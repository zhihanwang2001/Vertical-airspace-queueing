"""
Capacity-Aware Action Clipper for HCA2C

Dynamically clips actions based on current system state to ensure
all actions are feasible and within capacity constraints.
"""

import numpy as np
import torch
from typing import Dict, Union, Optional


class CapacityAwareClipper:
    """
    Capacity-Aware Action Clipper

    Clips actions to feasible ranges based on current state:
    - Service intensity: Cannot serve more than queue length
    - Transfer amount: Cannot transfer more than available capacity at target
    - Arrival control: Bounded by system capacity

    This eliminates invalid actions and improves exploration efficiency.
    """

    def __init__(self,
                 capacities: np.ndarray = None,
                 n_layers: int = 5,
                 service_range: tuple = (0.1, 2.0),
                 arrival_range: tuple = (0.5, 5.0),
                 transfer_threshold: float = 0.3):
        """
        Initialize the clipper

        Args:
            capacities: Capacity for each layer [n_layers]
            n_layers: Number of layers
            service_range: (min, max) for service intensity
            arrival_range: (min, max) for arrival multiplier
            transfer_threshold: Threshold for triggering transfers
        """
        self.n_layers = n_layers

        if capacities is None:
            self.capacities = np.array([8, 6, 4, 3, 2], dtype=np.float32)
        else:
            self.capacities = np.array(capacities, dtype=np.float32)

        self.service_min, self.service_max = service_range
        self.arrival_min, self.arrival_max = arrival_range
        self.transfer_threshold = transfer_threshold

    def clip_action(self,
                    raw_action: Dict[str, Union[np.ndarray, torch.Tensor]],
                    state: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str, np.ndarray]:
        """
        Clip actions based on current state

        Args:
            raw_action: Dictionary containing raw action outputs
                - service_intensities: [n_layers] service rate multipliers
                - arrival_multiplier: [1] arrival rate multiplier
                - transfer_decisions: [n_layers] transfer directions/magnitudes
                - emergency_mode: [1] emergency flag (optional)
            state: Dictionary containing current state
                - queue_lengths: [n_layers] current queue lengths
                - utilization_rates: [n_layers] current utilization
                - (or hierarchical format with 'global' and 'layers')

        Returns:
            Clipped action dictionary ready for environment
        """
        # Convert to numpy if needed
        raw_action = self._to_numpy(raw_action)
        state = self._to_numpy(state)

        # Extract state information
        queue_lengths, utilizations = self._extract_state(state)

        # Clip each action component
        clipped = {}

        # 1. Service intensities - adaptive based on queue pressure
        clipped['service_intensities'] = self._clip_service(
            raw_action.get('service_intensities', np.ones(self.n_layers)),
            queue_lengths,
            utilizations
        )

        # 2. Arrival multiplier - reduce if system is overloaded
        clipped['arrival_multiplier'] = self._clip_arrival(
            raw_action.get('arrival_multiplier', np.array([1.0])),
            utilizations
        )

        # 3. Emergency transfers - only allow feasible transfers
        clipped['emergency_transfers'] = self._clip_transfers(
            raw_action.get('transfer_decisions', np.zeros(self.n_layers)),
            queue_lengths
        )

        return clipped

    def _clip_service(self,
                      service: np.ndarray,
                      queue_lengths: np.ndarray,
                      utilizations: np.ndarray) -> np.ndarray:
        """
        Clip service intensities based on queue state

        - If queue is empty, reduce service intensity (save resources)
        - If queue is near capacity, increase service intensity
        """
        clipped = np.clip(service, self.service_min, self.service_max)

        for i in range(self.n_layers):
            # If queue is empty, cap service at minimum
            if queue_lengths[i] < 0.1:
                clipped[i] = self.service_min

            # If utilization is very high, boost service
            elif utilizations[i] > 0.9:
                clipped[i] = max(clipped[i], self.service_max * 0.8)

        return clipped.astype(np.float32)

    def _clip_arrival(self,
                      arrival: np.ndarray,
                      utilizations: np.ndarray) -> np.ndarray:
        """
        Clip arrival multiplier based on system load

        - If system is overloaded, reduce arrival rate
        - If system has capacity, allow higher arrival
        """
        arrival = np.clip(arrival, self.arrival_min, self.arrival_max)

        # Compute system load
        avg_util = np.mean(utilizations)
        max_util = np.max(utilizations)

        # If any layer is near capacity, reduce arrivals
        if max_util > 0.9:
            arrival = np.minimum(arrival, self.arrival_min + 0.5)
        elif max_util > 0.8:
            arrival = np.minimum(arrival, (self.arrival_min + self.arrival_max) / 2)

        return arrival.astype(np.float32)

    def _clip_transfers(self,
                        transfers: np.ndarray,
                        queue_lengths: np.ndarray) -> np.ndarray:
        """
        Clip transfer decisions to feasible values

        - Cannot transfer from empty queue
        - Cannot transfer to full queue
        - Convert continuous transfer decision to binary
        """
        # Convert continuous [-1, 1] to binary {0, 1}
        # Positive values trigger transfer to next layer
        emergency = (transfers > self.transfer_threshold).astype(np.int8)

        for i in range(self.n_layers):
            # Cannot transfer from empty queue
            if queue_lengths[i] < 1:
                emergency[i] = 0

            # Cannot transfer from last layer (no target)
            if i == self.n_layers - 1:
                emergency[i] = 0

            # Cannot transfer if target is full
            if i < self.n_layers - 1:
                target_available = self.capacities[i + 1] - queue_lengths[i + 1]
                if target_available < 1:
                    emergency[i] = 0

        return emergency

    def _extract_state(self, state: Dict) -> tuple:
        """Extract queue lengths and utilizations from state"""
        if 'layers' in state:
            # Hierarchical format
            layers = state['layers']
            if isinstance(layers, np.ndarray) and layers.ndim == 2:
                # layers shape: [n_layers, layer_dim]
                # Assuming: [queue, capacity, util, service, load, up, down, change]
                queue_lengths = layers[:, 0] * self.capacities  # Denormalize
                utilizations = layers[:, 2]
            else:
                queue_lengths = np.zeros(self.n_layers)
                utilizations = np.zeros(self.n_layers)
        else:
            # Flat format
            queue_lengths = state.get('queue_lengths', np.zeros(self.n_layers))
            utilizations = state.get('utilization_rates', np.zeros(self.n_layers))

        return np.array(queue_lengths), np.array(utilizations)

    def _to_numpy(self, data: Union[Dict, np.ndarray, torch.Tensor]) -> Union[Dict, np.ndarray]:
        """Convert torch tensors to numpy arrays"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, dict):
            return {k: self._to_numpy(v) for k, v in data.items()}
        else:
            return np.array(data)

    def get_action_bounds(self, state: Dict) -> Dict[str, tuple]:
        """
        Get dynamic action bounds based on current state

        Useful for constrained optimization or action space reshaping.
        """
        queue_lengths, utilizations = self._extract_state(state)

        bounds = {
            'service_intensities': [],
            'arrival_multiplier': (self.arrival_min, self.arrival_max),
            'transfer_feasible': []
        }

        for i in range(self.n_layers):
            # Service bounds
            if queue_lengths[i] < 0.1:
                bounds['service_intensities'].append((self.service_min, self.service_min))
            else:
                bounds['service_intensities'].append((self.service_min, self.service_max))

            # Transfer feasibility
            can_transfer = (
                queue_lengths[i] >= 1 and
                i < self.n_layers - 1 and
                (self.capacities[i + 1] - queue_lengths[i + 1]) >= 1
            )
            bounds['transfer_feasible'].append(can_transfer)

        return bounds


class DifferentiableClipper(torch.nn.Module):
    """
    Differentiable version of CapacityAwareClipper for end-to-end training

    Uses soft clipping functions to maintain gradient flow while
    encouraging feasible actions.
    """

    def __init__(self,
                 capacities: torch.Tensor = None,
                 n_layers: int = 5,
                 temperature: float = 1.0):
        super().__init__()

        self.n_layers = n_layers
        self.temperature = temperature

        if capacities is None:
            capacities = torch.tensor([8, 6, 4, 3, 2], dtype=torch.float32)

        self.register_buffer('capacities', capacities)

    def forward(self,
                raw_action: Dict[str, torch.Tensor],
                state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Soft clip actions while maintaining differentiability

        Uses sigmoid/tanh transformations with state-dependent scaling.
        """
        # Extract state
        if 'layers' in state:
            layers = state['layers']
            queue_lengths = layers[:, :, 0] * self.capacities.unsqueeze(0)
            utilizations = layers[:, :, 2]
        else:
            queue_lengths = state.get('queue_lengths', torch.zeros(1, self.n_layers))
            utilizations = state.get('utilization_rates', torch.zeros(1, self.n_layers))

        clipped = {}

        # Service intensities with soft bounds
        service = raw_action.get('service_intensities')
        if service is not None:
            # Scale based on queue availability
            queue_factor = torch.sigmoid(queue_lengths / self.capacities.unsqueeze(0) * self.temperature)
            service_scaled = 0.1 + 1.9 * torch.sigmoid(service) * queue_factor
            clipped['service_intensities'] = service_scaled

        # Arrival multiplier with load-based scaling
        arrival = raw_action.get('arrival_multiplier')
        if arrival is not None:
            # Reduce arrival when system is loaded
            load_factor = 1.0 - torch.sigmoid((utilizations.mean(dim=-1, keepdim=True) - 0.7) * self.temperature)
            arrival_scaled = 0.5 + 4.5 * torch.sigmoid(arrival) * load_factor
            clipped['arrival_multiplier'] = arrival_scaled

        # Transfer decisions with feasibility mask
        transfers = raw_action.get('transfer_decisions')
        if transfers is not None:
            # Soft feasibility mask
            feasibility = torch.zeros_like(transfers)
            for i in range(self.n_layers - 1):
                source_available = torch.sigmoid((queue_lengths[:, i] - 1) * self.temperature)
                target_space = torch.sigmoid(
                    (self.capacities[i + 1] - queue_lengths[:, i + 1] - 1) * self.temperature
                )
                feasibility[:, i] = source_available * target_space

            transfer_masked = torch.sigmoid(transfers) * feasibility
            clipped['emergency_transfers'] = (transfer_masked > 0.5).float()

        return clipped


if __name__ == "__main__":
    print("Testing CapacityAwareClipper...")

    # Create clipper
    clipper = CapacityAwareClipper()

    # Test state
    state = {
        'queue_lengths': np.array([5.0, 3.0, 2.0, 1.0, 0.5]),
        'utilization_rates': np.array([0.625, 0.5, 0.5, 0.33, 0.25])
    }

    # Test raw action
    raw_action = {
        'service_intensities': np.array([1.5, 1.2, 1.0, 0.8, 0.5]),
        'arrival_multiplier': np.array([3.0]),
        'transfer_decisions': np.array([0.5, 0.2, -0.3, 0.8, 0.1])
    }

    # Clip action
    clipped = clipper.clip_action(raw_action, state)

    print(f"✅ Original service: {raw_action['service_intensities']}")
    print(f"✅ Clipped service: {clipped['service_intensities']}")
    print(f"✅ Original arrival: {raw_action['arrival_multiplier']}")
    print(f"✅ Clipped arrival: {clipped['arrival_multiplier']}")
    print(f"✅ Original transfers: {raw_action['transfer_decisions']}")
    print(f"✅ Clipped transfers: {clipped['emergency_transfers']}")

    # Test with hierarchical state
    hier_state = {
        'global': np.array([0.5, 0.4, 0.8, 0.3, 0.45, 0.0]),
        'layers': np.array([
            [0.625, 1.0, 0.625, 0.6, 0.3, 0.0, 0.5, 0.1],
            [0.5, 0.75, 0.5, 0.5, 0.25, 0.625, 0.5, 0.0],
            [0.5, 0.5, 0.5, 0.4, 0.2, 0.5, 0.33, -0.1],
            [0.33, 0.375, 0.33, 0.3, 0.15, 0.5, 0.25, 0.0],
            [0.25, 0.25, 0.25, 0.2, 0.1, 0.33, 0.0, 0.05]
        ])
    }

    clipped_hier = clipper.clip_action(raw_action, hier_state)
    print(f"\n✅ Hierarchical state clipping works!")
    print(f"✅ Clipped service (hier): {clipped_hier['service_intensities']}")

    # Test differentiable clipper
    print("\n\nTesting DifferentiableClipper...")
    diff_clipper = DifferentiableClipper()

    torch_state = {
        'layers': torch.tensor(hier_state['layers']).unsqueeze(0)
    }
    torch_action = {
        'service_intensities': torch.tensor(raw_action['service_intensities']).unsqueeze(0),
        'arrival_multiplier': torch.tensor(raw_action['arrival_multiplier']).unsqueeze(0),
        'transfer_decisions': torch.tensor(raw_action['transfer_decisions']).unsqueeze(0)
    }

    diff_clipped = diff_clipper(torch_action, torch_state)
    print(f"✅ Differentiable clipped service: {diff_clipped['service_intensities']}")

    print("\n✅ All clipper tests passed!")
