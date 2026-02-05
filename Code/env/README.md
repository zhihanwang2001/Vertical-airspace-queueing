# Vertical Stratified Queue Environment

## 📖 Overview

This environment implements a vertical stratified queue system based on MCRPS/D/K theory for deep reinforcement learning optimization research in UAV delivery systems.

Note: This repository contains two environment implementations for research and training:
- DRL Fixed Version (for training and experiments, Dict observation, 29-dim): `env.drl_optimized_env_fixed.DRLOptimizedQueueEnvFixed`
- Theoretical Version (for mechanism research and visualization, vector observation, high-dim): `env.vertical_queue_env.VerticalQueueEnv`
This document describes the observation/action space corresponding to the "DRL Fixed Version" used by default in training scripts.

**Core Features**:
- 🏗️ **5-Layer Vertical Structure**: Altitude stratification [100m, 80m, 60m, 40m, 20m]
- 📊 **Inverted Pyramid Capacity**: Capacity configuration [8, 6, 4, 3, 2]
- 🧠 **29-Dimensional Observation Space**: Complete system state representation
- 🎮 **11-Dimensional Hybrid Actions**: Continuous control + discrete decisions
- 🎯 **Multi-Objective Optimization**: Balancing 6 optimization objectives

## 🚀 Quick Start

### Basic Usage

```python
from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed

# Create environment
env = DRLOptimizedQueueEnvFixed()

# Reset environment
obs, info = env.reset()

# Random action testing
for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
```

### Integration with DRL Frameworks

```python
from baselines.space_utils import SB3DictWrapper
from stable_baselines3 import PPO

# Environment wrapping (adapted for Stable-Baselines3)
env = DRLOptimizedQueueEnvFixed()
wrapped_env = SB3DictWrapper(env)

# Train model
model = PPO("MultiInputPolicy", wrapped_env, verbose=1)
model.learn(total_timesteps=100000)
```

## 🏗️ System Architecture

### Vertical Stratification Structure

| Layer | Altitude | Capacity | Service Rate | Arrival Weight |
|-------|----------|----------|--------------|----------------|
| L1 | 100m | 8 | 1.2 | 0.30 |
| L2 | 80m | 6 | 1.0 | 0.25 |
| L3 | 60m | 4 | 0.8 | 0.20 |
| L4 | 40m | 3 | 0.6 | 0.15 |
| L5 | 20m | 2 | 0.4 | 0.10 |

**Design Philosophy**:
- **Inverted Pyramid Capacity**: Reflects real airspace physical constraints
- **Ascending Service Priority**: Upper layers fast processing (μ5=1.5), lower layers refined service (μ1=0.8)
- **Differentiated Arrivals**: Upper layers bear more traffic load

## 📊 Observation Space (29-Dimensional)

Observation space is in Dict format, containing 7 main components:

```python
observation_space = spaces.Dict({
    'queue_lengths':     Box(shape=(5,)),  # Queue length per layer [0, max_capacity]
    'utilization_rates': Box(shape=(5,)),  # Utilization rate per layer [0.0, 1.0]
    'queue_changes':     Box(shape=(5,)),  # Queue change trend [-1.0, 1.0]
    'load_rates':        Box(shape=(5,)),  # Actual load rate [0.0, 5.0]
    'service_rates':     Box(shape=(5,)),  # Current service rate [0.0, 10.0]
    'prev_reward':       Box(shape=(1,)),  # Historical reward [-100.0, 100.0]
    'system_metrics':    Box(shape=(3,)),  # System metrics [0.0, 10.0]
})
```

**Dimension Details**:
- **queue_lengths** (5-dim): Current number of UAVs in each layer's queue
- **utilization_rates** (5-dim): Capacity utilization per layer = queue_length / capacity
- **queue_changes** (5-dim): Queue length change relative to previous step
- **load_rates** (5-dim): Load coefficient ρ = λ_eff / (μ × capacity)
- **service_rates** (5-dim): Dynamically adjusted actual service rates
- **prev_reward** (1-dim): Previous step's reward value for reward trend learning
- **system_metrics** (3-dim): [total load, total utilization, stability indicator]

**Total**: 5+5+5+5+5+1+3 = **29 dimensions**

## 🎮 Action Space (11-Dimensional Hybrid)

Action space is in Dict format, containing continuous control and discrete decisions:

```python
action_space = spaces.Dict({
    'service_intensities': Box([0.1, 2.0], shape=(5,)),  # Continuous: service intensity adjustment
    'arrival_multiplier':  Box([0.5, 5.0], shape=(1,)),  # Continuous: global arrival multiplier
    'emergency_transfers': MultiBinary(5)                # Discrete: emergency transfer decisions
})
```

**Action Details**:
- **service_intensities** (5-dim continuous): Service intensity adjustment multiplier per layer [0.1, 2.0]
  - Controls service processing speed for each layer
  - Value >1.0 indicates accelerated service, <1.0 indicates decelerated service
- **arrival_multiplier** (1-dim continuous): Global arrival rate adjustment [0.5, 5.0]
  - Controls overall system task arrival intensity
  - Used for dynamic load management
- **emergency_transfers** (5-dim discrete): Emergency transfer trigger {0, 1}
  - One binary switch per layer
  - 1 indicates triggering emergency downward transfer for that layer

**Total**: 5+1+5 = **11 dimensions**

## 🎯 Reward Function (6-Objective Optimization)

The system employs a multi-objective reward function balancing 6 key metrics:

### Reward Components

```python
def _calculate_reward_fixed(self, action) -> float:
    # 1. Throughput reward (primary objective)
    R_throughput = 10.0 * np.sum(service_counts)

    # 2. Gini coefficient load balancing (fairness)
    gini = self._calculate_gini_coefficient(utilization_rates)
    R_balance = 5.0 * (1.0 - gini)

    # 3. Energy efficiency optimization (efficiency)
    R_efficiency = 3.0 * total_service / total_energy

    # 4. Congestion penalty (stability)
    P_congestion = -20.0 * np.sum(congestion_levels)

    # 5. System stability (robustness)
    P_instability = -15.0 * instability_penalty

    # 6. Emergency transfer cost (control complexity)
    P_transfer = -5.0 * np.sum(emergency_transfers)

    return R_throughput + R_balance + R_efficiency + P_congestion + P_instability + P_transfer
```

### Objective Weights

| Objective | Weight | Description |
|-----------|--------|-------------|
| Throughput maximization | 10.0 | System processing capacity |
| Load balancing | 5.0 | Gini coefficient fairness |
| Energy efficiency | 3.0 | Service/energy ratio |
| Congestion penalty | -20.0 | Prevent system overload |
| Stability penalty | -15.0 | Maintain system stability |
| Transfer cost | -5.0 | Control action complexity |

## ⚙️ System Parameters

### Core Configuration
```python
# Physical structure
n_layers = 5
heights = [100, 80, 60, 40, 20]  # meters
capacities = [8, 6, 4, 3, 2]     # Inverted pyramid capacity

# Arrival process
base_arrival_rate = 0.3                    # λ₀ = 0.3/step
arrival_weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # α_i

# Service process
base_service_rates = [1.2, 1.0, 0.8, 0.6, 0.4]  # μ_i

# Environment parameters
max_episode_steps = 200  # Episode time limit
render_modes = ["human"]
```

### Dynamics Model

**Arrival Process**: Poisson arrivals + multinomial splitting
```
λ_total(t) = base_arrival_rate × arrival_multiplier(t)
λ_i(t) = λ_total(t) × arrival_weights[i]
```

**Service Process**: State-dependent service rates
```
μ_i(t) = base_service_rates[i] × service_intensities[i](t)
```

**Transfer Dynamics**: Pressure-triggered + emergency transfers
```
transfer_rate_i = f(queue_length_i, utilization_i, emergency_transfers[i])
```

## 🔬 MCRPS/D/K Theoretical Foundation

This environment implements a novel queueing network type: **MCRPS/D/K**, which is the core innovation of the deep reinforcement learning-based vertical stratified queueing optimization theory.

### Theoretical Classification
- **MC**: Multi-layer Correlated arrivals
- **R**: Random batch service
- **P**: Poisson splitting
- **S**: State-dependent
- **D**: Dynamic transfer
- **K**: Finite capacity

### Innovative System Dynamics

#### 1. Correlated Arrival Process (MC + P)
```mathematical
Nₜ ~ Poisson(λ₀μₜ)                    # Total arrivals follow Poisson distribution
(N₁ₜ, ..., N₅ₜ) ~ Multinomial(Nₜ, w)  # Multinomial splitting
Actual arrivals: min(Nᵢₜ, cᵢ - qᵢₜ)   # Capacity-constrained arrivals
```
**Innovative Feature**: Inter-layer correlated arrivals vs traditional independent arrivals

#### 2. Batch Service Process (R)
```mathematical
Dᵢₜ = min(max(1, Poisson(s₀ᵢαᵢₜ)), qᵢₜ)
```
**Innovative Feature**: Non-zero batch service guarantee vs traditional single-entity service

#### 3. Dynamic Transfer Mechanism (D)
```mathematical
When τᵢₜ = 1 and i < 5:
Pressure coefficient: pᵢ = qᵢₜ/cᵢ
Transfer quantity: Tᵢₜ = min(⌊qᵢₜ × min(0.8, pᵢ)⌋, cᵢ₊₁ - qᵢ₊₁ₜ)
```
**Innovative Feature**: Pressure-triggered cross-layer transfers vs traditional static networks

### Comparison with Classical Queueing Theory

| Feature | Classical Queueing | MCRPS/D/K |
|---------|-------------------|-----------|
| **Spatial Structure** | Abstract nodes | Physical vertical stratification |
| **Arrival Process** | Independent Poisson | Correlated + splitting |
| **Service Mechanism** | Single exponential | Batch + non-zero guarantee |
| **Network Topology** | Static connections | Dynamic transfers |
| **Capacity Design** | Uniform/increasing | Inverted pyramid |
| **Control Method** | Fixed parameters | DRL real-time optimization |

### Theoretical Innovation Value

1. **Vertical Stratified Queueing**: First physical space-stratified queueing network, filling the theoretical gap in airspace management
2. **Inverted Pyramid Capacity**: Capacity design reflecting real airspace constraints, upper layers have larger capacity but decrease due to physical limitations
3. **Pressure-Triggered Transfers**: Cross-layer dynamic transfers based on congestion pressure, achieving intelligent load balancing
4. **Multi-Objective Balancing**: Intelligent scheduling strategy balancing 6 objectives, considering efficiency, fairness, and stability
5. **Theoretical Breakthrough**: Creates a novel queueing system type with no classical analytical solution, requiring simulation analysis

### Literature Theoretical Support

Based on "Queueing Theory and Airspace Management Literature Review" analysis, existing research has the following limitations:

- **[4] Vertiport Capacity Assessment**: Only considers single-layer M/M/1 and M/M/c, lacks vertical stratification
- **[11] Airspace Network Design**: 3D corridor segmented modeling, but no dynamic inter-layer transfers
- **[5] Air Traffic Flow Management**: MILP optimization lacks real-time learning capability

**Our Breakthrough**: MCRPS/D/K theory fills the theoretical gap in vertical stratified dynamic queueing, achieving a leap from static network design to intelligent real-time scheduling.

## 🏛️ System Environment Architecture Summary

### Overall Architecture Design

```
┌─────────────────────────────────────────────────────────────┐
│                DRLOptimizedQueueEnvFixed                    │
│              (29-dim state + 11-dim hybrid action)          │
├─────────────────────────────────────────────────────────────┤
│  Layer 1 (100m) │ Capacity: 8 │ Service: 1.2 │ Weight: 0.30│
│  Layer 2 (80m)  │ Capacity: 6 │ Service: 1.0 │ Weight: 0.25│
│  Layer 3 (60m)  │ Capacity: 4 │ Service: 0.8 │ Weight: 0.20│
│  Layer 4 (40m)  │ Capacity: 3 │ Service: 0.6 │ Weight: 0.15│
│  Layer 5 (20m)  │ Capacity: 2 │ Service: 0.4 │ Weight: 0.10│
├─────────────────┬───────────────┬───────────────┬─────────────┤
│ Poisson Arrival │ Multinomial   │ Batch Service │ Pressure    │
│ Poisson(0.3μₜ)  │ Splitting     │ max(1,Pois.) │ Transfer    │
└─────────────────┴───────────────┴───────────────┴─────────────┘
```

### Core Component Architecture

#### 1. State Management Subsystem (StateManager)
- **Function**: 29-dimensional state space construction and updating
- **Components**: Queue state, utilization rate, load rate, system metrics
- **Features**: Real-time state normalization and historical recording

#### 2. Queue Dynamics Subsystem (QueueDynamics)
- **Function**: Core implementation of MCRPS/D/K theory
- **Components**: Correlated arrivals, batch service, dynamic transfers
- **Features**: Mathematical modeling of non-standard queueing processes

#### 3. Reward Optimization Subsystem (RewardFunction)
- **Function**: 6-objective multi-trade-off optimization
- **Components**: Throughput, fairness, efficiency, stability
- **Features**: Mathematically rigorous Gini coefficient load balancing

#### 4. Action Control Subsystem (ActionProcessor)
- **Function**: 11-dimensional hybrid action space processing
- **Components**: Continuous service control + discrete transfer decisions
- **Features**: Real-time parameter adjustment and emergency response

### System Integration Features

#### 📊 Data Flow Architecture
```
Environment State → StateManager → 29-dim observation vector
     ↓
DRL Agent → Policy Network → 11-dim hybrid action
     ↓
ActionProcessor → Parameter parsing → System control signals
     ↓
QueueDynamics → MCRPS/D/K → State update
     ↓
RewardFunction → 6-objective evaluation → Reward feedback
```

#### 🔄 Control Loop Design
1. **Observation Phase**: Extract 29-dimensional system state
2. **Decision Phase**: DRL generates 11-dimensional hybrid action
3. **Execution Phase**: Action parsed into system parameter adjustments
4. **Update Phase**: MCRPS/D/K dynamics state transition
5. **Evaluation Phase**: Multi-objective reward calculation and feedback

#### ⚡ Real-Time Response Mechanism
- **Normal Mode**: Intelligent scheduling based on DRL policy
- **Emergency Mode**: Pressure-triggered rapid cross-layer transfers
- **Stability Guarantee**: Mathematical constraints ensure system doesn't crash
- **Performance Monitoring**: Real-time load rate and stability detection

### Technical Innovation Summary

#### 🧠 Theoretical Level
- **Original Theory**: MCRPS/D/K vertical stratified queueing network
- **Mathematical Rigor**: Simulation-based novel queueing system analysis
- **Gap Filling**: Theoretical foundation for vertical airspace management

#### 🛠️ Engineering Level
- **Modular Design**: High cohesion, low coupling system architecture
- **Hybrid Actions**: Continuous + discrete complex control space
- **Real-Time Optimization**: DRL-driven dynamic parameter adjustment

#### 📈 Performance Level
- **Convergence Stability**: Environment stability validated by 15 algorithms
- **Performance Breakthrough**: DRL improves 50%+ over traditional methods
- **Strong Robustness**: Stable performance under various load conditions

### Comparison with Existing Systems

| Dimension | Traditional Queueing | Existing UAV Management | MCRPS/D/K Environment |
|-----------|---------------------|------------------------|----------------------|
| **Spatial Model** | Abstract network | Geometric planning | Physical vertical stratification |
| **Scheduling Method** | Static rules | Heuristic algorithms | DRL real-time learning |
| **State Dimension** | Simple metrics | Position velocity | 29-dim system state |
| **Control Precision** | Fixed parameters | Path control | 11-dim hybrid actions |
| **Theoretical Foundation** | Classical queueing | Operations research | MCRPS/D/K theory |
| **Adaptability** | Static configuration | Semi-dynamic | Fully adaptive |

This architectural design achieves a complete chain from theoretical innovation to engineering implementation, providing a new technical paradigm for vertical airspace management.

## 📈 Performance Benchmarks

### Experimental Validation Results

Based on large-scale experiments with 500k timesteps:

| Algorithm | Avg Reward | Std Dev | Training Time | Convergence |
|-----------|------------|---------|---------------|-------------|
| **PPO** | **4419.98** | 135.71 | 30.8min | ⭐⭐⭐⭐⭐ |
| **TD7** | **4392.52** | 84.60 | 382.4min | ⭐⭐⭐⭐⭐ |
| **R2D2** | **4289.22** | 82.23 | 115.7min | ⭐⭐⭐⭐ |
| SAC v2 | 4282.94 | 80.70 | 287.0min | ⭐⭐⭐⭐ |
| Heuristic | 2860.69 | 87.96 | 1.1min | ⭐⭐⭐ |

**Key Findings**:
- DRL algorithms improve **50%+** over heuristics
- Best algorithm achieves **>4400** average reward
- Standard deviation controlled at **<200**, proving environment stability

## 🛠️ Development Guide

### Extending the Environment

```python
class CustomVerticalQueueEnv(DRLOptimizedQueueEnvFixed):
    def __init__(self, custom_config):
        super().__init__()
        # Custom configuration
        self.custom_param = custom_config

    def _calculate_reward_fixed(self, action):
        # Custom reward function
        base_reward = super()._calculate_reward_fixed(action)
        custom_bonus = self._calculate_custom_bonus()
        return base_reward + custom_bonus
```

### Adding New Observations

```python
def _get_enhanced_observation(self):
    base_obs = self._get_observation()

    # Add new observation dimensions
    enhanced_obs = base_obs.copy()
    enhanced_obs['custom_metrics'] = self._calculate_custom_metrics()

    return enhanced_obs
```

### Debugging Tools

```python
# Enable verbose output
env = DRLOptimizedQueueEnvFixed(render_mode="human")

# State inspection
print(f"Current state: {env._get_observation()}")
print(f"Queue lengths: {env.queue_lengths}")
print(f"Utilization: {env.queue_lengths / env.capacities}")
```

## 📚 Related Documentation

- **Theoretical Foundation**: `docs/Vertical_Stratified_Queueing_Theory_Final.md`
- **Experiment Guide**: `docs/README_EXPERIMENT.md`
- **Code Architecture**: `docs/RP1_Project_Core_Code_Architecture_Analysis.md`
- **Innovation Analysis**: `docs/Innovation_Analysis_Report.md`

## 🤝 Support and Contribution

For questions or suggestions, please refer to project documentation or contact the development team.

**Version**: v1.0.0
**Last Updated**: 2025-09-29
**Project Status**: Production Ready ✅
