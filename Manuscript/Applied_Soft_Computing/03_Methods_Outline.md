# Methods Section Outline

## 3. Methodology (4-5 pages)

### 3.1 MCRPS/D/K Queueing Framework (1 page)

#### 3.1.1 Framework Overview
- **Purpose**: Model vertical layered queueing system for UAM airspace
- **Key innovation**: Dynamic inter-layer transfers based on pressure thresholds
- **Components**: MC-R-P-S-D-K notation explained

#### 3.1.2 System Architecture
- **Vertical structure**: 5 layers (L0 to L4) representing altitude zones
- **Capacity configuration**: Layer-specific finite capacities K = [k₀, k₁, k₂, k₃, k₄]
- **Service mechanism**: Batch service with random selection and non-zero guarantee
- **Transfer mechanism**: Pressure-triggered bidirectional transfers between adjacent layers

#### 3.1.3 Arrival Process (MC-P Components)
- **Total arrival rate**: λ_total follows Poisson process
- **Layer-specific arrivals**: Multinomial splitting with weights w = [w₀, w₁, w₂, w₃, w₄]
- **Correlation structure**: Arrivals correlated through shared Poisson source
- **Mathematical formulation**:
  - λᵢ = λ_total × wᵢ for layer i
  - Arrival weights: w = [0.3, 0.25, 0.2, 0.15, 0.1] (default configuration)

#### 3.1.4 Service Process (R-S Components)
- **Service rates**: Layer-specific rates μ = [μ₀, μ₁, μ₂, μ₃, μ₄]
- **Batch service**: Random selection from queue with minimum guarantee
- **State-dependent**: Service efficiency varies with queue length
- **Mathematical formulation**:
  - Service capacity: sᵢ = min(qᵢ, μᵢ) where qᵢ is queue length
  - Batch size: Bᵢ ~ Uniform(1, sᵢ) with minimum guarantee

#### 3.1.5 Dynamic Transfer Mechanism (D Component)
- **Pressure calculation**: pᵢ = qᵢ / kᵢ (utilization ratio)
- **Transfer conditions**:
  - Upward transfer: If pᵢ > threshold_up AND pᵢ₊₁ < threshold_down
  - Downward transfer: If pᵢ > threshold_up AND pᵢ₋₁ < threshold_down
- **Transfer volume**: Proportional to pressure difference
- **Constraints**: Respect capacity limits of receiving layer

#### 3.1.6 Capacity Constraints (K Component)
- **Finite capacity**: Each layer has maximum capacity kᵢ
- **Blocking**: New arrivals rejected if layer at capacity
- **Crash condition**: System terminates if any layer exceeds capacity
- **Capacity configurations tested**:
  - Inverted pyramid: [8, 6, 4, 3, 2] (total K=23)
  - Normal pyramid: [2, 3, 4, 6, 8] (total K=23)
  - Uniform: [k, k, k, k, k] for various k values

---

### 3.2 Deep Reinforcement Learning Algorithms (1-1.5 pages)

#### 3.2.1 Algorithm Categories
This study evaluates 15 state-of-the-art DRL algorithms across four categories:

**Policy Gradient Methods:**
- **A2C (Advantage Actor-Critic)**: Synchronous variant of A3C with advantage estimation
- **PPO (Proximal Policy Optimization)**: Clipped surrogate objective for stable policy updates

**Actor-Critic Methods:**
- **TD3 (Twin Delayed DDPG)**: Addresses overestimation bias with twin Q-networks
- **SAC (Soft Actor-Critic)**: Maximum entropy framework for exploration
- **TD7 (TD3 + 7 improvements)**: Enhanced TD3 with multiple algorithmic improvements
- **DDPG (Deep Deterministic Policy Gradient)**: Deterministic policy for continuous control

**Value-Based Methods:**
- **DQN (Deep Q-Network)**: Deep neural network for Q-value approximation
- **Rainbow**: Combines 6 DQN extensions (double Q, dueling, prioritized replay, etc.)
- **R2D2 (Recurrent Replay Distributed DQN)**: Recurrent architecture for partial observability

**Distributed Methods:**
- **IMPALA (Importance Weighted Actor-Learner Architecture)**: Decoupled acting and learning
- **APEX-DQN**: Distributed prioritized experience replay

**Additional Algorithms:**
- **QRDQN (Quantile Regression DQN)**: Distributional RL approach
- **C51**: Categorical distributional RL
- **IQN (Implicit Quantile Networks)**: Implicit quantile function approximation

#### 3.2.2 Algorithm Hyperparameters
**A2C Configuration:**
- Learning rate: 0.0007
- n_steps: 32
- Gamma: 0.99
- GAE lambda: 0.95
- Entropy coefficient: 0.01
- Value function coefficient: 0.5
- Max gradient norm: 0.5

**PPO Configuration:**
- Learning rate: 0.0003
- n_steps: 2048
- Batch size: 64
- n_epochs: 10
- Gamma: 0.99
- GAE lambda: 0.95
- Clip range: 0.2

**[Similar detail for other algorithms - abbreviated for space]**

#### 3.2.3 State Space Design (29 dimensions)
- **Queue lengths** (5 dims): q₀, q₁, q₂, q₃, q₄
- **Capacities** (5 dims): k₀, k₁, k₂, k₃, k₄
- **Utilization ratios** (5 dims): q₀/k₀, ..., q₄/k₄
- **Service rates** (5 dims): μ₀, μ₁, μ₂, μ₃, μ₄
- **Arrival rates** (5 dims): λ₀, λ₁, λ₂, λ₃, λ₄
- **Time step** (1 dim): Current timestep
- **Total system load** (1 dim): Σqᵢ
- **Average waiting time** (1 dim): Mean waiting time across layers
- **Crash indicator** (1 dim): Binary flag for capacity violation

#### 3.2.4 Action Space Design (11 dimensions)
- **Service allocation** (5 dims): Continuous [0,1] for each layer's service priority
- **Transfer decisions** (4 dims): Continuous [-1,1] for inter-layer transfers (L0↔L1, L1↔L2, L2↔L3, L3↔L4)
- **Admission control** (2 dims): Continuous [0,1] for accepting/rejecting arrivals at top/bottom layers

#### 3.2.5 Reward Function (6 objectives)
Multi-objective reward combining:
1. **Throughput reward**: +1 per successful service
2. **Waiting time penalty**: -0.1 × average waiting time
3. **Queue length penalty**: -0.05 × total queue length
4. **Crash penalty**: -10000 for capacity violation
5. **Balance reward**: +0.5 for balanced utilization across layers
6. **Transfer efficiency**: +0.2 per successful transfer, -0.1 per failed transfer

---

### 3.3 Experimental Design (1 page)

#### 3.3.1 Training Configuration
- **Total timesteps**: 500,000 per algorithm
- **Random seeds**: 5 seeds (42, 43, 44, 45, 46) for reproducibility
- **Training environment**: Stable-Baselines3 framework
- **Hardware**: [Specify GPU/CPU specifications]
- **Software versions**: Python 3.8+, PyTorch 1.10+, Stable-Baselines3 1.5+

#### 3.3.2 Evaluation Protocol
- **Evaluation frequency**: Every 10,000 timesteps during training
- **Evaluation episodes**: 50 episodes per evaluation
- **Deterministic policy**: Used during evaluation (no exploration noise)
- **Metrics recorded**:
  - Mean episode reward
  - Standard deviation of rewards
  - Mean episode length
  - Crash rate (percentage of episodes ending in capacity violation)
  - Training time (wall-clock minutes)

#### 3.3.3 Baseline Implementations
Four traditional heuristic baselines for comparison:

**1. FCFS (First-Come-First-Served)**
- Serves requests in arrival order
- No prioritization or optimization
- Represents naive baseline

**2. SJF (Shortest Job First)**
- Prioritizes requests with shortest expected service time
- Minimizes average waiting time in single-server systems
- Classical queueing heuristic

**3. Priority-Based**
- Assigns priority based on layer position (higher layers = higher priority)
- Reflects altitude-based urgency in UAM context
- Domain-specific heuristic

**4. Heuristic Baseline**
- Custom rule-based policy combining:
  - Load balancing across layers
  - Pressure-based transfer decisions
  - Threshold-based admission control
- Represents engineered baseline

#### 3.3.4 Ablation Studies
Three systematic ablation studies conducted:

**Study 1: Structural Comparison (5× load)**
- Compare inverted pyramid [8,6,4,3,2] vs normal pyramid [2,3,4,6,8]
- Algorithms: A2C and PPO (top performers)
- Sample size: n=30 per algorithm per structure (total n=60 per structure)
- Load multiplier: 5× baseline load

**Study 2: Capacity Scan (10× extreme load)**
- Test total capacities K ∈ {10, 15, 20, 25, 30, 40}
- Capacity shapes: Uniform, inverted, reverse pyramid
- Algorithms: A2C, PPO, and all heuristics
- Purpose: Identify capacity paradox and optimal capacity range

**Study 3: Generalization Testing**
- Test across 5 heterogeneous traffic patterns
- Vary arrival weights and service rates
- Validate robustness of findings
- Algorithms: Top 3 performers (A2C, PPO, TD7)

---

### 3.4 Statistical Analysis Methods (0.5 page)

#### 3.4.1 Hypothesis Testing
**Primary hypothesis**: DRL algorithms outperform traditional heuristics
- **Test**: Independent samples t-test
- **Null hypothesis**: μ_DRL = μ_heuristic
- **Alternative hypothesis**: μ_DRL > μ_heuristic
- **Significance level**: α = 0.05

**Secondary hypothesis**: Inverted pyramid outperforms normal pyramid
- **Test**: Independent samples t-test
- **Sample sizes**: n_inverted = 60, n_normal = 60
- **Effect size**: Cohen's d for practical significance

#### 3.4.2 Statistical Metrics
- **Mean and standard deviation**: For central tendency and variability
- **Standard error**: SE = σ/√n for precision estimation
- **t-statistic**: t = (μ₁ - μ₂) / SE_diff
- **p-value**: Probability of observing results under null hypothesis
- **Cohen's d**: d = (μ₁ - μ₂) / σ_pooled for effect size
- **95% Confidence intervals**: For parameter estimation

#### 3.4.3 Data Aggregation
- **Per-seed results**: Individual runs with fixed random seeds
- **Grouped statistics**: Mean and std across seeds for each algorithm
- **Comparative analysis**: Pairwise comparisons between algorithms and structures

#### 3.4.4 Reproducibility Measures
- **Fixed random seeds**: 42, 43, 44, 45, 46 for all experiments
- **Deterministic evaluation**: No exploration noise during testing
- **Complete hyperparameter documentation**: All settings recorded
- **Code availability**: Full implementation provided for verification

#### 3.4.5 Reward Reporting and Episode Length
**Reward Calculation**:
- All reported rewards are **cumulative per-episode rewards** (standard RL practice)
- Reward accumulates over the entire episode until termination
- No normalization or scaling applied to maintain interpretability

**Episode Termination Conditions**:
- **Maximum steps**: Episodes terminate at 10,000 timesteps (if system remains stable)
- **System crash**: Episodes terminate early if any queue exceeds capacity
- **Episode length varies** based on system stability and load conditions

**Reward Scale Variation**:
- **High load conditions**: Episodes often terminate early due to crashes → lower cumulative rewards
- **Low load conditions**: Episodes typically reach maximum 10,000 steps → higher cumulative rewards
- **Example**: At 3× load, mean episode length ≈ 10,000 steps (reward ≈ 437K); at 10× load, mean episode length ≈ 5,000 steps (reward ≈ 284K)
- **Implication**: Reward scales vary across experiments due to episode length differences, not different reward functions

**Reporting Standards**:
- Report both mean reward and mean episode length for transparency
- Consider reward-per-step as normalized metric when comparing across load conditions
- Crash rate reported separately as stability metric

#### 3.4.6 Effect Size Interpretation in Computational Experiments
**Context: Large Effect Sizes in Deterministic Systems**

This study reports Cohen's d effect sizes ranging from d=0.28 (small) to d=412.62 (extremely large) depending on load conditions. While effect sizes exceeding d=300 may appear unusual in social science research, they are legitimate and expected in computational experiments with the following characteristics:

**Why Large Effect Sizes Are Valid:**

1. **Extremely Low Variance in Converged Systems**
   - At high loads (7×-10×), coefficient of variation (CV) < 0.1%
   - Example: 10 runs at 7× load have range of only 831 (0.19% of mean)
   - Fixed random seeds + deterministic evaluation → minimal stochastic variation
   - Converged DRL policies produce highly consistent behavior

2. **Complete Distribution Separation**
   - Inverted pyramid group: [447,406 - 447,960] (range: 554)
   - Normal pyramid group: [387,198 - 387,829] (range: 631)
   - Separation distance: 59,577 (no overlap between groups)
   - When distributions don't overlap, large d values are mathematically inevitable

3. **Computational vs. Social Science Context**
   - **Social science**: d > 0.8 considered "large" due to high human variability
   - **Computational experiments**: d > 100 possible when variance is controlled
   - Cohen's d formula: d = (μ₁ - μ₂) / σ_pooled
   - Small σ_pooled (due to low variance) → large d (even with moderate mean differences)

**Load-Dependent Effect Size Pattern:**

| Load Level | Cohen's d | CV (%) | Interpretation |
|------------|-----------|--------|----------------|
| 3× | 0.28 | 2.1% | Small effect, high variance in A2C |
| 5× | 6.31 | 0.12% | Very large effect, low variance |
| 7× | 302.55 | 0.05% | Extremely large, complete separation |
| 10× | 412.62 | 0.02% | Extremely large, complete separation |

**Key Insight**: Effect sizes increase with load not because differences grow larger, but because variance decreases as system behavior becomes more deterministic under stress.

**Statistical Validity Confirmation:**
- Bootstrap 95% CI for d at 7× load: [241.77, 503.96] (excludes zero)
- Independent samples t-test: p < 10⁻⁴⁰ (highly significant)
- Welch's t-test (unequal variances): confirms significance
- All statistical tests support the validity of large effect sizes

**Interpretation Guidelines:**
- Focus on **practical significance**: 9.7%-19.7% performance improvement
- Report **CV** alongside effect sizes to show variance control
- Emphasize **complete separation** as evidence of robust differences
- Compare **absolute performance differences** for practical interpretation

**Literature Support:**
Computational experiments in operations research, algorithm comparison studies, and simulation-based research commonly report large effect sizes when:
- Systems are deterministic or near-deterministic
- Algorithms have converged to stable policies
- Evaluation protocols minimize stochastic variation
- Sample sizes are sufficient to detect small variance

This phenomenon is well-documented in computational science literature and should not be interpreted as a statistical error.

---

**End of Methods Section Outline**

**Estimated Length**: 4-5 pages (as required by Applied Soft Computing)

**Key Writing Notes**:
- Emphasize DRL algorithm diversity (15 algorithms across 4 categories)
- Provide sufficient detail for reproducibility
- Justify design choices (state space, action space, reward function)
- Highlight rigorous experimental design (500K timesteps, 5 seeds, 50 eval episodes)
- Include mathematical formulations where appropriate
- Reference Stable-Baselines3 documentation for implementation details
