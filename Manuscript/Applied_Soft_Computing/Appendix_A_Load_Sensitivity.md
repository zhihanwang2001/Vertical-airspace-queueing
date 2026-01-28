# Appendix A: Comprehensive Load Sensitivity Analysis (Priority 1)

## A.1 Motivation and Research Questions

The capacity paradox finding—where K=10 outperforms K=30 at extreme loads—raised critical questions about the generalizability and robustness of this phenomenon. This appendix presents a comprehensive load sensitivity analysis to address:

1. **Transition Point Identification**: At what load multiplier does the capacity paradox emerge?
2. **Consistency Across Loads**: Is the paradox consistent across different load levels?
3. **Algorithm Robustness**: Do both A2C and PPO exhibit the same pattern?

## A.2 Experimental Design

### A.2.1 Configuration

**Capacity Configurations**:
- K=10: Uniform distribution [2,2,2,2,2]
- K=30: Uniform distribution [6,6,6,6,6]

**Load Multipliers**: 3×, 4×, 6×, 7×, 8×, 9×, 10× (7 levels)

**Algorithms**: A2C, PPO

**Training**: 100,000 timesteps per run

**Evaluation**: 50 episodes per run

**Seeds**: 42, 43, 44, 45, 46 (n=5 independent runs)

**Total Runs**: 7 loads × 2 capacities × 2 algorithms × 5 seeds = **140 runs**

### A.2.2 Rationale

This design systematically varies load from moderate (3×) to extreme (10×) to identify:
- The critical transition point where K=10 begins to outperform K=30
- The stability of each capacity configuration across load levels
- The consistency of findings across two state-of-the-art algorithms

## A.3 Results

### A.3.1 Overview

The load sensitivity analysis reveals a clear three-phase pattern in the relationship between capacity and performance. At low loads (3-4×), K=30 significantly outperforms K=10 as expected by conventional queueing theory. However, a critical transition occurs at moderate loads (6-7×), where K=10 begins to outperform K=30. At extreme loads (8-10×), this capacity paradox becomes dramatic, with K=10 achieving stable performance while K=30 experiences complete system collapse.

### A.3.2 Summary Results

Table A.1 presents the mean rewards and crash rates across all load levels for both capacity configurations, averaged across A2C and PPO algorithms (n=10 per load level).

**Table A.1**: Performance comparison of K=10 vs K=30 across load multipliers

| Load | K=10 Reward | K=30 Reward | K=30 Crash Rate | Winner | Advantage |
|------|-------------|-------------|-----------------|--------|-----------|
| 3×   | 280,243     | 595,015     | 0%              | K=30   | +112%     |
| 4×   | 314,934     | 759,930     | 0%              | K=30   | +141%     |
| 6×   | 400,327     | 343,148     | 84%             | K=10   | +17%      |
| 7×   | 444,220     | 138,135     | 97%             | K=10   | +222%     |
| 8×   | 485,587     | 69,392      | 95%             | K=10   | +600%     |
| 9×   | 523,505     | 28.6        | 100%            | K=10   | +1,830,000%|
| 10×  | 558,555     | 16.9        | 100%            | K=10   | +3,304,000%|

**Key Findings**:
- **Transition point**: Between 4× and 6× load multiplier
- **Crash rate correlation**: K=30 crash rate increases from 0% (4×) to 84% (6×) to 100% (9-10×)
- **K=10 stability**: Maintains 0% crash rate across all load levels
- **Performance trend**: K=10 rewards increase monotonically with load (280K → 558K)

### A.3.3 Phase 1: Low Load (3-4×) - Conventional Behavior

At low load levels, the system behaves according to conventional queueing theory expectations:

**3× Load**:
- K=30 achieves 595,015 mean reward vs K=10's 280,243 (+112% advantage)
- Both configurations maintain 0% crash rate
- K=30's larger capacity enables higher throughput without stability issues

**4× Load**:
- K=30 achieves 759,930 mean reward vs K=10's 314,934 (+141% advantage)
- Both configurations remain stable (0% crash rate)
- K=30's advantage increases with load, as expected

**Interpretation**: At moderate loads, larger capacity provides clear benefits. The system can handle increased arrival rates without coordination challenges overwhelming the benefits of additional capacity.

### A.3.4 Phase 2: Transition (6-7×) - Capacity Paradox Emerges

The critical transition occurs between 6× and 7× load, where the capacity paradox first becomes evident:

**6× Load**:
- K=10 achieves 400,327 mean reward vs K=30's 343,148 (+17% advantage)
- K=30 crash rate jumps to 84% (from 0% at 4×)
- First load level where K=10 outperforms K=30

**7× Load**:
- K=10 achieves 444,220 mean reward vs K=30's 138,135 (+222% advantage)
- K=30 crash rate increases to 97%
- K=10's advantage becomes substantial

**Interpretation**: The transition reveals that coordination complexity in K=30 systems becomes overwhelming at moderate-high loads. The RL agents trained on K=30 struggle to maintain system stability, leading to frequent crashes that negate the throughput benefits of larger capacity.

### A.3.5 Phase 3: Extreme Load (8-10×) - Complete System Collapse

At extreme loads, the capacity paradox becomes dramatic, with K=30 experiencing complete system failure:

**8× Load**:
- K=10 achieves 485,587 mean reward vs K=30's 69,392 (+600% advantage)
- K=30 crash rate: 95%
- K=10 continues to improve performance

**9× Load**:
- K=10 achieves 523,505 mean reward vs K=30's 28.6 (+1,830,000% advantage)
- K=30 crash rate: 100%
- K=30 systems crash in nearly every episode

**10× Load**:
- K=10 achieves 558,555 mean reward vs K=30's 16.9 (+3,304,000% advantage)
- K=30 crash rate: 100%
- K=30 completely unable to maintain stability

**Interpretation**: At extreme loads, K=30 systems experience catastrophic failure. The coordination complexity becomes insurmountable, and RL agents cannot learn stable policies. In contrast, K=10's simpler state space enables robust learning and stable operation even under extreme stress.

## A.4 Statistical Analysis

### A.4.1 Algorithm Consistency

Both A2C and PPO exhibit nearly identical patterns across all load levels, demonstrating the robustness of the capacity paradox finding:

**Correlation between A2C and PPO performance**:
- K=10 configurations: r = 0.998 (p < 0.001)
- K=30 configurations: r = 0.995 (p < 0.001)

**Mean absolute difference between algorithms**:
- K=10: 1,247 reward units (0.3% of mean)
- K=30: 3,892 reward units (1.2% of mean)

**Interpretation**: The capacity paradox is algorithm-independent, suggesting it reflects fundamental properties of the system dynamics rather than algorithm-specific learning biases.

### A.4.2 Variance and Stability

Within-configuration variance reveals important stability differences:

**K=10 Standard Deviations**:
- 3× load: 126,248 (40% of mean)
- 7× load: 152 (0.03% of mean)
- 10× load: 265 (0.05% of mean)

**K=30 Standard Deviations**:
- 3× load: 347 (0.06% of mean)
- 7× load: 75,363 (55% of mean)
- 10× load: 1.06 (6% of mean, but mean is only 17)

**Interpretation**: K=10 shows high variance at low loads (exploration phase) but becomes extremely stable at high loads. K=30 shows the opposite pattern: stable at low loads but highly unstable at moderate-high loads where crashes occur.

## A.5 Discussion

### A.5.1 The Transition Point

The critical transition between 4× and 6× load multiplier represents a fundamental shift in system dynamics:

**Below 4× load**: System operates in the "capacity-advantaged" regime where larger capacity provides clear benefits. Coordination complexity is manageable, and throughput gains dominate.

**Between 4× and 6× load**: System enters the "complexity-dominated" regime where coordination challenges begin to overwhelm capacity benefits. This is the critical threshold where the capacity paradox emerges.

**Above 6× load**: System fully enters the "paradox regime" where smaller capacity consistently outperforms larger capacity due to stability advantages.

### A.5.2 Mechanisms Behind the Paradox

Three interrelated mechanisms explain the capacity paradox:

**1. State Space Complexity**: K=30 systems have exponentially larger state spaces (30^5 vs 10^5 possible configurations), making it harder for RL agents to learn effective policies within the same training budget (100K timesteps).

**2. Coordination Overhead**: Larger capacity requires more complex coordination decisions. At high loads, the cognitive burden of managing 30 units across 5 queues overwhelms the benefits of additional throughput capacity.

**3. Cascading Failures**: When K=30 systems begin to fail, they fail catastrophically. Queue imbalances in high-capacity systems lead to rapid cascading failures that the RL agent cannot recover from, resulting in 100% crash rates at extreme loads.

### A.5.3 Practical Implications

This load sensitivity analysis provides actionable guidance for system designers:

**Capacity Planning**: The transition point (4-6× load) should inform capacity decisions. Systems expected to operate below this threshold can benefit from larger capacity, while systems facing higher loads should prioritize smaller, more manageable capacity configurations.

**Load Forecasting**: Understanding the load regime is critical. A system designed for 4× load with K=30 capacity will fail catastrophically if load increases to 7×, whereas a K=10 system would handle the increase gracefully.

**Training Budget**: The 100K timestep training budget may be insufficient for K=30 systems to learn stable policies at high loads. Extended training (explored in optional experiments) may shift the transition point.

## A.6 Conclusions

This comprehensive load sensitivity analysis (140 runs across 7 load levels) provides definitive evidence for the capacity paradox and identifies its critical transition point:

**Key Findings**:
1. **Transition point identified**: The capacity paradox emerges between 4× and 6× load multiplier
2. **Three-phase pattern**: Capacity-advantaged (3-4×) → Complexity-dominated (6-7×) → Paradox regime (8-10×)
3. **Algorithm-independent**: Both A2C and PPO exhibit identical patterns (r > 0.99)
4. **Catastrophic failure mode**: K=30 experiences 100% crash rates at 9-10× load

**Implications for Main Study**:
- The 10× load condition used in the main study represents the extreme end of the paradox regime
- The capacity paradox is not an artifact of a single load level but a robust phenomenon across a range of high loads
- System designers should carefully consider expected load regimes when making capacity decisions

**Future Work**: Extended training experiments (500K timesteps) will test whether the transition point can be shifted through additional learning, or whether the paradox reflects fundamental system properties that persist regardless of training duration.

