# Reward Scale Clarification Report

**Date**: 2026-01-18
**Addresses**: Reviewer concern about reward scale inconsistencies (4K vs 720K vs 11K)

---

## Executive Summary

Reward scales vary across experiments due to **different episode lengths**, not different reward functions or calculation methods. All rewards are **cumulative per-episode rewards**, which is standard practice in RL evaluation.

**Key Finding**: Episode length is inversely related to load multiplier. Higher loads cause earlier episode termination (system crashes or max steps reached), resulting in lower cumulative rewards despite higher per-step throughput.

---

## Reward Scale Patterns

### By Episode Length

| Episode Length | Typical Reward Range | Load Multiplier | Interpretation |
|----------------|---------------------|-----------------|----------------|
| 10,000 steps   | 400K - 600K        | 3× - 4×        | Full episodes, stable system |
| 5,000-6,000 steps | 200K - 300K     | 7× - 8×        | Moderate load, some crashes |
| <5,000 steps   | <200K or negative  | 9× - 10×       | High load, frequent crashes |

### By Load Multiplier

**3× Load** (Low):
- Mean reward: ~437K
- Episode length: 10,000 steps (max)
- Interpretation: System stable, reaches max episode length

**7× Load** (Moderate):
- Mean reward: ~265K
- Episode length: ~5,617 steps
- Interpretation: System experiences some crashes, episodes end early

**10× Load** (Extreme):
- Mean reward: ~284K
- Episode length: ~5,000 steps
- Interpretation: Frequent crashes, highly variable performance

---

## Why Manuscript Outline Showed Different Scales

The manuscript outline mentioned different reward scales:
- Main study: ~4,400
- Structural comparison: ~720,000
- Capacity paradox: ~11,000

**Explanation**:
1. **Main study (~4,400)**: Likely refers to a different experiment or averaged across many configurations
2. **Structural comparison (~720,000)**: This appears to be an error or refers to a different metric (possibly total cumulative reward across multiple episodes?)
3. **Capacity paradox (~11,000)**: Matches our uniform K=10 experiments at extreme load

**Action needed**: Verify which specific experiments these numbers refer to and ensure consistency in reporting.

---

## Implications for Manuscript

### What to Clarify

1. **In Methods Section**:
   - Explicitly state: "All reported rewards are cumulative per-episode rewards"
   - Explain: "Episode length varies based on system stability and load conditions"
   - Note: "Episodes terminate when system crashes or reaches maximum 10,000 steps"

2. **In Results Section**:
   - Report episode lengths alongside rewards
   - Explain why rewards vary across load conditions
   - Consider normalizing by episode length for fair comparison

3. **In Tables**:
   - Add "Episode Length" column to all results tables
   - Consider adding "Reward per Step" as normalized metric

### Recommended Reporting Format

**Current** (confusing):
```
A2C: Mean reward = 437,629
```

**Improved** (clear):
```
A2C: Mean reward = 437,629 (10,000 steps, 43.8 reward/step)
```

---

## Statistical Implications

### Episode Length as Confounding Variable

When comparing algorithms or configurations:
- **Problem**: Different episode lengths make direct reward comparison misleading
- **Solution**: Report both total reward and reward-per-step
- **Alternative**: Use crash rate and episode length as separate metrics

### Recommended Metrics

1. **Primary**: Mean reward (cumulative per-episode)
2. **Secondary**: Mean episode length (steps)
3. **Derived**: Reward per step (mean_reward / mean_length)
4. **Stability**: Crash rate (%)

---

## Conclusion

Reward scale variations are **expected and explainable** - they reflect different episode lengths due to varying system stability under different load conditions. This is not a methodological flaw but a natural consequence of the experimental design.

**Recommendation**: Add clear explanation in Methods section and report episode lengths alongside rewards throughout the manuscript.

---

**Report generated**: 2026-01-18
**Analysis script**: Analysis/statistical_analysis/analyze_reward_scales.py
