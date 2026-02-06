# Complete Data Analysis Report
## Priority 1: Load Sensitivity Analysis

**Experiment Date**: 2026-01-17
**Analysis Time**: 16:23
**Data Scale**: 220 runs (7 load levels × 2 capacities × 2 algorithms × 5 seeds + heuristics)

---

## 1. Capacity Paradox Emergence Point Analysis

### Key Finding
🎯 **Capacity paradox emerges between 4× and 6× load**

| Load | K=10 Reward | K=30 Reward | Difference | Winner | Significance |
|------|-------------|-------------|------------|--------|--------------|
| 3×   | 280,244     | 595,016     | -314,772   | K=30   | p<0.001 *** |
| 4×   | 314,934     | 759,930     | -444,997   | K=30   | p<0.001 *** |
| **6×**   | **400,327**     | **343,148**     | **+57,179**    | **K=10**   | **p=0.025 *** |
| 7×   | 444,220     | 138,135     | +306,085   | K=10   | p<0.001 *** |
| 8×   | 485,587     | 69,392      | +416,196   | K=10   | p<0.001 *** |
| 9×   | 523,505     | 29          | +523,477   | K=10   | p<0.001 *** |
| 10×  | 558,555     | 17          | +558,538   | K=10   | p<0.001 *** |

### Statistical Significance
- **Low load (3×-4×)**: K=30 significantly outperforms K=10, Cohen's d > 4.0 (very large effect)
- **Turning point (6×)**: K=10 starts to surpass K=30, Cohen's d = 1.093 (medium effect)
- **High load (7×-10×)**: K=10 extremely significantly outperforms K=30, Cohen's d > 5.0 (extremely large effect)

---

## 2. System Stability Analysis

### Crash Rate Comparison

| Load | K=10 Crash | K=30 Crash | Difference |
|------|------------|------------|------|
| 3×   | 0.0%       | 0.0%       | Stable |
| 4×   | 0.0%       | 0.0%       | Stable |
| 6×   | 0.0%       | 84.6%      | ⚠️ K=30 starts crashing |
| 7×   | 0.0%       | 97.2%      | ❌ K=30 severe crashes |
| 8×   | 0.0%       | 95.0%      | ❌ K=30 severe crashes |
| 9×   | 0.0%       | 100.0%     | ❌ K=30 complete crash |
| 10×  | 0.0%       | 100.0%     | ❌ K=30 complete crash |

**Key Insights**:
- K=10 maintains 0% crash rate across all loads, showing excellent robustness
- K=30 starts showing high crash rate (84.6%) at load≥6×
- K=30 completely crashes (100%) at load≥9×

---

## 3. Heuristic Baselines Comparison Analysis

### Load 6× Performance Ranking

**K=10:**
1. SJF: 491,275 (0% crash) ⭐ **Best**
2. PPO: 400,464 (0% crash)
3. A2C: 400,190 (0% crash)
4. FCFS: -48,232 (0% crash)
5. Priority: -68 (100% crash)
6. Heuristic: -508 (100% crash)

**K=30:**
1. PPO: 347,612 (83.2% crash)
2. A2C: 338,684 (86.0% crash)
3. All others completely crash (100% crash)

### Load 7× Performance Ranking

**K=10:**
1. PPO: 444,385 (0% crash) ⭐ **Best**
2. A2C: 444,055 (0% crash)
3. All others completely crash (100% crash)

**K=30:**
1. A2C: 155,279 (97.6% crash)
2. PPO: 120,991 (96.8% crash)
3. All others completely crash (100% crash)

### Key Findings

1. **SJF excels at low-medium load**: At Load 6×, K=10 conditions, SJF reaches 491K reward, exceeding RL algorithms by 18.5%

2. **RL algorithms dominate at high load**: At Load 7×, all heuristics completely crash under K=10, only RL algorithms can run stably

3. **RL algorithms still have advantage under K=30**: Although crash rate is high, RL algorithms can still achieve positive returns under K=30, while heuristics all crash

---

## 4. Key Data Points for Paper

### Core Numbers for Paper

1. **Capacity paradox turning point**: Around load 5× (between 4× and 6×)

2. **K=10 advantage magnitude**:
   - Load 6×: +57K (+16.7%)
   - Load 7×: +306K (+222%)
   - Load 10×: +558K (+33,000 times)

3. **Stability comparison**:
   - K=10: 0% crash across all loads
   - K=30: 84.6% → 100% crash at loads ≥6×

4. **RL vs Heuristics**:
   - Load 6×, K=10: SJF best (491K)
   - Load 7×, K=10: RL exclusive (444K), heuristics all crash
   - RL algorithms show significant advantage at high loads

---

## 5. Statistical Test Results

All load level K=10 vs K=30 comparisons reach statistical significance (p < 0.05)

**Effect Size (Cohen's d)**:
- Load 3×-4×: d < -4.0 (K=30 advantage, very large effect)
- Load 6×: d = 1.093 (K=10 advantage, medium effect)
- Load 7×-10×: d > 5.0 (K=10 advantage, extremely large effect)

---

## 6. Next Steps

⏳ **To be completed**:
1. Priority 2: Structural comparison generalization analysis (inverted vs normal pyramid)
2. Interaction effect analysis (load × capacity × algorithm)
3. Update paper charts and tables

✅ **Completed**:
- Complete load sensitivity analysis (7 load levels)
- Heuristic baselines comparison
- Statistical significance testing
- Visualization chart generation
