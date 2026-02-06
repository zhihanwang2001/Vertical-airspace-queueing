# Final Ablation Study Data Analysis Report

**Generation Time**: 2026-01-05
**Total Experiments**: 21 (7 configurations × 3 algorithms)
**Evaluation Episodes**: 50 episodes
**High Load Multiplier**: 10x

---

## I. Experiment Overview

### Capacity Configuration Distribution

| Total Capacity | Capacity Distribution | Configuration Type | Number of Algorithms |
|--------|---------|---------|--------|
| 10 | [2, 2, 2, 2, 2] | Low capacity | 3 |
| 20 | [4, 4, 4, 4, 4] | Uniform20 | 3 |
| 23 | [8, 6, 4, 3, 2] | Inverted pyramid | 3 |
| 25 | [5, 5, 5, 5, 5] | Uniform | 3 |
| 30 | [6, 6, 6, 6, 6] | Uniform30 | 3 |
| 40 | [8, 8, 8, 8, 8] | High capacity | 3 |

## II. Key Findings

### 1. Optimal Capacity Configuration

**Ranked by Average Reward (A2C+PPO)**:

| Rank | Configuration | Type | Total Capacity | Avg Reward | Avg Crash Rate | Avg Completion Rate |
|------|------|------|--------|---------|-----------|-----------|
| 1 🥇 | low_capacity | Low capacity | 10 | 11180.17 | 0.0% | 100.0% |
| 2 🥈 | capacity_4x5 | Uniform20 | 20 | 10854.55 | 10.0% | 90.0% |
| 3 🥉 | inverted_pyramid | Inverted pyramid | 23 | 8843.70 | 29.0% | 71.0% |
| 4  | uniform | Uniform | 25 | 7817.07 | 35.0% | 65.0% |
| 5  | reverse_pyramid | Normal pyramid | 23 | 3950.14 | 65.0% | 35.0% |
| 6  | capacity_6x5 | Uniform30 | 30 | 13.50 | 100.0% | 0.0% |
| 7  | high_capacity | High capacity | 40 | -32.41 | 100.0% | 0.0% |

**✅ Optimal Configuration**: `low_capacity` (Low capacity, total capacity 10)
- Average Reward: 11180.17
- Average Crash Rate: 0.0%
- Average Completion Rate: 100.0%

### 2. Algorithm Performance Comparison

| Algorithm | Avg Reward | Reward Std Dev | Avg Crash Rate | Avg Completion Rate | Avg Episode Length |
|------|---------|-----------|-----------|-----------|----------------|
| A2C | 6454.84 | 4411.80 | 40.6% | 59.4% | 128.74 |
| PPO | 5724.23 | 4646.93 | 56.3% | 43.7% | 108.60 |
| TD7 | 375294.33 | 244253.89 | 28.6% | 71.4% | 7143.14 |

### 3. Capacity Effect Analysis

**Feasible Configurations (Non-100% crash)**:

| Total Capacity | Avg Reward | Avg Crash Rate | Status |
|--------|---------|-----------|------|
| 10 | 11180.17 | 0.0% | ✅ Excellent |
| 20 | 10854.55 | 10.0% | ✅ Excellent |
| 23 | 6396.92 | 47.0% | ⚠️ Usable |
| 25 | 7817.07 | 35.0% | ⚠️ Usable |

**Key Observation**: Capacity ≤ 25 can maintain system stability under 10x high load
