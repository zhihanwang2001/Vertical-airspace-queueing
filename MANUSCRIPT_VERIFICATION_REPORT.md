# Manuscript Data Verification Report

## Summary

- Total claims verified: 10
- ❌ Major: 6
- ✅ Match: 2
- ⚠️ Minor: 1
- ❌ Data Missing: 1

## Detailed Verification Results

| Category              | Claim                             | Manuscript   | Actual    | Diff (%)   | Status          | Source                                |
|:----------------------|:----------------------------------|:-------------|:----------|:-----------|:----------------|:--------------------------------------|
| Algorithm Performance | A2C Mean Reward                   | 4437.86      | 4391.40   | 1.05%      | ✅ Match        | Results/excel/SB3_A2C.csv             |
| Algorithm Performance | PPO Mean Reward                   | 4419.98      | 3281.15   | 25.77%     | ❌ Major        | Results/excel/SB3_PPO.csv             |
| Structural Comparison | A2C Inverted Pyramid              | 447,683      | 723,337   | 61.57%     | ❌ Major        | structural_5x_per_seed.csv            |
| Structural Comparison | A2C Normal Pyramid                | 387,514      | 661,165   | 70.62%     | ❌ Major        | structural_5x_per_seed.csv            |
| Structural Comparison | PPO Inverted Pyramid              | 445,892      | 722,568   | 62.05%     | ❌ Major        | structural_5x_per_seed.csv            |
| Structural Comparison | PPO Normal Pyramid                | 388,321      | 659,198   | 69.76%     | ❌ Major        | structural_5x_per_seed.csv            |
| Capacity Paradox      | K10 A2C Reward                    | 11,180       | 11,146    | 0.30%      | ✅ Match        | all_experiments_summary.csv           |
| Capacity Paradox      | K30 A2C Reward                    | 13           | 14        | 6.00%      | ⚠️ Minor        | all_experiments_summary.csv           |
| Capacity Paradox      | K40 A2C Reward                    | -245         | -30       | 87.67%     | ❌ Major        | all_experiments_summary.csv           |
| Statistical Claims    | 59.9% improvement over heuristics | 59.9%        | NOT FOUND | N/A        | ❌ Data Missing | Heuristic baseline data not available |

## Critical Findings

### Major Discrepancies (❌ Major)

| Category              | Claim                | Manuscript   | Actual   | Diff (%)   | Status   | Source                      |
|:----------------------|:---------------------|:-------------|:---------|:-----------|:---------|:----------------------------|
| Algorithm Performance | PPO Mean Reward      | 4419.98      | 3281.15  | 25.77%     | ❌ Major | Results/excel/SB3_PPO.csv   |
| Structural Comparison | A2C Inverted Pyramid | 447,683      | 723,337  | 61.57%     | ❌ Major | structural_5x_per_seed.csv  |
| Structural Comparison | A2C Normal Pyramid   | 387,514      | 661,165  | 70.62%     | ❌ Major | structural_5x_per_seed.csv  |
| Structural Comparison | PPO Inverted Pyramid | 445,892      | 722,568  | 62.05%     | ❌ Major | structural_5x_per_seed.csv  |
| Structural Comparison | PPO Normal Pyramid   | 388,321      | 659,198  | 69.76%     | ❌ Major | structural_5x_per_seed.csv  |
| Capacity Paradox      | K40 A2C Reward       | -245         | -30      | 87.67%     | ❌ Major | all_experiments_summary.csv |

### Data Availability Issues

| Category           | Claim                             | Manuscript   | Actual    | Diff (%)   | Status          | Source                                |
|:-------------------|:----------------------------------|:-------------|:----------|:-----------|:----------------|:--------------------------------------|
| Statistical Claims | 59.9% improvement over heuristics | 59.9%        | NOT FOUND | N/A        | ❌ Data Missing | Heuristic baseline data not available |