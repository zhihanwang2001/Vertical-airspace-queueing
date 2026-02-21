# HCA2C Final Comparison Analysis Report

**Generated:** 2026-02-21 17:05:09

**Total Experiments:** 45
- Algorithms: A2C, HCA2C, PPO
- Seeds: 42, 43, 44, 45, 46
- Loads: 3.0, 5.0, 7.0

## Descriptive Statistics

| Algorithm | Load | n | Mean ± SD | CV (%) | Crash Rate | Time (min) |
|-----------|------|---|-----------|--------|------------|------------|
| A2C | 3.0× | 5 | 428603.9 ± 174782.0 | 36.47 | 0.000 | 0.7 |
| A2C | 5.0× | 5 | 771222.5 ± 1646.8 | 0.19 | 0.000 | 0.7 |
| A2C | 7.0× | 5 | 112518.7 ± 60377.4 | 47.99 | 0.000 | 3.3 |
| HCA2C | 3.0× | 5 | 228878.8 ± 262.1 | 0.10 | 0.000 | 138.7 |
| HCA2C | 5.0× | 5 | 79457.9 ± 228.6 | 0.26 | 0.000 | 139.0 |
| HCA2C | 7.0× | 5 | -134253.8 ± 470.7 | 0.31 | 0.000 | 87.1 |
| PPO | 3.0× | 5 | 411085.5 ± 41963.8 | 9.13 | 0.000 | 0.7 |
| PPO | 5.0× | 5 | 482715.5 ± 57380.8 | 10.63 | 0.000 | 0.7 |
| PPO | 7.0× | 5 | 85312.4 ± 69.9 | 0.07 | 0.000 | 4.1 |

## Pairwise Comparisons

| Load | Comparison | Mean Diff | t-stat | p-value | Cohen's d | Significant |
|------|------------|-----------|--------|---------|-----------|-------------|
| 3.0× | HCA2C vs A2C | -199725.1 | -2.555 | 0.0630 | -1.616 | No |
| 3.0× | HCA2C vs PPO | -182206.7 | -9.709 | 0.0006 | -6.140 | Yes |
| 3.0× | A2C vs PPO | 0.0 | 0.218 | 0.8371 | 0.138 | No |
| 5.0× | HCA2C vs A2C | -691764.5 | -930.373 | 0.0000 | -588.420 | Yes |
| 5.0× | HCA2C vs PPO | -403257.6 | -15.714 | 0.0001 | -9.939 | Yes |
| 5.0× | A2C vs PPO | 0.0 | 11.238 | 0.0004 | 7.108 | Yes |
| 7.0× | HCA2C vs A2C | -246772.5 | -9.139 | 0.0008 | -5.780 | Yes |
| 7.0× | HCA2C vs PPO | -219566.2 | -1031.695 | 0.0000 | -652.501 | Yes |
| 7.0× | A2C vs PPO | 0.0 | 1.008 | 0.3707 | 0.637 | No |

## Key Findings

- **Load 3.0×**: A2C achieves highest mean reward (428603.9 ± 174782.0)
- **Load 5.0×**: A2C achieves highest mean reward (771222.5 ± 1646.8)
- **Load 7.0×**: A2C achieves highest mean reward (112518.7 ± 60377.4)

### Stability (Coefficient of Variation)
- **A2C**: Average CV = 28.22% (Less stable)
- **HCA2C**: Average CV = 0.22% (Most stable)
- **PPO**: Average CV = 6.61% (Less stable)