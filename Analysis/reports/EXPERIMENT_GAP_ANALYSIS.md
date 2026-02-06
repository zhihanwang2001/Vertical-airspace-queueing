# Experiment Completeness Analysis Report

**Analysis Date**: January 17, 2026
**Purpose**: Identify experiment gaps, ensure manuscript completeness

---

## Part 1: Existing Experiment Data Inventory

### 1.1 Completed Experiments

#### Experiment 1: Algorithm Comparison (15 algorithms)
- **Configuration**: Default inverted pyramid [8,6,4,3,2], baseline load
- **Algorithms**: A2C, PPO, TD7, SAC, TD3, R2D2, Rainbow, IMPALA, DDPG + 4 heuristics
- **Training**: 500K timesteps, 5 seeds
- **Status**: ✅ Complete
- **Data Location**: Results/comparison/

#### Experiment 2: Structural Comparison (Inverted vs Normal)
- **Configuration**: 5× load, K=23 total
- **Structure**: Inverted [8,6,4,3,2] vs Normal [2,3,4,6,8]
- **Algorithms**: A2C + PPO
- **Sample**: n=30 per algorithm per structure (total n=60 per structure)
- **Status**: ✅ Complete
- **Key Results**: Inverted outperforms Normal by 9.5% (p<1e-134, d=48.452)

#### Experiment 3: Capacity Scan at 5× Load
- **Configuration**: K∈{10, 30}, load=5×, uniform shape
- **Algorithms**: A2C, PPO + 4 heuristics
- **Seeds**: n=5
- **Status**: ✅ Complete
- **Key Findings**:
  - K=10: 352K reward, 0% crash
  - K=30: 737K reward, 12-17% crash
  - **Note**: K=30 actually performs better at 5× load!

#### Experiment 4: Capacity Scan at 6× Load (Running)
- **Configuration**: K∈{10, 30}, load=6×,7×, uniform shape
- **Algorithms**: A2C, PPO + 4 heuristics
- **Seeds**: n=5
- **Status**: 🔄 Running (started at 07:58, estimated 8-12 hours to complete)
- **Log**: logs/capacity_uniform_6_7_5s_100000t_50e.log

#### Experiment 5: Capacity Scan - Reverse Pyramid at 6× Load
- **Configuration**: K∈{10, 30}, load=6×, reverse pyramid [2,3,4,6,8]
- **Algorithms**: A2C, PPO + 4 heuristics
- **Seeds**: n=5
- **Status**: ✅ Complete
- **Data**: capacity_scan_summary_reverse_6.csv
- **Key Findings**:
  - K=10: 353K reward, 0% crash
  - K=30: 207K reward (A2C), 96-98% crash rate
  - **Important**: At 6× load, reverse pyramid already shows capacity paradox signs!

---

## Part 2: Key Findings and Problem Identification

### 2.1 Load Dependence of Capacity Paradox

**Observed Phenomena**:
- **5× load**: K=30 > K=10 (737K vs 352K, uniform)
- **6× load**: K=10 > K=30 (353K vs 207K, reverse pyramid)
- **10× load** (mentioned in Results outline): K=10 >> K=30 (11,180 vs 13)

**Key Question**: At what load level does capacity paradox begin to appear?

**Impact**: This is one of the paper's core findings, requires clear data support!

### 2.2 EJOR Review Requirements

EJOR review report (Major Comment 2) explicitly states:
> "Test across multiple load levels (3x, 5x, 7x, not just 10x)"

**Current Status**:
- ✅ 5× load: Have data
- 🔄 6×, 7× load: Running
- ❌ 3×, 4×, 8×, 9×, 10× load: **Missing**

---

## Part 3: Critical Experiment Gaps Identification

### Gap 1: Load Sensitivity Analysis ⚠️ **Highest Priority**

**Missing Content**:
- Systematic load scanning: 1×, 2×, 3×, 4×, 8×, 9×, 10× load
- Especially 10× load data (mentioned in Results outline but data file not found)

**Why Important**:
1. EJOR review explicitly requires
2. Determine critical load point of capacity paradox
3. Key support for paper's core finding

**Impact Scope**: Results Section 5.3 (Capacity Paradox)

**Suggested Experiment**:
```
Configuration: K∈{10, 30}, load∈{3, 4, 8, 9, 10}, uniform shape
Algorithms: A2C, PPO (top performers)
Seeds: n=5
Training: 100K timesteps (consistent with existing capacity scan)
Estimated Time: 5 loads × 2 capacities × 2 algos × 5 seeds = 100 runs × ~7min = ~12 hours
```

### Gap 2: Structural Comparison Load Generalization ⚠️ **High Priority**

**Missing Content**:
- Inverted vs Normal pyramid comparison at different loads
- Currently only have 5× load data

**Why Important**:
1. Verify if 9.5% advantage holds across all loads
2. Address EJOR review concerns about generalization

**Impact Scope**: Results Section 5.2 (Structural Analysis)

**Suggested Experiment**:
```
Configuration: Inverted [8,6,4,3,2] vs Normal [2,3,4,6,8], load∈{3, 7, 10}
Algorithms: A2C, PPO
Seeds: n=5 per algorithm per structure
Training: 100K timesteps
Estimated Time: 3 loads × 2 structures × 2 algos × 5 seeds = 60 runs × ~7min = ~7 hours
```

### Gap 3: Capacity Paradox Training Duration Verification ⚠️ **Medium Priority**

**Missing Content**:
- Test with longer training time (500K) whether K=30 still fails at 10× load

**Why Important**:
1. Rule out "insufficient training" hypothesis
2. Prove capacity paradox is system property not training artifact

**Impact Scope**: Results Section 5.3.3 (Theoretical Explanation)

**Suggested Experiment**:
```
Configuration: K∈{10, 30}, load=10, uniform shape
Algorithms: A2C, PPO
Seeds: n=5
Training: 500K timesteps (5× current)
Estimated Time: 2 capacities × 2 algos × 5 seeds = 20 runs × ~35min = ~12 hours
```

---

## Part 4: Priority Summary and Action Recommendations

### 4.1 Must-Complete Experiments (Before Applied Soft Computing Submission)

#### ✅ Priority 1: Load Sensitivity Analysis (Gap 1)
**Rationale**:
- EJOR review explicitly requires
- Key support for paper's core finding
- Determine critical point of capacity paradox

**Action**: Run capacity scan for load∈{3, 4, 8, 9, 10}
**Time**: ~12 hours
**Urgency**: ⚠️ **Must Complete**

#### ✅ Priority 2: Structural Comparison Load Generalization (Gap 2)
**Rationale**:
- Verify universality of 9.5% advantage
- Strengthen persuasiveness of Results Section 5.2

**Action**: Test Inverted vs Normal at load∈{3, 7, 10}
**Time**: ~7 hours
**Urgency**: ⚠️ **Strongly Recommended**

### 4.2 Optional Supplementary Experiments (Enhance Paper Quality)

#### ⭕ Priority 3: Training Duration Verification (Gap 3)
**Rationale**: Rule out "insufficient training" hypothesis
**Action**: Retest K=30 at 10× load with 500K timesteps
**Time**: ~12 hours
**Urgency**: 🔵 **Recommended but Not Required**

---

## Part 5: Specific Execution Plan

### 5.1 Priority 1: Load Sensitivity Analysis

**Command Line**:
```bash
# Run on server
cd /root/RP1

# Load 3×, 4×
python3 Code/training_scripts/run_capacity_scan.py \
  --capacities 10,30 \
  --loads 3,4 \
  --shape uniform \
  --algos A2C,PPO \
  --n-seeds 5 \
  --timesteps 100000 \
  --eval-episodes 50 \
  > logs/capacity_uniform_3_4_5s_100000t_50e.log 2>&1 &

# Load 8×, 9×, 10×
python3 Code/training_scripts/run_capacity_scan.py \
  --capacities 10,30 \
  --loads 8,9,10 \
  --shape uniform \
  --algos A2C,PPO \
  --n-seeds 5 \
  --timesteps 100000 \
  --eval-episodes 50 \
  > logs/capacity_uniform_8_9_10_5s_100000t_50e.log 2>&1 &
```

**Estimated Completion Time**: 2 batches × 6 hours = 12 hours

### 5.2 Priority 2: Structural Comparison Load Generalization

**Command Line**:
```bash
# Load 3×, 7×, 10× - Inverted vs Normal
python3 Code/training_scripts/run_structural_comparison_5x_load.py \
  --loads 3,7,10 \
  --algos A2C,PPO \
  --n-seeds 5 \
  --timesteps 100000 \
  > logs/structural_comparison_3_7_10.log 2>&1 &
```

**Note**: Need to modify `run_structural_comparison_5x_load.py` to support multiple load parameters

**Estimated Completion Time**: ~7 hours

---

## Part 6: Timeline Planning

### Option A: Conservative Plan (Complete Priority 1+2)
- **Day 1**: Wait for current experiments to complete (6×,7× load)
- **Day 2**: Run Priority 1 experiments (3×,4×,8×,9×,10× load)
- **Day 3**: Run Priority 2 experiments (structural comparison)
- **Day 4**: Data analysis and visualization
- **Total Time**: 4 days

### Option B: Aggressive Plan (Priority 1 Only)
- **Day 1**: Wait for current experiments to complete, immediately start Priority 1
- **Day 2**: Data analysis, start writing manuscript
- **Total Time**: 2 days

### Option C: Perfect Plan (Priority 1+2+3)
- **Day 1-2**: Priority 1+2
- **Day 3**: Priority 3 (training duration verification)
- **Day 4**: Data analysis
- **Total Time**: 5 days

---

## Part 7: Final Recommendation

### Recommended Plan: **Option A (Conservative Plan)**

**Rationale**:
1. ✅ Meets EJOR review requirements (multi-load testing)
2. ✅ Verifies universality of core findings (9.5% advantage)
3. ✅ Provides complete capacity paradox evolution curve
4. ⏱️ Time controllable (4 days)
5. 📊 Data sufficiently supports Applied Soft Computing submission

**Why Not Recommend Priority 3**:
- High time cost (additional 12 hours)
- Not required for submission
- Can be mentioned as future work in Discussion

---

## Summary

**Current Status**: Solid experimental foundation, but critical gaps exist
**Must Supplement**: Priority 1 (Load sensitivity)
**Strongly Recommended**: Priority 2 (Structural generalization)
**Optional**: Priority 3 (Training duration verification)

**Action Recommendation**: Adopt Option A, complete all necessary experiments within 4 days, then focus on manuscript writing.

**Estimated Submission Time**: End of January (4 days experiments + 2-3 weeks writing)

---

**Report Completion Time**: January 17, 2026
