# Peer Review Report: Applied Soft Computing
**Manuscript Title**: Deep Reinforcement Learning for Vertical Layered Queueing Systems in Urban Air Mobility: A Comparative Study of 15 Algorithms

**Review Date**: 2026-01-18

**Reviewer Role**: Acting as Applied Soft Computing peer reviewer

**Manuscript Status**: Outline form (not complete draft)

---

## Executive Summary

This manuscript presents a comprehensive comparative study of 15 deep reinforcement learning algorithms for optimizing vertical layered queueing systems in Urban Air Mobility. The research introduces the MCRPS/D/K queueing framework and reports three main findings: (1) DRL algorithms achieve 50%+ improvement over heuristics, (2) inverted pyramid capacity configuration outperforms normal pyramid by 9.5%, and (3) a capacity paradox where low-capacity systems (K=10) outperform high-capacity systems (K=30+) under extreme load.

**Current Status**: The manuscript is in outline form with detailed section plans but no complete prose. The Abstract is complete (237 words), and comprehensive outlines exist for Introduction, Methods, and Results sections. Missing components include Discussion, Conclusion, Appendices, and actual figures.

**Preliminary Recommendation**: **Major Revision** - The research has strong potential, but critical issues must be addressed before acceptance. The manuscript requires completion of full text, integration of supplementary experiments, and resolution of statistical concerns.

---

## Strengths

### 1. Comprehensive Experimental Design
✓ **Breadth of comparison**: 15 DRL algorithms across 4 categories (policy gradient, actor-critic, value-based, distributed)
✓ **Rigorous methodology**: 500K timesteps per algorithm, 5 independent seeds, 50 evaluation episodes
✓ **Systematic ablation studies**: Structural comparison, capacity scan, generalization testing
✓ **Appropriate baselines**: Four traditional heuristics (FCFS, SJF, Priority, Heuristic) for comparison

### 2. Clear Research Questions and Objectives
✓ **Well-defined scope**: Five specific research objectives clearly stated
✓ **Practical relevance**: Addresses real UAM challenges in vertical airspace management
✓ **Interdisciplinary approach**: Combines queueing theory, DRL, and UAM operations

### 3. Statistical Rigor
✓ **Proper significance testing**: t-tests, p-values, effect sizes (Cohen's d)
✓ **Multiple seeds**: 5 independent seeds for reproducibility
✓ **Confidence intervals**: 95% CIs reported for key comparisons
✓ **Appropriate sample sizes**: n=60 for structural comparison study

### 4. Novel and Counter-Intuitive Findings
✓ **Capacity paradox**: Interesting finding that K=10 outperforms K=30+ at extreme load
✓ **Structural insights**: Inverted pyramid advantage with capacity-flow matching principle
✓ **Practical value**: Actionable design guidelines for UAM infrastructure

### 5. Good Journal Fit
✓ **DRL focus**: Aligns with Applied Soft Computing scope (soft computing techniques)
✓ **Practical application**: Real-world UAM problem with industry relevance
✓ **Accessible presentation**: Measured language avoiding overclaiming
✓ **Interdisciplinary appeal**: Bridges DRL, operations research, and transportation

### 6. Reproducibility Measures
✓ **Fixed random seeds**: 42-46 explicitly stated
✓ **Deterministic evaluation**: No exploration noise during testing
✓ **Complete hyperparameters**: All algorithm settings documented
✓ **Framework specified**: Stable-Baselines3 with version requirements

---

## Major Concerns (Must Address for Acceptance)

### 1. Effect Sizes Extraordinarily Large ⚠️ CRITICAL

**Issue**: Cohen's d values reported are extraordinarily large and require investigation.

**Evidence**:
- Structural comparison: d = 48.452 (Results Table 2)
- DRL vs heuristics: d = 15.678 (Results Section 5.1.3)
- Typical effect sizes: small (0.2), medium (0.5), large (0.8)
- d > 15 suggests either: (1) perfect separation, (2) data issues, (3) calculation errors

**Concern**: Effect sizes this large are virtually never seen in experimental research. They suggest:
- Possible calculation error (using wrong standard deviation formula?)
- Data quality issues (outliers, measurement errors)
- Overfitting or training artifacts
- Unrealistic experimental conditions

**Recommendation**:
1. Verify Cohen's d calculations using pooled standard deviation: d = (μ₁ - μ₂) / σ_pooled
2. Check for data entry errors or outliers
3. If calculations are correct, provide detailed explanation for why effect sizes are so large
4. Consider reporting additional effect size metrics (e.g., Glass's Δ, Hedges' g)
5. Discuss whether such large effects are realistic or indicate experimental artifacts

**Impact**: This is the most serious concern and must be resolved before acceptance.

---

### 2. Reward Scale Inconsistencies ⚠️ CRITICAL

**Issue**: Reward scales vary dramatically across different experiments without explanation.

**Evidence**:
- Main study (Table 1): A2C = 4,437.86 reward
- Structural comparison (Table 2): Inverted = 722,952.90 reward
- Capacity paradox (Table 3): K=10 = 11,180 reward

**Concern**: These vastly different scales (4K vs 720K vs 11K) suggest:
- Different experiments with different reward functions
- Different episode lengths or evaluation protocols
- Inconsistent reporting (cumulative vs per-step rewards?)
- Lack of clarity about what "reward" means in each context

**Recommendation**:
1. Explicitly state whether rewards are per-episode, per-step, or cumulative
2. Clarify if different experiments use different reward functions
3. Explain why scales differ by orders of magnitude
4. Consider normalizing rewards for fair comparison
5. Add clear definitions in Methods section

**Impact**: Without clarification, readers cannot interpret or compare results across experiments.

---

### 3. Missing Supplementary Experiments Integration

**Issue**: Supplementary experiments (Priority 1 and Priority 2) are not integrated into manuscript.

**Evidence**:
- No appendix files found in manuscript directory
- Preparation summary mentions "100K timesteps may be insufficient" (line 280)
- Earlier context suggests Priority 1 (load sensitivity) and Priority 2 (structural comparison) experiments were completed

**Concern**:
- Supplementary experiments appear to use different configurations than main study
- Priority experiments: [3,3,2,1,1] capacity, 100K timesteps, n=5 seeds
- Main study: [8,6,4,3,2] capacity, 500K timesteps, n=60
- Relationship between main study and supplementary experiments unclear

**Recommendation**:
1. Create appendices integrating Priority 1 and Priority 2 experiments
2. Clearly explain relationship between main study and supplementary experiments
3. Justify why different configurations were used
4. Ensure consistency in reporting across all experiments

**Impact**: Missing appendices leave gaps in the experimental validation.

---

### 4. Manuscript Completeness

**Issue**: Manuscript is in outline form, not complete draft ready for submission.

**Missing Components**:
- Discussion section (2-3 pages needed)
- Conclusion section (1 page needed)
- Full prose for Introduction, Methods, Results (currently outlines only)
- Literature review expansion (30-40 references needed)
- All 6 figures (only specifications provided, no actual figures)
- All 4 tables (only outlines provided)
- Appendices (supplementary experiments)

**Recommendation**:
1. Complete full text expansion (estimated 2-3 weeks per preparation summary)
2. Generate all figures at publication quality (≥300 DPI)
3. Create properly formatted tables
4. Write Discussion section interpreting findings
5. Add comprehensive literature review with citations

**Impact**: Cannot be submitted in current form. Requires substantial additional work.

---

## Moderate Concerns (Should Address to Strengthen Paper)

### 1. Capacity Paradox Explanation Needs Strengthening

**Issue**: The capacity paradox finding is interesting but the theoretical explanation needs more support.

**Current Explanation** (Results Section 5.3.3):
- Hypothesis 1: State space complexity increases with capacity
- Hypothesis 2: Exploration challenge in high-capacity systems
- Hypothesis 3: System dynamics favor aggressive control

**Concerns**:
- Explanations are plausible but not rigorously tested
- Alternative explanations not considered (e.g., reward function design, training hyperparameters)
- Could this be a training artifact rather than fundamental phenomenon?
- Would longer training (>100K timesteps) resolve the paradox?

**Recommendation**:
1. Test capacity paradox with extended training (500K timesteps) to rule out training artifacts
2. Analyze learning curves for different capacity levels
3. Consider alternative explanations and rule them out systematically
4. Provide more rigorous theoretical justification or acknowledge as empirical observation

### 2. Sample Size Clarity and Consistency

**Issue**: Relationship between seeds (n=5) and sample sizes (n=30, n=60) is unclear.

**Evidence**:
- Methods states: "5 seeds (42, 43, 44, 45, 46)"
- Results Table 2 states: "n=30 per algorithm per structure"
- How does n=5 seeds become n=30 samples?

**Possible Explanations**:
- 6 different configurations × 5 seeds = 30 samples?
- Multiple evaluation episodes counted as separate samples?
- Different experiments use different sample sizes?

**Recommendation**:
1. Explicitly explain how n=5 seeds relates to n=30 or n=60 samples
2. Clarify whether samples are independent runs or evaluation episodes
3. Ensure consistent terminology throughout manuscript

### 3. Reward Function Weight Justification

**Issue**: Six reward components with specific weights, but no justification provided.

**Current Specification** (Methods Section 3.2.5):
- Throughput: +1
- Waiting time: -0.1
- Queue length: -0.05
- Crash: -10000
- Balance: +0.5
- Transfer efficiency: +0.2/-0.1

**Concerns**:
- How were these weights chosen?
- Was sensitivity analysis performed?
- Are results robust to weight variations?
- Could different weights change conclusions?

**Recommendation**:
1. Provide justification for weight selection (e.g., domain expertise, tuning experiments)
2. Conduct sensitivity analysis showing results are robust to weight variations
3. Discuss trade-offs between competing objectives

### 4. Heuristic Baseline Fairness

**Issue**: DRL algorithms have 500K timesteps to optimize, but heuristics are static.

**Concern**:
- Is the comparison fair?
- Could heuristics be tuned or optimized?
- Are heuristics representative of best traditional approaches?

**Recommendation**:
1. Discuss fairness of comparison in Methods section
2. Consider tuning heuristic parameters if applicable
3. Justify why these specific heuristics were chosen as baselines

---

## Minor Concerns (Consider Addressing for Improvement)

### 1. Terminology Consistency
- Ensure MCRPS/D/K notation is consistently used throughout
- Define all abbreviations on first use
- Maintain consistent terminology for "layers" vs "levels"

### 2. Figure and Table Quality
- Generate all figures at publication quality (≥300 DPI)
- Ensure axes are labeled clearly with units
- Use colorblind-friendly color schemes
- Provide comprehensive captions that can stand alone

### 3. Citation Completeness
- Expand literature review to 30-40 references
- Cite recent Applied Soft Computing papers on DRL
- Include key queueing theory references
- Cite UAM industry reports and standards

### 4. Notation Clarity
- Ensure mathematical notation is consistent throughout
- Define all symbols in a notation table if needed
- Use standard queueing theory notation where applicable

---

## Final Recommendation

**Decision**: **MAJOR REVISION**

**Rationale**:

This manuscript has strong potential and addresses an interesting problem at the intersection of deep reinforcement learning and operations research. The experimental design is comprehensive, the findings are novel (particularly the capacity paradox), and the practical relevance to UAM systems is clear. The research fits well within Applied Soft Computing's scope.

However, several critical issues must be resolved before the manuscript can be accepted:

**Critical Issues Requiring Resolution**:
1. **Effect sizes (d=48.452, d=15.678)** are extraordinarily large and require verification or explanation
2. **Reward scale inconsistencies** (4K vs 720K vs 11K) need clarification
3. **Manuscript completeness** - currently in outline form, needs full text expansion
4. **Missing appendices** - supplementary experiments need integration

**Moderate Issues to Address**:
1. Strengthen capacity paradox theoretical explanation
2. Clarify sample size relationships (n=5 seeds vs n=30 samples)
3. Justify reward function weight selection
4. Discuss heuristic baseline fairness

**Estimated Work Required**:
- 2-3 weeks for full text expansion (per preparation summary)
- Resolution of statistical concerns (effect sizes, reward scales)
- Integration of supplementary experiments as appendices
- Generation of all figures and tables

**Acceptance Probability After Revision**: 85-90% (as estimated in preparation summary)

The research has strong empirical results and practical value. Once the critical issues are addressed and the manuscript is completed, it should be suitable for publication in Applied Soft Computing.

---

## Specific Action Items for Authors

### Immediate Priority (Before Resubmission)
1. ✅ Verify Cohen's d calculations - investigate why effect sizes are so large
2. ✅ Clarify reward scale differences across experiments
3. ✅ Complete full text expansion from outlines
4. ✅ Integrate Priority 1 and Priority 2 experiments as appendices
5. ✅ Generate all 6 figures at publication quality
6. ✅ Create all 4 tables with proper formatting

### High Priority (Strengthen Paper)
7. ⚠️ Strengthen capacity paradox explanation with additional analysis
8. ⚠️ Clarify sample size methodology (seeds vs samples)
9. ⚠️ Add reward function weight justification
10. ⚠️ Write Discussion section (2-3 pages)
11. ⚠️ Write Conclusion section (1 page)
12. ⚠️ Expand literature review (30-40 references)

### Medium Priority (Polish)
13. 📝 Ensure terminology consistency throughout
14. 📝 Add comprehensive figure captions
15. 📝 Create notation table if needed
16. 📝 Proofread for grammar and clarity

---

## Reviewer's Summary

**What Works Well**:
- Comprehensive comparison of 15 DRL algorithms
- Novel capacity paradox finding
- Clear practical relevance to UAM systems
- Rigorous experimental design with multiple seeds
- Good fit with Applied Soft Computing scope

**What Needs Work**:
- Resolve extraordinarily large effect sizes
- Clarify reward scale inconsistencies
- Complete manuscript from outline to full text
- Integrate supplementary experiments
- Strengthen theoretical explanations

**Overall Assessment**: Strong research with interesting findings, but requires substantial revision before acceptance. The core contributions are valuable, and with proper attention to the identified concerns, this should become a solid publication.

---

**Review Completed**: 2026-01-18

**Reviewer Recommendation**: Major Revision

**Estimated Time to Acceptance**: 3-6 months (including revision time)

