# Batch 4 Deep Re-Review Plan
## Review Date: 2026-01-06

## Background
Completed reviews:
- ✅ Data Verification (BATCH4_VERIFICATION.md): 10/10 data points verified
- ✅ Logic Review (BATCH4_LOGIC_REVIEW.md): 2 critical issues fixed
- ✅ Quality Assessment (BATCH4_FINAL_CHECK.md): Total score 9.6/10

User requested "re-review", requiring in-depth inspection from different dimensions.

---

## Review Objectives
This review will focus on the following **insufficiently covered** dimensions:

### 1. **Cross-Batch Consistency Check**
**Objective**: Ensure Batch 1-4 are completely consistent in terminology, symbols, and data references

**Check Items**:
- [ ] Symbol consistency: Are symbols defined in Batch 3 used consistently in Batch 4
  - Example: $U_\ell$ vs $U_i$, $\rho_\ell$ vs $\rho_i$
  - Usage of $G_t$ vs $G^t_\ell$ across different batches
- [ ] Terminology consistency: Same concepts expressed in different batches
  - "occupancy rate" vs "utilization rate"
  - "pressure" vs "workload"
  - "emergency transfer" vs "downward transfer"
- [ ] Data cross-reference: Is data referenced in Batch 4 consistent with Batch 3
  - Capacity [2,3,4,6,8]
  - Service rates [0.4, 0.6, 0.8, 1.0, 1.2]
  - Arrival weights [0.10, 0.15, 0.20, 0.25, 0.30]
- [ ] Section reference accuracy: Does §4.1 referencing §4.3 form a logical loop

**Tools**:
- Read paper_batch1, batch2, batch3, batch4 in parallel
- Use Grep to search for key term definitions across batches

---

### 2. **Narrative Coherence Review**
**Objective**: Check logical transitions and narrative fluency between Batch 4 subsections

**Check Items**:
- [ ] §4.1 → §4.2 transition: How pressure trigger mechanism leads to Pareto optimization
- [ ] §4.2 → §4.3 transition: How Pareto objectives transform into DRL reward function
- [ ] Does each subsection opening have sufficient motivating context
- [ ] Does technical detail presentation order follow reader understanding logic

**Method**:
- Read paragraph by paragraph, mark abrupt logical jumps
- Check if each new concept introduction has sufficient groundwork

---

### 3. **Mathematical Rigor Review**
**Objective**: Beyond formula correctness, check mathematical expression completeness and rigor

**Check Items**:
- [ ] Are all variables defined at first appearance
  - $f^*$ (ideal point) undefined at line 69
  - $y$ (dominated solutions) undefined in HV formula
  - Role of $m$ (arrival_multiplier value) in projection formula
- [ ] Are formula domains and ranges clear
  - Gini coefficient range [0,1]
  - Theoretical range of Pressure $P^t_\ell$
  - Is HV normalization correct
- [ ] Strictness of inequalities and constraints
  - $\rho_\ell < 1$ vs $\rho_\ell \leq 1$
  - Values of $G_{target}$ and $\epsilon$ in $G_t \leq G_{target} + \epsilon$
- [ ] Is bold/non-bold symbol usage consistent
  - $\mathbf{f}(\mathbf{x})$ vs $f(x)$

**Method**:
- Check formula by formula
- Create symbol reference table

---

### 4. **Implementation Feasibility Verification**
**Objective**: Ensure described algorithms are feasible and unambiguous at code implementation level

**Check Items**:
- [ ] Can state space dimension calculation be derived from description
  - 29 dimensions = 5(queues) + 5(occupancy rates) + 5(service rates) + 5(arrival weights) + 5(transfer flags) + 1(timestep) + 3(global metrics)
  - Verify source of each item
- [ ] Is continuous/discrete division of action space clear
  - 6D continuous (5 service_intensities + 1 arrival_multiplier)
  - 5D discrete (emergency_transfers)
- [ ] Are projection algorithm steps complete
  - Can pseudocode at Algorithm lines 112-117 be directly implemented
  - Boundary condition handling
- [ ] Is reward function calculation unambiguous
  - Are input variables for each reward term in state space
  - Can transfer_benefit condition `upper_pressure > lower_util` be computed

**Method**:
- Simulate manual calculation of one timestep
- Check if all needed variables are in state

---

### 5. **Citations and Evidence Support**
**Objective**: Ensure all claims have appropriate support

**Check Items**:
- [ ] Are empirical statements supported by data
  - Where does "<0.2% transfer frequency" come from
  - Does "37% peak reduction" come from experiments (but experiment section is in Batch 6)
- [ ] Are design choices reasonably explained
  - Why β₃=0 simplifies pressure formula
  - Why Gini threshold chooses $G_{target}$
  - Design rationale for reward weights [10:5:3:2:2:-20:-15]
- [ ] Reasonableness of future section references
  - §4.1 references §4.3 (within same chapter, reasonable)
  - Does reference to §6.5 exist (fixed in Batch 3)
  - Is reference to Appendix C marked as "future supplement"

**Method**:
- Mark all quantitative claims
- Check source of each claim

---

### 6. **Academic English Quality**
**Objective**: Ensure compliance with high-level academic writing standards

**Check Items**:
- [ ] Reasonable use of passive vs active voice
- [ ] Technical terminology consistency (no switching synonyms in same sentence)
- [ ] Sentence length and complexity (avoid overly long clauses)
- [ ] Paragraph structure: Topic sentence → Supporting details → Transition
- [ ] Avoid colloquial expressions
- [ ] Tense consistency (present tense for system description, past tense for experiments)

**Method**:
- Read paragraph by paragraph
- Mark verbose or ambiguous sentences

---

### 7. **Quick Assessment of 4 Remaining Suggestions**
**Objective**: Assess whether 4 suggestions in BATCH4_FINAL_CHECK.md are worth immediate fixing

**Suggestion List**:
1. Clarify $G^t_\ell$ local fairness definition
2. Improve energy efficiency model expression (causality of "high μ corresponds to low energy consumption")
3. Refine transfer_benefit calculation condition variables
4. Explain why only arrival_multiplier is constrained, not service_intensities

**Assessment Criteria**:
- Fix workload (1 line vs 1 paragraph)
- Impact on reader understanding
- Potential to introduce new issues

---

## Review Process

### Phase 1: Parallel Data Collection (5 minutes)
- Read Batch 1-4 .tex files simultaneously
- Use Grep to search key term definitions across batches
- Extract all formulas and symbols

### Phase 2: Cross-Batch Consistency Check (10 minutes)
- Create symbol reference table
- Mark inconsistencies
- Check data references

### Phase 3: Deep Logic Review (15 minutes)
- Check narrative flow paragraph by paragraph
- Verify mathematical rigor
- Check implementation feasibility

### Phase 4: Quality Enhancement (10 minutes)
- Assess 4 remaining suggestions
- Check academic English standards
- Verify citations and evidence

### Phase 5: Generate Report (5 minutes)
- Integrate all findings
- Categorize: Critical issues vs improvement suggestions
- Give clear "proceed" or "needs fixing" conclusion

---

## Expected Output

1. **BATCH4_DEEP_CONSISTENCY_CHECK.md**
   - Cross-batch consistency issue list
   - Symbol reference table

2. **BATCH4_NARRATIVE_REVIEW.md**
   - Narrative flow issues
   - English quality improvement suggestions

3. **BATCH4_MATH_RIGOR_CHECK.md**
   - Undefined variable list
   - Mathematical expression improvements

4. **BATCH4_FINAL_DECISION.md**
   - Integrate all review results
   - Clear "proceed" or "blocking" decision
   - If proceeding, list optional improvement checklist

---

## Success Criteria

Markers of successful review:
- ✅ Discover all cross-batch inconsistencies (if any)
- ✅ Identify all undefined mathematical variables
- ✅ Provide clear proceed decision
- ✅ If blocking issues found, fix immediately and verify
- ✅ If no blocking issues, provide priority-sorted improvement checklist

---

## Time Estimate
- **Total Duration**: Approximately 45 minutes
- **Critical Path**: Phase 2 (Cross-batch consistency) → Phase 3 (Deep logic) → Phase 5 (Decision)

---

## Notes
- This review is the **final checkpoint** before Batch 4 proceeds
- User emphasizes "all content must be meticulous", requires absolute accuracy
- If any blocking issues found, must fix immediately and recompile for verification
