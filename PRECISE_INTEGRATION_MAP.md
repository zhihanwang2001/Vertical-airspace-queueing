# Precise Integration Map for Manuscript
**Date**: 2026-01-27 21:55
**Purpose**: Exact line numbers and insertion points for all ablation content

---

## Manuscript Structure Overview

**Total Lines**: 1,249 lines
**Key Sections**:
- Abstract: Lines 65-67
- Introduction: Line 90
- Methodology: Line 197
- Results: Line 780
- Discussion: Line 1076
- Conclusion: Line 1151

---

## Integration Point 1: Abstract (REPLACE)

**Location**: Lines 65-67
**Action**: REPLACE entire abstract content

**Current Content** (lines 65-67):
```latex
\begin{abstract}
[Current abstract text - approximately 2 lines]
\end{abstract}
```

**Replace With**: Content from `sections/revised_abstract.tex`

**How to do it**:
1. Open `manuscript.tex` in editor
2. Go to line 65
3. Select from `\begin{abstract}` to `\end{abstract}` (lines 65-67)
4. Delete selected content
5. Copy entire content from `sections/revised_abstract.tex`
6. Paste at line 65

**Expected Result**: Abstract now spans ~10 lines instead of 2-3 lines

---

## Integration Point 2: Ablation Study Section (INSERT)

**Location**: After line 1074 (before Discussion section at line 1076)
**Action**: INSERT new subsection

**Context**:
```
Line 1074: [End of last Results subsection]
Line 1075: [Blank line]
Line 1076: \section{Discussion}
```

**Insert At**: Line 1075 (between Results and Discussion)

**Content to Insert**:
```latex
\subsection{Ablation Study: Network Capacity and Architectural Components}
\label{subsec:ablation}

[Full content from sections/ablation_study.tex]
```

**How to do it**:
1. Open `manuscript.tex` in editor
2. Go to line 1075 (blank line before Discussion)
3. Position cursor at beginning of line 1075
4. Copy entire content from `sections/ablation_study.tex`
5. Paste at line 1075
6. Add blank line after the inserted content

**Expected Result**: New subsection appears as last subsection of Results, just before Discussion

---

## Integration Point 3: Discussion Additions (INSERT)

**Location**: After line 1148 (before Conclusion section at line 1151)
**Action**: INSERT new subsection

**Context**:
```
Line 1148: [End of last Discussion subsection]
Line 1149: [Blank line]
Line 1150: [Blank line]
Line 1151: \section{Conclusion}
```

**Insert At**: Line 1149 (between Discussion and Conclusion)

**Content to Insert**:
```latex
\subsection{The Performance-Stability Trade-off in Deep Reinforcement Learning}
\label{subsec:performance_stability}

[Full content from sections/ablation_discussion.tex]
```

**How to do it**:
1. Open `manuscript.tex` in editor
2. Go to line 1149 (blank line before Conclusion)
3. Position cursor at beginning of line 1149
4. Copy entire content from `sections/ablation_discussion.tex`
5. Paste at line 1149
6. Add blank line after the inserted content

**Expected Result**: New subsection appears as last subsection of Discussion, just before Conclusion

---

## Integration Point 4: Tables (INSERT)

**Location**: Within ablation study section (after line 1075 + ~20 lines)
**Action**: INSERT table reference

**The table is already defined in `tables/tab_ablation_results.tex`**

**Option A: Use \input command** (Recommended)
```latex
% In the ablation study section, after "Table~\ref{tab:ablation} presents..."
\input{tables/tab_ablation_results}
```

**Option B: Copy-paste directly**
1. Open `tables/tab_ablation_results.tex`
2. Copy entire content
3. Paste in ablation study section where table is referenced

**Note**: The table reference is already in `sections/ablation_study.tex` at line 25:
```latex
Table~\ref{tab:ablation} presents the complete ablation study results.
```

So you need to insert the actual table definition after this reference.

**How to do it**:
1. After inserting `sections/ablation_study.tex` (Integration Point 2)
2. Find the line that says "Table~\ref{tab:ablation} presents..."
3. A few lines after this, insert the table definition
4. Either use `\input{tables/tab_ablation_results}` or copy-paste the table content

---

## Integration Point 5: Figures (INSERT)

**Location**: Within ablation study section, after table
**Action**: INSERT figure definitions

**Figures to Add** (3 figures):

### Figure 1: Performance Comparison
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/ablation_performance_comparison.pdf}
\caption{Performance comparison between HCA2C-Full and A2C-Enhanced across three random seeds. While A2C-Enhanced achieves higher mean and peak performance, it shows extreme variance. HCA2C-Full provides consistent performance across all seeds.}
\label{fig:ablation_performance}
\end{figure}
```

### Figure 2: Stability Comparison
```latex
\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{figures/ablation_stability_comparison.pdf}
\caption{Stability and reliability comparison. (Left) A2C-Enhanced shows 965,000× higher variance than HCA2C-Full. (Right) HCA2C-Full achieves 100\% success rate while A2C-Enhanced has 33\% failure rate to low-performance modes.}
\label{fig:ablation_stability}
\end{figure}
```

### Figure 3: Bimodal Distribution
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/ablation_bimodal_distribution.pdf}
\caption{A2C-Enhanced exhibits bimodal distribution across random seeds. Seed 42 converges to low-performance mode (217K), while seeds 43-44 converge to high-performance mode (507K). HCA2C-Full (blue line) provides stable performance independent of random seed.}
\label{fig:ablation_bimodal}
\end{figure}
```

**Where to Insert**:
- Figure 1: After the paragraph mentioning "Figure~\ref{fig:ablation_performance}"
- Figure 2: After the paragraph mentioning "Figure~\ref{fig:ablation_stability}"
- Figure 3: After the paragraph mentioning "Figure~\ref{fig:ablation_bimodal}"

**Note**: The figure references are already in `sections/ablation_study.tex` at line 48 and 74:
```latex
Line 48: Figure~\ref{fig:ablation_bimodal} illustrates this bimodal distribution...
Line 74: Figure~\ref{fig:ablation_performance} shows the performance comparison boxplot, and Figure~\ref{fig:ablation_stability} illustrates...
```

---

## Integration Point 6: Introduction Update (OPTIONAL)

**Location**: After line 152 (in Introduction, after background)
**Action**: INSERT new paragraph

**Content to Insert**:
```latex
\textbf{The Performance-Stability Challenge.} A critical challenge in deep reinforcement learning is the \textit{performance-stability trade-off}. While large networks can achieve impressive peak performance, they often suffer from extreme variance and initialization sensitivity. In safety-critical applications like Urban Air Mobility, achieving high performance in 67\% of training runs is insufficient---we need 100\% reliability. This paper addresses this challenge by proposing a hierarchical architecture that provides consistent, reliable high performance across all random seeds, validated through comprehensive ablation studies comparing capacity-matched baselines.
```

**Where to Insert**: After line 152 (after background, before contributions)

---

## Integration Point 7: Contributions Update (OPTIONAL)

**Location**: Around line 154-190 (Main Contributions subsection)
**Action**: ADD new contribution bullet

**Content to Add**:
```latex
\item We conduct comprehensive ablation studies revealing that while capacity-matched baselines can achieve 121\% higher peak performance, they exhibit 965,000× higher variance and only 67\% reliability, demonstrating the value of architectural regularization for stable, predictable performance in safety-critical applications.
```

**Where to Insert**: As the last bullet point in the contributions list

---

## Integration Point 8: Conclusion Update (REPLACE)

**Location**: Around line 1151-1168 (Conclusion section)
**Action**: UPDATE conclusion statement

**Find This Text** (approximate):
```latex
We proposed HCA2C, a hierarchical capacity-aware actor-critic algorithm...
```

**Replace With**:
```latex
We proposed HCA2C, a hierarchical capacity-aware actor-critic algorithm that achieves stable, reliable high performance for vertical queueing control in Urban Air Mobility. Through comprehensive ablation studies, we demonstrated that while capacity-matched baselines can achieve 121\% higher peak performance, they exhibit 965,000× higher variance and only 67\% reliability. HCA2C's hierarchical decomposition provides essential architectural regularization, ensuring 100\% reliable performance across all random seeds. This stability is critical for practical deployment in safety-critical applications where single-run success is required.
```

---

## Step-by-Step Integration Sequence

### Phase 1: Preparation (5 minutes)
1. [ ] Backup `manuscript.tex`: `cp manuscript.tex manuscript_backup.tex`
2. [ ] Open `manuscript.tex` in LaTeX editor
3. [ ] Open `INTEGRATION_GUIDE.md` in separate window
4. [ ] Open all 4 LaTeX section files in separate tabs

### Phase 2: Core Integrations (30-45 minutes)
1. [ ] **Abstract** (Integration Point 1)
   - Go to line 65
   - Replace lines 65-67 with content from `sections/revised_abstract.tex`
   - Verify: Abstract now ~10 lines

2. [ ] **Ablation Study Section** (Integration Point 2)
   - Go to line 1075
   - Insert content from `sections/ablation_study.tex`
   - Verify: New subsection before Discussion

3. [ ] **Tables** (Integration Point 4)
   - Find "Table~\ref{tab:ablation} presents..." in newly inserted section
   - Insert `\input{tables/tab_ablation_results}` a few lines after
   - Verify: Table reference exists

4. [ ] **Figures** (Integration Point 5)
   - Find figure references in ablation study section
   - Insert 3 figure definitions after their references
   - Verify: All 3 figures defined

5. [ ] **Discussion Additions** (Integration Point 3)
   - Go to line 1149 (now shifted due to previous insertions)
   - Insert content from `sections/ablation_discussion.tex`
   - Verify: New subsection before Conclusion

### Phase 3: Optional Enhancements (15-30 minutes)
6. [ ] **Introduction Update** (Integration Point 6)
   - Go to line 152
   - Insert performance-stability challenge paragraph
   - Verify: Smooth transition

7. [ ] **Contributions Update** (Integration Point 7)
   - Find contributions list (~line 154-190)
   - Add ablation study contribution bullet
   - Verify: Consistent formatting

8. [ ] **Conclusion Update** (Integration Point 8)
   - Find conclusion statement (~line 1151-1168)
   - Update with ablation findings
   - Verify: Consistent message

### Phase 4: Compilation (15-20 minutes)
9. [ ] **First Compilation**
   ```bash
   cd Manuscript/Applied_Soft_Computing/LaTeX
   pdflatex manuscript.tex
   ```
   - Check for errors
   - Note any undefined references

10. [ ] **Bibliography**
    ```bash
    bibtex manuscript
    ```
    - Check for bibliography errors

11. [ ] **Second Compilation**
    ```bash
    pdflatex manuscript.tex
    ```
    - Resolve cross-references

12. [ ] **Third Compilation**
    ```bash
    pdflatex manuscript.tex
    ```
    - Final pass for all references

### Phase 5: Verification (20-30 minutes)
13. [ ] **Check PDF Output**
    - Open `manuscript.pdf`
    - Verify all sections appear
    - Check figure placement
    - Check table formatting
    - Verify page count increase (~3-4 pages)

14. [ ] **Verify Cross-References**
    - Search for "??" in PDF (indicates unresolved references)
    - Verify all Table~\ref{} work
    - Verify all Figure~\ref{} work
    - Verify all Section~\ref{} work

15. [ ] **Verify Numbers**
    - Check all statistical numbers match data
    - Verify consistency across sections
    - Check calculations (e.g., +121%, 965,000×)

### Phase 6: Proofreading (30-60 minutes)
16. [ ] **Read Through**
    - Read abstract
    - Read ablation study section
    - Read discussion additions
    - Check transitions between sections

17. [ ] **Final Checks**
    - Consistent terminology (HCA2C-Full, A2C-Enhanced)
    - Proper capitalization
    - Consistent notation
    - No typos or grammatical errors

---

## Expected Line Number Changes

After all integrations, the manuscript will grow from 1,249 lines to approximately:

- Abstract: +8 lines (from 3 to 11 lines)
- Ablation Study Section: +74 lines
- Tables: +46 lines
- Figures: +30 lines (3 figures × ~10 lines each)
- Discussion Additions: +88 lines
- Introduction/Conclusion: +10 lines (optional)

**Total Addition**: ~256 lines
**New Total**: ~1,505 lines

**Page Count**: +3-4 pages (from current page count)

---

## Common Issues and Solutions

### Issue 1: Undefined References
**Symptom**: "??" appears instead of numbers
**Solution**: Run pdflatex 2-3 times to resolve all references

### Issue 2: Figure Placement
**Symptom**: Figures appear in wrong location
**Solution**: Use `[h]` or `[H]` placement, or move to end of section

### Issue 3: Table Too Wide
**Symptom**: Table exceeds page width
**Solution**: Add `\small` before table, or use `\resizebox`

### Issue 4: Compilation Errors
**Symptom**: pdflatex fails with errors
**Solution**: Check error message, verify all `\ref{}` labels exist

---

## Quick Reference: File Paths

**LaTeX Sections**:
```
sections/revised_abstract.tex
sections/ablation_study.tex
sections/ablation_discussion.tex
tables/tab_ablation_results.tex
```

**Figures**:
```
figures/ablation_performance_comparison.pdf
figures/ablation_stability_comparison.pdf
figures/ablation_bimodal_distribution.pdf
```

**Manuscript**:
```
manuscript.tex (1,249 lines → ~1,505 lines after integration)
```

---

## Success Criteria Checklist

After integration, verify:

- [ ] Abstract mentions ablation studies and key findings
- [ ] Ablation study section appears in Results (before Discussion)
- [ ] Both ablation tables are included and referenced
- [ ] All three figures are included and referenced
- [ ] Discussion includes performance-stability trade-off section
- [ ] All cross-references work (no ??)
- [ ] Manuscript compiles without errors
- [ ] Page count increased by ~3-4 pages
- [ ] All numbers are consistent
- [ ] Smooth transitions between sections
- [ ] Professional formatting maintained

---

## Final Notes

**Estimated Time**: 3-4 hours total
- Preparation: 5 minutes
- Core integrations: 30-45 minutes
- Optional enhancements: 15-30 minutes
- Compilation: 15-20 minutes
- Verification: 20-30 minutes
- Proofreading: 30-60 minutes

**Backup Strategy**: Always keep `manuscript_backup.tex` until final verification complete

**Support**: Refer to `INTEGRATION_GUIDE.md` for detailed instructions and troubleshooting

**Ready to begin!** 🚀
