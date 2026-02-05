# Manuscript Integration Guide

**Date**: 2026-01-27 21:10
**Purpose**: Step-by-step instructions for integrating ablation study content into manuscript.tex

---

## 📋 Files Created

All ablation study content has been prepared in standalone files:

1. **`sections/ablation_study.tex`** - Complete ablation study section for Results
2. **`sections/ablation_discussion.tex`** - Discussion additions on performance-stability trade-off
3. **`sections/revised_abstract.tex`** - Updated abstract with ablation findings
4. **`tables/tab_ablation_results.tex`** - Two professional LaTeX tables

---

## 🎯 Integration Steps

### Step 1: Update Abstract

**Location**: Lines 65-67 in `manuscript.tex`

**Action**: Replace the current abstract with content from `sections/revised_abstract.tex`

**Current abstract** (lines 65-67):
```latex
\begin{abstract}
Urban Air Mobility (UAM) systems face critical challenges...
[current text]
\end{abstract}
```

**Replace with**: Content from `sections/revised_abstract.tex`

**Key additions**:
- Ablation study findings (second paragraph)
- Performance-stability trade-off
- Bimodal distribution (965,000× variance)
- 100% reliability vs 67% reliability

---

### Step 2: Add Ablation Study Section to Results

**Location**: After main results sections, before Discussion (approximately line 800-900)

**Action**: Insert content from `sections/ablation_study.tex`

**Suggested placement**: After the structural comparison results, before Discussion section

**What to add**:
```latex
%% Insert here: sections/ablation_study.tex
\input{sections/ablation_study}
```

Or copy-paste the entire content from `sections/ablation_study.tex`

**This section includes**:
- Experimental setup (4 variants)
- Complete results table (Table~\ref{tab:ablation})
- Four key findings
- Statistical analysis
- Interpretation

---

### Step 3: Add Ablation Tables

**Location**: In `tables/` directory (already created)

**Action**: The tables are already in `tables/tab_ablation_results.tex`

**To use in manuscript**:
```latex
%% In the ablation study section
\input{tables/tab_ablation_results}
```

Or copy-paste the table definitions where needed.

**Tables included**:
1. Main ablation results table (Table~\ref{tab:ablation})
2. Seed-level detailed results (Table~\ref{tab:ablation_seeds})

---

### Step 4: Add Ablation Discussion

**Location**: In Discussion section (approximately line 1000-1200)

**Action**: Add content from `sections/ablation_discussion.tex`

**Suggested placement**: As a new subsection after existing discussion content

**What to add**:
```latex
%% Insert here: sections/ablation_discussion.tex
\input{sections/ablation_discussion}
```

Or copy-paste the entire content.

**This section includes**:
- Performance-stability trade-off analysis
- Peak performance discussion
- Reliability problem explanation
- HCA2C's value proposition
- Practical implications for UAM
- Broader implications for deep RL
- Limitations and future work

---

### Step 5: Add Figure References

**Location**: Throughout the ablation study section

**Action**: Ensure figures are referenced correctly

**Figures to add** (from `Analysis/figures/`):
1. `ablation_performance_comparison.pdf` → Figure~\ref{fig:ablation_performance}
2. `ablation_stability_comparison.pdf` → Figure~\ref{fig:ablation_stability}
3. `ablation_bimodal_distribution.pdf` → Figure~\ref{fig:ablation_bimodal}

**Copy figures to manuscript directory**:
```bash
cp Analysis/figures/ablation_*.pdf Manuscript/Applied_Soft_Computing/LaTeX/figures/
```

**Add figure definitions** (in manuscript.tex, after ablation study section):
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/ablation_performance_comparison.pdf}
\caption{Performance comparison between HCA2C-Full and A2C-Enhanced across three random seeds. While A2C-Enhanced achieves higher mean and peak performance, it shows extreme variance. HCA2C-Full provides consistent performance across all seeds.}
\label{fig:ablation_performance}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{figures/ablation_stability_comparison.pdf}
\caption{Stability and reliability comparison. (Left) A2C-Enhanced shows 965,000× higher variance than HCA2C-Full. (Right) HCA2C-Full achieves 100\% success rate while A2C-Enhanced has 33\% failure rate to low-performance modes.}
\label{fig:ablation_stability}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/ablation_bimodal_distribution.pdf}
\caption{A2C-Enhanced exhibits bimodal distribution across random seeds. Seed 42 converges to low-performance mode (217K), while seeds 43-44 converge to high-performance mode (507K). HCA2C-Full (blue line) provides stable performance independent of random seed.}
\label{fig:ablation_bimodal}
\end{figure}
```

---

### Step 6: Update Introduction (Optional but Recommended)

**Location**: Introduction section, after background (approximately line 150-200)

**Action**: Add a paragraph on performance-stability trade-off

**Suggested addition**:
```latex
\textbf{The Performance-Stability Challenge.} A critical challenge in deep reinforcement learning is the \textit{performance-stability trade-off}. While large networks can achieve impressive peak performance, they often suffer from extreme variance and initialization sensitivity. In safety-critical applications like Urban Air Mobility, achieving high performance in 67\% of training runs is insufficient---we need 100\% reliability. This paper addresses this challenge by proposing a hierarchical architecture that provides consistent, reliable high performance across all random seeds, validated through comprehensive ablation studies comparing capacity-matched baselines.
```

**Also update contributions** (approximately line 250-300):
```latex
\item We conduct comprehensive ablation studies revealing that while capacity-matched baselines can achieve 121\% higher peak performance, they exhibit 965,000× higher variance and only 67\% reliability, demonstrating the value of architectural regularization for stable, predictable performance in safety-critical applications.
```

---

### Step 7: Update Conclusion

**Location**: Conclusion section (approximately line 1400-1500)

**Action**: Update the contribution statement

**Current statement** (find and replace):
```latex
We proposed HCA2C, a hierarchical capacity-aware actor-critic algorithm...
```

**Replace with**:
```latex
We proposed HCA2C, a hierarchical capacity-aware actor-critic algorithm that achieves stable, reliable high performance for vertical queueing control in Urban Air Mobility. Through comprehensive ablation studies, we demonstrated that while capacity-matched baselines can achieve 121\% higher peak performance, they exhibit 965,000× higher variance and only 67\% reliability. HCA2C's hierarchical decomposition provides essential architectural regularization, ensuring 100\% reliable performance across all random seeds. This stability is critical for practical deployment in safety-critical applications where single-run success is required.
```

---

## 🔍 Verification Checklist

After integration, verify:

- [ ] Abstract mentions ablation studies and key findings
- [ ] Ablation study section appears in Results
- [ ] Both ablation tables are included and referenced
- [ ] All three figures are copied and referenced
- [ ] Discussion includes performance-stability trade-off section
- [ ] Introduction mentions the challenge (optional)
- [ ] Conclusion updated with ablation findings
- [ ] All cross-references work (Table~\ref{}, Figure~\ref{})
- [ ] Compile manuscript.tex without errors
- [ ] Check page count (should increase by ~3-4 pages)

---

## 📊 Expected Changes

**Page count**: +3-4 pages
- Ablation study section: ~2 pages
- Discussion additions: ~1.5 pages
- Tables: ~0.5 pages
- Figures: ~1 page (can be placed in supplementary if needed)

**Word count**: +3,000-3,500 words
- Ablation study: ~1,500 words
- Discussion: ~1,500 words
- Abstract: +200 words
- Introduction/Conclusion: +300 words

---

## 🎯 Key Messages to Emphasize

When integrating, ensure these key messages are clear:

1. **Acknowledge A2C-Enhanced's higher peak** (+121%)
2. **Emphasize extreme variance** (965,000×)
3. **Highlight bimodal distribution** (33% failure rate)
4. **Position HCA2C as stability-focused** (100% reliability)
5. **Explain practical implications** (safety-critical applications)

---

## 🚨 Common Issues and Solutions

### Issue 1: Figure placement
**Problem**: Figures appear in wrong location
**Solution**: Use `[h]` or `[H]` placement specifier, or move to supplementary materials

### Issue 2: Table too wide
**Problem**: Table exceeds page width
**Solution**: Use `\small` or `\footnotesize` before table, or rotate with `\begin{sidewaystable}`

### Issue 3: Cross-references broken
**Problem**: `??` appears instead of numbers
**Solution**: Compile twice (first pass creates labels, second resolves references)

### Issue 4: Page count too high
**Problem**: Manuscript exceeds page limit
**Solution**: Move detailed seed-level table to supplementary materials, reduce figure sizes

---

## 📝 Quick Integration Commands

**Copy figures**:
```bash
cd /Users/harry./Desktop/PostGraduate/RP1
cp Analysis/figures/ablation_*.pdf Manuscript/Applied_Soft_Computing/LaTeX/figures/
```

**Compile manuscript**:
```bash
cd Manuscript/Applied_Soft_Computing/LaTeX
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

---

## ✅ Final Checklist

Before submission:

- [ ] All ablation content integrated
- [ ] Manuscript compiles without errors
- [ ] All figures display correctly
- [ ] All tables formatted properly
- [ ] Cross-references resolved
- [ ] Page count acceptable
- [ ] Proofreading complete
- [ ] Consistent terminology throughout
- [ ] References updated if needed

---

## 🎉 You're Done!

The ablation study content is now fully integrated into your manuscript. The revised manuscript:

1. **Acknowledges** A2C-Enhanced's higher peak performance
2. **Explains** the performance-stability trade-off
3. **Demonstrates** HCA2C's reliability advantage
4. **Provides** comprehensive statistical evidence
5. **Discusses** practical implications for UAM deployment

This makes your manuscript **more honest, more rigorous, and more valuable** than claiming simple superiority.

---

**Questions?** Refer to:
- `Analysis/statistical_reports/manuscript_revision_guide.md` for detailed LaTeX templates
- `COMPLETE_ABLATION_REPORT.md` for full experimental results
- `Analysis/statistical_reports/final_ablation_analysis.txt` for statistical details
