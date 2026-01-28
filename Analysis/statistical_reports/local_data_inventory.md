# Local Data Inventory
**Date**: 2026-01-17
**Total Size**: ~26MB

---

## 1. Experimental Data (220K)

### Priority 1: Load Sensitivity Analysis (220 runs)
- `capacity_scan_results_uniform_3_4.csv` - Loads 3×,4× (40 runs)
- `capacity_scan_results_uniform_6_7.csv` - Loads 6×,7× + heuristics (120 runs)
- `capacity_scan_results_uniform_8_9_10.csv` - Loads 8×,9×,10× (60 runs)
- `capacity_scan_results_complete.csv` - Merged all loads (220 runs)

### Priority 2: Structural Comparison (120 runs)
- `capacity_scan_results_inverted_3_7_10.csv` - Inverted pyramid (60 runs)
- `capacity_scan_results_reverse_3_7_10.csv` - Reverse pyramid (60 runs)

### Server Backup
- `capacity_scan_results_server_backup.csv` - Backup data (60 runs)
- `structural_5x_load/` - Trained models + results (21MB, 120 runs)

---

## 2. Analysis Results (92K)

### Statistical Reports
- `complete_load_sensitivity.csv` - Priority 1 statistical analysis
- `structural_comparison_results.csv` - Priority 2 statistical analysis
- `complete_analysis_summary.md` - Comprehensive findings report
- `plan_completion_status.md` - Execution plan status

### Additional Reports (32 CSV files total in Data/summary/)
- Capacity scan summaries (uniform, inverted, reverse)
- Bootstrap CI results
- Structural group tables
- Stability proxy analyses

---

## 3. Visualizations (4.6MB)

### New Analysis Figures (6 files)
- `capacity_paradox_comprehensive.png` - Main capacity paradox visualization
- `capacity_paradox_load_sensitivity.png` - Load sensitivity trends
- `capacity_uniform_k10k30_crash/reward.png/pdf/svg` - K=10 vs K=30 comparison
- `capacity_paradox_crash/reward_heuristics.png` - Heuristic baselines

### Manuscript Figures (15 files)
- English versions: figure1-5_en.png, fig1-5_en.png
- Chinese versions: figure1-5.png

### Other Analysis Charts (9 files)
- Structural comparison: reward bars, box plots, stability scatter
- Stability analysis: drift vs crash, load vs reward

---

## 4. Key Findings

### Priority 1: Capacity Paradox
- **Transition point**: Between 4× and 6× load
- **K=10 stability**: 0% crash rate across all loads
- **K=30 fragility**: 84.6%-100% crash rate at loads ≥6×
- **Statistical significance**: All comparisons p<0.05

### Priority 2: Structural Comparison
- **Inverted pyramid advantage**: 9.66%-19.65% at K=10
- **Load 3×**: +9.66% (p<0.001, d=33.75)
- **Load 7×**: +15.56% (p<10⁻⁴¹, d=302.55)
- **Load 10×**: +19.65% (p<10⁻⁴³, d=412.61)

---

## 5. Data Quality

**Completeness**: 100% (340 runs total)
- Priority 1: 220/220 runs ✅
- Priority 2: 120/120 runs ✅

**Integrity**: All quality checks passed ✅
**Backup**: Server data backed up ✅
**Reproducibility**: All scripts and configs available ✅

---

## 6. Next Steps

- [ ] Update manuscript figures (Figures 3-5)
- [ ] Update manuscript tables (Tables 2-4)
- [ ] Supplement Results section
- [ ] Update Discussion citations
