# Final Submission Checklist - Applied Soft Computing

**Date**: 2026-02-01
**Status**: ✅ READY FOR SUBMISSION

---

## ✅ Completed Tasks

### 1. Unicode Character Fixes
- [x] Fixed all × (multiplication) symbols → `$\times$` (150+ instances)
- [x] Fixed all ± (plus-minus) symbols → `$\pm$` (50+ instances)
- [x] Fixed all → (arrow) symbols → `$\rightarrow$` (3 instances)
- [x] Fixed all ≥, ≤ symbols → `$\geq$`, `$\leq$` (9 instances)
- [x] Verified 0 Unicode characters remain in source files

### 2. Figure Management
- [x] Deleted Figure 6 (as requested)
- [x] Automatic renumbering completed (7→6, 8→7, 9→8)
- [x] All figure references updated
- [x] Total figures: 8 (all at 300 DPI)

### 3. Compilation
- [x] pdflatex compilation successful (0 errors)
- [x] bibtex compilation successful
- [x] All cross-references resolved (no ??)
- [x] PDF generated: 46 pages, 1.2 MB

### 4. Submission Package
- [x] manuscript.pdf (1.2 MB, 46 pages)
- [x] manuscript_latex_source.zip (3.3 MB)
- [x] cover_letter.pdf (79 KB)
- [x] graphical_abstract.png (84 KB, 812×590 pixels)
- [x] figures.zip (276 KB, 8 figures)

### 5. Content Verification
- [x] Abstract ≤250 words
- [x] Keywords: 7 keywords
- [x] Highlights: 5 highlights (≤85 characters each)
- [x] All formulas verified (12, 18, 20, 21, 23, 27)
- [x] All mathematical symbols in proper LaTeX format

---

## 📋 Pre-Submission Verification

### Document Quality
- [x] No spelling errors
- [x] No grammar errors
- [x] Consistent terminology
- [x] All figures referenced in text
- [x] All tables referenced in text
- [x] All equations numbered correctly

### Format Compliance
- [x] Page count: 46 pages (within 20-50 page limit)
- [x] Font: 11pt (correct)
- [x] Margins: 1 inch (correct)
- [x] Line spacing: 1.0 (correct)
- [x] Document class: elsarticle (correct)

### Required Sections
- [x] Title
- [x] Abstract
- [x] Keywords
- [x] Highlights
- [x] Introduction
- [x] Methodology
- [x] Results
- [x] Discussion
- [x] Conclusion
- [x] References
- [x] Author information
- [x] CRediT author statement
- [x] Data availability statement
- [x] Conflict of interest statement
- [x] Funding statement

### Figures and Tables
- [x] All figures at 300 DPI
- [x] All figures in PDF or PNG format
- [x] All figure captions complete
- [x] All table captions complete
- [x] No missing figures or tables

### References
- [x] All citations in text have references
- [x] All references are cited in text
- [x] Reference format consistent
- [x] DOIs included where available

---

## 📦 Submission Package Contents

### Main Files
```
submission_ready/
├── manuscript.pdf (1.2 MB) ✅
├── cover_letter.pdf (79 KB) ✅
├── graphical_abstract.png (84 KB) ✅
├── manuscript_latex_source.zip (3.3 MB) ✅
└── figures.zip (276 KB) ✅
```

### LaTeX Source (in manuscript_latex_source.zip)
```
manuscript.tex ✅
elsarticle.cls ✅
sections/
  ├── ablation_study_simple.tex ✅
  ├── hca2c_ablation.tex ✅
  ├── hca2c_ablation_discussion.tex ✅
  └── [5 more section files] ✅
tables/
  ├── tab_hca2c_ablation.tex ✅
  ├── tab_ablation_simple.tex ✅
  └── [8 more table files] ✅
figures/
  ├── fig_system_architecture.pdf ✅
  ├── fig_hca2c_ablation_comprehensive.png ✅
  └── [6 more figure files] ✅
```

---

## 🎯 Journal Requirements Compliance

### Applied Soft Computing Requirements

| Requirement | Status | Details |
|-------------|--------|---------|
| Page limit | ✅ | 46 pages (20-50 allowed) |
| Abstract length | ✅ | ~250 words (≤250 allowed) |
| Keywords | ✅ | 7 keywords (1-7 allowed) |
| Highlights | ✅ | 5 highlights (3-5 required) |
| Figure resolution | ✅ | 300 DPI (≥300 required) |
| File format | ✅ | PDF + LaTeX source |
| Author info | ✅ | Complete |
| CRediT statement | ✅ | Included |
| Data availability | ✅ | Included |
| Conflicts | ✅ | Declared |
| Funding | ✅ | Declared |

---

## 🔍 Final Quality Checks

### Mathematical Content
- [x] All equations numbered sequentially
- [x] All mathematical symbols use LaTeX commands
- [x] No Unicode characters in source files
- [x] All Greek letters properly formatted
- [x] All subscripts and superscripts correct

### Statistical Content
- [x] All p-values reported correctly
- [x] All effect sizes (Cohen's d) reported
- [x] All means ± standard deviations formatted correctly
- [x] All statistical tests properly described

### Figures
- [x] Figure 1: System Architecture ✅
- [x] Figure 2: Algorithm Performance ✅
- [x] Figure 3: Structural Comparison ✅
- [x] Figure 4: Capacity Paradox ✅
- [x] Figure 5: State Space Analysis ✅
- [x] Figure 6: HCA2C Ablation (formerly Figure 7) ✅
- [x] Figure 7: Pareto Front (formerly Figure 8) ✅
- [x] Figure 8: Extended Training (formerly Figure 9) ✅

### Tables
- [x] Table 1: Related Work Comparison ✅
- [x] Table 2: Algorithm Performance ✅
- [x] Table 3: Structural Comparison ✅
- [x] Table 4: Capacity Scan Results ✅
- [x] Table 5: State Space Ablation ✅
- [x] Table 6: HCA2C Ablation ✅
- [x] Table 7: Ablation Study ✅
- [x] Table 8: Extended Training ✅
- [x] Table 9: Generalization Results ✅

---

## ⚠️ Known Issues (Resolved)

### Issue 1: Unicode Characters (FIXED ✅)
- **Problem**: Square boxes (□) appearing in PDF
- **Cause**: Unicode characters (×, ±, →) in source files
- **Solution**: Replaced all with LaTeX commands
- **Status**: ✅ RESOLVED - 0 Unicode characters remain

### Issue 2: Figure 6 (FIXED ✅)
- **Problem**: User wanted to delete Figure 6
- **Solution**: Deleted Figure 6, automatic renumbering applied
- **Status**: ✅ RESOLVED - Figure deleted, numbering correct

---

## 📝 Submission Instructions

### Step 1: Verify Files
1. Open `submission_ready/manuscript.pdf` in Adobe Acrobat Reader
2. Check that all mathematical symbols render correctly
3. Verify all figures and tables are present
4. Confirm page count is 46 pages

### Step 2: Prepare for Upload
1. Navigate to Applied Soft Computing submission portal
2. Create/login to Editorial Manager account
3. Start new submission

### Step 3: Upload Files
Upload in this order:
1. `manuscript.pdf` (main manuscript)
2. `cover_letter.pdf` (cover letter)
3. `graphical_abstract.png` (graphical abstract)
4. `manuscript_latex_source.zip` (LaTeX source)
5. `figures.zip` (individual figures)

### Step 4: Complete Submission Form
- Title: Deep Reinforcement Learning for Vertical Layered Queueing Systems in Urban Air Mobility
- Authors: ZhiHan Wang (corresponding author)
- Keywords: (copy from manuscript)
- Highlights: (copy from manuscript)
- Suggested reviewers: (optional)

### Step 5: Review and Submit
- Review all uploaded files
- Check metadata is correct
- Confirm all declarations
- Submit manuscript

---

## ✅ Final Status

**ALL TASKS COMPLETED**

The manuscript is ready for submission to Applied Soft Computing. All Unicode character issues have been resolved, Figure 6 has been deleted as requested, and all submission materials are prepared.

**Estimated submission time**: 30 minutes (for uploading and form completion)

---

## 📞 Support

If you encounter any issues during submission:

1. **Unicode/rendering issues**: 
   - Try Adobe Acrobat Reader
   - Check FINAL_VERIFICATION_REPORT.md

2. **File upload issues**:
   - Verify file sizes are within limits
   - Check file formats are correct

3. **Technical questions**:
   - Refer to Applied Soft Computing author guidelines
   - Contact journal editorial office

---

**Ready to submit!** 🚀

