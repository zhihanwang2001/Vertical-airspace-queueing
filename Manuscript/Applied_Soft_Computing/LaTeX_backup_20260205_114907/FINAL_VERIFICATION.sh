#!/bin/bash
# Final Verification Script

echo "==================================================================="
echo "Manuscript Final Verification"
echo "Date: $(date '+%Y-%m-%d %H:%M')"
echo "==================================================================="
echo ""

# Check PDF exists and size
echo "1. Checking manuscript.pdf..."
if [ -f "manuscript.pdf" ]; then
    SIZE=$(ls -lh manuscript.pdf | awk '{print $5}')
    PAGES=$(pdfinfo manuscript.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}')
    echo "   ✅ PDF exists: $SIZE, $PAGES pages"
else
    echo "   ❌ PDF not found!"
fi
echo ""

# Check key files exist
echo "2. Checking required files..."
FILES=(
    "manuscript.tex"
    "sections/ablation_study_simple.tex"
    "tables/tab_ablation_simple.tex"
)
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file missing!"
    fi
done
echo ""

# Check key numbers in manuscript
echo "3. Verifying key numbers in manuscript..."
echo "   Checking: 59.9% (DRL improvement)"
grep -q "59.9" manuscript.tex && echo "   ✅ 59.9% found" || echo "   ⚠️  59.9% not found"

echo "   Checking: 228,945 (HCA2C-Full reward)"
grep -q "228,945\|228945" manuscript.tex && echo "   ✅ 228,945 found" || echo "   ⚠️  228,945 not found"

echo "   Checking: 100% crash (HCA2C-Wide)"
grep -q "100.*crash\|100\\\\%" manuscript.tex && echo "   ✅ 100% crash found" || echo "   ⚠️  100% crash not found"

echo "   Checking: 85,650 (A2C-Baseline)"
grep -q "85,650\|85650" manuscript.tex && echo "   ✅ 85,650 found" || echo "   ⚠️  85,650 not found"
echo ""

# Check ablation study section
echo "4. Checking ablation study integration..."
grep -q "\\\\input{sections/ablation_study_simple}" manuscript.tex && echo "   ✅ Ablation section included" || echo "   ❌ Ablation section not included!"
grep -q "\\\\label{subsec:ablation}" sections/ablation_study_simple.tex && echo "   ✅ Ablation label exists" || echo "   ❌ Ablation label missing!"
grep -q "\\\\ref{tab:ablation}" sections/ablation_study_simple.tex && echo "   ✅ Table reference exists" || echo "   ❌ Table reference missing!"
echo ""

# Check highlights updated
echo "5. Checking highlights..."
grep -q "Ablation study proves capacity-aware clipping essential" manuscript.tex && echo "   ✅ Highlights updated with ablation" || echo "   ⚠️  Highlights may need update"
echo ""

# Check compilation status
echo "6. Checking compilation..."
if [ -f "manuscript.log" ]; then
    ERRORS=$(grep -c "^!" manuscript.log 2>/dev/null || echo "0")
    WARNINGS=$(grep -c "Warning" manuscript.log 2>/dev/null || echo "0")
    echo "   Errors: $ERRORS"
    echo "   Warnings: $WARNINGS"
    if [ "$ERRORS" -eq 0 ]; then
        echo "   ✅ No compilation errors"
    else
        echo "   ❌ Compilation errors found!"
    fi
else
    echo "   ⚠️  No log file found"
fi
echo ""

echo "==================================================================="
echo "Verification Complete"
echo "==================================================================="
