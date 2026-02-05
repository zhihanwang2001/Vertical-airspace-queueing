#!/bin/bash

# 备份所有文件
echo "Creating backups..."
cp manuscript.tex manuscript.tex.unicode_bak

# 修复 × (乘号) - 在文本模式中
echo "Fixing × symbols in text mode..."
sed -i '' 's/\([0-9]\)×/\1$\\times$/g' manuscript.tex
sed -i '' 's/×\([0-9]\)/$\\times$\1/g' manuscript.tex

# 修复 ± (加减号) - 在文本模式中
echo "Fixing ± symbols..."
sed -i '' 's/±/$\\pm$/g' manuscript.tex

# 修复 – (en-dash) - 应该用 --
echo "Fixing en-dash..."
sed -i '' 's/–/--/g' manuscript.tex

# 修复 ≥ (大于等于)
echo "Fixing ≥ symbols..."
sed -i '' 's/≥/$\\geq$/g' manuscript.tex

# 修复 ≤ (小于等于)
echo "Fixing ≤ symbols..."
sed -i '' 's/≤/$\\leq$/g' manuscript.tex

echo "Unicode fixes applied!"
