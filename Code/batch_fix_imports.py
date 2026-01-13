"""批量修复所有Python文件的导入路径"""
import os
from pathlib import Path

def fix_file(filepath):
    """修复单个文件的导入"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    modified = False
    new_lines = []

    for line in lines:
        original_line = line

        # 替换 baselines 导入
        if 'from baselines.' in line and 'algorithms.baselines' not in line:
            line = line.replace('from baselines.', 'from algorithms.baselines.')
            modified = True

        # 替换 advanced_algorithms 导入
        if 'from advanced_algorithms.' in line and 'algorithms.advanced' not in line:
            line = line.replace('from advanced_algorithms.', 'from algorithms.advanced.')
            modified = True

        new_lines.append(line)

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return True
    return False

# 修复training_scripts
training_scripts_dir = Path('/Users/harry./Desktop/PostGraduate/final/RP1/Code/training_scripts')
analysis_scripts_dir = Path('/Users/harry./Desktop/PostGraduate/final/RP1/Code/analysis_scripts')

fixed_count = 0
for script_dir in [training_scripts_dir, analysis_scripts_dir]:
    if script_dir.exists():
        for py_file in script_dir.glob('*.py'):
            if fix_file(py_file):
                print(f"✓ Fixed: {py_file.name}")
                fixed_count += 1

print(f"\n总计修复了 {fixed_count} 个文件")
