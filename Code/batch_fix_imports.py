"""Batch fix import paths in all Python files"""
import os
from pathlib import Path

def fix_file(filepath):
    """Fix imports in a single file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    modified = False
    new_lines = []

    for line in lines:
        original_line = line

        # Replace baselines imports
        if 'from baselines.' in line and 'algorithms.baselines' not in line:
            line = line.replace('from baselines.', 'from algorithms.baselines.')
            modified = True

        # Replace advanced_algorithms imports
        if 'from advanced_algorithms.' in line and 'algorithms.advanced' not in line:
            line = line.replace('from advanced_algorithms.', 'from algorithms.advanced.')
            modified = True

        new_lines.append(line)

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return True
    return False

# Fix training_scripts
training_scripts_dir = Path('/Users/harry./Desktop/PostGraduate/final/RP1/Code/training_scripts')
analysis_scripts_dir = Path('/Users/harry./Desktop/PostGraduate/final/RP1/Code/analysis_scripts')

fixed_count = 0
for script_dir in [training_scripts_dir, analysis_scripts_dir]:
    if script_dir.exists():
        for py_file in script_dir.glob('*.py'):
            if fix_file(py_file):
                print(f"✓ Fixed: {py_file.name}")
                fixed_count += 1

print(f"\nTotal fixed: {fixed_count} files")
