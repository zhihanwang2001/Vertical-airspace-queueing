"""
Comprehensive Fix for All File Path References

This script fixes:
1. Model file paths (./models/ → ../../Models/)
2. Result file paths (./comparison_results → ../../Results/comparison)
3. Figure save paths (relative paths in analysis_scripts)
"""

import os
import re
from pathlib import Path

def fix_paths_in_file(filepath):
    """Fix all paths in a single file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    filename = filepath.name
    parent_dir = filepath.parent.name

    # 1. Fix model paths
    # ./models/ → ../../Models/
    content = re.sub(
        r'(["\'])\.\/models\/',
        r'\1../../Models/',
        content
    )

    # 2. Fix result paths - comparison_results
    content = re.sub(
        r'(["\'])\.\/comparison_results(["\'/])',
        r'\1../../Results/comparison\2',
        content
    )

    # advanced_comparison_results → Results/comparison
    content = re.sub(
        r'(["\'])\.\/advanced_comparison_results\/',
        r'\1../../Results/comparison/',
        content
    )

    # 3. Fix result paths - generalization_results
    content = re.sub(
        r'(["\'])\.\/generalization_results(["\'/])',
        r'\1../../Results/generalization\2',
        content
    )

    # 4. Fix result paths - result_excel
    content = re.sub(
        r'(["\'])\.\/result_excel\/',
        r'\1../../Results/excel/',
        content
    )

    # 5. Fix figure save paths (only in analysis_scripts)
    if parent_dir == 'analysis_scripts':
        # Fix PNG files saved directly to current directory → Figures/analysis/
        # But preserve paths that already exist
        content = re.sub(
            r"plt\.savefig\(['\"]([a-zA-Z0-9_]+\.png)['\"](.*?)\)",
            r"plt.savefig('../../Figures/analysis/\1'\2)",
            content
        )

        # Fix output_dir with relative paths
        content = re.sub(
            r'output_dir\s*=\s*["\']\.\/paper_figures\/?["\']',
            'output_dir = "../../Figures/publication/"',
            content
        )

        content = re.sub(
            r'output_dir\s*=\s*Path\(["\']\.\/paper_figures\/?["\']\)',
            'output_dir = Path("../../Figures/publication/")',
            content
        )

    # 6. Fix output_dir paths in training_scripts
    if parent_dir == 'training_scripts':
        # Fix default output directory
        content = re.sub(
            r'save_path\s*=\s*["\']\.\/models\/',
            'save_path = "../../Models/',
            content
        )

    # 7. Fix paths in Path objects
    content = re.sub(
        r'Path\(["\']\.\/generalization_results["\']\)',
        'Path("../../Results/generalization")',
        content
    )

    # 8. Fix paths in makedirs
    content = re.sub(
        r'makedirs\(f?["\'][^"\']*\/models\/',
        'makedirs(f"../../Models/',
        content
    )

    # If content changed, write back to file
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Main function: Traverse all Python files and fix paths"""
    code_dir = Path(__file__).parent
    fixed_files = []

    print("="*80)
    print("Starting to fix all file paths...")
    print("="*80)

    # Traverse training_scripts and analysis_scripts
    for script_dir in ['training_scripts', 'analysis_scripts']:
        dir_path = code_dir / script_dir
        if not dir_path.exists():
            continue

        print(f"\nChecking {script_dir}/ ...")

        for py_file in dir_path.glob('*.py'):
            try:
                if fix_paths_in_file(py_file):
                    fixed_files.append(py_file)
                    print(f"  ✓ Fixed: {py_file.name}")
                else:
                    print(f"  - No change: {py_file.name}")
            except Exception as e:
                print(f"  ✗ Error fixing {py_file.name}: {e}")

    print(f"\n{'='*80}")
    print(f"Path fixing completed! Fixed {len(fixed_files)} files")
    print(f"{'='*80}")

    if fixed_files:
        print("\nList of fixed files:")
        for f in fixed_files:
            print(f"  - {f.relative_to(code_dir)}")

if __name__ == '__main__':
    main()
