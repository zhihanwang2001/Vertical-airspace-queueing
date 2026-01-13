"""
全面修复所有文件路径引用
Fix All File Path References

这个脚本会修复：
1. 模型文件路径 (./models/ → ../../Models/)
2. 结果文件路径 (./comparison_results → ../../Results/comparison)
3. 图表保存路径 (analysis_scripts中的相对路径)
"""

import os
import re
from pathlib import Path

def fix_paths_in_file(filepath):
    """修复单个文件中的所有路径"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    filename = filepath.name
    parent_dir = filepath.parent.name

    # 1. 修复模型路径
    # ./models/ → ../../Models/
    content = re.sub(
        r'(["\'])\.\/models\/',
        r'\1../../Models/',
        content
    )

    # 2. 修复结果路径 - comparison_results
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

    # 3. 修复结果路径 - generalization_results
    content = re.sub(
        r'(["\'])\.\/generalization_results(["\'/])',
        r'\1../../Results/generalization\2',
        content
    )

    # 4. 修复结果路径 - result_excel
    content = re.sub(
        r'(["\'])\.\/result_excel\/',
        r'\1../../Results/excel/',
        content
    )

    # 5. 修复图表保存路径（仅在analysis_scripts中）
    if parent_dir == 'analysis_scripts':
        # 修复直接保存到当前目录的PNG文件 → Figures/analysis/
        # 但保留已经有路径的
        content = re.sub(
            r"plt\.savefig\(['\"]([a-zA-Z0-9_]+\.png)['\"](.*?)\)",
            r"plt.savefig('../../Figures/analysis/\1'\2)",
            content
        )

        # 修复output_dir为相对路径的情况
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

    # 6. 修复training_scripts中output_dir的路径
    if parent_dir == 'training_scripts':
        # 修复默认输出目录
        content = re.sub(
            r'save_path\s*=\s*["\']\.\/models\/',
            'save_path = "../../Models/',
            content
        )

    # 7. 修复Path对象中的路径
    content = re.sub(
        r'Path\(["\']\.\/generalization_results["\']\)',
        'Path("../../Results/generalization")',
        content
    )

    # 8. 修复makedirs中的路径
    content = re.sub(
        r'makedirs\(f?["\'][^"\']*\/models\/',
        'makedirs(f"../../Models/',
        content
    )

    # 如果内容有变化，写回文件
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """主函数：遍历所有Python文件并修复路径"""
    code_dir = Path(__file__).parent
    fixed_files = []

    print("="*80)
    print("开始修复所有文件路径...")
    print("="*80)

    # 遍历training_scripts和analysis_scripts
    for script_dir in ['training_scripts', 'analysis_scripts']:
        dir_path = code_dir / script_dir
        if not dir_path.exists():
            continue

        print(f"\n正在检查 {script_dir}/ ...")

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
    print(f"路径修复完成！共修复了 {len(fixed_files)} 个文件")
    print(f"{'='*80}")

    if fixed_files:
        print("\n修复的文件列表:")
        for f in fixed_files:
            print(f"  - {f.relative_to(code_dir)}")

if __name__ == '__main__':
    main()
