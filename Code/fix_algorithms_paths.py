"""
修复algorithms目录中的所有路径引用
Fix All Path References in algorithms Directory
"""

import os
import re
from pathlib import Path

def fix_baselines_paths(filepath):
    """修复baselines文件中的路径（需要../../../Models/）"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 修复 ./models/ → ../../../Models/
    content = re.sub(
        r"(['\"])\.\/models\/",
        r"\1../../../Models/",
        content
    )

    # 修复单独的 './models' → '../../../Models'
    content = re.sub(
        r"(['\"])\.\/models(['\"])",
        r"\1../../../Models\2",
        content
    )

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def fix_advanced_paths(filepath):
    """修复advanced算法文件中的路径（需要../../../../Models/）"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 修复 ./models/ → ../../../../Models/
    content = re.sub(
        r"(['\"])\.\/models\/",
        r"\1../../../../Models/",
        content
    )

    # 修复 save_dir': './models/td7' 这样的配置
    content = re.sub(
        r"'save_dir':\s*'\.\/models\/",
        r"'save_dir': '../../../../Models/",
        content
    )

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """主函数"""
    code_dir = Path(__file__).parent
    fixed_files = []

    print("="*80)
    print("修复algorithms目录中的路径...")
    print("="*80)

    # 修复baselines
    baselines_dir = code_dir / 'algorithms' / 'baselines'
    if baselines_dir.exists():
        print(f"\n正在检查 algorithms/baselines/ ...")
        for py_file in baselines_dir.glob('*.py'):
            try:
                if fix_baselines_paths(py_file):
                    fixed_files.append(py_file)
                    print(f"  ✓ Fixed: {py_file.name}")
            except Exception as e:
                print(f"  ✗ Error fixing {py_file.name}: {e}")

    # 修复advanced
    advanced_dir = code_dir / 'algorithms' / 'advanced'
    if advanced_dir.exists():
        print(f"\n正在检查 algorithms/advanced/ ...")
        for subdir in advanced_dir.iterdir():
            if subdir.is_dir():
                for py_file in subdir.glob('*.py'):
                    try:
                        if fix_advanced_paths(py_file):
                            fixed_files.append(py_file)
                            print(f"  ✓ Fixed: {py_file.relative_to(code_dir)}")
                    except Exception as e:
                        print(f"  ✗ Error fixing {py_file.name}: {e}")

    print(f"\n{'='*80}")
    print(f"路径修复完成！共修复了 {len(fixed_files)} 个文件")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
