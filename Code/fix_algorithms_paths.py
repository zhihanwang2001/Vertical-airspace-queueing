"""
Fix All Path References in algorithms Directory
"""

import os
import re
from pathlib import Path

def fix_baselines_paths(filepath):
    """Fix paths in baselines files (need ../../../Models/)"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Fix ./models/ → ../../../Models/
    content = re.sub(
        r"(['\"])\.\/models\/",
        r"\1../../../Models/",
        content
    )

    # Fix standalone './models' → '../../../Models'
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
    """Fix paths in advanced algorithm files (need ../../../../Models/)"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Fix ./models/ → ../../../../Models/
    content = re.sub(
        r"(['\"])\.\/models\/",
        r"\1../../../../Models/",
        content
    )

    # Fix save_dir': './models/td7' style configurations
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
    """Main function"""
    code_dir = Path(__file__).parent
    fixed_files = []

    print("="*80)
    print("Fixing paths in algorithms directory...")
    print("="*80)

    # Fix baselines
    baselines_dir = code_dir / 'algorithms' / 'baselines'
    if baselines_dir.exists():
        print(f"\nChecking algorithms/baselines/ ...")
        for py_file in baselines_dir.glob('*.py'):
            try:
                if fix_baselines_paths(py_file):
                    fixed_files.append(py_file)
                    print(f"  ✓ Fixed: {py_file.name}")
            except Exception as e:
                print(f"  ✗ Error fixing {py_file.name}: {e}")

    # Fix advanced
    advanced_dir = code_dir / 'algorithms' / 'advanced'
    if advanced_dir.exists():
        print(f"\nChecking algorithms/advanced/ ...")
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
    print(f"Path fixing completed! Fixed {len(fixed_files)} files")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
