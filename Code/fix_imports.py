"""
Fix Import Paths Script

Fix import statements in all Python files to adapt to new folder structure
"""

import os
import re
from pathlib import Path

def fix_import_statements(file_path):
    """Fix import statements in a single file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 1. Fix 'from env.' Import
    content = re.sub(
        r'^from env\.',
        'from env.',
        content,
        flags=re.MULTILINE
    )

    # 2. Fix 'from baselines.' import to 'from algorithms.baselines.'
    content = re.sub(
        r'^from baselines\.',
        'from algorithms.baselines.',
        content,
        flags=re.MULTILINE
    )

    # 3. Fix 'from advanced_algorithms.' import to 'from algorithms.advanced.'
    content = re.sub(
        r'^from advanced_algorithms\.',
        'from algorithms.advanced.',
        content,
        flags=re.MULTILINE
    )

    # 4. Remove sys.path reference to rpTransition
    content = re.sub(
        r"sys\.path\.append\(os\.path\.join\(os\.path\.dirname\(__file__\), '\.\.', 'rpTransition'\)\)\n?",
        '',
        content
    )

    # 5. Fix heterogeneous_configs import
    # Remove dynamic import code block
    content = re.sub(
        r'# Import异质性配置生成器\nimport importlib\.util\nspec = importlib\.util\.spec_from_file_location\(\s*"heterogeneous_configs",\s*os\.path\.join\(os\.path\.dirname\(__file__\), \'\.\.\'.*?\n\).*?\nheterogeneous_configs = importlib\.util\.module_from_spec\(spec\)\nspec\.loader\.exec_module\(heterogeneous_configs\)\n\nHeterogeneousRegionConfigs = heterogeneous_configs\.HeterogeneousRegionConfigs',
        '# Import heterogeneous config generator\nfrom heterogeneous_configs import HeterogeneousRegionConfigs',
        content,
        flags=re.DOTALL
    )

    # 6. Add correct sys.path setup (if needed)
    if 'training_scripts' in str(file_path) or 'analysis_scripts' in str(file_path):
        # Check if sys.path.insert already exists
        if 'sys.path.insert(0' not in content and 'sys.path.append' in content:
            # Insert correct sys.path setup before first import
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('import sys'):
                    # Find location of import os
                    for j in range(i+1, min(i+10, len(lines))):
                        if 'import os' in lines[j]:
                            # Insert sys.path setup after
                            lines.insert(j+1, '# Add parent directory to path for imports')
                            lines.insert(j+2, 'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))')
                            break
                    break
            content = '\n'.join(lines)

    # If content changed, write back to file
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Main function: Traverse all Python files and fix imports"""
    code_dir = Path(__file__).parent
    fixed_files = []

    # Traverse all .py files
    for py_file in code_dir.rglob('*.py'):
        if py_file.name == 'fix_imports.py':
            continue

        try:
            if fix_import_statements(py_file):
                fixed_files.append(py_file)
                print(f"✓ Fixed: {py_file.relative_to(code_dir)}")
        except Exception as e:
            print(f"✗ Error fixing {py_file}: {e}")

    print(f"\n{'='*80}")
    print(f"Fix completed! Fixed {len(fixed_files)} files")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
