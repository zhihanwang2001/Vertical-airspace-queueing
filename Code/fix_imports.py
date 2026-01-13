"""
批量修复导入路径脚本
Fix Import Paths Script

修复所有Python文件中的import语句，适配新的文件夹结构
"""

import os
import re
from pathlib import Path

def fix_import_statements(file_path):
    """修复单个文件的import语句"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 1. 修复 'from env.' 导入
    content = re.sub(
        r'^from env\.',
        'from env.',
        content,
        flags=re.MULTILINE
    )

    # 2. 修复 'from baselines.' 导入为 'from algorithms.baselines.'
    content = re.sub(
        r'^from baselines\.',
        'from algorithms.baselines.',
        content,
        flags=re.MULTILINE
    )

    # 3. 修复 'from advanced_algorithms.' 导入为 'from algorithms.advanced.'
    content = re.sub(
        r'^from advanced_algorithms\.',
        'from algorithms.advanced.',
        content,
        flags=re.MULTILINE
    )

    # 4. 移除对rpTransition的sys.path引用
    content = re.sub(
        r"sys\.path\.append\(os\.path\.join\(os\.path\.dirname\(__file__\), '\.\.', 'rpTransition'\)\)\n?",
        '',
        content
    )

    # 5. 修复heterogeneous_configs的导入
    # 移除动态导入的代码块
    content = re.sub(
        r'# 导入异质性配置生成器\nimport importlib\.util\nspec = importlib\.util\.spec_from_file_location\(\s*"heterogeneous_configs",\s*os\.path\.join\(os\.path\.dirname\(__file__\), \'\.\.\'.*?\n\).*?\nheterogeneous_configs = importlib\.util\.module_from_spec\(spec\)\nspec\.loader\.exec_module\(heterogeneous_configs\)\n\nHeterogeneousRegionConfigs = heterogeneous_configs\.HeterogeneousRegionConfigs',
        '# 导入异质性配置生成器\nfrom heterogeneous_configs import HeterogeneousRegionConfigs',
        content,
        flags=re.DOTALL
    )

    # 6. 添加正确的sys.path设置（如果需要）
    if 'training_scripts' in str(file_path) or 'analysis_scripts' in str(file_path):
        # 检查是否已经有sys.path.insert
        if 'sys.path.insert(0' not in content and 'sys.path.append' in content:
            # 在第一个import之前插入正确的sys.path设置
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('import sys'):
                    # 找到import os的位置
                    for j in range(i+1, min(i+10, len(lines))):
                        if 'import os' in lines[j]:
                            # 在后面插入sys.path设置
                            lines.insert(j+1, '# Add parent directory to path for imports')
                            lines.insert(j+2, 'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))')
                            break
                    break
            content = '\n'.join(lines)

    # 如果内容有变化，写回文件
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """主函数：遍历所有Python文件并修复导入"""
    code_dir = Path(__file__).parent
    fixed_files = []

    # 遍历所有.py文件
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
    print(f"修复完成！共修复了 {len(fixed_files)} 个文件")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
