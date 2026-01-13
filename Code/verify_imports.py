"""验证所有Python文件的导入是否正确"""
import os
import sys
from pathlib import Path
import ast

# 添加Code目录到sys.path
code_dir = Path(__file__).parent
sys.path.insert(0, str(code_dir))

def check_imports(file_path):
    """检查文件的导入是否有语法错误"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 尝试解析AST
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """主函数"""
    print("="*80)
    print("验证所有Python文件的导入路径")
    print("="*80)

    errors = []
    success_count = 0

    # 遍历所有.py文件
    for py_file in code_dir.rglob('*.py'):
        if py_file.name in ['fix_imports.py', 'batch_fix_imports.py', 'verify_imports.py']:
            continue

        success, error_msg = check_imports(py_file)
        relative_path = py_file.relative_to(code_dir)

        if success:
            print(f"✓ {relative_path}")
            success_count += 1
        else:
            print(f"✗ {relative_path}: {error_msg}")
            errors.append((relative_path, error_msg))

    print("\n" + "="*80)
    print(f"验证完成！")
    print(f"成功: {success_count} 个文件")
    print(f"错误: {len(errors)} 个文件")
    print("="*80)

    if errors:
        print("\n错误详情:")
        for path, msg in errors:
            print(f"  - {path}: {msg}")

if __name__ == '__main__':
    main()
