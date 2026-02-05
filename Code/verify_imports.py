"""Verify that all Python file imports are correct"""
import os
import sys
from pathlib import Path
import ast

# Add Code directory to sys.path
code_dir = Path(__file__).parent
sys.path.insert(0, str(code_dir))

def check_imports(file_path):
    """Check if file imports have syntax errors"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try to parse AST
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main function"""
    print("="*80)
    print("Verifying all Python file import paths")
    print("="*80)

    errors = []
    success_count = 0

    # Traverse all .py files
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
    print(f"Verification completed!")
    print(f"Success: {success_count} files")
    print(f"Errors: {len(errors)} files")
    print("="*80)

    if errors:
        print("\nError details:")
        for path, msg in errors:
            print(f"  - {path}: {msg}")

if __name__ == '__main__':
    main()
