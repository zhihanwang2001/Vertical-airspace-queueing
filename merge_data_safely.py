"""
安全地合并服务器数据和本地数据
策略：
1. 检查本地已有的数据文件
2. 只复制本地不存在的文件
3. 保留本地已有的文件（不覆盖）
4. 生成详细的合并报告
"""
import os
import json
import shutil
from pathlib import Path
from datetime import datetime

def compare_and_merge():
    """安全地比对和合并数据"""
    
    # 定义路径
    server_backup = Path("server_backup_20260128/Data/hca2c_final_comparison")
    local_data = Path("Data/hca2c_final_comparison")
    
    # 确保本地目录存在
    local_data.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("数据合并安全检查")
    print("="*70)
    print(f"服务器备份: {server_backup}")
    print(f"本地数据: {local_data}")
    print()
    
    # 检查服务器备份是否存在
    if not server_backup.exists():
        print(f"❌ 错误: 服务器备份目录不存在: {server_backup}")
        return
    
    # 获取所有文件
    server_files = list(server_backup.glob("*"))
    local_files = {f.name: f for f in local_data.glob("*")}
    
    print(f"📊 统计:")
    print(f"  服务器文件: {len(server_files)} 个")
    print(f"  本地文件: {len(local_files)} 个")
    print()
    
    # 分类文件
    to_copy = []  # 需要复制的文件（本地不存在）
    existing = []  # 本地已存在的文件（保留本地版本）
    
    for server_file in server_files:
        filename = server_file.name
        
        if filename in local_files:
            existing.append(filename)
        else:
            to_copy.append(server_file)
    
    # 显示分析结果
    print("="*70)
    print("文件分析结果")
    print("="*70)
    print()
    
    print(f"✅ 本地已存在 (将保留): {len(existing)} 个")
    if existing:
        for fname in sorted(existing)[:10]:  # 只显示前10个
            print(f"  - {fname}")
        if len(existing) > 10:
            print(f"  ... 还有 {len(existing) - 10} 个文件")
    print()
    
    print(f"📥 需要从服务器复制: {len(to_copy)} 个")
    if to_copy:
        for fpath in sorted(to_copy):
            print(f"  - {fpath.name}")
    print()
    
    # 询问确认（自动确认模式）
    print("="*70)
    print("执行计划")
    print("="*70)
    print(f"将复制 {len(to_copy)} 个文件到本地")
    print(f"保留 {len(existing)} 个本地已有文件")
    print()
    
    # 执行复制
    copied_count = 0
    errors = []
    
    for server_file in to_copy:
        try:
            dest = local_data / server_file.name
            shutil.copy2(server_file, dest)
            copied_count += 1
            print(f"✓ 复制: {server_file.name}")
        except Exception as e:
            errors.append((server_file.name, str(e)))
            print(f"✗ 失败: {server_file.name} - {e}")
    
    print()
    print("="*70)
    print("合并完成")
    print("="*70)
    print(f"✅ 成功复制: {copied_count} 个文件")
    print(f"✅ 保留本地: {len(existing)} 个文件")
    if errors:
        print(f"❌ 失败: {len(errors)} 个文件")
        for fname, err in errors:
            print(f"  - {fname}: {err}")
    print()
    
    # 验证最终结果
    final_files = list(local_data.glob("*.json"))
    print(f"📊 最终统计:")
    print(f"  本地数据目录总文件数: {len(list(local_data.glob('*')))} 个")
    print(f"  JSON结果文件: {len(final_files)} 个")
    print()
    
    # 生成合并报告
    report = {
        "merge_time": datetime.now().isoformat(),
        "server_backup_path": str(server_backup),
        "local_data_path": str(local_data),
        "server_files_count": len(server_files),
        "local_existing_count": len(existing),
        "copied_count": copied_count,
        "errors_count": len(errors),
        "final_total_files": len(list(local_data.glob('*'))),
        "final_json_files": len(final_files),
        "existing_files": existing,
        "copied_files": [f.name for f in to_copy],
        "errors": errors,
    }
    
    report_file = Path("data_merge_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 详细报告已保存: {report_file}")
    print()
    
    return report

if __name__ == "__main__":
    report = compare_and_merge()
