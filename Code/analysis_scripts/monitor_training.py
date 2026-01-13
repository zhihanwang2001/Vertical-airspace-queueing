#!/usr/bin/env python3
"""
训练监控脚本
Training Monitor Script
"""

import os
import time
import subprocess
import glob
from datetime import datetime

def check_tensorboard():
    """检查TensorBoard是否运行"""
    try:
        result = subprocess.run(['pgrep', '-f', 'tensorboard'], 
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def check_training_processes():
    """检查训练进程"""
    try:
        result = subprocess.run(['pgrep', '-f', 'run_baseline_comparison'], 
                              capture_output=True, text=True)
        pids = result.stdout.strip().split('\n') if result.stdout.strip() else []
        return len(pids)
    except:
        return 0

def get_log_files():
    """获取日志文件"""
    logs = glob.glob('*.log')
    return logs

def monitor_training():
    """监控训练状态"""
    print("=" * 60)
    print(f"Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 检查TensorBoard
    tb_running = check_tensorboard()
    print(f"TensorBoard Status: {'🟢 Running' if tb_running else '🔴 Not Running'}")
    
    # 检查训练进程
    training_count = check_training_processes()
    print(f"Training Processes: {training_count} running")
    
    # 检查日志文件
    log_files = get_log_files()
    print(f"Log Files: {len(log_files)} found")
    
    if log_files:
        print("\nRecent Log Activity:")
        for log_file in log_files[:5]:  # 只显示前5个
            try:
                stat = os.stat(log_file)
                size = stat.st_size / 1024  # KB
                mtime = datetime.fromtimestamp(stat.st_mtime)
                print(f"  {log_file}: {size:.1f}KB, modified {mtime.strftime('%H:%M:%S')}")
            except:
                pass
    
    # 检查TensorBoard日志
    tb_dirs = glob.glob('tensorboard_logs/*')
    print(f"\nTensorBoard Logs: {len(tb_dirs)} directories")
    
    if not tb_running:
        print("\n🚀 To start TensorBoard:")
        print("nohup python3 -m tensorboard.main --logdir=./tensorboard_logs --host=0.0.0.0 --port=6006 > tensorboard.log 2>&1 &")
    else:
        print("\n🌐 Access TensorBoard:")
        print("Local: ssh -L 6006:localhost:6006 user@server")
        print("Then visit: http://localhost:6006")

if __name__ == "__main__":
    while True:
        try:
            monitor_training()
            print("\n" + "=" * 60)
            print("Press Ctrl+C to exit, or wait 30s for next update...")
            time.sleep(30)
            os.system('clear' if os.name == 'posix' else 'cls')
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)