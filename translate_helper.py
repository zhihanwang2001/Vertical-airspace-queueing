#!/usr/bin/env python3
"""
Helper script to translate Chinese comments to English in Python files.
This creates a translation mapping that can be reviewed before applying.
"""

# Common translation mappings for this project
TRANSLATIONS = {
    # General terms
    "重置时间步": "Reset time step",
    "重置所有组件": "Reset all components",
    "重置性能统计": "Reset performance statistics",
    "获取初始观测": "Get initial observation",
    "返回初始状态": "Return initial state",

    # Environment terms
    "环境步进": "Environment step",
    "执行动作": "Execute action",
    "更新状态": "Update state",
    "计算奖励": "Calculate reward",
    "检查终止条件": "Check termination condition",
    "处理动作": "Process action",
    "更新性能统计": "Update performance statistics",
    "计算系统利用率": "Calculate system utilization",
    "检查稳定性条件": "Check stability condition",
    "检查终止": "Check termination",
    "性能退化": "Performance degradation",

    # Queue terms
    "队列长度": "Queue length",
    "队列状态": "Queue state",
    "队列动力学": "Queue dynamics",
    "层间转移": "Inter-layer transfer",
    "转移决策": "Transfer decision",
    "服务率": "Service rate",
    "到达率": "Arrival rate",
    "等待时间": "Waiting time",
    "吞吐量": "Throughput",

    # Layer terms
    "层索引": "Layer index",
    "层容量": "Layer capacity",
    "层高度": "Layer height",
    "当前层": "Current layer",
    "下一层": "Next layer",
    "上一层": "Previous layer",

    # Statistics terms
    "总吞吐量": "Total throughput",
    "总等待时间": "Total waiting time",
    "层利用率": "Layer utilization",
    "成功转移": "Successful transfers",
    "阻塞到达": "Blocked arrivals",

    # Action terms
    "动作空间": "Action space",
    "观测空间": "Observation space",
    "状态向量": "State vector",
    "动作向量": "Action vector",

    # Reward terms
    "奖励函数": "Reward function",
    "奖励权重": "Reward weights",
    "多目标优化": "Multi-objective optimization",

    # Configuration terms
    "配置参数": "Configuration parameters",
    "系统参数": "System parameters",
    "环境参数": "Environment parameters",

    # Common phrases
    "基于": "Based on",
    "实现": "Implementation",
    "初始化": "Initialize",
    "更新": "Update",
    "计算": "Calculate",
    "检查": "Check",
    "获取": "Get",
    "设置": "Set",
    "返回": "Return",
    "处理": "Process",

    # Theory terms
    "理论": "theory",
    "理论文档": "theoretical framework",
    "01理论": "theoretical framework",
    "倒金字塔": "inverted pyramid",
    "垂直分层": "vertical layered",
    "分层队列": "layered queue",
}

def find_chinese_lines(filepath):
    """Find all lines containing Chinese characters."""
    import re
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    chinese_lines = []
    for i, line in enumerate(lines, 1):
        if chinese_pattern.search(line):
            chinese_lines.append((i, line.rstrip()))

    return chinese_lines

def suggest_translation(text):
    """Suggest English translation for Chinese text."""
    for chinese, english in TRANSLATIONS.items():
        if chinese in text:
            text = text.replace(chinese, english)
    return text

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python translate_helper.py <file_path>")
        sys.exit(1)

    filepath = sys.argv[1]
    chinese_lines = find_chinese_lines(filepath)

    print(f"Found {len(chinese_lines)} lines with Chinese in {filepath}")
    print("\nFirst 20 lines:")
    for line_num, line in chinese_lines[:20]:
        print(f"{line_num}: {line}")
        suggested = suggest_translation(line)
        if suggested != line:
            print(f"   -> {suggested}")
        print()
