"""
Major Revision Experiment 1.3: Empirical State Space Measurement
Empirically measure state space size

Key questions:
- Paper uses 3^K to estimate state space
- Reviewer questions: This is only theoretical upper bound, actual reachable states may be much smaller
- If actual state space is much smaller, "state space explosion" hypothesis doesn't hold

Experiment design:
1. Monte Carlo sampling: Run 100K steps, record all visited unique states
2. Test configurations: K=10, K=20, K=23, K=30
3. Use random policy to ensure broad exploration
4. Statistics:
   - Number of unique states
   - State visit frequency distribution
   - Ratio to 3^K theoretical value

Expected results:
- Best case: Actual states ≈ 3^K → Theory correct
- Worst case: Actual states << 3^K (e.g. 0.01×) → Theory incorrect
- Most likely: Actual states ≈ 0.1-0.5 × 3^K → 3^K is reasonable upper bound
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gymnasium as gym
import numpy as np
import json
from collections import defaultdict
from datetime import datetime
import hashlib

from env.config import VerticalQueueConfig
from env.configurable_env_wrapper import ConfigurableEnvWrapper
from env.drl_wrapper_fixed import DictToBoxActionWrapperFixed, ObservationWrapperFixed


def state_to_hashable(state_dict):
    """
    Convert state dictionary to hashable string

    Key state components (based on VerticalQueueConfig):
    - queue_lengths: [5] queue length per layer
    - service_classes: [3] number of orders per class in system
    - May have other features, but queue length is core
    """
    # Extract queue lengths (most critical state dimension)
    if isinstance(state_dict, dict):
        # If dictionary, try to extract queue-related info
        queue_info = []
        for key in sorted(state_dict.keys()):
            if 'queue' in key.lower() or 'layer' in key.lower():
                val = state_dict[key]
                if isinstance(val, (list, np.ndarray)):
                    queue_info.extend([int(v) for v in val])
                else:
                    queue_info.append(int(val))

        if queue_info:
            return tuple(queue_info)

    # If array, use directly
    if isinstance(state_dict, (list, np.ndarray)):
        return tuple([int(v) for v in state_dict])

    # Fallback: use hash
    return hashlib.md5(str(state_dict).encode()).hexdigest()


def extract_queue_lengths(env):
    """
    Extract current queue lengths from environment

    This is the core dimension of state space
    """
    try:
        # Environment structure: ObservationWrapperFixed -> DictToBoxActionWrapperFixed -> ConfigurableEnvWrapper -> DRLOptimizedQueueEnvFixed
        # Need to access the bottom-level DRLOptimizedQueueEnvFixed

        core_env = env
        depth = 0
        # Unwrap layer by layer until finding environment with queue_lengths
        while hasattr(core_env, 'env'):
            depth += 1
            core_env = core_env.env
            if depth > 10:  # Prevent infinite loop
                print(f"⚠️ Environment nesting too deep (>{depth})")
                break

        # Now core_env should be DRLOptimizedQueueEnvFixed
        if hasattr(core_env, 'queue_lengths'):
            # queue_lengths is numpy array, convert to integer tuple
            queue_lengths = core_env.queue_lengths
            if queue_lengths is not None and len(queue_lengths) > 0:
                return tuple(int(x) for x in queue_lengths)
            else:
                # Debug info: queue_lengths exists but is empty
                return None

        # Fallback: if still not found, try to access env's state directly
        if hasattr(core_env, 'state'):
            state = core_env.state
            if isinstance(state, dict) and 'queue_lengths' in state:
                return tuple(int(x) for x in state['queue_lengths'])

        # Last attempt: check if there's a get_state method
        if hasattr(core_env, 'get_state'):
            state = core_env.get_state()
            if isinstance(state, dict) and 'queue_lengths' in state:
                return tuple(int(x) for x in state['queue_lengths'])

        # Complete failure - only print warning on first time
        return None
    except Exception as e:
        print(f"⚠️ Failed to extract queue lengths: {e}")
        import traceback
        traceback.print_exc()
        return None


def measure_state_space(
    capacity_config,
    config_name,
    n_steps=100_000,
    n_runs=3
):
    """
    Measure state space size

    Args:
        capacity_config: Capacity configuration list[int]
        config_name: Configuration name str
        n_steps: Number of steps per run
        n_runs: Number of runs (multiple samples to ensure coverage)

    Returns:
        dict: Measurement results
    """

    print(f"\n{'='*80}")
    print(f"Measuring state space: {config_name}")
    print(f"Capacity config: {capacity_config} (Total K={sum(capacity_config)})")
    print(f"Sampling steps: {n_steps:,} × {n_runs} runs")
    print(f"{'='*80}\n")

    # 理论估计
    K = sum(capacity_config)
    theoretical_states_3K = 3 ** K
    theoretical_states_product = np.prod([c + 1 for c in capacity_config])

    print(f"理论估计:")
    print(f"  3^K = 3^{K} = {theoretical_states_3K:,}")
    print(f"  ∏(C_l+1) = {theoretical_states_product:,}")
    print()

    # 创建环境
    config = VerticalQueueConfig()
    config.layer_capacities = capacity_config
    config.arrival_weights = [0.3, 0.25, 0.2, 0.15, 0.1]

    # 10× 高负载 (相对于baseline 0.5)
    # Baseline = 0.5, 所以 10× baseline = 5.0
    config.base_arrival_rate = 5.0

    all_unique_states = set()
    state_frequencies = defaultdict(int)

    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}...")

        env = ConfigurableEnvWrapper(config)
        env = DictToBoxActionWrapperFixed(env)
        env = ObservationWrapperFixed(env)

        obs, _ = env.reset()
        run_unique_states = set()

        # 验证extract_queue_lengths是否工作
        if run == 0:  # 只在第一次run时验证
            test_state = extract_queue_lengths(env)
            if test_state is None:
                print(f"⚠️  警告: extract_queue_lengths返回None!")
                print(f"   环境类型: {type(env)}")
                # 尝试手动检查环境结构
                temp_env = env
                depth = 0
                print(f"   环境层次结构:")
                while hasattr(temp_env, 'env'):
                    print(f"     [{depth}] {type(temp_env).__name__}")
                    temp_env = temp_env.env
                    depth += 1
                    if depth > 10:
                        break
                print(f"     [{depth}] {type(temp_env).__name__} (底层)")
                if hasattr(temp_env, 'queue_lengths'):
                    print(f"   ✅ 底层有queue_lengths: {temp_env.queue_lengths}")
                else:
                    print(f"   ❌ 底层没有queue_lengths属性!")
            else:
                print(f"   ✅ extract_queue_lengths工作正常: {test_state}")

        for step in range(n_steps):
            # 提取队列长度状态
            queue_state = extract_queue_lengths(env)

            if queue_state is not None:
                run_unique_states.add(queue_state)
                all_unique_states.add(queue_state)
                state_frequencies[queue_state] += 1

            # 随机动作（确保广泛探索）
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()

            if (step + 1) % 10000 == 0:
                print(f"  Step {step+1:,}: {len(run_unique_states):,} unique states (cumulative: {len(all_unique_states):,})")

        env.close()

        print(f"  Run {run+1} 完成: {len(run_unique_states):,} 新状态")

    # 分析结果
    n_unique = len(all_unique_states)
    ratio_3K = n_unique / theoretical_states_3K if theoretical_states_3K > 0 else 0
    ratio_product = n_unique / theoretical_states_product if theoretical_states_product > 0 else 0

    # 访问频率统计
    frequencies = list(state_frequencies.values())
    top_states = sorted(state_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]

    # 检查是否有有效数据
    if len(frequencies) == 0:
        print(f"⚠️  警告: 未收集到任何状态数据！")
        frequency_stats = {
            'mean_visits': 0.0,
            'median_visits': 0.0,
            'max_visits': 0,
            'min_visits': 0,
        }
    else:
        frequency_stats = {
            'mean_visits': float(np.mean(frequencies)),
            'median_visits': float(np.median(frequencies)),
            'max_visits': int(np.max(frequencies)),
            'min_visits': int(np.min(frequencies)),
        }

    results = {
        'config_name': config_name,
        'capacity_config': capacity_config,
        'K': K,

        'theoretical': {
            '3^K': int(theoretical_states_3K) if theoretical_states_3K < 1e15 else str(theoretical_states_3K),
            'product_C+1': int(theoretical_states_product),
        },

        'empirical': {
            'n_steps_total': n_steps * n_runs,
            'n_runs': n_runs,
            'unique_states': n_unique,
            'ratio_to_3K': float(ratio_3K),
            'ratio_to_product': float(ratio_product),
        },

        'frequency_stats': frequency_stats,

        'top_10_states': [
            {'state': list(state), 'visits': int(freq)}
            for state, freq in top_states
        ],

        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'purpose': 'Major Revision Exp 1.3: State Space Measurement'
        }
    }

    # 输出结果
    print(f"\n{'='*80}")
    print(f"测量结果:")
    print(f"  唯一状态数: {n_unique:,}")
    print(f"  理论 3^K: {theoretical_states_3K:,}" if theoretical_states_3K < 1e15 else f"  理论 3^K: {theoretical_states_3K:.2e}")
    print(f"  实际/理论比率: {ratio_3K:.2%}")

    if len(frequencies) > 0:
        print(f"\n  平均访问次数: {np.mean(frequencies):.1f}")
        print(f"  中位数访问: {np.median(frequencies):.0f}")
        print(f"  最高访问: {np.max(frequencies):,}")
        print(f"\n  Top 3 most frequent states:")
        for i, (state, freq) in enumerate(top_states[:3]):
            pct = freq / (n_steps * n_runs) * 100
            print(f"    {i+1}. {list(state)} - {freq:,} visits ({pct:.2f}%)")
    else:
        print(f"\n  ⚠️  无频率数据（未收集到状态）")

    print(f"{'='*80}\n")

    return results


def main():
    """
    主函数：测量所有关键配置的状态空间
    """

    configurations = [
        ([2, 2, 2, 2, 2], "K=10 Low Capacity"),
        ([4, 4, 4, 4, 4], "K=20 Uniform"),
        ([8, 6, 4, 3, 2], "K=23 Inverted Pyramid"),
        ([5, 5, 5, 5, 5], "K=25 Uniform"),
        ([6, 6, 6, 6, 6], "K=30 Uniform"),
        # ([8, 8, 8, 8, 8], "K=40 High Capacity"),  # 可选：如果时间充足
    ]

    print(f"\n{'#'*80}")
    print(f"# Major Revision Experiment 1.3: State Space Measurement")
    print(f"# 配置数量: {len(configurations)}")
    print(f"# 预计时间: ~{len(configurations) * 10} 分钟")
    print(f"{'#'*80}\n")

    all_results = []

    for capacity_config, config_name in configurations:
        result = measure_state_space(
            capacity_config=capacity_config,
            config_name=config_name,
            n_steps=100_000,
            n_runs=3
        )

        all_results.append(result)

    # 保存结果
    output_file = Path("Results/major_revision_exp3/state_space_measurement.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'#'*80}")
    print(f"# 所有测量完成！")
    print(f"# 结果保存到: {output_file}")
    print(f"{'#'*80}\n")

    # 汇总表格
    print("\n汇总表格:")
    print("="*100)
    print(f"{'配置':<25} {'K':>5} {'理论3^K':>20} {'实际唯一':>15} {'比率':>10}")
    print("="*100)

    for r in all_results:
        K = r['K']
        theoretical = r['theoretical']['3^K']
        if isinstance(theoretical, str):
            theoretical_str = theoretical
        else:
            theoretical_str = f"{theoretical:,}"

        empirical = r['empirical']['unique_states']
        ratio = r['empirical']['ratio_to_3K']

        print(f"{r['config_name']:<25} {K:>5} {theoretical_str:>20} {empirical:>15,} {ratio:>9.2%}")

    print("="*100)

    # 关键发现
    print("\n关键发现:")
    print("-" * 80)

    for r in all_results:
        ratio = r['empirical']['ratio_to_3K']
        if ratio > 0.5:
            assessment = "✅ 理论估计准确（实际≈理论）"
        elif ratio > 0.1:
            assessment = "⚠️  理论是合理上界（实际为理论的10-50%）"
        elif ratio > 0.01:
            assessment = "⚠️  理论高估较多（实际<10%理论）"
        else:
            assessment = "❌ 理论严重高估（实际<<1%理论）"

        print(f"{r['config_name']}: {assessment}")

    print("-" * 80)
    print("\n分析完成！请用此数据修订论文的状态空间假说部分。")


if __name__ == "__main__":
    main()
