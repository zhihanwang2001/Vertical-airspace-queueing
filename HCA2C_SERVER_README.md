# HCA2C 服务器运行指南

## 文件说明

- `hca2c_package.tar.gz` - HCA2C算法代码包 (75KB)

## 服务器部署步骤

### 1. 上传并解压

```bash
# 上传到服务器后
tar -xzvf hca2c_package.tar.gz
```

### 2. 安装依赖

```bash
pip install torch numpy gymnasium stable-baselines3 rich
```

### 3. 运行完整实验

```bash
# 完整实验 (3算法 × 5种子 × 3负载 = 45次训练)
# 预计时间: 12-24小时 (取决于GPU)
python training_scripts/run_hca2c_comparison.py \
    --algorithms hca2c a2c ppo \
    --seeds 42 43 44 45 46 \
    --loads 3 5 10 \
    --timesteps 500000 \
    --output-dir ./hca2c_results
```

### 4. 快速测试 (验证代码可运行)

```bash
# 快速测试模式 (约5分钟)
python training_scripts/run_hca2c_comparison.py --test-mode
```

### 5. 中等规模实验 (如果时间有限)

```bash
# 中等规模 (3算法 × 3种子 × 2负载 = 18次训练)
# 预计时间: 4-8小时
python training_scripts/run_hca2c_comparison.py \
    --algorithms hca2c a2c ppo \
    --seeds 42 43 44 \
    --loads 5 10 \
    --timesteps 500000 \
    --output-dir ./hca2c_results
```

## 实验参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--algorithms` | 要比较的算法 | hca2c a2c ppo |
| `--seeds` | 随机种子列表 | 42 43 44 45 46 |
| `--loads` | 负载倍数列表 | 3 5 10 |
| `--timesteps` | 每次训练步数 | 500000 |
| `--output-dir` | 输出目录 | ./hca2c_results |

## 输出文件

实验完成后，输出目录包含：

```
hca2c_results/
├── 20260125_HHMMSS/           # 时间戳目录
│   ├── experiment_summary.json # 汇总结果
│   ├── hca2c_load3_seed42_results.json
│   ├── hca2c_load3_seed42_model.pt
│   ├── a2c_load3_seed42_results.json
│   ├── a2c_load3_seed42_model.zip
│   └── ...
```

## 预期结果

### HCA2C vs A2C 预期性能

| 负载 | HCA2C预期 | A2C基准 | 预期提升 |
|------|-----------|---------|----------|
| 3× | 高奖励, 0%崩溃 | 高奖励, 0%崩溃 | 相近 |
| 5× | 中等奖励, <10%崩溃 | 中等奖励, 10-20%崩溃 | +10-20% |
| 10× | 低奖励, <30%崩溃 | 极低奖励, 80-100%崩溃 | 显著优势 |

### 关键指标

- **Mean Reward**: 越高越好
- **Crash Rate**: 越低越好 (0%最佳)
- **Training Time**: HCA2C约为A2C的3-4倍

## 注意事项

1. HCA2C参数量是A2C的35倍，需要更长训练时间
2. 在极端负载(10×)下，HCA2C的优势更明显
3. 如果GPU内存不足，可减少batch_size或网络大小

## 问题排查

### 如果遇到NaN错误
代码已包含NaN处理，但如果仍有问题：
```python
# 在hca2c_agent.py中增加
torch.autograd.set_detect_anomaly(True)
```

### 如果内存不足
修改 `hca2c_agent.py` 中的 `n_steps` 参数：
```python
agent = HCA2C(env=env, n_steps=16)  # 默认64，可减小
```

## 联系

如有问题，请检查 `experiment_summary.json` 中的错误信息。
