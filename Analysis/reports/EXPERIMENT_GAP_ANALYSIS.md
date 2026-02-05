# 实验完整性分析报告

**分析日期**: 2026年1月17日
**目的**: 识别实验gaps，确保manuscript的完整性

---

## 第一部分：现有实验数据盘点

### 1.1 已完成的实验

#### Experiment 1: Algorithm Comparison (15 algorithms)
- **配置**: Default inverted pyramid [8,6,4,3,2], baseline load
- **算法**: A2C, PPO, TD7, SAC, TD3, R2D2, Rainbow, IMPALA, DDPG + 4 heuristics
- **训练**: 500K timesteps, 5 seeds
- **状态**: ✅ 完成
- **数据位置**: Results/comparison/

#### Experiment 2: Structural Comparison (Inverted vs Normal)
- **配置**: 5× load, K=23 total
- **结构**: Inverted [8,6,4,3,2] vs Normal [2,3,4,6,8]
- **算法**: A2C + PPO
- **样本**: n=30 per algorithm per structure (total n=60 per structure)
- **状态**: ✅ 完成
- **关键结果**: Inverted优于Normal 9.5% (p<1e-134, d=48.452)

#### Experiment 3: Capacity Scan at 5× Load
- **配置**: K∈{10, 30}, load=5×, uniform shape
- **算法**: A2C, PPO + 4 heuristics
- **种子**: n=5
- **状态**: ✅ 完成
- **关键发现**:
  - K=10: 352K reward, 0% crash
  - K=30: 737K reward, 12-17% crash
  - **注意**: K=30在5×负载下实际表现更好！

#### Experiment 4: Capacity Scan at 6× Load (正在运行)
- **配置**: K∈{10, 30}, load=6×,7×, uniform shape
- **算法**: A2C, PPO + 4 heuristics
- **种子**: n=5
- **状态**: 🔄 运行中（启动于07:58，预计8-12小时完成）
- **日志**: logs/capacity_uniform_6_7_5s_100000t_50e.log

#### Experiment 5: Capacity Scan - Reverse Pyramid at 6× Load
- **配置**: K∈{10, 30}, load=6×, reverse pyramid [2,3,4,6,8]
- **算法**: A2C, PPO + 4 heuristics
- **种子**: n=5
- **状态**: ✅ 完成
- **数据**: capacity_scan_summary_reverse_6.csv
- **关键发现**:
  - K=10: 353K reward, 0% crash
  - K=30: 207K reward (A2C), 96-98% crash rate
  - **重要**: 在6×负载下，reverse pyramid已经出现capacity paradox迹象！

---

## 第二部分：关键发现与问题识别

### 2.1 Capacity Paradox的负载依赖性

**观察到的现象**:
- **5× load**: K=30 > K=10 (737K vs 352K, uniform)
- **6× load**: K=10 > K=30 (353K vs 207K, reverse pyramid)
- **10× load** (Results outline中提到): K=10 >> K=30 (11,180 vs 13)

**关键问题**: Capacity paradox在哪个负载水平开始出现？

**影响**: 这是论文的核心发现之一，需要清晰的数据支撑！

### 2.2 EJOR评审意见中的要求

EJOR评审报告（Major Comment 2）明确指出：
> "Test across multiple load levels (3x, 5x, 7x, not just 10x)"

**当前状态**:
- ✅ 5× load: 有数据
- 🔄 6×, 7× load: 正在运行
- ❌ 3×, 4×, 8×, 9×, 10× load: **缺失**

---

## 第三部分：关键实验Gaps识别

### Gap 1: 负载敏感性分析（Load Sensitivity） ⚠️ **最高优先级**

**缺失内容**:
- 系统的负载扫描：1×, 2×, 3×, 4×, 8×, 9×, 10× load
- 特别是10× load的数据（Results outline中提到但未找到数据文件）

**为什么重要**:
1. EJOR评审明确要求
2. 确定capacity paradox的临界负载点
3. 论文核心发现的关键支撑

**影响范围**: Results Section 5.3 (Capacity Paradox)

**建议实验**:
```
配置: K∈{10, 30}, load∈{3, 4, 8, 9, 10}, uniform shape
算法: A2C, PPO (top performers)
种子: n=5
训练: 100K timesteps (与现有capacity scan一致)
预计时间: 5 loads × 2 capacities × 2 algos × 5 seeds = 100 runs × ~7min = ~12小时
```

### Gap 2: 结构比较的负载泛化性 ⚠️ **高优先级**

**缺失内容**:
- Inverted vs Normal pyramid在不同负载下的比较
- 当前只有5× load的数据

**为什么重要**:
1. 验证9.5%优势是否在所有负载下都成立
2. 回应EJOR评审关于泛化性的质疑

**影响范围**: Results Section 5.2 (Structural Analysis)

**建议实验**:
```
配置: Inverted [8,6,4,3,2] vs Normal [2,3,4,6,8], load∈{3, 7, 10}
算法: A2C, PPO
种子: n=5 per algorithm per structure
训练: 100K timesteps
预计时间: 3 loads × 2 structures × 2 algos × 5 seeds = 60 runs × ~7min = ~7小时
```

### Gap 3: Capacity Paradox的训练时长验证 ⚠️ **中优先级**

**缺失内容**:
- 用更长训练时间（500K）测试K=30在10× load下是否仍然失败

**为什么重要**:
1. 排除"训练不足"假说
2. 证明capacity paradox是系统属性而非训练artifact

**影响范围**: Results Section 5.3.3 (Theoretical Explanation)

**建议实验**:
```
配置: K∈{10, 30}, load=10, uniform shape
算法: A2C, PPO
种子: n=5
训练: 500K timesteps (5倍于当前)
预计时间: 2 capacities × 2 algos × 5 seeds = 20 runs × ~35min = ~12小时
```

---

## 第四部分：优先级总结与行动建议

### 4.1 必须补充的实验（Applied Soft Computing投稿前）

#### ✅ Priority 1: 负载敏感性分析（Gap 1）
**理由**:
- EJOR评审明确要求
- 论文核心发现的关键支撑
- 确定capacity paradox的临界点

**行动**: 运行load∈{3, 4, 8, 9, 10}的capacity scan
**时间**: ~12小时
**紧急程度**: ⚠️ **必须完成**

#### ✅ Priority 2: 结构比较的负载泛化（Gap 2）
**理由**:
- 验证9.5%优势的普适性
- 增强Results Section 5.2的说服力

**行动**: 在load∈{3, 7, 10}下测试Inverted vs Normal
**时间**: ~7小时
**紧急程度**: ⚠️ **强烈建议**

### 4.2 可选的补充实验（增强论文质量）

#### ⭕ Priority 3: 训练时长验证（Gap 3）
**理由**: 排除"训练不足"假说
**行动**: 用500K timesteps重新测试K=30 at 10× load
**时间**: ~12小时
**紧急程度**: 🔵 **建议但非必须**

---

## 第五部分：具体执行方案

### 5.1 Priority 1: 负载敏感性分析

**命令行**:
```bash
# 在服务器上运行
cd /root/RP1

# Load 3×, 4×
python3 Code/training_scripts/run_capacity_scan.py \
  --capacities 10,30 \
  --loads 3,4 \
  --shape uniform \
  --algos A2C,PPO \
  --n-seeds 5 \
  --timesteps 100000 \
  --eval-episodes 50 \
  > logs/capacity_uniform_3_4_5s_100000t_50e.log 2>&1 &

# Load 8×, 9×, 10×
python3 Code/training_scripts/run_capacity_scan.py \
  --capacities 10,30 \
  --loads 8,9,10 \
  --shape uniform \
  --algos A2C,PPO \
  --n-seeds 5 \
  --timesteps 100000 \
  --eval-episodes 50 \
  > logs/capacity_uniform_8_9_10_5s_100000t_50e.log 2>&1 &
```

**预计完成时间**: 2批次 × 6小时 = 12小时

### 5.2 Priority 2: 结构比较的负载泛化

**命令行**:
```bash
# Load 3×, 7×, 10× - Inverted vs Normal
python3 Code/training_scripts/run_structural_comparison_5x_load.py \
  --loads 3,7,10 \
  --algos A2C,PPO \
  --n-seeds 5 \
  --timesteps 100000 \
  > logs/structural_comparison_3_7_10.log 2>&1 &
```

**注意**: 需要修改`run_structural_comparison_5x_load.py`以支持多负载参数

**预计完成时间**: ~7小时

---

## 第六部分：时间线规划

### 方案A：保守方案（完成Priority 1+2）
- **Day 1**: 等待当前实验完成（6×,7× load）
- **Day 2**: 运行Priority 1实验（3×,4×,8×,9×,10× load）
- **Day 3**: 运行Priority 2实验（结构比较）
- **Day 4**: 数据分析和可视化
- **总时间**: 4天

### 方案B：激进方案（仅Priority 1）
- **Day 1**: 等待当前实验完成，立即启动Priority 1
- **Day 2**: 数据分析，开始撰写manuscript
- **总时间**: 2天

### 方案C：完美方案（Priority 1+2+3）
- **Day 1-2**: Priority 1+2
- **Day 3**: Priority 3（训练时长验证）
- **Day 4**: 数据分析
- **总时间**: 5天

---

## 第七部分：最终建议

### 推荐方案：**方案A（保守方案）**

**理由**:
1. ✅ 满足EJOR评审要求（多负载测试）
2. ✅ 验证核心发现的普适性（9.5%优势）
3. ✅ 提供完整的capacity paradox演化曲线
4. ⏱️ 时间可控（4天）
5. 📊 数据充分支撑Applied Soft Computing投稿

**不推荐Priority 3的原因**:
- 时间成本高（额外12小时）
- 对投稿不是必需的
- 可以在Discussion中作为future work提及

---

## 总结

**当前状态**: 实验基础扎实，但存在关键gaps
**必须补充**: Priority 1（负载敏感性）
**强烈建议**: Priority 2（结构泛化性）
**可选**: Priority 3（训练时长验证）

**行动建议**: 采用方案A，4天内完成所有必要实验，然后全力撰写manuscript。

**预计投稿时间**: 1月底（实验4天 + 撰写2-3周）

---

**报告完成时间**: 2026年1月17日
