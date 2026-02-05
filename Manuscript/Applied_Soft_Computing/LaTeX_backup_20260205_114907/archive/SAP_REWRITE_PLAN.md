# SAP Manuscript精简计划

## 当前状态
- **当前页数**: 90页
- **SAP要求**: 20-50页（最大50页）
- **需要减少**: 40页

## 精简策略

### 1. 将Section 6-7改为Appendix（立即减少约15-20页正文）
- Section 6: Load Sensitivity Analysis → Appendix A
- Section 7: Structural Comparison Generalization → Appendix B
- 在正文Results中简要引用appendix结果

### 2. 精简算法伪代码（减少15页）
**当前**: 18个算法伪代码块
**目标**: 保留3个代表性算法
- 保留：A2C（最佳性能）、PPO（经典算法）、TD3（代表off-policy）
- 移除：其余15个算法（或移至supplementary material）
- 用简表总结算法特征

### 3. 精简数学公式（减少5页）
**当前**: 25个独立公式
**目标**: 保留15个核心公式
- 保留：MDP formulation, Bellman equation, reward function核心公式
- 合并：相似的queue dynamics公式
- 简化：推导过程，保留结果

### 4. 优化图表（减少2-3页）
**当前**: 11个图表
**目标**: 保留7-8个最关键图表
- 必保留：
  - system_architecture.pdf
  - fig1_capacity (capacity paradox)
  - fig2_structure (structural comparison)
  - fig4_algorithm_radar (algorithm performance)
- 可合并/移除：
  - 3个structural细节图可合并为1个
  - 2个capacity细节图可合并为1个

### 5. 精简方法论（减少3-5页）
- 简化系统动力学详细描述
- 精简state/action space展开
- 保留核心MDP formulation

### 6. 精简结果讨论（减少3-5页）
- 聚焦3大核心发现：
  1. DRL vs heuristics（59.9% improvement）
  2. Structural advantage（9.7%-19.7%）
  3. Capacity paradox
- 次要发现简述，引用appendix

## 预期结果
- Section 6-7 → Appendix: -18页
- 算法伪代码精简: -15页
- 数学公式精简: -5页
- 图表优化: -3页
- 方法论精简: -4页
- 结果精简: -4页
**总计减少**: ~49页
**预期页数**: 41页 ✅（符合20-50页要求）

## 实施步骤
1. 创建新的manuscript_SAP.tex
2. 复制基础结构（frontmatter, introduction）
3. 精简Methodology section
4. 精简Results section
5. 将Section 6-7转换为Appendix
6. 创建supplementary_algorithms.pdf（包含15个算法详细伪代码）
7. 编译并验证页数

## SAP其他要求清单
- ✅ Highlights (3-5条, ≤85字符)
- ✅ Graphical Abstract (1644×3884 pixels)
- ⏳ Author biographies
- ⏳ CRediT contributions
- ⏳ Data availability statement
- ⏳ Funding statement
- ⏳ Competing interests declaration
