# RP1: Single-Objective Deep Reinforcement Learning for Vertical Layered Queueing Systems

## 📋 Project Overview

This directory contains the complete archive of **Research Point 1 (RP1)** from the PostGraduate research project on Urban Air Mobility (UAM) drone delivery optimization using deep reinforcement learning.

**Research Question**: "Which deep reinforcement learning algorithm is most suitable for vertical layered queueing systems?"

**Key Findings**:
- **Best Algorithm**: A2C with delayed cosine annealing (4437.86 reward, 6.9 min training)
- **Runner-up**: PPO (4419.98 reward, 30.8 min)
- **Notable**: TD7 with "double-jump learning" phenomenon (+857% and +95% performance jumps)
- **Overall**: DRL algorithms achieve 50%+ improvement over heuristic baselines

## 📂 Directory Structure

```
final/RP1/
├── Code/                    # Source code (1.7 MB)
│   ├── env/                # Environment implementation (~6,640 lines)
│   ├── algorithms/         # Algorithm implementations
│   │   ├── baselines/     # Stable-Baselines3 wrappers (A2C, PPO, SAC, TD3, DDPG)
│   │   └── advanced/      # Custom implementations (TD7, R2D2, Rainbow, IMPALA, SAC-v2)
│   ├── training_scripts/  # Training and testing scripts (12 files)
│   └── analysis_scripts/  # Visualization and analysis (23 files)
│
├── Models/                  # Trained models (1.2 GB)
│   ├── a2c/               # A2C model checkpoints
│   ├── ppo/               # PPO model checkpoints
│   ├── td7/               # TD7 model checkpoints
│   ├── sb3_a2c_best/      # Best A2C model
│   ├── sb3_ppo_best/      # Best PPO model
│   ├── *_final.pt         # Final model weights for custom algorithms
│   └── top3_training_summary.json
│
├── Results/                 # Experimental results (3.9 MB)
│   ├── comparison/        # Algorithm comparison data (JSON, TXT)
│   ├── generalization/    # Cross-region generalization tests (CSV, JSON)
│   └── excel/             # Excel format results
│
├── Figures/                 # Visualization results (10 MB)
│   ├── publication/       # Publication-quality figures (9 figures, 300 DPI)
│   └── analysis/          # Analysis figures (6 figures)
│
└── Documentation/           # Documentation and references (132 MB)
    ├── guides/            # 15 markdown guides and READMEs
    └── references/        # 60+ academic papers with analysis notes
```

## 🔬 Core Components

### 1. Environment Implementation (`Code/env/`)

**MCRPS/D/K Queueing Framework** - A novel queueing model for vertical airspace:
- **MC**: Multi-Class Correlated arrivals
- **R**: Random batch service
- **P**: Poisson splitting
- **S**: State-dependent control
- **D**: Dynamic inter-layer transfers
- **K**: Finite capacity (inverted pyramid: [8, 6, 4, 3, 2])

**Key Files**:
- `vertical_queue_env.py` - Main Gymnasium environment
- `delivery_cabinet.py` - Queue dynamics and routing logic
- `reward_function.py` - Multi-objective reward design
- `state_manager.py` - State representation (29-dim)
- `queue_dynamics.py` - Queueing theory implementation
- `config.py` - System configuration
- `README.md` - Complete environment specification (440 lines)

**Environment Specs**:
- **State Space**: 29 dimensions (queue lengths, service rates, transfers, etc.)
- **Action Space**: 11 dimensions (6 continuous + 5 discrete) - Hybrid action space
- **Reward Components**: Throughput, fairness (Gini), efficiency, stability, queue management

### 2. Algorithm Implementations (`Code/algorithms/`)

#### Baselines (Stable-Baselines3)
- `sb3_a2c_baseline.py` - Advantage Actor-Critic ✅ **BEST**
- `sb3_ppo_baseline.py` - Proximal Policy Optimization ✅ **RUNNER-UP**
- `sb3_sac_baseline.py` - Soft Actor-Critic
- `sb3_td3_baseline.py` - Twin Delayed DDPG
- `sb3_ddpg_baseline.py` - Deep Deterministic Policy Gradient
- `heuristic_baseline.py` - FCFS, Priority, SJF heuristics
- `comparison_runner.py` - Orchestration for all experiments

#### Advanced Algorithms (Custom Implementations)
- `td7/` - Temporal Difference Learning with 7 enhancements ✅ **INTERESTING**
- `r2d2/` - Recurrent Replay Distributed DQN
- `rainbow_dqn/` - Rainbow DQN (distributional RL + prioritized replay)
- `impala/` - Importance Weighted Actor-Learner Architecture
- `sac_v2/` - SAC version 2 with automatic entropy tuning

### 3. Training Scripts (`Code/training_scripts/`)

**Main Training Scripts**:
- `run_baseline_comparison.py` - Run all SB3 baselines
- `run_advanced_algorithm_comparison.py` - Run custom algorithms
- `run_ablation_experiments.py` - Ablation studies
- `train_top3_models.py` - Train top 3 algorithms (A2C, PPO, TD7)

**Testing Scripts**:
- `test_all_models_generalization_v3.py` - Cross-region generalization tests
- `test_td7_generalization.py` - TD7-specific generalization
- `test_sb3_save.py` - Model saving verification

**Total**: 12 training/testing scripts

### 4. Analysis Scripts (`Code/analysis_scripts/`)

**Figure Generation**:
- `generate_paper_figures.py` - All publication figures
- `generate_ccf_professional_figures.py` - CCF-style figures
- `generate_framework_architecture.py` - System architecture diagram
- `plot_beautiful_*.py` - Individual high-quality plots

**Analysis**:
- `pareto_analysis_final.py` - Multi-objective trade-off analysis
- `analyze_ablation_results.py` - Ablation study analysis

**Total**: 23 analysis/visualization scripts

### 5. Models (`Models/`)

**Pre-trained Models (1.2 GB)**:
- **15 trained models** from 500K training timesteps
- **Best performing**:
  - `sb3_a2c_best/` - A2C final model (Reward: 4437.86)
  - `sb3_ppo_best/` - PPO final model (Reward: 4419.98)
  - `td7/` - TD7 with jump learning
- **Training checkpoints**: Periodic snapshots for analysis
- **Custom algorithm weights**: `*.pt` files for PyTorch models

**Model Summary**: `top3_training_summary.json`

### 6. Results (`Results/`)

#### Comparison Results (`comparison/`)
- `comparison_data.json` - Complete comparison of all 15 algorithms
- `comparison_report.txt` - Statistical analysis report
- Individual algorithm histories (`*_history.json`)
- Model artifacts for heuristic baselines

**Key Metrics**:
- Final episode reward
- Training time
- Sample efficiency
- Stability (std dev)

#### Generalization Results (`generalization/`)
- `all_models_generalization_results_v3.json` - Complete test results
- `all_models_generalization_summary_v3.csv` - Summary table
- Cross-region performance data (5 regions: A, B, C, D, E)

**Region Characteristics**:
- **Region A**: Standard conditions (baseline)
- **Region B**: Weather disruption (-20% service rate)
- **Region C**: High traffic (+50% arrival rate) → +36.2% reward boost
- **Region D**: Strict regulation (-22% capacity)
- **Region E**: Energy constraints (-25% high-layer service)

#### Excel Results (`excel/`)
- Formatted results for publication tables
- Statistical analysis outputs

### 7. Figures (`Figures/`)

#### Publication Figures (`publication/` - 9 figures)
1. `figure1_3d_structure.png` - 3D visualization of inverted pyramid structure
2. `figure1_top_tier_comparison.png` - Top-tier algorithm comparison
3. `figure2_architecture.png` - System architecture diagram
4. `figure2_learning_curves.png` - Training curves for all algorithms
5. `figure3_performance_ranking.png` - Performance ranking bar chart
6. `figure4_td7_jump_learning.png` - TD7 double-jump phenomenon
7. `figure4_td7_jump_learning_zoom.png` - Zoomed view of jumps
8. `figure5_radar_chart.png` - Multi-dimensional performance comparison
9. `figure6_efficiency_analysis.png` - Sample efficiency analysis

**All figures**: 300 DPI, publication-ready

#### Analysis Figures (`analysis/` - 6 figures)
- A2C detailed training curves
- Optimization curves
- Pareto front 2D/3D visualizations
- Trade-off conflict analysis

### 8. Documentation (`Documentation/`)

#### Guides (`guides/` - 15 documents)
- `FINAL_TRANSITION_NARRATIVE.md` - Complete RP1 research story
- `Final_Paper_Chinese_Version.md` - Chinese manuscript (98 KB)
- `Related_Work.md` - Literature review (60+ papers)
- `RP1_完整研究报告.md` - Complete research report (Chinese)
- `GENERALIZATION_TEST_README.md` - Generalization experiment guide
- `V3_RESULTS_ANALYSIS.md` - Final results analysis
- `QUICK_REFERENCE_FINAL.md` - Quick reference guide
- Multiple implementation and design documents

#### References (`references/` - 60+ papers)

**Algorithm Papers** (with analysis notes):
- `A1_TD7.pdf` + `A1_TD7_SALE_Analysis.md`
- `A2_RainbowDQN.pdf` + `A2_Rainbow_DQN_Analysis.md`
- `A3_IMPALA.pdf` + `A3_IMPALA_Analysis.md`
- `A4_R2D2.pdf` + `A4_R2D2_Analysis.md`

**Application Papers**:
- Food delivery optimization
- Urban air mobility systems
- Queueing theory applications

**Total**: 132 MB of documentation

## 🎯 Key Research Contributions

### 1. Theoretical Innovation
- **MCRPS/D/K Framework**: First queueing model for vertical airspace
- **Inverted Pyramid Design**: Capacity structure [8,6,4,3,2] reflecting physical constraints
- **Pressure-Triggered Routing**: Dynamic inter-layer transfers

### 2. Algorithmic Discoveries
- **A2C Superiority**: Best balance of performance (4437.86) and speed (6.9 min)
- **TD7 Jump Learning**: Identified rare "double-jump" learning pattern
- **DRL Advantage**: 50%+ improvement over heuristic methods
- **Conservative Strategy Problem**: Queue utilization <1% in optimal policies

### 3. Cross-Scenario Generalization
- Tested on 5 heterogeneous regions
- Region C (high traffic) shows +36.2% reward improvement
- Strong generalization despite distribution shift

## 📊 Experimental Results Summary

| Algorithm | Final Reward | Training Time | Sample Efficiency | Rank |
|-----------|-------------|---------------|-------------------|------|
| A2C | 4437.86 | 6.9 min | High | 🥇 1st |
| PPO | 4419.98 | 30.8 min | Medium | 🥈 2nd |
| TD7 | 4324.12 | 382 min | Very High | 🥉 3rd |
| SAC | 4156.23 | 48.2 min | Medium | 4th |
| TD3 | 3987.45 | 52.1 min | Medium | 5th |
| R2D2 | 3654.87 | 156 min | Low | 6th |
| Rainbow | 3542.19 | 124 min | Low | 7th |
| IMPALA | 3234.56 | 89 min | Low | 8th |
| DDPG | 2987.34 | 41.5 min | Low | 9th |
| Heuristic | <2000 | N/A | N/A | Baseline |

**Training Configuration**: 500K timesteps per algorithm

## 🚀 Quick Start

### 1. Environment Setup
```bash
cd Code/env
python -c "from vertical_queue_env import VerticalQueueEnv; env = VerticalQueueEnv(); print('Environment loaded successfully!')"
```

### 2. Load Pre-trained Model
```python
from stable_baselines3 import A2C
model = A2C.load("../../Models/sb3_a2c_best/final_model")
```

### 3. Run Generalization Test
```bash
cd Code/training_scripts
python test_all_models_generalization_v3.py
```

### 4. Generate Figures
```bash
cd Code/analysis_scripts
python generate_paper_figures.py
```

## 📖 Usage Guidelines

### For Researchers
1. **Reproduce Results**: Use `Code/training_scripts/run_baseline_comparison.py`
2. **Analyze Data**: Results in `Results/comparison/comparison_data.json`
3. **Visualize**: Run scripts in `Code/analysis_scripts/`
4. **Read Documentation**: Start with `Documentation/guides/RP1_完整研究报告.md`

### For Practitioners
1. **Use Best Model**: `Models/sb3_a2c_best/final_model`
2. **Adapt Environment**: Modify `Code/env/config.py`
3. **Test Generalization**: Run `test_all_models_generalization_v3.py`

### For Paper Submission
1. **Figures**: Use `Figures/publication/` (300 DPI)
2. **Results**: Use `Results/comparison/comparison_report.txt`
3. **References**: Cite papers in `Documentation/references/`

## 🔧 Technical Details

### Dependencies
- Python 3.10+
- PyTorch ≥2.0.0
- Stable-Baselines3 ≥2.0.0
- Gymnasium ≥0.29.0
- NumPy ≥1.24.0
- Matplotlib ≥3.7.0

### Hardware Requirements
- **Training**: GPU recommended (A2C: 6.9 min on GPU, TD7: 382 min)
- **Inference**: CPU sufficient
- **Storage**: 1.3 GB for complete archive

### Code Statistics
- **Total Python Code**: ~6,640 lines in environment
- **Total Files**: ~230 files
- **Algorithms Implemented**: 15 (10 custom + 5 baseline)
- **Trained Models**: 15 models (500K timesteps each)

## 📝 Citation

If you use this work, please cite:

```bibtex
@mastersthesis{rp1_uam_drl,
  title={Single-Objective Deep Reinforcement Learning for Vertical Layered Queueing Systems in Urban Air Mobility},
  author={[Your Name]},
  year={2025},
  school={[Your University]},
  note={Research Point 1 of PostGraduate Thesis}
}
```

## 📄 License

This research archive is provided for academic and research purposes.

## 📧 Contact

For questions about this research, please refer to the documentation in `Documentation/guides/` or contact [your email].

---

**Archive Created**: January 2, 2026
**Research Completion**: December 2025
**Status**: Publication-Ready ✅
# Vertical-airspace-queueing
