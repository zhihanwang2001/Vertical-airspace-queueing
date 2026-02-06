# A1 Literature Analysis: TD7/SALE State-Action Representation Learning

**Full Citation**: S. Fujimoto, W.-D. Chang, E. J. Smith, S. Gu, D. Precup, and D. Meger, "For SALE: State-Action Representation Learning for Deep Reinforcement Learning," in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2023.

---

## 📄 Algorithm Basic Information

* **Algorithm Name**: TD7 (TD3 + SALE + Checkpoints + LAP + (optional) BC)
* **Publication Venue**: NeurIPS 2023 (37th Conference on Neural Information Processing Systems)
* **Year**: 2023
* **Algorithm Type**: **Actor-Critic** (deterministic policy gradient based on TD3, with behavior cloning term in offline scenarios)

## 🧠 Core Algorithm Innovation Analysis

1. **Algorithm Architecture**

   * **Base Framework**: Built on **TD3** with four key enhancements:
     **SALE** (State-Action Learned Embeddings), **Policy Checkpoints**, **LAP** (Loss-Adjusted Prioritized replay), **BC** (enabled in offline settings). Section 5 and Algorithm 1 (page 8) provide the combination and training procedure.
   * **Main Improvements**:

     * **SALE**: Jointly learns state embedding (z_s=f(s)) and state-action embedding (z_{sa}=g(z_s,a)), using **next state embedding** as learning target, and **only concatenates embeddings to Q and π inputs**, without world model planning (Section 4.1, Equations (1)-(3); Figure 1, page 4).
     * **Stability**: Proposes **AvgL1Norm** normalization, **fixed previous round embeddings** participating in current updates (avoiding input drift), and **target value clipping** to suppress **extrapolation error** in online environments (Section 4, Equations (4), (6)-(9), Figure 2).
     * **Checkpoints**: During training, saves the "most robust" policy based on **evaluation window minimum**, using checkpoint policy at test time (Section 5.1).
   * **Computational Complexity**: On HalfCheetah, 1M steps takes **approximately 1h50m**, increased compared to TD3 (about 47min), but still lower than some heavier methods (Figure 6, page 10).

2. **Key Technical Features**

   * **Action Space**: Continuous control (MuJoCo benchmarks); in offline settings, adds BC regularization to policy loss (Equation (11)).
   * **Observation Processing**: Focuses on **low-dimensional state** representation learning (not images), significantly improving sample efficiency through **state-action joint embeddings** (Sections 1, 4).
   * **Stability Mechanisms**: **Double Q minimization** (TD3), **target network**, **AvgL1Norm**, **target value clipping**, **prioritized replay** (LAP), **policy checkpoints**.

## 🔬 Technical Method Details

1. **Problem Modeling**: Standard MDP; learns (Q(s,a)), (π(s)), introducing (z_s, z_{sa}) as **additional features**. SALE approximates **next state embedding** by minimizing (|z_{sa}-|z_{s'}|_×|^2) (Equation (2)); **embedding and value/policy training are decoupled**.
2. **Theoretical/Empirical Framework**: Large-scale design space ablation (Section 4.2 and Figure 3, page 6), showing **Q input using z_s+z_{sa}+original(s,a)** is optimal; normalization using AvgL1Norm is most stable.
3. **Algorithm Framework**: **TD7 training procedure** in Algorithm 1 (page 8); online and offline versions differ only in BC weight.
4. **Key Techniques** (3-5 points)

   * **SALE joint embedding** (Equations (1)-(3))
   * **AvgL1Norm** normalization (Equation (4))
   * **Fixed embeddings** (Equations (6)-(8))
   * **Value function target clipping** to suppress extrapolation (Equation (9))
   * **Policy checkpoints** (Section 5.1)
5. **System Design Details**: Q/π inputs are (Q(z_{sa},z_s,s,a)), (π(z_s,s)) respectively; offline policy loss adds (λ)(π(s)−a)^2 (Equation (11)).

## 📊 Experimental Results and Performance

* **Benchmark Comparison**: On MuJoCo, compared with TD3, SAC, TQC, TD3+OFE (Figure 4 & Table 1, page 9); D4RL offline comparison with CQL, TD3+BC, IQL, X-QL (Table 2, page 10).
* **Performance Improvement**: Paper reports **significant superiority at 300k/5M steps**, e.g., **HalfCheetah** reaches **15031** at 300k, **18165** at 5M (Table 1), offline total score **784.4** is best (Table 2). Abstract also provides average improvement over TD3 (+276.7%@300k, +50.7%@5M).
* **Ablation**: Figure 5 (page 10) shows contribution order: **SALE > LAP > Checkpoints**.
* **System Scale/Complexity**: Full suite of MuJoCo continuous control tasks; single GPU, runtime curves in Figure 39 (Appendix I).
* **Limitations**: Mainly **lack of theoretical analysis** and **computational overhead doubled compared to TD3** (Appendix J), with default continuous action setting.

## 🔄 Technical Adaptability to Our System

**Our System Features**: 29-dimensional observations; **hybrid actions** (continuous service control + discrete emergency transfer); multi-objective (6 objectives); UAV scheduling real-time requirements.

### Adaptability Scores

1. **High-dimensional Observation Processing Capability**: **8/10** (excels at learning representations from low-dimensional states, SALE directly enhances Q/π inputs).
2. **Hybrid Action Space Support**: **5/10** (native continuous; requires extension for discrete head or gating/option policies).
3. **Multi-objective Optimization Capability**: **6/10** (native single-objective; can be extended through weighting/constraint formulation).
4. **Training Stability**: **9/10** (extrapolation clipping + prioritized replay + checkpoints significantly robust across multiple benchmarks).
5. **Real-time Inference Speed**: **8/10** (forward pass is small MLP + two encoders; authors report moderate runtime cost).
6. **Sample Efficiency**: **9/10** (300k steps significantly outperforms strong baselines).

### Technical Improvement Suggestions

1. **Observation Space Encoding**

   * Directly reuse **SALE**: feed 29-dimensional state into (f,g) to learn (z_s,z_{sa}).
   * If "layer-layer" structure exists (e.g., 5-layer airspace/queues), concatenate layer indices and cross-layer statistics to (s,a) before feeding into g, enabling embeddings to **explicitly capture inter-layer interactions**.

2. **Action Space Design (Hybrid)**

   * Add **discrete branch** (K types of emergency transfers) on top of existing continuous actor, sharing (z_s) representation: continuous branch uses DPG, discrete branch uses Gumbel-Softmax/REINFORCE + baseline; jointly optimize both branches with weighted total loss.
   * Or implement **gating policy**: first use discrete gate to decide whether to trigger "emergency transfer option", if triggered, use dedicated continuous head to output rate.

3. **Reward Function**

   * Adopt **stability-efficiency-fairness** decomposition: main objective is throughput/latency, supplemented by **Gini coefficient** load balancing, out-of-bounds/instability penalties; **combined with value clipping** to avoid divergence from short-term extremes.

4. **Network Architecture**

   * Follow paper's **fixed embeddings** and **AvgL1Norm**; retain **LAP** and **Checkpoints** (minimum value criterion, early evaluation=1, later=20).

## 🏆 Algorithm Integration Value

1. **Benchmark Comparison Value**: As a **strong baseline** (both online/offline), can demonstrate the upper limit of continuous control in our system.
2. **Technical Reference Value**: **SALE + target clipping + Checkpoints** trio significantly improves stability/sample efficiency.
3. **Performance Expectation**: Continuous scheduling part expected to outperform TD3/SAC; overall limited by hybrid action adaptation quality.
4. **Experimental Design**

   * **Comparison Groups**: TD3, SAC, PPO, TD7 (with/without SALE), "TD7 + hybrid action extension".
   * **Ablation**: Remove Checkpoints / Remove clipping / Don't use AvgL1Norm / Only use z_s or only use z_{sa}.
   * **Metrics**: Pareto frontier of 6 objectives + comprehensive scalar; stability (evaluation window minimum) and convergence speed.

**Algorithm Applicability Score**: **8/10**
**Integration Priority**: **High** (first implement continuous sub-problem + extend hybrid actions)

---

## 📋 Core Points Summary (for easy reference)

1. **SALE Definition and Training Objective**: Learn (z_s=f(s), z_{sa}=g(z_s,a)), supervised by **next state embedding**, embeddings only used as Q/π input enhancement (Section 4.1, Equations (1)-(3), Figure 1, pages 3–4).
2. **Extrapolation Error and Target Clipping**: In online RL, extrapolation error occurs due to action-related dimension explosion; **clipping targets with data min/max values** stabilizes training (Equation (9), Figure 2, page 5).
3. **Design Space Conclusions**: Q input must include **(z_{sa}, z_s, s, a)**; **AvgL1Norm** most stable; **end-to-end coupling** significantly worse than decoupling (Figure 3, page 6).
4. **TD7 Components and Training Procedure**: TD3 + **SALE** + **Checkpoints** + **LAP** + (offline) **BC**; Algorithm 1 provides training pseudocode (page 8).
5. **SOTA Results**: Online Table 1 (page 9) and offline Table 2 (page 10) show TD7 significantly leads at 300k/5M, D4RL; Figure 5 ablation shows **SALE** contributes most.

---

**Conclusion**: TD7/SALE provides a plug-and-play **representation learning and stability toolkit** for our "**low-dimensional queue state + continuous control-focused**" UAV queuing/scheduling problem. Engineering-level extensions needed for "**hybrid actions**" and "**multi-objectives**", but can proceed with implementation and experiments without changing core ideas (SALE + clipping + Checkpoints).

---

**Analysis Completion Date**: 2025-01-28
**Analysis Quality**: Detailed analysis with specific technical improvement suggestions and integration value assessment
**Recommended Use**: As core algorithm baseline for continuous control, focus on SALE representation learning and stability mechanisms