# Related Work

This section reviews the current state-of-the-art in three key research areas relevant to our MCRPS/D/K (Multi-Class correlated arrivals, Random batch service, Poisson splitting, State-dependent, Dynamic transfer, finite capacity K) vertical stratified queueing framework: system architecture and load balancing, recent algorithm research, and queueing theory foundations. Our comprehensive literature review of 28 high-quality papers from recent years reveals fundamental gaps that our proposed framework addresses.

## 2.1 System Architecture and Load Balancing

### 2.1.1 Fairness-Aware Scheduling and Load Distribution

Recent advances in scheduling systems have increasingly focused on fairness metrics and intelligent load distribution mechanisms. Li et al. [1] propose a Theil index-based fairness framework for Power IoT networks, achieving 10-35% allocation improvement through within-group and between-group fairness decomposition with Lyapunov optimization. However, their approach operates on a two-tier architecture without considering vertical spatial stratification or capacity heterogeneity.

Chen et al. [2] address temporal fairness in weighted fair queueing (WFQ) systems through EMA burst detection and nanosecond-level processing, demonstrating the importance of real-time fairness mechanisms. Do et al. [3] extend fairness concepts through generalized Gini index optimization for ranking systems, employing Moreau smoothing and Frank-Wolfe algorithms to achieve Lorenz efficiency coverage.

While these works establish strong foundations for fairness-aware scheduling, they operate on horizontal resource allocation paradigms without considering the unique challenges of vertical airspace management. Our MCRPS/D/K framework extends these fairness concepts to vertical stratified systems through Gini coefficient-based load balancing across altitude layers with heterogeneous capacities.

### 2.1.2 UAV System Architecture and Coordination

The UAV research landscape demonstrates growing sophistication in multi-agent coordination and airspace management. For target assignment and path planning, existing approaches focus on geometric optimization without queueing-theoretic considerations. The TANet-TD3 framework [4] achieves simultaneous target assignment and continuous 3D path planning through Hungarian algorithm matching, but operates on unlimited airspace assumptions without capacity constraints.

Multi-UAV cooperative search systems [5] introduce hierarchical coordination through three-layer altitude cooperation (high-medium-low), employing AM-MAPPO with action masking for collision avoidance. This work provides early evidence for the benefits of altitude stratification, though limited to three layers without capacity modeling or queueing analysis.

Communication-constrained UAV systems [6] address practical deployment challenges through MADDPG with explicit LSTM-enhanced messaging under communication radius and interference constraints. However, these constraints are modeled as geometric limitations rather than capacity-based queueing bottlenecks.

Recent work on data-driven drone delivery optimization [7] introduces sophisticated capacity management through learned link priorities and α-profile strategies for online demand, achieving 28-69% profit improvements over greedy baselines. Their surrogate ILP approach with β-weighted capacity reservation demonstrates the importance of intelligent capacity management, though applied to horizontal road networks rather than vertical airspace.

### 2.1.3 Advanced Airspace Management Systems

Low-altitude airspace management (LAAM) research [8] provides comprehensive frameworks for urban UAV traffic management through UTM and U-space paradigms, emphasizing service volume partitioning and dynamic capacity allocation. However, these approaches focus on regulatory and service-level management without algorithmic optimization for real-time scheduling.

Airspace network design research [9] explores 3D corridor construction through road projection to airways with BPR congestion functions and CSO traffic assignment. While providing 3D spatial modeling, these methods rely on static network planning without real-time adaptive optimization capabilities.

Hybrid AI approaches for 4D trajectory management [10] combine genetic algorithms, K-means clustering, and 3D A* search for demand-capacity balancing, achieving 99.74% overload resolution. However, their semi-real-time processing (30-40 minute intervals) limits applicability to dynamic environments requiring immediate response.

Cloud-based load balancing frameworks [11] introduce five-stage optimization with pheromone-based overload suppression, achieving 98% packet delivery ratios through triple-trigger mechanisms. These works provide valuable load distribution insights but lack spatial stratification and queueing-theoretic foundations.

**Research Gap Identification**: Existing system architectures either focus on horizontal resource allocation (fairness-aware scheduling) or geometric coordination (UAV systems) without integrating vertical spatial stratification with queueing-theoretic capacity management. No current work combines physical altitude constraints with queue stability guarantees and intelligent load balancing across stratified layers.

## 2.2 Recent Deep Reinforcement Learning Algorithm Research

### 2.2.1 State-of-the-Art Algorithm Development

The deep reinforcement learning landscape has witnessed significant advances in sample efficiency, stability, and scalability. TD7 with State-Action Learning Embeddings (SALE) [12] introduces joint learning of state embeddings z_s and state-action embeddings z_{sa}, combined with policy checkpoints and LAP prioritized replay, achieving substantial improvements over TD3 baselines. The algorithm demonstrates particular strength in continuous control tasks, though limited to pure continuous action spaces.

Rainbow DQN [13] represents a pinnacle of value-based methods, integrating six DQN improvements: Double Q-learning, prioritized experience replay, dueling networks, multi-step learning, distributional RL, and noisy networks. Achieving DQN's final performance in just 7M frames (vs 200M for vanilla DQN) and 231% normalized human performance on Atari, Rainbow establishes strong baselines for discrete action spaces but requires significant extension for mixed action environments.

Distributed learning approaches have advanced through IMPALA [14], which employs Actor-Learner architecture with V-trace off-policy correction, achieving 250K frames/s throughput across multiple environments. The architecture demonstrates excellent scalability for high-throughput applications, though requiring adaptation for hybrid action spaces.

Recurrent experience replay advances through R2D2 [15] incorporate LSTM memory with stored state mechanisms and burn-in procedures, achieving superhuman performance on 52 of 57 Atari games. The sequential decision-making capabilities provided by memory-enhanced learning show particular promise for complex dynamic environments.

### 2.2.2 Algorithm Integration Challenges

The chronological overview of DRL development [16] reveals increasing algorithm complexity and specialization, with recent trends toward hybrid architectures and domain-specific adaptations. However, survey analysis indicates a lack of queueing-aware state representations and mixed action space support in most state-of-the-art algorithms.

Learning-augmented queueing research [17] introduces prediction mechanisms into queueing systems through SOAP (Smoothed Online Algorithmic Predictor) analysis and Trail age-threshold policies, achieving 1.66-2.01× performance improvements. These approaches demonstrate the potential for ML-enhanced queueing, though focused on single-node prediction rather than multi-layer coordination.

**Research Gap Identification**: Current DRL algorithms excel in specific domains (continuous control, discrete decisions, distributed learning) but lack unified frameworks for hybrid action spaces required by multi-layer queueing systems. Additionally, most algorithms operate on domain-specific state representations rather than queueing-theoretic system states incorporating arrival rates, service rates, and capacity utilization metrics.

## 2.3 Queueing Theory and Network Analysis

### 2.3.1 Advanced Queueing Models and Analytical Methods

Modern queueing theory has evolved beyond classical M/M/1 models toward sophisticated network structures and correlated arrival processes. Blockchain queueing systems [18] employ Markovian arrival processes (MkAP) with matrix-geometric solutions for correlated arrivals in healthcare applications, providing analytical frameworks for GI/M/1 systems with dependencies. These methods establish foundations for analyzing correlated arrival processes but operate on unlimited capacity assumptions.

Multi-queue scheduling optimization [19] combines neural networks with discrete event simulation for parallel queue management, employing MDP policy iteration for load distribution. The hybrid analytical-simulation approach achieves significant performance improvements while maintaining theoretical rigor, though limited to parallel rather than hierarchical queue structures.

Specialized queueing models include orbit queues with repeated service attempts [20], analyzed through supplementary variable techniques and Laplace-Stieltjes transforms. Container transportation networks [21] employ closed Jackson network models with Mean Value Analysis (MVA) for optimization, demonstrating advanced network analysis capabilities for ground-based logistics.

### 2.3.2 State-Dependent and Network Queueing Systems

State-dependent queueing mechanisms have gained attention through battery swapping station analysis [22], which employs embedded Markov chain methods for dual-threshold control strategies. The work demonstrates sophisticated state-dependent service mechanisms but operates on dual-queue systems without multi-layer coordination.

Comprehensive queueing network surveys [23] review SOQN (Semi-Open Queueing Network) frameworks and decomposition methods for material handling systems, providing extensive network analysis foundations. However, these frameworks assume horizontal network topologies without vertical stratification considerations.

Hierarchical multi-agent system taxonomy [24] provides five-axis classification frameworks (control/information/role/time/communication) for complex system design, offering architectural guidance but lacking queueing-theoretic foundations.

Core allocation research [25] addresses multicore resource optimization with completion deadlines through KKT conditions and power-law speedup functions, providing resource allocation techniques applicable to capacity management but focused on deterministic batch processing rather than stochastic queueing systems.

### 2.3.3 Dynamic Scheduling and Real-Time Applications

Dynamic scheduling applications span diverse domains from food delivery [26] using LSTM encoder-decoder architectures for sequential order allocation to meal delivery optimization [27] employing event-driven DQN variants for Poisson arrival processes. Online DRL frameworks [28] achieve real-time order recommendation through attention-based state representation and behavior prediction, demonstrating the feasibility of real-time learning in dynamic environments.

These applications consistently operate on 2D spatial assumptions with horizontal resource coordination, lacking vertical stratification and capacity heterogeneity considerations essential for airspace management.

## 2.4 Identified Research Gaps and Theoretical Contributions

### 2.4.1 Fundamental Theoretical Gaps

Our comprehensive literature analysis reveals critical gaps in existing research:

1. **Absence of Vertical Spatial Queueing Theory**: No existing work addresses queueing systems with physical vertical stratification and altitude-dependent capacity constraints. Current queueing theory operates on abstract service networks without spatial physics integration.

2. **Lack of Inverted Pyramid Capacity Models**: Existing capacity allocation assumes uniform or increasing capacity with system hierarchy, contrary to physical airspace constraints where lower altitudes face tighter capacity limitations due to safety and coordination requirements.

3. **Missing Pressure-Triggered Dynamic Mechanisms**: While state-dependent queueing exists [22], no work implements congestion pressure-triggered inter-layer transfers with queue stability guarantees.

4. **Absence of MCRPS/D/K Network Types**: Current queueing literature lacks systems combining multi-class correlated arrivals, random batch service, Poisson splitting, state-dependent control, dynamic inter-node transfers, and finite capacity constraints in a unified framework.

### 2.4.2 Algorithmic Integration Gaps

1. **Hybrid Action Space Limitations**: State-of-the-art DRL algorithms [12-15] require significant adaptation for mixed continuous-discrete action spaces essential for queueing system control.

2. **Queueing-Unaware State Representation**: Current algorithms operate on domain-specific state representations rather than queueing-theoretic system states incorporating arrival rates, service patterns, and capacity metrics.

3. **Lack of Fairness-Performance Integration**: While fairness mechanisms exist [1, 3], no work integrates fairness objectives with queueing performance optimization in DRL frameworks.

### 2.4.3 Vertical Airspace Management Gaps

1. **Geometric vs. Queueing Paradigms**: UAV coordination research [4-6, 9-10] employs geometric optimization without queueing-theoretic foundations for congestion management and service guarantees.

2. **Static vs. Dynamic Capacity Management**: Airspace management systems [8-9] provide regulatory frameworks and static planning without real-time adaptive optimization capabilities.

3. **Single-Layer vs. Multi-Layer Optimization**: While layered cooperation exists [5], no work addresses capacity optimization across multiple vertical layers with differentiated service characteristics.

## 2.5 Positioning of MCRPS/D/K Framework

Our MCRPS/D/K vertical stratified queueing framework addresses these fundamental gaps through several key innovations:

1. **Theoretical Foundation**: To our knowledge, we present the first systematic queueing-theoretic model for vertical airspace with stratified capacity structure C = {8,6,4,3,2} reflecting physical altitude constraints.

2. **Algorithmic Innovation**: Our pressure-triggered cross-layer transfer mechanisms based on queue length, service rates, and Gini fairness coefficients provide theoretically grounded dynamic management.

3. **System Integration**: The 29-dimensional state space integrates queueing metrics with deep reinforcement learning for real-time multi-objective optimization (throughput, latency, fairness, stability, safety, transmission efficiency).

4. **Practical Application**: To our knowledge, this represents the first systematic implementation of vertical stratified queueing for UAV airspace management with experimental validation across 15 algorithms and 500,000 training timesteps.

The extensive experimental validation demonstrates that our framework achieves significant performance improvements: state-of-the-art algorithms (PPO: 4419.98±135.71, TD7: 4392.52±84.60, R2D2: 4289.22±82.23) substantially outperform traditional scheduling methods and establish new benchmarks for vertical stratified queueing optimization.

This comprehensive review establishes that while individual components of our framework (fairness metrics, DRL algorithms, UAV coordination, queueing analysis) exist in isolation, no prior work combines these elements into a unified theoretical framework for vertical stratified queueing with practical algorithmic implementation and experimental validation.

## References

[1] X. Li, X. Chen, and G. Li, "Fairness-aware task offloading and load balancing with delay constraints for Power Internet of Things," Ad Hoc Networks, vol. 153, p. 103333, 2024, DOI: 10.1016/j.adhoc.2023.103333.

[2] W. Chen, Y. Tian, X. Yu, B. Zheng, and X. Zhang, "Enhancing Fairness for Approximate Weighted Fair Queueing With a Single Queue," IEEE/ACM Transactions on Networking, vol. 32, no. 5, pp. 3901-3915, Oct. 2024, DOI: 10.1109/TNET.2024.3399212.

[3] V. Do and N. Usunier, "Optimizing Generalized Gini Indices for Fairness in Rankings," in Proc. 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2022, pp. 1706-1711, DOI: 10.1145/3477495.3532048.

[4] X. Kong, Y. Zhou, Z. Li, and S. Wang, "Multi-UAV simultaneous target assignment and path planning based on deep reinforcement learning in dynamic multiple obstacles environments," Frontiers in Neurorobotics, vol. 17, 2024, DOI: 10.3389/fnbot.2023.1302898.

[5] Y. Liu, X. Li, J. Wang, F. Wei, and J. Yang, "Reinforcement-Learning-Based Multi-UAV Cooperative Search for Moving Targets in 3D Scenarios," Drones, vol. 8, no. 8, p. 378, 2024, DOI: 10.3390/drones8080378.

[6] T.-T. Zhang, Y. Chen, R.-Z. Dong, X.-H. Li, J.-Y. Wang, and H.-L. Liu, "Autonomous decision-making of UAV cluster with communication constraints based on reinforcement learning," Journal of Cloud Computing, vol. 14, p. 12, 2025, DOI: 10.1186/s13677-025-00738-9.

[7] A. Paul, M. W. Levin, S. T. Waller, and D. Rey, "Data-driven optimization for drone delivery service planning with online demand," Transportation Research Part E: Logistics and Transportation Review, vol. 198, p. 104095, 2025, DOI: 10.1016/j.tre.2025.104095.

[8] N. Pongsakornsathien, N. El-Din Safwat, Y. Xie, A. Gardi, and R. Sabatini, "Advances in low-altitude airspace management for uncrewed aircraft and advanced air mobility," Progress in Aerospace Sciences, vol. 154, p. 101085, 2025, DOI: 10.1016/j.paerosci.2025.101085.

[9] L. Stuive and F. Gzara, "Airspace network design for urban UAV traffic management with congestion," Transportation Research Part C: Emerging Technologies, vol. 169, p. 104882, 2024, DOI: 10.1016/j.trc.2024.104882.

[10] Y. Xie, A. Gardi, M. Liang, and R. Sabatini, "Hybrid AI-based 4D trajectory management system for dense low altitude operations and Urban Air Mobility," Aerospace Science and Technology, vol. 153, p. 109422, 2024, DOI: 10.1016/j.ast.2024.109422.

[11] N. S. Albalawi, "Dynamic scheduling strategies for cloud-based load balancing in parallel and distributed systems," Journal of Cloud Computing, vol. 14, p. 33, 2025, DOI: 10.1186/s13677-025-00757-6.

[12] S. Fujimoto, W.-D. Chang, E. J. Smith, S. Gu, D. Precup, and D. Meger, "For SALE: State-Action Representation Learning for Deep Reinforcement Learning," in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2023.

[13] M. Hessel, J. Modayil, H. van Hasselt, T. Schaul, G. Ostrovski, W. Dabney, D. Horgan, B. Piot, M. Azar, and D. Silver, "Rainbow: Combining Improvements in Deep Reinforcement Learning," in Proc. AAAI Conference on Artificial Intelligence (AAAI), 2018, pp. 3215-3222.

[14] L. Espeholt, H. Soyer, R. Munos, K. Simonyan, V. Mnih, T. Ward, Y. Doron, V. Firoiu, T. Harley, I. Dunning, S. Legg, and K. Kavukcuoglu, "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures," in Proc. International Conference on Machine Learning (ICML), 2018, pp. 1407-1416.

[15] S. Kapturowski, G. Ostrovski, J. Quan, R. Munos, and W. Dabney, "Recurrent Experience Replay in Distributed Reinforcement Learning," in Proc. International Conference on Learning Representations (ICLR), 2019.

[16] J. Terven, "Deep Reinforcement Learning: A Chronological Overview and Methods," AI, vol. 6, no. 3, p. 46, 2025, DOI: 10.3390/ai6030046.

[17] M. Mitzenmacher and R. Shahout, "Queueing, Predictions, and Large Language Models: Challenges and Open Problems," Stochastic Systems, vol. 15, no. 3, pp. 195-219, 2025, DOI: 10.1287/stsy.2025.0106.

[18] S. Siddiqui, S. Fatima, A. Ali, S. K. Gupta, H. K. Singh, and S. Kim, "Modelling of queuing systems using blockchain based on Markov process for smart healthcare systems," Scientific Reports, vol. 15, no. 1, p. 1652, 2025, DOI: 10.1038/s41598-025-01652-5.

[19] D. Efrosinin, V. Vishnevsky, and N. Stepanova, "Optimal Scheduling in General Multi-Queue System by Combining Simulation and Neural Network Techniques," Sensors, vol. 23, no. 12, p. 5479, 2023, DOI: 10.3390/s23125479.

[20] G. Hanukov, Y. Barron, and U. Yechiali, "An M/G/1 Queue with Repeated Orbit While in Service," Mathematics, vol. 12, no. 23, p. 3722, 2024, DOI: 10.3390/math12233722.

[21] Y. Zhao, Y. Ji, and Y. Zheng, "Balanced truck dispatching strategy for inter-terminal container transportation with demand outsourcing," Mathematics, vol. 13, no. 13, p. 2163, 2025, DOI: 10.3390/math13132163.

[22] D. I. Choi and D.-E. Lim, "Analysis of the State-Dependent Queueing Model and Its Application to Battery Swapping and Charging Stations," Sustainability, vol. 12, no. 6, p. 2343, 2020, DOI: 10.3390/su12062343.

[23] M. Amjath, L. Kerbache, A. Elomri, S. Nachiappan, A. Diabat, and N. Govindan, "Queueing network models for the analysis and optimisation of material handling systems: a systematic literature review," Flexible Services and Manufacturing Journal, vol. 36, pp. 668–709, 2024, DOI: 10.1007/s10696-023-09505-x.

[24] D. J. Moore, "A Taxonomy of Hierarchical Multi-Agent Systems: Design Patterns, Coordination Mechanisms, and Industrial Applications," arXiv preprint arXiv:2508.12683, 2025.

[25] J. Kim and J. Park, "Core allocation to minimize total flow time in a multicore system," Queueing Systems, vol. 108, no. 3-4, pp. 475-577, 2024, DOI: 10.1007/s11134-024-09923-0.

[26] X. Wang, L. Wang, C. Dong, H. Ren, and K. Xing, "Reinforcement Learning-Based Dynamic Order Recommendation for On-Demand Food Delivery," Tsinghua Science and Technology, vol. 29, no. 2, pp. 356-367, 2024, DOI: 10.26599/TST.2023.9010041.

[27] H. Jahanshahi, A. Bozanta, M. Cevik, E. M. Kavuk, A. Tosun, and S. B. Sonuc, "A deep reinforcement learning approach for the meal delivery problem," Knowledge-Based Systems, vol. 243, p. 108489, 2022, DOI: 10.1016/j.knosys.2022.108489.

[28] X. Wang, L. Wang, C. Dong, H. Ren, and K. Xing, "An Online Deep Reinforcement Learning-Based Order Recommendation Framework for Rider-Centered Food Delivery System," IEEE Transactions on Intelligent Transportation Systems, vol. 24, no. 5, pp. 5640-5654, 2023, DOI: 10.1109/TITS.2023.3237580.