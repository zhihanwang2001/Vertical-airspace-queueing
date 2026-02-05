# Cover Letter - Applied Soft Computing

**Date**: January 28, 2026

**To**: Editor-in-Chief, Applied Soft Computing

**Subject**: Submission of Manuscript - "Deep Reinforcement Learning for Urban Air Mobility: A Systematic Comparison and Structural Analysis of Vertical Layered Queueing Systems"

---

Dear Editor,

We are pleased to submit our manuscript entitled **"Deep Reinforcement Learning for Urban Air Mobility: A Systematic Comparison and Structural Analysis of Vertical Layered Queueing Systems"** for consideration for publication in Applied Soft Computing.

## Research Significance

Urban Air Mobility (UAM) represents a transformative technology for future transportation, but faces critical challenges in managing vertical airspace congestion. Our work addresses a fundamental question: **which deep reinforcement learning algorithms and system configurations are most effective for optimizing vertical layered queueing systems?** This question is crucial for the safe and efficient deployment of UAM systems as drone traffic scales.

## Key Contributions

Our manuscript makes several significant contributions to the field:

1. **Comprehensive Algorithm Comparison**: We provide the first systematic evaluation of 15 state-of-the-art DRL algorithms (including A2C, PPO, TD7, SAC, TD3) for vertical queueing systems, establishing that DRL achieves 59.9% performance improvement over heuristic baselines (p<0.001).

2. **Counter-Intuitive Structural Findings**: We demonstrate that inverted pyramid capacity configurations consistently outperform normal pyramid structures by 9.7%-19.7% across load levels, supported by theoretical analysis proving optimal capacity should be proportional to arrival weights.

3. **Capacity Paradox Discovery**: We identify a load-dependent capacity paradox where low-capacity systems (K=10) outperform high-capacity systems (K=30+) under extreme load due to state space explosion—a finding with important implications for system design.

4. **Architectural Validation**: Through comprehensive ablation studies, we prove that capacity-aware action clipping is essential for system stability, with its removal causing 100% crash rate despite identical network capacity (821K parameters). This validates that performance stems from architectural design, not merely parameter scaling.

5. **Rigorous Statistical Validation**: Our findings are validated through 500,000 training timesteps per algorithm, Pareto analysis of 10,000 policy configurations, bootstrap confidence intervals, and analysis across 260+ experimental runs.

## Fit with Applied Soft Computing

This work is highly suitable for Applied Soft Computing for several reasons:

- **Soft Computing Methods**: Deep reinforcement learning represents a core soft computing technique, and our work advances understanding of DRL algorithm selection and architectural design.

- **Real-World Application**: UAM systems represent an emerging real-world application domain where soft computing methods can provide significant value.

- **Methodological Rigor**: Our systematic comparison methodology and statistical validation align with the journal's emphasis on rigorous evaluation of soft computing approaches.

- **Practical Impact**: Our findings provide evidence-based guidelines for practitioners designing UAM systems, addressing the journal's focus on practical applications.

## Novelty and Originality

To the best of our knowledge, this is the first work to:
- Systematically compare 15 DRL algorithms for vertical layered queueing systems
- Identify and theoretically explain the inverted pyramid advantage
- Discover the load-dependent capacity paradox
- Validate architectural contributions through ablation studies in this domain

All work is original, and the manuscript has not been submitted elsewhere for publication.

## Ethical Compliance

- All authors have approved the manuscript and agree with its submission to Applied Soft Computing
- There are no conflicts of interest to declare
- No human subjects or animal experiments were involved
- All data and code will be made available upon publication to ensure reproducibility

## Suggested Reviewers

We suggest the following potential reviewers with expertise in reinforcement learning, queueing systems, and UAM:

1. **Dr. [Name]**, [Institution] - Expert in deep reinforcement learning for transportation systems
2. **Dr. [Name]**, [Institution] - Expert in queueing theory and optimization
3. **Dr. [Name]**, [Institution] - Expert in Urban Air Mobility systems

## Conclusion

We believe this manuscript makes significant contributions to understanding how deep reinforcement learning can be effectively applied to Urban Air Mobility systems. The comprehensive evaluation, counter-intuitive findings, and rigorous validation make it well-suited for publication in Applied Soft Computing.

We look forward to your consideration of our manuscript.

Sincerely,

[Author Names]
[Affiliations]
[Contact Information]

---

## Manuscript Statistics

- **Pages**: 39
- **Words**: ~12,000
- **Figures**: 15+
- **Tables**: 17+
- **References**: 60+
- **Supplementary Materials**: Available upon request

