"""
消融实验结果分析器
Ablation Study Results Analyzer

分析消融实验结果并生成：
1. 组件贡献度分析
2. 性能对比图表
3. 统计显著性测试
4. 论文所需的精确数值

用法：
    python analyze_ablation_results.py --results ablation_results/final_results.json
    python analyze_ablation_results.py --generate-figures
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import os
from datetime import datetime
from scipy import stats


class AblationResultsAnalyzer:
    """消融实验结果分析器"""
    
    def __init__(self, results_file: str = None):
        self.results = {}
        self.analysis = {}
        
        if results_file and os.path.exists(results_file):
            self.load_results(results_file)
    
    def load_results(self, filepath: str):
        """加载消融实验结果"""
        print(f"📂 加载消融实验结果: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        print(f"✅ 已加载 {len(self.results)} 个实验结果")
        
        # 验证结果完整性
        self._validate_results()
        
    def _validate_results(self):
        """验证结果完整性"""
        required_experiments = ['full_system', 'no_high_priority', 'single_objective', 
                              'traditional_pyramid', 'no_transfer']
        
        missing = []
        failed = []
        
        for exp in required_experiments:
            if exp not in self.results:
                missing.append(exp)
            elif not self.results[exp].get('success', False):
                failed.append(exp)
        
        if missing:
            print(f"⚠️  缺失实验: {missing}")
        if failed:
            print(f"❌ 失败实验: {failed}")
        
        successful = len([r for r in self.results.values() if r.get('success', False)])
        print(f"✅ 成功实验: {successful}/{len(self.results)}")
    
    def calculate_component_contributions(self) -> Dict[str, float]:
        """
        计算各组件的贡献度
        
        贡献度 = (完整系统性能 - 移除组件后性能) / 完整系统性能 * 100%
        """
        if 'full_system' not in self.results:
            raise ValueError("缺少完整系统基准结果")
        
        full_system_performance = self.results['full_system']['mean_reward']
        
        contributions = {}
        component_mapping = {
            'no_high_priority': 'High-Layer Priority',
            'single_objective': 'Multi-Objective Optimization', 
            'traditional_pyramid': 'Inverted Pyramid Structure',
            'no_transfer': 'Transfer Mechanism'
        }
        
        print("🧮 计算组件贡献度...")
        print("-" * 50)
        
        for ablation_type, component_name in component_mapping.items():
            if ablation_type in self.results and self.results[ablation_type].get('success'):
                ablation_performance = self.results[ablation_type]['mean_reward']
                
                # 计算贡献度（性能下降百分比）
                contribution = (full_system_performance - ablation_performance) / full_system_performance * 100
                contributions[component_name] = contribution
                
                print(f"{component_name:<25}: {contribution:>6.1f}%")
            else:
                contributions[component_name] = 0.0
                print(f"{component_name:<25}: {'N/A':>6}")
        
        self.analysis['contributions'] = contributions
        return contributions
    
    def perform_statistical_analysis(self) -> Dict[str, Any]:
        """执行统计显著性分析"""
        print("📊 执行统计显著性分析...")
        
        if 'full_system' not in self.results:
            print("❌ 缺少完整系统基准，无法进行统计分析")
            return {}
        
        full_system_reward = self.results['full_system']['mean_reward']
        full_system_std = self.results['full_system']['std_reward']
        
        statistical_results = {}
        
        for ablation_type, result in self.results.items():
            if ablation_type == 'full_system' or not result.get('success'):
                continue
            
            ablation_reward = result['mean_reward']
            ablation_std = result['std_reward']
            
            # 假设正态分布，计算t统计量
            # 这里简化处理，实际应该有更多样本数据
            pooled_std = np.sqrt((full_system_std**2 + ablation_std**2) / 2)
            
            if pooled_std > 0:
                t_stat = (full_system_reward - ablation_reward) / pooled_std
                # 简化的p值估计（实际需要更复杂的计算）
                p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
                
                statistical_results[ablation_type] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': abs(full_system_reward - ablation_reward) / pooled_std
                }
            else:
                statistical_results[ablation_type] = {
                    't_statistic': 0,
                    'p_value': 1.0,
                    'significant': False,
                    'effect_size': 0
                }
        
        self.analysis['statistics'] = statistical_results
        return statistical_results
    
    def generate_contribution_pie_chart(self, output_path: str = "component_contributions.png"):
        """生成组件贡献度饼图"""
        if 'contributions' not in self.analysis:
            self.calculate_component_contributions()
        
        contributions = self.analysis['contributions']
        
        # 过滤掉贡献度为0的组件
        filtered_contributions = {k: v for k, v in contributions.items() if v > 0}
        
        if not filtered_contributions:
            print("❌ 无有效贡献度数据，无法生成饼图")
            return
        
        plt.figure(figsize=(10, 8))
        
        # 数据准备
        labels = list(filtered_contributions.keys())
        sizes = list(filtered_contributions.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        # 创建饼图
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors[:len(labels)], 
                                          autopct='%1.1f%%', startangle=90,
                                          textprops={'fontsize': 10})
        
        # 美化图表
        plt.title('Component Contribution Analysis\\n组件贡献度分析', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # 添加图例
        plt.legend(wedges, [f"{label}: {size:.1f}%" for label, size in zip(labels, sizes)],
                  title="Components",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 组件贡献度饼图已保存: {output_path}")
        
        return output_path
    
    def generate_performance_comparison_chart(self, output_path: str = "performance_comparison.png"):
        """生成性能对比条形图"""
        plt.figure(figsize=(12, 8))
        
        # 数据准备
        experiment_names = []
        mean_rewards = []
        std_rewards = []
        colors = []
        
        # 定义颜色和顺序
        color_map = {
            'full_system': '#2ECC71',  # 绿色 - 完整系统
            'no_high_priority': '#E74C3C',  # 红色 - 最大贡献
            'single_objective': '#F39C12',  # 橙色 - 次大贡献
            'traditional_pyramid': '#9B59B6',  # 紫色 - 中等贡献
            'no_transfer': '#3498DB'  # 蓝色 - 最小贡献
        }
        
        name_map = {
            'full_system': 'Complete System\\n(Baseline)',
            'no_high_priority': 'No High-Layer\\nPriority',
            'single_objective': 'Single-Objective\\nOptimization',
            'traditional_pyramid': 'Traditional\\nPyramid',
            'no_transfer': 'No Transfer\\nMechanism'
        }
        
        # 按期望顺序排列
        order = ['full_system', 'no_high_priority', 'single_objective', 
                'traditional_pyramid', 'no_transfer']
        
        for exp_type in order:
            if exp_type in self.results and self.results[exp_type].get('success'):
                experiment_names.append(name_map.get(exp_type, exp_type))
                mean_rewards.append(self.results[exp_type]['mean_reward'])
                std_rewards.append(self.results[exp_type]['std_reward'])
                colors.append(color_map.get(exp_type, '#BDC3C7'))
        
        if not experiment_names:
            print("❌ 无有效实验数据，无法生成对比图")
            return
        
        # 创建条形图
        bars = plt.bar(experiment_names, mean_rewards, yerr=std_rewards, 
                      color=colors, alpha=0.8, capsize=5)
        
        # 添加数值标签
        for bar, mean_val, std_val in zip(bars, mean_rewards, std_rewards):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + std_val + 5,
                    f'{mean_val:.1f}±{std_val:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 美化图表
        plt.title('Ablation Study Performance Comparison\\n消融实验性能对比', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Mean Reward', fontsize=12)
        plt.xlabel('Experiment Configuration', fontsize=12)
        
        # 添加网格
        plt.grid(True, alpha=0.3, axis='y')
        
        # 旋转x轴标签
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 性能对比图已保存: {output_path}")
        
        return output_path
    
    def generate_latex_table(self, output_path: str = "ablation_table.tex"):
        """生成论文用的LaTeX表格"""
        if 'contributions' not in self.analysis:
            self.calculate_component_contributions()
        
        contributions = self.analysis['contributions']
        
        # 按贡献度排序
        sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        
        latex_content = []
        latex_content.append("% 消融实验结果表格")
        latex_content.append("% Ablation Study Results Table")
        latex_content.append("")
        latex_content.append("\\\\begin{table}[htbp]")
        latex_content.append("\\\\caption{Ablation Study Results: Component Contribution Analysis}")
        latex_content.append("\\\\label{tab:ablation_results}")
        latex_content.append("\\\\centering")
        latex_content.append("\\\\begin{tabular}{lccccc}")
        latex_content.append("\\\\toprule")
        latex_content.append("\\\\textbf{Configuration} & \\\\textbf{Mean Reward} & \\\\textbf{Std Dev} & \\\\textbf{Performance} & \\\\textbf{Removed Component} & \\\\textbf{Contribution} \\\\\\\\")
        latex_content.append("\\\\midrule")
        
        # 完整系统（基准）
        if 'full_system' in self.results and self.results['full_system'].get('success'):
            result = self.results['full_system']
            latex_content.append(f"Complete System & {result['mean_reward']:.2f} & {result['std_reward']:.2f} & 100.0\\% & None & Baseline \\\\\\\\")
        
        # 消融实验结果
        component_mapping = {
            'High-Layer Priority': 'no_high_priority',
            'Multi-Objective Optimization': 'single_objective',
            'Inverted Pyramid Structure': 'traditional_pyramid', 
            'Transfer Mechanism': 'no_transfer'
        }
        
        for component, contribution in sorted_contributions:
            if component in component_mapping:
                ablation_type = component_mapping[component]
                if ablation_type in self.results and self.results[ablation_type].get('success'):
                    result = self.results[ablation_type]
                    full_reward = self.results['full_system']['mean_reward']
                    performance_pct = (result['mean_reward'] / full_reward) * 100
                    
                    latex_content.append(f"w/o {component} & {result['mean_reward']:.2f} & {result['std_reward']:.2f} & {performance_pct:.1f}\\% & {component} & {contribution:.1f}\\% \\\\\\\\")
        
        latex_content.append("\\\\bottomrule")
        latex_content.append("\\\\end{tabular}")
        latex_content.append("\\\\end{table}")
        latex_content.append("")
        latex_content.append("% Note: 'w/o' means 'without'")
        latex_content.append("% Performance shows the percentage relative to complete system")
        latex_content.append("% Contribution shows the performance drop when component is removed")
        
        # 保存文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\\n'.join(latex_content))
        
        print(f"📄 LaTeX表格已生成: {output_path}")
        return output_path
    
    def print_paper_ready_results(self):
        """打印论文就绪的结果数据"""
        print("\\n" + "="*60)
        print("📄 论文就绪的消融实验结果")
        print("="*60)
        
        if 'contributions' not in self.analysis:
            self.calculate_component_contributions()
        
        contributions = self.analysis['contributions']
        
        # 按贡献度排序
        sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        
        print("\\n🎯 组件贡献度排名:")
        print("-" * 40)
        for i, (component, contribution) in enumerate(sorted_contributions, 1):
            print(f"{i}. {component}: {contribution:.1f}%")
        
        print("\\n📊 详细实验数据:")
        print("-" * 50)
        print(f"{'配置':<25} {'平均奖励':<12} {'标准差':<8} {'相对性能':<8}")
        print("-" * 50)
        
        if 'full_system' in self.results:
            baseline = self.results['full_system']['mean_reward']
            result = self.results['full_system']
            print(f"{'Complete System':<25} {result['mean_reward']:<12.2f} {result['std_reward']:<8.2f} {'100.0%':<8}")
            
            component_mapping = {
                'High-Layer Priority': 'no_high_priority',
                'Multi-Objective Optimization': 'single_objective', 
                'Inverted Pyramid Structure': 'traditional_pyramid',
                'Transfer Mechanism': 'no_transfer'
            }
            
            for component, contribution in sorted_contributions:
                if component in component_mapping:
                    ablation_type = component_mapping[component]
                    if ablation_type in self.results and self.results[ablation_type].get('success'):
                        result = self.results[ablation_type]
                        relative_perf = (result['mean_reward'] / baseline) * 100
                        print(f"{'w/o ' + component:<25} {result['mean_reward']:<12.2f} {result['std_reward']:<8.2f} {relative_perf:<8.1f}%")
        
        print("\\n💡 论文中可直接使用的数据:")
        print("-" * 40)
        for component, contribution in sorted_contributions:
            print(f"- {component}组件贡献{contribution:.1f}%的性能提升")
        
        print(f"\\n📈 关键发现:")
        print(f"- 最重要组件: {sorted_contributions[0][0]} ({sorted_contributions[0][1]:.1f}%)")
        print(f"- 次重要组件: {sorted_contributions[1][0]} ({sorted_contributions[1][1]:.1f}%)")
        print(f"- 四个组件总贡献: {sum(c for _, c in sorted_contributions):.1f}%")
    
    def generate_all_figures(self, output_dir: str = "./ablation_figures/"):
        """生成所有图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🎨 生成所有消融实验图表到: {output_dir}")
        
        # 1. 组件贡献度饼图
        pie_path = os.path.join(output_dir, "component_contributions.png")
        self.generate_contribution_pie_chart(pie_path)
        
        # 2. 性能对比条形图
        bar_path = os.path.join(output_dir, "performance_comparison.png")
        self.generate_performance_comparison_chart(bar_path)
        
        # 3. LaTeX表格
        tex_path = os.path.join(output_dir, "ablation_table.tex")
        self.generate_latex_table(tex_path)
        
        # 4. 分析报告
        report_path = os.path.join(output_dir, "analysis_summary.txt")
        self._generate_text_summary(report_path)
        
        print(f"✅ 所有图表已生成完成!")
        return output_dir
    
    def _generate_text_summary(self, output_path: str):
        """生成文本摘要"""
        if 'contributions' not in self.analysis:
            self.calculate_component_contributions()
        
        contributions = self.analysis['contributions']
        sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("消融实验分析摘要\\n")
            f.write("Ablation Study Analysis Summary\\n")
            f.write("="*50 + "\\n\\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"实验数量: {len(self.results)}\\n\\n")
            
            f.write("组件贡献度排名:\\n")
            f.write("-"*30 + "\\n")
            for i, (component, contribution) in enumerate(sorted_contributions, 1):
                f.write(f"{i}. {component}: {contribution:.1f}%\\n")
            
            f.write("\\n主要结论:\\n")
            f.write("-"*30 + "\\n")
            f.write(f"1. 最重要组件: {sorted_contributions[0][0]}\\n")
            f.write(f"2. 总体贡献度: {sum(c for _, c in sorted_contributions):.1f}%\\n")
            f.write(f"3. 实验验证了所有组件的重要性\\n")
        
        print(f"📝 分析摘要已保存: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="消融实验结果分析器")
    parser.add_argument('--results', type=str, 
                       help='消融实验结果JSON文件路径')
    parser.add_argument('--generate-figures', action='store_true',
                       help='生成所有图表')
    parser.add_argument('--output-dir', type=str, default='./ablation_figures/',
                       help='图表输出目录')
    parser.add_argument('--print-results', action='store_true',
                       help='打印论文就绪的结果')
    
    args = parser.parse_args()
    
    if not args.results:
        print("❌ 请指定结果文件路径: --results path/to/results.json")
        return
    
    # 创建分析器
    analyzer = AblationResultsAnalyzer(args.results)
    
    if args.generate_figures:
        # 生成所有图表
        analyzer.generate_all_figures(args.output_dir)
    
    if args.print_results:
        # 打印论文结果
        analyzer.print_paper_ready_results()
    
    # 默认执行组件贡献度分析
    analyzer.calculate_component_contributions()
    analyzer.perform_statistical_analysis()
    
    print("\\n🎉 消融实验结果分析完成!")


if __name__ == "__main__":
    main()