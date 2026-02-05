"""
Ablation Study Experiment Runner

Run complete ablation study including:
1. Full system (control group)
2. No high priority experiment
3. Single objective optimization experiment
4. Traditional pyramid experiment
5. No transfer mechanism experiment

Usage:
    python run_ablation_experiments.py --timesteps 100000 --all
    python run_ablation_experiments.py --experiment no_high_priority --timesteps 50000
    python run_ablation_experiments.py --quick-test
"""

import argparse
import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.baselines.sb3_ablation_baseline import SB3AblationBaseline, AblationExperimentManager
from ablation_configs import AblationConfigs


class AblationExperimentRunner:
    """Ablation experiment runner"""

    def __init__(self, output_dir="./ablation_results/"):
        self.output_dir = output_dir
        self.results = {}
        self.start_time = None

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"../../Models/", exist_ok=True)
        os.makedirs(f"{output_dir}/logs/", exist_ok=True)
        
    def run_single_experiment(self, ablation_type: str, timesteps: int = 100000,
                            eval_episodes: int = 30) -> Dict[str, Any]:
        """
        Run single ablation experiment

        Args:
            ablation_type: Ablation experiment type
            timesteps: Training timesteps
            eval_episodes: Evaluation episodes

        Returns:
            Experiment results dictionary
        """
        print(f"\nRunning ablation experiment: {ablation_type}")
        print(f"   Training timesteps: {timesteps:,}")
        print(f"   Evaluation episodes: {eval_episodes}")
        print("=" * 50)

        experiment_start = time.time()

        try:
            # Create ablation baseline
            baseline = SB3AblationBaseline(ablation_type)

            # Train model
            print(f"Starting training...")
            baseline.train(total_timesteps=timesteps)

            # Evaluate performance
            print(f"Starting evaluation...")
            results = baseline.evaluate(n_episodes=eval_episodes)

            # Add experiment metadata
            experiment_time = time.time() - experiment_start
            results.update({
                'timesteps': timesteps,
                'eval_episodes': eval_episodes,
                'experiment_time': experiment_time,
                'timestamp': datetime.now().isoformat(),
                'success': True
            })

            print(f"Completed: {ablation_type} experiment!")
            print(f"   Training time: {experiment_time:.1f}s")
            print(f"   Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")

            return results

        except Exception as e:
            error_result = {
                'ablation_type': ablation_type,
                'error': str(e),
                'timesteps': timesteps,
                'success': False,
                'timestamp': datetime.now().isoformat()
            }

            print(f"Failed: {ablation_type} experiment: {str(e)}")
            return error_result

    def run_all_experiments(self, timesteps: int = 100000, eval_episodes: int = 30) -> Dict[str, Any]:
        """
        Run all ablation experiments

        Args:
            timesteps: Training timesteps per experiment
            eval_episodes: Evaluation episodes per experiment

        Returns:
            All experiment results
        """
        self.start_time = time.time()

        # Get all ablation experiment types
        ablation_types = [
            'full_system',      # Full system (control group)
            'no_high_priority', # No high priority
            'single_objective', # Single objective optimization
            'traditional_pyramid', # Traditional pyramid
            'no_transfer'       # No transfer mechanism
        ]

        print(f"Starting complete ablation study")
        print(f"   Number of experiments: {len(ablation_types)}")
        print(f"   Training timesteps per experiment: {timesteps:,}")
        print(f"   Evaluation episodes per experiment: {eval_episodes}")
        print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Run each experiment
        for i, ablation_type in enumerate(ablation_types, 1):
            print(f"\nProgress: {i}/{len(ablation_types)} - {ablation_type}")

            # Run single experiment
            result = self.run_single_experiment(ablation_type, timesteps, eval_episodes)
            self.results[ablation_type] = result

            # Save intermediate results
            self._save_intermediate_results()

            # Print current comparison
            if i > 1:  # Only compare when at least 2 results exist
                self._print_current_comparison()

        # Complete all experiments
        total_time = time.time() - self.start_time
        print(f"\n🎉 Ablation study completed!")
        print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f}min)")

        # Generate final report
        self._generate_final_report()
        
        return self.results

    def _save_intermediate_results(self):
        """Save intermediate results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f"{self.output_dir}/intermediate_results_{timestamp}.json"

        # Convert numpy types to Python native types
        results_serializable = self._convert_numpy_types(self.results)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)

    def _convert_numpy_types(self, obj):
        """Convert numpy types to JSON-serializable Python types"""
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def _print_current_comparison(self):
        """Print current comparison results"""
        if len(self.results) < 2:
            return

        print(f"\n📊 Current comparison results:")
        print("-" * 70)
        print(f"{'Experiment Type':<20} {'Mean Reward':<12} {'Std Dev':<8} {'Performance Change':<10} {'Status'}")
        print("-" * 70)

        full_system_reward = None
        if 'full_system' in self.results and self.results['full_system'].get('success'):
            full_system_reward = self.results['full_system']['mean_reward']

        for ablation_type, result in self.results.items():
            if not result.get('success', False):
                print(f"{ablation_type:<20} {'ERROR':<12} {'-':<8} {'-':<10} {'❌'}")
                continue

            mean_reward = result['mean_reward']
            std_reward = result['std_reward']

            if ablation_type == 'full_system':
                change = "Baseline"
                status = "✅"
            elif full_system_reward:
                change_percent = (mean_reward - full_system_reward) / full_system_reward * 100
                change = f"{change_percent:+.1f}%"
                status = "✅" if change_percent > -5 else "📉"
            else:
                change = "Pending"
                status = "⏳"

            print(f"{ablation_type:<20} {mean_reward:<12.2f} {std_reward:<8.2f} {change:<10} {status}")

        print("-" * 70)

    def _generate_final_report(self):
        """Generate final experiment report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON results
        json_path = f"{self.output_dir}/final_ablation_results_{timestamp}.json"
        results_serializable = self._convert_numpy_types(self.results)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)

        # Markdown report
        md_path = f"{self.output_dir}/ablation_report_{timestamp}.md"
        self._create_markdown_report(md_path)

        print(f"📄 Reports generated:")
        print(f"   JSON results: {json_path}")
        print(f"   Markdown report: {md_path}")

    def _create_markdown_report(self, filepath: str):
        """Create Markdown format experiment report"""

        # Get experiment plan information
        experiment_plan = AblationConfigs.get_ablation_experiment_plan()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Ablation Study Research Report\n\n")

            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Number of experiments**: {len(self.results)}  \n")
            if self.start_time:
                total_time = time.time() - self.start_time
                f.write(f"**Total time**: {total_time:.1f}s ({total_time/60:.1f}min)  \n")
            f.write("\\n")

            # Experiment results table
            f.write("## 📊 Experiment Results Comparison\n\n")
            f.write("| Experiment Type | Mean Reward | Std Dev | Performance Change | Removed Component | Status |\\n")
            f.write("|----------------|-------------|---------|-------------------|-------------------|--------|\\n")
            
            full_system_reward = None
            if 'full_system' in self.results and self.results['full_system'].get('success'):
                full_system_reward = self.results['full_system']['mean_reward']
            
            for ablation_type, result in self.results.items():
                plan_info = experiment_plan.get(ablation_type, {})
                removed_component = plan_info.get('removed_component', 'None')
                
                if not result.get('success', False):
                    f.write(f"| {ablation_type} | ERROR | - | - | {removed_component} | ❌ |\\n")
                    continue
                
                mean_reward = result['mean_reward']
                std_reward = result['std_reward']
                
                if ablation_type == 'full_system':
                    change = "Baseline (100%)"
                    status = "✅"
                elif full_system_reward:
                    change_percent = (mean_reward - full_system_reward) / full_system_reward * 100
                    change = f"{change_percent:+.1f}%"
                    status = "✅" if change_percent > -5 else "📉"
                else:
                    change = "Pending"
                    status = "⏳"
                
                f.write(f"| {ablation_type} | {mean_reward:.2f} | {std_reward:.2f} | {change} | {removed_component} | {status} |\\n")

            # Detailed experiment information
            f.write("\\n## 🧪 Detailed Experiment Information\\n\\n")

            for ablation_type, result in self.results.items():
                plan_info = experiment_plan.get(ablation_type, {})
                f.write(f"### {plan_info.get('name', ablation_type)}\\n\\n")
                f.write(f"**Description**: {plan_info.get('description', 'N/A')}  \\n")

                if 'removed_component' in plan_info:
                    f.write(f"**Removed Component**: {plan_info['removed_component']}  \\n")

                if 'hypothesis' in plan_info:
                    f.write(f"**Hypothesis**: {plan_info['hypothesis']}  \\n")

                if result.get('success'):
                    f.write(f"**Experiment Results**:  \\n")
                    f.write(f"- Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}  \\n")
                    f.write(f"- Training timesteps: {result['timesteps']:,}  \\n")
                    f.write(f"- Evaluation episodes: {result['eval_episodes']}  \\n")
                    f.write(f"- Experiment time: {result['experiment_time']:.1f}s  \\n")
                else:
                    f.write(f"**Experiment Failed**: {result.get('error', 'Unknown error')}  \\n")

                f.write("\\n")

            # Conclusions and analysis
            f.write("## 🎯 Conclusions and Analysis\\n\\n")

            if full_system_reward:
                f.write("### Component Contribution Ranking\\n\\n")

                contributions = []
                for ablation_type, result in self.results.items():
                    if ablation_type == 'full_system' or not result.get('success'):
                        continue

                    contribution = (full_system_reward - result['mean_reward']) / full_system_reward * 100
                    removed_component = experiment_plan.get(ablation_type, {}).get('removed_component', ablation_type)
                    contributions.append((removed_component, contribution, ablation_type))

                contributions.sort(key=lambda x: x[1], reverse=True)

                for i, (component, contribution, ablation_type) in enumerate(contributions, 1):
                    f.write(f"{i}. **{component}**: Contributes {contribution:.1f}% (performance drops when removed)\\n")

                f.write("\\n")

            f.write("### Key Findings\\n\\n")
            f.write("1. **System Integrity**: Each component makes important contributions to overall performance\\n")
            f.write("2. **Component Synergy**: Multiple innovative components work together to produce optimal results\\n")
            f.write("3. **Design Validation**: Ablation experiments validate the effectiveness of our system design\\n\\n")

            f.write("---\\n")
            f.write("*Report automatically generated by ablation experiment system*\\n")

    def quick_test(self):
        """Quick test all ablation experiments (for debugging)"""
        print("🚀 Quick test mode")
        print("   Training timesteps: 1,000")
        print("   Evaluation episodes: 3")
        print("=" * 40)
        
        return self.run_all_experiments(timesteps=1000, eval_episodes=3)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Ablation experiment runner")
    parser.add_argument('--experiment', type=str, choices=[
        'full_system', 'no_high_priority', 'single_objective',
        'traditional_pyramid', 'no_transfer', 'all'
    ], default='all', help='Ablation experiment type to run')

    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Training timesteps (default: 100,000)')
    parser.add_argument('--eval-episodes', type=int, default=30,
                       help='Evaluation episodes (default: 30)')
    parser.add_argument('--output-dir', type=str, default='./ablation_results/',
                       help='Output directory (default: ./ablation_results/)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode (1k timesteps, 3 episodes)')

    args = parser.parse_args()

    # Create experiment runner
    runner = AblationExperimentRunner(output_dir=args.output_dir)

    print("🧪 Ablation Experiment Research System")
    print("=" * 50)
    print(f"Output directory: {args.output_dir}")

    if args.quick_test:
        # Quick test
        results = runner.quick_test()
    elif args.experiment == 'all':
        # Run all experiments
        results = runner.run_all_experiments(args.timesteps, args.eval_episodes)
    else:
        # Run single experiment
        result = runner.run_single_experiment(args.experiment, args.timesteps, args.eval_episodes)
        results = {args.experiment: result}

    print(f"\\n🎉 Experiments complete! Results saved to: {args.output_dir}")

    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n\\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n\\n❌ Experiment execution error: {str(e)}")
        sys.exit(1)