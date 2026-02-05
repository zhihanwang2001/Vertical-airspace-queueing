"""
Validate Supplementary Experiment Results

This script validates the integrity and completeness of Priority 1 and Priority 2
experimental results to ensure data quality before analysis.

Priority 1: Load sensitivity analysis (loads 3,4,8,9,10)
Priority 2: Structural comparison generalization (loads 3,7,10)
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class ExperimentValidator:
    """Validates experimental results for data quality and completeness."""

    def __init__(self, base_path: str = "Data"):
        self.base_path = Path(base_path)
        self.errors = []
        self.warnings = []
        self.summary = {}

    def validate_priority1(self) -> Dict:
        """Validate Priority 1: Load sensitivity analysis."""
        print("\n" + "="*60)
        print("PRIORITY 1: Load Sensitivity Analysis Validation")
        print("="*60)

        expected_loads = [3, 4, 8, 9, 10]
        expected_capacities = [10, 30]
        expected_algos = ['A2C', 'PPO']
        expected_seeds = [42, 43, 44, 45, 46]
        expected_runs = len(expected_loads) * len(expected_capacities) * len(expected_algos) * len(expected_seeds)

        print(f"\nExpected configuration:")
        print(f"  Loads: {expected_loads}")
        print(f"  Capacities: {expected_capacities}")
        print(f"  Algorithms: {expected_algos}")
        print(f"  Seeds: {expected_seeds}")
        print(f"  Total expected runs: {expected_runs}")

        # Check CSV summary files
        csv_files = [
            self.base_path / "summary" / "capacity_scan_results_uniform_3_4.csv",
            self.base_path / "summary" / "capacity_scan_results_uniform_8_9_10.csv"
        ]

        all_data = []
        for csv_file in csv_files:
            if not csv_file.exists():
                self.errors.append(f"Missing CSV file: {csv_file}")
                print(f"❌ Missing: {csv_file.name}")
            else:
                df = pd.read_csv(csv_file)
                all_data.append(df)
                print(f"✅ Found: {csv_file.name} ({len(df)} rows)")

        if not all_data:
            print("\n❌ No data files found for Priority 1")
            return {"status": "failed", "errors": self.errors}

        # Combine all data
        df_combined = pd.concat(all_data, ignore_index=True)

        # Validate data integrity
        self._validate_data_integrity(df_combined, "Priority 1")

        # Check completeness
        self._check_completeness(df_combined, expected_loads, expected_capacities,
                                expected_algos, expected_seeds, "Priority 1")

        return {
            "status": "passed" if not self.errors else "failed",
            "total_runs": len(df_combined),
            "expected_runs": expected_runs,
            "errors": self.errors,
            "warnings": self.warnings
        }

    def _validate_data_integrity(self, df: pd.DataFrame, experiment_name: str):
        """Check for data quality issues."""
        print(f"\n--- Data Integrity Checks for {experiment_name} ---")

        # Check for NaN values
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            self.errors.append(f"{experiment_name}: NaN values found in columns: {nan_cols}")
            print(f"❌ NaN values in: {nan_cols}")
        else:
            print("✅ No NaN values")

        # Check for infinite values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
        if inf_cols:
            self.errors.append(f"{experiment_name}: Infinite values found in columns: {inf_cols}")
            print(f"❌ Infinite values in: {inf_cols}")
        else:
            print("✅ No infinite values")

        # Check reward ranges
        if 'mean_reward' in df.columns:
            min_reward = df['mean_reward'].min()
            max_reward = df['mean_reward'].max()
            print(f"✅ Reward range: [{min_reward:.1f}, {max_reward:.1f}]")

            # Flag suspicious rewards (negative or extremely high)
            if min_reward < -1000:
                self.warnings.append(f"{experiment_name}: Very low rewards detected (min={min_reward:.1f})")
                print(f"⚠️  Warning: Very low rewards (min={min_reward:.1f})")

        # Check crash rates
        if 'crash_rate' in df.columns:
            min_crash = df['crash_rate'].min()
            max_crash = df['crash_rate'].max()
            print(f"✅ Crash rate range: [{min_crash:.2%}, {max_crash:.2%}]")

            # Flag if all runs crashed
            if min_crash == 1.0:
                self.errors.append(f"{experiment_name}: All runs crashed (100% crash rate)")
                print(f"❌ All runs crashed")

    def _check_completeness(self, df: pd.DataFrame, expected_loads: List,
                           expected_capacities: List, expected_algos: List,
                           expected_seeds: List, experiment_name: str):
        """Check if all expected configurations are present."""
        print(f"\n--- Completeness Checks for {experiment_name} ---")

        # Check loads
        actual_loads = sorted(df['load_multiplier'].unique())
        missing_loads = set(expected_loads) - set(actual_loads)
        if missing_loads:
            self.errors.append(f"{experiment_name}: Missing loads: {sorted(missing_loads)}")
            print(f"❌ Missing loads: {sorted(missing_loads)}")
        else:
            print(f"✅ All loads present: {actual_loads}")

        # Check capacities
        if 'total_capacity' in df.columns:
            actual_capacities = sorted(df['total_capacity'].unique())
            missing_capacities = set(expected_capacities) - set(actual_capacities)
            if missing_capacities:
                self.errors.append(f"{experiment_name}: Missing capacities: {sorted(missing_capacities)}")
                print(f"❌ Missing capacities: {sorted(missing_capacities)}")
            else:
                print(f"✅ All capacities present: {actual_capacities}")

        # Check algorithms
        actual_algos = sorted(df['algorithm'].unique())
        missing_algos = set(expected_algos) - set(actual_algos)
        if missing_algos:
            self.errors.append(f"{experiment_name}: Missing algorithms: {sorted(missing_algos)}")
            print(f"❌ Missing algorithms: {sorted(missing_algos)}")
        else:
            print(f"✅ All algorithms present: {actual_algos}")

        # Check seeds
        if 'seed' in df.columns:
            actual_seeds = sorted(df['seed'].unique())
            missing_seeds = set(expected_seeds) - set(actual_seeds)
            if missing_seeds:
                self.warnings.append(f"{experiment_name}: Missing seeds: {sorted(missing_seeds)}")
                print(f"⚠️  Missing seeds: {sorted(missing_seeds)}")
            else:
                print(f"✅ All seeds present: {actual_seeds}")

    def validate_priority2(self) -> Dict:
        """Validate Priority 2: Structural comparison generalization."""
        print("\n" + "="*60)
        print("PRIORITY 2: Structural Comparison Generalization Validation")
        print("="*60)

        expected_loads = [3, 7, 10]
        expected_structures = ['inverted', 'normal']
        expected_algos = ['A2C', 'PPO']
        expected_seeds = [42, 43, 44, 45, 46]
        expected_runs = len(expected_loads) * len(expected_structures) * len(expected_algos) * len(expected_seeds)

        print(f"\nExpected configuration:")
        print(f"  Loads: {expected_loads}")
        print(f"  Structures: {expected_structures}")
        print(f"  Algorithms: {expected_algos}")
        print(f"  Seeds: {expected_seeds}")
        print(f"  Total expected runs: {expected_runs}")

        # Check for structural comparison data
        # Note: This will need to be adjusted based on actual output format
        summary_path = self.base_path / "ablation_studies" / "priority2_structural_generalization"

        if not summary_path.exists():
            self.errors.append(f"Priority 2 data directory not found: {summary_path}")
            print(f"❌ Directory not found: {summary_path}")
            return {
                "status": "failed",
                "total_runs": 0,
                "expected_runs": expected_runs,
                "errors": self.errors,
                "warnings": self.warnings
            }

        # Count result files
        result_files = list(summary_path.rglob("*_results.json"))
        print(f"\n✅ Found {len(result_files)} result files")

        if len(result_files) == 0:
            self.errors.append("Priority 2: No result files found")
            print("❌ No result files found")
            return {
                "status": "failed",
                "total_runs": 0,
                "expected_runs": expected_runs,
                "errors": self.errors,
                "warnings": self.warnings
            }

        # Load and validate individual results
        results_data = []
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    results_data.append(data)
            except Exception as e:
                self.errors.append(f"Failed to load {result_file.name}: {str(e)}")
                print(f"❌ Failed to load: {result_file.name}")

        print(f"✅ Successfully loaded {len(results_data)} result files")

        # Basic validation on loaded data
        if results_data:
            print("\n--- Sample Data Check ---")
            sample = results_data[0]
            print(f"Sample keys: {list(sample.keys())}")

        return {
            "status": "passed" if not self.errors else "failed",
            "total_runs": len(results_data),
            "expected_runs": expected_runs,
            "errors": self.errors,
            "warnings": self.warnings
        }

    def generate_report(self) -> str:
        """Generate a comprehensive validation report."""
        report = []
        report.append("\n" + "="*60)
        report.append("VALIDATION REPORT SUMMARY")
        report.append("="*60)

        if not self.errors and not self.warnings:
            report.append("\n✅ ALL VALIDATIONS PASSED")
            report.append("   Data is ready for analysis")
        else:
            if self.errors:
                report.append(f"\n❌ ERRORS FOUND: {len(self.errors)}")
                for i, error in enumerate(self.errors, 1):
                    report.append(f"   {i}. {error}")

            if self.warnings:
                report.append(f"\n⚠️  WARNINGS: {len(self.warnings)}")
                for i, warning in enumerate(self.warnings, 1):
                    report.append(f"   {i}. {warning}")

        report.append("\n" + "="*60)
        return "\n".join(report)


def main():
    """Main validation workflow."""
    validator = ExperimentValidator()

    # Validate Priority 1
    p1_result = validator.validate_priority1()

    # Validate Priority 2
    p2_result = validator.validate_priority2()

    # Generate final report
    report = validator.generate_report()
    print(report)

    # Save report to file
    report_path = Path("Analysis/statistical_reports/validation_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n📄 Report saved to: {report_path}")

    # Return exit code
    return 0 if not validator.errors else 1


if __name__ == '__main__':
    exit(main())
