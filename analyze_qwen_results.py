#!/usr/bin/env python3
"""
Qwen3-32B Evaluation Result Analyzer
Helps analyze and compare BFCL evaluation results for Qwen3-32B
"""

import csv
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import statistics

class BFCLResultAnalyzer:
    def __init__(self, bfcl_root: str):
        self.bfcl_root = Path(bfcl_root)
        self.score_dir = self.bfcl_root / "score"
        self.result_dir = self.bfcl_root / "result"
        
    def read_csv(self, filename: str) -> Dict:
        """Read CSV file and return as list of dicts"""
        filepath = self.score_dir / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found")
            return []
        
        data = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    
    def get_overall_scores(self) -> Dict:
        """Get overall evaluation scores"""
        data = self.read_csv("data_overall.csv")
        if not data:
            return {}
        
        scores = {}
        for row in data:
            model = row.get('Model', 'Unknown')
            scores[model] = row
        
        return scores
    
    def get_category_breakdown(self) -> Dict:
        """Get scores broken down by category"""
        categories = {
            'multi_turn': 'data_multi_turn.csv',
            'live': 'data_live.csv',
            'non_live': 'data_non_live.csv',
        }
        
        breakdown = {}
        for cat_name, filename in categories.items():
            breakdown[cat_name] = self.read_csv(filename)
        
        return breakdown
    
    def get_qwen_results(self) -> Dict:
        """Extract Qwen3-32B results specifically"""
        overall = self.get_overall_scores()
        
        qwen_results = {}
        for model_name, scores in overall.items():
            if 'qwen3' in model_name.lower() and '32b' in model_name.lower():
                qwen_results[model_name] = scores
        
        return qwen_results
    
    def print_overall_summary(self):
        """Print overall evaluation summary"""
        overall = self.get_overall_scores()
        
        if not overall:
            print("No overall scores found")
            return
        
        print("\n" + "="*80)
        print("OVERALL EVALUATION SUMMARY")
        print("="*80)
        
        # Print header
        print(f"{'Model':<40} {'Accuracy':<15} {'Samples':<10}")
        print("-"*80)
        
        # Print data
        for model_name, scores in overall.items():
            accuracy = scores.get('Accuracy', 'N/A')
            samples = scores.get('Samples', 'N/A')
            print(f"{model_name:<40} {accuracy:<15} {samples:<10}")
    
    def print_qwen_detailed(self):
        """Print detailed Qwen3-32B results"""
        qwen = self.get_qwen_results()
        
        if not qwen:
            print("\nNo Qwen3-32B results found")
            return
        
        print("\n" + "="*80)
        print("QWEN3-32B DETAILED EVALUATION")
        print("="*80)
        
        for model_name, scores in qwen.items():
            print(f"\n{model_name}:")
            print("-"*80)
            for key, value in scores.items():
                print(f"  {key:<30} {value}")
    
    def print_category_analysis(self):
        """Print analysis by category"""
        breakdown = self.get_category_breakdown()
        
        print("\n" + "="*80)
        print("CATEGORY-WISE ANALYSIS")
        print("="*80)
        
        for cat_name, data in breakdown.items():
            if not data:
                print(f"\n{cat_name}: No data available")
                continue
            
            print(f"\n{cat_name.upper()}:")
            print("-"*80)
            
            # Find Qwen rows
            qwen_rows = [row for row in data 
                        if 'qwen3' in row.get('Model', '').lower() 
                        and '32b' in row.get('Model', '').lower()]
            
            if qwen_rows:
                for row in qwen_rows[:5]:  # Show first 5
                    model = row.get('Model', 'Unknown')
                    print(f"\n  {model}:")
                    for key, value in row.items():
                        if key != 'Model':
                            print(f"    {key:<25} {value}")
            else:
                print("  No Qwen3-32B results in this category")
    
    def compare_models(self, models: List[str] = None):
        """Compare Qwen3-32B with other models"""
        overall = self.get_overall_scores()
        
        if not overall:
            print("No data to compare")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        # Get Qwen model
        qwen_model = None
        for model_name in overall.keys():
            if 'qwen3' in model_name.lower() and '32b' in model_name.lower():
                qwen_model = model_name
                break
        
        if not qwen_model:
            print("Qwen3-32B not found in results")
            return
        
        qwen_acc = float(overall[qwen_model].get('Accuracy', 0))
        
        print(f"\nQwen3-32B Accuracy: {overall[qwen_model].get('Accuracy')}%")
        print("\nOther models:")
        print(f"{'Model':<40} {'Accuracy':<15} {'Diff':<10}")
        print("-"*80)
        
        comparisons = []
        for model_name, scores in overall.items():
            if model_name == qwen_model:
                continue
            
            try:
                acc = float(scores.get('Accuracy', 0))
                diff = acc - qwen_acc
                comparisons.append((model_name, acc, diff))
            except ValueError:
                pass
        
        # Sort by difference
        comparisons.sort(key=lambda x: x[2], reverse=True)
        
        for model_name, acc, diff in comparisons[:10]:
            diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
            print(f"{model_name:<40} {acc:<15.1f} {diff_str:<10}")
    
    def generate_report(self, output_file: str = None):
        """Generate comprehensive report"""
        report_lines = []
        
        report_lines.append("="*80)
        report_lines.append("QWEN3-32B BFCL EVALUATION REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Overall summary
        overall = self.get_overall_scores()
        qwen = self.get_qwen_results()
        
        report_lines.append("EXECUTION SUMMARY")
        report_lines.append("-"*80)
        report_lines.append(f"Total Models Evaluated: {len(overall)}")
        report_lines.append(f"Qwen3-32B Variants: {len(qwen)}")
        report_lines.append("")
        
        # Qwen results
        if qwen:
            report_lines.append("QWEN3-32B RESULTS")
            report_lines.append("-"*80)
            for model_name, scores in qwen.items():
                report_lines.append(f"Model: {model_name}")
                for key, value in scores.items():
                    report_lines.append(f"  {key}: {value}")
                report_lines.append("")
        
        # Category breakdown
        breakdown = self.get_category_breakdown()
        report_lines.append("CATEGORY BREAKDOWN")
        report_lines.append("-"*80)
        for cat_name, data in breakdown.items():
            qwen_rows = [row for row in data 
                        if 'qwen3' in row.get('Model', '').lower() 
                        and '32b' in row.get('Model', '').lower()]
            if qwen_rows:
                report_lines.append(f"\n{cat_name}:")
                for row in qwen_rows[:3]:
                    report_lines.append(f"  {row.get('Model')}: {row}")
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")
        
        return report_text

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Qwen3-32B BFCL evaluation results"
    )
    parser.add_argument(
        '--bfcl-root',
        default='/Users/vgangal/phd_research_workspace/gorilla_bfcl',
        help='BFCL project root directory'
    )
    parser.add_argument(
        '--action',
        choices=['summary', 'detailed', 'category', 'compare', 'report'],
        default='summary',
        help='Analysis action to perform'
    )
    parser.add_argument(
        '--output',
        help='Output file for report (for --action report)'
    )
    
    args = parser.parse_args()
    
    analyzer = BFCLResultAnalyzer(args.bfcl_root)
    
    if args.action == 'summary':
        analyzer.print_overall_summary()
    elif args.action == 'detailed':
        analyzer.print_qwen_detailed()
    elif args.action == 'category':
        analyzer.print_category_analysis()
    elif args.action == 'compare':
        analyzer.compare_models()
    elif args.action == 'report':
        output = args.output or 'qwen3_32b_evaluation_report.txt'
        analyzer.generate_report(output)
    
    print("\n")

if __name__ == '__main__':
    main()
