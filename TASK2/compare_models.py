#!/usr/bin/env python3
"""
Model Comparison Script for Barkopedia Gender Classification.
Compare results from AST, Wav2Vec2 Custom, and Wav2Vec2 Pretrained models.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_results(results_dir):
    """Load training results from JSON files."""
    results = {}
    
    # Look for result files
    for file in os.listdir(results_dir):
        if file.endswith('_training_results.json'):
            model_type = file.replace('_training_results.json', '')
            try:
                with open(os.path.join(results_dir, file), 'r') as f:
                    results[model_type] = json.load(f)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    return results

def create_comparison_plots(results, save_dir):
    """Create comparison plots for all models."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison: AST vs Wav2Vec2 Custom vs Wav2Vec2 Pretrained', fontsize=16)
    
    # Extract data for plotting
    models = []
    final_accuracies = []
    final_f1_scores = []
    best_f1_scores = []
    
    # Training curves data
    training_data = {}
    
    for model_type, result in results.items():
        models.append(model_type.replace('_', ' ').title())
        final_accuracies.append(result['final_evaluation']['accuracy'])
        final_f1_scores.append(result['final_evaluation']['f1_score'])
        best_f1_scores.append(result['best_f1_score'])
        
        # Extract training curves
        eval_history = result['evaluation_history']
        if eval_history:
            training_data[model_type] = {
                'steps': [e['step'] for e in eval_history],
                'accuracy': [e['accuracy'] for e in eval_history],
                'f1_score': [e['f1_score'] for e in eval_history],
                'loss': [e['avg_loss'] for e in eval_history]
            }
    
    # Plot 1: Final Accuracy Comparison
    bars1 = axes[0, 0].bar(models, final_accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_title('Final Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, final_accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Final F1 Score Comparison
    bars2 = axes[0, 1].bar(models, final_f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 1].set_title('Final F1 Score Comparison')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, f1 in zip(bars2, final_f1_scores):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Training Curves - Accuracy
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (model_type, data) in enumerate(training_data.items()):
        axes[1, 0].plot(data['steps'], data['accuracy'], 
                       label=model_type.replace('_', ' ').title(), 
                       color=colors[i], linewidth=2, marker='o', markersize=4)
    
    axes[1, 0].set_title('Training Curves - Validation Accuracy')
    axes[1, 0].set_xlabel('Training Steps')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Training Curves - F1 Score
    for i, (model_type, data) in enumerate(training_data.items()):
        axes[1, 1].plot(data['steps'], data['f1_score'], 
                       label=model_type.replace('_', ' ').title(), 
                       color=colors[i], linewidth=2, marker='s', markersize=4)
    
    axes[1, 1].set_title('Training Curves - Validation F1 Score')
    axes[1, 1].set_xlabel('Training Steps')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comparison plot
    plot_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to: {plot_path}")
    
    # Create a summary table
    summary_df = pd.DataFrame({
        'Model': models,
        'Final Accuracy': final_accuracies,
        'Final F1 Score': final_f1_scores,
        'Best F1 Score': best_f1_scores
    })
    
    # Save summary table
    summary_path = os.path.join(save_dir, 'model_comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Summary table saved to: {summary_path}")
    print("\nModel Comparison Summary:")
    print(summary_df.to_string(index=False))
    
    return summary_df

def main():
    """Main comparison function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare training results from multiple models')
    parser.add_argument('--results_dir', type=str, default='./TASK2',
                       help='Directory containing training results')
    parser.add_argument('--save_dir', type=str, default='./model_comparison_results',
                       help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load results
    print("Loading training results...")
    results = load_results(args.results_dir)
    
    if not results:
        print("No training results found! Make sure the models have been trained.")
        return
    
    print(f"Found results for {len(results)} models: {list(results.keys())}")
    
    # Create comparison plots
    summary_df = create_comparison_plots(results, args.save_dir)
    
    # Save detailed comparison report
    report = {
        'comparison_timestamp': datetime.now().isoformat(),
        'models_compared': list(results.keys()),
        'summary': summary_df.to_dict('records'),
        'detailed_results': results
    }
    
    report_path = os.path.join(args.save_dir, 'detailed_comparison_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Detailed comparison report saved to: {report_path}")
    print("Comparison completed!")

if __name__ == "__main__":
    main()
