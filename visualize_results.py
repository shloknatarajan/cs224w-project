"""
Visualize hyperparameter search results

Creates plots to understand parameter impact and performance patterns.

Usage:
    python visualize_results.py logs/hyperparam_search_TIMESTAMP/hyperparam_search_results.json

Requires: matplotlib, seaborn (install with: pip install matplotlib seaborn)
"""
import json
import sys
import argparse
import os
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
except ImportError:
    print("Error: matplotlib and seaborn are required for visualization")
    print("Install with: pip install matplotlib seaborn")
    sys.exit(1)


def load_results(json_file):
    """Load results from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)


def plot_parameter_impact(results, param_name, output_dir):
    """Create box plot showing parameter impact on performance"""
    param_groups = defaultdict(list)

    for r in results:
        param_value = r['config'][param_name]
        param_groups[param_value].append(r['best_test_hits'])

    # Sort by parameter value
    sorted_params = sorted(param_groups.keys())
    data = [param_groups[p] for p in sorted_params]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=[str(p) for p in sorted_params], patch_artist=True)

    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Test Hits@20', fontsize=12)
    ax.set_title(f'Impact of {param_name} on Performance', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'impact_{param_name}.png'), dpi=150)
    plt.close()

    print(f"  Created: impact_{param_name}.png")


def plot_performance_distribution(results, output_dir):
    """Plot distribution of test performance"""
    test_scores = [r['best_test_hits'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(test_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(test_scores), color='red', linestyle='--',
               label=f'Mean: {np.mean(test_scores):.4f}')
    ax1.axvline(np.median(test_scores), color='green', linestyle='--',
               label=f'Median: {np.median(test_scores):.4f}')
    ax1.set_xlabel('Test Hits@20', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Performance Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(test_scores, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue'))
    ax2.set_ylabel('Test Hits@20', fontsize=12)
    ax2.set_title('Performance Box Plot', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_distribution.png'), dpi=150)
    plt.close()

    print(f"  Created: performance_distribution.png")


def plot_model_comparison(results, output_dir):
    """Compare performance across different models"""
    model_groups = defaultdict(list)

    for r in results:
        model_groups[r['model_name']].append(r['best_test_hits'])

    if len(model_groups) < 2:
        print(f"  Skipped: model_comparison.png (only one model type)")
        return

    models = sorted(model_groups.keys())
    data = [model_groups[m] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=models, patch_artist=True)

    # Color boxes differently
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Test Hits@20', fontsize=12)
    ax.set_title('Model Architecture Comparison', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150)
    plt.close()

    print(f"  Created: model_comparison.png")


def plot_val_test_gap(results, output_dir):
    """Plot validation-test gap analysis"""
    test_scores = [r['best_test_hits'] for r in results]
    gaps = [r['val_test_gap'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: test performance vs gap
    ax1.scatter(test_scores, gaps, alpha=0.5, c='blue')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Test Hits@20', fontsize=12)
    ax1.set_ylabel('Val-Test Gap', fontsize=12)
    ax1.set_title('Generalization Analysis', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Histogram of gaps
    ax2.hist(gaps, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(gaps), color='red', linestyle='--',
               label=f'Mean: {np.mean(gaps):.4f}')
    ax2.axvline(0, color='green', linestyle='--', alpha=0.5, label='Perfect generalization')
    ax2.set_xlabel('Val-Test Gap', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Gap Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generalization_analysis.png'), dpi=150)
    plt.close()

    print(f"  Created: generalization_analysis.png")


def plot_pareto_frontier(results, output_dir):
    """Plot Pareto frontier: performance vs generalization"""
    test_scores = [r['best_test_hits'] for r in results]
    gaps = [r['val_test_gap'] for r in results]

    # Find Pareto-optimal points (high test, low gap)
    pareto_points = []
    for i, (score, gap) in enumerate(zip(test_scores, gaps)):
        is_dominated = False
        for j, (other_score, other_gap) in enumerate(zip(test_scores, gaps)):
            if i != j and other_score >= score and other_gap <= gap and \
               (other_score > score or other_gap < gap):
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append(i)

    fig, ax = plt.subplots(figsize=(10, 6))

    # All points
    ax.scatter(test_scores, gaps, alpha=0.4, c='lightblue', s=50, label='All configs')

    # Pareto frontier
    pareto_scores = [test_scores[i] for i in pareto_points]
    pareto_gaps = [gaps[i] for i in pareto_points]
    ax.scatter(pareto_scores, pareto_gaps, alpha=0.8, c='red', s=100,
              marker='*', label='Pareto optimal', zorder=5)

    ax.set_xlabel('Test Hits@20 (higher is better)', fontsize=12)
    ax.set_ylabel('Val-Test Gap (lower is better)', fontsize=12)
    ax.set_title('Pareto Frontier: Performance vs Generalization', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_frontier.png'), dpi=150)
    plt.close()

    print(f"  Created: pareto_frontier.png")


def plot_training_efficiency(results, output_dir):
    """Plot training efficiency: performance vs epochs to convergence"""
    test_scores = [r['best_test_hits'] for r in results]
    epochs = [r['best_epoch'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(epochs, test_scores, alpha=0.5, c=test_scores,
                        cmap='viridis', s=50)
    ax.set_xlabel('Epochs to Best Validation', fontsize=12)
    ax.set_ylabel('Test Hits@20', fontsize=12)
    ax.set_title('Training Efficiency', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label='Test Hits@20')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_efficiency.png'), dpi=150)
    plt.close()

    print(f"  Created: training_efficiency.png")


def plot_parameter_heatmap(results, param1, param2, output_dir):
    """Create heatmap showing interaction between two parameters"""
    # Create grid of mean performance
    param1_values = sorted(set(r['config'][param1] for r in results))
    param2_values = sorted(set(r['config'][param2] for r in results))

    grid = np.zeros((len(param2_values), len(param1_values)))
    counts = np.zeros((len(param2_values), len(param1_values)))

    for r in results:
        i = param2_values.index(r['config'][param2])
        j = param1_values.index(r['config'][param1])
        grid[i, j] += r['best_test_hits']
        counts[i, j] += 1

    # Average
    with np.errstate(divide='ignore', invalid='ignore'):
        grid = np.where(counts > 0, grid / counts, np.nan)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(grid, annot=True, fmt='.3f', cmap='YlOrRd',
               xticklabels=[str(v) for v in param1_values],
               yticklabels=[str(v) for v in param2_values],
               ax=ax, cbar_kws={'label': 'Mean Test Hits@20'})

    ax.set_xlabel(param1, fontsize=12)
    ax.set_ylabel(param2, fontsize=12)
    ax.set_title(f'Parameter Interaction: {param1} vs {param2}', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmap_{param1}_vs_{param2}.png'), dpi=150)
    plt.close()

    print(f"  Created: heatmap_{param1}_vs_{param2}.png")


def create_all_plots(results, output_dir):
    """Create all visualization plots"""
    print(f"\nCreating visualizations in {output_dir}...")

    # Performance distribution
    plot_performance_distribution(results, output_dir)

    # Model comparison (if multiple models)
    plot_model_comparison(results, output_dir)

    # Generalization analysis
    plot_val_test_gap(results, output_dir)

    # Pareto frontier
    plot_pareto_frontier(results, output_dir)

    # Training efficiency
    plot_training_efficiency(results, output_dir)

    # Parameter impact plots
    key_params = ['hidden_dim', 'num_layers', 'dropout', 'lr', 'weight_decay']
    for param in key_params:
        # Check if parameter varies in results
        values = set(r['config'].get(param) for r in results)
        if len(values) > 1:
            plot_parameter_impact(results, param, output_dir)

    # Heatmaps for parameter interactions
    param_pairs = [
        ('hidden_dim', 'dropout'),
        ('num_layers', 'dropout'),
        ('lr', 'weight_decay'),
    ]

    for param1, param2 in param_pairs:
        # Check if both parameters vary
        values1 = set(r['config'].get(param1) for r in results)
        values2 = set(r['config'].get(param2) for r in results)
        if len(values1) > 1 and len(values2) > 1:
            try:
                plot_parameter_heatmap(results, param1, param2, output_dir)
            except Exception as e:
                print(f"  Skipped: heatmap_{param1}_vs_{param2}.png ({e})")

    print(f"\nAll visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize hyperparameter search results')
    parser.add_argument('json_file', help='Path to hyperparam_search_results.json')
    parser.add_argument('--output-dir', '-o', help='Output directory for plots',
                       default=None)

    args = parser.parse_args()

    # Load results
    try:
        results = load_results(args.json_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.json_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file: {args.json_file}")
        sys.exit(1)

    if not results:
        print("Error: No results found in file")
        sys.exit(1)

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use same directory as input file
        output_dir = os.path.join(os.path.dirname(args.json_file), 'plots')

    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("HYPERPARAMETER SEARCH VISUALIZATION")
    print("="*80)
    print(f"Results file: {args.json_file}")
    print(f"Total configurations: {len(results)}")
    print(f"Output directory: {output_dir}")

    # Create all plots
    create_all_plots(results, output_dir)

    print("\n" + "="*80)
    print("Visualization complete!")
    print(f"View plots in: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
