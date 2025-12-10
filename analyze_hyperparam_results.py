"""
Analyze hyperparameter search results

Usage:
    python analyze_hyperparam_results.py logs/hyperparam_search_TIMESTAMP/hyperparam_search_results.json
"""
import json
import sys
import argparse
from collections import defaultdict
import statistics


def load_results(json_file):
    """Load results from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)


def analyze_parameter_impact(results, param_name):
    """Analyze the impact of a single parameter on performance"""
    param_groups = defaultdict(list)

    for r in results:
        param_value = r['config'][param_name]
        param_groups[param_value].append(r['best_test_hits'])

    # Calculate statistics for each parameter value
    stats = {}
    for value, scores in param_groups.items():
        stats[value] = {
            'mean': statistics.mean(scores),
            'median': statistics.median(scores),
            'std': statistics.stdev(scores) if len(scores) > 1 else 0.0,
            'min': min(scores),
            'max': max(scores),
            'count': len(scores)
        }

    return stats


def find_best_configs(results, n=5, metric='best_test_hits'):
    """Find top N configurations by metric"""
    sorted_results = sorted(results, key=lambda x: x[metric], reverse=True)
    return sorted_results[:n]


def analyze_generalization(results):
    """Analyze val-test gap patterns"""
    gaps = [r['val_test_gap'] for r in results]

    return {
        'mean_gap': statistics.mean(gaps),
        'median_gap': statistics.median(gaps),
        'std_gap': statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
        'min_gap': min(gaps),
        'max_gap': max(gaps),
    }


def find_best_generalizing_configs(results, n=5):
    """Find configs with best generalization (small val-test gap but good performance)"""
    # Filter for reasonable performance (test hits > 10%)
    good_performers = [r for r in results if r['best_test_hits'] > 0.10]

    if not good_performers:
        return []

    # Sort by val-test gap (smaller is better)
    sorted_results = sorted(good_performers, key=lambda x: x['val_test_gap'])
    return sorted_results[:n]


def analyze_by_model(results):
    """Group and analyze results by model type"""
    model_groups = defaultdict(list)

    for r in results:
        model_groups[r['model_name']].append(r)

    model_stats = {}
    for model, model_results in model_groups.items():
        test_scores = [r['best_test_hits'] for r in model_results]
        val_scores = [r['best_val_hits'] for r in model_results]
        gaps = [r['val_test_gap'] for r in model_results]

        best = max(model_results, key=lambda x: x['best_test_hits'])

        model_stats[model] = {
            'count': len(model_results),
            'test_mean': statistics.mean(test_scores),
            'test_std': statistics.stdev(test_scores) if len(test_scores) > 1 else 0.0,
            'test_max': max(test_scores),
            'test_min': min(test_scores),
            'val_mean': statistics.mean(val_scores),
            'gap_mean': statistics.mean(gaps),
            'gap_std': statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
            'best_config': best
        }

    return model_stats


def print_analysis(results):
    """Print comprehensive analysis"""
    print("=" * 80)
    print("HYPERPARAMETER SEARCH ANALYSIS")
    print("=" * 80)
    print(f"\nTotal configurations tested: {len(results)}")

    # Overall statistics
    test_scores = [r['best_test_hits'] for r in results]
    print(f"\nOverall Performance:")
    print(f"  Mean Test Hits@20: {statistics.mean(test_scores):.4f}")
    print(f"  Std Dev: {statistics.stdev(test_scores) if len(test_scores) > 1 else 0:.4f}")
    print(f"  Best: {max(test_scores):.4f}")
    print(f"  Worst: {min(test_scores):.4f}")

    # Generalization analysis
    print(f"\nGeneralization (Val-Test Gap):")
    gen_stats = analyze_generalization(results)
    print(f"  Mean Gap: {gen_stats['mean_gap']:.4f}")
    print(f"  Std Dev: {gen_stats['std_gap']:.4f}")
    print(f"  Min Gap: {gen_stats['min_gap']:.4f}")
    print(f"  Max Gap: {gen_stats['max_gap']:.4f}")

    # Model-specific analysis
    print(f"\n" + "=" * 80)
    print("MODEL-SPECIFIC ANALYSIS")
    print("=" * 80)
    model_stats = analyze_by_model(results)

    for model, stats in sorted(model_stats.items()):
        print(f"\n{model}:")
        print(f"  Configurations tested: {stats['count']}")
        print(f"  Test Performance:")
        print(f"    Mean: {stats['test_mean']:.4f} ± {stats['test_std']:.4f}")
        print(f"    Range: [{stats['test_min']:.4f}, {stats['test_max']:.4f}]")
        print(f"  Val-Test Gap: {stats['gap_mean']:.4f} ± {stats['gap_std']:.4f}")
        print(f"  Best Config:")
        best_cfg = stats['best_config']['config']
        print(f"    Test Hits@20: {stats['best_config']['best_test_hits']:.4f}")
        print(f"    Val Hits@20: {stats['best_config']['best_val_hits']:.4f}")
        print(f"    hidden_dim={best_cfg['hidden_dim']}, layers={best_cfg['num_layers']}")
        print(f"    dropout={best_cfg['dropout']}, decoder_dropout={best_cfg['decoder_dropout']}")
        print(f"    lr={best_cfg['lr']}, weight_decay={best_cfg['weight_decay']}")
        if model in ['GAT', 'GraphTransformer']:
            print(f"    heads={best_cfg['heads']}")
        if model == 'GraphSAGE':
            print(f"    batch_norm={best_cfg['use_batch_norm']}")

    # Top configurations
    print(f"\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS (by Test Hits@20)")
    print("=" * 80)
    top_configs = find_best_configs(results, n=10)

    for i, r in enumerate(top_configs, 1):
        print(f"\n{i}. {r['model_name']} - Test Hits@20: {r['best_test_hits']:.4f}")
        print(f"   Val Hits@20: {r['best_val_hits']:.4f} | Gap: {r['val_test_gap']:.4f}")
        cfg = r['config']
        print(f"   Config: hidden={cfg['hidden_dim']}, layers={cfg['num_layers']}, "
              f"dropout={cfg['dropout']}, lr={cfg['lr']}, wd={cfg['weight_decay']}")
        if r['model_name'] in ['GAT', 'GraphTransformer']:
            print(f"   heads={cfg['heads']}, multi_strategy={cfg['use_multi_strategy']}")
        elif r['model_name'] == 'GraphSAGE':
            print(f"   batch_norm={cfg['use_batch_norm']}, multi_strategy={cfg['use_multi_strategy']}")

    # Best generalizing configs
    print(f"\n" + "=" * 80)
    print("TOP 5 BEST GENERALIZING CONFIGS (small val-test gap + good performance)")
    print("=" * 80)
    gen_configs = find_best_generalizing_configs(results, n=5)

    for i, r in enumerate(gen_configs, 1):
        print(f"\n{i}. {r['model_name']} - Gap: {r['val_test_gap']:.4f}")
        print(f"   Test Hits@20: {r['best_test_hits']:.4f} | Val Hits@20: {r['best_val_hits']:.4f}")
        cfg = r['config']
        print(f"   Config: hidden={cfg['hidden_dim']}, layers={cfg['num_layers']}, "
              f"dropout={cfg['dropout']}, lr={cfg['lr']}")

    # Parameter impact analysis
    print(f"\n" + "=" * 80)
    print("PARAMETER IMPACT ANALYSIS")
    print("=" * 80)

    key_params = ['hidden_dim', 'num_layers', 'dropout', 'lr', 'weight_decay']

    for param in key_params:
        print(f"\n{param.upper()}:")
        stats = analyze_parameter_impact(results, param)

        # Sort by mean performance
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True)

        for value, stat in sorted_stats:
            print(f"  {value:>10}: mean={stat['mean']:.4f}, "
                  f"std={stat['std']:.4f}, range=[{stat['min']:.4f}, {stat['max']:.4f}], "
                  f"n={stat['count']}")

    # Check for multi-strategy impact (if tested)
    if any('use_multi_strategy' in r['config'] for r in results):
        print(f"\nMULTI-STRATEGY DECODER:")
        ms_stats = analyze_parameter_impact(results, 'use_multi_strategy')
        for value, stat in sorted(ms_stats.items(), key=lambda x: x[1]['mean'], reverse=True):
            print(f"  {value}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, n={stat['count']}")

    print(f"\n" + "=" * 80)


def export_best_configs(results, output_file='best_configs.json'):
    """Export best configurations for each model to a file"""
    model_stats = analyze_by_model(results)

    best_configs = {}
    for model, stats in model_stats.items():
        best = stats['best_config']
        best_configs[model] = {
            'test_hits': best['best_test_hits'],
            'val_hits': best['best_val_hits'],
            'val_test_gap': best['val_test_gap'],
            'config': best['config']
        }

    with open(output_file, 'w') as f:
        json.dump(best_configs, f, indent=2)

    print(f"\nBest configurations exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter search results')
    parser.add_argument('json_file', help='Path to hyperparam_search_results.json')
    parser.add_argument('--export', '-e', help='Export best configs to file',
                       metavar='FILE', default=None)

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

    # Print analysis
    print_analysis(results)

    # Export best configs if requested
    if args.export:
        export_best_configs(results, args.export)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
