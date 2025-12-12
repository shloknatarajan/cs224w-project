"""
Hyperparameter search for minimal baseline models

This script systematically explores hyperparameter combinations to find optimal
configurations for GCN, GraphSAGE, GAT, and GraphTransformer models.

Search Strategy:
- Grid search over predefined hyperparameter ranges
- Logs all configurations and results
- Can be run in parallel or sequentially
- Focuses on most impactful hyperparameters first

Usage:
  python tune_baselines.py                    # Run GCN only (default)
  python tune_baselines.py --all-models       # Run all models
  python tune_baselines.py --search-type standard  # Use different search space
"""
import torch
import logging
import os
import json
import argparse
from datetime import datetime
from itertools import product
from dataclasses import dataclass, asdict

from src.data import load_dataset
from src.models.baselines import GCN, GraphSAGE, GraphTransformer, GAT
from src.training import train_minimal_baseline
from src.evals import evaluate


@dataclass
class HyperparamConfig:
    """Configuration for a single hyperparameter search run"""
    # Model architecture
    hidden_dim: int
    num_layers: int
    dropout: float
    decoder_dropout: float
    use_multi_strategy: bool

    # Model-specific
    heads: int = 2  # For GAT/Transformer
    use_batch_norm: bool = False  # For GraphSAGE

    # Training
    lr: float = 0.01
    weight_decay: float = 1e-4
    batch_size: int = 50000

    # Fixed params
    epochs: int = 200
    patience: int = 20
    eval_every: int = 5


@dataclass
class SearchResult:
    """Result from a single hyperparameter configuration"""
    model_name: str
    config: HyperparamConfig
    best_val_hits: float
    best_test_hits: float
    best_epoch: int
    val_test_gap: float


def create_search_space(search_type='minimal'):
    """
    Create hyperparameter search space

    Args:
        search_type: 'minimal', 'standard', or 'extensive'
    """
    if search_type == 'minimal':
        # Focused search on proven high-performers (based on DDI sweep results)
        # Generates ~24 configs for GCN (down from 72)
        return {
            'hidden_dim': [128, 256, 512],  # Top 2 from DDI sweeps (256=best, 192=second)
            'num_layers': [2, 3],
            'dropout': [0.0, 0.2],  # Best no-dropout, middle ground, best with dropout
            'decoder_dropout': [0.0],
            'use_multi_strategy': [False],
            'heads': [2, 4],  # GAT/Transformer only
            'use_batch_norm': [False, True],  # GraphSAGE only
            'lr': [0.001, 0.005],  # 0.005 best for 2-layer, 0.001 best for 3-layer
            'weight_decay': [1e-4],  # Both 1e-4 and 1e-3 worked; focus on 1e-4
            'batch_size': [50000],  # Best batch size from sweeps (64k)
        }
    elif search_type == 'standard':
        # Balanced search
        return {
            'hidden_dim': [128, 256, 512],
            'num_layers': [2, 3, 4],
            'dropout': [0.0, 0.1, 0.2, 0.3],
            'decoder_dropout': [0.0, 0.1],
            'use_multi_strategy': [False, True],
            'heads': [2, 4, 8],
            'use_batch_norm': [False, True],
            'lr': [0.001, 0.005, 0.01, 0.02],
            'weight_decay': [1e-5, 1e-4, 1e-3],
            'batch_size': [25000, 50000, 100000],
        }
    else:  # extensive
        # Comprehensive search
        return {
            'hidden_dim': [128, 256, 512],
            'num_layers': [2, 3, 4, 5],
            'dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
            'decoder_dropout': [0.0, 0.05, 0.1, 0.2],
            'use_multi_strategy': [False, True],
            'heads': [2, 4, 8, 16],
            'use_batch_norm': [False, True],
            'lr': [0.0005, 0.001, 0.005, 0.01, 0.02],
            'weight_decay': [0.0, 1e-5, 1e-4, 1e-3],
            'batch_size': [25000, 50000, 100000],
        }


def generate_configs(search_space, model_type='GCN'):
    """
    Generate all hyperparameter configurations for a model type

    Args:
        search_space: Dictionary of hyperparameter ranges
        model_type: 'GCN', 'GraphSAGE', 'GAT', or 'GraphTransformer'
    """
    # Determine which parameters are relevant for this model
    if model_type in ['GAT', 'GraphTransformer']:
        # Use heads parameter
        param_names = ['hidden_dim', 'num_layers', 'dropout', 'decoder_dropout',
                      'use_multi_strategy', 'heads', 'lr', 'weight_decay', 'batch_size']
    elif model_type == 'GraphSAGE':
        # Use batch_norm parameter
        param_names = ['hidden_dim', 'num_layers', 'dropout', 'decoder_dropout',
                      'use_multi_strategy', 'use_batch_norm', 'lr', 'weight_decay', 'batch_size']
    else:  # GCN
        # Basic parameters only
        param_names = ['hidden_dim', 'num_layers', 'dropout', 'decoder_dropout',
                      'use_multi_strategy', 'lr', 'weight_decay', 'batch_size']

    # Generate all combinations
    param_values = [search_space[name] for name in param_names]

    configs = []
    for values in product(*param_values):
        config_dict = dict(zip(param_names, values))
        # Set defaults for unused parameters
        if 'heads' not in config_dict:
            config_dict['heads'] = 2
        if 'use_batch_norm' not in config_dict:
            config_dict['use_batch_norm'] = False
        configs.append(HyperparamConfig(**config_dict))

    return configs


def run_experiment(model_name, model_class, config, data, splits, evaluate_fn, device, logger):
    """Run a single experiment with given configuration"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing {model_name} with config:")
    logger.info(f"  Architecture: hidden={config.hidden_dim}, layers={config.num_layers}, "
               f"dropout={config.dropout}, decoder_dropout={config.decoder_dropout}")
    if hasattr(config, 'heads') and model_name in ['GAT', 'GraphTransformer']:
        logger.info(f"  Heads: {config.heads}")
    if hasattr(config, 'use_batch_norm') and model_name == 'GraphSAGE':
        logger.info(f"  Batch norm: {config.use_batch_norm}")
    logger.info(f"  Multi-strategy: {config.use_multi_strategy}")
    logger.info(f"  Training: lr={config.lr}, wd={config.weight_decay}, bs={config.batch_size}")
    logger.info("="*80)

    # Create model based on type
    num_nodes = data.num_nodes
    if model_name in ['GAT', 'GraphTransformer']:
        model = model_class(
            num_nodes,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            heads=config.heads,
            dropout=config.dropout,
            decoder_dropout=config.decoder_dropout,
            use_multi_strategy=config.use_multi_strategy
        )
    elif model_name == 'GraphSAGE':
        model = model_class(
            num_nodes,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            decoder_dropout=config.decoder_dropout,
            use_multi_strategy=config.use_multi_strategy,
            use_batch_norm=config.use_batch_norm
        )
    else:  # GCN
        model = model_class(
            num_nodes,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            decoder_dropout=config.decoder_dropout,
            use_multi_strategy=config.use_multi_strategy
        )

    # Train model
    try:
        result = train_minimal_baseline(
            f"{model_name}-Tuning",
            model,
            data,
            splits['train_pos'],
            splits['valid_pos'],
            splits['valid_neg'],
            splits['test_pos'],
            splits['test_neg'],
            evaluate_fn,
            device=device,
            epochs=config.epochs,
            lr=config.lr,
            weight_decay=config.weight_decay,
            eval_every=config.eval_every,
            patience=config.patience,
            batch_size=config.batch_size,
            eval_batch_size=config.batch_size
        )

        val_test_gap = result.best_val_hits - result.best_test_hits

        return SearchResult(
            model_name=model_name,
            config=config,
            best_val_hits=result.best_val_hits,
            best_test_hits=result.best_test_hits,
            best_epoch=result.best_epoch,
            val_test_gap=val_test_gap
        )
    except Exception as e:
        logger.error(f"Error training {model_name} with config: {e}")
        return None


def save_results(results, log_dir):
    """Save all results to JSON file"""
    results_data = []
    for r in results:
        if r is not None:
            result_dict = {
                'model_name': r.model_name,
                'config': asdict(r.config),
                'best_val_hits': r.best_val_hits,
                'best_test_hits': r.best_test_hits,
                'best_epoch': r.best_epoch,
                'val_test_gap': r.val_test_gap
            }
            results_data.append(result_dict)

    results_file = os.path.join(log_dir, 'hyperparam_search_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    return results_file


def analyze_results(results, logger):
    """Analyze and report best configurations"""
    if not results:
        logger.warning("No results to analyze!")
        return

    # Filter out None results
    results = [r for r in results if r is not None]

    logger.info("\n" + "="*80)
    logger.info("HYPERPARAMETER SEARCH RESULTS SUMMARY")
    logger.info("="*80)

    # Best overall
    best_overall = max(results, key=lambda x: x.best_test_hits)
    logger.info(f"\nBest Overall Configuration:")
    logger.info(f"  Model: {best_overall.model_name}")
    logger.info(f"  Test Hits@20: {best_overall.best_test_hits:.4f}")
    logger.info(f"  Val Hits@20: {best_overall.best_val_hits:.4f}")
    logger.info(f"  Config: {asdict(best_overall.config)}")

    # Best per model
    logger.info("\nBest Configuration Per Model:")
    models = set(r.model_name for r in results)
    for model in sorted(models):
        model_results = [r for r in results if r.model_name == model]
        if model_results:
            best = max(model_results, key=lambda x: x.best_test_hits)
            logger.info(f"\n  {model}:")
            logger.info(f"    Test Hits@20: {best.best_test_hits:.4f}")
            logger.info(f"    Val Hits@20: {best.best_val_hits:.4f}")
            logger.info(f"    Hidden dim: {best.config.hidden_dim}, Layers: {best.config.num_layers}")
            logger.info(f"    Dropout: {best.config.dropout}, Decoder dropout: {best.config.decoder_dropout}")
            logger.info(f"    LR: {best.config.lr}, Weight decay: {best.config.weight_decay}")
            if model in ['GAT', 'GraphTransformer']:
                logger.info(f"    Heads: {best.config.heads}")
            if model == 'GraphSAGE':
                logger.info(f"    Batch norm: {best.config.use_batch_norm}")

    # Top 5 configurations
    logger.info("\nTop 5 Configurations Overall:")
    top_5 = sorted(results, key=lambda x: x.best_test_hits, reverse=True)[:5]
    for i, r in enumerate(top_5, 1):
        logger.info(f"\n  {i}. {r.model_name} - Test Hits@20: {r.best_test_hits:.4f}")
        logger.info(f"     hidden_dim={r.config.hidden_dim}, layers={r.config.num_layers}, "
                   f"dropout={r.config.dropout}, lr={r.config.lr}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Hyperparameter search for baseline models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tune_baselines.py                          # Run GCN only (default)
  python tune_baselines.py --all-models             # Run all models
  python tune_baselines.py --search-type standard   # Standard search space
        """
    )
    parser.add_argument(
        '--all-models',
        action='store_true',
        help='Run hyperparameter search for all models (GCN, GraphSAGE, GAT, GraphTransformer). Default: GCN only'
    )
    parser.add_argument(
        '--search-type',
        choices=['minimal', 'standard', 'extensive'],
        default='minimal',
        help='Search space size (default: minimal)'
    )
    args = parser.parse_args()

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'logs/hyperparam_search_{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f'{log_dir}/search.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Logging results to: {log_filename}")

    # Load dataset
    logger.info("Loading dataset...")
    data, split_edge, num_nodes, evaluator = load_dataset('ogbl-ddi', device=device)

    splits = {
        'train_pos': split_edge['train']['edge'].to(device),
        'valid_pos': split_edge['valid']['edge'].to(device),
        'valid_neg': split_edge['valid']['edge_neg'].to(device),
        'test_pos': split_edge['test']['edge'].to(device),
        'test_neg': split_edge['test']['edge_neg'].to(device),
    }

    # Create evaluation function wrapper
    def evaluate_fn(model, pos_edges, neg_edges, batch_size):
        return evaluate(model, data, evaluator, pos_edges, neg_edges, batch_size)

    # Configuration
    SEARCH_TYPE = args.search_type
    if args.all_models:
        MODELS_TO_SEARCH = ['GCN', 'GraphSAGE', 'GAT', 'GraphTransformer']
    else:
        MODELS_TO_SEARCH = ['GCN']

    logger.info(f"\nRunning {SEARCH_TYPE} hyperparameter search")
    logger.info(f"Models: {MODELS_TO_SEARCH}")

    # Generate search space
    search_space = create_search_space(SEARCH_TYPE)

    # Model mapping
    model_classes = {
        'GCN': GCN,
        'GraphSAGE': GraphSAGE,
        'GAT': GAT,
        'GraphTransformer': GraphTransformer
    }

    # Run experiments
    all_results = []
    for model_name in MODELS_TO_SEARCH:
        logger.info(f"\n{'#'*80}")
        logger.info(f"Searching hyperparameters for {model_name}")
        logger.info(f"{'#'*80}")

        model_class = model_classes[model_name]
        configs = generate_configs(search_space, model_name)

        logger.info(f"Generated {len(configs)} configurations to test")

        for i, config in enumerate(configs, 1):
            logger.info(f"\nConfiguration {i}/{len(configs)}")
            result = run_experiment(
                model_name, model_class, config, data, splits,
                evaluate_fn, device, logger
            )
            if result:
                all_results.append(result)

    # Save and analyze results
    results_file = save_results(all_results, log_dir)
    logger.info(f"\nResults saved to: {results_file}")

    analyze_results(all_results, logger)

    logger.info(f"\n{'='*80}")
    logger.info("Hyperparameter search complete!")
    logger.info(f"Full results: {results_file}")
    logger.info(f"Log file: {log_filename}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
