"""
Run focused hyperparameter experiments based on specific hypotheses

This script allows you to quickly test specific configurations or hypotheses
without running a full grid search.

Usage examples:
    # Test if dropout helps GCN
    python run_focused_experiment.py --model GCN --vary dropout --values 0.0 0.1 0.2 0.3

    # Test optimal hidden dimension for GraphSAGE
    python run_focused_experiment.py --model GraphSAGE --vary hidden_dim --values 128 256 512

    # Test multi-strategy decoder
    python run_focused_experiment.py --model GCN --vary use_multi_strategy --values False True

    # Test specific configuration
    python run_focused_experiment.py --model GCN --config '{"hidden_dim": 256, "dropout": 0.1}'
"""
import torch
import logging
import os
import json
import argparse
from datetime import datetime

from src.data import load_dataset
from src.models.baselines import GCN, GraphSAGE, GraphTransformer, GAT
from src.training import train_minimal_baseline
from src.evals import evaluate


# Default configuration
DEFAULT_CONFIG = {
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.0,
    'decoder_dropout': 0.0,
    'use_multi_strategy': False,
    'heads': 2,
    'use_batch_norm': False,
    'lr': 0.01,
    'weight_decay': 1e-4,
    'batch_size': 50000,
    'epochs': 200,
    'patience': 20,
    'eval_every': 5,
}


def parse_value(value_str):
    """Parse a string value to appropriate type"""
    # Boolean
    if value_str.lower() in ['true', 'false']:
        return value_str.lower() == 'true'
    # Try int
    try:
        return int(value_str)
    except ValueError:
        pass
    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass
    # Return as string
    return value_str


def create_configs_varying_param(param_name, values, base_config):
    """Create configurations varying a single parameter"""
    configs = []
    for value in values:
        config = base_config.copy()
        config[param_name] = value
        configs.append(config)
    return configs


def run_experiment(model_name, config, data, splits, evaluate_fn, device, logger):
    """Run a single experiment"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Running {model_name} with configuration:")
    for key, value in sorted(config.items()):
        logger.info(f"  {key}: {value}")
    logger.info("="*80)

    # Create model
    num_nodes = data.num_nodes
    model_classes = {
        'GCN': GCN,
        'GraphSAGE': GraphSAGE,
        'GAT': GAT,
        'GraphTransformer': GraphTransformer
    }

    model_class = model_classes[model_name]

    if model_name in ['GAT', 'GraphTransformer']:
        model = model_class(
            num_nodes,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            heads=config['heads'],
            dropout=config['dropout'],
            decoder_dropout=config['decoder_dropout'],
            use_multi_strategy=config['use_multi_strategy']
        )
    elif model_name == 'GraphSAGE':
        model = model_class(
            num_nodes,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            decoder_dropout=config['decoder_dropout'],
            use_multi_strategy=config['use_multi_strategy'],
            use_batch_norm=config['use_batch_norm']
        )
    else:  # GCN
        model = model_class(
            num_nodes,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            decoder_dropout=config['decoder_dropout'],
            use_multi_strategy=config['use_multi_strategy']
        )

    # Train
    result = train_minimal_baseline(
        f"{model_name}-Focused",
        model,
        data,
        splits['train_pos'],
        splits['valid_pos'],
        splits['valid_neg'],
        splits['test_pos'],
        splits['test_neg'],
        evaluate_fn,
        device=device,
        epochs=config['epochs'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        eval_every=config['eval_every'],
        patience=config['patience'],
        batch_size=config['batch_size'],
        eval_batch_size=config['batch_size']
    )

    return {
        'config': config,
        'best_val_hits': result.best_val_hits,
        'best_test_hits': result.best_test_hits,
        'best_epoch': result.best_epoch,
        'val_test_gap': result.best_val_hits - result.best_test_hits
    }


def main():
    parser = argparse.ArgumentParser(description='Run focused hyperparameter experiments')
    parser.add_argument('--model', required=True,
                       choices=['GCN', 'GraphSAGE', 'GAT', 'GraphTransformer'],
                       help='Model to test')
    parser.add_argument('--vary', help='Parameter to vary')
    parser.add_argument('--values', nargs='+', help='Values to test for the varied parameter')
    parser.add_argument('--config', help='JSON string with specific configuration to test')
    parser.add_argument('--base-config', help='JSON file with base configuration',
                       default=None)

    args = parser.parse_args()

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'logs/focused_experiment_{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f'{log_dir}/experiment.log'

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
    logger.info(f"Logging to: {log_filename}")

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

    def evaluate_fn(model, pos_edges, neg_edges, batch_size):
        return evaluate(model, data, evaluator, pos_edges, neg_edges, batch_size)

    # Load base configuration
    base_config = DEFAULT_CONFIG.copy()
    if args.base_config:
        with open(args.base_config, 'r') as f:
            custom_base = json.load(f)
            base_config.update(custom_base)

    # Create configurations to test
    configs = []

    if args.config:
        # Single specific configuration
        custom_config = json.loads(args.config)
        base_config.update(custom_config)
        configs = [base_config]
        logger.info("Testing single configuration")

    elif args.vary and args.values:
        # Vary a single parameter
        values = [parse_value(v) for v in args.values]
        configs = create_configs_varying_param(args.vary, values, base_config)
        logger.info(f"Testing {len(configs)} configurations varying {args.vary}")
        logger.info(f"Values: {values}")

    else:
        logger.error("Must specify either --config or both --vary and --values")
        return

    # Run experiments
    results = []
    for i, config in enumerate(configs, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"Experiment {i}/{len(configs)}")
        logger.info(f"{'#'*80}")

        result = run_experiment(args.model, config, data, splits, evaluate_fn, device, logger)
        results.append(result)

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)

    if args.vary and len(results) > 1:
        logger.info(f"\nParameter: {args.vary}")
        logger.info(f"{'Value':<15} {'Test Hits@20':<15} {'Val Hits@20':<15} {'Gap':<10}")
        logger.info("-" * 60)
        for r in results:
            value = r['config'][args.vary]
            logger.info(f"{str(value):<15} {r['best_test_hits']:<15.4f} "
                       f"{r['best_val_hits']:<15.4f} {r['val_test_gap']:<10.4f}")

        # Best configuration
        best = max(results, key=lambda x: x['best_test_hits'])
        logger.info(f"\nBest {args.vary}: {best['config'][args.vary]}")
        logger.info(f"  Test Hits@20: {best['best_test_hits']:.4f}")
        logger.info(f"  Val Hits@20: {best['best_val_hits']:.4f}")

    else:
        # Single config result
        r = results[0]
        logger.info(f"\nTest Hits@20: {r['best_test_hits']:.4f}")
        logger.info(f"Val Hits@20: {r['best_val_hits']:.4f}")
        logger.info(f"Val-Test Gap: {r['val_test_gap']:.4f}")
        logger.info(f"Best Epoch: {r['best_epoch']}")

    # Save results
    results_file = os.path.join(log_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Log saved to: {log_filename}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
