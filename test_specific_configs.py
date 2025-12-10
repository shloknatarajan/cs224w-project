"""
Test Specific Hyperparameter Configurations

Use this script to test specific configurations you want to try,
either based on intuition or from tuning results.

Simply add configurations to the CONFIGS_TO_TEST list below and run.
"""
import os
import logging
import json
import torch
from datetime import datetime

from src.data.data_loader import load_dataset_chemberta
from src.models.hybrid import HybridGCN
from src.training.hybrid_trainer import train_hybrid_model

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/specific_configs_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "specific_configs.log")
results_file = os.path.join(log_dir, "results.json")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load dataset
logger.info("Loading dataset...")
data, split_edge, num_nodes, evaluator = load_dataset_chemberta(
    dataset_name="ogbl-ddi",
    device=device,
    smiles_csv_path="data/smiles.csv",
    feature_cache_path="data/chemberta_features_768.pt",
    chemberta_model="seyonec/ChemBERTa-zinc-base-v1",
    batch_size=32,
)
logger.info(f"Dataset loaded: {data.x.shape}")

# Baseline for comparison
BASELINE = {
    'name': 'Current Baseline',
    'val_hits': 0.2349,
    'test_hits': 0.1933,
}

# ============================================================================
# CONFIGURATIONS TO TEST
# ============================================================================
# Add your configurations here. Each config should be a dictionary with all
# required parameters. You can copy the baseline and modify specific values.

CONFIGS_TO_TEST = [
    {
        'name': 'Baseline (reproduce)',
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.0,
        'decoder_dropout': 0.3,
        'lr': 0.01,
        'weight_decay': 1e-4,
        'batch_size': 5000,
        'epochs': 200,
    },
    {
        'name': 'Larger model',
        'hidden_dim': 256,  # Increase capacity
        'num_layers': 3,    # Deeper network
        'dropout': 0.0,
        'decoder_dropout': 0.3,
        'lr': 0.01,
        'weight_decay': 1e-4,
        'batch_size': 5000,
        'epochs': 200,
    },
    {
        'name': 'Higher learning rate',
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.0,
        'decoder_dropout': 0.3,
        'lr': 0.02,         # Double the lr
        'weight_decay': 2e-4,  # Also increase regularization
        'batch_size': 5000,
        'epochs': 200,
    },
    {
        'name': 'Lower learning rate',
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.0,
        'decoder_dropout': 0.3,
        'lr': 0.005,        # Half the lr
        'weight_decay': 5e-5,  # Reduce regularization
        'batch_size': 5000,
        'epochs': 200,
    },
    {
        'name': 'More regularization',
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.1,     # Add encoder dropout
        'decoder_dropout': 0.4,  # Increase decoder dropout
        'lr': 0.01,
        'weight_decay': 5e-4,  # Stronger weight decay
        'batch_size': 5000,
        'epochs': 200,
    },
    {
        'name': 'Larger batches',
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.0,
        'decoder_dropout': 0.3,
        'lr': 0.015,        # Slightly higher lr for larger batches
        'weight_decay': 1e-4,
        'batch_size': 20000,  # 4x larger batches
        'epochs': 200,
    },
]

# Fixed parameters (same for all configs)
FIXED_PARAMS = {
    'eval_every': 5,
    'patience': 20,
    'chemical_dim': 768,
    'decoder_type': 'simple',
    'use_gating': False,
}


def test_config(config_id, config):
    """Test a single configuration."""
    name = config.get('name', f'Config-{config_id}')

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Testing: {name}")
    logger.info("=" * 80)

    # Log configuration
    logger.info("Configuration:")
    for key in ['hidden_dim', 'num_layers', 'dropout', 'decoder_dropout',
                'lr', 'weight_decay', 'batch_size', 'epochs']:
        logger.info(f"  {key}: {config[key]}")

    # Create full config with fixed params
    full_config = {**config, **FIXED_PARAMS}

    try:
        # Create model
        model = HybridGCN(
            num_nodes=data.num_nodes,
            hidden_dim=full_config['hidden_dim'],
            num_layers=full_config['num_layers'],
            dropout=full_config['dropout'],
            chemical_dim=full_config['chemical_dim'],
            decoder_type=full_config['decoder_type'],
            decoder_dropout=full_config['decoder_dropout'],
            use_gating=full_config['use_gating'],
        ).to(device)

        logger.info(f"Model: {model.description}")

        # Train
        result = train_hybrid_model(
            name=name,
            model=model,
            data=data,
            train_pos=split_edge['train']['edge'].to(device),
            valid_pos=split_edge['valid']['edge'].to(device),
            valid_neg=split_edge['valid']['edge_neg'].to(device),
            test_pos=split_edge['test']['edge'].to(device),
            test_neg=split_edge['test']['edge_neg'].to(device),
            evaluate_fn=None,
            device=device,
            epochs=full_config['epochs'],
            lr=full_config['lr'],
            weight_decay=full_config['weight_decay'],
            batch_size=full_config['batch_size'],
            eval_every=full_config['eval_every'],
            patience=full_config['patience'],
        )

        improvement = ((result.best_val_hits - BASELINE['val_hits']) / BASELINE['val_hits']) * 100

        logger.info("")
        logger.info(f"Result: val@20={result.best_val_hits:.4f}, test@20={result.best_test_hits:.4f}")
        logger.info(f"Improvement over baseline: {improvement:+.1f}%")

        return {
            'config_id': config_id,
            'name': name,
            'config': config,
            'val_hits': result.best_val_hits,
            'test_hits': result.best_test_hits,
            'best_epoch': result.best_epoch,
            'improvement_pct': improvement,
            'success': True,
        }

    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

        return {
            'config_id': config_id,
            'name': name,
            'config': config,
            'success': False,
            'error': str(e),
        }


def main():
    logger.info("=" * 80)
    logger.info("TESTING SPECIFIC CONFIGURATIONS")
    logger.info("=" * 80)
    logger.info(f"Baseline: val@20={BASELINE['val_hits']:.4f}, test@20={BASELINE['test_hits']:.4f}")
    logger.info(f"Testing {len(CONFIGS_TO_TEST)} configurations...")
    logger.info("")

    results = []

    for i, config in enumerate(CONFIGS_TO_TEST, 1):
        result = test_config(i, config)
        results.append(result)

        # Save after each
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        torch.cuda.empty_cache()

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY OF RESULTS")
    logger.info("=" * 80)
    logger.info("")

    successful = [r for r in results if r.get('success', False)]

    if not successful:
        logger.info("No successful experiments!")
        return

    # Sort by performance
    successful.sort(key=lambda x: x['val_hits'], reverse=True)

    logger.info(f"Successful: {len(successful)}/{len(results)}")
    logger.info("")

    # Show all results
    for i, r in enumerate(successful, 1):
        status = "✅" if r['improvement_pct'] > 0 else "➖" if r['improvement_pct'] > -1 else "❌"
        logger.info(f"{i}. {status} {r['name']}")
        logger.info(f"   val@20={r['val_hits']:.4f}, test@20={r['test_hits']:.4f} ({r['improvement_pct']:+.1f}%)")
        logger.info("")

    # Best config
    best = successful[0]
    logger.info("=" * 80)
    logger.info("BEST CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Name: {best['name']}")
    logger.info(f"Performance: val@20={best['val_hits']:.4f}, test@20={best['test_hits']:.4f}")
    logger.info(f"Improvement: {best['improvement_pct']:+.1f}%")
    logger.info("")
    logger.info("Configuration:")
    for key, value in best['config'].items():
        if key != 'name':
            logger.info(f"  {key}: {value}")
    logger.info("")

    # Statistics
    improvements = [r['improvement_pct'] for r in successful]
    better = sum(1 for x in improvements if x > 0)
    logger.info("=" * 80)
    logger.info(f"Configs better than baseline: {better}/{len(successful)}")
    logger.info(f"Best improvement: {max(improvements):+.1f}%")
    logger.info(f"Average improvement: {sum(improvements)/len(improvements):+.1f}%")
    logger.info(f"Worst result: {min(improvements):+.1f}%")
    logger.info("")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Full log: {log_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
