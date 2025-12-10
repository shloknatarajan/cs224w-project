"""
Quick hyperparameter exploration for Hybrid GCN

This script runs faster experiments with fewer epochs (50 instead of 200) and
less patience (10 instead of 20) to quickly explore the hyperparameter space.
Use this for rapid prototyping, then run full experiments on promising configs.
"""
import os
import logging
import json
import torch
from datetime import datetime
import random

from src.data.data_loader import load_dataset_chemberta
from src.models.hybrid import HybridGCN
from src.training.hybrid_trainer import train_hybrid_model

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/hybrid_gcn_quick_tuning_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "quick_tuning.log")
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

# Device
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

# Quick exploration search space (focused on most impactful parameters)
SEARCH_SPACE = {
    # Architecture
    'hidden_dim': [96, 128, 160, 256],
    'num_layers': [2, 3],
    'decoder_dropout': [0.2, 0.3, 0.4],

    # Training
    'lr': [0.005, 0.01, 0.015, 0.02],
    'weight_decay': [5e-5, 1e-4, 2e-4, 5e-4],
}

# Fixed params (for quick runs)
FIXED_PARAMS = {
    'epochs': 50,  # Reduced from 200
    'eval_every': 5,
    'patience': 10,  # Reduced from 20
    'dropout': 0.0,
    'chemical_dim': 768,
    'decoder_type': 'simple',
    'use_gating': False,
    'batch_size': 5000,
}

BASELINE = {
    'val_hits': 0.2349,
    'config': {
        'hidden_dim': 128,
        'num_layers': 2,
        'decoder_dropout': 0.3,
        'lr': 0.01,
        'weight_decay': 1e-4,
    }
}


def run_quick_experiment(exp_id, config):
    """Run a quick experiment."""
    # Show only changed params
    changes = []
    for key in ['hidden_dim', 'num_layers', 'decoder_dropout', 'lr', 'weight_decay']:
        if config[key] != BASELINE['config'][key]:
            changes.append(f"{key}={config[key]}")

    if changes:
        logger.info(f"\n[Exp {exp_id}] Testing: {', '.join(changes)}")
    else:
        logger.info(f"\n[Exp {exp_id}] Testing: BASELINE")

    try:
        model = HybridGCN(
            num_nodes=data.num_nodes,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            chemical_dim=config['chemical_dim'],
            decoder_type=config['decoder_type'],
            decoder_dropout=config['decoder_dropout'],
            use_gating=config['use_gating'],
        ).to(device)

        result = train_hybrid_model(
            name=f"Quick-{exp_id}",
            model=model,
            data=data,
            train_pos=split_edge['train']['edge'].to(device),
            valid_pos=split_edge['valid']['edge'].to(device),
            valid_neg=split_edge['valid']['edge_neg'].to(device),
            test_pos=split_edge['test']['edge'].to(device),
            test_neg=split_edge['test']['edge_neg'].to(device),
            evaluate_fn=None,
            device=device,
            epochs=config['epochs'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            batch_size=config['batch_size'],
            eval_every=config['eval_every'],
            patience=config['patience'],
        )

        improvement = ((result.best_val_hits - BASELINE['val_hits']) / BASELINE['val_hits']) * 100

        logger.info(f"[Exp {exp_id}] Result: val@20={result.best_val_hits:.4f} ({improvement:+.1f}%)")

        return {
            'exp_id': exp_id,
            'config': config,
            'val_hits': result.best_val_hits,
            'test_hits': result.best_test_hits,
            'best_epoch': result.best_epoch,
            'improvement_pct': improvement,
            'success': True,
        }
    except Exception as e:
        logger.error(f"[Exp {exp_id}] FAILED: {e}")
        return {
            'exp_id': exp_id,
            'config': config,
            'success': False,
            'error': str(e),
        }


def quick_random_search(num_experiments=15):
    """Quick random search."""
    logger.info(f"\nRunning {num_experiments} quick experiments...")
    logger.info(f"Using {FIXED_PARAMS['epochs']} epochs (vs 200 for full runs)")
    logger.info("")

    results = []
    configs_tried = set()

    for exp_id in range(1, num_experiments + 1):
        # Sample config
        for _ in range(100):
            config = {key: random.choice(values) for key, values in SEARCH_SPACE.items()}
            config.update(FIXED_PARAMS)
            config_tuple = tuple(sorted(config.items()))
            if config_tuple not in configs_tried:
                configs_tried.add(config_tuple)
                break

        result = run_quick_experiment(exp_id, config)
        results.append(result)

        # Save after each
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        torch.cuda.empty_cache()

    return results


def analyze_quick_results(results):
    """Quick analysis."""
    successful = [r for r in results if r.get('success', False)]

    if not successful:
        logger.info("\nNo successful experiments!")
        return

    successful.sort(key=lambda x: x['val_hits'], reverse=True)

    logger.info("")
    logger.info("=" * 80)
    logger.info("QUICK TUNING RESULTS")
    logger.info("=" * 80)
    logger.info("")

    # Top configs
    logger.info("Top 5 configurations:")
    for i, r in enumerate(successful[:5], 1):
        cfg = r['config']
        logger.info(f"{i}. val@20={r['val_hits']:.4f} ({r['improvement_pct']:+.1f}%) | "
                   f"h={cfg['hidden_dim']}, L={cfg['num_layers']}, "
                   f"dec_drop={cfg['decoder_dropout']}, lr={cfg['lr']}, wd={cfg['weight_decay']}")

    # Best config
    best = successful[0]
    logger.info("")
    logger.info("Best configuration (recommend full training on this):")
    logger.info(f"  val@20={best['val_hits']:.4f} ({best['improvement_pct']:+.1f}%)")
    logger.info(f"  hidden_dim={best['config']['hidden_dim']}")
    logger.info(f"  num_layers={best['config']['num_layers']}")
    logger.info(f"  decoder_dropout={best['config']['decoder_dropout']}")
    logger.info(f"  lr={best['config']['lr']}")
    logger.info(f"  weight_decay={best['config']['weight_decay']}")

    # Summary stats
    improvements = [r['improvement_pct'] for r in successful]
    better_count = sum(1 for x in improvements if x > 0)
    logger.info("")
    logger.info(f"Better than baseline: {better_count}/{len(successful)} ({100*better_count/len(successful):.1f}%)")
    logger.info(f"Best improvement: {max(improvements):+.1f}%")
    logger.info(f"Average improvement: {sum(improvements)/len(improvements):+.1f}%")


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("QUICK HYPERPARAMETER EXPLORATION")
    logger.info("=" * 80)
    logger.info(f"Baseline: val@20={BASELINE['val_hits']:.4f}")
    logger.info(f"Config: {json.dumps(BASELINE['config'], indent=2)}")

    # Run quick experiments
    results = quick_random_search(num_experiments=20)

    # Analyze
    analyze_quick_results(results)

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Results: {results_file}")
    logger.info(f"Log: {log_file}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Review top configurations above")
    logger.info("2. Run full training (200 epochs) on promising configs")
    logger.info("3. Use tune_hybrid_gcn.py for thorough search")
    logger.info("=" * 80)
