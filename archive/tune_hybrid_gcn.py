"""
Hyperparameter tuning for Hybrid GCN-Simple model

This script performs a systematic search over hyperparameters to find the best
configuration for the hybrid GCN model. Current baseline (val@20=0.2349) uses:
- hidden_dim=128, num_layers=2, dropout=0.0, decoder_dropout=0.3
- lr=0.01, weight_decay=1e-4, batch_size=5000

We'll explore variations around these values to find improvements.
"""
import os
import logging
import json
import torch
from datetime import datetime
from itertools import product
import random

from src.data.data_loader import load_dataset_chemberta
from src.models.hybrid import HybridGCN
from src.training.hybrid_trainer import train_hybrid_model

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/hybrid_gcn_tuning_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "tuning.log")
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
logger.info(f"Logging results to: {log_file}")

# Load dataset once (reuse for all experiments)
logger.info("Loading dataset with ChemBERTa embeddings + SMILES mask...")
data, split_edge, num_nodes, evaluator = load_dataset_chemberta(
    dataset_name="ogbl-ddi",
    device=device,
    smiles_csv_path="data/smiles.csv",
    feature_cache_path="data/chemberta_features_768.pt",
    chemberta_model="seyonec/ChemBERTa-zinc-base-v1",
    batch_size=32,
)

logger.info(f"Dataset loaded: data.x.shape = {data.x.shape}")
logger.info(f"SMILES mask: {data.smiles_mask.sum().item():.0f} valid / {data.smiles_mask.size(0)} total")

# Hyperparameter search space
# Baseline: hidden_dim=128, num_layers=2, dropout=0.0, decoder_dropout=0.3
#           lr=0.01, weight_decay=1e-4, batch_size=5000
SEARCH_SPACE = {
    # Model architecture - explore around baseline
    'hidden_dim': [64, 128, 256],  # baseline: 128
    'num_layers': [2, 3, 4],  # baseline: 2
    'dropout': [0.0, 0.1],  # baseline: 0.0
    'decoder_dropout': [0.2, 0.3, 0.4, 0.5],  # baseline: 0.3

    # Training parameters - explore learning rate and regularization
    'lr': [0.005, 0.01, 0.02],  # baseline: 0.01
    'weight_decay': [1e-5, 1e-4, 5e-4, 1e-3],  # baseline: 1e-4
    'batch_size': [5000, 10000, 20000],  # baseline: 5000
}

# Fixed parameters
FIXED_PARAMS = {
    'epochs': 200,
    'eval_every': 5,
    'patience': 20,
    'chemical_dim': 768,
    'decoder_type': 'simple',
    'use_gating': False,
}

# Baseline result (current best)
BASELINE = {
    'name': 'Hybrid-GCN-Simple',
    'config': {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.0,
        'decoder_dropout': 0.3,
        'lr': 0.01,
        'weight_decay': 1e-4,
        'batch_size': 5000,
    },
    'val_hits': 0.2349,
    'test_hits': 0.1933,
}


def run_experiment(exp_id, config):
    """Run a single experiment with given configuration."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Experiment {exp_id}")
    logger.info("=" * 80)

    # Show which parameters differ from baseline
    diffs = []
    for key, value in config.items():
        if key in BASELINE['config'] and BASELINE['config'][key] != value:
            diffs.append(f"{key}: {BASELINE['config'][key]} → {value}")

    if diffs:
        logger.info(f"Changes from baseline: {', '.join(diffs)}")
    else:
        logger.info("Configuration: BASELINE")

    logger.info(f"Full config: {json.dumps({k: config[k] for k in sorted(config.keys())}, indent=2)}")

    # Create model
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

    logger.info(f"Model: {model.description}")

    # Train
    try:
        result = train_hybrid_model(
            name=f"Exp-{exp_id}",
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
        logger.info(f"Result: val@20={result.best_val_hits:.4f} ({improvement:+.1f}% vs baseline)")

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
        logger.error(f"Experiment {exp_id} failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'exp_id': exp_id,
            'config': config,
            'success': False,
            'error': str(e),
        }


def random_search(search_space, fixed_params, num_experiments=20, seed=42):
    """
    Perform random search over hyperparameters.
    More practical than grid search for large parameter spaces.
    """
    random.seed(seed)

    logger.info(f"Random search: {num_experiments} random configurations")

    # Calculate total possible combinations
    total_combinations = 1
    for values in search_space.values():
        total_combinations *= len(values)
    logger.info(f"Total possible combinations: {total_combinations}")

    results = []
    configs_tried = set()

    for exp_id in range(1, num_experiments + 1):
        # Sample random configuration (avoid duplicates)
        max_attempts = 100
        for _ in range(max_attempts):
            config = {key: random.choice(values) for key, values in search_space.items()}
            config.update(fixed_params)
            config_tuple = tuple(sorted(config.items()))
            if config_tuple not in configs_tried:
                configs_tried.add(config_tuple)
                break

        result = run_experiment(exp_id, config)
        results.append(result)

        # Save results after each experiment
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Clear GPU cache
        torch.cuda.empty_cache()

    return results


def focused_search(search_space, fixed_params, baseline_config, num_experiments=15):
    """
    Focused search: Start with baseline, then explore one parameter at a time.
    This helps understand individual parameter impact.
    """
    logger.info(f"Focused search: baseline + {num_experiments} variations")

    results = []

    # First, run baseline
    baseline_full = baseline_config.copy()
    baseline_full.update(fixed_params)
    result = run_experiment(0, baseline_full)
    results.append(result)

    # Save after baseline
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    torch.cuda.empty_cache()

    # Then explore variations
    exp_id = 1
    param_names = list(search_space.keys())
    random.shuffle(param_names)  # Random order for fairness

    for param_name in param_names:
        if exp_id > num_experiments:
            break

        # Try each value for this parameter
        for value in search_space[param_name]:
            if exp_id > num_experiments:
                break

            # Skip if it's the baseline value
            if param_name in baseline_config and baseline_config[param_name] == value:
                continue

            # Create config: baseline + one change
            config = baseline_config.copy()
            config[param_name] = value
            config.update(fixed_params)

            result = run_experiment(exp_id, config)
            results.append(result)
            exp_id += 1

            # Save after each experiment
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            torch.cuda.empty_cache()

    return results


def analyze_results(results):
    """Analyze and report best configurations."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER TUNING RESULTS")
    logger.info("=" * 80)

    # Filter successful experiments
    successful = [r for r in results if r.get('success', False)]

    if not successful:
        logger.info("No successful experiments!")
        return

    logger.info(f"Successful experiments: {len(successful)} / {len(results)}")

    # Sort by validation hits
    successful.sort(key=lambda x: x['val_hits'], reverse=True)

    # Display results
    logger.info("")
    logger.info("=" * 80)
    logger.info("TOP 10 CONFIGURATIONS")
    logger.info("=" * 80)
    logger.info("")

    for i, result in enumerate(successful[:10], 1):
        improvement = result.get('improvement_pct', 0)
        status = "✅" if improvement > 0 else "➖" if improvement > -1 else "❌"

        logger.info(f"{i}. {status} val@20={result['val_hits']:.4f}, test@20={result['test_hits']:.4f} ({improvement:+.1f}%)")

        # Show key params
        cfg = result['config']
        logger.info(f"   h={cfg['hidden_dim']}, L={cfg['num_layers']}, "
                   f"drop={cfg['dropout']}, dec_drop={cfg['decoder_dropout']}, "
                   f"lr={cfg['lr']}, wd={cfg['weight_decay']}, bs={cfg['batch_size']}")
        logger.info("")

    # Best configuration
    best = successful[0]
    logger.info("")
    logger.info("=" * 80)
    logger.info("BEST CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"val@20={best['val_hits']:.4f}, test@20={best['test_hits']:.4f}")
    logger.info(f"Improvement: {best['improvement_pct']:+.1f}%")
    logger.info("")
    logger.info("Configuration:")
    for key in sorted(SEARCH_SPACE.keys()):
        baseline_val = BASELINE['config'].get(key, 'N/A')
        new_val = best['config'][key]
        change_marker = " ← CHANGED" if baseline_val != new_val else ""
        logger.info(f"  {key}: {new_val} (baseline: {baseline_val}){change_marker}")
    logger.info("")

    # Parameter impact analysis
    logger.info("")
    logger.info("=" * 80)
    logger.info("PARAMETER IMPACT ANALYSIS")
    logger.info("=" * 80)

    for param_name in SEARCH_SPACE.keys():
        logger.info(f"\n{param_name}:")
        param_values = {}
        for result in successful:
            value = result['config'][param_name]
            if value not in param_values:
                param_values[value] = []
            param_values[value].append(result['val_hits'])

        for value in sorted(param_values.keys()):
            scores = param_values[value]
            avg_score = sum(scores) / len(scores)
            best_score = max(scores)
            is_baseline = (param_name in BASELINE['config'] and
                          BASELINE['config'][param_name] == value)
            baseline_marker = " ← baseline" if is_baseline else ""
            logger.info(f"  {value}: avg={avg_score:.4f}, best={best_score:.4f} (n={len(scores)}){baseline_marker}")

    # Improvement summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    improvements = [r['improvement_pct'] for r in successful]
    better = sum(1 for x in improvements if x > 0)
    logger.info(f"Configurations better than baseline: {better}/{len(successful)} ({100*better/len(successful):.1f}%)")
    logger.info(f"Best improvement: {max(improvements):+.1f}%")
    logger.info(f"Average improvement: {sum(improvements)/len(improvements):+.1f}%")
    logger.info(f"Worst result: {min(improvements):+.1f}%")


if __name__ == "__main__":
    logger.info("")
    logger.info("=" * 80)
    logger.info("HYBRID GCN HYPERPARAMETER TUNING")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Baseline: {BASELINE['name']}")
    logger.info(f"  val@20={BASELINE['val_hits']:.4f}, test@20={BASELINE['test_hits']:.4f}")
    logger.info(f"  Config: {json.dumps(BASELINE['config'], indent=4)}")
    logger.info("")
    logger.info("Search space:")
    for key, values in SEARCH_SPACE.items():
        baseline_val = BASELINE['config'].get(key, 'N/A')
        logger.info(f"  {key}: {values} (baseline: {baseline_val})")
    logger.info("")
    logger.info("Fixed parameters:")
    for key, value in FIXED_PARAMS.items():
        logger.info(f"  {key}: {value}")
    logger.info("")

    # Choose search strategy
    SEARCH_STRATEGY = "random"  # "random" or "focused"

    if SEARCH_STRATEGY == "focused":
        # Focused search: baseline + one-at-a-time variations
        logger.info("Strategy: FOCUSED (baseline + one parameter at a time)")
        results = focused_search(SEARCH_SPACE, FIXED_PARAMS, BASELINE['config'], num_experiments=25)
    else:
        # Random search: explore random combinations
        logger.info("Strategy: RANDOM (random combinations)")
        results = random_search(SEARCH_SPACE, FIXED_PARAMS, num_experiments=30)

    # Analyze results
    analyze_results(results)

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Full log: {log_file}")
    logger.info("=" * 80)
