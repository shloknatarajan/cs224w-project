#!/usr/bin/env python
"""
Hyperparameter sweep for OGBL-DDI with external features.

Based on OGB leaderboard analysis and DDI-specific considerations:
- Higher hidden dims help with high-dimensional external features (3054-dim)
- Moderate dropout (0.3-0.5) prevents overfitting
- Learning rates in 0.001-0.01 range work well
- Deeper models (3 layers) can capture more complex patterns
- Weight decay helps regularization
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from src.models.ogb_ddi_gnn.gnn_external import (
    GCNExternal,
    SAGEExternal,
    LinkPredictor,
    train_with_external,
    test_with_external,
)
from src.data.external_features import (
    FeatureConfig,
    load_external_features,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# RECOMMENDED HYPERPARAMETER RANGES
# ============================================================================

# Based on OGB leaderboard and external feature characteristics
RECOMMENDED_CONFIGS = {
    # Core architecture
    "hidden_channels": [256, 512],       # Larger for 3054-dim features
    "num_layers": [2, 3],                 # 2-3 layers typically optimal
    "dropout": [0.3, 0.5],                # Higher dropout for regularization

    # Training
    "lr": [0.001, 0.005],                 # Standard range
    "weight_decay": [0, 1e-5, 1e-4],      # Light regularization

    # Feature fusion
    "fusion": ["concat", "add"],          # Concat usually better
}

# Quick sweep (fewer configs for fast iteration)
QUICK_CONFIGS = {
    "hidden_channels": [256, 512],
    "num_layers": [2, 3],
    "dropout": [0.3, 0.5],
    "lr": [0.0005, 0.001, 0.005],
    "weight_decay": [0, 1e-4],
    "fusion": ["concat"],
}

# Full sweep (comprehensive search)
FULL_CONFIGS = {
    "hidden_channels": [256, 384, 512],
    "num_layers": [2, 3],
    "dropout": [0.3, 0.4, 0.5],
    "lr": [0.0005, 0.001, 0.005, 0.01],
    "weight_decay": [0, 1e-5, 1e-4],
    "fusion": ["concat", "add"],
}


def generate_configs(param_grid: Dict) -> List[Dict]:
    """Generate all combinations from parameter grid."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)

    return configs


def run_single_config(
    config: Dict,
    model_type: str,
    device: torch.device,
    adj_t,
    split_edge: Dict,
    external_features: Optional[Dict[str, torch.Tensor]],
    external_dims: Dict[str, int],
    evaluator,
    num_nodes: int,
    epochs: int = 500,
    batch_size: int = 64 * 1024,
    eval_steps: int = 10,
    seed: int = 42,
) -> Tuple[float, float, int]:
    """Run training with a single hyperparameter configuration."""
    set_seed(seed)

    # Create model
    ModelClass = SAGEExternal if model_type == "sage" else GCNExternal
    model = ModelClass(
        num_nodes=num_nodes,
        hidden_channels=config["hidden_channels"],
        out_channels=config["hidden_channels"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        external_dims=external_dims if external_dims else None,
        fusion=config["fusion"],
    ).to(device)

    predictor = LinkPredictor(
        config["hidden_channels"],
        config["hidden_channels"],
        1,
        config["num_layers"],
        config["dropout"],
    ).to(device)

    model.reset_parameters()
    predictor.reset_parameters()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0),
    )

    best_valid = 0.0
    best_test = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        loss = train_with_external(
            model, predictor, adj_t, split_edge,
            optimizer, batch_size, external_features,
        )

        if epoch % eval_steps == 0:
            scores = test_with_external(
                model, predictor, adj_t, split_edge,
                evaluator, batch_size, external_features,
            )

            _, valid_hits, test_hits = scores["Hits@20"]

            if valid_hits > best_valid:
                best_valid = valid_hits
                best_test = test_hits
                best_epoch = epoch

    return best_valid, best_test, best_epoch


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for DDI external features")

    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "recommended", "full", "custom"],
                        help="Sweep mode: quick (fast), recommended, full (comprehensive)")
    parser.add_argument("--model", type=str, default="gcn",
                        choices=["gcn", "sage"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Epochs per config (use less for quick sweeps)")
    parser.add_argument("--eval_steps", type=int, default=10,
                        help="Evaluate every N epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64 * 1024)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))

    # Feature selection (default: all)
    parser.add_argument("--no-morgan", action="store_true")
    parser.add_argument("--no-pubchem", action="store_true")
    parser.add_argument("--no-chemberta", action="store_true")
    parser.add_argument("--no-drug-targets", action="store_true")

    args = parser.parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/sweep_external_{args.model}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/sweep.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger()

    # Select parameter grid
    if args.mode == "quick":
        param_grid = QUICK_CONFIGS
    elif args.mode == "recommended":
        param_grid = RECOMMENDED_CONFIGS
    elif args.mode == "full":
        param_grid = FULL_CONFIGS
    else:
        param_grid = RECOMMENDED_CONFIGS

    configs = generate_configs(param_grid)
    logger.info(f"Generated {len(configs)} configurations to test")
    logger.info(f"Parameter grid: {param_grid}")

    # Setup device
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Device: {device}")

    # Load dataset
    logger.info("Loading OGBL-DDI dataset...")
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", transform=T.ToSparseTensor())
    data = dataset[0]
    adj_t = data.adj_t.to(device)
    split_edge = dataset.get_edge_split()
    num_nodes = data.num_nodes

    # Setup eval_train subset
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge["train"]["edge"].size(0))
    idx = idx[:split_edge["valid"]["edge"].size(0)]
    split_edge["eval_train"] = {"edge": split_edge["train"]["edge"][idx]}

    # Load external features
    logger.info("Loading external features...")
    feature_config = FeatureConfig(
        use_morgan=not args.no_morgan,
        use_pubchem=not args.no_pubchem,
        use_chemberta=not args.no_chemberta,
        use_drug_targets=not args.no_drug_targets,
        morgan_path=args.data_dir / "morgan_features.pt",
        pubchem_path=args.data_dir / "ogbl_ddi_properties.csv",
        chemberta_path=args.data_dir / "chemberta_embeddings.pt",
        drug_targets_path=args.data_dir / "drug_target_features.pt",
    )

    ext_features = load_external_features(feature_config, num_nodes)

    external_features = {}
    external_dims = {}
    if ext_features.total_dim > 0:
        external_dims = ext_features.feature_dims
        if ext_features.morgan is not None:
            external_features['morgan'] = ext_features.morgan.to(device)
        if ext_features.pubchem is not None:
            external_features['pubchem'] = ext_features.pubchem.to(device)
        if ext_features.chemberta is not None:
            external_features['chemberta'] = ext_features.chemberta.to(device)
        if ext_features.drug_targets is not None:
            external_features['drug_targets'] = ext_features.drug_targets.to(device)
        logger.info(f"Loaded features: {external_dims} (total: {ext_features.total_dim})")

    evaluator = Evaluator(name="ogbl-ddi")

    # Run sweep
    results = []
    best_config = None
    best_valid = 0.0

    logger.info("="*80)
    logger.info("Starting hyperparameter sweep...")
    logger.info("="*80)

    for i, config in enumerate(configs):
        logger.info(f"\n[Config {i+1}/{len(configs)}] {config}")

        try:
            valid_hits, test_hits, best_epoch = run_single_config(
                config=config,
                model_type=args.model,
                device=device,
                adj_t=adj_t,
                split_edge=split_edge,
                external_features=external_features if external_features else None,
                external_dims=external_dims,
                evaluator=evaluator,
                num_nodes=num_nodes,
                epochs=args.epochs,
                batch_size=args.batch_size,
                eval_steps=args.eval_steps,
                seed=args.seed,
            )

            result = {
                **config,
                "val_hits20": valid_hits,
                "test_hits20": test_hits,
                "best_epoch": best_epoch,
            }
            results.append(result)

            logger.info(f"  Result: val@20={100*valid_hits:.2f}%, test@20={100*test_hits:.2f}% (ep {best_epoch})")

            if valid_hits > best_valid:
                best_valid = valid_hits
                best_config = result
                logger.info(f"  *** New best config! ***")

        except Exception as e:
            logger.error(f"  Config failed: {e}")
            results.append({**config, "error": str(e)})

        # Save intermediate results
        with open(f"{log_dir}/results.json", "w") as f:
            json.dump({
                "results": results,
                "best_config": best_config,
                "completed": i + 1,
                "total": len(configs),
            }, f, indent=2)

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("SWEEP COMPLETE")
    logger.info("="*80)

    if best_config:
        logger.info(f"\nBest configuration:")
        for k, v in best_config.items():
            if k in ["val_hits20", "test_hits20"]:
                logger.info(f"  {k}: {100*v:.2f}%")
            else:
                logger.info(f"  {k}: {v}")

    # Top 5 configs
    valid_results = [r for r in results if "val_hits20" in r]
    sorted_results = sorted(valid_results, key=lambda x: x["val_hits20"], reverse=True)

    logger.info(f"\nTop 5 configurations:")
    for i, r in enumerate(sorted_results[:5]):
        logger.info(f"  {i+1}. val@20={100*r['val_hits20']:.2f}%, test@20={100*r['test_hits20']:.2f}%")
        logger.info(f"     hidden={r['hidden_channels']}, layers={r['num_layers']}, "
                   f"dropout={r['dropout']}, lr={r['lr']}, fusion={r['fusion']}")

    logger.info(f"\nResults saved to {log_dir}/")


if __name__ == "__main__":
    main()
