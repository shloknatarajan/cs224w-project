#!/usr/bin/env python
"""
Extends train_ddi_reference.py to incorporate external knowledge:
- Morgan fingerprints (--morgan)
- PubChem properties (--pubchem)
- ChemBERTa embeddings (--chemberta)
- Drug-target features (--drug-targets)

Usage:
    # Train with all external features
    python train_ddi_reference_external.py --use_sage --all

    # Train with specific features
    python train_ddi_reference_external.py --use_sage --morgan --chemberta

    # Train baseline (no external features)
    python train_ddi_reference_external.py --use_sage
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from src.models.advanced.gdinn import (
    GDINN,
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


def setup_logging(args) -> str:
    """Set up logging to file and console."""
    # Build feature suffix
    features = []
    if args.morgan:
        features.append("morgan")
    if args.pubchem:
        features.append("pubchem")
    if args.chemberta:
        features.append("chemberta")
    if args.drug_targets:
        features.append("dti")

    feature_suffix = "_" + "_".join(features) if features else "_baseline"

    if args.all:
        feature_suffix = "_all"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/ddi_gdinn{feature_suffix}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"ddi_gdinn{feature_suffix}.log")

    # Get the root logger and configure it directly (basicConfig may be ignored)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return log_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OGBL-DDI with external features (GDINN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model configuration
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--fusion", type=str, default="concat",
                        choices=["concat", "add"],
                        help="Feature fusion strategy")

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=64 * 1024)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # External feature selection
    feature_group = parser.add_argument_group("External Features")
    feature_group.add_argument("--all", action="store_true",
                               help="Enable all available external features")
    feature_group.add_argument("--morgan", action="store_true",
                               help="Use Morgan fingerprints")
    feature_group.add_argument("--pubchem", action="store_true",
                               help="Use PubChem properties")
    feature_group.add_argument("--chemberta", action="store_true",
                               help="Use ChemBERTa embeddings")
    feature_group.add_argument("--drug-targets", action="store_true",
                               help="Use drug-target features")

    # Data paths
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Directory for external feature files")

    args = parser.parse_args()

    log_dir = setup_logging(args)
    logger = logging.getLogger()

    set_seed(args.seed)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    logger.info("OGBL-DDI Link Prediction - GDINN with External Features")
    logger.info(f"Device: {device}")
    logger.info(f"Log directory: {log_dir}")
    logger.info("=" * 80)

    # Load dataset
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", transform=T.ToSparseTensor())
    data = dataset[0]
    adj_t = data.adj_t.to(device)
    split_edge = dataset.get_edge_split()
    num_nodes = data.num_nodes

    logger.info(f"Dataset: {num_nodes} nodes, {data.num_edges} edges")
    logger.info(f"Train edges: {split_edge['train']['edge'].size(0)}")
    logger.info(f"Valid edges: {split_edge['valid']['edge'].size(0)} pos, {split_edge['valid']['edge_neg'].size(0)} neg")
    logger.info(f"Test edges: {split_edge['test']['edge'].size(0)} pos, {split_edge['test']['edge_neg'].size(0)} neg")

    # Randomly pick eval_train subset
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge["train"]["edge"].size(0))
    idx = idx[: split_edge["valid"]["edge"].size(0)]
    split_edge["eval_train"] = {"edge": split_edge["train"]["edge"][idx]}

    # Determine which features to use
    if args.all:
        use_morgan = True
        use_pubchem = True
        use_chemberta = True
        use_drug_targets = True
    else:
        use_morgan = args.morgan
        use_pubchem = args.pubchem
        use_chemberta = args.chemberta
        use_drug_targets = args.drug_targets

    # Load external features
    external_features: Optional[Dict[str, torch.Tensor]] = None
    external_dims: Dict[str, int] = {}

    if any([use_morgan, use_pubchem, use_chemberta, use_drug_targets]):
        logger.info("Loading external features...")

        feature_config = FeatureConfig(
            use_morgan=use_morgan,
            use_pubchem=use_pubchem,
            use_chemberta=use_chemberta,
            use_drug_targets=use_drug_targets,
            morgan_path=args.data_dir / "morgan_features.pt",
            pubchem_path=args.data_dir / "ogbl_ddi_properties.csv",
            chemberta_path=args.data_dir / "chemberta_embeddings.pt",
            drug_targets_path=args.data_dir / "drug_target_features.pt",
        )

        ext_features = load_external_features(feature_config, num_nodes)

        if ext_features.total_dim > 0:
            external_dims = ext_features.feature_dims
            external_features = {}

            if ext_features.morgan is not None:
                external_features['morgan'] = ext_features.morgan.to(device)
            if ext_features.pubchem is not None:
                external_features['pubchem'] = ext_features.pubchem.to(device)
            if ext_features.chemberta is not None:
                external_features['chemberta'] = ext_features.chemberta.to(device)
            if ext_features.drug_targets is not None:
                external_features['drug_targets'] = ext_features.drug_targets.to(device)

            logger.info(f"Loaded features: {external_dims}")
            logger.info(f"Total feature dimension: {ext_features.total_dim}")
        else:
            logger.warning("No external features could be loaded! Training without them.")
            external_features = None
            external_dims = {}
    else:
        logger.info("Training without external features (baseline)")

    # Create model
    model = GDINN(
        num_nodes=num_nodes,
        hidden_channels=args.hidden_channels,
        out_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        external_dims=external_dims if external_dims else None,
        fusion=args.fusion,
    ).to(device)

    predictor = LinkPredictor(
        args.hidden_channels, args.hidden_channels, 1, args.num_layers, args.dropout
    ).to(device)

    logger.info("=" * 80)
    logger.info("Model Configuration:")
    logger.info(f"  architecture: GDINN + MLP")
    logger.info(f"  num_layers: {args.num_layers}")
    logger.info(f"  hidden_channels: {args.hidden_channels}")
    logger.info(f"  dropout: {args.dropout}")
    logger.info(f"  fusion: {args.fusion}")
    logger.info(f"  external_features: {list(external_dims.keys()) if external_dims else 'None'}")
    logger.info("Training Configuration:")
    logger.info(f"  epochs: {args.epochs}")
    logger.info(f"  lr: {args.lr}")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  eval_steps: {args.eval_steps}")
    logger.info(f"  runs: {args.runs}")
    logger.info("=" * 80)

    evaluator = Evaluator(name="ogbl-ddi")
    results: Dict[str, list[Tuple[float, float]]] = {
        "Hits@10": [],
        "Hits@20": [],
        "Hits@30": [],
    }

    for run in range(args.runs):
        if args.runs > 1:
            logger.info(f"Starting Run {run + 1}/{args.runs}")

        model.reset_parameters()
        predictor.reset_parameters()

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr,
        )

        best_valid = {k: 0.0 for k in results}
        best_test = {k: 0.0 for k in results}
        best_epoch = {k: 0 for k in results}

        for epoch in range(1, args.epochs + 1):
            loss = train_with_external(
                model,
                predictor,
                adj_t,
                split_edge,
                optimizer,
                args.batch_size,
                external_features,
            )

            if epoch % args.eval_steps == 0:
                scores = test_with_external(
                    model, predictor, adj_t, split_edge, evaluator,
                    args.batch_size, external_features
                )

                # Track best scores
                for key, (train_hits, valid_hits, test_hits) in scores.items():
                    if valid_hits > best_valid[key]:
                        best_valid[key] = valid_hits
                        best_test[key] = test_hits
                        best_epoch[key] = epoch

                if epoch % args.log_steps == 0:
                    run_prefix = f"[Run {run + 1:02d}] " if args.runs > 1 else ""
                    train_h20, valid_h20, test_h20 = scores["Hits@20"]
                    logger.info(
                        f"{run_prefix}[GDINN] Epoch {epoch:04d} | "
                        f"loss {loss:.4f} | "
                        f"val@20 {valid_h20:.4f} | "
                        f"test@20 {test_h20:.4f} | "
                        f"best {best_valid['Hits@20']:.4f} (ep {best_epoch['Hits@20']})"
                    )

        for key in results:
            results[key].append((best_valid[key], best_test[key]))

        if args.runs > 1:
            logger.info(f"Run {run + 1} - Best Results:")
            for key, (valid, test) in ((k, results[k][-1]) for k in results):
                logger.info(f"  {key}: Valid {100*valid:5.2f}% | Test {100*test:5.2f}%")

    logger.info("=" * 80)
    logger.info("Final Results" + (f" (averaged over {args.runs} runs)" if args.runs > 1 else ""))
    logger.info("=" * 80)

    for key, stats in results.items():
        valid_scores = torch.tensor([v for v, _ in stats])
        test_scores = torch.tensor([t for _, t in stats])

        if args.runs > 1:
            valid_mean, valid_std = valid_scores.mean().item(), valid_scores.std().item()
            test_mean, test_std = test_scores.mean().item(), test_scores.std().item()
            logger.info(
                f"{key}: "
                f"Valid {100*valid_mean:5.2f}% +/- {100*valid_std:5.2f}% | "
                f"Test {100*test_mean:5.2f}% +/- {100*test_std:5.2f}%"
            )
        else:
            valid_mean = valid_scores.mean().item()
            test_mean = test_scores.mean().item()
            logger.info(
                f"{key}: "
                f"Valid {100*valid_mean:5.2f}% | "
                f"Test {100*test_mean:5.2f}%"
            )

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
