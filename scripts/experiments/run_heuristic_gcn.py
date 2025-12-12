#!/usr/bin/env python3
"""
Run heuristic-enhanced GCN (simplified leaderboard approach)

This model incorporates graph-based link prediction heuristics:
- Common Neighbors (CN)
- Jaccard Coefficient
- Adamic-Adar (AA)
- Resource Allocation (RA)
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.heuristic import HeuristicGCN
from src.training.heuristic_trainer import train_heuristic_model
from src.data import load_dataset


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"heuristic_gcn_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_filename = log_dir / "heuristic_gcn.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Running Heuristic GCN (leaderboard-inspired)")
    logger.info(f"Device: {device}")
    logger.info(f"Log directory: {log_dir}")

    # Load data
    logger.info("Loading ogbl-ddi dataset...")
    data, split_edge, num_nodes, evaluator = load_dataset('ogbl-ddi', device=device)

    train_pos = split_edge["train"]["edge"].to(device)
    valid_pos = split_edge["valid"]["edge"].to(device)
    valid_neg = split_edge["valid"]["edge_neg"].to(device)
    test_pos = split_edge["test"]["edge"].to(device)
    test_neg = split_edge["test"]["edge_neg"].to(device)

    logger.info(f"Graph: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    logger.info(f"Train edges: {train_pos.size(0)}")
    logger.info(f"Valid edges: {valid_pos.size(0)} pos, {valid_neg.size(0)} neg")
    logger.info(f"Test edges: {test_pos.size(0)} pos, {test_neg.size(0)} neg")

    # Model configuration
    model_config = {
        'num_nodes': data.num_nodes,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.0,
        'decoder_dropout': 0.3,
        'heuristic_dim': 4,  # CN, Jaccard, AA, RA
        'use_degree_emb': True,
        'degree_emb_dim': 16,
    }

    # Training configuration
    train_config = {
        'epochs': 200,
        'lr': 0.01,
        'weight_decay': 1e-4,
        'batch_size': 50000,
        'eval_batch_size': 10000,
        'eval_every': 5,
        'patience': 20,
        'heuristic_batch_size': 10000,  # For heuristic computation
    }

    logger.info("=" * 80)
    logger.info("Model Configuration:")
    for k, v in model_config.items():
        logger.info(f"  {k}: {v}")
    logger.info("Training Configuration:")
    for k, v in train_config.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 80)

    # Create model
    model = HeuristicGCN(**model_config)
    logger.info(f"Model: {model.description}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    result = train_heuristic_model(
        name="HeuristicGCN",
        model=model,
        data=data,
        train_pos=train_pos,
        valid_pos=valid_pos,
        valid_neg=valid_neg,
        test_pos=test_pos,
        test_neg=test_neg,
        evaluate_fn=None,  # Not used in this trainer
        device=device,
        **train_config
    )

    # Summary
    logger.info("=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"HeuristicGCN | Best Val Hits@20: {result.best_val_hits:.4f} | "
                f"Test Hits@20: {result.best_test_hits:.4f} | Epoch: {result.best_epoch}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
