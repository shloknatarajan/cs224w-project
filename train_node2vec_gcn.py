"""
Train a GCN that augments learnable IDs with frozen Node2Vec embeddings.
"""
import argparse
import logging
import os
from datetime import datetime

import torch

from src.data import load_dataset
from src.training.node2vec_trainer import Node2VecConfig, train_node2vec_gcn


def parse_args():
    def int_from_str(val: str) -> int:
        """Accept integer-looking strings even if they include a trailing decimal (e.g., '20.')."""
        return int(float(val))

    parser = argparse.ArgumentParser(description="Train Node2Vec+ID GCN on ogbl-ddi")
    parser.add_argument('--hidden-dim', type=int_from_str, default=128)
    parser.add_argument('--num-layers', type=int_from_str, default=2)
    parser.add_argument('--use-multi-strategy', action='store_true', help="Enable multi-strategy decoder")
    parser.add_argument('--epochs', type=int_from_str, default=200)
    parser.add_argument('--patience', type=int_from_str, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int_from_str, default=50000)
    parser.add_argument('--eval-batch-size', type=int_from_str, default=50000)
    parser.add_argument('--eval-every', type=int_from_str, default=5)
    # Node2Vec hyperparameters
    parser.add_argument('--node2vec-dim', type=int_from_str, default=64)
    parser.add_argument('--node2vec-walk-length', type=int_from_str, default=20)
    parser.add_argument('--node2vec-context-size', type=int_from_str, default=10)
    parser.add_argument('--node2vec-walks-per-node', type=int_from_str, default=10)
    parser.add_argument('--node2vec-negative-samples', type=int_from_str, default=1)
    parser.add_argument('--node2vec-epochs', type=int_from_str, default=30)
    parser.add_argument('--node2vec-batch-size', type=int_from_str, default=256)
    parser.add_argument('--node2vec-lr', type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()

    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/node2vec_gcn_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "node2vec_gcn.log")

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
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Logging results to: {log_file}")

    # Load dataset
    data, split_edge, num_nodes, evaluator = load_dataset('ogbl-ddi', device=device)

    node2vec_cfg = Node2VecConfig(
        dim=args.node2vec_dim,
        walk_length=args.node2vec_walk_length,
        context_size=args.node2vec_context_size,
        walks_per_node=args.node2vec_walks_per_node,
        negative_samples=args.node2vec_negative_samples,
        epochs=args.node2vec_epochs,
        batch_size=args.node2vec_batch_size,
        lr=args.node2vec_lr,
    )

    result = train_node2vec_gcn(
        data,
        split_edge,
        num_nodes,
        evaluator,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_multi_strategy=args.use_multi_strategy,
        node2vec_cfg=node2vec_cfg,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        patience=args.patience,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS - NODE2VEC + ID GCN")
    logger.info("=" * 80)
    logger.info(f"  Validation Hits@20: {result.best_val_hits:.4f}")
    logger.info(f"  Test Hits@20:       {result.best_test_hits:.4f}")
    val_test_gap = result.best_val_hits - result.best_test_hits
    gap_pct = (val_test_gap / result.best_val_hits * 100) if result.best_val_hits != 0 else 0.0
    logger.info(f"  Val-Test Gap:       {val_test_gap:.4f} ({gap_pct:.1f}% relative)")
    logger.info(f"Results logged to: {log_file}")


if __name__ == "__main__":
    main()
