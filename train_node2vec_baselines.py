"""
Train all Node2Vec-enhanced baseline models on ogbl-ddi.

This script trains GCN, GraphSAGE, GraphTransformer, and GAT models,
all initialized with Node2Vec embeddings that are fine-tuned during training.
"""
import argparse
import logging
import os
from datetime import datetime

import torch

from src.data import load_dataset
from src.models.node2vec import Node2VecGCN, Node2VecGraphSAGE, Node2VecTransformer, Node2VecGAT
from src.training.node2vec_trainer import Node2VecConfig, train_node2vec_embeddings
from src.training.minimal_trainer import train_minimal_baseline
from src.evals import evaluate


def parse_args():
    def int_from_str(val: str) -> int:
        """Accept integer-looking strings even if they include a trailing decimal (e.g., '20.')."""
        return int(float(val))

    parser = argparse.ArgumentParser(description="Train Node2Vec-enhanced baselines on ogbl-ddi")
    parser.add_argument('--hidden-dim', type=int_from_str, default=128)
    parser.add_argument('--num-layers', type=int_from_str, default=2)
    parser.add_argument('--epochs', type=int_from_str, default=100)
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
    parser.add_argument('--node2vec-epochs', type=int_from_str, default=50)
    parser.add_argument('--node2vec-batch-size', type=int_from_str, default=256)
    parser.add_argument('--node2vec-lr', type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()

    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/node2vec_baselines_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "node2vec_baselines.log")

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
    logger.info("Loading dataset ogbl-ddi...")
    data, split_edge, num_nodes, evaluator = load_dataset('ogbl-ddi', device=device)
    
    train_pos = split_edge['train']['edge'].to(device)
    valid_pos = split_edge['valid']['edge'].to(device)
    valid_neg = split_edge['valid']['edge_neg'].to(device)
    test_pos = split_edge['test']['edge'].to(device)
    test_neg = split_edge['test']['edge_neg'].to(device)

    logger.info(f"Dataset loaded: {num_nodes} nodes")
    logger.info(f"Train pos edges: {train_pos.size(0)}, Valid pos: {valid_pos.size(0)}, Test pos: {test_pos.size(0)}")
    logger.info(f"Valid neg edges: {valid_neg.size(0)}, Test neg: {test_neg.size(0)}")

    # Create evaluation function wrapper
    def evaluate_fn(model, pos_edges, neg_edges, batch_size):
        return evaluate(model, data, evaluator, pos_edges, neg_edges, batch_size)

    # Pretrain Node2Vec embeddings
    logger.info("\n" + "=" * 80)
    logger.info("PRETRAINING NODE2VEC EMBEDDINGS")
    logger.info("=" * 80)
    
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
    
    logger.info(
        f"Pretraining Node2Vec embeddings (dim={node2vec_cfg.dim}, "
        f"walk_length={node2vec_cfg.walk_length}, context={node2vec_cfg.context_size}, "
        f"walks_per_node={node2vec_cfg.walks_per_node}, neg={node2vec_cfg.negative_samples}, "
        f"epochs={node2vec_cfg.epochs})"
    )
    
    node2vec_embeddings = train_node2vec_embeddings(
        data.edge_index,
        num_nodes,
        device,
        node2vec_cfg
    )
    
    logger.info(f"Node2Vec pretraining complete. Embeddings shape: {node2vec_embeddings.shape}")

    # Store results
    results = {}

    # Train Node2Vec-GCN
    logger.info("\n" + "=" * 80)
    logger.info("Training Node2Vec-GCN")
    logger.info("=" * 80)
    gcn_model = Node2VecGCN(
        node2vec_embeddings=node2vec_embeddings,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        decoder_dropout=0.0,
        use_multi_strategy=False,
    )
    gcn_result = train_minimal_baseline(
        "Node2Vec-GCN",
        gcn_model,
        data,
        train_pos,
        valid_pos,
        valid_neg,
        test_pos,
        test_neg,
        evaluate_fn,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size
    )
    results['Node2Vec-GCN'] = gcn_result

    # Train Node2Vec-GraphSAGE
    logger.info("\n" + "=" * 80)
    logger.info("Training Node2Vec-GraphSAGE")
    logger.info("=" * 80)
    sage_model = Node2VecGraphSAGE(
        node2vec_embeddings=node2vec_embeddings,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.0,
        decoder_dropout=0.0,
        use_multi_strategy=False,
        use_batch_norm=False,
    )
    sage_result = train_minimal_baseline(
        "Node2Vec-GraphSAGE",
        sage_model,
        data,
        train_pos,
        valid_pos,
        valid_neg,
        test_pos,
        test_neg,
        evaluate_fn,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size
    )
    results['Node2Vec-GraphSAGE'] = sage_result

    # Train Node2Vec-GraphTransformer
    logger.info("\n" + "=" * 80)
    logger.info("Training Node2Vec-GraphTransformer")
    logger.info("=" * 80)
    transformer_model = Node2VecTransformer(
        node2vec_embeddings=node2vec_embeddings,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=4,
        dropout=0.0,
        decoder_dropout=0.0,
        use_multi_strategy=False,
    )
    transformer_result = train_minimal_baseline(
        "Node2Vec-GraphTransformer",
        transformer_model,
        data,
        train_pos,
        valid_pos,
        valid_neg,
        test_pos,
        test_neg,
        evaluate_fn,
        device=device,
        epochs=args.epochs,
        lr=0.005,  # Lower LR for transformer
        patience=args.patience,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size
    )
    results['Node2Vec-GraphTransformer'] = transformer_result

    # Train Node2Vec-GAT
    logger.info("\n" + "=" * 80)
    logger.info("Training Node2Vec-GAT")
    logger.info("=" * 80)
    gat_model = Node2VecGAT(
        node2vec_embeddings=node2vec_embeddings,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=4,
        dropout=0.0,
        decoder_dropout=0.0,
        use_multi_strategy=False,
    )
    gat_result = train_minimal_baseline(
        "Node2Vec-GAT",
        gat_model,
        data,
        train_pos,
        valid_pos,
        valid_neg,
        test_pos,
        test_neg,
        evaluate_fn,
        device=device,
        epochs=args.epochs,
        lr=0.005,  # Lower LR for attention
        patience=args.patience,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size
    )
    results['Node2Vec-GAT'] = gat_result

    # Final results summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS - NODE2VEC BASELINES")
    logger.info("=" * 80)
    for model_name, result in results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Validation Hits@20: {result.best_val_hits:.4f}")
        logger.info(f"  Test Hits@20:       {result.best_test_hits:.4f}")
        val_test_gap = result.best_val_hits - result.best_test_hits
        gap_pct = (val_test_gap / result.best_val_hits * 100) if result.best_val_hits != 0 else 0.0
        logger.info(f"  Val-Test Gap:       {val_test_gap:.4f} ({gap_pct:.1f}% relative)")
    logger.info("=" * 80)
    logger.info(f"Results logged to: {log_file}")


if __name__ == "__main__":
    main()

