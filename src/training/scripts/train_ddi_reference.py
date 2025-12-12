"""
Reference-style trainer for OGBL-DDI using the official GCN/SAGE + MLP edge
predictor architecture (ported in `src/models/ogb_ddi_gnn/gnn.py`).

Defaults match the original OGB example (dropout=0.5, hidden=256, 2 layers) so
you can compare directly. Set `--dropout 0.0` if you want to align with this
repo's zero-dropout preference for DDI.
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from src.models.advanced.gcn_advanced import (
    GCNAdvanced,
    LinkPredictor,
    train as train_epoch,
    test as eval_epoch,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(args) -> str:
    """Set up logging to file and console."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/ddi_gcn_advanced_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "ddi_gcn_advanced.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return log_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="OGBL-DDI (GCNAdvanced)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64 * 1024)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log_dir = setup_logging(args)
    logger = logging.getLogger()
    
    set_seed(args.seed)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    logger.info("OGBL-DDI Link Prediction - GCNAdvanced")
    logger.info(f"Device: {device}")
    logger.info(f"Log directory: {log_dir}")
    logger.info("="*80)

    dataset = PygLinkPropPredDataset(name="ogbl-ddi", transform=T.ToSparseTensor())
    data = dataset[0]
    adj_t = data.adj_t.to(device)
    split_edge = dataset.get_edge_split()
    
    logger.info(f"Dataset: {data.num_nodes} nodes, {data.num_edges} edges")
    logger.info(f"Train edges: {split_edge['train']['edge'].size(0)}")
    logger.info(f"Valid edges: {split_edge['valid']['edge'].size(0)} pos, {split_edge['valid']['edge_neg'].size(0)} neg")
    logger.info(f"Test edges: {split_edge['test']['edge'].size(0)} pos, {split_edge['test']['edge_neg'].size(0)} neg")

    # Randomly pick eval_train subset (matching the reference script).
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge["train"]["edge"].size(0))
    idx = idx[: split_edge["valid"]["edge"].size(0)]
    split_edge["eval_train"] = {"edge": split_edge["train"]["edge"][idx]}

    model = GCNAdvanced(
        args.hidden_channels,
        args.hidden_channels,
        args.hidden_channels,
        args.num_layers,
        args.dropout,
    ).to(device)

    emb = torch.nn.Embedding(adj_t.size(0), args.hidden_channels).to(device)
    predictor = LinkPredictor(
        args.hidden_channels, args.hidden_channels, 1, args.num_layers, args.dropout
    ).to(device)
    
    logger.info("="*80)
    logger.info("Model Configuration:")
    logger.info(f"  architecture: GCNAdvanced + MLP")
    logger.info(f"  num_layers: {args.num_layers}")
    logger.info(f"  hidden_channels: {args.hidden_channels}")
    logger.info(f"  dropout: {args.dropout}")
    logger.info("Training Configuration:")
    logger.info(f"  epochs: {args.epochs}")
    logger.info(f"  lr: {args.lr}")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  eval_steps: {args.eval_steps}")
    logger.info(f"  runs: {args.runs}")
    logger.info("="*80)

    evaluator = Evaluator(name="ogbl-ddi")
    results: Dict[str, list[Tuple[float, float, float]]] = {
        "Hits@10": [],
        "Hits@20": [],
        "Hits@30": [],
    }

    for run in range(args.runs):
        if args.runs > 1:
            logger.info(f"Starting Run {run + 1}/{args.runs}")
        
        torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(emb.parameters()) + list(predictor.parameters()),
            lr=args.lr,
        )

        best_valid = {k: 0.0 for k in results}
        best_test = {k: 0.0 for k in results}
        best_epoch = {k: 0 for k in results}

        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(
                model,
                predictor,
                emb.weight,
                adj_t,
                split_edge,
                optimizer,
                args.batch_size,
            )

            if epoch % args.eval_steps == 0:
                scores = eval_epoch(
                    model, predictor, emb.weight, adj_t, split_edge, evaluator, args.batch_size
                )
                
                # Track best scores
                for key, (train_hits, valid_hits, test_hits) in scores.items():
                    if valid_hits > best_valid[key]:
                        best_valid[key] = valid_hits
                        best_test[key] = test_hits
                        best_epoch[key] = epoch

                if epoch % args.log_steps == 0:
                    run_prefix = f"[Run {run + 1:02d}] " if args.runs > 1 else ""
                    # Get Hits@20 for compact logging (matches user's style)
                    train_h20, valid_h20, test_h20 = scores["Hits@20"]
                    logger.info(
                        f"{run_prefix}[GCNAdvanced] Epoch {epoch:04d} | "
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

    logger.info("="*80)
    logger.info("Final Results" + (f" (averaged over {args.runs} runs)" if args.runs > 1 else ""))
    logger.info("="*80)
    
    for key, stats in results.items():
        valid_scores = torch.tensor([v for v, _ in stats])
        test_scores = torch.tensor([t for _, t in stats])
        
        if args.runs > 1:
            valid_mean, valid_std = valid_scores.mean().item(), valid_scores.std().item()
            test_mean, test_std = test_scores.mean().item(), test_scores.std().item()
            logger.info(
                f"{key}: "
                f"Valid {100*valid_mean:5.2f}% ± {100*valid_std:5.2f}% | "
                f"Test {100*test_mean:5.2f}% ± {100*test_std:5.2f}%"
            )
        else:
            valid_mean = valid_scores.mean().item()
            test_mean = test_scores.mean().item()
            logger.info(
                f"{key}: "
                f"Valid {100*valid_mean:5.2f}% | "
                f"Test {100*test_mean:5.2f}%"
            )
    
    logger.info("="*80)


if __name__ == "__main__":
    main()

