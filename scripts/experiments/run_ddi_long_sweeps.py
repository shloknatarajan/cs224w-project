"""
Long-horizon sweep launcher for top OGBL-DDI configs with LR scheduling.

Key differences vs. run_ddi_sweeps.py:
- Focused on the best-performing short-run configs.
- Longer default training (800 epochs) and one seed.
- Adds a MultiStepLR decay (gamma=0.5 at 40% and 70% of training).

Configs:
- A_l2_h256_do0.25_lr0.005_bs64k
- A_l2_h192_do0.25_lr0.005_bs64k
- B_l3_h192_do0.0_lr0.001_bs32k
- A_l2_h256_do0.0_lr0.005_bs64k (dropout-free counterpart)
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import torch
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from src.models.advanced.gcn_advanced import (
    GCNAdvanced,
    LinkPredictor,
    train as train_epoch,
    test as eval_epoch,
)


@dataclass
class SweepConfig:
    name: str
    num_layers: int
    hidden_channels: int
    dropout: float
    lr: float
    batch_size: int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_root_logger(log_dir: str) -> logging.Logger:
    """Configure sweep-level logging."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "sweep.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%%Y-%%m-%%d %%H:%%M:%%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("ddi_long_sweeps")


def build_search_space() -> List[SweepConfig]:
    """Return the focused list of configs to rerun with longer epochs."""
    return [
        # SweepConfig("A_l2_h256_do0.25_lr0.005_bs64k", 2, 256, 0.25, 0.005, 64 * 1024),
        SweepConfig("A_l2_h256_do0.5_lr0.005_bs64k", 2, 256, 0.5, 0.005, 64 * 1024),
        # SweepConfig("A_l2_h192_do0.25_lr0.005_bs64k", 2, 192, 0.25, 0.005, 64 * 1024),
        # SweepConfig("B_l3_h256_do0.5_lr0.001_bs64k", 3, 256, 0.5, 0.001, 64 * 1024),
    ]


def attach_config_handler(logger: logging.Logger, config_log_path: str) -> logging.Handler:
    handler = logging.FileHandler(config_log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    return handler


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int) -> torch.optim.lr_scheduler._LRScheduler:
    milestones = sorted({int(epochs * 0.4), int(epochs * 0.7)})
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)


def run_config(
    logger: logging.Logger,
    config: SweepConfig,
    *,
    sweep_log_dir: str,
    device: torch.device,
    epochs: int,
    eval_steps: int,
    runs: int,
    base_seed: int,
    data,
    adj_t,
    split_edge: Dict[str, Dict[str, torch.Tensor]],
    evaluator,
    patience: int = 100,
    min_performance: float = 0.0,
) -> None:
    """Run a single hyperparameter configuration for `runs` seeds."""
    config_dir = os.path.join(sweep_log_dir, config.name)
    os.makedirs(config_dir, exist_ok=True)
    config_log = os.path.join(config_dir, "config.log")
    config_handler = attach_config_handler(logger, config_log)

    logger.info(
        f"[{config.name}] Starting (GCNAdvanced) | layers={config.num_layers}, "
        f"hidden={config.hidden_channels}, dropout={config.dropout}, "
        f"lr={config.lr}, batch={config.batch_size}, epochs={epochs}, runs={runs}"
    )

    results: Dict[str, list[Tuple[float, float]]] = {"Hits@10": [], "Hits@20": [], "Hits@30": []}

    for run_idx in range(runs):
        set_seed(base_seed + run_idx)

        model = GCNAdvanced(
            config.hidden_channels,
            config.hidden_channels,
            config.hidden_channels,
            config.num_layers,
            config.dropout,
        ).to(device)

        emb = torch.nn.Embedding(adj_t.size(0), config.hidden_channels).to(device)
        predictor = LinkPredictor(
            config.hidden_channels, config.hidden_channels, 1, config.num_layers, config.dropout
        ).to(device)

        torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        predictor.reset_parameters()

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(emb.parameters()) + list(predictor.parameters()),
            lr=config.lr,
        )
        scheduler = make_scheduler(optimizer, epochs)

        best_valid = {k: 0.0 for k in results}
        best_test = {k: 0.0 for k in results}
        best_epoch = {k: 0 for k in results}
        
        # Early stopping tracking
        epochs_since_improvement = 0
        best_val_hits20 = 0.0
        early_stopped = False

        for epoch in range(1, epochs + 1):
            loss = train_epoch(
                model,
                predictor,
                emb.weight,
                adj_t,
                split_edge,
                optimizer,
                config.batch_size,
            )
            scheduler.step()

            if epoch % eval_steps == 0:
                scores = eval_epoch(
                    model,
                    predictor,
                    emb.weight,
                    adj_t,
                    split_edge,
                    evaluator,
                    config.batch_size,
                )

                for key, (_, valid_hits, test_hits) in scores.items():
                    if valid_hits > best_valid[key]:
                        best_valid[key] = valid_hits
                        best_test[key] = test_hits
                        best_epoch[key] = epoch

                train_h20, valid_h20, test_h20 = scores["Hits@20"]
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[{config.name}][run {run_idx + 1}/{runs}] "
                    f"Epoch {epoch:04d} | loss {loss:.4f} | "
                    f"val@20 {valid_h20:.4f} | test@20 {test_h20:.4f} | "
                    f"best {best_valid['Hits@20']:.4f} (ep {best_epoch['Hits@20']}) | "
                    f"lr {current_lr:.6f}"
                )
                
                # Early stopping check
                if valid_h20 > best_val_hits20:
                    best_val_hits20 = valid_h20
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += eval_steps
                
                # Stop if no improvement for patience epochs
                if epochs_since_improvement >= patience:
                    logger.info(
                        f"[{config.name}][run {run_idx + 1}/{runs}] "
                        f"Early stopping at epoch {epoch}: no improvement for {patience} epochs. "
                        f"Best val@20: {best_val_hits20:.4f}"
                    )
                    early_stopped = True
                    break
                
                # Stop if not meeting minimum performance threshold
                if min_performance > 0.0 and epoch >= 100 and best_val_hits20 < min_performance:
                    logger.info(
                        f"[{config.name}][run {run_idx + 1}/{runs}] "
                        f"Early stopping at epoch {epoch}: best val@20 {best_val_hits20:.4f} "
                        f"below threshold {min_performance:.4f}"
                    )
                    early_stopped = True
                    break

        for key in results:
            results[key].append((best_valid[key], best_test[key]))
        
        if early_stopped:
            logger.info(
                f"[{config.name}][run {run_idx + 1}/{runs}] "
                f"Stopped early. Final best: val@20={best_valid['Hits@20']:.4f}, "
                f"test@20={best_test['Hits@20']:.4f} at epoch {best_epoch['Hits@20']}"
            )

    for key in ["Hits@10", "Hits@20", "Hits@30"]:
        vals = results[key]
        best_valid = max(v for v, _ in vals)
        best_test = max(t for _, t in vals)
        logger.info(f"[{config.name}] {key}: best_valid={best_valid:.4f}, best_test={best_test:.4f}")

    logger.removeHandler(config_handler)
    config_handler.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Longer-run sweeps for top OGBL-DDI GCNAdvanced configs.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id (ignored if no CUDA).")
    parser.add_argument("--epochs", type=int, default=2000, help="Epochs per run (long horizon).")
    parser.add_argument("--eval_steps", type=int, default=5, help="Evaluation frequency.")
    parser.add_argument("--runs", type=int, default=1, help="Seeds per config.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility.")
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stopping patience: stop if no val improvement for N epochs (default: 100)",
    )
    parser.add_argument(
        "--min_performance",
        type=float,
        default=0.0,
        help="Minimum val Hits@20 threshold. Stop early if below this after 100 epochs (0.0 = disabled)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"ddi_long_sweeps_{timestamp}")
    logger = setup_root_logger(log_dir)

    device_str = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Device: {device}")
    logger.info(f"Logging to: {log_dir}")

    dataset = PygLinkPropPredDataset(name="ogbl-ddi", transform=T.ToSparseTensor())
    data = dataset[0]
    adj_t = data.adj_t.to(device)
    split_edge = dataset.get_edge_split()

    torch.manual_seed(12345)
    idx = torch.randperm(split_edge["train"]["edge"].size(0))
    idx = idx[: split_edge["valid"]["edge"].size(0)]
    split_edge["eval_train"] = {"edge": split_edge["train"]["edge"][idx]}

    evaluator = Evaluator(name="ogbl-ddi")
    logger.info(f"Dataset: {data.num_nodes} nodes, {data.num_edges} edges")
    logger.info(f"Configs: {len(build_search_space())} total")
    logger.info(f"Early stopping: patience={args.patience} epochs")
    if args.min_performance > 0.0:
        logger.info(f"Performance threshold: min val@20={args.min_performance:.4f} (checked after epoch 100)")

    for config in build_search_space():
        run_config(
            logger,
            config,
            sweep_log_dir=log_dir,
            device=device,
            epochs=args.epochs,
            eval_steps=args.eval_steps,
            runs=args.runs,
            base_seed=args.seed,
            data=data,
            adj_t=adj_t,
            split_edge=split_edge,
            evaluator=evaluator,
            patience=args.patience,
            min_performance=args.min_performance,
        )

    logger.info("Long-horizon sweep complete.")


if __name__ == "__main__":
    main()
