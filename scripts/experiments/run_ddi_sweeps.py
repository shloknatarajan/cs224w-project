"""
Sweep launcher for OGBL-DDI reference models (GCN/SAGE + LinkPredictor MLP).

Implements the hyperparameter grids described in the sweep proposal:
- Config A (baseline-ish): layers=2; hidden {128,192,256}; dropout {0.0,0.1,0.25};
  lr {0.005,0.003}; batch=64k.
- Config B (slightly deeper): layers=3; hidden {128,192,256}; dropout {0.0,0.1};
  lr {0.003,0.001}; batch=32k.
- Config C (regularization check): layers=2; hidden=256; dropout {0.5,0.0};
  lr {0.005,0.003}; batch=64k.

Each config runs for a configurable number of epochs/seeds and logs to
logs/ddi_sweeps_<timestamp>/<config_name>/config.log plus a sweep-level log.
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
    return logging.getLogger("ddi_sweeps")


def build_search_space() -> List[SweepConfig]:
    configs: List[SweepConfig] = []

    # Config A: layers=2; hidden {128,192,256}; dropout {0.0,0.1,0.25}; lr {0.005,0.003}; batch=64k
    for hidden in [128, 192, 256]:
        for dropout in [0.0, 0.1, 0.25]:
            for lr in [0.005, 0.003]:
                name = f"A_l2_h{hidden}_do{dropout}_lr{lr}_bs64k"
                configs.append(
                    SweepConfig(
                        name=name,
                        num_layers=2,
                        hidden_channels=hidden,
                        dropout=dropout,
                        lr=lr,
                        batch_size=64 * 1024,
                    )
                )

    # Config B: layers=3; hidden {128,192,256}; dropout {0.0,0.1}; lr {0.003,0.001}; batch=32k
    for hidden in [128, 192, 256]:
        for dropout in [0.0, 0.1]:
            for lr in [0.003, 0.001]:
                name = f"B_l3_h{hidden}_do{dropout}_lr{lr}_bs32k"
                configs.append(
                    SweepConfig(
                        name=name,
                        num_layers=3,
                        hidden_channels=hidden,
                        dropout=dropout,
                        lr=lr,
                        batch_size=32 * 1024,
                    )
                )

    # Config C: layers=2; hidden=256; dropout {0.5,0.0}; lr {0.005,0.003}; batch=64k
    for dropout in [0.5, 0.0]:
        for lr in [0.005, 0.003]:
            name = f"C_l2_h256_do{dropout}_lr{lr}_bs64k"
            configs.append(
                SweepConfig(
                    name=name,
                    num_layers=2,
                    hidden_channels=256,
                    dropout=dropout,
                    lr=lr,
                    batch_size=64 * 1024,
                )
            )

    return configs


def attach_config_handler(logger: logging.Logger, config_log_path: str) -> logging.Handler:
    handler = logging.FileHandler(config_log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    return handler


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

        best_valid = {k: 0.0 for k in results}
        best_test = {k: 0.0 for k in results}
        best_epoch = {k: 0 for k in results}

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
                logger.info(
                    f"[{config.name}][run {run_idx + 1}/{runs}] "
                    f"Epoch {epoch:04d} | loss {loss:.4f} | "
                    f"val@20 {valid_h20:.4f} | test@20 {test_h20:.4f} | "
                    f"best {best_valid['Hits@20']:.4f} (ep {best_epoch['Hits@20']})"
                )

        for key in results:
            results[key].append((best_valid[key], best_test[key]))

    for key in ["Hits@10", "Hits@20", "Hits@30"]:
        vals = results[key]
        best_valid = max(v for v, _ in vals)
        best_test = max(t for _, t in vals)
        logger.info(f"[{config.name}] {key}: best_valid={best_valid:.4f}, best_test={best_test:.4f}")

    logger.removeHandler(config_handler)
    config_handler.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hyperparameter sweeps for OGBL-DDI GCNAdvanced models.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id (ignored if no CUDA).")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per run (use shorter for screening).")
    parser.add_argument("--eval_steps", type=int, default=5, help="Evaluation frequency.")
    parser.add_argument("--runs", type=int, default=1, help="Seeds per config.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"ddi_sweeps_{timestamp}")
    logger = setup_root_logger(log_dir)

    device_str = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Device: {device}")
    logger.info(f"Logging to: {log_dir}")

    dataset = PygLinkPropPredDataset(name="ogbl-ddi", transform=T.ToSparseTensor())
    data = dataset[0]
    adj_t = data.adj_t.to(device)
    split_edge = dataset.get_edge_split()

    # eval_train subset matches the reference script for train-split Hits.
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge["train"]["edge"].size(0))
    idx = idx[: split_edge["valid"]["edge"].size(0)]
    split_edge["eval_train"] = {"edge": split_edge["train"]["edge"][idx]}

    evaluator = Evaluator(name="ogbl-ddi")
    logger.info(f"Dataset: {data.num_nodes} nodes, {data.num_edges} edges")
    logger.info(f"Configs: {len(build_search_space())} total")

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
        )

    logger.info("Sweep complete.")


if __name__ == "__main__":
    main()
