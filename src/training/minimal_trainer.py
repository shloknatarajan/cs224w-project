import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import negative_sampling

logger = logging.getLogger(__name__)


@dataclass
class BaselineRunResult:
    best_val_hits: float
    best_test_hits: float
    best_epoch: int


def _ensure_zero_dropout(model: nn.Module) -> None:
    """Baseline runs must keep dropout disabled to match minimal configs."""
    flagged_layers = []
    if hasattr(model, "dropout"):
        drop_value = getattr(model, "dropout")
        if isinstance(drop_value, (float, int)) and drop_value != 0:
            flagged_layers.append(f"model.dropout={drop_value}")

    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)) and module.p > 0:
            flagged_layers.append(f"{name} (p={module.p})")

    if flagged_layers:
        details = ", ".join(flagged_layers)
        raise ValueError(f"Baseline trainer enforces dropout=0. Found: {details}")


def train_minimal_baseline(
    name: str,
    model: nn.Module,
    data,
    train_pos: torch.Tensor,
    valid_pos: torch.Tensor,
    valid_neg: torch.Tensor,
    test_pos: torch.Tensor,
    test_neg: torch.Tensor,
    evaluate_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor, int], float],
    *,
    device: torch.device,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    eval_every: int = 5,
    patience: int | None = 20,
    batch_size: int = 50000,
    eval_batch_size: int | None = None,
) -> BaselineRunResult:
    """
    Ultra-minimal training loop tailored for the baseline models.

    - Random negative sampling only
    - Single AdamW optimizer
    - Batch decoding to keep memory predictable
    - Strictly enforces dropout=0 to match baseline assumptions
    """
    _ensure_zero_dropout(model)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_edge_index = data.edge_index.to(device)
    num_nodes = data.num_nodes

    best_val = float("-inf")
    best_test = float("-inf")
    best_epoch = 0
    epochs_no_improve = 0

    logger.info(
        f"[{name}] Starting minimal baseline training "
        f"(epochs={epochs}, lr={lr}, wd={weight_decay}, batch_size={batch_size}, eval_every={eval_every})"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Check if model needs input features (Morgan models) or uses embeddings
        if hasattr(data, 'x') and data.x is not None:
            z = model.encode(train_edge_index, x=data.x)
        else:
            z = model.encode(train_edge_index)

        neg_edges = negative_sampling(
            edge_index=train_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=train_pos.size(0),
            method="sparse",
        ).t().to(device)

        batch_losses = []
        for start in range(0, train_pos.size(0), batch_size):
            end = start + batch_size
            pos_batch = train_pos[start:end]
            neg_batch = neg_edges[start:end]

            pos_logits = model.decode(z, pos_batch)
            neg_logits = model.decode(z, neg_batch)

            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            batch_losses.append(pos_loss + neg_loss)

        total_loss = torch.stack(batch_losses).mean()
        total_loss.backward()
        optimizer.step()

        train_loss = float(total_loss.detach().cpu())

        if epoch == 1 or epoch % eval_every == 0:
            eval_bs = eval_batch_size or batch_size
            val_hits = float(evaluate_fn(model, valid_pos, valid_neg, eval_bs))
            test_hits = float(evaluate_fn(model, test_pos, test_neg, eval_bs))

            improved = val_hits > best_val
            if improved:
                best_val = val_hits
                best_test = test_hits
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            logger.info(
                f"[{name}] Epoch {epoch:04d} | loss {train_loss:.4f} | "
                f"val@20 {val_hits:.4f} | test@20 {test_hits:.4f} | "
                f"best {best_val:.4f} (ep {best_epoch})"
            )

            if patience is not None and epochs_no_improve >= patience:
                logger.info(f"[{name}] Early stopping at epoch {epoch} (no val improvement for {epochs_no_improve} evals)")
                break

    logger.info(
        f"[{name}] Done. Best val@20={best_val:.4f} | test@20={best_test:.4f} "
        f"(epoch {best_epoch})"
    )
    return BaselineRunResult(best_val, best_test, best_epoch)
