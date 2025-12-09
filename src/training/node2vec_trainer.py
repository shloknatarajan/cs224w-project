import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch_geometric.nn.models import Node2Vec

from src.evals import evaluate
from src.models import Node2VecGCN
from src.training.minimal_trainer import BaselineRunResult, train_minimal_baseline

logger = logging.getLogger(__name__)


@dataclass
class Node2VecConfig:
    dim: int = 64
    walk_length: int = 20
    context_size: int = 10
    walks_per_node: int = 10
    negative_samples: int = 1
    epochs: int = 30
    batch_size: int = 256
    lr: float = 0.01


def train_node2vec_embeddings(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    config: Optional[Node2VecConfig] = None,
) -> torch.Tensor:
    """
    Pre-train Node2Vec embeddings on the training graph to provide structural priors.

    Returns:
        Tensor of shape [num_nodes, dim] with pretrained embeddings (on CPU).
    """
    cfg = config or Node2VecConfig()

    if edge_index.numel() > 0 and int(edge_index.max()) + 1 > num_nodes:
        raise ValueError(
            f"Edge index contains node id >= num_nodes ({int(edge_index.max()) + 1} vs {num_nodes})"
        )

    logger.info(
        f"Pretraining Node2Vec embeddings "
        f"(dim={cfg.dim}, walk_length={cfg.walk_length}, context={cfg.context_size}, "
        f"walks_per_node={cfg.walks_per_node}, neg={cfg.negative_samples}, epochs={cfg.epochs})"
    )
    node2vec = Node2Vec(
        edge_index,
        embedding_dim=cfg.dim,
        walk_length=cfg.walk_length,
        context_size=cfg.context_size,
        walks_per_node=cfg.walks_per_node,
        num_negative_samples=cfg.negative_samples,
        sparse=True,
    ).to(device)

    loader = node2vec.loader(batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=cfg.lr)

    node2vec.train()
    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        if epoch == 1 or epoch % max(1, cfg.epochs // 5) == 0:
            avg_loss = total_loss / max(1, len(loader))
            logger.info(f"[node2vec] epoch {epoch:02d}/{cfg.epochs} | loss {avg_loss:.4f}")

    with torch.no_grad():
        embeddings = node2vec.embedding.weight.detach().clone().cpu()
    return embeddings


def train_node2vec_gcn(
    data,
    split_edge,
    num_nodes: int,
    evaluator,
    *,
    hidden_dim: int = 128,
    num_layers: int = 2,
    use_multi_strategy: bool = False,
    node2vec_cfg: Optional[Node2VecConfig] = None,
    device: torch.device,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    eval_every: int = 5,
    patience: int = 20,
    batch_size: int = 50000,
    eval_batch_size: int = 50000,
) -> BaselineRunResult:
    """
    Train a GCN that combines fixed Node2Vec embeddings with learnable ID embeddings.
    """
    train_pos = split_edge['train']['edge'].to(device)
    valid_pos = split_edge['valid']['edge'].to(device)
    valid_neg = split_edge['valid']['edge_neg'].to(device)
    test_pos = split_edge['test']['edge'].to(device)
    test_neg = split_edge['test']['edge_neg'].to(device)

    train_edge_index = train_pos.t().contiguous().to(device)

    # Pretrain Node2Vec on the training graph
    node2vec_embeddings = train_node2vec_embeddings(
        edge_index=train_edge_index,
        num_nodes=num_nodes,
        device=device,
        config=node2vec_cfg,
    )

    # Build model (dropout remains 0.0)
    model = Node2VecGCN(
        node2vec_embeddings=node2vec_embeddings,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        decoder_dropout=0.0,
        use_multi_strategy=use_multi_strategy,
    )
    logger.info(f"Model: {model.description}")

    # Evaluation wrapper
    def evaluate_fn(model, pos_edges, neg_edges, eval_bs):
        return evaluate(model, data, evaluator, pos_edges, neg_edges, eval_bs)

    result = train_minimal_baseline(
        "Node2Vec-GCN",
        model,
        data,
        train_pos,
        valid_pos,
        valid_neg,
        test_pos,
        test_neg,
        evaluate_fn,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        eval_every=eval_every,
        patience=patience,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
    )

    return result
