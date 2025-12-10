import argparse
import os
import random
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.nn.models import Node2Vec
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling


# -------------------------------
# Utils
# -------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LinkPredictor(nn.Module):
    """Simple dot-product decoder."""
    def forward(self, z: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        src, dst = edge[:, 0], edge[:, 1]
        return (z[src] * z[dst]).sum(dim=1)


def train_node2vec_embeddings(
    edge_index: torch.Tensor,
    num_nodes: int,
    embedding_dim: int,
    device: torch.device,
    *,
    walk_length: int = 20,
    context_size: int = 10,
    walks_per_node: int = 10,
    num_negative_samples: int = 1,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 0.01,
) -> torch.Tensor:
    """
    Lightweight Node2Vec pretraining to provide structural priors.

    Returns:
        Tensor of shape [num_nodes, embedding_dim] with pretrained embeddings.
    """
    if edge_index.numel() > 0 and int(edge_index.max()) + 1 > num_nodes:
        raise ValueError(
            f"Edge index contains node id >= num_nodes ({int(edge_index.max()) + 1} vs {num_nodes})"
        )

    node2vec = Node2Vec(
        edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        sparse=True,
    ).to(device)

    loader = node2vec.loader(batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=lr)

    node2vec.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        if epoch == 1 or epoch % max(1, epochs // 5) == 0:
            avg_loss = total_loss / max(1, len(loader))
            print(f"[node2vec] epoch {epoch:02d}/{epochs} | loss {avg_loss:.4f}")

    with torch.no_grad():
        embeddings = node2vec.embedding.weight.detach().clone().cpu()
    return embeddings


class SAGEModel(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.3,
                 use_bn: bool = True, model_type: str = "sage"):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        self.dropout = dropout
        self.use_bn = use_bn
        self.model_type = model_type

        convs = []
        norms = []
        for _ in range(num_layers):
            if model_type == "sage":
                convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif model_type == "gcn":
                convs.append(GCNConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            norms.append(nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity())
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.predictor = LinkPredictor()

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.emb.weight
        for conv, norm in zip(self.convs, self.norms):
            if isinstance(conv, GCNConv):
                x_new = conv(x, edge_index)
            else:
                x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            # Residual
            if x_new.shape == x.shape:
                x = x + x_new
            else:
                x = x_new
        return x

    def decode(self, z: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        return self.predictor(z, edge)


class Node2VecGCNModel(nn.Module):
    """
    GCN encoder that concatenates fixed Node2Vec embeddings with learnable IDs.
    """
    def __init__(
        self,
        node2vec_embeddings: torch.Tensor,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_bn: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_bn = use_bn
        num_nodes, node2vec_dim = node2vec_embeddings.size()

        self.id_emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.id_emb.weight)

        # Keep Node2Vec embeddings fixed to preserve the structural prior
        self.node2vec_emb = nn.Embedding.from_pretrained(node2vec_embeddings, freeze=True)

        input_dim = hidden_dim + node2vec_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        convs = []
        norms = []
        for _ in range(num_layers):
            convs.append(GCNConv(hidden_dim, hidden_dim))
            norms.append(nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity())
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.predictor = LinkPredictor()

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.id_emb.weight, self.node2vec_emb.weight], dim=1)
        x = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            if self.dropout > 0:
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            if x_new.shape == x.shape:
                x = x + x_new
            else:
                x = x_new
        return x

    def decode(self, z: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        return self.predictor(z, edge)


@torch.no_grad()
def evaluate_hits20(model: nn.Module, edge_index: torch.Tensor, z: torch.Tensor,
                    pos_edge: torch.Tensor, neg_edge: torch.Tensor, evaluator: Evaluator) -> float:
    model.eval()
    pos_scores = model.decode(z, pos_edge).view(-1).cpu()
    neg_scores = model.decode(z, neg_edge).view(-1).cpu()
    result = evaluator.eval({
        'y_pred_pos': pos_scores,
        'y_pred_neg': neg_scores,
    })
    return float(result['hits@20'])


def train_one_seed(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = PygLinkPropPredDataset('ogbl-ddi')
    data = dataset[0]
    split_edge = dataset.get_edge_split()

    # Use only training edges for message passing to avoid leakage
    train_pos = split_edge['train']['edge'].to(device)
    valid_pos = split_edge['valid']['edge'].to(device)
    test_pos = split_edge['test']['edge'].to(device)

    valid_neg = split_edge['valid']['edge_neg'].to(device)
    test_neg = split_edge['test']['edge_neg'].to(device)

    num_nodes = data.num_nodes
    edge_index = train_pos.t().contiguous().to(device)

    node2vec_embeddings = None
    if args.model == "gcn_node2vec":
        print("Pretraining Node2Vec embeddings on the training graph...")
        node2vec_embeddings = train_node2vec_embeddings(
            edge_index=edge_index,
            num_nodes=num_nodes,
            embedding_dim=args.node2vec_dim,
            device=device,
            walk_length=args.node2vec_walk_length,
            context_size=args.node2vec_context_size,
            walks_per_node=args.node2vec_walks_per_node,
            num_negative_samples=args.node2vec_negative_samples,
            epochs=args.node2vec_epochs,
            batch_size=args.node2vec_batch_size,
            lr=args.node2vec_lr,
        )

    if args.model in ["sage", "gcn"]:
        model = SAGEModel(
            num_nodes=num_nodes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_bn=not args.no_bn,
            model_type=args.model,
        ).to(device)
    else:
        if node2vec_embeddings is None:
            raise ValueError("Node2Vec embeddings were not initialized.")
        if args.dropout != 0.0:
            print("Node2Vec GCN enforces dropout=0.0 to match minimal baseline settings.")
        model = Node2VecGCNModel(
            node2vec_embeddings=node2vec_embeddings,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=0.0,
            use_bn=not args.no_bn,
        ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    evaluator = Evaluator(name='ogbl-ddi')

    best_val = -1.0
    best_test = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(edge_index)

        # Sample negatives equal to number of positives per epoch
        neg = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=train_pos.size(0),
            method='sparse'
        ).t().to(device)

        pos_out = model.decode(z, train_pos)
        neg_out = model.decode(z, neg)

        # BCE with logits, balanced positives/negatives
        pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        if epoch % args.eval_steps == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                z_eval = model.encode(edge_index)
                val_hits = evaluate_hits20(model, edge_index, z_eval, valid_pos, valid_neg, evaluator)
                test_hits = evaluate_hits20(model, edge_index, z_eval, test_pos, test_neg, evaluator)

            improved = val_hits > best_val
            if improved:
                best_val = val_hits
                best_test = test_hits
                best_epoch = epoch
                epochs_no_improve = 0
                # Save checkpoint
                if args.ckpt_dir:
                    os.makedirs(args.ckpt_dir, exist_ok=True)
                    torch.save({
                        'model_state': model.state_dict(),
                        'epoch': epoch,
                        'val_hits20': val_hits,
                        'test_hits20': test_hits,
                        'args': vars(args),
                    }, os.path.join(args.ckpt_dir, f"best_{args.model}.pt"))
            else:
                epochs_no_improve += 1

            print(f"Epoch {epoch:04d} | loss {loss.item():.4f} | val@20 {val_hits:.4f} | test@20 {test_hits:.4f} | best val {best_val:.4f} (ep {best_epoch})")

            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {epochs_no_improve} evals)")
                break

    return {
        'best_val_hits20': best_val,
        'best_test_hits20': best_test,
        'best_epoch': float(best_epoch),
    }


def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI Link Prediction Trainer')
    parser.add_argument('--model', type=str, default='sage', choices=['sage', 'gcn', 'gcn_node2vec'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--no_bn', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--node2vec-dim', dest='node2vec_dim', type=int, default=64)
    parser.add_argument('--node2vec-walk-length', dest='node2vec_walk_length', type=int, default=20)
    parser.add_argument('--node2vec-context-size', dest='node2vec_context_size', type=int, default=10)
    parser.add_argument('--node2vec-walks-per-node', dest='node2vec_walks_per_node', type=int, default=10)
    parser.add_argument('--node2vec-negative-samples', dest='node2vec_negative_samples', type=int, default=1)
    parser.add_argument('--node2vec-epochs', dest='node2vec_epochs', type=int, default=30)
    parser.add_argument('--node2vec-batch-size', dest='node2vec_batch_size', type=int, default=256)
    parser.add_argument('--node2vec-lr', dest='node2vec_lr', type=float, default=0.01)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_val = []
    all_test = []
    for seed in args.seeds:
        print("\n" + "=" * 30)
        print(f"Running seed {seed}")
        print("=" * 30)
        set_seed(seed)
        stats = train_one_seed(args)
        all_val.append(stats['best_val_hits20'])
        all_test.append(stats['best_test_hits20'])
        print(f"Seed {seed} -> best val@20 {stats['best_val_hits20']:.4f} | best test@20 {stats['best_test_hits20']:.4f} (epoch {int(stats['best_epoch'])})")

    val_mean = float(np.mean(all_val))
    val_std = float(np.std(all_val))
    test_mean = float(np.mean(all_test))
    test_std = float(np.std(all_test))

    print("\n==== FINAL (mean ± std) across seeds ====")
    print(f"Val Hits@20:  {val_mean:.4f} ± {val_std:.4f}")
    print(f"Test Hits@20: {test_mean:.4f} ± {test_std:.4f}")


if __name__ == '__main__':
    main()
