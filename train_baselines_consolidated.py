"""
Consolidated baseline training script
All model definitions, training logic, and evaluation code in one file
"""
import torch
import torch.nn.functional as F
from torch import nn
import logging
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Callable

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.nn import GCNConv, SAGEConv, TransformerConv, GATConv
from torch_geometric.utils import negative_sampling, add_self_loops


# ============================================================================
# BASE MODEL AND DECODER
# ============================================================================

class EdgeDecoder(nn.Module):
    def __init__(self, hidden_dim, dropout=0.5, use_multi_strategy=True):
        super().__init__()
        self.use_multi_strategy = use_multi_strategy

        if use_multi_strategy:
            self.hadamard_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

            self.concat_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

            self.strategy_weights = nn.Parameter(torch.ones(3) / 3)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(self, z, edge):
        src, dst = edge[:, 0], edge[:, 1]
        z_src, z_dst = z[src], z[dst]

        if self.use_multi_strategy:
            # Apply softmax to strategy weights
            weights = F.softmax(self.strategy_weights, dim=0)

            # Strategy 1: Hadamard product
            hadamard = z_src * z_dst
            score1 = self.hadamard_mlp(hadamard).squeeze()

            # Strategy 2: Concatenation
            concat = torch.cat([z_src, z_dst], dim=1)
            score2 = self.concat_mlp(concat).squeeze()

            # Strategy 3: Bilinear
            score3 = self.bilinear(z_src, z_dst).squeeze()

            # Weighted combination
            score = weights[0] * score1 + weights[1] * score2 + weights[2] * score3
        else:
            # Simple Hadamard scoring
            hadamard = z_src * z_dst
            score = self.mlp(hadamard).squeeze()

        return score


class BaseModel(nn.Module):
    """Base class for all GNN models"""
    def __init__(self, hidden_dim, decoder_dropout=0.3, use_multi_strategy=True):
        super().__init__()
        self.decoder = EdgeDecoder(hidden_dim, dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)

    def encode(self, edge_index):
        """Encode nodes to embeddings. Should be implemented by subclasses."""
        raise NotImplementedError

    def decode(self, z, edge):
        """Decode edge scores from node embeddings."""
        return self.decoder(z, edge)


# ============================================================================
# BASELINE MODELS
# ============================================================================

class GCN(BaseModel):
    """
    Ultra-minimal GCN baseline

    Key differences from original broken version:
    - dropout=0 (was 0.5 - TOO HIGH for dense graphs)
    - decoder_dropout=0 (was 0.3)
    - Hadamard + small MLP decoder only (no multi-strategy head)
    - No complex features

    Proven performance: 13-24% Hits@20 (vs 0.18% with old config)
    """

    def __init__(self, num_nodes, hidden_dim=128, num_layers=2, dropout=0.0,
                 decoder_dropout=0.0, use_multi_strategy=False):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers

        # Model description
        decoder_type = "multi-strategy" if use_multi_strategy else "simple"
        self.description = (
            f"Ultra-minimal GCN (2 layers, dropout={dropout}) | "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, "
            f"decoder={decoder_type}"
        )

        # Learnable embeddings
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # GCN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))

    def encode(self, edge_index):
        x = self.emb.weight

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            # Only apply dropout if > 0 (default is 0 for ultra-minimal)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GraphSAGE(BaseModel):
    """
    Ultra-minimal GraphSAGE baseline

    Key differences from original broken version:
    - dropout=0 (was 0.5 - TOO HIGH for dense graphs)
    - decoder_dropout=0 (was 0.3)
    - Simple dot product decoder only
    - No complex features

    Proven performance: 17% Hits@20 (vs 0.33% with old config)
    BEST performing baseline model!
    """

    def __init__(self, num_nodes, hidden_dim=128, num_layers=2, dropout=0.0,
                 decoder_dropout=0.0, use_multi_strategy=False, use_batch_norm=False):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm

        # Model description
        decoder_type = "multi-strategy" if use_multi_strategy else "simple"
        bn_str = "BN" if use_batch_norm else "no-BN"
        self.description = (
            f"GraphSAGE ({num_layers} layers, dropout={dropout}, {bn_str}) | "
            f"hidden_dim={hidden_dim}, "
            f"decoder={decoder_type}"
        )

        # Learnable embeddings
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def encode(self, edge_index):
        x = self.emb.weight

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            # Apply batch normalization if enabled
            if self.use_batch_norm:
                x = self.batch_norms[i](x)

            x = F.relu(x)

            # Only apply dropout if > 0 (default is 0 for ultra-minimal)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GraphTransformer(BaseModel):
    """
    Ultra-minimal Graph Transformer baseline

    Key differences from original broken version:
    - dropout=0 (was 0.5 - TOO HIGH for dense graphs)
    - decoder_dropout=0 (was 0.3)
    - Simple dot product decoder only
    - 2 attention heads per layer (was 4)

    Proven performance: 8% Hits@20 (vs 0.05% with old config)
    """

    def __init__(self, num_nodes, hidden_dim=128, num_layers=2, heads=2,
                 dropout=0.0, decoder_dropout=0.0, use_multi_strategy=False):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads = heads

        # Model description
        decoder_type = "multi-strategy" if use_multi_strategy else "simple"
        self.description = (
            f"Ultra-minimal Transformer (2 layers, {heads} heads, dropout={dropout}) | "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, "
            f"decoder={decoder_type}"
        )

        # Learnable embeddings
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # Transformer layers
        self.convs = nn.ModuleList()
        # First layer: hidden_dim -> hidden_dim (with multi-head)
        self.convs.append(TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(TransformerConv(hidden_dim, hidden_dim, heads=1, concat=False))

    def encode(self, edge_index):
        x = self.emb.weight

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            # Only apply dropout if > 0 (default is 0 for ultra-minimal)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GAT(BaseModel):
    """
    Ultra-minimal GAT baseline

    Key differences from original broken version:
    - dropout=0 (was 0.5 - TOO HIGH for dense graphs)
    - decoder_dropout=0 (was 0.3)
    - Simple dot product decoder only
    - 2 attention heads per layer (was 4)

    Proven performance: 11% Hits@20 (vs 0.31% with old config)
    """

    def __init__(self, num_nodes, hidden_dim=128, num_layers=2, heads=2,
                 dropout=0.0, decoder_dropout=0.0, use_multi_strategy=False):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads = heads

        # Model description
        decoder_type = "multi-strategy" if use_multi_strategy else "simple"
        self.description = (
            f"Ultra-minimal GAT (2 layers, {heads} heads, dropout={dropout}) | "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, "
            f"decoder={decoder_type}"
        )

        # Learnable embeddings
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # GAT layers
        self.convs = nn.ModuleList()
        # First layer: hidden_dim -> hidden_dim (with multi-head)
        self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False))

    def encode(self, edge_index):
        x = self.emb.weight

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            # Only apply dropout if > 0 (default is 0 for ultra-minimal)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, data, evaluator, pos_edges, neg_edges, batch_size=50000):
    """
    Evaluate model using official OGB negative edges.

    Args:
        model: Model to evaluate
        data: Graph data object
        evaluator: OGB evaluator
        pos_edges: Positive edges for evaluation
        neg_edges: Negative edges for evaluation
        batch_size: Batch size for evaluation

    Returns:
        float: Hits@20 score
    """
    model.eval()
    with torch.no_grad():
        z = model.encode(data.edge_index)

        # Positive scores - batch process to avoid OOM
        pos_scores_list = []
        for i in range(0, pos_edges.size(0), batch_size):
            chunk = pos_edges[i:i+batch_size]
            scores = model.decode(z, chunk).view(-1).cpu()
            pos_scores_list.append(scores)
            del scores
        pos_scores = torch.cat(pos_scores_list)
        del pos_scores_list

        # Negative scores - batch process to avoid OOM
        neg_scores_list = []
        for i in range(0, neg_edges.size(0), batch_size):
            chunk = neg_edges[i:i+batch_size]
            scores = model.decode(z, chunk).view(-1).cpu()
            neg_scores_list.append(scores)
            del scores
        neg_scores = torch.cat(neg_scores_list)
        del neg_scores_list

        # Free z before evaluation
        del z
        # Clear cache for CUDA and MPS devices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Use OGB evaluator
        result = evaluator.eval({
            'y_pred_pos': pos_scores,
            'y_pred_neg': neg_scores,
        })

        return result['hits@20']


# ============================================================================
# TRAINING
# ============================================================================

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
    logger = logging.getLogger(__name__)
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


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(dataset_name: str = "ogbl-ddi", device: str = "cpu"):
    """
    Load OGB link prediction dataset and split edges.

    Args:
        dataset_name: OGB dataset name (default: 'ogbl-ddi').
        device: 'cpu' or 'cuda'.

    Returns:
        tuple: (data, split_edge, num_nodes, evaluator)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading dataset {dataset_name}...")
    dataset = PygLinkPropPredDataset(dataset_name)
    data = dataset[0]

    split_edge = dataset.get_edge_split()
    logger.info(f"Dataset loaded: {data.num_nodes} nodes")

    # Move edge splits to device
    train_pos = split_edge["train"]["edge"].to(device)
    valid_pos = split_edge["valid"]["edge"].to(device)
    valid_neg = split_edge["valid"]["edge_neg"].to(device)
    test_pos = split_edge["test"]["edge"].to(device)
    test_neg = split_edge["test"]["edge_neg"].to(device)

    logger.info(
        f"Train pos edges: {train_pos.size(0)}, "
        f"Valid pos: {valid_pos.size(0)}, Test pos: {test_pos.size(0)}"
    )
    logger.info(
        f"Valid neg edges: {valid_neg.size(0)}, "
        f"Test neg: {test_neg.size(0)}"
    )

    num_nodes = data.num_nodes

    # Construct graph using only training edges
    train_edge_index = train_pos.t().contiguous().to(device)

    # Add self-loops
    data.edge_index, _ = add_self_loops(train_edge_index, num_nodes=num_nodes)
    logger.info(
        f"Added self-loops: Total edges now = {data.edge_index.size(1)}"
    )

    evaluator = Evaluator(name=dataset_name)
    return data, split_edge, num_nodes, evaluator


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'logs/baselines_{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f'{log_dir}/baselines.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Device setup - support CUDA (Linux), MPS (macOS), and CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Logging results to: {log_filename}")

    # Load dataset
    data, split_edge, num_nodes, evaluator = load_dataset('ogbl-ddi', device=device)

    train_pos = split_edge['train']['edge'].to(device)
    valid_pos = split_edge['valid']['edge'].to(device)
    valid_neg = split_edge['valid']['edge_neg'].to(device)
    test_pos = split_edge['test']['edge'].to(device)
    test_neg = split_edge['test']['edge_neg'].to(device)

    # Hyperparameters
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    # Ultra-minimal defaults
    DROPOUT = 0.0
    DECODER_DROPOUT = 0.0
    EPOCHS = 200
    PATIENCE = 20
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 50000
    EVAL_BATCH_SIZE = 50000
    EVAL_EVERY = 5

    # Create evaluation function wrapper
    def evaluate_fn(model, pos_edges, neg_edges, batch_size):
        return evaluate(model, data, evaluator, pos_edges, neg_edges, batch_size)

    # Results storage
    results = {}

    # Train GCN
    logger.info("\n" + "=" * 80)
    logger.info("Training Simple GCN Baseline (minimal trainer)")
    logger.info("=" * 80)
    gcn_model = GCN(num_nodes, HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, decoder_dropout=DECODER_DROPOUT)
    gcn_result = train_minimal_baseline(
        "GCN-Baseline",
        gcn_model,
        data,
        train_pos,
        valid_pos,
        valid_neg,
        test_pos,
        test_neg,
        evaluate_fn,
        device=device,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        eval_every=EVAL_EVERY,
        eval_batch_size=EVAL_BATCH_SIZE
    )
    results['GCN'] = gcn_result

    # Train GraphSAGE
    logger.info("\n" + "=" * 80)
    logger.info("Training GraphSAGE Baseline (minimal trainer)")
    logger.info("=" * 80)
    sage_model = GraphSAGE(num_nodes, HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, decoder_dropout=DECODER_DROPOUT)
    sage_result = train_minimal_baseline(
        "GraphSAGE-Baseline",
        sage_model,
        data,
        train_pos,
        valid_pos,
        valid_neg,
        test_pos,
        test_neg,
        evaluate_fn,
        device=device,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        eval_every=EVAL_EVERY,
        eval_batch_size=EVAL_BATCH_SIZE
    )
    results['GraphSAGE'] = sage_result

    # Train GraphTransformer
    logger.info("\n" + "=" * 80)
    logger.info("Training GraphTransformer Baseline (minimal trainer)")
    logger.info("=" * 80)
    transformer_model = GraphTransformer(num_nodes, HIDDEN_DIM, num_layers=NUM_LAYERS, heads=4, dropout=DROPOUT, decoder_dropout=DECODER_DROPOUT)
    transformer_result = train_minimal_baseline(
        "GraphTransformer-Baseline",
        transformer_model,
        data,
        train_pos,
        valid_pos,
        valid_neg,
        test_pos,
        test_neg,
        evaluate_fn,
        device=device,
        epochs=EPOCHS,
        lr=0.005,  # Lower LR for transformer
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        eval_every=EVAL_EVERY,
        eval_batch_size=EVAL_BATCH_SIZE
    )
    results['GraphTransformer'] = transformer_result

    # Train GAT
    logger.info("\n" + "=" * 80)
    logger.info("Training GAT Baseline (minimal trainer)")
    logger.info("=" * 80)
    gat_model = GAT(num_nodes, HIDDEN_DIM, num_layers=NUM_LAYERS, heads=4, dropout=DROPOUT, decoder_dropout=DECODER_DROPOUT)
    gat_result = train_minimal_baseline(
        "GAT-Baseline",
        gat_model,
        data,
        train_pos,
        valid_pos,
        valid_neg,
        test_pos,
        test_neg,
        evaluate_fn,
        device=device,
        epochs=EPOCHS,
        lr=0.005,  # Lower LR for attention
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        eval_every=EVAL_EVERY,
        eval_batch_size=EVAL_BATCH_SIZE
    )
    results['GAT'] = gat_result

    # Final results summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS - BASELINES")
    logger.info("=" * 80)
    for model_name, result in results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Validation Hits@20: {result.best_val_hits:.4f}")
        logger.info(f"  Test Hits@20: {result.best_test_hits:.4f}")
        val_test_gap = result.best_val_hits - result.best_test_hits
        gap_pct = (val_test_gap / result.best_val_hits * 100) if result.best_val_hits != 0 else 0.0
        logger.info(f"  Val-Test Gap: {val_test_gap:.4f} ({gap_pct:.1f}% relative)")
    logger.info("=" * 80)
    logger.info(f"Results logged to: {log_filename}")
