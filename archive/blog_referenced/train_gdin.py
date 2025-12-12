"""
Train GDIN (Graph Drug Interaction Network) model

Phase 1: AUC loss only (target: 20-25% Hits@20)
Phase 2: Add common neighbor features (target: 30-40% Hits@20)
"""
import torch
import logging
import os
from datetime import datetime

from src.data import load_dataset
from src.models.advanced import GDIN
from src.training.gdin_trainer import train_gdin, GDINRunResult
from src.evals import evaluate

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f'logs/gdin_{timestamp}'
os.makedirs(log_dir, exist_ok=True)
log_filename = f'{log_dir}/gdin.log'

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

# =============================================================================
# GDIN Hyperparameters (Phase 1)
# =============================================================================
HIDDEN_DIM = 256        # More capacity than baseline (128)
NUM_LAYERS = 2          # Keep simple, avoid over-smoothing
DROPOUT = 0.0           # Dense graph already regularizes
NUM_NEG = 3             # PLNLP uses 3 negatives per positive
LR = 0.005              # Slightly lower for ranking loss stability
WEIGHT_DECAY = 1e-4     # Same as baseline
BATCH_SIZE = 50000      # Same as baseline
EVAL_BATCH_SIZE = 50000
EPOCHS = 400            # More epochs for ranking loss
PATIENCE = 30           # More patience for ranking loss convergence
EVAL_EVERY = 5

# Phase 2 settings (set USE_CN=True to enable)
USE_CN = False  # Enable in Phase 2

logger.info("=" * 80)
logger.info("GDIN Configuration")
logger.info("=" * 80)
logger.info(f"Hidden dim: {HIDDEN_DIM}")
logger.info(f"Num layers: {NUM_LAYERS}")
logger.info(f"Dropout: {DROPOUT}")
logger.info(f"Num negatives: {NUM_NEG}")
logger.info(f"Learning rate: {LR}")
logger.info(f"Epochs: {EPOCHS}")
logger.info(f"Use CN features: {USE_CN}")
logger.info("=" * 80)

# Create evaluation function wrapper
def evaluate_fn(model, pos_edges, neg_edges, batch_size):
    return evaluate(model, data, evaluator, pos_edges, neg_edges, batch_size)

# Build sparse adjacency for CN computation (Phase 2)
adj_sparse = None
if USE_CN:
    from scipy.sparse import csr_matrix
    import numpy as np

    logger.info("Building sparse adjacency matrix for CN computation...")
    edge_index = data.edge_index.cpu()
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    adj_sparse = csr_matrix(
        (np.ones(len(row)), (row, col)),
        shape=(num_nodes, num_nodes)
    )
    logger.info(f"Adjacency matrix: {adj_sparse.shape}, nnz={adj_sparse.nnz}")

# Create GDIN model
logger.info("\n" + "=" * 80)
logger.info("Training GDIN (Phase 1: AUC Loss)")
logger.info("=" * 80)

model = GDIN(
    num_nodes=num_nodes,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    use_cn=USE_CN
)
logger.info(f"Model: {model.description}")

# Train
result = train_gdin(
    "GDIN",
    model,
    data,
    train_pos,
    valid_pos,
    valid_neg,
    test_pos,
    test_neg,
    evaluate_fn,
    device=device,
    epochs=EPOCHS,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    eval_every=EVAL_EVERY,
    num_neg=NUM_NEG,
    use_cn=USE_CN,
    adj_sparse=adj_sparse
)

# Final results
logger.info("\n" + "=" * 80)
logger.info("FINAL RESULTS - GDIN")
logger.info("=" * 80)
logger.info(f"Best Validation Hits@20: {result.best_val_hits:.4f}")
logger.info(f"Best Test Hits@20: {result.best_test_hits:.4f}")
logger.info(f"Best Epoch: {result.best_epoch}")
val_test_gap = result.best_val_hits - result.best_test_hits
gap_pct = (val_test_gap / result.best_val_hits * 100) if result.best_val_hits != 0 else 0.0
logger.info(f"Val-Test Gap: {val_test_gap:.4f} ({gap_pct:.1f}% relative)")
logger.info("=" * 80)
logger.info(f"Results logged to: {log_filename}")
