"""
Train baseline models (simple GCN, GraphSAGE, GraphTransformer, GAT)
"""
import torch
import logging
import os
from datetime import datetime

from src.data import load_dataset
from src.models.baselines import GCN, GraphSAGE, GraphTransformer, GAT
from src.training import train_model
from src.evals import evaluate

# Setup logging
os.makedirs('logs', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'logs/baselines_{timestamp}.log'

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
DROPOUT = 0.5
DECODER_DROPOUT = 0.3
EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 50000
EVAL_BATCH_SIZE = 50000
GRADIENT_ACCUMULATION_STEPS = 1

# Create evaluation function wrapper
def evaluate_fn(model, pos_edges, neg_edges, batch_size):
    return evaluate(model, data, evaluator, pos_edges, neg_edges, batch_size)

# Results storage
results = {}

# Train GCN
logger.info("\n" + "=" * 80)
logger.info("Training Simple GCN Baseline")
logger.info("=" * 80)
gcn_model = GCN(num_nodes, HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, decoder_dropout=DECODER_DROPOUT)
gcn_val, gcn_test = train_model(
    "GCN-Baseline",
    gcn_model,
    data,
    train_pos,
    valid_pos,
    valid_neg,
    test_pos,
    test_neg,
    num_nodes,
    evaluate_fn,
    device=device,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    patience=PATIENCE,
    use_hard_negatives=False,
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    weight_decay=WEIGHT_DECAY
)
results['GCN'] = (gcn_val, gcn_test)

# Train GraphSAGE
logger.info("\n" + "=" * 80)
logger.info("Training GraphSAGE Baseline")
logger.info("=" * 80)
sage_model = GraphSAGE(num_nodes, HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, decoder_dropout=DECODER_DROPOUT)
sage_val, sage_test = train_model(
    "GraphSAGE-Baseline",
    sage_model,
    data,
    train_pos,
    valid_pos,
    valid_neg,
    test_pos,
    test_neg,
    num_nodes,
    evaluate_fn,
    device=device,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    patience=PATIENCE,
    use_hard_negatives=False,
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    weight_decay=WEIGHT_DECAY
)
results['GraphSAGE'] = (sage_val, sage_test)

# Train GraphTransformer
logger.info("\n" + "=" * 80)
logger.info("Training GraphTransformer Baseline")
logger.info("=" * 80)
transformer_model = GraphTransformer(num_nodes, HIDDEN_DIM, num_layers=NUM_LAYERS, heads=4, dropout=DROPOUT, decoder_dropout=DECODER_DROPOUT)
transformer_val, transformer_test = train_model(
    "GraphTransformer-Baseline",
    transformer_model,
    data,
    train_pos,
    valid_pos,
    valid_neg,
    test_pos,
    test_neg,
    num_nodes,
    evaluate_fn,
    device=device,
    epochs=EPOCHS,
    lr=0.005,  # Lower LR for transformer
    patience=PATIENCE,
    use_hard_negatives=False,
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    weight_decay=WEIGHT_DECAY
)
results['GraphTransformer'] = (transformer_val, transformer_test)

# Train GAT
logger.info("\n" + "=" * 80)
logger.info("Training GAT Baseline")
logger.info("=" * 80)
gat_model = GAT(num_nodes, HIDDEN_DIM, num_layers=NUM_LAYERS, heads=4, dropout=DROPOUT, decoder_dropout=DECODER_DROPOUT)
gat_val, gat_test = train_model(
    "GAT-Baseline",
    gat_model,
    data,
    train_pos,
    valid_pos,
    valid_neg,
    test_pos,
    test_neg,
    num_nodes,
    evaluate_fn,
    device=device,
    epochs=EPOCHS,
    lr=0.005,  # Lower LR for attention
    patience=PATIENCE,
    use_hard_negatives=False,
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    weight_decay=WEIGHT_DECAY
)
results['GAT'] = (gat_val, gat_test)

# Final results summary
logger.info("\n" + "=" * 80)
logger.info("FINAL RESULTS - BASELINES")
logger.info("=" * 80)
for model_name, (val_hits, test_hits) in results.items():
    logger.info(f"{model_name}:")
    logger.info(f"  Validation Hits@20: {val_hits:.4f}")
    logger.info(f"  Test Hits@20: {test_hits:.4f}")
    val_test_gap = val_hits - test_hits
    logger.info(f"  Val-Test Gap: {val_test_gap:.4f} ({val_test_gap/val_hits*100:.1f}% relative)")
logger.info("=" * 80)
logger.info(f"Results logged to: {log_filename}")
