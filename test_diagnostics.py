"""
Quick test to verify diagnostic logging is working correctly.
Trains a single GCN model for 15 epochs to test the diagnostic system.
"""
import torch
import logging
import os
from datetime import datetime

from src.data import load_dataset
from src.models.baselines import GCN
from src.training import train_model
from src.evals import evaluate

# Setup logging
os.makedirs('logs', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'logs/diagnostic_test_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
logger.info(f"Using device: {device}")
logger.info(f"Logging results to: {log_filename}")

# Load dataset
logger.info("Loading dataset...")
data, split_edge, num_nodes, evaluator = load_dataset('ogbl-ddi', device=device)

train_pos = split_edge['train']['edge'].to(device)
valid_pos = split_edge['valid']['edge'].to(device)
valid_neg = split_edge['valid']['edge_neg'].to(device)
test_pos = split_edge['test']['edge'].to(device)
test_neg = split_edge['test']['edge_neg'].to(device)

# Hyperparameters - reduced for quick testing
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.5
DECODER_DROPOUT = 0.3
EPOCHS = 15  # Just 15 epochs to test
PATIENCE = 20
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 50000
EVAL_BATCH_SIZE = 50000
GRADIENT_ACCUMULATION_STEPS = 1

# Create evaluation function wrapper
def evaluate_fn(model, pos_edges, neg_edges, batch_size):
    return evaluate(model, data, evaluator, pos_edges, neg_edges, batch_size)

# Train GCN with diagnostics
logger.info("\n" + "=" * 80)
logger.info("Testing Diagnostic System with GCN")
logger.info("=" * 80)
gcn_model = GCN(num_nodes, HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, decoder_dropout=DECODER_DROPOUT)
gcn_val, gcn_test = train_model(
    "GCN-DiagTest",
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

logger.info("\n" + "=" * 80)
logger.info("DIAGNOSTIC TEST COMPLETE")
logger.info("=" * 80)
logger.info(f"Final Val Hits@20: {gcn_val:.4f}")
logger.info(f"Final Test Hits@20: {gcn_test:.4f}")
logger.info(f"Results logged to: {log_filename}")
logger.info("\nPlease review the diagnostic warnings above to identify any issues.")
