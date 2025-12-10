"""
Train Hybrid-GraphSAGE-Simple to compare with GCN
"""
import os
import logging
import torch
from datetime import datetime

from src.data.data_loader import load_dataset_chemberta
from src.models.hybrid import HybridGraphSAGE
from src.training.hybrid_trainer import train_hybrid_model

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/hybrid_sage_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "hybrid_sage.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
logger.info(f"Logging results to: {log_file}")

# Load dataset with ChemBERTa features
logger.info("Loading dataset with ChemBERTa embeddings + SMILES mask...")
data, split_edge, num_nodes, evaluator = load_dataset_chemberta(
    dataset_name="ogbl-ddi",
    device=device,
    smiles_csv_path="data/smiles.csv",
    feature_cache_path="data/chemberta_features_768.pt",
    chemberta_model="seyonec/ChemBERTa-zinc-base-v1",
    batch_size=32,
)

logger.info(f"Dataset loaded: data.x.shape = {data.x.shape}")
logger.info(f"SMILES mask: {data.smiles_mask.sum().item():.0f} valid / {data.smiles_mask.size(0)} total ({100*data.smiles_mask.sum().item()/data.smiles_mask.size(0):.1f}%)")

# Training configuration
config = {
    'epochs': 200,
    'lr': 0.01,
    'weight_decay': 1e-4,
    'batch_size': 5000,
    'eval_every': 5,
    'patience': 20,
}

# Model
name = 'Hybrid-GraphSAGE-Simple'
logger.info("\n" + "="*80)
logger.info(f"Training {name}")
logger.info("="*80)

model = HybridGraphSAGE(
    num_nodes=data.num_nodes,
    hidden_dim=128,
    num_layers=2,
    dropout=0.0,
    chemical_dim=768,
    decoder_type='simple',  # Simple decoder like best GCN
    decoder_dropout=0.3,
    use_gating=False,
).to(device)

logger.info(f"Model: {model.description}")

# Train
result = train_hybrid_model(
    name=name,
    model=model,
    data=data,
    train_pos=split_edge['train']['edge'].to(device),
    valid_pos=split_edge['valid']['edge'].to(device),
    valid_neg=split_edge['valid']['edge_neg'].to(device),
    test_pos=split_edge['test']['edge'].to(device),
    test_neg=split_edge['test']['edge_neg'].to(device),
    evaluate_fn=None,
    device=device,
    epochs=config['epochs'],
    lr=config['lr'],
    weight_decay=config['weight_decay'],
    batch_size=config['batch_size'],
    eval_every=config['eval_every'],
    patience=config['patience'],
)

# Print results
logger.info("\n" + "="*80)
logger.info("FINAL RESULTS - HYBRID GRAPHSAGE VS GCN")
logger.info("="*80)
logger.info("")
logger.info("Previous Results:")
logger.info(f"  Baseline GCN:              val@20=0.1118")
logger.info(f"  Hybrid-GCN-Simple:         val@20=0.2349  test@20=0.1933")
logger.info("")
logger.info("New Result:")
logger.info(f"  {name}:  val@20={result.best_val_hits:.4f}  test@20={result.best_test_hits:.4f}")
logger.info("")

# Calculate improvement
improvement = ((result.best_val_hits - 0.2349) / 0.2349) * 100
if improvement > 0:
    logger.info(f"  ✨ GraphSAGE is {improvement:.1f}% BETTER than GCN!")
elif improvement < -2:
    logger.info(f"  ⚠️  GraphSAGE is {abs(improvement):.1f}% worse than GCN")
else:
    logger.info(f"  ≈ GraphSAGE is similar to GCN ({improvement:+.1f}%)")

logger.info("")
logger.info("="*80)
logger.info(f"Results logged to: {log_file}")

