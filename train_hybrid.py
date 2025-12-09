"""
Train hybrid models: Structure-only encoder + Chemistry-aware decoder

These models test whether chemistry can be intelligently combined with topology
by keeping the encoder structure-only (like strong baselines) but incorporating
chemistry at the decoder level.
"""
import os
import logging
import torch
from datetime import datetime
from ogb.linkproppred import Evaluator

from src.data.data_loader import load_dataset_chemberta
from src.models.hybrid import HybridGCN
from src.training.hybrid_trainer import train_hybrid_model

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/hybrid_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "hybrid.log")

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

logger.info(f"Dataset loaded with ChemBERTa features: data.x.shape = {data.x.shape}")
logger.info(f"SMILES mask: {data.smiles_mask.sum().item():.0f} valid / {data.smiles_mask.size(0)} total ({100*data.smiles_mask.sum().item()/data.smiles_mask.size(0):.1f}%)")

# Training configuration
config = {
    'epochs': 200,
    'lr': 0.01,
    'weight_decay': 1e-4,
    'batch_size': 10000,  # Reduced from 50000 to avoid OOM with chemistry features
    'eval_every': 5,
    'patience': 20,
}

# Models to train (GCN variants only)
models_config = [
    # Chemistry-aware decoder (full version with 3 paths)
    {
        'name': 'Hybrid-GCN-ChemAware',
        'class': HybridGCN,
        'params': {
            'num_nodes': data.num_nodes,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.0,
            'chemical_dim': 768,  # ChemBERTa dimension
            'decoder_type': 'chemistry_aware',
            'decoder_dropout': 0.3,
            'use_gating': False,
        },
        'lr': 0.01,
    },
    # Simple decoder (additive: structure + α * chemistry)
    {
        'name': 'Hybrid-GCN-Simple',
        'class': HybridGCN,
        'params': {
            'num_nodes': data.num_nodes,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.0,
            'chemical_dim': 768,
            'decoder_type': 'simple',
            'decoder_dropout': 0.3,
            'use_gating': False,
        },
        'lr': 0.01,
    },
    # With gating (optional: dynamic gates per edge)
    {
        'name': 'Hybrid-GCN-ChemAware-Gated',
        'class': HybridGCN,
        'params': {
            'num_nodes': data.num_nodes,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.0,
            'chemical_dim': 768,
            'decoder_type': 'chemistry_aware',
            'decoder_dropout': 0.3,
            'use_gating': True,  # Dynamic gates
        },
        'lr': 0.01,
    },
]

# Store results
results = {}

# Train each model
for model_config in models_config:
    name = model_config['name']
    logger.info("\n" + "="*80)
    logger.info(f"Training {name}")
    logger.info("="*80)
    
    # Create model
    model = model_config['class'](**model_config['params']).to(device)
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
        evaluate_fn=None,  # Not used, we have custom eval in hybrid_trainer
        device=device,
        epochs=config['epochs'],
        lr=model_config['lr'],
        weight_decay=config['weight_decay'],
        batch_size=config['batch_size'],
        eval_every=config['eval_every'],
        patience=config['patience'],
    )
    
    best_val = result.best_val_hits
    test_at_best = result.best_test_hits
    
    results[name] = {
        'val': best_val,
        'test': test_at_best,
    }
    
    logger.info(f"[{name}] Done. Best val@20={best_val:.4f} | test@20={test_at_best:.4f}")

# Print final results
logger.info("\n" + "="*80)
logger.info("FINAL RESULTS - HYBRID GCN MODELS")
logger.info("="*80)
logger.info("NOTE: These models use structure-only GCN encoder + chemistry-aware decoders")
logger.info("Compare with baseline GCN: val@20=0.1118")
logger.info("")

for name, res in results.items():
    val_test_gap = res['val'] - res['test']
    gap_pct = 100 * val_test_gap / res['val'] if res['val'] > 0 else 0
    logger.info(f"{name}:")
    logger.info(f"  Validation Hits@20: {res['val']:.4f}")
    logger.info(f"  Test Hits@20: {res['test']:.4f}")
    logger.info(f"  Val-Test Gap: {val_test_gap:.4f} ({gap_pct:.1f}% relative)")

logger.info("="*80)
logger.info(f"Results logged to: {log_file}")

