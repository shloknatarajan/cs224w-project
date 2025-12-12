# Training Guide

This document describes how to train models using the unified `train.py` script.

**Available Models**: 14 total
- 4 baseline models (GCN, SAGE, Transformer, GAT)
- 5 chemistry models (ChemBERTa/Morgan-based) - GNN with drug features
- 4 hybrid models (structure + chemistry decoder)
- 2 advanced models (GDINN, GCNAdvanced)

## Quick Start

```bash
# Train baseline GCN (structure-only)
python train.py --model gcn

# Train GCN with drug features (ChemBERTa)
python train.py --model chemberta-gcn --external-features chemberta

# Train hybrid model (better GCN with chemistry-aware decoder)
python train.py --model hybrid-gcn --external-features morgan,chemberta

# Train GDINN with all features (best model)
python train.py --model gdinn --external-features all
```

## Available Models

### Baseline Models
Simple structure-only models using learnable node embeddings:
- `gcn` - Graph Convolutional Network
- `sage` - GraphSAGE
- `transformer` - Graph Transformer
- `gat` - Graph Attention Network

### Chemistry Models (GNN with Drug Features)
Models that use molecular features as node input instead of learnable embeddings:
- `chemberta-gcn` - GCN with ChemBERTa embeddings (768-dim)
- `chemberta-sage` - GraphSAGE with ChemBERTa embeddings
- `chemberta-gat` - GAT with ChemBERTa embeddings
- `chemberta-transformer` - Graph Transformer with ChemBERTa embeddings
- `morgan-gcn` - GCN with Morgan fingerprints (2048-dim)

**Note**: These models require `--external-features chemberta` or `morgan`.

### Hybrid Models (Structure + Chemistry Decoder)
Best of both worlds - structure-only encoding with chemistry-aware decoding:
- `hybrid-gcn` - Hybrid GCN
- `hybrid-sage` - Hybrid GraphSAGE
- `hybrid-transformer` - Hybrid Graph Transformer
- `hybrid-gat` - Hybrid GAT

**Note**: Work best with external features but can run without them.

### Advanced Models
- `gdinn` - Graph Drug Interaction Neural Network (requires external features, best performance)
- `gcn-advanced` - OGB reference GCN implementation (structure-only)

## External Features

Control which external features to use with `--external-features`:

```bash
# Use all available features
--external-features all

# Use no features (structure only)
--external-features none

# Use specific features (comma-separated)
--external-features morgan,chemberta
--external-features morgan,pubchem,drug-targets
```

Available features:
- `morgan` - Morgan fingerprints (molecular substructure)
- `pubchem` - PubChem physicochemical properties
- `chemberta` - ChemBERTa pre-trained embeddings
- `drug-targets` - Drug-target interaction features

## Common Training Options

### Model Architecture
```bash
--hidden-dim 256          # Hidden dimension (default: 128)
--num-layers 3            # Number of GNN layers (default: 2)
--dropout 0.1             # Dropout rate (default: 0.0)
--decoder-dropout 0.1     # Decoder dropout (default: 0.0)
--attention-heads 8       # Attention heads for GAT/Transformer (default: 4)
--fusion concat           # Feature fusion: concat, add (default: concat) - GDINN only
```

### Training Configuration
```bash
--epochs 300              # Maximum epochs (default: 200)
--lr 0.001                # Learning rate (default: 0.01)
--weight-decay 5e-4       # Weight decay (default: 1e-4)
--batch-size 64000        # Training batch size (default: 50000)
--eval-batch-size 64000   # Evaluation batch size (default: 50000)
--patience 30             # Early stopping patience (default: 20)
--eval-every 10           # Evaluate every N epochs (default: 5)
```

### System Configuration
```bash
--device cuda             # Device: cpu, cuda, mps (default: auto-detect)
--data-dir ./data         # Data directory (default: data)
--seed 42                 # Random seed (default: 42)
```

## Example Workflows

### Train Baseline Models

```bash
# GCN with default settings
python train.py --model gcn

# GraphSAGE with more layers and higher learning rate
python train.py --model sage --num-layers 3 --lr 0.02

# Graph Transformer with attention heads
python train.py --model transformer --attention-heads 8 --lr 0.005
```

### Train Chemistry Models (GNN with Drug Features)

```bash
# GCN with ChemBERTa embeddings
python train.py --model chemberta-gcn --external-features chemberta

# GraphSAGE with ChemBERTa
python train.py --model chemberta-sage --external-features chemberta

# GCN with Morgan fingerprints
python train.py --model morgan-gcn --external-features morgan

# GAT with ChemBERTa embeddings
python train.py --model chemberta-gat \
  --external-features chemberta \
  --attention-heads 8
```

### Train with External Features

```bash
# GDINN with all features (best model)
python train.py --model gdinn --external-features all

# GDINN with Morgan fingerprints only
python train.py --model gdinn --external-features morgan

# GCNAdvanced (structure-only, OGB reference implementation)
python train.py --model gcn-advanced --hidden-dim 256 --dropout 0.5

# Hybrid model with ChemBERTa
python train.py --model hybrid-gcn --external-features chemberta
```

### Hyperparameter Tuning

```bash
# Larger model with regularization
python train.py --model gcn \
  --hidden-dim 256 \
  --num-layers 3 \
  --dropout 0.2 \
  --decoder-dropout 0.1 \
  --weight-decay 5e-4

# Training with custom schedule
python train.py --model sage \
  --epochs 500 \
  --patience 50 \
  --eval-every 10 \
  --lr 0.005
```

### Best Performing Configuration (GDINN)

```bash
# Train GDINN with recommended settings (achieved 73.28% test Hits@20)
python train.py --model gdinn \
  --external-features all \
  --hidden-dim 256 \
  --dropout 0.5 \
  --lr 0.005 \
  --batch-size 65536 \
  --epochs 2000
```

## Output and Logging

Training progress is logged to both console and file:
- Log directory: `logs/{model}_{features}_{timestamp}/`
- Log file: `logs/{model}_{features}_{timestamp}/{model}.log`

Example log directory names:
- `logs/gcn_20231211_140530/`
- `logs/gdinn_all_features_20231211_140530/`
- `logs/hybrid-sage_morgan_chemberta_20231211_140530/`

## Advanced Usage

### GDINN-Specific Options

```bash
# GDINN with custom fusion strategy
python train.py --model gdinn \
  --external-features all \
  --num-neg 5 \
  --fusion concat

# GDINN with add fusion (alternative to concat)
python train.py --model gdinn \
  --external-features morgan,chemberta \
  --fusion add
```

### Device Selection

```bash
# Force CPU (for debugging)
python train.py --model gcn --device cpu

# Use CUDA (Linux/Windows with NVIDIA GPU)
python train.py --model gcn --device cuda

# Use MPS (macOS with Apple Silicon)
python train.py --model gcn --device mps
```

## Getting Help

```bash
# View all options
python train.py --help

# View model-specific help
python train.py --model gdin --help
```

## Migrating from Old Scripts

If you were using the old training scripts, here's how to migrate:

### `train_baselines.py` → `train.py`
```bash
# Old
python train_baselines.py

# New
python train.py --model gcn
python train.py --model sage
python train.py --model transformer
python train.py --model gat
```

### `train_gdin_external.py` → `train.py`
```bash
# Old
python train_gdin_external.py --all

# New
python train.py --model gdinn --external-features all
```

### `train_ddi_reference_external.py` → `train.py`
```bash
# Old
python src/training/scripts/train_ddi_reference_external.py --morgan --chemberta

# New
python train.py --model gdinn --external-features morgan,chemberta
```

## Troubleshooting

### Missing External Features
If you see errors about missing external features, run the data preparation scripts:

```bash
python -m src.data.fetch_smiles
python -m src.data.fetch_pubchem_properties
python -m src.data.extract_chemberta_embeddings
python -m src.data.fetch_drug_targets
```

### Out of Memory
Reduce batch sizes:

```bash
python train.py --model gcn \
  --batch-size 32000 \
  --eval-batch-size 32000
```

### Training Too Slow
Increase batch size and reduce evaluation frequency:

```bash
python train.py --model gcn \
  --batch-size 100000 \
  --eval-every 10
```

## Reference

Old training scripts are preserved in `src/training/scripts/` for reference:
- `src/training/scripts/train_baselines.py`
- `src/training/scripts/train_ddi_reference.py`
- `src/training/scripts/train_ddi_reference_external.py`
- `src/training/scripts/train_gdin_external.py`
- `src/training/scripts/trainer.py`
