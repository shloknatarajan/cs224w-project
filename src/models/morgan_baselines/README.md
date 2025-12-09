# Morgan Fingerprint-Based Baseline Models

This directory contains GNN baseline models that use **Morgan fingerprints** (2048-dimensional) as input features instead of learnable node embeddings.

## Models

All models follow the same architecture pattern:

1. **Input Layer**: Morgan fingerprints (2048-dim)
2. **Projection Layer**: Linear projection from 2048 → hidden_dim (default: 128)
3. **GNN Layers**: 2 layers of the respective GNN type
4. **Decoder**: Simple dot product or multi-strategy decoder

### Available Models

- **MorganGCN**: GCN with Morgan features
- **MorganGraphSAGE**: GraphSAGE with Morgan features
- **MorganGAT**: Graph Attention Network with Morgan features
- **MorganGraphTransformer**: Graph Transformer with Morgan features

## Usage

### 1. Prepare Data

First, ensure you have Morgan fingerprints loaded in your data:

```python
from src.data import load_dataset

data, split_edge, num_nodes, evaluator = load_dataset(
    'ogbl-ddi',
    device=device,
    smiles_csv_path="data/smiles.csv",  # CSV with columns: ogb_id, smiles
    feature_cache_path="data/morgan_features_2048.pt",  # Optional cache
    fp_n_bits=2048,
    fp_radius=2
)

# Verify features are loaded
print(f"Morgan features shape: {data.x.shape}")  # Should be [num_nodes, 2048]
```

### 2. Create Model

```python
from src.models.morgan_baselines import MorganGCN

model = MorganGCN(
    in_channels=2048,      # Morgan fingerprint dimension
    hidden_dim=128,        # Hidden dimension for GNN
    num_layers=2,          # Number of GNN layers
    dropout=0.0,           # Dropout (default 0.0)
    decoder_dropout=0.0,   # Decoder dropout (default 0.0)
    use_multi_strategy=False  # Use simple or multi-strategy decoder
)
```

### 3. Train

The existing trainer automatically detects if `data.x` exists and passes it to the model:

```python
from src.training import train_minimal_baseline

result = train_minimal_baseline(
    "Morgan-GCN",
    model,
    data,
    train_pos,
    valid_pos,
    valid_neg,
    test_pos,
    test_neg,
    evaluate_fn,
    device=device,
    epochs=200,
    lr=0.01,
    patience=20
)
```

### 4. Full Example

See `train_morgan_baselines.py` in the project root for a complete training script.

## Key Differences from Regular Baselines

| Feature | Regular Baselines | Morgan Baselines |
|---------|-------------------|------------------|
| Input | Learnable embeddings | Morgan fingerprints (2048-dim) |
| First layer | Direct to GNN | Projection layer (2048→128) |
| Encoder signature | `encode(edge_index)` | `encode(edge_index, x)` |
| Data requirement | No `data.x` needed | Requires `data.x` |

## Architecture Details

### Projection Layer

The projection layer maps high-dimensional Morgan fingerprints to a lower-dimensional space:

```
Morgan FP (2048) → Linear(2048, 128) → ReLU → GNN layers
```

This allows the GNN to work with more manageable feature dimensions while preserving the chemical information from the fingerprints.

### Comparison to Learnable Embeddings

**Advantages of Morgan features:**
- Chemistry-informed: Based on actual molecular structure
- No need to learn from scratch
- Better generalization to unseen molecules

**Potential disadvantages:**
- Fixed representation (can't adapt during training)
- Requires SMILES data
- Higher initial dimensionality

## Hyperparameters

Recommended starting values (same as regular baselines):

```python
IN_CHANNELS = 2048      # Morgan fingerprint size
HIDDEN_DIM = 128        # Hidden dimension
NUM_LAYERS = 2          # GNN layers
DROPOUT = 0.0           # No dropout (works well for dense graphs)
DECODER_DROPOUT = 0.0   # No decoder dropout
LEARNING_RATE = 0.01    # 0.005 for attention models
WEIGHT_DECAY = 1e-4
```

## Expected Performance

Performance will vary based on:
- Quality of SMILES data
- Morgan fingerprint parameters (radius, n_bits)
- Dataset characteristics

Compare against regular baselines to see if chemistry-informed features help!
