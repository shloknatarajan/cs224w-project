# Baseline Models - Performance Summary

**Last Updated:** 2025-12-08

## Performance Results

All models trained on OGBL-DDI with minimal configuration (2 layers, 128 hidden dim, 100 epochs):

| Model | Val Hits@20 | Test Hits@20 | Description |
|-------|-------------|--------------|-------------|
| **GraphSAGE** | **17.13%** | **21.52%** | Mean aggregation, best baseline |
| **GCN** | 13.34% | 17.45% | Graph convolution |
| **GAT** | 10.99% | 6.24% | Multi-head attention (2 heads) |
| **Transformer** | 8.38% | 15.81% | Self-attention mechanism |

## Models

### GraphSAGE (Best Baseline)
- **File:** `sage.py`
- **Architecture:** 2 SAGEConv layers, mean aggregation
- **Use Case:** Start here for improvements

### GCN (Graph Convolutional Network)
- **File:** `gcn.py`
- **Architecture:** 2 GCNConv layers
- **Use Case:** Simple baseline

### GAT (Graph Attention Network)
- **File:** `gat.py`
- **Architecture:** 2 GAT layers with 2 attention heads
- **Use Case:** Attention-based aggregation

### GraphTransformer
- **File:** `transformer.py`
- **Architecture:** 2 Transformer layers with 2 attention heads
- **Use Case:** Full self-attention over graph

## Usage

```python
from src.models.baselines import GraphSAGE, GCN, GAT, GraphTransformer

# Create model
model = GraphSAGE(num_nodes=4267, hidden_dim=128)

# Encode nodes
z = model.encode(edge_index)

# Decode edge scores
scores = model.decode(z, edge_pairs)
```

## Configuration

Default settings for all models:
- Hidden dimension: 128
- Number of layers: 2
- Dropout: 0.0
- Decoder: Dot product (z_u · z_v)
- Optimizer: Adam (lr=0.01)
- Embeddings: Learnable only (no structural features)
