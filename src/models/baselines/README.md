# Ultra-Minimal Baseline Models

**Last Updated:** 2025-12-08

## Overview

These baseline models have been updated to use **ultra-minimal configurations** that actually work, replacing the broken originals that had dropout=0.5.

## Performance Summary

| Model | Old Val Hits@20 | New Val Hits@20 | Improvement |
|-------|----------------|-----------------|-------------|
| **GraphSAGE** | 0.33% | **17.13%** | **52x better** ✅ |
| **GCN** | 0.18% | **13.34%** | **74x better** ✅ |
| **GAT** | 0.31% | **10.99%** | **35x better** ✅ |
| **Transformer** | 0.05% | **8.38%** | **168x better** ✅ |

## Key Changes

### What Was Fixed

**Primary Issue: dropout=0.5 → 0.0**
- Old: 50% of neurons dropped every forward pass
- New: No dropout (or very light dropout if specified)
- **This was the main killer** - ogbl-ddi is too dense for high dropout

**Secondary Fixes:**
- decoder_dropout: 0.3 → 0.0
- Simplified descriptions
- Cleaner code structure
- Better documentation

### Configuration Comparison

```python
# OLD (BROKEN)
def __init__(self, num_nodes, hidden_dim, num_layers=2, dropout=0.5, ...)
    #                                                     ^^^ KILLED IT

# NEW (WORKS)
def __init__(self, num_nodes, hidden_dim=128, num_layers=2, dropout=0.0, ...)
    #                                                         ^^^ FIXED
```

## Model Descriptions

### GCN (Graph Convolutional Network)
- **Performance:** 13-24% Hits@20
- **Architecture:** 2 GCN layers, 128 hidden dim
- **Features:** Learnable embeddings only
- **File:** `gcn.py`

### GraphSAGE (BEST BASELINE)
- **Performance:** 17% Hits@20
- **Architecture:** 2 SAGE layers, 128 hidden dim
- **Features:** Mean aggregation, learnable embeddings
- **File:** `sage.py`
- **Note:** Best performing baseline - use this for improvements

### GAT (Graph Attention Network)
- **Performance:** 11% Hits@20
- **Architecture:** 2 GAT layers, 2 attention heads
- **Features:** Multi-head attention, learnable embeddings
- **File:** `gat.py`

### GraphTransformer
- **Performance:** 8% Hits@20
- **Architecture:** 2 Transformer layers, 2 attention heads
- **Features:** Self-attention mechanism, learnable embeddings
- **File:** `transformer.py`

## Usage

All models follow the same interface:

```python
from src.models.baselines import GCN, GraphSAGE, GAT, GraphTransformer

# Create model (default: ultra-minimal config)
model = GraphSAGE(num_nodes=4267, hidden_dim=128)

# Encode nodes
node_embeddings = model.encode(edge_index)

# Decode edge scores
edge_scores = model.decode(node_embeddings, edge_pairs)
```

### Adding Light Regularization

If you want to experiment with dropout (not recommended initially):

```python
# Add very light dropout
model = GraphSAGE(
    num_nodes=4267,
    hidden_dim=128,
    dropout=0.1,  # MUCH lower than old 0.5
    decoder_dropout=0.0
)
```

**Recommended dropout values for ogbl-ddi:**
- Start: 0.0 (ultra-minimal)
- Light: 0.1
- Never: >0.2 (too high for dense graphs)

## Old Broken Models

The original broken models have been backed up to:
```
src/models/baselines/old_broken/
```

**DO NOT use these models.** They are kept for reference only.

## Next Steps

### Improving GraphSAGE (Current Best)

Start with GraphSAGE (17% val) and add improvements incrementally:

**Phase 1: Stability (Target: 25-30%)**
```python
1. Add BatchNorm after each layer
2. Train for 200 epochs (may still be improving)
3. Try 3 layers instead of 2
4. Add dropout=0.1 (optional, test if helps)
```

**Phase 2: Better Decoder (Target: 35-45%)**
```python
5. Replace dot product with MLP decoder
6. Increase hidden_dim to 256
```

**Phase 3: Advanced Techniques (Target: 50-70%)**
```python
7. Add distance encoding
8. Try SEAL approach
9. Careful hard negative sampling (late, low ratio)
```

## Why This Works

**ogbl-ddi Dataset Characteristics:**
- Very dense graph: 14.67% edge density
- High average degree: ~500 neighbors per node
- 1M+ edges, only 4.2k nodes

**Why dropout=0.5 failed:**
- GNNs aggregate information from neighbors
- With 500 neighbors and dropout=0.5, you lose half the aggregated information
- Model can't learn meaningful neighbor patterns
- Over-regularization prevents learning on dense graphs

**Why dropout=0.0 works:**
- Model can see full neighborhood information
- Learns proper aggregation patterns
- Dense graph provides natural regularization
- Simple is better for this task

## References

- Performance comparison: `baseline_comparison.md`
- Ultra-minimal script: `minimal_baselines.py`
- Original broken results: `logs/baselines_20251208_001009/`
- Ultra-minimal results: `logs/minimal_baselines_20251208_053057/`

---

**Bottom Line:** Use these ultra-minimal baselines as your starting point. The old ones with dropout=0.5 are fundamentally broken for ogbl-ddi.
