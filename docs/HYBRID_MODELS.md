# Hybrid Models: Smart Chemistry Integration

## Problem Summary

ChemBERTa embeddings performed **catastrophically worse** than structure-only baselines:

```
GCN:              0.1118 (baseline) → 0.0424 (ChemBERTa)  [62% worse]
GraphSAGE:        0.1293 (baseline) → 0.0005 (ChemBERTa)  [99.6% worse]
GraphTransformer: 0.1283 (baseline) → 0.0208 (ChemBERTa)  [84% worse]
GAT:              0.1190 (baseline) → 0.0039 (ChemBERTa)  [97% worse]
```

**Root Cause**: 
- Forcing chemistry into the node encoder overrides useful topological signals
- 46% of nodes have no SMILES → all share one fallback embedding
- Chemical similarity ≠ interaction similarity for DDIs

## Solution: Chemistry-Aware Decoder

**Key Insight**: Keep the encoder structure-only (preserving strong baseline performance), but incorporate chemistry intelligently at the decoder level.

### Architecture

```python
# ENCODER: Structure-only (like baselines)
z = GNN(edge_index)  # [N, hidden_dim] - learns topology

# DECODER: Chemistry-aware
structural_score = MLP(z_i ⊙ z_j)              # Pure topology
chemical_score = MLP(chem_i ⊙ chem_j)          # Pure chemistry
combined_score = MLP([z_i⊙z_j, chem_i⊙chem_j]) # Joint reasoning

# Learnable weighted combination
final_score = w1*structural + w2*chemical + w3*combined

# Smart fallback: only use chemistry when BOTH nodes have valid SMILES
if mask_i * mask_j == 0:
    final_score = structural_score  # Fall back to structure
```

## Usage

### Training

```bash
# Run all hybrid models
pixi run python train_hybrid.py
```

This trains 6 models:
- **Full decoder**: Hybrid-{GCN, GraphSAGE, GraphTransformer, GAT}-ChemAware
- **Simple decoder**: Hybrid-{GCN, GraphSAGE}-Simple

Results logged to `logs/hybrid_YYYYMMDD_HHMMSS/hybrid.log`

### Manual Usage

```python
from src.models.hybrid import HybridGCN
from src.training.hybrid_trainer import train_hybrid_model

# Create model
model = HybridGCN(
    num_nodes=4267,
    hidden_dim=128,
    num_layers=2,
    dropout=0.0,
    chemical_dim=768,  # ChemBERTa: 768, Morgan: 2048
    decoder_type='chemistry_aware',  # or 'simple'
    decoder_dropout=0.3,
    use_gating=False,  # optional: learned gates
)

# Load data (must have data.x and data.smiles_mask)
data, split_edge, evaluator = load_dataset(
    dataset_name="ogbl-ddi",
    use_chemberta=True,
    return_smiles_mask=True,
)

# Train
result = train_hybrid_model(
    name="Hybrid-GCN",
    model=model,
    data=data,
    train_pos=split_edge['train']['edge'],
    valid_pos=split_edge['valid']['edge'],
    valid_neg=split_edge['valid']['edge_neg'],
    test_pos=split_edge['test']['edge'],
    test_neg=split_edge['test']['edge_neg'],
    evaluate_fn=None,
    device=device,
    epochs=200,
    lr=0.01,
    weight_decay=1e-4,
)

print(f"Best Val: {result.best_val_hits:.4f}")
print(f"Test: {result.best_test_hits:.4f}")
```

## Why This Should Work

1. **Preserves strong baseline**: Structure-only encoder matches baselines
2. **Adds information**: Chemistry provides complementary signal
3. **Handles missing data**: Graceful fallback to structure for missing SMILES
4. **Learnable balance**: Model decides when chemistry helps
5. **No forced features**: Chemistry doesn't override topological patterns

## Model Variants

### Chemistry-Aware Decoder (Recommended)
- Three scoring paths with learnable weights
- Automatically learns optimal combination
- Falls back to structure for missing SMILES

**Files**: All models in `src/models/hybrid/` with `decoder_type='chemistry_aware'`

### Simple Decoder
- Additive: `score = structural + α * chemical`
- Single learnable parameter α
- Simpler but still intelligent

**Files**: Same models with `decoder_type='simple'`

### With Gating (Experimental)
- Dynamic gates per edge
- `gate = f(z_i, z_j, chem_i, chem_j, mask_i, mask_j)`
- Can learn edge-specific chemistry importance

**Enable**: Set `use_gating=True`

## Implementation Details

### New Files Created

```
src/models/hybrid/
├── __init__.py           # Module exports
├── README.md            # Detailed documentation
├── decoder.py           # ChemistryAwareDecoder, SimpleChemistryDecoder
├── gcn.py              # HybridGCN
├── sage.py             # HybridGraphSAGE
├── transformer.py      # HybridGraphTransformer
└── gat.py              # HybridGAT

src/training/
└── hybrid_trainer.py    # train_hybrid_model(), evaluate_hybrid()

train_hybrid.py          # Main training script
HYBRID_MODELS.md        # This file
```

### Key Differences from Previous Approaches

| Component | Baseline | ChemBERTa-Fallback | Hybrid (New) |
|-----------|----------|-------------------|--------------|
| **Encoder** | Structure-only | Chemistry features | Structure-only |
| **Decoder** | Simple dot product | Simple dot product | **Chemistry-aware** |
| **Missing SMILES** | N/A | Shared fallback | Structural fallback |
| **Performance** | Strong (0.11-0.13) | Weak (0.00-0.04) | **TBD** |

## Expected Results

Based on the design:

1. **Should not degrade**: Structure-only encoder preserves baseline performance
2. **Potential improvement**: Chemistry provides boost when both nodes have valid SMILES
3. **Robustness**: Graceful handling of 46% missing SMILES
4. **Better generalization**: Less overfitting to spurious chemical patterns

## Next Steps

1. **Run experiments**: `pixi run python train_hybrid.py`
2. **Compare with baselines**: Check if we match/beat structure-only baselines
3. **Analyze decoder weights**: See when model prefers topology vs chemistry
4. **Experiment with gating**: Try dynamic gates for per-edge chemistry importance

## Questions to Answer

- Do hybrid models match baseline performance?
- Does chemistry provide a meaningful boost?
- How do decoder weights evolve during training?
- Are there patterns in when chemistry helps vs hurts?

