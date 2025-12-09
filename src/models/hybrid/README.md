# Hybrid Models: Structure-Only Encoder + Chemistry-Aware Decoder

## The Problem

Previous approaches (ChemBERTa-based models) performed **worse than structure-only baselines**:

| Model | Baseline (structure-only) | ChemBERTa (chem features) | Performance |
|-------|---------------------------|---------------------------|-------------|
| GCN | 0.1118 | 0.0424 | **62% worse** |
| GraphSAGE | 0.1293 | 0.0005 | **99.6% worse** |
| GraphTransformer | 0.1283 | 0.0208 | **84% worse** |
| GAT | 0.1190 | 0.0039 | **97% worse** |

**Why?** Chemical features were forced into the node encoder, overriding useful topological signals. With 46% of nodes missing SMILES (using shared fallback embeddings), the models struggled to learn meaningful patterns.

## The Solution

**Hybrid Architecture**: Keep encoder structure-only, incorporate chemistry at decoder level.

```
┌─────────────────────────────────────────────────────────┐
│                    HYBRID MODEL                          │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Encoder (Structure-Only)                                │
│  ├── Learnable node embeddings                           │
│  ├── GNN layers (GCN/SAGE/Transformer/GAT)              │
│  └── Output: z = [N, hidden_dim]                        │
│                                                           │
│  Decoder (Chemistry-Aware)                               │
│  ├── Path 1: structural_score = MLP(z_i ⊙ z_j)         │
│  ├── Path 2: chemical_score = MLP(chem_i ⊙ chem_j)     │
│  ├── Path 3: combined_score = MLP([z_i⊙z_j, chem_i⊙chem_j]) │
│  └── Final: weighted combination of all paths           │
│                                                           │
│  Intelligence:                                           │
│  ✓ Uses chemistry only when both nodes have valid SMILES│
│  ✓ Learns when to trust topology vs chemistry           │
│  ✓ Falls back to structure-only for missing SMILES      │
└─────────────────────────────────────────────────────────┘
```

## Key Innovations

### 1. **Separation of Concerns**
- **Encoder**: Learns topological patterns (like strong baselines)
- **Decoder**: Incorporates chemistry when useful

### 2. **Smart Chemistry Usage**
```python
# Only use chemistry when BOTH nodes have valid SMILES
both_valid = smiles_mask[src] * smiles_mask[dst]
scores = torch.where(both_valid > 0.5, 
                     combined_score,      # Use chemistry
                     structural_score)    # Fall back to structure
```

### 3. **Multi-Path Scoring**
Three complementary strategies learned jointly:
- **Structural**: Pure topology `MLP(z_i ⊙ z_j)`
- **Chemical**: Pure chemistry `MLP(chem_i ⊙ chem_j)`
- **Combined**: Joint reasoning `MLP([z_i⊙z_j, chem_i⊙chem_j])`

Learnable weights determine optimal combination.

## Model Variants

### Chemistry-Aware Decoder (Full)
- Three scoring paths with learnable weights
- Automatically learns when to use each path
- Falls back to structural score for missing SMILES

### Simple Decoder
- Additive: `score = structural_score + α * chemical_score`
- Simpler, but still intelligent
- Single learnable parameter `α` (starts at 0.1)

### With Gating (Optional)
- Dynamic gates based on node features
- `gate = sigmoid(MLP([z_i, z_j, chem_i, chem_j, mask_i, mask_j]))`
- Can learn per-edge chemistry importance

## Usage

```python
from src.models.hybrid import HybridGCN
from src.training.hybrid_trainer import train_hybrid_model

# Create model (structure-only encoder)
model = HybridGCN(
    num_nodes=4267,
    hidden_dim=128,
    num_layers=2,
    dropout=0.0,
    chemical_dim=768,  # ChemBERTa dimension
    decoder_type='chemistry_aware',  # or 'simple'
    decoder_dropout=0.3,
    use_gating=False,
)

# Train with hybrid trainer
result = train_hybrid_model(
    name="Hybrid-GCN",
    model=model,
    data=data,  # Must have data.x (chemistry) and data.smiles_mask
    train_pos=train_edges,
    valid_pos=valid_edges,
    valid_neg=valid_neg_edges,
    test_pos=test_edges,
    test_neg=test_neg_edges,
    evaluate_fn=None,
    device=device,
    epochs=200,
    lr=0.01,
    weight_decay=1e-4,
)
```

## Training Script

Run all hybrid models:

```bash
pixi run python train_hybrid.py
```

This trains:
- Hybrid-GCN-ChemAware
- Hybrid-GraphSAGE-ChemAware
- Hybrid-GraphTransformer-ChemAware
- Hybrid-GAT-ChemAware
- Hybrid-GCN-Simple
- Hybrid-GraphSAGE-Simple

Results logged to `logs/hybrid_YYYYMMDD_HHMMSS/hybrid.log`

## Expected Improvements

Based on similar approaches in literature:

1. **Should match or beat baselines**: Structure-only encoder preserves strong topological learning
2. **Chemistry as boost**: When both nodes have valid SMILES, chemistry provides additional signal
3. **Graceful degradation**: Falls back to structure for missing SMILES
4. **Better generalization**: Less prone to overfitting on spurious chemical patterns

## Files

```
src/models/hybrid/
├── __init__.py           # Module exports
├── README.md            # This file
├── decoder.py           # Chemistry-aware decoders
├── gcn.py              # Hybrid GCN
├── sage.py             # Hybrid GraphSAGE
├── transformer.py      # Hybrid GraphTransformer
└── gat.py              # Hybrid GAT

src/training/
└── hybrid_trainer.py    # Training loop for hybrid models

train_hybrid.py          # Main training script
```

## Design Principles

1. **Keep what works**: Structure-only encoders match strong baselines
2. **Add intelligence**: Chemistry used only when helpful
3. **Handle missing data**: Graceful fallback for nodes without SMILES
4. **Learnable weighting**: Model decides optimal topology/chemistry balance
5. **No forced features**: Chemistry doesn't override topological signals

## Comparison with Previous Approaches

| Approach | Encoder | Decoder | Missing SMILES | Performance |
|----------|---------|---------|----------------|-------------|
| **Baseline** | Structure-only | Simple | N/A | Strong (0.11-0.13) |
| **ChemBERTa-Fallback** | Chem features | Simple | Shared embedding | Weak (0.00-0.04) |
| **Hybrid (Ours)** | Structure-only | Chem-aware | Structural fallback | **TBD** |

The key insight: **Don't poison the encoder with unreliable chemistry—keep it pure and add chemistry intelligently at the decoder.**

