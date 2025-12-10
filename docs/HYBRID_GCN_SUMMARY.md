# Hybrid GCN Models - Quick Reference

## The Problem

ChemBERTa-GCN performed **62% worse** than baseline GCN:
- **Baseline GCN** (structure-only): 0.1118 val Hits@20 ✓
- **ChemBERTa-GCN** (with features): 0.0424 val Hits@20 ✗

**Why?** Forcing chemistry into the encoder overrode useful topological signals.

## The Solution

**Hybrid GCN** = Structure-only encoder + Chemistry-aware decoder

```
┌─────────────────────────────────────────┐
│  ENCODER (Structure-Only)               │
│  • Learnable node embeddings            │
│  • 2-layer GCN                          │
│  • Learn topology (like baseline)       │
│  → z [N, 128]                           │
├─────────────────────────────────────────┤
│  DECODER (Chemistry-Aware)              │
│  • Path 1: structure only               │
│  • Path 2: chemistry only               │
│  • Path 3: combined                     │
│  • Learnable weights                    │
│  • Fallback to structure if missing     │
└─────────────────────────────────────────┘
```

## Quick Start

```bash
# Train all 3 GCN variants
pixi run python train_hybrid.py
```

Trains:
1. **Hybrid-GCN-ChemAware** - Full 3-path decoder with learnable weights
2. **Hybrid-GCN-Simple** - Additive: `structure + α * chemistry`
3. **Hybrid-GCN-ChemAware-Gated** - Dynamic gates per edge

Results → `logs/hybrid_YYYYMMDD_HHMMSS/hybrid.log`

## Expected Outcome

✓ **Match or beat baseline**: Structure-only encoder preserves strong baseline (0.1118)  
✓ **Chemistry boost**: When both nodes have valid SMILES (54% of pairs)  
✓ **Graceful fallback**: Falls back to structure for missing SMILES (46% of nodes)

## Variants

### 1. Chemistry-Aware (Recommended)
Three scoring paths with learnable combination:
```python
structural_score = MLP(z_i ⊙ z_j)
chemical_score = MLP(chem_i ⊙ chem_j)  
combined_score = MLP([z_i⊙z_j, chem_i⊙chem_j])

final = softmax(w) @ [structural, chemical, combined]
```

### 2. Simple
Additive combination:
```python
final = structural_score + α * chemical_score
# α is learnable (starts at 0.1)
```

### 3. Chemistry-Aware-Gated
Dynamic gates decide per-edge chemistry importance:
```python
gate = sigmoid(MLP([z_i, z_j, chem_i, chem_j, mask_i, mask_j]))
final = gate * [structural, chemical, combined]
```

## Files

```
src/models/hybrid/
├── gcn.py              # HybridGCN
└── decoder.py          # ChemistryAwareDecoder, SimpleChemistryDecoder

src/training/
└── hybrid_trainer.py   # train_hybrid_model()

train_hybrid.py         # Training script (3 GCN variants)
```

## Key Design Choices

| Choice | Rationale |
|--------|-----------|
| **Structure-only encoder** | Preserves proven baseline performance (0.1118) |
| **Chemistry in decoder** | Adds signal without overriding topology |
| **Fallback to structure** | Handles 46% missing SMILES gracefully |
| **Learnable weights** | Model learns when chemistry helps |
| **GCN only** | Simpler, cleaner experiments |

## Comparison Table

| Model | Encoder | Decoder | Val Hits@20 |
|-------|---------|---------|-------------|
| Baseline GCN | Structure-only | Simple | **0.1118** |
| ChemBERTa-GCN | Chem features | Simple | 0.0424 |
| **Hybrid-GCN-ChemAware** | Structure-only | Chem-aware | **?** |
| **Hybrid-GCN-Simple** | Structure-only | Additive | **?** |
| **Hybrid-GCN-Gated** | Structure-only | Gated | **?** |

## Next Steps

1. Train: `pixi run python train_hybrid.py`
2. Compare with baseline (0.1118)
3. Analyze decoder weights to see when model uses chemistry
4. If successful, extend to GraphSAGE/Transformer/GAT

