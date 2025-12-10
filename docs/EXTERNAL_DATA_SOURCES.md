# External Data Sources for OGBL-DDI

This document describes all external knowledge sources integrated into the DDI prediction pipeline to enhance model performance beyond the graph structure alone.

## Overview

| Feature Type | Dimensions | Source | Description |
|--------------|------------|--------|-------------|
| Morgan Fingerprints | 2048 | RDKit | Binary molecular substructure features |
| PubChem Properties | 9 | PubChem API | Continuous physicochemical properties |
| ChemBERTa Embeddings | 768 | HuggingFace | Pre-trained molecular representations |
| Drug-Target Features | ~229 | TDC (KIBA) | Binary target interaction vectors |
| **Total** | **3054** | | Combined feature dimension |

---

## 1. Morgan Fingerprints

### What They Are
Morgan fingerprints (also known as Extended Connectivity Fingerprints / ECFP) are circular topological fingerprints that encode molecular substructures. Each bit represents the presence or absence of a particular substructure pattern within a specified radius of atoms.

### Technical Details
- **Dimensions**: 2048 bits (binary)
- **Radius**: 2 (captures substructures within 2 bond hops)
- **Library**: RDKit (`rdkit-pypi`)
- **Input**: SMILES strings

### Why They Help DDI Prediction
- Capture local chemical structure patterns
- Similar drugs often share substructures
- Fast to compute and interpretable
- Well-established in cheminformatics

### How to Fetch
```bash
# Step 1: Fetch SMILES from PubChem
pixi run fetch-smiles

# Morgan features are built automatically from SMILES when first used
```

### Output Files
- `data/ogbl_ddi_smiles.csv` - SMILES strings per drug
- `data/morgan_features.pt` - Cached Morgan fingerprint tensor

### Code Location
- Fetching: [src/data/fetch_smiles.py](../src/data/fetch_smiles.py)
- Loading: [src/data/external_features.py](../src/data/external_features.py) → `load_morgan_features()`
- Building: [src/data/data_loader.py](../src/data/data_loader.py) → `build_smiles_feature_matrix()`

---

## 2. PubChem Properties

### What They Are
Molecular properties computed by PubChem that describe physicochemical characteristics relevant to drug behavior and interactions.

### Properties Included (9 dimensions)
| Property | Description | Relevance |
|----------|-------------|-----------|
| MolecularWeight | Total molecular mass | Drug absorption, distribution |
| XLogP | Octanol-water partition coefficient | Lipophilicity / membrane permeability |
| TPSA | Topological Polar Surface Area | Oral bioavailability |
| HBondDonorCount | Hydrogen bond donors | Binding interactions |
| HBondAcceptorCount | Hydrogen bond acceptors | Binding interactions |
| RotatableBondCount | Rotatable bonds | Molecular flexibility |
| HeavyAtomCount | Non-hydrogen atoms | Molecular size |
| Complexity | Structural complexity score | Overall molecular complexity |
| Charge | Formal charge | Electrostatic interactions |

### Technical Details
- **Dimensions**: 9 continuous values
- **Normalization**: StandardScaler (z-score)
- **Missing Values**: Filled with zero (default) or column mean
- **API**: PubChem via `pubchempy`

### Why They Help DDI Prediction
- Drugs with similar properties may have similar interaction profiles
- Lipophilicity affects CYP450 enzyme interactions
- Polar surface area influences transporter-mediated DDIs
- Complementary to structural fingerprints

### How to Fetch
```bash
# Requires SMILES data first
pixi run fetch-pubchem
```

### Output Files
- `data/ogbl_ddi_properties.csv` - Properties per drug

### Code Location
- Fetching: [src/data/fetch_pubchem_properties.py](../src/data/fetch_pubchem_properties.py)
- Loading: [src/data/external_features.py](../src/data/external_features.py) → `load_pubchem_features()`

---

## 3. ChemBERTa Embeddings

### What They Are
Pre-trained transformer embeddings from ChemBERTa, a BERT-style model trained on ~77 million SMILES strings from the ZINC database. These embeddings capture rich chemical semantics learned from large-scale molecular data.

### Technical Details
- **Model**: `seyonec/ChemBERTa-zinc-base-v1` (HuggingFace)
- **Dimensions**: 768 (BERT hidden size)
- **Pooling**: Mean pooling over token embeddings (excluding padding)
- **Max Length**: 128 tokens
- **Library**: `transformers`

### Why They Help DDI Prediction
- Learned representations capture complex chemical patterns
- Transfer learning from large molecular corpus
- Captures relationships not obvious from fingerprints
- State-of-the-art for many molecular property prediction tasks

### How to Fetch
```bash
# Requires SMILES data and GPU recommended
pixi run extract-chemberta
```

### Output Files
- `data/chemberta_embeddings.pt` - Embedding tensor [4267, 768]

### Code Location
- Extraction: [src/data/extract_chemberta_embeddings.py](../src/data/extract_chemberta_embeddings.py)
- Loading: [src/data/external_features.py](../src/data/external_features.py) → `load_chemberta_features()`

### Reference
- Paper: [ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction](https://arxiv.org/abs/2010.09885)
- HuggingFace: https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1

---

## 4. Drug-Target Interaction Features

### What They Are
Binary vectors indicating which protein targets each drug interacts with, derived from the KIBA dataset in the Therapeutics Data Commons (TDC).

### Technical Details
- **Dimensions**: Variable (~229 in practice, capped at 500)
- **Format**: Binary (1 = drug interacts with target, 0 = no known interaction)
- **Source**: TDC KIBA dataset
- **Matching**: By SMILES string (exact match)
- **Library**: `PyTDC`

### Why They Help DDI Prediction
**Critical for OGBL-DDI**: The OGBL-DDI dataset uses a **protein-target based split** where test edges involve drugs targeting previously unseen proteins. Drug-target knowledge directly addresses this out-of-distribution challenge:

- Drugs targeting similar proteins may have similar DDI profiles
- Target-based features provide biological context
- Helps generalize to test set by leveraging target information

### How to Fetch
```bash
# Requires SMILES data
pixi run fetch-drug-targets
```

### Output Files
- `data/drug_target_features.pt` - Binary target feature tensor [4267, num_targets]
- `data/drug_target_edges.pt` - Edge list for heterogeneous graphs (optional)

### Code Location
- Fetching: [src/data/fetch_drug_targets.py](../src/data/fetch_drug_targets.py)
- Loading: [src/data/external_features.py](../src/data/external_features.py) → `load_drug_target_features()`

### Data Statistics
The KIBA dataset contains:
- ~229K drug-target interactions
- ~2,116 unique drugs
- ~229 unique protein targets (after frequency filtering)

---

## Data Pipeline

### Fetch All Data at Once
```bash
pixi run fetch-data
```

This sequentially runs:
1. `fetch-smiles` - Get SMILES from PubChem
2. `fetch-pubchem` - Get molecular properties
3. `extract-chemberta` - Generate embeddings (GPU recommended)
4. `fetch-drug-targets` - Get target interactions

### Data Dependencies
```
nodeidx2drugid.csv.gz ─┬─> fetch-smiles ─> ogbl_ddi_smiles.csv
ddi_description.csv.gz ─┘                          │
                                                   ├─> fetch-pubchem ─> ogbl_ddi_properties.csv
                                                   ├─> extract-chemberta ─> chemberta_embeddings.pt
                                                   └─> fetch-drug-targets ─> drug_target_features.pt
```

### Coverage Statistics
From the log output:
- **4,267** total drugs in OGBL-DDI
- **~3,800** have SMILES from PubChem (~89%)
- **~3,800** have PubChem properties
- **~3,800** have ChemBERTa embeddings
- **~500-1000** matched to TDC drug-target data (~15-25%)

---

## Usage in Training

### Command Line Flags
```bash
# Train with all external features
python train_ddi_reference_external.py --all

# Train with specific features
python train_ddi_reference_external.py --morgan --chemberta

# Train baseline (no external features)
python train_ddi_reference_external.py
```

### Feature Flags
| Flag | Feature | Dimension |
|------|---------|-----------|
| `--morgan` | Morgan fingerprints | 2048 |
| `--pubchem` | PubChem properties | 9 |
| `--chemberta` | ChemBERTa embeddings | 768 |
| `--drug-targets` | Drug-target features | ~229 |
| `--all` | All of the above | 3054 |

### Feature Fusion Strategies
```bash
--fusion concat  # Concatenate all features (default)
--fusion add     # Average encoded features
```

---

## Code Architecture

### Feature Configuration
```python
from src.data.external_features import FeatureConfig, load_external_features

# Configure which features to use
config = FeatureConfig(
    use_morgan=True,
    use_pubchem=True,
    use_chemberta=True,
    use_drug_targets=True,
    normalize=True,
    fill_missing="zero",
)

# Load features
features = load_external_features(config, num_nodes=4267)

# Access individual features
features.morgan       # [4267, 2048]
features.pubchem      # [4267, 9]
features.chemberta    # [4267, 768]
features.drug_targets # [4267, 229]
features.combined     # [4267, 3054]
```

### Model Integration
The `GCNExternal` and `SAGEExternal` models in [src/models/ogb_ddi_gnn/gnn_external.py](../src/models/ogb_ddi_gnn/gnn_external.py):

1. Encode each external feature type to hidden dimension
2. Combine with learnable node embeddings via concat or add
3. Apply GNN layers on the fused representation

---

## Results Summary

### Performance Comparison (2000 epochs)

| Configuration | Val Hits@20 | Test Hits@20 |
|--------------|-------------|--------------|
| GCN Baseline (no features) | 56.09% | 39.84% |
| **GCN + All External** | **70.31%** | **73.28%** |

**Improvement**: +14.22% Val, +33.44% Test

The external features provide substantial gains, especially on the test set which has OOD characteristics due to the protein-target split.

---

## File Locations Summary

| File | Description |
|------|-------------|
| `data/ogbl_ddi_smiles.csv` | SMILES strings (ogb_id, drugbank_id, drug_name, smiles) |
| `data/ogbl_ddi_properties.csv` | PubChem properties (9 columns) |
| `data/morgan_features.pt` | Morgan fingerprints tensor [4267, 2048] |
| `data/chemberta_embeddings.pt` | ChemBERTa embeddings [4267, 768] |
| `data/drug_target_features.pt` | Drug-target binary features [4267, ~229] |
| `data/drug_target_edges.pt` | Drug-target edges for hetero graphs |

---

## Dependencies

```toml
# pixi.toml
[pypi-dependencies]
rdkit-pypi = ">=2022.9.5, <2023"      # Morgan fingerprints
transformers = ">=4.35.0, <5"          # ChemBERTa
PyTDC = ">=1.0.0, <2"                  # Drug-target data

[dependencies]
pubchempy = ">=1.0.5,<2"               # SMILES & properties
```
