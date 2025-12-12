# Drug-Drug Interaction Prediction with Graph Neural Networks

**CS224W Project: Link Prediction on ogbl-ddi**

This project explores link prediction on the Open Graph Benchmark drug-drug interaction (ogbl-ddi) dataset using various Graph Neural Network architectures and molecular features. We progressively build from simple structure-only baselines to feature-rich models that leverage drug chemistry and biology, achieving a **73.28% test Hits@20** with our best model.

---

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Project Progression](#project-progression)
   - [Phase 1: Running the Baselines](#phase-1-running-the-baselines)
   - [Phase 2: Structure-Only GCN Optimization](#phase-2-structure-only-gcn-optimization)
   - [Phase 3: Advanced GCN + Ablations](#phase-3-advanced-gcn--ablations)
   - [Phase 4: Incorporating Drug Features](#phase-4-incorporating-drug-features)
   - [Phase 5: Hybrid Models (Structure + Chemistry Decoder)](#phase-5-hybrid-models-structure--chemistry-decoder)
   - [Phase 6: Full Feature Integration (Best Model)](#phase-6-full-feature-integration-best-model)
3. [Final Results Summary](#final-results-summary)
4. [Quick Start](#quick-start)
5. [Repository Structure](#repository-structure)
6. [Key Insights](#key-insights)

---

## Dataset Overview

**ogbl-ddi** is a drug-drug interaction network from the Open Graph Benchmark:
- **Nodes**: 4,267 drugs
- **Edges**: ~2.1M drug-drug interactions
- **Task**: Link prediction (predict which drug pairs interact)
- **Metric**: Hits@K (K=10, 20, 30) - primary metric is Hits@20
- **Splits**:
  - Train: 1,067,911 positive edges
  - Validation: 133,489 positive, 101,882 negative
  - Test: 133,489 positive, 95,599 negative

---

## Project Progression

### Phase 1: Running the Baselines

**Objective**: Establish baseline performance using structure-only GNN architectures.

**Approach**: We trained four standard GNN architectures (GCN, GraphSAGE, GAT, GraphTransformer) using only graph structure with learnable node embeddings. No external drug features were used—models learned representations purely from network topology.

**Models Tested**:
- GCN (Graph Convolutional Network)
- GraphSAGE (inductive neighborhood aggregation)
- GAT (Graph Attention Network)
- GraphTransformer (attention-based message passing)

**Configuration**:
- Hidden dimension: 128
- Number of layers: 2
- Dropout: 0.0 (ultra-minimal)
- Learning rate: 0.01 (0.005 for attention models)
- Batch size: 50,000
- Training epochs: 200

**Results**:

| Model | Validation Hits@20 | Test Hits@20 | Val-Test Gap |
|-------|-------------------|--------------|--------------|
| **GCN** | 13.59% | **11.02%** | 2.58% (19.0%) |
| GraphSAGE | 9.61% | 4.46% | 5.15% (53.6%) |
| GAT | 9.53% | 4.88% | 4.65% (48.8%) |
| GraphTransformer | 11.66% | 3.73% | 7.93% (68.0%) |

**Key Takeaway**: GCN emerged as the best structure-only baseline with the lowest val-test gap, establishing it as our foundation architecture. The large gaps in attention-based models suggested overfitting without proper regularization or features.

**Log Reference**: `logs/baselines_20251209_185813/baselines.log`

---

### Phase 2: Structure-Only GCN Optimization

**Objective**: Maximize structure-only GCN performance through hyperparameter tuning and extended training.

**Why This Step**: Before adding external features, we wanted to understand the ceiling of structure-only learning. This establishes a strong baseline and helps isolate the contribution of molecular features later.

**Approach**: We systematically varied:
- Hidden dimensions (128 → 256)
- Dropout rates (0.0 → 0.5)
- Training duration (100 → 2000 epochs)
- Batch sizes (32k → 65k)
- Learning rates (0.001 → 0.01)

**Key Experiments**:

1. **Short-to-Medium Budget** (20-200 epochs):
   - 100 epochs: 22.32% test Hits@20
   - 200 epochs: **39.84% test Hits@20**

2. **Long-Horizon Training** (300-600+ epochs):
   - Best observed: **55.71% test Hits@20** (dropout=0.5, ~330 epochs)
   - Extended run: 42.40% test Hits@20 (dropout=0.25, 600+ epochs)

**Best Structure-Only Configuration**:
- Architecture: 2-layer GCN, 256 hidden channels
- Dropout: 0.5
- Learning rate: 0.005
- Batch size: 65,536
- Training: 2000 epochs (best at epoch 430)
- **Performance**: 64.78% validation, 61.69% test Hits@20

**Key Takeaway**: Proper regularization (dropout) and extended training significantly boost structure-only performance. The 5x improvement over the initial baseline (11% → 55%+) shows that careful tuning matters. However, we hit a ceiling around 60% test performance, motivating the need for external features.

**Log References**:
- `logs/ddi_gcn_20251210_030519/ddi_gcn.log` (200 epochs)
- `logs/ddi_long_sweeps_20251210_231641/sweep.log` (long horizon)

---

### Phase 3: Advanced GCN + Ablations

**Objective**: Understand the impact of architectural choices (depth, dropout, learning rate) on GCN performance.

**Why This Step**: Before investing in feature engineering, we needed to validate our architectural decisions. These ablations help us understand what contributes to model performance and guide future experiments.

**Approach**: Fast ablation study (200 epochs each) varying:
- Number of layers (2 vs 3)
- Hidden dimensions (192, 256, 512)
- Dropout rates (0.0, 0.1, 0.25, 0.5)
- Learning rates (0.001, 0.0015, 0.005)

**Results**:

| Experiment | Layers | Hidden | Dropout | LR | Val Hits@20 | Test Hits@20 |
|-----------|--------|--------|---------|-----|-------------|--------------|
| E1 (baseline) | 2 | 256 | 0.0 | 0.005 | 36.45% | 17.36% |
| **E2 (dropout)** | 2 | 256 | 0.25 | 0.005 | **50.69%** | **34.34%** |
| E3 (3-layer) | 3 | 192 | 0.0 | 0.001 | 31.82% | 18.13% |
| E4 (3-layer + drop) | 3 | 192 | 0.1 | 0.001 | 38.24% | 18.10% |
| E5 (3-layer, lr↑) | 3 | 192 | 0.0 | 0.0015 | 34.55% | 12.30% |

**Key Findings**:
1. **Dropout is crucial**: Adding dropout=0.25 nearly doubled test performance (17.36% → 34.34%)
2. **Shallow is better**: 2-layer models consistently outperformed 3-layer models
3. **Overfitting without regularization**: E1 shows 19% absolute val-test gap without dropout
4. **Learning rate sensitivity**: Optimal LR=0.005 for this architecture and batch size

**Key Takeaway**: For dense drug interaction graphs, a shallow 2-layer GCN with moderate dropout (0.25-0.5) works best. Adding depth doesn't help and may hurt generalization. These findings inform our final architecture choices.

**Log Reference**: `logs/ddi_gcn_ablations_20251211_005945/README.md`

---

### Phase 4: Incorporating Drug Features

**Objective**: Explore different molecular and biological feature representations to improve drug node embeddings.

**Why This Step**: Structure-only models peaked around 60% test performance. Drug-drug interactions depend heavily on molecular properties and biological mechanisms, so incorporating chemistry should provide significant gains.

**Feature Types Explored**:

#### 4.1 Morgan Fingerprints (2048-dim)
Chemical substructure fingerprints encoding molecular fragments.

**Results**:

| Model | Validation Hits@20 | Test Hits@20 | Val-Test Gap |
|-------|-------------------|--------------|--------------|
| Morgan-GCN | 6.10% | 5.36% | 0.73% (12.0%) |
| Morgan-GraphSAGE | 3.05% | 4.71% | -1.65% (-54.2%) |
| Morgan-GraphTransformer | 4.14% | 3.98% | 0.15% (3.7%) |
| Morgan-GAT | 4.32% | 1.12% | 3.20% (74.0%) |

**Observation**: Morgan features alone actually underperformed structure-only baselines, suggesting fixed fingerprints lack sufficient expressiveness for this task without graph context.

**Log**: `logs/morgan_baselines_20251209_191058/morgan_baselines.log`

#### 4.2 ChemBERTa Embeddings (768-dim)
Pre-trained transformer embeddings capturing semantic chemical information from SMILES strings.

**Challenge**: 46% of drugs lack SMILES data. We used learnable fallback embeddings for missing nodes to maintain gradient flow.

**Results**:

| Model | Validation Hits@20 | Test Hits@20 | Val-Test Gap |
|-------|-------------------|--------------|--------------|
| ChemBERTa-GCN | 11.37% | 6.94% | 4.43% (39.0%) |
| ChemBERTa-GraphSAGE | 11.29% | 9.07% | 2.22% (19.7%) |
| ChemBERTa-GraphTransformer | 9.41% | 4.99% | 4.42% (47.0%) |
| ChemBERTa-GAT | 6.19% | 2.60% | 3.59% (58.0%) |

**Observation**: ChemBERTa embeddings alone performed similarly to structure-only GCN but with higher val-test gaps. This suggested they capture useful chemistry but need graph context.

**Log**: `logs/chemberta_fallback_20251209_210608/chemberta_fallback.log`

#### 4.3 PubChem Properties (9-dim)
Hand-engineered physicochemical descriptors (molecular weight, logP, H-bond donors/acceptors, etc.)

**Observation**: Small feature dimension but captures important drug-likeness properties. Not tested in isolation but included in combined feature models.

#### 4.4 Drug-Target Interactions (229-dim)
Features derived from known drug-target binding profiles.

**Observation**: Biological context from protein targets provides complementary signal to pure chemistry. Included in best combined model.

**Key Takeaway**: Individual feature types show limited performance, suggesting the need for multi-modal feature fusion. Morgan and ChemBERTa alone are insufficient, but they may be valuable when combined with graph structure and other modalities.

---

### Phase 5: Hybrid Models (Structure + Chemistry Decoder)

**Objective**: Combine the strong structure-only GCN encoder with chemistry-aware decoding to leverage both graph topology and molecular features.

**Why This Step**: Phase 4 showed that chemistry features alone underperformed structure-only models. We hypothesized that a hybrid approach—encoding graph structure but decoding with chemistry—would capture both signals.

**Approach**: Three hybrid architectures tested:
1. **Hybrid-GCN-Simple**: Additive combination of structural and chemical scores with learnable weight
2. **Hybrid-GCN-ChemAware**: Three-path decoder (structural, chemical, combined) with learnable weights
3. **Hybrid-GCN-ChemAware-Gated**: Dynamic gate determining chemistry importance per edge

**Configuration**:
- Encoder: Structure-only GCN (preserves strong baseline)
- Decoder: Chemistry-aware (uses ChemBERTa + structural features)
- Features: ChemBERTa (768) + structural (6) = 774 dims

**Results**:

| Model | Validation Hits@20 | Test Hits@20 | Val-Test Gap |
|-------|-------------------|--------------|--------------|
| **Hybrid-GCN-Simple** | 23.49% | **19.33%** | 4.16% (17.7%) |
| Hybrid-GCN-ChemAware | 18.82% | 15.57% | 3.25% (17.3%) |
| Hybrid-GCN-ChemAware-Gated | 18.59% | 21.60% | -3.01% (-16.2%) |

**Key Findings**:
- Simple additive fusion worked best (~19% test Hits@20)
- Hybrid models outperformed chemistry-only models (~6-10%)
- Still underperformed optimized structure-only GCN (~55-60%)
- Chemistry signal is helpful but not sufficient with limited features

**Key Takeaway**: Hybrid decoding shows promise but doesn't close the gap to structure-only performance. This motivated our final approach: rich multi-modal feature concatenation fed directly into a strong GCN encoder.

**Log**: `logs/hybrid_20251209_231044/hybrid.log`

---

### Phase 6: Full Feature Integration (Best Model)

**Objective**: Achieve state-of-the-art performance by combining all available feature sources in a unified representation.

**Why This Step**: Previous phases showed:
- Structure-only GCN can reach ~60% with proper tuning
- Individual feature types underperform structure-only
- Hybrid decoding helps but isn't sufficient
- **Hypothesis**: Concatenating all features should provide the richest signal

**Approach**: Concatenate all feature modalities into a single 3054-dimensional node representation:
- Morgan fingerprints: 2048 dims (molecular substructure)
- ChemBERTa embeddings: 768 dims (semantic chemistry)
- PubChem properties: 9 dims (physicochemical)
- Drug-target features: 229 dims (biological context)

Feed these rich features into a 2-layer GCN encoder with MLP decoder.

**Architecture**:
- Encoder: 2-layer GCN, 256 hidden channels
- Decoder: Simple MLP (Hadamard product + feedforward)
- Input features: 3054 dims (all modalities concatenated)
- Dropout: 0.5
- Learning rate: 0.005
- Batch size: 65,536
- Training: 2000 epochs

**Results** (Best Model):

| Metric | Validation | Test | Best Epoch |
|--------|-----------|------|------------|
| **Hits@10** | - | **35.29%** | 1685 |
| **Hits@20** | **70.31%** | **73.28%** | 1685 |
| **Hits@30** | - | **84.64%** | 1685 |

**Performance Progression**:

| Approach | Test Hits@20 | Improvement |
|----------|--------------|-------------|
| Simple GCN baseline (Phase 1) | 11.02% | - |
| Optimized structure-only (Phase 2) | 55-61% | +5.0x |
| Hybrid models (Phase 5) | 19.33% | +1.75x from baseline |
| **All features (Phase 6)** | **73.28%** | **+6.65x from baseline** |

**Key Observations**:
1. **Excellent generalization**: Test (73.28%) > Validation (70.31%) by 3%
2. **Extended training helps**: Best results at epoch 1685 of 2000
3. **Feature fusion is key**: Combined features vastly outperform any single modality
4. **Multi-modal learning works**: Chemistry + biology + structure > sum of parts

**Why This Works**:
- **Rich signal**: 3054-dim features capture molecular structure, semantic chemistry, drug-likeness, and biological context
- **Graph context**: GCN message passing refines already-expressive features with network topology
- **Proper regularization**: Dropout prevents overfitting despite high feature dimensionality
- **Extended training**: Complex feature space benefits from longer optimization

**Key Takeaway**: The combination of comprehensive feature engineering and proper GNN architecture achieves a 6.6x improvement over structure-only baselines. Multi-modal fusion (chemistry + biology + structure) is essential for drug-drug interaction prediction.

**Log**: `logs/ddi_gcn_all_20251210_033923/ddi_gcn_all.log`

---

## Final Results Summary

### Performance Ranking (Test Hits@20)

| Rank | Model | Test Hits@20 | Phase |
|------|-------|--------------|-------|
| 🥇 1 | **GCN + All Features** | **73.28%** | Phase 6 |
| 2 | Plain GCN (structure, long, drop=0.5) | 55.71% | Phase 2 |
| 3 | Plain GCN (structure, long, drop=0.25) | 42.40% | Phase 2 |
| 4 | Plain GCN (structure, 200 epochs) | 39.84% | Phase 2 |
| 5 | Plain GCN (ablation, drop=0.25) | 34.34% | Phase 3 |
| 6 | Plain GCN (structure, 100 epochs) | 22.32% | Phase 2 |
| 7 | Hybrid-GCN-Simple | 19.33% | Phase 5 |
| 8 | GCN Baseline (structure-only) | 11.02% | Phase 1 |
| 9 | ChemBERTa-GraphSAGE | 9.07% | Phase 4 |
| 10 | ChemBERTa-GCN | 6.94% | Phase 4 |
| 11 | Morgan-GCN | 5.36% | Phase 4 |
| 12 | GraphSAGE Baseline | 4.46% | Phase 1 |

### Key Insights

1. **Feature Engineering is Critical**:
   - Best model (73.28%) uses all features vs. structure-only baseline (11.02%)
   - 6.6x improvement from comprehensive feature engineering
   - Multi-modal fusion outperforms individual feature types

2. **Architecture Choices Matter**:
   - Simple 2-layer GCN outperforms complex architectures (GAT, Transformer)
   - Shallow networks generalize better on dense drug graphs
   - Message passing amplifies good features but can't create them from nothing

3. **Regularization is Essential**:
   - Dropout (0.25-0.5) critical for generalization on DDI graphs
   - Structure-only models show dramatic improvement: 17% → 34% test Hits@20
   - Feature-rich models also benefit from regularization

4. **Training Considerations**:
   - Extended training helps: best model converges at epoch 1685
   - Structure-only models benefit from 300-600 epochs
   - Learning rate 0.005 stable for large batches (65k)

5. **Feature Fusion Strategy**:
   - Simple concatenation works better than complex hybrid decoders
   - All modalities contribute: chemistry + biology + structure
   - Chemistry-only features underperform but are valuable in combination

---

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd 224w-project

# Install dependencies (using pixi or conda)
pixi install
```

### Training Models

```bash
# Train best model (GCN + all features)
python train.py --model gcn --external-features all \
  --hidden-dim 256 --dropout 0.5 --lr 0.005 \
  --batch-size 65536 --epochs 2000

# Train structure-only baseline
python train.py --model gcn --external-features none

# Train hybrid model
python train.py --model hybrid-gcn --external-features chemberta,morgan
```

See `TRAINING.md` for detailed training instructions and all available options.

### Using the Consolidated Baseline Script

For a self-contained training script that doesn't require the `src` package:

```bash
# Train all baselines (GCN, GraphSAGE, GAT, GraphTransformer)
python train_baselines_consolidated.py
```

This script includes all model definitions, training logic, and evaluation code in one file.

---

## Repository Structure

```
224w-project/
├── src/
│   ├── data/               # Data loading and feature extraction
│   ├── models/             # Model architectures
│   │   ├── baselines/      # Structure-only GNNs
│   │   ├── chemberta_*/    # ChemBERTa-based models
│   │   ├── hybrid/         # Hybrid encoder-decoder models
│   │   ├── advanced/       # GDIN and other advanced architectures
│   │   └── morgan_*/       # Morgan fingerprint models
│   ├── training/           # Training loops and utilities
│   └── evals/              # Evaluation functions
├── logs/                   # Training logs and results
├── train.py                # Unified training script
├── train_baselines_consolidated.py  # Self-contained baseline script
├── TRAINING.md             # Training guide
└── README.md               # This file
```

---

## Key Takeaways

This project demonstrates a systematic approach to drug-drug interaction prediction:

1. **Start Simple**: Establish structure-only baselines to understand task difficulty
2. **Optimize Architecture**: Tune hyperparameters and training procedures before adding complexity
3. **Validate Choices**: Use ablations to understand what contributes to performance
4. **Engineer Features**: Invest in multi-modal feature engineering (chemistry + biology)
5. **Combine Signals**: Simple fusion of rich features outperforms complex architectures on sparse signals

**Final Performance**: Our best model achieves **73.28% test Hits@20**, a 6.6x improvement over the simple structure-only baseline, through comprehensive feature engineering and careful architectural choices.

---

## Citation

Dataset: Open Graph Benchmark (OGB)
```bibtex
@article{hu2020ogb,
  title={Open graph benchmark: Datasets for machine learning on graphs},
  author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={22118--22133},
  year={2020}
}
```