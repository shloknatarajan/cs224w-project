# Intro

This project tackles link prediction on the ogbl-ddi dataset (4,267 drugs; ~2.1M interactions), targeting Hits@K (K=10/20/30) with validation-driven model selection. We began with structure-only GNN baselines, then explored architectural variants and long-horizon training, and finally integrated drug-level features (Morgan fingerprints, ChemBERTa embeddings, PubChem properties, and drug-target context). Each section below cites specific runs with their logs for reproducibility.

Key references for setup and context:
- Dataset and sources: ../docs/EXTERNAL_DATA_SOURCES.md
- Aggregate results summary: ../docs/MODEL_RESULTS_SUMMARY.md


# Baselines

## Explanation of the initial run
We established baseline performance using four different structure-only GNN models: GCN, GraphSAGE, GAT, and GraphTransformer. These models were trained for 200 epochs with a learning rate of 0.01 and a batch size of 50,000.

The results showed that the GCN model performed the best among the baselines:
- **GCN**: Validation Hits@20: 13.59%, Test Hits@20: 11.02%
- **GraphSAGE**: Validation Hits@20: 9.61%, Test Hits@20: 4.46%
- **GraphTransformer**: Validation Hits@20: 11.66%, Test Hits@20: 3.73%
- **GAT**: Validation Hits@20: 9.53%, Test Hits@20: 4.88%

This GCN run establishes the anchor performance for the structure-only setting.

Key log(s):
- Baselines (GCN, GraphSAGE, GAT, GraphTransformer): ../logs/baselines_20251209_185813/baselines.log

## Baseline takeaway
This minimal GCN+MLP (structure-only) run establishes the baseline reference without any sweep results or extended training schedules.


# Top Performing Architecture (no drug information)

## Explanation of architecture
A 2-layer GCN encoder with 256 hidden channels and an MLP decoder trained purely on graph structure (no external features). This serves as the best-performing structure-only model family after tuning.

The best performing model from the long-horizon sweep achieved:
- **Validation Hits@20**: 64.78%
- **Test Hits@20**: 61.69%

This was achieved with the following configuration:
- **Architecture**: GCN with 2 layers and 256 hidden channels.
- **Dropout**: 0.5
- **Training**: 2000 epochs with a learning rate of 0.005 and a batch size of 65536. The model with the best validation score was found at epoch 430.

Representative log(s):
- Short/medium budget (structure-only): ../logs/ddi_gcn_20251210_030519/ddi_gcn.log
- Long-horizon (structure-only): ../logs/ddi_long_sweeps_20251210_231641/sweep.log
- Broader GCN sweeps (structure-only variants): ../logs/ddi_sweeps_20251210_200632/sweep.log

## Some findings from the sweeps (info stored in logs)
- **Epochs**: Increasing epochs from 100 to 200 significantly improves performance. The long-horizon run shows that performance continues to improve up to ~430 epochs.
- **Dropout**: Dropout is crucial for generalization. The best results were achieved with dropout rates of 0.25 and 0.5. For example, the configuration `A_l2_h256_do0.25_lr0.005_bs64k` achieved a validation Hits@20 of 38.60%, while the same configuration with 0.5 dropout reached 64.78% in the long run.
- **Learning Rate and Batch Size**: A learning rate of 0.005 with a large batch size of 65536 was found to be a stable and effective combination.
- **Hidden Channels**: Increasing hidden channels from 128 to 256 consistently improved performance across different configurations.

## How the other experiments support this
Comparative experiments reinforce that, without node features, GCN is the most reliable baseline among the tested architectures, and that variants like Node2Vec+GCN and GDIN provide useful context but do not surpass tuned GCN in this setting. The hybrid models, which combine structural and chemical features, show promising results, with the `Hybrid-GCN-Simple` model outperforming the structure-only baseline. The SEAL model, which takes a different subgraph-based approach, achieves high accuracy (76% validation accuracy after only 3 epochs), but its results are not directly comparable to the Hits@K metrics of the other models.

Supporting log(s):
- Baselines (structure-only comparison): ../logs/baselines_20251209_185813/baselines.log
- Node2Vec + GCN: ../logs/node2vec_gcn_20251209_225240/node2vec_gcn.log
- GDIN (deconfounded): ../logs/gdin_20251209_233115/gdin.log
- SEAL (subgraph-based reference): ../logs/seal_20251208_231608/seal.log
- Hybrid GCN Models: ../logs/hybrid_20251209_231044/hybrid.log

# Top Performing Architecture (with drug information)

## Explanation of architecture
Our best model concatenates multiple feature sources (Morgan 2048 + ChemBERTa 768 + PubChem properties + drug-target context) and feeds them into a 2-layer GCN encoder with an MLP decoder. The total feature dimension is 3054. The model was trained for 2000 epochs with a learning rate of 0.005 and a batch size of 65536.

This approach yielded the best performance, with a **validation Hits@20 of 70.31%** and a **test Hits@20 of 73.28%** at epoch 1685. This demonstrates the significant benefit of combining multiple feature sources.

Key log(s) and artifacts:
- Best model log: ../logs/ddi_gcn_all_20251210_033923/ddi_gcn_all.log
- Training curves: ../logs/ddi_gcn_all_20251210_033923/training_curves.png (and .pdf)
- Comparative summaries: ../docs/MODEL_RESULTS_SUMMARY.md, ../docs/FINAL_DDI_MODEL_REPORT.md


# Introducing Drug Features

Adding drug-level features dramatically lifts performance. We evaluated feature families individually and in combination.

## Morgan Finger Prints
We generated 2048-d Morgan fingerprints from SMILES and trained GCN, GraphSAGE, GAT, and GraphTransformer models on these features. The GCN model performed the best among these, but all models underperformed the structure-only baselines.

The results for the Morgan fingerprint baselines were:
- **Morgan-GCN**: Validation Hits@20: 6.10%, Test Hits@20: 5.36%
- **Morgan-GraphSAGE**: Validation Hits@20: 3.05%, Test Hits@20: 4.71%
- **Morgan-GraphTransformer**: Validation Hits@20: 4.14%, Test Hits@20: 3.98%
- **Morgan-GAT**: Validation Hits@20: 4.32%, Test Hits@20: 1.12%

As a standalone signal, Morgan features underperformed structure-only baselines but are valuable when combined with other modalities.

Key log(s):
- Morgan-only baselines: ../logs/morgan_baselines_20251209_191058/morgan_baselines.log
- SMILES processing and Morgan build (example run output): ../logs/real_drugbank_run.log

## ChemBERTa embeddings
ChemBERTa provides 768-d semantic embeddings of SMILES. Because 46% of nodes lack SMILES, we used a learnable fallback embedding for missing chemistry to avoid zero vectors and preserve gradient flow. We ran baselines with GCN, GraphSAGE, GAT, and GraphTransformer models on top of these embeddings.

The results showed that the GCN and GraphSAGE models performed the best:
- **ChemBERTa-GCN-Fallback**: Validation Hits@20: 11.37%, Test Hits@20: 6.94%
- **ChemBERTa-GraphSAGE-Fallback**: Validation Hits@20: 11.29%, Test Hits@20: 9.07%
- **ChemBERTa-GraphTransformer-Fallback**: Validation Hits@20: 9.41%, Test Hits@20: 4.99%
- **ChemBERTa-GAT-Fallback**: Validation Hits@20: 6.19%, Test Hits@20: 2.60%

Standalone ChemBERTa models lag structure-only GCN but are useful as part of a combined feature set.

Key log(s) and reference:
- ChemBERTa fallback runs: ../logs/chemberta_fallback_20251209_210608/chemberta_fallback.log, ../logs/chemberta_fallback_20251209_212319/chemberta_fallback.log
- Design/implementation details: ../docs/CHEMBERTA_FALLBACK.md

## PubChem properties
Hand-engineered PubChem descriptors (e.g., physico-chemical properties) were included in the final combined feature representation. These 9 features represent a small part of the total 3054-dimensional feature vector. While we did not run a PubChem-only baseline, their contribution is captured in the combined model logs below.

Reference evidence in combined model:
- See feature breakdown and outcomes in: ../logs/ddi_gcn_all_20251210_033923/ddi_gcn_all.log
- External sources and properties catalog: ../docs/EXTERNAL_DATA_SOURCES.md

## Hybrid GCN Models
To address the performance degradation observed when using only chemical features, we developed hybrid models that combine a structure-only GCN encoder with a chemistry-aware decoder. This approach preserves the strong baseline performance of the structure-only GCN while allowing the model to leverage chemical information when available.

We experimented with three different hybrid GCN models:
- **Hybrid-GCN-ChemAware**: A three-path decoder with learnable weights for structural, chemical, and combined scores.
- **Hybrid-GCN-Simple**: An additive model that combines structural and chemical scores with a learnable weight.
- **Hybrid-GCN-ChemAware-Gated**: A model with a dynamic gate that determines the importance of chemistry for each edge.

The results showed that the `Hybrid-GCN-Simple` model performed the best among the hybrid models, and also outperformed the structure-only baseline:
- **Hybrid-GCN-Simple**: Validation Hits@20: 23.49%, Test Hits@20: 19.33%
- **Hybrid-GCN-ChemAware**: Validation Hits@20: 18.82%, Test Hits@20: 15.57%
- **Hybrid-GCN-ChemAware-Gated**: Validation Hits@20: 18.59%, Test Hits@20: 21.60%

Key log(s) and reference:
- Hybrid GCN Models: ../logs/hybrid_20251209_231044/hybrid.log
- Hybrid GCN Summary: ../docs/HYBRID_GCN_SUMMARY.md



# Ablations (Structure-Only GCN)

We ran fast ablations on the minimal baseline (structure-only GCN+MLP) to assess the impact of depth, dropout, and learning rate while keeping runs short and comparable (200 epochs, eval every 10).

The results are summarized in the table below:

| Tag | num_layers | hidden | dropout | lr | batch_size | Hits@20 (val/test) | Hits@30 (val/test) | Log |
| --- | ---------- | ------ | ------- | -- | ---------- | ------------------ | ------------------ | --- |
| E1 (baseline) | 2 | 256 | 0.0 | 0.0050 | 65536 | 36.45 / 17.36 | 49.51 / 27.76 | ddi_gcn_20251210_233435/ddi_gcn.log |
| E2 (dropout) | 2 | 256 | 0.25 | 0.0050 | 65536 | **50.69 / 34.34** | **59.37 / 41.69** | ddi_gcn_20251210_233939/ddi_gcn.log |
| E3 (3-layer, low lr) | 3 | 192 | 0.0 | 0.0010 | 32768 | 31.82 / 18.13 | 36.09 / 29.38 | ddi_gcn_20251210_234445/ddi_gcn.log |
| E4 (3-layer + dropout) | 3 | 192 | 0.1 | 0.0010 | 32768 | 38.24 / 18.10 | 44.92 / 26.22 | ddi_gcn_20251210_235031/ddi_gcn.log |
| E5 (3-layer, lr bump) | 3 | 192 | 0.0 | 0.0015 | 32768 | 34.55 / 12.30 | 41.09 / 24.37 | ddi_gcn_20251210_235621/ddi_gcn.log |

Key log(s) and summary:
- Ablation summary and per-run links: ../logs/ddi_gcn_ablations_20251211_005945/README.md

Findings:
- **Dropout helps significantly**: The best performing model (E2) used a dropout of 0.25, which resulted in a large improvement in both validation and test Hits@20 compared to the baseline with no dropout (E1).
- **Deeper is not always better**: The 3-layer models (E3, E4, E5) did not outperform the 2-layer models. This suggests that for this dataset and architecture, 2 layers are sufficient.
- **Hyperparameter sensitivity**: The results show that the model is sensitive to hyperparameters like learning rate and dropout. The best results were obtained with a higher learning rate (0.005) and a non-zero dropout.
