# Intro

This project tackles link prediction on the ogbl-ddi dataset (4,267 drugs; ~2.1M interactions), targeting Hits@K (K=10/20/30) with validation-driven model selection. We began with structure-only GNN baselines, then explored architectural variants and long-horizon training, and finally integrated drug-level features (Morgan fingerprints, ChemBERTa embeddings, PubChem properties, and drug-target context). Each section below cites specific runs with their logs for reproducibility.

Key references for setup and context:
- Dataset and sources: ../docs/EXTERNAL_DATA_SOURCES.md
- Aggregate results summary: ../docs/MODEL_RESULTS_SUMMARY.md


# Baselines

## Explanation of the initial run
We use a minimal baseline architecture: a 2-layer GCN encoder with an MLP decoder, trained on graph structure only (no node features). This run establishes the anchor performance for the structure-only setting.

Key log(s):
- Minimal baseline (GCN, 200 epochs): ../logs/ddi_gcn_20251210_030519/ddi_gcn.log

## Some explanations of sweeps and what this means for likely top performing model
We ran sweeps over layers, hidden size, dropout, learning rate, and batch size for the same minimal baseline (GCN+MLP, structure-only), including short/medium budgets and long-horizon training. These show that tuning and longer schedules substantially lift performance; the best structure-only peaks land in the mid-0.5 Hits@20 range.

Key log(s) — minimal baseline only:
- GCN sweeps (2-layer focus: A_l2_* configs): ../logs/ddi_sweeps_20251210_200632/sweep.log
- Long-horizon GCN (2-layer): ../logs/ddi_long_sweeps_20251210_231641/sweep.log
- Additional long-horizon variant: ../logs/ddi_long_sweeps_20251210_223703/sweep.log

Implications:
- Longer schedules and tuned dropout (often ~0.5) markedly improve structure-only GCN.


# Top Performing Architecture (no drug information)

## Explanation of architecture
A 2-layer GCN encoder with 256 hidden channels and an MLP decoder trained purely on graph structure (no external features). This serves as the best-performing structure-only model family after tuning.

Representative log(s):
- Short/medium budget (structure-only): ../logs/ddi_gcn_20251210_030519/ddi_gcn.log
- Long-horizon (structure-only): ../logs/ddi_long_sweeps_20251210_231641/sweep.log
- Broader GCN sweeps (structure-only variants): ../logs/ddi_sweeps_20251210_200632/sweep.log

## Some findings from the sweeps (info stored in logs)
- Increasing epochs from 20 → 200 markedly improves test Hits@20; extending to 300–600+ continues to help.
- Dropout around 0.5 often yields the strongest structure-only generalization in these runs.
- Learning rate 0.003–0.005 with large batches (32k–65k) is a stable region.

## How the other experiments support this
Comparative experiments reinforce that, without node features, GCN is the most reliable baseline among the tested architectures, and that variants like Node2Vec+GCN and GDIN provide useful context but do not surpass tuned GCN in this setting.

Supporting log(s):
- Baselines (structure-only comparison): ../logs/baselines_20251209_185813/baselines.log
- Node2Vec + GCN: ../logs/node2vec_gcn_20251209_225240/node2vec_gcn.log
- GDIN (deconfounded): ../logs/gdin_20251209_233115/gdin.log
- SEAL (subgraph-based reference): ../logs/seal_20251208_231608/seal.log


# Introducing Drug Features

Adding drug-level features dramatically lifts performance. We evaluated feature families individually and in combination.

## Morgan Finger Prints
We generated 2048-d Morgan fingerprints from SMILES and trained shallow GNN variants on these features. As a standalone signal, Morgan features underperformed structure-only baselines but are valuable when combined with other modalities.

Key log(s):
- Morgan-only baselines: ../logs/morgan_baselines_20251209_191058/morgan_baselines.log
- SMILES processing and Morgan build (example run output): ../logs/real_drugbank_run.log

## ChemBERTa embeddings
ChemBERTa provides 768-d semantic embeddings of SMILES. Because 46% of nodes lack SMILES, we used a learnable fallback embedding for missing chemistry to avoid zero vectors and preserve gradient flow. Standalone ChemBERTa models lag structure-only GCN but are useful as part of a combined feature set.

Key log(s) and reference:
- ChemBERTa fallback runs: ../logs/chemberta_fallback_20251209_210608/chemberta_fallback.log, ../logs/chemberta_fallback_20251209_212319/chemberta_fallback.log
- Design/implementation details: ../docs/CHEMBERTA_FALLBACK.md

## PubChem properties
Hand-engineered PubChem descriptors (e.g., physico-chemical properties) were included in the final combined feature representation. While we did not run a PubChem-only baseline, their contribution is captured in the combined model logs below.

Reference evidence in combined model:
- See feature breakdown and outcomes in: ../logs/ddi_gcn_all_20251210_033923/ddi_gcn_all.log
- External sources and properties catalog: ../docs/EXTERNAL_DATA_SOURCES.md

## Explanation of the final integration with all the drug features
Our best model concatenates multiple feature sources (Morgan 2048 + ChemBERTa 768 + PubChem properties + drug-target context) and feeds them into a 2-layer GCN encoder with an MLP decoder. Message passing refines the rich initial representations, leading to strong generalization and the overall best performance observed.

Key log(s) and artifacts:
- Best model log: ../logs/ddi_gcn_all_20251210_033923/ddi_gcn_all.log
- Training curves: ../logs/ddi_gcn_all_20251210_033923/training_curves.png (and .pdf)
- Comparative summaries: ../docs/MODEL_RESULTS_SUMMARY.md, ../docs/FINAL_DDI_MODEL_REPORT.md

Highlights from best run (Hits@20): validation ~70.3%, test ~73.3% at ~epoch 1685, with positive test generalization relative to validation.


# Ablations (Minimal Baseline GCN)

We ran fast ablations on the minimal baseline (structure-only GCN+MLP) to assess the impact of depth, dropout, and learning rate while keeping runs short and comparable (200 epochs, eval every 10).

Key log(s) and summary:
- Ablation summary and per-run links: ../logs/ddi_gcn_ablations_20251211_005945/README.md
- Runs (each contains `ddi_gcn.log`):
  - Baseline 2-layer, no dropout: ../logs/ddi_gcn_ablations_20251211_005945/ddi_gcn_20251210_233435/ddi_gcn.log
  - 2-layer with dropout=0.25: ../logs/ddi_gcn_ablations_20251211_005945/ddi_gcn_20251210_233939/ddi_gcn.log
  - 3-layer, low lr (0.001): ../logs/ddi_gcn_ablations_20251211_005945/ddi_gcn_20251210_234445/ddi_gcn.log
  - 3-layer + dropout=0.1: ../logs/ddi_gcn_ablations_20251211_005945/ddi_gcn_20251210_235031/ddi_gcn.log
  - 3-layer, lr=0.0015: ../logs/ddi_gcn_ablations_20251211_005945/ddi_gcn_20251210_235621/ddi_gcn.log

Findings:
- Dropout helps: 2-layer with dropout=0.25 significantly improves generalization over no-dropout.
- Depth tradeoff: Moving to 3 layers at smaller hidden size (192) with lower lr underperforms 2-layer 256-d settings in test metrics.
- Short budgets remain sensitive to lr; modest increases don’t close the gap versus tuned 2-layer configs with dropout.
