# DDI Reference Architecture

Reference link prediction stack for OGBL-DDI, mirroring the official OGB example while fitting this repo's conventions.

## Files
- Training entrypoint: `train_ddi_reference.py`
- Model components: `src/models/ogb_ddi_gnn/gnn.py`

## High-Level Flow
```
learnable node embeddings (N x H)
        ↓
GCN or GraphSAGE encoder (L layers, ReLU, optional dropout)
        ↓
node embeddings z ∈ R^{N x H}
        ↓
edge scoring: z_i ⊙ z_j → MLP → sigmoid
```

The ultra-minimal GCN baseline in this repo follows the same flow with dropout forced to **0.0** (enforced in `train_minimal_baseline`) and a Hadamard + 2-layer MLP decoder from `src/models/base.py`—it is not a pure dot-product head.

## Components
- **Node embeddings**: `torch.nn.Embedding` of size `[num_nodes, hidden_channels]`, randomly initialized per run and updated jointly with the encoder and decoder.
- **Encoder**: `GCN` or `SAGE` (select via `--use_sage`), each an `L`-layer stack (`cached=True` for GCN). Hidden size is constant across layers. Nonlinearity is ReLU. Dropout is applied after hidden layers only; set `--dropout 0.0` to align with the repo-wide zero-dropout guideline.
- **Decoder (LinkPredictor)**: Elementwise product of endpoint embeddings followed by an MLP with `num_layers` linear blocks, ReLU between hidden layers, final sigmoid to produce edge probabilities.
- **Parameter reset**: All modules expose `reset_parameters()` mirroring the OGB reference for clean restarts.

### GCN vs. GraphSAGE in this reference
- **Convolution:** GCN uses `GCNConv(..., cached=True)` with normalized aggregation; SAGE uses `SAGEConv` (mean aggregation) without caching.
- **Caching:** The GCN variant caches the normalized adjacency for speed; SAGE recomputes message passing each call.
- **Outputs/decoder:** Both feed the resulting node embeddings into the same LinkPredictor MLP decoder; only the encoder changes when `--use_sage` is set.

## Training Loop (matches OGB reference)
- **Objective**: Binary log-likelihood on positive edges plus equal-count negative samples (`torch_geometric.utils.negative_sampling` with dense mode).
- **Batching**: Mini-batch over positive edges (`batch_size`), with fresh negatives per batch.
- **Optimization**: Adam over encoder + embeddings + decoder; gradient clipping at 1.0 for all three parameter sets.
- **Evaluation**: Hits@{10,20,30} via `ogb.linkproppred.Evaluator`. A validation-sized subset of training edges (`eval_train`) is sampled for train-split metrics to match the reference script.
- **Logging**: Metrics and config saved under `logs/ddi_{gcn|sage}_TIMESTAMP/`.

## Default/Recommended Hyperparameters
- Layers: `--num_layers 2`
- Hidden width: `--hidden_channels 256`
- Dropout: **default 0.5 from the OGB example**
- LR: `0.005`
- Batch size: `64k` positive edges
- Epochs: `200`, eval every `--eval_steps` epochs (default 5), up to 10 runs averaged

## Fit for Purpose
- Baseline-quality reproduction of the official OGBL-DDI example for apples-to-apples comparisons.
- Minimal feature engineering: relies solely on learned node embeddings plus message passing, keeping the architecture clean for future ablations or feature injection experiments.

## Best Run in Repo Logs
- Log: `logs/ddi_gcn_all_20251210_033923/ddi_gcn_all.log`
- Setup: GCN encoder (2 layers, 256 hidden), MLP decoder (2 layers), dropout 0.5, lr 0.005, batch size 65,536, 2,000 epochs, eval every 5 epochs. External features concatenated: Morgan (2048), PubChem (9), ChemBERTa (768), drug-target (229) → total 3,054 dims.
- Result: Best validation at epoch **1685** with Hits@20 **0.7031**, corresponding test Hits@20 **0.7328**. Final logged metrics after 2,000 epochs: Hits@10 35.29%, Hits@20 73.28%, Hits@30 84.64%.

## How This Differs from Repo Baselines
- **Dropout**: Default 0.5 to mirror the OGB script; minimal baselines enforce `dropout=0.0` (trainer raises if any dropout layer is active).
- **Decoder**: Both use Hadamard-product MLPs; the baseline keeps it to 2 linear layers (H → H/2 → 1) with dropout=0.0, while the reference uses the configurable `LinkPredictor` (default dropout=0.5).
- **Edge sampling**: Dense-mode negative sampling + eval_train subset exactly like OGB; our other trainers often use lighter eval cadence and sometimes uniform negatives.
- **Feature usage**: Core reference stack is structure-only with learned embeddings; the best logged run above adds concatenated external chemical features (Morgan, PubChem, ChemBERTa, drug-targets), whereas baseline GCN/SAGE scripts typically remain structure-only.
- **Training horizon**: Reference-style runs can go long (2,000 epochs in the best run); baseline sweeps usually stop at 200–400 epochs unless specified.

### Architecture Side-by-Side (Baseline vs. DDI Reference)
```
MINIMAL GCN BASELINE (structure-only)     DDI REFERENCE (with optional externals)
─────────────────────────────────────     ─────────────────────────────────────────
learnable node emb (N x H)                learnable node emb (N x H)
        │                                         │
   2x GCN (ReLU, dropout=0)                 GCN/SAGE stack (L layers, dropout 0.5)
        │                                         │
   node reps z (no dropout)                      node reps z
        │                                         │
Hadamard product z_i ⊙ z_j                 elementwise product z_i ⊙ z_j
        │                                         │
2-layer MLP scorer (H→H/2→1, p=0.0)         MLP decoder (configurable depth, p=0.5)
        │                                         │
   sigmoid prob                              sigmoid prob

Optional in DDI reference: concatenate external chemical features to inputs before the encoder (Morgan, PubChem, ChemBERTa, drug-targets) when using the extended script/logged best run.
```
- Decoder: Baseline uses the Hadamard + 2-layer MLP in `src/models/base.py`; reference keeps the LinkPredictor MLP with configurable depth and dropout.
- Dropout placement/usage: Reference defaults to dropout (p=0.5) on encoder hidden layers and MLP hidden layers; baseline enforces zero-dropout end-to-end.
- Training protocol: Reference mirrors OGB with dense negative sampling per batch, an eval_train subset for Hits, and gradient clipping on embeddings/encoder/decoder. Baseline runs lighter eval/sampling and often omits clipping.
- Horizon: Reference is set up for long runs (up to 200–2000 epochs) to fully converge; baseline usually stops earlier.

So decoder swap + depth/width are key, but dropout and the reference-style sampling/eval/clipping are also differences in behavior.

## Sweep runner (reference model)
- Script: `run_ddi_sweeps.py`
- Config grids implemented:
  - A: layers=2; hidden {128,192,256}; dropout {0.0,0.1,0.25}; lr {0.005,0.003}; batch=64k
  - B: layers=3; hidden {128,192,256}; dropout {0.0,0.1}; lr {0.003,0.001}; batch=32k
  - C: layers=2; hidden=256; dropout {0.5,0.0}; lr {0.005,0.003}; batch=64k
- Defaults: `--epochs 100 --runs 1 --eval_steps 5`, GCN encoder. Add `--use_sage` to switch encoders.
- Example (screening all configs, 100 epochs, 1 seed): `pixi run python run_ddi_sweeps.py --epochs 100 --runs 1`
- Logs: `logs/ddi_sweeps_<timestamp>/sweep.log` plus per-config logs under `logs/ddi_sweeps_<timestamp>/<config>/config.log`.

The best logged run in docs/gcn_reference.md used the GCN encoder (2-layer, hidden 256) with the LinkPredictor MLP. SAGE is supported, but that specific successful run was GCN.

### Training differences in depth

- Optimization stack
      - Minimal: AdamW over model params (encoder + embeddings + decoder) with weight decay; no grad clipping; single run with optional early stopping (patience on val Hits); batch-wise BCE logits for pos/neg; dropout must be zero
        (raises otherwise).
      - Reference: Adam over encoder + embeddings + LinkPredictor; gradient clipping at 1.0 on embeddings, encoder, and decoder each step; fixed epoch budget per run (no early stop) and multiple runs averaged; dropout applied per
        config (default 0.5).
  - Negative sampling & batching
      - Minimal: negative_sampling(..., method="sparse") each epoch; negatives count matches number of train positives; processes positives and sampled negatives in edge mini-batches during decoding to control memory.
      - Reference: negative_sampling(..., method="dense") inside each batch; negatives count matches batch positives; uses dense edge index from adj_t COO for sampling.
  - Forward/decoding pattern
      - Minimal: Computes full-node embeddings once per epoch (encode on edge_index), then decodes batches of pos/neg edges via Hadamard + 2-layer MLP head; loss is mean of per-batch BCE logits.
      - Reference: Recomputes h = model(x, adj_t) inside each training batch; decoder is LinkPredictor MLP with sigmoid; loss is -log likelihood for pos and neg (no BCE wrapper), averaged over batch.
  - Eval protocol
      - Minimal: Evaluates on valid/test splits every eval_every epochs (default 5), no eval_train subset; uses provided evaluate helper with batch scoring; early stopping based on best val Hits.
      - Reference: Evaluates every eval_steps epochs (default 5); includes eval_train subset drawn from train edges for train-split Hits; reports Hits@{10,20,30} per run; tracks best per metric without early stop.
  - Hyperparameter defaults affecting training
      - Minimal defaults: hidden 128, lr 0.01, batch_size 50k, weight_decay 1e-4, epochs 200, dropout=0.0 enforced.
      - Reference defaults: hidden 256, lr 0.005, batch_size 64k, epochs 200 (often used longer), dropout=0.5 allowed on encoder/decoder, runs=10 averaged.
  - Stability/regularization levers
      - Minimal: Stability comes mainly from zero-dropout, AdamW, and early stopping; no clipping or eval_train subset.
      - Reference: Stability comes from dropout, gradient clipping, and dense negatives; no early stopping.
Notes:

• - LinkPredictor MLP: The edge scorer used in the DDI reference. It takes the Hadamard (elementwise) product of the two endpoint embeddings, then passes it through a small MLP: linear → ReLU → (dropout if enabled) repeated for
    num_layers-1 hidden blocks, ending in a final linear to 1 and a sigmoid. Depth, hidden width, and dropout come from the run config (defaults: depth = num_layers, dropout = 0.5 in the reference).
  - Gradient clipping: A training safeguard that caps the norm of gradients before the optimizer step. In the reference loop, torch.nn.utils.clip_grad_norm_(...) is applied to embeddings, encoder, and decoder with a max norm of 1.0.
    This prevents exploding gradients, stabilizes updates, and can improve convergence when using larger learning rates or deeper models.
## Desigining Build Up Experiments
### Chat recommendations
• I’d avoid a full grid (too expensive) and do a small random/Latin hypercube sweep over the big movers:

- Encoder width: e.g., 128, 192, 256, 320 (maybe 384 if memory allows); keep 2–3 layers max.
- MLP depth/width: 2 vs 3 layers; hidden 128 vs 256 to match/decouple the encoder.
- Dropout: start at the repo-preferred 0.0; maybe test 0.1 and 0.25 if you think regularization helps. Skip 0.5 unless you need to replicate the OGB script; it’s usually suboptimal here.

- Establish a clean baseline: train the existing GCN with zero-dropout, simple scorer, and current sampling/eval. Log Hits@K and loss curves for a few seeds.
- Probe the decoder: replace the simple scorer with an elementwise-product MLP (2 layers) while keeping encoder fixed. If Hits@K jump, you’ve identified a better edge function.
- Align encoder capacity: sweep a small set of widths (e.g., 128→256) at fixed depth (2 layers). If gains saturate, no need to go wider; if not, increase cautiously until memory/overfit shows up.
- Add dropout cautiously: toggle dropout on hidden layers (encoder and MLP) at low values (0.1–0.25). Keep runs short to see if regularization helps or hurts.
- Tighten training protocol: introduce dense negative sampling and gradient clipping (norm=1.0) to stabilize training at larger batch sizes. Observe if variance drops and best Hits improve.
- Extend training horizon: once stable, allow longer training (e.g., up to 200–500 epochs initially) and check if the model keeps improving; if so, consider longer runs to reach the reference behavior.
- Confirm eval parity: add the eval_train subset for Hits measurement to compare fairly across runs.
- Iterate with small sweeps (width, decoder depth, dropout) guided by the above signals, stopping when gains flatten.

## Build Up
1. Train baselines a couple times with just the GCN
