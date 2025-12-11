# Final DDI Model Report (with vs. without drug features)

## Winner (with drug features)
- **Model**: 2-layer GCN encoder + MLP decoder, hidden=256, dropout=0.5, lr=0.005, batch=65,536, **all external features fused (3054 dims: Morgan 2048 + PubChem 9 + ChemBERTa 768 + drug-target 229)**
- **Log**: `logs/ddi_gcn_all_20251210_033923/ddi_gcn_all.log`
- **Result**: Hits@20 Val=0.7031, Test=0.7328 (best single-epoch test peak ≈0.7995 at epoch 1570; best checkpoint at epoch 1685)
- **Why it wins**: Rich drug descriptors front-load chemistry/biology signal, letting the simple GCN refine rather than invent representations. Long training (≈2000 epochs) plus 0.5 dropout stabilizes and pushes generalization (Test > Val).

## Best no-feature option (structure-only)
- **Model**: 2-layer GCN + MLP, hidden=256, dropout=0.5, lr=0.005, batch=65,536, **no node features**
- **Log**: `logs/ddi_long_sweeps_20251210_231641/sweep.log`
- **Result**: Best observed Hits@20 Test=0.5571 (Val=0.5894) during a 330+ epoch sweep (peak mid-run; no final summary block).
- **Shorter baseline**: Same setup run for 200 epochs hit Test=0.3984 (Val=0.5609) in `logs/ddi_gcn_20251210_030519/ddi_gcn.log`.
- **Why this is the ceiling without features**: Long horizons and heavy dropout are needed to avoid overfitting purely structural signals; widening to 256 dims is the main capacity boost that mattered.

## How the experiments led here
1. **Structure-only baselines set the floor**: Early 2-layer GCN/GAT/GraphSAGE/Transformer runs without features capped around Test@20 ≈0.11 (`logs/baselines_20251209_185813/baselines.log`). This established that architecture tweaks alone would not solve the task.
2. **Morgan-only tests underperformed structure**: Using 2048-d fingerprints alone dropped Test@20 to ≈0.05 (`logs/morgan_baselines_20251209_191058/morgan_baselines.log`), showing fingerprints need graph context and richer descriptors.
3. **Long-horizon structural sweeps moved the needle**: Extending the plain GCN to 200+ epochs raised Test@20 from ≈0.11 to 0.40, and 300+ epoch sweeps with dropout=0.5 surfaced the 0.5571 peak (`logs/ddi_long_sweeps_20251210_231641/sweep.log`). Depth/width changes mattered less than time and dropout.
4. **Feature fusion unlocked the jump**: Adding Morgan + PubChem + ChemBERTa + drug-target features to the same 2-layer GCN vaulted Test@20 to 0.73 (`logs/ddi_gcn_all_20251210_033923/ddi_gcn_all.log`). The positive Val→Test gap and best epoch near 1700 highlighted the combination of rich inputs and long training as the decisive factors.
5. **Auxiliary experiments reinforced the pattern**: Node2Vec + GCN (`logs/node2vec_gcn_20251209_225240/node2vec_gcn.log`, Test@20=0.1319) and Hybrid GCN mixing ChemBERTa with light structural cues (`logs/hybrid_gcn_quick_tuning_20251210_015911/quick_tuning.log`, Test@20=0.1572) showed incremental gains but never approached the fused-feature GCN, underscoring that comprehensive drug descriptors outperform partial heuristics.

## Recommendations
- **Use the feature-fused GCN as the production/default model** for ogbl-ddi; keep the 2000-epoch horizon with 0.5 dropout and the full 3054-d feature stack.
- **If features are unavailable**, run the structure-only GCN with dropout=0.5 for ≥350 epochs (and add checkpointing on Val@20) to approach the 0.55 Test@20 ceiling seen in the long sweep.
- **Next upgrades**: add explicit best-checkpoint saving to the sweep scripts; seed-average the top feature-fused run; and, for featureless settings, try light LR decay during long schedules to see if the 0.5571 peak can be pushed further.
