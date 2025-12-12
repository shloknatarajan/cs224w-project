echo "Training with Morgan fingerprints (1/5)"
pixi run python train_ddi_reference_external.py --morgan

echo "Training with PubChem properties (2/5)"
pixi run python train_ddi_reference_external.py --pubchem

echo "Training with ChemBERTa embeddings (3/5)"
pixi run python train_ddi_reference_external.py --chemberta

echo "Training with Drug-target features (4/5)"
pixi run python train_ddi_reference_external.py --drug-targets

echo "Training with all features (5/5)"
pixi run python train_ddi_reference_external.py --all