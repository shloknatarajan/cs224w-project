# SMILES Data for ogbl-ddi

## Current Status

The `smiles.csv` file contains **REAL SMILES** data extracted from DrugBank!

The data has been validated and is ready for experiments.

## How to Update or Re-extract SMILES

### Option 1: DrugBank (Official, Requires Account)

1. Create a free account at https://www.drugbank.ca/
2. Download the "Structures" dataset (SDF format)
3. Use `scripts/extract_drugbank_smiles.py` to extract SMILES for the drugs in the dataset

### Option 2: ChEMBL/PubChem (May have incomplete coverage)

Run the scripts in `scripts/` directory:
- `scripts/fetch_smiles_fast.py` - Attempts to fetch from public APIs (may have low success rate)

### Option 3: Manual Download

Some research groups have shared DrugBank SMILES mappings:
- Search GitHub for "drugbank smiles csv"
- Check Zenodo for pharmaceutical datasets
- Look for OGB-related repositories that may have preprocessed this data

## File Format

The `smiles.csv` file must have two columns:
- `ogb_id`: Node index (0 to 4266)
- `smiles`: SMILES string for the drug

Example:
```csv
ogb_id,smiles
0,CC(=O)Nc1nnc(s1)S(=O)(=O)N
1,Cc1onc(-c2ccccc2)c1C(=O)N[C@H]2[C@H]3SC(C)(C)[C@@H](N3C2=O)C(=O)O
...
```

## Using the SMILES Data

To use SMILES features in your models:
- Load the data with `smiles_csv_path` parameter in `load_dataset()`
- The loader will automatically convert SMILES to Morgan fingerprints
- Features are cached to speed up subsequent loads
