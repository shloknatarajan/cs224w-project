"""
Extract SMILES from DrugBank SDF file.

Usage:
    python scripts/extract_drugbank_smiles.py path/to/drugbank.sdf

This script reads a DrugBank structures SDF file and creates the smiles.csv
file needed for training.
"""
import sys
import pandas as pd
import gzip
from pathlib import Path
import logging
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_smiles_from_sdf(sdf_path: str) -> dict:
    """
    Extract SMILES from DrugBank SDF file.

    Args:
        sdf_path: Path to DrugBank structures SDF file

    Returns:
        dict: Mapping from DrugBank ID to SMILES
    """
    logger.info(f"Reading SDF file: {sdf_path}")

    drugbank_to_smiles = {}
    supplier = Chem.SDMolSupplier(sdf_path)

    for i, mol in enumerate(supplier):
        if mol is None:
            continue

        # Get DrugBank ID from molecule properties
        if mol.HasProp('DATABASE_ID'):
            drugbank_id = mol.GetProp('DATABASE_ID')
        elif mol.HasProp('DRUGBANK_ID'):
            drugbank_id = mol.GetProp('DRUGBANK_ID')
        elif mol.HasProp('_Name'):
            drugbank_id = mol.GetProp('_Name')
        else:
            logger.warning(f"Molecule {i} has no DrugBank ID, skipping")
            continue

        # Get SMILES
        smiles = Chem.MolToSmiles(mol)
        drugbank_to_smiles[drugbank_id] = smiles

        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i+1} molecules, found {len(drugbank_to_smiles)} with DrugBank IDs")

    logger.info(f"Extracted SMILES for {len(drugbank_to_smiles)} DrugBank compounds")
    return drugbank_to_smiles


def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python extract_drugbank_smiles.py path/to/drugbank.sdf")
        sys.exit(1)

    sdf_path = sys.argv[1]

    if not Path(sdf_path).exists():
        logger.error(f"File not found: {sdf_path}")
        sys.exit(1)

    # Extract SMILES from SDF
    drugbank_to_smiles = extract_smiles_from_sdf(sdf_path)

    # Read node to DrugBank ID mapping
    mapping_path = "dataset/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
    logger.info(f"Reading node-to-drug mapping from {mapping_path}")

    with gzip.open(mapping_path, 'rt') as f:
        df = pd.read_csv(f)

    logger.info(f"Found {len(df)} nodes/drugs in ogbl-ddi dataset")

    # Create smiles.csv
    results = []
    missing = []

    for _, row in df.iterrows():
        node_id = row['node idx']
        drug_id = row['drug id']

        if drug_id in drugbank_to_smiles:
            smiles = drugbank_to_smiles[drug_id]
            results.append({'ogb_id': node_id, 'smiles': smiles})
        else:
            # Empty SMILES for missing drugs
            results.append({'ogb_id': node_id, 'smiles': ''})
            missing.append(drug_id)

    # Save results
    output_path = "data/smiles.csv"
    Path("data").mkdir(exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    logger.info(f"\n{'='*80}")
    logger.info(f"✓ Saved SMILES data to {output_path}")
    logger.info(f"✓ Found SMILES for {len(results) - len(missing)}/{len(results)} drugs ({100*(len(results)-len(missing))/len(results):.1f}%)")

    if missing:
        logger.info(f"✗ Missing SMILES for {len(missing)} drugs")
        logger.info(f"  First 10 missing: {', '.join(missing[:10])}")

        # Save missing IDs
        missing_path = "data/missing_drugbank_ids.txt"
        with open(missing_path, 'w') as f:
            f.write('\n'.join(missing))
        logger.info(f"  Full list saved to {missing_path}")

    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
