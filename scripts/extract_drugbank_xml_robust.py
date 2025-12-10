"""
Extract SMILES from DrugBank full database XML file (robust version).

This version uses lxml for better error handling with large/malformed XML files.

Usage:
    python scripts/extract_drugbank_xml_robust.py
"""
import pandas as pd
import gzip
from pathlib import Path
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_smiles_regex(xml_path: str) -> dict:
    """
    Extract SMILES using regex patterns (more robust for malformed XML).

    Args:
        xml_path: Path to DrugBank XML file

    Returns:
        dict: Mapping from DrugBank ID to SMILES
    """
    logger.info(f"Reading XML file: {xml_path}")
    logger.info("This may take a moment (reading large file)...")

    with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    logger.info(f"File read successfully ({len(content) / 1e6:.1f} MB)")
    logger.info("Extracting DrugBank IDs and SMILES...")

    # Pattern to match drug entries
    # Matches: <drug ...> ... </drug>
    drug_pattern = re.compile(
        r'<drug[^>]*>(.*?)</drug>',
        re.DOTALL
    )

    # Pattern to extract primary DrugBank ID
    drugbank_id_pattern = re.compile(
        r'<drugbank-id[^>]*primary="true"[^>]*>([^<]+)</drugbank-id>'
    )

    # Pattern to extract SMILES
    smiles_pattern = re.compile(
        r'<property>.*?<kind>SMILES</kind>.*?<value>([^<]+)</value>.*?</property>',
        re.DOTALL
    )

    drugbank_to_smiles = {}
    drug_count = 0

    for drug_match in drug_pattern.finditer(content):
        drug_count += 1
        drug_xml = drug_match.group(1)

        # Extract DrugBank ID
        id_match = drugbank_id_pattern.search(drug_xml)
        if not id_match:
            continue

        drugbank_id = id_match.group(1)

        # Extract SMILES
        smiles_match = smiles_pattern.search(drug_xml)
        if smiles_match:
            smiles = smiles_match.group(1)
            drugbank_to_smiles[drugbank_id] = smiles

        if drug_count % 1000 == 0:
            logger.info(f"Processed {drug_count} drugs, found {len(drugbank_to_smiles)} with SMILES")

    logger.info(f"Completed! Processed {drug_count} drugs, extracted SMILES for {len(drugbank_to_smiles)}")
    return drugbank_to_smiles


def main():
    xml_path = "data/drugbank/full_database.xml"

    if not Path(xml_path).exists():
        logger.error(f"File not found: {xml_path}")
        return

    # Extract SMILES from XML
    drugbank_to_smiles = extract_smiles_regex(xml_path)

    # Read node to DrugBank ID mapping from OGB dataset
    mapping_path = "dataset/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
    logger.info(f"\nReading node-to-drug mapping from {mapping_path}")

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

    # Remove the feature cache so it gets regenerated with real SMILES
    cache_path = "data/morgan_features_2048.pt"
    if Path(cache_path).exists():
        Path(cache_path).unlink()
        logger.info(f"\n✓ Removed old feature cache: {cache_path}")
        logger.info(f"  Morgan fingerprints will be regenerated from real SMILES on next run")

    logger.info(f"{'='*80}")
    logger.info("\nYou can now run: python train_morgan_baselines.py")


if __name__ == "__main__":
    main()
