"""
Extract SMILES from DrugBank full database XML file.

Usage:
    python scripts/extract_drugbank_xml.py data/drugbank/full_database.xml
"""
import sys
import pandas as pd
import gzip
from pathlib import Path
import logging
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# DrugBank XML namespace
DRUGBANK_NS = '{http://www.drugbank.ca}'


def extract_smiles_from_xml(xml_path: str) -> dict:
    """
    Extract SMILES from DrugBank XML file.

    Args:
        xml_path: Path to DrugBank full_database.xml file

    Returns:
        dict: Mapping from DrugBank ID to SMILES
    """
    logger.info(f"Parsing XML file: {xml_path}")
    logger.info("This may take a few minutes...")

    drugbank_to_smiles = {}

    # Parse XML iteratively to handle large file
    context = ET.iterparse(xml_path, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)

    drug_count = 0
    smiles_count = 0

    for event, elem in context:
        if event == 'end' and elem.tag == f'{DRUGBANK_NS}drug':
            drug_count += 1

            # Get primary DrugBank ID
            drugbank_id = None
            for db_id in elem.findall(f'{DRUGBANK_NS}drugbank-id'):
                if db_id.get('primary') == 'true':
                    drugbank_id = db_id.text
                    break

            if not drugbank_id:
                logger.debug(f"Drug {drug_count} has no primary DrugBank ID, skipping")
                elem.clear()
                continue

            # Find SMILES in calculated-properties
            smiles = None
            calc_props = elem.find(f'{DRUGBANK_NS}calculated-properties')

            if calc_props is not None:
                for prop in calc_props.findall(f'{DRUGBANK_NS}property'):
                    kind = prop.find(f'{DRUGBANK_NS}kind')
                    value = prop.find(f'{DRUGBANK_NS}value')

                    if kind is not None and value is not None:
                        if kind.text == 'SMILES':
                            smiles = value.text
                            break

            if smiles:
                drugbank_to_smiles[drugbank_id] = smiles
                smiles_count += 1
                logger.debug(f"Found SMILES for {drugbank_id}: {smiles[:50]}...")
            else:
                logger.debug(f"No SMILES found for {drugbank_id}")

            # Clear element to free memory
            elem.clear()

            if drug_count % 1000 == 0:
                logger.info(f"Processed {drug_count} drugs, found SMILES for {smiles_count}")

    logger.info(f"Completed! Processed {drug_count} drugs, extracted SMILES for {smiles_count}")
    return drugbank_to_smiles


def main():
    if len(sys.argv) < 2:
        logger.info("Usage: python extract_drugbank_xml.py path/to/full_database.xml")
        logger.info("Using default path: data/drugbank/full_database.xml")
        xml_path = "data/drugbank/full_database.xml"
    else:
        xml_path = sys.argv[1]

    if not Path(xml_path).exists():
        logger.error(f"File not found: {xml_path}")
        sys.exit(1)

    # Extract SMILES from XML
    drugbank_to_smiles = extract_smiles_from_xml(xml_path)

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

    # Also remove the feature cache so it gets regenerated with real SMILES
    cache_path = "data/morgan_features_2048.pt"
    if Path(cache_path).exists():
        Path(cache_path).unlink()
        logger.info(f"\n✓ Removed old feature cache: {cache_path}")
        logger.info(f"  Morgan fingerprints will be regenerated from real SMILES on next run")

    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
