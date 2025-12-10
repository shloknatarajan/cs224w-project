"""
Fetch SMILES strings for DrugBank IDs in the ogbl-ddi dataset.

This script reads the node-to-drug mapping from the OGB dataset and attempts
to fetch SMILES strings from PubChem for each DrugBank ID.
"""
import pandas as pd
import gzip
import requests
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_smiles_from_pubchem(drugbank_id: str) -> str | None:
    """
    Fetch SMILES string for a DrugBank ID from PubChem.

    Args:
        drugbank_id: DrugBank ID (e.g., 'DB00001')

    Returns:
        SMILES string or None if not found
    """
    try:
        # Search PubChem for the DrugBank ID
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/{drugbank_id}/cids/JSON"
        response = requests.get(search_url, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        if 'IdentifierList' not in data or 'CID' not in data['IdentifierList']:
            return None

        cid = data['IdentifierList']['CID'][0]

        # Get SMILES for the CID
        smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
        response = requests.get(smiles_url, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
            smiles = data['PropertyTable']['Properties'][0].get('CanonicalSMILES')
            return smiles

    except Exception as e:
        logger.debug(f"Error fetching SMILES for {drugbank_id}: {e}")
        return None

    return None


def main():
    # Read node to DrugBank ID mapping
    mapping_path = "dataset/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
    logger.info(f"Reading node-to-drug mapping from {mapping_path}")

    with gzip.open(mapping_path, 'rt') as f:
        df = pd.read_csv(f)

    logger.info(f"Found {len(df)} nodes/drugs")

    # Create output dataframe
    results = []
    failed_drugs = []

    # Fetch SMILES for each drug
    for idx, row in df.iterrows():
        node_id = row['node idx']
        drug_id = row['drug id']

        logger.info(f"Fetching SMILES for node {node_id}: {drug_id} ({idx+1}/{len(df)})")

        smiles = fetch_smiles_from_pubchem(drug_id)

        if smiles:
            results.append({'ogb_id': node_id, 'smiles': smiles})
            logger.info(f"  ✓ Found SMILES: {smiles[:50]}...")
        else:
            # Add placeholder for missing SMILES
            results.append({'ogb_id': node_id, 'smiles': ''})
            failed_drugs.append(drug_id)
            logger.warning(f"  ✗ Could not fetch SMILES for {drug_id}")

        # Rate limiting - be nice to PubChem
        time.sleep(0.2)

    # Save results
    output_path = "data/smiles.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    logger.info(f"\n{'='*80}")
    logger.info(f"Saved SMILES data to {output_path}")
    logger.info(f"Successfully fetched: {len(results) - len(failed_drugs)}/{len(results)} drugs")
    logger.info(f"Failed to fetch: {len(failed_drugs)} drugs")

    if failed_drugs:
        logger.info(f"\nFailed DrugBank IDs (first 10): {', '.join(failed_drugs[:10])}")

        # Save failed IDs for reference
        failed_path = "data/failed_drugbank_ids.txt"
        with open(failed_path, 'w') as f:
            f.write('\n'.join(failed_drugs))
        logger.info(f"Full list of failed IDs saved to {failed_path}")

    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
