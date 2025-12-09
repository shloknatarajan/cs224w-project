"""
Smoke tests for the SMILES → Morgan fingerprint pipeline.

This test validates the entire data pipeline:
1. OGB node ID → DrugBank ID
2. DrugBank ID → SMILES
3. SMILES → Morgan fingerprint
4. Verify fingerprints are valid and have expected properties
"""
import pytest
import pandas as pd
import gzip
import torch
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

from src.data.data_loader import smiles_to_morgan, build_smiles_feature_matrix


class TestSMILESPipeline:
    """Test the complete SMILES data pipeline."""

    @pytest.fixture
    def drugbank_mapping(self):
        """Load the OGB node ID to DrugBank ID mapping."""
        mapping_path = "dataset/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
        with gzip.open(mapping_path, 'rt') as f:
            df = pd.read_csv(f)
        return df

    @pytest.fixture
    def smiles_data(self):
        """Load the SMILES CSV."""
        smiles_path = "data/smiles.csv"
        df = pd.read_csv(smiles_path, keep_default_na=False)
        return df

    def test_mapping_file_exists(self, drugbank_mapping):
        """Test that the DrugBank mapping file exists and is valid."""
        assert drugbank_mapping is not None
        assert 'node idx' in drugbank_mapping.columns
        assert 'drug id' in drugbank_mapping.columns
        assert len(drugbank_mapping) == 4267  # ogbl-ddi has 4267 nodes

    def test_smiles_file_exists(self, smiles_data):
        """Test that the SMILES file exists and has correct structure."""
        assert smiles_data is not None
        assert 'ogb_id' in smiles_data.columns
        assert 'smiles' in smiles_data.columns
        assert len(smiles_data) == 4267

    def test_node_to_drugbank_mapping(self, drugbank_mapping):
        """Test mapping from OGB node ID to DrugBank ID."""
        # Test a few known mappings
        node_0 = drugbank_mapping[drugbank_mapping['node idx'] == 0]
        assert len(node_0) == 1
        assert node_0['drug id'].values[0] == 'DB00001'

        node_1 = drugbank_mapping[drugbank_mapping['node idx'] == 1]
        assert len(node_1) == 1
        assert node_1['drug id'].values[0] == 'DB00002'

    def test_drugbank_to_smiles_mapping(self, drugbank_mapping, smiles_data):
        """Test that we can map DrugBank IDs to SMILES."""
        # Find a node with SMILES
        non_empty_smiles = smiles_data[smiles_data['smiles'] != '']
        assert len(non_empty_smiles) > 0, "Should have at least some SMILES strings"

        # Pick the first node with SMILES
        test_node_id = non_empty_smiles.iloc[0]['ogb_id']
        test_smiles = non_empty_smiles.iloc[0]['smiles']

        # Verify the DrugBank ID exists for this node
        drug_id = drugbank_mapping[drugbank_mapping['node idx'] == test_node_id]['drug id'].values[0]
        assert drug_id is not None
        assert drug_id.startswith('DB')

        # Verify SMILES is valid
        assert test_smiles is not None
        assert len(test_smiles) > 0

    def test_smiles_coverage(self, smiles_data):
        """Test that we have reasonable SMILES coverage."""
        non_empty = smiles_data[smiles_data['smiles'] != '']
        coverage = len(non_empty) / len(smiles_data)

        print(f"\nSMILES Coverage: {len(non_empty)}/{len(smiles_data)} ({coverage*100:.1f}%)")

        # We should have SMILES for at least 50% of drugs (small molecules)
        # The rest are likely biotech drugs (proteins, peptides, antibodies)
        assert coverage >= 0.50, f"Expected at least 50% SMILES coverage, got {coverage*100:.1f}%"

    def test_smiles_to_morgan_conversion(self, smiles_data):
        """Test converting SMILES to Morgan fingerprints."""
        # Get a valid SMILES string
        valid_smiles = smiles_data[smiles_data['smiles'] != ''].iloc[0]['smiles']

        # Convert to Morgan fingerprint
        fp = smiles_to_morgan(valid_smiles, n_bits=2048, radius=2)

        # Verify fingerprint properties
        assert fp is not None
        assert isinstance(fp, torch.Tensor)
        assert fp.shape == (2048,)
        assert fp.dtype == torch.float32

        # Fingerprint should be binary (only 0s and 1s)
        assert torch.all((fp == 0) | (fp == 1))

        # Fingerprint should not be all zeros (valid SMILES should have some bits set)
        assert torch.sum(fp) > 0, "Valid SMILES should produce non-zero fingerprint"

    def test_invalid_smiles_returns_zeros(self):
        """Test that invalid SMILES returns zero vector."""
        invalid_smiles = "INVALID_SMILES_STRING"
        fp = smiles_to_morgan(invalid_smiles, n_bits=2048, radius=2)

        assert fp is not None
        assert fp.shape == (2048,)
        assert torch.all(fp == 0), "Invalid SMILES should return zero vector"

    def test_empty_smiles_returns_zeros(self):
        """Test that empty SMILES returns zero vector."""
        empty_smiles = ""
        fp = smiles_to_morgan(empty_smiles, n_bits=2048, radius=2)

        assert fp is not None
        assert fp.shape == (2048,)
        assert torch.all(fp == 0), "Empty SMILES should return zero vector"

    def test_morgan_fingerprint_validity(self, smiles_data):
        """Test multiple SMILES to ensure fingerprints are valid."""
        valid_smiles = smiles_data[smiles_data['smiles'] != ''].head(10)

        for _, row in valid_smiles.iterrows():
            smiles = row['smiles']
            fp = smiles_to_morgan(smiles, n_bits=2048, radius=2)

            # Basic checks
            assert fp.shape == (2048,)
            assert torch.all((fp == 0) | (fp == 1))

            # Verify using RDKit directly
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:  # Valid molecule
                # Should produce non-zero fingerprint
                assert torch.sum(fp) > 0, f"Valid SMILES '{smiles[:50]}...' should produce non-zero fingerprint"

    def test_different_molecules_different_fingerprints(self, smiles_data):
        """Test that different molecules produce different fingerprints."""
        valid_smiles = smiles_data[smiles_data['smiles'] != ''].head(5)

        fingerprints = []
        for _, row in valid_smiles.iterrows():
            fp = smiles_to_morgan(row['smiles'], n_bits=2048, radius=2)
            fingerprints.append(fp)

        # Check that not all fingerprints are identical
        first_fp = fingerprints[0]
        all_same = all(torch.all(fp == first_fp) for fp in fingerprints)

        assert not all_same, "Different molecules should produce different fingerprints"

    def test_build_feature_matrix(self, smiles_data):
        """Test building the full feature matrix."""
        # Create a temporary SMILES file with just 10 nodes
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data = smiles_data.head(10)
            test_data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            # Build feature matrix
            feat = build_smiles_feature_matrix(
                num_nodes=10,
                smiles_csv_path=temp_path,
                n_bits=2048,
                radius=2
            )

            # Verify shape and type
            assert feat.shape == (10, 2048)
            assert feat.dtype == torch.float32

            # Check that we have some non-zero features (for nodes with SMILES)
            non_zero_nodes = torch.sum(feat, dim=1) > 0
            assert torch.sum(non_zero_nodes) > 0, "Should have some nodes with features"

        finally:
            # Clean up temp file
            Path(temp_path).unlink()

    def test_specific_example_pipeline(self, drugbank_mapping, smiles_data):
        """Test the complete pipeline with a specific example."""
        # Find node 4 (DB00006) which should have SMILES
        node_id = 4

        # Step 1: Get DrugBank ID
        drug_id = drugbank_mapping[drugbank_mapping['node idx'] == node_id]['drug id'].values[0]
        print(f"\nNode {node_id} → DrugBank ID: {drug_id}")

        # Step 2: Get SMILES
        smiles = smiles_data[smiles_data['ogb_id'] == node_id]['smiles'].values[0]
        print(f"DrugBank {drug_id} → SMILES: {smiles[:80]}...")

        # Step 3: Convert to Morgan fingerprint
        fp = smiles_to_morgan(smiles, n_bits=2048, radius=2)
        print(f"SMILES → Morgan FP: shape={fp.shape}, dtype={fp.dtype}")
        print(f"  Non-zero bits: {torch.sum(fp).item()}/{fp.shape[0]}")
        print(f"  First 20 bits: {fp[:20].tolist()}")

        # Verify the pipeline worked
        if smiles and smiles != '':
            # Should have valid fingerprint
            assert fp.shape == (2048,)
            assert torch.sum(fp) > 0, "Valid SMILES should produce non-zero fingerprint"

            # Verify with RDKit
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Compare with RDKit directly
                rdkit_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                rdkit_bits = set(rdkit_fp.GetOnBits())
                our_bits = set(torch.nonzero(fp).squeeze().tolist())

                # Should match
                assert rdkit_bits == our_bits, "Our fingerprint should match RDKit's"
        else:
            # Should have zero fingerprint
            assert torch.all(fp == 0), "Empty SMILES should produce zero fingerprint"


if __name__ == "__main__":
    """Run tests with pytest or as standalone script."""
    import sys

    # Run with pytest if available, otherwise run manually
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v", "-s"]))
    except ImportError:
        print("pytest not available, running tests manually...\n")

        # Create test instance
        test = TestSMILESPipeline()

        # Load fixtures
        mapping_path = "dataset/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
        with gzip.open(mapping_path, 'rt') as f:
            drugbank_mapping = pd.read_csv(f)

        smiles_path = "data/smiles.csv"
        smiles_data = pd.read_csv(smiles_path, keep_default_na=False)

        # Run tests
        tests = [
            ("Mapping file exists", lambda: test.test_mapping_file_exists(drugbank_mapping)),
            ("SMILES file exists", lambda: test.test_smiles_file_exists(smiles_data)),
            ("Node to DrugBank mapping", lambda: test.test_node_to_drugbank_mapping(drugbank_mapping)),
            ("DrugBank to SMILES mapping", lambda: test.test_drugbank_to_smiles_mapping(drugbank_mapping, smiles_data)),
            ("SMILES coverage", lambda: test.test_smiles_coverage(smiles_data)),
            ("SMILES to Morgan conversion", lambda: test.test_smiles_to_morgan_conversion(smiles_data)),
            ("Invalid SMILES returns zeros", lambda: test.test_invalid_smiles_returns_zeros()),
            ("Empty SMILES returns zeros", lambda: test.test_empty_smiles_returns_zeros()),
            ("Morgan fingerprint validity", lambda: test.test_morgan_fingerprint_validity(smiles_data)),
            ("Different molecules different fingerprints", lambda: test.test_different_molecules_different_fingerprints(smiles_data)),
            ("Build feature matrix", lambda: test.test_build_feature_matrix(smiles_data)),
            ("Complete pipeline example", lambda: test.test_specific_example_pipeline(drugbank_mapping, smiles_data)),
        ]

        passed = 0
        failed = 0

        for test_name, test_fn in tests:
            try:
                print(f"Testing: {test_name}...", end=" ")
                test_fn()
                print("✓ PASSED")
                passed += 1
            except Exception as e:
                print(f"✗ FAILED: {e}")
                failed += 1

        print(f"\n{'='*80}")
        print(f"Results: {passed} passed, {failed} failed")
        print(f"{'='*80}")

        sys.exit(0 if failed == 0 else 1)
