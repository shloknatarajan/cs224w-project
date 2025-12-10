"""
Standalone smoke tests for SMILES/Morgan fingerprint generation.
Tests only the Morgan fingerprint functions without heavy dependencies.
"""
import torch
import tempfile
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def smiles_to_morgan(smiles: str, n_bits: int = 2048, radius: int = 2) -> torch.Tensor:
    """
    Convert a SMILES string to a Morgan/ECFP fingerprint vector [n_bits].
    Returns an all-zero vector if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(n_bits, dtype=torch.float32)

    bitvect = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits
    )
    arr = torch.zeros(n_bits, dtype=torch.float32)
    on_bits = list(bitvect.GetOnBits())
    arr[on_bits] = 1.0
    return arr


def build_smiles_feature_matrix(
    num_nodes: int,
    smiles_csv_path: str,
    n_bits: int = 2048,
    radius: int = 2,
) -> torch.Tensor:
    """
    Build a [num_nodes, n_bits] feature matrix from a CSV with columns:
      - ogb_id  (int: 0..num_nodes-1)
      - smiles  (str)
    Nodes without a SMILES entry (or invalid SMILES) get all-zero features.
    """
    df = pd.read_csv(smiles_csv_path)
    if "ogb_id" not in df.columns or "smiles" not in df.columns:
        raise ValueError(
            f"Expected columns 'ogb_id' and 'smiles' in {smiles_csv_path}, "
            f"found {df.columns.tolist()}"
        )

    smiles_by_id = dict(
        zip(df["ogb_id"].astype(int), df["smiles"].astype(str))
    )

    feat = torch.zeros((num_nodes, n_bits), dtype=torch.float32)
    missing = 0

    for node_id in range(num_nodes):
        smi = smiles_by_id.get(node_id, "")
        if not smi or not isinstance(smi, str):
            missing += 1
            continue
        feat[node_id] = smiles_to_morgan(smi, n_bits=n_bits, radius=radius)

    logger.info(
        f"Built SMILES feature matrix: shape={feat.shape}, "
        f"missing/invalid SMILES for {missing}/{num_nodes} nodes"
    )
    return feat


def test_smiles_to_morgan():
    """Test basic SMILES to Morgan fingerprint conversion"""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: smiles_to_morgan() basic functionality")
    logger.info("="*80)

    # Test 1: Valid SMILES - Aspirin
    logger.info("\nTest 1a: Valid SMILES (Aspirin)")
    aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
    fp = smiles_to_morgan(aspirin, n_bits=2048, radius=2)

    assert fp.shape == (2048,), f"Expected shape (2048,), got {fp.shape}"
    assert fp.dtype == torch.float32, f"Expected dtype float32, got {fp.dtype}"
    assert torch.all((fp == 0) | (fp == 1)), "Fingerprint should be binary (0 or 1)"

    num_bits_on = torch.sum(fp).item()
    sparsity = 1 - (num_bits_on / 2048)
    logger.info(f"✓ Aspirin fingerprint shape: {fp.shape}")
    logger.info(f"✓ Bits set: {num_bits_on}/2048 ({num_bits_on/2048*100:.1f}%)")
    logger.info(f"✓ Sparsity: {sparsity*100:.1f}%")

    # Test 2: Another valid SMILES - Caffeine
    logger.info("\nTest 1b: Valid SMILES (Caffeine)")
    caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    fp2 = smiles_to_morgan(caffeine, n_bits=2048, radius=2)

    assert fp2.shape == (2048,), f"Expected shape (2048,), got {fp2.shape}"
    num_bits_on2 = torch.sum(fp2).item()
    logger.info(f"✓ Caffeine fingerprint shape: {fp2.shape}")
    logger.info(f"✓ Bits set: {num_bits_on2}/2048 ({num_bits_on2/2048*100:.1f}%)")

    # Test 3: Different molecules should have different fingerprints
    logger.info("\nTest 1c: Different molecules have different fingerprints")
    assert not torch.equal(fp, fp2), "Different molecules should have different fingerprints"
    # Calculate Tanimoto similarity
    intersection = torch.sum(fp * fp2).item()
    union = torch.sum((fp + fp2) > 0).item()
    similarity = intersection / union if union > 0 else 0.0
    logger.info(f"✓ Tanimoto similarity between Aspirin and Caffeine: {similarity:.3f}")

    # Test 4: Invalid SMILES
    logger.info("\nTest 1d: Invalid SMILES returns zero vector")
    invalid_smiles = "INVALID_SMILES_XYZ123"
    fp_invalid = smiles_to_morgan(invalid_smiles, n_bits=2048, radius=2)

    assert fp_invalid.shape == (2048,), f"Expected shape (2048,), got {fp_invalid.shape}"
    assert torch.sum(fp_invalid).item() == 0, "Invalid SMILES should return all-zero vector"
    logger.info(f"✓ Invalid SMILES returns zero vector")

    # Test 5: Empty SMILES
    logger.info("\nTest 1e: Empty SMILES returns zero vector")
    fp_empty = smiles_to_morgan("", n_bits=2048, radius=2)
    assert torch.sum(fp_empty).item() == 0, "Empty SMILES should return all-zero vector"
    logger.info(f"✓ Empty SMILES returns zero vector")

    # Test 6: Different fingerprint sizes
    logger.info("\nTest 1f: Different fingerprint sizes")
    for n_bits in [512, 1024, 2048, 4096]:
        fp_size = smiles_to_morgan(aspirin, n_bits=n_bits, radius=2)
        assert fp_size.shape == (n_bits,), f"Expected shape ({n_bits},), got {fp_size.shape}"
        logger.info(f"✓ n_bits={n_bits} works, bits set: {torch.sum(fp_size).item()}/{n_bits}")

    # Test 7: Different radius values
    logger.info("\nTest 1g: Different radius values")
    for radius in [1, 2, 3, 4]:
        fp_radius = smiles_to_morgan(aspirin, n_bits=2048, radius=radius)
        num_bits = torch.sum(fp_radius).item()
        logger.info(f"✓ radius={radius} works, bits set: {num_bits}/2048")

    logger.info("\n✓ All smiles_to_morgan() tests passed!")
    return True


def test_build_smiles_feature_matrix():
    """Test building feature matrix from CSV file"""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: build_smiles_feature_matrix()")
    logger.info("="*80)

    # Create temporary CSV file with test data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_csv = f.name
        # Write test data: 10 nodes with various SMILES
        f.write("ogb_id,smiles\n")
        f.write("0,CC(=O)OC1=CC=CC=C1C(=O)O\n")  # Aspirin
        f.write("1,CN1C=NC2=C1C(=O)N(C(=O)N2C)C\n")  # Caffeine
        f.write("2,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O\n")  # Ibuprofen
        f.write("3,CC(C)NCC(COC1=CC=CC=C1)O\n")  # Propranolol
        f.write("4,INVALID_SMILES\n")  # Invalid
        f.write("5,\n")  # Empty
        f.write("6,C\n")  # Methane (simplest)
        f.write("7,CCCCCCCCCCCCCCCC\n")  # Long chain
        # Node 8 missing (no entry)
        f.write("9,CC(C)C\n")  # Isobutane

    try:
        # Test 1: Basic functionality
        logger.info("\nTest 2a: Basic feature matrix construction")
        num_nodes = 10
        feat_matrix = build_smiles_feature_matrix(
            num_nodes=num_nodes,
            smiles_csv_path=temp_csv,
            n_bits=2048,
            radius=2
        )

        assert feat_matrix.shape == (10, 2048), f"Expected shape (10, 2048), got {feat_matrix.shape}"
        assert feat_matrix.dtype == torch.float32, f"Expected dtype float32, got {feat_matrix.dtype}"
        logger.info(f"✓ Feature matrix shape: {feat_matrix.shape}")
        logger.info(f"✓ Feature matrix dtype: {feat_matrix.dtype}")

        # Test 2: Check valid SMILES have non-zero features
        logger.info("\nTest 2b: Valid SMILES have non-zero features")
        valid_nodes = [0, 1, 2, 3, 6, 7, 9]
        for node_id in valid_nodes:
            num_bits = torch.sum(feat_matrix[node_id]).item()
            assert num_bits > 0, f"Node {node_id} should have non-zero features"
            logger.info(f"✓ Node {node_id}: {num_bits}/2048 bits set ({num_bits/2048*100:.1f}%)")

        # Test 3: Check invalid/missing SMILES have zero features
        logger.info("\nTest 2c: Invalid/missing SMILES have zero features")
        invalid_nodes = [4, 5, 8]  # Invalid, empty, missing
        for node_id in invalid_nodes:
            num_bits = torch.sum(feat_matrix[node_id]).item()
            assert num_bits == 0, f"Node {node_id} should have zero features"
            logger.info(f"✓ Node {node_id}: all zeros (as expected)")

        # Test 4: Check features are binary
        logger.info("\nTest 2d: Features are binary")
        assert torch.all((feat_matrix == 0) | (feat_matrix == 1)), "All features should be 0 or 1"
        logger.info(f"✓ All features are binary (0 or 1)")

        # Test 5: Overall statistics
        logger.info("\nTest 2e: Overall statistics")
        total_bits_set = torch.sum(feat_matrix).item()
        total_bits = feat_matrix.numel()
        overall_sparsity = 1 - (total_bits_set / total_bits)
        logger.info(f"✓ Total bits set: {int(total_bits_set)}/{total_bits} ({total_bits_set/total_bits*100:.1f}%)")
        logger.info(f"✓ Overall sparsity: {overall_sparsity*100:.1f}%")

        # Test 6: Different fingerprint sizes
        logger.info("\nTest 2f: Different fingerprint sizes work")
        for n_bits in [512, 1024]:
            feat = build_smiles_feature_matrix(num_nodes, temp_csv, n_bits=n_bits, radius=2)
            assert feat.shape == (num_nodes, n_bits), f"Expected shape ({num_nodes}, {n_bits}), got {feat.shape}"
            logger.info(f"✓ n_bits={n_bits} works: shape {feat.shape}")

        logger.info("\n✓ All build_smiles_feature_matrix() tests passed!")
        return True

    finally:
        # Clean up temp file
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
            logger.info(f"\n✓ Cleaned up temporary file")


def test_feature_consistency():
    """Test that features are consistent across multiple calls"""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Feature consistency")
    logger.info("="*80)

    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin

    # Generate fingerprint multiple times
    logger.info("\nTest 3a: Same SMILES produces same fingerprint")
    fps = [smiles_to_morgan(smiles, n_bits=2048, radius=2) for _ in range(5)]

    for i in range(1, len(fps)):
        assert torch.equal(fps[0], fps[i]), f"Fingerprint {i} differs from fingerprint 0"

    logger.info(f"✓ Generated fingerprint 5 times - all identical")
    logger.info(f"✓ Bits set: {torch.sum(fps[0]).item()}/2048")

    logger.info("\n✓ All consistency tests passed!")
    return True


def test_common_molecules():
    """Test fingerprints for common drug molecules"""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Common drug molecules")
    logger.info("="*80)

    common_drugs = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
        "Nicotine": "CN1CCCC1C2=CN=CC=C2",
        "Dopamine": "C1=CC(=C(C=C1CCN)O)O",
        "Serotonin": "C1=CC2=C(C=C1O)C(=CN2)CCN",
        "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
    }

    logger.info(f"\nTesting {len(common_drugs)} common drug molecules:\n")

    fingerprints = {}
    for name, smiles in common_drugs.items():
        fp = smiles_to_morgan(smiles, n_bits=2048, radius=2)
        bits_set = torch.sum(fp).item()
        fingerprints[name] = fp

        assert fp.shape == (2048,), f"{name}: Expected shape (2048,), got {fp.shape}"
        assert bits_set > 0, f"{name}: Should have non-zero fingerprint"

        logger.info(f"✓ {name:15s}: {int(bits_set):4d}/2048 bits ({bits_set/2048*100:5.1f}%)")

    # Check all are unique
    logger.info(f"\nTest 4b: All drug fingerprints are unique")
    names = list(fingerprints.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            fp1, fp2 = fingerprints[names[i]], fingerprints[names[j]]
            assert not torch.equal(fp1, fp2), f"{names[i]} and {names[j]} have identical fingerprints"

    logger.info(f"✓ All {len(names)} drug fingerprints are unique")

    # Calculate some similarities
    logger.info(f"\nTest 4c: Sample Tanimoto similarities")
    pairs = [
        ("Aspirin", "Ibuprofen"),  # Both NSAIDs
        ("Dopamine", "Serotonin"),  # Both neurotransmitters
        ("Aspirin", "Glucose"),  # Very different
    ]

    for name1, name2 in pairs:
        fp1, fp2 = fingerprints[name1], fingerprints[name2]
        intersection = torch.sum(fp1 * fp2).item()
        union = torch.sum((fp1 + fp2) > 0).item()
        tanimoto = intersection / union if union > 0 else 0.0
        logger.info(f"  {name1:15s} vs {name2:15s}: Tanimoto = {tanimoto:.3f}")

    logger.info("\n✓ All common molecule tests passed!")
    return True


def main():
    """Run all smoke tests"""
    logger.info("\n" + "="*80)
    logger.info("MORGAN FINGERPRINT SMOKE TESTS (Standalone)")
    logger.info("="*80)

    try:
        # Run all tests
        test_smiles_to_morgan()
        test_build_smiles_feature_matrix()
        test_feature_consistency()
        test_common_molecules()

        # Summary
        logger.info("\n" + "="*80)
        logger.info("ALL TESTS PASSED! ✓")
        logger.info("="*80)
        logger.info("\nSummary:")
        logger.info("  ✓ smiles_to_morgan() works correctly")
        logger.info("  ✓ Handles invalid/empty SMILES gracefully")
        logger.info("  ✓ build_smiles_feature_matrix() works correctly")
        logger.info("  ✓ Features are binary, sparse, and consistent")
        logger.info("  ✓ Different molecules have different fingerprints")
        logger.info("  ✓ Common drug molecules processed successfully")
        logger.info("\nYour Morgan fingerprint generation is working correctly!")
        logger.info("Ready for production use with your models.")

        return True

    except AssertionError as e:
        logger.error(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        logger.error(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
