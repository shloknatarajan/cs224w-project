# Tests

This directory contains smoke tests for the Morgan fingerprint generation functionality and the complete SMILES data pipeline.

## Test Files

### `test_smiles_pipeline.py` ⭐ (NEW)

**Complete end-to-end pipeline validation for ogbl-ddi with real DrugBank data:**

Tests the entire data flow:
1. OGB node ID → DrugBank ID mapping
2. DrugBank ID → SMILES string lookup
3. SMILES → Morgan fingerprint conversion
4. Fingerprint validation and properties

**Key Tests:**
- ✓ Validates node-to-DrugBank ID mapping (4267 nodes)
- ✓ Verifies SMILES data coverage (should be ~53.6%)
- ✓ Tests SMILES-to-Morgan conversion with real molecules
- ✓ Ensures different molecules produce different fingerprints
- ✓ Validates fingerprint properties (binary, sparse, 2048-dim)
- ✓ Tests edge cases (empty SMILES, invalid SMILES)
- ✓ Complete example walkthrough with specific nodes

**Running:**
```bash
# Quick way
./tests/run_tests.sh

# Or with pytest directly
PYTHONPATH=/home/ubuntu/cs224w-project python tests/test_smiles_pipeline.py

# Or with pytest verbose
PYTHONPATH=/home/ubuntu/cs224w-project pytest tests/test_smiles_pipeline.py -v -s
```

**Expected Results:**
```
✓ 12 tests passed
✓ SMILES Coverage: 2287/4267 (53.6%)
✓ Node 4 → DB00006 → Valid SMILES → 107 bits set
✓ All fingerprints are binary and valid
```

### `test_morgan_standalone.py`

Comprehensive smoke tests for SMILES/Morgan fingerprint generation:

- ✓ Tests `smiles_to_morgan()` function with valid/invalid SMILES
- ✓ Tests `build_smiles_feature_matrix()` with CSV input
- ✓ Validates feature properties (shape, dtype, sparsity, binary values)
- ✓ Tests edge cases (invalid SMILES, empty strings, missing nodes)
- ✓ Tests consistency across multiple calls
- ✓ Tests common drug molecules (Aspirin, Caffeine, Ibuprofen, etc.)
- ✓ Calculates Tanimoto similarities between molecules

### Running the Tests

```bash
python tests/test_morgan_standalone.py
```

Expected output: All tests pass with detailed logging showing:
- Fingerprint dimensions and sparsity
- Bits set for each molecule
- Tanimoto similarities between molecules
- Validation of edge cases

### Test Results Summary

From the last run:
- ✓ Aspirin: 24/2048 bits set (1.2% density, 98.8% sparse)
- ✓ Caffeine: 25/2048 bits set (1.2% density)
- ✓ All 8 drug molecules have unique fingerprints
- ✓ Invalid/empty SMILES correctly return zero vectors
- ✓ Features are binary (0 or 1) and consistent

### What Gets Tested

1. **Basic Functionality**
   - SMILES → Morgan fingerprint conversion
   - Different fingerprint sizes (512, 1024, 2048, 4096 bits)
   - Different radius values (1, 2, 3, 4)

2. **Edge Cases**
   - Invalid SMILES strings
   - Empty SMILES strings
   - Missing nodes in CSV
   - Different molecular sizes (methane to long chains)

3. **Feature Properties**
   - Correct shape and dtype
   - Binary values (0 or 1)
   - Sparsity (~98-99% sparse is normal)
   - Consistency across multiple calls

4. **Real-World Molecules**
   - Common drugs (Aspirin, Caffeine, Ibuprofen, etc.)
   - Neurotransmitters (Dopamine, Serotonin)
   - Small molecules (Glucose, Methane)

5. **Similarity Metrics**
   - Tanimoto similarity calculations
   - Similar drugs (NSAIDs) have higher similarity
   - Dissimilar molecules have low similarity

## Notes

- The tests may show NumPy compatibility warnings with RDKit - these are harmless
- Morgan fingerprints are sparse (~98-99% zeros) - this is expected
- Tanimoto similarity ranges from 0 (completely different) to 1 (identical)
- All drug molecules should have unique fingerprints
