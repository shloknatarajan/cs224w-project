#!/bin/bash
# Convenience script to run tests with proper PYTHONPATH

cd "$(dirname "$0")/.."
export PYTHONPATH=/home/ubuntu/cs224w-project

echo "Running SMILES pipeline smoke tests..."
python tests/test_smiles_pipeline.py "$@"
