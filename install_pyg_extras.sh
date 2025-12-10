#!/bin/bash
# Install PyTorch Geometric optional dependencies (torch-sparse, torch-scatter, torch-cluster)
# These are required for SparseTensor support in PyG

set -e

echo "Detecting PyTorch and CUDA versions..."
TORCH_VERSION=$(pixi run python -c "import torch; print(torch.__version__)")
CUDA_VERSION=$(pixi run python -c "import torch; print(torch.version.cuda if torch.version.cuda else 'cpu')")

echo "PyTorch: $TORCH_VERSION"
echo "CUDA: $CUDA_VERSION"

# Determine the wheel URL based on PyTorch and CUDA versions
if [ "$CUDA_VERSION" == "cpu" ]; then
    WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}.html"
else
    # Convert CUDA version (e.g., "12.1" -> "cu121")
    CUDA_SHORT=$(echo $CUDA_VERSION | sed 's/\.//g')
    WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_SHORT}.html"
fi

echo "Installing PyG extras from: $WHEEL_URL"
pixi run pip install torch-sparse torch-scatter torch-cluster -f $WHEEL_URL

echo "Verifying installation..."
pixi run python -c "import torch_sparse; from torch_sparse import SparseTensor; print('✓ torch-sparse installed successfully')"

echo "Done!"

