"""
Advanced GNN models for drug-drug interaction prediction.

This module contains the two main model architectures:
- GCNAdvanced: Structure-only GCN achieving 61.69% test Hits@20
- GDINN: Full model with external features achieving 73.28% test Hits@20
"""
from .gcn_advanced import GCNAdvanced, LinkPredictor, train, test
from .gdinn import GDINN, FeatureEncoder, train_with_external, test_with_external

__all__ = [
    # Models
    "GCNAdvanced",
    "GDINN",
    # Decoders and encoders
    "LinkPredictor",
    "FeatureEncoder",
    # Training functions
    "train",
    "test",
    "train_with_external",
    "test_with_external",
]
