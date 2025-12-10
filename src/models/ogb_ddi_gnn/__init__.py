"""OGB DDI reference GNN models."""

from .gnn import GCN, SAGE, LinkPredictor, train, test
from .gnn_external import (
    GCNExternal,
    SAGEExternal,
    LinkPredictor as LinkPredictorExternal,
    train_with_external,
    test_with_external,
)

__all__ = [
    # Original reference models
    'GCN',
    'SAGE',
    'LinkPredictor',
    'train',
    'test',
    # Models with external feature support
    'GCNExternal',
    'SAGEExternal',
    'LinkPredictorExternal',
    'train_with_external',
    'test_with_external',
]
