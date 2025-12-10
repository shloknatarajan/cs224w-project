from .gcn_structural_variations import (
    GCNStructural,
    GCNStructuralV2,
    GCNStructuralV3,
    GCNStructuralV4
)

from .seal import SEALGCN, SEALGIN
from .sage_variations import (
    SAGEVariant1Baseline,
    SAGEVariant2BatchNorm,
    SAGEVariant3Depth,
    SAGEVariant4Both
)
from .gdin import GDIN

__all__ = [
    'GCNStructural',
    'GCNStructuralV2',
    'GCNStructuralV3',
    'GCNStructuralV4',
    'SEALGCN',
    'SEALGIN',
    'SAGEVariant1Baseline',
    'SAGEVariant2BatchNorm',
    'SAGEVariant3Depth',
    'SAGEVariant4Both',
    'GDIN'
]
