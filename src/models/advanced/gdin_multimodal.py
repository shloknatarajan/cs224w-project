"""
GDIN with Multi-Modal External Features

Extends GDIN to incorporate external knowledge from multiple sources:
- Phase 1: Morgan fingerprints (molecular substructure)
- Phase 2: PubChem properties (physicochemical features)
- Phase 3: ChemBERTa embeddings (pre-trained molecular representations)
- Phase 4: Drug-target interactions (biological context)

Fusion strategies:
- 'concat': Simple concatenation of all features
- 'attention': Cross-modal attention fusion
- 'gated': Gated fusion with learnable weights
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

from ..base import BaseModel


class FeatureEncoder(nn.Module):
    """Encode external features to hidden dimension."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CrossModalAttention(nn.Module):
    """Cross-modal attention for fusing multiple feature sources."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [num_nodes, num_modalities, hidden_dim]

        Returns:
            [num_nodes, hidden_dim] fused representation
        """
        # Self-attention across modalities
        attended, _ = self.attention(features, features, features)
        attended = self.norm(attended + features)  # Residual

        # Mean pool across modalities
        return attended.mean(dim=1)


class GatedFusion(nn.Module):
    """Gated fusion with learnable weights per modality."""

    def __init__(self, hidden_dim: int, num_modalities: int):
        super().__init__()
        self.gates = nn.Linear(hidden_dim * num_modalities, num_modalities)
        self.num_modalities = num_modalities

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [num_nodes, num_modalities, hidden_dim]

        Returns:
            [num_nodes, hidden_dim] fused representation
        """
        # Flatten for gate computation
        flat = features.reshape(features.size(0), -1)

        # Compute gates
        gates = torch.softmax(self.gates(flat), dim=1)  # [N, num_modalities]

        # Weighted sum
        weighted = features * gates.unsqueeze(-1)
        return weighted.sum(dim=1)


class GDINMultiModal(BaseModel):
    """
    GDIN with multi-modal external feature support.

    Combines:
    - Structural pathway: GNN on drug-drug interaction graph
    - External pathways: Encoders for each external feature type

    Args:
        num_nodes: Number of drug nodes
        hidden_dim: Hidden dimension for all pathways
        num_layers: Number of GNN layers
        dropout: Dropout rate
        use_cn: Enable common neighbor features (from original GDIN)
        external_dims: Dict mapping feature name to input dimension
            e.g., {'morgan': 2048, 'pubchem': 9, 'chemberta': 768, 'drug_targets': 500}
        fusion: Fusion strategy ('concat', 'attention', 'gated')
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_cn: bool = False,
        external_dims: dict[str, int] | None = None,
        fusion: str = 'attention',
    ):
        # Determine decoder input dim based on fusion strategy
        decoder_hidden = hidden_dim  # After fusion, all strategies output hidden_dim

        super().__init__(decoder_hidden, decoder_dropout=0.0, use_multi_strategy=False)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_cn = use_cn
        self.fusion_type = fusion
        self.external_dims = external_dims or {}

        # Structural pathway: learnable embeddings + GNN
        self.struct_emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.struct_emb.weight)

        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # External feature encoders
        self.feature_encoders = nn.ModuleDict()
        for name, dim in self.external_dims.items():
            self.feature_encoders[name] = FeatureEncoder(
                input_dim=dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

        # Count total modalities (structural + external)
        self.num_modalities = 1 + len(self.external_dims)

        # Fusion layer
        if fusion == 'attention':
            self.fusion = CrossModalAttention(hidden_dim, num_heads=4, dropout=dropout)
        elif fusion == 'gated':
            self.fusion = GatedFusion(hidden_dim, self.num_modalities)
        elif fusion == 'concat':
            # Concat all and project back to hidden_dim
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * self.num_modalities, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")

        # Common neighbor weight (from original GDIN)
        if use_cn:
            self.cn_weight = nn.Parameter(torch.tensor(0.5))

        # Model description
        ext_str = ", ".join(f"{k}={v}" for k, v in self.external_dims.items())
        self.description = (
            f"GDINMultiModal ({num_layers} layers, {fusion} fusion) | "
            f"hidden_dim={hidden_dim}, external=[{ext_str}]"
        )

    def encode(
        self,
        edge_index: torch.Tensor,
        external_features: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Encode nodes using structural GNN and external features.

        Args:
            edge_index: Graph edge index [2, E]
            external_features: Dict mapping feature name to tensor [N, dim]

        Returns:
            Node embeddings [N, hidden_dim]
        """
        # Structural pathway
        x_struct = self.struct_emb.weight
        for conv in self.convs:
            x_struct = conv(x_struct, edge_index)
            x_struct = F.relu(x_struct)
            if self.dropout > 0 and self.training:
                x_struct = F.dropout(x_struct, p=self.dropout, training=True)

        # If no external features, return structural only
        if not external_features or not self.external_dims:
            return x_struct

        # Encode external features
        modality_features = [x_struct]  # Start with structural

        for name in self.external_dims.keys():
            if name in external_features and external_features[name] is not None:
                encoded = self.feature_encoders[name](external_features[name])
                modality_features.append(encoded)
            else:
                # Use zeros for missing modalities
                modality_features.append(torch.zeros_like(x_struct))

        # Fusion
        if self.fusion_type in ['attention', 'gated']:
            # Stack modalities: [N, num_modalities, hidden_dim]
            stacked = torch.stack(modality_features, dim=1)
            fused = self.fusion(stacked)
        else:  # concat
            # Concatenate and project
            concat = torch.cat(modality_features, dim=1)
            fused = self.fusion(concat)

        return fused

    def decode(
        self,
        z: torch.Tensor,
        edge: torch.Tensor,
        cn_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Decode edge scores from node embeddings.

        Args:
            z: Node embeddings [N, hidden_dim]
            edge: Edges to score [E, 2]
            cn_scores: Optional common neighbor scores [E]

        Returns:
            Edge scores [E]
        """
        # Base score from inherited decoder
        base_score = self.decoder(z, edge)

        # Add CN contribution if enabled
        if self.use_cn and cn_scores is not None:
            return base_score + self.cn_weight * cn_scores

        return base_score

    def forward(
        self,
        edge_index: torch.Tensor,
        edge: torch.Tensor,
        external_features: dict[str, torch.Tensor] | None = None,
        cn_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            edge_index: Graph structure [2, E_train]
            edge: Edges to score [E_query, 2]
            external_features: Dict of external feature tensors
            cn_scores: Optional common neighbor scores

        Returns:
            Edge scores [E_query]
        """
        z = self.encode(edge_index, external_features)
        return self.decode(z, edge, cn_scores)


# Factory function for easy model creation
def create_gdin_multimodal(
    num_nodes: int,
    feature_dims: dict[str, int],
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    fusion: str = 'attention',
    use_cn: bool = False,
) -> GDINMultiModal:
    """
    Create a GDINMultiModal model with the specified feature configuration.

    Args:
        num_nodes: Number of drug nodes
        feature_dims: Dict from load_external_features().feature_dims
        hidden_dim: Hidden dimension
        num_layers: Number of GNN layers
        dropout: Dropout rate
        fusion: 'attention', 'gated', or 'concat'
        use_cn: Enable common neighbor features

    Returns:
        Configured GDINMultiModal model
    """
    return GDINMultiModal(
        num_nodes=num_nodes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_cn=use_cn,
        external_dims=feature_dims,
        fusion=fusion,
    )
