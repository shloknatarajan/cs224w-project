"""
ChemBERTa-based GraphSAGE with learnable fallback embeddings for missing SMILES
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from ..base import BaseModel


class ChemBERTaGraphSAGEFallback(BaseModel):
    """
    GraphSAGE with ChemBERTa embeddings and learnable fallback for missing SMILES.
    """

    def __init__(self, num_nodes, in_channels=768, hidden_dim=128, num_layers=2,
                 dropout=0.0, decoder_dropout=0.0, use_multi_strategy=False,
                 use_batch_norm=False, use_id_embeddings=False, id_dim=64,
                 use_structural_fallback=True, num_structural_features=6):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout,
                         use_multi_strategy=use_multi_strategy)

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.use_id_embeddings = use_id_embeddings
        self.id_dim = id_dim
        self.use_structural_fallback = use_structural_fallback
        self.num_structural_features = num_structural_features

        # Model description
        decoder_type = "multi-strategy" if use_multi_strategy else "simple"
        bn_str = "BN" if use_batch_norm else "no-BN"
        id_str = f"+ID({id_dim})" if use_id_embeddings else ""
        struct_str = "+struct" if use_structural_fallback else ""
        self.description = (
            f"ChemBERTa-GraphSAGE-Fallback ({num_layers} layers, dropout={dropout}, {bn_str}{id_str}{struct_str}) | "
            f"in={in_channels}, hidden={hidden_dim}, "
            f"decoder={decoder_type}"
        )

        # ChemBERTa projection
        self.chem_proj = nn.Linear(in_channels, hidden_dim)

        # Structural projection (always available for fusion if enabled)
        self.struct_proj = nn.Linear(num_structural_features, hidden_dim) if use_structural_fallback else None

        # Fuse ChemBERTa/fallback with structural features
        self.fuse_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Learnable fallback embedding
        self.no_chem_emb = nn.Parameter(torch.randn(1, hidden_dim))

        # Optional node ID embeddings (added after fusion)
        if use_id_embeddings:
            self.id_emb = nn.Embedding(num_nodes, id_dim)
            gnn_input_dim = hidden_dim + id_dim
        else:
            self.id_emb = None
            gnn_input_dim = hidden_dim

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        # First layer
        self.convs.append(SAGEConv(gnn_input_dim, hidden_dim))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def encode(self, edge_index, x=None, smiles_mask=None, structural_features=None):
        """
        Encode nodes to embeddings using ChemBERTa + fallback.

        Args:
            edge_index: Graph edge index
            x: Node features (ChemBERTa embeddings), shape [num_nodes, in_channels]
            smiles_mask: Binary mask indicating valid SMILES, shape [num_nodes]
            structural_features: Optional structural features, shape [num_nodes, num_structural_features]

        Returns:
            Node embeddings of shape [num_nodes, hidden_dim]
        """
        if x is None:
            raise ValueError("ChemBERTaGraphSAGEFallback requires input features x")

        num_nodes = x.size(0)
        device = x.device

        # Project ChemBERTa embeddings
        chem_emb = self.chem_proj(x)

        # Expand fallback embedding
        fallback = self.no_chem_emb.expand(num_nodes, -1)

        # Structural branch
        if self.use_structural_fallback and structural_features is not None and self.struct_proj is not None:
            struct_feats = self.struct_proj(structural_features)
        else:
            struct_feats = torch.zeros_like(chem_emb)

        # Select based on mask
        if smiles_mask is not None:
            mask = smiles_mask.unsqueeze(-1)
            chem_used = mask * chem_emb + (1 - mask) * fallback
        else:
            chem_used = chem_emb

        chem_used = F.relu(chem_used)

        # Concatenate ChemBERTa/fallback with structural features, then project
        fused = torch.cat([chem_used, struct_feats], dim=-1)
        node_features = F.relu(self.fuse_proj(fused))

        # Optionally add node ID embeddings
        if self.use_id_embeddings:
            id_emb = self.id_emb(torch.arange(num_nodes, device=device))
            node_features = torch.cat([node_features, id_emb], dim=-1)

        # Apply GraphSAGE layers
        for i, conv in enumerate(self.convs):
            node_features = conv(node_features, edge_index)

            # Apply batch normalization if enabled
            if self.use_batch_norm:
                node_features = self.batch_norms[i](node_features)

            node_features = F.relu(node_features)

            if self.dropout > 0:
                node_features = F.dropout(node_features, p=self.dropout, training=self.training)

        return node_features
