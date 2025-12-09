"""
Hybrid GCN: Structure-only encoder + Chemistry-aware decoder

Key innovation: Keep the GNN encoder structure-only (like strong baselines),
but incorporate chemistry at the decoder level where it can be used intelligently.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from .decoder import ChemistryAwareDecoder, SimpleChemistryDecoder


class HybridGCN(nn.Module):
    """
    Hybrid GCN with structure-only encoder and chemistry-aware decoder.
    
    Architecture:
        1. Encoder: Standard GCN (no node features, structure-only like baselines)
        2. Decoder: Chemistry-aware decoder that combines topology + chemistry
    
    This design allows the model to:
        - Learn strong topological patterns (like baselines)
        - Incorporate chemistry when it's useful
        - Ignore chemistry when it's unreliable (missing SMILES)
    """
    
    def __init__(self,
                 num_nodes: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 chemical_dim: int = 768,
                 decoder_type: str = "chemistry_aware",
                 decoder_dropout: float = 0.3,
                 use_gating: bool = False):
        """
        Args:
            num_nodes: Number of nodes in the graph
            hidden_dim: Hidden dimension for GNN
            num_layers: Number of GCN layers
            dropout: Dropout for GCN layers (should be 0 per repo rules)
            chemical_dim: Dimension of chemical features (768 for ChemBERTa, 2048 for Morgan)
            decoder_type: "chemistry_aware" or "simple"
            decoder_dropout: Dropout for decoder MLPs
            use_gating: Use learned gates in decoder (only for chemistry_aware)
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.chemical_dim = chemical_dim
        self.decoder_type = decoder_type
        
        # Model description
        decoder_str = f"{decoder_type}" + ("+gate" if use_gating else "")
        self.description = (
            f"Hybrid-GCN ({num_layers} layers, dropout={dropout}) | "
            f"Structure-only encoder + {decoder_str} decoder | "
            f"hidden={hidden_dim}, chem_dim={chemical_dim}"
        )
        
        # Learnable node embeddings (structure-only, like baselines)
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        
        # GCN layers (structure-only)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))
        
        # Chemistry-aware decoder
        if decoder_type == "chemistry_aware":
            self.decoder = ChemistryAwareDecoder(
                structural_dim=hidden_dim,
                chemical_dim=chemical_dim,
                dropout=decoder_dropout,
                use_gating=use_gating
            )
        elif decoder_type == "simple":
            self.decoder = SimpleChemistryDecoder(
                structural_dim=hidden_dim,
                chemical_dim=chemical_dim,
                dropout=decoder_dropout
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")
    
    def encode(self, edge_index, x=None):
        """
        Encode nodes using structure-only GCN.
        
        Args:
            edge_index: Graph edge index [2, E]
            x: Ignored (we use learnable embeddings)
        
        Returns:
            z: Node embeddings [N, hidden_dim]
        """
        # Start with learnable node embeddings
        z = self.node_emb.weight  # [N, hidden_dim]
        
        # Apply GCN layers
        for conv in self.convs:
            z = conv(z, edge_index)
            z = F.relu(z)
            if self.dropout > 0:
                z = F.dropout(z, p=self.dropout, training=self.training)
        
        return z
    
    def decode(self, z, chemistry, edge, smiles_mask=None):
        """
        Decode edge scores using structural + chemical information.
        
        Args:
            z: Structural node embeddings [N, hidden_dim]
            chemistry: Chemical node features [N, chemical_dim]
            edge: Edge indices [E, 2]
            smiles_mask: Binary mask [N] for valid chemistry
        
        Returns:
            scores: Edge scores [E]
        """
        return self.decoder(z, chemistry, edge, smiles_mask)
    
    def forward(self, edge_index, chemistry, pos_edge, neg_edge, smiles_mask=None):
        """
        Forward pass for link prediction.
        
        Args:
            edge_index: Graph edge index [2, E]
            chemistry: Chemical features [N, chemical_dim]
            pos_edge: Positive edges [E_pos, 2]
            neg_edge: Negative edges [E_neg, 2]
            smiles_mask: Binary mask [N] for valid chemistry
        
        Returns:
            pos_pred: Predictions for positive edges [E_pos]
            neg_pred: Predictions for negative edges [E_neg]
        """
        # Encode nodes (structure-only)
        z = self.encode(edge_index)
        
        # Decode edges (structure + chemistry)
        pos_pred = self.decode(z, chemistry, pos_edge, smiles_mask)
        neg_pred = self.decode(z, chemistry, neg_edge, smiles_mask)
        
        return pos_pred, neg_pred

