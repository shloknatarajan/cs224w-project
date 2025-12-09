"""
Hybrid GraphTransformer: Structure-only encoder + Chemistry-aware decoder
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import TransformerConv
from .decoder import ChemistryAwareDecoder, SimpleChemistryDecoder


class HybridGraphTransformer(nn.Module):
    """
    Hybrid GraphTransformer with structure-only encoder and chemistry-aware decoder.
    """
    
    def __init__(self,
                 num_nodes: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.0,
                 chemical_dim: int = 768,
                 decoder_type: str = "chemistry_aware",
                 decoder_dropout: float = 0.3,
                 use_gating: bool = False):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.chemical_dim = chemical_dim
        self.decoder_type = decoder_type
        
        # Model description
        decoder_str = f"{decoder_type}" + ("+gate" if use_gating else "")
        self.description = (
            f"Hybrid-GraphTransformer ({num_layers} layers, {num_heads} heads, dropout={dropout}) | "
            f"Structure-only encoder + {decoder_str} decoder | "
            f"hidden={hidden_dim}, chem_dim={chemical_dim}"
        )
        
        # Learnable node embeddings
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        
        # GraphTransformer layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
        
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
        """Encode nodes using structure-only GraphTransformer."""
        z = self.node_emb.weight
        
        for conv in self.convs:
            z = conv(z, edge_index)
            z = F.relu(z)
            if self.dropout > 0:
                z = F.dropout(z, p=self.dropout, training=self.training)
        
        return z
    
    def decode(self, z, chemistry, edge, smiles_mask=None):
        """Decode edge scores using structural + chemical information."""
        return self.decoder(z, chemistry, edge, smiles_mask)
    
    def forward(self, edge_index, chemistry, pos_edge, neg_edge, smiles_mask=None):
        """Forward pass for link prediction."""
        z = self.encode(edge_index)
        pos_pred = self.decode(z, chemistry, pos_edge, smiles_mask)
        neg_pred = self.decode(z, chemistry, neg_edge, smiles_mask)
        return pos_pred, neg_pred

