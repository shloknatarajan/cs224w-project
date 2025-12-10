"""
Heuristic-Enhanced GCN (Simplified Leaderboard Approach)

Key features from leaderboard:
1. Node embeddings + degree embeddings
2. Graph heuristics (CN, Jaccard, AA, RA) as edge features
3. Standard GCN layers (simplified from ComHG)
4. Decoder concatenates node embeddings with edge heuristics

Simplifications from full leaderboard:
- Uses standard GCNConv instead of custom ComHG layers
- No distance heuristic (requires expensive shortest path computation)
- Simpler attention mechanism
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class HeuristicGCN(nn.Module):
    """
    GCN with graph heuristics (simplified leaderboard approach).

    Architecture:
        1. Input: Node embeddings + degree embeddings
        2. Encoder: Standard GCN layers
        3. Decoder: MLP([z_i * z_j, heuristics(i,j)])

    This is simpler than the leaderboard but captures the key idea:
    combining learned representations with graph-based heuristics.
    """

    def __init__(self,
                 num_nodes: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 decoder_dropout: float = 0.3,
                 heuristic_dim: int = 4,  # CN, Jaccard, AA, RA
                 use_degree_emb: bool = True,
                 degree_emb_dim: int = 16):
        """
        Args:
            num_nodes: Number of nodes in graph
            hidden_dim: Hidden dimension for GNN
            num_layers: Number of GCN layers
            dropout: Dropout for GCN layers
            decoder_dropout: Dropout for decoder
            heuristic_dim: Dimension of heuristic features (4 by default)
            use_degree_emb: Whether to use degree embeddings
            degree_emb_dim: Dimension of degree embeddings
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.heuristic_dim = heuristic_dim
        self.use_degree_emb = use_degree_emb
        self.degree_emb_dim = degree_emb_dim

        # Model description
        self.description = (
            f"Heuristic-GCN (leaderboard-inspired) | "
            f"{num_layers} layers, hidden={hidden_dim}, "
            f"degree_emb={use_degree_emb}, heuristics=[CN,Jaccard,AA,RA]"
        )

        # Node embeddings (learnable)
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # Degree embeddings (optional)
        if use_degree_emb:
            # We'll bin degrees into 100 buckets
            self.degree_emb = nn.Embedding(100, degree_emb_dim)
            nn.init.xavier_uniform_(self.degree_emb.weight)

            # Project combined embeddings to hidden_dim
            self.input_proj = nn.Linear(hidden_dim + degree_emb_dim, hidden_dim)

        # GCN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))

        # Heuristic embeddings (LEADERBOARD APPROACH)
        # Each heuristic type gets its own embedding table
        # This allows the model to learn different representations for different values
        self.heuristic_emb_dim = 16  # Embedding dimension per heuristic
        self.max_heuristic_bins = 100  # Number of bins for discretization
        
        self.heuristic_embs = nn.ModuleList([
            nn.Embedding(self.max_heuristic_bins, self.heuristic_emb_dim)
            for _ in range(heuristic_dim)  # One embedding per heuristic (CN, Jaccard, AA, RA)
        ])
        
        # Initialize embeddings
        for emb in self.heuristic_embs:
            nn.init.xavier_uniform_(emb.weight)

        # Decoder: takes [z_i * z_j, heuristic_embeddings] and predicts edge score
        # Input size: hidden_dim (Hadamard) + (heuristic_dim * heuristic_emb_dim)
        decoder_input_dim = hidden_dim + (heuristic_dim * self.heuristic_emb_dim)
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(decoder_dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(decoder_dropout),
            nn.Linear(64, 1)
        )

    def encode(self, edge_index, degrees=None):
        """
        Encode nodes using GCN with optional degree embeddings.

        Args:
            edge_index: Graph edges [2, E]
            degrees: Node degrees [N] (optional, for degree embeddings)

        Returns:
            z: Node embeddings [N, hidden_dim]
        """
        # Start with node embeddings
        z = self.node_emb.weight  # [N, hidden_dim]

        # Add degree embeddings if enabled
        if self.use_degree_emb and degrees is not None:
            # Bin degrees (log-scale for better distribution)
            degree_bins = torch.log1p(degrees).long().clamp(0, 99)  # [N]
            deg_emb = self.degree_emb(degree_bins)  # [N, degree_emb_dim]

            # Concatenate and project
            z = torch.cat([z, deg_emb], dim=-1)  # [N, hidden_dim + degree_emb_dim]
            z = self.input_proj(z)  # [N, hidden_dim]

        # Apply GCN layers
        for conv in self.convs:
            z = conv(z, edge_index)
            z = F.relu(z)
            if self.dropout > 0:
                z = F.dropout(z, p=self.dropout, training=self.training)

        return z

    def decode(self, z, edge, heuristics):
        """
        Decode edge scores using structural embeddings + heuristics.
        
        LEADERBOARD APPROACH: Uses embedding lookup for discretized heuristics.

        Args:
            z: Node embeddings [N, hidden_dim]
            edge: Edge indices [E, 2]
            heuristics: DISCRETIZED edge heuristics [E, heuristic_dim] as LongTensor

        Returns:
            scores: Edge scores [E]
        """
        src, dst = edge[:, 0], edge[:, 1]

        # Structural representation (Hadamard product)
        z_src, z_dst = z[src], z[dst]
        structural = z_src * z_dst  # [E, hidden_dim]

        # Embed heuristics (LEADERBOARD APPROACH)
        # Each heuristic value is looked up in its own embedding table
        heuristic_embeddings = []
        for i in range(self.heuristic_dim):
            # heuristics[:, i] contains bin indices for heuristic type i
            emb = self.heuristic_embs[i](heuristics[:, i])  # [E, heuristic_emb_dim]
            heuristic_embeddings.append(emb)
        
        # Concatenate all heuristic embeddings
        heuristic_features = torch.cat(heuristic_embeddings, dim=-1)  # [E, heuristic_dim * heuristic_emb_dim]

        # Concatenate structural + heuristics
        combined = torch.cat([structural, heuristic_features], dim=-1)  # [E, hidden_dim + total_heuristic_dims]

        # MLP decoder
        scores = self.decoder(combined).squeeze(-1)  # [E]

        return scores

    def forward(self, edge_index, pos_edge, neg_edge, pos_heuristics, neg_heuristics, degrees=None):
        """
        Forward pass for link prediction.

        Args:
            edge_index: Training graph edges [2, E]
            pos_edge: Positive edges [E_pos, 2]
            neg_edge: Negative edges [E_neg, 2]
            pos_heuristics: Heuristics for positive edges [E_pos, heuristic_dim]
            neg_heuristics: Heuristics for negative edges [E_neg, heuristic_dim]
            degrees: Node degrees [N] (optional)

        Returns:
            pos_pred: Predictions for positive edges [E_pos]
            neg_pred: Predictions for negative edges [E_neg]
        """
        # Encode nodes
        z = self.encode(edge_index, degrees)

        # Decode edges with heuristics
        pos_pred = self.decode(z, pos_edge, pos_heuristics)
        neg_pred = self.decode(z, neg_edge, neg_heuristics)

        return pos_pred, neg_pred
