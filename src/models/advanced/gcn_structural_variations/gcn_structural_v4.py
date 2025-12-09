import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge
from ...base import BaseModel


class GCNStructuralV4(BaseModel):
    """
    Advanced GCN model V4 with multi-strategy decoder.

    Key Improvements from V3:
    - **Multi-strategy decoder enabled** (was disabled in V3) - combines Hadamard, Concat, and Bilinear scoring
    - **Reduced diversity loss** (0.02 default vs 0.05) - we achieved good diversity, can lower weight
    - **Kept from V3**: 3 layers, LeakyReLU, full residuals, LayerNorm

    Architecture Details:
    - 3 GCN layers with residual connections at ALL layers
    - LeakyReLU activation (prevents dead neurons)
    - Layer normalization for stable training
    - Structural features (degree, clustering, core number, PageRank, neighbor stats)
    - Edge dropout for regularization
    - Embedding diversity regularization
    - Multi-strategy edge decoder (3 strategies combined with learnable weights)

    The multi-strategy decoder is critical for ogbl-ddi's dense graph structure:
    - Hadamard: captures feature interactions
    - Concatenation: preserves all information
    - Bilinear: learns relation-specific patterns
    """

    def __init__(self, num_nodes, hidden_dim=256, num_layers=3, dropout=0.2, decoder_dropout=0.3,
                 use_structural_features=True, num_structural_features=6, structural_features=None,
                 use_multi_strategy=True, diversity_weight=0.02):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_structural_features = use_structural_features
        self.num_structural_features = num_structural_features
        self.structural_features = structural_features
        self.diversity_weight = diversity_weight

        # Model description with hyperparameters
        decoder_type = "multi-strategy" if use_multi_strategy else "simple"
        struct_status = f"with {num_structural_features} structural features" if use_structural_features else "without structural features"
        self.description = (
            f"GCN-V4 {struct_status}, multi-strategy decoder, layer norm, LeakyReLU, full residuals | "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}, "
            f"decoder_dropout={decoder_dropout}, decoder={decoder_type}, diversity_weight={diversity_weight}"
        )

        # Learnable embeddings with improved initialization
        emb_dim = hidden_dim - num_structural_features if use_structural_features else hidden_dim
        self.emb = nn.Embedding(num_nodes, emb_dim)
        # Xavier uniform initialization for better gradient flow
        nn.init.xavier_uniform_(self.emb.weight)

        # Feature projection layer to transform structural features
        if use_structural_features:
            self.feature_proj = nn.Sequential(
                nn.Linear(num_structural_features, num_structural_features),
                nn.LayerNorm(num_structural_features),
                nn.LeakyReLU(0.2),  # LeakyReLU instead of ReLU
                nn.Dropout(dropout * 0.5)
            )

        # Multiple GCN layers with LayerNorm
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))
            self.lns.append(nn.LayerNorm(hidden_dim))

    def encode(self, edge_index):
        x = self.emb.weight

        # Concatenate structural features with learnable embeddings
        if self.use_structural_features and self.structural_features is not None:
            struct_feats = self.feature_proj(self.structural_features)
            if self.training:
                struct_feats = F.dropout(struct_feats, p=0.1, training=True)
            x = torch.cat([x, struct_feats], dim=1)

        # Moderate edge dropout for regularization - only during training
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=0.15, training=self.training)

        # GCN layers with residual connections on ALL layers
        for i, (conv, ln) in enumerate(zip(self.convs, self.lns)):
            x_prev = x
            x = conv(x, edge_index)
            x = ln(x)
            # Use LeakyReLU instead of ReLU to prevent dead neurons
            x = F.leaky_relu(x, negative_slope=0.2)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection on ALL layers (including first layer)
            # Scale factor helps with gradient flow
            x = x + 0.5 * x_prev

        return x

    def compute_diversity_loss(self, embeddings):
        """
        Compute embedding diversity loss to prevent collapse.
        Encourages embeddings to be different from each other.

        Args:
            embeddings: Node embeddings [num_nodes, hidden_dim]

        Returns:
            Diversity loss (lower when embeddings are more diverse)
        """
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix (sample a subset for efficiency)
        sample_size = min(1000, embeddings_norm.size(0))
        indices = torch.randperm(embeddings_norm.size(0))[:sample_size]
        sampled_emb = embeddings_norm[indices]

        # Compute pairwise cosine similarities
        similarity_matrix = torch.mm(sampled_emb, sampled_emb.t())

        # Exclude diagonal (self-similarity)
        mask = torch.eye(sample_size, device=embeddings.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, 0)

        # Diversity loss: penalize high similarity
        # We want low similarity, so we minimize the mean absolute similarity
        diversity_loss = similarity_matrix.abs().mean()

        return diversity_loss

    def forward(self, edge_index, pos_edge, neg_edge):
        """
        Forward pass with diversity loss.

        Returns:
            pos_pred: Predictions for positive edges
            neg_pred: Predictions for negative edges
            diversity_loss: Embedding diversity loss
        """
        z = self.encode(edge_index)

        # Compute diversity loss
        diversity_loss = self.compute_diversity_loss(z) if self.training else torch.tensor(0.0, device=z.device)

        # Decode edges using multi-strategy decoder
        pos_pred = self.decode(z, pos_edge)
        neg_pred = self.decode(z, neg_edge)

        return pos_pred, neg_pred, diversity_loss

    def forward_simple(self, edge_index, pos_edge, neg_edge):
        """
        Simple forward pass without diversity loss (for compatibility).
        """
        pos_pred, neg_pred, _ = self.forward(edge_index, pos_edge, neg_edge)
        return pos_pred, neg_pred
