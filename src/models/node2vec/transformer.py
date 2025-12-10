import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import TransformerConv
from ..base import BaseModel


class Node2VecTransformer(BaseModel):
    """
    Graph Transformer encoder that combines Node2Vec embeddings with learnable IDs.
    
    Node2Vec embeddings are initialized from pretraining but fine-tuned during training.
    Dropout is kept at 0.0 to comply with baseline settings.
    """
    def __init__(
        self,
        node2vec_embeddings: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.0,
        decoder_dropout: float = 0.0,
        use_multi_strategy: bool = False,
    ):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)
        num_nodes, node2vec_dim = node2vec_embeddings.size()

        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.node2vec_dim = node2vec_dim
        self.heads = heads

        # Learnable ID embeddings
        self.id_emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.id_emb.weight)

        # Node2Vec embeddings (initialized from pretraining, but fine-tunable)
        self.node2vec_emb = nn.Embedding.from_pretrained(node2vec_embeddings, freeze=False)

        input_dim = hidden_dim + node2vec_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer layers
        self.convs = nn.ModuleList()
        # First layer: hidden_dim -> hidden_dim (with multi-head)
        self.convs.append(TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(TransformerConv(hidden_dim, hidden_dim, heads=1, concat=False))

        decoder_type = "multi-strategy" if use_multi_strategy else "simple"
        self.description = (
            f"Transformer + Node2Vec (fine-tuned) | hidden_dim={hidden_dim}, "
            f"node2vec_dim={node2vec_dim}, layers={num_layers}, heads={heads}, "
            f"decoder={decoder_type}, dropout={dropout}"
        )

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.id_emb.weight, self.node2vec_emb.weight], dim=1)
        x = self.input_proj(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            # Only apply dropout if > 0
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

