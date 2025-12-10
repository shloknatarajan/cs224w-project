import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from ..base import BaseModel


class Node2VecGCN(BaseModel):
    """
    GCN encoder that combines Node2Vec embeddings with learnable IDs.
    
    Node2Vec embeddings are initialized from pretraining but fine-tuned during training.
    Dropout is kept at 0.0 to comply with baseline settings.
    """
    def __init__(
        self,
        node2vec_embeddings: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 2,
        decoder_dropout: float = 0.0,
        use_multi_strategy: bool = False,
    ):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)
        num_nodes, node2vec_dim = node2vec_embeddings.size()

        self.dropout = 0.0  # baseline requirement: keep dropout disabled
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.node2vec_dim = node2vec_dim

        # Learnable ID embeddings
        self.id_emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.id_emb.weight)

        # Node2Vec embeddings (initialized from pretraining, but fine-tunable)
        self.node2vec_emb = nn.Embedding.from_pretrained(node2vec_embeddings, freeze=False)

        input_dim = hidden_dim + node2vec_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))

        decoder_type = "multi-strategy" if use_multi_strategy else "simple"
        self.description = (
            f"GCN + Node2Vec (fine-tuned) | hidden_dim={hidden_dim}, "
            f"node2vec_dim={node2vec_dim}, layers={num_layers}, "
            f"decoder={decoder_type}, dropout=0.0"
        )

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.id_emb.weight, self.node2vec_emb.weight], dim=1)
        x = self.input_proj(x)

        for conv in self.convs:
            x_new = conv(x, edge_index)
            x_new = F.relu(x_new)
            # Dropout disabled by design (dropout must remain 0.0)
            if x_new.shape == x.shape:
                x = x + x_new
            else:
                x = x_new
        return x

