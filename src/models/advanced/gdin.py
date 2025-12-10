"""
GDIN: Graph Drug Interaction Network

Key improvements over GraphSAGE baseline:
1. Pairwise ranking loss (AUC loss) instead of BCE - directly optimizes Hits@K
2. Common neighbor features at decode time (Phase 2) - NCN-style

Design philosophy: Minimal changes to proven GraphSAGE base.
Based on PLNLP (90.88% Hits@20) and NCN research.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

from ..base import BaseModel


class GDIN(BaseModel):
    """
    Graph Drug Interaction Network

    Phase 1: GraphSAGE encoder + simple decoder + AUC loss (in trainer)
    Phase 2: Add common neighbor features at decode time

    Key differences from baseline GraphSAGE:
    - hidden_dim=256 (vs 128) for more capacity
    - Trained with pairwise ranking loss (AUC loss) instead of BCE
    - Phase 2: CN features added at decode time (not encoding time)
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_cn: bool = False,  # Enable in Phase 2
    ):
        # Use simple decoder (no multi-strategy), no dropout
        super().__init__(hidden_dim, decoder_dropout=0.0, use_multi_strategy=False)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_cn = use_cn

        # Learnable embeddings (same init as GraphSAGE)
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # GraphSAGE layers (identical to baseline)
        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Common neighbor weight (Phase 2)
        if use_cn:
            self.cn_weight = nn.Parameter(torch.tensor(0.5))

        # Model description
        cn_str = "CN-enabled" if use_cn else "no-CN"
        self.description = (
            f"GDIN ({num_layers} layers, {cn_str}) | "
            f"hidden_dim={hidden_dim}, dropout={dropout}"
        )

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Encode nodes to embeddings.
        Identical to GraphSAGE baseline - proven to work.
        """
        x = self.emb.weight

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def decode(
        self,
        z: torch.Tensor,
        edge: torch.Tensor,
        cn_scores: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Decode edge scores from node embeddings.

        Phase 1: Just use base decoder (Hadamard + MLP)
        Phase 2: Add common neighbor signal

        Args:
            z: Node embeddings [N, hidden_dim]
            edge: Edges to score [E, 2]
            cn_scores: Optional common neighbor scores [E] (Phase 2)

        Returns:
            Edge scores [E]
        """
        # Base score from inherited decoder
        base_score = self.decoder(z, edge)

        # Add CN contribution if enabled (Phase 2)
        if self.use_cn and cn_scores is not None:
            return base_score + self.cn_weight * cn_scores

        return base_score
