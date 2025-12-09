"""
Chemistry-Aware Edge Decoder

Combines structural (GNN) and chemical (ChemBERTa/Morgan) signals at the decoder level.
This allows the model to learn when to use topology vs chemistry for edge prediction.
"""
import torch
import torch.nn.functional as F
from torch import nn


class ChemistryAwareDecoder(nn.Module):
    """
    Edge decoder that intelligently combines structural and chemical information.
    
    Key idea: Keep GNN encoder structure-only (strong baseline), but incorporate
    chemistry when scoring edges. The model learns when chemistry is useful.
    
    Architecture:
        1. Structural path: MLP(z_i ⊙ z_j) - topology-based scoring
        2. Chemical path: MLP(chem_i ⊙ chem_j) - chemistry-based scoring
        3. Combined path: MLP([z_i⊙z_j, chem_i⊙chem_j]) - joint reasoning
        4. Learnable weights to combine all three paths
    """
    
    def __init__(self, 
                 structural_dim: int = 128,
                 chemical_dim: int = 768,
                 dropout: float = 0.3,
                 use_gating: bool = False):
        """
        Args:
            structural_dim: Dimension of GNN node embeddings
            chemical_dim: Dimension of chemical features (768 for ChemBERTa, 2048 for Morgan)
            dropout: Dropout rate for MLPs
            use_gating: If True, use learned gates instead of fixed weights
        """
        super().__init__()
        self.structural_dim = structural_dim
        self.chemical_dim = chemical_dim
        self.use_gating = use_gating
        
        # Path 1: Pure structural scoring (z_i ⊙ z_j)
        self.structural_mlp = nn.Sequential(
            nn.Linear(structural_dim, structural_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(structural_dim // 2, 1)
        )
        
        # Path 2: Pure chemical scoring (chem_i ⊙ chem_j)
        self.chemical_mlp = nn.Sequential(
            nn.Linear(chemical_dim, chemical_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(chemical_dim // 4, 1)
        )
        
        # Path 3: Combined scoring [z_i⊙z_j, chem_i⊙chem_j]
        self.combined_mlp = nn.Sequential(
            nn.Linear(structural_dim + chemical_dim, structural_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(structural_dim, 1)
        )
        
        if use_gating:
            # Learned gating network: decides how much to trust each path
            # Input: [z_i, z_j, chem_i, chem_j, mask_i, mask_j]
            self.gate_network = nn.Sequential(
                nn.Linear(2 * structural_dim + 2 * chemical_dim + 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 3),  # 3 gates for 3 paths
                nn.Softmax(dim=-1)
            )
        else:
            # Fixed learnable weights (simpler, often works well)
            self.path_weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, z, chemistry, edge, smiles_mask=None):
        """
        Compute edge scores using structural + chemical information.
        
        Args:
            z: Structural node embeddings [N, structural_dim]
            chemistry: Chemical node features [N, chemical_dim]
            edge: Edge indices [E, 2]
            smiles_mask: Binary mask [N] indicating valid chemistry (1=valid, 0=missing)
        
        Returns:
            scores: Edge scores [E]
        """
        src, dst = edge[:, 0], edge[:, 1]
        
        # Get embeddings for source and destination nodes
        z_src, z_dst = z[src], z[dst]  # [E, structural_dim]
        chem_src, chem_dst = chemistry[src], chemistry[dst]  # [E, chemical_dim]
        
        # Path 1: Structural scoring (Hadamard product)
        structural_prod = z_src * z_dst  # [E, structural_dim]
        score_structural = self.structural_mlp(structural_prod).squeeze(-1)  # [E]
        
        # Path 2: Chemical scoring (Hadamard product)
        chemical_prod = chem_src * chem_dst  # [E, chemical_dim]
        score_chemical = self.chemical_mlp(chemical_prod).squeeze(-1)  # [E]
        
        # Path 3: Combined scoring
        combined_prod = torch.cat([structural_prod, chemical_prod], dim=-1)  # [E, structural_dim + chemical_dim]
        score_combined = self.combined_mlp(combined_prod).squeeze(-1)  # [E]
        
        # Combine paths
        if self.use_gating:
            # Dynamic gating based on node features
            if smiles_mask is not None:
                mask_src, mask_dst = smiles_mask[src].unsqueeze(-1), smiles_mask[dst].unsqueeze(-1)  # [E, 1]
            else:
                mask_src = torch.ones(src.size(0), 1, device=z.device)
                mask_dst = torch.ones(dst.size(0), 1, device=z.device)
            
            gate_input = torch.cat([z_src, z_dst, chem_src, chem_dst, mask_src, mask_dst], dim=-1)
            gates = self.gate_network(gate_input)  # [E, 3]
            
            # Weighted combination using learned gates
            scores = (gates[:, 0] * score_structural + 
                     gates[:, 1] * score_chemical + 
                     gates[:, 2] * score_combined)
        else:
            # Fixed weighted combination
            weights = F.softmax(self.path_weights, dim=0)
            scores = (weights[0] * score_structural + 
                     weights[1] * score_chemical + 
                     weights[2] * score_combined)
        
        # Optional: Mask out chemical contribution for edges with missing SMILES
        if smiles_mask is not None:
            # Only use chemistry when BOTH nodes have valid SMILES
            both_valid = smiles_mask[src] * smiles_mask[dst]  # [E]
            # If either node missing, fall back to structural score only
            scores = torch.where(both_valid > 0.5, scores, score_structural)
        
        return scores


class SimpleChemistryDecoder(nn.Module):
    """
    Simplified version: Just adds chemical similarity as a bonus to structural score.
    
    score = structural_score + α * chemical_score (only when both nodes have valid SMILES)
    """
    
    def __init__(self, 
                 structural_dim: int = 128,
                 chemical_dim: int = 768,
                 dropout: float = 0.3):
        super().__init__()
        self.structural_dim = structural_dim
        self.chemical_dim = chemical_dim
        
        # Structural decoder
        self.structural_mlp = nn.Sequential(
            nn.Linear(structural_dim, structural_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(structural_dim // 2, 1)
        )
        
        # Chemical decoder
        self.chemical_mlp = nn.Sequential(
            nn.Linear(chemical_dim, chemical_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(chemical_dim // 4, 1)
        )
        
        # Learnable weight for chemical contribution
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Start small
    
    def forward(self, z, chemistry, edge, smiles_mask=None):
        """
        Args:
            z: Structural embeddings [N, structural_dim]
            chemistry: Chemical features [N, chemical_dim]
            edge: Edge indices [E, 2]
            smiles_mask: Binary mask [N] for valid chemistry
        
        Returns:
            scores: Edge scores [E]
        """
        src, dst = edge[:, 0], edge[:, 1]
        
        # Structural score (always computed)
        z_src, z_dst = z[src], z[dst]
        structural_prod = z_src * z_dst
        score_structural = self.structural_mlp(structural_prod).squeeze(-1)
        
        # Chemical score (only used when both nodes have valid SMILES)
        chem_src, chem_dst = chemistry[src], chemistry[dst]
        chemical_prod = chem_src * chem_dst
        score_chemical = self.chemical_mlp(chemical_prod).squeeze(-1)
        
        # Combine: structure + α * chemistry (only when both valid)
        if smiles_mask is not None:
            both_valid = smiles_mask[src] * smiles_mask[dst]
            score_chemical = both_valid * score_chemical
        
        scores = score_structural + self.alpha * score_chemical
        
        return scores

