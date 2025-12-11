# Predicting Drug-Drug Interactions using GNNs - Midway Report

**Avi Udash**  
Stanford University  
udashavi@stanford.edu

**Shlok Natarajan**  
Stanford University  
shlok.natarajan@stanford.edu

**Tanvir Bhathal**  
Stanford University  
tanvirb@stanford.edu

## 1 Introduction

The aim of our project is to predict interactions between drugs. Many patients take multiple medications concurrently, but interactions between taking multiple drugs could cause severe reactions or adverse side-effects. The interactions between different drugs can be modeled via a graph, and our plan is to run a link prediction task on this graph using GNNs.

### 1.1 Dataset Overview

For our project, we plan to use the Open Graph Benchmark - Drug Drug Interaction (ogbl-ddi) public dataset from Hu et al. (2020). This dataset is homogenous, undirected, unweighted, and contains 4,267 nodes and 1,334,889 edges. Each node in the database refers to a specific drug and each edge represents a known drug-drug interaction. This dataset already has a train-validation-test split which was done via a protein-target split, meaning that drug edges are split according to the proteins that the drugs target in the body. So the drugs in the test set act differently in the body than the ones in the training and validation sets. Our task is link prediction on the dataset, with the goal being to predict new edges which will represent drug pairs that are likely to interact but not known yet.

To evaluate, we use the Hits@K ranking performance, where K=10, 20, 50. The Hits@K metric captures the percentage of times that the correct answer appears in the top k predictions (Hoyt et al. (2022)). The equation for Hits@K is the following:

$$\text{Hits@K} = \frac{|\{t \in K_{test}|\text{rank}(t) \leq k\}|}{|K_{test}|}$$

In addition, we will also report Mean Reciprocal Rank (MRR), which evaluates how highly ranked the first correct item in a ranked list is (Hoyt et al. (2022)). Unlike Hits@K, MRR is sensitive to rank and Hoyt et al. (2022) compares it as a "softer" version of Hits@K. MRR is calculated by the following equation:

$$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_i}$$

where |Q| is the number of total queries.

We chose this dataset because understanding drug-drug interactions is an important problem in healthcare and it could have real-world impact by predicting new connections. Moreover, this particular dataset splits the train-validate-test sets by protein-target splits which is a more realistic evaluation setting and it tests how we would be able to generalize the model to different types of drugs.

We used the following code to work with loading the dataset:

```python
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, TransformerConv
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

dataset = PygLinkPropPredDataset('ogbl-ddi')
data = dataset[0]
split_edge = dataset.get_edge_split()

train_pos = split_edge['train']['edge'].to(device)
valid_pos = split_edge['valid']['edge'].to(device)
test_pos = split_edge['test']['edge'].to(device)

num_nodes = data.num_nodes

# Construct graph using *only* training edges (IMPORTANT: prevents leakage)
data.edge_index = train_pos.t().contiguous().to(device)

evaluator = Evaluator(name='ogbl-ddi')
```

### 1.2 Graph ML Techniques

In terms of our Graph ML models, we plan to use three progressively complex GNN models as a baseline: Graph Convolutional Networks, GraphSAGE, and GraphTransformer. We'll use a link prediction decoder for all three models as well. We can use these baselines to see how our novel GNN method compares.

#### 1.2.1 Graph Convolutional Network

We'll start with the Graph Convolutional Network (GCN) architecture introduced by Kipf and Welling (2017). GCNs work by aggregating and passing messages from a node's neighbors through graph convolutions. The update function for a GCN will look like the following:

$$h_v^{(l)} = \sigma\left(W^{(l)} \sum_{u \in N(v)} \frac{h_u^{(l-1)}}{|N(v)|}\right)$$

where $h_v^{(l)}$ is the embedding of a node v at layer k, $W^{(l)}$ is a learnable weight matrix, and σ is a non-linearity function. Since the ogbl-ddi dataset contains no node features, we initialize each node with a learnable embedding vector, which the GCN layers then refine through neighborhood aggregation.

#### 1.2.2 GraphSAGE

We'll also be working with GraphSAGE, which was proposed by Hamilton et al. (2018) and is an inductive method. In contrast to GCN which takes the entire K-hop neighborhood, GraphSAGE will do neighborhood sampling to learn an aggregator function. This is inductive because it can be scaled to extremely large graphs and can be used when the graph expands, unlike GCN which has to be retrained if new nodes are added. This update function looks like:

$$h_v^{(l)} = \sigma\left(W^{(l)} \cdot \text{CONCAT}\left(h_v^{(l-1)}, \text{AGG}\{h_u^{(l-1)}, \forall u \in N(v)\}\right)\right)$$

#### 1.2.3 Graph Transformer

Graph Transformer Networks is a more recent architecture introduced by Dwivedi and Bresson (2021) that adapt the self-attention mechanism from the standard Transformer architecture. Unlike message passing GNNs like GCN and GraphSAGE, Graph transformers allow attending over all nodes and edges, which allows it to better capture global structural patterns. Figure 1 provides a visual overview of the graph transformer architecture as shown in the paper by Dwivedi and Bresson (2021).

![Figure 1: Graph transformer layers proposed by Dwivedi and Bresson (2021)](figure_1_description)

#### 1.2.4 Reasons for selecting these models

We believe these models are well suited for the dataset. This is because all three models can learn purely from graph structure, since the dataset doesn't contain node features. All three models can also be used to for the downstream link prediction task. It will be interesting to compare the results, since they offer different advantages. GraphSAGE and GCN captures more local neighborhoods, while graph transformers can identify global structural similarities better.

## 2 Summary of Progress - Midway Report

So far, we have implemented basic versions of each model and the first version of our Graph Drug Interaction Network (GDIN) model.

After creating the initial versions of each model, we focused on primarily optimizing the Graph Convolutional Network (GCN) architecture to develop GDIN v1.

### 2.1 Baseline Initial Architectures

Our initial architectures consisted of very simple implementations. For node representations, we used random initial features. We also had 2 GNN layers in the initial version (e.g. for GCN, we had 2 GCN layers - for GraphSage, we had 2 GraphSage layers, etc.) Our decoder was a simple dot product and we did not use any regularization. In addition, we did random negative sampling and used SGD for training.

For example, our initial models looked like the following:

BaseModel that everything was built on:

```python
class BaseModel(nn.Module):
    def decode(self, z, edge):
        src, dst = edge[:, 0], edge[:, 1]
        return (z[src] * z[dst]).sum(dim=1)
```

The GCN code:

```python
class GCN(BaseModel):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
    
    def encode(self, edge_index):
        # Random non-trainable node features -> prevents memorization
        x = torch.randn(num_nodes, self.hidden_dim, device=device)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
```

The Graph SAGE code:

```python
class GraphSAGE(BaseModel):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
    
    def encode(self, edge_index):
        x = torch.randn(num_nodes, self.hidden_dim, device=device)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
```

The Graph Transformer code:

```python
class GraphTransformer(BaseModel):
    def __init__(self, hidden_dim, heads=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads)
    
    def encode(self, edge_index):
        x = torch.randn(num_nodes, self.hidden_dim, device=device)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
```

### 2.2 GDIN Architecture v1

Our GDIN architecture as of now focuses on aggregating neighborhood information through spectral graph convolutions (Kipf and Welling, 2017).

$$h_v^{(l+1)} = \sigma\left(W^{(l)} \sum_{u \in N(v) \cup v} \frac{1}{\sqrt{|N(v)||N(u)|}} h_u^{(l)}\right)$$

Our architectural details are as follows:

- Input: Learnable embeddings + 6D structural features
- Feature projection: MLP to combine embeddings and structural features
- Hidden dimensions: 192
- Layers: 3 GCN Layers with batch normalization and scaled residual connections
- Dropout: 0.5 in GNN layers and 0.1 for structural features
- Edge dropout: 0.15 during training to avoid overfitting

We used 6 dimensional structual features that took the following into account: degree, clustering, core, PageRank, neighbor statistics. We increased the number of layers, increased the hidden dimensions, and added batch normalization to regularize. In our training, we used binary cross-entropy with L2 embedding regularization:

$$L = \text{BCE}(\text{pos edges}) + \text{BCE}(\text{neg edges}) + 0.005 \cdot ||E||_2^2$$

This improved architecture is shown in the following code snippet:

```python
class GDIN(BaseModel):
    def __init__(self, num_nodes, hidden_dim, num_layers=3,
                 dropout=0.4, decoder_dropout=0.3, use_structural_features=True,
                 num_structural_features=6, use_multi_strategy=True):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout,
                        use_multi_strategy=use_multi_strategy)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_structural_features = use_structural_features
        self.num_structural_features = num_structural_features
        
        # Learnable embeddings with improved initialization
        emb_dim = hidden_dim - num_structural_features if use_structural_features else hidden_dim
        self.emb = nn.Embedding(num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)
        
        # Feature projection layer to transform structural features
        if use_structural_features:
            self.feature_proj = nn.Sequential(
                nn.Linear(num_structural_features, num_structural_features),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)  # Lighter dropout on features
            )
        
        # Multiple GCN layers with residual connections - keep BatchNorm as it worked in baseline
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim,
                                     add_self_loops=False))  # We already added self-loops
            self.bns.append(nn.BatchNorm1d(hidden_dim))
    
    def encode(self, edge_index):
        x = self.emb.weight
        
        # Concatenate rich structural features with learnable embeddings
        if self.use_structural_features:
            # Project structural features through a small MLP
            struct_feats = self.feature_proj(node_structural_features)
            # Add feature dropout for regularization
            if self.training:
                struct_feats = F.dropout(struct_feats, p=0.1, training=True)
            x = torch.cat([x, struct_feats], dim=1)
        
        # Moderate edge dropout for regularization - only during training
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=0.2, training=self.training)
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_prev = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection with scaling (for all layers except first)
            if i > 0:
                x = x + 0.5 * x_prev  # Scaled residual for better gradient flow
        
        return x
```

Also unlike our previous decoder, we implemented a learnable multi-strategy decoder that would combine three scoring functions:

$$\text{score}(v, u) = w_1 \cdot \text{Hadamard}(h_v, h_u) + w_2 \cdot \text{Concat}(h_v, h_u) + w_3 \cdot \text{Bilinear}(h_v, h_u)$$

Each decoder has 2 layers with dropout which would prevent overfitting.

### 2.3 Early Results

We achieved substantial improvements through our iterations on the GDIN model, with our baselines being our initial results shown in Table 1. The final GDIN v1 architecture described in the previous section showed improvement: getting a 27.67% validation hits@20 and 24.18% test hits@20. An important thing to initially highlight, is the lack of overfitting in GDIN. In Table 1 we see how the Validation and Test Gap is significantly large, but in the GDIN model, this is only 1.14x.

**Table 1: Initial Results**

| Model | Val Hits@20 | Test Hits@20 | Val/Test Gap |
|-------|-------------|--------------|--------------|
| GCN | 16.47% | 9.35% | 1.76x |
| GraphTransformer | 15.38% | 6.77% | 2.27x |
| GraphSAGE | 1.53% | 0.89% | 1.72x |

### 2.4 Next Steps

Based on our early results, we plan to integrate more learnings from the GraphSAGE and GraphTransformer models into the GDIN. We also want to try more feature engineering. We are open to suggestions on what we should try as we are currently doing a lot of literature review to find what is optimal.
