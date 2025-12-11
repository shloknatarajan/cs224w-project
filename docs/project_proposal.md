# Predicting Drug-Drug Interactions using GNNs - Project Proposal

**Avi Udash**  
Stanford University  
udashavi@stanford.edu

**Shlok Natarajan**  
Stanford University  
shlok.natarajan@stanford.edu

**Tanvir Bhathal**  
Stanford University  
tanvirb@stanford.edu

## 1 Project Proposal

The aim of our project is to predict interactions between drugs. Many patients take multiple medications concurrently, but interactions between taking multiple drugs could cause severe reactions or adverse side-effects. The interactions between different drugs can be modeled via a graph, and our plan is to run a link prediction task on this graph using GNNs.

### 1.1 Application Domain

For our project, we plan to use the Open Graph Benchmark - Drug Drug Interaction (ogbl-ddi) public dataset from Hu et al. (2020). This dataset is homogenous, undirected, unweighted, and contains 4,267 nodes and 1,334,889 edges. Each node in the database refers to a specific drug and each edge represents a known drug-drug interaction. This dataset already has a train-validation-test split which was done via a protein-target split, meaning that drug edges are split according to the proteins that the drugs target in the body. So the drugs in the test set act differently in the body than the ones in the training and validation sets. Our task is link prediction on the dataset, with the goal being to predict new edges which will represent drug pairs that are likely to interact but not known yet.

To evaluate, we use the Hits@K ranking performance, where K=10, 20, 50. The Hits@K metric captures the percentage of times that the correct answer appears in the top k predictions (Hoyt et al. (2022)). The equation for Hits@K is the following:

$$\text{Hits@K} = \frac{|\{t \in K_{test}|\text{rank}(t) \leq k\}|}{|K_{test}|}$$

In addition, we will also report Mean Reciprocal Rank (MRR), which evaluates how highly ranked the first correct item in a ranked list is (Hoyt et al. (2022)). Unlike Hits@K, MRR is sensitive to rank and Hoyt et al. (2022) compares it as a "softer" version of Hits@K. MRR is calculated by the following equation:

$$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_i}$$

where |Q| is the number of total queries.

We chose this dataset because understanding drug-drug interactions is an important problem in healthcare and it could have real-world impact by predicting new connections. Moreover, this particular dataset splits the train-validate-test sets by protein-target splits which is a more realistic evaluation setting and it tests how we would be able to generalize the model to different types of drugs.

### 1.2 Graph ML Techniques

In terms of our Graph ML models, we plan to use three progressively complex GNN models and compare them with each other: Graph Convolutional Networks, GraphSAGE, and GraphTransformer. We'll use a link prediction decoder for all three models as well.

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

## References

V. P. Dwivedi and X. Bresson. A generalization of transformer networks to graphs, 2021. URL https://arxiv.org/abs/2012.09699.

W. L. Hamilton, R. Ying, and J. Leskovec. Inductive representation learning on large graphs, 2018. URL https://arxiv.org/abs/1706.02216.

C. T. Hoyt, M. Berrendorf, M. Galkin, V. Tresp, and B. M. Gyori. A unified framework for rank-based evaluation metrics for link prediction in knowledge graphs, 2022. URL https://arxiv.org/abs/2203.07544.

W. Hu, M. Fey, M. Zitnik, Y. Dong, H. Ren, B. Liu, M. Catasta, and J. Leskovec. Open graph benchmark: Datasets for machine learning on graphs. arXiv preprint arXiv:2005.00687, 2020.

T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks, 2017. URL https://arxiv.org/abs/1609.02907.