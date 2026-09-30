# Information Bottleneck-Guided Adaptive Hypergraph Transformer for Brain Disease Diagnosis

Jingxi Feng Xudong Chen Yifan Zhang Heming Xu Hongcheng Han Xijing Wang Dong Zhang Shaoyi Du

## Abstract

Exploring high-order correlations and long-range dependencies in brain networks holds significant value for both neuroscience research and clinical diagnosis. However, previous studies have lacked a unified integration of high-order and long-range dependency information in brain networks, and there is substantial redundancy behind various types of information. These issues limit their effectiveness in the diagnosis of brain diseases. To address this, we propose an Information Bottleneck-Guided Adaptive HyperGraph Transformer (IBAHGT). By incorporating the information bottleneck (IB) principle, this approach enables adaptive learning of high-order correlations and both short- and long-range dependencies within a unified framework for brain network analysis, achieving high-precision brain disease diagnosis. IBAHGT consists of three key components: an information bottleneckguided adaptive hypergraph convolution, which introduces a novel hypergraph information bottleneck (HIB) principle to adaptively learn hypergraph messagepassing weights between nodes and hyperedges, optimizes information flow and captures high-order information in brain networks that is maximally informative and minimally redundant (MIMR). The Transformer encoder captures global information within brain networks through the attention mechanism, specifically modeling short- and long-range dependencies. An information bottleneck-guided node-level adaptive fusion employs the IB principle to learn independent weights for each node, facilitating the fine-grained integration of high-order information and global information to obtain an efficient representation for downstream tasks. Extensive experiments demonstrate that the proposed method outperforms current state-of-the-art methods and can identify biomarkers for clinical applications.

## 1 Introduction

The brain network describes the interactions and information processing patterns between regions of interest (ROIs) [1]. Brain network analysis, which explores interactions between ROIs, reveals the brain’s functional organization and is crucial for understanding the pathogenesis of neurological diseases and their early intervention [2, 3]. As shown in Fig.1, many studies have shown the presence of extensive high-order correlations [4], along with short- and long-range dependencies, within brain networks [5, 6]. High-order correlations reflect the joint activation of multiple ROIs during functional activities such as cognitive tasks, and are crucial for understanding the activity patterns of the brain’s neural system [7]. Furthermore, during functional activities, the brain selectively integrates information through the coordinated work of high-order neural circuits to improve task execution efficiency and accuracy [8]. Short-range dependencies reflect the interactions between ROIs in the neighboring space [9], while long-range dependencies reflect long-distance communication between ROIs and play a critical role in understanding brain functions [10] and communication [11]. Therefore, it is necessary to simultaneously capture the high-order correlations and long-range dependencies in the brain network for unified analysis.

Many studies use graph learning methods for brain network analysis [12, 13], modeling the brain network as a graph and capturing pairwise dependencies in feature interactions through Graph Neural Networks (GNNs). However, these methods overlook the crucial high-order correlations and fail to capture the collaborative interactions among multiple ROIs involved in brain functions. Hypergraphs, a mathematical model that can express high-order correlations, have been increasingly used in brain network analysis [14]. For example, FC-HAT effectively captures high-order information in brain networks through hypergraph attention networks [15], thereby improving the accuracy of brain disease diagnosis.

![](images/4213e4fc38f3b410162bcbfb87718dbea2b269bc47da64cee9656cd25f9c6892.jpg)  
Figure 1: The brain network contains extensive high-order correlations as well as short-range and long-range dependencies. Multi-source information fusion, guided by the information bottleneck, aims to optimize the fused representation by capturing the minimal sufficient information from highorder representation X and global representation X for predicting the classification label Y.

GNNs and Hypergraph Neural Networks (HGNNs) [16, 17], which aggregate neigh borhood information through message passing, struggle to capture long-range dependencies in

brain networks. Unlike the local information aggregation mechanisms of GNNs and HGNNs, the attention mechanism in Transformer can capture long-range dependencies between nodes [18]. Brain network Transformer [19] has introduced Transformer into brain network analysis to learn long-range dependencies between ROIs, achieving promising performance in brain disease diagnosis.

However, these methods fail to integrate the high-order correlations and long-range dependencies according to the specific characteristics of each ROI. Furthermore, as shown in Fig.1, both the high-order information captured by general hypergraph convolution and the information captured based on different dependencies contain substantial redundancy, which dilutes the disease-related discriminative information. Thus, it is crucial to condense the information during the extraction of high-order features and the fusion of multi-source information to achieve efficient representations.

To address these issues, we propose an information bottleneck-guided adaptive hypergraph Transformer method that effectively captures the MIMR high-order correlations as well as short- and long-range dependencies in the brain network. Specifically, first, an information bottleneck-guided adaptive hypergraph convolution introduces an innovative hypergraph information bottleneck (HIB) principle to optimize the message passing process, achieving the goal of capturing MIMR high-order information. Secondly, the Transformer encoder captures short- and long-range dependencies between ROIs through the self-attention mechanism. Finally, an information bottleneck-guided node-level adaptive fusion method considers the unique roles and needs of each ROI, and adaptively learns the weights for each node (i.e., ROI) to fuse representations that contain high-order information and global information. In summary, this paper makes three main contributions:

1) We propose an information bottleneck-guided adaptive hypergraph Transformer method that captures high-order correlations as well as short- and long-range dependencies in the brain network within a unified framework, achieving high-precision brain disease diagnosis.

2) We introduce an information bottleneck-guided adaptive hypergraph convolution method. By incorporating the HIB principle, it adaptively learns the message passing weights between nodes and hyperedges, capturing MIMR high-order information in the brain network.

3) We propose an information bottleneck-guided node-level adaptive fusion method that finegrained integrates high-order and global information in the brain network. Further optimization through the IB principle effectively eliminates redundancy during multi-source information fusion, resulting in efficient representations.

![](images/c2937be233369711177053c7ac0591be5222ec913ae53ebb2a6d3eda0a30c56d.jpg)  
Figure 2: The overall framework of the proposed IBAHGT.

## 2 Related work

## 2.1 Deep Learning-Based Brain Network Analysis

GNNs are commonly used deep learning methods for graph-structured brain networks learning [20, 21]. BrainGNN [22] introduces a novel ROI-aware graph convolution (Ra-GConv) layer for functional brain network analysis. However, graph-based methods overlook the high-order correlations that are widespread in brain networks. Recent studies have incorporated hypergraph learning methods into brain network analysis [15]. For example, dwHGCN [23] learns high-order correlations in brain networks through dynamic hypergraph learning with learnable hyperedge weights. Graphor hypergraph-based methods are unable to effectively capture long-range dependencies between ROIs. BrainNetTF [19] introduces a Transformer encoder to learn long-range dependencies and has demonstrated outstanding performance in brain disease diagnosis tasks. However, the above methods fail to provide a unified analysis that incorporates both the high-order correlations as well as shortand long-range dependencies present in brain networks.

## 2.2 Information Bottleneck

The Information Bottleneck (IB) [24] is a data compression technique aimed at encouraging representations derived from raw data to capture the maximum relevant information about the target, while excluding extraneous information unrelated to the prediction task. Deep learning methods based on the IB principle have been widely applied in various fields, such as computer vision [25, 26] and natural language processing [27]. Recently, BrainIB [28] has incorporated the IB principle into brain network analysis, effectively identifying disease-specific prominent brain network connections, thus enabling interpretable brain disease diagnosis.

## 3 Method

In brain network analysis, the ROIs are treated as nodes, and the functional connectivity (FC) matrix $\mathbf { X } ^ { 0 } \in \mathbb { R } ^ { N \times N }$ is obtained by calculating Pearson correlation between ROIs, serving as the initial node features. Our goal is to comprehensively capture various types of correlations within brain networks and distill this multi-source information into efficient feature representations for predicting the target Y , (i.e., the ground truth class label). The overall framework of IBAHGT is shown in Fig.2, with its core consisting of L layers of information bottleneck-guided adaptive hypergraph Transformer layers, each comprising three main components: information bottleneck-guided adaptive hypergraph convolution, an adaptive message passing process guided by the HIB principle, which produces high-order representation $\mathbf { X } _ { h } ^ { l } .$ . Transformer encoder captures short- and long-range dependencies to obtain global representation $\mathbf { X } _ { t } ^ { l }$ . Information bottleneck-guided adaptive node-level fusion learns the fusion weights for each node, enabling fine-grained integration of MIMR high-order and global information to obtain updated node features $\mathbf { X } ^ { \tilde { l } }$ . After L layers, the final node embeddings $\mathbf { \bar { X } } ^ { L }$ is flattened and fed into a multilayer perceptron (MLP) to predict class label.

## 3.1 Information Bottleneck-Guided Adaptive Hypergraph Convolution

Hypergraph-based methods can model high-order correlations among ROIs. However, general hypergraph convolution employs equal-weight message passing when capturing high-order information, which introduces substantial redundant information. To address this, the information bottleneckguided adaptive hypergraph convolution adaptively learns weights during the message passing and introduces the HIB principle to optimize this process, capturing the MIMR high-order information.

## 3.1.1 Adaptive Hypergraph Convolution

The hypergraph is constructed using the K Nearest Neighbor (KNN) method to model high-order corre lations in the brain network. Specifically, ROIs are represented as a set of nodes $\mathcal { V } = \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { N } \}$ The hypergraph structure is represented by an incidence matrix $\mathbf { H } \in \mathbb { R } ^ { N \times E }$ , reflecting the nodehyperedge relation. The proximity between nodes is measured by the inverse of the Euclidean distance based on the node features $\mathbf { X } ^ { 0 }$

Based on the constructed hypergraph structure, the adaptive hypergraph convolution performs a two-stage message passing process: adaptive gathering of node features to hyperedges and adaptive aggregating of hyperedge features to nodes. In the first stage, each node adaptively learns the weight and performs weighted aggregation to obtain the MIMR hyperedge feature $\mathbf { Z } ^ { \dot { l } }$ , with the HIB principle introduced to optimize the weight learning process. In the l-th layer, for a given hyperedge $e _ { j } .$ , the process of generating the hyperedge feature $\mathbf { Z } _ { j } ^ { l }$ based on the features of the nodes in its associated node set $\mathcal { N } _ { v } ( e _ { j } )$ can be expressed as follows:

$$
\mathbf { Z } _ { j } ^ { l } = \sigma \left( \sum _ { v _ { i } \in \mathcal { N } _ { v } ( e _ { j } ) } \boldsymbol { \alpha } _ { i , j } ^ { l } \mathbf { W } _ { 1 } ^ { l } \mathbf { X } _ { i } ^ { l - 1 } \right) , \quad \boldsymbol { \alpha } _ { i , j } ^ { l } = \frac { \exp \big ( \mathbf { a } _ { 1 } ^ { l } \mathbf { W } _ { 1 } ^ { l } \mathbf { X } _ { i } ^ { l - 1 } \big ) } { \sum _ { v _ { k } \in \mathcal { N } _ { v } ( e _ { j } ) } \exp \big ( \mathbf { a } _ { 1 } ^ { l } \mathbf { W } _ { 1 } ^ { l } \mathbf { X } _ { k } ^ { l - 1 } \big ) } ,\tag{1}
$$

where $\mathbf { a } _ { 1 } ^ { l }$ is a learnable vector, $\mathbf { W } _ { 1 } ^ { l }$ is a learnable weight matrix, σ represents the nonlinear activation function. $\alpha _ { i , j } ^ { l }$ represents the weight when the feature of node $v _ { i }$ is aggregated into the hyperedge $e _ { j }$ In the second stage, each hyperedge adaptively learns the weight for the information transfer back to the nodes and performs a weighted aggregation to obtain the high-order representation $\mathbf { X } _ { h } ^ { l }$ This process is also optimized through the HIB principle. For node $v _ { i }$ , the process of weighted aggregation of the hyperedge features from its connected hyperedge set $\mathcal { N } _ { e } ( v _ { i } )$ to obtain the highorder representation $\mathbf { X } _ { h , i } ^ { l }$ is formalized as follows:

$$
\mathbf { X } _ { h , i } ^ { l } = \sigma \left( \sum _ { e _ { j } \in N _ { e } ( v _ { i } ) } \beta _ { i , j } ^ { l } \mathbf { W } _ { 2 } ^ { l } \mathbf { Z } _ { j } ^ { l } \right) , \quad \beta _ { i , j } ^ { l } = \frac { \exp { ( \mathbf { a } _ { 2 } ^ { l } \mathbf { W } _ { 2 } ^ { l } \mathbf { Z } _ { j } ^ { l } ) } } { \sum _ { e _ { p } \in N _ { e } ( v _ { i } ) } \exp { ( \mathbf { a } _ { 2 } ^ { l } \mathbf { W } _ { 2 } ^ { l } \mathbf { Z } _ { p } ^ { l } ) } } ,\tag{2}
$$

where $\mathbf { a } _ { 2 } ^ { l }$ is a learnable vector, $\mathbf { W } _ { 2 } ^ { l }$ is a learnable weight matrix, and $\beta _ { i j } ^ { l }$ represents the weight when the feature of hyperedge $e _ { j }$ is transmitted back to node $v _ { i }$

## 3.1.2 Optimization Based on the HIB Principle

There is redundancy in the high-order information within the brain captured by adaptive hypergraph convolution, for this reason, we propose the HIB principle to optimize the information flow process to capture the MIMR high-order information. In the l-th layer of adaptive hypergraph convolution, the optimization objective based on the HIB principle is:

$$
\operatorname* { m i n } _ { \mathbb { P } ( \mathbf { Z } ^ { l } , \mathbf { X } ^ { l } \mid \mathbf { H } , \mathbf { X } ^ { l - 1 } ) \in \Omega } { \mathrm { H I B } } _ { \gamma _ { 1 } } ( \mathbf { H } , \mathbf { X } ^ { l - 1 } , Y ; \mathbf { Z } ^ { l } , \mathbf { X } _ { h } ^ { l } ) \triangleq - I ( Y ; \mathbf { Z } ^ { l } , \mathbf { X } _ { h } ^ { l } ) + \gamma _ { 1 } I ( \mathbf { X } ^ { l - 1 } , \mathbf { H } ; \mathbf { Z } ^ { l } , \mathbf { X } _ { h } ^ { l } ) .\tag{3}
$$

Minimizing $- I ( Y ; \mathbf { Z } ^ { l } , \mathbf { X } _ { h } ^ { l } )$ ) ensures that the hypergraph message passing process focuses on taskrelevant information, while minimizing $I ( \mathbf { X } ^ { l - 1 } , \mathbf { H } ; \mathbf { Z } ^ { l } , \mathbf { X } _ { h } ^ { l } )$ effectively removes redundant information. $\gamma _ { 1 }$ is a hyperparameter that balances information and redundancy. However, accurately

calculating mutual information terms in $\operatorname { E q . } ( 3 )$ is intractable. Therefore, we introduce the lower bound of $\breve { I } ( Y ; \mathbf { Z } ^ { l } , \mathbf { X } _ { h } ^ { l } )$ and the upper bound of $\bar { I } ( \mathbf { X } _ { h } ^ { l } , \mathbf { Z } ^ { l } ; \mathbf { X } ^ { l - 1 }$ , H) for subsequent estimation.

Proposition 1. For any distribution $\mathbb { Q } _ { 1 } ( Y | \mathbf { Z } ^ { l } )$ and $\mathbb { Q } _ { 2 } ( Y )$ , we have

$$
I ( Y ; { \mathbf { Z } } ^ { l } , { \mathbf { X } } _ { h } ^ { l } ) = I ( Y ; { \mathbf { Z } } ^ { l } ) + I ( Y ; { \mathbf { X } } _ { h } ^ { l } \mid { \mathbf { Z } } ^ { l } ) .\tag{4}
$$

To represent the lower bounds of two terms on the right-hand side of the equation, in the adaptive hypergraph convolution layer, $\mathbf { Z } ^ { l }$ and $\mathbf { X } _ { h } ^ { l }$ are read out and then mapped through the learnable parameter matrix to the predicted labels $\hat { y } _ { z } ^ { l }$ and $\hat { y } _ { x } ^ { l }$ , respectively. Therefore, the two terms on the right-hand side of the Eq.(4) can reduce to the cross-entropy loss without constants as follows:

$$
I ( Y ; { \bf Z } ^ { l } ) \to - \mathcal { L } _ { C E } ( \hat { y } _ { z } ^ { l } , Y ) \quad , \quad I ( Y ; { \bf X } _ { h } ^ { l } \mid { \bf Z } ^ { l } ) \to - \mathcal { L } _ { C E } ( \hat { y } _ { x } ^ { l } , Y ) .\tag{5}
$$

The proof of the above process is given in Appendix B. Subsequently, we derive the upper bound of the second term in $\operatorname { E q . } ( { \bar { 3 } } )$ ).

Proposition 2. For any distributions $\mathbb { Q } ( \mathbf { Z } ^ { l } )$ and $\mathbb { Q } ( \mathbf { X } _ { h } ^ { l } )$ , we have

$$
I ( \mathbf { X } ^ { l - 1 } , \mathbf { H } ; \mathbf { X } _ { h } ^ { l } , \mathbf { Z } ^ { l } ) \leq I ( \mathbf { X } ^ { l - 1 } , \mathbf { H } ; \mathbf { Z } ^ { l } ) + I ( \mathbf { Z } ^ { l } , \mathbf { H } ; \mathbf { X } _ { h } ^ { l } ) \leq \mathbf { Z } \mathbf { I } \mathbf { B } ^ { l } + \mathbf { X } \mathbf { I } \mathbf { B } ^ { l } ,\tag{6}
$$

$$
\mathbf { Z } \mathbf { I B } ^ { l } = D _ { K L } \left( \mathbb { P } ( \mathbf { Z } ^ { l } \mid \mathbf { X } ^ { l - 1 } , \mathbf { H } ) \parallel \mathbb { Q } ( \mathbf { Z } ^ { l } ) \right) \mathrm { , ~ } \mathbf { X } \mathbf { I B } ^ { l } = D _ { K L } \left( \mathbb { P } ( \mathbf { X } _ { h } ^ { l } \mid \mathbf { Z } ^ { l } , \mathbf { H } ) \parallel \mathbb { Q } ( \mathbf { X } _ { h } ^ { l } ) \right)\tag{7}
$$

The proof of the proposition can be found in Appendix B. To estimate $Z \mathrm { I B } ^ { l }$ and $\mathrm { X I B } ^ { l }$ , we set both $\mathbb { Q } ( \mathbf { Z } ^ { l } )$ and $\dot { \mathbb { Q } } ( \dot { \mathbf { X } _ { h } ^ { l } } )$ as a mixture of Gaussians with learnable parameters [29]. Concretely, set $\begin{array} { r } { \mathbb { Q } ( \mathbf { Z } ^ { l } ) \sim \sum _ { i = 1 } ^ { m } w _ { i } } \end{array}$ Gaussian $( \mu _ { 0 , i } , \sigma _ { 0 , i } ^ { 2 } )$ and $\begin{array} { r } { \mathbb { Q } ( \mathbf { X } _ { h } ^ { l } ) \sim \sum _ { i = 1 } ^ { m } \hat { w } _ { i } } \end{array}$ Gaussian(ˆµ<sub>0,i</sub>, σˆ<sup>2</sup><sub>0,i</sub>), where $w _ { i } , \mu _ { 0 , i } , \sigma _ { 0 , i }$ <sub>i</sub> and $\hat { w } _ { i } , \hat { \mu } _ { 0 , i } , \hat { \sigma } _ { 0 , i }$ <sub>i</sub> are learnable parameters. Based on the aggregated $\mathbf { Z } ^ { l }$ and $\mathbf { X } ^ { l }$ , the parameters are calculated and samples are drawn from a Gaussian distribution to obtain $\mathbb { P } ( \mathbf { Z } ^ { l } \mid$ $\mathbf { X } ^ { l - 1 } , \mathbf { H } ) \sim \mathrm { G a u s s i a n } ( \mu _ { l } , \sigma _ { l } ^ { 2 } )$ and $\mathbb { P } ( \mathbf { X } _ { h } ^ { l } \mid \mathbf { Z } ^ { l } , \mathbf { H } ) \sim \mathrm { G a u s s i a n } ( \hat { \mu } _ { l } , \hat { \sigma } _ { l } ^ { 2 } )$ . The estimation of $\mathbf { \epsilon } _ { Z \mathrm { I B } } l$ and $\mathrm { X I B } ^ { l }$ can be written as:

$$
\widehat { \mathbf { Z } \mathbf { I } \mathbf { B } } ^ { l } = \log ( \Phi ( \mathbf { Z } ^ { l } ; \mu _ { l } , \sigma _ { l } ^ { 2 } ) ) - \log ( \sum _ { i = 1 } ^ { m } w _ { i } \Phi ( \mathbf { Z } ^ { l } ; \mu _ { 0 , i } , \sigma _ { 0 , i } ^ { 2 } ) ) ,\tag{8}
$$

$$
\widehat { \mathbf { X I B } } ^ { l } = \log ( \Phi ( \mathbf { X } _ { h } ^ { l } ; \hat { \mu } _ { l } , \hat { \sigma } _ { l } ^ { 2 } ) ) - \log ( \sum _ { i = 1 } ^ { m } \hat { w } _ { i } \Phi ( \mathbf { X } _ { h } ^ { l } ; \hat { \mu } _ { 0 , i } , \hat { \sigma } _ { 0 , i } ^ { 2 } ) ) .\tag{9}
$$

Plugging the estimations of both mutual information terms in $\operatorname { E q . } ( 3 )$ , the loss function for the information bottleneck-guided adaptive hypergraph convolution in the l-th layer can be obtained:

$$
\mathcal { L } _ { H } ^ { l } = \mathcal { L } _ { C E } ( \hat { y } _ { z } ^ { l } , Y ) + \mathcal { L } _ { C E } ( \hat { y } _ { x } ^ { l } , Y ) + \gamma _ { 1 } \widehat { [ \mathbf { Z } \mathbf { I } \mathbf { B } } ^ { l } + \widehat { \mathbf { X } \mathbf { I } \mathbf { B } } ^ { l } ] .\tag{10}
$$

## 3.2 Transformer Encoder

Hypergraph-based learning methods struggle to capture long-range dependencies, whereas the selfattention mechanism in the Transformer encoder can effectively model both short- and long-range dependencies between ROIs in the brain network. Formally, in the l-th layer, a multi-head selfattention mechanism is used to learn global representation $\mathbf { X } _ { t } ^ { \bar { l } }$

$$
\mathbf { X } _ { t } ^ { l } = \left( \left. _ { m = 1 } ^ { M } \mathbf { h } ^ { l , m } \right) \mathbf { W } _ { O } ^ { l } \right) , \quad \mathbf { h } ^ { l , m } = \operatorname { s o f t m a x } \left( \frac { \mathbf { W } _ { Q } ^ { l , m } \mathbf { X } ^ { l - 1 } \left( \mathbf { W } _ { K } ^ { l , m } \mathbf { X } ^ { l - 1 } \right) ^ { \top } } { \sqrt { d _ { K } ^ { l , m } } } \right) \mathbf { W } _ { V } ^ { l , m } \mathbf { X } ^ { l - 1 } ,\tag{11}
$$

where ∥ denotes the concatenation operator, M is the number of heads, $\mathbf { W } _ { O } ^ { l } , \mathbf { W } _ { Q } ^ { l , m } , \mathbf { W } _ { K } ^ { l , m }$ , and $\mathbf { W } _ { V } ^ { l , m }$ are learnable model parameters, and $d _ { K } ^ { l , m }$ is the first dimension of $\mathbf { W } _ { K } ^ { l , m }$

## 3.3 Information Bottleneck-Guided Node-Level Adaptive Fusion

Considering the varying roles of each ROI in the brain network and their different tendencies in capturing various dependencies [30], the information bottleneck-guided node-level adaptive fusion learns an independent weight for each node to balance the high-order information from the information bottleneck-guided adaptive hypergraph convolution and the short- and long-range dependencies information from the Transformer encoder.

## 3.3.1 Node-Level Adaptive Fusion

In the l-th layer of the information bottleneck-guided adaptive hypergraph Transformer, node-level adaptive fusion integrates the high-order representation $\mathbf { X } _ { h , i } ^ { \bar { l } }$ and global representation $\mathbf { X } _ { t , \ i } ^ { l }$ for each node $v _ { i } .$ , resulting in the updated node features $\mathbf { X } _ { i } ^ { l }$ , which can be formulated as follows:

$$
\mathbf { X } _ { i } ^ { l } = \theta _ { i } ^ { l } \mathbf { X } _ { h , i } ^ { l } + ( 1 - \theta _ { i } ^ { l } ) \mathbf { X } _ { t , i } ^ { l } ,\tag{12}
$$

where $\theta _ { i } ^ { l }$ denotes the learnable fusion weight of node $v _ { i }$

## 3.3.2 Optimization Based on the IB Principle

There is redundancy between the high-order representation of the brain network $\mathbf { X } _ { h } ^ { l }$ and the global representation of the brain network $\mathbf { X } _ { t } ^ { l }$ that contains short- and long-range dependencies. To reduce redundant information while preserving effective information in the updated node features $\mathbf { X } ^ { l }$ , we apply the IB principle to optimize the fusion process. The optimization objective is:

$$
\operatorname* { m i n } _ { \mathbb { P } ( \mathbf { X } ^ { l } , \mathbf { X } _ { h } ^ { l } , \mathbf { X } _ { t } ^ { l } ) \in \Omega } - I ( Y ; \mathbf { X } ^ { l } ) + \gamma _ { 2 } I ( \mathbf { X } _ { h } ^ { l } , \mathbf { X } _ { t } ^ { l } ; \mathbf { X } ^ { l } ) .\tag{13}
$$

Minimizing $- I ( Y ; \mathbf { X } ^ { l } )$ ensures that the fused features effectively retain task-relevant information, while minimizing $I ( \mathbf { X } _ { h } ^ { l } , \mathbf { X } _ { t } ^ { l } ; \mathbf { X } ^ { l } )$ removes redundant information. The trade-off hyperparameter $\gamma _ { 2 }$ is to balance the weights of two items. In the l-th layer, $\mathbf { X } ^ { l }$ is read out and mapped to the predicted label $\hat { y } _ { f } ^ { l }$ through a learnable parameter matrix. Similar to Section 3.1.2, the minimization of the first term in Eq.(13) can be replaced by the cross-entropy loss, i.e.,

$$
I ( Y ; { \bf X } ^ { l } ) \to - \mathcal { L } _ { C E } ( \hat { y } _ { f } ^ { l } , Y ) .\tag{14}
$$

The second term in Eq.(13) has an optimizable upper bound, which is given as follows:

Proposition 3. For any distribution $\mathbb { Q } ( \mathbf { X } ^ { l } )$ , we have

$$
I ( \mathbf { X } _ { h } ^ { l } , \mathbf { X } _ { t } ^ { l } ; \mathbf { X } ^ { l } ) \leq D _ { K L } \left( \mathbb { P } ( \mathbf { X } ^ { l } \mid \mathbf { X } _ { h } ^ { l } , \mathbf { X } _ { t } ^ { l } ) \mid \mid \mathbb { Q } ( \mathbf { X } ^ { l } ) \right) .
$$

The proof of the proposition can be found in Appendix B. To specify the upper bound for $I ( \mathbf { X } _ { h } ^ { l ^ { \prime } } , \mathbf { X } _ { t } ^ { l } ; \mathbf { X } ^ { l } )$ , we set $\mathbb { Q } ( \mathbf { X } ^ { l } )$ as a mixture of Gaussians with learnable parameters: $\mathbb { Q } ( \mathbf { X } ^ { l } ) \sim$ $\textstyle \sum _ { i = 1 } ^ { m ^ { * } } { \tilde { w } } _ { i }$ Gaussian $( \tilde { \mu } _ { 0 , i } , \tilde { \sigma } _ { 0 , i } ^ { 2 } )$ . The parameters are computed based on the fused representation $\mathbf { X } ^ { l }$ , and sampling is performed from a Gaussian distribution to obtain $\mathbb { P } ( \mathbf { X } ^ { l } \ \mid \ \mathbf { X } _ { h } ^ { l } , \mathbf { X } _ { t } ^ { l } ) \ \sim$ Gaussian $( \tilde { \mu } _ { l } , \tilde { \sigma } _ { l } ^ { 2 } )$ ). Thus, the estimation of $I ( \mathbf { X } _ { h } ^ { l } , \mathbf { X } _ { t } ^ { l } ; \mathbf { X } ^ { l } )$ is written as:

$$
I ( { \bf X } _ { h } ^ { l } , { \bf X } _ { t } ^ { l } ; { \bf X } ^ { l } ) \to \log ( \Phi ( { \bf X } ^ { l } ; \tilde { \mu } _ { l } , \tilde { \sigma } _ { l } ^ { 2 } ) ) - \log ( \sum _ { i = 1 } ^ { m } \tilde { w } _ { i } \Phi ( { \bf X } ^ { l } ; \tilde { \mu } _ { 0 , i } , \tilde { \sigma } _ { 0 , i } ^ { 2 } ) ) .\tag{15}
$$

By plugging the estimations of both mutual information terms into Eq.(13), the loss function for the node-level adaptive fusion process guided by IB principle in the l-th layer can be obtained:

$$
\mathcal { L } _ { F } ^ { l } = \mathcal { L } _ { C E } ( \hat { y } _ { f } ^ { l } , Y ) + \gamma _ { 2 } ( \log ( \Phi ( \mathbf { X } ^ { l } ; \tilde { \mu } _ { l } , \tilde { \sigma } _ { l } ^ { 2 } ) ) - \log ( \sum _ { i = 1 } ^ { m } \tilde { w } _ { i } \Phi ( \mathbf { X } ^ { l } ; \tilde { \mu } _ { 0 , i } , \tilde { \sigma } _ { 0 , i } ^ { 2 } ) ) ) .\tag{16}
$$

## 3.4 Information Bottleneck-Guided Adaptive Hypergraph Transformer Layer

The information bottleneck-guided adaptive hypergraph Transformer layer is composed of parallel information bottleneck-guided adaptive hypergraph convolution (IBAHGConv) layer and Transformer encoder (TransEncoder) layer with information bottleneck-guided node-level adaptive fusion (IBNAFusion) module. In the l-th layer, the high order representation $\mathbf { X } _ { h } ^ { l }$ from the IBAHGConv<sup>l</sup> layer and the global representation $\mathbf { X } _ { t } ^ { l }$ from the TransEncoder<sup>l</sup> layer are adaptively aggregated by IBNAFusion<sup>l</sup> to obtain updated node features $\mathbf { X } ^ { l }$ , as shown in the following equation:

$$
{ \bf X } _ { h } ^ { l } = \mathrm { I B A H G C o n v } ^ { l } \left( { \bf X } ^ { l - 1 } , { \bf H } \right) ,\tag{17}
$$

$$
\mathbf { X } _ { t } ^ { l } = \operatorname { T r a n s E c o d e r } ^ { l } \left( \mathbf { X } ^ { l - 1 } \right) ,\tag{18}
$$

$$
\mathbf { X } ^ { l } = \mathrm { I B N A F u s i o n } ^ { l } \left( \mathbf { X } _ { h } ^ { l } , \mathbf { X } _ { t } ^ { l } \right) .\tag{19}
$$

After L layers, the final node embeddings $\mathbf { X } ^ { L }$ effectively capture information from various dependencies in the brain network.

## 3.5 Loss Function

By concatenating the node embeddings $\mathbf { X } ^ { L }$ , the graph-level representation ˆx is obtained, which is then fed into an MLP to predict the label $\hat { y } .$ . During training, the overall loss function $\mathcal { L }$ is composed of the cross-entropy loss function $\mathcal { L } _ { C E }$ for supervised disease prediction, the loss $\mathcal { L } _ { H }$ for the adaptive hypergraph convolution part, and the loss $\mathcal { L } _ { F }$ for the node-level adaptive fusion part:

$$
\mathcal { L } = \mathcal { L } _ { C E } ( \hat { y } , Y ) + \sum _ { l = 1 } ^ { L } ( \lambda _ { 1 } \mathcal { L } _ { H } ^ { l } + \lambda _ { 2 } \mathcal { L } _ { F } ^ { l } ) ,\tag{20}
$$

where L denotes the number of layers in the information bottleneck-guided adaptive hypergraph Transformer, and $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are trade-off hyperparameters balancing different losses.

## 4 Experiments

## 4.1 Experiments Settings

Datasets and Preprocessing. The proposed method is evaluated on two brain network analysis related fMRI datasets, ABIDE [31] and ADNI [32] . The ABIDE dataset includes brain imaging data from 871 subjects collected across 17 international sites, with 403 subjects diagnosed with Autism Spectrum Disorder (ASD). A subset of the ADNI dataset is used in this study, consisting of 66 patients with Alzheimer’s Disease (AD) and 139 Normal Controls (NCs).

We preprocess the fMRI data using the standard steps provided by the Data Processing Assistant for Resting-State Function (DPARSF) toolkit [33], then use the Anatomical Automatic Labeling (AAL) brain template [34] to divide the brain space into ROIs and extract the corresponding BOLD signals. The FC matrix is obtained by calculating the Pearson correlation coefficients between the ROIs.

Metrics. Four machine learning and medical diagnostic-specific metrics are used: accuracy (ACC), sensitivity (SEN), specificity (SPE) and area under the receiver operating characteristic curve (AUC). We record the mean and standard deviation across 10 random runs on the test dataset.

Implementation Details. In the proposed method, we set the value of k to 3 in the KNN hypergraph modeling. The number of layers L in the adaptive hypergraph Transformer and the number of heads M in the multi-head attention module are set to 2 and 4, respectively. For all datasets, we randomly divide the training set, evaluation set, and test set by the ratio of 7 : 1 : 2. During training, we use the Adam optimizer [35], with an initial learning rate set to $1 \times 1 0 ^ { - 4 }$ . The number of epochs is set to 300. All our experiments are implemented in PyTorch and trained on one NVIDIA 4090.

## 4.2 Comparison Experiments

Baselines. The selected baselines correspond to three categories. The first category includes CNNbased method BrainNetCNN [36]. The second category includes graph or hypergraph-based learning methods, such as ContrastPool [37], BrainGNN [22], BrainIB [28], HGNN<sup>+</sup> [38], and FC-HAT [15]. The third category consists of Transformer-based methods, including BrainNetTF [19], Com-BrainTF [39] and ALTER [40]. The codes are reproduced based on the released codes.

Results. Table 1 shows the comparison results between the proposed and baseline methods. On all datasets, IBAHGT significantly outperforms the baseline methods. The hypergraph-based methods achieved higher accuracy in most cases compared to graph-based methods, indicating that hypergraphs can capture high-order correlations that are beneficial for classification. Compared to these methods, our proposed approach further improves accuracy on the ABIDE dataset (ASD vs. NC: 77.6%) and the ADNI dataset (AD vs. NC: 86.3%). The performance improvement is attributed to our method, which captures the MIMR high-order correlations and long-range dependencies in the brain network, and condenses the information from both at the ROI level to obtain an efficient representation.

## 4.3 Ablation Study

To validate the effectiveness of information bottleneck-guided adaptive hypergraph convolution (IBAHGConv) and information bottleneck-guided adaptive node-level fusion (IBNAFusion), ablation study are performed on two datasets, with results shown in Table 2. Using only IBAHGConv or the Transformer encoder (TransEncoder) performs worse than the combined method, indicating that integrating high-order information with long-range dependency information for comprehensive analysis enables a more thorough capture of abnormal changes in brain networks. On the ABIDE dataset, replacing IBAHGConv with $\mathrm { H G N N ^ { + } }$ and AHGConv (without the HIB principle) led to a decrease in accuracy by 3.5% and 2.5%, respectively. This demonstrates that IBAHGConv optimizes the message-passing process and is more effective at capturing MIMR high-order information. Furthermore, diagnostic accuracy on the ABIDE dataset decreases when directly average fusion or NAFusion without IB principle, compared to IBNAFusion, which demonstrates the effectiveness of information concise during fusion at the node level guided by the IB principle.

Table 1: Experimental results of the comparison methods.
<table><tr><td>Datasets (Tasks)</td><td colspan="4">ABIDE (ASD vs. NC)</td><td colspan="4">ADNI (AD vs. NC)</td></tr><tr><td>Method</td><td>ACC (%)</td><td>SPE (%)</td><td>SEN (%)</td><td>AUC (%)</td><td>ACC (%)</td><td>SPE (%)</td><td>SEN (%)</td><td>AUC (%)</td></tr><tr><td>BrainNetCNN</td><td> $6 6 . 2 \pm 1 . 9$ </td><td></td><td>74.3±4.1 56.8±3.1</td><td>68.3±2.6</td><td>73.2±2.7</td><td>79.3±4.2</td><td> $6 0 . 0 { \pm } 5 . 8 $ </td><td> $6 6 . 3 { \pm } 4 . 7$ </td></tr><tr><td>ContrastPool</td><td> $6 3 . 0 { \pm } 2 . 5 $ </td><td></td><td></td><td>71.1±8.3 53.5±4.6 68.1±7.7 70.7±4.1 72.1±5.7</td><td></td><td></td><td> $6 7 . 7 { \pm } 1 0 . 2 $ </td><td> $6 9 . 5 { \pm } 3 . 9 $ </td></tr><tr><td>BrainGNN</td><td> $6 4 . 5 { \pm } 2 . 7 $ </td><td></td><td></td><td>72.6±7.7 55.0±5.5 69.7±7.2</td><td>73.2±4.1</td><td> $7 4 . 3 { \pm } 4 . 2 $ </td><td> $7 0 . 8 { \pm } 1 2 . 3 $ </td><td> $7 1 . 3 { \pm } 6 . 1 $ </td></tr><tr><td>BrainIB</td><td> $6 8 . 3 { \pm } 1 . 9 $ </td><td></td><td></td><td>76.4±5.4 58.8±5.0 70.2±1.9</td><td>74.1±2.5 78.6±5.1</td><td></td><td> $6 4 . 6 { \pm } 1 1 . 5 $ </td><td> $6 9 . 0 { \pm } 6 . 5 $ </td></tr><tr><td>HGNN+</td><td> $6 8 . 9 { \pm } 2 . 1 $ </td><td></td><td></td><td></td><td></td><td></td><td>77.0±4.4 59.3±5.9 71.0±1.9 74.6±2.9 81.4±4.2 60.0±5.8</td><td> $6 9 . 2 { \pm } 6 . 8 $ </td></tr><tr><td>FC-HAT</td><td> $7 0 . 7 \pm 3 . 5$ </td><td></td><td></td><td></td><td></td><td></td><td>75.5±5.0 65.0±6.5 72.0±4.8 77.1±3.7 84.3±3.6 61.5±4.9</td><td> $6 9 . 1 \pm 2 . 0$ </td></tr><tr><td>BrainNetTF</td><td> $7 1 . 4 { \pm } 1 . 1$ </td><td></td><td></td><td></td><td></td><td></td><td>76.8±5.0 65.0±7.2 72.2±1.4 78.0±2.7 85.7±2.3 61.5±4.9</td><td> $7 0 . 6 \pm 1 . 8$ </td></tr><tr><td>Com-BrainTF</td><td> $7 2 . 1 \pm 1 . 5$ </td><td></td><td></td><td></td><td></td><td></td><td>74.3±3.3 69.5±6.0 73.3±2.4 79.0±3.386.4±2.7 63.1±5.8</td><td> $7 1 . 4 \pm 2 . 5$ </td></tr><tr><td>ALTER</td><td> $7 4 . 4 \pm 1 . 2$ </td><td></td><td></td><td>78.5±1.8 69.5±3.0 75.3±1.9 81.5±2.5 87.9±1.8</td><td></td><td></td><td> $6 7 . 7 { \pm } 5 . 8 $ </td><td> $7 1 . 9 { \pm } 2 . 8 $ </td></tr><tr><td>IBAHGT</td><td> ${ \bf 7 7 . 6 { \pm 0 . 8 } }$ </td><td> $\mathbf { 8 0 . 4 \pm 3 . 1 }$ </td><td> $\mathbf { 7 4 . 3 \pm 3 . 8 }$ </td><td> $\mathbf { 8 0 . 5 \pm 2 . 0 }$ </td><td> ${ \bf 8 6 . 3 \pm 1 . 2 }$ </td><td> ${ \bf 8 8 . 6 \pm 4 . 7 }$ </td><td> $\mathbf { 8 1 . 5 \pm 7 . 9 }$ </td><td> $\mathbf { 8 } 2 . \mathbf { 8 } \pm 5 . 2$ </td></tr></table>

Table 2: Experimental results of the ablation study.
<table><tr><td>Datasets (Tasks)</td><td colspan="4">ABIDE (ASD vs. NC)</td><td colspan="4">ADNI (AD vs. NC)</td></tr><tr><td>Method</td><td>ACC (%)</td><td>SPE (%)</td><td>SEN (%)</td><td>AUC (%)</td><td>ACC (%)</td><td>SPE (%)</td><td>SEN (%)</td><td>AUC (%)</td></tr><tr><td>TransEnoder</td><td>69.3±0.9</td><td>77.9±4.5</td><td>59.3±4.2</td><td>72.7±5.3</td><td>76.1±2.8</td><td>77.9±6.9</td><td>72.3±10.4</td><td>75.6±5.8</td></tr><tr><td>IBAHGconv</td><td>70.9±1.3</td><td>81.7±3.3</td><td>58.3±2.9</td><td>73.4±5.1</td><td>78.0±2.2</td><td>82.9±5.3</td><td>67.7±10.2</td><td>77.3±3.8</td></tr><tr><td>IBAHGT w/o IBAHGCony</td><td>74.1±1.4 76.6±4.9</td><td></td><td>71.3±6.4</td><td>77.8±3.4</td><td>81.9±2.5</td><td>84.3±3.6</td><td>76.9±8.4</td><td>80.4±2.7</td></tr><tr><td>IBAHGT w/o HIB</td><td></td><td>75.1±1.5 78.1±4.1 71.5±6.5</td><td></td><td>79.2±2.3</td><td>83.4±2.4</td><td>84.3±3.6</td><td>81.5±7.9</td><td>81.0±5.1</td></tr><tr><td>IBAHGT w/o IBNAFusion</td><td>73.0±1.3 77.5±4.6 67.8±6.477.4±3.5</td><td></td><td></td><td></td><td>81.0±1.8</td><td></td><td>85.7±3.9 70.8±5.8</td><td>80.3±3.9</td></tr><tr><td>IBAHGT w/o IB</td><td>74.8±1.2 76.8±4.7 72.5±7.3 78.9±2.3 83.9±2.0</td><td></td><td></td><td></td><td></td><td></td><td>85.7±3.2 80.0±6.2</td><td>81.8±4.6</td></tr><tr><td>IBAHGT</td><td>77.6±0.8 80.4±3.1 74.3±3.8 80.5±2.0 86.3±1.2 88.6±4.7 81.5±7.9</td><td></td><td></td><td></td><td></td><td></td><td></td><td>82.8±5.2</td></tr></table>

## 4.4 Interpretation Analysis

Discriminative Hyperedge Analysis We analyze the discriminative connectivity of brain networks in diseased and normal groups for each classification task, from a high-order correlation perspective. By performing a t-test on the hyperedge features, we identify significant hyperedges, which are visualized in Fig.3. For ASD vs. NC, ASD patients show missing connections with right middle frontal gyrus (MFG.R) as well as right middle temporal gyrus (MTG.R), which foreshadows defects in individual external goal-related behavior and self-consciousness, further validating the conclusions of the previous study [41]. For AD vs. NC, it can be observed that the high-order correlarions represented by the hyperedge in AD patients show missing connections with PHG.R and HIP.R. Previous studies have indicated that functional network impairments in AD patients often occur in the PHG and HIP, which affect memory [42].

Analysis of Fusion Weights for Important ROIs To evaluate the influence of ROIs on the classification tasks, we measure the significance of the nodes using the t-test method and visualize the top 10 most important ROIs with the smallest p-values, displaying the fusion weights of each ROI through color differences. For ASD vs. NC, the identified important ROIs include the left precuneus (PCUN.L), right rectus (REC.R), right middle temporal gyrus (MTG.R), superior frontal gyrus, medial (SFGmed.L) and IPL.R, etc. As mentioned in [43], the activations in PCUN and REC will be reduced when ASD patients inference mental states themselves or others. In addition, we found that the fusion weight of the MTG.R is larger, indicating a greater inclination towards high-order correlations. Previous studies have shown that MTG is critical for high-order cognitive functions, especially in language comprehension through collaboration with multiple other ROIs [44]. The fusion weight of the PCUN.L is relatively small, indicating that PCUN is more inclined to capture long-range dependencies. For AD vs. NC, the identified important ROIs include the HIP.R, left superior frontal gyrus, medial (SFGmed.L), right inferior frontal gyrus, triangular part (IFGtriang.R), right amygdala (AMYG.R), and left caudate nucleus (CAU.L), etc. The fusion weight of the HIP is relatively small, indicating that it relies more on long-range dependencies. Previous research has reported that low-frequency activity in the hippocampus can drive functional connectivity between the cortices of the two hemispheres [45], highlighting the HIP’s key role in long-distance communication.

(b) Influence of two key hyperparameters for model performance  
![](images/a8505571e07ab5f2f915f6eaebe23007820c333746d7b3218c8ee1891453887b.jpg)

![](images/8152f562f5d1aa2ec0e80998c300008da37f207ff54436160ccb30cb4ee28c20.jpg)  
(a) ASD vs .NC

![](images/6a46b254f6fa1c239d0fac6907e1df234f393cc3ffb7e37f33176485742421f0.jpg)

![](images/2afd638d08cd71286a3c1369e851a8b51e9c0e7dc3dfe2cf203166c3d08e7db7.jpg)  
(b) AD vs .NC

Figure 3: Visualization of discriminative hyperedges for each classification. In each figure, the blue balls are the ROI contained in the hyperedge and the green balls are the center of the involved ROIs.  
![](images/2ab8f20502d6baef4b12aa27ea488fcf5df4efd263e1f31abda817aaec31df0e.jpg)

![](images/b41092a127987776c9fc415e58b76a3326b5afa73e47927f3930004d011d5458.jpg)

![](images/24315392104c3ed555255532cfad9345b7fa01a462d06cf2300ed011dea8ee96.jpg)  
Figure 4: Visualization of important ROIs with their fusion weights (Color bar represents the magnitude of the fusion weights) and the hyperparameters influence.

## 4.5 Hyperparameter Sensitivity Analysis

To analyze how the trade-off hyperparameters $\gamma _ { 1 }$ and $\gamma _ { 2 }$ affect model performance, we perform a hyperparameter search and plot the optimal model performance on the ABIDE and ADNI datasets for each case where $\gamma _ { 1 } \in \{ 0 . 0 0 0 1 , 0 . 0 0 1 , 0 . 0 1 , 0 . 1 , 1 \}$ and $\gamma _ { 2 } \in \{ 0 . 0 0 0 1 , 0 . 0 0 1 , 0 . 0 1 , 0 . 1 , 1 \}$ , as shown in Fig.4(b). We observe that as the magnitude of $\gamma _ { 1 }$ increases, the accuracy on both datasets initially rises and then decreases, reaching a peak at $\gamma _ { 1 } = 0 .$ 1. A similar trend is observed with $\gamma _ { 2 }$ , where the IBAHGT achieves the highest classification accuracy when $\gamma _ { 2 } = 0 . 1$

## 5 Conclusion

In this paper, the proposed IBAHGT learns the MIMR high-order correlations and short- and longrange dependencies in brain networks within a unified framework. It suppresses redundancy while enabling fine-grained fusion of multi-source information to obtain efficient representations for brain disease diagnosis. The information bottleneck-guided adaptive hypergraph convolution introduces the HIB principle, adaptively learning message-passing weights to capture MIMR high-order information. Information bottleneck-guided node-level fusion assigns weights based on each node’s information preference, effectively integrating high-order and global information. The proposed method achieves state-of-the-art performance on two datasets, offering new perspectives for understanding ROIs’ interactions and brain information processing patterns.

## References

[1] Andrea Avena-Koenigsberger, Bratislav Misic, and Olaf Sporns. Communication dynamics in complex brain networks. Nature reviews neuroscience, 19(1):17–33, 2018.

[2] Jin Liu, Min Li, Yi Pan, Wei Lan, Ruiqing Zheng, Fang-Xiang Wu, and Jianxin Wang. Complex brain network analysis and its applications to brain disorders: a survey. Complexity, 2017(1):8362741, 2017.

[3] Kaustubh Supekar, Vinod Menon, Daniel Rubin, Mark Musen, and Michael D Greicius. Network analysis of intrinsic functional brain connectivity in alzheimer’s disease. PLoS computational biology, 4(6):e1000100, 2008.

[4] Austin R Benson, David F Gleich, and Jure Leskovec. Higher-order organization of complex networks. Science, 353(6295):163–166, 2016.

[5] Hae-Jeong Park and Karl Friston. Structural and functional brain networks: from connections to cognition. Science, 342(6158):1238411, 2013.

[6] Pablo Barttfeld, Bruno Wicker, Sebastián Cukier, Silvana Navarta, Sergio Lew, and Mariano Sigman. A big-world network in asd: dynamical connectivity analysis reflects a deficit in longrange connections and an excess of short-range connections. Neuropsychologia, 49(2):254–263, 2011.

[7] Christoph Von Der Malsburg. The correlation theory of brain function. In Models of neural networks: Temporal aspects of coding and information processing in biological systems, pages 95–119. Springer, 1994.

[8] Beatriz Luna and John A Sweeney. The emergence of collaborative brain function: Fmri studies of the development of response inhibition. Annals of the New York Academy of Sciences, 1021(1):296–309, 2004.

[9] Dina R Dajani and Lucina Q Uddin. Local brain connectivity across development in autism spectrum disorder: A cross-sectional investigation. Autism Research, 9(1):43–54, 2016.

[10] Gustavo Deco, Yonathan Sanz Perl, Peter Vuust, Enzo Tagliazucchi, Henry Kennedy, and Morten L Kringelbach. Rare long-range cortical connections enhance human information processing. Current Biology, 31(20):4436–4448, 2021.

[11] Caio Seguin, Olaf Sporns, and Andrew Zalesky. Brain network communication: concepts, models and applications. Nature reviews neuroscience, 24(9):557–574, 2023.

[12] Alaa Bessadok, Mohamed Ali Mahjoub, and Islem Rekik. Graph neural networks in network neuroscience. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5):5833– 5848, 2022.

[13] Gang Qu, Wenxing Hu, Li Xiao, Junqi Wang, Yuntong Bai, Beenish Patel, Kun Zhang, and Yu-Ping Wang. Brain functional connectivity analysis via graphical deep learning. IEEE Transactions on Biomedical Engineering, 69(5):1696–1706, 2021.

[14] Ju Niu and Yuhui Du. Applications of hypergraph-based methods in classifying and subtyping psychiatric disorders: a survey. Radiology Science, 2(01):83–95, 2023.

[15] Junzhong Ji, Yating Ren, and Minglong Lei. Fc–hat: Hypergraph attention network for functional brain network classification. Information Sciences, 608:1301–1316, 2022.

[16] Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, and Yue Gao. Hypergraph neural networks. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 3558–3565, 2019.

[17] Song Bai, Feihu Zhang, and Philip HS Torr. Hypergraph convolution and hypergraph attention. Pattern Recognition, 110:107637, 2021.

[18] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[19] Xuan Kan, Wei Dai, Hejie Cui, Zilong Zhang, Ying Guo, and Carl Yang. Brain network transformer. Advances in Neural Information Processing Systems, 35:25586–25599, 2022.

[20] Yanwu Yang, Chenfei Ye, Xutao Guo, Tao Wu, Yang Xiang, and Ting Ma. Mapping multimodal brain connectome for brain disorder diagnosis via cross-modal mutual learning. IEEE Transactions on Medical Imaging, 43(1):108–121, 2023.

[21] Hongting Ye, Yalu Zheng, Yueying Li, Ke Zhang, Youyong Kong, and Yonggui Yuan. Rhbrainfs: regional heterogeneous multimodal brain networks fusion strategy. Advances in Neural Information Processing Systems, 36:59286–59303, 2023.

[22] Xiaoxiao Li, Yuan Zhou, Nicha Dvornek, Muhan Zhang, Siyuan Gao, Juntang Zhuang, Dustin Scheinost, Lawrence H Staib, Pamela Ventola, and James S Duncan. Braingnn: Interpretable brain graph neural network for fmri analysis. Medical Image Analysis, 74:102233, 2021.

[23] Junqi Wang, Hailong Li, Gang Qu, Kim M Cecil, Jonathan R Dillman, Nehal A Parikh, and Lili He. Dynamic weighted hypergraph convolutional network for brain functional connectome analysis. Medical image analysis, 87:102828, 2023.

[24] Naftali Tishby, Fernando C Pereira, and William Bialek. The information bottleneck method. arXiv preprint physics/0004057, 2000.

[25] Yawei Luo, Ping Liu, Tao Guan, Junqing Yu, and Yi Yang. Significance-aware information bottleneck for domain adaptive semantic segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6778–6787, 2019.

[26] Xue Bin Peng, Angjoo Kanazawa, Sam Toyer, Pieter Abbeel, and Sergey Levine. Variational discriminator bottleneck: Improving imitation learning, inverse rl, and gans by constraining information flow. arXiv preprint arXiv:1810.00821, 2018.

[27] Rundong Wang, Xu He, Runsheng Yu, Wei Qiu, Bo An, and Zinovi Rabinovich. Learning efficient multi-agent communication: An information bottleneck approach. In International conference on machine learning, pages 9908–9918. PMLR, 2020.

[28] Kaizhong Zheng, Shujian Yu, Baojuan Li, Robert Jenssen, and Badong Chen. Brainib: Interpretable brain network-based psychiatric diagnosis with graph information bottleneck. IEEE Transactions on Neural Networks and Learning Systems, 2024.

[29] Nat Dilokthanakul, Pedro AM Mediano, Marta Garnelo, Matthew CH Lee, Hugh Salimbeni, Kai Arulkumaran, and Murray Shanahan. Deep unsupervised clustering with gaussian mixture variational autoencoders. arXiv preprint arXiv:1611.02648, 2016.

[30] James M Shine, Patrick G Bissett, Peter T Bell, Oluwasanmi Koyejo, Joshua H Balsters, Krzysztof J Gorgolewski, Craig A Moodie, and Russell A Poldrack. The dynamics of functional brain networks: integrated network states during cognitive task performance. Neuron, 92(2):544– 554, 2016.

[31] Anibal Sólon Heinsfeld, Alexandre Rosa Franco, R Cameron Craddock, Augusto Buchweitz, and Felipe Meneguzzi. Identification of autism spectrum disorder using deep learning and the abide dataset. NeuroImage: Clinical, 17:16–23, 2018.

[32] Clifford R Jack Jr, Matt A Bernstein, Nick C Fox, Paul Thompson, Gene Alexander, Danielle Harvey, Bret Borowski, Paula J Britson, Jennifer L. Whitwell, Chadwick Ward, et al. The alzheimer’s disease neuroimaging initiative (adni): Mri methods. Journal ofMagnetic Resonance Imaging: An Official Journal of the International Society for Magnetic Resonance in Medicine, 27(4):685–691, 2008.

[33] Xiao-Wei Song, Zhang-Ye Dong, Xiang-Yu Long, Su-Fang Li, Xi-Nian Zuo, Chao-Zhe Zhu, Yong He, Chao-Gan Yan, and Yu-Feng Zang. Rest: a toolkit for resting-state functional magnetic resonance imaging data processing. PloS one, 6(9):e25031, 2011.

[34] Edmund T Rolls, Chu-Chung Huang, Ching-Po Lin, Jianfeng Feng, and Marc Joliot. Automated anatomical labelling atlas 3. Neuroimage, 206:116189, 2020.

[35] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[36] Jeremy Kawahara, Colin J Brown, Steven P Miller, Brian G Booth, Vann Chau, Ruth E Grunau, Jill G Zwicker, and Ghassan Hamarneh. Brainnetcnn: Convolutional neural networks for brain networks; towards predicting neurodevelopment. NeuroImage, 146:1038–1049, 2017.

[37] Jiaxing Xu, Qingtian Bian, Xinhang Li, Aihu Zhang, Yiping Ke, Miao Qiao, Wei Zhang, Wei Khang Jeremy Sim, and Balázs Gulyás. Contrastive graph pooling for explainable classification of brain networks. IEEE Transactions on Medical Imaging, 2024.

[38] Yue Gao, Yifan Feng, Shuyi Ji, and Rongrong Ji. Hgnn+: General hypergraph neural networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(3):3181–3199, 2022.

[39] Anushree Bannadabhavi, Soojin Lee, Wenlong Deng, Rex Ying, and Xiaoxiao Li. Communityaware transformer for autism prediction in fmri connectome. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 287–297. Springer, 2023.

[40] Shuo Yu, Shan Jin, Ming Li, Tabinda Sarwar, and Feng Xia. Long-range brain graph transformer. Advances in Neural Information Processing Systems, 37:24472–24495, 2024.

[41] Anne Pankow, Teresa Katthagen, Sarah Diner, Lorenz Deserno, Rebecca Boehme, Nobert Kathmann, Tobias Gleich, Michael Gaebler, Henrik Walter, Andreas Heinz, et al. Aberrant salience is related to dysfunctional self-referential processing in psychosis. Schizophrenia bulletin, 42(1):67–76, 2016.

[42] Saul L Miller, Kim Celone, Kristina DePeau, Eli Diamond, Bradford C Dickerson, Dorene Rentz, Maija Pihlajamäki, and Reisa A Sperling. Age-related memory impairment associated with loss of parietal deactivation but preserved hippocampal activation. Proceedings of the National Academy ofSciences, 105(6):2181–2186, 2008.

[43] Rajesh K Kana, Jose O Maximo, Diane L Williams, Timothy A Keller, Sarah E Schipul, Vladimir L Cherkassky, Nancy J Minshew, and Marcel Adam Just. Aberrant functioning of the theory-of-mind network in children and adolescents with autism. Molecular autism, 6:1–12, 2015.

[44] Karalyn Patterson, Peter J Nestor, and Timothy T Rogers. Where do you know what you know? the representation of semantic knowledge in the human brain. Nature reviews neuroscience, 8(12):976–987, 2007.

[45] Russell W Chan, Alex TL Leong, Leon C Ho, Patrick P Gao, Eddie C Wong, Celia M Dong, Xunda Wang, Jufang He, Ying-Shing Chan, Lee Wei Lim, et al. Low-frequency hippocampal– cortical activity drives brain-wide resting-state functional mri connectivity. Proceedings ofthe National Academy ofsciences, 114(33):E6972–E6981, 2017.

[46] Iva Ilioska, Marianne Oldehinkel, Alberto Llera, Sidhant Chopra, Tristan Looden, Roselyne Chauvin, Daan Van Rooij, Dorothea L Floris, Julian Tillmann, Carolin Moessnang, et al. Connectome-wide mega-analysis reveals robust patterns of atypical functional connectivity in autism. Biological psychiatry, 94(1):29–39, 2023.

[47] Elysa J Marco, Leighton BN Hinkley, Susanna S Hill, and Srikantan S Nagarajan. Sensory processing in autism: a review of neurophysiologic findings. Pediatric research, 69(8):48–54, 2011.

[48] Nicholas Myers, Lorenzo Pasquini, Jens Göttler, Timo Grimmer, Kathrin Koch, Marion Ortner, Julia Neitzel, Mark Mühlau, Stefan Förster, Alexander Kurz, et al. Within-patient correspondence of amyloid-β and intrinsic network connectivity in alzheimer’s disease. Brain, 137(7):2052–2064, 2014.

[49] Yigal Agam, Robert M Joseph, Jason JS Barton, and Dara S Manoach. Reduced cognitive control of response inhibition by the anterior cingulate cortex in autism spectrum disorders. Neuroimage, 52(1):336–347, 2010.

[50] Qinghua Zhao, Hong Lu, Hichem Metmer, Will XY Li, and Jianfeng Lu. Evaluating functional connectivity of executive control network and frontoparietal network in alzheimer’s disease. Brain research, 1678:262–272, 2018.

[51] ADHD-200 consortium. The adhd-200 consortium: a model to advance the translational potential of neuroimaging in clinical neuroscience. Frontiers in systems neuroscience, 6:62, 2012.

## A Algorithm

The IBAHGT algorithm is described in Algorithm 1. To clearly present the process of our algorithm, the IBAHGT algorithm is described in the form of Algorithm 1. In the implementation process, we express the information-bottleneck-guided adaptive hypergraph convolution and the informationbottleneck-guided node-level adaptive fusion in matrix form to accelerate computation.

Algorithm 1: Information-Bottleneck-Guided Adaptive Hypergraph Transformer   
Input :Initial node features $\mathbf { X } ^ { 0 } ;$ Incidence matrix H   
Initialize : The learnable weight matrix $\mathbf { W } _ { 1 } ^ { l } , \mathbf { W } _ { 2 } ^ { l }$ ; The learnable vector $\mathbf { a } _ { 1 } ^ { l } , \mathbf { a } _ { 2 } ^ { l } , \theta ^ { l }$   
Output :The predicted class label yˆ   
1 for Layers $l = \dot { 1 } , 2 , \dots , L$ do   
2 Stage 1: Information-Bottleneck-Guided Adaptive Hypergraph Convolution   
3 Phase 1:Adaptive Gathering of Node Features to Hyperedges   
4 for $e _ { j } \in \bar { \mathcal { E } }$ do   
5 for $v _ { i } \in \mathcal V$ do   
6 Construct Node Set $\mathcal { N } _ { v } ( e _ { j } ) = \{ v _ { i } \in \mathcal { V } | \mathbf { H } _ { i , j } \neq 0 \}$   
7 $\begin{array} { r } { \alpha _ { : \textit { \doteq } } ^ { l } = \frac { \exp \big ( \mathbf { a } _ { 1 } ^ { l } \mathbf { W } _ { 1 } ^ { l } \mathbf { X } _ { i } ^ { l - 1 } \big ) } { \mathfrak { e } ^ { \mathrm { i } } } } \end{array}$   
i,j $\begin{array} { r } { \overrightarrow { \sum _ { v _ { k } \in \mathcal { N } _ { v } ( e _ { j } ) } \exp \left( \mathbf { a } _ { 1 } ^ { l } \mathbf { W } _ { 1 } ^ { l } \mathbf { X } _ { k } ^ { l - 1 } \right) } , } \end{array}$   
8 end   
9 $\begin{array} { r } { \mathbf Z _ { j } ^ { l } = \sigma \left( \sum _ { v _ { i } \in \mathcal { N } _ { v } ( e _ { j } ) } { \alpha _ { i , j } ^ { l } \mathbf W _ { 1 } ^ { l } \mathbf X _ { i } ^ { l - 1 } } \right) } \end{array}$   
10 end   
11 $\mu _ { l }  \mathbf { Z } ^ { l } [ 0 : f _ { z } ^ { l } ] ;$   
12 $\sigma _ { l } ^ { 2 } \gets$ softplus $\bar { ( } \mathbf { Z } ^ { l } [ f _ { z } ^ { l } : 2 f _ { z } ^ { l } ] ) \colon$   
13 P(Z<sup>l</sup> | X<sup>l−1</sup>, H) ∼ Gaussian $( \mu _ { l } , \sigma _ { l } ^ { 2 } )$   
14 Phase 2:Adaptive Aggregating of Hyperedge Features to Nodes   
15 for $v _ { i } \in \mathcal V$ do   
16 for $e _ { j } \in \mathcal { E }$ do   
17 Construct Hyperedge Set $\mathcal { N } _ { e } ( v _ { i } ) = \{ e _ { j } \in \mathcal { E } \mid \mathbf { H } _ { i , j } \neq 0 \}$   
$( \mathbf { a } _ { 2 } ^ { l } \mathbf { W } _ { 2 } ^ { l } \mathbf { Z } _ { j } ^ { l } )$   
18 $\begin{array} { r } { \beta _ { i , j } ^ { l } = \frac { \exp { ( \mathbf { a } _ { 2 } \mathbf { w } _ { \mathrm { 2 } } \mathbf { Z } _ { j } ) } } { \sum _ { e _ { p } \in \mathcal { N } _ { e } ( v _ { i } ) } \exp { ( \mathbf { a } _ { 2 } ^ { l } \mathbf { W } _ { 2 } ^ { l } \mathbf { Z } _ { p } ^ { l } ) } } ; } \end{array}$   
19 end   
20 $\begin{array} { r } { \mathbf { X } _ { h , i } ^ { l } = \sigma \left( \sum _ { e _ { j } \in \mathcal { N } _ { e } ( v _ { i } ) } \beta _ { i , j } ^ { l } \mathbf { W } _ { 2 } ^ { l } \mathbf { Z } _ { j } ^ { l } \right) } \end{array}$   
21 end   
22 $\hat { \mu } _ { l } \gets \mathbf { X } _ { h } ^ { l } [ 0 : f _ { h } ^ { l } ] ;$   
23 $\hat { \sigma } _ { l } ^ { 2 } \gets$ softplus $( \mathbf { \dot { X } } _ { h } ^ { l } [ f _ { h } ^ { l } : 2 f _ { h } ^ { l } ] )$ ;   
24 $\mathbb { P } ( \mathbf { X } _ { h } ^ { l } \mid \mathbf { Z } ^ { l } , \mathbf { H } ) \sim$ Gaussian $( \hat { \mu } _ { l } , \hat { \sigma } _ { l } ^ { 2 } )$   
25 Stage 2: Transformer Encoder   
26 $\mathbf { \Delta } \bar { \mathbf { X } } _ { t } ^ { l } \gets$ TransEncoder $\mathbf { \nabla } \cdot ( \mathbf { X } ^ { l - 1 } )$   
27 Stage 3: Information-Bottleneck-Guided Node-Level Adaptive Fusion   
28 for $v _ { i } \in \mathcal V$ do   
29 $\begin{array} { r } { \dot { \mathbf { X } } _ { i } ^ { l } = \theta _ { i } ^ { l } \mathbf { X } _ { h , i } ^ { l } + ( 1 - \theta _ { i } ^ { l } ) \mathbf { X } _ { t , i } ^ { l } ; } \end{array}$   
30 end   
31 $\tilde { \mu } _ { l } \gets \mathbf { X } ^ { l } [ 0 : f ^ { l } ] ;$   
32 $\tilde { \sigma } _ { l } ^ { 2 } \gets$ softplus $( \mathbf { X } ^ { l } [ f ^ { l } : 2 f ^ { l } ] ) ;$   
33 $\ddot { \mathbb { P } } ( \mathbf { X } ^ { l } \mid \mathbf { X } _ { h } ^ { l } , \mathbf { X } _ { t } ^ { l } ) \sim$ Gaussian $( \tilde { \mu } _ { l } , \tilde { \sigma } _ { l } ^ { 2 } )$   
34 end   
35 xˆ = Readout $( \mathbf { X } ^ { L } ) )$   
36 $\hat { y } = \mathbf { M } \mathbf { L P } ( \hat { \mathbf { x } } )$   
37 return yˆ;

![](images/1b81202fd408e96c564695b6676254dce9ec97be5f8d8f9cc4bb24fd24a8b134.jpg)  
Figure 5: (a) The Markovian chain of information bottleneck-guided adaptive hypergraph convolution. (b) The Markovian chain of information bottleneck-guided node-level adaptive fusion.

## B Proof

Proof for Proposition 1 We restate Proposition 1:

$$
I ( Y ; { \mathbf { Z } } ^ { l } , { \mathbf { X } } _ { h } ^ { l } ) = I ( Y ; { \mathbf { Z } } ^ { l } ) + I ( Y ; { \mathbf { X } } _ { h } ^ { l } \mid { \mathbf { Z } } ^ { l } ) .\tag{21}
$$

For any probabilistic distribution functions $\mathbb { Q } _ { 1 } ( Y \mid \mathbf { Z } ^ { l } ) , \mathbb { Q } _ { 1 } ( Y \mid \mathbf { Z } ^ { l } , \mathbf { X } _ { h } ^ { l } )$ , and $\mathbb { Q } _ { 2 } ( Y )$ , we can express $I ( Y ; \mathbf { Z } ^ { l } )$ and $I ( Y ; { \mathbf { X } } _ { h } ^ { l } \mid { \mathbf { Z } } ^ { l } )$ as follows:

$$
I ( Y ; \mathbf { Z } ^ { l } ) \geq 1 + \mathbb { E } \left[ \log \frac { \mathbb { Q } _ { 1 } ( Y \mid \mathbf { Z } ^ { l } ) } { \mathbb { Q } _ { 2 } ( Y ) } \right] + \mathbb { E } _ { \mathbb { P } ( Y ) \mathbb { P } ( \mathbf { Z } ^ { l } ) } \left[ \frac { \mathbb { Q } _ { 1 } ( Y \mid \mathbf { Z } ^ { l } ) } { \mathbb { Q } _ { 2 } ( Y ) } \right] ,\tag{22}
$$

$$
I ( Y ; \mathbf { X } _ { h } ^ { l } \mid \mathbf { Z } ^ { l } ) \geq 1 + \mathbb { E } _ { \mathbb { P } ( Y , \mathbf { X } _ { h } ^ { l } \mid \mathbf { Z } ^ { l } ) } \left[ \log \frac { \mathbb { Q } _ { 1 } ( Y \mid \mathbf { Z } ^ { l } , \mathbf { X } _ { h } ^ { l } ) } { \mathbb { Q } _ { 2 } ( Y ) } \right] + \mathbb { E } _ { \mathbb { P } ( Y ) \mathbb { P } ( \mathbf { X } _ { h } ^ { l } \mid \mathbf { Z } ^ { l } ) } \left[ \frac { \mathbb { Q } _ { 1 } ( Y \mid \mathbf { Z } ^ { l } , \mathbf { X } _ { h } ^ { l } ) } { \mathbb { Q } _ { 2 } ( Y ) } \right]\tag{23}
$$

The Nguyen, Wainright & Jordan’s bound is used here:

Lemma 1 For any two random variables $\mathbf { X } _ { 1 }$ and $\mathbf { X } _ { 2 }$ and any function g: $g ( \mathbf { X } _ { 1 } , \mathbf { X } _ { 2 } ) \in \mathbb { R }$ , we have

$$
I ( X _ { 1 } , X _ { 2 } ) \geq \mathbb { E } [ g ( X _ { 1 } , X _ { 2 } ) ] - \mathbb { E } _ { P ( X _ { 1 } ) P ( X _ { 2 } ) } \left[ \exp ( g ( X _ { 1 } , X _ { 2 } ) - 1 ) \right] .\tag{24}
$$

The above lemma is used to $( Y ; \mathbf { Z } ^ { l } )$ . Plugging in $1 + \log \frac { \mathbf { C a t } ( \hat { y } _ { z } ^ { l } ) } { \mathbb { P } ( Y ) }$ , the right hand side of Eq.(24) is substituted by the cross-entropy loss, i.e.,

$$
I ( Y ; { \bf Z } ^ { l } ) \to - \mathcal { L } _ { C E } ( \hat { y } _ { z } ^ { l } , Y ) .\tag{25}
$$

The above lemma is also used to $( Y ; \mathbf { X } _ { h } ^ { l } \mid \mathbf { Z } ^ { l } )$ . Plugging in $1 + \log \frac { \mathbf { C a t } ( \hat { y } _ { x } ^ { l } ) } { \mathbb { P } ( Y ) }$ , it can be transformed into the following cross-entropy loss, i.e.,

$$
I ( Y ; { \bf X } _ { h } ^ { l } \mid { \bf Z } ^ { l } )  - { \mathcal L } _ { C E } ( \hat { y } _ { x } ^ { l } , Y ) .\tag{26}
$$

Proof for Proposition 2 According to the Markovian chain in Fig. 5(a), we restate Proposition 2: For any distributions $\mathbb { Q } ( \mathbf { Z } ^ { l } )$ and $\mathbb { Q } ( \mathbf { \bar { X } } _ { h } ^ { l } )$ ), we have

$$
\begin{array} { r l } & { I ( \mathbf { X } ^ { l - 1 } , \mathbf { H } ; \mathbf { X } _ { h } ^ { l } , \mathbf { Z } ^ { l } ) \leq I ( \mathbf { X } ^ { l - 1 } , \mathbf { H } ; \mathbf { Z } ^ { l } ) + I ( \mathbf { Z } ^ { l } , \mathbf { H } ; \mathbf { X } _ { h } ^ { l } ) } \\ & { \qquad = \mathbb { E } \left( \log \frac { \mathbb { P } ( \mathbf { Z } ^ { l } \mid \mathbf { X } ^ { l - 1 } , \mathbf { H } ) } { \mathbb { Q } ( \mathbf { Z } ^ { l } ) } \right) - D _ { K L } \left( \mathbb { P } ( \mathbf { Z } ^ { l } ) \| \mathbb { Q } ( \mathbf { Z } ^ { l } ) \right) } \\ & { \qquad + \mathbb { E } \left( \log \frac { \mathbb { P } ( \mathbf { X } _ { h } ^ { l } \mid \mathbf { Z } ^ { l } , \mathbf { H } ) } { \mathbb { Q } ( \mathbf { X } _ { h } ^ { l } ) } \right) - D _ { K L } \left( \mathbb { P } ( \mathbf { X } _ { h } ^ { l } ) \| \mathbb { Q } ( \mathbf { X } _ { h } ^ { l } ) \right) } \\ & { \qquad \leq \mathbb { E } \left( \log \frac { \mathbb { P } ( \mathbf { Z } ^ { l } \mid \mathbf { X } ^ { l - 1 } , \mathbf { H } ) } { \mathbb { Q } ( \mathbf { Z } ^ { l } ) } \right) + \mathbb { E } \left( \log \frac { \mathbb { P } ( \mathbf { X } _ { h } ^ { l } \mid \mathbf { Z } ^ { l } , \mathbf { H } ) } { \mathbb { Q } ( \mathbf { X } _ { h } ^ { l } ) } \right) } \\ & { \qquad = \mathbf { Z } \mathbf { I } \mathbb { B } ^ { l } + \mathbf { X } \mathbf { I } \mathbf { B } ^ { l } } \end{array}\tag{27}
$$

$$
\mathbf { Z } \mathbf { I B } ^ { l } = D _ { K L } \left( \mathbb { P } ( \mathbf { Z } ^ { l } \mid \mathbf { X } ^ { l - 1 } , \mathbf { H } ) \parallel \mathbb { Q } ( \mathbf { Z } ^ { l } ) \right) , \mathbf { X } \mathbf { I B } ^ { l } = D _ { K L } \left( \mathbb { P } ( \mathbf { X } _ { h } ^ { l } \mid \mathbf { Z } ^ { l } , \mathbf { H } ) \parallel \mathbb { Q } ( \mathbf { X } _ { h } ^ { l } ) \right)\tag{28}
$$

![](images/0a15bcf122d439e24283138e5c1a40ff7fbf7eac28555926da6b22c30e6dd6f6.jpg)  
(a) ASD vs. NC

![](images/168082de6d5662fec99e4f7602245a103457aad6fb71218edf8326a811d26627.jpg)  
(b) AD vs. NC

![](images/f46004ea660d049f32c2fd5b596608bf8a65cd1c654d95add6c0985bc774669f.jpg)  
(c) ASD vs. NC

![](images/f56f9b9fa87039c082fb9d7cb68ff9b8c5a5067d671cc3cbd73a4db39b326a6e.jpg)  
(d) AD vs. NC  
Figure 6: Visualization of attention maps and discriminative connections. connections within the same neural system (VN, AN, BLN, DMN, SMN, SN, MN, CCN) are colored accordingly, while connections across different systems are colored gray.

Proof for Proposition 3 According to the Markovian chain in Fig. 5(b), we restate Proposition 3: For any distribution Q(X<sup>l</sup>), we have

$$
\begin{array} { r l } & { I ( \mathbf { X } _ { h } ^ { l } , \mathbf { X } _ { t } ^ { l } ; \mathbf { X } ^ { l } ) = \mathbb { E } ( \log \frac { \mathbb { P } ( \mathbf { X } ^ { l } \mid \mathbf { X } _ { h } ^ { l } , \mathbf { X } _ { t } ^ { l } ) } { \mathbb { Q } ( \mathbf { X } ^ { l } ) } ) - D _ { K L } ( \mathbb { P } ( \mathbf { X } ^ { l } ) \| \mathbb { Q } ( \mathbf { X } ^ { l } ) ) } \\ & { \qquad \leq \mathbb { E } ( \log \frac { \mathbb { P } ( \mathbf { X } ^ { l } \mid \mathbf { X } _ { h } ^ { l } , \mathbf { X } _ { t } ^ { l } ) } { \mathbb { Q } ( \mathbf { X } ^ { l } ) } ) } \\ & { \qquad = D _ { K L } ( \mathbb { P } ( \mathbf { X } ^ { l } \mid \mathbf { X } _ { h } ^ { l } , \mathbf { X } _ { t } ^ { l } ) \parallel \mathbb { Q } ( \mathbf { X } ^ { l } ) . } \end{array}\tag{29}
$$

## C Additional Experiments.

## C.1 Analysis of Attention Maps and Discriminative Connections

To evaluate the contribution of the self-attention mechanism in the Transformer module, we visualize the average attention maps along the group level, as shown in Fig.6. Additionally, to clearly display the variations within and between sub-networks, we divide the entire brain network into eight subnetworks: visual network (VN), auditory network (AN), basal ganglia network (BLN), default mode network (DMN), sensorimotor network (SMN), salience network (SN), motor network (MN), and cognitive control network (CCN). In the ASD vs. NC classification task, the attention scores within the SMN are relatively high, indicating that the inter-ROI connectivity within the SMN plays a significant role in ASD prediction. This is consistent with previous studies that have shown reduced functional connectivity within the SMN [46], reflecting alterations in sensory and motor processing [47]. In the AD vs. NC classification task, the positions with higher attention scores are primarily located within the DMN. Previous studies have indicated that the substantial accumulation of β- amyoid protein in the DMN of AD patients leads to reduced functional connectivity in this region, which is associated with cognitive impairments in AD patients [48].

We perform a t-test on attention maps to identify discriminative connections. For clarity, only connections with significant differences (p-value < 0.01) are shown in Fig. 6. In the ASD vs. NC classification task, the identified discriminative connections include those between the left anterior cingulate and paracingulate gyri (ACG.L) and right Inferior parietal (IPL.R), as well as between the right superior frontal gyrus, dorsolateral (SFGdor.R). This validates previous research, which suggests that changes in these connections are often associated with the repeat stereotypical behaviors commonly observed in individuals with ASD [49]. In the AD vs. NC classification task, the discriminative connections include the connection between the left middle frontal gyrus (MFG.L) and the left superior frontal gyrus, dorsolateral (SFGdor.L). This is consistent with previous findings [50], which suggest that changes in functional connectivity between the right frontal lobe and the superior frontal gyrus in AD patients are associated with cognitive decline.

## C.2 Hyperparameter Analysis

To evaluate the influence of hyperparameters on classification performance, we conduct controlled variable experiments on two main hyperparameters: the value of k for KNN-based hypergraph construction and the number of layers L in the information bottleneck-guided hypergraph Transformer. Fig.7(a) and $\mathrm { F i g . 7 ( b ) }$ illustrate the impact of different values of k on two datasets. The accuracy is highest when $k = 3$ for both classification tasks. The accuracy is highest when $k = 3$ for both classification tasks. When $k = 2 ,$ , the hypergraph degenerates into a graph, causing a loss of highorder information in the brain network. When $k > 3 ,$ , the information aggregation range increases, and the representations of different nodes become more homogeneous, resulting in a drop in accuracy.

![](images/6201dc26b0b635164167619fdc364139ad5355fa1b9896c31a786a973f42a0d4.jpg)  
Figure 7: The effect of varying hyperparameters. (a) and (c) are experiments on ABIDE dataset. (b) and (d) are experiments on ADNI dataset.

Table 3: Running time with different methods.
<table><tr><td>Method</td><td>Running Time on ABIDE(s) Running Time on ADNI(s)</td></tr><tr><td>BrainNetCNN</td><td> $3 9 . 7 5 \pm 1 . 0 3$   $8 . 0 4 \pm 0 . 9 8$ </td></tr><tr><td>ContrastPool  $3 3 4 . 8 9 \pm 4 . 2 3$ </td><td> $8 3 . 6 2 \pm 2 . 5 0$ </td></tr><tr><td>BrainGNN</td><td> $2 9 4 . 0 1 \pm 1 . 1 1$ </td></tr><tr><td> $7 4 . 9 7 \pm 2 . 0 8$ </td><td> $2 1 2 . 4 3 \pm 1 . 3 2$   $5 1 . 6 6 \pm 1 . 5 8$ </td></tr><tr><td>BrainIB HGNN+</td><td> $3 6 . 5 5 \pm 1 . 2 2$   $7 . 6 3 \pm 3 . 2 2$ </td></tr><tr><td>FC-HAT</td><td> $8 2 . 5 6 \pm 0 . 6 9$   $2 2 . 7 6 \pm 1 . 0 8$ </td></tr><tr><td>BrainNetTF  $4 9 . 7 2 \pm 1 . 0 2$ </td><td> $1 0 . 8 2 \pm 0 . 3 4$ </td></tr><tr><td>Com-BrainTF</td><td> $6 1 . 2 6 \pm 2 . 3 1$   $1 4 . 3 2 \pm 0 . 7 4$ </td></tr><tr><td>ALTER</td><td> $5 4 . 8 7 \pm 0 . 9 8$   $1 1 . 3 5 \pm 2 . 6 2$ </td></tr><tr><td>IBAHGT  $7 2 . 4 3 \pm 0 . 8 8$ </td><td> $1 5 . 9 7 \pm 0 . 4 9$ </td></tr></table>

Fig.7(c) and Fig.7(d) show the variation in accuracy with respect to the number of layers L on two datasets. We find that when the number of layers $\dot { L } = 2$ , the model achieved the highest accuracy in both classification tasks. When $L < 2 ,$ , the model’s depth was insufficient to effectively capture the complex dependencies within the brain network. On the other hand, when $L > 2 .$ , an excessive number of layers led to overfitting, resulting in a decline in performance.

## C.3 Running Time

Table 3 shows the running time of IBAHGT compared to other methods on ABIDE and ADNI datasets. BrainGNN and ContrastPool are significantly slower than the other methods, primarily due to their pooling processes, which consume more time. BrainIB dynamically samples subgraphs during training and exhibits slower convergence, resulting in a longer training time. The hypergraph based method FC-HAT is slower than both our IBAHGT and HGNN<sup>+</sup>, mainly because FC-HAT dynamically updates the hypergraph structure at each layer using KNN and k-means methods, which adds additional time cost. The adaptive hypergraph convolution and node-level fusion in the proposed IBAHGT are formulated in matrix form during implementation to accelerate training on GPU. Although the training time is slightly longer compared to Transformer-based methods, this increase is acceptable given the significant performance improvement.

## C.4 Baselines and Additional Comparative Experiments on ADHD-200 Dataet

The selected baselines mainly include three categories. The first category consists of convolutional neural network (CNN)-based method, such as BrainNetCNN [36]. BrainNetCNN employs new feature extraction structures based on the topological locality characteristics of brain networks. The second category includes graph or hypergraph-based methods, such as ContrastPool [37], BrainGNN [22], BrainIB [28], $\mathrm { H G N N ^ { + } }$ [38], and FC-HAT [15]. ContrastPool generates contrast graph through a dual-attention module to guide differentiable graph pooling, which is used to produce effective brain network representations for disease classification. BrainIB introduces the IB principle into brain network analysis to effectively identify disease-specific prominent brain network connections. $\mathrm { H G N N ^ { + } }$ is a general hypergraph convolution method. FC-HAT combines KNN and k-means to dynamically construct hypergraphs and captures information within the hypergraph during the hypergraph aggregation attention phase. The third category consists of Transformer-based methods, including BrainNetTF [19], Com-BrainTF [39], and ALTER [40]. BrainNetTF employs self-attention to learn the long-range dependencies between ROIs and utilizes orthogonal clustering for graph-level representation. Com-BrainTF is a hierarchical local-global transformer architecture that learns intraand inter-community aware node embeddings. ALTER proposes a brain graph transformer to capture long-range dependencies between ROIs utilizing biased random walk. To ensure a fair comparison, we conduct experiments using the open-source code provided in the papers and the optimal parameters specified therein.

Table 4: Comparison experiments results on the ADHD-200 dataset.
<table><tr><td>Datasets (Tasks)</td><td colspan="2">ADHD-200 (ADHD vs. NC)</td></tr><tr><td>Method</td><td></td><td>ACC (%) SPE (%) SEN (%) AUC (%)</td></tr><tr><td>BrainNetCNN ContrastPool</td><td> $6 2 . 3 { \pm } 1 . 7 $   $6 7 . 6 { \pm } 2 . 7 $ </td><td> $5 3 . 5 { \pm } 1 . 5 $   $6 1 . 8 { \pm } 1 . 4 $ </td></tr><tr><td>BrainGNN</td><td> $6 1 . 9 { \pm } 1 . 6 $   $6 9 . 0 { \pm } 4 . 0 $   $6 2 . 5 { \pm } 1 . 9 $   $6 7 . 3 { \pm } 3 . 3 $ </td><td> $5 0 . 4 \pm 4 . 2$   $6 0 . 6 { \pm } 3 . 5 $   $5 4 . 7 { \pm } 4 . 5 $   $6 2 . 3 { \pm } 2 . 1 $ </td></tr><tr><td>BrainIB</td><td> $6 3 . 5 { \pm } 2 . 0 $   $6 8 . 7 { \pm } 2 . 8 $ </td><td> $5 5 . 2 { \pm } 5 . 0 $   $6 3 . 4 \pm 2 . 2$ </td></tr><tr><td>HGNN+</td><td> $6 4 . 4 \pm 1 . 0$   $6 8 . 9 { \pm } 2 . 5 $ </td><td> $5 7 . 2 { \pm } 5 . 8 $   $6 4 . 1 { \pm } 1 . 6 $ </td></tr><tr><td>FC-HAT</td><td> $6 6 . 5 { \pm } 1 . 1$   $7 0 . 1 \pm 2 . 0$ </td><td> $6 0 . 6 { \pm } 4 . 5 $   $6 6 . 3 { \pm } 1 . 2 $ </td></tr><tr><td>BrainNetTF</td><td> $6 7 . 2 \pm 1 . 7$   $7 0 . 4 \pm 5 . 7$ </td><td> $6 2 . 0 { \pm } 5 . 0 \ $   $6 6 . 6 { \pm } 1 . 1$ </td></tr><tr><td>Com-BrainTF</td><td> $6 7 . 7 { \pm } 1 . 7 $   $6 9 . 7 { \pm } 1 . 7 $ </td><td></td></tr><tr><td>ALTER</td><td></td><td> $6 4 . 5 { \pm } 3 . 9 $   $6 7 . 6 { \pm } 2 . 5 $ </td></tr><tr><td>IBAHGT</td><td> $6 8 . 8 { \pm } 2 . 1 $   ${ \bf 7 4 . 1 \pm 4 . 6 }$   ${ \bf 7 2 . 3 \pm 1 . 3 }$   $7 3 . 7 { \pm } 1 . 9 $ </td><td> $6 0 . 3 { \pm } 7 . 6 $   $6 8 . 0 { \pm } 2 . 4 $   ${ \bf 6 9 . 9 { \pm 4 . 0 } }$   $7 2 . 2 { \pm } 1 . 2 $ </td></tr></table>

To further validate the performance of IBAHGT, the ADHD-200 dataset [51] is also used for comparative experiments. The ADHD-200 dataset is a multi-site neuroimaging resource for attention deficit hyperactivity disorder (ADHD) research, consisting of 357 ADHD patients and 573 normal controls (NCs) from eight international imaging centers. As shown in Table 4, the experimental results demonstrate that IBAHGT outperforms the baselines on the majority of metrics, further validating the superiority of the proposed method in brain disease diagnosis.

## D Further Discussion.

## D.1 Limitations.

This study primarily focuses on functional brain networks constructed from fMRI data, providing a unified analysis of high-order relationships and both short- and long-range dependencies in brain network. However, the influence of structural brain networks constructed from DTI data on the diverse dependencies among ROIs in functional brain networks are also worthy of exploration. In future work, we will delve into the effects of structural connectivity on multiple types of dependencies in functional brain network and propose methods to integrate functional and structural brain networks for more comprehensive analyses of high-order correlations and both short- and long-range dependencies among ROIs.

## D.2 Possible Societal Impacts.

This research involves brain disease diagnosis, and it is therefore necessary to acknowledge its potential societal impact. The proposed method contributes to the discovery of potential biomarkers, offering positive impacts for neuroscience research and computer-aided diagnosis. However, the use of artificial intelligence in diagnostic assistance may lead to misdiagnoses, which are difficult to completely avoid and could have negative impacts for both patients and society. In real-world clinical settings, AI-based methods should serve solely as decision-support tools, with final diagnostic decisions remaining the responsibility of medical professionals.