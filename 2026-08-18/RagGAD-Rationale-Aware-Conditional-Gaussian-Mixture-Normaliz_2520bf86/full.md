# RagGAD: Rationale-Aware Conditional Gaussian Mixture Normalizing Flow for Unsupervised Graph Anomaly Detection

Junxin Lu, Jing Zhao, Shiliang Sun, Senior Member, IEEE

Abstract—Graph anomaly detection aims to identify nodes that deviate from normal behavioral patterns within graphs. However, existing methods largely rely on the homophily assumption, which makes it difficult to distinguish spurious affinities and to capture the diverse behaviors of normal nodes, limiting their robustness in complex real-world scenarios. To address this problem, we propose RagGAD, an unsupervised graph anomaly detection framework based on rationale-aware conditional Gaussian mixture normalizing flow. RagGAD introduces an adaptive rationale disentangler to disentangle stable rationales from spurious correlations within node interrelationships, and further decomposes stable rationales into robust and fragile components. The learned rationales capture underlying interaction patterns that characterize normal behaviors under varying conditions, while anomalies emerge as deviations associated with unstable or spurious correlations. To model the intricate distributions of normal and abnormal nodes, RagGAD integrates rationalenon-rationale Gaussian mixture modeling with a robust-fragile rationale mixture learning strategy. By mitigating spurious homophilic correlations and embracing the heterogeneity of normal patterns, RagGAD identifies anomalies as low-density regions within a structure-aware distribution space. Extensive experiments on multiple benchmark datasets demonstrate that RagGAD outperforms state-of-the-art methods.

Index Terms—Graph anomaly detection, node homophily, stable rationales, rationale-aware conditional Gaussian mixture normalizing flow

## I. INTRODUCTION

Graph anomaly detection (GAD) aims to identify nodes in a graph that exhibit behaviors or characteristics significantly deviating from the majority [1]–[3]. GAD is crucial in various real-world applications, such as detecting irregularities in financial [4], [5] and social [6] networks. Existing GAD methods are predominantly based on supervised or semi-supervised learning paradigms to model node-level abnormalities, relying on labeled data to guide model training [7], [8]. However, in real-world scenarios, data is often vast and complex, with anomalies that cannot be pre-identified or labeled.

In this context, unsupervised graph anomaly detection (UGAD) [9]–[11] emerges as a more practical yet challenging paradigm, as it eliminates the need for labeled supervision, thereby expanding the practical applicability of GAD. Existing UGAD works are comparatively few, which can be roughly categorized into two groups: (i) data reconstruction-based methods, which identify anomalies by quantifying discrepancies between reconstructed node attributes/structures and the original inputs [12]–[15]; (ii) self-supervised learningbased methods, which utilize pre-text tasks such as proxy classification, surrogate contrastive learning, or pre-trained models to provide additional supervisions [16]–[18].

![](images/ccab5064b1ab6e43f407b582ae23a88d2de81443ba367a967bcbb7dd820f1d35.jpg)  
Fig. 1. In UGAD, (a) existing methods fall into the homophily trap when abnormal nodes mimic normal connection patterns (e.g., abnormal node v<sub>2</sub> imitating the affinity of normal node $v _ { 1 }$ with its neighbors). Additionally, they incorrectly identify $v _ { 4 }$ as abnormal due to overlooking the diversity of normal patterns. (b) RagGAD disentangles stable rationales, enabling the model to capture true mutual influence while eliminate spurious connections/affinities. Furthermore, it decomposes rationales into robust and fragile components, allowing for fine-grained modeling of both stable and context-sensitive normal patterns.

Despite achieving promising detection performance, when dealing with complex and diverse graphs, above-mentioned works inherently face the following limitations: (i) Overreliance on the homophily assumption, which posits that normal nodes have stronger affinity with each other than anomalies, may lead to the homophily trap [11], [12], [19], [20]. When normal nodes exhibit weak affinity or abnormal nodes deliberately conceal their suspicious activity patterns, the homophily assumption is violated, resulting in misidentification. As illustrated in Figure 1(a), the abnormal node $v _ { 2 }$ disguises its abnormal behavior by mimicking the normal node $v _ { 1 }$ , sharing same neighbors and inter-nodes affinities with $v _ { 1 }$ . Consequently, $v _ { 2 }$ is incorrectly identified as a normal node. This is attributed to their neglect of deeper interaction patterns among nodes, focusing solely on statistical co-occurrence or superficial similarity. However, there exist intricate yet stable interdependencies among nodes, referred to as robust rationales. These rationales capture the intricate yet stable interdependencies among nodes, explaining the robust interaction patterns underlying specific nodes. By modeling stable rationales, the model can uncover the genuine mutual influences between nodes, eliminating spurious connections or affinities even when abnormal nodes obscure their behavior, as shown in Figure 1(b). (ii) In real-world graphs, nodes often exhibit diverse normal patterns, such as varying preferences and personalities within a community. As shown in Figure 1(a), nodes $v _ { 3 }$ and $v _ { 4 }$ are normal and connected to abnormal node $v _ { 2 }$ . If the model fails to distinguish the varying directions and strengths of interactions with $v _ { 2 } .$ , it may misclassify $v _ { 4 }$ abnormal. Therefore, the model must be sufficiently sensitive to these variations, ensuring that diverse and fine-grained normal behaviors are not misidentified as anomalies.

To address these challenges, we propose RagGAD, a novel framework based on Rationale-aware conditional gaussian mixture normalizing flow for UGAD, as shown in Figure 2. RagGAD employs an adaptive rationale disentangler (ARD) to disentangles stable rationale from spurious correlations in node interrelationships. The rationales encapsulate the underlying influence mechanisms governing normal node behaviors, while anomalies are revealed via unstable spurious correlations. To address the diversity of normal patterns, RagGAD employs a learnable rationale soft mask to decompose stable rationales into robust and fragile components. This enables fine-grained modeling of both stable and context-sensitive normal patterns. Based on the disentangled stable rationales and unstable correlations, we propose a node-level rationale-aware conditional Gaussian mixture normalizing flow (RGMN) model to capture complex normal-abnormal distributions, with anomalies identified as low-density regions within the mixture distribution. A key component of RGMN is rationale-non-rationale Gaussian mixture modeling (RRGM), which leverages conditional Gaussian mixture modeling to effectively separate and model the distributions of rationale and non-rationale correlations. Additionally, to capture nuanced variations within rationale representations, we introduce a robust-fragile rationale mixture learning (RFRM) strategy. This strategy models diverse combinations of robust and fragile components, enabling RagGAD to learn fine-grained divergent traits across normal patterns and provide a comprehensive understanding of the underlying mixture distribution. The main contributions of this paper are summarized as follows:

• We disentangle stable rationales from spurious correlations within complex node interrelationships, further refining rationales into robust and fragile components using an adaptive rationale disentangler. This enables RagGAD to capture the underlying influence mechanisms governing normal node behaviors, while anomalies are revealed through unstable non-rationale correlations.

• We propose a node-level rationale-aware conditional Gaussian mixture normalizing flow model, which learns the diversity of node normal behaviors and models the intricate distributions of both normal and abnormal patterns, enabling efficient anomaly detection in low-density regions.

• Experimental results on multiple datasets demonstrate that RagGAD significantly outperforms state-of-the-art baselines for unsupervised graph anomaly detection.

## II. RELATIVE WORKS

## A. Unsupervised Graph Anomaly Detection

Compared to supervised [8], [21] and semi-supervised [22]–[25] paradigms, unsupervised graph anomaly detection (UGAD) aims to identify abnormal nodes among predominantly normal nodes without relying on labeled data [7], [14].

A practical and intuitive strategy is data reconstructionbased, where nodes are reconstructed based on their similarity or affinity with neighbors, and those with high reconstruction errors are identified as anomalies [7], [12]–[14]. Another line of work leverages self-supervised learning, incorporating contrastive learning [17], [18], [26], [27], proxy classification [28], and auxiliary objectives [15], [29]–[32]. Additionally, general anomaly detection methods, such as ARC [33], UNPrompt [9] and FreeGAD [34], aim to detect anomalies across domains without requiring fine-tuning and retraining, but suffer from low accuracy [9], [33], [35]. FreeGAD [34] leverages an affinity-gated encoder and anchor-guided statistical deviations to compute anomaly scores, however, its reliance on heuristic anchors and shallow statistical measures limits its ability to capture complex dependencies. ADA-GAD [12] and HUGE [30] alleviate the homophily trap through learning-free augmentation strategy and label-free heterogeneity measurement to reduce false detection rates. CoCo [10] leverages Transformers to jointly model multi-hop local and global contextual features, and detects anomalies by measuring discrepancies in their correlations. GCTAM [36] extends TAM [11], which identifies anomalies by maximizing normal-node similarity while truncating anomalous ones, by incorporating contextual information and global similarities to avoid rigid thresholds thereby improving the modeling of node-specific characteristics and high-order affinities. However, they are still limited to the statistical node associations provided by the dataset, without further exploring more robust underlying rationale relationships among variables.

The aforementioned works typically rely on statistical cooccurrence/similarity or homophily-aware anomaly metrics to model intrinsic normal patterns for UGAD. However, they struggle to eliminate spurious connections and affinities, making them susceptible to the homophily trap when anomalies conceal their suspicious behavior. In contrast, RagGAD disentangles more robust dependencies, i.e., rationales, from unstable non-rationale correlations, thereby revealing the underlying mechanisms of interactions and generative patterns among normal nodes, while exposing anomalies through non rationale correlations.

## B. Normalizing Flows in Anomaly Detection

Normalizing flows (NFs) leverage invertible transformations to map complex data distributions to simpler ones (e.g., standard normal distribution), enabling precise probability density estimation. NFs maximize the log-likelihood of normal samples, assuming that normal samples map to high-density regions, while anomalies fall into low-density regions [37], [38]. NFs have been widely explored for anomaly detection in non-graph data [39]–[43]. DifferNet [39] uses NFs to assign meaningful likelihoods to images and develops a scoring function to detect defects. BGAD [44] enhances model distinguishability through a boundary-guided semi-pull contrastive learning mechanism based on NFs, while HGAD [43] introduces a hierarchical Gaussian mixture normalizing flow for unified anomaly detection. GANF [45] combine normalizing flow with a graph auto-encoder to create a generative model of graph structures. For graph data, FANFOLD [38] proposes a graph normalizing flows-driven asymmetric network for unsupervised graph-level anomaly detection. However, how to utilize normalizing flows for the more challenging task of node-level UGAD, while disentangling inter-node robust rationale relationships, remains unexplored in previous works.

![](images/2edb404344e663d2de408ba6018072c22dcdb307bae6db896582214d019c1acb.jpg)  
Fig. 2. Overview of RagGAD. RagGAD introduces an adaptive rationale disentangler (ARD) to disentangle rationales $\mathcal { G } _ { c }$ and non-rationale correlations $\mathcal { G } _ { o }$ from complex node interrelationships I. The robust rationales $\mathcal { G } _ { r \mid c }$ and fragile rationales $\mathcal { G } _ { f | c }$ are further decomposed from $\mathcal { G } _ { c }$ using a learnable soft mask $\hat { \mathcal { M } } .$ Based on $\mathcal { G } _ { r \mid c } , \mathcal { G } _ { f \mid c } ,$ and $\mathcal { G } _ { o } , \mathrm { R a g G A D }$ extracts the robust rationale representation $H _ { r | c } ,$ fragile rationale representation $H _ { f | c } ,$ , and non-rationale representation $H _ { o } ,$ respectively, which are then transformed into latent conditional embeddings $z _ { r | c } , z _ { f | c } ,$ and $z _ { o }$ through NCNF. Subsequently, RRGM instantiates rationales and non-rationale correlations as distinct classes for Gaussian mixture modeling to fit the distribution of node attribute x˜ conditioned on $z _ { r | c } , z _ { f | c }$ and $z _ { o }$ . Meanwhile, RFRM captures the latent diversification of normal distributions among nodes through fine-grained robust-fragile rational Gaussian mixture learning.

These works inspire us to leverage NFs for learning data distributions and identifying anomalies in low-density regions. However, existing methods based on normalizing flows rely solely on normal data during training, which is impractical and leads to ambiguous decision boundaries between normal and abnormal data, thereby reducing distinguishability. In contrast, RagGAD facilitates coexistence of normal and abnormal data during training, leveraging disentangled robust rationales and non-rationale correlations for discriminative rationale Gaussian mixture modeling in UGAD. Furthermore, existing methods overlook the diversity of normal patterns, misclassifying diverse fine-grained normal behaviors as anomalies.

## III. PRELIMINARIES

## A. Problem Statement

An attributed graph is represented as $G = \{ \nu , \mathcal { E } , \mathcal { X } \}$ , where $\mathcal { V } = \{ v _ { 1 } , v _ { 2 } , \cdot \cdot \cdot , v _ { N } \}$ and $\mathcal { E } = \{ \dots , e _ { i j } , \dots \} _ { i , j = 1 } ^ { N }$ are the sets of nodes and edges, respectively. $\mathcal { X } \overset { \cdot } { = } \{ x _ { i } \} _ { i = 1 } ^ { N } \ \in \ \mathbb { R } ^ { N \times d _ { x } }$ is the node-level attribute matrix, where $x _ { i }$ indicates the attribute vector of node $v _ { i }$ with $d _ { x }$ dimensions. The internode connectivity of $G$ is represented by an adjacent matrix $A = \{ a _ { i j } \} _ { i . j = 1 } ^ { N } \stackrel { \cdot } { \in } \mathbb { R } ^ { N \times N }$ , where $a _ { i j } = 1$ indicates an edge between nodes $v _ { i }$ and $v _ { j } .$ , and $a _ { i j } = 0$ indicates no edge.

Given an attributed graph $G \ : = \ : \{ \nu , \mathcal { E } , \mathcal { X } \}$ , UGAD aims to identify abnormal nodes $\gamma _ { a }$ (the minority) from normal nodes $\nu _ { n }$ (the majority), where $\mathcal { V } _ { a } \cup \mathcal { V } _ { n } = \mathcal { V } , \mathcal { V } _ { a } \cap \mathcal { V } _ { n } = \emptyset$ and $| \nu _ { a } | \ll | \nu _ { n } |$ , without access to any class labels during training. The objective of UGAD model is to learn an anomaly scoring function $\mathcal { S } : \mathcal { V }  \mathbb { R }$ , such that ${ \cal S } ( v _ { n } ) < { \cal S } ( v _ { a } )$ for all $\forall v _ { n } \in \mathcal { V } _ { n }$ and $\forall v _ { a } \in \mathcal { V } _ { a }$

## B. Normalizing Flow

The normalizing flow [46]–[48] is a probabilistic density estimation model that maps an unknown data distribution $p ( \mathcal { X } )$ to a tractable latent distribution $p ( Z )$ by a sequence of invertible transformations. Specifically, a normalizing flow model is defined as a bijective transformation $\mathcal { F } : x \in \mathbb { R } ^ { d _ { x } }  z \in \mathbb { R } ^ { d _ { z } }$ establishing a mapping from the data space $x \in \mathcal { X }$ to the latent space $z \in Z$ . According to the change of variables formula in calculus [49], the log-likelihood of any $x \in \mathcal { X }$ can be computed as:

$$
\log p _ { \mathcal { X } } ( x ) = \log p _ { Z } \left( \mathcal { F } _ { \theta } ( x ) \right) + \log \lvert \mathrm { d e t } J \rvert ,\tag{1}
$$

where $\begin{array} { r } { J = \nabla _ { x } \mathcal { F } _ { \theta } ( x ) = \frac { \delta \mathcal { F } _ { \theta } ( x ) } { \delta x } } \end{array}$ is the Jacobian matrix of the transformation. θ represents the learnable model parameters of ${ \mathcal F } .$ In most flow-based models, $p _ { Z } ( z )$ is typically assumed to follow a standard multivariate normal distribution $\mathcal { N } ( 0 , \mathbb { I } )$ for simplicity. The normalizing flow $\mathcal { F } _ { \theta }$ can be optimized by maximizing the log-likelihood over the training distribution $p ( \mathcal { X } )$ . Consequently, the loss function is defined as follows:

$$
\mathcal { L } _ { n f } = \mathbb { E } _ { x \sim p ( x ) } [ - \mathrm { l o g } p _ { \mathcal { X } } ( x ) ] .\tag{2}
$$

Conditional normalizing flow [50]–[52] extends normalizing flow by introducing an external condition C, allowing the model to estimate the conditional distribution $p _ { \mathcal { X } } ( x | \mathcal { C } ) ~ =$ $p _ { Z } \left( \mathcal { F } _ { \theta } ( x ; \mathcal { C } ) \right) \lvert \mathrm { d e t } \nabla _ { x } \mathcal { F } _ { \theta } ( x ; \mathcal { C } ) \rvert$ , thereby enabling conditional density estimation. The log-density of $x$ conditioned on C is defined as follows:

$$
\begin{array} { r } { \log p _ { \mathcal { X } } ( x | \mathcal { C } ) = \log p _ { Z } \left( \mathcal { F } _ { \theta } ( x ; \mathcal { C } ) \right) + \log \left| \mathsf { d e t } \nabla _ { x } \mathcal { F } _ { \theta } ( x ; \mathcal { C } ) \right| . } \end{array}\tag{3}
$$

## IV. METHODOLOGY

We propose RagGAD, a node-level unsupervised graph anomaly detection framework based on rationale-aware conditional Gaussian mixture normalizing flow.

## A. Adaptive Rationale Disentanglement

In UGAD, only the node attribute matrix X and the adjacent matrix $A$ are accessible during training, while the stable rationales $\mathcal { G } _ { c }$ and unstable non-rationale correlations $\mathcal { G } _ { o }$ are unknown. To tackle this, we propose an adaptive rationale disentanglement module, which effectively disentangles the $\mathcal { G } _ { c }$ and $\mathcal { G } _ { o }$ from node interdependencies I.

RagGAD first employs an attribute projector $\mathcal { P } ( \cdot )$ to reduce the attribute dimension of nodes [33], [53]. Specifically, given the attribute matrix $\mathcal { X } \in \mathbb { R } ^ { N \times d _ { x } }$ , the attribute projection is defined as follows:

$$
\tilde { \mathcal { X } } \in \mathbb { R } ^ { N \times d _ { \tilde { \mathcal { X } } } } = \mathcal { P } ( \mathcal { X } ) = \mathcal { X } \boldsymbol { W } _ { \mathcal { P } } ,\tag{4}
$$

where $\tilde { \mathcal X }$ is the projected attribute matrix. $W _ { \mathcal { P } } ~ \in ~ \mathbb { R } ^ { N \times d _ { \tilde { x } } }$ is a dataset-specify linear projection weight matrix, and $d _ { \tilde { x } }$ is a pre-defined dimension of the projected attribute for nodes. The attribute projector $\mathcal { P } ( \cdot )$ can employ commonly used dimensionality reduction methods, such as singular value decomposition [54] or principal component analysis [55].

a) Adaptive Rationale Disentangler. We propose an adaptive rationale disentangler (ARD) to capture the stable rationales $\mathcal { G } _ { c }$ among nodes, while simultaneously disentangling robust rationales $\mathcal { G } _ { r | c }$ and fragile rationale relationships $\mathcal { G } _ { f | c } .$ Specifically, ARD integrates a node interrelation attention module to learn the structure interrelationships between nodes, forming the node interrelationships I. This involves calculating the cross attention coefficient $\varepsilon _ { i j } .$ , which reflects the interrelationship of node $v _ { j }$ to $v _ { i } ,$ defined as:

$$
\varepsilon _ { i j } = \frac { \mathbb { I } \left[ a _ { i j } = 1 \right] \cdot \exp \left( Q _ { i } K _ { j } ^ { T } / \sqrt { d _ { \tilde { x } } } \right) } { \sum _ { k } \mathbb { I } \left[ a _ { i k } = 1 \right] \cdot \exp \left( Q _ { i } K _ { k } ^ { T } / \sqrt { d _ { \tilde { x } } } \right) } ,\tag{5}
$$

where $Q ~ = ~ \tilde { x } _ { i } W _ { Q } ~ \in ~ \mathbb { R } ^ { d _ { \tilde { x } } }$ is the query embedding of $v _ { i } ,$ and $K ~ = ~ \tilde { x } _ { j } W _ { K } ~ \in ~ \mathbb { R } ^ { d _ { \tilde { x } } }$ is the key embedding of $v _ { j }$ $W _ { Q } \ \in \ \mathbb { R } ^ { d _ { \tilde { x } } \times \check { d _ { \tilde { x } } } }$ and $W _ { K } \in \mathbb { R } ^ { d _ { \widetilde { x } } \times d _ { \widetilde { x } } }$ are the query and key weights, respectively. According to the adjacent matrix A, node $v _ { j }$ is constrained to be a neighbor of node $v _ { i } .$ , i.e., $v _ { j } \in \tilde { \mathcal { N } } ( v _ { i } ) = \{ v _ { k } | a _ { i k } = 1 \}$

RagGAD treats the disentanglement of $\mathcal { G } _ { c }$ and $\mathcal { G } _ { o }$ as an edge selection problem. Some edges reflect stable underlying dependencies between nodes, i.e., rationales, while others represent unstable relationships determined by proximity or statistical correlations. Therefore, we use a binary mask as disentangler, denoted as $\mathcal { M } = \{ m _ { i j } \} _ { i , j = 1 } ^ { N } \in \{ 0 , \overset { . } { 1 } \} ^ { N \times N }$ , as shown in Figure 2. This allows us to disentangle $\mathcal { G } _ { c } \in \mathbb { R } ^ { N \times N }$ and $\mathcal { G } _ { o } \in \mathbb { R } ^ { \bar { N } \times N }$ from $\mathcal { T } = \{ \varepsilon _ { i j } \} _ { i , j = 1 } ^ { N }$ as follows:

$$
\left\{ \begin{array} { l l } { \mathcal { G } _ { c } = \mathcal { T } \odot \mathcal { M } , } \\ { \mathcal { G } _ { o } = \mathcal { T } \odot ( 1 - \mathcal { M } ) , } \end{array} \right. \quad \mathrm { s . t . } \ \mathcal { M } = \mathbb { I } _ { \tilde { m } _ { i j } > \tau } \sigma ( \tilde { \mathcal { M } } ) ,\tag{6}
$$

where $\tilde { \mathcal { M } } = \{ \tilde { m } _ { i j } \} _ { i , j = 1 } ^ { N } \in \mathbb { R } ^ { N \times N }$ is the trainable mask of M. $\sigma ( \cdot )$ is the sigmoid function, τ is the hyperparameter for selecting rationale edges, and ⊙ represents element-wise multiplication.

In UGAD, relying solely on the disentanglement of rationales and non-rationale correlations remains insufficient for fully capturing the complex and diverse patterns underlying normal nodes. This limitation may cause certain nodes, which exhibit diverse behaviors but are inherently normal, to be mistakenly identified as anomalies. To address this, we further decompose $\mathcal { G } _ { c }$ into stable rationales $\mathcal { G } _ { r \mid c }$ and fragile rationales $\mathcal { G } _ { f | c } . ~ \mathcal { G } _ { r | c }$ captures persistent, robust relationships, while $\mathcal { G } _ { f | c }$ reflects context-sensitive relationships that are more susceptible to data perturbations or neighborhood fluctuations. To achieve this decomposition, RagGAD introduce a learnable soft mask $\hat { \mathcal { M } } \in \mathbb { R } ^ { \hat { N } \times N }$ , which partitions $\mathcal { G } _ { c }$ into $\mathcal { G } _ { r \mid c }$ and $\mathcal { G } _ { f | c }$ as follows:

$$
\mathcal { G } _ { r | c } = \mathcal { G } _ { c } \odot \sigma ( \hat { \mathcal { M } } ) , \quad \mathcal { G } _ { f | c } = \mathcal { G } _ { c } \odot ( 1 - \sigma ( \hat { \mathcal { M } } ) ) ,\tag{7}
$$

where $\hat { \mathcal { M } } = \{ \hat { m } _ { i j } \} _ { i , j = 1 } ^ { N } ;$ , with $\hat { m } _ { i j } \in [ 0 , 1 ]$ representing the learnable weight for the stability of the rationales from node $v _ { j }$ to node $v _ { i }$ . This decomposition enables a fine-grained reweighting partitioning of $\mathcal { G } _ { c } ,$ , where each rationale edge is assigned a stability score via $\sigma ( { \hat { \mathcal { M } } } )$ . By combining diverse robust and fragile rationales, RagGAD effectively delineates the boundaries between stable and fragile rationale relationships, offering a nuanced understanding of the underlying anomalies within the graph while comprehensively capturing the behavioral diversity of normal nodes.

Upon extracted $\mathcal { G } _ { r \mid c }$ and $\mathcal { G } _ { f | c }$ , we introduce two rationale augmented graph convolution layers, $\mathrm { G C N } _ { r | c } ( \cdot )$ and $\mathrm { G C N } _ { f | c } ( \cdot )$ , followed by a reconstructor $\mathcal { R } ( \cdot )$ , to generate the node attribute reconstruction. The reconstruction process is defined as follows:

$$
\begin{array} { r l } & { \qquad \hat { \mathcal { X } } = \mathcal { R } \left( \left[ H _ { r | c } \| H _ { f | c } \right] ; \theta _ { \mathcal { R } } \right) , } \\ & { H _ { r | c } { = } \mathrm { G C N } _ { r | c } ( \mathcal { G } _ { r | c } , \tilde { \mathcal { X } } ) = W _ { r | c } ^ { 2 } \Big ( \mathrm { R e L U } \Big ( \mathcal { G } _ { r | c } \tilde { \mathcal { X } } W _ { r | c } ^ { 1 } \Big ) \Big ) { + } b _ { r | c } ^ { 2 } , } \\ & { H _ { f | c } { = } \mathrm { G C N } _ { f | c } ( \mathcal { G } _ { f | c } , \tilde { \mathcal { X } } ) = W _ { f | c } ^ { 2 } \Big ( \mathrm { R e L U } \Big ( \mathcal { G } _ { f | c } \tilde { \mathcal { X } } W _ { f | c } ^ { 1 } \Big ) \Big ) { + } b _ { f | c } ^ { 2 } , } \end{array}\tag{8}
$$

where $W _ { \ast | c } ^ { 1 } \in \mathbb R ^ { d _ { \tilde { x } } \times d _ { \tilde { x } } } , \ W _ { \ast | c } ^ { 2 } \in \mathbb R ^ { d _ { \tilde { x } } \times d _ { h } }$ , and $b _ { * | c } ^ { 2 } \in \mathbb { R } ^ { d _ { h } }$ are the learnable parameters of $\mathrm { G C N } _ { r | c } ( \cdot )$ and $\mathrm { G C N } _ { f | c } ( \cdot )$ respectively. Inspired by the success of GCNs in node representation learning through the aggregation of neighborhood effects [56], [57], $\mathrm { G C N } _ { r | c } ( \cdot )$ and $\mathrm { G C N } _ { f | c } ( \cdot )$ aggregate the attributes of parent nodes to encode dependency relationships, yielding node-level rationale representations $\dot { H } _ { r | c } \in \mathbb { R } ^ { N \times \bar { d } _ { h } }$ and $H _ { f | c } \in \mathbb { R } ^ { N \times d _ { h } }$ . The [·∥·] denotes the concatenation of representations. $\mathcal { R } ( \cdot )$ , parameterized by $\theta _ { \mathcal { R } }$ , is a MLP that transforms the fused rationale representation $H \ = \ \left[ H _ { r | c } | | H _ { f | c } \right]$ into the reconstructed node attributes $\hat { \mathcal X }$ . Similarity, another graph convolution layer, $\mathrm { G C N } _ { o } ( \cdot )$ , is employed to extract the non-rationale representation $\dot { H } _ { o } ~ \in ~ \mathbb { R } ^ { \bar { N } \times \bar { d } _ { h } }$ , i.e., $H _ { o } \ =$ $\mathrm { G C N } _ { o } ( \mathcal { G } _ { o } , \tilde { \mathcal { X } } )$ .

B. Rationale-Aware Node-Level Gaussian Mixture Normalizing Flow Model

Inspired by the success of anomaly detection strategies based on normalizing flow, we leverage normalizing flow to learn a robust anomaly detection criterion for UGAD. Normalizing flow maps data into learned distributions, with the assumption that abnormal instances tend to reside in lowdensity regions, while normal instances are concentrated in high-density regions [38], [58], [59]. However, applying normalizing flow to UGAD presents several challenges: $( i )$ how to accurately estimating the density of a mixture distribution containing both normal and abnormal data, thereby distinguishing high-density (normal) and low-density (abnormal) regions; (ii) how to capture subtle variations in rationale representations to enable the model to learn the diverse patterns inherent in normal nodes, providing a more nuanced understanding of the underlying mixture distribution.

To address these challenges, we propose a node-level rationale-aware conditional Gaussian mixture normalizing flow model (RGMN), as illustrated in Figure 2. RGMN comprises three core modules: (i) a Node-Level Conditional Normalizing Flow (NCNF) block, which conditionally transforms both rationale and non-rationale node representations into latent conditional embeddings, thereby facilitating conditional density estimation; (ii) a Rationale-Non-Rationale Gaussian Mixture Modeling (RRGM) module, which separates and models rationale and non-rationale distributions using conditional Gaussian mixture modeling, enabling a more precise distinction between normal and abnormal nodes; and (iii) a Robust-Fragile Rationale Mixture Learning (RFRM) strategy, which learns the Gaussian mixture distribution of rationale representations, capturing fine-grained divergent patterns among normal nodes and characterizing their diverse traits.

a) Node-Level Conditional Normalizing Flows. The normalizing flow $\mathcal { F }$ in RagGAD is built upon the Real-NVP [48], comprising $\eta$ affine coupling layers with an identical structure, as shown in Figure 2. Specifically, given the robust rationale representation $\bar { h } _ { r | c } ^ { i } \in \mathbb { R } ^ { d _ { h } }$ , the affine transformation that maps the projected attribute $\tilde { x } _ { i } ~ \in ~ \mathbb { R } ^ { d _ { \tilde { x } } }$ of node $v _ { i }$ , conditioned on $h _ { r | c } ^ { i } \ ( \mathrm { i } . \mathrm { e } .$ , condition $\begin{array} { r } { \mathcal { C } = h _ { r | c } ^ { i } ) } \end{array}$ , to the latent conditional embedding $\boldsymbol { z } _ { r | c } ^ { i } \in \mathbb { R } ^ { d _ { z } }$ is defined as:

$$
z _ { r | c } ^ { i } = \mathcal { F } _ { \boldsymbol { \theta } } ( \tilde { x } _ { i } ; h _ { r | c } ^ { i } ) \Rightarrow z _ { r | c } ^ { i } = \left[ z _ { r | c } ^ { i , 1 } \lVert z _ { r | c } ^ { i , 2 } \right] ,\tag{9}
$$

$$
\tilde { x } _ { i } ^ { 1 } , \tilde { x } _ { i } ^ { 2 } = \Theta ( \tilde { x } _ { i } ) , z _ { r | c } ^ { i , 1 } = \tilde { x } _ { i } ^ { 1 } ,\tag{10}
$$

$$
\begin{array} { r } { z _ { r | c } ^ { i , 2 } = \tilde { x } _ { i } ^ { 2 } \odot \exp \Bigl ( s \Bigl ( \Bigl [ \tilde { x } _ { i } ^ { 1 } \| h _ { r | c } ^ { i } W _ { h } \Bigr ] \Bigr ) \Bigr ) + t \left( \Bigl [ \tilde { x } _ { i } ^ { 1 } \| h _ { r | c } ^ { i } W _ { h } \Bigr ] \right) , } \end{array}\tag{11}
$$

where Θ is a attribute partitioning function [50] that splits ${ \tilde { x } } _ { i }$ into $\tilde { x } _ { i } ^ { 1 } \in \mathbb { R } ^ { d _ { \tilde { x } / 2 } }$ and $\tilde { x } _ { i } ^ { 2 } \in \mathbb { R } ^ { d _ { \tilde { x } / 2 } }$ , respectively. $s ( \cdot )$ and $t ( \cdot )$ are transformation coefficients predicted by a learnable neural network, and is implemented using a MLPs [43], [48]. $W _ { h } \in \mathbb { R } ^ { d _ { h } \times d _ { \tilde { x } / 2 } }$ is a learnable parameters, and ⊙ represents the element-wise product. Based on $\operatorname { E q . } ( 3 )$ , the robust rationale conditional density of $\tilde { x } _ { i }$ can be defined as:

$$
\begin{array} { r } { \log p _ { \mathcal { X } } ( \tilde { x } _ { i } | h _ { r | c } ^ { i } ) { = } { \log p _ { Z } } \Big ( \mathcal { F } _ { \theta } ( \tilde { x } _ { i } ; h _ { r | c } ^ { i } ) \Big ) { + } { \log } \Big | \mathrm { d e t } \nabla _ { x } \mathcal { F } _ { \theta } ( \tilde { x } _ { i } ; h _ { r | c } ^ { i } ) \Big | . } \end{array}\tag{12}
$$

Similarly, conditioned on the fragile rationale representation $h _ { f | c } ^ { i }$ and the non-rationale representation $h _ { o } ^ { i } .$ , we can obtain the latent conditional embeddings $z _ { f \mid c } ^ { i }$ and $z _ { o } ^ { i }$ for node $v _ { i } ,$ along with the corresponding fragile rationale conditional density logp $\nu _ { \mathcal { X } } ( \tilde { x } _ { i } | h _ { f | c } ^ { i } )$ and non-rationale conditional density $\log p _ { \mathcal { X } } ( \tilde { x } _ { i } | h _ { o } ^ { i } )$ . For simplicity, we omit the superscript i in the subsequent sections without loss of generality.

b) Rationale-Non-Rationale Gaussian Mixture Modeling. Based on NCNF (IV-B), we can derive the conditional densities $\log p _ { \mathcal { X } } ( \tilde { x } _ { i } | h _ { r | c } ^ { i } )$ , $\log p _ { \mathcal { X } } ( \tilde { x } _ { i } | h _ { f | c } ^ { i } )$ , and $\log p _ { \mathcal { X } } ( \tilde { x } _ { i } | h _ { o } ^ { i } )$ for $\tilde { x }$ of node $v _ { i }$ under the representations $h _ { r | c } ^ { i } , \ h _ { f | c } ^ { i }$ , and $h _ { o } ^ { i } .$ , respectively. However, existing normalizing flow models typically adopt a standard multivariate normal distribution $\mathcal { N } ( 0 , \mathbb { I } )$ as the prior and assume that all latent conditional embeddings z follow an identical distribution. This assumption is inherently restrictive in UGAD, as it fails to capture the distinct characteristics of normal and abnormal nodes in the latent space, thereby weakening the ability to distinguish between normal and abnormal patterns.

Therefore, to better fit the normal distribution in the presence of anomalies, we introduce rationale-non-rationale Gaussian mixture modeling (RRGM), which models rationale and non-rationale representations as distinct latent classes. Specifically, RRGM formulates a conditional Gaussian mixture model based on the dependencies among representation classes, where rationale representations correspond to the normal class $( \mathrm { i . e . }$ , label $Y = 0 )$ and non-rationale representations as the abnormal class (i.e., label $Y = 1 )$ . Consequently, RagGAD uses class-dependent mean $u _ { y }$ and covariance $\Sigma _ { y }$ for each class as the prior distribution for $z ,$ defined as:

$$
\begin{array} { r l } { p _ { Z } ( z ) = \displaystyle \sum _ { y \in \{ 0 , 1 \} } p ( y ) \mathcal { N } \left( z ; u _ { y } , \Sigma _ { y } \right) , } & { { } } \\ { p _ { Z | Y } ( z | y ) = \mathcal { N } ( z ; u _ { y } , \Sigma _ { y } ) , } \end{array}\tag{13}
$$

where $u _ { y }$ and $\Sigma _ { y }$ are the mean and covariance matrix of class $y ,$ respectively. p(y) = softmax $\left( W _ { y } ^ { T } z + b _ { y } \right)$ represents the class prior distribution of $z ,$ where $\breve { W } _ { y } \in \breve { \mathbb { R } } ^ { d _ { z } }$ and $b _ { y } \in \mathbb { R }$ are the learnable parameters. RRGM setting the covariance matrix $\Sigma _ { y }$ is the identity matrix I for better convergence. Then, we can calculate the log-likelihood for latent conditional embedding z as follows:

$$
\log p _ { Z } ( z ) = \log \left[ \sum _ { y \in \{ 0 , 1 \} } p ( y ) \mathcal { N } ( z ; u _ { y } , \varSigma _ { y } ) \right]\tag{14}
$$

$$
= \log \left[ \sum _ { y \in \{ 0 , 1 \} } p ( y ) \mathcal { N } ( z ; u _ { y } , \mathbb { I } ) \right]\tag{15}
$$

$$
\Downarrow p _ { Z } ( z ) = ( 2 \pi ) ^ { - \frac { d _ { z } } { 2 } } e ^ { - \frac { 1 } { 2 } } z ^ { T } z ,\tag{16}
$$

$$
= - \frac { d _ { z } } { 2 } \mathrm { l o g } ( 2 \pi ) + \log \left( \sum _ { y \in \{ 0 , 1 \} } e ^ { \epsilon _ { y } } \cdot e ^ { - \frac { \| z - u _ { y } \| _ { 2 } ^ { 2 } } { 2 } } \right)\tag{17}
$$

$$
= - \frac { d _ { z } } { 2 } \mathrm { l o g } ( 2 \pi ) + \mathrm { l o g } \left( \sum _ { y \in \{ 0 , 1 \} } e ^ { - \frac { \| z - u _ { y } \| _ { 2 } ^ { 2 } } { 2 } + \epsilon _ { y } } \right)\tag{18}
$$

$$
= - \frac { d _ { z } } { 2 } \mathrm { l o g } ( 2 \pi ) + \underset { y \in \{ 0 , 1 \} } { \mathrm { o g s u m e x p } } \left( - \frac { \| z - u _ { y } \| _ { 2 } ^ { 2 } } { 2 } + \epsilon _ { y } \right) ,\tag{19}
$$

where $\epsilon _ { y } ~ = ~ \mathrm { l o g } p ( y )$ denotes the logarithmic class weight, scaling the class prior distribution. Further, we bring logp (z) into Eq. (12), and reformulate the conditional log-density $\log p _ { \mathcal { X } } ( \tilde { { \boldsymbol { x } } } | h )$ of x˜ as follows:

$$
\begin{array} { c } { { \displaystyle \log p _ { \mathcal { X } } \big ( \tilde { x } | h \big ) = \log \operatorname * { s u m e x p } _ { y } \left( - \frac { \left\| \mathcal { F } _ { \theta } ( \tilde { x } ; h ) - u _ { y } \right\| _ { 2 } ^ { 2 } } { 2 } + \epsilon _ { y } \right) } } \\ { { - \displaystyle \frac { d _ { z } } { 2 } \mathrm { l o g } ( 2 \pi ) + \log \left| \operatorname* { d e t } J \right| , } } \end{array}\tag{20}
$$

where $J ~ = ~ \nabla _ { \boldsymbol { x } } \mathcal { F } _ { \boldsymbol { \theta } } ( \tilde { \boldsymbol { x } } ; h )$ . The loss function of RRGM is defined as follows:

$$
\begin{array} { r l } & { \mathcal { L } _ { r r g m } = \mathbb { E } _ { \tilde { x } \sim p ( \mathcal { X } ) } [ - \log p \chi ( \tilde { x } | h ) ] } \\ & { \quad \quad \quad = \mathbb { E } _ { \tilde { x } \sim p ( \mathcal { X } ) } [ - \log \operatorname* { s u m e x p } ( - \frac { \| \mathcal { F } _ { \theta } ( \tilde { x } ; h ) - u _ { \mathcal { Y } } \| _ { 2 } ^ { 2 } } { 2 } + \epsilon _ { \mathcal { Y } } ) } \\ & { \quad \quad \quad - \log | \mathrm { d e t } J | + \frac { d _ { z } } { 2 } \mathrm { l o g } ( 2 \pi ) ] . } \end{array}\tag{21}
$$

c) Robust-Fragile Rationale Mixture Learning. In realworld large-scale graphs, the diversification of normal patterns among nodes is a critical problem that must be considered for UGAD. If a normalizing flow model can learn the diverse fine-grained normal distributions of nodes, it can help suppress false detections for divergent patterns within normals. Consequently, we propose RFRM strategy, which captures the diversity among normals nodes through fine-grained rationale Gaussian mixture learning. Specifically, we first extends the Gaussian prior $p _ { Z | Y } ( z | y ) = \mathcal N ( z ; u _ { y } , \varSigma _ { y } )$ to mixture Gaussian prior $\begin{array} { r } { p _ { Z | Y } ( z | y ) = \sum _ { k = 1 } ^ { K } p _ { k } ( y ) \mathcal { N } ( z ; u _ { y } ^ { k } , \varSigma _ { y } ^ { k } ) } \end{array}$ , where K is the number of Gaussian components [43], [60]. If, we further assume $\begin{array} { r } { \sum _ { y } ^ { k } \ = \ \mathbb { I } . } \end{array}$ then the likelihood of latent conditional embedding z can be calculate as follows:

$$
p _ { Z } ( z ) = \sum _ { y } p ( y ) \left( \sum _ { k = 1 } ^ { K } p _ { k } ( y ) \mathcal { N } ( z ; u _ { y } ^ { k } , \Sigma _ { y } ^ { k } ) \right) .\tag{22}
$$

When $p _ { Z } ( z )$ is obtained, the log-likelihood logp<sub>Z</sub>(z) of z can be defined as follows:

$$
\log p _ { Z } ( z ) = \log \left( \sum _ { y } p ( y ) \mathrm { s u m e x p } \Biggl [ - \frac { \left\| z - u _ { y } ^ { k } \right\| _ { 2 } ^ { 2 } } { 2 } + \epsilon _ { y } ^ { k } - \frac { d _ { z } } { 2 } \log ( 2 \pi ) \Biggr ] \right) ,\tag{23}
$$

where the latent conditional embedding $z = \mathcal { F } _ { \boldsymbol { \theta } } ( \tilde { x } ; h ) . ~ \epsilon _ { y } ^ { k } =$ $\log p _ { k } ( y ) = \mathrm { l o g s o f t m a x } _ { k } ( \psi _ { y } )$ denotes the logarithmic component center weights, and $\psi _ { y }$ is a learnable vector specify for class $y ,$ with $\psi _ { y } \in \mathbb { R } ^ { K }$ , to adaptively learn the component weights. Further, we can reformulate Eq. (20) to get the conditional log-density lo $\mathrm { g } p _ { \mathcal { X } } ( \tilde { { \boldsymbol { x } } } | h )$ for any $\tilde { x }$ on condition h when upon mixture Gaussian prior:

$$
\begin{array} { r l } & { \log p _ { \mathcal { X } } ( \tilde { x } | h ) { = } { \log } \left( { \displaystyle \sum _ { y } p ( y ) \mathrm { s u m e x p } \Bigg [ - \frac { \left\| z - u _ { y } ^ { k } \right\| _ { 2 } ^ { 2 } } { 2 } + \epsilon _ { y } ^ { k } - \frac { d _ { z } } { 2 } { \log } ( 2 \pi ) } \Bigg ] \right) } \\ & { \qquad + \log | \mathrm { d e t } J | . } \end{array}\tag{24}
$$

Furthermore, in RFRM, for each class y, we learn a central component $u _ { y } ^ { c }$ and a set offsets $\{ \varDelta u _ { y } ^ { k } \} _ { k = 1 } ^ { K }$ , used to calculate the centers $u _ { y } ^ { \check { k } } = \{ u _ { y } ^ { c } + \varDelta u _ { y } ^ { k } \} _ { k = 1 } ^ { K }$ for other components. We can directly optimize the central component $u _ { y } ^ { c }$ using Eq. (21) for each class y. However, when optimizing $\mathbf { \widetilde { \mathbf { \Gamma } } } u _ { y } ^ { k }$ for other components, we detach the gradient of $u _ { y } ^ { c }$ and optimize the offset $\varDelta u _ { y } ^ { k }$ by the following loss function:

$$
\begin{array} { r l } & { \mathcal { L } _ { r f r m } = \mathbb { E } _ { ( \tilde { x } , y ) \sim p ( \mathcal { X } , Y ) } \big [ - \log p _ { \mathcal { X } } ( \tilde { x } | h ) \big ] } \\ & { \qquad = \mathbb { E } _ { ( \tilde { x } , y ) \sim p ( \mathcal { X } , Y ) } \left[ - \log \left| \operatorname* { d e t } J \right| \right. } \\ & { \qquad \left. - \log \operatorname* { s u m e x p } \Biggl ( - \frac { \left\| z - \left( \mathcal { A } \left[ u _ { y } ^ { c } \right] + \varDelta u _ { y } ^ { k } \right) \right\| _ { 2 } ^ { 2 } } { 2 } + \epsilon _ { y } ^ { k } \Biggr ) \right] , } \end{array}\tag{25}
$$

where $\mathcal { \pmb { G } } \left[ \cdot \right]$ is stop gradient back-propagation. $\textstyle { \frac { d _ { z } } { 2 } } \log ( 2 \pi )$ is a constant term that does not affect the gradient during model optimization, so it is omitted.

For normal class $( \mathrm { i . e . , ~ } Y ~ = ~ 0 )$ , to model the diverse combinations of robust rationales $\mathcal { G } _ { r | c }$ and fragile rationales $\mathcal { G } _ { f | c }$ , we assign rationale latent conditional embedding $z _ { r | c }$ as central component and fragile rationale latent conditional embedding $z _ { f | c }$ as a complementary component. Specifically, we initialize the central component as $u _ { r | c }$ for $z _ { r | c } .$ , and the learnable offset $\varDelta u _ { f | c }$ for $z _ { f | c }$ is defined as follows:

$$
\Delta u _ { f | c } = f _ { \Delta } ( \alpha z _ { f | c } ; \theta _ { \Delta } ) , \quad \alpha = \sigma \left( f _ { \alpha } \left( \left[ z _ { f | c } \middle | | z _ { r | c } \right] ; \theta _ { \alpha } \right) \right) ,\tag{26}
$$

where $f _ { \alpha }$ computes the similarity α between $z _ { r | c }$ and $z _ { f | c } ,$ parameterized by $\theta _ { \alpha }$ , while $f _ { \varDelta }$ generates the offset $\varDelta u _ { f | c } ^ { 1 }$ based on $\alpha z _ { f | c } ,$ parameterized by $\theta _ { \varDelta } . \varDelta u _ { f | c }$ quantifies the offset of the fragile rationale component relative to robust rationale component, and the center of fragile component is $\begin{array} { r } { u _ { f | c } = u _ { r | c } + \varDelta u _ { f | c } . } \end{array}$ . Formally, the loss function of RFRM for $\boldsymbol { u } _ { r | c }$ is defined as $\dot { \mathcal { L } } _ { r r g m } ^ { c }$ directly based on Eq. (21), while the loss function $\mathcal { L } _ { r f r m } ^ { c }$ for $\boldsymbol { u } _ { f | c }$ can be defined by reformulating Eq. (25), as follows:

$$
\begin{array} { l } { { \displaystyle \mathcal { L } _ { r r g m } ^ { c } = \mathbb { E } _ { { \tilde { x } } \sim p ( { \boldsymbol { x } } ) } \left[ - \log \operatorname * { s u m e x p } _ { { \boldsymbol { y } } = 0 } \left( - \frac { \left\| \mathcal { F } _ { \boldsymbol { \theta } } \left( { \boldsymbol { \tilde { x } } } | h _ { r \vert c } \right) - u _ { r \vert c } \right\| _ { 2 } ^ { 2 } } { 2 } + \epsilon _ { \boldsymbol { y } } \right) \right. } } \\ { { \displaystyle \left. - \log \left. \mathrm { d e t } \boldsymbol { J } \right. + \frac { d _ { z } } { 2 } \mathrm { l o g } ( 2 \pi ) \right] , } } \end{array}\tag{27}
$$

$$
\begin{array} { r l } & { \mathcal { L } _ { r f r m } ^ { c } = \mathbb { E } _ { ( \tilde { x } , y ) \sim p ( \mathcal { X } , Y ) } [ - \log | \operatorname* { d e t } J |  } \\ & {  - \log \operatorname* { s u m e x p } \ ( - \frac { | | \mathcal { F } _ { \theta } ( \tilde { x } | h _ { f | c } ) - ( \pmb { \mathscr { E } } [ u _ { r | c } ] + \varDelta u _ { f | c  } | | _ { 2 } ^ { 2 } } { 2 } + \epsilon _ { y } ^ { k } ) ] . } \end{array}\tag{28}
$$

For the abnormal class $( \mathsf { i . e . , } Y = 1 )$ , we employ a twocomponent Gaussian mixture model. The class central $u _ { o }$ and the learnable offset $\varDelta u _ { o }$ are optimized by reformulating Eq. (21) and Eq. (25). The corresponding loss functions $\mathcal { L } _ { r r g m } ^ { o }$ and $\mathcal { L } _ { r f r m } ^ { o }$ are defined as follows:

$$
\begin{array} { r l } & { \mathcal { L } _ { r r g m } ^ { o } = \mathbb { E } _ { \tilde { x } \sim p ( \mathcal { X } ) } \left[ - \underset { y = 1 } { \mathrm { l o g s u m e x p } } \Bigg ( - \frac { \left\| \mathcal { F } _ { \theta } \left( \tilde { x } \left| h _ { o } \right) - u _ { o } \right\| _ { 2 } ^ { 2 } \right)} { 2 } + \epsilon _ { y }  \right. } \\ & { \qquad \left. - \mathrm { l o g } \left| \mathrm { d e t } J \right| + \frac { d _ { z } } { 2 } \mathrm { l o g } ( 2 \pi ) \right] , } \end{array}\tag{29}
$$

$$
\begin{array} { r l } & { \mathcal { L } _ { r f r m } ^ { o } = \mathbb { E } _ { ( \tilde { x } , y ) \sim p ( \mathcal { X } , Y ) } [ - \log | \operatorname* { d e t } { J } |  } \\ & { \qquad - \log \underset { y = 1 } { \operatorname { g s u m e x p } } \Biggl ( - \frac { \| \mathcal { F } _ { \theta } ( \tilde { x } | h _ { o } ) - ( \pmb { \mathscr { G } } [ u _ { o } ] + \varDelta u _ { o } ) \| _ { 2 } ^ { 2 } } { 2 } + \epsilon _ { y } ^ { k } \Biggr ) ] . } \end{array}\tag{30}
$$

Finally, the overall loss function of our RGMN model is defined as:

$$
\mathcal { L } _ { r g m n } = \gamma ( \mathcal { L } _ { r r g m } ^ { c } + \mathcal { L } _ { r f r m } ^ { c } ) + ( 1 - \gamma ) ( \mathcal { L } _ { r r g m } ^ { o } + \mathcal { L } _ { r f r m } ^ { o } ) ,\tag{31}
$$

where hyperparameter $\gamma$ controls the balance between the rationale and non-rationale Gaussian mixture normalizing flow losses.

## C. The Overall Objective and Anomaly Scoring

a) Overall Objective. In the training phase, the overall objective of RagGAD is formulated as follows:

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { 1 } \mathcal { L } _ { r g m n } + \lambda _ { 2 } \mathcal { L } _ { r c o n } + \lambda _ { 3 } \mathcal { L } _ { s p r } , } \end{array}\tag{32}
$$

where $\mathcal { L } _ { \boldsymbol { r c o n } } = \| \mathcal { X } - \hat { \mathcal { X } } \| _ { 2 } ^ { 2 }$ is the attribute reconstruction loss. $\mathcal { L } _ { s p r } = | \mathcal { G } _ { c } |$ is the rationale sparsity loss, which encourages RagGAD to automatically discard weak non-rationale correlations during the optimization process. The hyperparameters $\lambda _ { 1 } , \lambda _ { 2 } .$ , and $\lambda _ { 3 }$ control the relative contributions of each loss term and are tuned via grid search.

b) Anomaly Scoring. Since anomalies deviate significantly from the majority of data instances, we hypothesize that their densities are low. Therefore, for each node $v _ { i } ,$ based on $\operatorname { E q } .$ (24), the overall log-density conditioned on the rationale and non-rationale representations $\begin{array} { r } { h _ { o } , \ h _ { r | c } , } \end{array}$ and $h _ { f | c }$ is computed as follows:

$$
\log p _ { \mathcal { X } } \left( \tilde { x } \vert h \right) = \log p _ { \mathcal { X } } \left( \tilde { x } \vert h _ { o } \right) + \log p _ { \mathcal { X } } \left( \tilde { x } \vert h _ { r \vert c } \right) + \log p _ { \mathcal { X } } \left( \tilde { x } \vert h _ { f \vert c } \right) .\tag{33}
$$

During testing, the anomaly score $S ( x _ { i } )$ for each node $v _ { i }$ is defined as the negative log-density, i.e., $S ( x _ { i } ) = - \mathrm { l o g } p _ { \mathcal X } ( \tilde { x } | h )$ A higher $ { \boldsymbol { S } } (  { \boldsymbol { { x } } } _ { i } )$ indicates a higher likelihood of $v _ { i }$ being abnormal.

## V. EXPERIMENTS

## A. Experimental Setup

a) Datasets. We evaluate RagGAD on ten benchmark datasets for UGAD, following TAM [11] and HUGE [30]. These datasets comprise six real-world datasets, including BlogCatalog [62], ACM [63], Amazon [64], Facebook [65], Reddit [66], and YelpChi [66]. Additionally, four large-scale datasets are used, namely Amazon-all [11], YelpChi-all [26], T-Finance [8], and OGB-Proteins [67]. Among these datasets, BlogCatalog, ACM, and OGB-Protein contain two types of injected anomalies, contextual and structural anomalies. Following [11], [30], all graphs are converted to homogeneous undirected graphs for consistency during evaluation.

b) Baselines. We compare RagGAD with a comprehensive set of state-of-the-art baselines, including Anomalous [61], DOMINANT [7], CoLA [17], SL-GAD [18], HCM-A [29], ComGA [14], GADAM [27], TAM [11], HUGE [30], SmoothGNN [15], GCTAM [36], CoCo [10], and FreeGAD [34]. Following [11], [33], two evaluation metrics, Area Under the Receiver Operating Characteristic Curve (AUROC) and Area Under the precision recall curve (AUPRC), are used for UGAD. The reported AUROC and AUPRC results are averaged over five independent runs with different random seeds. Reported performance for all baselines is obtained from publicly available results or reproduced using their official source code.

c) Implementation Details. RagGAD adopts a consistent strategy for hyperparameter tuning, early stopping, and model selection, following TAM [11] and HUGE [30]. All experiments are performed using the PyTorch framework on an NVIDIA GeForce RTX 3090 (24GB) GPU. The model is optimized with the Adam optimizer over 500 epochs with a learning rate of 1e-4. The projector $\mathcal { P } ( \cdot )$ employs the PCA method, while the graph convolution layer GCN(·) is implemented as a two-layer GCN. The reconstructor $\mathcal { R } ( \cdot )$ is designed as a two-layer MLP to reconstruct node attributes.

## B. Results

a) Comparison with Baselines. Table I presents the AU-ROC and AUPRC results for all baselines across six realworld UGAD datasets with both real and injected anomalies. Overall, RagGAD consistently achieves the best performance across all datasets under both metrics, demonstrating its strong effectiveness and robustness. In terms of AUPRC, which is more sensitive to class imbalance, RagGAD also achieves the best performance across all datasets. RagGAD surpasses TAM on BlogCatalog and ACM by +4.50% (46.32 vs. 41.82) and +3.17% (55.27 vs. 52.10), respectively. On Amazon, Rag-GAD improves over FreeGAD by +2.61% (77.67 vs. 75.06), while on Facebook, it exceeds HUGE by +2.59% (39.33 vs. 36.74). Although the absolute gains on Reddit and YelpChi are relatively smaller (+0.34% and +1.20%), RagGAD still consistently ranks first, demonstrating stable performance un der highly imbalanced and noisy conditions. In terms of AU-ROC, RagGAD ranks first on all six datasets, outperforming the second-best methods by clear margins. Specifically, it improves over TAM on BlogCatalog by +4.84% (87.32 vs. 82.48), over GADAM on ACM by +2.58% (97.24 vs. 94.66), and over CoCo on Amazon by +3.08% (92.04 vs. 88.96). On Facebook, RagGAD achieves 98.62%, exceeding the strong baseline HUGE (97.60) by +1.02%. Even on more challenging datasets such as Reddit and YelpChi, RagGAD still attains the highest scores (63.71% and 81.13%), surpassing the secondbest methods by +1.79% and +2.13%, respectively. Moreover, compared with recent strong methods such as HUGE, TAM, and FreeGAD, which exhibit dataset-dependent performance variations, RagGAD maintains consistently high performance across all scenarios.

b) Performance on Large-scale Graphs. We evaluated RagGAD on four large-scale graph datasets with substantial node and edge counts to assess its effectiveness in handling complex graph structures. The results in Table II demonstrate that RagGAD maintains high effectiveness despite the increased complexity of anomalies, delivering consistent and significant performance across all four datasets. In terms of AUROC, RagGAD outperforms the strongest baselines on all datasets, achieving 93.15% on Amazon-all (+4.23% over HUGE), 63.12% on YelpChi-all (+3.55% over GCTAM),

TABLE I  
AUROC AND AUPRC RESULTS (IN PERCENTAGE) ON SIX REAL-WORLD UGAD DATASETS WITH INJECTED/REAL ANOMALIES. HIGHER AUROC/AUPRC INDICATES BETTER PERFORMANCE. THE BEST RESULTS ARE SHOWN IN BOLD, AND THE SECOND-BEST RESULTS ARE BOLD.
<table><tr><td rowspan="2">Metric</td><td rowspan="2">Method</td><td colspan="6">Dataset</td></tr><tr><td>BlogCatalog</td><td>ACM</td><td>Amazon</td><td>Facebook</td><td>Reddit</td><td>YelpChi</td></tr><tr><td rowspan="12">AUROC</td><td>Anomalous [61]</td><td> $5 6 . 5 2 { \scriptstyle \pm 2 . 5 0 }$ </td><td> $6 8 . 5 6 { \scriptstyle \pm 6 . 3 0 }$ </td><td> $4 4 . 5 7 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $9 0 . 2 1 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $5 3 . 8 7 _ { \pm 1 . 2 0 }$ </td><td> $4 9 . 5 6 _ { \pm 0 . 3 0 }$ </td></tr><tr><td>DOMINANT [7]</td><td> $7 5 . 9 0 { \scriptstyle \pm 1 . 0 0 }$ </td><td> $8 5 . 6 9 { \scriptstyle \pm 2 . 0 0 }$ </td><td> $5 9 . 9 6 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $5 6 . 7 7 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $5 5 . 5 5 { \scriptstyle \pm 1 . 1 0 }$ </td><td> $4 1 . 3 3 { \scriptstyle \pm 1 . 0 0 }$ </td></tr><tr><td>CoLA [17]</td><td> $7 7 . 4 6 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $8 2 . 3 3 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $5 8 . 9 8 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $8 4 . 3 4 { \scriptstyle \pm 1 . 1 0 }$ </td><td> $6 0 . 2 8 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $4 6 . 3 6 { \scriptstyle \pm 0 . 1 0 }$ </td></tr><tr><td>SL-GAD [18]</td><td> $8 1 . 2 3 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $8 4 . 7 9 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $5 9 . 3 7 { \scriptstyle \pm 1 . 1 0 }$ </td><td> $7 9 . 3 6 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $5 6 . 7 7 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $3 3 . 1 2 { \scriptstyle \pm 3 . 5 0 }$ </td></tr><tr><td>HCM-A [29]</td><td> $7 9 . 8 0 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $8 0 . 6 0 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $3 9 . 5 6 { \scriptstyle \pm 1 . 4 0 }$ </td><td> $7 3 . 8 7 { \scriptstyle \pm 3 . 2 0 }$ </td><td> $4 5 . 9 3 { \scriptstyle \pm 1 . 1 0 }$ </td><td> $4 5 . 9 3 { \scriptstyle \pm 0 . 5 0 }$ </td></tr><tr><td>ComGA [14]</td><td> $7 6 . 8 3 _ { \pm 0 . 4 0 }$ </td><td> $8 2 . 2 1 { \scriptstyle \pm 2 . 5 0 }$ </td><td> $5 8 . 9 5 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $6 0 . 5 5 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $5 4 . 5 3 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $4 3 . 9 1 _ { \pm 0 . 0 0 }$ </td></tr><tr><td>GADAM [27]</td><td> $7 9 . 9 2 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $\mathbf { 9 4 . 6 6 { \scriptstyle \pm 0 . 0 0 } }$ </td><td>一  $6 1 . 7 1 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $9 4 . 0 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $5 6 . 5 4 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $4 1 . 7 7 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>TAM [11]</td><td> $8 2 . 4 8 _ { \pm 0 . 3 0 }$ </td><td> $8 8 . 7 8 { \scriptstyle \pm 2 . 4 0 }$ </td><td> $7 0 . 6 4 { \scriptstyle \pm 1 . 0 0 }$ </td><td> $9 1 . 4 4 _ { \pm 0 . 8 0 }$ </td><td> $6 0 . 2 3 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $5 6 . 4 3 _ { \pm 0 . 7 0 }$ </td></tr><tr><td>HUGE [30]</td><td> $6 2 . 1 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $8 3 . 2 7 { \scriptstyle \pm 0 . 0 0 }$ </td><td>85.16±0.00</td><td> $9 7 . 6 0 { \scriptstyle \pm 0 . 0 0 }$  </td><td> $5 9 . 0 6 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $6 0 . 1 3 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>CoCo [10]</td><td> $7 9 . 1 0 { \scriptstyle \pm 0 . 9 3 }$ </td><td> $8 8 . 3 7 { \scriptstyle \pm 1 . 5 6 }$ </td><td> ${ \bf 8 8 . 9 6 _ { \pm 3 . 1 1 } }$  </td><td> $9 6 . 7 5 { \scriptstyle \pm 0 . 0 1 }$ </td><td> ${ \bf 6 1 . 9 2 _ { \pm 0 . 5 1 } }$  </td><td> $6 3 . 8 9 _ { \pm 0 . 0 6 }$ </td></tr><tr><td>SmoothGNN [15]</td><td></td><td></td><td> $8 3 . 1 7 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $4 4 . 4 6 { \scriptstyle \pm 1 . 0 8 }$ </td><td> $5 6 . 8 7 { \scriptstyle \pm 1 . 6 9 }$ </td><td> $5 7 . 4 8 { \scriptstyle \pm 0 . 0 6 }$ </td></tr><tr><td>GCTAM [36]</td><td> $6 2 . 4 7 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $9 0 . 7 7 { \scriptstyle \pm 1 . 5 0 }$ </td><td> $8 4 . 3 8 { \scriptstyle \pm 1 . 4 0 }$ </td><td> $9 2 . 3 8 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $5 9 . 2 1 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $\mathbf { 7 9 . 0 0 _ { \pm 0 . 5 0 } }$ </td></tr><tr><td>RagGAD</td><td>FreeGAD [34]</td><td> $7 4 . 8 4 { \scriptstyle \pm 0 . 0 0 }$   $8 7 . 3 2 _ { \pm 0 . 6 7 }$ </td><td> $8 4 . 8 4 _ { \pm 0 . 0 0 }$   $9 7 . 2 4 _ { \pm 0 . 3 4 } ^ { - }$ </td><td> $8 8 . 5 7 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $9 1 . 5 1 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $5 7 . 2 1 _ { \pm 0 . 0 0 }$ </td><td>1  $7 8 . 5 5 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td rowspan="10">AUPRC</td><td></td><td></td><td></td><td> $9 2 . 0 4 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $9 8 . 6 2 _ { \pm 0 . 2 7 }$  </td><td> $6 3 . 7 1 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $8 1 . 1 3 { \scriptstyle \pm 0 . 1 4 }$  一</td></tr><tr><td>Anomalous [61]</td><td> $6 . 5 2 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $6 . 3 5 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $5 . 5 8 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $1 8 . 9 8 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $3 . 7 5 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $5 . 1 9 { \scriptstyle \pm 0 . 2 0 }$ </td></tr><tr><td>DOMINANT [7]</td><td> $3 1 . 0 2 { \scriptstyle \pm 1 . 1 0 }$ </td><td> $4 4 . 0 2 { \scriptstyle \pm 3 . 6 0 }$ </td><td> $1 4 . 2 4 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $3 . 1 4 { \scriptstyle \pm 4 . 1 0 }$ </td><td> $3 . 5 6 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $3 . 9 5 { \scriptstyle \pm 2 . 0 0 }$ </td></tr><tr><td>CoLA [17]</td><td> $3 2 . 7 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $3 2 . 3 5 { \scriptstyle \pm 1 . 7 0 }$ </td><td> $6 . 7 7 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $2 1 . 0 6 _ { \pm 1 . 7 0 }$ </td><td> $4 . 4 9 _ { \pm 0 . 2 0 }$ </td><td> $4 . 4 8 _ { \pm 0 . 2 0 }$ </td></tr><tr><td>SL-GAD [18]</td><td> $3 8 . 8 2 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $3 7 . 8 4 { \scriptstyle \pm 1 . 1 0 }$ </td><td> $6 . 3 4 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $1 3 . 1 6 { \scriptstyle \pm 2 . 0 0 }$ </td><td> $4 . 0 6 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $3 . 5 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>HCM-A [29]</td><td> $3 1 . 3 9 _ { \pm 0 . 1 0 }$ </td><td> $3 4 . 1 3 _ { \pm 0 . 4 0 }$ </td><td> $5 . 2 7 { \scriptstyle \pm 1 . 5 0 }$ </td><td> $7 . 1 3 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $2 . 8 7 _ { \pm 0 . 5 0 }$ </td><td> $2 . 8 7 _ { \pm 1 . 2 0 }$ </td></tr><tr><td>ComGA [14]</td><td> $3 2 . 9 3 { \scriptstyle \pm 2 . 8 0 }$ </td><td> $2 8 . 7 3 { \scriptstyle \pm 1 . 2 0 }$ </td><td> $1 1 . 5 3 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $3 . 5 4 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $3 . 7 4 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $4 . 2 3 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>GADAM [27]</td><td> $2 8 . 1 4 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $3 3 . 7 7 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 1 . 7 4 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $2 4 . 0 6 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $4 . 6 2 _ { \pm 0 . 0 0 }$ </td><td> $4 . 2 3 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>TAM [11]</td><td> $4 1 . 8 2 _ { \pm 0 . 5 0 }$ </td><td>一  $5 1 . 2 4 { \scriptstyle \pm 1 . 8 0 }$ </td><td> $2 6 . 3 4 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $2 2 . 3 3 { \scriptstyle \pm 1 . 6 0 }$ </td><td> $4 . 4 6 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $7 . 7 8 { \scriptstyle \pm 0 . 9 0 }$ </td></tr><tr><td>HUGE [30]</td><td> $7 . 4 4 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $3 2 . 9 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $6 6 . 7 2 { \scriptstyle \pm 0 . 0 0 }$ </td><td>1  $3 6 . 7 4 { \scriptstyle \pm 0 . 0 0 }$ </td><td> ${ \bf 5 . 1 1 { \scriptstyle \pm 0 . 0 0 } }$ </td><td> $7 . 0 8 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>CoCo [10]</td><td></td><td> $3 7 . 5 3 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $4 0 . 1 9 _ { \pm 2 . 4 4 }$ </td><td> $3 9 . 2 0 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $3 6 . 3 1 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $5 . 0 2 _ { \pm 0 . 1 7 }$ </td><td> $7 . 1 8 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td>SmoothGNN [15]</td><td></td><td></td><td></td><td> $3 9 . 0 5 { \scriptstyle \pm 2 . 3 3 }$ </td><td> $1 . 9 5 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $4 . 1 7 { \scriptstyle \pm 0 . 1 6 }$ </td><td> ${ \bf 1 8 . 1 8 { \scriptstyle \pm 0 . 0 3 } }$  1</td></tr><tr><td>GCTAM [36]</td><td></td><td> $1 2 . 9 0 { \scriptstyle \pm 0 . 6 5 }$ </td><td>_  $5 2 . 1 0 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $5 0 . 6 9 { \scriptstyle \pm 8 . 1 0 }$ </td><td> $2 2 . 8 1 _ { \pm 0 . 6 0 }$ </td><td> $4 . 1 7 _ { \pm 0 . 1 0 }$ </td><td> $1 6 . 0 4 _ { \pm 1 . 0 0 }$ </td></tr><tr><td>FreeGAD [34]</td><td></td><td> $3 4 . 0 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $3 7 . 1 5 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $7 5 . 0 6 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $2 2 . 5 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $3 . 8 5 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 5 . 8 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td colspan="2">RagGAD</td><td> $4 6 . 3 2 _ { \pm 1 . 2 1 }$ </td><td> $5 5 . 2 7 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $7 7 . 6 7 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $3 9 . 3 3 _ { \pm 0 . 4 2 }$ </td><td> $5 . 4 5 _ { \pm 0 . 2 0 }$ </td><td> $1 9 . 3 8 _ { \pm 0 . 4 3 }$  _</td></tr></table>

TABLE II  
RESULTS ON LARGE-SCALE GRAPHS DEMONSTRATE THE EFFECTIVENESS OF EACH BASELINE FOR HANDLING LARGE NUMBERS OF NODES AND EDGES. OOM INDICATES OUT-OF-MEMORY ON A 24GB GPU.

RagGAD remains stable and achieves strong performance across all settings. These results demonstrate that modeling stable rationales while mitigating spurious correlations enables RagGAD to better generalize across diverse graph structures and anomaly types, leading to superior performance in largescale graph anomaly detection.

<table><tr><td rowspan=2 colspan=1>Metric</td><td rowspan=2 colspan=1>Method</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=3>DatasetAmazon-all YelpChi-all T-Finance OGB-Proteins</td></tr><tr><td rowspan=10 colspan=1>AUROC</td><td rowspan=10 colspan=1>DOMINANT [7]ComGA [14]CoLA [17]SL-GAD[18]GADAM[27]TAM[11]GCTAM [36]CoCo[10]HUGE[30]FreeGAD[34]RagGAD</td><td rowspan=1 colspan=3>69.37     53.90    53.80      72.67</td></tr><tr><td rowspan=1 colspan=3>71.54     53.52    55.42      71.34</td></tr><tr><td rowspan=1 colspan=3>26.14     48.01     48.29      71.42</td></tr><tr><td rowspan=1 colspan=3>27.28     55.51    46.48      73.71</td></tr><tr><td rowspan=2 colspan=2>41.55     47.9584.76     58.18</td><td rowspan=1 colspan=1>25.38      73.38</td></tr><tr><td rowspan=1 colspan=1>61.75      74.49</td></tr><tr><td rowspan=1 colspan=2>88.89     59.5785.61     OOM</td><td rowspan=1 colspan=1>OOM      OOMOOM     OOM</td></tr><tr><td rowspan=1 colspan=1>88.92</td><td rowspan=1 colspan=1>57.67</td><td rowspan=1 colspan=1>OOM     OOM</td></tr><tr><td rowspan=1 colspan=1>87.83</td><td rowspan=1 colspan=1>54.17</td><td rowspan=1 colspan=1>92.13      66.51</td></tr><tr><td rowspan=1 colspan=1>93.15</td><td rowspan=1 colspan=1>63.12</td><td rowspan=1 colspan=1>92.98      78.83</td></tr><tr><td rowspan=11 colspan=1>AUPRC</td><td rowspan=11 colspan=1>DOMINANT [7]]ComGA [14]CoLA[17]GADAM[27]SL-GAD[18]TAM[11]GCTAM[36]CoCo[10]HUGE[30]FreeGAD[34]RagGAD</td><td rowspan=1 colspan=1>10.15</td><td rowspan=1 colspan=1>16.38</td><td rowspan=1 colspan=1>4.74      22.17</td></tr><tr><td rowspan=1 colspan=1>18.54</td><td rowspan=1 colspan=1>16.58</td><td rowspan=1 colspan=1>4.81      15.54</td></tr><tr><td rowspan=1 colspan=1>5.16</td><td rowspan=1 colspan=1>13.61</td><td rowspan=1 colspan=1>4.10      13.49</td></tr><tr><td rowspan=1 colspan=1>5.78</td><td rowspan=1 colspan=1>13.79</td><td rowspan=1 colspan=1>3.04      21.04</td></tr><tr><td rowspan=1 colspan=1>4.44</td><td rowspan=1 colspan=1>17.11</td><td rowspan=1 colspan=1>3.86      17.71</td></tr><tr><td rowspan=1 colspan=1>43.46</td><td rowspan=1 colspan=1>18.86</td><td rowspan=1 colspan=1>5.47      21.73</td></tr><tr><td rowspan=1 colspan=1>67.18</td><td rowspan=1 colspan=1>20.13</td><td rowspan=1 colspan=1>OOM     OOM</td></tr><tr><td rowspan=1 colspan=1>70.36</td><td rowspan=1 colspan=1>OOM</td><td rowspan=1 colspan=1>OOM     OOM</td></tr><tr><td rowspan=1 colspan=1>76.68</td><td rowspan=1 colspan=1>18.69</td><td rowspan=1 colspan=1>OOM     OOM</td></tr><tr><td rowspan=1 colspan=1>66.32</td><td rowspan=1 colspan=1>16.96</td><td rowspan=1 colspan=1>73.71      25.21</td></tr><tr><td rowspan=1 colspan=2>80.02     23.25</td><td rowspan=1 colspan=1>74.81      31.44</td></tr></table>

c) Visualization of Anomaly Score. Figure 3(a), presents the homophily distribution of normal and abnormal nodes across benchmark datasets. The results indicate that the Facebook and Amazon-all datasets exhibit one-class homophily, characterized by stronger connectivity or affinity among normal nodes compared to abnormal nodes. In contrast, the YelpChi dataset deviates from this assumption, with connectivity or affinity among abnormal nodes being comparable to, or even surpassing, that among normal nodes. Figure 3(b)- (e), depict the anomaly score distributions for TAM, HUGE, GCTAM, and RagGAD. As shown in Figure 3(b)-(e), TAM fails to establish a clear boundary between the distributions of normal and abnormal nodes across all datasets. HUGE demonstrates improved performance on the Facebook and Amazonall datasets, yet does not differentiate normal and abnormal nodes on the Facebook dataset. Although HUGE identifies the homophily trap and introduces a label-free heterophily measure strategy, it falls short of substantially enhancing the distributional distinction between normal and abnormal nodes. GCTAM extends TAM by incorporating contextual and global affinity to filter anomalous nodes, but it still fails to effectively distinguish between normal and anomalous distributions on the YelpChi dataset. Conversely, RagGAD achieves a clear distributional distinction between normal and abnormal nodes across all datasets. While some overlap in distribution density

92.98% on T-Finance (+0.85% over FreeGAD), and 78.83% on OGB-Proteins (+4.34% over TAM). On more challenging datasets such as T-Finance and OGB-Proteins, where anomalies are highly subtle and distributions are more complex, RagGAD still maintains clear advantages, indicating its robustness in capturing intricate structural patterns. Consistent improvements are also observed under AUPRC, with notable gains such as +3.34% on Amazon-all and +6.23% on OGB-Proteins, demonstrating its effectiveness under highly imbalanced settings. Moreover, several recent methods encounter out-of-memory (OOM) issues on larger datasets, whereas occurs on the YelpChi dataset, the anomaly score distributions of normal and abnormal nodes remain separated, exhibiting clearly differentiated peaks.

TABLE III  
ABLATION STUDIES FOR KEY COMPONENTS IN RAGGAD (IN PERCENTAGE). WE REPORT THE AVERAGE AUROC AND AUPRC OVER FIVE INDEPENDENT RANDOM RUNS. IMP: THE AVERAGE IMPROVEMENT OF EACH VARIANT OVER THE REST (IN PERCENTAGE).
<table><tr><td>Variant</td><td>Amazon AUROC AUPRC</td><td></td><td>Facebook AUROC AUPRC</td><td></td><td>Reddit AUROC AUPRC</td><td>YelpChi-all AUROC AUPRC</td><td></td><td>T-Finance AUROC AUPRC</td><td>Avg (IMP) AUROC AUPRC</td></tr><tr><td>w/o ARD</td><td>75.49</td><td>61.35</td><td>93.15 19.88</td><td>58.01</td><td>4.27</td><td>54.52</td><td>17.84</td><td>78.81 66.08</td><td>72.00 (0.00) 33.88 (0.00)</td></tr><tr><td>w/o RGMN</td><td>73.23</td><td>60.07</td><td>95.36 22.68</td><td>59.36</td><td>4.30</td><td>55.68</td><td>18.09 77.80</td><td>66.02</td><td>72.29 (0.29) 34.23</td></tr><tr><td>w/o RRGM</td><td>86.53</td><td>66.88</td><td>91.08 23.03</td><td>60.37</td><td>4.58</td><td>57.21</td><td>19.60 84.23</td><td>66.79 75.88</td><td>(0.35) (3.60) 35.82 (1.94)</td></tr><tr><td>w/o RFRM</td><td>90.58</td><td>76.03</td><td>95.85 33.03</td><td>61.04</td><td>4.80</td><td>61.57</td><td>21.94 90.84</td><td>70.23 79.97</td><td>(4.09) 38.91 (5.03)</td></tr><tr><td>RagGAD</td><td>92.04</td><td>77.67</td><td>98.62 39.33</td><td>63.71</td><td>5.45</td><td>63.12</td><td>23.25 92.98</td><td>74.81 82.09</td><td>(2.21) 44.10 (2.90)</td></tr></table>

![](images/65a7db2600d630d19deea77d07d3b9f6084e67102747913de5b3142876340c45.jpg)  
（a）Homophily  
（b）TAM  
（c）HUGE  
（d）GCTAM  
（e）RagGAD  
Fig. 3. Visualization of (a) homophily distribution of normal and abnormal nodes, and (b)-(e) the anomaly score of TAM (local affinity) [11], HUGE (label-fre heterophily measure) [30], GCTAM (contextual and global affinity) [36], and RagGAD (rationale-aware conditional log-density) on YelpChi [66], Faceboo [65], and Amazon-all [11] datasets.

d) Visualization of Rationale Graphs. In disentangling non-rationale correlations and rationales between nodes, we visualize the disentangled non-rationale correlations $\mathcal { G } _ { o }$ and rationales $( { \mathcal { G } } _ { r \mid c }$ and $\mathcal { G } _ { f | c } )$ on the Facebook dataset, as shown in Figure 4. The highly challenging Facebook dataset, has the lowest anomaly ratio (2.49%) and the smallest number of nodes, making it an ideal choice for this validation. To improve visualization clarity, we randomly extract 20×20 subgraphs from the 1,080-node of Facebook for display. As shown in Figure 4(b), the disentangled rationales $\mathcal { G } _ { c }$ are sparse, yet they retain most of the node correlations while eliminating certain spurious correlations (highlighted by the red box) from A. Figure 4(c)-(e) demonstrate that robust rationales $\mathcal { G } _ { r \mid c }$ dominate the graph, accounting for approximately 80% of the relationships, compared to fragile rationales $\mathcal { G } _ { f | c } .$ . However, despite their lower proportion, fragile rationales are crucial for diversifying normal patterns. When extended to the entire graph (1081 nodes), the number of fragile rationales becomes non-negligible. Essentially, the presence of fragile rationales greatly expands the diversity of normal node patterns. Therefore, as illustrated in Figure 2, RagGAD further decomposes the rationales into robust and fragile components and performs robust-fragile rationale mixture learning to enhance generalization for anomaly detection.

e) Visualization of Representation. Figure 5 illustrates the t-SNE visualization of robust rationale, fragile rationale, and non-rationale representations extracted by RagGAD from benchmark datasets. The visualization demonstrates that Rag-GAD effectively disentangles rationale and non-rationale representations, further decomposing the rationale components into robust and fragile components. Within RagGAD, the adaptive rationale disentangler (ARD) disentangles rationale and non-rationale correlations from node interdependencies, subsequently decomposing rationale into robust and fragile components. Leveraging rationales, RagGAD derives robust and fragile rationale representations, while non-rationale representations extracted from non-rationale correlations. This enablesprecise modeling and discrimination of conditional densities for normal and abnormal nodes. As shown in Figure 5, RagGAD achieves a compact intra-class feature distribution while distinctly separating clusters of different class representations.

![](images/a9f421300b25b886bcb2c356a05e96836011bee0c03fac6d5d696b278e8eb584.jpg)

![](images/a4f1cca5d6c1503ec7ff4e934d29d515d51d8d5aba43ca7fc1ecf03975bfb858.jpg)  
（e）Non-Rationales Correlations

（c）Robust Rationales  
（d）Fragile Rationales  
Fig. 4. The visualization of inter-node interrelationships, disentangled nonrationale correlations and rationales (robust and fragile) relationships by RagGAD. The deeper the color, the stronger the relationship between nodes. The red box □ indicates correlations between nodes but are not rationales and can thus be removed.  
t-SNE Visualization of Amazon  
![](images/fd12cb35c13bd6f9e43b979ee66dbbe35ae09fc9fe570d0430c792b60d3813e7.jpg)  
t-SNE Visualization of YelpChi

t-SNE Visualization of Tfinance  
![](images/3a0adf5bf4fe57c339be6681cce0e79432dd531f3ef59d7cf6df3c27a8fd9f39.jpg)

![](images/98596ae5d232e992292d14a9f9277330a3e8036131506ff1ab47161d150eeb05.jpg)

t-SNE Visualization of Facebook  
![](images/95ee84a5e99e9daa84b2fe2ae705ae0c2ed88461a6fe881bdcaef822e1f7c98d.jpg)  
Fig. 5. The t-SNE visualization shows the robust rationale representation (■), fragile rationale representation (▲), and non-rationale representation (⋆) learned on four benchmark datasets.

## C. Ablation Study

a) Components Analysis. We investigate the effectiveness of each component in RagGAD across five datasets, as detailed in Table III. Table III demonstrates a progressive improvement in accuracy as more components are integrated, highlighting the essential role of each component in the overall effectiveness of RagGAD. To evaluate the effectiveness of our ARD (w/o ARD), we remove rationale disentangler and instead utilize node interrelationships I to guide information propagation between nodes, performing density estimation based on conventional normalizing flow. The results in Table III indicate that compared to w/o RGMN, which incorporates ARD followed by a conventional normalizing flow, ARD improves model performance, further validating its effectiveness.

We compare RagGAD with two variants that exclude the corresponding components, i.e., rationale-non-rationale gaussian mixture modeling (w/o RRGM) and robust-fragile rationale mixture learning (w/o RFRM), to evaluate the effectiveness of these key components. The results in the lower part of Table III confirm that integrating RRGM and RFRM significantly improves model performance, with RRGM alone demonstrating strong competitiveness. Notably, RRGM plays a pivotal role, as its removal results in a substantial drop in AUPRC and AUROC by approximately 1.94% (35.82 vs. 33.88) and 3.88% (75.88 vs. 72.00) across all datasets compared to RagGAD.

![](images/be6e1ab074ac3fa40f064f1c632ba0ef97e823dc82323860eb5920f3bdb7cb43.jpg)  
Facebook BlogCatalog YelpChi Amazon\_all

![](images/6c7e086c6373836d837e2c74415fcb0d0d7b037ee19bc835dbddeae9b9ba371b.jpg)  
Fig. 6. AUROC and AUPRC of RagGAD w.r.t hyperparameter τ.

![](images/95771c73ba2852b2f09634269499dc66de0e22ddde177a6e09d76931ba75e11d.jpg)  
Fig. 7. Sensitivity analysis of hyperparameter γ on the Facebook dataset.

b) Sensitivity Analysis of τ. In order to evaluate the sensitivity of RagGAD to the hyperparameter τ, we vary its value within the range [0.5, 1] in increments of 0.05 for validation. The parameter τ , which governs the selection of edges as rationales (see Eq.(6)), is a crucial hyperparameter for RagGAD. As shown in Figure 6, as τ increases, RagGAD exhibits a tred of initially rising, then flattening, and finally gradually decreaseing. This variation pattern is reasonable, as a very low value of τ forces the model to incorrectly classify non-rationale correlations as rationales, resulting in excessively high false positives. In contrast, as τ increases, the rationale edge selection strategy becomes stricter. When τ is too high, the model may miss relatively fragile rationales, leading to an increase in rationale false negatives. For AUROC, the impact of changing τ is relatively significant, with a noticeable performance growth between 0.5 and 0.8. Whether for AUROC or AUPRC, when τ is around 0.85, the model achieves its best performance across all datasets. Therefore, in all our experiments, we set the default value of $\tau$ to 0.85.

TABLE IV  
THE EFFECT OF PARAMETERS λ<sub>1</sub>, λ<sub>2</sub>, AND λ<sub>3</sub> ON DIFFERENT DATASETS.
<table><tr><td colspan="2">Dataset</td><td colspan="2">BlogCatalog AURÓC AUPRC</td><td colspan="2">Amazon AUROC AUPRC</td><td colspan="2">Facebook AUROC AUPRC</td><td colspan="2">Reddit AUROĆ AUPRC</td><td colspan="2">YelpChi-all AUROC AUPRC</td><td colspan="2">T-Finance AUROC AUPRC</td><td colspan="2">Avg AUROC ÅUPRC</td></tr><tr><td rowspan="8">0.01  $\lambda _ { 1 } \quad 0 . 1$ </td><td>0.001</td><td>77.10</td><td>35.11</td><td>88.57</td><td>67.94</td><td>89.63</td><td>25.60</td><td>63.33</td><td>5.60</td><td>62.40</td><td>22.37</td><td></td><td>91.76</td><td>59.27</td><td>78.30</td></tr><tr><td></td><td></td><td>34.95</td><td>90.84</td><td>70.80</td><td>91.90</td><td>37.55</td><td>65.62</td><td>5.79</td><td>62.80</td><td>22.79</td><td>89.73</td><td>60.37</td><td></td><td>36.65 79.56 39.69</td></tr><tr><td></td><td>79.50</td><td>45.47</td><td>91.85</td><td>74.85</td><td>98.62</td><td>40.26</td><td>63.49</td><td>5.35</td><td>62.86</td><td>22.79</td><td>92.98</td><td></td><td>68.37</td><td>81.35 39.91</td></tr><tr><td>1</td><td>86.31 87.32</td><td>46.32</td><td>92.04</td><td>77.67</td><td>94.10</td><td>30.58</td><td>63.60</td><td>5.59</td><td>63.12</td><td>23.33</td><td>91.20</td><td></td><td>74.81</td><td>80.88</td></tr><tr><td></td><td>77.04</td><td></td><td>91.93</td><td></td><td>90.20</td><td>22.18</td><td>60.75</td><td>5.15</td><td>62.77</td><td>23.36</td><td>84.77</td><td></td><td></td><td>39.17 77.12</td></tr><tr><td>10</td><td></td><td>36.41</td><td></td><td>75.32</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>47.46</td><td></td><td>35.19</td></tr><tr><td>0.001</td><td>86.74</td><td>47.60</td><td>93.07</td><td>76.20</td><td>93.88</td><td>26.70</td><td>64.28</td><td>6.31</td><td>63.41</td><td>22.63</td><td>89.28</td><td>55.26</td><td>80.96</td><td>40.06</td></tr><tr><td>0.01 0.1</td><td>86.88</td><td>46.68</td><td>91.73</td><td>78.64</td><td>98.60</td><td>34.74</td><td>64.53</td><td>6.49</td><td>64.32</td><td>23.55</td><td>89.41</td><td>56.00</td><td>81.66</td><td>41.96</td></tr><tr><td colspan="2"> $\lambda _ { 2 }$  1 10</td><td>87.32</td><td>46.20 46.70</td><td>91.92 91.15</td><td>74.38 74.00</td><td>99.03 99.08</td><td>37.36 31.41</td><td>65.99 63.03</td><td>6.73 5.58</td><td>62.94 62.15</td><td>22.50 22.30</td><td>91.02 91.50</td><td>66.99</td><td>82.09 81.24</td><td>41.34 40.69</td></tr><tr><td colspan="2"></td><td>86.96 87.13</td><td>46.00</td><td>92.91</td><td>77.01</td><td></td><td>96.40 31.87</td><td>63.04</td><td>5.11</td><td>62.94</td><td>22.45</td><td></td><td>93.45</td><td>86.43 67.30</td><td>81.51 40.65</td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">0.001</td><td>84.55</td><td>47.05</td><td>92.12</td><td>73.02</td><td>89.50</td><td>20.23</td><td>61.27</td><td>5.15</td><td>63.19</td><td>23.06</td><td>78.40</td><td>38.55</td><td></td><td>77.80 36.92</td></tr><tr><td colspan="2">0.01  $\lambda _ { 3 }$ </td><td>87.01</td><td>47.40</td><td>91.45</td><td>76.40</td><td>94.02</td><td>24.84</td><td>65.26</td><td>5.60</td><td>64.38</td><td>22.70</td><td>82.28</td><td>53.78</td><td>80.26</td><td>38.93</td></tr><tr><td colspan="2">0.1 1</td><td>86.32</td><td>47.10</td><td>91.62 91.30</td><td>78.68 73.60</td><td>95.90 99.20</td><td>26.00 36.40</td><td>65.00 63.50</td><td>6.00</td><td>63.04</td><td>22.18</td><td>93.50</td><td>77.00</td><td>81.90 80.93</td><td>40.12</td></tr><tr><td colspan="2">10</td><td>86.80</td><td>46.30</td><td>92.80</td><td></td><td>96.10</td><td></td><td></td><td>5.80</td><td>62.70</td><td>22.95</td><td>88.10</td><td>55.20</td><td></td><td>39.71</td></tr><tr><td colspan="2"></td><td>85.60</td><td>47.90</td><td></td><td>74.60</td><td></td><td>26.90</td><td>62.00</td><td>5.70</td><td>62.90</td><td>22.50</td><td>82.50</td><td>47.10</td><td></td><td>80.00 38.45</td></tr></table>

![](images/5f5898b151718e8511df7d554f2382b84b3d3c06403a62ba9685346851489925.jpg)

![](images/a6b9f7cae69d8b75c4058ee5db72d85a6573f531c949d80c7352e71715f37d83.jpg)  
(a) Performance of CGAD with different  
60.0 40.0 20.0 0.0 20.0 40.0 60.0 80.0 100.0 (b) Performance of CGAD with different  
Fig. 8. Results of parameters sensitivity analysis. (a) shows the effects of different η, (b) shows the influence of embedding dimension $d _ { \tilde { x } }$ on the ACM dataset.

c) Sensitivity Analysis $o f \gamma$ . The hyperparameter γ controls the contributions of the rationales and non-rationale gaussian mixture normalizing flow loss terms (see Eq.(31)). We adjust its value across $[ 0 , 1 ]$ with increments of 0.1 on the Facebook dataset. When $\gamma$ is set to 0, it indicates that the rationale loss term in RGMN is inactive. In constast, when $\gamma$ is set to 1, the model optimizes RGMN based solely on the rationale loss term. As shown in Figure 7, when $\gamma$ is set to 0, both AUROC and AUPRC perform poorly, indicating that our rationale-aware Gaussian mixture normalizing flow plays a crucial role in RagGAD. However, when γ is set to 1, despite the absence of guidance from the non-rationale loss term, the model still performs relatively well, achieving 93.82 and 29.08 for AUROC and AUPRC, respectively. When all loss terms are enabled and γ is set to 0.7 to balance rationale and non-rationale losses, RagGAD performs optimally, achieving AUROC and AUPRC of 98.62 and 39.33, respectively.

d) Sensitivity Analysis of η. The hyperparameter η defines the number of coupling layers in NCNF. We evaluate performance of RagGAD under varying η, with the results presented in Figure 8(a). Overall, RagGAD demonstrates robustness to changes in $\eta$ across both small-scale graphs (BlogCatalog, Facebook, and YelpChi) and the large-scale graph (Amazonall). However, the highest AUROC is achieved at $\eta = 1$ for small-scale graphs, while for large-scale graphs, the optimal setting is $\eta = 2$ . To ensure stable performance across different graph sizes while minimizing parameter redundancy, we set $\eta = 2$ as the default value for all datasets.

TABLE V  
THE MODEL COMPLEXITY, PARAMETER COUNT, AND TRAINING AND TESTING TIME (IN SECONDS) OF DIFFERENT BASELINES. R DENOTES THE NUMBER OF EVALUATION ROUNDS, c DENOTES THE NUMBER OF NODES IN THE LOCAL SUBGRAPH, k IS THE NUMBER OF GLOBAL AFFINITY TRUNCATION GRAPH NEIGHBORS [36], AND ω IS THE AVERAGE NODE DEGREE OF G [27].
<table><tr><td>Method</td><td>Overall Complexity</td><td>Parameters</td><td>Train</td><td>Test</td></tr><tr><td>DOMINANT [7]</td><td> $\mathcal { O } ( \mathcal { E } + N ^ { 2 } )$ </td><td>15769</td><td>67.549</td><td>0.283</td></tr><tr><td>CoLA [17] GADAM [27]</td><td> $\mathcal { O } ( c \dot { N } R ( c + \dot { \omega } ) )$ </td><td>5762</td><td>285</td><td>2.740</td></tr><tr><td></td><td> ${ \dot { \mathcal { O } } } ( { \mathcal { E } } + N )$ </td><td>4226</td><td>6.418</td><td>0.029</td></tr><tr><td>TAM [11]</td><td> $\mathcal { O } ( \mathcal { E } d _ { x } \overset { \cdot } { + } \mathcal { E } d _ { h } \overset { \cdot } { + } N ^ { 2 } )$ </td><td>105091</td><td>1039</td><td>217</td></tr><tr><td>CoCo [10]</td><td> $\mathcal { O } ( \mathcal { E } + N d _ { \tilde { x } } ^ { 2 } + N ^ { 2 } d _ { \tilde { x } } )$ </td><td>12283</td><td>11.26</td><td>0.034</td></tr><tr><td>HUGE [30] GCTAM [36]</td><td> $\mathcal { O } ( N ^ { 2 } d _ { h } )$   $\mathcal { O } ( N d \dot { + } \mathcal { E } d \dot { + } N k )$ </td><td>36355 72320</td><td>18.38</td><td>13.45</td></tr><tr><td></td><td></td><td></td><td>41.72</td><td>0.3876</td></tr><tr><td>RagGAD</td><td> $\mathcal { O } ( N ^ { 2 } d _ { \tilde { x } } )$ </td><td>21439</td><td>87.72</td><td>0.6488</td></tr></table>

e) Sensitivity Analysis of $d _ { \tilde { x } } .$ . The hyperparameter $d _ { \tilde { x } }$ in Eq.(4) denotes the projected attribute dimension for each node. Following GADAM [27], for datasets with low-dimension attributes, we set $d _ { \tilde { x } }$ to 64. For high-dimension attribute datasets, to investigate the impact of $d _ { \tilde { x } }$ on the performance of RagGAD, we vary $d _ { \tilde { x } }$ within the range of 4 to 1024 on the ACM dataset, increasing it in powers of 2. As shown in Figure 8(b), the performance improves as $d _ { \tilde { x } }$ increases but starts to decline when it exceeds 256. To strike a balance between effectiveness and computation efficiency, we set $d _ { \tilde { x } }$ to 128 for all high-dimensional attribute datasets, including BlogCatalog, ACM, and Facebook.

f) Sensitivity Analysis of $\lambda _ { 1 } , \lambda _ { 2 } ,$ and $\lambda _ { 3 } .$ . We perform a grid search to tune these hyperparameters separately for each dataset, with the search range set between 0.001 and 1. As shown in Table IV, for $\lambda _ { 1 } .$ , the optimal performance is obtained in the range of 0.01 to 1. $\gamma _ { 1 }$ controls the strength of loss term $\mathcal { L } _ { r g m n }$ , which is the fundamental loss of RagGAD, the best performance is obtained when $\gamma _ { 1 } = 0 . 1 . \ \gamma _ { 2 }$ and $\lambda _ { 3 }$ control the contributions of the attribute reconstruction loss $\mathcal { L } _ { r c o n }$ and the rationale sparsity loss $\mathcal { L } _ { s p r }$ , respectively. From Table IV, we observe that $\lambda _ { 2 }$ and $\lambda _ { 3 }$ exhibit higher sensitivity compared to $\lambda _ { 1 }$ . Specifically, $\lambda _ { 2 }$ remains relatively stable in the range of 0.001 to 0.1, with a notable decrease in accuracy observed when it exceeds 0.1. $\lambda _ { 3 }$ remains relatively stable between 0.01 and 1, while the performance of RagGAD is suboptimal when $\lambda _ { 3 }$ falls below 0.01. This is mainly because setting $\lambda _ { 2 }$ too high may cause the model to prioritize attribute reconstruction at the expense of accurately disentangling rationales. Conversely, setting $\lambda _ { 3 }$ too low leads RagGAD to ignore the sparsity of rationales, resulting in the misclassification of weak nonrationale correlations as rationales.

## D. Efficiency and Complexity Analysis

We evaluate efficiency from model complexity, parameter count, and empirical runtime, as summarized in Table V. All runtimes are measured on the Amazon dataset, where the training time corresponds to 100 epochs and the testing time reflects standard inference cost.

In terms of complexity, most baselines scale linearly with E or N, such as GADAM and GCTAM, while methods modeling pairwise node interactions introduce quadratic terms $\mathcal { O } ( N ^ { 2 } )$ including DOMINANT, TAM, and HUGE. RagGAD falls into this category with complexity $\mathcal { O } ( N ^ { 2 } d _ { \tilde { x } } )$ , due to global nodewise interaction and rationale disentanglement in the ARD module. Compared with local or sampling-based methods, this design captures long-range dependencies and high-order structural correlations, which are critical for accurate anomaly characterization. Despite the quadratic complexity, RagGAD maintains a moderate parameter size of 21,439, which is significantly smaller than TAM and comparable to GCTAM and HUGE. This indicates that the additional cost stems from dense computation rather than over-parameterization, reflecting efficient parameter utilization. Empirically, RagGAD achieves competitive efficiency. Its training time of 87.72 seconds is much lower than TAM and CoLA, and its inference time of 0.6488 seconds remains practical, outperforming several high-complexity methods. This suggests that the theoretical complexity does not directly translate into prohibitive runtime, as optimized matrix operations and parallelism mitigate the overhead in practice.

Overall, RagGAD introduces additional computation for global dependency modeling and rationale disentanglement, leading to improved representation robustness, while maintaining a favorable balance between effectiveness and efficiency without excessive parameter or runtime overhead.

## VI. CONCLUSION

In this paper, we propose RagGAD, a novel unsupervised graph anomaly detection framework that disentangles robust rationales from non-rationale correlations in node interrelationships. By further decomposing rationales into robust and fragile components, RagGAD captures the underlying mechanisms governing normal behavior while identifying anomalies through deviations in non-rationale interactions. To model the complex distributions of both normal and abnormal nodes, we incorporate a node-level rationale-aware conditional Gaussian mixture normalizing flow that enhances sensitivity to the diversity of normal patterns and mitigates the homophily trap. Extensive experiments on benchmark datasets demonstrate that RagGAD outperforms state-of-the-art baselines, confirming its effectiveness in addressing the challenges of unsupervised graph anomaly detection. Future work will focus on optimizing the model to improve efficiency without sacrificing performance, and expanding its applicability to dynamic and heterogeneous graph scenarios.

## REFERENCES

[1] X. Li, X. Li, J. Jia, L. Li, J. Yuan, Y. Gao, and S. Yu, “A high accuracy and adaptive anomaly detection model with dual-domain graph convolutional network for insider threat detection,” IEEE Transactions on Information Forensics and Security, vol. 18, pp. 1638–1652, 2023.

[2] H. Qiao, H. Tong, B. An, I. King, C. Aggarwal, and G. Pang, “Deep graph anomaly detection: A survey and new perspectives,” arXiv preprint arXiv:2409.09957, 2024.

[3] X. Li, X. Jiang, H. Wan, and X. Zhao, “Tered: Normal behavior-based efficient provenance graph reduction for large-scale attack forensics,” IEEE Transactions on Information Forensics and Security, 2025.

[4] R. Li, Z. Liu, Y. Ma, D. Yang, and S. Sun, “Internet financial fraud detection based on graph learning,” IEEE Transactions on Computational Social Systems, vol. 10, no. 3, pp. 1394–1401, 2022.

[5] Y. Liu, X. Ao, Z. Qin, J. Chi, J. Feng, H. Yang, and Q. He, “Pick and choose: A gnn-based imbalanced learning approach for fraud detection,” in Proceedings of the Web Conference, 2021, pp. 3168–3177.

[6] C. Wang and H. Zhu, “Wrongdoing monitor: A graph-based behavioral anomaly detection in cyber security,” IEEE Transactions on Information Forensics and Security, vol. 17, pp. 2703–2718, 2022.

[7] K. Ding, J. Li, R. Bhanushali, and H. Liu, “Deep anomaly detection on attributed networks,” in Proceedings of the SIAM International Conference on Data Mining. SIAM, 2019, pp. 594–602.

[8] J. Tang, J. Li, Z. Gao, and J. Li, “Rethinking graph neural networks for anomaly detection,” in International Conference on Machine Learning, 2022, pp. 21 076–21 089.

[9] C. Niu, H. Qiao, C. Chen, L. Chen, and G. Pang, “Zero-shot generalist graph anomaly detection with unified neighborhood prompts,” arXiv preprint arXiv:2410.14886, 2024.

[10] R. Wang, L. Xi, F. Zhang, H. Fan, X. Yu, L. Liu, S. Yu, and V. C. Leung, “Context correlation discrepancy analysis for graph anomaly detection,” IEEE Transactions on Knowledge and Data Engineering, vol. 37, no. 1, pp. 174–187, 2025.

[11] H. Qiao and G. Pang, “Truncated affinity maximization: One-class homophily modeling for graph anomaly detection,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[12] J. He, Q. Xu, Y. Jiang, Z. Wang, and Q. Huang, “Ada-gad: Anomalydenoised autoencoders for graph anomaly detection,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 8, 2024, pp. 8481–8489.

[13] A. Roy, J. Shu, J. Li, C. Yang, O. Elshocht, J. Smeets, and P. Li, “Gad-nr: Graph anomaly detection via neighborhood reconstruction,” in Proceedings of the ACM International Conference on Web Search and Data Mining, 2024, pp. 576–585.

[14] X. Luo, J. Wu, A. Beheshti, J. Yang, X. Zhang, Y. Wang, and S. Xue, “Comga: Community-aware attributed graph anomaly detection,” in Proceedings of the ACM International Conference on Web Search and Data Mining, 2022, pp. 657–665.

[15] X. Dong, X. Zhang, Y. Sun, L. Chen, M. Yuan, and S. Wang, “Smoothgnn: Smoothing-aware gnn for unsupervised node anomaly detection,” in Proceedings of the ACM on Web Conference 2025, 2025, pp. 1225–1236.

[16] J. Liu, M. He, X. Shang, J. Shi, B. Cui, and H. Yin, “Bourne: Bootstrapped self-supervised learning framework for unified graph anomaly detection,” in IEEE International Conference on Data Engineering. IEEE, 2024, pp. 2820–2833.

[17] Y. Liu, Z. Li, S. Pan, C. Gong, C. Zhou, and G. Karypis, “Anomaly detection on attributed networks via contrastive self-supervised learning,” IEEE Transactions on Neural Networks and Learning Systems, vol. 33, no. 6, pp. 2378–2392, 2021.

[18] Y. Zheng, M. Jin, Y. Liu, L. Chi, K. T. Phan, and Y.-P. P. Chen, “Generative and contrastive self-supervised learning for graph anomaly detection,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 12, pp. 12 220–12 233, 2021.

[19] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” arXiv preprint arXiv:1609.02907, 2016.

[20] N. Chen, Z. Liu, B. Hooi, B. He, R. Fathony, J. Hu, and J. Chen, “Consistency training with learnable data augmentation for graph anomaly detection with limited supervision,” in International Conference on Learning Representations, 2024.

[21] J. Tang, F. Hua, Z. Gao, P. Zhao, and J. Li, “Gadbench: Revisiting and benchmarking supervised graph anomaly detection,” Advances in Neural Information Processing Systems, vol. 36, pp. 29 628–29 653, 2023.

[22] G. AI, H. QIAO, H. YAN, and G. PANG, “Semi-supervised graph anomaly detection via robust homophily learning.” in Proceedings of the Conference on Neural Information Processing Systems, 2025, pp. 2–7.

[23] M. Huang, Y. Liu, X. Ao, K. Li, J. Chi, J. Feng, H. Yang, and Q. He, “Auc-oriented graph neural network for fraud detection,” in Proceedings of the ACM Web Conference, 2022, pp. 1311–1321.

[24] H. QIAO, Q. WEN, and X. LI, “Generative semi-supervised graph anomaly detection.” in Advances in Neural Information Processing Systems, pp. 10–15.

[25] Y. Gao, X. Wang, X. He, Z. Liu, H. Feng, and Y. Zhang, “Addressing heterophily in graph anomaly detection: A perspective of graph spectrum,” in Proceedings of the ACM Web Cconference, 2023, pp. 1528–1538.

[26] B. Chen, J. Zhang, X. Zhang, Y. Dong, J. Song, P. Zhang, K. Xu, E. Kharlamov, and J. Tang, “Gccad: Graph contrastive coding for anomaly detection,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 8, pp. 8037–8051, 2022.

[27] J. Chen, G. Zhu, C. Yuan, and Y. Huang, “Boosting graph anomaly detection with adaptive message passing,” in International Conference on Learning Representations, 2024.

[28] X. Wang, B. Jin, Y. Du, P. Cui, Y. Tan, and Y. Yang, “One-class graph neural networks for anomaly detection in attributed networks,” Neural Computing and Applications, vol. 33, pp. 12 073–12 085, 2021.

[29] T. Huang, Y. Pei, V. Menkovski, and M. Pechenizkiy, “Hop-count based self-supervised anomaly detection on attributed networks,” in Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, 2022, pp. 225–241.

[30] J. Pan, Y. Liu, X. Zheng, Y. Zheng, A. W.-C. Liew, F. Li, and S. Pan, “A label-free heterophily-guided approach for unsupervised graph fraud detection,” arXiv preprint arXiv:2502.13308, 2025.

[31] J. Pan, Y. Liu, Y. Zheng, and S. Pan, “Prem: A simple yet effective approach for node-level graph anomaly detection,” in IEEE International Conference on Data Mining. IEEE, 2023, pp. 1253–1258.

[32] T. Zhao, C. Deng, K. Yu, T. Jiang, D. Wang, and M. Jiang, “Errorbounded graph anomaly loss for gnns,” in Proceedings of the ACM International Conference on Information & Knowledge Management, 2020, pp. 1873–1882.

[33] Y. Liu, S. Li, Y. Zheng, Q. Chen, C. Zhang, and S. Pan, “Arc: A generalist graph anomaly detector with in-context learning,” in Advances in Neural Information Processing Systems, 2024.

[34] Y. Zhao, Y. Liu, S. Li, Q. Chen, Y. Zheng, and S. Pan, “Freegad: A training-free yet effective approach for graph anomaly detection,” in Proceedings of the ACM International Conference on Information and Knowledge Management, 2025, pp. 4379–4389.

[35] Q. Wang, G. Pang, M. Salehi, W. Buntine, and C. Leckie, “Cross-domain graph anomaly detection via aanomaly-aware contrastive alignment,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 4, 2023, pp. 4676–4684.

[36] X. Zhang, H. Peng, Z. He, C. Xie, X. Jin, and H. Jiang, “Gctam: global and contextual truncated affinity combined maximization model for unsupervised graph anomaly detection,” in Proceedings of the International Joint Conference on Artificial Intelligence, 2025, pp. 3642– 3650.

[37] G. Papamakarios, E. Nalisnick, D. J. Rezende, S. Mohamed, and B. Lakshminarayanan, “Normalizing flows for probabilistic modeling and inference,” Journal of Machine Learning Research, vol. 22, no. 57, pp. 1–64, 2021.

[38] R. Cao, S. Xue, J. Li, Q. Wang, and Y. Chang, “Fanfold: Graph normalizing flows-driven asymmetric network for unsupervised graphlevel anomaly detection,” arXiv preprint arXiv:2407.00383, 2024.

[39] M. Rudolph, B. Wandt, and B. Rosenhahn, “Same same but differnet: Semi-supervised defect detection with normalizing flows,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2021, pp. 1907–1916.

[40] E. Dai and J. Chen, “Graph-augmented normalizing flows for anomaly detection of multiple time series,” in International Conference on Learning Representations, 2022.

[41] D. Gudovskiy, S. Ishizaka, and K. Kozuka, “Cflow-ad: Real-time unsupervised anomaly detection with localization via conditional normalizing flows,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2022, pp. 98–107.

[42] M. Rudolph, T. Wehrbein, B. Rosenhahn, and B. Wandt, “Fully convolutional cross-scale-flows for image-based defect detection,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2022, pp. 1088–1097.

[43] X. Yao, R. Li, Z. Qian, L. Wang, and C. Zhang, “Hierarchical gaussian mixture normalizing flow modeling for unified anomaly detection,” in European Conference on Computer Vision. Springer, 2025, pp. 92–108.

[44] X. Yao, R. Li, J. Zhang, J. Sun, and C. Zhang, “Explicit boundary guided semi-push-ppull contrastive learning for supervised anomaly detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 24 490–24 499.

[45] J. Liu, A. Kumar, J. Ba, J. Kiros, and K. Swersky, “Graph normalizing flows,” Advances in Neural Information Processing Systems, vol. 32, 2019.

[46] L. Dinh, D. Krueger, and Y. Bengio, “Nice: Non-linear independent components estimation,” arXiv preprint arXiv:1410.8516, 2014.

[47] D. P. Kingma and P. Dhariwal, “Glow: Generative flow with invertible 1x1 convolutions,” Advances in Neural Information Processing Systems, vol. 31, 2018.

[48] L. Dinh, J. Sohl-Dickstein, and S. Bengio, “Density estimation using real-nvp,” arXiv preprint arXiv:1605.08803, 2016.

[49] C. Villani, Topics in Optimal Transportation. American Mathematical Soc., 2021, vol. 58.

[50] L. Ardizzone, C. Luth, J. Kruse, C. Rother, and U. K¨ othe, “Guided image¨ generation with conditional invertible neural networks,” arXiv preprint arXiv:1907.02392, 2019.

[51] Q. Zhou, S. He, H. Liu, J. Chen, and W. Meng, “Label-free multivariate time series anomaly detection,” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 7, pp. 3166–3179, 2024.

[52] A. Abdelhamed, M. A. Brubaker, and M. S. Brown, “Noise flow: Noise modeling with conditional normalizing flows,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 3165–3173.

[53] H. Zhao, A. Chen, X. Sun, H. Cheng, and J. Li, “All in one and one for all: A simple yet effective method towards cross-domain graph pretraining,” in Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2024, pp. 4443–4454.

[54] G. W. Stewart, “On the early history of the singular value decomposition,” SIAM Review, vol. 35, no. 4, pp. 551–566, 1993.

[55] H. Abdi and L. J. Williams, “Principal component analysis,” Wiley Interdisciplinary Reviews: Computational Statistics, vol. 2, no. 4, pp. 433–459, 2010.

[56] J. Lu and S. Sun, “CauDiTS: Causal disentangled domain adaptation of multivariate time series,” in International Conference on Machine Learning, vol. 235, 2024, pp. 33 113–33 146.

[57] C. Wu, J. Sun, J. Chen, M. Alazab, Y. Liu, and Y. Xiang, “Tcgids: Robust network intrusion detection via temporal contrastive graph learning,” IEEE Transactions on Information Forensics and Security, vol. 20, pp. 1475–1486, 2025.

[58] Q. Zhou, J. Chen, H. Liu, S. He, and W. Meng, “Detecting multivariate time series anomalies with zero known label,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 4, 2023, pp. 4963–4971.

[59] R. Wang, K. Nie, T. Wang, Y. Yang, and B. Long, “Deep learning for anomaly detection,” in Proceedings of the International Conference on Web Search and Data Mining, 2020, pp. 894–896.

[60] P. Izmailov, P. Kirichenko, M. Finzi, and A. G. Wilson, “Semi-supervised learning with normalizing flows,” in International Conference on Machine Learning, 2020, pp. 4615–4630.

[61] Z. Peng, M. Luo, J. Li, H. Liu, Q. Zheng et al., “Anomalous: A joint modeling approach for anomaly detection on attributed networks.” in International Joint Conference on Artificial Intelligence, vol. 18, 2018, pp. 3513–3519.

[62] L. Tang and H. Liu, “Relational learning via latent social dimensions,” in Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2009, pp. 817–826.

[63] J. Tang, J. Zhang, L. Yao, J. Li, L. Zhang, and Z. Su, “Arnetminer: Extraction and mining of academic social networks,” in Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2008, pp. 990–998.

[64] Y. Dou, Z. Liu, L. Sun, Y. Deng, H. Peng, and P. S. Yu, “Enhancing graph neural network-based fraud detectors against camouflaged fraudsters,” in Proceedings of the ACM International Conference on Information & Knowledge Management, 2020, pp. 315–324.

[65] Z. Xu, X. Huang, Y. Zhao, Y. Dong, and J. Li, “Contrastive attributed network anomaly detection with data augmentation,” in Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, 2022, pp. 444–457.

[66] S. Kumar, X. Zhang, and J. Leskovec, “Predicting dynamic embedding trajectory in temporal interaction networks,” in Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2019, pp. 1269–1278.

[67] W. Hu, M. Fey, M. Zitnik, Y. Dong, H. Ren, B. Liu, M. Catasta, and J. Leskovec, “Open graph benchmark: Datasets for machine learning on graphs,” Advances in Neural Information Processing Systems, vol. 33, pp. 22 118–22 133, 2020.