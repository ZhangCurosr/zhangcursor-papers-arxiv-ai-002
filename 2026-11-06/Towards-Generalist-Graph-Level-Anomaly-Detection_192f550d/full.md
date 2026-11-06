# Towards Generalist Graph-Level Anomaly Detection

Junjun Pan<sup>1</sup>, Yixin Liu<sup>1∗</sup>, Yu Zheng<sup>1</sup>, Fuyi Li<sup>2</sup>, Alan Wee-Chung Liew<sup>1</sup>, Shirui Pan<sup>1</sup> <sup>1</sup>School of Information and Communication Technology, Griffith University, Australia <sup>2</sup>South Australian immunoGENomics Cancer Institute (SAiGENCI), Adelaide University, Australia junjun.pan@griffithuni.edu.au, fuyi.li@adelaide.edu.au {yixin.liu, yu.zheng, a.liew, s.pan}@griffith.edu.au

## Abstract

Graph-level anomaly detection (GLAD) aims to identify graphs that deviate significantly from the majority and plays a crucial role in applications such as molecular analysis and fraud detection. While recent advances have achieved competitive performance, existing approaches mainly follow a dataset-specific paradigm, requiring substantial in-domain data and training cost when applied to new application scenarios. This limits their scalability and practicality in real-world applications where labeled data is scarce and diverse domains are continuously emerging. In this paper, we investigate the problem of generalist GLAD, which seeks to learn a single unified model capable of detecting anomalous graphs across multiple domains with minimal supervision on the target data. This problem introduces key challenges, including learning transferable patterns from heterogeneous graph datasets and effective adaptation under limited supervision. To address these challenges, we propose GenGLAD, a generalist GLAD approach that leverages discriminative subgraph extraction as the core mechanism to learn transferable anomaly knowledge and enable efficient adaptation to new domains. Specifically, GenGLAD employs a graph neural network-based discriminative subgraph extraction model to learn the anomaly-discriminative substructures that distinguish anomalous and normal graphs across diverse domains. During the test-time adaptation phase, multi-dimensional discrepancies between the extracted subgraph and multi-level contextual information are calculated to characterize anomalies from multiple perspectives. For efficient adaptation, a lightweight scoring module is introduced to automatically decide the most informative signals, improving robustness against misleading patterns. Extensive experiments on diverse benchmark datasets demonstrate that the proposed approach achieves strong performance, superior generalization ability, and improved data efficiency compared to existing methods.

## 1 Introduction

Graphs are a ubiquitous data structure for modeling discrete objects and their relationships, which makes them widely used in fields such as chemistry, biology, and social network analysis [46, 56]. Building on this foundation, graph-level anomaly detection (GLAD) aims to identify graph instances that are significantly different from the majority of graphs [18]. Due to its strong capability for solving various real-world challenges, such as identifying toxic chemicals [2], spotting fraudster networks [25] and safeguarding multi-agent systems [28], GLAD has attracted sustained research attention. Recently, graph neural network (GNN)-based methods have become the dominant approach for the GLAD problem, showcasing strong performance compared to traditional kernel-based methods [18, 25, 43].

![](images/a95342a161f47d155b6124f4d4301164bed188909b6377bc0a6515329839fc98.jpg)  
(a) Generalist GLAD paradigm.

![](images/903c9f81ad6355c5ec1e1b83e7914425f4cbfaed7d470dabbad10d56b1cd6a5f.jpg)  
(b) AUROC comparison.  
Figure 1: (a) Sketch map the generalist GLAD paradigm, where one model fits multiple new datasets. (b) Performance of unsupervised GLAD methods reduces with the reduction of training set size.

Despite the competitive detection performance they achieved, existing GLAD approaches follow the traditional “one model per dataset” paradigm, requiring large amounts of in-domain data to train a dedicated detection model for each dataset. This learning paradigm brings significant limitations for GLAD applications. On the one hand, deploying a dataset-specific model requires sufficient in-domain data for model training, which limits its practical applicability in domains where data are scarce or difficult to collect. On the other hand, training a dataset-specific model is usually computationally expensive, involving high time and hardware costs, making it inefficient for large-scale databases.

Given the aforementioned limitations, a natural question arises: Can we train a “one-for-all” GLAD model that can generalize to detecting anomalous graphs across diverse application scenarios, without heavy adaptation cost? Following the previous success of generalist anomaly detection methods [19, 58], developing a generalist GLAD approach is a promising solution to the question. As shown in Figure 1a, a generalist GLAD model can be trained on a large collection of diverse datasets, acquiring strong generalization capabilities for various GLAD tasks. When applied to an arbitrary unseen target dataset, it only requires a few-shot normal samples for minimal-cost adaptation to detect anomalous graphs. As the knowledge learned from large-scale pre-training can be transferred to target datasets, the generalist GLAD paradigm eliminates the need for heavy dataset-specific training, reducing both data requirements and training costs.

Despite the promising potential of the generalist paradigm, it is non-trivial to develop a generalist GLAD approach. While recent efforts such as ARC [19] and AnomalyGFM [34] have promising generalist capabilities for node-level graph anomaly detection, extending such paradigms to GLAD comes with unique challenges. Firstly, (C1) How to learn generalizable patterns from diverse training datasets? Different graph-level applications lie in different semantic spaces and different structural priors. As a result, the characteristics of normal and abnormal graphs can vary widely across domains, which makes obtaining generalizable knowledge difficult. For example, in a social network, a fraudster community is usually characterized by dense local structures due to spamming behavior, while in molecular graphs, a small motif that exhibits specific functional groups can determine biotoxicity. In this case, how to develop a generalizable model that can capture transferable patterns across diverse domains remains a fundamental challenge. Moreover, (C2) How can models effectively adapt to a new test domain under limited supervision? Even if a generalist model can learn transferable patterns across domains, making reliable prediction on a target dataset still requires leveraging a few-shot set of normal samples, which characterize the specific distributional properties of the dataset. Nevertheless, unsupervised GLAD models typically rely heavily on sufficient data to capture the normal distribution: as shown in Figure 1b, their performance degrades sharply as the amount of training data decreases. In this context, how to effectively exploit the limited few-shot normal samples for efficient adaptation of pre-trained generalist GLAD models to unseen domains still remains an open problem.

To address the aforementioned challenges, we propose GenGLAD, a Generalist GLAD approach based on discriminative subgraph extraction. Our core idea is to learn transferable anomalydiscriminative subgraphs (e.g., the substructures in red in Figure 1a) through cross-domain pretraining, and adapt them to unseen domains via discrepancy-based evidence modeling under few-shot supervision. Specifically, to tackle (C1), GenGLAD incorporates a memory-guided discriminative subgraph extraction model as its backbone network, learning to capture shared fundamental knowledge for identifying the most anomaly-discriminative substructures that distinguish normal and anomalous graphs from arbitrary graph samples. To enhance transferable knowledge learning, GenGLAD empowers cross-domain pre-training by storing learned knowledge in a memory bank, and also facilitating task-relevant knowledge retrieval with the help of contextual priors. Moreover, to address (C2), we construct multi-dimensional discrepancy scores between the discriminative subgraph of each graph sample and the contextual information, using them as prediction evidence. In this way, GenGLAD not only leverages few-shot contextual samples for domain-specific adaptation, but also captures anomaly evidence from fine-grained instance-level cues to coarse-grained cross-domain knowledge. With the discrepancy-based evidence, we can perform efficient adaptation by learning a lightweight scoring module with a linear layer to support anomaly prediction on new datasets. To summarize, this paper makes the following contributions:

• Problem. To the best of our knowledge, we are the first to formalize and investigate the generalist GLAD problem, aiming to detect anomalous graphs from various domains with a single pre-trained GLAD model with limited in-domain supervision when adapting to arbitrary unseen datasets.

• Method. We propose a novel generalist GLAD method, GenGLAD, which learns transferable anomaly-discriminative substructures through memory-guided cross-domain pre-training and performs efficient few-shot adaptation via discrepancy-based evidence modeling.

• Experiments. We conduct extensive experiments to demonstrate the strong performance of GenGLAD, and illustrate its generalizability and efficiency across 10 benchmark datasets from various application domains.

## 2 Related Work

Graph-Level Anomaly Detection aims to identify anomalous graphs that significantly deviate from the majority of normal graphs in a dataset [24, 43]. Existing methods can be broadly categorized into two groups, namely, reconstruction-based methods and contrastive learning-based methods. The former aim to reconstruct input graphs and detect anomalies based on reconstruction errors, assuming that anomalous graphs are harder to reconstruct [22, 5, 14, 55]. The latter learn graph representations by contrasting multiple views of the data, using consistency or agreement between views to distinguish normal patterns from anomalies [15, 18, 50, 11]. There are also several studies that leverage advanced techniques, such as evolutionary mapping [25], mixture-of-experts [4], and knowledge distillation [23], to address the GLAD problem. Existing GLAD methods usually follow a one-dataset-one-model paradigm, where models are trained using either purely normal data (one-class unsupervised setting [43]) or labeled normal and anomalous data (supervised setting [52, 25, 47]), and evaluated on the same dataset. This paradigm increases the requirement for large amounts of labeled or curated data and limits its broad applicability.

Generalist Anomaly Detection is an emerging paradigm that aims to train a universal anomaly detection model on diverse datasets and generalize it to unseen datasets with minimal supervision and cost [58, 31]. Recently, generalist anomaly detection approaches have achieved significant progress and demonstrated strong performance across various domains, such as industrial image data [58], time-series data [37], relational data [17], and tabular data [16]. In text data, such techniques can be utilized to spot hallucination [33, 6] or fraudsters [30]. Recent visual approaches further enhance generalization through residual feature modeling [51], normal-abnormal guidance [44], and residual guided normality learning [42]. For graph data, recent studies investigate generalist approaches for node-level anomaly detection [19, 34, 20]. While these methods have shown promising performance, they cannot be directly applied to GLAD due to fundamental differences between node-level and graph-level abnormal patterns: node-level methods typically rely on several assumptions (e.g., the homophily assumption [29]), which may not hold in the graph-level counterpart [32].

A more comprehensive and detailed literature review is given in Appendix A.

## 3 Problem Statement

Notations. In general, a graph can be presented as $\mathcal { G } = ( \nu , \mathcal { E } )$ , where $\mathcal { V } = \{ v _ { 1 } , \ldots , v _ { n } \}$ is the set of n nodes and $\mathcal { E } \subseteq \mathcal { V } \times \mathcal { V }$ is the set of edges, which can also be presented in adjacency matrix form $\mathbf { A } \in \{ 0 , 1 \} ^ { n \times n }$ , with $\mathbf { A } _ { i , j } = 1 \mathrm { i f f } \left( v _ { i } , v _ { j } \right) \in \mathcal { E }$ . For a node $v \in \mathcal { V } , \mathcal { N } ( v )$ denotes its one-hop neighborhood and $\deg ( v ) = | \tilde { \mathcal { N } } ( v )$ | its degree.

Conventional GLAD Problem. GLAD aims to identify anomalous graphs from a collection of graphs by learning an anomaly scoring function $f : { \mathcal { G } }  ]$ R where higher scores correspond to greater degrees of abnormality. The function $f$ is optimized on the training split of target datasets $\mathbf { \bar { \boldsymbol { D } } } = \{ \mathcal { G } _ { 1 } , \ldots , \mathcal { G } _ { K } \}$ . The mainstream studies of GLAD usually follow an one-class (sometimes a.k.a.

![](images/6a02afeb2ee50ad957a5caf10e825cbf93bfea86ea29064a38b6e56e03055387.jpg)  
Figure 2: The overall framework of GenGLAD. During the pre-training phase, a discriminative subgraph extraction model learns the transferable patterns for generalist GLAD. During adaptation, multi-dimensional discrepancy is constructed for anomaly scoring on target dataset. Code available at https://github.com/CampanulaBells/Generalist-Graph-Level-Anomaly-Detection.

unsupervised) setting, where the model is trained on a collection of normal graphs and is then applied to detect anomalies in previously unseen graphs. The inductive property makes GLAD challenging, as anomalous patterns are not observed during training.

Generalist GLAD Problem. In this paper, we study the generalist GLAD problem, which aims to develop a generalist GLAD model capable of detecting anomalous graphs across diverse application scenarios, with only lightweight fine-tuning on few-shot normal samples from target dataset. Formally, we define $\mathcal { T } ^ { \mathrm { t r a i n } } = \mathbf { \bar { \Phi } } \{ \mathcal { D } _ { 1 } ^ { \mathrm { t r a i n } } , \dots , \mathcal { D } _ { N } ^ { \mathrm { t r a i n } } \}$ as a collection of training datasets, where each dataset $\mathcal { D } _ { i } ^ { \mathrm { t r a i n } } \dot { = }$ $\{ G _ { i , j } ^ { \mathrm { t r a i n } } \} _ { j = 1 } ^ { N _ { i } }$ consists of $N _ { i }$ normal graphs from a particular application scenario. The goal is to train a generalist GLAD model f on T<sup>train</sup> such that it can effectively detect anomalies in unseen test datasets $\mathcal { T } ^ { \mathrm { t e s t } }$ , where $\mathcal { D } _ { i } ^ { \mathrm { t e s t } } \in \mathcal { T } ^ { \mathrm { t e s t } }$ . During inference, we adopt the commonly used normalfew-shot setting [19, 34, 54, 58]. Specifically, for each test dataset $\mathcal { D } _ { i } ^ { \mathrm { t e s t } }$ <sup>t</sup>, only a small number $( n _ { k } )$ of normal graphs are available for lightweight adaptation. The model f is then required to assign anomaly scores to the remaining graphs in the dataset. In this paper, we assume that all graphs are unattributed to cover a wide range of real-world scenarios, because ❶ data from several domains do not come with node or edge features [43], and ❷ node features may play a less critical role in GLAD [48, 3].

## 4 Method

In this section, we introduce GenGLAD, a generalist GLAD approach that can identify anomalous graph samples on arbitrary testing datasets with few-shot adaptation. As shown in Figure 2, the pretrained model in GenGLAD learns to extract discriminative subgraphs for anomaly discrimination, with a memory bank to capture transferable knowledge and a cross-domain motif injection strategy to generate pseudo-anomalies (Sec. 4.1). During adaptation phase, GenGLAD leverages few-shot normal samples to compute multi-dimensional discrepancy for anomaly detection, with a lightweight scoring module for final prediction (Sec. 4.2).

## 4.1 Memory-Guided Discriminative Subgraph Extraction

To establish a generalist model, a fundamental challenge is to capture generalizable patterns across diverse graph domains for normal-anomaly discrimination. However, such patterns can vary significantly across domains due to differences in graph structures and distributions, making it difficult to identify a specific type of pattern as transferable knowledge. To address this challenge, instead of relying on a specific type of pattern, we propose a generalizable graph neural network (GNN) model that learns to extract the most discriminative substructures from arbitrary graph samples across diverse domains, serving as the transferable backbone of GenGLAD (Sec. 4.1.1). The discriminative subgraph can serve as key clues for distinguishing normal and abnormal samples, facilitating downstream discrimination. To capture domain-specific knowledge during subgraph extraction, we introduce a memory bank to store transferable patterns, supporting subgraph extraction with domain-relevant information (Sec. 4.1.2). To optimize the model using normal-only data, we propose a cross-domain motif injection strategy to generate pseudo-anomalous samples for training, together with a tailored objective function to guide the learning process (Sec. 4.1.3).

## 4.1.1 GNN-based Subgraph Extractor

While anomaly patterns across different domains can be diverse, the abnormality of anomalous graphs is usually characterized by a few key substructures [18, 50]. For example, in molecular graphs, abnormality $( \mathrm { e . g . }$ , chemical instability) can often be attributed to specific functional groups, while in social networks, anomalous behavior may be associated with unusual local irregular interaction motifs. Motivated by this, we propose a GNN backbone model that identifies the most discriminative subgraphs from various graph samples, learning the transferable patterns for GLAD across diverse domains. This capability is generalizable to unseen graphs for anomaly discrimination [25].

Structural Attribute Construction. As we focus on a more general unattributed setting, the first step is to construct a unified and semantically rich node attributes as the input features for the GNN. To this end, we incorporate structural features as node attributes. Specifically, for each graph G, node attributes are constructed by concatenating one-hot encodings of node degrees with random-walk diffusion-based structural encodings [10], i.e, $\mathbf { X } = \mathbf { X } ^ { \mathrm { d e g } } \oplus \mathbf { \breve { X } } ^ { \mathrm { R W } }$ . This design not only unifies the dimensionality and semantics of node attributes across datasets but also equips them with informative structural signals, thereby providing a consistent and effective foundation for generalist GLAD.

Subgraph Extraction Network. Given an input graph $\mathcal { G } = ( \nu , \mathcal { E } )$ with unified node attributes $\mathbf { X } ,$ we aim to extract a discriminative subgraph $\hat { \mathcal { G } } ^ { \prime } = ( \hat { \mathcal { V } } ^ { \prime } , \mathcal { E } ^ { \prime } )$ that preserves anomaly-discriminative information while filtering out irrelevant structures. To achieve this, we first obtain node representations H via a graph encoder, and then compute the node importance scores p through an importance estimation module $f _ { \theta } ( \cdot )$

$$
\mathbf { H } = \operatorname { G N N } ( \mathbf { X } , \mathbf { A } ) , \quad \mathbf { p } = f _ { \theta } ( \mathbf { H } ) .\tag{1}
$$

To extract the task-relevant subgraph while obtaining a differentiable node selection, we then apply Gumbel-Softmax with temperature τ to transform the importance scores into a binary mask:

$$
\begin{array} { r } { \mathbf { m } = \mathrm { G u m b e l S o f t m a x } ( \mathbf { p } ; \tau ) , \quad \mathbf { m } \in \{ 0 , 1 \} ^ { | \mathcal { V } | } . } \end{array}\tag{2}
$$

The resulting mask is used to filter nodes and edges, yielding the extracted subgraph $\mathcal { G } ^ { \prime }$ . This subgraph focuses the model on the most informative regions of the input graph while ignoring the irrelevant structures, serving as the discriminative evidence for downstream detection.

## 4.1.2 Importance Estimation via Memory Retrieva

In the subgraph extraction network, it is important to ensure that $f _ { \theta } ( \cdot )$ consistently identifies the most discriminative subgraphs. However, what constitutes such discriminative structures is inherently domain-dependent. Due to the diversity of application scenarios, topological patterns may vary significantly across domains, implying that the estimation module $f _ { \theta } { \bar { ( } } \cdot { \bar { ) } }$ must incorporate domainspecific knowledge. To improve the domain adaptability of $f _ { \theta } ( \cdot )$ , we equip it with a memory bank that encodes anomaly-related knowledge accumulated from diverse training domains. Given a new dataset, we can retrieve domain-relevant memory conditioned on its target context, thereby guiding the extraction process.

Specifically, we maintain a set of $M$ trainable memory tokens $\mathbf { T } = \{ \mathbf { t } _ { 1 } , \dots , \mathbf { t } _ { M } \}$ , where each token is responsible for identifying a discriminating graph pattern associated with anomaly detection across datasets. Then, given a target dataset D with available graph samples $s \subset D _ { i } .$ , we summarize a compact context vector that serves as the condition for retrieval: $\begin{array} { r } { \mathbf { c } = \frac { 1 } { \| S \| } \sum _ { \mathcal { G } \in \mathcal { S } } } \end{array}$ Readout<sub>G</sub>(X), with mean pooling operation for readout. Here, $s$ can be a batch of normal sample during training, or the few-shot supervision during the adaptation stage. The obtained context vector c is then used to adaptively retrieve relevant knowledge from the memory bank. In detail, we choose the tokens with top- $K$ relevance by computing cosine similarity between c and memory tokens, resulting in the retrieved token set $\tilde { \mathbf { T } }$ which encodes the domain-specific knowledge. Finally, the node importance is estimated by measuring the alignment between node embeddings and the retrieved memory tokens:

$$
\mathbf { p } _ { i } = \operatorname* { m a x } _ { \mathbf { t } \in \tilde { \mathbf { T } } } \cos ( \mathbf { h } _ { u } , \mathbf { t } ) ,\tag{3}
$$

where $\mathbf { h } _ { u }$ denotes the encoded embedding of node u in $\mathcal { G } _ { : }$ , and $\cos ( \cdot , \cdot )$ is the cosine similarity. By obtaining the maximum similarity score, we capture the most significant aligned pattern, ensuring the extracted subgraph is discriminative. In this way, the memory bank supplies transferable knowledge, while context-conditioned retrieval ensures domain-specific subgraph extraction.

## 4.1.3 Model Training

To train a generalist model for GLAD, a straightforward but effective learning objective is teaching it to distinguish between anomalous and normal samples. However, annotating abnormal graph samples demands heavy domain expertise, making the training data for GLAD usually under a one-class assumption, i.e., only normal samples are available [18]. To optimize the generalist model in such an unsupervised pre-training setting, we generate pseudo-anomalies via a cross-domain motifinjection strategy, and leverage them to guide model training with a bi-level loss function.

Cross-Domain Motif Injection. To generate pseudo-anomalies for training datasets from diverse domains, we propose a cross-domain motif injection strategy that leverages motifs from one domain to perturb graphs in another. The detailed procedure can be found at Appendix B. Concretely, given a batch of graphs sampled from a training dataset $\mathcal { D } _ { i } ^ { \mathrm { t r a i n } } \in \mathcal { T } ^ { \mathrm { t r a i n } }$ , we treat these graphs as normal samples for the corresponding graph application, allocated with pseudo label $\tilde { y } = 0 ,$ . Pseudoanomalies $( \tilde { y } = 1 )$ are then constructed by incorporating cross-dataset substructure, where each normal graph is augmented with a motif randomly sampled from graphs drawn from other datasets. As such subgraphs originate from different data distributions, they can disrupt the inherent structural patterns of the original graph, naturally introducing anomalous signals. To merge the graph and the motif, we link them with additional edges to form a single connected graph. To further provide fine-grained supervision signals, we assign a node-level pseudo-label vector $\tilde { \mathbf { y } } ^ { ( n ) } \in \{ 0 , 1 \} ^ { | \tilde { \nu } _ { b } | }$ for each graph, where each entry indicates whether a node is injected from a motif: $\tilde { \mathbf { y } } _ { u } ^ { ( n ) } = 1$ if node u belongs to any injected motif, and $\tilde { \mathbf { y } } _ { u } ^ { ( n ) } = 0$ otherwise.

Bi-Level Training Loss. To ensure the model can distinguish anomalies across diverse scenarios, we train the model by a pseudo-label-guided bi-level loss. We first formulate a graph-level objective that encourages the extracted subgraph $\mathcal { G } ^ { \prime }$ to be semantically consistent with domain-specific knowledge while deviating from anomalous ones. Given that the memory tokens encode prototypical discriminative patterns, we compute the similarity between the graph-level representation $\mathbf { z } _ { \mathcal { G } ^ { \prime } } = \operatorname { R e a d o u t } _ { \mathcal { G } ^ { \prime } } ( \mathbf { H } )$ and the aggregated in-domain memory token $\mathbf { t _ { c } } = \mathrm { M e a n } ( \mathbf { T } ^ { \prime } )$ , and optimize it with a binary cross-entropy (BCE) objective:

$$
\mathcal { L } _ { \mathrm { g r a p h } } = \mathrm { B C E } \big ( \sigma ( 1 - \cos ( \mathbf { z } _ { \mathcal { G } ^ { \prime } } , \mathbf { t _ { c } } ) \big ) , \ : \tilde { y } \big ) ,\tag{4}
$$

where $\cos ( \cdot , \cdot )$ denotes cosine similarity and $\sigma ( \cdot )$ is a sigmoid function. This objective encourages the extracted in-domain knowledge to be aligned with normal graphs while diverging from anomalies, thereby promoting cross-domain generalization.

To provide node-level supervisions, we leverage the injected motifs to form node-level objective. Specifically, we utilize the labeled injected nodes as hard negatives and penalize false positive predictions, encouraging the model to exclude them from the learned discriminative subgraph:

$$
\mathcal { L } _ { \mathrm { n o d e } } = - \mathbb { E } _ { i \sim \tilde { \mathbf { y } } _ { i } ^ { ( n ) } = 1 } [ \log ( 1 - m _ { i } ) ] .\tag{5}
$$

Finally, we integrate the graph- and node-level objectives into a unified pre-training loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p r e } } = \mathcal { L } _ { \mathrm { g r a p h } } + \lambda _ { \mathrm { n } } \mathcal { L } _ { \mathrm { n o d e } } + \lambda _ { \mathrm { s } } \| \mathbf { m } \| _ { 1 } , } \end{array}\tag{6}
$$

where the sparsity regularization term $\| \mathbf { m } \| _ { 1 }$ encourages the model to focus on compact and informative substructures, while $\lambda _ { \mathrm { n } }$ and $\lambda _ { \mathrm { s } }$ are balance hyperparameters. Overall, the training objective utilizes the informative pseudo-labels to guide the learning of GenGLAD with both coarse-grained and fine-grained losses, grounding robust generalization across diverse graph application domains.

## 4.2 Multi-Dimensional Discrepancy Scoring for Few-Shot Adaptation

Although the pre-trained backbone model has the capability to capture informative subgraphs that are useful for anomaly discrimination, simply applying it to unseen datasets may not always generalize well due to domain shift and variations in abnormality patterns. To achieve effective target-dataset adaptation, a key idea is to fully exploit the knowledge encoded in the pre-trained model and the memory bank, together with the few-shot samples that reflects the characteristics of the target domain. To this end, we construct multi-dimensional discrepancy signals that characterize the deviation of the discriminative subgraph of an incoming sample from multiple perspectives, including prior knowledge, domain-level context, and feature semantics. To adaptively leverage these discrepancy signals for anomaly detection, we further learn a lightweight scoring module with the few-shot normal samples, ensuring reliable predictions are made based on the discrepancy-based evidence.

Multi-Dimensional Discrepancy Construction. While the extracted subgraph captures key characteristics of each graph sample, it is not straightforward to characterize its position within the normal distribution of the target dataset for anomaly detection. To achieve this, we model anomaly evidence as discrepancy scores across multiple semantic spaces. We construct a set of complementary discrepancy signals for each incoming graph, providing a holistic characterization of it. Specifically, we construct multi-dimensional discrepancy measures from four perspectives.

1. Prior Discrepancy, which measures the deviation of the discriminative subgraph embedding $\mathbf { z } _ { \mathcal { G } ^ { \prime } }$ from the pre-trained semantic prior, characterized by its similarity to the pre-trained prototype $\mathbf { t _ { c } } .$ This perspective captures how well the current graph aligns with the general knowledge learned during pre-training, serving as a basic indicator for anomaly detection.

2. Domain Discrepancy, which measures the deviation of $\mathbf { z } _ { \mathcal { G } ^ { \prime } }$ from the domain-level context, characterized by its distance to the global domain prototype c. This perspective reflects the extent to which the current graph deviates from the overall data distribution, providing a coarse-grained signal of distributional shift and potential anomaly.

3. Class-level Discrepancy, which evaluates the mismatch between $\mathbf { z } _ { \mathcal { G } ^ { \prime } }$ and the class prototype q calculated as the average representations of few-shot normal samples. This perspective characterizes how well the current graph adheres to the semantic pattern of the normal class, offering a more targeted cue for distinguishing abnormal instances.

4. Instance-level Discrepancy, which captures the inconsistency between the similarity pattern of the query graph to support instances and the intrinsic relational structure within the support set. This perspective reveals whether the current graph maintains the fine-grained relational regularities of normal data, yielding a distribution-aware signal beyond point-wise comparisons.

Notably, the choice of similarity/distance metrics (e.g., cosine and L2) can influence the characterization of discrepancies; therefore, we adopt multiple metrics for certain discrepancies to improve coverage. A formal formulation of the discrepancy scores is provided in Appendix C.

Lightweight Scoring Module. In practice, different datasets may exhibit varying dependence on different discrepancy scores for anomaly detection, due to differences in data distributions. To account for this, we learn a lightweight scoring module, which adopts a linear combination of discrepancy scores: $s = \sigma ( \mathbf { w } ^ { \top } \mathbf { s } )$ , where s is the anomaly score of a query graph, s is the vector composed of multiple discrepancy scores, and σ is a Sigmoid function. $\mathbf { w } \in \mathbb { R } ^ { k \times 1 }$ is the only parameter that is learnable during the adaptation phase, while the rest of the model remains frozen.

To optimize w, a straightforward approach is to use a binary cross-entropy loss with binary normal/anomalous labeled samples. Nevertheless, the few-shot set only contains normal instances, leading to an urgent need for a mechanism to distinguish pseudo-anomalies from normal samples. As large-scale training datasets are not available at this stage, we employ two simple yet effective strategies to distinguish pseudo-anomalies from few-shot normal samples (details of pseudo-anomaly construction is given in Appendix D): ❶ Structure perturbation, which constructs outlier graphs by randomly shuffling graph structures to destroy meaningful relational patterns, and ❷ Sample bridging, which generates outlier graphs by merging normal graphs to simulate over-amplified in-domain structural patterns. With the original samples as normal instances and pseudo-anomalies as negative instances, we can optimize w with only a few gradient steps during adaptation phase. Algorithm description and complexity analysis of GenGLAD are in Appendices E and F.

## 5 Experiments

## 5.1 Experimental Setup

Datasets. We train both the baseline approaches and GenGLAD on one set of graph datasets and evaluate them on a separate set of datasets. For a comprehensive evaluation, we utilize real-world GLAD datasets drawn from diverse domains, including molecular graphs, biochemical graphs, and social networks. Following [19], we train the models on the largest dataset within each domain and evaluate them on the remaining datasets. Specifically, the training datasets include NCI1, DD, and COLLAB, while the testing datasets consist of molecular graphs: AIDS, BZR, COX2, DHFR, MUTAG, PTC\_MR; biochemical graphs: ENZYMES, PROTEINS; and social networks: IMDB-BINARY, and REDDIT-BINARY. We also denote the collection of molecular and chemical graphs as science graphs. For more detailed information, please refer to the Appendix G.

Table 1: Anomaly detection performance in terms of AUROC (in percent, mean±std). Highlighted are the results ranked first, second, and third. OOM indicates out-of-memory.
<table><tr><td>Method</td><td>AIDS</td><td>BZR</td><td>COX2</td><td>DHFR</td><td>MUTAG</td><td>PTC_MR</td><td>ENZYMES</td><td>PROTEINS</td><td>IMDB-B</td><td>REDDIT-B</td></tr><tr><td colspan="9">Pre-training+Few-shot Fine-tuning</td><td></td></tr><tr><td> $_ \mathrm { G L o c a l K D }$ </td><td> $5 0 . 6 7 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $5 0 . 2 9 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $5 2 . 7 3 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $5 6 . 9 0 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $7 9 . 5 5 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $4 9 . 3 4 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $5 0 . 1 2 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $4 9 . 9 0 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $6 7 . 8 2 \pm 0 . 2 3 $ </td><td> $6 7 . 8 2 { \scriptstyle \pm 0 . 2 3 }$ </td></tr><tr><td> $\mathrm { C V T G A D }$ </td><td> $4 0 . 6 5 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $4 5 . 9 6 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $5 2 . 0 0 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $5 5 . 8 1 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $8 9 . 5 5 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $5 1 . 8 3 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $4 5 . 3 1 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $4 0 . 3 6 { \scriptstyle \pm 0 . 7 1 }$ </td><td>OOM</td><td>OOM</td></tr><tr><td>SIGNET</td><td> $9 0 . 5 9 { \pm } 0 . 0 0 \ $ </td><td>21.46±0.02</td><td> $5 3 . 5 6 { \pm } 0 . 0 1 $ </td><td> $5 3 . 5 6 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $8 2 . 4 7 { \scriptstyle \pm 0 . 0 2 }$ </td><td>53.19±0.02</td><td> $4 7 . 2 8 { \scriptstyle \pm 0 . 0 0 }$ </td><td>57.19±0.01</td><td> $5 0 . 6 6 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $5 0 . 6 6 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>MUSE</td><td> $7 7 . 3 0 { \scriptstyle \pm 1 . 7 3 }$ </td><td> $5 2 . 7 \dot { 7 } \pm 0 . 4 2$ </td><td>45.27±0.27</td><td> $5 7 . 2 4 { \pm } 0 . 8 8 $ </td><td> $8 4 . 0 5 { \scriptstyle \pm 1 . 6 7 }$ </td><td> $4 7 . 3 2 { \scriptstyle \pm 0 . 3 3 }$ </td><td>52.48±0.47</td><td> $5 2 . 5 7 { \scriptstyle \pm 1 . 2 5 }$ </td><td> $6 7 . 8 4 \pm 1 . 0 6 $ </td><td>67.84±1.06</td></tr><tr><td>GLADPro</td><td> $5 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $4 7 . 3 9 { \scriptstyle \pm 1 . 7 2 }$ </td><td> $4 8 . 1 0 { \scriptstyle \pm 0 . 6 9 }$ </td><td> $5 3 . 0 4 { \scriptstyle \pm 1 . 8 4 }$ </td><td> $5 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $5 3 . 3 3 { \pm } 3 . 6 1 $ </td><td> $5 3 . 1 3 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $3 8 . 5 1 { \pm } 1 . 3 6 $ </td><td>64.16±7.36</td><td> $6 4 . 1 6 { \scriptstyle \pm 7 . 3 6 }$ </td></tr><tr><td>HC-GLAD</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>FANFOLD</td><td> $7 3 . 5 4 { \pm } 0 . 0 0$ </td><td> $5 0 . 3 6 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $3 7 . 9 4 { \pm } 0 . 0 2$ </td><td> $5 8 . 1 8 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $8 7 . 8 9 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $4 1 . 7 0 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $4 9 . 5 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $4 9 . 4 1 { \pm } 0 . 0 0$ </td><td> $5 6 . 6 4 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 8 . 5 1 { \pm } 0 . 0 0 $ </td></tr><tr><td colspan="10">Few-shot Fine-tuning Only</td></tr><tr><td> $_ \mathrm { G L o c a l K D }$ </td><td> $5 7 . 7 1 { \scriptstyle \pm 2 . 9 1 }$ </td><td> $5 8 . 9 1 { \scriptstyle \pm 4 . 5 4 }$ </td><td> $5 2 . 1 4 { \scriptstyle \pm 2 . 3 9 }$ </td><td> $6 3 . 3 4 \pm 1 . 5 4 $ </td><td> $8 9 . 5 4 \pm 0 . 2 3 $ </td><td> $4 8 . 8 3 { \scriptstyle \pm 0 . 9 9 }$ </td><td> $4 7 . 6 0 { \scriptstyle \pm 2 . 0 8 }$ </td><td> $5 3 . 0 5 { \scriptstyle \pm 2 . 8 9 }$ </td><td> $5 9 . 7 9 { \scriptstyle \pm 1 . 6 3 }$ </td><td> $7 2 . 3 7 { \pm } 1 . 5 6 $ </td></tr><tr><td> $\mathrm { C V T G A D }$ </td><td> $6 9 . 4 9 { \scriptstyle \pm 7 . 7 8 }$ </td><td> $4 3 . 3 0 { \scriptstyle \pm 3 . 8 0 }$ </td><td> $4 3 . 3 6 { \scriptstyle \pm 3 . 9 6 }$ </td><td> $4 6 . 2 4 \pm 1 . 4 7$ </td><td> $7 9 . 4 1 \pm 8 . 9 3 $ </td><td> $4 3 . 2 9 2 2 . 7 9$ </td><td> $5 4 . 5 0 { \scriptstyle \pm 2 . 9 0 }$ </td><td> $6 9 . 8 1 { \pm } 0 . 8 6 $ </td><td> $5 5 . 5 1 { \pm } 4 . 7 9$ </td><td>OOM</td></tr><tr><td>SIGNET</td><td> $6 8 . 2 1 { \scriptstyle \pm 5 . 4 9 }$ </td><td> $4 4 . 6 0 { \scriptstyle \pm 3 . 5 2 }$ </td><td> $4 4 . 3 8 { \scriptstyle \pm 3 . 0 2 }$ </td><td> $3 5 . 2 1 { \pm } 3 . 1 2$ </td><td> $6 8 . 8 6 \pm 1 4 . 4 5$ </td><td> $5 0 . 9 3 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $\overline { { 4 1 . 2 8 { \scriptstyle \pm 2 . 0 2 } } }$ </td><td> $5 5 . 4 8 { \scriptstyle \pm 1 . 7 8 }$ </td><td> $4 5 . 9 7 { \scriptstyle \pm 2 . 3 7 }$ </td><td> $5 2 . 3 6 { \pm } 1 . 9 5 $ </td></tr><tr><td>MUSE</td><td> $8 4 . 5 6 \pm 5 . 8 2 $ </td><td> $4 7 . 3 2 { \scriptstyle \pm 0 . 8 7 }$ </td><td> $4 1 . 8 0 { \scriptstyle \pm 1 . 3 5 }$ </td><td> $6 1 . 7 4 { \pm } 1 . 9 1 $ </td><td> $8 5 . 7 7 { \scriptstyle \pm 6 . 0 1 }$ </td><td> $5 0 . 7 6 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $4 9 . 5 0 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $6 1 . 8 1 { \pm } 0 . 1 7$ </td><td> $5 8 . 1 8 { \scriptstyle \pm 2 . 3 8 } $ </td><td> $7 3 . 8 5 { \pm } 1 . 8 2 $ </td></tr><tr><td>GLADPro</td><td> $\overline { { 7 0 . 2 1 \pm 1 3 . 9 2 } }$ </td><td> $5 6 . 2 4 \pm 3 . 3 7$ </td><td> $4 3 . 6 2 \pm 3 . 8 0 $ </td><td> $\overline { { 4 3 . 0 3 2 3 . 7 2 } }$ </td><td> $8 2 . 9 6 { \pm } 8 . 1 0 $ </td><td> $5 5 . 7 3 { \scriptstyle \pm 3 . 0 5 }$ </td><td> $5 3 . 9 8 { \scriptstyle \pm 1 . 4 5 }$ </td><td> $\overline { { 4 3 . 4 4 \pm 4 . 0 9 } }$ </td><td> $5 4 . 0 1 { \scriptstyle \pm 2 . 8 9 }$ </td><td> $6 4 . 2 9 2 6 . 0 7$ </td></tr><tr><td> $_ \mathrm { H C - G L A D }$ </td><td> $3 . 6 4 \pm 0 . 5 5$ </td><td> $3 7 . 5 0 { \scriptstyle \pm 3 . 3 2 }$ </td><td> $4 6 . 1 6 { \scriptstyle \pm 1 . 5 6 }$ </td><td> $5 4 . 9 9 2 . 6 6 $ </td><td> $8 4 . 6 9 { \scriptstyle \pm 2 . 7 9 }$ </td><td> $4 8 . 9 4 \pm 0 . 7 0$ </td><td> $6 3 . 0 3 { \pm } 4 . 4 5 $ </td><td> $3 7 . 8 5 { \scriptstyle \pm 3 . 6 7 }$ </td><td> $6 1 . 8 5 { \scriptstyle \pm 5 . 1 1 }$ </td><td>OOM</td></tr><tr><td>FANFOLD</td><td> $6 0 . 9 5 { \scriptstyle \pm 1 8 . 8 9 }$ </td><td> $4 4 . 9 2 { \scriptstyle \pm 1 . 0 4 }$ </td><td> $4 1 . 8 0 { \scriptstyle \pm 3 . 8 4 }$ </td><td> $6 0 . 2 8 { \scriptstyle \pm 2 . 3 5 }$ </td><td> $7 4 . 0 0 { \scriptstyle \pm 2 0 . 0 9 }$ </td><td> $4 7 . 0 4 2 3 . 6 0$ </td><td> $5 2 . 1 5 { \scriptstyle \pm 3 . 9 5 }$ </td><td> $5 8 . 2 4 { \scriptstyle \pm 3 . 6 1 }$ </td><td> $5 1 . 5 4 { \scriptstyle \pm 3 . 9 7 }$ </td><td> $2 3 . 2 4 \pm 2 . 9 2$ </td></tr><tr><td colspan="10"> $_ \mathrm { G e n e r a l i s t }$ </td></tr><tr><td> $_ \mathrm { G e n G L A D }$ </td><td> $9 5 . 8 4 \pm 0 . 0 7$ </td><td> $6 3 . 8 9 { \pm } 1 . 1 8 $ </td><td> $6 2 . 9 2 2 0 . 3 0 $ </td><td> $7 4 . 9 6 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $9 0 . 7 5 { \pm } 0 . 1 7 $ </td><td> $6 0 . 5 8 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $5 4 . 3 7 { \pm } 1 . 7 4 $ </td><td> $7 3 . 6 3 { \pm } 0 . 1 6 $ </td><td> $7 4 . 1 1 { \pm } 0 . 2 5 $ </td><td> $7 3 . 6 3 { \pm } 0 . 2 0 $ </td></tr></table>

<table><tr><td>Training Datasets</td><td>AIDS</td><td>BZR</td><td>COX2</td><td>DHFR</td><td>MUTAG</td><td>PTC_MR</td><td>ENZYMES</td><td>PROTEINS</td><td>IMDB-B</td><td>REDDIT-B</td></tr><tr><td>NCI1</td><td> $8 7 . 5 6 { \pm } 2 . 6 8 $ </td><td> $5 6 . 6 4 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $5 7 . 7 2 { \scriptstyle \pm 1 . 2 7 }$ </td><td> $7 0 . 9 2 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $7 5 . 1 9 { \scriptstyle \pm 3 . 4 5 }$ </td><td> $5 5 . 8 3 { \scriptstyle \pm 1 . 4 6 }$ </td><td> $5 2 . 7 0 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $6 3 . 2 6 { \pm } 1 . 4 5 $ </td><td> $5 6 . 1 4 { \pm } 1 . 2 0 $ </td><td> $6 4 . 0 6 { \scriptstyle \pm 0 . 1 6 }$ </td></tr><tr><td>DD</td><td> $8 8 . 3 3 { \scriptstyle \pm 4 . 2 6 }$ </td><td> $5 6 . 9 8 { \scriptstyle \pm 1 . 2 5 }$ </td><td> $5 7 . 8 9 { \scriptstyle \pm 1 . 3 9 }$ </td><td> $7 0 . 8 7 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $7 3 . 7 4 \pm 2 . 9 3 $ </td><td> $5 5 . 8 3 { \scriptstyle \pm 1 . 4 6 }$ </td><td> $5 2 . 8 2 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $6 3 . 2 0 { \scriptstyle \pm 1 . 4 0 }$ </td><td> $5 5 . 8 0 { \scriptstyle \pm 1 . 0 9 } $ </td><td> $6 3 . 4 6 { \pm } 1 . 0 4 $ </td></tr><tr><td> $_ \mathrm { C O L L A B }$ </td><td> $8 7 . 8 8 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $5 7 . 4 8 { \scriptstyle \pm 1 . 6 7 }$ </td><td> $5 7 . 7 4 { \scriptstyle \pm 1 . 7 6 }$ </td><td> $7 1 . 0 9 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $7 6 . 2 5 { \pm } 1 . 6 9$ </td><td> $5 5 . 6 4 \pm 1 . 4 3 $ </td><td> $5 2 . 5 2 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $6 3 . 3 0 { \scriptstyle \pm 1 . 1 5 }$ </td><td> $5 5 . 1 5 { \pm } 1 . 3 8 $ </td><td> $6 5 . 4 4 2 0 . 4 8$ </td></tr><tr><td>NCI1+DD</td><td> $8 9 . 7 2 { \scriptstyle \pm 2 . 7 1 }$ </td><td>58.18±1.52</td><td> $5 8 . 2 3 { \pm } 1 . 5 3 $ </td><td>71.10±0.50</td><td> $8 0 . 0 8 { \scriptstyle \pm 3 . 9 9 }$ </td><td>57.70±1.42</td><td>53.19±0.54</td><td> $6 5 . 3 7 { \scriptstyle \pm 1 . 0 9 }$ </td><td>66.77±1.22</td><td>70.96±1.09</td></tr><tr><td> $\mathbf { N C I 1 + C O L L A B }$ </td><td> $9 1 . 6 2 \pm 3 . 3 8 $ </td><td> $5 9 . 9 2 { \pm } 1 . 1 0 $ </td><td> $5 8 . 2 2 { \scriptstyle \pm 1 . 4 6 }$ </td><td> $7 2 . 5 1 { \pm } 0 . 8 9 $ </td><td> $8 2 . 5 5 { \pm } 2 . 6 4$ </td><td> $\overline { { 5 7 . 6 3 \pm 1 . 4 0 } }$ </td><td> $5 3 . 4 9 { \pm } 0 . 1 4 $ </td><td> $6 7 . 0 4 \pm 0 . 8 6$ </td><td> $6 6 . 6 2 \pm 1 . 7 6$ </td><td> $\overline { { 6 9 . 2 2 \pm 1 . 2 8 } }$ </td></tr><tr><td> $\mathrm { D D + C O L L A B }$ </td><td> $\overline { { 9 1 . 9 9 \pm 2 . 9 4 } }$ </td><td> $5 9 . 9 9 { \scriptstyle \pm 1 . 0 6 }$ </td><td> $5 8 . 5 5 { \pm } 1 . 3 2 $ </td><td> $7 2 . 8 5 { \pm } 0 . 8 0 $ </td><td> $8 3 . 0 9 \pm 2 . 6 2 $ </td><td> $5 7 . 3 2 { \pm } 1 . 2 1 $ </td><td> $5 3 . 7 6 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $6 7 . 3 3 { \pm } 1 . 1 3 $ </td><td> $6 8 . 5 9 { \scriptstyle \pm 1 . 4 4 }$ </td><td> $6 5 . 3 4 \pm 0 . 4 0$ </td></tr><tr><td>All</td><td> $9 5 . 8 4 \pm 0 . 0 7$ </td><td> $\underline { { 6 3 . 8 9 \pm 1 . 1 8 } }$ </td><td> $6 2 . 9 2 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $7 4 . 9 6 { \pm } 0 . 4 2 $ </td><td> $9 0 . 7 5 { \scriptstyle \pm 0 . 1 7 } $ </td><td> $6 0 . 5 8 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $5 4 . 3 7 { \pm } 1 . 7 4 $ </td><td> $7 3 . 6 3 { \pm } 0 . 1 6 $ </td><td> $7 4 . 1 1 { \pm } 0 . 2 5 $ </td><td> $7 3 . 6 3 { \pm } 0 . 2 0 $ </td></tr></table>

Table 2: Performance (in AUROC) with varying pre-training datasets ${ \mathcal { T } } ^ { \mathrm { t r a i n } } .$

Baselines. We compare GenGLAD with several unsupervised GLAD baselines, including contrastive methods such as GLocalKD [23], GLADPro [50], CVTGAD [15], SIGNET [18], and HC-GLAD [11]; reconstruction-based methods such as FANFOLD [5] and MUSE [14].

Evaluation and Implementation. Following [43], we adopt AUROC and AUPRC as evaluation metrics for GLAD. We report the mean AUROC/AUPRC along with standard deviations over five independent runs. We train GenGLAD jointly on all datasets in $\bar { \mathcal { T } } ^ { \mathrm { t r a i n } }$ and evaluate it on each dataset in $\mathcal { T } ^ { \mathrm { t e s t } }$ with few-shot tuning, where $n = 1 0$ normal graphs are provided as supervision by default. For all baselines, we consider two settings: "Pre-training+Few-shot Fine-tuning" and "Few-shot Finetuning $\mathrm { O n l y " }$ . In the latter, baselines are trained directly using only the few-shot normal supervision to avoid potential performance degradation caused by negative transfer. We utilize random search to determine the optimal hyperparameters for both the baselines and GenGLAD (Details in Appendix G). Since our goal is to train generalist GLAD models, we do not perform dataset-specific hyperparameter tuning, but instead use the same set of hyperparameters for all testing datasets.

## 5.2 Experimental Results

## Performance Comparison.

Table 1 presents a comparison between GenGLAD and baseline methods in terms of AUROC. Results for AUPRC are provided in Appendix H.1. We make the following observations: ❶ GenGLAD demonstrates strong anomaly detection capability in the generalist GLAD setting, despite using only few-shot finetuning on normal samples. Specifically, GenGLAD achieves state-of-the-art performance on 9 out of 10 datasets, while still obtaining competitive performance on the ENZYMES dataset. Moreover, GenGLAD demonstrates significant improvements compared to the best baselines on AIDS, REDDIT, COX2, and DHFR. ❷ Pre-training typically does not lead to performance improvements for GLAD baselines, highlighting the limited generalizability of conventional GLAD methods. ❸ With a limited amount of training data, baseline methods perform poorly. Notably, in some cases, they achieve an AUROC below 0.5 (worse than random guessing). This highlights that conventional GLAD methods are data-hungry, making them difficult to apply in emerging GLAD scenarios where training data is scarce.

![](images/71b00578414a404510241052cb99850045c54f1fee7f4b46979fbce51f9718d2.jpg)  
Figure 3: Sensitivity of #shot.

Effectiveness of Data Scale. To investigate the impact of data scale on the performance of GenGLAD, we vary $\mathcal { T } ^ { \mathrm { t r a i n } }$ . From Table 2, we observe that the performance of GenGLAD consistently improves as more pre-training datasets are incorporated, demonstrating its ability to learn transferable patterns across diverse domains. With the help of memory retrieval, the model can further retrieve domain-relevant knowledge, mitigating the negative transfer issue. Moreover, we report the average performance on social networks (Figure 3, more results are in Appendix H.2) with varying number of few-shot samples for adaptation. We observe that increasing the number of fine-tuning graphs consistently improves performance, which highlights the effectiveness of our method in leveraging additional adaptation data when available.

Ablation Study. To verify the effectiveness of each component of GenGLAD, we conduct ablation studies by removing/modifying each design, with the results on selected datasets in Table 3 (full results are in Appendix H.3). Specifically, for the pre-training phase, we construct variants by removing Lnode and the $\ell _ { 1 }$ regularization on m. We find that both components contribute to overall performance. In particular, Lnode has a larger impact, highlighting the importance of informative node-level pseudo anomaly supervision compared to unsupervised regularization. Moreover, we investigate the effectiveness of each discrepancy measure in the fine-tuning stage and the scoring module. Generally, removing any of the discrepancy measures degrades the performance in most cases. Among them, the domain and class discrepancy scores have the most significant impact, as their removal can lead to performance close to or even below random guessing. To assess the necessity of the scoring module, we replace it with simple score aggregation strategies, including mean, max, and min pooling. As illustrated, these replacements significantly degrade performance and can even lead to AUROC scores below 0.5. This suggests that the model fails to distinguish between normal and anomaly-indicative cues even with normal graph supervisions, indicating the importance of our adaptation framework.

Table 3: Ablation study.
<table><tr><td rowspan=1 colspan=1>Variant</td><td rowspan=1 colspan=3>DHFR PROTEINS IMDB-B</td></tr><tr><td rowspan=1 colspan=1>GenGLAD</td><td rowspan=1 colspan=3>74.96    73.63     74.11</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Pre-training</td></tr><tr><td rowspan=1 colspan=1>w/o ${ \mathcal { L } } ^ { \mathrm { n o d e } }$ w/o ||m||1</td><td rowspan=1 colspan=3>70.98    66.03     58.0371.30    68.53     60.06</td></tr><tr><td rowspan=1 colspan=1>Di</td><td rowspan=1 colspan=2>screpanc</td><td rowspan=1 colspan=1>y Measures</td></tr><tr><td rowspan=4 colspan=1>w/o Priorw/o Domainw/o Cls.w/o Inst.</td><td rowspan=1 colspan=2>73.05</td><td rowspan=1 colspan=1>71.66     71.15</td></tr><tr><td rowspan=1 colspan=3>73.77    63.76     51.65</td></tr><tr><td rowspan=1 colspan=3>45.63    60.20     63.72</td></tr><tr><td rowspan=1 colspan=3>74.25    73.75     72.96</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Scoring Module</td></tr><tr><td rowspan=3 colspan=1>MEANMAXMIN</td><td rowspan=1 colspan=3>28.92    41.54     42.00</td></tr><tr><td rowspan=1 colspan=3>40.32    61.94     30.12</td></tr><tr><td rowspan=1 colspan=3>29.56    38.05     43.57</td></tr></table>

Effectiveness of #Retrieved Tokens. To verify the effectiveness of retrieved memory tokens, we study the effect of retrieved memory token number K on performance (AUROC) in Figure 4a. Overall, both groups of datasets exhibit a consistent trend: after a certain threshold, increasing K leads to performance degradation, as less relevant tokens may introduce noisy or unhelpful information for the target task. Moreover, scientific graphs and social networks exhibit different sensitivities to K, where scientific graphs prefer larger K for diverse motifs, while social networks favor smaller K.

Efficiency Analysis. To investigate the efficiency of GenGLAD, we compare fine-tuning time on the AIDS dataset. As shown in Figure 4b, GenGLAD outperforms all baselines in terms of efficiency. Compared with our method, the fastest baseline, GLADPro, is still 2.1× slower. A comprehensive runtime breakdown is further discussed in Appendix H.5.

Visualization. To investigate how the pre-trained GenGLAD adapts to target applications, we visualize the weights of the scoring module. As shown in Figure 4c, GenGLAD learns distinct weighting patterns across different test domains. This indicates that the scoring module successfully captures domainspecific dependencies, enabling effective yet efficient few-shot adaptation. We also visualized the importance scores of individual memory tokens in Appendix H.4.

![](images/b16e08ecf7d379215191c2c183a678b3472a4abac30ee4a3538889b53956b35f.jpg)  
(a) Sensitivity to # mem. tokens.

![](images/f12578f21debc8974d700c41e84dd3b8ea83a0e760d35f67e192369e47876873.jpg)  
(b) Time comparison.

![](images/ac6e6a08c3253a4a1036369f52bea634a06837966b38566dc8c4e4c23cece915.jpg)  
(c) Visualization results.  
Figure 4: Analysis results.

## 6 Conclusion

In this paper, we take a pioneering step toward addressing the generalist GLAD problem, which aims to detect anomalous graphs across diverse emerging applications under a one-for-all paradigm with only a few-shot normal graph supervision for adaptation. To this end, we propose GenGLAD, a dis criminative subgraph extraction-based generalist GLAD approach capable of identifying anomalous graphs with limited in-domain supervision. Extensive experiments on real-world GLAD datasets from various domains demonstrate the effectiveness, generalizability, and efficiency of GenGLAD. The main limitation of GenGLAD is that it still requires few-shot supervision for adaptation and therefore cannot yet achieve fully zero-shot generalist GLAD. Future work may explore incorporating expert knowledge or external knowledge bases to enable zero-shot GLAD across diverse domains.

## Acknowledgments and Disclosure of Funding

The work of S. Pan was partially supported by the Australian Research Council (ARC) under Grant Nos. DP240101547 and FT210100097. The work of Y. Liu was partially supported by the ARC under Grant No. DE260101172. F. Li was supported by the Australian National Health and Medical Research Council Investigator grants [2041439].

## References

[1] Lemar Abdi, Amaan Valiuddin, Francisco Caetano, Christiaan Viviers, and Fons Van Der Som men. Zero-shot image anomaly detection using generative foundation models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3554–3563, 2025.

[2] Charu C Aggarwal and Haixun Wang. Graph data management and mining: A survey of algorithms and applications. In Managing and mining graph data, pages 13–68. Springer, 2010.

[3] Jinyu Cai, Yunhe Zhang, Zhoumin Lu, Wenzhong Guo, and See-Kiong Ng. Towards effective federated graph anomaly detection via self-boosted knowledge distillation. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 5537–5546, 2024.

[4] Jinyu Cai, Yunhe Zhang, Pengyang Wang, and See-Kiong Ng. Moegad: A mixture-of-experts framework with pseudo-anomaly generation for graph-level anomaly detection. IEEE Transac tions on Pattern Analysis and Machine Intelligence, 2025.

[5] Rui Cao, Shijie Xue, Jindong Li, Qi Wang, and Yi Chang. Fanfold: Graph normalizing flowsdriven asymmetric network for unsupervised graph-level anomaly detection. arXiv preprint arXiv:2407.00383, 2024.

[6] Baiyu Chen, Wilson Wongso, Xiaoqian Hu, Yue Tan, and Flora D Salim. Multi-stage verificationcentric framework for mitigating hallucination in multi-modal rag. In 2025 KDD Cup Workshop for Multimodal Retrieval Augmented Generation, 2026.

[7] Xuhai Chen, Yue Han, and Jiangning Zhang. A zero-/few-shot anomaly classification and segmentation method for cvpr 2023 (vand) workshop challenge tracks 1 &2. 1st Place on Zero-shot AD and 4th Place on Few-shot AD, 2305:17382, 2023.

[8] Asim Kumar Debnath, Rosa L Lopez de Compadre, Gargi Debnath, Alan J Shusterman, and Corwin Hansch. Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds. correlation with molecular orbital energies and hydrophobicity. Journal of medicinal chemistry, 34(2):786–797, 1991.

[9] Paul D Dobson and Andrew J Doig. Distinguishing enzyme structures from non-enzymes without alignments. Journal ofmolecular biology, 330(4):771–783, 2003.

[10] Vijay Prakash Dwivedi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Graph neural networks with learnable structural and positional representations. In International Conference on Learning Representations, 2022.

[11] Yali Fu, Jindong Li, Jiahong Liu, Qianli Xing, Qi Wang, and Irwin King. Hc-glad: dual hyperbolic contrastive learning for unsupervised graph-level anomaly detection. Neural Networks, page 109009, 2026.

[12] Gastón García González, Pedro Casas, Emilio Martínez, and Alicia Fernández. Towards foundation auto-encoders for time-series anomaly detection. arXiv preprint arXiv:2507.01875, 2025.

[13] Christoph Helma, Ross D. King, Stefan Kramer, and Ashwin Srinivasan. The predictive toxicology challenge 2000–2001. Bioinformatics, 17(1):107–108, 2001.

[14] Sunwoo Kim, Soo Yong Lee, Fanchen Bu, Shinhwan Kang, Kyungho Kim, Jaemin Yoo, and Kijung Shin. Rethinking reconstruction-based graph-level anomaly detection: limitations and a simple remedy. Advances in Neural Information Processing Systems, 37:95931–95962, 2024.

[15] Jindong Li, Qianli Xing, Qi Wang, and Yi Chang. Cvtgad: Simplified transformer with crossview attention for unsupervised graph-level anomaly detection. In Joint European conference on machine learning and knowledge discovery in databases, pages 185–200. Springer, 2023.

[16] Shiyuan Li, Yixin Liu, Yu Zheng, Xiaofeng Cao, Shirui Pan, and Heng Tao Shen. Towards one-for-all anomaly detection for tabular data. arXiv preprint arXiv:2603.14407, 2026.

[17] Shiyuan Li, Yunfeng Zhao, Yue Tan, Qingfeng Chen, Yixin Liu, and Shirui Pan. Towards anomaly detection on relational data. arXiv preprint arXiv:2606.18621, 2026.

[18] Yixin Liu, Kaize Ding, Qinghua Lu, Fuyi Li, Leo Yu Zhang, and Shirui Pan. Towards selfinterpretable graph-level anomaly detection. Advances in Neural Information Processing Systems, 36:8975–8987, 2023.

[19] Yixin Liu, Shiyuan Li, Yu Zheng, Qingfeng Chen, Chengqi Zhang, and Shirui Pan. Arc: A generalist graph anomaly detector with in-context learning. Advances in Neural Information Processing Systems, 37:50772–50804, 2024.

[20] Yujing Liu, Yixin Liu, Yu Zheng, Alan Wee-Chung Liew, Xiaofeng Cao, and Shirui Pan. Rethinking feature alignment in generalist graph anomaly detection: A relational fingerprintbased approach. In International Conference on Machine Learning, 2026.

[21] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, et al. Swin transformer v2: Scaling up capacity and resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12009–12019, 2022.

[22] Xuexiong Luo, Jia Wu, Jian Yang, Shan Xue, Hao Peng, Chuan Zhou, Hongyang Chen, Zhao Li, and Quan Z Sheng. Deep graph level anomaly detection with contrastive learning. Scientific Reports, 12(1):19867, 2022.

[23] Rongrong Ma, Guansong Pang, Ling Chen, and Anton Van Den Hengel. Deep graph-level anomaly detection by glocal knowledge distillation. In Proceedings of the fifteenth ACM international conference on web search and data mining, pages 704–714, 2022.

[24] Xiaoxiao Ma, Jia Wu, Shan Xue, Jian Yang, Chuan Zhou, Quan Z Sheng, Hui Xiong, and Leman Akoglu. A comprehensive survey on graph anomaly detection with deep learning. IEEE transactions on knowledge and data engineering, 35(12):12012–12038, 2021.

[25] Xiaoxiao Ma, Jia Wu, Jian Yang, and Quan Z Sheng. Towards graph-level anomaly detection via deep evolutionary mapping. In Proceedings ofthe 29th ACM SIGKDD conference on knowledge discovery and data mining, pages 1631–1642, 2023.

[26] Christopher Morris, Nils M Kriege, Franka Bause, Kristian Kersting, Petra Mutzel, and Marion Neumann. Tudataset: A collection of benchmark datasets for learning with graphs. arXiv preprint arXiv:2007.08663, 2020.

[27] Chaoxi Niu, Guansong Pang, and Ling Chen. Graph-level anomaly detection via hierarchical memory networks. In Joint European conference on machine learning and knowledge discovery in databases, pages 201–218. Springer, 2023.

[28] Junjun Pan, Yixin Liu, Rui Miao, Kaize Ding, Yu Zheng, Quoc Viet Hung Nguyen, Alan Wee-Chung Liew, and Shirui Pan. Explainable and fine-grained safeguarding of llm multi-agent systems via bi-level graph anomaly detection. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 30491–30506, 2026.

[29] Junjun Pan, Yixin Liu, Xin Zheng, Yizhen Zheng, Alan Wee-Chung Liew, Fuyi Li, and Shirui Pan. A label-free heterophily-guided approach for unsupervised graph fraud detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 12443–12451, 2025.

[30] Junjun Pan, Yixin Liu, Yu Zheng, Lianhua Chi, Alan Wee-Chung Liew, and Shirui Pan. Camera: Adapting to semantic camouflage in unsupervised text-attributed graph fraud detection. In International Joint Conference on Artificial Intelligence, 2026.

[31] Junjun Pan, Yixin Liu, Chuan Zhou, Fei Xiong, Alan Wee-Chung Liew, and Shirui Pan. Correcting false alarms from unseen: Adapting graph anomaly detectors at test time. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 24691–24699, 2026.

[32] Junjun Pan, Yu Zheng, Yue Tan, and Yixin Liu. A Survey of Generalization of Graph Anomaly Detection: From Transfer Learning to Foundation Models . In 2025 IEEE International Conference on Knowledge Graph (ICKG), pages 316–323, Los Alamitos, CA, USA, November 2025. IEEE Computer Society.

[33] Yanyu Qian, Yue Tan, Yixin Liu, Wang Yu, and Shirui Pan. Dynhd: Hallucination detection for diffusion large language models via denoising dynamics deviation learning. In Findings of the Associationfor Computational Linguistics: EMNLP 2026, 2026.

[34] Hezhe Qiao, Chaoxi Niu, Ling Chen, and Guansong Pang. Anomalygfm: Graph foundation model for zero/few-shot anomaly detection. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 2326–2337, 2025.

[35] Chen Qiu, Marius Kloft, Stephan Mandt, and Maja Rudolph. Raising the bar in graph-level anomaly detection. In International Joint Conference on Artificial Intelligence, 2022.

[36] Ida Schomburg, Antje Chang, Christian Ebeling, Marion Gremse, Christian Heldt, Gregor Huhn, and Dietmar Schomburg. Brenda, the enzyme database: updates and major new developments. Nucleic acids research, 32(suppl\_1):D431–D433, 2004.

[37] Qichao Shentu, Beibu Li, Kai Zhao, Yang Shu, Zhongwen Rao, Lujia Pan, Bin Yang, and Chenjuan Guo. Towards a general time series anomaly detector with adaptive bottlenecks and dual adversarial decoders. In The Thirteenth International Conference on Learning Representations, 2025.

[38] Jeffrey J Sutherland, Lee A O’brien, and Donald F Weaver. Spline-fitting with a genetic algorithm: A method for developing classification structure- activity relationships. Journal of chemical information and computer sciences, 43(6):1906–1915, 2003.

[39] Yue Tan, Guodong Long, Jing Jiang, and Chengqi Zhang. Influence-oriented personalized federated learning. In IEEE International Conference on Data Mining, 2026.

[40] S Vichy N Vishwanathan, Nicol N Schraudolph, Risi Kondor, and Karsten M Borgwardt. Graph kernels. The Journal ofMachine Learning Research, 11:1201–1242, 2010.

[41] Nikil Wale, Ian A Watson, and George Karypis. Comparison of descriptor spaces for chemical compound retrieval and classification. Knowledge and Information Systems, 14(3):347–375, 2008.

[42] Xiaolei Wang, Yuexin Wang, Tianhong Dai, Huihui Bai, Yao Zhao, and Jimin Xiao. Hunting normality from query sample via residual learning for generalist anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 43093–43102, 2026.

[43] Yili Wang, Yixin Liu, Xu Shen, Chenyu Li, Kaize Ding, Rui Miao, Ying Wang, Shirui Pan, and Xin Wang. Unifying unsupervised graph-level anomaly detection and out-of-distribution detection: A benchmark. In International Conference on Learning Representations, 2025.

[44] Yuexin Wang, Xiaolei Wang, Yizheng Gong, and Jimin Xiao. Normal-abnormal guided generalist anomaly detection. Advances in Neural Information Processing Systems, 38:62560– 62585, 2026.

[45] Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S Pappu, Karl Leswing, and Vijay Pande. Moleculenet: a benchmark for molecular machine learning. Chemical science, 9(2):513–530, 2018.

[46] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and Philip S Yu. A comprehensive survey on graph neural networks. IEEE transactions on neural networks and learning systems, 32(1):4–24, 2020.

[47] Dong Xiangyu, Zhang Xingyi, and Wang Sibo. Rayleigh quotient graph neural networks for graph-level anomaly detection. In ICLR, 2024.

[48] Chunjing Xiao, Shikang Pang, Wenxin Tai, Yanlong Huang, Goce Trajcevski, and Fan Zhou. Motif-consistent counterfactuals with adversarial refinement for graph-level anomaly detection. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3518–3526, 2024.

[49] Pinar Yanardag and Svn VN Vishwanathan. Deep graph kernels. In Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining, pages 1365–1374, 2015.

[50] Zhenyu Yang, Ge Zhang, Jia Wu, Jian Yang, Shan Xue, Amin Beheshti, Hao Peng, and Quan Z Sheng. Global interpretable graph-level anomaly detection via prototype. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 3586–3597, 2025.

[51] Xincheng Yao, Zixin Chen, Chao Gao, Guangtao Zhai, and Chongyang Zhang. Resad: A simple framework for class generalizable anomaly detection. Advances in Neural Information Processing Systems, 37:125287–125311, 2024.

[52] Ge Zhang, Zhenyu Yang, Jia Wu, Jian Yang, Shan Xue, Hao Peng, Jianlin Su, Chuan Zhou, Quan Z Sheng, Leman Akoglu, et al. Dual-discriminative graph neural network for imbalanced graph-level anomaly detection. Advances in Neural Information Processing Systems, 35:24144– 24157, 2022.

[53] Si Zhang, Hanghang Tong, Jiejun Xu, and Ross Maciejewski. Graph convolutional networks: a comprehensive review. Computational Social Networks, 6(1):1–23, 2019.

[54] Xiong Zhang, Hong Peng, Changlong Fu, Xin Jin, Yun Yang, and Cheng Xie. Ta-ggad: Testing-time adaptive graph model for generalist graph anomaly detection. arXiv preprint arXiv:2603.09349, 2026.

[55] Yunfeng Zhao, Yixin Liu, Qingfeng Chen, Shiyuan Li, Yue Tan, and Shirui Pan. Fedcigar: A personalized reconstruction approach for federated graph-level anomaly detection. In International Joint Conference on Artificial Intelligence, 2026.

[56] Xin Zheng, Yixin Liu, Zhifeng Bao, Meng Fang, Xia Hu, Alan Wee-Chung Liew, and Shirui Pan. Towards data-centric graph machine learning: Review and outlook. arXiv preprint arXiv:2309.10979, 2023.

[57] Qihang Zhou, Guansong Pang, Yu Tian, Shibo He, and Jiming Chen. Anomalyclip: Objectagnostic prompt learning for zero-shot anomaly detection. In The Twelfth International Conference on Learning Representations, 2024.

[58] Jiawen Zhu and Guansong Pang. Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17826–17836, 2024.

## A Detailed Related Work

## A.1 Graph-Level Anomaly Detection

Graph-level anomaly detection (GLAD) focuses on identifying abnormal graphs that significantly deviate from the dominant graph patterns [52, 27, 38]. In many real-world applications, anomaly annotations are scarce or unavailable, making one-class learning the mainstream setting for GLAD. Under this setting, models are trained solely on normal graphs, while anomalous graphs are identified during inference based on their deviations from learned normal patterns (also referred to as the unsupervised setting in prior studies).

Recently, GNN-based approaches have become the prevailing solutions for GLAD and achieved promising performance. These methods can be broadly categorized into two groups, namely, reconstruction-based methods and contrastive learning-based methods. The reconstruction-based methods aim to reconstruct input graphs and detect anomalies based on reconstruction errors, assuming that anomalous graphs are harder to reconstruct [22, 5, 14]. For example, GLADC [22] takes autoencoder-based reconstruction as an important supervision signal for GLAD. MUSE [14] introduces a two-stage strategy that combines graph reconstruction with reconstruction-error modeling for anomaly identification. FedCIGAR [55] employs a personalized reconstruction approach for Federated learning [39].

The contrastive learning-based methods learn graph representations by contrasting multiple views of the data, using consistency or agreement between views to distinguish normal patterns from anomalies [15, 18, 50, 11]. For instance, SIGNET [18] measures graph abnormality by maximizing the mutual information between the original graph view and its corresponding dual hypergraph representation. CVTGAD [15] further introduces a transformer-based cross-view attention mechanism to directly capture the inter-relationship between augmented graph views for anomaly detection.

There are also several studies that leverage advanced techniques, such as evolutionary mapping [25], mixture-of-experts [4], and knowledge distillation [23], to address the GLAD problem. Specifically, GmapAD [25] adopts an evolutionary search strategy to identify representative nodes and detects anomalies according to the similarity between graphs and the selected representatives. OCGTL [35] improves graph-level anomaly detection by integrating self-supervised transformation learning to alleviate issues such as hypersphere collapse and performance flip in deep one-class learning.

Existing GLAD methods usually follow a one-dataset-one-model paradigm, where models are trained using either purely normal data (one-class unsupervised setting [43]) or labeled normal and anomalous data (supervised setting [52, 25, 47]), and evaluated on the same dataset. This paradigm increases the requirement for large amounts of labeled or curated data and limits its broad applicability.

## A.2 Generalist Anomaly Detection

Recently, anomaly detection research has increasingly shifted toward generalist paradigms, where a unified pre-trained model is expected to generalize across different datasets and application scenarios. Instead of retraining separate models for each domain, these methods usually rely on pretrained foundation architectures together with lightweight test-time adaptation strategies such as prompts, contextual examples, or conditional modules.

In computer vision, many generalist approaches are built upon vision-language models (VLMs). For example, WinCLIP reformulates industrial anomaly detection by matching image features with compositional textual prompts at multiple granularities in CLIP, supporting zero-shot and few-shot anomaly recognition without additional task-specific optimization [7, 57]. Besides VLM-based approaches, generative foundation models have also been explored. Diffusion-based methods reuse denoising processes as generic perceptual priors and identify anomalies according to reconstruction or score discrepancies [1]. To further reduce dependence on manually designed prompts, InCTRL introduces an in-context residual learning framework, where query samples are compared with a few normal examples during inference for adaptive anomaly estimation [58]. Recent works further enhance generalization through residual feature modeling [51], normal-abnormal guidance [44], and residual-guided normality learning [42].

For time-series anomaly detection, recent studies also investigate reusable pretrained detectors across domains. FAE pretrains a variational autoencoder-based generative model on large-scale time-series corpora and combines it with dilated convolutions to model complex temporal dependencies, aiming to support zero-shot anomaly detection on unseen datasets [12]. DADA further improves crossdomain transferability by introducing adaptive bottlenecks and dual adversarial decoders to handle domain divergence and diverse anomaly patterns [37].

Generalist anomaly detection has also been explored in graph learning, but they are all focus on node-level anomaly detection. ARC adopts an in-context adaptation strategy that extracts datasetspecific regularities from a few normal nodes at inference time, combining feature alignment, residual encoding, and cross-attention scoring for dynamic adaptation [19]. Similarly, AnomalyGFM develops a graph foundation model by aligning learnable normal and abnormal prototypes with residual neighborhood representations, enabling both zero-shot inference and few-shot adaptation on unseen graph datasets [34]. Recently, REFI-GAD introduces a fingerprint-based approach to boost the performance [20]. However, due to the fundamental differences between node-level and graph-level anomaly detection, these node-level approaches cannot be directly applied to GLAD.

Overall, existing generalist anomaly detection studies suggest that strong transferability mainly relies on robust pretrained representations together with effective test-time adaptation mechanisms, rather than dataset-specific retraining. Following this idea, our method targets a new research direction, i.e., generalist model for GLAD.

## B Pseudo-anomalies Construction for Pre-training

Training generalist models typically depends on large volumes of labeled data [21]. However, annotating graph-structured data demands substantial domain expertise, making it infeasible in practice [18]. In this section, we provide a detailed version of how we construct $\tau ^ { \mathrm { t r a i n } }$ via crossdomain motif injection. Specifically, given a batch of graphs $B = \{ \mathcal { G } _ { 1 } ^ { B } , \ldots , \mathcal { G } _ { B } ^ { B } \}$ sampled from a dataset $\mathcal { D } _ { \mathrm { c u r r } } ^ { \mathrm { t r a i n } } \in \mathcal { T } ^ { \mathrm { t r a i n } }$ , we treat these graphs as inliers for the corresponding graph application. Pseudo-anomalies are then constructed by incorporating cross-dataset information, where each normal graph is augmented with motifs sampled from graphs drawn from other datasets to introduce anomalous information. Formally, for each graph $\mathcal { G } _ { i } ^ { B } \in \bar { B } ,$ , we build a corresponding outlier graph $\bar { \mathcal { G } } _ { i } ^ { B }$ as a pseudo-anomaly. Let $\{ \mathcal { M } _ { 1 } , . . . , \mathcal { M } _ { K } \}$ denote the set of motifs sampled via random walks from graphs drawn from $\mathcal { T } ^ { \mathrm { t r a i n } } \setminus \{ \mathcal { D } _ { \mathrm { c u r r } } ^ { \mathrm { t r a i n } } \}$ , where each motif is defined as $\bar { \mathcal { M } } _ { k } = ( \mathcal { V } _ { k } ^ { \mathrm { m o t i f } } , \mathcal { E } _ { k } ^ { \mathrm { m o t i f } } , \mathbf { X } _ { k } ^ { \mathrm { m o t i f } } )$ We then introduce these motifs to $\mathcal { G } _ { i } ^ { B }$ by connecting a random node with the root node of the motif, resulting in the pseudo anomaly graph:

$$
\bar { \mathcal { G } } _ { i } ^ { \mathcal { B } } = ( \gamma _ { i } ^ { \mathcal { B } } \cup \gamma _ { 1 } ^ { \mathrm { m o t i f } } \cup . . . , \mathcal { E } _ { i } ^ { \mathcal { B } } \cup \mathcal { E } _ { 1 } ^ { \mathrm { m o t i f } } \cup \cdot \cdot \cdot \cup \mathcal { E } ^ { \mathrm { j o i n } } , \mathbf { X } _ { i } ^ { \mathcal { B } } \oplus \mathbf { X } _ { 1 } ^ { \mathrm { m o t i f } } \oplus . . . ) ,\tag{7}
$$

where ${ \mathcal { E } } ^ { \mathrm { j o i n } }$ contains the edges that join the root node of motifs and the graph. Moreover, we denote the mask of injected nodes (i.e., nodes in out-domain motifs) as $\tilde { \mathbf { m } } _ { i } \in \{ 0 , 1 \} ^ { | \tilde { \mathcal { V } } _ { i } ^ { B } | }$ <sup>|</sup>, where

$$
( \tilde { \mathbf { m } } _ { i } ) _ { j } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f } v _ { j } \in \bigcup \mathcal { V } _ { k } ^ { \mathrm { m o t i f } } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

The cross-domain motif injection provides a simple yet effective way to construct negative samples. In many graph applications, the distinction between normal and anomalous instances is often determined by discriminative subgraphs that encapsulate the graph’s key information [18]. $( \mathrm { e . g . }$ , functional groups in molecular graphs that determine chemical properties.) Injecting out-of-domain motifs perturbs the semantic consistency of inlier graphs, thereby yielding realistic pseudo-anomalies. Moreover, it also offers a fine-grained supervisory signal with injected out-of-domain motifs to guide the training of the generalist GLAD model.

## C Discrepancy Design

We construct a set of complementary discrepancy signals for each incoming graph, providing a holistic characterization of the incoming graph.

Prior Discrepancy. We first utilize the pre-trained model to obtain the subgraph $G ^ { \prime }$ on the test graph. Hence, the prior discrepancy is defined as:

$$
\begin{array} { r } { s ^ { \mathrm { p r i o r } } = - \cos ( \mathbf { z } _ { \mathcal { G } ^ { \prime } } , \mathbf { t } _ { \mathbf { c } } ) , } \end{array}\tag{8}
$$

Algorithm 1 GenGLAD: Cross-domain Motif Injection for Constructing Pseudo-anomaly   
Input: Training task collection $\tau _ { \ast }$ , batch of normal graph $B \subset \mathcal { D } _ { \mathrm { c u r r } } ^ { \mathrm { t r a i n } } \in \mathcal { T }$ , number of injected motifs   
$K ,$ random walk length l.   
Output: A batch of pseudo-anomalies $\bar { B }$ with injected node masks.   
1: $\bar { B }  \emptyset$   
2: for graph $\mathcal { G } _ { i } ^ { \mathcal { B } } = ( \mathcal { V } _ { i } ^ { \mathcal { B } } , \mathcal { E } _ { i } ^ { \mathcal { B } } , \mathbf { X } _ { i } ^ { \mathcal { B } } ) \in \mathcal { B } _ { i }$ do   
3: Sample motifs $\{ \mathcal { M } _ { 1 } , . . . , \mathcal { M } _ { K } \}$ from $\mathcal { T } ^ { \mathrm { t r a i n } } \setminus \{ \mathcal { D } _ { \mathrm { c u r r } } ^ { \mathrm { t r a i n } } \}$ with random walk of length l.   
4: ${ \bar { \mathcal { G } } } _ { i } ^ { B } \gets { \mathcal { G } } _ { i } ^ { B } .$   
5: for $k = \mathrm { { i } } , \ldots , K$ do   
6: Construct the joining edge $e _ { k } ^ { \mathrm { j o i n } }$ by connecting the root node of $\mathcal { M } _ { k }$ with a randomly selected   
nodes in $\mathcal { V } _ { i } ^ { B }$   
7: Attach $\mathcal { M } _ { k }$ to current graph: $\bar { \mathcal { G } } _ { i } ^ { \mathcal { B } } \gets ( \bar { \mathcal { V } } _ { i } ^ { \mathcal { B } } \cup \mathcal { V } _ { k } ^ { \mathrm { m o t i f } } , \bar { \mathcal { E } } _ { i } ^ { \mathcal { B } } \cup \mathcal { E } _ { k } ^ { \mathrm { m o t i f } } \cup \{ e _ { k } ^ { \mathrm { j o i n } } \} , \bar { \mathbf { X } } _ { i } ^ { \mathcal { B } }$ ⊕ X<sup>motif</sup><sub>k</sub> ).   
8: end for   
9: Obtain the injected node mask $\tilde { \mathbf { m } } _ { i } \in \{ 0 , 1 \}$   
10: $\bar { \mathcal { B } }  \bar { \mathcal { B } } \cup \{ ( \breve { \bar { \mathcal { G } } } _ { i } ^ { \mathcal { B } } , \tilde { \mathbf { m } } _ { i } ) \}$   
11: end for   
12: return ${ \bar { B } } .$

which estimates the difficulty of estimating the discriminative subgraph with the knowledge learned during pre-training.

Domain Discrepancy. We quantify how much the graph deviates from the domain-level context. Let $\mathbf { x } _ { \mathcal { G } ^ { \prime } } = \mathrm { R e a d o u t } _ { \mathcal { G } ^ { \prime } } ( \mathbf { X } )$ denote the readout attribute on the discriminative subgraph. To capture complementary geometry, we consider both cosine and Euclidean distances to get discrepancy in both angular and scale differences:

$$
\begin{array} { r } { s _ { \mathrm { c o s } } ^ { \mathrm { d o m a i n } } = - \cos ( \mathbf { x } _ { \mathcal { G } ^ { \prime } } , \mathbf { c } ) , \quad s _ { \mathrm { e u c } } ^ { \mathrm { d o m a i n } } = \| \mathbf { x } _ { \mathcal { G } ^ { \prime } } - \mathbf { c } \| _ { 2 } . } \end{array}\tag{9}
$$

Class-level Discrepancy. We further measure deviation from the class prototype q estimated from few-shot normal graphs:

$$
\mathbf { q } = \frac { 1 } { | \mathcal { T } ^ { \mathrm { t e s t } } | } \sum _ { \mathcal { G } \in \mathcal { T } ^ { \mathrm { t e s t } } } \mathrm { R e a d o u t } _ { \mathcal { G } ^ { \prime } } ( \mathbf { H } ) ,\tag{10}
$$

$$
\begin{array} { r } { s _ { \mathrm { c o s } } ^ { \mathrm { c l a s s } } = - \cos ( \mathbf { z } _ { \mathcal { G } ^ { \prime } } , \mathbf { q } ) , \quad s _ { \mathrm { e u c } } ^ { \mathrm { c l a s s } } = \| \mathbf { z } _ { \mathcal { G } ^ { \prime } } - \mathbf { q } \| _ { 2 } . } \end{array}\tag{11}
$$

Instance-level Discrepancy. Beyond prototype-level statistics, we capture fine-grained relational structure via similarity distributions. Let $\mathcal { D } _ { \mathcal { S } }$ denote the pairwise similarity distribution among support graphs, and $\mathcal { D } _ { G }$ denote the similarity distribution between the query graph and each support instance:

$$
\mathcal { D } _ { G } = \{ \cos ( { \bf z } _ { \mathcal { G } ^ { \prime } } , { \bf z } _ { \mathcal { G } _ { s } } ) \} _ { s = 1 } ^ { K } .\tag{12}
$$

We then define the instance-level discrepancy as the 1-D Wasserstein distance:

$$
s ^ { \mathrm { i n s } } = W _ { 1 } ( { \cal { D } } _ { S } , { \cal { D } } _ { G } ) ,\tag{13}
$$

where the Wasserstein distance captures distributional discrepancies beyond point-wise similarity.

## D Pseudo-anomalies Construction for Adaptation

In this section, we provide a detailed version of how we construct pseudo-anomalies from few-shot normal data supervision. Since the few-shot supervision only contains normal instances, directly learning the w for the scoring function is infeasible. However, as all normal instances come from the same domain, it is unsuitable apply cross-domain motif injection. While reusing the motif sampling during training can be a potential bandit, this may raise potential data leakage and privacy concerns.

To address the aforementioned concerns, we introduce an alternative pseudo-anomaly construction way that effectively utilizes the few-shot normal graphs to generate semantically corrupted graphs and graphs with over-strong patterns, thereby supervising the contrastive learning objective with target domain-specific insights in a data-efficient manner. Specifically, to simulate anomalous graphs with out-of-domain characteristics, we build pseudo-anomaly graphs (negative outliners) by structure perturbation, i.e., applying structural corruption to normal graphs by shuffling edges, thereby destroying meaningful relational structures while introducing out-of-domain structure patterns. Formally, given a normal graph $\mathcal { G } = ( \nu , \mathcal { E } , \mathbf { X } )$ , we generate a corrupted variant by replacing its edge set with randomly sampled node pairs. Specifically, we construct a new edge set E<sup>˜</sup> by sampling without replacement from the set of all possible unordered node pairs:

$$
\tilde { \mathcal { E } } \sim \mathrm { U n i f o r m } \Big ( \{ ( u , v ) \mid u , v \in \mathcal { V } , u \neq v \} \Big ) , \quad | \tilde { \mathcal { E } } | = | \mathcal { E } | .\tag{14}
$$

The resulting pseudo-anomalous graph is defined as:

$$
\tilde { \mathcal { G } } _ { \mathrm { s h u f } } ^ { \mathrm { a n o m a l y } } = ( \mathcal { V } , \tilde { \mathcal { E } } , { \bf X } ) .\tag{15}
$$

In addition to structural corruption, real-world anomalies may also arise from over-expressed or exaggerated in-domain patterns [14], where certain structures appear with abnormally high intensity or frequency. For example, in molecular graphs, excessive presence of certain functional groups may lead to toxicity; in social networks, while active users are typically normal, abnormally high interaction frequency can indicate spammers. Motivated by this, we conduct sample bridging to construct positive outliers by amplifying patterns observed in few-shot normal samples.

Specifically, given two normal graphs $\mathcal { G } _ { i } = ( \mathcal { V } _ { i } , \mathcal { E } _ { i } , { \bf X } _ { i } )$ and $\mathcal { G } _ { j } = ( \mathcal { V } _ { j } , \mathcal { E } _ { j } , { \bf X } _ { j } ) .$ , we generate a bridged graph by merging them and introducing additional cross-graph connections. The resulting graph is defined as:

$$
\tilde { \mathcal { G } } _ { \mathrm { b r i d g e } } ^ { \mathrm { a n o m a l y } } = ( \mathcal { V } _ { i } \cup \mathcal { V } _ { j } , \mathcal { E } _ { i } \cup \mathcal { E } _ { j } \cup \mathcal { E } _ { \mathrm { b r i d g e } } , \mathbf { X } _ { i } \oplus \mathbf { X } _ { j } ) ,\tag{16}
$$

where $\mathcal { E } _ { \mathrm { b r i d g e } }$ is the randomly sampled edge set to connect two graphs together. By bridging graphs together, this construction amplifies structural patterns beyond their normal range, thereby guilding GenGLAD to spot any over-strong in-domain patterns.

We then utilize the constructed pseudo-anomalies with the few-shot supervisions to optimize the loss function:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { a d a p t } } = \mathrm { B C E } \big ( \sigma ( s ) , y _ { \mathrm { p s e u d o } } ^ { \mathrm { a d a p t } } \big ) . } \end{array}\tag{17}
$$

while keeping the encoder and memory bank frozen. This preserves the transferable knowledge learned during pre-training and avoids overfitting under few-shot supervision. Overall, structure perturbation and sampling bridging enable GenGLAD to learn target-application specific knowledge for GLAD, thereby complementing the transferable knowledge learned during the pre-training stage with cross-domain motif injection.

## E Algorithm

The procedure of pre-training and few-shots adaptation GenGLAD is summarized in Algorithm 2 and Algorithm 3, respectively.

## F Complexity Analysis

We analyze the computational complexity of each stage in GenGLAD. Let N denote the total number of graphs in T<sup>train</sup>, V and E the average numbers of nodes and edges in a graph, respectively, M the number of memory tokens, and K the number of retrieved memory. We also denote the number of few-shot normal graphs as $N ^ { \mathrm { a d a p t } }$

During pre-training, the time complexity of training set construction is $\mathcal { O } ( N ( V + E ) )$ , where motif caching is employed to reduce computational overhead. Since the random-walk-sampled motifs are relatively small, the overall complexity is dominated by the graph merging operation, which primarily incurs copying overhead. When it comes to pre-training, for each graph, the GNN encoder requires ${ \mathcal { O } } ( E )$ for message passing. The subsequent subgraph sampling costs $\mathcal { O } ( K V + E )$ . In addition, it also takes $\mathcal { O } ( M )$ to obtain the domain-related memory tokens for each batch. Overall, during pre-training, GenGLAD requires $\mathcal { O } ( N ( K V + E ) )$ ) time per training epoch, in addition to a one-time training set construction cost of $\mathcal { O } ( \dot { N } ( \dot { V } + E ) )$ and a per-batch overhead of $\mathcal { O } ( M )$ . As K and M are small constants, our method maintains good training efficiency and scales comparably to training a vanilla GCN [53].

Algorithm 2 GenGLAD: Generalist Pre-training   
Input: Training task collection $\overline { { \mathcal { T } ^ { \mathrm { t r a i n } } } } = \{ \mathcal { D } _ { 1 } ^ { \mathrm { t r a i n } } , \cdot \cdot \cdot , \mathcal { D } _ { N } ^ { \mathrm { t r a i n } } \}$ , training epochs $E ^ { \mathrm { p r e - t r a i n } }$ , batch size $B ,$   
top-K retrieval number, Gumbel temperature τ, learning rate lr.   
Output: Pre-trained GenGLAD.   
1: Randomly initialize the parameters of GenGLAD.   
2: $\mathcal { P }  \{ ( \mathcal { \bar { D } } _ { i } ^ { \mathrm { t r a i n } } : \mathcal { M } ^ { \mathrm { p o o l } } ) \} _ { i = 1 } ^ { \bar { N } }$ (Pre-sampling motifs for each dataset to avoid computation overhead   
during cross-domain motif injection)   
3: for $e \bar { p } o c h = 1 , \ldots , E ^ { \mathrm { p r e - t r a i n } }$ do   
4: $B ^ { \mathrm { a l l } }  ($ (randomly split input datasets into batches of size B.)   
5: for batch $B \in \breve { B ^ { \mathrm { a l l } } }$ do   
6: $\mathcal { L } _ { \mathrm { g r a p h } }  0$   
7: $\dot { \mathcal { L } _ { \mathrm { n o d e } } }  0$   
8: $\mathcal { L } _ { \mathrm { r e g } }  0$   
9: Retrieve top-K relevant memory tokens $\tilde { \mathbf { T } }$ from T according to cosine similarity with c.   
10: Obtain pseudo-anomalies B<sup>¯</sup> following Algorithm 1 with the cached motifs $\mathcal { P }$   
11: ${ \tilde { B } } _ { i } \gets B _ { i } \cup { \bar { B } } _ { i }$   
12: $\begin{array} { r } { \mathbf { c } \gets \frac { 1 } { | B _ { i } | } \sum _ { \mathcal { G } \in B _ { i } } } \end{array}$ Readou $\operatorname { \dot { \rho } } _ { \mathcal { G } } ( \mathbf { X } )$   
13: $\mathbf { t _ { c } } \gets \mathrm { M e a n } ( \tilde { \mathbf { T } } ) .$   
14: for graph $\tilde { \mathcal { G } } = ( \tilde { \mathcal { V } } , \tilde { \mathcal { E } } , \tilde { \mathbf { X } } ) \in \tilde { \mathcal { B } } _ { i }$ do   
15: H $\dot { { \bf \theta } } \mathrm { ~ G N N } ( { \bf \dot { X } } , { \bf A } )$   
16: Obtain $\mathcal { G } ^ { \prime }$ , m following Section 4.1.1.   
17: $\mathbf { z } _ { \mathcal { G } ^ { \prime } }  \operatorname { R e a d o u t } _ { \mathcal { G } ^ { \prime } } ( \mathbf { H } ) \overline { { \mathbf { \Psi } } }$   
18: $\mathcal { L } _ { \mathrm { g r a p h } } \gets \mathcal { L } _ { \mathrm { g r a p h } } + \mathrm { B C E } \Big ( \sigma \big ( 1 - \cos ( \mathbf { z } _ { \mathcal { G } ^ { \prime } } , \mathbf { t } _ { \mathbf { c } } ) \big ) , \tilde { y } \Big )$   
19: $\mathcal { L } _ { \mathrm { n o d e } }  \mathcal { L } _ { \mathrm { n o d e } } - \mathbb { E } _ { u \sim \tilde { \mathbf { y } } _ { u } ^ { ( n ) } = 1 } [ \log ( 1 - m _ { u } ) ] .$   
20: $\mathcal { L } _ { \mathrm { r e g } }  \mathcal { L } _ { \mathrm { r e g } } + \| \mathbf { m } \| _ { 1 }$   
21: end for   
22: $\mathcal { L } _ { \mathrm { p r e } }  \mathcal { L } _ { \mathrm { g r a p h } } + \lambda _ { n } \mathcal { L } _ { \mathrm { n o d e } } + \lambda _ { s } \mathcal { L } _ { \mathrm { r e g } } .$   
23: Back-propagate $\mathcal { L } _ { \mathrm { p r e } }$ to update the parameters of GenGLAD with learning rate lr.   
24: end for   
25: end for   
26: return pre-trained GenGLAD model.

During adaptation, pseudo-anomalies are constructed from $N ^ { \mathrm { a d a p t } }$ few-shot normal graphs through structure perturbation and sample bridging. Structure perturbation requires $\mathcal { O } ( E ) { }$ , while graph bridging costs ${ \mathcal { O } } ( V + E )$ . Overall, each round of pseudo-anomaly construction requires $\mathcal { O } ( N ^ { \mathrm { a d a p t } } ( \dot { V } +$ E)) as we obtain a balanced number of normal and pseudo-anomaly during training. As inference only involves memory-guided subgraph extraction and discrepancy scoring, resulting in a complexity of $\mathcal { O } ( K V + E )$ . To summarize, each adaptation round incurs a time complexity of $\breve { \mathcal { O } } ( N ^ { \mathrm { a d a p t } } ( \dot { K } V + \dot { E } ) )$ , in addition to an $\mathcal { O } ( M )$ overhead for obtaining the domain-related memory tokens at the beginning of training. With the help of light-weight module design, the adaptation of GenGLAD is extremely efficient as demonstrated in Figure 4b.

## G Detailed Experimental Setup

## G.1 Description of Datasets

In this section, we present additional details of all graph datasets used in our experiments. Table 5 reports key statistics, including the number of graphs as well as the average number of nodes and edges. All datasets are sourced from TUDataset [26], a publicly available graph benchmark repository. We follow UB-GOLD [43] to configure the GLAD experimental settings. Specifically, the detailed descriptions for the datasets are given as follows:

NCI1 [41]: A dataset of chemical compounds screened for anti-cancer activity, where each graph corresponds to a molecular structure. It is widely used for evaluating graph learning methods in anti-cancer drug discovery tasks.

```latex
Algorithm 3 GenGLAD: Few-shot Adaptation
Input: Pre-trained GenGLAD, few-shot normal set ${ \mathcal { S } } \subset { \mathcal { D } } _ { i } ^ { \mathrm { t e s t } }$ , adaptation epochs $E ^ { \mathrm { a d a p t } }$ , learning rate
lr.
Output: Anomaly scores for graphs in $\mathcal { D } _ { i } ^ { \mathrm { t e s t } } .$
1: Randomly initialize scoring parameter w.
2: Retrieve top-K relevant memory tokens T<sup>˜</sup> from T according to cosine similarity with c.
3: Construct the pseudo-anomaly set $\bar { S } \ B ^ { \mathrm { a d a p t } }$ using following Appendix D.
4: $\mathbf { c } \gets \frac { 1 } { | S | } \sum _ { \mathcal { G } \in S }$ Readou $\operatorname { \dot { \rho } } _ { \mathcal { G } } ( \mathbf { X } )$
5: $\begin{array} { r } { \mathbf { q }  \frac { 1 } { | S | } \sum _ { \mathcal { G } \in S } \mathbf { z } _ { \mathcal { G } ^ { \prime } } } \end{array}$
6: for $e p o c h = 1 , \ldots , E ^ { \mathrm { a d a p t } }$ do
7: $\mathcal { L } _ { \mathrm { a d a p t } }  0$
8: Obtain S<sup>¯</sup> following Algorithm 1 and form fine-tuning batch ${ \tilde { S } } _ { i } \gets S _ { i } \cup { \bar { S } } _ { i }$
9: for graph $\tilde { \mathcal { G } } _ { i } \in \tilde { \mathcal { S } }$ do
10: Compute the four discrepancies following Section 4.2.
11: $\mathcal { L } _ { \mathrm { a d a p t } } \gets \mathcal { L } _ { \mathrm { a d a p t } } + \mathrm { B C E } ( \sigma ( \mathbf { w } ^ { \top } \mathbf { s } ) , y _ { i } ^ { \mathrm { a d a p t } } )$
12: end for
13: Update scoring parameter w by back-propagating $\mathcal { L } _ { \mathrm { a d a p t } }$ with learning rate lr.
14: end for
15: for query graph ${ \mathcal { G } } \in { \mathcal { D } } _ { i } ^ { \mathrm { t e s t } }$ do
16: Compute the four discrepancies following Section 4.2.
17: $s _ { i } \gets \sigma ( \mathbf { w } ^ { \top } \mathbf { s } )$
18: end for
19: return anomaly scores s.
```

Table 4: Statistics of the datasets.
<table><tr><td>Datasets</td><td># Graphs</td><td>Avg. Nodes</td><td>Avg. Edges</td></tr><tr><td colspan="4">Molecular Graphs</td></tr><tr><td>NCI1</td><td>4110</td><td>29.87</td><td>64.60</td></tr><tr><td>AIDS</td><td>2000</td><td>15.59</td><td>47.98</td></tr><tr><td>BZR</td><td>405</td><td>35.75</td><td>76.72</td></tr><tr><td>COX2</td><td>467</td><td>41.22</td><td>86.89</td></tr><tr><td>DHFR</td><td>756</td><td>42.43</td><td>89.09</td></tr><tr><td>MUTAG</td><td>188</td><td>17.93</td><td>57.52</td></tr><tr><td>PTC_MR</td><td>344</td><td>14.29</td><td>29.38</td></tr><tr><td colspan="4">Biochemistry Graphs</td></tr><tr><td>DD</td><td>1178</td><td>284.32</td><td>1715.63</td></tr><tr><td>ENZYMES</td><td>600</td><td>32.63</td><td>124.27</td></tr><tr><td>PROTEINS</td><td>1113</td><td>39.06</td><td>145.63</td></tr><tr><td colspan="4">Social Networks</td></tr><tr><td>COLLAB</td><td>5000</td><td>74.49</td><td>4914.43</td></tr><tr><td>IMDB-BINARY</td><td>1000</td><td>19.77</td><td>212.84</td></tr><tr><td>REDDIT-BINARY</td><td>2000</td><td>429.63</td><td>1425.13</td></tr></table>

AIDS [26]: A collection of molecular graphs representing chemical compounds tested for antiviral activity against HIV, commonly used in antiviral drug discovery research.

BZR [45]: A molecular dataset of benzodiazepine receptor ligands, where nodes denote atoms and edges denote chemical bonds, designed for studying molecular interactions with benzodiazepine receptors.

COX2 [26]: A dataset consisting of cyclooxygenase-2 inhibitor molecules represented as graphs, used for analyzing the structural properties of anti-inflammatory compounds.

DHFR [26]: A collection of molecular graphs for compounds that inhibit the dihydrofolate reductase (DHFR) enzyme, important for research in cancer therapy and antibacterial drug development.

MUTAG [8]: A benchmark dataset containing 188 mutagenic aromatic and heteroaromatic nitro compounds. Each graph represents a molecule, and the classification task determines whether the compound exhibits mutagenic effects on the bacterium Salmonella typhimurium.

PTC-MR [13]: A subset of the Predictive Toxicology Challenge (PTC) dataset, where molecular compounds are labeled according to their carcinogenicity. The suffix $\mathbf { \cdots } \mathbf { \mathit { M } } \mathbf { \mathbf { R } } ^ { \prime \prime }$ indicates experiments conducted on male rats.

DD [40]: A protein-protein interaction dataset in which graphs capture the complex relationships among proteins, supporting the study of biological processes and cellular functions.

ENZYMES [36]: A dataset of protein tertiary structures, where each graph represents an enzyme with nodes corresponding to secondary structure elements and edges indicating spatial adjacency relationships.

PROTEINS [9]: A protein graph dataset containing structural and interaction information, where nodes represent amino acids and edges capture interactions between them, facilitating the analysis of protein functions and biological interactions.

COLLAB [49]: A scientific collaboration network dataset in which nodes correspond to researchers and edges indicate co-authorship relationships across multiple research domains.

IMDB-BINARY [49]: A movie collaboration network dataset where nodes represent actors and edges denote co-appearances in films, enabling the analysis of collaboration patterns in the film industry.

REDDIT-BINARY [49]: A dataset of Reddit discussion threads represented as graphs, where nodes correspond to users and edges capture interactions between them, reflecting the structural dynamics of online discussions.

As the native node and edge attributes may be unavailable or semantically incompatible across domains, we construct structure attributes, which not only addresses that issue but also unify the dimension. Such formulation can provides strong discriminative information for graph-level tasks from graph topology. Specifically, we follow the commonly used structure-based attribute construction, which consists of two components: the degree encoding $\mathbf { X } ^ { \mathrm { d e g } }$ and the random-walk positional encoding $\mathbf { X } ^ { \mathrm { R W } }$ . We concatenate them to obtain the structural embedding. The degree encoding is a one-hot representation of each node’s degree, designed to characterize its local connectivity. We use a fixed dimensionality of 32, with degrees exceeding the supported range clipped to the final bin. To capture higher-order structural information, we additionally employ random-walk positional encoding that denotes as the probability that a k-step random walk returns to the starting point. By concatenating them together, we obtain a 64-dimensional structural embedding.

## G.2 Details of Implementation

Hyper-parameters. We tune the key hyper-parameters of GenGLAD via random search over predefined grids. Specifically, the search space includes $\lambda _ { n } \in \{ 0 . 0 , 0 . 1 , 0 . 5 , 1 . 0 , 1 . 5 , 2 . 0 \} , \lambda _ { s } \in$ $\{ 0 . 0 , 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 1 5 , 0 . 2 \}$ , and the pre-training learning rate $\mathrm { l r ^ { p r e - t r a i n } } \in \{ 1 \times 1 0 ^ { - 4 } , 1 \times 1 0 ^ { - 3 } \}$ For the remaining hyper-parameters, we set the number of memory tokens M to 128 and the number of retrieved in-domain memory tokens K to 12. For structural encoding, we set the dimensionality of both $\mathbf { X } ^ { \mathrm { d e g } }$ and $\mathbf { X } ^ { \mathrm { R W } }$ to 32, yielding a 64-dimensional concatenated structure representation. The temperature parameter τ for Gumbel-Softmax and the hidden embedding dimension are set to 0.5 and 64, respectively. We pre-train the model for 30 epochs. For adaptation, we use a $\mathrm { l r } ^ { \mathrm { a d a p t } } = 1 \times 1 0 ^ { - 4 }$ and train for 10 epochs. We utilize the Adam optimizer for both pre-training and fine-tuning.

Implementation Pipeline. We employ a fixed set of hyperparameters to construct a generalist GLAD model across all datasets. First, all methods (including baselines and GenGLAD) are trained on the training set $\mathcal { T } ^ { \mathrm { t r a i n } }$ . Then, each method is adapted separately to the corresponding dataset using $\mathcal { T } ^ { \mathrm { t e s t } }$ . For adaptation and few-shot fine-tuning, we ensure that the few-shot supervision is identical across all baselines and GenGLAD to guarantee a fair comparison. During pretraining, the memory tokens are randomly initialized and jointly optimized with the model through backpropagation, while in finetuning they are frozen to avoid overfitting.

Evaluation Metrics. Following standard practice in GAD and GLAD [24, 43], we evaluate performance using two widely adopted and complementary metrics: the area under the receiver operating characteristic curve (AUROC) and the area under the precision–recall curve (AUPRC). Higher AUROC and AUPRC values indicate better detection performance. We report the mean AUROC and AUPRC along with standard deviations over five independent runs.

Table 6: AUROC of GenGLAD and its variants.  
Table 5: Anomaly detection performance in terms of AUPRC (in percent, mean±std). Highlighted are the results ranked first, second, and third.
<table><tr><td>Method</td><td>AIDS</td><td>BZR</td><td>COX2</td><td>DHFR</td><td>MUTAG</td><td>PTC_MR</td><td>ENZYMES</td><td>PROTEINS</td><td>IMDB-B</td><td>REDDIT-B</td></tr><tr><td colspan="9">Pre-training+Few-shot Fine-tuning</td></tr><tr><td>GLocalKD</td><td>82.47±0.06</td><td>26.54±0.06</td><td>25.83±0.09</td><td>69.52±0.13</td><td>87.22±0.05</td><td>44.04±0.03</td><td>81.69±0.02</td><td>53.43±0.05</td><td>50.21±0.05</td><td>69.93±0.13</td></tr><tr><td>CVTGAD</td><td>76.10±0.00</td><td>21.15±0.03</td><td>23.30±0.01</td><td>67.41±0.00</td><td>94.71±0.00</td><td>48.36±0.02</td><td>82.58±0.00</td><td>30.82±0.00</td><td>42.19±0.49</td><td>O0M</td></tr><tr><td>SIGNET</td><td>95.64±0.00</td><td>13.55±0.02</td><td>25.13±0.02</td><td>67.25±0.01</td><td>82.78±0.03</td><td>51.26±0.19</td><td>81.72±0.00</td><td>40.01±0.00</td><td>52.33±0.01</td><td>52.30±0.00</td></tr><tr><td>MUSE</td><td>92.67±0.44</td><td>24.30±0.59</td><td>20.95±0.30</td><td>67.87±0.72</td><td>86.26±1.96</td><td>46.07±0.93</td><td>84.65±0.16</td><td>39.38±0.37</td><td>50.16±1.06</td><td>63.30±0.59</td></tr><tr><td>GLADPro</td><td>80.00±0.00</td><td>20.76±0.23</td><td>24.14±0.73</td><td>64.35±2.85</td><td>66.49±0.00</td><td>46.49±2.60</td><td>84.34±0.22</td><td>42.18±2.96</td><td>45.47±1.02</td><td>63.01±7.06</td></tr><tr><td>HC-GLAD</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>FANFOLD</td><td>91.63±0.00</td><td>21.09±0.01</td><td>16.70±0.01</td><td>69.28±0.00</td><td>94.52±0.01</td><td>43.96±0.01</td><td>82.90±0.00</td><td>41.71±0.01</td><td>56.12±0.01</td><td>34.01±0.00</td></tr><tr><td colspan="9">Few-shot Fine-tuning Only</td></tr><tr><td>GLocalKD</td><td>85.24±1.05</td><td>26.83±3.24</td><td>23.23±0.97</td><td>77.15±0.62</td><td>94.93±0.24</td><td>43.21±0.81</td><td>81.54±0.86</td><td>43.71±3.22</td><td>55.69±1.28</td><td>62.14±1.40</td></tr><tr><td>CVTGAD</td><td>89.90±3.71</td><td>21.82±6.16</td><td>20.77±3.09</td><td>58.70±3.62</td><td>83.76±8.41</td><td>41.03±2.03</td><td>86.22±1.74</td><td>32.98±1.35</td><td>53.45±7.46</td><td>OOM</td></tr><tr><td>SIGNET</td><td>86.33±2.17</td><td>18.94±3.97</td><td>18.45±1.54</td><td>52.52±2.12</td><td>74.93±11.66</td><td>45.08±0.62</td><td>79.56±0.61</td><td>48.12±3.36</td><td>51.78±3.49</td><td>48.22±2.48</td></tr><tr><td>MUSE</td><td>95.23±1.85</td><td>19.21±0.39</td><td>18.19±0.59</td><td>76.14±1.54</td><td>89.45±6.26</td><td>48.44±1.31</td><td>84.10±0.25</td><td>41.96±0.64</td><td>53.17±2.05</td><td>68.27±1.44</td></tr><tr><td>GLADPro</td><td>86.57±7.03</td><td>28.85±7.18</td><td>27.28±4.40</td><td>55.70±3.70</td><td>88.42±8.01</td><td>51.52±1.72</td><td>85.19±0.45</td><td>43.32±5.06</td><td>44.64±2.32</td><td>68.24±3.85</td></tr><tr><td>HC-GLAD</td><td>60.55±0.13</td><td>18.04±2.13</td><td>20.33±0.95</td><td>65.31±1.67</td><td>92.86±1.09</td><td>43.97±1.27</td><td>90.35±1.61</td><td>32.97±3.45</td><td>38.59±4.51</td><td>OOM</td></tr><tr><td>FANFOLD</td><td>85.76±7.21</td><td>18.58±0.76</td><td>17.90±1.26</td><td>71.24±3.06</td><td>85.06±13.52</td><td>46.54±1.80</td><td>84.58±1.49</td><td>51.46±4.15</td><td>52.80±2.68</td><td>35.58±1.00</td></tr><tr><td colspan="10">Generalist</td></tr><tr><td>GenGLAD</td><td>98.79±0.01</td><td>38.87±3.46</td><td>31.64±1.50</td><td>82.21±0.10</td><td>95.06±0.13</td><td>56.05±0.56</td><td>85.47±0.71</td><td>69.26±0.11</td><td>74.72±0.64</td><td>74.77±0.76</td></tr></table>

<table><tr><td>Variant</td><td>AIDS</td><td>BZR</td><td>COX2 62.92±0.30</td><td>DHFR 74.96±0.42</td><td>MUTAG</td><td>PTC_MR</td><td>ENZYMES</td><td>PROTEINS 73.63±0.16</td><td>IMDB-B</td><td>REDDIT-B</td></tr><tr><td colspan="9">95.84±0.07</td><td>74.11±0.25 73.63±0.20</td></tr><tr><td>w/o Lnode</td><td></td><td></td><td></td><td>70.98±0.94</td><td>Pre-training</td><td></td><td></td><td>66.03±0.19</td><td>58.03±0.95</td><td>62.77±0.32</td></tr><tr><td colspan="9">91.43±0.73</td></tr><tr><td>w/o ||m||1</td><td>93.82±0.32</td><td>57.80±0.73 59.74±1.37</td><td>57.32±0.95 58.20±0.17</td><td>77.98±0.32 71.30±0.34 86.32±0.43</td><td>58.20±1.69 56.68±0.19</td><td></td><td>54.47±0.60 53.70±0.29</td><td>68.53±0.37</td><td>60.06±2.43</td><td>67.07±0.33</td></tr><tr><td colspan="9">Discrepancy Measures</td></tr><tr><td>w/o Prior Disc.</td><td>93.75±0.16</td><td>60.82±3.90</td><td>59.03±1.49</td><td>73.05±0.49</td><td>88.00±0.59</td><td>57.31±1.66</td><td>52.54±1.19</td><td>71.66±0.37</td><td>71.15±0.99</td><td>71.38±0.43</td></tr><tr><td>w/o Domain Disc.</td><td>93.31±3.41</td><td>57.82±4.22</td><td>46.87±6.66</td><td>73.77±0.52</td><td>88.58±0.36</td><td>46.65±0.67</td><td>47.58±2.00</td><td>63.76±1.57</td><td>51.65±2.44</td><td>65.93±1.78</td></tr><tr><td>w/o Cls. Disc. w/o Inst. Disc.</td><td>92.60±3.58</td><td>52.41±3.05 56.62±2.39</td><td>59.28±0.96 59.03±1.80</td><td>45.63±1.53 74.25±0.51</td><td>87.07±4.98 89.38±2.21</td><td>55.56±0.55 58.84±0.74</td><td>53.58±0.64 53.75±0.80</td><td>60.20±1.48 73.75±0.14</td><td>63.72±2.81 72.96±0.99</td><td>61.92±1.45 72.16±1.69</td></tr><tr><td colspan="9">95.77±0.13</td></tr><tr><td></td><td>6.15±0.06</td><td>49.06±0.33</td><td></td><td></td><td>Scoring Module 13.34±0.72</td><td>46.90±0.22</td><td>46.29±0.21</td><td>41.54±0.10</td><td>42.00±0.17</td><td>50.32±0.06</td></tr><tr><td>MEAN MAX</td><td>27.63±0.18</td><td>33.71±0.12</td><td>56.86±1.27 57.56±0.01</td><td>28.92±0.69 40.32±0.05</td><td>56.57±0.23</td><td>53.89±0.09</td><td>44.40±0.08</td><td>61.94±0.25</td><td>30.12±0.04</td><td>67.78±0.18</td></tr><tr><td>MIN</td><td>8.44±0.25</td><td>47.02±0.52</td><td>58.34±1.84</td><td>29.56±0.87</td><td>16.36±1.26</td><td>45.19±0.41</td><td>46.90±0.53</td><td>38.05±0.24</td><td>43.57±0.24</td><td>45.77±0.25</td></tr></table>

Hardware Configuration. All the experiments are conducted on a Windows desktop equipped with an AMD Ryzen 5800X processor, 32 GB of RAM, and a single NVIDIA GeForce RTX4090 GPU with 24GB of VRAM.

## H Additional Experimental Results

## H.1 Main Results in AUPRC

Table 5 reports the anomaly detection performance in terms of AUPRC. Overall, GenGLAD achieves consistently strong performance across diverse datasets and demonstrates clear advantages in the generalist GLAD setting. Specifically, GenGLAD obtains the best performance on 9 out of 10 datasets, while remaining highly competitive on the remaining datasets. Notably, substantial improvements can be observed on datasets such as DHFR, IMDB-B, and REDDIT-B, indicating the effectiveness of GenGLAD in capturing transferable anomaly-related patterns across domains. In contrast, conventional GLAD baselines generally exhibit limited gains from pre-training and often suffer noticeable performance degradation under the few-shot fine-tuning setting. This trend is particularly evident on datasets with limited training samples, where several baselines achieve relatively low AUPRC scores or even unstable performance. These observations further validate that existing GLAD approaches heavily rely on sufficient in-domain training data and struggle to generalize across domains. The strong AUPRC performance of GenGLAD demonstrates that the proposed discriminative subgraph extraction and discrepancy-based adaptation strategy can effectively identify anomalous graphs while maintaining robust cross-domain generalization under limited supervision.

## H.2 More Results of Sensitivity to #shot

Table 7: AUPRC of GenGLAD and its variants.
<table><tr><td>Variant</td><td>AIDS</td><td>BZR</td><td>COX2</td><td>DHFR</td><td>MUTAG</td><td>PTC_MR</td><td>ENZYMES</td><td>PROTEINS</td><td>IMDB-B</td><td>REDDIT-B</td></tr><tr><td colspan="9">98.79±0.01 38.87±3.46</td></tr><tr><td></td><td></td><td></td><td>31.64±1.50</td><td>82.21±0.10 95.06±0.13</td><td>Pre-training</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">97.76±0.12 30.72±0.14</td></tr><tr><td>w/o ||m||1</td><td>98.35±0.14</td><td>37.22±7.62</td><td>25.73±0.33 25.87±0.34</td><td>80.16±1.33 87.61±1.58 80.18±0.15 92.94±0.13</td><td>55.29±1.44 54.66±0.11</td><td></td><td>86.69±1.58 83.76±0.75</td><td>63.80±0.48 65.35±0.97</td><td>59.63±0.59 55.08±1.31</td><td>67.24±1.74  $6 9 . 3 5 { \scriptstyle \pm 1 . 7 4 } _ { }$ </td></tr><tr><td colspan="9">Discrepancy Measures</td></tr><tr><td colspan="9"></td></tr><tr><td>w/o Prior Disc. w/o Domain Disc.</td><td>97.79±0.05</td><td>31.92±5.12</td><td>27.32±1.23 80.18±0.42</td><td>93.77±0.35</td><td>54.67±1.42</td><td>82.44±0.60</td><td>68.04±0.25</td><td>71.85±1.11</td><td>72.28±0.92</td></tr><tr><td>w/o Cls. Disc.</td><td>97.11±1.06</td><td>30.19±8.18</td><td>20.34±3.63 81.21±0.35</td><td> $9 4 . 3 0 { \scriptstyle \pm 0 . 1 7 }$ </td><td>43.49±0.25</td><td>81.74±1.89</td><td> $5 9 . 7 5 { \scriptstyle \pm 1 . 2 7 }$ </td><td>52.02±2.92</td><td>70.22±1.95</td></tr><tr><td>w/o Inst. Disc.</td><td>96.60±1.17 96.78±0.05</td><td>21.32±2.38 23.59±1.52</td><td>25.77±0.47 63.11±1.57 26.43±1.31 81.78±0.11</td><td> $9 2 . 7 1 { \scriptstyle \pm 3 . 4 1 }$  93.91±1.56</td><td>53.54±0.27 55.22±0.47</td><td>84.19±0.30 84.60±0.49</td><td>54.69±1.92 69.18±0.08</td><td>62.62±2.21 73.25±1.41</td><td>60.04±1.35 73.69±1.38</td></tr><tr><td colspan="10">Scoring Module</td></tr><tr><td colspan="10">48.40±0.47 59.26±0.11</td></tr><tr><td>MEAN MAX</td><td>60.97±0.02 74.09±0.06</td><td>19.30±0.18</td><td>24.81±0.94</td><td>48.75±0.28</td><td>43.97±0.65 49.46±0.08</td><td>80.82±0.19 79.82±0.06</td><td>35.05±0.06</td><td>43.70±0.09 37.89±0.02</td><td>51.79±0.11</td></tr><tr><td>MIN</td><td>61.64±0.08</td><td>15.92±0.02 18.63±0.09</td><td>24.38±0.01 26.41±2.57</td><td>81.51±0.16 49.04±0.75 49.45±0.55</td><td>42.87±0.78</td><td>81.24±0.22</td><td>49.20±0.55 33.96±0.35</td><td>44.73±0.12</td><td>60.52±0.11 49.42±0.24</td></tr></table>

Figure 6 presents the performance of GenGLAD with varying numbers of few-shot samples for adaptation on individual datasets, while Figure 5 reports the averaged results on science graph datasets. Overall, GenGLAD demonstrates stable adaptation behavior under different few-shot settings. In most cases, increasing the number of fine-tuning graphs leads to improved AUROC and AUPRC performance, indicating that the proposed adaptation framework can effectively exploit additional normal samples when available.

![](images/1d1f80d7a6c483d1093392ec3901e69edd6ecd27be1d8577dc624d3c2d9a7153.jpg)  
Figure 5: Sensitivity of #shot (Science groups).

Moreover, even under extremely limited supervision, GenGLAD still achieves competitive performance on most datasets, demonstrating the strong transferability of the pre-trained backbone model. We also observe that the sensitivity to the number of few-shot samples varies across datasets. For example, datasets such as IMDB-B and DHFR exhibit relatively consistent performance improvements as more adaptation samples are introduced, while datasets such as BZR and MUTAG show larger fluctuations due to their relatively small scales and higher structural diversity. Nevertheless, the overall trends suggest that GenGLAD maintains robust adaptation capability across diverse domains and does not heavily rely on large amounts of target-domain data.

## H.3 Full Results of Ablation Study

Full Ablation Results. Tables 6 and 7 present the complete ablation results of GenGLAD in terms of AUROC and AUPRC, respectively. Overall, the observations are consistent with those reported in the main paper, further validating the effectiveness of each component in GenGLAD.

For the pre-training stage, removing either the node-level pseudo anomaly supervision $\mathcal { L } _ { \mathrm { n o d e } }$ or the sparsity regularization on m consistently degrades performance on most datasets. In particular, removing ${ \mathcal { L } } _ { \mathrm { n o d e } }$ leads to more noticeable performance drops, demonstrating the importance of finegrained pseudo anomaly supervision for learning transferable discriminative substructures.

For the adaptation stage, all discrepancy measures contribute positively to the final performance. Among them, the domain discrepancy and class discrepancy generally have the largest impact. Their removal causes substantial degradation on several datasets and even results in near-random performance in certain cases, highlighting the importance of capturing both domain-aware and class-level abnormality evidence. In comparison, the prior discrepancy and instance discrepancy provide complementary gains by refining the anomaly characterization from different perspectives.

We further evaluate the effectiveness of the scoring module by replacing it with simple aggregation strategies, including mean, max, and min pooling. As shown in the tables, these heuristic strategies consistently underperform the learnable scoring module and often lead to unstable results across datasets. In some cases, they even produce performance close to or below random guessing. These observations indicate that different discrepancy signals contribute unequally across datasets, and adaptive weighting is crucial for reliable anomaly detection under few-shot adaptation settings.

![](images/6ccae5e3f63d50460760e6c3a34d22623c53a4c4680196e4d3dcb7cf3953e960.jpg)  
(a) AIDS

![](images/640086cb1104cf7cfa85f38a7e632616812949f2c259685e3552786f3373ae4d.jpg)  
(b) BZR

![](images/19d938a89929dc7135980d51653ab887dca8c0cafbba0e99944c9301aff7c32e.jpg)  
(c) COX2

![](images/11fce696624c9234d63e8efae2a5f7a93f64ac1d56d71d8118ae3e4c709d79eb.jpg)  
(d) DHFR

![](images/121911d682115becc09a0ca20bf87a88d2af5d0e585ab53e4c9f2a908b649e40.jpg)  
(e) ENZYMES

![](images/67ca8038773db42a5e030c27fdfe218736da3a046ba0646f08c4acf288ae817d.jpg)  
(f) IMDB-B

![](images/6f677227f5e0fd623132f0e2c58444d757b28a48e18e8ce62ab7672cacc135b4.jpg)  
(g) MUTAG

![](images/a7c867721fdbfe569d607864c2471263d6d33f152187896b24ab570a5300f3ee.jpg)  
(h) PROTEINS

![](images/fd10165e9b53ad905ede889f2b94612f848bf78fffe5ed47bc69885ad4fce19d.jpg)  
(i) PTC\_MR

![](images/f1e658519a315837119fe39b8a84dd7456029ed71586f57102a109fc2ae2aec8.jpg)  
(j) REDDIT-B

Figure 6: Performance with varying numbers of few-shot samples for adaptation.  
![](images/31a3789f828d776fd19c2c1d6a76ff5e4b8d005d45ccd85af0a729ef947d0b55.jpg)  
Figure 7: Visualization results of memory token importance.

## H.4 Visualization of Memory Retrieval Weight

As illustrated in Figure 7, token activations vary across application domains. With the help of context-guided memory retrieval, GenGLAD can selectively leverage the most relevant pre-trained knowledge for GLAD, thereby improving generalization.

## H.5 More Analysis on Efficiency

We further provide a detailed runtime breakdown in Table 8. As the results show, synthesizing the pseudo-graphs requires even less time than a single training epoch, demonstrating the negligible computational overhead of our graph synthesis procedure. Notably, the pretraining dataset, consisting of NCI1, DD, and COLLAB, contains 10,288 graphs, 830,144 nodes, and 26,858,668 edges in total. The reported runtime and memory consumption on this relatively large collection further demonstrate the practical efficiency and scalability of our method.

## H.6 Quantitative Evaluation of Discriminative Subgraph

We further provide a quantitative evaluation for the extracted discriminative subgraph on BM-MN and BM-MT. We compare GenGLAD with SIGNET, which is trained under the conventional indomain setting. As shown in Table 9, despite being pretrained for cross-dataset generalization rather than optimized separately on each target dataset (like SIGNET), GenGLAD achieves interpretation performance close to SIGNET, which is trained on 500 normal graphs on BM-MN and BM-MT. Overall, these results suggest that the pretrained discriminative-subgraph extractor can identify meaningful explanatory structures across diverse anomaly patterns.

Table 8: Runtime breakdown of GenGLAD in second. Pretraining is conducted on NCI1, DD, and COLLAB, while fine-tuning is evaluated on AIDS.
<table><tr><td>Process</td><td>Time (s)</td></tr><tr><td>Pretraining: pseudo-anomaly synthesis (motif pool P)</td><td>0.85</td></tr><tr><td>Pretraining: per epoch</td><td>10.86</td></tr><tr><td>Pretraining: total</td><td>325.93</td></tr><tr><td>Fine-tuning: pseudo-anomaly synthesis (pool S)</td><td>0.81</td></tr><tr><td>Fine-tuning: per epoch</td><td>0.04</td></tr><tr><td>Fine-tuning: total</td><td>0.42</td></tr></table>

Table 9: Quantitative Evaluation of Discriminative Subgraph. Results in AUROC (%).
<table><tr><td>Method</td><td>BM-MN</td><td>BM-MT</td></tr><tr><td>SIGNET</td><td>76.6</td><td>78.4</td></tr><tr><td>GenGLAD</td><td>72.1</td><td>76.7</td></tr></table>

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

• You should answer [Yes], [No], or [N/A].

• [N/A] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.

• Please provide a short (1–2 sentence) justification right after your answer (even for [N/A]).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will also be asked to include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While [Yes] is generally preferable to [No], it is perfectly acceptable to answer [No] provided a proper justification is given (e.g., error bars are not reported because it would be too computationally expensive” or “we were unable to find the license for the dataset we used”). In general, answering [No] or [N/A] is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

• Delete this instruction block, but keep the section heading “NeurIPS Paper Checklist",

• Keep the checklist subsection headings, questions/answers and guidelines below.

• Do not modify the questions and only use the provided macros for your answers.

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: Our paper aims to introduce and address generalist GLAD, which is consistent with what we state in the abstract and introduction.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We mentioned limitations in the conclusion.

## Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: Does not include theoretical results.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We report all the hyperparameters in the appendix. The detailed algorithm can also be find in the appendix.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in the supplemental material?

Answer: [Yes]

Justification: We use publicly available datasets, and the source code is provided.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: We provide all hyperparameters together with range , and details of pre-training and fine-tuning dataset construction in the appendix.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Standard deviation error bar is provided.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We include the hardware configuration in the appendix.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: We confirm compliance with the Code of Ethics.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: In the introduction, we highlight that generalist GLAD helps to safeguard diverse graph applications, including new domains with limited in-domain data.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper poses no such risk.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We include citations for all used datasets and baselines we compared.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: We provide documentation for the code in READEME.md.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

## Answer: [N/A]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: LLM is only used for grammar and word choice.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.