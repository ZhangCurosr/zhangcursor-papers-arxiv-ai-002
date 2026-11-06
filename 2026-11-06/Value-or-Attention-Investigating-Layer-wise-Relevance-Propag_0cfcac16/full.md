# Value or Attention: Investigating Layer-wise Relevance Propagation for Graph Attention Networks

Ken Schreiber<sup>1</sup> René P. Klausen<sup>1</sup> Sebastian Lapuschkin<sup>1,2</sup> Wojciech Samek<sup>1,3,4</sup> Jonas Naujoks<sup>1,†</sup>

<sup>1</sup>Department of Artificial Intelligence, Fraunhofer Heinrich Hertz Institute, Berlin, Germany <sup>2</sup>Centre of eXplainable Artificial Intelligence, Technological University Dublin, Ireland <sup>3</sup>Department of Electrical Engineering, Technische Universität Berlin, Germany <sup>4</sup>BIFOLD – Berlin Institute for the Foundations of Learning and Data, Berlin, Germany

<sup>†</sup>Corresponding author: jonas.naujoks@hhi.fraunhofer.de

## Abstract

Graph Attention Networks (GATs) are increasingly used in domains where scientists need to read a model’s reasoning, from medicine to materials science, yet the routing they learn remains opaque to most explainers. Layer-wise Relevance Propagation (LRP) follows the model’s computation, but in an attention layer the relevance reaching a node is the sum of two mechanically different quantities: its content relevance through the value path and its routing relevance through the attention path. Existing rules either drop one or add them into a single number. We write a single product rule with free weights on the two factors that recovers current approaches as special cases. The value path measures what content a node supplied against a zero baseline; the attention path measures how the node shaped what was passed relative to its neighborhood, and its sign records whether the node’s features moved the gating the way the prediction demanded. On a ground-truth classification task, we show that the same node receives negative, positive and near-zero relevance under three settings of the rule. The separated relevance maps yield a hypothesis about the model: it gains by not reading that node. We give reading rules for both paths and argue that they should be interpreted separately.

## 1 Introduction

From particles to planets, social networks to logical formulas, many scientific relations can naturally be described as graphs. Graph Neural Networks (GNNs) [24] learn on such graphs directly by passing messages between neighboring nodes. How a node aggregates its neighbors’ messages is their central design choice: Graph Convolutional Networks (GCNs) [14] use coefficients fixed by the graph structure, while Graph Attention Networks (GATs) allow input-dependent routing [7, 27]. GATs are deployed in high-stakes domains such as biomedical systems [6, 12, 19], electronic health records [18, 21], drug discovery [16], and materials science [25], where inspectability of a model’s reasoning is needed for trust and scientific insight. In these settings, an explanation proposes a hypothesis about which nodes mattered and why, which requires that the importance assigned to a node have a single, stateable meaning.

For GNNs, most explainability approaches are perturbation- or surrogate-based [13, 15, 28, 29, 31]: they score a node by the effect of intervening on it. In an attention layer, however, a node influences the prediction through two mechanisms: the message it sends and the attention weights it helps compute. We refer to these as the value path and the attention path. Any intervention at the input moves both, and black-box methods thus cannot distinguish the two mechanisms. Layer-wise Relevance Propagation (LRP) [5], and its graph extension GNN-LRP [26], instead decompose the prediction along the model’s computation and can attribute to each mechanism through the model’s internals. There is no canonical way to split the relevance across the two paths. The question has been studied in the context of Transformers: CP-LRP [2] keeps only the value path, while AttnLRP [1] distributes evenly across both paths. These rules trade places across metrics and model families [4, 30]. Neither rule considers the two paths separately, which only matters if they carry different information.

We show that they do. By writing the relevance through one GATv2 layer as a product rule with free weights on the two factors, we find that the two paths answer different questions: the value path says how much each node’s content contributed; the attention path, how each node shaped which content was routed. Existing rules are special cases of this formulation. Dropping either path omits information, while collapsing them into a single attribution entangles distinct stories. On a graph-classification task with known ground truth BA-HOUSE [15], the same node receives a positive, negative or near-zero relevance depending on the setting of the rule. We therefore recommend that the two paths be interpreted individually. We show how this can lead to deeper mechanistic understanding, which we argue increases trust and opens up potential for scientific discovery.

## 2 Background and related work

Graph Attention Networks We study GATv2 [7], the dynamic-attention variant of GAT. A GATv2 layer updates node i from its neighborhood $\mathcal { N } ( i )$ by message passing

$$
y _ { i } = \sum _ { j \in \mathcal { N } ( i ) } \alpha _ { i j } m _ { j } ,\tag{1}
$$

where messages from all source nodes $j$ are aggregated over a single destination i. The attention parameter $\alpha _ { i j }$ and message $m _ { j }$ are calculated as

$$
\begin{array} { r } { m _ { j } = W _ { l } x _ { j } , \qquad s _ { i j } = a ^ { \top } \sigma ( W _ { l } x _ { j } + W _ { r } x _ { i } + W _ { e } e _ { i j } ) , \qquad \alpha _ { i j } = \mathrm { s o f t m a x } _ { j } ( s _ { i j } ) . } \end{array}\tag{2}
$$

where $\sigma$ is the LeakyReLU, $e _ { i j }$ optional edge attributes, $W _ { l } , W _ { r } , W _ { e }$ learned source-, destinationand edge-projections, a a learned score vector, and $x _ { j } , x _ { i }$ source- and destination-node features from the layer input x. A message passing step differs from a Transformer attention layer in two ways: (i) the embeddings: $W _ { l } x _ { j }$ is both the message and the source-side input to $s _ { i j }$ , and optional edge attributes $e _ { i j }$ enter only the attention path; (ii) the logit $s _ { i j }$ calculation, which mimics a feed-forward layer compared to the Transformer’s normalized matrix multiplication $s _ { i j } = q _ { i } ^ { \top } k _ { j } / \sqrt { d _ { k } }$

Layer-wise Relevance Propagation (LRP) LRP [5] is a well-established and efficient explanation method settled between gradient- and decomposition-based attribution: originally motivated as a rule-based, layer-by-layer redistribution under a conservation principle, modern rule formulations often follow a Deep Taylor Decomposition motive [17] and can be implemented as a modified Gradient×Input (G×I) pass [2, 4]. Because LRP rules rest on local linearizations, a central objective of LRP research is deriving faithful rules for the non-linear operations that violate this assumption.

Related work As a special extension for GNNs, a higher-order method was developed that nests first-order explanations into relevant walks [26]. However, they build on the first-order rules, where GAT message passing faces the same non-linearities that are unsettled for Transformer attention: the bilinear product and the softmax function. Ali et al. [2] motivate CP-LRP by showing that G×I breaks conservation at the softmax whenever its logits and the inputs covary; treating the attention weights as constants removes that term. Achtibat et al. [1] maintain that discarding how the routing was produced costs faithfulness. Their rule, motivated by Deep Taylor Decomposition and Shapley decomposition at a zero baseline, preserves conservation at the bilinear product via uniform relevance split and accepts the softmax as a legitimate relevance sink. In the recent literature, both rules achieve mixed results across model families and metrics [4, 30] — the cause is stated as an open question [4].

## 3 Methodology

We suggest that these results reflect a genuine difference in what the two rules measure, not necessarily a shortcoming of either. We argue that deep understanding of how both relevance paths function

individually will lead to mechanistic understanding beyond what’s possible with a single reading. This consequently gives insights into the different questions that CP-LRP and AttnLRP answer. We make this separate reading precise for GAT message passing.

## 3.1 The two paths gradient rule

For a single message passing step (Eqs. (1) and (2)), we write the modified gradient of an output feature $y _ { i }$ with regard to the input vector x with an adapted product rule with a set of free weights $\mathbf { w } = ( w _ { \mathrm { v a l } } , w _ { \mathrm { a t t } } )$ , so that the existing methods appear as special cases:

$$
\begin{array} { r } { \left. \frac { \partial y _ { i } } { \partial x } \right| _ { \mathbf { w } } = \displaystyle \sum _ { j \in \mathcal { N } ( i ) } \Big [ \underbrace { w _ { \mathrm { v a l } } \alpha _ { i j } \frac { \partial m _ { j } } { \partial x } } _ { \mathrm { v a l u e \ p a t h } } + \underbrace { w _ { \mathrm { a t t } } m _ { j } \frac { \partial \alpha _ { i j } } { \partial x } } _ { \mathrm { a t t e n t i o n \ p a t h } } \Big ] . } \end{array}\tag{3}
$$

We recover the formulations for G×I at $\mathbf { w } = ( 1 , 1 )$ , the CP-LRP rule at $\boldsymbol { \mathsf { v } } = ( 1 , 0 )$ , the AttnLRP rule at $\begin{array} { r } { { \bf w } = ( \frac { 1 } { 2 } , \frac { 1 } { 2 } ) } \end{array}$ , and inspect the attention path individually at $\mathbf { w } = ( 0 , 1 )$ . In deeper networks the choice recurs at each layer and rules face the question of the downstream implications of relevance aggregation between the paths. A caveat to the distinctiveness of the two paths is that in GATv2 the two paths share a derivative, which is further discussed in App. A.

## 3.2 Two types of object

Let $g _ { i }$ be the incoming gradient signal at destination node i for the explained scalar output. We recover the value and attention path relevances by multiplying with the activation. For the full edge-level algebra, we refer to App. B.

The value path as a content share A $\mathbf { t } \ \mathbf { w } = ( 1 , 0 )$ only the value term of Eq. (3) remains, and the edge relevance is $r _ { i j } ^ { \mathrm { v a l } } = \alpha _ { i j } g _ { i } ^ { \top } m _ { j }$ . Summing over the destinations that $j$ sends to gives $\begin{array} { r } { R _ { j } ^ { \mathrm { v a l } } = \sum _ { i : j \in \mathcal { N } ( i ) } r _ { i j } ^ { \mathrm { v a l } } } \end{array}$ . With fixed routing, this is a linear function with factors $\alpha _ { i j } \geq 0$ . Layerlocally, at fixed $g _ { i }$ , the ablation effect is exact: zeroing node $j ^ { \circ } \mathbf { s }$ message removes exactly $r _ { i j } ^ { \mathrm { v a l } }$ from the relevance $g _ { i } ^ { \textsf { T } } y _ { i }$ that destination i received, and exactly $R _ { j } ^ { \mathrm { v a l } }$ across all of $j ^ { \prime } { \bf s }$ outgoing edges (App. B.2).

The attention path as a routing influence $\mathrm { A t } { \mathbf { w } } = ( 0 , 1 )$ the attention path is isolated. Differentiating through the softmax leaves the relevance on the logit $s _ { i j }$

$$
r _ { i j } ^ { \mathrm { a t t } } = r _ { i j } ^ { \mathrm { v a l } } - \alpha _ { i j } \sum _ { k \in \mathcal { N } ( i ) } r _ { i k } ^ { \mathrm { v a l } } , \qquad \sum _ { j \in \mathcal { N } ( i ) } r _ { i j } ^ { \mathrm { a t t } } = 0 ,\tag{4}
$$

which is the value path relevance but subtracts from each edge its share of the neighborhood total: it measures the effect of re-routing relevance among neighbors. Withholding attention from one neighbor necessarily grants it to the others, and relevance can only exist in relation (App. B).

Reaching an input node takes one further step. The logit is computed from the source’s features $x _ { j }$ and the destination’s features $x _ { i } , \mathbf { A }$ node n therefore collects relevance in two roles,

$$
R _ { n } ^ { \mathrm { a t t } } = \sum _ { { \underbrace { i : n \in { \cal N } ( i ) } } _ { n { \mathrm { ~ a s ~ s o u r c e } } } } r _ { i n } ^ { \mathrm { a t t } } u _ { i n } + \sum _ { { \underbrace { j \in { \cal N } ( n ) } _ { n { \mathrm { ~ a s ~ d e s t i n a t i o n } } } } } r _ { n j } ^ { \mathrm { a t t } } v _ { n j } , \qquad v _ { i n } = x _ { n } ^ { \top } W _ { l } ^ { \top } p _ { i n } ,\tag{5}
$$

with $p _ { i j } = \sigma ^ { \prime } ( W _ { l } x _ { j } + W _ { r } x _ { i } + W _ { e } e _ { i j } )$ ⊙ a the local linearization of the scorer (App. B.4). Both factors in each product are signed and meet at the logit from opposite directions: $r _ { i j } ^ { \mathrm { a t } \bar { \mathrm { t } } }$ is a demand from above — which way the target wants this edge scored — while $u _ { i n }$ and $v _ { n j }$ are supplies from below, how much of that logit $n \mathrm { { : } }$ own features produced. The product is positive when demand and supply agree: the edge relevance $r _ { i j } ^ { \mathrm { a t t } }$ says score this edge higher and n’s features push the logit up, or it says score it lower and they push it down.

The roles differ in what survives aggregation. The destination role sum ranges over $\mathcal { N } ( n )$ , the same set on which the attention parameters are normed, but it does not inherit the edge relevances’ zero-sum (Eq. (4)) property due to the non-linear LeakyReLU. It asks how $n \mathrm { { : } }$ features affected the way it distributed attention to its neighbors within a constrained attention fund; its magnitude is set by how much those features separate one neighbor from another. The source role sum ranges over multiple constrained destination neighborhoods and is itself unconstrained. It measures how n’s features affected the way it was read by its neighbors; its magnitude is set by how much the neighborhoods that read n differ in what they demand of it.

![](images/55fd2330cffd740d54f783aeb19a847467a7a65bc182a6c117e6841d72dfe04f.jpg)  
Figure 1: Node relevance on one BA-HOUSE [15] graph, red positive, blue negative.

Consequences for interpreting attributions Value path and attention path attributions read differently. A value path content share is measured against a 0 baseline, it vanishes when a node contributes nothing. The sign means: j’s content moved the target this way. Positive attributions have a clear reading: my content helped the target by this much, where helping means pushing the class logit up. CP-LRP will assign this relevance, but miss all information from the attention calculation. An attention path routing influence has no fixed reference, its magnitude depends on contrast within the neighborhood, it vanishes when all content matters the same, regardless of what they contribute. Its sign measures a demand-supply agreement: in the source role a positive sign means skip me, and skipping me helps or read me, and reading me helps. For a flowchart of all interpretations find App. C. AttnLRP will combine the information of both paths, but content helped and steering helped become a single number with overlapping magnitude meanings and sign readings.

## 4 Experiments

Section 3 argues algebraically that the two paths answer different questions. In order to make these readings tangible, we present a case study as a proof of concept. It centers on a demonstrative instance where each path tells a coherent, yet distinct story about the same prediction on a singular node. For further experiments on a toy dataset and a real-world motivation, we refer to App. E. For the case study, we evaluate the value path (CP-LRP), the attention path, and AttnLRP $( \mathbf { \bar { w } } \in \{ ( 1 , 0 ) , ( 0 , 1 )$ $\bigl ( { \textstyle { \frac { 1 } { 2 } } } , { \textstyle { \frac { 1 } { 2 } } } \bigr ) \bigr \}$ in Eq. (3)) at a single GATv2 layer on BA-HOUSE, a graph-classification special case of the synthetic BA-2Motifs benchmark [15], chosen for its interpretable ground truth. A positive graph contains a house motif consisting of two floor nodes, two eave nodes, and one rooftop node (Fig. 1). The attachment node becomes the anchor: a floor or rooftop anchor reaches degree three, an eave anchor reaches degree four. The node embeddings are their one-hot encoded degree.

Fig. 1 shows the three attribution maps on one graph with a degree-four anchor node 16 and highlights disagreement on the same node: the value path assigns negative and the attention path positive relevance, the sum returns it to near zero. On the base-graph hubs (0, 1, 2, 3, 5) the two paths agree in sign. The trend between the two paths attributions can be similar or opposite, depending on the nodes and neighborhoods, and a combined reading entangles this information.

Fig. 2 shows that this is not an isolated case. 100% of evaluated degree-four nodes receive negative value path relevance (max −0.031), which, by Section 3.2, reads as a negative contribution to the class logit. Read as content alone, the negative attribution is reasonable, because house nodes are naturally degree two or degree three, the only case a degree-four node can belong to a house is as an anchor node at the eave: only 18.2% of degree-four nodes belong to a house (41.6% for degree-three and 27.9% for degree-two). The attention path attributes them positive relevance and lifts degree-four anchors’ relevance substantially $( \Delta z = + 1 . 8 8 )$ relative to their main graph counterparts. This signal sits in the source role: across the evaluation set the source term carries 93.4% of $| R ^ { \mathrm { a t t } } |$ at degree-four anchor nodes. Following Section 3.2 routing influence logic, the positive relevance means the skip me, and skipping me helps case: For node 16 (Fig. 1), 100% of its 5 sender edges have both demand $r ^ { \mathrm { a t t } } < 0$ and supply $u _ { i n } < 0 \left( \mathrm { E q . } \left( 5 \right) \right)$ .

![](images/1d1497c0d9c77d7fa2014099e0ae20269d1a7401ac746370f72b0af61e7ad273.jpg)  
degree three anchor ΔR +0.22 Δz +0.14 degree four anchor ΔR −0.20 Δz −0.13

![](images/d98471e5f801537c3b3e7617be17f740446daad1b5164354a1d9a0f5fe2e77fa.jpg)  
degree three anchor ΔR +0.17 Δz +0.33 degree four anchor ΔR +0.96 Δz +1.88

![](images/0915c9ef86a43f890e438ec70e39d7cfd47ac7706601e9107840f407889b7c08.jpg)  
degree three anchor ΔR +0.19 Δz +0.20 degree four anchor ΔR +0.38 Δz +0.40  
Figure 2: Node relevance for the house logit against degree, all nodes of 600 positive graphs across 5 initializations. The 4.6% of nodes with degree higher than 7 are omitted here for clarity; Fig. 6 shows the full range. Grey: ordinary (non-anchor) nodes, from both the BA graph and the house; black bars: their per-degree mean. Blue / orange: anchor nodes of degree three / four; dark blue / dark orange bars: their means. ∆R: mean anchor relevance minus mean ordinary-node relevance at the same degree. $\bar { \Delta z } \colon \Delta R$ divided by the map’s standard deviation over all nodes, including those not drawn, which makes it independent of each method’s overall scale (per initialization, then averaged).

Combining both maps flattens the separation to $\Delta z = + 0 . 4 0 ~ \mathrm { ( F i g . ~ } 2$ , right) and node 16 to near zero (Fig. 1, right). This is not “wrong”: in this example, one could even argue that the combined path explanation finds the best house representation. What is lost is the disentanglement between interpretations. This content helped and this steering helped (Section 3.2) get reported as a single value, and a small entry no longer distinguishes a node that neither contributes value nor routing from a node whose two roles point in opposite directions with high influence.

In this case study, the value path reports “degree 4 nodes contribute negatively to the classification logit”, the attention path reports “it is important that degree 4 nodes don’t send their information, but especially in high value neighborhoods”, and their combination says “in some way, degree 4 nodes are important to the prediction”. It shows that close, differentiated inspection between the content contribution and routing roles can lead to meaningful insights about how the model comes to conclusions. This should also be kept in mind when evaluating models with metrics. For example, some graph explainability metrics mask messages that get sent to and from a node; they can’t evaluate nodes that contribute by not being read in the first place (App. H).

## 5 Conclusion

Graph Attention Networks (GATs) are applied in domains where explanations must support trust and scientific insights. A standard way to explain GNNs is to intervene at the input, but such an intervention moves both a node’s message and its attention weights. Separating the two requires attributions at the model’s internals, which Layer-wise Relevance Propagation (LRP) provides for all nodes with one backward pass per path. We have shown that in an attention layer, the relevance reaching a node is the sum of a value path share and an attention path routing influence. These two differ in baseline, in conservation, and interpretation. Existing rules either drop one or add them into a single number.

Since these rules are settings of one product rule, considering the paths separately is straightforward and gives each a definite meaning. On BA-HOUSE, where the ground truth is known, our case study shows the node attaching the motif to the graph receives negative, positive and near-zero relevance under three different settings of the rule. We read them separately and show: the value path says the node’s content contributes negatively to the predicted class and the attention path says the model’s gating of it works in the model’s favor. Which rule to use therefore depends on what is asked of an explanation. For a domain expert, the separated reading turns one ambiguous number into two statements that can be analyzed independently. In high-stakes domains in particular, the distinction matters for trust: a near-zero attribution because a node has little influence and a near-zero attribution because its counterproductive content is suppressed by the routing are different statements. Beyond trust, for a model that predicts well, the separate readings can yield more specific hypotheses about what it has learned. Tested against the data by domain experts, such hypotheses can contribute to scientific insight. We see this as a step towards a mechanistic understanding of both GATs by themselves and their attention mechanism, in particular, compared to attention-free GNNs.

Limitations and future work Our attention split is posed at a single attention layer, where it is exact. Furthermore, our main experiments use single-head models and synthetic data with known ground truth. In GATv2, the message and the source embedding share their projection, so information can be entangled even though the computation is decoupled.

Next steps are to extend the experiments to real-world data, to tasks where attention is provably necessary, and to models with multiple attention heads and layers. The main open question is whether a combination exists that retains the information of the individual readings better than the sum used in AttnLRP, so that a more meaningful relevance propagates through preceding layers. Perturbation studies of node-level attention path attributions could guide this.

In principle, the different readings can be applied to other architectures that share the bilinear, softmaxnormalized attention. To what extent the split accounts for the disagreement between CP-LRP and AttnLRP in Transformers remains open.

## Acknowledgments

This work was supported by the Fraunhofer Internal Programs (Fraunhofer) as grant ESPINN (PREPARE 40-08394).

## References

[1] Reduan Achtibat, Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer, Aakriti Jain et al. AttnLRP: Attention-aware layer-wise relevance propagation for transformers. In Proceedings ofthe 41st International Conference on Machine Learning, pages 135–168. PMLR, 2024.

[2] Ameen Ali, Thomas Schnake, Oliver Eberle, Grégoire Montavon et al. XAI for transformers: Better explanations through conservative propagation. In Proceedings ofthe 39th International Conference on Machine Learning, pages 435–451. PMLR, 2022.

[3] Kenza Amara, Zhitao Ying, Zitao Zhang, Zhichao Han et al. GraphFramEx: Towards systematic evaluation of explainability methods for graph neural networks. In Proceedings of the First Learning on Graphs Conference, pages 44:1–44:23. PMLR, 2022.

[4] Leila Arras, Bruno Puri, Patrick Kahardipraja, Sebastian Lapuschkin et al. A close look at decomposition-based XAI-methods for transformer language models, 2025. arXiv:2502.15886.

[5] Sebastian Bach, Alexander Binder, Grégoire Montavon, Frederick Klauschen et al. On pixelwise explanations for non-linear classifier decisions by layer-wise relevance propagation. PLOS ONE, 10(7):e0130140, 2015.

[6] Mayank Baranwal, Abram Magner, Jacob Saldinger, Emine S. Turali-Emre et al. Struct2Graph: A graph attention network for structure based predictions of protein–protein interactions. BMC Bioinformatics, 23(1):370, 2022.

[7] Shaked Brody, Uri Alon, and Eran Yahav. How attentive are graph attention networks? In International Conference on Learning Representations, 2022.

[8] Matthias Fey and Jan E. Lenssen. Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.

[9] Matthias Fey, Jinu Sunil, Akihiro Nitta, Rishi Puri et al. PyG 2.0: Scalable learning on real world graphs. In Temporal Graph Learning Workshop @ KDD, 2025.

[10] Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12216–12235. Association for Computational Linguistics, 2023.

[11] Aric A. Hagberg, Daniel A. Schult, and Pieter J. Swart. Exploring network structure, dynamics, and function using NetworkX. In Proceedings of the 7th Python in Science Conference (SciPy2008), pages 11–15, 2008.

[12] Yidong Hu, Wenbin Shi, and Chien-Hung Yeh. Spatiotemporal convolution sleep network based on graph attention mechanism with automatic feature extraction. Computer Methods and Programs in Biomedicine, 244:107930, 2024.

[13] Qiang Huang, Makoto Yamada, Yuan Tian, Dinesh Singh et al. GraphLIME: Local interpretable model explanations for graph neural networks. IEEE Transactions on Knowledge and Data Engineering, 35(7):6968–6972, 2023.

[14] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017.

[15] Dongsheng Luo, Wei Cheng, Dongkuan Xu, Wenchao Yu et al. Parameterized explainer for graph neural network. In Advances in Neural Information Processing Systems, volume 33, pages 19620–19631. Curran Associates, Inc., 2020.

[16] Qiujie Lv, Guanxing Chen, Ziduo Yang, Weihe Zhong et al. Meta learning with graph attention networks for low-data drug discovery. IEEE Transactions on Neural Networks and Learning Systems, 35(8):11218–11230, 2024.

[17] Grégoire Montavon, Sebastian Lapuschkin, Alexander Binder, Wojciech Samek et al. Explaining nonlinear classification decisions with deep Taylor decomposition. Pattern Recognition, 65: 211–222, 2017.

[18] Heloísa Oss Boll, Ali Amirahmadi, Mirfarid Musavian Ghazani, Wagner Ourique de Morais et al. Graph neural networks for clinical risk prediction based on electronic health records: A survey. Journal ofBiomedical Informatics, 151:104616, 2024.

[19] Hakan T. Otal, Abdulhamit Subasi, Furkan Kurt, M. Abdullah Canbaz et al. Analysis of gene regulatory networks from gene expression using graph neural networks. In Digital Healthcare, Digital Transformation and Citizen Empowerment in Asia-Pacific and Europe for a Healthier Society, volume 5 of Information Technologies in Healthcare Industry, pages 249–270. Academic Press, 2025.

[20] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer et al. PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

[21] Fahmida Liza Piya, Mehak Gupta, and Rahmatollah Beheshti. HealthGAT: Node classifications in electronic health records using graph attention networks. In 2024 IEEE/ACM Conference on Connected Health: Applications, Systems and Engineering Technologies (CHASE), pages 132–141. IEEE, 2024.

[22] Phillip E. Pope, Soheil Kolouri, Mohammad Rostami, Charles E. Martin et al. Explainability methods for graph convolutional neural networks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10772–10781. IEEE, 2019.

[23] RDKit contributors. RDKit: Open-source cheminformatics. https://www.rdkit.org, 2026. Version 2026.03.5.

[24] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner et al. The graph neural network model. IEEE Transactions on Neural Networks, 20(1):61–80, 2009.

[25] Jonathan Schmidt, Love Pettersson, Claudio Verdozzi, Silvana Botti et al. Crystal graph attention networks for the prediction of stable materials. Science Advances, 7(49):eabi7948, 2021.

[26] Thomas Schnake, Oliver Eberle, Jonas Lederer, Shinichi Nakajima et al. Higher-order explanations of graph neural networks via relevant walks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(11):7581–7596, 2022.

[27] Petar Velickoviˇ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero et al. Graph attention´ networks. In International Conference on Learning Representations, 2018.

[28] Minh Vu and My T. Thai. PGM-Explainer: Probabilistic graphical model explanations for graph neural networks. In Advances in Neural Information Processing Systems, volume 33, pages 12225–12235. Curran Associates, Inc., 2020.

[29] Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik et al. GNNExplainer: Generating explanations for graph neural networks. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

[30] Weiqiu You, Siqi Zeng, Yao-Hung Hubert Tsai, Makoto Yamada et al. When LRP diverges from leave-one-out in transformers. In Proceedings of the 8th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pages 176–188. Association for Computational Linguistics, 2025.

[31] Hao Yuan, Haiyang Yu, Jie Wang, Kang Li et al. On explainability of graph neural networks via subgraph explorations. In Proceedings ofthe 38th International Conference on Machine Learning, pages 12241–12252. PMLR, 2021.

[32] Hao Yuan, Haiyang Yu, Shurui Gui, and Shuiwang Ji. Explainability in graph neural networks: A taxonomic survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5): 5782–5799, 2023.

## A GATv2 vs. Transformer attention

A Transformer attention layer has separate projections for the input on the sender side: $W _ { k }$ for the key that enters the attention computation, and $\hat { W } _ { v }$ for the value that gets sent.

$$
y _ { i } = \sum _ { j } { \mathrm { s o f t m a x } } _ { j } \left( { \frac { q _ { i } ^ { \top } k _ { j } } { \sqrt { d _ { k } } } } \right) v _ { j } , \qquad q _ { i } = W _ { q } x _ { i } , \quad k _ { j } = W _ { k } x _ { j } , \quad v _ { j } = W _ { v } x _ { j } .\tag{6}
$$

With s the logit, the backward pass splits schematically into

$$
\frac { \partial y } { \partial x } = \underbrace { \frac { \partial y } { \partial \alpha } \frac { \partial \alpha } { \partial s } \left( \frac { \partial s } { \partial q } \frac { \partial q } { \partial x } + \frac { \partial s } { \partial k } \frac { \partial k } { \partial x } \right) } _ { \mathrm { a t t e n t i o n \ p a t h } } + \underbrace { \frac { \partial y } { \partial v } \frac { \partial v } { \partial x } } _ { \mathrm { v a l u e \ p a t h } } .\tag{7}
$$

In GATv2 message passing, the source node that sends the message has a single projection $W _ { l }$ that enters the attention computation and gets transmitted as the message,

$$
y _ { i } = \sum _ { j } { \mathrm { s o f t m a x } } _ { j } \left( a ^ { \top } { \mathrm { L e a k y R e L U } } \left( W _ { r } x _ { i } + W _ { l } x _ { j } \right) \right) W _ { l } x _ { j } .\tag{8}
$$

$$
\frac { \partial y } { \partial x } = \underbrace { \frac { \partial y } { \partial \alpha } \frac { \partial \alpha } { \partial s } \frac { \partial s } { \partial \hat { z } } \frac { \partial \hat { z } } { \partial z } \left( \frac { \partial z } { \partial q } \frac { \partial q } { \partial x } + \frac { \partial z } { \partial m } \frac { \partial m } { \partial x } \right) } _ { \mathrm { a t t e n t i o n ~ p a t h } } + \underbrace { \frac { \partial y } { \partial m } \frac { \partial m } { \partial x } } _ { \mathrm { v a l u e ~ p a t h } } ,\tag{9}
$$

where, as in Eq. (2) with edge attributes omitted for simplicity, $q _ { i } = W _ { r } x _ { i }$ and $m _ { j } = W _ { l } x _ { j }$ are the destination and source projections, $z _ { i j } = q _ { i } + m _ { j }$ the pre-activation, $\hat { z } _ { i j } = \sigma ( z _ { i j } )$ its LeakyReLU, and $\boldsymbol { s } _ { i j } = \boldsymbol { a } ^ { \intercal } \hat { \boldsymbol { z } } _ { i j }$ the logit.

In the GATv2 backward pass, both the attention path and the value path carry the factor $\partial m / \partial x = W _ { l }$ This does not couple the decomposition. Eq. (3) is exact and its two terms are disjoint, so $w _ { \mathrm { a t t } } = 0$ removes the attention term entirely and $R ^ { \mathrm { v a l } }$ carries no routing contribution, and conversely. What the sharing does limit is how far agreement between the two maps can be read as independent evidence, since the shared projection was fit under both objectives at once. It does not force that agreement: both readings express a node through the same vector $W _ { l } x _ { n } .$ , but score it against different covectors — the upstream signal $g _ { i }$ on the value path, the scorer’s linearization $p _ { i n }$ in the source role (Eq. (5)). In Transformer attention, the two paths share no such factor.

Empirically, the architecture control in App. G (Fig. 7), which compares GATv2 with TransformerConv (separate $W _ { k } , W _ { v } )$ and an attention-free GCN, suggests that the shared projection has little effect, and that the architectural similarities of the bilinear product and softmax normalization outweigh the different embedding and logit computation, at least on our case study dataset. The shared projection does not change what the two maps report qualitatively.

## B Edge-level algebra and the step to nodes

## B.1 Edge relevances

Let T be the explained scalar and $g _ { i }$ the signal that reaches the layer output $y _ { i } { \mathrm { : } }$ the standard gradient ${ \partial T } / { \partial g _ { i } }$ or its modified-gradient counterpart under any upstream rules. The value path edge relevance is one application of G×I at the message, with

$$
r _ { i j } ^ { \mathrm { v a l } } = m _ { j } ^ { \top } \frac { \partial T } { \partial m _ { j } } = m _ { j } ^ { \top } \frac { \partial T } { \partial y _ { i } } \frac { \partial y _ { i } } { \partial m _ { j } } = \alpha _ { i j } g _ { i } ^ { \top } m _ { j } .\tag{10}
$$

For the attention path we need

$$
\frac { \partial T } { \partial \alpha _ { i j } } = g _ { i } ^ { \top } m _ { j } , \qquad \frac { \partial \alpha _ { i k } } { \partial s _ { i j } } = \alpha _ { i k } \left( \delta _ { k j } - \alpha _ { i j } \right) ,\tag{11}
$$

so that

$$
\begin{array} { l } { { r _ { i j } ^ { \mathrm { a t t } } = \displaystyle \frac { \partial T } { \partial s _ { i j } } = \sum _ { k \in \mathcal { N } ( i ) } \displaystyle \frac { \partial T } { \partial \alpha _ { i k } } \displaystyle \frac { \partial \alpha _ { i k } } { \partial s _ { i j } } = \sum _ { k \in \mathcal { N } ( i ) } g _ { i } ^ { \top } m _ { k } \alpha _ { i k } \left( \delta _ { k j } - \alpha _ { i j } \right) } } \\ { { = \alpha _ { i j } g _ { i } ^ { \top } m _ { j } - \alpha _ { i j } \displaystyle \sum _ { k \in \mathcal { N } ( i ) } \alpha _ { i k } g _ { i } ^ { \top } m _ { k } = \alpha _ { i j } g _ { i } ^ { \top } ( m _ { j } - y _ { i } ) } } \\ { { = r _ { i j } ^ { \mathrm { v a l } } - \alpha _ { i j } \displaystyle \sum _ { k \in \mathcal { N } ( i ) } r _ { i k } ^ { \mathrm { v a l } } . } } \end{array}\tag{12}
$$

The middle form $\alpha _ { i j } g _ { i } ^ { \top } ( m _ { j } - y _ { i } )$ can also be interpreted as a statement of what the map measures: the neighbor’s content score against the attention-weighted level of its own neighborhood.

## B.2 Exactness of the value path ablation

With the attention weights frozen the layer is linear in the messages, so setting $m _ { j } ~  ~ 0$ gives $\Delta y _ { i } = - \alpha _ { i j } m _ { j }$ with no remainder and no interaction with the other neighbors, whose coefficients do not depend on $m _ { j }$ . With $g _ { i }$ fixed,

$$
\Delta \left( g _ { i } ^ { \top } y _ { i } \right) = - r _ { i j } ^ { \mathrm { v a l } } , \qquad \sum _ { i : j \in N ( i ) } \Delta \left( g _ { i } ^ { \top } y _ { i } \right) = - R _ { j } ^ { \mathrm { v a l } } ,\tag{13}
$$

so $\mathrm { G } { \times } \mathrm { I }$ at the edge level message is not an estimate of the ablation effect but the effect itself. The statement requires an intervention that leaves every $\alpha _ { i j }$ untouched: masking the edge message qualifies, since the scorer never reads $m _ { j }$ , whereas zeroing $x _ { j }$ or deleting the node moves the logits as well and perturbs both paths at once, as in an ordinary forward pass the routing is never frozen.

The attention path cannot work this way: the softmax is nonlinear in the logits and its row normalization couples all neighbors of a destination.

## B.3 Row-sum cancellation

With $\begin{array} { r } { \sum _ { j \in \mathcal { N } ( i ) } \alpha _ { i j } = 1 } \end{array}$ , the second term of Eq. (12) sums to the first,

$$
\sum _ { j \in \mathcal { N } ( i ) } r _ { i j } ^ { \mathrm { a t t } } = \sum _ { j \in \mathcal { N } ( i ) } r _ { i j } ^ { \mathrm { v a l } } - \Big ( \sum _ { j \in \mathcal { N } ( i ) } \alpha _ { i j } \Big ) \sum _ { k \in \mathcal { N } ( i ) } r _ { i k } ^ { \mathrm { v a l } } = 0 .\tag{14}
$$

Attention is a fixed budget: what one neighbor is given is taken from the others. The question at the logit is not how much an edge delivers but whether it delivers more than the neighbors it competes with. A below-average neighbor is scored negative however much relevance its edge carries in total. Thus, ordering of attention path relevances across edges can be completely different from the value path, and it works on a different scale, depending on how different the neighborhood value path relevances are.

## B.4 From logits to nodes

The relevance of Eq. (12) sits on $s _ { i j }$ . With $z _ { i j } = W _ { l } x _ { j } + W _ { r } x _ { i } + W _ { e } e _ { i j }$ and σ applied elementwise,

$$
\frac { \partial s _ { i j } } { \partial x _ { j } } = W _ { l } ^ { \top } p _ { i j } , \qquad \frac { \partial s _ { i j } } { \partial x _ { i } } = W _ { r } ^ { \top } p _ { i j } , \qquad p _ { i j } = \sigma ^ { \prime } ( z _ { i j } ) \odot a .\tag{15}
$$

Applying G×I at $x _ { n }$ and collecting the logits $x _ { n }$ enters — as a source in $s _ { i n }$ for each destination i, and as a destination in $s _ { n j }$ for each of its own neighbors j — gives Eq. (5).

Two remarks. First, the source-side input to the scorer is $W _ { l } x _ { j }$ , the same projection that produces the message; as the attention path explanation uses those weights in its gradient, it inherits information from the training of the value path. The projection still influences both paths separately, as different upstream relevance signals arrive at that projection. Second, with self-loops a node is its own neighbor and the self-edge contributes to both sums.

## C Path reading flowchart

Table 1 and Table 2 show the possible reading of the value and the attention path attributions.

Table 1: Readings of a positive attribution. The value path does not branch; the attention path branches by role and, within each role, by the direction on which demand and supply agreed. The negative case is given in Table 2.
<table><tr><td></td><td>value path</td><td colspan="4">attention path</td></tr><tr><td></td><td></td><td colspan="2">n as source</td><td colspan="2">n as destination</td></tr><tr><td>node map edge term</td><td> $R _ { n } ^ { \mathrm { v a l } }$   $\alpha _ { i n } g _ { i } ^ { \top } m _ { n }$ </td><td colspan="2"> $\textstyle \sum _ { i : n \in { \mathcal { N } } ( i ) } r _ { i n } ^ { \mathrm { a t t } } u _ { i n }$   $r _ { i n } ^ { \mathrm { a t t } } u _ { i n }$ </td><td colspan="2"> $\textstyle \sum _ { j \in { \mathcal { N } } ( n ) } r _ { n j } ^ { \mathrm { a t t } } v _ { n j }$   $r _ { n j } ^ { \mathrm { a t t } } \ v _ { n j }$ </td></tr><tr><td>reference scales with</td><td>0 baseline content  $m _ { n }$ </td><td colspan="2">neighborhoods reading n contrast in demand</td><td colspan="2"> ${ \mathcal { N } } ( n ) .$  , contrasted twice contrast in demand and in supply</td></tr><tr><td>supply (below)</td><td></td><td> $u _ { i n } > 0$  n pushes logit up n pushes logit down n rates j above</td><td> $u _ { i n } < 0$ </td><td> $v _ { n j } > 0$ </td><td> $v _ { n j } < 0$  n rates j below</td></tr><tr><td>demand (above)</td><td></td><td> $r _ { i n } ^ { \mathrm { a t t } } > 0$  score edge up</td><td> $r _ { i n } ^ { \mathrm { a t t } } < 0$  score edge down</td><td> $r _ { n j } ^ { \mathrm { a t t } } > 0$  shift toward j</td><td> $r _ { n j } ^ { \mathrm { a t t } } < 0$  shift away from j</td></tr><tr><td>reading</td><td>my content helped the target</td><td>read me, and reading me helps</td><td>skip me, and skipping me helps</td><td>helps</td><td>I favor this one, I pass this one over, and favoring it and passing it over helps</td></tr></table>

In the combined AttnLRP map, the attribution of a single node sums three components: the value path, and the source and destination roles of the attention path (Tables 1 and 2). A single value entangles them, and cancellation can make a node whose components conflict indistinguishable from one with little influence. Each role is itself a sum over edges and can cancel internally, as in any attribution that sums over features. Within one role, however, the summed relevance keeps a single, clear reading for the node, whereas cancellation across paths and roles mixes different interpretable readings.

In general, the granularity of an attribution should follow the question asked of it. The case study in Section 4 shows the skip me, and skipping me helps reading of a positive attention path attribution, because the source role dominates. The attribution is positive because the demand on the anchor’s outgoing logits is negative, and its features push these logits down. This lowers the attention its neighbors pay to it and limits the impact of its negative content. If neither role dominates, the roles may have to be interpreted separately. Otherwise, the attention attribution can again hide canceling effects, although they can both be attributed to routing influences.

Table 2: Readings of a negative attribution: the mirror of Table 1, where demand and supply disagree. Note that within one neighborhood $\begin{array} { r } { \sum _ { j \in \mathcal { N } ( n ) } r _ { n j } ^ { \mathrm { a t t } } = 0 , } \end{array}$ , so a node acting as destination always faces demands of both signs.
<table><tr><td></td><td>value path</td><td colspan="4">attention path</td></tr><tr><td></td><td></td><td colspan="2">n as source</td><td colspan="2">n as destination</td></tr><tr><td>edge term</td><td> $\alpha _ { i n } g _ { i } ^ { \top } m _ { n }$ </td><td colspan="2"> $r _ { i n } ^ { \mathrm { a t t } } u _ { i n }$ </td><td colspan="2"> $r _ { n j } ^ { \mathrm { a t t } } v _ { n j }$ </td></tr><tr><td>supply (below)</td><td></td><td> $u _ { i n } > 0$  n pushes logit up n pushes logit down n rates j above</td><td> $u _ { i n } < 0$ </td><td> $v _ { n j } > 0$ </td><td> $v _ { n j } < 0$ </td></tr><tr><td>demand (above)</td><td></td><td> $r _ { i n } ^ { \mathrm { a t t } } < 0$  score edge down</td><td> $r _ { i n } ^ { \mathrm { a t t } } > 0$  score edge up</td><td> $r _ { n j } ^ { \mathrm { a t t } } < 0$  shift away from j</td><td>n rates j below  $r _ { n j } ^ { \mathrm { a t t } } > 0$  shift toward j</td></tr><tr><td>reading</td><td>my content</td><td>read me,</td><td>skip me,</td><td>I favor this one, I pass this one over,</td><td></td></tr><tr><td></td><td></td><td>but reading</td><td>but skipping</td><td></td><td></td></tr><tr><td></td><td>hurt</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>but favoring it</td><td>but passing it over</td></tr><tr><td></td><td>the target</td><td>me hurts</td><td>me hurts</td><td>hurts</td><td>hurts</td></tr></table>

All sign combinations across the value path and the two attention roles can occur, because no component constrains the sign of another. First, the value path is measured against a zero baseline, whereas the attention path demand is measured against the neighborhood, whose extent depends on the role. In the destination role, n’s features steer which of its 1-hop neighbors’ messages it reads, and n receives positive relevance if it rates neighbors with above-average messages higher and those with below-average messages lower. In the source role, the 2-hop neighborhood matters, because each destination evaluates n relative to its own neighborhood. Thus, content that hurts the target can still receive positive demand if its destinations’ other neighbors carry even less valuable content, and helpful content can receive negative demand if the competing content is more valuable. Second, value and supply project the same message $W _ { l } x _ { n }$ onto different directions $( g _ { i } \ \mathbf { v s } . p _ { i n } ) .$ , so their signs are not constrained either. Fig. 1 shows both agreement (hubs) and disagreement (node 16).

## D Experimental setup

All models were implemented in PyTorch [20] using PyTorch Geometric [8, 9] for message passing. NetworkX [11] was used for graph generation and RDKit [23] for molecular featurization. All experiments were run on CPU (Apple M4). Training a single model takes approximately a couple of seconds for BA-HOUSE and QKV-BA and up to 1.5 minutes on ESOL.

Each synthetic BA dataset is generated once and shared by all five runs per setting. These five seeds are for the model initialization only and not for data generation. Each attention-free control is trained under the same protocol as the model it controls for.

Layer hyperparameters are PyTorch Geometric defaults, with four exceptions. We use a single attention head throughout. Self-loops are written into the edge set rather than added inside the layer (add\_self\_loops=False), so that a loop is an ordinary edge with an ordinary attention weight and the message-masking metrics of App. H operate on exactly the edge set the model sees. Source and destination projections are kept separate (share\_weights=False). For the TransformerConv control we disable the root weight (root\_weight=False) and remove the unused lin\_skip. The LeakyReLU slope is 0.2. No attention dropout is used; feature dropout is 0.1 between the ESOL convolutions and 0 elsewhere.

Datasets, architectures, training schedules and the set of attributed graphs are summarized in Table 3.

Nothing is selected on the QKV-BA and BA-HOUSE held-out sets: the epoch budget is fixed, with no early stopping and no best-weight restore, and the sets are independently generated rather than partitioned, so they serve as test sets. ESOL is the only model with selection, and reports on a test split untouched by it; its split is random rather than scaffold-based. On BA-HOUSE, the explained class is the model’s argmax, and no attributed graph is misclassified (0 of 3000), so the predictedand true-class conventions coincide.

Table 3: Datasets, models and evaluation. All models share one skeleton: L message-passing layers of width h, one head, ReLU between convolutions, linear readout, sum-pooled where the task is graph-level. QKV-BA has no activation after the last convolution, its target being linear in v.
<table><tr><td></td><td>QKV-BA</td><td>BA-HOUSE ESOL</td></tr><tr><td>graphs</td><td> $\mathbf { B A } n { = } 8 , m { = } 2$ </td><td>pos.: BA n=15,  $m { = } 2 ~ + ~ 1 1 1 6$  Delaney molecules HoUSEMOTIF; neg.: BA (1128 before de-duplication)  $n { = } 2 0 , m { = } 2 ;$  balanced</td></tr><tr><td>node feat.</td><td> $\begin{array} { r c l c l } { x _ { i } } & { = } & { \left[ q _ { i } | k _ { i } | v _ { i } \right] } & { \in } & { \mathbb { R } ^ { 1 5 } } \end{array}$   $d _ { q } = d _ { k } = \dot { d } _ { v } = 5 ,$ </td><td>, one-hot true degree, 10 bins 35-dim one-hot (element, de- iid (loops excluded from the de- gree, charge, nH, hybrid., gree) aromaticity, ring)</td></tr><tr><td>edge feat.</td><td> $\dot { U ( - 1 , 1 ) }$ </td><td>9-dim one-hot bond</td></tr><tr><td>task</td><td>head (below), MSE</td><td>regress one Transformer graph classification, cross- graph regression on stan- entropy dardized log S, MSE</td></tr><tr><td>model</td><td>par. par.</td><td>GATv2 h=16, L=1, 629 GATv2 h=32, L=1, 834 GATv2 h=64, L=4, 32449 par.</td></tr><tr><td>training</td><td>Adam, lr  $1 0 ^ { - 2 }$  , cosine, 60 Adam, lr epochs, batch 64</td><td> $1 0 ^ { - 2 } ,$  cosine, 30 AdamW, lr  $1 0 ^ { - 3 } .$  epochs, batch 32 600 epochs, batch 32, best- validation weights restored</td></tr><tr><td>split</td><td>2000 train / 500 held- 600 train / 1200 held- ated (seeds 0 and 10000)</td><td> $8 9 2 / 1 1 1 / 1 1 3 ,$  one fixed out, independently gener- out, independently gener- random split (seed 0) ated (seeds 0 and 10000)</td></tr><tr><td>attributed on</td><td>60 further graphs (seed 99), the 600 positive held-out all 1116 molecules node 0 only</td><td>graphs, ×5 seeds = 3000 records; per seed 248 degree-four and 352 degree-</td></tr><tr><td>explained scalar</td><td> $\scriptstyle \sum$  of node 0&#x27;s output</td><td>three anchors sum-pooled logit of the pre- standardized output dicted class</td></tr></table>

QKV-BA A constructed task where the ground truth of the split is known. Node features come in three blocks, $x _ { i } = \left[ q _ { i } \mid k _ { i } \mid v _ { i } \right]$ , and the target is one transformer attention head on the graph’s neighborhoods,

$$
y _ { i } = \sum _ { j \in \mathcal { N } ( i ) } \alpha _ { i j } v _ { j } , \qquad \alpha _ { i } = \operatorname { s o f t m a x } _ { j \in \mathcal { N } ( i ) } \bigl ( \beta \langle q _ { i } , k _ { j } \rangle / \sqrt { d _ { q } } \bigr ) , \qquad \beta = 8 .
$$

Routing depends on q and k alone, content on v alone. A model that fits this target must read $q ,$ k in its scorer and v in its messages, so a correct path split has to put value path relevance on v and attention path relevance on $q , k \colon$ the answer is fixed by the construction, not by our reading of it. Three constants make the task genuinely require attention rather than merely permit it.

$\beta = 8$ sets how far the softmax commits. At low $\beta$ the weights are near-uniform and the target collapses towards a plain mean of the neighbors’ $v ,$ which any fixed-weight aggregator can fit — at $\beta = 1$ a mean aggregator already reaches $R ^ { 2 } = 0 . 9 3$ , leaving almost nothing for routing to explain. $\operatorname { A t } \beta = 8$ the head commits (normalized attention entropy 0.46) and the best attention-free predictor reaches only $R ^ { 2 } = 0 . 4 1 5$ , so most of the target is reachable exclusively through input-dependent routing.

$m \geq 2$ makes neighborhoods overlap: two nodes share two non-self neighbors in 99.8% of graphs. A static scorer (GATv1) ranks all sources in the same order regardless of who reads them [7], and so cannot rank the same two sources differently in two overlapping neighborhoods. The task therefore needs the dynamic scorer of GATv2.

• Separate q and k blocks keep a node’s score for itself an ordinary random draw, so the self-loop does not turn “copy $\boldsymbol { v } _ { i } ^ { \flat }$ into a shortcut.

Finally $h > d _ { v } ,$ so that the readout is not a square, generically invertible map, which would force the value path onto v by construction rather than by learning; the sweep $h \in \{ 5 , 1 0 , 1 6 \}$ is reported in Table 4. One limit is inherent to the construction: $\alpha _ { i }$ has d $\mathtt { g } _ { i } - 1$ free parameters against the $d _ { v }$ equations $y _ { i }$ supplies, so routing is identifiable only where d $\operatorname { s g } _ { i } - 1 \leq d _ { v }$

BA-HOUSE PyG’s HOUSEMOTIF (eaves 0−1, floors 2−3, square $0 - 1 - 2 - 3$ , rooftop 4) is wired in by one edge from a uniformly random house node to a uniformly random main-graph node. Its house-side endpoint is the anchor: attaching at an eave moves its degree 3 → 4 (degree-four anchor), at a floor or the rooftop $2  3$ (degree-three anchor). The (before, after) pair is stored per graph, so statistics can be stratified without re-deriving the geometry.

ESOL A real molecular regression: aqueous solubility (log S) of 1116 small molecules, one graph per molecule, atoms as nodes and bonds as edges. Atoms carry a 35-dimensional one-hot of chemical facts (element, degree, formal charge, attached hydrogens, hybridization, aromaticity, ring membership), bonds a 9-dimensional one (bond type, conjugation, self-loop). There is no input encoder, so every input coordinate stays a named chemical fact and relevance can be read per chemical property rather than per latent dimension; eight of the 35 coordinates are all-zero on this dataset and necessarily receive zero relevance. It is also the only dataset here with edge features, which in GATv2 enter the score and never the message (Eq. (2)).

## D.1 Computing the two paths

Eq. (3) scales the gradient entering each factor of the product $\alpha _ { i j } m _ { j } \colon w _ { \mathrm { a t t } }$ into $\alpha , w _ { \mathrm { v a l } }$ into m. Two implementations are available. A custom backward per operation, as in AttnLRP [1], evaluates the product once but has to be hand-written. A reweighted forward, as in the detach form of CP-LRP [2], rewrites the aggregation as

$$
\begin{array} { l } { { y _ { i } = w _ { \mathrm { a t t } } \displaystyle \sum _ { j \in { \mathcal { N } } ( i ) } \alpha _ { i j } [ m _ { j } ] . \mathsf { d e t a c h } ( \ l ) ~ + ~ w _ { \mathrm { v a l } } \displaystyle \sum _ { j \in { \mathcal { N } } ( i ) } [ \alpha _ { i j } ] . \mathsf { d e t a c h } ( \ l ) ~ m _ { j } } } \\ { { ~ + ~  ( 1 - w _ { \mathrm { a t t } } - w _ { \mathrm { v a l } } ) \displaystyle [ \sum _ { j \in { \mathcal { N } } ( i ) } \alpha _ { i j } m _ { j } ] . \mathsf { d e t a c h } ( \ l ) , } } \end{array}\tag{16}
$$

which is their trick with the weights freed — CP-LRP is ${ \bf w } = ( 1 , 0 )$ , where only the middle term survives. The three terms sum to y<sub>i</sub>, so the prediction never moves and only the backward is reweighted. We use this second form: stock autograd, one code path for every w, at the price of three product evaluations. The two agree to float precision.

Gradient × Input on the reweighted forward gives G×I at $\mathbf { w } = ( 1 , 1 )$ , CP-LRP at (1, 0), AttnLRP at $\bigl ( \frac { 1 } { 2 } , \frac { 1 } { 2 } \bigr )$ (identity rule at the activation) and the attention path at (0, 1). The split is taken at layer 1 throughout.

## E Further experiments

Table 4: QKV-BA: Mean fraction of |R| mass per block in node 0.
<table><tr><td>Model</td><td> $R ^ { 2 }$ </td><td>Value path (q/k/v)</td><td>Attention path  $\mathrm { ( q / k / v ) }$ </td></tr><tr><td>TransformerConv h=5</td><td>1.000</td><td>0.000 / 0.000 / 1.000</td><td>0.323 / 0.677 / 0.000</td></tr><tr><td>GATv2 h=5</td><td>0.413</td><td>0.019 / 0.043 / 0.939</td><td>0.168 / 0.166 / 0.666</td></tr><tr><td>GATv2 h=10</td><td>0.780</td><td>0.008 / 0.019 / 0.974</td><td>0.303 / 0.611 / 0.087</td></tr><tr><td> $\mathrm { G A T v } 2 \mathrm { h } { = } 1 6$ </td><td>0.950</td><td>0.005 / 0.003 / 0.992</td><td>0.326 / 0.610 / 0.064</td></tr></table>

QKV-BA: each path is blind to what only the other one reaches On QKV-BA, the input $x _ { i } = [ q _ { i } | k _ { i } | v _ { i } ]$ is blocked such that the target uses $q ,$ k only for routing and v only for content. The path split recovers this partition (Table 4, Fig. 3): a well-fitting GATv2 places 0.992 of its value path mass on the value block and 0.936 of its attention path mass on query and key. The separation is not exact because the forward function is not learned perfectly, some features from routing blocks reach the value block, and vice versa. Under-trained models thus separate less cleanly. A GATv2 with Transformer attention (TransformerConv), whose forward pass is the target after the linear projections are learned, separates perfectly.

ESOL: a molecular example Edge attributes enter only the attention path by definition $\left( \operatorname { E q . } \left( 2 \right) \right)$ and therefore obtain no relevance in pure value path attributions. This is a structural property, which we confirm empirically on ESOL: the output has a nonzero gradient with respect to the edge attributes, whereas CP-LRP assigns them an exact zero.

![](images/543d7074706d43a6047d4d97fe42a274952c80911cff61136db6a20a19c197fb.jpg)  
Figure 3: Shows separation of attributions by input block on separate path explanations.

![](images/25b7f3527c9bcf05105e7a53cce9b516df144d4864a1cfe8e70034021377b4e5.jpg)  
Figure 4: ESOL: attention-path share $\sigma _ { c } = | R _ { , c } ^ { \mathrm { a t t } } | / ( | R _ { , c } ^ { \mathrm { a t t } } | + | R _ { , c } ^ { \mathrm { v a l } } | )$ per live input coordinate (27 of 35; the other 8 are always zero on ESOL), at the layer-1 split with the value path in all layers above, attributed over all 1116 molecules and sorted by $\sigma _ { c } .$ Markers: mean ± SD over 5 initializations. Grey bars: 95% bootstrap interval over molecules (2000 resamples) for the best-validation model, centered on that model’s own estimate. Dashed line: global attention-path share, averaged over initializations (0.240; per initialization 0.201–0.280). Hollow markers: coordinates present on fewer than 300 atoms, excluded from all statistics.

Fig. 4 illustrates, on real-world data, that some input features receive proportionally more attention path relevance than the average and others less. The pattern does not depend on which molecules are evaluated: computing $\sigma _ { c }$ separately on two disjoint halves of the molecules gives almost the same ordering (Spearman $\rho = 0 . 9 8 )$ . It also largely persists across trainings, with an average Spearman $\rho = 0 . 7 2$ over the 10 pairs of initializations (range 0.61–0.87). In every initialization, degree 3, charge +0, SP2 and SP3 hybridization and aromaticity lean towards the attention path, while 0, 2 or 3 attached hydrogens, degrees 1 and 2 and chlorine lean towards the value path. The figure motivates real-world applications in which mechanistic understanding could contribute to scientific discovery.

Fig. 5 is a further, anecdotal illustration that the two paths pick up distinct information. However, node level agreement between the two paths does not reproduce across initializations in our experiments $( \rho \approx 0 . 0 2 )$ , only at the feature level. Future work could investigate input-feature ablations that induce stronger weighting of edge embeddings, or a learned embedding MLP on the chemical data, so that the network can form features suited to the routing and content roles of the message passing layer, which could be a step towards node-level agreement. Propagating relevance through such an embedding MLP would still recover the chemical feature relevances.

![](images/9ca42a3a89edda7fa82cf0d2fb8aa4cd719ef245ae110b2323a26961252c13d4.jpg)  
Figure 5: Three molecules spanning the solubility range. Top row: the value map summed over features per atom. Bottom row: the routing map, same atoms, same layer-1 split. Color is signed relevance normalized by the column maximum — value against routing on the same molecule is the comparison that is meaningful; magnitudes are not comparable across columns. Molecules are chosen deterministically for legibility, not for what they show.

## F Additional case study visualization

![](images/2c69ed14350843ac61ef7bf39acef7188e70ce743e5526ce95d61c7a8b5b200c.jpg)  
Figure 6: Node relevance for the house logit against degree, full version of Fig. 2.

## G Architecture control

The two attention architectures build their score differently: GATv2 adds a source and a destination projection, passes the sum through a LeakyReLU, and reuses the source projection as the message; TransformerConv takes a dot product of separate query and key projections. What they share is a bilinear product of attention weight and message under a softmax.

Trained under the identical protocol, they behave alike. All three architectures, the attention-free GCN included, learn the same content reading: its sign changes between degree three and degree four in every architecture and seed combination (Fig. 7a). The routing reading is the part that is attention-specific — the degree-four anchor separation is present for GATv2 and TransformerConv, and absent for the GCN. The shared source projection is therefore not what produces the reading, and the remaining differences in embedding and score do not change it, at least on this dataset.

## H What metrics reward

Fig. 8 places the value path near the ceiling (Fid+ = 0.98, char. 0.99) and the attention path far behind it (0.50, char. 0.61), still outperforming random. Fidelity+ cannot measure the suppressionbased positive evidence, it only rewards removals that lower the prediction. On node 16 of Fig. 1, masking its edges without renormalizing raises the logit margin by +1.23, and letting the softmax redistribute the attention it freed raises it by +4.39. Either way, the model classifies the graph better once it can no longer read from that node, and this holds for every 3 → 4 anchor node in the evaluation set. The gap between the value and attention paths reflects, at least in part, the question posed by the metric rather than simply the quality of the explanations.

![](images/e17d4771ea17f05736e9d14aaf67f58b2bf1db1e99f63ed373f7621e897fa73f.jpg)

![](images/f7f723aabcd67659219b21d41e6ffa507776c800dfb2e80068cbf96defa60658.jpg)  
Figure 7: (a) Every content reading crosses between degree three and four; each curve is divided by its own largest value, since only the ordering and the sign boundary are the claim. (b) Margin gain from suppressing the anchor minus the same quantity for an ordinary node of the same degree in the same graph, split into what removing content buys and what rerouting the freed attention buys on top. Degree-three anchors are the control. Bars are 95% cluster-bootstrap intervals.

![](images/71c01f9affffcea58649b2994f2391570ae4c152cb9477995e79738c696ae191.jpg)

![](images/b11456b7ca69160be57382c22f6676dc3f7b9679cea92c2d6b14d74b38365bb2.jpg)

![](images/6e8964f8c10c120132afc7776421378d20f07efcd09c36ee6db1a27d42b9c4fe.jpg)  
Figure 8: Fidelity metrics of Yuan et al. [32], with Fid+ and sparsity going back to Pope et al. [22], which remove a node by masking the messages from and to it. Fid+: drop in predicted-class probability when the explanation is removed, so high means necessary. Fid-: the same drop when only the explanation is kept, so low means sufficient. Sparsity is the fraction of the graph not in the explanation; characterization score [3] is the harmonic mean of Fid+ and (1 - Fid-).

![](images/83fa0febce7151e3a7f965d6fcee489a69d3162db0901e8122712a2247666baf.jpg)  
Figure 9: Fidelity+ with attention knockout instead of message masking: the most positive edges of each edge attribution map have their logits set to −∞ together, allowing neighborhood attention to renormalize. Same models and graphs as Fig. 8. E is the set of all edges. Up to sparsity of 0.70: 30% of edges removed.

In Transformer research, a useful tool for mechanistic interpretability next to value zeroing is attention knockout [10]. It answers the question: what would happen if this logit did not exist? Deletion here does not mean zero but $s _ { i j } \to - \infty$ , because this lets the neighborhood distribute its attention as if the edge did not exist. Over all 12,900 edges of the graphs in Fig. 8, the logit relevance $r _ { i j } ^ { a t t } \left( \mathrm { E q . } \left( 4 \right) \right)$ predicts single-edge knockout with Spearman $\rho = 0 . 9 9$ , while the value path’s edge relevance $r _ { i j } ^ { v a l }$ reaches $\rho = 0 . 8 6$ , because it measures an edge’s content against zero rather than against the rest of its neighborhood. Used as a fidelity metric like Fig. 8, knocking out the most relevant edges of each map together (Fig. 9), the attention path lowers the class probability faster while fewer edges are removed. Knocking out edges with highest content, those of the value path, catches up as the number of removed edges increases and the single-edge prediction no longer holds.

That the attention path’s logit relevance predicts the removal of edges well (Fig. 9), while its nodelevel relevance predicts the removal of nodes poorly (Fig. 8), goes along with the demand supply interpretation (Section 3.2, App. C). Each node’s features can contribute positively or negatively to a logit whose relevance already has a clear, individually signed role within its neighborhood. These examples illustrate how the choice of evaluation metric affects the interpretation of explanation scores. Further perturbation research, especially at the node level, and an analysis of how to leverage disentangled attention path and value path attributions for downstream explanations are left for future work.