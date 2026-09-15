# When Correlations Mislead: Confounder-Aware Multi-View Urban Region Representation Learning – Extended Version

Sean Bin Yang<sup>†</sup>, Ying Sun<sup>‡</sup>, Zongyi Xu<sup>‡</sup>, Tung Kieu<sup>†</sup>,

Jilin Hu<sup>§</sup>, Bin Yang<sup>§</sup>, Kristian Torp<sup>†</sup>, Hua Lu<sup>†</sup>, Torben Bach Pedersen<sup>†</sup>

<sup>†</sup>Aalborg University, Aalborg, Denmark

<sup>‡</sup>Chongqing University of Posts and Telecommunications, Chongqing, China <sup>§</sup>East China Normal University, Shanghai, China

{seany, tungkvt, torp, luhua, tbp}@cs.aau.dk, {sunying, xuzy}@cqupt.edu.cn, {jlhu, byang}@dase.ecnu.edu.cn,

Abstract—Urban region representation learning commonly combines heterogeneous data sources, such as mobility flows, points of interest, and land-use information, to support tasks including mobility analysis, public safety forecasting, and service demand estimation. Existing multi-view methods typically improve region embeddings by strengthening interactions across views. However, such methods often overlook view-specific regional structures and may propagate correlations induced by shared latent factors, which can reduce the stability of downstream predictions. To overcome this major limitation, we propose CURE, a confounder-aware framework for multi-view urban region representation learning. CURE first encodes each view with its regional graph structure, estimates a shared latent component, and then reduces its projected influence before crossview interaction. A hierarchical graph-aware fusion module subsequently aggregates the residual view representations using local and global regional contexts. Experiments on three real-world cities show that CURE improves predictive performance, remains robust under missing and noisy input views, and provides reliable cross-view integration through shared component separation and context-dependent view weighting.

This is an extended version of ” When Correlations Mislead: Confounder-Aware Multi-View Urban Region Representation Learning”, to appear in ICDE 2027.

Index Terms—Multi-view data integration, confounder-aware learning, urban region representation learning.

## I. INTRODUCTION

Urban region embeddings support a wide range of city analytics tasks, including mobility analysis [1]–[7], locationbased services [1], [8]–[19], and crime prediction [20]–[24]. These tasks increasingly rely on heterogeneous observations, such as human mobility flows, points of interest (POIs), landuse records, and sensing data. Region representation learning provides a reusable way to encode such observations into low-dimensional vectors that preserve functional, spatial, and socioeconomic characteristics [25]–[30].

Early methods primarily derive region embeddings from a dominant data source, such as mobility flows, POIs, building footprints, or geographic observations. Mobility-based approaches characterize regions through human transition patterns, while RegionDCL [26] shows that building footprints, complemented by POIs, can also reveal useful regional semantics. A single view, however, captures only part of the information relevant to modeling an urban region.

Therefore, recent methods combine heterogeneous urban data. Multi-View Joint Graph Representation Learning [27] models mobility patterns and inherent region properties through cross-view information sharing and adaptive fusion. HAFusion [28] focuses on attentive feature learning and fusion, while CGAP [31] captures local and global graph structures through coarsened graph attention pooling. FlexiReg [25] further adapts region embeddings to different spatial partitions and downstream tasks. These methods rely heavily on observed dependencies across views and regions. However, such dependencies are not always reliable. Mobility, POI, and land-use views may be correlated because they reflect complementary urban semantics, but they may also respond to shared latent factors, such as population density, commercial intensity, transportation accessibility, and socioeconomic activity. As a result, directly strengthening these correlations can amplify shared biases and weaken view-specific signals.

Fig. 1 illustrates the distinction between existing paradigms and our motivation. Single-view methods learn region representations from one data source, limiting their ability to capture heterogeneous urban semantics. Conventional multiview methods alleviate this limitation by integrating multiple views through general interaction and fusion. However, they typically assume that observed cross-view correlations are informative and reliable, while overlooking shared latent factors that may simultaneously influence different views. In urban data, factors such as population density, commercial intensity, transportation accessibility, and socioeconomic activity may induce correlations among mobility, POI, and land-use views without necessarily reflecting stable or taskrelevant semantics [32]–[36]. As a result, existing single-view and multi-view methods may encode biased shared signals or propagate spurious cross-view dependencies into learned region representations.

In this work, we distinguish reliability from robustness in multi-view urban data integration. Reliability refers to whether the integrated representation reflects stable and task-relevant urban semantics rather than spurious cross-view correlations induced by shared latent factors. Robustness refers to whether the learned representation remains stable when the input views are incomplete or noisy. Under this distinction, reliability is challenged by correlation-driven view interaction and contextdependent view contribution, whereas robustness is challenged by data-quality variations such as missing or noisy views. This distinction motivates the following three challenges:

![](images/69d763feb2508bad0eba1cb3cfba54796fe7741d460eccf4903ff32fd8999129.jpg)  
Fig. 1. Comparison of region representation learning paradigms. (a) Single-view region representation learning constructs embeddings from one urban data source, which limits its ability to capture heterogeneous urban semantics. (b) Conventional multi-view region representation learning integrates multiple views through general cross-view interaction, but may directly propagate correlation-induced shared signals. (c) CURE explicitly models potential latent confounding factors and performs confounder-aware view interaction before fusion, enabling the learned region representations to emphasize stable and task-relevant view specific information for downstream urban analytics.

Challenge I: View-specific regional structure. Different urban views encode different types of regional dependencies. For example, mobility flows describe region-to-region movement connectivity, POI distributions capture functional similarity, and land-use records reflect planning-oriented spatial organization. Existing methods often treat these views mainly as heterogeneous feature sources to be fused, while overlooking the structural dependencies within each view. As a result, view-specific local patterns may be weakened before crossview interaction, reducing the model’s ability to preserve structured semantics from each data source. This also limits robustness, since the model may fail to exploit the remaining structured views when one view is incomplete or corrupted.

Challenge II: Correlation-driven cross-view interaction. Most multi-view methods strengthen observed dependencies across views, implicitly assuming that stronger cross-view correlations are beneficial. In urban data, however, different views may be correlated because they are influenced by shared latent factors, such as population density, commercial intensity, transportation accessibility, or socioeconomic activity. Directly propagating such correlations may amplify biased shared signals and introduce spurious cross-view dependencies. This creates a reliability risk: the learned representation may appear predictive by exploiting strong observed correlations, while these correlations may not correspond to stable or task-relevant urban semantics.

Challenge III: Context-dependent view contribution. Different urban views do not contribute equally across regions and tasks. Mobility signals may be highly informative in commercial centers but sparse or unstable in suburban regions. Land-use information may provide stable semantics in residential areas but may be less responsive to short-term activity patterns. POI distributions may capture functional composition but can be incomplete or unevenly updated. Existing fusion strategies mainly aim to increase representation strength, but they rarely model whether each view provides stable and taskrelevant information under local and global regional contexts. This affects reliability, since a view that is informative in one region or task may introduce less relevant or unstable signals in another context, causing fixed or globally shared fusion strategies to overemphasize misleading view-specific information.

To address these challenges, we propose CURE, a ConfoUnder-awaRE framework for reliable multi-view urban region representation learning. CURE is designed following three principles. First, to address view-specific regional structure, CURE employs a graph-guided intra-view encoder that constructs and exploits a regional graph for each urban view. This enables mobility, POI, and land-use views to preserve their own structural semantics before cross-view interaction, providing more stable view-specific signals when some views are incomplete or noisy. Second, to address correlation-driven cross-view interaction, as shown in Fig. 1(c), CURE introduces a confounder-aware inter-view interaction module that estimates a shared latent component across views and performs interaction in the residual representation space. This design reduces the direct propagation of shared latent correlations and encourages cross-view learning to focus on complementary view-specific information. Third, to address context-dependent view contribution, CURE develops a hierarchical graph-aware residual fusion module that aggregates residual view representations under both local and global graph contexts. By learning region-wise adaptive view weights, this module allows the model to emphasize views that provide more stable and task-relevant information under different urban structures and downstream tasks. Together, these designs enable CURE to improve predictive performance, provide reliable cross-view integration, and remain robust under incomplete or noisy urban observations.

Extensive experiments on three real-world cities demonstrate that CURE learns reliable region representations for urban analytics tasks, including check-in prediction, crime forecasting, and service call prediction. Compared with representative baselines, CURE consistently improves predictive performance across cities and tasks. Ablation studies verify the contribution of each component, while further analyses demonstrate both robustness under incomplete and noisy views and reliability in shared component separation and adaptive view weighting.

The main contributions are summarized as follows:

• We propose CURE, a confounder-aware framework that integrates heterogeneous urban views into reliable regionlevel representations for downstream urban analytics.

• We design a graph-guided intra-view encoder that preserves view-specific regional structures and provides stable structured signals under incomplete or noisy views.

• We introduce a confounder-aware inter-view interaction module that estimates a shared latent component and performs cross-view learning in the residual space, reducing spurious dependencies induced by shared latent factors.

• We develop a hierarchical graph-aware residual fusion module that adaptively aggregates view-specific signals according to their context-dependent task relevance under local and global regional contexts.

• We conduct extensive experiments on three real-world cities and domain-specific analytics tasks, showing that CURE improves predictive performance, remains robust under incomplete and noisy views, and provides reliable cross-view integration.

## II. RELATED WORK

## A. Urban Region Representation Learning

Urban region representation learning encodes urban observations into compact embeddings for tasks such as landuse classification, crime prediction, and population estimation. Early methods are often centered on a dominant data source. For instance, MGFN [37] derives temporal mobility patterns from multiple mobility graphs, whereas RegionDCL [26] primarily uses OpenStreetMap building footprints, complemented by POIs, to capture urban morphology and function. Such methods can model specific regional characteristics effectively, but they offer only a partial description of the urban environment. Multi-view methods combine heterogeneous signals to obtain more comprehensive embeddings. Multi-View Joint Graph Representation Learning [27] introduces crossview information sharing and adaptive fusion for mobility and regional attributes. HREP [8] models intra-region features together with inter-region relations, while CGAP [31] incorporates local graph structure and urban-level global information. HAFusion [28] further studies attentive interaction within and across views. Other work improves the applicability of region embeddings across spatial partitions and tasks [25], [26]. Most of these methods treat stronger cross-view dependencies as useful signals. However, heterogeneous urban views may also be correlated through shared latent factors. Our work focuses on reducing the influence of such factors before crossview interaction and fusion.

## B. Reliable Multi-View Urban Data Integration

Multi-view learning integrates heterogeneous data sources that describe the same entities from complementary perspectives. In urban analytics, mobility flows, POI distributions, and land-use records respectively characterize movement connectivity, functional composition, and planning-oriented spatial structure. Existing urban representation methods exploit heterogeneous information through cross-view information sharing and adaptive aggregation [27], attentive feature learning and fusion [28], and graph-based integration of regional attributes and mobility contexts [31]. Other methods employ contrastive learning to improve the generalizability of region embeddings across spatial partitions [26]. Although effective, these approaches primarily focus on strengthening correlations across views or regions, and often treat the observed dependencies as reliable evidence. This assumption may be problematic because heterogeneous urban views can be jointly influenced by shared latent factors, such as population density, transportation accessibility, commercial activity, and socioeconomic status. Directly reinforcing such correlations may amplify biased shared signals and obscure informative view-specific patterns. To address this issue, CURE estimates a shared latent component, reduces its projected influence through soft residualization, and performs inter-view interaction in the residual representation space. It further aggregates view-specific residual representations under local and global regional contexts, enabling context-dependent integration of stable and task-relevant urban signals.

## C. Confounder-Aware Representation Learning

Confounding is a fundamental challenge in data-driven modeling, where observed dependencies may arise from latent factors rather than meaningful causal or semantic relationships [38]–[43]. In representation learning, such confounders may cause embeddings to encode unstable or biased associations, thereby reducing robustness and generalization [44]–[47]. For example, stable learning [38] incorporates ideas from causal inference to improve predictive modeling when stability, explainability, and fairness are important. Causal representation learning (CRL) [45] further studies how interventional data can facilitate the identification of latent causal factors from low-level sensory observations. It shows that interventions provide geometric signatures that enable provable identification of latent factors, even without distributional or dependency assumptions. This issue is particularly relevant to heterogeneous urban data, where latent domain factors may simultaneously influence mobility patterns, POI distributions, land-use characteristics, and downstream prediction targets. Although existing urban representation learning methods [26], [28], [31] capture rich correlations across heterogeneous views, they rarely account for latent confounding during cross-view interaction and fusion. Unlike prior work, CURE introduces confounder-aware modeling into multi-view urban data integration. Specifically, it estimates a shared latent component, reduces its projected influence via soft residualization, and performs cross-view interaction only in the residual space. This design encourages complementary view-specific learning while suppressing spurious shared correlations, leading to more robust and generalizable urban representations.

TABLE I COMMON NOTATIONS.
<table><tr><td>Notation</td><td>Description</td></tr><tr><td> $N$ </td><td>Number of urban regions</td></tr><tr><td> $V$ </td><td>Number of urban data views</td></tr><tr><td> $F _ { v }$ </td><td>Feature dimension of the v-th view</td></tr><tr><td> $^ d$ </td><td>Embedding dimension</td></tr><tr><td> $\mathcal { R } = \{ \mathcal { R } _ { 1 } , . . . , \mathcal { R } _ { N } \}$ </td><td>Set of urban regions</td></tr><tr><td> $X ^ { ( v ) } \in \mathbb { R } ^ { N \times F _ { v } }$ </td><td>Input feature matrix of the v-th view</td></tr><tr><td> $A ^ { ( v ) } \in \mathbb { R } ^ { N \times N }$ </td><td>View-specific regional graph of the v-th view</td></tr><tr><td> $A ^ { \mathrm { l o c a l } } \in \mathbb { R } ^ { N \times N }$ </td><td>Local regional graph used in fusion</td></tr><tr><td> $A ^ { \mathrm { g l o b a l } } \mathbf { \Phi } \in \mathbb { R } ^ { N \times N }$ </td><td>Global functional-similarity graph used in fusion</td></tr><tr><td> $H ^ { ( v ) } \in \mathbb { R } ^ { N \times d }$ </td><td>Intra-view representation of the v-th view</td></tr><tr><td> $C \in \mathbb { R } ^ { \bar { N } \times d }$ </td><td>Estimated shared latent component across views</td></tr><tr><td> $R ^ { ( v ) } \in \mathbb { R } ^ { N \times d }$ </td><td>Residual representation of the v-th view</td></tr><tr><td> $\boldsymbol { \alpha } ^ { ( v ) } \in \mathbb { R } ^ { N \times 1 }$ </td><td>Adaptive fusion weight of the v-th view</td></tr><tr><td> $H \in \mathbb { R } ^ { N \times d }$ </td><td>Final urban region embedding</td></tr><tr><td> $\lambda$ </td><td>Weight of the decorrelation regularizer</td></tr></table>

## III. PRELIMINARIES

We first introduce the basic concepts used throughout the paper and then formulate the problem. The main notation is given in Table I.

## A. Definitions

Definition 1 (Urban Regions): Given a study area A, let $\mathcal { R } = \{ \mathcal { R } _ { 1 } , . . . , \mathcal { R } _ { N } \}$ denote a set of non-overlapping urban regions that partition A, where N is the number of regions. Formally, these regions satisfy $\mathcal { R } _ { i } \cap \mathcal { R } _ { j } = \varnothing$ for $i \neq j$ and $\textstyle \bigcup _ { i = 1 } ^ { N } { \mathcal { R } } _ { i } = { \mathcal { A } } .$ . Each region $\mathcal { R } _ { i }$ corresponds to a geographically bounded spatial unit, such as a grid cell, census tract, or administrative zone. In this work, urban regions serve as the basic analytical units for integrating heterogeneous urban data, constructing region-level representations, and supporting downstream urban analytics tasks.

Definition 2 (Mobility, POI, and Land-Use Views): The mobility view characterizes movement connectivity among regions and is represented by a region-level mobility matrix $X ^ { ( 1 ) } ~ \in ~ \mathbb { R } ^ { N \times N }$ . Each entry $X _ { i j } ^ { ( 1 ) }$ records the flow volume from region $\mathcal { R } _ { i }$ to region $\mathcal { R } _ { j }$ during a given observation period, such as taxi trips, human mobility flows, or commuting records. This view reflects dynamic spatial interactions and transportation-related dependencies between regions.

The POI view describes the functional composition of each region based on the distribution of POI categories. It is represented by a feature matrix ${ \cal X } ^ { ( 2 ) } \in \mathbb { R } ^ { N \times F _ { 2 } }$ , where each feature dimension corresponds to a POI category or a derived functional attribute. The POI view captures region-level urban functions, such as commercial activity, residential services, education, healthcare, recreation, and transportation facilities.

The land-use view reflects the planning-oriented spatial structure of each region and is represented by a feature matrix $X ^ { ( 3 ) } \in \mathbb { R } ^ { N \times \bar { F } _ { 3 } }$ . Each row summarizes the land-use composition of a region, such as residential, commercial, industrial, public service, green space, or transportation-related land-use types. Compared with the POI view, which reflects fine-grained functional facilities, the land-use view provides a more structural and planning-level description of urban space.

Definition 3 (Multi-View Urban Features): Given the region set R, each region is associated with multiple heterogeneous urban data views that describe different aspects of urban semantics. We denote the collection of multi-view features as

$$
\mathcal { X } = \{ X ^ { ( 1 ) } , X ^ { ( 2 ) } , \ldots , X ^ { ( V ) } \} ,\tag{1}
$$

where V is the number of views and $X ^ { ( v ) } \in \mathbb { R } ^ { N \times F _ { v } }$ denotes the feature matrix of the v-th view. The i-th row $X _ { i } ^ { ( v ) } \in \mathbb { R } ^ { F _ { v } }$ represents the feature vector of region $\mathcal { R } _ { i }$ under view v, and $F _ { v }$ is the corresponding feature dimensionality. Different views may have different feature dimensionalities, data distributions, and semantic meanings. In this work, we consider three representative urban views: mobility, points of interest (POIs), and land-use information. Nevertheless, our method naturally extends to more views.

## B. Problem Definition

Given the region set R and the associated multi-view urban features $\bar { \mathcal { X } } = \{ X ^ { ( 1 ) } , X ^ { ( 2 ) } , \ldots , X ^ { ( V ) } \}$ , the objective of reliable multi-view urban region representation learning is to learn a unified embedding matrix

$$
H \in \mathbb { R } ^ { N \times d } ,\tag{2}
$$

where d is the embedding dimensionality and the i-th row $H _ { i } \in \mathbb { R } ^ { d }$ denotes the learned representation of region $\mathcal { R } _ { i } .$ In this work, H is learned by optimizing a structure-preserving objective that reconstructs pairwise regional similarities from mobility, POI, and land-use views, together with a decorrelation regularizer that reduces the dependence between residual view representations and the estimated shared latent component. Therefore, a desirable embedding should preserve view-specific regional structures, suppress misleading shared correlations, and support downstream urban analytics tasks, such as check-in prediction, crime forecasting, and service call prediction.

## IV. METHODOLOGY

We propose CURE, a confounder-aware framework for reliable multi-view urban region representation learning. Given heterogeneous urban views, including mobility, POIs, and land-use information, CURE aims to learn unified region embeddings that preserve view-specific semantics while mitigating spurious cross-view dependencies caused by shared latent factors.

![](images/17d687eed02e1895edf2839e432df34c505df8102b00958ea9cb62f814a58de8.jpg)  
Fig. 2. Overview of CURE. CURE encodes mobility, POI, and land-use views with graph-guided intra-view encoders, estimates a shared latent component and reduces its projected influence before cross-view interaction, and adaptively fuses residual view representations under local and global graph contexts. The resulting region embeddings are used for downstream tasks such as check-in prediction, crime forecasting, and service call prediction.

## A. Framework Overview

Let $\mathcal { V } = \{ v _ { 1 } , v _ { 2 } , . . . , v _ { V } \}$ denote the set of urban views. For each view $ { \boldsymbol { v } } \in  { \mathcal { V } } .$ , we denote its feature matrix by $X ^ { ( v ) } \in \mathbb { R } ^ { N \times F _ { v } }$ , where N is the number of urban regions and $F _ { v }$ is the feature dimensionality of view v. In this work, we consider three representative urban views, namely mobility, POIs, and land-use. The objective is to learn a unified region embedding matrix $H \in \mathbb { R } ^ { \bar { N } \times d }$ that can support downstream urban analytics tasks.

As illustrated in Fig. 2, the framework consists of four key components: graph-guided intra-view encoding, confounderaware inter-view interaction, hierarchical graph-aware residual fusion, and region-level refinement. Specifically, CURE first preserves view-specific regional structures through graphguided intra-view encoding, which provides stable structured signals under incomplete or noisy views. It then estimates a shared latent component and performs inter-view interaction in the residual space, improving reliability by reducing spurious cross-view dependencies induced by shared latent factors. Next, hierarchical graph-aware residual fusion adaptively aggregates residual view representations under local and global graph contexts, improving context-dependent reliability and robustness under input perturbations. Finally, region-level refinement captures higher-order dependencies and produces the unified region embedding H.

## B. Graph-Guided Intra-View Encoding

Different urban views encode different regional structures that may exhibit dependencies across views. To preserve such view-specific dependencies, we associate each view with a regional graph $\dot { A } ^ { ( v ) } \in \mathbb { R } ^ { N \times N }$ and encode it independently.

For view v, we first project its raw features into a shared latent space:

$$
Z _ { 0 } ^ { ( v ) } = \phi _ { v } ( X ^ { ( v ) } ) ,\tag{3}
$$

where $\phi _ { v } ( \cdot )$ is a learnable projection and $Z _ { 0 } ^ { ( v ) } \in \mathbb { R } ^ { N \times d }$

We first add self-loops and row-normalize the view-specific adjacency matrix:

$$
\tilde { A } ^ { ( v ) } = \left( D ^ { ( v ) } \right) ^ { - 1 } \left( A ^ { ( v ) } + I \right) ,\tag{4}
$$

where $D ^ { ( v ) }$ is the corresponding degree matrix.

At the l-th encoder block, graph propagation is performed as:

$$
G _ { l } ^ { ( v ) } = \tilde { A } ^ { ( v ) } Z _ { l } ^ { ( v ) } W _ { g , l } ^ { ( v ) } .\tag{5}
$$

The graph-enhanced representation is obtained through a residual connection and layer normalization:

$$
Z _ { l } ^ { \prime ( v ) } = \mathrm { L a y e r N o r m } \left( Z _ { l } ^ { ( v ) } + \mathrm { D r o p o u t } ( G _ { l } ^ { ( v ) } ) \right) .\tag{6}
$$

Multi-head self-attention (MHA) and a feed-forward network (FFN) are then applied:

$$
\begin{array} { r } { S _ { l } ^ { ( v ) } = \mathrm { M H A } \left( Z _ { l } ^ { \prime ( v ) } , Z _ { l } ^ { \prime ( v ) } , Z _ { l } ^ { \prime ( v ) } \right) , } \end{array}\tag{7}
$$

$$
Z _ { l } ^ { \prime \prime ( v ) } = \mathrm { L a y e r N o r m } \left( Z _ { l } ^ { \prime ( v ) } + \mathrm { D r o p o u t } ( S _ { l } ^ { ( v ) } ) \right) ,\tag{8}
$$

$$
Z _ { l + 1 } ^ { ( v ) } = \mathrm { L a y e r N o r m } \left( Z _ { l } ^ { \prime \prime ( v ) } + \mathrm { D r o p o u t } \left( \mathrm { F F N } ( Z _ { l } ^ { \prime \prime ( v ) } ) \right) \right)\tag{9}
$$

After stacking $L _ { G }$ graph-guided encoder blocks, we obtain the intra-view representation:

$$
{ \cal H } ^ { ( v ) } = \phi _ { \mathrm { o u t } } ^ { ( v ) } \left( Z _ { L _ { G } } ^ { ( v ) } \right) ,\tag{10}
$$

where $\begin{array} { r l r } { H ^ { ( v ) } } & { { } \in } & { \mathbb { R } ^ { N \times d } } \end{array}$ and $\phi _ { \mathrm { o u t } } ^ { ( v ) } ( \cdot )$ is a learnable output transformation.

## C. Confounder-Aware Inter-View Interaction

Heterogeneous urban views often share latent factors, which may induce spurious cross-view correlations. The intuition is that signals consistently shared across views may capture common urban background factors, such as population density, commercial intensity, or transportation accessibility, rather than complementary task-relevant semantics. Therefore, instead of directly interacting the original view representations, CURE estimates a shared latent component as a learnable proxy for such cross-view commonality.

Given this component, CURE reduces its projected influence from each view through soft residualization. The resulting residual representations retain more view-specific information while suppressing correlations already explained by the shared component. Cross-view interaction is then performed in the residual space, encouraging the model to focus on complementary view-specific semantics:

$$
C = f \left( \frac { 1 } { V } \sum _ { v = 1 } ^ { V } H ^ { ( v ) } \right) ,\tag{11}
$$

where $f ( \cdot )$ is a learnable nonlinear mapping and $C \in \mathbb { R } ^ { N \times d }$ denotes the shared latent component.

For each view, we reduce the projected influence of C through soft residualization:

$$
\begin{array} { r } { R ^ { ( v ) } = H ^ { ( v ) } - \gamma _ { v } g _ { v } ( C ) , } \end{array}\tag{12}
$$

where $g _ { v } ( \cdot )$ is a view-specific projection and $\gamma _ { v } = \sigma ( \theta _ { v } )$ is a learnable coefficient constrained to (0, 1). This design reduces the influence of dominant shared signals while avoiding overremoval of useful common semantics.

The residual representations are organized into a regionwise view sequence:

$$
\begin{array} { r } { Q _ { 0 } = \mathrm { S t a c k } \left( R ^ { ( 1 ) } , \ldots , R ^ { ( V ) } \right) \in \mathbb { R } ^ { N \times V \times d } . } \end{array}\tag{13}
$$

In particular, we denote the inter-view attentive feature learning module as InterAFL, which models dependencies among different residual views for each region.

At the l-th inter-view block, the residual representations are projected into an intermediate attention space:

$$
E _ { l } = Q _ { l } W _ { k , l } ,\tag{14}
$$

where $W _ { k , l } \in \mathbb { R } ^ { d \times d _ { m } }$

The attention features are normalized across views:

$$
A _ { l , n , v , s } = \frac { \exp ( E _ { l , n , v , s } ) } { \sum _ { u = 1 } ^ { V } \exp ( E _ { l , n , u , s } ) } .\tag{15}
$$

where $n ,$ v, and s index the region, view, and intermediate attention dimension, respectively.

The attention features are further normalized along the intermediate dimension:

$$
\bar { A } _ { l , n , v , s } = \frac { A _ { l , n , v , s } } { \sum _ { j = 1 } ^ { d _ { m } } A _ { l , n , v , j } } .\tag{16}
$$

The block output is:

$$
Q _ { l + 1 } = \bar { A } _ { l } W _ { o , l } ,\tag{17}
$$

where $W _ { o , l } \in \mathbb { R } ^ { d _ { m } \times d }$ is a learnable output projection.

After stacking $L _ { I }$ blocks, the interacted residual representations are:

$$
\Big [ \tilde { R } ^ { ( 1 ) } , \ldots , \tilde { R } ^ { ( V ) } \Big ] = \mathrm { M L P } ( Q _ { L _ { I } } ) .\tag{18}
$$

We further combine the interacted and original residual representations:

$$
\begin{array} { r } { \hat { R } ^ { ( v ) } = \beta \tilde { R } ^ { ( v ) } + ( 1 - \beta ) R ^ { ( v ) } , } \end{array}\tag{19}
$$

where $\beta = \sigma ( \theta _ { \beta } )$ is a learnable mixing coefficient constrained to $( 0 , 1 )$ . This encourages cross-view learning to focus on complementary view-specific information.

## D. Hierarchical Graph-Aware Residual Fusion

The contribution of each view to reliable integration may vary across regions and structural contexts. Therefore, we fuse residual view representations with local and global graph contexts. Let $A ^ { \mathrm { l o c a l } } \in \mathbf { \mathbb { R } } ^ { N \times N }$ denote a local regional graph and $A ^ { \mathrm { g l o b a l } } \in \mathbb { R } ^ { N \times N }$ denote a global functional-similarity graph. For each view $v ,$ we compute:

$$
H _ { \mathrm { l o c a l } } ^ { ( v ) } = A ^ { \mathrm { l o c a l } } \hat { R } ^ { ( v ) } W _ { l } ,\tag{20}
$$

$$
H _ { \mathrm { g l o b a l } } ^ { ( v ) } = A ^ { \mathrm { g l o b a l } } \hat { R } ^ { ( v ) } W _ { h } ,\tag{21}
$$

where $W _ { l } , W _ { h } \in \mathbb { R } ^ { d \times d }$ are learnable matrices.

The residual representation and its graph-enhanced contexts are combined as:

$$
\begin{array} { r } { \bar { R } ^ { ( v ) } = \psi \left( [ \hat { R } ^ { ( v ) } \| H _ { \mathrm { l o c a l } } ^ { ( v ) } \| H _ { \mathrm { g l o b a l } } ^ { ( v ) } ] \right) , } \end{array}\tag{22}
$$

where $\psi ( \cdot )$ is a nonlinear transformation and ∥ denotes concatenation.

Region-wise fusion weights are computed by:

$$
e ^ { ( v ) } = s ( \bar { R } ^ { ( v ) } ) ,\tag{23}
$$

$$
\alpha ^ { ( v ) } = \frac { \exp ( e ^ { ( v ) } ) } { \sum _ { u = 1 } ^ { V } \exp ( e ^ { ( u ) } ) } ,\tag{24}
$$

where $s ( \cdot )$ is a learnable scoring function and $\boldsymbol { \alpha } ^ { ( v ) } \in \mathbb { R } ^ { N \times 1 }$ The fused representation is given as follows:

$$
H _ { f } = \sum _ { v = 1 } ^ { V } \alpha ^ { ( v ) } \odot \bar { R } ^ { ( v ) } + C .\tag{25}
$$

This fusion module adaptively emphasizes view-specific signals that are more stable and task-relevant under local and global regional structures.

## E. Region-Level Refinement

The fused representation is further refined to capture higherorder dependencies among regions:

$$
H = \operatorname { R e g i o n F u s i o n } ( H _ { f } ) ,\tag{26}
$$

where RegionFusion(·) denotes a Transformer-style region refinement module composed of multi-head self-attention, residual connections, layer normalization, and a feed-forward network.

## F. Training of CURE

We train CURE with a multi-objective loss that preserves the region-level similarity structures encoded by mobility, POI, and land-use features. Let $x _ { i } ^ { s }$ and $\ v x _ { i } ^ { t }$ denote the outgoing and incoming mobility flow vectors of region i, respectively, and let $x _ { i } ^ { p }$ and $x _ { i } ^ { l }$ denote its POI and land-use feature vectors. We first map the unified region embedding H into feature-oriented representations:

$$
H ^ { q } = \phi _ { q } ( H ) , \qquad q \in \{ s , t , p , l \} ,\tag{27}
$$

where $\phi _ { q } ( \cdot )$ is a learnable decoder and $H ^ { q } = \{ h _ { 1 } ^ { q } , \ldots , h _ { N } ^ { q } \}$

Specifically, for each feature type $q \in \{ s , t , p , l \}$ , we define

$$
\mathcal { L } _ { q } = \frac { 1 } { N ^ { 2 } } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \left| S _ { i j } ^ { q } - \mathrm { s i m } \left( h _ { i } ^ { q } , h _ { j } ^ { q } \right) \right| ,\tag{28}
$$

where s and t denote outgoing and incoming mobility structures, while p and l denote POI and land-use structures, respectively. $S _ { i j } ^ { q } ~ = ~ \sin ( x _ { i } ^ { q } , x _ { j } ^ { q } )$ is the pairwise similarity computed from the original feature type $q ,$ and $h _ { i } ^ { q }$ is the corresponding decoded representation of region $\mathcal { R } _ { i }$

The mobility objective jointly preserves outgoing and incoming flow structures:

$$
\mathcal { L } _ { \mathrm { m o b } } = \mathcal { L } _ { s } + \mathcal { L } _ { t } .\tag{29}
$$

The POI and land-use objectives are defined as $\mathcal { L } _ { \mathrm { p o i } } = \mathcal { L } _ { p }$ and $\mathcal { L } _ { \mathrm { l a n d } } = \mathcal { L } _ { l }$ , respectively. The main training objective is:

$$
\mathcal { L } _ { \mathrm { m a i n } } = \mathcal { L } _ { \mathrm { m o b } } + \mathcal { L } _ { \mathrm { p o i } } + \mathcal { L } _ { \mathrm { l a n d } } .\tag{30}
$$

To reduce the dependence between the shared latent component and the residual view representations, we introduce a decorrelation regularizer:

$$
\mathcal { L } _ { \mathrm { d e c o r } } = \frac { 1 } { V } \sum _ { v = 1 } ^ { V } \left. \mathrm { C o r r } \Big ( R ^ { ( v ) } , C \Big ) \right. _ { F } ^ { 2 } ,\tag{31}
$$

where $\mathrm { C o r r } \big ( R ^ { ( v ) } , C \big ) \ \in \ \mathbb { R } ^ { d \times d }$ denotes the dimension-wise Pearson correlation matrix between $R ^ { ( v ) }$ and C, computed across the N urban regions.

The final objective is:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { m a i n } } + \lambda \mathcal { L } _ { \mathrm { d e c o r } } , } \end{array}\tag{32}
$$

where λ controls the strength of the decorrelation regularizer. In practice, we gradually introduce ${ \mathcal { L } } _ { \mathrm { d e c o r } }$ during training to stabilize optimization. Through joint optimization, CURE learns region representations that preserve view-specific structures, reduce the influence of shared latent factors, and support reliable urban analytics.

TABLE II DATASET STATISTICS
<table><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>NY</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>SF</td></tr><tr><td rowspan=1 colspan=1>#Regions</td><td rowspan=1 colspan=1>180</td><td rowspan=1 colspan=4>77</td><td rowspan=1 colspan=1>175</td></tr><tr><td rowspan=1 colspan=1>#POIs           2</td><td rowspan=1 colspan=1>4,496        5</td><td rowspan=1 colspan=4>7,891</td><td rowspan=1 colspan=1>28,578</td></tr><tr><td rowspan=1 colspan=1>#POI categories</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=4>26</td><td rowspan=1 colspan=1>26</td></tr><tr><td rowspan=1 colspan=1>#Land use categories</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=4>12</td><td rowspan=1 colspan=1>23</td></tr><tr><td rowspan=1 colspan=1>#Taxi trips</td><td rowspan=1 colspan=1>10,953,879     3</td><td rowspan=1 colspan=4>,381,807</td><td rowspan=1 colspan=1>357,749</td></tr><tr><td rowspan=1 colspan=1>(Collection period)</td><td rowspan=1 colspan=1>2015/06 - 2015/07</td><td rowspan=1 colspan=4>72021/01 - 2022/01|</td><td rowspan=1 colspan=1>2008/05 - 2008/06</td></tr><tr><td rowspan=1 colspan=1>#Crime records</td><td rowspan=1 colspan=1>35,335</td><td rowspan=1 colspan=4>18,200</td><td rowspan=1 colspan=1>48,489</td></tr><tr><td rowspan=1 colspan=1>(Collection period)</td><td rowspan=1 colspan=1>unknown</td><td rowspan=1 colspan=4>2022/12 - 2022/12|</td><td rowspan=1 colspan=1>2011/01 - 2022/12</td></tr><tr><td rowspan=1 colspan=1>#Check-ins</td><td rowspan=1 colspan=1>106,902</td><td rowspan=1 colspan=4>167,232</td><td rowspan=1 colspan=1>87,750</td></tr><tr><td rowspan=1 colspan=1>(Collection period)</td><td rowspan=1 colspan=1>2012/04 - 2013/09|2</td><td rowspan=1 colspan=4>012/04 - 2013/09|</td><td rowspan=1 colspan=1>2012/04 - 2013/09</td></tr><tr><td rowspan=1 colspan=1>#Service calls</td><td rowspan=1 colspan=1>516,187       2</td><td rowspan=1 colspan=4>4,350</td><td rowspan=1 colspan=1>34,385</td></tr><tr><td rowspan=1 colspan=1>(Collection period)</td><td rowspan=1 colspan=1>2023/01 - 2023/032</td><td rowspan=1 colspan=4>022/12 - 2022/12|2</td><td rowspan=1 colspan=1>022/01 - 2022/12</td></tr></table>

## V. EXPERIMENTS

## A. Experimental Settings

Datasets. We evaluate CURE on three metropolitan-scale urban datasets collected from New York City (NY)<sup>1</sup>, Chicago (Chi)<sup>2</sup>, and San Francisco (SF)<sup>3</sup>. These datasets cover diverse urban environments with different spatial layouts, population distributions, and activity patterns. Following prior work [28], each city is partitioned into spatial regions, which serve as the basic units for feature construction and downstream analytics.

For each city, we construct heterogeneous urban views from multiple data sources. The mobility view is derived from taxi trip records aggregated into region-level origin–destination flows. The POI view is constructed from OpenStreetMap<sup>4</sup> and characterizes the functional composition of each region. The land-use view describes planning-oriented regional attributes and reflects relatively stable urban structures. We further collect three types of region-level urban activity records, including Foursquare check-ins<sup>5</sup>, crime incidents, and civic service calls, which are used as prediction targets. Dataset statistics are reported in Table II.

Evaluation Scenarios. We evaluate the learned region representations on three analytics tasks. Check-in prediction estimates the volume of user check-ins in each region and reflects human mobility and location-based activity intensity. Crime forecasting predicts regional crime occurrences and evaluates whether the embeddings capture public-safety-related spatial patterns. Service call prediction estimates civic service requests and measures the ability to model urban demand and operational needs.

Following prior studies [8], [27], [28], [37], all tasks are formulated as region-level regression problems. The learned region embeddings are used as input features to a lightweight regression model that predicts aggregated activity counts. We use Lasso regression [48] as the downstream predictor, since its sparsity regularization mitigates overfitting and provides a fair evaluation of embedding quality.

TABLE III  
OVERALL PERFORMANCE ON THREE DATASETS. ’↓’ DENOTES THAT SMALLER VALUES ARE PREFERRED, AND ’↑’ DENOTES THAT LARGER VALUES ARE PREFERRED. WE BOLD THE BEST PERFORMANCE IN RED AND UNDERLINE THE SECOND-BEST RESULTS IN BLUE FOR EACH DATASET.
<table><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>Method</td><td rowspan=1 colspan=3>NY</td><td rowspan=1 colspan=3>Chi</td><td rowspan=1 colspan=3>SF</td></tr><tr><td rowspan=1 colspan=1>MAE (↓)</td><td rowspan=1 colspan=1>RMSE (↓)</td><td rowspan=1 colspan=1> $R ^ { 2 } \left( \uparrow \right)$ </td><td rowspan=1 colspan=1> $\mathbf { M A E \left( \downarrow \right) }$ </td><td rowspan=1 colspan=1> $\mathbf { R M S E \left( \downarrow \right) }$ </td><td rowspan=1 colspan=1> $R ^ { 2 } \left( \uparrow \right)$ </td><td rowspan=1 colspan=1> $\mathbf { M A E \left( \downarrow \right) }$ </td><td rowspan=1 colspan=1> $\mathbf { R M S E \left( \downarrow \right) }$ </td><td rowspan=1 colspan=1> $R ^ { 2 } \left( \uparrow \right)$ </td></tr><tr><td rowspan=6 colspan=1>Chec-in</td><td rowspan=1 colspan=1>MVURE</td><td rowspan=1 colspan=1> $3 0 6 . 7 \pm 8 . 2 0$ </td><td rowspan=1 colspan=1> $4 9 9 . 6 \pm 1 2 . 9$ </td><td rowspan=1 colspan=1> $0 . 6 2 7 \pm 0 . 0 1 9$ </td><td rowspan=1 colspan=1> $1 6 9 3 \pm 7 4$ </td><td rowspan=1 colspan=1> $\lvert 3 1 7 1 \pm 1 2 8$ </td><td rowspan=1 colspan=1> $0 . 6 5 6 \pm 0 . 0 2 9$ </td><td rowspan=1 colspan=1> $3 4 6 . 8 \pm 8 . 7$ </td><td rowspan=1 colspan=1> $6 5 9 . 3 \pm 1 5 . 7$ </td><td rowspan=1 colspan=1> $0 . 5 6 2 \pm 0 . 0 2 1$ </td></tr><tr><td rowspan=1 colspan=1>MGFN</td><td rowspan=1 colspan=1> $2 9 2 . 6 \pm 1 7 . 1$ </td><td rowspan=1 colspan=1> $4 5 1 . 8 \pm 2 8 . 1$ </td><td rowspan=1 colspan=1> $0 . 6 9 0 \pm 0 . 0 4 0$ </td><td rowspan=1 colspan=1> $1 2 8 1 \pm 4 1$ </td><td rowspan=1 colspan=1> $\lvert 2 2 7 6 \pm 8 6$ </td><td rowspan=1 colspan=1> $0 . 8 1 7 \pm 0 . 0 1 1$ </td><td rowspan=1 colspan=1> $3 1 0 . 8 \pm 9 . 1 $ </td><td rowspan=1 colspan=1> $5 4 2 . 1 \pm 1 7 . 6$ </td><td rowspan=1 colspan=1> $0 . 7 0 8 \pm 0 . 0 1 0$ </td></tr><tr><td rowspan=1 colspan=1>RDCL</td><td rowspan=1 colspan=1> $3 7 1 . 2 \pm 1 0 . 3 $ </td><td rowspan=1 colspan=1> $4 9 5 . 5 \pm 1 5 . 9$ </td><td rowspan=1 colspan=1> $0 . 4 7 1 \pm 0 . 0 2 3$ </td><td rowspan=1 colspan=1> $2 4 2 7 \pm 1 2 3$ </td><td rowspan=1 colspan=1> $\left| 4 1 8 4 \pm 1 3 6 \right.$ </td><td rowspan=1 colspan=1> $0 . 4 0 2 \pm 0 . 0 4 2$ </td><td rowspan=1 colspan=1> $3 9 8 . 8 \pm 9 . 9$ </td><td rowspan=1 colspan=1> $7 4 8 . 1 \pm 1 7 . 8$ </td><td rowspan=1 colspan=1> $0 . 4 3 7 \pm 0 . 0 2 4$ </td></tr><tr><td rowspan=1 colspan=1>HREP</td><td rowspan=1 colspan=1> $2 7 6 . 3 \pm 1 1 . 7$ </td><td rowspan=1 colspan=1> $4 4 8 . 2 \pm 1 7 . 1 $ </td><td rowspan=1 colspan=1> $0 . 7 0 3 \pm 0 . 0 2 1$ </td><td rowspan=1 colspan=1> $1 6 7 9 \pm 7 1$ </td><td rowspan=1 colspan=1> $\lvert 3 1 3 5 \pm 7 9$ </td><td rowspan=1 colspan=1> $0 . 6 6 4 \pm 0 . 0 1 7$ </td><td rowspan=1 colspan=1> $3 3 0 . 9 \pm 9 . 3$ </td><td rowspan=1 colspan=1> $6 0 6 . 7 \pm 2 5 . 8$ </td><td rowspan=1 colspan=1> $0 . 6 2 9 \pm 0 . 0 3 2$ </td></tr><tr><td rowspan=1 colspan=1>HAF</td><td rowspan=1 colspan=1> $2 0 2 . 8 \pm 7 . 2$ </td><td rowspan=1 colspan=1> $3 2 2 . 8 \pm 1 2 . 6$ </td><td rowspan=1 colspan=1> $0 . 8 4 4 \pm 0 . 0 1 2$ </td><td rowspan=1 colspan=1> $9 2 9 \pm 6 2$ </td><td rowspan=1 colspan=1> $1 9 4 7 \pm 7 5$ </td><td rowspan=1 colspan=1> $0 . 8 7 0 \pm 0 . 0 1 0$ </td><td rowspan=1 colspan=1> $2 3 3 . 1 \pm 9 . 5 $ </td><td rowspan=1 colspan=1> $4 2 9 . 6 \pm 2 8 . 1$ </td><td rowspan=1 colspan=1> $0 . 8 1 3 \pm 0 . 0 2 4$ </td></tr><tr><td rowspan=1 colspan=1>CURE</td><td rowspan=1 colspan=1> $1 8 6 . 7 \pm 2 . 2$ </td><td rowspan=1 colspan=1> $2 8 9 . 0 \pm 4 . 6$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 8 7 5 \pm 0 . 0 0 4 }$ </td><td rowspan=1 colspan=1> $7 6 2 \pm 5 6$ </td><td rowspan=1 colspan=1> $\mathbf { 1 5 6 9 } \pm \mathbf { 1 9 }$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 9 1 6 \ : \pm { \ : 0 . 0 1 1 } }$ </td><td rowspan=1 colspan=1> $2 1 5 . 8 \pm 5 . 5$ </td><td rowspan=1 colspan=1> $3 7 3 . 4 \pm 6 . 0$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 8 6 0 \pm 0 . 0 0 4 }$ </td></tr><tr><td rowspan=6 colspan=1>Crime</td><td rowspan=1 colspan=1>MVURE</td><td rowspan=1 colspan=1> $\left| 6 7 . 9 \pm 1 . 1 \right.$ </td><td rowspan=1 colspan=1> $\left| 9 3 . 8 \pm 1 . 9 \right.$ </td><td rowspan=1 colspan=1> $\left| 0 . 5 9 1 \pm 0 . 0 1 6 \right.$ </td><td rowspan=1 colspan=1> $1 0 0 . 4 \pm 6 . 6$ </td><td rowspan=1 colspan=1> $1 2 9 . 2 \pm 7 . 3$ </td><td rowspan=1 colspan=1> $0 . 4 6 1 \pm 0 . 0 6 2$ </td><td rowspan=1 colspan=1> $1 3 0 . 3 \pm 1 . 7$ </td><td rowspan=1 colspan=1> $2 0 1 . 7 \pm 3 . 2$ </td><td rowspan=1 colspan=1> $0 . 5 9 4 \pm 0 . 0 1 3$ </td></tr><tr><td rowspan=1 colspan=1>MGFN</td><td rowspan=1 colspan=1> $7 0 . 2 \pm 2 . 3$ </td><td rowspan=1 colspan=1> $8 9 . 6 \pm 2 . 5$ </td><td rowspan=1 colspan=1> $0 . 6 3 0 \pm 0 . 0 2 0$ </td><td rowspan=1 colspan=1> $1 0 7 . 4 \pm 5 . 4$ </td><td rowspan=1 colspan=1> $1 3 7 . 9 \pm 5 . 2$ </td><td rowspan=1 colspan=1> $0 . 3 8 6 \pm 0 . 0 4 7$ </td><td rowspan=1 colspan=1> $1 2 8 . 4 \pm 3 . 3 $ </td><td rowspan=1 colspan=1> $1 9 9 . 9 \pm 4 . 3 $ </td><td rowspan=1 colspan=1> $0 . 6 0 1 \pm 0 . 0 1 7$ </td></tr><tr><td rowspan=1 colspan=1>RDCL</td><td rowspan=1 colspan=1> $\left| 9 8 . 7 \pm 3 . 1 \right.$ </td><td rowspan=1 colspan=1> $1 2 7 . 9 \pm 5 . 2$ </td><td rowspan=1 colspan=1> $0 . 2 5 1 \pm 0 . 0 2 6$ </td><td rowspan=1 colspan=1> $1 2 1 . 7 \pm 4 . 8$ </td><td rowspan=1 colspan=1> $1 5 9 . 6 \pm 6 . 3$ </td><td rowspan=1 colspan=1> $0 . 1 7 9 \pm 0 . 0 5 3$ </td><td rowspan=1 colspan=1> $1 5 6 . 3 \pm 2 . 1$ </td><td rowspan=1 colspan=1> $2 4 2 . 3 \pm 4 . 6$ </td><td rowspan=1 colspan=1> $0 . 4 1 3 \pm 0 . 0 2 1$ </td></tr><tr><td rowspan=1 colspan=1>HREP</td><td rowspan=1 colspan=1> $\left| 6 2 . 8 \pm 2 . 1 \right.$ </td><td rowspan=1 colspan=1> $\left| 8 3 . 1 \pm 2 . 3 \right.$ </td><td rowspan=1 colspan=1> $\left| 0 . 6 8 0 \pm 0 . 0 1 4 \right.$ </td><td rowspan=1 colspan=1> $8 8 . 3 \pm 6 . 4 $ </td><td rowspan=1 colspan=1> $1 1 4 . 4 \pm 5 . 5$ </td><td rowspan=1 colspan=1> $0 . 5 7 8 \pm 0 . 0 4 1$ </td><td rowspan=1 colspan=1> $1 2 4 . 4 \pm 2 . 3$ </td><td rowspan=1 colspan=1> $1 9 6 . 9 \pm 3 . 9$ </td><td rowspan=1 colspan=1> $0 . 6 1 2 \pm 0 . 0 1 4$ </td></tr><tr><td rowspan=1 colspan=1>HAF</td><td rowspan=1 colspan=1> $\left| 5 6 . 1 \pm 1 . 3 \right.$ </td><td rowspan=1 colspan=1> $\lvert \underline { { 7 6 . 1 } } \pm 2 . 2$ </td><td rowspan=1 colspan=1> $\underline { { \vert 0 . 7 3 4 \pm 0 . 0 1 5 } }$ </td><td rowspan=1 colspan=1> $7 7 . 8 \pm 3 . 6$ </td><td rowspan=1 colspan=1> $\underline { { 1 0 7 . 1 \pm 5 . 4 } }$ </td><td rowspan=1 colspan=1> $\underline { { 0 . 6 3 1 \pm 0 . 0 3 6 } }$ </td><td rowspan=1 colspan=1> $\underline { { 1 0 1 . 5 \pm 3 . 3 } }$ </td><td rowspan=1 colspan=1> $\underline { { 1 7 8 . 4 \pm 3 . 6 } }$ </td><td rowspan=1 colspan=1> $\underline { { 0 . 6 8 2 \pm 0 . 0 1 3 } }$ </td></tr><tr><td rowspan=1 colspan=1>CURE</td><td rowspan=1 colspan=1> $\left| 5 3 . 2 \pm 1 . 5 \right.$ </td><td rowspan=1 colspan=1> $\mathbf { \left| 7 3 . 1 \pm 0 . 9 \right. }$ </td><td rowspan=1 colspan=1> $\mathbf { \left| 0 . 7 5 5 \pm 0 . 0 0 6 \right. }$ </td><td rowspan=1 colspan=1> ${ \bf 6 9 . 6 \pm 3 . 2 }$ </td><td rowspan=1 colspan=1> $\mathbf { \left| 9 1 . 2 \pm 0 . 9 \right. }$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 3 2 \pm 0 . 0 0 5 }$ </td><td rowspan=1 colspan=1> $\mathbf { \left| 9 9 . 1 \pm 0 . 6 \right| }$ </td><td rowspan=1 colspan=1> ${ \bf 1 6 8 . 0 \pm 1 . 5 }$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 1 8 \ : \pm { \ : 0 . 0 0 5 } }$ </td></tr><tr><td rowspan=6 colspan=1>Sere eereal</td><td rowspan=1 colspan=1>MVURE</td><td rowspan=1 colspan=1> $1 4 2 8 \pm 3 3$ </td><td rowspan=1 colspan=1> $\left| 2 1 8 0 \pm 4 6 \right|$ </td><td rowspan=1 colspan=1> $\left| 0 . 3 6 7 \pm 0 . 0 2 7 \right.$ </td><td rowspan=1 colspan=1> $1 9 0 . 3 \pm 9 . 8 $ </td><td rowspan=1 colspan=1> $\left| 2 6 6 . 9 \pm 1 2 . 1 \right.$ </td><td rowspan=1 colspan=1> $0 . 4 4 1 \pm 0 . 0 5 0$ </td><td rowspan=1 colspan=1> $1 0 2 . 1 \pm 4 . 8 $ </td><td rowspan=1 colspan=1> $1 6 4 . 7 \pm 2 . 7$ </td><td rowspan=1 colspan=1> $0 . 4 7 9 \pm 0 . 0 1 7$ </td></tr><tr><td rowspan=1 colspan=1>MGFN</td><td rowspan=1 colspan=1> $1 5 5 4 \pm 8 1$ </td><td rowspan=1 colspan=1> $\vert 2 2 8 6 \pm 1 1 5$ </td><td rowspan=1 colspan=1> $\left| 0 . 3 0 3 \pm 0 . 0 6 9 \right.$ </td><td rowspan=1 colspan=1> $2 0 8 . 2 \pm 1 1 . 3$ </td><td rowspan=1 colspan=1> $2 9 3 . 4 \pm 1 6 . 6$ </td><td rowspan=1 colspan=1> $0 . 3 2 9 \pm 0 . 0 7 7$ </td><td rowspan=1 colspan=1> $1 0 2 . 8 \pm 2 . 2$ </td><td rowspan=1 colspan=1> $1 6 6 . 3 \pm 2 . 5$ </td><td rowspan=1 colspan=1> $0 . 4 6 8 \pm 0 . 0 2 1$ </td></tr><tr><td rowspan=1 colspan=1>RDCL</td><td rowspan=1 colspan=1> $1 7 8 3 \pm 2 1$ </td><td rowspan=1 colspan=1> $\left| 2 5 9 7 \pm 3 8 \right.$ </td><td rowspan=1 colspan=1> $\left| 0 . 1 0 3 \pm 0 . 0 2 6 \right.$ </td><td rowspan=1 colspan=1> $1 9 5 . 7 \pm 7 . 6$ </td><td rowspan=1 colspan=1> $\left| 2 7 2 . 1 \pm 1 0 . 1 \right.$ </td><td rowspan=1 colspan=1> $0 . 4 4 5 \pm 0 . 0 4 1$ </td><td rowspan=1 colspan=1> $1 1 6 . 6 \pm 2 . 3$ </td><td rowspan=1 colspan=1> $1 9 6 . 7 \pm 3 . 2 $ </td><td rowspan=1 colspan=1> $0 . 2 5 6 \pm 0 . 0 2 4$ </td></tr><tr><td rowspan=1 colspan=1>HREP</td><td rowspan=1 colspan=1> $1 4 3 0 \pm 2 9$ </td><td rowspan=1 colspan=1> $\lvert 2 2 8 6 \pm 3 4$ </td><td rowspan=1 colspan=1> $\left| 0 . 3 9 8 \pm 0 . 0 2 1 \right.$ </td><td rowspan=1 colspan=1> $1 8 5 . 7 \pm 6 . 1 $ </td><td rowspan=1 colspan=1> $\left| 2 6 2 . 2 \pm 1 0 . 8 \right.$ </td><td rowspan=1 colspan=1> $0 . 4 6 8 \pm 0 . 0 2 2$ </td><td rowspan=1 colspan=1> $1 0 3 . 4 \pm 3 . 2$ </td><td rowspan=1 colspan=1> $1 6 7 . 4 \pm 4 . 6$ </td><td rowspan=1 colspan=1> $0 . 4 6 1 \pm 0 . 0 2 9$ </td></tr><tr><td rowspan=1 colspan=1>HAF</td><td rowspan=1 colspan=1> $1 2 7 3 \pm 2 0$ </td><td rowspan=1 colspan=1> $1 9 5 1 \pm 2 7$ </td><td rowspan=1 colspan=1> $\left| \underline { { 0 . 4 9 3 \pm 0 . 0 1 4 } } \right.$ </td><td rowspan=1 colspan=1> $\underline { { 1 5 9 . 3 \pm 1 3 . 9 } }$ </td><td rowspan=1 colspan=1> $\underline { { 2 2 2 . 0 \pm 1 8 . 9 } }$ </td><td rowspan=1 colspan=1> $\underline { { 0 . 6 1 3 \pm 0 . 0 6 7 } }$ </td><td rowspan=1 colspan=1> $\underline { { 8 1 . 5 \pm 2 . 5 } }$ </td><td rowspan=1 colspan=1> $\underline { { 1 4 2 . 1 \pm 3 . 2 } }$ </td><td rowspan=1 colspan=1> $\underline { { 0 . 6 1 2 \pm 0 . 0 1 8 } }$ </td></tr><tr><td rowspan=1 colspan=1>CURE</td><td rowspan=1 colspan=1> $\mathbf { | 1 2 4 4 \pm 1 2 }$ </td><td rowspan=1 colspan=1> $1 9 1 3 \pm 4$ </td><td rowspan=1 colspan=1> $\mathbf { \left| 0 . 5 1 3 \pm 0 . 0 0 2 \right| }$ </td><td rowspan=1 colspan=1> $1 3 1 . 0 \pm 2 . 8$ </td><td rowspan=1 colspan=1> $1 8 4 . 3 \pm 2 . 2$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 3 6 \pm 0 . 0 0 6 }$ </td><td rowspan=1 colspan=1> $\mathbf { \left| 7 3 . 8 \pm 1 . 6 \right. }$ </td><td rowspan=1 colspan=1> $1 2 0 . 3 \pm 1 . 5$ </td><td rowspan=1 colspan=1> $\mathbf { \left| 0 . 7 2 2 \pm 0 . 0 0 7 \right. }$ </td></tr></table>

Evaluation Metrics. We use three standard regression metrics: Mean Absolute Error (MAE): $\begin{array} { r } { \mathrm { M A E } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n ^ { - } } \left| y _ { i } - \hat { y } _ { i } \right| } \end{array}$ , Root Mean Square Error (RMSE): RMSE = $\textstyle { \sqrt { \frac { 1 } { n } } } \sum _ { i = 1 } ^ { n } ( y _ { i } - { \hat { y } } _ { i } ) ^ { 2 }$ and the coefficient of determination $\begin{array} { l l l } { { \dot { ( R ^ { 2 } ) \colon ~ R ^ { 2 } } } } & { { = } } & { { 1 ~ - } } \end{array}$ $\frac { \sum _ { i = 1 } ^ { n } ( y _ { i } - { \hat { y } } _ { i } ) ^ { 2 } } { \sum _ { i = 1 } ^ { n } ( y _ { i } - { \bar { y } } ) ^ { 2 } }$ , where n is the number of regions, y<sub>i</sub> and $\hat { y } _ { i }$ are the ground-truth and predicted values of region R , and y¯ is the mean ground-truth value. Lower MAE and RMSE and higher $R ^ { 2 }$ indicate better performance.

lines cover mobility-based, semantic-based, heterogeneous graph-based, contrastive, and attention-based representation paradigms.

Implementation Details. We implement CURE using Python 3.8.18 and PyTorch 1.10, and conduct experiments on an NVIDIA RTX 8000 GPU. The hidden dimension is set to 256. Multi-head attention is used in the graph-guided intra-view encoder, confounder-aware inter-view interaction module, and region-level refinement module. The numbers of intra-view encoding blocks, inter-view interaction blocks, and region fusion blocks are selected using a validation set. Dropout is applied after input projection and within attention-based blocks.

Baselines. We compare CURE with five representative urban region representation methods. MVURE [27] jointly models mobility patterns and regional attributes for multi-view region embedding. MGFN [37] constructs multiple mobility graphs to capture diverse movement patterns. RegionDCL (RDCL) [26] uses dual contrastive learning over building footprints and POI information. HREP [8] models heterogeneous urban relations with relation-aware graph convolution. HAFusion (HAF) [28] integrates intra-view, inter-view, and cross-region correlations through attentive feature learning and fusion. These base-

All methods are evaluated under the same data split and downstream regression protocol. Models are trained for at most 2000 epochs with early stopping based on validation performance. We use Adam with an initial learning rate of $5 \times 1 0 ^ { - 4 }$ and weight decay for regularization.

TABLE IV ABLATION STUDIES FOR CURE.
<table><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>Method</td><td rowspan=1 colspan=3>NY</td><td rowspan=1 colspan=3>Chi</td><td rowspan=1 colspan=3>SF</td></tr><tr><td rowspan=1 colspan=1>MAE (↓)</td><td rowspan=1 colspan=1>RMSE (↓)</td><td rowspan=1 colspan=1> $R ^ { 2 } \left( \uparrow \right)$ </td><td rowspan=1 colspan=1>MAE (↓)</td><td rowspan=1 colspan=1>RMSE (↓)</td><td rowspan=1 colspan=1> $R ^ { 2 } \left( \uparrow \right)$ </td><td rowspan=1 colspan=1>MAE (↓)</td><td rowspan=1 colspan=1>RMSE (↓)</td><td rowspan=1 colspan=1> $R ^ { 2 } \left( \uparrow \right)$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>w/o GIVE</td><td rowspan=1 colspan=1> $1 9 4 . 9 \pm 3 . 4$ </td><td rowspan=1 colspan=1> $\left| 2 9 2 . 9 \pm 5 . 3 \right.$ </td><td rowspan=1 colspan=1> $\left| 0 . 8 7 2 \pm 0 . 0 0 5 \right.$ </td><td rowspan=1 colspan=1> $1 0 0 0 \pm 7 5 . 4$ </td><td rowspan=1 colspan=1> $1 9 0 6 \pm 6 4 . 4$ </td><td rowspan=1 colspan=1> $\lvert 0 . 8 7 6 \pm 0 . 0 0 8$ </td><td rowspan=1 colspan=1> $\left| 2 2 8 . 1 \pm 8 . 2 \right.$ </td><td rowspan=1 colspan=1>1 $3 9 3 . 5 \pm 1 8 . 6 $ </td><td rowspan=1 colspan=1> $0 . 8 4 4 \pm 0 . 0 1 5$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>w/o CIVI</td><td rowspan=1 colspan=1> $1 9 5 . 7 \pm 5 . 9$ </td><td rowspan=1 colspan=1> $\lvert 3 1 1 . 7 \pm 1 0 . 6$ </td><td rowspan=1 colspan=1>0.855 ± 0.010|</td><td rowspan=1 colspan=1>998 ± 90.4</td><td rowspan=1 colspan=1> $1 9 1 2 \pm 1 5 0 . 5$ </td><td rowspan=1 colspan=1>0.874 ± 0.019|</td><td rowspan=1 colspan=1>242.9 ± 8.5</td><td rowspan=1 colspan=1> $4 3 0 . 9 \pm 2 1 . 3$ </td><td rowspan=1 colspan=1> $0 . 8 1 3 \pm 0 . 0 1 8$ </td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>w/o HGRF</td><td rowspan=1 colspan=1> $\left. 2 0 9 . 0 \pm 1 . 9 \right.$ </td><td rowspan=1 colspan=1> $3 2 7 . 5 \pm 3 . 4$ </td><td rowspan=1 colspan=1> $\left| 0 . 8 3 9 \pm 0 . 0 0 3 \right.$ </td><td rowspan=1 colspan=1> $1 0 3 7 \pm 6 8 . 3$ </td><td rowspan=1 colspan=1> $1 9 4 7 \pm 7 2 . 4$ </td><td rowspan=1 colspan=1> $\left| 0 . 8 7 0 \pm 0 . 0 1 0 \right.$ </td><td rowspan=1 colspan=1> $2 2 7 . 8 \pm 4 . 8$ </td><td rowspan=1 colspan=1> $4 2 1 . 1 \pm 1 9 . 5$ </td><td rowspan=1 colspan=1> $0 . 8 2 1 \pm 0 . 0 1 7$ </td></tr><tr><td rowspan=1 colspan=1>CURE</td><td rowspan=1 colspan=1> $1 8 6 . 7 \pm 2 . 2$ </td><td rowspan=1 colspan=1> $\mathbf { \left| 2 8 9 . 0 \pm 4 . 6 \right. }$ </td><td rowspan=1 colspan=1>0.875 ± 0.004</td><td rowspan=1 colspan=1>762 ± 56</td><td rowspan=1 colspan=1> $\mathbf { 1 5 6 9 } \pm \mathbf { 1 9 }$ </td><td rowspan=1 colspan=1> $\mathbf { \left| 0 . 9 1 6 \pm 0 . 0 1 1 \right| }$ </td><td rowspan=1 colspan=1> $2 1 5 . 8 \pm 5 . 5$ </td><td rowspan=1 colspan=1> $3 7 3 . 4 \pm 6 . 0$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 8 6 0 \pm 0 . 0 0 4 }$ </td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>w/o GIVE</td><td rowspan=1 colspan=1> $6 0 . 0 \pm 2 . 0$ </td><td rowspan=1 colspan=1> $\lvert 8 1 . 5 \pm 2 . 4$ </td><td rowspan=1 colspan=1>0.696 ± 0.018|</td><td rowspan=1 colspan=1>83.0 ± 4.8</td><td rowspan=1 colspan=1> $1 1 0 . 6 \pm 5 . 6$ </td><td rowspan=1 colspan=1> $\left| 0 . 6 0 5 \pm 0 . 0 4 0 \right.$ </td><td rowspan=1 colspan=1> $1 0 3 . 1 \pm 2 . 6$ </td><td rowspan=1 colspan=1> $1 7 4 . 9 \pm 3 . 6$ </td><td rowspan=1 colspan=1> $\left| 0 . 6 9 4 \pm 0 . 0 1 3 \right.$ </td></tr><tr><td rowspan=1 colspan=1>w/o CIVI</td><td rowspan=1 colspan=1> $\left| 5 7 . 4 \pm 1 . 3 \right.$ </td><td rowspan=1 colspan=1> $\left| 7 9 . 9 \pm 1 . 8 \right.$ </td><td rowspan=1 colspan=1>0.708 ± 0.013</td><td rowspan=1 colspan=1> $7 8 . 4 \pm 4 . 6$ </td><td rowspan=1 colspan=1> $1 0 7 . 5 \pm 5 . 8$ </td><td rowspan=1 colspan=1> $\left| 0 . 6 2 7 \pm 0 . 0 4 1 \right.$ </td><td rowspan=1 colspan=1> $1 0 7 . 5 \pm 3 . 5$ </td><td rowspan=1 colspan=1> $1 8 6 . 6 \pm 2 . 7$ </td><td rowspan=1 colspan=1> $\left| 0 . 6 5 2 \pm 0 . 0 1 0\right.$ </td></tr><tr><td rowspan=2 colspan=1>Crrime</td><td rowspan=1 colspan=1>w/o HGRF</td><td rowspan=1 colspan=1> $\lvert 5 7 . 0 \pm 0 . 3$ </td><td rowspan=1 colspan=1> $\left| 7 9 . 0 \pm 1 . 7 \right.$ </td><td rowspan=1 colspan=1>0.714 ± 0.012</td><td rowspan=1 colspan=1> $\left| 7 8 . 0 \pm 1 . 0 \right.$ </td><td rowspan=1 colspan=1> $\left| 1 0 7 . 8 \pm 3 . 9 \right.$ </td><td rowspan=1 colspan=1> $\left| 0 . 6 2 5 \pm 0 . 0 2 7 \right.$ </td><td rowspan=1 colspan=1> $1 0 5 . 6 \pm 5 . 1$ </td><td rowspan=1 colspan=1> $1 7 6 . 4 \pm 0 . 7$ </td><td rowspan=1 colspan=1> $\left| 0 . 6 8 9 \pm 0 . 0 0 2\right.$ </td></tr><tr><td rowspan=1 colspan=1>CURE</td><td rowspan=1 colspan=1> $\left. 5 3 . 2 \pm 1 . 5 \right.$ </td><td rowspan=1 colspan=1> $\mathbf { \left| 7 3 . 1 \pm 0 . 9 \right. }$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 5 5 \pm 0 . 0 0 6 } \mathb</td><td rowspan=1 colspan=1>f { \lvert 6 9 . 6 \pm 3 . 2 }$ </td><td rowspan=1 colspan=1> $\mathbf { \left| 9 1 . 2 \pm 0 . 9 \right. }$ </td><td rowspan=1 colspan=1> $\mathbf { \lvert 0 . 7 3 2 \ t 0 . 0 0 5 }$ </td><td rowspan=1 colspan=1> $\mathbf { \lvert 9 9 . 1 \pm 0 . 6 }$ </td><td rowspan=1 colspan=1> ${ \bf 1 6 8 . 0 \pm 1 . 5 }$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 1 8 \pm 0 . 0 0 5 }$ </td></tr><tr><td rowspan=1 colspan=1> $|</td><td rowspan=1 colspan=1>_ { \mathsf { W } } / \mathsf { o }$ GIVE</td><td rowspan=1 colspan=1> $1 2 6 7 \pm 9 . 6$ </td><td rowspan=1 colspan=1> $1 9 2 0 \pm 1 7 . 8$ </td><td rowspan=1 colspan=1>0.510 ± 0.009</td><td rowspan=1 colspan=1> $1 6 3 . 7 \pm 7 . 2$ </td><td rowspan=1 colspan=1> $\lvert 2 2 6 . 1 \pm 4 . 1$ </td><td rowspan=1 colspan=1> $\left| 0 . 6 0 3 \pm 0 . 0 1 4 \right.$ </td><td rowspan=1 colspan=1> $\left. 8 2 . 6 \pm 5 . 0 \right.$ </td><td rowspan=1 colspan=1>137.8 ± 10.0|</td><td rowspan=1 colspan=1>0.633 ± 0.054</td></tr><tr><td rowspan=3 colspan=1>See rvvll $\l</td><td rowspan=1 colspan=1>eft| \mathrm { w } / \mathrm { o } \right|$ CIVI</td><td rowspan=1 colspan=1> $1 2 7 8 \pm 6 . 4$ </td><td rowspan=1 colspan=1> $1 9 6 5 \pm 4 . 3$ </td><td rowspan=1 colspan=1> $\left| 0 . 4 8 6 \pm 0 . 0 0 2\right.$ </td><td rowspan=1 colspan=1> $1 6 1 . 4 \pm 2 . 9$ </td><td rowspan=1 colspan=1> $\vert 2 2 5 . 7 \pm 1 . 7$ </td><td rowspan=1 colspan=1> $\left| 0 . 6 0 5 \pm 0 . 0 0 6 \right.$ </td><td rowspan=1 colspan=1> $\mathbf { 8 4 . 0 \pm 2 . 3 }$ </td><td rowspan=1 colspan=1> $1 4 5 . 0 \pm 4 . 2$ </td><td rowspan=1 colspan=1> $\left| 0 . 5 9 6 \pm 0 . 0 2 3 \right.$ </td></tr><tr><td rowspan=1 colspan=1>w/o HGRF</td><td rowspan=1 colspan=1> $1 3 4 7 \pm 7 5 . 8$ </td><td rowspan=1 colspan=1> $2 0 3 1 \pm 9 0 . 1$ </td><td rowspan=1 colspan=1> $\lvert 0 . 4 5 0 \pm 0 . 0 5 0$ </td><td rowspan=1 colspan=1> $1 5 3 . 8 \pm 5 . 9$ </td><td rowspan=1 colspan=1> $\lvert 2 1 4 . 0 \pm 5 . 7$ </td><td rowspan=1 colspan=1> $\left| 0 . 6 4 4 \pm 0 . 0 1 9 \right.$ </td><td rowspan=1 colspan=1> $\lceil 7 8 . 7 \pm 2 . 8$ </td><td rowspan=1 colspan=1> $1 3 4 . 1 \pm 7 . 8$ </td><td rowspan=1 colspan=1> $\left| 0 . 6 5 4 \pm 0 . 0 4 1 \right.$ </td></tr><tr><td rowspan=1 colspan=1>CURE</td><td rowspan=1 colspan=1> $\mathbf { | } 1 2 4 4 \pm 1 2$ </td><td rowspan=1 colspan=1> $\mathbf { \left| 1 9 1 3 \pm 4 \right| }$ </td><td rowspan=1 colspan=1> $\mathbf { \lvert 0 . 5 1 3 \pm 0 . 0 0 2 }$ </td><td rowspan=1 colspan=1> $1 3 1 . 0 \pm 2 . 8$ </td><td rowspan=1 colspan=1> $1 8 4 . 3 \pm 2 . 2$ </td><td rowspan=1 colspan=1> $\mathbf { \left| 0 . 7 3 6 \pm 0 . 0 0 6 \right| }$ </td><td rowspan=1 colspan=1> $\mathbf { 7 3 . 8 \pm 1 . 6 }$ </td><td rowspan=1 colspan=1> $1 2 0 . 3 \pm 1 . 5$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 2 2 \pm 0 . 0 0 7 }$ </td></tr></table>

Table III, CURE achieves the best results on all three datasets. In NY, it reduces MAE from 1273 to 1244 and RMSE from 1951 to 1913 over HAF, while increasing $R ^ { 2 }$ from 0.493 to 0.513. Larger improvements are observed in Chi, where $R ^ { 2 }$ increases from 0.613 to 0.736 and RMSE decreases from 222.0 to 184.3. In SF, CURE improves $R ^ { 2 }$ from 0.612 to 0.722 and reduces RMSE from 142.1 to 120.3. These results demonstrate that CURE is effective for heterogeneous urban analytics tasks with different spatial regularities and uncertainty levels.

## B. Main Results

Check-in Prediction. We first evaluate all methods on checkin prediction, which measures whether the learned representations capture human mobility and location-activity patterns. As shown in Table III, CURE consistently achieves the best performance across all cities and metrics. Compared with the strongest baseline HAF, CURE reduces MAE from 202.8 to 186.7 and RMSE from 322.8 to 289.0 in NY, while improving $R ^ { 2 }$ from 0.844 to 0.875. In Chi, CURE improves $R ^ { 2 }$ from 0.870 to 0.916 and reduces RMSE from 1947 to 1569. In SF, it increases $R ^ { 2 }$ from 0.813 to 0.860. These results indicate that CURE more effectively captures mobility-related urban semantics and cross-region dependencies than correlationdriven baselines.

## C. Evaluation on the Chengdu Dataset

Crime Forecasting. We next evaluate crime forecasting, which requires modeling public-safety-related spatial patterns and socio-environmental context. Table III shows that CURE again outperforms all baselines. In NY, CURE improves $R ^ { 2 }$ from 0.734 to 0.755 over HAF, while reducing MAE from 56.1 to 53.2 and RMSE from 76.1 to 73.1. The improvement is more substantial in Chi, where $R ^ { 2 }$ increases from 0.631 to 0.732 and RMSE decreases from 107.1 to 91.2. In SF, CURE achieves the highest $R ^ { 2 }$ of 0.718. These gains suggest that confounder-aware multi-view integration helps capture localized urban risks and spatially dependent social activities. Service Call Prediction. Finally, we evaluate service call prediction, which reflects civic service demand and operational urban needs. This task is challenging because service requests are often event-driven and spatially irregular. As reported in

We additionally evaluate CURE on the Chengdu dataset, which contains 836 regions. The dataset provides three types of regional information, namely mobility outflow, mobility inflow, and POI attributes, which can be naturally regarded as three complementary views under the CURE framework. For each view, we construct a view-specific regional graph and apply CURE’s shared-component estimation, residualization, and multi-view fusion mechanisms. We evaluate the resulting representations on three downstream urban socioeconomic prediction tasks: carbon emissions, GDP, and population.

As shown in Table V, CURE consistently achieves competitive performance across all three tasks, demonstrating that the proposed framework remains effective when applied to a substantially larger and heterogeneous multi-view urban dataset.

TABLE V  
PERFORMANCE COMPARISON ON THE CHENGDU (CD) DATASET.
<table><tr><td></td><td colspan="3">Carbon</td><td colspan="3">GDP</td><td colspan="3">Population</td></tr><tr><td>Method</td><td>MAE</td><td>RMSE</td><td> $\scriptstyle \mathbf { R } ^ { 2 }$ </td><td>MAE</td><td>RMSE</td><td> $\scriptstyle \mathbf { R } ^ { 2 }$ </td><td>MAE</td><td>RMSE</td><td> $\scriptstyle \mathbf { R } ^ { 2 }$ </td></tr><tr><td>HAF</td><td>81.4</td><td>145.7</td><td>0.189</td><td>198.1</td><td>350.2</td><td>0.168</td><td>807.6</td><td>1366.3</td><td>0.108</td></tr><tr><td>ComSRE [49]</td><td>62.3</td><td>119.5</td><td>0.412</td><td>161.6</td><td>307.9</td><td>0.313</td><td>600.2</td><td>1024.5</td><td>0.432</td></tr><tr><td>CURE</td><td>58.2</td><td>113.5</td><td>0.466</td><td>158.4</td><td>302.5</td><td>0.358</td><td>598.3</td><td>1021.8</td><td>0.477</td></tr></table>

D. Comparison with Shared–Private and Adversarial Multi-View Methods

Confounder-aware cross-view interaction is a key distinction of CURE. Existing urban region representation methods primarily focus on multi-view fusion and cross-view interaction, without explicitly accounting for potentially misleading dependencies induced by latent shared factors. To provide a more direct comparison with alternative multi-view representation paradigms, we further consider three representative methods beyond the urban-specific baselines: MISA [50], which performs shared–private representation learning; Domain Separation Networks (DSN) [51], which employs adversarial learning for shared/private feature separation; and MEGAN [52], which adopts adversarial learning for multi-view network representation.

All methods are evaluated on the same datasets, downstream tasks, and evaluation protocol as CURE. As shown in Table VI, CURE consistently achieves the best overall performance across New York, Chicago, and San Francisco. These results demonstrate that merely separating shared and private representations or employing adversarial multi-view learning is insufficient to address potentially confounding cross-view dependencies. In contrast, CURE explicitly attenuates latent shared variation before cross-view interaction while preserving view-specific urban graph structures, leading to more robust regional representations.

## E. Ablation Study

To assess the contribution of each component in CURE, we compare the full model with three ablated variants: w/o GIVE removes graph-guided intra-view encoding (GIVE); w/o CIVI removes confounder-aware inter-view interaction (CIVI); and $\mathtt { w / o }$ HGRF removes hierarchical graph-aware residual fusion (HGRF). As shown in Table IV, the full CURE consistently outperforms all variants across tasks, cities, and metrics, confirming the effectiveness of the proposed design. Effect of GIVE. Removing GIVE leads to consistent performance degradation, especially in Chi. For example, on checkin prediction, MAE increases from 762 to 1000 and $R ^ { 2 }$ drops from 0.916 to 0.876. Similar drops are observed in crime forecasting and service call prediction, where Chi $R ^ { 2 }$ decreases from 0.732 to 0.605 and from 0.736 to 0.603, respectively. The results indicate that graph-guided intra-view encoding is essential for preserving view-specific spatial structures and local urban dependencies.

Effect of CIVI. The variant w/o CIVI also underperforms the full model on all tasks. In check-in prediction, $R ^ { 2 }$ decreases from 0.875 to 0.855 in NY, from 0.916 to 0.874 in Chi, and from 0.860 to 0.813 in SF. The degradation is also evident in service call prediction, where SF $R ^ { 2 }$ drops from 0.722 to 0.596. These results demonstrate that confounder-aware inter-view interaction improves the reliability of multi-view integration by mitigating spurious cross-view correlations.

Effect of HGRF. Removing HGRF causes notable declines, particularly on check-in prediction and service call prediction. For check-in prediction, NY MAE increases from 186.7 to 209.0, while Chi MAE increases from 762 to 1037. For service call prediction in NY, $R ^ { 2 }$ decreases from 0.513 to 0.450. These results verify that hierarchical graph-aware residual fusion is important for integrating multi-level urban semantics while retaining useful low-level structural information.

Comparison with Alternative Cross-View Interaction Designs: To further justify the proposed cross-view interaction design, we compare CURE with two simpler alternatives. Direct Concatenation (DC) directly concatenates the representations learned from different views without explicit cross-view interaction, while Cross-Attention (CA) replaces the proposed confounder-aware interaction mechanism with standard crossview attention.

As shown in Table VII, CURE consistently outperforms both DC and CA across New York, Chicago, and San Francisco. These results suggest that the performance gain cannot be attributed solely to increased fusion or interaction capacity. Instead, explicitly attenuating shared latent variation before cross-view interaction leads to more effective and robust multiview regional representations.

## F. Parameter Sensitivity Analysis

Effects of Region Embedding Size. We vary the embedding size d in {36, 72, 96, 144, 288} to examine the sensitivity of CURE to representation dimensionality. As shown in Fig. 3, small embeddings generally yield larger MAE and RMSE, indicating insufficient capacity to encode heterogeneous urban semantics and spatial dependencies. Increasing d improves performance in most cases, but the gains saturate at moderate dimensionalities. Across tasks and cities, d = 96 and d = 144 usually achieve the best or near-best results, whereas d = 288 does not consistently provide further improvement. This suggests that overly large embeddings may introduce redundant dimensions and increase overfitting risk. We therefore select a moderate embedding size in the main experiments to balance expressiveness and generalization.

(a) NY  
TABLE VI  
COMPARISON WITH SHARED–PRIVATE AND ADVERSARIAL MULTI-VIEW REPRESENTATION METHODS ON NEW YORK (NY), CHICAGO (CHI), AND SAN FRANCISCO (SF).
<table><tr><td></td><td colspan="3">NY</td><td colspan="3">Chi</td><td colspan="3">SF</td></tr><tr><td>Method</td><td>MAE</td><td>RMSE</td><td> $\scriptstyle \mathbf { R } ^ { 2 }$ </td><td>MAE</td><td>RMSE</td><td> $\scriptstyle \mathbf { R } ^ { 2 }$ </td><td>MAE</td><td>RMSE</td><td> $\scriptstyle \mathbf { R } ^ { 2 }$ </td></tr><tr><td>MISA</td><td>196.9</td><td>317.5</td><td>0.849</td><td>854</td><td>1722</td><td>0.899</td><td>244.5</td><td>443.8</td><td>0.802</td></tr><tr><td>DSN</td><td>201.7</td><td>310.5</td><td>0.856</td><td>918</td><td>1960</td><td>0.869</td><td>233.7</td><td>417.8</td><td>0.824</td></tr><tr><td>MEGAN</td><td>209.2</td><td>327.3</td><td>0.840</td><td>975</td><td>1931</td><td>0.873</td><td>261.0</td><td>416.2</td><td>0.825</td></tr><tr><td>CURE</td><td>186.7</td><td>289.0</td><td>0.875</td><td>762</td><td>1569</td><td>0.916</td><td>215.8</td><td>373.4</td><td>0.860</td></tr></table>

TABLE VII  
COMPARISON WITH ALTERNATIVE CROSS-VIEW INTERACTION DESIGNS ON NEW YORK (NY), CHICAGO (CHI), AND SAN FRANCISCO (SF).
<table><tr><td></td><td colspan="3">NY</td><td colspan="3">Chi</td><td colspan="3">SF</td></tr><tr><td>Method</td><td>MAE</td><td>RMSE</td><td> $\scriptstyle \mathbf { R } ^ { 2 }$ </td><td>MAE</td><td>RMSE</td><td> $\mathbf { R ^ { 2 } }$ </td><td>MAE</td><td>RMSE</td><td> $\scriptstyle \mathbf { R } ^ { 2 }$ </td></tr><tr><td>DC</td><td>206.4</td><td>319.2</td><td>0.848</td><td>897</td><td>1749</td><td>0.896</td><td>265.1</td><td>438.3</td><td>0.806</td></tr><tr><td>CA</td><td>194.7</td><td>317.3</td><td>0.850</td><td>795</td><td>1643</td><td>0.907</td><td>240.6</td><td>434.1</td><td>0.810</td></tr><tr><td>CURE</td><td>186.7</td><td>289.0</td><td>0.875</td><td>762</td><td>1569</td><td>0.916</td><td>215.8</td><td>373.4</td><td>0.860</td></tr></table>

![](images/d676461da3f83b5bfc618931b376cdb4fc92053ba2ba4981806e0aa26dad9115.jpg)  
Fig. 3. Effects of Embedding Size.

Effects of λ. Fig. 4 shows the effect of the deconfounding weight λ on the three datasets. In general, the model is stable across different values of λ, indicating low sensitivity to this hyperparameter. A moderate value of λ generally improves performance, especially for crime and service call prediction, suggesting that deconfounding helps reduce biased correlations and learn more robust representations. Across NY, Chi, and SF, Check-in consistently achieves the highest and most stable $R ^ { 2 }$ . For crime and service call, performance usually increases when λ grows from 0 to 0.5 or 0.7, but drops when λ becomes too large, $\mathrm { e . g . }$ , when $\lambda ~ = ~ 0 . 9$ This indicates that weak deconfounding may be insufficient, while overly strong deconfounding may remove useful taskrelated information. Taken together, $\lambda ~ = ~ 0 . 7$ achieves the best or near-best performance in most cases, providing a good balance between prediction accuracy and confounder-reduced representation learning. Therefore, we set $\lambda ~ = ~ 0 . 7$ as the default value in our experiments.

![](images/adc9006d891fadbeacd6176731fc045841d26bbbdb9a78356f9ef4badd2301e7.jpg)

![](images/4f300661aef358fd5690633030ae8340f66214ea08c7f4a4363ed19ffd5c3fd8.jpg)

![](images/e72e707dd0bcc0ff3030427044fc90b267ad730cc185f6c84605d66063326231.jpg)  
Fig. 4. Effects of Lambda (λ).

## G. Model Scalability

Scalability w.r.t. Number of Regions. We evaluate the scalability of CURE by varying the number of urban regions and reporting the downstream prediction performance. As shown in Fig. 5, the performance generally improves as the number of regions increases across cities and tasks. This trend indicates that using more regions provides richer spatial coverage and more complete inter-region dependencies, enabling the model to learn more informative urban representations. The improvement is particularly evident in NY, where the $R ^ { 2 }$ of check-in prediction and crime forecasting increases substantially as the number of regions grows from 36 to 180. Similar trends are observed in SF, where all three tasks benefit from larger region sets. In Chi check-in prediction already achieves relatively high performance with fewer regions, while crime forecasting and service call prediction continue to improve as more regions are included.

![](images/da796ba59cd85b66f9b76a0691b7266ca07ab7c8d56b33fb195637eba7348dd0.jpg)

![](images/acfae9deff9809769e3a8db32c404446ad0d76060db27ee6d5f5932b4b917392.jpg)  
Fig. 5. Scalability Analysis with respect to the Number of Regions.

![](images/7a18518fcbb45f924d56edb9eb264648c518a335f65d18ad007e7b96d5985b34.jpg)

Computational Scalability. Table VIII reports the computational cost of CURE on the check-in prediction task under different scale ratios. The scale ratio denotes the proportion of regions sampled from the original city-level region set. AET and PA denote the average training epoch time and peak GPU memory allocation, respectively. As the scale ratio increases from 0.2 to 1.0, both AET and PA increase consistently across the three datasets. This is expected because larger region sets lead to larger regional graphs and more pairwise region dependencies to be processed during graph-guided encoding and multi-view fusion. Nevertheless, the increase in training time remains moderate. For example, in NY, AET increases from 0.1269s to 0.1590s, while in SF it increases from 0.1153s to 0.1505s. This indicates that CURE can handle larger urban graphs without introducing prohibitive training overhead. The memory consumption shows a clearer growth pattern than training time. In NY, PA increases from 168.14MB to 328.91MB, and in SF it increases from 167.12MB to 319.89MB as the full region set is used. This trend is mainly due to the storage and propagation of multiple view-specific regional graphs, including mobility, POI, and land-use relations. Since graph-based operations depend on the number of regions and regional connections, larger spatial scales naturally require more GPU memory. Compared with NY and SF, Chi exhibits lower memory usage and slower growth in AET. This is because Chi contains fewer regions under the same scale ratios, leading to smaller graph structures and lower computation cost. Even at the full scale, Chi only requires 0.0839s per epoch and 117.40MB peak memory allocation.

## H. Robustness and Reliability Analysis

We evaluate CURE from two complementary perspectives: robustness, which tests whether the learned representations remain stable under missing views and noisy input features, and reliability, which examines whether cross-view integration suppresses spurious shared dependencies while preserving stable and task-relevant view-specific information.

TABLE VIII  
COMPUTATIONAL SCALABILITY OF CURE ON THE CHECK-IN PREDICTION TASK UNDER DIFFERENT SCALE RATIOS. AET DENOTES THE AVERAGE EPOCH TIME, AND PA DENOTES PEAK GPU MEMORY ALLOCATION.
<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=2>NY</td><td rowspan=1 colspan=2>Chi</td><td rowspan=1 colspan=2>SF</td></tr><tr><td rowspan=1 colspan=1>AET (s)</td><td rowspan=1 colspan=1>PA (MB)</td><td rowspan=1 colspan=1>AET (s)</td><td rowspan=1 colspan=1>PA (MB)</td><td rowspan=1 colspan=1>AET (s)</td><td rowspan=1 colspan=1>PA (MB)</td></tr><tr><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>0.1269</td><td rowspan=1 colspan=1>168.14</td><td rowspan=1 colspan=1>0.0763</td><td rowspan=1 colspan=1>88.54</td><td rowspan=1 colspan=1>0.1153</td><td rowspan=1 colspan=1>167.12</td></tr><tr><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>0.1315</td><td rowspan=1 colspan=1>193.75</td><td rowspan=1 colspan=1>0.0798</td><td rowspan=1 colspan=1>92.98</td><td rowspan=1 colspan=1>0.1203</td><td rowspan=1 colspan=1>191.55</td></tr><tr><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>0.1393</td><td rowspan=1 colspan=1>229.16</td><td rowspan=1 colspan=1>0.0809</td><td rowspan=1 colspan=1>100.03</td><td rowspan=1 colspan=1>0.1308</td><td rowspan=1 colspan=1>224.47</td></tr><tr><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=1>0.1546</td><td rowspan=1 colspan=1>277.14</td><td rowspan=1 colspan=1>0.0825</td><td rowspan=1 colspan=1>107.45</td><td rowspan=1 colspan=1>0.1417</td><td rowspan=1 colspan=1>266.38</td></tr><tr><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>0.1590</td><td rowspan=1 colspan=1>328.91</td><td rowspan=1 colspan=1>0.0839</td><td rowspan=1 colspan=1>117.40</td><td rowspan=1 colspan=1>0.1505</td><td rowspan=1 colspan=1>319.89</td></tr></table>

![](images/360f34be7dd2ee238e067cec9a0bba33240c38691a891297ff255e12b71b3d99.jpg)

![](images/0b12c7fe0efb3b3aa06c287c951c2d08f1244de4ca4f9d7a46be6ed7e2bbb106.jpg)

![](images/36477f40c25e44aad9d83393b7f420d2548680fb684fcae5f6a2e78251682a9e.jpg)

![](images/99d03ce093d26e3ba3b5f137997c5a5d1ecf40364f2154d13b22f6309e80c02e.jpg)

![](images/a5eefb552d419f688df8e33ee4b9eb26c4e8f78fa5b019c206ac32c4964b812e.jpg)

![](images/abbc2167713105223025edfa296ea16ed28e9f8cfdf41548c876ff0a3ded7822.jpg)

![](images/05e56111f2d7d7ed3dd4ce08c3be21d9a2d4b6b6d12d3815b52fd23992da4346.jpg)

![](images/c77c547f6e89887d81515100c9c18caa203af2421e4db896e4debba98f85fd5f.jpg)

![](images/d4426b5e9dfb4933a29543b831768bcb1128edd28660ec50735ad16af23c7107.jpg)  
Fig. 6. Effects of missing views.

Robustness to Missing Views. To evaluate robustness under incomplete urban observations, we mask one input view at a time during inference, including mobility, POI, and landuse views. As shown in Fig. 6, removing any view generally degrades performance, indicating that all three views contribute useful information to region representation learning. The performance drop is often most pronounced when the mobility view is missing, especially for check-in prediction and service call prediction. This is expected because mobility flows directly capture human movement intensity and inter-region interactions, which are closely related to location activity and urban service demand.

In contrast, missing POI or land-use information usually causes a smaller degradation. This suggests that POI and landuse views provide complementary functional and planningoriented semantics, while mobility supplies more direct dynamic signals for downstream prediction. Nevertheless, CURE maintains reasonable performance when a single view is removed, showing that it can exploit the remaining views rather than depending entirely on one data source.

TABLE IX SHARED COMPONENT ANALYSIS OF CURE.  
![](images/8ab08ef368147bc248fd821565c8fac8efde0466180c6afbe3cf0a714e7c4f25.jpg)  
Fig. 7. Effects of feature noise.

Robustness to Feature Noise. We next examine robustness to noisy urban observations by injecting Gaussian noise into each view separately. The noise ratio is varied from 0.05 to 0.5, where larger values indicate stronger perturbations. Fig. 7 shows that the performance generally decreases as the noise ratio increases, confirming that noisy urban observations can weaken region representation quality. However, the degradation patterns differ across views, tasks, and cities. Perturbing the mobility view often leads to a sharper decline, particularly in tasks that are strongly associated with human movement and spatial interaction. POI and land-use perturbations tend to cause more moderate degradation in several cases, suggesting that semantic and planning-oriented views provide relatively stable contextual information.

Overall, CURE exhibits gradual rather than abrupt performance degradation under feature noise. This indicates that its graph-guided intra-view encoding and hierarchical residual fusion help preserve reliable signals from less corrupted views, while the confounder-aware interaction module reduces overreliance on unstable cross-view correlations. Together with the missing-view results, the feature-noise study verifies that CURE can learn robust urban region representations under incomplete and noisy multi-view data conditions.

Reliability of Shared Component Separation. We further examine whether the deconfounding module reduces sharedfactor-induced dependencies, which is a key aspect of reliability defined in the Introduction. Since the true latent factors are unobserved, we use two diagnostic metrics. For each view representation $H ^ { ( v ) }$ and its residual representation $R ^ { ( v ) }$ , the shared component ratio (SR), defined as $\| H ^ { ( v ) } - R ^ { ( v ) } \| _ { 2 } / \| H ^ { ( v ) } \| _ { 2 }$ , measures how much information is removed as the estimated shared component. The residual dependence (RD), measured by the mean absolute cosine similarity between $R ^ { ( v ) }$ and C, measures whether the residual representation remains aligned with the shared component. A non-trivial SR together with a small RD indicates that CURE separates shared signals and reduces their influence on residual cross-view interaction. Since representation learning is citylevel and task-agnostic, we report these statistics once for each city.

<table><tr><td>City | View</td><td>一</td><td>SR</td><td>RD</td></tr><tr><td rowspan="3">NY</td><td>POI</td><td> $0 . 2 2 6 7 \pm 0 . 0 9 5 3$ </td><td> $1 . 5 1 { \times } 1 0 ^ { - 8 } \pm 5 . 2 5 { \times } 1 0 ^ { - 9 }$ </td></tr><tr><td>Land-use</td><td> $0 . 2 1 2 6 \pm 0 . 0 7 2 7$ </td><td> $1 . 4 1 \times 1 0 ^ { - 8 } \pm 4 . 4 1 \times 1 0 ^ { - 9 }$ </td></tr><tr><td>Mobility</td><td> $0 . 1 7 5 7 \pm 0 . 0 4 9 4$ </td><td> $1 . 2 1 \times 1 0 ^ { - 8 } \pm 2 . 5 4 \times 1 0 ^ { - 9 }$ </td></tr><tr><td rowspan="3">Chi</td><td>POI</td><td> $0 . 1 6 8 5 \pm 0 . 0 6 7 7$ </td><td> $1 . 2 2 \times 1 0 ^ { - 8 } \pm 3 . 5 2 { \times } 1 0 ^ { - 9 }$ </td></tr><tr><td>Land-use</td><td> $0 . 2 7 2 3 \pm 0 . 1 4 9 7$ </td><td> $1 . 8 1 { \times } 1 0 ^ { - 8 } \pm 9 . 8 8 { \times } 1 0 ^ { - 9 }$ </td></tr><tr><td>Mobility</td><td> $0 . 2 3 1 1 \pm 0 . 0 9 8 7$ </td><td> $1 . 4 8 \times 1 0 ^ { - 8 } \pm 5 . 2 3 \times 1 0 ^ { - 9 }$ </td></tr><tr><td rowspan="2">SF</td><td>POI Land-use</td><td> $0 . 1 7 8 0 \pm 0 . 0 7 1 9$   $0 . 1 9 0 7 \pm 0 . 0 5 2 3$ </td><td> $1 . 2 6 \times 1 0 ^ { - 8 } \pm 3 . 8 6 \times 1 0 ^ { - 9 }$   $1 . 3 0 { \times } 1 0 ^ { - 8 } \pm 2 . 8 8 { \times } 1 0 ^ { - 9 }$ </td></tr><tr><td>Mobility</td><td> $0 . 1 6 8 8 \pm 0 . 0 7 7 9$ </td><td> $1 . 1 8 \times 1 0 ^ { - 8 } \pm 3 . 7 3 \times 1 0 ^ { - 9 }$ </td></tr></table>

As shown in Table IX, all views contain non-negligible shared components, indicating that heterogeneous urban views are not independent but are jointly influenced by latent urban factors. The SR values vary across cities and views. In NY, POI has the largest SR, suggesting that functional urban semantics are more strongly aligned with the estimated shared factor. In Chi, land-use and mobility show higher SR values, indicating stronger coupling between static urban structure, movement patterns, and shared city-level factors. In SF, the SR values are relatively balanced across views, suggesting more evenly distributed shared information among heterogeneous urban signals. The RD values are consistently close to zero across all cities and views. This confirms that the residualization step effectively removes the component aligned with the estimated shared factor, leaving residual representations that are nearly independent of the shared component.

Adaptive View Weighting Analysis. We further analyze the adaptive fusion weights learned by the hierarchical graphaware residual fusion module to examine context-dependent view contribution. These weights reflect how CURE allocates importance among confounder-reduced urban views when forming the final region representation. In the context of our reliability definition, such adaptive weighting complements shared component separation by showing whether the model further adjusts residual view contributions according to cityand task-specific contexts, rather than relying on fixed crossview integration.

As shown in Fig. 8, the learned weights vary across both cities and downstream tasks. In NY, check-in prediction assigns the largest weight to mobility, which is consistent with the close relation between check-in activity and human movement. Crime forecasting places more emphasis on POI, suggesting that regional functional composition provides useful signals for public-safety-related prediction. Service call prediction assigns the largest weight to land-use, reflecting the relevance of stable urban structure and planning-oriented attributes. Similar task- and city-dependent patterns are observed in Chi and SF. These results indicate that CURE adaptively integrates residual view representations according to city and task contexts, supporting its ability to model context-dependent view contribution in reliable multi-view data integration.

![](images/e62dec79b545d405e28811a2102e0d7ea28b8585cfc3817634b9cabd5427e5f2.jpg)

![](images/fae4526010f09a17468f6f32eecd5182881c327cb9206381e2040017b816eeef.jpg)

![](images/109e39250cde02c20b13b18756a989321add5a5d709ddd565f22246e5f5c4577.jpg)  
Fig. 8. Adaptive Fusion Weights Analysis.

## VI. CONCLUSION

We studied reliable multi-view urban data integration for urban region representation learning and proposed CURE, a confounder-aware framework for heterogeneous urban views. By combining graph-guided intra-view encoding, confounderaware inter-view interaction, and hierarchical graph-aware residual fusion, CURE preserves view-specific structures, reduces shared-factor-induced dependencies, and adaptively integrates stable and task-relevant residual view representations. Experiments on three cities and three tasks show that CURE improves predictive performance, remains robust to incomplete and noisy observations, and supports reliable integration through shared component separation and adaptive view weighting.

## REFERENCES

[1] L. Gong, S. Guo, Y. Lin, Y. Liu, E. Zheng, Y. Shuang, Y. Lin, J. Hu, and H. Wan, “STCDM: spatio-temporal contrastive diffusion model for check-in sequence generation,” IEEE Trans. Knowl. Data Eng., vol. 37, no. 4, pp. 2141–2154, 2025.

[2] S. B. Yang, Y. Sun, Y. Cheng, Y. Lin, K. Torp, and J. Hu, “Spatiotemporal trajectory foundation model - recent advances and future directions,” CoRR, vol. abs/2511.20729, 2025.

[3] S. B. Yang, C. Guo, J. Hu, J. Tang, and B. Yang, “Unsupervised path representation learning with curriculum negative sampling,” CoRR, vol. abs/2106.09373, 2021.

[4] S. B. Yang, C. Guo, J. Hu, B. Yang, J. Tang, and C. S. Jensen, “Weaklysupervised temporal path representation learning with contrastive curriculum learning,” in ICDE, 2022, pp. 2873–2885.

[5] C. Han, S. B. Yang, and J. Hu, “Diffmm: Efficient method for accurate noisy and sparse trajectory map matching via one step diffusion,” CoRR, vol. abs/2601.08482, 2026.

[6] S. B. Yang, J. Hu, C. Guo, B. Yang, and C. S. Jensen, “Lightpath: Lightweight and scalable path representation learning,” in KDD, 2023, pp. 2999–3010.

[7] Z. Pan, Y. Wang, Y. Zhang, S. B. Yang, Y. Cheng, P. Chen, C. Guo, Q. Wen, X. Tian, Y. Dou, Z. Zhou, C. Yang, A. Zhou, and B. Yang, “Magicscaler: Uncertainty-aware, predictive autoscaling,” Proc. VLDB Endow., vol. 16, no. 12, pp. 3808–3821, 2023.

[8] S. Zhou, D. He, L. Chen, S. Shang, and P. Han, “Heterogeneous region embedding with prompt learning,” in AAAI, 2023, pp. 4981–4989.

[9] L. Gong, H. Wan, S. Guo, X. Li, Y. Lin, E. Zheng, T. Wang, Z. Zhou, and Y. Lin, “Spatial-temporal cross-view contrastive pre-training for checkin sequence representation learning,” IEEE Trans. Knowl. Data Eng., vol. 36, no. 12, pp. 9308–9321, 2024.

[10] S. B. Yang, Y. Sun, Y. Cheng, Y. Lin, T. Kristian, and J. Hu, “Spatiotemporal trajectory foundation model-recent advances and future directions,” arXiv preprint arXiv:2511.20729, 2025.

[11] S. B. Yang, C. Guo, and B. Yang, “Context-aware path ranking in road networks,” TKDE, vol. 34, no. 7, pp. 3153–3168, 2022.

[12] Y. Wei, Y. Lin, H. Gao, R. Xu, S. B. Yang, and J. Hu, “Path-llm: A multi-modal path representation learning by aligning and fusing with large language models,” in WWW, 2025, pp. 2289–2298.

[13] R. Xu, H. Cheng, C. Guo, H. Gao, J. Hu, S. B. Yang, and B. Yang, “Mm-path: Multi-modal, multi-granularity path representation learning,” in KDD, 2025, pp. 1703–1714.

[14] S. B. Yang, C. Guo, J. Hu, B. Yang, J. Tang, and C. S. Jensen, “Weaklysupervised temporal path representation learning with contrastive curriculum learning - extended version,” CoRR, vol. abs/2203.16110, 2022.

[15] C. Han, S. B. Yang, and J. Hu, “Diffmm: Efficient method for accurate noisy and sparse trajectory map matching via one step diffusion,” in AAAI, 2026, pp. 14 783–14 791.

[16] S. B. Yang and B. Yang, “Learning to rank paths in spatial networks,” in ICDE, 2020, pp. 2006–2009.

[17] S. B. Yang, Y. Sun, J. Hu, Z. Xu, K. Torp, H. Lu, B. Yang, and C. S. Jensen, “Refine: Trajectory representation learning via closedlooptranscription,” in KDD, 2026.

[18] S. B. Yang, C. Guo, J. Hu, J. Tang, and B. Yang, “Unsupervised path representation learning with curriculum negative sampling,” in IJCAI, 2021, pp. 3286–3292.

[19] S. B. Yang and B. Yang, “Pathrank: A multi-task learning framework to rank paths in spatial networks,” CoRR, vol. abs/1907.04028, 2019.

[20] Z. Li, C. Huang, L. Xia, Y. Xu, and J. Pei, “Spatial-temporal hypergraph self-supervised learning for crime prediction,” in ICDE, 2022, pp. 2984– 2996.

[21] U. M. Butt, S. Letchmunan, M. Ali, and H. H. R. Sherazi, “START: A spatiotemporal autoregressive transformer for enhancing crime prediction accuracy,” IEEE Trans. Comput. Soc. Syst., vol. 12, no. 6, pp. 4650–4664, 2025.

[22] C. Wang, Z. Lin, X. Yang, J. Sun, M. Yue, and C. Shahabi, “HAGEN: homophily-aware graph convolutional recurrent network for crime fore casting,” in AAAI, 2022, pp. 4193–4200.

[23] X. Zhao, W. Fan, H. Liu, and J. Tang, “Multi-type urban crime prediction,” in AAAI. AAAI Press, 2022, pp. 4388–4396.

[24] S. Zhao, R. Liu, B. Cheng, and D. Zhao, “Classification-labeled continuousization and multi-domain spatio-temporal fusion for fine-grained urban crime prediction,” IEEE Trans. Knowl. Data Eng., vol. 35, no. 7, pp. 6725–6738, 2023.

[25] F. Sun, Y. Chang, E. Tanin, S. Karunasekera, and J. Qi, “Flexireg: Flexible urban region representation learning,” in KDD, 2025, pp. 2702– 2713.

[26] Y. Li, W. Huang, G. Cong, H. Wang, and Z. Wang, “Urban region representation learning with openstreetmap building footprints,” in KDD, 2023, pp. 1363–1373.

[27] M. Zhang, T. Li, Y. Li, and P. Hui, “Multi-view joint graph representation learning for urban region embedding,” in IJCAI, 2020, pp. 4431–4437.

[28] F. Sun, J. Qi, Y. Chang, X. Fan, S. Karunasekera, and E. Tanin, “Urban region representation learning with attentive fusion,” in ICDE, 2024, pp. 4409–4421.

[29] S. B. Yang, H. Miao, Z. Xu, J. Hu, X. Wang, H. Lu, B. Yang, and C. S. Jensen, “Dgcpath: Distribution-aware generative contrastive framework for self-supervised path representation learning – extended version,” CoRR, vol. abs/2609.07316, 2026.

[30] S. B. Yang, Y. Sun, J. Hu, Z. Xu, K. Torp, H. Lu, B. Yang, and C. S. Jensen, “Refine: Trajectory representation learning via closed-loop transcription–extended version,” CoRR, vol. abs/2609.07206, 2026.

[31] Z. Xu and X. Zhou, “CGAP: urban region representation learning with coarsened graph attention pooling,” in IJCAI, 2024, pp. 7518–7526.

[32] Y. Zhao, P. Deng, J. Liu, X. Jia, and M. Wang, “Causal conditional hidden markov model for multimodal traffic prediction,” in AAAI, 2023, pp. 4929–4936.

[33] Y. Xia, Y. Liang, H. Wen, X. Liu, K. Wang, Z. Zhou, and R. Zimmermann, “Deciphering spatio-temporal graph forecasting: A causal lens and treatment,” in NeurIPS, 2023.

[34] K. Wang, H. Wu, Y. Duan, G. Zhang, K. Wang, X. Peng, Y. Zheng, Y. Liang, and Y. Wang, “Nuwadynamics: Discovering and updating in causal spatio-temporal modeling,” in ICLR, 2024.

[35] C. Gong, C. Zhang, D. Yao, J. Bi, W. Li, and Y. Xu, “Causal discovery from temporal data: An overview and new perspectives,” ACM Comput. Surv., vol. 57, no. 4, pp. 100:1–100:38, 2025.

[36] W. Li, D. Yao, C. Gong, X. Chu, Q. Jing, X. Zhou, Y. Zhang, Y. Fan, and J. Bi, “Causaltad: Causal implicit generative model for debiased online trajectory anomaly detection,” in ICDE, 2024, pp. 4477–4490.

[37] S. Wu, X. Yan, X. Fan, S. Pan, S. Zhu, C. Zheng, M. Cheng, and C. Wang, “Multi-graph fusion networks for urban region embedding,” in IJCAI, 2022, pp. 2312–2318.

[38] P. Cui and S. Athey, “Stable learning establishes some common ground between causal inference and machine learning,” Nat. Mach. Intell., vol. 4, no. 2, pp. 110–115, 2022.

[39] Y. Zhang, S. B. Yang, A. Khan, and C. G. Akcora, “ATEX-CF: attackinformed counterfactual explanations for graph neural networks,” CoRR, vol. abs/2602.06240, 2026.

[40] F. Lv, J. Liang, S. Li, B. Zang, C. H. Liu, Z. Wang, and D. Liu, “Causality inspired representation learning for domain generalization,” in CVPR, 2022, pp. 8036–8046.

[41] Z. Chu, R. Li, S. L. Rathbun, and S. Li, “Continual causal inference with incremental observational data,” in ICDE, 2023, pp. 3430–3439.

[42] D. Zhou, B. Wu, K. Wang, Q. Yang, Y. Deng, and S. Yiu, “Interventiondriven correlation reduction: A data generation approach for achieving counterfactually fair predictors,” in ICDE, 2025, pp. 2066–2079.

[43] J. Liu, S. Xia, D. Alabi, and E. Wu, “Suna: Scalable causal confounder discovery over relational data,” PVLDB, vol. 18, no. 11, pp. 4158–4170, 2025.

[44] Y. Huang, Z. Fang, Z. Zeng, L. Chen, and Y. Gao, “Causal spatiotemporal prediction: An effective and efficient multi-modal approach,”

CoRR, vol. abs/2505.17637, 2025.

[45] K. Ahuja, D. Mahajan, Y. Wang, and Y. Bengio, “Interventional causal representation learning,” in ICML, 2023, pp. 372–407.

[46] E. Acarturk, B. Varici, K. Shanmugam, and A. Tajer, “Sample com-¨ plexity of interventional causal representation learning,” in NeurIPS , 2024.

[47] G. Rajendran, S. Buchholz, B. Aragam, B. Scholkopf, and P. Ravikumar,¨ “From causal to concept-based representation learning,” in NeurIPS, 2024.

[48] S. Kumar, M. Srivastava, and V. Prakash, “Comparative analysis of arima, deep learning, and lasso regression models for time series forecasting: Assessing accuracy, robustness, and computational efficiency,” in AIDS, vol. 3619, 2023, pp. 12–22.

[49] Z. Li, H. Jia, K. Zhao, W. Huang, and M. Chen, “Multi-view urban region embedding via commonality-specificity disentanglement,” in KDD, 2026, pp. 748–758.

[50] D. Hazarika, R. Zimmermann, and S. Poria, “MISA: modality-invariant and -specific representations for multimodal sentiment analysis,” in MM, 2020, pp. 1122–1131.

[51] K. Bousmalis, G. Trigeorgis, N. Silberman, D. Krishnan, and D. Erhan, “Domain separation networks,” in NeurIPS, 2016, pp. 343–351.

[52] Y. Sun, S. Wang, T. Hsieh, X. Tang, and V. G. Honavar, “MEGAN: A generative adversarial network for multi-view network embedding,” in IJCAI, 2019, pp. 3527–3533.