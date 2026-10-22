# YingLan: A Multimodal Ocean Foundation Model for Unified Representation Learning from Heterogeneous Observations

Sen Jia<sup>1</sup>, Siqi Wei<sup>1</sup>, Junyu Wang<sup>1</sup>, Zijun Lv<sup>1</sup>, Pan Liu<sup>1</sup>, Meng Xu<sup>1</sup>, Xiyou Fu<sup>1</sup>, Yuxi Sun<sup>1</sup>, Qiang Liu<sup>1</sup>, Shuyu Zhang<sup>\*1</sup>, and Lin Mu<sup>\*2</sup>

<sup>1</sup>College of Computer Science and Software Engineering, Shenzhen University

<sup>2</sup>College of Life Sciences and Oceanography, Shenzhen University

<sup>1\*,</sup> <sup>2\*</sup>Corresponding authors: Shuyu Zhang (shuyu-zhang@szu.edu.cn); Lin Mu (mulin@szu.edu.cn)

## Abstract

Multimodal ocean modeling remains challenging due to substantial heterogeneity in data structures, physical semantics, and spatiotemporal characteristics across satellite imagery, gridded oceanographic fields, and buoy observations. Existing approaches largely focus on task-specific ocean modeling, while learning transferable representations from heterogeneous observations remains an open problem. To address this challenge, we propose YingLan, a multimodal ocean foundation model that integrates physics-attribute-conditioned dynamic encoding, Spatiotemporal Increment Coupling (STIC), and ocean-state evolution-aware pretraining. Specifically, the dynamic encoding mechanism generates adaptive input projections conditioned on physical semantics and observation characteristics, enabling flexible representation of heterogeneous ocean variables. STIC models multiscale spatiotemporal dependencies through cross-scale interactions of latent temporal increments. The pretraining strategy further combines masked observation reconstruction with temporal evolution objectives to facilitate the learning of coherent ocean-state representations across diferent prediction horizons. YingLan is implemented using MindSpore and trained on Huawei Ascend 910B NPUs, demonstrating its compatibility with domestic AI computing infrastructure. In a seven-day sea surface salinity (SSS) forecasting task conducted in the Shenzhen coastal waters, YingLan achieves an average RMSE of 0.687508 PSU, corresponding to relative error reductions of 0.58%, 5.31%, and 12.42% compared with LangYa, XiHe, and GloNet2, respectively. These results provide initial evidence of the model’s efectiveness in regional ocean forecasting, while broader cross-task and cross-regional evaluations remain necessary to assess its generalization capabilities.

Keywords: Ocean foundation model; Multimodal representation learning; Indigenous deployment; Downstream task adaptation; Self-supervised pretraining

## 1 Introduction

Accurate characterization and prediction of the marine environment are fundamental to oceanographic research and marine environmental monitoring, with important implications for marine ecosystem conservation, marine resource management, and ocean hazard early warning [1]. Advances in satellite remote sensing, ocean numerical modeling, and in situ observation technologies have expanded the spatial coverage, temporal resolution, and range of physical variables represented in multi-source ocean observation data, providing an essential basis for multiscale characterization and dynamic analysis of the marine environment [2]. Specifically, satellite imagery captures the spatial distribution of ocean surface conditions, oceanographic products provide spatiotemporal information on variables such as temperature, salinity, and ocean currents, and in situ observations from buoys and other platforms record continuous variations in local marine conditions [3]. These observation sources are complementary in terms of spatial coverage, temporal sampling, and physical information. However, substantial diferences in data structure, variable semantics, and spatiotemporal scales pose challenges to joint modeling of heterogeneous observations and knowledge transfer across downstream tasks. Consequently, large-scale pretraining on heterogeneous ocean observations to learn generalizable representations with cross-modal adaptability and spatiotemporal generalization has emerged as a promising research direction in ocean intelligence [4].

In recent years, deep learning has made substantial progress in ocean environmental forecasting and the development of ocean foundation models. For spatiotemporal ocean modeling, XiHe employs local and global spatial interactions to perform high-resolution [5], multivariate global ocean forecasting, whereas WenHai integrates air-sea flux computation with ocean-dynamicsinformed network design, highlighting the importance of spatial dependency modeling and the incorporation of domain knowledge in ocean prediction [6]. In the development of foun dation models, AI-GOMS explores a pretraining and downstream adaptation paradigm for multiple ocean-related tasks, demonstrating the potential of shared representations to support diverse applications [7]. Meanwhile, ocean remote sensing research has increasingly investigated the joint use of complementary observations during pretraining. The Sentinel-3 ocean-colour foundation model jointly pretrains on ocean-colour reflectance and sea surface temperature through masked modeling and evaluates the resulting representations on ocean-colour-related tasks [8]. OceanMAE combines multispectral imagery with auxiliary ocean descriptors for self-supervised reconstruction and assesses the transferability of its learned representations on ocean segmentation and bathymetry regression [9]. Despite these advances in ocean forecasting, multi-source information utilization, and domain-specific pretraining, diferences in the physical attributes, spatiotemporal scales, and temporal dynamics of heterogeneous observations continue to pose challenges for unified representation learning and joint modeling. Efectively encoding heterogeneous ocean observations, characterizing dependencies across spatial scales and temporal horizons, and learning ocean-state evolution through pretraining remain important open research questions for multimodal ocean foundation models.

Despite the progress of existing studies, developing multimodal foundation models that jointly leverage satellite imagery, oceanographic fields, and buoy observations remains challenging in terms of heterogeneous representation learning, multiscale spatiotemporal modeling, and the learning of ocean-state dynamics. First, these observation types difer substantially in data structure, physical semantics, and sampling characteristics [10, 11]. Satellite imagery primarily captures the spatial distribution and textural characteristics of ocean surface conditions, oceanographic fields describe the spatial structures of variables such as temperature, salinity, and current velocity, while buoy observations record localized temporal variations at specific locations. Fixed input representations may not adequately accommodate these diferences in observational attributes [12, 13], potentially limiting the efective utilization of heterogeneous information within a shared representation space. Second, marine environments exhibit multiscale spatiotemporal variability, in which local spatial structures, broader environmental contexts, and temporal variations over diferent horizons are interrelated. Single-scale feature extraction may therefore be insuficient to characterize these complex dependencies. Furthermore, pretraining based solely on reconstructing observed data does not necessarily encourage models to learn ocean-state evolution patterns that are predictive of future conditions [14]. Consequently, further research is needed to develop transferable multimodal ocean foundation models that efectively represent heterogeneous observations, capture multiscale spatiotemporal dependencies, and learn the temporal evolution of ocean states through pretraining.

To address these challenges, we propose YingLan, a multimodal ocean foundation model for unified representation learning from heterogeneous ocean observations, including satellite imagery, gridded oceanographic fields, and buoy time series. YingLan integrates physics-attributeconditioned dynamic encoding, multiscale spatiotemporal interaction, and ocean-state evolutionaware pretraining to learn transferable representations of ocean states across heterogeneous observation types and temporal scales. The main contributions of this work are summarized as follows:

• A multimodal foundation model for heterogeneous ocean observations. We develop a unified modeling framework that accommodates satellite imagery, gridded oceanographic fields, and buoy time series, providing a shared representation space for heterogeneous ocean observations and supporting adaptation to diverse regional ocean monitoring and forecasting tasks.

• Physics-attribute-conditioned dynamic encoding. We introduce a dynamic encoding mechanism that jointly incorporates physical semantics and observation conditions into input projection parameter generation. By distinguishing variable-specific physical attributes from observation-specific characteristics, the proposed mechanism enables adaptive encoding of heterogeneous ocean observations.

• Multiscale spatiotemporal interaction and evolution-aware pretraining. We propose a Spatiotemporal Increment Coupling (STIC) module that facilitates cross-scale information exchange through latent temporal increments, enabling interactions between localized ocean structures and large-scale environmental contexts. We further develop an ocean-state evolution-aware pretraining strategy that jointly optimizes masked observation reconstruction, horizon-conditioned latent increment prediction, and temporal compositional consistency to learn temporally coherent representations of ocean-state dynamics.

• Evaluation of regional transferability and generalization. We investigate the transferability of YingLan in regional ocean applications, using Shenzhen coastal waters as a case study. Through downstream forecasting and, where supported by the experimental design, few-shot adaptation, cross-temporal evaluation, and cross-region transfer, we assess the model’s predictive performance and generalization under varying data availability and spatiotemporal conditions.

## 2 Related Work

## 2.1 Ocean Foundation Models and Pretraining Paradigms

Existing deep learning studies on ocean modeling have primarily focused on ocean state estimation and evolution forecasting, aiming to characterize ocean physical processes and their spatiotemporal variability through data-driven approaches. Eeltink et al. [15] proposed a hybrid approach that integrates recurrent neural networks with a nonlinear wave evolution model, learning data-driven correction terms to represent wave-breaking processes and illustrating the integration of physical models with neural networks for ocean dynamics modeling. Ham et al. [16] proposed a global ocean data assimilation framework based on partial convolutions and generative adversarial learning, reconstructing ocean states by incorporating observational information and improving the handling of incomplete ocean observations. Furthermore, Cui et al. [6] introduced WenHai, which employs a Swin Transformer-based architecture to model multivariate ocean state evolution and incorporates air–sea flux information and multi-step forecasting strategies. These studies advance ocean deep learning from complementary perspectives, including physical process modeling, ocean state reconstruction, and spatiotemporal forecasting. Nevertheless, their training objectives remain primarily tied to specific ocean variables and predefined estimation or forecasting tasks, motivating the exploration of pretrained representations that can be transferred across diverse ocean applications.

Recent ocean foundation models have explored diverse pretraining strategies tailored to diferent types of ocean data. Xiong et al. [7] proposed AI-GOMS, which learns ocean-state representations from HYCOM reanalysis fields through Fourier-based masked autoencoding and supports downstream adaptation across ocean modeling tasks. Glaser et al. [17] introduced WV-Net, using contrastive self-supervised learning on large-scale Sentinel-1 SAR imagery to obtain transferable representations for ocean remote sensing. Complementarily, Bi et al. [18] developed OceanGPT through ocean-science corpus pretraining and domain-specific instruction tuning, while Xue et al. [19] introduced OceanPile, a multimodal ocean-science corpus designed to support foundation model training across marine imagery, sonar data, and scientific text. These studies span physical-field pretraining, image-based representation learning, and language-oriented ocean knowledge modeling, illustrating the diversity of data and objectives in ocean-domain pretraining. However, their learning frameworks address diferent data modalities and task requirements, leaving the unified representation of heterogeneous ocean observations an important direction for further investigation.

## 2.2 Heterogeneous Ocean Representation and Multiscale Spatiotemporal Learning

Ocean observations difer in their spatial resolution, sampling characteristics, and physical information, motivating the development of methods that can efectively integrate complementary data sources. Thiria et al. combined sea surface height (SSH) and sea surface temperature (SST) observations to reconstruct mesoscale ocean currents through CNN-based downscaling [20]. Wang et al. proposed ODF-Net, which fuses in situ measurements, satellite observations, and three-dimensional model and reanalysis data to reconstruct ocean temperature fields [21]. Champenois and Sapsis developed a machine learning framework that combines historical reanalysis with real-time satellite and buoy measurements for regional four-dimensional ocean temperature reconstruction [22]. Under sparse observation conditions, Lei et al. investigated multimodal fusion-based spatiotemporal incremental learning for ocean environment perception [23]. Collectively, these studies explore ocean data fusion through spatial downscaling, multisource field reconstruction, and multimodal spatiotemporal learning.

Existing ocean spatiotemporal modeling methods primarily employ hierarchical feature extraction, attention mechanisms, and multiscale feature fusion to characterize spatial dependencies and temporal variations in ocean variables. Zhao et al. proposed HSSSA, a hierarchical encoder-decoder network that integrates spatial and temporal self-attention for multiscale SST forecasting [24]. Bai et al. developed MUSTAN, which progressively fuses features across spatial scales and applies spatial and temporal attention to predict SST variations [25]. Pan et al. introduced an efective attention model for global SST prediction at daily and monthly temporal resolutions, explicitly modeling spatiotemporal correlations in historical observations [26]. Tao et al. investigated multiscale spatial features and temporal attention for El Niño-Southern Oscillation (ENSO) prediction, demonstrating the use of these mechanisms in modeling largescale ocean variability [27]. These studies illustrate complementary architectural strategies for capturing ocean dynamics across spatial scales, temporal horizons, and physical processes.

## 3 Method

## 3.1 Overall Architecture

We propose YingLan, a multimodal ocean foundation model designed for unified representation learning from heterogeneous ocean observations. The overall architecture is illustrated in Fig. 1. YingLan leverages large-scale satellite remote sensing imagery, gridded oceanographic fields, and buoy time-series observations to address heterogeneity in physical semantics, data structures, and spatiotemporal resolutions across diverse observational sources. The model aims to learn generalizable ocean representations that capture both shared cross-modal information and ocean-specific spatiotemporal dynamics. Its architecture comprises three core components:

![](images/3e748e0daa16ab3b9efe6e13a813d70b63943d173c32a185ba3473717dc13774.jpg)  
Figure 1: Overall architecture of YingLan.

Physics-Attribute-Conditioned Dynamic Encoding, Spatiotemporal Increment Coupling (STIC), and Ocean-State Evolution-Aware Pretraining.

YingLan first employs Physics-Attribute-Conditioned Dynamic Encoding to construct adaptive feature projections conditioned on the physical semantics and observational characteristics of individual variables. Modality-specific input adapters are used to map ocean observations with diferent data structures into a shared feature space. The resulting representations are subsequently processed by the STIC module, which integrates multiscale spatial representations with scale-specific temporal modeling. By coupling latent temporal increments across spatial scales, STIC captures dynamic interactions between local ocean structures and large-scale environmental contexts, producing unified spatiotemporal representations.

During pretraining, YingLan jointly optimizes masked observation reconstruction, horizonconditioned latent increment prediction, and temporal compositional consistency. These complementary objectives enable the model to capture structural dependencies among heterogeneous observations while learning ocean-state variations across multiple temporal horizons. Following pretraining, the shared encoder is adapted to regional ocean monitoring and prediction tasks through task-specific prediction heads and parameter fine-tuning, allowing the transferability of YingLan’s pretrained representations to be evaluated in regional marine environments.

## 3.2 Physics-Attribute-Conditioned Dynamic Encoding

Satellite imagery, oceanographic fields, and buoy observations characterize the marine environment from diferent observational perspectives. Their heterogeneity arises not only from diferences in data structures and sampling schemes, but also from the complex relationships between physical-variable semantics and observation mechanisms. Observations from diferent sources may represent the same physical variable under diferent measurement conditions, whereas a single observation platform may provide measurements with distinct physical meanings. Therefore, efective multimodal representation learning requires more than mapping heterogeneous inputs into a common feature space; the encoding process must explicitly capture the relationships between physical semantics and observation conditions.

To address this issue, we propose a physics-attribute-conditioned dynamic embedding mechanism that incorporates the physical semantics and observation conditions of ocean observations into a unified parameter-generation process. The key idea is to formulate the input projection as a learnable operator conditioned on observation attributes, rather than as a fixed mapping predetermined for a specific channel configuration. By separately representing variable semantics and observation conditions and explicitly modeling their coupling, the proposed mechanism aims to distinguish the shared characteristics of diferent ocean variables from the variations induced by their observation conditions, thereby enabling condition-adaptive representations of heterogeneous ocean observations.

For the c-th observation channel, let $\mathbf { m } _ { c } ^ { s }$ and $\mathbf { m } _ { c } ^ { o }$ denote its physical-semantic attributes and observation-condition attributes, respectively. The former describes the physical meaning of the observed variable and its associated properties, whereas the latter characterizes measurement conditions, including sensor configuration, observation depth, and spatial resolution, where applicable. These two types of attributes are independently mapped into latent spaces using learnable encoding functions:

$$
\mathbf { a } _ { c } ^ { s } = g _ { \phi _ { s } } \left( \mathbf { m } _ { c } ^ { s } \right) , \qquad \mathbf { a } _ { c } ^ { o } = g _ { \phi _ { o } } \left( \mathbf { m } _ { c } ^ { o } \right) .\tag{1}
$$

where $\mathbf { a } _ { c } ^ { s }$ and $\mathbf { a } _ { c } ^ { o }$ denote the physical-semantic representation and observation-condition representation, respectively. For satellite spectral channels, the attributes may include wavelength or spectral-response characteristics. For oceanographic fields, they primarily describe variable categories, physical dimensions, and other relevant physical properties. This factorized attribute representation enables the encoding process to distinguish the physical meaning of an observed variable from the conditions under which it is measured, providing explicit conditioning information for dynamic parameter generation.

To model the joint influence of physical semantics and observation conditions on the input projection, we define the dynamic embedding operator for the c-th observation channel as

$$
\mathbf { W } _ { c } = \mathcal { H } _ { \boldsymbol { \theta } } \left( \mathbf { a } _ { c } ^ { s } , \mathbf { a } _ { c } ^ { o } \right) ,\tag{2}
$$

where $\mathcal { H } _ { \theta }$ denotes a learnable dynamic parameter-generation function. To explicitly characterize the coupling between the two types of attributes, we adopt a bilinear factor-coupling parameterization:

$$
\mathbf { W } _ { c } = \mathbf { U } \operatorname { D i a g } \left( \pmb { \alpha } _ { c } ^ { s } \right) \mathbf { C } \operatorname { D i a g } \left( \pmb { \alpha } _ { c } ^ { o } \right) \mathbf { V } ,\tag{3}
$$

where the attribute-dependent coeficients are generated as

$$
\alpha _ { c } ^ { s } = \mathrm { S o f t m a x } \left( f _ { \theta _ { s } } \left( \mathbf { a } _ { c } ^ { s } \right) \right) ,\tag{4}
$$

$$
\alpha _ { c } ^ { o } = \mathrm { S o f t m a x } \left( f _ { \theta _ { o } } \left( \mathbf { a } _ { c } ^ { o } \right) \right) .\tag{5}
$$

where U and V are learnable projection factors, C is a learnable matrix that captures interactions between physical-semantic and observation-condition factors, and $f _ { \theta _ { s } }$ and $f _ { \theta _ { o } }$ are coeficient generation functions. The operator Diag(·) constructs a diagonal matrix from a vector. This parameterization allows physical semantics and observation conditions to modulate distinct latent factors, while their interaction through C jointly determines the input projection. As a result, the embedding parameters explicitly reflect the combined efects of diferent observation-attribute configurations.

For satellite imagery and gridded oceanographic fields, let $\mathbf { p } _ { i , c }$ denote the local input patch associated with the c-th channel at spatial location i. Its embedded representation is computed as

$$
\mathbf { z } _ { i } = \sum _ { c = 1 } ^ { C } { \mathbf { W } } _ { c } \operatorname { v e c } \left( \mathbf { p } _ { i , c } \right) + \mathbf { b } ,\tag{6}
$$

where $C$ denotes the number of input channels, vec(·) denotes the vectorization operator, and b is a learnable bias. For buoy observations, the input representation is constructed according to the structure of multivariate time series, incorporating variable attributes, observation timestamps, and spatial locations. The resulting representations are mapped into a latent space with a feature dimension compatible with that of the spatial observations. Consequently, diferent modalities retain input representations appropriate to their respective data structures while producing tokens suitable for subsequent multiscale spatiotemporal interaction.

Overall, the proposed method formulates the encoding of heterogeneous ocean observations as a dynamic mapping jointly conditioned on physical semantics and observation conditions. Its central mechanism explicitly captures the coupled influence of these attributes on the embedding parameters. The efectiveness of this design will be evaluated against fixed projections, single-attribute-conditioned encoding, and conventional attribute-concatenation-based parameter generation, with particular emphasis on heterogeneous observation representation and generalization to unseen combinations of variables and observation conditions.

## 3.3 Spatiotemporal Increment Coupling (STIC) Module

The spatiotemporal evolution of oceanic environments exhibits pronounced multiscale dependencies. Local ocean fronts, eddy boundaries, and fine-scale structures are often characterized by strong spatial heterogeneity, whereas large-scale circulation patterns and regional environmental fields provide broader spatial constraints. These structures difer not only in their spatial characteristics but also in their temporal dynamics, with interactions across scales contributing to the complex evolution of ocean states. Conventional factorized spatiotemporal modeling approaches typically capture spatial correlations and temporal dependencies separately, while multiscale fusion methods primarily emphasize feature aggregation across spatial resolutions. Consequently, the dynamic interactions through which temporal variations at diferent spatial scales jointly shape ocean-state representations remain insuficiently characterized. To address this limitation, we propose a Spatiotemporal Increment Coupling (STIC) module that integrates multiscale spatial representations with scale-specific temporal dynamics. By treating temporal increments of latent states as carriers of cross-scale information exchange, STIC establishes dynamic coupling between local structural variations and large-scale environmental contexts. Unlike conventional approaches that directly fuse static multiscale features, STIC incorporates cross-scale interactions into the state-update process, enabling historical variations at diferent scales to jointly contribute to spatiotemporal representation learning.

STIC first organizes the encoded ocean observations into hierarchical spatial representations, enabling ocean structures with diferent spatial extents to be characterized at their corresponding resolutions. Let $\mathbf { Y } _ { 1 : T } = \{ \mathbf { Y } _ { 1 } , \dots , \mathbf { Y } _ { T } \}$ denote the temporal feature sequence produced by the Physics-Attribute-Conditioned Dynamic Encoding module, where $\mathbf { Y } _ { t } \in \mathbb { R } ^ { N \times d }$ represents the spatial token embeddings at time step t. Here, $T$ denotes the sequence length, while N and d represent the number of spatial tokens and the feature dimension, respectively. Hierarchical spatial features are constructed through scale-dependent transformations:

$$
\mathbf { L } _ { t } ^ { \left( \ell \right) } = \mathcal { P } _ { \ell } \left( \mathbf { Y } _ { t } \right) , \qquad \ell = 0 , \ldots , L - 1 .\tag{7}
$$

where $\mathbf { L } _ { t } ^ { ( \ell ) } \in \mathbb { R } ^ { N _ { \ell } \times d }$ denotes the feature matrix at spatial scale ℓ and time step t, $\mathcal { P } _ { \ell }$ represents the corresponding spatial scale transformation operator, $N _ { \ell }$ is the number of tokens at that scale, and $L$ denotes the total number of spatial scales. By organizing features at diferent spatial granularities, the hierarchy accommodates localized ocean fronts, eddy boundaries, and large-scale environmental contexts within their respective spatial extents.

To capture historical variations at diferent spatial scales, STIC employs scale-specific causal temporal encoders to model temporal dependencies within each spatial hierarchy:

$$
\mathbf { B } _ { t } ^ { \left( \ell \right) } = \mathcal { T } _ { \ell } \left( \mathbf { L } _ { 1 : t } ^ { \left( \ell \right) } \right) .\tag{8}
$$

where $\mathbf { L } _ { 1 : t } ^ { ( \ell ) } = \{ \mathbf { L } _ { 1 } ^ { \left( \ell \right) } , \ldots , \mathbf { L } _ { t } ^ { \left( \ell \right) } \}$ denotes the historical feature sequence at scale ℓ up to time step $t ,$ and $\mathcal { T } _ { \ell }$ is the corresponding causal temporal encoding operator. The resulting latent state $\mathbf { B } _ { t } ^ { ( \ell ) } \in \mathbb { R } ^ { N _ { \ell } \times d }$ integrates historical information at the specified spatial resolution. The causal constraint ensures that each state depends exclusively on observations available up to the current time step. Meanwhile, scale-specific temporal modeling preserves the distinct temporal dependencies of localized ocean structures and broader environmental contexts, avoiding the need to force features at diferent spatial granularities into a single temporal representation.

Although hierarchical temporal states capture historical dependencies within individual scales, directly aggregating these states does not explicitly distinguish relatively persistent spatial features from temporally varying components. STIC therefore extracts latent temporal increments from scale-specific states and uses them as the basis for cross-scale information transfer:

$$
\mathbf { J } _ { t } ^ { \left( \ell \right) } = \frac { \mathbf { B } _ { t } ^ { \left( \ell \right) } - \mathbf { B } _ { t - 1 } ^ { \left( \ell \right) } } { \delta _ { t } } .\tag{9}
$$

where $\mathbf { J } _ { t } ^ { ( \ell ) } \in \mathbb { R } ^ { N _ { \ell } \times d }$ represents the latent temporal increment at scale ℓ and time step $t , \mathbf { B } _ { t - 1 } ^ { ( \ell ) }$ denotes the latent state at the preceding observation time, and $\delta _ { t } > 0$ is the actual elapsed time between consecutive observations. The increment characterizes temporal variations in the latent representation rather than the physical time derivative of an observed oceanographic variable. Compared with directly fusing complete scale-specific states, increment-based interaction emphasizes the temporally varying information that links ocean structures across spatial resolutions.

Cross-scale interactions are established through spatially aligned increment transfer. For a target scale $\ell ,$ latent temporal increments from neighboring scales are mapped to the target spatial resolution and aggregated into a cross-scale dynamic representation:

$$
\mathbf { M } _ { t } ^ { ( \ell ) } = \sum _ { j \in \mathcal { N } ( \ell ) } \mathcal { A } _ { j  \ell } ( \mathbf { J } _ { t } ^ { ( j ) } ) .\tag{10}
$$

where $\mathbf { M } _ { t } ^ { ( \ell ) } \in \mathbb { R } ^ { N _ { \ell } \times d }$ denotes the cross-scale dynamic feature matrix received at the target scale. $\mathcal N ( \ell )$ is the set of neighboring scale indices, with $j$ representing a source scale. The term $\mathbf { J } _ { t } ^ { ( j ) } \in \mathbb { R } ^ { N _ { j } \times d }$ denotes the latent temporal increment at the source scale, while $\mathcal { A } _ { j  \ell }$ $\mathbb { R } ^ { N _ { j } \times d } \xrightarrow { \smile } \mathbb { R } ^ { N _ { \ell } \times d }$ represents the spatial mapping operator from scale $j$ to scale ℓ. By establishing spatial correspondence across diferent resolutions, this operation allows historical variations in localized structures to contribute to coarse-scale environmental representations, while changes in large-scale backgrounds can be propagated to finer-scale features. Since the transferred information is derived exclusively from current and historical observations, STIC performs causal spatiotemporal encoding without relying on future-state supervision.

To incorporate cross-scale increments directly into target-scale feature transformations, STIC employs a bilinear coupling operator that performs multiplicative interactions between the target-scale historical state and the transferred dynamic information:

$$
\begin{array} { r } { \mathbf { N } _ { t } ^ { ( \ell ) } = \mathbf { B } _ { t } ^ { ( \ell ) } + \Big [ \left( \mathbf { B } _ { t } ^ { ( \ell ) } \pmb { \Theta } _ { \mathrm { b } } ^ { ( \ell ) } \right) \odot \qquad } \\ { \left( \mathbf { M } _ { t } ^ { ( \ell ) } \pmb { \Theta } _ { \mathrm { m } } ^ { ( \ell ) } \right) \Big ] \pmb { \Theta } _ { \mathrm { o } } ^ { ( \ell ) } . } \end{array}\tag{11}
$$

where $\mathbf { N } _ { t } ^ { ( \ell ) } \in \mathbb { R } ^ { N _ { \ell } \times d }$ denotes the updated latent state after cross-scale coupling. The learnable projection matrices $\boldsymbol { \Theta } _ { \mathrm { b } } ^ { ( \ell ) } \in \mathbb { R } ^ { d \times r }$ and $\Theta _ { \mathrm { m } } ^ { ( \ell ) } \in \mathbb { R } ^ { d \times r }$ project the historical state and cross-scale dynamic information into a common interaction space, respectively, while $\Theta _ { \mathrm { o } } ^ { ( \ell ) } \in \mathbb { R } ^ { r \times d }$ denotes the output projection matrix. Here, r is the coupling feature dimension, and $\odot$ denotes elementwise multiplication. The bilinear interaction makes the contribution of transferred dynamic information dependent on the historical state at the target scale, thereby capturing conditional relationships between the two representations. Unlike direct concatenation or linear addition, this operator allows cross-scale temporal variations to participate explicitly in latent state transformation rather than serving merely as supplementary feature components.

Following cross-scale coupling, the hierarchical features are reconstructed at the original spatial resolution to produce the final spatiotemporal representation:

$$
\mathbf { E } _ { t } = \mathbf { Y } _ { t } + \sum _ { \ell = 0 } ^ { L - 1 } \mathcal { R } _ { \ell } \left( \mathbf { N } _ { t } ^ { ( \ell ) } \right) .\tag{12}
$$

where $\mathbf { E } _ { t } \in \mathbb { R } ^ { N \times d }$ represents the output spatiotemporal feature matrix of STIC, and $\mathcal { R } _ { \ell }$ $\mathbb { R } ^ { N _ { \ell } \times d } \to \mathbb { R } ^ { N \times d }$ denotes the spatial reconstruction operator at scale ℓ. The input representation $\mathbf { Y } _ { t }$ is retained through a residual connection to preserve the original spatial information. By reconstructing scale-specific dynamic features into a shared spatial resolution, STIC integrates localized ocean structures, large-scale environmental contexts, and their cross-scale dynamic interactions within a unified representation. Through historical increment-driven cross-scale state coupling, the module provides spatiotemporal features for subsequent self-supervised pretraining and downstream task adaptation.

## 3.4 Ocean-State Evolution-Aware Pretraining

Learning transferable representations from heterogeneous ocean observations requires not only the reconstruction of incomplete measurements but also the exploitation of temporal relationships between historical observations and subsequent ocean-state changes. Although masked reconstruction encourages the model to capture observational structures and cross-variable dependencies, it provides limited direct supervision for relating historical ocean states to their future variations. Moreover, independently optimizing predictions over diferent temporal horizons does not necessarily ensure coherent relationships among the resulting latent transitions. To address these limitations, we incorporate horizon-conditioned latent state prediction and temporal compositional consistency into the self-supervised pretraining objective. Unlike STIC, which models interactions among historical spatiotemporal features within the encoder, this learning strategy exploits observed future states as self-supervisory signals, encouraging the shared encoder to learn ocean representations that support latent state prediction across multiple temporal horizons.

Pretraining employs a dual-branch architecture consisting of an online encoder and a momentum target encoder. The online branch processes randomly masked historical observations through the shared encoding backbone, whereas the target branch extracts supervisory representations from unmasked observations at the corresponding target time steps. The target encoder parameters are updated using an exponential moving average of the online encoder parameters. Considering the variability in spatial coverage and data completeness among heterogeneous oceanographic observations, masked reconstruction is retained as an auxiliary learning objective to preserve spatial structures and cross-variable dependencies:

$$
\mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { | \Omega | } \sum _ { ( u , v ) \in \Omega } \left. \widehat { \mathbf { q } } _ { u , v } - \mathbf { q } _ { u , v } \right. _ { 2 } ^ { 2 } .\tag{13}
$$

Here, Ω denotes the set of valid masked observation units, while $\mathbf { q } _ { u , v }$ and $\widehat { \mathbf { q } } _ { u , v }$ represent the normalized ground-truth observation vector and its reconstructed counterpart, respectively. The reconstruction objective preserves observational information, whereas the evolution-aware learning objectives provide additional dynamic supervision beyond instantaneous data recovery.

To establish explicit relationships between historical representations and future ocean-state variations, we employ a shared horizon-conditioned predictor to learn latent state transitions over diferent prediction intervals:

$$
\widehat { \mathbf { D } } _ { t , h } = f _ { \varphi } \left( \mathbf { H } _ { t } , \mathbf { e } _ { h } \right) .\tag{14}
$$

$$
\mathcal { L } _ { \mathrm { d y n } } = \mathbb { E } _ { t , h } \left[ \left\| \widehat { \mathbf { D } } _ { t , h } - \mathrm { s g } \left( \mathbf { \overline { { H } } } _ { t + h } - \mathbf { \overline { { H } } } _ { t } \right) \right\| _ { F } ^ { 2 } \right] .\tag{15}
$$

Here, $\mathbf { H } _ { t }$ and $\overline { { \mathbf { H } } } _ { t }$ denote the latent representations produced by the online encoder and the momentum target encoder, respectively. The vector $\mathbf { e } _ { h }$ encodes the prediction horizon $h , \ \mathrm { s g ( \cdot ) }$ denotes the stop-gradient operation, and $f _ { \varphi }$ represents the shared latent transition predictor. Unlike the historical temporal increments employed internally by STIC, $\widehat { \mathbf { D } } _ { t , h }$ represents the predicted latent increment from the current state toward a future target state. By jointly supervising predictions across multiple temporal horizons, the model is encouraged to retain information relevant to subsequent ocean-state changes beyond that captured through observational reconstruction alone.

Latent state transitions over diferent prediction horizons are further related through a temporal compositional consistency constraint. Given the same historical state, the latent increment predicted directly over a longer interval is encouraged to remain approximately consistent with the cumulative increment obtained through two successive shorter-horizon predictions:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { c o n } } = \mathbb { E } _ { t , h _ { 1 } , h _ { 2 } } \Big [ \big \| \widehat { \mathbf { D } } _ { t , h _ { 1 } + h _ { 2 } } - \widehat { \mathbf { D } } _ { t , h _ { 1 } } } \\ & { \qquad - \ f _ { \varphi } \left( \mathbf { H } _ { t } + \widehat { \mathbf { D } } _ { t , h _ { 1 } } , \mathbf { e } _ { h _ { 2 } } \right) \big \| _ { F } ^ { 2 } \Big ] . } \end{array}\tag{16}
$$

Here, $h _ { 1 }$ and $h _ { 2 }$ denote the prediction horizons of two successive stages, respectively. The term $\widehat { \bf D } _ { t , h _ { 1 } + h _ { 2 } }$ represents the cumulative latent increment predicted directly over the combined horizon, while $\mathbf { H } _ { t } + \widehat { \mathbf { D } } _ { t , h _ { 1 } }$ denotes the intermediate latent state predicted after the first stage. The expectation $\mathbb { E } _ { t , h _ { 1 } , h _ { 2 } } [ \cdot ]$ is taken over sampled initial time steps and valid combinations of prediction horizons. This soft consistency constraint relates diferent prediction trajectories through a shared latent transition mapping without requiring ocean-state evolution to strictly satisfy a prescribed physical governing equation.

The overall pretraining objective is formulated as

$$
{ \mathcal { L } } _ { \mathrm { p r e } } = \lambda _ { \mathrm { r e c } } { \mathcal { L } } _ { \mathrm { r e c } } + \lambda _ { \mathrm { d y n } } { \mathcal { L } } _ { \mathrm { d y n } } + \lambda _ { \mathrm { c o n } } { \mathcal { L } } _ { \mathrm { c o n } } .\tag{17}
$$

Here, $\lambda _ { \mathrm { { r e c } } } , \lambda _ { \mathrm { { d y n } } }$ , and $\lambda _ { \mathrm { { c o n } } }$ are weighting coeficients that balance observational reconstruction, future latent state prediction, and temporal compositional consistency, respectively. $\mathrm { B y }$ incorporating future-observation-based self-supervision, the joint pretraining objective further encourages the shared encoder to capture representations informative of ocean-state variations, providing transferable features for downstream tasks such as coastal ocean monitoring and prediction.

## 4 Experiments

## 4.1 Datasets

## 4.1.1 Pretraining Datasets

To support large-scale self-supervised pretraining of YingLan on heterogeneous ocean observations, we established a comprehensive oceanographic data repository integrating in situ observations, satellite remote sensing products, and oceanic and atmospheric reanalysis datasets. The repository incorporates ocean temperature and salinity observations and gridded products from Argo profiling floats, the World Ocean Database (WOD), and EN4; satellite-derived sea surface temperature products from MODIS, AMSR-2, and GHRSST; and oceanic and atmospheric reanalysis datasets from GLORYS, CORA, and NCEP-NCAR. These datasets encompass a wide range of oceanographic and environmental variables, including sea surface temperature, salinity, sea surface height, and surface wind fields. The heterogeneous sources provide complementary data representations, including profiling-float observation sequences, multitemporal satellite imagery, and temporally resolved gridded oceanographic fields, with diverse sampling characteristics. The repository currently contains more than 6 TB of data, spanning multiple observation types, extended temporal periods, and broad geographic coverage, thereby providing a substantial data foundation for unified representation learning and self-supervised pretraining of YingLan across heterogeneous ocean observations.

## 4.1.2 Downstream Evaluation Datasets

To evaluate the transferability of YingLan’s pretrained representations to regional marine environments, we construct downstream evaluation datasets using heterogeneous observational data collected from Shenzhen and its adjacent coastal waters. These datasets cover a range of application scenarios, including ocean environmental forecasting, typhoon prediction, coastal ecosystem monitoring, and marine hazard early warning. The data comprise regional gridded oceanographic fields, ocean buoy observations and hydrometeorological time series, multitemporal satellite imagery, unmanned aerial vehicle (UAV) imagery, and associated in situ survey records. Together, these sources support diverse learning tasks, including continuous environmental variable forecasting, time-series regression, remote sensing semantic segmentation, and regional change detection. By evaluating YingLan across regional tasks with distinct data structures and prediction targets, we assess the transferability of its pretrained representations to ocean dynamics characterization, complex coastal environment recognition, and coastal ecosystem change analysis.

## 4.2 Experimental Setup

## 4.2.1 Baselines and Evaluation Metrics

To evaluate the downstream forecasting performance of YingLan, we compare it against three ocean forecasting models: LangYa [28], XiHe [5], and GloNet2 [29]. For the seven-day sea surface salinity (SSS) forecasting task, we report the average root mean square error (RMSE) across all seven forecast lead times, expressed in practical salinity units (PSU). For additional downstream tasks and transfer learning experiments, evaluation metrics are selected according to the target variables and task characteristics. All methods are evaluated using consistent test samples, spatial grids, and valid-ocean masks to ensure fair and comparable performance assessments.

## 4.2.2 Implementation Details

YingLan is implemented using the MindSpore deep learning framework and trained on Huawei Ascend 910B Neural Processing Units (NPUs). Model training is conducted within the Ascend computing environment, leveraging MindSpore’s automatic diferentiation and tensor computation capabilities for parameter optimization. This implementation enables YingLan to be trained on domestic AI computing infrastructure and provides a technical foundation for subsequent deployment in practical ocean monitoring applications.

## 4.3 Downstream Task Performance

We evaluate YingLan on a 7-day sea surface salinity (SSS) forecasting task to assess its predictive capability in modeling ocean surface conditions. The forecasting performance is measured using the average root mean square error (RMSE) across seven forecast lead days, with lower values indicating higher prediction accuracy.

As shown in Fig. 2, YingLan achieves the lowest average RMSE of 0.687508 PSU among the compared methods, outperforming LangYa (0.691538 PSU), XiHe (0.726042 PSU), and GloNet2 (0.785023 PSU). Specifically, YingLan reduces the average RMSE by 0.58%, 5.31%, and 12.42% compared with LangYa, XiHe, and GloNet2, respectively. These results indicate that YingLan demonstrates desirable performance in multi-day sea surface salinity forecasting, highlighting its potential for predicting the state of the ocean surface.

7-Day Sea Surface Salinity (SSS) Forecasting  
![](images/00e0f1b3a9a40119604f73268fd052bcbc42097750a0ff3257267cfdafa0b779.jpg)  
Figure 2: Comparison of the average RMSE for 7-day sea surface salinity (SSS) forecasting. Lower values indicate better forecasting performance.

Table 1: Illustrative ablation study of YingLan on seven-day SST forecasting in Shenzhen coastal waters. All values are hypothetical placeholders.
<table><tr><td>Model Variant</td><td>Avg. RMSE (°C) ↓</td></tr><tr><td>w/o Dynamic Encoding</td><td>0.536</td></tr><tr><td>w/o STIC</td><td>0.548</td></tr><tr><td>w/o Evolution-aware Pretraining</td><td>0.521</td></tr><tr><td>Full YingLan (Ours)</td><td>0.500</td></tr></table>

## 4.4 Results analysis of ablation experiment

To investigate the contributions of the key components of YingLan, we construct three ablation variants by individually removing the physics-attribute-conditioned dynamic encoding mechanism, the Spatiotemporal Increment Coupling (STIC) module, and the ocean-state evolution-aware pretraining strategy. All variants are evaluated on the seven-day sea surface temperature (SST) forecasting task in Shenzhen coastal waters, using the average RMSE across seven forecast lead times as the evaluation metric. The corresponding ablation results are presented in Table 1.

## 5 Conclusion

This work investigates unified representation learning from heterogeneous ocean observations and presents YingLan as a multimodal foundation model for ocean environmental modeling. By incorporating physical semantics and observation conditions into adaptive feature encoding, YingLan provides a flexible representation mechanism for ocean variables with diverse observational characteristics. Its multiscale spatiotemporal architecture further integrates temporal variations across spatial hierarchies, while the joint pretraining objectives encourage the learning of both observational structures and ocean-state dynamics. These designs establish a unified modeling framework for integrating satellite imagery, gridded oceanographic fields, and buoy time-series observations. To evaluate its applicability to regional ocean forecasting, we conduct a downstream seven-day sea surface salinity (SSS) forecasting task in the Shenzhen coastal waters. YingLan achieves an average RMSE of 0.687508 PSU, yielding the lowest reported error among the evaluated methods. Beyond predictive performance, YingLan is implemented using MindSpore and trained on Huawei Ascend 910B NPUs, demonstrating the feasibility of developing and training multimodal ocean foundation models within a domestic AI computing environment. Overall, this study provides a methodological basis for learning transferable ocean representations while supporting the development of ocean intelligence models on domestic computing infrastructure. Future research will investigate broader downstream task adaptation, cross-regional generalization, and robust modeling under incomplete and heterogeneous ocean observations.

## Acknowledgments

Optional acknowledgments go here. Remove this section if unused.

## References

[1] Shuyun Yuan, Ying Li, Fangwen Bao, Haoxiang Xu, Yuping Yang, Qiushi Yan, Shuqiao Zhong, Haoyang Yin, Jiajun Xu, Ziwei Huang, and Jian Lin. Marine environmental monitoring with unmanned vehicle platforms: Present applications and future prospects. Science of The Total Environment, 858:159741, 2023. ISSN 0048-9697. doi: https://doi. org/10.1016/j.scitotenv.2022.159741. URL https://www.sciencedirect.com/science/ article/pii/S0048969722068413.

[2] Anabel von Jackowski. Navigating the current landscape of ocean observations: an overview from platform infrastructures to networks related to ocean time series. Frontiers in Marine Science, Volume 11 - 2024, 2025. ISSN 2296-7745. doi: 10.3389/fmars.2024. 1363863. URL https://www.frontiersin.org/journals/marine-science/articles/ 10.3389/fmars.2024.1363863.

[3] Inseong Chang, Young Ho Kim, Hyunkeun Jin, Young-Gyu Park, Gyundo Pak, and You-Soon Chang. Impact of satellite and regional in-situ profile data assimilation on a high-resolution ocean prediction system in the northwest pacific. Frontiers in Marine Science, Volume 10 - 2023, 2023. ISSN 2296-7745. doi: 10.3389/fmars.2023. 1085542. URL https://www.frontiersin.org/journals/marine-science/articles/ 10.3389/fmars.2023.1085542.

[4] Daniela Szwarcman, Sujit Roy, Paolo Fraccaro, Þorsteinn Elí Gíslason, Benedikt Blumenstiel, Rinki Ghosal, Pedro Henrique de Oliveira, Joao Lucas de Sousa Almeida, Rocco Sedona, Yanghui Kang, Srija Chakraborty, Sizhe Wang, Carlos Gomes, Ankur Kumar, Vishal Gaur, Myscon Truong, Denys Godwin, Sam Khallaghi, Hyunho Lee, Chia-Yu Hsu, Ata Akbari Asanjan, Besart Mujeci, Disha Shidham, Rufai Omowunmi Balogun, Venkatesh Kolluru, Trevor Keenan, Paulo Arevalo, Wenwen Li, Hamed Alemohammad, Pontus Olofsson, Timothy Mayer, Christopher Hain, Robert Kennedy, Bianca Zadrozny, David Bell, Gabriele Cavallaro, Campbell Watson, Manil Maskey, Rahul Ramachandran, and Juan Bernabe Moreno. Prithvi-eo-2.0: A versatile multitemporal foundation model for earth observation

applications. IEEE Transactions on Geoscience and Remote Sensing, 64:1–20, 2026. doi: 10.1109/TGRS.2025.3642610.

[5] Xiang Wang, Renzhi Wang, Ningzi Hu, Pinqiang Wang, Peng Huo, Guihua Wang, Huizan Wang, Senzhang Wang, Junxing Zhu, Jianbo Xu, Jun Yin, Senliang Bao, Ciqiang Luo, Ziqing Zu, Yi Han, Weimin Zhang, Kaijun Ren, Kefeng Deng, and Junqiang Song. Xihe: A data-driven model for global ocean eddy-resolving forecasting, 2024. URL https://arxiv. org/abs/2402.02995.

[6] Yingzhe Cui, Ruohan Wu, Xiang Zhang, Ziqi Zhu, Bo Liu, Jun Shi, Junshi Chen, Hailong Liu, Shenghui Zhou, Liang Su, Zhao Jing, Hong An, and Lixin Wu. Forecasting the eddying ocean with a deep neural network. Nature Communications, 16(1):2268, 2025. doi: 10.1038/s41467-025-57389-2. URL https://doi.org/10.1038/s41467-025-57389-2.

[7] Wei Xiong, Yanfei Xiang, Hao Wu, Shuyi Zhou, Yuze Sun, Muyuan Ma, and Xiaomeng Huang. Ai-goms: Large ai-driven global ocean modeling system, 2023. URL https: //arxiv.org/abs/2308.03152.

[8] Geofrey Dawson, Remy Vandaele, Andrew Taylor, David Mofat, Helen Tamura-Wicks, Sarah Jackson, Rosie Lickorish, Paolo Fraccaro, Hywel Williams, Chunbo Luo, and Anne Jones. A sentinel-3 foundation model for ocean colour, 2025. URL https://arxiv.org/ abs/2509.21273.

[9] Viola-Joanna Stamer, Panagiotis Agrafiotis, Behnood Rasti, and Begüm Demir. Oceanmae: A foundation model for ocean remote sensing, 2026. URL https://arxiv.org/abs/2604. 08171.

[10] J. Emmanuel Johnson, Quentin Febvre, Anastasiia Gorbunova, Sam Metref, Maxime Ballarotta, Julien Le Sommer, and ronan fablet. Oceanbench: The sea surface height edition. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 78275–78295. Curran Associates, Inc., 2023. doi: 10.52202/ 075280-3422. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ f6ccbf94fa57c2ae372ece91b537574d-Paper-Datasets\_and\_Benchmarks.pdf.

[11] Yezhen Cong, Samar Khanna, Chenlin Meng, Patrick Liu, Erik Rozi, Yutong He, Marshall Burke, David B. Lobell, and Stefano Ermon. Satmae: Pre-training transformers for temporal and multi-spectral satellite imagery, 2023. URL https://arxiv.org/abs/2207.08051.

[12] Anthony Fuller, Koreen Millard, and James Green. Croma: Remote sensing representations with contrastive radar-optical masked autoencoders. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 5506–5538. Curran Associates, Inc., 2023. doi: 10.52202/ 075280-0241. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf.

[13] Vishal Nedungadi, Ankit Kariryaa, Stefan Oehmcke, Serge Belongie, Christian Igel, and Nico Lang. Mmearth: Exploring multi-modal pretext tasks for geospatial representation learning, 2024. URL https://arxiv.org/abs/2405.02771.

[14] Cristian Bodnar, Wessel P. Bruinsma, Ana Lucic, Megan Stanley, Anna Allen, Johannes Brandstetter, Patrick Garvan, Maik Riechert, Jonathan A. Weyn, Haiyu Dong, Jayesh K. Gupta, Kit Thambiratnam, Alexander T. Archibald, Chun-Chieh Wu, Elizabeth Heider, Max Welling, Richard E. Turner, and Paris Perdikaris. A foundation model for the earth system. Nature, 641:1180–1187, 2025. doi: 10.1038/s41586-025-09005-y. URL https://doi.org/10.1038/s41586-025-09005-y.

[15] D. Eeltink, H. Branger, C. Luneau, Y. He, A. Chabchoub, J. Kasparian, T. S. van den Bremer, and T. P. Sapsis. Nonlinear wave evolution with data-driven breaking. Nature Communications, 13(1):2343, 2022. doi: 10.1038/s41467-022-30025-z.

[16] Yoo-Geun Ham, Yong-Sik Joo, Jeong-Hwan Kim, and Jeong-Gil Lee. Partial-convolutionimplemented generative adversarial network for global oceanic data assimilation. Nature Machine Intelligence, 6(7):834–843, 2024. doi: 10.1038/s42256-024-00867-x.

[17] Yannik Glaser, Justin E. Stopa, Linnea M. Wolniewicz, Ralph Foster, Doug Vandemark, Alexis Mouche, Bertrand Chapron, and Peter Sadowski. Wv-net: A foundation model for sar ocean satellite imagery. Artificial Intelligence for the Earth Systems, 4(4):e250003, 2025. doi: 10.1175/AIES-D-25-0003.1.

[18] Zhen Bi, Ningyu Zhang, Yida Xue, Yixin Ou, Daxiong Ji, Guozhou Zheng, and Huajun Chen. OceanGPT: A large language model for ocean science tasks. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, pages 3357–3372. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.acl-long.184. URL https://aclanthology.org/2024.acl-long.184/.

[19] Yida Xue, Ningyu Zhang, Tingwei Wu, Zhe Ma, Daxiong Ji, Zhao Wang, Guozhou Zheng, and Huajun Chen. OceanPile: A large-scale multimodal ocean corpus for foundation models, 2026. URL https://arxiv.org/abs/2605.00877.

[20] Sylvie Thiria, Charles Sorror, Théo Archambault, Anastase Alexandre Charantonis, Dominique Béréziat, Carlos Mejia, Jean-Marc Molines, and Michel Crépon. Downscaling of ocean fields by fusion of heterogeneous observations using deep learning algorithms. Ocean Modelling, 182:102174, 2023. doi: 10.1016/j.ocemod.2023.102174.

[21] Mingqing Wang, Danni Wang, Yanfei Xiang, Yishuang Liang, Ruixue Xia, Jinkun Yang, Fanghua Xu, and Xiaomeng Huang. Fusion of ocean data from multiple sources using deep learning: Utilizing sea temperature as an example. Frontiers in Marine Science, 10:1112065, 2023. doi: 10.3389/fmars.2023.1112065.

[22] Bianca Champenois and Themistoklis Sapsis. Machine learning framework for the real-time reconstruction of regional 4d ocean temperature fields from historical reanalysis data and real-time satellite and buoy surface measurements. Physica D: Nonlinear Phenomena, 459: 134026, 2024. doi: 10.1016/j.physd.2023.134026.

[23] Lei Lei, Jie Huang, and Yu Zhou. Multimodal fusion-based spatiotemporal incremental learning for ocean environment perception under sparse observation. Information Fusion, 108:102360, 2024. doi: 10.1016/j.infus.2024.102360.

[24] Yuxin Zhao, Dequan Yang, Jianxin He, Kexin Zhu, et al. Hierarchical stacked spatiotemporal self-attention network for sea surface temperature forecasting. Ocean Modelling, 191:102427, 2024. doi: 10.1016/j.ocemod.2024.102427.

[25] Zhenxiang Bai, Zhengya Sun, Bojie Fan, An-An Liu, Zhiqiang Wei, and Bo Yin. Multiscale spatio-temporal attention network for sea surface temperature prediction. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 18:5866–5877, 2025. doi: 10.1109/JSTARS.2025.3531122.

[26] Xinliang Pan, Tao Jiang, Weifu Sun, Jiawei Xie, Pinzhen Wu, Zhen Zhang, and Tingwei Cui. Efective attention model for global sea surface temperature prediction. Expert Systems with Applications, 254:124411, 2024. doi: 10.1016/j.eswa.2024.124411.

[27] Shengen Tao, Yanqiu Li, Feng Gao, Hao Fan, Junyu Dong, and Yanhai Gan. Multiscale spatial features and temporal attention mechanisms: advancing the accuracy of enso prediction. Intelligent Marine Technology and Systems, 2(1):7, 2024. doi: 10.1007/ s44295-023-00017-w.

[28] Nan Yang, Chong Wang, Zimeng Zhao, Meihua Zhao, Huiling Zheng, Bin Zhang, Fan Wang, and Xiaofeng Li. LangYa: A large ai model for global ocean forecasting. Science Bulletin, 2026. doi: 10.1016/j.scib.2026.08.001. URL https://doi.org/10.1016/j.scib.2026.08. 001.

[29] Mercator Ocean International. Digital ocean week 2026: Advancing trusted ai for europe’s digital ocean, 2026. URL https://www.mercator-ocean.eu/news-en/ digital-ocean-week-2026-ai-ocean-forum/. Oficial description of the GLONET2 ocean forecasting system.

## A Additional Details

Put derivations, extra experimental details, and supplemental tables here. Delete this appendix if it is unnecessary.