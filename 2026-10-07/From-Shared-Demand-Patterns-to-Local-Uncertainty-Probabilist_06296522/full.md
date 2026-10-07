# From Shared Demand Patterns to Local Uncertainty: Probabilistic Load Forecasting by Mixing Compact Adaptations

Haoran Li, Member, IEEE, Zhe Cheng, Student Member, IEEE, Yang Weng, Senior Member, IEEE,

Abstract—Probabilistic load forecasting has been widely studied for power-system operation and planning, but customerand transformer-level forecasting introduces a distinct scalability challenge. At these levels, load uncertainty is strongly affected by customer behavior, weather, and mixed load composition, making it difficult for a single shared model to capture heterogeneous patterns. Using separate probabilistic models can improve local accuracy, but becomes costly to train, store, update, and validate at scale. To address this challenge, we develop a scalable customer-aware forecasting framework that learns common demand behavior through a shared model while adapting only a compact subset of parameters. Rather than using an independent model for each load or assigning each load to a specialized model, the proposed design learns a small bank of low-dimensional adaptation components and allows each load to combine them according to its forecasting characteristics. This preserves shared knowledge across customers while providing sufficient flexibility for heterogeneous and mixed load compositions. Experiments on 590 load profiles from the SMART-DS dataset show consistent improvements in deterministic accuracy and probabilistic quality over statistical, neural-network, Transformer-based, and pretrained time-series baselines, while retaining low storage and inference costs. The source code and data processing pipeline are available at M<sup>2</sup>oE<sup>2</sup>-mLoRA.

Index Terms—Probabilistic load forecasting, distribution systems, uncertainty quantification, parameter-efficient learning, low-rank adaptation, mixture of experts.

## I. INTRODUCTION

Load forecasting is a foundational function for modern power systems, providing essential information for economic dispatch [1], demand response [2], reliability assessment [3], event detection [4]–[6], and infrastructure planning [7]. As smart meters and advanced sensing infrastructures become increasingly available, forecasting practice is moving beyond aggregate system-level prediction toward fine-grained, bottom-up forecasting for individual customers in distribution feeders [8]–[10]. This shift is particularly important for distribution-system operation and can support distributed energy resource coordination [11], demand-side management [12], local market participation [13], and capacity planning [14]. Compared with point forecasting, probabilistic load forecasting further characterizes predictive uncertainty and provides richer information for risk-aware decision-making under variable and disaggregated demand [15].

Nevertheless, customer- and transformer-level probabilistic forecasting remains highly challenging in realistic distribution feeders with massive heterogeneous loads, including residential, commercial, industrial, and mixed transformer-level demand. Unlike more aggregated load forecasting, local power measurements reflect the combined effects of largely unobserved factors such as occupancy, business activity, appliance usage, electric-vehicle charging, distributed energy resources, and customer composition behind transformers. These factors can produce strong differences in consumption and uncertainty patterns across customers, phases, locations, and aggregation levels, making it difficult for a single global forecaster to fully capture local behavior [16], [17]. At the same time, training and maintaining an independent deep probabilistic model for each customer or transformer is impractical for utilities, as storage, computation, and model-management costs grow rapidly with the number of loads. Therefore, scalable forecasting requires a mechanism that shares common knowledge across massive customers while adapting compactly to customer-specific demand and uncertainty characteristics.

Early approaches address local heterogeneity by building separate forecasting models for different customers, buildings, or load groups. Clustering-based methods partition customers according to load profiles and train separate predictors for the resulting clusters [18], [19]. Transfer-learning-based methods instead transfer knowledge learned from data-rich source loads, buildings, or customer groups to related target domains with limited data [20]–[22]. While clustering can substantially reduce complexity compared with maintaining an independent forecaster for every customer, storage and model-management costs may still become considerable when many clusters are required to represent heterogeneous loads. Transfer learning improves knowledge reuse, but often relies on source-to-target transfer and repeated adaptation across different loads.

The third category follows a shared-model adaptation strategy, where knowledge is learned jointly from many related loads and then adapted to specific customers or customer groups. Meta-learning methods learn a shared initialization or meta-model that can be quickly adapted to target household loads using limited local data [23], [24], but typically require an additional learning-to-adapt optimization procedure. Personalized federated learning trains a common model across consumers while allowing local parameter or gradient updates to preserve customer-specific characteristics [25]– [27]. Hypernetwork-based methods instead generate customerspecific model parameters or compact sub-parameters conditioned on customer information [10], [28], while finetuning approaches adapt only selected parameters of a pretrained model for different customer types [29]. These approaches substantially improve scalability by sharing most model knowledge across loads, but still face a capacityefficiency tradeoff: the adapted parameter space must remain compact for large-scale deployment while being expressive enough to capture heterogeneous customer behavior and mixed transformer-level load compositions.

To address the capacity-efficiency challenge, we first investigate where customer-specific adaptation should be introduced within a shared probabilistic forecaster. A key observation from our parameter-selection and ablation tests is that the forecasting model can be naturally separated into two functional stages: learning generalizable temporal and contextual representations across customers, and adapting these shared representations to customer-specific demand and uncertainty patterns. While the former can be largely shared, different customers may require different mappings from the common representation to their predictive distributions. This suggests that customer-specific adaptation should be concentrated at the post-fusion prediction stage, identifying the probabilistic output head as the key parameter subset for adaptation.

We therefore introduce low-rank adaptation (LoRA) [30] into the output head, freezing the shared parameters while learning compact low-rank updates for different load groups. This preserves the common output-mapping structure while substantially reducing the number of adapted parameters. However, a single LoRA update restricts adaptation and may not provide sufficient flexibility for heterogeneous customers and mixed transformer-level loads; simply increasing its rank improves capacity but also increases parameter cost. We thus propose a mixture-of-LoRA mechanism that represents each adaptation update as a weighted combination of multiple lowrank components, allowing diverse adaptation patterns to be composed from a small reusable parameter bank. The shared probabilistic forecaster is implemented using $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ [10], which incorporates external information as meta-knowledge and has demonstrated strong performance on benchmarks including GEFCom2014 [31].

The resulting model is termed $\mathbf { M } ^ { 2 } \mathbf { o } \mathbf { E } ^ { 2 } \mathbf { - m } \mathbf { L } \mathbf { o } \mathbf { R } \mathbf { A }$ . We evaluate it on 590 residential, commercial, and industrial load profiles from the SMART-DS dataset [32] against statistical models, recurrent and convolutional neural networks, Transformers, and pretrained time-series foundation models. The evaluation covers deterministic accuracy, probabilistic forecasting quality, inference time, and storage cost. Results show that $\bar { \mathbf { M } } ^ { 2 } \mathbf { o } \bar { \mathbf { E } } ^ { 2 }$ -mLoRA improves both deterministic and probabilistic forecasting performance while maintaining low computational and storage overhead, demonstrating a favorable accuracy– uncertainty–cost tradeoff for customer- and transformer-level forecasting.

The rest of the paper is as follows. Section II formulates the problem. Section III provides preliminaries of $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ . Section IV illustrates $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 } \mathbf { - } \mathbf { m } \mathrm { L o } \mathbf { R } \mathbf { A }$ framework. Section V presents numerical results, and Section VI concludes the paper.

## II. PROBLEM FORMULATION

We define the customer-aware probabilistic load forecasting problem for massive heterogeneous feeder loads as follows.

• Goal: Learn a scalable probabilistic forecasting model that shares common knowledge across massive customerand transformer-level loads, while adapting compactly to different load groups in realistic distribution feeders.

• Given: A customer/transformer index set C, where each $c \in { \mathcal { C } }$ denotes one customer-level or transformer-level load. The historical load measurements of c are denoted as $\{ x _ { c , i } \} _ { i = 1 } ^ { N }$ , where N is the number of measurement points. Optional external measurements are denoted as $\{ w _ { j , i } \mid j = 1 , \cdots , M ; i = 1 , \cdots , N \}$ , where M is the number of external data sources, such as weather, calendar variables, and electricity prices. The set C may contain K load groups, including residential, commercial, industrial, and transformer-level loads with different mixed customer compositions.

• Find: A shared $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ probabilistic forecaster $f _ { \Theta }$ and a set of adaptation parameters $\{ \theta _ { k } \} _ { k = 1 } ^ { K }$ , where Θ denotes the shared global model parameters and $\theta _ { k }$ denotes the LoRA parameters associated with the k-th load cluster.

## III. PRELIMINARY: $\mathbf { M } ^ { 2 } \mathrm { O E } ^ { 2 }$ GLOBAL FORECASTER

This section briefly reviews our previous $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ probabilistic forecaster [10], which serves as the shared global forecasting backbone $f _ { \Theta }$ in this paper. The model is applicable to each customer or transformer $c \in { \mathcal { C } }$ and is trained using load data from all $c \in { \mathcal { C } } .$ . For notational simplicity, we omit the customer/transformer subscript c in this section. $\mathbf { \dot { M } } ^ { 2 } \mathbf { o E } ^ { 2 }$ builds on a probabilistic sequence base model and further treats external data as meta-knowledge to guide feature extraction. It uses MetaNets to transform heterogeneous external sources into candidate experts, a sparse gating network to select informative experts, and a residual anchor to preserve the noexternal-data case. For more details, please review [10].

## A. Variational Sequence As A Base Model

Let $\{ x _ { i } \} _ { i = 1 } ^ { N }$ denote a load measurement sequence, where $x _ { i }$ is a scalar load measurement. A sequence base model, such as RNN, GRU, or LSTM, first encodes historical load information into a hidden state $\boldsymbol { h } _ { i }$ . For example, an RNN-type base model updates the hidden state as

$$
\pmb { h } _ { i } = \rho \left( W _ { h x } x _ { i } + W _ { h h } \pmb { h } _ { i - 1 } + \pmb { b } _ { h } \right) ,\tag{1}
$$

where $\rho ( \cdot )$ is the activation function, and $W _ { h x } , \ W _ { h h }$ , and $ { \boldsymbol { b } } _ { h }$ are trainable parameters. The hidden state $\boldsymbol { h } _ { i }$ summarizes historical load patterns and provides the temporal feature for future forecasting.

To quantify uncertainty, $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ adopts a variational probabilistic forecasting structure on top of the sequence base model. A latent vector $z _ { i + 1 }$ is introduced to represent stochastic factors that affect the future load. The encoder approximates the posterior distribution of $z _ { i + 1 }$ as

$$
\begin{array} { r l } & { q _ { \Theta } \left( z _ { i + 1 } ~ | ~ x _ { i } , { h } _ { i - 1 } \right) } \\ & { ~ = \mathcal { N } \Big ( \mu _ { z } ( x _ { i } , { h } _ { i - 1 } ) , \mathrm { d i a g } \big ( \pmb { \sigma } _ { z } ^ { 2 } ( x _ { i } , { h } _ { i - 1 } ) \big ) \Big ) . } \end{array}\tag{2}
$$

where $\mu _ { z } ( \cdot )$ and $\sigma _ { z } ^ { 2 } ( \cdot )$ are neural-network mappings. The decoder then predicts the mean and variance of the future load distribution:

$$
p _ { \Theta } \left( x _ { i + 1 } \mid z _ { i + 1 } \right) = \mathcal { N } \left( \mu _ { x } \left( z _ { i + 1 } \right) , \sigma _ { x } ^ { 2 } \left( z _ { i + 1 } \right) \right) .\tag{3}
$$

The model is trained by minimizing the negative evidence lower bound:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { E L B O } } ( \Theta ) = - \mathbb { E } _ { q _ { \Theta } ( z _ { i + 1 } \mid x _ { i } , h _ { i - 1 } ) } \left[ \log p _ { \Theta } \left( x _ { i + 1 } ~ \middle \vert ~ z _ { i + 1 } \right) \right] } \\ { + \lambda \mathrm { K L } \left( q _ { \Theta } \left( z _ { i + 1 } ~ \middle \vert ~ x _ { i } , h _ { i - 1 } \right) \Vert p \left( z _ { i + 1 } \right) \right) , } \end{array}\tag{4}
$$

where $p \left( z _ { i + 1 } \right) \ = \ { \mathcal { N } } ( \mathbf { 0 } , I )$ $\operatorname { K L } ( \cdot )$ is the Kullback–Leibler (KL) divergence-based regularization, and λ balances the likelihood term and latent-space regularization.

In general, the variational sequence base model maps historical load information into a conditional predictive distribution for future loads. Depending on the forecasting requirement, the decoder can directly provide distributional parameters, such as the predictive mean and variance, or generate Monte Carlo samples by repeatedly drawing $z _ { i + 1 }$ and decoding the corresponding load realizations. These samples can further be used to estimate other probabilistic statistics, including quantiles and prediction intervals, making the base model flexible for both point and probabilistic forecasting. Such a model serves as the base model for $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ , as shown in the GRU encoder and mean/variance head blocks in Fig. 1.

## B. External Meta-Knowledge through MetaNet and Gating

After establishing the variational sequence base model, the next step is to incorporate external data in a way that helps the model adapt to changing environmental and contextual factors. Instead of simply concatenating external variables with load inputs, $\mathbf { M } ^ { 2 } \mathrm { o E } ^ { \hat { 2 } }$ treats them as meta-knowledge that dynamically guides how the base model extracts load features.

For each time index i, optional external measurements are denoted as $\{ w _ { j , i } \mid j = 1 , \cdot \cdot \cdot , M \}$ , where M is the number of external data sources, such as weather, calendar variables, and electricity prices. Thus, we introduce the following modules to construct the final meta representation $\phi _ { i }$ in Eq. 6.

Module 1. MetaNet for building external experts. For the j-th external source, a MetaNet $g _ { j } ( \cdot )$ transforms $w _ { j , i }$ into a candidate meta-representation:

$$
\psi _ { j , i } = \mathrm { L N } \left( g _ { j } \left( w _ { j , i } \right) \right) , \quad j = 1 , \cdots , M ,\tag{5}
$$

where LN(·) denotes layer normalization. The layer normalization reduces scale and distributional heterogeneity among different external sources, so that each $\psi _ { j , i }$ can be treated as a candidate expert for guiding the forecasting model.

Module 2. Gating network for selecting informative experts. Because the usefulness of external sources changes over time, $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ employs a gating network $l ( \cdot )$ to automatically select informative experts according to the historical hidden state. The combined meta-representation is

$$
\phi _ { i } = \sum _ { j = 1 } ^ { M } l _ { j , i } \psi _ { j , i } + \phi _ { 0 } ,\tag{6}
$$

where $l _ { j , i } ~ = ~ l ( h _ { i - 1 } ) [ j ]$ is the gating weight for the j-th external expert and ϕ<sub>0</sub> is a trainable residual anchor. A top-m sparse softmax is applied to activate only the most relevant experts:

$$
l _ { j , i } = \frac { \exp { \left( \tilde { l } _ { j , i } \right) } \mathbb { I } \left( j \in \mathcal { T } _ { i } \right) } { \sum _ { r \in \mathcal { T } _ { i } } \exp { \left( \tilde { l } _ { r , i } \right) } } ,\tag{7}
$$

where $\tilde { l } _ { j , i }$ is the pre-activation score, $\mathcal { T } _ { i }$ contains the indices of the selected top-m experts, and I(·) is the indicator function. Module 3. Skip connection for preserving the no-externaldata case. The residual anchor $\phi _ { 0 }$ allows $\bar { \mathbf { M } } ^ { 2 } \mathbf { o } \mathbf { E } ^ { 2 }$ to fall back to the base forecasting structure when external information is uninformative. Specifically, if the selected external contribution vanishes, i.e., $\begin{array} { r } { \sum _ { j = 1 } ^ { M } \dot { l } _ { j , i } \psi _ { j , i } = \mathbf { 0 } } \end{array}$ , then $\phi _ { i } = \phi _ { 0 }$ . Therefore, the external-data module can be bypassed, preventing the model from being worse than the corresponding no-externaldata base model. The MetaNets, gate, and skip-connection are shown in the top-left part of Fig. 1.

## C. Plug-and-Play Parameter Modulation

The meta-representation $\phi _ { i }$ can be used to modulate different sub-parameters of the sequence base model in Subsection $\mathrm { I I I - A }$ , such as input-to-hidden $W _ { h x } .$ hidden-to-hidden $W _ { h h }$ , or output-layer parameters in $\mu _ { x }$ and $\sigma _ { x }$ . In $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ , a detailed analysis of parameter size, implementation cost, and adaptation effect shows that modulating the input-to-hidden matrix $W _ { h x }$ is an efficient choice: it allows external conditions to change how load inputs are encoded into temporal features while avoiding the large overhead of modulating recurrent or output parameters.

Directly replacing the internal matrix $W _ { h x }$ in Eq. (1), however, requires modifying the internal implementation of the sequence base model. To obtain a plug-and-play design, $\mathbf { M } ^ { 2 } \mathbf { o } \mathbf { E } ^ { 2 }$ uses an equivalent input transformation:

$$
\mathbf { \Delta } x _ { i } ^ { \prime } = \phi _ { i } x _ { i } .\tag{8}
$$

The transformed input $\mathbf { \Delta } \mathbf { x } _ { i : } ^ { \prime }$ , as shown on the top left of Fig. 1, is then fed into the sequence base model. This formulation enables external meta-knowledge to adapt the feature extraction without changing the internal architecture of the base model. As a result, $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ can be integrated with different sequence base models, including RNN, GRU, and LSTM, and serves as an efficient global probabilistic forecasting backbone $f _ { \Theta }$ . Here, $\Theta = \Theta _ { \mathrm { b a s e } } \cup \Theta _ { \mathrm { m e t a } } \cup \Theta _ { \mathrm { g a t e } } \cup \{ \phi _ { 0 } \}$ includes the parameters of the variational sequence base model, the MetaNets $\{ g _ { j } ( \cdot ) \} _ { j = 1 } ^ { M }$ the gating network $l ( \cdot )$ , and the residual anchor ϕ<sub>0</sub>.

In this paper, we further extend this backbone from external-data-aware forecasting to customer-aware forecasting by adapting sub-parameters $\{ \theta _ { k } \} _ { k = 1 } ^ { K }$ for different load groups $k \in \{ 1 , 2 , \cdots , K \}$ , defined in Section II.

## IV. PROPOSED MODEL

In this section, we propose $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ -mLoRA, as shown in Fig. 1. In the upper-left module (Stage 1), load history and external information are used to train a shared probabilistic forecaster; the pretrained backbone is then fixed during adaptation. In the lower-left module (Stage 2), load-profile features are clustered to identify representative customer and transformer patterns and determine the mixture weights used for adaptation. In the right module (Stage 3), these weights combine a shared bank of LoRA components to form customer-specific adaptations, which are applied to the probabilistic output heads for mean and variance prediction (Stage 4). This design reuses the same pretrained backbone across all loads while adapting only a compact set of parameters, avoiding full-model finetuning and supporting scalable deployment.

## A. Load Clustering for Customer-Specific Fine-Tuning

Although the globally pre-trained $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ backbone captures common temporal and external-context knowledge across massive loads, it may not fully address the strong heterogeneity among residential, commercial, industrial, and transformerlevel loads. Therefore, after global pre-training, customeraware adaptation is required to further specialize the model to different load behaviors by re-updating selected subparameters using specific groups of load data. Before performing such fine-tuning, it is necessary to understand how the load data are distributed across different behavioral patterns. To this end, we first conduct load clustering based on profile characteristics, so that loads with similar consumption patterns can provide structured guidance for the subsequent mixtureof-LoRA adaptation.

![](images/54fa1bac51232dc78d843f665ab69e2d5c632c45b3756f96e4796ce859385055.jpg)  
Fig. 1: $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 } .$ -mLoRA framework.

For each customer or transformer $c \in { \mathcal { C } } .$ , we use its historical training load data to construct one clustering representation. Specifically, the training data of load c are divided into $Q$ weekly profiles, where each weekly profile is denoted by $\pmb { x } _ { c , q } \in \mathbb { R } ^ { S } \mathrm { ~ ( 1 ~ \leq ~ } q \mathrm { ~ \leq ~ } Q \mathrm { ) ~ }$ , and $S$ is the number of time slots in one week. Since clustering is expected to capture the load-shape characteristics rather than the absolute load magnitude, each weekly profile is normalized by the maximum training-period load of customer c. Specifically, we have: $\begin{array} { r } { x _ { c } ^ { \mathrm { m a x } } = \mathrm { m a x } _ { 1 \leq q \leq Q , 1 \leq s \leq S } x _ { c , q , s } , \tilde { \pmb { x } } _ { c , q } = \frac { { \pmb x } _ { c , q } } { x _ { c } ^ { \mathrm { m a x } } + \epsilon } } \end{array}$ , where ϵ is a small constant for numerical stability.

Then, all normalized weekly profiles are averaged to obtain a representative weekly load shape ${ \pmb p } _ { c }$ for each customer or transformer. To characterize load heterogeneity beyond the average shape, we further consider week-to-week variability and load magnitude. Specifically, we define $\begin{array} { r l } { p _ { c } } & { { } = } \end{array}$ $\begin{array} { r } { \frac { 1 } { Q } \sum _ { q = 1 } ^ { Q } \tilde { \pmb { x } } _ { c , q } } \end{array}$ and $\begin{array} { r } { \boldsymbol { v _ { c , s } } = \left\lceil \frac { 1 } { Q } \sum _ { q = 1 } ^ { \bar { Q } } \left( \tilde { x } _ { c , q , s } - p _ { c , s } \right) ^ { 2 } \right\rceil ^ { 1 / 2 } } \end{array}$ . Here, $\pmb { p _ { c } }$ captures the average normalized weekly load shape, while $\pmb { v } _ { c } = [ v _ { c , 1 } , \ldots , v _ { c , S } ] ^ { \top }$ characterizes the variability of each time slot across weeks. To retain absolute demand information removed by normalization, we further define $\begin{array} { r l } { \mathbf { \delta } a _ { c } } & { { } = } \end{array}$ $\begin{array} { r } { \left\lceil \frac { 1 } { Q S } \sum _ { q = 1 } ^ { Q } \sum _ { s = 1 } ^ { S } x _ { c , q , s } , x _ { c } ^ { \operatorname* { m a x } } \right\rceil ^ { ! } } \end{array}$ , which contains the mean and maximum load magnitudes of the original profiles. The resulting feature vector therefore jointly characterizes typical load shape, temporal variability, and demand magnitude, corresponding to the three load-profile feature components in Fig. 1:

$$
\mathbf { } f _ { c } = \left[ \pmb { p } _ { c } ^ { \top } , \pmb { v } _ { c } ^ { \top } , \pmb { a } _ { c } ^ { \top } \right] ^ { \top } .\tag{9}
$$

Finally, the feature dimensions are standardized across the training loads and reduced using principal component analysis (PCA). The resulting low-dimensional representation ${ \bf u } _ { c }$ is used for subsequent clustering.

Given the reduced profile representations $\{ { \pmb u } _ { c } \} _ { c \in \mathcal { C } } ,$ we further apply a Gaussian mixture model (GMM) to identify representative load groups. The use of GMM is motivated by the fact that load groups in realistic distribution feeders may not have clear hard boundaries. In particular, transformerlevel loads can aggregate multiple types of customers, such as residential, commercial, and industrial loads, leading to mixed profile characteristics. In addition, customer behavior may vary under different weather conditions, calendar effects, and operating regimes. Therefore, instead of assigning each load to a single deterministic cluster, GMM provides soft membership weights that describe how strongly each customer or transformer is associated with different load clusters.

$$
p ( { \pmb u } _ { c } ) = \sum _ { k = 1 } ^ { K } \omega _ { k } \mathcal { N } \left( { \pmb u } _ { c } \mid m _ { k } , \Sigma _ { k } \right) ,\tag{10}
$$

where K is the number of load groups, defined in Section II. $\omega _ { k }$ is the mixture coefficient satisfying $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \omega _ { k } = 1 } \end{array}$ , and $\mathbf { \nabla } m _ { k }$ and $\Sigma _ { k }$ are the mean vector and covariance matrix of the k-th Gaussian component. The soft membership weight of load c with respect to cluster k is given by the posterior probability

$$
\alpha _ { c , k } ^ { 0 } = \frac { \omega _ { k } \mathcal { N } \left( \pmb { u } _ { c } \mid \pmb { m } _ { k } , \Sigma _ { k } \right) } { \sum _ { r = 1 } ^ { K } \omega _ { r } \mathcal { N } \left( \pmb { u } _ { c } \mid \pmb { m } _ { r } , \Sigma _ { r } \right) } , \quad k \in \{ 1 , \cdots , K \} .\tag{11}
$$

The resulting vector ${ \pmb { \alpha } } _ { c } ^ { 0 } = [ \alpha _ { c , 1 } ^ { 0 } , \alpha _ { c , 2 } ^ { 0 } , \cdot \cdot \cdot , \alpha _ { c , K } ^ { 0 } ]$ serves as the clustering-based soft membership assignment of customer c over the K load groups. This profile construction, dimensionality reduction, and soft-membership estimation correspond to the load-profile features, PCA, and GMM blocks in the lowerleft part of Fig. 1.

## B. Efficient Selection of Adapted Sub-Parameters

The clustering provides a structured description of load distributions across different customers. However, these clustering assignments only suggest which data subsets may require distinct customer-specific adaptation; they do not specify which sub-parameters of $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ should be adapted. A direct solution is to fine-tune the whole pre-trained model for each load cluster, but this would introduce large storage and computation overhead and weaken the scalability advantage of the shared backbone. Therefore, before constructing the mixtureof-LoRA adaptation, we first identify a small but effective subset of $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ parameters for customer-specific fine-tuning.

For customer-aware forecasting, different sub-parameters of $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ play different roles. The input transformation, externaldata gating, and sequence feature extraction modules mainly learn common temporal and contextual representations shared by different loads. Directly adapting these modules for each customer group may disturb the globally learned representation and introduce unnecessary storage and computation overhead. In contrast, the probabilistic output head directly maps the learned latent representation to the predictive mean and variance, which are more sensitive to customer-specific load level, volatility, and uncertainty characteristics. Therefore, we select the output-head parameters as the adaptation target, so that each load group can adjust its predictive distribution while preserving the shared $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ backbone.

Following Eq. (3), for each customer or transformer $c \in { \mathcal { C } }$ the frozen $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ backbone first produces the latent representation $z _ { c , i + 1 }$ . The probabilistic output head then maps $z _ { c , i + 1 }$ to the mean and variance of the future load distribution:

$$
\begin{array} { r l } & { p _ { \Theta } \left( x _ { c , i + 1 } \mid z _ { c , i + 1 } \right) = \mathcal { N } \left( \mu _ { x } \left( z _ { c , i + 1 } \right) , \sigma _ { x } ^ { 2 } \left( z _ { c , i + 1 } \right) \right) , } \\ & { \quad \quad \quad \quad \quad \mu _ { x } \left( z _ { c , i + 1 } \right) = \rho \left( W _ { \mu } ^ { ( 0 ) } z _ { c , i + 1 } + b _ { \mu } ^ { ( 0 ) } \right) , } \\ & { \quad \quad \quad \quad \quad \log { \sigma _ { x } ^ { 2 } \left( z _ { c , i + 1 } \right) } = \rho \left( W _ { \sigma } ^ { ( 0 ) } z _ { c , i + 1 } + b _ { \sigma } ^ { ( 0 ) } \right) , } \end{array}\tag{12}
$$

where $W _ { \mu } ^ { ( 0 ) }$ and $W _ { \sigma } ^ { ( 0 ) }$ are the pre-trained output-head weight matrices for predicting the mean and variance, respectively, and ${ \pmb b } _ { \mu } ^ { ( 0 ) }$ and ${ \pmb b } _ { \sigma } ^ { ( 0 ) }$ are the corresponding bias terms. Here, the superscript (0) is added to $W _ { \mu } ^ { ( 0 ) }$ and $\overline { { W _ { \sigma } ^ { ( 0 ) } } }$ to denote the globally pre-trained output-head weights that remain frozen during customer-specific adaptation. We select ${ \cal W } _ { \mu } ^ { ( 0 ) }$ and $W _ { \sigma } ^ { ( 0 ) }$ as the target sub-parameters for customer-specific adaptation, as shown in the upper-right module of Fig. 1.

## C. Soft-Membership-Guided Mixture-of-LoRA

Subsequently, we need to determine how these parameters should be adapted for different customers and transformers. A straightforward strategy is to learn one independent LoRA update for each load group. However, this hard assignment ignores the soft membership information obtained from GMM and may be insufficient for transformer-level loads with mixed load profiles. Therefore, in this subsection, we propose a softmembership-guided mixture-of-LoRA mechanism.

The basis of the proposed mixture-of-LoRA mechanism is a single LoRA update for one load cluster. Therefore, we first define how LoRA adapts the selected output-head parameters for the k-th cluster. Instead of updating the full pre-trained matrices ${ \cal W } _ { \mu } ^ { ( 0 ) }$ and $W _ { \sigma } ^ { ( 0 ) }$ , LoRA represents the cluster-specific updates by low-rank decompositions:

$$
\begin{array} { l l } { { W _ { \mu } ^ { ( k ) } = W _ { \mu } ^ { ( 0 ) } + \Delta W _ { \mu } ^ { ( k ) } , } } & { { \Delta W _ { \mu } ^ { ( k ) } = B _ { \mu } ^ { ( k ) } A _ { \mu } ^ { ( k ) } , } } \\ { { W _ { \sigma } ^ { ( k ) } = W _ { \sigma } ^ { ( 0 ) } + \Delta W _ { \sigma } ^ { ( k ) } , } } & { { \Delta W _ { \sigma } ^ { ( k ) } = B _ { \sigma } ^ { ( k ) } A _ { \sigma } ^ { ( k ) } , } } \end{array}\tag{13}
$$

where $A _ { \mu } ^ { ( k ) } \in \mathbb { R } ^ { r \times d _ { z } } , B _ { \mu } ^ { ( k ) } \in \mathbb { R } ^ { d _ { x } \times r } , A _ { \sigma } ^ { ( k ) } \in \mathbb { R } ^ { r \times d _ { z } }$ , and $B _ { \sigma } ^ { ( k ) } \in \dot { \mathbb { R } } ^ { d _ { x } \times r }$ are the trainable low-rank factors for the k-th cluster, with $r \leq \operatorname* { m i n } ( d _ { z } , d _ { x } )$ . Here, $d _ { z }$ is the dimension of the latent representation $z _ { c , i + 1 }$ , and $d _ { x }$ is the output dimension of the forecasted load. In multi-step forecasting, $d _ { x }$ is usually larger than 1 because the output head predicts a future horizon; for example, $d _ { x } = 2 4$ for day-ahead hourly forecasting. During adaptation, the pre-trained matrices $W _ { \mu } ^ { ( 0 ) }$ and $W _ { \sigma } ^ { ( 0 ) }$ remain frozen, and only the low-rank factors are updated by gradients computed from the ELBO loss (see Eq. (4)) on the load data associated with the k-th cluster. Such a mixture of LoRA banks is presented on the bottom right of Fig. 1.

As discussed in Subsection IV-A, realistic load data often exhibit soft cluster memberships. Therefore, instead of selecting a single cluster-specific LoRA update, we construct a customer-specific adaptation as a linear mixture of multiple cluster-level LoRA updates. Let $\alpha _ { c , k }$ denote the mixture weight of customer or transformer c on the k-th cluster, where $\alpha _ { c , k } \geq 0$ and $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \alpha _ { c , k } = 1 } \end{array}$ . The mixture-of-LoRA update is defined as:

$$
\begin{array} { l } { { \displaystyle \Delta W _ { \mu , c } = \sum _ { k = 1 } ^ { K } \alpha _ { c , k } \Delta W _ { \mu } ^ { ( k ) } = \sum _ { k = 1 } ^ { K } \alpha _ { c , k } B _ { \mu } ^ { ( k ) } A _ { \mu } ^ { ( k ) } , } } \\ { { \displaystyle \Delta W _ { \sigma , c } = \sum _ { k = 1 } ^ { K } \alpha _ { c , k } \Delta W _ { \sigma } ^ { ( k ) } = \sum _ { k = 1 } ^ { K } \alpha _ { c , k } B _ { \sigma } ^ { ( k ) } A _ { \sigma } ^ { ( k ) } . } } \end{array}\tag{14}
$$

The adapted output-head matrices for customer or transformer c are then given by

$$
\begin{array} { r } { W _ { \mu , c } = W _ { \mu } ^ { ( 0 ) } + \Delta W _ { \mu , c } , } \\ { W _ { \sigma , c } = W _ { \sigma } ^ { ( 0 ) } + \Delta W _ { \sigma , c } . } \end{array}\tag{15}
$$

Eq. (13) studies the LoRA update for a single load group, where the k-th cluster has its own adapted output-head matrices $W _ { \mu } ^ { ( k ) }$ and $W _ { \sigma } ^ { ( k ) }$ . In contrast, Eq. (15) focuses on a specific customer $c ,$ which may have soft memberships to multiple load clusters rather than belonging to only one group. Therefore, the customer-specific output heads $W _ { \mu , c }$ and $W _ { \sigma , c }$ are constructed by combining the LoRA updates from different clusters according to the membership weights $\{ \alpha _ { c , k } \} _ { k = 1 } ^ { K } .$

The remaining question is how to determine the mixture weight $\alpha _ { c , k }$ in Eq. (14). The GMM posterior $\alpha _ { c , k } ^ { 0 }$ from Eq. (11) provides a useful intuition: for example, if customer or transformer c is very close to the first load cluster, then the first entry of $\alpha _ { c } ^ { 0 }$ will be close to 1 while the other entries will be close to 0. In this case, Eq. (14) should mainly activate the first LoRA update and suppress the others. Therefore, a direct option is to set $\alpha _ { c , k } = \alpha _ { c , k } ^ { 0 } .$

However, $\alpha _ { c , k } ^ { 0 }$ is computed only from the load-profile clustering space, while the optimal LoRA contribution is determined by the forecasting loss in the parameter-adaptation space. These two spaces are related but not identical. Thus, directly fixing $\alpha _ { c , k }$ as $\alpha _ { c , k } ^ { 0 }$ may overly restrict the mixture. To preserve the clustering intuition while allowing forecasting loss-based adjustment, we treat $\alpha _ { c } ^ { 0 }$ as the initialization of $\pmb { \alpha } _ { c }$ and introduce a trainable correction vector $\begin{array} { r l } { r _ { c } } & { { } = } \end{array}$ $[ r _ { c , 1 } , \cdots , r _ { c , K } ]$ . Consequently, we can define:

$$
\pmb { \alpha } _ { c } = \mathrm { s o f t m a x } \left( \mathrm { l o g } ( \pmb { \alpha } _ { c } ^ { 0 } + \epsilon ) + \tau \operatorname { t a n h } ( \pmb { r } _ { c } ) \right) ,\tag{16}
$$

where ϵ is a small constant for numerical stability, and $\tau \geq 0$ controls the maximum deviation from the GMM-based soft membership assignment. The log operator converts the initial membership vector $\alpha _ { c } ^ { 0 }$ from probability space to score space, so that applying softmax with $\mathbf { \nabla } _ { \mathbf { r } _ { c } } = \mathbf { 0 }$ recovers $\alpha _ { c } ^ { 0 } .$ . Meanwhile, since tanh $\mathbf { \Psi } _ { 1 } ( r _ { c } )$ is bounded within $[ - 1 , 1 ]$ , the correction term cannot arbitrarily dominate log $( \alpha _ { c } ^ { 0 } + \epsilon )$ . Therefore, the mixture starts from the clustering-based membership assignment and remains guided by $\alpha _ { c } ^ { 0 }$ during training, while still allowing forecasting-loss-driven adjustment through the learnable correction vector $\mathbf { \nabla } _ { \mathbf { r } _ { c } . }$

We generate the correction score $\mathbf { \nabla } _ { \mathbf { r } _ { c } }$ using a shared lightweight neural network. For each customer-cluster pair $( c , k )$ , the correction score is computed from the reduced customer profile ${ \pmb u } _ { c }$ and the cluster center $m _ { k } .$ , defined in Eq. (11). Specifically, we have:

$$
r _ { c , k } = g _ { \Omega } \left( [ { \pmb u } _ { c } ; { \pmb m } _ { k } ] \right) , \quad k \in \{ 1 , \cdots , K \} ,\tag{17}
$$

where $g _ { \Omega } ( \cdot )$ is a small MLP shared by all clusters. The resulting correction vector is $\pmb { r } _ { c } = [ r _ { c , 1 } , \cdots , r _ { c , K } ]$

## D. Training Algorithm and Online Prediction

In this subsection, we describe the offline training and online prediction procedures of the proposed M<sup>2</sup>oE<sup>2</sup>-mLoRA. The offline training procedure, summarized in Algorithm 1, consists of four stages: global pre-training, load clustering, clusterwise LoRA pretraining, and joint mixture-of-LoRA adaptation. First, the $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ backbone is trained on all available loads by minimizing the ELBO objective in Eq. (4), and the pretrained backbone is then frozen. Second, feature preparation, PCA, and GMM are applied to obtain the clustering-based soft membership assignment $\alpha _ { c } ^ { 0 } .$ . Third, each single-cluster LoRA basis in Eq. (13) is pretrained using the load data associated with its corresponding cluster. Since GMM provides soft memberships, the contribution of customer or transformer c to the k-th LoRA pretraining loss is weighted by $\alpha _ { c , k } ^ { 0 } \mathrm { : }$

Algorithm 1 Offline Training of $\mathbf { M } ^ { 2 } \mathbf { o } \mathbf { E } ^ { 2 } \mathbf { - m } \mathbf { L } \mathbf { o } \mathbf { R } \mathbf { A }$   
Require: Training load data $\{ \{ x _ { c , i } \} _ { i = 1 } ^ { N _ { c } } \} _ { c \in \mathcal { C } } ,$ , external data   
$\{ w _ { j , i } \} _ { j = 1 } ^ { M } .$ , number of clusters $K ,$ , LoRA rank $r ,$ and   
correction bound τ.   
Ensure: Frozen backbone $\Theta ^ { ( 0 ) }$ , load features and statistics $f _ { c } ,$   
GMM parameters $\{ \omega _ { k } , \pmb { m } _ { k } , \pmb { \Sigma } _ { k } \} _ { k = 1 } ^ { K } ,$ mixture-of-LoRA   
factors $\Phi _ { \mathrm { L o R A } }$ , and parameters Ω in $g _ { \Omega } ( \cdot )$   
1: Step 1: Global pre-training   
2: Train the $\mathbf { M } ^ { 2 } \mathbf { o } \mathbf { E } ^ { 2 }$ backbone using all loads by minimizing   
the ELBO objective in Eq. (4).   
3: Obtain the pretrained $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ , i.e., $f _ { \Theta ^ { ( 0 ) } }$ , and freeze $\Theta ^ { ( 0 ) }$   
4: Step 2: Load clustering   
5: for each customer or transformer $c \in { \mathcal { C } }$ do   
6: Construct the load-profile feature vector $f _ { c }$ by Eq. (9).   
7: end for   
8: Conduct PCA dimension reduction and obtain ${ \pmb u } _ { c } .$   
9: Fit a GMM with $K$ clusters on $\{ { \pmb u } _ { c } \} _ { c \in \mathcal { C } }$ using Eq. (10).   
10: Obtain the soft membership assignment $\alpha _ { c } ^ { 0 }$ by Eq. (11).   
11: Step 3: Cluster-wise LoRA pretraining   
12: for each load cluste $k = 1 , \cdots , K$ do   
13: Pretrain the single-cluster LoRA basis in Eq. (13) by   
minimizing the weighted ELBO loss in Eq. (18).   
14: Obtain the pretrained LoRA basis $\Phi _ { \mathrm { L o R A } } ^ { ( k ) , \mathrm { i n i t } ^ { \mathbf { i } } }$   
15: end for   
16: Set $\Phi _ { \mathrm { { L o R A } } }  \{ \Phi _ { \mathrm { { L o R A } } } ^ { ( k ) , \mathrm { { i n i t } } } \} _ { k = 1 } ^ { K } ,$   
17: Step 4: Joint mixture-of-LoRA adaptation   
18: while not converged do   
19: Sample a mini-batch of customer-time pairs B.   
20: for each sampled customer or transformer c in B do   
21: Compute the correction vector $\mathbf { \nabla } _ { \mathbf { r } _ { c } }$ by Eq. (17).   
22: Compute the bounded corrected mixture weight $\pmb { \alpha } _ { c }$   
by Eq. (16).   
23: Construct $W _ { \mu , c }$ and $W _ { \sigma , c }$ by Eqs. (14) and (15).   
24: end for   
25: Obtain the predictive mean and variance by replacing   
$W _ { \mu } ^ { ( 0 ) }$ and $\dot { W } _ { \sigma } ^ { ( 0 ) }$ in Eq. (12) with the customer-specific   
adapted matrices $W _ { \mu , c }$ and $W _ { \sigma , c } .$   
26: Compute the ELBO-based loss in Eq. (4) on the sam  
pled mini-batch B.   
27: Update only $\Phi _ { \mathrm { L o R A } }$ and Ω in g<sub>Ω</sub>(·) by backpropagation.   
28: end while   
29: return $\Theta ^ { ( 0 ) } , \{ \omega _ { k } , \pmb { m } _ { k } , \pmb { \Sigma } _ { k } \} _ { k = 1 } ^ { K } , \ \Phi _ { \mathrm { L o R A } }$ , and Ω.

$$
\begin{array} { r l } { \Phi _ { \mathrm { L o R A } } ^ { ( k ) , \mathrm { i n i t } } = \displaystyle \arg \operatorname* { m i n } _ { \Phi _ { \mathrm { L o R A } } ^ { ( k ) } } \sum _ { c \in \mathcal { C } } \alpha _ { c , k } ^ { 0 } } & { } \\ { \qquad \quad \times \displaystyle \sum _ { i = 1 } ^ { N _ { c } } \mathcal { L } _ { \mathrm { E L B O } } \left( x _ { c , i + 1 } ; W _ { \mu } ^ { ( k ) } , W _ { \sigma } ^ { ( k ) } \right) , } \end{array}\tag{18}
$$

where $\Phi _ { \mathrm { L o R A } } ^ { ( k ) } = \{ A _ { \mu } ^ { ( k ) } , B _ { \mu } ^ { ( k ) } , A _ { \sigma } ^ { ( k ) } , B _ { \sigma } ^ { ( k ) } \}$ . The resulting pretrained single-cluster LoRA bases are then used to initialize the mixture-of-LoRA bank: $\Phi _ { \mathrm { { L o R A } } }  \{ \Phi _ { \mathrm { { L o R A } } } ^ { ( k ) , \mathrm { { i n i t } } } \} _ { k = 1 } ^ { K }$ . This warm-start step gives each LoRA basis an explicit clusterlevel meaning before the final joint adaptation stage. After the warm-start, the cluster-level LoRA bases and the correction network $g _ { \Omega } ( \cdot )$ are jointly optimized while the $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ backbone remains frozen. During joint adaptation, only $\Phi _ { \mathrm { L o R A } }$ and Ω are updated by minimizing the ELBO-based loss in Eq. (4), where the predictive mean and variance are computed using the customer-specific adapted output heads $W _ { \mu , c }$ and $W _ { \sigma , c } .$

TABLE I: Parameter-efficiency comparison of different customer-specific adaptation strategies.
<table><tr><td>Method</td><td>Required Parameters for Adaptation</td><td>Complexity</td><td>Scaling w.r.t. |C|</td></tr><tr><td>Full output-head adaptation</td><td> $\{ W _ { \mu , c } , W _ { \sigma , c } \} _ { c \in \mathcal { C } }$ </td><td> $O ( 2 | C | d _ { x } d _ { z } )$ </td><td>Linear in |C|</td></tr><tr><td>Independent LoRA per customer</td><td> $\{ A _ { \mu , c } , \bar { B _ { \mu , c } } , A _ { \sigma , c } , \bar { B _ { \sigma , c } } \} _ { c \in \mathcal { C } }$ </td><td> $O ( 2 | \dot { c } | \dot { r } ( \dot { d } _ { x } + \dot { d } _ { z } ) )$   $\mathcal { O } ( 2 K r ( d _ { x } + d _ { z } ) )$ </td><td>Linear in |C|</td></tr><tr><td>Proposed  $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 } .$  -mLoRA</td><td> $\{ A _ { \mu } ^ { ( k ) } , B _ { \mu } ^ { ( k ) } , A _ { \sigma } ^ { ( k ) } , B _ { \sigma } ^ { ( k ) } \} _ { k = 1 } ^ { K } , g _ { \Omega } , \mathrm { P C A } ,$  GMM</td><td> $+ \mathcal { O } ( 2 d _ { u } + 1 )$   $+ \mathcal { O } ( \mathit { S } d _ { u } + \mathit { S } )$   $\begin{array} { r l r } { \mathrm { ~ } } & { { } } & { + \mathcal { O } \Bigl ( K d _ { u } + \frac { K d _ { u } ( d _ { u } + 1 ) } { 2 } + K \Bigr ) } \end{array}$ </td><td>Independent of |C|</td></tr></table>

During online prediction, the frozen $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ backbone is reused for all customers. For each customer or transformer $c ,$ the trained PCA, GMM, correction network, and mixtureof-LoRA factors are used to construct the customer-specific adapted output heads $W _ { \mu , c }$ and $W _ { \sigma , c }$ . The final predictive mean and variance are then obtained by applying Eq. (12) with ${ \cal W } _ { \mu } ^ { ( 0 ) }$ and $W _ { \sigma } ^ { ( 0 ) }$ replaced by $W _ { \mu , c }$ and $W _ { \sigma , c }$

## E. Parameter Efficiency and Deployment Complexity

In realistic distribution feeders, |C| (the number of customers or transformers) can be very large because utilities may need to forecast thousands or even hundreds of thousands of customer- and transformer-level loads. Therefore, the storage complexity and scaling with respect to |C| becomes a critical deployment bottleneck. Table I presents the parameterefficiency comparison among three customer-specific adaptation strategies. Here, $d _ { z }$ is the dimension of the latent representation, $d _ { x }$ is the output dimension of the forecasting horizon, r is the LoRA rank, K is the number of load clusters, S is the dimension of the normalized weekly profile, and $d _ { u }$ is the PCA-reduced profile dimension.

For full output-head adaptation, each customer requires its own mean and variance output matrices, resulting in $O ( 2 | \mathcal { C } | d _ { x } d _ { z } )$ additional parameters. This cost grows linearly with the number of loads and becomes expensive in large-scale feeders. Independent LoRA per customer reduces the cost by replacing each full matrix update with low-rank factors. For each customer, the mean and variance heads require two LoRA updates, leading to $O ( 2 | \mathcal { C } | r ( d _ { x } + d _ { z } ) )$ ) additional parameters. Although this is much smaller than full outputhead adaptation when $r \ll \operatorname* { m i n } ( d _ { x } , d _ { z } )$ , the storage cost still scales linearly with |C|, since each customer or transformer needs an independent adapter.

In contrast, the proposed $\mathbf { M } ^ { 2 } \mathrm { o E } ^ { 2 }$ -mLoRA stores only K cluster-level LoRA updates shared by all customers, giving $\mathcal { O } ( 2 K r ( d _ { x } + d _ { z } ) )$ parameters for the mean and variance heads. In addition, the lightweight (1 layer) neural network $g _ { \Omega }$ maps $[ { \pmb u } _ { c } ; { \pmb m } _ { k } ] \in \mathbb { R } ^ { 2 d _ { u } }$ to the scalar correction score $r _ { c , k }$ , requiring $O ( 2 d _ { u } + 1 )$ parameters. The clustering-related components introduce $\mathcal { O } ( S d _ { u } + S )$ parameters for the PCA projection and mean profile, and $\begin{array} { r } { \mathcal { O } ( \tilde { K d } _ { u } + \frac { K d _ { u } ( d _ { u } + 1 ) } { 2 } + K ) } \end{array}$ parameters for the full-covariance GMM. Since these components are shared across all customers and $K \ll | \mathcal { C } |$ , the proposed method avoids storing customer-specific adapters and keeps the additional parameter cost independent of the number of loads.

## V. EXPERIMENTS

## A. Setup

SMART-DS (Synthetic Models for Advanced, Realistic Testing: Distribution Systems and Scenarios) provides public distribution network models and load time series for several U.S. regions, including San Francisco, Greensboro, and Austin [32]. The models are designed to reproduce realistic statistical and operational characteristics of utility distribution systems, with active and reactive power profiles available at 15-minute intervals. We use 590 residential, commercial, and industrial load profiles from the 2018 dataset and aggregate active power to hourly resolution for forecasting. Selected profiles contain complete observations throughout 2018 and have a 95th-percentile hourly demand of at least 4 kW from January to October, spanning demand levels from 4 kW to above 128 kW.

Training uses January-October data, validation uses November, and testing uses December. Each day at midnight, the model forecasts the next 24 hours. Using complete weekly samples yields 30 forecasting times per profile from December 1–30, 2018, corresponding to 17,700 daily forecasts and 424,800 hourly predictions. Each sample spans two weeks. The model first encodes 168 hours of historical load, then updates this representation with newly available observations before each daily forecast and predicts the next 24 hours’ load.

We evaluate forecasting performance using seven metrics [10], [33]: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Weighted Absolute Percentage Error (WAPE), Continuous Ranked Probability Score (CRPS), Quantile Score (QS), and Winkler Score (WS). MSE measures profile-normalized squared error; RMSE and MAE quantify prediction errors in kW; WAPE measures absolute error relative to aggregate demand; CRPS evaluates the overall predictive distribution; QS assesses the accuracy of predicted quantiles; and WS evaluates prediction intervals by jointly considering their coverage and width. Lower values indicate better performance for all metrics.

We compare M<sup>2</sup>oE<sup>2</sup>-mLoRA against statistical, recurrent/convolutional, Transformer-based, and pretrained timeseries foundation models. The benchmarks include ARIMA + Fourier, GRU [34], LSTM [35], CNN-GRU, DeepAR [36], iTransformer [37], PatchTST [38], Chronos-2 [39], TimesFM-2.5 [40], [41], and TimesFM-2.5 + LoRA [30], covering deterministic and probabilistic forecasting as well as zero-shot and adapted foundation models. We further compare against an $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ ablation [10] without the proposed mLoRA adaptation, which serves as the shared base model for evaluating the contribution of customer-specific adapters.

TABLE II: Forecasting performance and computational cost.
<table><tr><td>Model</td><td>MSE  $( 1 0 ^ { - 3 } )$ </td><td>RMSE</td><td>MAE</td><td>WAPE</td><td>CRPS</td><td>QS</td><td> ${ \mathrm { W S } } _ { 9 0 }$ </td><td> ${ \mathrm { W S } } _ { 5 0 }$ </td><td>Time</td><td>Storage</td></tr><tr><td>ARIMA + Fourier</td><td>5.47</td><td>8.79</td><td>3.26</td><td>8.19</td><td>2.45</td><td>1.10</td><td>20.96</td><td>11.16</td><td>6.86*</td><td>0.18</td></tr><tr><td>CNN-GRU</td><td>3.92</td><td>6.58</td><td>2.37</td><td>5.95</td><td>1.75</td><td>0.79</td><td>15.52</td><td>7.79</td><td>0.04</td><td>0.33</td></tr><tr><td>TimesFM-2.5 zero shot</td><td>3.74</td><td>7.22</td><td>2.35</td><td>5.91</td><td>1.80</td><td>0.83</td><td>17.72</td><td>8.03</td><td>4.92</td><td>882.30</td></tr><tr><td>iTransformer</td><td>3.69</td><td>6.22</td><td>2.14</td><td>5.37</td><td>1.67</td><td>0.77</td><td>16.99</td><td>7.42</td><td>0.06</td><td>18.77</td></tr><tr><td>LSTM</td><td>3.51</td><td>4.90</td><td>1.77</td><td>4.45</td><td>1.31</td><td>0.59</td><td>11.88</td><td>5.84</td><td>0.10</td><td>0.79</td></tr><tr><td>GRU</td><td>3.49</td><td>5.44</td><td>1.98</td><td>4.97</td><td>1.48</td><td>0.68</td><td>14.26</td><td>6.57</td><td>0.06</td><td>0.60</td></tr><tr><td>PatchTST</td><td>3.35</td><td>5.43</td><td>1.87</td><td>4.68</td><td>1.45</td><td>0.68</td><td>15.36</td><td>6.44</td><td>0.04</td><td>2.32</td></tr><tr><td>Chronos-2 zero shot</td><td>3.21</td><td>6.16</td><td>2.02</td><td>5.07</td><td>1.48</td><td>0.67</td><td>12.59</td><td>6.61</td><td>10.15</td><td>455.77</td></tr><tr><td>DeepAR</td><td>3.18</td><td>5.08</td><td>1.71</td><td>4.28</td><td>1.45</td><td>0.72</td><td>12.78</td><td>6.98</td><td>11.62</td><td>0.77</td></tr><tr><td> $\mathrm { T i m e s F M - } 2 . 5 + \mathrm { L o R A }$ </td><td>3.15</td><td>5.32</td><td>1.73</td><td>4.35</td><td>1.37</td><td>0.64</td><td>14.99</td><td>6.07</td><td>4.93</td><td>882.30</td></tr><tr><td> $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ </td><td>2.98</td><td>4.87</td><td>1.79</td><td>4.49</td><td>1.32</td><td>0.59</td><td>11.29</td><td>5.90</td><td>0.08</td><td>0.40</td></tr><tr><td> $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 } { \mathrm { - m L o R A } }$ </td><td>2.72</td><td>4.50</td><td>1.60</td><td>4.16</td><td>1.18</td><td>0.48</td><td>11.01</td><td>5.54</td><td>0.11</td><td>0.58</td></tr></table>

MSE uses each customer’s training load range and is reported in units of $1 0 ^ { - 3 } .$ . RMSE, MAE, CRPS, QS, $\mathrm { W S _ { 9 0 } } ,$ and ${ \mathrm { W S } } _ { 5 0 }$ are in kW, while WAPE is reported in percent. Time is reported in ms/forecast and storage in MiB. Bold values indicate the best result in each column before rounding. Accuracy is evaluated on the full December test set. Time is the median processing time divided by 4,130 daily forecasts from December 3–9, with each forecast covering 24 hours. Neural models are evaluated on an NVIDIA Tesla T4. <sup>∗</sup>ARIMA is evaluated on an Intel Xeon CPU with two configured threads and is excluded from GPU speed comparisons. Storage includes the forecasting model and required static routing data, with one MiB equal to $\mathbf { \bar { 2 } } ^ { 2 0 }$ bytes.

![](images/ce678a0ee552ef1b407706b6bafe99f3df171475fb0c5a543f7741f66e1aafe0.jpg)

![](images/922bcdd403fbc9a78ed750bf363a2c8afbef4ef1287a7a5ef225344545287408.jpg)

![](images/fe4b3be31a5b7b3219bbe5afbc6c8e6dcf36ef5d32f98d84129c51189d0b216f.jpg)

![](images/bee0900f4478eb234bbde6d8306cccf67d5419097511063a663d461c481709a5.jpg)  
Fig. 2: Examples of load forecasts from $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ and $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ -mLoRA.

## B. Forecasting Performance and Computational Cost

Table II shows that $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 } .$ -mLoRA achieves the best performance across all evaluated deterministic and probabilistic forecasting metrics. Its MSE is 8.72% lower than that of $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ and 13.65% lower than that of TimesFM-2.5 with LoRA, demonstrating the benefit of cluster-specific mLoRA adaptation. The following figures further illustrate the forecasting behavior for representative load profiles.

Fig. 2 compares the actual loads with forecasts from $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ and $\mathbf { M } ^ { 2 } \mathbf { o } \mathbf { E } ^ { 2 } \mathbf { - m } \mathbf { L } \mathbf { o } \mathbf { R } \mathbf { A }$ for representative profiles with different demand levels and temporal patterns. Each panel shows seven consecutive day-ahead forecasts. For the first two profiles, customer-aware adaptation reduces overestimation between daily peaks, while for the third profile, it more accurately captures the lower-load period around hours 72-90. The last profile exhibits more irregular fluctuations, for which $\mathbf { M } ^ { 2 } \mathrm { o E } ^ { 2 } .$ mLoRA better tracks low-load intervals and more accurately recovers a sharp demand spike. Fig. 3 compares one-week forecasts for six test profiles across all benchmark models. Overall, $\mathbf { M } ^ { 2 } \mathbf { o } \mathbf { E } ^ { 2 } \mathbf { - m } \mathbf { L } \mathbf { o } \mathbf { R } \mathbf { A }$ tracks the observed load levels and daily variations more closely than the compared methods.

TABLE III: Sensitivity to the number of clusters.
<table><tr><td>K</td><td>Adapter parameters</td><td>November MSE  $( \times 1 0 ^ { - 3 } )$ </td></tr><tr><td>14</td><td>1,971,200</td><td>2.97</td></tr><tr><td>15</td><td>2,112,000</td><td>2.96</td></tr><tr><td>16</td><td>2,252,800</td><td>2.97</td></tr></table>

Inference cost is evaluated on the December 3-9 test period under the same forecasting setting for all methods. As shown in Table II, $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 } \mathbf { - m L } \mathbf { o } \mathrm { R } \mathbf { A }$ requires only 0.11 ms per forecast and 0.58 MiB of storage, remaining substantially more efficient than large pretrained foundation models while adding only modest overhead over the shared $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ base. This demonstrates that the proposed cluster-specific adaptation improves forecasting accuracy without sacrificing computational scalability.

## C. Parameter Sensitivity

We evaluate the sensitivity of $\mathrm { M ^ { 2 } o E ^ { 2 } { \cdot } m L o R A }$ to the number of load clusters using the same 590 profiles and chronological data split. The clusters are constructed from training data based on similarities in training-period load-profile features, and each cluster is assigned its own mLoRA adapter. As K increases, the model can represent more diverse customerspecific adaptation patterns, but the number of adapter parameters and overall model size also increase approximately linearly. Table III compares $K \ = \ 1 4 , 1 5 , 1 6 ;$ the lowest

![](images/403cdfc134a9401fa176bbaf274dc671852d8e896df02809d260c7809218b701.jpg)  
Hour (h)  
Fig. 3: Weekly load forecasts for six selected SMART-DS profiles. Rows show all nine benchmark configurations, and $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 } .$ mLoRA. Each column uses the same actual loads and vertical scale throughout.

November validation MSE is obtained with $K = 1 5$ , while the overall performance remains stable across the tested values, supporting the use of 15 clusters in the final model.

## VI. CONCLUSION

This paper studied scalable customer-aware probabilistic forecasting for heterogeneous customer- and transformer-level loads without training a separate full model for each load. The proposed $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ -mLoRA framework identifies the probabilistic output head as the key parameter subset for customerspecific adaptation and introduces a soft-membership-guided mixture-of-LoRA mechanism, allowing each load to combine compact cluster-level adaptations while preserving the shared forecasting backbone. Experiments on 590 load profiles from the SMART-DS dataset show that $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ -mLoRA improves all evaluated deterministic and probabilistic forecasting metrics over the shared $\mathbf { M } ^ { 2 } \mathbf { o } \mathrm { E } ^ { 2 }$ model and achieves the best overall forecasting performance among the evaluated methods, with low additional storage and inference cost. These results demonstrate that mixture-based parameter-efficient adaptation provides a scalable approach to capturing heterogeneous demand and uncertainty patterns in customer- and transformerlevel probabilistic forecasting.

[1] J. Han, L. Yan, and Z. Li, “A task-based day-ahead load forecasting model for stochastic economic dispatch,” IEEE Transactions on Power Systems, vol. 36, no. 6, pp. 5294–5304, 2021.

[2] S. H. Pramono, M. Rohmatillah, E. Maulana, R. N. Hasanah, and F. Hario, “Deep learning-based short-term load forecasting for supporting demand response program in hybrid energy system,” Energies, vol. 12, no. 17, p. 3359, 2019.

[3] A. Saxena, R. Shankar, E. El-Saadany, M. Kumar, O. Al Zaabi, K. Al Hosani, and U. R. Muduli, “Intelligent load forecasting and renewable energy integration for enhanced grid reliability,” IEEE Transactions on Industry Applications, 2024.

[4] H. Li, H. Tong, and Y. Weng, “Domain adaptation in physical systems via graph kernel,” in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, ser. KDD ’22. New York, NY, USA: Association for Computing Machinery, 2022, p. 868–876. [Online]. Available: https://doi.org/10.1145/3534678.3539380

[5] H. Li, Z. Ma, and Y. Weng, “A transfer learning framework for power system event identification,” IEEE Transactions on Power Systems, vol. 37, no. 6, pp. 4424–4435, 2022.

[6] H. Li, Z. Ma, Y. Weng, E. Blasch, and S. Santoso, “Structural tensor learning for event identification with limited labels,” IEEE Transactions on Power Systems, vol. 38, no. 6, pp. 5314–5328, 2023.

[7] X. Zhang, T. K. Chau, Y. H. Chow, T. Fernando, and H. H.-C. Iu, “A novel sequence to sequence data modelling based cnn-lstm algorithm for three years ahead monthly peak load forecasting,” IEEE Transactions on Power Systems, vol. 39, no. 1, pp. 1932–1947, 2023.

[8] C. Ye, Y. Ding, P. Wang, and Z. Lin, “A data-driven bottom-up approach for spatial and temporal electric load forecasting,” IEEE Transactions on Power Systems, vol. 34, no. 3, pp. 1966–1979, 2019.

[9] H. Li, M. Guo, Y. Weng, M. Ilic, and G. Ruan, “Exarnn: An environment-driven adaptive rnn for learning non-stationary power dynamics,” arXiv preprint arXiv:2505.17488, 2025.

[10] H. Li, M. Guo, M. Ilic, Y. Weng, and G. Ruan, “External data-enhanced meta-representation for adaptive probabilistic load forecasting,” IEEE Transactions on Smart Grid, pp. 1–1, 2026.

[11] A. Attarha, P. Scott, and S. Thiebaux, “Affinely adjustable robust´ admm for residential der coordination in distribution networks,” IEEE Transactions on Smart Grid, vol. 11, no. 2, pp. 1620–1629, 2019.

[12] X. Kong, C. Li, F. Zheng, and C. Wang, “Improved deep belief network for short-term load forecasting considering demand-side management,” IEEE transactions on power systems, vol. 35, no. 2, pp. 1531–1538, 2019.

[13] S. Schreck, I. P. de La Comble, S. Thiem, and S. Niessen, “A methodological framework to support load forecast error assessment in local energy markets,” IEEE Transactions on Smart Grid, vol. 11, no. 4, pp. 3212–3220, 2020.

[14] J. Ponocko and J. V. Milanovi´ c, “Forecasting demand flexibility of´ aggregated residential load using smart meter data,” IEEE Transactions on Power Systems, vol. 33, no. 5, pp. 5446–5455, 2018.

[15] B. Wang, M. Mazhari, and C. Chung, “A novel hybrid method for short-term probabilistic load forecasting in distribution networks,” IEEE Transactions on Smart Grid, vol. 13, no. 5, pp. 3650–3661, 2022.

[16] Y. Wang, N. Zhang, Y. Tan, T. Hong, D. S. Kirschen, and C. Kang, “Combining probabilistic load forecasts,” IEEE Transactions on Smart Grid, vol. 10, no. 4, pp. 3664–3674, 2018.

[17] H. Li, Y. Weng, V. Vittal, and E. Blasch, “Distribution grid topology and parameter estimation using deep-shallow neural network with physical consistency,” IEEE Transactions on Smart Grid, vol. 15, no. 1, pp. 655– 666, 2024.

[18] T. K. Wijaya, M. Vasirani, S. Humeau, and K. Aberer, “Cluster-based aggregate forecasting for residential electricity demand using smart meter data,” in 2015 IEEE international conference on Big data (Big data). IEEE, 2015, pp. 879–887.

[19] F. Fahiman, S. M. Erfani, S. Rajasegarar, M. Palaniswami, and C. Leckie, “Improving load forecasting based on deep learning and k-shape clustering,” in 2017 international joint conference on neural networks (IJCNN). IEEE, 2017, pp. 4134–4141.

[20] M. Ribeiro, K. Grolinger, H. F. ElYamany, W. A. Higashino, and M. A. Capretz, “Transfer learning with seasonal and trend adjustment for crossbuilding energy forecasting,” Energy and Buildings, vol. 165, pp. 352– 363, 2018.

[21] D. Wu and W. Lin, “Efficient residential electric load forecasting via transfer learning and graph neural networks,” IEEE Transactions on Smart Grid, vol. 14, no. 3, pp. 2423–2431, 2022.

[22] Y. Tian, L. Sehovac, and K. Grolinger, “Similarity-based chained transfer learning for energy forecasting with big data,” IEEE Access, vol. 7, pp. 139 895–139 908, 2019.

[23] Y. He, F. Luo, and G. Ranzi, “Transferrable model-agnostic metalearning for short-term household load forecasting with limited training data,” IEEE Transactions on Power Systems, vol. 37, no. 4, pp. 3177– 3180, 2022.

[24] G. Tsoumplekas, C. Athanasiadis, D. I. Doukas, A. Chrysopoulos, and P. Mitkas, “Few-shot load forecasting under data scarcity in smart grids: A meta-learning approach,” Energies, vol. 18, no. 3, p. 742, 2025.

[25] Y. Wang, N. Gao, and G. Hug, “Personalized federated learning for individual consumer load forecasting,” CSEE Journal of Power and Energy Systems, vol. 9, no. 1, pp. 326–330, 2022.

[26] C. Briggs, Z. Fan, and P. Andras, “Federated learning for short-term residential load forecasting,” IEEE open access journal of power and energy, vol. 9, pp. 573–583, 2022.

[27] Y. He, F. Luo, M. Sun, and G. Ranzi, “Privacy-preserving and hierarchically federated framework for short-term residential load forecasting,” IEEE Transactions on Smart Grid, vol. 14, no. 6, pp. 4409–4423, 2023.

[28] M. U. Danish and K. Grolinger, “Leveraging hypernetworks and learnable kernels for consumer energy forecasting across diverse consumer types,” IEEE Transactions on Power Delivery, vol. 40, no. 1, pp. 75–87, 2024.

[29] D. Gupta, A. Bhatti, S. Parmar, C. Dan, Y. Liu, B. Shen, and S. Lee, “Low-rank adaptation of time series foundational models for out-ofdomain modality forecasting,” in Proceedings of the 26th International Conference on Multimodal Interaction, 2024, pp. 382–386.

[30] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” Iclr, vol. 1, no. 2, p. 3, 2022.

[31] T. Hong, P. Pinson, S. Fan, H. Zareipour, A. Troccoli, and R. J. Hyndman, “Probabilistic energy forecasting: Global energy forecasting competition 2014 and beyond,” pp. 896–913, 2016.

[32] B. Palmintier et al., “SMART-DS synthetic electrical network data OpenDSS models for SFO, GSO, and AUS,” Open Energy Data Initiative, 2020. [Online]. Available: https://data.openei.org/submissions/ 2981

[33] H. Li, Z. Cheng, M. Guo, Y. Weng, Y. Sun, V. Tran, and J. Chainaranont, “M2oe2-gl: A family of probabilistic load forecasters that scales to massive customers,” in 2026 IEEE Power & Energy Society General Meeting (PESGM), 2026, pp. 1–5.

[34] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares,¨ H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder-decoder for statistical machine translation,” in Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, 2014, pp. 1724–1734. [Online]. Available: https://aclanthology.org/D14-1179/

[35] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.

[36] D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski, “DeepAR: Probabilistic forecasting with autoregressive recurrent networks,” International Journal of Forecasting, vol. 36, no. 3, pp. 1181–1191, 2020.

[37] Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, “itransformer: Inverted transformers are effective for time series forecasting,” arXiv preprint arXiv:2310.06625, 2023.

[38] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in International Conference on Learning Representations, 2023. [Online]. Available: https://arxiv.org/abs/2211.14730

[39] A. F. Ansari et al., “Chronos-2: From univariate to universal forecasting,” arXiv:2510.15821, 2025. [Online]. Available: https: //arxiv.org/abs/2510.15821

[40] A. Das, W. Kong, R. Sen, and Y. Zhou, “A decoder-only foundation model for time-series forecasting,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235, 2024, pp. 10 148–10 167. [Online]. Available: https://proceedings.mlr.press/v235/das24c.html

[41] Google Research, “TimesFM: Official implementation and TimesFM 2.5 release documentation,” GitHub repository, accessed September 27, 2026. [Online]. Available: https://github.com/google-research/timesfm