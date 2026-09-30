# IronLLM: Forging Compact Edge-Native Language Models for Real-Time Embodied Intelligence

Changdi Yang<sup>\*</sup>, Fengquan Jiao<sup>\*</sup>, Haochih Lin<sup>\*</sup>, Haoran Yang<sup>\*</sup>, Jing Xiao<sup>\*</sup>, Liangyu Huo<sup>\*</sup>, Suxin Lu<sup>\*</sup>, Tiance Chen<sup>\*</sup>, Wei Liu<sup>\*</sup>, Yinggan Xu<sup>\*</sup>, Yunxiang Lu<sup>\*†</sup>, Zai Zheng<sup>\*</sup>, Zhirui Xie<sup>\*</sup>, Zhongyang Che<sup>\*†</sup>, Ziyan Tang<sup>\*†</sup>, Zuoxiang Zhao<sup>\*</sup>, Jian Yao<sup>‡</sup>

Robotics Foundation Model Team, Xpeng Inc.

## Abstract

We present IronLLM-0.6B, a 654M-parameter language model designed for efficient on-device inference. IronLLM-0.6B combines a hybrid attention architecture with X-MTP, a lightweight shared-KV multi-token prediction design that eliminates per-depth KV-cache replay and employs a lightweight verification head for rollback-free drafting, achieving a 1.48x decoding speedup. The model is pretrained on approximately 6.2 trillion tokens using a quality-oriented data pipeline and is further post-trained with Multi-Domain On-Policy Distillation to integrate capabilities from domain-specialized teachers. To better meet the low-latency requirements of on-device scenarios, IronLLM-0.6B adopts an Instruct-Only design. Evaluations show that IronLLM-0.6B achieves competitive performance relative to larger models such as Qwen3.5- 0.8B and MiniCPM5-1B, while producing more concise responses on many tasks. We further present IronLLM-0.6B-Light, which replaces RMSNorm with Dynamic Tanh and simplifies several computationally expensive components to improve inference and quantization efficiency. Together, the IronLLM models provide an effective performance–efficiency trade-off for resource-constrained deployment.

Project: https://xpeng-robotics.github.io/iron-fm

![](images/521f0493f76ba405c0f6f47495ab944cdf2eaee1352b544d3ff715d990984441.jpg)

![](images/ffdc46905b7a45df833cb800cdfc0853cb60fdc982df2d6f598144121cceece7.jpg)

![](images/26c4c8672e6b533d94c89e9b7b98da13c5a1938a16f799fee0a324f4a97abdc7.jpg)

![](images/f555b7e0008bebc3532a67993364805afb107e49f86fb645f7c7b568ec686cd5.jpg)  
Figure 1: Capability and inference efficiency of compact language models. Left: scores across eight evaluation categories. Upper right: decoding throughput relative to IronLLM-0.6B at 32K context. Lower right: benchmark-level score differences versus relative generation time, compared with IronLLM-0.6B MTP.

## Contents

1 Introduction 4   
2 Architecture 5   
2.1 IronLLM-0.6B Basic Architecture 5   
2.2 IronLLM-0.6B-Light: Edge-Oriented Model Structure 6   
3 Training Data 8   
3.1 Data Sources 8   
3.2 Data Processing Pipeline 9   
3.3 Data Contamination Analysis . 10   
4 Pre-Training 10   
4.1 Pre-Training Data Construction . 10   
4.2 Multi-Stage Pre-Training Strategy 10   
4.3 Pre-Training Setup 11   
4.4 Evaluations 12   
4.4.1 Evaluation Setup 12   
4.4.2 Evaluation Results 13   
4.5 Long-Context Mid-Training 13   
5 Post-Training 15   
5.1 Post-Training Pipeline 15   
5.2 General Supervised Fine-Tuning 15   
5.3 Domain-Specialist Training . 16   
5.4 Multi-Domain On-Policy Distillation . 17   
5.5 Evaluations 18   
5.5.1 Evaluation Setup 18   
5.5.2 Evaluation Results 19   
5.5.3 Inference Efficiency 19   
5.5.4 MOPD Capability Integration 20   
6 Lightweight Multi-Token Prediction 22   
6.1 X-MTP Design 22   
6.1.1 Standard MTP Architecture 22   
6.1.2 Cross-Step Parameter and KV Sharing . 23   
6.1.3 Inference Speed . 24   
6.2 Lightweight Verification Head 25   
6.2.1 Verification Head Design 25   
6.2.2 Training Strategy 26   
6.2.3 Effectiveness of Verification Head . 27   
7 Conclusion, Limitation, and Future Work 27   
A Data Contamination Details 33   
B Training and Evaluation Details 33   
B.1 Pre-training Loss Curve . 33   
B.2 Base Model Evaluation Protocol 34   
B.3 MOPD Training Curves . 35   
B.4 Additional Results of the Verification Head 35

## 1 Introduction

The rapid advancement of large language models (LLMs) has led to unprecedented capabilities in natural language understanding, reasoning, and generation, with today’s frontier models scaling to hundreds of billions or even trillions of parameters [1–4]. However, the practical deployment of AI assistants in real-world settings demands a fundamentally different set of priorities. This is particularly evident in embodied intelligence, where robots, in-vehicle systems, and smart cockpits must understand instructions, reason about their surroundings, and respond in real time. In these on-device contexts, factors such as inference latency, memory footprint, energy consumption, and operational reliability often outweigh the raw capacity offered by massive scale. Consequently, embodied intelligence creates a critical need for compact, high-performance models that deliver robust capabilities entirely on the edge, without reliance on network connectivity or remote computation.

Achieving strong performance within a severely constrained parameter budget poses unique challenges across the model development lifecycle. On the data front, small models exhibit heightened sensitivity to corpus quality and distribution, requiring substantially more efficient data curation than their larger counterparts, for which massive data scale can often compensate for noise [5]. Architecturally, components that yield marginal gains in large models may introduce disproportionate overhead in compact regimes, calling for careful co-design of model structure and inference efficiency under stringent onboard compute and power budgets. Furthermore, post-training paradigms must be tailored to preserve and integrate diverse capabilities without inducing the catastrophic forgetting or cross-domain interference that small models are particularly susceptible to. These considerations motivate treating on-device deployment for embodied applications not as an afterthought but as a first-class design constraint spanning data, architecture, and training methodology.

In this work, we introduce IronLLM-0.6B, a compact yet high-performance large language model with a hybrid attention architecture [4, 6, 7], purpose-built to push the boundaries of on-device inference. Our primary contributions are as follows:

• X-MTP: A Lightweight Multi-Token Prediction Architecture. We propose a shared-KV MTP design that eliminates the per-depth KV-cache replay overhead inherent in conventional MTP [1, 8], substantially reducing the auxiliary parameter count and memory traffic during speculative decoding. This design is specifically optimized for the stringent latency and memory constraints of edge devices. A lightweight verification head further enables adaptive, rollback-free drafting on the hybrid linear-attention backbone.

• IronLLM-0.6B-Light: An Ultra-Efficient Architectural Variant. We further explore the absolute limits of on-device efficiency by introducing IronLLM-0.6B-Light, which incorporates DyT (Dynamic Tanh) [9] to pioneer the RMSNorm-free LLM architecture. Combined with a series of aggressive yet principled simplifications—including learnable upper-bounded ReLUx activations [10] and data-independent gating— this variant systematically eliminates computational bottlenecks to maximize inference throughput.

• A Large-Scale, Data-Efficient Processing Framework. We develop a unified data pipeline that achieves extreme data efficiency [11], consuming merely 6.2 trillion pre-training tokens while yielding performance competitive with models trained on substantially larger corpora. This framework integrates systematic quality enhancement, data composition optimization, and a continuous data-model co-optimization loop tailored to the heightened data sensitivity of small-parameter models.

• Multi-Domain On-Policy Distillation (MOPD) for Post-Training. We employ an MOPD paradigm [12– 14] during post-training that decouples capability production from capability integration. By distilling multiple domain-specialized teachers into a unified student through on-policy distillation with verifiable rewards, we achieve robust multi-domain performance while mitigating the cross-domain interference that commonly afflicts compact models.

Through these combined innovations, IronLLM-0.6B achieves performance on par with top-tier edge models such as Qwen3.5-0.8B [6] and MiniCPM5-1B [15], while its Instruct-Only design generates substantially shorter and more concise responses—a critical advantage for latency-sensitive applications. As summarized in Figure 1, the IronLLM models occupy a favorable position on the capability–efficiency frontier of compact language models: at a 32K context length, Qwen3-0.6B decodes at only 0.44× the speed of IronLLM-0.6B. Moreover, the proposed X-MTP architecture and IronLLM-0.6B-Light variant collectively establish a new paradigm for on-device efficiency, unlocking unprecedented possibilities for extreme-speed inference in future edge-centric scenarios, from autonomous robotics to next-generation intelligent cockpits.

## 2 Architecture

In this section, we present the architecture of IronLLM-0.6B, together with IronLLM-0.6B-Light, a structurally streamlined variant optimized for on-chip deployment that strikes a balance between competitive performance and substantially reduced inference cost.

## 2.1 IronLLM-0.6B Basic Architecture

As illustrated in Figure 2, IronLLM-0.6B follows a design philosophy similar to Qwen3.5 [6], adopting a decoder-only Transformer [16] with a hybrid attention mechanism and a parameter budget of approximately 650M, carefully calibrated for efficient deployment on edge devices while retaining strong bilingual (Chinese– English) capabilities. The model consists of 24 Transformer blocks in total, with embedding and output projection weights tied to reduce the parameter overhead introduced by the large vocabulary.

![](images/b91a8e20fc8c1a3addc15697c8216bd5dd7e8d4e6374f655ed3f0897e2a41e5c.jpg)  
Figure 2: Overview of the IronLLM-0.6B architecture. Gated DeltaNet (GDN) linear-attention layers and gated global attention (GA) layers are interleaved at a 3:1 ratio, with Zero-RMSNorm adopted throughout for training stability. An X-MTP module with a shared attention block and tied embedding/LM-head weights provides multi-token prediction.

Hybrid Attention. Recent mainstream LLMs increasingly replace a fraction of full self-attention layers with linear attention variants [4, 6, 7], among which Kimi Delta Attention (KDA) [17] and Gated DeltaNet (GDN) [18] are the two most representative designs. Given edge-deployment constraints, we adopt the simpler GDN formulation as our linear attention layer. Following our early experiments and common industry practice [6], linear and global attention layers are interleaved at a 3:1 ratio and evenly distributed across the network depth.

Each GDN layer first applies a short causal convolution to the projected queries, keys, and values to capture local token-mixing patterns, with queries and keys further $\ell _ { 2 } \cdot$ -normalized to stabilize training. The layer then maintains a fixed-size recurrent state $\mathbf { S } _ { t } \in \mathbb { R } ^ { d \times d } ,$ , updated at each step via a gated delta rule:

$$
\mathbf { S } _ { t } = \mathbf { S } _ { t - 1 } \left( \alpha _ { t } \left( \mathbf { I } - \beta _ { t } \pmb { k } _ { t } \pmb { k } _ { t } ^ { \top } \right) \right) + \beta _ { t } \pmb { v } _ { t } \pmb { k } _ { t } ^ { \top } , \qquad \pmb { o } _ { t } = \mathbf { S } _ { t } \pmb { q } _ { t } ,\tag{1}
$$

where $\alpha _ { t } , \beta _ { t } \in ( 0 , 1 )$ are input-dependent gates controlling state decay and write strength, respectively. An additional output gate rescales $\mathbf { } _ { o _ { t } }$ to compensate for the absence of softmax-style normalization.

The remaining global attention (GA) layers retain standard Grouped-Query Attention (GQA) [19] with QK-Norm [20] for precise long-range retrieval, and adopt Partial Rotary Positional Embeddings (RoPE) [21] to further strengthen long-context extrapolation. This hybrid design substantially reduces inference cost on long sequences while preserving retrieval fidelity.

Training Stability. Following Qwen3.5 [6], we employ output gating in both linear and global attention layers to suppress the attention and residual sinks caused by outlier activations [22], and also adopt Zero-RMSNorm for training stability. Formally, for an input vector x $\in \mathbb { R } ^ { d }$ , Zero-RMSNorm is defined as

$$
\mathrm { Z e r o - R M S N o r m } ( { \pmb x } ) = \frac { { \pmb x } } { \sqrt { \frac { 1 } { d } \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } + \epsilon } } \odot ( { \bf 1 } + { \boldsymbol \gamma } ) , \qquad { \boldsymbol \gamma } \gets { \bf 0 } ,\tag{2}
$$

where the learnable gain $\gamma$ is initialized to 0 rather than 1 as in conventional RMSNorm [23] to avoid the systematic scale drift.

Tokenizer. Given our focus on Chinese–English bilingual scenarios, we adopt the Qwen3 tokenizer [2] with a vocabulary size of 152K. Its high compression rate on multilingual text is critical for small edge-side models, where the embedding and output layers account for a substantial fraction of the total parameter count. Compared with the Qwen3.5 tokenizer, this choice reduces the parameter count by approximately 100M.

Table 1: Core architecture configuration of IronLLM-0.6B.
<table><tr><td>Component</td><td>IronLLM-0.6B</td></tr><tr><td>Total parameters</td><td>654M</td></tr><tr><td>Number of layers</td><td>24</td></tr><tr><td>Hidden size</td><td>1024</td></tr><tr><td>Intermediate size</td><td>3584</td></tr><tr><td>Attention type</td><td>Hybrid</td></tr><tr><td>Positional encoding</td><td>Partial RoPE</td></tr><tr><td>Number of GA layers</td><td>6</td></tr><tr><td>GA heads (Q/KV)</td><td>8/2</td></tr><tr><td>GA head dimensions (QK/V)</td><td>256</td></tr><tr><td>Linear heads (Q/KV)</td><td>16/16</td></tr><tr><td>Linear head dimensions (QK/V)</td><td>128/128</td></tr><tr><td>Vocabulary size</td><td>152K</td></tr></table>

Table 1 summarizes the detailed configuration of IronLLM-0.6B. The model comprises 18 GDN layers and 6 GA layers, interleaved at a 3:1 ratio to balance linear-attention efficiency with full-attention retrieval fidelity. In total, IronLLM-0.6B contains 654M parameters, with a hidden size of 1024 and an FFN intermediate dimension of 3,584. The GA layers adopt GQA with 8 query heads and 2 key–value heads (head dimension 256), while the GDN layers employ 16 linear-attention heads (head dimension 128 for Q, K, and V). This heterogeneous head configuration, together with the stabilization designs described above, yields a compact yet expressive architecture tailored for efficient on-device inference.

## 2.2 IronLLM-0.6B-Light: Edge-Oriented Model Structure

To push the practical limits of on-device inference, we introduce IronLLM-0.6B-Light (Figure 3), a streamlined variant of IronLLM-0.6B tailored for extreme edge-side deployment, with a primary focus on maximizing inference throughput and latency efficiency while keeping performance degradation under control.

Our core design philosophy is to systematically identify and simplify modules that incur significant computational overhead or hinder quantization-friendly deployment. Specifically, we target two major bottlenecks: (1) computationally intensive RMSNorm operations [23], and (2) sigmoid-based activations, which exhibit poor compatibility with post-training quantization.

![](images/ccdef28f8f09b0a531fac973a3706bb56dac8350b51f139b6bed973c1643c1d1.jpg)  
Figure 3: Overview of the IronLLM-0.6B-Light architecture. Building on IronLLM-0.6B, the Light variant replaces all RMSNorm layers with Dynamic Tanh (DyT), removes the attention output gate and QK-Norm, omits the SiLU activation after the causal convolution in GDN, and adopts the upper-bounded ReLUx activation, reducing inference cost and improving quantization friendliness.

Dynamic Tanh (DyT) for Normalization. We replace all RMSNorm layers with Dynamic Tanh (DyT) [9] operation, defined as:

$$
\begin{array} { r } { \mathrm { D y T } ( \pmb { x } ) = \gamma \odot \mathrm { t a n h } \left( \alpha \pmb { x } \right) + \beta , } \end{array}\tag{3}
$$

where α, γ, and $\beta$ are learnable parameters. This substitution not only reduces computational cost but also eliminates the reliance on root-mean-square statistics, which are expensive to compute on resourceconstrained devices. Additionally, we remove the QK-Norm [20] previously used in the attention module, as its functionality is largely subsumed by DyT.

ReLUx: Learnable Upper-Bounded ReLU. To make the activation function friendly to on-chip deployment, we introduce ReLUx, a variant of the standard ReLU with a learnable upper bound [10]. While conventional ReLU is defined as

$$
{ \mathrm { R e L U } } ( x ) = \operatorname* { m a x } ( 0 , x ) ,\tag{4}
$$

ReLUx is formulated as

$$
{ \mathrm { R e L U x } } ( x ) = \operatorname* { m i n } ( \operatorname* { m a x } ( 0 , x ) , \theta ) ,\tag{5}
$$

where θ is a learnable parameter, applied per layer or per head. The bounded activation improves numerical stability under quantization while preserving the nonlinear expressiveness of the original ReLU.

Learnable embedding scaling. Unlike conventional architectures, we introduce a learnable global scaling factor $\gamma _ { \mathrm { e m b } }$ applied directly to the output of the token embedding layer:

$$
\boldsymbol e _ { t } ^ { \prime } = \gamma _ { \mathrm { e m b } } \cdot \boldsymbol e _ { t }\tag{6}
$$

where $e _ { t }$ is the raw token embedding, and $\gamma _ { \mathrm { e m b } }$ is initialized to $\sqrt { d _ { \mathrm { m o d e l } } }$ and thereafter optimized jointly with all other model parameters. Empirically, we find that this simple modification helps our Light model achieve better convergence and more stable training.

Simplification of Gating and Kernel Modules. Our ablations on architectural variants show that several advanced design choices—while effective in larger models—yield only marginal gains at the 0.6B scale, where the capacity bottleneck imposed by the limited parameter count outweighs the benefits of more sophisticated modules. We therefore adopt a series of simplifications that reduce computational overhead with negligible performance regression.

Specifically, we first remove the output gating from the full-attention layers entirely, as its contribution proves negligible in this regime. Second, we revisit the SiLU [24] activation following the causal Conv1D kernel in Gated DeltaNet. When seeking a quantization-friendlier substitute, we find that ReLU-style activations actually degrade performance; counterintuitively, directly omitting the activation yields better results than replacing SiLU with ReLU or ReLUx. We attribute this to ReLU-style gating suppressing a large fraction of feature channels to zero, which over-restricts the representational capacity of an already compact model. By contrast, removing the activation preserves full feature flow while reducing computation, leading to a more favorable efficiency–accuracy trade-off.

Moreover, in the Gated DeltaNet recurrence

$$
\mathbf { S } _ { t } = \alpha _ { t } \mathbf { S } _ { t - 1 } + \beta _ { t } \left( \pmb { v } _ { t } - \alpha _ { t } \mathbf { S } _ { t - 1 } \pmb { k } _ { t } \right) \pmb { k } _ { t } ^ { \top } .\tag{7}
$$

the forget and write coefficients are originally data-dependent. For each token $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and head h, the baseline computes

$$
\beta _ { t , h } = \sigma \big ( \mathbf { W } _ { b } \pmb { x } _ { t } \big ) _ { h } ,\tag{8}
$$

$$
g _ { t , h } = - \mathrm { e } ^ { A _ { h } } \mathrm { s o f t p l u s } \big ( ( \mathbf { W } _ { a } \mathbf { x } _ { t } ) _ { h } + b _ { h } ^ { \mathrm { d t } } \big ) , \qquad \alpha _ { t , h } = \mathrm { e } ^ { g _ { t , h } } ,\tag{9}
$$

where $\mathbf { W } _ { a } , \mathbf { W } _ { b }$ are input projections, $A _ { h }$ and $b _ { h } ^ { \mathrm { d t } }$ are learnable per-head parameters, and σ denotes the sigmoid.   
This requires dynamic projection and nonlinearities at every timestep.

In our Light variant, we replace both coefficients with data-independent, learnable head-wise scalars, eliminating ${ \mathbf W } _ { a }$ and $\mathbf { W } _ { b } \mathbf { : }$

$$
\beta _ { h } = \sigma \big ( \hat { \beta } _ { h } \big ) ,\tag{10}
$$

$$
g _ { h } = - \mathrm { ( s o f t p l u s } ( \gamma _ { h } ) + \varepsilon ) , \qquad \alpha _ { h } = \mathrm { e } ^ { g _ { h } } ,\tag{11}
$$

where $\hat { \beta } _ { h } , \gamma _ { h } \in \mathbb { R }$ are trainable per-head parameters $( \hat { \beta } _ { h }$ initialized at 0, hence $\beta _ { h } { = } 0 . 5 )$ , and $\varepsilon { = } 1 0 ^ { - 6 }$ ensures $\alpha _ { h } \in ( 0 , 1 )$ . For the decay logits $\gamma _ { h }$ , we adopt a Lightning-Attention-style [25, 26] slope initialization with layer-wise scaling:

$$
\boldsymbol { r } _ { l , h } = \boldsymbol { s } _ { h } \cdot \Big ( 1 - \frac { l } { L - 1 + \delta } + \delta \Big ) , \qquad \gamma _ { l , h } = \mathrm { s o f t p l u s } ^ { - 1 } ( \boldsymbol { r } _ { l , h } ) ,\tag{12}
$$

where $s _ { h }$ is the standard power-of-two slope schedule over value heads, l is the layer index, L is the total number of layers, and $\delta { = } \dot { 1 } 0 ^ { - 5 }$ is a small offset that keeps $r _ { l , h }$ strictly positive at all layers. Collectively, these changes remove token-wise dynamic gating, reduce projection cost, and yield more predictable inference behavior across varying input lengths.

## 3 Training Data

In this section, we present the data pipeline underlying our model training. We begin by introducing the diverse data sources used to construct the training corpus, followed by our unified data processing pipeline for large-scale data construction, curation, and optimization. Finally, we describe our benchmark contamination analysis, which helps ensure a consistent and high-quality training corpus.

## 3.1 Data Sources

We construct our bilingual training corpus through a unified data acquisition framework that continuously integrates diverse public and internally processed data sources. Rather than relying on a fixed collection of datasets, we continuously expand and refine the corpus throughout the model development lifecycle.

The training data spans a broad range of domains, including general web content, academic literature, code, mathematics, educational content, reasoning-oriented data, and other high-value sources. These data provide complementary signals for language understanding, knowledge acquisition, reasoning, mathematical problem solving, and code generation.

Beyond existing public resources, we continuously incorporate newly available data through our in-house data acquisition workflow, allowing the composition and coverage of the training corpus to evolve over time. The detailed data processing and optimization pipeline is introduced in the next section.

## 3.2 Data Processing Pipeline

To support scalable and high-quality model training across the entire training lifecycle, we build a unified data processing pipeline that continuously transforms raw data into a high-quality training corpus. As illustrated in Figure 4, the pipeline integrates data construction, curation, strategy, and continuous data iteration into a unified framework, connecting diverse data sources with different stages of model training. Standardized data construction transforms heterogeneous raw data into structured training samples, quality-aware data curation improves the consistency and reliability of the training corpus, flexible data strategies organize stage-specific data compositions for different training objectives, and continuous data iteration enables the training corpus to evolve alongside model development through a closed-loop feedback process.

![](images/f22a8354494750d0b02d1f20aef642d1f89e23e64a63ecd6e6423233acc47f17.jpg)  
Figure 4: Training Data Ecosystem.

Building upon this unified data engine, we continuously improve the training corpus through three key capabilities:

Systematic Data Quality Enhancement. We establish a unified data quality framework with standardized processing procedures across diverse data sources. Instead of relying on isolated heuristics, we develop in-house quality assessment models and ensemble multiple quality signals, including model-based scoring, rule-based signals, and domain-specific indicators, to achieve more comprehensive and reliable data evaluation. Based on the quality assessment results, we apply differentiated optimization strategies: low-quality samples are filtered, medium-quality samples are refined through targeted enhancement techniques, and high-quality samples are further utilized for synthetic data generation. This quality-aware data refinement process enables continuous improvement of the training corpus beyond simple data collection and filtering.

Data Composition Optimization. Beyond improving individual sample quality, we optimize the overall data distribution through systematic data mixture experiments. We establish a unified domain taxonomy and perform fine-grained domain classification to guide mixture design, enabling data-driven optimization of training recipes. With extensive mixture ratio exploration, we achieve competitive model performance while consuming only approximately 6.2T pre-training tokens, demonstrating substantially improved data efficiency. More importantly, the optimized mixture strategy is not limited to a single training stage. The same recipe search methodology can be applied across different stages and capability domains, enabling efficient construction of stage-specific data mixtures.

Data-Model Co-optimization Loop. We build a continuous data optimization loop that integrates model evaluation, capability analysis, and targeted data improvement. By analyzing model performance and identifying capability gaps, we further attribute these gaps to potential data deficiencies, such as insufficient coverage, imbalanced distributions, or limited high-quality supervision. Based on these insights, we develop targeted data strategies, including domain-specific data expansion, custom processing pipelines, and synthetic data generation. This iterative optimization paradigm has been applied across multiple capability domains, including Chinese language understanding, mathematical reasoning, and code generation, enabling the training corpus to continuously evolve with model development.

Together, these capabilities transform training data from a static collection into a measurable, controllable, and continuously evolving data asset, enabling long-term model improvement.

## 3.3 Data Contamination Analysis

Benchmark contamination may lead to overly optimistic evaluation results if benchmark samples appear in the training corpus. To ensure the reliability of our reported performance, we perform comprehensive contamination analysis across all major evaluation benchmarks.

Our contamination analysis combines multiple complementary detection methods, including n-gram overlap detection and embedding-based semantic retrieval, to identify potential overlaps. Based on the detected overlaps, we construct cleaned evaluation sets and compare model performance on the original and cleaned versions of the benchmarks to quantify the impact of contamination.

Across most evaluated benchmarks, performance differences between original and cleaned sets remain within 2.5 points, with several benchmarks even showing slight improvements after decontamination. This suggests that benchmark contamination has limited impact on our reported results. Detailed contamination detection procedures and experimental results are provided in Appendix A.

## 4 Pre-Training

This section details our pre-training methodology, specifically optimized for small-parameter language models targeting edge deployment. We first describe our data construction strategy, which prioritizes quality and mixture efficiency over raw scale, followed by the multi-stage training curriculum. We then present the pre-training setup, including hyperparameters and training strategy. Finally, we introduce our small-modeloriented evaluation framework and the long-context mid-training pipeline that extends the model’s native context window from 4,096 to 65,536 tokens.

## 4.1 Pre-Training Data Construction

Given the inherent capacity constraints of edge-deployed, small-parameter models, our pre-training data strategy diverges substantially from mainstream large-scale LLM training practices, which often prioritize data volume within fixed compute budget. Instead, we prioritize data quality and mixture efficiency over raw scale [11].

Building on the data pipeline described in Sec. 3, we curate a high-quality training corpus from a pre-processed pool of nearly 20 trillion tokens. Through systematic data mixture experiments, the final pre-training consumes only 6.2 trillion tokens, approximately 1/6 of the 36 trillion tokens used to train Qwen3 series [2].

The curated corpus spans a diverse set of domains, including public web text, books, academic papers, mathematics, and code, complemented by high-quality synthetic data in a variety of formats (e.g., QA-style pairs). Consistent with our bilingual focus, the corpus is predominantly composed of Chinese and English text.

## 4.2 Multi-Stage Pre-Training Strategy

We adopt a three-stage pre-training strategy with a Warmup-Stable-Decay (WSD) [5] learning rate schedule, as illustrated in Figure 5. The data composition is progressively shifted from general web corpora toward math, code, and reasoning-intensive sources as training proceeds, while the learning rate follows a corresponding WSD trajectory across stages.

![](images/8328d60adfa900b3975bdbb8d0aaa6e3181e9f9c5f0ddf0c8d786a37c4813847.jpg)

(b) Learning-rate schedule  
![](images/d2eac1ec345ecf4833a4f8b36b006bf8a2d561dbc1143239825d1ff24975a8d0.jpg)  
Figure 5: Illustration of the multi-stage pre-training pipeline. (a) Data composition across stages; (b) corresponding learning-rate schedule following the Warmup-Stable-Decay (WSD) pattern.

Stage 1. During the first 4.2 trillion tokens, the model is trained predominantly on general web corpora, with a small proportion of math and code data introduced early to establish foundational language understanding and seed elementary STEM reasoning capabilities. We include wiki-style encyclopedic knowledge throughout this period, but down-sample highly specialized content, such as domain-specific PDFs and arXiv papers, as our experiments indicate that small models struggle to absorb such knowledge at this point, and early exposure may even hurt general language performance.

Stage 2. Over the next 1.0 trillion tokens, we reduce the proportion of general web data and increase the share of math and code corpora to strengthen the model’s logical reasoning capabilities. We also introduce a modest amount of synthetic data, including QA-style pairs, to improve instruction-following and structured-response generation.

Stage 3. In the final 1.0 trillion tokens, we further reduce general web content, keeping only the highestquality subset after rigorous filtering, while continuing to raise the overall proportion of math, code, and high-quality synthetic data. This allows us to maximize reasoning performance within the remaining budget, using the declining learning rate to refine the model toward targeted capabilities.

Notably, even as the overall share of general web data decreases across stages, we maintain, or even increase, the proportion of Chinese web data within that category, and we carefully preserve the Chinese-data ratio within each individual data type throughout all three stages. In our experiments, high-quality Chinese data substantially improves the model’s Chinese-language performance without degrading English benchmark results, underscoring the importance of balanced bilingual curation even under an aggressive, quality-driven data reduction strategy.

## 4.3 Pre-Training Setup

Training Hyper-parameters. We employ the AdamW optimizer [27] with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ , and a weight decay of 0.1, which is applied exclusively to two-dimensional parameters, including embedding layers. Gradient clipping is applied with a maximum norm of 1.0. Model parameters are initialized with a uniform range of 0.02. The rotary positional embedding (RoPE) [21] base theta is set to 10,000 and is applied to the first 64 channels of the head dimension.

The training process consists of three stages with distinct learning rate schedules. In Stage 1, the learning rate linearly warms up from 0 to $4 . 0 \times 1 0 ^ { - 4 }$ over the first 80B tokens. Stage 2 maintains a constant learning rate of $4 . 0 \times 1 0 ^ { - 4 }$ . In Stage 3, the learning rate undergoes a linear decay from $4 . 0 \times 1 0 ^ { - 4 }$ down to 0. The global batch size is fixed at 1,024 throughout the entire pre-training process, and the training sequence length is set to $4 { , } 0 9 6$ across all stages. We use FlashAttention2 [28], Flash Linear Attention [29] and Liger Kernel [30] to accelerate training.

The training objective follows the standard next-token prediction cross-entropy loss. To improve training efficiency, we initialize a single Multi-Token Prediction (MTP) [1] head for one-step-ahead prediction starting from Stage 3, with the MTP loss weight set to 0.1.

Per-batch data proportioning. We examined a training paradigm in which each global batch of 1,024 samples is constructed to exactly mirror the overall source distribution, with the goal of ensuring consistent data composition at every optimization step and thereby yielding more stable gradients. Empirically, however, this strict enforcement did not lead to improved downstream task performance. We hypothesize that the stochasticity introduced by standard global shuffling serves as an implicit regularizer: permitting per-batch ratios to fluctuate naturally facilitates broader exploration of the loss landscape and reduces the likelihood of convergence to sharp minima. Conversely, rigidly constraining per-batch composition restricts the optimization trajectory and degrades generalization. Accordingly, we enforce data proportions only at the global level over the full training run, relaxing per-batch constraints. In addition, to preserve stable gradient norms, we augment standard gradient clipping with a mechanism that skips optimizer steps when anomalous data triggers severe gradient spikes.

Cross-document attention. We also evaluated cross-document attention [31], which prevents attention interference between distinct documents concatenated within the same sequence. Ablations across training stages showed that applying this masking during the pretraining annealing phase improved performance on some benchmarks while degrading it on others, and these discrepancies were largely neutralized after subsequent supervised fine-tuning (SFT). We therefore omit this masking during the main pretraining stage, treating unmasked document concatenation as an implicit form of data augmentation. However, we explicitly enable it during long-context training, where spurious inter-document attention in significantly longer sequences consistently degrades nearly all evaluated capabilities.

Model Merging. We apply model merging [32, 33] during pretraining to mitigate performance instability on challenging downstream tasks. By merging intermediate checkpoints, we create a more robust initialization for subsequent training stages. Additionally, this approach enables the rapid evaluation of different data mixture ratios.

As shown in Table 2, the merged checkpoint $\left( X _ { \mathrm { m e r g e } } \right)$ outperforms or matches both the intermediate (X − 10k) and final (X) checkpoints across all benchmarks, achieving substantial gains on GSM8K (+7.51) and MATH (+5.18). This indicates that merging intermediate checkpoints effectively stabilizes pretraining and provides a stronger foundation for downstream tasks.

Table 2: Performance comparison across benchmarks. Best results are highlighted in bold.
<table><tr><td>Checkpoint</td><td>MMLU</td><td>CMMLU</td><td>CEval</td><td>BBH</td><td>GSM8K</td><td>MATH</td><td>MBPP</td></tr><tr><td>Step X – 10k</td><td>52.32</td><td>48.39</td><td>48.11</td><td>35.65</td><td>51.55</td><td>19.06</td><td>36.19</td></tr><tr><td>Step X</td><td>53.18</td><td>48.90</td><td>47.67</td><td>36.52</td><td>50.34</td><td>17.64</td><td>40.47</td></tr><tr><td>Xmerge</td><td>54.17</td><td>49.93</td><td>49.49</td><td>37.70</td><td>57.85</td><td>22.82</td><td>40.47</td></tr><tr><td>Gain over Step X</td><td>+0.99</td><td>+1.03</td><td>+1.82</td><td>+1.18</td><td>+7.51</td><td>+5.18</td><td>+0.00</td></tr></table>

## 4.4 Evaluations

## 4.4.1 Evaluation Setup

We evaluate IronLLM-0.6B-Base and IronLLM-0.6B-Light-Base against representative pretrained base models in up to one-billion parameter regime, including Qwen3-0.6B-Base [2], Qwen3.5-0.8B-Base [6], and

MiniCPM5-1B-Base [15]. The evaluation aims to provide a comprehensive comparison of small language models across general knowledge, reasoning, Chinese language understanding, mathematical reasoning, and code generation. All models are evaluated using a unified evaluation pipeline [34] with consistent prompting, decoding, and post-processing settings whenever applicable.

We organize the evaluation benchmarks into the following capability categories:

• General Tasks: MMLU (5-shot) [35], MMLU-Pro (5-shot, CoT) [36], MMLU-redux (5-shot) [37], ARC-Easy & ARC-Challenge (0-shot) [38], HellaSwag (0-shot) [39], CSQA (8-shot) [40], OBQA (0-shot) [41], PIQA (0-shot) [42], WinoGrande (0-shot) [43], BBH (3-shot, CoT) [44], TriviaQA (5-shot) [45].

• Mathematical Reasoning: GSM8K (4-shot, CoT) [46], MATH (4-shot, CoT) [47].

• Code Generation: HumanEval (0-shot) [48], MBPP (3-shot) [49].

• Chinese Understanding: CMMLU (5-shot) [50], C-Eval (5-shot) [51], C<sup>3</sup> (0-shot) [52].

Small-Scale Model Evaluation. When evaluating the Base model, we observe that most existing benchmarks are primarily designed for large-scale models, making them overly challenging for small-parameter models (e.g., 0.6B scale). This leads to unstable performance measurements [53] and complicates fair comparisons across different architectural variants, especially for models trained from scratch. To address this, we tailor our evaluation strategy specifically for the small-model setting. During early-stage pretraining, we adopt both standard MCF (Multiple-Choice Fill) and custom-designed CF (Cloze Fill) metrics [54] to continuously track model capabilities. Furthermore, we enhance the robustness of evaluation prompts for several benchmarks through refined template designs, which helps stabilize performance trajectories throughout training. Detailed descriptions are provided in Appendix B.2.

## 4.4.2 Evaluation Results

As shown in Table 3, IronLLM-0.6B-Base achieves the best performance on 8 of the 19 benchmarks, including the full MMLU suite (MMLU, MMLU-Pro, and MMLU-redux), TriviaQA, HellaSwag, WinoGrande, GSM8K, and C<sup>3</sup>, while remaining on par with the best baseline on MATH. It outperforms MiniCPM5-1B-Base on 17 of the 19 benchmarks, although the latter has over 53% more parameters (e.g., a 21.0-point margin on GSM8K). It also leads Qwen3.5-0.8B-Base, which is approximately 25% larger, on 13 benchmarks and performs on par with the identically sized Qwen3-0.6B-Base, highlighting the parameter efficiency of our model. The remaining benchmarks are led by Qwen3.5-0.8B-Base (ARC, BBH, CMMLU, and C-Eval) or Qwen3-0.6B-Base (CommonsenseQA, HumanEval, and MBPP).

We also report results for IronLLM-0.6B-Light-Base, which trades a modest amount of raw capability for improved inference and quantization efficiency. Overall, it retains most of the capabilities of IronLLM-0.6B-Base, with an average deficit of about 5.5 points across the 19 benchmarks, and remains competitive against larger baselines: it outperforms MiniCPM5-1B-Base on 15 of 19 benchmarks and leads Qwen3.5-0.8B-Base on 8 of them, offering an attractive performance–efficiency trade-off for resource-constrained deployment.

We note that MiniCPM5-1B-Base only publicly releases the checkpoint prior to its mid-training stage, and we therefore evaluate this officially released version. Moreover, we find that continuing to increase the proportion of instruction-style data in the pre-training corpus can improve the performance of base models. However, to avoid front-loading benchmark gains that rightfully belong to post-training and, more importantly, to preserve the model’s plasticity and headroom for the post-training stage, we deliberately restrain our use of such data. As such, the results in Table 3 faithfully reflect the capabilities that IronLLM-0.6B-Base acquires purely from pre-training, providing a cleaner basis for attributing the gains brought by subsequent post-training.

## 4.5 Long-Context Mid-Training

Following the main pretraining, we extend the model’s native 4K context window to 64K tokens through a dedicated mid-training phase. We first describe the stage-wise extension schedule with the corresponding RoPE adaptation (Table 4), then introduce the length-bucketed data mixing strategy and the curation of targeted synthetic data.

Two-stage extension with progressive RoPE scaling. We extend the context window from 4K to 32K and then to 64K tokens, trained on approximately 42B and 21B tokens, respectively. Since the model employs RoPE [21], whose relatively small base (θ = 10,000) limits the distinguishability of distant positions, we scale the base to $1 0 ^ { 6 }$ at the first stage and keep it fixed thereafter, allowing the model to adapt smoothly while preserving pretrained capabilities. Simply pushing the context length further yields no significant gains, and ablations show that allocating more tokens to the first stage brings little additional benefit, suggesting that this schedule balances training efficiency and long-context capability.

Table 3: Comparison of IronLLM-0.6B-Base with other representative base models. Bold and underlined values indicate the best and second-best non-thinking results, respectively.
<table><tr><td>Benchmark</td><td># Shots</td><td>Mode</td><td>IronLLM-0.6B Base</td><td>IronLLM-0.6B -Light Base</td><td>MiniCPM5-1B Base1</td><td>Qwen3.5-0.8B Base</td><td>Qwen3-0.6B Base</td></tr><tr><td colspan="8">General Tasks</td></tr><tr><td>MMLU</td><td>5-shot</td><td>PPL</td><td>55.67</td><td>50.62</td><td>45.21</td><td>49.94</td><td>54.46</td></tr><tr><td>MMLU-Pro (CoT)</td><td>5-shot</td><td>Gen</td><td>26.45</td><td>23.24</td><td>21.07</td><td>25.71</td><td>23.93</td></tr><tr><td>MMLU-redux</td><td>5-shot</td><td>PPL</td><td>57.11</td><td>51.57</td><td>46.61</td><td>51.02</td><td>55.69</td></tr><tr><td>ARC-Challenge</td><td>0-shot</td><td>PPL</td><td>64.41</td><td>61.02</td><td>48.14</td><td>72.54</td><td>66.10</td></tr><tr><td>ARC-Easy</td><td>0-shot</td><td>PPL</td><td>79.72</td><td>72.66</td><td>62.26</td><td>84.83</td><td>83.07</td></tr><tr><td>CommonsenseQA</td><td>8-shot</td><td>PPL</td><td>61.92</td><td>48.40</td><td>39.64</td><td>59.46</td><td>63.39</td></tr><tr><td>OpenBookQA</td><td>0-shot</td><td>PPL</td><td>75.00</td><td>66.02</td><td>63.60</td><td>75.20</td><td>75.20</td></tr><tr><td>PIQA</td><td>0-shot</td><td>PPL</td><td>71.16</td><td>71.22</td><td>72.52</td><td>69.15</td><td>69.91</td></tr><tr><td>HellaSwag</td><td>0-shot</td><td>PPL</td><td>52.65</td><td>51.90</td><td>50.57</td><td>49.25</td><td>47.13</td></tr><tr><td>WinoGrande</td><td>0-shot</td><td>PPL</td><td>57.30</td><td>56.67</td><td>55.09</td><td>55.96</td><td>55.25</td></tr><tr><td>BBH (CoT)</td><td>3-shot</td><td>Gen</td><td>38.84</td><td>35.43</td><td>38.29</td><td>46.87</td><td>41.52</td></tr><tr><td>TriviaQA</td><td>5-shot</td><td>Gen</td><td>31.22</td><td>27.54</td><td>28.55</td><td>24.53</td><td>26.50</td></tr><tr><td colspan="8">Mathematical Reasoning</td></tr><tr><td>GSM8K (CoT)</td><td>4-shot</td><td>Gen</td><td>61.71</td><td>53.45</td><td>40.71</td><td>45.72</td><td>59.82</td></tr><tr><td>MATH (CoT)</td><td>4-shot</td><td>Gen</td><td>31.32</td><td>22.62</td><td>19.36</td><td>22.64</td><td>31.82</td></tr><tr><td colspan="8">Code Generation</td></tr><tr><td>HumanEval</td><td>0-shot</td><td>Gen</td><td>25.61</td><td>21.34</td><td>13.41</td><td>20.73</td><td>28.05</td></tr><tr><td>MBPP</td><td>3-shot</td><td>Gen</td><td>30.00</td><td>23.60</td><td>33.20</td><td>26.20</td><td>37.00</td></tr><tr><td colspan="8">Chinese Understanding</td></tr><tr><td>CMMLU</td><td>5-shot</td><td>PPL</td><td>51.62</td><td>44.55</td><td>35.71</td><td>53.90</td><td>52.00</td></tr><tr><td>C-Eval</td><td>5-shot</td><td>PPL</td><td>50.38</td><td>43.80</td><td>36.31</td><td>55.88</td><td>54.75</td></tr><tr><td>C3</td><td>0-shot</td><td>PPL</td><td>57.86</td><td>50.14</td><td>49.64</td><td>55.51</td><td>52.93</td></tr></table>

Length-bucketed data mixing. To strengthen long-context learning without eroding short-context performance, we partition the training corpus into length-based buckets and tune the cross-bucket sampling ratios through extensive ablations. The overall mixture otherwise closely follows that of the final pretraining stage, with only selected buckets upsampled for long-context learning. The resulting mixture consistently improves long-context benchmarks while remaining competitive on standard short-context evaluations.

Targeted synthetic long-context data. We find that capabilities such as retrieval, counting, and other specialized long-context reasoning are difficult to acquire from naturally occurring long documents alone. We therefore curate high-quality synthetic data specifically targeting these skills. Incorporating these datasets during either mid-training or SFT substantially improves the corresponding abilities without degrading general or short-context performance, making targeted synthetic data an efficient means of acquiring specialized long-context skills.

Table 4: Context extension schedule. The RoPE base is scaled at Stage 1 and kept fixed at Stage 2.
<table><tr><td>Stage</td><td>Context Window</td><td>RoPE Base</td><td>Tokens</td></tr><tr><td>Pretrain</td><td>4,096</td><td>10,000</td><td>6.2T</td></tr><tr><td>Stage 1</td><td>32,768</td><td>10,000 → 1,000,000</td><td>42B</td></tr><tr><td>Stage 2</td><td>65,536</td><td>1,000,000 (fixed)</td><td>21B</td></tr></table>

## 5 Post-Training

## 5.1 Post-Training Pipeline

Our post-training pipeline transforms the pre-trained base model into a capable instruction-following assistant through a three-stage process, as shown in Figure 6.

Stage 1: General Supervised Fine-Tuning. We first fine-tune the base model on a large-scale, multi-domain instruction dataset to establish broad conversational and instruction-following capabilities. This stage converts the base model into a general-purpose instruct model.

Stage 2: Domain-Specialist Training. Starting from the General SFT model, we independently train a set of domain-specialist models, each optimized for a specific capability domain. These specialists serve as high-quality teachers in the subsequent distillation stage.

Stage 3: Multi-Domain On-Policy Distillation. We distill the complementary strengths of all domain specialists back into the General SFT model via on-policy distillation, yielding a single unified model that retains broad general ability while approaching specialist-level performance in each domain.

![](images/340adb6d821909aa2a4d0f7a0d28f278c19877ca829f71463b28007ea209f161.jpg)  
Figure 6: Overview of the IronLLM post-training pipeline. IronLLM-0.6B follows three stages: General SFT, independent domain-specialist training, and Multi-Domain On-Policy Distillation (MOPD). During MOPD, frozen specialists provide token-level guidance through teacher prefill, while domain verifiers supply sequence-level verifiable rewards (VRs) on student-generated trajectories. IronLLM-0.6B-Light skips specialist training and reuses the specialists trained from IronLLM-0.6B.

Training Strategy Variants. We apply distinct post-training strategies for different model variants to balance performance and efficiency. For IronLLM-0.6B, we execute the complete three-stage pipeline described above. For IronLLM-0.6B-Light, we streamline the process by skipping Stage 2 and directly applying MOPD after Stage 1, utilizing the domain specialists trained from IronLLM-0.6B as teachers. This design choice is motivated by three key advantages: (1) Higher Performance Ceiling: Teachers derived from the full-capacity IronLLM-0.6B offer superior upper-bound guidance compared to specialists trained from the lightweight variant; (2) Distributional Alignment: Since both variants share identical pre-training and General SFT stages, their output distributions remain highly similar, ensuring that cross-variant distillation remains effective without significant mismatch; and (3) Iteration Efficiency: Bypassing the redundant training of Light-specific specialists significantly accelerates the development cycle, enabling faster experimentation and deployment.

## 5.2 General Supervised Fine-Tuning

Data. We curate a large-scale instruction-tuning dataset comprising several million high-quality conversation samples, encompassing both single-turn and multi-turn dialogues across diverse categories—including general QA, instruction following, mathematical reasoning, code generation, and tool-use scenarios. Notably, general QA data constitutes over one-third of the dataset, to expose the model to a wide spectrum of response formats and stylistic conventions across varied task contexts. Similarly, mathematical reasoning accounts for more than one-third of the corpus, serving as a critical foundation for cultivating robust logical reasoning capabilities; this emphasis ensures that the model can autonomously generate diverse and coherent chain-of-thought (CoT) trajectories during subsequent reinforcement learning training. This comprehensive and strategically balanced coverage guarantees that the resulting model develops strong generalist competencies prior to any domain-specific specialization.

![](images/a97b33e670b58fcdb6ac419d69df18f55dce3df0f93235348f35255b879e218b.jpg)  
Figure 7: Composition of the general supervised fine-tuning (SFT) dataset.

Training Configuration. The model is trained for 3 epochs using the same optimizer configuration as in the pre-training stage to maintain training stability and consistency. The learning rate schedule begins with a warm-up phase over the first 100 steps, linearly increasing from zero to a peak of $\mathbf { \tilde { 8 . 0 } \times 1 0 ^ { - 5 } }$ , before smoothly decaying to zero via a cosine annealing schedule. Throughout instruction tuning, the RoPE base is kept identical to the final setting used during mid-training, preserving the positional encoding distribution to which the model has already adapted. One notable distinction from pre-training is the application of an attention mask tailored to the conversational format of SFT samples; specifically, the loss is computed exclusively on the assistant’s response tokens, excluding user prompts and system instructions from the optimization objective. This alignment ensures a seamless transition from pre-training to instruction tuning, minimizing disruption to learned representations and supporting stable convergence.

Model Design Choice: Instruct-Only. A notable design decision in post-training is that we train a pure instruct model without a thinking mode. While recent work [2] has demonstrated that extended reasoning traces can improve performance on complex tasks, we prioritize inference efficiency given our target deployment scenario on edge/on-device platforms. Eliminating the thinking mode significantly reduces output token count and thus latency, which is critical for real-time interactive applications on resource-constrained hardware.

## 5.3 Domain-Specialist Training

Building upon the General SFT Model, we develop a suite of domain-specialist models to advance performance boundaries on specific high-value tasks. Rather than applying a uniform training protocol, each specialist adopts a tailored strategy—employing either domain-specific Supervised Fine-Tuning (SFT), Reinforcement Learning (RL), or a sequential combination of both—depending on the unique characteristics and requirements of the target domain.

Domain-Specific SFT. For domains requiring precise knowledge injection or format adherence, we curate high-quality, domain-focused instruction datasets to fine-tune the General SFT model. This phase establishes robust in-domain foundational capabilities and ensures the model internalizes task-specific conventions.

Domain-Specific RL. For tasks demanding advanced reasoning or nuanced preference alignment, we apply RL either as a standalone training method or as a refinement stage following SFT. This component is critical for optimizing reasoning patterns, enhancing answer correctness, and aligning outputs with complex domain-specific preferences that are difficult to capture through supervised learning alone.

Concretely, for each training prompt x, the model generates a set of candidate responses $\left\{ y _ { 1 } , y _ { 2 } , \dotsc , y _ { K } \right\} \sim$ $p _ { \theta } ( \cdot \mid x )$ via policy sampling. Each candidate is scored by a domain-specific reward signal $r ( x , y _ { i } )$ , and the model parameters are updated to maximize the expected reward. We adopt Group Relative Policy Optimization (GRPO) [55] as our RL algorithm, which estimates advantages within each sampled group without requiring a separate value network:

$$
\hat { A } _ { i } = \frac { r ( x , y _ { i } ) - \mathrm { m e a n } \left( r ( x , y _ { j } ) _ { j = 1 } ^ { K } \right) } { \mathrm { s t d } \left( r ( x , y _ { j } ) _ { j = 1 } ^ { K } \right) + \epsilon } , \qquad i = 1 , \dots , K ,\tag{13}
$$

where $\epsilon > 0$ is a small constant for numerical stability. The policy is then updated with a clipped surrogate objective analogous to PPO [56], applied at the token level.

## The specialist family includes:

• Mathematics. Fine-tuned on mathematical reasoning and symbolic computation datasets, followed by RL optimization against verifiable correctness rewards. During RL, the final answer is extracted from a predefined output format and evaluated by a dedicated mathematical verifier that checks the numerical or algebraic equivalence of the predicted result against the ground truth. This pipeline enables robust step-by-step problem solving and substantially reduces arithmetic and logical errors.

• Code. Trained on code completion, synthesis, and debugging corpora spanning multiple programming languages, with subsequent RL leveraging execution-based rewards. Specifically, the executable function enclosed within the code block is extracted and paired with curated test cases; the reward signal is derived from the outcomes of assertion checks on the function’s inputs and expected outputs, thereby directly optimizing for functional correctness and code quality.

• Instruction Following. Further enhanced on complex, multi-constraint instruction datasets. Each training sample is annotated with a structured set of constraint conditions—such as output language, word count limits, and formatting requirements. During RL, a dedicated instruction verifier evaluates the model’s response against every individual constraint, yielding a fine-grained compliance score that serves as the reward signal. This design encourages precise and reliable adherence to composite instructions.

• MCQA. Trained via RL on multiple-choice question-answering datasets that span a broad range of disciplines and subject areas, aiming to strengthen the model’s general knowledge capacity. The reward is computed in a straightforward yet effective manner: the final selected option is directly extracted from the model’s output and compared against the ground-truth answer, providing a binary correctness signal without the need for auxiliary verification tools.

• Open-Ended QA. Trained with a reward model that scores response quality in alignment with human preferences. The training data encompass diverse tasks such as creative writing, editing, factual question answering, and role-playing, with a controlled proportion of safety-oriented samples. For tasks demanding high factual accuracy—such as explanation, advice, and planning—we deliberately exclude content from highly specialized domains such as law and finance to mitigate the risk of generating misleading or unverifiable claims.

These specialists operate as independent teacher models and serve as the knowledge source for the subsequent distillation stage.

## 5.4 Multi-Domain On-Policy Distillation

LLM post-training typically involves integrating capabilities across multiple domains. Existing approaches, including Mix-RL [2, 57], Cascade RL [58], Off-Policy Fine-Tuning [59, 60], and Parameter Merging [32, 61], provide practical means of integrating capabilities across multiple domains. However, these approaches often suffer from cross-domain interference and capability forgetting, leading to unstable or suboptimal capability integration.

We employ Multi-Domain On-Policy Distillation (MOPD) [12–14] to decouple capability production from capability integration. During MOPD training, we balance data sampling across domains such that approximately the same amount of training data is routed to each domain teacher, promoting balanced capability integration across domains. Specifically, domain-specialized teachers are trained independently to develop their respective capabilities, while a unified student integrates these capabilities through on-policy distillation and verifiable task rewards. Let D denote the set of domains and $\{ \pi _ { \phi _ { d } } \} _ { d \in \mathcal { D } }$ the corresponding frozen teacher policies. The number of teachers is configurable and can be extended with the domain set. Given a prompt $x ,$ its domain metadata $d ( x )$ routes the student-generated trajectory to the corresponding teacher $\pi _ { \phi _ { d ( x ) } }$ . The student policy $\pi _ { \theta }$ samples multiple responses

$$
y ^ { ( i ) } \sim \pi _ { \theta } ( \cdot \mid x ) , \qquad i = 1 , \ldots , N ,\tag{14}
$$

and the routed teacher evaluates each response on the student-generated prefixes, providing token-level log-probabilities. In this way, domain teachers can be trained independently, while capability integration is performed on the student policy’s own trajectory distribution. For a trajectory $y = ( y _ { 1 } , \dots , y _ { T } )$ , we align the student with the routed teacher by minimizing the token-level reverse KL divergence:

$$
\mathcal { L } _ { \mathrm { R K L } } ( \theta ) = \mathbb { E } _ { x , y \sim \pi _ { \theta } } \left[ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } D _ { \mathrm { K L } } \left( \pi _ { \theta } ( \cdot \vert x , y _ { < t } ) \Vert \pi _ { \phi _ { d ( x ) } } ( \cdot \vert x , y _ { < t } ) \right) \right] .\tag{15}
$$

In practice, we estimate the reverse KL divergence over student-generated tokens using the sampled-token $k _ { 1 }$ estimator. For each generated token y , it is defined as

$$
\ell _ { \mathrm { K D } , t } = \log \pi _ { \boldsymbol { \theta } } ( y _ { t } \mid \boldsymbol { x } , y _ { < t } ) - \log \pi _ { \phi _ { d ( \boldsymbol { x } ) } } ( y _ { t } \mid \boldsymbol { x } , y _ { < t } ) .\tag{16}
$$

For training stability, the estimator is clipped and converted into a detached token-level distillation advantage:

$$
\begin{array} { r } { \hat { A } _ { \mathrm { K D } , t } = - \mathrm { s g } \left[ \mathrm { c l i p } \left( \ell _ { \mathrm { K D } , t } , - c , c \right) \right] , } \end{array}\tag{17}
$$

where $\mathrm { s g } [ \cdot ]$ denotes stop-gradient. The corresponding distillation objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { K D } } ( \theta ) = \mathcal { L } _ { \mathrm { P G } } \left( \theta ; \hat { A } _ { \mathrm { K D } } \right) . } \end{array}\tag{18}
$$

This objective provides dense token-level supervision on trajectories generated by the current student, thereby reducing the distribution mismatch associated with off-policy teacher trajectories.

Beyond teacher supervision, verifiable rewards (VRs) derived from task outcomes provide an independent and complementary learning signal. While distillation transfers domain-specific behavior from teacher policies, VRs directly indicate whether complete responses successfully solve the underlying tasks. Incorporating this signal avoids relying exclusively on policy imitation and explicitly anchors optimization to task success. We evaluate each response using a domain-specific verifier, such as answer verification, execution-based testing, or instruction-constraint checking, and optimize the resulting VR objective alongside the distillation objective.

The overall training objective is

$$
\mathcal { L } _ { \mathrm { t o t a l } } ( \theta ) = \mathcal { L } _ { \mathrm { K D } } ( \theta ) + \lambda _ { \mathrm { V R } } \mathcal { L } _ { \mathrm { V R } } ( \theta ) ,\tag{19}
$$

where $\lambda _ { \mathrm { V R } }$ controls the relative contribution of verifiable rewards.

The distillation objective provides dense token-level guidance from the routed domain teacher, while the VR objective supplies sequence-level feedback derived from task outcomes. Together, they enable the student to integrate capabilities from multiple domain teachers while explicitly optimizing for task success.

## 5.5 Evaluations

## 5.5.1 Evaluation Setup

We evaluate IronLLM-0.6B and IronLLM-0.6B-Light against representative state-of-the-art open-source posttrained models in the sub-2B parameter regime, including Qwen3-0.6B [2], Qwen3.5-0.8B [6], MiniCPM5- 1B [15], and LFM2-700M [62]. Notably, Qwen3-0.6B, Qwen3.5-0.8B, and MiniCPM5-1B are hybrid thinking models; we report their performance under both thinking and non-thinking modes to provide a comprehensive comparison, while LFM2-700M only has an instruct version. The evaluation covers seven core capability categories: general knowledge, instruction following, subjective quality, mathematics, code generation, reasoning and function calling. Furthermore, to highlight our model’s advantages in inference efficiency, we present a detailed statistical analysis correlating output token length with benchmark scores.

For subjective evaluation, we adopt DeepSeek-V4-Flash-0731 as the judge model, which scores individual responses or conducts pairwise comparisons between model outputs. This LLM-as-a-Judge protocol provides complementary signals beyond objective metrics, capturing dimensions such as helpfulness, coherence, and user preference that are difficult to quantify with rule-based scoring.

The evaluation benchmarks are organized into the following capability categories:

• General Knowledge: MMLU-Pro [36], MMLU-Redux [37], C-Eval [51] and CMMLU [50].

• Instruction Following: IFEval [63], IFBench [64] and Multi-IF [65].

• Subjective Quality: AlpacaEval 2.0 [66] and ArenaHard [67].

• Mathematics: MATH-500 [68], GSM8K [46], AIME 2025, AIME 2026, and HMMT Feb. 2026 [69].

• Code Generation: HumanEval [48], MBPP [49], and LiveCodeBench v6 [70].

• Reasoning: Big-Bench Hard [44] and ZebraLogic [71].

• Function Calling: BFCL v3 [72].

All evaluations are conducted using EvalScope v1.10.0 [73] with 0-shot settings and default prompt templates. For high-difficulty benchmarks such as AIME, HMMT, and LiveCodeBench, we report avg@k or pass@k metrics to ensure statistical reliability, while all baseline models utilize their officially recommended sampling parameters.

## 5.5.2 Evaluation Results

As shown in Table 5, IronLLM-0.6B demonstrates particularly strong instruction following, mathematical reasoning, and subjective response quality, while also achieving competitive performance in long-context understanding and function calling. Compared with Qwen3.5-0.8B under the non-thinking setting, IronLLM 0.6B achieves consistent and often substantial improvements across nearly all benchmarks. Overall, IronLLM 0.6B also outperforms the non-thinking MiniCPM5-1B on the majority of general knowledge, instructionfollowing, mathematics, and function-calling benchmarks, although the latter has over 53% more parameters, highlighting the parameter efficiency of our model. In addition, the lighter variant IronLLM-0.6B-Light retains most of these gains and remains highly competitive among models of similar scale.

Table 5: Comparison of IronLLM-0.6B with representative post-trained models. Bold and underlined values indicate the best and second-best non-thinking results, respectively.
<table><tr><td>Benchmark (Metric)</td><td>|IronLLM-0.6B²</td><td>IronLLM-0.6B-Light²</td><td>Qwen3-0.6B3</td><td></td><td>LFM2-700M Qwen3.5-0.8B3,4</td><td>MiniCPM5-1B3</td></tr><tr><td colspan="7">Long Context</td></tr><tr><td>RULER⁵</td><td>87.7</td><td>82.1</td><td>40.4 / 60.0</td><td>56.2</td><td>87.5 / 82.1</td><td>67.5 /78.2</td></tr><tr><td>LongBench v2</td><td>27.0</td><td>27.2</td><td>28.4/ 28.2</td><td>16.1</td><td>27.8 / 25.8</td><td>24.3 / 26.4</td></tr><tr><td colspan="7">General Knowledge</td></tr><tr><td>MMLU-Pro</td><td>42.1</td><td>35.9</td><td>24.4 / 37.6</td><td>21.5</td><td>35.4 / 45.9</td><td>36.8 / 47.5</td></tr><tr><td>MMLU-Redux</td><td>60.4</td><td>56.2</td><td>46.5 / 56.0</td><td>47.1</td><td>54.2 / 63.3</td><td>59.7 / 69.5</td></tr><tr><td>C-Eval</td><td>47.3</td><td>44.7</td><td>42.3 / 51.8</td><td>37.8</td><td>29.0 / 31.4</td><td>54.6 / 66.6</td></tr><tr><td>CMMLU</td><td>49.4</td><td>45.2</td><td>45.3 / 49.6</td><td>38.3</td><td>33.0 / 47.9</td><td>66.4 / 72.2</td></tr><tr><td colspan="7">Instruction Following</td></tr><tr><td>IFEval</td><td>75.6</td><td>69.5</td><td>56.6 / 55.1</td><td>62.9</td><td>45.3 / 57.3</td><td>70.4 / 77.8</td></tr><tr><td>IFBench</td><td>18.0</td><td>16.3</td><td>15.7 /15.7</td><td>17.0</td><td>15.7 / 19.7</td><td>32.3 / 43.2</td></tr><tr><td>Multi-IF6</td><td>36.3</td><td>28.0</td><td>30.7 / 33.1</td><td>26.7</td><td>21.8 / 32.3</td><td>30.2 / 41.3</td></tr><tr><td colspan="7">Subjective Quality</td></tr><tr><td>AlpacaEval 2.0</td><td>10.2</td><td>7.1</td><td>3.4 / 2.6</td><td>7.0</td><td>1.6 / 2.9</td><td>4.0 / 3.4</td></tr><tr><td>ArenaHard v0.1</td><td>11.9</td><td>6.1</td><td>3.5/5.9</td><td>9.0</td><td>2.6 / 4.6</td><td>6.3 / 6.4</td></tr><tr><td colspan="7">Mathematics</td></tr><tr><td>MATH-500</td><td>67.4</td><td>57.4</td><td>52.2 / 74.8</td><td>27.6</td><td>43.8 / 17.0</td><td>56.4 / 89.0</td></tr><tr><td>GSM8K</td><td>78.6</td><td>73.6</td><td>61.8 /78.3</td><td>52.5</td><td>49.4 / 31.1</td><td>67.6 / 86.4</td></tr><tr><td>AIME 2025 (Avg@16)</td><td>10.2</td><td>6.3</td><td>9.0 / 16.3</td><td>3.3</td><td>1.7 / 0.0</td><td>0.0/31.5</td></tr><tr><td>AIME 2026 (Avg@16)</td><td>11.0</td><td></td><td>2.1 / 12.3</td><td>0.2</td><td>0.0 / 0.0</td><td>0.2 / 36.0</td></tr><tr><td>HMMT Feb. 2026 (Avg@16)</td><td>7.6</td><td>#.3</td><td>3.0/11.0</td><td>0.4</td><td>1.5 /0.0</td><td>9.5 / 26.5</td></tr><tr><td colspan="7">Code Generation</td></tr><tr><td>HumanEval</td><td>46.3</td><td>31.7</td><td>32.9 / 52.4</td><td>30.5</td><td>15.2 / 29.3</td><td>68.9 / 83.5</td></tr><tr><td>MBPP</td><td>45.0</td><td>42.4</td><td>31.8 /39.6</td><td>23.4</td><td>8.4 / 14.6</td><td>48.8 / 72.0</td></tr><tr><td>LiveCodeBench v6 (Pass@3)</td><td>16.6</td><td>14.3</td><td>11.4 / 16.4</td><td>3.4</td><td>6.9 / 5.7</td><td>11.4/37.7</td></tr><tr><td colspan="7">Reasoning</td></tr><tr><td>BBH (3-shot)</td><td>37.6</td><td>30.1</td><td>30.9 / 56.0</td><td>21.9</td><td>37.0 / 60.3</td><td>67.4 / 70.8</td></tr><tr><td>ZebraLogic</td><td>3.5</td><td>2.5</td><td>3.8/ 29.0</td><td>1.2</td><td>3.3 / 23.4</td><td>1.0/ 11.2</td></tr><tr><td colspan="7">Function Calling</td></tr><tr><td>BFCL v3</td><td>49.4</td><td>46.5</td><td>47.9/49.3</td><td>37.7</td><td>38.8 / 39.0</td><td>41.3 / 49.6</td></tr></table>

## 5.5.3 Inference Efficiency

As discussed in Section 5.2, we deliberately adopt a pure instruct architecture without a thinking mode, priori tizing inference efficiency for edge/on-device deployment scenarios. In this section, we provide quantitative evidence supporting the effectiveness of this design decision.

To measure how efficiently a model converts generation time into task performance, we introduce the Score Efficiency metric, defined as:

$$
\mathrm { S c o r e ~ E f f i c i e n c y } = \frac { \mathrm { S c o r e } \times \mathrm { T P S } } { \mathrm { A v e r a g e ~ O u t p u t ~ T o k e n ~ L e n g t h } } ,\tag{20}
$$

where Score denotes the benchmark accuracy (or pass rate) achieved by the model, TPS denotes the decoding throughput in tokens per second measured under the same serving setup, and Average Output Token Length is the mean number of generated tokens across all evaluation samples on the corresponding benchmark. Equivalently, since Average Output Token Length divided by TPS corresponds to the average generation time per sample, Score Efficiency can be interpreted as the task score delivered per unit of wall-clock generation time. A higher Score Efficiency indicates that the model achieves comparable or superior task performance with fewer output tokens and higher decoding speed, which directly translates to lower inference latency and reduced computational cost during serving.

The GDN layers greatly boost the decoding speed of IronLLM over long generations. Table 6 measures single-user decoding speed while each model generates 32K tokens (vLLM, RTX 4090, batch size 1, no speculative decoding). IronLLM-0.6B decodes at 460 tokens/s at 1K tokens and still at 377 tokens/s at 32K, a 1.22× increase in per-token latency over the whole generation, and is 8–9% faster than its architectural counterpart Qwen3.5-0.8B at every length. The full-attention Qwen3-0.6B starts at a similar speed but slows by 2.87× and falls to 166 tokens/s at 32K, making IronLLM 2.27× faster there. The shallower LFM2-700M (16 layers) remains about 10% faster than IronLLM throughout; with X-MTP (Section 6.1.3), IronLLM exceeds it on mathematics and code.

Table 6: Single-user decoding speed over long generations. Decode speed (tokens/s) at position L of a 32K-token generation at batch size 1, i.e. the inverse of the per-token latency (TPOT) averaged over the 512 tokens preceding L, and the growth of TPOT from 1K to 32K. RTX 4090, vLLM 0.17.1, BF16, CUDA graphs, greedy decoding; median of 3 prompts. No speculative decoding.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Attention</td><td colspan="4">Decode speed at position L (tokens/s)</td><td rowspan="2">TPOT growth 1K→32K</td></tr><tr><td>1K</td><td>4K</td><td>16K</td><td>32K</td></tr><tr><td>IronLLM-0.6B</td><td>GDN hybrid</td><td>460</td><td>445</td><td>410</td><td>377</td><td>1.22×</td></tr><tr><td>IronLLM-0.6B-Light</td><td>GDN hybrid</td><td>483</td><td>465</td><td>428</td><td>392</td><td>1.23×</td></tr><tr><td>Qwen3.5-0.8B</td><td>GDN hybrid</td><td>422</td><td>409</td><td>380</td><td>352</td><td>1.20×</td></tr><tr><td>LFM2-700M</td><td>Conv hybrid</td><td>508</td><td>498</td><td>457</td><td>416</td><td>1.22×</td></tr><tr><td>Qwen3-0.6B</td><td>Full attention</td><td>476</td><td>402</td><td>251</td><td>166</td><td>2.87×</td></tr><tr><td>MiniCPM5-1B</td><td>Full attention</td><td>368</td><td>360</td><td>322</td><td>279</td><td>1.32×</td></tr></table>

Table 7 reports the average output token length and Score Efficiency for each model across all evaluation benchmarks. Note that Qwen3-0.6B, Qwen3.5-0.8B, LFM2-700M, and MiniCPM5-1B only show results in non-thinking modes, as explicit thinking-mode generation is inherently at odds with the goal of inference efficiency. Under this comparison, IronLLM-0.6B demonstrates a substantial advantage, achieving the highest Score Efficiency across nearly all benchmarks by generating significantly shorter outputs while maintaining competitive accuracy, which directly validates our Instruct-Only design choice that prioritizes concise, direct responses over verbose reasoning traces. MiniCPM5-1B shows moderate efficiency but still trails IronLLM by a notable margin, while both Qwen models exhibit the lowest Score Efficiency due to residual over-generation behavior inherited from their thinking-oriented training, producing disproportionately long outputs even without explicit chain-of-thought prompting.

## 5.5.4 MOPD Capability Integration

We compare the General SFT checkpoint, independently optimized Domain Experts, a TIES parametermerging baseline, and the unified model trained with MOPD. The Domain Experts serve as task-specific reference points, whereas TIES and MOPD consolidate capabilities from all specialists into a single model.

As shown in Table 8, MOPD improves over General SFT and outperforms TIES Merging on all 13 benchmarks, demonstrating consistent integration without the negative transfer observed under parameter merging. It also matches or exceeds the corresponding Domain Expert on MMLU-Pro, CMMLU, and ArenaHard, and approaches expert performance on several instruction-following and knowledge benchmarks. Although gaps remain on some mathematics and code tasks, the overall results show that MOPD effectively integrates complementary specialist capabilities into a unified model.

Table 7: Inference-efficiency comparison between IronLLM-0.6B and representative models. A.T. and S.E. denote average output token length and Score Efficiency, respectively.
<table><tr><td rowspan="2">Benchmark (Metric)</td><td colspan="2">IronLLM-0.6B</td><td colspan="2">IronLLM-0.6B-Light</td><td colspan="2">Qwen3-0.6B7</td><td colspan="2">LFM2-700M7</td><td colspan="2">Qwen3.5-0.8B7</td><td colspan="2">MiniCPM5-1B7</td></tr><tr><td>A.T.</td><td>S.E.</td><td>A.T.</td><td>S.E.</td><td>A.T.</td><td>S.E.</td><td>A.T.</td><td>S.E.</td><td>A.T.</td><td>S.E.</td><td>A.T.</td><td>S.E.</td></tr><tr><td colspan="9">Long Context RULER</td><td>694.94</td><td></td><td>117.0</td><td>161.06</td></tr><tr><td>LongBench v2</td><td>47.0 173.9</td><td>703.71 58.62</td><td>49.3 201.8</td><td>652.56 52.91</td><td>125.1 868.1</td><td>53.57 5.44</td><td>50.4 1371.6</td><td>463.96 1.69</td><td>44.3 652.7</td><td>15.01</td><td>1064.1</td><td>4.27</td></tr><tr><td colspan="9">General Knowledge</td><td></td><td></td><td></td><td>5.93</td></tr><tr><td>MMLU-Pro MMLU-Redux</td><td>345.1 177.0</td><td>46.01 128.54</td><td>401.5 212.9</td><td>35.01 103.46</td><td>36.9 103.6</td><td>109.68 74.46</td><td>675.7 336.3</td><td>13.26 58.25</td><td>4652.2 2225.3</td><td>2.68 8.57</td><td>1731.3 385.2</td><td>43.24</td></tr><tr><td>C-Eval CMMLU</td><td>13.8 86.4</td><td>1293.00 215.60</td><td>137.3 126.3</td><td>127.68 140.41</td><td>7.3 182.9</td><td>961.21 41.14</td><td>562.3 392.1</td><td>27.98 40.62</td><td>1311.9 1476.4</td><td>7.77 7.87</td><td>1027.2 897.6</td><td>14.83 20.63</td></tr><tr><td colspan="9">Instruction Following</td><td></td><td></td><td></td></tr><tr><td>IFEval</td><td>265.5</td><td>107.35</td><td>269.6</td><td>101.05</td><td>1130.6</td><td>8.30</td><td>1560.7</td><td>16.75</td><td>334.2</td><td>47.70</td><td>806.1</td><td>24.38</td></tr><tr><td colspan="9">IFBench</td><td></td><td></td><td>2992.2</td><td>3.01</td></tr><tr><td>Multi-IF</td><td>665.0 226.7</td><td>10.22 60.33</td><td>783.7 358.6</td><td>8.17 30.58</td><td>1080.3 522.6</td><td>2.40 9.74</td><td>1329.8 705.3</td><td>5.32 15.75</td><td>672.7 368.6</td><td>8.19 38.23</td><td>2743.2</td><td>4.95</td></tr><tr><td>Subjective Quality</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">AlpacaEval 2.0</td><td>0.73</td><td></td><td>1028.4</td><td>1.08</td></tr><tr><td>ArenaHard</td><td>510.2 1058.7</td><td>7.53 4.25</td><td>555.6 1231.9</td><td>5.00 1.94</td><td>586.5 1064.4</td><td>0.95 0.54</td><td>502.6 1016.0</td><td>5.81 3.68</td><td>772.8 1981.1</td><td>0.46</td><td>2650.3</td><td>0.66</td></tr><tr><td colspan="9">Mathematics</td><td></td><td></td><td></td><td></td></tr><tr><td>MATH-500 GSM8K</td><td>1029.4 360.8</td><td>24.68 82.15</td><td>1865.9 400.7</td><td>12.06 72.02</td><td>714.2</td><td>12.13</td><td>873.2</td><td>13.15</td><td>8686.4</td><td>1.77</td><td>5727.6</td><td>2.75</td></tr><tr><td colspan="9"></td><td></td><td>4.47</td><td>681.7</td><td>27.65</td></tr><tr><td>AIME 2025 (Avg@16)</td><td>2949.3</td><td>1.31</td><td>3864.0</td><td>0.63</td><td>319.5 1611.0</td><td>32.10 0.92</td><td>417.4 2019.5</td><td>52.36 0.69</td><td>3884.8 14074.7</td><td>0.042</td><td>8198.9</td><td>0.00</td></tr><tr><td>AIME 2026 (Avg@16)</td><td>3398.0</td><td>1.22</td><td>5304.8</td><td>0.32</td><td>1596.4</td><td>0.22</td><td>1996.3</td><td>0.044</td><td>14155.3</td><td>0.00</td><td>12720.9</td><td>0.005</td></tr><tr><td>HMMT Feb. 2026 (Avg@16)</td><td>2239.6</td><td>1.28</td><td>4115.9</td><td>0.50</td><td>1670.3</td><td>0.30</td><td>1375.9</td><td>0.11</td><td>12558.7</td><td>0.043</td><td>4673.2</td><td>0.57</td></tr><tr><td colspan="9">Code Generation</td><td></td><td></td><td></td><td></td></tr><tr><td>HumanEval MBPP</td><td>101.6 248.9</td><td>171.95 68.16</td><td>125.2 133.1</td><td>76.36 117.81</td><td>155.4</td><td>35.18</td><td>86.8 658.0</td><td>146.13</td><td>841.6</td><td>6.37</td><td>827.9</td><td>23.22</td></tr><tr><td colspan="9">LiveCodeBench v6 (Pass@3)</td><td>848.3</td><td>3.49</td><td>4140.7</td><td>3.29</td></tr><tr><td></td><td>287.4</td><td>21.74</td><td>357.4</td><td>13.79</td><td>433.9 1348.5</td><td>12.17 1.41</td><td>503.1</td><td>14.79 2.84</td><td>13512.2</td><td>0.18</td><td>3870.6</td><td>0.82</td></tr><tr><td>Reasoning</td><td>58.3</td><td>243.14</td><td>88.4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">BBH ZebraLogic</td><td>2580.0</td><td>5.05</td><td>680.5</td><td></td><td>27.62</td></tr><tr><td></td><td>512.0</td><td>2.58</td><td>1339.5</td><td>133.43 0.73</td><td>85.8 478.3</td><td>59.76 1.32</td><td>198.7 1826.1</td><td>45.83 0.27</td><td>15448.8</td><td>0.075</td><td>19148.7</td><td>0.015</td></tr><tr><td>Function Calling</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">BFCL v3</td><td></td><td>224.83</td><td></td><td>51.6</td><td>223.15</td></tr><tr><td></td><td>381.4</td><td>48.86</td><td>541.8</td><td>33.67</td><td>81.7</td><td>97.28</td><td>61.5</td><td>255.28</td><td>60.7</td><td></td><td></td><td></td></tr></table>

Table 8: Capability integration results for IronLLM-0.6B. Domain Expert denotes the independently trained specialist for each domain, while TIES Merging and MOPD produce unified multi-domain models. Bold and underlined values indicate the best and second-best results, respectively.
<table><tr><td>Category</td><td>Benchmark</td><td>General SFT</td><td>Domain Expert</td><td>TIES Merging</td><td>MOPD</td></tr><tr><td rowspan="2">Mathematics</td><td>GSM8K</td><td>61.0</td><td>83.9</td><td>68.9</td><td>78.6</td></tr><tr><td>AIME 2026 (Avg@16)</td><td>0.6</td><td>14.0</td><td>0.6</td><td>11.0</td></tr><tr><td rowspan="3">Code</td><td>HumanEval</td><td>37.8</td><td>70.1</td><td>13.4</td><td>46.3</td></tr><tr><td>MBPP</td><td>25.0</td><td>57.8</td><td>43.8</td><td>45.0</td></tr><tr><td>LiveCodeBench v6 (Pass@3)</td><td>10.9</td><td>24.0</td><td>14.9</td><td>16.6</td></tr><tr><td rowspan="2">Instruction Following</td><td>IFEval IFBench</td><td>45.8 11.6</td><td>77.5</td><td>38.5</td><td>75.6</td></tr><tr><td></td><td></td><td>19.1</td><td>16.0</td><td>18.0</td></tr><tr><td rowspan="4">MCQA</td><td>MMLU-Pro</td><td>31.2</td><td>41.5</td><td>28.7</td><td>42.1</td></tr><tr><td>MMLU-Redux</td><td>54.7</td><td>60.6</td><td>50.3</td><td>60.4</td></tr><tr><td>C-Eval</td><td>43.3</td><td>49.6</td><td>36.2</td><td>47.3</td></tr><tr><td>CMMLU</td><td>44.5</td><td>37.5</td><td>36.1</td><td>49.4</td></tr><tr><td rowspan="2">Open-Ended QA</td><td>AlpacaEval 2.0</td><td>1.5</td><td>13.5</td><td>1.4</td><td>10.2</td></tr><tr><td>ArenaHard</td><td>1.2</td><td>11.0</td><td>2.3</td><td>11.9</td></tr></table>

## 6 Lightweight Multi-Token Prediction

## 6.1 X-MTP Design

Inspired by recent advances in Multi-Token Prediction (MTP), including DeepSeek-V3 and MiMo-V2- Flash [1, 8], we investigate lightweight MTP designs for accelerating the autoregressive decoding of IronLLM-0.6B in on-device scenarios. We consider both standard sequential MTP and a shared-KV variant, as described below.

## 6.1.1 Standard MTP Architecture

![](images/e910cfb9c9e3b677e347fcb818a3901d6b4d96b09d63fb18555804c92d8ec321.jpg)  
Figure 8: The conventional MTP architecture. Each prediction depth uses an independent MTP block.

Let $x _ { 1 : T } = ( x _ { 1 } , \ldots , x _ { T } ) \in \mathcal { V } ^ { T }$ denote an input sequence and $h _ { t } ^ { ( 0 ) } = \mathcal { F } ( x _ { < t } )$ the final hidden state produced by the backbone at position t. As shown in Figure 8, a standard MTP module predicts future tokens through a chain of auxiliary Transformer blocks. At prediction depth $k \in \{ 1 , \ldots , K \}$ , the embedding of the teacherforced token $x _ { t + k }$ is fused with the hidden state from the preceding depth:

$$
z _ { t } ^ { ( k ) } = \mathbf { W _ { f u s e } ^ { ( k ) } } \left[ \operatorname { N o r m } _ { e } ^ { ( k ) } ( \mathbf { E } ( x _ { t + k } ) ) ; \operatorname { N o r m } _ { h } ^ { ( k ) } \left( h _ { t } ^ { ( k - 1 ) } \right) \right] , \qquad h _ { t } ^ { ( k ) } = \mathcal { D } ^ { ( k ) } \left( z _ { \le t } ^ { ( k ) } \right) ,\tag{21}
$$

where E is the token embedding, $\mathcal { D } ^ { ( k ) }$ is the k-th MTP Transformer block, and $[ \cdot ; \cdot ]$ denotes concatenation. The output distribution

$$
\pmb { p } _ { t } ^ { ( k ) } = \mathrm { s o f t m a x } \Big ( \mathbf { W } _ { \mathrm { o u t } } \mathrm { N o r m } _ { o } ^ { ( k ) } \left( \pmb { h } _ { t } ^ { ( k ) } \right) \Big )\tag{22}
$$

is supervised by $x _ { t + k + 1 }$ . Thus, the first MTP depth consumes $\mathbf { E } ( x _ { t + 1 } )$ and predicts $x _ { t + 2 } .$ , the second consumes $ { \mathbf { E } } ( x _ { t + 2 } )$ and predicts $x _ { t + 3 } ,$ , and so forth. IronLLM shares both E and $\mathbf { W _ { \mathrm { o u t } } }$ with the backbone, avoiding two vocabulary-sized parameter matrices. The auxiliary objective is

$$
{ \mathcal { L } } _ { \mathrm { M T P } } = \frac { \sum _ { k = 1 } ^ { K } \sum _ { t } m _ { t } ^ { ( k ) } \mathrm { C E } \left( p _ { t } ^ { ( k ) } , x _ { t + k + 1 } \right) } { \sum _ { k = 1 } ^ { K } \sum _ { t } m _ { t } ^ { ( k ) } } , \qquad { \mathcal { L } } = { \mathcal { L } } _ { \mathrm { A R } } + \lambda _ { \mathrm { M T P } } { \mathcal { L } } _ { \mathrm { M T P } } ,\tag{23}
$$

where $m _ { t } ^ { ( k ) } \in \{ 0 , 1 \}$ denotes whether the target $x _ { t + k + 1 }$ is valid. Positions affected by padding, packedsequence boundaries, or sequence tails are masked out by setting $m _ { t } ^ { ( k ) } = 0$

Besides enabling speculative decoding, this objective encourages the backbone representation to encode information about a longer prediction horizon. However, standard sequential MTP may exhibit a traininference discrepancy. During training, the module at depth k is conditioned on the ground-truth future token $x _ { t + k }$ through teacher forcing. During decoding, by contrast, deeper MTP steps consume draft tokens generated by the preceding steps. An inaccurate early draft can therefore shift the inputs of subsequent steps away from their training distribution and propagate errors through the prediction chain, reducing the acceptance probability at deeper depths. This discrepancy does not negate the advantages of standard MTP, which remains an effective drafting approach, but motivates us to explore a complementary architecture.

## 6.1.2 Cross-Step Parameter and KV Sharing

Inspired by GLM-5.2 [3], we further investigate shared-KV MTP, which anchors all prediction depths to the K/V states constructed at the first MTP step while retaining depth-dependent queries. Our experiments suggest that this design provides a favorable trade-off between robust multi-step drafting and inference efficiency.

![](images/e211ecc034bd308696b1dfeca1fad6fcbf6ac13adef12c7a571629ce5ed9fce4.jpg)  
Figure 9: The shared-KV MTP architecture. All prediction depths reuse one MTP block, while keys and values are constructed only at the first depth and shared by subsequent depths.

The shared-KV variant replaces the depth-specific modules $\{ \mathcal { D } ^ { ( 1 ) } , \ldots , \mathcal { D } ^ { ( K ) } \}$ with a single recurrently applied MTP block $\mathcal { D } _ { \theta }$ . The embedding normalization, hidden-state normalization, fusion projection, attention, feed-forward network, and output normalization are all shared across prediction depths:

$$
z _ { t } ^ { ( k ) } = \mathbf { W } _ { \mathrm { f u s e } } \left[ \operatorname { N o r m } _ { e } ( \mathbf { E } ( x _ { t + k } ) ) ; \operatorname { N o r m } _ { h } \left( h _ { t } ^ { ( k - 1 ) } \right) \right] .\tag{24}
$$

Parameter sharing alone reduces the auxiliary model size from approximately $K P _ { \mathrm { M T P } }$ to $P _ { \mathrm { M T P } }$ , but naively reapplying the shared block would still reconstruct keys and values at every prediction depth. The shared-KV variant further removes this redundancy throughfirst-step KV sharing.

Specifically, after the input normalization inside ${ \mathcal { D } } _ { \theta } .$ , the first MTP depth constructs a static KV bank:

$$
\begin{array} { r } { \bar { \bf K } = \mathrm { R o P E } \left( { \bf W } _ { K } \widetilde { \bf Z } ^ { ( 1 ) } \right) , \qquad \bar { \bf V } = { \bf W } _ { V } \widetilde { \bf Z } ^ { ( 1 ) } , } \end{array}\tag{25}
$$

where $\widetilde { \mathbf { Z } } ^ { ( 1 ) }$ denotes the normalized first-depth fused states. Every depth retains its own query,

$$
{ \bf Q } ^ { ( k ) } = \mathrm { R o P E } \left( { \bf W } _ { Q } { \widetilde { \bf Z } } ^ { ( k ) } \right) , \qquad { \bf A } ^ { ( k ) } = \mathrm { A t t e n t i o n } \left( { \bf Q } ^ { ( k ) } , { \bar { \bf K } } , { \bar { \bf V } } ; { \bf M } _ { \mathrm { c a u s a l } } \right) ,\tag{26}
$$

but depths $k > 1$ neither execute ${ \bf W } _ { K } / { \bf W } _ { V }$ nor append new entries to the KV bank. They only compute a new fusion state and query, attend to the same first-depth causal memory, and propagate the resulting hidden state to the next depth. Hence, shared-KV MTP preserves depth-dependent representations while sharing both parameters and memory.

For K prediction depths, the conventional design performs K sets of Q/K/V projections and maintains K MTP caches. Shared-KV MTP performs K query projections but only one set of K/V projections and maintains a single MTP cache. Ignoring the embedding and language-model head already shared with the backbone, the auxiliary parameter overhead is reduced by approximately a factor of K, while the K/V projection and cache costs are reduced from $K ( C _ { Q } + C _ { K } + \bar { C _ { V } } )$ and $K M _ { \mathrm { K V } }$ to $K C _ { Q } + C _ { K } + C _ { V }$ and M , respectively. This reduction is particularly important on memory-constrained edge devices.

During training, the first depth builds an immutable full-sequence KV bank, which remains differentiable so that losses from deeper depths can update the first-depth fusion and K/V projections. During autoregressive inference, the backbone cache and the shared MTP cache are kept separate. Only the first MTP depth appends K/V for tokens that have been committed after verification; deeper depths access this cache in read-only mode. Consequently, rejected draft tokens never contaminate the persistent shared MTP cache, and no per-depth cache replay is required after a rejection.

The concrete parameter counts are summarized in Table 9. A single MTP block contains only 20.45M parameters, corresponding to approximately 3.1% of the 653.70M-parameter main model, demonstrating the lightweight nature of the MTP module. With three prediction depths, standard MTP uses three such blocks, resulting in 715.05M total parameters, whereas shared-KV MTP reuses a single block and reduces the total parameter count to 674.15M.

Table 9: Parameter comparison between three-depth standard MTP and shared-KV MTP. Component counts are approximate.
<table><tr><td>Architecture</td><td>Main Model</td><td>MTP Module</td><td>Total Parameters</td></tr><tr><td>Standard MTP (K = 3)</td><td>653.70M</td><td>20.45M×3</td><td>715.05M</td></tr><tr><td>Shared-KV MTP (K = 3)</td><td>653.70M</td><td>20.45M×1</td><td>674.15M</td></tr></table>

## 6.1.3 Inference Speed

We measure the end-to-end decoding speedup of shared-KV X-MTP with K = 3 in vLLM on a single RTX 4090 at batch size 1, the regime of on-device serving. The backbone verifies every draft exactly under greedy decoding, so X-MTP never changes the output distribution; the speedup is therefore a pure latency gain. Table 10 reports decode throughput on 100 prompts from each of IFEval, GSM8K, MBPP, MATH-500, and ShareGPT, the last consisting of first user turns from real conversations. On mathematics and code, the drafter commits 3.3–3.6 tokens per backbone step and X-MTP accelerates decoding by 1.37–1.48×, lifting IronLLM-0.6B from 460 to 629–678 tokens/s. Open-ended conversation is harder to anticipate: on ShareGPT X-MTP commits 2.8 tokens per step for a 1.17× speedup, and on IFEval, whose constrained instruction-following responses are least predictable from the backbone state, the gain drops to 1.04×. A single draft depth $( \boldsymbol { K } = 1 )$ does not pay for its drafting overhead (0.83–0.93×), whereas each additional depth adds a further speedup, confirming that the recurrent shared-KV block remains accurate at deeper depths: at K = 3 the per-depth acceptance rates on MATH-500 are 95%, 89%, and 82%.

Table 10: Decoding speedup of shared-KV X-MTP. Batch size 1, greedy decoding with exact verification by the backbone, so drafts never change the output distribution. N counts the prompts (of 100 sampled per benchmark) whose response ends with EOS in every run, τ is the mean number of tokens committed per backbone step at K=3, and TPS is the decode throughput with prefill excluded. RTX 4090, vLLM 0.17.1.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">N</td><td rowspan="2">Avg. out len</td><td rowspan="2">T</td><td colspan="2">Decode TPS</td><td colspan="3">Speedup</td></tr><tr><td>w/o MTP</td><td> $K = 3$ </td><td>K = 1</td><td>K = 2</td><td>K = 3</td></tr><tr><td>IFEval</td><td>81</td><td>237</td><td>2.48</td><td>460</td><td>477</td><td>0.83×</td><td>1.02×</td><td>1.04×</td></tr><tr><td>GSM8K</td><td>99</td><td>139</td><td>3.42</td><td>460</td><td>662</td><td>0.92×</td><td>1.27×</td><td>1.44×</td></tr><tr><td>MBPP</td><td>99</td><td>93</td><td>3.27</td><td>460</td><td>629</td><td>0.91×</td><td>1.23×</td><td>1.37×</td></tr><tr><td>MATH-500</td><td>93</td><td>317</td><td>3.55</td><td>460</td><td>678</td><td>0.93×</td><td>1.29×</td><td>1.48×</td></tr><tr><td>ShareGPT</td><td>87</td><td>268</td><td>2.82</td><td>460</td><td>540</td><td>0.87×</td><td>1.11×</td><td>1.17×</td></tr></table>

## 6.2 Lightweight Verification Head

## 6.2.1 Verification Head Design

In deployment and inference speed tests with the MTP module, we observe that the wall-clock speedup falls noticeably short of the ideal gain implied by the acceptance rate, because predicting several tokens per step also inflates the latency of each step. Under the conventional drafting schedule, one decoding step must run all K MTP depths to completion before any draft can be verified, and verification happens only in the next backbone forward pass. Each depth therefore pays for a full MTP block plus a vocabulary-sized output projection, the latter being by far the dominant cost. Since prefix acceptance discards every depth beyond the first rejection, this compute is spent unconditionally and becomes pure overhead whenever an early draft fails.

Verification itself is a second source of deployment complexity, as it requires speculatively advancing the backbone state over the proposed tokens and then rolling it back to the last accepted position. For softmaxattention layers a rollback reduces to truncating the KV cache, but IronLLM interleaves GDN layers whose fixed-size recurrent memory and short convolution state summarize the entire prefix in place and cannot be restored by slicing; an exact rollback demands either re-running the accepted prefix or snapshotting and restoring these states at every step. Both issues share a root cause: acceptance is decided too late and by too heavy a mechanism. We therefore attach a lightweight verification head to each MTP depth, which predicts whether a draft token will be accepted at the moment that draft is produced, without consulting the backbone.

![](images/45644795f2ba09db0803757d1a4e62d4eb6611ea6c910a47785719eea36ce0eb.jpg)  
Figure 10: The lightweight verification head. At every MTP depth, a shared two-layer MLP consumes the aligned input hidden state and the output hidden state of that depth and emits a scalar acceptance confidence, which gates whether drafting proceeds to the next depth.

Architecture. As illustrated in Figure 10, the verification head is a two-layer MLP placed after each MTP depth. It observes the two hidden states that already characterize the prediction made at that depth: the aligned input state $\boldsymbol { z } _ { t } ^ { ( k ) }$ produced by $\mathbf { W } _ { \mathrm { f u s e } } ,$ , and the output state ${ \bf \Lambda } _ { h _ { t } ^ { ( k ) } }$ from which the draft token is decoded. Following the same norm-then-concatenate fusion used by the MTP block, each stream is normalized separately before concatenation:

$$
c _ { t } ^ { ( k ) } = \sigma \Big ( \mathbf { w } _ { 2 } ^ { \top } \operatorname { S i L U } \Big ( \mathbf { W } _ { 1 } \Big [ \operatorname { N o r m } _ { a } \Big ( z _ { t } ^ { ( k ) } \Big ) ; \operatorname { N o r m } _ { m } \Big ( h _ { t } ^ { ( k ) } \Big ) \Big ] \Big ) \Big ) ,\tag{27}
$$

The scalar $c _ { t } ^ { ( k ) } \in ( 0 , 1 )$ estimates the probability that the draft proposed at depth k matches the token the backbone would have committed. As with the shared-KV MTP block, one set of head parameters is reused across all K depths, so the overhead is constant in K: 2.10M parameters, about 0.32% of the 653.70Mparameter backbone and roughly one tenth of an MTP block.

Confidence-gated drafting. At inference time the head turns drafting from a fixed K-step pipeline into an adaptive one. Each depth is assigned its own threshold $\tau _ { k }$ , and once depth k has produced its draft token the corresponding confidence is evaluated immediately, in parallel with decoding that token: clearing $\tau _ { k }$ retains the draft and drafting continues to depth $k + 1$ , otherwise drafting terminates and depths $k + 1 , \ldots , K$ are never executed. To prevent errors from accumulating silently along the chain, the decision is made on the joint confidence, i.e. the product of the per-depth confidences accumulated so far:

$$
C _ { t } ^ { ( k ) } = \prod _ { j = 1 } ^ { k } c _ { t } ^ { ( j ) } , \qquad \hat { k } _ { t } = \operatorname* { m a x } \left\{ k \in \{ 0 , \dots , K \} \Big | C _ { t } ^ { ( j ) } \geq \tau _ { j } \ \forall j \leq k \right\} ,\tag{28}
$$

with $\hat { k } _ { t }$ the number of accepted drafts. Because each factor lies in $( 0 , 1 ) , C _ { t } ^ { ( k ) }$ is non-increasing in k and approximates the probability that the entire draft prefix of length k is correct, rather than that of one isolated depth. A locally plausible depth can thus no longer extend a prefix whose joint reliability has already decayed.

The scheme also removes the rollback problem entirely. Acceptance is settled before the backbone is invoked, so the backbone forward pass extends over the committed prefix only, and rejected drafts never reach the backbone KV cache, the GDN recurrent and convolution states, or the shared MTP cache. No truncation, replay, or state snapshotting is required at any point, which is precisely what makes speculative decoding practical for a hybrid linear-attention model on device.

## 6.2.2 Training Strategy

Following PIPO [74], we train the verification head in two stages. In both, the head is a purely auxiliary regressor: the two hidden states feeding Equation (27) and the regression target are both detached, so its gradients never perturb the backbone or the MTP block. Adding the head is therefore behavior-preserving for the base model, and the overall objective simply extends Equation (23) with one more weighted term,

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { A R } } + \lambda _ { \mathrm { M T P } } \mathcal { L } _ { \mathrm { M T P } } + \lambda _ { \mathrm { v e r } } \mathcal { L } _ { \mathrm { v e r } } , \qquad \mathcal { L } _ { \mathrm { v e r } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { \sum _ { t } m _ { t } ^ { ( k ) } \mathrm { B C E } \left( c _ { t } ^ { ( k ) } , y _ { t } ^ { ( k ) } \right) } { \sum _ { t } m _ { t } ^ { ( k ) } } ,\tag{29}
$$

where $m _ { t } ^ { ( k ) }$ is the same validity mask used by the MTP objective.

Stage I: probability matching during SFT. During SFT, MTP is trained with teacher forcing, so no on-policy draft tokens exist and the acceptance event is not observable. We instead supervise the head with the MTP head’s own probability of the ground-truth continuation,

$$
y _ { t } ^ { ( k ) } = \mathrm { s g } \Big [ p _ { t } ^ { ( k ) } ( x _ { t + k + 1 } ) \Big ] ,\tag{30}
$$

where sg[·] denotes the stop-gradient operator. Under a deterministic verifier that commits the ground-truth token, this quantity is precisely the acceptance probability of the draft at depth $k .$ It remains only a surrogate, since it is computed from teacher-forced inputs instead of self-generated drafts and scores the ground-truth token instead of the one the backbone would commit. Its value is that the signal is dense and essentially free at every valid position and depth, yielding a well-scaled head that initializes the sparser, draft-dependent objective of the next stage.

Stage II: acceptance matching during OPD. The on-policy distillation (OPD) stage removes both mismatches: drafts are generated by the MTP chain itself, and a teacher distribution acts as the verifier. As observed in PIPO, the OPD teacher plays exactly the role of the speculative-decoding verifier, under which a draft $\hat { y }$ proposed from $q$ and verified against $p$ is accepted with probability min $\{ 1 , \bar { p } ( \hat { y } ) / q ( \hat { y } ) \}$ }. Taking the depth-k MTP head as the proposal and the teacher as the verifier gives

$$
\hat { y } _ { t } ^ { ( k ) } \sim p _ { t } ^ { ( k ) } , \qquad y _ { t } ^ { ( k ) } = \mathrm { s g } \left[ \operatorname* { m i n } \left\{ 1 , \ : \frac { p _ { t } ^ { \mathrm { t e a } } \Bigl ( \hat { y } _ { t } ^ { ( k ) } \Bigr ) } { p _ { t } ^ { ( k ) } \Bigl ( \hat { y } _ { t } ^ { ( k ) } \Bigr ) } \right\} \right] .\tag{31}
$$

The head stays detached here as well, so this objective leaves the distillation of the model itself untouched.

## 6.2.3 Effectiveness of Verification Head

We jointly train the verification head during the MOPD stage. We evaluate the resulting head on three representative benchmarks—MMLU-Redux, IFEval, and GSM8K—with the inference threshold uniformly set to 0.9 across all depths. Table 11 reports the evaluation results. We observe that for general QA tasks such as MMLU-Redux and IFEval, the verification head achieves nearly lossless performance while maintaining high acceptance rates. However, for complex reasoning tasks like GSM8K, using the verification head leads to noticeable score degradation. This is primarily because for tasks that require rigorous reasoning, such as mathematics and code generation, a single verification error introduced by the verification head can disrupt the model’s reasoning chain, and the accumulated errors ultimately lead to incorrect final answers. More results—including the accuracy trajectory during joint training and a systematic sweep of the inference verification threshold are provided in Appendix B.4.

Table 11: Evaluation of the jointly trained verification head with the inference threshold uniformly set to 0.9. Main Model denotes non-speculative decoding with the backbone; Avg. Accept. Len. is the average number of tokens committed per decoding step, and Match Rate is the fraction of confidence-accepted draft tokens that agree with the tokens committed by the backbone, both measured under confidence-gated MTP decoding.
<table><tr><td>Benchmark</td><td>Main Model</td><td>w/ Ver. Head</td><td>∆</td><td>Avg. Accept. Len.</td><td>Match Rate</td></tr><tr><td>MMLU-Redux</td><td>60.35</td><td>60.43</td><td>+0.08</td><td>2.41</td><td>0.983</td></tr><tr><td>IFEval</td><td>75.60</td><td>72.83</td><td>-2.77</td><td>2.23</td><td>0.968</td></tr><tr><td>GSM8K</td><td>78.62</td><td>66.26</td><td>-12.36</td><td>2.12</td><td>0.967</td></tr></table>

## 7 Conclusion, Limitation, and Future Work

In this work, we present IronLLM-0.6B, a compact language model for efficient and capable on-device inference. It combines a hybrid attention architecture, a quality-oriented pre-training pipeline, and a posttraining recipe for small models. Multi-Domain On-Policy Distillation integrates complementary capabilities from domain-specialized teachers without the negative transfer of parameter merging, and an Instruct-Only objective yields concise responses. The streamlined IronLLM-0.6B-Light improves deployment and quantization efficiency, while X-MTP delivers a 1.48× decoding speedup via cross-step parameter and KV sharing. Together, these designs balance capability and efficiency, making the IronLLM models particularly well-suited to embodied systems with stringent compute, latency, and power constraints.

Several open challenges remain, pointing to clear directions for future work. Performance gaps persist on specialized mathematics and code-generation tasks, and the gains from our efficiency techniques may vary across hardware platforms, inference engines, and quantization settings. Future work will therefore pursue more capable and efficient edge-native architectures alongside data-efficient training recipes, extend the models toward compact multimodality and robot-side deployment, and broaden evaluation to production-grade edge hardware and embodied tasks, covering multilingual generation, safety, robustness, and tool use. We hope IronLLM models serve as a solid step toward bringing capable and efficient language models to embodied platforms.

## Acknowledgments

This work was made possible by the strong support of XPENG Robotics and the Foundation Model Team, along with the joint efforts of our research and engineering teams. We thank XPENG Robotics for providing computing and engineering resources, and for its continued commitment to building efficient foundation models for embodied intelligence. Special thanks go to all colleagues who contributed to data preparation, infrastructure support, model development, experimentation and evaluation.

## References

[1] DeepSeek-AI. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. URL https: //arxiv.org/abs/2412.19437.

[2] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[3] Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

[4] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. Kimi k3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

[5] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

[6] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/ blog?id=qwen3.5.

[7] William Merrill, Yanhong Li, Tyler Romero, Anej Svete, Caia Costello, Pradeep Dasigi, Dirk Groeneveld, David Heineman, Bailey Kuehl, Nathan Lambert, et al. Olmo hybrid: From theory to practice and back. arXiv preprint arXiv:2604.03444, 2026.

[8] Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen, Chen Zhang, Chenhong He, Chiheng Lou, Fuli Luo, Gang Wang, et al. Mimo-v2-flash technical report. arXiv preprint arXiv:2601.02780, 2026.

[9] Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, and Zhuang Liu. Transformers without normalization. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14901–14911. IEEE, 2025.

[10] Jungwook Choi, Zhuo Wang, Swagath Venkataramani, Pierce I-Jen Chuang, Vijayalakshmi Srinivasan, and Kailash Gopalakrishnan. Pact: Parameterized clipping activation for quantized neural networks, 2018. URL https://arxiv.org/abs/1805.06085.

[11] Team Olmo, Allyson Ettinger, Amanda Bertsch, et al. Olmo 3, 2026. URL https://arxiv.org/abs/ 2512.13961.

[12] Wenhan Ma, Jianyu Wei, Liang Zhao, Hailin Zhang, Bangjun Xiao, Lei Li, Qibin Yang, Bofei Gao, Yudong Wang, Rang Li, et al. Mopd: Multi-teacher on-policy distillation for capability integration in llm post-training. arXiv preprint arXiv:2606.30406, 2026.

[13] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, volume 2024, pages 21246–21263, 2024.

[14] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. In The twelfth international conference on learning representations, 2024.

[15] Team MiniCPM. Minicpm4: Ultra-efficient llms on end devices. arXiv preprint arXiv:2506.07900, 2025.

[16] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017. URL https://arxiv.org/abs/1706.03762.

[17] Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, et al. Kimi linear: An expressive, efficient attention architecture. arXiv preprint arXiv:2510.26692, 2025.

[18] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. In International Conference on Learning Representations, volume 2025, pages 29687–29707, 2025.

[19] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023. URL https://arxiv.org/abs/2305.13245.

[20] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In International conference on machine learning, pages 7480–7512. PMLR, 2023.

[21] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864, 2021. URL https: //arxiv.org/abs/2104.09864.

[22] Zihan Qiu, Zeyu Huang, Kaiyue Wen, Peng Jin, Bo Zheng, Yuxin Zhou, Haofeng Huang, Zekun Wang, Xiao Li, Huaqing Zhang, Yang Xu, Haoran Lian, Siqi Zhang, Rui Men, Jianwei Zhang, Ivan Titov, Dayiheng Liu, Jingren Zhou, and Junyang Lin. A unified view of attention and residual sinks: Outlierdriven rescaling is essential for transformer training, 2026. URL https://arxiv.org/abs/2601. 22966.

[23] Zixuan Jiang, Jiaqi Gu, Hanqing Zhu, and David Pan. Pre-rmsnorm and pre-crmsnorm transformers: equivalent and efficient pre-ln transformers. Advances in Neural Information Processing Systems, 36: 45777–45793, 2023.

[24] Prajit Ramachandran, Barret Zoph, and Quoc V. Le. Searching for activation functions, 2017. URL https://arxiv.org/abs/1710.05941.

[25] Zhen Qin, Weigao Sun, Dong Li, Xuyang Shen, Weixuan Sun, and Yiran Zhong. Lightning attention-2: A free lunch for handling unlimited sequence lengths in large language models. arXiv preprint arXiv:2401.04658, 2024. URL https://arxiv.org/abs/2401.04658.

[26] Ofir Press, Noah A. Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. In International Conference on Learning Representations, 2022. URL https://arxiv.org/abs/2108.12409.

[27] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[28] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations, volume 2024, pages 35549–35562, 2024.

[29] Songlin Yang and Yu Zhang. Fla: A triton-based library for hardware-efficient implementations of linear attention mechanism, January 2024. URL https://github.com/fla-org/ flash-linear-attention.

[30] Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, Qingquan Song, Shao Tang, Siyu Zhu, Steven Shimizu, Shivam Sahni, Haowen Ning, Yanning Chen, and Zhipeng Wang. Liger-kernel: Efficient triton kernels for LLM training. In Championing Open-source DEvelopment in ML Workshop @ ICML25, 2025. URL https://openreview.net/forum?id=36SjAIT42G.

[31] Hantian Ding, Zijian Wang, Giovanni Paolini, Varun Kumar, Anoop Deoras, Dan Roth, and Stefano Soatto. Fewer truncations improve language modeling. arXiv preprint arXiv:2404.10830, 2024.

[32] Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. arXiv preprint arXiv:2203.05482, 2022.

[33] Yunshui Li, Yiyuan Ma, Shen Yan, Chaoyi Zhang, Jing Liu, Jianqiao Lu, Ziwen Xu, Mengzhao Chen, Minrui Wang, Shiyi Zhan, et al. Model merging in pre-training of large language models. Advances in Neural Information Processing Systems, 38:133668–133691, 2026.

[34] Maosong Cao, Kai Chen, Haodong Duan, et al. Opencompass: A universal evaluation platform for large language models, 2026. URL https://arxiv.org/abs/2605.19276.

[35] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021.

[36] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multitask language understanding benchmark. Advances in Neural Information Processing Systems, 37: 95266–95290, 2024.

[37] Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, et al. Are we done with mmlu? In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5069–5096, 2025.

[38] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

[39] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791–4800, 2019.

[40] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4149–4158, 2019.

[41] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 2381–2391, 2018.

[42] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, 2020.

[43] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An adversarial winograd schema challenge at scale. In Proceedings ofthe AAAI Conference on Artificial Intelligence, 2020.

[44] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. Challenging BIG-Bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.

[45] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Associationfor Computational Linguistics, pages 1601–1611, 2017.

[46] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[47] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. arXiv preprint arXiv:2103.03874, 2021.

[48] Mark Chen et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

[49] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc V. Le, and Charles Sutton. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

[50] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. CMMLU: Measuring massive multitask language understanding in chinese. arXiv preprint arXiv:2306.09212, 2023.

[51] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-Eval: A multi-level multi-discipline chinese evaluation suite for foundation models. arXiv preprint arXiv:2305.08322, 2023.

[52] Kai Sun, Dian Yu, Dong Yu, and Claire Cardie. Investigating prior knowledge for challenging chinese machine reading comprehension. Transactions of the Association for Computational Linguistics, 8: 141–155, 2020.

[53] Zhengxiao Du, Aohan Zeng, Yuxiao Dong, and Jie Tang. Understanding emergent abilities of language models from the loss perspective. Advances in neural information processing systems, 37:53138–53167, 2024.

[54] Yuling Gu, Oyvind Tafjord, Bailey Kuehl, Dany Haddad, Jesse Dodge, and Hannaneh Hajishirzi. Olmes: A standard for language model evaluations. In Findings ofthe Associationfor Computational Linguistics: NAACL 2025, pages 5020–5048, 2025.

[55] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[56] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. URL https://arxiv.org/abs/ 1707.06347.

[57] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

[58] Boxin Wang, Chankyu Lee, Nayeon Lee, Sheng-Chieh Lin, Wenliang Dai, Yang Chen, Yangyi Chen, Zhuolin Yang, Zihan Liu, Mohammad Shoeybi, et al. Nemotron-cascade: Scaling cascaded reinforcement learning for general-purpose reasoning models. arXiv preprint arXiv:2512.13607, 2025.

[59] Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025.

[60] Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471, 2025.

[61] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In The Eleventh International Conference on Learning Representations, 2022.

[62] Liquid AI. Lfm2 technical report. arXiv preprint arXiv:2511.23404, 2025.

[63] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

[64] Valentina Pyatkin, Saumya Malik, Victoria Graf, Hamish Ivison, Shengyi Huang, Pradeep Dasigi, Nathan Lambert, and Hanna Hajishirzi. Generalizing verifiable instruction following. Advances in Neural Information Processing Systems, 38, 2026.

[65] Yun He, Di Jin, Chaoqi Wang, Chloe Bi, Karishma Mandyam, Hejia Zhang, Chen Zhu, Ning Li, Tengyu Xu, Hongjiang Lv, et al. Multi-if: Benchmarking llms on multi-turn and multilingual instructions following. arXiv preprint arXiv:2410.15553, 2024.

[66] Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. Length-controlled alpacaeval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475, 2024.

[67] Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline. arXiv preprint arXiv:2406.11939, 2024.

[68] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations, volume 2024, pages 39578–39601, 2024.

[69] Jasper Dekoninck, Nikola Jovanovic, Tim Gehrunger, Kári Rögnvaldsson, Ivo Petrov, Chenhao Sun, and´ Martin Vechev. Beyond benchmarks: Matharena as an evaluation platform for mathematics with llms. arXiv preprint arXiv:2605.00674, 2026.

[70] Naman Jain, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In International Conference on Learning Representations, volume 2025, pages 58791–58831, 2025.

[71] Bill Yuchen Lin, Ronan Le Bras, Kyle Richardson, Ashish Sabharwal, Radha Poovendran, Peter Clark, and Yejin Choi. Zebralogic: On the scaling limits of llms for logical reasoning. arXiv preprint arXiv:2502.01100, 2025.

[72] Shishir G. Patil, Huanzhi Mao, Charlie Cheng-Jie Ji, Fanjia Yan, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025.

[73] ModelScope Team. EvalScope: Evaluation framework for large models, 2024. URL https://github. com/modelscope/evalscope.

[74] Wenhui Tan, Minghao Li, Xiaoqian Ma, Siqi Fan, Xiusheng Huang, Liujie Zhang, Ruihua Song, and Weihang Chen. Pair-in, pair-out: Latent multi-token prediction for efficient llms. arXiv preprint arXiv:2605.27255, 2026.

[75] Xeron Du, Yifan Yao, Kaijing Ma, Bingli Wang, Tianyu Zheng, Minghao Liu, Yiming Liang, Xiaolong Jin, Zhenlin Wei, Chujie Zheng, et al. Supergpqa: Scaling llm evaluation across 285 graduate disciplines. Advances in Neural Information Processing Systems, 38, 2026.

## A Data Contamination Details

This appendix provides additional details of our benchmark contamination analysis, together with the corresponding experimental results.

N-gram Overlap Detection. We detect potential benchmark contamination between the pre-training corpus and evaluation benchmarks using an IDF-weighted 5-gram overlap framework. An inverted 5-gram index is constructed over benchmark question stems to efficiently retrieve candidate training documents with high lexical overlap.

For each retrieved candidate, both question and answer overlaps are jointly considered. We define the contamination score as

$$
\mathrm { s c = 0 . 7 5 \times \ s t e m _ { - } i d f _ { - } o v e r l a p + 0 . 2 5 \times \ a i o , }\tag{32}
$$

where stem\_idf\_overlap denotes the IDF-weighted lexical overlap between question stems, and aio (answer IDF overlap) measures lexical consistency between reference answers.

Candidate pairs with retrieval scores above 0.3 are retained for further verification. Samples are considered contaminated if sc ≥ 0.8 or aio ≥ 0.5, while borderline cases are resolved through benchmark-specific rules and manual inspection.

Embedding-based Semantic Retrieval. To identify benchmark contamination beyond lexical overlap, we further perform embedding-based semantic retrieval on the SFT corpus. Each SFT instruction is compared with benchmark questions using cosine similarity between sentence embeddings, allowing semantically similar but lexically different samples to be retrieved.

Candidate pairs with cosine similarity above 0.8 are further verified using an LLM as a judge, followed by human inspection. A sample is considered contaminated only when both the underlying semantics and evaluation target are consistent with the benchmark item, while semantically related but independently constructed examples are excluded from contamination statistics.

Results Table 12 reports IronLLM-0.6B performance on the original and non-contaminated subsets across the evaluated benchmarks. Overall, the differences are small on most benchmarks, with no systematic performance degradation after decontamination.

Table 12: Contamination Analysis. Contaminated samples are identified as the union of N-gram overlap detection against the pretraining corpus and embedding-based semantic retrieval against the SFT data.
<table><tr><td rowspan="2">Test Set</td><td colspan="3">IronLLM-0.6B</td></tr><tr><td>Orig.</td><td>Non-Contam.</td><td>Δ</td></tr><tr><td>MMLU-Pro</td><td>42.1</td><td>43.1</td><td>+1.0</td></tr><tr><td>MMLU-Redux</td><td>60.4</td><td>61.0</td><td>+0.6</td></tr><tr><td>C-Eval</td><td>47.3</td><td>47.0</td><td>-0.3</td></tr><tr><td>CMMLU</td><td>49.4</td><td>49.7</td><td>+0.3</td></tr><tr><td>IFEval</td><td>75.6</td><td>75.7</td><td>+0.1</td></tr><tr><td>GSM8K</td><td>78.6</td><td>76.9</td><td>-1.7</td></tr><tr><td>MBPP</td><td>45.0</td><td>47.3</td><td>+2.3</td></tr><tr><td>BBH</td><td>37.6</td><td>37.0</td><td>-0.6</td></tr></table>

## B Training and Evaluation Details

## B.1 Pre-training Loss Curve

Figure 11 shows the training loss over the full pre-training run (1.5M steps, 6.2 trillion tokens in total). Training remains stable throughout. After warmup, the loss decreases smoothly with no loss spikes or divergence observed. The downward jumps at stage boundaries reflect shifts in data composition toward higher proportions of math, code, and reasoning data. During the decay phase of Stage 3, the loss further decreases correspond to the learning-rate decay phase of the WSD schedule.

![](images/5c43204d24785c86ce821cb0223ed1a4c9a1a278b957274ecc4e882ded540e4a.jpg)  
Figure 11: Pre-training loss curve across the three training stages. The gray trace shows the raw per-step loss and the colored curves show the loss smoothed with a rolling average over 1K steps. The loss decreases smoothly throughout training.

## B.2 Base Model Evaluation Protocol

Base model evaluation can be divided into two categories: perplexity-based metrics and free-generation-based metrics. Perplexity-based approaches further split into the multiple-choice formulation (MCF), which requires the model to choose among explicit options (e.g., A/B/C/D), and the cloze formulation (CF), which compares likelihoods of candidate completions without exposing options in the prompt.

Prior work [53, 54] shows that MCF is especially difficult in early training, particularly for small models. We therefore use CF with benchmark-specific probability normalization for early-stage diagnostics in from-scratch runs, and report final base-model results using MCF. Unlike prior work that selectively reports the more favorable metric for each benchmark, we keep this protocol fixed throughout. We view MCF as better suited to practical option-based use cases once the model is mature. For the free-generation benchmarks, we apply minor prompt refinement and more carefully selected few-shot examples to reduce benchmark-specific sensitivity.

Figure 12 illustrates the distinction on MMLU, CMMLU, C-Eval, and their average. CF curves from two independent runs are smooth and clearly separable, while MCF curves are highly variable, frequently cross, and remain near the 25% random baseline, indicating that the model struggles with four-way multiple-choice questions during early training.

![](images/6ff15bb24c9a01ae6ebaaf160de7f92ca5c7041303e85435664d3e1b93684492.jpg)  
Figure 12: Comparison of CF and MCF evaluation trajectories for two pretraining runs conducted from scratch.

## B.3 MOPD Training Curves

Figure 13 shows the loss and reward over the first 10K steps of MOPD training. The distillation term decreases from 1.59 to 0.14, i.e. the student’s average per-token disagreement with its routed teacher is reduced by roughly an order of magnitude, and the total loss follows the same trajectory, falling from 6.41 to 0.57. Meanwhile the verifiable reward rises from near zero to 0.41, steeply over the first few hundred steps and then slowly but steadily. Training is stable throughout, with no loss spikes or reward collapse, and the two objectives improve jointly rather than trading off against each other.

![](images/87bbe0e0fc11103d58cfc5afc60ffddee3788c3b143339c83169c156456a99bf.jpg)

![](images/a93f64228820cf5e5bc0938b82308929f9c1d6b4566641b40b7738b9decf7bd3.jpg)  
Figure 13: MOPD training curves over the first 10K steps. Left: total optimized loss (dark) and the distillation term ${ \mathcal { L } } _ { \mathrm { K D } }$ alone (light); the warm-up steps, where the total loss peaks at 6.4, fall outside the plotted y-range. Right: batch-mean verifiable reward. Faint traces show raw per-step values, bold curves an exponential moving average (decay 0.97).

## B.4 Additional Results of the Verification Head

This appendix supplements the verification head evaluation in Section 6.2.3 with two additional studies of the jointly trained verification head: (i) the accuracy trajectory observed during the MOPD stage, and (ii) a systematic sweep of the inference confidence threshold, covering accuracy, average acceptance length, and confidence–verification consistency.

Accuracy during Joint Training. During the MOPD stage, we evaluate confidence-gated MTP decoding on GSM8K at every epoch with the uniform threshold τ = 0.9 used in Section 6.2.3. Figure 14 plots the resulting trajectory. Starting from a low level at MOPD initialization, the accuracy rises steeply within the first epoch of on-policy distillation, keeps climbing over the next few epochs, and then saturates into a stable plateau for the remainder of the stage. Most of the total gain is realized within the first epoch, indicating that the on-policy objective rapidly aligns the MTP chain, and hence the acceptance events that supervise the head, with the teacher distribution.

![](images/6d9362f47e454da1d835aebedff08e26e2613bbfce8175fe9ae740a1d201fe7e.jpg)  
Figure 14: GSM8K accuracy of MTP inference w/ verification head (uniform threshold $\tau = 0 . 9 )$ during MOPD training. The accuracy rises steeply in the early training phase and saturates into a stable plateau by the end of the stage.

Effect of the Confidence Threshold. The inference threshold controls how aggressively drafts are admitted, and hence the trade-off between the number of tokens committed per step and the risk of accepting an incorrect draft. Using the final MOPD checkpoint, we sweep the uniform per-depth threshold $\tau _ { k } \equiv \tau$ from 0.50 to 0.95 on IFEval. Figure 15 plots the accuracy, the average acceptance length, and the confidence–verification match rate for the ten uniform settings.

![](images/685663ed0bbdd657cc3f823f23851872a5ee2f53391111ca846aac0d21adefe2.jpg)

![](images/e91781f06bde0f5300b0b7d39664cdc4301f5538752d023b61f54323b5e51ea3.jpg)

![](images/89d8ba8fbc90a6d84cda8735a9e13111920c3d88115596a40eab2f686ee0f326.jpg)  
Figure 15: Effect of the uniform confidence threshold τ on IFEval: accuracy (left), average acceptance length (middle), and confidence–verification match rate (right). Ten uniform thresholds from 0.50 to 0.95 are evaluated.

The sweep exhibits a clear accuracy–speed trade-off. Relaxing the threshold admits more draft tokens per step but costs accuracy, whereas stricter settings progressively close the gap to the non-speculative baseline while still committing multiple tokens per step. The match rate improves accordingly as the threshold tightens, confirming that the head’s confidence scores are informative: drafts admitted with high confidence are almost always verified as correct, and at the strict end of the sweep nearly every accepted draft agrees with the backbone.