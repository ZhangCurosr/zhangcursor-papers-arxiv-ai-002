# ProQuant: Preserving Expert Quality over Quantity in Ultra-Low-Bit Quantization of Mixture-of-Experts Models

Jiaqi Chen<sup>1</sup>\*, Xianglong Yan<sup>1</sup>\*, Chengzhu Bao<sup>1</sup>, Xie Chen<sup>1</sup>, Yulun Zhang<sup>1†</sup> <sup>1</sup>Shanghai Jiao Tong University Correspondence: yulun100@gmail.com

## Abstract

Mixture-of-Experts (MoE) large language models reduce computation through sparse activation but still incur substantial storage costs. Mixed-precision post-training quantization (PTQ) reduces these costs by quantizing different model components at different bit widths, but existing methods often struggle to preserve generation quality under tight bit budgets. We identify two key challenges: (1) nonlinear quantization errors at extremely low precision are underestimated when crosslayer propagation is ignored; (2) direct router fine-tuning may cause routing drift, and calibration gains may not generalize to downstream tasks. To address these issues, we propose ProQuant, a weight-only PTQ framework for MoE models. Prune-or-Prioritize Mixed-Precision Allocation (PoP) jointly optimizes expert pruning and bit-width assignment. It allows higher-precision experts to substitute for severely degraded ones while improving bit-budget utilization. Routing-Consistent RM-SNorm Finetune (RoCo) adapts expert input representations through mathematically equivalent transformations. It compensates for quantization errors to reduce routing drift. Extensive experiments demonstrate that ProQuant outperforms existing methods across diverse generative tasks. Code is available at https: //github.com/ThrowBug/ProQuant.git.

## 1 Introduction

Large language models (LLMs) (Team et al., 2026; Zeng et al., 2026; Chen et al., 2026; Blakeman et al., 2026; Agarwal et al., 2025) have demonstrated remarkable capabilities in natural language understanding and complex reasoning, while scaling model capacity remains an important path toward further performance improvements. Mixtureof-Experts (MoE) architectures (Yang et al., 2025; Liu et al., 2024a; Xu et al., 2026a) increase model capacity by activating only a small subset of experts for each token, maintaining relatively low active parameters and computation. However, sparse activation does not reduce the need to store the full set of expert weights. As MoE models continue to scale, the resulting storage, memory, and weight-movement costs increasingly become major bottlenecks for efficient deployment.

![](images/a7f7cbffed3c493b51119fc0d41aa547598fc093fec2af7313b4bda713d786f0.jpg)

![](images/2abe281fbf20c2d66c347dfdb25dc4bc1899c1c9d63fc2d994ae87eac883510b.jpg)  
Figure 1: Performance comparison of MoE mixedprecision quantization methods on Qwen3-30B-A3B-Instruct-2507 under 2.5-bit average expert quantization.

To address these deployment challenges, weightonly post-training quantization (PTQ) (Frantar et al., 2023; Lin et al., 2024b; Dettmers et al., 2024; Kim et al., 2024; Tseng et al., 2024) provides an efficient solution for compressing largescale MoE models. Since autoregressive decoding is typically memory-bandwidth bound, reducing weight precision lowers both storage requirements and weight transfer costs during inference. Compared with quantization-aware training (QAT) (Liu et al., 2024b; Chen et al., 2025a; Du et al., 2024; Liu et al., 2025b; Panferov et al., 2025), PTQ requires no large-scale retraining and only a small amount of calibration data, making it particularly suitable for efficient MoE deployment.

Existing MoE quantization studies (Qin et al., 2026; Chen et al., 2025c) mainly improve model performance through mixed-precision allocation (Huang et al., 2025b; Duanmu et al., 2025; Xie et al., 2025; Deng et al., 2026) and post-quantization fine-tuning (Deng et al., 2026; Chen et al., 2025b). The former assigns different bit-widths to experts based on expert sensitivity (Chitty-Venkata et al., 2025; Xie et al., 2025), routing information (Huang et al., 2025b; Duanmu et al., 2025), or quantization error (Deng et al., 2026; Duanmu et al., 2025; Qi et al., 2026), while the latter adapts to the post-quantization distribution by updating the remaining full-precision parameters. However, these approaches still face two key challenges. First, existing mixed-precision allocation strategies typically rely on manually designed importance metrics and consecutive bitwidth search spaces. Although these metrics can capture local quantization errors, our statistical experiments demonstrate that they may still underestimate how extremely low-bit quantization affects model performance (see Fig. 2). Second, although existing router fine-tuning methods can reduce perplexity on calibration data, such improvements do not consistently translate into performance gains on real-world downstream generation tasks.

![](images/ba6ba5437150e2ee63cae2c88fb4316f2392446451fd0c23ceed3ef8156784df.jpg)  
Figure 2: Relative hidden-state MSE at each decoder layer, measured against the corresponding full-precision model outputs. All experts in layer 0 are quantized to 1–4 bits, while those in other layers use 2 bits.

To overcome these challenges, we propose Pro-Quant, an efficient PTQ framework for largescale MoE models. First, we introduce Prune-or-Prioritize Mixed-Precision Allocation (PoP), which jointly determines expert pruning and bit-width assignment within a unified optimization. PoP allows low-contribution experts to be removed while prioritizing sufficient precision for more valuable experts, thereby avoiding severe representation degradation caused by extremely low bit-widths. Second, we propose Routing-Consistent RMSNorm Finetune (RoCo), which compensates for expert quantization errors through mathematically equivalent transformations while preserving the original routing behavior. As illustrated in Fig. 1, ProQuant consistently outperforms all competing methods across all six benchmarks, maintaining strong capabilities even on tasks where other methods suffer substantial degradation.

Our contributions are summarized as follows:

• We propose ProQuant, a weight-only PTQ framework for large-scale MoE models that integrates mixed-precision allocation with lightweight post-quantization compensation.

• We design PoP, which adopts a prune-orprioritize joint decision to avoid severe degradation from extremely low-precision experts and better utilize limited bit budgets.

• We develop RoCo, which decouples representation adaptation from routing behavior and compensates for quantization errors while preserving routing consistency. This avoids routing drift caused by directly fine-tuning fullprecision parameters.

• Experiments on Qwen3-30B-A3B-Instruct-2507 and Qwen3.5-35B-A3B validate Pro-Quant on diverse generative tasks at average expert precisions of 1.5, 2.0, and 2.5 bits.

## 2 Related Works

## 2.1 Post-Training Quantization

Post-training quantization (PTQ) directly converts pretrained models into low-precision representations without costly retraining. A representative direction is second-order error compensation, where GPTQ (Frantar et al., 2023) leverages Hessian information to compensate for quantization errors in the remaining weights during sequential columnwise quantization. Building on GPTQ, GPTAQ (Li et al., 2025a) and BoA (Kim et al., 2025) extend second-order error compensation by accounting for inter-layer error accumulation and cross-projection dependencies, respectively. Activation-aware scaling is employed in AWQ (Lin et al., 2024b) to protect salient weight channels, whereas other studies (Dettmers et al., 2024; Lee et al., 2024; Kim et al., 2024; Yuan et al., 2024; Huang et al., 2024, 2025c) exploit differences in weight saliency or quantization sensitivity to adopt differentiated quantization strategies. Many PTQ methods use mathematically equivalent transformations to ease quantization. SmoothQuant (Xiao et al., 2023)

![](images/b31da02d25dc40b91046b8286dbc2d9b5c7c29eb11e003c4ca1efa2adfe71bd4.jpg)  
Figure 3: Overview of ProQuant, illustrating the workflows of (a) Prune-or-Prioritize Mixed-Precision Allocation (PoP) and (b) Routing-Consistent RMSNorm Finetune (RoCo).

adopts channel-wise scaling, while rotation-based methods (Ashkboos et al., 2024; Liu et al., 2025a; Lin et al., 2024a) redistribute outliers. Other approaches (Ma et al., 2024; Shao et al., 2024; Wei et al., 2023) explore affine transformations and equivalent scaling or shifting operations. Recent studies have extended PTQ to ultra-low-bit regimes, including ternary, binary, and even sub-1-bit weight representations (Yan et al., 2026; Wang et al., 2026; Chong et al., 2026; Li et al., 2025b). However, when applied to MoE architectures, these generalpurpose PTQ methods often underutilize the bit budget, degrading performance.

## 2.2 Quantization of MoE Models

MoE models pose additional quantization challenges due to sparse expert activation. To capture token–expert relevance, MoEQuant (Chen et al., 2025c) and VEQ (Qin et al., 2026) incorporate token–expert affinity into calibration and quantization objectives, with VEQ further introducing modality information for VLMs to account for textual and visual differences. Router-aware methods (Fu et al., 2025; Chen et al., 2025b; Fang and Huang, 2026) explicitly align the quantized routing behavior with its full-precision counterpart to preserve expert selection. Expert heterogeneity also motivates mixed-precision quantization. Sensitivity-aware methods (Li et al., 2024; Huang et al., 2025b; Duanmu et al., 2025; Xie et al., 2025) allocate precision according to structural or expert sensitivity. More recent studies develop principled sensitivity metrics (Chowdhury et al., 2026) or perform global expert-level bit allocation with router fine-tuning (Deng et al., 2026). Beyond mixedprecision allocation, other works directly improve low-bit representations or compensate for quantization errors. MiLo (Huang et al., 2025a) reconstructs quantization residuals using adaptive lowrank compensators. CodeQuant (Yin et al., 2026) combines rotation and clustering to reduce quantization errors, whereas KBVQ-MoE (Xu et al., 2026b) removes cross-expert redundancy and applies bias correction. Our method prunes less important experts to prioritize precision for others and preserves routing via router-decoupled RMSNorm fine-tuning.

## 3 Method

## 3.1 Preliminaries

Mixture-of-Experts. MoE-based large language models (Liu et al., 2024a; Xu et al., 2026a; Yang et al., 2025) replace each FFN layer of a dense model with an MoE layer consisting of a router and multiple parallel experts, such that only a subset of experts is activated during forward propagation. Given the input hidden state z of the t-th token to the l-th MoE layer, the router computes routing scores over all experts and selects the top-K experts for computation. The output of the MoE layer

can be formulated as:

$$
\mathbf { y } = \sum _ { i \in \mathrm { T o p } - K } g _ { i } ( \mathbf { z } ) E _ { i } ( \mathbf { z } ) ,\tag{1}
$$

where $E _ { i }$ denotes the i-th expert, and Top-K is the index set of top-K experts with the highest routing scores. The routing weight of expert i is obtained by renormalizing its routing probability within Top-K:

$$
g _ { i } ( \mathbf { z } ) = \frac { \pi _ { i } ( \mathbf { z } ) } { \sum _ { j \in \mathrm { T o p } - K } \pi _ { j } ( \mathbf { z } ) } ,\tag{2}
$$

where $\pi _ { i } ( \mathbf { z } )$ denotes the routing probability assigned to expert i.

Weight-Only Quantization. Asymmetric quantization maps floating-point weights to low-bit integer representations, with a scaling factor and a zero point stored for each weight group. Given a weight matrix $\mathbf { W } \in \mathbb { R } ^ { C _ { \mathrm { { o u t } } } \times C _ { \mathrm { { i n } } } }$ , each row is partitioned into multiple groups. For each group $\mathbf { w } \in \mathbb { R } ^ { 1 \times c }$ its low-bit integer representation is obtained via the round-to-nearest rule as follows:

$$
\mathbf { q } = \mathrm { c l a m p } ( \lfloor \frac { \mathbf { w } } { s } \rceil + z , q _ { \mathrm { m i n } } , q _ { \mathrm { m a x } } ) ,\tag{3}
$$

where s and z denote the scaling factor and zero point, respectively, $[ q _ { \mathrm { m i n } } , q _ { \mathrm { m a x } } ]$ denotes the representable range of the quantized integer, and $\lfloor \cdot \rceil$ denotes the rounding operation. The corresponding quantized approximation of the original weights can be expressed as:

$$
\hat { \mathbf { w } } = s ( \mathbf { q } - z ) .\tag{4}
$$

GPTQ (Frantar et al., 2023) has demonstrated relatively robust performance under low-bit quantization. Therefore, once the target bit-width of a weight matrix is determined, we adopt it for quantization. Specifically, GPTQ minimizes the following layer-wise reconstruction loss:

$$
\operatorname* { m i n } { \| \mathbf { W X } - \hat { \mathbf { W } } \mathbf { X } \| _ { F } ^ { 2 } } ,\tag{5}
$$

where X denotes the calibration activations and W<sup>ˆ</sup> the quantized weights. It uses the Hessian approximation $\mathbf { H } = 2 \mathbf { X } \mathbf { X } ^ { \top }$ to guide quantization error compensation, helping preserve reconstruction quality at low precision.

## 3.2 Prune-or-Prioritize Mixed-Precision Allocation

To quantify the quantization error of an expert, we introduce the definition of expert quantization cost. As MoE outputs are weighted by routing scores, we incorporate their renormalized values into its definition:

$$
C _ { l , i , b } = \frac { \sum _ { t \in A _ { l , i } } g _ { l , i } ( \mathbf { z } _ { t , l } ^ { B } ) \| E _ { l , i } ( \mathbf { z } _ { t , l } ^ { B } ) - \hat { E } _ { l , i } ^ { b } ( \mathbf { z } _ { t , l } ^ { B } ) \| _ { 2 } } { | A _ { l , i } | } ,\tag{6}
$$

![](images/221dc38c6c95072c1b7d706c799d914c40bf1615915a29cb1124020e25964202.jpg)  
Figure 4: Top-32 expert Jaccard similarity across MoE layers in Qwen3-30B-A3B-Instruct-2507, comparing C4 with MATH-500 and GPQA-Diamond using activation-count and mean-routing-score rankings.

where $C _ { l , i , b }$ represents the quantization cost of expert (l, i) at b bits, with l and i indexing the MoE layer and the expert within that layer. $E _ { l , i }$ and $\hat { E } _ { l , i } ^ { b }$ correspond to the full-precision expert and its b-bit quantized counterpart. $\mathbf { \sigma } _ { \mathbf { z } _ { t , l } ^ { B } }$ is obtained by propagating token t through the preceding layers quantized to an average bit-width of B. $A _ { l , i }$ contains the indices of tokens routed to expert i in layer l. As shown in Fig. 4, we compare the similarity of the top-32 experts ranked by activation count and mean routing score between C4 and two downstream benchmarks. Mean routing score yields more consistent expert selections across datasets, suggesting that it better reflects stable expert importance. We therefore use routing-score-weighted quantization error and normalize it by $| A _ { l , i } |$ to remove the effect of activation frequency.

As shown in Fig. 2, we quantize all experts in layer 0 to 1, 2, 3, or 4 bits, while keeping subsequent layers at 2 bits, to investigate crosslayer error propagation. Compared with other bitwidths, the errors introduced by 1-bit quantization cannot be effectively compensated for by subsequent higher-precision layers. This suggests that extremely low-bit quantization can cause persistent degradation in hidden representations across layers. However, some existing methods (Huang et al., 2025b; Duanmu et al., 2025) estimate quantization sensitivity using local output errors, potentially underestimating such cross-layer effects and the actual impact of extremely low precision. Based on this observation, we exclude 1-bit quantization from the candidate bit-widths under average expert bit-width budgets of 1.5, 2.0, and 2.5 bits. Instead, we introduce a 0-bit pruning option, whose expert quantization cost is defined as follows:

$$
C _ { l , i , 0 } = \frac { \sum _ { t \in A _ { l , i } } g _ { l , i } ( \mathbf { z } _ { t , l } ^ { B } ) \| E _ { l , i } ( \mathbf { z } _ { t , l } ^ { B } ) \| _ { 2 } } { | A _ { l , i } | } ,\tag{7}
$$

where 0-bit means pruning the expert. Replacing

1-bit with 0-bit brings two benefits: (1) it frees additional bit budget for more important experts, allowing them to retain higher precision; (2) it removes severely compressed experts with limited representational capacity and allows higher-precision experts to take their place, which better exploits the sparse activation property of MoE models.

Based on the above definitions, we formulate the bit-width allocation as the following integer linear programming (ILP) problem:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { \eta } \quad \displaystyle \sum _ { l = 1 } ^ { L } \displaystyle \sum _ { i = 1 } ^ { L } \sum _ { b \in \mathbb { R } } C _ { l , i , b } \eta _ { i , b } } \\ { \mathrm { s . t . } \quad \eta _ { i , b } \in \{ 0 , 1 \} , \forall l , i , b } \\ { \displaystyle \sum _ { b \in B } \eta _ { i , b } = 1 , \forall l , i } \\ { \displaystyle \sum _ { i = 1 } ^ { L } \eta _ { i , 0 } \le \operatorname* { m i n } \{ K - K , \rho E \} , \forall l } \\ { \displaystyle \sum _ { l = 1 } ^ { L } \sum _ { i = 1 } ^ { L } \sum _ { b \in B } \eta _ { i , i , b } \le B L E , } \end{array}\tag{8}
$$

where $\eta _ { l , i , b }$ indicates whether expert $( l , i )$ is quantized to b bits, and $\boldsymbol { B }$ denotes the set of candidate bit-widths, which is set to {0, 2, 3} in our settings. To avoid excessive pruning, we further bound the number of 0-bit experts by the maximum ratio $\rho .$ Solving this ILP jointly determines expert pruning and the bit-widths of retained experts.

## 3.3 Routing-Consistent RMSNorm Finetune

Prior studies either calibrate the router through routing alignment (Fu et al., 2025; Chen et al., 2025b), or adapt it after quantization using a global output-level objective (Deng et al., 2026). However, we find that neither strategy yields consistent improvements on downstream tasks. Routing alignment constrains the quantized model to match the routing decisions of its full-precision counterpart. However, such local alignment lacks global information about the quantized model. Therefore, it cannot effectively adjust routing to compensate for quantization errors accumulated across layers. In contrast, directly optimizing the router with a global output-level objective can adapt routing according to the quantized model. However, fitting the router to the calibration set may bias it toward experts that are more frequently activated by the calibration data. Inspired by (Xiao et al., 2023; Lin et al., 2024b), we propose an RMSNorm finetuning method based on mathematically equivalent transformations. It adapts hidden representations for quantization-error compensation while preserving the original routing behavior.

![](images/0f646e7938dcb335834c94cb239a54a433ca66f893e4a218579c9a686f3c35c6.jpg)  
Figure 5: Architecture of $\mathbf { R o C o } .$ (a) Fine-tuning process and (b) parameter folding process. Superscripts i and $p$ denote “in” and “post”, respectively.

Given an input hidden state $\mathbf { x } \in \mathbb { R } ^ { 1 \times d }$ to the RMSNorm layer following the attention module, its output is given by

$$
\mathbf { z } = \gamma \odot \mathrm { N o r m } ( \mathbf { x } ) ,\tag{9}
$$

where $\boldsymbol { \gamma } \in \mathbb { R } ^ { 1 \times d }$ denotes the learnable scaling parameter, and Norm $\begin{array} { r } { ( { \bf x } ) = \frac { { \bf x } } { \sqrt { \frac { 1 } { d } { \bf x } { \bf x } ^ { \top } + \epsilon } } , } \end{array}$

We introduce $\pmb { \delta } \in \mathbb { R } ^ { 1 \times d }$ as the optimization variable and parameterize a positive scaling vector as ${ \bf s } = \exp ( \delta )$ . The scaling vector is applied to the learnable RMSNorm parameter $\gamma$ , while the corresponding input channels of the router weight matrix are inversely scaled to preserve the routing outputs: $\widetilde { \gamma } = \gamma \odot \mathbf { s } , \ \widetilde { \mathbf { W } } _ { \mathrm { r o u t e r } } = \mathbf { W } _ { \mathrm { r o u t e r } } \odot \big ( \mathbf { 1 } ^ { \top } \mathbf { s } ^ { - 1 } \big ) .$ , (10) where $\odot$ denotes element-wise multiplication, $\mathbf { 1 \in }$ $\mathbb { R } ^ { 1 \times C _ { \mathrm { o u t } } }$ is an all-ones vector, and $\mathbf { s } ^ { - 1 }$ denotes the element-wise reciprocal of s. After applying the scaling, the RMSNorm output becomes $\widetilde { \mathbf z } = \widetilde { \gamma } \odot$ $\mathrm { N o r m } ( \mathbf { x } ) = \mathbf { z } \odot \mathbf { s }$ . The inverse scaling on $\mathbf { W _ { \mathrm { r o u t e r } } }$ exactly cancels this change when computing the router logits:

$$
\widetilde \mathbf { z } \widetilde { \mathbf { W } } _ { \mathrm { r o u t e r } } ^ { \top } = \mathbf { z } \mathbf { W } _ { \mathrm { r o u t e r } } ^ { \top } .\tag{11}
$$

Therefore, the equivalent transformation allows RMSNorm fine-tuning to compensate for expert quantization errors while preserving the original routing behavior, thereby avoiding routing flips induced by RMSNorm updates. We optimize the RMSNorm with the following KL divergence:

$$
\mathcal { L } = D _ { \mathrm { K L } } ( \mathbf { p } _ { \mathrm { F P } } \Vert \mathbf { p } _ { \mathrm { Q } } )\tag{12}
$$

where $\mathbf { p } _ { \mathrm { F P } }$ and $\mathbf { p } _ { \mathrm { Q } }$ denote the output probability distributions of the full-precision and quantized models, respectively. Since only a small number of RMSNorm weights are updated, the fine-tuning is highly lightweight. Moreover, it consistently outperforms direct fine-tuning without inverse scaling of the router weights, as shown in ablation study. After this step, the learned scaling vector is absorbed into the RMSNorm and router weights, introducing no additional overhead during inference. The process is illustrated in Fig. 5.

![](images/f22cba3843f4607266d65c060fbaff72b2327c164c9ee3d0e10c75b62a33227d.jpg)  
Figure 6: Layer-wise Jaccard similarity between the full-precision and quantized models’ top-32 experts ranked by mean routing score on three datasets.

Fig. 6 reports the Jaccard similarity of top-32 expert sets ranked by mean routing score between the full-precision model and the same quantized model under three strategies. RoCo achieves higher Jaccard similarity across most MoE layers, indicating that it better preserves the expert-selection preference of the full-precision model for the target task. By maintaining the original routing behavior, RoCo allows the quantized model to continue prioritizing experts favored by the full-precision model, while adapting hidden representations to compensate for quantization errors. RoCo stabilizes routing and strengthens experts.

## 4 Experiments

## 4.1 Settings

Implementation Details. All experiments are conducted using PyTorch (Paszke et al., 2019) and HuggingFace Transformers (Wolf et al., 2020) on two NVIDIA A800 GPUs. We evaluate our method on Qwen3-30B-A3B-Instruct-2507 (Yang et al., 2025) and Qwen3.5-35B-A3B (Qwen Team, 2026). We use 128 sequences of 2,048 tokens sampled from C4 (Raffel et al., 2020) with seed 0 as the calibration set. For both softmax and linear attention layers, the Q, K, V, and O projection weights are uniformly quantized to 4 bits, while all other linear-attention weights and router weights remain in full precision. We consider average expert bitwidths of 2.5, 2.0, and 1.5 bits with a candidate set of 0, 2, 3 bits, and quantize the shared experts of Qwen3.5 to 4 bits. See Appendix B for details.

Models and Evaluation. We observe that the conventional zero-shot evaluation with lm-eval (Biderman et al., 2024), which is widely adopted in prior work, may not fully reflect model capability on practical generative tasks. Therefore, we primarily use EvalScope (Team, 2024) to evaluate generation-based performance across diverse capabilities, such as mathematical reasoning, scientific reasoning, and instruction following, on six generative benchmarks: GPQA-Diamond (Rein et al., 2024), MATH-500 (Hendrycks et al., 2021; Lightman et al., 2024), IFEval (Zhou et al., 2023), GSM8K (Cobbe et al., 2021), MMLU-Redux (Gema et al., 2025) and BFCL-v3 (Patil et al., 2025). We also report perplexity and zeroshot results as auxiliary metrics.

Baselines. We compare ProQuant with representative mixed-precision quantization methods for MoE LLMs, including PMQ from MC-MoE (Huang et al., 2025b), MxMoE (Duanmu et al., 2025), and GEMQ (Deng et al., 2026). For fair comparison, all methods use GPTQ (Frantar et al., 2023) with their respective bit-width allocation or post-quantization compensation strategies.

## 4.2 Main Results

Generation-based Evaluation. We first evaluate the performance of quantized models on realistic generation tasks. Table 1 shows that ProQuant consistently achieves the best or competitive performance across different models and bit-width budgets, with more pronounced advantages under low-bit settings. As the expert average bit-width decreases from 2.5 bits to 1.5 bits, existing methods suffer from substantial performance degradation, while ProQuant maintains strong generation capability. For example, under the 1.5-bit setting, Pro-

<table><tr><td>Model</td><td>Bits</td><td>Method</td><td>GSM8K</td><td>MATH-500</td><td>GPQA-D</td><td>IFEval</td><td>MMLU-R</td><td>BFCL-v3</td><td>AVG</td></tr><tr><td rowspan="12">Qwen3-30B- A3B- Instruct- 2507</td><td>16.0</td><td>BF16</td><td>96.36</td><td>95.40</td><td>55.05</td><td>87.66</td><td>88.18</td><td>63.02</td><td>80.94</td></tr><tr><td rowspan="4">2.5</td><td>MxMoE PMQ</td><td>93.86 95.45</td><td>51.80</td><td>36.36 30.81</td><td>81.17 82.33</td><td>81.26 79.56</td><td>30.01 29.82</td><td>62.41</td></tr><tr><td>GEMQ</td><td>93.93</td><td>39.80</td><td></td><td></td><td></td><td></td><td>59.63</td></tr><tr><td></td><td></td><td>45.60</td><td>31.31</td><td>82.66</td><td>81.56</td><td>30.22</td><td>60.88</td></tr><tr><td>ProQuant</td><td>95.53</td><td>93.20</td><td>53.03</td><td>87.19</td><td>85.44</td><td>56.31</td><td>78.45</td></tr><tr><td rowspan="5">2.0</td><td>GPTQ MxMoE</td><td>91.28 81.43</td><td>76.00 15.60</td><td>36.36</td><td>81.17 75.80</td><td>81.75</td><td>45.29 30.11</td><td>68.64</td></tr><tr><td>PMQ</td><td>84.61</td><td>12.80</td><td>21.72</td><td></td><td>74.65</td><td></td><td>49.89</td></tr><tr><td></td><td>84.31</td><td></td><td>24.24</td><td>75.03</td><td>71.44</td><td>29.75</td><td>49.65</td></tr><tr><td>GEMQ ProQuant</td><td>94.47</td><td>18.20</td><td>24.24</td><td>73.95</td><td>71.75</td><td>30.08</td><td>50.42</td></tr><tr><td>MxMoE</td><td>5.38</td><td>90.20 3.60</td><td>45.96 13.64</td><td>84.47 39.93</td><td>79.82</td><td>52.05</td><td>74.49</td></tr><tr><td rowspan="3">1.5</td><td>PMQ</td><td>1.44</td><td>1.40</td><td>19.19</td><td>36.51</td><td>55.96 44.26</td><td>29.61 29.73</td><td>24.69 22.09</td></tr><tr><td>GEMQ</td><td>29.72</td><td>2.80</td><td>22.22</td><td>46.90</td><td>53.14</td><td>29.73</td><td></td></tr><tr><td>ProQuant</td><td>93.93</td><td></td><td></td><td></td><td></td><td></td><td>30.75</td></tr><tr><td rowspan="11">Qwen3.5- 35B-A3B</td><td rowspan="4">16.0</td><td>BF16</td><td></td><td>86.00</td><td>32.83</td><td>77.07</td><td>67.26</td><td>46.69</td><td>67.30</td></tr><tr><td>MxMoE</td><td>97.04</td><td>93.60</td><td>81.82</td><td>90.18</td><td>92.39</td><td>68.87</td><td>87.32</td></tr><tr><td>PMQ</td><td>96.36 96.29</td><td>81.80</td><td>56.06</td><td>87.95</td><td>90.04</td><td>63.13</td><td>79.22</td></tr><tr><td>GEMQ</td><td>95.68</td><td>71.20</td><td>39.39</td><td>86.82</td><td>86.53</td><td>56.80</td><td>72.84</td></tr><tr><td>ProQuant</td><td>96.89</td><td>67.40</td><td>45.45</td><td>85.51</td><td>87.77</td><td>45.23</td><td>71.17</td></tr><tr><td rowspan="5">2.0</td><td>GPTQ</td><td>93.10</td><td>93.60</td><td>70.20</td><td>88.03</td><td>91.33</td><td>60.75</td><td>83.47</td></tr><tr><td>MxMoE</td><td>85.60</td><td>83.00</td><td>53.54</td><td>82.83</td><td>88.44</td><td>59.79</td><td>76.78</td></tr><tr><td>PMQ</td><td>94.31</td><td>47.20</td><td>27.78</td><td>79.22</td><td>82.30</td><td>40.79</td><td>60.48</td></tr><tr><td>GEMQ</td><td>72.40</td><td>42.60</td><td>26.26</td><td>77.01</td><td>77.09</td><td>45.12</td><td>60.40</td></tr><tr><td>ProQuant</td><td></td><td>29.80</td><td>24.24</td><td>75.37</td><td>78.91</td><td>48.13</td><td>54.81</td></tr><tr><td rowspan="4">1.5</td><td>MxMoE</td><td>95.53</td><td>82.60</td><td>57.07</td><td>87.76</td><td>89.05</td><td>57.66</td><td>78.28</td></tr><tr><td>PMQ</td><td>69.29</td><td>17.60</td><td>22.73</td><td>47.05</td><td>69.28</td><td>22.57</td><td>41.42</td></tr><tr><td>GEMQ</td><td>22.29</td><td>6.60</td><td>14.65</td><td>48.46</td><td>58.54</td><td>17.96</td><td>28.08</td></tr><tr><td>ProQuant</td><td>36.92 94.84</td><td>11.80 40.20</td><td>27.27 34.85</td><td>55.46 81.16</td><td>64.88 81.72</td><td>12.03 32.31</td><td>34.73 60.85</td></tr></table>

Table 1: Performance of quantized Qwen3-30B-A3B-Instruct-2507 and Qwen3.5-35B-A3B on six generation-based benchmarks. We compare ProQuant with representative mixed-precision methods for MoE LLMs at varying average expert bit-widths. GPQA-D and MMLU-R denote GPQA-Diamond and MMLU-Redux, respectively.
<table><tr><td>Model</td><td>Bits</td><td>Method</td><td>Wiki. PPL↓</td><td>C4 PPL↓</td><td>PIQA↑</td><td>ARC-E↑</td><td>ARC-C↑</td><td>HS↑</td><td>WING↑</td><td>AVG↑</td></tr><tr><td rowspan="4">Qwen3-30B-A3B- Instruct-2507</td><td rowspan="4">16.0</td><td rowspan="4">BF16 GPTQ</td><td>7.26</td><td>12.49</td><td>80.09</td><td>83.88</td><td>62.37</td><td>79.72</td><td>73.40</td><td>75.89</td></tr><tr><td>9.83</td><td>15.45</td><td>78.29</td><td>80.18</td><td>56.83</td><td>73.70</td><td>71.74</td><td>72.15</td></tr><tr><td rowspan="2">MxMoE 9.23</td><td rowspan="2">13.97</td><td rowspan="2">79.54</td><td rowspan="2">79.21</td><td rowspan="2">55.20</td><td rowspan="2">77.84</td><td rowspan="2">72.30</td><td rowspan="2">72.82</td></tr><tr><td>10.61</td></tr><tr><td rowspan="4"></td><td rowspan="4">2.0</td><td colspan="2">PMQ</td><td>14.84</td><td>79.71</td><td>78.87</td><td>54.69</td><td>77.73</td><td>72.93</td><td>72.79</td></tr><tr><td rowspan="3">GEMQ ProQuant</td><td>8.22</td><td>13.49</td><td>79.65</td><td>77.99</td><td>56.31</td><td>76.97</td><td></td><td>72.79</td></tr><tr><td></td><td>8.94 14.57</td><td>79.22</td><td>79.08</td><td>57.17</td><td>75.19</td><td>72.61 70.64</td><td>72.26</td></tr><tr><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td></tr></table>

Table 2: Perplexity and 0-shot log-likelihood accuracy on Qwen3-30B-A3B-Instruct-2507. Wiki., ARC-E, ARC-C, HS, and WING denote WikiText-2, ARC-Easy, ARC-Challenge, HellaSwag, and WinoGrande, respectively.

Quant achieves average scores of 67.30 and 60.85 on Qwen3-30B-A3B-Instruct-2507 and Qwen3.5- 35B-A3B, respectively, significantly outperforming existing mixed-precision quantization methods. These results demonstrate that ProQuant can better preserve the capabilities of MoE LLMs under extremely constrained bit budgets.

Perplexity and Log-Likelihood Evaluation. Furthermore, following previous works, we report perplexity on WikiText-2 and C4, and conduct zeroshot likelihood-based evaluation using lm-eval in Table 2. The results show that ProQuant improves performance on realistic generation tasks while maintaining comparable results on conventional likelihood-based benchmarks. However, when considered together with Table 1, these results suggest that strong performance on PPL and zeroshot likelihood-based metrics does not necessarily translate into better generation ability. In particular, some mixed-precision quantization methods achieve favorable results on traditional likelihoodbased evaluations, yet perform worse than a uniformly quantized 2.0-bit baseline on generationbased tasks. This indicates that token-level likelihood estimation, which underlies these conventional metrics, cannot fully reflect the practical capabilities of quantized models in open-ended generation scenarios and may lead to biased conclusions about the effectiveness of MoE quantization methods. Therefore, we argue that generation-based evaluation is essential for a comprehensive assessment of MoE quantization models.

## 4.3 Ablation Study

Effect of PoP. PoP adopts a fixed candidate bit-width set of {0, 2, 3}. To validate this design choice, we construct two ablation variants:

<table><tr><td>Candidate Bits</td><td>GSM8K</td><td>MATH-500</td><td>GPQA-D</td><td>IFEval</td><td>MMLU-R</td><td>BFCL-v3</td><td>AVG</td></tr><tr><td>{0, 2, 3}</td><td>94.47</td><td>90.20</td><td>45.96</td><td>84.47</td><td>79.82</td><td>52.05</td><td>74.49</td></tr><tr><td>{1, 2, 3}</td><td>93.33</td><td>85.80</td><td>36.87</td><td>80.96</td><td>71.40</td><td>42.40</td><td>68.46</td></tr><tr><td>{0, 1, 2, 3}</td><td>93.93</td><td>85.40</td><td>36.36</td><td>77.88</td><td>69.44</td><td>38.87</td><td>66.98</td></tr></table>

Table 3: Ablation of candidate bit-width sets on Qwen3-30B-A3B-Instruct-2507 at a 2.0-bit average expert precision.

<table><tr><td>Method</td><td>GSM8K</td><td>MATH-500</td><td>GPQA-D</td><td>IFEval</td><td>MMLU-R</td><td>BFCL-v3</td><td>AVG</td></tr><tr><td>ProQuant</td><td>93.93</td><td>86.00</td><td>32.83</td><td>77.07</td><td>67.26</td><td>46.69</td><td>67.30</td></tr><tr><td>w/o RoCo</td><td>94.16</td><td>85.80</td><td>26.77</td><td>75.35</td><td>63.75</td><td>42.79</td><td>64.77</td></tr><tr><td>Direct RMSNorm FT</td><td>93.33</td><td>86.60</td><td>30.81</td><td>74.96</td><td>65.88</td><td>43.89</td><td>65.91</td></tr></table>

Table 4: Fine-tuning strategy ablation on Qwen3-30B-A3B-Instruct-2507 at a 1.5-bit average expert precision.

<table><tr><td>Model</td><td>Bits</td><td>ρ</td><td>GSM8K</td><td>MATH-500</td><td>GPQA-D</td><td>IFEval</td><td>MMLU-R</td><td>BFCL-v3</td><td>AVG</td></tr><tr><td rowspan="4">Qwen3-30B- A3B- Instruct-</td><td rowspan="5">2.0</td><td>0.10</td><td>95.00</td><td>88.00</td><td>48.99</td><td>82.10</td><td>82.56</td><td>49.71</td><td>74.39</td></tr><tr><td>0.15</td><td>94.69</td><td>88.20</td><td>42.93</td><td>84.21</td><td>81.30</td><td>52.66</td><td>74.00</td></tr><tr><td>0.20</td><td>95.22</td><td>87.80</td><td>44.44</td><td>84.92</td><td>80.12</td><td>51.67</td><td>74.03</td></tr><tr><td>0.25</td><td>95.45</td><td>91.80</td><td>46.46</td><td>82.57</td><td>79.49</td><td>54.38</td><td>75.03</td></tr><tr><td>2507</td><td>0.30</td><td>94.24</td><td>90.60</td><td>45.45</td><td>83.97</td><td>79.00</td><td>51.04</td><td>74.05</td></tr></table>

Table 5: Sensitivity to ρ on Qwen3-30B-A3B-Instruct-2507 with 2.0-bit experts, without RoCo finetuning.

{1, 2, 3} removes the 0-bit option and forces all experts to be retained, while {0, 1, 2, 3} additionally introduces 1-bit quantization. The full PoP setting achieves the highest average score of 74.07, compared with 68.46 and 66.98 for the two variants, respectively. Notably, introducing the 0-bit option does not make {0, 1, 2, 3} outperform {1, 2, 3}, indicating that the gain does not come from pruning itself, but from avoiding severely quantized 1-bit experts that continue to participate in the forward pass. Thanks to the sparse routing mechanism of MoE models, a removed expert can be replaced by a slightly lower-ranked expert that retains higher precision. PoP prunes experts impaired by extremely low precision and reallocates their bit budget to preserve others’ effectiveness.

Effect of RoCo. We evaluate the effectiveness of RoCo by comparing the full ProQuant framework with two ablation variants: w/o RoCo, which completely removes RoCo, and Direct RMSNorm FT, which directly fine-tunes RMSNorm without preserving routing consistency. The average score drops from 67.30 to 64.77 without RoCo, indicating that lightweight RMSNorm fine-tuning can effectively mitigate quantization-induced performance degradation. Direct RMSNorm FT improves the average score to 65.91, but still underperforms the full ProQuant framework. This suggests that directly updating RMSNorm can provide partial recovery, but may also alter the router input and disturb the original routing behavior. RoCo decouples representation adaptation from routing, correcting quantization errors while preserving expert selection and improving recovery on generative tasks.

Sensitivity to the Pruning Ratio. Since PoP and RoCo are decoupled, we analyze the effect of the maximum pruning ratio $\rho$ without RoCo finetuning, so as to isolate the sensitivity of PoP to this hyperparameter. On Qwen3-30B-A3B-Instruct-2507, the average performance varies only slightly as ρ ranges from 0.10 to 0.30, indicating that PoP is relatively robust to ρ within this range on this model. When ρ exceeds 0.30, the constraint is no longer active, and further increasing it does not affect the final allocation. The results also show that a moderate maximum pruning ratio is generally favorable for achieving higher average accuracy.

## 5 Conclusion

In this work, we presented ProQuant, which preserves MoE LLM performance on downstream generation tasks. More importantly, our study offers two insights into MoE quantization. First, we advocate a principle for extremely low-bit mixedprecision allocation: “discard it rather than destroy it.” When the bit budget cannot adequately preserve an expert’s functionality, removing it may be preferable to retaining a severely degraded version. Second, we caution against over-optimizing for calibration data or specific metrics and emphasize generalization to downstream tasks. This also calls for reexamining the use of perplexity and loglikelihood metrics: they measure a model’s ability to assign probabilities to reference sequences, rather than to generate text autoregressively based on its own outputs, and cannot replace real-world generation evaluations. We hope these findings encourage MoE quantization research to prioritize model reliability and cross-task generalization.

## Limitations

In this work, we focus on mixed-precision PTQ for large-scale MoE LLMs and evaluate ProQuant on models with up to 35 billion total parameters. Although ProQuant demonstrates strong generation performance at this scale, resource constraints prevented us from extending our experiments to models with hundreds of billions or trillions of parameters. Its effectiveness and scalability at these larger scales therefore remain to be validated.

## References

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. 2025. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925.

Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. 2024. Quarot: Outlier-free 4-bit inference in rotated llms. In NeurIPS.

Stella Biderman, Hailey Schoelkopf, Lintang Sutawika, Leo Gao, Jonathan Tow, Baber Abbasi, Alham Fikri Aji, Pawan Sasanka Ammanamanchi, Sidney Black, Jordan Clive, et al. 2024. Lessons from the trenches on reproducible evaluation of language models. arXiv preprint arXiv:2405.14782.

Aaron Blakeman, Aaron Thomas, Aastha Jhunjhunwala, Abhibha Gupta, Abhinav Khattar, Adam Rajfer, Adi Renduchintala, Adil Asif, Aditya Vavre, Adriana Flores Miranda, et al. 2026. Nemotron 3 ultra: Open, efficient mixture-of-experts hybrid mambatransformer model for agentic reasoning. arXiv preprint arXiv:2606.15007.

Aili Chen, Aonian Li, Baichuan Zhou, Bangwei Gong, Binyang Jiang, Boji Dan, Changhao Zhang, Changqing Yu, Chao Wang, Cheng Ma, et al. 2026. The minimax-m2 series: Mini activations unleashing max real-world intelligence. arXiv preprint arXiv:2605.26494.

Mengzhao Chen, Wenqi Shao, Peng Xu, Jiahao Wang, Peng Gao, Kaipeng Zhang, and Ping Luo. 2025a. Efficientqat: Efficient quantization-aware training for large language models. In ACL.

Yuanteng Chen, Yuantian Shao, Peisong Wang, and Jian Cheng. 2025b. Eac-moe: Expert-selection aware compressor for mixture-of-experts large language models. In ACL.

Zhixuan Chen, Xing Hu, Dawei Yang, Zukang Xu, Xu Chen, Zhihang Yuan, Sifan Zhou, and Jiangyong Yu. 2025c. Moequant: Enhancing quantization for mixture-of-experts large language models via expertbalanced sampling and affinity guidance. In ICML.

Krishna Teja Chitty-Venkata, Jie Ye, and Murali Emani. 2025. Mopeq: Mixture of mixed precision quantized experts. In ICCV workshops.

Hyochan Chong, Dongkyu Kim, Changdong Kim, and Minseop Choi. 2026. Nanoquant: Efficient sub-1-bit quantization of large language models. In ICML.

Mohammed Nowaz Rabbani Chowdhury, Kaoutar El Maghraoui, Hsinyu Tsai, Naigang Wang, Geoffrey Burr, Liu Liu, and Meng Wang. 2026. Efficient quantization of mixture-of-experts with theoretical generalization guarantees. In ICLR.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Jianing Deng, Song Wang, Dongwei Wang, Zijie Liu, Tianlong Chen, Huanrui Yang, and Jingtong Hu. 2026. Gemq: Global expert-level mixed-precision quantization for moe llms. In ICML.

Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, and Dan Alistarh. 2024. Spqr: A sparse-quantized representation for near-lossless llm weight compression. In ICLR.

Dayou Du, Yijia Zhang, Shijie Cao, Jiaqi Guo, Ting Cao, Xiaowen Chu, and Ningyi Xu. 2024. Bitdistiller: Unleashing the potential of sub-4-bit llms via selfdistillation. In ACL.

Haojie Duanmu, Xiuhong Li, Zhihang Yuan, Size Zheng, Jiangfei Duan, Xingcheng Zhang, and Dahua Lin. 2025. Mxmoe: Mixed-precision quantization for moe with accuracy and performance co-design. In ICML.

Yi-Zeng Fang and Juinn-Dar Huang. 2026. Beyond freezing the router: Rank-aligned post-training quantization for mixture-of-experts models. TMLR.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2023. Gptq: Accurate post-training quantization for generative pre-trained transformers. In ICLR.

Zhongqian Fu, Tianyi Zhao, Ning Ding, Xianzhi Yu, Xiaosong Li, Yehui Tang, and Yunhe Wang. 2025. Eaquant: Enhancing post-training quantization for moe models via expert-aware optimization. arXiv preprint arXiv:2506.13329.

Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, et al. 2025. Are we done with mmlu? In NAACL.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical

problem solving with the math dataset. In NeurIPS Datasets and Benchmarks Track.

Beichen Huang, Yueming Yuan, Zelei Shao, and Minjia Zhang. 2025a. Milo: Efficient quantized moe inference with mixture of low-rank compensators. In MLSys.

Wei Huang, Yue Liao, Jianhui Liu, Ruifei He, Haoru Tan, Shiming Zhang, Hongsheng Li, Si Liu, and Xiaojuan Qi. 2025b. Mixture compressor for mixtureof-experts llms gains more. In ICLR.

Wei Huang, Yangdong Liu, Haotong Qin, Ying Li, Shiming Zhang, Xianglong Liu, Michele Magno, and Xiaojuan Qi. 2024. Billm: Pushing the limit of posttraining quantization for llms. In ICML.

Wei Huang, Haotong Qin, Yangdong Liu, Yawei Li, Qinshuo Liu, Xianglong Liu, Luca Benini, Michele Magno, Shiming Zhang, and Xiaojuan Qi. 2025c. Slim-llm: Salience-driven mixed-precision quantization for large language models. In ICML.

Junhan Kim, Ho-young Kim, Eulrang Cho, Chungman Lee, Joonyoung Kim, and Yongkweon Jeon. 2025. Boa: Attention-aware post-training quantization without backpropagation. In ICML.

Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W Mahoney, and Kurt Keutzer. 2024. Squeezellm: Dense-andsparse quantization. In ICML.

Changhun Lee, Jungyu Jin, Taesu Kim, Hyungjun Kim, and Eunhyeok Park. 2024. Owq: Outlier-aware weight quantization for efficient fine-tuning and inference of large language models. In AAAI.

Pingzhi Li, Xiaolong Jin, Zhen Tan, Yu Cheng, and Tianlong Chen. 2024. Quantmoe-bench: Examining post-training quantization for mixture-of-experts. arXiv preprint arXiv:2406.08155.

Yuhang Li, Ruokai Yin, Donghyun Lee, Shiting Xiao, and Priyadarshini Panda. 2025a. Gptaq: Efficient finetuning-free quantization for asymmetric calibration. In ICML.

Zhiteng Li, Xianglong Yan, Tianao Zhang, Haotong Qin, Dong Xie, Jiang Tian, Zhongchao Shi, Linghe Kong, Yulun Zhang, and Xiaokang Yang. 2025b. Arb-llm: Alternating refined binarizations for large language models. In ICLR.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In ICLR.

Haokun Lin, Haobo Xu, Yichen Wu, Jingzhi Cui, Yingtao Zhang, Linzhan Mou, Linqi Song, Zhenan Sun, and Ying Wei. 2024a. Duquant: Distributing outliers via dual transformation makes stronger quantized llms. In NeurIPS.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2024b. Awq: Activation-aware weight quantization for ondevice llm compression and acceleration. In MLSys.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024a. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra. 2024b. Llm-qat: Data-free quantization aware training for large language models. In Findings of ACL.

Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. 2025a. Spinquant: Llm quantization with learned rotations. In ICLR.

Zechun Liu, Changsheng Zhao, Hanxian Huang, Sijia Chen, Jing Zhang, Jiawei Zhao, Scott Roy, Lisa Jin, Yunyang Xiong, Yangyang Shi, et al. 2025b. Paretoq: Improving scaling laws in extremely low-bit llm quantization. In NeurIPS.

Yuexiao Ma, Huixia Li, Xiawu Zheng, Feng Ling, Xuefeng Xiao, Rui Wang, Shilei Wen, Fei Chao, and Rongrong Ji. 2024. Affinequant: Affine transformation quantization for large language models. In ICLR.

Andrei Panferov, Jiale Chen, Soroush Tabesh, Mahdi Nikdan, and Dan Alistarh. 2025. QuEST: Stable training of LLMs with 1-bit weights and activations. In ICML.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS.

Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. 2025. The berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In ICML.

Huachen Qi, Ruiyu Zhuo, Bowen Shi, Xiang Chang, Fei Chao, Changjing Shang, and Qiang Shen. 2026. Profiling-free mixed-precision quantization for MoE LLMs via fuzzy rule interpolation. In ACL.

Guangshuo Qin, Zhiteng Li, Zheng Chen, Weihang Zhang, Linghe Kong, and Yulun Zhang. 2026. Veq: Modality-adaptive quantization for moe visionlanguage models. In ICML.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2024. GPQA: A graduate-level google-proof q&a benchmark. In COLM.

Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Gao Peng, Yu Qiao, and Ping Luo. 2024. Omniquant: Omnidirectionally calibrated quantization for large language models. In ICLR.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Ziwei Chai, Y Charles, HS Che, Cheng Chen, et al. 2026. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276.

ModelScope Team. 2024. EvalScope: Evaluation framework for large models.

Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. 2024. QuIP#: Even better LLM quantization with hadamard incoherence and lattice codebooks. In ICML.

Shigeng Wang, Chao Li, Yangyuxuan Kang, Jiawei Fan, and Anbang Yao. 2026. Cat-q: Cost-efficient and accurate ternary quantization for llms. In ICML.

Xiuying Wei, Yunchen Zhang, Yuhang Li, Xiangguo Zhang, Ruihao Gong, Jinyang Guo, and Xianglong Liu. 2023. Outlier suppression+: Accurate quantization of large language models by equivalent and effective shifting and scaling. In EMNLP.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2020. Transformers: State-of-the-art natural language processing. In EMNLP System Demonstrations.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. 2023. Smoothquant: Accurate and efficient post-training quantization for large language models. In ICML.

Zhanhao Xie, Yuexiao Ma, Xiawu Zheng, Fei Chao, Wanchen Sui, Yong Li, Shen Li, and Rongrong Ji. 2025. Automated fine-grained mixture-of-experts quantization. In Findings of ACL.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. 2026a. Deepseek-v4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348.

Zukang Xu, Zhixiong Zhao, Xing Hu, Zhixuan Chen, and Dawei Yang. 2026b. Kbvq-moe: Klt-guided svd with bias-corrected vector quantization for moe large language models. In ICLR.

Xianglong Yan, ChengZhu Bao, Zhiteng Li, Tianao Zhang, Kaicheng Yang, Haotong Qin, Ruobing Xie, Xingwu Sun, and Yulun Zhang. 2026. PT<sup>2</sup>-LLM: Post-training ternarization for large language models. In ICLR.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Xiangyang Yin, Xingyu Liu, Tianhua Xia, Bo Bao, Vithursan Thangarasa, Valavan Manohararajah, Eric Sather, and Sai Qian Zhang. 2026. Codequant: Unified clustering and quantization for enhanced outlier smoothing in low-precision mixture-of-experts. In ICLR.

Zhihang Yuan, Yuzhang Shang, and Zhen Dong. 2024. Pb-llm: Partially binarized large language models. In ICLR.

Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. 2026. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.

Algorithm 1 Prune-or-Prioritize Mixed-Precision Algorithm 2 Routing-Consistent RMSNorm Fine-  
Allocation tune   
Require: MoE model $\mathcal { M } ,$ calibration set ${ \mathcal { D } } ,$ av- Require: Quantized model $\hat { \mathcal { M } } ,$ , calibration set $\mathcal { D } _ { \mathrm { { : } } }$   
erage bit-width $B ,$ candidate bit-width set $B ,$ learning rate $\alpha ,$ training steps $T$   
maximum pruning ratio $\rho ,$ top- $K$ value $K$ Ensure: Fine-tuned quantized model $\hat { \mathcal { M } }$   
Ensure: Bit-width assignment η 1: Freeze all model weights   
1: $\mathbf { Z } _ { 1 } ^ { B }  \mathbf { G e r T M o E I N P U T } ( \ b { M } , \mathcal { D } )$ 2: Initialize $\delta _ { l } ^ { \mathrm { i n } }  \mathbf { 0 }$ and $\delta _ { l } ^ { \mathrm { p o s t } }  \mathbf { 0 } ,$ ∀l   
2: for $l = 1$ to $L$ do 3: for $t = 1$ to $T$ do   
3: for $i = 1$ to $E$ do 4: for $l = 1$ to L do   
4: Obtain $A _ { l , i }$ and $g _ { l , i }$ using $\mathbf { Z } _ { l } ^ { B }$ 5: $\mathbf { s } _ { l } ^ { \mathrm { i n } } \gets \exp ( \delta _ { l } ^ { \mathrm { i n } } )$   
5: for all $b \in B$ do 6: $\mathbf { s } _ { l } ^ { \mathrm { p o s t } }  \exp ( \delta _ { l } ^ { \mathrm { p o s t } } )$   
6: $\hat { E } _ { l , i } ^ { b } \gets \mathrm { Q U A N T I Z E } ( E _ { l , i } , b )$ 7: end for   
7: $C _ { l , i , b } \gets \mathbf { C a L C C o s T } ( l , i , b , \mathbf { Z } _ { l } ^ { B } )$ 8: Forward $\hat { \mathcal { M } }$ with the RMSNorm scaling   
8: end for vectors and its element-wise reciprocal   
9: end for 9: Compute $\mathcal { L }$   
10: $\hat { \mathcal { M } } _ { l } ^ { B } \gets \mathrm { Q u N T I Z E L A Y E R } ( \mathcal { M } _ { l } , B )$ 10: Update $\{ \delta _ { l } ^ { \mathrm { i n } } , \delta _ { l } ^ { \mathrm { p o s t } } \} _ { l = 1 } ^ { L }$   
11: $\mathbf { Z } _ { l + 1 } ^ { \tilde { B } } \gets \mathrm { L A Y E R F O R W A R D } ( \hat { \mathcal { M } } _ { l } ^ { B } , \mathbf { Z } _ { l } ^ { B } )$ 11: end for   
12: end for 12: for $l = 1$ to $L$ do   
13: $\eta \gets \mathrm { S o L V E I L P } ( C , B , \rho , K , \mathcal { B } , L , E )$ 13: Fuse $\mathbf { s } _ { l } ^ { \mathrm { i n } }$ and $\mathbf { s } _ { l } ^ { \mathrm { p o s t } }$ into the corresponding   
14: return η RMSNorm weights, and $( \mathbf { s } _ { l } ^ { \mathrm { p o s t } } ) ^ { - 1 }$ into the   
router weights   
14: end for

## A ProQuant Pipeline and Algorithms

As shown in Fig. 3, ProQuant consists of three stages. First, as illustrated in Algorithm 1, we traverse the MoE layers of the pretrained model and evaluate the quantization cost of each expert under different candidate bit-widths, thereby determining the final bit-width of each expert under the global bit budget. We then apply GPTQ according to the resulting allocation to obtain the integer representations of the quantized weights, together with the corresponding scales and zero-points. Finally, as shown in Algorithm 2, we freeze all other model parameters and fine-tune the RMSNorm scaling vectors to compensate for quantization errors. After fine-tuning, the learned scaling vectors are fused into the corresponding RMSNorm weights, while their element-wise inverses are fused into the associated router weights, yielding the final ProQuant model.

## B Additional Implementation Details

We randomly sample 128 sequences, each containing 2,048 tokens, from C4 using a fixed random seed of 0. The calibration set is used for expertwise cost estimation prior to bit-width allocation, GPTQ calibration, and RoCo fine-tuning. We solve the ILP problem in Eq. 8 using the HiGHS solver via SciPy. Following GEMQ, we apply GPTQ with a group size of 128 and asymmetric quantization

$$
\hat { \mathcal { M } }
$$

under the final bit-width assignment. The clipping ranges are determined by minimizing the following weight quantization error:

$$
\operatorname* { m i n } _ { l , u } \sum _ { i , j } \left| W _ { i j } - \hat { W } _ { i j } ( l , u ) \right| ^ { 2 . 4 } ,\tag{13}
$$

where l and u denote the lower and upper clipping bounds, respectively, and $\hat { \mathbf { W } } ( l , u )$ denotes the quantized-dequantized weights under the corresponding clipping range.

For Qwen3.5-35B-A3B, the maximum pruning ratio $\rho$ is set to 0.1, 0.15, and 0.3 for average expert bit-widths of 2.5, 2.0, and 1.5 bits, respectively, with a RoCo learning rate of $3 \times 1 0 ^ { - 3 }$ . In contrast, Qwen3-30B-A3B-Instruct-2507 is less sensitive to the maximum pruning ratio, maintaining strong performance even at $\rho = 1 . 0 $ , which imposes no additional upper bound on the pruning ratio. We therefore set $\rho = 1 . 0$ and use a RoCo learning rate of $5 \times 1 0 ^ { - 3 }$ for this model. For both models, RoCo is optimized using AdamW for one epoch with a batch size of 1, $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , and $\epsilon = 1 0 ^ { - 8 }$

For downstream evaluation, we use EvalScope and follow the official sampling recommendations for both Qwen3-30B-A3B-Instruct-2507 and

Qwen3.5-35B-A3B in non-thinking mode: temperature = 0.7, top-p = 0.8, top-k = 20, and min-p = 0.0. The maximum generation lengths are set to 8,192 tokens for GPQA-Diamond and MATH-500, 4,096 tokens for IFEval and GSM8K, and 2,048 tokens for MMLU-Redux and BFCL-v3.

## C Efficiency Analysis

![](images/a6a2e10e5dd36b85630a9cd974cfc8d97b8f5d453df62338f7950901e1bb2ba0.jpg)  
(a) Model Size under Different Bit-Widths

![](images/b0da6d4ceca2efef948c9ae1435de663e710265f97574bb93ba73bc753d0268d.jpg)  
Figure 7: (a) Model size comparison of Qwen3-30B-A3B-Instruct-2507 and Qwen3.5-35B-A3B under different expert average bit-width settings. (b) End-to-end quantization time comparison of different methods on Qwen3-30B-A3B-Instruct-2507 under a 2.0-bit expert average bit-width setting.

Model Size. As shown in Fig. 7(a), ProQuant achieves substantial model compression under different expert average bit-width settings. Since the expert average bit-width does not directly reflect the overall storage cost due to non-expert components in MoE models, we report the model size after quantization. For Qwen3-30B-A3B-Instruct-2507, ProQuant reduces the BF16 model size from 61.06 GB to 8.09 GB, achieving 7.55× storage compression. Combined with the generation-based evaluation results, ProQuant effectively reduces storage requirements while maintaining strong downstream performance.

Quantization Time. We further compare the quantization time required to obtain the final quantized models. As shown in Fig. 7(b), although Pro-Quant introduces additional statistical computation for bit-width allocation and RoCo fine-tuning compared with GPTQ, it only incurs a 1.21× overhead. Compared with GEMQ, which determines bitwidth allocation through gradient-based backpropagation, ProQuant adopts a much lighter-weight allocation process. These results demonstrate that Pro-Quant improves low-bit model performance while maintaining high quantization efficiency.

## C.1 Model Size Calculation

We calculate model size based on the parameter counts and quantization precision of different components. All original weights are stored in BF16, with 16 bits per parameter. For asymmetric quantization with a group size of 128, we account for two 16-bit metadata values per group, corresponding to the scale and zero-point. Accordingly, the storage ratio of b-bit quantized weights relative to their original BF16 representation is:

$$
r ( b ) = { \frac { b } { 1 6 } } + { \frac { 1 6 + 1 6 } { 1 2 8 \times 1 6 } } = { \frac { b } { 1 6 } } + { \frac { 1 } { 6 4 } } .\tag{14}
$$

We divide model parameters into three categories according to their quantization configurations: experts subject to mixed-precision allocation, parameters fixed at 4-bit precision, and parameters retained in BF16. The total model size under an average expert bit-width of B is calculated as:

$$
S _ { \mathrm { E } } r ( B ) + S _ { \mathrm { 4 } } r ( 4 ) + S _ { \mathrm { F P } } ,\tag{15}
$$

where $S _ { \mathrm { E } } , S _ { 4 }$ , and $S _ { \mathrm { F P } }$ denote the original BF16 storage sizes of these three parameter categories, respectively, measured in GB. Here, B denotes the parameter-weighted average expert bit-width. When expert pruning is considered, Eq. (15) provides a conservative upper bound on parameter storage, as it still accounts for quantization metadata associated with pruned experts.

Qwen3-30B-A3B-Instruct-2507. The model contains 48 MoE layers with 128 routed experts per layer. According to the quantization configuration, the expert weights, attention weights, and remaining BF16 parameters occupy 57.982 GB, 1.812 GB, and 1.270 GB in their original BF16 representation, respectively. Therefore, the model size under an average expert bit-width B is

$$
S _ { \mathrm { Q w e n 3 } } ( B ) = 5 7 . 9 8 2 r ( B ) + 1 . 8 1 2 r ( 4 ) + 1 . 2 7 0 .\tag{16}
$$

The resulting model sizes are 11.717 GB, 9.905 GB, and 8.093 GB under average expert bit-widths of 2.5, 2.0, and 1.5 bits, respectively.

Qwen3.5-35B-A3B. The model consists of 40 MoE layers with 256 routed experts and one shared expert per layer. Its attention module combines 30 Gated DeltaNet layers and 10 full-attention layers. Under our quantization configuration, the routed experts, shared experts, QKVO projections, and remaining BF16 parameters occupy 64.425 GB, 0.252 GB, 2.055 GB, and 2.590 GB in BF16 storage, respectively. The corresponding model size is calculated as

$$
\begin{array} { l } { S _ { \mathrm { Q w e n 3 . 5 } } ( B ) = 6 4 . 4 2 5 r ( B ) } \\ { \qquad + \left( 0 . 2 5 2 + 2 . 0 5 5 \right) r ( 4 ) } \\ { \qquad + 2 . 5 9 0 . } \end{array}\tag{17}
$$

This yields model sizes of 14.276 GB, 12.263 GB, and 10.249 GB under average expert bitwidths of 2.5, 2.0, and 1.5 bits, respectively. The calculation considers only the language model parameters and excludes the vision encoder.

## D Comparative Analysis of Expert Importance Metrics

We compare the effects of different expert importance metrics on mixed-precision allocation and further examine the need to explicitly remove the influence of expert activation frequency. For a fair comparison, we use {1, 2, 3} as the candidate expert bit-widths, fix the average expert bit-width at 2.0 bits, and keep the subsequent allocation procedure and all other quantization settings identical. No post-quantization finetuning is performed for any method.

As shown in Table 6, our importance metric achieves an average score of 61.40 across six benchmarks, outperforming the metrics used in MxMoE, PMQ, and GEMQ without finetuning by 11.51, 11.75, and 11.18 percentage points, respectively. These results demonstrate that the definition of expert importance substantially affects downstream performance under the same bit-width budget and allocation settings.

To examine the influence of activation frequency, we introduce $\mathrm { P o P ^ { * } }$ , which does not explicitly divide the expert quantization cost by the expert activation count, while keeping all other settings identical to PoP. Explicitly removing the influence of activation frequency increases the average score from 47.84 to 61.40 and improves performance on five of the six benchmarks, with a decrease on MMLU-Redux. We hypothesize that expert activation patterns on the C4 calibration set may be more closely aligned with those on MMLU-Redux. Such alignment could favor importance metrics that incorporate activation frequency on MMLU-Redux, while retaining this frequency dependence leads to lower performance on all other evaluated benchmarks.

## E Dialog Examples

## Example 1: MATH-500 (Level 2/5)

Prompt: A 90<sup>◦</sup> rotation around the origin in the counterclockwise direction is applied to $7 + 2 i$ . What is the resulting complex number? Please reason step by step, and put your final answer within \boxed{}.

## BF16 response

We are given a complex number: $7 + 2 i ,$ , and we are to apply a 90<sup>◦</sup> counter-clockwise rotation around the origin. Step 1: Understand the rotation in the complex plane. In the complex plane, multiplying a complex number by i corresponds to $\textup { i } 9 0 ^ { \circ }$ counter-clockwise rotation about the origin. Why? Because $i = \cos ( 9 0 ^ { \circ } ) + i \sin ( 9 0 ^ { \circ } )$ ; multiplying by i rotates the number by 90<sup>◦</sup> counter-clockwise. Step 2: Multiply the complex number by $i _ { \pm } i ( 7 + 2 i ) =$ $i \cdot { \bar { 7 } } + i \cdot 2 i { \bar { ; } } = 7 i + 2 { \bar { i } } ^ { 2 } .$ . Recall that $i ^ { 2 } \stackrel { \cdot } { = } - 1$ , so $7 i + 2 ( - 1 ) = 7 i - 2 = - 2 + 7 i$   
Final Answer: \boxed $| \{ - 2 \ + \ 7 \mathrm { i } \}$ . This is the result of rotating $7 + 2 i$ by $9 0 ^ { \circ }$ counter-clockwise around the origin.

Correct.

## MxMoE response

We are given a complex number: $7 + 2 i$ , and we are to apply a 90<sup>◦</sup> rotation counter-clockwise around the origin. Let’s break this down step by step.

Step 1: Understand what a $9 0 ^ { \circ }$ counter-clockwise rotation means. Rotating a point (or complex number) by $9 0 ^ { \circ }$ counter-clockwise around the origin means we are transforming its coordinates by rotating it in the plane. In complex numbers, this is equivalent to rotating the vector from the origin (i.e., the complex number itself) by $9 0 ^ { \circ }$ But here’s the key: the rotation is around the origin, so we are rotating the vector $7 + 2 i$ as a vector from the origin, by $9 0 ^ { \circ }$ counter-clockwise.

Step 2: Recall how rotation affects complex numbers. The real part is the x-axis. The imaginary part is the y-axis. So 7 + 2i is a point at $( 7 , 2 )$ in the plane. The vector (7, 2) is rotated so that its new coordinates become $( - 2 , 7 ) ?$ Wait—let’s think carefully. A 90<sup>◦</sup> CCW rotation of $( x , y )$ gives $( - y , x ) ? { \mathrm { ~ W a i t \mathrm { - l e t ' s ~ v e r i f y } } }$ Take the vector (1, 0): rotating 90<sup>◦</sup> CCW gives (0, 1).

Incorrect: repeated rotation checks without a final answer.

## PMQ response

We are given a complex number: $7 + 2 i$ , and we are to apply a $\mathsf { \bar { g } 0 ^ { \circ } }$ rotation counter-clockwise around the origin to this number. Let’s break this down step by step.

Step 1: Understand the meaning of $7 + 2 i .$ . The real part is 7. The imaginary part is 2i. So the complex number is $z = 7 + 2 i$ . This is the same as $z = 7 + 2 j$ in engineering notation, but here i is the imaginary unit.

Step 2: What does a $9 0 ^ { \circ }$ counter-clockwise rotation mean? Rotating a complex number by $9 0 ^ { \circ }$ counterclockwise around the origin is equivalent to multiplying the complex number by i. Wait—let’s clarify this. Actually, in complex numbers, rotation is not done by multiplying by i. Instead, we use rotation in the plane, which is geometric.

<table><tr><td>Model</td><td>Bit Budget</td><td>Method</td><td>GSM8K</td><td>MATH-500</td><td>GPQA-D</td><td>IFEval</td><td>MMLU-R</td><td>BFCL-v3</td><td>AVG</td></tr><tr><td rowspan="3">Qwen3-30B- A3B-</td><td rowspan="6">2.0</td><td>MxMoE</td><td>81.43</td><td>15.60</td><td>21.72</td><td>75.80</td><td>74.65</td><td>30.11</td><td>49.89</td></tr><tr><td>PMQ</td><td>84.61</td><td>12.80</td><td>24.24</td><td>75.03</td><td>71.44</td><td>29.75</td><td>49.65</td></tr><tr><td>GEMQ w/o FT</td><td>83.71</td><td>17.20</td><td>25.76</td><td>73.74</td><td>71.14</td><td>29.74</td><td>50.22</td></tr><tr><td rowspan="2"></td><td> $\mathsf { P o P } ^ { * } \left( \left\{ 1 , 2 , 3 \right\} \right)$ </td><td>86.13</td><td>15.40</td><td>21.72</td><td>73.22</td><td>61.98</td><td>28.61</td><td>47.84</td></tr><tr><td> $\underline { { \mathsf { P o P } ( \{ 1 , 2 , 3 \} ) } }$ </td><td>91.74</td><td>78.60</td><td>22.22</td><td>77.87</td><td>58.18</td><td>39.76</td><td>61.40</td></tr></table>

Table 6: Comparison of expert importance metrics on Qwen3-30B-A3B-Instruct-2507 with an average expert bit-width of 2.0 bits. All methods use candidate bit-widths {1, 2, 3} and the same allocation procedure, without post-quantization finetuning. $\mathrm { P o P ^ { * } }$ omits explicit normalization by the expert activation count.

Step 3: Apply rotation geometrically. Rotation of a point $( x , y )$ by $9 0 ^ { \circ }$ counter-clockwise: $( x , y )$ becomes $( y , x ) ?$ Wait—let’s recall the rotation matrix. The rotation matrix for $9 0 ^ { \circ }$ counter-clock clockwise is:

\left \left \left. . .

Incorrect: malformed rotation-matrix loop; no final answer.

## GEMQ response

We are given a complex number: $7 + 2 i ,$ and we are to apply $\mathrm { ~ a ~ } \bar { 9 } 0 ^ { \circ }$ counter-clockwise rotation around the origin to it. We need to find the resulting complex number.

Step 1: Understand the complex number. The complex number $7 + 2 i$ means 7 is the real part and 2i is the imaginary part. This is a point in the complex plane: (7, 2).

Step 2: Understand what $\mathbf { a } \ 9 0 ^ { \circ }$ counter-clockwise rotation means. We are rotating the point about the origin $( 0 , 0 )$ , not about itself. A 90<sup>◦</sup> counter-clockwise rotation about the origin corresponds to multiplying the complex number by i. But wait: rotation about the origin is not the same as multiplying by i, unless you’re rotating the vector from the origin. To rotate it around the origin, we must translate it to the origin? No—rotation about the origin means we keep the origin fixed.

Step 3: Use geometric rotation. The new x-coordinate becomes 2 (original y); the new y-coordinate becomes 7 (original x), but negated because of rotation?

I think the correct answer is $2 + 7 i .$ So, the resulting complex number is \boxed{2 + 7i}.

So, \boxed{2 + 7i}.

So, \boxed{2 + 7i}.

After $9 0 ^ { \circ }$ CCW rotation, the new $x = y ,$ new $y = x .$ So, the point at (7, 2) becomes $( 2 , 7 ) .$ . Therefore, the resulting complex number is \boxed{2 + 7i}.

But I think it’s wrong. Let’s use the correct value. After 90<sup>◦</sup> CCW rotation, the new complex number is $2 + 7 i$ So, \boxed $\{ 2 ~ + ~ 7 \mathrm { i } \}$ . I think that’s the answer.

Incorrect: wrong sign after rotating (7, 2).

## ProQuant response

We are given a complex number: $7 + 2 i ,$ and we are to apply a 90<sup>◦</sup> counter-clockwise rotation around the origin. Step 1: Understand what a $9 0 ^ { \circ }$ counter-clockwise rotation does to a complex number. In the complex plane, rotating a complex number counter-clockwise by 90<sup>◦</sup> is equivalent to multiplying it by i. This is because $i = \cos ( \mathrm { 9 0 ^ { \circ } } ) + i \sin ( \mathrm { 9 0 ^ { \circ } } )$ , so multiplying by i rotates the number by 90<sup>◦</sup> counter-clockwise.

Step 2: Multiply 7 + 2i by $i . \ i ( 7 + 2 i ) = i \cdot 7 + i \cdot 2 i ;$

$= 7 i + 2 i ^ { 2 } .$ . Recall that $i ^ { 2 } = - 1 , \mathrm { { s o } } 7 i + 2 ( - 1 ) = 7 i - 2 .$   
So the result $\mathbf { i s } - 2 + 7 i .$   
Final Answer: \boxed{-2 + 7i}.   
Correct.

## Example 2: MATH-500 (Level 2/20)

Prompt: Simplify ${ \frac { \sec x } { \sin x } } - { \frac { \sin x } { \cos x } } .$ Please reason step by step, and put your final answer within \boxed{}.

## BF16 response

We are asked to simplify the expression ${ \begin{array} { l } { { \frac { \sec x } { \sin x } } \mathrm { ~ - ~ } { \frac { \sin x } { \cos x } } . } \\ { { \frac { \sin x } { \cos x } } \mathrm { ~ - ~ } { \frac { \cos x } { \cos x } } . } \end{array} }$ Step 1: Recall trigonometric identities. We know $\begin{array} { r } { \sec x = \frac { 1 } { \cos x } . } \end{array}$

Step 2: Rewrite sec x in terms of cos x. Replace sec x

with ${ \scriptstyle { \frac { 1 } { \cos x } } } .$ Simplify the $\begin{array} { r } { \mathrm { { f i r s t ~ t e r m : } } \frac { \frac { 1 } { \cos x } } { \sin x } = \frac { 1 } { \cos x \sin x } . } \end{array}$ The expression becomes $\begin{array} { r } { \frac { 1 } { \cos x \sin x } - \frac { \sin x } { \cos x } . } \end{array}$

Step 3: Get a common denominator. Both terms have cos x in the denominator. The least common denominator is cos x sin x. Rewrite the second term: $\begin{array} { l } { \frac { \sin x } { \cos x } = } \end{array}$ ${ \frac { \sin ^ { 2 } x } { \cos x \sin x } } .$

Step 4: Combine the fractions. $\begin{array} { r } { \frac { 1 } { \cos x \sin x } - \frac { \sin ^ { 2 } x } { \cos x \sin x } ; } \end{array}$ $\begin{array} { r } { = \frac { 1 - \sin ^ { 2 } x } { \cos x \sin x } . } \end{array}$

Step 5: Use Pythagorean identity. Since $1 - \sin ^ { 2 } x =$ $\cos ^ { 2 } x ,$ substitute to get ${ \frac { \cos ^ { 2 } x } { \cos x \sin x } } .$

Step 6: Simplify. Cancel one cos x from numerator and denominator: $\begin{array} { r } { \frac { \cos x } { \sin x } = \cot x . } \end{array}$

Final Answer: \boxed{\cot x}.

Correct.

## MxMoE response

We are given the expression $\begin{array} { r } { \frac { \sec x } { \sin x } - \frac { \sin x } { \cos x } } \end{array}$ . Let’s simplify this step by step.

Step 1: Understand the terms. Sec x means the secant of angle x, which is $\textstyle { \frac { 1 } { \cos x } } . \ S 0$ <sup>sec</sup> <sup>x</sup> is the same as ${ \frac { 1 / \cos x } { \sin x } } ,$ sin x which is $\textstyle { \frac { 1 } { \cos x \sin x } } .$ But let’s rewrite it clearly.

Actually, sec x is $\begin{array} { r } { \frac { 1 } { \cos x } , \ \operatorname { s o } \ \frac { \sec x } { \sin x } = \frac { 1 } { \sin x \cos x } . } \end{array}$

Wait—let’s be careful. The expression is: \frac{\sec x}{\sin x} = \frac \text{sec x} / \sin x. But actually, that’s not quite right. Let’s clarify.

But this is messy.

Incorrect: loops on fraction rewriting; no final answer.

## PMQ response

We are given the expression $\begin{array} { r } { { \frac { \sec x } { \sin x } } \_ - \ { \frac { \sin x } { \cos x } } } \end{array}$ . But this notation is ambiguous and likely contains a misunderstanding. Let’s carefully analyze what’s going on.

Step 1: Understand the notation. “Sec x” likely refers to the secant of x; “sin x” is sine and “cos x” is cosine. But this is not standard mathematical notation. It’s written as if we are dividing “sec x” by “sin x”, and subtracting “sin x” divided by “cos x”.

```latex
It doesn’t simplify. Therefore, the simplified form is the
expression itself. So the answer is:
\boxed{\frac{\sec x}{\sin x} - \frac{\sin
x}{\cos}}.
But that’s not simplified.
Final answer:
\boxed{\frac{\sec x}{\sin x} - \frac{\ \sin
x}{\cos}}.
But that’s not simplified.
Incorrect: stops at an unsimplified expression.
```

Wait—perhaps this is not a mathematical expression but a typographical or notational error?

Step 2: Consider trigonometric identities. But perhaps the user meant sec x/ sin x as a ratio? But still, the expression is not standard. Could this be a miswritten version of a trigonometric identity or formula?

The expression is not valid or cannot be simplified due to ambiguous and non-standard notation. Therefore, the expression cannot be simplified as written.

Final Answer: \boxed{Cannot be simplified}.

Incorrect: treats a valid expression as ambiguous.

## GEMQ response

We are given the expression <sup>sec</sup> <sup>x</sup><sub>sin x</sub> sin x . Let’s simplify this step by step.

Step 1: Understand the terms. We are dealing with trigonometric functions: sec x is the secant, sin x is the sine, and cos x is the cosine. But here, we are dealing with ratios of these values.

Step 2: Rewrite the expression. Wait—there’s a typo in the expression. But let’s clarify: the expression is sec x <sup>sin</sup> <sup>x</sup> . We are to simplify this expression as written. sin x cos x   
sec x 1/ cos x = 1 Wait—actually, this is sin x sin x cos x sin x   
ambiguous. sec x is a function, not a number.

## ProQuant response

We are asked to simplify the expression sec x sin x sin x cos x   
Step 1: Recall identities. We know sec x 二 1 cos x   
So, rewrite sec x in terms of cos x: sec x <sup>1/</sup> <sup>cos</sup> <sup>x</sup><sub>sin x</sub> ; =   
<sup>1</sup> . Now the expression becomes cos x sin x sin x   
Step 2: Find a common denominator. The least com  
mon denominator is cos x sin x. Rewrite the second term:   
sin x = sin x . Now subtract: cos x sin x <sup>sin2</sup> <sup>x</sup> ;   
= 1−sin<sup>2</sup> x cos x sin x

Step 3: Use identity 1 − sin<sup>2</sup> x = cos<sup>2</sup> x. 1−sin<sup>2</sup> x <sup>cos2</sup> <sup>x</sup> . Now simplify: cos<sup>2</sup> x = cos x = cot x. cos x sin x cos x sin x sin x Final Answer: \boxed{\cot x}.

Correct.

## Example 3: BFCL-v3 (simple/1)

Prompt: Calculate the factorial of 5 using math functions.

Tool: math\_factorial(number: integer). Description: Calculate the factorial of a given number (Python 3 syntax). The required number argument is an integer: the number for which factorial needs to be calculated.

## BF16 response

[{"function":{   
"arguments":{"number":5},   
"name":"math\_factorial"},   
"id":"chatcmpl-tool-1e13c937663d4ebeb0ae5f34   
96935db8",   
"internal":null,"parse\_error":null,   
"type":null,"view":null}]   
Correct.

MxMoE response   
<tool\_call>   
{"name": math\_factorial,   
"arguments": math\_factorial, 5)   
</tool\_call>   
Incorrect: malformed tool-call arguments.

PMQ response   
<tool\_call>   
{"name": "mathdba",   
"arguments": {"number": {"type": "integer",   
"description": "The number for which factorial   
needs to be calculated."%}}   
</tool\_call>

Incorrect: wrong function name and malformed arguments.

GEMQ response   
[{"function":{   
"arguments":{"number":"number"},   
"name":"math\_factorial"},   
"id":"chatcmpl-tool-d32f30a02e1d4fd8a066626a   
e647681b",   
"internal":null,"parse\_error":null,   
"type":null,"view":null}]   
Incorrect: passes a string instead of the integer 5.

ProQuant response   
[{"function":{   
"arguments":{"number":5},   
"name":"math\_factorial"},   
"id":"chatcmpl-tool-2b59039d87e24cac98c20e5c   
f00a4494",   
"internal":null,"parse\_error":null,   
"type":null,"view":null}]   
Correct.

## Example 4: BFCL-v3 (simple/6)

Prompt: What are the roots of the quadratic equation where a = 2, b = 5 and c = 3?

Tool: solve\_quadratic(a: integer, b: integer, c: integer). Description: Find the roots of a quadratic equation and return both roots (Python 3 syntax). Required integer arguments: a, coefficient of x<sup>2</sup>; b, coefficient of x; and c, constant term.

BF16 response   
[{"function":{   
"arguments":{"a":2,"b":5,"c":3},   
"name":"solve\_quadratic"},   
"id":"chatcmpl-tool-4157e7aca65c477abd6658c2   
09fb7164",   
"internal":null,"parse\_error":null,   
"type":null,"view":null}]   
Correct.

MxMoE response   
Do you want to solve the quadratic equation with $a = 2 ,$   
b = 5, and c = 3?   
The quadratic equation is $2 x ^ { 2 } + 5 x + 3 = 0 .$ . We can   
solve this equation using the quadratic formula:   
x = \frac{-b \pm   
\sqrt{b^2 - c^2) } / (2a)   
Let’s solve it step-by-step. First, compute the discrimi   
nant: $b ^ { 2 } - c ^ { 2 } = \hat { 5 } ^ { 2 } - 3 ^ { 2 } = 2 5 - 9 = \hat { 1 } 6$   
Then compute the roots: x = \frac(-5 \pm \sqrt(16))   
/ (2\*2)   
= (-5 \pm 4) / 4.   
$x = ( - 5 + 4 ) / 4 = - 1 / 4 ; x = ( - 5 - 4 ) / 4 = - 9 / 4 .$   
So the roots are x = −0.25 and x = −2.25. Therefore,   
the roots are −0.25 and −2.25.   
Incorrect: no tool call; incorrect roots.   
PMQ response   
<tool\_call>   
{"name": "solve\_quadratic",   
"arguments $: 0 . 1 ^ { - } a ^ { i } : 2 , ^ { n } b ^ { n } : 5 , ^ { n } c ^ { n } : 3 ^ { n } \}$   
</tool\_call>   
Incorrect: malformed tool-call JSON.   
GEMQ response   
<tool\_call>   
{"name": "solve\_quadratic",   
"argument $\mathsf { s } ^ { \prime \prime } \colon \{ ^ { \prime \prime } \mathsf { a } ^ { \prime \prime } \colon$ : and "b" and "c": 2, 5, 3}}   
</tool\_call>   
Incorrect: malformed argument mapping.   
ProQuant response   
[{"function":{   
"arguments $" : \{ " { \sf a } ^ { \prime } : 2 , " { \sf b } ^ { \prime \prime } : 5 , " { \sf c } " : 3 \}$   
"name":"solve\_quadratic"},   
$" \mathrm { i } \mathrm { d } ^ { \prime \prime } : "$ chatcmpl-tool-e455b126f3664211b67c43e5   
2594e61f",   
"internal":null,"parse\_error":null,   
"type":null,"view":null}]   
Correct.  
Table 7: Qualitative comparison of model responses on four examples using Qwen3-30B-A3B-Instruct-2507. All quantized methods have an average expert bit-width of 2.5 bits and use the same prompt and decoding settings within each example. For readability, responses have been abridged and reformatted, with repetitive passages and off-topic digressions omitted while retaining key reasoning steps. For the two BFCL examples, tool definitions are summarized rather than reproduced in full JSON format.

## F License and Attribution

Our implementation builds on GEMQ (Deng et al., 2026), which is released under the MIT License. We retain the original copyright and license notices in our released code.

## G AI Use Statement

AI assistants were used only to modify code according to our instructions and to polish the manuscript wording. The literature review and development of the research contributions were conducted by the authors.