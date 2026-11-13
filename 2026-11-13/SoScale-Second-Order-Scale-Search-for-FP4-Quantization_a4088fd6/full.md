# SoScale: Second-Order Scale Search for FP4 Quantization

Pengxiang Zhao, Xing Li, Hui-Ling Zhen, Weizhe Lin, Xianzhi Yu, Zhenhua Dong

Huawei Technologies Co., Ltd.

## Abstract

Low-bit quantization reduces the memory footprint and inference cost of large language models (LLMs), yet accurate FP4 quantization remains challenging. Practical formats such as MXFP4, NVFP4, and HiF4 pair a 4-bit floating-point payload with block-wise or hierarchical scale codes, making accuracy highly sensitive to how each local distribution is scaled. Default maximum-absolute-value scaling preserves dynamic range, but can spend scarce FP4 levels on rarely occupied outlier regions and leave dense tensor values poorly resolved. Existing scale-search methods minimize local element-wise reconstruction error, optimizing tensor fidelity rather than layeroutput fidelity. We propose SoScale, a second-order weight scale-code search method for hierarchical FP4 quantization. SoScale searches hardware-valid scale codes under a Hessianaware layer reconstruction objective and couples this search with GPTQ-style error recovery, selecting scales by their impact on layer outputs. Across Qwen3 models, SoScale consistently reduces downstream quantization degradation under MXFP4, NVFP4, and HiF4.

## Introduction

Large language models (LLMs) have demonstrated strong capabilities across a wide range of natural language processing tasks (OpenAI 2026; DeepSeek-AI 2026; Qwen Team 2026), but their deployment remains constrained by substantial memory footprint and inference cost. Post-training quantization has therefore become an important technique for reducing model size and improving serving eficiency without expensive retraining (Frantar et al. 2023; Xiao et al. 2023; Lin et al. 2024; Shao et al. 2024; Zhao and Yuan 2025). Among recent low-bit formats, FP4 is appealing for LLM inference because it compresses each value to four bits while retaining exponent-based dynamic range and enabling hardware acceleration for low-bit floating-point computation. Practical FP4 formats, such as MXFP4 (Rouhani et al. 2023), NVFP4 (Alvarez et al. 2025), and HiF4 (Luo et al. 2026), further introduce hierarchical scaling mechanisms to better accommodate the non-uniform distributions of LLM weights and activations.

Despite this promise, accurate quantization with hierarchical FP4 formats remains highly sensitive to scale-code selection. Since the FP4 payload provides only 16 representable values, the block scale determines where these quantization levels fall on the real-valued axis. When a block contains outliers, the standard maximum-absolute-value scaling rule expands the grid to cover the largest-magnitude element. This preserves range, but can spend levels on rarely occupied regions and reduce resolution near the dense mass of tensor values. Figure 1a illustrates this efect on a real Qwen3- 8B MXFP4 weight block. With default scaling, the grid is stretched by the outliers, yet the two outermost representable levels receive no assigned values because the outliers are rounded to the next-inner levels. This leaves fewer useful levels near zero, whereas the searched scale contracts the grid toward the occupied region and improves resolution for dense values. Figure 1b provides a complementary error view: after sorting values within the block, the absolute errors at the two tails remain nearly unchanged, while errors in the middle region, corresponding to values near zero, drop substantially under the searched scale.

The example above suggests that adapting the scale can be an efective way to reduce FP4 quantization error, and prior work has explored several related directions. Some methods reshape tensor distributions before quantization (Xiao et al. 2023; Ashkboos et al. 2024; Liu et al. 2025; Shao et al. 2025); others compensate residual quantization errors after or during quantization (Frantar et al. 2023; Meng et al. 2026); and FP4-specific methods adapt block scales with format-specific heuristics or local distortion objectives (Cook et al. 2026; Gupta et al. 2026). These approaches show that distributionaware FP4 quantization is important, but most scale-selection objectives remain local: they choose a scale code by how well it reconstructs a block, even though the code ultimately matters through its efect on the layer output. As shown in Figure 1c, reducing local element-wise error does not necessarily reduce model-level perplexity. This motivates treating scale selection as a layer reconstruction problem rather than a local tensor fitting problem.

In this work, we propose SoScale, a second-order posttraining quantization (PTQ) method for weight and activation scale-code selection in hierarchical FP4 quantization. SoScale operates on a fixed pretrained model, without retraining or modifying the target FP4 format; instead, it searches within the format’s legal scale-code space and chooses codes by minimizing a Hessian-weighted layer reconstruction loss. At quantization time, SoScale formulates scale selection as a sequential second-order reconstruction problem: after each block is committed, its induced error is propagated through the inverse-Hessian factor, so later blocks are optimized in a compensated parameter space rather than against the original weights (Frantar et al. 2023). This candidate-set view makes the method format-agnostic: given the legal scale-code set and quantizer of a hierarchical FP4 format, the same Hessian-aware search and recovery procedure can be applied without format-specific redesign. Since weight quantization is performed ofline and the resulting representation preserves the target FP4 encoding, SoScale does not introduce serving-time search and remains compatible with hardware inference paths that already support the corresponding format. We instantiate this general formulation for MXFP4, NVFP4, and HiF4, where it remains complementary to preprocessing methods such as block Hadamard rotations (Shao et al. 2025).

![](images/455a68e37ee80fc1601b35180f1b383b7e0d94b71df6aa4652b433d1b99aacb2.jpg)  
(a) Scale-code alignment.

![](images/e618d40dedf42347d8a6c0ca604380952ae5e7799054a61aa0b27f86468d2d43.jpg)  
(b) Element-wise error.

![](images/c6d27aacf373f5833794f2d2e59e34ff769e34916734ece0995aadcf43da5eb4.jpg)  
(c) PPL comparison.  
Figure 1: Scale-code alignment and model-level impact. On a real Qwen3-8B weight block, scale search changes the MXFP4 scale from 0.125 to 0.0625. (a) Default scaling preserves a wider range but leaves extreme levels unused, while the searched scale concentrates levels near dense tensor values. (b) The searched scale reduces element-wise reconstruction error for most elements. (c) Local element-wise scale search does not consistently reduce perplexity, whereas SoScale does.

We evaluate SoScale on Qwen3-8B and Qwen3-14B (Yang et al. 2025) across three hierarchical FP4 formats: MXFP4, NVFP4, and HiF4. Across downstream benchmarks, the best SoScale variants achieve the highest average score in every model-format group, outperforming RTN, GPTQ (Frantar et al. 2023), local ScaleSearch (Gupta et al. 2026), and BlockRotation (Ashkboos et al. 2024) baselines. On Qwen3- 14B, these variants retain 97.17% of BF16 average accuracy under MXFP4, 99.20% under NVFP4, and 99.40% under HiF4. These results indicate that optimizing scale codes with second-order layer information recovers accuracy that local tensor-level objectives miss.

Our contributions are summarized as follows:

• We identify an outlier-induced scale mismatch in hierarchical FP4 quantization: default range-preserving scales can reserve scarce FP4 levels for regions that receive no assignments, reducing resolution for dense weight values.

• We introduce SoScale, an ofline scale-code optimizer that preserves the target FP4 format, combines Hessianweighted layer reconstruction with quantization error compensation, and retains compatibility with existing FP4 hardware inference paths.

<table><tr><td></td><td>MXFP4</td><td>NVFP4</td><td>HiF4</td></tr><tr><td>Element format</td><td>E2M1</td><td>E2M1</td><td>E1M2</td></tr><tr><td>Scale hierarchy</td><td>E8M0</td><td>FP32 + E4M3</td><td>E6M2 + E1 + E1</td></tr><tr><td>Granularity</td><td>32</td><td>16 + tensor</td><td>64 / 8 / 4</td></tr><tr><td>Bits / element</td><td>4.25</td><td>≥4.50</td><td>4.50</td></tr></table>

Table 1: Compact comparison of hierarchical FP4 formats considered in this work.

• We show consistent gains over representative baselines across two Qwen3 models and three FP4 formats.

## Related Work

Hierarchical FP4 formats. Low-bit floating-point formats reduce LLM inference cost while preserving exponent-based dynamic range and enabling hardware acceleration. Practical FP4 formats, including MXFP4 (Rouhani et al. 2023), NVFP4 (Alvarez et al. 2025), and HiF4 (Luo et al. 2026), pair a 4-bit payload with block-wise or hierarchical scales; their main structural diferences are summarized in Table 1. Despite diferences in scale hierarchy, granularity, and codebooks, these formats share the same abstraction: the scale code places the FP4 payload levels on the real-valued axis for each block. Consequently, scale selection becomes a primary source of accuracy loss when block distributions are heavy-tailed or contain outliers.

Post-training quantization methods. PTQ reduces the memory and compute cost of pretrained LLMs without expensive retraining. Simple round-to-nearest (RTN) quantization is widely used as an eficient baseline, but often degrades accuracy at very low bit-widths. Calibration-based PTQ methods improve low-bit accuracy by using activation statistics, saliency estimates, or second-order approximations to preserve layer behavior (Frantar et al. 2023; Lin et al. 2024; Shao et al. 2024; Zhao and Yuan 2025). A complementary line smooths, rotates, or rebalances weights and activations before quantization to reduce outlier-induced dificulty (Xiao et al. 2023; Ashkboos et al. 2024; Liu et al. 2025). A common lesson is that quantization errors should be evaluated by how they afect layer outputs, not only by element-wise tensor distortion. SoScale adopts this layer-centric view, but applies it to the discrete scale codes of hierarchical FP4 formats.

FP4 methods. Recent FP4 methods typically handle outlier sensitivity through preprocessing, compensation, or scale adaptation. Block Hadamard rotations shrink each MXFP4 block before quantization, but rely on a downstream scale rule (Shao et al. 2025). ARCQuant repairs NVFP4 errors with reordering and residual channels, adding compensation instead of optimizing scales (Meng et al. 2026). The closest work adapts scales directly: Four Over Six switches between two NVFP4 rules (Cook et al. 2026), and Scale-Search minimizes local mean squared error (MSE) over block scales (Gupta et al. 2026). These methods establish scale selection as important, but remain tied to format-specific rules or local tensor loss. SoScale uses a Hessian-weighted layer reconstruction loss to search legal scale codes across MXFP4, NVFP4, and HiF4, with GPTQ recovery.

## Methodology

This section presents SoScale, a weight scale-code search method for hierarchical FP4 quantization. For each linear layer, SoScale selects legal scale codes with a Hessianweighted layer reconstruction objective and formulates blockwise quantization as a sequential second-order reconstruction process with error recovery. We first formulate scale-code search, then describe error recovery, and finally summarize the full pipeline.

## Hessian-Aware Scale-Code Search

Modern LLMs are dominated by linear projections in attention and feed-forward modules, making layer reconstruction a natural objective for post-training quantization. Consider a linear layer with calibration inputs $\mathbf { \bar { \boldsymbol { X } } } \in \mathbb { R } ^ { T \times n }$ and fullprecision weights $\pmb { W } \in \mathbb { R } ^ { m \times n }$

$$
\mathbf { Y } = \mathbf { X } \mathbf { W } ^ { \top } ,\tag{1}
$$

where $T$ denotes the number of calibration tokens, and n and m denote the input and output dimensions. Given quantized weights $\widehat { W }$ , SoScale measures quantization quality by the calibration-based layer reconstruction error

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } ( W , \widehat { W } ) = \left\| X W ^ { \top } - X \widehat { W } ^ { \top } \right\| _ { F } ^ { 2 } } } \\ { { \displaystyle \quad = \sum _ { i = 1 } ^ { m } e _ { i } ^ { \top } H e _ { i } } , } \end{array}\tag{2}
$$

where $\begin{array} { r } { e _ { i } = { \pmb w } _ { i } - \hat { \pmb w } _ { i } , { \pmb w } _ { i } } \end{array}$ and $\widehat { \pmb { w } } _ { i }$ are the i-th rows of W and $\widehat { W }$ , and $H = X ^ { \top } X$ is the empirical Hessian. Unlike element-wise distortion metrics, this objective does not treat all weight errors equally: errors that induce larger changes in the calibrated layer outputs receive larger penalties.

The reconstruction objective specifies the desired layer behavior, while the target FP4 format defines the discrete feasible set for $\widehat { W }$ . In hierarchical FP4 quantization, each row of W is partitioned into blocks along the input dimension, and each block is quantized by combining a low-bit FP4 payload with one or more scale codes, as summarized in Table 1. Let ${ \pmb w } _ { i , b } \in \mathbb { R } ^ { B }$ denote block b of row i, where B is the format-specific block size. For a target format ${ \mathcal F } ,$ , we write the quantized block as

$$
\begin{array} { r } { \widehat { \pmb w } _ { i , b } ( \pmb c ) = \pmb Q _ { \mathcal { F } } ( \pmb w _ { i , b } ; \pmb c ) , } \end{array}\tag{3}
$$

where $Q _ { \mathcal { F } }$ is the format-specific FP4 quantizer and $\ b { c } \in \mathbb { C } _ { \mathcal { F } }$ denotes a legal scale configuration. For each candidate c, the payload values are obtained by the standard rounding and clipping rule of the target format; SoScale only changes which legal scale codes are used. For concreteness, consider MXFP4. Each block uses an E2M1 payload and an E8M0 block-scale exponent. Let $c \in \mathbb { C } _ { \mathrm { M X } }$ denote the exponent code and let $s ( c ) = 2 ^ { c }$ be the corresponding block scale. The quantized block is

$$
\widehat { \pmb { w } } _ { i , b } ( c ) = s ( c ) \Pi _ { \mathrm { E 2 M 1 } } \left( \frac { \pmb { w } _ { i , b } } { s ( c ) } \right) ,\tag{4}
$$

where $\Pi _ { \mathrm { E 2 M 1 } } ( \cdot )$ denotes rounding and clipping to the E2M1 payload codebook. For a fixed E2M1 codebook, the exponent code c induces the scaled representable set $s ( c ) \mathcal { C } _ { \mathrm { E 2 M 1 } } \mathrm { : }$ increasing $s ( c )$ expands the covered range but coarsens the spacing between levels, while decreasing $s ( c )$ contracts the range and improves resolution for smaller magnitudes, as illustrated in Figure 1a. The same notation covers the other hierarchical formats considered in this work: for NVFP4, SoScale searches the E4M3 block-scale code while keeping the FP32 tensor scale fixed; for HiF4, it searches the E6M2 scale code and derives the E1 scales accordingly.

After fixing the FP4 payload and legal scale space, scalecode optimization reduces to selecting one code for each block. The ideal choice would minimize the layer reconstruction loss above over all scale codes, but the full Hessian couples errors across blocks and makes joint search combinatorial. SoScale therefore uses a block-local second-order surrogate aligned with the FP4 layout. Let $\mathbb { I } _ { b }$ denote the input indices of block $b ,$ and let

$$
H _ { b } = H _ { \mathbb { I } _ { b } , \mathbb { I } _ { b } }\tag{5}
$$

be the corresponding Hessian submatrix. For a candidate scale configuration c, define the row-block error

$$
\begin{array} { r } { \pmb { e } _ { i , b } ( \pmb { c } ) = \pmb { w } _ { i , b } - \pmb { \widehat { w } } _ { i , b } ( \pmb { c } ) . } \end{array}\tag{6}
$$

SoScale scores c by

$$
\ell _ { i , b } ( \pmb { c } ) = \pmb { e } _ { i , b } ( \pmb { c } ) ^ { \top } \pmb { H } _ { b } \pmb { e } _ { i , b } ( \pmb { c } ) ,\tag{7}
$$

The matrix $\pmb { H } _ { b }$ is what distinguishes this score from local MSE: if $\pmb { H } _ { b }$ were the identity, $\ell _ { i , b }$ would reduce to squared block error, whereas the empirical Hessian weights errors and correlations according to the calibration activations. For each row-block pair (i, b), SoScale selects

$$
{ \pmb { c } } _ { i , b } ^ { \star } = \arg \operatorname* { m i n } _ { { \pmb { c } } \in \mathbb { C } _ { i , b } } \ell _ { i , b } ( { \pmb { c } } ) ,\tag{8}
$$

where $\mathbb { C } _ { i , b }$ is formed by applying a small format-specific ofset window around the default scale code and clipping invalid codes to the legal range of the target format.

![](images/3da939979b41ab2de74ebd437fcad4c6862b5bfd15fe2f6dafa86f41a53ace3f.jpg)  
Figure 2: Workflow of SoScale for one linear layer. Calibration builds the empirical Hessian and damped inverse-Hessian factor. In 2.1, SoScale searches legal FP4 scale codes with a Hessian-aware layer loss, whereas scale-only baselines such as ScaleSearch and Four Over Six use local MSE-style objectives without Hessian information; in 2.2, it propagates the selected block’s error to future blocks. Inference uses only stored FP4 payloads, legal scale codes, and lightweight per-token activation scales.

## Second-Order Error Recovery

The previous subsection defines how to choose scale codes for a fixed block of weights. In sequential quantization, however, committing one block introduces an error that can be partially absorbed by the remaining unquantized blocks. If later blocks were searched on the original weights, their scale choices would ignore this compensation. This coupling is natural for hierarchical FP4 formats because both scale selection and error recovery operate at the block level. SoScale therefore follows the GPTQ principle (Frantar et al. 2023): after each block is quantized, it propagates a second-order correction to future blocks, and the next scale-code search is performed on the compensated weights.

Let $\widetilde { W }$ denote the working copy of the weights, initialized as $\widetilde { W } = W$ . Following GPTQ, we compute an uppertriangular factor U of the damped inverse Hessian,

$$
U ^ { \top } U = \left( \pmb { H } + \lambda \pmb { I } \right) ^ { - 1 } ,\tag{9}
$$

where λ is a damping coeficient. At one sequential step, let $\mathbb { Q } \subset \{ 1 , \ldots , n \}$ denote the input-column indices of the current FP4 block, and let F denote the columns belonging to the remaining future blocks. The slice $\widetilde { W } _ { : , \mathbb { Q } }$ contains the current block for all output rows. Each row in the slice has its own scale code or scale configuration, and we collect these row-wise codes in $c _ { \mathbb { Q } }$ . Before committing the block, SoScale searches these codes on the compensated weights:

$$
\pmb { c } _ { \mathbb { Q } } ^ { \star } = \arg \operatorname* { m i n } _ { \pmb { c } _ { \mathbb { Q } } \in \mathbb { C } _ { \mathbb { Q } } } \ell _ { \mathbb { Q } } ( \pmb { c } _ { \mathbb { Q } } ) .\tag{10}
$$

Here $\ell _ { \mathbb { Q } }$ sums the row-block scores from the previous subsection over all output rows. Using the selected scale codes, SoScale commits the quantized block

$$
\widehat { W } _ { : , \mathbb { Q } } = Q _ { \mathcal { F } } \left( \widetilde { W } _ { : , \mathbb { Q } } ; \pmb { c } _ { \mathbb { Q } } ^ { \star } \right) ,\tag{11}
$$

and obtains the block error

$$
\begin{array} { r } { { \cal E } _ { \mathbb { Q } } = \widetilde { { \cal W } } _ { : , \mathbb { Q } } - \widehat { { \cal W } } _ { : , \mathbb { Q } } . } \end{array}\tag{12}
$$

The block error is then propagated to the future blocks through the back-substitution:

$$
\widetilde { W } _ { : , \mathbb { F } } \gets \widetilde { W } _ { : , \mathbb { F } } - E _ { \mathbb { Q } } U _ { \mathbb { Q } , \mathbb { Q } } ^ { - 1 } U _ { \mathbb { Q } , \mathbb { F } } .\tag{13}
$$

Here $U _ { \mathbb { Q } , \mathbb { Q } }$ and $U _ { \mathbb { Q } , \mathbb { F } }$ are the submatrices of U indexed by the current block and the future input dimensions. This update changes only the working copy $\widetilde { W }$ used for subsequent blocks; the committed block $\mathcal { \hat { W } } _ { : , \mathbb { Q } }$ remains fixed. As a result, scale-code search and error recovery are interleaved: each block selects scales on weights that already include secondorder compensation from earlier quantization errors.

## Dynamic Per-Token Activation Quantization

SoScale optimizes the scale codes of weight blocks. Activations are handled by an online dynamic per-token FP4 rule under the target format. For token t, let $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ denote its activation vector and let $\mathbf { \Delta x } _ { t , b }$ denote block b under the target format. Each activation block is quantized as

$$
\widehat { \pmb { x } } _ { t , b } = Q _ { \mathcal { F } } ^ { \mathrm { a c t } } \left( \pmb { x } _ { t , b } ; c _ { t , b } ^ { \mathrm { a c t } } \right) , \qquad c _ { t , b } ^ { \mathrm { a c t } } = D _ { \mathcal { F } } ( \pmb { x } _ { t , b } ) ,\tag{14}
$$

where $D _ { \mathcal { F } }$ is the format-specific dynamic scale rule for activations. Because activation scales are computed online for every token, $D _ { \mathcal { F } }$ is kept lightweight and selects $c _ { t , b } ^ { \mathrm { a c t } }$ for each token block using an element-wise MSE criterion.

## Overall SoScale Framework

Algorithm 1 summarizes SoScale for one linear layer, and Figure 2 visualizes the same workflow by separating Hessian-aware search (2.1) from error recovery (2.2). The algorithm builds the empirical Hessian from calibration activations, optionally applies a preprocessing transform to both weights and Hessian, and then processes FP4 input blocks sequentially. For each block, scale codes are searched on the current compensated weights, the selected quantized block is committed, and its error is propagated to future blocks through the back-substitution. Scale-only baselines such as ScaleSearch (Gupta et al. 2026) and Four Over Six (Cook et al. 2026) mainly correspond to the 2.1 search stage and optimize local MSE-style objectives without Hessian information; SoScale instead scores legal scale codes with the Hessian-aware layer loss and then applies 2.2 recovery to update future blocks. The output is a target-format FP4 representation consisting of quantized payloads and legal scale codes. This weight quantization procedure is performed offline during calibration; inference uses only the stored FP4 payloads and scale codes.

Algorithm 1 SoScale for one linear layer   
Require: Full-precision weights W, calibration inputs $X ,$   
target FP4 format ${ \mathcal F } ,$ damping λ, optional transform R   
Ensure: Quantized weights $\widehat { W }$ and scale-code setting $c ^ { \star }$   
1: Compute empirical Hessian $H  X ^ { \top } X$   
2: if preprocessing transform R is used then   
3: Transform weights and Hessian: $W  W R ,$ H ←   
$R ^ { \top } H R$   
4: end if   
5: Initialize working weights $\widetilde { W }  W$ , quantized weights   
$\widehat { W }$ , and scale codes $c ^ { \star }$   
6: Compute upper-triangular factor $U ^ { \top } U = ( { \pmb { H } } + \lambda { \pmb { I } } ) ^ { - 1 }$   
7: for each FP4 block $\mathbb { Q }$ and future-block index set F do   
8: Construct legal candidate set $\mathbb { C } _ { \mathbb { Q } }$   
9: Search on weights: $c _ { \mathbb { Q } } ^ { \star }  \mathrm { a r g }$ mi $\mathsf { 1 } _ { \pmb { c } _ { \mathbb { Q } } \in \mathbb { C } _ { \mathbb { Q } } } \ell _ { \mathbb { Q } } ( \pmb { c } _ { \mathbb { Q } } )$   
10: Commit quantized block: $\widehat { W } _ { : , \mathbb { Q } } \gets Q _ { \mathcal { F } } ( \widetilde { W } _ { : , \mathbb { Q } } ; \pmb { c } _ { \mathbb { Q } } ^ { \star } )$   
11: Compute block error $\pmb { { \cal E } } _ { \mathbb { Q } }  \widetilde { \pmb { W } } _ { : , \mathbb { Q } } - \widehat { \pmb { W } } _ { : , \mathbb { Q } }$   
12: Propagate error to future blocks: $\widetilde { W } _ { : , \mathbb { F } } \ :  \ : \widetilde { W } _ { : , \mathbb { F } } \ : -$   
$\pmb { E } _ { \mathbb { Q } } \pmb { U } _ { \mathbb { Q } , \mathbb { Q } } ^ { - 1 } \pmb { U } _ { \mathbb { Q } , \mathbb { F } }$   
13: Store $c _ { \mathbb { Q } } ^ { \star }$ in $c ^ { \star }$   
14: end for   
15: return $\hat { W }$ and $c ^ { \star }$

## Experiments

We evaluate SoScale on Qwen3-family LLMs under W4A4 quantization with three hierarchical FP4 formats: MXFP4, NVFP4, and HiF4. We report downstream benchmark accuracy against representative baselines.

## Experimental Setup

Models. We evaluate SoScale on Qwen3-8B and Qwen3- 14B (Yang et al. 2025).

Quantization formats. We evaluate MXFP4, NVFP4, and HiF4, the three hierarchical FP4 formats summarized in Table 1. Our main setting is W4A4: weights are quantized by the compared weight quantizer, and activations use the online per-token FP4 rule described in the methodology. We quantize linear layers in transformer blocks, while keeping token embeddings, the LM head, and normalization layers in BF16. We also report a direct-activation setting for scalesearch methods, where activations use direct RTN quantization; this isolates the weight-side contribution while keeping the activation path closer to hardware-native FP4 execution.

Baselines. We use BF16 as the full-precision reference. RTN denotes the default format-specific round-to-nearest FP4 quantizer. GPTQ denotes GPTQ-style second-order weight quantization (Frantar et al. 2023). Since GPTQ is originally a weight-only PTQ method, in the main W4A4 comparison we pair its quantized weights with the online per-token activation rule described above. ScaleSearch denotes local block-scale search that minimizes element-wise MSE (Gupta et al. 2026). BlockRotation denotes the standalone block Hadamard preprocessing method proposed for MXFP4 (Shao et al. 2025); in our experiments, we apply the same preprocessing to MXFP4, NVFP4, and HiF4. In tables, the sufix w/ Had. denotes applying block Hadamard preprocessing before the corresponding quantization method, following QuaRot (Ashkboos et al. 2024) and block rotation (Shao et al. 2025). We use these names consistently in the result tables.

Calibration and implementation. For methods that require calibration, we sample 128 sequences of 2,048 tokens from the first WikiText shard (Merity et al. 2016), following a widely used PTQ setup (Frantar et al. 2023). For SoScale, these calibration activations are used to estimate the empirical Hessian for the Hessian-aware scale-code search. We load Qwen3 checkpoints and tokenizers from their public Hugging Face repositories, use datasets (Lhoest et al. 2021) and transformers (Wolf et al. 2020) for data preprocessing and model execution, and implement all quantization routines in PyTorch (Paszke et al. 2019).

Evaluation benchmarks. We evaluate quantized models using the Language Model Evaluation Harness (Gao et al. 2024) with batch size 64 on GSM8K (Cobbe et al. 2021), HellaSwag (Zellers et al. 2019), MMLU (Hendrycks et al. 2021), ARC-Challenge and ARC-Easy (Clark et al. 2018), BoolQ (Clark et al. 2019), and PIQA (Bisk et al. 2020). We report exact-match accuracy for GSM8K and accuracy for the remaining tasks. Unless otherwise noted, GSM8K is evaluated in a 5-shot setting and the other benchmarks are evaluated in a zero-shot setting. We report the average score across tasks and its relative ratio to the BF16 reference.

## Main Results

SoScale improves downstream accuracy. Table 2 reports downstream accuracy in the main W4A4 setting. Across both Qwen3 models and all three FP4 formats, a SoScale variant obtains the best average score in every model-format group, preserving 96.44%–97.17% of the BF16 average accuracy under MXFP4, 98.99%–99.20% under NVFP4, and 98.38%–99.40% under HiF4. Compared with the strongest non-SoScale baseline in each group, the best SoScale variant improves the average score by 0.21 to 0.77 percentage points; relative to local ScaleSearch, the gains are 0.47 to

<table><tr><td>Model</td><td>Bit</td><td>Format</td><td>Method</td><td>GSM8K</td><td>HellaSwag</td><td>MMLU</td><td>ARC-C</td><td>ARC-E</td><td>BoolQ</td><td>PIQA Avg.</td><td>Rel.</td></tr><tr><td rowspan="9"></td><td rowspan="9"></td><td rowspan="9">BF16</td><td>RTN</td><td>0.8810 0.8211</td><td>0.5713 0.5292 0.5329</td><td>0.7292 0.6718 0.6862</td><td>0.5538 0.4923</td><td>0.8371 0.7735</td><td>0.8663 0.8425</td><td>0.7628 0.7431 0.7492</td><td>1.0000 0.9381</td></tr><tr><td>GPTQ</td><td>0.8362</td><td></td><td>0.5068</td><td>0.7971</td><td>0.8520</td><td>0.7514</td><td>0.6971 0.7089</td><td>0.9540</td></tr><tr><td>BlockRotation MXFP4</td><td>0.7839</td><td>0.5329</td><td>0.6772 0.5077</td><td>0.7921</td><td>0.8440</td><td>0.7421</td><td>0.6971</td><td>0.9381</td></tr><tr><td>ScaleSearch</td><td>0.8317</td><td>0.5303</td><td>0.6802 0.5171</td><td>0.7837</td><td>0.8505</td><td>0.7459</td><td>0.7056</td><td>0.9496</td></tr><tr><td>SoScale SoScale w/ Had.</td><td>0.8575 0.8271</td><td>0.5334 0.6924 0.5455 0.7010</td><td>0.5000</td><td>0.7875</td><td>0.8517 0.8636</td><td>0.7421</td><td>0.7092</td><td>0.9545</td></tr><tr><td>RTN</td><td>0.8544</td><td></td><td>0.5137</td><td>0.8072</td><td></td><td>0.7584 0.7579</td><td>0.7166</td><td>0.9644</td></tr><tr><td>GPTQ</td><td>0.8582</td><td>0.5542 0.5515</td><td>0.7029 0.7117</td><td>0.5401 0.5247</td><td>0.8178 0.8178</td><td>0.8560 0.8584 0.7573</td><td>0.7262 0.7257</td><td>0.9773 0.9765</td></tr><tr><td>BlockRotation ScaleSearch</td><td>0.8423 0.8620</td><td>0.5439 0.5551</td><td>0.6966</td><td>0.5187</td><td>0.8093</td><td>0.8523</td><td>0.7600 0.7176</td><td>0.9657</td></tr><tr><td>SoScale SoScale w/ Had.</td><td>0.8787 0.8582</td><td>0.5556 0.5543</td><td>0.7100 0.7153</td><td>0.5324 0.5401</td><td>0.8215 0.8349</td><td>0.8621 0.8664</td><td>0.7535 0.7579 0.7628</td><td>0.7281 0.9798 0.7356 0.9899</td></tr><tr><td rowspan="5">HiF4</td><td rowspan="3"></td><td>RTN GPTQ</td><td>0.8529 0.8643</td><td>0.5555 0.5577</td><td>0.7204 0.7053 0.7109</td><td>0.5384 0.5119 0.5273</td><td>0.8161 0.8056</td><td>0.8651 0.8541 0.7622</td><td>0.7308 0.7211</td><td>0.9834 0.9704</td></tr><tr><td>BlockRotation</td><td>0.8567</td><td>0.5475</td><td>0.7058 0.5154</td><td>0.8173 0.8081</td><td>0.8630</td><td>0.7617 0.8587 0.7584</td><td>0.7289 0.7215</td><td>0.9809</td></tr><tr><td>ScaleSearch SoScale</td><td>0.8552</td><td>0.5587</td><td>0.7093 0.5213</td><td></td><td>0.8081</td><td>0.8554</td><td>0.7617 0.7242</td><td>0.9710 0.9747</td></tr><tr><td>SoScale w/ Had.</td><td>0.8696 0.8605</td><td>0.5582 0.5563</td><td>0.7154</td><td>0.5299</td><td>0.8165</td><td>0.8590 0.8694</td><td>0.7579 0.7617</td><td>0.7295 0.7310</td><td>0.9817 0.9838</td></tr><tr><td rowspan="7"></td><td rowspan="2">BF16</td><td></td><td>0.8863</td><td>0.6094</td><td>0.7163 0.7710</td><td>0.5367 0.5845</td><td>0.8161 0.8418</td><td>0.8936</td><td>0.8020</td><td>0.7698 1.0000</td></tr><tr><td>RTN GPTQ</td><td>0.8408 0.8711</td><td>0.5792</td><td>0.7243</td><td>0.5623</td><td>0.8186</td><td>0.8728</td><td>0.7671</td><td>0.7379</td></tr><tr><td rowspan="6"></td><td rowspan="2">MXFP4</td><td>BlockRotation</td><td>0.8400</td><td>0.5794 0.5759</td><td>0.7313 0.7281</td><td>0.5546 0.5469</td><td>0.8157 0.8186</td><td>0.8804 0.7753 0.8798 0.7780</td><td>0.7440 0.7382</td><td>0.9585 0.9664</td></tr><tr><td>ScaleSearch SoScale</td><td>0.8522</td><td>0.5839</td><td>0.7334 0.5631</td><td>0.8119</td><td>0.8789</td><td>0.7677</td><td>0.7416 0.7479</td><td>0.9589 0.9633</td></tr><tr><td>SoScale w/ Had. RTN</td><td>0.8726 0.8658</td><td>0.5811 0.5898</td><td>0.7321 0.7414</td><td>0.5691 0.5546</td><td>0.8215 0.8203</td><td>0.8865 0.8847</td><td>0.7726 0.7797</td><td>0.7480</td><td>0.9716 0.9717</td></tr><tr><td rowspan="2">GPTQ NVFP4</td><td>0.8757 0.8719</td><td>0.5937 0.5993</td><td>0.7496 0.7554</td><td>0.5529 0.5546</td><td>0.8224 0.8338</td><td></td><td>0.8887 0.7791 0.8850</td><td>0.7517 0.7922 0.7560</td><td>0.9765 0.9821</td></tr><tr><td>BlockRotation</td><td></td><td></td><td></td><td></td><td></td><td>0.8826</td><td>0.7480</td><td>0.9717</td></tr><tr><td rowspan="6"></td><td rowspan="3"></td><td>ScaleSearch SoScale</td><td>0.8817 0.8772</td><td>0.5847 0.5976</td><td>0.7418 0.7550</td><td>0.5478 0.5683</td><td>0.8047 0.8253</td><td>0.8856</td><td>0.7927</td><td></td><td>0.9827 0.9894</td></tr><tr><td></td><td>0.8901</td><td>0.5991</td><td>0.7580</td><td>0.5811</td><td>0.8291</td><td>0.8890</td><td>0.7862 0.7851</td><td>0.7565 0.7616</td></tr><tr><td>SoScale w/ Had.</td><td>0.8954</td><td>0.5956</td><td>0.7615</td><td>0.5802</td><td>0.8338</td><td>0.8896</td><td>0.7894</td><td>0.7636 0.9920</td></tr><tr><td rowspan="3">RTN GPTQ</td><td></td><td></td><td>0.5987</td><td>0.7564</td><td>0.5657</td><td>0.8333</td><td>0.8936</td><td>0.7922</td><td>0.7593</td><td>0.9863</td></tr><tr><td></td><td>0.8749</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.7905</td><td></td></tr><tr><td rowspan="4"></td><td rowspan="2">HiF4</td><td></td><td>0.8946</td><td>0.5993</td><td>0.7564</td><td>0.5700</td><td>0.8304</td><td>0.8896</td><td>0.7889</td><td>0.7613</td><td>0.9890</td></tr><tr><td>BlockRotation</td><td>0.8795</td><td>0.5927</td><td>0.7512</td><td>0.5794</td><td>0.8367</td><td>0.8893</td><td></td><td>0.7599</td><td>0.9871</td></tr><tr><td>ScaleSearch SoScale</td><td>0.8772</td><td>0.6007 0.5965</td><td>0.7558</td><td>0.5700</td><td>0.8375</td><td>0.8899</td><td>0.7927</td><td>0.7605</td><td>0.9880</td></tr><tr><td></td><td>SoScale w/ Had.</td><td>0.8840 0.8954</td><td>0.6000</td><td>0.7559</td><td>0.5922</td><td>0.8422</td><td>0.8911</td><td>0.7943 0.7884</td><td>0.7652 0.7631</td><td>0.9940 0.9913</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.7595</td><td>0.5700</td><td>0.8392</td><td>0.8890</td><td></td><td></td></tr></table>

Table 2: Downstream accuracy under W4A4 hierarchical FP4 quantization; Avg. averages tasks and Rel. normalizes by the BF16 average for the same model. Highlighted rows are SoScale variants, w/ Had. denotes block Hadamard preprocessing, bold marks the best Avg. in each model-format group, and underlining marks the second best.

1.10 points. The largest gains appear under MXFP4 because its E8M0 scale codes allow only coarse power-of-two adjustments: outliers can force a coarse FP4 grid that leaves dense values under-resolved, creating more room for Hessian-aware scale selection. NVFP4 and HiF4 provide stronger baselines and leave less headroom, yet SoScale still improves them, indicating that layer-level sensitivity remains useful even when the scale space is finer.

Hadamard gains are format-dependent. BlockRotation alone gives mixed improvements: it can help in some modelformat groups, but it does not consistently outperform Scale-Search and sometimes trails RTN. Combining Hadamard preprocessing with SoScale is most useful for MXFP4, where it gives the best average score for both Qwen3-8B and Qwen3-14B, and it also gives the best NVFP4 result on Qwen3-14B. However, the best row is not always the

Hadamard variant: Qwen3-8B under NVFP4 and Qwen3- 14B under HiF4 are best with SoScale alone. This format dependence is consistent with recent microscaling FP4 analysis, which highlights MXFP4’s power-of-two scale error and NVFP4’s distinct small-group behavior (Egiazarian et al. 2026). Overall, preprocessing changes the distribution presented to the quantizer, whereas SoScale selects the hardware-valid scale codes used in the final FP4 representation. This explains why rotations can help MXFP4, but do not replace Hessian-aware scale selection.

Direct activation RTN remains viable. Table 3 evaluates a deployment-oriented variant in which activations use direct RTN quantization instead of activation-side scale search. This setting keeps the activation path aligned with hardware-supported FP4 formats and avoids introducing format-specific activation search into the runtime. Under this direct activation path, the best SoScale-w variant still outperforms ScaleSearch-w in every model-format group, with gains ranging from 0.08 to 1.92 percentage points in average accuracy. Hadamard preprocessing remains useful here because the paired orthogonal transforms applied to weights and activations preserve the linear map while spreading outliers on both sides of the multiplication. This is especially helpful when activations follow a direct RTN path, where no activation-side search is available to correct outlier-induced scale choices; the clearest gains again appear for MXFP4. Some direct-RTN rows also exceed their counterparts in Table 2. This does not imply that direct activation RTN is uniformly better; rather, the online activation rule is a local heuristic and is not jointly optimized with the weight scale codes or downstream loss. When activations are already well matched to the target FP4 format, avoiding this extra activation-side adaptation can be beneficial, while SoScale continues to address the dominant weight-side scale-code error.

<table><tr><td>Model</td><td>Bit</td><td>Format Method</td><td></td><td>GSM8K</td><td>HellaSwag</td><td>MMLU</td><td>ARC-C</td><td>ARC-E</td><td>BoolQ</td><td>PIQA</td><td>Avg.</td><td>Rel.</td></tr><tr><td rowspan="5">Qwen3-8B</td><td rowspan="5">W4A4</td><td>MXFP4</td><td>ScaleSearch-w SoScale-w</td><td>0.8241 0.8317</td><td>0.5273 0.5292</td><td>0.6738 0.6856</td><td>0.5009 0.5000</td><td>0.7774 0.7997</td><td>0.8492 0.8508</td><td>0.7492 0.7405</td><td>0.7003 0.7054</td><td>0.9424 0.9492</td></tr><tr><td rowspan="4"></td><td>SoScale-w w/ Had.</td><td>0.8461</td><td>0.5438</td><td>0.6966</td><td>0.5205</td><td>0.8064</td><td>0.8635</td><td>0.7601</td><td>0.7195</td><td>0.9683</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ScaleSearch-w SoScale-w</td><td>0.8605 0.8726</td><td>0.5533 0.5542</td><td>0.7124 0.7082</td><td>0.5341 0.5384</td><td>0.8114 0.8157</td><td>0.8575 0.8602</td><td>0.7655 0.7459</td><td>0.7278 0.7279</td><td>0.9795</td></tr><tr><td>NVFP4 SoScale-w w/ Had.</td><td>0.8560</td><td>0.5541</td><td>0.7121</td><td>0.5410</td><td>0.8136</td><td>0.8657</td><td>0.7579</td><td>0.7286</td><td>0.9796 0.9806</td></tr><tr><td rowspan="5">Qwen3-14B W4A4</td><td></td><td>HiF4</td><td>ScaleSearch-w SoScale-w SoScale-w w/ Had.</td><td>0.8400 0.8613 0.8582</td><td>0.5567 0.5574 0.5591</td><td>0.7074 0.7124 0.7140</td><td>0.5060 0.5230 0.5444</td><td>0.8035 0.8144 0.8186</td><td>0.8514 0.8636 0.8700</td><td>0.7650 0.7557 0.7655</td><td>0.7186 0.7268 0.7328</td><td>0.9670 0.9781 0.9861</td></tr><tr><td rowspan="4"></td><td rowspan="2">MXFP4</td><td>ScaleSearch-w SoScale-w</td><td>0.8567</td><td>0.5795</td><td>0.7328</td><td>0.5469</td><td>0.8140</td><td>0.8670</td><td>0.7584</td><td>0.7365</td><td>0.9567</td></tr><tr><td></td><td>0.8749 0.8673</td><td>0.5762 0.5877</td><td>0.7318</td><td>0.5700</td><td>0.8249</td><td>0.8813</td><td>0.7715</td><td>0.7472</td><td>0.9707</td></tr><tr><td rowspan="2">SoScale-w w/ Had. ScaleSearch-w</td><td></td><td></td><td>0.7421</td><td>0.5555</td><td></td><td>0.8236</td><td>0.8853</td><td>0.7829 0.7492</td><td>0.9732</td></tr><tr><td></td><td>0.8901 0.8878</td><td>0.5970 0.5971</td><td>0.7569</td><td>0.5631</td><td>0.8237</td><td>0.8829</td><td>0.7753 0.7556</td><td>0.9815</td></tr><tr><td rowspan="4"></td><td></td><td>NVFP4 SoScale-w</td><td>SoScale-w w/ Had.</td><td>0.8878</td><td>0.5975</td><td>0.7584 0.7587</td><td>0.5794 0.5768</td><td>0.8249 0.8333</td><td>0.8902 0.8911</td><td>0.7829 0.7813</td><td>0.7601 0.7609</td><td>0.9874</td></tr><tr><td rowspan="3">HiF4</td><td></td><td>0.8825</td><td>0.6002</td><td>0.7528</td><td>0.5717</td><td></td><td></td><td></td><td></td><td></td><td>0.9885</td></tr><tr><td>ScaleSearch-w SoScale-w</td><td></td><td>0.8893</td><td>0.5954</td><td>0.7578</td><td>0.5657</td><td>0.8316 0.8413</td><td>0.8838 0.8917</td><td>0.7835 0.7905</td><td>0.7580 0.7617</td><td>0.9847 0.9894</td></tr><tr><td>SoScale-w w/ Had.</td><td>0.8946</td><td>0.6008</td><td>0.7609</td><td>0.5836</td><td>0.8380</td><td>0.8902</td><td>0.7954</td><td>0.7662</td><td>0.9953</td></tr></table>

Table 3: W4A4 comparison with direct RTN activation quantization. The sufix “-w” denotes variants that use weight-side scale search while keeping activations on a direct RTN path; w/ Had. denotes block Hadamard preprocessing before weight quantization. Highlighted cells denote SoScale variants, and bold values mark the best average accuracy within each modelformat group.

<table><tr><td>Format</td><td>Variant</td><td>Loss</td><td>Rec.</td><td>Avg.</td><td>Rel.</td></tr><tr><td rowspan="4">MXFP4</td><td>RTN</td><td>N/A</td><td>No</td><td>0.6971</td><td>0.9381</td></tr><tr><td>ScaleSearch</td><td>MSE</td><td>No</td><td>0.7056</td><td>0.9496</td></tr><tr><td>MSE + recovery</td><td>MSE</td><td>√</td><td>0.7061</td><td>0.9503</td></tr><tr><td>SoScale</td><td>Hessian</td><td>√</td><td>0.7092</td><td>0.9545</td></tr><tr><td rowspan="4">NVFP4</td><td>RTN</td><td>N/A</td><td>No</td><td>0.7262</td><td>0.9773</td></tr><tr><td>ScaleSearch</td><td>MSE</td><td>No</td><td>0.7281</td><td>0.9798</td></tr><tr><td>MSE + recovery</td><td>MSE</td><td>√</td><td>0.7270</td><td>0.9784</td></tr><tr><td>SoScale</td><td>Hessian</td><td>√</td><td>0.7356</td><td>0.9899</td></tr><tr><td rowspan="4">HiF4</td><td>RTN</td><td>N/A</td><td>No</td><td>0.7211</td><td>0.9704</td></tr><tr><td>ScaleSearch</td><td>MSE</td><td>No</td><td>0.7242</td><td>0.9747</td></tr><tr><td>MSE + recovery</td><td>MSE</td><td>√</td><td>0.7263</td><td>0.9774</td></tr><tr><td>SoScale</td><td>Hessian</td><td>√</td><td>0.7295</td><td>0.9817</td></tr></table>

Table 4: Qwen3-8B ablation under W4A4 quantization; Rec. denotes the 2.2 error-propagation stage in Figure 2.

Ablating search and recovery. Table 4 compares MSE scale search, MSE search with blockwise recovery, and full SoScale on Qwen3-8B under fixed FP4 formats. Adding recovery to the MSE objective gives mixed gains over Scale-Search: it is marginal under MXFP4, decreases NVFP4, and improves HiF4. In contrast, full SoScale is best for all three formats, indicating that recovery is most efective when coupled with Hessian-aware scale-code selection rather than a local MSE objective.

## Conclusion

We presented SoScale, a second-order scale-code search method for hierarchical FP4 quantization. SoScale addresses a practical failure mode of low-bit floating-point formats: range-preserving or locally MSE-optimal scales can place the FP4 grid poorly for layer outputs, especially when outliers interact with discrete scale codes. By optimizing hardwarevalid weight scale codes under a Hessian-aware layer reconstruction objective and interleaving this search with GPTQstyle error recovery, SoScale improves W4A4 quantization without changing the target FP4 representation. Experiments on Qwen3-8B and Qwen3-14B across MXFP4, NVFP4, and HiF4 show consistent gains over RTN, GPTQ, local Scale-Search, and BlockRotation baselines, with the largest headroom under MXFP4 and persistent gains for stronger formats. The direct-activation setting further shows that SoScale can improve weight quantization while keeping the activation path simple and hardware aligned. These results suggest that scale-code selection should be treated as a layer-aware optimization problem, rather than merely a local tensor-fitting rule, for accurate low-bit floating-point LLM inference.

## References

Alvarez, E.; Almog, O.; Chung, E.; Layton, S.; Stosic, D.; Krashinsky, R.; and Aubrey, K. 2025. Introducing NVFP4 for Eficient and Accurate Low-Precision Inference. https://developer.nvidia.com/blog/introducing-nvfp4-foreficient-and-accurate-low-precision-inference/. NVIDIA Technical Blog.

Ashkboos, S.; Mohtashami, A.; Croci, M. L.; Li, B.; Cameron, P.; Jaggi, M.; Alistarh, D.; Hoefler, T.; and Hensman, J. 2024. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. arXiv:2404.00456.

Bisk, Y.; Zellers, R.; Le Bras, R.; Gao, J.; and Choi, Y. 2020. PIQA: Reasoning about Physical Commonsense in Natural Language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, 7432–7439.

Clark, C.; Lee, K.; Chang, M.-W.; Kwiatkowski, T.; Collins, M.; and Toutanova, K. 2019. BoolQ: Exploring the Surprising Dificulty of Natural Yes/No Questions. In Proceedings of the 2019 Conference of the North American Chapter of the Associationfor Computational Linguistics, 2924–2936.

Clark, P.; Cowhey, I.; Etzioni, O.; Khot, T.; Sabharwal, A.; Schoenick, C.; and Tafjord, O. 2018. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457.

Cobbe, K.; Kosaraju, V.; Bavarian, M.; Chen, M.; Jun, H.; Kaiser, L.; Plappert, M.; Tworek, J.; Hilton, J.; Nakano, R.; Hesse, C.; and Schulman, J. 2021. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168.

Cook, J.; Guo, J.; Xiao, G.; Lin, Y.; Wyss, K.; Nazemi, M.; Mishra, A.; del Mundo, C.; Blankevoort, T.; and Han, S. 2026. Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling. arXiv:2512.02010.

DeepSeek-AI. 2026. DeepSeek-V4: Towards Highly Eficient Million-Token Context Intelligence.

Egiazarian, V.; Castro, R. L.; Kuznedelev, D.; Panferov, A.; Kurtic, E.; Pandit, S.; Marques, A.; Kurtz, M.; Ashkboos, S.; Hoefler, T.; and Alistarh, D. 2026. Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization. arXiv:2509.23202.

Frantar, E.; Ashkboos, S.; Hoefler, T.; and Alistarh, D. 2023. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. arXiv:2210.17323.

Gao, L.; Tow, J.; Abbasi, B.; Biderman, S.; Black, S.; DiPofi, A.; Foster, C.; Golding, L.; Hsu, J.; Le Noac’h, A.; Li, H.; McDonell, K.; Muennighof, N.; Ociepa, C.; Phang, J.; Reynolds, L.; Schoelkopf, H.; Skowron, A.; Sutawika, L.; Tang, E.; Thite, A.; Wang, B.; Wang, K.; and Zou, A. 2024. The Language Model Evaluation Harness.

Gupta, T.; Prairie, H.; Wu, X.; Abhyankar, R.; Wu, Q.; Silveria, A.; Ponnusamy, P.; Wang, J.; Athiwaratkun, B.; Song, L.; Dao, T.; Fu, D. Y.; and Sa, C. D. 2026. Search Your Block Floating Point Scales! arXiv:2605.12464.

Hendrycks, D.; Burns, C.; Basart, S.; Zou, A.; Mazeika, M.; Song, D.; and Steinhardt, J. 2021. Measuring Massive Multitask Language Understanding. In International Conference on Learning Representations.

Lhoest, Q.; Villanova del Moral, A.; Jernite, Y.; Thakur, A.; von Platen, P.; Patil, S.; Chaumond, J.; Drame, M.; Plu, J.; Tunstall, L.; Davison, J.; Sajjad, H.; Chhablani, G.; Malik, B.; Brandeis, S.; Le Scao, T.; Sanh, V.; Xu, C.; Patry, N.; McMillan-Major, A.; Schmid, P.; Gugger, S.; Delangue, C.; Matussiere, T.; Debut, L.; Bekman, S.; Cistac, P.; Goehringer, T.; Mustar, V.; Lagunas, F.; Rush, A. M.; and Wolf, T. 2021. Datasets: A Community Library for Natural Language Processing. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 175–184.

Lin, J.; Tang, J.; Tang, H.; Yang, S.; Chen, W.-M.; Wang, W.-C.; Xiao, G.; Dang, X.; Gan, C.; and Han, S. 2024. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. Proceedings of machine learning and systems, 6: 87–100.

Liu, Z.; Zhao, C.; Fedorov, I.; Soran, B.; Choudhary, D.; Krishnamoorthi, R.; Chandra, V.; Tian, Y.; and Blankevoort, T. 2025. SpinQuant: LLM quantization with learned rotations. arXiv:2405.16406.

Luo, Y.; Huang, J.; Cheng, Y.; Yu, Z.; Tang, K.; Ma, X.; Wang, X.; Tong, A.; Hu, G.; Xu, Y.; Taghian, M.; Wu, P.; Li, G.; Peng, Y.; Hu, T.; Chen, M.; Mi, M. B.; Liu, H.; Zhou, X.; Wang, J.; Lin, Q.; and Liao, H. 2026. HiFloat4 Format for Language Model Inference. arXiv:2602.11287.

Meng, H.; Luo, Y.; Zhao, Y.; Liu, W.; Zhang, P.; and Ma, X. 2026. ARCQuant: Boosting NVFP4 Quantization with Augmented Residual Channels for LLMs. arXiv:2601.07475.

Merity, S.; Xiong, C.; Bradbury, J.; and Socher, R. 2016. Pointer Sentinel Mixture Models. arXiv:1609.07843.

OpenAI. 2026. ChatGPT.

Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.; Chanan, G.; Killeen, T.; Lin, Z.; Gimelshein, N.; Antiga, L.; Desmaison, A.; Kopf, A.; Yang, E.; DeVito, Z.; Raison, M.; Tejani, A.; Chilamkurthy, S.; Steiner, B.; Fang, L.; Bai, J.; and Chintala, S. 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems, volume 32.

Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents.

Rouhani, B. D.; Zhao, R.; More, A.; Hall, M.; Khodamoradi, A.; Deng, S.; Choudhary, D.; Cornea, M.; Dellinger, E.; Denolf, K.; et al. 2023. Microscaling data formats for deep learning. arXiv preprint arXiv:2310.10537.

Shao, W.; Chen, M.; Zhang, Z.; Xu, P.; Zhao, L.; Li, Z.; Zhang, K.; Peng, G.; Qiao, Y.; and Luo, P. 2024. Omniquant: Omnidirectionally calibrated quantization for large language models. In International Conference on Learning Representations, volume 2024, 45472–45496.

Shao, Y.; Wang, P.; Chen, Y.; Xu, C.; Wei, Z.; and Cheng, J. 2025. Block Rotation is All You Need for MXFP4 Quantization. arXiv:2511.04214.

Wolf, T.; Debut, L.; Sanh, V.; Chaumond, J.; Delangue, C.; Moi, A.; Cistac, P.; Rault, T.; Louf, R.; Funtowicz, M.; Davison, J.; Shleifer, S.; von Platen, P.; Ma, C.; Jernite, Y.; Plu, J.; Xu, C.; Scao, T. L.; Gugger, S.; Drame, M.; Lhoest, Q.; and

Yu, B.; Gao, C.; Huang, C.; Lv, C.; Zheng, C.; Liu, D.; Zhou,

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.;

Rush, A. M. 2020. Transformers: State-of-the-Art Natural Language Processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 38–45.

Xiao, G.; Lin, J.; Seznec, M.; Wu, H.; Demouth, J.; and Han, S. 2023. Smoothquant: Accurate and eficient post-training quantization for large language models. In International conference on machine learning, 38087–38099. PMLR.

F.; Huang, F.; Hu, F.; Ge, H.; Wei, H.; Lin, H.; Tang, J.; Yang,

J.; Tu, J.; Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Zhou, J.;

Lin, J.; Dang, K.; Bao, K.; Yang, K.; Yu, L.; Deng, L.; Li,

M.; Xue, M.; Li, M.; Zhang, P.; Wang, P.; Zhu, Q.; Men,

R.; Gao, R.; Liu, S.; Luo, S.; Li, T.; Tang, T.; Yin, W.; Ren,

X.; Wang, X.; Zhang, X.; Ren, X.; Fan, Y.; Su, Y.; Zhang,

Y.; Zhang, Y.; Wan, Y.; Liu, Y.; Wang, Z.; Cui, Z.; Zhang,

Z.; Zhou, Z.; and Qiu, Z. 2025. Qwen3 Technical Report. arXiv:2505.09388.

Zellers, R.; Holtzman, A.; Bisk, Y.; Farhadi, A.; and Choi, Y. 2019. HellaSwag: Can a Machine Really Finish Your Sentence? In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics, 4791–4800.

Zhao, P.; and Yuan, X. 2025. GANQ: GPU-Adaptive Non-Uniform Quantization for Large Language Models. In International Conference on Machine Learning, 77911–77925. PMLR.