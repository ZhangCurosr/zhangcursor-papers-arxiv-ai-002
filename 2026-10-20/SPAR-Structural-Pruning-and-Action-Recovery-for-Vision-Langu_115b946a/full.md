# SPAR: Structural Pruning and Action Recovery for Vision-Language-Action Models

Shenli Zheng<sup>1</sup>, Zhou Fang<sup>1</sup>, Yuan Du<sup>1</sup>, Yijiang Liu<sup>2</sup>, Li Du<sup>1,∗</sup> <sup>1</sup>Nanjing University, Nanjing, China

<sup>2</sup>Nanjing University of Information Science and Technology, Nanjing, China shenlizheng@smail.nju.edu.cn, zfang@nju.edu.cn yuandu@nju.edu.cn, liuyijiang@nuist.edu.cn <sup>∗</sup>Corresponding author: ldu@nju.edu.cn

Model Compression with PAT and Policy Recovery with RL

![](images/4b7f84ab389f3afc81b73bcd402ca53d66a8464bdadb6f4d8a6105c9235315c8.jpg)  
Fig. 1. Model Compression and Policy Recovery on LIBERO. (a) Average success rates before and after RL, with gains shown in green (percentage points). (b) Parameter counts of the pruned models and the unpruned baseline (right), broken down into the LLM backbone and remaining components. With 0.09B LLM parameters (1.12B total), SPAR reaches 93.45% success after RL, an 8.90-percentage-point gain over its PAT-only counterpart.

Abstract— The limited onboard computing resources of robots make the inference demands of vision-language-action (VLA) models costly to accommodate. Reducing these demands through aggressive compression, however, can degrade manipulation performance beyond what supervised fine-tuning alone recovers. We present SPAR (Structural Pruning and Action Recovery), which simultaneously prunes the hidden and feedforward intermediate dimensions of the large language model (LLM) backbone through pruning-aware tuning, then applies reinforcement learning (RL) to recover policy performance. Applied to $\pi _ { 0 . 5 } ,$ , SPAR reduces the language backbone from 2.34B to 0.09B parameters and the full model from 3.37B to 1.12B, removing approximately 96% and 67% of their parameters, respectively. At this compression level, RL raises average success across the four LIBERO suites from 84.55% to 93.45%, an 8.90-percentage-point gain. The compressed policy achieves a 1.98× full-policy inference speedup over the unpruned model. These results identify substantial intermediatedimension redundancy and show that post-pruning RL can recover manipulation performance in compact policies.

## I. INTRODUCTION

Vision-language-action (VLA) models such as RT-2 [1], OpenVLA [2], and $\pi _ { 0 . 5 }$ [3] transfer pretrained representations to robotic manipulation, but their inference demands strain limited onboard compute. In $\pi _ { 0 . 5 } ,$ the LLM accounts for 2.34B of 3.37B parameters. This motivates a central question: how much language-model capacity do manipulation tasks require, and how can a smaller policy retain useful capabilities?

Efficiency approaches include compact policies such as SmolVLA [4] and NORA [5], preference-based adaptation in NORA-1.5 [6], and synthetic pretraining in GraspVLA [7]. For existing policies, SP-VLA [8] and DART [9] reduce tokens, while EfficientVLA [10] and MoLe-VLA [11] reduce layer computation. Token reduction alone leaves backbone weights intact. EcoVLA [12] dynamically selects channels and attention heads; we instead permanently reduce backbone dimensions to produce a smaller dense model.

Permanent compression introduces a second challenge: recovering manipulation capabilities lost when parameters are removed. RLRC [13] combines structured pruning with supervised fine-tuning (SFT) and reinforcement learning (RL), while SimpleVLA-RL [14] demonstrates that interaction can improve policies beyond SFT. These results motivate studying how the allocation of a small parameter budget and adaptation during pruning affect subsequent recovery. The hidden dimension couples representations across Transformer blocks, whereas the intermediate dimension controls expansion within feed-forward layers. Their different roles motivate separate retention targets for the two dimensions.

![](images/8375b0ac0734fe4c354ad1e0e2fe67d097f988a342af88992095e77faa44ea83.jpg)  
Fig. 2. Overview of the SPAR Framework. Top: SPAR compresses the LLM backbone of a VLA policy, which integrates visual and language inputs to guide the action expert. Bottom: (1) Through pruning-aware tuning, SPAR jointly learns channel masks for the hidden dimension and the FFN intermediate dimension while adapting the policy to manipulation tasks. (2) These masks guide structural pruning, which removes inactive channels to form a smaller dense backbone. (3) After supervised fine-tuning of the compact policy, RL recovery further improves task performance through simulated interaction, updating the action expert and value head while keeping the pruned LLM backbone frozen.

We introduce SPAR (Structural Pruning and Action Recovery), which couples dimension-aware pruning with postpruning RL to recover task performance (Fig. 2). Building on pruning-aware tuning (PAT) [15], SPAR jointly learns masks for the hidden and intermediate dimensions with separate retention targets and a regularized mask-learning objective. Channel selection and supervised adaptation are performed together, followed by physical pruning and fine-tuning of the reduced model. The resulting compact policy provides the initialization for RL, which further improves manipulation performance through interaction.

Our evaluation on LIBERO [16] and LIBERO-Plus [17] examines both task success and robustness under compression. Pruning only the intermediate dimension preserves high task success, while additional hidden-dimension pruning at a fixed intermediate-dimension retention target causes larger losses that RL can substantially mitigate. As a representative result, SPAR achieves 93.45% average LIBERO success with a 0.09B language backbone (1.12B total), approximately 4% of the original LLM size (Fig. 1). These findings support dimension-aware compression and recovery as complementary tools for compact robotic policies.

Our main contributions are:

• A dual-dimension pruning framework that jointly learns masks for the hidden and intermediate dimensions with separate retention targets and a regularized mask-learning objective.

• An empirical analysis of dimension sensitivity: pruning only the intermediate dimension preserves high task success, while additional hidden-dimension pruning causes larger performance losses in our evaluated tasks.

• An evaluation of recovery and efficiency across compression levels, showing improvements in task success and robustness after RL, together with measured fullpolicy inference acceleration.

## II. RELATED WORK

## A. Vision-Language-Action (VLA) models

VLA models connect pretrained vision-language representations to robotic control. OpenVLA [2] predicts discretized actions autoregressively, whereas OpenVLA-OFT [18] uses parallel decoding and action chunking. CogACT [19] separates action generation into a diffusion expert. The π<sub>0</sub> family [20] couples a PaliGemma backbone [21] with a flow-matching action expert; π [3] adds semantic subtask prediction. TinyVLA [22] instead combines a compact multimodal backbone with a diffusion decoder.

## B. Pruning for VLA models

LLM compression includes weight sparsification in SparseGPT [23], coupled-structure removal in LLM-Pruner [24], and row/column deletion to form smaller dense matrices in SliceGPT [25]. PAT [15] jointly performs hiddendimension pruning and fine-tuning through trainable channel masks and Hybrid Sparsification Modules. For VLA models, DART [9] and SP-VLA [8] reduce visual tokens; EfficientVLA [10] removes low-impact layers, and MoLe-VLA [11] dynamically skips layers. EcoVLA [12] adaptively selects channels and attention heads with parallel sparsity updates and custom kernels. Building on PAT, SPAR learns masks for both the hidden and intermediate dimensions with separate retention targets, physically reduces the dense backbone, and applies RL recovery. Our evaluation examines the effects of compression on both manipulation success and robustness.

## C. Reinforcement learning for VLA models

RL for VLA models includes both offline and online approaches. Offline RL learns from a fixed dataset without additional environment interaction, whereas online RL collects new experience as training proceeds. The RECAP recipe for $\pi _ { 0 . 6 } ^ { * }$ [26] starts with offline RL pretraining, then alternates robot data collection with offline updates using advantage-conditioned policies. For online VLA training, SimpleVLA-RL [14] optimizes policies through environment interaction with outcome-based rewards, while RLinf [27] provides scalable training infrastructure. In the compression setting, RLRC [13] combines structured pruning, supervised fine-tuning, and RL to recover policy performance.

## D. Manipulation Benchmarks

LIBERO [16] evaluates knowledge transfer across language-conditioned manipulation tasks. ManiSkill2 [28] offers 20 task families spanning stationary and mobile manipulation, single- and dual-arm robots, and rigid and soft objects. Robustness-oriented extensions probe behavior beyond standard task completion: LIBERO-PRO [29] perturbs manipulated objects, initial states, instructions, and environments to examine sensitivity to memorized task conditions. LIBERO-Plus [17] evaluates seven perturbation dimensions, including camera viewpoints, robot initial states, and scene appearance. We use LIBERO to evaluate task success and LIBERO-Plus to assess robustness under perturbations after compression.

## III. METHODOLOGY

SPAR combines structural pruning with post-pruning RL (Fig. 2). We adapt pruning-aware tuning (PAT) [15] to jointly learn hidden and intermediate channel masks with separate retention targets. After mask learning, we physically remove inactive channels and finetune the compact policy before RL recovery. In our experiments, PAT only denotes this pruning and supervised adaptation stage within SPAR, and PAT+RL denotes the complete framework.

a) PAT background: PAT’s Hybrid Sparsification Module [15] combines channel selection with adaptation of pretrained weights. For a linear map $\textbf { W } \in \ \mathbb { R } ^ { d _ { o } \times d _ { i } }$ , its forward operation can be written as

$$
\mathbf { Z } = \mathrm { d i a g } ( \mathbf { M } ) \mathbf { D } \mathbf { W } \mathbf { X } ,\tag{1}
$$

where M selects output channels and D is the knowledgetransformation operator. The original formulation regularizes channel sparsity and the low-rank transformation’s orthogonality while optimizing the supervised objective. This joint optimization allows pretrained representations to adapt as channel selection changes. SPAR adopts the pruning-aware training principle and introduces separate retention targets for the hidden and intermediate dimensions, a sigmoid mask schedule, and a differentiable count penalty. The following subsections formulate these channel-selection components and their role in preparing the compact policy for recovery.

## A. Dual-Dimension Structural Pruning

The hidden dimension determines the width of the representation passed between Transformer blocks, whereas the intermediate dimension controls the expansion within each feed-forward network (FFN). Pruning the hidden dimension therefore requires coordinated changes to the attention and FFN projections to preserve residual connections. In contrast, pruning the intermediate dimension reduces the FFN expansion while leaving the residual-stream width unchanged. SPAR uses separate retention targets for the two dimensions to control these distinct aspects of model capacity.

Let $d _ { m o d e l }$ and $d _ { f f n }$ denote the hidden dimension and the FFN intermediate dimension, respectively. We learn a mask for each dimension, $\mathbf { M } _ { h i d } \in [ 0 , 1 ] ^ { d _ { m o d e l } }$ and ${ \bf { M } } _ { i n t } \in$ $[ 0 , 1 ] ^ { d _ { f f n } }$ , with retention targets $\rho _ { h i d }$ and $\rho _ { i n t }$ . Both masks are shared across layers. Writing their diagonal forms as $\mathbf { D } _ { h } = \mathrm { d i a g } ( \mathbf { M } _ { h i d } )$ and $\mathbf { D } _ { i } = \mathrm { d i a g } ( \mathbf { M } _ { i n t } )$ , we apply the hidden-dimension mask to the attention projections:

$$
\begin{array} { c } { { \mathbf W _ { \{ Q , K , V \} } ^ { \prime } = \mathbf W _ { \{ Q , K , V \} } \mathbf D _ { h } , } } \\ { { \mathbf W _ { O } ^ { \prime } = \mathbf D _ { h } \mathbf W _ { O } . } } \end{array}\tag{2}
$$

The FFN projections depend on both dimensions and are masked as

$$
\begin{array} { r l } { \mathbf { W } _ { \{ u p , g a t e \} } ^ { \prime } = \mathbf { D } _ { i } \mathbf { W } _ { \{ u p , g a t e \} } \mathbf { D } _ { h } , } & { { } } \\ { \mathbf { W } _ { d o w n } ^ { \prime } = \mathbf { D } _ { h } \mathbf { W } _ { d o w n } \mathbf { D } _ { i } . } & { { } } \end{array}\tag{3}
$$

In $\mathbf { W } _ { u p }$ and $\mathbf { W } _ { g a t e }$ , rows correspond to the intermediate dimension and columns to the hidden dimension; this arrangement is reversed in $\mathbf { W } _ { d o w n }$ . Sharing the hiddendimension mask across blocks ensures that their residual connections use the same retained channels. Once the masks are binarized, we remove the rows and columns associated with inactive channels. The resulting backbone uses smaller dense weight matrices while preserving the visual-token sequence and Transformer depth.

## B. Mask Learning and Regularization

At low retention, mask learning must select a small set of channels while preserving task-relevant representations. We combine continuous gates for gradient-based adaptation, a count penalty for the retention target, and binary regularizers for decisive channel selection. This formulation allows the supervised objective to guide channel importance while progressively preparing the masks for physical pruning.

a) Soft-to-hard masking: For either mask, trainable mask logits $\mathbf { W } _ { M }$ are initialized to +1.5 for an initial candidate set of retained channels and −1.5 for the remaining channels. These assignments remain learnable during supervised adaptation. During training,

$$
\mathbf { M } ( s ) = \sigma ( \tau ( s ) \mathbf { W } _ { M } ) ,\tag{4}
$$

where $\sigma$ is the logistic sigmoid and s is the training step. The inverse temperature increases linearly from 1 to 10:

$$
\tau ( s ) = 1 + 9 \mathrm { m i n } \biggl ( \frac { s } { r _ { s t o p } s _ { 0 } } , 1 \biggr ) ,\tag{5}
$$

where $s _ { 0 }$ is the reference training horizon for the schedule and $r _ { s t o p }$ controls its endpoint (typically 0.8). The inverse temperature reaches 10 at step $r _ { s t o p } s _ { 0 }$ , progressively sharpening the mask toward binary values. After mask learning, we set $\mathbf { M } _ { b i n } = \mathbb { 1 } [ \mathbf { W } _ { M } \geq 0 ]$ and use the resulting channel selection for physical pruning.

The gate’s derivative makes the optimization trade-off explicit:

$$
\frac { \partial M _ { j } } { \partial W _ { M , j } } = \tau ( s ) M _ { j } ( 1 - M _ { j } ) .\tag{6}
$$

The supervised loss updates channel selection through these gradients, with larger responses near the decision boundary and smaller responses near the binary endpoints. Annealing progressively sharpens the masks as the policy adapts. Once the masks are thresholded, the retained channel set is fixed and supervised fine-tuning adapts the reduced dense matrices to the manipulation tasks.

b) Pruning objective: We use three regularizers to encourage target retention and binary channel selection. For a mask with d entries and target active-channel count $N _ { t a r g e t }$ the count penalty is

$$
\mathcal { L } _ { c o u n t } = \lambda _ { 1 } \left( \frac { \sum _ { j = 1 } ^ { d } M _ { j } - N _ { t a r g e t } } { N _ { t a r g e t } } \right) ^ { 2 } .\tag{7}
$$

The soft count $\textstyle \sum _ { j } M _ { j }$ replaces a hard active-channel count during optimization. Its derivative with respect to any mask entry is

$$
\frac { \partial \mathcal { L } _ { c o u n t } } { \partial M _ { j } } = \frac { 2 \lambda _ { 1 } } { N _ { t a r g e t } ^ { 2 } } \left( \sum _ { k = 1 } ^ { d } M _ { k } - N _ { t a r g e t } \right) .\tag{8}
$$

Thus, a count above the target encourages smaller mask values, and a count below it encourages larger values. The supervised loss guides the selection of individual channels, while this penalty encourages the aggregate soft count to approach the target. To promote near-binary channel assignments, we add entropy regularization:

$$
\begin{array} { l } { \displaystyle { \mathcal { L } _ { e n t r o p y } = - \lambda _ { 2 } \sum _ { j = 1 } ^ { d } \big [ M _ { j } \ln ( M _ { j } + \epsilon ) } } \\ { \displaystyle { ~ + ( 1 - M _ { j } ) \ln ( 1 - M _ { j } + \epsilon ) \big ] , } } \end{array}\tag{9}
$$

The discreteness term further encourages separation from the threshold at 0.5:

$$
\mathcal { L } _ { d i s c } = \lambda _ { 3 } \left( 0 . 5 - \sum _ { j = 1 } ^ { d } | M _ { j } - 0 . 5 | \right) ,\tag{10}
$$

where $\lambda _ { 1 } , \lambda _ { 2 } .$ , and $\lambda _ { 3 }$ weight the terms and $\epsilon \ : = \ : 1 0 ^ { - 7 }$ stabilizes the logarithms. The training objective combines these penalties with the supervised policy fine-tuning loss $\mathcal { L } _ { S F T } \mathrm { : \quad }$

$$
\begin{array} { r } { \mathcal { L } _ { t o t a l } = \mathcal { L } _ { S F T } + \mathcal { L } _ { c o u n t } } \\ { + \mathcal { L } _ { e n t r o p y } + \mathcal { L } _ { d i s c } . } \end{array}\tag{11}
$$

Together, these soft penalties encourage near-binary masks with soft counts close to the target channel budgets, while the supervised loss guides task-relevant channel selection. We then binarize the masks, physically remove inactive channels, and continue supervised fine-tuning of the pruned model. Table VI evaluates each regularizer when pruning only the intermediate dimension.

c) Mask-budget convergence: Fig. 3 shows $\pi _ { 0 . 5 }$ mask learning on LIBERO at 10% hidden and intermediate retention targets. Both soft counts are within 0.05% of their aligned targets by step 3000.

![](images/5569027e86147a23a6f1740a4130cb20f53ba55924187d0261c0405384182fbf.jpg)  
Fig. 3. Soft-mask budget convergence. Hidden and intermediate soft counts approach their respective 32-channel-aligned targets of 224 and 1664 (horizontal dotted lines). The vertical line marks physical pruning at step 6000.

## C. Post-Pruning RL Recovery

After mask learning, physical pruning, and supervised fine-tuning, the compact policy may still exhibit reduced task success. We therefore use RL to improve manipulation behavior through simulated interaction, building on the representations retained during compression.

We use Proximal Policy Optimization (PPO) [30] with RLinf [27] for parallel simulation rollouts and distributed policy updates. A sparse binary reward assigns 1 to successful task completion and 0 otherwise, providing a direct training signal without task-specific dense reward shaping.

TABLE I  
Effect of FFN Intermediate-Dimension Pruning on LIBERO Performance. We vary the FFN intermediate-dimension retention target in $\pi _ { 0 . 5 }$ while keeping the hidden dimension unchanged, and evaluate each policy after SPAR’s PAT stage without RL. The ratios $\rho _ { h i d } / \rho _ { i n t }$ specify retention targets for the hidden and intermediate dimensions. Parameter counts are in billions, and pruning ratios indicate the approximate percentages of parameters removed from the full model and LLM relative to the unpruned baseline. Success rates are reported as percentages, with $\operatorname { A v g } .$ denoting the mean across the four suites and bold highlighting the best score in each column.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td rowspan="2"> $\rho _ { h i d } / \rho _ { i n t }$ </td><td rowspan="2">Params (B) Total / LLM</td><td rowspan="2">Speedup (↑)</td><td colspan="2">Pruning Ratio</td><td rowspan="2">Spatial</td><td rowspan="2">Object</td><td rowspan="2">Goal</td><td rowspan="2">Long</td><td rowspan="2">Avg.</td></tr><tr><td>Model</td><td>LLM</td></tr><tr><td>π0.5</td><td>baseline</td><td>100% / 100%</td><td>3.37 / 2.34</td><td>1.00</td><td>0%</td><td>0%</td><td>98.80</td><td>98.80</td><td>97.40</td><td>92.20</td><td>96.80</td></tr><tr><td rowspan="6">π0.5</td><td rowspan="6">PAT only</td><td>100% / 90%</td><td>3.20 / 2.17</td><td>1.05</td><td>5%</td><td>7%</td><td>99.80</td><td>98.20</td><td>96.60</td><td>94.60</td><td>97.30</td></tr><tr><td>100% / 70%</td><td>2.86 / 1.83</td><td>1.17</td><td>15%</td><td>22%</td><td>98.40</td><td>98.00</td><td>98.00</td><td>96.20</td><td>97.65</td></tr><tr><td>100% / 50%</td><td>2.52 /  1.49</td><td>1.33</td><td>25%</td><td>36%</td><td>98.40</td><td>98.60</td><td>98.20</td><td>94.40</td><td>97.40</td></tr><tr><td>100% / 30%</td><td>2.19 / 1.16</td><td>1.52</td><td>35%</td><td>51%</td><td>98.80</td><td>98.20</td><td>97.80</td><td>94.60</td><td>97.35</td></tr><tr><td>100% / 20%</td><td>2.02 / 0.99</td><td>1.65</td><td>40%</td><td>58%</td><td>97.80</td><td>98.00</td><td>97.40</td><td>93.40</td><td>96.65</td></tr><tr><td>100% / 10%</td><td>1.85 / 0.82</td><td>1.73</td><td>45%</td><td>65%</td><td>97.40</td><td>97.60</td><td>96.80</td><td>94.00</td><td>96.45</td></tr></table>

TABLE II

LIBERO Performance of SPAR and Comparison Methods. PAT only denotes pruning and supervised adaptation, and PAT+RL adds RL recovery. Notation follows Table I. Parentheses show RL gains in percentage points. Bold marks the best success rates within each PAT group. EcoVLA speedups are reported on RTX 3090 with custom Triton kernels [12], whereas SPAR uses TensorRT on Jetson AGX Orin. These speedups are not same-platform comparisons.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td rowspan="2">ρhid/ρint</td><td rowspan="2">Params (B) Total / LLM</td><td rowspan="2">Speedup (↑)</td><td colspan="2">Pruning Ratio</td><td rowspan="2"></td><td rowspan="2">Spatial Object</td><td rowspan="2">Goal</td><td rowspan="2">Long</td><td rowspan="2">Avg. (∆)</td></tr><tr><td>Model</td><td>LLM</td></tr><tr><td colspan="10">Compact VLA Models</td></tr><tr><td>NORA [5]</td><td></td><td></td><td>3.00 / -</td><td>一</td><td></td><td></td><td>92.20</td><td>95.40</td><td>89.40</td><td>74.60</td><td>87.90</td></tr><tr><td>NORA-1.5 [6]</td><td></td><td>1</td><td>3.00 / -</td><td>一</td><td>一</td><td></td><td>98.00</td><td>96.00</td><td>95.40</td><td>90.50</td><td>95.00</td></tr><tr><td>SmolVLA [4]]</td><td></td><td></td><td>2.25 / –</td><td>一</td><td>一</td><td></td><td>93.00</td><td>94.00</td><td>91.00</td><td>77.00</td><td>88.75</td></tr><tr><td>GraspVLA [7]</td><td></td><td>一</td><td>1.80 / -</td><td>一</td><td>一</td><td></td><td></td><td>94.10</td><td>91.20</td><td>82.00</td><td>89.10</td></tr><tr><td colspan="10">VLA Pruning Methods</td></tr><tr><td></td><td>EcoVLA (25%)</td><td>一</td><td>1</td><td>1.31</td><td>一</td><td>1</td><td>98.20</td><td>98.60</td><td>98.40</td><td>91.60</td><td>96.70</td></tr><tr><td>π0.5</td><td>EcoVLA (37.5%)</td><td></td><td></td><td>1.46</td><td>一</td><td>一</td><td>97.80</td><td>98.40</td><td>96.80</td><td>87.00</td><td>95.00</td></tr><tr><td>π0.5</td><td>baseline</td><td>100% / 100%</td><td>3.37 / 2.34</td><td>1.00</td><td>0%</td><td>0%</td><td>98.80</td><td>98.80</td><td>97.40</td><td>92.20</td><td>96.80</td></tr><tr><td rowspan="5">π0.5</td><td rowspan="5">PAT only</td><td>90% /  10%</td><td>1.78 / 0.74</td><td>1.78</td><td>47%</td><td>68%</td><td>97.80</td><td>96.00</td><td>97.20</td><td>92.60</td><td>95.90</td></tr><tr><td>70% /  10%</td><td>1.61 / 0.58</td><td>1.81</td><td>52%</td><td>75%</td><td>95.80</td><td>94.60</td><td>95.60</td><td>89.20</td><td>93.80</td></tr><tr><td>50% / 10%</td><td>1.44 / 0.41</td><td>1.89</td><td>57%</td><td>82%</td><td>94.00</td><td>95.20</td><td>94.00</td><td>86.00</td><td>92.30</td></tr><tr><td>30% / 10%</td><td>1.29 / 0.26</td><td>1.95</td><td>62%</td><td>89%</td><td>94.00</td><td>90.60</td><td>94.40</td><td>83.80</td><td>90.70</td></tr><tr><td>10% / 10%</td><td>1.12 / 0.09</td><td>1.98</td><td>67%</td><td>96%</td><td>94.20</td><td>82.80</td><td>86.20</td><td>75.00</td><td>84.55</td></tr><tr><td>π0.5</td><td>baseline+RL 100% / 100%</td><td>3.37 / 2.34</td><td>1.00</td><td></td><td>0%</td><td>0% 99.20</td><td>99.60</td><td>99.20</td><td>92.80</td><td></td><td>97.70(+0.90)</td></tr><tr><td rowspan="5">π0.5</td><td rowspan="5">PAT+RL</td><td>90% /  10%</td><td>1.78 / 0.74</td><td>1.78</td><td>47%</td><td>68%</td><td>98.40</td><td>99.60</td><td>97.40</td><td>95.80</td><td>97.80(+1.90)</td></tr><tr><td>70% /  10%</td><td>1.61 / 0.58</td><td>1.81</td><td>52%</td><td>75%</td><td>98.60</td><td>99.80</td><td>96.80</td><td>95.00</td><td>97.55 (+3.75)</td></tr><tr><td>50% /  10%</td><td>1.44 / 0.41</td><td>1.89</td><td>57%</td><td>82%</td><td>97.80</td><td>99.60</td><td>96.60</td><td>92.00</td><td>96.50 (+4.20)</td></tr><tr><td>30% / 10%</td><td>1.29 / 0.26</td><td>1.95</td><td>62%</td><td>89%</td><td>95.40</td><td>98.60</td><td>95.60</td><td>92.20</td><td>95.45 (+4.75)</td></tr><tr><td>10% / 10%</td><td>1.12 / 0.09</td><td>1.98</td><td>67%</td><td>96%</td><td>97.00</td><td>99.20</td><td>93.00</td><td>84.60</td><td>93.45 (+8.90)</td></tr></table>

Rewards and log-probabilities are aggregated at the action chunk level. PPO maximizes the clipped policy surrogate

$$
\begin{array} { r } { J _ { c l i p } ( \theta ) = \mathbb { E } _ { t } \big [ \operatorname* { m i n } \big ( r _ { t } ( \theta ) \hat { A } _ { t } , ~ } \\ { \mathrm { c l i p } ( r _ { t } ( \theta ) , 1 - \delta , 1 + \delta ) \hat { A } _ { t } \big ) \big ] , } \end{array}\tag{12}
$$

where $s _ { t }$ is the policy’s multimodal observation, $a _ { t }$ is an action chunk, and $r _ { t } ( \theta ) = \pi _ { \theta } ( a _ { t } \mid s _ { t } ) / \pi _ { \theta _ { o l d } } ( a _ { t } \mid s _ { t } )$ compares its probability under the current and rollout policies. The normalized generalized advantage estimate ${ \hat { A } } _ { t }$ guides the policy update, while δ controls the clipping range. Using taskcompletion feedback, the actor–critic loss trains the action expert and value head while keeping the pruned backbone frozen. The architecture remains fixed; paired evaluations before and after RL measure improvement over PAT only.

## IV. EXPERIMENTS

a) Tasks and Training Protocol: We train one policy per configuration across all 40 tasks in LIBERO’s Spatial, Object, Goal, and Long (LIBERO-10) suites [16], covering spatial, object, goal, and long-horizon variation. Using OpenPI’s PyTorch $\pi _ { 0 . 5 }$ implementation [31] with a Gemma 2B backbone, we vary retention targets for the hidden and intermediate dimensions and report suite success rates and their mean.

b) Baselines: We compare with unpruned $\pi _ { 0 . 5 }$ before and after RL and published results for NORA [5], NORA-

TABLE III  
Robustness of Unpruned and Pruned $\pi _ { 0 . 5 }$ on LIBERO-Plus [17]. For each configuration and RL setting, we evaluate the same checkpoint on LIBERO and LIBERO-Plus. Retention targets, parameter counts, and pruning ratios follow Table II. Scores are percentages, with cross-suite results by perturbation and the aggregate under Overall. Parentheses show RL gains in percentage points. Bold highlights the best pruned score within each RL setting.
<table><tr><td>Model</td><td>Method</td><td> $\rho _ { h i d } / \rho _ { i n t }$ </td><td>Camera</td><td>Robot</td><td>Language</td><td>Light</td><td>Background</td><td>Noise</td><td>Layout</td><td>Overall (∆)</td></tr><tr><td>π0.5</td><td>baseline</td><td>100% / 100%</td><td>71.04</td><td>74.77</td><td>85.82</td><td>95.80</td><td>94.89</td><td>86.51</td><td>85.84</td><td>83.98</td></tr><tr><td rowspan="5">π0.5</td><td rowspan="5">PAT only</td><td>90% /  10%</td><td>60.60</td><td>54.65</td><td>68.77</td><td>92.03</td><td>87.27</td><td>76.89</td><td>78.43</td><td>72.68</td></tr><tr><td>70% / 10%</td><td>59.60</td><td>41.23</td><td>56.60</td><td>87.83</td><td>79.65</td><td>75.27</td><td>70.49</td><td>65.82</td></tr><tr><td>50% / 10%</td><td>56.66</td><td>32.58</td><td>52.37</td><td>82.14</td><td>71.47</td><td>66.83</td><td>62.95</td><td>59.35</td></tr><tr><td>30% /  10%</td><td>58.47</td><td>30.19</td><td>41.77</td><td>82.84</td><td>65.43</td><td>64.21</td><td>59.15</td><td>56.08</td></tr><tr><td>10% /  10%</td><td>59.47</td><td>24.00</td><td>31.36</td><td>76.71</td><td>48.98</td><td>64.15</td><td>34.43</td><td>47.46</td></tr><tr><td>π0.5</td><td>baseline+RL</td><td>100% / 100%</td><td>69.04</td><td>77.94</td><td>84.26</td><td>94.92</td><td>96.19</td><td>88.82</td><td>86.69</td><td>84.45 (+0.47)</td></tr><tr><td rowspan="5">π0.5</td><td rowspan="5">PAT+RL</td><td>90% /  10%</td><td>65.92</td><td>62.00</td><td>76.06</td><td>95.27</td><td>91.82</td><td>82.01</td><td>83.28</td><td>78.20 (+5.52)</td></tr><tr><td>70% /  10%</td><td>64.98</td><td>52.00</td><td>64.87</td><td>93.78</td><td>89.41</td><td>82.32</td><td>74.95</td><td>73.14 (+7.32)</td></tr><tr><td>50% /  10%</td><td>61.79</td><td>41.68</td><td>58.04</td><td>89.14</td><td>81.23</td><td>75.27</td><td>67.28</td><td>66.29 (+6.94)</td></tr><tr><td>30% /  10%</td><td>64.98</td><td>38.84</td><td>45.93</td><td>90.37</td><td>73.42</td><td>72.77</td><td>62.69</td><td>62.71 (+6.63)</td></tr><tr><td>10% /  10%</td><td>66.23</td><td>34.77</td><td>37.54</td><td>85.73</td><td>58.27</td><td>75.58</td><td>46.36</td><td>56.81 (+9.35)</td></tr></table>

TABLE IV

Real-Robot Task Results. We evaluate unpruned and pruned π<sub>0.5</sub> on an AgileX dual-arm platform and report successes out of 20 trials per task. Target retention ratios and parameter reductions follow Table I. Avg. is the mean success rate across the six tasks described below.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td rowspan="2"> $\rho _ { h i d } / \rho _ { i n t }$ </td><td colspan="2">Pruning Ratio</td><td rowspan="2">Task 1 Bottle</td><td rowspan="2">Task 2 Cubes</td><td rowspan="2">Task 3 Cup</td><td rowspan="2">Task 4 Block</td><td rowspan="2">Task 5 Sorting</td><td rowspan="2">Task 6 Backpack</td><td rowspan="2">Avg. (%)</td></tr><tr><td>Model</td><td>LLM</td></tr><tr><td>π0.5</td><td>baseline</td><td>100% /  100%</td><td>0%</td><td>0%</td><td>19/20</td><td>17/20</td><td>15/20</td><td>18/20</td><td>12/20</td><td>4/20</td><td>70.83</td></tr><tr><td>π0.5</td><td>PAT</td><td>100% / 20%</td><td>40%</td><td>58%</td><td>19/20</td><td>16/20</td><td>13/20</td><td>16/20</td><td>9/20</td><td>3/20</td><td>63.33</td></tr></table>

Task Instruction  
Task Instruction  
Task 1 Pick up the bottle and place it upright.  
Task 4 Put the red block into the basket.  
Task 2 Place the blue cube to the right of the yellow cube.  
Task 3 Pick up the cup and place it on the plate.

![](images/0a094c8c9b533657d6053fa8658d1028620e8aa7c0c171c0d7c5c05eeaa0be80.jpg)  
(a)

![](images/2bb374fd5163196d68841d266f13529bd6ba0bb0725f4b2e9c2d58634d179e97.jpg)  
(b)  
Task 5 Put the screws in the blue basket and the blocks in the red basket. Task 6 Put the object in the backpack.

![](images/e40db46dd0c1ec2e725a90ee800239c5d118085e7b706cf5be323bb59395108c.jpg)  
(c)

![](images/72a3da005e7b307ebfb00d1c96aa6e3c24f8a9fe23425c135fc2f9e83699d4a7.jpg)  
(d)  
Fig. 4. Real-Robot Sorting Task on the AgileX Dual-Arm Platform. Four frames from one recorded execution of Task 5: (a) initial arrangement, (b) block manipulation, (c) screw manipulation, and (d) the final recorded scene. Quantitative results are reported in Table IV.

1.5 [6], SmolVLA [4], GraspVLA [7], and EcoVLA [12]. SPAR’s parameter-pruning ratios are relative to the unpruned full model (3.37B) and LLM (2.34B), with pruning percentages rounded to integers. EcoVLA’s labels denote its reported channel/head pruning settings.

c) Implementation Details: SFT uses a batch size of 256 for 30,000 steps: 6,000 for PAT mask learning and 24,000 for post-pruning fine-tuning. Training and RL use NVIDIA A800 GPUs. We deploy on NVIDIA Jetson AGX Orin and measure median full-policy inference latency using TensorRT.

d) PPO Configuration: PPO uses eight NVIDIA A800 GPUs on one node, with 128 training and 80 evaluation environments across all four suites. The actor global batch size is 2,048, with a per-rank micro-batch size of 128 and two gradient-accumulation steps. The runner allows up to 200 RL epochs.

## A. Main Results

Pruning only the intermediate dimension preserves high task success (Table I). A 10% intermediate-dimension retention target reduces the LLM from 2.34B to 0.82B with 96.45% average success, only 0.35 percentage points below the baseline’s 96.80%. This suggests substantial FFN intermediate-dimension redundancy on these tasks, motivating $\rho _ { i n t } = 0 . 1$ for subsequent experiments.

Additional hidden-dimension pruning causes larger losses (Table II): reducing $\rho _ { h i d }$ from 0.9 to 0.1 lowers average success without RL from 95.90% to 84.55%, most sharply on Long. RL substantially mitigates these losses, improving all five pruned averages by 1.90–8.90 percentage points versus

0.90 for the baseline. Notably, at $\rho _ { h i d } = 0 . 7$ and $\rho _ { i n t } = 0 . 1$ RL achieves 97.55%, just 0.15 points below the unpruned RL baseline’s 97.70%, with approximately 52% fewer total and 75% fewer LLM parameters. Even the smallest policy (1.12B total / 0.09B LLM) improves from 84.55% to 93.45%, recovering much of the loss while remaining 4.25 points below that baseline.

## B. Robustness on LIBERO-Plus

LIBERO-Plus [17] tests changes in camera viewpoints, robot initial states, language instructions, lighting, background textures, sensor noise, and object layouts. Table III reports cross-suite scores by perturbation category and the aggregate score under Overall.

RL substantially improves all five pruned overall scores by 5.52–9.35 percentage points, versus 0.47 for the baseline. The smallest policy improves from 47.46% to 56.81% but remains 27.64 points below the unpruned RL baseline, particularly under robot-initial-state and language perturbations. Even at $\rho _ { h i d } = 0 . 9$ and $\rho _ { i n t } = 0 . 1$ , the recovered policy scores 78.20% versus the baseline’s 84.45%. This 6.25-point gap, despite nearly matched standard LIBERO performance, indicates greater sensitivity to perturbations after compression. Robustness-critical applications may therefore require less aggressive pruning or further RL adaptation with broader perturbation exposure.

## C. Real-Robot Evaluation

On an AgileX dual-arm platform, we evaluate six tasks with 20 trials each (Table IV); Fig. 4 illustrates Task 5. Pruning only the intermediate dimension with $\rho _ { i n t } = 0 . 2$ yields 2.02B total / 0.99B LLM parameters, approximately 40% / 58% fewer than the baseline. The compact policy retains much of the baseline’s manipulation performance: 76/120 successes (63.33%) versus 85/120 (70.83%), a 7.50- percentage-point decrease. It matches the baseline on Task 1 and records successes on all six tasks. These results show that intermediate-dimension pruning preserves practical manipulation capability on physical hardware despite substantial parameter reduction.

## D. Inference Latency

Table V reports full-policy latency and memory on NVIDIA Jetson AGX Orin. At 10% intermediate retention, latency falls from 295.04 ms to 170.64 ms (1.73×); adding hidden-dimension pruning at the same retention reaches 148.71 ms, a 1.98× speedup and 49.60% latency reduction. GPU memory decreases from 7.098 to 2.627 GiB, approximately 63% lower than the baseline.

## E. Ablation Study

Table VI shows that the full objective in Table I achieves the highest average success rate. Removing the count, entropy, or discreteness term lowers the average by 0.75, 0.35, or 0.40 percentage points, respectively. The full objective also achieves the highest Long-suite score, supporting the combined use of budget and binarization penalties in this configuration.

## TABLE V

Inference Latency and GPU Memory of $\pi _ { 0 . 5 } .$ We report median full-policy latency and GPU memory using TensorRT on NVIDIA Jetson AGX Orin. Memory is in GiB, speedup is the unpruned-to-pruned latency

<table><tr><td colspan="3">Latency</td><td colspan="2">GPU Memory</td></tr><tr><td>Model</td><td> $\rho _ { h i d } / \rho _ { i n t }$ </td><td>(ms, ↓)</td><td>Speedup (↑)</td><td>(GiB, ↓)</td></tr><tr><td>π0.5</td><td>100% / 100%</td><td>295.04</td><td>1.00</td><td>7.098</td></tr><tr><td rowspan="6">π0.5</td><td>100% / 90%</td><td>282.32</td><td>1.05</td><td>6.713</td></tr><tr><td>100% / 70%</td><td>252.36</td><td>1.17</td><td>6.039</td></tr><tr><td>100% / 50%</td><td>222.65</td><td>1.33</td><td>5.395</td></tr><tr><td>100% /  30%</td><td>194.20</td><td>1.52</td><td>4.752</td></tr><tr><td>100% / 20%</td><td>178.93</td><td>1.65</td><td>4.400</td></tr><tr><td>100% / 10%</td><td>170.64</td><td>1.73</td><td>4.086</td></tr><tr><td rowspan="6">π0.5</td><td>90% /  10%</td><td>165.52</td><td>1.78</td><td>3.908</td></tr><tr><td>70% / 10%</td><td>162.76</td><td>1.81</td><td>3.596</td></tr><tr><td>50% /  10%</td><td>156.10</td><td>1.89</td><td>3.281</td></tr><tr><td>30% /  10%</td><td>151.57</td><td>1.95</td><td>2.973</td></tr><tr><td>10% / 10%</td><td>148.71</td><td>1.98</td><td>2.627</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

TABLE VI

Pruning-Loss Ablation for $\pi _ { 0 } . 5 ,$ Evaluated on LIBERO. We remove one loss at a time from the full objective, targeting $\rho _ { h i d } = 1 0 0 \%$ and $\rho _ { i n t } = 1 0 \%$ without RL. Scores are percentages, Long denotes LIBERO-10, and $\operatorname { A v g } .$ is the four-suite mean. Bold identifies the best value in each column.
<table><tr><td>Method</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg.</td></tr><tr><td>PAT w/o  $\mathcal { L } _ { c o u n t }$ </td><td>97.60</td><td>98.60</td><td>97.40</td><td>89.20</td><td>95.70</td></tr><tr><td>PAT w/o  $\mathcal { L } _ { e n t r o p y }$ </td><td>97.40</td><td>98.40</td><td>96.60</td><td>92.00</td><td>96.10</td></tr><tr><td>PAT w/o  $\mathcal { L } _ { d i s c }$ </td><td>97.60</td><td>98.80</td><td>97.20</td><td>90.60</td><td>96.05</td></tr><tr><td>PAT (full objective)</td><td>97.40</td><td>97.60</td><td>96.80</td><td>94.00</td><td>96.45</td></tr></table>

## V. LIMITATIONS AND FUTURE WORK

Although SPAR approaches or slightly exceeds the unpruned RL baseline on LIBERO under some aggressive compression settings, a robustness gap remains on LIBERO-Plus. Evaluation is also limited to one backbone and one intermediate-dimension pruning configuration across six real-robot tasks, motivating broader backbone studies, joint hidden-dimension and intermediate-dimension pruning on robots, and energy measurements across deployment platforms.

To further improve recovery, we envision using Sim-Foundry [32] to reconstruct real-world AgileX task scenes as simulation environments. TiPToP [33] and ASPIRE [34] could then generate trajectories through planning and program execution, with MimicGen [35] expanding their coverage for supervised adaptation of compact policies. Building on this initialization, the policies would undergo online reinforcement learning with PPO through direct interaction in the real-to-sim environments reconstructed by SimFoundry.

## VI. CONCLUSION

SPAR jointly prunes the hidden dimension and intermediate dimension with separate retention targets, combining pruning-aware adaptation with PPO-based recovery.

Intermediate-only pruning preserves high success, while RL partially restores hidden-dimension losses. With a 0.09B LLM and 1.12B total parameters, the compact policy achieves a 1.98× inference speedup, while RL raises its LIBERO success from 84.55% to 93.45%. On AgileX, 20% intermediate retention leaves a 7.50-point gap to the baseline. These results support combining dimension-aware compression and RL to improve the performance–size tradeoff of VLA models.

## REFERENCES

[1] A. Brohan et al., “RT-2: Vision-language-action models transfer web knowledge to robotic control,” 2023. [Online]. Available: https://arxiv.org/abs/2307.15818

[2] M. J. Kim et al., “OpenVLA: An open-source vision-language-action model,” 2024. [Online]. Available: https://arxiv.org/abs/2406.09246

[3] Physical Intelligence et al., “π<sub>0.5</sub>: a vision-language-action model with open-world generalization,” 2025. [Online]. Available: https: //arxiv.org/abs/2504.16054

[4] M. Shukor et al., “SmolVLA: A vision-language-action model for affordable and efficient robotics,” 2025. [Online]. Available: https://arxiv.org/abs/2506.01844

[5] C.-Y. Hung et al., “NORA: A small open-sourced generalist vision language action model for embodied tasks,” 2025. [Online]. Available: https://arxiv.org/abs/2504.19854

[6] ——, “NORA-1.5: A vision-language-action model trained using world model- and action-based preference rewards,” 2025. [Online]. Available: https://arxiv.org/abs/2511.14659

[7] S. Deng et al., “GraspVLA: a grasping foundation model pre-trained on billion-scale synthetic action data,” 2025. [Online]. Available: https://arxiv.org/abs/2505.03233

[8] Y. Li et al., “SP-VLA: A joint model scheduling and token pruning approach for VLA model acceleration,” 2025. [Online]. Available: https://arxiv.org/abs/2506.12723

[9] Z. Wen et al., “Stop looking for important tokens in multimodal language models: Duplication matters more,” 2025. [Online]. Available: https://arxiv.org/abs/2502.11494

[10] Y. Yang et al., “EfficientVLA: Training-free acceleration and compression for vision-language-action models,” 2025. [Online]. Available: https://arxiv.org/abs/2506.10100

[11] R. Zhang et al., “MoLe-VLA: Dynamic layer-skipping vision language action model via mixture-of-layers for efficient robot manipulation,” 2025. [Online]. Available: https://arxiv.org/abs/2503.20384

[12] Y. Huang et al., “Environment-aware adaptive pruning with interleaved inference orchestration for vision-language-action models,” 2026. [Online]. Available: https://arxiv.org/abs/2602.00780

[13] Y. Chen, Y. Han, Y. Huang, and X. Li, “RLRC: Reinforcement learning-based recovery for compressed vision-language-action models,” IEEE Robotics and Automation Letters, vol. 11, no. 7, pp. 8864– 8871, 2026.

[14] H. Li et al., “SimpleVLA-RL: Scaling VLA training via reinforcement learning,” 2025. [Online]. Available: https://arxiv.org/abs/2509.09674

[15] Y. Liu et al., “PAT: Pruning-aware tuning for large language models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 23, 2025, pp. 24 686–24 695. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/34649

[16] B. Liu et al., “LIBERO: Benchmarking knowledge transfer for lifelong robot learning,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36. Curran Associates, Inc., 2023, pp. 44 776–44 791.

[17] S. Fei et al., “LIBERO-Plus: In-depth robustness analysis of vision-language-action models,” 2025. [Online]. Available: https: //arxiv.org/abs/2510.13626

[18] M. J. Kim, C. Finn, and P. Liang, “Fine-tuning vision-language-action models: Optimizing speed and success,” 2025. [Online]. Available: https://arxiv.org/abs/2502.19645

[19] Q. Li et al., “CogACT: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation,” 2024. [Online]. Available: https://arxiv.org/abs/2411.19650

[20] K. Black et al., “π : A vision-language-action flow model for general robot control,” in Proceedings of Robotics: Science and Systems, 2025. [Online]. Available: https://www.roboticsproceedings.org/rss21/ p010.html

[21] L. Beyer et al., “PaliGemma: A versatile 3B VLM for transfer,” 2024. [Online]. Available: https://arxiv.org/abs/2407.07726

[22] J. Wen et al., “TinyVLA: Toward fast, data-efficient vision-languageaction models for robotic manipulation,” IEEE Robotics and Automation Letters, vol. 10, no. 4, pp. 3988–3995, 2025.

[23] E. Frantar and D. Alistarh, “SparseGPT: Massive language models can be accurately pruned in one-shot,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202, 2023, pp. 10 323–10 337. [Online]. Available: https://proceedings.mlr.press/v202/frantar23a.html

[24] X. Ma, G. Fang, and X. Wang, “LLM-Pruner: On the structural pruning of large language models,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 21 702–21 720. [Online]. Available: https://arxiv.org/abs/2305.11627

[25] S. Ashkboos, M. L. Croci, M. Gennari do Nascimento, T. Hoefler, and J. Hensman, “SliceGPT: Compress large language models by deleting rows and columns,” in International Conference on Learning Representations, 2024. [Online]. Available: https: //arxiv.org/abs/2401.15024

[26] Physical Intelligence et al., “π<sup>∗</sup><sub>0.6</sub>: a VLA that learns from experience,” 2025. [Online]. Available: https://arxiv.org/abs/2511.14759

[27] H. Zang et al., “RLinf-VLA: A unified and efficient framework for VLA+RL training,” 2025. [Online]. Available: https://arxiv.org/abs/ 2510.06710

[28] J. Gu et al., “ManiSkill2: A unified benchmark for generalizable manipulation skills,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https: //openreview.net/forum?id=b CQDy9vrD1

[29] X. Zhou et al., “LIBERO-PRO: Towards robust and fair evaluation of vision-language-action models beyond memorization,” arXiv preprint arXiv:2510.03827, 2025. [Online]. Available: https://arxiv.org/abs/ 2510.03827

[30] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017. [Online]. Available: https: //arxiv.org/abs/1707.06347

[31] Physical Intelligence, “OpenPI: Open-source models and packages for robotics,” Software repository, 2025. [Online]. Available: https://github.com/Physical-Intelligence/openpi

[32] N. Ranawaka et al., “SimFoundry: Modular and automated scene generation for policy learning and evaluation,” arXiv preprint arXiv:2606.28276, 2026. [Online]. Available: https://arxiv.org/abs/ 2606.28276

[33] W. Shen et al., “TiPToP: A modular open-vocabulary robot manipulation system that plans,” 2026. [Online]. Available: https: //arxiv.org/abs/2603.09971

[34] R. Lu et al., “ASPIRE: Agentic /skills discovery for robotics,” arXiv preprint arXiv:2607.00272, 2026. [Online]. Available: https: //arxiv.org/abs/2607.00272

[35] A. Mandlekar et al., “MimicGen: A data generation system for scalable robot learning using human demonstrations,” in Proceedings of the 7th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 229, 2023, pp. 1820–1864. [Online]. Available: https://proceedings.mlr.press/v229/mandlekar23a.html