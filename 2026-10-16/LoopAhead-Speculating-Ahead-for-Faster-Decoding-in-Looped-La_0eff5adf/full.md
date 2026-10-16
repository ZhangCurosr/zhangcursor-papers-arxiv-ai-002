# LoopAhead: Speculating Ahead for Faster Decoding in Looped Language Models

Zixuan Liu<sup>1</sup>, Nan Xue<sup>1</sup>, Rui Bao<sup>1</sup>, Yuhang Xu<sup>1</sup>, Xiaofeng Cao<sup>2</sup>, Jiangchao Yao<sup>1</sup>, Zhiyong Chen<sup>1</sup>\*

<sup>1</sup>Shanghai Jiao Tong University

<sup>2</sup>Tongji University

{lzx20020405,nan.xue,851756936,xuyuhangtmx,Sunarker,zhiyongchen}@sjtu.edu.cn xiaofengcao@tongji.edu.cn

## Abstract

Looped Transformers improve parameter efficiency by repeatedly applying the same Transformer layers. However, generating each token still involves sequential computation and repeated access to model weights, leading to high decoding latency. Recent pipelined selfspeculative decoding methods reduce this delay by drafting tokens from early-loop predictions and asynchronously advancing tokens at different recurrent depths in the same batch, but they do not fully exploit early-loop features for longer lookahead and still spend computa tion verifying incorrect speculative branches. We introduce LoopAhead, a speculative decoding framework for looped transformers that enables earlier drafting of future tokens and assigns complementary roles to different recurrent depths. A lightweight lookahead drafter (LAD) predicts multiple future tokens with early-loop states, and deeper-loop feedback helps decide which draft branches to continue and which to pause, while the full-depth model makes the final token decisions. LoopAhead achieves up to 7.82× decoding speedup over autoregressive decoding while preserving the target model’s output distribution.

## 1 Introduction

Looped Transformers have recently attracted increasing attention as a parameter-efficient architecture for large language models (McLeish et al., 2025; Geiping et al., 2026). By repeatedly applying a shared Transformer stack across recurrent depths, they achieve performance comparable to substantially deeper and larger models while using significantly fewer parameters (Saunshi et al., 2025; Dehghani et al., 2018; Zhu et al., 2025). However, under full-depth autoregressive decoding, a looped model completes all configured recurrent passes before producing each token (Cho et al., 2026), and its parameters are repeatedly accessed across iterations for every generated token, leading to increased memory-bandwidth pressure (Hooper et al., 2025). The resulting latency makes inference acceleration particularly relevant to looped models (Schwethelm et al., 2026; Cho et al., 2026).

![](images/b6d9d3a1a3fc0721987c8a0e558fde3e5bf1fa80ffac1a5013114b406d84a880.jpg)

![](images/03758019b17c728e43be0705e87563c99e914c3a6d29b9480e3a984c4da8c452.jpg)

![](images/ef05dab6fe3114b5758f66b49771fb12d9b43c12cf5bd38ef3ebba9888332b1a.jpg)  
Figure 1: Key observations and decoding speedups of LoopAhead. Top left shows Ouro-1.4B’s task accuracy when generating at a fixed loop depth r. Top right shows token agreement between depth-r and full-depth predictions under the same prefixes. The bottom panel shows representative decoding speedups over AR, with percentages indicating throughput gains over LoopSpec.

Speculative decoding accelerates autoregressive inference by generating candidate tokens with a lightweight drafter and verifying them in parallel with the target model (Leviathan et al., 2023; Chen et al., 2023; Jiang et al., 2026). Feature-based methods further improve drafting using target hidden representations (Li et al., 2024b,a, 2025; Zhang et al., 2025). However, these methods generally separate drafting from verification and do not fully exploit the shared recurrent structure of looped Transformers. Recent methods such as LoopSpec (Cho et al., 2026) and WaveFront (Ha and Kim, 2026) address this by pipelining early-loop proposals with full-depth verification, allowing tokens at different recurrent depths to be processed together. In particular, LoopSpec achieves substantial speedups without training an additional drafter. Nevertheless, each new proposal requires its predecessor to reach a specified recurrent depth, limiting how early future tokens can be generated. Moreover, incorrect branches may continue consuming computation until full-depth verification.

To go beyond the speedup achieved by pipelining alone, we explore how recurrent states can support both drafting and early candidate checking. Prior work has shown the value of target features and early-layer predictions for speculative drafting (Li et al., 2024b; Elhoushi et al., 2024). Our observation in Figure 1 shows that generating complete answers using only early loops can substantially reduce task accuracy. Yet early-loop predictions already show high token agreement with full-depth outputs under the same prefixes, and this agreement rises rapidly over subsequent loops. These observations motivate using early-loop states to draft multiple future tokens and using intermediateloop feedback to guide execution of speculative branches before full-depth verification.

Motivated by these observations, we introduce LoopAhead, a novel speculative decoding framework for looped language models that combines early-loop multi-token drafting with intermediateloop feedback-guided scheduling. A separately trained lightweight lookahead drafter predicts multiple future candidates from selected early-loop states and a native proposal anchor. Feedback from intermediate loops guides the recurrent computation, suspension and resumption. Our experiments show that feedback-guided scheduling reduces computation on unpromising branches while improving decoding throughput, making early lookahead more efficient. LoopAhead remains lossless since the scheduler only controls speculative execution, while all final output decisions remain determined by the full-depth target verification rule. Our contributions are:

• We introduce LoopAhead, a lossless speculative decoding framework for looped language models. Its lookahead drafter uses early-loop states and native proposal anchors to predict multiple future positions, and uses a compact draft vocabulary to reduce drafting overhead.

• We design a feedback-guided scheduling mechanism that leverages intermediate recurrent representations to dynamically suspend and resume speculative branches under bounded execution budgets. Experimental results demonstrate that our scheduling strategy effectively limits wasted computation caused by unpromising speculative paths.

• We evaluate LoopAhead across diverse looped language models and benchmarks under both greedy and sampling decoding settings. LoopAhead achieves up to 7.82× decoding speedup over AR as shown in Figure 1. Further analysis and ablation studies demonstrate the effectiveness of components and their complementary contributions to performance.

## 2 Preliminaries

Looped Transformer. Looped Transformers increase computational depth by repeatedly applying a shared stack $\mathcal { F } _ { \theta }$ (Dehghani et al., 2018; Geiping et al., 2026). Suppressing input-conditioning and cache arguments, the recurrent core follows

$$
\begin{array} { r } { { \bf Z } ^ { ( r ) } = \mathcal { F } _ { \boldsymbol { \theta } } \big ( { \bf Z } ^ { ( r - 1 ) } \big ) , \qquad r = 1 , \ldots , R , } \end{array}
$$

where $\mathbf { Z } ^ { ( 0 ) }$ contains the initial hidden states. Repeating $n _ { \mathrm { l a y e r } }$ shared layers gives an effective core depth of $n _ { \mathrm { l a y e r } } R$ without additional parameters.

Learned Drafting for Speculative Decoding. Speculative decoding verifies draft tokens in parallel using a target model (Leviathan et al., 2023; Chen et al., 2023). Blockwise decoding, MTP, and Medusa predict multiple positions in parallel (Stern et al., 2018; Gloeckle et al., 2024; Cai et al., 2024). Hydra and ReDrafter introduce dependencies between draft tokens (Ankner et al., 2024; Cheng et al., 2024). EAGLE drafts autoregressively from target features (Li et al., 2024b). EAGLE-2 adapts draft trees (Li et al., 2024a), and EAGLE-3 fuses multi-layer features for direct token prediction (Li et al., 2025). DistillSpec improves draft–target alignment through distillation (Zhou et al., 2024), while HASS aligns training and decoding contexts and objectives (Zhang et al., 2025). LayerSkip trains early exits for self-speculation (Elhoushi et al., 2024). DFlash generates draft blocks in parallel with a diffusion model (Chen et al., 2026).

Pipelined Speculative Decoding. SPEED pipelines decoding with cyclic parameter sharing, but invalidates dependent states and restarts computation when intermediate predictions change (Hooper et al., 2025). LoopSpec (Cho et al., 2026) provides training-free pipelining specially designed for looped models with gated deeper proposals and full-depth verification. However, each native readout predicts only the next token position and drafting further ahead requires additional recurrent computation.

![](images/66f98772aae27d6414737db5af53dff2180e0600b864ac58ac0cb258f55ba202.jpg)  
Figure 2: Illustration of LoopAhead with an Ouro-1.4B decoding trajectory. LAD uses early-loop context and the native anchor to draft future tokens before verification. R2 feedback suspends conflicting branches while retaining their KV states for resumption. Full-depth R4 verification determines acceptance and pruning. The complete traces appear in Appendix E and Figure 13.

## 3 Method

Framework Overview Figure 1 shows that earlyloop predictions often agree with full-depth outputs, motivating us to draft future tokens early and use deeper-loop feedback to guide their execution. Figure 2 illustrates how LoopAhead combines these roles with full-depth verification.

## 3.1 Early-Loop LookAhead Drafting

Anchor-Conditioned Drafting from Early-Loop States Let $h _ { t } ^ { ( r ) }$ denote the target state immediately before the vocabulary projection after r loops. It is computed at token $x _ { t - 1 }$ and predicts position t. We collect w cached depth-r states along this branch.

$$
\mathbf { H } _ { t } ^ { ( r ) } = [ h _ { t - w + 1 } ^ { ( r ) } , . . . , h _ { t } ^ { ( r ) } ] .\tag{1}
$$

For greedy decoding, the native prediction supplies an anchor at position t.

$$
a _ { t } = \arg \operatorname* { m a x } _ { v \in \mathcal { V } _ { \mathrm { t g t } } } [ W _ { \mathrm { o u t } } h _ { t } ^ { ( r ) } ] _ { v } ,\tag{2}
$$

where $W _ { \mathrm { o u t } }$ is the frozen target vocabulary projection and $\nu _ { \mathrm { t g t } }$ is the target vocabulary. We call this direct target-model prediction a native proposal.

Under sampling, the anchor is drawn after temperature scaling and nucleus truncation. The anchor need not be verified before drafting. Its embedding supplies the token at position t, while the cached states describe the preceding context.

The LAD drafter $\mathcal { D } _ { \phi }$ uses $\mathbf { H } _ { t } ^ { ( r ) }$ and $E ( a _ { t } )$ to predict K additional tokens at positions $t + 1 , \ldots , t +$ K, where E denotes the frozen target token embedding. It projects the context into a smaller hidden space and appends one query for each future position. Each query combines the anchor embedding, the latest projected context state, and a position embedding. A lightweight causal Transformer processes the context and queries together. Its output is projected back to the target hidden size and added to the latest early-loop state.

$$
\begin{array} { r l } & { \widetilde { h } _ { t , j } = h _ { t } ^ { ( r ) } + \mathcal { D } _ { \phi , j } \big ( \mathbf { H } _ { t } ^ { ( r ) } , E ( a _ { t } ) \big ) , } \\ & { q _ { \phi , j } = \mathrm { s o f t m a x } \big ( W _ { \mathrm { o u t } } \widetilde { h } _ { t , j } \big ) . } \end{array}\tag{3}
$$

Here $q _ { \phi , j }$ is the base drafter’s token distribution at position $t + j$ . The base drafter computes all future hidden states in one forward pass. The causal mask allows each query to attend to the context and earlier query representations. All queries are constructed before token selection, so the base drafter predicts future positions without feeding one selected draft token into the next prediction. Ouro and Raven-TinyLlama select draft tokens directly from these predictions. Raven-Llama and Raven-OLMo add a small logit correction conditioned on the preceding draft token. Appendix A.5 gives the model-specific adaptations. Draft tokens are selected by argmax in both decoding regimes. Sampling randomness is handled by native proposals and full-depth verification.

Teacher Trajectories and Training Objectives The frozen target generates full-depth greedy continuations of mixed-domain prompts. We record its early-loop states and use its generated tokens as training labels. For a teacher trajectory y, a training example at prediction position t is

$$
\begin{array} { r } { \big ( \mathbf { H } _ { t } ^ { ( r ) } , ~ a _ { t } = y _ { t } , ~ y _ { t + 1 : t + K _ { \mathrm { t r a i n } } } \big ) . } \end{array}\tag{4}
$$

Training supervises up to $K _ { \mathrm { t r a i n } }$ future positions, and inference uses $K \le K _ { \mathrm { t r a i n } }$ . Base training uses teacher tokens as anchors. At inference, the anchor comes from the native early-loop proposal. Labels beyond the recorded continuation are masked. We train the base drafter with cross-entropy over valid teacher-label indices I.

$$
\mathcal { L } _ { \mathrm { L A D } } = - \frac { 1 } { | \mathcal { T } | } \sum _ { ( t , j ) \in \mathcal { I } } \log q _ { \phi , j } \left( y _ { t + j } \mid \mathbf { H } _ { t } ^ { ( r ) } , y _ { t } \right) .\tag{5}
$$

The target model, token embeddings, and output projection remain frozen. The base drafter learns to turn early context into hidden states that predict the target’s later tokens. Training objectives for the residual adaptations are given in Appendix A.5.

Low-Overhead Drafter Execution At inference, LAD projects to a request-specific vocabulary $\nu _ { \mathrm { d r a f t } }$ containing frequent training-output tokens, stop tokens, prompt tokens, and already committed tokens. We gather the corresponding rows of $W _ { \mathrm { o u t } }$ and restrict draft argmax and score normalization to this set. This reduces projection cost for each future position. Native proposals and final verification use the full target vocabulary $\nu _ { \mathrm { t g t } }$

## 3.2 Feedback-Guided Scheduling

Feedback from Intermediate Loops Intermediate-loop readouts provide early feedback for candidates awaiting full-depth verification. A parent’s readout predicts the token represented by its child. Under greedy decoding, a candidate conflicts if its token differs from the parent’s prediction at the deepest available feedback depth. Under sampling, it conflicts if its token differs from all native proposal tokens recorded by the parent. Ouro uses one feedback depth, while Raven updates feedback across multiple depths. Appendix A.5 gives the selected depths and sampling rules.

Candidate Tree and Recurrent Execution The root represents the last committed token, and its descendants represent speculative continuations. Each retained node represents a candidate token and stores its parent, hidden state, KV states, and completed loop depth. Native proposals and LAD drafts enter the same tree. Proposals with the same token and parent share one node. The tree therefore reuses computations when different drafting paths propose the same continuation. Within each LAD extension, the candidate at position t+1 is attached to the native anchor $a _ { t } .$ , and each later candidate is attached to the preceding draft token. Different candidate tokens extending the same prefix form separate branches.

Nodes advance from their own loop depths, allowing candidates at different stages to run in the same batch. At each depth, a node attends to itself, the committed prefix, and its ancestors on the same branch. Figure 2 illustrates this attention mask. Required ancestor KV states must be available at the corresponding depth, either from the cache or from ancestors advanced in the same batch.

Execution Policy $B _ { \mathrm { r e t } }$ limits the nodes kept in the tree, while $\boldsymbol { B } _ { \mathrm { e x e } }$ limits those advanced in each batch. A path conflicts if any token on it conflicts. Nodes within $d _ { \mathrm { g u a r d } }$ token positions of the committed prefix are protected from conflict-based suspension. The scheduler applies three operations.

(1) Execution. Select at most $\boldsymbol { B } _ { \mathrm { e x e } }$ unfinished nodes with available predecessor states. Eligible nodes have conflict-free paths or lie within the protected region. The scheduling priority chooses among eligible nodes when the batch is full.

(2) Suspension and resumption. Suspend conflicting paths outside the protected region and retain their hidden states, KV states, and completed loop depths. A node becomes eligible again when feedback resolves the conflict or root advancement brings it into the protected region. Nodes left out of a batch remain available for later execution.

(3) Pruning. Full-depth verification selects the next output token. Keep its continuation and cached states, and discard competing subtrees. If the selected token has no retained node, create one and compute its states from the committed prefix.

Recursive Draft Expansion After each recurrent step, our scheduling policy considers nodes newly reaching the drafting depth and selects at most one of their native proposal children as an anchor, prioritizing those nearest the root. Drafting requires a cached context window and an anchor with no existing children. Each LAD call adds a chain of at most K tokens, subject to retention capacity and model-specific admission rules. A drafted node can later reach the same early depth and support another call, extending lookahead beyond one draft block. Appendix A.5 shows the details of admission and execution priorities.

Full-Depth Verification and Commitment The full-depth readout of a node determines the token that follows it. Greedy decoding selects the argmax. Sampling applies native proposal acceptance and residual correction against the full-depth distribution, as described in Appendix D. The selected token is then matched to a retained child, whether proposed natively or by LAD. This child becomes the root and preserves its computed states. If its full-depth readout is ready, commitment continues within the same step.

## 3.3 Theoretical Analysis

For greedy decoding, consider a length-L chain $x _ { 1 : L }$ formed by recursive LAD calls after a correct prefix, and let $y _ { 1 : L } ^ { \star }$ be the full-depth continuation. Let N be their matching-prefix length and $J = N +$ 1 the first mismatch, with $J = L + 1$ if all tokens match. Define $\alpha _ { i } = \mathrm { P r } ( N \ge i \mid N \ge i - 1 )$ $\begin{array} { r } { S _ { i } = \operatorname* { P r } ( N \geq i ) = \prod _ { i = 1 } ^ { i } \alpha _ { j } } \end{array}$ with $S _ { 0 } = 1$ , and $\begin{array} { r } { A _ { L } = \mathbb { E } [ N ] = \sum _ { i = 1 } ^ { L } S _ { i } } \end{array}$

Fix a feedback depth $r \ < \ R$ . Let $\hat { y } _ { i } ^ { ( r ) }$ be its native prediction under the same preceding tokens, and $\gamma _ { i } = \mathrm { P r } ( \hat { y } _ { i } ^ { ( r ) } = y _ { i } ^ { \star } \mid J = i )$ denote the probability that feedback predicts the target token at the first mismatch. Compare the same candidate workload with and without suspension, keeping other policies fixed. Let $W _ { \mathrm { a v o i d } }$ count advances on the rejected suffix performed without suspension and skipped with it, counting through final pruning. One advance runs one recurrent loop for one candidate. Let $\bar { G } _ { i } \geq 0$ be a lower bound on $\mathbb { E } [ W _ { \mathrm { a v o i d } } \mid J = i , \hat { y } _ { i } ^ { ( r ) } = y _ { i } ^ { \star } ]$

Theorem 1 (Feedback-Guided Recursive Draft Filtering). Under this matched-workload comparison,

$$
\mathbb { E } [ W _ { \mathrm { a v o i d } } ] \geq \sum _ { i = 1 } ^ { L } S _ { i - 1 } ( 1 - \alpha _ { i } ) \gamma _ { i } \bar { G } _ { i } .\tag{6}
$$

$I f \gamma _ { i } \geq \gamma$ and $\bar { G } _ { i } \geq g ( L - i + 1 )$ for all i, where $\gamma , g \geq 0$ , then

$$
\mathbb { E } [ W _ { \mathrm { a v o i d } } ] \geq g \gamma ( L - A _ { L } ) .\tag{7}
$$

The first bound sums the contributions from correct feedback at each possible first mismatch. In the second bound, $A _ { L }$ is the expected matchingprefix length and $L - A _ { L }$ is the expected rejectedsuffix length. An early mismatch can leave many recursive successors on which feedback can save computation. The proof and extensions to multiple feedback depths and variable chain lengths are provided in Appendix C.1 in detail.

## 4 Experiments

## 4.1 Settings

Models. We evaluate Ouro-1.4B, Ouro-2.6B, Ouro-1.4B-Thinking (Zhu et al., 2025), Raven-Llama-3.2, Raven-OLMo-2, and Raven-TinyLlama (McLeish et al., 2025) from two model families, using 4 recurrences for Ouro and 32 for Raven. Inference runs on NVIDIA A40 GPUs (48 GB), with one request per GPU and BF16 target computation in a local PyTorch-based backend.

Draft Training and Inference Details. Each target uses a separately trained LAD drafter shared across all benchmarks. Teacher trajectories are collected from a pool of 9,025 training prompts spanning GSM8K (Cobbe et al., 2021), CodeAlpaca (Chaudhary, 2023), MATH (Hendrycks et al., 2021), BBH (Suzgun et al., 2023), and UltraChat (Ding et al., 2023), with 782 development and 782 confirmation prompts. LAD base drafters are twolayer Transformers with hidden size 512, trained for three epochs using AdamW with a peak learning rate of $2 \times 1 0 ^ { - 4 }$ . Raven-Llama/OLMo additionally use R4 residual adapters, as detailed in Appendix A.5. The base drafter is trained to predict up to $K _ { \mathrm { t r a i n } } = 7$ future tokens beyond the native anchor. At inference time, each LAD invocation generates at most three additional tokens for Ouro and Raven-TinyLlama, and two for Raven-Llama/OLMo. All main LoopAhead results use an 8,192-token base draft vocabulary without changing the target vocabulary.

Baselines. We compare LoopAhead with three baselines. (i) AR uses standard autoregressive decoding at full recurrent depth. (ii) LoopSpec is our reproduction of pipelined self-speculative decoding (Cho et al., 2026), with proposal depths (1, 2) for Ouro and (2, 8) for Raven, following the dual-proposal configuration provided in the official LoopSpec implementation (Cho et al., 2026). (iii) DFlash is a locally trained adaptation of DFlash (Chen et al., 2026). Its training and development-based block-size selection are detailed in Appendix A.3.

<table><tr><td>Model</td><td>Method</td><td>GSM8K</td><td>MATH500</td><td>HumanEval</td><td>MBPP</td><td>BBH</td></tr><tr><td rowspan="3">Ouro-1.4B</td><td>DFlash</td><td>2.19 (-24.8%)</td><td>2.35 (−11.9%)</td><td>1.29 (−60.8%)</td><td>1.64 (-45.4%)</td><td>3.21 (+20.6%)</td></tr><tr><td>LoopSpec</td><td>2.92(→)</td><td>2.66(→)</td><td>3.29(-)</td><td>3.00(→)</td><td>2.66()</td></tr><tr><td>Ours</td><td>4.27(+46.4%)</td><td>3.73(+40.0%)</td><td>4.07(+23.8%)</td><td>4.01(+33.6%)</td><td>4.30(+61.6%)</td></tr><tr><td rowspan="3">Ouro-2.6B</td><td>DFlash</td><td>2.24 (-30.0%)</td><td>2.30 (-18.0%)</td><td>1.35 (−60.9%)</td><td>1.72 (−46.6%)</td><td>3.42 (+19.8%)</td></tr><tr><td>LoopSpec</td><td>3.19(→)</td><td>2.80(→)</td><td>3.45(→)</td><td>3.22(-)</td><td>2.85 (→)</td></tr><tr><td>Ours</td><td>5.21(+63.1%)</td><td>4.37(+55.9%)</td><td>4.72(+36.9%)</td><td>4.76(+47.8%)</td><td>5.12(+79.2%)</td></tr><tr><td rowspan="3">Ouro-1.4B-Thinking LoopSpec</td><td>DFlash</td><td>2.52 (-6.1%)</td><td>2.40 (-1.6%)</td><td>1.88 (-28.9%)</td><td>1.92 (-27.9%)</td><td>2.62 (+18.8%)</td></tr><tr><td></td><td>2.68 (→)</td><td>2.44(→)</td><td>2.64(→)</td><td>2.66(→)</td><td>2.21 (→)</td></tr><tr><td>Ours</td><td>3.49(+30.2%)</td><td>3.10(+27.0%)</td><td>3.18(+20.6%)</td><td>3.17(+19.0%)</td><td>2.82(+27.6%)</td></tr><tr><td rowspan="3">Raven-Llama-3.2</td><td>DFlash</td><td>1.69 (-62.6%)</td><td>1.60 (−71.1%)</td><td>1.29 (−78.1%)</td><td>1.47 (-72.1%)</td><td>3.15 (−38.0%)</td></tr><tr><td>LoopSpec</td><td>4.51 (→)</td><td>5.51 (→)</td><td>5.90(→)</td><td>5.28(→)</td><td>5.09 (→)</td></tr><tr><td>Ours</td><td>5.67(+25.9%)</td><td>6.05 (+9.7%)</td><td>6.06 (+2.7%)</td><td>5.86(+11.0%)</td><td>6.19(+21.6%)</td></tr><tr><td rowspan="3">Raven-OLMo-2</td><td>DFlash</td><td>1.93 (-58.5%)</td><td>2.65 (-30.2%)</td><td>1.51 (−73.7%)</td><td>1.94 (-65.2%)</td><td>3.70 (-7.3%)</td></tr><tr><td>LoopSpec</td><td>4.65(→)</td><td>3.80(→)</td><td>5.73 (→)</td><td>5.59 (→)</td><td>3.99(→)</td></tr><tr><td>Ours</td><td>6.01(+29.4%)</td><td>5.74(+51.0%)</td><td>6.13 (+6.8%)</td><td>6.22(+11.4%)</td><td>6.14(+54.0%)</td></tr><tr><td rowspan="3">Raven-TinyLlama</td><td>DFlash</td><td>1.85 (-65.8%)</td><td>2.13 (-13.6%)</td><td>1.37 (-79.7%)</td><td>1.70 (−72.7%)</td><td>3.18 (-45.9%)</td></tr><tr><td>LoopSpec</td><td>5.41 (→)</td><td>2.46(→)</td><td>6.76(→)</td><td>6.21 (→)</td><td>5.88(→)</td></tr><tr><td>Ours</td><td>6.69(+23.5%)</td><td> $4 . 8 3 \ : ( + 9 6 . 2 \% )$ </td><td>7.82(+15.7%)</td><td>7.47(+20.3%)</td><td>7.51(+27.8%)</td></tr></table>

Table 1: Greedy decoding speedups (×) over AR. Results for LoopAhead are highlighted with shading. Parentheses show throughput changes relative to LoopSpec, with increases in green and decreases in red. Bold marks the best result. Timing and aggregation are detailed in Appendix A.5.

![](images/fb24d7f5f64f821d9aaa1b36d7dc4a8583e548f67b1e5c17b5d532da65ce2482.jpg)  
(a) Loops / output token

![](images/90bf080c8cfeb067f9b650c19f0bb17ebf8574c1f9bf404372bfea09da656573.jpg)  
(b) Candidate advances / output token

![](images/e646ff193b7876a0cc24ebaabb37e0a0026fcb30d00c60a797ad0ec9247bfa65.jpg)  
(c) Candidates advanced / loop  
Figure 3: Recurrent computation profiles on Ouro-1.4B, Ouro-2.6B, Raven-Llama-3.2, and Raven-OLMo-2, pooled over five tasks with 3,023 requests per model. (a) Average loops per output token (relative to AR), $\widetilde { D } = S _ { \mathrm { r e c } } / ( R N _ { \mathrm { o u t } } )$ . (b) Average candidate advances per output token (relative to AR), $\widetilde { W } = W / ( R { N _ { \mathrm { o u t } } } )$ . (c) Average candidates advanced per loop, ${ \bar { B } } = W / S _ { \mathrm { r e c } }$

Benchmarks. We evaluate GSM8K (1,319 questions) (Cobbe et al., 2021), MATH500 (500) (Lightman et al., 2024), HumanEval (164) (Chen et al., 2021), MBPP test (500) (Austin et al., 2021), and a fixed BBH held-out split (540 with 20 per subtask) (Suzgun et al., 2023). The BBH evaluation split is disjoint from our training split. These benchmarks cover mathematical reasoning, code generation, and general reasoning, while additional evaluations on summarization and broader instructionfollowing tasks are reported in Appendix B.7.

Metrics. We report speedups of decoding throughput over AR decoding throughput. Timing includes drafting, scheduling, and verification, but excludes prefill, model loading, and warmup. We also report average loops per output token relative to AR, $\widetilde { D } = S _ { \mathrm { r e c } } / ( R N _ { \mathrm { o u t } } )$ , and average candidate advances per output token relative to AR, $\widetilde { W } = W / ( R { N _ { \mathrm { o u t } } } )$ . Here R is the full target depth, $N _ { \mathrm { o u t } }$ excludes the first output token produced during prefill, $S _ { \mathrm { r e c } }$ counts the total number of loops, and W sums the number of candidates advanced in each loop. The average number of candidates advanced per loop is ${ \bar { B } } = W / S _ { \mathrm { r e c } }$

<table><tr><td>Model</td><td>Method</td><td>GSM8K</td><td>MATH500</td><td>HumanEval</td><td>MBPP</td><td>BBH</td></tr><tr><td rowspan="3">Ouro-1.4B</td><td>DFlash</td><td> $1 . 7 2 { \pm } 0 . 0 3 \left( - 3 0 . 4 \% \right)$ </td><td> $1 . 9 4 { \pm } 0 . 0 0 1 \left( - 1 8 . 2 \% \right)$ </td><td> $1 . 2 4 { \pm } 0 . 0 2 \left( - 5 6 . 8 \% \right)$ </td><td> $1 . 4 0 { \pm } 0 . 0 3 \left( - 4 5 . 8 \% \right)$ </td><td> $2 . 1 5 { \pm } 0 . 0 3 \left( - 5 . 1 \% \right)$ </td></tr><tr><td>LoopSpec</td><td> $2 . 4 7 { \pm } 0 . 0 2 ( - )$ </td><td> $2 . 3 7 { \pm } 0 . 0 0 1 ( - )$ </td><td> $2 . 8 7 { \pm } 0 . 0 1 ( - )$ </td><td> $2 . 5 8 { \pm } 0 . 0 2 ( - )$ </td><td> $2 . 2 6 { \pm } 0 . 0 2 ( - )$ </td></tr><tr><td>Ours</td><td> $3 . 0 3 { \pm } 0 . 0 4 \ : ( + 2 2 . 8 \% )$ </td><td>–  $3 . 0 6 { \pm } 0 . 0 1 \ : ( + 2 8 . 9 \% )$ </td><td> $3 . 4 2 { \pm } 0 . 0 2 \ : ( { + } 1 8 . 9 \% )$ </td><td> $3 . 0 0 { \pm } 0 . 0 4 \left( + 1 6 . 2 \% \right)$ </td><td> $3 . 0 9 { \pm } 0 . 0 3 \left( + 3 6 . 4 \% \right)$ </td></tr><tr><td rowspan="3">Raven-Llama-3.2 LoopSpec</td><td>DFlash</td><td> $1 . 1 2 { \pm } 0 . 0 3 \left( - 5 2 . 6 \% \right)$ </td><td> $1 . 1 6 { \pm } 0 . 0 1 \left( - 5 8 . 2 \% \right)$ </td><td> $1 . 0 3 { \pm } 0 . 0 3 \left( - 6 6 . 6 \% \right)$ </td><td> $1 . 0 7 { \pm } 0 . 0 1 \left( - 6 1 . 2 \% \right)$ </td><td> $2 . 0 0 { \pm } 0 . 0 6 \left( - 3 8 . 5 \% \right)$ </td></tr><tr><td></td><td> $2 . 3 7 { \pm } 0 . 0 5 ( - )$ </td><td> $2 . 7 8 { \pm } 0 . 0 2 ( - )$ </td><td> $3 . 1 0 { \pm } 0 . 1 3 ( - )$ </td><td> $2 . 7 6 { \pm } 0 . 0 8 \left( - \right)$ </td><td> $3 . 2 6 { \pm } 0 . 0 7 ( - )$ </td></tr><tr><td>Ours</td><td> $3 . 2 4 { \pm } 0 . 0 3 \ : ( { + 3 6 . 8 \% } )$ </td><td> $3 . 6 0 { \pm } 0 . 0 7 \ : ( + 2 9 . 4 \% )$ </td><td> $3 . 8 0 { \pm } 0 . 0 2 \left( + 2 2 . 9 \% \right)$ </td><td> $3 . 5 8 { \pm } 0 . 0 2 \ : ( { \pm } 2 9 . 9 \% )$  </td><td> $4 . 3 2 { \pm } 0 . 0 3 \ : ( + 3 2 . 6 \% )$ </td></tr></table>

Table 2: Sampling decoding speedups (×) over AR at T = 1 and top-p = 0.9, reporting mean ± sample standard deviation over seeds 0, 42, and 271828. Parentheses show average throughput changes relative to LoopSpec under three seeds. Colors and boldface follow Table 1.

## 4.2 Evaluation Results

Under greedy decoding, LoopAhead outperforms LoopSpec and our DFlash adaptation on all 30 model–benchmark pairs while reproducing AR outputs, as shown in Table 1. Speedups over AR range from 2.82× to 7.82×, with the highest on Raven-TinyLlama/HumanEval. The largest additional gains over LoopSpec occur on mathematical or general reasoning tasks for every model, reaching 79.2% on Ouro-2.6B/BBH and 96.2% on Raven-TinyLlama/MATH500. The speedups also extend to HumanEval and MBPP, whose prompts are not used for drafter training, demonstrating LoopAhead’s ability to generalize to code-generation benchmarks not used for drafter training.

Under sampling at $T = 1 \mathrm { ~ a n d ~ t o p } { - p } = 0 . 9 $ LoopAhead remains the fastest method on all ten pairs, achieving mean speedups of 3.00–4.32× over AR and throughput gains of 16.2–36.8% over LoopSpec as shown in Table 2. The gains on reasoning tasks persist under sampling, exceeding 28% on both MATH500 and BBH for both models. Results are consistent across three seeds, with LoopAhead speedup standard deviations of 0.01– 0.07×. Further 95% interval analysis under both settings is provided in Appendix B.3.

## 4.3 Ablation and Analysis

We conduct ablation studies using 428 held-out requests across five benchmarks per model to answer the following questions.

## Q1: Is our method a simple combination of the LAD drafter and the scheduling policy?

Table 3 and Figure 5 show that LAD alone can slow decoding, while combining it with scheduling gives substantial gains and reduces candidate advances spent on discarded nodes for both models. With budgets, admission rules, and priorities fixed, enabling conflict-based suspension improves speedup on every benchmark for both models. This shows that feedback helps use the execution budget more effectively. LAD and scheduling therefore serve complementary roles, generating candidates early and controlling their computation.

![](images/f1ee1352986310a6249be9072a06c1972a14157a026cf8b3242047654a275e41.jpg)

![](images/c97a820c7c79023cd45757fa834e0a6597b846774ee3302a7cc33dabe65d3197.jpg)  
(a) Drafter parameters  
(b) Time per loop  
Figure 4: Drafter size and per-loop decode cost. (a) Learned parameters on Ouro-1.4B, Ouro-2.6B, and Raven-Llama-3.2, excluding shared target weights and including R4 residual for Raven, $P _ { \mathrm { a d d } }$ . (b) Average decode time per loop, including drafting and scheduling, on Ouro-1.4B and Raven-Llama-3.2, pooled over five tasks with 3,023 requests per model, $\bar { t } _ { \mathrm { r e c } } = T _ { \mathrm { d e c } } / S _ { \mathrm { r e c } } .$

Q2: How much parallel computation is needed? Figure 3 shows that LoopAhead reduces average loops per output token relative to AR from 0.285 to 0.162 on Ouro-1.4B compared with LoopSpec, while increasing average candidate advances per output token relative to AR from 1.14 to 1.91. However, Raven-Llama and Raven-OLMo achieve higher throughput while advancing fewer candidates per loop and making 17.7% and 30.6% fewer candidate advances per output token, respectively. LoopAhead also makes fewer candidate advances per output token than DFlash on all four models shown. Thus, wider parallelism and selective execution offer two routes to faster decoding, with the latter reducing target computation on Raven.

<table><tr><td>Model</td><td>LAD</td><td>Sched.</td><td>Susp.</td><td>Vocab.</td><td>GSM8K</td><td>MATH500</td><td>HumanEval</td><td>MBPP</td><td>BBH</td></tr><tr><td rowspan="6">Ouro-1.4B</td><td>X</td><td>X</td><td>X</td><td></td><td>2.88×</td><td>2.60×</td><td>3.24×</td><td>2.90×</td><td>2.60×</td></tr><tr><td>X</td><td>√</td><td>√</td><td>一</td><td>2.87×</td><td>2.60×</td><td>3.23×</td><td>2.89×</td><td>2.59×</td></tr><tr><td>V</td><td>X</td><td>X</td><td>X</td><td>2.89×</td><td>2.38×</td><td>2.52×</td><td>2.60×</td><td>2.78×</td></tr><tr><td>√</td><td>√</td><td>√</td><td>X</td><td>4.13×</td><td>3.53×</td><td>3.90×</td><td>3.70×</td><td>4.03×</td></tr><tr><td>√</td><td>√</td><td>X</td><td>√</td><td>3.86×</td><td>3.19×</td><td>3.43×</td><td>3.45×</td><td>3.71×</td></tr><tr><td>√</td><td>V</td><td>V</td><td>V</td><td>4.26×</td><td>3.61×</td><td>4.01×</td><td>3.80×</td><td>4.13×</td></tr><tr><td rowspan="6">Raven-Llama-3.2</td><td>X</td><td>X</td><td>X</td><td>一</td><td>4.32×</td><td>4.42×</td><td>4.68×</td><td>4.53×</td><td>4.37×</td></tr><tr><td>X</td><td>√</td><td>√</td><td></td><td>4.30×</td><td>4.42×</td><td>4.67×</td><td>4.52×</td><td>4.36×</td></tr><tr><td>√</td><td>X</td><td>X</td><td>X</td><td>3.82×</td><td>3.56×</td><td>3.75×</td><td>3.63×</td><td>4.01×</td></tr><tr><td>√</td><td>√</td><td>√</td><td>X</td><td>5.44×</td><td>5.74×</td><td>5.59×</td><td>5.56×</td><td>5.91×</td></tr><tr><td>√</td><td>√</td><td>X</td><td>√</td><td>4.79×</td><td>4.61×</td><td>4.80×</td><td>4.67×</td><td>5.04×</td></tr><tr><td>√</td><td>V</td><td>V</td><td>V</td><td>5.79×</td><td>6.09×</td><td>5.97×</td><td>5.93×</td><td>6.23×</td></tr></table>

Table 3: Greedy decoding speedups (×) over AR on 428 requests per model. Sched. denotes our full scheduling framework, with conflict-based suspension controlled separately by Susp. Vocab. denotes draft vocabulary reduction with 8,192 base tokens. Each model’s last two rows are measured together, varying only suspension with budgets, admission rules, and priorities fixed. Native proposal depths are R1/R2 for Ouro and R4/R8 for Raven. Bold marks the best result per model and task.

![](images/90371c649f415c547671ab1b8f83e33e52a870f5921dcc527322d815a5df4650.jpg)  
(a) Ouro-1.4B

![](images/3f00498d68e6571b0c20a8d0d9a90775a26a270545cc42323e856c28c2026c02.jpg)  
(b) Raven-Llama-3.2

<table><tr><td>Metric</td><td>Off</td><td>On</td></tr><tr><td> $\widetilde { W } _ { \mathrm { d i s c } }$ </td><td>2.19</td><td>0.93</td></tr><tr><td>D</td><td>0.725</td><td>0.656</td></tr><tr><td>B</td><td>17.87</td><td>11.99</td></tr><tr><td>thr</td><td>108.2</td><td>160.4</td></tr></table>

<table><tr><td>Metric</td><td>Off</td><td>On</td></tr><tr><td> $\widetilde { W } _ { \mathrm { d i s c } }$ </td><td>1.72</td><td>0.13</td></tr><tr><td>D</td><td>3.867</td><td>2.709</td></tr><tr><td>B</td><td>22.86</td><td>13.62</td></tr><tr><td>thr</td><td>70.6</td><td>109.8</td></tr></table>

Figure 5: Scheduling results with LAD and draft vocabulary reduction held fixed over 428 questions per model across five benchmarks. Bars show $\widetilde { W } = W / ( R { N _ { \mathrm { o u t } } } )$ the average number of candidate advances per output token relative to AR, and the dashed line denotes AR. Red annotations indicate the percentage reduction in candidate advances when scheduling is enabled. The tables report candidate advances on subsequently discarded nodes normalized by the same AR reference, $\widetilde { W } _ { \mathrm { d i s c } } = W _ { \mathrm { d i s c } } / ( R N _ { \mathrm { o u t } } ) ;$ average loops per output token, $D = S _ { \mathrm { r e c } } / N _ { \mathrm { o u t } } ;$ average candidates advanced per loop, $\bar { B } = W / S _ { \mathrm { r e c } } ;$ and throughput (thr, in tokens/s).

## Q3: Does each loop become more expensive due to the introduction of the draft model?

As shown in Figure 4, average decode time per loop increases from 7.72 to 9.45 ms on Ouro-1.4B compared with LoopSpec, but decreases from 4.34 to 3.37 ms on Raven-Llama. Ouro compensates for the higher cost with fewer loops per output token, while Raven benefits from cheaper execution. The drafter adds 11.1M parameters on Ouro, compared with 277.9M for DFlash. On Raven-Llama, LAD and its residual adapter together contain 19.5M parameters, compared with 325.1M for DFlash, keeping the additional model storage small. The draft model adds overhead, but fewer loops or candidate advances can more than offset it.

## 5 Conclusion

We present LoopAhead as a speculative decoding framework for looped language models that uses early-loop states to predict future candidates and intermediate feedback to guide their execution. Experiments show that our framework achieves consistent speedups over AR and LoopSpec under both greedy and stochastic decoding, while remaining lossless. The execution statistics reveal model-dependent tradeoffs between parallel computation and speedups. LoopAhead’s key value lies in exploiting early predictions and translating them into useful computation, ultimately delivering substantial inference speedups.

## Limitations

We discuss three main limitations of our work: LoopAhead currently requires model-specific adaptations in drafter training and scheduling. Models with fewer recurrent iterations and those with more iterations differ in their intermediate representations and the depths at which useful feedback becomes available. We therefore adapt the drafter and select feedback depths and scheduling settings for each model family. Applying LoopAhead to a new architecture requires drafter training and scheduling choices based on development data.

Although LoopAhead consistently accelerates decoding in our evaluations, its gains over the training-free LoopSpec baseline are modest on some model–benchmark pairs. Task type and difficulty may affect how reliably the drafter predicts several future tokens. When these predictions are less accurate, additional lookahead can introduce verification work without producing enough accepted tokens to offset its cost. Scheduling helps control this work but cannot fully compensate for poor drafts. The additional training effort therefore does not yield equally large improvements on every task, and understanding these task-dependent differences requires further study.

Our scheduler responds dynamically to intermediate-loop feedback, but its rules and budgets are fixed in advance. Our evaluation uses one request per GPU. Concurrent serving introduces changing request lengths, GPU load, and memory pressure. Wider execution may also become less beneficial when the GPU is already busy. Future work can adapt drafting and execution budgets to candidate quality and changing serving workloads.

## References

Zachary Ankner, Rishab Parthasarathy, Aniruddha Nrusimha, Christopher Rinard, Jonathan Ragan-Kelley, and William Brandon. 2024. Hydra: Sequentially-dependent draft heads for medusa decoding. arXiv preprint arXiv:2402.05109.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. 2021. Program synthesis with large language models. arXiv preprint arXiv:2108.07732.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. 2024. Medusa: Simple LLM inference acceleration framework with multiple decoding heads. arXiv preprint arXiv:2401.10774.

Sahil Chaudhary. 2023. Code alpaca: An instruction-following llama model for code generation. https://github.com/sahil280114/ codealpaca.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John

Jumper. 2023. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318.

Jian Chen, Yesheng Liang, and Zhijian Liu. 2026. DFlash: Block diffusion for flash speculative decoding. arXiv preprint arXiv:2602.06036.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Yunfei Cheng, Aonan Zhang, Xuanyu Zhang, Chong Wang, and Yi Wang. 2024. Recurrent drafter for fast speculative decoding in large language models. arXiv preprint arXiv:2403.09919.

Sang Lyul Cho, Langqing Cui, Sehoon Kim, Dongsu Han, and Insu Han. 2026. LoopSpec: Pipelined selfspeculative decoding for looped transformers. arXiv preprint arXiv:2609.17184.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. 2018. Universal transformers. arXiv preprint arXiv:1807.03819.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. 2023. Enhancing chat language models by scaling high-quality instructional conversations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3029–3051, Singapore. Association for Computational Linguistics.

Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, et al. 2024. Layerskip: Enabling early exit inference and self-speculative decoding. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 12622–12642.

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. 2026. Scaling up test-time compute with latent reasoning: A recurrent depth approach. Advances in Neural Information Processing Systems, 38:41340– 41391.

Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, and Gabriel Synnaeve. 2024. Better & faster large language models via multi-token prediction. arXiv preprint arXiv:2404.19737.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Hyeongju Ha and Jae-Joon Kim. 2026. Wavefront decoding: Parallelized self-speculative decoding for looped language models. arXiv preprint arXiv:2609.23033.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Hasan Genc, Kurt Keutzer, Amir Gholami, and Yakun Sophia Shao. 2025. Speed: Speculative pipelined execution for efficient decoding. In Enhancing LLM Performance: Efficacy, Fine-Tuning, and Inference Techniques, pages 19–32. Springer.

Haoyun Jiang, Junqi He, Feng Hong, Xinlong Yang, Jianwei Zhang, Zheng Li, Zhengyang Zhuge, Zhiyong Chen, Bo Han, Junyang Lin, et al. 2026. Trispec: Ternary speculative decoding via lightweight proxy verification. arXiv preprint arXiv:2601.23180.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast inference from transformers via speculative decoding. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 19274–19286. PMLR.

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Alpacaeval: An automatic evaluator of instructionfollowing models. https://github.com/ tatsu-lab/alpaca\_eval.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024a. EAGLE-2: Faster inference of language models with dynamic draft trees. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 7421–7432, Miami, Florida, USA. Association for Computational Linguistics.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024b. Eagle: Speculative sampling requires rethinking feature uncertainty. arXiv preprint arXiv:2401.15077.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2025. EAGLE-3: Scaling up inference acceleration of large language models via training-time test. arXiv preprint arXiv:2503.01840.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In International

Conference on Learning Representations, volume 2024, pages 39578–39601.

Sean McLeish, Ang Li, John Kirchenbauer, Dayal Singh Kalra, Brian R Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Jonas Geiping, Tom Goldstein, and Micah Goldblum. 2025. Teaching pretrained language models to think deeper with retrofitted recurrence. arXiv preprint arXiv:2511.07384.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, et al. 2024. 2 olmo 2 furious. arXiv preprint arXiv:2501.00656.

Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J Reddi. 2025. Reasoning with latent thoughts: On the power of looped transformers. In International Conference on Learning Representations, volume 2025, pages 14855–14881.

Kristian Schwethelm, Daniel Rueckert, and Georgios Kaissis. 2026. Depth-adaptive inference of looped language models via continuous depth batching. arXiv preprint arXiv:2608.09444.

Abigail See, Peter J Liu, and Christopher D Manning. 2017. Get to the point: Summarization with pointergenerator networks. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1073– 1083.

Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. 2018. Blockwise parallel decoding for deep autoregressive models. Advances in Neural Information Processing Systems, 31.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed H Chi, Denny Zhou, et al. 2023. Challenging big-bench tasks and whether chain-of-thought can solve them. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 13003–13051.

Lefan Zhang, Xiaodan Wang, Yanhua Huang, and Ruiwen Xu. 2025. Learning harmonized representations for speculative sampling. In International Conference on Learning Representations, volume 2025, pages 35367–35388.

Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. 2024. Tinyllama: An open-source small language model. arXiv preprint arXiv:2401.02385.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Preprint, arXiv:2306.05685.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. Preprint, arXiv:2311.07911.

Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh, Sanjiv Kumar, Jean-François Kagy, and Rishabh Agarwal. 2024. Distillspec: Improving speculative decoding via knowledge distillation. In International Conference on Learning Representations, volume 2024, pages 32011–32050.

Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, et al. 2025. Scaling latent reasoning via looped language models. arXiv preprint arXiv:2510.25741.

## A Experimental Setting

## A.1 Models

We evaluate six models from the Ouro (Zhu et al., 2025) and Raven (McLeish et al., 2025) families. Table 4 lists the checkpoint identifiers and the revisions recorded in our experiment configurations. All target-model parameters remain frozen during drafter training and inference.

Ouro. We use Ouro-1.4B, Ouro-2.6B, and Ouro-1.4B-Thinking (Zhu et al., 2025). The 1.4B checkpoints contain 24 shared Transformer layers, while Ouro-2.6B contains 48. Their hidden size is 2,048, feed-forward width is 5,632, and attention uses 16 query heads and 16 key/value heads with head dimension 128. The vocabulary contains 49,152 tokens. We fix the target computation to four recurrent passes and disable adaptive early exit.

Raven. We use the recurrence-32 checkpoints (McLeish et al., 2025) retrofitted from Llama-3.2 (Grattafiori et al., 2024), OLMo-2-0425 (OLMo et al., 2024), and TinyLlama-3T (Zhang et al., 2024). All three targets execute 32 recurrent passes while retaining their checkpoint-specific prelude, recurrent core, and coda. For example, Raven-Llama-3.2 has four prelude layers, six shared core layers and four coda layers. Raven-TinyLlama has eight layers in its recurrent core. Intermediate proposals are obtained through the corresponding coda readout, rather than by directly applying the vocabulary projection to an unfinished recurrent latent.

Tokenization and thinking mode. Inputs are tokenized separately with each checkpoint’s tokenizer. Base-model prompts use add\_special\_tokens=False, without an additional automatically inserted beginningof-sequence token. For Ouro-1.4B-Thinking, we wrap the task prompt in the released chat template with enable\_thinking=True and a generation prompt. Generation stops when the endof-sequence token is produced or the task-specific output budget is reached. Task-specific completion delimiters are disabled for this checkpoint to avoid interrupting its reasoning.

## A.2 Datasets

Evaluation benchmarks. We evaluate five benchmarks covering mathematical reasoning, code generation, and general reasoning. GSM8K (Cobbe et al., 2021) contains gradeschool mathematical word problems. We use all 1,319 test questions with three fixed chain-ofthought demonstrations. MATH500 (Lightman et al., 2024) contains 500 competition-level mathematics problems from the MATH test set, covering several mathematical subjects and difficulty levels. We use four fixed Minerva-style demonstrations, while drafter training draws from the separate 7,500-example MATH training split (Hendrycks et al., 2021). HumanEval (Chen et al., 2021) contains 164 Python functioncompletion problems with function signatures and specifications. MBPP (Austin et al., 2021) contains short Python programming tasks, and we use its 500 test examples with task identifiers 11–510. Both code benchmarks use zero-shot prompts. HumanEval retains the original function specification, while MBPP includes the task description and first public example assertion. BBH (Suzgun et al., 2023) covers 27 challenging reasoning subtasks. We use 20 held-out questions per subtask, giving 540 evaluation questions, with three fixed chain-of-thought demonstrations for each subtask. Each subtask also contributes 75 training, 10 development, and 10 confirmation questions to the drafter data. These partitions are disjoint, and demonstration questions are excluded from data selection. Table 5 lists sample counts, prompt formats, and output budgets.

Additional evaluation benchmarks. The supplementary experiments cover CNN/DailyMail (See et al., 2017), AlpacaEval (Li et al., 2023), IFEval (Zhou et al., 2023), and MT-Bench (Zheng et al., 2023). CNN/DailyMail pairs news articles with summary highlights and provides our news-summarization prompts. AlpacaEval contains diverse user instructions, while IFEval contains instructions with verifiable requirements such as length and formatting constraints. We select 128 prompts from each of these three datasets. For MT-Bench, we use the first turns of 38 questions from writing, roleplay, extraction, and humanities. The maximum output lengths are 256, 512, 1,024, and 512 tokens for the four datasets, respectively. All evaluated methods use the same selected prompts and output budgets. Results appear in Appendix B.7. For these tasks, Ouro uses its released chat template and Raven uses completion prompts.

Mixed-domain drafter data. All teacher trajectories for training LAD drafters are based on the same prompt pool in Table 6. The sources are GSM8K training questions, Python-filtered CodeAlpaca (Chaudhary, 2023) instructions, the original MATH training split, the disjoint BBH partitions described above, and the train\_sft split of UltraChat (Ding et al., 2023). CodeAlpaca examples explicitly targeting other programming languages are excluded. For UltraChat, we retain the first user request rather than using the supplied assistant response as the training target. We construct three disjoint prompt sets containing 9,025 training, 782 development, and 782 confirmation prompts. The development set is used to select the checkpoint with the highest average accepted prefix length across the five domains. The selected checkpoint is then evaluated once on the confirmation set, which is not used for checkpoint selection.

<table><tr><td>Model</td><td>Hugging Face checkpoint</td><td>Revision</td></tr><tr><td>Ouro-1.4B</td><td>ByteDance/Ouro-1.4B</td><td>574fa66cb8bf</td></tr><tr><td>Ouro-2.6B</td><td> $\mathtt { B y t e D a n c e / O u r o - 2 . 6 B }$ </td><td>1ed04250da1a</td></tr><tr><td>Ouro-1.4B-Thinking</td><td> $\bar { \mathrm { B y t } } \mathsf { e D a n c e } / \mathsf { o u r o } - 1 . 4 \mathsf { B } \mathrm { - T h i n k i n g }$ </td><td>3aaa2224253a</td></tr><tr><td>Raven-Llama-3.2</td><td> $\mathtt { s m c l e i s h } / \mathtt { R e c u r r e n t - L 1 a m a - 3 } . \hat { 2 } \mathrm { - t } x \mathrm { a i n - r } \mathtt { e c u r r e n c e - } 3 2 \mathtt { i n - t } . \mathtt { d e l } \mathtt { a } \mathtt { m } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m } \mathtt { - } \mathtt { m - } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m - } \mathtt { - } \mathtt { i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - } \mathtt { i } \mathtt { m - i } \mathtt { m - } \mathtt { i } \mathtt { m - } \mathtt { i } \mathtt { m - } \mathtt { - } \mathtt { i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt { m - i } \mathtt \mathtt { m - i } \mathtt { m - i } \mathtt \mathtt { m - i } \mathtt \mathtt { m - i } \mathtt \mathtt { m - i } \mathtt \mathtt { m - i } \mathtt \mathtt { m - i } \mathtt \mathtt \mathtt { m - i } \mathtt \mathtt \mathtt { m - i } \mathtt$ </td><td> $\mathtt { a } 5 \mathtt { f } 6 \mathtt { f } 1 2 6 \mathtt { e } 9 \mathtt { d } 9$ </td></tr><tr><td>Raven-OLMo-2</td><td> $\mathtt { s m c l e i s h \mathrm { / R e c u r r e n t - O L M o - } 2 - 0 4 2 5 - t r a i n - r e c u r r e n c e - 3 2 \pi i t s h } _ { \alpha } .$ </td><td> $\ e { 1 2 0 8 } \ e { 9 a \pm 6 1 d }$ </td></tr><tr><td>Raven-TinyLlama</td><td> $\mathtt { s m c l e i s h } / \mathtt { R e c u r r e n t - T i n y L 1 a m a - 3 T - t r a i n - r e c u r r e n c e - 3 2 \mu _ { 0 } }$ </td><td> $2 5 7 9 \mathrm { b } 6 \mathsf { a } 0 \mathsf { e } 9 8 3$ </td></tr></table>

Table 4: Target checkpoints. Revisions are abbreviated to their first 12 hexadecimal characters.

<table><tr><td>Benchmark</td><td>Examples</td><td>Prompt</td><td>Max. tokens</td></tr><tr><td>GSM8K</td><td>1,319</td><td>3-shot CoT</td><td>512</td></tr><tr><td>MATH500</td><td>500</td><td>4-shot</td><td>2,048</td></tr><tr><td>HumanEval</td><td>164</td><td>0-shot</td><td>1,024</td></tr><tr><td>MBPP</td><td>500</td><td>0-shot</td><td>1,024</td></tr><tr><td>BBH</td><td>540</td><td>3-shot CoT</td><td>1,024</td></tr></table>

Table 5: Benchmark settings for our evaluation experiments, including example numbers, prompt format and output budgets.
<table><tr><td>Source</td><td>Train</td><td>Dev.</td><td>Confirm.</td></tr><tr><td>GSM8K train</td><td>2,000</td><td>128</td><td>128</td></tr><tr><td>Python CodeAlpaca</td><td>2,000</td><td>128</td><td>128</td></tr><tr><td>MATH train</td><td>2,000</td><td>128</td><td>128</td></tr><tr><td>BBH custom split</td><td>2,025</td><td>270</td><td>270</td></tr><tr><td>UltraChat</td><td>1,000</td><td>128</td><td>128</td></tr><tr><td>Total</td><td>9,025</td><td>782</td><td>782</td></tr></table>

Table 6: Training, development, and confirmation prompts used to collect teacher trajectories for modelspecific LAD drafters.

Data isolation. We protect evaluation questions and few-shot demonstrations before selecting training, development, and confirmation requests. Exact duplicates are detected after Unicode NFKC normalization, case normalization, and whitespace normalization. Near-duplicate filtering uses token 5-gram Jaccard similarity of at least 0.9 with at least eight shared 5-grams, while retaining punctuation and mathematical operators. For BBH, 5- grams shared by at least 20% of the examples in the same subtask are ignored when computing similarity to reduce the influence of repeated task instructions. These checks remove exact and nearduplicate questions across the data partitions.

## A.3 Baselines

Autoregressive decoding. The AR baseline completes all recurrent passes before producing the next token. It uses the same target checkpoint, tokenizer, prompts, and generation settings as the speculative methods. AR provides the full-depth reference for greedy output checks and the denominator for the reported decoding throughput speedups.

LoopSpec. Our principal baseline is the trainingfree, dual-proposal LoopSpec method (Cho et al., 2026). Its first and second proposal depths are (1, 2) for Ouro and (2, 8) for Raven, with final verification at depths 4 and 32, respectively. We retain its gated second proposal, cascading verification and reuse of accepted speculative descendants, and reproduce the baseline’s scheduling semantics from the official implementation and align its target backend with LoopAhead.

DFlash. We train a local adaptation of DFlash (Chen et al., 2026) for each of the six targets, using frozen target features and the mixed-domain training trajectories. The five-layer drafter is trained for 10,000 updates with batch size 32, 32 sampled anchors per sequence, learning rate $6 \times 1 0 ^ { - 4 }$ , maximum sequence length 3,072, and training block size 16. The masked-block cross-entropy uses positional decay scale 7. We evaluate checkpoints every 500 updates on 40 fixed development prompts, with eight from each training domain. We select the checkpoint with

<table><tr><td>Model group</td><td> $d _ { 1 }$ </td><td> $d _ { 2 }$ </td><td>R</td><td>K</td><td> $B _ { \mathrm { r e t } }$ </td><td> $B _ { \mathrm { e x e } }$ </td><td> $d _ { \mathrm { g u a r d } }$ </td><td>Stride</td></tr><tr><td>All Ouro checkpoints</td><td>1</td><td>2</td><td>4</td><td>3</td><td>32</td><td>16</td><td>2</td><td>1</td></tr><tr><td>Raven-Llama / OLMo</td><td>4</td><td>8</td><td>32</td><td>2</td><td>32</td><td>16</td><td>1</td><td>2</td></tr><tr><td>Raven-TinyLlama</td><td>2</td><td>8</td><td>32</td><td>3</td><td>48</td><td>32</td><td>1</td><td>2</td></tr></table>

Table 7: Greedy LoopAhead execution parameters.

ing off and on.

the highest mean accepted-prefix length across domains, counting the anchor token. Serving block size is selected from {4, 8, 16} on 20 fixed development prompts separately for greedy decoding and sampling. Our implementation of DFlash uses the same target checkpoints, saved evaluation inputs, recurrent depths, and stopping rules as the other methods. We use hidden states from five layers of the target network unrolled to its full recurrent depth. For $L _ { \mathrm { t a r g e t } }$ layer executions, the selected zero-based indices are round $( 1 + j ( L _ { \mathrm { t a r g e t } } - 4 ) / 4 )$ for $j = 0 , \ldots , 4$ Raven’s layer count includes the prelude and final coda. The selected states are concatenated, linearly projected, and normalized, then supplied to the key and value projections of every drafter layer. For Ouro-1.4B and Ouro-2.6B, the best checkpoints occur at updates 5,500 and 6,000, with development accepted-prefix lengths of 4.59 and 4.84, compared with 4.35 and 4.74 at update 10,000. Ouro-1.4B-Thinking selects the final checkpoint, with its accepted-prefix length increasing only from 3.500 at update 8,000 to 3.503 at update 10,000. For Raven-Llama, Raven-OLMo, and Raven-TinyLlama, the selected checkpoints occur at updates 5,000, 9,000, and 5,500, with development accepted-prefix lengths of 4.14, 5.24, and 4.22. At update 10,000, these values are 3.90, 5.19, and 4.03, respectively.

Across the five greedy benchmarks, DFlash commits an average of 2.47–2.94 tokens per verification across the six models, pooling 3,023 requests per model. Ouro-1.4B and Raven-Llama-3.2 commit 2.88 and 2.47 tokens per verification, with drafttoken acceptance rates of 27.1% and 9.9%, respectively. Their development-selected block sizes are 8 and 16. The committed-token counts include correction or bonus tokens.

## A.4 Additional Experimental Details

Scheduling-only controls. The LAD-off rows in Table 3 retain native proposals and the same feedback depths as the corresponding LAD-on configurations. Ouro uses R1/R2, and Raven-Llama uses R4/R8. Figure 5 instead holds LAD and draft vocabulary reduction fixed and compares schedul-

Factorial ablation scope. The component ablations use a fixed subset of 428 evaluation requests per model, comprising 128 from GSM8K, 64 each from MATH500, HumanEval, and MBPP, and 108 from BBH. This subset is disjoint from the drafter training, development, and confirmation sets. LAD, scheduling, and the compact vocabulary are toggled within this fixed scope. The scheduling switch jointly controls execution limits, conflict suspension, and the associated admission and priority rules. Raven retains R4/R8 proposals and R32 verification even with LAD disabled.

## A.5 Implementation Details

Hardware and inference backend. Experiments use a server with four NVIDIA A40 GPUs, each with 48 GB of memory. Each GPU processes one request at a time, and the GPUs run independent experiment shards. Our local implementation uses PyTorch 2.6.0, Triton 3.2.0, Transformers 4.54.1, FlashAttention 2.8.3.post1, and CUDA 12.4. All methods use BF16 target weights and caches, with FP32 accumulation in target linear kernels.

Drafter architecture. The base drafter consumes a causal window of 64 actual target states and the embedding of a native anchor token. It contains two causal Transformer layers with hidden size 512, eight attention heads, and feed-forward width 1,536, using RMSNorm and a SiLU-gated feedforward network. Its outputs are mapped back to the target hidden dimension and read through the frozen target vocabulary projection described in Section 3.1. Training supervises up to seven future positions, whereas the deployed extension length is smaller and model dependent (Table 7).

Base-drafter optimization. We train each base drafter from scratch for three epochs with crossentropy supervision over valid future positions. Each epoch samples 32 anchor positions per training request. The microbatch size is eight with eight gradient-accumulation steps, giving an effective batch size of 64. We use AdamW with $( \beta _ { 1 } , \beta _ { 2 } ) ~ = ~ ( 0 . 9 , 0 . 9 5 )$ , weight decay 0.01, and gradient-norm clipping at 1.0. The learning rate warms up linearly over the first 5% of updates to $2 \times 1 0 ^ { - \hat { 4 } }$ and then follows cosine decay toward $2 \times 1 0 ^ { - 5 }$ . The training seed is 42.

Development evaluation is performed every 500 optimizer updates and at the final update, using eight fixed anchors per development request. Checkpoint selection maximizes the mean accepted-prefix length across the five development domains, with cross-entropy used to break ties. For each anchor, we count the longest correct prefix consisting of the native anchor followed by up to three drafter predictions, using the full-depth target’s greedy trajectory as reference. The selected checkpoint is evaluated once on the separate confirmation split. Target-model parameters remain frozen throughout optimization.

R4 adaptation for Raven. Raven-Llama and Raven-OLMo additionally use a token-conditioned residual head with a 64-dimensional bottleneck, adapted to actual R4 features. The previously trained R2 drafter backbone remains frozen. The residual head applies trainable normalization, projects both the drafter hidden representation and the preceding-token embedding into the 64- dimensional latent space, and maps their combined representation to a vocabulary-logit correction. It is part of the deployed drafter, not a second independently executed language model. For a base hidden prediction h and predecessor token embedding e, the correction is

$$
\begin{array} { r l } & { \mathcal { R } _ { \psi } ( h , e ) = \mathbf { U } _ { \psi } \mathrm { S i L U } \big ( \mathbf { V } _ { \psi } \mathrm { R M S N o r m } _ { \psi } ( h ) } \\ & { \qquad + \mathbf { B } _ { \psi } e \big ) . } \end{array}\tag{8}
$$

The matrices $\mathbf { V } _ { \psi }$ and $\mathbf { B } _ { \psi }$ project their inputs into a 64-dimensional latent space, while $\mathbf { U } _ { \psi }$ maps the latent representation to a vocabulary-logit correction. At inference time, the frozen backbone, originally trained on R2 features, takes the actual R4 context window $\mathbf { H } _ { t } ^ { ( 4 ) }$ and the native anchor $a _ { t }$ as input and jointly produces the base hidden predictions $\widetilde { h } _ { t , 1 : K }$ using the formulation in Section 3.1, with $r = 4$ The R4-adapted residual head then corrects the corresponding vocabulary logits by conditioning each prediction on its predecessor token, using the native anchor for the first prediction and the previously selected draft token for later predictions:

$$
\begin{array} { r l } & { z _ { 0 } = a _ { t } , } \\ & { \ell _ { t , j } = W _ { \mathrm { o u t } } \widetilde { h } _ { t , j } + \mathcal { R } _ { \psi } \big ( \widetilde { h } _ { t , j } , E ( z _ { j - 1 } ) \big ) , } \\ & { \quad z _ { j } = \arg \underset { v \in \mathcal { V } _ { \mathrm { d r a f t } } } { \operatorname* { m a x } } \ell _ { t , j } ( v ) , \qquad j = 1 , \ldots , K . } \end{array}\tag{9}
$$

Here, $z _ { j }$ is the candidate for output position $t + j ,$ and $\nu _ { \mathrm { d r a f t } }$ is the current request’s draft vocabulary. The frozen backbone computes all base hidden predictions jointly. Each logit correction then uses the preceding selected draft token.

For Raven-Llama, the base drafter contains 11,051,008 learned parameters and the residual head adds $8 , 4 7 2 , 5 7 6 .$ giving 19,523,584 learned parameters in total. During residual adaptation, the base drafter is frozen and only the residual head is optimized. For each model, this stage uses 100 training and 16 development requests from each of the five domains, totaling 500 and 80 requests, respectively. It additionally collects at most 16 online windows from each of 32 GSM8K training requests. Each training batch contains 25% online windows and 75% offline trajectory windows.

The residual head is trained for 600 updates with a microbatch size of four, effective batch size 32, and peak learning rate $5 \times 1 0 ^ { - 4 }$ . The objective averages a teacher-forced loss conditioned on teacher-generated predecessor tokens, and a selfrollout loss conditioned on the model’s own selected predecessors. The three prediction positions are weighted by (1, 0.75, 0.5). For the self-rollout term, a position is supervised only if all preceding rollout predictions are correct, so supervision is masked after the first rollout error. Development selection is performed every 100 updates using self-generated accepted-prefix length. Raven-TinyLlama instead uses its R2 base drafter without the R4 residual head.

Model-specific execution settings. Table 7 summarizes the adopted greedy LoopAhead configurations. Here, $d _ { 1 }$ is the native anchor and drafterfeature depth, $d _ { 2 }$ is the second native proposal depth, R is the final target depth, and $K$ counts additional draft tokens, excluding the native anchor. Stride is the number of loops in each pipeline step.

Ouro uses R2 feedback and prioritizes nodes closer to the committed root. Under sampling, ties are broken by the estimated probability of the candidate path matching the final output. Raven prioritizes paths formed by native proposals, then distance. Raven-Llama/OLMo use R4 feedback followed by R8, while Raven-TinyLlama uses R2 followed by R8. Suspended nodes are reconsidered as feedback or the root changes. When new feedback arrives, Raven-TinyLlama prunes an uncommitted LAD node and its descendants if its token differs from the parent’s deepest available prediction and its subtree contains no nodes on native-proposal paths. Their cached states are released.

For Ouro greedy decoding, LAD is called for an eligible anchor when free retention capacity is available. Each call predicts $K = 3$ additional tokens and retains the first min $( K , f )$ predictions, where f is the number of free retention slots.

For Raven, the call and retained lengths are both limited by K, free retention slots, spare executionbatch capacity, and the remaining output budget with one position reserved. The capacity threshold used for LAD admission can expand from 8 to 16 for Raven-Llama/OLMo, and from 16 to 20 or 20 to 24 for Raven-TinyLlama. Successive expansions require at least four newly committed tokens. Raven-Llama/OLMo allow the first expansion immediately, while Raven-TinyLlama waits until four output tokens are committed. These thresholds control draft admission, while $\boldsymbol { B } _ { \mathrm { e x e } }$ limits the number of nodes advanced in each batch.

For Ouro sampling, we calibrate candidate quality and execution costs on 16 development prompts, with eight each from GSM8K and CodeAlpaca. Let $p _ { j }$ estimate the probability that the candidate path through draft position $j$ matches the final output. Before drafting, this estimate uses draft position and distance from the root. After drafting, it also uses token confidence over the draft vocabulary, grouped into four equal-width bins. The call length $\ell$ and retained length k maximize

$$
\begin{array} { r l } & { U _ { \mathrm { c a l l } } ( \ell ) = t _ { \mathrm { s a v e } } \displaystyle \sum _ { j = 1 } ^ { \ell } p _ { j } - t _ { \mathrm { h e a d } } - \ell t _ { \mathrm { n o d e } } , } \\ & { U _ { \mathrm { k e e p } } ( k ) = t _ { \mathrm { s a v e } } \displaystyle \sum _ { j = 1 } ^ { k } \widehat { p } _ { j } - k t _ { \mathrm { n o d e } } , } \end{array}\tag{10}
$$

where $\widehat { p } _ { j }$ incorporates the observed confidence. Here $t _ { \mathrm { s a v e } }$ estimates the time saved per accepted draft token, $t _ { \mathrm { h e a d } }$ is the cost of one LAD call, and $t _ { \mathrm { n o d e } }$ is the added execution cost per retained candidate. We search $1 \leq \ell \leq \operatorname* { m i n } ( K , f )$ and $1 \leq k \leq \ell .$ , select zero when the maximum is nonpositive, and favor shorter lengths in ties. The calibrated costs are approximately $t _ { \mathrm { s a v e } } = 7 . 5 2 4$ ms, $t _ { \mathrm { h e a d } } = 0 . 6 0 2$ ms, and $t _ { \mathrm { n o d e } } = 0 . 4 0 7$ ms. We fix these estimates across benchmarks and seeds.

Draft vocabulary. For each target model, we select the 8,192 most frequent tokens in teachergenerated training outputs as the base draft vocabulary, breaking ties by token ID. At inference, we add tokens from the prompt, the EOS token, and already committed outputs. LAD uses the resulting request-specific vocabulary for its output projection. Native proposals and full-depth verification use the complete target vocabulary.

Decoding settings. Greedy evaluation covers all six checkpoints. Sampling evaluation covers Ouro-1.4B and Raven-Llama-3.2 with temperature $T = 1$ , top- $\cdot p = 0 . 9$ , and three sampling seeds {0, 42, 271828}. For each seed, every example– method pair is executed once. For sampling, temperature scaling and nucleus truncation are applied to each native distribution before constructing residual distributions. LAD uses argmax candidates, while native proposals and full-depth verification provide sampling randomness.

Timing and aggregation. Models and CUDA graphs are loaded, captured, and warmed up before timed generation. Decode time is measured with a wall-clock timer, with CUDA synchronization before and after the timed region. We measure the iterative decoding phase after prompt prefill and initial request setup. Timing includes learned drafting, native readouts, recurrent verification, sampling, branch scheduling, stopping checks, and decode-time candidate-vocabulary maintenance. It excludes model loading, graph capture, warm-up, prefill, post-run checks, and result serialization.

For a fixed model, benchmark, and decoding regime, throughput is computed as

$$
\nu _ { m } = \frac { \sum _ { i = 1 } ^ { n } N _ { \mathrm { o u t } , m , i } } { \sum _ { i = 1 } ^ { n } t _ { m , i } ^ { \mathrm { d e c o d e } } } ,\tag{11}
$$

where $N _ { \mathrm { o u t } , m , i }$ counts generated tokens in the iterative decoding phase for method m on request i. The first generated token is produced during prefill and is excluded from this count.

## B Supplementary Experiments

## B.1 Predictions at Intermediate Depths

The depth and LAD diagnostics use the same 310 questions per model, comprising 64 each from GSM8K, MATH500, HumanEval, and MBPP, and

![](images/d73fc34a46c8213afe534b538e4d178dca41e0fc9112cbc226b353b78c143f36.jpg)  
(a) Ouro-1.4B All generated

![](images/90ad1821e57ab1b9373e52cfaf5af40f1220d1aae1d8bc849e4928cb10de606f.jpg)  
(b) Raven-Llama-3.2 All generated

![](images/68befe49d9fd06750f642f164f3ab68749e0f41bdc88782976b1d93156f40c45.jpg)  
(c) Ouro-1.4B Verified

![](images/0000d5fde1127497fb19a91a63df0d5e5d44874a83d7e591320aba121cc4508a.jpg)  
(d) Raven-Llama-3.2 Verified

Figure 6: LAD token quality by extra-token offset i on five benchmarks. Panels (a,b) show full-depth agreement rate among all generated candidates $q _ { i } ^ { \mathrm { a l l } }$ . Panels (c,d) show acceptance rate among candidates checked during verification $a _ { i } ^ { \mathrm { c h e c k } }$  
![](images/c93c06256c9031871c35041aef12de3e2333acf6ae8d8e0e75f7315d7bf8ea05.jpg)

![](images/6676a486e7b898aa5327c8008665c1e3489a26571ab65cfdb9b83bccbad1f512.jpg)  
(a) Accuracy  
(b) Token agreement  
Figure 7: Raven-Llama-3.2 task accuracy at fixed exit depths and token agreement with full-depth predictions across five benchmarks, supplementing Figure 1.

54 from BBH. We measure task accuracy by generating complete answers at a fixed exit depth. Token agreement $\rho _ { r }$ compares predictions at depth r with full-depth predictions on the same AR prefixes.

Figure 7 extends the Ouro analysis to Raven-Llama-3.2. Exiting at R2 gives 50.0% accuracy on GSM8K and 20.3% on MATH500, compared with 65.6% and 32.8% at R32. Token agreement improves rapidly in the early loops, reaching 97.0– 99.1% at R4 and further increasing to 99.3–99.7% at R8 across the five tasks. These results support intermediate-loop drafting and scheduling with fulldepth verification.

## B.2 Quality of LAD Draft Tokens

Figure 6 evaluates each additional position i beyond the native anchor. We report $q _ { i } ^ { \mathrm { a l l } }$ , the fraction of all generated candidates matching the full-depth prediction under their own prefixes, and $a _ { i } ^ { \mathrm { c h } \mathrm { { \bar { e } c k } } }$ the fraction accepted among candidates checked against their committed parent’s final prediction.

Draft quality decreases at later positions. On Ouro/GSM8K, $q _ { i } ^ { \mathrm { a l l } }$ falls from 64.3% at the first extra token to 36.1% at the third, while $a _ { i } ^ { \mathrm { c h e c k } }$ falls from 68.7% to 55.9%. On Raven/GSM8K, the corresponding values decrease from 51.9% to 34.8% and from 55.1% to 44.3% over two extra positions. Later drafts remain useful, but their lower quality favors short extensions and selective execution.

## B.3 Confidence Intervals for Main Results

We compute pointwise 95% confidence intervals from the saved full-benchmark results using 10,000 paired-question bootstrap resamples. Each resample keeps methods and sampling seeds paired by question and recomputes their pooled throughputs. BBH is stratified by its 27 subtasks. Sampling gains are averaged across seeds 0, 42, and 271828 after computing the throughput ratio within each seed. These intervals reflect variation across question samples.

Figures 8 and 9 show throughput gains over LoopSpec, with intervals above zero for 29 of 30 greedy pairs and all ten sampling pairs.

## B.4 Choice of Drafting and Feedback Depths

We evaluate 11 combinations of draft depth $d _ { 1 }$ and intermediate-feedback depth $d _ { 2 }$ using 428 questions per model from the evaluation set, disjoint from the drafter training, development, and confirmation sets. The main configurations were fixed before these sweeps. Full-depth verification remains at R32, and the other inference settings are unchanged. For each target model, all depth configurations share its frozen R2-trained drafter backbone. We use no residual correction at R2, reuse the existing R4 residual heads, and train R6 residual heads following the procedure in Appendix A.5.

Throughput gain (%)  
![](images/39c3cf0b5ad01c5d908565259bcf5396dd7270e4a340a499440c29ac41a35d9c.jpg)  
(a) Ouro-1.4B

![](images/f9aaebc13f6811d6371e81ee6862a439e0279279a0cb08fcb6951259ab3a9390.jpg)

![](images/90f0826067c39d4b245bd78df635858af634ace1347371e9e346566602aace31.jpg)

![](images/5ce7aac94b3bf3a1b33ee7714d2353b7c267edb13a915197c558f10765b7b8ed.jpg)  
(d) Raven-Llama-3.2

(b) Ouro-2.6B  
(c) Ouro-1.4B-Thinking  
![](images/a87d1a4cb602f24f86946f848463712214ff3f8785df13739a1538bb500da5b7.jpg)  
(e) Raven-OLMo-2

![](images/eb54dce05a4e998eb1b42aac2b847d7652b8712f3054793281ae176ed74c865c.jpg)  
(f) Raven-TinyLlama

Figure 8: Greedy throughput gains of LoopAhead over LoopSpec across six models and five benchmarks. Points show observed gains and black bars show 95% confidence intervals. The dashed line marks equal throughput.  
![](images/7f049b9a1f09d8e6de0115e65a5d5027fd7675a02cd045bd46e40a278c1af9ae.jpg)  
(a) Ouro-1.4B

![](images/d5634a9caa56d3fc1235ed76d817cad949286d601141a7e67ab4ce4622e9b1de.jpg)  
(b) Raven-Llama-3.2  
Figure 9: Sampling throughput gains of LoopAhead over LoopSpec at $T = 1$ and top- $\cdot p = 0 . 9$ . Points show mean gains over three seeds, and black bars show 95% confidence intervals. The dashed line marks equal throughput.

As shown in Figure 10, Raven-Llama and Raven-OLMo achieve their highest average relative throughput with R4 drafting and R8 feedback, while R4–R12 performs similarly. Raven-TinyLlama instead favors R2–R8. These results support the model-specific drafting depths and shared R8 feedback used in our main experiments. Deeper readouts do not consistently improve speed, since they also delay candidate generation or scheduling feedback.

## B.5 Effect of Draft Length

Using the same 428 requests per model as the component ablation, we vary the number of additional LAD tokens K while keeping the trained drafter and scheduling settings fixed. Throughput is pooled across the five tasks, and confidence intervals use paired question resampling within each task.

In Figure 11, Ouro reaches its highest measured speedup of 3.89× at $K = 3 .$ , close to 3.87× at $K \ = \ 2 .$ , and falls to 3.80× at $K = 4$ . Raven improves from 5.85× at K = 1 to 6.03× at $K = 2$ with little change at $K = 3$ . The benefit of longer drafts therefore levels off at a small K, supporting the main settings of three extra tokens for Ouro and

<table><tr><td>Model</td><td>Method</td><td>CNN/DailyMail</td><td>AlpacaEval</td><td>IFEval</td><td>MT-Bench</td></tr><tr><td rowspan="3">Ouro-2.6B</td><td>DFlash</td><td> $0 . 9 4 \left( - 6 3 . 4 \% \right)$ </td><td> $1 . 1 6 \left( - 6 2 . 6 \% \right)$ </td><td> $1 . 1 4 \left( - 6 3 . 2 \% \right)$ </td><td> $1 . 0 6 ( - 6 1 . 6 \% )$ </td></tr><tr><td>LoopSpec</td><td>2.56(→)</td><td>3.11 (→)</td><td> $3 . 1 1 ( - )$ </td><td>2.75 (→)</td></tr><tr><td>Ours</td><td>2.62(+2.4%)</td><td> $3 . 4 3 \ : ( + 1 0 . 4 \% )$ </td><td>3.33(+7.3%)</td><td>2.81(+2.4%)</td></tr><tr><td rowspan="3">Raven-Llama-3.2 LoopSpec</td><td>DFlash</td><td>0.94 (-76.5%)</td><td> $1 . 3 3 \ : ( - 7 1 . 2 \% )$ </td><td> $1 . 2 0 ( - 7 6 . 7 \% )$ </td><td>1.17 (−74.0%)</td></tr><tr><td></td><td>4.01 (→)</td><td>4.60(→)</td><td> $5 . 1 3 ( - )$ </td><td>4.51 (→)</td></tr><tr><td>Ours</td><td>4.51(+12.3%)</td><td>5.31 (+15.4%)</td><td>5.39(+5.0%)</td><td>5.14(+14.0%)</td></tr><tr><td rowspan="3">Raven-OLMo-2</td><td>DFlash</td><td>1.05 (−73.1%)</td><td>1.34 (-74.3%)</td><td>1.49 (-72.1%)</td><td>1.33 (−70.7%)</td></tr><tr><td>LoopSpec</td><td>3.89 (→)</td><td>5.22(→)</td><td>5.32(→)</td><td>4.55(→)</td></tr><tr><td>Ours</td><td>4.48(+15.1%)</td><td>5.56(+6.5%)</td><td> ${ \pm . 7 \mathbf { 0 } } \left( { + 7 . 3 \% } \right)$ </td><td>5.36(+17.8%)</td></tr><tr><td rowspan="3">Raven-TinyLlama LoopSpec</td><td>DFlash</td><td> $0 . 9 9 \left( - 7 7 . 7 \% \right)$ </td><td> $1 . 3 1 \ : ( - 7 4 . 5 \% )$ </td><td> $1 . 2 7 \ : ( - 7 6 . 8 \% )$ </td><td> $1 . 1 8 \left( - 7 6 . 1 \% \right)$ </td></tr><tr><td></td><td> $4 . 4 6 ( - )$ </td><td> $5 . 1 1 ( - )$ </td><td> $5 . 4 6 ( - )$ </td><td> $4 . 9 3 ( - )$ </td></tr><tr><td>Ours</td><td> ${ \pm } 5 . 2 8 \left( + 1 8 . 4 \% \right)$ </td><td> ${ \bf 6 . 0 5 } \left( + 1 8 . 4 \% \right)$ </td><td> $6 . 3 7 \left( + 1 6 . 8 \% \right)$ </td><td> ${ \pmb 5 } . 7 7 ( + 1 7 . 1 \% )$ </td></tr></table>

Table 8: Greedy decoding speedups (×) over AR on additional generation tasks, measured using pooled decode-only throughput with one pass per method. We evaluate 128 prompts each from CNN/DailyMail, AlpacaEval, and IFEval, and 38 first-turn MT-Bench prompts from writing, roleplay, extraction, and humanities. Shading, parentheses, and colors follow Table 1.

![](images/8cea5e05983f55853a7ccd547b9a4d6673004c85f190b895e10518598bfdc8a8.jpg)  
(a) Raven-Llama-3.2

![](images/b392a3ddc6d7c321a73fc494a719a65d12c595d46da515360d202d3870959cb8.jpg)  
(b) Raven-OLMo-2

![](images/5fba8bcb1bcae4e20b1b3d777923851bc4bf582dac4d95cacba0f2c12b49446f.jpg)  
(c) Raven-TinyLlama  
Figure 10: Effect of draft depth $d _ { 1 }$ and intermediate-feedback depth $d _ { 2 }$ on greedy decoding. Cells show the geometric mean of per-task throughput ratios relative to each model’s main configuration. Black boxes mark the configurations chosen for our main results in Section 4.2.

two for Raven-Llama.

## B.6 Execution Depth of Discarded Drafts

We compare scheduling on and off using 155 questions per model, comprising 32 each from GSM8K, MATH500, HumanEval, and MBPP, and 27 from BBH. For each discarded LAD candidate, we record its maximum executed loop depth $r _ { \mathrm { e n d } }$ The distribution $p _ { \mathrm { d i s c } } ( r )$ pools candidate counts across tasks and normalizes by the total number discarded. Candidates removed by direct rejection or ancestor pruning are included.

Figure 12 shows that scheduling shifts discarded candidates toward earlier depths. On Ouro, the fraction reaching R4 before removal drops from 41.79% to 9.21%. On Raven, the fraction reaching R32 falls from 44.06% to 0.07%, and 98.8% of discarded candidates execute at most eight loops with scheduling enabled. The scheduler thus limits deep execution on unproductive branches, especially in the model with more loops.

## B.7 Additional Generation Tasks

We further evaluate Ouro-2.6B and the three Raven models on CNN/DailyMail, AlpacaEval, IFEval, and a first-turn subset of MT-Bench, covering summarization and broader instruction-following tasks. We reuse the main experiments’ trained drafters and scheduling settings without further tuning.

Table 8 shows that LoopAhead outperforms LoopSpec and our DFlash adaptation across all 16 model–benchmark pairs, achieving 2.62–6.37× speedups over AR. Throughput gains over Loop-Spec range from 2.4% to 10.4% on Ouro-2.6B and from 5.0% to 18.4% on Raven. Raven-TinyLlama shows the most consistent gains, improving by 16.8–18.4% across the four tasks. In comparison, DFlash reaches 0.94–1.49× over AR. These results show that LoopAhead’s benefits extend beyond the reasoning and code benchmarks in the main evaluation, using the same trained drafters and execution policies.

![](images/3b4e9ec8a5a1cf963ba021b6614ff29e7b8a92f2f982309aec4eff140b676f9b.jpg)  
(a) Ouro-1.4B

![](images/70266e83a6a89baecd527319c027868b231cfee1e6b06e3b4f74117f82c46732.jpg)  
(b) Raven-Llama-3.2

Figure 11: Decode speedup over AR versus additional LAD draft length $K .$ , pooled across five benchmarks. Larger circles mark the main settings. Shading and error bars show 95% confidence intervals.  
![](images/f19cb8c43814fb5bf451a9a31d4f3fad0b85448209e38fd491119528c4f14d09.jpg)

![](images/17be12aff3db4ad41fa11bb4e92b8ecdf72cc751c4605a794f2593e568ee2a77.jpg)  
(a) Ouro-1.4B  
(b) Raven-Llama-3.2  
Figure 12: Distribution of discarded LAD candidates by their executed loop depth $r _ { \mathrm { e n d } }$ , with scheduling off or on. Annotations show the fraction reaching full depth. The Raven bin (8, 32) sums frequencies at depths strictly between 8 and 32.

## C Proof of Theoretical Analysis

## C.1 Proof of Theorem 1

Setup. Consider a length-L draft chain $x _ { 1 : L }$ following a correct prefix, and let $y _ { 1 : L } ^ { \star }$ be the fulldepth target’s greedy continuation from that prefix. Correctness means agreement with this continuation rather than a dataset reference answer. The chain can contain tokens from several recursive LAD calls. Probabilities are taken over requests and their length-L draft chains.

Let N be the number of consecutive matching tokens and set $J = N + 1$ . The first mismatch is at J, with $J = L + 1$ when all L tokens match. Define

$$
\alpha _ { i } = \operatorname* { P r } \ ( x _ { i } = y _ { i } ^ { \star } \mid x _ { 1 : i - 1 } = y _ { 1 : i - 1 } ^ { \star } ) ,\tag{12}
$$

and let $S _ { i } = \mathrm { P r } ( N \geq i ) , S _ { 0 } = 1$ , and $A _ { L } = \mathbb { E } [ N ]$

We compare the same candidate workload with suspension disabled and enabled, using the same admissions, budgets, priorities, and cache rules. A candidate advance runs one recurrent loop for one candidate. For the candidate with prefix $x _ { 1 : k } .$ , let $c _ { \mathrm { o f f } } ( k , d )$ and $c _ { \mathrm { o n } } ( k , d )$ count evaluations of loop d in the two executions. The counts cover the candidate’s lifetime, including any resumption. The number of avoided advances on the rejected suffix is

$$
W _ { \mathrm { a v o i d } } = \sum _ { k = J } ^ { L } \sum _ { d = 1 } ^ { R } \bigl [ c _ { \mathrm { o f f } } ( k , d ) - c _ { \mathrm { o n } } ( k , d ) \bigr ] _ { + } ,\tag{13}
$$

where $[ z ] _ { + } = \operatorname* { m a x } ( z , 0 )$ . Each candidate is identified by its full token prefix, so the same advance is matched across the two executions.

Fix a feedback depth $r \ < \ R$ . Let $\hat { y } _ { i } ^ { ( r ) }$ be the native greedy prediction for position i under prefix $x _ { 1 : i - 1 }$ , and write

$$
\gamma _ { i } = \mathrm { P r } \big ( \hat { y } _ { i } ^ { ( r ) } = y _ { i } ^ { \star } \mid J = i \big ) .\tag{14}
$$

Thus, $\gamma _ { i }$ measures how often intermediate feedback predicts the target token where the draft first goes wrong. Let $\bar { G } _ { i }$ be a nonnegative lower bound on the expected avoided advances in these cases:

$$
\mathbb { E } \big [ W _ { \mathrm { a v o i d } } \mid J = i , \hat { y } _ { i } ^ { ( r ) } = y _ { i } ^ { \star } \big ] \geq \bar { G } _ { i } .\tag{15}
$$

This mean accounts for feedback timing, near-root protection, execution budgets, and resumption.

Proof. Step 1: Locate thefirst mismatch. The first i tokens match when the preceding $i - 1$ tokens match and token i also matches. By the chain rule,

$$
S _ { i } = S _ { i - 1 } \alpha _ { i } = \prod _ { j = 1 } ^ { i } \alpha _ { j } .\tag{16}
$$

For the first mismatch to occur at $i ,$ the preceding tokens must match and token i must differ. Hence,

$$
\begin{array} { r l } { \operatorname* { P r } ( J = i ) = S _ { i - 1 } - S _ { i } } \\ { = S _ { i - 1 } ( 1 - \alpha _ { i } ) . } \end{array}\tag{17}
$$

Each position contributes one to N when the prefix through that position matches. Therefore,

$$
\begin{array} { c } { { N = \displaystyle \sum _ { i = 1 } ^ { L } \mathbf { 1 } \{ N \ge i \} , } } \\ { { { } } } \\ { { A _ { L } = \displaystyle \mathbb { E } [ N ] = \displaystyle \sum _ { i = 1 } ^ { L } S _ { i } . } } \end{array}\tag{18}
$$

Step 2: Count avoided advances under correctfeedback. Define the event

$$
E _ { i } = \{ J = i , \hat { y } _ { i } ^ { ( r ) } = y _ { i } ^ { \star } \} .\tag{19}
$$

When $J = i ,$ , all preceding tokens match the target continuation. The intermediate and full-depth predictions therefore share the same prefix. On $E _ { i } ,$ , the draft token differs from the target while feedback agrees with it:

$$
x _ { i } \neq y _ { i } ^ { \star } = \hat { y } _ { i } ^ { ( r ) } .\tag{20}
$$

This gives a feedback conflict on the draft path. By the definition of $\gamma _ { i }$ , the event has probability

$$
\begin{array} { r l } { \mathrm { P r } ( E _ { i } ) = \mathrm { P r } ( J = i ) \gamma _ { i } } & { { } } \\ { = S _ { i - 1 } ( 1 - \alpha _ { i } ) \gamma _ { i } . } \end{array}\tag{21}
$$

The events $E _ { 1 } , \dots , E _ { L }$ are mutually exclusive because a chain has only one first mismatch. Let $E _ { \mathrm { r e s t } }$ contain the remaining cases. Splitting the expectation over these events gives

$$
\begin{array} { r l } & { \mathbb { E } [ W _ { \mathrm { a v o i d } } ] = \displaystyle \sum _ { i = 1 } ^ { L } \operatorname* { P r } ( E _ { i } ) \mathbb { E } [ W _ { \mathrm { a v o i d } } \mid E _ { i } ] } \\ & { ~ + \mathbb { E } \big [ W _ { \mathrm { a v o i d } } \mathbf { 1 } \big \{ E _ { \mathrm { r e s t } } \big \} \big ] } \\ & { ~ \geq \displaystyle \sum _ { i = 1 } ^ { L } S _ { i - 1 } ( 1 - \alpha _ { i } ) \gamma _ { i } \bar { G } _ { i } . } \end{array}\tag{22}
$$

Here we use Equation 15 for each conditional mean and drop the nonnegative contribution from $E _ { \mathrm { r e s t } }$ Each term has three parts: the probability of a first mismatch at i, the probability of correct feedback there, and a lower bound on the average advances avoided in those cases. This proves the first bound.

Step 3: Express the bound through the rejected suffix. Suppose $\gamma _ { i } \geq \gamma$ and $\bar { G } _ { i } \geq g ( L - i + 1 )$ for all i, where $\gamma , g \geq 0$ . Substitution gives

$$
\mathbb { E } [ W _ { \mathrm { a v o i d } } ] \geq g \gamma \sum _ { i = 1 } ^ { L } S _ { i - 1 } ( 1 - \alpha _ { i } ) ( L - i + 1 ) .\tag{23}
$$

If the first mismatch is at $i ,$ the rejected suffix contains $L - i + 1$ tokens. The sum on the right is therefore its expected length. Since the matching prefix and rejected suffix together contain $L$ tokens,

$$
\begin{array} { r l r } {  { \sum _ { i = 1 } ^ { L } S _ { i - 1 } ( 1 - \alpha _ { i } ) ( L - i + 1 ) } } \\ & { } & { = \mathbb { E } [ L - N ] = L - A _ { L } . } \end{array}\tag{24}
$$

Equivalently, using $S _ { i - 1 } ( 1 - \alpha _ { i } ) = S _ { i - 1 } - S _ { i }$

$$
\begin{array} { c } { { \displaystyle \sum _ { i = 1 } ^ { L } ( S _ { i - 1 } - S _ { i } ) ( L - i + 1 ) } } \\ { { = L S _ { 0 } - \displaystyle \sum _ { i = 1 } ^ { L } S _ { i } = L - A _ { L } . } } \end{array}\tag{25}
$$

Combining these results yields

$$
\mathbb { E } [ W _ { \mathrm { a v o i d } } ] \geq g \gamma ( L - A _ { L } ) ,\tag{26}
$$

which proves the second bound. Here $L - A _ { L }$ gives the expected rejected-suffix length, while $g \gamma$ relates that length to the number of avoided advances.

Constant match probability. When $\alpha _ { i } = \alpha$ at every position, $S _ { i } = \alpha ^ { i }$ and the expected matchingprefix length has the closed form

$$
A _ { L } = \left\{ \begin{array} { l l } { \displaystyle \frac { \alpha ( 1 - \alpha ^ { L } ) } { 1 - \alpha } , } & { 0 \leq \alpha < 1 , } \\ { L , } & { \alpha = 1 . } \end{array} \right.\tag{27}
$$

Substitution gives the second bound directly in terms of L and $\alpha .$

Multiple feedback depths. For a set of feedback depths ${ \mathcal { R } } _ { \mathrm { f b } }$ , define

$$
\begin{array} { l } { { \displaystyle F _ { i } ^ { \mathrm { m u l t i } } = \bigcup _ { r \in \mathcal { R } _ { \mathrm { f b } } } \{ \hat { y } _ { i } ^ { ( r ) } = y _ { i } ^ { \star } \} } , } \\ { { \displaystyle \gamma _ { i } ^ { \mathrm { m u l t i } } = \operatorname* { P r } ( F _ { i } ^ { \mathrm { m u l t i } } \mid J = i ) . } } \end{array}\tag{28}
$$

This event collects cases in which at least one selected depth predicts the target token at the first mismatch. Let $\bar { G } _ { i } ^ { \mathrm { m u l t i } } ~ \geq ~ 0$ lower-bound $\mathbb { E } [ W _ { \mathrm { a v o i d } } \mid J = i , F _ { i } ^ { \mathrm { m u l t i } } ]$ . Repeating Step 2 with $E _ { i } ^ { \mathrm { m u l t i } } = \{ J = i \} \cap F _ { i } ^ { \mathrm { m u l t i } }$ gives

$$
\mathbb { E } [ W _ { \mathrm { a v o i d } } ] \geq \sum _ { i = 1 } ^ { L } S _ { i - 1 } ( 1 - \alpha _ { i } ) \gamma _ { i } ^ { \mathrm { m u l t i } } \bar { G } _ { i } ^ { \mathrm { m u l t i } } .\tag{29}
$$

Each first-mismatch case is counted once, including when several depths predict correctly.

Recursive drafting and variable lengths. The chain $x _ { 1 : L }$ may span several LAD calls: a candidate produced by one call can provide the earlyloop states for a later call. The conditional probabilities $\alpha _ { i }$ account for the preceding draft tokens across these calls, so the same proof applies to the resulting recursive chain. An early mismatch can therefore affect drafts from several calls.

When the scheduler produces a bounded random chain length Λ, apply the theorem conditional on $\Lambda = \ell .$ . Write $A _ { \ell } = \mathbb { E } [ N \mid \Lambda = \ell ]$ , and define the match probabilities, feedback probabilities, and conditional savings bounds within each length group. If the uniform bounds $g$ and $\gamma$ hold in every group, averaging over lengths gives

$$
\begin{array} { r l } {  { \mathbb { E } [ W _ { \mathrm { a v o i d } } ] \geq g \gamma \sum _ { \ell } \operatorname* { P r } ( \Lambda = \ell ) ( \ell - A _ { \ell } ) } } \\ & { = g \gamma \big ( \mathbb { E } [ \Lambda ] - \mathbb { E } [ N ] \big ) . } \end{array}\tag{30}
$$

Thus, the same bound uses the expected rejectedsuffix length when recursive expansion produces chains of different lengths.

## D Algorithm Pseudocode

Algorithm 1 uses the trained drafter $\mathsf { D } ^ { \star }$ and target $F _ { \theta } = \left( \mathcal { P } , B , \mathcal { C } \right)$ , whose components are the prelude, recurrent core, and readout. The retained tree is A, with committed root $b _ { 0 }$ and KV caches $\kappa .$ Each node b has state $\mathbf { h } _ { b }$ , depth $r _ { b } .$ , and token prefix $X ^ { ( b ) }$ . Its readout predicts the next token. The policy π includes priorities, admission and cache rules, budgets $B _ { \mathrm { r e t } } , B _ { \mathrm { e x e } }$ , and guard $d _ { \mathrm { g u a r d } }$

For logits ℓ, Π(ℓ) is a point mass at the argmax for greedy decoding, or the distribution after temperature scaling and top-p truncation for sampling. The ordered list $Q _ { b }$ stores native proposals and their distributions. Starting from the full-depth distribution $p ,$ NativeVerify accepts each proposal $u \sim q$ with probability min $\{ 1 , p ( u ) / q ( u ) \}$ . After rejection, it updates $p \gets p \ominus q$ , where

$$
\begin{array} { c l l } { { ( p \ominus q ) ( v ) = \displaystyle \frac { [ p ( v ) - q ( v ) ] _ { + } } { \sum _ { u \in \mathcal { V } _ { \mathrm { t g t } } } [ p ( u ) - q ( u ) ] _ { + } } , } }  \\ { { \lbrack a \rbrack _ { + } = \operatorname* { m a x } ( a , 0 ) . } } \end{array}\tag{31}
$$

If all proposals are rejected, it samples from the remaining p.

SelectWork follows Section 3.2, checking candidate paths against parent feedback at the configured depths. AdvanceBatch advances selected nodes by $\Delta$ loops and updates their states, caches, and depths. ReclaimDrafts applies the additional pruning rule only to Raven-TinyLlama, as described in Appendix A.5. FillNativeChildren adds or reuses eligible native-proposal children within $B _ { \mathrm { r e t } }$ . Reroot preserves committed-prefix KV states and keeps the selected subtree, or clears the speculative tree if the selected child is absent. Spawn creates the missing child with depth zero. S checks the output limit and stopping tokens. SelectAnchor chooses an unexpanded native anchor whose parent newly reached $d _ { 1 }$ . ActualWindow and DraftVocab provide the branch context and request-specific vocabulary. PrecallLength, DraftHorizon, and AdmitLength determine the allowed draft length, actual call length, and retained prefix length, respectively, following the model-specific admission rules in Appendix A.5. GraftChain attaches the retained prefix below the anchor.

## E Case Study

Complete decoding trace. Figure 13 shows the full Ouro-1.4B trace for the example in the main system figure. Starting from the anchor dozen, LAD drafts eggs, eggs, and a. Intermediate feedback suspends the conflicting suffix, and fulldepth verification commits the first eggs and the correction contain together. A later LAD call also allows the final eggs and period to be committed together. These earlier computations reduce the loop count from 15 with LoopSpec to 13 with LoopAhead.

Code generation. Table 9 illustrates both fully matching and partially matching LAD drafts. One call correctly predicts all three future tokens. Another matches i and in, while lst is corrected to t. LoopAhead reduces the loop count from 40 to 23 and improves throughput by 45.2%. Useful draft prefixes therefore help accelerate generation even when a later draft token is incorrect.

Less work per loop. Table 10 shows a different source of acceleration on Raven-Llama-3.2. LoopAhead uses a similar number of loops to Loop-Spec (252 versus 248), but reduces candidate advances from 4,786 to 3,100. Decode time falls from 1,008.1 to 775.3 ms, giving a 30.0% throughput improvement. This case shows the benefit of controlling candidate execution per loop.

Recursive drafting. Table 11 shows how LAD extends its own drafts on a BBH Booleanexpression task. The R1 state of a drafted ) supports another LAD call before full-depth verification. With expressions as the native anchor, this call correctly drafts inside, brackets, and are. LoopAhead completes the request in 55 loops instead of 146, improving throughput by 113.5% over LoopSpec.

Algorithm 1: LoopAhead Inference   
Input: Prompt x , target $F _ { \theta } ,$ trained drafter D<sup>⋆</sup>, base draft vocabulary $\mathcal { V } _ { \mathrm { b a s e } } .$ , depths d < d $< R ,$ window $w ,$   
extension limit K, stride $\Delta ,$ policy π, stopping rule S.   
Output: Generated tokens Y.   
1 $( \ell _ { 0 } , \mathbf { \bar { \mathcal { K } } } ) \gets \mathrm { P r e f i l l } _ { R } ( F _ { \theta } , x _ { 1 : n } )$   
2 $\dot { x } \sim \dot { \Pi } ( \ell _ { 0 } )$   
3 $Y \gets [ x ]$   
4 if $\bar { \cal { S } } ( \dot { Y } )$ then   
5 return Y   
6 b ← Spawn $\left( x _ { 1 : n } \| Y \right)$   
7 $\mathcal { A }  \{ b _ { 0 } \}$   
8 while ${ \check { \mathbf { \alpha } } } { \mathcal { S } } ( Y )$ do   
9 W ← SelectWork<sub>π</sub> $\cdot \left( \mathcal { A } , b _ { 0 } \right)$   
10 $\mathcal { N }  \emptyset$   
11 AdvanceBatch $( \mathcal { W } ; F _ { \theta } , \mathcal { K } , \Delta )$   
12 foreach $b \in \mathcal W$ do   
13 if $r _ { b } = d _ { 1 }$ then   
14 $q _ { 1 } \gets \Pi ( { \mathcal { C } } ( { \mathbf { h } } _ { b } ) )$   
15 $u _ { 1 } \sim q _ { 1 }$   
16 $Q _ { b } \gets \mathsf { \bar { [ ( } } u _ { 1 } , q _ { 1 } ) \mathsf { ] }$   
17 ${ \mathcal { N } } \gets { \dot { N } } \cup \{ b \}$   
18 else if $r _ { b } = d _ { 2 }$ then   
19 $\widetilde { q } _ { 2 } \gets \Pi ( { \mathcal { C } } ( { \mathbf { h } } _ { b } ) )$   
20 $( u _ { 1 } , q _ { 1 } )  Q _ { b } [ 1 ]$   
21 if $\widetilde { q } _ { 2 } ( u _ { 1 } ) < q _ { 1 } ( u _ { 1 } )$ then   
22 $q _ { 2 }  \widetilde { q } _ { 2 } \ominus q _ { 1 }$   
23 u<sub>2</sub> ∼ q<sub>2</sub>   
24 $Q _ { b }  Q _ { b } \vert \vert [ ( u _ { 2 } , q _ { 2 } ) ]$   
25 else if $r _ { b } = R$ then   
26 v ← NativeVerify $\left( b , \Pi ( { \mathcal { C } } ( \mathbf { h } _ { b } ) ) \right)$   
27 if ∃b $\in \mathcal { W } , \ r _ { b } \in \{ d _ { 1 } , d _ { 2 } , R \}$ then   
28 A ← ReclaimDrafts $\cdot ( \dot { \boldsymbol { A } } , \boldsymbol { b } _ { 0 } )$   
29 FillNativeChildren(A; B )   
30 while $r _ { b _ { 0 } } = R$ do   
31 $x ^ { \star }  v _ { b _ { 0 } }$   
32 c ← FindChild(b , x<sup>⋆</sup>)   
33 $Y \gets Y \| x ^ { \star }$   
34 if $S ( Y )$ then   
35 return Y   
36 $X _ { \mathrm { n e x t } }  X ^ { ( b _ { 0 } ) } \| x ^ { \star }$   
37 A ← Reroot(A, c; K)   
38 if c = ∅ then   
39 $c \gets \operatorname { S p a w n } ( X _ { \mathrm { n e x t } } )$   
40 $A  \{ c \}$   
41 $b _ { 0 }  c$   
42 FillNativeChildren(A; B<sub>ret</sub>)   
43 (b, a) ← SelectAnchor<sub>π</sub> $( \mathcal { N } , \mathcal { A } , b _ { 0 } ; w )$   
44 if a ̸= ∅ then   
45 ℓ ← PrecallLength<sub>π</sub> $( b , a , { \mathcal { A } } ; K )$   
46 if ℓ > 0 then   
47 H ← ActualWindow $( b , d _ { 1 } , w )$   
48 V ← DraftVocab(V<sub>base</sub>, x<sub>1:n</sub>, Y)   
49 ℓ ← DraftHorizon (ℓ)   
50 z, b c <sup>←</sup> Draft(D<sup>⋆</sup>, H, Token $\iota ( a ) , \ell _ { \mathrm { c a l l } } ; \mathcal { V } )$   
51 k ← AdmitLength $( \widehat { z } , \mathbf { c } , b , a , \mathcal { A } ; \ell )$   
52 <sup>A</sup> <sup>←</sup> GraftChain(<sup>A</sup>, a, zb1:k)   
53 return Y

Prompt: How many eggs are in two dozen eggs? Output: Two dozen eggs contain twenty-four eggs.  
![](images/06b46c0dde8e06fb41a7455b9918880db1ed0aa62e6c567e0eaf70fa7e82a99d.jpg)  
Figure 13: Complete LoopSpec and LoopAhead decoding traces for the example in Figure 2, using Ouro-1.4B under greedy decoding. Rows denote output positions and columns denote global loop steps. LoopAhead produces the eight tokens after prefill in 13 steps, compared with 15 for LoopSpec. Two of the nine LAD calls are shown in detail. Colors and symbols follow Figure 2.

<table><tr><td colspan="3">Case 1: Drafting ahead in code generation</td></tr><tr><td>Question</td><td colspan="3">Write a function to find tuples which have all elements divisible by k from the given list of tuples.</td></tr><tr><td>Output excerpt</td><td colspan="3">def find_tuples(tuples, k): return [t for t in tuples if all(i % k == 0 for i in t)]</td></tr><tr><td>LAD drafts</td><td colspan="3">Context + anchor LAD proposals def find_tuples(tuples, k):√√ return√</td></tr><tr><td rowspan="3">Execution</td><td>== 0 for</td><td>i√ in√ lst × lst →t</td></tr><tr><td>Method Loops Decode time (ms)</td><td></td></tr><tr><td>LoopSpec 40 269.7 LoopAhead 23 185.8</td><td></td></tr><tr><td>Observation</td><td colspan="3">42.5% fewer loops and 45.2% higher throughput. LAD supplies matching continuations ahead of</td></tr></table>

Table 9: A greedy code-generation case from MBPP (sample 75). Both methods produce the same output. Blue marks the native anchor. ✓ and × mark draft-token matches and mismatches against the full-depth continuation. ←- denotes a newline with indentation, and leading token spaces are omitted.

<table><tr><td colspan="6">Case 2: Less work per loop</td></tr><tr><td>Question</td><td colspan="5">Jane has saved $4 of her allowance every week for the past 8 weeks. If she continues to save the same amount each week, how many more weeks will it take for her to have saved a total of $60?</td></tr><tr><td>Output excerpt</td><td colspan="5">Jane has saved $4 each week for 8 weeks. So she has saved 8 × 4 = $32. [.. .] So she needs to save $60 – $32 = $28 more. To save $28, she needs to save for 28 ÷ 4 = 7 more weeks. The answer is 7.</td></tr><tr><td>Execution</td><td>Method Loops</td><td>Candidate</td><td>Candidates per loop</td><td>Decode time</td></tr><tr><td></td><td>LoopSpec</td><td>advances</td><td>19.3</td><td>(ms) 1,008.1</td></tr><tr><td></td><td></td><td>4,786</td><td>12.3</td><td>775.3</td></tr><tr><td></td><td>LoopAhead 252</td><td>3,100</td><td></td><td></td></tr><tr><td>Observation</td><td colspan="5">35.2% fewer candidate advances and 30.0% higher throughput. LoopAhead executes fewer candidates</td></tr></table>

Table 10: A greedy math-generation case from GSM8K (sample 1134). Both methods produce the same output. Candidate advances count the total candidates advanced across all loops. Candidates per loop gives their average.

<table><tr><td colspan="4">Case 3: Recursive drafting</td></tr><tr><td>Question</td><td colspan="4">Evaluatenot (True and True) or False.</td></tr><tr><td>Output excerpt</td><td colspan="4">Remember that (i) expressions inside brackets are always evaluated first [... ] Plugging in A and B, we get: Z = A or B = False or False = False. So the answer is False.</td></tr><tr><td>Recursive LAD</td><td>Call Context ending</td><td>Native anchor</td><td>LAD proposals</td></tr><tr><td></td><td>1 Remember 2</td><td>that</td><td>(|i|)√</td></tr><tr><td></td><td>(i)</td><td>expressions</td><td>inside |brackets|are√</td></tr><tr><td></td><td colspan="4">Call 2 uses the R1 state of ) drafted in Call 1, before it is fully verified.</td></tr><tr><td>Execution</td><td>Method</td><td>Loops</td><td>Decode time (ms)</td></tr><tr><td></td><td>LoopSpec</td><td>146</td><td>1,070.4</td></tr><tr><td></td><td>LoopAhead</td><td>55</td><td>501.3</td></tr><tr><td>Observation</td><td colspan="5">62.3% fewer loops and 113.5% higher throughput. Earlier drafts become new sources of lookahead before full-depth verification.</td></tr></table>

Table 11: A recursive-drafting case from BBH Boolean Expressions (sample 148), using greedy decoding. Both methods produce the same output. Vertical bars separate draft tokens. ✓ indicates that all three match the full-depth continuation.