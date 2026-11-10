# Spectral Rank and Step-Direction Dynamics in Language-Model Post-Training

Khurram Khalil Department of Electrical Engineering & Computer Science University of Missouri-Columbia, MO, USA

Khaza Anuarul Hoque

Department of Electrical Engineering & Computer Science University of Missouri-Columbia, MO, USA

khurram.khalil@missouri.edu

hoquek@missouri.edu

Reviewed on OpenReview: https: //openreview. net/forum?id=UGIcayxHMI

## Abstract

Understanding how language models change during post-training requires characterizing both the structure of cumulative weight updates and the directions of successive training displacements. Existing studies report seemingly contrasting spectral properties, including nearly full-rank updates and low stable-rank cumulative changes. We show that these observations are mathematically compatible and investigate how weight-update directions evolve during training. For singular values $\sigma _ { i } \propto i ^ { - \alpha } , 0 < \alpha \leq 1 / 2$ , we prove that stable rank becomes negligible relative to fixed-energy rank as dimension grows. Full singular value decompositions of selected OLMo-2-7B updates verify this separation. Three attention updates from proximal policy optimization (PPO) have stable ranks 9.6–18.3 but require 532–1,222 components for 80% of their energy. We then measure consecutive checkpoint-displacement cosines using full-model inner products. Across nine public stages of reinforcement learning from verifiable rewards (RLVR), all 131 adjacent pairs have negative cosine on their native checkpoint grids, including PPO and Group Relative Policy Optimization (GRPO) releases. Supervised comparisons across four model families reveal configuration-dependent alignment, with negative late-training cosines in Mistral and FP32 Llama runs. A controlled OLMo-2-1B study with three paired seeds and a 2,048-token response cap compares grouprelative advantages with a learned critic and generalized advantage estimation. At matched 60-rollout windows, their late-training mean cosines are +0.259 and +0.012, with a paired diference of −0.247 and a seed-level 95% interval of [−0.358, −0.137]. Together, these results separate spectral concentration, energy coverage, and trajectory direction, and show that the advantage system afects directional alignment under matched training conditions.

## 1 Introduction

Language-model post-training changes a model’s behavior by changing its weights. Supervised fine-tuning (sft) learns from demonstration responses, while reinforcement learning from verifiable rewards (rlvr) learns from sampled responses scored by an automatic checker. Weight-space measurements describe how these procedures alter a pretrained model and inform methods that operate on weight diferences, including task arithmetic and low-rank adaptation (Ilharco et al., 2023; Hu et al., 2022). Interpreting those measurements requires separating the mathematical object being measured from the functional applied to it.

That distinction matters when comparing recent accounts of post-training geometry. Mukherjee et al. (2025) report sparse, nearly full-algebraic-rank rlvr updates. Zhu et al. (2025) analyze update localization, numerical precision, and alignment with principal weight subspaces. Shen et al. (2026) compare cumulative updates in on-policy distillation, rlvr, and sft, reporting low stable rank for distillation and the rlvr endpoint. These studies characterize diferent aspects of adaptation. Yet their rank descriptions can appear incompatible if algebraic rank, stable rank, and energy coverage are treated as interchangeable. A comparison on the same update matrix is needed to determine what each description actually implies. We address this question with an explicit spectral construction and full-spectrum measurements of selected updates.

Even a complete endpoint spectrum leaves a second question unanswered. It does not determine whether consecutive training displacements reinforce or oppose one another. Analyses of full weight matrices (He & Cao, 2025) and trajectory-level linear trends (Wang et al., 2026) provide complementary information, but neither an endpoint change nor a fitted trend determines the inner products between consecutive finite displacements. We therefore measure direction at specified checkpoint widths and separations. These measurements describe local alignment on the saved checkpoint grid.

Diferences in trajectory alignment raise a third question. How does the advantage estimator afect the direction of weight updates under a shared training schedule? We investigate this question through a controlled comparison of group-relative advantages and a learned critic with generalized advantage estimation. Holding the actor schedule and sampled prompt stream fixed connects the observed geometry to a specific change in the training procedure.

These investigations provide complementary views of post-training geometry. Spectral analysis characterizes how the cumulative weight change is distributed across singular directions, while trajectory analysis examines how successive weight changes interact during training. The controlled comparison then investigates whether changing the advantage system alters these directional relationships. Together, these measurements distinguish the structure of the final update from the geometry of the path taken to reach it.

Our contributions follow these three questions. First, we prove that low stable rank and large fixed-energy rank coexist for a family of power-law singular spectra, and verify the separation with full decompositions of OLMo-2-7B cumulative updates. Second, we document negative consecutive-displacement cosines across nine public rlvr stage trajectories, including ppo and grpo releases, quantify their temporal extent and opposing components, and test supervised alignment across four model families and two Llama precision settings. Third, we identify an advantage-system efect in six matched OLMo-2-1B runs. The critic–GAE arm reduces late-training alignment by 0.247 on average relative to the group-relative arm, with a paired seed-level 95% interval of [−0.358, −0.137] for the signed diference. Together, these contributions reconcile spectral descriptions of post-training and establish trajectory direction as a separately measurable aspect of adaptation.

## 2 Background and Geometric Quantities

## 2.1 How post-training changes the weights

Let $\pi _ { \theta } ( y \mid x )$ denote the probability of response y to prompt x under parameters θ. For a batch B of tokenized demonstrations z, sft minimizes a masked next-token negative log likelihood

$$
L _ { \mathrm { S F T } } ( \theta ) = - \frac { \sum _ { z \in B } \sum _ { j = 2 } ^ { | z | } m _ { j } ( z ) \log \pi _ { \theta } ( z _ { j } \mid z _ { < j } ) } { \sum _ { z \in B } \sum _ { j = 2 } ^ { | z | } m _ { j } ( z ) } .\tag{1}
$$

Here $m _ { j } ( z )$ selects supervised positions and excludes padding. Response-only supervision selects response positions; our sft experiments supervise all non-padding conversation tokens with a preceding token, including prompt and response text. The target tokens are fixed demonstrations. The loss supplies gradients, and an optimizer such as Adam converts those gradients into parameter changes. A saved weight displacement therefore aggregates the optimizer-preconditioned updates between checkpoints.

In rlvr, the current policy samples responses and a checker supplies a reward $R ( x , y )$ , for example whether a final mathematical answer is correct. A common objective balances expected reward against deviation from a frozen reference policy $\pi _ { \mathrm { r e f } }$ . Practical implementations optimize sample-based surrogate losses. Proximal policy optimization (ppo) (Schulman et al., 2017) uses a token probability ratio $r _ { j } ( \theta ) = \pi _ { \theta } ( y _ { j } \mid x , y _ { < j } ) / \pi _ { \mathrm { o l d } } ( y _ { j } \mid x , y _ { < j } )$ and the clipped objective

$$
L _ { \mathrm { a c t o r } } ( \theta ) = - \mathbb { E } _ { j } \operatorname* { m i n } \Big ( r _ { j } ( \theta ) \widehat { A } _ { j } , \mathrm { c l i p } ( r _ { j } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) \widehat { A } _ { j } \Big ) + \beta \widehat { D } _ { \mathrm { K L } } .\tag{2}
$$

Here $\widehat { A } _ { j }$ estimates the advantage of a sampled token or response, ϵ controls ratio clipping, and $\beta$ weights the reference-policy penalty. Multiple optimization passes may reuse the same sampled responses.

The advantage estimator distinguishes important implementations of this objective. A learned value function estimates expected future reward, and generalized advantage estimation (GAE) combines temporal-diference residuals (Schulman et al., 2016). Group Relative Policy Optimization (grpo) instead compares rewards among responses to the same prompt without a learned value function (Shao et al., 2024). Our controlled arms share Equation 2 and difer in this advantage-system package. We name the two arms group-relative and $_ { c r i t i c - G A E }$ after their advantage estimators. Appendix B specifies their estimators and training configuration.

The reference for a weight displacement must follow this training sequence. The public OLMo-2-7B instruction model follows sft, direct preference optimization (dpo), and then rlvr (OLMo Team et al., 2025). dpo learns from preferred and dispreferred responses (Rafailov et al., 2023). Consequently, the pure rlvr displacement is $W _ { \mathrm { R L V R } } - W _ { \mathrm { D P O } }$ . Subtracting the sft checkpoint instead adds the intervening dpo change. All PPO endpoint measurements use the immediate dpo initialization.

## 2.2 Endpoint functionals

For a nonzero matrix displacement $\Delta = W _ { T } - W _ { 0 } \in \mathbb { R } ^ { m \times n }$ , let $\sigma _ { 1 } \geq \cdot \cdot \cdot \geq \sigma _ { d } \geq 0 , d = \operatorname* { m i n } ( m , n )$ , be its singular values. Algebraic rank counts nonzero singular values. Stable rank measures total squared singular energy relative to the largest squared singular value

$$
r _ { \mathrm { s t a b l e } } ( \Delta ) = \frac { \| \Delta \| _ { F } ^ { 2 } } { \| \Delta \| _ { 2 } ^ { 2 } } = \frac { \sum _ { i } \sigma _ { i } ^ { 2 } } { \sigma _ { 1 } ^ { 2 } } , \qquad E _ { k } ( \Delta ) = \frac { \sum _ { i = 1 } ^ { k } \sigma _ { i } ^ { 2 } } { \sum _ { i = 1 } ^ { d } \sigma _ { i } ^ { 2 } } , \qquad k _ { \tau } ( \Delta ) = \operatorname* { m i n } \{ k \mid E _ { k } ( \Delta ) \geq \tau \} .\tag{3}
$$

We call $k _ { \tau }$ the energy-threshold rank. This avoids ambiguity with entropy-based efective rank (Roy & Vetterli, 2007). Stable rank is a continuous measure of spectral concentration, while $k _ { 0 . 8 }$ explicitly counts the components needed for 80% of the squared Frobenius norm. These definitions imply $1 \leq r _ { \mathrm { s t a b l e } } \leq \mathrm { r a n k } ( \Delta )$ and $k _ { \tau } \geq \tau r _ { \mathrm { s t a b l e } } ,$ since $\begin{array} { r } { \sum _ { i = 1 } ^ { k } \sigma _ { i } ^ { 2 } \le k \sigma _ { 1 } ^ { 2 } } \end{array}$ . They supply no converse bound of comparable size, as the next proposition shows.

Intuitively, smaller stable rank indicates stronger concentration around the largest singular value, whereas energy-threshold rank measures how many singular components are needed to capture a specified fraction of the total energy. For example, an update may contain one dominant singular value and hundreds of smaller but collectively significant singular values. Such an update can have low stable rank while requiring hundreds of components to capture a specified energy threshold (e.g., 80%, as used in our experiments).

Proposition 1 (Separation of stable rank and fixed-energy rank). For $\sigma _ { i } = i ^ { - \alpha } , i = 1 , . . . , n$ , and fixed $\tau \in ( 0 , 1 ) , r _ { \mathrm { s t a b l e } } / k _ { \tau } \to 0$ as $n \to \infty$ whenever $0 < \alpha \le 1 / 2$ . For $0 < \alpha < 1 / 2$ , with $b = 1 - 2 \alpha$

$$
r _ { \mathrm { s t a b l e } } \leq 1 + \frac { n ^ { b } - 1 } { b } , \qquad k _ { \tau } \geq \tau ^ { 1 / b } ( n ^ { b } - 1 ) ^ { 1 / b } = \Theta ( n ) .\tag{4}
$$

For α = 1/2, r<sub>stable</sub> ≤ 1 + log n and $k _ { \tau } \geq n ^ { \tau } / e$

Appendix A proves the proposition by integral comparison. The construction shows how a leading singular value and a broad energetic tail produce the two rank descriptions simultaneously.

These spectral quantities do not specify how many entries of an update are zero. We therefore consider support sparsity separately. Exact zero-entry fraction $1 - \| \Delta \| _ { 0 } / ( m n )$ and thresholded sparsity depend on the coordinate basis and numerical precision. A diagonal matrix with all diagonal entries nonzero has fixed support and an arbitrary positive singular spectrum. It can have stable rank close to one or equal to n without changing that support. Orthogonal changes of basis preserve its spectrum while generally making it dense. Thus support sparsity does not determine either spectral rank.

## 2.3 Finite-displacement direction

Endpoint functionals do not specify the path to the endpoint. Let ${ \theta _ { t } } ~ \in { \mathbb { R } ^ { P } }$ be the vector obtained by flattening and concatenating the model parameters at checkpoint t, with tied parameters counted once; P is the number of parameter coordinates measured. On a grid with width h, define the full-parameter displacement $D _ { j } ^ { ( h ) } = \theta _ { j h } - \theta _ { ( j - 1 ) h }$ . Thus $D _ { j } ^ { ( h ) } \in \mathbb { R } ^ { P }$ . We use the Euclidean inner product $\langle x , y \rangle = x ^ { \top }$ y and norm $\| x \| _ { 2 } = { \sqrt { x ^ { \top } x } }$ . For nonzero displacements, define their directional correlation at separation ℓ as

$$
\rho _ { j , \ell } ^ { ( h ) } = \frac { \langle D _ { j } ^ { ( h ) } , D _ { j - \ell } ^ { ( h ) } \rangle } { \| D _ { j } ^ { ( h ) } \| _ { 2 } \| D _ { j - \ell } ^ { ( h ) } \| _ { 2 } } .\tag{5}
$$

A positive cosine indicates that the compared displacements point in broadly similar directions, whereas a negative cosine indicates that they contain opposing directional components. Importantly, negative cosine does not imply that training is reversing or that model performance is deteriorating. Instead, it describes only the geometric relationship between the compared weight changes. For consecutive displacements, $\ell = 1$ Width h and separation ℓ are distinct. Increasing h aggregates more training updates, whereas increasing ℓ compares displacements further apart on a fixed grid. We refer to a negative consecutive cosine as consecutive anti-alignment.

The amount projected against the previous displacement also depends on norms. Writing $D _ { j } = a _ { j } D _ { j - 1 } + D _ { j } ^ { \scriptscriptstyle 1 }$ with $\langle D _ { j } ^ { \perp } , D _ { j - 1 } \rangle = 0$ , gives

$$
a _ { j } = \frac { \left. D _ { j } , D _ { j - 1 } \right. } { \| D _ { j - 1 } \| _ { 2 } ^ { 2 } } = \rho _ { j , 1 } ^ { ( h ) } \frac { \| D _ { j } \| _ { 2 } } { \| D _ { j - 1 } \| _ { 2 } } .\tag{6}
$$

For example, cosine −0.25 gives $a _ { j } = - 0 . 0 2 5$ if the current norm is one tenth of the previous norm, and $a _ { j } = - 0 . 5$ if it is twice as large.

## 3 Data and Measurement Protocol

## 3.1 Public trajectories and supervised comparisons

Our public-release analysis covers nine rlvr stages in the OLMo-2, OLMoE, and Olmo-3 releases. Table 3 lists each stage and its native checkpoint width. The OLMo-2 stages include 7B and 13B ppo releases and 1B and 32B grpo releases. We report successive stages within each model lineage separately and retain their shared lineage in the checkpoint inventory.

For sft, we analyze author-run trajectories from four model families on a uniform 500-optimizer-step grid through step 5,000. The comparison includes OLMo-2-7B and Llama-3.2-3B at seeds 42, 43, and 44, Mistral-$7 \mathrm { B - v 0 . 3 }$ at seeds 42 and 43, and Qwen2.5-3B at seed 42. Each trajectory contains ten displacements and nine adjacent pairs. All runs use the Tülu-3 OLMo-2 data mixture (Lambert et al., 2025; OLMo Team et al., 2025), the same plain-text role formatting and all-token objective in Equation 1, efective batch size 64, sequence cap 4,096, and learning rate $1 0 ^ { - 5 }$ with 3% warmup and linear decay over the two-epoch horizon. Training stops at step 5,000 while preserving that schedule. The OLMo seed-42 run continues to step 29,356 and supplies the sft endpoint for spectral analysis.

The cross-family runs store trainable parameters and Adam states in BF16. Two additional Llama runs at seeds 42 and 43 use FP32 parameters and Adam states with BF16 autocast, retaining the data mixture, efective batch size, learning-rate schedule, and measurement grid. Appendix G reports each completed run, the precision comparison, and the full-model measurement checks. These comparisons match example and optimizer-step budgets; tokenization and realized token counts vary across families.

The spectral validation uses fourteen matrices selected by layer position and shape before inspecting approximation errors. For each of the 7B PPO and author-run 7B sft endpoints, we select query projections at layers 0, 16, and 31, up projections at layers 0 and 31, the down projection at layer 16, and the embedding matrix. Table 1 states their dimensions and the complete 7B matrix inventory. The endpoints difer in objective, initialization, data exposure, and horizon. We examine each matrix’s spectral functionals and compare their concentration and energy coverage across these endpoints.

Table 1: Matrix dimensions in the OLMo-2-7B endpoint analysis. Stored orientation is rows by columns. The 226 matrices exclude one-dimensional normalization weights, which remain included in full-model directional measurements.
<table><tr><td>Matrix class</td><td>Rows</td><td>Columns</td><td>Count</td></tr><tr><td>Attention query, key, value, output</td><td>4,096</td><td>4,096</td><td>128</td></tr><tr><td>MLP up and gate</td><td>11,008</td><td>4,096</td><td>64</td></tr><tr><td>MLP down</td><td>4,096</td><td>11,008</td><td>32</td></tr><tr><td>Token embedding and language-model head</td><td>100,352</td><td>4,096</td><td>2</td></tr></table>

## 3.2 Controlled advantage-system comparison

The controlled study starts both arms from OLMo-2-0425-1B-DPO and trains on RLVR-GSM for 360 rollout batches. Seeds 42, 43, and 44 each define a paired prompt stream. Both arms use eight prompts and eight sampled responses per prompt, four update passes, eight-response minibatches, learning rate $4 \times 1 0 ^ { - 7 }$ with linear decay, ratio clipping 0.2, KL coeficient 0.05, and prompt and response caps of 2,048 tokens. This response cap follows the published OLMo-2 PPO recipe. The rollout batch is smaller than that published recipe, and the experiment uses a shared custom trainer. The shared trainer holds the actor schedule fixed while changing the advantage system.

Each rollout batch generates 64 responses and performs 32 actor optimizer updates. A displacement width of 60 rollouts therefore represents 3,840 responses and 1,920 actor optimizer updates. We match these quantities between arms and report realized generated-token counts in Appendix C. Generated-token totals vary with the evolving policies. Public trajectories retain the release-specific step units listed in Table 3.

The prespecified primary outcome is each seed’s mean consecutive cosine at width 60 for endpoints after rollout 180, comprising endpoints 240, 300, and 360. We report all three seed values and the paired criticminus-group diference. A Student interval uses the three seed-level diferences with two degrees of freedom. Widths 10, 20, and 40 and all-window summaries are secondary descriptive checks. Appendix B records the full protocol.

## 3.3 Exact directional measurements and spectral validation

Directional measurements accumulate full-parameter dot products and squared norms in FP64 over coordinate chunks, including normalization weights. For eight retained trajectories, we independently compute the displacement Gram matrix $G _ { i j } = \langle D _ { i } , D _ { j } \rangle$ ⟩, which supplies both higher-separation correlations and normaware projections. Online and saved-checkpoint cosines for the six matched runs agree within $1 . 4 \times 1 0 ^ { - 1 2 }$ in absolute value.

For the fourteen selected spectral cases, we compute a full FP64 singular decomposition. Reduced QR followed by full SVD preserves the complete singular spectrum of high-aspect-ratio matrices. Leading singulartriplet residuals are below $1 0 ^ { - 1 0 }$ . We compare these results with the original randomized SVD configuration, ten oversamples and four power iterations (Halko et al., 2011), using seeds 42, 43, and 44, and a stronger setting with forty oversamples and eight iterations at seed 42. Rank-256 and rank-512 approximations are computed independently. We separate singular-value error from error introduced by the original FP32 norm and denominator regularizer.

Energy-threshold reporting requires suficient spectral accuracy. If an approximate top-K energy remains below τ, approximation error alone may explain the shortfall. Without a certified error bound, we label the threshold unresolved at the computed budget, not $k _ { \tau } > K$ . All energy-threshold ranks used as quantitative evidence below come from full decompositions. Appendix F documents the corrected initialization and coordinate-sketch errors in the earlier analysis.

Table 2: Full-SVD endpoint measurements for the shape-stratified 7B sample. PPO subtracts the immediate DPO checkpoint. SFT subtracts the base checkpoint. Each row refers to the same named matrix in the two endpoints. $k _ { 0 . 8 }$ is computed from the complete spectrum. $E _ { 5 1 2 }$ abbreviates $E _ { k } ( \Delta )$ from Equation 3 at $k = 5 1 2$ , the fraction of squared singular energy in the leading 512 components.
<table><tr><td rowspan="2">Matrix</td><td colspan="3">PPO</td><td colspan="3">SFT</td></tr><tr><td> $r _ { \mathrm { s t a b l e } }$ </td><td> $E _ { 5 1 2 }$ </td><td> $k _ { 0 . 8 }$ </td><td> $r _ { \mathrm { s t a b l e } }$ </td><td> $E _ { 5 1 2 }$ </td><td> $k _ { 0 . 8 }$ </td></tr><tr><td>Query 0</td><td>9.58</td><td>0.790</td><td>532</td><td>1.10</td><td>0.987</td><td>1</td></tr><tr><td>MLP up 0</td><td>115.60</td><td>0.368</td><td>2,077</td><td>356.95</td><td>0.289</td><td>2,221</td></tr><tr><td>Query 16</td><td>18.14</td><td>0.536</td><td>1,222</td><td>1.48</td><td>0.848</td><td>255</td></tr><tr><td>MLP down 16</td><td>58.78</td><td>0.345</td><td>2,120</td><td>792.32</td><td>0.277</td><td>2,250</td></tr><tr><td>Query 31</td><td>18.27</td><td>0.564</td><td>1,149</td><td>1.99</td><td>0.779</td><td>599</td></tr><tr><td>MLP up 31</td><td>45.72</td><td>0.342</td><td>2,114</td><td>200.56</td><td>0.291</td><td>2,210</td></tr><tr><td>Embedding</td><td>1026.35</td><td>0.200</td><td>2,831</td><td>916.65</td><td>0.175</td><td>2,982</td></tr></table>

![](images/c2a9e193e3c5f389db181473c9691d0788f175fc036fa5cd3dbc2a75596d904b.jpg)  
Figure 1: Cumulative singular energy from full FP64 decompositions. The PPO query update at layer 0 has stable rank 9.58 but requires 532 components for 80% energy. The layer-16 MLP down update requires 2,120. The SFT layer-16 query curve also illustrates a threshold near the rank-256 budget.

## 4 Results

## 4.1 Low stable rank coexists with large energy rank

Full decompositions verify the spectral separation on the same update matrices. Table 2 reports all fourteen cases, and Figure 1 shows representative cumulative energy curves. The three PPO query updates have stable ranks 9.58, 18.14, and 18.27, while their 80%-energy ranks are 532, 1,222, and 1,149. These matrices combine strong leading-value concentration with an energetic tail spanning hundreds of components.

The same distinction applies to the MLP updates. The selected PPO down projection at layer 16 has stable rank 58.78 but $k _ { 0 . 8 } = 2 , 1 2 0$ . Its sft counterpart has stable rank 792.32 and $k _ { 0 . 8 } = 2 , 2 5 0$ . The stable ranks difer by more than an order of magnitude while their energy ranks are close. Conversely, the sft query update at layer 0 has $k _ { 0 . 8 } = 1$ . The two functionals thus reveal diferent aspects of variation across matrices, with stable rank distinguishing concentration even when energy ranks are similar.

These observations reconcile the logical content of the prior rank descriptions. Near-full algebraic rank permits a dominant leading singular value, and a dominant leading value permits a long energetic tail. Proposition 1 supplies an explicit family with this behavior. Thus, low stable rank alone does not generally imply that a small number of singular components captures most of the update energy.

Table 3: Consecutive full-model cosines on native public checkpoint grids. Width is in release step labels. Every listed pair is negative. Counts refer to adjacent pairs within each stage. Successive stages retain their shared model lineage.
<table><tr><td>Public stage</td><td>Released method</td><td>Width</td><td>Pairs</td><td>Mean cosine</td></tr><tr><td>OLMo-2 7B</td><td>PPO</td><td>60</td><td>5</td><td>-0.2202</td></tr><tr><td>OLMo-2 13B RLVR1</td><td>PPO</td><td>60</td><td>12</td><td>-0.2446</td></tr><tr><td>OLMo-2 13B RLVR2</td><td>PPO</td><td>60</td><td>12</td><td>-0.2541</td></tr><tr><td>OLMo-2 13B final</td><td>PPO</td><td>60</td><td>12</td><td>-0.2582</td></tr><tr><td>OLMo-2 1B RLVR1</td><td>GRPO</td><td>200</td><td>12</td><td>-0.2293</td></tr><tr><td>OLMo-2 1B final</td><td>GRPO</td><td>200</td><td>12</td><td>-0.2327</td></tr><tr><td>OLMo-2 32B</td><td>GRPO</td><td>20</td><td>16</td><td>-0.1929</td></tr><tr><td>OLMoE 1B active / 7B</td><td>RLVR</td><td>40</td><td>43</td><td>-0.2633</td></tr><tr><td>Olmo-3 7B</td><td>OlmoRL</td><td>50</td><td>7</td><td>-0.2153</td></tr></table>

Table 4: Directional correlations at fixed displacement width. Entries are means with pair counts in parentheses. Separation changes within each row; width does not. Each trajectory retains its native width and training exposure. All six controlled runs appear in Appendix D.
<table><tr><td>Trajectory</td><td>Width and unit</td><td>l = 1</td><td>l = 2</td></tr><tr><td>Public 7B PPO</td><td>60 release steps</td><td>−0.2202 (5)</td><td>−0.0244 (4)</td></tr><tr><td>7B SFT seed 42</td><td>500 trainer steps</td><td>+0.2275 (9)</td><td>-0.0087 (3) +0.1740 (8) +0.1411 (7)</td></tr></table>

## 4.2 Production anti-alignment extends beyond PPO

Every native-grid adjacent pair in the nine public stage trajectories has negative full-model cosine. Table 3 reports 131 such pairs, with stage means between −0.263 and −0.193. Consecutive anti-alignment is consistent across this collection of PPO and GRPO releases, model sizes, and checkpoint widths.

The supervised comparison reveals configuration-dependent alignment across four model families. Table 14 reports all seeds on the common 500-step grid. The BF16 OLMo, Llama, and Qwen trajectories have positive cosines throughout, with seed means between +0.2226 and +0.2381. Both Mistral seeds instead have seven negative pairs out of nine and means of −0.0100 and −0.0099. The Mistral runs thus provide a same-size-class comparison in which sft also exhibits consecutive anti-alignment.

The Llama precision comparison further shows that alignment depends on the numerical training configuration. With FP32 parameters and Adam states, the seed-42 and seed-43 means fall to +0.0206 and +0.0209, and the final five pairs are negative in both runs. Table 16 reports the paired series. Positive alignment therefore replicates across several BF16 settings, while the Mistral and FP32 Llama results establish that cosine sign varies within sft itself.

Higher-separation correlations quantify how far the directional relationship extends along the trajectory.   
Table 4 reports these correlations for the public 7B PPO trajectory at a fixed width of 60 release steps.   
Mean cosine changes from −0.2202 at separation one to −0.0244 and −0.0087 at separations two and three.   
The negative correlation is strongest between adjacent displacements and approaches zero at the larger separations measured here.

Cosine describes directional alignment but does not measure the magnitude of the component opposing the previous displacement. The projection coeficient accounts for the ratio of displacement norms. In the same public 7B run, Equation 6 yields coeficients from −0.2204 to −0.1741. Each displacement has a component opposite the preceding displacement with magnitude 17–22% of the preceding norm. These observations demonstrate that consecutive weight displacements can exhibit systematic directional opposition, even though their cumulative efect is not characterized by the individual displacement cosines.

Table 5: Primary matched outcome. Each seed value averages the width-60 cosines ending at rollouts 240, 300, and 360. The interval uses the three paired seed means. Diference means critic–GAE minus grouprelative.
<table><tr><td>Seed</td><td>Group-relative</td><td>Critic-GAE</td><td>Difference</td></tr><tr><td>42</td><td>+0.3178</td><td>+0.0193</td><td>-0.2985</td></tr><tr><td>43</td><td>+0.2363</td><td>+0.0094</td><td>-0.2269</td></tr><tr><td>44</td><td>+0.2229</td><td>+0.0060</td><td>-0.2168</td></tr><tr><td>Mean</td><td>+0.2590</td><td>+0.0116</td><td>-0.2474</td></tr><tr><td>95% interval</td><td>[+0.1314, +0.3866]</td><td>[-0.0056, +0.0287]</td><td>[-0.3580, -0.1368]</td></tr></table>

## 4.3 A controlled efect on alignment magnitude

To test whether the advantage system changes alignment under a shared training schedule, we turn from the public trajectories to the matched study. Table 5 reports all three paired seeds for the primary late-training outcome. The group-relative mean is +0.259 and the critic–GAE mean is +0.012. The paired critic-minusgroup diference is −0.247, with a seed-level 95% Student interval of [−0.358, −0.137]. Every paired seed has lower alignment in the critic arm. The paired contrast shows that the advantage system changes directiona alignment consistently across the three seeds.

The critic mean is close to zero, and its own seed-level interval, [−0.0056, +0.0287], includes zero. Replacing group-relative advantages with the critic–GAE package therefore lowers mean alignment toward zero. Neither local arm reproduces the consistently negative production pattern.

The efect persists in the secondary matched-width summaries. At widths 10, 20, and 40, all paired latetraining diferences are negative (Appendix C). Their magnitudes change with width, illustrating why equal aggregation units matter. The group-relative seed mean increases from 0.071 at width 10 to 0.259 at width 60. Matching aggregation width makes the measured advantage-system efect interpretable at a shared response and optimizer-update budget.

Training-health checks accompany the geometry measurements. The runs record realized response lengths, EOS completion, and finite updates. Training cap-hit fractions range from 0.213% to 1.189%. The fixed 32-example held-out diagnostic shows no consistent improvement over the initial policy. Appendix B reports the per-run counts. The geometry measurements show sensitivity to the advantage-system configuration, while the held-out diagnostic shows no consistent performance improvement.

## 4.4 Numerical accuracy depends on the reported functional

The SVD validation supports accurate leading-value estimates in the selected sample but exposes material errors in threshold and angle interpretations. With an FP64 Frobenius denominator, the baseline randomized approximation’s maximum relative stable-rank error is $8 . 7 8 \times 1 0 ^ { - 5 }$ , or 0.0088%. Including the original FP32 norm and denominator regularizer raises the maximum original-pipeline error to 2.38%. The maximum absolute $E _ { 5 1 2 }$ error with the FP64 denominator is 0.01353, or 1.353 percentage points. Appendix E gives the per-matrix errors.

An energy error of this size matters near a reporting threshold. For the sft layer-16 query update, full SVD gives $k _ { 0 . 8 } = 2 5 5$ and $E _ { 2 5 6 } = 0 . 8 0 0 2 7 9 2$ . The baseline rank-256 approximation gives 0.7975442. All three baseline seeds and the stronger randomized setting fall below 0.8 when normalized by the accurate Frobenius norm. These are four configuration disagreements on one matrix. They demonstrate why an unreached approximate threshold is not a certified lower bound on energy rank.

Subspace angles are more sensitive still. The largest absolute discrepancy between the original FP32 angle calculation and the full-SVD reference is 3.09 degrees in this sample. For the PPO embedding, the original estimate is 0.09279 degrees and the full-SVD value is 0.00103 degrees. Randomized SVD supports eficient broad scans, while these functional-specific errors motivate full decompositions for the reported energy ranks and subspace angles. These findings highlight the importance of validating numerical approximations against the specific spectral or geometric quantity being reported.

## 5 Discussion and Limitations

The spectral and trajectory results describe complementary aspects of adaptation. Stable rank measures concentration around the leading singular value, while energy-threshold rank measures the breadth of the spectrum needed to retain a specified energy fraction. Consecutive-displacement cosine adds information about how successive changes interact. The public trajectories show anti-alignment across several RLVR recipes, and the controlled study demonstrates that the advantage system afects alignment magnitude under a shared actor schedule.

The rank reconciliation explains how the reported spectral descriptions coexist. The power-law construction establishes this compatibility analytically, and the full decompositions demonstrate it on selected updates. Its practical implication is that a small stable rank can accompany substantial energy outside a low-rank approximation. Assessing the task-level value of that energy requires evaluating models after spectral truncation.

The public survey measures aggregated changes on each release’s native checkpoint grid. Diferences in schedules, exposure budgets, storage precision, and related stages limit causal comparisons across releases. The supervised study spans four model families under one data mixture and schedule, with the FP32 followup covering two Llama seeds. Its negative Mistral and FP32 Llama pairs show why trajectory sign should be interpreted together with the model and numerical training configuration. The controlled RL study covers one 1B initialization, one task source, one shared trainer, and three paired seeds. Its seed-level intervals characterize variation within that setting.

The controlled intervention combines critic fitting, GAE, and advantage normalization, so its efect applies to that package. Generated-token exposure remains policy-dependent, and the small held-out diagnostic shows no consistent performance improvement. Both local arms have higher mean alignment than the public RLVR trajectories. Identifying the source of this diference requires interventions within the production implementation, with matched exposure and adequate task evaluation.

Full-SVD validation covers fourteen matrices selected by shape and layer position. The resulting accuracy measurements apply to this sample. All geometric quantities describe stored checkpoint weights, whose numerical precision afects apparent support sparsity and small displacements.

## 6 Conclusion

In this work, we studied language-model post-training by separating the structure of cumulative weight updates from the directions of successive training displacements. Our power-law construction showed that low stable rank and large fixed-energy rank are mathematically compatible, and full decompositions verified this separation in selected OLMo-2-7B updates. A dominant singular value can therefore coexist with hundreds of components needed to retain most of an update’s energy. This distinction clarifies why diferent rank measures can give contrasting descriptions of the same matrix. Trajectory measurements add information about how successive weight changes interact. All 131 measured adjacent pairs across nine public RLVR stages from the OLMo-2, OLMoE, and Olmo-3 releases had negative cosine on their native checkpoint grids, including PPO and GRPO releases. Supervised comparisons of OLMo-2-7B, Llama-3.2-3B, Qwen2.5-3B, and Mistral-7B-v0.3 showed configuration-dependent alignment, with late anti-alignment in Mistral and FP32 Llama runs. In a three-seed matched RL study initialized from OLMo-2-0425-1B-DPO and trained on the RLVR-GSM mathematical-answer dataset, replacing group-relative advantages with the critic–GAE package consistently lowered alignment under a shared actor schedule. Together, these findings support a more precise account of post-training geometry. Spectral concentration, energy coverage, and trajectory direction answer diferent questions and should be reported separately. Comparisons also require explicit initialization checkpoints, aggregation widths, and numerical settings. Our full-SVD checks further showed why approximation accuracy must be assessed for the particular quantity being reported. These distinctions provide a basis for interpreting weight-space measurements across post-training procedures.

## Acknowledgments

This work was supported by the NVIDIA Academic Grant Program, which provided 8× NVIDIA A100 80 GB GPU resources on the Brev platform.

The authors would like to acknowledge the use of Grammarly (an AI-assisted editing tool), for language editing, proofreading, and typographical corrections during the preparation of this manuscript. All technical content, analyses, and conclusions were manually drafted and verified by the authors.

## Reproducibility and Broader Impact

Appendices B–G specify the controlled protocol, exposure units, numerical checks, corrections, and crossfamily supervised comparisons. The accompanying revision artifacts retain per-case results and checkpoint identifiers. No new human-subject data are collected. The study analyzes open-weight models and mathematical-reasoning data. General risks associated with language models remain applicable.

## References

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. URL https://arxiv.org abs/2110.14168.

Nathan Halko, Per-Gunnar Martinsson, and Joel A. Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review, 53(2):217–288, 2011. doi: 10.1137/090771806.

Xinyu He and Xianghui Cao. Understanding post-training structural changes in large language models. arXiv preprint arXiv:2509.17866, 2025. URL https://arxiv.org/abs/2509.17866.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In International Conference on Learning Representations, 2023.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Xinxi Lyu, et al. Tülu 3: Pushing frontiers in open language model post-training. In Conference on Language Modeling, 2025. URL https://openreview.net/forum? id=i1uGbfHHpH.

Sagnik Mukherjee, Lifan Yuan, Dilek Hakkani-Tur, and Hao Peng. Reinforcement learning finetunes smal subnetworks in large language models. In Advances in Neural Information Processing Systems, volume 38, pp. 132119–132138, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ bf235a1d6780afd979f2f81676f43413-Abstract-Conference.html.

OLMo Team, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, et al. 2 OLMo 2 Furious (COLM’s version). In Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=2ezugTT9kU.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, 2023.

Olivier Roy and Martin Vetterli. The efective rank: A measure of efective dimensionality. In Proceedings of the 15th European Signal Processing Conference, pp. 606–610, 2007. URL https://www.eurasip.org/ Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. In International Conference on Learning Repre sentations, 2016.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. URL https://arxiv.org/abs/1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. URL https://arxiv.org/abs/2402.03300.

Zhennan Shen, Yanshu Li, Qingyu Yin, Chak Tou Leong, Zhilin Wang, Yanxu Chen, Rongduo Han, Sunbowen Lee, and Yi R. Fung. On the geometry of on-policy distillation. arXiv preprint arXiv:2606.07082, 2026. URL https://arxiv.org/abs/2606.07082.

Tianle Wang, Jiayu Liu, Zhongyuan Wu, Shenghao Jin, Wei Chen, Hao Xu, and Ning Miao. Linear dynamics in the RLVR training of large language models. arXiv preprint arXiv:2601.04537, 2026. URL https: //arxiv.org/abs/2601.04537.

Hanqing Zhu, Zhenyu Zhang, Hanxian Huang, DiJia Su, Zechun Liu, Jiawei Zhao, Igor Fedorov, Hamed Pirsiavash, Zhizhou Sha, Jinwon Lee, David Z. Pan, Zhangyang Wang, Yuandong Tian, and Kai Sheng Tai. The path not taken: RLVR provably learns of the principals. arXiv preprint arXiv:2511.08567, 2025. URL https://arxiv.org/abs/2511.08567.

## A Proof and Support-Sparsity Examples

Write $\textstyle S _ { k } = \sum _ { i = 1 } ^ { k } i ^ { - 2 \alpha }$ . Because $\sigma _ { 1 } = 1 , r _ { \mathrm { s t a b l e } } = S _ { n }$ and $k _ { \tau } = \operatorname* { m i n } \{ k \mid S _ { k } \geq \tau S _ { n } \}$ . For $0 < \alpha < 1 / 2$ , set $b = 1 - 2 \alpha$ . Integral comparison gives

$$
S _ { n } \leq 1 + \int _ { 1 } ^ { n } x ^ { - 2 \alpha } d x = 1 + { \frac { n ^ { b } - 1 } { b } } , \qquad S _ { n } \geq \int _ { 1 } ^ { n + 1 } x ^ { - 2 \alpha } d x \geq { \frac { n ^ { b } - 1 } { b } } .
$$

Also $\begin{array} { r } { S _ { k } \le \int _ { 0 } ^ { k } x ^ { - 2 \alpha } d x = k ^ { b } / b . } \end{array}$ . Hence $S _ { k } \geq \tau S _ { n }$ implies

$$
k ^ { b } \geq \tau ( n ^ { b } - 1 ) , \qquad k \geq \tau ^ { 1 / b } ( n ^ { b } - 1 ) ^ { 1 / b } .
$$

The latter expression grows linearly with n. Since $k _ { \tau } \leq n , k _ { \tau } = \Theta ( n )$ , whereas $r _ { \mathrm { s t a b l e } } = O ( n ^ { b } )$ and $b < 1$ Thus $r _ { \mathrm { s t a b l e } } / k _ { \tau } = O ( n ^ { - 2 \alpha } ) \to 0$

For $\alpha = 1 / 2 , S _ { k } = H _ { k }$ is the harmonic sum, with $\log ( k { + } 1 ) \leq H _ { k } \leq 1 { + } \log k$ . Consequently $r _ { \mathrm { s t a b l e } } \leq 1 + \log n .$ If $H _ { k } \geq \tau H _ { n }$ , then $1 + \log k \geq \tau \log n ,$ so $k \geq n ^ { \tau } / e$ . This proves $r _ { \mathrm { s t a b l e } } / k _ { \tau } = O ( \log n / n ^ { \tau } ) \to 0$ for every fixed $\tau \in ( 0 , 1 )$ and completes Proposition 1.

For an illustrative finite dimension, $n = 4 0 9 6 , \alpha = 0 . 4 .$ and $\tau = 0 . 8$ give the relaxed upper bound $r _ { \mathrm { s t a b l e } } \le$ $1 + 4 0 9 6 ^ { 0 . 2 } / 0 . 2 \ < \ 2 7 . 4$ and the lower bound $k _ { 0 . 8 } \gtrsim 4 6 9$ . This finite-dimensional example illustrates the separation in the constructed spectrum.

The support examples do not require a power law. Let $D = \mathrm { d i a g } ( d _ { 1 } , \ldots , d _ { n } )$ with every $d _ { i } > 0$ . Its support always contains exactly n entries, while its singular values are the $d _ { i }$ in decreasing order. Setting all $d _ { i } = 1$ gives stable rank $n ;$ setting $d _ { 1 } = 1$ and the others to a suficiently small positive value gives stable rank arbitrarily close to one and $k _ { \tau } = 1$ for fixed $\tau < 1$ . For generic orthogonal $U , V ,$ , the matrix $U D V ^ { \top }$ has the same spectrum and dense support. Conversely, dense unit vectors u, v give a dense rank-one matrix $u v ^ { \top }$ These examples show that support sparsity and spectral concentration describe distinct matrix properties.

## B Controlled Training Protocol and Health Checks

## B.1 Shared configuration

Both arms initialize the policy and frozen reference from allenai/OLMo-2-0425-1B-DPO. Training examples come from allenai/RLVR-GSM, a verifiable mathematical-answer task source related to GSM8K (Cobbe et al., 2021). We use the same prompt template, numeric final-answer checker, prompt stream, sampling configuration, and actor optimizer configuration within each seed pair. Table 6 lists the implementation settings.

Each response is trimmed after its first EOS token. That EOS remains in the loss, and subsequent padding does not. Within an eight-response minibatch, we sum token losses and divide by the total retained responsetoken count. Microbatch accumulation therefore preserves the same token-weighted minibatch objective in both arms. The reference penalty is the nonnegative sample estimator $\exp ( z _ { j } ) - z _ { j } - 1$ , where $z _ { j } = \log \pi _ { \mathrm { r e f } } ( y _ { j } \mid$ $x , y _ { < j } ) - \log \pi _ { \theta } ( y _ { j } \mid x , y _ { < j } )$

The group-relative arm assigns all tokens in response i the normalized outcome advantage $( R _ { i } - \bar { R } ) / ( s _ { R } +$ $1 0 ^ { - 6 } )$ , where the mean and population standard deviation are over the eight responses to that prompt. The critic–GAE arm instead forms terminal-reward temporal-diference residuals

$$
\delta _ { j } = r _ { j } + \gamma V ( s _ { j + 1 } ) - V ( s _ { j } ) , \qquad \widehat { A } _ { j } = \sum _ { l = 0 } ^ { T - j } ( \gamma \lambda ) ^ { l } \delta _ { j + l } ,
$$

Here $r _ { j }$ denotes the token reward, with the outcome reward assigned to the terminal token and zero reward at earlier tokens; the terminal bootstrap value is zero. The arm whitens advantages over all retained response tokens in the rollout. The critic target is the unwhitened GAE return. Its loss is one half the maximum of unclipped and clipped squared value errors, token averaged within the minibatch. The critic has a separate optimizer, so this experiment changes an advantage-system package that includes critic fitting and normalization.

Table 6: Configuration of the six controlled runs. A rollout batch is the unit that advances the learning-rate schedule and geometry clock.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Paired seeds</td><td>42, 43, 44</td></tr><tr><td>Rollout batches</td><td>360 per run</td></tr><tr><td>Prompts and responses</td><td>8 prompts, 8 responses per prompt</td></tr><tr><td>Prompt cap and response cap</td><td>2,048 tokens each</td></tr><tr><td>Sampling</td><td>Temperature 1, top-p 1, no top-k restriction</td></tr><tr><td>Actor update passes</td><td>4 per rollout batch</td></tr><tr><td>Actor minibatch and microbatch</td><td>8 responses and 1 response</td></tr><tr><td>Actor optimizer updates</td><td>32 per rollout, 11,520 per run</td></tr><tr><td>Policy parameters and optimizer states</td><td>FP32 with BF16 autocast</td></tr><tr><td>Optimizer</td><td>AdamW, zero weight decay,  $\epsilon _ { \mathrm { A d a m } } = 1 0 ^ { - 8 }$ </td></tr><tr><td>Actor learning rate</td><td> $4 \times 1 0 ^ { - 7 }$  , linear decay over rollouts</td></tr><tr><td>Ratio clip and gradient norm limit</td><td>0.2 and 1</td></tr><tr><td>Reference KL penalty coefficient</td><td>0.05</td></tr><tr><td>Reward</td><td>Correct final answer 1, incorrect 0, missing EOS -10</td></tr><tr><td>Critic architecture</td><td>Separate model backbone and zero-initialized scalar head</td></tr><tr><td>Critic optimizer</td><td>Separate AdamW, learning rate  $4 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>GAE discount and trace parameter</td><td> $\gamma = 1 , \lambda = 0 . 9 5$ </td></tr><tr><td>Value clipping</td><td>0.2 around the rollout value estimate</td></tr><tr><td>Full policy checkpoints</td><td>Every 60 rollouts and at completion</td></tr><tr><td>Online geometry widths</td><td>10, 20, 40, 60 rollout batches</td></tr></table>

The shared implementation checks finite losses and gradients and validates saved parameters. The experiment ran on the existing eight-A100 node, with six independent training processes for the paired seeds. The protocol was written before the corrected runs. An execution-order amendment started seeds 43 and 44 concurrently after smoke tests instead of waiting for the seed-42 60-rollout checkpoint. The outcome definitions, hyperparameters, and acceptance criteria remained unchanged, and all completed seeds enter the analysis.

## B.2 Scope relative to published recipes

Table 18 of the expanded OLMo-2 technical report lists a 2,048-token response cap, learning rate $4 \times 1 0 ^ { - 7 }$ with a linear schedule, and KL coeficient 0.05 (OLMo Team et al., 2025). The table is available in the expanded technical report. Our controlled study uses these settings but a 1B model and a smaller rollout batch. The shared custom trainer and critic initialization are specified in Table 6. Public release step labels are retained in their original units.

The 2,048-token cap expands the response budget beyond the earlier local 384/768-token experiments. Table 7 reports the realized cap-hit fractions for the controlled runs.

## B.3 Training health and small held-out diagnostic

Each run uses the same fixed 32 held-out examples with greedy decoding and the 2,048-token response cap. The initial policy answers 20 of 32 correctly. Table 7 reports final counts and training cap-hit fractions. The final counts range from 16 to 20 correct answers, compared with 20 initially. This small diagnostic shows no consistent improvement.

Table 7: Health diagnostics for the matched runs. Cap-hit fraction counts training responses that reach 2,048 tokens without EOS. Held-out results use 32 fixed examples and are descriptive.
<table><tr><td>Arm</td><td>Seed</td><td>Final correct</td><td>Final EOS</td><td>Training cap hit</td></tr><tr><td>Group-relative</td><td>42</td><td>16/32</td><td>32/32</td><td>1.102%</td></tr><tr><td>Group-relative</td><td>43</td><td>18/32</td><td>32/32</td><td>1.189%</td></tr><tr><td>Group-relative</td><td>44</td><td>20/32</td><td>32/32</td><td>0.786%</td></tr><tr><td>Critic-GAE</td><td>42</td><td>19/32</td><td>27/32</td><td>0.291%</td></tr><tr><td>Critic-GAE</td><td>43</td><td>18/32</td><td>28/32</td><td>0.269%</td></tr><tr><td>Critic-GAE</td><td>44</td><td>18/32</td><td>29/32</td><td>0.213%</td></tr></table>

## C Exposure Budgets and Matched-Width Results

All six runs generate 23,040 responses and perform 11,520 actor optimizer updates. Actual generated-token totals difer across policies and seeds, as Table 8 records. These totals count each generated token once across the four update passes. The saved results also retain token counts for every aggregation window.

Table 8: Realized exposure in the matched runs. Every row contains 360 rollouts, 23,040 responses, and 11,520 actor optimizer updates.
<table><tr><td>Arm</td><td>Seed</td><td>Generated tokens</td></tr><tr><td>Group-relative</td><td>42</td><td>5,286,865</td></tr><tr><td>Critic-GAE</td><td>42</td><td>5,832,426</td></tr><tr><td>Group-relative</td><td>43</td><td>4,779,950</td></tr><tr><td>Critic-GAE</td><td>43</td><td>5,861,391</td></tr><tr><td>Group-relative</td><td>44</td><td>5,072,345</td></tr><tr><td>Critic-GAE</td><td>44</td><td>4,776,819</td></tr></table>

For a width h, the paired comparison matches 64h responses and 32h actor optimizer updates. The latewindow rule retains cosine endpoints greater than 180. At widths 10, 20, 40, and 60 this gives 18, 9, 5, and 3 cosines per seed, respectively. The width-40 grid includes a window crossing the midpoint, under the prespecified endpoint-based rule. The secondary intervals in Table 9 use the same three seed pairs at each width. These descriptive intervals are unadjusted for multiple comparisons across widths.

Table 9: Late-window means and paired uncertainty at matched local widths. Seed values are ordered 42, 43, 44. Intervals are seed-level Student intervals.
<table><tr><td>Width</td><td>Group-relative seeds</td><td>Critic-GAE seeds</td><td>Paired 95% interval</td></tr><tr><td>10</td><td>+0.0897, +0.0669, +0.0576</td><td>+0.0220, +0.0140, +0.0223</td><td>[-0.0922, -0.0118]</td></tr><tr><td>20</td><td> $+ 0 . 1 3 9 2 , + 0 . 1 0 0 7 , + 0 . 0 9 0 4$ </td><td> $+ 0 . 0 0 6 3 , + 0 . 0 0 9 3 , + 0 . 0 0 9 0$ </td><td>[-0.1698, -0.0341]</td></tr><tr><td>40</td><td> $+ 0 . 2 5 9 4 , + 0 . 1 9 9 6 , + 0 . 1 7 6 2$ </td><td> $+ 0 . 0 1 3 2 , + 0 . 0 1 2 1 , + 0 . 0 0 4 9$ </td><td>[-0.2997, -0.1037]</td></tr><tr><td>60</td><td> $+ 0 . 3 1 7 8 , + 0 . 2 3 6 3 , + 0 . 2 2 2 9$ </td><td> $+ 0 . 0 1 9 3 , + 0 . 0 0 9 4 , + 0 . 0 0 6 0$ </td><td>[-0.3580, -0.1368]</td></tr></table>

## D Fixed-Width Temporal Checks and Checkpoint Inventory

Table 10 reports correlations at separations one, two, and three from full-parameter displacement Gram matrices. Each public or local RL row has six native displacements and therefore five, four, and three pairs. The SFT row has ten displacements and nine, eight, and seven pairs. Each summary averages the available pairs within its trajectory.

Table 10: Fixed-width directional correlations. Local controlled widths are 60 rollouts, the public PPO width is 60 release steps, and SFT width is 500 trainer steps. Only the local controlled arms have matched exposure units.
<table><tr><td>Trajectory</td><td>Separation 1</td><td>Separation 2</td><td>Separation 3</td></tr><tr><td>Public 7B PPO</td><td>-0.220162</td><td>-0.024420</td><td>-0.008670</td></tr><tr><td>7B SFT seed 42</td><td>+0.227479</td><td>+0.173982</td><td>+0.141078</td></tr><tr><td>Group-relative seed 42</td><td>+0.394134</td><td>+0.297558</td><td>+0.227189</td></tr><tr><td>Group-relative seed 43</td><td>+0.349763</td><td>+0.251865</td><td>+0.210152</td></tr><tr><td>Group-relative seed 44</td><td>+0.317653</td><td>+0.245254</td><td>+0.184630</td></tr><tr><td>Critic-GAE seed 42</td><td>+0.008941</td><td>+0.015477</td><td>+0.014242</td></tr><tr><td>Critic-GAE seed 43</td><td>+0.023189</td><td>+0.013839</td><td>-0.001997</td></tr><tr><td>Critic-GAE seed 44</td><td>+0.010831</td><td>+0.015073</td><td>+0.004019</td></tr></table>

The three OLMo-2-7B SFT seeds are assessed on the same 500-step grid through step 5,000. Their means and sign counts are given below. The longer endpoint trajectory continues the same seed-42 run. Appendix G extends this comparison to the other model families and precision settings.
<table><tr><td>SFT seed</td><td>Mean cosine</td><td>Positive pairs</td></tr><tr><td>42</td><td>+0.227479</td><td>9/9</td></tr><tr><td>43</td><td>+0.227273</td><td>9/9</td></tr><tr><td>44</td><td>+0.226147</td><td>9/9</td></tr></table>

The public 7B PPO calculation includes the immediate DPO initialization and six released checkpoints at labels 60 through 360. Its full-parameter Gram matrix covers 7,298,617,344 stored parameter coordinates. Adjacent cosines are −0.14379, −0.21128, −0.23898, −0.25304, and −0.25372. Equation 6 combines these cosines with displacement-norm ratios to obtain the projection coeficients. Across the nine public stages, the per-stage coeficient ranges are in Table 11.

Table 11: Projection coeficient onto the preceding displacement at native public widths. A negative coeficient describes one component of a step.
<table><tr><td>Stage</td><td>Minimum coefficient</td><td>Maximum coefficient</td></tr><tr><td>OLMo-2 7B</td><td>-0.2204</td><td>-0.1741</td></tr><tr><td>OLMo-2 13B RLVR1</td><td>-0.2534</td><td>-0.1807</td></tr><tr><td>OLMo-2 13B RLVR2</td><td>-0.2619</td><td>-0.1848</td></tr><tr><td>OLMo-2 13B final</td><td>-0.2643</td><td>-0.1837</td></tr><tr><td>OLMo-2 1B RLVR1</td><td>-0.2779</td><td>-0.1042</td></tr><tr><td>OLMo-2 1B final</td><td>-0.2762</td><td>-0.1774</td></tr><tr><td>OLMo-2 32B</td><td>-0.2603</td><td>-0.1172</td></tr><tr><td>OLMoE 1B active / 7B</td><td>-0.2751</td><td>-0.1884</td></tr><tr><td>Olmo-3 7B</td><td>-0.2615</td><td>-0.1930</td></tr></table>

The release repositories and immediate references used in the survey are listed below. All names have the allenai/ organization prefix. The accompanying machine-readable results retain full snapshot identifiers where available. Stage names identify the released training segment within each model lineage.

## OLMo-2 7B

Release OLMo-2-1124-7B-Instruct.   
Immediate reference OLMo-2-1124-7B-DPO.

OLMo-2 13B RLVR1 Release OLMo-2-1124-13B-Instruct-RLVR1. Immediate reference OLMo-2-1124-13B-DPO.

## OLMo-2 13B RLVR2

Release OLMo-2-1124-13B-Instruct-RLVR2.   
Immediate reference OLMo-2-1124-13B-Instruct-RLVR1. OLMo-2 13B final   
Release OLMo-2-1124-13B-Instruct.   
Immediate reference OLMo-2-1124-13B-Instruct-RLVR2.

## OLMo-2 1B RLVR1

Release OLMo-2-0425-1B-RLVR1.   
Immediate reference OLMo-2-0425-1B-DPO.

OLMo-2 1B final Release OLMo-2-0425-1B-Instruct. Immediate reference OLMo-2-0425-1B-RLVR1.

OLMo-2 32B Release OLMo-2-0325-32B-Instruct. Immediate reference OLMo-2-0325-32B-DPO.

OLMoE 1B active / 7B Release OLMoE-1B-7B-0125-Instruct. Immediate reference OLMoE-1B-7B-0125-DPO.

Olmo-3 7B   
Release Olmo-3-7B-Instruct.   
Immediate reference Olmo-3-7B-Instruct-DPO.

Table 12: Additional matrix-shape inventory for the public releases. Counts cover matrices of the stated shape and exclude one-dimensional normalization weights. The latter remain in full-parameter cosine calculations. OLMoE counts include individual expert matrices and router matrices.
<table><tr><td>Model</td><td>Rows</td><td>Columns</td><td>Count</td></tr><tr><td>OLMo-2 1B final</td><td>100,352</td><td>2,048</td><td>2</td></tr><tr><td rowspan="6">OLMo-2 13B final</td><td>2,048</td><td>8,192</td><td>16</td></tr><tr><td>8,192</td><td>2,048</td><td>32</td></tr><tr><td>2,048</td><td>2,048</td><td>64</td></tr><tr><td>100,352</td><td>5,120</td><td>2</td></tr><tr><td>5,120</td><td>13,824</td><td>40</td></tr><tr><td>13,824</td><td>5,120</td><td>80</td></tr><tr><td rowspan="2">OLMo-2 32B</td><td>5,120</td><td>5,120</td><td>160</td></tr><tr><td>100,352</td><td>5,120</td><td>2</td></tr><tr><td rowspan="6">Olmo-3 7B</td><td>5,120 27,648</td><td>27,648 5,120</td><td>64 128</td></tr><tr><td>1,024</td><td></td><td>128</td></tr><tr><td>5,120</td><td>5,120 5,120</td><td>128</td></tr><tr><td>100,278</td><td>4,096</td><td>2</td></tr><tr><td>4,096</td><td>11,008</td><td>32</td></tr><tr><td>11,008</td><td>4,096</td><td>64</td></tr><tr><td rowspan="5">OLMoE 1B active / 7B</td><td></td><td></td><td></td></tr><tr><td>4,096 50,304</td><td>4,096</td><td>128 2</td></tr><tr><td>2,048</td><td>2,048</td><td>1,024</td></tr><tr><td>1,024</td><td>1,024</td><td>2,048</td></tr><tr><td>64</td><td>2,048 2,048</td><td>16</td></tr><tr><td></td><td>2,048</td><td>2,048</td><td>64</td></tr></table>

## E Full-SVD Validation

The validation uses NumPy 1.26.4, scikit-learn 1.8.0, and PyTorch 2.12.0+cu130. A reduced QR factorization followed by LAPACK full SVD preserves the complete singular spectrum for rectangular cases. The largest selected matrix is the 100352 × 4096 embedding, for which reduced QR makes the full-spectrum calculation tractable.

For subspaces, let $U _ { 0 } , U _ { T }$ contain the leading 64 left singular vectors of the reference and endpoint weight matrices. We compute the largest principal angle as arcsin $\| ( I - U _ { 0 } U _ { 0 } ^ { \top } ) U _ { T } \| _ { 2 }$ using orthonormal FP64 bases. This avoids the small-angle floor of an arccos calculation. We separately reproduce the original FP32 arccos calculation. Table 13 gives baseline approximation errors and both angle estimates.

Original angle is the largest principal angle computed by the original randomized-SVD and FP32 procedure. Let $\widehat { U } _ { 0 } , \widehat { U } _ { T }$ be the first 64 left singular vectors returned by the rank-256 approximations. In degrees, this calculation is

$$
\phi _ { \mathrm { o r i g i n a l } } = { \frac { 1 8 0 } { \pi } } \operatorname { a r c c o s } \left( \mathrm { c l i p } _ { [ - 1 , 1 ] } \Big [ \sigma _ { \mathrm { m i n } } ( \widehat { U } _ { 0 } ^ { \top } \widehat { U } _ { T } ) \Big ] \right) .
$$

The cross-product and its singular values are computed in FP32. For energy, the approximation and absolute error are

$$
\widehat { E } _ { 5 1 2 } = \frac { \sum _ { i = 1 } ^ { 5 1 2 } \widehat { \sigma } _ { i } ^ { 2 } } { \| \Delta \| _ { F } ^ { 2 } } , \qquad \varepsilon _ { E } = \left| \widehat { E } _ { 5 1 2 } - E _ { 5 1 2 } ^ { \mathrm { f u l l } } \right| .
$$

Here ${ \widehat { \sigma } } _ { i }$ are the rank-512 randomized-SVD singular values, $E _ { 5 1 2 } ^ { \mathrm { f u l l } }$ comes from the full decomposition, and both fractions use the same FP64 squared Frobenius norm. Table 13 reports $\varepsilon _ { E }$ as a fraction; 0.01 equals one percentage point.

Table 13: Baseline numerical audit on the fourteen selected matrices. The stable-rank error isolates singularvalue error with an FP64 Frobenius norm. Energy error is absolute error in $E _ { 5 1 2 }$ . Here $E _ { 5 1 2 } = E _ { k } ( \Delta )$ from Equation 3 at $k = 5 1 2 ;$ ; the energy-error and original-angle formulas appear above. Angles are the largest of the 64 principal angles, in degrees. The original-pipeline stable-rank error can be larger because its normalization difers.
<table><tr><td>Run</td><td>Matrix</td><td>Relative stable-rank error</td><td>Energy error</td><td>Original angle</td><td>Full-SVD angle</td></tr><tr><td>PPO</td><td>Query 0</td><td>3.9e-07</td><td>0.00514</td><td>0.31404</td><td>0.29726</td></tr><tr><td>PPO</td><td>MLP up 0</td><td>1.5e-06</td><td>0.00939</td><td>0.09487</td><td>0.07653</td></tr><tr><td>PPO</td><td>Query 16</td><td>5.7e-07</td><td>0.00818</td><td>0.09691</td><td>0.05868</td></tr><tr><td>PPO</td><td>MLP down 16</td><td>4.3e-07</td><td>0.01029</td><td>0.08623</td><td>0.08062</td></tr><tr><td>PPO</td><td>Query 31</td><td>1e-07</td><td>0.00761</td><td>0.13848</td><td>0.13671</td></tr><tr><td>PPO</td><td>MLP up 31</td><td>1.7e-07</td><td>0.01044</td><td>0.13122</td><td>0.17695</td></tr><tr><td>PPO</td><td>Embedding</td><td>8.8e-05</td><td>0.01350</td><td>0.09279</td><td>0.00103</td></tr><tr><td>SFT</td><td>Query 0</td><td>4.2e-07</td><td>0.00029</td><td>76.20685</td><td>76.26711</td></tr><tr><td>SFT</td><td>MLP up 0</td><td>1.1e-06</td><td>0.01247</td><td>1.62522</td><td>1.96539</td></tr><tr><td>SFT</td><td>Query 16</td><td>2e-07</td><td>0.00296</td><td>8.71772</td><td>8.72792</td></tr><tr><td>SFT</td><td>MLP down 16</td><td>7.5e-05</td><td>0.01282</td><td>1.94431</td><td>2.76619</td></tr><tr><td>SFT</td><td>Query 31</td><td>2e-07</td><td>0.00412</td><td>5.86628</td><td>5.85589</td></tr><tr><td>SFT</td><td>MLP up 31</td><td>3.9e-07</td><td>0.01231</td><td>5.37735</td><td>2.28540</td></tr><tr><td>SFT</td><td>Embedding</td><td>1.9e-05</td><td>0.01353</td><td>0.10279</td><td>0.06163</td></tr></table>

The threshold failure occurs in the SFT layer-16 query projection. Full SVD gives $E _ { 2 5 6 } = 0 . 8 0 0 2 7 9 2 0$ and $k _ { 0 . 8 } = 2 5 5$ . Independent baseline rank-256 calls at seeds 42, 43, and 44 give 0.79754418, 0.79760081, and 0.79759122. Forty oversamples and eight iterations at seed 42 give 0.79972833. All four approximate energies use the same FP64 denominator as the full result. The original FP32 denominator introduces an additional shift. Full-spectrum ranks resolve the threshold for these selected cases.

## F Corrections and Disposition of Earlier Analyses

The original public 7B PPO endpoint subtracted SFT and therefore contained both DPO and PPO changes. Revised PPO spectral cases subtract the immediate DPO checkpoint. We withdraw the earlier MLP stablerank headline near 37 as a pure-PPO statistic. Separately, the intended reservoir sketch admitted no coordinates after its first two million entries. All reported full-model cosines now use full-coordinate accumulation, and we withdraw the earlier sketch-error estimate.

The reference cancels between two fixed checkpoints but afects the initial displacement and all cumulative descriptors. Independently, correcting the sketch can change every previously sampled global cosine.

The Qwen ablations had unequal optimizer-update counts, unset critic seeds, and post-EOS padding entering losses. We replace them with the shared, paired implementation in Section 3.2 and withdraw claims that clipping is ruled out or that an advantage estimator explains production anti-alignment.

The original local 7B GRPO run started from SFT, not DPO, and its requested 60-step reconstruction is unavailable in the retained inventory. We remove the matched-initialization claim. The paired 1B experiment supplies the controlled comparison in the revised analysis. Energy-rank and angle claims follow Section 4.4. These corrections preserve the spectral compatibility proposition.

## G Cross-Family SFT and Numerical Precision

## G.1 Shared training and measurement protocol

The cross-family study extends the OLMo-2-7B comparison to meta-llama/Llama-3.2-3B, Qwen/Qwen2. 5-3B, and mistralai/Mistral-7B-v0.3. All runs train on allenai/tulu-3-sft-olmo-2-mixture with the same plain-text system, user, and assistant role formatting. The causal language-model loss supervises al retained non-padding conversation tokens with a preceding token, as specified in Equation 1. The maximum sequence length is 4,096 tokens. The efective batch contains 64 examples, and AdamW uses learning rate $1 0 ^ { - 5 }$ , zero weight decay, 3% warmup, and linear decay over the full two-epoch schedule of 29,356 optimizer steps. A stop callback ends the comparison at step 5,000 without rescaling that schedule.

The baseline runs hold model parameters and Adam states in BF16. The Llama precision follow-ups hold both in FP32 and perform forward and backward computation under BF16 autocast. The BF16 Llama and Qwen runs use microbatches of two examples per device, while the Mistral and FP32 Llama runs use one; gradient accumulation and device count maintain the efective batch of 64. All arms use gradient checkpointing. The precision comparison consequently concerns the numerical training configuration, including parameter and optimizer-state precision.

At steps 0, 500, . . . , 5,000, we record the full parameter vector. The checkpoint-based and in-memory implementations both form displacements in FP64 before accumulating their full-coordinate Gram matrix. For tied parameters, each independent parameter is counted once. The in-memory method avoids retaining another complete checkpoint series on disk. A comparison on Mistral seed 43 at steps 0, 500, and 1,000 gives a maximum absolute Gram-entry discrepancy of $1 . 2 8 \times 1 0 ^ { - 1 3 }$ and relative discrepancy of $1 . 7 5 \times 1 0 ^ { - 1 5 }$ across 7,248,023,552 parameter coordinates. Table 14 reports every completed 5,000-step trajectory in the study.

The extensions were sequential. Three Llama seeds formed the initial cross-family test; Mistral and Qwen were added subsequently. The negative Mistral outcome motivated a second Mistral seed and the Llama precision follow-ups. A 2,000-step FP32 seed-42 pilot preceded a fresh 5,000-step run with the same seed and schedule. Its first three cosines, +0.1766, +0.0678, and +0.0233, agree closely with the corresponding values in Table 16. The tables summarize the complete 5,000-step grids, with the pilot treated as a diagnostic of the same setting.

## G.2 Family and precision dependence

The BF16 Llama comparison reproduces positive alignment across three seeds, with means +0.2375, +0.2381, and +0.2372. The Qwen run also has nine positive pairs and mean +0.2226. In contrast, the two Mistral seeds each have seven negative pairs, beginning with the pair ending at step 2,000. Their full-grid means are −0.0100 and −0.0099. Thus the shared supervised recipe produces diferent directional patterns across models.

The temporal summaries retain this contrast at larger widths. Table 15 holds width fixed when increasing separation and separately aggregates displacements into widths 1,000 and 1,500. The BF16 Llama and Qwen means remain positive under these summaries. Mistral’s width-1,000 means are approximately −0.0159 for both seeds.

Within Llama, changing the numerical training configuration substantially lowers alignment. The FP32 seed-42 and seed-43 full-grid means are +0.0206 and +0.0209, with four positive and five negative pairs

Table 14: SFT comparisons at width 500 optimizer steps through step 5,000. Precision denotes trainable parameters and Adam states. Each run contains nine adjacent displacement pairs. OLMo rows reproduce the seed summary in Appendix D.
<table><tr><td>Model</td><td>Precision</td><td>Seed</td><td>Mean cosine</td><td>Positive</td><td>Negative</td></tr><tr><td>OLMo-2-7B</td><td>BF16</td><td>42</td><td>+0.2275</td><td>9/9</td><td>0/9</td></tr><tr><td>OLMo-2-7B</td><td>BF16</td><td>43</td><td>+0.2273</td><td>9/9</td><td>0/9</td></tr><tr><td>OLMo-2-7B</td><td>BF16</td><td>44</td><td>+0.2261</td><td>9/9</td><td>0/9</td></tr><tr><td>Llama-3.2-3B</td><td>BF16</td><td>42</td><td>+0.2375</td><td>9/9</td><td>0/9</td></tr><tr><td>Llama-3.2-3B</td><td>BF16</td><td>43</td><td>+0.2381</td><td>9/9</td><td>0/9</td></tr><tr><td>Llama-3.2-3B</td><td>BF16</td><td>44</td><td>+0.2372</td><td>9/9</td><td>0/9</td></tr><tr><td>Qwen2.5-3B</td><td>BF16</td><td>42</td><td>+0.2226</td><td>9/9</td><td>0/9</td></tr><tr><td>Mistral-7B-v0.3</td><td>BF16</td><td>42</td><td>-0.0100</td><td>2/9</td><td>7/9</td></tr><tr><td>Mistral-7B-v0.3</td><td>BF16</td><td>43</td><td>-0.0099</td><td>2/9</td><td>7/9</td></tr><tr><td>Llama-3.2-3B</td><td>FP32</td><td>42</td><td>+0.0206</td><td>4/9</td><td>5/9</td></tr><tr><td>Llama-3.2-3B</td><td>FP32</td><td>43</td><td>+0.0209</td><td>4/9</td><td>5/9</td></tr></table>

each. Table 16 shows the change over training. Both runs first become negative at the pair ending at step 3,000 and remain negative through step 5,000. Full-grid means remain positive because of the early positive pairs. These results establish that the positive sign in the BF16 Llama runs is sensitive to numerical training configuration; they also show that supervised updates can exhibit consecutive anti-alignment.

Table 15: Additional cross-family SFT temporal summaries. Separation-two and separation-three means use width 500 and contain eight and seven pairs. Width-1,000 and width-1,500 means aggregate consecutive displacements and contain four and two adjacent pairs. The width-1,500 grid ends at step 4,500.
<table><tr><td>Model</td><td>Precision</td><td>Seed</td><td>Sep. 2</td><td>Sep. 3</td><td>Width 1,000</td><td>Width 1,500</td></tr><tr><td>Llama-3.2-3B</td><td>BF16</td><td>42</td><td>+0.1889</td><td>+0.1593</td><td>+0.3195</td><td>+0.3609</td></tr><tr><td>Llama-3.2-3B</td><td>BF16</td><td>43</td><td>+0.1891</td><td>+0.1578</td><td>+0.3194</td><td>+0.3594</td></tr><tr><td>Llama-3.2-3B</td><td>BF16</td><td>44</td><td>+0.1881</td><td>+0.1584</td><td>+0.3172</td><td>+0.3605</td></tr><tr><td>Qwen2.5-3B</td><td>BF16</td><td>42</td><td>+0.1784</td><td>+0.1510</td><td>+0.3037</td><td>+0.3449</td></tr><tr><td>Mistral-7B-v0.3</td><td>BF16</td><td>42</td><td>-0.0031</td><td>-0.0005</td><td>-0.0159</td><td>-0.0144</td></tr><tr><td>Mistral-7B-v0.3</td><td>BF16</td><td>43</td><td>-0.0035</td><td>-0.0004</td><td>-0.0159</td><td>-0.0150</td></tr><tr><td>Llama-3.2-3B</td><td>FP32</td><td>42</td><td>+0.0273</td><td>+0.0275</td><td>+0.0408</td><td>+0.0691</td></tr><tr><td>Llama-3.2-3B</td><td>FP32</td><td>43</td><td>+0.0269</td><td>+0.0273</td><td>+0.0408</td><td>+0.0683</td></tr></table>

Table 16: Llama precision comparison at width 500. Each entry is the cosine between the displacement ending at the listed step and its immediate predecessor. BF16 and FP32 denote parameter and Adam-state precision.
<table><tr><td rowspan="2">Endpoint</td><td colspan="2">Seed 42</td><td colspan="2">Seed 43</td></tr><tr><td>BF16</td><td>FP32</td><td>BF16</td><td>FP32</td></tr><tr><td>1000</td><td>+0.3724</td><td>+0.1765</td><td>+0.3676</td><td>+0.1774</td></tr><tr><td>1500</td><td>+0.3763</td><td>+0.0678</td><td>+0.3767</td><td>+0.0708</td></tr><tr><td>2000</td><td>+0.2993</td><td>+0.0232</td><td>+0.3023</td><td>+0.0226</td></tr><tr><td>2500</td><td>+0.2454</td><td>+0.0031</td><td>+0.2478</td><td>+0.0031</td></tr><tr><td>3000</td><td>+0.2120</td><td>-0.0066</td><td>+0.2072</td><td>-0.0095</td></tr><tr><td>3500</td><td>+0.1868</td><td>-0.0142</td><td>+0.1911</td><td>-0.0141</td></tr><tr><td>4000</td><td>+0.1648</td><td>-0.0184</td><td>+0.1660</td><td>-0.0177</td></tr><tr><td>4500</td><td>+0.1493</td><td>-0.0217</td><td>+0.1481</td><td>-0.0216</td></tr><tr><td>5000</td><td>+0.1309</td><td>-0.0242</td><td>+0.1359</td><td>-0.0231</td></tr></table>