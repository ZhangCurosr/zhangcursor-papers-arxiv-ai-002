# Stochastic Pruning of the DPO Computation Graph: 1.4× Faster Training and Lower Peak Memory

Yubing Wu<sup>1,†</sup> Wei Chen<sup>1,†</sup> Junmei Yang<sup>1</sup> Delu Zeng<sup>1,\*</sup> Tiewen Pan<sup>2</sup>

<sup>1</sup>South China University of Technology

<sup>2</sup> Shenzhen Technology University

<sup>†</sup>Equal contribution. <sup>\*</sup>Corresponding author: dlzeng@scut.edu.cn

## Abstract

Direct preference optimization (DPO) incurs training costs in output scoring, sequence execution, and backpropagation, so reducing supervision alone does not necessarily save computation. We introduce a unified framework for stochastic pruning of the DPO computation graph that independently samples response positions for scoring, response prefixes for execution, and preference branches for backpropagation. By making these selections before the corresponding operations, the framework omits computation at each stage of a training step. Our analysis characterizes the objective changes induced by position and sequence pruning and uses inverse-probability weighting to obtain a conditionally unbiased singlebranch gradient estimator for the sampled objective. Controlled measurements show that position and sequence pruning primarily reduce peak memory, with most output-memory savings attributable to an implementation that retains full response supervision. Branch pruning yields a 1.426× step-time speedup over a concatenated DPO baseline and reduces training time by about 30% relative to stock DPO across two model families, with slightly lower mean accuracy and benchmark-dependent changes in open-ended scores. These findings relate the resource benefits of stochastic graph pruning to the computation omitted and the resulting changes in the training objective.

## 1 Introduction

Direct preference optimization (DPO) formulates alignment as a logistic comparison of policy– reference likelihood ratios, with gradient contributions from both chosen and rejected responses (Rafailov et al., 2023). In large-vocabulary language-model training, materialized logits can dominate memory consumption (Wijmans et al., 2025), and attention I/O and operator fusion also affect memory use and training throughput (Dao et al., 2022; Hsu et al., 2025). These costs motivate our question of which computations a preferencetraining step can omit.

Prior work changes token supervision through confidence-based token selection or weighting (Yoon et al., 2025; Jiang et al., 2025), reward-based selection (Yang et al., 2025b), and weighting from contrastive model predictions (Liu et al., 2025), while memory-efficient kernels change outputscore computation (Wijmans et al., 2025; Hsu et al., 2025). This distinction motivates our question: which parts of a DPO step can be omitted, and how do their resource savings relate to changes in the optimization objective?

Preference-objective structure affects optimization (Chen et al., 2026), while shared-parameter updates can couple response likelihoods (Yuan et al., 2025) and reduce preferred-response probability (Razin et al., 2025). Our token-level analysis examines this coupling (Appendix C.1), and Proposition 1 separates reward preservation in expectation from the nonlinear loss.

We introduce a framework that samples the DPO computation graph to control training cost (Fig. 1). Each training step jointly selects the executed response prefixes, the positions used to estimate rewards, and the branches that carry gradients. These selections precede sequence execution, vocabulary projection, and backpropagation, allowing computation to be omitted throughout the step. This view unifies position, sequence, and branch pruning through their effects on the sampled objective and its gradient. Our analysis characterizes the objective changes induced by reward sampling, while inverse-probability weighting makes the branch gradient conditionally unbiased for that objective.

Our contributions are:

• a unified framework for sampling DPO computation across response execution, token scoring, and backpropagation, with separately configurable sampling decisions.

![](images/3837cb7b57a5724854f79afdbdc69e59883cfc9b54c17448b63a2931b29779db.jpg)  
Figure 1: Stochastic interventions on the DPO computation graph. The two response branches share policy parameters. Rewards average policy–reference log-ratios over scored response positions. Fixed-reference computations use matching prefixes and positions and are omitted. Sampling outcomes and counts are schematic.

• a characterization of the position and prefix objective changes, tolerance conditions that relate their optimal gaps to the retention rate and the truncation level, and a conditionally unbiased branch estimator with a stated convergence rate.

• controlled comparisons showing that full supervision retains most output-memory savings, that branch pruning reduces training time across model families, and that mean accuracy is slightly lower while open-ended score changes vary by benchmark (Secs. 4.2, 4.4 and 4.5).

## 2 Related Work

Token-level credit in preference optimization. A growing family of methods changes where DPO assigns credit through confidence-based token selection (Yoon et al., 2025), confidence-dependent weighting of varied tokens (Jiang et al., 2025), trained oracles for token rewards (Yang et al., 2025b), and contrastive token weights (Liu et al., 2025). Other approaches use optimal-transport reweighting (Li et al., 2025a), gradient-attribution priors (Yang et al., 2026), prefix-level ratio matching (Nguyen et al., 2026), implicit-reward credit (Zhao et al., 2026), attention-derived weights (Huang et al., 2026), minimal-edit emphasis in clinical dialogue (Lai et al., 2026a), or token-level KL control (Zeng et al., 2024). Our framework places supervision selection within the computation of a complete DPO step. Matched selection controls motivate uniform membership (Sec. E.1), and fullretention controls distinguish supervision deletion from implementation savings (Tab. 1).

Efficient preference training. Cut Cross-Entropy avoids materializing the token-byvocabulary logit matrix (Wijmans et al., 2025), and Liger-Kernel uses fusion and chunked cross-entropy computation to reduce memory and improve training throughput (Hsu et al., 2025). FlashAttention relates attention efficiency to memory accesses rather than FLOP counts alone (Dao et al., 2022), while activation checkpointing trades recomputation for lower activation storage (Chen et al., 2016). Related efficiency work covers flow training (Du et al., 2022) and diffusion inference (Li et al., 2025b). SePO reports reduced GPU training hours through offline token selection (Yang et al., 2025b). Our framework considers how sampling throughout a DPO step changes both executed computation and the training objective. Controlled profiles separate implementation savings from the effects of reduced supervision and execution (Sec. 4.2).

Stochastic computation and optimization dynamics. Dropout randomly removes units (Srivastava et al., 2014) and stochastic depth bypasses sampled layers (Huang et al., 2016), while importance sampling can reduce stochastic-gradient variance using gradient-dependent probabilities (Katharopoulos and Fleuret, 2018). Preferenceoptimization analyses study gradient entanglement (Yuan et al., 2025), likelihood displacement (Razin et al., 2025), logit-space squeezing (Luo et al., 2026), and gated rejected contributions (Mouiche, 2026). Density-ratio estimation provides a distribution-comparison perspective (Chen et al., 2025) for preference objectives such as DDRO (Higuchi and Suzuki, 2025), whose update structure is analyzed alongside margin-based losses (Chen et al., 2026). Our analysis separates objective changes from gradient-estimation noise under composed sampling (Proposition 2 and Sec. 3.5). Shared-parameter response changes are analyzed in Appendix C.1.

## 3 Stochastic Graph Pruning

## 3.1 Formulation

Let $\mathcal { D } = \{ ( \pmb { x } , \pmb { y } _ { w } , \pmb { y } _ { l } ) \}$ contain prompts x with chosen and rejected responses ${ \mathbf { } } _ { \mathbf { } } \mathbf { } _ { \mathbf { } } \mathbf { } \mathbf { , } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { \Psi } \mathbf  \mathbf { } \mathbf \Psi \mathbf { } \mathbf \mathbf { } \mathbf \Psi \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { \Psi \Psi } \mathbf \mathbf { \mathbf \Psi } \mathbf \mathbf  \mathbf \Psi \Psi \mathbf $ . Write $\pi _ { \pmb { \theta } }$ for the policy with trainable parameters θ and $\pi _ { \mathrm { r e f } }$ for the fixed reference (Rafailov et al., 2023). For side $b \in \{ w , l \}$ , let $T _ { b } \geq 1$ count response tokens, excluding prompt and padding. At position t, define the target-token log-ratio

$$
u _ { b , t } = \log { \frac { \pi _ { \pmb { \theta } } ( y _ { b , t } \mid \pmb { x } , \pmb { y _ { b , < t } } ) } { \pi _ { \mathrm { r e f } } ( y _ { b , t } \mid \pmb { x } , \pmb { y _ { b , < t } } ) } } .\tag{1}
$$

Here $y _ { b , t }$ is the target token and $_  { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { \mathbf { } } { \mathbf { } } ^ { } { \mathbf { } } ^ { } { } { \mathbf { } } ^ { } { }$ its preceding response prefix. With mean rewards $R _ { b } \ =$ $T _ { b } ^ { - 1 } \sum _ { t = 1 } ^ { \overline { { T _ { b } } } } u _ { b , t }$ , margin $\Delta = R _ { w } - R _ { l }$ , and scale $\beta > 0$ , DPO minimizes the loss (Rafailov et al., 2023):

$$
F ( \pmb \theta ) = \mathbb { E } _ { \mathcal { D } } [ \mathcal { L } _ { \mathrm { D P O } } ] = \mathbb { E } _ { \mathcal { D } } [ \log \left( 1 + e ^ { - \beta \Delta } \right) ] .\tag{2}
$$

This length normalization parallels SimPO (Meng et al., 2024), but retains policy–reference log-ratios.

Fig. 1 separates three costs: dense vocabulary projection materializes $T _ { b } \times V$ logits for vocabulary size V (Wijmans et al., 2025), sequence execution computes attention and stores or recomputes activations (Dao et al., 2022; Chen et al., 2016), and backpropagation includes both responses (Rafailov et al., 2023).

Write $\mathbb { I } _ { M } = \{ 1 , \dots , M \}$ . On side b, sequence pruning selects length $M _ { b } \in \mathbb { I } _ { T _ { b } }$ through cutoff fraction $\rho _ { b } \in ( 0 , 1 ]$ , retaining $\tilde { \pmb { y } } _ { b } = \pmb { y } _ { b , 1 : M _ { b } }$ . Position pruning selects $\mathbb { S } _ { b } \subseteq \mathbb { I } _ { M _ { k } }$ with $k _ { b } = | \mathbb { S } _ { b } |$ controlled by retention $\alpha \in ( 0 , 1 ]$ . Branch pruning selects $b ^ { \star } \in \{ w , l \}$ with probabilities $q _ { w } , q _ { l } > 0$ $q _ { w } + q _ { l } = 1$ . All sampling laws are parameterindependent, and rewards average over selected positions. We use $\sigma$ for the sigmoid and $\mathrm { s g }$ for stop-gradient, which preserves values and blocks derivatives. Appendix A summarizes the notation.

## 3.2 Stochastic Position Pruning

Position pruning retains a uniform subset of response positions for scoring while preserving the executed response as context. Draw $\mathbb { S } _ { b } \sim$ Select $( M _ { b } , \alpha )$ by

$$
\begin{array} { c } { k _ { b } = \operatorname* { m a x } ( 1 , \lceil \alpha M _ { b } \rceil ) , } \\ { \mathrm { S e l e c t } ( M _ { b } , \alpha ) = \mathrm { U n i f } \{ \mathbb { S } \subseteq \mathbb { I } _ { M _ { b } } : | \mathbb { S } | = k _ { b } \} . } \end{array}\tag{3}
$$

Standalone position pruning uses $M _ { b } = T _ { b }$ . Define the selected-position reward

$$
R _ { b } ( \mathbb { S } _ { b } ) = \frac { 1 } { k _ { b } } \sum _ { t \in \mathbb { S } _ { b } } u _ { b , t } ,\tag{4}
$$

with masked margin $\Delta = R _ { w } ( \mathbb { S } _ { w } ) - R _ { l } ( \mathbb { S } _ { l } )$ . The subset is drawn before the language-model head, without computing policy probabilities. Applying the head only at $\mathbb { S } _ { b }$ evaluates Eq. (4) exactly, skipping $M _ { b } - k _ { b }$ vocabulary projections, their logits, and their backward traffic.

Uniform sampling preserves the mean reward, but the nonlinear loss introduces a gap.

Proposition 1. For a fully executed response, let $\mathbb { U } _ { k }$ be a uniform size-k subset of I and $\phi _ { t } ~ =$ $\nabla _ { \theta }$ log $\pi _ { \pmb { \theta } } ( y _ { t } \ \mid \ x , \pmb { y } _ { < t } )$ . Let $\Delta _ { U }$ be the margin under independent side masks and $\Delta _ { \mathrm { f u l l } }$ the fullresponse margin. At anyfixed policy,

$$
\begin{array} { c } { \mathbb { E } [ R ( \mathbb { U } _ { k } ) ] = R ( \mathbb { I } _ { T } ) , } \\ { \mathbb { E } \Big [ \frac { 1 } { k } \sum _ { t \in \mathbb { U } _ { k } } \phi _ { t } \Big ] = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \phi _ { t } , } \end{array}\tag{5}
$$

and

$$
0 \leq \mathbb { E } [ \ell ( \Delta _ { U } ) ] - \ell ( \Delta _ { \mathrm { f u l l } } ) \leq \frac { \beta ^ { 2 } } { 8 } \mathrm { V a r } ( \Delta _ { U } ) ,\tag{6}
$$

where Var is over the sampled masks and $\begin{array} { r } { \mathrm { V a r } ( \Delta _ { U } ) ~ = ~ \sum _ { b : T _ { b } > 1 } \frac { T _ { b } - k _ { b } } { k _ { b } ( T _ { b } - 1 ) } \stackrel { \sim } { v _ { b } } } \end{array}$ . Here $\begin{array} { r l } { \bar { u } _ { b } } & { { } = } \end{array}$ $T _ { b } ^ { - 1 } \sum u _ { b , t }$ and $\begin{array} { r } { v _ { b } = T _ { b } ^ { - 1 } \sum _ { t } ( u _ { b , t } - \bar { u } _ { b } ) ^ { 2 } } \end{array}$ are the population mean and variance. Single-token sides contribute zero variance.

At α = 1, selected-head execution preserves the full-response objective while avoiding projection outside valid response positions (Tab. 1).

To bound the sampled optimum, assume throughout this subsection and Secs. B.5 and B.6 that $| u _ { b , t } | \leq U , \| \nabla _ { \theta } u _ { b , t } \| \leq G , v _ { b } \leq \bar { v }$ , and $T _ { b } \geq$ $T _ { 0 } \geq 2$ on the analyzed region. The objectives are twice differentiable there and attain their minima.

Corollary 1 (Optimality gap of position pruning). Let $\pmb { \theta } ^ { * }$ minimize thefull-response objective F and $\pmb { \theta } _ { A } ^ { * }$ minimize $F _ { A } ( \pmb \theta ) = \mathbb { E } _ { \pmb { \mathcal D } } \mathbb { E } _ { S } [ \ell ( \Delta _ { U } ) ]$ . Then

$$
0 \leq F ( \pmb { \theta } _ { A } ^ { * } ) - F ( \pmb { \theta } ^ { * } ) \leq \frac { \beta ^ { 2 } } { 8 } \mathbb { E } _ { \mathcal { D } } \big [ \mathrm { V a r } _ { S } ( \Delta _ { U } ) \big ] .\tag{7}
$$

The variance is evaluated at $\pmb { \theta } ^ { * }$ over masks, then averaged over data. Since $k _ { b } \geq \alpha T _ { b }$ , Proposition 1 gives $\mathbb { E } _ { \mathcal { D } } [ \mathrm { V a r } _ { S } ( \Delta _ { U } ) ] \le 2 \bar { v } r _ { A }$ , where

$$
r _ { A } : = \frac { 1 - \alpha } { \alpha ( T _ { 0 } - 1 ) } .\tag{8}
$$

Thus the bound decays as $1 / T _ { 0 }$ at fixed retention and vanishes at $\alpha = 1 . \mathrm { ~ A ~ }$ sufficient condition for loss tolerance $\varepsilon _ { F } > 0$ is

$$
\alpha \ge \frac { 1 } { 1 + 4 \varepsilon _ { F } ( T _ { 0 } - 1 ) / ( \beta ^ { 2 } \bar { v } ) } ,\tag{9}
$$

A second-order expansion also gives $\parallel \nabla F _ { A } -$ $\nabla F \| = { \mathcal O } ( r _ { A } )$ (Appendix B.6). If $\nabla ^ { 2 } F \succeq \mu I$ $\mu ~ > ~ 0$ , along the segment joining interior stationary points of $F _ { A }$ and F within the analyzed region, their displacement is $\mathcal { O } ( r _ { A } )$ rather than $\mathcal { O } ( \sqrt { r _ { A } } )$ . The loss bound requires no curvature assumption. Sufficient curvature conditions appear in Appendix B.7.

## 3.3 Stochastic Sequence Pruning

Sequence pruning truncates each response before policy and reference execution. For minimum retained fraction $\rho _ { \mathrm { m i n } } \in ( 0 , 1 )$ , draw $( \tilde { \pmb { y } } _ { b } , M _ { b } ) \sim$ Prefix $( { \boldsymbol { \mathbf { \mathit { y } } } } _ { b } , \rho _ { \mathrm { m i n } } )$ by

$$
\begin{array} { r l } & { \rho _ { b } \sim \mathcal { U } ( \rho _ { \operatorname* { m i n } } , 1 ) , } \\ & { M _ { b } = \operatorname* { m a x } ( 1 , \lceil \rho _ { b } T _ { b } \rceil ) , } \\ & { \tilde { y } _ { b } = y _ { b , 1 : M _ { b } } . } \end{array}\tag{10}
$$

Both models repack the prompt with $\tilde { \mathbf { y } } _ { b } ,$ removing suffix attention, activations, and recomputation. The reward is $\begin{array} { r } { R _ { b } = M _ { b } ^ { - 1 } \sum _ { t = 1 } ^ { M _ { b } } u _ { b , t } , } \end{array}$ , and $\rho _ { b } \equiv 1$ recovers the full-response objective. Sec. 4.3 evaluates $\rho _ { \mathrm { m i n } }$

Prefix inclusion and length normalization favor earlier positions:

Lemma 1 (Implicit temporal weighting). For one response of length T, let $M \in \{ 1 , \dots , T \}$ be the sampled cutoff, with law $q ( m ) = \operatorname* { P r } ( M = m )$ and prefix reward $R _ { M } = M ^ { - 1 } \Sigma _ { t = 1 } ^ { M } u _ { t }$ . Define the temporal weights $\begin{array} { r } { w _ { t } : = \sum _ { m = t } ^ { T } q ( m ) / m } \end{array}$ . Then

$$
\mathbb { E } _ { M } [ R _ { M } ] = \sum _ { t = 1 } ^ { T } w _ { t } u _ { t } .\tag{11}
$$

The weights satisfy $w _ { 1 } \ \geq \ \cdot \cdot \ \geq \ w _ { T } \ \geq \ 0$ and $\textstyle \sum _ { t = 1 } ^ { T } w _ { t } = 1$ . Uniform weights $w _ { t } = 1 / T$ occur exactly when $M = T$ almost surely.

Appendix B.2 proves this identity and characterizes the cutoff laws. The nonlinear loss adds a Jensen gap:

Lemma 2 (Prefix Jensen gap). Let $\Delta _ { M }$ be the random-prefix margin, $\mu _ { \mathrm { p r e f } } = \mathbb { E } [ \Delta _ { M } ]$ its mean, and $\mathcal { L } _ { \mathrm { p r e f } } = \mathbb { E } [ \ell ( \Delta _ { M } ) ]$ the expected prefix loss with Jensen gap $J _ { \mathrm { p r e f } }$ . Then

$$
\begin{array} { c } { { \mathcal { L } _ { \mathrm { p r e f } } = \ell ( \mu _ { \mathrm { p r e f } } ) + J _ { \mathrm { p r e f } } , } } \\ { { 0 \leq J _ { \mathrm { p r e f } } \leq \displaystyle \frac { \beta ^ { 2 } } { 8 } \mathrm { V a r } ( \Delta _ { M } ) . } } \end{array}\tag{12}
$$

Unlike position pruning, sequence pruning can shift the mean margin. The shift $\ell ( \mu _ { \mathrm { p r e f } } ) - \ell ( \Delta _ { \mathrm { f u l l } } )$ has no fixed sign, whereas $J _ { \mathrm { p r e f } } \geq 0$ vanishes for deterministic cutoffs or $\beta = 0$ (Appendix B.3). Parameter-independent cutoffs make $\nabla _ { \pmb { \theta } } \ell ( \Delta _ { M } )$ unbiased for $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { p r e f } }$ , generally different from the full-response gradient. Sec. 4.4 evaluates the resulting performance.

Assume causal execution with the prompt retained, so prefix token scores and gradients match full execution. For truncation fraction $h _ { b } = 1$ $M _ { b } / T _ { b }$ , Eq. (10) gives $\mathbb { E } [ h _ { b } ] \leq ( 1 - \rho _ { \operatorname* { m i n } } ) / 2$ . Prefix and full mean rewards differ by at most $2 U h _ { b }$ and their gradients by $2 G h _ { b }$ . The $\beta .$ -Lipschitz loss therefore yields

$$
| F _ { P } ( \pmb \theta ) - F ( \pmb \theta ) | \leq 2 \beta U ( 1 - \rho _ { \mathrm { m i n } } ) ,\tag{13}
$$

where $F _ { P } = \mathbb { E } [ \mathcal { L } _ { \mathrm { p r e f } } ]$ . Consequently, $F ( \pmb { \theta } _ { P } ^ { * } ) \textrm { -- }$ $F ( \pmb \theta ^ { * } ) \leq 4 \beta U ( 1 - \rho _ { \operatorname* { m i n } } )$ . Unlike Eq. (7), this bound has no response-length decay. At the tested $\rho _ { \mathrm { m i n } } = 1 / 2$ , the optimality bound equals the full loss range over $| \Delta | \leq 2 U$ , making it vacuous. This comparison concerns theoretical bounds, not measured deviations. Appendix B.8 also derives gradient and displacement bounds.

## 3.4 Stochastic Branch Pruning

Branch pruning removes one backward graph while preserving the expected gradient conditional on the sampled positions and prefixes. With $\lambda \ =$ $\beta \sigma ( - \beta \Delta )$ and $\sigma ( z ) = ( 1 + e ^ { - z } ) ^ { - 1 }$ , the DPO gradient decomposes as (Rafailov et al., 2023)

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { D P O } } = - \lambda \nabla _ { \boldsymbol { \theta } } R _ { w } + \lambda \nabla _ { \boldsymbol { \theta } } R _ { l } .\tag{14}
$$

Independently of positions and prefixes, draw the active backward-branch set and its probability, $( \mathbb { B } , q ^ { \star } ) \sim \operatorname { B r a n c h } ( q _ { w } )$ , with $q _ { l } = 1 - q _ { w }$ and

$$
\operatorname* { P r } \left[ ( \mathbb { B } , q ^ { \star } ) = ( \{ b \} , q _ { b } ) \right] = q _ { b } , b \in \{ w , l \} .\tag{15}
$$

Only $b ^ { \star } \in \mathbb { B }$ retains a policy backward graph. The opposite side runs under $\mathtt { n o \_ g r a d }$ , providing the other reward needed for λ. Disabling branch pruning sets $\mathbb { B } = \{ w , l \}$ and $q ^ { \star } = 1$ . Inverseprobability weighting corrects each selected contribution. With $\zeta ( z ) = \log ( 1 + e ^ { z } )$ , use

$$
\begin{array} { c } { { \displaystyle \widehat { \mathcal { L } } _ { w } = \frac { 1 } { q _ { w } } \zeta \big ( { - \beta \big ( R _ { w } - \mathrm { s g } ( R _ { l } ) \big ) } \big ) , } } \\ { { \displaystyle \widehat { \mathcal { L } } _ { l } = \frac { 1 } { q _ { l } } \zeta \big ( { - \beta \big ( \mathrm { s g } ( R _ { w } ) - R _ { l } \big ) } \big ) . } } \end{array}\tag{16}
$$

The sampled loss $\widehat { \mathcal { L } } = \widehat { \mathcal { L } } _ { b }$ ⋆ preserves the numerical margin and differentiates only the selected reward.

Proposition 2 (Conditional unbiasedness). Fix the sampled response positions and prefixes. Then $\mathbb { E } _ { b ^ { \star } } \big [ \nabla _ { \theta } \widehat { \mathcal { L } } \big ] = \nabla _ { \theta } \log \big ( 1 + e ^ { - \beta ( R _ { w } - R _ { l } ) } \big )$ . With position and sequence pruning disabled, this is the full-response per-pair gradient in Eq. (2).

We default to $q _ { w } = q _ { l } = 0 . 5$ and sweep $q _ { w } \in$ $\{ 0 . 3 , 0 . 5 , 0 . 7 \}$ in Sec. 4.3. Gradient norms govern importance-sampling variance (Katharopoulos and Fleuret, 2018), as specialized in Appendix B.4.

Proposition 3 (Branch estimator variance). Fix positions and prefixes, and set $\begin{array} { r } { \pmb { c } _ { w } = - \lambda \nabla _ { \pmb { \theta } } R _ { w } , } \end{array}$ $\begin{array} { r } { \mathbf { c } _ { l } ~ = ~ \lambda \nabla _ { \theta } R _ { l } } \end{array}$ . The estimator $\hat { \textbf { \textit { g } } } = \pmb { c } _ { b ^ { \star } } / q _ { b ^ { \star } }$ ⋆ has mean $\boldsymbol c _ { w } + \boldsymbol c _ { l }$ and variance

$$
\mathrm { V a r } ( \hat { \pmb g } ) = \frac { \| \pmb c _ { w } \| ^ { 2 } } { q _ { w } } + \frac { \| \pmb c _ { l } \| ^ { 2 } } { q _ { l } } - \| \pmb c _ { w } + \pmb c _ { l } \| ^ { 2 } .\tag{17}
$$

At $q _ { w } = q _ { l } = 1 / 2$ , the variance is $\| \pmb { c } _ { w } - \pmb { c } _ { l } \| ^ { 2 } \le$ $4 \beta ^ { 2 } G ^ { 2 }$ under the bounds of Sec. 3.2. Averaging m estimators independent in preference pairs, positions, prefixes, and branches divides the variance by m. Re-drawing only the branch reduces only its variance component.

Branch pruning adds gradient noise without objective bias. For the optimized objective $\Phi _ { \pm }$ assume µ-strong convexity and L-smoothness on a closed convex region Θ with minimizer $\theta _ { \Phi } ^ { * }$ in its interior. Starting from $\pmb { \theta } _ { 0 } \in \Theta$ , projected SGD with m independent estimators per step and the schedule in Appendix B.9 gives $\mathbb { E } \| \pmb { \theta } _ { N } - \pmb { \theta } _ { \Phi } ^ { * } \| ^ { 2 } = \mathcal { O } ( 1 / N )$ That appendix supplies the constants and a highprobability bound. Gradient unbiasedness does not imply unbiased finite-step parameters.

## 3.5 Composition

Alg. 1 composes the interventions in execution order: truncate the response, select scoring positions, then select the backward branch and apply its probability correction. Each mode is optional. Full disables all three, using full responses, all response scores, and both backward branches. Tables abbreviate the modes as Position, Sequence, and Branch, with $" + "$ denoting composition.

Algorithm 1 Stochastic DPO step   
Require: Pair $( \pmb { x } , \pmb { y } _ { w } , \pmb { y } _ { l } )$ , policy π<sub>θ</sub>, frozen $\pi _ { \mathrm { r e f } }$   
Require: Rates α, ρ<sub>min</sub>, q<sub>w</sub>, enabled pruning modes   
1: for $b \in \{ w , l \}$ do   
2: $( \tilde { \mathbf { y } } _ { b } , \mathbf { \dot { M } } _ { b } ) ^ { \circ } \gets ( \mathbf { y } _ { b } , T _ { b } )$   
3: if sequence pruning then   
4: $( \mathbf { \hat { y } } _ { b } , M _ { b } \mathbf { \hat { ) } } \sim \operatorname { P r e f i x } ( \pmb { y } _ { b } , \rho _ { \operatorname* { m i n } } )$ ▷ Eq. (10)   
5: $\mathbb { S } _ { b } \gets \mathbb { I } _ { M _ { b } }$   
6: if position pruning then   
7: $\mathbb { S } _ { b } \sim \tilde { \mathrm { S e l e c t } } ( \bar { M } _ { b } , \alpha )$ ▷ Eq. (3)   
8: $\mathbb { B } \gets \{ w , l \}$   
9: $q ^ { \star }  \mathrm { \hat { 1 } }$   
10: if branch pruning then   
11: (B, q<sup>⋆</sup>) ∼ Branch(q<sub>w</sub>) ▷ Eq. (15)   
12: for $b \in \{ w , l \}$ do   
13: z<sub>b</sub> ← LogProbs(π<sub>θ</sub>, x, y˜<sub>b</sub>, S<sub>b</sub>, b ∈ B)   
14: $z _ { b } ^ { \mathrm { r e f } } \gets \mathrm { L o g P r o b s } ( \pi _ { \mathrm { r e f } } , \pmb { x } , \tilde { \pmb { y } } _ { b } , \mathbb { S } _ { b } , \pounds \mathrm { a l } s \in )$   
15: R<sub>b</sub> ← mean $( z _ { b } - z _ { b } ^ { \mathrm { r e f } } )$ ▷ Eq. (4)   
16: $\mathcal { L } \gets \ell ( R _ { w } - R _ { l } ) / q ^ { \star }$ ▷ Eq. (16)   
17: L. backward()   
18: Update θ

For a policy or reference model $\pi ,$ the scored log-probability vector is

$$
\begin{array} { r } { \mathrm { L o g P r o b s } ( \pi , \pmb { x } , \pmb { y } , \mathbb { S } , g ) = \qquad } \\ { \left( \log \pi ( y _ { t } \mid \pmb { x } , \pmb { y } _ { < t } ) \right) _ { t \in \mathbb { S } _ { - t } ^ { \pmb { \mathscr { s } } } } . } \end{array}\tag{18}
$$

Both models score the same prefixes and positions, with entries in response order. Position pruning applies the head only at S; otherwise, the full head is evaluated before gathering. The argument g controls gradient recording through the backbone and head: only policy branches in B retain a backward graph, while the reference always runs without gradients. Numerical values are unchanged. The outputs $z _ { b } , z _ { b } ^ { \mathrm { r e f } }$ give $R _ { b }$ by the mean difference in Eq. (4). The step loss is $\mathcal { L } \ = \ \ell ( R _ { w } - R _ { l } ) / q ^ { \star }$ (Eq. (16)), recovering the per-pair loss in Eq. (2) on the selected prefixes and positions when $\boldsymbol q ^ { \star } = 1$

The minimizer $\pmb { \theta } _ { A P } ^ { * }$ of the composed objective $F _ { A P }$ satisfies

$$
\begin{array} { r l } & { F ( \pmb { \theta } _ { A P } ^ { * } ) - F ( \pmb { \theta } ^ { * } ) \leq \displaystyle \frac { \beta ^ { 2 } \bar { v } ( 1 - \alpha ) } { 4 \alpha ( M _ { 0 } - 1 ) } } \\ & { ~ + ~ 4 \beta U ( 1 - \rho _ { \operatorname* { m i n } } ) , } \end{array}\tag{19}
$$

where $M _ { 0 } = \operatorname* { m a x } ( 2 , \lceil \rho _ { \mathrm { m i n } } T _ { 0 } \rceil )$ The positionpruning bias bound decays as $1 / T _ { 0 }$ at fixed retention. The prefix bound scales with $1 - \rho _ { \mathrm { m i n } }$ without length decay, while branch pruning adds no objective bias and has the $\mathcal { O } ( 1 / N )$ mean-squared optimization error above. Appendix B.10 proves the composed loss bound and Eqs. (43) and (45), which allocate a total tolerance across α, $\rho _ { \mathrm { m i n } }$ , and N.

## 4 Experiments

## 4.1 Experimental Setup

We use Qwen3-4B, Qwen3-8B (Yang et al., 2025a), and Mistral-7B-Instruct-v0.3 (Jiang et al., 2023). The core resource profiles, eight-configuration factorial, and parameter sweeps use Qwen3-4B on HelpSteer3-Preference (Wang et al., 2025). LoRA (Hu et al., 2021) has rank 64, scaling 128, and dropout 0.05. Training uses FlashAttention-2 (Dao, 2023), gradient checkpointing (Chen et al., 2016), AdamW (Loshchilov and Hutter, 2019), DPO $\beta =$ 1. Global batch is 64 with per-device batch 2, accumulation 4, and eight Ascend 910B accelerators with 32 GiB each. Optimizer and corpus details are in Appendix D.2; matched token-selection controls are in Tabs. 11 and 12. Defaults are $\alpha = 0 . 6 $ $\rho _ { b } \sim \mathcal { U } ( 0 . 5 , 1 )$ , and $q _ { w } = q _ { l } = 0 . 5$ . The factorial and sweeps use training and selection seed 42. A second Qwen seed (43) and Mistral seed 42 test the Full–Branch comparison.

Metrics. We group benchmarks by evaluation protocol. Accuracy-based benchmarks comprise ARC-Challenge (Clark et al., 2018), GSM8K (Cobbe et al., 2021), TruthfulQA (Lin et al., 2022), BBH (Suzgun et al., 2023), MATH-Hard (Hendrycks et al., 2021), MMLU-Pro (Wang et al., 2024), and MuSR (Sprague et al., 2024). For openended generation, MiniMax-M3 (Lai et al., 2026b) judges MT-Bench (MT, 0–10) (Zheng et al., 2023),

Arena-Hard (AH, tie-adjusted win rate in %) (Li et al., 2024), and WildBench (WB, 1–10) (Lin et al., 2024). Mean Accuracy averages the seven accuracy-based benchmark percentages. Mean Generation Score equally averages 10 MT, AH, and 10 WB on a 100-point scale. Task-level means are retained in Tabs. 9, 10 and 13.

## 4.2 Resource Effects and Attribution

Baselines and measurement. Full denotes the unpruned DPO baseline using the original padded stack. Full-concat batches chosen and rejected responses together to reduce padding; both retain all response supervision and both backward graphs. Position, Branch, and Position+Branch profiles use Full-concat; Sequence and end-to-end trainings use Full. The concatenated profiler measures Full-concat at 17.343 s and 23,627 MB, vs. Full at 18.204 s and 18,571 MB: 1.049× faster with 27% more memory. The separate prefix profiler measures Full at 18.826 s. Timings are compared within a profiler. The concatenated profile uses 16 paired measured rounds. Peak high-bandwidth memory (HBM) is reset per step and maximized over 8 ranks.

Output head versus supervision. Position combines selected-head execution with random position pruning. In the separate attribution profile (Tab. 1), retaining all supervision already saves 8,905 MB; sampling adds only 642 MB. Thus selected-head execution accounts for 93.3% of the saving, vs. 6.7% from supervision deletion. Fullretention rewards match Full-concat exactly, and selected-head and dense-reduced rewards match under identical masks (Appendix D.7). The implementation supplies most memory savings, while deletion introduces the loss gap of Proposition 1.

<table><tr><td>Configuration</td><td>Retention</td><td></td><td>Step (s) HBM (MB)</td></tr><tr><td>Full-concat</td><td>1.0</td><td>17.081</td><td>23,627</td></tr><tr><td>Selected head</td><td>1.0</td><td>16.827</td><td>14,722</td></tr><tr><td>Selected head</td><td>0.6</td><td>16.791</td><td>14,080</td></tr></table>

Table 1: Qwen output-head attribution in a separate matched profile. Retention 1.0 preserves all response supervision. Shading highlights full-supervision selectedhead execution. Details: Appendix D.7.

Backward computation. Against Full-concat, Branch reduces step time by 29.9% (1.426× speedup) and peak HBM by 26.7%. Position+Branch reaches 1.443× speedup and 48.6% lower HBM (Fig. 2). Its 11,471 MB saving is below the 15,855 MB sum of the separate savings; the cause of this sub-additivity remains unresolved.

![](images/2b667979d6bc2f444e4b49a68e90d8c325acefad743a681d4af3d966b5138fa5.jpg)  
Figure 2: Resource ratios at the default settings. Dashed lines mark each baseline. Panel (c) counts executed input tokens, including prompt and padding.

Sequence execution. Sequence pruning executes 0.85–0.94 of Full’s input tokens and saves 8–11% peak HBM (Tab. 2). Across random and fixed cutoffs, steps take 18.430–18.482 s, at most 2.1% below Full. Differences between cutoffs are smaller than the 0.18–0.44 s spread across repeated measurements of one configuration. Token reduction therefore has a much smaller timing effect than branch pruning at this geometry; its physical cause remains unresolved.

<table><tr><td>Sequence setting</td><td>Tokens HBM (MB)</td><td></td><td>∆HBM Step (s)</td><td></td></tr><tr><td>Full</td><td>1.000×</td><td>18,571</td><td></td><td>18.826</td></tr><tr><td>U(0.25, 1)</td><td>0.854×</td><td>16,718</td><td>-10.0%</td><td>18.480</td></tr><tr><td>U(0.5, 1)</td><td>0.893×</td><td>16,827</td><td>-9.4%</td><td>18.430</td></tr><tr><td>U(0.75, 1)</td><td>0.943×</td><td>17,072</td><td>-8.1%</td><td>18.482</td></tr><tr><td>fixed 0.75</td><td>0.894×</td><td>16,561</td><td>-10.8%</td><td>18.471</td></tr></table>

Table 2: Sequence profile against Full. Tokens counts executed input, including prompt and padding, relative to Full. Shading marks the default U(0.5, 1).

## 4.3 Parameter Sensitivity

Tab. 3 varies one axis at a time using the factorial recipe. Realized pruning fractions are reported in Appendix D.4. Sequence input counts include prompt and padding, so fractions are not comparable across axes or directly to $\rho _ { b }$

For Position, increasing α from 0.3 to 0.9 raises mean accuracy from 57.90 to 58.32, while peak HBM changes little (11,211–11,580 MB) and AH is non-monotone (Tab. 10). This weak memory dependence agrees with the implementationdominated attribution above. Sequence cutoffs track input work and HBM but barely affect step time. Fixed $\rho _ { b } = 0 . 7 5$ and U(0.5, 1) execute similar token fractions yet differ in temporal weights and randomness, so their score difference cannot isolate the Jensen gap. Branch probabilities yield similar training times (127.7–128.4 min) and varying open-ended point estimates. We retain $q _ { w } \ = \ 0 . 5$ HBM was not measured across $q _ { w }$ Position step times are 3.29–3.41 s in its separate paired, interleaved retention profiler; resource comparisons remain within each axis.

<table><tr><td>Axis</td><td>Setting</td><td>Resource accuracy (%)</td><td>Mean</td><td>Mean generation score</td></tr><tr><td>Position</td><td>α=0.30</td><td>11,211 MB</td><td>57.90</td><td>47.73</td></tr><tr><td>Position</td><td>α=0.60</td><td>11,320 MB</td><td>58.06</td><td>48.38</td></tr><tr><td>Position</td><td>α=0.90</td><td>11,580 MB</td><td>58.32</td><td>48.63</td></tr><tr><td></td><td>Sequence U(0.25, 1)</td><td>16,718 MB</td><td>58.09</td><td>47.91</td></tr><tr><td></td><td>Sequence U(0.50, 1)1</td><td>16,827 MB</td><td>58.04</td><td>47.33</td></tr><tr><td></td><td>Sequence U(0.75,1)</td><td>17,072 MB</td><td>58.04</td><td>49.53</td></tr><tr><td></td><td>Sequence fixed 0.75</td><td>16,561 MB</td><td>58.08</td><td>47.03</td></tr><tr><td>Branch</td><td> $q _ { w } { = } 0 . 3 0$ </td><td>127.7 min</td><td>58.06</td><td>48.80</td></tr><tr><td>Branch</td><td> $q _ { w } { = } 0 . 5 0$ </td><td>128.4 min</td><td>58.08</td><td>50.25</td></tr><tr><td>Branch</td><td> $q _ { w } { = } 0 . 7 0$ </td><td>128.4 min</td><td>57.97</td><td>48.74</td></tr></table>

Table 3: Ablation on Qwen3-4B/HelpSteer3-Preference. Shading marks defaults. Means follow Sec. 4.1. Full task scores: Tab. 10.

## 4.4 The Eight-Configuration Factorial

In Tab. 4, all four Branch configurations finish in 126–128 min vs. 185–188 min without Branch, preserving a 31–32% time saving across matched contrasts. Position and Sequence retain that benefit when composed with Branch. All 12 single-axis contrasts lower mean accuracy (unrounded contrasts in Appendix D.4); Fig. 3 shows the resulting time–accuracy tradeoff.

Open-ended scores move differently. Branch raises MT and WB point estimates with AH close to Full. Configurations containing Position or Sequence fall below Full on AH, while their MT changes depend on the base configuration. Generation means are descriptive single-seed summaries and do not establish score superiority.

<table><tr><td>Method</td><td>Wall (min)</td><td>Mean accuracy (%)</td><td>Mean generation score</td></tr><tr><td>Full</td><td>187.8</td><td>58.33</td><td>49.33</td></tr><tr><td>Position</td><td>185.9</td><td>58.06</td><td>48.38</td></tr><tr><td>Sequence</td><td>186.4</td><td>58.04</td><td>47.33</td></tr><tr><td>Branch</td><td>128.4</td><td>58.08</td><td>50.25</td></tr><tr><td>Position+Sequence</td><td>185.0</td><td>57.97</td><td>48.85</td></tr><tr><td>Position+Branch</td><td>127.8</td><td>57.94</td><td>48.17</td></tr><tr><td>Sequence+Branch</td><td>126.3</td><td>57.86</td><td>47.67</td></tr><tr><td>Position+Sequence +Branch</td><td>127.0</td><td>57.69</td><td>48.35</td></tr></table>

Table 4: Qwen3-4B factorial, one training and one selection seed. Bold/underline mark best/second-best values, including ties. Full scores: Tab. 9.

![](images/9633c66cd0880b23c1c631a0365616da926401619b430f24feef4ca44895d017.jpg)  
Figure 3: Training time versus seven-task mean accuracy for the eight factorial runs. The gray staircase is the observed frontier of single-seed point estimates.

## 4.5 Branch Pruning Across Seeds and Models

The matched Full–Branch trainings in Tab. 5 save 29.9–32.7% wall-clock across two Qwen seeds and Mistral, with mean accuracy lower by 0.25– 0.45 pp. In the Mistral profile in Tab. 6, Branch gives 1.507× speedup and saves 1,347 MB (6.4%) against Full. The time benefit transfers across families.

MT and WB checkpoint means rise in all three comparisons, while Mistral AH falls. These checkpoint scores are descriptive and do not establish superiority or characterize training-seed variability. Full task scores and scoring coverage are in Appendix D.7. Cross-family validation covers Branch only; Position, Sequence, and their compositions are evaluated on Qwen.

Branch-gradient consistency. We check the conditional expectation in Proposition 2 by enumerating both backward branches. On a small Mistral model in Float64, the probability-weighted average of both branch gradients matches Full to relative L2 error at most $1 . 7 \times 1 0 ^ { - 7 }$ . On Mistral-7B in BFloat16, the gradient cosine is 0.9997 and the relative L2 error is 0.0236, against 0.0071 for repeated computation of the same configuration, which sets the numerical noise floor for this check. These checks disable LoRA dropout and retain gradient checkpointing using dropout 0.05.

<table><tr><td>Config. Wall (min)</td><td>Speedup</td><td>(%)</td><td>Mean acc. Mean gen. score</td></tr><tr><td>Qwen3-4B, seed 42</td><td></td><td></td><td></td></tr><tr><td>Full 187.8</td><td>1.000×</td><td>58.33</td><td>49.33</td></tr><tr><td>Branch</td><td>128.4 1.463×</td><td>58.08</td><td>50.25</td></tr><tr><td>Qwen3-4B, seed 43</td><td></td><td></td><td></td></tr><tr><td>Full</td><td>187.0 1.000×</td><td>58.26</td><td>47.64</td></tr><tr><td>Branch</td><td>131.0 1.428×</td><td>57.81</td><td>48.49</td></tr><tr><td>Mistral-7B, seed 42</td><td></td><td></td><td></td></tr><tr><td>Full</td><td>211.0 1.000×</td><td>44.86</td><td>33.17</td></tr><tr><td>Branch</td><td>142.0 1.486×</td><td>44.55</td><td>33.93</td></tr></table>

Table 5: Full–Branch validation. Speedup uses each pair’s Full training time. Bold marks the better value within each pair. Score means follow Sec. 4.1; task scores are in Tab. 13.
<table><tr><td>Configuration Step (s) HBM (MB)</td><td></td><td></td></tr><tr><td>Qwen3-4B</td><td></td><td></td></tr><tr><td>Full</td><td>18.204</td><td>18,571</td></tr><tr><td>Branch</td><td>12.161</td><td>17,319</td></tr><tr><td>Mistral-7B</td><td></td><td></td></tr><tr><td>Full</td><td>22.438</td><td>21,185</td></tr><tr><td>Branch</td><td>14.888</td><td>19,838</td></tr></table>

Table 6: Paired Full–Branch step-time and peakmemory profiles. Bold marks lower values within each model.

## 5 Conclusion

We separate output scoring, sequence execution, and backward computation to locate where stochastic DPO computation saves resources and how it changes optimization. Most output-memory savings come from the implementation rather than from supervision deletion, the branch-pruning time benefit reproduces across Qwen seeds and Mistral, and both accompany lower mean accuracy with mixed open-ended results, so configuration choice should follow the resource constraint and the acceptable objective change.

## Limitations

Evaluation covers parameter-efficient text preference training with an online reference on one eightaccelerator setup. Savings may vary with hardware, batch geometry, and reference execution.

## References

Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. 2023. Pythia: A suite for analyzing large language models across training and scaling. Preprint, arXiv:2304.01373.

Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. 2016. Training deep nets with sublinear memory cost. Preprint, arXiv:1604.06174.

Wei Chen, Shigui Li, Jiacheng Li, Junmei Yang, John Paisley, and Delu Zeng. 2025. Dequantified diffusion-schrödinger bridge for density ratio estimation. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 8427–8452. PMLR.

Wei Chen, Yubing Wu, Junmei Yang, Delu Zeng, Qibin Zhao, John Paisley, Min Chen, and Zhou Wang. 2026. Towards disentangled preference optimization dynamics: Suppress the loser, preserve the winner. In International Conference on Machine Learning. ArXiv:2604.18239.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2024. Ultrafeedback: Boosting language models with scaled ai feedback. Preprint, arXiv:2310.01377.

Tri Dao. 2023. FlashAttention-2: Faster attention with better parallelism and work partitioning. Preprint, arXiv:2307.08691.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems, volume 35.

Shian Du, Yihong Luo, Wei Chen, Jian Xu, and Delu Zeng. 2022. TO-FLOW: Efficient continuous normalizing flows with temporal optimization adjoint with moving speed. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12570–12580.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. Preprint, arXiv:2103.03874.

Rei Higuchi and Taiji Suzuki. 2025. Direct density ratio optimization: A statistically consistent approach to aligning large language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 23219–23234. PMLR.

Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, Qingquan Song, Shao Tang, Siyu Zhu, Steven Shimizu, Shivam Sahni, Haowen Ning, and Yanning Chen. 2025. Liger kernel: Efficient triton kernels for llm training. Preprint, arXiv:2410.10989.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. LoRA: Low-rank adaptation of large language models. Preprint, arXiv:2106.09685.

Chengyu Huang, Zhuohang Li, Sheng-Yen Chou, and Claire Cardie. 2026. Token-weighted direct preference optimization with attention. Preprint, arXiv:2605.21883.

Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q. Weinberger. 2016. Deep networks with stochastic depth. In Computer Vision – ECCV 2016, volume 9908 of Lecture Notes in Computer Science, pages 646–661. Springer.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7B. Preprint, arXiv:2310.06825.

Yuxin Jiang, Bo Huang, Yufei Wang, Xingshan Zeng, Liangyou Li, Yasheng Wang, Xin Jiang, Lifeng Shang, Ruiming Tang, and Wei Wang. 2025. Bridging and modeling correlations in pairwise data for direct preference optimization. In International Conference on Learning Representations, volume 2025, pages 85386–85405.

Angelos Katharopoulos and Francois Fleuret. 2018. Not all samples are created equal: Deep learning with importance sampling. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 2525–2534. PMLR.

Shuzhong Lai, Junhong Lai, Chenxi Li, Qing Zhou, Haifeng Li, Gang Pan, Lin Yao, and Yueming Wang. 2026a. Td-dpo: Difference-aware preference optimization for mitigating sycophancy in clinical autism intervention dialogue. Preprint, arXiv:2607.18304.

Xunhao Lai, Weiqi Xu, Yufeng Yang, Qiaorui Chen, Yang Xu, Lunbin Zeng, Xiaolong Li, Haohai Sun, Haichao Zhu, Vito Zhang, Jinkai Hu, Jiayao Li, Rui Gao, Zekun Li, Songquan Zhu, Jingkai Zhou, and Pengyu Zhao. 2026b. Minimax sparse attention. Preprint, arXiv:2606.13392.

Meng Li, Guangda Huzhang, Haibo Zhang, Xiting Wang, and Anxiang Zeng. 2025a. Optimal transportbased token weighting scheme for enhanced preference optimization. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 21311– 21334.

Shigui Li, Wei Chen, and Delu Zeng. 2025b. EVODiff: Entropy-aware variance optimized diffusion inference. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. 2024. From crowdsourced data to highquality benchmarks: Arena-hard and benchbuilder pipeline. Preprint, arXiv:2406.11939.

Bill Yuchen Lin, Yuntian Deng, Khyathi Chandu, Faeze Brahman, Abhilasha Ravichander, Valentina Pyatkin, Nouha Dziri, Ronan Le Bras, and Yejin Choi. 2024. Wildbench: Benchmarking llms with challenging tasks from real users in the wild. Preprint, arXiv:2406.04770.

Stephanie C. Lin, Jacob Hilton, and Owain Evans. 2022. TruthfulQA: Measuring how models mimic human falsehoods. In Annual Meeting of the Association for Computational Linguistics.

Aiwei Liu, Haoping Bai, Zhiyun Lu, Yanchao Sun, Xiang Kong, Simon Wang, Jiulong Shan, Albin Madappally Jose, Xiaojiang Liu, Lijie Wen, Philip S. Yu, and Meng Cao. 2025. TIS-DPO: Token-level importance sampling for direct preference optimization with estimated weights. In International Conference on Learning Representations.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. Preprint, arXiv:1711.05101.

Haocheng Luo, Zehang Deng, Thanh-Toan Do, Mehrtash Harandi, Dinh Phung, and Trung Le. 2026. Sharpness-aware minimization in logit space efficiently enhances direct preference optimization. Preprint, arXiv:2603.18258.

Yu Meng, Mengzhou Xia, and Danqi Chen. 2024. SimPO: Simple preference optimization with a reference-free reward. In Advances in neural information processing systems.

Inoussa Mouiche. 2026. Gradient-gated DPO: Stabilizing preference optimization in language models. arXiv preprint arXiv:2605.02626.

Truong Nguyen, Tien-Phat Nguyen, Linh Ngo Van, Duy Minh Ho Nguyen, Khoa D Doan, and Trung Le. 2026. TokenRatio: Principled token-level preference optimization via ratio matching. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 92636–92658. PMLR.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems.

Noam Razin, Sadhika Malladi, Adithya Bhaskar, Danqi Chen, Sanjeev Arora, and Boris Hanin. 2025. Unintentional unalignment: Likelihood displacement in direct preference optimization. In International Conference on Learning Representations, volume 2025, pages 24791–24834.

Zayne Sprague, Xi Ye, Kaj Bostrom, Swarat Chaudhuri, and Greg Durrett. 2024. Musr: Testing the limits of chain-of-thought with multistep soft reasoning. Preprint, arXiv:2310.16049.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014. Dropout: A simple way to prevent neural networks from overfitting. Journal ofMachine Learning Research, 15(56):1929–1958.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed H Chi, Denny Zhou, et al. 2023. Challenging big-bench tasks and whether chain-of-thought can solve them. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 13003–13051.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. 2024. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Preprint, arXiv:2406.01574.

Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Hoo-Chang Shin, Felipe Soares, Alexander Bukharin, Ellie Evans, Yi Dong, and Oleksii Kuchaiev. 2025. Helpsteer3-preference: Open human-annotated preference data across diverse tasks and languages. Preprint, arXiv:2505.11475.

Erik Wijmans, Brody Huval, Alexander Hertzberg, Vladlen Koltun, and Philipp Krähenbühl. 2025. Cut your losses in large-vocabulary language models. In International Conference on Learning Representations, volume 2025, pages 68174–68193.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41

others. 2025a. Qwen3 technical report. Preprint, arXiv:2505.09388.

Kailai Yang, Zhiwei Liu, Qianqian Xie, Jimin Huang, Erxue Min, and Sophia Ananiadou. 2025b. Selective preference optimization via token-level reward function estimation. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 7032–7056, Suzhou, China. Association for Computational Linguistics.

Ning Yang, Hai Lin, Yibo Liu, Baoliang Tian, Guoqing Liu, and Haijun Zhang. 2026. Token-importance guided direct preference optimization. In International Conference on Learning Representations.

Hee Suk Yoon, Eunseop Yoon, Mark A. Hasegawa-Johnson, Sungwoong Kim, and Chang D. Yoo. 2025. ConfPO: Exploiting policy model confidence for critical token selection in preference optimization. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 72641–72655. PMLR.

Hui Yuan, Yifan Zeng, Yue Wu, Huazheng Wang, Mengdi Wang, and Liu Leqi. 2025. A common pitfall of margin-based language model alignment: Gradient entanglement. In International Conference on Learning Representations, volume 2025, pages 5278– 5297.

Yongcheng Zeng, Guoqing Liu, Weiyu Ma, Ning Yang, Haifeng Zhang, and Jun Wang. 2024. Token-level direct preference optimization. In International Conference on Machine Learning.

Wenxiao Zhao, Shu Wang, and Ying Nian Wu. 2026. Sedpo: Self-evolving token credit for direct preference optimization. Preprint, arXiv:2608.09568.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Preprint, arXiv:2306.05685.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.

## A Notation

We group the main notation below. Symbols specific to individual proofs are defined where introduced. In single-response results, the side index b is suppressed, so $T , k , R , u _ { t }$ denote the corresponding side quantities.

Preference pairs and loss. $\mathcal { D }$ is the preference dataset of prompts x, chosen responses ${ \mathbf { } } y _ { w } .$ , and rejected responses ${ \mathbf { } } ^ { \mathbf { \eta } } \mathbf { \mathbf { 3 } } \mathbf { \eta } ^ { \mathrm { ~ t ~ } }$ . The policy $\pi _ { \pmb { \theta } }$ has trainable parameters $\theta ,$ and $\pi _ { \mathrm { r e f } }$ is the fixed reference. For side $b \in \{ w , l \} , T _ { b }$ is the response length, $y _ { b , t }$ the target token at position $t ,$ and $_ { { \pmb { y } } _ { b , < t } }$ its response prefix. The token log-ratio $u _ { b , t }$ gives the mean reward $R _ { b }$ and margin $\Delta = R _ { w } - R _ { l }$ . The scale $\beta$ defines $\ell ( d ) = \log ( 1 + e ^ { - \beta d } )$ , with per-pair loss $\mathcal { L } _ { \mathrm { D P O } } = \ell ( \Delta ) \left( \mathrm { S e c . } \ 3 . 1 \right)$

Three pruning axes. Sequence pruning uses cutoff fraction $\rho _ { b }$ to retain prefix $\tilde { \mathbf { \pmb { y } } } _ { b }$ of length $M _ { b }$ , with minimum fraction $\rho _ { \mathrm { m i n } }$ . Position pruning selects $\mathbb { S } _ { b } \subseteq \mathbb { I } _ { M _ { b } } = \{ 1 , \dots , M _ { b } \}$ with count $k _ { b }$ controlled by retention α. Branch pruning samples backward branch $b ^ { \star }$ with probability $q _ { b }$ (Secs. 3.2 to 3.4).

Sampling and execution. Prefix, Select, and Branch denote the prefix, position, and branch sampling laws. The active backward branches form B, and $q ^ { \star }$ supplies the probability correction. The operator LogProbs returns scored logprobabilities, with Boolean flag g controlling gradient recording. Its policy and reference outputs are $z _ { b }$ and $\boldsymbol { z } _ { b } ^ { \mathrm { r e f } }$ , and $\mathcal { L }$ is the step loss. We write σ for sigmoid, sg for stop-gradient, and V for vocabulary size (Secs. 3.1, 3.4 and 3.5).

Analysis. E and Var refer to the indicated sampling. $\mathbb { U } _ { k }$ is a uniform size-k position subset, $\phi _ { t }$ a token log-probability gradient, and $\Delta _ { U }$ and $\Delta _ { \mathrm { f u l l } }$ the sampled and full margins (Proposition 1). The objectives $F , F _ { A } , F _ { P } , F _ { A P }$ correspond to full responses, position pruning, sequence pruning, and their composition, with minimizers $\theta ^ { * } , \theta _ { A } ^ { * } , \theta _ { P } ^ { * } , \theta _ { A P } ^ { * }$ . The analysis uses token-score and gradient bounds $U , G ,$ variance bound v¯, and length bounds $T _ { 0 } , M _ { 0 }$ (Secs. 3.2 and 3.5).

## B Proofs

This appendix collects the proofs of every stated result: Proposition 1 (Sec. B.1), Lemma 1 (Sec. B.2), Lemma 2 (Sec. B.3), Proposition 2 (Sec. B.4).

## B.1 Proof of Proposition 1

Proof of Proposition 1. Writing $\mathcal { H } \{ E \}$ for the indicator of event E, Eq. (5) follows from $\mathbb { E } [ \mathbb { H } \mathscr  \{ t \in$ $\mathbb { U } _ { k } \} ] = k / T$ and sampling without replacement. Jensen’s inequality gives the lower bound in Eq. (6). Since $0 ~ \leq ~ \ell ^ { \prime \prime } ( d ) ~ \leq ~ \beta ^ { 2 } / 4$ , a second-order upper bound around $\Delta _ { \mathrm { f u l l } }$ gives the upper bound. The variance is the finite-population variance of a without-replacement sample, summed over independent sides. □

The finite-population identities behind the proposition are as follows. Let $I _ { t }$ indicate membership in a uniform size-k subset of T positions. Then $\mathbb { E } I _ { t } =$ $k / T .$ , and for $i \neq j , \mathbb { E } [ I _ { i } I _ { j } ] = k ( k - 1 ) / [ T ( T - 1 ) ]$ These identities establish Eq. (5) and the variance in Proposition 1. For vector gradients and $T > 1$ let $\bar { \phi } \stackrel { - } { = } T ^ { - 1 } \sum _ { t } \phi _ { t }$ be their population mean and $\Sigma _ { \phi }$ their population covariance. Cov denotes covariance. Then

$$
\begin{array} { r } { \mathrm { C o v } \left[ \frac { 1 } { k } \displaystyle \sum _ { t } I _ { t } \phi _ { t } \right] = \frac { T - k } { k ( T - 1 ) } \pmb { \Sigma } _ { \phi } , } \\ { \pmb { \Sigma } _ { \phi } = \frac { 1 } { T } \displaystyle \sum _ { t } ( \phi _ { t } - \bar { \phi } ) ( \phi _ { t } - \bar { \phi } ) ^ { \top } , } \end{array}\tag{20}
$$

For $T = 1$ , the covariance is zero. These identities concern the side-reward gradient. The sampled loss gradient is $- \beta \sigma ( - \beta \Delta _ { U } ) \nabla _ { \pmb { \theta } } \Delta _ { U }$ . Its mask-dependent multiplier means that rewardgradient unbiasedness alone does not imply fullloss-gradient unbiasedness. Equation (6) quantifies the expected loss difference.

Remark (target-logit diagnostics). Let $p _ { t }$ be the current-policy target-token probability at position $t ,$ and S a selected set. The target-logit diagnostic $\mathcal { E } _ { z }$ (subscript z denotes the target logit) is $\mathcal { E } _ { z } ( \mathbb { S } ) =$ $\begin{array} { r } { \frac { 1 } { | \mathbb { S } | ^ { 2 } } \sum _ { t \in \mathbb { S } } ( 1 - p _ { t } ) ^ { 2 } } \end{array}$ on a fixed response. For the full position set $\mathbb { I } _ { T }$ , linearity gives

$$
\begin{array} { r } { \mathbb { E } [ \mathcal { E } _ { z } ( \mathbb { U } _ { k } ) ] = \frac { 1 } { k T } \sum _ { t } ( 1 - p _ { t } ) ^ { 2 } , } \\ { \mathcal { E } _ { z } ( \mathbb { I } _ { T } ) = \frac { 1 } { T ^ { 2 } } \sum _ { t } ( 1 - p _ { t } ) ^ { 2 } . } \end{array}\tag{21}
$$

Their ratio is $T / k ,$ not $k / T$ . If a diagnostic instead reports the retained numerator fraction $f ( \mathbb { S } ) =$ $\textstyle \sum _ { t \in \mathbb { S } } ( 1 - p _ { t } ) ^ { 2 } / \sum _ { t = 1 } ^ { T } ( 1 - p _ { t } ) ^ { 2 } ,$ , its normalized reward-derivative ratio is $( T / k ) ^ { 2 } f ( \mathbb { S } )$ for that response. With varying response lengths and rounded counts, conversion must occur per response before aggregation. Neither quantity measures conditional KL or downstream quality.

Remark (alternative denominator). Keeping the $1 / T _ { b }$ denominator would scale the selected mean by the realized fraction $k _ { b } / T _ { b }$ . At a common realized fraction $\gamma = k _ { b } / T _ { b }$ on both sides, this equals selected-mean normalization at temperature $\gamma \beta$ . We use the $1 / k _ { b }$ mean throughout.

## B.2 Proof of Lemma 1

ProofofLemma 1. Exchange the order of summation:

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { M } [ R _ { M } ] = \sum _ { m = 1 } ^ { T } q ( m ) \frac { 1 } { m } \sum _ { t = 1 } ^ { m } u _ { t } } \\ { \displaystyle = \sum _ { t = 1 } ^ { T } u _ { t } \sum _ { m = t } ^ { T } \frac { q ( m ) } { m } = \sum _ { t = 1 } ^ { T } w _ { t } u _ { t } . } \end{array}\tag{22}
$$

For $\begin{array} { r } { 1 \leq t < T , w _ { t } - w _ { t + 1 } = q ( t ) / t \geq 0 , } \end{array}$ and $w _ { T } ~ = ~ q ( T ) / T ~ \ge ~ 0$ . Also, $\begin{array} { r } { \sum _ { t = 1 } ^ { T } w _ { t } \ = } \end{array}$ $\textstyle \sum _ { m = 1 } ^ { T } q ( m ) \sum _ { t = 1 } ^ { m } 1 / m = 1$ . Conversely, for a nonnegative, normalized, non-increasing vector ${ \pmb w } = ( w _ { 1 } , \dots , w _ { T } )$ , set $w _ { T + 1 } : = 0$ and $q ( t ) =$ $t ( w _ { t } - w _ { t + 1 } )$ . Then $q ( t ) \geq 0$ and $\begin{array} { r } { \sum _ { t = 1 } ^ { T } q ( t ) = } \end{array}$ $\begin{array} { r } { \sum _ { t = 1 } ^ { T } t ( w _ { t } - w _ { t + 1 } ) = \sum _ { t = 1 } ^ { T } w _ { t } = 1 } \end{array}$ . Substitution recovers $\begin{array} { r } { w _ { t } = \sum _ { m = t } ^ { T } q ( m ) / m } \end{array}$ . Uniform weights $w _ { t } = 1 / T$ give $q ( t ) = 0$ for $t < T$ and $q ( T ) = 1$ so $M = T$ almost surely, recovering the fixedcutoff case $\rho _ { b } \equiv 1$ □

## B.3 Proof of Lemma 2

ProofofLemma 2. $\ell ( d ) ~ = ~ \log ( 1 + e ^ { - \beta d } )$ satisfies $\begin{array} { r l r } { \ell ^ { \prime } ( d ) } & { { } = } & { - \beta \sigma ( - \beta d ) \mathrm { a n d } \ell ^ { \prime \prime } ( d ) } \end{array}$ $\begin{array} { r l } { \ell ^ { \prime \prime } ( d ) } & { { } = } \end{array}$ $\beta ^ { 2 } \sigma ( \beta d ) \sigma ( - \beta d )$ . For $\beta > 0 , 0 < \ell ^ { \prime \prime } \leq \beta ^ { 2 } / 4$ and $\ell ^ { \prime } < 0$ , so ℓ is strictly convex and strictly decreasing. For $\beta = 0 , \ell \equiv \log 2$ with $\ell ^ { \prime } = \ell ^ { \prime \prime } = 0$ Assume $\beta > 0$ unless a degenerate case is named. Jensen’s inequality gives $\mathbb { E } [ \ell ( \Delta _ { M } ) ] \geq \ell ( \mu _ { \mathrm { p r e f } } )$ hence $J _ { \mathrm { p r e f } } : = \mathcal { L } _ { \mathrm { p r e f } } - \ell ( \mu _ { \mathrm { p r e f } } ) \geq 0 .$ For the upper bound, expand ℓ to second order around $\mu _ { \mathrm { p r e f } } { \mathrm { : } }$ for each scalar margin x, $, \ell ( x ) = \ell ( \mu _ { \mathrm { p r e f } } ) +$ $\begin{array} { r } { \ell ^ { \prime } ( \mu _ { \mathrm { p r e f } } ) ( x - \mu _ { \mathrm { p r e f } } ) + \frac { 1 } { 2 } \ell ^ { \prime \prime } ( \xi ) ( x - \mu _ { \mathrm { p r e f } } ) ^ { 2 } } \end{array}$ for some ξ between x and $\mu _ { \mathrm { p r e f } } .$ . Taking the expectation removes the linear term, and $\ell ^ { \prime \prime } \leq \beta ^ { 2 } / 4$ gives

$$
\begin{array} { r } { J _ { \mathrm { p r e f } } = \frac { 1 } { 2 } \mathbb { E } [ \ell ^ { \prime \prime } ( \xi ) ( \Delta _ { M } - \mu _ { \mathrm { p r e f } } ) ^ { 2 } ] \leq \frac { \beta ^ { 2 } } { 8 } \mathrm { V a r } ( \Delta _ { M } ) . } \end{array}\tag{23}
$$

Subtracting the full-response loss from Eq. (12) gives

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { p r e f } } - \ell ( \Delta _ { \mathrm { f u l l } } ) = \ell ( \mu _ { \mathrm { p r e f } } ) - \ell ( \Delta _ { \mathrm { f u l l } } ) } & { { } } \\ { + J _ { \mathrm { p r e f } } . } \end{array}\tag{24}
$$

Because ℓ is strictly decreasing, $\ell ( \mu _ { \mathrm { p r e f } } ) ~ -$ $\ell ( \Delta _ { \mathrm { f u l l } } )$ is positive precisely when the temporally reweighted mean margin $\mu _ { \mathrm { p r e f } }$ is below the fullresponse margin, which is a data- and distributiondependent event and therefore carries no fixed sign. Equation (11) gives $\begin{array} { r } { \mu _ { \mathrm { p r e f } } = \sum _ { t } w _ { t } ^ { ( w ) } u _ { w , t } - } \end{array}$ $\Sigma _ { t } w _ { t } ^ { ( l ) } u _ { l , t }$ , with the side-specific weights $w _ { t } ^ { ( b ) }$

Two degeneracies: a deterministic cutoff makes $\Delta _ { M }$ constant, so $\mathrm { V a r } ( \Delta _ { M } ) = 0$ and $J _ { \mathrm { p r e f } } = 0$ . At $\beta = 0$ both terms of Eq. (24) vanish.

Finally, the cutoff variables $M _ { w } , M _ { l }$ are drawn independently of θ and have finite support, so the expectation in $\mathcal { L } _ { \mathrm { p r e f } } ( \pmb { \theta } ) = \mathbb { E } [ \ell ( \Delta _ { M } ( \pmb { \theta } ) ) ]$ is a finite sum and differentiation passes through it exactly:

$$
\nabla _ { \pmb { \theta } } \mathcal { L } _ { \mathrm { p r e f } } = \mathbb { E } \left[ \nabla _ { \pmb { \theta } } \ell ( \Delta _ { M } ) \right] .\tag{25}
$$

Thus $\nabla _ { \pmb { \theta } } \ell ( \Delta _ { M } )$ is an unbiased stochastic gradient of ${ \mathcal { L } } _ { \mathrm { p r e f } }$ and need not coincide in expectation with the full-response gradient. The temporal shift and Jensen gap in Eq. (24) can cancel at a fixed policy. Equality of loss values at one point does not establish equality of gradients. □

## B.4 Proof of Proposition 2

Proof of Proposition 2. Write $\mathcal { L } _ { w } ^ { \mathrm { s g } } = \zeta ( - \beta ( R _ { w } -$ $\mathrm { s g } ( R _ { l } ) )$ . Because the stop-gradient blocks the flow into $R _ { l }$ , and because $\lambda = \beta \sigma ( - \beta ( R _ { w } - R _ { l } ) )$ is evaluated at the current parameters as a chainrule coefficient (it is differentiated no further),

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathcal { L } _ { w } ^ { \mathrm { s g } } = \sigma \big ( - \beta ( R _ { w } - R _ { l } ) \big ) ( - \beta ) \nabla _ { \theta } R _ { w } } \\ & { \qquad = - \lambda \nabla _ { \theta } R _ { w } , } \end{array}\tag{26}
$$

and symmetrically $\nabla _ { \pmb { \theta } } \mathcal { L } _ { l } ^ { \mathrm { s g } } \ = \ + \lambda \nabla _ { \pmb { \theta } } R _ { l }$ With the $1 / q _ { b }$ importance factors, $\mathbb { E } _ { b ^ { \star } } [ \nabla \widehat { \mathcal { L } } ] \ = \ q _ { w } \ .$ $( - \lambda / q _ { w } ) \nabla R _ { w } + q _ { l } \cdot ( \lambda / q _ { l } ) \nabla R _ { l } \ = \ - \lambda \nabla R _ { w } +$ $\lambda \nabla R \boldsymbol { l } = \nabla _ { \pmb { \theta } } \log ( 1 + e ^ { - \beta ( R _ { w } - R _ { l } ) } )$ . All quantities are those of the sampled response positions and prefixes: with sequence pruning active, $R _ { b }$ is the prefix mean over $M _ { b }$ . With position pruning active, it is the masked mean over $\mathbb { S } _ { b } ,$ , and the gradients flow only through the selected positions. When position and sequence pruning are disabled, the sampled computation is the full-response objective and the claim recovers the length-normalized loss in Eq. (2). □

Remark (branch-estimator variance). With ∥ · ∥ denoting the Euclidean norm, for fixed sampled rewards the single-branch estimator’s second moment is $\lambda ^ { 2 } \big ( \| \bar { \nabla } R _ { w } \| ^ { 2 } / q _ { w } + \| \nabla R _ { l } \| ^ { 2 } / q _ { l } \big )$ Its variance-optimal probability therefore balances the branch gradient norms, the usual importancesampling criterion (Katharopoulos and Fleuret, 2018). We use $q _ { w } = 0 . 5$ as a symmetric default. The $q _ { w } \in \{ 0 . 3 , 0 . 5 , 0 . 7 \}$ sweep (Tab. 3) has similar wall-clock times and differing quality point estimates.

## B.5 Proof of Corollary 1 and the retention rule

ProofofCorollary 1. Since ℓ is convex, Jensen’s inequality gives $\begin{array} { r c l } { \mathbb { E } _ { S } [ \ell ( \Delta _ { U } ) ] } & { \ge } & { \ell ( \mathbb { E } _ { S } [ \Delta _ { U } ] ) } \end{array} =$ $\ell ( \Delta _ { \mathrm { f u l l } } )$ at every θ, so $F _ { A } \geq F$ pointwise. Hence

$$
\begin{array} { l } { { \displaystyle F ( \pmb { \theta } _ { A } ^ { * } ) \leq F _ { A } ( \pmb { \theta } _ { A } ^ { * } ) \leq F _ { A } ( \pmb { \theta } ^ { * } ) } } \\ { { \displaystyle \qquad \leq F ( \pmb { \theta } ^ { * } ) + \frac { \beta ^ { 2 } } { 8 } \mathbb { E } _ { \mathcal { D } } \big [ \mathrm { V a r } _ { S } ( \Delta _ { U } ) \big ] , } } \end{array}\tag{27}
$$

where the first step uses $F \leq F _ { A }$ , the second the optimality of $\pmb { \theta } _ { A } ^ { * }$ for $F _ { A } .$ , and the third $\operatorname { E q . } \left( 6 \right)$ evaluated at $\pmb { \theta } ^ { * }$ . Subtracting $F ( \pmb \theta ^ { * } )$ gives Eq. (7).

For Eq. $( 8 ) , k _ { b } \ge \alpha T _ { b }$ implies $1 / k _ { b } \le 1 / ( \alpha T _ { b } )$ and $\begin{array} { r } { T _ { b } - k _ { b } \le ( 1 - \alpha ) T _ { b } . } \end{array}$ , while $v _ { b } \leq \bar { v }$ and $T _ { b } \geq T _ { 0 }$ give

$$
\begin{array} { r l r } {  { \operatorname { V a r } _ { S } ( \Delta _ { U } ) = \sum _ { b : T _ { b } > 1 } \frac { T _ { b } - k _ { b } } { k _ { b } ( T _ { b } - 1 ) } v _ { b } } } \\ & { } & { \leq \sum _ { b } \frac { ( 1 - \alpha ) T _ { b } } { \alpha T _ { b } ( T _ { 0 } - 1 ) } \bar { v } = 2 \bar { v } r _ { A } . } \end{array}\tag{28}
$$

The display is per response, so averaging over the dataset gives $\mathbb { E } _ { \mathcal { D } } [ \mathrm { V a r } _ { S } ( \Delta _ { U } ) ] \le 2 \bar { v } r _ { A }$ . Requiring $\begin{array} { r } { \frac { \beta ^ { 2 } } { 8 } \mathbb { E } _ { \mathcal { D } } [ \mathrm { V a r } _ { S } ( \Delta _ { U } ) ] \leq \frac { \beta ^ { 2 } \bar { v } } { 4 } r _ { A } \leq \varepsilon _ { F } } \end{array}$ and solving for α yields Eq. (9). □

## B.6 Proof of Eq. (31) and the stationary displacement

Proof. Fix one preference pair and abbreviate $d =$ $\Delta _ { \mathrm { f u l l } } , g = \nabla _ { \pmb { \theta } } \Delta _ { \mathrm { f u l l } } , \delta = \Delta _ { U } - d ,$ and $e = \nabla _ { \pmb { \theta } } \Delta _ { U } -$ g. Unbiasedness gives $\mathbb { E } \delta = 0$ and $\mathbb { E } e = 0 ,$ , so $V _ { u } = \mathbb { E } \delta ^ { 2 }$ and $V _ { g } = \mathbb { E } \| e \| ^ { 2 }$ . Since $\nabla _ { \pmb { \theta } } \ell ( \Delta _ { U } ) =$ $\ell ^ { \prime } ( d + \delta ) ( g + e )$ and $\nabla _ { \pmb { \theta } } \ell ( d ) = \ell ^ { \prime } ( d ) g$

$$
\begin{array} { r l } & { \nabla _ { \pmb { \theta } } \ell ( \Delta _ { U } ) - \nabla _ { \pmb { \theta } } \ell ( d ) = \big [ \ell ^ { \prime } ( d + \delta ) - \ell ^ { \prime } ( d ) \big ] g } \\ & { ~ + \ell ^ { \prime } ( d + \delta ) e . } \end{array}\tag{29}
$$

The linear term in the first bracket has zero mean, so only $\textstyle { \frac { 1 } { 2 } } \ell ^ { \prime \prime \prime } ( \xi ) \delta ^ { 2 }$ survives, and its norm is at most $\begin{array} { r } { \frac { 1 } { 2 } \operatorname* { s u p } { \lvert \ell ^ { \tilde { \prime \prime \prime } } \rvert } \hat { V _ { u } } \rvert | g \rvert | } \end{array}$ . Because $\ell ^ { \prime \prime \prime } = \beta ^ { 3 } s ( 1 - s ) ( 1 -$ $2 s )$ with $s = \sigma ( \beta d )$ satisfies ma $\mathrm { x } _ { s } | s ( 1 - s ) ( 1 -$ $2 s ) | = 1 / ( 6 \sqrt { 3 } )$ , we have sup $\lvert \ell ^ { \prime \prime \prime } \rvert = \beta ^ { 3 } / ( 6 \sqrt { 3 } )$ For the second term, centering with $\mathbb { E } e = 0$ and applying Cauchy–Schwarz with sup $\vert \ell ^ { \prime \prime } \vert = \beta ^ { 2 } / 4$ bounds it by $\frac { \beta ^ { 2 } } { 4 } \sqrt { V _ { u } V _ { g } }$ . Therefore

$$
\begin{array} { r l r } & { } & { \| \mathbb { E } \nabla _ { \pmb { \theta } } \ell ( \Delta _ { U } ) - \nabla _ { \pmb { \theta } } \ell ( d ) \| \le \displaystyle \frac { \beta ^ { 3 } } { 1 2 \sqrt { 3 } } V _ { u } \| g \| } \\ & { } & { \quad + \displaystyle \frac { \beta ^ { 2 } } { 4 } \sqrt { V _ { u } V _ { g } } . } \end{array}\tag{30}
$$

Averaging over pairs replaces the single-pair quantities by their population counterparts. Substituting $V _ { u } \le 2 \bar { v } r _ { A } , V _ { g } \le 2 G ^ { 2 } r _ { A }$ , and $\| g \| \leq 2 G$ bounds the two terms by ${ \frac { \beta ^ { 3 } G \bar { v } } { 3 \sqrt { 3 } } } r _ { A }$ and ${ \frac { \beta ^ { 2 } G { \sqrt { \bar { v } } } } { 2 } } r _ { A }$ , so that

$$
\begin{array} { c } { \displaystyle \| \nabla F _ { A } ( \pmb \theta ) - \nabla F ( \pmb \theta ) \| \le C _ { A } r _ { A } , } \\ { \displaystyle C _ { A } : = \frac { \beta ^ { 3 } G \bar { v } } { 3 \sqrt { 3 } } + \frac { \beta ^ { 2 } G \sqrt { \bar { v } } } { 2 } . } \end{array}\tag{31}
$$

Now let $\theta _ { A }$ be an interior stationary point of $F _ { A }$ and $\pmb { \theta } ^ { * }$ an interior stationary point of $F ,$ both in the analyzed region, and suppose $\nabla ^ { 2 } F \succeq \mu I$ with $\mu > 0$ along the segment joining them. Then $\| \nabla F ( \pmb \theta _ { A } ) \| = \| \nabla F ( \pmb \theta _ { A } ) - \nabla F _ { A } ( \pmb \theta _ { A } ) \| \le C _ { A } r _ { A }$ Integrating the Hessian along that segment gives $\langle \nabla F ( \pmb { \theta } _ { A } ) , \pmb { \theta } _ { A } - \pmb { \theta } ^ { * } \rangle \geq \mu \| \pmb { \theta } _ { A } - \pmb { \theta } ^ { * } \| ^ { 2 } \mathrm { . }$ ; Cauchy– Schwarz together with the previous display yields $\| \pmb { \theta } _ { A } - \pmb { \theta } ^ { * } \| \leq C _ { A } r _ { A } / \mu$ □

## B.7 Curvature constants from the margin geometry

Suppose $| \beta \Delta _ { \pmb { \theta } } | \leq C$ on the analyzed convex region, $\mathbb { E } _ { \mathcal { D } } [ \nabla _ { \pmb { \theta } } \Delta \nabla _ { \pmb { \theta } } \Delta ^ { \top } ] \succeq \nu I$ , and $\lvert | \nabla _ { \theta } ^ { 2 } \Delta \rvert | _ { \mathrm { o p } } \leq H$ With $m _ { C } = \sigma ( C ) \sigma ( - C )$ , the chain rule gives

$$
\begin{array} { r l } & { \nabla ^ { 2 } F = \mathbb { E } \big [ \ell ^ { \prime \prime } ( \Delta ) \nabla _ { \pmb { \theta } } \Delta \nabla _ { \pmb { \theta } } \Delta ^ { \top } + \ell ^ { \prime } ( \Delta ) \nabla _ { \pmb { \theta } } ^ { 2 } \Delta \big ] } \\ & { \qquad \succeq \big ( \beta ^ { 2 } m _ { C } \nu - \beta H \big ) I , } \end{array}\tag{32}
$$

because $\ell ^ { \prime \prime } \geq \beta ^ { 2 } m _ { C }$ on the region and $| \ell ^ { \prime } | \leq \beta .$ Hence $\beta ^ { 2 } m _ { C } \nu > \beta H$ is a sufficient stability condition and one may take $\mu = \beta ^ { 2 } m _ { C } \nu - \beta H$ . In a linear-margin model $H = 0 .$ , so $\mu > 0$ as soon as the comparison matrix has positive smallest eigenvalue ν.

## B.8 Proof of Eq. (13)

Proof. Let $h _ { b } ~ = ~ 1 - { M _ { b } } / { T _ { b } }$ . Since $\begin{array} { r l } { M _ { b } } & { { } = } \end{array}$ max $( 1 , \lceil \rho _ { b } T _ { b } \rceil )$ and $\rho _ { b } \sim \mathcal { U } ( \rho _ { \mathrm { m i n } } , 1 )$ ,

$$
\begin{array} { c l c r } { \displaystyle \mathbb { E } [ h _ { b } ] = 1 - \frac { \mathbb { E } M _ { b } } { T _ { b } } \leq 1 - \frac { \mathbb { E } \rho _ { b } T _ { b } } { T _ { b } } } \\ { \displaystyle = 1 - \frac { 1 + \rho _ { \mathrm { m i n } } } { 2 } = \frac { 1 - \rho _ { \mathrm { m i n } } } { 2 } . } \end{array}\tag{33}
$$

For one realization, split the reward difference into its retained and removed parts:

$$
\begin{array} { l } { { \displaystyle R _ { b , M _ { b } } - R _ { b } = \left( \frac { 1 } { M _ { b } } - \frac { 1 } { T _ { b } } \right) \sum _ { t \leq M _ { b } } u _ { b , t } } } \\ { { \displaystyle ~ - \frac { 1 } { T _ { b } } \sum _ { t > M _ { b } } u _ { b , t } } , } \end{array}\tag{34}
$$

so $\begin{array} { r } { | R _ { b , M _ { b } } - R _ { b } | \leq ( 1 - \frac { M _ { b } } { T _ { h } } ) U + ( 1 - \frac { M _ { b } } { T _ { h } } ) U = } \end{array}$ $2 U h _ { b }$ . The same argument with $\lvert | \nabla _ { \pmb { \theta } } u _ { b , t } \rvert | \leq G$ gives $\| \nabla _ { \pmb { \theta } } R _ { b , M _ { b } } - \nabla _ { \pmb { \theta } } R _ { b } \| \leq 2 G h _ { b }$ . The margin difference is the difference of the two sides, so $| \Delta _ { M } - \Delta _ { \mathrm { f u l l } } | \ \leq \ 2 U ( h _ { w } + h _ { l } )$ and $\Vert \nabla _ { \pmb { \theta } } \Delta _ { M } - \qquad $ $\begin{array} { r } { \nabla _ { \theta } \Delta _ { \mathrm { f u l l } } \| \leq 2 G ( h _ { w } + h _ { l } ) } \end{array}$

The loss is β-Lipschitz, so $\begin{array} { r } { | \ell ( \Delta _ { M } ) - \ell ( \Delta _ { \mathrm { f u l l } } ) | \leq } \end{array}$ $2 \beta U ( h _ { w } + h _ { l } )$ ; taking expectations gives the loss bound of Eq. (13). For the gradient, decompose

$$
\begin{array} { r l } & { \ell ^ { \prime } ( \Delta _ { M } ) \nabla _ { \pmb { \theta } } \Delta _ { M } - \ell ^ { \prime } ( \Delta _ { \mathrm { f u l l } } ) \nabla _ { \pmb { \theta } } \Delta _ { \mathrm { f u l l } } } \\ & { = \left[ \ell ^ { \prime } ( \Delta _ { M } ) - \ell ^ { \prime } ( \Delta _ { \mathrm { f u l l } } ) \right] \nabla _ { \pmb { \theta } } \Delta _ { M } } \\ & { \quad + \ell ^ { \prime } ( \Delta _ { \mathrm { f u l l } } ) \left[ \nabla _ { \pmb { \theta } } \Delta _ { M } - \nabla _ { \pmb { \theta } } \Delta _ { \mathrm { f u l l } } \right] . } \end{array}\tag{35}
$$

Each side’s mean-reward gradient is an average of token gradients of norm at most $G ,$ so $\| \nabla _ { \pmb { \theta } } \Delta _ { M } \| ~ \le ~ 2 G$ at every truncation, with no truncation-dependent factor. The first term is therefore at most $\begin{array} { r } { \frac { \beta ^ { 2 } } { 4 } \mathrm { ~  ~ \cdot ~ } 2 U ( h _ { w } \mathrm { ~ + ~ } h _ { l } ) \mathrm { ~  ~ \cdot ~ } 2 G = } \end{array}$ $\beta ^ { 2 } U G ( h _ { w } + h _ { l } )$ , using sup $\begin{array} { r } { | \ell ^ { \prime \prime } | = \beta ^ { 2 } / 4 . } \end{array}$ and the second at most $\beta \cdot 2 G ( h _ { w } + h _ { l } )$ . Taking expectations and using $\mathbb { E } [ h _ { w } + h _ { l } ] \le 1 - \rho _ { \operatorname* { m i n } }$ from above gives an expected gradient bias of at most $G ( \beta ^ { 2 } U + 2 \beta ) ( 1 - \rho _ { \mathrm { m i n } } )$ , which is the constant $C _ { P }$ used in Sec. B.10.

For the optimality statement, $F \leq F _ { P } + \Delta$ with $\Delta = 2 \beta U \big ( 1 - \rho _ { \mathrm { m i n } } \big )$ , so $F ( { \pmb \theta } _ { P } ^ { * } ) \le F _ { P } ( { \pmb \theta } _ { P } ^ { * } ) + \Delta \le$ $F r ( { \pmb \theta } ^ { * } ) + \Delta \ \leq \ F ( { \pmb \theta } ^ { * } ) + 2 \Delta$ The displacement bound follows from the segment argument of Sec. B.6 applied to the gradient bound above, and inherits the same requirements: an interior stationary point of each objective lying in the analyzed region, and $\nabla ^ { 2 } F \succeq \mu I$ with $\mu > 0$ along the joining segment.

The loss bound is a scaling statement, not a guarantee at the tested setting. Since $| R _ { b } | \ \leq \ U$ we have $| \Delta | \leq 2 U$ , so the total range of the loss on the relevant interval is $\ell ( - 2 U ) - \ell ( 2 U ) ~ =$ $2 \beta U$ . At $\rho _ { \mathrm { m i n } } = 1 / 2$ the optimality-gap bound $4 \beta U ( 1 - \rho _ { \mathrm { m i n } } )$ reaches $2 \beta U$ , exactly that range, so it is vacuous at the tested cutoff; the loss bound itself is $2 \beta U ( 1 - \rho _ { \mathrm { m i n } } ) ~ = ~ \beta U$ , half that range. Both become restrictive only for cutoffs closer to one. □

## B.9 Proof of Proposition 3 and Eq. (41)

Proof. With $c _ { w } , c _ { l }$ as defined in Proposition 3, $\mathbb { E } [ \hat { \pmb { g } } ] = \pmb { c } _ { w } + \pmb { c } _ { l }$ follows from Proposition 2, and

$$
\mathbb { E } \Vert \hat { \pmb { g } } \Vert ^ { 2 } = q _ { w } \frac { \Vert \pmb { c } _ { w } \Vert ^ { 2 } } { q _ { w } ^ { 2 } } + q _ { l } \frac { \Vert \pmb { c } _ { l } \Vert ^ { 2 } } { q _ { l } ^ { 2 } } = \frac { \Vert \pmb { c } _ { w } \Vert ^ { 2 } } { q _ { w } } + \frac { \Vert \pmb { c } _ { l } \Vert ^ { 2 } } { q _ { l } } ,\tag{36}
$$

so $\mathrm { V a r } ( \hat { \pmb g } ) = \mathbb { E } \| \hat { \pmb g } \| ^ { 2 } - \| \pmb { c } _ { w } + \pmb { c } _ { l } \| ^ { 2 }$ is Eq. (17). At $q _ { w } = q _ { l } = 1 / 2$ it equals $2 \| \pmb { c } _ { w } \| ^ { 2 } + 2 \| \pmb { c } _ { l } \| ^ { 2 } - \| \pmb { c } _ { w } +$ $\pmb { c } _ { l } \| ^ { 2 } = \| \pmb { c } _ { w } - \pmb { c } _ { l } \| ^ { 2 }$ . Because $| \lambda | = \beta \sigma ( - \beta \Delta ) \le \beta$ and $\begin{array} { r } { \| \nabla _ { \pmb { \theta } } R _ { b } \| \le G _ { \frac { } { } } } \end{array}$ , we have $\| \pmb { c } _ { w } \| , \| \pmb { c } _ { l } \| \le \beta G$ hence $\begin{array} { r } { \mathrm { V a r } ( \hat { g } ) \leq 4 \beta ^ { 2 } G ^ { 2 } } \end{array}$ , and at the symmetric default also $\| \hat { \pmb { g } } \| = \| \pmb { c } _ { w } \| / q _ { w } \le 2 \beta G$ . Averaging m estimators divides the variance by m when the m draws are independent in their preference pairs, response positions, prefixes and branches; if only the branch is re-drawn, only the branch component is reduced.

For the rate, take the symmetric default $q _ { w } =$ $q _ { l } = 1 / 2$ so that each step’s noise has second moment at most $4 \beta ^ { 2 } G ^ { 2 } / m$ . Let $\pmb { \theta } _ { \Phi } ^ { * }$ minimize Φ and $D _ { n } = \mathbb { E } \| \pmb { \theta } _ { n } - \pmb { \theta } _ { \Phi } ^ { * } \| ^ { 2 }$ . Projection is non-expansive and $\theta _ { \Phi } ^ { * } \in \Theta$ , so

$$
\begin{array} { r l r } & { } & { D _ { n + 1 } \leq D _ { n } - 2 \eta _ { n } \mathbb { E } \langle \nabla \Phi ( \pmb { \theta } _ { n } ) , \pmb { \theta } _ { n } - \pmb { \theta } _ { \Phi } ^ { * } \rangle } \\ & { } & { \qquad + \eta _ { n } ^ { 2 } \Big ( \mathbb { E } \| \nabla \Phi ( \pmb { \theta } _ { n } ) \| ^ { 2 } + \frac { 4 \beta ^ { 2 } G ^ { 2 } } { m } \Big ) . } \end{array}\tag{37}
$$

Let A be the average Hessian along the segment from $\pmb { \theta } _ { \Phi } ^ { * }$ to $\pmb \theta _ { n }$ . Strong convexity gives $A \succeq \mu I$ and smoothness gives $A \preceq L I$ , so $A ^ { 2 } \preceq L A$ . Since $\nabla \Phi ( \pmb \theta _ { n } ) = A ( \pmb \theta _ { n } - \pmb \theta _ { \Phi } ^ { * } )$

$$
\begin{array} { r } { \| \nabla \Phi ( \pmb { \theta } _ { n } ) \| ^ { 2 } \leq L \langle \nabla \Phi ( \pmb { \theta } _ { n } ) , \pmb { \theta } _ { n } - \pmb { \theta } _ { \Phi } ^ { \ast } \rangle . } \end{array}\tag{38}
$$

Substituting this and using $\left. \nabla \Phi , \pmb \theta - \pmb \theta _ { \Phi } ^ { * } \right. \geq \mu D _ { n }$ together with $\eta _ { n } L \leq 1$ , which $t _ { 0 } \geq 2 L / \mu$ ensures,

$$
D _ { n + 1 } \leq ( 1 - \mu \eta _ { n } ) D _ { n } + \frac { 4 \beta ^ { 2 } G ^ { 2 } } { m } \eta _ { n } ^ { 2 } .\tag{39}
$$

Insert $\eta _ { n } = 2 / [ \mu ( n + t _ { 0 } ) ]$ , write $s = n + t _ { 0 }$ , and suppose $D _ { n } \leq A _ { B } / s$ . Then

$$
\begin{array} { c } { { D _ { n + 1 } \leq \displaystyle \frac { A _ { B } ( s - 2 ) } { s ^ { 2 } } + \displaystyle \frac { 1 6 \beta ^ { 2 } G ^ { 2 } } { \mu ^ { 2 } m s ^ { 2 } } } } \\ { { \leq \displaystyle \frac { A _ { B } ( s - 1 ) } { s ^ { 2 } } \leq \displaystyle \frac { A _ { B } } { s + 1 } , } } \end{array}\tag{40}
$$

where the middle step uses $A _ { B } \geq 1 6 \beta ^ { 2 } G ^ { 2 } / ( \mu ^ { 2 } m )$ and the last uses $( s - 1 ) ( s + 1 ) \leq s ^ { 2 }$ . Since $D _ { 0 } \leq$ $A _ { B } / t _ { 0 }$ , induction gives

$$
\begin{array} { r } { \mathbb { E } \| \pmb { \theta } _ { N } - \pmb { \theta } _ { \Phi } ^ { * } \| ^ { 2 } \leq \displaystyle \frac { A _ { B } } { N + t _ { 0 } } , \qquad } \\ { A _ { B } : = \operatorname* { m a x } \Bigl \{ t _ { 0 } \| \pmb { \theta } _ { 0 } - \pmb { \theta } _ { \Phi } ^ { * } \| ^ { 2 } , } \\ { \frac { 1 6 \beta ^ { 2 } G ^ { 2 } } { \mu ^ { 2 } m } \Bigr \} } \end{array}\tag{41}
$$

for $\eta _ { n } = 2 / [ \mu ( n + t _ { 0 } ) ]$ and $t _ { 0 } \geq$ max $\lbrace 2 , 2 L / \mu \rbrace$ Markov’s inequality applied to $\lVert \pmb { \theta } _ { N } - \pmb { \theta } _ { \Phi } ^ { * } \rVert ^ { 2 }$ gives $\operatorname* { P r } ( \| \pmb { \theta } _ { N } - \pmb { \theta } _ { \Phi } ^ { * } \| > \varepsilon _ { \theta } ) \leq A _ { B } / [ ( N + t _ { 0 } ) \varepsilon _ { \theta } ^ { 2 } ]$ □

## B.10 Proof of the composed bias and error budget

Proof. Write $r _ { A | P } = ( 1 - \alpha ) / [ \alpha ( M _ { 0 } - 1 ) ]$ . Write $\nabla F _ { A P } - \nabla F = ( \nabla F _ { A P } - \nabla F _ { P } ) + ( \nabla F _ { P } - \nabla F )$ Conditioned on the executed lengths $( M _ { w } , M _ { l } )$ the first difference is the position-pruning bias of Eq. (31) evaluated at those lengths. A fixed prefix can have a larger token-score variance than the whole response, so this difference must be averaged over the cutoff law rather than bounded prefix by prefix. Two facts supply that average. First, the within-prefix sum of squares cannot exceed the whole-response sum of squares: splitting the total sum of squares into the prefix, the suffix and the part between their means gives

$$
\begin{array} { l } { { \displaystyle \sum _ { t \le T } ( u _ { t } - \bar { u } _ { T } ) ^ { 2 } = \sum _ { t \le M } ( u _ { t } - \bar { u } _ { M } ) ^ { 2 } } } \\ { { + \displaystyle \sum _ { t > M } ( u _ { t } - \bar { u } _ { > M } ) ^ { 2 } } } \\ { { + \displaystyle \frac { M ( T - M ) } { T } \big ( \bar { u } _ { M } - \bar { u } _ { > M } \big ) ^ { 2 } , } } \end{array}\tag{42}
$$

so $M v _ { M } \ \leq \ T v _ { T }$ and hence $v _ { M } ~ \le ~ ( T / M ) v _ { T }$ with the suffix terms read as zero when $M \ =$ T. Second, for $M _ { b } ~ = ~ \operatorname* { m a x } ( 1 , \lceil \rho _ { b } T _ { b } \rceil )$ with $\rho _ { b } \sim \mathcal { U } ( \rho _ { \mathrm { m i n } } , 1 )$ , prefixes of length one carry zero variance and the remaining lengths satisfy $\mathbb { E } [ \mathbf { 1 } _ { \{ M _ { b } > 1 \} } T _ { b } / ( M _ { b } ( M _ { b } - 1 ) ) ] \le 1 / ( M _ { 0 } - 1 )$ with $M _ { 0 } { \bf { \bar { \alpha } } } = \operatorname* { m a x } ( 2 , \lceil \rho _ { \mathrm { { m i n } } } T _ { 0 } \rceil )$ . Combining the two bounds the prefix-averaged position-pruning bias by $C _ { A } r _ { A | P }$ , the factor two accounting for the two sides. The second difference is the prefix bias of $\operatorname { E q . } \left( 1 3 \right)$ , contributing $C _ { P } ( 1 - \rho _ { \mathrm { m i n } } )$ with $C _ { P } = G ( \beta ^ { 2 } U + 2 \beta )$ . Adding the two gives

$$
\begin{array} { r l } & { \| \nabla F _ { A P } - \nabla F \| \leq C _ { A } r _ { A | P } + C _ { P } ( 1 - \rho _ { \operatorname* { m i n } } ) , } \\ & { \qquad C _ { P } : = G ( \beta ^ { 2 } U + 2 \beta ) . } \end{array}\tag{43}
$$

If the composed objective has interior minimizer $\pmb { \theta } _ { A P } ^ { * }$ and F has curvature at least $\mu _ { F }$ on the relevant segment, the segment argument of Sec. B.6 gives

$$
\| \pmb { \theta } _ { A P } ^ { * } - \pmb { \theta } ^ { * } \| \leq \frac { C _ { A } r _ { A | P } + C _ { P } ( 1 - \rho _ { \operatorname* { m i n } } ) } { \mu _ { F } } .\tag{44}
$$

The $L ^ { 2 }$ triangle inequality $\lVert { \pmb \theta } _ { N } - { \pmb \theta } ^ { * } \rVert \leq \lVert { \pmb \theta } _ { N }$ $\pmb { \theta } _ { A P } ^ { * } \| + \| \pmb { \theta } _ { A P } ^ { * } - \pmb { \theta } ^ { * } \|$ , combined with Eq. (41) ap-

plied to the composed objective, yields

$$
\begin{array} { r l } { \sqrt { \mathbb { E } \| \pmb { \theta } _ { N } - \pmb { \theta } ^ { * } \| ^ { 2 } } \leq \frac { C _ { A } r _ { A | P } + C _ { P } ( 1 - \rho _ { \operatorname* { m i n } } ) } { \mu _ { F } } } & { } \\ { + \sqrt { \frac { A _ { B , A P } } { N + t _ { 0 , A P } } } , } \end{array}\tag{45}
$$

where $\mu _ { F }$ bounds the curvature of $F$ on the relevant segment and $A _ { B , A P } , t _ { 0 , A P }$ are the constants of Eq. (41) evaluated for the composed objective.

For the composed optimal loss, $F _ { A P } \geq F _ { P }$ by Jensen’s inequality in the position masks and $F \leq$ $F _ { P } + 2 \beta U ( 1 - \rho _ { \mathrm { m i n } } )$ . Chaining these as in Sec. B.5 gives

$$
F ( \pmb { \theta } _ { A P } ^ { * } ) - F ( \pmb { \theta } ^ { * } ) \leq \frac { \beta ^ { 2 } \bar { v } } { 4 } r _ { A | P } + 4 \beta U ( 1 - \rho _ { \operatorname* { m i n } } ) .\tag{46}
$$

## C Shared-Parameter Analysis

This appendix expands the shared-parameter coupling discussed in the introduction.

## C.1 Shared-Parameter Updates and Signed Token Movement

Let i index token positions across a pair, $p _ { i }$ be the current-policy target-token probability, and $\phi _ { i } =$ $\nabla _ { \theta }$ log p<sub>i</sub> its log-probability gradient. Set $c _ { i } =$ $1 / k _ { w }$ for a selected chosen token, $c _ { i } = - 1 / k _ { l }$ for a selected rejected token, and $c _ { i } = 0$ otherwise. Within a region of fixed mask membership,

$$
\pmb { g } = \nabla _ { \pmb { \theta } } \Delta = \sum _ { i } c _ { i } \phi _ { i } .\tag{47}
$$

For a small gradient-descent step of size $\eta > 0$ on the pair loss, the parameter change is $\eta \beta \sigma ( - \beta \Delta ) \underline { { \theta } }$ Writing δ for the resulting change, to first order

$$
\begin{array} { r l r } & { } & { \delta \log p _ { i } = \eta \beta \sigma ( - \beta \Delta ) \phi _ { i } ^ { \top } { \pmb g } + { \mathcal O } ( \eta ^ { 2 } ) , } \\ & { } & { \delta \Delta = \eta \beta \sigma ( - \beta \Delta ) \| { \pmb g } \| ^ { 2 } + { \mathcal O } ( \eta ^ { 2 } ) . } \end{array}\tag{48}
$$

The pair margin increases to first order, yet an individual chosen probability can decrease if $\phi _ { i } ^ { \top } \pmb { g } < 0$ Token confidence alone does not determine these inner products (Yuan et al., 2025; Razin et al., 2025). Minibatching and AdamW further change the actual update. The single-pair gradient calculation identifies the missing interaction rather than modeling that optimizer exactly.

## D Experimental Settings and Results

This appendix documents the experimental settings and additional results: the UltraFeedback motivation experiments (Sec. D.1), the HelpSteer3 corpus and recipe (Sec. D.2), the retention-sweep geometry (Sec. D.3), the factorial and parameter sweeps (Sec. D.4), the full per-task tables (Sec. D.5), the detailed HelpSteer3 tables (Sec. D.6), and the supplemental attribution and Full–Branch validation (Sec. D.7).

## D.1 The UltraFeedback Experiments

The UltraFeedback collection uses sequence length 512, prompt length 256, one training epoch, AdamW, cosine decay, BFloat16 computation, 4- bit loading, gradient checkpointing, and LoRA on attention and MLP projections. The reference is the frozen policy initialization. Qwen thinking mode is disabled for evaluation. Experiments run on Ubuntu 22.04 with two NVIDIA RTX 5090 GPUs (32 GiB each), Python 3.12, PyTorch 2.8.0, and CUDA 12.8. Masks are selected independently on the two sides each step, with the chosen side ranked by current-policy confidence (Low) or uniformly (Random), and gradients bypass the discrete selection.

All three models use LoRA dropout 0.05. The principal comparison uses retention 0.8 and training seed 42 for each rule. Pythia-2.8B uses learning rate $5 \times 1 0 ^ { - 7 }$ , DPO $\beta = 0 . 1$ , LoRA rank 16 and scaling alpha 32, per-device batch size 2, 16 gradient accumulation steps, and warmup fraction 0.10. Qwen3-4B uses learning rate $1 . 5 \times 1 0 ^ { - 6 } , \beta = 0 . 5 .$ LoRA rank 64 and scaling alpha 128, per-device batch size 2, 16 gradient accumulation steps, and warmup fraction 0.08. Qwen3-8B uses learning rate $7 . 5 \times 1 0 ^ { - 7 } , \beta = 0 . 5$ , LoRA rank 32 and scaling alpha 64, per-device batch size 1, 32 gradient accumulation steps, and warmup fraction 0.10.

The tasks include ARC Challenge (Clark et al., 2018), BBH (Suzgun et al., 2023), MMLU-Pro (Wang et al., 2024), MuSR (Sprague et al., 2024), TruthfulQA MC1 (Lin et al., 2022), IFEval promptlevel strict accuracy (Zhou et al., 2023), GSM8K (Cobbe et al., 2021), and MATH-Hard drawn from the MATH dataset (Hendrycks et al., 2021). ARC uses 25-shot normalized accuracy, and GSM8K uses 5-shot exact-answer evaluation. The pertask tables (Tabs. 7 and 8) preserve evaluatorreported standard errors, which do not measure training-seed variability. Mean accuracy is a point summary. Pythia’s mean accuracy excludes GSM8K and MATH-Hard because all its variants score zero on both. The summary averages cited in Sec. E.1 are: $\mathrm { L o w { - } A l l + 1 . 9 5 / + 0 . 5 1 / + }$ 0.32, $\mathrm { R a n d o m { - } A l l + 1 . 5 8 / + 0 . 0 0 / + 0 . 0 6 , }$ , and Low−Random $+ 0 . 3 7 / + 0 . 5 1 / + 0 . 2 6$ points for Pythia-2.8B, Qwen3-4B, and Qwen3-8B, respectively. The High/High row and the cross-side rules (Low/High, High/Low) are shown for completeness only. They are not part of the matched Low– Random comparison and are not interpreted in the main text. The UltraFeedback and HelpSteer3 collections differ in recipe and evaluation protocol, so absolute scores are not comparable across them.

## D.2 The HelpSteer3 Corpus and Recipe

Preference conversion and recipe. The preference conversion retains 36,264 of 38,459 training pairs and 1,917 of 2,017 validation pairs, removing ties and empty responses. The main experiments use Qwen3-4B with DPO $\beta = 1$ , LoRA rank 64 and alpha 128, dropout 0.05 on attention and MLP projections, BFloat16 computation, FlashAttention-2, gradient checkpointing, AdamW at learning rate $1 0 ^ { - 6 }$ , cosine decay, warmup fraction 0.05, and maximum gradient norm 0.5. Sequence and prompt limits are 2,048 and 1,024. Per-device batch is 2. The factorial and parameter sweeps use training and selection seed 42. The reference is computed online. The supplemental Full–Branch comparisons retain this recipe at Qwen3-4B seed 43 and Mistral-7B seed 42 (Sec. D.7).

## D.3 Retention-Sweep Geometry

The sweep in Tab. 12 compares Low/Low and Random/Random at retentions 0.1–0.9, with masks recomputed every step. Gradient accumulation is 16 for retentions 0.1–0.4 and 4 for 0.5–0.9, giving effective batches of 32 (1,133 updates) and 64 (566 updates) for one epoch. Each Low–Random pair within a block shares its geometry and data order. Comparisons are therefore interpreted within each retention level, and the two blocks do not form one controlled dose-response curve. Tab. 11 gives the full per-task scores.

## D.4 The Factorial and Parameter Sweeps

Factorial. The factorial of Sec. 4.1 uses the same corpus and recipe with a fixed batch geometry: per-device batch 2, gradient accumulation 4, eight workers, global batch 64 pairs, 566 optimizer steps for one epoch. Position pruning uses $\alpha ~ = ~ 0 . 6$ (uniform response-position subsets, selected-token mean reward). Sequence pruning uses $\rho _ { b } \sim \mathcal { U } ( 0 . 5 , 1 . 0 )$ (prefix truncation with repacking and a matching reference cutoff). Branch pruning uses $q _ { w } = q _ { l } = 0 . 5$ (one backward branch per pair, opposite branch forwarded without gradient, $1 / q _ { b }$ importance correction and stop-gradient surrogate). Composition follows Alg. 1: truncation first, then response-position pruning inside the retained prefix, then branch pruning. Policy and reference see identical truncations and masks. All eight configurations in Tab. 4 were trained endto-end under this recipe and evaluated on all ten benchmarks.

Realized sparsity matches the nominal targets. With all three modes enabled, the selected fraction is 0.6013, the mean $\rho _ { b }$ is 0.7545, and the response retention is 0.7662.

Parameter sweeps. The within-axis sweeps of Tab. 3 are ten trainings that share the factorial’s recipe and differ from it only in the ablated knob. The sweep midpoints $( \alpha { = } 0 . 6 , \mathcal { U } ( 0 . 5 , 1 ) , q _ { w } { = } 0 . 5 )$ coincide with the factorial’s Position, Sequence, and Branch rows. Tab. 2 gives the sequencepruning resource profile in full. Realized sparsity matches nominal for position pruning (selected fraction $0 . 3 0 2 / 0 . 6 0 1 / 0 . 9 0 2 )$ and branch pruning (global chosen fractions $0 . 2 9 8 / 0 . 4 9 8 / 0 . 6 9 9$ for $q _ { w } ~ = ~ 0 . 3 / 0 . 5 / 0 . 7 )$ For sequence pruning, the sampled cutoff matches its distribution (mean $\rho _ { b } ~ 0 . 6 3 2 / 0 . 7 5 5 / 0 . 8 7 7$ against theory $0 . 6 2 5 / 0 . 7 5 0 / 0 . 8 7 5$ , and $\rho _ { b } = 0 . 7 5 0$ exactly for the fixed-cutoff configuration), with corresponding response-token retention $0 . 6 5 / 0 . 7 7 / 0 . 8 8$ . The Tokens column in Tab. $2 \left( 0 . 8 5 4 / 0 . 8 9 3 / 0 . 9 4 3 / 0 . 8 9 4 \right)$ is a third quantity: the executed-sequence fraction, the ratio of executed sequence tokens, prompt and padding included, to the stock baseline. It lies above the response-token retention and must not be compared with the nominal $\rho _ { b }$ . Peak HBM was not measured across $q _ { w } ,$ so branch pruning is compared on training wall-clock only. The per-task scores of all ten runs are in Tab. 10.

Mean generation score in every table uses the reported task means and the scaling in Sec. 4.1.

Descriptive factorial contrasts. Raw differences of aggregate columns across the factorial are descriptive contrasts on single-seed runs. Contrasts use unrounded aggregates and can differ from differences of the displayed columns by 0.01. Adding position pruning changes mean accuracy by −0.27, −0.07, −0.14, and −0.17 pp over Full, Sequence, Branch, and Sequence+Branch, respectively. Adding sequence pruning changes it by −0.29, −0.09, −0.22, and −0.25 pp over Full, Position, Branch, and Position+Branch. Adding branch pruning changes it by −0.26, −0.12, −0.19, and −0.28 pp over Full, Position, Sequence, and Position+Sequence.

<table><tr><td>Benchmark</td><td>All/All</td><td>Random/Random</td><td>High/High</td><td>Low/Low</td></tr><tr><td> $Q w e n 3 – 4 B$ </td><td></td><td></td><td></td><td></td></tr><tr><td>ARC</td><td> $6 2 . 8 0 { \scriptstyle \pm 1 . 4 1 }$ </td><td> $6 2 . 9 7 { \scriptstyle \pm 1 . 4 1 }$ </td><td> $6 2 . 4 6 { \scriptstyle \pm 1 . 4 2 }$ </td><td> $6 2 . 9 7 { \scriptstyle \pm 1 . 4 1 }$ </td></tr><tr><td>BBH</td><td> $5 6 . 1 6 { \scriptstyle \pm 3 . 0 1 }$ </td><td> $5 6 . 1 6 { \scriptstyle \pm 3 . 0 1 }$ </td><td> $5 6 . 0 8 { \scriptstyle \pm 3 . 0 1 }$ </td><td> $5 6 . 3 3 { \scriptstyle \pm 3 . 0 1 }$ </td></tr><tr><td>MMLU-Pro</td><td> $4 2 . 4 6 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $4 2 . 2 5 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $4 2 . 2 4 { \pm } 0 . 4 5$ </td><td> $4 2 . 5 0 { \scriptstyle \pm 0 . 4 5 }$ </td></tr><tr><td>MuSR</td><td> $4 2 . 4 6 { \pm } 1 . 7 3$ </td><td> $4 3 . 3 9 { \scriptstyle \pm 1 . 7 4 }$ </td><td> $4 2 . 7 2 { \scriptstyle \pm 1 . 7 4 }$ </td><td> $4 3 . 2 5 { \scriptstyle \pm 1 . 7 4 }$ </td></tr><tr><td>TruthfulQA</td><td> $3 7 . 9 4 { \scriptstyle \pm 1 . 7 0 }$ </td><td> $3 8 . 0 7 { \scriptstyle \pm 1 . 7 0 }$ </td><td> $3 6 . 1 1 { \pm } 1 . 6 8$ </td><td> $3 8 . 5 6 { \scriptstyle \pm 1 . 7 0 }$ </td></tr><tr><td>IFEval</td><td> $4 3 . 9 9 2 . 1 4$ </td><td> $4 0 . 6 7 { \scriptstyle \pm 2 . 1 1 }$ </td><td> $4 4 . 3 6 { \scriptstyle \pm 2 . 1 4 }$ </td><td> $4 3 . 5 6 { \pm } 2 . 1 0$ </td></tr><tr><td>GSM8K</td><td> $7 4 . 1 5 { \pm } 1 . 2 1 $ </td><td> $7 4 . 9 8 { \pm } 1 . 1 9$ </td><td> $6 8 . 7 6 { \scriptstyle \pm 1 . 2 8 } $ </td><td> $7 5 . 3 6 { \pm } 1 . 1 9$ </td></tr><tr><td>MATH-Hard</td><td> $3 7 . 9 4 { \scriptstyle \pm 3 . 4 4 }$ </td><td> $3 9 . 3 9 { \scriptstyle \pm 3 . 4 5 }$ </td><td> $3 7 . 2 5 { \scriptstyle \pm 3 . 3 6 }$ </td><td> $3 9 . 5 0 { \scriptstyle \pm 3 . 4 6 }$ </td></tr><tr><td>Mean accuracy (%)</td><td>49.74</td><td>49.74</td><td>48.75</td><td>50.25</td></tr><tr><td>Qwen3-8B</td><td></td><td></td><td></td><td></td></tr><tr><td>ARC</td><td> $6 8 . 1 7 { \scriptstyle \pm 1 . 3 6 }$ </td><td> $6 8 . 0 9 { \scriptstyle \pm 1 . 3 6 }$ </td><td> $6 8 . 3 4 { \scriptstyle \pm 1 . 3 6 }$ </td><td> $6 8 . 2 9 { \scriptstyle \pm 1 . 3 6 }$ </td></tr><tr><td>BBH</td><td> $6 0 . 8 4 { \pm } 2 . 9 8 $ </td><td> $6 0 . 8 0 { \scriptstyle \pm 2 . 9 8 }$ </td><td> $6 0 . 5 5 { \scriptstyle \pm 2 . 9 9 }$ </td><td> $6 1 . 3 4 { \pm } 2 . 9 8$ </td></tr><tr><td>MMLU-Pro</td><td> $4 7 . 6 3 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $4 7 . 7 0 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $4 7 . 6 1 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $4 7 . 6 8 { \scriptstyle \pm 0 . 4 6 }$ </td></tr><tr><td>MuSR</td><td> $4 3 . 3 9 { \scriptstyle \pm 1 . 7 7 }$ </td><td> $4 3 . 7 8 { \scriptstyle \pm 1 . 7 7 }$ </td><td> $4 2 . 8 6 { \scriptstyle \pm 1 . 7 7 }$ </td><td> $4 3 . 3 9 { \scriptstyle \pm 1 . 7 7 }$ </td></tr><tr><td>TruthfulQA</td><td> $3 6 . 7 2 { \scriptstyle \pm 1 . 6 9 }$ </td><td> $3 6 . 9 6 { \scriptstyle \pm 1 . 6 9 }$ </td><td> $3 6 . 8 4 { \scriptstyle \pm 1 . 6 9 }$ </td><td> $3 6 . 8 4 { \pm } 1 . 6 9 $ </td></tr><tr><td>IFEval</td><td> $2 5 . 8 8 { \scriptstyle \pm 1 . 8 8 }$ </td><td> $2 5 . 3 2 { \scriptstyle \pm 1 . 8 7 }$ </td><td> $2 4 . 0 3 { \scriptstyle \pm 1 . 8 4 }$ </td><td> $2 6 . 0 6 { \scriptstyle \pm 1 . 8 9 }$ </td></tr><tr><td>GSM8K</td><td> $7 8 . 0 9 { \scriptstyle \pm 1 . 1 4 }$ </td><td> $7 7 . 6 3 { \pm } 1 . 1 5$ </td><td> $7 7 . 8 6 { \pm } 1 . 1 4 $ </td><td> $7 8 . 6 2 { \scriptstyle \pm 1 . 1 3 }$ </td></tr><tr><td>MATH-Hard</td><td> $4 6 . 2 5 { \scriptstyle \pm 3 . 4 5 }$ </td><td> $4 7 . 1 3 { \scriptstyle \pm 3 . 4 4 }$ </td><td> $4 6 . 3 8 { \scriptstyle \pm 3 . 4 9 }$ </td><td> $4 7 . 2 9 { \scriptstyle \pm 3 . 4 6 }$ </td></tr><tr><td>Mean accuracy (%)</td><td>50.87</td><td>50.93</td><td>50.56</td><td>51.19</td></tr></table>

Table 7: UltraFeedback Qwen3 task scores (%) with evaluator-reported standard errors. Mean accuracy is the eight-benchmark point mean. Non-All rules use retention 0.8. These errors do not measure training variability.
<table><tr><td>Method</td><td>ARC</td><td>BBH</td><td>MMLU-Pro</td><td>MuSR</td><td>IFEval</td><td></td><td>Mean TruthfulQA accuracy (%)</td></tr><tr><td>All/All</td><td> $2 5 . 5 1 { \scriptstyle \pm 1 . 2 7 }$ </td><td> $2 8 . 4 6 { \scriptstyle \pm 2 . 6 4 }$ </td><td> $1 0 . 5 9 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $3 5 . 8 5 { \scriptstyle \pm 1 . 7 1 }$ </td><td> $1 1 . 4 6 { \scriptstyle \pm 1 . 3 7 }$ </td><td> $2 2 . 4 0 { \scriptstyle \pm 1 . 4 6 }$ </td><td>22.38</td></tr><tr><td>Random/Random</td><td> $2 8 . 1 6 { \pm } 1 . 3 1 $ </td><td> $2 7 . 4 7 { \scriptstyle \pm 2 . 5 5 }$ </td><td> $1 0 . 8 6 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $3 6 . 6 4 { \scriptstyle \pm 1 . 7 2 }$ </td><td> $1 5 . 5 3 { \scriptstyle \pm 1 . 5 6 }$ </td><td> $2 5 . 0 9 { \scriptstyle \pm 1 . 5 2 }$ </td><td>23.96</td></tr><tr><td>Low/Low</td><td> $2 7 . 7 3 { \scriptstyle \pm 1 . 3 1 }$ </td><td> $2 8 . 4 7 { \scriptstyle \pm 2 . 6 9 }$ </td><td> $1 0 . 8 4 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $3 9 . 9 5 { \scriptstyle \pm 1 . 7 3 }$ </td><td> $1 1 . 4 6 { \scriptstyle \pm 1 . 3 7 }$ </td><td> $2 7 . 5 4 { \scriptstyle \pm 1 . 5 6 }$ </td><td>24.33</td></tr><tr><td>Low/High</td><td> $2 4 . 9 1 { \scriptstyle \pm 1 . 2 6 }$ </td><td> $2 8 . 9 2 { \scriptstyle \pm 2 . 6 9 }$ </td><td> $1 1 . 6 6 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $3 5 . 5 8 { \scriptstyle \pm 1 . 7 0 }$ </td><td> $1 8 . 3 0 { \scriptstyle \pm 1 . 6 6 }$ </td><td> $2 2 . 6 4 { \scriptstyle \pm 1 . 4 7 }$ </td><td>23.67</td></tr><tr><td>High/Low</td><td> $2 6 . 1 1 { \scriptstyle \pm 1 . 2 8 }$ </td><td> $2 9 . 4 7 { \scriptstyle \pm 2 . 7 2 }$ </td><td> $1 1 . 6 6 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $3 4 . 7 9 { \scriptstyle \pm 1 . 6 9 }$ </td><td> $1 2 . 5 7 { \scriptstyle \pm 1 . 4 3 }$ </td><td> $2 3 . 0 1 { \scriptstyle \pm 1 . 4 7 }$ </td><td>22.93</td></tr><tr><td>High/High</td><td> $2 5 . 5 1 { \scriptstyle \pm 1 . 2 7 }$ </td><td> $2 9 . 9 9 { \scriptstyle \pm 2 . 7 6 }$ </td><td> $1 1 . 1 6 { \scriptstyle \pm 0 . 2 9 } $ </td><td> $3 4 . 3 9 { \scriptstyle \pm 1 . 6 8 }$ </td><td> $1 2 . 2 0 { \scriptstyle \pm 1 . 4 1 }$ </td><td> $2 3 . 1 3 { \pm } 1 . 4 8$ </td><td>22.73</td></tr></table>

Table 8: UltraFeedback Pythia-2.8B task scores (%) at retention 0.8. Per-task errors are evaluator-reported standard errors. Mean accuracy is the six-task point mean; GSM8K and MATH-Hard are excluded because all variants score zero on both.

mean accuracy contrasts are negative. The branchpruning effect on training wall-clock is consistent in sign: −59.4, −58.1, −60.1, and −58.0 min in the four pairings.

## D.5 Full Task Tables

Tab. 11 reports per-task scores for Low/Low and Random/Random at each retention level on HelpSteer3-Preference.

Contrasts in mean generation score use the scaling in Sec. 4.1 and have mixed signs. Position pruning gives −0.95, +1.52, −2.07, and +0.68 in the four pairings above. Sequence pruning gives −2.01, +0.47, −2.58, and +0.17. Branch pruning gives +0.92, −0.21, +0.35, and −0.50. All twelve

## D.6 Detailed HelpSteer3 Tables

The following tables retain the count-matched motivation results and the per-axis parameter settings summarized in the main text; the completed facto-

<table><tr><td></td><td colspan="2">Efficiency</td><td colspan="8">Capability (accuracy, %)</td><td colspan="4">Open-ended</td></tr><tr><td>Method</td><td>Wall</td><td>Speed</td><td>ARC</td><td>GSM8K</td><td>TQA</td><td>BBH</td><td>MATH</td><td>MMLU</td><td>MuSR</td><td>Mean accuracy (%)</td><td>MT</td><td>AH</td><td>WB</td><td>Mean generation score</td></tr><tr><td>Full</td><td>187.8</td><td>1.000×</td><td>62.54</td><td>87.41</td><td>53.88</td><td>55.64</td><td>63.59</td><td>42.28</td><td>42.99</td><td>58.33</td><td>6.649</td><td>50.12</td><td>3.139</td><td>49.33</td></tr><tr><td>Position</td><td>185.9</td><td>1.010×</td><td>62.46</td><td>86.43</td><td>53.51</td><td>55.58</td><td>63.26</td><td>42.18</td><td>42.99</td><td>58.06</td><td>6.829</td><td>44.85</td><td>3.200</td><td>48.38</td></tr><tr><td>Sequence</td><td>186.4</td><td>1.008×</td><td>62.37</td><td>86.28</td><td>53.40</td><td>55.92</td><td>63.27</td><td>42.20</td><td>42.86</td><td>58.04</td><td>6.432</td><td>45.80</td><td>3.186</td><td>47.33</td></tr><tr><td>Branch</td><td>128.4</td><td>1.463×</td><td>62.03</td><td>86.66</td><td>53.06</td><td>55.73</td><td>63.65</td><td>42.16</td><td>43.25</td><td>58.08</td><td>6.800</td><td>50.11</td><td>3.263</td><td>50.25</td></tr><tr><td>Position+Sequence</td><td>185.0</td><td>1.015×</td><td>62.20</td><td>85.97</td><td>53.06</td><td>55.74</td><td>63.63</td><td>42.07</td><td>43.12</td><td>57.97</td><td>6.893</td><td>45.54</td><td>3.208</td><td>48.85</td></tr><tr><td>Position+Branch</td><td>127.8 126.31.487×</td><td>1.469×</td><td>61.95 62.03</td><td>86.50 86.20</td><td>52.92</td><td>55.64 55.69</td><td>63.31 63.06</td><td>42.12 42.05</td><td>43.12 43.12</td><td>57.94 57.86</td><td>6.676 6.600</td><td>45.80 45.05</td><td>3.196 3.197</td><td>48.17 47.67</td></tr><tr><td>Sequence+Branch Position+Sequence</td><td></td><td></td><td></td><td></td><td>52.84</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+Branch</td><td>127.01.479×62.03</td><td></td><td></td><td>85.97</td><td>52.75</td><td>55.65</td><td>62.24</td><td>42.05</td><td>43.12</td><td>57.69</td><td>6.693</td><td>45.86</td><td>3.225</td><td>48.35</td></tr></table>

Table 9: Full factorial task scores underlying Tab. 4. Metrics and score units follow Sec. 4.1. Bold/underline mark the best/second-best displayed column values.
<table><tr><td colspan="10">Mean</td><td rowspan="2">Mean generation</td><td rowspan="2">score</td></tr><tr><td>Setting</td><td>ARC</td><td>GSM8K</td><td>TQA</td><td>BBH</td><td>MATH MMLU MuSR</td><td></td><td></td><td>accuracy (%) MT</td><td>AH</td><td>WB</td></tr><tr><td>Position α=0.3</td><td>62.29</td><td>86.20</td><td>53.15</td><td>55.61</td><td>62.78</td><td>42.15</td><td>43.12</td><td>57.90</td><td>6.568</td><td>45.56</td><td>3.195</td><td>47.73</td></tr><tr><td>Position α=0.6</td><td>62.46</td><td>86.43</td><td>53.51</td><td>55.58</td><td>63.26</td><td>42.18</td><td>42.99</td><td>58.06</td><td>6.829</td><td>44.85</td><td>3.200</td><td>48.38</td></tr><tr><td>Position α=0.9</td><td>62.54</td><td>87.34</td><td>53.83</td><td>55.64</td><td>63.54</td><td>42.26</td><td>43.12</td><td>58.32</td><td>6.707</td><td>48.00</td><td>3.081</td><td>48.63</td></tr><tr><td>Sequence U(0.25, 1)</td><td>62.37</td><td>86.35</td><td>53.30</td><td>55.80</td><td>63.53</td><td>42.15</td><td>43.12</td><td>58.09</td><td>6.552</td><td>47.05</td><td>3.117</td><td>47.91</td></tr><tr><td>Sequence U(0.5, 1)</td><td>62.37</td><td>86.28</td><td>53.40</td><td>55.92</td><td>63.27</td><td>42.20</td><td>42.86</td><td>58.04</td><td>6.432</td><td>45.80</td><td>3.186</td><td>47.33</td></tr><tr><td>Sequence U(0.75, 1)</td><td>62.46</td><td>86.28</td><td>53.44</td><td>56.03</td><td>63.16</td><td>42.16</td><td>42.72</td><td>58.04</td><td>6.874</td><td>47.92</td><td>3.193</td><td>49.53</td></tr><tr><td>Sequence fixed 0.75</td><td>62.29</td><td>86.73</td><td>53.38</td><td>55.95</td><td>63.23</td><td>42.13</td><td>42.86</td><td>58.08</td><td>6.562</td><td>43.33</td><td>3.214</td><td>47.03</td></tr><tr><td>Branch  $q _ { w } { = } 0 . 3$ </td><td>62.37</td><td>87.34</td><td>53.03</td><td>55.44</td><td>62.70</td><td>42.16</td><td>43.39</td><td>58.06</td><td>6.669</td><td>47.82</td><td>3.189</td><td>48.80</td></tr><tr><td>Branch  $q _ { w } { = } 0 . 5$ </td><td>62.03</td><td>86.66</td><td>53.06</td><td>55.73</td><td>63.65</td><td>42.16</td><td>43.25</td><td>58.08</td><td>6.800</td><td>50.11</td><td>3.263</td><td>50.25</td></tr><tr><td>Branch  $q _ { w } { = } 0 . 7$ </td><td>62.03</td><td>86.58</td><td>52.97</td><td>55.75</td><td>62.92</td><td>42.15</td><td>43.39</td><td>57.97</td><td>6.753</td><td>46.64</td><td>3.205</td><td>48.74</td></tr></table>

Table 10: Per-task single-axis sweeps, one training seed. Shading marks defaults; rows specify settings. Metric units follow Sec. 4.1. Evaluator standard errors (0.45–3.02 pp) do not test aggregate differences or training variability.

rial is in Tab. 4. All factorial and sweep configurations use seed 42.

## D.7 Supplemental Attribution and Full–Branch Validation

The attribution and Full–Branch validation add four complete trainings and two resource profiles on the same eight-accelerator hardware. Results are in Tabs. 1, 5 and 6; this section gives reproduction and scoring details.

Output implementation control. One Qwen profiler compares Full-concat, selected-head execution at retention 1.0, and selected-head execution at 0.6. The configurations share weights, inputs, reference computation, precision, and the $2 \times 4 \times 8$ training geometry (per-device batch 2, gradient accumulation 4, eight workers). Four warm-up rounds precede 16 measured rounds. Configuration order rotates within rounds, memory counters reset before each measurement, and peak HBM is the maximum over ranks. Full-retention selected-head rewards match Full-concat exactly, as do dense-reduced and selected-head rewards under identical masks. Numerical checks also compare LoRA gradients with a row-order control. Mean step times and peak HBM are reported in Tab. 1. Memory differences are computed within this profile: 8,905 MB for fullsupervision selected-head execution and a further 642 MB for random deletion.

Matched training pairs. Qwen Full and Branch at seed 43 use the same initialization, dataset version, data order, training budget, optimizer, schedule, precision, attention backend, checkpointing, and online reference. Mistral-7B-Instruct-v0.3 uses the same recipe and 566-step budget at seed 42. Both models fit the $2 \times 4 \times 8$ geometry without a batch-size change. Branch pruning uses $q _ { w } = q _ { l } = 0 . 5$ , with position and sequence pruning disabled. The global chosen-branch fractions are 0.5088 for Qwen seed 43 and 0.4956 for Mistral, averaged over all optimizer steps.

The Mistral setup is verified on chat template, prompt masking, EOS padding, the online reference, and the selected/no-gradient branches. The gradient checks in Sec. 4.5 disable LoRA dropout while retaining checkpointing; the trainings use

<table><tr><td colspan="10">Mean GSM8K TruthfulQA BBH MATH-Hard MMLU-Pro MuSR</td></tr><tr><td>Selection</td><td>Retention 0.1</td><td>ARC 63.48</td><td>85.82</td><td></td><td></td><td></td><td></td><td></td><td>accuracy (%)</td></tr><tr><td>Low/Low</td><td>0.1</td><td>62.88</td><td>86.73</td><td>54.57 53.11</td><td>55.54 55.41</td><td>62.39 63.36</td><td>42.26 42.00</td><td>42.86 42.59</td><td>58.13</td></tr><tr><td>Random/Random</td><td></td><td></td><td>86.58</td><td>55.19</td><td>55.32</td><td>60.29</td><td>42.33</td><td>42.59</td><td>58.01</td></tr><tr><td>Low/Low</td><td>0.2 0.2</td><td>63.82 63.05</td><td>86.35</td><td>53.62</td><td>55.06</td><td>63.14</td><td>42.02</td><td>42.46</td><td>58.02</td></tr><tr><td>Random/Random</td><td>0.3</td><td>63.82</td><td>87.26</td><td>55.49</td><td>55.47</td><td>58.47</td><td>42.35</td><td>42.59</td><td>57.96</td></tr><tr><td>Low/Low Random/Random</td><td>0.3</td><td>63.23</td><td>86.50</td><td>54.46</td><td>55.51</td><td>63.20</td><td>42.04</td><td>42.72</td><td>57.92</td></tr><tr><td>Low/Low</td><td>0.4</td><td>63.99</td><td>88.02</td><td>55.70</td><td>55.43</td><td>57.67</td><td>42.32</td><td>42.86</td><td>58.24</td></tr><tr><td>Random/Random</td><td>0.4</td><td>63.40</td><td>86.88</td><td>54.85</td><td>55.64</td><td>61.92</td><td>42.10</td><td></td><td>58.00</td></tr><tr><td>Low/Low</td><td>0.5</td><td>63.14</td><td>87.19</td><td>53.81</td><td>55.63</td><td></td><td>42.25</td><td>42.59</td><td>58.20</td></tr><tr><td>Random/Random</td><td>0.5</td><td>63.05</td><td>86.50</td><td>53.38</td><td>55.63</td><td>63.66 63.92</td><td>42.22</td><td>43.12</td><td>58.40</td></tr><tr><td>Low/Low</td><td>0.6</td><td>63.40</td><td>87.26</td><td>53.85</td><td>55.60</td><td>63.33</td><td>42.22</td><td>43.12 43.12</td><td>58.26</td></tr><tr><td>Random/Random</td><td>0.6</td><td>63.05</td><td>86.81</td><td>53.52</td><td>55.71</td><td>64.11</td><td>42.20</td><td>42.99</td><td>58.40</td></tr><tr><td>Low/Low</td><td>0.7</td><td>63.31</td><td>87.26</td><td>53.88</td><td>55.53</td><td>63.39</td><td>42.25</td><td>42.99</td><td>58.34</td></tr><tr><td>Random/Random</td><td>0.7</td><td>63.23</td><td>86.81</td><td>53.67</td><td>55.77</td><td>64.17</td><td>42.20</td><td>42.86</td><td>58.37 58.39</td></tr><tr><td>Low/Low</td><td>0.8</td><td>63.14</td><td>87.34</td><td>53.91</td><td>55.60</td><td>63.28</td><td>42.27</td><td>42.99</td><td>58.36</td></tr><tr><td>Random/Random</td><td>0.8</td><td>63.05</td><td>87.49</td><td>53.76</td><td>55.78</td><td>64.01</td><td>42.24</td><td>42.59</td><td>58.42</td></tr><tr><td>Low/Low</td><td>0.9</td><td>63.23</td><td>87.26</td><td>53.92</td><td>55.57</td><td>63.43</td><td>42.30</td><td>42.99</td><td>57.55*</td></tr><tr><td>Random/Random</td><td>0.9</td><td>63.14</td><td>87.11</td><td>53.86</td><td>55.61</td><td></td><td>42.26</td><td>43.12</td><td>57.52*</td></tr><tr><td>Qwen3-4B</td><td></td><td>62.80</td><td>85.90</td><td>52.27</td><td>55.59</td><td>62.80</td><td>41.98</td><td>43.39</td><td></td></tr><tr><td>Unmasked DPO</td><td></td><td>62.63</td><td>87.57</td><td></td><td>53.9155.61</td><td>63.33</td><td>42.30</td><td>43.12</td><td>57.82 58.35</td></tr></table>

Table 11: HelpSteer3-Preference task scores (%). Low and Random are compared within a retention level. Retentions 0.1–0.4 and 0.5–0.9 use different training schedules. Mean accuracy is the seven-task mean. <sup>∗</sup>At 0.9 both means use six tasks, excluding MATH-Hard, which is missing for Random. The last two rows are baselines: the untrained Qwen3-4B starting checkpoint and unmasked DPO, which provides context but does not use a matched training schedule. A dash denotes a missing result.

<table><tr><td>α Low/Low Random/Random Difference</td></tr><tr><td>Batch 32</td></tr><tr><td>0.1 58.13 58.01 +0.12</td></tr><tr><td>0.2 58.02 57.96 +0.06</td></tr><tr><td>0.3 57.92 58.24 -0.32</td></tr><tr><td>0.4 58.00 58.20 -0.20</td></tr><tr><td>Batch 64</td></tr><tr><td>0.5 58.40 58.26 +0.14</td></tr><tr><td>0.6 58.40 58.34 +0.06</td></tr><tr><td>0.7 58.37 58.39 -0.02</td></tr><tr><td>0.8 58.36 58.42 -0.06</td></tr><tr><td>0.9 57.55 57.52 +0.03*</td></tr></table>

Table 12: Qwen3-4B on HelpSteer3-Preference: mean capability scores (%) for confidence-ranked (Low/Low) versus matched uniform (Random/Random) masks at retention α. Comparisons are within rows. <sup>∗</sup>At 0.9 both means use six tasks, excluding MATH-Hard, which is missing for Random. Differences use displayed means.

dropout 0.05.

Per-task results. Tab. 13 retains every benchmark score for the repeated Qwen pair and the Mistral pair. Tab. 13 retains the native open-ended scores; Tab. 5 reports category means. The conventional Branch-minus-Full differences are −0.25 and −0.45 pp for the two Qwen seeds (a descriptive mean of −0.35 pp), and −0.31 pp for Mistral.

Open-ended scoring. All supplemental generations are judged by MiniMax-M3. The evaluator reports valid-item means separately for each checkpoint. The supplemental WildBench suite has 260 prompts, four more than the main evaluation’s 256. The Qwen seed-43 Full Arena-Hard generation has 499 judged entries and 443 valid scores. Tab. 13 reports per-checkpoint scores. These summaries do not characterize training-seed variability.

## E Additional Token-Selection Analysis

This section reports the matched token-selection comparisons referenced in the main text.

## E.1 Supervision Membership Is a Limited Lever

We compare selecting positions with the lowest current-policy target-token probabilities (Low) against uniform random selection (Random). Low/Low and Random/Random use the same selection rule on both chosen and rejected responses. At equal retained ratios, Low/Low exceeds Random/Random by 0.37, 0.51, and 0.26 meanaccuracy points on UltraFeedback (Cui et al., 2024) for Pythia-2.8B (Biderman et al., 2023), Qwen3- 4B, and Qwen3-8B. On HelpSteer3-Preference, the difference changes sign across retention levels and stays within 0.32 points (Tab. 12). Task differences are larger: at low retention, Low trails Random on MATH-Hard while leading TruthfulQA (Tab. 11). These comparisons motivate uniform position pruning before vocabulary projection. We compare selection rules at a fixed number of scored positions. The factorial experiment separately assesses the model-quality cost of reduced supervision relative to the Full baseline.

<table><tr><td>Model / seed</td><td>Config.</td><td>ARC</td><td>GSM8K</td><td>TQA</td><td>BBH</td><td>MATH</td><td>MMLU</td><td>MuSR</td><td>Mean accuracy (%)</td><td>MT</td><td>AH</td><td>WB</td><td>Mean generation score</td></tr><tr><td rowspan="2">Qwen3-4B / 42</td><td>Full</td><td>62.54</td><td>87.41</td><td>53.88</td><td>55.64</td><td>63.59</td><td>42.28</td><td>42.99</td><td>58.33</td><td>6.649</td><td>50.12</td><td>3.139</td><td>49.33</td></tr><tr><td>Branch</td><td>62.03</td><td>86.66</td><td>53.06</td><td>55.73</td><td>63.65</td><td>42.16</td><td>43.25</td><td>58.08</td><td>6.800</td><td>50.11</td><td>3.263</td><td>50.25</td></tr><tr><td rowspan="2">Qwen3-4B / 43</td><td>Full</td><td>62.54</td><td>87.04</td><td>53.88</td><td>55.55</td><td>63.45</td><td>42.34</td><td>42.99</td><td>58.26</td><td>6.623</td><td>44.36</td><td>3.233</td><td>47.64</td></tr><tr><td>Branch</td><td>62.03</td><td>86.20</td><td>53.05</td><td>55.55</td><td>62.57</td><td>42.14</td><td>43.12</td><td>57.81</td><td>6.750</td><td>45.03</td><td>3.295</td><td>48.49</td></tr><tr><td rowspan="2">Mistral-7B / 42</td><td>Full</td><td>67.06</td><td>52.46</td><td>64.82</td><td>46.37</td><td>12.50</td><td>30.47</td><td>40.34</td><td>44.86</td><td>5.130</td><td>24.60</td><td>2.361</td><td>33.17</td></tr><tr><td>Branch</td><td>65.27</td><td>51.18</td><td>62.87</td><td>46.51</td><td>12.96</td><td>30.32</td><td>42.72</td><td>44.55</td><td>5.373</td><td>24.00</td><td>2.405</td><td>33.93</td></tr></table>

Table 13: Complete Full–Branch scores in the units of Sec. 4.1. Qwen seed-42 values repeat the factorial evaluations in Tab. 4.

## F LLM Usage Disclosure

In answer to the submission questions on the use of large language models, a large language model was used to aid and polish the writing of this paper, for retrieval and discovery such as finding related work, and to draft and revise sections of the manuscript.