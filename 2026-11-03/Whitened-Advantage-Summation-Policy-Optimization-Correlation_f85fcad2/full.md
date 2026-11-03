# Whitened Advantage Summation Policy Optimization: Correlation-Aware Aggregation for Multi-Reward Reinforcement Learning

Olaitan Olaleye

ProofMed AI

olaitan@proofmed.ai

Jayson Meribe

ProofMed AI and Stanford University

Temidayo Adepoju

ProofMed AI

meribejayson@gmail.com

dayo.olaleye25@gmail.com

## Abstract

Reinforcement learning for language models often combines several rewards by normalizing each one and adding them with fixed weights. When the rewards are correlated, the variation they share is counted once for every reward that carries it. In terms of the ZCA-decorrelated components of the reward signal, the weighted sum applies weights $\pmb { \Sigma } ^ { 1 / 2 } \mathbf { w }$ rather than the chosen weights w, where Σ is the reward correlation matrix. We study WAS-PO, which whitens the normalized advantages with $\Sigma ^ { - 1 / 2 }$ before adding them, so that the chosen weights apply to those components. We derive the scalarized objective whose gradient it follows when normalization statistics are held fixed, show that its weights on the raw rewards can be negative, give an exact two-reward condition under which the lighter-weighted reward is reversed, and give a loose bound on the error from estimating Σ during training. On tool calling with Qwen2.5-1.5B-Instruct, WAS-PO reached a higher mean training reward than GDPO in all seven seeds with three rewards (3.8% higher) and all three seeds with four (3.9% higher). WAS-PO also replaces GDPO’s per-prompt normalization with batch-level normalization, and we did not separate the two changes with three or four rewards. The training-reward gains did not carry over to held-out accuracy, which was slightly lower than GDPO’s with three rewards (not significant) and 0.03 lower in absolute terms with four (two seeds evaluated). With two rewards, WAS-PO trailed GDPO in training reward, and a three-seed control weakly suggests that the gap comes from batch-level normalization rather than whitening. In a math setting where the two rewards were nearly identical by construction, WAS-PO drove down the lighter-weighted reward, as the sign condition predicts.

## 1 Introduction

Reinforcement learning for language models now routinely optimizes several rewards at once, such as correctness, output format and length. Group Relative Policy Optimization (GRPO; Shao et al., 2024) handles this by summing the rewards and normalizing the sum within each group of completions sampled for a prompt. Liu et al. (2026) showed that summing first maps distinct combinations of rewards to the same advantage, and proposed Group reward-Decoupled Normalization Policy Optimization (GDPO), which normalizes each reward within the group before taking a weighted sum.

The weights in that sum are meant to say how much each objective matters, but correlated rewards change what they do. When normalized advantages of correlated rewards are added, the variation they share enters the sum once per reward. In terms of the decorrelated components of the reward signal (Section 3.1), a weighted sum with weights w applies weights $\pmb { \Sigma } ^ { 1 / 2 } \mathbf { w }$ , where Σ is the correlation matrix of the rewards. Correlation of this kind is common in multi-reward training. In our tool-calling setup, following GDPO’s conditional rewards, the format, length and structure rewards are set to zero whenever accuracy is not positive, so they rise and fall together largely by construction. At the batch-level correlations that WAS-PO estimated in our runs, the efective weight vector of the batch-normalized sum points away from w by $9 . 6 ^ { \circ }$ $1 6 . 0 ^ { \circ }$ and $2 0 . 7 ^ { \circ }$ with two, three and four rewards. Whether this is a distortion depends on how the weights are read, which we discuss in Section 3.1.

Whitening the normalized advantages with $\Sigma ^ { - 1 / 2 }$ (ZCA whitening) before summing applies w to the decorrelated components that are closest to the individual rewards. We call the result Whitened Advantage Summation Policy Optimization (WAS-PO). Whitening is not free. WAS-PO is a scalarized policy gradient whose weights on the raw rewards, $\mathbf { D } ^ { - 1 } \pmb { \Sigma } ^ { - 1 / 2 } \mathbf { w }$ with D the diagonal matrix of reward standard deviations, can be negative (Proposition 2). With two rewards weighted $a > b > 0$ , the lighter-weighted reward receives a negative weight exactly when the correlation exceeds $2 a b / ( a ^ { 2 } + b ^ { 2 } )$ , so the policy is trained to lower that reward, and the negative weight grows without bound as the correlation approaches one (Proposition 3). We call this ZCA-Inversion and observe it in a math setting, where the format reward fell from 0.67 to 0.13 during training.

In our three- and four-reward tool-calling settings, WAS-PO reached higher training reward than GDPO in all ten paired runs (seven at $K = 3$ , three at $K = 4 )$ . The gain did not carry over to held-out accuracy, and with two rewards WAS-PO trailed GDPO even in training reward. WAS-PO also replaces GDPO’s per-prompt normalization with batch-level normalization, so the comparison with GDPO mixes two design changes. With two rewards, a control that keeps batch-level normalization but drops whitening (WAS-PO-Diag) scores slightly below WAS-PO, which weakly suggests that normalization is the source of the two-reward gap. We have not run this control with three or four rewards, so we report the gains there as gains of WAS-PO as a whole.

## Our contributions are:

1. an exact statement of how correlation changes the efective weights of a weighted sum of normalized advantages, with a variance decomposition (Section 3.1);

2. the objective that whitened aggregation follows, and the corresponding per-prompt objective of GDPO (Section 3.2);

3. an exact two-reward condition for sign reversal, its extension to near-duplicate rewards among K, and an inversion index that can be logged during training (Section 3.3);

4. an error bound for the estimated whitening matrix (Section 3.4);

5. experiments on tool calling with two, three and four rewards, component ablations, an observed instance of ZCA-Inversion, and runs in which a nearly singular estimate did not cause collapse (Section 5).

## 2 Background

For a prompt x, a policy $\pi _ { \theta }$ samples G completions $y _ { 1 } , \dotsc , y _ { G }$ . Each completion receives a vector of K rewards $\mathbf { r } _ { i } \in \mathbb { R } ^ { K }$ , and the user supplies weights $\mathbf { w } \in \mathbb { R } _ { > 0 } ^ { K }$ . A training batch contains N completions, which we call rollouts. We write ${ \mathrm { m e a n } } _ { g }$ and $\operatorname { s t d } _ { g }$ for statistics over the G completions of one prompt, $\epsilon = 1 0 ^ { - 4 }$ for a constant that prevents division by zero, and $\mathrm { B N } ( u ) _ { i } = ( u _ { i } - \mathrm { m e a n } ( u ) ) / ( \mathrm { s t d } ( u ) + \epsilon )$ for normalization across the batch. Standard deviations are sample standard deviations, as in the trainer we use.

GRPO forms a scalar reward and normalizes it within the group:

$$
\hat { A } _ { i } ^ { \mathrm { G R P O } } = \frac { \mathbf { w } ^ { \top } \mathbf { r } _ { i } - \mathrm { m e a n } _ { g } ( \mathbf { w } ^ { \top } \mathbf { r } ) } { \mathrm { s t d } _ { g } ( \mathbf { w } ^ { \top } \mathbf { r } ) + \epsilon } .
$$

GDPO normalizes each reward within the group, takes the weighted sum, and normalizes the result across the batch:

$$
\hat { A } _ { i } ^ { ( k ) } = \frac { r _ { i } ^ { ( k ) } - \mathrm { m e a n } _ { g } ( r ^ { ( k ) } ) } { \mathrm { s t d } _ { g } ( r ^ { ( k ) } ) + \epsilon } , \qquad \hat { A } _ { i } ^ { \mathrm { G D P O } } = \mathrm { B N } \Big ( \sum _ { k = 1 } ^ { K } w _ { k } \hat { A } ^ { ( k ) } \Big ) _ { i } .
$$

Within a group, each $\hat { A } ^ { ( k ) }$ has zero mean and close to unit variance, or is identically zero if reward k is constant on that group. The vector $\hat { \mathbf { A } } _ { i } = ( \hat { A } _ { i } ^ { ( 1 ) } , \ldots , \hat { A } _ { i } ^ { ( K ) } )$ keeps the correlation of the rewards. If Σ is the covariance of $\hat { \bf A }$ and has unit diagonal, with entries $\rho _ { j k }$ , the covariance of the summed advantage with the k-th normalized advantage is

$$
\operatorname { C o v } \left( \mathbf { w } ^ { \top } \hat { \mathbf { A } } , \hat { A } ^ { ( k ) } \right) = ( \pmb { \Sigma } \mathbf { w } ) _ { k } = w _ { k } + \sum _ { j \neq k } w _ { j } \rho _ { j k } ,\tag{1}
$$

so positively correlated rewards raise each other’s covariance with the aggregate. Both methods update the policy with the clipped surrogate objective of PPO (Schulman et al., 2017), a variant of the score-function estimator (Williams, 1992).

## 3 What Weighted Summation Does to Reward Weights

## 3.1 Efective Weights

Let $\hat { \mathbf { A } } \in \mathbb { R } ^ { K }$ have zero mean and covariance $\Sigma \succ 0$ with unit diagonal, and let $\pmb { \Sigma } ^ { \pm 1 / 2 }$ denote the symmetric positive definite square root and its inverse. Define

$$
\begin{array} { r } { \mathbf { z } : = \Sigma ^ { - 1 / 2 } \hat { \mathbf { A } } , \qquad \mathrm { s o ~ t h a t } \qquad \hat { \mathbf { A } } = \Sigma ^ { 1 / 2 } \mathbf { z } \quad \mathrm { a n d } \quad \mathrm { C o v } ( \mathbf { z } ) = \mathbf { I } . } \end{array}\tag{2}
$$

The coordinates of z are uncorrelated, and we call them the decorrelated components of the reward signal.

Any W with $\mathbf { W } ^ { \top } \mathbf { W } = \pmb { \Sigma } ^ { - 1 }$ decorrelates $\hat { \bf A } .$ . The symmetric choice $\mathbf { W } = \mathbf { \sum } ^ { - 1 / 2 }$ is zero-phase component analysis (ZCA) whitening. Because the coordinates of $\hat { \bf A }$ are standardized, it coincides with what Kessy et al. (2018) call ZCA-cor whitening of the raw rewards, which among all whitening transforms maximizes the total correlation $\textstyle \sum _ { k } \operatorname { C o r r } ( z _ { k } , { \tilde { A } } ^ { \ j } )$ between each component and its own reward. Principal component analysis (PCA) whitening produces components with no correspondence to individual rewards, and Cholesky whitening depends on the order in which the rewards are listed.

Proposition 1 (Efective weights). For any w $\in \mathbb { R } ^ { K }$

$$
\begin{array} { r } { \mathbf { w } ^ { \top } \hat { \mathbf { A } } = \left( \pmb { \Sigma } ^ { 1 / 2 } \mathbf { w } \right) ^ { \top } \mathbf { z } \quad \quad a n d \quad \quad \mathbf { w } ^ { \top } \pmb { \Sigma } ^ { - 1 / 2 } \hat { \mathbf { A } } = \mathbf { w } ^ { \top } \mathbf { z } . } \end{array}\tag{3}
$$

Weighted summation applies weights $\pmb { \Sigma } ^ { 1 / 2 } \mathbf { w }$ to the decorrelated components, and summation after ZCA whitening applies w. The two agree for all w if and only i $f \Sigma = \mathbf { I }$

The proof is in Appendix A. We measure the discrepancy by the angle $\varphi _ { \mathbf { w } }$ between $\boldsymbol { \Sigma } ^ { 1 / 2 } \mathbf { w }$ and w.

Proposition 1 does not say which weighting is appropriate, because that depends on what w is meant to weight. Let $J _ { k } ( \theta ) = \mathbb { E } _ { x , y \sim \pi _ { \theta } } [ r ^ { ( k ) } ]$ . If w weights the raw rewards, the target is $\sum _ { k } w _ { k } J _ { k }$ and correlation calls for no correction; summing the rewards before normalizing, as GRPO does, follows it, while summing separately normalized advantages follows $\sum _ { k } ( w _ { k } / \sigma _ { k } ) J _ { k }$ (Proposition 2). If w is meant to weight the decorrelated components that track each reward, whitening is what applies those weights. We adopt the second reading, which matches the intuition that a weight such as 0.3 on length should not be inflated because length moves with accuracy, and we examine its consequences in Section 5. A third reading, under which Cov(aggregate, A<sup>ˆ(k)</sup>) = w<sub>k</sub> in (1), would call for the regression weights ${ \boldsymbol { \Sigma } } ^ { - 1 } \mathbf { w }$ , which we do not consider further.

Corollary 1 (Variance decomposition). Let $\kappa _ { \mathbf { w } } : = \mathbf { w } ^ { \top } \pmb { \Sigma } \mathbf { w } / \| \mathbf { w } \| ^ { 2 }$ . Then

$$
\operatorname { V a r } ( \mathbf { w } ^ { \top } \hat { \mathbf { A } } ) = \mathbf { w } ^ { \top } \pmb { \Sigma } \mathbf { w } = \| \mathbf { w } \| ^ { 2 } + \sum _ { j \neq k } w _ { j } w _ { k } \rho _ { j k } , \qquad \operatorname { V a r } ( \mathbf { w } ^ { \top } \pmb { \Sigma } ^ { - 1 / 2 } \hat { \mathbf { A } } ) = \| \mathbf { w } \| ^ { 2 } ,
$$

so cross-reward terms account for a fraction $1 - 1 / \kappa _ { \mathbf { w } }$ of the variance of the summed advantage.

We computed $\varphi _ { \mathbf { w } }$ and $\kappa _ { \mathbf { w } }$ from Σ<sup>¯</sup> , the time average of the correlation estimate that WAS-PO maintained during training (Table 13). The angle is $9 . 6 ^ { \circ } , ~ 1 6 . 0 ^ { \circ }$ and $2 0 . 7 ^ { \circ }$ for $K = 2 , 3 , 4$ , and $\kappa _ { \mathbf { w } }$ is 1.52, 1.95 and 2.30, so cross-reward terms carry 34%, 49% and 57% of the variance of the unwhitened sum. Because these use batch-level correlations, they describe the unwhitened sum under batch-level normalization (the WAS-PO-Diag control of Section 4). For GDPO, Proposition 1 applies with Σ replaced by the pooled covariance of its group-normalized advantages, rescaled to unit diagonal; it reflects within-group correlation only, and we did not log it. Without renormalization, $\sqrt { \kappa _ { \mathbf { w } } }$ would be the factor by which correlation inflates the scale of the summed advantage. GDPO and WAS-PO both renormalize across the batch, which removes this efect, so here $\kappa _ { \mathbf { w } }$ describes only the composition of the aggregate.

## 3.2 The Objective Whitened Aggregation Follows

Let $\mathbf { v } : = \Sigma ^ { - 1 / 2 } \mathbf { w }$ be the vector that whitened aggregation applies to the normalized advantages, and let $\mathbf { D } = \mathrm { d i a g } ( \sigma _ { 1 } , \dots , \sigma _ { K } )$ hold the reward standard deviations used for normalization.

Proposition 2 (Efective objectives). Consider the sequence-level score-function estimator without a KL penalty or length normalization. Treat all normalization statistics, the matrix Σ and the final batch scale $s > 0$ (the denominator of BN) as constants.

(a) Suppose the aggregate advantage is $\tilde { A } _ { i } = s ^ { - 1 } \mathbf { v } ^ { \top } \hat { \mathbf { A } } _ { i } ,$ with $\hat { A } _ { i } ^ { ( k ) } = ( r _ { i } ^ { ( k ) } - \mu _ { k } ) / \sigma _ { k }$ computed from batch statistics. Then the expected update $\mathbb { E } [ \tilde { A } \nabla _ { \theta } \log \pi _ { \theta } ( y \mid x ) ]$ equals $s ^ { - 1 } \nabla _ { \theta } J _ { \omega } ( \theta )$ , where

$$
J _ { \omega } ( \theta ) = \sum _ { k = 1 } ^ { K } \omega _ { k } J _ { k } ( \theta ) , \qquad \omega = \mathbf { D } ^ { - 1 } \Sigma ^ { - 1 / 2 } \mathbf { w } .\tag{4}
$$

(b) With per-prompt statistics $\mu _ { k } ( x ) , \sigma _ { k } ( x ) > 0$ and $\mathbf { v } = \mathbf { w }$ , as in $G D P O ,$ the same holds with $J =$ $\begin{array} { r } { \mathbb E _ { x } \left[ \sum _ { k } \bigl ( w _ { k } / \sigma _ { k } ( x ) \bigr ) \mathbb E _ { y \sim \pi _ { \theta } ( \cdot | x ) } [ r ^ { ( k ) } ] \right] } \end{array}$

Under these idealizations WAS-PO is a scalarized policy gradient with weights ω on the raw rewards. The idealizations are standard but not exact. The batch statistics depend on the sampled completions (an $O ( 1 / N )$ efect), the trainer’s token-level loss difers from the sequence-level estimator, and ω changes during training as the estimate of Σ and the scales D change. In our runs each batch is used for a single update, so the clipping in the surrogate is inactive. Part (b) shows that GDPO weights each reward per prompt by the inverse of its within-group standard deviation, so a reward that varies little on a prompt receives a larger weight there (at most $w _ { k } / \epsilon ;$ a reward that is constant on the group contributes nothing). For binary rewards with $G = 4$ , the within-group standard deviation is 0.50 or 0.58 when nonzero, so this reweighting matters mainly for continuous rewards. The batch-normalized sum without whitening follows $\mathbf { D } ^ { - 1 } \mathbf { w }$ , whose entries are all positive. For two positively correlated rewards, and at all our measured correlation matrices, $\Sigma ^ { - 1 / 2 }$ has negative of-diagonal entries, so ω can have negative entries. A negative $\omega _ { k }$ rewards the policy for lowering reward k when the other rewards are held fixed.

## 3.3 When Whitening Reverses a Reward

Proposition 3 (Sign reversal with two rewards). Let $K = 2$ and $\mathbf { w } = ( a , b )$ with $a \geq b > 0$ , and let the correlation be $\rho \in ( - 1 , 1 )$ . Then $v _ { 1 } > 0$ , and

$$
v _ { 2 } < 0 \quad \Longleftrightarrow \quad \rho > \rho ^ { \star } ( a , b ) : = \frac { 2 a b } { a ^ { 2 } + b ^ { 2 } } .
$$

If $a > b ,$ , then $| v _ { 2 } | \to \infty$ and $v _ { 2 } / v _ { 1 }  - 1$ as $\rho  1$ . If a = b, then $\rho ^ { \star } = 1$ and no reversal occurs.

The threshold depends only on the ratio of the weights. For the math setting of Section 5.4, $\mathbf { w } = ( 1 . 0 , 0 . 3 )$ gives $\rho ^ { \star } = 0 . 5 5 ;$ for the two-reward tool-calling setting, $\mathbf { w } = ( 1 . 0 , 0 . 5 )$ gives $\rho ^ { \star } = 0 . 8 0$ . As an example, at $\rho = 0 . 9 8$ and $\mathbf { w } = ( 1 . 0 , 0 . 3 ) , \mathbf { v } = ( 2 . 9 4 , - 2 . 0 1 )$ : an accurate but unformatted completion gains advantage relative to an accurate and formatted one. As $\rho  1$ , the aggregate approaches a positive multiple of $\hat { A } ^ { ( 1 ) } - \hat { A } ^ { ( 2 ) }$ . Batch-level renormalization removes the growth in magnitude but not this direction. Corollary 2 extends the result to K rewards when two of them become nearly collinear.

Table 1: Diagnostics at the estimated correlations. $\varphi _ { \mathbf { w } }$ is the angle between $\pmb { \Sigma } ^ { 1 / 2 }$ w and $\mathbf { w }$ (Proposition 1); v = Ww with the eigenvalue floor of Section 4; ψ is the inversion index (5). “Time-averaged” rows use $\dot { \Sigma }$ (Table 13). “Largest” rows set every pair to its largest logged value; the maxima need not occur at the same step. At $K = 3 ,$ 4 these matrices are singular to numerical precision, so there the magnitudes of v are set by the floor and only its signs and $\psi$ are meaningful. $\Delta \mathrm { A U C }$ is WAS-PO minus GDPO.
<table><tr><td>Setting</td><td>W</td><td>4w</td><td>V</td><td>ψ</td><td>∆AUC</td></tr><tr><td>Tool, K = 2, time-averaged</td><td>(1.0, 0.5)</td><td> $9 . 6 ^ { \circ }$ </td><td>(1.00, 0.17)</td><td>0.00</td><td>-0.014</td></tr><tr><td>Tool, K = 2, largest</td><td>(1.0, 0.5)</td><td></td><td>(1.24, −0.14)</td><td>0.11</td><td></td></tr><tr><td>Tool, K = 3, time-averaged</td><td>(1.0, 0.5, 0.3)</td><td> $1 6 . 0 ^ { \circ }$ </td><td>(1.05, 0.30, −0.15)</td><td>0.15</td><td>+0.041</td></tr><tr><td>Tool, K = 3, largest</td><td>(1.0, 0.5, 0.3)</td><td></td><td>(1.22, 9.84, −9.95)</td><td>1.01</td><td></td></tr><tr><td>Tool, K = 4, time-averaged</td><td>(1.0, 0.5, 0.3, 0.2)</td><td> $2 0 . 7 ^ { \circ }$ </td><td>(1.10, 0.39, −0.06, −0.26)</td><td>0.23</td><td>+0.049</td></tr><tr><td>Tool, K = 4, largest</td><td>(1.0, 0.5, 0.3, 0.2)</td><td></td><td>(1.32, 16.5, −3.25, −13.5)</td><td>0.82</td><td></td></tr><tr><td>Math, step 10  $( \rho \approx 0 . 9 8 )$ </td><td>(1.0, 0.3)</td><td></td><td>(2.94, −2.01)</td><td>0.69</td><td>-0.228</td></tr></table>

Corollary 2 (Near-duplicate rewards). Let $\Sigma _ { n }$ be a sequence of correlation matrices whose smallest eigenvalue $\lambda _ { 1 } ^ { ( n ) }$ is simple and tends to zero, with unit eigenvector ${ \bf u } _ { n }  { \bf u }$ , while the other eigenvalues stay above some $\begin{array} { r } { \underline { { \lambda } } > 0 . ~ I f \mathbf { u } ^ { \top } \mathbf { w } \neq 0 , } \end{array}$ , then $\mathbf { v } _ { n } / \lVert \mathbf { v } _ { n } \rVert \to \mathrm { s i g n } ( \mathbf { u } ^ { \top } \mathbf { w } ) \mathbf { u }$ . In particular, if under the same hypotheses $\rho _ { i j } \to 1$ for one pair $i \neq j$ , then $\mathbf { u } = \pm ( \mathbf { e } _ { i } - \mathbf { e } _ { j } ) / \sqrt { 2 }$ , where $\mathbf { e } _ { i }$ is the i-th standard basis vector, and the whitened weights approach a multiple of $\mathbf { e } _ { i } - \mathbf { e } _ { j }$ whenever $w _ { i } \neq w _ { j }$ , so the lighter-weighted of the two rewards is reversed.

For $K > 2$ we summarize the sign pattern of v with an inversion index,

$$
\psi ( \Sigma , \mathbf { w } ) : = \frac { \operatorname* { m a x } \{ 0 , - \operatorname* { m i n } _ { k } v _ { k } \} } { \operatorname* { m a x } _ { k } v _ { k } } ,\tag{5}
$$

which is zero when all whitened weights are nonnegative and approaches one when two rewards become collinear (the pair case of Corollary 2). The denominator is positive because $\mathbf { w } > 0$ and $\mathbf { w } ^ { \top } \mathbf { v } = \mathbf { w } ^ { \top } \Sigma ^ { - 1 / 2 } \mathbf { w } > 0$ force some $v _ { k }$ to be positive. For $K = 2 , \psi < 1$ because $v _ { 1 } + v _ { 2 } = ( a + b ) / \sqrt { 1 + \rho } > 0 ;$ for $K > 2$ it can exceed one. In practice we compute v and ψ from $\mathbf { v } _ { t } = \mathbf { W } _ { t } \mathbf { w }$ , which includes the eigenvalue floor of Section 4.

Table 1 lists v and $\psi$ for our settings. At Σ<sup>¯</sup> , WAS-PO places negative weights on length (−0.15 at $K = 3 )$ and on length and structure (−0.06 and −0.26 at $K = 4 )$ , because both are largely predicted by accuracy and format. In all three tool-calling settings, $\psi \leq 0 . 2 3$ at Σ<sup>¯</sup> . A negative weight on length means that WAS-PO penalizes length beyond what accuracy and format predict; it does not by itself make the policy shorter. The estimate was sometimes much worse conditioned: its format–length entry reached 1.000 during the threeand four-reward runs (Table 13). If every pair were at its largest logged value, WAS-PO would put a large positive weight on format and a large negative weight on length at $K = 3 \ ( \psi = 1 . 0 1 )$ , and on structure and length at $K = 4 \ ( \psi = 0 . 8 2 )$ . None of the three- or four-reward runs collapsed, unlike the math runs; Section 5.4 discusses why.

## 3.4 Estimation Error

In practice, Σ is estimated during training from N rollouts per batch.

Assumption 1. An estimate Σ<sup>ˆ</sup> of Σ from N rollouts satisfies $\mathbb { E } \lVert \hat { \boldsymbol { \Sigma } } - \boldsymbol { \Sigma } \rVert _ { F } ^ { 2 } \leq c K ^ { 2 } / ( N - 1 )$ for a constant c.

Under bivariate normality, the asymptotic variance of a sample correlation is $( 1 - \rho ^ { 2 } ) ^ { 2 } / N \leq 1 / N$ , which gives $c \approx 1$ . For skewed binary rewards this variance can be several times larger, and correlation among the $G$ completions of a prompt can reduce the efective sample size by up to a factor of G. We use $c = 4$ as a working value and have not verified it for our rewards.

Theorem 1 (Whitening estimation error). Let Σ be symmetric, let $\hat { \Sigma }$ be a random symmetric matrix satisfying Assumption 1, and suppose that almost surely all eigenvalues $o f \Sigma$ and Σ<sup>ˆ</sup> are at least $\lambda _ { 0 } > 0$ . Then

$$
\mathbb { E } \big \| ( \hat { \boldsymbol { \Sigma } } ^ { - 1 / 2 } - \boldsymbol { \Sigma } ^ { - 1 / 2 } ) \mathbf { w } \big \| ^ { 2 } \leq \frac { c K ^ { 2 } \| \mathbf { w } \| ^ { 2 } } { 4 \lambda _ { 0 } ^ { 3 } ( N - 1 ) } .\tag{6}
$$

The proof (Appendix B) bounds the Fréchet derivative of $\pmb { \Sigma } \mapsto \pmb { \Sigma } ^ { - 1 / 2 }$ with the Daleckii–Krein formula (Bhatia, 1997, Section V.3) and integrates it along the segment from Σ to Σ<sup>ˆ</sup> . Every matrix on that segment has eigenvalues at least $\lambda _ { 0 }$ because the smallest eigenvalue is a concave function of a symmetric matrix.

We compare (6) with the size of the correction, $\mathcal { E } : = \| \Sigma ^ { - 1 / 2 } \mathbf { w } - \mathbf { w } \| ^ { 2 }$ , which is 0.112, 0.251 and 0.360 for $K = 2 , 3 , 4$ . WAS-PO estimates Σ from all rollouts in a batch $( N = 3 2$ in our main runs) and smooths the estimate with an exponential moving average (EMA) that gives weight 1 − β to the newest batch. If successive batch estimates were independent with mean Σ, the EMA started from the first batch would multiply the variance of each entry at step t by

$$
f _ { t } = \beta ^ { 2 ( t - 1 ) } + \frac { 1 - \beta } { 1 + \beta } \big ( 1 - \beta ^ { 2 ( t - 1 ) } \big ) .
$$

This factor tends to $( 1 - \beta ) / ( 1 + \beta ) = 1 / 1 9 9$ at $\beta = 0 . 9 9$ , but equals 0.38 at $t = 5 0 , 0 . 1 4$ at $t = 1 0 0$ and 0.023 at $t = 2 0 0$ ; its average over 500 steps is 0.105. Appendix C evaluates the bound under this idealization, with $\lambda _ { 0 }$ set to the smallest eigenvalue of Σ<sup>¯</sup> . At $K = 3$ and $N = 3 2$ , the stationary bound is 0.26, about the size of the correction, and at $t = 1 0 0$ it is 7.3. The bound is therefore too loose to rule out estimation noise comparable to the correction for much of training, and it does not apply when the estimate approaches singularity, as it did at $K = 3$ and $K = 4$ . Its value is qualitative: the bound grows as $K ^ { 2 } \lambda _ { \operatorname* { m i n } } ^ { - 3 } ,$ so adding near-redundant rewards (small $\lambda _ { \operatorname* { m i n } } )$ makes the whitening matrix much harder to estimate, and it falls with the efective number of rollouts behind the estimate.

## 4 Method

Each training iteration of WAS-PO has three stages (Algorithm 1), plus an optional inversion check. From here on, $\hat { A } ^ { ( k ) }$ denotes the batch-normalized advantage defined below.

Batch-level normalization. Each reward is centered and scaled by its mean and standard deviation over all N rollouts in the batch. The covariance of A<sup>ˆ</sup> on a batch is then the batch correlation matrix $\mathbf { C } _ { t } ,$ which feeds the EMA estimate used for whitening. Batch-level normalization also changes the baseline and the scale relative to GDPO. Under GDPO a completion is compared with other completions to the same prompt, and each reward is scaled by its within-group standard deviation (Proposition 2(b)). Under batch-level normalization, completions to easy prompts tend to receive positive advantages, much as in the global advantage normalization of REINFORCE++ (Hu et al., 2025). The WAS-PO-Diag control below separates this change from whitening.

Correlation estimate. The Pearson correlation matrix $\mathbf { C } _ { t }$ of the raw rewards is computed over the batch and replaced by I if any reward has zero variance in the batch. It is smoothed as $\hat { \Sigma } _ { t } = \beta \hat { \Sigma } _ { t - 1 } + ( 1 - \beta ) \mathbf { C } _ { t }$ with $\beta = 0 . 9 9$ and $\hat { \mathbf { \boldsymbol { \Sigma } } } _ { 1 } = \mathbf { \boldsymbol { C } } _ { 1 }$

Whitening and aggregation. With the eigendecomposition $\hat { \pmb { \Sigma } } _ { t } = \mathbf { U } _ { t } \mathrm { d i a g } ( \eta _ { 1 } , \dots , \eta _ { K } ) \mathbf { U } _ { t } ^ { \top }$ , we set $\mathbf { W } _ { t } =$ $\mathbf { U } _ { t } \mathrm { d i a g } \big ( \mathrm { m a x } ( \eta _ { i } , 1 0 ^ { - 4 } ) ^ { - 1 / 2 } \big ) \mathbf { U } _ { t } ^ { \top }$ and compute

$$
\begin{array} { r } { \hat { A } _ { i } ^ { \mathrm { W A S - P O } } = \operatorname { B N } \big ( \mathbf { w } ^ { \top } \mathbf { W } _ { t } \hat { \mathbf { A } } \big ) _ { i } . } \end{array}\tag{7}
$$

The eigenvalue floor bounds $\| \mathbf { W } _ { t } \| _ { 2 }$ by 100. It has no efect when every eigenvalue of $\hat { \Sigma } _ { t }$ exceeds $1 0 ^ { - 4 }$ , and for two rewards it removes a reversal only when $\rho > 1 - 1 0 ^ { - 4 }$ and $a / b < 1 . 0 1 4 2$

Inversion check (optional). Since $\mathbf { v } _ { t } = \mathbf { W } _ { t } \mathbf { w }$ is available at every step, $\psi ( \hat { \Sigma } _ { t } , { \bf w } )$ can be logged at negligible cost, and $\mathbf { W } _ { t }$ can be replaced by I when needed (Section 6). The experiments reported here did not use this check.

Controls. WAS-PO-Diag zeros the of-diagonal entries of $\hat { \Sigma } _ { t }$ , which sets $\mathbf { W } _ { t } = \mathbf { I }$ because the diagonal is one. It runs through the same code path as WAS-PO, keeps batch-level normalization and drops whitening, so it difers from GDPO only in normalization and from WAS-PO only in whitening. WAS-PO-LW applies Ledoit–Wolf shrinkage (Ledoit & Wolf, 2004) to $\hat { \Sigma } _ { t }$ before inversion.

Algorithm 1 WAS-PO advantages for one training iteration   
Require: rewards $\mathbf { R } \in \mathbb { R } ^ { N \times K }$ , weights w, EMA state $\hat { \Sigma } .$ , decay $\beta$   
1: $\hat { A } _ { i } ^ { ( k ) } \gets ( R _ { i k } - \mathrm { m e a n } _ { i } R _ { i k } ) / ( \mathrm { s t d } _ { i } R _ { i k } + \epsilon )$   
2: C ← Corr(R), or C ← I if some column of R is constan   
3: Σ<sup>ˆ</sup> ← C at the first iteration, else $\hat { \Sigma }  \beta \hat { \Sigma } + ( 1 - \beta ) \mathbf { C }$   
4: $\mathbf { W }  \mathbf { U } \mathrm { d i a g } ( \operatorname* { m a x } ( \eta _ { i } , 1 0 ^ { - 4 } ) ^ { - 1 / 2 } ) \mathbf { U } ^ { \top }$ , where $\begin{array} { r } { \hat { \dot { \mathbf { \Sigma } } } = \mathbf { U } \dot { { \mathrm { d i a g } } } ( \eta ) \mathbf { U } ^ { \top } } \end{array}$   
5: $\begin{array} { r } { \tilde { A } _ { i } \gets \mathbf { w } ^ { \top } \mathbf { W } \hat { \mathbf { A } } _ { i } ; \hat { A } _ { i } ^ { \mathrm { W A S - P O } } \gets \mathrm { B N } ( \tilde { A } ) _ { i } } \end{array}$   
6: (optional) if the inversion check fires, recompute ${ \tilde { A } } _ { i }$ with W = I   
7: update $\pi _ { \theta }$ with the clipped surrogate objective using $\{ \hat { A } _ { i } ^ { \mathrm { W A S - P O } } \}$

## 5 Experiments

## 5.1 Setup

Training. We train Qwen2.5-1.5B-Instruct (Qwen Team, 2024) for 500 steps on the single-turn subset of the Berkeley Function Calling Leaderboard v3 data (BFCL; Patil et al., 2025), excluding the irrelevance category, using the copy distributed as llamastack/bfcl\_v3. The last 20% of this subset, in dataset order, is held out from training. Each step samples 8 prompts with $G = 4$ completions each, so $N = 3 2$ , with temperature 1.0 and at most 128 new tokens (all hyperparameters are in Table 14). GRPO, GDPO and WAS-PO share one implementation, $\mathrm { G D P O ^ { \circ } s }$ patched version of TRL 0.18, and difer only in how advantages are computed; rewards are gathered across the eight GPUs before advantages are formed. There is no KL penalty, and each batch is used for one update. Appendix E gives the prompt template and other details, including a reward-wiring pitfall that reduces multi-reward training to a single reward.

Rewards. The accuracy reward parses the first JSON object in a completion and compares it with the BFCL ground truth. It is 0 if no object parses, 0.1 if the function name is wrong, and otherwise 0.5 times the fraction of ground-truth argument names present plus 0.5 times the fraction of argument values that match. The format reward is 1 if a JSON object parses and 0 otherwise. The length reward is ex $\mathrm { { \cdot p \{ - \frac { 1 } { 2 } ( \ell / 5 1 2 - 1 ) ^ { 2 } \} } }$ for a completion of ℓ whitespace-separated words; with completions capped at 128 tokens, it lies between 0.61 and about 0.76 and increases with length. The structure reward $( K = 4 )$ is 1 for an object with a function name and nonempty arguments, 0.5 for a name alone, and 0 otherwise. We use GDPO’s conditional rewards: every secondary reward is set to zero when the accuracy reward is not positive, and we call such rewards gated. The gated format reward is therefore exactly the indicator that accuracy is positive, and the length and structure rewards are zero whenever it is zero. This shared gate is the main source of the correlations in Table 13.

Metrics. Our main metric is training AUC, the mean over all $T = 5 0 0$ steps of the weighted batch-mean reward,

$$
\mathrm { A U C } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbf { w } ^ { \top } \bar { \mathbf { r } } _ { t } ,
$$

where $\bar { \mathbf { r } } _ { t }$ is the batch-mean reward vector at step t. Its maximum is $\sum _ { k } w _ { k }$ (1.5, 1.8 and 2.0 for $K = 2 , 3 , 4 )$ so AUC values are not comparable across K. We also report held-out accuracy: the accuracy reward of the final checkpoint under greedy decoding on the first 500 prompts of the held-out 20%, the same prompts for every method, K and seed. Seeds are paired across methods. Training comparisons use exact two-sided Wilcoxon signed-rank tests on the paired diferences; with seven seeds the smallest attainable p-value is $2 / 1 2 8 \approx 0 . 0 1 6 .$ , and with three seeds it is 0.25. We report $\Delta$ as the mean paired diference and $d _ { z }$ as that mean divided by the standard deviation of the paired diferences. For held-out accuracy we kept per-method summaries rather than per-seed pairs, so we use unpaired two-sided Welch t-tests.

![](images/7619221f528f67f3bc450844e874e6ed46f816cd1d7d25a92c7fffa6967ffa7e.jpg)  
(a) Two rewards.

![](images/e6daf11fc9d36b26623dc39ffd057581e807c561bed23bff1cad251d95fc4214.jpg)  
(b) Three rewards.  
Figure 1: Training reward of GRPO, GDPO and WAS-PO. Curves and ±1 SD bands are computed over a subset of the seeds (three at $K = 2 )$ . The per-seed AUCs in Tables 7 and 5 cover all seven seeds.

## 5.2 Three and Four Rewards

Three rewards. Figure 1 shows training curves for two and three rewards. The rewards are accuracy $( w _ { 1 } = 1 . 0 )$ , format $( w _ { 2 } = 0 . 5 )$ and length $( w _ { 3 } = 0 . 3 )$ . Over seven seeds, training AUC was $1 . 0 1 3 \pm 0 . 0 1 1$ for GRPO, $1 . 0 8 9 \pm 0 . 0 0 6$ for GDPO and $1 . 1 3 0 \pm 0 . 0 0 7$ for WAS-PO (mean ± SD; Table 5, Figure 1b). WAS-PO was ahead of GDPO in every seed, by $\Delta = + 0 . 0 4 1$ $( p = 0 . 0 1 6$ $d _ { z } = 4 . 6 )$

Held-out accuracy was $0 . 3 9 7 \pm 0 . 0 7 6$ for GRPO, $0 . 4 4 6 \pm 0 . 0 1 2$ for GDPO and $0 . 4 3 8 \pm 0 . 0 1 1$ for WAS-PO (Table 9). WAS-PO was 0.008 below GDPO, which is not significant (Welch $p = 0 . 2 2 )$ and within the standard error of a single run’s mean accuracy on 500 prompts (at most about 0.022). The exact-match rate, the fraction of completions with accuracy reward 1, was 0.133 for WAS-PO and 0.131 for GDPO, and the format rate was 0.99 for both.

Four rewards. With the structure reward added $( w _ { 4 } = 0 . 2 )$ , training AUC over three seeds was 1.116±0.024 for $\mathrm { G R P O } , 1 . 2 5 7 \pm 0 . 0 0 7$ for GDPO and $1 . 3 0 6 \pm 0 . 0 0 9$ for WAS-PO (Table 6, Figure 4). WAS-PO was ahead in all three seeds, by +0.040, +0.050 and +0.058. Three seeds cannot reach significance in a Wilcoxon test (the smallest two-sided $p$ is 0.25), so we read this as consistent with the three-reward result rather than as independent evidence. Averaged over training, WAS-PO was ahead on every reward: accuracy $+ 0 . 0 1 4$ , format +0.038, length +0.011 and structure +0.032 (Table 12). Weighted by w, about two-thirds of this gain comes from the three gated rewards; the format mean, which equals the rate of completions with positive accuracy, was 0.871 under WAS-PO against 0.833 under GDPO. On the held-out set we recorded scores for two of the three seeds: WAS-PO scored 0.485 and 0.487, GDPO 0.517 and 0.515, and GRPO 0.521 and 0.517. With four rewards, WAS-PO was therefore about 0.03 below GDPO on held-out accuracy, and GRPO matched or exceeded GDPO. Relative to GDPO, the gain in training AUC was 3.8% with three rewards and 3.9% with four (Table 2; Figure 2 in Appendix D).

## 5.3 Two Rewards and Component Ablations

With accuracy and format only, training AUC over seven seeds was 0.986 for GRPO, 0.985 for GDPO and 0.970 for WAS-PO, which trailed GDPO in every seed (∆ = −0.014, $p = 0 . 0 1 6$ $d _ { z } = - 4 . 5 ;$ Table $^ { 7 , }$ Figure 1a). Held-out accuracy was $0 . 4 7 9 \pm 0 . 0 0 6$ for GRPO, $0 . 4 7 4 \pm 0 . 0 0 8$ for GDPO and $0 . 4 5 7 \pm 0 . 0 2 2$ for WAS-PO (Welch $p = 0 . 0 9$ against GDPO).

Changing how Σ is estimated did not close the gap (Table 3). EMA decays from $\beta = 0 . 5$ to 0.99, Ledoit–Wolf shrinkage (which lowered the largest estimated correlation from 0.87 to 0.37) and larger groups all left WAS-PO below GDPO. At $G = 8$ and at $G = 1 6$ , WAS-PO was below GDPO in each of the three seeds (Table 8).

Table 2: Training AUC by number of rewards. ∆ is the mean paired diference, WAS-PO minus GDPO, and Relative is ∆ divided by $\mathrm { G D P O ^ { \circ } s }$ mean, both computed from unrounded per-seed values. cond(Σ<sup>¯</sup> ) is the condition number of the time-averaged correlation estimate. The settings difer in which rewards they contain as well as in how many, and AUC is not comparable across rows. acc, fmt, len and struct are the accuracy, format, length and structure rewards.
<table><tr><td>K</td><td>Seeds</td><td>Rewards</td><td>GDPO</td><td>WAS-PO</td><td>Δ</td><td>Relative</td><td>cond(Σ)</td></tr><tr><td>2</td><td>7</td><td>acc, fmt</td><td>0.985</td><td>0.970</td><td>-0.014</td><td>-1.5%</td><td>4.6</td></tr><tr><td>3</td><td>7</td><td>acc, fmt, len</td><td>1.089</td><td>1.130</td><td>+0.041</td><td>+3.8%</td><td>12.2</td></tr><tr><td>4</td><td>3</td><td>acc, fmt, len, struct</td><td>1.257</td><td>1.306</td><td>+0.049</td><td>+3.9%</td><td>16.2</td></tr></table>

Table 3: Ablations with two rewards. Training AUC. ∆ is relative to GDPO trained with the same group size G on the same seeds (seeds 42, 123 and 456 for the three-seed rows; seed 42 for the decay sweep) and is computed from unrounded values. WAS-PO-Diag keeps batch-level normalization and removes whitening. The decay sweep is a separate set of single-seed runs with its own GDPO baseline (0.990). Its $\beta = 0 . 9 9$ run repeats the default configuration and scored 0.966, against 0.972 for seed 42 in Table 7, so reruns at a fixed seed vary by about 0.006.
<table><tr><td>Variant (G = 4 unless stated)</td><td>Seeds</td><td>GDPO, same G</td><td>AUC</td><td> $\Delta$ </td></tr><tr><td>WAS-PO</td><td>7</td><td>0.985</td><td>0.970</td><td>-0.014</td></tr><tr><td>WAS-PO-LW</td><td>7</td><td>0.985</td><td>0.968</td><td>-0.017</td></tr><tr><td>WAS-PO-Diag</td><td>3</td><td>0.986</td><td>0.965</td><td>-0.021</td></tr><tr><td> $\mathrm { W A S - P O } , G = 8 \ ( N = 6 4 )$ </td><td>3</td><td>1.003</td><td>0.984</td><td>-0.019</td></tr><tr><td> $\mathrm { W A S - P O } , G = 1 6 \ ( N = 1 2 8 )$ </td><td>3</td><td>1.011</td><td>0.993</td><td>-0.018</td></tr><tr><td> $\mathrm { W A S - P O } , \beta = 0 . 5$ </td><td>1</td><td>0.990</td><td>0.967</td><td>-0.023</td></tr><tr><td> $\mathrm { W A S - P O } , \beta = 0 . 7$ </td><td>1</td><td>0.990</td><td>0.970</td><td>-0.020</td></tr><tr><td> $\mathrm { W A S - P O } , \beta = 0 . 9$ </td><td>1</td><td>0.990</td><td>0.955</td><td>-0.035</td></tr><tr><td> $\mathrm { W A S - P O } , \beta = 0 . 9 5$ </td><td>1</td><td>0.990</td><td>0.955</td><td>-0.035</td></tr><tr><td> $\mathrm { W A S - P O } , \beta = 0 . 9 9$ </td><td>1</td><td>0.990</td><td>0.966</td><td>-0.025</td></tr></table>

WAS-PO-Diag scored 0.965 on seeds 42, 123 and 456, while WAS-PO scored 0.972 on the same seeds. The gap (0.007, three seeds) is about the size of the rerun variation in Table 3. It weakly suggests that, with two rewards, the deficit relative to GDPO comes from batch-level normalization rather than from whitening. The theory does not point to whitening as the cause either. At Σ<sup>¯</sup> , $\psi = 0 ;$ the estimate started at 0.87, above $\rho ^ { \star } = 0 . 8 0$ , and stayed above it only for the first few dozen steps (Figure 3, top).

In an earlier math run, the format reward was a JSON-format check applied to math answers and was therefore always zero. With a constant reward, $\mathbf { C } _ { t } = \mathbf { I }$ at every step and WAS-PO reduced to WAS-PO-Diag. Its gap to GDPO was −0.004 over three seeds (Figure 3, bottom; Table 10), so batch-level normalization does not always cost much training reward.

## 5.4 ZCA-Inversion

We train the same model on the GSM8K training set (Cobbe et al., 2021), with a system prompt asking it to end with the answer in the form The answer is: X. The accuracy reward is 1 if the extracted answer matches the reference and 0 otherwise; the extractor takes a boxed answer if there is one and otherwise the last number in the completion. The format reward is 1 if the completion contains The answer is: or a boxed answer, and 0 otherwise. Both rewards are gated as in tool calling, with $\mathbf { w } = ( 1 . 0 , 0 . 3 ) \ ( \rho ^ { \star } = 0 . 5 5 )$ $N = 3 2$ , with at most 384 new tokens. Since the gated format reward is 1 exactly when the answer is correct and formatted, the batch correlation of the two binary rewards follows from their means $p _ { \mathrm { a c c } }$ and $p _ { \mathrm { f m t } } \colon$ $\rho = \sqrt { p _ { \mathrm { f m t } } ( 1 - p _ { \mathrm { a c c } } ) / ( p _ { \mathrm { a c c } } ( 1 - p _ { \mathrm { f m t } } ) ) }$ (Appendix A). From the means in Table 11, $\rho \approx 0 . 9 6$ under GDPO and 0.98 under WAS-PO at step 10, well above $\rho ^ { \star }$ . Under GDPO the means are equal at steps 50 and 500, so $\rho = 1$ . These values are approximate because the means are averaged over batches and seeds.

Over three seeds, training AUC was 1.050 for GRPO, 1.049 for GDPO and 0.821 for WAS-PO. Under WAS-PO the format reward fell from 0.666 at step 10 to 0.132 at step 500 while accuracy kept rising; under GDPO both rose. This matches the sign that Proposition 3 assigns to the format reward: whenever $\rho > \rho ^ { \star }$ the completions that are correct and unformatted receive the largest advantages. Because the extractor falls back to the last number, such completions are easy to produce, and the policy learned to produce them. As the format reward drops, the two rewards decouple: by the same formula, the batch correlation under WAS-PO was about 0.41 at step 50 and 0.19 at step 500. The EMA forgets slowly (half-life 69 steps at $\beta = 0 . 9 9 )$ : starting from 0.98, the estimate cannot fall below $\rho ^ { \star }$ before step 58 unless new batches are negatively correlated, so the reversal stayed in force while the format reward fell to 0.277 by step 50.

The three- and four-reward runs show that a nearly singular estimate does not by itself cause collapse. In those runs the format–length entry of the estimate reached 1.000; this probably happened early in training, since the estimate is initialized from the first batch and the two-reward trace (Figure 3) is highest at the start. Had every pair been at its maximum, ψ would have been 0.82–1.01. Among completions that pass the gate, however, the length reward varies only between 0.61 and about 0.76, so a reversal along that direction changes the reward means little. In the math run, by contrast, the reversed reward could be switched of entirely at no cost to accuracy. Whether reward models trained on overlapping data reach correlations like those of the math rewards is untested here; ensembles of such models are known to share errors (Eisenstein et al., 2023).

## 5.5 Why Does WAS-PO Raise the Weighted Training Reward?

Training AUC weights the raw rewards by w. By Proposition 2, neither method follows exactly that objective: GDPO weights reward k by $w _ { k } / \sigma _ { k } ( x )$ on each prompt, and WAS-PO follows $\omega ,$ which puts negative weight on length or structure at $\bar { \Sigma } .$ That WAS-PO nonetheless scores higher on ${ \bf w } ^ { \top } { \bf r }$ is therefore not explained by its applying the intended weights. At $K = 4$ , the per-reward means show that most of the gain is in the gated rewards, through a higher rate of completions that parse and score above zero (0.871 against 0.833), which also explains why every reward moved up together. Three explanations fit. First, whitening shifts weight toward accuracy relative to w $( v _ { 1 } / v _ { 2 }$ is 3.6 at $K = 3$ and 2.8 at $K = 4$ , against $w _ { 1 } / w _ { 2 } = 2 )$ , and every secondary reward is gated on accuracy being positive. Second, GDPO may spend gradient on noise. In a group where all four completions pass the gate, the format reward is constant and drops out, while the small length diferences among them (all within 0.61–0.76) are standardized to unit variance, so by Proposition 2(b) GDPO gives them a large weight on that prompt; batch-level normalization scales them by the much larger batch standard deviation. Third, batch-level normalization gives nonzero advantages to groups in which every completion scored the same, which GDPO does not. Comparing against WAS-PO-Diag and against batch-normalized summation with fixed weights v, both at $K \geq 3$ , would separate these explanations.

## 6 Practical Guidance

Items marked observed are supported by our experiments on one task; items marked proposed have not been tested.

Training reward is not held-out accuracy (observed). With three or four rewards, WAS-PO reached higher training reward than GDPO in every seed but equal or lower held-out accuracy, and GRPO matched or exceeded GDPO on held-out accuracy at $K = 2$ and $K = 4 .$ In these settings the secondary rewards were close to copies of the accuracy gate. On this evidence we do not recommend WAS-PO as a way to improve held-out accuracy; what we have shown is that it changes which weighted training objective is optimized.

Inspect the whitened weights (proposed). Log $\mathbf { v } _ { t }$ and $\psi ( \hat { \Sigma } _ { t } , { \bf w } )$ from the matrix that is actually inverted. Any positive ψ means some reward is being reversed, but in our runs $\psi$ did not separate the collapsed run (0.69) from the worst cases assembled from per-pair maxima in runs that did not collapse (0.82–1.01), so we give no threshold. What mattered was whether the policy could move along the reversed direction cheaply. Two simple guards are to clip negative entries of $\mathbf { v } _ { t }$ to zero, or to fall back to $\mathbf { W } _ { t } = \mathbf { I }$ when the batch mean of a reward with $\omega _ { k } < 0$ falls over a sustained window; neither has been tested. For two rewards, Proposition 3 gives the exact boundary $\rho ^ { \star } = 2 a b / ( a ^ { 2 } + b ^ { 2 } )$ for the correlation being whitened.

Screen before training (proposed). Log rewards for a few dozen steps, estimate Σ from the batches, and compute $\varphi _ { \mathbf { w } } , ~ \kappa _ { \mathbf { w } }$ and v. We give no threshold for the angle; at $9 . 6 ^ { \circ }$ WAS-PO still difered from GDPO, mainly through normalization (Table 3).

Merge near-duplicate rewards (follows from Corollary 2). Two rewards that measure nearly the same thing, such as a gated format reward that is almost always earned, or two reward models trained on similar data, should be merged or left unwhitened. With unequal weights, whitening reverses the lighter-weighted one; with two equally weighted rewards it reverses neither (Proposition 3). Our own three- and four-reward settings contain such near-duplicates (Table 13), so the gains reported there come from a configuration this item advises against.

## 7 Limitations

WAS-PO changes both normalization and aggregation relative to GDPO, and the control that separates them was run only with two rewards and three seeds. The gains at $K \geq 3$ are gains of WAS-PO as a whole, and Section 5.5 lists competing explanations that we have not tested.

The gains are in training reward only. Held-out accuracy did not improve and was lower than GDPO’s with four rewards. Training AUC is measured on temperature-1 samples from the training prompts, while held-out accuracy uses greedy decoding on new prompts, so a method that sharpens the policy faster can raise the first without the second. The held-out prompts come from the same BFCL subset as the training prompts, taken in dataset order, so their category mix may difer from the training portion, and we have no evidence about transfer to other tool-calling distributions or to the oficial BFCL evaluation.

The tool-calling results come from one task, one 1.5B model and a small batch (N = 32); the math runs have three seeds each. Each K uses one reward set, and moving from two to three rewards adds the length reward, so the number of rewards is confounded with which rewards are used. The four-reward setting has three seeds and held-out scores for two of them. Conditional reward gating produces much of the correlation we study, and our diagnostics use batch-level correlations, not the within-group correlations that govern GDPO.

The analysis relies on a stated reading of w (Proposition 1), on treating normalization statistics as constants (Proposition 2), and on Assumption 1 with an idealized EMA (Theorem 1). The inversion diagnostics rest on one collapsed run and two runs that did not collapse despite a nearly singular estimate.

## 8 Related Work

Group-based policy optimization. Shao et al. (2024) introduced GRPO, which normalizes a scalar reward within a group of completions and needs no learned value function. Liu et al. (2026) showed that GRPO’s summed reward maps distinct reward combinations to the same advantage, and proposed GDPO. On BFCL with Qwen2.5-1.5B they report 32.81% average accuracy for GDPO against 30.18% for GRPO over five runs, with an evaluation protocol that difers from ours. Liu et al. (2025) analyze biases introduced by the length and standard-deviation normalization terms in GRPO, and Hu et al. (2025) argue for normalizing advantages over the whole batch. WAS-PO uses batch-level normalization in the spirit of the latter and adds a correction for correlation between rewards, which these methods do not consider.

Multiple rewards in language model alignment. Bai et al. (2022) train with a preference model for helpfulness and harmlessness, and Touvron et al. (2023) combine separate helpfulness and safety reward models with a threshold rule rather than a weighted sum. Wu et al. (2023) combine several fine-grained reward models with fixed weights, and Dai et al. (2024) treat helpfulness and harmlessness as a constrained problem. Ramé et al. (2023) interpolate the weights of policies fine-tuned on diferent rewards to trace a Pareto front. Coste et al. (2024) and Eisenstein et al. (2023) combine reward models in ensembles and find that models trained on similar data share errors, and Moskovitz et al. (2024) find that the correlation between component reward models afects where overoptimization sets in. We instead ask how correlation changes what a fixed weight vector does.

Multi-task loss weighting. Multi-task learning faces a similar problem with conflicting or redundant losses. Kendall et al. (2018) learn task weights from homoscedastic uncertainty, Sener & Koltun (2018) cast multi-task learning as multi-objective optimization, and Yu et al. (2020) project away conflicting gradient components. These methods operate on losses or parameter gradients, whereas WAS-PO changes per-sample advantages before any gradient is formed.

Whitening and covariance estimation. ZCA whitening was used in early work on the independent components of natural images (Bell & Sejnowski, 1997). Kessy et al. (2018) characterize the family of whitening transforms and the optimality of ZCA, and Huang et al. (2018) use ZCA whitening inside network layers and find that PCA whitening performs worse there. Estimating covariance from few samples is a classical dificulty (Ledoit & Wolf, 2004); in our two-reward runs, Ledoit–Wolf shrinkage did not close the gap to GDPO. Amari (1998) and Kakade (2001) precondition parameter updates with the Fisher information; WAS-PO can loosely be viewed as preconditioning in reward space instead.

Reward hacking. Skalse et al. (2022) define reward hacking formally, Gao et al. (2023) measure how optimizing a learned reward model overshoots the reward it approximates, and Pan et al. (2022) show that misspecified rewards are exploited more as agents become more capable. ZCA-Inversion difers from reward hacking in that it occurs even when every reward is correctly specified, because it arises from how the rewards are combined.

## 9 Conclusion

When normalized rewards are correlated, adding them with fixed weights does not apply those weights to the decorrelated components of the reward signal. We quantified the discrepancy, derived the objectives that GDPO and whitened aggregation follow under fixed normalization statistics, and gave an exact two-reward condition, with an extension to near-duplicate rewards, under which whitening reverses a reward. In tool calling with three or four correlated rewards, WAS-PO reached higher training reward than GDPO in every seed, but its held-out accuracy was not higher, and with four rewards it was about 0.03 lower (two seeds). In our two-reward setting, GDPO was better. With two nearly identical rewards, WAS-PO lowered the lighterweighted reward, as the sign condition predicts. The open questions are how much of the training-reward gain comes from whitening rather than batch-level normalization, why it does not carry over to held-out accuracy, and whether either finding holds on other tasks and larger models.

## Broader Impact Statement

Multi-reward training is used to combine helpfulness, format and safety objectives. ZCA-Inversion is relevant to safety because whitening two highly correlated rewards with unequal weights can train the policy to lower the lighter-weighted one, which could be a safety reward. Proposition 3 gives the exact condition for two rewards in terms of the correlation being whitened, but the estimate actually inverted can difer from the population value (Table 13), and negatively correlated pairs are never reversed. We ran no experiments with reward models or safety rewards. We advise against whitening a safety reward unless its weight in ω is constrained to be nonnegative. We release no new datasets or models; all resources used are publicly available.

## Code Availability

Code is available at https://github.com/lleye/waspo.

## References

Shun-ichi Amari. Natural gradient works eficiently in learning. Neural Computation, 10(2):251–276, 1998. doi: 10.1162/089976698300017746.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Anthony J. Bell and Terrence J. Sejnowski. The “independent components” of natural scenes are edge filters. Vision Research, 37(23):3327–3338, 1997. doi: 10.1016/S0042-6989(97)00121-1.

Rajendra Bhatia. Matrix Analysis, volume 169 of Graduate Texts in Mathematics. Springer, New York, 1997.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Thomas Coste, Usman Anwar, Robert Kirk, and David Krueger. Reward model ensembles help mitigate overoptimization. In International Conference on Learning Representations, 2024.

Juntao Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. Safe RLHF: Safe reinforcement learning from human feedback. In International Conference on Learning Representations, 2024.

Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D’Amour, DJ Dvijotham, Adam Fisch, Katherine Heller, Stephen Pfohl, Deepak Ramachandran, Peter Shaw, and Jonathan Berant. Helping or herding? Reward model ensembles mitigate but do not eliminate reward hacking. arXiv preprint arXiv:2312.09244, 2023.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 10835–10866, 2023.

Jian Hu, Jason Klein Liu, Haotian Xu, and Wei Shen. REINFORCE++: Stabilizing critic-free policy optimization with global advantage normalization. arXiv preprint arXiv:2501.03262, 2025.

Lei Huang, Dawei Yang, Bo Lang, and Jia Deng. Decorrelated batch normalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018. doi: 10.1109/CVPR.2018.00089.

Sham M. Kakade. A natural policy gradient. In Advances in Neural Information Processing Systems, volume 14, pp. 1531–1538, 2001.

Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7482–7491, 2018.

Agnan Kessy, Alex Lewin, and Korbinian Strimmer. Optimal whitening and decorrelation. The American Statistician, 72(4):309–314, 2018. doi: 10.1080/00031305.2016.1277159.

Olivier Ledoit and Michael Wolf. A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88(2):365–411, 2004. doi: 10.1016/S0047-259X(03)00096-4.

Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, Jan Kautz, and Pavlo Molchanov. GDPO: Group reward-decoupled normalization policy optimization for multi-reward RL optimization. arXiv preprint arXiv:2601.05242, 2026.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding R1-Zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

Ted Moskovitz, Aaditya K. Singh, DJ Strouse, Tuomas Sandholm, Ruslan Salakhutdinov, Anca Dragan, and Stephen Marcus McAleer. Confronting reward model overoptimization with constrained RLHF. In International Conference on Learning Representations, 2024.

Alexander Pan, Kush Bhatia, and Jacob Steinhardt. The efects of reward misspecification: Mapping and mitigating misaligned models. In International Conference on Learning Representations, 2022.

Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The Berkeley Function Calling Leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 48371–48392, 2025.

Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Alexandre Ramé, Guillaume Couairon, Corentin Dancette, Jean-Baptiste Gaya, Mustafa Shukor, Laure Soulier, and Matthieu Cord. Rewarded soups: Towards Pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards. In Advances in Neural Information Processing Systems, volume 36, pp. 71095–71134, 2023.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In Advances in Neural Information Processing Systems, volume 31, 2018.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward hacking. In Advances in Neural Information Processing Systems, volume 35, pp. 9460–9471, 2022.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3–4):229–256, 1992. doi: 10.1007/BF00992696.

Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A. Smith, Mari Ostendorf, and Hannaneh Hajishirzi. Fine-grained human feedback gives better rewards for language model training. In Advances in Neural Information Processing Systems, volume 36, 2023.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems, volume 33, pp. 5824–5836, 2020.

## A Proofs for Section 3

Proposition 1. Since $\Sigma \succ 0 .$ its symmetric positive definite square root and the inverse of that root exist and are unique. With $\mathbf { z } = \pmb { \Sigma } ^ { - 1 / 2 } \hat { \mathbf { A } }$ we have $\mathrm { C o v } ( { \bf z } ) = \Sigma ^ { - 1 / 2 } \Sigma \Sigma ^ { - 1 / 2 } = { \bf I }$ and $\hat { \mathbf { A } } = \pmb { \Sigma } ^ { 1 / 2 } \mathbf { z }$ . Then $\mathbf { w } ^ { \top } \hat { \mathbf { A } } = \mathbf { w } ^ { \top } \pmb { \Sigma } ^ { 1 / 2 } \mathbf { z } = ( \pmb { \Sigma } ^ { 1 / 2 } \mathbf { w } ) ^ { \top } \mathbf { z }$ by symmetry, and $\mathbf { w } ^ { \top } \pmb { \Sigma } ^ { - 1 / 2 } \hat { \mathbf { A } } = \mathbf { w } ^ { \top } \pmb { \Sigma } ^ { - 1 / 2 } \pmb { \Sigma } ^ { 1 / 2 } \mathbf { z } = \mathbf { w } ^ { \top } \mathbf { z }$ . The two agree for every w if and only if $\boldsymbol { \Sigma } ^ { 1 / 2 } \mathbf { w } = \mathbf { w }$ for every w, that is, ${ \pmb { \Sigma } } ^ { 1 / 2 } = \mathbf { I } .$ , which is equivalent to $\Sigma = \mathbf { I } .$ □

Corollary 1. For a fixed vector m, $\mathrm { V a r } ( \mathbf { m } ^ { \top } \hat { \mathbf { A } } ) = \mathbf { m } ^ { \top } \pmb { \Sigma } \mathbf { m }$ . With m = w and unit diagonal this is $\begin{array} { r } { \| \mathbf { w } \| ^ { 2 } + \sum _ { j \neq k } w _ { j } w _ { k } \rho _ { j k } } \end{array}$ . With $\begin{array} { r } { \mathbf { m } = \pmb { \Sigma } ^ { - 1 / 2 } \mathbf { w } \mathrm { ~ i t ~ } \mathrm { i s ~ } \mathbf { w } ^ { \top } \pmb { \Sigma } ^ { - 1 / 2 } \pmb { \Sigma } \pmb { \Sigma } ^ { - 1 / 2 } \mathbf { w } = \| \mathbf { w } \| ^ { 2 } . \ \bigtriangledown } \end{array}$

Proposition $\mathbf { 2 } , \mathbf { \Gamma } ( \mathrm { a } )$ With the stated constants, $\begin{array} { r } { \tilde { A } _ { i } = s ^ { - 1 } \sum _ { k } v _ { k } ( r _ { i } ^ { ( k ) } - \mu _ { k } ) / \sigma _ { k } = s ^ { - 1 } \big ( { \omega } ^ { \top } { \bf r } _ { i } - \gamma _ { 0 } \big ) } \end{array}$ , where $\omega = \mathbf { D } ^ { - 1 } \mathbf { v }$ and $\begin{array} { r } { \gamma _ { 0 } = \sum _ { k } v _ { k } \mu _ { k } / \sigma _ { k } } \end{array}$ does not depend on $y _ { i } .$ For any $q$ that does not depend on $\theta ,$ the score-function identity gives $\mathbb { E } _ { y \sim \pi _ { \theta } } [ q ( y ) \nabla _ { \theta }$ log $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { y } \mid \boldsymbol { x } ) ] = \nabla _ { \boldsymbol { \theta } } \mathbb { E } _ { \boldsymbol { y } \sim \pi _ { \boldsymbol { \theta } } } [ q ( \boldsymbol { y } ) ]$ , and $\mathbb { E } _ { y \sim \pi _ { \theta } } [ \nabla _ { \theta } \log \pi _ { \theta } ( y \mid x ) ] = 0$ Applying both to $q = \omega ^ { \top } \mathbf { r }$ and to the constant $\gamma _ { 0 } .$ , and averaging over prompts, gives $\mathbb { E } [ \tilde { A } \nabla _ { \theta } \log \pi _ { \theta } ] =$ $s ^ { - 1 } \nabla _ { \theta } J _ { \omega }$ . Setting $\boldsymbol { \Sigma } = \mathbf { I }$ gives $\mathbf { D } ^ { - 1 } \mathbf { w }$ for the batch-normalized sum without whitening. (b) For a fixed prompt $\begin{array} { r } { x , \tilde { A } _ { i } = s ^ { - 1 } \bigl ( \sum _ { k } ( w _ { k } / \sigma _ { k } ( x ) ) r _ { i } ^ { ( k ) } - \gamma _ { 0 } ( x ) \bigr ) } \end{array}$ with $\gamma _ { 0 } ( x )$ independent of $y _ { i } .$ , and the same argument applied conditionally on $x ,$ followed by averaging over prompts, gives the stated objective. Using the group mean that includes $y _ { i }$ itself, with $\sigma _ { k } ( x )$ still held fixed, adds a factor $( 1 - 1 / G )$ common to all rewards, because the other completions are independent of $y _ { i } ;$ it does not change the direction of the gradient. □

Proposition 3. For $\Sigma = { \bigl ( } { } _ { 2 } ^ { 1 } _ { 1 } ^ { \rho } { \bigr ) }$ the eigenvectors are $\mathbf { u } _ { \pm } = ( 1 , \pm 1 ) / \sqrt { 2 }$ with eigenvalues $1 \pm \rho ,$ so $\pmb { \Sigma } ^ { - 1 / 2 } =$ $( 1 + \rho ) ^ { - 1 / 2 } \mathbf { u } _ { + } \mathbf { u } _ { + } ^ { \top } + ( 1 - \rho ) ^ { - 1 / 2 } \mathbf { u } _ { - } \mathbf { u } _ { - } ^ { \top }$ and

$$
v _ { 1 } = \frac { a + b } { 2 \sqrt { 1 + \rho } } + \frac { a - b } { 2 \sqrt { 1 - \rho } } , \qquad v _ { 2 } = \frac { a + b } { 2 \sqrt { 1 + \rho } } - \frac { a - b } { 2 \sqrt { 1 - \rho } } .
$$

Both terms of $v _ { 1 }$ are nonnegative and the first is positive, so $v _ { 1 } > 0$ . Both terms in the expression for $v _ { 2 }$ are nonnegative, so $v _ { 2 } < 0$ if and only if $( a - b ) ^ { 2 } ( 1 + \rho ) > ( a + b ) ^ { 2 } ( 1 - \rho )$ , that is, $\rho \big > ( a - b ) ^ { \bar { 2 } } + ( a + b ) ^ { 2 } \big >$ $( a + b ) ^ { 2 } - ( a - b ) ^ { 2 }$ , which simplifies to $2 \rho ( a ^ { 2 } + b ^ { 2 } ) > 4 a b ,$ or $\rho > 2 a b / ( a ^ { 2 } + b ^ { 2 } )$ . If $a = b ,$ the second terms vanish and $v _ { 1 } = v _ { 2 } > 0$ . If $a > b ,$ as $\rho  1$ the first terms stay bounded while the second terms grow as $( a - b ) / ( 2 \sqrt { 1 - \rho } )$ , so $| v _ { 2 } | \to \infty$ and $v _ { 2 } / v _ { 1 }  - 1$ . □

Corollary 2. Write $\begin{array} { r } { \mathbf { v } _ { n } = \sum _ { m } ( \boldsymbol { \lambda } _ { m } ^ { ( n ) } ) ^ { - 1 / 2 } ( \mathbf { u } _ { n , m } ^ { \top } \mathbf { w } ) \mathbf { u } _ { n , m } } \end{array}$ in an orthonormal eigenbasis of $\Sigma _ { n } .$ , with $\mathbf { u } _ { n , 1 } = \mathbf { u } _ { n }$ The terms with $m \geq 2$ have norm at most $\boldsymbol { \underline { { \lambda } } ^ { - 1 / 2 } } \| \mathbf { w } \|$ , while the first has norm $( \lambda _ { 1 } ^ { ( n ) } ) ^ { - 1 / 2 } | \mathbf { u } _ { n } ^ { \top } \mathbf { w } |  \infty$ because $\mathbf { u } _ { n } ^ { \top } \mathbf { w }  \mathbf { u } ^ { \top } \mathbf { w } \ne 0$ . Dividing by $\| \mathbf { v } _ { n } \|$ gives the limit. For the pair case let $\mathbf { d } = ( \mathbf { e } _ { i } - \mathbf { e } _ { j } ) / \sqrt { 2 }$ Then d $^ { \intercal } \Sigma _ { n } \mathbf { d } = 1 - \rho _ { i j }  0$ , while $\begin{array} { r } { \mathbf { \tilde { d } } ^ { \top } \Sigma _ { n } \mathbf { \tilde { d } } \geq \underline { { \lambda } } \sum _ { m > 2 } ( \mathbf { u } _ { n , m } ^ { \top } \mathbf { d } ) ^ { 2 } } \end{array}$ . Hence $( \mathbf { u } _ { n } ^ { \top } \mathbf { d } ) ^ { 2 }  1$ , so ${ \bf u } = \pm { \bf d }$ and $\mathbf { u } ^ { \top } \mathbf { w } = \pm ( w _ { i } - w _ { j } ) / \sqrt { 2 }$ , which is nonzero when $w _ { i } \neq w _ { j } . ~ \bigsqcup$

Correlation of gated binary rewards. Let A and F be binary with $F \leq A$ , as for an accuracy reward and a format reward gated on it, with means $p _ { \mathrm { a c c } }$ and $p _ { \mathrm { f m t } }$ . Then $\mathbb { E } [ A F ] = p _ { \mathrm { f m t } } .$ , so $\operatorname { C o v } ( A , F ) = p _ { \mathrm { f m t } } ( 1 - p _ { \mathrm { a c c } } )$ and dividing by $\sqrt { p _ { \mathrm { a c c } } ( 1 - p _ { \mathrm { a c c } } ) p _ { \mathrm { f m t } } ( 1 - p _ { \mathrm { f m t } } ) } \mathrm { ~ g i v e s ~ } \rho = \sqrt { p _ { \mathrm { f m t } } ( 1 - p _ { \mathrm { a c c } } ) / ( p _ { \mathrm { a c c } } ( 1 - p _ { \mathrm { f m t } } ) ) } . \mathrm { ~ } \Omega$

## B Proof of Theorem 1

Step 1. Let $h ( \mathbf { X } ) = \mathbf { X } ^ { - 1 / 2 }$ on symmetric matrices with all eigenvalues at least $\lambda _ { 0 } ,$ and write $\mathbf { X } = \mathbf { Q } \pmb { \Lambda } \mathbf { Q } ^ { \top }$ with eigenvalues $\lambda _ { 1 } , \ldots , \lambda _ { K } . \mathrm { ~ B y ~ }$ the Daleckii–Krein formula (Bhatia, 1997, Section V.3), the Fréchet derivative is $\mathrm { d } h ( \mathbf { X } ) [ \mathbf { H } ] = \mathbf { Q } \big ( \mathbf { M } \circ ( \mathbf { Q } ^ { \top } \mathbf { H } \mathbf { Q } ) \big ) \mathbf { Q } ^ { \top }$ , where ◦ is the entrywise product and

$$
m _ { i i } = - \frac { 1 } { 2 } \lambda _ { i } ^ { - 3 / 2 } , \qquad m _ { i j } = \frac { \lambda _ { i } ^ { - 1 / 2 } - \lambda _ { j } ^ { - 1 / 2 } } { \lambda _ { i } - \lambda _ { j } } = - \frac { 1 } { \sqrt { \lambda _ { i } \lambda _ { j } } \left( \sqrt { \lambda _ { i } } + \sqrt { \lambda _ { j } } \right) } \quad ( \lambda _ { i } \ne \lambda _ { j } ) ,
$$

with $\begin{array} { r } { m _ { i j } = - \frac { 1 } { 2 } \lambda _ { i } ^ { - 3 / 2 } } \end{array}$ when $\lambda _ { i } = \lambda _ { j }$ . Since $\sqrt { \lambda _ { i } \lambda _ { j } } \ge \lambda _ { 0 }$ and $\sqrt { \lambda _ { i } } + \sqrt { \lambda _ { j } } \ge 2 \sqrt { \lambda _ { 0 } } ,$ , every $| m _ { i j } | \leq 1 / ( 2 \lambda _ { 0 } ^ { 3 / 2 } )$ The Frobenius norm is invariant under orthogonal conjugation and $\begin{array} { r } { \| { \bf M } \circ { \bf B } \| _ { F } \leq \operatorname* { m a x } _ { i j } | m _ { i j } | \| { \bf B } \| _ { F } , } \end{array}$ , so $\| \mathrm { d } h ( \mathbf { X } ) [ \mathbf { H } ] \| _ { F } \leq \| \mathbf { H } \| _ { F } / ( 2 \lambda _ { 0 } ^ { 3 / 2 } )$

Step 2. Let $\mathbf { H } = \hat { \Sigma } - \Sigma$ and $\mathbf { X } _ { \xi } = \pmb { \Sigma } + \xi \mathbf { H }$ for $\xi \in [ 0 , 1 ]$ . The smallest eigenvalue is concave on symmetric matrices, so $\lambda _ { \operatorname* { m i n } } ( \mathbf { X } _ { \xi } ) \geq ( 1 - \xi ) \lambda _ { \operatorname* { m i n } } ( \Sigma ) + \xi \lambda _ { \operatorname* { m i n } } ( \hat { \Sigma } ) \geq \lambda _ { 0 }$ . Hence h is continuously diferentiable along the segment and $\begin{array} { r } { \hat { \boldsymbol { \Sigma } } ^ { - 1 / 2 } - \boldsymbol { \Sigma } ^ { - 1 / 2 } = \int _ { 0 } ^ { 1 } \mathrm { d } h ( \mathbf { X } _ { \xi } ) [ \mathbf { H } ] d \xi } \end{array}$ , which by Step 1 gives $\lVert \hat { \boldsymbol { \Sigma } } ^ { - 1 / 2 } - \boldsymbol { \Sigma } ^ { - 1 / 2 } \rVert _ { F } \leq \lVert \mathbf { H } \rVert _ { F } / ( 2 \lambda _ { 0 } ^ { 3 / 2 } )$ .

Step 3. $\begin{array} { r } { \| ( \hat { { \boldsymbol \Sigma } } ^ { - 1 / 2 } - { \boldsymbol \Sigma } ^ { - 1 / 2 } ) { \mathbf w } \| ^ { 2 } \leq \| \hat { { \boldsymbol \Sigma } } ^ { - 1 / 2 } - { \boldsymbol \Sigma } ^ { - 1 / 2 } \| _ { F } ^ { 2 } \| { \mathbf w } \| ^ { 2 } \leq \| \mathbf { H } \| _ { F } ^ { 2 } \| { \mathbf w } \| ^ { 2 } / ( 4 \lambda _ { 0 } ^ { 3 } ) } \end{array}$ almost surely. Taking expectations and applying Assumption 1 gives (6). □

EMA variance factor. Unrolling the recursion $\hat { \Sigma } _ { t } = \beta \hat { \Sigma } _ { t - 1 } + ( 1 - \beta ) \mathbf { C } _ { t }$ from $\hat { \mathbf { \Sigma } } _ { 1 } = \mathbf { C } _ { 1 }$ gives

$$
\hat { \mathbf { \boldsymbol { \Sigma } } } _ { t } = \beta ^ { t - 1 } \mathbf { \boldsymbol { \mathbf { C } } } _ { 1 } + \left( 1 - \beta \right) \sum _ { j = 0 } ^ { t - 2 } \beta ^ { j } \mathbf { \boldsymbol { \mathbf { C } } } _ { t - j } .
$$

If the $\mathbf { C } _ { t }$ are independent with mean Σ and entrywise variance $\tau ^ { 2 }$ , each entry of $\hat { \Sigma } _ { t }$ has variance $\tau ^ { 2 } \big [ \beta ^ { 2 ( t - 1 ) } +$ $\begin{array} { r } { ( 1 - \beta ) ^ { 2 } \sum _ { i = 0 } ^ { t - 2 } \beta ^ { 2 j } \big ] = \tau ^ { 2 } f _ { t } } \end{array}$ , with $f _ { t }$ as in Section 3.4. Bias in the batch estimates, which is $O ( 1 / N )$ for sample correlations, is not reduced by the EMA.

## C Evaluating the Bound

Table 4 evaluates (6) with $c = 4$ and $\lambda _ { 0 }$ set to the smallest eigenvalue of $\bar { \Sigma } ,$ for one batch of N rollouts and for the EMA under the idealization of Section 3.4, both at its stationary factor 1/199 and at step $t = 1 0 0 \ ( f _ { 1 0 0 } = 0 . 1 4 1 )$ . We use $N = 3 2$ in the main runs and $N = 1 2 8$ in the $G = 1 6$ ablation. Using the time-averaged $\lambda _ { \mathrm { m i n } }$ is itself an idealization: the estimate became nearly singular at times in the three- and four-reward runs, and single-batch estimates can be worse, so for them Theorem 1 holds only with a much smaller $\lambda _ { 0 } .$ , or not usefully at all. At the largest two-reward estimate of 0.869, $\lambda _ { \operatorname* { m i n } } = 0 . 1 3 1$ , and the constant $K ^ { 2 } \| \mathbf { w } \| ^ { 2 } / \lambda _ { \operatorname* { m i n } } ^ { 3 }$ rises from 111.8 to about 2224. Step 3 of the proof is tight only when $\hat { \boldsymbol { \Sigma } } ^ { - 1 / 2 } - \boldsymbol { \Sigma } ^ { - 1 / 2 }$ is close to rank one with w in its row space.

Table 4: Bound (6) against the size of the correction $\varepsilon .$
<table><tr><td></td><td></td><td></td><td></td><td colspan="2">One batch</td><td colspan="2">EMA, stationary</td><td colspan="2">EMA, t = 100</td></tr><tr><td>K</td><td> $\lambda _ { \mathrm { m i n } }$ </td><td>E</td><td> $K ^ { 2 } \| \mathbf { w } \| ^ { 2 } / \lambda _ { \operatorname* { m i n } } ^ { 3 }$ </td><td> $N = 3 2$ </td><td> $N = 1 2 8$ </td><td> $N = 3 2$ </td><td> $N = 1 2 8$ </td><td> $N = 3 2$ </td><td>N = 128</td></tr><tr><td>2</td><td>0.355</td><td>0.112</td><td>111.8</td><td>3.61</td><td>0.880</td><td>0.018</td><td>0.004</td><td>0.508</td><td>0.124</td></tr><tr><td>3</td><td>0.196</td><td>0.251</td><td>1601.7</td><td>51.7</td><td>12.6</td><td>0.260</td><td>0.063</td><td>7.29</td><td>1.78</td></tr><tr><td>4</td><td>0.196</td><td>0.360</td><td>2932.5</td><td>94.6</td><td>23.1</td><td>0.475</td><td>0.116</td><td>13.3</td><td>3.26</td></tr></table>

## D Full Results

![](images/2d939d1a9e743d5b0f219a5874ec7f8c7fe3f5e57f0c3d79f7bccd7b11d29149.jpg)

![](images/1ff31674ece4e1e352d1ac68969c0cbd72b64321302e81fe7e839d00412b0adc.jpg)

![](images/5c5ced137560030a572e9fe84d921dece293df0f350e9ad2ef6b6fb4b309cbd9.jpg)  
Figure 2: Training AUC for two, three and four rewards $( \mathrm { m e a n } \pm \mathrm { S D }$ over seeds). Each panel has its own axis because AUC is not comparable across $K$

Table 5: Training AUC with three rewards, seven seeds. Exact two-sided Wilcoxon $p = 0 . 0 1 6 ; d _ { z } = 4 . 6$
<table><tr><td>Seed</td><td>GRPO</td><td>GDPO</td><td>WAS-PO</td><td> $\Delta$ </td></tr><tr><td>42</td><td>1.0303</td><td>1.0958</td><td>1.1252</td><td>+0.0294</td></tr><tr><td>123</td><td>0.9995</td><td>1.0862</td><td>1.1386</td><td>+0.0524</td></tr><tr><td>456</td><td>1.0028</td><td>1.0903</td><td>1.1213</td><td>+0.0310</td></tr><tr><td>789</td><td>1.0188</td><td>1.0947</td><td>1.1391</td><td>+0.0444</td></tr><tr><td>1011</td><td>1.0059</td><td>1.0798</td><td>1.1303</td><td>+0.0505</td></tr><tr><td>2024</td><td>1.0110</td><td>1.0863</td><td>1.1246</td><td>+0.0383</td></tr><tr><td>9999</td><td>1.0225</td><td>1.0879</td><td>1.1315</td><td>+0.0436</td></tr><tr><td>Mean ± SD</td><td> $1 . 0 1 3 \pm 0 . 0 1 1$ </td><td> $1 . 0 8 9 \pm 0 . 0 0 6$ </td><td> $1 . 1 3 0 \pm 0 . 0 0 7$ </td><td>+0.041</td></tr></table>

Table 6: Training AUC with four rewards, three seeds. $d _ { z } = 5 . 7 ;$ the smallest attainable two-sided Wilcoxon p with three seeds is 0.25.
<table><tr><td>Seed</td><td>GRPO</td><td>GDPO</td><td>WAS-PO</td><td> $\Delta$ </td></tr><tr><td>42</td><td>1.1049</td><td>1.2633</td><td>1.3036</td><td>+0.0403</td></tr><tr><td>123</td><td>1.0995</td><td>1.2488</td><td>1.2987</td><td>+0.0499</td></tr><tr><td>456</td><td>1.1439</td><td>1.2579</td><td>1.3154</td><td>+0.0575</td></tr><tr><td>Mean ± SD</td><td>1.116 ± 0.024</td><td>1.257 ± 0.007</td><td>1.306 ± 0.009</td><td>+0.049</td></tr></table>

Table 7: Training AUC with two rewards, seven seeds. Exact two-sided Wilcoxon $p = 0 . 0 1 6 ; d _ { z } = - 4 . 5 .$
<table><tr><td>Seed</td><td>GRPO</td><td>GDPO</td><td>WAS-PO</td><td> $\Delta$ </td></tr><tr><td>42</td><td>0.9903</td><td>0.9870</td><td>0.9721</td><td>-0.0149</td></tr><tr><td>123</td><td>0.9759</td><td>0.9826</td><td>0.9722</td><td>-0.0104</td></tr><tr><td>456</td><td>0.9918</td><td>0.9873</td><td>0.9713</td><td>-0.0160</td></tr><tr><td>789</td><td>0.9898</td><td>0.9833</td><td>0.9727</td><td>-0.0106</td></tr><tr><td>1011</td><td>0.9825</td><td>0.9870</td><td>0.9691</td><td>-0.0179</td></tr><tr><td>2024</td><td>0.9825</td><td>0.9814</td><td>0.9683</td><td>-0.0131</td></tr><tr><td>9999</td><td>0.9908</td><td>0.9849</td><td>0.9668</td><td>-0.0181</td></tr><tr><td> $\mathrm { M e a n } \pm \mathrm { S D }$ </td><td> $0 . 9 8 6 \pm 0 . 0 0 6$ </td><td> $0 . 9 8 5 \pm 0 . 0 0 2$ </td><td> $0 . 9 7 0 \pm 0 . 0 0 2$ </td><td>-0.014</td></tr></table>

Table 8: Training AUC at larger group sizes, two rewards, three seeds. Eight prompts per step, so N = 64 at G = 8 and $N = 1 2 8$ at $G = 1 6 \AA$
<table><tr><td>G</td><td>Seed</td><td>GRPO</td><td>GDPO</td><td>WAS-PO</td><td> $\Delta$ </td></tr><tr><td>8</td><td>42</td><td>1.0021</td><td>1.0061</td><td>0.9849</td><td rowspan="4">-0.0212 -0.0139 -0.0220</td></tr><tr><td>8</td><td>123</td><td>0.9964</td><td>0.9976</td><td>0.9837</td></tr><tr><td>8</td><td>456</td><td>1.0020</td><td>1.0050</td><td>0.9830</td></tr><tr><td>16</td><td>42</td><td>1.0125</td><td>1.0111</td><td>0.9938</td></tr><tr><td>16</td><td>123</td><td>1.0084</td><td>1.0064</td><td>0.9905</td><td rowspan="2">-0.0173 -0.0159 -0.0193</td></tr><tr><td>16</td><td>456</td><td>1.0169</td><td>1.0153</td><td>0.9960</td></tr></table>

Table 9: Held-out accuracy (accuracy reward of the final checkpoint under greedy decoding on 500 held-out prompts; mean ± SD over seeds). p is from an unpaired two-sided Welch t-test computed from these summaries. At K = 4, scores were recorded for two of three seeds.
<table><tr><td>Setting</td><td>GRPO</td><td>GDPO</td><td>WAS-PO</td><td>∆ vs. GDPO</td><td> $p$ </td></tr><tr><td>K = 2 (seven seeds)</td><td> $0 . 4 7 9 \pm 0 . 0 0 6$ </td><td> $0 . 4 7 4 \pm 0 . 0 0 8$ </td><td> $0 . 4 5 7 \pm 0 . 0 2 2$ </td><td>-0.017</td><td>0.09</td></tr><tr><td> $K = 3 \ \mathrm { ( s e v e n \ s e e d s ) }$ </td><td> $0 . 3 9 7 \pm 0 . 0 7 6$ </td><td> $0 . 4 4 6 \pm 0 . 0 1 2$ </td><td> $0 . 4 3 8 \pm 0 . 0 1 1$ </td><td>-0.008</td><td>0.22</td></tr><tr><td> $K = 4$  (two seeds)</td><td> $0 . 5 2 1 , 0 . 5 1 7$ </td><td> $0 . 5 1 7 , 0 . 5 1 5$ </td><td> $0 . 4 8 5 , 0 . 4 8 7$ </td><td>-0.030</td><td></td></tr></table>

Table 10: Training AUC in the earlier math run with a constant format reward, where WAS-PO reduces to WAS-PO-Diag (Section 5.3). This is not the run of Section 5.4.
<table><tr><td>Seed</td><td>GRPO</td><td>GDPO</td><td>WAS-PO</td><td> $\Delta$ </td></tr><tr><td>42</td><td>0.8077</td><td>0.8041</td><td>0.8085</td><td>+0.0044</td></tr><tr><td>123</td><td>0.7988</td><td>0.8120</td><td>0.7943</td><td>-0.0177</td></tr><tr><td>456</td><td>0.8075</td><td>0.7994</td><td>0.7995</td><td>+0.0001</td></tr><tr><td>Mean</td><td>0.805</td><td>0.805</td><td>0.801</td><td>-0.004</td></tr></table>

Table 11: Per-reward batch-mean training reward in the math setting (three seeds).
<table><tr><td>Step</td><td>WAS-PO acc</td><td>WAS-PO fmt</td><td>GDPO acc</td><td>GDPO fmt</td></tr><tr><td>10</td><td>0.675</td><td>0.666</td><td>0.691</td><td>0.675</td></tr><tr><td>50</td><td>0.697</td><td>0.277</td><td>0.775</td><td>0.775</td></tr><tr><td>500</td><td>0.806</td><td>0.132</td><td>0.863</td><td>0.863</td></tr></table>

Table 12: Per-reward training reward with four rewards, averaged over 500 steps (three seeds). Per-reward values were not logged for GRPO. The weighted sums of these means (1.262 and 1.305) agree with the AUCs in Table 6 to within 0.006.
<table><tr><td>Method</td><td>Accuracy</td><td>Format</td><td>Length</td><td>Structure</td></tr><tr><td>GDPO</td><td>0.521</td><td>0.833</td><td>0.528</td><td>0.831</td></tr><tr><td>WAS-PO</td><td>0.535</td><td>0.871</td><td>0.539</td><td>0.863</td></tr><tr><td>∆</td><td>+0.014</td><td>+0.038</td><td>+0.011</td><td>+0.032</td></tr></table>

<sub>�</sub>�<sup>�</sup> ��������������������������������������� �����������������������  
Table 13: The correlation estimate $\hat { \Sigma } _ { t }$ maintained by WAS-PO (seed 42 for K = 2, 3; three seeds for $K = 4 )$ . Mean and range are over the 500 training steps; Σ<sup>¯</sup> is the matrix of means, and its eigenvalues, cond(Σ<sup>¯</sup> ) and E are listed.
<table><tr><td>K</td><td>Pair</td><td>Mean ρ</td><td>Range</td><td>cond(Σ)</td><td>E</td></tr><tr><td colspan="2">2 acc-fmt eigenvalues</td><td>0.645 [0.355, 1.645]</td><td>[0.530, 0.869]</td><td>4.6</td><td>0.112</td></tr><tr><td>3</td><td>acc-fmt acc-len</td><td>0.643 0.642 0.804</td><td>[0.545, 0.828] [0.544, 0.827]</td><td>12.2</td><td>0.251</td></tr><tr><td colspan="2">eigenvalues 4</td><td></td><td>[0.196, 0.408, 2.396]</td><td></td><td></td></tr><tr><td rowspan="5"></td><td>acc-fmt</td><td>0.652</td><td>[0.561, 0.828]</td><td rowspan="5"></td><td rowspan="5">16.2 0.360</td></tr><tr><td>acc-len</td><td>0.651</td><td>[0.560, 0.827]</td></tr><tr><td>acc-struct</td><td>0.643</td><td>[0.551, 0.828]</td></tr><tr><td>fmt-len</td><td>0.804</td><td>[0.700, 1.000]</td></tr><tr><td>fmt-struct</td><td>0.796</td><td>[0.693, 1.000]</td></tr><tr><td colspan="2">len-struct eigenvalues</td><td>0.796 [0.196, 0.207, 0.420, 3.177]</td><td>[0.694, 1.000]</td><td></td><td></td></tr></table>

![](images/fee697f92ee31c862ebc7903be797b6d92cba1673d97b3e280473209f5ffe470.jpg)

![](images/c28fbdaceb6353bc62bef1a67e30ea366627bde26adf400069636456a59cbd54.jpg)  
Figure 3: Correlation estimate and reward diference (seed 42, logged every 10 steps). Blue: the EMA estimate of $\rho ( \mathrm { a c c } , \mathrm { f m t } )$ . Red: training reward of WAS-PO minus GDPO, labeled “advantage” in the legend but meaning a diference in reward. Top: tool calling with two rewards; the two series have correlation 0.44, but both trend over training, so we do not interpret this. Bottom: the earlier math run with a constant format reward, where the estimate is the identity and the diference fluctuates around zero.

![](images/53ebb20752f41b719657c64636b31e0d13d5475d74c0aa6e0e8dcf6ecc304257.jpg)  
Figure 4: Training reward with four rewards (three seeds, mean ± SD).

## E Hyperparameters and Implementation Details

Table 14: Training hyperparameters. Settings not listed are the defaults of TRL 0.18 and Transformers.
<table><tr><td></td><td>Tool, K = 2</td><td>Tool, K = 3</td><td>Tool, K = 4</td><td>Math</td></tr><tr><td>Model</td><td colspan="4">Qwen2.5-1.5B-Instruct</td></tr><tr><td>Data</td><td colspan="4">BFCL v3, single-turn, irrelevance excluded; first 80%</td></tr><tr><td>Group size G</td><td>4</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Prompts per step</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Rollouts per step N</td><td>32</td><td>32</td><td>32</td><td>32</td></tr><tr><td>Max new tokens</td><td>128</td><td>128</td><td>128</td><td>384</td></tr><tr><td>Max prompt length</td><td colspan="4">512 tokens</td></tr><tr><td>Training steps</td><td>500</td><td>500</td><td>500</td><td>500</td></tr><tr><td>Seeds</td><td>7</td><td>7</td><td>3</td><td>3</td></tr><tr><td>Reward weights</td><td>(1.0, 0.5)</td><td>(1.0, 0.5, 0.3)</td><td>(1.0, 0.5, 0.3, 0.2)</td><td>(1.0, 0.3)</td></tr><tr><td>Conditional (gated) rewards</td><td colspan="4">yes</td></tr><tr><td>EMA decay β</td><td colspan="4">0.99</td></tr><tr><td>Eigenvalue floor; €</td><td colspan="4">10−4; 10−4</td></tr><tr><td>KL coefficient</td><td colspan="4">0 (no reference model)</td></tr><tr><td>Learning rate</td><td colspan="4">1 × 10−⁶</td></tr><tr><td>Optimizer</td><td colspan="4">AdamW, linear decay, no warmup</td></tr><tr><td>Updates per batch</td><td colspan="4">1</td></tr><tr><td>Sampling</td><td colspan="4">temperature 1.0, top-p 1.0</td></tr><tr><td>Held-out decoding</td><td colspan="4">greedy</td></tr><tr><td>Precision</td><td colspan="4">bfloat16</td></tr><tr><td>Framework</td><td colspan="4">TRL 0.18 with the GDPO patch</td></tr><tr><td>Hardware</td><td colspan="4">8×H100</td></tr></table>

Prompts. Tool-calling prompts list the names and descriptions (truncated to 80 characters) of the first three available functions, then the user request (truncated to 300 characters), then the instruction to respond only with a JSON function call, followed by the example {"name": "function\_name", "arguments": {"param": "value"}}. The held-out evaluation uses the same content with the system message “You are a helpful assistant.” Ground-truth argument values in BFCL can list several acceptable values; the accuracy reward compares against the first, as a case-insensitive string match. The math system prompt asks the model to end its response with The answer is: X, where X is the numeric answer only.

Compute. The reported runs used about 120 GPU-hours, and preliminary and failed runs about 40 more.   
The G = 8 and G = 16 ablations also use eight prompts per step.

Reward wiring. A natural way to add several rewards to a GRPO trainer is to wrap all reward functions in one function that returns their weighted sum. The trainer then sees a single reward, and GRPO, GDPO and WAS-PO all run the same single-reward path, so no comparison between them is meaningful. All results here use an implementation that passes each reward function to the trainer separately, so that the reward tensor has one column per reward. We recommend checking that the reward tensor has K columns before comparing multi-reward methods.