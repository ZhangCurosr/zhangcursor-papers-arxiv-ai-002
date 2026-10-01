# OPTIMAL DESIGN FOR ACTIVE PREFERENCE LEARN-ING WITH BIASED LLM JUDGES

Zhongman Du<sup>1</sup> Huiming Zhang<sup>1</sup> Haodong Zhu<sup>1,2</sup> Baochang Zhang<sup>1,3</sup>

<sup>1</sup>Beihang University <sup>2</sup>Zhongguancun Academy

<sup>3</sup>Hangzhou Innovation Institute of Beihang University

## ABSTRACT

Learning from human preferences is central to large language model (LLM) alignment, but human preference annotation is costly. Active preference learning reduces this cost by selecting informative comparisons, and LLM judges can provide additional scalable feedback. However, the preferences of the judges may deviate from those of the target human population. Even after calibration on trusted reference data, active acquisition can shift the comparison distribution and expose residual judge bias. We therefore incorporate judge deviations into the acquisition design rather than relying on a separate calibration stage. Under joint estimation, comparisons that appear highly informative about the reward may also reflect judge bias and therefore provide less information about human preferences. To address this issue, we propose Nuisance-Adjusted Optimal Design (NAOD), a comparison-selection strategy that prioritizes policy-relevant target information after nuisance adjustment and uses the Frank–Wolfe algorithm for optimization. Theoretically, we establish a sharp conditional local asymptotic minimax lower bound on policy risk and construct an estimator that attains it. We further characterize the finite-sample cost of learning the nuisance representation and show that representation error can reverse an oracle design advantage. Finally, we validate these predictions experimentally and evaluate NAOD on Chatbot Arena data across 17 judges, 15 budget configurations, and 15 random cluster-level splits. NAOD reduces the mean regret of proxy policy by 29.1% relative to a matched target-information design, outperforms existing methods, and improves humanpreference prediction on held-out data.

## 1 INTRODUCTION

Preference feedback (Christiano et al., 2017) provides a standard way to align large language models (LLMs) with human preferences when desired behavior is difficult to specify directly (Ouyang et al., 2022; Rafailov et al., 2023). Reward models are trained on pairwise response comparisons and then used to guide downstream policy optimization (Stiennon et al., 2020). The resulting policy therefore depends not only on how the reward model is trained, but also on which comparisons are selected for feedback. When human annotation is costly or limited, active preference learning and experimental design make more efficient use of a fixed annotation budget by selecting informative comparisons (Muldrew et al., 2024; Mukherjee et al., 2024). Recent methods further incorporate reward-model information (Lin et al., 2026) or downstream policy objectives (Feng et al., 2025) into their acquisition criteria.

A complementary approach to reducing annotation cost is to obtain preference feedback from LLM judges (Lee et al., 2024). However, their judgments can differ systematically from the preferences of the target human population (Zheng et al., 2023). A natural response is to calibrate the judge against trusted human data before relying on its feedback (Polo et al., 2025). However, calibration can deteriorate under distribution shift (Ovadia et al., 2019; Chan et al., 2020), while active acquisition itself changes the comparison distribution by selecting which pairs receive feedback (Farquhar et al., 2021). Residual judge biases that cancel under the reference calibration distribution can therefore become unbalanced on the acquired distribution. Thus, pair selection determines both how much information is collected and which judge errors can influence the learned reward.

![](images/a0a8b6b73402f98a604a0c07fcc1a2316c4d15287a6b0ba2ae866236a9175d94.jpg)  
Figure 1: Overview of the NAOD acquisition and estimation pipeline. Historical human–judge feedback is used to construct the frozen target and judge-deviation representations. Before current outcomes are observed, NAOD selects comparisons using nuisance-adjusted target information weighted by policy relevance. After selection, trusted human and judge feedback are combined in a joint estimator to update the target reward and induced policy.

It is additionally difficult to model these judge deviations during estimation. The human reward and judge-specific deviation must now be estimated jointly and distinguished from one another. On some selected pairs, changes in the human reward and changes in the judge deviation can have nearly the same effect on label probabilities. A design that treats all apparent reward information as target information can therefore overstate the precision with which the human target can be estimated.

For acquisition, what matters is not the raw information about reward parameters, but the infor mation about the human target that remains after judge-specific deviations have been accounted for. Even this remaining information is not equally important across reward directions, because some reward errors substantially alter downstream policy decisions whereas others have little effect on the induced policy. Reward-model accuracy alone therefore need not reflect downstream policy performance (Gao et al., 2023; Wen et al., 2025; Frick et al., 2025). We therefore propose Nuisance-Adjusted Optimal Design (NAOD), a design strategy that selects comparisons using nuisance-adjusted target information weighted by downstream policy relevance. Figure 1 provides an overview of the NAOD pipeline from pre-acquisition design to post-selection estimation.

A sharp conditional local asymptotic minimax lower bound for policy risk is established, together with an estimator that attains it. The resulting risk coincides with the nuisance-adjusted, policyweighted objective optimized by NAOD. The same target–nuisance cross-information controls both residual exposure and the loss of target information. Further, when the judge-deviation representation is learned, a design-dependent risk term appears that can reverse oracle design rankings.

Empirically, controlled experiments validate the predicted policy-risk behavior and the designranking reversal. On Chatbot Arena, across 17 judges, 15 budget configurations, and 15 random cluster-level splits, NAOD reduces mean proxy policy regret by 29.1% relative to the matched Target-info design and improves held-out human-preference prediction. Its gains tend to be larger when target–nuisance coupling is stronger, consistent with the mechanism predicted by our theory.

Our contributions can be summarized as follows:

• We formulate active preference learning with biased LLM judges as a joint target–nuisance design problem and characterize how acquisition exposes judge residuals.

• We introduce NAOD, establish its sharp local minimax interpretation, and characterize the additional risk from learning the judge-deviation representation.

• We develop a practical comparison-selection algorithm for finite candidate pools, validate our theoretical risk predictions in synthetic experiments, and evaluate NAOD on Chatbot Arena with held-out human evaluation.

## 2 RELATED WORK

Active preference acquisition and nuisance-aware design. Active preference learning reduces feedback cost by selecting informative comparisons. Prior work studies adaptive pairwise comparisons (Maystre & Grossglauser, 2017), preference-based reinforcement learning (Lee et al., 2021), query-efficient learning from pairwise preferences (Wu & Sun, 2024), and subset selection for reliable LLM ranking (Liusie et al., 2024). Information-based methods use Fisher information for reward modeling (Shen et al., 2025) or D-optimal design with randomized Frank–Wolfe optimization (Thekumparampil et al., 2025), while other work incorporates downstream policy relevance or reward uncertainty into acquisition (Hu et al., 2024; Sun et al., 2025b). When observations jointly inform target and nuisance parameters, acquisition must also account for nuisance uncertainty (Sloman et al., 2024). Our work combines policy-aware preference acquisition with nuisance-aware experimental design under LLM-judge feedback.

LLM judges and calibration. LLM judges provide scalable evaluation but exhibit systematic biases, including position effects (Wang et al., 2024), stylistic preferences (Feuer et al., 2025), and broader recurring biases across human and model evaluators (Chen et al., 2024; Ye et al., 2025). Related work also studies pairwise feedback from systematically biased evaluators (Tang et al., 2025). Existing mitigation approaches characterize limits of debiasing with limited high-quality labels (Dorner et al., 2025), fine-tune evaluators on debiased data (Park et al., 2024), optimize prompts (Zhou et al., 2024), calibrate pairwise prediction distributions (Li et al., 2025), or use human cali bration data to correct judge estimates and allocate calibration samples (Lee et al., 2026). Our focus is instead on how judge discrepancies should change which comparisons are acquired for human reward estimation.

## 3 PROBLEM SETTING

The Bradley-Terry model of human preferences. Let $z = ( s , y _ { A } , y _ { B } )$ denote an ordered comparison, where s is a prompt and $y _ { A } , y _ { B }$ are two candidate responses. Let $\phi ( s , y ) \in \mathbb { R } ^ { d }$ be a fixed d-dimensional feature map. We model the reward as linear in this feature representation, $r _ { \theta } ( s , y ) = \phi ( s , y ) ^ { \top } \theta .$ , where $\theta \in \mathbb { R } ^ { d }$ is the reward parameter. We define the comparison feature $X ( z ) = \phi ( s , y _ { A } ) - \phi ( s , y _ { B } )$ . Let $Y \in \{ 0 , 1 \}$ denote the human preference label, with $Y = 1$ indicating that $y _ { A }$ is preferred to $y _ { B }$ . Under the Bradley-Terry model (Bradley & Terry, 1952), the preference probability is determined by the reward difference between the two responses,

$$
p _ { \theta } ( z ) = \mathbb { P } _ { \theta } ( Y = 1 \mid z ) = \sigma ( r _ { \theta } ( s , y _ { A } ) - r _ { \theta } ( s , y _ { B } ) ) = \sigma \big ( X ( z ) ^ { \top } \theta \big ) ,\tag{1}
$$

where $\sigma ( u ) = ( 1 + e ^ { - u } ) ^ { - 1 }$ is the sigmoid function. We write $\theta ^ { \star }$ for the reward parameter of the target human population and $p ^ { \star } = p _ { \theta }$ ⋆ for its preference probability.

Performance metric. We evaluate the learned reward through the policy it induces. Let $\mu _ { e }$ denote the evaluation distribution over prompts. For each prompt s, let $\mathcal { A } ( s )$ be a finite set of candidate responses and let $\pi _ { \mathrm { r e f } } ( \cdot \mid s )$ be a reference policy on $\mathcal { A } ( s )$ with full support. Given a regularization parameter $\tau > 0$ , let $\pi _ { \theta } ( \cdot \mid s )$ denote the optimizer of the KL-regularized reward objective (Azar et al., 2012; Liu et al., 2024). Its closed form is

$$
\pi _ { \boldsymbol { \theta } } ( y \mid s ) = \frac { \pi _ { \mathrm { r e f } } ( y \mid s ) \exp ( r _ { \boldsymbol { \theta } } ( s , y ) / \tau ) } { \sum _ { y ^ { \prime } \in A ( s ) } \pi _ { \mathrm { r e f } } ( y ^ { \prime } \mid s ) \exp ( r _ { \boldsymbol { \theta } } ( s , y ^ { \prime } ) / \tau ) } .\tag{2}
$$

The corresponding optimal regularized value is

$$
F ( \theta ) = \tau \mathbb { E } _ { s \sim \mu _ { e } } \log \sum _ { y \in A ( s ) } \pi _ { \mathrm { r e f } } ( y \mid s ) \exp ( r _ { \theta } ( s , y ) / \tau ) .\tag{3}
$$

When the target reward parameter is $\theta ^ { \star }$ , deploying the policy induced by θ incurs the loss

$$
D _ { F } ( \theta ^ { \star } , \theta ) = F ( \theta ^ { \star } ) - F ( \theta ) - \nabla F ( \theta ) ^ { \top } ( \theta ^ { \star } - \theta ) = \tau \mathbb { E } _ { s \sim \mu _ { c } } { \mathrm { K L } } ( \pi _ { \theta } ( \cdot \mid s ) \parallel \pi _ { \theta ^ { \star } } ( \cdot \mid s ) ) .\tag{4}
$$

See Appendix A.1 for derivations. Thus, our performance metric measures error through its effect on the induced policy.

LLM judge and active acquisition. For a fixed LLM judge and annotation protocol, let $q ( z ) \in$ [0, 1] denote its preference probability for comparison $z .$ Its preferences need not match those of the target human population. We define the judge discrepancy on the probability scale as $\delta ( z ) =$ $q ( z ) - p ^ { \star } ( z )$ . Let $\nu$ denote a reference distribution over comparisons and let $Z \sim \nu .$ Conventional probability calibration with respect to the target human population requires $\mathbb { E } _ { \boldsymbol { \nu } } [ \boldsymbol { Y } \mid \boldsymbol { q } ( \boldsymbol { Z } ) ] = \boldsymbol { q } ( \boldsymbol { Z } )$ (Guo et al., 2017; Futami & Fujisawa, 2024; Chidambaram & Ge, 2025). Under the target preference model in equation 1, this is equivalent to

$$
\mathbb { E } _ { \nu } [ \delta ( Z ) \mid q ( Z ) ] = 0 .
$$

Thus, under $\nu ,$ calibration constrains the conditional mean of the judge discrepancy given $q ( Z )$ However, active acquisition changes the comparison distribution. Let $\xi$ denote the acquired comparison distribution and assume $\xi \ll \nu ,$ so that the density ratio $\omega ( z ) = \bar { d } \xi / d \nu ( z )$ is well defined.

Acquisition-dependent residual exposure. Probability calibration is defined under the reference distribution ν and is not guaranteed to hold under the acquired distribution ξ (Park et al., 2020). For reward estimation, the relevant issue is how judge discrepancies align with the reward-feature directions. Under the acquired distribution $\xi ,$ define the residual contribution to the target score as $b _ { \xi } = \mathbb { E } _ { \xi } [ \delta ( Z ) X ( Z ) ]$ ], and define $b _ { \nu } = \mathbb { E } _ { \nu } [ \delta ( Z ) X ( Z ) ]$ ] analogously under the reference distribution. This residual-feature moment can be viewed as a multiaccuracy-style residual moment over linear reward-feature tests (Kim et al., 2019; Kern et al., 2024). A change of measure gives

$$
b _ { \xi } = b _ { \nu } + \mathbb { E } _ { \nu } \left[ \{ \omega ( Z ) - 1 \} \delta ( Z ) X ( Z ) \right] .\tag{5}
$$

The second term captures how acquisition reweights judge discrepancies across reward-feature directions. Calibration does not imply $b _ { \nu } = 0$ , and even when $b _ { \nu } = 0$ , acquisition can yield $b _ { \xi } \neq 0$ In that case, treating judge feedback as target feedback shifts the population score away from zero at $\theta ^ { \star }$ , potentially moving the fitted reward away from the human target. Appendix B.1 gives the corresponding parameter and policy-regret consequences.

Thus, active acquisition can expose judge discrepancies that are balanced under the reference distribution. This motivates accounting for judge deviations directly in the acquisition design rather than relying on reference calibration alone.

## 4 NUISANCE-ADJUSTED OPTIMAL DESIGN

## 4.1 JOINT TARGET-NUISANCE MODEL

We model human and judge preferences jointly. Let $W ( z ) \in \mathbb { R } ^ { r }$ be a fixed representation of judge deviations, where $r$ is its dimension, and let $a \in \mathbb { R } ^ { r }$ be the associated nuisance parameter. We model the fixed judge preference probability $q ( z )$ introduced in Section 3 using a Bradley–Terry model augmented with a structured judge-specific component (Movva et al., 2026),

$$
\begin{array} { r } { q _ { \theta , a } ( z ) = \sigma \big ( X ( z ) ^ { \top } \theta + W ( z ) ^ { \top } a \big ) . } \end{array}\tag{6}
$$

Human labels follow the preference model $p _ { \theta }$ in equation 1, while $W ( z ) ^ { \top }$ a represents the difference between judge and human log odds. For the theory in this section, we assume that $q ( z ) = q _ { \theta ^ { \star } , a ^ { \star } } ( z )$ for some unknown nuisance parameter $a ^ { \star }$ . Thus, $W ( z ) ^ { \top } a ^ { \star }$ parameterizes the judge-human discrepancy on the log-odds scale, while $\delta ( z )$ measures the corresponding discrepancy on the probability scale. Throughout this section, $W$ is treated as fixed. Section 5.1 studies the additional error when W is learned from historical human and judge feedback.

For the approximate-design formulation, we first consider a finite support of $L$ comparison types $z _ { 1 } , \dots , z _ { L }$ . Write $X _ { i } = \breve { X } ( z _ { i } )$ and $W _ { i } = W ( z _ { i } )$ . At type $i ,$ we collect $n _ { i }$ judge labels, each indicating a preference for the first response with probability $q _ { \theta , a } ( z _ { i } )$ . An independent trusted sample contains $n _ { c }$ human labels, whose preference probabilities are given by $p _ { \theta }$ . Conditional on the comparison features, all current labels are independent Bernoulli observations. Let $\begin{array} { r } { n = \sum _ { i = 1 } ^ { L } n _ { i } } \end{array}$ be the total number of judge labels. We represent the asymptotic allocation by ${ \boldsymbol \xi } = ( \xi _ { 1 } , \dots , \bar { \xi _ { L } } )$ , where $n _ { i } / n \to \xi _ { i } , \xi _ { i } \geq 0$ , and $\textstyle \sum _ { i = 1 } ^ { L } \xi _ { i } = 1$ . We also assume that $n _ { c } / n \to \kappa$ for some $\kappa \in [ 0 , \infty )$ We condition on the historical data used to construct $W$ and select the allocation. These data are independent of the current labels.

## 4.2 EFFECTIVE TARGET INFORMATION

Joint estimation must separate target-score variation from nuisance-score variation (Sloman et al., 2024; Barker & Kavalieris, 2001). Let $k = d + r$ be the joint parameter dimension. Define $\gamma =$ $( \theta ^ { \top } , \dot { a } ^ { \top } ) ^ { \top } \in \mathbb { R } ^ { k }$ and the corresponding feature $v _ { i } = ( X _ { i } ^ { \top } , \mathbf { \bar { W } } _ { i } ^ { \top } ) ^ { \top }$ . Fix a local analysis center $\gamma _ { 0 } =$ $( \theta _ { 0 } ^ { \top } , a _ { 0 } ^ { \top } ) ^ { \top }$ , where $\theta _ { 0 }$ and $a _ { 0 }$ are the target and nuisance components, respectively. The Bernoulli variance of the judge label at this center is $t _ { i } = q _ { \theta _ { 0 } , a _ { 0 } } ( z _ { i } ) \{ 1 - q _ { \theta _ { 0 } , a _ { 0 } } ( z _ { i } ) \}$ , and one judge label contributes Fisher information $J _ { i } ~ = ~ t _ { i } v _ { i } v _ { i } ^ { \top }$ Let $H _ { c }$ denote the limiting Fisher information per trusted label at $\theta _ { 0 } .$ Let $X _ { c }$ denote a random comparison feature in the trusted sample. Then $H _ { c } =$ $\mathbb { E } [ \sigma ^ { \prime } ( X _ { c } ^ { \top } \theta _ { 0 } ) X _ { c } \breve { X } _ { c } ^ { \top } ]$ . Let $A _ { \kappa } ( \xi ) , D _ { \xi }$ , and $C _ { \xi }$ denote the target, nuisance, and cross-information blocks, respectively. The joint information normalized by the judge sample size is

$$
M _ { \kappa } ( \xi ) = \kappa \mathrm { d i a g } ( H _ { c } , 0 _ { r \times r } ) + \sum _ { i = 1 } ^ { L } \xi _ { i } J _ { i } = \left( \begin{array} { c c } { A _ { \kappa } ( \xi ) } & { C _ { \xi } } \\ { C _ { \xi } ^ { \top } } & { D _ { \xi } } \end{array} \right) .\tag{7}
$$

Here, $A _ { \kappa } ( \boldsymbol { \xi } )$ is the target-information block, $D _ { \xi }$ is the nuisance-information block, and $C _ { \xi }$ is the target–nuisance cross-information block. Specifically, $\begin{array} { r } { A _ { \kappa } ( \xi ) = { \kappa } H _ { c } + \sum _ { i = 1 } ^ { L } \xi _ { i } t _ { i } X _ { i } X _ { i } ^ { \top } , C _ { \xi } = } \end{array}$ $\scriptstyle \sum _ { i = 1 } ^ { L } \xi _ { i } t _ { i } X _ { i } W _ { i } ^ { \top }$ , and $\begin{array} { r } { D _ { \xi } \ = \ \sum _ { i = 1 } ^ { L } \xi _ { i } t _ { i } W _ { i } W _ { i } ^ { \top } } \end{array}$ . When $M _ { \kappa } ( \boldsymbol { \xi } )$ is positive definite, the effective information for the human reward parameter is

$$
I _ { \mathrm { e f f } , \kappa } ( \boldsymbol { \xi } ) = A _ { \kappa } ( \boldsymbol { \xi } ) - C _ { \boldsymbol { \xi } } D _ { \boldsymbol { \xi } } ^ { - 1 } C _ { \boldsymbol { \xi } } ^ { \top } .\tag{8}
$$

Estimating the nuisance parameter reduces the information available for θ from $A _ { \kappa } ( \boldsymbol { \xi } )$ to the Schur complement $I _ { \mathrm { e f f } , \kappa } ( \boldsymbol { \xi } )$ , with information loss $C _ { \xi } D _ { \xi } ^ { - 1 } C _ { \xi } ^ { \top }$ (Fewster & Jupp, 2013). The inverse of $I _ { \mathrm { e f f } , \kappa } ( \boldsymbol { \xi } )$ is the target block of $M _ { \kappa } ( \boldsymbol { \xi } ) ^ { - 1 }$

The cross-information also connects this adjustment to the residual exposure in equation 5. Holding the human parameter at $\theta _ { 0 }$ , define $\begin{array} { r } { b _ { \xi } ( a ) = \sum _ { i = 1 } ^ { L } \xi _ { i } \{ q _ { \theta _ { 0 } , a } ( z _ { i } ) - p _ { \theta _ { 0 } } ( z _ { i } ) \} X _ { i } } \end{array}$ . Let $C _ { \xi } ( a )$ denote the cross-information evaluated at $( \theta _ { 0 } , a )$ , so that $\dot { C } _ { \xi } ( a _ { 0 } ) = \dot { C } _ { \xi }$ . Differentiation gives the exact Jacobian identity

$$
\frac { \partial b _ { \xi } ( a ) } { \partial a ^ { \top } } = C _ { \xi } ( a ) .\tag{9}
$$

The same cross-information governs the sensitivity of the exposed target score and the information loss from nuisance estimation. See Appendices ${ \tt A . 2 }$ and ${ \mathrm { A } } . 3$ for the statistical foundations of the joint and effective information, and Appendices B.1 and B.3 for the residual-exposure and target– nuisance coupling results.

## 4.3 POLICY-WEIGHTED DESIGN AND MINIMAX RISK

Effective information describes how precisely the human reward can be estimated after nuisance adjustment. The policy regret in equation 4 determines which estimation errors matter. Define the policy curvature at $\theta _ { 0 }$ by

$$
G _ { 0 } = \nabla ^ { 2 } F ( \theta _ { 0 } ) = \frac { 1 } { \tau } \mathbb { E } _ { s \sim \mu _ { e } } \operatorname { C o v } _ { y \sim \pi _ { \theta _ { 0 } } ( \cdot | s ) } [ \phi ( s , y ) ] .
$$

For $h \in \mathbb { R } ^ { d }$ , the local regret satisfies $\begin{array} { r } { D _ { F } ( \theta _ { 0 } , \theta _ { 0 } + h ) = \frac { 1 } { 2 } h ^ { \top } G _ { 0 } h + o ( \| h \| ^ { 2 } ) } \end{array}$ as $h  0$ . Thus, $G _ { 0 }$ weights reward-parameter directions by their effect on the induced policy. For a reward estimator $\widehat { \theta } _ { n }$ , we measure policy risk by $\mathbb { E } [ D _ { F } ( \theta ^ { \star } , \widehat { \theta } _ { n } ) ]$ ]. Let $G _ { e } = \mathrm { d i a g } ( G _ { 0 } , 0 _ { r \times r } )$ , and let tr denote the matrix trace. Combining policy curvature with effective target information gives

$$
\Phi _ { \kappa } ( \xi ) = \frac 1 2 \operatorname { t r } \bigl ( G _ { 0 } I _ { \mathrm { e f f } , \kappa } ( \xi ) ^ { - 1 } \bigr ) = \frac 1 2 \operatorname { t r } \bigl ( G _ { e } M _ { \kappa } ( \xi ) ^ { - 1 } \bigr ) .\tag{10}
$$

This is a weighted optimal design criterion for the target parameter (Pukelsheim, 2006). Let Ξ be a nonempty compact convex set of allocations on the $L$ comparison types, and assume that $M _ { \kappa } ( \boldsymbol { \xi } )$ is uniformly positive definite over Ξ. We define nuisance-adjusted optimal design (NAOD) by

$$
\xi ^ { \star } \in \mathop { \arg \operatorname* { m i n } } _ { \xi \in \Xi } \Phi _ { \kappa } ( \xi ) .
$$

Since $M _ { \kappa } ( { \boldsymbol \xi } )$ is affine in $\xi ,$ the objective is convex. To state its local minimax characterization, let $\boldsymbol { \eta } = ( h ^ { \top } , \ddot { v } ^ { \top } ) ^ { \top }$ , where $\boldsymbol { v } \in \mathbb { R } ^ { r }$ , and define $\gamma _ { n , \eta } = \gamma _ { 0 } + \eta / \sqrt { n }$ , with target component $\theta _ { n , h } =$ $\theta _ { 0 } + \dot { h } / \sqrt { n }$ . Write $\mathbb { E } _ { \eta }$ for expectation under this local model, conditional on the historical data.

Theorem 4.1 (Conditional local policy risk). Fix ξ and condition on the historical data. Under the regularity conditions in Appendix A.1, suppose $M _ { \kappa } ( \xi ) \succ 0 . \ I f G _ { 0 } \succ 0 ,$ , then

$$
\operatorname* { l i m } _ { R \to \infty } \operatorname* { l i m } _ { n \to \infty } \operatorname* { i n f } _ { T _ { n } } \operatorname* { s u p } _ { \| \eta \| \le R } n \mathbb { E } _ { \eta } D _ { F } ( \theta _ { n , h } , T _ { n } ) \ge \Phi _ { \kappa } ( \xi ) ,\tag{11}
$$

where $T _ { n }$ ranges over reward estimators based on the current observations. Under the additional trusted-sample and initialization conditions in Appendix B.2, with $\kappa > 0 ,$ , the projected one-step estimator defined there attains this bound uniformly on every fixed local ball.

Thus, for the projected one-step estimator, $\mathbb { E } _ { \eta } D _ { F } ( \theta _ { n , h } , \widehat { \theta } _ { n } ) = \Phi _ { \kappa } ( \xi ) / n + o ( n ^ { - 1 } )$ uniformly over every fixed local ball. The case $G _ { 0 } \succeq 0$ is handled by removing policy-null directions.

A matched Target-info design replaces $I _ { \mathrm { e f f } , \kappa } ( \boldsymbol { \xi } )$ by $A _ { \kappa } ( \boldsymbol { \xi } )$ in equation 10 while retaining the same final joint estimator. The two criteria can select different designs, and Proposition B.1 shows that Target-info can incur strictly larger leading policy risk.

## 4.4 FINITE-POOL COMPARISON SELECTION

We now translate the approximate design into the selection of B distinct comparisons from a finite pool of N candidates. Let $x _ { i } ~ \in ~ [ 0 , 1 ]$ be the relaxed inclusion weight of candidate i, and write ${ \boldsymbol x } = ( x _ { 1 } , \dots , x _ { N } )$ . Before querying the pool, we evaluate the information atoms and policy weight at a center fitted from historical feedback, yielding ${ \widehat { J } } _ { i }$ and $\widehat { G } _ { e }$ . Let $\widehat { I } _ { c }$ denote the corresponding total trusted information, embedded in the joint parameter space. The fitted information under x is $\begin{array} { r } { \widehat { \mathcal { T } } ( x ) = \widehat { I } _ { c } + \sum _ { i = 1 } ^ { N } x _ { i } \widehat { J } _ { i } } \end{array}$

Let C partition the candidate indices into groups from which at most one comparison may be selected. Without group restrictions, each candidate forms its own group. Let $S _ { 0 }$ be a fixed feasible seed, possibly empty, used when needed to ensure positive definite fitted information. Its comparisons count toward the budget B. The relaxed feasible set is

$$
\mathcal { X } _ { B } = \left\{ x \in [ 0 , 1 ] ^ { N } \bigg | \sum _ { i = 1 } ^ { N } x _ { i } = B , \quad x _ { i } = 1 \mathrm { f o r } i \in S _ { 0 } , \quad \sum _ { i \in c } x _ { i } \le 1 \mathrm { f o r } c \in \mathcal { C } \right\} .
$$

We solve

$$
\operatorname * { m i n } _ { x \in \mathcal { X } _ { B } } f _ { B } ( x ) , \qquad f _ { B } ( x ) = \frac { 1 } { 2 } \operatorname { t r } \left[ \widehat { G } _ { e } \widehat { \mathcal { T } } ( x ) ^ { - 1 } \right] .\tag{12}
$$

For $\xi _ { i } = x _ { i } / B$ and $\kappa = n _ { c } / B$ , this is the fitted finite-pool counterpart of $\Phi _ { \kappa } ( \boldsymbol { \xi } ) / B$ , preserving the total-information scaling of policy risk.

We solve the convex relaxation using the Frank–Wolfe algorithm (Jaggi, 2013), followed by feasible largest-weight rounding and improving exchanges. Algorithm 1 in Appendix C.1 gives the complete acquisition procedure, and the same appendix provides a computable optimization certificate for the returned subset.

Because rounding need not preserve the relaxed allocation proportions, the statistical guarantee is stated in terms of the information of the selected comparisons themselves. For an asymptotic sequence of selected sets $S _ { B }$ containing B distinct comparisons selected before their outcomes are observed, let $J _ { B i }$ denote the true information of candidate i at $\gamma _ { 0 }$ , and let $I _ { c , B } ^ { \mathrm { e x p } }$ denote the total expected trusted information. Define the normalized information of the selected set by $M _ { B } \ =$ $\begin{array} { r } { B ^ { - 1 } \Big ( I _ { c , B } ^ { \mathrm { e x p } } + \sum _ { i \in S _ { B } } J _ { B i } \Big ) } \end{array}$ . Let $\widehat { \theta } _ { B }$ be the target component of the projected one-step estimator based on the trusted sample and the selected judge labels.

Theorem 4.2 (Policy risk for distinct comparisons). Condition on the historical data, candidate features, and selected set. Under the distinct-comparison regularity conditions in Appendix C.2, suppose $M _ { B }$ is uniformly positive definite. Under the local model $\gamma _ { 0 } + \eta / \sqrt { B }$

$$
B \mathbb { E } _ { \eta } D _ { F } \Bigg ( \theta _ { 0 } + \frac { h } { \sqrt { B } } , \widehat { \theta } _ { B } \Bigg ) - \frac { 1 } { 2 } \mathrm { t r } ( G _ { e } M _ { B } ^ { - 1 } ) \longrightarrow 0\tag{13}
$$

uniformly over bounded local parameters. $I f M _ { B }$ converges to a positive definite limit, the corresponding conditional local minimax lower bound also holds.

The theorem evaluates the information of the realized subset and therefore retains the effect of round ing. Appendix C.2 gives the proof, while Appendix C.3 gives the plug-in analysis and computational details.

## 5 LEARNING THE JUDGE-DEVIATION REPRESENTATION

In this section, we study the additional error introduced when the judge-deviation representation is learned from historical feedback. This error contributes a design-dependent term to policy risk and can reverse the ordering of acquisition rules.

## 5.1 FINITE-SAMPLE REPRESENTATION COST

We use the same finite support $z _ { 1 } , \dots , z _ { L }$ and evaluate policy risk at the human target $\theta _ { 0 } = \theta ^ { \star }$ . Let m denote the number of independent historical units used to learn the judge-deviation representation. Write $\widehat { W } _ { m } \in \mathbb { R } ^ { L \times r }$ for the learned representation, with row i given by $\widehat { W } _ { m , i } ^ { \top }$ . Let $a _ { 0 m } \in \mathbb { R } ^ { r }$ be the nuisance component of the local center $\gamma _ { 0 m } = ( \theta _ { 0 } ^ { \top } , a _ { 0 m } ^ { \top } ) ^ { \top }$ . The nuisance parameter a is reestimated from the current data. Write logi $\mathsf { t } ( u ) = \log \{ u / ( 1 - u ) \}$ for $u \in ( 0 , 1 )$ ), and define the remaining logit error $\boldsymbol { e } _ { m } = ( e _ { m , 1 } , \ldots , e _ { m , L } ) ^ { \top }$ b

$$
e _ { m , i } = \mathrm { l o g i t } q ( z _ { i } ) - X _ { i } ^ { \top } \theta _ { 0 } - \widehat { W } _ { m , i } ^ { \top } a _ { 0 m } , \qquad i = 1 , \ldots , L .\tag{14}
$$

Let $\xi _ { m , i } = n _ { i } / n$ denote the judge allocation proportion for type i. Define $v _ { m , i } = ( X _ { i } ^ { \top } , \widehat { W } _ { m , i } ^ { \top } ) ^ { \top }$ and $t _ { m , i } = \sigma ^ { \prime } ( v _ { m , i } ^ { \top } \gamma _ { 0 m } )$ . Let $M _ { m }$ denote the normalized joint information at $\gamma _ { 0 m }$ , formed as in equation 7 from the learned representation, the actual judge allocation, and the expected trusted information. Set $\begin{array} { r } { \Phi _ { m } = \frac { 1 } { 2 } \mathrm { t r } ( \dot { G } _ { e } M _ { m } ^ { - 1 } ) } \end{array}$ , and let $P _ { \theta } = [ I _ { d } ~ 0 _ { d \times r } ]$ select the target coordinates. For $e = ( e _ { 1 } , \ldots , e _ { L } ) ^ { \top } \in \mathbb { R } ^ { L }$ , define

$$
B _ { m } e = P _ { \theta } M _ { m } ^ { - 1 } \sum _ { i = 1 } ^ { L } \xi _ { m , i } t _ { m , i } v _ { m , i } e _ { i } , \qquad R _ { m } = B _ { m } ^ { \top } G _ { 0 } B _ { m } .\tag{15}
$$

The first-order effect of $e _ { m }$ on the target parameter is $\boldsymbol { B _ { m } } \boldsymbol { e _ { m } }$ . Let $\widehat { \theta } _ { n , m }$ be the target component of the projected one-step estimator based on the learned representation and the current observations. Expectations below are over both the historical and current data.

Theorem 5.1 (Policy risk with a learned representation). Under the regularity conditions in $A p \cdot$ pendix D.2, suppose $n _ { c } / n  \kappa > 0$ and the smallest eigenvalue of $M _ { m }$ is bounded away from zero. Let $m , n \to \infty$ with $n / m  \lambda _ { P } \in [ 0 , \infty )$ . Suppose $( \Phi _ { m } , R _ { m } )  ( \Phi _ { 0 } , R _ { 0 } )$ in probability for deterministic limits, and $\sqrt { m } e _ { m }$ converges in distribution to a random vector $Z _ { P }$ with mean $b _ { P }$ and covariance $\Sigma _ { P }$ . Then

$$
\mathbb { E } \Big [ D _ { F } ( \theta _ { 0 } , \widehat { \theta } _ { n , m } ) \Big ] = \frac { \Phi _ { 0 } } { n } + \frac { C _ { P } } { m } + o ( n ^ { - 1 } + m ^ { - 1 } ) , \qquad C _ { P } = \frac { 1 } { 2 } \left\{ \operatorname { t r } ( R _ { 0 } \Sigma _ { P } ) + b _ { P } ^ { \top } R _ { 0 } b _ { P } \right\} .\tag{16}
$$

The first term is the oracle sampling risk and the second is the representation-learning cost, which depends on the direction of the representation error through $R _ { 0 }$ and includes both covariance and bias contributions. Proofs and additional representation-learning results are given in Appendix D.

## 5.2 DESIGN RANKING REVERSALS

The representation-cost term in equation 16 can reverse the ordering induced by the oracle sampling risk. We compare NAOD with the matched Target-info design under the same learned representation, label budgets, and final estimator. Let $\mathcal { L } _ { \mathrm { N } } ( n , m )$ and $\mathcal { L } _ { \mathrm { T } } ( n , m )$ denote their expected policy risks. Write $\Phi _ { \mathrm { { N } } } , \Phi _ { \mathrm { { T } } }$ for their limiting oracle risk constants and $C _ { P , \mathrm { N } } , C _ { P , \mathrm { T } }$ for their representation-cost coefficients. Applying Theorem 5.1 to the two designs gives

$$
n \{ { \mathcal L } _ { \mathrm { N } } ( n , m ) - { \mathcal L } _ { \mathrm { T } } ( n , m ) \} \longrightarrow \Phi _ { \mathrm { N } } - \Phi _ { \mathrm { T } } + \lambda _ { P } ( C _ { P , \mathrm { N } } - C _ { P , \mathrm { T } } ) .\tag{17}
$$

If $\Phi _ { \mathrm { N } } < \Phi _ { \mathrm { T } }$ , the leading-order ordering reverses whenever $\lambda _ { P } ( C _ { P , \mathrm { N } } - C _ { P , \mathrm { T } } ) > \Phi _ { \mathrm { T } } - \Phi _ { \mathrm { N } }$ . Thus, NAOD can have lower oracle sampling risk while being more sensitive to representation error.

When $n / m  0$ , the representation-cost term vanishes and the oracle ordering is recovered. When $n / m  \lambda _ { P } > 0$ , it can contribute at leading order. Thus, NAOD optimizes the fitted design criterion rather than the full two-stage risk.

## 6 EXPERIMENTS

## 6.1 SYNTHETIC EXPERIMENTS

Effective information predicts policy risk. We first consider a scalar three-type Bernoulli construction with a constant nuisance component and equal trusted and judge sample sizes, $n _ { c } = n$ NAOD, Target-info, and Random use the same correctly specified joint model and projected one-step estimator and differ only in the acquisition design. Each setting is evaluated over 6,000 independent replications. Across $n \in \{ 2 4 0 , 9 6 0 , 3 8 4 0 , 1 5 3 6 0 \}$ , the empirical scaled risks in Figure 2(a) closely track their corresponding theoretical information constants, supporting the effective-information characterization of leading policy risk.

![](images/9c88f8c4f5efbd47cd094432f0b3fc18d9450726fd1705cf62d5a1e1c9406f10.jpg)  
(a) Effective information.

![](images/2c67201f357c637548496ffc90d8c1d473b5ee14b0ad39bbe61ba42f195543e9.jpg)

![](images/5afad930bba406153a40cbf3033216dabd912e7ef8b1d799d459ac0edc3ec959.jpg)  
Figure 2: Controlled tests of the theoretical predictions. Dashed references in panels (a) and (b) denote the corresponding theoretical predictions. Panel (c) reports the NAOD-minus-Target-info scaled-risk difference.

Learning the representation adds the predicted risk term. We next use a scalar two-type Bernoulli construction with $n _ { c } = n$ , in which the first-order cost of learning the judge-deviation representation is explicit. For this construction, $\Phi _ { 0 } = 5 0 / 4 9$ and $C _ { P } = 2 5 / 9 8$ . Figure 2(b) tests the risk expansion in equation 16 across $n \in \{ 9 6 0$ , 3840, 15360} and $m / n \in \{ 1 / 4 , 1 , 4 \}$ . The empirical scaled risks closely follow the theoretical prediction $\Phi _ { 0 } + ( n / m ) \dot { C _ { P } }$ , with the largest finite-sample deviation occurring at the smallest representation-learning sample size.

Representation error can reverse design rankings. We then learn the representation in the threetype experiment and compare NAOD with Target-info under otherwise identical conditions, with $n = n _ { c } = 9 6 0$ . Figure 2(c) shows the paired NAOD-minus-Target-info scaled-risk difference as $m / n$ varies. With limited historical data, the difference is positive and Target-info has lower risk. As the historical sample grows, the gap shrinks, becomes indistinguishable from zero around $m / n = 1$ and turns negative by $m / n = 2$ , where NAOD has lower risk. This confirms the design-ranking reversal mechanism in Section 5.2. Full constructions, finite-sample diagnostics, and additional analyses are reported in Appendices E.2 and E.3.

## 6.2 CHATBOT ARENA EVALUATION

Experimental setup. We evaluate finite-pool acquisition on a frozen Chatbot Arena LLM-judge archive (Chiang et al., 2024) with 49,635 usable comparisons in 42,875 connected text clusters and 17 judges. Current budgets are $H \ \in \ \{ 3 2 , 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \}$ human labels and $B \in$ {256, 512, 1024} judge records, yielding 15 budget combinations. Each combination is evaluated over 15 cluster-level splits. Within each split, representation learning, acquisition, and evaluation use disjoint clusters. At most one comparison is selected from each candidate cluster.

All acquisition methods differ only in the acquisition rule. Frozen Qwen3 embeddings (Zhang et al., 2025) are used to construct judge-specific two-dimensional target and nuisance representations, with the full procedure given in Appendix F.1. Representation learning uses 8,575 shared upstream human–judge comparisons, and initialization uses 128 historical comparisons. These shared resources are fixed across acquisition rules and are not counted in the current (H, B) budgets. We compare NAOD against five acquisition baselines, Random, Entropy, D-opt, PA D-opt (Shen et al., 2025), and Target-info. Initial and Human-only serve as reference baselines from the common initialization. Judge feedback uses the archived A/B/tie probabilities (Sun et al., 2025a) through the soft target $p _ { A } + { \frac { 1 } { 2 } } p _ { T }$ , where $p _ { A }$ and $p _ { T }$ are the archived probabilities of an A win and a tie, respectively. For these bounded soft targets, Appendix A.4 gives the corresponding sandwich-risk characterization. Under mean correctness, Theorem A.4 shows that the population curvature crite rion underlying NAOD equals the worst-case leading sandwich policy-risk coefficient over bounded soft-feedback laws with the same conditional means.

Evaluation. The primary endpoint is proxy policy regret relative to a human-preference reference fitted only from upstream data. We additionally report held-out human cross-entropy and tie-aware human choice accuracy. Test losses are first averaged within connected clusters and then equally across clusters. Within each split, judges and budget combinations are macro-averaged. Paired uncertainty is reported using two-sided 95% t-intervals across the 15 split-level differences.

Results. NAOD achieves the lowest mean proxy policy regret and human cross-entropy in Table 1, together with the highest mean human choice accuracy. Relative to the matched Target-info design, NAOD reduces mean proxy policy regret by 29.11%, with 14 of 15 split averages favoring NAOD. Relative to D-opt, the reduction is 18.87%, with 12 of 15 split averages favoring NAOD.

Table 1: Overall Chatbot Arena results. Means over 17 judges, 15 budget regimes, and 15 random cluster-level splits. Regret is reported in $1 0 ^ { - 3 }$ units, CE in nats per pair, and accuracy in percent. The last two columns report NAOD’s relative regret reduction and paired split wins against each row. Bold denotes the best mean.
<table><tr><td>Method</td><td>Proxy regret ↓</td><td>Human CE↓</td><td>Acc. ↑</td><td>Drop (%)</td><td>Wins</td></tr><tr><td>Initial</td><td>8.86</td><td>0.6752</td><td>57.82</td><td>77.53</td><td>13/15</td></tr><tr><td>Human-only</td><td>18.25</td><td>0.6829</td><td>56.84</td><td>89.09</td><td>15/15</td></tr><tr><td>Random</td><td>4.88</td><td>0.6727</td><td>58.22</td><td>59.18</td><td>15/15</td></tr><tr><td>Entropy</td><td>13.42</td><td>0.6784</td><td>57.10</td><td>85.17</td><td>15/15</td></tr><tr><td>D-opt</td><td>2.45</td><td>0.6703</td><td>58.19</td><td>18.87</td><td>12/15</td></tr><tr><td>PA D-opt</td><td>2.46</td><td>0.6703</td><td>58.19</td><td>18.93</td><td>12/15</td></tr><tr><td>Target-info</td><td>2.81</td><td>0.6705</td><td>58.21</td><td>29.11</td><td>14/15</td></tr><tr><td>NAOD</td><td>1.99</td><td>0.6699</td><td>58.27</td><td>一</td><td></td></tr></table>

Budget dependence. NAOD reduces mean proxy policy regret relative to Target-info in all 15 budget settings. At a fixed human budget H, the relative advantage consistently narrows as the judge budget B increases. Moreover, settings with the same ratio $\bar { H } / B$ can exhibit substantially different gains, showing that absolute budget size matters in addition to the human-to-judge budget ratio. The complete $5 \times 3$ budget grid is reported in Figure 4 in Appendix F.3.

Target–nuisance coupling. The acquisition gain varies across judges and tends to be larger when Target-info induces stronger target–nuisance coupling. This pattern is consistent with the theoretical prediction that nuisance-aware acquisition is most valuable when target and nuisance information are strongly coupled. The corresponding judge-level analysis is reported in Figure 5 in Appendix F.3.

## 7 CONCLUSION

We studied active preference learning with LLM-judge feedback that may systematically deviate from target human preferences. NAOD prioritizes policy-relevant target information after nuisance adjustment. The design criterion has a sharp local minimax characterization, while controlled experiments validate the predicted risk behavior and Arena results show that gains tend to be larger under stronger target–nuisance coupling. Representation error can nevertheless change the preferred acquisition design, motivating future work on jointly learning judge-deviation representations and selecting comparisons.

## AI USE STATEMENT

Generative AI tools were used to assist with language polishing and to check the algebraic consistency of author-derived mathematical derivations. All AI-assisted suggestions and checks were reviewed and independently verified by the authors. The authors take full responsibility for the final content of the paper.

## ETHICS STATEMENT

This work uses archived Chatbot Arena preference data and collects no new human-subject data. The available preference feedback and LLM-judge outputs may reflect population- and model-specific biases and may not be universally representative. Preference models and acquisition strategies based on such feedback should therefore be evaluated with respect to the intended target population and deployment setting.

## REPRODUCIBILITY STATEMENT

We provide the code required to reproduce the reported experimental results in the supplementary material. The main text and appendices provide the statistical model, acquisition and estimation procedures, optimization details, and experimental protocols required for reproduction, together with the assumptions and proofs underlying the theoretical results.

## REFERENCES

Mohammad Gheshlaghi Azar, Vicenc¸ Gomez, and Hilbert J. Kappen. Dynamic policy programming.´ Journal ofMachine Learning Research, 13(103):3207–3245, 2012.

Richard J. Barker and Laimonis Kavalieris. Efficiency gain from auxiliary data requiring additional nuisance parameters. Biometrics, 57(2):563–566, 2001.

Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3-4):324–345, 1952.

Alex Chan, Ahmed Alaa, Zhaozhi Qian, and Mihaela Van Der Schaar. Unlabelled data improves Bayesian uncertainty calibration under covariate shift. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 1392–1402. PMLR, 2020.

Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, and Benyou Wang. Humans or LLMs as the judge? a study on judgement bias. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 8301–8327. Association for Computational Linguistics, 2024.

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael Jordan, Joseph E. Gonzalez, and Ion Stoica. Chatbot Arena: An open platform for evaluating LLMs by human preference. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 8359–8388. PMLR, 2024.

Muthu Chidambaram and Rong Ge. Reassessing how to compare and improve the calibration of machine learning models. In International Conference on Learning Representations, 2025.

Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, volume 30, pp. 4299–4307, 2017.

Florian E. Dorner, Vivian Y. Nastl, and Moritz Hardt. Limits to scalable evaluation at the frontier: LLM as judge won’t beat twice the data. In International Conference on Learning Representations, 2025.

Sebastian Farquhar, Yarin Gal, and Tom Rainforth. On statistical bias in active learning: How and when to fix it. In International Conference on Learning Representations, 2021.

Yunzhen Feng, Ariel Kwiatkowski, Kunhao Zheng, Julia Kempe, and Yaqi Duan. PILAF: Optimal human preference sampling for reward modeling. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 16744–16776. PMLR, 2025.

Benjamin Feuer, Micah Goldblum, Teresa Datta, Sanjana Nambiar, Raz Besaleli, Samuel Dooley, Max Cembalest, and John P. Dickerson. Style outweighs substance: Failure modes of LLM judges in alignment benchmarking. In International Conference on Learning Representations, 2025.

R. M. Fewster and P. E. Jupp. Information on parameters of interest decreases under transformations. Journal ofMultivariate Analysis, 120:34–39, 2013.

Evan Frick, Tianle Li, Connor Chen, Wei-Lin Chiang, Anastasios N. Angelopoulos, Jiantao Jiao, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. How to evaluate reward models for RLHF. In International Conference on Learning Representations, 2025.

Futoshi Futami and Masahiro Fujisawa. Information-theoretic generalization analysis for expected calibration error. In Advances in Neural Information Processing Systems, volume 37, pp. 84246– 84297, 2024.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceed ings of Machine Learning Research, pp. 10835–10866. PMLR, 2023.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings ofMachine Learning Research, pp. 1321–1330. PMLR, 2017.

Xiao Hu, Jianxiong Li, Xianyuan Zhan, Qing-Shan Jia, and Ya-Qin Zhang. Query-policy misalignment in preference-based reinforcement learning. In International Conference on Learning Representations, 2024.

Martin Jaggi. Revisiting Frank-Wolfe: Projection-free sparse convex optimization. In Proceedings of the 30th International Conference on Machine Learning, volume 28 of Proceedings of Machine Learning Research, pp. 427–435. PMLR, 2013.

Christoph Kern, Michael P. Kim, and Angela Zhou. Multi-accurate CATE is robust to unknown covariate shifts. Transactions on Machine Learning Research, 2024.

Michael P. Kim, Amirata Ghorbani, and James Zou. Multiaccuracy: Black-box post-processing for fairness in classification. In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society, pp. 247–254. ACM, 2019.

Chungpa Lee, Thomas Zeng, Jongwon Jeong, Jy-yong Sohn, and Kangwook Lee. How to correctly report LLM-as-a-judge evaluations. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research. PMLR, 2026.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Ren Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, and Sushant Prakash. RLAIF vs. RLHF: Scaling reinforcement learning from human feedback with AI feedback. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 26874–26901. PMLR, 2024.

Kimin Lee, Laura M. Smith, and Pieter Abbeel. PEBBLE: Feedback-efficient interactive reinforcement learning via relabeling experience and unsupervised pre-training. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 6152–6163. PMLR, 2021.

Haitao Li, Junjie Chen, Qingyao Ai, Zhumin Chu, Yujia Zhou, Qian Dong, and Yiqun Liu. CalibraEval: Calibrating prediction distribution to mitigate selection bias in LLMs-as-judges. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16537–16552. Association for Computational Linguistics, 2025.

Xiaoqiang Lin, Arun Verma, Zhongxiang Dai, Daniela Rus, See-Kiong Ng, and Bryan Kian Hsiang Low. ActiveDPO: Active direct preference optimization for sample-efficient alignment. In International Conference on Learning Representations, 2026.

Zhihan Liu, Miao Lu, Shenao Zhang, Boyi Liu, Hongyi Guo, Yingxiang Yang, Jose Blanchet, and Zhaoran Wang. Provably mitigating overoptimization in RLHF: Your SFT loss is implicitly an adversarial regularizer. In Advances in Neural Information Processing Systems, volume 37, pp. 138663–138697, 2024.

Adian Liusie, Vatsal Raina, Yassir Fathullah, and Mark Gales. Efficient LLM comparative assessment: A product of experts framework for pairwise comparisons. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 6835–6855. Association for Computational Linguistics, 2024.

Lucas Maystre and Matthias Grossglauser. Just sort it! A simple and effective approach to active preference learning. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings ofMachine Learning Research, pp. 2344–2353. PMLR, 2017.

Rajiv Movva, Smitha Milli, Sewon Min, and Emma Pierson. What’s in my human feedback? learning interpretable descriptions of preference data. In International Conference on Learning Rep resentations, 2026.

Subhojyoti Mukherjee, Anusha Lalitha, Kousha Kalantari, Aniket Deshmukh, Ge Liu, Yifei Ma, and Branislav Kveton. Optimal design for human preference elicitation. In Advances in Neural Information Processing Systems, volume 37, pp. 90132–90159, 2024.

William Muldrew, Peter Hayes, Mingtian Zhang, and David Barber. Active preference learning for large language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 36577–36590. PMLR, 2024.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pp. 27730–27744, 2022.

Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model’s uncertainty? evaluating predictive uncertainty under dataset shift. In Advances in Neural Information Processing Systems, volume 32, pp. 13991–14002, 2019.

Junsoo Park, Seungyeon Jwa, Ren Meiying, Daeyoung Kim, and Sanghyuk Choi. OffsetBias: Leveraging debiased data for tuning evaluators. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pp. 1043–1067. Association for Computational Linguistics, 2024.

Sangdon Park, Osbert Bastani, James Weimer, and Insup Lee. Calibrated prediction with covariate shift via unsupervised domain adaptation. In Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pp. 3219–3229. PMLR, 2020.

Felipe Maia Polo, Xinhe Wang, Mikhail Yurochkin, Gongjun Xu, Moulinath Banerjee, and Yuekai Sun. Bridging human and LLM judgments: Understanding and narrowing the gap. In Advances in Neural Information Processing Systems, volume 38, pp. 16857–16908, 2025.

Friedrich Pukelsheim. Optimal Design of Experiments. Society for Industrial and Applied Mathematics, 2006.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pp. 53728–53741, 2023.

Yunyi Shen, Hao Sun, and Jean-Francois Ton. Active reward modeling: Adaptive preference labeling for large language model alignment. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 54410–54430. PMLR, 2025.

Sabina J. Sloman, Ayush Bharti, Julien Martinelli, and Samuel Kaski. Bayesian active learning in the presence of nuisance parameters. In Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence, volume 244 of Proceedings of Machine Learning Research, pp. 3245– 3263. PMLR, 2024.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F. Christiano. Learning to summarize with human feedback. In Advances in Neural Information Processing Systems, volume 33, pp. 3008–3021, 2020.

Guangzhi Sun, Anmol Kagrecha, Potsawee Manakul, Phil Woodland, and Mark Gales. SkillAggregation: Reference-free LLM-dependent aggregation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15532–15548. Association for Computational Linguistics, 2025a.

Zexu Sun, Yiju Guo, Yankai Lin, Xu Chen, Qi Qi, Xing Tang, Xiuqiang He, and Ji-Rong Wen. Uncertainty and influence aware reward model refinement for reinforcement learning from human feedback. In International Conference on Learning Representations, 2025b.

Ming Tang, Yuxuan Zhou, and Chao Huang. Tackling biased evaluators in dueling bandits. In Advances in Neural Information Processing Systems, volume 38, pp. 83569–83602, 2025.

Kiran Koshy Thekumparampil, Gaurush Hiranandani, Kousha Kalantari, Shoham Sabach, and Branislav Kveton. Comparing few to rank many: Active human preference learning using randomized Frank-Wolfe method. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 59355–59376. PMLR, 2025.

Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and Zhifang Sui. Large language models are not fair evaluators. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9440–9450. Association for Computational Linguistics, 2024.

Xueru Wen, Jie Lou, Yaojie Lu, Hongyu Lin, Xing Yu, Xinyu Lu, Ben He, Xianpei Han, Debing Zhang, and Le Sun. Rethinking reward model evaluation: Are we barking up the wrong tree? In International Conference on Learning Representations, 2025.

Runzhe Wu and Wen Sun. Making RL with preference-based feedback efficient via randomization. In International Conference on Learning Representations, 2024.

Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao, Werner Geyer, Chao Huang, Pin-Yu Chen, Nitesh V. Chawla, and Xiangliang Zhang. Justice or prejudice? quantifying biases in LLM-as-a-judge. In International Conference on Learning Representations, 2025.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, pp. 46595–46623, 2023.

Han Zhou, Xingchen Wan, Yinhong Liu, Nigel Collier, Ivan Vulic, and Anna Korhonen. Fairer´ preferences elicit improved human-aligned large language model judgments. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 1241–1252. Association for Computational Linguistics, 2024.

## APPENDIX

## A TECHNICAL FOUNDATIONS

This appendix collects the common statistical and policy-loss facts used throughout the analysis. We specify the conditional experiment and policy loss, establish the local likelihood expansion, characterize efficient target information and policy-null directions, and finally separate likelihood curvature from score covariance for soft feedback.

## A.1 NOTATION, REGULARITY, AND POLICY LOSS

Conditioning and notation. Let $\mathcal { F } _ { P }$ denote the historical information available before the current outcomes are observed, including the frozen nuisance representation, comparison features, allocation, and numerical settings. Unless explicitly stated otherwise, probabilities and expectations in the fixed-representation theory are conditional on $\mathcal { F } _ { P }$ . Write $\gamma = \overline { { ( \theta ^ { \top } , a ^ { \top } ) ^ { \top } } } \in \mathbb { R } ^ { k }$ , with $k = d + r ,$ and let $\overset { \cdot } { P _ { \theta } } = \left[ I _ { d } \right. \left. \right. \left. 0 \right]$ select the target coordinates. A judge observation of comparison type i has augmented feature $\dot { v _ { i } } = ( X _ { i } ^ { \top } , W _ { i } ^ { \top } ) ^ { \top }$ , while a trusted observation with feature $X _ { c j }$ has padded feature $v _ { c j } = ( X _ { c j } ^ { \top } , 0 ^ { \top } ) ^ { \top }$ . There are $n _ { i }$ judge observations of type i, $\begin{array} { r } { n = \sum _ { i = 1 } ^ { L } n _ { i } } \end{array}$ judge observations in total, and $n _ { c }$ trusted observations. For bounded $\eta = ( h ^ { \top } , v ^ { \top } ) ^ { \top }$ , the local parameter is $\gamma _ { n , \eta } = \gamma _ { 0 } + \eta / \sqrt { n }$ , with target component $\theta _ { n , h } = \theta _ { 0 } + h / \sqrt { n }$

Regularity. For the fixed-support theory, $L , d , r$ are fixed and all feature vectors are uniformly bounded. Conditional on their features, current judge and trusted outcomes are mutually independent Bernoulli variables under the joint model in equation 6. A compact convex product set $\mathcal { K } = \mathcal { K } _ { \theta } \times \mathcal { K } _ { a }$ contains $\gamma _ { 0 }$ in its interior, with a bounded open neighborhood available for derivative expansions. The proportions satisfy $n _ { i } / n \to \xi _ { i }$ and $n _ { c } / n  \kappa < \infty$ . Uniformly over bounded root-n neighborhoods, the normalized expected information converges to the positive-definite ma trix $M _ { \kappa } ( { \boldsymbol \xi } )$ in equation 7. Trusted covariates may be bounded deterministic arrays with the stated information limit or independent bounded random covariates independent of the judge block. The policy value $F$ is twice continuously differentiable near the target region, with bounded uniformly continuous Hessian. For the Gibbs policy considered here, boundedness follows directly from the covariance representation below, while uniform continuity follows on the compact target neighborhood. Bounded features and the compact parameter region also imply a common interior probability bound $q ( 1 - q ) \geq t _ { \operatorname* { m i n } } > 0$ . Additional conditions required by the concrete preliminary estimator are stated in Appendix B.2.

Gibbs policy and directed policy loss. For the Gibbs policy in equation 2 and value function in equation 3, substituting the log probability ratio into the regularized policy objective gives

$$
V _ { \theta } ( \pi ) = F ( \theta ) - \tau \mathbb { E } _ { \mu _ { e } } \mathrm { K L } ( \pi ( \cdot \mid s ) \parallel \pi _ { \theta } ( \cdot \mid s ) ) .
$$

Thus $\pi _ { \theta }$ is the unique optimizer. Differentiating the log partition function yields

$$
\begin{array} { r } { \nabla F ( \theta ) = \mathbb { E } _ { s \sim \mu _ { e } } \mathbb { E } _ { y \sim \pi _ { \theta } ( \cdot | s ) } [ \phi ( s , y ) ] , \qquad \nabla ^ { 2 } F ( \theta ) = \frac { 1 } { \tau } \mathbb { E } _ { s \sim \mu _ { e } } \operatorname { C o v } _ { y \sim \pi _ { \theta } ( \cdot | s ) } [ \phi ( s , y ) ] . } \end{array}
$$

In particular, $F$ is convex. If the policy optimized at parameter $y$ is evaluated under reward parameter x, linearity of the reward gives $\dot { V } _ { x } ( \pi _ { y } ) \dot { = } F ( y ) + \dot { \nabla } F ( y ) ^ { \top } ( x - y )$ , and therefore

$$
F ( x ) - V _ { x } ( \pi _ { y } ) = D _ { F } ( x , y ) = \tau \mathbb { E } _ { \mu _ { e } } \mathrm { K L } ( \pi _ { y } ( \cdot \mid s ) \parallel \pi _ { x } ( \cdot \mid s ) ) .
$$

Hence the first Bregman argument is the reward under which performance is evaluated, whereas the second determines the deployed policy. At the local center,

$$
D _ { F } ( \theta _ { 0 } , \theta _ { 0 } + h ) = \frac { 1 } { 2 } h ^ { \top } G _ { 0 } h + o ( \| h \| ^ { 2 } ) , \qquad G _ { 0 } = \nabla ^ { 2 } F ( \theta _ { 0 } ) .\tag{18}
$$

This gives the policy weighting used by NAOD.

## A.2 SCORES, INFORMATION, AND LOCAL ASYMPTOTIC NORMALITY

Let $\operatorname { s p } ( u ) = \log ( 1 + e ^ { u } )$ ). For judge labels $Y _ { i \ell }$ and trusted labels $H _ { j }$ , the total negative log likelihood is

$$
Q _ { n } ( \gamma ) = \sum _ { i = 1 } ^ { L } \sum _ { \ell = 1 } ^ { n _ { i } } \left\{ \mathrm { s p } ( v _ { i } ^ { \top } \gamma ) - Y _ { i \ell } v _ { i } ^ { \top } \gamma \right\} + \sum _ { j = 1 } ^ { n _ { c } } \left\{ \mathrm { s p } ( X _ { c j } ^ { \top } \theta ) - H _ { j } X _ { c j } ^ { \top } \theta \right\} .
$$

Define the total score and observed information by $U _ { n } ( \gamma ) = - \nabla Q _ { n } ( \gamma )$ and $I _ { n } ( \gamma ) = \nabla ^ { 2 } Q _ { n } ( \gamma )$ A judge observation with probability $q _ { i } ( \gamma ) = \sigma ( v _ { i } ^ { \top } \gamma )$ contributes $( Y _ { i \ell } - q _ { i } ( \gamma ) ) v _ { i }$ to the score and $q _ { i } ( \gamma ) \{ 1 - q _ { i } ( \gamma ) \} v _ { i } v _ { i } ^ { \top }$ to the information. Trusted observations have the same form with padded features. Hence $Q _ { n }$ is globally convex.

At the correctly specified Bernoulli center,

$$
M _ { n } : = \frac { 1 } { n } \mathbb { E } I _ { n } ( \gamma _ { 0 } ) = \frac { 1 } { n } \operatorname { C o v } \{ U _ { n } ( \gamma _ { 0 } ) \} \longrightarrow M _ { \kappa } ( \xi ) .
$$

The equality between score covariance and curvature is specific to the correctly specified Bernoulli experiment. Appendix A.4 treats soft feedback separately. Let $\Delta _ { n } = U _ { n } ( \gamma _ { 0 } ) \dot { / } \sqrt { n }$ . Bounded independent score summands in fixed dimension give sup $\phantom { } _ { n } \mathbb { E } \| \Delta _ { n } \| ^ { 4 } < \infty ,$ , while the normalized Hessian differs from its expectation by $O _ { L ^ { 4 } } ( n ^ { - 1 / 2 } )$ when current trusted covariates are random. For fixed covariate arrays, it is deterministic conditional on the features.

Proposition A.1 (Conditional LAN). Under the regularity conditions of Appendix A.1,

$$
\Delta _ { n } \Rightarrow N ( 0 , M _ { \kappa } ( \xi ) ) ,
$$

and, writing $\ell _ { n } = - Q _ { n } ,$ for every fixed $R < \infty$

$$
\ell _ { n } \bigg ( \gamma _ { 0 } + \frac { \eta } { \sqrt { n } } \bigg ) - \ell _ { n } ( \gamma _ { 0 } ) = \eta ^ { \top } \Delta _ { n } - \frac { 1 } { 2 } \eta ^ { \top } M _ { \kappa } ( \xi ) \eta + o _ { P } ( 1 )
$$

uniformly over $\| \eta \| \le R$ . The same derivative and moment bounds hold uniformly along bounded local truth sequences.

Proof. Every scalar projection of $\Delta _ { n }$ is a triangular array of independent centered bounded terms whose largest summand is $O ( n ^ { - 1 / 2 } )$ and whose total variance converges to the corresponding quadratic form of $M _ { \kappa } ( \boldsymbol { \xi } )$ . The triangular-array central limit theorem and the Cramer–Wold device´ give the Gaussian limit. A third-order Taylor expansion of the log likelihood around γ<sub>0</sub> gives the displayed quadratic likelihood ratio. Bounded third derivatives make the remainder $O ( n ^ { - \bar { 1 } / 2 } )$ uniformly on each fixed local ball, while normalized information converges uniformly to $\dot { M } _ { \kappa } ( \boldsymbol { \xi } )$ □

## A.3 EFFICIENT TARGET INFORMATION AND POLICY-NULL DIRECTIONS

Suppress $( \kappa , \xi )$ and partition the correctly specified Bernoulli information as

$$
M = \left( \begin{array} { l l } { { A } } & { { C } } \\ { { C ^ { \top } } } & { { D } } \end{array} \right) \succ 0 .
$$

Let $( S _ { \theta } , S _ { a } )$ denote the target and nuisance components of the limiting LAN score, whose covariance is M. Since $D \succ 0$ , define $S _ { \mathrm { e f f } } = S _ { \theta } - C \bar { D ^ { - 1 } } S _ { a }$ . Then

$$
\mathrm { C o v } ( S _ { \mathrm { e f f } } , S _ { a } ) = 0 , \qquad \mathrm { C o v } ( S _ { \mathrm { e f f } } ) = A - C D ^ { - 1 } C ^ { \top } = I _ { \mathrm { e f f } } .
$$

Thus the Schur complement removes exactly the target-score component linearly reproducible by nuisance-score variation. Equivalently, for every target direction $h ,$

$$
\boldsymbol { h } ^ { \top } \boldsymbol { I } _ { \mathrm { e f f } } \boldsymbol { h } = \operatorname* { m i n } _ { \boldsymbol { b } \in \mathbb { R } ^ { r } } \left\{ \boldsymbol { h } ^ { \top } \boldsymbol { A } \boldsymbol { h } - 2 \boldsymbol { h } ^ { \top } \boldsymbol { C } \boldsymbol { b } + \boldsymbol { b } ^ { \top } \boldsymbol { D } \boldsymbol { b } \right\} ,
$$

with minimizer $b = D ^ { - 1 } C ^ { \top } h$ . Block elimination gives

$$
P _ { \theta } M ^ { - 1 } P _ { \theta } ^ { \top } = I _ { \mathrm { e f f } } ^ { - 1 } .\tag{19}
$$

Finally, with $K = A ^ { - 1 / 2 } C D ^ { - 1 / 2 }$

$$
A ^ { - 1 / 2 } I _ { \mathrm { e f f } } A ^ { - 1 / 2 } = I - K K ^ { \top } .\tag{20}
$$

The latter is the whitening identity used by the target–nuisance coupling analysis.

At the saturation extreme, if the judge target-score directions lie entirely in the nuisance-score span, judge feedback contributes no additional effective information for the human target. In particular, if $W _ { i } = X _ { i }$ with unrestricted nuisance coefficients, the judge likelihood identifies only $\theta + a .$ , and $I _ { \mathrm { e f f } } = \kappa H _ { c }$

Policy-null directions. Theorem 4.1 states the lower bound directly for $G _ { 0 } \succ 0$ . For the Gibbs policy, the reduction required when $G _ { 0 } \succeq 0$ is exact.

Proposition A.2 (Exact policy-null quotient). Let $\mathcal { N } = \ker \nabla ^ { 2 } F ( \theta _ { 0 } )$ . For thefinite-response Gibbs policy with full-support reference policy, $\mathcal { N }$ is the same at every finite parameter. For every $h \in \mathcal N$ andfinite t,

$$
\pi _ { \theta + t h } ( \cdot \mid s ) = \pi _ { \theta } ( \cdot \mid s ) f o r \mu _ { e } { - a l m o s t e \nu e r y s } .
$$

In orthogonal coordinates $\theta = ( \beta , \zeta ) \in \mathcal { N } ^ { \perp } \oplus \mathcal { N }$ , the value decomposes as $F ( \beta , \zeta ) = f ( \beta ) + c ^ { \top } \zeta ,$ so

$$
D _ { F } \big ( ( \beta , \zeta ) , ( \beta ^ { \prime } , \zeta ^ { \prime } ) \big ) = D _ { f } ( \beta , \beta ^ { \prime } ) .
$$

Unless the policy loss is identically zero, $\nabla ^ { 2 } f ( \beta _ { 0 } ) \succ 0 .$

Proof. For $h \in \mathcal N$ , the covariance representation of the policy Hessian gives

$$
0 = h ^ { \top } \nabla ^ { 2 } F ( \theta _ { 0 } ) h = \frac { 1 } { \tau } \mathbb { E } _ { \mu _ { e } } \operatorname { V a r } _ { \pi \theta _ { 0 } ( \cdot | s ) } \{ h ^ { \top } \phi ( s , y ) \} .
$$

Because the Gibbs policy has full support, $h ^ { \top } \phi ( s , y )$ must be constant over responses for almost every prompt. Adding th therefore multiplies every unnormalized Gibbs weight at that prompt by the same factor, leaving the normalized policy unchanged and making $F$ affine along h. The same response-constant characterization holds at every finite parameter, so the null space is parameterindependent. The affine component cancels from the Bregman divergence, giving the quotient representation. □

Policy-null target coordinates may therefore be treated as additional nuisance coordinates. Applying the positive-curvature argument on $\mathcal { N } ^ { \perp }$ yields the same policy-weighted trace criterion in the original coordinates.

## A.4 SOFT FEEDBACK AND SANDWICH POLICY RISK

For binary evaluation comparisons with equal reference weights and $\tau ~ = ~ 1$ , let $F _ { e } ( \theta ) =$ $\mathbb { E } _ { e } \operatorname { s p } ( X ^ { \dag } \theta )$ up to an affine term. Under the correctly specified human model,

$$
\operatorname { C E } ( \theta ) - \operatorname { C E } ( \theta ^ { \star } ) = D _ { F _ { c } } ( \theta , \theta ^ { \star } ) , \qquad \operatorname { R e g r e t } ( \pi _ { \theta } ; \theta ^ { \star } ) = D _ { F _ { c } } ( \theta ^ { \star } , \theta ) .
$$

The two losses therefore have opposite Bregman orientations, although both have the same local quadratic expansion

$$
\frac { 1 } { 2 } ( \theta - \theta ^ { \star } ) ^ { \top } \nabla ^ { 2 } F _ { e } ( \theta ^ { \star } ) ( \theta - \theta ^ { \star } ) + o ( \| \theta - \theta ^ { \star } \| ^ { 2 } ) .
$$

This distinction is used in Appendix F.2, where proxy policy regret and held-out human crossentropy are separate endpoints.

Now allow conditionally independent responses $Y _ { n j } \in [ 0 , 1 ]$ . The logistic loss remains $\mathrm { s p } ( v _ { n j } ^ { \top } \gamma ) -$ $Y _ { n j } v _ { n j } ^ { \top } \gamma$ , with

$$
U _ { n } ( \gamma ) = \sum _ { j } v _ { n j } \{ Y _ { n j } - q _ { n j } ( \gamma ) \} , \qquad I _ { n } ( \gamma ) = \sum _ { j } q _ { n j } ( \gamma ) \{ 1 - q _ { n j } ( \gamma ) \} v _ { n j } v _ { n j } ^ { \top } ,
$$

where $q _ { n j } ( \gamma ) = \sigma ( v _ { n j } ^ { \top } \gamma )$ . Hence soft and Bernoulli responses share the same logistic curvature, while their score covariances need not agree.

Proposition A.3 (Soft-label sandwich policy risk). Suppose interior population roots $\gamma _ { n } ^ { \dagger } ~ =$ $( \theta _ { n } ^ { \dagger } { } ^ { \dagger } , a _ { n } ^ { \dagger } { } ^ { \top } ) ^ { \top }  \gamma ^ { \dagger }$ satisfy $\mathbb { E } U _ { n } ( \gamma _ { n } ^ { \dagger } ) = 0 ;$ , and

$$
M _ { n } : = \frac { 1 } { n } \mathbb { E } I _ { n } ( \gamma _ { n } ^ { \dagger } )  M \succ 0 , \qquad \Omega _ { n } : = \frac { 1 } { n } \operatorname { C o v } \{ U _ { n } ( \gamma _ { n } ^ { \dagger } ) \}  \Omega \succeq 0 .
$$

Assume a regular estimator satisfies

$$
\sqrt { n } ( \widehat { \gamma } _ { n } - \gamma _ { n } ^ { \dagger } ) = M ^ { - 1 } \frac { U _ { n } ( \gamma _ { n } ^ { \dagger } ) } { \sqrt { n } } + o _ { L ^ { 2 } } ( 1 ) .
$$

Partition $\boldsymbol { M } = \left( \begin{array} { c c } { \boldsymbol { A } } & { \boldsymbol { C } } \\ { \boldsymbol { C } ^ { \top } } & { \boldsymbol { D } } \end{array} \right)$ , set $I _ { \mathrm { e f f } } = A - C D ^ { - 1 } C ^ { \top } , L = [ I _ { d } ~ - C D ^ { - 1 } ]$ , and $\Omega _ { \mathrm { e f f } } = L \Omega L ^ { \top }$ . Then, with $G ^ { \dagger } = \nabla ^ { 2 } F ( \tilde { \theta } ^ { \dagger } )$

$$
n \mathbb { E } D _ { F } ( \theta _ { n } ^ { \dagger } , \widehat { \theta } _ { n } ) \longrightarrow \Psi _ { \mathrm { s o f t } } : = \frac { 1 } { 2 } \operatorname { t r } \left( G ^ { \dagger } I _ { \mathrm { e f f } } ^ { - 1 } \Omega _ { \mathrm { e f f } } I _ { \mathrm { e f f } } ^ { - 1 } \right) .
$$

Ifevery row is additionally mean-correct, $\mathbb { E } [ Y _ { n j } \mid v _ { n j } ] = q _ { n j } ( \gamma _ { n } ^ { \dagger } )$ , then

$$
\Omega \preceq M , \qquad \Omega _ { \mathrm { e f f } } \preceq I _ { \mathrm { e f f } } , \qquad \Psi _ { \mathrm { s o f t } } \leq \frac { 1 } { 2 } \operatorname { t r } \left( G ^ { \dagger } I _ { \mathrm { e f f } } ^ { - 1 } \right) .
$$

For correctly specified Bernoulli feedback, all three inequalities become equalities.

Proof. The assumed asymptotic linear expansion gives target covariance $P _ { \theta } M ^ { - 1 } \Omega M ^ { - 1 } P _ { \theta } ^ { \top }$ . Since $L M = \left[ I _ { \mathrm { e f f } } \left( \begin{array} { l l } { } \end{array} \right) \right]$ , block elimination gives $P _ { \theta } M ^ { - 1 } = I _ { \mathrm { e f f } } ^ { - 1 } L .$ , which yields the stated sandwich covariance and, after the local Bregman expansion, the policy-risk limit.

Under mean correctness and $0 \leq Y _ { n j } \leq 1$

$$
\operatorname { V a r } ( Y _ { n j } \mid v _ { n j } ) = q _ { n j } ( 1 - q _ { n j } ) - \mathbb { E } [ Y _ { n j } ( 1 - Y _ { n j } ) \mid v _ { n j } ] \le q _ { n j } ( 1 - q _ { n j } ) .
$$

Conditional independence and summation of the score-covariance contributions give $\Omega \preceq M$ . Congruence by L gives $\Omega _ { \mathrm { e f f } } \preceq I _ { \mathrm { e f f } }$ and hence the risk bound. For Bernoulli responses, $Y _ { n j } ( 1 - Y _ { n j } ) = 0$ almost surely, so equality is recovered. □

Theorem A.4 (Least-favorable bounded soft feedback). Fix an admissible design $\xi \in \Xi$ and suppose the conditions ofProposition A.3 hold under mean correctness with limiting root $\gamma ^ { \dagger } = \gamma _ { 0 } .$ . Let $\Psi _ { \mathrm { s o f t } } ( \xi ; P )$ denote the limiting sandwich policy-risk coefficient under a conditionally independent response law P with the same conditional means and responses in $[ 0 , 1 ] .$ . The supremum over all such response lawsfor which the asymptotic expansion in Proposition A.3 holds is

$$
\operatorname* { s u p } _ { P } \Psi _ { \mathrm { s o f t } } ( \xi ; P ) = \Phi _ { \kappa } ( \xi ) .
$$

The supremum is attained by conditionally Bernoulli feedback with the same conditional means. Consequently,

$$
\arg \operatorname* { m i n } _ { \xi \in \Xi } \operatorname* { s u p } _ { P } \Psi _ { \mathrm { s o f t } } ( \xi ; P ) = \arg \operatorname* { m i n } _ { \xi \in \Xi } \Phi _ { \kappa } ( \xi ) .
$$

Thus the NAOD criterion minimizes the worst-case leading sandwich policy-risk coefficient over the bounded mean-correct soft-feedback class.

Proof. Fix ξ. Under mean correctness, the logistic curvature depends on the conditional means but not on the conditional variances of the responses. Hence all response laws considered in the theorem have the same limiting curvature $M _ { \kappa } ( \boldsymbol { \xi } )$ and the same effective information $I _ { \mathrm { e f f } , \kappa } ( \boldsymbol { \xi } )$

For any such response law P, Proposition A.3 gives

$$
\Omega _ { P } \preceq M _ { \kappa } ( \xi ) .
$$

With $L = [ I _ { d } - C _ { \xi } D _ { \xi } ^ { - 1 } ]$ , congruence preserves the Loewner order and therefore

$$
\Omega _ { \mathrm { e f f } , P } = L \Omega _ { P } L ^ { \top } \preceq L M _ { \kappa } ( \boldsymbol { \xi } ) L ^ { \top } = I _ { \mathrm { e f f } , \kappa } ( \boldsymbol { \xi } ) .
$$

Since $G _ { 0 } \succeq 0$ and $I _ { \mathrm { e f f } , \kappa } ( { \boldsymbol { \xi } } ) \succ 0$

$$
\Psi _ { \mathrm { s o f t } } ( \xi ; P ) \leq \frac { 1 } { 2 } \operatorname { t r } \bigl ( G _ { 0 } I _ { \mathrm { e f f , } \kappa } ( \xi ) ^ { - 1 } \bigr ) = \Phi _ { \kappa } ( \xi ) .
$$

Now take conditionally Bernoulli responses with the same conditional means. Then

$$
\operatorname { V a r } ( Y _ { n j } \mid v _ { n j } ) = q _ { n j } ( \gamma _ { 0 } ) \{ 1 - q _ { n j } ( \gamma _ { 0 } ) \} ,
$$

so the score covariance equals the logistic curvature. Hence $\Omega _ { P } = M _ { \kappa } ( \xi )$ and $\Omega _ { \mathrm { e f f } , P } = I _ { \mathrm { e f f } , \kappa } ( \boldsymbol { \xi } )$ which gives $\Psi _ { \mathrm { s o f t } } ( \xi ; P ) = \Phi _ { \kappa } ( \xi )$ . The upper bound is therefore attainable and the first claim follows.

Because the equality $\mathrm { s u p } _ { P } \Psi _ { \mathrm { s o f t } } ( \xi ; P ) = \Phi _ { \kappa } ( \xi )$ holds pointwise for every admissible design, minimizing both sides over Ξ gives the stated equality of minimizer sets. □

The Arena experiment in Section 6.2 uses the archived soft target $\begin{array} { r } { Y _ { J } = p _ { A } + \frac { 1 } { 2 } p _ { T } \in [ 0 , 1 ] } \end{array}$ . NAOD therefore uses the same logistic curvature geometry as in the Bernoulli theory without identifying curvature with score covariance. Under mean correctness, Proposition A.3 gives the corresponding sandwich risk, while Theorem A.4 shows that the population curvature criterion underlying NAOD is the worst-case leading sandwich policy-risk coefficient over bounded mean-correct soft feedback. Under mean misspecification, the sandwich result describes fluctuations around the joint population root rather than eliminating displacement of that root from the human target.

## B THEORY FOR NUISANCE-ADJUSTED OPTIMAL DESIGN

This appendix develops the theory used in Section 3 and Sections 4.1–4.3. We first connect acquisition-dependent residual exposure to target displacement and policy loss, then prove the conditional local minimax characterization and attainment of the NAOD criterion. We finally give the explicit separation from Target-info and formalize the target–nuisance coupling diagnostic used in the empirical analysis.

## B.1 ACQUISITION-DEPENDENT RESIDUAL EXPOSURE

Section 3 defines the probability-scale judge discrepancy $\delta ( z ) = q ( z ) - p ^ { \star } ( z )$ and the acquired residual score $b _ { \xi } = \mathbb { E } _ { \xi } [ \delta ( Z ) X ( Z ) ]$ ]. The change-of-measure identity in equation 5 does not require a parametric model for the judge discrepancy. We now give its consequences for a nuisance-ignorant reward fit.

Suppose an interior population root $\theta _ { \xi }$ satisfies $\mathbb { E } _ { \xi } [ ( q - p _ { \theta _ { \xi } } ) X ] = 0$ , and write $\Delta = \theta _ { \xi } - \theta ^ { \star }$ . Define the integrated statistical and policy curvatures

$$
H _ { \xi } = \int _ { 0 } ^ { 1 } \mathbb { E } _ { \xi } \big [ \sigma ^ { \prime } \big ( X ^ { \top } ( \theta ^ { \star } + t \Delta ) \big ) X X ^ { \top } \big ] d t , \qquad G _ { \xi } = 2 \int _ { 0 } ^ { 1 } ( 1 - t ) \nabla ^ { 2 } F ( \theta _ { \xi } - t \Delta ) d t .
$$

Subtracting the population score equations and integrating the logistic derivative along the segment from $\theta ^ { \star }$ to $\theta _ { \xi }$ gives $H _ { \xi } \Delta = b _ { \xi }$ . Hence, whenever $H _ { \xi } \succ 0$

$$
\theta _ { \xi } - \theta ^ { \star } = { \cal H } _ { \xi } ^ { - 1 } b _ { \xi } , \qquad { \cal D } _ { F } ( \theta ^ { \star } , \theta _ { \xi } ) = \frac 1 2 b _ { \xi } ^ { \top } { \cal H } _ { \xi } ^ { - 1 } G _ { \xi } { \cal H } _ { \xi } ^ { - 1 } b _ { \xi } .
$$

Thus policy impact depends on how the residual aligns with reward-score directions under the acquired law rather than on residual magnitude alone.

The joint target–nuisance model gives a complementary local interpretation. Holding the human parameter at $\theta _ { 0 } ,$ , define $\begin{array} { r } { b _ { \xi } ( a ) \ = \ \sum _ { i = 1 } ^ { L } \xi _ { i } \{ q _ { \theta _ { 0 } , a } ( z _ { i } ) \ - \ p _ { \theta _ { 0 } } ( z _ { i } ) \} X _ { i } } \end{array}$ . Differentiation gives $\partial b _ { \xi } ( a ) / \partial a ^ { \top } = C _ { \xi } ( a )$ , as in equation 9. Since $b _ { \xi } ( 0 ) = 0$ , the fundamental theorem of calculus yields

$$
b _ { \xi } ( a ) = \int _ { 0 } ^ { 1 } C _ { \xi } ( t a ) a d t , \qquad b _ { \xi } \left( a _ { 0 } + \frac { v } { \sqrt { n } } \right) - b _ { \xi } ( a _ { 0 } ) = \frac { C _ { \xi } ( a _ { 0 } ) v } { \sqrt { n } } + { \cal O } ( n ^ { - 1 } ) .
$$

Hence the same cross-information that produces the Schur-complement adjustment in equation 8 also controls the local sensitivity of the exposed target score.

## B.2 CONDITIONAL LOCAL MINIMAX RISK AND ATTAINMENT

We condition throughout on the historical information ${ \mathcal { F } } _ { P } ,$ so the representation W, allocation $\xi ,$ and all design quantities are fixed before the current outcomes are observed. Write $M = M _ { \kappa } ( { \boldsymbol \xi } )$ as in equation 7 and $G _ { e } = \mathrm { d i a g } ( G _ { 0 } , 0 )$ .

Preliminary estimation and projected one-step estimator. For attainment, assume $\kappa > 0 .$ , a common positive lower bound on normalized trusted likelihood curvature over $\kappa _ { \theta } .$ , a common positive lower bound on the weighted nuisance Gram matrix $\begin{array} { r } { \sum _ { i } ( n _ { i } / n ) W _ { i } W _ { i } ^ { \top } } \end{array}$ , and a fixed positive distance of the local truths from the boundary of $\mathcal { K } = \mathcal { K } _ { \theta } \times \overline { { \mathcal { K } } } _ { a } ^ { \ast }$ . Let $c _ { s } > 0$ be smaller than one half of a valid asymptotic lower bound on $\lambda _ { \operatorname* { m i n } } ( \bar { M } )$

Let $\widetilde { \theta } _ { n }$ minimize the trusted negative log likelihood on $\scriptstyle { \mathcal { K } } _ { \theta }$ . Holding its target contribution fixed as an offset, let

$$
\widetilde { a } _ { n } \in \arg \operatorname* { m i n } _ { a \in \mathcal { K } _ { a } } \sum _ { i = 1 } ^ { L } \sum _ { \ell = 1 } ^ { n _ { i } } \left\{ \mathrm { s p } ( X _ { i } ^ { \top } \widetilde { \theta } _ { n } + W _ { i } ^ { \top } a ) - Y _ { i \ell } ( X _ { i } ^ { \top } \widetilde { \theta } _ { n } + W _ { i } ^ { \top } a ) \right\} .
$$

Set $\widetilde { \gamma } _ { n } = ( \widetilde { \theta } _ { n } ^ { \top } , \widetilde { a } _ { n } ^ { \top } ) ^ { \top }$ . If the declared nuisance-curvature check fails, a fixed interior anchor may be used. Under the stated conditions, this event is asymptotically negligible. The guarded one-step update is

$$
\gamma _ { n } ^ { \circ } = \left\{ \begin{array} { l l } { \widetilde { \gamma } _ { n } + I _ { n } ( \widetilde { \gamma } _ { n } ) ^ { - 1 } U _ { n } ( \widetilde { \gamma } _ { n } ) , } & { \lambda _ { \operatorname* { m i n } } \{ I _ { n } ( \widetilde { \gamma } _ { n } ) / n \} \ge c _ { s } , } \\ { \widetilde { \gamma } _ { n } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad \quad \widehat { \gamma } _ { n } = \mathrm { P r o j } _ { \mathcal { K } } ( \gamma _ { n } ^ { \circ } ) ,
$$

with target output $\widehat { \theta } _ { n } = P _ { \theta } \widehat { \gamma } _ { n }$ . The preliminary and one-step update may use the same current observations. No independence between them is assumed.

Strong convexity of the trusted likelihood and the bounded-score moment bound from Appendix A.2 give $\lVert \widetilde { { \boldsymbol { \theta } } } _ { n } - { \boldsymbol { \theta } } _ { n , h } \rVert _ { L ^ { 4 } } = O ( n ^ { - 1 / 2 } )$ . At the true nuisance coefficient, the normalized offset score is a centered judge-score average plus a term Lipschitz in $\widetilde { \theta } _ { n } - \theta _ { n , h }$ . The nuisance Gram lower bound and the common interior probability bound therefore give $\lVert \widetilde { \boldsymbol { a } } _ { n } - \boldsymbol { a } _ { n , v } \rVert _ { L ^ { 4 } } = O ( n ^ { - 1 / 2 } )$ , and hence $\| \widetilde { \gamma } _ { n } - \gamma _ { n , \eta } \| _ { L ^ { 4 } } = O ( n ^ { - 1 / 2 } )$ uniformly on every fixed local parameter ball.

To obtain the one-step expansion, put $e _ { n } = \widetilde { \gamma } _ { n } - \gamma _ { n , \eta } , \widehat { H } _ { n } = I _ { n } ( \widetilde { \gamma } _ { n } ) / n$ , and $\begin{array} { r } { \overline { { H } } _ { n } = \int _ { 0 } ^ { 1 } I _ { n } ( \gamma _ { n , \eta } + } \end{array}$ $t e _ { n } ) / n d t$ . Since the derivative of the score is minus the information, on the guard event the Newton update satisfies the exact identity

$$
\sqrt { n } ( \gamma _ { n } ^ { \circ } - \gamma _ { n , \eta } ) = \widehat { H } _ { n } ^ { - 1 } \frac { U _ { n } ( \gamma _ { n , \eta } ) } { \sqrt { n } } + \widehat { H } _ { n } ^ { - 1 } ( \widehat { H } _ { n } - \overline { { H } } _ { n } ) \sqrt { n } e _ { n } .
$$

Normalized Hessians are uniformly Lipschitz on the compact neighborhood, so $\| { \widehat { H } } _ { n } - { \overline { { H } } } _ { n } \| _ { \mathrm { o p } } \leq$ $C \| e _ { n } \|$ . The second term is therefore $O _ { L ^ { 2 } } ( n ^ { - 1 / 2 } )$ . Hessian concentration from Appendix $\mathbf { A . } 2 .$ local information convergence, and the fixed spectral guard permit replacing $\widehat { H } _ { n } ^ { - 1 }$ by $M ^ { - 1 }$ . The guard complement has vanishing squared contribution by the fourth-moment bound, and the fixed boundary margin makes the final projection asymptotically inactive in mean square. Consequently,

$$
\sqrt { n } ( \widehat { \gamma } _ { n } - \gamma _ { n , \eta } ) = M ^ { - 1 } \frac { U _ { n } ( \gamma _ { n , \eta } ) } { \sqrt { n } } + o _ { L ^ { 2 } } ( 1 )
$$

uniformly on bounded local parameter sets.

The same argument will be used in Appendix D.2 around a nearby population root rather than the correctly specified center. More generally, if an interior population root $\gamma _ { n } ^ { \dagger }$ has normalized expected Hessian and score covariance converging to the same $M \succ 0$ , and a preliminary satisfies $r _ { n } \overset { \cdot } { = } \Vert \widetilde { \gamma } _ { n } - \gamma _ { n } ^ { \dagger } \Vert _ { L ^ { 4 } } = o ( n ^ { - 1 / 4 } )$ , then the nonlinear Newton remainder is $O _ { L ^ { 2 } } ( \sqrt { n } r _ { n } ^ { 2 } ) = o ( 1 )$ and

$$
\sqrt { n } ( \widehat { \gamma } _ { n } - \gamma _ { n } ^ { \dagger } ) = M ^ { - 1 } \frac { U _ { n } ( \gamma _ { n } ^ { \dagger } ) } { \sqrt { n } } + o _ { L ^ { 2 } } ( 1 ) .
$$

Conditional local minimax lower bound. We now prove the lower-bound part of Theorem 4.1. Assume first $G _ { 0 } \succ 0$ . Fix $\varepsilon \in ( 0 , 1 )$ and a bounded local parameter set. Continuity of the policy Hessian gives a neighborhood of $\theta _ { 0 }$ on which $\nabla ^ { 2 } F \succeq ( 1 - \dot { \varepsilon } ) G _ { 0 }$ . Project the rescaled output of an arbitrary reward estimator onto a fixed $G _ { 0 }$ -metric ball containing the target local parameters. Metric projection cannot increase its quadratic distance to any point in this ball. For decisions outside the local neighborhood, convexity together with positive local curvature gives a fixed positive policy loss, whereas the projected quadratic loss remains bounded. Hence, for all sufficiently large n,

$$
n D _ { F } \bigg ( \theta _ { 0 } + \frac { h } { \sqrt { n } } , T _ { n } \bigg ) \geq \frac { 1 - \varepsilon } { 2 } \| \overline { { T } } _ { n } - h \| _ { G _ { 0 } } ^ { 2 } ,
$$

where ${ \overline { { T } } } _ { n }$ denotes the corresponding localized rescaled decision.

Place a smooth product prior on η supported on $( - t , t ) ^ { k }$ , with prior Fisher information $\pi ^ { 2 } t ^ { - 2 } I _ { k }$ Let $E _ { n } = \overline { { T } } _ { n } - P _ { \theta } \eta$ and let $S _ { n }$ be the joint score of the conditional likelihood and this local prior. Integration by parts in the local parameter gives $\mathbb { E } [ E _ { n } S _ { n } ^ { \top } ] = P _ { \theta }$ . Writing $J _ { n } = \mathbb { E } [ S _ { n } S _ { n } ^ { \top } ]$ , positive semidefiniteness of

$$
\mathbb { E } \big [ ( E _ { n } - P _ { \theta } J _ { n } ^ { - 1 } S _ { n } ) ( E _ { n } - P _ { \theta } J _ { n } ^ { - 1 } S _ { n } ) ^ { \top } \big ]
$$

therefore gives

$$
\mathbb { E } [ E _ { n } E _ { n } ^ { \top } ] \succeq P _ { \theta } J _ { n } ^ { - 1 } P _ { \theta } ^ { \top } , \qquad J _ { n } = \mathbb { E } _ { \eta } \mathcal { T } _ { n } ( \eta ) + \pi ^ { 2 } t ^ { - 2 } I _ { k } ,
$$

where $\mathcal { T } _ { n } ( \eta )$ is Fisher information for the local coordinate η. The regularity conditions in $\mathsf { A p - }$ pendix A.1 imply that ${ \mathcal { T } } _ { n } ( \eta ) \to M$ uniformly on every fixed prior support. Taking the trace against $G _ { 0 }$ , using worst-case risk to dominate prior-averaged risk, and then letting $n \to \infty , t \to \infty .$ , and $\varepsilon \downarrow 0$ yields equation 11, because equation 19 gives $P _ { \theta } M ^ { - 1 } P _ { \theta } ^ { \top } = I _ { \mathrm { e f f } , \kappa } ^ { - 1 }$ . When $G _ { 0 } \succeq 0$ is singular, Proposition $\mathrm { A } . 2$ removes the exact policy-null coordinates and the same argument applies on the policy-relevant quotient.

Attainment. Under correct specification, $U _ { n } ( \gamma _ { n , \eta } ) / \sqrt { n }$ is centered, its covariance converges to M, and its fourth moments are uniformly bounded. Combining the one-step expansion with the local Bregman expansion in equation 18 and the resulting uniform integrability gives

$$
n \mathbb { E } _ { \eta } D _ { F } ( \theta _ { n , h } , \widehat { \theta } _ { n } ) \longrightarrow \frac { 1 } { 2 } \operatorname { t r } ( G _ { e } M ^ { - 1 } ) = \Phi _ { \kappa } ( \xi )
$$

uniformly on every fixed local parameter ball. Thus the projected one-step estimator attains the lower-bound constant in Theorem 4.1.

## B.3 SEPARATION AND TARGET–NUISANCE COUPLING

Proposition B.1 (Separation from Target-info). Let $\lambda > 1$ , and consider the scalar two-type model with (X , X ) = (2λ, 2), (W , W ) = (2, −2λ), θ = a = 0, and $H _ { c } = ( \lambda ^ { 2 } + \bar { 1 } \bar { ) } / 2$ . For unrestricted allocation with $\kappa > 0$ and $G _ { 0 } > 0 $ , Target-info selects $\xi _ { \mathrm { T } } = ( 1 , 0 )$ , whereas NAOD selects $\xi ^ { \star } = ( \lambda , 1 ) / ( \lambda + 1 )$ . Their leading risk ratio is

$$
\frac { \Phi _ { \kappa } ( \xi _ { \mathrm { T } } ) } { \Phi _ { \kappa } ( \xi ^ { \star } ) } = 1 + \frac { 2 ( \lambda ^ { 2 } + 1 ) } { \kappa ( \lambda + 1 ) ^ { 2 } } > 1 .
$$

Proof. At the zero center, every judge variance is $1 / 4 . \mathrm { ~ H ~ } x \in [ 0 , 1 ]$ denotes the fraction of judge labels assigned to the first type, the two judge information atoms are

$$
J _ { 1 } = \left( \begin{array} { l l } { { \lambda ^ { 2 } } } & { { \lambda } } \\ { { \lambda } } & { { 1 } } \end{array} \right) , \qquad J _ { 2 } = \left( \begin{array} { l l } { { 1 } } & { { - \lambda } } \\ { { - \lambda } } & { { \lambda ^ { 2 } } } \end{array} \right) .
$$

The raw target information is $: H _ { c } + x \lambda ^ { 2 } + ( 1 - x )$ , which is strictly increasing in x because $\lambda > 1$ Hence Target-info selects $x = 1$

For the judge block $J ( x ) = x J _ { 1 } + ( 1 - x ) J _ { 2 }$ , the nuisance entry is $D ( x ) = x + \lambda ^ { 2 } ( 1 - x )$ and det $J ( x ) { \stackrel { - } { = } } { \bar { x } } ( 1 - x ) ( \lambda ^ { 2 } + 1 ) ^ { 2 }$ . Because trusted information enters only the target block, the effective target information is

$$
I _ { \mathrm { e f f } } ( x ) = \kappa H _ { c } + \frac { x ( 1 - x ) ( \lambda ^ { 2 } + 1 ) ^ { 2 } } { x + \lambda ^ { 2 } ( 1 - x ) } .
$$

The derivative of the second term has roots $\lambda / ( \lambda + 1 )$ and $\lambda / ( \lambda - 1 )$ Only the first lies in $[ 0 , 1 ]$ and the derivative changes from positive to negative there. Thus NAOD selects $x ^ { \star } = \lambda / ( \lambda + \bar { 1 } )$ . At this allocation, $I _ { \mathrm { e f f } } ( x ^ { \star } ) \stackrel {  } { = } \kappa H _ { c } + ( \lambda ^ { 2 } + 1 ) ^ { 2 } \stackrel {  } { = } / ( \lambda + 1 ) ^ { 2 }$ , whereas $I _ { \mathrm { e f f } } ( 1 ) = \kappa H _ { c }$ . Since the scalar leading policy risk is $\dot { G } _ { 0 } / \{ 2 I _ { \mathrm { e f f } } ( x ) \}$ by equation 10, substituting $H _ { c } \dot { = } ( \lambda ^ { 2 } + 1 ) / 2$ gives the stated ratio. □

This construction isolates the acquisition criterion because both rules use the same correctly specified joint model and final estimator, while Target-info favors the type with the largest raw target block even when that type cannot separate target from nuisance variation.

Target–nuisance coupling. For any positive-definite design, define

$$
K _ { \xi } = A _ { \kappa } ( \xi ) ^ { - 1 / 2 } C _ { \xi } D _ { \xi } ^ { - 1 / 2 } , \qquad \rho ^ { 2 } ( \xi ) = \| K _ { \xi } \| _ { \mathrm { o p } } ^ { 2 } .
$$

Positive definiteness of the Schur complement implies $0 \le \rho ^ { 2 } ( \xi ) < 1$ . By the whitening identity in equation 20, if $\rho _ { j } ^ { 2 }$ are the eigenvalues of $K _ { \xi } K _ { \xi } ^ { \top }$ , then $1 - \rho _ { j } ^ { 2 }$ are the corresponding fractions of whitened target information remaining after nuisance adjustment. Thus $\rho ^ { 2 } ( \xi )$ records the strongest target–nuisance coupling. Values near one indicate a target-score direction that is nearly reproducible by nuisance variation and for which raw Target-info can substantially overstate usable target information.

This coupling measure is geometric rather than policy-weighted, so its consequence for policy risk also depends on whether the coupled target directions receive substantial weight under $G _ { 0 }$ . The Arena analysis in Figure 5 and Table 12 evaluates $\rho ^ { 2 }$ for the Target-info-selected design and uses it as a diagnostic of target–nuisance confounding rather than as a finite-sample prediction of the realized NAOD gain.

## C FINITE-POOL SELECTION AND COMPUTATION

This appendix gives the optimization, statistical, and computational details for the finite-pool procedure in Section 4.4. We first justify the convex relaxation and its executable optimality certificate, then prove Theorem 4.2 directly for the realized set of distinct comparisons. We finally separate plug-in error from numerical optimization error and record the computational cost of the implementation.

## C.1 FINITE-POOL OPTIMIZATION AND CERTIFICATES

For the fitted finite-pool criterion in equation 12, write ${ \widehat { \cal T } } _ { x } = { \widehat { \cal T } } ( x )$ The seed $S _ { 0 }$ is chosen only when needed to keep fitted information positive definite. Because all information atoms are positive semidefinite, positive definiteness of the trusted-plus-seed information then implies $\widehat { \cal T } _ { x } \succ 0$ for every $x \in \mathcal { X } _ { B }$

The map $M \mapsto { \textstyle \frac { 1 } { 2 } } \operatorname { t r } ( G M ^ { - 1 } )$ is convex for $G \succeq 0$ on the positive-definite cone. Indeed, for a symmetric perturbation H,

$$
\frac { d ^ { 2 } } { d t ^ { 2 } } \left. \frac 1 2 \mathrm { t r } \big \{ G ( M + t H ) ^ { - 1 } \big \} \right| _ { t = 0 } = \mathrm { t r } \big ( G M ^ { - 1 } H M ^ { - 1 } H M ^ { - 1 } \big ) \ge 0 .
$$

Since $\widehat { \cal T } ( x )$ is affine in x, $f _ { B }$ is convex on $\mathcal { X } _ { B }$ . Differentiation gives the gradient used by the linear oracle.

For a feasible relaxed point x, define the full Frank–Wolfe gap

$$
g _ { \mathrm { F W } } ( x ) = \operatorname* { m a x } _ { u \in \mathcal { X } _ { B } } \nabla f _ { B } ( x ) ^ { \top } ( x - u ) .
$$

Convexity gives $\begin{array} { r } { 0 \le f _ { B } ( x ) - \operatorname* { m i n } _ { u \in \mathcal { X } _ { B } } f _ { B } ( u ) \le g _ { \mathrm { F W } } ( x ) } \end{array}$ . The word full is important because the linear oracle must optimize over the complete declared feasible set. Under the group constraints in Section 4.4, this oracle is exact. After fixing the seed coordinates, it selects the smallest gradient coefficient in each available group and then the required number of groups with the smallest such coefficients.

Let $\widehat { x }$ be the final relaxed iterate and let $S$ be the returned feasible subset after rounding and exchanges. Write $f _ { B } ( S ) = f _ { B } ( \mathbf { 1 } _ { S } )$ , and let $f _ { B , \mathrm { i n t } } ^ { \star }$ denote the minimum over feasible integer selections. Since every integer design is feasible for the relaxation,

$$
0 \leq f _ { B } ( S ) - f _ { B , \mathrm { i n t } } ^ { \star } \leq f _ { B } ( S ) - f _ { B } ( \widehat { x } ) + g _ { \mathrm { F W } } ( \widehat { x } ) .\tag{21}
$$

Algorithm 1 NAOD finite-pool acquisition   
Input Fitted information atoms $\{ \widehat { J } _ { i } \} _ { i = 1 } ^ { N } ,$ policy weight $\widehat { G } _ { e } ,$ , trusted information $\widehat { I } _ { c }$ , budget B,   
feasible set $\mathcal { X } _ { B } ,$ , seed $S _ { 0 } ,$ tolerance $\varepsilon > 0$ , and iteration limit $T _ { \mathrm { m a x } } .$   
1: Choose x $\in { \mathcal { X } } _ { B }$ with ${ \widehat { \cal T } } ( x ) \succ 0 .$   
2: for $t = 1 , \dots , T _ { \mathrm { m a x } }$ do   
3: Compute $s _ { t } \in$ arg min $_ { \cdot u \in \mathcal { X } _ { B } } \nabla f _ { B } ( x ) ^ { \top } u .$   
4: if $\nabla f _ { B } ( x ) ^ { \top } ( x - s _ { t } ) \leq \varepsilon$ then   
5: break   
6: end if   
7: Choose $\begin{array} { r } { \alpha _ { t } \in \arg \operatorname* { m i n } _ { \alpha \in [ 0 , 1 ] } f _ { B } ( x + \alpha ( s _ { t } - x ) ) . } \end{array}$   
8: Update $x  x + \alpha _ { t } ( s _ { t } - x )$   
9: end for   
10: Set ${ \widehat { x } } \gets x$ and recompute $g _ { \mathrm { F W } } ( \widehat { x } ) .$   
11: Round xb by largest weights to a feasible set $S$ of size B, including $S _ { 0 }$ and respecting all group   
capacities.   
12: Apply feasible exchanges that decrease $f _ { B } ( \boldsymbol { S } )$ and preserve positive-definite fitted information.   
13: Freeze S before observing its judge labels.   
14: return S and the certificate in equation 21.

Thus the final Frank–Wolfe gap and the actual rounding-and-exchange change give a computable certificate for the returned subset under the frozen fitted criterion. This is an optimization certificate, not a statistical confidence interval and not a certificate that the fitted information equals population information. The complete optimization and integerization procedure is summarized in Algorithm 1.

The final gap is recomputed regardless of whether optimization stops by tolerance or by the iteration limit. Reaching the limit is not treated as convergence. Improving exchanges can only reduce the certificate because equation 21 is evaluated at the final returned subset. The selected IDs and their fitted information are frozen before any selected judge outcome is revealed.

## C.2 STATISTICAL GUARANTEES FOR DISTINCT COMPARISONS

We now prove Theorem 4.2. Unlike the fixed-support analysis in Appendix B.2, the selected comparisons need not repeat a finite set of feature types. The proof therefore retains each selected row and works with the realized information matrix $M _ { B }$ appearing in the theorem.

Distinct-comparison regularity. Condition on the historical data, candidate features, and the selected set $S _ { B }$ , all fixed before the selected outcomes are observed. The dimensions d and r are fixed. Selected and trusted feature rows are uniformly bounded. Current labels are conditionally independent and correctly specified under the joint Bernoulli model. The parameter centers lie a fixed positive distance from the boundary of a common compact product set. Assume $n _ { c , B } / B$ is bounded above and away from zero, trusted likelihood curvature has a common positive lower bound, and $\lambda _ { \operatorname* { m i n } } ( M _ { B } ) \geq \mathbf { \bar { \mu } } > 0$ . The guards, projection region, and derivative bounds are the same as in Appendix B.2.

For $\gamma _ { B , \eta } = \gamma _ { 0 } + \eta / \sqrt { B }$ , let $M _ { B } ( \eta )$ be the expected joint information divided by B at the local truth. Bounded rows and bounded logistic derivatives imply, for every fixed $R < \infty ,$

$$
\operatorname* { s u p } _ { \| \eta \| \leq R } \| M _ { B } ( \eta ) - M _ { B } \| _ { \mathrm { o p } } = O _ { R } ( B ^ { - 1 / 2 } ) .
$$

No convergence of $M _ { B }$ is needed for this comparison.

Let $U _ { B }$ denote the total joint score from the trusted block and the selected judge records, and define $\Delta _ { B , \eta } \ : = \ : U _ { B } ( \gamma _ { B , \eta } ) / \sqrt { B }$ There are $O ( B )$ independent bounded score summands, so $\Delta _ { B , \eta }$ has uniformly bounded fourth moments on bounded local parameter sets, and the normalized observed information has $L ^ { 4 }$ fluctuation $O ( B ^ { - 1 / 2 } )$ when trusted covariates are random. Trusted curvature gives the target preliminary an $O _ { L ^ { 4 } } ( B ^ { - 1 / 2 } )$ error. Moreover, the nuisance principal block of $M _ { B }$ is at least $\mu I _ { r }$ . Since logistic variances are at most $\begin{array} { r } { 1 / 4 , B ^ { - 1 } \sum _ { i \in S _ { B } } W _ { B i } { W _ { B i } ^ { \top } } \succeq 4 \dot { \mu } I _ { r } } \end{array}$ , which supplies the strong convexity required by the offset nuisance fit. The full preliminary is therefore root-B in $L ^ { 4 }$

Applying the exact Newton expansion from Appendix B.2 with the finite-array information retained gives

$$
\sqrt { B } ( \widehat { \gamma } _ { B } - \gamma _ { B , \eta } ) = M _ { B } ( \eta ) ^ { - 1 } \Delta _ { B , \eta } + o _ { L ^ { 2 } } ( 1 )
$$

uniformly over bounded η and eligible frozen selected arrays. The guard complement and final projection are negligible by the common information floor, fourth-moment bounds, and boundary margin.

Under correct specification, $\mathrm { C o v } ( \Delta _ { B , \eta } ) = M _ { B } ( \eta )$ . Hence the leading expected policy quadratic form is $\scriptstyle { \frac { 1 } { 2 } } \operatorname { t r } \{ G _ { e } M _ { B } ( \eta ) ^ { - 1 } \}$ The preceding $O _ { R } ( B ^ { - 1 / 2 } )$ information comparison, the inverseinformation bound, and the local Bregman expansion in equation 18 then yield exactly equation 13, uniformly over bounded local parameters. This proves the risk statement in Theorem 4.2 without requiring the selected empirical proportions to approach a fixed-support allocation.

If $M _ { B }  M _ { \infty } \succ 0$ , the bounded triangular-array central limit theorem and the same third-order likelihood expansion as in Proposition A.1 give a LAN experiment with information $M _ { \infty }$ . The localized lower-bound argument of Appendix B.2 therefore gives the corresponding conditional local minimax bound. If $G _ { 0 }$ is singular, Proposition A.2 applies on the policy-relevant quotient.

The theorem is conditional on the realized selected set and uses its true information. It therefore preserves the statistical effect of rounding, but it does not assert that every rounded set satisfies the information floor, nor does fitted positive definiteness imply true positive definiteness. Those are separate plug-in questions addressed next.

## C.3 PLUG-IN ERROR, NUMERICAL ACCURACY, AND COMPUTATION

Plug-in error. Define the population counterpart by $\begin{array} { r } { f _ { B } ^ { 0 } ( x ) = \frac { 1 } { 2 } \mathrm { t r } \{ G _ { e } \mathcal { T } _ { 0 } ( x ) ^ { - 1 } \} } \end{array}$ , where $\mathcal { T } _ { 0 } ( x ) =$ $\begin{array} { r } { I _ { c } + \sum _ { i = 1 } ^ { N } x _ { i } J _ { i } } \end{array}$ , and recall that $f _ { B }$ in equation 12 uses the fitted quantities $\widehat { G } _ { e }$ and $\widehat { \cal T } ( x )$ . Suppose both information matrices have eigenvalues at least $\underline { { \mu } } _ { B } > 0$ over the feasible family. The inverse identity and the trace inequality give

$$
| f _ { B } ( x ) - f _ { B } ^ { 0 } ( x ) | \leq \frac { \| \widehat { G } _ { e } - G _ { e } \| _ { * } } { 2 \underline { { \mu } } _ { B } } + \frac { \mathrm { t r } ( G _ { e } ) } { 2 \underline { { \mu } } _ { B } ^ { 2 } } \| \widehat { \mathbb { Z } } ( x ) - \mathcal { Z } _ { 0 } ( x ) \| _ { \mathrm { o p } } .
$$

For example, bounds $\| \widehat { I } _ { c } - I _ { c } \| _ { \mathrm { o p } } \ \leq \ \delta _ { c }$ and max<sub>i</sub> $\| \widehat { J } _ { i } - J _ { i } \| _ { \mathrm { o p } } \leq \delta _ { J }$ imply $\operatorname* { s u p } _ { x \in \mathcal { X } _ { B } } \| \widehat { \mathcal { T } } ( x ) -$ $\mathcal { T } _ { 0 } ( x ) \lVert _ { \mathrm { o p } } \stackrel { \cdot } { \leq } \delta _ { c } + B \delta _ { J }$ because every feasible design has total mass B.

Let $\begin{array} { r } { \Delta _ { B } = \operatorname* { s u p } _ { x \in \mathcal { X } _ { B } } | f _ { B } ( x ) - f _ { B } ^ { 0 } ( x ) | } \end{array}$ . If $S _ { B } ^ { 0 , \star }$ minimizes the population criterion over feasible integer selections, then the fitted optimization certificate and two applications of the uniform plugin bound give

$$
0 \leq f _ { B } ^ { 0 } ( S ) - f _ { B } ^ { 0 } ( S _ { B } ^ { 0 , \star } ) \leq 2 \Delta _ { B } + f _ { B } ( S ) - f _ { B } ( \widehat { x } ) + g _ { \mathrm { F W } } ( \widehat { x } ) .
$$

Thus plug-in error and numerical/rounding error enter separately. A small Frank–Wolfe certificate establishes accurate optimization of the fitted objective. It does not by itself establish that the fitted objective accurately represents the population criterion or that the joint model is correctly specified.

Numerical accuracy of the estimator. The statistical proofs in Appendix B.2 are written for exact trusted and offset preliminary minimizers, but approximate convex solutions preserve the required rates under a quantitative stopping rule. For a normalized convex objective $Q$ with Hessian bounded below by $^ { c I , }$ let $t ^ { \star }$ be its constrained minimizer and define the global first-order gap $g _ { Q } ( t ) ~ =$ max $\boldsymbol { u } \in \mathcal { K } \left. \nabla Q ( t ) ^ { \top } ( t - u ) \right.$ . Strong convexity and convexity give

$$
\frac { c } { 2 } \| t - t ^ { \star } \| ^ { 2 } \leq Q ( t ) - Q ( t ^ { \star } ) \leq g _ { Q } ( t ) .
$$

Hence gaps of order $O ( n _ { c } ^ { - 1 } )$ for the normalized trusted objective and $O ( B ^ { - 1 } )$ for the normalized nuisance objective yield numerical errors of order $O ( n _ { c } ^ { - 1 / 2 } )$ and $O ( B ^ { - 1 / 2 } )$ , respectively, or smaller, leaving the one-step expansions unchanged. Selection line searches need not themselves prove convergence because the final Frank–Wolfe gap is recomputed at the returned relaxed point and is the quantity entering equation 21.

Computation. For $k = d + r$ , each fitted information atom has rank-one form $\widehat { J } _ { i } = \widehat { t } _ { i } \widehat { v } _ { i } \widehat { v } _ { i } ^ { \top }$ Storing $( \widehat { t } _ { i } , \widehat { v } _ { i } )$ ) rather than a dense $k \times k$ matrix for every candidate requires $O ( N k + k ^ { 2 } )$ memory. After factorizing $\widehat { \mathcal { T } } _ { x }$ , the gradient coordinate can be evaluated as

$$
\frac { \partial f _ { B } ( x ) } { \partial x _ { i } } = - \frac { 1 } { 2 } \widehat { t } _ { i } \widehat { v } _ { i } ^ { \top } \widehat { \mathcal { T } } _ { x } ^ { - 1 } \widehat { G } _ { e } \widehat { \mathcal { T } } _ { x } ^ { - 1 } \widehat { v } _ { i } .
$$

A dense full information-and-gradient evaluation costs $O ( N k ^ { 2 } + k ^ { 3 } )$ , with the $k ^ { 3 }$ term coming from the factorization. The implementation uses positive-definite factorizations and linear solves rather than forming matrix inverses explicitly. Total selection time additionally depends on the number of Frank–Wolfe iterations, line-search evaluations, rounding, and improving exchanges.

## D LEARNING THE JUDGE-DEVIATION REPRESENTATION

This appendix proves the representation-learning results in Sections 5.1 and 5.2. We first characterize the target effect of representation error and the exact cancellation of modeled nuisance directions, then prove the finite-sample representation cost in Theorem 5.1 and its design-ranking consequence. We finally give the higher-order population-root expansion that clarifies the limit of first-order cancellation.

## D.1 REPRESENTATION ERROR AND ORACLE RECOVERY

Recall the learned-representation residual $e _ { m }$ in equation 14 and the target influence operator $B _ { m }$ and policy kernel $R _ { m }$ in equation 15. We first justify the first-order interpretation of these quantities.

Proposition D.1 (Local representation-error expansion). Fix the learned representation, nuisance center, and allocation, and suppose the regularity conditions ofAppendices A.1 and B.2 hold with positive joint information $M _ { m }$ . If the true judge logits equal the nominal logits at $\gamma _ { 0 m }$ plus a sufficiently small deterministic vector e ${ \bf \Psi } : \in \mathbb { R } ^ { L }$ , while the trusted model remains correctly specified at $\theta _ { 0 } ,$ , then thefitted population score has a unique local root $\gamma _ { m } ^ { \dag } ( e )$ satisfying

$$
\gamma _ { m } ^ { \dagger } ( e ) - \gamma _ { 0 m } = M _ { m } ^ { - 1 } \sum _ { i = 1 } ^ { L } \xi _ { m , i } t _ { m , i } v _ { m , i } e _ { i } + O ( \Vert e \Vert ^ { 2 } ) .
$$

Consequently,

$$
P _ { \theta } \{ \gamma _ { m } ^ { \dag } ( e ) - \gamma _ { 0 m } \} = B _ { m } e + O ( \Vert e \Vert ^ { 2 } ) .
$$

Proof. Let $\Psi _ { m } ( \gamma , e )$ denote the normalized expected fitted score when the true judge logit at type i is $v _ { m , i } ^ { \top } \gamma _ { 0 m } + \dot { e } _ { i }$ , while the trusted probabilities remain those of $\theta _ { 0 } . \mathrm { ~ A t ~ } ( \gamma _ { 0 m } , 0 )$ , the score is zero, its derivative with respect to $\gamma \ \mathrm { i s } \ - M _ { m }$ , and its derivative with respect to e applied to a direction e is $\textstyle \sum _ { i } \xi _ { m , i } t _ { m , i } v _ { m , i } e _ { i }$ . The inverse-information bound and bounded logistic derivatives give a local implicit-function expansion with the stated linear term and an $O ( \| e \| ^ { 2 } )$ remainder. Taking target coordinates and using the definition of $B _ { m }$ in equation 15 gives the second display. □

The operator $B _ { m }$ annihilates representation errors that lie in the learned nuisance span. In particular, for any $c \in \mathbb { R } ^ { r }$

$$
B _ { m } \widehat { W } _ { m } c = P _ { \theta } M _ { m } ^ { - 1 } \sum _ { i = 1 } ^ { L } \xi _ { m , i } t _ { m , i } v _ { m , i } \widehat { W } _ { m , i } ^ { \top } c = P _ { \theta } M _ { m } ^ { - 1 } M _ { m } \Big ( { 0 } _ { c } \Big ) = 0 ,
$$

so $B _ { m } \widehat { W } _ { m } = 0$ . Thus an error component in the column span of $\widehat { W } _ { m }$ has no first-order effect on the target parameter. More strongly, if $e = \widehat { W } _ { m } c$ and $a _ { 0 m } + c$ remains in the admissible interior neighborhood, then $\gamma _ { 0 m } + ( 0 ^ { \top } , c ^ { \top } ) ^ { \top }$ reproduces every perturbed judge logit exactly while leaving the trusted model unchanged. Such a represented error is therefore absorbed entirely by the nuisance coefficient, although estimating that coefficient still reduces effective target information through equation 8.

Proposition D.1 also gives a generic oracle-recovery condition. Along a deterministic sequence of learned representations and designs, if $\| e _ { m } \| = o ( \bar { n } ^ { - 1 / 4 } )$ and ${ \sqrt { n } } \left\| B _ { m } e _ { m } \right\| \to 0$ , then the target population-root displacement is $o ( n ^ { - 1 / 2 } )$ . The nearby-root one-step expansion in Appendix B.2 then recovers the oracle leading policy risk. For random representation errors, expected-risk recovery additionally requires the moment control used below.

## D.2 FINITE-SAMPLE REPRESENTATION COST AND DESIGN REVERSALS

Representation-learning regularity. Let $\mathcal { F } _ { m }$ denote the historical information used to learn the judge-deviation representation. Conditional on $\mathcal { F } _ { m }$ , the aligned representation $\widehat { W } _ { m }$ , nuisance center $a _ { 0 m }$ , and realized allocation proportions $\xi _ { m , i }$ are fixed before the current outcomes are observed. We assume fixed support and parameter dimensions, common bounded feature and logit ranges, a common compact product parameter set whose nominal centers $\gamma _ { 0 m } = ( \theta _ { 0 } ^ { \top } , a _ { 0 m } ^ { \top } ) ^ { \top }$ remain a fixed positive distance from the boundary, and the trusted-curvature, nuisance-curvature, guard, and projection conditions of Appendix B.2 with common constants. Conditional on ${ \mathcal { F } } _ { m } .$ , current judge outcomes are independent Bernoulli variables with their actual probabilities and are independent of a correctly specified trusted block. The trusted-to-judge sample ratio is bounded above and away from zero, and $M _ { m } \succeq \mu I$ for some fixed $\mu > 0$ . In addition to the convergence assumptions in Theorem 5.1, assume su $\begin{array} { r } { \operatorname { p } _ { m } \mathbb { E } \| \sqrt { m } e _ { m } \| ^ { 4 } < \infty . } \end{array}$

ProofofTheorem 5.1. Condition on $\mathcal { F } _ { m }$ . On the event that $\| e _ { m } \|$ lies in the common local neighborhood, Proposition D.1 gives a population root $\gamma _ { m } ^ { \dagger }$ satisfying

$$
\gamma _ { m } ^ { \dagger } - \gamma _ { 0 m } = M _ { m } ^ { - 1 } \sum _ { i = 1 } ^ { L } \xi _ { m , i } t _ { m , i } v _ { m , i } e _ { m , i } + O ( \Vert e _ { m } \Vert ^ { 2 } ) .
$$

The fourth-moment assumption gives $\mathbb { P } ( \| e _ { m } \| > \varepsilon ) = O ( m ^ { - 2 } )$ for every fixed sufficiently small $\varepsilon > 0$ . Since $n / m$ is bounded and policy loss is bounded on the common compact parameter set, these exceptional historical samples contribute $o ( 1 )$ to the expected policy loss after multiplication by n.

On the local event, let

$$
\Delta _ { m } = \frac { U _ { n } ( \gamma _ { m } ^ { \dagger } ) } { \sqrt { n } } , \qquad A _ { m } ^ { \dagger } = \frac { 1 } { n } \mathbb { E } [ I _ { n } ( \gamma _ { m } ^ { \dagger } ) \mid \mathcal { F } _ { m } ] .
$$

Because $\gamma _ { m } ^ { \dagger }$ is the conditional population root, $\mathbb { E } [ \Delta _ { m } \mid \mathcal { F } _ { m } ] = 0$ . Bounded logistic derivatives and Proposition D.1 imply $A _ { m } ^ { \dagger } = \hat { M } _ { m } ^ { \bullet } + O ( \lVert e _ { m } \rVert )$ and $\operatorname { C o v } ( \Delta _ { m } \mid \mathcal { F } _ { m } ) = M _ { m } + \operatorname { \bar { O } } ( \| e _ { m } \| )$

The trusted-plus-offset preliminary has $L ^ { 4 }$ distance $O ( n ^ { - 1 / 2 } + m ^ { - 1 / 2 } )$ from $\gamma _ { m } ^ { \dagger } .$ Since $n / m$ is bounded, the Newton argument of Appendix B.2, retaining the conditional matrix $A _ { m } ^ { \dagger }$ , gives

$$
\sqrt { n } ( \widehat { \gamma } _ { n , m } - \gamma _ { m } ^ { \dagger } ) = ( A _ { m } ^ { \dagger } ) ^ { - 1 } \Delta _ { m } + o _ { L ^ { 2 } } ( 1 ) .
$$

Combining this expansion with the population-root expansion and taking target coordinates yields

$$
\sqrt { n } ( \widehat { \theta } _ { n , m } - \theta _ { 0 } ) = P _ { \theta } ( A _ { m } ^ { \dagger } ) ^ { - 1 } \Delta _ { m } + \sqrt { n } B _ { m } e _ { m } + o _ { L ^ { 2 } } ( 1 ) .
$$

Indeed, the omitted quadratic root remainder is negligible because $\sqrt { n } \| e _ { m } \| _ { I ^ { 4 } } ^ { 2 } = O ( \sqrt { n } / m ) =$ $o ( 1 )$ . The first leading term has conditional mean zero given $\mathcal { F } _ { m }$ , whereas $\boldsymbol { B _ { m } } \boldsymbol { e _ { m } }$ <sub>m</sub> is F<sub>m</sub>-measurable, $\mathcal { F } _ { m }$ so their cross second moment vanishes.

Using the local policy expansion in equation $^ { 1 8 , }$ the common Hessian bound, and the fourth-moment controls therefore gives the finite-sequence identity

$$
n \mathbb { E } D _ { F } ( \theta _ { 0 } , \widehat { \theta } _ { n , m } ) = \mathbb { E } \Phi _ { m } + \frac { n } { 2 } \mathbb { E } [ e _ { m } ^ { \top } R _ { m } e _ { m } ] + o ( 1 ) .
$$

This identity separates current-sample variation from representation-induced displacement before taking the historical-sample limit.

Write $Z _ { m } = { \sqrt { m } } e _ { m }$ . The information floor uniformly bounds $\Phi _ { m }$ and $\| R _ { m } \| _ { \mathrm { o p } }$ , so $\Phi _ { m } \to \Phi _ { 0 }$ in probability implies $\mathbb { E } \Phi _ { m } \to \Phi _ { 0 }$ . Moreover, $Z _ { m } \Rightarrow Z _ { P }$ and $R _ { m } \ \to \ \stackrel { . } { R } _ { 0 }$ in probability imply $Z _ { m } ^ { \dagger } R _ { m } Z _ { m } \stackrel { . } { \Rightarrow } \bar { Z _ { P } ^ { \intercal } } R _ { 0 } Z _ { P }$ . The uniform fourth-moment bound makes these quadratic forms uniformly integrable, and hence

$$
\mathbb { E } [ Z _ { m } ^ { \top } R _ { m } Z _ { m } ] \longrightarrow \mathbb { E } [ Z _ { P } ^ { \top } R _ { 0 } Z _ { P } ] = \mathrm { t r } ( R _ { 0 } \Sigma _ { P } ) + b _ { P } ^ { \top } R _ { 0 } b _ { P } .
$$

Since $n / m  \lambda _ { P }$ , substitution into the preceding finite-sequence identity and division by n give exactly the risk expansion in equation 16. □

The same expansion gives the ranking result in Section 5.2. Applying equation 16 to NAOD and Target-info under the same representation-learning protocol, label budgets, and final estimator, and subtracting the two risks, gives equation 17. Thus the oracle ordering is recovered when $n / m  0$ whereas for $n / m  \lambda _ { P } > 0$ the design-dependent representation cost can contribute at the same leading order as the oracle sampling risk.

## D.3 HIGHER-ORDER REPRESENTATION EFFECTS

First-order influence does not by itself characterize finite representation error. We therefore record the second-order population-root expansion that determines when cancellation of $\boldsymbol { B _ { m } } \boldsymbol { e _ { m } }$ is sufficient for oracle recovery.

For a fixed regular design, let $V = [ v _ { 1 } , \ldots , v _ { L } ] , \Lambda _ { \xi } = \operatorname { d i a g } ( \xi _ { 1 } , \ldots , \xi _ { L } ) , T = \operatorname { d i a g } ( t _ { 1 } , \ldots , t _ { L } )$ , and define $\begin{array} { r } { \mathcal { L } _ { \xi } = M ^ { - 1 } V \Lambda _ { \xi } T } \end{array}$ . For $v _ { c } = ( X _ { c } ^ { \top } , 0 ^ { \top } ) ^ { \top }$ , define

$$
\begin{array} { l } { { \displaystyle Q _ { J } ( e , f ) = \sum _ { i = 1 } ^ { L } \xi _ { i } \sigma ^ { \prime \prime } ( v _ { i } ^ { \top } \gamma _ { 0 } ) v _ { i } e _ { i } f _ { i } , } } \\ { { \displaystyle Q _ { M } ( h , g ) = \sum _ { i = 1 } ^ { L } \xi _ { i } \sigma ^ { \prime \prime } ( v _ { i } ^ { \top } \gamma _ { 0 } ) v _ { i } ( v _ { i } ^ { \top } h ) ( v _ { i } ^ { \top } g ) + \kappa \mathbb { E } \big [ \sigma ^ { \prime \prime } ( X _ { c } ^ { \top } \theta _ { 0 } ) v _ { c } ( v _ { c } ^ { \top } h ) ( v _ { c } ^ { \top } g ) \big ] . } } \end{array}
$$

Proposition D.2 (Second-order population-root expansion). Suppose the true judge logits are $v _ { i } ^ { \top } \gamma _ { 0 } + e _ { i } ,$ , while the trusted model remains correctly specified at $\theta _ { 0 }$ . For sufficiently small $e ,$ the nearbyfitted population root satisfies

$$
\begin{array} { c } { { \gamma ^ { \dagger } ( e ) - \gamma _ { 0 } = s _ { 1 } ( e ) + s _ { 2 } ( e ) + O ( \left. e \right. ^ { 3 } ) , } } \\ { { s _ { 2 } ( e ) = \displaystyle \frac 1 2 M ^ { - 1 } \{ Q _ { J } ( e , e ) - Q _ { M } ( s _ { 1 } ( e ) , s _ { 1 } ( e ) ) \} , } } \end{array}
$$

where $s _ { 1 } ( e ) = \mathcal { L } _ { \xi } e . \ I f e = W b$ and $a _ { 0 } + b$ remains in the admissible interior neighborhood, then $\gamma ^ { \dagger } ( e ) = \gamma _ { 0 } + ( 0 ^ { \top } , b ^ { \top } ) ^ { \top }$ exactly, so the target displacement vanishes to every order and in particular $s _ { 2 } ( W b ) = 0 .$

Proof. Let $\ell _ { i } = v _ { i } ^ { \top } \gamma _ { 0 }$ . The normalized population score at $\gamma _ { 0 } + h$ under logit perturbation e is

$$
\Psi ( h , e ) = \sum _ { i = 1 } ^ { L } \xi _ { i } \{ \sigma ( \ell _ { i } + e _ { i } ) - \sigma ( \ell _ { i } + v _ { i } ^ { \top } h ) \} v _ { i } + \kappa \mathbb { E } \big [ \{ \sigma ( X _ { c } ^ { \top } \theta _ { 0 } ) - \sigma ( X _ { c } ^ { \top } \theta _ { 0 } + v _ { c } ^ { \top } h ) \} v _ { c } \big ] .
$$

At $( h , e ) = ( 0 , 0 )$ , the derivatives with respect to h and e are −M and $V \Lambda _ { \xi } T$ , respectively, so the local root satisfie $h = O ( \left\| e \right\| )$ . A second-order Taylor expansion of the true-mean and fitted-mean terms gives

$$
0 = V \Lambda _ { \xi } T e + \frac { 1 } { 2 } Q _ { J } ( e , e ) - M h - \frac { 1 } { 2 } Q _ { M } ( h , h ) + O ( \| e \| ^ { 3 } + \| h \| ^ { 3 } ) .
$$

The first-order relation gives $h - s _ { 1 } ( e ) = O ( \| e \| ^ { 2 } )$ . Bilinearity of $Q _ { M }$ then allows $Q _ { M } ( h , h )$ to be replaced by $Q _ { M } ( s _ { 1 } , s _ { 1 } )$ at third-order error, yielding the stated expansion. I $: e = W b ,$ shifting only the nuisance coefficient reproduces the perturbed judge logits exactly and leaves the trusted probabilities unchanged, establishing exact cancellation. □

Proposition D.2 explains why first-order cancellation alone is insufficient at the $n ^ { - 1 / 4 }$ scale. Along a sequence of learned representations and designs, a sufficient generic condition for representation error to be negligible at the oracle root-n scale is

$$
\sqrt { n } \| B _ { m } e _ { m } \| \to 0 , \qquad \| e _ { m } \| = o ( n ^ { - 1 / 4 } ) ,
$$

together with the corresponding moment control when $e _ { m }$ is random. Errors lying exactly in the learned nuisance span are different because they are absorbed by recentering the nuisance coefficient and are not subject to this generic first-order remainder condition.

## E SYNTHETIC EXPERIMENTS

This appendix provides the constructions and diagnostics underlying the synthetic results in Section 6.1. Section E.1 specifies the common protocol, Section E.2 gives the complete numerical evidence behind Figure 2, and Section E.3 tests mechanism boundaries not needed in the main text.

## E.1 EXPERIMENTAL PROTOCOL

Common experimental setup. Unless an intervention explicitly changes one component, all controlled studies use independent Bernoulli observations from the canonical joint-logit model in equation 6, with the same target and nuisance representations, parameter region, and final estimator across acquisition rules. Oracle studies condition on a fixed nuisance representation. Representation-learning studies first generate independent historical human–judge data, freeze the learned representation and acquisition rule, and then generate independent current trusted and judge observations. Thus acquisition comparisons change the information collected rather than the outcome-generating law or final estimation procedure.

Estimator and numerical implementation. The main studies use the projected one-step estimator of Appendix B.2. Trusted logistic regression supplies the target preliminary, an offset nuisance fit supplies the nuisance preliminary, and one guarded joint update is followed by projection. Numerical search and safeguard settings are summarized in Table 2. Numerical damping is used only to stabilize optimization and is never counted as statistical information. Fixed-support theoretical criteria are evaluated using the realized integer counts after rounding. The distinct-comparison experiment instead uses the finite-pool procedure of Algorithm 1 and the certificate in equation 21.

Table 2: Common controlled-study protocol. Experiment-specific constructions and sample sizes are given in Sections E.2–E.3.
<table><tr><td>Component</td><td>Convention</td></tr><tr><td>Current outcomes</td><td>Independent Bernoulli trusted and judge labels</td></tr><tr><td>Final estimator</td><td>Trusted preliminary + offset nuisance preliminary + one joint update</td></tr><tr><td>Target / nuisance radii</td><td>2 and 3, respectively, unless stated otherwise</td></tr><tr><td>Preliminary gap tolerance</td><td> $1 0 ^ { - 6 }$  divided by fit sample size</td></tr><tr><td>Nuisance-Gram threshold</td><td> $1 0 ^ { - 1 0 }$ </td></tr><tr><td>Joint-information guard</td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>Numerical damping</td><td> $1 0 ^ { - 1 2 } I ,$  search only</td></tr><tr><td>Final likelihood</td><td>Unpenalized</td></tr><tr><td>Fixed-support design solver</td><td>SLSQP, analytic gradient, tolerance  $1 0 ^ { - 1 2 }$  , cap 600</td></tr><tr><td>Finite-pool procedure</td><td>Algorithm 1 framework, iteration cap 180</td></tr><tr><td>Primary endpoint</td><td>Scaled policy loss n  ${ \cal D } _ { F } ,$  unless stated otherwise</td></tr></table>

Replication units and uncertainty. For fixed-support studies, one independently generated Bernoulli dataset is one Monte Carlo replication, and MC SE is the sample standard deviation across replications divided by the square root of their number. In nested representation-learning studies, current losses are first averaged within each independent historical realization and uncertainty is computed across those historical realizations. In the distinct-pool study, current losses are first averaged within each independently generated pool and uncertainty is computed across pools. Error bars in the appendix figures are 1.96 MC SE. Inner current draws are therefore not treated as additional independent outer replications.

## E.2 CONTROLLED MECHANISM STUDIES

Effective information and policy risk. The experiment behind Figure 2(a) uses target features $X ~ = ~ ( 3 , 2 , - 1 )$ , constant nuisance feature $\dot { W } = 1$ , target center $\theta _ { 0 } \ = \ 0$ , nuisance center $a _ { 0 } ~ = ~ \log ( 3 / 2 )$ , and $n _ { c } ~ = ~ n$ trusted labels with feature one. The judge probability is 0.6 at every comparison type. The evaluation policy has feature gap two and temperature one, so $F ( \theta ) \stackrel { \cdot } { = } \log ( \stackrel { . } { 1 } + e ^ { 2 \theta } )$ up to an additive constant and $G _ { 0 } ~ = ~ 1$ . Every allocation has coordinate floor 0.05. NAOD, Target-info, and Random share the same correctly specified joint model and projected one-step estimator and differ only in acquisition. Each method–budget pair uses 6,000 independent datasets, and $\Phi _ { \kappa }$ is evaluated at the realized integer allocation.

Table 3: Effective-information experiment underlying Figure 2(a). Each row uses 6,000 independent repetitions. Φ is evaluated using the realized integer counts.
<table><tr><td>n</td><td>Design</td><td>Φ</td><td>Mean  $n D _ { F }$ </td><td>MC SE</td></tr><tr><td>240</td><td>NAOD</td><td>0.426</td><td>0.430</td><td>0.008</td></tr><tr><td>240</td><td>Target-info</td><td>1.139</td><td>1.164</td><td>0.021</td></tr><tr><td>240</td><td>Random</td><td>0.530</td><td>0.527</td><td>0.010</td></tr><tr><td>960</td><td>NAOD</td><td>0.426</td><td>0.441</td><td>0.008</td></tr><tr><td>960</td><td>Target-info</td><td>1.139</td><td>1.161</td><td>0.022</td></tr><tr><td>960</td><td>Random</td><td>0.530</td><td>0.518</td><td>0.010</td></tr><tr><td>3840</td><td>NAOD</td><td>0.426</td><td>0.426</td><td>0.008</td></tr><tr><td>3840</td><td>Target-info</td><td>1.139</td><td>1.105</td><td>0.020</td></tr><tr><td>3840</td><td>Random</td><td>0.530</td><td>0.526</td><td>0.009</td></tr><tr><td>15360</td><td>NAOD</td><td>0.426</td><td>0.431</td><td>0.008</td></tr><tr><td>15360</td><td>Target-info</td><td>1.139</td><td>1.144</td><td>0.021</td></tr><tr><td>15360</td><td>Random</td><td>0.530</td><td>0.541</td><td>0.010</td></tr></table>

Table 3 shows that empirical risks track the information constants across all four budgets. Because the likelihood and final estimator are identical across methods, the Target-info gap isolates acquisition geometry. Raw target-score variation is valuable only to the extent that it remains distinguishable from nuisance variation. This variance mechanism is separate from the exposure effect in equation 5. Under reference weights $( 0 . 1 , 0 . 2 , 0 . 7 )$ , the probability-residual score balances to zero, whereas uniform exposure gives $b _ { \xi } = 2 / 1 5 .$

Finite-sample representation-learning cost. The experiment behind Figure $2 ( \mathsf { b } )$ uses two equally weighted comparison types with $X = \breve { ( 1 , - 1 ) } ^ { \top }$ , true nuisance direction $\bar { W } _ { 0 } = ( 1 , 1 ) ^ { \top } , \theta _ { 0 } = \bar { 0 }$ , and $a _ { 0 } = c = \log ( 3 / 2 )$ . Human probabilities are 0.5 and judge probabilities are 0.6. The historical sample contains m independent units, each contributing one human and one judge label, with the units split equally across the two comparison types. Let $\widehat { h } _ { i }$ and $\widehat { j } _ { i }$ denote the corresponding sample means, let $s = 1 / 4 , t = 0 . 2 4$ , and set $g = 1 / 2$ . The historical learner first estimates the rotation

$$
\delta _ { m } ^ { \mathrm { r a w } } = \frac { g \{ ( \widehat { j } _ { 1 } - \widehat { h } _ { 1 } ) - ( \widehat { j } _ { 2 } - \widehat { h } _ { 2 } ) \} } { 2 c t } .
$$

We then set $\widehat { \delta } _ { m } = \mathrm { c l i p } ( \delta _ { m } ^ { \mathrm { r a w } } , - 0 . 5 , 0 . 5 )$ and $\widehat { W } _ { m } = ( 1 + \widehat { \delta } _ { m } , 1 - \widehat { \delta } _ { m } ) ^ { \top }$ . The current design is equally weighted with $n _ { c } ~ = ~ n .$ Propagating the historical rotation through the target-influence operator gives $\Phi _ { 0 } = 5 0 / 4 9$ and $C _ { P } = 2 5 / 9 8$ , so equation 16 predicts scaled risk $\Phi _ { 0 } \bar { + } ( n / m ) C _ { P }$ Each setting below uses $6 { , } 0 0 0$ independent historical–current replications.

Table 4: Representation-learning cost underlying Figure 2(b). Prediction is $\Phi _ { 0 } + ( n / m ) C _ { P }$ “Clip” is the percentage of historical fits for which the bounded rotation is active.
<table><tr><td>n</td><td> $m / n$ </td><td>Prediction</td><td>Mean n  $D _ { F }$ </td><td>MC SE</td><td>Clip (%)</td></tr><tr><td>960</td><td> $\overline { { 1 / 4 } }$ </td><td>2.041</td><td>1.891</td><td>0.034</td><td>3.1</td></tr><tr><td>960</td><td>1</td><td>1.276</td><td>1.252</td><td>0.023</td><td>0.0</td></tr><tr><td>960</td><td>4</td><td>1.084</td><td>1.085</td><td>0.020</td><td>0.0</td></tr><tr><td>3840</td><td> $1 / 4$ </td><td>2.041</td><td>1.993</td><td>0.037</td><td>0.0</td></tr><tr><td>3840</td><td>1</td><td>1.276</td><td>1.281</td><td>0.024</td><td>0.0</td></tr><tr><td>3840</td><td>4</td><td>1.084</td><td>1.116</td><td>0.020</td><td>0.0</td></tr><tr><td>15360</td><td> $1 / 4$ </td><td>2.041</td><td>2.046</td><td>0.037</td><td>0.0</td></tr><tr><td>15360</td><td>1</td><td>1.276</td><td>1.299</td><td>0.024</td><td>0.0</td></tr><tr><td>15360</td><td>4</td><td>1.084</td><td>1.091</td><td>0.020</td><td>0.0</td></tr></table>

At fixed $m / n ,$ , the predicted scaled risk is asymptotically constant rather than vanishing with the absolute current sample size. The empirical values in Table 4 approach these common levels as both samples grow, confirming that representation learning contributes at the same leading order as current-sample variance when $n / m$ does not vanish. The largest discrepancy occurs in the only setting with nonzero clipping, identifying a finite-sample departure from the local regime rather than a different asymptotic mechanism.

Design-ranking reversal. The experiment behind Figure $2 ( \mathrm { c } )$ returns to the three-type persistentresidual construction with $n = n _ { c } = 9 6 0$ . For each $m / n \in \{ 1 / 4 , 1 / 2 , 1 , 2 \}$ , we generate 300 independent balanced historical datasets, each containing m independent units with one human and one judge label per unit. At each comparison type, clipped human and judge sample means are transformed to logits and differenced. The resulting discrepancy vector is normalized to form the learned one-dimensional nuisance representation. A historical trusted logistic fit supplies the target center and an offset logistic fit supplies the nuisance coefficient. NAOD and Target-info use this same learned representation, current budgets, and final estimator.

Each historical realization generates eight current datasets per design. Losses are first averaged within the historical realization, and paired uncertainty is computed across the 300 outer averages. Table 5 reports the NAOD-minus-Target-info scaled-risk difference. Positive values favor Targetinfo.

Table 5: Design-ranking reversal underlying Figure 2(c). Differences are paired over 300 independent historical realizations, each averaging eight current datasets. Intervals are pointwise twosided $t _ { 2 9 9 }$ intervals.
<table><tr><td> $m / n$ </td><td>NAOD-Target-info</td><td>Outer SE</td><td>95% interval</td></tr><tr><td> $\overline { { 1 / 4 } }$ </td><td>+1.588</td><td>0.237</td><td>[1.122, 2.054]</td></tr><tr><td> $1 / 2$ </td><td>+0.772</td><td>0.200</td><td>[0.378, 1.165]</td></tr><tr><td>1</td><td>-0.039</td><td>0.100</td><td>[-0.235, 0.157]</td></tr><tr><td>2</td><td>-0.446</td><td>0.049</td><td>[−0.541, -0.350]</td></tr></table>

The sign change is the empirical counterpart of equation 17. Representation sensitivity dominates when historical information is scarce, the two leading contributions nearly balance around $m / n = 1$ and the oracle acquisition advantage dominates once representation error is sufficiently reduced. Because both designs use the same learned nuisance model and final estimator, the reversal is induced by how acquisition responds to representation error rather than by a difference in fitted model class.

## E.3 ADDITIONAL ABLATIONS AND DIAGNOSTICS

The direction of omitted error matters. The representation analysis in Appendix D.1 predicts that policy cost depends on how omitted logit error passes through the target-influence operator, not on its norm alone. At the zero-center NAOD allocation, let $u _ { 1 }$ be the unit direction aligned with the target-influence row. A near-null direction $u _ { 2 }$ is obtained by projecting the all-ones vector onto the orthogonal complement of $u _ { 1 }$ and rotating it by $\pi / 1 2$ toward $u _ { 1 } . \mathrm { A t } n = 3 8 4 0$ , both directions receive the same magnitudes $\rho \in \{ 0 , 1 , 2 , 3 , 4 , 6 , 8 \}$ , with true judge logits perturbed by $\rho u _ { j } / \sqrt { n }$ Each point uses 6,000 independent datasets.

Figure 3(a) shows that equal error magnitude can produce sharply different policy costs. The aligned perturbation follows the predicted quadratic increase, whereas the near-null perturbation remains much less consequential. Residual magnitude alone therefore cannot rank acquisition risk. The relevant quantity is its policy-weighted target projection through the target-influence operator and $G _ { 0 }$

Policy weighting changes which information is useful. To isolate $G _ { 0 } .$ , we use target rows $( \bar { 3 } , 0 ) \bar { , } ( 2 , 0 \bar { ) } , ( - \bar { 1 , } 0 ) , ( \bar { 0 , 3 } ) , ( 0 , 2 ) , ( 0 , - 1 )$ , constant nuisance feature $W = 1 , \theta _ { 0 } = 0 ,$ and $a _ { 0 } =$ log(3/2). Half of the trusted observations use each coordinate direction, giving $H _ { c } = I _ { 2 } / 8$ . The evaluation distribution weights two binary-action contexts by 0.8 and 0.2, each with feature gap two, so

$$
F ( \theta ) = 0 . 8 \log ( 1 + e ^ { 2 \theta _ { 1 } } ) + 0 . 2 \log ( 1 + e ^ { 2 \theta _ { 2 } } ) , \qquad G _ { 0 } = \mathrm { d i a g } ( 0 . 8 , 0 . 2 ) .
$$

The Unweighted ablation replaces this policy curvature by $I _ { 2 } / 2$ only during acquisition and is evaluated under the same true policy loss. The allocation floor is $0 . 0 3 , n _ { c } = n$ , and each setting uses 6,000 repetitions.

![](images/5cca580765e4c64efbc192216ca2373095f3574b5c70d709576740a3103ddb17.jpg)  
(a) Directional omitted-error.

![](images/997bf9697fe1fbd230af1a0172cb2b7bba0b9ae37863ad690bed5c8ef6eb6904.jpg)  
(b) Policy anisotropy.  
Figure 3: Controlled diagnostics. (a) Directional omitted-error diagnostic. Dashed curves are local predictions from the frozen logit-influence kernel. Solid curves are empirical mean scaled policy risks. The two perturbation directions have identical Euclidean magnitudes. (b) Policyanisotropy diagnostic. Dashed curves are information-based predictions. Solid curves are empirical mean scaled policy risks. The Unweighted rule changes only the acquisition weight and is evaluated under the same policy loss. Error bars show 1.96 MC SE over 6,000 independent datasets.

Figure 3(b) shows that NAOD remains below the Unweighted design across all budgets, even though both account for nuisance estimation. The gap therefore isolates policy weighting. $I _ { \mathrm { e f f } }$ determines which target directions remain identifiable after nuisance adjustment, while $\breve { G } _ { 0 }$ determines which of those directions matter for downstream policy loss. Target-info incurs the additional cost of ignoring target–nuisance coupling.

Nuisance rank separates misspecification from information loss. Using the same twodimensional construction at $n = 3 8 4 0$ , we vary only the fitted nuisance rank while keeping the actual judge probability at 0.6. Underfit uses no nuisance feature, Correct uses the constant feature, Overfit adds an unnecessary axis-group contrast, and Saturated uses $W = I _ { 6 }$ . In the saturated model, every judge target-score direction lies in the nuisance span.

Table 6: Nuisance-rank ablation at $n = 3 8 4 0 .$ . The Underfit nominal criterion is not a true-risk prediction because its fitted model omits the actual judge discrepancy.
<table><tr><td>Fitted model</td><td>r</td><td>Nominal Φ</td><td> $\overline { { { \mathrm { M e a n } n D _ { F } } } }$ </td><td>MC SE</td></tr><tr><td>Underfit</td><td>0</td><td>0.391</td><td>27.562</td><td>0.064</td></tr><tr><td>Correct</td><td>1</td><td>0.686</td><td>0.699</td><td>0.010</td></tr><tr><td>Overfit</td><td>2</td><td>0.772</td><td>0.764</td><td>0.011</td></tr><tr><td>Saturated</td><td>6</td><td>4.000</td><td>4.015</td><td>0.059</td></tr></table>

Table 6 separates three distinct effects. Underfitting creates omitted-discrepancy bias, so its favorable nominal criterion is misleading. Overfitting remains correctly specified but incurs a variance cost from an unnecessary nuisance direction. Saturation is also correctly specified, but the Schur complement removes all judge-derived target information because nuisance variation can reproduce every judge target-score direction. Nuisance dimension is therefore neither uniformly beneficial nor uniformly harmful. Its effect depends on representation adequacy and target–nuisance geometry.

Distinct finite-pool selection. We generate 24 independent pools, each containing $N ~ = ~ 6 0 0$ target-feature rows sampled uniformly from $[ - 2 , 2 ] ^ { 2 }$ . We set $W _ { i } = 1 , \theta _ { 0 } = 0 , a _ { 0 } = \log ( 3 / 2 )$ and use 300 trusted labels with balanced coordinate features and the same $0 . 8 / 0 . 2$ policy weighting as above. Judge budgets are $B \in \{ 6 0 , 1 8 0 , 3 6 0 \}$ . NAOD and Target-info use the finite-pool relaxation, rounding, and exchange procedure of Algorithm 1 with their respective frozen objectives and a common charged seed, while Random samples the remaining IDs without replacement. For each pool and budget, 300 independent outcome datasets are generated. Only selected outcomes enter the fit, and uncertainty is computed across the 24 pool-level mean losses.

Table 7 shows that the realized-subset criterion tracks empirical policy loss without requiring the rounded subset to approximate a fixed-support fractional allocation. Across the 24 pools, the largest

Table 7: Distinct finite-pool experiment. Risks are reported in $1 0 ^ { - 3 }$ units. Nominal risk uses the actual selected-subset information. SE is computed across 24 independent pool-level means, each averaging 300 current datasets.
<table><tr><td> $\overline { { B } }$ </td><td>Design</td><td>Nominal risk  $( \times 1 0 ^ { - 3 } )$ </td><td>Mean  $\overline { { D _ { F } } }$   $( \times 1 0 ^ { - 3 } )$ </td><td>Pool SE  $( \times 1 0 ^ { - 3 } )$ </td></tr><tr><td>60</td><td>NAOD</td><td>6.108</td><td> $\overline { { 6 . 1 4 6 } }$ </td><td>0.081</td></tr><tr><td>60</td><td>Target-info</td><td>6.145</td><td>6.181</td><td>0.081</td></tr><tr><td>60</td><td>Random</td><td>8.945</td><td>8.851</td><td>0.117</td></tr><tr><td>180</td><td>NAOD</td><td>3.420</td><td>3.361</td><td>0.053</td></tr><tr><td>180</td><td>Target-info</td><td>3.430</td><td>3.373</td><td>0.051</td></tr><tr><td>180</td><td>Random</td><td>5.287</td><td>5.191</td><td>0.082</td></tr><tr><td>360</td><td>NAOD</td><td>2.459</td><td>2.376</td><td>0.029</td></tr><tr><td>360</td><td>Target-info</td><td>2.461</td><td>2.385</td><td>0.030</td></tr><tr><td>360</td><td>Random</td><td>3.293</td><td>3.224</td><td>0.046</td></tr></table>

NAOD Frank–Wolfe gap is $4 . 0 \times 1 0 ^ { - 8 }$ and the largest integer certificate is $1 . 1 \times 1 0 ^ { - 6 }$ . The corresponding Target-info maxima are $5 . 7 \times 1 0 ^ { - 9 }$ and $1 . 2 \times \mathsf { \bar { 1 0 ^ { - 7 } } }$ . These values certify numerical optimization of each frozen criterion, not statistical uncertainty.

As a numerical check, we also compared the projected one-step estimator with a fully converged constrained joint likelihood on paired datasets. Their scaled-risk differences decreased with sample size and were negligible relative to the acquisition gaps.

## F CHATBOT ARENA EVALUATION

## F.1 DATA, REPRESENTATIONS, AND EXPERIMENTAL PROTOCOL

Frozen archive and cluster construction. We use the frozen revision of the Chatbot Arena LLMjudge archive. The raw archive contains 49,938 rows. After removing non-string or empty text and identical-response comparisons, 49,635 usable rows remain in 42,875 connected text clusters. Text is Unicode NFKC normalized and whitespace is collapsed. Connected components are formed before filtering by transitive exact-text matching on the prompt or either response. Cluster identity is the ownership unit for splitting and the capacity unit for acquisition.

Table 8: Arena cluster partition within each repeated split. Roles are cluster-disjoint. Candidate clusters may contain multiple rows, but at most one row from a connected cluster can be acquired.
<table><tr><td>Role</td><td>Clusters Use</td><td></td></tr><tr><td>Upstream</td><td> $\overline { { 8 , 5 7 5 } }$ </td><td>Target representation, nuisance representation, and hu- man reference</td></tr><tr><td>Historical initialization</td><td>128</td><td>Initial target and nuisance estimates</td></tr><tr><td>Policy support</td><td>1,024</td><td>Policy weighting covariates with no labels</td></tr><tr><td>Current human reservoir</td><td>512</td><td>Nested trusted budgets  $H \leq 5 1 2$ </td></tr><tr><td>Candidate</td><td>24,061</td><td>Judge acquisition with cluster capacity one</td></tr><tr><td>Test</td><td>8,575</td><td>Held-out human evaluation</td></tr></table>

Table 8 summarizes these role assignments. Upstream, historical initialization, policy support, and the human reservoir use one representative row from each assigned cluster. Candidate retains all eligible rows subject to cluster capacity one. Test performance is first averaged within connected clusters. We use 15 prespecified repeated cluster-level splits. Each split relearns the downstream representations and human reference, while all 15 budget combinations within a split share these upstream objects.

Target and nuisance representations. Response features are extracted with Qwen3-Embedding-0.6B. The pair representation is the difference between the A and B response embeddings. An upstream-fitted whitened PCA reduces this representation to 128 coordinates.

For each judge, we fit four ridge heads separately to human preferences and to that judge’s hard preferences. The leading two-dimensional singular-vector span of the resulting coefficient vectors defines the target representation $X _ { J }$ . Policy-weighted residual learning is performed separately. The resulting standardized swap-antisymmetric residual score $r _ { J } ( z )$ defines the nuisance representation $W _ { J } ( z ) \ = \ ( 1 , r _ { J } ( z ) ) ^ { \top }$ Thus the target and nuisance representations are learned from upstream data and frozen before current acquisition. Table 9 records the frozen numerical configuration used across methods.

Table 9: Frozen Arena configuration. All acquisition methods use the same representation, initialization, current budgets, and final estimator unless required otherwise by the method definition.
<table><tr><td>Component</td><td>Configuration</td></tr><tr><td>Embedding</td><td>Qwen3-Embedding-0.6B, last-token pooling,  $L _ { 2 }$  normalization</td></tr><tr><td>Maximum input</td><td>2,048 tokens with prompt cap 512</td></tr><tr><td>Base reduction</td><td>Upstream-fitted whitened PCA, 128 dimensions</td></tr><tr><td>Target representation</td><td>Judge-specific SVD span, dimension  $d = 2$ </td></tr><tr><td>Target-head penalties</td><td>32, 128, 512, 2048</td></tr><tr><td>Nuisance representation</td><td>Intercept and standardized learned residual, dimension  $r = 2$ </td></tr><tr><td>Residual learner</td><td>ExtraTrees, 128 trees, depth 10, min leaf 20, feature fraction 0.5, weight clip [0.1, 10]</td></tr><tr><td>Inner residual folds</td><td>4</td></tr><tr><td>Human budgets</td><td> $H \in \{ 3 2 , 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \}$ </td></tr><tr><td>Judge budgets</td><td> $B \in \{ 2 5 6 , 5 1 2 , 1 0 2 4 \}$ </td></tr><tr><td>Parameter boxes</td><td> $\| \theta \| _ { \infty } \leq 2 0 , \| a \| _ { \infty } \leq 1 0$ </td></tr><tr><td>Final step size</td><td> $\alpha = 1$ </td></tr><tr><td>Finite-pool solver</td><td>Frank-Wolfe cap 180, tolerance  $1 0 ^ { - 5 }$ </td></tr><tr><td>Cluster capacity</td><td>At most one selected comparison per connected cluster</td></tr></table>

Feedback and common estimator. Human outcomes are encoded as 1 for A, 0 for B, and $1 / 2$ for ties. For judge feedback, the archived A/B/tie probabilities are converted to the soft target $y _ { J } =$ $p _ { A } + \frac { 1 } { 2 } p _ { T }$ . The logistic quasi-likelihood therefore uses the archived graded preference signal directly.

A common historical set of 128 comparisons supplies the initial target and nuisance estimates before current acquisition. After current trusted feedback and the selected judge feedback are revealed, every acquisition method uses the same unpenalized joint observed-information update and the same fixed parameter boxes. Historical initialization rows are not reused in the final score. NAOD and Target-info use the same charged feasibility anchors. Their comparison therefore changes the acquisition criterion while holding the representation, budgets, and final estimator fixed.

Resource accounting. The varying current resource is $( H , B )$ . The 8,575 upstream human–judge comparisons used for representation learning and the 128 historical comparisons used for initialization are shared across acquisition rules and are not counted in H or B. Policy support contributes covariates without labels, while test human outcomes are used only for evaluation.

## F.2 METRICS, BASELINES, AND STATISTICAL ANALYSIS

Acquisition baselines. Random samples feasible candidate comparisons uniformly. Entropy prioritizes candidates with high predictive uncertainty. D-opt and PA D-opt use Fisher-information acquisition on the common judge-specific target representation, with PA D-opt additionally incorporating the historical initialization as past information. Target-info minimizes the policy-weighted target-information criterion without nuisance adjustment. NAOD instead uses effective target information after nuisance adjustment. Initial performs no current update. Human-only updates from the common initialization using current trusted feedback without acquired judge outcomes.

Evaluation endpoints. The primary endpoint is proxy policy regret ${ \cal D } _ { F } ( \theta _ { \mathrm { r e f } } , \widehat { \theta } )$ , where $\theta _ { \mathrm { r e f } }$ is fitted only from upstream human feedback. We additionally report held-out human cross-entropy and tie-aware human choice accuracy. Cross-entropy uses the full predicted probability. For hard

choice accuracy, we predict A when the fitted preference probability is at least 0.5 and B otherwise.   
Human ties receive half credit.

Aggregation and paired uncertainty. Test losses are first averaged within connected clusters and then equally across clusters. Judges are macro-averaged within each split. For overall results, the 15 budget combinations are also equally weighted. The repeated split is the inference unit. For loss endpoints, paired gain is baseline minus NAOD. For accuracy, paired gain is NAOD minus baseline. Positive values therefore favor NAOD throughout. Two-sided 95% Student t intervals are computed from the 15 paired split-level differences using $t _ { 1 4 } .$ Budget and judge intervals are pointwise intervals. Relative proxy-regret reduction is $1 0 0 ( \hat { 1 } - \overline { { R } } _ { \mathrm { N A O D } } / \overline { { R } } _ { b } )$ , which is a ratio of aggregated mean regrets rather than an average of split-specific percentage reductions.

## F.3 COMPLETE RESULTS AND DIAGNOSTICS

Paired comparisons across methods. The main text reports absolute endpoint means. Table 10 reports the corresponding paired split-level effects and their uncertainty.

Table 10: Overall paired Arena comparisons. Positive values favor NAOD. Proxy-regret and human-CE gains are in $1 0 ^ { - 3 }$ units. Accuracy gains are percentage points. Brackets give paired 95% $t _ { 1 4 }$ intervals across the 15 repeated splits.
<table><tr><td>Baseline</td><td>Proxy-regret gain</td><td>Human-CE gain</td><td>Accuracy gain</td></tr><tr><td>Initial</td><td>+6.870 [3.020, 10.721]</td><td>+5.308 [1.952, 8.664]</td><td>+0.448 [0.117, 0.780]</td></tr><tr><td>Human-only</td><td>+16.259 [8.849, 23.668]</td><td>+12.981 [7.214, 18.748]</td><td>+1.426 [0.331, 2.520]</td></tr><tr><td>Random</td><td>+2.886 [2.455, 3.318]</td><td>+2.814 [2.388, 3.241]</td><td>+0.050 [−0.069, 0.169]</td></tr><tr><td>Entropy</td><td>+11.433 [4.783, 18.083]</td><td>+8.527 [3.669, 13.385]</td><td>+1.166 [0.079, 2.252]</td></tr><tr><td>D-opt</td><td>+0.463 [0.093, 0.833]</td><td>+0.385 [0.011, 0.759]</td><td>+0.075 [0.028, 0.121]</td></tr><tr><td>PA D-opt</td><td>+0.465 [0.095, 0.834]</td><td>+0.386 [0.013, 0.760]</td><td>+0.075 [0.029, 0.121]</td></tr><tr><td>Target-info</td><td>+0.818 [0.348, 1.287]</td><td>+0.531 [0.211, 0.850]</td><td>+0.058 [0.015, 0.101]</td></tr></table>

Relative to Target-info, the proxy-regret gain is accompanied by lower held-out human cross-entropy and higher held-out choice accuracy. The D-opt and PA D-opt differences are smaller but remain positive in the overall paired analysis. The smaller accuracy gain is expected because hard choice accuracy changes only when the predicted preference crosses the 0.5 threshold, whereas cross-entropy remains sensitive to probability improvements that preserve the predicted winner. The resulting accuracy gain is positive in 12 of 15 repeated splits.

Budget dependence. Figure 4 reports relative proxy-regret reductions over the complete $5 \times 3$ budget grid. Table 11 reports the paired absolute gains and uncertainty for Target-info and D-opt.  
![](images/820d5c8e05c94fed7b5c4b130f166a3f62dde216b05bb0e321651e01b7528152.jpg)  
(a) Relative to Target-info.

![](images/6569b212438c958aebdb0cb17f038ba5ee6f9e570b5098c943b9951d29db6f09.jpg)  
(b) Relative to D-opt.  
Figure 4: Budget dependence on Chatbot Arena. NAOD relative reduction in mean proxy policy regret across the complete 5×3 Arena budget grid. Positive values favor NAOD. Panel (a) compares against Target-info and panel (b) against D-opt. Cell values are ratios of macro-averaged mean regrets rather than averages of split-specific percentage reductions.

Table 11: Paired proxy-regret gains across the Arena budget grid. Entries are baseline-minus-NAOD gains in $1 0 ^ { - 3 }$ units with pointwise paired 95% $t _ { 1 4 }$ intervals.
<table><tr><td> $\overline { { H } }$ </td><td> $\overline { { B } }$ </td><td>Target-info-NAOD</td><td>D-opt-NAOD</td></tr><tr><td>32</td><td>256</td><td>+2.103 [0.263, 3.944]</td><td>+1.543 [0.612, 2.474]</td></tr><tr><td>32</td><td>512</td><td>+1.141 [0.447, 1.835]</td><td>+0.715 [0.138, 1.291]</td></tr><tr><td>32</td><td>1024</td><td>+0.408 [0.143, 0.673]</td><td>+0.160 [−0.192, 0.512]</td></tr><tr><td>64</td><td>256</td><td>+1.731 [0.507, 2.955]</td><td>+1.234 [0.359, 2.109]</td></tr><tr><td>64</td><td>512</td><td>+1.123 [0.476, 1.770]</td><td>+0.703 [0.125, 1.281]</td></tr><tr><td>64</td><td>1024</td><td>+0.471 [0.184, 0.758]</td><td>+0.234 [-0.119, 0.587]</td></tr><tr><td>128</td><td>256</td><td>+1.172 [0.205, 2.138]</td><td>+0.814 [0.062, 1.565]</td></tr><tr><td>128</td><td>512</td><td>+0.843 [0.071, 1.615]</td><td>+0.434 [−0.117, 0.986]</td></tr><tr><td>128</td><td>1024</td><td>+0.367 [-0.044, 0.778]</td><td>+0.092 [−0.303, 0.487]</td></tr><tr><td>256</td><td>256</td><td>+0.844 [0.211, 1.477]</td><td>+0.466 [0.008, 0.924]</td></tr><tr><td>256</td><td>512</td><td>+0.553 [0.221, 0.885]</td><td>+0.186 [-0.222, 0.594]</td></tr><tr><td>256</td><td>1024</td><td>+0.176 [−0.044, 0.396]</td><td>-0.084 [−0.405, 0.238]</td></tr><tr><td>512</td><td>256</td><td>+0.664 [0.203, 1.124]</td><td>+0.395 [0.001, 0.790]</td></tr><tr><td>512</td><td>512</td><td>+0.515 [0.199, 0.831]</td><td>+0.157[-0.186, 0.500]</td></tr><tr><td>512</td><td>1024</td><td>+0.155 [-0.044, 0.354]</td><td>-0.103 [-0.369, 0.162]</td></tr></table>

The Target-info mean comparison favors NAOD in all 15 budget cells, with 12 pointwise paired intervals entirely above zero. At every fixed human budget, the relative gain narrows as the judge budget increases. The pattern is not determined by $H / \breve { B }$ alone. Along ${ \bf \bar { \cal H } } / B = 1 / 8 { \it \Delta \mathrm { . } }$ , the relative reductions are 50.1%, 36.7%, and 15.3% for (H, B) = (32, 256), (64, 512), (128, 1024). Selection overlap increases from 0.446 to 0.505 to 0.568 over the same sequence. Absolute information scale and finite-pool geometry therefore matter in addition to the human-to-judge budget ratio.

The D-opt gap narrows more sharply at large judge budgets. At (256, 1024) and (512, 1024), the mean difference changes sign, while both paired intervals include zero. The separation from D-opt is therefore budget dependent rather than uniform across the grid.

Judge-level target–nuisance coupling. Figure 5 summarizes the judge-level association between Target-info coupling and realized policy gain, while Table 12 reports the corresponding coupling values and paired policy gains with pointwise 95% intervals.

![](images/fc2558f9d34a083867c88a721efd9e334f738179714d897eee4a755af911ccc1.jpg)  
Figure 5: Judge-level coupling and realized acquisition gain. The horizontal axis is Target-info target–nuisance coupling $\rho ^ { \bar { 2 } }$ and the vertical axis is the Target-info-minus-NAOD proxy-regret gain in $\mathbf { \bar { 1 0 } ^ { - 3 } }$ units. Each point represents one judge after averaging over the 15 budget combinations and 15 repeated cluster-level splits. The Spearman rank correlation is $r _ { s } = 0 . 5 7 6$ . Positive vertical values favor NAOD.

Mean policy gain is positive for 15 of the 17 judges. The largest gains occur for Meta-Llama-3- 8B and the two dolphin judges, whose Target-info coupling values all exceed 0.95. By contrast, Meta-Llama-3-70B-Instruct has strong residual predictability but substantially weaker coupling and essentially no policy gain. This contrast separates residual learnability from target–nuisance interference. The Spearman association in Figure 5 summarizes this judge-level relationship.

Table 12: Judge-level coupling and policy gain. $\rho _ { T } ^ { 2 }$ is the squared target–nuisance coupling under Target-info. Policy gain is Target-info minus NAOD in $1 0 ^ { - 3 }$ units with pointwise paired 95% $t _ { 1 4 }$ intervals.
<table><tr><td>Judge</td><td> $\overline { { \rho _ { T } ^ { 2 } } }$ </td><td>Policy gain</td></tr><tr><td>Athene-70B</td><td>0.570</td><td>+0.428 [-0.290, 1.145]</td></tr><tr><td>dolphin-2.1-mistral-7b</td><td>0.956</td><td>+2.063 [1.076, 3.049]</td></tr><tr><td>dolphin-2.5-mixtral-8x7b</td><td>0.960</td><td>+2.819 [1.248, 4.390]</td></tr><tr><td>Hermes-3-Llama-3.1-70B</td><td>0.610</td><td>+0.366 [−0.305, 1.038]</td></tr><tr><td>Meta-Llama-3-70B-Instruct</td><td>0.459</td><td>-0.037 [-0.358, 0.285]</td></tr><tr><td>Meta-Llama-3-8B-Instruct</td><td>0.716</td><td>+0.733 [0.069, 1.397]</td></tr><tr><td>Meta-Llama-3-8B</td><td>0.958</td><td>+4.024 [1.184, 6.865]</td></tr><tr><td>Mistral-7B-Instruct-v0.1</td><td>0.713</td><td>+0.096 [-0.035, 0.227]</td></tr><tr><td>Mistral-7B-Instruct-v0.2</td><td>0.666</td><td>+0.491 [0.129, 0.854]</td></tr><tr><td>Mistral-7B-OpenOrca</td><td>0.646</td><td>+0.067 [-0.151, 0.285]</td></tr><tr><td>Mixtral-8x7B-Instruct-v0.1</td><td>0.604</td><td>+0.749 [0.332, 1.166]</td></tr><tr><td>OpenHermes-2-Mistral-7B</td><td>0.676</td><td>+0.194 [−0.141, 0.529]</td></tr><tr><td>OpenHermes-2.5-Mistral-7B</td><td>0.611</td><td>+0.531 [0.153, 0.909]</td></tr><tr><td>Qwen2-72B-Instruct</td><td>0.551</td><td>+0.211 [-0.252, 0.673]</td></tr><tr><td>StableBeluga-7B</td><td>0.915</td><td>+0.873 [0.234, 1.511]</td></tr><tr><td>Starling-LM-7B-alpha</td><td>0.646</td><td>+0.391 [0.045, 0.736]</td></tr><tr><td>zephyr-7b-beta</td><td>0.647</td><td>-0.099 [−0.283, 0.084]</td></tr></table>

Across all budget and split designs, mean squared coupling decreases from 0.700 under Target-info to 0.290 under NAOD. The mean selected-set overlap is 0.521. NAOD therefore changes which comparisons are acquired in a direction that reduces target–nuisance coupling rather than applying a different estimator to essentially the same selected set.

Representation and numerical diagnostics. All $2 5 5 \ : = \ : 1 5 \times 1 7$ split-by-judge nuisance fits achieve positive out-of-fold soft-label cross-entropy gain, with mean gain 0.0370. The learned nuisance representation therefore captures systematic judge residual structure across the full evaluation.

The audit finds no budget, duplicate-ID, duplicate-cluster, out-of-pool, projection, or design failures in 22,950 acquisition arrays. Across 7,650 NAOD and Target-info designs, the maximum Frank– Wolfe gap is $\mathbf { \bar { 8 . 6 2 } \times 1 0 ^ { - 8 } }$ and the maximum integer certificate is $6 . 0 1 \times 1 0 ^ { - 7 }$ , indicating that the observed acquisition differences are not explained by unresolved selection optimization.