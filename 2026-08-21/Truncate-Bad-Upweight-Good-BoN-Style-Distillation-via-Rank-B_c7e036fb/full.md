# Truncate Bad, Upweight Good: BoN-Style Distillation via Rank-Based Classification

Yarin Bar<sup>1</sup> Yaniv Romano<sup>1,2</sup>

<sup>1</sup>Department of Electrical and Computer Engineering

<sup>2</sup>Department of Computer Science

Technion–Israel Institute of Technology

yarinbar@campus.technion.ac.il yromano@technion.ac.il

## Abstract

Inference-time selection methods, such as Best-of-N, improve generation by sampling a pool of candidates and selecting the top-ranked completion according to a reward model. Distillation seeks to amortize this procedure into a single policy by replacing raw rewards with in-pool ranks and learning a policy that upweights higher-ranked completions. However, existing rank-based policies typically use smooth full-support reweighting, so low-ranked completions receive less mass but remain in the target support. Although a sharper reweighting reduces lower-tail mass, it also increases reliance on brittle ranking at the top made by a single reward model. We propose TUP: a Truncate-bad, Upweight-good Policy that removes low-ranked completions from the support and reweights only the retained upper tail with a tunable sharpness. TUP admits a closed-form, prompt-independent normalization and can be trained fully ofline via binary cross-entropy, using shifted-truncated win-rates as soft labels and distilled-to-reference log-likelihood ratios as logits. Theoretically, under certain assumptions, we show that for any unknown oracle reward, the best monotone rank-reweighting can be matched by a lower-tail truncation rule, providing formal support for removing the lower tail rather than merely downweighting it. Empirically, we show that TUP is competitive with strong ofline alignment baselines.

## 1 Introduction

Inference-time selection methods such as best-of-N (BoN) have emerged as an efective tool for improving language models’ generation quality [4, 15]. BoN samples several completions from a base policy, scores them using a reward model, and returns the highest-scoring one [1, 17, 36]. The price is computational, since each prompt requires multiple generated completions and reward evaluations [5]. This has motivated a growing line of work on BoN-style distillation [3, 4, 8, 14, 26, 34], where rather than carrying out selection during deployment, one trains a single policy to internalize the behavior of the inference-time selector.

A key insight from recent BoN-style distillation work is that the relevant training signal for a completion is not its raw reward in isolation, but its relative rank within the sampled pool. This relative rank is often referred to as the completion win-rate. In turn, existing distillation methods aim to maximize the win-rate while penalizing the KL distance from the base policy [3, 14, 26]. This results in policies that apply smooth full-support reweighting, favoring higher-ranked completions while keeping every completion in the target support. Sharper reweighting reduces the probability mass of clearly low-ranked completions, but it also concentrates more mass on the very top of the reward-model ranking. As a result, over-sharpening can move the policy further from the reference model (smaller KL-penalty), increasing the risk of reward hacking [19, 28, 32, 38]. Since reward models are imperfect proxies for unknown oracle or human preferences [10, 11, 15, 20, 21, 29, 32, 45], recent work shows that coarse preferences tend to persist across diferent evaluators, while fine-grained distinctions among top-ranked completions are less consistently agreed upon [23, 24, 43].

Consequently, we argue that the design of a BoN-style distillation policy should separate two choices: (i) how much of the lower-ranked tail should be removed from the support, and (ii) how sharply the retained higherranked completions should be upweighted. To this end, we propose TUP, a Truncate-bad, Upweight-good Policy for BoN-style distillation. TUP decouples these choices with a shifted-truncated win-rate transform. Completions whose win-rate falls below the threshold are assigned zero mass, while those above it are softly reweighted by their relative ranking. The threshold parameter controls lower-tail truncation, whereas the upper-tail sharpness parameter controls the upweighting strength within the retained set. By introducing separate parameters for lower-tail truncation and upper-tail sharpness, this design preserves the BoN principle of favoring higher-ranked completions while (i) preventing probability mass from being assigned to clearly undesirable completions, and (ii) avoiding over-sharpening due to possible ranking mistakes at the top.

![](images/c40a4b819b4da481a47c361f7268f471ee01303f7f06d4e7b6109fd75b32b801.jpg)  
Figure 1: Illustration of TUP.

The resulting target policy remains simple to train from ofline data. Each prompt is paired with a pool of completions sampled from the reference policy and scored by a reward model. From these scores, we compute empirical in-pool ranks and convert them into shifted-truncated win-rate labels; by subtracting the truncation parameter, we assign the lower-tail completions an exact zero label. Pairing with the logit transformation, we then train the distilled language model with binary cross-entropy (BCE) loss.

Contributions. In this work, we make three key contributions.

1. A truncate-bad, upweight-good policy for BoN-style distillation. We propose a rank-based target policy that removes low-ranked completions using a win-rate threshold and softly reweights the retained upper tail.

2. Theoretical support for lower-tail truncation and within-tail upweighting. Under certain assumptions, we give two theoretical justifications for TUP, using the oracle win-rate as the performance criterion. First, we show that the best hard lower-tail truncation rule can match the best monotone policy that reweights completions by proxy-reward win-rate. Second, after fixing a truncation threshold, we show that smooth upweighting of retained completions can further improve the oracle win-rate when, informally, higher proxy-reward win-rates tend to correspond to higher oracle win-rates within the retained tail.

3. Benchmark results and ablations. We evaluate TUP on the QRPO benchmark training Llama-8B Tülu 3 SFT on UltraFeedbak and Magpie Air and Mistral-7B-Instruct-v0.2 on Magpie Air, comparing to four leading ofline-alignment baseline methods. We evaluate performance using three diferent reward models, both on held-out data and AlpacaEval [9]. Full implementation of TUP and the experiments are available at github.com/yarinbar/truncate-bad-upweight-good.

## 2 Preliminaries and related works

We denote a language model as a policy $\pi ( y \mid x )$ mapping a prompt $x \in \mathcal { X }$ to a distribution over completions $y \in \mathcal { V }$ . Prompts are drawn from a distribution D over X. Let $\pi _ { \mathrm { r e f } } ( y \mid x )$ denote the reference policy, e.g., obtained via supervised fine-tuning, and let $r ( x , y ) \in \mathbb { R }$ be a reward function.

The standard post-training alignment objective seeks a policy that maximizes expected reward while remaining close to the reference policy:

$$
\pi _ { r , \beta } ^ { \star } ( \cdot | x ) = \arg \operatorname* { m a x } _ { \pi } \mathbb { E } _ { y \sim \pi ( \cdot | x ) } [ r ( x , y ) ] - \beta D _ { \mathrm { K L } } ( \pi ( \cdot | x ) \| \pi _ { \mathrm { r e f } } ( \cdot | x ) ) ,\tag{1}
$$

where $\beta > 0$ controls the reward–KL tradeof. This KL-regularized reward-maximization objective is a standard formulation in RLHF for language-model alignment [2, 6, 30, 37, 46], and is commonly optimized with online reinforcement-learning methods such as PPO [33] and GRPO [35].

An alternative paradigm, which we follow in this paper, builds directly on the closed-form solution to (1), known as the Gibbs policy:

$$
\pi _ { r , \beta } ^ { \star } ( y | x ) = \frac { 1 } { Z ( x ) } \pi _ { \mathrm { r e f } } ( y | x ) \exp \left( \frac { r ( x , y ) } { \beta } \right) , \quad Z ( x ) = \mathbb { E } _ { Y \sim \pi _ { \mathrm { r e f } } ( \cdot | x ) } \left[ \exp \left( \frac { r ( x , Y ) } { \beta } \right) \right] ,\tag{2}
$$

where $Z ( x )$ is the partition function. Rearranging (2) yields

$$
r ( x , y ) = \beta \log Z ( x ) + \beta \log \frac { \pi _ { r , \beta } ^ { \star } ( y | x ) } { \pi _ { \mathrm { r e f } } ( y | x ) } .\tag{3}
$$

This relation suggests a direct fitting strategy. Replacing the unknown optimal policy $\pi _ { r , \beta } ^ { \star }$ with a parameterized model $\pi _ { \theta }$ would allow one to train the policy by regressing the right-hand side of $\left( 3 \right) ^ { ^ { \prime } }$ to predict the reward. The dificulty is that the partition function $Z ( x )$ is prompt-dependent and impractical to compute. Preferenceoptimization methods avoid the need to compute $Z ( x )$ in diferent but related ways. DPO [31], IPO [13], and SimPO [27] learn from pairwise or binary preference signals, while REBEL [12] regresses relative reward diferences between completions.

Although standard post-training alignment typically treats the raw reward $r ( x , y )$ as the quantity to optimize, this is not the most natural choice for inference-time selection algorithms such as BoN. Recent inference-aware and rank-based methods, including InfAlign [3] and QRPO [26], motivate replacing raw reward values with relative-rank quantities. A natural choice is the population win-rate

$$
w _ { r } ( x , y ) : = \mathbb { P } _ { Y ^ { \prime } \sim \pi _ { \mathrm { r e f } } ( \cdot | x ) } \big ( r ( x , Y ^ { \prime } ) \leq r ( x , y ) \big ) ,\tag{4}
$$

which measures how often y beats an independent reference sample. In practice, one can estimate the win-rate from a finite pool of completions.

This win-rate parametrization is useful not only because it matches the relative nature of BoN-style selection, but also because it removes the prompt-dependent partition function in (3). When $Y \sim \pi _ { \mathrm { r e f } } ( \cdot \mid x )$ and the conditional reward distribution is continuous, the probability integral transform implies that $w _ { r } ( x , Y ) \sim U [ 0 , 1 ]$ Therefore, for any nonnegative reweighting function $g : [ 0 , 1 ]  [ 0 , \infty )$ with finite integral, the rank-based policy

$$
\pi _ { g } ( y \mid x ) = { \frac { 1 } { Z _ { g } } } g ( w _ { r } ( x , y ) ) \pi _ { \mathrm { r e f } } ( y \mid x ) , \qquad Z _ { g } = \int _ { 0 } ^ { 1 } g ( u ) d u ,\tag{5}
$$

has a normalizer $Z _ { g }$ independent of the prompt x. Many rank-based policies follow this construction. For example, classic Best-of-N induces $g _ { \mathrm { B o N } , N } ( w ) \stackrel { \textstyle \sim } { \propto } w ^ { N - 1 } \left[ 4 , 1 4 \right]$ , and the standard QRPO formulation induces $g _ { \mathrm { Q R P O } , \beta } ( w ) \propto e ^ { w / \beta } \ [ 2 6 ]$ . Both assign positive weight over the entire support, leaving open how the win-rate should be transformed into a target-policy reweighting.

This full-support property clarifies how our target difers from related alignment methods. Iterative BoNdistillation methods such as BOND [34] and Faster-WIND [41] also aim to approximate or distill Best-of-N-type behavior, but they inherit the smooth reweighting view rather than defining a hard lower-tail removal rule.

![](images/447eb4d0d466ca1a3ced640d1f22d8ff59ab2587d6d4d8f70d0c902ba77f1774.jpg)

![](images/1514f2f08b6208ca2a4200c5ef1353692e80513a943253f51c6dae1897a0d672.jpg)  
Figure 2: Reward model agreement (left) and the cost–benefit tradeof of λ-truncation (right) on UltraFeedback reference completions. We compare ArmoRM, used as the proxy reward model, against the independent Skywork-v2-Llama reward model. Left: Fraction of completions placed by ArmoRM in the top or bottom p region that Skywork also places in the same region, as a function of $p .$ Right: Probability that λ-truncation discards a Skywork-top-25% completion or retains a Skywork-bottom-25% completion. The crossover marks where these error probabilities are equal.

Other objectives, including RAFT [8], SLiC-HF [44], and RRHF [42], use reward filtering or relative preference rankings to construct training signals, but they do not specify a prompt-independent rank-transform density that first truncates the lower tail and then softly reweights the retained upper tail. Our work makes this truncation-and-reweighting operation explicit, yielding a finite prompt-independent normalization and a target that can be fit as a soft-classification problem using the transformed win-rate as labels.

## 3 Method

## 3.1 The proposed truncate-bad, upweight-good target policy

Gibbs-like policies (5) that reweight $\pi _ { \mathrm { r e f } }$ by a smooth monotone function of the proxy win-rate, such as QRPO [26], face the sharpness tradeof we discussed above.

Figure 2 (left) provides an empirical illustration motivating our choice to decouple lower-tail truncation from upper-tail sharpness. When the same reference completions are ranked by ArmoRM [39] and by an independent Skywork-v2-Llama reward [25], the models tend to agree more on the bottom of the pool than on the top. Thus, removing the lower tail targets a comparatively stable region across reward models, whereas sharpening toward the top places disproportionate trust in fine-grained top-rank distinctions with weaker cross-model agreement. Additional such comparisons are available in Appendix B.

We instantiate this principle by introducing a truncation level $\lambda \in \mathsf { \Gamma } ( 0 , 1 )$ , which determines whether a completion remains in the target support, and a sharpness parameter $\beta > 0$ , which controls how strongly the retained completions are reweighted. Given λ, we define the shifted-truncated win-rate

$$
w _ { \lambda , r } ( x , y ) : = \operatorname* { m a x } ( w _ { r } ( x , y ) - \lambda , 0 ) ,\tag{6}
$$

and use it to define the transformed reward

$$
R _ { \lambda } ( x , y ) : = \mathrm { l o g i t } \bigl ( w _ { \lambda , r } ( x , y ) \bigr ) = \mathrm { l o g } \frac { w _ { \lambda , r } ( x , y ) } { 1 - w _ { \lambda , r } ( x , y ) } .\tag{7}
$$

The truncation level λ acts as a quality threshold. Completions with $w _ { r } ( x , y ) \leq \lambda$ are mapped to $R _ { \lambda } ( x , y ) =$ $- \infty .$ , while completions above the threshold are retained and scored by the log-odds of their shifted win-rate. Thus, λ imposes a hard quality floor, whereas the transformed reward preserves a smooth ordering among the retained completions. As shown in Figure 2 (right), increasing λ lowers the probability of retaining completions that the independent reward model rank poorly, but also raises the probability of discarding completions this model rank highly.

![](images/246a7200c014988663df4af417740d01e444ea192d5ee746b1b40d3faf87d2d5.jpg)  
Figure 3: Fixed-threshold illustration for the quadratic un-normalized oracle-proxy profile $m _ { s } ( w ) = 1 - ( w - s ) ^ { 2 }$ at $\lambda = 1 / 2$ . The x axis is the proxy-rank location s where oracle utility is maximized, and the y axis shows the unnormalized oracle utility $\mathcal { U } _ { s } .$ . The curves compare the best QRPO policy, pure truncation, and the best TUP reweighting at the fixed threshold $\begin{array} { r } { \operatorname* { s u p } _ { \beta \in ( 0 , \infty ] } \mathcal { U } _ { s } ( 1 / 2 , \beta ) } \end{array}$ . The vertical dashed line marks $s _ { \beta } .$ , above which finite $\beta$ improves over pure truncation, and the dotted line marks $s _ { T }$ , where pure truncation no longer outperforms the best QRPO curve.

We now derive the KL-regularized target policy induced by the transformed reward $R _ { \lambda } ( x , y )$ . The key point is that truncation preserves the tractability of win-rate-based objectives. When the reward distribution is continuous under $Y \sim \pi _ { \mathrm { r e f } } ( \cdot \mid x )$ , the population win-rate $w _ { r } ( x , Y )$ is uniform on [0, 1]. After shifting and truncating this variable, the Gibbs normalizer remains independent of the prompt and can be computed in closed form.

Proposition 3.1 (Closed-form target under the truncated reward). Assume that, for the fixed prompt x, the distribution $o f r ( x , Y )$ under $Y \sim \pi _ { \mathrm { r e f } } ( \cdot \mid x )$ is continuous. Then, for every truncation level $\lambda \in ( 0 , 1 )$ and regularization parameter $\beta > 0$ , the KL-regularized objective in (1) with reward $R _ { \lambda }$ admits the unique Gibbs solution

$$
\pi _ { \lambda , \beta } ^ { \star } ( \boldsymbol { y } \mid \boldsymbol { x } ) = \frac { 1 } { Z _ { \lambda , \beta } } \pi _ { \mathrm { r e f } } ( \boldsymbol { y } \mid \boldsymbol { x } ) \left( \frac { w _ { \lambda , r } ( \boldsymbol { x } , \boldsymbol { y } ) } { 1 - w _ { \lambda , r } ( \boldsymbol { x } , \boldsymbol { y } ) } \right) ^ { 1 / \beta } ,\tag{8}
$$

where the normalization constant is

$$
Z _ { \lambda , \beta } \ = \ \int _ { 0 } ^ { 1 - \lambda } t ^ { 1 / \beta } ( 1 - t ) ^ { - 1 / \beta } d t \ = \ \mathrm { B e t a } _ { 1 - \lambda } \big ( 1 + { \textstyle \frac { 1 } { \beta } } , 1 - { \textstyle \frac { 1 } { \beta } } \big ) .\tag{9}
$$

Above, Bet $\textstyle \cdot { \mathrm { a } } _ { u } ( a , b )$ denotes the incomplete Beta function.

The proposition follows by substituting $R _ { \lambda }$ into the Gibbs solution in $( 2 ) { \mathrm { ; } }$ ; Appendix A.1 gives the population normalizer calculation. The result separates support selection from within-support reweighting, where λ controls which completions remain in the support and the sharpening parameter $\beta$ controls how strongly the target policy departs from $\pi _ { \mathrm { r e f } }$ within the retained set.

## 3.2 Theoretical merits of truncate-bad, upweight-good policies

We now provide a theoretical analysis that sheds light on the roles of truncation and reweighting in our target policy. To set the stage, fix a prompt $x ,$ and let $Y \sim \pi _ { \mathrm { r e f } } ( \cdot \mid x )$ . Recall that $w _ { r } ( x , Y )$ is the win-rate computed using the proxy reward $r ( x , \cdot )$ . Define also the oracle win-rate

$$
w _ { u } ( x , Y ) : = \mathbb { P } _ { Y ^ { \prime } \sim \pi _ { \operatorname { r e f } } ( \cdot \vert x ) } \bigl ( u ( x , Y ^ { \prime } ) \leq u ( x , Y ) \bigr ) ,
$$

where $u ( x , \cdot )$ is an unknown oracle reward function.

Algorithm 1 TUP ofline BoN distillation with BCE   
Require: Ofline prompts $\{ x _ { i } \sim \mathcal { D } \} _ { i = 1 } ^ { n }$ , reference policy $\pi _ { \mathrm { r e f } } ,$ , reward model $r ,$ truncation $\lambda ,$ sharpness $\beta$   
Require: For each $x _ { i } .$ , a poo $\{ y _ { i , j } \sim \pi _ { \mathrm { r e f } } ( \cdot \mid x _ { i } ) \} _ { j = 1 } ^ { K }$ with empirical in-pool win-rates   
$\hat { w } _ { r } ( x _ { i } , y _ { i , j } ) = \frac { 1 + \sum _ { \ell \neq j } 1 [ r ( x _ { i } , y _ { i , j } ) \geq r ( x _ { i } , y _ { i , \ell } ) ] } { K } \in \left\{ \frac { 1 } { K } , \ldots , 1 \right\}$   
1: Precompute truncated win-rates and bias term:   
$\hat { w } _ { \lambda , r } ( x _ { i } , y _ { i , j } ) = \operatorname* { m a x } \bigl ( \hat { w } _ { r } ( x _ { i } , y _ { i , j } ) - \lambda , 0 \bigr ) , \qquad b _ { \lambda , \beta } = \beta \log Z _ { \lambda , \beta }$   
2: for each training step do   
3: Sample a minibatch of indexed pool elements $( x _ { i } , y _ { i , j } )$ from the reference-sampled pool   
4: Compute the logit and predicted probability:   
$s _ { \theta } ( x _ { i } , y _ { i , j } ) = \beta \log \frac { \pi _ { \theta } ( y _ { i , j } \mid x _ { i } ) } { \pi _ { \mathrm { r e f } } ( y _ { i , j } \mid x _ { i } ) } + b _ { \lambda , \beta } , \qquad p _ { \theta } ( x _ { i } , y _ { i , j } ) = \sigma \big ( s _ { \theta } ( x _ { i } , y _ { i , j } ) \big )$   
5: Take a gradient step to minimize the BCE loss:   
$\mathcal { L } _ { \mathrm { B C E } } ( \theta ) = - \hat { w } _ { \lambda , r } ( x _ { i } , y _ { i , j } ) \log p _ { \theta } ( x _ { i } , y _ { i , j } ) - \left( 1 - \hat { w } _ { \lambda , r } ( x _ { i } , y _ { i , j } ) \right) \log \left( 1 - p _ { \theta } ( x _ { i } , y _ { i , j } ) \right)$   
6: end for

The following analysis builds on the general formulation of rank-based policies $\pi _ { g }$ from $( 5 )$ . Notably, our proposed policy also falls under this construction, with the function $\begin{array} { r } { g _ { \lambda , \beta } ( w ) \propto \left( \frac { w - \lambda } { 1 - w + \lambda } \right) ^ { 1 / \beta } \mathbb { 1 } \{ w > \lambda \} } \end{array}$ . With this in place, we can evaluate a candidate policy $\pi _ { g }$ using the unknown oracle reward, through the expected oracle win-rate of completions $Y$ sampled from $\pi _ { g } \colon$

$$
\mathcal { U } _ { x } ( g ) : = \mathbb { E } _ { Y \sim \pi _ { g } ( \cdot | x ) } [ w _ { u } ( x , Y ) ] .
$$

Our first result shows that, among monotone reweightings of proxy rank, it is enough to consider hard truncation rules. Let $\mathcal { F } _ { \uparrow }$ denote the set of nondecreasing densities on $[ 0 , 1 ]$ , and for each $\lambda \in [ 0 , 1 )$ , let $\nu _ { \lambda } ( w ) : = { \textstyle { \frac { 1 } { 1 - \lambda } } } \mathbb { 1 } \{ w \in [ \lambda , 1 ] \}$ denote the uniform density on the upper tail, which is the limit $g _ { \lambda , \infty } ( w ) \propto \mathbb { 1 } \{ w >$ $\lambda \}$ . With this notation, we state the following result.

Theorem 3.2 (Informal). Fix a prompt x, and assume that the proxy reward $r ( x , Y )$ is continuous under $Y \sim \pi _ { \mathrm { r e f } } ( \cdot \mid x )$ , so that the proxy win-rate $w _ { r } ( x , Y )$ is uniform on [0, 1]. Let $\mathcal { F } _ { \uparrow }$ be the class of nonnegative nondecreasing densities on [0, 1]. Then

$$
\operatorname* { s u p } _ { f \in \mathcal { F } _ { \uparrow } } \mathcal { U } _ { x } ( f ) = \operatorname* { s u p } _ { \lambda \in [ 0 , 1 ) } \mathcal { U } _ { x } ( \nu _ { \lambda } ) .
$$

The formal statement and proof are given in Appendix A.2.1. This result implies that the best hard-threshold rule, obtained by optimizing λ for each prompt $x ,$ can match the best rank-based rule in those families, such as QRPO and BoNBoN. In practice, tuning λ for each prompt $x$ is infeasible, and thus we treat it as a global quality floor. To empirically quantify the potential gain of prompt-adaptive truncation over a fixed global threshold, we consider a simplified, training-free experiment that treats a strong reward model as an “oracle” evaluator. We show that the diference is relatively insignificant. For full details, refer to Section B.3 of the Appendix.

Once the threshold is fixed, we show that tail sharpening can further improve the policy if the proxy win rates within the retained tail is informative about oracle preferences. To formalize this relationship, define the oracle–proxy profile as $m _ { x } ( w ) : = \mathbb { E } _ { Y \sim \pi _ { \mathrm { r e f } } ( \cdot \vert x ) } [ w _ { u } ( x , Y ) ~ \vert ~ w _ { r } ( x , Y ) = w ]$ , the average oracle win rate among completions with proxy win rate w. In oracle rank space, the following result states that a finite $\beta$ can improve over pure truncation when this profile is positively aligned with the within-tail log-odds tilt. The formal statement and proof are given in Appendix A.2.2.

Proposition 3.3 (Informal). Fix a threshold $\lambda _ { 0 } \in ( 0 , 1 )$ . If, within the retained tail $[ \lambda _ { 0 } , 1 ]$ , the oracle profile $m _ { x }$ has positive covariance with the within-tail log-odds tilt $\log ( ( w - \lambda _ { 0 } ) / ( 1 - w + \lambda _ { 0 } ) )$ ), then there exists a finite $\beta _ { 0 } \in ( 0 , \infty )$ such that $\pi _ { \lambda _ { 0 } , \beta _ { 0 } } ^ { \star }$ achieves strictly larger expected oracle win-rate than pure truncation policy $\pi _ { \lambda _ { 0 } , \infty } ^ { \star }$

As with the truncation threshold λ, the theoretically optimal tail-sharpening parameter $\beta$ is prompt-specific. In practice, however, we use a single global value of $\beta$ across all prompts. In this sense, Theorem 3.2 and Proposition 3.3 provide idealized theoretical support for separating truncation from sharpness, but they do not directly explain how a training model can benefit from it.

To demonstrate the efects of threshold λ and the reweighting efect of a finite $\beta ,$ , Appendix A.2.3 and Figure 3 use a stylized, unnormalized quadratic oracle–proxy example; the appendix shows that normalization does not afect the relevant comparisons. The improvement of the pure-truncation policy $\pi _ { \lambda , \infty } ^ { \star }$ over QRPO reflects the gain from using a threshold to remove low-ranked completions. The improvement of the best finite-β TUP policy over $\pi _ { \lambda , \infty } ^ { \star }$ reflects the additional gain from reweighting within the retained tail once the threshold is fixed.

## 3.3 Training objective: policy learning as a classification problem

Having established the theoretical properties of our proposed policy $\pi _ { \lambda , \beta } ^ { \star } ,$ we now turn to formulate the training objective induced by this target. For completions in the support of $\pi _ { \lambda , \beta } ^ { \star } ,$ rearranging (8) gives

$$
w _ { \lambda , r } ( x , y ) \ : = \ : \sigma \left( \beta \log \frac { \pi _ { \lambda , \beta } ^ { \star } ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } + \beta \log Z _ { \lambda , \beta } \right) ,\tag{10}
$$

where $\sigma$ is the sigmoid function and logit is its inverse on $( 0 , 1 )$ , so $\sigma ( \mathrm { l o g i t } ( x ) ) = x$ for $x \in ( 0 , 1 )$ , with endpoint limits 0 and 1 as $x \to 0 ^ { + }$ and $x \to 1 ^ { - }$ . The above relation suggests that we can fit a parameterized policy $\pi _ { \theta }$ that estimates $\pi _ { \lambda , \beta } ^ { \star }$ using logistic regression of the shifted-truncated win-rate. Indeed, we can define our classifier as

$$
p _ { \theta } ( x , y ) : = \sigma \left( \beta \log \frac { \pi _ { \theta } ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } + \beta \log Z _ { \lambda , \beta } \right) ,\tag{11}
$$

where β log $Z _ { \lambda , \beta }$ acts as a constant and known intercept. For empirical finite-pool ranks, the normalizer has an exact finite-K analogue of the same form, with the integral replaced by a discrete average over rank locations. As in QRPO, our reported experiments use the population intercept, while Appendix A.1 gives the finite-pool alternative. Empirically, the population and finite-pool normalizers performs similarly to the analytical normalizer, except at extremely small $\beta ,$ where the finite-pool form is needed to keep the normalizer finite.

The construction above naturally results in a fully ofline binary cross-entropy (BCE) objective with the shifted-truncated win-rate as a soft label:

$$
\begin{array} { r l } & { \mathrm { T h e ~ B C E ~ T U P ~ d i s t i l l a t i o n ~ o b j e c t i v e ~ i s : } } \\ & { \qquad \mathcal { L } ( \theta ) \ = \ \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \mathrm { r e f } } ( \cdot \vert x ) } \Big [ { - } w _ { \lambda , r } ( x , y ) \log \bigr ( p _ { \theta } ( x , y ) \bigr ) - ( 1 - w _ { \lambda , r } ( x , y ) ) \log \bigr ( 1 - p _ { \theta } ( x , y ) \bigr ) \Big ] . } \end{array}\tag{12}
$$

This loss requires only precomputed scalar rank labels $w _ { \lambda , r } ( x , y )$ and the global intercept $\beta$ log $Z _ { \lambda , \beta }$ . It requires no pairwise preferences, no online sampling, and no runtime estimation of a prompt-dependent partition function.

Algorithm 1 summarizes the resulting fully ofline training procedure. In practice, for each prompt we first sample a reference pool, compute each completion’s empirical in-pool win-rate $\hat { w } _ { r } ( x , y )$ against the other $K - 1$ completions in that same pool, apply the global truncation $\lambda ,$ and then fit $\pi _ { \theta }$ by binary cross-entropy on the distilled-to-reference log-likelihood ratio.

## 4 Experiments

Experimental setup. We build on the QRPO experimental framework of Matrenok et al. [26], which evaluates ofline alignment methods across multiple models and datasets. We compare TUP with six baselines across two model families, two datasets, three reward-model evaluators, and a GPT judge. We use UltraFeedback [7] and Magpie Air [40] as training datasets. In our experiments, we apply each alignment method to Llama 8B Tülu 3 SFT [22] separately on both datasets and to Mistral-7B [18] on Magpie Air; fo Mistral, we first perform dedicated supervised fine-tuning (SFT) to obtain the initial reference policy.

Following QRPO, we use their published datasets, which use ArmoRM [39] as the ofline reward model for scoring the reference-model training completions. To test generalization beyond the training reward model, we evaluate using two Skywork-v2 reward models, Skywork-Reward-V2-Llama-3.1-8B and Skywork-Reward-V2-Qwen3-8B [25], referred to as Skywork-Llama and Skywork-Qwen respectively. These models are ranked #1 and #2, respectively, among available classifier reward models on the RewardBench leaderboard at the time of writing [23].<sup>1</sup> We additionally evaluate on AlpacaEval using gpt-4o as a judge [16, 45]. Table 8 of the Appendix reports the RewardBench standings of all reward and judge models in our experiments. We report length-controlled (LC) rewards in the main tables; raw reward results are given in Appendix B.1.

The most natural baselines for our setting are rank-based distillation methods. We include BoNBoN [14] and the QRPO variant that computes the loss with random completions [26], denoted QRPO (random). While less related, we also compare to strong ofline-alignment baselines using pairwise or relative-reward signals, namely DPO [31] and REBEL [12]. Unless marked random, DPO and REBEL use the best–worst pair from each pool; REBEL (random) uses random pairs. We also include QRPO trained on best–worst pairs. For all DPO, REBEL, and QRPO variants, we adopt the best-performing configurations from the of-policy setting reported in Matrenok et al. [26]. To avoid clutter, tables omit the best–worst label when it is the default. We do not include SimPO [27], whose length-normalized objective makes it structurally diferent from baselines that do not explicitly account for length. Because BoNBoN is not covered by the QRPO benchmark, we run a dedicated hyperparameter search and select the best checkpoint by validation reward. Full implementation details are provided in Appendix C.

In all tables, unless stated otherwise, we show TUP with 3 diferent global truncation values; λ = 0.2 (mild) which truncates only the worst rollout out of the 6 completions, λ = 0.5 (mid.) and λ = 0.8 (aggressive) that keeps only the top two completions.

Results. Table 1 reports two in-dataset experiments on Llama 8B, where each method is evaluated on the held-out split of its respective training dataset, UltraFeedback [7] or Magpie Air [40]. As shown, TUP is competitive with strong ofline alignment baselines on both datasets. Under ArmoRM, all methods achieve broadly similar results. Larger diferences appear under the two Skywork reward models. On both UltraFeedback and Magpie Air, TUP obtains the best in-dataset results under both Skywork-Llama and Skywork-Qwen. If these reward models are viewed as diferent proxies for the oracle reward, the results suggest that TUP generalizes well across proxy rewards compared to the baselines considered.

Table 1: In-dataset evaluation for Llama 8B Tülu 3 SFT trained separately on UltraFeedback and Magpie Air. We report the LC reward metric for each reward model.
<table><tr><td rowspan="2">Method</td><td colspan="3">UltraFeedback eval split</td><td colspan="3">Magpie Air eval split</td></tr><tr><td>ArmoRM</td><td>Skywork-Llama</td><td>Skywork-Qwen</td><td>ArmoRM</td><td>Skywork-Llama</td><td>Skywork-Qwen</td></tr><tr><td>Initial</td><td>0.1300±0.0008</td><td>9.18±0.24</td><td>3.80±0.19</td><td>0.1531±0.0012</td><td>10.14±0.30</td><td>5.08±0.25</td></tr><tr><td>DPO</td><td>0.1499±0.0001</td><td>16.16±0.06</td><td>7.97±0.03</td><td>0.1727±0.0005</td><td>16.33±0.15</td><td>9.89±0.05</td></tr><tr><td>REBEL</td><td>0.1494±0.0003</td><td>16.37±0.14</td><td>8.01±0.07</td><td>0.1709±0.0007</td><td>17.32±0.34</td><td>9.94±0.15</td></tr><tr><td>REBEL (random)</td><td>0.1496±0.0005</td><td>16.84±0.14</td><td>8.14±0.08</td><td>0.1718±0.0001</td><td>15.08±0.19</td><td>9.27±0.06</td></tr><tr><td>QRPO</td><td>0.1521±0.0003</td><td>17.20±0.29</td><td>8.22±0.08</td><td>0.1748±0.0003</td><td>16.32±0.29</td><td>9.07±0.18</td></tr><tr><td>QRPO (random)</td><td>0.1522±0.0004</td><td>16.92±0.08</td><td>8.05±0.07</td><td>0.1695±0.0005</td><td>15.31±0.30</td><td>8.54±0.24</td></tr><tr><td>BoNBoN</td><td>0.1409±0.0003</td><td>12.74±0.05</td><td>5.92±0.04</td><td>0.1641±0.0008</td><td>13.76±0.20</td><td>7.60±0.07</td></tr><tr><td>TUP mild</td><td>0.1500±0.0006</td><td>17.47±0.28</td><td>8.38±0.12</td><td>0.1798±0.0006</td><td>20.68±0.46</td><td>12.27±0.25</td></tr><tr><td>TUP mid.</td><td>0.1499±0.0002</td><td>17.45±0.13</td><td>8.39±0.07</td><td>0.1806±0.0013</td><td>21.24±0.20</td><td>12.49±0.11</td></tr><tr><td>TUP aggressive</td><td>0.1499±0.0003</td><td>17.54±0.16</td><td>8.43±0.05</td><td>0.1794±0.0037</td><td>19.64±0.58</td><td>11.38±0.24</td></tr></table>

Table 2: AlpacaEval performance for Llama 8B Tülu 3 SFT trained on UltraFeedback (top) and Magpie Air (bottom). We report the LC reward metric for each reward model, and win-rate using gpt-4o judge results; length is measured in characters. Methods marked “random” are trained on a random pair of completions from each pool, whereas unmarked pairwise methods use the best and worst completions. TUP is trained on a single random completion per pool.
<table><tr><td colspan="2"></td><td colspan="3">Reward model</td><td colspan="3">GPT Judge</td></tr><tr><td>Train set Method</td><td></td><td>ArmoRM</td><td></td><td>Skywork-Llama Skywork-Qwen</td><td>LC Win</td><td>Win</td><td>Len</td></tr><tr><td rowspan="10">UItrack</td><td>Initial</td><td> $0 . 1 4 4 8 { \scriptstyle \pm 0 . 0 0 3 0 }$ </td><td> $1 3 . 4 4 { \scriptstyle \pm 1 . 6 0 }$ </td><td> $6 . 6 5 { \scriptstyle \pm 0 . 8 9 }$ </td><td> $1 5 . 1 8 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $8 . 5 7 { \scriptstyle \pm 0 . 9 8 }$ </td><td>1011</td></tr><tr><td>DPO</td><td> $0 . 1 6 1 9 { \scriptstyle \pm 0 . 0 0 0 1 }$ </td><td> $2 1 . 5 9 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $1 0 . 9 1 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $\mathbf { 4 2 . 3 0 { \scriptstyle \pm 0 . 1 6 } }$ </td><td> $3 7 . 0 8 { \scriptstyle \pm 1 . 7 0 }$ </td><td>1807</td></tr><tr><td>REBEL</td><td> $0 . 1 6 2 0 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $2 2 . 1 5 { \scriptstyle \pm 0 . 1 2 }$ </td><td>11.19±0.08</td><td> $4 1 . 8 1 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $3 9 . 8 1 { \scriptstyle \pm 1 . 7 3 }$ </td><td>1938</td></tr><tr><td>REBEL (random)</td><td> $0 . 1 6 2 5 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $2 2 . 5 0 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $1 1 . 3 1 { \pm } 0 . 0 5$ </td><td> $\mathbf { 4 2 . 4 4 { \scriptstyle \pm 0 . 1 5 } }$ </td><td> $4 0 . 9 3 { \scriptstyle \pm 1 . 7 3 }$ </td><td>1959</td></tr><tr><td>QRPO</td><td> $\mathbf { 0 . 1 6 6 8 { \scriptstyle \pm 0 . 0 0 0 0 } }$ </td><td> $2 0 . 6 9 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $1 0 . 2 3 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $3 9 . 6 4 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $\mathbf { 5 3 . 1 7 { \scriptstyle \pm 1 . 7 6 } }$ </td><td>3100</td></tr><tr><td>QRPO (random)</td><td> $0 . 1 6 6 3 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $2 2 . 2 6 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $1 1 . 1 1 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $\mathbf { 4 2 . 3 5 { \scriptstyle \pm 0 . 1 3 } }$ </td><td> $4 8 . 5 1 { \scriptstyle \pm 1 . 7 6 }$ </td><td>2436</td></tr><tr><td>BoNBoN</td><td> $0 . 1 5 0 3 { \scriptstyle \pm 0 . 0 0 2 8 }$ </td><td> $1 6 . 7 5 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $8 . 4 4 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $2 3 . 2 8 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $1 5 . 4 0 { \scriptstyle \pm 1 . 2 7 } $ </td><td>1325</td></tr><tr><td>TUP mild</td><td> $0 . 1 6 3 8 { \scriptstyle \pm 0 . 0 0 0 0 }$ </td><td> $2 2 . 8 5 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $1 1 . 4 1 { \pm } 0 . 1 5$ </td><td> $3 9 . 4 1 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $4 9 . 5 0 { \scriptstyle \pm 1 . 7 6 }$ </td><td>2730</td></tr><tr><td>TUP mid.</td><td> $0 . 1 6 3 3 { \scriptstyle \pm 0 . 0 0 0 5 }$ </td><td> $\mathbf { 2 3 . 0 5 } \pm \mathbf { 0 . 2 3 }$ </td><td> $\mathbf { 1 1 . 5 4 2 0 . 0 5 }$ </td><td> $4 0 . 2 7 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $4 9 . 1 3 { \scriptstyle \pm 1 . 7 6 }$ </td><td>2670</td></tr><tr><td>TUP aggressive</td><td> $0 . 1 6 3 8 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $2 2 . 9 0 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $1 1 . 4 6 { \scriptstyle \pm 0 . 1 9 } $ </td><td> $\mathbf { 4 2 . 3 6 { \scriptstyle \pm 0 . 1 4 } }$ </td><td> $5 1 . 2 4 { \scriptstyle \pm 1 . 7 6 }$ </td><td>2709</td></tr><tr><td rowspan="10">Mae AAir</td><td>Initial</td><td> $0 . 1 4 4 8 { \scriptstyle \pm 0 . 0 0 3 0 }$ </td><td> $1 3 . 4 4 { \scriptstyle \pm 1 . 6 0 }$ </td><td> $6 . 6 5 { \scriptstyle \pm 0 . 8 9 }$ </td><td> $1 5 . 1 8 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $8 . 5 7 { \scriptstyle \pm 0 . 9 8 }$ </td><td>1011</td></tr><tr><td>DPO</td><td> $0 . 1 5 8 1 { \scriptstyle \pm 0 . 0 0 0 8 }$ </td><td> $1 8 . 9 4 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $9 . 7 3 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $3 2 . 1 1 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $2 6 . 1 5 { \scriptstyle \pm 1 . 5 5 }$ </td><td>1644</td></tr><tr><td>REBEL</td><td> $0 . 1 5 9 1 { \scriptstyle \pm 0 . 0 0 0 7 }$ </td><td> $2 0 . 6 5 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $1 0 . 3 7 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $3 9 . 6 6 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $3 6 . 5 8 { \scriptstyle \pm 1 . 7 0 }$ </td><td>1872</td></tr><tr><td>REBEL (random)</td><td> $0 . 1 5 6 5 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $1 7 . 8 1 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $9 . 1 5 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $3 1 . 3 7 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $2 6 . 2 1 { \scriptstyle \pm 1 . 5 5 }$ </td><td>1675</td></tr><tr><td>QRPO</td><td> $0 . 1 5 7 8 { \scriptstyle \pm 0 . 0 0 0 0 }$ </td><td> $1 9 . 0 0 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $9 . 3 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $3 2 . 3 9 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $3 2 . 4 8 { \scriptstyle \pm 1 . 6 5 }$ </td><td>1992</td></tr><tr><td>QRPO (random)</td><td> $0 . 1 5 5 3 { \scriptstyle \pm 0 . 0 0 0 9 }$ </td><td> $1 8 . 6 4 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $9 . 2 4 { \pm } 0 . 1 5$ </td><td> $2 7 . 9 7 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $2 3 . 5 4 { \scriptstyle \pm 1 . 4 9 }$ </td><td>1628</td></tr><tr><td>BoNBoN</td><td> $0 . 1 4 4 6 { \scriptstyle \pm 0 . 0 0 1 1 }$ </td><td> $1 4 . 9 2 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $7 . 3 2 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $1 9 . 8 2 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $1 3 . 7 3 { \scriptstyle \pm 1 . 2 1 }$ </td><td>1346</td></tr><tr><td>TUP mild</td><td> $0 . 1 5 8 5 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $2 0 . 5 8 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $1 0 . 2 1 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $3 6 . 8 2 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $\mathbf { 4 3 . 2 9 2 1 . 7 5 }$ </td><td>2712</td></tr><tr><td>TUP mid.</td><td> $0 . 1 5 8 5 { \scriptstyle \pm 0 . 0 0 0 6 }$ </td><td> $\mathbf { 2 1 . 0 8 { \scriptstyle \pm 0 . 4 4 } }$ </td><td> $\mathbf { 1 0 . 4 4 { \scriptstyle \pm 0 . 0 8 } }$ </td><td> $3 5 . 8 1 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $4 2 . 5 5 { \scriptstyle \pm 1 . 7 4 }$ </td><td>2645</td></tr><tr><td>TUP aggressive</td><td> $0 . 1 5 7 4 { \scriptstyle \pm 0 . 0 0 0 6 }$ </td><td> $1 9 . 7 6 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $9 . 7 7 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $3 2 . 3 1 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $3 8 . 9 4 { \scriptstyle \pm 1 . 7 2 }$ </td><td>2626</td></tr></table>

Table 2 reports AlpacaEval results for the above mentioned trained Llama models. Under both datasets, TUP obtains the best results under both Skywork-Llama and Skywork-Qwen, including on AlpacaEval LC reward, and remains highly competitive under the GPT judge. While the Magpie-Air-trained models on AlpacaEval results are more mixed, TUP remains competitive. The supplementary raw reward tables in Appendix B.1 provide the corresponding non-LC scores and in-dataset response lengths, showing that the main trends are not limited to the LC metric. Table 3, which reports both in-dataset and AlpacaEval LC rewards, shows that TUP remains competitive when applied to the Mistral model family.

In addition to the LC-reward results reported above, we conduct a pairwise length-matched reward comparison to further isolate TUP’s reward gains from the efect of its longer average responses. As reported in Table 7 of the Appendix, TUP responses are largely preferred against similar-length responses from the baseline methods on the same prompt. This provides additional evidence that our advantage does not stem from response length alone. Full details are in Section B.2 of the Appendix.

Figure 4 shows the efects of $\mathrm { T U P } { \mathrm { s } }$ truncation threshold λ and preference sharpness β, using the same validation-based checkpoint-selection protocol as the main experiments (Tables 1–5). To approximate the no-truncation setting, we use $\lambda = 0 . 2$ , which removes only the lowest-ranked of the six responses, a response that would otherwise receive an extremely low weight. To approximate “pure” truncation, we use $\beta = 1 0 0$ which produces an approximately uniform weighting over the retained tail while still remaining finite and allowing for some deviation from $\pi _ { \mathrm { r e f } } .$ Generally, intermediate values of both λ and $\beta$ tend to yield the highest

Table 3: LC rewards for Mistral 7B SFT, trained on Magpie Air. Tested on in-dataset test split (MA) and AlpacaEval (AE) evaluated by ArmoRM, Skywork-Llama and Skywork-Qwen reward models, with AlpacaEva also judged by $\tt { g p t - 4 0 }$
<table><tr><td></td><td colspan="2">ArmoRM</td><td colspan="2">Skywork-v2-Llama</td><td colspan="2"> $\mathrm { S k y w o r k - v 2 - Q w e n }$ </td><td colspan="3">GPT Judge</td></tr><tr><td>Method</td><td>MA</td><td>AE</td><td>MA</td><td>AE</td><td>MA</td><td>AE</td><td>LC Win</td><td>Win</td><td>Len</td></tr><tr><td>Initial</td><td>0.1646±0.0006</td><td>0.1477±0.0002</td><td>17.36±0.11</td><td>16.53±0.19</td><td>11.57±0.07</td><td> $8 . 7 6 { \scriptstyle \pm 0 . 0 9 }$ </td><td>21.94±0.27</td><td> $1 8 . 7 6 { \scriptstyle \pm 1 . 3 8 }$ </td><td>1736</td></tr><tr><td>DPO</td><td> $0 . 1 7 9 2 { \scriptstyle \pm 0 . 0 0 0 8 }$ </td><td> $0 . 1 5 8 3 { \scriptstyle \pm 0 . 0 0 1 0 }$ </td><td> $2 2 . 6 2 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $2 1 . 5 0 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $1 6 . 9 4 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $1 2 . 4 3 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $3 1 . 8 0 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $3 7 . 1 4 { \pm } 1 . 7 0 $ </td><td>2860</td></tr><tr><td>REBEL</td><td> $0 . 1 7 3 2 { \scriptstyle \pm 0 . 0 0 2 1 }$ </td><td> $0 . 1 4 6 2 { \scriptstyle \pm 0 . 0 0 1 6 }$ </td><td> $2 0 . 8 3 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $1 3 . 5 8 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $1 4 . 0 8 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $7 . 5 1 { \pm } 0 . 1 5$ </td><td> $2 2 . 1 1 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $1 9 . 2 5 { \scriptstyle \pm 1 . 3 9 } $ </td><td>8470</td></tr><tr><td>REBEL (random)</td><td> $0 . 1 8 2 2 { \scriptstyle \pm 0 . 0 0 2 6 }$ </td><td> $0 . 1 5 8 8 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $\mathbf { 2 4 . 6 4 } \pm \mathbf { 0 . 2 3 }$ </td><td> $\mathbf { 2 1 . 9 1 } \pm \mathbf { 0 . 4 6 }$ </td><td> $\mathbf { 1 8 . 3 2 { \scriptstyle \pm 0 . 2 6 } }$ </td><td> $\mathbf { 1 2 . 6 7 { \scriptstyle \pm 0 . 1 8 } }$ </td><td> $3 4 . 4 5 { \pm } 0 . 1 2 $ </td><td> $4 0 . 1 2 { \scriptstyle \pm 1 . 7 3 }$ </td><td>3013</td></tr><tr><td>QRPO</td><td> $0 . 1 7 2 0 { \scriptstyle \pm 0 . 0 0 0 5 }$ </td><td> $0 . 1 5 6 5 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $2 1 . 0 9 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $2 0 . 2 9 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $1 4 . 2 3 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 1 . 0 2 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $3 1 . 9 7 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $3 4 . 7 8 { \scriptstyle \pm 1 . 6 8 }$ </td><td>2261</td></tr><tr><td>QRPO (random)</td><td> $0 . 1 8 5 8 { \scriptstyle \pm 0 . 0 0 1 3 }$ </td><td> $0 . 1 4 2 5 { \scriptstyle \pm 0 . 0 0 7 8 }$ </td><td> $2 1 . 9 6 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $1 7 . 2 3 { \pm } 1 . 5 3 $ </td><td> $1 5 . 8 6 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $9 . 5 4 { \pm } 1 . 0 4 $ </td><td> $2 7 . 5 3 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $2 9 . 0 7 { \scriptstyle \pm 1 . 6 0 }$ </td><td>4740</td></tr><tr><td>BoNBoN</td><td> $0 . 1 6 6 7 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $0 . 1 4 9 4 { \scriptstyle \pm 0 . 0 0 0 6 }$ </td><td> $1 8 . 4 0 { \scriptstyle \pm 0 . 3 2 } $ </td><td> $1 7 . 4 6 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $1 2 . 3 5 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $9 . 3 4 { \pm } 0 . 0 9$ </td><td> $2 4 . 8 5 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $2 2 . 1 1 { \pm } 1 . 4 6$ </td><td>1828</td></tr><tr><td>TUP mild</td><td> $\mathbf { 0 . 1 8 6 4 { \scriptstyle \pm 0 . 0 0 0 4 } }$ </td><td> $0 . 1 6 1 2 { \scriptstyle \pm 0 . 0 0 3 2 }$ </td><td> $2 3 . 4 7 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $2 1 . 0 4 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $1 7 . 0 1 { \pm } 0 . 1 0$ </td><td> $1 1 . 7 5 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $\mathbf { 3 6 . 8 4 { \scriptstyle \pm 0 . 0 4 } }$ </td><td> $\mathbf { 4 0 . 6 2 { \scriptstyle \pm 1 . 7 3 } }$ </td><td>3382</td></tr><tr><td>TUP mid.</td><td> $0 . 1 8 3 2 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $0 . 1 6 2 0 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $2 2 . 8 1 { \pm } 0 . 1 8$ </td><td> $2 1 . 3 4 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $1 6 . 9 6 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $1 2 . 2 5 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $3 5 . 6 9 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $3 9 . 2 5 { \scriptstyle \pm 1 . 7 2 }$ </td><td>2765</td></tr><tr><td>TUP aggressive</td><td> $0 . 1 8 2 6 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $\mathbf { 0 . 1 6 2 5 { \scriptstyle \pm 0 . 0 0 2 4 } }$ </td><td> $2 1 . 3 2 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $2 0 . 5 0 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $1 6 . 0 0 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $1 1 . 5 4 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $3 3 . 2 2 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $3 9 . 6 3 { \scriptstyle \pm 1 . 7 3 }$ </td><td>2688</td></tr></table>

![](images/6cbba1684070697799da3180d127c2d90499c68cbf7fd7d478ca4726118d0088.jpg)

![](images/8008c1d4bd588682597f3facddf0ca44515284e00c45b76faee47ba387b26a83.jpg)  
Figure 4: Ablation study of $\mathrm { T U P } { \mathrm { s } }$ truncation threshold λ and sharpness $\beta$ for Llama 8B Tülu 3 SFT trained on Magpie Air and evaluated on AlpacaEval with Skywork-Llama reward model. TUP uses $K = 6$ reference completions per prompt. Cells report the mean and sample standard deviation across three generation seeds. Left: LC rewards; higher is better. Right: Mean response length in tokens.

LC rewards. These results suggest that the best performance comes from combining moderate truncation with non-uniform within-tail reweighting, supporting our choice to treat λ and $\beta$ as separate parameters.

## 5 Discussion

One limitation of our method is that both λ and $\beta$ must be tuned based on validation performance, whereas other baselines tune only $\beta .$ This may increase the computational cost when a suitable initial range for λ is unknown. A practical way to narrow this range is to construct a plot such as the right panel of Figure 2. Given a dataset, a proxy reward model, and an auxiliary reward model treated as an “oracle,” this plot provides an eficient way to identify a promising range of truncation levels. Once a truncation level that retains responses ranked highly by the “oracle” while removing poorly ranked ones is identified, the remaining hyperparameter search is once again only on $\beta$ values.

TUP also leaves several directions for future work. First, although TUP improves rank-based policy distillation, it is still exposed to reward hacking; incorporating reward-hacking mitigation methods could further improve its performance. Second, future work could replace the fixed global λ and $\beta$ with potentially learned prompt-specific parameters.

More broadly, TUP may support safety alignment when unsafe or otherwise undesirable completions are consistently assigned low proxy ranks, allowing truncation to remove them from the target support rather than merely down-weight them.

## References

[1] Zachary Ankner, Mansheej Paul, Brandon Cui, Jonathan Daniel Chang, and Prithviraj Ammanabrolu. Critique-out-Loud Reward Models. In Pluralistic Alignment Workshop at NeurIPS, 2024.

[2] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

[3] Ananth Balashankar, Ziteng Sun, Jonathan Berant, Jacob Eisenstein, Michael Collins, Adrian Hutter, Jong Lee, Chirag Nagpal, Flavien Prost, and Aradhana Sinha. InfAlign: Inference-aware language model alignment. In Forty-second International Conference on Machine Learning, 2025.

[4] Ahmad Beirami, Alekh Agarwal, Jonathan Berant, Alexander D’Amour, Jacob Eisenstein, Chirag Nagpal, and Ananda Theertha Suresh. Theoretical guarantees on the best-of-N alignment policy. In Proceedings of the 42nd International Conference on Machine Learning, 2025.

[5] Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.

[6] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, 2017.

[7] Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. Ultrafeedback: Boosting language models with high-quality feedback. 2023.

[8] Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, KaShun Shum, and Tong Zhang. RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment. Transactions on Machine Learning Research, 2023.

[9] Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. Length-controlled alpacaeval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475, 2024.

[10] Evan Frick, Tianle Li, Connor Chen, Wei-Lin Chiang, Anastasios Nikolas Angelopoulos, Jiantao Jiao, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. How to Evaluate Reward Models for RLHF. In The Thirteenth International Conference on Learning Representations, 2025.

[11] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In Proceedings of the 40th International Conference on Machine Learning, 2023.

[12] Zhaolin Gao, Jonathan D. Chang, Wenhao Zhan, Owen Oertell, Gokul Swamy, Kianté Brantley, Thorsten Joachims, J. Andrew Bagnell, Jason Lee, and Wen Sun. REBEL: Reinforcement Learning via Regressing Relative Rewards. In Advances in Neural Information Processing Systems, 2024.

[13] Shivank Garg, Ayush Singh, Shweta Singh, and Paras Chopra. IPO: Your Language Model is Secretly a Preference Classifier. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, 2025.

[14] Lin Gui, Cristina Gârbacea, and Victor Veitch. BoNBoN alignment for large language models and the sweetness of best-of-n sampling. In Advances in Neural Information Processing Systems, 2024.

[15] Audrey Huang, Adam Block, Qinghua Liu, Nan Jiang, Akshay Krishnamurthy, and Dylan J Foster. Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment. In Proceedings of the 42nd International Conference on Machine Learning, 2025.

[16] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

[17] Hamish Ivison, Yizhong Wang, Jiacheng Liu, Zeqiu Wu, Valentina Pyatkin, Nathan Lambert, Noah A. Smith, Yejin Choi, and Hannaneh Hajishirzi. Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback. In Advances in Neural Information Processing Systems, 2024.

[18] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

[19] Yuu Jinnai, Tetsuro Morimura, Kaito Ariu, and Kenshi Abe. Regularized best-of-n sampling to mitigate reward hacking for language model alignment. In ICML 2024 Workshop on Models of Human Feedback for AI Alignment, 2024.

[20] Hadi Khalaf, Claudio Mayrink Verdun, Alex Oesterling, Himabindu Lakkaraju, and Flavio du Pin Calmon. Inference-Time Reward Hacking in Large Language Models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[21] Sunghwan Kim, Dongjin Kang, Taeyoon Kwon, Hyungjoo Chae, Dongha Lee, and Jinyoung Yeo. Rethinking reward model evaluation through the lens of reward overoptimization. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025.

[22] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James Validad Miranda, Alisa Liu, Nouha Dziri, Xinxi Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Christopher Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tulu 3: Pushing frontiers in open language model post-training. In Second Conference on Language Modeling, 2025.

[23] Nathan Lambert, Valentina Pyatkin, Jacob Morrison, Lester James Validad Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, and Yejin Choi. RewardBench: Evaluating reward models for language modeling. In Findings of the Association for Computational Linguistics: NAACL 2025, 2025.

[24] Eddie Landesberg. When LLM judge scores look good but best-of-n decisions fail. arXiv preprint arXiv:2603.12520, 2026.

[25] Chris Yuhao Liu, Liang Zeng, Yuzhen Xiao, Jujie He, Jiacai Liu, Chaojie Wang, Rui Yan, Wei Shen, Fuxiang Zhang, and Jiacheng Xu. Skywork-reward-v2: Scaling preference data curation via human-ai synergy. In The Fourteenth International Conference on Learning Representations, 2026.

[26] Simon Matrenok, Skander Moalla, and Caglar Gulcehre. Quantile Reward Policy Optimization: Alignment with Pointwise Regression and Exact Partition Functions. In Advances in Neural Information Processing Systems, 2025.

[27] Yu Meng, Mengzhou Xia, and Danqi Chen. SimPO: Simple preference optimization with a reference-free reward. In Advances in Neural Information Processing Systems, 2024.

[28] Eric J Michaud, Adam Gleave, and Stuart Russell. Understanding learned reward functions. arXiv preprint arXiv:2012.05862, 2020.

[29] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jef Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

[30] Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, and Alex Ray. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 2022.

[31] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, 2023.

[32] Rafael Rafailov, Yaswanth Chittepu, Ryan Park, Harshit Sikchi, Joey Hejna, W. Bradley Knox, Chelsea Finn, and Scott Niekum. Rafailov laws for reward model overoptimization in direct alignment algorithms. In Advances in Neural Information Processing Systems, 2024.

[33] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[34] Pier Giuseppe Sessa, Robert Dadashi-Tazehozi, Leonard Hussenot, Johan Ferret, Nino Vieillard, Alexandre Rame, Bobak Shahriari, Sarah Perrin, Abram L Friesen, and Geofrey Cideron. BOND: Aligning LLMs with Best-of-N Distillation. In The Thirteenth International Conference on Learning Representations, 2025.

[35] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[36] Yifan Song, Guoyin Wang, Sujian Li, and Bill Yuchen Lin. The Good, The Bad, and The Greedy: Evaluation of LLMs Should Not Ignore Non-Determinism. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies, 2025.

[37] Nisan Stiennon, Long Ouyang, Jefrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. In Advances in Neural Information Processing Systems, 2020.

[38] Jeremy Tien, Jerry Zhi-Yang He, Zackory Erickson, Anca Dragan, and Daniel S. Brown. Causal confusion and reward misidentification in preference-based reward learning. In The Eleventh International Conference on Learning Representations, 2023.

[39] Haoxiang Wang, Wei Xiong, Tengyang Xie, Han Zhao, and Tong Zhang. Interpretable preferences via multi-objective reward modeling and mixture-of-experts. In Findings of the Association for Computational Linguistics: EMNLP 2024, 2024.

[40] Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. Magpie: Alignment data synthesis from scratch by prompting aligned LLMs with nothing. In The Thirteenth International Conference on Learning Representations, 2025.

[41] Tong Yang, Jincheng Mei, Hanjun Dai, Zixin Wen, Shicong Cen, Dale Schuurmans, Yuejie Chi, and Bo Dai. Faster wind: Accelerating iterative best-of-n distillation for LLM alignment. In Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research, 2025.

[42] Hongyi Yuan, Zheng Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. RRHF: Rank Responses to Align Language Models with Human Feedback. In Advances in Neural Information Processing Systems, volume 36, 2023.

[43] Junkai Zhang, Zihao Wang, Lin Gui, Swarnashree Mysore Sathyendra, Jaehwan Jeong, Victor Veitch, Wei Wang, Yunzhong He, Bing Liu, and Lifeng Jin. Chasing the Tail: Efective Rubric-based Reward Modeling for Large Language Model Post-Training. In The Fifteenth International Conference on Learning Representations, 2026.

[44] Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J Liu. SLIC-HF: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425, 2023.

[45] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLMas-a-judge with MT-bench and chatbot arena. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023.

[46] Daniel M Ziegler, Nisan Stiennon, Jefrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geofrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

## A Theory and proofs

## A.1 Normalization constant for Proposition 3.1

Proof. By the Gibbs solution in (2), substituting $r = R _ { \lambda }$ yields

$$
\pi ^ { \star } ( y \mid x ) = \frac { 1 } { Z ( x ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \biggl ( \frac { R _ { \lambda } ( x , y ) } { \beta } \biggr ) = \frac { 1 } { Z ( x ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \biggl ( \frac { w _ { \lambda , r } ( x , y ) } { 1 - w _ { \lambda , r } ( x , y ) } \biggr ) ^ { 1 / \beta } .
$$

It therefore remains to identify the normalizer $Z ( x )$ and show that it is independent of x. Under the induced uniform win-rate variable $w \sim \mathrm { U n i f o r m } [ 0 , 1 ]$ , the normalization term becomes

$$
Z ( x ) = \int _ { 0 } ^ { 1 } \left( \frac { \mathrm { m a x } ( w - \lambda , 0 ) } { 1 - \mathrm { m a x } ( w - \lambda , 0 ) } \right) ^ { 1 / \beta } d w .\tag{13}
$$

For $w \leq \lambda$ , the shifted win-rate is zero, so the integrand vanishes. Therefore,

$$
Z ( x ) = \int _ { \lambda } ^ { 1 } \left( \frac { w - \lambda } { 1 - ( w - \lambda ) } \right) ^ { 1 / \beta } d w .\tag{14}
$$

Applying the change of variables $u : = w - \lambda$ gives

$$
Z ( x ) = \int _ { 0 } ^ { 1 - \lambda } \left( \frac { u } { 1 - u } \right) ^ { 1 / \beta } d u = \int _ { 0 } ^ { 1 - \lambda } u ^ { 1 / \beta } ( 1 - u ) ^ { - 1 / \beta } d u .\tag{15}
$$

Recalling that the incomplete Beta function is defined in Proposition 3.1 as $\begin{array} { r } { B _ { z } ( a , b ) = \int _ { 0 } ^ { z } t ^ { a - 1 } ( 1 - t ) ^ { b - 1 } d t } \end{array}$ we identify

$$
\begin{array} { r } { a = 1 + \frac { 1 } { \beta } , \qquad b = 1 - \frac { 1 } { \beta } . } \end{array}
$$

Hence

$$
Z ( x ) = \int _ { 0 } ^ { 1 - \lambda } u ^ { ( 1 + 1 / \beta ) - 1 } ( 1 - u ) ^ { ( 1 - 1 / \beta ) - 1 } d u = \mathrm { B e t a } _ { 1 - \lambda } \Big ( 1 + { \textstyle { \frac { 1 } { \beta } } } , 1 - { \textstyle { \frac { 1 } { \beta } } } \Big ) = Z _ { \lambda , \beta } ,\tag{16}
$$

which is independent of $x .$

Finite-pool normalizer. The closed-form normalizer in Proposition 3.1 is the population normalizer induced by the continuous win-rate $w _ { r } ( x , Y )$ . In the experiments, labels are computed from finite promptspecific pools. Under the empirical-rank convention in Algorithm 1, the corresponding discrete promptindependent normalizer is

$$
Z _ { K , \lambda , \beta } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \left( \frac { ( \frac { j } { K } - \lambda ) _ { + } } { 1 - ( \frac { j } { K } - \lambda ) _ { + } } \right) ^ { 1 / \beta } .
$$

This finite-pool form can replace $Z _ { \lambda , \beta }$ in the BCE intercept without changing the labels, the truncation support, or the loss. In our experiments, we use the population normalizer $Z _ { \lambda , \beta }$ , following the continuous win-rate normalization used in QRPO; in our experiments, replacing it with the finite-pool expression led to negligible performance diferences. In extremely small-β regimes, where the continuous calculation can become numerically unstable, the discrete expression above provides the exact finite-K alternative for this empirical-rank convention.

## A.2 Formal oracle rank-space results for Section 3.2

This appendix formalizes the oracle rank-space discussion from Section 3.2. The aim is to separate two decisions that smooth full-support reweighting conflates: which proxy ranks should remain in the support at all, and how the retained ranks should be reweighted. The first result shows that, among monotone proxy-rank reweightings, the oracle-best rule is always a hard lower-tail truncation. The second result is a fixed-threshold refinement statement: once a truncation level has been chosen, finite within-tail reweighting can improve over pure truncation when oracle value remains positively aligned with proxy rank inside the retained tail. We close with the quadratic worked example behind Figure 3.

Fix a prompt $x ,$ and let $Y \sim \pi _ { \mathrm { r e f } } ( \cdot \mid x )$ . Define the proxy and oracle population win-rates by

$$
w _ { r } ( x , y ) : = \mathbb { P } _ { Y ^ { \prime } \sim \pi _ { \tau \in \mathsf { f } } ( \cdot | x ) } \big ( r ( x , Y ^ { \prime } ) \le r ( x , y ) \big ) , \qquad w _ { u } ( x , y ) : = \mathbb { P } _ { Y ^ { \prime } \sim \pi _ { \tau \in \mathsf { f } } ( \cdot | x ) } \big ( u ( x , Y ^ { \prime } ) \le u ( x , y ) \big ) .
$$

and write

$$
W _ { r } : = w _ { r } ( x , Y ) , \qquad W _ { u } : = w _ { u } ( x , Y ) .
$$

Since BoN-style methods act on relative rank rather than raw reward, the natural policy class in this analysis reweights a completion according to its proxy win-rate. For any measurable $g : [ 0 , 1 ]  [ 0 , \infty )$ satisfying

$$
0 < \mathbb { E } _ { \pi _ { \mathrm { r e f } } } [ g ( W _ { r } ) ] < \infty ,
$$

define

$$
\pi _ { g } ( y \mid x ) : = \frac { g ( w _ { r } ( x , y ) ) \pi _ { \mathrm { r e f } } ( y \mid x ) } { \mathbb { E } _ { Y ^ { \prime } \sim \pi _ { \mathrm { r e f } } ( \cdot \mid x ) } [ g ( w _ { r } ( x , Y ^ { \prime } ) ) ] } , \qquad \mathcal { U } _ { x } ( g ) : = \mathbb { E } _ { Y \sim \pi _ { g } ( \cdot \mid x ) } [ W _ { u } ] .
$$

Finally, let

$$
m _ { x } ( w ) : = \mathbb { E } [ W _ { u } \mid W _ { r } = w ]
$$

denote a measurable version of the conditional oracle profile. Once the problem is reduced to rank space, $m _ { x }$ is the only prompt-specific object that matters.

## A.2.1 Adaptive truncation among monotone reweightings

We first isolate the support-selection question of which proxy ranks should receive positive mass at all. Lemma A.1 (Exact reduction to proxy-rank space). For every measurable g as above,

$$
\mathcal { U } _ { x } ( g ) = \frac { \mathbb { E } _ { \pi _ { \mathrm { r e f } } } [ m _ { x } ( W _ { r } ) g ( W _ { r } ) ] } { \mathbb { E } _ { \pi _ { \mathrm { r e f } } } [ g ( W _ { r } ) ] } .
$$

If moreover the law of $r ( x , Y )$ under $Y \sim \pi _ { \mathrm { r e f } } ( \cdot \mid x )$ is continuous, then $W _ { r } \sim \mathrm { U n i f } [ 0 , 1 ]$ , and therefore

$$
\mathcal { U } _ { x } ( g ) = \frac { \int _ { 0 } ^ { 1 } m _ { x } ( w ) g ( w ) d w } { \int _ { 0 } ^ { 1 } g ( w ) d w } .
$$

Proof. By definition of $\pi _ { g } .$

$$
\mathcal { U } _ { x } ( g ) = \frac { \mathbb { E } _ { \pi _ { \mathrm { r e f } } } [ W _ { u } g ( W _ { r } ) ] } { \mathbb { E } _ { \pi _ { \mathrm { r e f } } } [ g ( W _ { r } ) ] } .
$$

Applying conditional expectation with respect to $W _ { r }$ gives

$$
\operatorname { \mathbb { E } } _ { \pi _ { \mathrm { r e f } } } [ W _ { u } g ( W _ { r } ) ] = \operatorname { \mathbb { E } } _ { \pi _ { \mathrm { r e f } } } [ \operatorname { \mathbb { E } } [ W _ { u } g ( W _ { r } ) \mid W _ { r } ] ] = \operatorname { \mathbb { E } } _ { \pi _ { \mathrm { r e f } } } [ g ( W _ { r } ) \mathbb { E } [ W _ { u } \mid W _ { r } ] ] = \operatorname { \mathbb { E } } _ { \pi _ { \mathrm { r e f } } } [ m _ { x } ( W _ { r } ) g ( W _ { r } ) ] .
$$

This proves the first identity. If the law of $r ( x , Y )$ is continuous, the probability integral transform yields $W _ { r } \sim \mathrm { U n i f } [ 0 , 1 ]$ , and the second identity follows by writing expectations as Lebesgue integrals on [0, 1].

Under the continuity assumption, the problem becomes a pure rank-space optimization over densities on [0, 1]. Let

$$
\mathcal { F } _ { \uparrow } : = \left\{ f : [ 0 , 1 ] \to [ 0 , \infty ) : \int _ { 0 } ^ { 1 } f ( w ) d w = 1 , \ f \ \mathrm { i s ~ n o n d e c r e a s i n g } \right\} ,
$$

and for each threshold $\lambda \in [ 0 , 1 )$ , define the lower-tail truncation density

$$
\nu _ { \lambda } ( w ) : = { \frac { 1 } { 1 - \lambda } } \mathbb { 1 } \{ w \in [ \lambda , 1 ] \} .
$$

The next lemma shows that every monotone density is a mixture of these top-interval uniforms. This is the structural reason hard truncation is enough.

Lemma A.2 (Mixture representation of nondecreasing densities). Let $f \in { \mathcal { F } } _ { \uparrow }$ . Then there exists a probability measure ξ on [0, 1) such that

$$
f ( w ) = \int _ { [ 0 , 1 ) } \nu _ { \lambda } ( w ) \xi ( d \lambda ) \qquad f o r ~ a . e . ~ w \in [ 0 , 1 ] .
$$

Proof. Define a finite Stieltjes measure on [0, 1) by

$$
\xi : = f ( 0 ) \delta _ { 0 } + ( 1 - t ) d f ( t ) .
$$

Since $f$ is nondecreasing and $\begin{array} { r } { \int _ { 0 } ^ { 1 } f ( w ) d w = 1 } \end{array}$ , integration by parts gives

$$
\xi ( [ 0 , 1 ) ) = f ( 0 ) + \int _ { ( 0 , 1 ) } ( 1 - t ) d f ( t ) = 1 .
$$

Thus $\xi$ is a probability measure. For $\mathrm { a . e . ~ } w \in [ 0 , 1 ]$ 2

$$
\int _ { [ 0 , 1 ) } \nu _ { \lambda } ( w ) \xi ( d \lambda ) = \int _ { [ 0 , w ] } \frac { 1 } { 1 - \lambda } \xi ( d \lambda ) = f ( 0 ) + \int _ { ( 0 , w ] } d f ( \lambda ) = f ( w ) .
$$

Theorem A.3 (Hard truncation is optimal among monotone reweightings). Assume the law $o f r ( x , Y )$ under $Y \sim \pi _ { \mathrm { r e f } } ( \cdot \mid x )$ is continuous. Then

$$
\operatorname* { s u p } _ { f \in \mathcal { F } _ { \uparrow } } \mathcal { U } _ { x } ( f ) = \operatorname* { s u p } _ { \lambda \in [ 0 , 1 ) } \mathcal { U } _ { x } ( \nu _ { \lambda } ) .
$$

Equivalently, in the oracle rank-space view, optimizing over monotone proxy-rank reweightings reduces to optimizing over hard lower-tail truncation rules.

Proof. By Lemma A.1, the continuity assumption implies

$$
\mathcal { U } _ { x } ( f ) = \int _ { 0 } ^ { 1 } m _ { x } ( w ) f ( w ) d w \qquad \mathrm { f o r ~ e v e r y ~ } f \in \mathcal { F } _ { \uparrow } .
$$

Fix $f \in { \mathcal { F } } _ { \uparrow }$ . By Lemma A.2,

$$
f ( w ) = \int _ { [ 0 , 1 ) } \nu _ { \lambda } ( w ) \xi ( d \lambda ) \qquad \mathrm { f o r ~ a . e . } ~ w ,
$$

for some probability measure $\xi$ on [0, 1). Therefore,

$$
\mathcal { U } _ { x } ( f ) = \int _ { 0 } ^ { 1 } m _ { x } ( w ) \left( \int _ { [ 0 , 1 ) } \nu _ { \lambda } ( w ) \xi ( d \lambda ) \right) d w .
$$

By Fubini’s theorem,

$$
\mathcal { U } _ { x } ( f ) = \int _ { [ 0 , 1 ) } \left( \int _ { 0 } ^ { 1 } m _ { x } ( w ) \nu _ { \lambda } ( w ) d w \right) \xi ( d \lambda ) \leq \operatorname* { s u p } _ { \lambda \in [ 0 , 1 ) } \mathcal { U } _ { x } ( \nu _ { \lambda } ) .
$$

Taking the supremum over $f \in { \mathcal { F } } _ { \uparrow }$ gives

$$
\operatorname* { s u p } _ { f \in \mathcal { F } _ { \uparrow } } \mathcal { U } _ { x } ( f ) \leq \operatorname* { s u p } _ { \lambda \in [ 0 , 1 ) } \mathcal { U } _ { x } ( \nu _ { \lambda } ) .
$$

The reverse inequality is immediate because each $\nu _ { \lambda }$ belongs to $\mathcal { F } _ { \uparrow }$

Theorem A.3 is an oracle statement, so it does not identify a single global threshold for practice. Its role is structural: in rank space, support restriction is the right family for the first decision, namely which proxy ranks to keep.

Corollary A.4 (QRPO comparison in the same rank-space model). Under the assumptions of Theorem A.3, let

$$
q _ { \tau } ( w ) : = \frac { \tau e ^ { \tau w } } { e ^ { \tau } - 1 } , \qquad \tau > 0 ,
$$

with $q _ { 0 } ( w ) \equiv 1$ . Then

$$
\operatorname* { s u p } _ { \tau \geq 0 } \mathcal { U } _ { x } ( q _ { \tau } ) \leq \operatorname* { s u p } _ { \lambda \in [ 0 , 1 ) } \mathcal { U } _ { x } ( \nu _ { \lambda } ) .
$$

Proof. Each $q _ { \tau }$ is nondecreasing on [0, 1], so $\{ q _ { \tau } : \tau \geq 0 \} \subseteq { \mathcal { F } } _ { \uparrow }$ . The claim follows immediately from Theorem A.3. □

Thus, within the same oracle rank-space model, the best hard truncation rule weakly dominates the best QRPO-style full-support monotone tilt.

## A.2.2 Within-tail reweighting after truncation

We now hold the truncation level fixed and show that, when oracle value remains positively aligned with proxy rank inside the retained tail, uniform weighting over the retained interval is locally suboptimal, so a finite within-tail tilt can improve over pure truncation at the same threshold.

For a fixed threshold $\lambda \in ( 0 , 1 )$ and parameter $\beta \in ( 0 , \infty )$ , define the truncated-soft family

$$
f _ { \lambda , \beta } ( w ) : = \frac { 1 } { Z _ { \lambda } ( \beta ) } \left( \frac { w - \lambda } { 1 - w + \lambda } \right) ^ { 1 / \beta } \mathbb { 1 } \{ w \in ( \lambda , 1 ) \} ,
$$

where

$$
Z _ { \lambda } ( \beta ) : = \int _ { \lambda } ^ { 1 } \left( \frac { w - \lambda } { 1 - w + \lambda } \right) ^ { 1 / \beta } d w .
$$

This is the rank-space form induced by the shifted-truncated odds-ratio tilt at threshold λ. We use the convention $f _ { \lambda , \infty } = \nu _ { \lambda ; }$ , so $\beta = \infty$ recovers hard truncation, while finite $\beta$ tilts mass toward larger retained proxy ranks without changing the support. Define

$$
\mathcal { U } _ { x } ( \lambda , \beta ) : = \int _ { \lambda } ^ { 1 } { m _ { x } ( w ) f _ { \lambda , \beta } ( w ) d w } , \qquad \mathcal { U } _ { x } ( \lambda , \infty ) : = \mathcal { U } _ { x } ( \nu _ { \lambda } ) .
$$

Proposition A.5 (Local criterion for improving over pure truncation). Assume the law $o f r ( x , Y )$ under $Y \sim \pi _ { \mathrm { r e f } } ( \cdot \mid x )$ is continuous, and fix $\lambda \in ( 0 , 1 )$ . Let

$$
h _ { \lambda } ( w ) : = \log \left( \frac { w - \lambda } { 1 - w + \lambda } \right) , \qquad w \in ( \lambda , 1 ) .
$$

Then

$$
\frac { \partial } { \partial ( \beta ^ { - 1 } ) } \mathcal { U } _ { x } ( \lambda , \beta ) \bigg | _ { \beta = \infty } = \mathrm { C o v } _ { W _ { r } \sim \mathrm { U n i f } [ \lambda , 1 ] } \big ( m _ { x } ( W _ { r } ) , h _ { \lambda } ( W _ { r } ) \big ) .
$$

In particular, if this covariance is positive, then there exists a finite $\beta$ such that

$$
\mathcal { U } _ { x } ( \lambda , \beta ) > \mathcal { U } _ { x } ( \lambda , \infty ) = \mathcal { U } _ { x } ( \nu _ { \lambda } ) .
$$

Proof. Let $f _ { \lambda , \infty } = \nu _ { \lambda }$ denote the uniform density on [λ, 1]. Then

$$
f _ { \lambda , \beta } ( w ) = \frac { e ^ { h _ { \lambda } ( w ) / \beta } f _ { \lambda , \infty } ( w ) } { \mathbb { E } _ { W _ { r } \sim \mathrm { U n i f } [ \lambda , 1 ] } \left[ e ^ { h _ { \lambda } ( W _ { r } ) / \beta } \right] } .
$$

Therefore,

$$
\mathcal { U } _ { x } ( \lambda , \beta ) = \frac { \mathbb { E } _ { W _ { r } \sim \mathrm { U n i f } [ \lambda , 1 ] } \left[ m _ { x } ( W _ { r } ) e ^ { h _ { \lambda } ( W _ { r } ) / \beta } \right] } { \mathbb { E } _ { W _ { r } \sim \mathrm { U n i f } [ \lambda , 1 ] } \left[ e ^ { h _ { \lambda } ( W _ { r } ) / \beta } \right] } .
$$

Diferentiating with respect to $\beta ^ { - 1 }$ at $\beta = \infty$ gives

$$
\frac { \partial } { \partial ( \beta ^ { - 1 } ) } \mathcal { U } _ { x } ( \lambda , \beta ) \bigg | _ { \beta = \infty } = \mathbb { E } [ m _ { x } ( W _ { r } ) h _ { \lambda } ( W _ { r } ) ] - \mathbb { E } [ m _ { x } ( W _ { r } ) ] \mathbb { E } [ h _ { \lambda } ( W _ { r } ) ] ,
$$

where all expectations are under $W _ { r } \sim \mathrm { U n i f } [ \lambda , 1 ]$ . The diferentiation may be passed under the expectation because $h _ { \lambda }$ is integrable on (λ, 1) under the uniform law. This is exactly

$$
\mathrm { C o v } _ { W _ { r } \sim \mathrm { U n i f } [ \lambda , 1 ] } \big ( m _ { x } ( W _ { r } ) , h _ { \lambda } ( W _ { r } ) \big ) .
$$

If this covariance is positive, then the derivative at $\beta = \infty$ in the $\beta ^ { - 1 }$ direction is positive, so by continuity there exists a finite $\beta$ such that

$$
\mathcal { U } _ { x } ( \lambda , \beta ) > \mathcal { U } _ { x } ( \lambda , \infty ) .
$$

This is a fixed-λ refinement statement, not a global dominance claim. It does not say that finite-β reweighting beats the best threshold when λ is optimized freely. Instead, once a global quality floor has been chosen, the covariance criterion identifies when a softer within-tail tilt extracts additional oracle value from the retained tail.

## A.2.3 Quadratic illustration underlying Figure 3

This subsection gives a minimal analytic example that separates the two efects shown in Figure 3: changing the retained support and reweighting points within a fixed retained tail. Consider the stylized oracle-utility shape

$$
m _ { s } ( w ) : = 1 - ( w - s ) ^ { 2 } , \qquad s \in ( 0 , 1 ) ,
$$

where s is the proxy-rank location at which oracle value is maximized. We use $m _ { s }$ only as an unnormalized analytic shape. It is not itself a realizable conditional oracle win-rate profile, since

$$
\int _ { 0 } ^ { 1 } m _ { s } ( w ) d w = \frac { 2 } { 3 } + s - s ^ { 2 } ,
$$

which is generally not $1 / 2$ . A normalized profile with the same shape is

$$
\bar { m } _ { s } ( w ) : = \frac 1 2 + \frac 1 2 \left( m _ { s } ( w ) - \left( \frac 2 3 + s - s ^ { 2 } \right) \right) .
$$

Then $\begin{array} { r } { \int _ { 0 } ^ { 1 } \bar { m } _ { s } ( w ) d w = 1 / 2 } \end{array}$ , and $\bar { m } _ { s } ( w ) \in [ 1 / 6 , 2 / 3 ]$ for all $s , w \in [ 0 , 1 ]$ . For any density f on $[ 0 , 1 ]$

$$
\bar { \mathcal { U } } _ { s } ( f ) : = \int _ { 0 } ^ { 1 } \bar { m } _ { s } ( w ) f ( w ) d w = \frac { 1 } { 2 } + \frac { 1 } { 2 } \left( \mathcal { U } _ { s } ( f ) - \left( \frac { 2 } { 3 } + s - s ^ { 2 } \right) \right) ,
$$

where

$$
\mathcal { U } _ { s } ( f ) : = \int _ { 0 } ^ { 1 } m _ { s } ( w ) f ( w ) d w .
$$

Thus, for each fixed $s , \bar { \mathcal { U } } _ { s }$ is a positive afine transformation of $\mathcal { U } _ { s }$ . All maximizers, strict comparisons, crossing locations, and the threshold $s _ { \beta }$ below are unchanged by this normalization. We therefore use the shorter unnormalized form $m _ { s }$ in the algebra.

For the QRPO-style retained-tail family, write

$$
\mathcal { U } _ { s } ( \lambda , \beta ) : = \int _ { \lambda } ^ { 1 } \boldsymbol { m } _ { s } ( w ) f _ { \lambda , \beta } ( w ) d w , \qquad \mathcal { U } _ { s } ( \lambda , \infty ) : = \mathcal { U } _ { s } ( \nu _ { \lambda } ) .
$$

The first proposition isolates the value of adaptive support restriction by comparing the best hard truncation rule with the best full-support QRPO tilt. The second proposition fixes $\lambda = 1 / 2$ , matching Figure $^ { 3 , }$ and asks when finite within-tail reweighting improves over pure truncation at that same threshold.

Proposition A.6 (Adaptive truncation beats best QRPO under the quadratic profile). For every $s > 1 / 2$ the hard truncation threshold

$$
\lambda ^ { \star } ( s ) : = \frac { 3 s - 1 } { 2 }
$$

satisfies

$$
\mathcal { U } _ { s } ( \nu _ { \lambda ^ { \star } ( s ) } ) = 1 - \frac { ( 1 - s ) ^ { 2 } } { 4 } > \operatorname* { s u p } _ { \tau \geq 0 } \mathcal { U } _ { s } ( q _ { \tau } ) .
$$

Proof. Since $m _ { s } ( w ) = 1 - ( w - s ) ^ { 2 }$ , maximizing $\mathcal { U } _ { s } ( f )$ is equivalent to minimizing

$$
R ( f ; s ) : = \mathbb { E } _ { W _ { r } \sim f } \big [ ( W _ { r } - s ) ^ { 2 } \big ] .
$$

For the top-interval uniform law $\nu _ { \lambda }$ , we have

$$
\mathbb { E } _ { \nu _ { \lambda } } [ W _ { r } ] = \frac { 1 + \lambda } { 2 } , \qquad \mathrm { V a r } _ { \nu _ { \lambda } } ( W _ { r } ) = \frac { ( 1 - \lambda ) ^ { 2 } } { 1 2 } .
$$

Therefore

$$
R ( \nu _ { \lambda } ; s ) = \left( \frac { 1 + \lambda } 2 - s \right) ^ { 2 } + \frac { ( 1 - \lambda ) ^ { 2 } } { 1 2 } .
$$

Diferentiating with respect to λ gives

$$
\frac { d } { d \lambda } R ( \nu _ { \lambda } ; s ) = \frac { 2 \lambda } { 3 } + \frac { 1 } { 3 } - s .
$$

Thus the unique minimizer over $\lambda \in [ 0 , 1 ]$ is

$$
\lambda ^ { \star } ( s ) = \frac { 3 s - 1 } { 2 } ,
$$

which lies in (0, 1) for $s \in ( 1 / 2 , 1 )$ . Substituting this value gives

$$
R ( \nu _ { \lambda ^ { \star } ( s ) } ; s ) = \frac { ( 1 - s ) ^ { 2 } } { 4 } , \qquad \mathcal { U } _ { s } ( \nu _ { \lambda ^ { \star } ( s ) } ) = 1 - \frac { ( 1 - s ) ^ { 2 } } { 4 } .
$$

By Lemma $\mathrm { A . 2 , }$ every $f \in { \mathcal { F } } _ { \uparrow }$ is a mixture of the top-interval uniform laws $\nu _ { \lambda }$ . Since $\lambda \mapsto \mathcal { U } _ { s } ( \nu _ { \lambda } )$ has the unique maximizer $\lambda ^ { \star } ( s )$ , the unique maximizer over $\mathcal { F } _ { \uparrow }$ is $\nu _ { \lambda ^ { \star } ( s ) }$ . Hence

$$
\operatorname* { s u p } _ { \tau \geq 0 } \mathcal { U } _ { s } ( q _ { \tau } ) \leq \mathcal { U } _ { s } ( \nu _ { \lambda ^ { \star } ( s ) } ) .
$$

It remains only to rule out equality. No finite-τ QRPO density equals $\nu _ { \lambda ^ { \star } ( s ) } \colon q _ { 0 }$ is uniform on $[ 0 , 1 ]$ , while $q _ { \tau }$ is strictly positive on all of [0, 1] for every $\tau > 0$ , whereas $\nu _ { \lambda ^ { \star } ( s ) }$ has zero density below $\lambda ^ { \star } ( s ) > 0$ . Moreover, since $m _ { s }$ is bounded and continuous and $q _ { \tau }$ converges weakly to a point mass at $w = 1$ as $\tau  \infty .$

$$
\operatorname* { l i m } _ { \tau  \infty } \mathcal { U } _ { s } ( q _ { \tau } ) = m _ { s } ( 1 ) = 1 - ( 1 - s ) ^ { 2 } .
$$

This limiting value is strictly smaller than

$$
1 - \frac { ( 1 - s ) ^ { 2 } } { 4 } = \mathcal { U } _ { s } ( \nu _ { \lambda ^ { \star } ( s ) } ) .
$$

Therefore neither a finite QRPO tilt nor its limiting point mass at $w = 1$ attains the hard-truncation optimum, and the desired strict inequality follows. □

This comparison isolates the gain from support restriction itself. Even in the quadratic example, a full-support QRPO tilt cannot recover the oracle value achieved by the best hard truncation rule.

Proposition $\mathbf { A . 7 \ ( A t \ ) } = 1 / 2$ , finite $\beta$ helps above an explicit threshold). Define

$$
s _ { \beta } : = { \frac { { \frac { 1 1 } { 1 2 } } - \log 2 } { 1 - \log 2 } } \approx 0 . 7 2 8 4 .
$$

Then

$$
\frac { \partial } { \partial ( \beta ^ { - 1 } ) } \mathcal { U } _ { s } ( 1 / 2 , \beta ) \bigg | _ { \beta = \infty } = ( 1 - \log 2 ) s - \bigg ( \frac { 1 1 } { 1 2 } - \log 2 \bigg ) .
$$

In particular, $i f s > s _ { \beta }$ , then there exists a finite $\beta$ such that

$$
\mathcal { U } _ { s } ( 1 / 2 , \beta ) > \mathcal { U } _ { s } ( 1 / 2 , \infty ) = \mathcal { U } _ { s } ( \nu _ { 1 / 2 } ) .
$$

Proof. We instantiate Proposition A.5 at the fixed threshold $\lambda = 1 / 2$ . On the retained interval $[ 1 / 2 , 1 ]$ , the local log-odds tilt is

$$
h ( w ) : = \log \left( \frac { w - \frac { 1 } { 2 } } { \frac { 3 } { 2 } - w } \right) , \qquad w \in ( 1 / 2 , 1 ) .
$$

Proposition A.5 gives

$$
\frac { \partial } { \partial ( \beta ^ { - 1 } ) } \mathcal { U } _ { s } ( 1 / 2 , \beta ) \bigg | _ { \beta = \infty } = \mathrm { C o v } _ { W _ { r } \sim \mathrm { U n i f } [ 1 / 2 , 1 ] } \big ( m _ { s } ( W _ { r } ) , h ( W _ { r } ) \big ) .
$$

Since $m _ { s } ( W _ { r } ) = 1 - ( W _ { r } - s ) ^ { 2 }$ , this is

$$
\frac \partial { \partial ( \beta ^ { - 1 } ) } \mathcal { U } _ { s } ( 1 / 2 , \beta ) \bigg | _ { \beta = \infty } = - \operatorname { C o v } _ { W _ { r } \sim \mathrm { U n i f } [ 1 / 2 , 1 ] } \bigl ( ( W _ { r } - s ) ^ { 2 } , h ( W _ { r } ) \bigr ) .
$$

The required covariance is explicit. Under $W _ { r } \sim \mathrm { U n i f } [ 1 / 2 , 1 ]$ 」，

$$
\mathbb { E } [ h ( W _ { r } ) ] = 2 \int _ { 1 / 2 } ^ { 1 } \log \left( \frac { w - \frac { 1 } { 2 } } { \frac { 3 } { 2 } - w } \right) d w = - \log 4 ,
$$

and

$$
\mathbb { E } [ ( W _ { r } - s ) ^ { 2 } ] = \left( s - { \frac { 3 } { 4 } } \right) ^ { 2 } + { \frac { 1 } { 4 8 } } = s ^ { 2 } - { \frac { 3 } { 2 } } s + { \frac { 7 } { 1 2 } } .
$$

A direct integration also gives

$$
\mathbb { E } [ ( W _ { r } - s ) ^ { 2 } h ( W _ { r } ) ] = \frac { 1 1 } { 1 2 } - s + \left( - 2 s ^ { 2 } + 4 s - \frac { 1 3 } { 6 } \right) \log 2 .
$$

Subtracting the product of the first two moments yields

$$
\operatorname { C o v } \bigl ( ( W _ { r } - s ) ^ { 2 } , h ( W _ { r } ) \bigr ) = \frac { 1 1 } { 1 2 } - \log 2 - ( 1 - \log 2 ) s .
$$

Therefore

$$
\frac { \partial } { \partial ( \beta ^ { - 1 } ) } \mathcal { U } _ { s } ( 1 / 2 , \beta ) \bigg | _ { \beta = \infty } = ( 1 - \log 2 ) s - \bigg ( \frac { 1 1 } { 1 2 } - \log 2 \bigg ) .
$$

The threshold $s _ { \beta }$ is the unique zero of this afine function. Hence, for every $s > s _ { \beta }$ , the derivative in the $\beta ^ { - 1 }$ direction is positive at $\beta = \infty ,$ , so for suficiently small positive $\beta ^ { - 1 }$ , equivalently for suficiently large finite $\beta _ { ; }$ we have

$$
\mathcal { U } _ { s } ( 1 / 2 , \beta ) > \mathcal { U } _ { s } ( 1 / 2 , \infty ) .
$$

![](images/aabda9a0dafdffb04a0e220c25a554a3142133b9bf914dd449f440b270155a24.jpg)  
Figure 5: Optimal finite sharpness in the fixed-threshold quadratic illustration. For $\lambda = 1 / 2$ and $m _ { s } ( w ) =$ $1 - ( w - s ) ^ { 2 }$ , we numerically maximize $\mathcal { U } _ { s } ( 1 / 2 , \beta )$ over finite $\beta$ for each oracle peak location $s > s _ { \beta }$ . The dashed vertical line marks $s _ { \beta } \approx 0 . 7 2 8 4$ , below which pure truncation is locally optimal and the optimizer is $\beta = \infty$ . Just above $s _ { \beta } .$ , the optimal finite tilt is arbitrarily weak, so $\beta ^ { \star } ( s )$ is large; as s approaches one, the optimal policy concentrates more strongly near the top proxy ranks, and $\beta ^ { \star } ( s )$ decreases.

The proposition identifies the point at which pure truncation becomes locally suboptimal, but it does not assign a single universal value of $\beta .$ The optimal value depends on the oracle peak location s. To compute it, write $\alpha = 1 / \beta$ and view $\mathcal { U } _ { s } ( 1 / 2 , \beta )$ as a one-dimensional function of $\alpha \geq 0$ . Diferentiating the tilted expectation gives

$$
\frac { \partial } { \partial \alpha } \mathcal { U } _ { s } ( 1 / 2 , \beta ) = \mathrm { C o v } _ { f _ { 1 / 2 , \beta } } \left( m _ { s } ( W ) , \log \left( \frac { W - \frac { 1 } { 2 } } { \frac { 3 } { 2 } - W } \right) \right) .
$$

Thus an interior optimum $\beta ^ { \star } ( s )$ is obtained by solving this scalar covariance equation, while for $s \leq s _ { \beta }$ the optimum remains the boundary value $\beta = \infty$ , corresponding to pure truncation. Figure 5 plots the resulting finite optimizer for $s > s _ { \beta } \colon \beta ^ { \star } ( s )$ diverges as s approaches $s _ { \beta }$ from above, because the beneficial tilt is infinitesimal at the threshold, and decreases as s approaches one, where stronger concentration near the top proxy ranks becomes optimal. When the proxy is highly reliable $s \to 1$ and thus $\beta ^ { \star } \to 0$

## B Additional experiments and results

## B.1 Raw rewards

Tables 4 and 5 report the corresponding raw reward scores and generated lengths. On UltraFeedback, TUP remains strongest under both independent Skywork reward models, including on AlpacaEval. On Magpie Air, TUP remains competitive with the strongest baselines and improves over QRPO and BoNBoN under the independent Skywork evaluators. These raw results should be read together with the LC tables because several methods, including TUP and QRPO, tend to produce longer responses.

Figures 6 and 7 extend the motivating observation in Figure 2 beyond the UltraFeedback/Skywork-v2-Llama setting. Across the additional dataset–evaluator combinations, ArmoRM-defined bottom regions are more consistently recovered by the independent Skywork evaluator than ArmoRM-defined top regions. The corresponding λ curves show the same truncation tradeof: increasing λ reduces the chance of retaining completions that the independent evaluator ranks poorly, but also increases the chance of discarding completions it ranks highly. This supports using moderate truncation rather than an aggressive threshold.

We also estimate the efective sharpness learned by the trained TUP checkpoint by reversing the construction used during training. During training, the truncated win-rate label $w _ { \lambda }$ determines the target policy-reference log-ratio through the configured sharpness $\beta .$ After training, we instead observe the checkpoint log-ratio

Table 4: Raw reward for Llama 8B Tülu 3 SFT on UltraFeedback eval split (UF) and AlpacaEval (AE), with reporting token length.
<table><tr><td rowspan="2">Method</td><td colspan="2">ArmoRM</td><td colspan="2">Skywork-v2-Llama</td><td colspan="2"> $\mathrm { S k y w o r k - v 2 - Q w e n }$ </td><td>UF</td></tr><tr><td>UF</td><td>AE</td><td>UF</td><td>AE</td><td>UF</td><td>AE</td><td>Len</td></tr><tr><td>Initial</td><td> $0 . 1 3 0 0 { \scriptstyle \pm 0 . 0 0 0 8 }$ </td><td>0.1370±0.0003</td><td> $9 . 1 5 { \pm } 0 . 1 8 $ </td><td> $1 0 . 7 8 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $3 . 7 8 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $4 . 9 7 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $3 8 9 . 1 { \scriptstyle \pm 8 . 0 }$ </td></tr><tr><td>DPO</td><td> $0 . 1 4 9 7 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $0 . 1 6 2 0 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $1 6 . 2 7 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $2 1 . 3 6 { \pm } 0 . 1 8$ </td><td> $8 . 0 0 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $1 0 . 8 7 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $5 0 1 . 5 { \pm } 2 . 3 $ </td></tr><tr><td>REBEL</td><td> $0 . 1 4 8 9 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $0 . 1 6 2 0 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $1 6 . 3 5 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $2 2 . 1 5 { \pm } 0 . 1 6$ </td><td> $8 . 0 0 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $1 1 . 2 0 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $5 1 6 . 1 { \scriptstyle \pm 3 . 6 }$ </td></tr><tr><td>REBEL (rand)</td><td> $0 . 1 4 3 6 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $0 . 1 5 6 8 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $1 4 . 2 8 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $2 0 . 0 5 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $6 . 8 1 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $1 0 . 1 8 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $4 7 3 . 9 { \pm } 5 . 2 $ </td></tr><tr><td>QRPO</td><td> $0 . 1 5 1 7 { \scriptstyle \pm 0 . 0 0 0 7 }$ </td><td> $0 . 1 6 6 6 { \scriptstyle \pm 0 . 0 0 0 1 }$ </td><td> $1 7 . 4 1 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $2 1 . 0 8 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $8 . 2 4 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $1 0 . 3 1 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $7 3 0 . 1 { \pm } 4 . 8 $ </td></tr><tr><td>QRPO (rand)</td><td> $0 . 1 4 6 5 { \scriptstyle \pm 0 . 0 0 0 1 }$ </td><td> $0 . 1 6 1 0 { \scriptstyle \pm 0 . 0 0 0 1 }$ </td><td> $1 5 . 8 3 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $2 1 . 8 7 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $7 . 4 5 { \pm } 0 . 0 6$ </td><td> $1 0 . 9 1 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $5 0 8 . 6 { \pm } 2 . 8 $ </td></tr><tr><td>BoNBoN</td><td> $0 . 1 4 9 8 { \scriptstyle \pm 0 . 0 0 0 5 }$ </td><td> $0 . 1 6 3 2 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $1 8 . 3 0 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $2 4 . 2 9 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $8 . 8 9 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $1 2 . 2 5 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $5 4 8 . 3 { \pm } 3 . 0 $ </td></tr><tr><td>TUP (ours)</td><td> $0 . 1 5 3 1 { \scriptstyle \pm 0 . 0 0 0 1 }$ </td><td> $0 . 1 6 7 4 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $\mathbf { 2 0 . 2 0 { \scriptstyle \pm 0 . 0 7 } }$ </td><td> $\mathbf { 2 6 . 8 8 { \scriptstyle \pm 0 . 0 9 } }$ </td><td> $\mathbf { 9 . 7 5 { \scriptstyle \pm 0 . 0 2 } }$ </td><td> $\mathbf { 1 3 . 3 4 { \scriptstyle \pm 0 . 0 2 } }$ </td><td> $6 4 5 . 5 { \pm } 2 . 7 \ $ </td></tr></table>

![](images/566e41f469f7f10083e113fd2bea93805fbc0708dd9e9586533080338fffd413.jpg)

![](images/92b4079b3499374911e05d123dea6ab685b1d96d2ea465bf6125111987e42978.jpg)

![](images/ebd300cd77c62731a7975da67bdde4c6ba49062381549ec7159b4534d8f48561.jpg)  
Figure 6: Additional cross-model agreement rates between ArmoRM and independent Skywork reward models. For each selected fraction $p ,$ the plots show the fraction of completions placed by ArmoRM in the top or bottom $p$ region that the Skywork evaluator also places in the corresponding region. Left: UltraFeedback with Skywork-v2-Qwen. Center: Magpie Air with Skywork-v2-Llama. Right: Magpie Air with Skywork-v2-Qwen.

![](images/26adc0419587e94b23110e928952c6115f109cd6f33f87dc6fe6d81b33f36e31.jpg)

![](images/fa501437f383c91f67008a6f674527eea81be53eb2394f707be4750693e77083.jpg)

![](images/3b870db0de730fe265d22c8b6d0a6107faa844917710f6d503225c06996bde17.jpg)  
Figure 7: Additional cost–benefit curves for λ-truncation. Each plot compares the probability that truncation discards a completion ranked in the top 25% by the independent Skywork evaluator with the probability that it retains a completion ranked in the bottom 25% by that evaluator. Left: UltraFeedback with Skywork-v2- Qwen. Center: Magpie Air with Skywork-v2-Llama. Right: Magpie Air with Skywork-v2-Qwen.

Table 5: Raw reward for Llama 8B Tülu 3 SFT on Magpie Air eval split (MA) and AlpacaEval (AE), with reporting token length.
<table><tr><td></td><td colspan="2">ArmoRM</td><td colspan="2">Skywork-v2-Llama</td><td colspan="2">Skywork-v2-Qwen</td><td>MA</td></tr><tr><td>Method</td><td>MA</td><td>AE</td><td>MA</td><td>AE</td><td>MA</td><td>AE</td><td>Len</td></tr><tr><td>Initial</td><td> $0 . 1 5 3 0 { \scriptstyle \pm 0 . 0 0 1 0 }$ </td><td> $0 . 1 3 7 0 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $1 0 . 0 1 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $1 0 . 7 8 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $4 . 9 9 2 0 . 2 9 $ </td><td> $4 . 9 7 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $3 3 1 . 6 { \pm } 2 . 4 $ </td></tr><tr><td>DPO</td><td> $0 . 1 8 0 0 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $0 . 1 5 8 1 { \scriptstyle \pm 0 . 0 0 0 6 }$ </td><td> $1 9 . 2 3 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $1 8 . 8 2 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $1 1 . 9 3 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $9 . 6 9 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $4 8 4 . 5 { \pm } 3 . 2 $ </td></tr><tr><td>REBEL</td><td> $0 . 1 7 9 5 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $0 . 1 5 9 1 { \scriptstyle \pm 0 . 0 0 0 7 }$ </td><td> $2 0 . 6 9 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $2 0 . 6 5 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $1 2 . 3 1 { \pm } 0 . 1 4$ </td><td> $1 0 . 3 7 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $5 4 3 . 2 { \pm } 4 . 5 $ </td></tr><tr><td>REBEL (random)</td><td> $0 . 1 7 7 8 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $0 . 1 5 6 5 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $1 7 . 6 0 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $1 7 . 8 1 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $1 0 . 8 8 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $9 . 1 4 { \pm } 0 . 0 8$ </td><td> $4 7 9 . 8 { \pm } 4 . 4 $ </td></tr><tr><td>QRPO</td><td> $0 . 1 7 7 2 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $0 . 1 5 7 8 { \scriptstyle \pm 0 . 0 0 0 0 }$ </td><td> $1 8 . 2 3 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $1 8 . 9 9 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $1 0 . 2 5 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $9 . 3 0 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $5 5 4 . 0 { \pm } 2 . 2 $ </td></tr><tr><td>QRPO (random)</td><td> $0 . 1 7 1 7 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $0 . 1 5 4 9 { \scriptstyle \pm 0 . 0 0 0 7 }$ </td><td> $1 6 . 4 6 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $1 8 . 2 0 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $9 . 2 8 { \pm } 0 . 1 1 $ </td><td> $9 . 0 3 { \pm } 0 . 1 5 $ </td><td> $4 7 3 . 2 { \pm } 0 . 6 $ </td></tr><tr><td>BoNBoN</td><td> $0 . 1 6 5 4 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $0 . 1 4 5 3 { \scriptstyle \pm 0 . 0 0 0 8 }$ </td><td> $1 4 . 8 0 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $1 4 . 6 9 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $8 . 3 0 { \pm } 0 . 0 6$ </td><td> $7 . 2 1 { \pm } 0 . 1 3$ </td><td> $4 1 6 . 2 { \pm } 1 . 6 $ </td></tr><tr><td>TUP (ours)</td><td> $0 . 1 7 9 1 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $0 . 1 5 8 3 { \scriptstyle \pm 0 . 0 0 0 6 }$ </td><td> $2 2 . 3 2 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $2 1 . 4 8 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $1 3 . 1 2 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $1 0 . 5 2 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $7 0 0 . 9 { \pm } 4 . 3 $ </td></tr></table>

Table 6: Raw reward for Mistral 7B SFT on Magpie Air eval split (MA) and AlpacaEval (AE), with reporting token length.
<table><tr><td rowspan="2">Method</td><td colspan="2">ArmoRM</td><td colspan="2"> $\mathrm { S k y w o r k - v 2 - L l a m a }$ </td><td colspan="2"> $\mathrm { S k y w o r k - v 2 - Q w e n }$ </td><td>MA</td></tr><tr><td>MA</td><td>AE</td><td>MA</td><td>AE</td><td>MA</td><td>AE</td><td>Len</td></tr><tr><td>Initial</td><td> $0 . 1 6 2 8 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $0 . 1 4 7 8 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $1 5 . 9 9 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $1 6 . 5 5 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $9 . 8 6 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $8 . 7 7 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $4 9 3 . 1 { \pm } 4 . 1 $ </td></tr><tr><td>DPO</td><td> $0 . 1 7 4 3 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $0 . 1 5 3 0 { \scriptstyle \pm 0 . 0 0 0 5 }$ </td><td> $2 2 . 7 1 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $2 0 . 6 4 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $1 6 . 7 5 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $1 1 . 8 4 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $7 5 0 . 1 { \scriptstyle \pm 2 . 7 }$ </td></tr><tr><td>REBEL</td><td> $0 . 1 5 9 3 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $0 . 1 4 2 4 { \scriptstyle \pm 0 . 0 0 1 4 }$ </td><td> $1 4 . 2 7 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $1 1 . 6 8 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $1 0 . 1 8 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $6 . 5 0 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $1 3 7 5 . 7 { \scriptstyle \pm 6 . 8 }$ </td></tr><tr><td>REBEL (random)</td><td> $0 . 1 7 1 7 { \scriptstyle \pm 0 . 0 0 0 1 }$ </td><td> $0 . 1 5 3 9 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $\mathbf { 2 3 . 2 9 } \pm \mathbf { 0 . 0 5 }$ </td><td> $\mathbf { 2 0 . 9 7 { \scriptstyle \pm 0 . 0 4 } }$ </td><td> $\mathbf { 1 7 . 1 9 2 0 . 1 0 }$ </td><td> $\mathbf { 1 2 . 0 8 { \scriptstyle \pm 0 . 0 1 } }$ </td><td> $8 6 1 . 0 { \scriptstyle \pm 6 . 5 }$ </td></tr><tr><td>QRPO</td><td> $0 . 1 7 1 0 { \scriptstyle \pm 0 . 0 0 0 6 }$ </td><td> $0 . 1 5 3 2 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $2 1 . 1 4 { \pm } 0 . 1 3$ </td><td> $1 9 . 8 0 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $1 4 . 2 8 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 0 . 8 6 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $6 2 9 . 1 { \pm } 2 . 2 $ </td></tr><tr><td>QRPO (random)</td><td> $0 . 1 5 2 2 { \scriptstyle \pm 0 . 0 0 0 5 }$ </td><td> $0 . 1 3 0 1 { \scriptstyle \pm 0 . 0 0 0 5 }$ </td><td> $1 8 . 3 9 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $1 5 . 2 9 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $1 2 . 9 3 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $8 . 2 7 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $1 1 5 5 . 9 { \pm } 6 . 2 $ </td></tr><tr><td>BoNBoN</td><td> $0 . 1 6 5 5 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $0 . 1 4 9 1 { \scriptstyle \pm 0 . 0 0 0 6 }$ </td><td> $1 7 . 3 3 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $1 7 . 4 2 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $1 1 . 0 7 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $9 . 3 2 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $5 1 6 . 4 { \pm } 1 . 7 $ </td></tr><tr><td>TUP mild</td><td> $0 . 1 7 5 7 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $0 . 1 5 3 7 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $2 2 . 3 8 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 9 . 9 4 { \pm } 0 . 1 0 $ </td><td> $1 6 . 0 2 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $1 1 . 0 9 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $8 0 8 . 2 { \scriptstyle \pm 1 0 . 1 }$ </td></tr><tr><td>TUP mid.</td><td> $\mathbf { 0 . 1 7 6 7 { \scriptstyle \pm 0 . 0 0 0 3 } }$ </td><td> $0 . 1 5 6 2 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $2 2 . 2 1 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $2 0 . 2 3 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $1 6 . 4 5 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $1 1 . 5 2 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $7 3 1 . 2 { \pm } 3 . 9 $ </td></tr><tr><td>TUP aggressive</td><td> $0 . 1 7 4 9 { \scriptstyle \pm 0 . 0 0 0 5 }$ </td><td> $\mathbf { 0 . 1 5 6 4 { \scriptstyle \pm 0 . 0 0 0 6 } }$ </td><td> $2 0 . 4 7 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $1 9 . 3 3 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $1 5 . 3 1 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $1 0 . 9 1 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $7 4 4 . 4 { \pm } 1 . 4 $ </td></tr></table>

$$
\Delta _ { \theta } ( y \mid x ) = \log \pi _ { \theta } ( y \mid x ) - \log \pi _ { \mathrm { r e f } } ( y \mid x ) ,
$$

and ask which value of $\beta$ best explains its dependence on the reward-derived truncated win-rate. On the test split, we construct $w _ { \lambda }$ by ranking each response reward against the 12 reference completions for the same prompt and applying the same threshold $\lambda = 0 . 5$ used in training. We then fit the linear relation between $\Delta _ { \theta } ( { \boldsymbol { y } } \mid { \boldsymbol { x } } )$ and $z = \log \mathrm { i t } ( w _ { \lambda } )$ on the finite, unclipped subset. This gives an estimate of $\beta _ { \mathrm { e f f } } \approx 0 . 0 1 4 7$ . Thus, although training used the nominal value $\beta = 0 . 0 1$ , the learned policy behaves on this diagnostic as if its efective sharpness is roughly 1.5 times larger.

## B.2 Length-matched rewards

While the LC-reward results in Tables 1 and 2 already account for response length at the metric level, we seek further evidence that verbosity is not the sole contributor to $\mathrm { T U P } { \mathrm { s } }$ strong performance. To this end, Table 7 compares TUP with each baseline on responses of similar length. TUP is preferred over all benchmark methods under ArmoRM and Skywork-Llama. The results under Skywork-Qwen are more mixed across methods, although $\mathrm { T U P } { \mathrm { s } }$ average win rate remains above 50% under all three reward models. Importantly, the matching criterion retains only a fraction of the response pairs and produces a diferent subset for each baseline. These results should therefore be viewed as complementary evidence rather than as an estimate of performance on the full test set. Together with the LC-reward results, they further suggest that $\mathrm { T U P } { \mathrm { s } }$ strong performance is not solely due to its longer responses.

Table 7: TUP (with λ = 0.5) win rates against each benchmark method on similar-length AlpacaEval responses for Llama 8B Tülu 3 SFT trained on Magpie Air. For each comparison, we retain response pairs whose lengths difer by at most 10%. Values above 50% favor TUP and are shown in bold. The final column gives the number of retained pairs out of 2,415.
<table><tr><td>Method</td><td>ArmoRM</td><td>Skywork-Llama</td><td>Skywork-Qwen</td><td>Prompts retained (of 2415)</td></tr><tr><td>DPO</td><td>53.1%</td><td>55.3%</td><td>49.1%</td><td>276</td></tr><tr><td>REBEL</td><td>50.9%</td><td>50.2%</td><td>45.8%</td><td>463</td></tr><tr><td>REBEL (random)</td><td>64.9%</td><td>63.2%</td><td>61.8%</td><td>238</td></tr><tr><td>QRPO</td><td>51.4%</td><td>56.6%</td><td>56.6%</td><td>415</td></tr><tr><td>QRPO (random)</td><td>54.2%</td><td>53.1%</td><td>46.7%</td><td>240</td></tr><tr><td>BoNBoN</td><td>71.0%</td><td>60.7%</td><td>56.5%</td><td>169</td></tr><tr><td>Average</td><td>57.6%</td><td>56.5%</td><td>52.7%</td><td>N/A</td></tr></table>

## B.3 Global vs. per-prompt truncation

To estimate the potential headroom of prompt-adaptive truncation over a fixed global threshold λ, we consider an idealized empirical setting in which Skywork-Llama serves as the “oracle” evaluator. For each UltraFeedback prompt i, let $( y _ { i } ^ { ( 1 ) } , y _ { i } ^ { ( 2 ) } , \dots , y _ { i } ^ { ( N ) } )$ denote the responses ranked by ArmoRM. We then:

1. Compute the ideal prompt-specific threshold

$$
\lambda _ { i } ^ { * } = \frac { \arg \operatorname* { m a x } _ { j } r _ { \mathrm { S k y w o r k } } ( y _ { i } ^ { ( j ) } ) - 1 } { N } ,
$$

which discards all responses ranked below the Skywork-best response by ArmoRM while retaining that response.

2. Select the single global threshold λ<sup>∗</sup> that maximizes the average Skywork-Llama win rate across all prompts.

To isolate the efect of the truncation threshold, we apply no within-tail reweighting. Under this idealized setting, the prompt-specific thresholds achieve an empirical Skywork-based win rate of 0.9325, compared with 0.8820 for the best fixed global threshold. Thus, although prompt-specific truncation ofers some additional headroom, the fixed global threshold captures most of the attainable win rate in this analysis.

## C Implementation details

## C.1 Codebase, models, and data

The experiments in Tables 1–6 use the released QRPO benchmark data rather than regenerating reference completions or reward annotations. These data contain prompt-specific reference-completion pools scored by ArmoRM.

All experiments are implemented using the oficial QRPO repository<sup>2</sup>, together with its released preprocessed reference datasets<sup>3</sup>. We use Llama 8B Tülu 3 SFT as the initial policy. On Mistral 7B we run a dedicated supervised fine-tuning (STF) on the original model (Mistral-7B-Instruct-v0.2), we closely followed QRPO code, using the prompt–chosen-response from Magpie-Air-DPO-100K-v0.1. After filtering sequences to at most 2,048 tokens, the training set contained 97,812 examples, with 1,992 held out for evaluation. We trained for one epoch in bfloat16 using an efective batch size of 128 across four GPUs, a learning rate of $5 \times 1 0 ^ { - 7 }$ 10% warm-up, and cosine decay, without LoRA or other parameter-eficient tuning methods. To reproduce, please refer to our or QRPO GitHub repository.

Table 8: RewardBench standings (regardless of public availability) for the reward and judge models used in our experiments, retrieved in April 2026. Global Rank denotes the model’s overall position on the RewardBench leaderboard, and Type Rank denotes its position within the corresponding model category. The asterisk marks gpt-4o; because AlpacaEval uses an API-hosted judge, the exact served model version may not match the RewardBench entry.
<table><tr><td>Model</td><td>Role in this work</td><td>Model type</td><td>Global Rank</td><td>Type Rank</td></tr><tr><td>ArmoRM-Llama3-8B-v0.1</td><td>Training proxy reward model</td><td>Classifier</td><td>#34</td><td>#26</td></tr><tr><td>Skywork-Reward-V2-Llama-3.1-8B</td><td>Evaluation reward model</td><td>Classifier</td><td>#1</td><td>#1</td></tr><tr><td>Skywork-Reward-V2-Qwen3-8B</td><td>Evaluation reward model</td><td>Classifier</td><td>#6</td><td>#3</td></tr><tr><td>gpt-4o*</td><td>GPT judge</td><td>Generative</td><td>#36</td><td>#10</td></tr></table>

The training reward model in the released data is RLHFlow/ArmoRM-Llama3-8B-v0.1. For transfer evaluation, we rescore generated completions with Skywork/Skywork-Reward-V2-Llama-3.1-8B and Skywork/Skywork-Reward-V2-Qwen3-8B.

For AlpacaEval, we report the win rate and length-controlled win rate against the default gpt4-turbo reference outputs, which correspond to gpt-4-1106-preview; gpt-4o is used only as the judge due to the deprecation of gpt-4-1106-preview.

The UltraFeedback setting contains 61,024 training prompts and a held-out set of 1,995 prompts, split into 998 checkpoint-selection prompts and 997 reward-reporting prompts. The Magpie Air setting contains 97,812 training prompts and a held-out set of 1,992 prompts, split into 996 checkpoint-selection prompts and 996 reward-reporting prompts. AlpacaEval contains 805 evaluation prompts. The QRPO benchmark reference pools contain 50 reference completions per prompt.

For DPO, REBEL, and QRPO, we use the best-performing non-SFT, of-policy configurations reported in Tables 12 and 14 of Matrenok et al. [26]. We use the offpolicy2best and offpolicy2random variants as reported in Tables 9 and 10. BoNBoN and TUP are not included in the original QRPO benchmark, so we perform a separate hyperparameter search for these methods.

## C.2 Reward models used for evaluation

Table 8 summarizes the external reward models used in our experiments. ArmoRM is used only as the ofline proxy reward model for training labels, whereas the Skywork-v2 models and gpt-4o are used only for evaluation. The RewardBench ranks are provided only as external context, since leaderboard positions can change as new models are added.

## C.3 TUP implementation

TUP uses the same precomputed completion pools and ArmoRM scores as the QRPO benchmark. For each prompt, we convert the ArmoRM scores in the pool into empirical in-pool win-rates, apply the shifted truncation $\hat { w } _ { \lambda , r } = \operatorname* { m a x } ( \hat { w } _ { r } - \lambda , 0 )$ , and train with the BCE objective in Algorithm 1. Relative to the QRPO pipeline, the implementation only changes the scalar training label, the known intercept, and the loss used. In the reported experiments the intercept uses the population normalizer β log $Z _ { \lambda , \beta } ;$ Appendix A.1 also gives the exact finite-pool alternative $Z _ { K , \lambda , \beta }$ for the empirical-rank convention used in Algorithm 1.

Numerical evaluation of the population normalizer. Although Proposition 3.1 writes $Z _ { \lambda , \beta }$ as an incomplete Beta function, the implementation evaluates the equivalent continuous integral directly with high-precision arithmetic. We do not use standard regularized incomplete-Beta routines such as scipy.special.betainc, since these APIs normalize by the complete Beta function and assume positive shape parameters, whereas here the second parameter is $1 - 1 / \beta < 0$ for the $\beta$ values used in our sweeps. We

Table 9: Training hyperparameters Llama 8B Tülu 3 SFT on UltraFeedback reported in Table 1, Table 2 and Table 4.
<table><tr><td>Method</td><td>Dataset variant</td><td> $\beta$ </td><td> $l r$ </td><td>Num ref</td><td>λ</td><td>Checkpoint</td></tr><tr><td>DPO</td><td>offpolicy2best</td><td>0.01</td><td>3e-07</td><td>6</td><td>-</td><td>300</td></tr><tr><td>REBEL</td><td>offpolicy2best</td><td>1e-06</td><td>3e-07</td><td>6</td><td>-</td><td>200</td></tr><tr><td>REBEL (rand.)</td><td>offpolicy2random</td><td>1e-06</td><td>1e-06</td><td>6</td><td>-</td><td>100</td></tr><tr><td>QRPO</td><td>offpolicy2best</td><td>3e-04</td><td>3e-07</td><td>3</td><td>=</td><td>100</td></tr><tr><td>QRPO (rand.)</td><td>offpolicy2random</td><td>3e-04</td><td>3e-07</td><td>3</td><td>–</td><td>100</td></tr><tr><td>BoNBoN</td><td>offpolicy2best</td><td>1e-03</td><td>1e-06</td><td>6</td><td>-</td><td>476</td></tr><tr><td>TUP mild</td><td>offpolicy2random</td><td>0.01</td><td>1e-07</td><td>6</td><td>0.2</td><td>400</td></tr><tr><td>TUP mid.</td><td>offpolicy2random</td><td>0.01</td><td>1e-07</td><td>6</td><td>0.5</td><td>400</td></tr><tr><td>TUP aggressive</td><td>offpolicy2random</td><td>0.01</td><td>1e-07</td><td>6</td><td>0.8</td><td>476</td></tr></table>

Table 10: Training hyperparameters Llama 8B Tülu 3 SFT on Magpie Air reported in Table 1, Table 2 and Table 5.
<table><tr><td>Method</td><td>Dataset variant</td><td> $\beta$ </td><td> $l r$ </td><td>Num ref</td><td>λ</td><td>Checkpoint</td></tr><tr><td>DPO</td><td>offpolicy2random</td><td>0.01</td><td>1e-06</td><td>6</td><td></td><td>480</td></tr><tr><td>REBEL</td><td>offpolicy2best</td><td>1e-06</td><td>1e-07</td><td>6</td><td></td><td>640</td></tr><tr><td>REBEL (rand.)</td><td>offpolicy2random</td><td>1e-04</td><td>1e-06</td><td>6</td><td></td><td>320</td></tr><tr><td>QRPO</td><td>offpolicy2best</td><td>1e-03</td><td>1e-07</td><td>1</td><td></td><td>320</td></tr><tr><td>QRPO (rand.)</td><td>offpolicy2random</td><td>1e-03</td><td>1e-07</td><td>1</td><td></td><td>320</td></tr><tr><td>BoNBoN</td><td>offpolicy2best</td><td>1e-03</td><td>1e-06</td><td>6</td><td>-</td><td>764</td></tr><tr><td>TUP mild</td><td>offpolicy2random</td><td>0.01</td><td>1e-07</td><td>6</td><td>0.2</td><td>320</td></tr><tr><td>TUP mid.</td><td>offpolicy2random</td><td>0.01</td><td>1e-07</td><td>6</td><td>0.5</td><td>320</td></tr><tr><td>TUP aggressive</td><td>offpolicy2random</td><td>0.01</td><td>1e-07</td><td>6</td><td>0.8</td><td>320</td></tr></table>

therefore compute

$$
Z _ { \lambda , \beta } = \int _ { 0 } ^ { 1 - \lambda } u ^ { 1 / \beta } ( 1 - u ) ^ { - 1 / \beta } d u
$$

using mpmath with mp.workdps(256). This avoids the underflow that can occur in standard double-precision quadrature for small $\beta$ and makes the BCE intercept $\beta$ log $Z _ { \lambda , \beta }$ reproducible.

## C.4 Hyperparameter search and model selection

For BoNBoN, we use only the IPO-BoN component of the original objective. This gives the closest functional comparison in our setting, which is ofline BoN-style distillation without an additional SFT loss. We tune only the learning rate:

$$
\mathrm { l r } \in \{ 1 \mathrm { e } - 6 , 3 \mathrm { e } - 7 , 1 \mathrm { e } - 7 \} .
$$

For TUP using Llama model on UltraFeedback and Magpie Air, we tune the learning rate, preference sharpness, and truncation threshold over

$$
\mathrm { l r } \in \{ 1 \mathrm { e } - 6 , 3 \mathrm { e } - 7 , 1 \mathrm { e } - 7 \} , \quad \beta \in \{ 3 \mathrm { e } - 3 , 1 \mathrm { e } - 2 , 3 \mathrm { e } - 2 \} , \quad \lambda \in \{ 0 . 2 , 0 . 5 , 0 . 8 \} .
$$

For Mistral 7B on Magpie Air, due to computational constraints, we ran the search on a narrower set of hyper parameters:

$$
\mathrm { l r } \in \{ 1 \mathrm { e } { - } 6 , 3 \mathrm { e } { - } 7 \} , \quad \beta \in \{ 1 \mathrm { e } { - } 2 , 3 \mathrm { e } { - } 2 \} , \quad \lambda \in \{ 0 . 2 , 0 . 5 , 0 . 8 \} .
$$

Checkpoint selection uses only the validation split. We select checkpoints by length-controlled reward under ArmoRM. For both model families, UltraFeedback checkpoints are saved and evaluated every 100 steps and Magpie Air checkpoints are saved and evaluated every 160 steps. For UltraFeedback, we only consider checkpoints with mean generated length below 650 tokens. For Magpie Air, we do not impose an analogous length threshold. Tables 9 and 10 report the selected hyperparameters and checkpoints for each method.

Table 11: Training hyperparameters Mistral 7B SFT on Magpie Air reported in Table 3 and Table 6.
<table><tr><td>Method</td><td>Dataset variant</td><td>β</td><td>lr</td><td>Num ref</td><td>λ</td><td>Checkpoint</td></tr><tr><td>DPO</td><td>offpolicy2random</td><td>0.03</td><td>3e-07</td><td>6</td><td></td><td>480</td></tr><tr><td>REBEL</td><td>offpolicy2best</td><td>1e-04</td><td>1e-07</td><td>6</td><td></td><td>480</td></tr><tr><td>REBEL (rand.)</td><td>offpolicy2random</td><td>1e-04</td><td>3e-07</td><td>6</td><td></td><td>320</td></tr><tr><td>QRPO</td><td>offpolicy2best</td><td>3e-03</td><td>1e-07</td><td>1</td><td></td><td>764</td></tr><tr><td>QRPO (rand.)</td><td>offpolicy2random</td><td>3e-03</td><td>1e-06</td><td>1</td><td></td><td>764</td></tr><tr><td>BoNBoN</td><td>offpolicy2best</td><td>1e-03</td><td>1e-07</td><td>6</td><td>1</td><td>480</td></tr><tr><td>TUP mild</td><td>offpolicy2random</td><td>0.01</td><td>1e-06</td><td>6</td><td>0.2</td><td>764</td></tr><tr><td>TUP mid.</td><td>offpolicy2random</td><td>0.01</td><td>1e-06</td><td>6</td><td>0.5</td><td>764</td></tr><tr><td>TUP aggressive</td><td>offpolicy2random</td><td>0.01</td><td>1e-06</td><td>6</td><td>0.8</td><td>764</td></tr></table>

## C.5 Training settings

Unless stated otherwise, training settings follow the QRPO benchmark configuration. Training uses bfloat16 precision with Accelerate and DeepSpeed ZeRO-1. We use a cosine learning-rate schedule with warmup ratio 0.1. Gradient clipping is efectively disabled in the Llama benchmark runs by setting max\_grad\_norm to 10<sup>8</sup>.

The per-device training batch size is 2, the per-device evaluation batch size is 4, and gradient accumulation is 16. With 4 GPUs, this gives an efective training batch size of 128.

The maximum prompt length, maximum completion length, and maximum total sequence length are all 2048. The released reference completions were generated with temperature 1.0 and top-p = 1.0. For online evaluation, we generate one completion per prompt using temperature 0.6, top-p = 0.9, and a maximum of 2048 new tokens.

## C.6 Evaluation and uncertainty

For reward-model evaluation, we follow the QRPO evaluation pipeline. For each generated completion, the evaluator computes both raw reward and length-controlled reward using the reference-completion pool for the same prompt. For UltraFeedback and Magpie Air, we use the released QRPO reference pools. For AlpacaEval, which is not part of the released QRPO reference data, we construct the reference pool by generating 16 completions per prompt from the base model using vLLM. To compute length-controlled metrics under a given reward model, we score both the generated completions and the corresponding prompt-specific reference pools with that reward model. The length-controlled score normalizes reward and length relative to the prompt-specific reference pool, then removes the estimated linear efect of length. Prompts with degenerate reference-length variance are masked by the evaluation code.

For reward-model metrics, we report uncertainty as the sample standard deviation across three generation seeds. For AlpacaEval, we replace the recently gpt-4-1106-preview judge used in earlier benchmark configurations with gpt-4o. This change may make the AlpacaEval scores not directly comparable to results computed with the older judge.

## C.7 Deviation from the QRPO implementation

The QRPO codebase constructs training examples using both chosen and rejected completions by default. For TUP, we instead use a single randomly selected response per prompt. This matches the scalar-label formulation of Algorithm 1, where each completion receives one shifted-truncated win-rate label. It also means that TUP receives less per-prompt training signal than methods that use both responses.

## C.8 Compute resources

The reported training runs were performed on a machine with 4 H200 GPUs and 64 CPU cores.

## C.9 Compliance and asset licenses

Table 12 summarizes the external code, model, API, and dataset assets used in the experiments. License and terms information was retrieved from the oficial repository, model cards, dataset cards, and provider documentation on April 30, 2026. We use these assets only for ofline training or evaluation as described in Appendix C, and we do not redistribute the underlying external model checkpoints, API-hosted judge, or source datasets as part of this submission.

Table 12: External assets used in the experiments and their reported licenses or terms. For the QRPO reference data, the table lists the released preprocessed datasets actually loaded by our experiments; these are derived reference-completion pools and rewards rather than newly collected data in this work.
<table><tr><td>Asset</td><td>Role in this work</td><td>Reported license or terms</td></tr><tr><td>QRPO reference codebase</td><td>Experimental framework and baseline implementation.</td><td>MIT license.</td></tr><tr><td>QRPO UltraFeedback reference datasets offpolicy2best, offpolicy2random</td><td>Preprocessed UltraFeedback reference- MIT license. completion pools and ArmoRM scores used for training and evaluation.</td><td></td></tr><tr><td>QRPO Magpie Air reference datasets offpolicy2best, offpolicy2random</td><td>Preprocessed Magpie Air reference- completion pools and ArmoRM scores used for training and evaluation.</td><td>MIT license for the QRPO pre- processed releases; the upstream Magpie Air dataset card does not declare a separate license field.</td></tr><tr><td>openbmb/UltraFeedback</td><td>Source instruction-tuning dataset underlying the UltraFeedback QRPO setting.</td><td>MIT license.</td></tr><tr><td>tatsu-lab/alpaca_eval</td><td>AlpacaEval prompt set used for GPT- CC-BY-NC-4.0 license. judge evaluation.</td><td></td></tr><tr><td>allenai/reward-bench</td><td>RewardBench leaderboard context for the reward and judge models in Table 8.</td><td>ODC-BY-1.0 license.</td></tr><tr><td>allenai/Llama-3.1-Tulu-3-8B-SFT</td><td>Initial policy for the reported Llama benchmark runs.</td><td>Llama 3.1 Community License.</td></tr><tr><td>mistralai/Mistral-7B-Instruct-vO.2</td><td>Initial policy for the reported Mistral benchmark runs (SFT on this model was ran separately).</td><td>Mistral Community License.</td></tr><tr><td>RLHFlow/ArmoRM-Llama3-8B-vO.1</td><td>Offline proxy reward model used in the released QRPO labels and for checkpoint selection.</td><td>Llama 3 Community License.</td></tr><tr><td>Skywork/Skywork-Reward-V2-Llama-3.1-8B</td><td>Independent reward model used for transfer evaluation.</td><td>Llama 3.1 Community License.</td></tr><tr><td>Skywork/Skywork-Reward-V2-Qwen3-8B</td><td>Independent reward model used for transfer evaluation.</td><td>Apache-2.0 license.</td></tr><tr><td>gpt-4o</td><td>API-hosted judge for AlpacaEval.</td><td>Proprietary OpenAI API model used under OpenAI service terms; no weights are redis- tributed.</td></tr></table>