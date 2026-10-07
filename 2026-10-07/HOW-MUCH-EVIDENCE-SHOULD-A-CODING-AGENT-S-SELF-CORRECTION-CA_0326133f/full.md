# HOW MUCH EVIDENCE SHOULD A CODING AGENT’S SELF-CORRECTION CARRY? ADAPTIVE DIRICHLET EVIDENCE FOR SELF-DISTILLATION

Yunbo Long<sup>1</sup>, Guangya Hao<sup>1</sup>, Yuhan Liu<sup>1</sup>, Yiting Duan<sup>2</sup> Longyan Tan<sup>1</sup>, Yunchen Long<sup>3</sup>, Hao Wu<sup>2</sup>

<sup>1</sup>University of Cambridge <sup>2</sup>Western Sydney University

<sup>3</sup>Guangdong University of Technology

## ABSTRACT

Execution feedback lets coding agents revise programs and learn from their own corrections. A correction’s learning weight should reflect both the transitions supported by its executions and the amount of evidence behind that support. We introduce Effective-Evidence Self-Distillation (EESD), which represents these quantities separately. Normalized execution relevance determines relative transition support and an effective pseudo-count mass; a Dirichlet posterior then produces an uncertainty-penalized weight for KL-anchored correction learning. Under a symmetric prior, changing mass preserves category ordering, and effective mass yields a supervised coefficient bounded by its matched fixed-mass counterpart. Across four model–domain history sweeps, increasing visible observations from one to eight reduces future-outcome NLL by 55.0–59.3%. At eight observations, effective mass achieves lower NLL than fixed mass in all four comparisons. In the primary matched DeepSeek/RunBugRun study, argmax predictions agree on all 3,000 examples, with the largest NLL gain under concentrated relevance. After one correction-learning round, DeepSeek/CodeARC all-tests Pass@1 increases from 15.0% to 20.4%, with a paired 95% source-bootstrap interval of [+2.8, +8.0] percentage points. The twelve-setting downstream evaluation establishes the model– domain scope of this update. These results show how separating evidence support from evidence mass changes probability estimation and correction learning in coding agents.

## 1 INTRODUCTION

Coding agents use language models to generate and revise programs from execution feedback, including compiler errors and unit-test outcomes. This supports self-correction in code generation and program repair. Self-Refine and Reflexion reuse model-generated feedback or reflection at inference time (Madaan et al., 2023; Shinn et al., 2023); Self-Debugging revises programs after observing execution behavior (Chen et al., 2024); and execution-guided repair, search, and verification systems use tests to select or refine candidate programs (Ni et al., 2023; Tang et al., 2024; Li et al., 2025b;a; Tao et al., 2026). These methods use feedback to improve the next candidate program.

Self-training also requires a decision about how strongly to learn from each trajectory. Correctness filtering, reward weighting, and confidence-aware training assess the quality of generated behavior (Jiang et al., 2025; Jang et al., 2025; Yang et al., 2025). Execution evidence supplies two further quantities: which before/after transitions the observations support and how much evidence supports those transitions. Representing both quantities makes the correction weight sensitive to evidence concentration even when relative support stays fixed.

Figure 1 illustrates the distinction through an off-by-one repair. Extending a Python range endpoint makes all four public executions pass, so relative support is $r = ( 1 , 0 , 0 , \bar { 0 } )$ under either relevance pattern. Uniform relevance assigns comparable weight to every execution and gives $E ( p ) = 4 ;$ concentrated relevance $p = ( 0 . 9 7 , 0 . 0 1 , 0 . 0 1 , 0 . 0 1 )$ gives $E ( p ) \approx 1 . 0 6 2$ . A fixed-mass control restores total mass to four in both cases. EESD instead lets relevance concentration change the supervised weight while retaining the same correction target. The analytical relevance vectors isolate the effect of concentration at fixed transition support.

![](images/e05f6cc1a224cba4f973db2efa33c185635dfcc300cce7c98b02a6e1e6700e8a.jpg)  
Figure 1: Direction and strength in a coding agent’s self-correction. (a) Fixing an off-by-one range error changes four public executions from fail to pass. (b) Aggregating execution relevance by transition gives $r = \bar { ( 1 , 0 , 0 , 0 ) }$ : all support favors FIX. (c) The same transitions under uniform and concentrated relevance give $E ( p ) = 4 . 0 0 0$ and 1.062; at fixed prior and utility settings, they yield different supervised weights. The p bars index executions, whereas the r bars index transition categories. The relevance vectors define an analytical example at fixed transition support.

The problem becomes more consequential when corrections are written back into model parameters. A mistaken inference-time correction affects one answer; an over-weighted training correction changes the policy that will generate future answers and future training data, a feedback pattern relevant to recursive self-improvement (RSI) and self-evolving agents (Yin et al., 2025; Ranaldi, 2026; Xiao et al., 2026; Li et al., 2026). We study the correction-to-update step within this feedback process: given a self-generated correction, how much should the next policy learn from its supporting execution evidence?

We introduce Effective-Evidence Self-Distillation (EESD), a posterior-based rule for weighting correction learning. Given nonnegative relevance scores s , EESD normalizes them into p. Relative support r describes what transition categories the observations favor, while $E ( p ) = \textstyle { \dot { ( } } \sum _ { i } p _ { i } ^ { 2 } ) ^ { - 1 }$ describes how much pseudo-count mass that support contributes. A Dirichlet posterior produces an uncertainty-penalized correction utility; its positive part scales correction cross-entropy while a KL term anchors the student to the entering policy. The upstream correction generator and relevance estimator remain modular: the contribution is how their execution record is converted into a learning coefficient. Our contributions are:

1. We formulate correction-level direction–strength separation: distinguishing relative execution support from the total evidence mass assigned to that support.

2. We introduce an effective-evidence Dirichlet estimator (EED), which converts relevance concentration into adaptive Dirichlet pseudo-count mass, and connect its posterior utility to KL-anchored LoRA self-distillation. We characterize count recovery, ordering invariance, and the resulting control of the supervised coefficient.

3. We evaluate the construction through visible-history scaling, a matched-mass mechanism study with fixed argmax, and one-round correction-based self-training followed by fresh code generation across multiple models and domains.

## 2 RELATED WORK

Table 1 positions EESD relative to representative self-correction and self-training approaches. The central distinction is between improving an answer, assessing its suitability for learning, and determining how much execution evidence supports the resulting update.

Table 1: Positioning by the role of feedback. ✓: explicit use; –: absent in the cited formulation; : varies within a grouped family. Generator updates refer to the code-generating policy. Adaptive training includes filtering, preferences, and reward weighting. The final three columns describe correction-level support–mass construction. Representative methods are discussed in Section 2.
<table><tr><td>Method</td><td>Inference revision / selection</td><td>Generator update</td><td>Execution feedback</td><td>Adaptive training</td><td>Transition support r</td><td>Effective mass E(p)</td><td>Support / mass separation</td></tr><tr><td>Self-Refine</td><td>√</td><td></td><td>一</td><td></td><td></td><td></td><td></td></tr><tr><td>Reflexion (coding)</td><td>√</td><td></td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>Self-Debugging</td><td>√</td><td></td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>Execution-guided verification / reranking</td><td>√</td><td></td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>Importance-weighted self-training</td><td></td><td>√</td><td></td><td>√</td><td></td><td></td><td></td></tr><tr><td>Confidence-aware self-training</td><td></td><td>√</td><td></td><td>√</td><td></td><td></td><td></td></tr><tr><td>Reward / process-weighted learning</td><td>O</td><td>o</td><td>√</td><td>0</td><td></td><td></td><td></td></tr><tr><td>EESD (ours)</td><td></td><td>√</td><td>√</td><td>5</td><td>√</td><td>√</td><td>√</td></tr></table>

Self-correction and execution-guided repair. Self-Refine iteratively revises outputs using modelgenerated critiques, and Reflexion reuses verbal feedback to guide later attempts (Madaan et al., 2023; Shinn et al., 2023). Reflexion’s coding setting and Self-Debugging use execution feedback to revise programs (Shinn et al., 2023; Chen et al., 2024). Execution-guided verification and reranking evaluate candidate programs with test outcomes (Ni et al., 2023; Li et al., 2025a). REx uses Bayesian beliefs to allocate repair-search effort, while ExecCritic couples testing with repair-agent learning (Tang et al., 2024; Tao et al., 2026). LEVER trains an auxiliary verifier for candidate selection (Ni et al., 2023). EESD operates on the correction record produced by such upstream procedures and determines its contribution to a generator update.

Learning from self-generated corrections. Correctness filtering, importance weighting, and confidence-aware self-training select or weight generated trajectories using outcome and model signals (Jiang et al., 2025; Jang et al., 2025). CodePRM learns a process evaluator from execution feedback, and iterative preference learning updates models from self-generated data (Li et al., 2025b; Yang et al., 2025). Self-evolving systems also reuse experience across updates or skill construction (Yin et al., 2025; Ranaldi, 2026; Xiao et al., 2026; Li et al., 2026). EESD adds a correction-level evidence representation: relative transition support and the pseudo-count mass assigned to that support jointly determine the training weight.

Evidential uncertainty and effective sample size. Dirichlet models separate categorical probabilities from distributional concentration and support evidential uncertainty modeling (Sensoy et al., 2018). The inverse squared concentration of normalized weights is an established effective-samplesize functional (Martino et al., 2017). EESD combines these tools at the level of an execution record. Relative transition support r allocates pseudo-counts across categories, effective mass E(p) sets their total, and posterior utility sets the supervised coefficient in KL-anchored self-distillation. This execution-to-learning construction links evidence concentration directly to the strength of a correction update.

## 3 EFFECTIVE-EVIDENCE SELF-DISTILLATION

EESD converts a coding agent’s execution feedback into a correction-specific training weight. The key is to separate relative transition support from total evidence mass, then use both to determine how strongly to learn from a correction.

![](images/7793a07252f662c193b1a9bd11fbbaf07814091733a183b9f0ca423d1bc799dd.jpg)  
Figure 2: Effective-Evidence Self-Distillation. The public correction record yields p and transition labels $z _ { i }$ Parallel support and mass summaries form a pseudo-count posterior and weight w. Teacher and student consume the same prompt and past correction tokens; the current target $a _ { i } ^ { 1 }$ enters CE only. Teacher probabilities supply the forward-KL reference. The weight w scales CE. The dashed coral arrow updates LoRA parameters only. Code and decoder blocks schematically represent the correction record and frozen model bases.

## 3.1 CORRECTION EVIDENCE

For a programming task x, a coding agent with policy $\pi _ { t }$ produces an initial program $a ^ { 0 }$ , and an upstream repair procedure returns a correction $a ^ { 1 }$ . Both programs are evaluated on the same n $\geq 1$ public executions $\{ q _ { i } \} _ { i = 1 } ^ { n } .$ , yielding before/after outcomes $\overline { { ( o _ { i } ^ { 0 } , o _ { i } ^ { 1 } ) } }$ . Let h denote this correction record. Each outcome pair defines a transition

$$
z _ { i } \in \{ \mathrm { F I X , R E G R E S S , P R E S E R V E , U N R E S O L V E D } \} ,\tag{1}
$$

corresponding to fail pass, pass fail, pass pass, and fail fail, respectively.

Each execution receives a relevance score $s _ { i } \geq 0$ , with $\textstyle \sum _ { i } s _ { i } > 0$ . Define

$$
p _ { i } = \frac { s _ { i } } { \sum _ { j } s _ { j } } ,\tag{2}
$$

$$
r _ { k } = \sum _ { i } p _ { i } \mathbb { I } [ z _ { i } = k ] .\tag{3}
$$

The four-vector r describes which transitions the observations support. Direction denotes the ordering of transition categories. Our implementation obtains $s _ { i }$ from a frozen lexical kernel comparing the code change with public execution descriptors; Appendix B gives the details.

## 3.2 FROM EFFECTIVE EVIDENCE TO CORRECTION UTILITY

Our fixed-mass control uses $c _ { k } ^ { \mathrm { f i x e d } } = n r _ { k }$ , preserving relative support while assigning every record of length n the same total mass. EED replaces this nominal mass with

$$
E ( p ) = \left( \sum _ { i } p _ { i } ^ { 2 } \right) ^ { - 1 } ,\tag{4}
$$

$$
\alpha _ { k } ^ { \mathrm { p o s t } } = \alpha + E ( p ) r _ { k } , \qquad \alpha > 0 ,\tag{5}
$$

and defines $\theta \mid h \sim \operatorname { D i r } ( \alpha ^ { \mathrm { p o s t } } )$ . Thus r allocates pseudo-counts across transitions, while $E ( p )$ sets their total mass. We have $1 \leq E ( p ) \leq n ;$ uniform relevance gives $E ( p ) p _ { i } = 1$ and recovers ordinary counts. The posterior is conditional on the relevance weights, and its pseudo-count mass summarizes their concentration.

To reward fixes and penalize regressions, set $v = ( 1 , - 1 , 0 , 0 )$ . Writing $\begin{array} { r } { \alpha _ { 0 } = \sum _ { k } \alpha _ { k } ^ { \mathrm { p o s t } } } \end{array}$ and $\mu _ { k } = \alpha _ { k } ^ { \mathrm { p o s t } } / \alpha _ { 0 }$ , the posterior utility moments are

$$
\bar { u } = \mathbb { E } [ v ^ { \top } \theta ] = \mu _ { \mathrm { F I X } } - \mu _ { \mathrm { R E G R E S S } } ,\tag{6}
$$

$$
\sigma _ { u } ^ { 2 } = \frac { \mu _ { \mathrm { F I X } } + \mu _ { \mathrm { R E G R E S S } } - \bar { u } ^ { 2 } } { \alpha _ { 0 } + 1 } .\tag{7}
$$

The correction weight is

$$
w ( h ) = \left[ \bar { u } - \lambda \sigma _ { u } \right] _ { + } , \qquad \lambda \geq 0 ,\tag{8}
$$

where $[ a ] _ { + } = \operatorname* { m a x } ( a , 0 )$ . The uncertainty penalty reduces the supervised contribution of corrections with less decisive posterior utility; nonpositive utility yields zero supervised weight. The score combines expected utility with a posterior standard-deviation penalty.

For the all-FIX example in Figure 1, $\alpha = 0 . 1$ and $\lambda = 0 . 5$ give weights 0.832 and 0.542 for uniform and concentrated relevance. The matched fixed-mass control assigns 0.832 to both. Appendix G.1 gives the calculation. The example holds correction targets and KL coefficients fixed.

## 3.3 KL-ANCHORED SELF-DISTILLATION

The student learns from $a ^ { 1 }$ while a frozen copy of $\pi _ { t }$ supplies a forward-KL anchor:

$$
\mathcal { L } _ { \mathrm { E E S D } } ( h ) = w ( h ) \mathcal { L } _ { \mathrm { C E } } ( h ) + \beta D _ { \mathrm { K L } } ^ { ( h ) } ( \pi _ { t } \| \pi _ { t + 1 } ) .\tag{9}
$$

Here $\mathcal { L } _ { \mathrm { C E } }$ is correction cross-entropy, and $D _ { \mathrm { K L } } ^ { ( h ) }$ is forward KL averaged over the same responsetoken prefixes. Only the supervised term is weighted by $w ( h )$ . We train LoRA parameters with the quantized base frozen; token-level definitions and fixed hyperparameters are in Appendix B.

Proposition 1 (Outcome ordering and correction weight). Fix r, a symmetric prior $\alpha > 0 , \lambda \ge 0$ and $v = ( 1 , - 1 , 0 , 0 )$ . Replace $\breve { E } ( p )$ by any mass $\tau > 0$ in Equations 5–8. The posterior mean satisfies

$$
\mu _ { k } ( \tau ) = \gamma ( \tau ) r _ { k } + \frac { 1 - \gamma ( \tau ) } { 4 } , \qquad \gamma ( \tau ) = \frac { \tau } { 4 \alpha + \tau } .\tag{10}
$$

Consequently, category ordering and the argmax set are invariant $t o \tau .$ . Moreover, the correction weight $w _ { \tau }$ is nondecreasing in $\tau ,$ so

$$
w _ { E ( p ) } \leq w _ { n } .\tag{11}
$$

At fixed support, effective mass attenuates the supervised coefficient relative to nominal mass while preserving category ordering. Its effect on log loss depends on the observed category. Appendix A gives the proofs and the outcome-dependent log-loss condition.

## 4 EXPERIMENTS

We evaluate three questions. RQ1 examines whether additional visible executions improve outcome prediction. RQ2 isolates how effective mass changes predictive probabilities at fixed relative support. RQ3 tests the complete correction-learning update through fresh program generation. Figures 3 and 4 organize these measurements, with full numerical reports in the appendix.

## 4.1 EXPERIMENTAL SETUP

Tasks and models. We use RunBugRun and CodeARC, together with APPS Replay and CodeContests Replay (Prenner & Robbes, 2023; Wei et al., 2025). RunBugRun contains executable buggy programs; CodeARC tests program synthesis from behavioral observations. The replay domains are controlled synthesis extensions evaluated on the recorded source populations and execution budgets. CodeARC uses the fixed-source execution protocol specified in Appendix B; the original benchmark uses interactive differential testing (Wei et al., 2025). The downstream matrix covers Qwen2.5-Coder-7B-Instruct, DeepSeek-Coder-6.7B-Instruct, and Gemma-3-4B-it, with 500 primary sources per model–domain cell.

Measurements and comparisons. The prediction studies measure future-outcome negative loglikelihood (NLL) and categorical accuracy. The history sweep compares fixed mass n with effective mass $E ( p )$ as visible history grows. A separate matched comparison holds relative support and predictor parameters fixed, changing only the pseudo-count mass. Downstream evaluation generates one fresh candidate per source after one update; all-tests Pass@1 requires passing all ten evaluator tests. It compares full EESD (eesd full) with the unchanged entering policy (no update). The paired effect, in percentage points, is

$$
\Delta _ { \mathrm { p p } } = 1 0 0 \big [ S ( \mathsf { e e s d \_ f u l 1 } ) - S ( \mathsf { n o \_ u p d a t e } ) \big ] ,\tag{12}
$$

where $S$ is the fraction of solved sources. Paired 95% intervals use 10,000 whole-source bootstrap draws and quantify source-level uncertainty conditional on the evaluated policies. Evidence scoring uses public executions; hidden outcomes are reserved for evaluation. Fresh generation produces a new completion for each recorded source. Protocol and replication details are in Appendix B.

Training and hardware. We adapt LoRA parameters on a frozen 4-bit NF4 base with bfloat16 computation. The correction scorer uses $\alpha = 0 . 1$ and $\lambda = 0 . 5 ;$ the forward-KL coefficient is $\beta = 0 . 0 3$ . Experiments use NVIDIA RTX A6000 hardware with 48 GB nominal memory per GPU. Evidence weights are computed from public records before student fitting, and the frozen entering policy supplies the KL reference. Appendix B specifies the relevance kernel, response-token objective, and optimization settings.

## 4.2 RQ1: DOES ADDITIONAL VISIBLE EVIDENCE IMPROVE PREDICTION?

We first ask whether the available executions carry useful predictive information. Figure 3 varies visible history size over $n \in \{ 1 , 2 , 4 , 8 \}$ for DeepSeek and Qwen on RunBugRun and CodeARC. Here n is the number of visible observations. The sweep measures the predictive value of a longer execution history.

Prediction improves with longer histories. EED NLL decreases at every measured step in all four model–domain cells. On RunBugRun, the endpoint changes are 0.4696 0.2114 for DeepSeek and 0.4737 0.2113 for Qwen. On CodeARC, they are 0.7103 0.2892 and $0 . 7 7 8 5  0 . 3 4 1 5$ Table 2 expresses these changes as relative NLL reductions of 55.0–59.3%. Categorical accuracy improves as well: by 2.5 and 1.9 percentage points on RunBugRun, and by 5.5 and 6.1 points on CodeARC, respectively. Additional history therefore benefits both the probability assigned to the observed outcome and the categorical prediction. RQ3 evaluates the resulting policy update.

Effective mass adds a distinct comparison. Fixed-mass prediction also benefits from additional observations. $\mathbf { A } \mathbf { { t } } n = 8$ , EED further reduces NLL relative to Fixed in all four cells, as summarized in Figure 4(a): 0.0046 and 0.0075 on RunBugRun, and 0.0113 and 0.0024 on CodeARC, for DeepSeek and Qwen respectively. All four values are absolute NLL differences. $\mathbf { A } \mathbf { t } n = 1$ , the two constructions coincide because $p _ { 1 } = 1$ and $E ( p ) = n = 1$ ; differences at shorter histories are generally small. Figure 3 and Appendix E retain the complete trajectories. The history sweep varies the available observations; RQ2 holds relative support fixed to isolate the mass intervention.

Qwen

Table 2: Endpoint summary of the EED history sweep. NLL reduction and accuracy change compare $n = 1$ with $n = 8 ; \Delta \mathrm { N L L } _ { 8 } = \mathrm { N L L } _ { \mathrm { F i x e d } } - \mathrm { N L L } _ { \mathrm { E E D } }$ at $n = 8 .$ The reductions are calculated from the reported rounded measurements.
<table><tr><td>Benchmark</td><td>Backbone</td><td>EED NLL 1→8</td><td>Reduction (%)</td><td>Acc. change (pp)</td><td> $\Delta \mathrm { N L L _ { 8 } } \uparrow$ </td></tr><tr><td>RunBugRun</td><td>DeepSeek 6.7B</td><td>0.4696→0.2114</td><td>55.0</td><td>+2.5</td><td>+0.0046</td></tr><tr><td>RunBugRun</td><td>Qwen2.5 7B</td><td> $0 . 4 7 3 7 { \scriptstyle  } 0 . 2 1 1 3$ </td><td>55.4</td><td>+1.9</td><td>+0.0075</td></tr><tr><td>CodeARC</td><td>DeepSeek 6.7B</td><td> $0 . 7 1 0 3 \substack {  } 0 . 2 8 9 2$ </td><td>59.3</td><td>+5.5</td><td>+0.0113</td></tr><tr><td>CodeARC</td><td>Qwen2.5 7B</td><td>0.7785→0.3415</td><td>56.1</td><td>+6.1</td><td>+0.0024</td></tr></table>

DeepSeek

![](images/ef768822f0aee5ee2a39f646371a7feaa8f447416b41e97925641d8bbf1ca805.jpg)

![](images/144b5af282ee0b0c2742189254b2f158100dbde8ad57d53b1a19dc4ede7bd11e.jpg)

$$
\times 1 0 ^ { - 3 }
$$

(c) RunBugRun  
![](images/5dc2f89a8470f6f1997b962d39fba310348f79736c5d535bf9197380ff388dce.jpg)

(d) CodeARC  
![](images/95696ec2c53e02712711a7067d2d37b83f8d3efb85f0cfcbf6043c796d6b37a9.jpg)  
Visible executions, n  
Figure 3: History size and effective evidence answer different questions. (a,b) EED future-outcome NLL: lower is better. (c,d) $\Delta \mathrm { N L L } = \mathrm { N L L } _ { \mathrm { F i x e d } } - \mathrm { N L L } _ { \mathrm { E E D } } ;$ higher is better, in units of $1 0 ^ { - 3 }$ . All panels use $n = 1 , 2 , 4 ,$ 8 visible observations. Differences are calculated from the four-decimal measurements in Table 8. The matched-parameter comparison is reported separately in Section 4.3.

## 4.3 RQ2: DOES EFFECTIVE EVIDENCE CHANGE STRENGTH WITHOUT CHANGING DIRECTION?

The primary matched comparison uses DeepSeek/RunBugRun with the same relative support and effective-arm hyperparameters for both predictors. Only total pseudo-count mass changes. Their categorical argmax predictions agree on all 3,000 future-outcome examples, while effective mass reduces aggregate NLL by 0.00609 with a 95% source-bootstrap interval of [0.00211, 0.01075]. Thus better probability estimates do not require changing the selected category. The ordering argument in Proposition 1 applies to any categorical predictor with a symmetric prior, as detailed in Appendix A.4; the four-transition utility is the separate training construction.

The largest gain occurs under concentrated relevance. Table 3 and Figure 4(b) divide the matched comparison into three effective-mass bins. The NLL reduction is 0.0159 for $1 \leq E ( p ) < 2$ 0.0077 for $2 \overset { \cdot } { \leq } E ( p ) < 3$ , and 0.0018 for $3 \leq E ( p ) \leq 4$ . The corresponding mean masses retain 37.8%, 63.0%, and 92.0% of the nominal count of four (Appendix G). In this comparison, the largest predictive benefit is associated with the largest reduction in pseudo-count mass. When support is already broadly distributed, effective and nominal mass are closer and the observed difference is smaller.

Changing mass adjusts shrinkage toward the symmetric prior at fixed observations and relative support. The favored category remains fixed while its assigned probability changes. The reported mass fractions quantify this pseudo-count intervention.

Table 3: Concentration-stratified matched-mass comparison. Primary DeepSeek/RunBugRun comparison. Gain is fixed-mass NLL minus EED NLL. Both predictors agree in argmax on all 3,000 examples.
<table><tr><td>Effective-mass bin</td><td>Examples</td><td>Mean  $E ( p )$ </td><td>NLL gain ↑</td></tr><tr><td>[1, 2)</td><td>568</td><td>1.511</td><td>+0.0159</td></tr><tr><td>[2, 3)</td><td>815</td><td>2.521</td><td>+0.0077</td></tr><tr><td>[3, 4]</td><td>1617</td><td>3.681</td><td>+0.0018</td></tr></table>

From probability estimates to learning. The matched study isolates the evidence estimator. For training, the posterior is further mapped to a correction-specific utility and supervised coefficient. Proposition 1 characterizes this mapping; RQ3 examines the complete resulting update. The other recorded mechanism repetitions and their intervals are provided in Appendix F, separately from the history sweep. The log-loss analysis in Appendix A.4 explains why a change in mass need not have the same benefit for every outcome.

## 4.4 RQ3: DOES CORRECTION LEARNING IMPROVE FRESH GENERATION?

We now move from outcome prediction to the coding agent’s code-generating policy. Each evaluated program is freshly sampled after learning. Figure 4(c) presents four selected model–domain contrasts against the unchanged policy; Appendix D reports the complete twelve-cell evaluation.

![](images/62613352086f337ebb7b1cea2c8018eea532ad9693f52c550bbd4944ead62ab3.jpg)

![](images/1afea6aa57a4711553a996bdb7bbd8d912887779e6bca9ad601ad78152d17f8b.jpg)

![](images/31dbb0870cad8540f50feacfc61698bd75d447f20abdba3fabd799992d8c883e.jpg)  
Figure 4: Prediction and fresh-generation performance. (a) Fixed-minus-EED NLL at n = 8 in the four history-sweep cells. (b) The matched DeepSeek/RunBugRun comparison by effective-mass bin; argmax agrees on all 3,000 examples. Both NLL panels use units of $\mathrm { \check { 1 0 } } ^ { - 3 }$ and positive values favor EED. (c) Four selected one-round contrasts against no update, with paired 95% source-bootstrap intervals and solved counts out of 500. The complete model–domain evaluation is in Table 7.

The clearest gain is on DeepSeek/CodeARC. All-tests Pass@1 rises from 75/500 (15.0%) to 102/500 (20.4%), a net increase of 27 solved sources. The paired improvement is +5.4 percentage points, with a 95% source-bootstrap interval of $[ + 2 . 8 , + 8 . { \dot { 0 } } ]$ . Each solved program passes all ten evaluator tests. The comparison measures the effect of the complete correction-learning update.

Further gains appear on RunBugRun and CodeContests Replay. DeepSeek increases from 184/500 to 198/500 on RunBugRun (+2.8 points), and Qwen increases from 228/500 to 240/500 (+2.4 points). DeepSeek also increases from 18/500 to 24/500 on CodeContests Replay (+1.2 points). These correspond to 14, 12, and 6 additional solved sources in the respective aggregate counts. The paired intervals for these three contrasts span zero. Across all twelve settings, changes range from 6.8 to +5.4 points (Table 7), placing the DeepSeek/CodeARC gain within a model– domain-dependent response.

Together, the studies examine successive parts of the evidence-to-learning path: additional observations improve prediction, matched mass changes probability estimates without changing category ordering, and the complete weighted update yields positive fresh-generation point estimates in several model–domain settings. The downstream comparison evaluates the complete update; the matched-mass comparison isolates the probability estimator.

## 5 LIMITATIONS

Our evaluation covers executable coding tasks and one-round adaptation with a frozen lexical relevance kernel. Effective mass measures normalized relevance concentration; accounting for duplicated or dependent executions requires an additional dependence model. The uncertaintypenalized utility is a training score with no asserted confidence-coverage level. Downstream results compare the full update with the entering policy, so isolating the contributions of weighting and KL anchoring requires matched training controls. The reported source-bootstrap intervals condition on one fitting run per setting. Source-disjoint transfer and repeated RSI updates are further evaluation settings beyond the present protocol.

## 6 CONCLUSION

EESD represents a correction’s execution evidence through relative transition support and effective pseudo-count mass. A Dirichlet posterior maps this representation to an uncertainty-penalized supervised weight within KL-anchored self-distillation. The analysis establishes category-order invariance and the ordering of effective- and fixed-mass coefficients. Experiments show lower outcome NLL with longer histories, lower NLL than fixed mass in all four eight-observation comparisons, and a primary matched-mass gain at identical argmax predictions. One correction-learning round increases DeepSeek/CodeARC all-tests Pass@1 by 5.4 percentage points. The full model–domain matrix defines the scope of this result. Together, these findings support explicit evidence-mass modeling when learning from a coding agent’s self-corrections.

## REFERENCES

Xinyun Chen, Maxwell Lin, Nathanael Scharli, and Denny Zhou. Teaching large language models¨ to self-debug. In International Conference on Learning Representations, 2024. URL https: //arxiv.org/abs/2304.05128.

Hyosoon Jang, Yunhui Jang, Sungjae Lee, Jungseul Ok, and Sungsoo Ahn. Self-training large language models with confident reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 14925–14939, 2025. doi: 10.18653/v1/2025.findings-emnlp.806.

Chunyang Jiang, Chi-Min Chan, Wei Xue, Qifeng Liu, and Yike Guo. Importance weighting can help large language models self-improve. Proceedings of the AAAI Conference on Artificial Intelligence, 39(23):24257–24265, 2025. doi: 10.1609/aaai.v39i23.34602.

Dacheng Li, Shiyi Cao, Chengkun Cao, Xiuyu Li, Shangyin Tan, Kurt Keutzer, Jiarong Xing, Joseph E. Gonzalez, and Ion Stoica. S\*: Test time scaling for code generation. In Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 15964–15978, 2025a. doi: 10.18653/v1/2025.findings-emnlp.865.

Qingyao Li, Xinyi Dai, Xiangyang Li, Weinan Zhang, Yasheng Wang, Ruiming Tang, and Yong Yu. CodePRM: Execution feedback-enhanced process reward model for code generation. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 8169–8182, 2025b. doi: 10.18653/v1/2025.findings-acl.428.

Yanzhou Li, Yiran Zhang, Xiaoyu Zhang, Xiaoxia Liu, and Yang Liu. CODESKILL: Learning selfevolving skills for coding agents, 2026. URL https://arxiv.org/abs/2605.25430.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, 2023.

Luca Martino, V´ıctor Elvira, and Francisco Louzada. Effective sample size for importance sampling based on discrepancy measures. Signal Processing, 131:386–401, 2017. doi: 10.1016/j.sigpro. 2016.08.025.

Ansong Ni, Srini Iyer, Dragomir Radev, Veselin Stoyanov, Wen-Tau Yih, Sida Wang, and Xi Victoria Lin. LEVER: Learning to verify language-to-code generation with execution. In Proceedings of the 40th International Conference on Machine Learning, volume 202, pp. 26106–26128. PMLR, 2023.

Julian Aron Prenner and Romain Robbes. Runbugrun—an executable dataset for automated program repair, 2023. URL https://arxiv.org/abs/2304.01102.

Leonardo Ranaldi. Evolving agents. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16561–16569, 2026. doi: 10.18653/ v1/2026.acl-long.754.

Murat Sensoy, Lance Kaplan, and Melih Kandemir. Evidential deep learning to quantify classification uncertainty. In Advances in Neural Information Processing Systems, 2018.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, 2023.

Hao Tang, Keya Hu, Jin Peng Zhou, Sicheng Zhong, Wei-Long Zheng, Xujie Si, and Kevin Ellis. Code repair with llms gives an exploration-exploitation tradeoff. In Advances in Neural Information Processing Systems, 2024. doi: 10.52202/079017-3746.

Leitian Tao, Baolin Peng, Haorui Wang, Hang Wang, Hao Cheng, Wenlin Yao, Qianhui Wu, Tao Ge, Sharon Li, and Jianfeng Gao. ExecCritic: Learn to test, test to improve for coding agents, 2026. URL https://arxiv.org/abs/2609.09133.

Anjiang Wei, Tarun Suresh, Jiannan Cao, Naveen Kannan, Yuheng Wu, Kai Yan, Thiago S. F. X. Teixeira, Ke Wang, and Alex Aiken. CodeARC: Benchmarking reasoning capabilities of LLM agents for inductive program synthesis, 2025. URL https://arxiv.org/abs/2503.23145.

Chuan Xiao, Zhengbo Jiao, Shaobo Wang, Wei Wang, Bing Zhao, Hu Wei, Linfeng Zhang, and Lin Qu. Socratic-swe: Self-evolving coding agents via trace-derived agent skills, 2026. URL https://arxiv.org/abs/2606.07412.

Haoyan Yang, Khiem Le, Ting Hua, Shangqian Gao, Binfeng Xu, Zheng Tang, Jie Xu, Nitesh V. Chawla, Hongxia Jin, and Vijay Srinivasan. Dynamic noise preference optimization: Selfimprovement of large language models with self-synthetic data, 2025. URL https://arxiv. org/abs/2502.05400.

Xunjian Yin, Xinyi Wang, Liangming Pan, Li Lin, Xiaojun Wan, and William Yang Wang. Godel ¨ agent: A self-referential agent framework for recursively self-improvement. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 27890–27913, 2025. doi: 10.18653/v1/2025.acl-long.1354.

## A PROOFS AND INTERPRETATION OF EFFECTIVE EVIDENCE

## A.1 PSEUDO-COUNT CONSTRUCTION AND ELEMENTARY PROPERTIES

Weighted-likelihood representation. Condition on the observed relevance weights $p .$ Starting from a symmetric Dirichlet prior, the weighted likelihood

$$
\widetilde L ( \theta ; h ) = \prod _ { i = 1 } ^ { n } \theta _ { z _ { i } } ^ { E ( p ) p _ { i } } = \prod _ { k = 1 } ^ { 4 } \theta _ { k } ^ { E ( p ) r _ { k } }\tag{13}
$$

gives

$$
\widetilde { p } ( \boldsymbol { \theta } \mid h ) \propto \prod _ { k = 1 } ^ { 4 } \theta _ { k } ^ { \alpha - 1 + E ( p ) r _ { k } } .\tag{14}
$$

This is the density of $\mathrm { D i r } ( \alpha { \bf 1 } + E ( p ) r )$ . This weighted likelihood provides an algebraic interpretation of the relevance-conditioned pseudo-count update.

Bounds and count recovery. Since $\textstyle \sum _ { i } p _ { i } = 1$ and $p _ { i } \geq 0 ,$ , Cauchy–Schwarz yields

$$
1 = \left( \sum _ { i } p _ { i } \right) ^ { 2 } \leq n \sum _ { i } p _ { i } ^ { 2 } ,
$$

while $\textstyle \sum _ { i } p _ { i } ^ { 2 } \leq 1$ . Therefore $1 \leq E ( p ) \leq n$ . Equality at the upper bound requires uniform weights, for which $E ( p ) p _ { i } = 1$ and

$$
E ( p ) r _ { k } = \sum _ { i } \mathbb { I } [ z _ { i } = k ] .
$$

Equality at the lower bound requires a point mass. Multiplying every $s _ { i }$ by the same positive constant leaves $p , r , E ( p )$ , and the pseudo-count posterior unchanged.

## A.2 POSTERIOR UTILITY MOMENTS

For a Dirichlet vector with mean $\mu$ and total concentration $\alpha _ { 0 }$

$$
\operatorname { C o v } ( \theta ) = \frac { \operatorname { d i a g } ( \mu ) - \mu \mu ^ { \top } } { \alpha _ { 0 } + 1 } .
$$

Thus $\mathbb { E } [ v ^ { \top } \theta ] = v ^ { \top } \mu$ and

$$
\operatorname { V a r } ( v ^ { \top } \theta ) = \frac { \sum _ { k } \mu _ { k } v _ { k } ^ { 2 } - ( v ^ { \top } \mu ) ^ { 2 } } { \alpha _ { 0 } + 1 } .
$$

Taking $v = ( 1 , - 1 , 0 , 0 )$ gives Equations 6 and 7. The training score $\bar { u } - \lambda \sigma _ { u }$ combines these utility moments.

## A.3 PROOF OF PROPOSITION 1

Proof. Set $A = 4 \alpha$ . The posterior mean is

$$
\mu _ { k } ( \tau ) = \frac { \alpha + \tau r _ { k } } { A + \tau } = \frac { \tau } { A + \tau } r _ { k } + \frac { A } { A + \tau } \frac { 1 } { 4 } .
$$

For any categories $j , k ,$

$$
\mu _ { j } ( \tau ) - \mu _ { k } ( \tau ) = \frac { \tau ( r _ { j } - r _ { k } ) } { A + \tau } .
$$

Its sign is independent of $\tau > 0 .$ , proving invariance of category ordering, including ties.

To establish monotonicity of the training weight, write

$$
d = r _ { \mathrm { F I X } } - r _ { \mathrm { R E G R E S S } } , \qquad b = r _ { \mathrm { F I X } } + r _ { \mathrm { R E G R E S S } } .
$$

Then

$$
\bar { u } _ { \tau } = \frac { \tau d } { A + \tau } ,\tag{15}
$$

$$
\sigma _ { \tau } ^ { 2 } = \frac { ( 2 \alpha + \tau b ) ( A + \tau ) - \tau ^ { 2 } d ^ { 2 } } { ( A + \tau ) ^ { 2 } ( A + \tau + 1 ) } .\tag{16}
$$

If $d \leq 0$ , then $\bar { u } _ { \tau } \leq 0$ and $w _ { \tau } = 0$ for every $\tau .$

Suppose $d > 0 ,$ . All Dirichlet parameters are positive, so $\sigma _ { \tau } > 0$ . The ratio $T _ { \tau } = \bar { u } _ { \tau } / \sigma _ { \tau }$ satisfies

$$
T _ { \tau } ^ { 2 } = \frac { d ^ { 2 } ( A + \tau + 1 ) } { b - d ^ { 2 } + ( 2 \alpha + A b ) / \tau + 2 \alpha A / \tau ^ { 2 } } .\tag{17}
$$

Because $| d | \leq b \leq 1$ , we have $b - d ^ { 2 } \geq 0$ . The numerator is strictly increasing in $\tau ,$ , and the positive denominator is strictly decreasing. Therefore $T _ { \tau }$ is strictly increasing. Since ${ { \bar { u } } _ { \tau } }$ is also positive and increasing,

$$
w _ { \tau } = \bar { u } _ { \tau } \left[ 1 - { \frac { \lambda } { T _ { \tau } } } \right] _ { + }
$$

is nondecreasing for every $\lambda \geq 0$ . Finally, $E ( p ) \leq n { \mathrm { ~ g i v e s ~ } } w _ { E ( p ) } \leq w _ { n }$

The proposition characterizes the supervised coefficient under fixed $r , \alpha , \lambda$ , and utility vector.

## A.4 WHEN REDUCING EVIDENCE MASS CHANGES LOG LOSS

For a predictive label set of size $K ,$ , the symmetric-prior construction is $\mu _ { k } ( \tau ) = ( \alpha + \tau r _ { k } ) / ( K \alpha + \tau )$ The difference $\mu _ { j } - \mu _ { k } = \tau ( r _ { j } - r _ { k } ) / ( K \alpha + \tau )$ preserves category ordering for every $K$ . The category-order statement holds for any predictive label set. The correction-utility result uses $K = 4$ and $v = ( 1 , - 1 , 0 , 0 )$

For an observed transition $y ,$ define

$$
\ell _ { \tau } ( y ) = - \log \mu _ { y } ( \tau ) = - \log \frac { \alpha + \tau r _ { y } } { 4 \alpha + \tau } .
$$

Differentiation gives

$$
\frac { \partial \ell _ { \tau } ( y ) } { \partial \tau } = \frac { \alpha ( 1 - 4 r _ { y } ) } { ( \alpha + \tau r _ { y } ) ( 4 \alpha + \tau ) } .\tag{18}
$$

Consequently, when $E ( p ) < n$ reducing mass from n to $E ( p )$ lowers log loss exactly when $r _ { y } < 1 / 4 _ { \cdot }$ increases it when $r _ { y } > 1 / 4$ , and leaves it unchanged when $r _ { y } = 1 / 4$ . The log-loss effect is therefore outcome-dependent: empirical gains reflect how relevance concentration aligns with errors in relative support.

## A.5 INTERPRETIVE LIMITS OF EFFECTIVE MASS

Consider replacing every observation by m identical copies. Each copy then has normalized weight $p _ { i } / m$ , so

$$
E ( \boldsymbol { p ^ { \prime } } ) = \left( \sum _ { i } \sum _ { \ell = 1 } ^ { m } ( p _ { i } / m ) ^ { 2 } \right) ^ { - 1 } = m E ( \boldsymbol { p } ) , \qquad r ^ { \prime } = r .
$$

Duplicating observations increases mass while leaving relative support fixed. Effective mass consequently summarizes normalized relevance concentration, while dependence and absolute score reliability are separate properties of the execution record.

## B EXECUTION PROTOCOL AND IMPLEMENTATION

## B.1 MEASUREMENT PROTOCOLS

Table 4: Evaluation protocols and outcomes. Candidate selection, outcome prediction, and post-update generation use the protocols below.
<table><tr><td>Study</td><td>Recorded protocol</td><td>Reported outcome</td></tr><tr><td>Candidate-bank audit</td><td>500 source groups per benchmark; eight candidates and four public executions per group</td><td>Visible-selected and retrospective hidden-oracle success rates</td></tr><tr><td>History-size prediction</td><td> $n \in \{ 1 , 2 , 4 , 8 \}$  visible observations; RunBugRun and CodeARC; DeepSeek and Qwen; primary run</td><td>Future-outcome NLL and categorical accuracy</td></tr><tr><td>Matched-mass prediction</td><td>DeepSeek/RunBugRun; same relative support and effective-arm settings within each contrast</td><td>NLL difference and argmax agreement; three repetitions</td></tr><tr><td>One-round learning</td><td>Three backbones and four domains; 500 primary sources per cell; one fitting run and one fresh candidate per source</td><td>Pass@1 requiring all ten evaluator tests; paired source-bootstrap interval</td></tr></table>

The retrospective oracle in Appendix C selects within a fixed candidate bank using evaluator outcomes. The downstream reference is the unchanged entering policy. Outcome prediction is scored with NLL and categorical accuracy; post-update generation is scored with all-tests Pass@1. Evaluation uses the recorded source populations. The supplied aggregate reports leave the overlap between training and evaluator source populations unspecified.

## B.2 RANDOMNESS AND UNCERTAINTY

The recorded random-number-generator seeds are 1701 for the primary history sweep, matchedmass comparison, and downstream evaluation, and 1702 and 1703 for the additional matched-mass repetitions. Each downstream model–domain cell reports one fitting run. Its interval is a nominal per cell 95% paired source-bootstrap interval conditional on the evaluated policies. We use the recorded 10,000-draw intervals and report fitting-run replication separately from source-level uncertainty.

## B.3 RELEVANCE SCORING AND TOKEN-LEVEL OBJECTIVE

Let $\Delta a$ be the unified diff between $a ^ { 0 }$ and $a ^ { 1 }$ , and let $f$ be the frozen 256-dimensional signed-hash text encoder. Execution descriptor $e _ { i }$ contains the input, expected output, actual output, stderr, and categorical outcome. The one-round correction scorer uses

$$
s _ { i } = \exp \bigl ( \kappa \cos \bigl ( f ( \Delta a ) , f ( e _ { i } ) \bigr ) \bigr ) , \qquad \kappa = 1 6 .\tag{19}
$$

The scorer is frozen and uses public execution descriptors. Equation 19 specifies the one-round correction scorer; the prediction studies retain their recorded matched-comparison settings.

For scored correction-response length $L _ { h }$ , let $c _ { h , j }$ be the actual training prompt followed by $a _ { < j } ^ { 1 } .$ The two token-mean losses are

$$
\mathcal { L } _ { \mathrm { C E } } ( h ) = - \frac { 1 } { L _ { h } } \sum _ { j = 1 } ^ { L _ { h } } \log \pi _ { t + 1 } ( a _ { j } ^ { 1 } \mid c _ { h , j } ) ,\tag{20}
$$

$$
D _ { \mathrm { K L } } ^ { ( h ) } ( \pi _ { t } \| \pi _ { t + 1 } ) = \frac { 1 } { L _ { h } } \sum _ { j = 1 } ^ { L _ { h } } D _ { \mathrm { K L } } \big ( \pi _ { t } ( \cdot  { | } c _ { h , j } )  { \| } \pi _ { t + 1 } ( \cdot  { | } c _ { h , j } ) \big ) .\tag{21}
$$

Teacher and student use the same response prefixes. The execution record determines the scalar $w ( h )$ and the training prompt supplies the token context. At $w ( h ) = 0$ , the objective retains its KL term.

## B.4 OPTIMIZATION SETTINGS

Table 5: Recorded one-round training configuration. GPU memory denotes nominal capacity.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Hardware</td><td>NVIDIA RTX A6000; 48 GB per GPU</td></tr><tr><td>Student adaptation / base</td><td>LoRA only / frozen</td></tr><tr><td>Quantization</td><td>4-bit NF4, double quantization</td></tr><tr><td>Compute dtype</td><td>bfloat16</td></tr><tr><td>LoRA rank / scaling alpha / dropout</td><td>16 / 32 / 0</td></tr><tr><td>Optimizer / learning rate</td><td> $\mathrm { A d a m W / 2 \times 1 0 ^ { - 4 } }$ </td></tr><tr><td>Gradient accumulation / clipping</td><td>16 / 1.0</td></tr><tr><td>Maximum sequence length</td><td>4608</td></tr><tr><td>Relevance encoder / κ</td><td>256-dimensional signed hash / 16</td></tr><tr><td>Dirichlet prior α</td><td>0.1</td></tr><tr><td>Utility / uncertainty coefficient</td><td> $\left( 1 , - 1 , 0 , 0 \right) / \lambda = 0 . 5$ </td></tr><tr><td>Forward-KL coefficient β</td><td>0.03</td></tr><tr><td>Loss normalization</td><td>Response tokens</td></tr><tr><td>Training budget policy</td><td>Response-token accounting</td></tr></table>

## B.5 ONE-CORRECTION ALGORITHM

Given $( a ^ { 0 } , a ^ { 1 } )$ and their public before/after execution outcomes:

1. Compute $s _ { i }$ using Equation 19 and normalize $p _ { i } = s _ { i } / \sum _ { j } s _ { j }$

2. Assign each outcome pair its transition $z _ { i }$ and aggregate $\begin{array} { r } { r _ { k } = \sum _ { i } p _ { i } \mathbb { I } [ z _ { i } = k ] } \end{array}$

3. Set $\begin{array} { r } { E ( p ) = 1 / \sum _ { i } p _ { i } ^ { 2 } } \end{array}$ and $\alpha _ { k } ^ { \mathrm { p o s t } } = \alpha + E ( p ) r _ { k }$

4. Calculate u¯ and $\sigma _ { u }$ under $v = ( 1 , - 1 , 0 , 0 )$ , then set $w = [ \bar { u } - \lambda \sigma _ { u } ] _ { + }$

5. Fit the LoRA student with $w \mathcal { L } _ { \mathrm { C E } } + \beta D _ { \mathrm { K L } } ^ { ( h ) } ( \pi _ { t } \Vert \pi _ { t + 1 } )$ while the entering-policy teacher is frozen.

## C CANDIDATE-BANK SELECTION DIAGNOSTIC

For each of 500 source groups per benchmark, visible-score selection chooses among eight fixed candidates using four public executions. A retrospective hidden oracle selects the candidate with the highest additional evaluator score in that bank. This evaluation-only diagnostic measures selection headroom among the generated candidates.

Table 6: Selection headroom within a fixed candidate bank. Both selectors operate on the same eight generated candidates; oracle selection uses evaluator outcomes.
<table><tr><td>Benchmark</td><td>Visible selected (%)</td><td>Hidden oracle (%)</td><td>Gap (pp)</td></tr><tr><td>CodeARC</td><td>27.0</td><td>31.2</td><td>+4.2</td></tr><tr><td>RunBugRun</td><td>71.4</td><td>75.4</td><td>+4.0</td></tr></table>

The gaps quantify the selection headroom available within each fixed candidate bank.

## D COMPLETE ONE-ROUND DOWNSTREAM RESULTS

Table 7: Complete one-round downstream matrix for the primary run. Each model–domain setting contains 500 sources and reports fresh all-tests Pass@1. Intervals are paired source-bootstrap 95% intervals.
<table><tr><td>Benchmark Backbone</td><td></td><td></td><td>EESD ↑</td><td>∆(pp) ↑</td><td>95% CI (pp)</td></tr><tr><td rowspan="3">RunBugRun</td><td>Qwen2.5 Coder 7B</td><td>no_update↑</td><td>240/500</td><td>+2.4</td><td>[-2.0,+6.8]</td></tr><tr><td>DeepSeek Coder 6.7B</td><td>228/500 184/500</td><td>198/500</td><td>+2.8</td><td>[-1.8,+7.4]</td></tr><tr><td>Gemma 3 4B</td><td>142/500</td><td>132/500</td><td>-2.0</td><td>[-5.6,+1.4]</td></tr><tr><td rowspan="3">CodeARC</td><td>Qwen2.5 Coder 7B</td><td>85/500</td><td>83/500</td><td>-0.4</td><td>[-2.8,+2.0]</td></tr><tr><td>DeepSeek Coder 6.7B</td><td>75/500</td><td>102/500</td><td>+5.4</td><td>[+2.8,+8.0]</td></tr><tr><td>Gemma 3 4B</td><td>59/500</td><td>56/500</td><td>-0.6</td><td>[-2.6,+1.4]</td></tr><tr><td rowspan="3">APPS Replay</td><td>Qwen2.5 Coder 7B</td><td>57/500</td><td>57/500</td><td>+0.0</td><td>[-1.8,+1.8]</td></tr><tr><td>DeepSeek Coder 6.7B</td><td>23/500</td><td>17/500</td><td>-1.2</td><td>[-3.2,+0.8]</td></tr><tr><td>Gemma 3 4B</td><td>36/500</td><td>2/500</td><td>-6.8</td><td>[-9.0,-4.8]</td></tr><tr><td rowspan="3">CodeContests Replay</td><td>Qwen2.5 Coder 7B</td><td>47/500</td><td>39/500</td><td>-1.6</td><td>[-3.2,+0.0]</td></tr><tr><td>DeepSeek Coder 6.7B</td><td>18/500</td><td>24/500</td><td>+1.2</td><td>[-0.2,+2.8]</td></tr><tr><td>Gemma 3 4B</td><td>31/500</td><td>4/500</td><td>-5.4</td><td>[-7.6,-3.2]</td></tr></table>

Table 7 reports all twelve model–domain settings. DeepSeek has positive point estimates in three domains; Qwen improves on RunBugRun and matches its entering-policy solved count on APPS Replay. The full range of paired differences is [ 6.8, +5.4] percentage points. Intervals quantify source-level uncertainty for the primary fitting run.

## E HISTORY-SIZE MECHANISM DETAILS

Table 8: Primary history-size sweep with both fixed and effective mass. Values are future-outcome NLL; lower is better. Displayed precision is four decimals.
<table><tr><td colspan="2"></td><td colspan="2"> $n = 1$ </td><td colspan="2"> $n = 2$ </td><td colspan="2"> $n = 4$ </td><td colspan="2"> $n = 8$ </td></tr><tr><td>Benchmark</td><td>Backbone</td><td>Fixed</td><td>EED</td><td>Fixed</td><td>EED</td><td>Fixed</td><td>EED</td><td>Fixed</td><td>EED</td></tr><tr><td>RunBugRun</td><td>DeepSeek</td><td>.4696</td><td>.4696</td><td>.3656</td><td>.3646</td><td>.2703</td><td>.2693</td><td>.2160</td><td>.2114</td></tr><tr><td>RunBugRun</td><td>Qwen</td><td>.4737</td><td>.4737</td><td>.3292</td><td>.3291</td><td>.2704</td><td>.2704</td><td>.2188</td><td>.2113</td></tr><tr><td>CodeARC</td><td>DeepSeek</td><td>.7103</td><td>.7103</td><td>.5440</td><td>.5440</td><td>.4149</td><td>.4149</td><td>.3005</td><td>.2892</td></tr><tr><td>CodeARC</td><td>Qwen</td><td>.7785</td><td>.7785</td><td>.6176</td><td>.6176</td><td>.4725</td><td>.4727</td><td>.3439</td><td>.3415</td></tr></table>

Table 8 supplies Figure 3. The history-size sweep and the matched-mass study in Appendix F retain their respective recorded predictor settings. The DeepSeek/RunBugRun difference at $n = 4$ is 0.0010 in the sweep; the primary matched-mass contrast is 0.00609. Displayed ties refer to the four-decimal reporting precision.

Table 9: Categorical accuracy (%) in the reported history-size sweep. Values are reported to one decimal place. Accuracy is measured on future categorical outcomes.
<table><tr><td>Benchmark</td><td>Backbone</td><td> $n = 1$ </td><td> $n = 2$ </td><td> $n = 4$ </td><td> $n = 8$ </td></tr><tr><td>RunBugRun</td><td>DeepSeek 6.7B</td><td>90.7</td><td>90.9</td><td>92.2</td><td>93.2</td></tr><tr><td>RunBugRun</td><td>Qwen2.5 7B</td><td>90.6</td><td>91.4</td><td>91.5</td><td>92.5</td></tr><tr><td>CodeARC</td><td>DeepSeek 6.7B</td><td>83.3</td><td>83.9</td><td>84.7</td><td>88.8</td></tr><tr><td>CodeARC</td><td>Qwen2.5 7B</td><td>80.5</td><td>81.0</td><td>83.7</td><td>86.6</td></tr></table>

## F MATCHED-MASS MECHANISM AND REPLICATION

Concentration bins. Table 3 and Figure 4(b) report the same primary DeepSeek/RunBugRun comparison at matched effective-arm settings. The bins contain 3,000 examples in total, with identical argmax predictions between estimators. Positive NLL differences favor effective mass. The table reports the recorded bin-level point estimates.

Recorded repetitions. Table 10 gives all three recorded repetitions. Each contrast matches the effective-arm parameters between predictors. The table records the initialization-specific contrasts under the outcome-dependent log-loss characterization in Appendix A.4. Initialization identifiers are defined in Appendix B.2.

Table 10: DeepSeek/RunBugRun mechanism sensitivity across recorded initializations at the effective-arm parameters. Positive NLL gain favors effective evidence over fixed mass.
<table><tr><td>Initialization</td><td>NLL gain ↑</td><td>95% CI</td></tr><tr><td>1701</td><td>+0.00609</td><td>[+0.00211,+0.01075]</td></tr><tr><td>1702</td><td>-0.00367</td><td>[-0.00447,-0.00287]</td></tr><tr><td>1703</td><td>+0.00067</td><td>[+0.00023,+0.00118]</td></tr></table>

## G DERIVED ARITHMETIC AND PAIRED-COUNT INTERPRETATION

## G.1 CODING ILLUSTRATION AND SUPERVISION WEIGHTS

In Figure 1, sum $\left( \mathrm { r a n g e } \left( 1 , \mathrm { ~ \ n ~ } \right) \right)$ gives 0, 1, 3, 6 on inputs 1, 2, 3, 4, while sum(range(1, $\mathrm { n } { + } \mathrm { 1 } ) \ \mathrm { ) }$ ) gives 1, 3, 6, 10. All transitions are FIX, so $r = ( 1 , 0 , 0 , 0 )$ for either stipulated relevance vector: (0.25, 0.25, 0.25, 0.25) or (0.97, 0.01, 0.01, 0.01). The relevance vectors define the analytical example independently of the lexical kernel. Their masses are 4 and $1 / 0 . 9 4 1 2 \approx 1 . 0 6 2 4 7 3$ . With $\alpha = 0 . 1 , \lambda = 0 . 5$ , and $v = ( 1 , - 1 , 0 , 0 )$ , Equations 5–8 give w  0.832081 and 0.541946. Fixed mass assigns 0.832081 to both. Targets and KL coefficients are held fixed while the supervised coefficients change with effective mass.

## G.2 REPORTED MASS FRACTIONS AND PAIRED COUNTS

For the four-observation matched comparison in Table 3, the retained pseudo-count fraction is $E ( p ) / 4$ . The three bin means therefore retain $1 0 0 ( 1 . 5 1 1 / 4 ) = 3 7 . 7 7 5 \% , 1 \bar { 0 } 0 ( 2 . 5 2 1 / 4 ) = 6 3 . 0 2 5 \%$ and $\mathrm { i 0 0 ( 3 . 6 8 1 / 4 ) } = 9 2 . 0 2 5 \%$ of fixed mass. The main text rounds these to one decimal place. These ratios summarize retained pseudo-count mass.

For paired downstream outcomes, let $N _ { 0 1 }$ count sources that fail under no update and pass under EESD, and $N _ { 1 0 }$ count the reverse. The difference in solved-source counts equals $N _ { 0 1 } - \bar { N } _ { 1 0 }$ , giving

$$
\Delta _ { \mathrm { p p } } = 1 0 0 \frac { N _ { 0 1 } - N _ { 1 0 } } { N } , \qquad N = 5 0 0 .
$$

Aggregate solved counts identify the net difference $N _ { 0 1 } - N _ { 1 0 }$ . The paired uncertainty intervals are taken from the recorded whole-source bootstrap comparisons.