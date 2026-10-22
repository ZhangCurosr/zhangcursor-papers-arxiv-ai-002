# RUBRICOPT: SELF-REVISION OF SEMANTIC RUBRICS FOR PREFERENCE OPTIMIZATION

Wei Chen<sup>1∗</sup> Yubing Wu<sup>1</sup> Shigui Li<sup>1</sup> Delu Zeng<sup>1†</sup> Jingfeng Liu<sup>2</sup> Junmei Yang<sup>1</sup> Zhiheng Zhou<sup>1</sup> Bo Li<sup>1</sup> Min Chen<sup>1</sup> Zhou Wang<sup>3</sup> <sup>1</sup>South China University of Technology <sup>2</sup>LinkSprite <sup>3</sup>University of Waterloo

## ABSTRACT

A model’s failures can reveal gaps in the behavioral criteria guiding its training. We introduce RubricOpt, a framework that lets a policy revise these criteria within preference optimization. The policy diagnoses its own rubric-free outputs, retains requirements already covered by the rubric, and edits clauses that leave recurring failures unaddressed. A shared semantic representation connects task demands, preference-data contrasts, and rubric clauses, while a deterministic gate checks evidence references and fixed semantic anchors. The rubric then guides Direct Preference Optimization (DPO) during training, with preference pairs, labels, and the objective held fixed, and is removed at evaluation. Our analysis relates the local effect on rubric-free success to semantic alignment and a residual beyond the measured factors. Experiments on capability and open-ended generation benchmarks show that rubric conditioning can improve outcomes over ordinary DPO, with gains and regressions varying across models and datasets. A matched continuation isolates the contribution of revision from additional training and finds mixed incremental effects for the tested edit, whose feedback overlaps the evaluation set. RubricOpt thus brings the behavioral specification itself into the policy’s feedback loop.

## 1 INTRODUCTION

Language models can supply rewards for their own iterative preference training (Yuan et al., 2024). They can also improve their judgment skills by assessing their own judgments (Wu et al., 2025a). The corresponding question for semantic training rubrics is how revising their behavioral requirements affects learned behavior, including when a revision yields gains beyond continuing training with the existing rubric.

Written rubrics specify desired response properties and can supply reward signals (Gunjal et al., 2025) or guide the construction of synthetic preference data (Gallego, 2025). EvoRubrics jointly trains a policy and a rubric generator to adapt rubric-based rewards as learning proceeds (Ding et al., 2026). SERPO co-evolves response evidence, query-specific rubrics, and policy parameters, using rubric scores as rewards for test-time reinforcement learning (Wang et al., 2026a). We study rubric revision in Direct Preference Optimization (DPO) (Rafailov et al., 2023), holding the pairs, labels, and objective fixed while changing the rubric context. A matched continuation with the existing rubric measures the edit’s incremental contribution.

We introduce RubricOpt to study how a model revises its own training rubric across preferencetraining rounds (Fig. 1). Fine-grained evaluation describes response quality through explicit skills (Ye et al., 2024). We use a fixed eight-factor ontology to connect the model’s task failures, the contrasts in its preference data, and the rubric’s Prefer/Avoid clauses. Diagnosis-guided selfevolution uses failure histories to direct subsequent learning (Wei et al., 2026). RubricOpt applies this diagnostic idea to rubric revision, with the current policy reading task outcomes, self-judged factor scores, and judge feedback on its own outputs to attribute recurring failures to the eight factors. It retains explicit requirements for further learning and proposes clause edits where the wording leaves recurring failures uncovered. A deterministic gate checks evidence references, factor anchors, and edit size before carrying the rubric into the next round. Training-only Semantic Rubric Steering uses the resulting rubric to condition both responses of a preference pair during DPO (Rafailov et al., 2023), with the pairs, labels, and objective held fixed. Its exposure decreases during training, and evaluation uses the resulting policy without the rubric. This use of training-only information has a precedent in privileged-information learning (Vapnik & Vashist, 2009). Our analysis relates the extra update induced by a rubric to rubric-free success through its alignment with task demand (Sec. 3.3).

![](images/54a1e85c12670070efca3694df1dc5773ab62f7083327211d1e4fe850d0a24d6.jpg)  
Figure 1: The RubricOpt loop. (a) Task demand and preference-data supply share eight factor axes. (b) Matched Prefer/Avoid clauses condition both responses during DPO. (c) Rubric-free evaluation supplies feedback for policy self-revision and the next REFINE/KEEP decision. Dashed frames group stages. Green boxes denote computation.

The experiments evaluate how changes to the training specification affect comprehensive capability and open-ended generation. Fixed-state probes and criterion comparisons show that rubric text changes the learning update and can alter rubric-free outcomes (Sec. 4.1). Across model families and preference corpora, rubric conditioning improves some capability and generation evaluations over ordinary DPO, with gains and regressions varying by setting (Tabs. 1 and 2). The dataset comparison relates these effect profiles to the behavioral contrasts available in each corpus (Sec. 4.2). A matched continuation separates revision from further optimization and finds mixed incremental results from the tested edit (Tab. 3).

RubricOpt makes the behavioral specification itself an object of policy-driven revision within preference learning. Shared semantic coordinates connect failure diagnosis and preference-data contrasts to editable rubric clauses, while training-only steering connects those clauses to subsequent parameter updates. This closes the loop from learned behavior to the specification of further learning, with the effect of revision evaluated independently of additional training.

## 2 RELATED WORK

Preference Optimization and Credit Assignment. The family descends from RLHF, in which a reward model is fit to pairwise human judgements (Stiennon et al., 2020) and a policy is then optimised against it at assistant scale (Ouyang et al., 2022; Bai et al., 2022). DPO instead fits chosen– rejected pairs with a classification objective (Rafailov et al., 2023), whose implicit reward is the log density ratio between the policy and the reference (Sugiyama et al., 2012; Chen et al., 2025). Later objectives change one component of that recipe: IPO replaces it with a squared loss on the pairwise margin (Azar et al., 2024), KTO drops the pairing requirement by scoring each response against a reference point (Ethayarajh et al., 2024), and SimPO removes the reference model and normalises the margin by length (Meng et al., 2024). Two analyses place the family in a wider frame: one connects these objectives to imitation learning (Xiao et al., 2025), and one disentangles how their update suppresses the rejected response and preserves the chosen one (Chen et al., 2026). A complementary line assigns credit inside a response rather than to the response as a whole: TDPO reweights tokens by a forward-KL penalty (Zeng et al., 2024), TIS-DPO reweights them by importance sampling against an estimated weight (Liu et al., 2025), SEPO selects tokens using a token-level reward estimate (Yang et al., 2025b), T-REG uses a token-level reward as a regulariser (Zhou et al., 2025), and

TIDPO guides the update by token importance (Yang et al., 2026). RubricOpt operates at a different level: a semantic factor names a behavioral distinction that can span several tokens or reasoning steps, while the pairwise loss itself is left unchanged.

Criteria and Behavioral Diagnosis. Written criteria already enter learning through several interfaces. As reward signals, Rubrics as Rewards scores on-policy responses against a rubric (Gunjal et al., 2025). As data, Configurable Preference Tuning builds rubric-conditioned synthetic preference pairs (Gallego, 2025). And as weights, CoRT contrasts rubric-conditioned with rubric-free replays to form token-level weights (Zhang et al., 2026). Criterion-conditioned training also has close relatives outside preference optimisation: learning using privileged information trains on a signal that is absent at test time (Vapnik & Vashist, 2009). Context distillation trains on a context and then evaluates without it, introduced as an alignment technique (Askell et al., 2021) and developed as a standalone method (Snell et al., 2022). And classifier-free guidance interpolates between conditioned and unconditioned predictions (Ho & Salimans, 2022). On the diagnostic side, fine-grained evaluation describes response quality through interpretable skill sets and error taxonomies—FLASK by grading along skill dimensions (Ye et al., 2024), EvalTree by profiling weaknesses as a hierarchical capability tree (Zeng et al., 2025), and ErrorMap and ErrorAtlas by charting the failure landscape into error categories (Ashury-Tahan et al., 2026)—and diagnosis-driven post-training then acts on what those tools find, with DiagEvo feeding diagnosed errors back into self-evolution through a hierarchical error memory (Wei et al., 2026) and APTER turning expert-grounded rubrics into posttraining supervision (Wang et al., 2026b). RubricOpt uses shared semantic coordinates to connect task diagnosis, preference-data contrasts, and edits to the training rubric.

Self-Improvement and Recursive Feedback. Self-improvement methods differ in the component they revise and how that revision enters subsequent behavior. Self-Refine edits outputs from self-feedback (Madaan et al., 2023), and automatic prompt optimisation edits inference-time text from feedback (Pryzant et al., 2023). In parameter learning, Self-Rewarding uses the model to generate its own reward signals (Yuan et al., 2024), Meta-Rewarding trains judgment skills through a meta-judge (Wu et al., 2025a), and self-play preference optimisation regenerates preferences (Wu et al., 2025b). Rubric evolution also enters the reward interface through joint policy–generator learning (Ding et al., 2026) and co-evolving query-specific rubrics during test-time reinforcement learning (Wang et al., 2026a). RubricOpt places self-revision in the semantic rubric that conditions preference training. The current policy proposes clause edits from feedback about its own behavior, then learns under the resulting rubric in the next round. The initial clause bank, factor ontology, task outcomes, and fixed scoring prompt provide the reference points for revision.

## 3 RUBRICOPT: METHOD AND ANALYSIS

RubricOpt alternates preference learning under a written rubric with revision of that rubric by the resulting policy.

Problem Setup. We study preference optimization on a fixed dataset $\begin{array} { r c l } { { \mathcal D } } & { { = } } & { { \{ ( { \pmb x } , { \pmb y } ^ { + } , { \pmb y } ^ { - } ) \} } } \end{array}$ of chosen–rejected pairs, with a trainable policy $\pi _ { \theta }$ and a frozen reference $\pi _ { \mathrm { r e f } }$ trained by DPO (Rafailov et al., 2023),

$$
\mathcal { L } = - \mathbb { E } \left[ \log \sigma \left( \beta \left[ \bar { \ell } _ { \theta } ( y ^ { + } \mid x ) - \bar { \ell } _ { \mathrm { r e f } } ( y ^ { + } \mid x ) - \bar { \ell } _ { \theta } ( y ^ { - } \mid x ) + \bar { \ell } _ { \mathrm { r e f } } ( y ^ { - } \mid x ) \right] \right) \right] ,\tag{1}
$$

with ${ \bar { \ell } } ( { \pmb y } \mid { \pmb x } ) = L ^ { - 1 } \log \pi ( { \pmb y } \mid { \pmb x } )$ averaged over the L response tokens. The pairs, labels, and objective are held fixed; only the text that conditions a pair may change, which is the degree of freedom RubricOpt introduces.

Behavior is described by a fixed eight-factor ontology $\mathcal { F } = \{ f _ { 1 } , \ldots , f _ { 8 } \}$ (appendix A.2), on which a blind judge returns one ordinal score per factor, $z _ { k } ( \pmb { x } , \pmb { y } ) \in \{ 0 , \dotsc , 4 \}$ (appendix A.3). Every behavioral quantity here is a contrast between two groups of these scores, and only two contrasts are used. Responses to a benchmark B are grouped by the task outcome $o \in \{ 0 , 1 \}$ , and semantic demand is the mean score gap between the groups, measured for one model on one task,

$$
D _ { m , B , k } = \mathbb { E } [ z _ { k } \mid o = 1 , m , B ] - \mathbb { E } [ z _ { k } \mid o = 0 , m , B ] .\tag{2}
$$

Inside a preference pair the two responses are contrasted by $\Delta z _ { k } = z _ { k } ( x , y ^ { + } ) - z _ { k } ( x , y ^ { - } )$ , and semantic supply summarizes its distribution,

$$
\mu _ { k } = \mathbb { E } [ \Delta z _ { k } ] , \qquad C _ { k } = \operatorname* { P r } ( \Delta z _ { k } \neq 0 ) , \qquad P _ { k } = \operatorname* { P r } ( \Delta z _ { k } > 0 ) ,\tag{3}
$$

as the mean signed contrast, the contrast coverage, and the share of pairs whose contrast agrees with the label. Demand is a property of the model–task pair and supply a property of the dataset, and both live on the same eight axes.

The artifact RubricOpt edits is a rubric $R _ { r }$ , a short natural-language specification whose clauses are keyed to factors, with r indexing rounds and a frozen clause bank supplying the initial $R _ { 0 }$ . For factor group ${ \mathcal { G } } \subseteq { \mathcal { F } }$ the training context is

$$
c _ { r } = \operatorname { r e n d e r } ( R _ { r } , { \mathcal { G } } ) ,\tag{4}
$$

where the renderer concatenates the current Prefer/Avoid clauses in the fixed factor order and ∅ denotes the absent criterion. Here $\mathcal { G } = \mathcal { F }$ except in the factor-set ablation of Sec. 4.2. A conditioning context is therefore u $\in \{ c _ { r } , \emptyset \} , \pi _ { \theta } ( \pmb { y } \mid \pmb { x } , \boldsymbol { u } )$ is the policy under that context, the main evaluations use $\pi _ { \boldsymbol { \theta } } ( \pmb { y } \mid \pmb { x } )$ , and Sec. 4.2 tests rubric reinsertion at inference time. Appendix A.1 collects the symbols used throughout.

## 3.1 DIAGNOSIS AND CONDITIONING

Shared Behavioral Coordinates. With the coordinates of Sec. 3 fixed, the diagnosis is a measurement on one model and one dataset: we score responses from Qwen3-4B on each benchmark and each HelpSteer3 pair, then read out demand $D _ { m , B , k }$ (Eq. (2)) and supply $\mu _ { k } , C _ { k } , P _ { k } \left( \mathrm { E q . } \left( 3 \right) \right)$ . We mark demand gaps of at least 0.15 on the $_ { 0 - 4 }$ scale as meaningful and contrast coverage below 0.25 as sparse. Tab. 7 reports the complete Qwen3-4B demand matrix. Judge checks and length controls appear in appendices A.3 and D.1.

The measurement separates the factors into Common $( f _ { 2 }$ reasoning validity, $f _ { 5 }$ completeness), Specific $( f _ { 4 }$ factual grounding, $f _ { 7 }$ clarity, $f _ { 8 }$ efficiency), and Low-demand $( f _ { 1 }$ comprehension, $f _ { 3 }$ verification, $f _ { 6 }$ instruction adherence). The shared factor ontology links task diagnosis, preference-data contrasts, and clause-level feedback. All eight factors enter the initial rubric from the frozen clause bank. The demand and supply measurements characterize the learning setting; recurring failures whose requirements are incompletely expressed determine subsequent wording edits.

Training-Only Conditioning. Each round fixes $R _ { r }$ and its full-factor context $\begin{array} { r l } { c _ { r } } & { { } = } \end{array}$ render $[ R _ { r } , { \mathcal { F } } )$ . At step t of $N$ , let $\rho _ { t } = t / N$ and independently sample $u _ { t } = c _ { r }$ with probability $q ( \rho _ { t } )$ and $u _ { t } = \emptyset$ otherwise for each pair, sharing it across both responses, the policy, and the frozen reference. Exposure is $q ( \rho ) = 1$ for $\rho \leq 0 . 1$ , {1+cos[π $\cdot ( \rho - 0 . 1 ) / \mathrm { \dot { 0 } } . 8 ] \} / 2$ for $0 . 1 < \rho \leq 0 . 9$ and 0 thereafter. With response-token mean log-probability $\tilde { \ell } ( \pmb { y } \mid \pmb { x } , \ b { u } ) = \bar { L } ( \pmb { y } ) ^ { - 1 } \log \pi ( \pmb { y } \mid \pmb { x } , \ b { u } )$ training uses the unchanged DPO objective

$$
\begin{array} { r l } & { M _ { u } = \bar { \ell } _ { \theta } ( y ^ { + } \mid x , u ) - \bar { \ell } _ { \mathrm { r e f } } ( y ^ { + } \mid x , u ) - \bar { \ell } _ { \theta } ( y ^ { - } \mid x , u ) + \bar { \ell } _ { \mathrm { r e f } } ( y ^ { - } \mid x , u ) , } \\ & { \quad \mathcal { L } = - \mathbb { E } [ \log \sigma ( \beta M _ { u _ { t } } ) ] . } \end{array}\tag{5}
$$

Rubric tokens are masked from the loss and excluded from response-length normalization. Evaluation uses $\pi _ { \boldsymbol { \theta } } ( \pmb { y } \mid \pmb { x } )$ without a rubric.

## 3.2 EVIDENCE-GROUNDED RECURSIVE REFINEMENT

Given rubric $R _ { r } ,$ the policy trained with its rendered context $c _ { r }$ , and item-level benchmark feedback $\boldsymbol { B } _ { \boldsymbol { r } + 1 }$ , one iteration is

$$
R _ { r } \longrightarrow c _ { r } \longrightarrow \pi _ { r + 1 } \longrightarrow B _ { r + 1 } \longrightarrow R _ { r + 1 } .\tag{6}
$$

Five prompt stages extract benchmark failures, synthesize recurring patterns, attribute them to the eight factors, propose conservative clause updates, and audit the candidate rubric. The policy $\pi _ { r + 1 }$ performs this analysis and proposes the edits itself, using task outcomes, self-judged factor scores, and fixed-judge feedback on its own rubric-free outputs. Attribution compares each recurring failure with the current clauses. An explicit requirement is retained for further learning, while an incompletely expressed requirement is eligible for refinement. A separate deterministic gate checks evidence references, factor anchors, and edit size before accepting a change. A covered requirement or a failed check returns KEEP. The accepted rubric is rendered as $c _ { r + 1 } = \operatorname { r e n d e r } ( R _ { r + 1 } , \mathcal { G } )$ for the next training round; KEEP retains the corresponding clauses from $R _ { r }$ . Full prompts, stage contracts, and the deterministic acceptance checks are in appendix C.2.

![](images/2f51906e34d6a18e6fac5ef3df899bfd641167208c34c0e14ff5429e10b4527a.jpg)  
Figure 2: Policy-driven rubric refinement (appendix C.2). Prompt stages 1–5 diagnose failures, attribute gaps, propose edits, and audit the candidate. The audit is advisory; step 6 decides acceptance. Covered requirements and rejected edits retain the original clause. The excerpt shows the factualgrounding edit. The dashed frame groups prompt stages. Green boxes denote computation.

## 3.3 ANALYSIS OF THE STEERING EFFECT

The Update and Its Rubric-Free Effect. Fix one round, with rubric context $c = c _ { r }$ and absent context $0 = \emptyset$ . For $s _ { u } ( \pmb { y } ) = L ^ { - 1 } \nabla _ { \theta } \log \pi _ { \theta } ( \pmb { y } \mid \pmb { x } , \pmb { u } )$ , the per-pair loss gradient is

$$
g _ { u } = \nabla _ { \theta } [ - \log \sigma ( \beta M _ { u } ) ] = - \kappa _ { u } [ s _ { u } ( \pmb { y } ^ { + } ) - s _ { u } ( \pmb { y } ^ { - } ) ] , \qquad \kappa _ { u } = \beta \sigma ( - \beta M _ { u } ) ,\tag{7}
$$

Let $d _ { u } = s _ { u } ( { y } ^ { + } ) - s _ { u } ( { y } ^ { - } )$ and $v _ { u } = \mathbb { E } _ { \mathcal { D } } [ \kappa _ { u } d _ { u } ]$ be the mean descent direction. Independent pairlevel context sampling gives $v _ { q } = v _ { 0 } + q \Delta v$ , where $\Delta v = v _ { c } - v _ { 0 }$ is the rubric’s extra update direction.

Fix the evaluation prompt law $\nu ,$ binary success rule $O _ { B } .$ , and factor scores $z = ( z _ { 1 } , \ldots , z _ { 8 } ) ^ { \top }$ . Assume finite prompt and response spaces and a positive, continuously differentiable policy. Rubricfree deployment uses $P _ { \theta } ( \pmb { x } , \pmb { y } ) = \nu ( \pmb { x } ) \pi _ { \theta } ( \pmb { y } | \ \pmb { x } , 0 )$ , with success probability $p ~ = ~ p _ { B } ( \theta ) ~ =$ $\mathbb { E } _ { P _ { \theta } } [ o _ { B } ] \in ( 0 , 1 )$ . The extra direction changes deployment log-probabilities at rate

$$
h ( \pmb { x } , \pmb { y } ) = \langle \nabla _ { \theta } \log \pi _ { \theta } ( \pmb { y } \mid \pmb { x } , 0 ) , \Delta \ b { v } \rangle .\tag{8}
$$

Let $D = \mathbb { E } [ z ~ \vert ~ o _ { B } = 1 ] - \mathbb { E } [ z ~ \vert ~ o _ { B } = 0 ]$ be population demand under $P _ { \theta } { \mathrm { . } }$ , and a the least-squares coefficients of h on centered factor scores. Write $\epsilon _ { h } , \epsilon _ { o }$ for the root-mean-square projection residuals of h and $o _ { B } \mathrm { ~ - ~ } p$ , respectively; the full construction is in appendix B.2.

Proposition 1 (Local success from semantic alignment) At a shared parameter state and reference, with the update directions heldfixed,

$$
\begin{array} { r } { \langle \nabla _ { \theta } p _ { B } , v _ { q } - v _ { 0 } \rangle = q p ( 1 - p ) \big ( \boldsymbol D ^ { \top } \boldsymbol a + \mathcal { R } \big ) , } \end{array}\tag{9}
$$

where $| \mathcal { R } | \leq \epsilon _ { o } \epsilon _ { h } / [ p ( 1 - p ) ]$ ]. For $q > 0 , i f D ^ { \top } a > \epsilon _ { o } \epsilon _ { h } / [ p ( 1 { - } p ) ]$ , then $p _ { B } ( \theta + \eta v _ { q } ) > p _ { B } ( \theta + \eta v _ { 0 } )$ for all sufficiently small $\eta > 0 .$ . The comparison reverses when $D ^ { \top } a < - \epsilon _ { o } \epsilon _ { h } / [ p ( 1 - p ) ]$

Demand D describes the behaviors associated with success; a expresses the rubric-induced change in deployment log-probabilities in those semantic coordinates. Their alignment sets the sign of the local success effect when it dominates the residual contribution outside the measured factors. Changing the preference corpus can change the update directions and hence this alignment, even under the same rubric.

For two rendered rubrics $c _ { 0 } , c _ { 1 }$ at a shared policy and reference, replacing $\Delta v$ by $v _ { c _ { 1 } } - v _ { c _ { 0 } }$ gives the local edit comparison at a common exposure rate. This comparison motivates the matched KEEP/REFINE continuation in Sec. 4.3, which measures the edit’s incremental effect over further training with the existing rubric.

## 4 EXPERIMENTS

The analysis in Sec. 3.3 organizes three empirical questions: how rubric text changes the training update, how conditioning affects rubric-free outcomes across datasets, and what an edit adds beyond continuing with the existing rubric. We examine them through fixed-state probes and a criterionidentity pilot, dataset-matched ordinary-DPO controls, and matched KEEP/REFINE continuations, respectively.

Models and Datasets. We cover two families and three sizes: Qwen3-4B-Instruct and Qwen3- 8B-Instruct from the Qwen3 family (Yang et al., 2025a), and Mistral-7B-Instruct-v0.3 from the Mistral family (Jiang et al., 2023). Formal runs train a rank-64 LoRA adapter $( \alpha = 1 2 8 ,$ , dropout 0.05) at learning rate $1 0 ^ { - 6 } \cdot$ , global batch 64, DPO $\beta = 1$ . We use two public preference datasets, HelpSteer3 (Wang et al., 2025) and Tulu3 (Lambert et al., 2025).

Benchmark Classes. Evaluation covers two benchmark classes, summarized separately. The comprehensive-capability suite is seven accuracy benchmarks—GSM8K (Cobbe et al., 2021), MATH-Hard (Hendrycks et al., 2021), BBH (Suzgun et al., 2023), MuSR (Sprague et al., 2024), ARC (Clark et al., 2018), TruthfulQA (Lin et al., 2022), and MMLU-Pro (Wang et al., 2024)—each with a task check defining the binary success outcome o in Prop. 1. The open-ended generative suite is three protocols—MT-Bench (Zheng et al., 2023), Arena-Hard (Li et al., 2024), and Wild-Bench (Lin et al., 2024)—on which a judge scores free-form responses and there is no task check. Generative protocols are scored on matched item sets under one judge.

Feedback and Evaluation. The refinement engine reads item-level outputs from first-round rubric-free benchmark evaluations: judge rationales on Arena-Hard, MT-Bench, and WildBench, and correctness feedback on GSM8K, MATH-Hard, and MMLU-Pro. The KEEP and REFINE continuations are then compared on matched evaluation items under the same benchmark protocols. This evaluates a rubric update informed by benchmark feedback, with the unchanged-rubric branch controlling for additional training. Feedback items overlap the final evaluation set, so the continuation comparison measures adaptation to the observed benchmark feedback. The diagnostic/held-out split in Sec. 3.1 assesses measurement stability. Generation, judging, extraction, paired tests, and all absolute scores are in appendix D.1.

## 4.1 HOW RUBRIC TEXT CHANGES LEARNING

Context-Dependent Updates. In Eq. (7), context enters the update through both the margindependent weight $\kappa _ { u }$ and the pair direction $d _ { u }$ . A fixed-state probe over 7,253 preference pairs gives conditioned–unconditioned gradient cosines of 0.74–0.79 and relative gradient changes of 0.65–0.72. The likelihood probe measures the semantic sensitivity of the pairwise margin: the $f _ { 5 }$ completeness criterion’s margin-change slope is 0.0226 nats per response token per unit of $\Delta z _ { 5 }$ compared with 0.0116 when read against reasoning contrast $\Delta z _ { 2 }$

Criterion Identity at Fixed Exposure. Changing the criterion text can change $v _ { c }$ in Sec. 3.3 while preserving the conditioning interface. A 150-step pilot compares reasoning $( f _ { 2 } )$ and clarity $( f _ { 7 } )$ criteria of nearly equal length, using the same pairs and conditioned rows at $q = 0 . 2$ . MuSR pass@4 is 70.8% under the reasoning criterion, 68.6% under the clarity criterion, and 70.8% under ordinary DPO. The paired $f _ { 2 ^ { - } } f _ { 7 }$ comparison has 17 prompts passing only under $f _ { 2 }$ and 6 only under $f _ { 7 }$ (appendix C.1). The two criterion texts produce different rubric-free outcomes on MuSR pass@4, with the reasoning criterion matching ordinary DPO there.

## 4.2 DATASET CONTRASTS AND RUBRIC-FREE OUTCOMES

The Two Datasets Contrast Different Factors. The shared coordinates of Sec. 3 are what make the two datasets comparable. Fig. 3a reports contrast coverage $C _ { k }$ on both, and Tab. 8 gives the HelpSteer3 statistics behind it. HelpSteer3 leaves three factors sparse against the 0.25 sparsity rule: $f _ { 2 }$ reasoning validity (0.146), $f _ { 4 }$ factual and evidential grounding (0.149), and $f _ { 3 }$ verification and calibration (0.220). Reasoning validity $( f _ { 2 } )$ is meaningful on five of the seven benchmarks but is contrasted in only 14.6% of HelpSteer3 pairs, while completeness $( f _ { 5 } )$ is meaningful on four and contrasted in 47.4%. Tulu3 contrasts seven of the eight factors more often than HelpSteer3 does, and no factor of its eight falls below that rule; its lowest coverage is 0.294. Its largest advantages over HelpSteer3 are on $f _ { 2 } ( + 0 . 2 6 8 ) , f _ { 8 } ( + 0 . 1 9 9 ) , f _ { 4 } ( + 0 . 1 6 1 )$ , and $f _ { 7 } \left( + 0 . 1 3 5 \right)$ . These profiles describe the frequency of behavioral contrasts on the same axes used for task demand in Sec. 3.1. The full demand matrix is Tab. 7, and across diagnostic and held-out subsets its largest change is 0.045, with every gap keeping its sign. Fig. 3b groups the same gaps by task family: verification and calibration separates success from failure about six times more strongly on the knowledge benchmarks than on the reasoning benchmarks (0.131 against 0.020), and clarity and organization runs the other way (0.051 against 0.105).

Demand profiles differ between the two Qwen3 models in a task-dependent way (Fig. 4). On BBH, the eight-factor profiles of Qwen3-4B and Qwen3-8B are negatively correlated $( r = - 0 . 7 5 1 )$ : completeness and clarity change from positive to negative gaps, while verification and calibration becomes more strongly associated with success. TruthfulQA retains a similar relative profile across the two models $( r = 0 . 9 5 3 )$ , although the gap magnitudes change. These measurements support diagnosing semantic priorities for each model and task.

![](images/5446adaf7e144ca673d6bff901e4f542ce4bf617f74ab53b7cbdd40598b0aa78.jpg)

![](images/0d9d027e43942f589a85167f90c5c7cb4b38be02288452b7cb847d6a1a9490f1.jpg)  
Figure 3: Diagnosis on shared semantic factors. (a) Contrast coverage (fraction of pairs) on both corpora using the same frozen judge; the dashed line marks the 0.25 sparsity threshold. (b) Qwen3- 4B demand, averaged equally over four reasoning or three knowledge benchmarks. Demand is the success–failure gap on the 0–4 judge scale. Factors follow appendix A.2.

![](images/06be2acde55c5e00b21bc6ec10dbfcdffa75de56eafb2a7ff53b178db58d29b6.jpg)

![](images/42406e4670f6da52ee06ee761503fa7b3c133c3d0524a338b867a78646fd78df.jpg)  
Figure 4: Semantic demand in Qwen3-4B and Qwen3-8B. Lines connect the same factor; error bars show recorded 95% confidence intervals. r is Pearson correlation across the eight factors.

Conditioning Effects Across Datasets. Tab. 1 reports the Tulu3 capability results and the Help-Steer3 open-ended results for both Qwen3 models. BBH improves in both Qwen3 models, with effects measured against their own ordinary-DPO controls. On Qwen3-4B, conditioning raises BBH from 15.43 to 19.64 (+4.20) and MATH-Hard from 39.15 to 43.35, and the seven-task average from 48.20 to 50.02. The other five benchmarks move by at most 2.28 points. Qwen3-8B gains 9.61 BBH points. The Mistral-7B Tulu3 run decreases BBH by 7.95 points relative to ordinary DPO (Tab. 2; appendix D.2).

On Qwen3-4B, the four rubric compositions raise Arena-Hard by 1.4–2.2 percentage points relative to ordinary DPO (Tab. 10). Common+Specific has the highest WildBench point estimate, while the full rubric has the highest MT-Bench score and average relative change across the three protocols.

Table 1: Rubric-free Qwen3 results on both benchmark suites, each from its own dataset. Capability scores and their unweighted seven-task mean are percentages from the Tulu3 runs. The openended protocols come from the HelpSteer3 runs: MT-Bench and WildBench use native scores and Arena-Hard is the win rate against GPT-4-0314 (%). Open-ended Avg. $\begin{array} { r } { \Delta = \frac { 1 0 0 } { 3 } \sum _ { b } ( s _ { b } / s _ { b } ^ { \mathrm { D P O } } - 1 ) } \end{array}$ averages relative changes across all three benchmarks against the same block’s DPO control. Bold marks maxima within each model block (including ties). Shading marks rubric-trained rows.
<table><tr><td rowspan="2"></td><td colspan="8">Tulu3 comprehensive capability (%)</td><td colspan="4">HelpSteer3 open-ended generation</td></tr><tr><td colspan="2">MATH-</td><td colspan="2"></td><td colspan="2"></td><td colspan="2">Truthful MMLU-</td><td colspan="2">MT-</td><td colspan="2">Wild- Avg. ∆</td></tr><tr><td>Method</td><td>GSM8K</td><td>Hard</td><td>BBH MuSR</td><td></td><td>ARC</td><td>QA</td><td>Pro</td><td>Avg.</td><td></td><td>Arena- Bench Hard (%)</td><td>Bench</td><td>(%)</td></tr><tr><td>Qwen3-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Instruct</td><td>85.90</td><td>62.24</td><td></td><td>12.82 43.39 62.80</td><td></td><td>52.27</td><td></td><td>41.98 51.63</td><td>6.320</td><td></td><td>49.0 3.730</td><td>-2.96</td></tr><tr><td>Ordinary DPO</td><td>84.53</td><td>39.15</td><td>15.43</td><td>42.59</td><td>61.09</td><td>52.71</td><td>41.88</td><td>48.20</td><td>6.890</td><td></td><td>50.93.617</td><td>0.00</td></tr><tr><td>RUBRICOPT-R1</td><td>86.81</td><td>43.35</td><td>19.64</td><td>43.12</td><td>61.35</td><td>53.74</td><td>42.10</td><td></td><td>50.02 6.930</td><td></td><td></td><td>53.03.727 +2.58</td></tr><tr><td>RUBRICOPT-R2</td><td>80.74</td><td>37.12 22.73</td><td></td><td>42.06</td><td>59.47</td><td>55.16</td><td>40.43</td><td>48.24</td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Instruct</td><td>91.90</td><td>70.13 37.43</td><td></td><td>54.9094.10</td><td></td><td>78.20</td><td></td><td>67.60 70.61 7.410</td><td></td><td></td><td></td><td>64.5 4.074 -0.05</td></tr><tr><td>Ordinary DPO</td><td>91.80</td><td></td><td></td><td>38.66 55.2094.65</td><td></td><td>76.80</td><td>67.35</td><td></td><td>7.219</td><td></td><td>64.2 4.211</td><td>0.00</td></tr><tr><td>RUBRICOPT-R1</td><td>91.40</td><td>72.77 48.26</td><td></td><td>55.15 95.00</td><td></td><td>77.85</td><td></td><td></td><td>68.30 72.68 7.416</td><td></td><td></td><td>65.7 4.329 +2.62</td></tr></table>

Variation Across Datasets and Models. In Prop. 1, the local effect depends on task demand and the corpus-dependent update direction. The ordinary-DPO controls measure the full rubric’s finite-training effect separately within each dataset. On Qwen3-4B, the capability effect is larger on Tulu3: BBH improves by 4.20 points, compared with 0.23 on HelpSteer3, and the seven-task mean by 1.82 points, compared with 0.18 (Tabs. 1 and 10). Both methods train for one epoch within each dataset. On HelpSteer3, Qwen3-8B gains 0.20 MT-Bench points and 1.5 Arena-Hard points (Tab. 1). Mistral-7B gains 0.29 MT-Bench points and 1.8 Arena-Hard points (Tab. 2).

Table 2: Mistral-7B results on both benchmark suites, each from its own dataset. Capability scores and their unweighted seven-task mean are percentages. Open-ended Avg. ∆ follows the definition in Tab. 1. Bold marks maxima within each dataset block; shading marks rubric-trained rows.
<table><tr><td></td><td colspan="8">Tulu3 comprehensive capability (%)</td><td colspan="4">HelpSteer3 open-ended generation</td></tr><tr><td></td><td colspan="9">MATH-</td><td rowspan="2">MT- Arena-</td><td rowspan="2"></td><td rowspan="2">Wild- Avg. ∆ Bench (%)</td></tr><tr><td>Method</td><td>GSM8K</td><td>Hard</td><td>BBH MuSR</td><td></td><td>ARC QA</td><td>Truthful MMLU-</td><td>Pro</td><td>Bench Hard (%)</td></tr><tr><td>Instruct</td><td>45.75</td><td>19.65</td><td>36.95</td><td>52.45</td><td>79.45</td><td>58.75</td><td>30.05</td><td>46.15</td><td>5.090</td><td>18.1</td><td>2.336</td><td>-8.44</td></tr><tr><td>Ordinary DPO</td><td>44.45</td><td>20.90</td><td>40.40</td><td>50.95</td><td>76.50</td><td>53.90</td><td>30.00</td><td>45.30</td><td>4.850</td><td>24.0</td><td>2.477</td><td>0.00</td></tr><tr><td>RUBRICOPT-R1</td><td>47.90</td><td>21.50</td><td>32.45</td><td>51.85</td><td>75.60</td><td>55.40</td><td>32.15</td><td></td><td>45.26 5.140</td><td>25.8 2.480</td><td></td><td>+4.53</td></tr></table>

Adding Factors to a Common Rubric. Four HelpSteer3 compositions each contain the Common factors $f _ { 2 }$ and $f _ { 5 }$ , with the Specific or Low-demand factors added. Their seven-task means differ by at most 0.13 points and exceed ordinary DPO’s 52.56 by at most 0.18. Common raises BBH by 0.90 points relative to ordinary DPO, adding Specific yields 0.47, adding Low yields 0.40, and including both yields 0.23. All four compositions improve Arena-Hard by 1.4–2.2 points, with a different ordering from BBH. The small average differences among compositions do not identify a superior factor set. The full ontology supplies a fixed set of clauses for subsequent editing. Per-benchmark results are in Tab. 10.

Rubric Dependence at Inference Time. Reinserting a rubric at inference time has task-dependent effects on the same trained checkpoint (appendix D.1). On matched sampled BBH items, the trained $R _ { 1 }$ wording changes accuracy by −1.25 points and the candidate $R _ { 2 } \ \mathsf { b y } + 3 . 7 0 $ points. The separate greedy MATH-Hard evaluation changes by +0.68 and +0.76 points respectively.

## 4.3 WHAT THE RUBRIC EDIT CONTRIBUTES

The edit comparison in Sec. 3.3 fixes the starting policy and varies only the rubric-induced update direction; it is carried through a matched continuation, with KEEP controlling for additional training. The trained Qwen3-4B policy reviews its first-round feedback and finds a recurring $f _ { 4 }$ gap in its training rubric. The frozen clause prohibited unsupported assertions but left the evidence requirement for “common knowledge” ambiguous. The revised Prefer/Avoid pair reads:

Prefer: factual claims are grounded in the given information or explicitly sourced common knowledge. Avoid: invented facts, unsupported assertions, or inferences from vague common knowledge.

Round 1 refines only $f _ { 4 } ;$ the other seven factors return KEEP because no recurring uncovered requirement is identified. The edit adds seven words across the Prefer/Avoid pair in a 234-token rubric. Tab. 3 evaluates it through two HelpSteer3 continuations from the same Qwen3-4B policy, using the same pairs, frozen reference, optimizer settings, seed, and pair-level exposure mask. The KEEP branch retains $R _ { 0 } ;$ the REFINE branch uses $R _ { 1 }$ . Unchanged-rubric continuation raises BBH from 14.96 to 18.05, a gain of 3.09 points. Relative to that continuation, the edit changes BBH by −0.25 points, MT-Bench by +0.406, and Arena-Hard by −0.1 percentage points. The edit’s incremental effects differ across tasks: BBH decreases slightly, while the MT-Bench point estimate recovers part of the decline under unchanged-rubric continuation, from 6.93 in round 1 to 6.419 under KEEP and 6.825 under REFINE. Evaluation protocols and paired comparisons are in appendix D.2.

Table 3: Qwen3-4B matched continuation on HelpSteer3. Both branches start from RUBRICOPT-R1 and differ only in rubric text. Evaluation is rubric-free. Open-ended Avg. ∆ follows the relative-DPO definition in Tab. 1. Bold marks column maxima.
<table><tr><td>Method</td><td>BBH</td><td>MATH-Hard</td><td>MT-Bench</td><td>Arena-Hard</td><td>WildBench</td><td>Avg.∆(%)</td></tr><tr><td>RUBRICOPT-R1</td><td>14.96</td><td>64.18</td><td>6.93</td><td>53.0</td><td>3.727</td><td>+2.58</td></tr><tr><td>RUBRICOPT-R2 (KEEP)</td><td>18.05</td><td>63.82</td><td>6.419</td><td>54.6</td><td>3.781</td><td>+1.66</td></tr><tr><td>RUBRICOPT-R2 (REFINE)</td><td>17.80</td><td>63.72</td><td>6.825</td><td>54.5</td><td>3.836</td><td>+4.06</td></tr></table>

The revision trace records how failure attribution and edit constraints determine the next rubric. Factor $f _ { 3 }$ receives the strongest failure signal, but its existing Avoid clause already names the concrete requirement. The engine nevertheless proposes an $f _ { 3 }$ rewrite. The gate rejects it because it drops part of the frozen semantic anchor; cross-factor vocabulary also triggers a warning. Factor $f _ { 3 }$ therefore returns to KEEP. This rejection preserves the original $f _ { 3 }$ clause for further learning. On Qwen3-8B, attribution marks every requirement in the feedback set as already covered and the operator returns KEEP for all factors. The subsequent 4B feedback pass produces a three-factor $R _ { 2 }$ candidate for a future training round.

## 5 CONCLUSION

RubricOpt connects policy self-evaluation to revision of the behavioral specification guiding prefer ence learning. Its semantic rubric translates failure feedback into training-time criteria while preference pairs, labels, and the optimization objective remain fixed. Experiments across comprehensive capability and open-ended generation show that rubric semantics can shape rubric-free behavior, with benefits depending on the model and preference corpus. The tested revision yields mixed incremental results over unchanged-rubric continuation. This makes the behavioral specification itself adaptable through the policy’s feedback loop.

Scope and Future Work. The study uses an associative demand measure and a manually specified ontology. Evaluating successive revisions on feedback-disjoint tasks and comparing proposals from initial and updated policies under a shared training budget would test the scope for sustained selfimprovement.

## REPRODUCIBILITY STATEMENT

Every trained method in this paper is a rank-64 LoRA adapter over a released checkpoint, and the appendices specify the dataset, optimizer, schedule, seed, conditioning rate, reference convention, evaluation protocols, and judge configuration for every reported run. The two preference datasets are public, and the seven accuracy benchmarks and three open-ended protocols are used through their released forms. The initial formal rubrics can be reconstructed from the canonical bank in appendix A.2 and the template in appendix C.1. The accepted HelpSteer3 $R _ { 1 }$ edit is quoted in Sec. 4.3. The refinement operator is specified as five stages with their inputs, outputs, and the deterministic acceptance checks (appendix C.2).

## BROADER IMPACT STATEMENT

RubricOpt exposes behavioral criteria as short, inspectable training artifacts, with an evidence trail for each accepted edit. The policy under diagnosis proposes the edits, so its interpretation of failures can reinforce its own biases. Deterministic checks enforce evidence references and textual constraints; rubric-free evaluation measures the resulting behavior. Deployments should review proposed rubric changes against their intended behaviors and evaluate each edit with a matched continuation control.

## REFERENCES

Shir Ashury-Tahan, Yifan Mai, Elron Bandel, Michal Shmueli-Scheuer, and Leshem Choshen. ErrorMap and ErrorAtlas: Charting the failure landscape of large language models. arXiv preprint arXiv:2601.15812, 2026. doi: 10.48550/arXiv.2601.15812. URL https://arxiv.org/ab s/2601.15812.

Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, and Jared Kaplan. A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861, 2021. doi: 10.48550/arXiv.2112.00861. URL https://arxiv.org/abs/2112.00861.

Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland,´ Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings ofMachine Learning Research, pp. 4447– 4455. PMLR, 2024. URL https://proceedings.mlr.press/v238/gheshlaghi-a zar24a.html.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback, 2022. URL https://arxiv.org/abs/2204.05862.

Wei Chen, Shigui Li, Jiacheng Li, Junmei Yang, John Paisley, and Delu Zeng. Dequantified diffusion-schrodinger bridge for density ratio estimation. In¨ International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=zvyHCOcwsw.

Wei Chen, Yubing Wu, Junmei Yang, Delu Zeng, Qibin Zhao, John Paisley, Min Chen, and Zhou Wang. Towards disentangled preference optimization dynamics: Suppress the loser, preserve the winner. In International Conference on Machine Learning, 2026. URL https://openrevi ew.net/forum?id=TaNH4XiQ6P.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? Try ARC, the AI2 reasoning challenge, 2018. URL https://arxiv.org/abs/1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.or g/abs/2110.14168.

Hongxin Ding, Baixiang Huang, Yue Fang, Weibin Liao, Zheng Li, Jinyang Zhang, Zhijing Wu, Junfeng Zhao, and Yasha Wang. EvoRubrics: Dynamic rubrics as rewards via adversarial coevolution for LLM reinforcement learning, 2026. URL https://arxiv.org/abs/2606 .23038.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Model alignment as prospect theoretic optimization. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 12634–12651. PMLR, 2024. URL https://proceedings.mlr.press/v235/ethayarajh24a.h tml.

V´ıctor Gallego. Configurable preference tuning with rubric-guided synthetic data. arXiv preprint arXiv:2506.11702, 2025. doi: 10.48550/arXiv.2506.11702. URL https://arxiv.org/ab s/2506.11702. ICML 2025 Workshop on Models of Human Feedback for AI Alignment.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean Hendryx. Rubrics as rewards: Reinforcement learning beyond verifiable domains. arXiv preprint arXiv:2507.17746, 2025. doi: 10.48550/arXiv.2507.17746. URL https://arxiv.org/ab s/2507.17746.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset, 2021. URL https://arxiv.org/abs/2103.03874.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. doi: 10.48550/arXiv.2207.12598. URL https://arxiv.or g/abs/2207.12598. NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lelio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril,´ Thomas Wang, Timothee Lacroix, and William El Sayed. Mistral 7b, 2023. URL´ https: //arxiv.org/abs/2310.06825.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tulu 3: Pushing frontiers in open language model post-training, 2025. URL https: //arxiv.org/abs/2411.15124.

Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline, 2024. URL https://arxiv.org/abs/2406.11939.

Bill Yuchen Lin, Yuntian Deng, Khyathi Chandu, Faeze Brahman, Abhilasha Ravichander, Valentina Pyatkin, Nouha Dziri, Ronan Le Bras, and Yejin Choi. WildBench: Benchmarking LLMs with challenging tasks from real users in the wild, 2024. URL https://arxiv.org/abs/2406 .04770.

Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods, 2022. URL https://arxiv.org/abs/2109.07958.

Aiwei Liu, Haoping Bai, Zhiyun Lu, Yanchao Sun, Xiang Kong, Xiaoming Wang, Jiulong Shan, Albin Madappally Jose, Xiaojiang Liu, Lijie Wen, Philip S. Yu, and Meng Cao. TIS-DPO: Tokenlevel importance sampling for direct preference optimization with estimated weights. In The Thirteenth International Conference on Learning Representations, 2025. URL https://proc eedings.iclr.cc/paper\_files/paper/2025/hash/7fb9f39075a520247267 6a7531568212-Abstract-Conference.html.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, volume 36, 2023. doi: 10.52202/075280-2019. URL https://papers.neurips.cc/pa per\_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstr act-Conference.html.

Yu Meng, Mengzhou Xia, and Danqi Chen. SimPO: Simple preference optimization with a reference-free reward. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-3946. URL https://proceedings.neurips.cc/paper\_fil es/paper/2024/hash/e099c1c9699814af0be873a175361713-Abstract-Con ference.html.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://arxiv.org/abs/2203.02155.

Reid Pryzant, Dan Iter, Jerry Li, Yin Lee, Chenguang Zhu, and Michael Zeng. Automatic prompt optimization with “gradient descent” and beam search. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 7957–7968, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.494. URL https://aclanthology.org/2023.em nlp-main.494/.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, 2023. doi: 10.52202/07528 0-2338. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html.

Charlie Snell, Dan Klein, and Ruiqi Zhong. Learning by distilling context. arXiv preprint arXiv:2209.15189, 2022. doi: 10.48550/arXiv.2209.15189. URL https://arxiv.or g/abs/2209.15189.

Zayne Sprague, Xi Ye, Kaj Bostrom, Swarat Chaudhuri, and Greg Durrett. MuSR: Testing the limits of chain-of-thought with multistep soft reasoning, 2024. URL https://arxiv.org/abs/ 2310.16049.

Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. Learning to summarize from human feedback. In Advances in Neural Information Processing Systems, volume 33, 2020. URL https://arxiv.org/ abs/2009.01325.

Masashi Sugiyama, Taiji Suzuki, and Takafumi Kanamori. Density-ratio matching under the bregman divergence: a unified framework of density-ratio estimation. Annals of the Institute of Sta tistical Mathematics, 64(5):1009–1044, 2012.

Mirac Suzgun, Nathan Scales, Nathanael Scharli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung,¨ Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. Challenging BIG-Bench tasks and whether chain-of-thought can solve them. In The Eleventh International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2210.09261. arXiv:2210.09261.

Vladimir Vapnik and Akshay Vashist. A new learning paradigm: Learning using privileged information. Neural Networks, 22(5–6):544–557, 2009. doi: 10.1016/j.neunet.2009.06.042. URL https://doi.org/10.1016/j.neunet.2009.06.042.

Jianze Wang, Kunwang Zheng, Ying Liu, Yu Cao, Qilong Zhang, Jinlong Chen, Hua Yang, and Qianglong Chen. SERPO: Self-evolving rubric policy optimization for open-ended test-time reinforcement learning, 2026a. URL https://arxiv.org/abs/2607.26873.

Xukai Wang, Liangqi Li, Zhiyue Xu, Jingang Zhou, Xiaoyu Shi, Jiansheng Cai, Bo Zhang, Zhe Li, and Xu-Yao Zhang. APTER: Adaptive post-training with expert-grounded rubrics, 2026b. URL https://arxiv.org/abs/2608.14212.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. MMLU-Pro: A more robust and challenging multi-task language understanding benchmark, 2024. URL https://arxiv.org/abs/2406.01574.

Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Hoo-Chang Shin, Felipe Soares, Alexander Bukharin, Ellie Evans, Yi Dong, and Oleksii Kuchaiev. Helpsteer3-preference: Open human-annotated preference data across diverse tasks and languages, 2025. URL https://arxiv.org/abs 2505.11475.

Xincheng Wei, Yifan Ding, Yoshua Li, Dongsheng Ma, Rongxiang Weng, Xunliang Cai, Wenjian Ding, and Yao Zhang. DiagEvo: Diagnosis-guided self-evolution via hierarchical error memory, 2026. URL https://arxiv.org/abs/2609.00768v1.

Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason E. Weston, and Sainbayar Sukhbaatar. Meta-rewarding language models: Self-improving alignment with LLM-as-a-meta-judge. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, Suzhou, China, 2025a. Association for Computational Linguistics. doi: 10.18653/v1/2025.emnlp-main.583. URL https://aclanthology.org/2025.emnl p-main.583/.

Yue Wu, Zhiqing Sun, Huizhuo Yuan, Kaixuan Ji, Yiming Yang, and Quanquan Gu. Self-play preference optimization for language model alignment. In The Thirteenth International Conference on Learning Representations, 2025b. URL https://arxiv.org/abs/2405.00675. arXiv:2405.00675.

Teng Xiao, Yige Yuan, Mingxiao Li, Zhengyu Chen, and Vasant G. Honavar. On a connection between imitation learning and RLHF. In The Thirteenth International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2503.05079. arXiv:2503.05079.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025a. URL https://arxiv.org/abs/2505.09388.

Kailai Yang, Zhiwei Liu, Qianqian Xie, Jimin Huang, Erxue Min, and Sophia Ananiadou. Selective preference optimization via token-level reward function estimation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 7032–7056, Suzhou, China, 2025b. Association for Computational Linguistics. doi: 10.18653/v1/2025.emnlp-main. 359. URL https://aclanthology.org/2025.emnlp-main.359/.

Ning Yang, Hai Lin, Yibo Liu, Baoliang Tian, Guoqing Liu, and Haijun Zhang. Token-importance guided direct preference optimization. In The Fourteenth International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_files/pape r/2026/hash/172fe9f8cc55953bad5c24774bf0142b-Abstract-Conferenc e.html. ICLR 2026 Poster.

Seonghyeon Ye, Doyoung Kim, Sungdong Kim, Hyeonbin Hwang, Seungone Kim, Yongrae Jo, James Thorne, Juho Kim, and Minjoon Seo. FLASK: Fine-grained language model evaluation based on alignment skill sets. In The Twelfth International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2307.10928. ICLR 2024 Spotlight; arXiv:2307.10928.

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, and Jason E. Weston. Self-rewarding language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 57905–57923. PMLR, 2024. URL https://proceedings.mlr.press/v235/yua n24d.html.

Yongcheng Zeng, Guoqing Liu, Weiyu Ma, Ning Yang, Haifeng Zhang, and Jun Wang. Token-level direct preference optimization. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 58348–58365. PMLR, 2024. URL https://proceedings.mlr.press/v235/zeng24c.html.

Zhiyuan Zeng, Yizhong Wang, Hannaneh Hajishirzi, and Pang Wei Koh. EvalTree: Profiling lan guage model weaknesses via hierarchical capability trees. In Second Conference on Language Modeling, 2025. URL https://arxiv.org/abs/2503.08893. arXiv:2503.08893.

Bo-Wen Zhang, Junwei He, Wen Wang, Song-Lin Lv, Wentao Ma, Rongyi Lin, Shuhan Zhong, and Lan-Zhe Guo. CoRT: Counterfactual replay for token-level rubric-guided policy optimization. arXiv preprint arXiv:2607.25659, 2026. doi: 10.48550/arXiv.2607.25659. URL https: //arxiv.org/abs/2607.25659.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 46595–46623. Curran Associates, Inc., 2023. doi: 10.52202/075280-2020. URL https: //proceedings.neurips.cc/paper\_files/paper/2023/file/91f18a1287b 398d378ef22505bf41832-Paper-Datasets\_and\_Benchmarks.pdf.

Wenxuan Zhou, Shujian Zhang, Lingxiao Zhao, and Tao Meng. T-REG: Preference optimization with token-level reward regularization. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 27876–27889, Vienna, Austria, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.acl-long.1353. URL https://aclanthology.org/2025.acl-long.1353/.

LLM Usage Disclosure. Large language models (LLMs) were used throughout this work to assist with several aspects of the research and writing process. Specifically, we used LLMs to improve the clarity and presentation of the manuscript, to support literature retrieval and the discovery of relevant prior work, and to assist in drafting portions of the paper. The authors reviewed, revised, and verified all LLM-assisted content and remain fully responsible for the final manuscript.

## APPENDIX CONTENTS

A Preliminaries and Measurement 15   
A.1 Notation . 15   
A.2 Measurement and the Factor Ontology 16   
A.3 Judge Configuration and Reliability 17   
B Proofs 17   
B.1 Local Behavioral Transfer 17   
B.2 Proof Of Prop. 1 . 19   
B.3 Retention After Rubric Withdrawal . 19   
C Semantic Rubric Steering and the RubricOpt Loop 20   
C.1 The Criterion Bank and Training-Only Conditioning 20   
C.2 Stages, Contracts and the Deterministic Gate . 21   
C.3 Matched Re-Optimization and the Dataset Control 23   
D Experimental Settings and Results 23   
D.1 Evaluation and Measurement Protocols 23   
D.2 Paired Effects and Results Across Datasets and Models 26   
A PRELIMINARIES AND MEASUREMENT

## A.1 NOTATION

Models and data $m , \pi _ { \theta } , \pi _ { \mathrm { r e f } } \colon$ Current model, its trainable policy, and the frozen reference. $B , o \colon$ Diagnostic task and its binary outcome. $\mathcal { D } , \pmb { x } , \pmb { y } , \pmb { y } ^ { + } , \pmb { y } ^ { - } ;$ Preference data, prompt, response, and chosen/rejected responses. L: Number of response tokens in a response; <sup>¯</sup>ℓ averages a log-probability over them.

Behavioral measurements $\mathcal { F } = \{ f _ { 1 } , \ldots , f _ { 8 } \}$ : Fixed factor ontology of eight factors. $z _ { k } ( \pmb { x } , \pmb { y } ) \in$ $\{ 0 , \ldots , 4 \}$ : Ordinal score a blind judge assigns to factor $f _ { k } . ~ D _ { m , B , k } \mathrm { ; }$ Model- and task-specific semantic demand $( \mathrm { E q . ~ } ( 2 ) ) . ~ \Delta z _ { k } , \bar { \mu _ { k } } , \bar { C } _ { k } , P _ { k }$ : Pair contrast, its mean, contrast coverage, and labelagreement rate (Eq. (3)).

Conditioning and updates $\mathcal { G } \subseteq \mathcal { F } , c _ { r } , \emptyset \colon$ Selected factors (all eight in the main method), the context rendered from rubric $R _ { r }$ by Eq. (4), and the absent criterion. $u , u _ { t } , q ( \rho _ { t } )$ : Conditioning context u $\in \{ c _ { r } , \emptyset \}$ , its draw at training step t within round $r ,$ and the conditioning rate at progress $\rho _ { t }$ $\beta , \sigma , \mathcal { L } \colon \{ \mathrm { D P O }$ scale, logistic function, and Semantic Rubric Steering loss. $M _ { u } , s _ { u } , g _ { u } , \kappa _ { u } \colon$ Lengthnormalized chosen–rejected margin, mean per-token response score gradient, loss gradient, and scalar multiplier, all in context u. $\nu , A _ { k } , H _ { k } ;$ Evaluation prompt distribution, rubric-free expected factor score $\mathbb { E } _ { \nu } [ \bar { z } _ { k } ( { \pmb x } ) ]$ , and its gradient Lipschitz constant. $G _ { c } , G _ { 0 } , G _ { q } \mathrm { . }$ : Mean loss gradients with and without the criterion, and their mixture at rate $q . \ \eta , \theta _ { q } , \theta _ { 0 } , r _ { k } \colon$ Step size, two local updated parameter states, and the transfer remainder. $\bar { z } _ { k } ( { \pmb x } ) , \delta _ { c } , \delta _ { 0 } \mathrm { ~ : ~ }$ : Prompt-specific expected factor score and two actual optimizer increments from a shared state. Shift : Semantic shift, the change in mean factor score after an update.

Rounds and probabilistic analysis $R _ { r } , \pi _ { r + 1 } , B _ { r + 1 } \colon$ Round-r rubric, the policy trained under its context $c _ { r } ,$ and the resulting behavioral feedback (Eq. (6)). ψ , F: Full-response deployment score (Eq. (10)) and its Fisher matrix (Eq. (26)). $D , \Sigma , \Gamma _ { u } , \mathcal { P } _ { \tau : T } \colon$ Deployment demand vector and factor covariance (Eq. (19)), training–deployment gradient alignment (Eq. (11)), and remaining Fisher path length. RUBRICOPT-R2 (KEEP): Matched re-optimization branch that keeps the criterion $R _ { 0 }$ RUBRICOPT-R2 (REFINE): Matched re-optimization branch that refines the criterion to $R _ { 1 }$ . The two branches differ only in the criterion.

## A.2 MEASUREMENT AND THE FACTOR ONTOLOGY

The measurements use the definitions of Sec. 3 and the blind judge of appendix A.3.

Worked Demand Example. A GSM8K problem asks the cost of six pounds of butter at \$4 per pound when every second pound is half price. A successful response answers \$18, while a failed response answers \$16. Their judge vectors are [2, 3, 1, 3, 2, 3, 3, 3] and [2, 1, 1, 3, 2, 3, 3, 3], differing only on reasoning validity. This illustrates scoring, not estimation of demand, which aggregates the success–failure gap over the benchmark population.

Worked Supply Example. In one HelpSteer3 joke-request pair, the chosen–rejected score difference is [0, 0, 0, 0, +1, 0, 0, 0]: it contrasts completeness but not reasoning validity. A pair about female US presidents contrasts several factors at once, with difference $[ + 2 , \dot { + } 2 , 0 , \dot { + } 2 , + \dot { 1 } , + 2 , 0 , + 1 ]$ Supply therefore summarizes a distribution of within-pair contrasts rather than assigning each pair to one factor.

The Factor Ontology. Every quantity above is indexed by the fixed eight-factor ontology, whose demand profile over the seven benchmarks is Tab. 7. The frozen bank supplies one Prefer and one matched Avoid clause per factor, as rendered for training:

f1 task comprehension   
Prefer: show a correct understanding of what was asked   
Avoid: answering a different question or ignoring stated conditions   
f2 reasoning validity   
Prefer: reasoning steps that are logically grounded and internally   
consistent   
Avoid: unsupported intermediate inferences or invalid deductions   
f3 verification and calibration   
Prefer: verify key intermediate results and express uncertainty in a   
calibrated way   
Avoid: unchecked steps or unsupported overconfidence   
f4 factual and evidential grounding   
Prefer: factual claims are grounded in the given information or common   
knowledge   
Avoid: invented facts or unsupported assertions   
f5 completeness   
Prefer: responses that include the reasoning and information needed to   
fully address the task   
Avoid: missing necessary reasoning, information, or important parts of   
the task   
f6 instruction adherence   
Prefer: follow the stated instructions and stay on task   
Avoid: drifting off-topic or violating requested formats   
f7 clarity and organization   
Prefer: clearly expressed and logically organized   
Avoid: jumbled steps or answers that are hard to locate   
f8 efficiency and conciseness   
Prefer: keep only the necessary content   
Avoid: repeating reasoning or padding with filler

## A.3 JUDGE CONFIGURATION AND RELIABILITY

Production Configuration. Semantic factor scores reported in this paper come from a self-judge. The policy under diagnosis scores its own rubric-free outputs with one fixed, version-tagged judge prompt. Decoding is greedy (temperature 0, top-p 1, top-k 1) with at most 1100 new tokens, at batch size 64. The self-judge is blind to the task outcome, the gold answer, the preference role, and the identity of the generating model. Each call returns one raw integer score per factor together with a supporting sentence, and the scores are not normalized across factors, responses, or runs.

Score Anchors. The shared scale is 0 = clearly poor or absent, 1 = weak, 2 = mixed or partial, 3 = good, and 4 = strong. The production prompt defines the top of the scale as an absolute standard rather than a comparative one: a 4 is earned by concrete extra rigor, such as re-checking a computation, stating and handling an assumption or edge case, noticing an ambiguity, or flagging its own uncertainty, and it is explicitly not awarded for being correct and fluent. A 3 is competent, complete, routine work with no extra rigor; a 2 does the minimum with visible gaps; a 1 is missing or merely asserts an important behavior. The prompt also asks the judge to list two strengths and two weaknesses before scoring, and gives per-factor rules that cap a score at 1. For reasoning validity, the cap applies when any step is a non-sequitur or an equation does not follow. The prompt permits NA only when a behavior cannot occur for that response kind, and states that a required behavior which is missing still scores 0 or 1 rather than NA. In practice that rule never fired: across the 108,685 scored responses, every factor was assessable on every response.

Agreement Audits. A different judge model rescored 1,200 responses. On items where the audit judge gave a decisive score, mean agreement with the production self-judge was 0.741. Per factor, agreement was highest for task comprehension (0.909), factual and evidential grounding (0.857), and instruction adherence (0.800), and lowest for clarity and organization (0.633), verification and calibration (0.616), and efficiency and conciseness (0.559). Tab. 4 gives the per-factor agreement. The three generative benchmarks are scored by MiniMax-M3.

Table 4: Blind pairwise agreement across four benchmarks. Pairwise verdicts are compared with the direction of independent per-response scores. Decisive excludes ties in either judgment; agreement rates use this subset.
<table><tr><td>Factor</td><td>Decisive</td><td>Agree</td><td>Decisive-only rate</td></tr><tr><td> $f _ { 1 }$  task comprehension</td><td>11</td><td>10</td><td>0.9091</td></tr><tr><td> $f _ { 2 }$  reasoning validity</td><td>52</td><td>41</td><td>0.7885</td></tr><tr><td> $f _ { 3 }$  verification and calibration</td><td>232</td><td>143</td><td>0.6164</td></tr><tr><td> $f _ { 4 }$  factual and evidential grounding</td><td>7</td><td>6</td><td>0.8571</td></tr><tr><td> $f _ { 5 }$  completeness</td><td>105</td><td>80</td><td>0.7619</td></tr><tr><td> $f _ { 6 }$  instruction adherence</td><td>15</td><td>12</td><td>0.8000</td></tr><tr><td> $f _ { 7 }$  clarity and organization</td><td>199</td><td>126</td><td>0.6332</td></tr><tr><td> $f _ { 8 }$  efficiency and conciseness</td><td>247</td><td>138</td><td>0.5587</td></tr><tr><td colspan="4">Mean</td></tr></table>

Factor Separability. Across 108,685 complete responses, the production self-judge yields mean absolute inter-factor correlation 0.195 and maximum 0.362. The older judge prompt, used as a collapse control, reaches maximum correlation 1.000 and mean 0.378, and produces only four distinct score vectors across 288 responses, whereas the production prompt produces 822. The eight factors are therefore not collapsed onto one quality axis in this analysis.

## B PROOFS

## B.1 LOCAL BEHAVIORAL TRANSFER

The Deployment Space. The analysis uses finite prompt and response spaces and a strictly positive, continuously differentiable policy on a neighborhood of every path considered. The full-

response deployment score is

$$
\psi _ { 0 } ( { \pmb x } , { \pmb y } ) = \nabla _ { \theta } \log \pi _ { \theta } ( \pmb y \mid { \pmb x } , 0 ) = L ( \pmb y ) s _ { 0 } ( \pmb y ) ,\tag{10}
$$

For a training triple $\xi = ( \pmb { x } ^ { \prime } , \pmb { y } ^ { + } , \pmb { y } ^ { - } ) \sim \mathcal { D }$ , write $d _ { u } ( \xi )$ for its pair direction. The incremental rate of change of deployment log-probability along the rubric-induced direction is

$$
\begin{array} { r l } & { \Gamma _ { u } ( { \pmb x } , { \pmb y } ; { \pmb \xi } ) = \langle \psi _ { 0 } ( { \pmb x } , { \pmb y } ) , d _ { u } ( { \pmb \xi } ) \rangle , } \\ & { \qquad h ( { \pmb x } , { \pmb y } ) = \langle \psi _ { 0 } ( { \pmb x } , { \pmb y } ) , \Delta { \pmb v } \rangle = \mathbb { E } _ { { \pmb \xi } \sim \mathcal { D } } [ \kappa _ { c } \Gamma _ { c } ( { \pmb x } , { \pmb y } ; { \pmb \xi } ) - \kappa _ { 0 } \Gamma _ { 0 } ( { \pmb x } , { \pmb y } ; { \pmb \xi } ) ] . } \end{array}\tag{11}
$$

The score identity gives $\mathbb { E } _ { P _ { \theta } } [ \psi _ { 0 } \mid { \pmb x } ] = 0$ and hence $\mathbb { E } _ { P _ { \theta } } [ h \mid { \pmb x } ] = 0 .$ , so each fixed preference pair contributes through the alignment of its training gradient with the deployment score.

The Behavioral Readout. What remains is to characterize that direction. For a rubric-free evaluation distribution ν, define the behavioural readout

$$
A _ { k } ( \theta ) = \mathbb { E } _ { \pmb { x } \sim \nu , \pmb { y } \sim \pi _ { \theta } ( \cdot | \pmb { x } ) } [ z _ { k } ( \pmb { x } , \pmb { y } ) ] ,\tag{12}
$$

and let $G _ { c }$ and $G _ { 0 }$ be mean preference-loss gradients with and without the criterion, with $G _ { q } =$ $( 1 - q ) \pmb { G } _ { 0 } + q \pmb { G } _ { c } .$

Proposition 2 Suppose $A _ { k }$ has an H -Lipschitz gradient on a neighborhood containing the two update segments. For $\theta _ { q } = \theta - \eta { G _ { q } }$ and $\theta _ { 0 } = \theta - \eta G _ { 0 }$ ，

$$
A _ { k } ( \theta _ { q } ) - A _ { k } ( \theta _ { 0 } ) = - \eta q \langle \nabla _ { \theta } A _ { k } , { \cal G } _ { c } - { \cal G } _ { 0 } \rangle + r _ { k } ,\tag{13}
$$

$$
| r _ { k } | \leq \frac { H _ { k } \eta ^ { 2 } } { 2 } \left( \Vert G _ { q } \Vert ^ { 2 } + \Vert G _ { 0 } \Vert ^ { 2 } \right) .\tag{14}
$$

The Score-Function Identity. Assume differentiation and expectation can be interchanged and define $\bar { z } _ { k } ( \pmb { x } ) = \mathbb { E } _ { \pmb { y } \sim \pi _ { \theta } ( \cdot | \pmb { x } ) } [ z _ { k } ( \pmb { x } , \pmb { y } ) ]$ at the current parameter state. The score-function identity gives

$$
\nabla _ { \theta } A _ { k } = \mathbb { E } _ { { \pmb { x } } \sim { \pmb { \nu } } , { \pmb { y } } \sim \pi _ { \theta } ( \cdot | { \pmb { x } } ) } \left[ \big ( z _ { k } ( { \pmb { x } } , { \pmb { y } } ) - \bar { z } _ { k } ( { \pmb { x } } ) \big ) \nabla _ { \theta } \log \pi _ { \theta } ( { \pmb { y } } \mid { \pmb { x } } ) \right] .\tag{15}
$$

The baseline term vanishes because the conditional mean score gradient is zero. A finite response space with a smooth, positive policy provides a sufficient setting for this identity, and the factor score itself need not be differentiable in the response: a randomized judge enters through its fixed conditional mean score. Eq. (15) is taken on generated responses under the rubric-free policy, which is a different quantity from the training-pair contrast $\Delta z _ { k }$

Actual Optimizer Steps. For two parameter increments $\delta _ { c }$ and $\delta _ { \mathrm { 0 } }$ from the same parameter state, the same Taylor argument yields

$$
\begin{array} { r l r } & { } & { A _ { k } ( \theta + \pmb { \delta } _ { c } ) - A _ { k } ( \theta + \pmb { \delta } _ { 0 } ) = \langle \nabla _ { \theta } A _ { k } , \pmb { \delta } _ { c } - \pmb { \delta } _ { 0 } \rangle + r _ { k } , } \\ & { } & { | r _ { k } | \leq \frac { 1 } { 2 } H _ { k } \big ( \| \pmb { \delta } _ { c } \| ^ { 2 } + \| \pmb { \delta } _ { 0 } \| ^ { 2 } \big ) . } \end{array}\tag{16}
$$

Here the same smoothness assumption holds along both update segments, and the form accommodates adaptive updates and momentum, which is the regime of the experiments.

Eq. (16) compares updates from shared parameters, optimizer state, and minibatch. Their effect on a behavior depends on the alignment of the increment difference with that behavior’s gradient. The context probes in Sec. 4.1 measure training-pair gradient and margin changes at a fixed state. The benchmark comparisons measure rubric-free outcomes after training.

Proof Of Prop. 2. For either $G = G _ { q } \ \mathrm { o r } \ G = G _ { 0 }$ , the Lipschitz-gradient assumption gives the Taylor expansion

$$
\begin{array} { r } { A _ { k } ( { \boldsymbol { \theta } } - { \boldsymbol { \eta } } { \pmb { G } } ) = A _ { k } ( { \boldsymbol { \theta } } ) - { \boldsymbol { \eta } } \langle \nabla _ { \boldsymbol { \theta } } A _ { k } , { \pmb { G } } \rangle + e ( { \pmb { G } } ) , \qquad | e ( { \pmb { G } } ) | \leq \frac { 1 } { 2 } H _ { k } { \boldsymbol { \eta } } ^ { 2 } \| { \pmb { G } } \| ^ { 2 } . } \end{array}\tag{17}
$$

Subtract the two expansions and use $G _ { q } - G _ { 0 } = q ( G _ { c } - G _ { 0 } )$ . The remainder is $r _ { k } = e ( G _ { q } )  – e ( G _ { 0 } )$ whose absolute value is bounded by Eq. (14) by the triangle inequality. In particular, a sufficient condition for strict improvement over the ordinary step is

$$
- q \langle \nabla _ { \theta } A _ { k } , G _ { c } - { \cal G } _ { 0 } \rangle > \frac { H _ { k } \eta } { 2 } \big ( \| { \cal G } _ { q } \| ^ { 2 } + \| { \cal G } _ { 0 } \| ^ { 2 } \big ) .\tag{18}
$$

A nonzero first-order term dominates the quadratic remainder for sufficiently small positive $\eta ,$ proving the sign statements. A zero inner product leaves the sign undetermined. Improvement over the initial policy is a different comparison: its first-order term $\mathrm { i } \mathrm { \bar { s } } - \eta \langle \nabla _ { \theta } A _ { k } , G _ { q } \rangle$ □

## B.2 PROOF OF PROP. 1

Semantic Projections and the Log-Odds Form. At a fixed state, write $p = p _ { B } ( \theta ) , Z = z - \mathbb { E } [ z ]$ and

$$
{ \pmb { D } } = \mathbb { E } [ z \mid o _ { B } = 1 ] - \mathbb { E } [ z \mid o _ { B } = 0 ] , \qquad { \pmb { \Sigma } } = \mathbb { E } [ Z Z ^ { \top } ] ,\tag{19}
$$

$$
\begin{array} { r l r l r l } & { a = \Sigma ^ { \dagger } \mathbb { E } [ Z h ] , \quad } & & { \quad \boldsymbol { r } _ { h } = h - a ^ { \top } Z , \quad } & & { \quad \boldsymbol { \epsilon } _ { h } ^ { 2 } = \mathbb { E } [ r _ { h } ^ { 2 } ] , } \\ & { b = \Sigma ^ { \dagger } \mathbb { E } [ Z ( o _ { B } - p ) ] , \quad } & & { \quad \boldsymbol { r } _ { o } = o _ { B } - p - b ^ { \top } Z , \quad } & & { \quad \boldsymbol { \epsilon } _ { o } ^ { 2 } = \mathbb { E } [ r _ { o } ^ { 2 } ] , } \end{array}\tag{20}
$$

Here $\Sigma ^ { \dagger }$ is the Moore–Penrose pseudoinverse, a is the coefficient vector for the log-probability rate h, and $\epsilon _ { h } , \epsilon _ { o }$ are the corresponding root-mean-square residuals. The residual contribution is

$$
\mathcal { R } = \frac { \mathbb { E } [ r _ { o } r _ { h } ] } { p ( 1 - p ) } , \qquad | \mathcal { R } | \le \frac { \epsilon _ { o } \epsilon _ { h } } { p ( 1 - p ) } .\tag{21}
$$

If either projection residual vanishes, then $\mathcal { R } = 0$ . For $\theta _ { q } ( \eta ) = \theta + \eta v _ { q }$ and $\theta _ { 0 } ( \eta ) = \theta + \eta v _ { 0 }$ , with directions fixed at θ and $\mathrm { l o g i t } ( p ) = \log [ p / ( 1 - p )$ ], the equivalent log-odds form is

$$
\frac d { d \eta } \left\{ \mathrm { l o g i t } p _ { B } ( \theta _ { q } ( \eta ) ) - \mathrm { l o g i t } p _ { B } ( \theta _ { 0 } ( \eta ) ) \right\} \bigg | _ { \eta = 0 } = q \bigl ( D ^ { \top } a + \mathcal { R } \bigr ) ,\tag{22}
$$

Proof. The deployment score satisfies $\begin{array} { r } { \mathbb { E } [ \psi _ { 0 } \ \mid \ x ] = \sum _ { \pmb { u } } \nabla _ { \pmb { \theta } } \pi _ { \pmb { \theta } } ( \pmb { y } \ \mid \ \pmb { x } , 0 ) = 0 } \end{array}$ . For any fixed direction w, differentiating the deployment expectation gives

$$
\nabla _ { \boldsymbol { \theta } } p _ { B } ^ { \top } \boldsymbol { w } = \mathbb { E } [ ( o _ { B } - p ) \langle \psi _ { 0 } , \boldsymbol { w } \rangle ] .\tag{23}
$$

Taking $w = \Delta v$ and dividing by $p ( 1 - p )$ expresses the incremental log-odds derivative as

$$
{ \frac { \mathbb { E } [ ( o _ { B } - p ) h ] } { p ( 1 - p ) } } = \mathbb { E } [ h \mid o _ { B } = 1 ] - \mathbb { E } [ h \mid o _ { B } = 0 ] .\tag{24}
$$

Likewise, conditioning on the binary outcome gives $\mathbb { E } [ Z ( o _ { B } - p ) ] = p ( 1 - p ) D$

The normal equations for Eq. (20) hold even when Σ is singular. Indeed, $w \in$ ker Σ implies $\mathbb { E } [ ( w ^ { \top } Z ) ^ { 2 } ] = 0$ , hence $w ^ { \top } \bar { Z } = 0$ almost surely. Both covariance vectors being projected are therefore in the range of Σ. Thus E $[ Z r _ { h } ] = \mathbb { E } [ Z r _ { o } ] = 0$ , and

$$
\mathbb { E } [ ( o _ { B } - p ) h ] = p ( 1 - p ) \boldsymbol { D } ^ { \top } \boldsymbol { a } + \mathbb { E } [ r _ { o } r _ { h } ] .\tag{25}
$$

Cauchy–Schwarz bounds the last expectation by $\epsilon _ { o } \epsilon _ { h }$ . The least-squares residual variance for $o _ { B } - p$ is $p ( \bar { 1 } ^ { \cdot } - p ) - \mathbb { E } [ Z ( o _ { B } - p ) ] ^ { \top } \Sigma ^ { \dagger } \mathbb { E } \big [ Z ( o _ { B } - \bar  p _ { \} } ) \big ]$ , which gives Eq. (21). Finally, $v _ { q } - v _ { 0 } = q \Delta v$ proves Eq. (22); multiplying by $p ( 1 - p )$ gives Eq. (9). A strictly positive derivative of the log-odds difference, which is zero at $\eta = 0$ , makes that difference positive for sufficiently small positive $\eta .$ The logit is strictly increasing. The negative case follows identically. □

## B.3 RETENTION AFTER RUBRIC WITHDRAWAL

Let τ mark the end of rubric exposure and T the final checkpoint, with success probability $p _ { t } =$ $p _ { B } ( \theta ( t ) )$ and its monotone transform $\alpha _ { t } = \arcsin \sqrt { p _ { t } }$ . The Fisher path length $\mathcal { P } _ { \tau : T }$ measures accumulated movement of the rubric-free response distribution over the remaining training.

Fisher Path Length and Probability Bounds. For the piecewise continuously differentiable path $\theta ( t )$ , define

$$
F ( \theta ) = \mathbb { E } _ { P _ { \theta } } [ \psi _ { 0 } \psi _ { 0 } ^ { \top } ] , \qquad \mathcal { P } _ { \tau : T } = \int _ { \tau } ^ { T } \sqrt { \dot { \theta } ( t ) ^ { \top } F ( \theta ( t ) ) \dot { \theta } ( t ) } d t .\tag{26}
$$

Proposition 3 (Retention after withdrawal) Along a piecewise continuously differentiable parameter path with $0 < p _ { t } < 1$ andfinite $\mathcal { P } _ { \tau : T ; \dag }$

$$
\begin{array} { r } { | \alpha _ { T } - \alpha _ { \tau } | \leq \frac 1 2 \mathcal { P } _ { \tau : T } , } \end{array}\tag{27}
$$

For a fixed comparison success rate $p _ { \star } \in ( 0 , p _ { \tau } )$ , set α = arcsin $\sqrt { p _ { \star } } . \ I f { \mathcal { P } } _ { \tau : T } < 2 ( \alpha _ { \tau } - \alpha _ { \star } )$ , then $p _ { T } > p _ { \star }$

Under the regularity assumptions of appendix B.1, Eq. (27) gives

$$
\begin{array} { r } { p _ { B } \big ( \theta ( T ) \big ) \geq \sin ^ { 2 } \left( \operatorname* { m a x } \Big \{ 0 , \arcsin \sqrt { p _ { B } \big ( \theta ( \tau ) \big ) } - \frac { 1 } { 2 } \mathcal { P } _ { \tau : T } \Big \} \right) . } \end{array}\tag{28}
$$

For a fixed comparison probability $p _ { \star } \in \mathsf { \Gamma } ( 0 , 1 )$ under the same prompt law and success rule, a sufficient condition for $p _ { B } ( \theta ( T ) ) > p _ { \star }$ is

$$
p _ { B } ( \boldsymbol { \theta } ( \tau ) ) > p _ { \star } , \qquad \mathcal { P } _ { \tau : T } < 2 \left( \arcsin \sqrt { p _ { B } ( \boldsymbol { \theta } ( \tau ) ) } - \arcsin \sqrt { p _ { \star } } \right) .\tag{29}
$$

Proof. Write $p _ { t } = p _ { B } ( { \boldsymbol { \theta } } ( t ) )$ and $h _ { t } = \langle \psi _ { 0 } , { \dot { \theta } } ( t ) \rangle$ ⟩, with $\psi _ { 0 }$ evaluated at $\theta ( t )$ . The score identity gives E $[ h _ { t } ] = 0$ and $\dot { p } _ { t } = \mathbb { E } [ ( o _ { B } - p _ { t } ) h _ { t } ]$ ]. Hence

$$
\begin{array} { r } { | \dot { p } _ { t } | \leq \sqrt { p _ { t } ( 1 - p _ { t } ) } \sqrt { \mathbb { E } [ h _ { t } ^ { 2 } ] } = \sqrt { p _ { t } ( 1 - p _ { t } ) } \sqrt { \dot { \theta } ( t ) ^ { \top } F ( \theta ( t ) ) \dot { \theta } ( t ) } . } \end{array}\tag{30}
$$

Since $\dot { \alpha } _ { t } = \dot { p } _ { t } / [ 2 \sqrt { p _ { t } ( 1 - p _ { t } ) } ]$ , integration on each smooth segment and the triangle inequality prove Eq. (27). Clipping its lower bound at zero and using monotonicity of $\sin ^ { 2 }$ on $[ 0 , \pi / 2 ]$ gives Eq. (28) and Eq. (29). For sharpness, consider one prompt and two responses, with success probabilities $( \sin ^ { 2 } \alpha , \cos ^ { 2 } \alpha )$ and $\alpha \in ( 0 , \pi / 2 )$ The Fisher information for parameter α is 4. Every monotone path then has $\mathcal { P } _ { \tau : T } = 2 | \alpha _ { T } - \alpha _ { \tau } |$ , achieving equality. □

For actual optimizer increments $\delta _ { j } = \theta _ { j + 1 } - \theta _ { j }$ , linear interpolation gives the exact path length

$$
\mathcal { P } _ { \tau : T } = \sum _ { j = \tau } ^ { T - 1 } \int _ { 0 } ^ { 1 } \sqrt { \delta _ { j } ^ { \top } F ( \theta _ { j } + s \delta _ { j } ) \delta _ { j } } d s .\tag{31}
$$

This form applies to a realized AdamW trajectory, including momentum and weight decay.

For the idealized mean-gradient path $\dot { \theta } \ : = \ : v _ { 0 } + q \Delta v$ the Fisher seminorm satisfies $\sqrt { { \dot { \theta } } ^ { \top } F { \dot { \theta } } } \leq$ $\sqrt { v _ { 0 } ^ { \top } F v _ { 0 } } + q \sqrt { \Delta v ^ { \top } F \Delta v } .$ with every term evaluated at the evolving policy.

## C SEMANTIC RUBRIC STEERING AND THE RUBRICOPT LOOP

## C.1 THE CRITERION BANK AND TRAINING-ONLY CONDITIONING

Appendix A.2 gives the frozen canonical names and Prefer/Avoid clauses used in formal training. For each selected factor, the renderer takes its current clauses in the order $f _ { 1 } - f _ { 8 }$ . The main method selects all eight factors, while the ablation selects the groups in Tab. 10. The renderer stays fixed across rounds:

Capability priorities: <factor A>, <factor B>   
Prefer:   
- <Prefer clause of factor A>;   
- <Prefer clause of factor B>.   
Avoid:   
- <Avoid clause of factor A>;   
- <Avoid clause of factor B>.

The Common rubric uses the $f _ { 2 }$ and $f _ { 5 }$ entries of appendix A.2; larger sets extend the same lists. With the Qwen3-4B tokenizer, Common, Common+Specific, Common+Low, and the full rubric cost 70, 151, 153, and 234 suffix tokens respectively. Gate-accepted edits change clauses, not the factor order or renderer.

The Development Pilot. The pilot compares ordinary DPO, a single-factor $f _ { 2 }$ criterion, and a single-factor $f _ { 7 }$ criterion while holding the pairs, labels, DPO objective, data order, and LoRA configuration fixed. It trains Qwen3-4B for 150 steps at a fixed conditioning rate $q = 0 . 2$ , so the $f _ { 2 }$ and $f _ { 7 }$ conditions condition exactly the same training rows and the comparison varies the criterion text alone. Six Ascend 910B1 NPUs with per-device batch 2 and accumulation 4 give 48 pairs per step and about 7,200 pairs over those 150 steps (0.20 epoch). Maximum prompt and sequence lengths are 1,024 and 2,048 tokens, in bf16 with FlashAttention-2, and prompt truncation retains the final 1,024 tokens, including the criterion suffix. Each run takes about 47 minutes.

Pilot evaluation uses its own sampling schedule. It draws 500 prompts per benchmark with 4 samples per prompt at temperature 0.7, top-p 0.8, and top-k 20. The unit of analysis is the prompt, and the paired tests are exact McNemar tests on the pass@4 indicator together with a two-sided sign-flip permutation test of 10,000 resamples (seed 20260904) on the paired per-prompt mean-correctness differences. The two criteria differ only in their text and are within one token of each other in length.

Under the paired protocol, the $f _ { 2 }$ criterion is favored over the $f _ { 7 }$ criterion on all four pass@4 comparisons (Tab. 5). The $f _ { 2 }$ criterion is ahead on GSM8K and on MuSR pass@4 (17 passes unique to $f _ { 2 }$ against 6 unique to $f _ { 7 } )$ , and MATH-Hard and BBH show the same direction. The $f _ { 7 }$ criterion shifts negatively relative to ordinary DPO on MuSR pass@4 (3 passes unique to $f _ { 7 }$ against 14 unique to ordinary DPO) and does not separate from ordinary DPO on the other three benchmarks. The $f _ { 2 }$ criterion and ordinary DPO do not differ on any of the four benchmarks, and ordinary DPO does not differ from the Instruct model on any of them. Changing the criterion text alone therefore moved the two criteria apart on MuSR pass@4, and the $f _ { 7 }$ criterion is the direction the diagnosis does not support. Across the four benchmarks, counts of prompts passing only under the first versus only under the second condition are [4/1, 4/2, 17/6, 23/17] for $f _ { 2 }$ versus $f _ { 7 }$ and [4/5, 2/3, 3/14, 15/13] for $f _ { 7 }$ versus ordinary DPO.

Table 5: Pilot accuracy and pass@4. Instruct is the untrained starting model. Trained conditions share pairs, labels, objective, data order, and LoRA configuration, differing only in criterion text. Bold marks row maxima, including ties.
<table><tr><td>Benchmark</td><td>Instruct</td><td>Ordinary DPO</td><td> $f _ { 2 }$  Criterion</td><td> $f _ { 7 }$  Criterion</td></tr><tr><td>GSM8K accuracy</td><td>0.8945</td><td>0.8955</td><td>0.9015</td><td>0.8910</td></tr><tr><td>GSM8K pass@4</td><td>0.928</td><td>0.928</td><td>0.932</td><td>0.926</td></tr><tr><td>MATH-Hard accuracy</td><td>0.9070</td><td>0.9105</td><td>0.9125</td><td>0.9095</td></tr><tr><td>MATH-Hard pass @4</td><td>0.964</td><td>0.964</td><td>0.966</td><td>0.962</td></tr><tr><td>MuSR accuracy</td><td>0.5870</td><td>0.5825</td><td>0.5795</td><td>0.5840</td></tr><tr><td>MuSR pass@4</td><td>0.706</td><td>0.708</td><td>0.708</td><td>0.686</td></tr><tr><td>BBH accuracy</td><td>0.0520</td><td>0.0535</td><td>0.0560</td><td>0.0495</td></tr><tr><td>BBH pass@4</td><td>0.142</td><td>0.126</td><td>0.142</td><td>0.130</td></tr></table>

## C.2 STAGES, CONTRACTS AND THE DETERMINISTIC GATE

Evidence Set and Engine. Fig. 2 in the main text gives the six-step workflow. The diagnostic stage reads item-level judge rationales for the all-eight-factor method against ordinary DPO and the Instruct model on Arena-Hard, MT-Bench (including a dedicated second-turn batch), and Wild-Bench, together with generated correctness items carrying a gold answer on GSM8K, MATH-Hard, and MMLU-Pro. That evidence is the first-round rubric-free benchmark evaluation output, read item by item, as described in Sec. 4. The analysis and proposal engine is the current policy itself, the alleight-factor merged checkpoint, so the model that reads the benchmark feedback is the model the feedback is about. The external anchors are the gold answers and benchmark outcomes, the fixed judge’s rationales and checklists, and the frozen $\bar { R } _ { 0 }$ text.

Five prompt stages process feedback from policy $\pi _ { r + 1 }$ , trained with the frozen round-r rubric $R _ { r }$ , in the order 1 (benchmark diagnostic), 2 (cross-benchmark synthesis), 3 (factor attribution), 4 (conservative update), and 5 (rubric audit), followed by step 6 (deterministic gate). Each stage is a separate call with its own schema, and no stage sees the next stage’s output.

Stage 1 sees the factor schema, the criterion currently in force, and one benchmark’s evidence, and returns a four-key YAML document:

```yaml
preserve:
- behavior: / factor: / evidence:
improve:
```

```yaml
- failure mode: / factor: / evidence:
criterion gaps:
- factor: / current problem: / suggested direction:
overall diagnosis:
```

Stage 1 uses only f<sub>1</sub>–f<sub>8</sub> and separates behaviors to preserve from those to improve. An isolated example requires strong evidence, and attributing a benchmark change to the criterion requires behavioral evidence. The output must be concrete and concise.

Stage 2 aggregates every stage-1 report for one policy. It may not rewrite the criterion or propose wording. It organises evidence and records disagreement, adding the benchmarks that support each pattern, an evidence-strength level (strong, moderate, weak), and an explicit record of contradictions. A benchmark where the policy did not change still counts as evidence about what is stable.

Stage 3 attributes each recurring behaviour onto the ontology and decides whether the current wording already covers it. The coverage decision follows this rule:

If the current Prefer/Avoid wording states this failure mode in full, the model simply did not learn it: the decision is KEEP. If the wording does not state the failure mode, or states only part of it, that is the only case in which REFINE may be considered.

Attribution disambiguates reasoning validity, verification and calibration, completeness, and instruction adherence, naming the primary factor when several are involved. It attributes only what the evidence supports.

Stage 4 proposes the smallest clause edits that address high-confidence recurring gaps while preserving factor identity and behaviors marked for preservation. Edits may clarify wording, narrow an overly broad reading, make an implicit requirement explicit, or refine a matched Avoid clause. Stage 4 prohibits examples, task-specific tricks, benchmark or dataset names, scores, numerical weights, and intensifiers. Unsupported changes return KEEP. The output is a candidate $R _ { r + 1 }$ and a per-factor ledger with the action, reason, and old and new clauses.

Stage 5 audits semantic identity, benchmark and dataset leakage, evidential support, specificity, Prefer/Avoid matching, preserved behaviors, and edit magnitude. It emits a justification per axis and one PASS or FAIL. The audit is advisory rather than decisive: in one executed round it returned a clean eight-of-eight PASS on a candidate that the deterministic gate then rejected.

Step 6: The Deterministic Gate. The binding checks constrain eligibility, semantic identity, and procedural consistency.

Acceptance is decided by eight checks in round 1 and ten in round 2. Round 2 adds that only gate-accepted factors may differ, and that the recorded decisions are consistent with the rendered rubric.

1. All eight factors must occur exactly once, each with a KEEP/REFINE decision.

2. The input clauses of every refinement must match the   
frozen canonical wording byte-for-byte.

3. Each new clause must retain at least 0.60 of the old clause’s content tokens, with its length change inside [−5, +10] words.

4. Every factor must retain one matched Prefer/Avoid pair.

5. Clauses must contain no benchmark or dataset names, digits, intensifiers, or example markers.

6. Every refinement must be traceable to an attribution judging the failure not fully covered.

7. Edit magnitude must remain within the specified budget.

8. Cross-factor vocabulary imports raise a warning rather than a rejection.

9. In round 2, only gate-accepted factors may differ.

10. In round $^ { 2 , }$ the recorded decisions must agree with the rendered rubric.

Failures of the binding checks reject the affected edit, so a factor that violates a check is reverted to KEEP.

## C.3 MATCHED RE-OPTIMIZATION AND THE DATASET CONTROL

The shared recipe is in Sec. 4: rank-64 LoRA (α = 128, dropout 0.05), mean-per-token DPO with $\beta = 1$ at learning rate $1 0 ^ { - 6 }$ , global batch 64, and seed 42. The formal runs and the HelpSteer3 continuations take 566 steps. The HelpSteer3 continuations start from the frozen RUBRICOPT-R1 checkpoint and use fresh AdamW and FP32 LoRA adapters, cosine decay with 5% warmup, and one shared exposure mask on identical ordered pairs. Their reference is the frozen RUBRICOPT-R1 checkpoint with the new adapter disabled, and the starting policy of every other block is the released Instruct checkpoint with its own adapter disabled. Only the rubric text differs: KEEP uses $R _ { 0 }$ and RE-FINE uses $R _ { 1 }$ . The primary comparison RUBRICOPT-R2 (REFINE) minus RUBRICOPT-R2 (KEEP) measures editing; comparison with RUBRICOPT-R1 includes further training.

Tulu3 uses the same initial rubric and training recipe on an independently assembled dataset, over 1,818 steps at global batch 64. The budgets match ordinary DPO within each dataset by epoch count, not across datasets by step count. The Tulu3 second-round change combines editing and additional training.

## D EXPERIMENTAL SETTINGS AND RESULTS

This appendix gives the measurement and protocol detail behind the main-text results: evaluation protocols, training-protocol audits, and the per-cell measurements that the main tables summarize.

## D.1 EVALUATION AND MEASUREMENT PROTOCOLS

Splits and Evaluation Protocols. The factor measurements use diagnostic and held-out subsets assigned by an md5 hash of the prompt identifier, and comparing these subsets tests the stability of the measured demand gaps. Evaluation prompts come from benchmark sources, while the preference-training pairs come from HelpSteer3. Rubric refinement uses first-round benchmark evaluation outputs as feedback and compares the continuation branches on matched evaluation items (Sec. 4). The formal study uses the released benchmarks through the standard evaluation harness, with adapter evaluation enabled and the adapter merged, and the reported baselines use this protocol (Tabs. 1 and 10). The demand values in Tab. 7 are for Qwen3-4B.

Open-Ended Protocols. Arena-Hard and WildBench use greedy decoding with a 4,096-token generation limit and one response per prompt. WildBench uses 256 prompts and MiniMax-M3 scores on a 1–10 scale, averaged over usable verdicts (255 for Qwen3-8B RUBRICOPT-R1, 256 for the other reported conditions). MT-Bench uses a 1,024-token generation limit. Every Arena-Hard condition is judged against the fixed answers of the standard Arena-Hard anchor (GPT-4-0314), with the candidate always in slot B, so Arena-Hard here is a fixed-anchor win rate: every method faces the same anchor on the same items, rather than a head-to-head between methods. Per-item outcomes are wins, ties, or losses, with ties counted as half-wins, and all conditions use the same tie rule, anchor, and items.

Accuracy Evaluation Protocols. Absolute scores are reported in Tabs. 1 to 3 and 10. BBH covers all 27 subtasks in the project’s sampled evaluation. The paired differences in Tab. 9 use the same (prompt id, sample idx) keys within each model–dataset cell. MATH-Hard uses the lm-eval task template with 7 subtasks. Scores are the unweighted mean exact match over subtasks under vLLM/bf16 with merged adapters.

Training-Protocol Details. The trainer computes Eq. (5) with each response’s own unpadded token count. Policy and frozen reference receive identical batch tensors and context; reference logprobabilities are recomputed with the adapter disabled and no cache. Rubric tokens remain in the masked prompt region and enter neither response-token numerator nor denominator, and the chosen and rejected response text is unchanged by conditioning.

Matched-Length Criterion Comparison. At formal scale, Common+Specific and Common+Low vary the criterion text, and different additions to Common. Common+Specific exceeds Common+Low by 0.07 BBH points, 0.01 points on the seven-task average, and 0.5 Arena-Hard points.

Formal Runs. Formal runs use eight Ascend NPUs under distributed data parallel. The realized conditioning exposure is 0.5009 on HelpSteer3 and 0.4971 on the Tulu3 conditioned runs, and the criterion is withdrawn for the final 10% of training and for main evaluation.

Inference-Time Rubric Reinsertion. The same Qwen3-4B RUBRICOPT-R2 checkpoint is evaluated with no rubric, the trained wording $R _ { 1 } { \mathrm { . } }$ , and the untrained candidate $R _ { 2 }$ . The rubric is inserted at the last message before templating, adding 243 and 256 prompt tokens respectively. Tab. 6 reports six benchmarks generated in one matched batch window, with identical (prompt id, sample idx) sets of 2,000 rows per benchmark. On MATH-Hard, OFF scores 63.17%, $R _ { 1 }$ ON 63.85%, and $R _ { 2 }$ ON 63.93%.

Table 6: Rubric reinsertion on the same checkpoint and matched sampled items (accuracy fractions).
<table><tr><td>Condition</td><td>GSM8K</td><td>MMLU-Pro</td><td>TruthfulQA</td><td>ARC</td><td>BBH</td><td>MuSR</td></tr><tr><td>OFF</td><td>0.8995</td><td>0.6390</td><td>0.6955</td><td>0.9070</td><td>0.4310</td><td>0.5755</td></tr><tr><td> $R _ { 1 }$  ON</td><td>0.8960</td><td>0.6205</td><td>0.6895</td><td>0.8945</td><td>0.4185</td><td>0.5640</td></tr><tr><td> $R _ { 2 }$  ON</td><td>0.9055</td><td>0.6285</td><td>0.6905</td><td>0.9000</td><td>0.4680</td><td>0.5775</td></tr></table>

The Demand Matrices. Tab. 7 reports the full Qwen3-4B demand matrix with effect sizes. The largest diagnostic–held-out gap difference is 0.045, with no sign reversal. The 8B matrix is in Tab. 11.

Factual and evidential grounding separates success from failure more strongly on the knowledge benchmarks than on the reasoning benchmarks (0.111 against 0.031, Fig. 3b). Within the knowledge group, ARC is driven by reasoning validity, TruthfulQA by verification and factual grounding, and MMLU-Pro by completeness. MuSR is close to flat on every factor. A deployment target usually contains several benchmarks with different semantic needs, and how they are combined changes which factor appears most important. On the knowledge group the equal-per-benchmark average puts reasoning validity first (0.213) and completeness second (0.193), whereas pooling by response count reverses the order, because MMLU-Pro supplies about 86% of the responses in the group, raising completeness to 0.356. The per-benchmark profiles are kept. The equal-per-benchmark average is a descriptive summary.

Model-Dependent Capability Signatures. Demand changes with the evaluated model as well as with the task. The Qwen3-4B and Qwen3-8B profiles correlate strongly on TruthfulQA $( r = 0 . 9 5 3 )$ GSM8K $( r = 0 . 9 0 6 )$ , and MMLU-Pro $( r = 0 . 8 6 8 )$ , moderately on MuSR $( r = 0 . 5 2 2 )$ , MATH-Hard $( r = 0 . 4 6 1 )$ , and ARC (r = 0.457), and negatively on BBH $( r = - 0 . 7 5 1 )$ . The disagreement is not confined to one factor (Fig. 4a): on BBH, $f _ { 5 } , f _ { 7 } , f _ { 6 } ,$ , and $f _ { 2 }$ all change sign. The $f _ { 5 }$ gap changes from +0.232 on Qwen3-4B to −0.100 on Qwen3-8B, and the $f _ { 7 }$ gap from +0.235 to −0.171. The demand scope changes as well, and that matters more for a diagnosis than any single gap: reasoning validity is meaningful on five of the seven benchmarks for Qwen3-4B but on only one for Qwen3-8B, verification and calibration rises from none to two, and clarity and organization remains meaningful on two (Tab. 11). A priority read off the weaker model would not describe the stronger one, which is consistent with a fixed ontology and a changing capability signature rather than one universal factor ranking, and it makes the grouping of Sec. 3.1 specific to Qwen3-4B.

Demand and Supply. Reasoning validity $( f _ { 2 } )$ is meaningful on five benchmarks but has the lowest coverage (0.146), while completeness $( f _ { 5 } )$ is meaningful on four and has the highest coverage

Supply Detail and Length Matching. Tab. 8 reports mean contrasts, coverage, positive fractions, and tie rates. Negative contrasts are a minority for every factor (at most 0.171), the contrast SDs lie between 0.476 and 0.835, and Cohen’s $d _ { z }$ between 0.089 and 0.193.

Table 7: Qwen3-4B diagnostic matrix: mean success–failure gaps on the 0–4 judge scale, followed by Cohen’s d. Bold shaded gaps are at least 0.15.
<table><tr><td colspan="8">Mean factor-score gap</td></tr><tr><td>Factor</td><td>GSM8K</td><td>MATH-Hard</td><td>BBH</td><td>MuSR</td><td>ARC</td><td>TruthfulQA</td><td>MMLU-Pro</td></tr><tr><td> $f _ { 1 }$ </td><td>+0.080</td><td>+0.039</td><td>+0.023</td><td>+0.005</td><td>+0.110</td><td>+0.100</td><td>+0.069</td></tr><tr><td> $f _ { 2 }$ </td><td>+0.170</td><td>+0.246</td><td>+0.066</td><td>+0.047</td><td>+0.248</td><td>+0.157</td><td>+0.235</td></tr><tr><td> $f _ { 3 }$ </td><td>-0.051</td><td>+0.090</td><td> $+ 0 . 0 1 8$ </td><td>+0.025</td><td>+0.105</td><td>+0.145</td><td>+0.142</td></tr><tr><td> $f _ { 4 }$ </td><td>+0.036</td><td>+0.054</td><td> $+ 0 . 0 2 8$ </td><td>+0.006</td><td>+0.093</td><td>+0.150</td><td>+0.091</td></tr><tr><td> $f _ { 5 }$ </td><td>+0.158</td><td>+0.486</td><td> $\mathbf { + 0 . 2 3 2 }$ </td><td>+0.038</td><td>+0.148</td><td>+0.034</td><td>+0.399</td></tr><tr><td> $f _ { 6 }$ </td><td>+0.086</td><td>+0.065</td><td> $+ 0 . 0 7 5$ </td><td>+0.010</td><td>-0.007</td><td>+0.035</td><td>+0.038</td></tr><tr><td> $f _ { 7 }$ </td><td>-0.048</td><td>+0.164</td><td>+0.235</td><td>+0.069</td><td> $+ 0 . 0 4 0$ </td><td>-0.020</td><td>+0.132</td></tr><tr><td> $f _ { 8 }$ </td><td>+0.081</td><td>+0.162</td><td> $+ 0 . 0 9 5$ </td><td> $+ 0 . 0 6 9$ </td><td> ${ \bf + 0 . 1 7 9 }$ </td><td>+0.110</td><td>+0.219</td></tr></table>

<table><tr><td>Factor</td><td>GSM8K</td><td>MATH-Hard</td><td>BBH</td><td>MuSR</td><td>ARC</td><td>TruthfulQA</td><td>MMLU-Pro</td></tr><tr><td> $f _ { 1 }$ </td><td>+0.578</td><td>+0.326</td><td>+0.103</td><td>+0.021</td><td>+0.680</td><td>+0.280</td><td>+0.275</td></tr><tr><td> $f _ { 2 }$ </td><td>+0.756</td><td>+0.709</td><td>+0.160</td><td>+0.123</td><td>+0.988</td><td>+0.362</td><td>+0.485</td></tr><tr><td> $f _ { 3 }$ </td><td>-0.112</td><td>+0.167</td><td>+0.035</td><td>+0.050</td><td>+0.210</td><td>+0.288</td><td>+0.283</td></tr><tr><td> $f _ { 4 }$ </td><td>+0.319</td><td>+0.313</td><td> $+ 0 . 1 2 4$ </td><td>+0.049</td><td>+0.570</td><td>+0.327</td><td>+0.256</td></tr><tr><td> $f _ { 5 }$ </td><td>+0.588</td><td>+1.105</td><td>+0.408</td><td>+0.069</td><td>+0.297</td><td>+0.065</td><td>+0.700</td></tr><tr><td> $f _ { 6 }$ </td><td>+0.270</td><td>+0.267</td><td>+0.200</td><td>+0.038</td><td>-0.023</td><td>+0.081</td><td>+0.122</td></tr><tr><td> $f _ { 7 }$ </td><td>-0.093</td><td>+0.315</td><td> $+ 0 . 4 4 7$ </td><td> $+ 0 . 1 4 7$ </td><td>+0.090</td><td>-0.043</td><td>+0.278</td></tr><tr><td> $f _ { 8 }$ </td><td>+0.222</td><td> $+ 0 . 3 6 6$ </td><td> $+ 0 . 1 9 6$ </td><td> $+ 0 . 1 3 6$ </td><td> $+ 0 . 3 1 7$ </td><td>+0.232</td><td>+0.387</td></tr></table>

(0.474). The length controls below show that their supply contrasts have different sensitivity to response length.

Chosen responses are longer, so we also report a length-matched subset: pairs whose chosen and rejected responses differ by at most 164 characters, the lower-quartile bound of the length-difference distribution, giving $n = 9 , 5 5 1$ . For each factor we additionally fit $\Delta z _ { k } = \alpha + \gamma z ( \Delta L )$ over the full set and report γ per standard deviation of the length difference. Across the factors we examined, γ is 0.040 for reasoning validity, 0.021 for factual grounding, 0.175 for completeness, 0.105 for clarity, and −0.177 for efficiency.

Table 8: HelpSteer3 supply statistics over all assessable pairs: mean signed contrast $\mu _ { k } .$ , contrast coverage $C _ { k } = \mathrm { P r } ( | \delta | \geq 1 )$ , and positive fraction $P _ { k } = \mathrm { P r } ( \delta > 0 )$ . Shading marks $C _ { k } < 0 . 2 5$ Reduction is the relative percentage drop in mean contrast after length matching, where reported.
<table><tr><td>Factor</td><td> $\mu _ { k }$ </td><td> $C _ { k }$ </td><td> $P _ { k }$ </td><td>Tie rate</td><td>Reduction</td></tr><tr><td> $f _ { 1 }$  task comprehension</td><td>0.1042</td><td>0.3379</td><td>0.2118</td><td>0.6621</td><td></td></tr><tr><td> $f _ { 2 }$  reasoning validity</td><td>0.0908</td><td>0.1463</td><td>0.1032</td><td>0.8537</td><td>13%</td></tr><tr><td> $f _ { 3 }$  verification and calibration</td><td>0.0818</td><td>0.2202</td><td>0.1506</td><td>0.7798</td><td></td></tr><tr><td> $f _ { 4 }$  factual and evidential grounding</td><td>0.0866</td><td>0.1493</td><td>0.0992</td><td>0.8507</td><td></td></tr><tr><td> $f _ { 5 }$  completeness</td><td>0.1556</td><td>0.4743</td><td>0.3037</td><td>0.5257</td><td>38%</td></tr><tr><td> $f _ { 6 }$  instruction adherence</td><td>0.0988</td><td>0.3405</td><td>0.2046</td><td>0.6595</td><td></td></tr><tr><td> $f _ { 7 }$  clarity and organization</td><td>0.1428</td><td>0.3950</td><td>0.2611</td><td>0.6050</td><td>47%</td></tr><tr><td> $f _ { 8 }$  efficiency and conciseness</td><td>0.0657</td><td>0.3248</td><td>0.1850</td><td>0.6752</td><td></td></tr></table>

The 38,173 assessable pairs include 1,917 held out from training, and the same coverage values are reused in the 8B diagnosis. All factors have more positive than negative contrast mass, but $f _ { 2 }$ and

$f _ { 4 }$ tie on about 85% of pairs. 59.2% of pairs have a longer chosen response, with mean +180 and median +106 characters. On the length-matched subset $( | \Delta L | \le 1 6 4 , n = 9 , 5 5 1 , \mathrm { F i g . \ 5 ) }$ , mean contrasts change from 0.156 to 0.097 for $f _ { 5 } , 0 . 1 4 3$ to 0.076 for f , 0.091 to 0.079 for $f _ { 2 } ,$ and 0.087 to 0.094 for $f _ { 4 } .$ Among these four factors, reasoning validity and factual grounding change less than completeness and clarity. The length slope is $\gamma = + 0 . 1 7 5$ for $f _ { 5 } , + 0 . 1 0 5$ for $\bar { f } _ { 7 } , - 0 . \bar { 1 } 7 7$ for $f _ { 8 } ,$ 0.040 for $f _ { 2 } ,$ , and 0.021 for $f _ { 4 }$ . This control assesses the length sensitivity of the supply diagnostic; the effect of rubric self-revision is evaluated by the matched continuations in Tab. 3.

![](images/54cf9f3613ca6ee5af035d0d5011c7bfc33b9a196d83201ee2ef05ed81f86b01.jpg)  
Figure 5: Length sensitivity of HelpSteer3 supply estimates. Points show mean chosen–rejected gaps for four factors on the 0–4 judge scale, using all 38,173 pairs or 9,551 length-matched pairs $( | \Delta L | \leq 1 6 4 $ characters). $\Delta$ is matched minus all.

## D.2 PAIRED EFFECTS AND RESULTS ACROSS DATASETS AND MODELS

Tab. 9 reports BBH contrasts from the full-subtask evaluation. MATH-Hard changes use the separate greedy protocol of appendix D.1.

Table 9: Full-subtask BBH differences within each model–dataset cell (accuracy fractions). Shading marks the matched REFINE-minus-KEEP comparison.
<table><tr><td>Cell</td><td>Comparison</td><td></td><td>△ Reading</td></tr><tr><td>4B × HS3</td><td>Common – Ordinary DPO</td><td>+0.0090 positive</td><td></td></tr><tr><td></td><td>Common+Specific — Ordinary DPO</td><td>+0.0047</td><td>positive</td></tr><tr><td></td><td>Common+Low – Ordinary DPO</td><td>+0.0040</td><td>positive</td></tr><tr><td></td><td>RUBRICOPT-R1 – Ordinary DPO</td><td>+0.0023</td><td>null</td></tr><tr><td></td><td>RUBRICOPT-R2(KEEP) – RUBRICOPT-R1</td><td>+0.0309</td><td>positive</td></tr><tr><td></td><td>RUBRICOPT-R2(REFINE) - RUBRICOPT-R2(KEEP) -0.0025 null</td><td></td><td></td></tr><tr><td>4B × Tulu3</td><td>RUBRICOPT-R1 – Ordinary DPO</td><td>+0.0420 positive</td><td></td></tr><tr><td>8B × Tulu3</td><td>RUBRICOPT-R1 – Ordinary DPO</td><td>+0.0961 positive</td><td></td></tr></table>

Per-Benchmark Differences and the Matched Continuation. Five comparisons are fixed in advance, each holding pairs, labels, objective, budget, and schedule fixed: A (Common minus ordinary DPO) asks whether the broadly demanded factors help, B (Common + Specific minus Common) measures the additional specific factors, C (Common + Low minus Common) measures the additional low-demand factors, D (full rubric minus Common + Specific) measures adding Low once Specific is present, and E (Common + Specific minus Common + Low) compares the two addi tions. The absolute scores for these comparisons are in Tab. 10. Reported per-benchmark differences range from −0.53 to +0.90 percentage points. The corresponding changes in the unweighted seven-benchmark mean are +0.08, −0.02, −0.03, +0.12, and +0.01 points for A–E, respectively.

Table 10: Per-benchmark factor-set results on Qwen3-4B with HelpSteer3. Accuracy and Arena-Hard win rate are percentages. Avg. is the unweighted seven-task accuracy mean. Open-ended protocols use native scores. Avg. $\Delta$ is the relative change against ordinary DPO across MT-Bench, Arena-Hard, and WildBench. Each rubric includes Common, with Specific and/or Low-demand factors added. Bold marks column maxima. Shading marks rubric-trained rows.
<table><tr><td rowspan="3"></td><td colspan="8">HelpSteer3 comprehensive capability (%)</td><td colspan="4">HelpSteer3 open-ended generation</td></tr><tr><td colspan="2">MATH-</td><td colspan="2"></td><td colspan="4"></td><td rowspan="2">MT-</td><td rowspan="2">Arena- Hard</td><td rowspan="2">Wild- Bench</td><td rowspan="2">Avg. ∆</td></tr><tr><td></td><td>GSM8K</td><td>Hard</td><td>BBH MuSR</td><td>ARC</td><td>Truthful MMLU- QA</td><td>Pro</td><td>Avg. Bench</td></tr><tr><td>Qwen3-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Instruct</td><td>85.90</td><td>62.24</td><td>12.8243.39</td><td></td><td>62.80</td><td>52.27</td><td>41.98</td><td>51.63</td><td>6.320</td><td>49.0%</td><td>3.730</td><td>-2.96</td></tr><tr><td>Ordinary DPO</td><td>87.57</td><td>63.64</td><td>14.73</td><td>43.12</td><td>62.63</td><td>53.91</td><td>42.30</td><td>52.56</td><td>6.890</td><td>50.9%</td><td>3.617</td><td>0.00</td></tr><tr><td>Common</td><td>87.04</td><td>63.48</td><td>15.64</td><td>43.12</td><td>63.23</td><td>53.72</td><td>42.23</td><td>52.64</td><td>6.720</td><td>52.3%</td><td>3.711</td><td>+0.96</td></tr><tr><td>Common+Specific</td><td>87.04</td><td>63.74</td><td>15.21</td><td>42.99 </td><td>63.40</td><td>53.71</td><td>42.25</td><td>52.62</td><td>6.610</td><td>53.1%</td><td>3.840</td><td>+2.14</td></tr><tr><td>Common+Specific+Low</td><td>87.49</td><td>64.18</td><td>14.96</td><td>43.25</td><td>63.31</td><td>53.74</td><td>42.25</td><td>52.74</td><td>6.930</td><td>53.0%</td><td>3.727 +2.58</td><td></td></tr><tr><td>Common+Low</td><td>86.88</td><td>64.02</td><td>15.14</td><td>43.12</td><td>63.14</td><td>53.69</td><td>42.26</td><td>52.61</td><td>6.900</td><td>52.6%</td><td>3.602 +1.02</td><td></td></tr></table>

Tab. 3 reports the starting policy and both HelpSteer3 continuations. The shared configuration in appendix C.3 makes RUBRICOPT-R2 (REFINE) minus RUBRICOPT-R2 (KEEP) the edit comparison. Besides the main-text BBH and MT-Bench changes, the edit changes MATH-Hard by −0.10 points and WildBench by +0.055 native-score units.

The Qwen3 and Mistral benchmark suites are in Tabs. 1 and 2; factor-set results are in Tab. 10. INSTRUCT is the released checkpoint every method starts from, DPO is rubric-free preference optimisation on the same pairs, and RUBRICOPT-R1 is the policy after the first RubricOpt round, trained under the initial rubric $R _ { 0 }$ over all eight factors. RUBRICOPT-R2 (KEEP) and RUBRICOPT-R2 (REFINE) are the two matched second-round branches: both start from the same frozen RUBRICOPT-R1, run one epoch over the same pairs with the same exposure mask and opti miser, and differ only in whether the rubric is the unchanged $R _ { 0 }$ or the refined $R _ { 1 }$

The Second Dataset. The Tulu3 refined continuation starts from the merged RUBRICOPT-R1 policy and changes $f _ { 3 } , f _ { 4 } ,$ , and $f _ { 8 } .$ . Its absolute scores are in Tab. 1, and the continuation changes MATH-Hard accuracy from 43.35% to 37.12%, a −6.23-point joint change from additional training and editing.

The Second Model. Tabs. 7 and 11 report the demand gaps for the two scales. The 8B block trains an ordinary-DPO method and a RUBRICOPT-R1 method on the first dataset, 566 steps each from the released checkpoint at global batch 64 with the same LoRA and seed, differing only in the rubric block. Both methods are evaluated on all seven benchmarks, and the conditioned method has 14,000 factor-scored responses.  
Table 11: Qwen3-8B success–failure gaps on the 0–4 judge scale, pooling diagnostic and held-out halves within the model. Bold shaded cells have |gap| ≥ 0.15. The Qwen3-4B matrix is in Tab. 7.
<table><tr><td>Benchmark</td><td> $f _ { 1 }$ </td><td> $f _ { 2 }$ </td><td> $f _ { 3 }$ </td><td> $f _ { 4 }$ </td><td> $f _ { 5 }$ </td><td> $f _ { 6 }$ </td><td> $f _ { 7 }$ </td><td> $f _ { 8 }$ </td></tr><tr><td>GSM8K</td><td>+0.113</td><td>+0.130</td><td>-0.096</td><td>+0.014</td><td>+0.189</td><td>+0.058</td><td>-0.065</td><td>+0.171</td></tr><tr><td>MATH-Hard</td><td>+0.011</td><td>+0.079</td><td>+0.182</td><td>+0.013</td><td>+0.144</td><td>+0.058</td><td>+0.163</td><td>+0.080</td></tr><tr><td>BBH</td><td>-0.038</td><td>-0.007</td><td>+0.142</td><td>+0.001</td><td>-0.100</td><td>-0.028</td><td>-0.171</td><td>+0.073</td></tr><tr><td>MuSR</td><td>+0.022</td><td>+0.042</td><td>+0.039</td><td>+0.012</td><td>+0.042</td><td>+0.022</td><td>+0.038</td><td>+0.024</td></tr><tr><td>ARC</td><td>+0.072</td><td>+0.167</td><td>+0.109</td><td>+0.089</td><td>+0.141</td><td>+0.033</td><td>-0.006</td><td>-0.035</td></tr><tr><td>TruthfulQA</td><td>+0.016</td><td>+0.053</td><td>+0.078</td><td>+0.100</td><td>-0.017</td><td>-0.031</td><td>-0.131</td><td>+0.039</td></tr><tr><td>MMLU-Pro</td><td>+0.057</td><td>+0.136</td><td>+0.190</td><td>+0.059</td><td>+0.521</td><td>+0.092</td><td>+0.183</td><td>+0.137</td></tr></table>

Mistral Evaluation. Mistral-7B uses the same recipe, with Tulu3 MATH-Hard and BBH read through the extraction ladder rather than the Qwen3 protocols. Tab. 2 reports both suites. BBH falls by 7.95 points against its own ordinary-DPO control, with the seven-task average unchanged.