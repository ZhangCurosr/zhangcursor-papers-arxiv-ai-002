# When Agents Disagree: Bayesian Backward Reasoning as a Label-Free Anchor for Multi-Agent Collective Decision-Making

Ken Chen<sup>1</sup> Wei Wang<sup>1</sup> Sachith Seneviratne<sup>1</sup> Hansani Weeratunge<sup>2</sup> Saman Halgamuge<sup>1</sup>

<sup>1</sup>Department of Mechanical Engineering, The University of Melbourne, Melbourne, Australia <sup>2</sup>Department of Mechanical Engineering, Sri Lanka Institute of Information Technology, Sri Lanka kenchen1@student.unimelb.edu.au

## Abstract

When multiple LLM agents yield conflicting answers, the decision-making process dictates whether agent diversity improves performance or merely compounds shared errors. Existing collective decision-making methods, including voting, electoral rules, and LLM judges, rely on forward reasoning: they map evidence to la bels in one direction. Although these methods can combine diverse forward traces, they still aggregate estimates that share this evidence-tolabel factorization and can inherit correlated errors within the forward pool. We therefore construct a reverse posterior for each instance through Bayesian backward reasoning from an explicit likelihood. The forward and reverse posteriors provide differently factorized approximations of the underlying posterior. Because estimates from different factorizations may tend to share the same error less often, we use Jensen-Shannon divergence to rank agents by cross-path consistency. This cross-path con sistency signal underlies three strategies: hard selection (MinJS), soft reweighting (FwdJS), and log-linear fusion (LogLin). Evaluated on DDXPlus across five LLM backbones, our pro posed strategies show consistent improvements: MinJS outperforms random selection across all backbones, FwdJS generally improves over the strongest baseline, and LogLin achieves the best performance among the evaluated meth ods, with its largest gains on the subset where the agents disagree. Despite its weaker stan dalone accuracy, the reverse posterior serves as a more useful anchor than forward-only alter natives, providing complementary information for collective decision-making. When labeled data are available, a lightweight two-stage cal ibration can further refine the reverse anchor and improve aggregation performance.

## 1 Introduction

Large Language Model (LLM)-based multi-agent systems coordinate multiple agents to collectively solve complex tasks. One prominent paradigm extends single-model capabilities by prompting heterogeneous agents to address the same query, shifting the focus toward collaborative decision-making, where diverse outputs must be aggregated into a final consensus. This paradigm is rooted in human collective intelligence (Woolley et al., 2010; Zhou et al., 2026) and cognitive diversity, positing that a team of diversely skilled solvers can outperform a uniform group of top individuals (Hong and Page, 2004). However, diversity alone does not guarantee superior group performance. When agents disagree, the aggregation mechanism ultimately determines whether complementary insights are synthesized or shared errors are compounded.

Existing collective decision-making pipelines typically rely on voting-based mechanisms (Zhao et al., 2024) or a designated LLM judge (Zhao et al., 2026). However, these approaches present distinct limitations. Voting and electoral rules-based methods rely on simple consensus and lack a reliable, objective anchor to resolve conflicts when agents disagree. Conversely, dictatorial approaches such as LLM-as-a-judge attempt to supply an external decision, but the evaluation remains grounded in the same forward-generated traces. Furthermore, they have also been observed to exhibit systematic biases, such as position bias, verbosity bias, or self-enhancement bias (Zheng et al., 2023). While multi-round debate can refine opinions (Du et al., 2024; Liang et al., 2024), it incurs high computational costs and remains vulnerable to conformity, identity-driven sycophancy, and persona instability (Baltaji et al., 2024; Choi et al., 2026; Li et al., 2026).

Focusing on single-round aggregation, we argue that the critical missing ingredient is a reliable external anchor, a reference distribution that guides the system in selecting and fusing conflicting answers. The root cause of aggregation failure is directional rather than merely algorithmic. Plurality voting, electoral rules, and LLM judges all operate exclusively withinforward reasoning: they reason directly from evidence to conclusions along the exact same conditioning path as the agents they evaluate. Consequently, errors within this pool are highly collinear. If the majority hallucinates, a vote or a judge is likely to echo the same mistake, trapping the system in a collective error. To break this cycle, the system requires a reference that resides in the same class-posterior space but is differently factorized, constructed via a path distinct from standard forward elicitation.

Bayes’ theorem provides two complementary factorizations of the same posterior. While forward reasoning executes one as a discriminative evidence-to-conclusion mapping, we explicitly construct the other through Bayesian backward reasoning. By defining an explicit likelihood $P ( e \mid d )$ over evidence e and labels $d \in \mathcal { V }$ , together with a prior, we invert the inference once per instance, a process of Bayes-style backward reasoning that yields a shared reverse posterior $R ( d \mid x )$ , where x denotes the observed input. Consequently, each agent’s forward posterior $F _ { i }$ and R serve as differently factorized, biased approximations of the same class posterior $P ( d \mid x )$ . Our working hypothesis is that their respective errors are far less collinear than those strictly within the forward pool. Because biased estimates derived from different factorizations are less likely to converge on the same incorrect conclusion than estimates from the same factorization, cross-path agreement provides a useful consistency signal for identifying more reliable predictions. Therefore, we utilize the Jensen-Shannon divergence, $\operatorname { J S } ( F _ { i } , R )$ , as a ranking signal for cross-path consistency rather than a definitive certificate of correctness. This symmetric metric treats neither channel as absolute ground truth, and its bounded nature ensures comparability across instances. Crucially, measuring distance to the pool’s internal consensus, such as the majority vote or the average prediction, cannot fulfill this role: it merely rewards conformity, actively penalizing the rare agent that happens to be correct when the majority is wrong.

Figure 1 summarizes our reverse-anchored framework: a shared reverse posterior serves as a reference for selecting, reweighting, and fusing the forward agents. Concretely, a single backward inference generates the shared reverse posterior R via explicit Bayesian backward reasoning over the candidate shortlist proposed by the agents’ forward reasoning, while each $F _ { i }$ represents an agent’s forward posterior. Unlike existing backward checks that merely score individual reasoning chains in math problems (Weng et al., 2023; Jiang et al., 2024), our R serves as a complete class distribution used to evaluate and fuse multiple forward agents. We leverage the Jensen–Shannon divergence, $\mathrm { J S } ( F _ { i } , R )$ , through three progressive aggregation strategies:

• MinJS (Hard Selection): Selects the single forward agent closest to R. While the ranking is informative, hard selection can be brittle because cross-path agreement measures compatibility with R, not an absolute guarantee of correctness.

• FwdJS (Soft Reweighting): Softly reweights the forward pool based on their alignment with R. Crucially, R solely determines the aggregation weights, meaning the final prediction remains a pure mixture of the forward posteriors: R shapes the weights but contributes no probability mass of its own.

• LogLin (Log-Linear Fusion): Actively blends R into the final prediction as a multiplicative factor with a small fixed weight $( w _ { R } { = } 0 . 2 )$ . This combination captures complementary cross-path information that the first two methods miss, as they rely solely on the forward outputs.

## Our main contributions are summarized as follows:

1. Cross-path reverse anchor. We introduce a reverse posterior obtained by Bayesian backward reasoning that acts as a reliable, label-free reference for multi-agent decisionmaking. We demonstrate that $\operatorname { J S } ( F _ { i } , R )$ effectively ranks and routes conflicting agents based on cross-path consistency.

2. Label-free aggregation suite. We propose three training-free strategies (MinJS, FwdJS, and LogLin). Extensive experiments demonstrate the progressive effectiveness of our proposed suite: MinJS consistently outperforms the selection baseline, FwdJS improves over the strongest baseline in most settings, and LogLin achieves strictly superior performance against all baselines across all scenarios.

3. Anchor indispensability. We establish that despite often being the weakest standalone predictor, the reverse posterior R acts as an irreplaceable anchor. Replacing it with the poolmean forward posterior (MeanF) or a poolexternal general single-agent forward posterior (GenF) strictly degrades both FwdJS and LogLin, confirming the unique value of crosspath diversity.

4. Lightweight labeled calibration. When a limited labeled split is available, we apply a two-stage calibration that repairs the reverse anchor’s quality and further elevates the aggregation strategies.

## 2 Related Work

Multi-agent collective decision-making. LLM multi-agent systems commonly coordinate through debate, role specialization, or layered synthesis (Du et al., 2024; Liang et al., 2024; Wang et al., 2025). Once multiple agents produce answers to the same instance, the system must aggregate their individual decisions into a single collective outcome. Existing approaches largely do so either by aggregating agents’ ballots through majority voting or more elaborate electoral rules (Zhao et al., 2024; Ai et al., 2025), or by delegating the decision to a designated judge (Zheng et al., 2023; Liu et al., 2023; Chan et al., 2024; Zhao et al., 2026). Confidence-weighted consensus further refines the former through discussion (Chen et al., 2024), while recent analyses suggest that simple majority already captures much of the benefit attributed to multi-round debate (Choi et al., 2025). Despite these differences, both voting and judging derive their decision signal from the agents’ forward outputs. In contrast, we derive an instance-level reference from a generative reverse posterior, providing a training-free alternative to another vote or judge.

Forward–backward reasoning. A separate line of work uses backward reasoning to evaluate a proposed answer or label rather than to aggregate decisions from multiple agents. Self-Verification (Weng et al., 2023) introduces backward checks at inference time to assess candidate solutions by testing whether the candidate is consistent with the underlying conditions. FOBAR (Jiang et al., 2024) further combines Self-Consistencystyle forward answer votes (Wang et al., 2023) and backward probabilities through a geometric mean. RevThink (Chen et al., 2025) instead trains a model to internalize forward and backward reasoning, while inference still produces a single forward answer. These methods thus use reverse reasoning primarily as a candidate-level verification signal. Rather than using the reverse signal only to verify individual candidate-level predictions, we derive a full reverse posterior and use its cross-path consistency with the forward posteriors to guide selection, reweighting, and fusion across multiple agents.

Ensembling and unlabeled routing. Model ensembling and routing also combine information from multiple models, but they formulate the problem as prediction fusion or model selection rather than collective decision-making among explicit agents. LLM-Blender (Jiang et al., 2023) merges candidate answer texts. DeePEn (Huang et al., 2024) and PackLLM (Mavromatis et al., 2024) combine next-token predictions across models. SMOOTHIE (Guha et al., 2024) performs unlabeled routing by scoring each model’s sample-wise quality from the unlabeled outputs and sending the input to the highest-scoring model. These methods operate on forward predictions and use agreement or predictive compatibility as their quality signal. In contrast, we retain the multi-agent collectivedecision setting and introduce a reverse posterior as an external reference without relying on another forward reasoning agent or a learned judge.

## 3 Method

## 3.1 Collective decision over agent posteriors

Consider an instance x and a finite label set $\mathcal { V } ,$ shown at the left of Figure 1. K heterogeneous agents $\mathcal { A } = A _ { 1 } , \ldots , A _ { K }$ each answer the same x and return a forward posterior $F _ { i } ( d \mid x )$ over $d \in \mathcal { V }$ . Our goal is to make a collective decision from the agent posteriors $F _ { i _ { i } = 1 } ^ { K }$ . We consider two forms of collective decision: selection, which adopts the posterior of one agent, and fusion, which fuses the pool into a single posterior $P ( d \mid x )$ and predicts arg max<sub>d</sub> $P ( d \mid x )$ . Both are implemented within the same label-free and training-free inference framework.

## 3.2 Reverse posterior as a consistency anchor

The class label $d \in \mathcal { V }$ is latent; what differs across inference paths is how the evidence is conditioned on d. Forward agents elicit $F _ { i } ( d \mid x )$ through a discriminative route from the instance x to the label space. We instead construct a complementary reverse view by specifying how the observed evidence factorizes conditioned on d and then applying Bayes’ rule.

![](images/745e5ca1011bd16527dc89605e70523eac1fe8698110c3609fbfb0f90eab1982.jpg)  
Figure 1: Reverse-anchored aggregation, left to right. Input x is decomposed into contextual evidence a and remaining evidence e under the structure $a  d  e$ . The forward path (yellow) produces agent posteriors $F _ { i } .$ , while the reverse path (blue) produces a shared reverse posterior $R \propto P ( e \mid d ) P ( d \mid a )$ . The shared anchor signal (green), $D _ { i } = \mathrm { J S } ( F _ { i } , R )$ , guides all three heads: MinJS selects one $F _ { i } ,$ FwdJS reweights $\{ F _ { i } \}$ , and LogLin additionally incorporates R into the output. The dashed box denotes optional labeled calibration $R {  } R ^ { \prime }$ . The dashed blue arrow indicates the direct contribution of R to LogLin.

We decompose the input x into two evidence components, a and $e ,$ and posit the conditionalindependence structure $a \ \to \ d \ \to \ e .$ Here, a represents contextual evidence, while e provides additional evidence about the latent label d. This implies $P ( e \mid a , d ) = P ( e \mid d )$ , and Bayes’ rule gives

$$
P ( d \mid a , e ) \propto P ( e \mid d ) P ( d \mid a ) .\tag{1}
$$

We thus obtain a reverse likelihood $P ( e \mid d )$ and a label prior $P ( d \mid a )$ , and define the resulting reverse posterior

$$
R ( d \mid x ) \propto P ( e \mid d ) P ( d \mid a ) ,\tag{2}
$$

normalized over Y. Forward and reverse inference therefore provide two different conditionalization perspectives on the same latent label: direct evidence-to-label prediction versus generative inversion.

Neither path is expected to recover the exact posterior $P ( d \mid x )$ . LLM elicitation and model bias make both $F _ { i }$ and R biased approximations rather than ground-truth posteriors. In particular, no exact Bayes identity links an elicited forward posterior $F _ { i }$ to our constructed reverse posterior $R .$ The motivation for introducing R is weaker and does not require either estimate to be exact: forward agents can exhibit correlated errors due to shared model priors or reasoning patterns, while the reverse factorization provides a structurally different source of evidence. We therefore do not treat R as a ground-truth posterior, but as a structurally distinct consistency signal whose errors need not be fully aligned with those of the forward predictors. This motivates using R to rank or weight the forward posteriors rather than introducing another forward vote or judge.

We instantiate the reverse anchor with one shared reverse inversion per instance, which uses ordinal likelihood and context-activation maps together with two LLM elicitation steps over the agents’ top-k labels, followed by exact replay on the Bayesian network. The construction is an implementation of the reverse anchor rather than a requirement of the aggregation framework itself. For the label-free experiments, the resulting reverse distribution is used without any training data.

Because the forward and reverse estimates arise from different factorizations, their errors can be partially complementary. This motivates a productlike composition that emphasizes labels supported by both channels, without requiring the stronger assumption that either distribution is exact or that the two estimates are statistically independent.

## 3.3 Reverse-referenced selection and fusion

We measure the consistency between each forward posterior and the reverse anchor (cross-path consis-

tency) using the Jensen-Shannon divergence:

$$
\begin{array} { r l } & { D _ { i } = \mathrm { J S } ( F _ { i } , R ) , } \\ & { \quad \quad = \frac { 1 } { 2 } \mathrm { K L } ( F _ { i } \| M _ { i } ) + \frac { 1 } { 2 } \mathrm { K L } ( R \| M _ { i } ) , } \\ & { \quad M _ { i } = \frac { 1 } { 2 } ( F _ { i } + R ) . } \end{array}\tag{3}
$$

The reverse posterior R serves as a shared external reference, shown in green in Figure 1, and the same consistency score is reused by the three heads on the right: hard selection, soft reweighting, and forward–reverse fusion.

MinJS (hard selection). MinJS selects the agent whose posterior is most consistent with the reverse anchor:

$$
\begin{array} { c } { i ^ { \star } = \arg \underset { i } { \operatorname* { m i n } } D _ { i } , } \\ { \hat { d } = \arg \operatorname* { m a x } _ { d } F _ { i ^ { \star } } ( d ) . } \end{array}\tag{4}
$$

The output $\hat { d }$ is obtained from the selected agent’s forward posterior $F _ { i ^ { \star } }$ . Thus, R is used only to rank the agents and does not contribute probability mass to the final prediction.

FwdJS (soft reweighting). MinJS uses reverse consistency for hard selection. FwdJS converts the same signal into soft weights over the forward pool:

$$
\begin{array} { r } { w _ { i } = \frac { \displaystyle \exp ( - \tau D _ { i } ) } { \displaystyle \sum _ { j } \exp ( - \tau D _ { j } ) } , \ } \\ { P _ { \mathrm { F w d J S } } ( d ) = \displaystyle \sum _ { i } w _ { i } F _ { i } ( d ) . \ } \end{array}\tag{5}
$$

We fix the temperature to $\tau ~ = ~ 5 . 0$ throughout all experiments. The collective label is arg max<sub>d</sub> $P _ { \mathrm { F w d J S } } ( d )$ . Agents whose posteriors are more consistent with R receive larger weights, while the resulting posterior remains a convex combination of the forward posteriors. In this operator, R shapes the routing weights but does not itself contribute probability mass to the output.

LogLin (forward–reverse fusion). When the forward and reverse estimates contain partially complementary errors, a product-like composition can emphasize labels supported by both channels. LogLin therefore builds on the FwdJS aggregate and directly incorporates the reverse posterior:

$$
{ P } _ { \mathrm { L o g L i n } } ( d ) \propto { P } _ { \mathrm { F w d J S } } ( d ) ^ { 1 - w _ { R } } R ( d ) ^ { w _ { R } } ,\tag{6}
$$

where we use $w _ { R } = 0 . 2$ by default unless stated otherwise (Appendix D). FwdJS determines how much each forward agent contributes, whereas

LogLin additionally allows the reverse anchor to reshape the final posterior (dashed arrow in Figure 1). This fixed-weight product-like form is related at the operator level to the geometric-mean composition used in FOBAR (Jiang et al., 2024), but serves a different purpose: FOBAR combines forward and backward evidence for candidate verification, whereas we use the reverse posterior as a shared anchor for multi-agent aggregation.

Summary. The reverse posterior R is not treated as another competing prediction. Instead, it provides a shared consistency reference that supports three levels of collective decision: hard selection (MinJS), soft reweighting (FwdJS), and direct forward–reverse fusion (LogLin). Importantly, cross-path consistency is a measure of compatibility rather than a certificate of correctness. Consequently, R need not be the top-1 predictor itself to be useful as an aggregation anchor.

## 3.4 Labeled calibration of the reverse anchor

The ordinal likelihood and context-activation maps used to construct R may be misspecified, making the resulting reverse posterior a biased approximation of the underlying posterior. When a labeled training split is available, we optionally calibrate the reverse anchor R while keeping MinJS, FwdJS, and LogLin fixed (dashed box in Figure 1). The low-capacity parameterization is designed to permit calibration with limited labeled data, without retraining the underlying LLM. Calibration only modifies the reverse anchor used by the aggregation heads.

Stage 1 calibrates the two ordinal maps used to construct the reverse posterior: the ordinal likelihood map used to parameterize $P ( e \mid d )$ and the contextual-activation map fused to parameterize $P ( d \mid a )$ . Each map converts an ordinal rank $k \in \{ 0 , \ldots , 6 \}$ into a monotone continuous value through a logit-linear curve:

$$
v _ { k } = \ell + \left( h - \ell \right) \sigma \left( a + b \cdot k / 6 \right) , \quad b > 0 ,\tag{7}
$$

where $\sigma$ is the logistic function and $( a , b )$ are learned separately for the two maps. A temperature $T$ is additionally fitted to rescale the replayed reverse posterior. After fitting, the two maps and $T$ are frozen, and the reverse posterior R is recomputed by replaying the Bayesian network with the calibrated parameters.

Stage 2 then applies a scalar class-prior correc-

tion

$$
{ \cal R } ^ { \prime } ( d ) \propto \frac { { \cal R } ( d ) } { m ( d ) ^ { \gamma } } ,\tag{8}
$$

where $m ( d )$ is the class marginal of $R$ and $\gamma$ is a single scalar parameter. The resulting calibrated reverse anchor $R ^ { \prime }$ therefore adds only this scalar correction and does not introduce a class-specific bias $b _ { d } .$ . Fitting details and a higher-capacity variant with a class-specific bias vector $b \in \mathbb { R } ^ { | \mathcal { V } | }$ , used as a capacity upper bound, are provided in Appendix B.

## 3.5 Computational cost

Inference requires K forward agent inferences and one shared reverse inversion per instance, followed by lightweight divergence computation and posterior aggregation. The reverse inversion is implemented with two LLM elicitation steps composed through the Bayesian network, and the same reverse posterior R is reused by all aggregation operators. No parameter updates are performed at test time; labeled calibration, when used, is fit on a training split only. Thus, the method introduces no fine-tuning cost and requires only a single shared reverse construction in addition to the $K$ forward agent inferences.

## 4 Experimental Setup

## 4.1 Dataset

We evaluate on DDXPlus (Fansi Tchango et al., 2022), a synthetic diagnostic benchmark whose cases pair structured clinical evidence with a closed set of 49 disease labels. Each case records demographics and observed findings, together with contextual information that precedes the remaining evidence. For this benchmark, the contextual component corresponds to $^ { a , }$ the remaining observed evidence to $e ,$ , and the latent class d to the diagnosis. Forward agents map the full case description to a disease posterior, while the reverse procedure constructs $R ( d \mid x ) \propto P ( e \mid d ) P ( d \mid a )$ once per case from the same evidence. We use a fixed 2,000-case test slice. Because every aggregator must use the same five forward posteriors and the same shared reverse anchor, we restrict evaluation to the per-backbone intersection of complete forward cases and cases with an available reverse posterior. We apply the same availability and parsing criteria to all methods within each backbone, yielding a fixed evaluation pool independent of the aggregation method. All denotes this complete perbackbone pool. Disagree is the subset of All in which the five forward top-1 predictions are not unanimous. Headline tables use these two slices.

## 4.2 Baselines

Collective-decision baselines operate entirely within the forward channel. Random Agent (Rand.) uniformly samples one of the five forward agents per case. We report the pooled accuracy over three fixed seeds. It serves as the selection control for MinJS. Voting-based aggregators follow GEDI (Zhao et al., 2024) and operate on the same five forward ballots or posteriors without reverse input. Plurality counts top-1 votes; Range voting treats each posterior as a cardinal score vector and selects the disease with the highest total score; Borda Count, Bucklin, IRV, Minimax, and Ranked Pairs apply the corresponding ordinal scoring or runoff rules. The dictatorial-based baselines are two LLM judges over the same five agents: Informed dictatorial judge (Informed Dicta.) and ACH-inspired structured judge (Zhao et al., 2026).

## 4.3 Models and implementation details

The forward pool comprises five heterogeneous prompting strategies applied to each backbone: Tree-of-Thought (ToT), few-shot MedPrompt, vanilla chain-of-thought, common-bias, and rarebias. Each agent produces a top-5 posterior over the official 49-label set. We use unified sampling settings across all agents, with temperature $= 0 . 6 \ \mathrm { a n d } \ \mathrm { t o p } { - p } \ = \ 0 . 9 5$ , and generate $n { = } 5$ responses per query, one for each agent. We evaluate the resulting pools on five backbones: Qwen3- 30B-A3B (Yang et al., 2025) (hereafter Qwen3- 30B), Mistral-Small-3.1-24B (Mistral AI, 2025), Llama-3.1-8B (Grattafiori et al., 2024), GLM-4- 32B (Team GLM et al., 2024), and DeepSeek-V4-Flash (DeepSeek-AI et al., 2026). The primary metric is Gold Top-Pathology Accuracy at 1 (GTPA@1), where a prediction is counted as correct when the gold pathology is the arg max of the final posterior. The shared reverse posterior is the label-free inversion described in Section 3.2 and is reused by all reverse-referenced operators for each case. Our methods are MinJS selection, FwdJS reweighting with $\tau = 5 . 0$ , and LogLin forward– reverse fusion with fixed $w _ { R } = 0 . 2$ (Tables 1–2; see Appendix D for the $w _ { R }$ sweep). Within each backbone, all methods are evaluated on the same fixed case set.

## 5 Results

## 5.1 Main Results: Reverse-Anchored Collective Decision

A shared reverse posterior turns an unanchored forward pool into a cross-path collective decision. Tables 1 and 2 report GTPA@1 on All and Disagree. A double rule separates single-agent selection from pool aggregation. For selection methods, the best result is shown in bold; for voting, judging, and fusion methods, the best and runner-up results are shown in bold and underline, respectively.

LogLin consistently outperforms the strongest electoral baseline. On Disagree, it exceeds the best forward-only electoral rule on all five backbones by 1.2–4.7 pp and also outperforms both LLM judges. On All, which includes cases where the forward agents already agree, LogLin achieves the best result for every backbone. Gains on Disagree are therefore diluted in the aggregate score. FwdJS likewise outperforms the strongest electoral rule on Disagree by 0.4–3.5 pp. The comparison on GLM illustrates that reverse-guided reweighting is not uniformly dominant: ACH reaches 44.07%, slightly above FwdJS at 42.60%, while LogLin remains best at 44.41%.

The three operators also exhibit the intended progression from hard selection to soft aggregation and direct reverse fusion. MinJS already exploits reverse consistency, outperforming random agent on every backbone, but hard selection can remain below the strongest election (e.g., 71.64% vs. 72.69% for Range voting on Qwen All). FwdJS improves on this by distributing weight across agents according to their consistency with R, while LogLin further incorporates R directly into the fused posterior. Thus, the gains arise from using R as a cross-path reference for routing and fusion.

## 5.2 Anchor Utility Is Not Standalone Accuracy

Table 3 reports standalone GTPA@1 and frozenhead fusion performance when each candidate distribution is used as the shared anchor. The central pattern is that R is often weaker as a standalone predictor, yet more useful as an anchor. MeanF denotes the equal-weight average of the in-pool forward posteriors $\{ F _ { i } \}$ and is used as a replacement anchor in Table 3. GenF is a generic pool-external single agent: one forward call with a neutral persona and the same top-5 output schema as the pool agents. We use it in two roles: as a standalone baseline to test whether a generic single agent can replace collective aggregation, and as a replacement anchor to test whether a pool-external forward posterior can substitute for R.

On All, R is the weakest standalone predictor among {R, MeanF, GenF} for every backbone, trailing MeanF by 7.8–26.4 pp, yet using R as the shared anchor yields the highest FwdJS and LogLin performance. On Disagree, R is the weakest of the three on Mistral, Llama, and DeepSeek. GLM is the exception (40.68% for R vs. 40.23% for MeanF), while on Qwen GenF (40.30%) is slightly weaker than R (40.90%). GenF is often a stronger standalone predictor than R, yet performs worse as a replacement anchor. Using MeanF as the frozen LogLin anchor yields $\Delta { \le } 0$ relative to the strongest forward-only electoral baseline on every backbone, whereas R remains $+ 1 . 2 \mathrm { t o } + 4 . 7 \mathrm { p p }$ above the same electoral baseline. The electoral gain is therefore attributable to the reverse channel, not the log-linear operator alone. Replacing R with either MeanF or GenF lowers both fusion heads on every backbone.

The reverse construction $R ( d ) \propto P ( e \mid d ) P ( d \mid$ a) factorizes into a likelihood term and a contextual prior. Neither one-factor variant matches the full shared R on Disagree LogLin for Qwen, Mistral, GLM, or DeepSeek. On Llama Disagree, $R _ { \mathrm { p r i o r } }$ slightly exceeds R (51.69 vs. 50.90), but remains weaker than MeanF and GenF as a standalone predictor. Appendix C extends Table 3 with the one-factor reverse anchors $R _ { \mathrm { l i k } }$ (likelihood only) and $R _ { \mathrm { p r i o r } }$ (contextual prior only).

## 5.3 What the Reverse Anchor Adds Beyond the Forward Pool

The two analyses below examine complementary aspects of what the reverse anchor R contributes beyond the forward pool. The first measures how often a reference predictor reproduces the forward consensus’ same incorrect label, while the second evaluates whether distance to the reference provides a useful signal for ranking the five forward agents.

Reverse errors collide less often on the same incorrect label. When plurality top-1 and a reference predictor are both wrong, π denotes the conditional probability that they assign the same incorrect label. Table 4 reports three matched comparisons on the same headline pool: plurality versus $R ,$ versus the pool-external GenF, and versus each in-pool agent $F _ { i }$ , with results pooled across the five agents. Across all five backbones, π(F, R) is the lowest of the three (0.196–0.413), below π(F, GenF) (0.594–0.765) and $\pi ( F , F _ { i } )$ (0.680– 0.829). The same ordering holds on Disagree (Appendix E). GenF serves as an external extraforward control, while $\pi ( F , F _ { i } )$ measures an inpool echo rate because plurality is itself constructed from $F _ { i }$ . Thus, R shares the same mistaken label with the forward consensus less often than either a pool-external forward predictor or the agents that constitute that consensus. Appendix E further reports the 2×2 correctness grids and Matthews ϕ coefficients for the corresponding error indicators.

<table><tr><td></td><td colspan="2">Selection</td><td colspan="7">Voting-based</td><td colspan="2">|Dictatorial-based </td><td colspan="2">Our method</td></tr><tr><td>Model</td><td>Rand. MinJS</td><td></td><td>Plurality Range</td><td></td><td>Borda Count</td><td>Bucklin</td><td>IRV</td><td>Minimax</td><td>Ranked | Pairs</td><td>| Informed Dicta.</td><td>ACH</td><td></td><td>FwdJS LogLin</td></tr><tr><td>Qwen3-30B</td><td>70.10</td><td>71.64</td><td>71.54</td><td>72.69</td><td>71.34</td><td>71.94</td><td>72.14</td><td>71.54</td><td>71.69</td><td>72.84</td><td>71.84</td><td>73.85</td><td>74.25</td></tr><tr><td>Mistral-Small-3.1-24B</td><td>69.18</td><td>71.37</td><td>71.37</td><td>73.20</td><td>72.54</td><td>72.19</td><td>72.29</td><td>72.03</td><td>72.29</td><td>72.39</td><td>72.13</td><td>73.51</td><td>74.58</td></tr><tr><td>Llama-3.1-8B</td><td>48.39</td><td>49.87</td><td>54.11</td><td>57.90</td><td>54.16</td><td>54.42</td><td>54.01</td><td>52.75</td><td>54.47</td><td>56.39</td><td>50.38</td><td>58.15</td><td>58.46</td></tr><tr><td>GLM-4-32B</td><td>62.16</td><td>64.26</td><td>65.32</td><td>65.12</td><td>64.31</td><td>65.42</td><td>64.82</td><td>64.66</td><td>65.57</td><td>65.93</td><td>66.78</td><td>66.18</td><td>67.04</td></tr><tr><td>DeepSeek-V4-Flash</td><td>75.35</td><td>77.60</td><td>76.84</td><td>77.80</td><td>77.60</td><td>77.29</td><td>77.19</td><td>76.99</td><td>77.14</td><td>76.58</td><td>75.87</td><td>78.61</td><td>79.07</td></tr></table>

Table 1: GTPA@1 (%) on All. ‘Rand.’ and ‘Dicta.’ denote ‘random’ and ‘dictatorial’. The double rule separates single-agent selection from pool aggregation. For selection, the best result is shown in bold; among pool-aggregation methods, the best and runner-up results are shown in bold and underlined, respectively.
<table><tr><td></td><td colspan="2">Selection</td><td colspan="7">Voting-based</td><td colspan="2">|Dictatorial-based</td><td colspan="2">Our method</td></tr><tr><td>Model</td><td>Rand. MinJS</td><td></td><td>Plurality Range</td><td></td><td>Borda Count</td><td>Bucklin</td><td>IRV</td><td>Minimax</td><td>Ranked | Pairs</td><td>| Informed Dicta.</td><td>ACH</td><td></td><td>FwdJS LogLin</td></tr><tr><td>Qwen3-30B</td><td>40.35</td><td>45.26</td><td>44.96</td><td>48.12</td><td>44.06</td><td>45.86</td><td>46.47</td><td>44.96</td><td>45.41</td><td>48.57</td><td>46.17</td><td>51.58</td><td>52.78</td></tr><tr><td>Mistral-Small-3.1-24B</td><td>40.69</td><td>46.24</td><td>46.24</td><td>50.59</td><td>48.88</td><td>47.95</td><td>48.22</td><td>47.56</td><td>48.22</td><td>48.48</td><td>49.01</td><td>51.39</td><td>53.90</td></tr><tr><td>Llama-3.1-8B</td><td>35.64</td><td>37.76</td><td>44.14</td><td>49.70</td><td>44.14</td><td>44.52</td><td>43.92</td><td>42.04</td><td>44.59</td><td>47.60</td><td>40.62</td><td>50.08</td><td>50.90</td></tr><tr><td>GLM-4-32B</td><td>33.63</td><td>38.19</td><td>40.79</td><td>40.23</td><td>38.42</td><td>40.90</td><td>39.55</td><td>39.32</td><td>41.36</td><td>42.03</td><td>44.07</td><td>42.60</td><td>44.41</td></tr><tr><td>DeepSeek-V4-Flash</td><td>41.38</td><td>48.15</td><td>45.93</td><td>48.44</td><td>47.85</td><td>46.96</td><td>46.67</td><td>46.07</td><td>46.52</td><td>45.19</td><td>44.15</td><td>50.81</td><td>52.30</td></tr></table>

Table 2: GTPA@1 (%) on Disagree. Column groups and formatting match Table 1.

JS divergence to R ranks agents by accuracy. For each case, we rank the five forward agents by $\mathrm { J S } ( F _ { i } , \cdot )$ with respect to a given anchor and evaluate each rank k by the corresponding agent’s GTPA@1. In aggregate, agents with lower JS divergence tend to be more accurate than those with higher divergence. We show Qwen as a representative case in Figure 2 using R, MeanF, and GenF as anchors, and report all backbones in $\mathsf { A p - }$ pendix A. The ranking is more consistently monotone under R than under MeanF or GenF. Under R, accuracy decreases with rank on Qwen and Mistral, is nearly tied between rank-1 and rank-2 on DeepSeek, and peaks at rank-2 and rank-3 on Llama and GLM, respectively. By comparison, the ranking under MeanF is monotone only on Llama, while the other four backbones break the rank ordering. Under GenF, only Mistral is monotone.

![](images/8e5bab9460c39e0b681eb9563953c203bee9821909177820758ffa9191b9c69d.jpg)  
Figure 2: GTPA@1 (%) by JS rank on Disagree (Qwen3-30B). Reverse R (blue), MeanF (orange), and GenF (green). Dashed line: random agent.

On Qwen, Llama, and GLM, rank-2 achieves the highest accuracy, indicating that the agent closest to GenF is not necessarily the most accurate.

## 5.4 Calibrating the Reverse Anchor

Section 3.4 calibrates the reverse anchor R on a labeled training split while keeping MinJS, FwdJS, and LogLin fixed. Table 5 reports test GTPA@1 gains from the calibrated reverse anchor R<sup>′</sup> over the uncalibrated R on the same evaluation pool. On All, all three aggregation heads improve on all five backbones: LogLin by 0.05–2.83 pp, FwdJS by 0.20–1.51 pp, and MinJS by 1.06–4.64 pp. The standalone reverse top-1 accuracy also improves by 1.4–8.6 pp, while the higher-capacity class-specific variant is reported separately in Appendix B. The gains are concentrated on Disagree: LogLin improves by 0.59–4.51 pp on four backbones, with Mistral the sole exception $( - 0 . 2 6 ~ \mathsf { p p } )$ . Adding a class-specific bias $b \in \mathbb { R } ^ { | \mathcal { V } | }$ further improves fusion performance, but increases reverse top-1 accuracy by up to 26 pp on Disagree. This pattern is consistent with the additional capacity primarily capturing class-prior effects rather than providing a stronger routing signal.

<table><tr><td></td><td></td><td colspan="3">All</td><td colspan="3">Disagree</td></tr><tr><td>Model</td><td>Anchor</td><td>Stand.</td><td>FwdJS</td><td>LogLin</td><td>Stand.</td><td>FwdJS</td><td>LogLin</td></tr><tr><td rowspan="3">Qwen3-30B</td><td>R</td><td>61.50</td><td>73.85</td><td>74.25</td><td>40.90</td><td>51.58</td><td>52.78</td></tr><tr><td> $M e a n F$ </td><td>72.69</td><td>72.54</td><td>72.69</td><td>48.12</td><td>47.67</td><td>48.12</td></tr><tr><td> $G e n F$ </td><td>68.47</td><td>72.94</td><td>72.44</td><td>40.30</td><td>48.87</td><td>48.27</td></tr><tr><td rowspan="3">Mistral-S.3.1-24B</td><td> $R$ </td><td>59.96</td><td>73.51</td><td>74.58</td><td>40.82</td><td>51.39</td><td>53.90</td></tr><tr><td> $M e a n F$ </td><td>73.26</td><td>73.00</td><td>73.10</td><td>50.73</td><td>50.07</td><td>50.33</td></tr><tr><td> $G e n F$ </td><td>72.24</td><td>72.90</td><td>73.10</td><td>48.75</td><td>49.80</td><td>50.20</td></tr><tr><td rowspan="3">Llama-3.1-8B</td><td>R</td><td>31.50</td><td>58.15</td><td>58.46</td><td>30.63</td><td>50.08</td><td>50.90</td></tr><tr><td> $M e a n F$ </td><td>57.90</td><td>57.45</td><td>57.60</td><td>49.70</td><td>49.02</td><td>49.25</td></tr><tr><td> $G e n F$ </td><td>46.64</td><td>57.60</td><td>56.28</td><td>35.96</td><td>49.25</td><td>47.52</td></tr><tr><td rowspan="3">GLM-4-32B</td><td>R</td><td>57.34</td><td>66.18</td><td>67.04</td><td>40.68</td><td>42.60</td><td>44.41</td></tr><tr><td> $M e a n F$ </td><td>65.12</td><td>65.47</td><td>65.47</td><td>40.23</td><td>41.02</td><td>41.02</td></tr><tr><td> $G e n F$ </td><td>61.48</td><td>65.88</td><td>64.41</td><td>34.80</td><td>41.92</td><td>39.55</td></tr><tr><td rowspan="3">DeepSeek-V4-Flash</td><td>R</td><td>67.76</td><td>78.61</td><td>79.07</td><td>36.59</td><td>50.81</td><td>52.30</td></tr><tr><td> $M e a n F$ </td><td>77.80</td><td>77.90</td><td>77.80</td><td>48.44</td><td>48.74</td><td>48.44</td></tr><tr><td> $G e n F$ </td><td>75.22</td><td>77.65</td><td>77.75</td><td>43.85</td><td>48.00</td><td>48.44</td></tr></table>

Table 3: Anchor utility on the headline pool (GTPA@1 %). Each row uses one anchor distribution for standalone prediction and for frozen FwdJS / LogLin $( w _ { R } { = } 0 . 2 )$ . Bold: best of {R, MeanF, GenF} in that column, per backbone. GenF: general single agent (pool-external, neutral persona). Stand.: standalone anchor argmax.

<table><tr><td>Model</td><td> $\pi ( F , R )$ </td><td> $\pi ( F , G e n F )$ </td><td> $\pi ( F , F _ { i } )$ </td></tr><tr><td>Qwen3-30B</td><td>0.413</td><td>0.765</td><td>0.829</td></tr><tr><td>Mistral-S.3.1-24B</td><td>0.404</td><td>0.725</td><td>0.775</td></tr><tr><td>Llama-3.1-8B</td><td>0.196</td><td>0.594</td><td>0.680</td></tr><tr><td>GLM-4-32B</td><td>0.348</td><td>0.658</td><td>0.750</td></tr><tr><td>DeepSeek-V4-Flash</td><td>0.393</td><td>0.691</td><td>0.781</td></tr></table>

Table 4: Label-collision rates on All (headline pool). $\pi { = } P ( \mathrm { p r e d } _ { F } { = } \mathrm { p r e d } _ { X }$ | both wrong) for plurality versus $X \in \{ R , G e n F , F _ { i } \} . \ \pi ( F , F _ { i } )$ is pooled over the five pool agents. Bold: lowest π per row. Disagree π, 2×2 grids, and ϕ are in Appendix E.

## 6 Conclusion

LLM-based multi-agent systems aggregate heterogeneous answers, yet voting and LLM judges remain confined to the same evidence-to-conclusion path and can inherit correlated errors from the agent pool.

We introduce a shared reverse posterior R by Bayesian backward reasoning as a structurally distinct reference and use $\operatorname { J S } ( F _ { i } , R )$ to guide collective decision-making. This signal drives three training-free aggregators: hard selection (MinJS), reverse-guided reweighting (FwdJS), and lightweight forward–reverse fusion (LogLin).

Across five LLM backbones on DDXPlus, reverse-anchored aggregation improves over baseline methods, with the largest gains on Disagree. Notably, R is often weaker as a standalone predictor, yet replacing it with either MeanF or GenF degrades FwdJS and LogLin, highlighting the distinction between predictive accuracy and anchor utility. When both the forward consensus and a reference are wrong, R is also less likely to share the same incorrect label with the forward consensus than GenF or an in-pool agent. Two-stage calibration further improves the frozen aggregation rules by refining the reverse anchor. Together, these results support reverse consistency as a useful complementary signal for multi-agent collective decision-making.

## Limitations

The reverse anchor is defined over a finite label set. It inverts an explicit likelihood $P ( e \mid d )$ over a closed candidate set, using agents’ shortlisted candidates, and therefore targets evidence-to-label aggregation rather than open-ended generation without a discrete candidate space. Constructing R also incurs one additional generative inversion per case on top of the K forward agents.

Future work could characterize how the reliability of R interacts with aggregation performance, including the point at which a sufficiently weak reverse posterior begins to hurt aggregation. For the labeled extension, studying calibration under progressively smaller labeled sets would clarify its sample efficiency and establish how much supervision is sufficient for reliable anchor refinement.

<table><tr><td></td><td colspan="4">All (∆ pp vs. uncalibrated)</td><td colspan="4">Disagree (∆ pp vs. uncalibrated)</td></tr><tr><td>Backbone</td><td>∆rev</td><td>∆minJS</td><td>∆FwdJS</td><td>∆log-lin.</td><td>∆rev</td><td>∆minJS</td><td>∆FwdJS</td><td>∆log-lin.</td></tr><tr><td>Qwen3-30B</td><td>+4.97</td><td>+1.86</td><td>+0.80</td><td>+1.61</td><td>+7.22</td><td>+5.41</td><td>+2.41</td><td>+4.51</td></tr><tr><td>Mistral-S.3.1-24B</td><td>+3.06</td><td>+1.53</td><td>+0.20</td><td>+0.05</td><td>+3.96</td><td>+3.96</td><td>+0.53</td><td>-0.26</td></tr><tr><td>Llama-3.1-8B</td><td>+8.63</td><td>+4.64</td><td>+1.51</td><td>+2.83</td><td>+5.93</td><td>+6.91</td><td>+2.25</td><td>+3.30</td></tr><tr><td>GLM-4-32B</td><td>+2.27</td><td>+1.06</td><td>+0.20</td><td>+0.35</td><td>+2.82</td><td>+2.49</td><td>+0.45</td><td>+2.03</td></tr><tr><td>DeepSeek-V4-Flash</td><td>+1.42</td><td>+1.22</td><td>+0.25</td><td>+0.10</td><td>+3.56</td><td>+3.41</td><td>+0.74</td><td>+0.59</td></tr></table>

Table 5: Test GTPA@1 gain (pp) of calibrated R over uncalibrated R. MinJS, FwdJS, and LogLin heads are frozen. The same case pools as in Tables 1–2 are used.

Our evaluation focuses on a single-round, samebackbone setting. The forward pool consists of different prompting strategies applied to one LLM backbone at a time, leaving mixed-backbone pools and multi-round debate as natural extensions. The label-collision statistic π captures how often two incorrect predictors assign the same wrong label, while future work could further investigate the mechanisms underlying such error overlap. Finally, all experiments are conducted on DDXPlus, a synthetic closed-set diagnostic benchmark with 49 diseases. Evaluating the framework across additional evidence-to-label domains would therefore be an important next step, and the proposed aggregators should not be interpreted as clinical decisionsupport systems.

## References

Rui Ai, Yuqi Pan, David Simchi-Levi, Milind Tambe, and Haifeng Xu. 2025. Beyond majority voting: LLM aggregation by leveraging higher-order information. arXiv preprint arXiv:2510.01499.

Razan Baltaji, Babak Hemmatian, and Lav Varshney. 2024. Conformity, confabulation, and impersonation: Persona inconstancy in multi-agent LLM collaboration. In Proceedings ofthe 2nd Workshop on Cross-Cultural Considerations in NLP, pages 17–31, Bangkok, Thailand. Association for Computational Linguistics.

Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. 2024. ChatEval: Towards better LLM-based evaluators through multi-agent debate. In The Twelfth International Conference on Learning Representations.

Justin Chen, Swarnadeep Saha, and Mohit Bansal. 2024. ReConcile: Round-table conference improves reasoning via consensus among diverse LLMs. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long

Papers), pages 7066–7085, Bangkok, Thailand. Association for Computational Linguistics.

Justin Chen, Zifeng Wang, Hamid Palangi, Rujun Han, Sayna Ebrahimi, Long Le, Vincent Perot, Swaroop Mishra, Mohit Bansal, Chen-Yu Lee, and Tomas Pfister. 2025. Reverse thinking makes LLMs stronger reasoners. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8611–8630, Albuquerque, New Mexico. Association for Computational Linguistics.

Hyeong Kyu Choi, Jerry Zhu, and Sharon Li. 2026. When identity skews debate: Anonymization for biasreduced multi-agent reasoning. In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14284–14311, San Diego, California, United States. Association for Computational Linguistics.

Hyeong Kyu Choi, Xiaojin Zhu, and Sharon Li. 2025. Debate or vote: Which yields better decisions in multi-agent large language models? In Advances in Neural Information Processing Systems.

DeepSeek-AI, Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chengyu Hou, Chenhao Xu, Chenze Shao, Chong Ruan, Conner Sun, and 300 others. 2026. Deepseek-v4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. 2024. Improving factuality and reasoning in language models through multiagent debate. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 11733–11763. PMLR.

Arsene Fansi Tchango, Rishab Goel, Zhi Wen, Julien Martel, and Joumana Ghosn. 2022. Ddxplus: A new dataset for automatic medical diagnosis. Advances in neural information processing systems, 35:31306– 31318.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh

Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Neel Guha, Mayee F. Chen, Trevor Chow, Ishan S. Khare, and Christopher Ré. 2024. Smoothie: Label free language model routing. In Advances in Neural Information Processing Systems.

Lu Hong and Scott E. Page. 2004. Groups of diverse problem solvers can outperform groups of highability problem solvers. Proceedings ofthe National Academy ofSciences, 101(46):16385–16389.

Yichong Huang, Xiaocheng Feng, Baohang Li, Yang Xiang, Hui Wang, Ting Liu, and Bing Qin. 2024. Ensemble learning for heterogeneous large language models with deep parallel collaboration. In Advances in Neural Information Processing Systems.

Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin. 2023. LLM-blender: Ensembling large language models with pairwise ranking and generative fusion. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 14165–14178, Toronto, Canada. Association for Computational Linguistics.

Weisen Jiang, Han Shi, Longhui Yu, Zhengying Liu, Yu Zhang, Zhenguo Li, and James Kwok. 2024. Forward-backward reasoning in large language models for mathematical verification. In Findings of the Association for Computational Linguistics: ACL 2024, pages 6647–6661, Bangkok, Thailand. Association for Computational Linguistics.

Jiayi Li, Xiao Liu, and Yansong Feng. 2026. From single to societal: Analyzing persona-induced bias in multi-agent interactions. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 31609–31617.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. 2024. Encouraging divergent thinking in large language models through multi-agent debate. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 17889–17904, Miami, Florida, USA. Association for Computational Linguistics.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: NLG evaluation using GPT-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522, Singapore. Association for Computational Linguistics.

Costas Mavromatis, Petros Karypis, and George Karypis. 2024. Pack of LLMs: Model fusion at testtime via perplexity optimization. In First Conference on Language Modeling.

Mistral AI. 2025. Mistral small 3.1. https://mistral. ai/news/mistral-small-3-1/.

Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, and 39 others. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793.

Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou. 2025. Mixture-of-agents enhances large language model capabilities. In The Thirteenth International Conference on Learning Representations.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations.

Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. 2023. Large language models are better reasoners with self-verification. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 2550–2575, Singapore. Association for Computational Linguistics.

Anita Williams Woolley, Christopher F. Chabris, Alex Pentland, Nada Hashmi, and Thomas W. Malone. 2010. Evidence for a collective intelligence factor in the performance of human groups. Science, 330(6004):686–688.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Xiutian Zhao, Ke Wang, and Wei Peng. 2024. An electoral approach to diversify LLM-based multiagent collective decision-making. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 2712–2727, Miami, Florida, USA. Association for Computational Linguistics.

Xuyang Zhao, Shiwan Zhao, Hualong Yu, Liting Zhang, and Qicheng Li. 2026. AgentCDM: Enhancing multi-agent collaborative decision-making via ACHinspired structured reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 34985–34993. AAAI Press.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot

arena. Advances in neural information processing systems, 36:46595–46623.

Zhilun Zhou, Zihan Liu, Jiahe Liu, Yihan Wang, Qingyu Shao, Fengli Xu, Depeng Jin, and Yong Li. 2026. Identifying collective intelligence factor in LLM agent groups for generalizable multi-agent system design. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 12827–12842, San Diego, California, United States. Association for Computational Linguistics.

## A JS-based agent ranking across backbones

Table 6 reports GTPA@1 after ranking the five forward agents by $\operatorname { J S } ( F _ { i } , R )$ on Disagree. Figure 2 in the main text shows the Qwen Disagree slice and includes MeanF and GenF as alternative anchors. Figure 3 summarizes the remaining backbone–subset curves for both All and Disagree.

Under R, the Disagree rank curve is strictly monotone on Qwen and Mistral. Llama and GLM attain their highest accuracy after rank-1, while rank-1 and rank-2 on DeepSeek differ by only one case. The ranking is less consistently monotone under MeanF and GenF.

<table><tr><td>Model</td><td>Rank-1</td><td>Rank-5</td><td>Rand.</td></tr><tr><td>Qwen3-30B</td><td>45.26</td><td>27.52</td><td>39.88</td></tr><tr><td>Mistral-S.3.1-24B</td><td>46.24</td><td>29.19</td><td>40.42</td></tr><tr><td>Llama-3.1-8B</td><td>37.76</td><td>26.73</td><td>35.36</td></tr><tr><td>GLM-4-32B</td><td>38.19</td><td>24.63</td><td>34.21</td></tr><tr><td>DeepSeek-V4-Flash</td><td>48.15</td><td>25.48</td><td>41.19</td></tr></table>

Table 6: GTPA@1 (%) by reverse JS rank on Disagree. Rank-k: the forward agent with the k-th lowest $\mathrm { J S } ( F _ { i } , R )$ on each case. Rand.: mean GTPA@1 across the five ranks.

## B Labeled calibration: fitting and capacity upper bound

Section 3.4 defines the two-stage calibration of R. This appendix details the fitting protocol for the calibrated reverse anchor and the higher-capacity class-specific bias variant.

## B.1 Fitting protocol

The reverse inversion uses ordinal evidencelikelihood and context-activation bins. Each axis in (7) has seven ordinal ranks $k \in { 0 , \ldots , 6 }$ , with axis-specific clamps ℓ, h. Stage 1 fits four map parameters $( a _ { s } , b _ { s } , a _ { a } , b _ { a } )$ together with a temperature T for rescaling the replayed reverse posterior. The objective is to minimize the label NLL of the reverse posterior R obtained by replaying the Bayesian network on the training cases. After fitting, the two maps and the temperature T are frozen. Stage 2 fits the scalar γ in (8) by k-fold minimization of the same NLL on the training fit split. Evaluation then applies the frozen MinJS, FwdJS, and LogLin heads on the same evaluation pool used for the label-free results.

## B.2 Capacity upper bound (L5CURVE\_L1GB)

Table 7 reports the same frozen-head evaluation as Section 5.4, but gives each label its own bias term $b _ { d }$ in addition to the scalar correction γ:

$$
R ^ { \prime } ( d ) \propto R ( d ) m ( d ) ^ { - \gamma } e ^ { b _ { d } } .\tag{9}
$$

This gives the calibration more flexibility and further improves both the standalone accuracy of R and the downstream fusion results. However, the large gain in reverse top-1 accuracy suggests that much of this improvement may come from adjusting label-specific prior preferences rather than from making R a better routing signal. We therefore use this higher-capacity version only as an upperbound reference, rather than as the primary calibrated method.

<table><tr><td rowspan="2">Backbone</td><td colspan="4">All (∆ pp vs. uncalibrated)</td><td colspan="4">Disagree (∆ pp vs. uncalibrated)</td></tr><tr><td>∆rev</td><td>∆minJS</td><td>∆FwdJS</td><td>∆log-lin.</td><td>∆rev</td><td>∆minJS</td><td>∆FwdJS</td><td>∆log-lin.</td></tr><tr><td>Qwen3-30B</td><td>+12.05</td><td>+3.71</td><td>+2.06</td><td></td><td>+5.77 +19.10</td><td>+10.98</td><td>+6.17</td><td>+13.23</td></tr><tr><td>Mistral-S.3.1-24B</td><td>+7.90</td><td>+2.29</td><td>+0.97</td><td></td><td>+2.75 +11.76</td><td>+5.55</td><td>+2.51</td><td>+5.81</td></tr><tr><td>Llama-3.1-8B</td><td>+16.15</td><td>+7.32</td><td>+3.33</td><td></td><td>+6.81 +14.49</td><td>+10.96</td><td>+4.95</td><td>+8.86</td></tr><tr><td>GLM-4-32B</td><td>+10.70</td><td>+3.33</td><td>+1.41</td><td></td><td>+4.24 +12.66</td><td>+7.57</td><td>+3.16</td><td>+8.70</td></tr><tr><td>DeepSeek-V4-Flash +12.52</td><td></td><td>+4.36</td><td>+1.32</td><td></td><td>+3.75 +26.52</td><td>+12.59</td><td>+3.85</td><td>+11.11</td></tr></table>

Table 7: Test GTPA@1 gain (pp) from class-specific calibration of R over uncalibrated R. Same evaluation pool as Table 5.

## C Likelihood-only and prior-only reverse anchors

Table 3 in the main text compares the full reverse posterior R with the forward-pool anchors MeanF and GenF. Table 8 uses the same evaluation pools and frozen FwdJS/LogLin heads, but replaces R with one-factor reverse constructions: $R _ { \mathrm { l i k } }$ uses only the likelihood term $P ( e \mid d )$ , whereas $R _ { \mathrm { p r i o r } }$ uses only the contextual prior $P ( d \mid a )$ . On Disagree with LogLin, neither one-factor construction matches the full R on Qwen, Mistral, GLM, or DeepSeek. Llama is the exception: $R _ { \mathrm { p r i o r } }$ reaches 51.69%, slightly above R at 50.90%, but remains weaker than MeanF and GenF as a standalone predictor.

![](images/b5f8fd6e77e0dbe73719cd873a09bff3021855f9a7a46dc720e0f4f64cd69c4a.jpg)  
Figure 3: GTPA@1 (%) by JS rank on the evaluation pool. Reverse R, MeanF, and GenF are used as anchors. Dashed line: random agent.

<table><tr><td rowspan="2">Model</td><td rowspan="2"></td><td colspan="3">All</td><td colspan="3">Disagree</td></tr><tr><td>Anchor Stand. FwdJS</td><td></td><td>LogLin</td><td></td><td></td><td>Stand. FwdJS LogLin</td></tr><tr><td rowspan="2">Qwen3-30B</td><td> $R _ { \mathrm { l i k } }$ </td><td>51.53</td><td>72.90</td><td>72.90</td><td>30.23</td><td>48.87</td><td>48.87</td></tr><tr><td>Rprior</td><td>43.69</td><td>73.20</td><td>73.40</td><td>31.58</td><td>49.77</td><td>50.38</td></tr><tr><td rowspan="2">Mistral-S.3.1-24B</td><td></td><td>44.74</td><td>73.06</td><td>74.23</td><td>34.44</td><td>50.20</td><td>53.11</td></tr><tr><td> $\begin{array} { l } { R _ { \mathrm { l i k } } } \\ { R _ { \mathrm { p r i o r } } } \end{array}$ </td><td>43.21</td><td>73.21</td><td>73.83</td><td>26.23</td><td>50.60</td><td>52.19</td></tr><tr><td rowspan="2">Llama-3.1-8B</td><td> $R _ { \mathrm { l i k } }$ </td><td>19.67</td><td>57.89</td><td>56.88</td><td>16.98</td><td>49.74</td><td>48.53</td></tr><tr><td>Rprior</td><td>30.59</td><td>58.29</td><td>59.20</td><td>29.53</td><td>50.34</td><td>51.69</td></tr><tr><td rowspan="2">GLM-4-32B</td><td> $R _ { \mathrm { l i k } }$ </td><td>47.32</td><td>64.86</td><td>65.12</td><td>33.75</td><td>39.64</td><td>40.32</td></tr><tr><td> $R _ { \mathrm { p r i o r } }$ </td><td>39.89</td><td>65.57</td><td>66.03</td><td>25.37</td><td>41.22</td><td>42.13</td></tr><tr><td rowspan="2">DeepSeek-V4-Flash</td><td></td><td>59.49</td><td>78.02</td><td>77.87</td><td>32.89</td><td>49.19</td><td>48.74</td></tr><tr><td> $\begin{array} { l } { R _ { \mathrm { l i k } } } \\ { R _ { \mathrm { p r i o r } } } \end{array}$ </td><td>48.93</td><td>77.92</td><td>78.58</td><td>23.41</td><td>48.89</td><td>50.81</td></tr></table>

Table 8: One-factor reverse anchors (GTPA@1 (%)) on the same evaluation pools as Table 3. $R _ { \mathrm { l i k } }$ : likelihoodonly, using $P ( e \mid d )$ $R _ { \mathrm { p r i o r } } { \mathrm { : } }$ prior-only, using $P ( d \mid a )$

## D Log-linear $w _ { R }$ sweep

Table 9 reports LogLin GTPA@1 for $w _ { R } \in \mathbf { \Sigma }$ 0.2, 0.3, 0.4, 0.5 on the same evaluation pools as Tables 1, 2. The w<sub>R</sub>=0.2 column matches the LogLin results reported in the main tables. Because FwdJS has already incorporated R through JS-based reweighting, we use $w _ { R } { = } 0 . 2$ as a light direct contribution from the reverse anchor rather than selecting the weight separately for each backbone on the test set. On Disagree subset, larger w<sub>R</sub> can yield higher accuracy on some backbones. For example, GLM improves from 44.41% at $w _ { R } { = } 0 . 2$ to 47.57% at $w _ { R } { = } 0 . 5$

<table><tr><td></td><td colspan="4">All</td><td colspan="4">Disagree</td></tr><tr><td>Model</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td></tr><tr><td>Qwen3-30B</td><td>74.25</td><td>74.25</td><td>74.25</td><td>73.85</td><td>52.78</td><td>53.08</td><td>53.38</td><td>53.08</td></tr><tr><td>Mistral-Small-3.1-24B</td><td>74.58</td><td>74.73</td><td>73.71</td><td>73.10</td><td>53.90</td><td>54.29</td><td>52.97</td><td>51.78</td></tr><tr><td>Llama-3.1-8B</td><td>58.46</td><td>57.40</td><td>55.83</td><td>53.36</td><td>50.90</td><td>49.92</td><td>49.40</td><td>47.97</td></tr><tr><td>GLM-4-32B</td><td>67.04</td><td>66.99</td><td>66.89</td><td>66.89</td><td>44.41</td><td>45.31</td><td>45.88</td><td>47.57</td></tr><tr><td>DeepSeek-V4-Flash</td><td>79.07</td><td>79.02</td><td>78.97</td><td>78.71</td><td>52.30</td><td>52.00</td><td>52.15</td><td>52.00</td></tr></table>

Table 9: Log-linear GTPA@1 (%) as a function of reverse weight $w _ { R }$ on the evaluation pools. Bold denotes the best $w _ { R }$ within each block. The main tables use the fixed default $w _ { R } { = } 0 . 2$

## E Label collision, co-occurrence, and error-indicator ϕ

Table 10 reports the matched-unit π statistic from Table 4 on Disagree subset. Across all five backbones, $\pi ( F , R )$ remains the lowest of the three comparisons, consistent with the All results in the main text. Tables 11 and 12 provide complementary views that are not used for the main same-incorrect-label claim: a $2 \times 2$ correctness grid and the Matthews $\phi$ coefficient computed from binary error indicators. The grid uses the same plurality GTPA@1 indicator as Tables 1–2, so Both✓+F✓R× matches the plurality column of those tables. Here, ϕ measures the association between whether plurality is wrong and whether the named reference is wrong. Unlike π, it does not measure whether the two predictors assign the same incorrect label.

<table><tr><td>Model</td><td> $\pi ( F , R )$ </td><td> $\pi ( F , G e n F )$ </td><td> $\pi ( F , F _ { i } )$ </td></tr><tr><td>Qwen3-30B</td><td>0.333</td><td>0.639</td><td>0.717</td></tr><tr><td>Mistral-S.3.1-24B</td><td>0.332</td><td>0.621</td><td>0.677</td></tr><tr><td>Llama-3.1-8B</td><td>0.162</td><td>0.497</td><td>0.600</td></tr><tr><td>GLM-4-32B</td><td>0.296</td><td>0.588</td><td>0.663</td></tr><tr><td>DeepSeek-V4-Flash</td><td>0.352</td><td>0.609</td><td>0.716</td></tr></table>

Table 10: Label-collision rates on Disagree. Definitions match Table 4. Bold denotes the lowest π in each row.

<table><tr><td></td><td colspan="4">All</td><td colspan="4">Disagree</td></tr><tr><td>Model</td><td>Both√</td><td>F√R×</td><td>F×R√</td><td>Both×</td><td>Both√</td><td>F√R×</td><td>F×R√</td><td>Both×</td></tr><tr><td>Qwen3-30B</td><td>50.8</td><td>20.8</td><td>10.5</td><td>17.9</td><td>22.6</td><td>22.4</td><td>20.3</td><td>34.7</td></tr><tr><td>Mistral-S.3.1-24B</td><td>50.8</td><td>20.6</td><td>9.6</td><td>19.1</td><td>25.4</td><td>20.9</td><td>17.7</td><td>36.1</td></tr><tr><td>Llama-3.1-8B</td><td>21.6</td><td>32.6</td><td>11.1</td><td>34.8</td><td>18.0</td><td>26.1</td><td>13.8</td><td>42.0</td></tr><tr><td>GLM-4-32B</td><td>45.3</td><td>20.0</td><td>13.3</td><td>21.4</td><td>21.4</td><td>19.4</td><td>21.4</td><td>37.9</td></tr><tr><td>DeepSeek-V4-Flash</td><td>62.0</td><td>14.9</td><td>6.8</td><td>16.4</td><td>23.4</td><td>22.5</td><td>15.6</td><td>38.5</td></tr></table>

Table 11: Forward/reverse correctness co-occurrence (% of the slice). Columns denote Both✓, plurality correct and R wrong, plurality wrong and R correct (recovery), and Both×. Plurality correctness is the same GTPA@1 indicator as in Tables 1–2.

<table><tr><td></td><td colspan="3">All</td><td colspan="3">Disagree</td></tr><tr><td>Model</td><td>φ(F, R)</td><td>φ(F, GenF)</td><td> $\phi ( F , F _ { i } )$ </td><td>φ(F, R)</td><td>φ(F, GenF)</td><td> $\phi ( F , F _ { i } )$ </td></tr><tr><td>Qwen3-30B</td><td>0.31</td><td>0.73</td><td>0.81</td><td>0.13</td><td>0.41</td><td>0.53</td></tr><tr><td>Mistral-S.3.1-24B</td><td>0.35</td><td>0.79</td><td>0.81</td><td>0.24</td><td>0.61</td><td>0.60</td></tr><tr><td>Llama-3.1-8B</td><td>0.16</td><td>0.61</td><td>0.71</td><td>0.17</td><td>0.45</td><td>0.56</td></tr><tr><td>GLM-4-32B</td><td>0.30</td><td>0.74</td><td>0.82</td><td>0.16</td><td>0.53</td><td>0.60</td></tr><tr><td>DeepSeek-V4-Flash</td><td>0.47</td><td>0.71</td><td>0.78</td><td>0.23</td><td>0.46</td><td>0.52</td></tr></table>

Table 12: Matthews ϕ on binary error indicators. Each ϕ measures the association between whether plurality is wrong and whether the named reference is wrong, rather than whether they assign the same incorrect label. Unlike Table 11, the error indicators here use stringlevel top-1 equality with the gold label, not plurality GTPA@1. $\phi ( F , F _ { i } )$ is pooled across the five in-pool agents. Bold denotes the lowest $\phi$ within each All or Disagree block.