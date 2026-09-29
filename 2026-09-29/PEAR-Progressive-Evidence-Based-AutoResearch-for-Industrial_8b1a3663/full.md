# PEAR: Progressive Evidence-Based AutoResearch for Industrial Search Systems

Global E-Commerce Agentic Search Team

## Abstract

AutoResearch improves systems through iterative experimentation: agents propose candidate modifications, evaluate them, and use the results to guide subsequent exploration. Applying this paradigm to industrial search presents two challenges. (1) Common AutoResearch approaches follow a keep-if-better rule, retaining the highest-scoring candidate for subsequent experiments. Under non-stationary trafic, transient gains may be mistaken for persistent improvements, impairing reliable accumulation of search knowledge. (2) Candidate modifications can be evaluated at multiple fidelity levels, from low-cost proxies to online validation, difering in cost, objective alignment, and statistical reliability. Existing methods rely on individual signals or task-specific procedures, lacking a unified basis for using evidence across levels to guide search. We introduce Progressive Evidence-Based AutoResearch (PEAR) with two complementary components. Evidence-driven AutoResearch maintains an independent, hypothesis-guided research state for each strategy task within a predefined objective and intervention scope. Each state evolves through a Plan– Execute–Evaluate–Update transition that links experimentation to context-aware evidence interpretation and hypothesis revision. Confidence-Gated Verifier Ladder organizes evaluation into four levels of increasing fidelity: Ofline Replay, Shadow-Trafic Evaluation, Rapid Online Evaluation, and Decision-Grade Online Evaluation. A unified confidence-based gate promotes candidates only when evidence supports a statistically significant positive efect, enabling broad low-cost exploration while reserving costly online experiments for promoted candidates. In a real-world industrial search system, strategies optimized with PEAR significantly increased Main Order/DAU by 2.7336% and 3.2957% relative to their respective baselines in two A/B experiments.

## 1 Introduction

Industrial search systems integrate query understanding, candidate retrieval, and result ranking to determine which results are presented to users. Recent work has substantially advanced individual components through instruction-aware retrieval and LLM-based reranking [3, 14, 26, 27]. In production, however, these components operate within a multi-stage decision process in which models, ranking strategies, and business constraints interact. Shifts in user demand, content supply, and business objectives require these strategies to be continually revisited, while dependencies across stages make the efects of candidate modifications context-dependent and dificult to predict ofline. Engineers therefore improve the system through iterative experimentation: they formulate hypotheses from observed system behavior, implement candidate strategies, evaluate them under real trafic, and use the results to guide subsequent iterations.

Recent advances in large language model (LLM) agents have demonstrated multi-step reasoning, tool use, and the ability to act in external environments [17, 18, 20, 23–25], creating an opportunity to automate portions of this workflow. AutoResearch turns these capabilities into an iterative experimentation loop in which agents propose and implement candidate modifications, evaluate their outcomes, and use the resulting evidence to guide subsequent proposals [1, 5, 15, 19]. Recent work has applied this paradigm to search and recommendation systems, exploring interventions that range from ranking parameters to model architectures and reward functions [4, 22], and evaluating them using ofline proxy metrics, model diagnostics, simulated user responses, and online A/B test outcomes [7, 9]. Collectively, these studies demonstrate the potential of agent-driven strategy optimization, while making experimental evidence the primary basis for candidate selection and subsequent exploration.

Reliably using evidence from both ofline and online evaluations in AutoResearch for industrial search presents two challenges arising from non-stationary outcomes and heterogeneous evaluation procedures. (1) How can an agent accumulate reliable experimental knowledge when candidate efects are context-dependent? Many AutoResearch systems apply a keep-if-better rule, retaining the highest-scoring candidate as the reference for subsequent experiments [15, 19]. This rule implicitly assumes that evaluation scores remain comparable across rounds. In industrial search, trafic composition and system state evolve over time; consequently, experiments conducted under diferent conditions may estimate candidate efects that are not directly comparable. A transient or context-specific gain may therefore be recorded as a persistent improvement, weakening hypothesis assessment and misdirecting subsequent exploration. (2) How should evidence from diferent evaluation procedures govern candidate promotion? Low-cost proxy evaluations support broad exploration, but their improvements may not translate into online gains. Online experiments provide evidence that is more directly aligned with deployment objectives, but suficiently evaluating every candidate online is costly and risky [6, 8, 11, 12, 16]. Existing approaches use individual evaluation signals or task-specific evaluation procedures [4, 7, 9, 22], leaving unclear how evidence from diferent procedures should jointly inform promotion decisions. The resulting challenge is to accumulate reliable experimental knowledge under changing contexts and make sound candidate-promotion decisions from heterogeneous evaluation evidence.

To address these challenges, we introduce Progressive Evidence-Based AutoResearch (PEAR), a framework for industrial search systems that couples persistent strategy states with progressive candidate evaluation through two complementary components. Evidence-driven AutoResearch maintains an independent research state for each strategy task, comprising the experimental context, current hypotheses, candidate intervention space, and accumulated evidence. The state evolves through a Plan–Execute–Evaluate–Update transition augmented with Bayesian-optimization-inspired reasoning. This reasoning balances refinement of evidence-supported regions with exploration of underexplored regions, while experimental results are interpreted within their contexts to support hypothesis revision and knowledge accumulation across experiments. Confidence-Gated Verifier Ladder organizes candidate evaluation into four progressively higher-fidelity levels through a unified confidence-based promotion gate: Ofline Replay, Shadow-Trafic Evaluation, Rapid Online Evaluation, and Decision-Grade Online Evaluation. Evaluation results and promotion decisions from each level update the corresponding research state and inform subsequent search. Lower-cost evaluations support broad exploration, while costly online experiments are reserved for promoted strategies. In a real-world industrial search system, strategies optimized with PEAR significantly increased Main Order/DAU by 2.7336% and 3.2957% relative to their respective baselines in two A/B experiments at the Decision-Grade Online Evaluation stage.

## 2 Contributions

1. We introduce Evidence-driven AutoResearch to support reliable knowledge accumulation for industrial search optimization under non-stationary trafic. For each strategy task, it maintains an independent research state that links hypotheses and candidate interventions to their evaluation outcomes and experimental contexts. The state evolves through a Plan–Execute–Evaluate–Update transition, in which contextual evidence informs revisions to both the hypotheses and the candidate intervention space across

experimental rounds.

2. We develop Confidence-Gated Verifier Ladder, which organizes candidate evaluation into four levels of progressively greater fidelity and cost, from Ofline Replay to Decision-Grade Online Evaluation. The ladder preserves each level’s evaluation signal in a unified evidence record and applies a common confidence-based promotion rule, allowing low-cost evaluations to support broad exploration while reserving costly online experiments for statistically supported candidates. The resulting evidence and promotion decisions are returned to the research state to guide subsequent search.

3. We deploy and evaluate PEAR in a real-world industrial search system using three strategy tasks. The evaluation documents evidence-driven hypothesis revision and, for two promoted candidates, directionally consistent positive efects from Shadow-Trafic Evaluation to Decision-Grade Online Evaluation. In two Decision-Grade Online A/B experiments, strategies optimized with PEAR significantly increased Main Order/DAU by 2.7336% and 3.2957% relative to their respective baselines.

## 3 Related Work

## 3.1 AutoResearch

AutoResearch organizes candidate generation, experimentation, and evaluation into iterative research workflows [5, 15, 21]. A common update mechanism is the keep-if-better rule, which retains candidates that improve the observed experimental score as references for subsequent search [15, 19]. The interpretation of these improvements depends on the comparability of evaluation outcomes across experiments. Recent systems preserve experimental history to inform subsequent proposals. Meta-Harness optimizes harness code using the code, execution traces, and scores of prior candidates [10]. AiScientist maintains a persistent workspace that retains experimental artifacts across implementation, experimentation, and diagnosis [2]. AutoResearchClaw combines multi-agent deliberation, execution recovery, and cross-run experience accumulation to support iterative hypothesis generation and experimentation [13]. These approaches provide mechanisms for retaining and reusing information across experiments.

The keep-if-better rule uses observed candidate scores to guide subsequent search. In industrial search optimization, non-stationary trafic complicates the interpretation of these scores across experimental contexts. Reliable search knowledge accumulation requires distinguishing persistent improvements from transient gains over multiple experiments.

## 3.2 AutoResearch in Search and Recommendation

Recent studies apply AutoResearch to search and recommendation through task-specific candidate generation and evaluation procedures. Self-Evolving Recommendation System combines a fast ofline loop for generating and filtering model candidates with a slower online loop for validating delayed business outcomes [22]. Self-EvolveRec combines semantic feedback from a user simulator with model-diagnostic signals to guide candidate modifications across the recommendation pipeline [7]. These studies use proxy evaluation and structured feedback to guide subsequent candidate generation.

Other systems incorporate online experimental feedback into iterative optimization. Sortify estimates the relationship between ofline signals and online outcomes and maintains cross-round memory to inform subsequent parameter search [4]. AgentX connects candidate generation, implementation, online experimental evaluation, and experience feedback through specialized agents [9]. These systems demonstrate how online outcomes can inform subsequent search in deployed search and recommendation systems.

Existing approaches organize candidate evaluation around individual signals or task-specific procedures. Proxy and online evaluation results difer in acquisition cost, objective alignment, and statistical reliability. A unified basis for using these results to guide candidate selection and subsequent search remains an open problem.

## 4 Problem Setup

AutoResearch for industrial search optimization iteratively refines search strategies through candidate generation, experimentation, and evaluation. Each strategy task � specifies a predefined objective and intervention scope. A candidate strategy � specifies a testable modification to the search system, such as a parameter configuration or a change to the ranking mechanism. Multiple strategy tasks may proceed concurrently within their respective intervention scopes.

At experimental round $t ,$ the agent generates a candidate set $\mathcal { A } _ { k , t }$ for strategy task �. An evaluation procedure assigns scores to the candidates according to the optimization objective. The agent uses the resulting scores to refine subsequent proposals. Repeated candidate generation, evaluation, and proposal revision form an iterative optimization process.

In AutoResearch for industrial search systems, evaluations range from low-cost proxy evaluation to online validation. These evaluation sources difer in acquisition cost, objective alignment, and statistical reliability. Proxy evaluation supports broad candidate exploration at low cost. Online validation measures candidate efects through real-user outcomes and requires additional trafic and observation time. Let $S ( c )$ denote the set of evaluation stages applied to candidate $^ { c , }$ and let $\ \varepsilon _ { \ell } ( c )$ denote the evidence acquired at stage ℓ. The collected evidence is

$$
\mathcal { E } ( c ) = \{ \mathcal { E } _ { \ell } ( c ) \ | \ \ell \in S ( c ) \} .\tag{1}
$$

For strategy task $k , \mathcal { E } _ { k , t } ^ { ( \ell ) }$ denotes the set of stage-ℓ evidence records acquired in round �. Each result reflects its evaluation protocol and trafic context. Under non-stationary trafic, results from diferent rounds must be interpreted within their corresponding experimental contexts.

The objective is to optimize industrial search strategies through iterative experimentation guided by heteroge neous evaluation results. Evaluation results provide the basis for hypothesis revision, candidate selection, and subsequent search. This process requires reliable search knowledge accumulation across experimental rounds and suficient online validation to support deployment decisions.

## 5 Progressive Evidence-Based AutoResearch (PEAR)

As illustrated in Figure 1, PEAR couples persistent strategy states with progressive candidate evaluation through two complementary components: Evidence-driven AutoResearch and Confidence-Gated Verifier Ladder.

Evidence-driven AutoResearch maintains an independent, hypothesis-guided research state for each strategy task. Each state preserves the hypotheses, candidate history, and experimental contexts associated with that task. A Plan–Execute–Evaluate–Update transition connects candidate generation and experimentation with evidence interpretation and hypothesis revision.

Confidence-Gated Verifier Ladder organizes candidate evaluation into four progressively higher-fidelity levels: Ofline Replay, Shadow-Trafic Evaluation, Rapid Online Evaluation, and Decision-Grade Online Evaluation. Unified evidence records and a confidence-based promotion gate connect these evaluation levels. Evaluation results and promotion decisions update the corresponding research states and inform subsequent search. Lower-cost evaluations support broad exploration, and costly online experiments are reserved for promoted candidates.

![](images/75c672b7365eb4a555d9775e7e9f4747e06cae7d6065e24fb1d126094b42a86f.jpg)  
Figure 1 Overview of PEAR. Evidence-driven AutoResearch maintains parallel strategy-level research states, while the Confidence-Gated Verifier Ladder progressively evaluates candidates, governs their promotion using evidence of increasing fidelity, and returns evaluation results to the corresponding research states.

## 5.1 Evidence-driven AutoResearch

Evidence-driven AutoResearch associates optimization hypotheses with the evaluation evidence acquired for each strategy task. A hypothesis describes the expected efect of a candidate modification under specified conditions. Its associated evidence records the candidate, evaluation outcome, and experimental context. This association provides the basis for interpreting non-stationary evaluation results and revising hypotheses across experiments.

For strategy task � at experimental round �, the research state is

$$
\mathcal { H } _ { k , t } = \left( C _ { k } , M _ { k , t } , \Omega _ { k , t } , \mathcal { D } _ { k , t } \right) ,\tag{2}
$$

where $C _ { k }$ specifies the search-system implementation, optimization objective, intervention scope, and experimental protocol. $\mathcal { M } _ { k , t }$ contains the current hypotheses, $\Omega _ { k , t }$ denotes the candidate intervention space within the specified scope, and $\mathcal { D } _ { k , i }$ <sub>�</sub> contains the accumulated evidence and its associations with the hypotheses. Independent states preserve the experimental context of each strategy task and support concurrent exploration across strategy tasks.

The research state evolves through a Plan–Execute–Evaluate–Update transition:

$$
\mathcal { H } _ { k , t } \xrightarrow { \mathrm { \tiny ~ P _ { L A N } ~ } } \mathcal { A } _ { k , t } \xrightarrow { \mathrm { \tiny ~ E x e c u r g e } } \Delta \mathcal { D } _ { k , t } \xrightarrow { \mathrm { \tiny ~ E v a u s r g e } } \mathcal { R } _ { k , t } \xrightarrow { \mathrm { \tiny ~ U p o a r g e } } \mathcal { H } _ { k , t + 1 } .\tag{3}
$$

Here, ${ \mathcal A } _ { k , \ast }$ <sub>�</sub> is the candidate set, $\Delta \mathcal { D } _ { k , t }$ is the newly acquired evidence, and $\mathcal { R } _ { k , \iota }$ records the evidence-based assessment of the current hypotheses.

Plan: Hypothesis-Guided Candidate Generation. The agent formulates testable hypotheses from the mechanisms of the industrial search system, the optimization objective, and accumulated experimental evidence. These hypotheses describe how candidate modifications are expected to afect the objective under the corresponding experimental conditions. Each candidate in $\mathcal { A } _ { k , t }$ is associated with a hypothesis and an expected observation. Candidate generation includes interventions that examine individual efects, test interactions, or replicate previous observations.

Repeated refinement around previously promising candidates can concentrate exploration within a narrow region. To broaden candidate generation, the agent incorporates Bayesian-optimization-inspired reasoning:

$$
\mathcal { A } _ { k , t } = \mathcal { G } \big ( C _ { k } , \mathcal { M } _ { k , t } , \mathcal { D } _ { k , t } , \Omega _ { k , t } ; \mathcal { K } _ { \mathrm { B O } } \big ) , \qquad \mathcal { A } _ { k , t } \subseteq \Omega _ { k , t } ,\tag{4}
$$

where $\mathcal { G }$ denotes the agent’s candidate-generation procedure and $\mathcal { K } _ { \mathrm { B O } }$ denotes Bayesian-optimization-inspired reasoning principles. Conditioned on the search-system context, current hypotheses, and accumulated evidence, this procedure combines refinement of evidence-supported regions with exploration of underexplored regions. These principles guide qualitative reasoning about observed efects and uncertainty rather than specifying a fitted numerical optimization model.

Execute: Experiment Execution and Evidence Collection. The agent invokes stage-specific experimentation tools to apply candidate modifications within the corresponding evaluation environments and initiate experiments under Confidence-Gated Verifier Ladder. As candidates progress through the evaluation stages, the agent retrieves the evaluation results and promotion decisions produced by the corresponding verifiers. These outputs form the newly acquired evidence $\Delta \mathcal { D } _ { k , t }$ , with each record linked to its candidate, hypothesis, and experimental context.

Evaluate: Contextual Hypothesis Assessment. The agent assesses each hypothesis using its associated evidence in $\mathcal { D } _ { k , t } \cup \Delta \mathcal { D } _ { k , t }$ . This assessment considers the candidate modifications, evaluation stages, and experimental contexts underlying the observed efects. Consistent observations provide support for a hypothesis. Conflicting observations motivate examination of its assumptions and applicability conditions. Inconclusive observations remain unresolved evidence for further experimentation. The assessment record $\mathcal { R } _ { k , t }$ preserves these distinctions for hypothesis revision.

Update: Evidence-Based Hypothesis Revision. The assessment record updates the hypotheses and the candidate intervention space:

$$
\begin{array} { r } { \left( \boldsymbol { M } _ { k , t + 1 } , \Omega _ { k , t + 1 } \right) = \mathcal { U } ( \boldsymbol { M } _ { k , t } , \Omega _ { k , t } , \mathcal { R } _ { k , t } ) , } \\ { \mathcal { D } _ { k , t + 1 } = \mathcal { D } _ { k , t } \cup \Delta \mathcal { D } _ { k , t } , } \end{array}\tag{5}
$$

where $\mathcal { U }$ denotes evidence-based revision of the hypotheses and candidate intervention space. Supported hypotheses guide further refinement and replication. Conflicting evidence motivates revised hypotheses and alternative candidate modifications within the specified intervention scope. Evaluation results and promotion decisions remain available in the updated state, allowing subsequent proposals to build on accumulated evidence and its experimental context.

## 5.2 Confidence-Gated Verifier Ladder

Confidence-Gated Verifier Ladder organizes heterogeneous candidate evaluations into four levels: Ofline Replay (L1), Shadow-Trafic Evaluation (L2), Rapid Online Evaluation (L3), and Decision-Grade Online Evaluation (L4). All four levels are aligned with the same optimization objective but difer in signal source, evaluation fidelity, acquisition cost, trafic exposure, and observation horizon. L1 and L2 use a model-based proxy score that estimates this objective from ranking outputs, and L3 and L4 use the objective directly observed from online user outcomes. Each strategy task instantiates an ordered sequence of active levels according to data availability and validation requirements. A unified confidence-based promotion rule connects consecutive levels in this task-specific sequence. Each active level produces evaluation results under its corresponding experimental protocol. Evaluation results and promotion decisions provide evidence for hypothesis assessment and subsequent search.

## 5.2.1 Unified Promotion Policy

The promotion policy uses the estimated candidate efect and its confidence interval at each evaluation stage. For candidate � at stage ℓ, the verifier produces

$$
\begin{array} { r } { \mathcal { E } _ { \ell } ( c ) = \big ( c , b _ { \ell } , m _ { \ell } , n _ { \ell } ^ { c } , n _ { \ell } ^ { b _ { \ell } } , \widehat { \Delta } _ { \ell } ( c ) , \mathbf { C I } _ { \ell } ( c ) , \nu _ { \ell } ( c ) \big ) , } \end{array}\tag{6}
$$

where $\ \varepsilon _ { \ell } ( c )$ is the evidence record produced for candidate � at stage ℓ, which packages the comparison and its statistical summary into a single object. Within it, $b _ { \ell }$ is the stage-specific baseline and $m _ { \ell }$ is the evaluation signal. The quantities $n _ { \ell } ^ { c }$ and $n _ { \ell } ^ { b _ { \ell } }$ denote the numbers of valid evaluation units for the candidate and baseline. $\widehat { \Delta } _ { \ell } ( c )$ denotes the estimated candidate efect, $\mathrm { C I } _ { \ell } ( c )$ its confidence interval, and $\nu _ { \ell } ( c )$ the promotion decision. L1 and L2 use model-based proxy scores. L3 and L4 use observed online outcomes. This record is the unit of evidence accumulated in the research state and interpreted together with its experimental context.

Let $\widehat { \mu } _ { \ell } ( c )$ and $\widehat { \mu } _ { \ell } ( b _ { \ell } )$ denote the estimated values of the evaluation signal for the candidate and baseline. With the signal oriented so that larger values indicate better outcomes, the estimated efect is

$$
\begin{array} { r } { \widehat { \Delta } _ { \ell } ( c ) = \widehat { \mu } _ { \ell } ( c ) - \widehat { \mu } _ { \ell } ( b _ { \ell } ) . } \end{array}\tag{7}
$$

The confidence interval is centered on this efect and widened by its standard error,

$$
\begin{array} { r } { \mathbf { C I } _ { \ell } ( c ) = \Big [ \widehat { \Delta } _ { \ell } ( c ) - t _ { \nu _ { \ell } , 1 - \alpha / 2 } \mathbf { S } \mathbf { E } _ { \ell } ( c ) , ~ \widehat { \Delta } _ { \ell } ( c ) + t _ { \nu _ { \ell } , 1 - \alpha / 2 } \mathbf { S } \mathbf { E } _ { \ell } ( c ) \Big ] , } \end{array}\tag{8}
$$

where $\mathrm { S E } _ { \ell } ( c )$ is the standard error of the efect, $t _ { \nu _ { \ell } , 1 - \alpha / 2 }$ is the critical value at significance level $\alpha$ and $\nu _ { \ell }$ the associated degrees of freedom. The evidence representation and promotion rule are shared across levels, whereas efect estimation follows the sampling structure of each evaluation protocol. The shared representation retains stage-specific signal definitions and confidence intervals rather than combining heterogeneous results into a single score. The stage-specific efect estimators and confidence computations are detailed in Appendix A.

Let $\mathrm { C I } _ { \ell } ( c ) = [ L _ { \ell } ( c ) , U _ { \ell } ( c ) ]$ denote the confidence interval at the prescribed confidence level. The promotion decision is

$$
\nu _ { \ell } ( c ) = \left\{ \begin{array} { l l } { \mathsf { P R O M O T E } , } & { L _ { \ell } ( c ) > 0 , } \\ { \mathsf { R E T A l N } , } & { L _ { \ell } ( c ) \leq 0 \leq U _ { \ell } ( c ) , } \\ { \mathsf { S T O P } , } & { U _ { \ell } ( c ) < 0 . } \end{array} \right.\tag{9}
$$

PROMOTE indicates statistically significant evidence of a positive efect and advances the candidate to the next evaluation stage. At L4, it indicates that the candidate satisfies the promotion rule for deployment consideration. RETAIN indicates inconclusive evidence and retains the candidate for further evaluation at the current stage. STOP indicates statistically significant evidence of a negative efect and terminates the candidate’s progression through the ladder.

The promotion decision progressively narrows the candidate set along the active evaluation sequence. For strategy task �, let $\mathcal { L } _ { k } = ( \ell _ { k , 1 } , \dots , \ell _ { k , J _ { k } } )$ denote its ordered sequence of active levels, where $\ell _ { k , j } \in \{ 1 , 2 , 3 , 4 \}$ and $\ell _ { k , j } \ : < \ell _ { k , j + 1 }$ . The candidates entering the first active level are those generated by Evidence-driven AutoResearch, $\mathcal { A } _ { k , t } ^ { [ 1 ] } = \mathcal { A } _ { k , t }$ , and each active level advances only the candidates it promotes:

$$
\mathcal { A } _ { k , t } ^ { [ j + 1 ] } = \big \{ c \in \mathcal { A } _ { k , t } ^ { [ j ] } : \nu _ { \ell _ { k , j } } ( c ) = \mathsf { P R O M O T E } \big \} , \qquad j \in \{ 1 , \dots , J _ { k } \} ,\tag{10}
$$

which yields the nested selection $\mathcal { A } _ { k , t } ^ { [ 1 ] } \supseteq \mathcal { A } _ { k , t } ^ { [ 2 ] } \supseteq \cdots \supseteq \mathcal { A } _ { k , t } ^ { [ J _ { k } + 1 ] }$ . A candidate reaches an active level only after being promoted at every preceding level in the task-specific sequence. For deployment consideration, the sequence terminates at Decision-Grade Online Evaluation (L4), and $\mathcal { A } _ { k , t } ^ { [ J _ { k } + 1 ] }$ contains the candidates promoted at that level. Thus, lower-cost active levels screen candidates before costly online evaluation.

Evaluation results and promotion decisions are returned to the corresponding research state together with their experimental contexts. For strategy task � in experimental round $\hat { t } , \mathcal { E } _ { k , t } ^ { ( \ell ) }$ collects the records of candidates evaluated at stage ℓ. These records support evidence interpretation across experiments while preserving the conditions under which each result was obtained.

## 5.2.2 Model-Based Proxy Evaluation

L1 and L2 evaluate candidate ranking outputs without exposing users to those outputs. A list-level evaluation model provides a proxy score for each candidate. Given request context � and the ranked list $L ^ { g } ( x )$ produced by strategy �, the proxy evaluation signal is

$$
m _ { \ell } ( g ; x ) = R _ { \psi } ( x , L ^ { g } ( x ) ) , \qquad \ell \in \{ 1 , 2 \} ,\tag{11}
$$

where $R _ { \psi }$ estimates the online objective from the request context and the complete ranked list. The model accounts for the composition and relative positions of multiple content formats. It therefore evaluates the combined ranking output rather than individual content scores in isolation.

The resulting proxy scores provide the evaluation signal for estimating candidate efects at L1 and L2. Both stages use the same scoring model but evaluate candidates under diferent request and execution conditions. Ofline Replay uses fixed historical requests. Shadow-Trafic Evaluation uses duplicated real-time requests in the online execution environment.

## 5.2.3 L1: Ofline Replay

L1 evaluates candidate modifications on a fixed set of historical requests. The candidate and baseline ranking procedures process the recorded request contexts and produce their respective ranked lists. The evaluation model assigns a proxy score to each list. For strategy $g \in \{ c , b _ { 1 } \}$ , the estimated proxy score is

$$
\widehat { \mu } _ { 1 } ( g ) = \widehat { \mathbb { E } } _ { x \sim \chi _ { \mathrm { h i s t } } } [ m _ { 1 } ( g ; x ) ] ,\tag{12}
$$

where $\chi _ { \mathrm { h i s t } }$ is the historical request set over which the proxy signal is averaged. Because the candidate and baseline process the same historical requests, L1 estimates the candidate efect from request-level paired diferences.

The verifier forms the efect $\widehat { \Delta } _ { 1 } ( c )$ and its confidence interval $\mathrm { C I } _ { 1 } ( c )$ from these scores under the Ofline Replay protocol. This evidence supports low-cost candidate screening. The promoted subset $\mathcal { A } _ { k , t } ^ { ( 2 ) }$ proceeds to Shadow-Trafic Evaluation.

## 5.2.4 L2: Shadow-Trafic Evaluation

L2 evaluates candidate modifications using duplicated real-time requests in the industrial search system. Candidate ranking procedures execute with current request contexts and online features. Their outputs are used for evaluation without replacing the results shown to users. The per-request proxy signal $m _ { 2 } { \left( g ; x \right) }$ follows the same model-based definition, with $L ^ { g } ( x )$ produced in the online execution environment. The scoring model is shared with L1, but its inputs reflect current requests, online features, and the ranking outputs generated under these conditions. For strategy $g \in \{ c , b _ { 2 } \}$ , the estimated proxy score is

$$
\widehat { \mu } _ { 2 } ( g ) = \widehat { \mathbb { E } } _ { x \sim \chi _ { \mathrm { h a d o w } } ^ { g } } \left[ m _ { 2 } ( g ; x ) \right] ,\tag{13}
$$

where $\chi _ { \mathrm { s h a d o w } } ^ { g }$ is the set of requests received by the shadow-trafic group for strategy $g .$ . Candidate and baseline statistics are computed from independent shadow-trafic groups that receive requests replicated from the live trafic stream, without request-level pairing across groups.

The verifier forms the efect $\widehat { \Delta } _ { 2 } ( c )$ and its confidence interval $\mathrm { C I } _ { 2 } ( c )$ from these scores, which determine candidate promotion under the unified policy. This stage examines candidate performance under current operating conditions while retaining model-based proxy evaluation. The promoted subset $\mathcal { A } _ { k , t } ^ { ( 3 ) }$ proceeds to Rapid Online Evaluation.

## 5.2.5 L3: Rapid Online Evaluation

L3 applies candidate modifications to limited randomized trafic. Users in the candidate group receive results produced by the candidate strategy, and users in the baseline group receive results produced by the baseline strategy. The experiment collects observed user outcomes over a short observation window. These outcomes replace model-based proxy scores as the evaluation signal.

Let $\mathcal { N } _ { 3 } ^ { g }$ denote the evaluation units observed under strategy $^ { g , }$ , and let $y _ { 3 } ( g ; u )$ denote the observed objective signal for unit �. For strategy $g \in \{ c , b _ { 3 } \}$ , the estimated value is

$$
\widehat { \mu } _ { 3 } ( g ) = \widehat { \mathbb { E } } _ { u \sim N _ { 3 } ^ { g } } \left[ y _ { 3 } ( g ; u ) \right] ,\tag{14}
$$

where the expectation is taken over the units randomly assigned to each strategy. Evaluation units and outcome definitions follow the online experimental protocol. The verifier forms the efect $\widehat { \Delta } _ { 3 } ( c )$ and its confidence interval $\mathrm { C I } _ { 3 } ( c )$ under that protocol and applies the unified promotion policy. This stage assesses whether candidates selected through proxy evaluation produce positive efects in real-user outcomes. The promoted subset $\mathcal { A } _ { k , t } ^ { ( 4 ) }$ proceeds to Decision-Grade Online Evaluation.

## 5.2.6 L4: Decision-Grade Online Evaluation

L4 evaluates promoted candidates through online experiments with broader randomized trafic and a longer observation window. Candidate and baseline groups provide observed outcomes aligned with the optimization objective. The expanded experiment covers a wider range of trafic conditions and temporal variation.

Using the same notation for observed objective signals, the estimated value is

$$
\widehat { \mu } _ { 4 } ( g ) = \widehat { \mathbb { E } } _ { u \sim \mathcal { N } _ { 4 } ^ { g } } \left[ y _ { 4 } ( g ; u ) \right] , \qquad g \in \{ c , b _ { 4 } \} ,\tag{15}
$$

where $\mathcal { N } _ { 4 } ^ { g }$ contains the evaluation units used in Decision-Grade Online Evaluation and $y _ { 4 } ( g ; u )$ is the objective signal measured over its observation window. The resulting efect $\widehat { \Delta } _ { 4 } ( c )$ and its confidence interval $\mathrm { C I } _ { 4 } ( c )$ support the final deployment decision. Candidates satisfying the promotion rule at L4 are eligible for deployment consideration. Evaluation results and final decisions update the corresponding research states and inform subsequent search.

## 6 Experiments

## 6.1 Experimental Setup and Research Questions

The experiments evaluate PEAR in a real-world industrial search system. Three anonymized strategy tasks provide records of candidate exploration and evaluation. These records include experimental configurations, evaluation results, and hypothesis revisions. A/B results from Decision-Grade Online Evaluation are reported for strategies identified for two of these tasks.

The evaluation addresses three research questions:

RQ1: Evidence-driven hypothesis revision. How are accumulated evaluation results incorporated into hypothesis revision and subsequent candidate generation?

RQ2: Progressive candidate evaluation. How do proxy evaluation results relate to observed online outcomes, and how do selected candidates behave across evaluation stages?

RQ3: Online efectiveness. Do strategies optimized with PEAR improve the online optimization objective during Decision-Grade Online Evaluation?

Implementation Detail. Experiments at L1–L3 run on timescales of hours, and L4 experiments run on timescales of days. Main Order/DAU is the primary outcome reported for the final online A/B experiments; the remaining reported outcomes are auxiliary business metrics for the e-commerce search setting. Trafic conditions, observation windows, and evaluation signals difer across stages. Cross-stage analysis therefore examines the direction of candidate efects rather than directly comparing their magnitudes. For industrial search optimization, each strategy task instantiates an ordered sequence of active evaluation levels according to data availability and validation requirements. Candidates must satisfy the promotion criterion at each active level before advancing to the next level in that sequence. In the following experiments, data-access restrictions across regions preclude the use of Ofline Replay (L1), so the evaluated sequence begins at Shadow-Trafic Evaluation (L2) and the subsequent experimental sections do not involve L1 validation. Although L1 is not used in these experiments, it has assisted algorithm engineers in optimizing other strategies in production.

## 6.2 RQ1: Evidence-Driven Hypothesis Revision

Table 1 L2 search scale and candidate selection. Unique configurations are deduplicated within each phase; promotion rate is the number of candidates promoted from L2 divided by the number of experimental groups.
<table><tr><td>Task phase</td><td>Experimental rounds</td><td>Groups</td><td>Unique configs.</td><td>Promoted from L2</td><td>Promotion rate</td></tr><tr><td>A</td><td>19</td><td>133</td><td>20</td><td>2</td><td>1.50%</td></tr><tr><td>B-I</td><td>12</td><td>84</td><td>23</td><td>5</td><td>5.95%</td></tr><tr><td>B-ⅡI</td><td>2</td><td>14</td><td>6</td><td>1</td><td>7.14%</td></tr><tr><td>B-III</td><td>21</td><td>147</td><td>53</td><td>2</td><td>1.36%</td></tr><tr><td>C</td><td>16</td><td>76</td><td>54</td><td>3</td><td>3.95%</td></tr></table>

Candidate exploration and selection. Table 1 summarizes candidate exploration at L2 for three anonymized strategy tasks. Each task evaluates multiple configurations and retains a small subset for online evaluation. Task B comprises three consecutive search phases: the first two explore local parameter neighborhoods, and Phase III expands the candidate range based on accumulated evaluation results. The following trajectories examine how this evidence informs hypothesis revision and subsequent candidate generation.

Table 2 Representative hypothesis-revision trajectory for Task A. The table shows four early experimental rounds from a 19-round search; $G = ( w , q )$ denotes the global window and capacity constraint.
<table><tr><td>Experimental round / research question</td><td>Controlled interventions</td><td>Evidence-driven update</td></tr><tr><td>1. Which constraint axis accounts for the observed gain?</td><td>Apply one-axis changes to global and format-specific constraints; also test the coupled global change  $G = ( 3 , 2 )$ </td><td>Only  $G = ( 3 , 2 )$  yields a significant positive order signal. The effects of increasing w, increasing q, and experimental variation remain entangled.</td></tr><tr><td>2. Can  $G = ( 3 , 2 )$  be reproduced, and which variable explains its gain?</td><td>Replicate  $G = ( 3 , 2 ) ;$  compare window-only  $G = ( 3 , 1 )$  , capacity-only  $G = ( 2 , 2 )$  , and boundary settings including  $G = ( 3 , 3 ) .$ </td><td> $G = ( 3 , 3 )$  produces a stronger signal. Replications of  $G = ( 3 , 2 )$  yield mixed directional effects and indicate potential degradation in the exit metric. The next experimental round prioritizes capacity and</td></tr><tr><td>3. Is the gain driven by the coupled Replicate change or by capacity alone?</td><td> $G = ( 3 , 3 )$  and  $G = ( 3 , 2 )$  test capacity-only  $G = ( 2 , 2 )$  and wider-window boundary settings.</td><td>replication.  $G = ( 2 , 2 )$  achieves the strongest observed result. The revised hypothesis attributes the gain primarily to capacity relaxation without requiring a wider window.</td></tr><tr><td>4. Are the marginal effects of window and capacity stable?</td><td>Concentrate replications on  $G = ( 2 , 2 ) ,$   $G = ( 3 , 2 )$  and  $G = ( 3 , 3 )$  retain window-only  $G = ( 3 , 1 )$  as a counterfactual.</td><td> $G = ( 3 , 3 )$  remains directionally favorable on order and exit signals but is not significant. The capacity-centered hypothesis is retained for further replication.</td></tr></table>

Hypothesis revision through controlled interventions. Task A illustrates how controlled interventions distinguish competing explanations for an observed gain. The task optimizes constraints on e-commerce result density. Let $G = ( w , q )$ denote the global constraint, where � is the window size and $q$ is the maximum content capacity within that window; the deployed baseline is $G = ( 2 , 1 )$ . Table 2 traces four representative early experimental rounds. Other constraint families remain at their baseline values after initial screening.

The initial gain from $G \ : = \ : ( 3 , 2 )$ leaves the efects of window size and capacity unresolved. The next experimental round separates these factors and replicates the coupled setting. Mixed directional efects across replications motivate further tests of capacity relaxation. In Experimental Round 3, the capacity-only setting $G = ( 2 , 2 )$ achieves the strongest observed result, shifting the hypothesis toward capacity relaxation rather than window expansion. Experimental Round 4 then examines the consistency of this hypothesis through additional controlled comparisons and replication.

Hypothesis revision from online evaluation. Task B optimizes position-dependent exposure settings for e-commerce search results. Table 3 summarizes three search phases and their outcomes from Decision-Grade Online Evaluation.  
Table 3 Outcomes from Decision-Grade Online Evaluation across search phases for Task B.
<table><tr><td>Phase</td><td>Experimental rounds</td><td>Groups</td><td>Unique configs.</td><td>Promoted from L2</td><td>L4 outcome</td></tr><tr><td>I</td><td>12</td><td>84</td><td>23</td><td>5</td><td>No significant gain</td></tr><tr><td>ⅡI</td><td>2</td><td>14</td><td>6</td><td></td><td>Positive, not significant</td></tr><tr><td>ⅢII</td><td>21</td><td>147</td><td>53</td><td></td><td>Significant order gain</td></tr></table>

In the first two phases, candidates promoted from L2 do not achieve significant improvements at L4. These online evaluation results are incorporated into the research state to reassess the hypotheses underlying candidate generation. The revised hypotheses guide further exploration at L2, expanding the candidate range beyond the initial parameter neighborhoods. Figure 2 illustrates this exploration process: the agent uses accumulated evaluation evidence to revise hypotheses, terminate unsupported branches, and redirect candidate generation. In Phase III, the revised exploration identifies candidates for further online evaluation, one of which subsequently achieves a significant improvement at L4. The trajectory illustrates how evidence from Decision-Grade Online Evaluation is incorporated into hypothesis revision and subsequent candidate generation, connecting candidate promotion with continued search.

× hypothesis test failed -O hypothesis validated  branch endpoint  
![](images/e70ef2391fecfdad37849f49d6eb96239aea4dc03f1266252e5c0e2cac77e864.jpg)  
Figure 2 Hypothesis-driven search trajectory for Task B. Candidate evaluations update a retained hypothesis state through validation, rejection, and branch termination until Decision-Grade Online Evaluation.

## 6.3 RQ2: Progressive Candidate Evaluation

Aggregate agreement between proxy predictions and online outcomes. An independent controlled experiment compares mean predicted and observed click, order, and GMV outcomes at a common session granularity. Table 4 reports the signed relative errors between these aggregate means. The order component has the smallest absolute relative error, with a signed value of −3.49%. This comparison assesses aggregate agreement rather than the accuracy of predicted efects for individual candidate interventions.

Table 4 Signed relative errors between the aggregate means of proxy predictions and observed online outcomes.
<table><tr><td></td><td>Click</td><td>Order</td><td>GMV</td></tr><tr><td>Signed relative error</td><td>-16.02%</td><td>-3.49%</td><td>-18.23%</td></tr></table>

Candidate efects across evaluation stages. Table 5 follows two Task A candidates from L2 through L3 to L4. Both traced candidates have positive order proxy efects at L2 and retain positive observed efects at L3 and L4. These cases illustrate directional consistency from proxy-based screening to Decision-Grade Online Evaluation.

Together, the aggregate comparison and the two candidate trajectories provide illustrative evidence that proxy evaluation can support candidate screening, while online evaluation remains necessary to establish efects under real-user trafic.

Table 5 Candidate efects across evaluation stages for two Task A candidates. L2 reports changes in the order proxy, L3 reports changes in observed orders, and L4 reports changes in Main Order/DAU.
<table><tr><td>Candidate</td><td>L2: Shadow-Traffic Evaluation</td><td>L3: Rapid Online Evaluation</td><td>L4: Decision-Grade Online Evaluation</td></tr><tr><td>A</td><td>Order proxy +45.88%</td><td>Observed order +15.48%</td><td>Main Order/DAU +2.73%</td></tr><tr><td>B</td><td>Order proxy +61.64%</td><td>Observed order +29.76%</td><td>Main Order/DAU +2.16%</td></tr></table>

## 6.4 RQ3: Online Efectiveness

Table 6 reports Decision-Grade Online Evaluation results for strategies optimized with PEAR for two search optimization tasks. Relative to their respective baselines, the strategies for Task A and Task B significantly improve Main Order/DAU by 2.7336% and 3.2957%, respectively. Both strategies also achieve significant gains in ASN, SKU Order/DAU, Main OPMS, and SKU OPMS.

Table 6 Relative metric changes in online A/B evaluations of strategies optimized with PEAR for two search optimization tasks. All values are relative to the corresponding baseline; dark-green boldface indicates statistical significance at � < 0.05.
<table><tr><td>Task</td><td>SearchPV/DAU</td><td>ASN</td><td>GMV/DAU</td><td>Main Order/DAU</td><td>SKU Order/DAU</td><td>PayPV/PV</td><td>Main OPMS</td><td>SKU OPMS</td></tr><tr><td>A</td><td>+0.0063%</td><td>+1.1512%</td><td>+2.2136%</td><td>+2.7336%</td><td>+2.7259%</td><td>+1.0441%</td><td>+2.7407%</td><td>+2.7375%</td></tr><tr><td>B</td><td>+0.0470%</td><td>+1.3841%</td><td>+1.6633%</td><td>+3.2957%</td><td>+2.9826%</td><td>+1.6298%</td><td>+3.2595%</td><td>+2.9448%</td></tr></table>

## 7 Conclusion

PEAR is an AutoResearch framework for industrial search optimization under non-stationary outcomes and heterogeneous evaluation signals. Evidence-driven AutoResearch associates hypotheses with evidence and its experimental context to support reliable search knowledge accumulation. Confidence-Gated Verifier Ladder connects four evaluation levels through a unified confidence-based promotion gate. The two components jointly turn evaluation results into signals that guide both candidate promotion and hypothesis revision, linking lower-cost exploration with Decision-Grade Online Evaluation.

Experiments in a real-world industrial search system show how accumulated evidence informs controlled interventions and how online evaluation results redirect subsequent search. Strategies optimized with PEAR increased Main Order/DAU by 2.7336% and 3.2957% relative to their respective baselines in two A/B experiments at the Decision-Grade Online Evaluation stage. These results support the use of contextual evidence and progressive evaluation for iterative industrial search optimization. Future work will examine broader intervention scopes, including model architectures and training procedures.

## 8 Limitations

The current work is confined to a single e-commerce general search environment, and PEAR’s generality across other search platforms, geographic regions, and evaluation configurations remains to be established. Where compliance constraints preclude Ofline Replay (L1), evaluation proceeds from L2 onward while retaining the higher-fidelity validation stages. The search space is currently restricted to strategy-level parameters; its extension to model parameters, architectures, and training procedures is left to future work. Confidence-Gated Verifier Ladder also depends on the compute clusters, deployed evaluation and search models, and live trafic of the underlying search environment, which constrains its applicability in settings where such resources are unavailable.

## 9 Ethical Considerations

In the evaluated deployment, AutoResearch is employed to optimize an e-commerce general search system. As optimization feedback, the agents receive model-based proxy scores produced by the list-level evaluation model and aggregate efect estimates of the objective signal from online experiments, rather than userlevel records, personal identifiers, sensitive attributes, or other sensitive information. All data used for evaluation and experimentation are processed under applicable data-governance and compliance requirements. The optimization loop therefore operates on compliant evaluation signals within the existing authorized experimentation pipeline and does not require additional access to personal or sensitive data.

Before any candidate is exposed to real-user trafic, the proposed change and supporting evidence undergo human review and require explicit human confirmation; automated promotion within the Verifier Ladder cannot independently initiate such exposure.

## 10 Acknowledgments

We thank our colleagues on the E-commerce General Search, Architecture, Product, and Data Science teams for their invaluable contributions to the design, development, and deployment of the systems described in this work. We are grateful to the AutoSearch tooling team for building the compute cluster and infrastructure that enabled large-scale AutoResearch for industrial search system optimization.

## Author List

Global E-Commerce Agentic Search Team: Yifan Wang<sup>\*</sup>, Shipeng Zhu, Fei Xiong, Yuqin Yang, Yonghui Huang, Kunyao Wu, Yue Wang, Weichao Meng, Yu Gong<sup>†</sup>.

<sup>\*</sup>: First author, yifanwang993w@gmail.com

<sup>†</sup>: Corresponding author, gy910210@gmail.com

## References

[1] Boiko, D. A., MacKnight, R., Kline, B., and Gomes, G. (2023). Autonomous chemical research with large language models. Nature, 624(7992):570–578.

[2] Chen, G., Chen, J., Chen, L., Zhao, J., Meng, F., Zhao, W. X., Song, R., Chen, C., Wen, J.-R., and Jia, K. (2026). Toward autonomous long-horizon engineering for ml research. arXiv preprint arXiv:2604.13018.

[3] Chen, Y., Liu, Q., Zhang, Y., Sun, W., Ma, X., Yang, W., Shi, D., Mao, J., and Yin, D. (2025). Tourrank: Utilizing large language models for documents ranking with a tournament-inspired strategy. In Proceedings ofthe ACM on Web Conference 2025, WWW ’25, page 1638–1652, New York, NY, USA. Association for Computing Machinery.

[4] Cheng, Y., Zhou, L., Liang, X., Luo, D., Lee, T., Zheng, K., Zhang, W., Cai, M., Dong, J., and Zhang, A. (2026). Let the agent steer: Closed-loop ranking optimization via influence exchange. arXiv preprint arXiv:2603.27765.

[5] Huang, Q., Vora, J., Liang, P., and Leskovec, J. (2024). MLAgentBench: Evaluating language agents on machine learning experimentation. In Proceedings ofthe 41st International Conference on Machine Learning.

[6] Kandasamy, K., Dasarathy, G., Schneider, J., and Póczos, B. (2017). Multi-fidelity bayesian optimisation with continuous approximations. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70, pages 1799–1808. PMLR.

[7] Kim, S., Park, S., Kang, H., Kim, W., Seo, J., In, Y., Yoon, K., Jeon, H., and Park, C. (2026). Self-evolverec: Self-evolving recommender systems with llm-based directional feedback. arXiv preprint arXiv:2602.12612.

[8] Kohavi, R., Longbotham, R., Sommerfield, D., and Henne, R. M. (2009). Controlled experiments on the web: Survey and practical guide. Data Mining and Knowledge Discovery, 18:140–181.

[9] Lao, C., Pan, F., Ma, G., Li, H., Lin, H., Shi, J., Zhao, K., Gai, K., Zhou, M., Zhou, Q., et al. (2026). Agentx: Towards agent-driven self-iteration of industrial recommender systems. arXiv preprint arXiv:2606.26859.

[10] Lee, Y., Nair, R., Zhang, Q., Lee, K., Khattab, O., and Finn, C. (2026). Meta-harness: End-to-end optimization of model harnesses. arXiv preprint arXiv:2603.28052.

[11] Li, L., Chu, W., Langford, J., and Wang, X. (2011). Unbiased ofline evaluation of contextual-bandit-based news article recommendation algorithms. In Proceedings of the Fourth ACM International Conference on Web Search and Data Mining, pages 297–306.

[12] Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., and Talwalkar, A. (2018). Hyperband: A novel bandit-based approach to hyperparameter optimization. Journal ofMachine Learning Research, 18(185):1–52.

[13] Liu, J., Qiu, S., Li, M., Li, B., Ji, H., Han, S., Ye, X., Xia, P., Dong, Z., Chen, M., et al. (2026). Autoresearchclaw: Self-reinforcing autonomous research with human-ai collaboration. arXiv preprint arXiv:2605.20025.

[14] Liu, Z., Li, C., Xiao, S., Li, C., Zhang, C. J., Liao, H., Lian, D., and Shao, Y. (2025). Fitting into any shape: A flexible llm-based re-ranker with configurable depth and width. In Proceedings ofthe ACM on Web Conference 2025, WWW ’25, page 3942–3951, New York, NY, USA. Association for Computing Machinery.

[15] Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., and Ha, D. (2024). The AI scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292.

[16] Mann, T. A., Gowal, S., Gyorgy, A., Hu, H., Jiang, R., Lakshminarayanan, B., and Srinivasan, P. (2019). Learning from delayed outcomes via proxies with applications to recommender systems. In Proceedings ofthe 36th International Conference on Machine Learning, volume 97, pages 4324–4332. PMLR.

[17] Park, J. S., O’Brien, J., Cai, C. J., Morris, M. R., Liang, P., and Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, pages 1–22.

[18] Qian, C., Liu, W., Liu, H., Chen, N., Dang, Y., Li, J., Yang, C., Chen, W., Su, Y., Cong, X., Xu, J., Li, D., Liu, Z., and Sun, M. (2024). ChatDev: Communicative agents for software development. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15174–15186.

[19] Romera-Paredes, B., Barekatain, M., Novikov, A., Balog, M., Kumar, M. P., Dupont, E., Ruiz, F. J. R., Ellenberg, J. S., Wang, P., Fawzi, O., Kohli, P., and Fawzi, A. (2024). Mathematical discoveries from program search with large language models. Nature, 625(7995):468–475.

[20] Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Hambro, E., Zettlemoyer, L., Cancedda, N., and Scialom, T. (2023). Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems, volume 36.

[21] Tie, G., Shi, J., Song, D., Huang, Y., Sheng, Z., Zhou, X., Liu, D., Zhou, P., Chen, Y., Xu, R., et al. (2026). Autoresearch ai: Towards ai-powered research automation for scientific discovery. arXiv preprint arXiv:2605.23204.

[22] Wang, H., Wu, Y., Chang, D., Wei, L., and Heldt, L. (2026). Self-evolving recommendation system: End-to-end autonomous model optimization with llm agents. arXiv preprint arXiv:2602.10226.

[23] Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., Tang, J., Chen, X., Lin, Y., Zhao, W. X., Wei, Z., and Wen, J. (2024). A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345.

[24] Yao, S., Yu, D., Zhao, J., Shafran, I., Grifiths, T., Cao, Y., and Narasimhan, K. (2023a). Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems, volume 36, pages 11809–11822.

[25] Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. (2023b). React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations.

[26] Yoon, S., Kim, G., Cho, G.-H., and won Hwang, S. (2025). Acurank: Uncertainty-aware adaptive computation for listwise reranking.

[27] Zhou, J., Zheng, Y., Chen, W., Zheng, Q., Su, H., Zhang, W., Meng, R., and Shen, X. (2025). Beyond content relevance: Evaluating instruction following in retrieval models.

## A Confidence Computation for the Unified Promotion Policy

This appendix specifies the stage-specific estimators used by the unified promotion policy (Section 5.2) to obtain the candidate efect $\widehat { \Delta } _ { \ell } ( c )$ and its confidence interval $\mathrm { C I } _ { \ell } ( c )$ . The evidence schema and promotion rule are shared across levels, while the estimator follows the sampling structure of each evaluation protocol. L1 uses request-aligned paired observations. L2 uses independent shadow-trafic groups, and L3 and L4 use independent randomized online groups.

Paired estimation for Ofline Replay (L1). Let $x _ { i }$ denote the �-th historical request shared by candidate � and baseline $b _ { 1 }$ , and define the request-level proxy-score diference as

$$
d _ { i } = m _ { 1 } ( c ; x _ { i } ) - m _ { 1 } ( b _ { 1 } ; x _ { i } ) , \qquad i \in \{ 1 , \ldots , n \} .\tag{16}
$$

The L1 efect, variance of the paired diferences, standard error, and degrees of freedom are

$$
\widehat { \Delta } _ { 1 } ( c ) = \bar { d } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } d _ { i } , \qquad s _ { d } ^ { 2 } = \frac { 1 } { n - 1 } \sum _ { i = 1 } ^ { n } ( d _ { i } - \bar { d } ) ^ { 2 } ,\tag{17}
$$

$$
{ \bf S E } _ { 1 } ( c ) = \frac { s _ { d } } { \sqrt { n } } , \qquad \nu _ { 1 } = n - 1 .\tag{18}
$$

This paired estimator controls for request-level variation because both strategies are evaluated on the same historical requests.

Independent-group estimation for L2–L4. For $\ell \in \{ 2 , 3 , 4 \}$ and strategy $g \in \{ c , b _ { \ell } \}$ , let $x _ { g , i } ^ { ( \ell ) }$ denote its �-th valid evaluation unit and let $n _ { g , \ell }$ be the number of such units. L2 units are collected from independent shadow-trafic groups without request-level pairing, whereas L3 and L4 units are collected from independently randomized online groups. The per-strategy mean and variance are

$$
\bar { x } _ { g , \ell } = \frac { 1 } { n _ { g , \ell } } \sum _ { i = 1 } ^ { n _ { g , \ell } } x _ { g , i } ^ { ( \ell ) } , \qquad s _ { g , \ell } ^ { 2 } = \frac { 1 } { n _ { g , \ell } - 1 } \sum _ { i = 1 } ^ { n _ { g , \ell } } \bigl ( x _ { g , i } ^ { ( \ell ) } - \bar { x } _ { g , \ell } \bigr ) ^ { 2 } .\tag{19}
$$

The estimated efect and Welch standard error are

$$
\widehat { \Delta } _ { \ell } ( c ) = \bar { x } _ { c , \ell } - \bar { x } _ { b _ { \ell } , \ell } , \qquad \mathbf { S } \mathbf { E } _ { \ell } ( c ) = \sqrt { \frac { s _ { c , \ell } ^ { 2 } } { n _ { c , \ell } } + \frac { s _ { b _ { \ell } , \ell } ^ { 2 } } { n _ { b _ { \ell } , \ell } } } ,\tag{20}
$$

with Welch–Satterthwaite degrees of freedom

$$
\nu _ { \ell } = \frac { \left( \displaystyle \frac { s _ { c , \ell } ^ { 2 } } { n _ { c , \ell } } + \displaystyle \frac { s _ { b _ { \ell } , \ell } ^ { 2 } } { n _ { b _ { \ell } , \ell } } \right) ^ { 2 } } { \displaystyle \frac { \left( s _ { c , \ell } ^ { 2 } / n _ { c , \ell } \right) ^ { 2 } } { n _ { c , \ell } - 1 } + \displaystyle \frac { \left( s _ { b _ { \ell } , \ell } ^ { 2 } / n _ { b _ { \ell } , \ell } \right) ^ { 2 } } { n _ { b _ { \ell } , \ell } - 1 } } .\tag{21}
$$

Confidence interval. For every active level, the two-sided confidence interval at significance level � is

$$
\begin{array} { r } { \mathbf { C I } _ { \ell } ( c ) = \left[ \widehat { \Delta } _ { \ell } ( c ) - t _ { \nu _ { \ell } , 1 - \alpha / 2 } \mathbf { S E } _ { \ell } ( c ) , \widehat { \Delta } _ { \ell } ( c ) + t _ { \nu _ { \ell } , 1 - \alpha / 2 } \mathbf { S E } _ { \ell } ( c ) \right] . } \end{array}\tag{22}
$$

When a relative efect is reported, the efect and interval endpoints are normalized by the corresponding baseline mean. By default, $\alpha = 0 . 0 5$ , corresponding to 95% confidence intervals.

Promotion decision. With the evaluation signal oriented so that larger values indicate better outcomes, the promotion decision applies the interval endpoints $L _ { \ell } ( c )$ and $U _ { \ell } ( c )$ as in Section 5.2. A candidate is promoted when the interval lies entirely above zero $( L _ { \ell } ( c ) > 0 )$ , which provides statistically significant evidence of a positive efect at the current significance level. It is retained for further evaluation when the interval contains zero $( L _ { \ell } ( c ) \leq 0 \leq U _ { \ell } ( c ) )$ , which indicates that the observed efect is not yet distinguishable from no efect. It is stopped when the interval lies entirely below zero $( U _ { \ell } ( c ) < 0 )$ , which provides statistically significant evidence of a negative efect. The relative interval $\mathrm { C I } _ { c } ^ { \mathrm { r e l } }$ shares the sign of the absolute interval and yields the same decision; it is reported for interpretability.

## B Prompt Templates and Structured Outputs for Research-State Updates

The following templates summarize the instructions for hypothesis revision and cross-round evidence reuse in Evidence-driven AutoResearch. They retain the core inputs, update rules, and outputs while omitting deployment-specific identifiers and serialization details.

## B.1 Hypothesis Update Prompt

## Prompt Template for Hypothesis Update

Objective and inputs. Maintain testable parameter regularities across experimental rounds. Read the completed round summary, candidate records, confidence evidence, final review, lessons, and existing hypotheses. Update instructions.

1. Generate at most three new hypotheses per round. Specify the parameter range, expected metric direction, risk boundary, and implications for candidate design; do not merely restate the best candidate.

2. Merge equivalent claims with overlapping scopes. Record narrower findings as scoped evidence without broadening their support. Keep claims with diferent directions or risk conditions separate and link them through Related.

3. Assign support when covered evidence supports the claim under the evaluation criteria, refute when it contradicts the claim or violates a decision constraint, and neutral when scope coverage or evidence is insuficient. Each round contributes at most one verdict per hypothesis.

4. Increment the corresponding counter and set Score = Support - Refute. Scores of at least 2 support active status; scores of at most −2 indicate risky hypotheses. Archive superseded or persistently unused hypotheses.

Plan use. Retrieve relevant active hypotheses with scores of at least 2 and risky hypotheses with scores of at most −2. Require a matching strategy version and explain each hypothesis’s efect on candidate design. If none applies, mark the round as exploratory. These scores summarize hypothesis evidence; they are not verifier confidence estimates.

Hypothesis Output Template.

```markdown
### H-{id}: {testable parameter regularity}
**Status:** active / risky / archived
- **Strategy Version:** {strategy fingerprint}
- **Score:** {Support - Refute}
- **Support:** {supporting-round count}
```

\*\*Refute:\*\* {refuting-round count}   
\*\*Neutral:\*\* {neutral-round count}   
\*\*Scope:\*\* {parameter ranges and experimental conditions}   
\*\*Claim:\*\* {expected metric direction under these conditions}   
\*\*Plan Use:\*\* {implications for subsequent candidate design}   
\*\*Risk:\*\* {trade-offs, applicability limits, or counterexamples}   
\*\*Evidence:\*\* {round:candidate verdict; ...}   
\*\*Related:\*\* {related hypothesis identifiers or none}   
\*\*Updated At:\*\* {timestamp}   
Plan-Stage Output Template.   
Used hypotheses: H-{id} score={value}: {design implication}   
Candidate design: {candidate}: {modification; hypothesis tested}

## B.2 Cross-Round Memory Update Prompt

## Prompt Template for Cross-Round Memory Update

Objective and inputs. Maintain two complementary records: structured lessons preserve round-level facts and reasoning; compressed memory summarizes cross-round findings and guides subsequent candidate design. Use all round summaries for the current strategy and the previous memory.

## Update instructions.

1. After the final review, append one non-failure lesson per round with an Outcome matching the recorded decision. Record execution failures separately as crash lessons with their causes.

2. Preserve the strategy version, experimental context, decision rationale, and evidence references. When summaries disagree with measured evidence or experimental records, correct the summaries.

3. Before planning, retrieve relevant lessons with a matching strategy version. Use supported findings, avoid repeatedly unsuccessful directions, and resolve recurring execution failures. Incompatible lessons remain available for audit, not direct candidate guidance.

4. Rewrite compressed memory after each round using the current strategy’s summaries and preceding memory. Preserve current-run lessons, down-weight older findings, and compress related historical lessons without losing evidence references. Provide concrete guidance for subsequent candidate design.

Outcome semantics. keep, discard, and iterate summarize the recorded round decision; pivot denotes a change of strategy family, crash an execution failure, and summary a historical synthesis.

```markdown
Lesson Output Template.
### L-{N}: {title}
**Strategy:** {candidate settings and search design}
**Strategy Version:** {strategy fingerprint}
**Outcome:** keep / discard / iterate / pivot / crash / summary
**Insight:** {actionable finding to reuse or avoid}
**Context:** goal=...; scope=...; metric=...; direction=...
**Iteration:** {round identifier}#{task identifier}
**Reason:** {rationale consistent with the recorded decision}
**Refs:** trace={record}; experiment={id}; round={id}
**Timestamp:** {timestamp}
Compressed Memory. The source protocol specifies its content rather than a fixed output schema: reusable
findings, their applicability conditions and evidence references, and concrete candidate-design guidance for the
next round.
```

## B.3 Structured Experimental Records

Structured Output Templates for Experimental Records   
Recording instructions. Link candidate designs, confidence evidence, and the final review through shared task   
and experimental-round identifiers. Retain one baseline per round and record each candidate’s parameters and   
design rationale. Base selection and the final decision on the confidence-evaluation record, not legacy score logs.   
The templates below retain selected source fields; placeholders replace deployment-specific values.   
State and decision. The source protocol uses cycle for an experimental round. Its state is RUNNING, CLOSED, or   
BLOCKED. Completed rounds record KEEP, DISCARD, or ITERATE, mapped to the corresponding lesson outcomes.   
Blocked rounds record a failure cause without a decision. These round-level labels are distinct from the verifier   
promotion labels in the Method.   
Final Review Template.   
{   
"task\_name": "<task identifier>",   
"cycle\_id": "<round identifier>",   
"stage": "review",   
"status": "DONE",   
"decision": "<KEEP | DISCARD | ITERATE>",   
"reason": "<evidence-based decision rationale>",   
"best\_version": "<selected candidate identifier>",   
"best\_selected\_by": "llm\_from\_confidence\_pipeline",   
"confidence\_method": "<evaluation method>",   
"confidence\_file": "<confidence-evidence record>",   
"best\_confidence": {},   
"strategy\_contract\_version": "<strategy fingerprint>",   
"versions\_file": "<candidate-design records>"   
}   
Evidence Trace Template.   
{   
"task\_name": "<task identifier>",   
"round": "<round label>",   
"cycle": {   
"cycle\_id": "<round identifier>",   
"status": "CLOSED",   
"best\_version": "<selected candidate identifier>",   
"confidence\_method": "<evaluation method>",   
"confidence\_file": "<confidence-evidence record>",   
"best\_confidence": {},   
"decision": "<KEEP | DISCARD | ITERATE>",   
"versions\_file": "<candidate-design records>"   
},   
"decision": "<same round decision>"   
}   
The best\_confidence object contains the selected candidate’s confidence evidence. Candidate, decision, and   
evidence references must agree across the review, trace, and lesson records.

## B.4 Knowledge-Augmented Candidate Generation Prompt

This template summarizes the Bayesian-optimization-inspired reasoning principles $\mathcal { K } _ { \mathrm { B O } }$ used in Equation 4.   
These principles guide qualitative reasoning about observed efects, uncertainty, and candidate diversity.

## Prompt Template for Knowledge-Augmented Candidate Generation

Inputs. Mutable parameters and their types, intervention bounds, baseline, candidate budget, historical evaluations, hypotheses, memory, and exact-replication requirements.

1. Classify the parameter space and diagnose the search state as cold\_start, local\_refine, stagnant, or pivot. Use repeated evidence, coverage, and contradictions rather than the largest point estimate alone.

2. Select one reasoning strategy. Use GP-inspired reasoning for low-dimensional continuous spaces: contrast supported regions with uncertainty gaps. Use TPE-inspired reasoning for mixed continuous and integer spaces: distinguish supported, unfavorable, and unresolved parameter patterns. Use SMAC-inspired reasoning for categorical or conditional spaces: compare supported branches and structural counterfactuals.

3. Form a candidate pool at least twice the available new-candidate budget. Cover refinement, boundary extension, uncertainty probes, and interaction tests. Preserve negative and inconclusive evidence when selecting candidates.

4. Under stagnation or a change of search direction, increase exploration. Include a feasible point beyond the explored range on a meaningful axis and an interaction test or uncovered branch. Remain within the intervention bounds.

5. Remove invalid, excluded, or duplicate proposals and select a diverse subset. Keep exact replications separate from new candidates: replication preserves the full configuration, while every new configuration is generated through the selected reasoning strategy.

Output Template. Report concise, evidence-backed selection summaries and candidate roles. The following layout summarizes the required fields without reproducing the full planning schema.

```erb
ideation_strategy.method: <GP-/TPE-/SMAC-inspired>
search_state: <cold_start | local_refine | stagnant | pivot>
cycle_mode: <current experiment-allocation mode>
exploration_level: <level; high for stagnant or pivot>
selection_reason: <brief evidence-backed rationale>
Allocation: <exact-confirmation and new-candidate counts>
For each candidate:
candidate_source: <confirmation_queue | algorithm>
search_role: <replication or method-specific exploration role>
```