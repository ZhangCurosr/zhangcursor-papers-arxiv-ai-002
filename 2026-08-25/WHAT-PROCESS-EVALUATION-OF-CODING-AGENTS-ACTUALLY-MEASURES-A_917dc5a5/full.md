# WHAT PROCESS EVALUATION OF CODING AGENTS ACTUALLY MEASURES: ACTION, TASK, AND STEP ARE THREE DIFFERENT LEVELS

Jiawei He<sup>1,†</sup>, Mengyu Shi<sup>2,†</sup>, Jie jia<sup>1</sup>, Xikai Yang<sup>1,∗</sup>, Dong Sun<sup>1,∗</sup>,

<sup>1</sup>Amap, Alibaba Group, China

<sup>2</sup>State Key Laboratory for Novel Software Technology, Nanjing University, Nanjing, China <sup>†</sup>Contribute equally <sup>∗</sup>Corresponding author

## ABSTRACT

Coding agents are increasingly evaluated not only by whether they solve a task, but also by how they execute it. However, existing process-level evaluations often treat action prediction, task uncertainty, and step attribution as if they were the same problem, which makes it unclear what such evaluations actually measure. In this paper, we introduce a measurement framework for process evaluation in coding agents and instantiate step-level causal attribution with SCAE, a replaybased estimator derived from a structural causal model of agent execution. Our framework combines prefix-conditioned identification, replay/intervention-based estimation, and controlled judge-information manipulation to study process evaluation at the action, task, and step levels. Experiments on 499 file-localization episodes from 12 repositories show that next actions are driven primarily by execution provenance rather than code-graph transitions, execution uncertainty is structured at the task rather than step level, and full-trace judges exhibit systematic collider bias, suggesting that current process evaluation often measures semantic relevance rather than certified causal contribution.

## 1 INTRODUCTION

Large language model (LLM) coding agents are increasingly evaluated not only by whether they solve a task, but also by how they execute it (Si et al., 2026). In practice, developers want more than a final success label: they want to understand which steps advanced the task, which steps caused failure, whether the execution remained interpretable and controllable, and whether process-level signals can be trusted for debugging, benchmarking, or training (Prasai et al., 2026). As codingagent trajectories grow longer and more tool-centric, these questions become increasingly difficult to answer from final outcomes alone (He et al., 2026).

Despite its growing importance, current process evaluation remains conceptually underspecified. Existing methods often assume that a trajectory can be meaningfully evaluated step by step, but in practice they mix together three different levels of analysis (Wang et al., 2025a; Sun et al., 2026). At the action level, the question is what the agent is likely to do next (Wen et al., 2024). At the task level, the question is whether the overall execution remains uncertain or recoverable (Wang et al., 2025b). At the step level, the question is whether a particular action causally changed the final outcome (Wang et al., 2025a). Process reward models (Setlur et al., 2025), full-trace LLM judges (Chen et al., 2026), and restart-based ablations (Kontonis et al., 2026) are frequently discussed as if they answered the same question, even though they condition on different information and target different quantities. This makes their outputs difficult to interpret and even harder to validate.

In this paper, we argue that action, task, and step are three different levels, and that process evaluation should distinguish them explicitly. To study this problem, we introduce a measurement framework for coding-agent process evaluation and instantiate step-level attribution with SCAE, a replay-based estimator derived from a structural causal model of agent execution. Rather than presenting SCAE primarily as a better attribution tool, we use it as a measurement instrument: by making step-local causal effects explicit and estimable in principle, it allows us to test what process evaluation can and cannot identify in practice.

Our setting is file localization, where an episode consists of an issue, a repository snapshot, a sequence of agent actions, and a verifiable target set of files. This setting is especially useful because it provides the ingredients required for principled process analysis: replayable environments, fully logged prefixes, and a checkable terminal outcome. We reconstruct prefixes, replay continuations, and estimate step-local effects under explicit assumptions about context observability, decoding noise, and environment replayability. When local overlap is available, the effect is estimated observationally; when the policy is effectively deterministic, we switch to an intervention branch. The same framework also allows us to analyze next-action prediction, uncertainty structure, and attribution behavior within a unified experimental design.

Our experiments reveal a consistent picture. At the action level, the next operation is often predictable, but not from the structure one might expect: the strongest signal is the agent’s own execution provenance, namely the paths it has just seen in tool outputs, rather than the code dependency graph. At the task level, execution variance is structured mainly by the instance rather than by individual steps, which implies that uncertainty is primarily a property of the task. At the step level, the identifiable causal quantity proves too weak to localize at feasible replay cost, yet full-trace judges still shift their blame systematically when later steps are made visible. This isolates a bias in the evaluation mechanism itself and suggests that many current process-level evaluators measure semantic relevance or late-trace evidence rather than certified causal contribution.

These findings lead to a simple conclusion: process evaluation is not one problem but several. Some process-level signals are useful because they predict what the agent will do next. Others are useful because they characterize how uncertain the task remains. But these signals should not automatically be interpreted as step-level causal attribution. Once the levels are separated, the apparent tractability of process evaluation becomes easier to understand: action prediction can be strong, task-level uncertainty can be structured, and step-level causal localization can still remain infeasible.

Our main contributions are as follows:

• We formulate process evaluation in coding agents as a three-level measurement problem, distinguishing action prediction, task uncertainty, and step attribution instead of treating them as a single notion of process quality.

• We introduce SCAE, a replay-based measurement framework grounded in a structural causal model, and use it to study what step-level process evaluation can identify in principle and what it can measure in practice.

• We show empirically that next actions are governed primarily by execution provenance rather than code-graph transitions, and that execution uncertainty resides mainly at the task rather than step level.

• We isolate collider bias in full-trace LLM judging through a controlled manipulation of the judge’s information set, showing that current process attribution is better interpreted as semantic relevance than as certified causal contribution.

## 2 RELATED WORK

Counterfactual Reasoning in Discrete Sequential Systems. Counterfactual reasoning over discrete action sequences is commonly grounded in the Gumbel–max structural causal model (Ravfogel et al., 2025), which also provides the core mechanism underlying SCAE. By representing categorical sampling as perturb-and-argmax (Khanmohammadi et al., 2025), this line of work yields a tractable posterior over latent noise variables conditioned on an observed draw. Building on this formulation, prior studies (Chatzi et al., 2025; Kazemi et al., 2024) have developed counterfactual explanations for action sequences under known MDPs, extended the framework to token-level generation, and analyzed how counterfactual influence decays as trajectories diverge. However, these approaches generally assume access to known transition mechanisms and a fixed action space. In contrast, LLM agents operating over code repositories provide neither. This introduces two challenges that are largely absent in standard simulators: positivity may fail locally at individual decision points, and context compaction may destroy the invertibility of the state mapping. Accordingly, SCAE does not assume a known transition mechanism; instead, it estimates one through replay. It also treats positivity as an empirical property to be measured rather than a standing assumption, and regard context compaction as an identification failure rather than an approximation error.

Step-Level Credit Assignment and Process Supervision. Process reward models supervise intermediate reasoning steps and have been shown to outperform outcome-only supervision on reasoning tasks (Zhang et al., 2026b; Setlur et al., 2025; Wang et al., 2024). Their primary purpose, however, is to improve policy learning rather than to explain the role of a particular step in a single trajectory. In practice, their supervision signals are often derived from Monte Carlo estimates over completed rollouts, for example by judging whether a given prefix can still reach a correct answer. As a result, information from downstream steps is reintroduced into the label, and the contribution of the current step is not cleanly separated from the effect of what follows. A related observation applies to hindsight and counterfactual credit assignment, which deliberately use future information to reduce the variance of policy-gradient estimates (Arjona-Medina et al., 2019; Liang et al., 2026; Yang et al., 2026). Such quantities are well suited for optimization, but they do not correspond to per-episode causal contributions. Another relevant line of work concerns axiomatic attribution methods, such as Shapley values (Horovicz & Goldshmidt, 2024), which satisfy an additive property analogous to our telescoping identity, but at the cost of exponentially many re-evaluations. In contrast, our goal is not to propose a stronger step scorer. Rather, we clarify what information a step score must condition on in order to admit a causal interpretation, show that this distinction systematically changes which step is blamed, and provide an estimator that preserves additivity through a single fitted object instead of requiring 2<sup>T</sup> re-executions.

Process-Oriented Evaluation of Coding Agents. Current evaluation of coding agents remains centered primarily on terminal task success (Merrill et al., 2026; Xu et al., 2026a; Xia et al., 2024). In parallel, a complementary line of work studies recurrent execution defects and calibrates detector outputs against human process annotations (He et al., 2026). These approaches target a different estimand: the probability that an annotator would mark a step as defective given trajectory evidence (Zhang et al., 2026a; Li & Cao, 2026). As such, they address a measurement problem about annotation agreement rather than a causal inference problem. By contrast, our focus is the causal effect of substituting one step on the final outcome. This estimand requires that the randomization induced by the logging policy be preserved and is not identifiable under context compaction (§3.4). Because the two estimands are fundamentally different, a step may be consistently judged defective by annotators while contributing no identifiable causal effect to the final outcome. Annotation agreement alone therefore cannot establish whether a process score measures what it is intended to measure. SCAE differs in that it evaluates attribution against a verifiable terminal outcome, thereby turning the question of what a process score measures into an empirically testable one rather than a purely definitional one.

Uncertainty Quantification for Language Models. Research on calibration and confidence estimation for language models has largely focused on single-turn generation, where each output is associated with a single scalar confidence value (Xia et al., 2025; Ulmer et al., 2024; Nakkiran et al., 2026). More broadly, decomposing predictive uncertainty into aleatoric and epistemic components is standard in Bayesian deep learning (Melo et al., 2024; Ross et al., 2026), while conformal prediction provides distribution-free coverage guarantees under exchangeability (Vovk et al., 2005; Angelopoulos & Bates, 2023). These tools are well suited to settings in which one prediction corresponds to one confidence estimate. The agent setting introduces two complications that they do not directly address. First, uncertainty compounds across sequential decisions. Second, the agent’s own actions determine what information becomes observable later, making uncertainty inherently trajectory-dependent. Consequently, a single terminal confidence score cannot reveal where reliability was lost, nor does it specify the level of the trajectory at which confidence should be attached. File localization itself is a classical subproblem in automated program repair, typically evaluated by retrieval-style accuracy against the files touched by a reference patch (?Xia et al., 2024). We adopt localization not as a performance target to improve, but as the minimal controllable setting in which the identification conditions required by our analysis can be satisfied. On this basis, we treat the level at which uncertainty resides as an empirical question rather than a modeling choice, and examine whether uncertainty is primarily a property of the task or of individual steps.

Structural Priors for Agent Navigation. Existing coding-agent systems make substantial use of repository structure. Retrieval and localization modules commonly rank files according to similarity or dependency information, and agent scaffolds expose repository-shaped tools through which the model navigates the codebase (Xu et al., 2026b; ?; Edge et al., 2024). As a result, the assumption that code structure guides the agent’s next move is already built into current system designs. Empirically, the files visited within a single episode do tend to cluster in the dependency graph. However, such clustering does not by itself imply that local step-to-step transitions are governed by graph structure. To our knowledge, prior work has not explicitly tested this assumption against a null model that holds the working region fixed, and therefore has not separated descriptive clustering from the actual transition mechanism. Our analysis fills this gap and shows that repository structure is better understood as a prior over the agent’s working region than as a local transition model for next-step prediction. Moreover, we identify an alternative structural signal—the agent’s own execution provenance—that is more directly predictive of local behavior. This signal provides a more appropriate conditioning basis for step-level prediction and context-selection policies.

## 3 METHOD

This section presents our methodological framework for measuring process evaluation at the action, task, and step levels. Our goal is not only to define a step-level causal quantity, but also to build a practical measurement pipeline that can be instantiated on real coding-agent traces. To this end, we introduce SCAE, a replay-based framework grounded in a structural causal model of agent execution. SCAE serves two roles in this paper. First, it provides the formal object needed to distinguish step-local causal attribution from other process-level signals. Second, it acts as a measurement instrument for testing what process evaluation can and cannot identify in practice.

At a high level, the framework takes as input a coding-agent trajectory, the corresponding repository snapshot, and a verifiable task outcome, and produces three kinds of measurements. At the action level, it characterizes what structure predicts the next operation. At the task level, it measures execution uncertainty under replayed continuations. At the step level, it estimates prefix-conditioned causal effects when the required assumptions hold. The method proceeds in three stages. We first formalize agent execution with a structural causal model and specify the conditions under which step-local effects are identified. We then estimate an anytime value function by replaying reconstructed prefixes, switching between observational replay and intervention depending on local overlap. Finally, we derive attribution, forecasting, and uncertainty quantities from the same fitted object and use them in the experiments of Section 4.

## 3.1 METHOD OVERVIEW

Figure 1 summarizes the overall pipeline. Starting from an issue, a repository snapshot, and a gold target set, we execute the agent once to obtain a factual trajectory and a terminal outcome. We then reconstruct selected prefixes from the recorded trace and resume execution from those prefixes multiple times. These replays provide two kinds of information. First, they estimate an anytime value function that tracks how the expected terminal outcome changes over the course of the trajectory. Second, they probe whether the local action distribution retains enough randomness for observational identification to be meaningful.

This distinction is important because overlap is not globally available in coding-agent execution. Some decision points remain stochastic under replay, while others are effectively deterministic. SCAE therefore uses a hybrid estimation strategy: if replayed continuations reveal local action variation, step-level quantities are estimated observationally from replay; if the policy is effectively deterministic, we substitute an alternative action and execute the resulting continuation in the real environment. This hybrid design is not introduced for convenience but forced by the data-generating process.

The same estimated object supports three downstream readings. The increments of the anytime value define step-level attribution quantities. The left endpoint defines a pre-execution forecast. The replay distribution around each point defines execution uncertainty and its decomposition. In this way, one framework supports all three levels studied in the paper while making explicit that they are not interchangeable.

![](images/8bb6ad93b04b02fe7caa05c3a62edb396b9070298ac011ae048f84245deae6a3.jpg)

Figure 1: Overview of SCAE. An episode is rolled out once to obtain $( \tau _ { 1 : T } , Y )$ . Prefix replay reconstructs $\tau _ { 1 : t }$ by rewriting the rollout log at item granularity, which requires no modification to the agent. An entropy probe at each estimation point decides which identification strategy applies: where the policy retains randomness, $v _ { t }$ is estimated observationally by fork-resampling (Proposition 1); where it is effectively deterministic, an alternative action is substituted and actually executed (Assumption 4). The single fitted object $\widehat { v } _ { t }$ then supports three readings. We report what each delivered rather than what it promised: the increments do not localize at feasible cost (§4.5), the trajectory yields a task-level rather than step-level uncertainty (§4.5), and the endpoint $v _ { 0 }$ does not forecast the outcome (leave-one-repository-out AUROC 0.481). The decomposition of §3.8 is validated on a synthetic generating process, not on these traces. In the figure, shaded boxes are data and plain boxes are pipeline stages; the two branches below the entropy probe are the two cases of equation 14. Algorithm 1 in Appendix B.1 states the procedure.

## 3.2 PROBLEM FORMULATION

We study step-level process evaluation through a prefix-conditioned causal quantity. Let $\tau _ { 1 : t - 1 }$ denote the realized execution prefix before step $t ,$ let $A _ { t }$ denote the action taken at that step, and let $Y$ denote the terminal task outcome. The step-level effect of replacing action a with an alternative action $a ^ { \prime }$ is defined as

$$
\theta _ { t } ( a , a ^ { \prime } ) = \mathbb { E } \big [ Y _ { A _ { t } = a ^ { \prime } } - Y _ { A _ { t } = a } \mid \tau _ { 1 : t - 1 } \big ] .\tag{1}
$$

This estimand asks whether changing a single action, while holding the realized history fixed, would change the final outcome.

This quantity is different from the targets implicitly measured by several common process-evaluation practices. Full-trace judges condition on downstream actions and the final outcome; restart-based ablations discard the realized history and resample the whole continuation; process reward labels are often derived from completed trajectories. These are not only procedural differences. They correspond to different conditioning sets, and therefore to different measurement targets. Our framework is designed to make this distinction explicit and to test what can be identified under the prefixconditioned target in Eq. equation 1; Appendix A tabulates each practice, the bias it incurs, and the empirical signature it should leave.

Figure 2 illustrates the core causal distinction. The valid conditioning set for step-local attribution is the realized prefix only. Conditioning additionally on later actions or on the terminal outcome changes the estimand and, in the case of full-trace judging, introduces post-treatment bias. This is why process evaluation can appear informative while still failing to provide valid step-level causal attribution. A related conflation, in which data-flow reachability is offered as a proxy for causal irrelevance, comes apart in both directions; Appendix A.7 gives counterexamples.

## 3.3 AN AGENT–ENVIRONMENT STRUCTURAL CAUSAL MODEL

Let $\mathcal { M } = \langle U , V , F , P ( U ) \rangle$ be a structural causal model with exogenous variables

$$
U = \big ( Z , \{ G _ { \pi } ^ { ( t ) } \} , \{ U _ { \mathrm { e n v } } ^ { ( t ) } \} \big ) ,
$$

![](images/1608425427d68966ef2ed1080a1c8133e4dd0d0fea7947b5de4d0a23fdac62df.jpg)  
Figure 2: Three conditioning sets and their causal status. The conditioning set, not the scorer, determines whether attribution is causal. Latent task difficulty Z (dashed) influences both behaviour and outcome, but conditioning on the prefix $\tau _ { 1 : t - 1 }$ blocks it, because the policy observes nothing that is not logged and decoding noise is exogenous. Conditioning additionally on $A _ { t + 1 }$ and $Y ,$ which is what full-trace judging does, conditions on descendants of $A _ { t }$ and opens a biasing path. The predicted signature is attribution displaced toward later steps. We test it in §4.6 by a controlled manipulation of the judge’s information set, which is the one attribution result here that requires no causal reference.

where Z denotes latent task difficulty, $G _ { \pi } ^ { ( t ) }$ denotes decoding noise, and $U _ { \mathrm { e n v } } ^ { ( t ) }$ denotes environment noise. The endogenous system is defined by

$$
A _ { t } : = \arg \operatorname* { m a x } _ { k } \big ( \log \pi _ { k } ( S _ { t } ) + G _ { \pi , k } ^ { ( t ) } \big ) , \qquad G _ { \pi , k } ^ { ( t ) } \sim \mathrm { G u m b e l } ( 0 , 1 ) ,\tag{2}
$$

$$
O _ { t } : = \mathcal { E } \big ( A _ { t } , G ; U _ { \mathrm { e n v } } ^ { ( t ) } \big ) ,\tag{3}
$$

$$
S _ { t + 1 } : = \mathcal { T } ( S _ { t } , A _ { t } , O _ { t } ) , \qquad S _ { 1 } : = \sigma ( q , \mathrm { s y s } ) ,\tag{4}
$$

$$
\hat { F } : = \phi ( S _ { T } ) , \qquad Y : = \mathcal { k } \big [ F ^ { \star } \subseteq \hat { F } \big ] .\tag{5}
$$

Two modeling choices are central. First, the policy is written in Gumbel–max form, which makes categorical decisions representable as perturb-and-argmax draws and gives a tractable account of local policy stochasticity. Second, the task outcome is defined by containment rather than exact set equality, because localization should not be penalized simply for naming reasonable auxiliary files such as tests. This outcome remains verifiable while aligning better with the semantics of repository search.

## 3.4 IDENTIFICATION

We now state the conditions under which the prefix-conditioned effect in Eq. equation 1 is identified. Assumption 1 (Context observability). $S _ { t }$ is a deterministicfunction of $\left( q , \tau _ { 1 : t - 1 } \right)$

Assumption 2 (Exogenous decoding noise). $G _ { \pi } ^ { ( t ) } \perp ( Z , U ^ { ( < t ) } , U _ { \mathrm { e n v } } )$

Assumption 3 (Replayable environment). Given $( A _ { t } , G , U _ { \mathrm { e n v } } ^ { ( t ) } )$ , the observation $O _ { t }$ is determined. Proposition 1 (Sequential randomization by construction). Under Assumptions $I { - } 3 ,$ , we have $A _ { t } \perp$ $( Z , \bar { U } ^ { ( \geq t ) } ) \mid \tau _ { 1 : t - 1 }$ . Consequently,for any action a with $\pi ( \boldsymbol { a } \mid \tau _ { 1 : t - 1 } ) > 0$

$$
\mathbb { E } \left[ Y \mid \tau _ { 1 : t - 1 } , d o ( A _ { t } { = } a ) \right] = \mathbb { E } \left[ Y \mid \tau _ { 1 : t - 1 } , A _ { t } { = } a \right] .\tag{6}
$$

Thus, step-level causal effects are nonparametrically identified from observational traces given the realized prefix.

The intuition is straightforward. Conditioning on the realized prefix fixes the state $S _ { t }$ , so the remaining variation in $A _ { t }$ comes only from exogenous policy noise. Since this noise is independent of latent task difficulty and of future continuation noise, the step-t action behaves as a locally randomized treatment assignment. This yields identification of the prefix-conditioned causal effect without the usual no-unmeasured-confounding assumption. Appendix A.6 gives the full argument and shows exactly which step of it fails under each departure from this conditioning set.

The assumptions are also empirically meaningful. Context observability fails under compaction, because dropped context becomes unobserved state. Replayability fails if the environment cannot be reproduced from the logged state and action. Decoding exogeneity is a structural property of the serving policy rather than something directly testable from traces. We therefore verify the first two assumptions empirically in the experimental setup and treat the third as a modeling assumption. Appendix A.5 states the three failure modes formally, none of which is confounding.

## 3.5 AN ANYTIME VALUE AND STEP-LEVEL CONTRIBUTION

Rather than estimating each step effect independently, we define an anytime value function along the trajectory:

$$
v _ { t } : = \mathbb { E } [ Y \mid \tau _ { 1 : t } , q , G ] ,\tag{7}
$$

and its increment

$$
\Delta _ { t } : = v _ { t } - v _ { t - 1 } .\tag{8}
$$

This construction supports a natural attribution view in which each step contributes by changing the expected terminal outcome relative to the immediately preceding prefix.

Theorem 1 (Exactly additive step contributions). Under Assumptions 1–3,

$$
\sum _ { t = 1 } ^ { T } \Delta _ { t } = v _ { T } - v _ { 0 } = Y - \mathbb { E } [ Y \mid q , G ] .\tag{9}
$$

Moreover, each increment can be decomposed into an action effect relative to the policy’s own baseline and an observation effect induced by the returned tool output.

This theorem shows that the total surprise of an episode, relative to its pre-execution expectation, can be distributed over steps without residual. In practice, the quantity of interest is not only whether the final outcome changed, but when and where the expected outcome changed along the trajectory. The increments $\Delta _ { t }$ therefore serve as our step-level attribution object. Appendix A.6 proves the theorem and states the action/observation split explicitly, since the two halves imply different remedies: better planning versus better tools or indexing.

## 3.6 REPLAY-BASED ESTIMATION UNDER LOCAL POSITIVITY

The quantities above are population-level objects. To estimate them on real coding-agent traces, we reconstruct a factual prefix, resume execution from that prefix multiple times, and average the resulting outcomes. Concretely, for an estimation point $t ,$ we replay the continuation from $\tau _ { 1 : t }$ for m repetitions and estimate

$$
\widehat { v } _ { t } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } Y _ { t } ^ { ( i ) } ,\tag{10}
$$

where $Y _ { t } ^ { ( i ) }$ is the terminal outcome of the i-th replay from prefix $\tau _ { 1 : t }$ . The corresponding empirical contribution is then

$$
\widehat { \Delta } _ { t } = \widehat { v } _ { t } - \widehat { v } _ { t - 1 } .\tag{11}
$$

Because all arms share the reconstructed prefix and the pinned environment noise, this is a commonrandom-numbers design, which is what makes a small replay budget usable at all.

This replay estimator is valid only when the local action distribution retains sufficient overlap. In coding-agent execution, however, positivity is not globally available: some prefixes generate diverse next actions under replay, while others repeatedly produce the same action and therefore provide no observational support for nearby alternatives. We therefore treat positivity as a local property that must be measured rather than assumed.

To detect whether local overlap is present, we probe the replayed action distribution at each estimation point. Let C denote action buckets defined at the level of tool and target object. Using m replayed actions at step t, we define the empirical action distribution

$$
\widehat { \pi } _ { t } ( c ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathcal { k } \bigl [ \mathrm { b u c k e t } ( A _ { t } ^ { ( i ) } ) = c \bigr ] , \qquad c \in \mathcal { C } ,\tag{12}
$$

and its empirical entropy

$$
\widehat { H } ( \widehat { \pi } _ { t } ) = - \sum _ { c \in \mathcal { C } } \widehat { \pi } _ { t } ( c ) \log \widehat { \pi } _ { t } ( c ) .\tag{13}
$$

Intuitively, $\widehat { H } ( \widehat { \pi } _ { t } )$ measures whether the next action remains stochastic under replay from the same realized prefix. Appendix B.4 reports the measured entropy at every probed decision point, which is what shows that overlap varies within an episode rather than globally.

This leads to a hybrid estimation strategy. When the replayed action entropy is positive, we use observational replay and estimate the contribution by the difference of replay-based value estimates. When the entropy is effectively zero, we switch to an intervention branch that substitutes an alternative action and executes the resulting continuation in the real environment. Formally,

$$
\begin{array} { r } { \widehat { \Delta } _ { t } = \left\{ \begin{array} { l l } { \widehat { v } _ { t } - \widehat { v } _ { t - 1 } , } & { \mathrm { i f ~ } \widehat { H } ( \widehat { \pi } _ { t } ) > \tau _ { H } \quad \mathrm { ( o b s e r v a t i o n a l ~ b r a n c h ) } , } \\ { \widehat { \mathbb { E } } [ Y \mid \tau _ { 1 : t - 1 } , d o ( A _ { t } = a ^ { \prime } ) ] - \widehat { \mathbb { E } } [ Y \mid \tau _ { 1 : t - 1 } , d o ( A _ { t } = a _ { t } ) ] , } & { \mathrm { i f ~ } \widehat { H } ( \widehat { \pi } _ { t } ) \leq \tau _ { H } \quad \mathrm { ( i n t e r v e n t i o n ~ b r a n c h ) } . } \end{array} \right. } \end{array}\tag{14}
$$

The intervention branch is needed because a deterministic policy step provides no observational support for nearby counterfactual alternatives. In that case, we substitute an alternative action $a ^ { \prime }$ into the reconstructed prefix, execute it for real, splice its genuine tool output into the continuation, and then replay the rest of the trajectory. This preserves the semantics of an action intervention. We never fabricate tool outputs, because replacing an observation rather than an action would target a different causal quantity.

Assumption 4 (Perturbation transferability). The substituted action a<sup>′</sup> lies in the support of a suitably perturbed version of the deployed policy.

Assumption 4 makes explicit that the low-entropy branch is not observational identification under the original policy. Instead, it estimates the effect of moving from the realized action to a nearby action that remains behaviorally plausible under a perturbed policy. We label this distinction explicitly in the experiments, and Appendix A.4 states where the resulting estimates sit on the causal ladder.

## 3.7 OUTCOME CHOICE AND PRACTICAL ESTIMATION

The framework above applies to any bounded terminal outcome. In principle, one could use a binary success variable, exact file-set equality, or recall over the gold file set. In practice, this choice matters because it determines how much signal remains available to replay-based estimation. Binary success is often too coarse in file localization: once a trajectory already fails, many step substitutions do not change the indicator at all. Exact set equality is even more brittle because it penalizes otherwise reasonable auxiliary predictions such as test files. We therefore use recall-based outcomes in the main experiments, which preserve more variation and make step- and task-level changes empirically detectable. Appendix B.6 reports the screening of all six candidate outcomes and the degenerate rate that motivated the choice.

Replay-based estimation also requires selecting a finite set of estimation points and a finite replay budget per point. We therefore estimate values at a set of sampled prefixes rather than at every step in the trajectory. The resulting contributions should be interpreted as segment-level approximations rather than per-token or per-tool-call infinitesimals. This design trades off granularity against replay cost and is one reason we later distinguish what is formally identifiable from what is empirically measurable at feasible scale.

## 3.8 UNCERTAINTY DECOMPOSITION

The same replay-based value function also supports uncertainty analysis. Let

$$
v _ { 0 } = \mathbb { E } [ Y \mid q , G ]\tag{15}
$$

denote the pre-execution success probability conditioned only on the task description and repository structure. Its predictive uncertainty can be decomposed conceptually into three components:

$$
\mathbb { H } [ Y \mid q , G ] = U _ { \mathrm { t a s k } } + U _ { \mathrm { p o l i c y } } + U _ { \mathrm { m o d e l } } .\tag{16}
$$

Here $U _ { \mathrm { t a s k } }$ denotes irreducible uncertainty arising from task ambiguity, $U _ { \mathrm { p o l i c y } }$ denotes variability induced by stochastic agent execution, and $U _ { \mathrm { m o d e l } }$ denotes epistemic uncertainty in the forecasting model itself.

This decomposition is useful because the three terms correspond to different operational responses. High task uncertainty suggests ambiguity in the problem itself; high policy uncertainty suggests repeated execution and aggregation; high model uncertainty suggests abstention or escalation. In this paper, however, we do not claim that all three components are directly and fully identified on real traces. Our empirical use of this decomposition is more modest. We estimate policy-driven execution variance directly from replay, use a simple forecasting model to probe the left endpoint $v _ { 0 } ,$ and test whether the uncertainty relevant to execution outcome is structured at the step level or at the task level. The decomposition therefore serves as a principled organizational framework for uncertainty analysis rather than as a fully solved inference problem in the main benchmark setting. Appendix C.6 gives an estimator for each component and the synthetic self-test on which the threeway separation is validated; Appendix C.7 gives the $v _ { 0 }$ model and Appendix C.8 the set-valued variant.

## 3.9 EVALUATION QUANTITIES

The experiments evaluate process-level measurements rather than only agent outcomes, so we define the main reported quantities here.

For attribution comparisons, let $\hat { t } _ { i } ^ { M }$ denote the step selected by method M in episode $i ,$ and let $\mathrm { p o s } ( t ) = t / T _ { i }$ denote normalized position within the trajectory. We compare two attribution methods using top-1 disagreement and positional displacement:

$$
D _ { 1 } ( M , M ^ { \prime } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { k } \bigl [ \hat { t } _ { i } ^ { M } \neq \hat { t } _ { i } ^ { M ^ { \prime } } \bigl ] , \qquad D _ { 3 } ( M , M ^ { \prime } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bigl ( \mathrm { p o s } ( \hat { t } _ { i } ^ { M } ) - \mathrm { p o s } ( \hat { t } _ { i } ^ { M ^ { \prime } } ) \bigr ) .\tag{17}
$$

In this paper, the displacement statistic is more informative than raw disagreement, because disagreement against an uncertified causal reference is not itself evidence of bias, whereas systematic displacement remains meaningful even when the reference is noisy. Appendix C.10 works out why $D _ { 1 }$ is near its ceiling under the null and should therefore be retired.

When a method selects a blamed step, we also ask whether that step aligns with the most harmful estimated segment, whether it lands on a zero-contribution segment, and what signed contribution it receives. Let $\sigma _ { i } ( t )$ denote the estimation segment containing step t, and let $\breve { \mathcal { N } }$ be the subset of non-degenerate episodes whose estimated contributions are not identically zero. We define

$$
R ( M ) = \frac { 1 } { | \mathcal { N } | } \sum _ { i \in \mathcal { N } } \mathcal { k } \Big [ \sigma _ { i } ( \widehat { t } _ { i } ^ { M } ) = \arg \operatorname* { m i n } _ { t } \widehat { \Delta } _ { i , t } \Big ] ,\tag{18}
$$

$$
Z ( \boldsymbol { M } ) = \frac { 1 } { | \mathcal { N } | } \sum _ { i \in \mathcal { N } } \boldsymbol { k } ^ { \intercal } \Big [ \widehat { \Delta } _ { i , \sigma _ { i } ( \widehat { t } _ { i } ^ { M } ) } = 0 \Big ] ,\tag{19}
$$

and

$$
\bar { \Delta } ( M ) = \frac { 1 } { | \mathcal { N } | } \sum _ { i \in \mathcal { N } } \widehat { \Delta } _ { i , \sigma _ { i } ( \widehat { t } _ { i } ^ { M } ) } .\tag{20}
$$

These quantities are descriptive tools for comparing where a process evaluator places blame relative to the intervention-based estimate. They should not be interpreted as proof of attribution correctness unless the underlying step-level reference is itself validated.

For pre-execution forecasting, we evaluate both discrimination and calibration. Let ${ \widehat { v } } _ { 0 , \cdot }$ <sub>i</sub> denote the predicted success probability for instance i. We compute the Expected Calibration Error and Brier score as

$$
\mathrm { E C E } = \sum _ { b = 1 } ^ { B } \frac { \left| \mathcal { B } _ { b } \right| } { n } \left| \overline { { y } } _ { b } - \overline { { p } } _ { b } \right| , \qquad \mathrm { B r i e r } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( \widehat { v } _ { 0 , i } - Y _ { i } \right) ^ { 2 } .\tag{21}
$$

Because the benchmark has a high success base rate, we also report ranking-based or selective-risk quantities in the experiments to distinguish genuinely informative forecasts from nearly constant but calibrated predictors.

At the action level, next-action prediction is evaluated with top-k accuracy, negative log-likelihood, calibration error, and effective candidate count. At the task level, uncertainty is evaluated through replay variance, residualized variance-structure probes, and its relationship to cheap predictive entropy from the action model. Together, these measurements allow us to answer three separate questions in the experiments: what predicts the next action, where uncertainty resides, and what process attribution methods actually measure. The full inventory of comparison baselines and nulls behind each measurement is given in Appendix B.3.

## 4 EXPERIMENTS

This section evaluates the proposed framework from the three-level perspective of the paper. Our goal is not only to report empirical results, but to determine what can be measured reliably at the action, task, and step levels of coding-agent execution. We therefore organize the experiments around five research questions, each tied to specific figures, tables, and comparison baselines.

We study the following questions. RQ1 asks which structural signal best predicts the next action of a coding agent. RQ2 asks what role the code dependency graph actually plays in execution. RQ3 examines whether commonly reported reuse patterns reflect genuine temporal behavior or permutation-invariant artifacts. RQ4 asks at which level execution uncertainty resides and whether step-level attribution is measurable at feasible replay cost. RQ5 asks what observational process attribution methods, especially full-trace judges, actually measure when no certified step-level causal reference is available.

## 4.1 EXPERIMENTAL SETUP

We study 499 file-localization instances derived from SWE-bench-Verified, covering 12 repositories. Each episode contains a natural-language issue, a repository snapshot, an agent trajectory, and a gold file set F<sup>⋆</sup>. The primary task outcome is recall over the gold file set, which we use because it preserves substantially more intervention-sensitive variation than binary success. Under this outcome, 98.6% of instances can move under intervention, compared with only 73.2% under the binary outcome. Appendix C.1 gives the per-repository composition, which is dominated by django and is why leave-one-repository-out is used wherever a predictor is fitted, and Appendix B.2 gives the agent, model, and environment-pinning details.

The causal analysis relies on replayability and prefix observability, so we verify these preconditions empirically. Environment replayability holds at 1.00 over 100 actions with 3 re-executions per action, and context compaction occurs in 0 of all rollouts. For attribution and uncertainty estimation, we replay 8 continuations from each of 190 prefixes across 38 instances, yielding 1901 replayed continuations over 12.7 hours. Confidence intervals are bootstrap intervals at the episode or instance level; null comparisons use within-episode permutation tests or closed-form expectations where available; and multiple comparisons are controlled by BH-FDR at q = 0.10. Appendix B.3 lists every comparison predictor and null used below, and Appendix B.5 states the statistical conventions in full.

## 4.2 RQ1: WHAT PREDICTS THE NEXT ACTION?

RQ1 asks which structural signal best predicts what the agent does next. The main competitors are execution provenance and repository structure. We evaluate this question with Figure 3, Table 1, and Table 2.

Figure 3(a) gives the central result. On the 67.0% of steps that move to a previously unvisited object, a provenance-only predictor using the most recent tool output attains top-3 = 0.326 with confidence interval [0.276, 0.375], compared with 0.058 for a strengthened dependency-graph predictor and 0.072 for a repository-frequency baseline. The paired improvement over the graph predictor is +0.136 at top-3 with confidence interval [+0.102, +0.173], and +0.259 at top-10 with confidence interval [+0.210, +0.313]. This comparison is especially informative because first-visit steps remove visit history by construction, so the gain cannot be explained by trivial revisitation effects. Instead, it shows that the agent’s next move is guided primarily by what it has just seen in tool outputs.

<table><tr><td>Object model</td><td>coV.</td><td>top-1</td><td>top-3</td><td>top-10</td><td>NLL</td><td>ECE exp(H)</td><td></td></tr><tr><td>Routed by predicted tool Factored, shared object</td><td>0.619</td><td>0.150</td><td>0.260</td><td>0.484</td><td>2.76</td><td>0.0123</td><td>12.8</td></tr><tr><td>model</td><td>0.645</td><td>0.109</td><td>0.241</td><td>0.432</td><td></td><td>3.41 0.0139</td><td>39.9</td></tr><tr><td>Paired difference</td><td></td><td>+0.041</td><td>+0.018</td><td>+0.052</td><td></td><td></td><td></td></tr><tr><td>95% CI</td><td></td><td> $- \ [ + . 0 3 4 , + . 0 4 7 ]$ </td><td> $[ + . 0 0 9 , + . 0 2 6 ]$ </td><td> $[ + . 0 4 7 , + . 0 5 8 ]$ </td><td></td><td></td><td></td></tr></table>

Table 1: Predicting the complete next action (tool, obj) over 7616 decision points from 488 episodes. Both models share the same tool component and the same denominator: every decision point counts, and points whose realized action falls outside the candidate set are scored as misses rather than dropped. top-k are closed-form expectations over random tie-breaking. NLL is over covered points only, so it measures sharpness rather than coverage. $\exp ( H )$ is the median effective candidate count. Routing the construction of the candidate set on the predicted next tool—reads take paths, searches take free-text queries, handler tools take no object—is what the dependence $I = 0 . 1 5 9$ nats $( p = 5 \times 1 0 ^ { - 4 } )$ between tool and object type requires; it improves every top-k and tightens the candidate set 3.1×.

<table><tr><td>Stratum (observable before predicting)</td><td>n</td><td>top-3</td><td>ECE</td></tr><tr><td>Current step is a search</td><td>228</td><td>0.469</td><td>0.0125</td></tr><tr><td>Current step is a read</td><td>206</td><td>0.160</td><td>0.0247</td></tr><tr><td>Current step lists a directory</td><td>35</td><td>0.057</td><td>0.0278</td></tr><tr><td>Predicted  $\exp ( H ) \leq 5$ </td><td>154</td><td>0.351</td><td>0.0762</td></tr><tr><td>Predicted exp  $( H ) \in ( 5 , 1 5 ]$ </td><td>198</td><td>0.409</td><td>0.0168</td></tr><tr><td>Predicted exp  $( H ) > 1 5$ </td><td>157</td><td>0.159</td><td>0.0157</td></tr><tr><td>Last output lists 1–3 paths</td><td>263</td><td>0.357</td><td>0.0411</td></tr><tr><td>Last output lists  $4 { - } 1 0$  paths</td><td>151</td><td>0.298</td><td>0.0050</td></tr><tr><td>Last output lists  $\geq 1 1$  paths</td><td>95</td><td>0.221</td><td>0.0125</td></tr></table>

Table 2: When the next-action distribution can be trusted. Only strata observable before the prediction are listed, because a stratum defined by the realized answer is a conditional accuracy and not a rule: conditioning on coverage gives top- $3 = 0 . 6 1 1$ at $\mathrm { E C E } = 0 . 0 1 0 3 .$ , which is not actionable and which we therefore exclude. Two readings are practical. Prediction is trustworthy immediately after a search and untrustworthy after a read, an eight-fold spread. And low entropy is not sufficient for trust: the $\exp ( H ) \leq 5$ stratum is the worst calibrated, because a small candidate set signals coverage risk rather than confidence.

The same figure also reveals why this effect holds. Among first-visit targets, 0.689 of them had already been mentioned in earlier tool outputs, with confidence interval [0.642, 0.739]. In other words, execution provenance already covers most of the objects that later become next-step targets. This is a much stronger signal than graph proximity, which remains weak even after strengthening the graph baseline. The candidate efficiency numbers make the same point: provenance achieves its top-3 performance with only 3.46 effective candidates, while broader graph-based search is les accurate despite a more diffuse candidate set.

Figure 3(b) shows that next-action predictability is strongly heterogeneous by object type. Handler steps, which account for 27.3% of all steps, reach $\displaystyle { \mathrm { t o p } } - 3 = 0 . 5 8 2 ;$ path steps, which account for 51.1%, reach 0.185; and steps whose object is a novel free-text query, which account for 21.6%, reach only 0.029. This heterogeneity matters because a single aggregate accuracy would hide the fact that query generation, not path selection, is the hard part of next-action prediction. That query number survives a deliberate attempt to improve it, and a parser defect we found and measured rather than assumed moves it down rather than up; Appendix C.4 reports the audit, because the direction of the defect determines how the claim should be read.

<table><tr><td>Next-object predictor</td><td>cov.</td><td>top-3</td><td>95% CI</td><td>top-10</td><td>exp(H)</td></tr><tr><td colspan="6">Repository structure</td></tr><tr><td>Repository frequency (marginal)</td><td>0.349</td><td>0.072</td><td>[0.047, 0.099]</td><td>0.163</td><td>97.62</td></tr><tr><td>Code dependency graph, strengthened</td><td>0.677</td><td>0.058</td><td>[0.039, 0.078]</td><td>0.172</td><td>1719.13</td></tr><tr><td>Visit history</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Visit-history frequency</td><td>0.000</td><td>0.000</td><td>[0.000, 0.000]</td><td>0.000</td><td>4.84</td></tr><tr><td>Execution provenance Uniform over mentioned paths</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.689 0.689</td><td>0.136</td><td>[0.116, 0.155]</td><td>0.354</td><td>25.00</td></tr><tr><td>Frequency-weighted history</td><td>0.689</td><td>0.194 0.308</td><td>[0.162, 0.227]</td><td>0.431</td><td>22.17</td></tr><tr><td>Recency-weighted history</td><td>0.501</td><td>0.297</td><td>[0.267, 0.348]</td><td>0.499 0.431</td><td>18.75</td></tr><tr><td>Last output only Last output, position-ranked</td><td>0.501</td><td>0.326</td><td>[0.256, 0.338] [0.276, 0.375]</td><td>0.455</td><td>4.00 3.46</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Tool-routed candidate sets</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Routed on provenance</td><td>0.689</td><td>0.156</td><td>[0.126, 0.188]</td><td>0.409</td><td>21.64</td></tr><tr><td>Routed on the code graph</td><td>0.677</td><td>0.009</td><td>[0.005, 0.014]</td><td>0.109</td><td>283.28</td></tr></table>

Table 3: The RQ1 predictor suite on the 341 first-visit decision points, where the realized object has never been visited before. This is the discriminating regime: visit history covers none of these steps by construction, so a predictor cannot score by restating the agent’s own past, which is why the visit-history row sits at zero coverage. All rows share one denominator—points whose realized object falls outside the candidate set are scored as misses rather than dropped—and top-k are closedform expectations over random tie-breaking. The code-graph row is strengthened relative to the obvious implementation: it seeds a multi-source proximity ranking from the entire visited set rather than from the last step alone, and it still loses to every provenance variant listed, including the uninformative uniform one. The paired contrast quoted in §4.2 uses the frequency-weighted row rather than the best row, so it understates the gap. Higher is better for coverage, top-3 and top-10; lower is better for the effective candidate count $\bar { \exp ( H ) }$ . Best value per column in bold.

Table 1 strengthens this interpretation by showing that the next action should be modeled jointly rather than with an independent factorization over tool and object. Tool identity and object type are statistically dependent, with mutual information I = 0.159 nats and permutation $p = \bar { 5 } \times \mathrm { 1 0 ^ { - 4 } }$ . A routed predictor that constructs the candidate set conditional on the predicted tool improves top-1 from 0.109 to 0.150, a gain of +0.041 with confidence interval [+0.034, +0.047], and improves top-10 by +0.052 with confidence interval [+0.047, +0.058]. It also shrinks the effective candidate set from 39.93 to 12.85 while preserving calibration. This confirms that action prediction is structured not only by provenance but also by a nontrivial interaction between tool choice and object type.

Table 2 turns these findings into an actionable reliability analysis. Immediately after a search step, the next action is both accurate and calibrated, with top-3 = 0.469 and ECE = 0.013. After a read step, the same prediction problem becomes much harder, with top-3 only 0.160. Moreover, low predicted entropy is not by itself a reliable trust signal: the low-effective-candidate regime is in fact the worst calibrated, because a small candidate set often reflects coverage failure rather than genuine confidence.

Taken together, these results answer RQ1 clearly: the dominant predictor of a coding agent’s next action is its own execution provenance, not the code dependency graph. Repository structure matters much less for local next-step prediction than what the agent has just observed.

## 4.3 RQ2: WHAT IS THE CODE DEPENDENCY GRAPH ACTUALLY USEFUL FOR?

RQ2 asks whether the code dependency graph is useless, or merely useful at a different level than local action prediction. We answer this with Figure 4 and the associated graph-ranking statistics.

Figure 4(a) evaluates whether the graph explains step-to-step movement. The null here is important: we compare the observed trajectory order against a within-episode permutation null that preserves the visited set but destroys only temporal order. Against this null, consecutive steps are not meaningfully closer in the graph than chance. The one-hop lift is 0.94, the three-hop lift is 1.02, and the same-community lift is 1.01. The only quantity that clearly exceeds chance is immediate selfrepetition, with lift 1.47, but this is a temporal reuse effect rather than evidence that the graph determines local transitions. This result directly complements RQ1: if provenance predicts next-step behavior well and graph proximity does not outperform a permutation null, then the graph should not be interpreted as the local transition kernel of agent navigation.

![](images/0674cb0b4bde1db67170ff06fdd7622fef6cc7de6ac801a48021f1862027bbb4.jpg)

![](images/8b40096d357fe94fc3dad53ad80d6e43227d06a1e8851f6ea66c44a59bfa8fb7.jpg)  
Figure 3: Execution provenance versus code structure on first-visit steps. The structure that predicts an agent’s next object is its own execution provenance, not the code dependency graph. (a) On the 67.0% of steps whose object has never been visited—where visit history is empty by construction—a predictor reading only the most recent tool output reaches top-3 = 0.326, against 0.058 for a code graph strengthened to seed a multi-source proximity ranking from the entire visited set, and 0.072 for repository frequency. Bars are closed-form expectations over random tie-breaking; whiskers are episode-level bootstrap 95% CIs. (b) Predictability is heterogeneous, and bar width is the share of all 7616 steps: handler steps are nearly determined, path steps are moderately predictable, and steps taking a novel free-text search query are not. A single aggregate number would hide this.

Figure 4(b) shows a different role for the same graph. The visited set has internal edge density 0.269, compared with a null of only 0.0087, corresponding to a lift of 30.9×. This means that the files touched by the agent form a graph-coherent region even though the order of movement inside that region is not graph-driven. The graph also ranks unseen gold files highly once leakage is controlled: the leakage-controlled rank quantile for unvisited gold files is 0.049 with confidence interval [0.021, 0.084], and remains 0.103 when seeded from an early prefix. These quantities show that the graph is informative about which area of the repository matters, even if it is not informative about which file comes next.

The two findings are therefore complementary rather than contradictory. The graph describes the semantic region in which the agent is working, but not the temporal ordering of its steps within that region. RQ2 thus answers that the code dependency graph is useful as a scoping prior, not as a next-step transition model. This conclusion is confined to the 46.2% of agent objects that resolve to a graph node, a coverage limit we return to in §5.

## 4.4 RQ3: WHICH REUSE PATTERNS ARE REAL?

RQ3 asks which reuse phenomena reflect real temporal behavior and which are artifacts of the visited multiset. This distinction matters because reuse statistics are often interpreted as evidence of agent tendency without checking whether they are permutation-sensitive.

We begin with reuse rates. For both tools and objects, the raw probability of reusing a previously seen element is largely determined by the multiset of visited elements and therefore does not, by itself, reveal a temporal preference. This means that the observed tool reuse rate of 79.1% and object reuse rate of 41.6% are not informative as behavioral evidence in isolation. They are arithmetic consequences of vocabulary size and trajectory length.

(a) the graph does not describe transitions  
![](images/2063eeb256554545823f88d86aba7244d30f9c0d5e0d49f70eda52d889d2713f.jpg)

(b) but it does describe regions  
![](images/5286c2bef5735d58cfc314724140dc6292fdcc6bd788348ea953b6d19702070f.jpg)  
Figure 4: What the dependency graph does and does not describe. The graph describes where an agent works, not how it moves. (a) Against a within-episode permutation null—which preserves the multiset of visited files and destroys only their order—consecutive steps are no closer in the graph than chance (one-hop lift 0.94 with self-repeats removed, three-hop 1.02, same-community 1.01). The only quantity above chance is distance zero, i.e. immediate self-repetition, which is a temporal effect (§4.4). (b) The same graph separates the visited set from random same-size sets by 30.9× in internal edge density. The two panels are consistent: inside a region of density 0.269 an arbitrary pair is already likely to be one hop apart, and the permutation null removes exactly that region effect.

What does carry behavioral content is temporal organization. Objects are repeated in immediately adjacent steps 1.83× more often than expected under within-episode permutation, with permutation $p \sim 1 0 ^ { - 1 4 1 }$ . Successive visits to the same object are also temporally clustered, with inter-visit gaps compressed to a ratio of 0.90 relative to chance. Among revisit events, the target’s previous frequency exceeds the closed-form uniform expectation by 1.70× with confidence interval [1.63, 1.76], and its recency rank exceeds expectation by 1.74× with confidence interval [1.67, 1.81]. These are genuine temporal effects and indicate that agents preferentially stay near recently or frequently used objects.

The same conclusion does not hold for tools. Tool self-repetition has lift only 1.02 with $p = 0 . 5 1$ which is not significant. Instead, the tool channel exhibits structured transitions, with mutualinformation lift 1.13× chance and permutation $p \sim 1 0 ^ { - 2 2 }$ This matches the action-prediction results in RQ1: tool usage is predictable, but not because agents blindly repeat the same tool.

RQ3 therefore distinguishes two notions that are often conflated. Reuse rates are often uninformative, while reuse patterns—especially immediate object repetition, temporal clustering, and recency/frequency bias—capture genuine execution behavior.

## 4.5 RQ4: AT WHICH LEVEL DOES EXECUTION UNCERTAINTY LIVE?

RQ4 addresses the main empirical question behind the paper: whether the uncertainty relevant to execution outcome is localized at the step level or structured more globally at the task level. Figure 5 provides the core evidence, and the attribution statistics show the consequence for step-level measurement.

Figure 5(a) compares several structural probes of replay variance. The key null result is that zerocost predictive entropy from the action model does not proxy replay-based execution uncertainty. The correlation is $\rho ~ = ~ - 0 . 0 8 8$ at the step level with confidence interval $[ - 0 . 2 4 4 , + 0 . 0 6 2 ]$ , and $\rho = - 0 . 2 1 9$ after aggregation to the instance level with confidence interval $[ - 0 . 5 1 8 , + 0 . 1 2 6 ]$ . Neither indicates a useful relationship. This rules out a convenient but misleading interpretation in which cheap action uncertainty could stand in for actual execution variance. A control quantity does respond at $\rho = - 0 . 4 2 5$ with confidence interval [−0.695, −0.102], so the null is not an artifact of the outcome measure.

![](images/d8ead5851708f0aaed481425c5f0821cc78eb67de29c79a1d4264409add465c7.jpg)

(b) the mechanical part, removed before (a)  
![](images/824a14bee5a91e6a872ae97b030d81eea67d67abc5ba1ce0dada89cf5f567a64.jpg)  
Figure 5: The level at which execution uncertainty resides. Execution uncertainty is a property of the task, not of the step. (a) Probes for structure in the replay variance $U ^ { \mathrm { e x e c } }$ , all expressed as a share of variance explained. Instance identity explains 0.640 against a permutation null of 0.196 $( p = 5 \times 1 0 ^ { - 4 } )$ ; step position and tool type explain nothing. (b) The mechanical mean–variance relation that had to be removed before (a) is credible: $U ^ { \mathrm { e x e c } }$ collapses to zero in the top quintiles of $v _ { t } .$ , because variance is constrained by the mean. Panel (a) is computed on residuals after binning on $v _ { t }$ . Given the absence of step-level structure, the failure of step-level effects to survive FDR correction is expected rather than remediable by a denser scan.

At the same time, the replay variance is not structureless. Figure 5(a) shows that instance identity explains a substantial share of residual variance after removing the mechanical mean–variance relationship, with $\mathrm { I C C } = 0 . 6 4 0$ compared against a permutation null of 0.196 and $p = 5 \times 1 0 ^ { - 4 }$ By contrast, step position contributes little, with $\rho = - 0 . 1 3 1$ , and tool type shows no effect, with $p = 0 . 6 5$ . Figure 5(b) explains why residualization is necessary in the first place: variance is mechanically constrained by the mean, and replay variance collapses near the ends of the bounded outcome range. Once this mechanical relation is removed, the remaining signal is clearly task-level rather than step-level. One apparent exception to this null, produced by thresholding a continuous response at $p < 0 . 0 5 .$ , does not replicate and decays monotonically under a threshold sweep; $\mathsf { A p - }$ pendix C.5 reports it, because dichotomising a continuous quantity produces clean-looking strata at small n.

This task-level locus of uncertainty has a direct consequence for attribution. Of 190 interventionestimated step effects, only 5.3% reach uncorrected $p < 0 . 0 5$ , indistinguishable from the nominal chance rate, and 0 survive BH-FDR at $q = 0 . 1 0$ . The replay budget behind this result is already substantial—1901 replayed continuations over 12.7 hours—so the negative result is informative. It shows that formal identification does not automatically imply empirical measurability at the desired resolution. The limiting factor is not only engineering, but the location of uncertainty itself. We also report the coarseness of the instrument: at $m = 8$ the median replay variance is zero and 50.5% of segments have variance exactly zero, while 22.9% of episodes are degenerate in the sense that every estimated contribution is identically zero.

RQ4 therefore gives the core answer of the paper: execution uncertainty resides primarily at the task level rather than at the step level, and this is why step-level causal attribution remains underpowered at feasible replay budgets.

## 4.6 RQ5: WHAT DO OBSERVATIONAL ATTRIBUTION METHODS ACTUALLY MEASURE?

RQ5 asks what can still be said about observational attribution methods when no certified step-level causal reference is available. We answer this using Figure 6, Table 4, and Table 5.

(a) same judge, same rubric  
![](images/98cbe7626712e62ac818c54844212b86ecd2d61ee5ad79ec69949c824f748c47.jpg)

(b) it selects on semantic relevance  
![](images/39e4dcbc346c482a9eb43e6423c4dab15ab7208e100cb83e00a5fc3ce9ecca03.jpg)

Figure 6: Collider isolation in full-trace judging. Conditioning on descendants of the scored step displaces the verdict later, and the manipulation identifies this by construction rather than by inference. (a) Each line is one episode. Judge model and rubric are fixed; the only difference is whether steps downstream of the scored step are visible. The verdict moves later in 59 of 70 episodes (10 ties), by +0.537 in mean normalized position. Because the manipulation is of the judge’s information set rather than of the world, this requires no causal reference—which matters here, since none can be certified (§4.5). (b) What the full-trace judge selects on: steps that operate on or reveal a gold file, at 0.900 against the correct baseline of 0.413, and biased late within that class against a closed-form uniform null. That is semantic relevance—cheap and reproducible, but not demonstrably causal contribution.
<table><tr><td>Zero-LLM structural rule</td><td>agree</td><td>chance</td><td>diff</td><td>sig.</td></tr><tr><td>First step revealing a gold file</td><td>0.043</td><td>0.053</td><td>-0.010</td><td></td></tr><tr><td>First step operating on a gold file</td><td>0.100</td><td>0.053</td><td>+0.047</td><td></td></tr><tr><td>Last gold-provenance step</td><td>0.243</td><td>0.053</td><td>+0.190</td><td>√</td></tr><tr><td>Last step of the trace (position only)</td><td>0.171</td><td>0.053</td><td>+0.118</td><td>√</td></tr><tr><td>Step revealing the most gold files</td><td>0.043</td><td>0.053</td><td>-0.010</td><td></td></tr></table>

Table 4: Can a rule with no language model reproduce the full-trace judge’s choice? Agreement is exact same-step agreement over 70 episodes; chance is the closed-form expectation $1 / \hat { T } _ { i }$ for a rule selecting uniformly among steps. The judge’s class of choice is almost fully explained—it selects gold-provenance steps at 0.900 against a 0.413 baseline, and is biased late within that class (0.746 against a uniform null of 0.5). Its exact choice is not: with 8.7 gold-provenance steps per episode the within-class chance rate is 0.139, so the best rule is only 1.7× better than guessing inside the class. We report this residual rather than presenting the judge as a lookup table.

Figure 6(a) gives the central experiment. We hold the judge model and rubric fixed and vary only whether downstream steps are visible. If full-trace judges were implicitly approximating a prefixconditioned step-local quantity, this manipulation should not systematically change the location of blame. Instead, the verdict moves +0.537 later in normalized position, with confidence interval [+0.459, +0.610], and it moves later in 59 of 70 episodes, with sign test $p = 1 . 1 \times 1 0 ^ { - 1 6 }$ . Because the world being judged is unchanged, this displacement isolates a bias in the evaluation mechanism itself rather than disagreement with a noisy causal reference. In causal terms, full-trace judging conditions on descendants of the scored step, and the later shift is exactly the empirical signature one would expect from such post-treatment adjustment.

Figure 6(b) then asks what the full-trace judge is selecting when it sees the whole trajectory. The answer is not arbitrary. The judge places 0.900 of its top-1 verdicts on steps that either operate on or reveal a gold file, compared with a baseline frequency of only 0.413 among all steps. The enrichment over baseline is therefore +0.487 with confidence interval [+0.411, +0.557]. This means that the

<table><tr><td></td><td colspan="4">Collider displacement</td><td colspan="2">Gold-prov. enrichment</td></tr><tr><td>Judge</td><td>mean</td><td>95% CI</td><td>later</td><td>sign p</td><td>rate</td><td>diff CI</td></tr><tr><td>gpt-5.5 (original)</td><td>+0.533</td><td>[+.399, +.656]</td><td>19/23</td><td> $4 \times 1 0 ^ { - 6 }$ </td><td>0.826*</td><td>[+.241, +.500]</td></tr><tr><td>qwen3.6-plus (general)</td><td>+0.434</td><td> $[ + . 2 9 9 , + . 5 5 7 ]$ </td><td>18/23</td><td> $8 \times 1 0 ^ { - 6 }$ </td><td>0.565</td><td>[-.068, +.297]</td></tr><tr><td>qwen3-coder-plus (code)</td><td>+0.394</td><td> $[ + . 2 5 1 , + . 5 5 1 ]$ </td><td>15/23</td><td> $6 \times 1 0 ^ { - 5 }$ </td><td>0.826*</td><td> $[ + . 2 2 2 , + . 5 1 4 ]$ </td></tr></table>

Between-judge heterogeneity (permuting judge labels within episode): displacement $p = 0 . 1 8 1$ , enrichment $p = 0 . 0 4 4 .$

Table 5: The same manipulation across 3 judges on their 23 common episodes. Displacement—the effect of making steps downstream of the scored step visible—appears in every judge and shows no detectable between-judge heterogeneity; the spread of 0.14 across judges is small against per-judge intervals of roughly ±0.13, which is what supports the claim, since a p of 0.181 on 23 episodes means only that no difference was detected. Enrichment behaves oppositely: it reaches significance in 2 of 3 judges (<sup>∗</sup>) and its heterogeneity is significant. The baseline for the enrichment column is 0.443, the share of steps that touch or reveal a gold file. A critique of the mechanism therefore transfers across scorers; a critique of what they score does not.

judge does capture semantically relevant evidence. However, within that semantically relevant class, it is biased late: the within-class lateness is 0.746 against a uniform null of 0.5, with difference +0.246 and confidence interval [+0.172, +0.311]. Thus, full-trace judging is neither random nor causally faithful. It preferentially selects semantically relevant steps, but it does so using information that is unavailable to a valid prefix-conditioned attribution rule.

Table 4 sharpens this point by asking whether a simple non-LLM structural rule can reproduce the judge’s top-1 selection. The best such rule—choosing the last gold-provenance step—achieves agreement 0.243, well above the closed-form chance level of 0.053, but still far from perfect. This is a useful result for interpretation. It shows that the judge is not behaving like a trivial lookup rule, yet its behavior is largely captured by a preference for late, gold-provenance-related evidence. In other words, what the judge extracts is semantically meaningful, but not demonstrably causal.

We next compare this observational signal to the intervention-based estimate. On the estimation points available to SCAE, the intervention-based top-1 lands on gold-provenance steps at rate 0.286, against its own baseline of 0.362. Restricting to the subset of steps that are directly comparable between the judge and the intervention estimate gives 0.826 versus 0.261, with difference +0.565 and confidence interval [+0.304, +0.783]; only 32.9% ofjudge verdicts are step-comparable at all, which is why the two baselines must be computed separately. Because RQ4 showed that the interventionbased step reference does not survive FDR correction, we do not interpret this gap as proof that one selector is correct and the other is wrong. Instead, we interpret it more conservatively: the two procedures are measuring different quantities, and the judge’s quantity is better understood as downstream-informed semantic relevance.

Table 5 shows that the mechanism-level result generalizes across judge models. Repeating the same downstream-visibility manipulation with 3 judges over their 23 common episodes yields positive displacement in all cases: +0.534 for the original judge, +0.435 for the second general model, and +0.394 for the code-specialized model. Each judge’s confidence interval excludes zero, and each sign test is below $1 0 ^ { - 4 }$ . Moreover, the heterogeneity test for displacement is not significant $( p = 0 . 1 8 1 )$ , and the spread of only 0.14 across judges is small relative to the per-judge intervals. This supports the claim that collider displacement is a stable property of the judging mechanism rather than an idiosyncrasy of a single scorer.

At the same time, Table 5 also shows that the content of judging is more model-dependent. Only 2 of 3 judges exhibit significant enrichment on gold-provenance steps above the baseline of 0.443, and the heterogeneity test for this enrichment is significant $( p = 0 . 0 4 4 )$ . This contrast is important for the paper’s broader argument. The mechanism-level critique transfers across scorers: full-trace judging is systematically biased by downstream visibility. The content-level interpretation does not transfer as cleanly, because different judge models rely on somewhat different semantic cues.

Finally, we revisit a pre-registered criterion that turns out to fail as a criterion. The judge/causal top-1 disagreement rate is 97.1%, which might at first glance appear to support a strong attribution critique.

However, once the intervention-based step reference is known not to survive FDR correction, this disagreement rate is no longer diagnostic. Under the null, a noisy selector over a small set of estimation points and a judge choosing from roughly twenty trajectory steps will disagree almost all the time. We therefore retire this criterion rather than treating it as positive evidence; Appendix C.10 works out the null and Appendix D.2 records the withdrawal. The visibility-manipulation contrast above is the correct replacement, because it identifies a property of the judge without relying on an uncertified causal reference.

Taken together, RQ5 shows that observational attribution methods in this setting do not provide validated step-level causal explanations. What they measure is better described as semantically enriched, downstream-informed relevance, and the controlled visibility experiment identifies the source of this mismatch directly.

## 4.7 SUMMARY OF EXPERIMENTAL FINDINGS

Across all five research questions, the results support the same three-level account of process evaluation. At the action level, agent behavior is strongly structured and often predictable, but mainly by execution provenance rather than repository transitions. At the task level, execution uncertainty is real and structured, but primarily at the instance level rather than the step level. At the step level, formal identifiability does not translate into practical measurability at feasible replay cost, and observational attribution methods instead rely heavily on downstream semantic evidence.

These findings explain why process evaluation can appear simultaneously useful and difficult to validate. Some process-level signals are useful because they predict what the agent will do next. Others are useful because they characterize whether the task remains uncertain. But those signals should not automatically be interpreted as step-level causal attribution. The experiments therefore support the central claim of the paper: action, task, and step are different levels, and process evaluators should be interpreted according to the level they actually measure. Appendix D.3 states, for each of these claims, the observation that would undercut it.

## 5 LIMITATIONS

This paper makes a measurement claim about process evaluation in coding agents, and its conclusions should be read within the scope of that claim. The first limitation is task scope. Our analysis is conducted on file localization rather than the full space of coding-agent behaviors, because this setting provides the replayability, prefix reconstruction, and verifiable outcomes required by the method. File localization is therefore not a proxy chosen for convenience, but the smallest realistic environment in which the identification assumptions can be checked empirically. At the same time, this choice limits external validity: process structure in code editing, patch synthesis, or open-ended tool use may differ substantially from the structure observed here.

A second limitation is that the intervention-based step reference is formally identified but empirically weak. Of 190 intervention-estimated effects, 0 survive BH-FDR correction at $q = 0 . 1 0$ , and only 5.3% reach uncorrected $p < 0 . 0 5$ , which is consistent with the nominal chance rate. This means that we cannot certify which step mattered in any individual episode. The strongest attribution result in the paper is therefore not a correctness claim about a judge’s selected step, but a mechanism claim about how the judge’s verdict changes when downstream information becomes visible. More broadly, our conclusions about step-level attribution should be read as limits on measurement at feasible replay cost rather than as proof that step-local causal structure never exists.

A third limitation concerns the replay budget and the granularity of estimation. We estimate values at selected prefixes rather than at every step in the trajectory, and each estimation point is backed by a finite number of replays. As a result, the reported contributions are segment-level approximations rather than dense per-step measurements. Increasing replay density would improve resolution, but only up to the point allowed by the structure of the underlying uncertainty. Since our results place that uncertainty mainly at the task level rather than the step level, simply scaling replay count is unlikely to remove the core limitation by itself.

A fourth limitation is that several assumptions are only partially testable. Replayability and compaction can be checked empirically in the current harness, and we do so. By contrast, exogeneity of decoding noise is a structural assumption about the serving policy and cannot be verified directly from traces. Similarly, the intervention branch relies on a transferability assumption about the plausibility of substituted actions under a perturbed policy. These assumptions are explicit in the method, but they remain assumptions, and future work should test how sensitive the conclusions are to departures from them.

A fifth limitation concerns judge evaluation. We replicate the collider-displacement finding across 3 judge models and find no detectable heterogeneity in displacement, but the cross-model sample remains modest and the heterogeneity test is necessarily low-powered. The content-level characterization of what judges score is even less stable across models than the displacement mechanism itself. Our strongest claim therefore concerns the information-set effect in full-trace judging, not a complete universal account of all LLM-based process evaluators.

A sixth limitation is that the uncertainty decomposition is only partially validated on real traces. The policy-driven component is directly estimated from repeated executions, but the full three-way decomposition into task, policy, and model uncertainty is validated only on a synthetic knowngenerating process. In the main benchmark setting, we use this decomposition as an organizing framework for uncertainty analysis rather than as a fully solved inference problem. The empirical claim supported by the experiments is narrower and more robust: the uncertainty that matters for execution outcome is primarily structured at the task level.

Finally, the graph-based analyses are limited by structural coverage. Only 46.2% of agent objects resolve cleanly to nodes in the dependency graph, and free-text search queries remain only weakly represented by graph structure. This does not affect the main provenance-based findings, but it does limit how strongly graph-based conclusions can be generalized beyond the resolvable subset.

These limitations do not undermine the central measurement result of the paper, but they define its proper scope. Our claim is not that process evaluation is useless, nor that step-level causal attribution is impossible in every setting. Our claim is that, in a replayable coding-agent benchmark where the causal target can be defined precisely, action prediction, task uncertainty, and step attribution behave differently enough that they should not be treated as a single evaluative object.

## 6 CONCLUSION

Process evaluation is increasingly used to analyze coding agents beyond final task success, but existing practice often treats all process-level signals as if they supported the same kind of interpretation. This paper argues that such a view is too coarse. Action prediction, task uncertainty, and step attribution are different levels of analysis, and conclusions drawn at one level do not automatically transfer to the others.

To study this problem, we introduced a measurement framework centered on SCAE, a replay-based estimator derived from a structural causal model of agent execution. This framework allowed us to define a prefix-conditioned step-level causal quantity, estimate it under replay and intervention, and compare it against observational process evaluators under controlled changes to their information sets. Using this instrument, we asked three connected questions: what predicts the next action, where execution uncertainty resides, and what full-trace attribution methods actually measure.

The experiments yield three main findings. First, next actions are predicted primarily by execution provenance rather than by code-graph transitions, which shows that agents move mainly according to what they have just observed rather than according to repository structure alone. Second, execution uncertainty has meaningful instance-level structure but little step-level structure, which implies that the uncertainty most relevant to outcome is a property of the task rather than of individual steps. Third, full-trace judges shift their selected blame later when downstream steps become visible, isolating collider bias in the evaluation mechanism and showing that observational attribution is better understood as semantically enriched, downstream-informed relevance than as certified causal contribution.

Taken together, these findings explain a tension that has become common in process evaluation. Process-level signals can be useful and still fail to support the interpretation often assigned to them. Action-level predictors can be strong, task-level uncertainty can be structured, and full-trace judges can select semantically meaningful steps, while step-level causal localization remains infeasible at practical replay budgets. This is not a contradiction once the levels are separated; it is the main reason the paper insists on separating them.

The broader implication is methodological. Process evaluation should not be treated as a single benchmark target whose meaning is fixed in advance. Instead, evaluation methods should be validated against the level they actually measure. We hope this work encourages future research to build process evaluators whose targets are explicit, whose assumptions are testable, and whose outputs are interpreted according to the kind of evidence they genuinely provide.

## REFERENCES

Anastasios N. Angelopoulos and Stephen Bates. Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning, 16(4):494–591, 2023.

Anonymous. Evaluating process-level defects and control preservation in LLM coding agents. arXiv preprint, 2026. Citation anonymized for double-blind review.

Jose A. Arjona-Medina, Michael Gillhofer, Michael Widrich, Thomas Unterthiner, Johannes Brandstetter, and Sepp Hochreiter. RUDDER: Return decomposition for delayed rewards. In Advances in Neural Information Processing Systems (NeurIPS), 2019. arXiv:1806.07857.

Ivi Chatzi, Nina Corvelo Benz, Eleni Straitouri, Stratis Tsirtsis, and Manuel Gomez-Rodriguez. Counterfactual token generation in large language models. In Causal Learning and Reasoning (CLeaR), 2025. arXiv:2409.17027.

Mengzhuo Chen, Junjie Wang, Fangwen Mu, Yawen Wang, Zhe Liu, Huanxiang Feng, and Qing Wang. Seeing the whole elephant: A benchmark for failure attribution in llm-based multi-agent systems. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 19888–19905, 2026.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.

Jiawei He, Jie Jia, Chenbo Liu, Chaoyi Xue, Yapeng Song, Xikai Yang, and Dong Sun. Procctrlbench: Evaluating process-level defects and control preservation in llm coding agents. arXiv preprint arXiv:2605.20251, 2026.

Miriam Horovicz and Roni Goldshmidt. Tokenshap: Interpreting large language models with monte carlo shapley value estimation. In Proceedings of the 1st Workshop on NLP for Science (NLP4Science), pp. 1–8, 2024.

Milad Kazemi, Jessica Lally, Ekaterina Tishchenko, Hana Chockler, and Nicola Paoletti. Counterfactual influence in markov decision processes. arXiv preprint arXiv:2402.08514, 2024.

Reza Khanmohammadi, Erfan Miahi, Mehrsa Mardikoraem, Simerjot Kaur, Ivan Brugere, Charese Smiley, Kundan S Thind, and Mohammad M Ghassemi. Calibrating llm confidence by probing perturbed representation stability. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 10459–10525, 2025.

Vasilis Kontonis, Yuchen Zeng, Shivam Garg, Lingjiao Chen, Hao Tang, Ziyan Wang, Ahmed Awadallah, Eric Horvitz, John Langford, and Dimitris Papailiopoulos. Memento: Teaching llms to manage their own context. arXiv preprint arXiv:2604.09852, 2026.

Rui Li and Shuang Cao. From trajectories to graphs: Contract-checked editing for verifier-guided llm reasoning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 43259–43306, 2026.

Kaiqu Liang, Haimin Hu, Ryan Liu, Thomas L Griffiths, and Jaime Fernandez Fisac. Rlhs: Miti-´ gating misalignment in rlhf with hindsight simulation. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pp. 11457–11483, 2026.

Luckeciano C Melo, Panagiotis Tigas, Alessandro Abate, and Yarin Gal. Deep bayesian active learning for preference modeling in large language models. Advances in Neural Information Processing Systems, 37:118052–118085, 2024.

Mike Merrill, Alexander Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Shin, Thomas Walshe, E Kelly Buchanan, et al. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces. In International Conference on Learning Representations, volume 2026, pp. 40903–40986, 2026.

Susan A. Murphy. Optimal dynamic treatment regimes. Journal of the Royal Statistical Society: Series B, 65(2):331–355, 2003.

Preetum Nakkiran, Arwen Bradley, Adam Golinski, Eugene Ndiaye, Michael Kirchhof, and Sinead Williamson. Trained on tokens, calibrated on concepts: The emergence of semantic calibration in llms. In International Conference on Learning Representations, volume 2026, pp. 34128–34192, 2026.

Judea Pearl. Causality: Models, Reasoning, and Inference. Cambridge University Press, 2nd edition, 2009.

Suraj Prasai, Mengnan Du, Ying Zhang, and Fan Yang. Knowthyself: An agentic assistant for llm interpretability. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 41661–41663, 2026.

Shauli Ravfogel, Anej Svete, Vesteinn Snæbjarnarson, and Ryan Cotterell. Gumbel counterfactual´ generation from language models. In International Conference on Learning Representations, volume 2025, pp. 90486–90512, 2025.

James Robins. A new approach to causal inference in mortality studies with a sustained exposure period. Mathematical Modelling, 7(9–12):1393–1512, 1986.

Brendan Ross, Noel Vouitsis, Atiyeh Ashari Ghomi, Rasa Hosseinzadeh, Ji Xin, Zhaoyan Liu, ¨ Yi Sui, Shiyi Hou, Kin Kwan Leung, Gabriel Loaiza-Ganem, et al. Textual bayes: Quantifying prompt uncertainty in llm-based systems. In International Conference on Learning Representations, volume 2026, pp. 133607–133630, 2026.

Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar. Rewarding progress: Scaling automated process verifiers for llm reasoning. In International Conference on Learning Representations, volume 2025, pp. 60808–60838, 2025.

Chenglei Si, Tatsunori Hashimoto, and Diyi Yang. The ideation-execution gap: Execution outcomes of llm-generated versus human research ideas. In International conference on learning representations, volume 2026, pp. 74195–74240, 2026.

Lihao Sun, Hang Dong, Bo Qiao, Qingwei Lin, Dongmei Zhang, and Saravan Rajmohan. Llm reasoning as trajectories: Step-specific representation geometry and correctness signals. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 26872–26887, 2026.

Dennis Ulmer, Martin Gubri, Hwaran Lee, Sangdoo Yun, and Seong Oh. Calibrating large language models using their generations only. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15440–15459, 2024.

Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. Algorithmic Learning in a Random World. Springer, 2005.

Hanlin Wang, Jian Wang, Chak Tou Leong, and Wenjie Li. Steca: Step-level trajectory calibration for llm agent learning. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 11597–11614, 2025a.

Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce LLMs step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 2024. arXiv:2312.08935.

Qian Wang, Tianyu Wang, Zhenheng Tang, Qinbin Li, Nuo Chen, Jingsheng Liang, and Bingsheng He. Megaagent: A large-scale autonomous llm-based multi-agent system without predefined sops. In Findings of the association for computational linguistics: ACL 2025, pp. 4998–5036, 2025b.

Muning Wen, Ziyu Wan, Jun Wang, Weinan Zhang, and Ying Wen. Reinforcing llm agents via policy optimization with action decomposition. Advances in Neural Information Processing Systems, 37: 103774–103805, 2024.

Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. Agentless: Demystifying LLM-based software engineering agents. arXiv preprint arXiv:2407.01489, 2024.

Yuxi Xia, Pedro Henrique Luz De Araujo, Klim Zaporojets, and Benjamin Roth. Influences on llm calibration: A study of response agreement, loss functions, and prompt styles. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3740–3761, 2025.

Frank Fangzheng Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, et al. Theagentcompany: benchmarking llm agents on consequential real world tasks. Advances in Neural Information Processing Systems, 38, 2026a.

Xiaoliang Xu, Huang Yuan, Junmei Wang, and Can Xu. Gcig: Graphrag-based cross-document instruction generation for boosting llm reasoning. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pp. 17593–17609, 2026b.

Matthew Yang, Hao Bai, Ian Wu, Gene Yang, Amrith Setlur, and Aviral Kumar. Int: Self-proposed interventions enable credit assignment in llm reasoning. In International Conference on Learning Representations, volume 2026, pp. 85054–85091, 2026.

Guibin Zhang, Junhao Wang, Junjie Chen, Wangchunshu Zhou, Kun Wang, and Shuicheng Yan. Agentracer: Who is inducing failure in the llm agentic systems? In International Conference on Learning Representations, volume 2026, pp. 11377–11399, 2026a.

Zheng Zhang, Ziwei Shan, Kaitao Song, Yexin Li, and Kan Ren. Linking process to outcome: Conditional reward modeling for llm reasoning. In International Conference on Learning Representations, volume 2026, pp. 87995–88016, 2026b.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and chatbot arena. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023. arXiv:2306.05685.

## A NOTATION AND FORMAL MATERIAL

This appendix supplies the formal material behind $\ S 3 { \mathrm { : } }$ the notation used throughout (Table 6), the consequence of each conditioning set for current practice, where on the causal ladder our estimates sit, the three ways identification can fail, and the proofs of Proposition 1 and Theorem 1.

## A.1 NOTATION

<table><tr><td>Symbol</td><td>Definition</td></tr><tr><td>q</td><td>natural-language issue text of an episode</td></tr><tr><td> $\overset { \cdot } { G } = ( V , E )$ </td><td>repository graph over files, modules and symbols</td></tr><tr><td> $F ^ { \star }$ </td><td>gold target set: files modified by the reference patch</td></tr><tr><td> $\hat { F }$ </td><td>file set predicted by the agent</td></tr><tr><td> $Y$ </td><td>task-level outcome  $\mathcal { H } [ F ^ { \star } \subseteq \hat { F } ]$ </td></tr><tr><td> $Y _ { f } , Y _ { \mathrm { r e c } }$ </td><td>per-file indicator; recall  $| F ^ { \star } \cap \hat { F } | / | F ^ { \star } |$ </td></tr><tr><td> $A _ { t } , O _ { t }$ </td><td>action taken at step t; tool output it returns</td></tr><tr><td> $S _ { t }$ </td><td>agent state at step  $t ; S _ { 1 } = \sigma ( q , \mathrm { s y s } )$ </td></tr><tr><td> $\tau _ { 1 : t }$ </td><td>trace prefix  $( A _ { 1 } , O _ { 1 } , \dots , A _ { t } , O _ { t } )$ </td></tr><tr><td> $T$ </td><td>number of steps in an episode;  $t / T$  is normalized position</td></tr><tr><td> $Z$ </td><td>latent task difficulty (exogenous)</td></tr><tr><td> $G _ { \pi } ^ { ( t ) }$ </td><td>decoding noise at step t, Gumbel(0, 1) (exogenous)</td></tr><tr><td> $U _ { \mathrm { e n v } } ^ { ( t ) }$ </td><td>environment noise at step t (exogenous, pinned)</td></tr><tr><td> $U ^ { ( \geq t ) }$ </td><td>all exogenous noise from step t onward</td></tr><tr><td> $\pi , { \tilde { \pi } }$ </td><td>deployed policy; raised-temperature perturbed policy</td></tr><tr><td> $\widehat { H } ( \widehat { \pi } _ { t } )$ </td><td>estimated action entropy at step t, Eq. equation 13</td></tr><tr><td> $\tau _ { H }$ </td><td>entropy threshold selecting the branch of Eq. equation 14</td></tr><tr><td> $m$ </td><td>replayed continuations per estimation point</td></tr><tr><td> $v _ { t }$ </td><td>anytime value  $\mathbb { E } [ Y \mid \tau _ { 1 : t } , q , G ]$ </td></tr><tr><td> $v _ { 0 }$ </td><td>pre-execution prediction  $\mathbb { E } [ Y \mid q , G ]$ </td></tr><tr><td> $\Delta _ { t }$ </td><td>step contribution  $v _ { t } - v _ { t - 1 }$ </td></tr><tr><td> $U _ { t } ^ { \mathrm { e x e c } }$ </td><td>replay variance of the outcome at estimation point t</td></tr><tr><td> $\theta _ { t } ( a , a ^ { \prime } )$ </td><td>step-level causal contrast of  $\operatorname { E q . }$  equation 1</td></tr><tr><td> $\mathrm { P N } _ { t }$ </td><td>probability of necessity at step t</td></tr><tr><td> $\mathrm { I n f l } ( t )$ </td><td>counterfactual influence at step t</td></tr><tr><td> $U _ { \mathrm { t a s k } } , U _ { \mathrm { p o l i c y } } , U _ { \mathrm { m o d e l } }$ </td><td>the three components of Eq. equation 16</td></tr><tr><td> $\Sigma$ </td><td>Laplace posterior covariance of the vo model</td></tr><tr><td> $\mathcal { C } ( q )$ </td><td>conformal prediction set for the target files</td></tr></table>

Table 6: Notation used throughout the paper. Symbols are introduced in $\ S 3$ and reused without redefinition in §4.

## A.2 CONSEQUENCES OF THE CONDITIONING SET FOR CURRENT PRACTICE

Proposition 1 licenses exactly one conditioning set: the realized prefix. Table 7 states, for each departure from it that appears in current practice, the resulting bias and the signature it should leave in data. The mechanism in the first row is standard — conditioning on a descendant of the treatment opens a non-causal path and biases the estimate (Pearl, 2009) — but it is not usually applied to LLM-as-a-judge attribution (Zheng et al., 2023), which presents the entire trajectory and the realized outcome and then asks which step was at fault. That row is also the only one yielding a falsifiable prediction without requiring that our own estimator be correct, which is why §4.6 tests it and why we foreground it.

## A.3 THE CAUSAL LADDER, INSTANTIATED FOR LOCALIZATION

## A.4 POSITION ON THE CAUSAL LADDER

With API-served models we obtain per-token log-probabilities only in a reference arm; in the main arm we estimate $\widehat { \pi } _ { t }$ over action equivalence classes by resampling. The intervened step therefore admits Gumbel–max abduction and is genuinely Level III, whereas the continuation marginalizes residual policy noise conditional on the prefix and is Level II.5 in the terms of Table 8. Environment noise is Level III throughout, being pinned by construction. We therefore claim action-level counterfactuals with a prefix-conditioned continuation, and not exact token-level counterfactuals. Quantifying the gap requires the reference arm, whose construction we describe but whose results are not included here.

<table><tr><td>Practice</td><td>Conditions on</td><td>Consequence</td></tr><tr><td>Full-trace LLM judge</td><td>prefix + successors + outcome</td><td>Successors are descendants of  $A _ { t } ,$  and condition- ing on a descendant opens a biasing path. Pre- dicted signature: blame displaced toward later</td></tr><tr><td>Process reward models</td><td>nominally the prefix</td><td>steps. Labels back-filled from completed trajectories reintroduce future information.</td></tr><tr><td>Ablation with restart</td><td>nothing</td><td>Resamples all noise, so it estimates a marginal interventional contrast and discards the realized history.</td></tr><tr><td>analysis</td><td>Data-flow“dead step&quot; syntactic dependencies</td><td>Conflates data-flow reachability with causal ef- fect on the outcome (§A.7).</td></tr></table>

Table 7: Each row misestimates Eq. equation 1 for a distinct reason, and none of the four reasons is unmeasured confounding. Only the first row yields a signature testable without a causal reference, which is why it is the attribution result we foreground in §4.6.

<table><tr><td>Level</td><td>Expression</td><td>Meaning for localization</td><td>Typical practice</td></tr><tr><td>I</td><td> $P ( \hat { F } \mid q )$ </td><td>association between query and result</td><td>retrieval recall@k</td></tr><tr><td>II</td><td> $P ( \hat { F } \mid d o ( a ^ { \prime } ) )$ </td><td>alter a step and restart</td><td>tool-ablation studies</td></tr><tr><td>II.5</td><td> $P ( \hat { F } \mid \tau _ { 1 : t - 1 } , d o ( a ^ { \prime } ) )$ </td><td>fix realized history, alter one step</td><td>SCAE (continuation)</td></tr><tr><td>III</td><td> $P ( \hat { F } _ { a ^ { \prime } } \mid \tau _ { \mathrm { o b s } } )$ </td><td>fix realized noise, alter one step</td><td>SCAE (intervened step)</td></tr></table>

Table 8: Levels II and II.5 differ in whether the episode’s own history is retained. Most agent ablation studies operate at Level II while being described in counterfactual language, which is the confusion the distinction is meant to prevent. §A.4 states where the boundary falls for SCAE.

## A.5 WHERE IDENTIFICATION BREAKS

Three failure modes deserve naming, and none of them is confounding.

Positivity. This is the binding constraint in practice. Necessity-type quantities are the ones it binds hardest on: the probability of necessity at step t,

$$
\mathrm { P N } _ { t } = P \big ( Y _ { A _ { t } = a _ { t } ^ { \star } } = 1 \mid \tau _ { 1 : t - 1 } , A _ { t } = a _ { t } , Y = 0 \big ) ,
$$

asks whether the outcome would have been correct had the agent taken the better action $a _ { t } ^ { \star }$ , given that it took $a _ { t }$ and failed. It is defined only when $\pi ( \boldsymbol { a } _ { t } ^ { \star } \mid \tau _ { 1 : t - 1 } ) > 0$ . When the policy is effectively deterministic at step t this fails, and no amount of observational data recovers the quantity, because the required comparison arm is never realized. This is why §B.4 measures $\pi ( \cdot \mid \tau _ { 1 : t - 1 } )$ rather than assuming it, and why Eq. equation 14 switches to a soft intervention exactly where the measurement says overlap is absent.

Context compaction. Let $\varphi$ denote the harness’s compaction map, so that the state actually conditioned on is $\varphi ( h _ { t } )$ rather than $h _ { t }$ . When $\varphi$ discards content, $\bar { h _ { t } } \stackrel { - } { \mapsto } \varphi ( h _ { t } )$ is not invertible and there exists $D _ { t } \notin \tau _ { 1 : t - 1 }$ that influenced $A _ { t }$ . Assumption 1 then fails, Step 1 of the proof in $\ S \mathrm { A } . 6$ does not go through, and $D _ { t }$ acts as an unmeasured confounder between $A _ { t }$ and $Y$ . The estimand is unidentified, not merely harder to estimate: no reweighting of the observed trace recovers it, because the confounder is absent from the log by construction. What makes this tractable in principle is that compaction is triggered by the harness rather than by nature, so it can be randomized — assigning compaction at a fixed prefix converts a sensitivity-analysis problem into a controlled experiment. We reserved the hook for this but did not run it, since no episode in our sweep compacted spontaneously.

Counterfactual influence decay. Even with Gumbel–max abduction at the intervened step, the coupling between the factual and counterfactual continuations weakens as the two paths diverge, so a nominal Level III quantity degrades toward a Level II one over long horizons (Kazemi et al., 2024). To keep this measurable rather than rhetorical we define

$$
\mathrm { I n f } ( t ) = \mathrm { T V } \Bigl ( P \bigl ( \hat { F } \mid \tau _ { \mathrm { o b s } } , d o ( a ^ { \prime } ) \bigr ) , P \bigl ( \hat { F } \mid d o ( a ^ { \prime } ) \bigr ) \Bigr ) ,
$$

the total-variation distance between the counterfactual predicted-set distribution that retains the observed trace and the one that discards it. Infl(t) near zero means the realized history has stopped constraining the continuation, so the estimate should be reported as interventional rather than counterfactual.

## A.6 PROOFS

Throughout, we read Assumption 2 in its joint form: the decoding noises $\{ G _ { \pi } ^ { ( t ) } \} _ { t \geq 1 }$ are mutually independent and independent of $( Z , U _ { \mathrm { e n v } } )$ . This is the standard exogeneity assumption for a sampler whose temperature is a decoder parameter, and it is what “exogenous” asserts; we make it explicit because the proof uses independence across steps, not only at step t.

## A.6.1 PROOF OF PROPOSITION 1

Step 1: conditional on the prefix, the action is a function of decoding noise alone. Fix a prefix value $\tau _ { 1 : t - 1 } = \tau$ with $P ( \tau _ { 1 : t - 1 } = \tau ) > 0$ . By Assumption 1 there is a deterministic map $f$ with $S _ { t } = f ( q , \tau )$ , so on the conditioning event $S _ { t }$ takes a single fixed value. By Eq. equation 2,

$$
A _ { t } = \arg \operatorname* { m a x } _ { k } \big ( \log \pi _ { k } ( f ( q , \tau ) ) + G _ { \pi , k } ^ { ( t ) } \big ) = : g _ { \tau } \big ( G _ { \pi } ^ { ( t ) } \big ) ,
$$

a measurable function of $G _ { \pi } ^ { ( t ) }$ only.

Step 2: that noise is independent of the prefix and of the continuation noise. The prefix $\tau _ { 1 : t - 1 }$ is, by the structural equations, a measurable function of $\left( q , Z , \{ G _ { \pi } ^ { ( s ) } \} _ { s < t } , \{ U _ { \mathrm { e n v } } ^ { ( s ) } \} _ { s < t } \right)$ . Write $U ^ { ( \geq t ) } = \left( \{ G _ { \pi } ^ { ( s ) } \} _ { s > t } , \{ U _ { \mathrm { e n v } } ^ { ( s ) } \} _ { s \geq t } \right)$ . By Assumption 2 in its joint form, $G _ { \pi } ^ { ( t ) }$ is independent of the collection $( Z , \{ G _ { \pi } ^ { ( s ) } \} _ { s \neq t } , U _ { \mathrm { e n v } } )$ , hence

$$
G _ { \pi } ^ { ( t ) } \perp ( Z , { \cal U } ^ { ( \geq t ) } , \tau _ { 1 : t - 1 } ) .\tag{22}
$$

Combining Eq. equation 22 with Step 1, and noting that conditioning on $\tau _ { 1 : t - 1 }$ leaves the law of $G _ { \pi } ^ { ( t ) }$ unchanged,

$$
A _ { t } \perp \left( Z , U ^ { ( \geq t ) } \right) \mid \tau _ { 1 : t - 1 } ,
$$

which is the first claim.

Step 3: the potential outcome is a function of the continuation noise. Fix an action a. Recursively applying $O _ { s } = \mathcal { E } ( A _ { s } , G ; U _ { \mathrm { e n v } } ^ { ( s ) } ) , S _ { s + 1 } = \mathcal { T } ( S _ { s } , A _ { s } , O _ { s } )$ and $\hat { F } = \phi ( S _ { T } )$ from step t onward, with $A _ { t }$ set to $^ { a , }$ the potential outcome under the intervention is a deterministic function

$$
Y _ { A _ { t } = a } = h \big ( a , \tau _ { 1 : t - 1 } , U ^ { ( \geq t ) } \big ) ,\tag{23}
$$

where determinism of each $O _ { s }$ given $( A _ { s } , G , U _ { \mathrm { e n v } } ^ { ( s ) } )$ is exactly Assumption 3. By Step 2 and Eq. equation $2 3 , A _ { t } \perp Y _ { A _ { t } = a } \mid \tau _ { 1 : t - 1 }$ for every a, which is conditional ignorability at step t.

Step 4: identification. Consistency holds by construction in a structural causal model: on the event $\left\{ A _ { t } \right. = { { a } } \}$ we have $Y = Y _ { A _ { t } = a }$ . Therefore, for any a with $\pi ( { a \ } | \ \tau _ { 1 : t - 1 } ) > 0$ , so that the conditional expectation is well defined,

$$
\begin{array} { r l } & { \mathbb { E } [ Y \mid \tau _ { 1 : t - 1 } , d o ( A _ { t } { = } a ) ] \stackrel { \mathrm { ( i ) } } { = } \mathbb { E } [ Y _ { A _ { t } = a } \mid \tau _ { 1 : t - 1 } ] } \\ & { \stackrel { \mathrm { ( i i ) } } { = } \mathbb { E } [ Y _ { A _ { t } = a } \mid \tau _ { 1 : t - 1 } , A _ { t } { = } a ] } \\ & { \stackrel { \mathrm { ( i i i ) } } { = } \mathbb { E } [ Y \mid \tau _ { 1 : t - 1 } , A _ { t } { = } a ] . } \end{array}
$$

where (i) is the definition of the interventional expectation in M, (ii) is Step 3, and (iii) is consistency. This is Eq. equation 6, and it is the one-step g-formula (Robins, 1986) with a known assignment mechanism. □

Relation to dynamic treatment regimes. The sequential structure here is formally that of a dynamic treatment regime (Robins, 1986; Murphy, 2003), but the difficulty is inverted. In epidemiological applications the central obstacle is time-varying confounding: the analyst must argue that everything the treatment decision depended on was recorded. Here that argument is free, because the policy is a function of a context we log in full and its decoding noise is a decoder property. What is not free is positivity, which those applications typically obtain from natural variation and which we must measure and then work around (§A.5, §B.4).

Remark on what the proof does and does not deliver. The positivity requirement in Step 4 is not a technicality but the binding constraint in practice, which is what forces the hybrid estimator of Eq. equation 14. Note also that Step 2 is where the argument fails for the conditioning sets of Table 7: conditioning additionally on $A _ { t ^ { \prime } }$ for $t ^ { \prime } > t$ or on $Y$ conditions on functions of $U ^ { ( \geq i ) }$ , which breaks Eq. equation 22 and hence Step 3. Similarly, if the harness compacts context then $S _ { t }$ is not a function of $\left( q , \tau _ { 1 : t - 1 } \right)$ , Assumption 1 fails, and Step 1 does not go through.

## A.6.2 PROOF OF THEOREM 1

Telescoping. By definition $\begin{array} { r } { \Delta _ { t } = v _ { t } - v _ { t - 1 } , \mathrm { s o } \sum _ { t = 1 } ^ { T } \Delta _ { t } = v _ { T } - v _ { 0 } } \end{array}$ as a finite telescoping sum. The outcome $Y \overset { \cdot } { = } \mathcal { H } [ F ^ { \star } \subseteq \phi ( S _ { T } ) ]$ ] is a deterministic function of $S _ { T }$ , which by Assumption 1 is a function of $( q , \tau _ { 1 : T } ) ;$ ; therefore $v _ { T } \stackrel { \cdot } { = } \mathbb { E } [ Y \mid \tau _ { 1 : T } , q , G ] = Y$ almost surely, and $v _ { 0 } = \mathbb { E } [ Y \mid q , G ]$ by definition. This gives Eq. equation 9.

The action/observation decomposition. Insert the intermediate conditioning set $( \tau _ { 1 : t - 1 } , A _ { t } { = } a _ { t } )$ which lies between $\tau _ { 1 : t - 1 }$ and $\tau _ { 1 : t } = ( \tau _ { 1 : t - 1 } , a _ { t } , o _ { t } )$

$$
\begin{array} { r } { \Delta _ { t } = \underbrace { \left( \mathbb { E } [ Y \mid \tau _ { 1 : t - 1 } , A _ { t } = a _ { t } ] - \mathbb { E } [ Y \mid \tau _ { 1 : t - 1 } ] \right) } _ { \mathrm { ( A ) } } + \underbrace { \left( \mathbb { E } [ Y \mid \tau _ { 1 : t } ] - \mathbb { E } [ Y \mid \tau _ { 1 : t - 1 } , A _ { t } = a _ { t } ] \right) } _ { \mathrm { ( B ) } } . } \end{array}\tag{24}
$$

Term (B) of Eq. equation 24 is the contribution of the observation: the change in expected outcome from learning $O _ { t } = o _ { t }$ , holding the action fixed. For term (A), the law of total expectation over the policy’s action distribution gives

$$
\mathbb { E } [ Y \mid \tau _ { 1 : t - 1 } ] = \sum _ { a } \pi ( a \mid \tau _ { 1 : t - 1 } ) \mathbb { E } [ Y \mid \tau _ { 1 : t - 1 } , A _ { t } = a ] = \mathbb { E } _ { a \sim \pi } { \left[ \mathbb { E } [ Y \mid \tau _ { 1 : t - 1 } , A _ { t } = a ] \right] } ,
$$

where the sum ranges over the support of $\pi ( \cdot \mid \tau _ { 1 : t - 1 } )$ , so every conditional expectation appearing in it is well defined. Applying Proposition 1 to each term replaces every observational conditional by its interventional counterpart, giving the action effect relative to the policy’s own baseline,

$$
\begin{array} { r l } & { \Delta _ { t } = \underbrace { \left( \mathbb { E } [ Y \mid \tau _ { 1 : t - 1 } , d o ( A _ { t } = a _ { t } ) ] - \mathbb { E } _ { a \sim \pi } [ \mathbb { E } [ Y \mid \tau _ { 1 : t - 1 } , d o ( A _ { t } = a ) ] ] \right) } _ { \mathrm { ~ w s t i o n ~ e f f e c t ~ } } } \\ & { \qquad + \underbrace { \left( \mathbb { E } [ Y \mid \tau _ { 1 : t } ] - \mathbb { E } [ Y \mid \tau _ { 1 : t - 1 } , A _ { t } = a _ { t } ] \right) } _ { \mathrm { ~ o b s e r v a t i o n ~ } } . } \end{array}\tag{25}
$$

Remark. Two readings of Eq. equation 25 are excluded. The baseline is the policy’s own action distribution, so $\Delta _ { t }$ answers how much better or worse a step was than what this agent would typically have done there, not than the optimal action — the latter is a necessity-type contrast requiring intervention (§A.5). And Eq. equation 9 holds for population quantities, so with finitely many replays per $v _ { t }$ the empirical sum carries error; we use that error as an internal consistency check rather than reporting it as exact. The proof uses boundedness of $Y$ only through the existence of the conditional expectations, so both identities hold verbatim for the recall outcome $Y _ { \mathrm { r e c } } \in [ 0 , 1 ]$ and for the per-file indicator $Y _ { f }$ , which is what licenses the outcome choice of §3.7 without restating the theorem.

## A.7 DATA-FLOW REACHABILITY IS NOT CAUSAL EFFECT

Data-flow “inertness” is sometimes offered as a proxy for causal irrelevance: if a step’s output is never read downstream, the step is taken to have had no effect. The two notions come apart in both directions, which is why Table 7 lists this as a distinct failure mode.

Inert but causally potent. A grep that returns nothing contributes no content to any subsequent context window, so it is inert by any data-flow criterion. It may nonetheless cause the agent to abandon the correct directory, because absence of information is itself information: the empty result licenses the inference that the target is elsewhere. Under our estimand this shows up as a large $| \Delta _ { t } |$ at a step with no outgoing data flow.

Data-carrying but causally inert. Conversely, a long file read may inject thousands of tokens into the context while leaving every subsequent action unchanged. Fixing the prefix and deleting the read verifies this directly: if the continuation is unchanged across replays, the step’s contribution is zero despite its data-flow footprint being large. The asymmetry matters because taxonomies that score steps by downstream data flow will systematically misrank exactly the steps that a causal criterion identifies as decisive.

## B IMPLEMENTATION AND EXPERIMENTAL DETAILS

## B.1 ESTIMATION ALGORITHM

Algorithm 1 SCAE: step contributions with per-step identification   
Require: episode $( q , \mathrm { r e p o } )$ , rollout $\tau _ { 1 : T : }$ , outcome $Y .$ , estimation points $\mathcal { P } \subseteq \{ 1 , \dots , T \}$ , replays   
m, threshold $\tau _ { H }$   
Ensure: $\{ \widehat { \Delta } _ { t } \} _ { t \in \mathcal { P } } ,$ entropy $\{ \widehat { H } _ { t } \}$   
1: reset repository to base commit; pin locale, timezone, terminal width, hash seeds   
2: for $t \in { \dot { \mathcal { P } } }$ do   
3: $\tau _ { 1 : t - 1 } \gets \tau$ TRUNCATEBEFORESTE $\mathbf { \Psi } ^ { \mathrm { p } } ( \tau , t )$ ▷ rewrite rollout log at item granularity   
4: ASSERTNODANGLINGCALLS(τ <sub>−</sub> )   
5: $A  \emptyset$   
6: for $i = 1$ to m do ▷ common random numbers: shared prefix and pinned env   
7: reset repository; fork agent from $\tau _ { 1 : t - 1 } ;$ resume with the byte-identical continuation   
prompt   
8: $A  A \cup$ {ACTIONBUCKET $( A _ { t } ^ { ( i ) } ) \}$ ▷ bucket = (tool, target node in G)   
9: end for   
10: $\hat { H } _ { t } \gets$ ENTROPY(A)   
11: if $| \{ A \} | \ge 2$ and $\hat { H } _ { t } > \tau _ { H }$ then ▷ overlap available: observational branch   
12: $\begin{array} { r } { \widehat { v _ { t } }  \frac { 1 } { m } \sum _ { i } Y ^ { ( i ) } } \end{array}$ from replays at $t ; \widehat { v } _ { t - 1 }$ likewise from replays at $t - 1$   
13: $\widehat { \Delta } _ { t } \gets \widehat { v } _ { t } - \widehat { v } _ { t - 1 }$   
14: else ▷ degenerate policy: soft intervention, Asm. 4   
15: draw $a ^ { \prime }$ adjacent to $a _ { t }$ in $G ,$ , or an oracle action inspecting a gold file   
16: $o ^ { \prime } \gets$ execute $a ^ { \prime }$ for real in the reset repository ▷ never a fabricated output   
17: splice $( a ^ { \prime } , o ^ { \prime } )$ into $\tau _ { 1 : t - 1 } ;$ resume m times; average to get $\widehat { \mathbb { E } } [ Y \mid \tau _ { 1 : t - 1 } , d o ( A _ { t } = a ^ { \prime } ) ]$   
18: $\widehat { \Delta } _ { t } \gets \widehat { \mathbb { E } } [ Y \mid \tau _ { 1 : t - 1 } , d o ( A _ { t } = a ^ { \prime } ) ] - \widehat { \mathbb { E } } [ Y \mid \tau _ { 1 : t - 1 } , d o ( A _ { t } = a _ { t } ) ]$   
19: end if   
20: record which branch was taken, so that the reported estimand is labelled π or $\tilde { \pi }$   
21: end for

Two implementation invariants are load-bearing and easy to violate. First, the continuation prompt must be byte-identical across all arms: it enters the context and is therefore part of the estimand, so

any difference between arms becomes an undeclared intervention. Second, the spliced observation $o ^ { \prime }$ must be the genuine result of executing $a ^ { \prime } ;$ substituting a plausible-looking output intervenes on the observation rather than the action, which is a different causal quantity.

## B.2 EXPERIMENTAL SETTINGS

Agent and model. Codex CLI driven over its JSON-RPC app-server with capabilities.experimentalApi enabled, so that thread paths are exposed and forking by path is available. Sandbox read-only, approval policy never; we verify that zero server-side approval requests are received, since a nonzero count would mean the read-only restriction was not in force. Backing model gpt-5.5-0424-global. Per-turn timeout 900 s.

Environment pinning. Locale, timezone, terminal width and hash seeds are fixed; tool outputs are totally ordered and truncation points fixed. §4.1 reports the empirical verification of this pinning, at 1.00 byte-identical replay over 100 actions.

Estimation points and replays. Attribution uses $m \ = \ 8$ replays per estimation point and 5 equidistant estimation points per episode, with the recall outcome of §3.7. Entropy probes use 4–6 replays per point, which is why Table 9 reports entropies on a coarse grid: with 4 replays the attainable nonzero values are 0.56, 1.04 and 1.39 nats. The entropy threshold is $\tau _ { H } = 0 ;$ , so the observational branch is taken exactly when the replays realize at least two distinct action buckets; this is the most permissive threshold, and it makes the reported 50% an upper bound on how often observational identification is available.

Judge conditions. Both judge conditions use the same model and the same rubric, and differ only in the visible portion of the trace: the full-trace condition sees $\tau _ { 1 : T }$ and the realized outcome $Y ;$ the prefix-only condition sees a fixed 60% prefix with the outcome withheld. The truncation fraction is fixed in advance and does not depend on any estimate. This matters, and §D.2 records why: an earlier design truncated just before the step selected by the causal estimate, which would have produced a positive displacement even under the null.

Covariates for $v _ { 0 }$ . The 7 pre-execution covariates are: issue description length; count of explicitly named files; presence of a traceback; error-term density; repository size; distinct identifier count; and the number of repository files containing those identifiers, which serves as an ambiguity proxy for $U _ { \mathrm { t a s k } }$ . None of these may read the trace, since doing so would leak execution-time information into a pre-execution predictor.

## B.3 THE BASELINE SUITE IN FULL

Because this paper compares measurement procedures rather than agents, each research question carries its own reference set, and every reference is either a competing predictor evaluated on the identical points or a null with a closed form. We list the full inventory here, since §4.1 only summarizes it.

Next-action prediction (§4.2). Nine predictors, all scored on the same decision points with uncovered points counted as misses rather than dropped (Table 3): a repository-frequency marginal; a visit-history frequency model; a strengthened code-dependency-graph predictor that seeds a multi-source proximity ranking from the entire visited set rather than from the last step alone; four execution-provenance variants (uniform over previously mentioned paths, recency-weighted, frequency-weighted, and last-output-only with within-output position ranking); and two tool-routed candidate-set constructions. We additionally contrast a factored $P ( \mathrm { t o o l } ) \bar { P ( \mathrm { o b j } ) }$ model against a routed one (Table 1). The graph predictor is deliberately strengthened rather than strawmanned, because the claim of §4.2 is that code structure loses even when given every advantage we could construct.

Graph and reuse structure (§4.3–§4.4). A within-episode permutation null, which preserves the multiset of visited objects and destroys only their order, and random same-size file sets for the region comparison. The permutation null is the load-bearing choice: it holds the working region fixed, which is exactly what separates “the graph describes where the agent works” from “the graph describes how it moves”. Revisit statistics are compared against closed-form uniform expectations rather than sampled draws.

Execution uncertainty (§4.5). Two controls in opposite directions. The superseded Bernoulli entropy approximation serves as a negative control, and a quantity known to respond serves as a positive control at $\rho = - 0 . 4 2 5$ ; without the latter, a null relating action entropy to replay variance would be uninterpretable, since it could reflect a dead outcome measure rather than an absent relationship. Structural probes with no mechanical coupling to the mean (step position, tool type, instance identity) are used in place of probes that are mechanically coupled.

Attribution (§4.6). Four reference sets. First, 3 judge models including a code-specialised one (Table 5). Second, a prefix-only against a full-trace ablation of the judge’s own information set, which is the only comparison here that needs no external reference. Third, five zero-LLM structural rules (Table 4), which test whether the judge is reducible to a positional or provenance heuristic. Fourth, a uniform-random selector reported at its exact per-episode expectation $1 / T _ { i }$ rather than as a sampled draw, so that the chance level carries no Monte Carlo noise.

Two families we do not implement. Back-filled process reward models and ablation-with-restart are absent, and both leave a gap we state rather than paper over. A trained process reward model might recover causal structure despite misspecified supervision, which our theoretical account predicts it will not but our experiments do not test. Ablation-with-restart, being a different estimand (Table 7), would need a separate comparison asking how far the marginal contrast departs from the history-conditioned one — a measurable quantity we have not measured.

## B.4 POSITIVITY PROBE: ENTROPY AT EACH DECISION POINT

Identification of Eq. equation 1 needs overlap at the step being estimated, and §3.6 treats this as something to measure rather than grant. Probing repeated replays from identical prefixes, action entropy is non-zero at 50% of 10 probed decision points across 3 episodes, with a rank correlation with normalized position of +0.072. Table 9 gives every probed point and Figure 7 plots them.
<table><tr><td>Episode</td><td>Entropy (nats) by normalized position</td><td>Pattern</td></tr><tr><td> $\mathrm { d j a n g o - 1 0 0 9 7 ( 1 6 s t e p s ) }$ </td><td> $0 . 1 2 \colon 0  0 . 3 8 \colon 1 . 0 4  0 . 6 2 \colon 1 . 0 4$ </td><td>det.—stoch.-stoch.</td></tr><tr><td>astropy-14369 (29 steps)</td><td> $0 . 1 0 \colon 0  0 . 3 1 \colon 0 . 5 6  0 . 5 2 \colon 0 . 5 6  0 . 7 6 \colon 0$ </td><td>det.—stoch.-det.</td></tr><tr><td>astropy-12907 (8 steps)</td><td> $0 . 2 5 \colon 0 . 4 5  0 . 5 0 \colon 0  0 . 7 5 \colon 0$ </td><td>stoch.-det.</td></tr></table>

Table 9: Action entropy at all 10 probed decision points across 3 episodes. Overlap is present at some steps and absent at others within the same episode, and the three episodes realize three different patterns. This is what forces the per-step branch selection of Eq. equation 14: neither a globally observational nor a globally interventional estimator is correct. Entropies fall on a coarse grid because each point is probed with 4–6 replays.

## B.5 STATISTICAL CONVENTIONS

Conventions are fixed once and applied throughout, so that no comparison is reported under a rule chosen after seeing it. Confidence intervals are episode- or instance-level bootstrap. Nulls are within-episode permutation or closed-form expectation, and never sampled where a closed form exists. Ranking metrics use the closed-form expectation over random tie-breaking, since count-based scorers tie heavily and a single arbitrary tie-break would be an undeclared design choice. Predictors with different candidate-set sizes are compared on a common denominator, with points whose realized action falls outside the candidate set scored as misses rather than dropped — otherwise a predictor could buy accuracy by abstaining. Multiple comparisons use BH-FDR at $q = 0 . 1 0$ , and we label exploratory versus confirmatory findings accordingly. Our Wilcoxon implementation declines to compute a p-value below $n = 6$ rather than reporting an unreliable one.

![](images/f6c74c5c0e8ced57d1b66fd74752d68512328d58134db66a8699003cd14ed5ad.jpg)

(b) neither branch dominates  
![](images/117273838eddbc682374f4534992c40fd5fb28a2d51a862a4afe084b479b8961.jpg)  
Figure 7: Overlap is a local property, not a global one. Each marker is one probed decision point, positioned by its normalized position in the episode and coloured by episode. Points on the zero line admit no observational identification and are routed to the intervention branch of Eq. equation 14; points above it are estimated observationally under Proposition 1. 50% of points lie above zero, and the rank correlation with position is only +0.072, so overlap is not a monotone function of trajectory depth. We report the sample size plainly: 10 points establish that overlap varies within episodes, which is all the estimator requires, but not the distribution of that variation, and the near-zero rank correlation should not be read as evidence of no relationship.

## B.6 CHOICE OF OUTCOME VARIABLE

§3.7 adopts recall over the gold file set. The choice was made by screening six candidate outcomes on intervention sensitivity, measured as the paired effect divided by its standard deviation: recall attains 2.505, against 2.165 for $F _ { 1 }$ and 1.643 for binary success. The second criterion is headroom: 98.6% of instances can move under recall against only 73.2% under the binary outcome, whose failures cannot fall further.

The mechanism behind those numbers is worth stating, because it is the reason a formally adequate outcome can still be empirically useless. With Y binary at the task level, single-target instances are solved almost always, so $\Delta _ { t } \equiv 0$ and attribution says nothing. Refining to the per-file indicator $Y _ { f }$ localizes the question to which step decided whether f was found, but degenerates whenever f is unreachable under every prefix. Recall retains signal from the remaining targets when one is hopeless, which is what reduces the degenerate rate to the 22.9% reported in §4.5. Theorem 1 requires only boundedness, so this substitution changes no formal result (§A.6).

## C ADDITIONAL RESULTS

## C.1 PER-REPOSITORY LOCALIZATION

The high aggregate success rate is itself a constraint on what the corpus can support. It is why the binary outcome has almost no headroom (§B.6), why attribution is restricted to the multi-target subset, and why a constant predictor is nearly calibrated — which in turn is why §C.7 reports selective-risk quantities alongside ECE rather than ECE alone.

## C.2 ADDITIONAL MEASUREMENTS BEHIND THE ACTION-LEVEL RESULT

§4.2 reports the headline predictor comparison. Four supporting measurements are recorded here.

The provenance set, not the repository, is the action space. Agents mention 50.5 distinct files per episode and actually visit 6.5, a conversion rate of 18.0%. This is the mechanism behind Table 3: the candidate set an agent draws from is two orders of magnitude smaller than the repository and is constructed by the agent’s own earlier outputs. It also explains why the strengthened graph predictor carries a median effective candidate count in the thousands while the provenance predictor carries 3.46 — the graph ranks the repository, whereas provenance ranks what has been seen.

<table><tr><td>Repository</td><td>n</td><td>Localization success</td></tr><tr><td>django</td><td>230</td><td>93.5%</td></tr><tr><td>sympy</td><td>75</td><td>94.7%</td></tr><tr><td>sphinx-doc</td><td>44</td><td>88.6%</td></tr><tr><td>matplotlib</td><td>34</td><td>91.2%</td></tr><tr><td>scikit-learn</td><td>32</td><td>100.0%</td></tr><tr><td>astropy</td><td>22</td><td>100.0%</td></tr><tr><td>pydata</td><td>22</td><td>90.9%</td></tr><tr><td>pytest-dev</td><td>19</td><td>100.0%</td></tr><tr><td> $\mathtt { p y l i n t - d e v }$ </td><td>10</td><td>90.0%</td></tr><tr><td>psf</td><td>8</td><td>100.0%</td></tr><tr><td>mwaskom</td><td>2</td><td>100.0%</td></tr><tr><td>pallets</td><td>1</td><td>100.0%</td></tr><tr><td>Total</td><td>499</td><td>93.99%</td></tr></table>

Table 10: Localization success by repository under the criterion $F ^ { \star } \subseteq { \hat { F } } .$ The corpus is dominated by django at 46% of instances, which is why leave-one-repository-out is the evaluation protocol for v (§C.7) and why repository-conditional calibration is used for conformal sets (§C.8). Higher is better.

Using the full history rather than the last output is not free. Reading the entire observation history rather than only the most recent output adds +0.011 (CI [+0.005, +0.019]) at top-3 while inflating effective candidates from 3.46 to 18.8. The last-output variant is therefore the one we report as headline, since it is both sharper and cheaper; the comparison is what rules out the reading that provenance works merely by accumulating a large candidate pool.

Sample-size expansion from dropping the code graph. Provenance requires only string normalization, because agent commands and shell outputs draw paths from the same vocabulary. Dropping the code graph therefore raises the evaluable sample from 509 decision points to 7616 points over 488 episodes, out of 491 episodes usable without a graph. We verified that this expansion does not shift conclusions: on the overlapping graph-built subset the headline metric differs by under 20%, and the residual gap is explained by object-type composition rather than by normalization. The strengthened graph predictor’s top-3 on first-visit steps is 0.058 (CI [0.039, 0.078]).

Tool and object type are dependent, and the routed model stays calibrated. The mutual information between tool identity and object type is 0.159 nats against a permutation null of 0.006 $( p = 5 \times 1 0 ^ { - 4 } )$ , which is what makes the factored $P ( \mathrm { t o o l } ) P ( \mathrm { o } \tilde { \mathrm { b } } \mathrm { j } )$ model misspecified rather than merely suboptimal. Routing does not buy accuracy at the cost of calibration: the routed model retains ECE = 0.012 (Table 1).

## C.3 ADDITIONAL MEASUREMENTS BEHIND THE GRAPH AND REUSE RESULTS

Raw rates behind the permutation lifts. The one-hop lift of 0.94 reported in §4.3 is the ratio of an observed adjacent-step one-hop rate of 0.284 to a within-episode permutation expectation of 0.301, with immediate self-repeats removed. We report the raw pair because the lift alone can be read as evidence that adjacency is rare, which it is not: roughly 28% of consecutive steps are one hop apart, and the point is that a null which preserves the visited multiset predicts the same rate. This is the sense in which the graph describes the region rather than the movement.

Leakage control in the gold-ranking measurement. 65% of gold files have already been visited by the time the ranking is computed, so leaving them in the candidate set would restate the agent’s own history rather than predict anything. The leakage-controlled figures of §4.3 are computed on unvisited gold files only, which is why they rest on n = 39 (unseen) and n = 54 (prefix-seeded) instances — small samples that we prefer to a larger but contaminated one.

Sample for the reuse analysis. The reuse statistics of §4.4 are computed over 487 episodes. The permutation null for the reuse rate has standard deviation exactly zero in our data, which is the for mal sense in which that rate carries no information: the number of first occurrences among positions 2..n is exactly #distinct − 1, so the reuse rate is a function of the visited multiset and is invariant to order.

## C.4 THE QUERY CHANNEL AND A MEASURED PARSER DEFECT

§4.2 reports that steps taking a novel free-text search query are predicted at top-3 = 0.029, and treats that as the binding constraint on next-action prediction. The number deserves scrutiny, because our command parser recovers the search pattern via shlex, which fails on 26.6% of command segments — an unbalanced quote such as ${ \mathfrak { g r e p } } \ { \mathfrak { n } } { \mathfrak { a } } \setminus { \mathfrak { b } } ^ { * } - -$ and then falls back to splitting on whitespace, retaining the quote character and truncating. 75.3% of query objects are affected, and 249 of them collapse onto the single fragment $\operatorname { q } : " \operatorname { d e f }$ . We therefore re-extracted the patterns with a parser that does not depend on shlex succeeding, and re-ran the channel over 1661 query steps.

The repair moves the number down, not up: those frequent fragments were a spurious repetition mode, and removing them costs -0.036 in top-3 (CI [−0.049, −0.022]), taking the same model from 0.067 to 0.031. Improving the model then recovers ground. Weighting history by frequency rather than recency helps marginally, and adding identifiers mined from the issue text helps substantially (+0.041, CI [+0.029, +0.054], with coverage rising from 0.060 to 0.181), while adding symbol names harvested from prior tool outputs actively hurts (-0.007). Our best configuration reaches top-3 = 0.076 at coverage 0.181, and the net change from the original pipeline is not significant $( + 0 . 0 0 2 , \mathbf { C I } \left[ - 0 . 0 1 6 , + 0 . 0 1 9 \right] )$ .

Three consequences follow. The conclusion that queries bound next-action prediction survives, and is now the outcome of a deliberate attempt rather than of a weak baseline. The defect biases in the conservative direction for that claim — it made queries look more predictable than they are — so it does not threaten it. And there is a positive mechanism worth recording, which contrasts instructively with the path channel of §4.2: an agent’s paths come from what it just read in an output, whereas its query terms come from the issue text, and importing symbol names from outputs into the query model only dilutes it. We report the defect rather than silently fixing it because its direction is what licenses the claim to stand.

## C.5 A BINARY CRITERION THAT MANUFACTURED AN EXCEPTION

An earlier pass over the data of §4.5 thresholded the per-point test at $p < 0 . 0 5$ and found that highentropy steps carried a nominally higher rate of significant effects, which would have contradicted that section’s null. It does not replicate. The stratified difference is +0.056 with $\mathrm { C I } \left[ - 0 . 0 0 8 , + 0 . 1 3 5 \right] ,$ and replacing the binary indicator with either $\left| \Delta _ { t } \right| ( \rho = + 0 . 0 0 2 ) \mathrm { o r } - \log _ { 1 0 } p$ leaves no relationship at all. The diagnostic is the threshold sweep: as α is relaxed and the event count grows from 8 to 24, the ratio decays monotonically from 3.36 to 1.71 — the signature of a small-sample artefact rather than of an effect. We record it because dichotomising a continuous quantity produces clean-looking strata at small n, and the continuous version plus a threshold sweep is what distinguishes the two.

## C.6 UNCERTAINTY COMPONENTS AND THEIR ESTIMATORS

<table><tr><td>Component</td><td>Type</td><td>Estimated by</td><td>Implied action</td></tr><tr><td> $U _ { \mathrm { t a s k } }$ </td><td>aleatoric, irreducible</td><td>residual; graph ambiguity covariates</td><td>ask the user to disambiguate</td></tr><tr><td> $U _ { \mathrm { p o l i c y } }$ </td><td>aleatoric, reducible</td><td>between-rollout variance at fixed q</td><td>resample and aggregate</td></tr><tr><td> $U _ { \mathrm { m o d e l } }$ </td><td>epistemic</td><td>posterior variance through the link</td><td>abstain, escalate, collect data</td></tr></table>

Table 11: The three components of Eq. equation 16. Each answers a different question, so conflating them discards the decision-relevant content: a system that reports only total uncertainty cannot distinguish “the request is ambiguous, ask the user” from “the model is out of its depth, escalate”.

Status on our data. We state this plainly, because it is weaker than the derivation suggests. Only $U _ { \mathrm { p o l i c y } }$ is estimated directly, at 0.049 from the 23 instances carrying K = 4 repeated rollouts. The three-way separation is validated on a synthetic generating process with known truth, not on these traces: parameter recovery reaches correlation 0.994, mean absolute probability error is 0.022, and

(b) AUROC 0.481: at chance

(a) ECE 0.015: calibrated

predictive entropy 0.464 splits as 0.308/0.146/0.012 across the three components, with $U _ { \mathrm { m o d e l } }$ inflating from 0.012 in support to 0.100 out of support, a factor of 8.1. Figure 8 shows the self-test together with the forecasting result it does not rescue. The identifiability of the three terms on real traces is therefore not established here, and the falsifiable prediction that additional sampling cannot help where $U _ { \mathrm { t a s k } }$ dominates is not tested, because it presupposes the calibrated $v _ { 0 }$ that $\bar { \ S } { C . \bar { 7 } }$ reports as absent. Table 11 is a specification with a synthetic self-test attached, and should be read as such.

(c) epistemic term responds

![](images/f7c56544b8b44e82892b8a4afb7b2de04d56977f0f1babb06748fcae4501c68c.jpg)  
Figure 8: The uncertainty decomposition, validated synthetically and unvalidated empirically. The panels report the forecasting behaviour of the left endpoint $v _ { 0 }$ on the real corpus $( n = 4 9 9$ leave-one-repository-out AUROC 0.481, ECE 0.015) alongside the synthetic self-test in which the generating process is known. On the real corpus the risk–coverage curve is essentially flat at the base error rate, which is the signature of a calibrated but uninformative predictor and is why we report AURC and AUROC rather than ECE alone. On the synthetic process the decomposition behaves as specified, with epistemic uncertainty inflating 8.1× outside the training support. The contrast is the point: the machinery works where the truth is known and does not deliver a usable forecast where it is not.

## C.7 ESTIMATING THE LEFT ENDPOINT v<sub>0</sub>

Our corpus has strong group structure by repository and a high base rate, which is a small-data regime. We therefore fit $v _ { 0 }$ as a Bayesian logistic model with a Gaussian prior $w \sim \mathcal { N } ( 0 , I )$ (precision $\lambda = 1 )$ , taking the MAP by IRLS and the posterior covariance Σ by Laplace approximation, then propagating parameter uncertainty by the probit approximation to obtain $U _ { \mathrm { m o d e l } } \approx$ $p ( 1 - p ) x ^ { \top } \Sigma x$ . No hyperparameter search was performed. Evaluation is leave-one-repository-out over 12 repositories. The claims at issue concern calibration and identifiability rather than predictive ceiling, so a small interpretable model with a tractable posterior serves them better than a tuned ensemble whose uncertainty is heuristic.

The measured outcome is negative. Leave-one-repository-out AUROC is 0.481 over 7 covariates, which is at chance against a base rate of 93.99%, with ECE and Brier score as defined in Eq. equation 21 reaching 0.015 and 0.058, and area under the risk–coverage curve 0.058. The low ECE is not reassuring: a high base rate makes calibration easy to satisfy by a nearly constant predictor, and the flat risk–coverage curve confirms that the predictor supplies no usable ordering. We therefore report this as a negative result rather than as a calibrated forecaster, and it is the reason the falsifiable prediction attached to Eq. equation 16 is untested (§C.6). The predictor may read only $q$ and $G ,$ never the trace, since doing so would leak execution-time information.

## C.8 SET-VALUED PREDICTION AND EXCHANGEABILITY

Localization is naturally set-valued, so alongside the scalar $v _ { 0 }$ we report conformal sets $\mathcal { C } ( q )$ satisfying $P ( F ^ { \star } \subseteq { \mathcal { C } } ( q ) ) \geq 1 - \alpha .$ , together with average set size, since a guarantee attained by returning the whole repository is vacuous. Exchangeability requires care rather than assertion in this setting:

traces are sequentially dependent, the observations an agent collects are self-selected by its own earlier actions, and repositories are heavily imbalanced (Table 10). We therefore use split conformal with repository-conditional (Mondrian) calibration rather than pooled calibration, report groupconditional coverage gaps rather than marginal coverage alone, and apply covariate-shift weighting for cross-repository transfer. Marginal coverage that is correct on average while failing badly on the smaller repositories is the failure mode this guards against, and pooled calibration on a corpus that is 46% django would conceal it.

## C.9 ADDITIONAL MEASUREMENTS BEHIND THE ATTRIBUTION RESULTS

§4.6 reports the displacement and enrichment results. The supporting quantities are recorded here, since several of them are what fix the interpretation.

Absolute positions, not just the displacement. The collider contrast moves the verdict from a mean normalized position of 0.214 under the prefix-only condition to 0.751 under the full-trace condition, with a median displacement of +0.667. The absolute positions matter for reading the effect: the prefix-only judge is not merely earlier, it sits in the first quarter of the trajectory, while the full-trace judge sits in the last quarter. For comparison, the intervention-based selector has mean position 0.333. Reporting only the difference would leave open the possibility that both conditions cluster in the same region.

Enrichment interval and the comparable subset. The full-trace judge places 0.900 (CI [0.829, 0.971]) of its top-1 verdicts on gold-provenance steps. The restriction to steps that are directly comparable between the judge and the intervention estimate rests on n = 23 episodes, because only 32.9% of judge verdicts fall inside the estimator’s window set at all. The two baselines are therefore computed separately over their own selection spaces, and the comparable-subset contrast is reported alongside the unrestricted one rather than instead of it.

Cross-model budget and per-judge enrichment. The replication uses one additional generalpurpose model (qwen3.6-plus) and one code-specialised model, at 48 API calls per judge, with the prompt held byte-identical and the trace set fixed. Displacement is positive in all 3 of 3 judges. Per-judge enrichment on gold-provenance steps is 0.826 for the original judge, 0.565 for the second general model and 0.826 for the code-specialised one, against the common-episode baseline of 0.443 (Table 5). We note but do not claim the pattern that the code-specialised model behaves like the original rather than like its own general-purpose sibling: 3 judges cannot separate that explanation from model scale, training data or rubric sensitivity, so we record it as a hypothesis.

Attribution subsample composition. Attribution is computed over 70 episodes drawn from 10 repositories, of which 19 are failing episodes. The mean replay variance across estimation segments is 0.034, and the mean normalized position of the high-entropy steps identified by the positivity probe is 0.416 — close to the middle of the trajectory, which is the sense in which overlap is not concentrated early.

## C.10 WHY $D _ { 1 }$ IS RETIRED

$D _ { 1 }$ appears in Eq. equation 17 for completeness and because we pre-registered it, but it is not a valid test and we report it nowhere as evidence. A disagreement rate is informative only against a reference whose own selection is reliable. Ours is not: §4.5 shows the identified top-1 does not survive FDR correction (0 of 190 effects), so it behaves as a draw over the $| \mathcal { P } _ { i } | \approx \bar { 5 }$ estimation points while a judge selects among $T _ { i } \approx 2 0$ steps. Two selectors drawing from spaces of such different granularity disagree with probability close to one under the null, which is what we observe (97.1%). The criterion was ill-posed rather than merely conservative, and we flag this rather than quietly dropping the metric, since reporting $D _ { 1 }$ as a positive finding is a mistake another group could repeat.

The same defect propagates to the three statistics of $\operatorname { E q . }$ equation 20. R, Z and $\bar { \Delta }$ are all computed against $\widehat { \Delta }$ , so each inherits its noise, and none can adjudicate between “the judge applies a different but defensible criterion” and “the judge extracts no causal signal”. We retain them in §3.9 as definitions, note that only 54 episodes are non-degenerate and 54.7% of estimation points have $\widehat { \Delta }$ exactly zero, and report them nowhere as evidence. The comparisons that do adjudicate need no causal reference: the provenance-class enrichment and the information-set isolation contrast, both in §4.6.

## C.11 COMPARISON WITH ANNOTATION-BASED PROCESS EVALUATION

Process-level evaluations of coding agents (Anonymous, 2026) and SCAE target different estimands, and the difference determines what each can license.

Estimands. That line of work estimates $p _ { j } = P ( y _ { j } = 1 \mid x _ { j } , c )$ , the probability that a human annotator would mark defect $j$ given trajectory evidence $x _ { j }$ and context $c .$ This is a measurementtheoretic quantity: it is about agreement between a detector and a rater, and it is identified from logged trajectories without any causal assumption, since nothing is intervened upon. SCAE estimates $\mathbb { E } [ { \dot { Y } } \mid \tau _ { 1 : t - 1 } , d o ( A _ { t } { = } a ^ { \dot { \prime } } ) ]$ ], the effect on the terminal outcome of substituting one step. This requires the logging policy’s randomization to be preserved (Proposition 1) and is not identified under context compaction (§A.5).

Calibration targets. The two therefore calibrate against different things. There, calibration means detector-versus-annotator agreement, and a well-calibrated detector is one whose confidence matches the rate at which raters agree. Here, calibration means predicted-versus-realized outcome, and a well-calibrated $v _ { 0 }$ is one whose predicted success probability matches the realized success frequency (§C.7). Neither target substitutes for the other: a detector perfectly aligned with annotators tells us nothing about whether the step it flags changed the outcome, and a perfectly calibrated outcome predictor tells us nothing about whether humans would call any particular step defective.

Why this is not a criticism. That work does not claim causal identification, so the comparison is a delineation of scope rather than an objection. It is, however, the reason we do not adopt annotation agreement as a validation target for attribution: a step can be unanimously judged defective and have zero causal contribution, and a step can look unremarkable to every annotator and be decisive, as §A.7 illustrates.

## D REPRODUCIBILITY, PRE-REGISTRATION, AND FALSIFIABILITY

## D.1 REPRODUCTION COMMANDS AND SENSITIVITY ANALYSES

Commands. Every reported number and every data figure is regenerated by a CPU-only script that fixes its random seed and writes its output to JSON. The uncertainty self-test of §C.6 is reproduced by python3 -m pilot.analyze.uq selftest; the entropy summary of §B.4 by python3 -m pilot.analyze.entropy scan; the localization and forecasting tables by python3 -m pilot.analyze.full report; and the data figures by python3 paper/figs/make figures rq.py and python3 paper/figs/make figures.py. The figure scripts read only the recorded rollout outputs, so plotted and tabulated values cannot drift apart. A further script, python3 paper/analysis/verify macros.py, checks every numeric macro in the manuscript against the analysis output that produced it, and python3 paper/analysis/check tex.py verifies that every reference, label, macro, citation and figure path resolves.

Outcome definition. §4.1 reports the primary criterion $F ^ { \star } \subseteq { \hat { F } }$ alongside exact set equality. The gap between 93.99% and 15.23% is accounted for entirely by the agent naming test files that gold patches exclude: a filtered variant that discards test paths before comparison coincides with the primary criterion to within rounding. We therefore treat exact equality as a diagnostic of the agent’s naming behaviour rather than as a metric of localization ability.

Effect of the leakage reset. Every number is measured after a hard reset of the repository snapshot to its base commit. This is a silent confound rather than a detail: the same pipeline run against the distributed snapshots as shipped would measure an agent that can read the answer out of its own working tree, and would report inflated localization accuracy with no visible symptom.

Measurement error in regression diagnostics. Estimated contributions carry Monte Carlo error, which attenuates regression coefficients toward zero. For the hypothesis that data-flow reachability fails to predict causal effect (§A.7), this bias is favourable to us, so we state its direction explicitly. A correction such as SIMEX, or a larger m, is required before that null is asserted rather than merely observed.

Discrete estimation points. Because $v _ { t }$ is estimated at equidistant points rather than at every step, reported contributions are segment effects and their sum does not satisfy Eq. equation $9$ exactly. We therefore present no additivity check as evidence; a genuine test requires contiguous windows. Equidistant rather than contiguous-tail points was a deliberate choice, because a window concentrated late in the trace would bias the position comparison on which §4.6 depends.

## D.2 PRE-REGISTRATION RECORD

The directions, intervals and falsification thresholds in Table 12 were fixed before any attribution experiment was run. We report the record in full, including the four entries that did not hold and the one criterion we retire rather than bank.
<table><tr><td>Quantity</td><td>Pre-registered</td><td>Outcome</td></tr><tr><td> $D _ { 1 }$  top-1 disagree- ment</td><td>[40%, 75%], point 55%; falsi-  $\mathrm { \dot { f } e d \ i f < 2 0 \% }$ </td><td>criterion retired, not passed: observed  $9 7 . 1 \%$   $( 6 8 / 7 0 ) , \mathrm { C I } [ 9 2 . 9 \% , 1 0 \bar { 0 } \% ]$  . We initially recorded this as “direction confirmed&quot;; that reading is with- drawn, because the null itself sits near 97.1% once</td></tr><tr><td> $D _ { 3 }$  displacement</td><td> $[ + 0 . 2 5 , + 0 . 4 0 ] ;$  falsified  $\mathrm { i f } \le$  0</td><td>the reference is known to be noise  $( \ S C . 1 0 )$  above interval: +0.418 over  $n = 6 8 , z = 6 . 0 7 ,$   $p < 0 . 0 0 1$ </td></tr><tr><td>Collider isolation</td><td> $[ + 0 . 1 5 , + 0 . 3 0 ] ;$  sign is the substantive claim</td><td>above interval:  $+ 0 . 5 3 7$  over  $n = 6 0 , z = 6 . 7 0 ,$   $p \ < \ 0 . 0 0 1$  . Sign confirmed; magnitude implies descendant conditioning accounts for most of the</td></tr><tr><td>Wilcoxon p</td><td>expected to be uncomputable at small n</td><td>displacement superseded: the full attribution sweep makes it computable, and both signed-rank tests reject at  $p < \mathsf { \bar { 0 . 0 0 1 } }$ </td></tr><tr><td>Degenerate  $\Delta \ \equiv \ 0$  rate</td><td> $3 0 { - } 4 0 \%$ </td><td>better than predicted: 22.9%  $( 1 6 / 7 0 ) .$  An early two-episode reading suggested 50%; the full sweep does not support it</td></tr><tr><td> $U _ { \mathrm { p o l i c y } }$  estimable</td><td>requires  $K \geq 2 ;$  not available from the main sweep</td><td>met on the  $K { = } 4$  subset:  $U _ { \mathrm { p o l i c y } } = 0 . 0 4 9$  over 23 instances</td></tr><tr><td>Entropy vs. position</td><td>monotone decay,  $\bar { \rho } \ \leq \ - 0 . 2$  (“early-high, late-low&quot;)</td><td>not supported:  $\rho = + 0 . 0 7 2$  over 10 points; re- ported as non-monotone with episode-dependent inflection</td></tr><tr><td> $v _ { 0 } \mathrm { L O R O } \mathrm { A U R O C }$ </td><td> $\geq 0 . 6 0$  for the forecasting end-</td><td>not met: 0.481, at chance; reported as a negative</td></tr><tr><td> $U _ { \mathrm { m o d e l } } \mathrm { i n f l a t i o n }$ </td><td>point to be useful  $> 2 \times$ </td><td>result (§C.7)</td></tr><tr><td>Replay consistency</td><td>out of training support  $\geq 0 . 9 9$  byte-identical</td><td>met:  $0 . 0 1 2  0 . 1 0 0 , \mathrm { a }$  factor of 8.1 met: 1.00 over 100 actions, zero mismatches</td></tr></table>

Table 12: The pre-registration record and its outcomes. Three directional predictions held at magnitudes above the intervals we fixed in advance, which we report as the intervals having been too conservative rather than as successes. Four entries did not hold as stated: the monotone entropy decay is contradicted, the forecasting endpoint failed its usefulness threshold, the degenerate-outcome rate came in below the predicted range, and the $D _ { 1 }$ criterion was ill-posed.

What this record does not cover. The decisive-step recovery rate R, the mean blamed contribution $\bar { \Delta }$ and the null-blame rate $Z$ of Eq. equation 20 are absent from the table because they were not pre-registered. We introduced them after observing the displacement, to distinguish “the judge applies a different but defensible criterion” from “the judge extracts no causal signal”. We now withdraw that use of them, for the reason given in §C.10. Both comparisons that do adjudicate — provenance-class enrichment and the information-set isolation contrast — were reachable from the same recorded rollouts, so the correction cost no additional agent execution.

The defect the record caught. Our initial design truncated the prefix-only judge just before the step selected by the causal estimate. Because a judge cannot select a step it cannot see, this caps the prefix-only condition’s reachable positions below the comparison point, and would have produced a positive displacement even if conditioning on descendants had no effect at all. Writing the falsification criterion forced us to ask what would happen under the null, which is what surfaced the defect. The corrected design uses a truncation fraction fixed in advance and independent of any estimate (§B.2).

## D.3 WHAT WOULD FALSIFY THE FRAMEWORK

§5 states the limitations that bound how our results should be read. This subsection states the sharper thing: the observations that would undercut the central argument rather than merely bound it.

Three would undercut the framework itself. If prefix replay failed to reproduce agent behaviour — that is, if forking from a reconstructed prefix systematically diverged from the original continuation — then Assumption 1 would be false in practice and the identification result would not apply to real harnesses. If the isolation contrast of §4.6 produced no displacement at scale, the collider-bias explanation ofjudge behaviour would be wrong even though the structural argument stands; this one has now been run, and the displacement is +0.537 with a sign test at $p = 1 . 1 \mathrm { \bar { \times } } 1 0 ^ { - 1 6 }$ . And if action entropy were zero at essentially all decision points, the observational branch of Eq. equation 14 would never apply and the framework would reduce to intervention-based ablation with respect to a perturbed policy.

Three further falsifiers attach to the specific empirical claims. If a code-graph predictor in any form matched execution provenance on first-visit steps, §4.2–§4.3 would collapse; we strengthened the graph predictor deliberately, seeding a multi-source proximity ranking from the whole visited set, and it still loses, but a better graph formulation is the obvious line of attack. If the residualised instance-level ICC of §4.5 fell back to its permutation null, then replay variance would carry no de tectable non-mechanical structure at $m = 8 ,$ , and our null result about structural entropy would revert to a statement about power rather than about the world — which is why we report that ICC before, not after, the null it licenses. And if the judge’s placement on gold-provenance steps regressed to its 0.413 baseline under a different judge or rubric, the characterisation of what such judges measure would not generalise beyond the ones we tested.