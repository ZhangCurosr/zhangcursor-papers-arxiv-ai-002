# Monitor Accuracy Isn’t the Answer: A Decision-Value Framework for Evaluating AI Agent Restart Monitors

Siddharth Jain Intuit

Venkat N Vedam Intuit

## Abstract

Failure monitors for AI agents can be judged by classification accuracy and true-positive and false-positive rates on labeled failures. We show that these metrics alone do not determine what actually matters: whether a restart improves success. Using an exact alarm-to-action identity for an independent clean restart with zero monitoring overhead, we show analytically that a monitor with fixed 90% accuracy can swing from a five-percentage-point loss to a 35-point gain in success rate, depending solely on the replacement attempt’s residual-budget success probability, a quantity accuracy never measures. Building on this, we introduce two techniques for evaluating monitors and progress signals directly in decision terms rather than classification terms. First, a decision-suficiency criterion, a binary-action specialization of Blackwell’s comparison of experiments, identifies exactly which progress representations or context compressions preserve the information needed to choose optimally between the specified actions, independent of how faithfully they reconstruct the trace. Second, a finite-sample, model-free decision audit uses simultaneous binomial intervals over existing agent traces, without additional model calls, to certify a preference for “restart” or “continue” under stated sampling assumptions, or return an explicit “abstain” when evidence is insuficient, instead of reporting an unqualified point estimate. Applied to a study of 194 externally evaluated model executions on 19 selected SymPy issues from SWE-bench Verified, with a retrospective audit of 130 of these traces, this approach exposes what accuracy-based evaluation alone cannot establish: a 2.29-point apparent restart gain traced to a single surviving failed attempt, a 42.29-point loss on a second model, no positive replication of the initial gain on prospective task cohorts, and zero of 103 live comparisons meeting the specified certification bar. Together, these results motivate evaluating agent monitors through recovery opportunity, decision value, and statistical support, rather than classification performance alone.

## 1 Introduction

An agent that is unlikely to finish may also be unlikely to succeed after a restart. This distinction matters whenever a harness acts on a failure prediction: abandoning a trajectory removes its remaining chance of success, and the replacement receives only the residual budget. A monitor can therefore classify failures accurately while inducing harmful decisions. Conversely, a modest signal may be useful when it resolves a small number of decisions with large consequences.

The problem has a substantial intellectual history. Dynamic restart policies use observations of a solver’s activity to update runtime beliefs [1]; learned schedules address distributions of problem instances [2]; Bayesian early stopping derives policies from observed run features [3]. Recent agent systems demonstrate useful termination and recovery, including EET [4] and FailFast-RestartSmart [5]. Our contribution is not the premise that progress can inform stopping. We ask how to make a claimed progress signal’s decision value and evidential support explicit under a finite interaction budget.

Three questions organize the study:

## RQ1

Which quantities determine whether a failure alarm improves success after a restart?

## RQ2

What information must a progress representation retain, and when do available traces justify a continuation or restart decision?

## RQ3

How do these distinctions explain observed outcomes of restart, horizon extension, context recovery, and failed-branch reuse?

We make the intervention precise. For a live attempt, continuation retains its state and spends the remaining budget; a clean restart restores the original task state and spends the same budget on a fresh attempt. A context reset that retains edits is a separate action, with a separate outcome law. Under this definition, we derive an exact relation between alarm true-positive rate, false-positive rate, continuation success, and restartability. An analytical control keeps failure classification accuracy at 90% while changing restart gain from −5 to +35 points solely by changing the success probability of the replacement. This is a controlled mathematical example, not a performance claim about an implemented detector.

The information question follows naturally. Progress need not be a monotone scalar. It can be a representation that preserves distinctions relevant to the action. We give a binary-action specialization of classical information comparison [6]: compressing a history loses decision value when the compressed observation combines positive and negative continuation advantages. Exact decision suficiency is an ideal benchmark; an imperfect representation can still improve utility. We then implement a conservative audit that exposes the statistical support for each action, including an abstention when the evidence does not resolve the sign.

The empirical evidence consists of 194 model-driven executions of mini-swe-agent on 19 selected SymPy issues in SWE-bench Verified [7, 8]. Each attempt begins from the benchmark-provided repository state. The reference patch and evaluator remain outside the sandbox, and external verification after each environment action records the first passing state. Earlier experiments comprise restart replay, prospective task transfer, additional horizon, conversational recovery, and two bounded uses of failed branches. We retain the negative results and distinguish their design status: later collection stages used prospectively frozen protocols, while the decision audit introduced here is retrospective.

The audit sharpens the interpretation of the original results. Across 130 executions with a 30-call horizon, the 16-call Luna gain comes from a single eligible failed trajectory on one task. On the second model, every task with a live attempt at that decision has an empirical advantage for continuation. Across six decision times, all 103 comparisons with live data remain unresolved by the simultaneous certificate. These findings distinguish visible efect size from evidence suficient for individual deployment decisions. They do not demonstrate that every alternative statistical method must abstain.

## Contributions.

1. An intervention-specific account of agent progress, including an alarm-to-action value identity and an information-loss criterion for progress representations and context compression.

2. A dependency-free audit that estimates both action values, respects live eligibility and observed horizons, and returns simultaneous sign certificates or abstention under explicit assumptions.

3. A reproducible retrospective analysis identifying the support of the restart efect, together with analytical controls that separate detector accuracy, restartability, and sample resolution.

4. A sequence of externally verified agent experiments showing model-dependent restart efects and the distinction between protocol recovery and verified task completion.

The classical decision identities are attributed and used as foundations. The empirical and software contributions concern how those identities can discipline evaluation of current agent controllers. The study does not train a monitor, directly compare against learned recovery systems, or establish performance outside its selected tasks and runtime configurations.

## 2 Related Work

## 2.1 Restarts and inference-time allocation

Restart policies are well studied for randomized algorithms. Luby, Sinclair, and Zuckerman give a logarithmic expected-runtime comparison with a distribution-specific fixed strategy under their assumptions [9]. This is not a guarantee that a chosen finite-budget truncation improves success. Dynamic restart policies already condition on solver observations and runtime beliefs [1]. Schedule learning across instances [2] and Bayesian stopping from run features [3] directly precede our formulation. The no-dominance and observation-ambiguity examples below are elementary specializations; novelty is not claimed for feature-dependent restarting or the need to calibrate runtime distributions.

Our decision-value expression specializes Blackwell’s comparison of experiments [6] to two actions. The abstaining audit applies exact binomial intervals [10] and simultaneous comparison in the spirit of safe policy improvement [11]. These foundations identify the scope of the contribution: an explicit connection between alarm quality, intervention-specific restartability, and the statistical support of agent decisions, implemented on externally verified traces.

Inference-time scaling allocates additional computation through sampling, voting, or search. Self-consistency samples reasoning paths and aggregates answers [12]; Tree of Thoughts searches over explicit intermediate candidates [13]; inference scaling studies compare compute-allocation strategies under fixed budgets [14]. Mixture-of-Agents aggregates outputs across model instances and layers [15]. These methods can exploit independent or weakly correlated trials and informative scoring. Classical ensemble analysis shows why both accuracy and error diversity matter [16, 17]. Our setting removes a central advantage: the controller receives no reliable scalar score before external task completion. We study whether schedule choice or failure-conditioned branching alone can recover that missing information.

Iterative self-improvement methods use generated feedback to revise an answer or policy [18, 19]. Other work shows that intrinsic self-correction can fail when the feedback does not add external information [20]. Our context-rejuvenation and semantic-resolver experiments test a related distinction at the harness level: changing the context or synthesizing failed branches can change behavior without improving the externally verified outcome.

## 2.2 Agent progress and externally grounded evaluation

The closest contemporary system is FailFast-RestartSmart [5]. It trains a prefix monitor and combines alarms with a fresh rollout that can inspect or apply previous edits. The paper reports both cost savings and recovery gains, providing positive evidence that richer learned signals and interventions can work. EET also demonstrates useful termination from prior execution experience [4]. Our experiments do not reproduce or compare against either system. A timeout is a diferent controller, and our mandatory workspace retention difers from an optional edit overlay. Our audit complements such systems by requiring the post-alarm action and remaining-budget success to be evaluated jointly with failure prediction.

AgentBoard introduced a progress-rate metric to complement final success across multi-turn environments [21]. Such metrics are valuable when an environment exposes meaningful subgoals. Park and Choi show that an agent can report improvement while externally measured state does not improve, and argue for evaluation grounded outside the transcript [22]. We take external grounding as a design constraint and ask what remains identifiable when the grounded signal is binary completion rather than a dense progress measure.

Semantic early stopping detects convergence between successive drafts and can reduce tokens in iterative question answering at comparable measured quality [23]. Geometric analyses represent iterative outputs as trajectories, attractors, or oscillations in an embedding space [24]. These representations describe change in semantic space, but geometric convergence is not automatically decision suficient for externally verified completion. VLAA-GUI uses visual completeness checks and loop breaking in GUI automation [25]. These systems have richer task structure than our passor-fail coding controller. Their results are consistent with our boundary condition: useful stopping and recovery require a signal calibrated to the relevant external state, not elapsed calls or semantic movement alone.

Long-horizon software agents have also motivated active context management. Context as a Tool lets an agent compress history into structured memory [26], while later work learns or implements agent-compatible context policies [27, 28]. Our recovery intervention is deliberately simpler. It discards conversational history after a protocol terminal event, retains the exact workspace, and supplies a deterministic state summary. Its null result does not evaluate learned context management. It isolates whether stale conversation alone explains observed failure in one fixed harness.

## 2.3 Software engineering agents and reproducible evaluation

SWE-bench evaluates repository-level issue resolution against executable tests [7]. SWE-agent established the efectiveness of purpose-built agent-computer interfaces for these tasks [8]. We use the compact mini-swe-agent implementation as a versioned, open-source harness. We do not compare agent architectures or report a leaderboard estimate. Selected SymPy tasks provide controlled, expensive trajectories with externally testable completion.

Agent evaluation is sensitive to cost, holdout design, and reproducibility [29]. Test-suite weakness can also overstate issue resolution [30]. We therefore run the benchmark evaluator outside the agent workspace after each environment action, gate tasks by base-fail and reference-pass checks, audit every successful dif for verifier tampering, and select cutofs with task-level holdout rather than attempt-level splitting.

## 2.4 Position of this work

Prior work includes both successful stopping systems and formal restart optimization. This study contributes an evaluation and evidence audit for a specified action pair. The distinction is summarized in table 1; the negative screens do not invalidate richer systems with diferent observations, training, or recovery semantics.

Table 1: The object measured by adjacent lines of work. A task-grounded state can support progress decisions; a trace-only controller must infer it.
<table><tr><td>Line of work</td><td>Observed quantity</td><td>Primary question</td></tr><tr><td>Inference-time search</td><td>Candidate scores or votes</td><td>Which candidate should receive compute?</td></tr><tr><td>Dynamic restart theory</td><td>Run features and runtime beliefs</td><td>Which restart policy minimizes expected runtime?</td></tr><tr><td>Learned termination and recovery</td><td>Prefix alarms and retained edits</td><td>Does a trained controller save cost or improve resolution?</td></tr><tr><td>Semantic stopping</td><td>Change between successive outputs</td><td>Has an iterative answer stabilized?</td></tr><tr><td>Trajectory evaluation</td><td>Task-specific subgoal state</td><td>How much of the task is complete?</td></tr><tr><td>Context management</td><td>Structured or compressed history</td><td>What information should the agent retain?</td></tr><tr><td>This work</td><td>Action values, live eligibility, sampling uncertainty</td><td>Does the intervention have decision value, and is its sign supported?</td></tr></table>

## 3 Black-Box Progress as a Decision Problem

## 3.1 Setup

Fix a task, agent, model, provider policy, prompt, and tool interface. One attempt induces latent states $X _ { 0 } , X _ { 1 } , \ldots$ and observations $O _ { 1 } , O _ { 2 } , \dots$ . This is a partially observed control process [31]. The controller’s information after call t is the filtration

$$
\mathcal { F } _ { t } = \sigma ( O _ { 1 } , \ldots , O _ { t } ) ,\tag{1}
$$

where observations may include actions, tool outputs, workspace summaries, external verifier results, and explicit protocol failures. Let

$$
T = \operatorname* { i n f } \{ t \geq 1 : V ( X _ { t } ) = 1 \}\tag{2}
$$

be the first call whose resulting state passes the external verifier, with $T = \infty$ if no such state is reached. The belief state $b _ { t } ( d x ) = \mathbb { P } ( X _ { t } \in d x \ | \ \mathcal { F } _ { t } )$ is suficient for optimal control when the generative process is known. A black-box harness does not know this process and must approximate decision value from its observations. A budget policy π maps observable histories to continuation, restart, recovery, branch, or termination actions under total call budget B.

We use semantic progress for a latent property of state that predicts future verified completion. This term does not presume that progress is monotone or one dimensional. Let $Q _ { a } ( h _ { t } , b )$ denote expected utility when action a is selected after history $h _ { t }$ with b calls remaining. For the binary continue-or-restart problem, define the continuation advantage

$$
\Delta ( h _ { t } , b ) = Q _ { \mathrm { c o n t } } ( h _ { t } , b ) - Q _ { \mathrm { r e s t a r t } } ( h _ { t } , b ) .\tag{3}
$$

The utility may combine externally verified success, calls, money, and latency, but it must be declared before evaluating the policy.

Definition 3.1 (Decision-suficient progress representation) An observable representation $G _ { t } = g ( O _ { 1 : t } )$ is decision suficient for a policy class and runtime population if the optimal action is measurable with respect to $( G _ { t } , b )$ . For binary control, this requires the sign of equation (3) to be determined by $( G _ { t } , b )$ almost surely.

![](images/88f947fae20a35b2f3a97786322a051afe806fedfede46d1e468dfbd6b3e965e.jpg)  
Figure 1: Observational aliasing in black-box control. Distinct latent states can induce the same observable history while requiring opposite actions. A useful progress sensor must separate histories according to continuation advantage, not surface activity.

This definition describes lossless action selection relative to the richer history. A representation can be useful without attaining exact suficiency. Embedding displacement, code churn, test counts, and fluent self-reports may describe a trajectory while discarding a distinction that changes the preferred action. Two histories may be close in a semantic space yet have diferent continuation advantages, or look dissimilar while implying the same action.

Proposition 3.2 (Progress error is advantage-weighted decision error) Let $\begin{array} { r l } { d ( h _ { t } , b ) } & { { } \in } \end{array}$ {cont, restart} be any stopping rule and let $d ^ { * } ( h _ { t } , b )$ select the action with larger conditional value. Its expected regret within the binary action class is

$$
\mathcal { R } ( d ) = \mathbb { E } [ | \Delta ( H _ { t } , b ) | \mathbb { I } \{ d ( H _ { t } , b ) \neq d ^ { * } ( H _ { t } , b ) \} ] .\tag{4}
$$

Proof. Conditioned on a history, choosing the inferior action loses exactly the absolute diference between the two action values; choosing the superior action loses zero. Taking expectation gives equation (4). □

The identity shows why label accuracy is not the right primary metric for a progress detector. Errors matter in proportion to the value diference between continuing and restarting. A detector can achieve high accuracy on histories where the actions are nearly tied and still lose most of the attainable utility on the few histories where |∆| is large.

## 3.2 Fixed restarts depend on the completion scale

Suppose fresh attempts are independent and identically distributed, restart overhead is zero, and a live attempt is truncated at cutof c. Write $F ( t ) = \mathbb { P } ( T \leq t )$ and decompose the total budget as $B = q c + r$ , where $q = \lfloor B / c \rfloor$ and $0 \leq r < c$ . If unused budget is assigned to a final residual episode, the success probability is

$$
P _ { c } ( B ) = 1 - ( 1 - F ( c ) ) ^ { q } \left( 1 - F ( r ) \right) ,\tag{5}
$$

with $F ( 0 ) = 0$ . This expression assumes that noncompletion consumes the assigned episode budget. An explicit terminal event can end an episode earlier and release additional budget, which is handled by trace-level replay in the experiments. An uninterrupted attempt succeeds with probability $F ( B )$

The discrete completion hazard makes the missing scale explicit. Let

$$
\lambda _ { u } = \mathbb { P } ( T = u \mid T \geq u ) .\tag{6}
$$

With r calls remaining and no observation beyond survival through t, continuation succeeds with probability

$$
C ( t , r ) = 1 - \prod _ { u = t + 1 } ^ { t + r } ( 1 - \lambda _ { u } ) ,\tag{7}
$$

whereas a fresh attempt succeeds with probability $R ( r ) = F ( r )$ . Under equal success utility and zero reset cost, restart is preferred exactly when $R ( r ) > C ( t , r )$ . Thus elapsed time helps only through assumptions about the hazard. A decreasing hazard favors restarts; delayed completion mass can favor continuation. With trace features $G _ { t }$ , the relevant object is the feature-conditiona residual law, not the marginal curve alone.

Proposition 3.3 (No uniformly dominant proper cutof) For any $B \geq 2$ and any fixed cutof $1 \leq c < B$ , there exists a completion-time distribution for which the uninterrupted policy strictly dominates cutof c. There also exists a completion-time distribution for which cutof c strictly dominates the uninterrupted policy whenever $\lfloor B / c \rfloor \geq 2$

Proof. For the first claim, place all probability mass at $T = c + 1 \leq B$ . Then $F ( c ) = 0$ and $F ( B ) = 1$ . For the second, let $T = c$ with probability $p \in ( 0 , 1 )$ and let $T = \infty$ otherwise. Then $F ( c ) = F ( B ) = p , F ( r ) = 0$ , and equation (5) equals $1 - ( 1 - p ) ^ { q } > p$ for $q = \lfloor B / c \rfloor \geq 2$ □

The proposition is elementary but constraining. Any positive universal-restart claim must introduce assumptions on the completion distribution, a prior over task-agent tuples, or an observation that predicts the residual completion time. Schedule shape cannot supply this information.

Corollary 3.4 (Finite-scale sensitivity) Consider any finite restart schedule whose largest episode has length $s < B$ . There is a distribution supported on completion times in $( s , B ]$ for which the schedule has zero success and an uninterrupted attempt succeeds with positive probability.

This corollary does not contradict the classical Luby guarantee [9]. It identifies the finite-budget role of the base scale. A universal ordering of episode lengths is not a universal calibration of model calls.

## 3.3 The continuation decision is not identified by survival

At call c, suppose the only task-grounded event is that no verified state has appeared. The remaining budget is enough for either one more call on the current attempt or one call on a fresh attempt.

Proposition 3.5 (Observational ambiguity) There are two latent completion regimes that induce the same observable no-success history through call c but require opposite optimal decisions. In one regime continuation succeeds with probability one and restart fails. In the other, conditional continuation fails while restart has positive success probability. Consequently, no rule measurable with respect to that history is optimal in both regimes.

Proof. In the late regime, let the current attempt complete deterministically at $c + 1$ and let a fresh attempt require more than one call. Continuation is uniquely optimal. In the fast-or-never regime, an attempt completes on its first call with probability $p > 0$ and never completes otherwise. Conditional on survival through $^ { c , }$ the current attempt is in the never component, so continuation fails and a fresh attempt succeeds with probability $p .$ Both regimes can emit the same observation sequence before external completion. A deterministic rule chooses the wrong action in one regime. $\mathrm { A }$ randomized rule cannot attain the regime-specific optimum in both. □

The result is about identification, not metaphysical opacity. A task-specific sensor may distinguish the regimes. Absent such a sensor or a calibrated prior, a controller that restarts because the trace looks stagnant imports an untested mapping from appearance to residual value.

Corollary 3.6 (Regret under observational aliasing) In the construction of proposition $3 . 5 ,$ suppose a fresh attempt in the fast-or-never regime succeeds with probability $p \in ( 0 , 1 )$ . For any randomized policy that observes only the aliased history, the worst-case regret over the two regimes is at least $p / ( 1 + p )$ . The bound approaches $1 / 2$ as p approaches one.

Proof. Let the policy continue with probability α. Its regret is $1 - \alpha$ in the late regime and pα in the fast-or-never regime. Minimizing max $\{ 1 - \alpha , p \alpha \}$ gives $\alpha = 1 / ( 1 + p )$ and regret $p / ( 1 + p )$ . □

The corollary gives an operational meaning to unobservability. It is not merely that a trace classifier may be inaccurate. Without an identifying assumption, error remains even with unlimited computation applied to the same observable history.

An additional failure-selection example and the standard correlated-ensemble variance identity appear in Appendix G. They motivate the branch-reuse screens but are not empirical estimates of our patch population.

## 4 From Failure Prediction to Restart Decisions

The useful question is whether a signal changes expected utility under a specified intervention. This section connects the classical value of information [6] to a finite-budget audit that can be computed without model access. The identities specialize existing decision theory; the contribution is their operationalization for agent termination and recovery, together with an explicit account of what the available traces can identify.

## 4.1 Failure prediction must be combined with restartability

Fix one task and runtime, a live decision at call $t ,$ and residual budget $b = B - t$ . Let $Y _ { C }$ be the potential binary outcome of uninterrupted continuation to the deadline. Let $q = \mathbb { P } ( Y _ { C } = 1 )$ , and let $p$ be the success probability of one independent clean restart with the same residual budget. We call $p$ the restartability of this decision. An alarm $A = 1$ triggers the restart. Define its failure true-positive rate $\tau = \mathbb { P } ( A = 1 \mid Y _ { C } = 0 )$ and false-positive rate $\phi = \mathbb { P } ( A = 1 \mid Y _ { C } = 1 )$

Proposition 4.1 (Alarm-to-action value identity) Assume the fresh outcome is independent of $( A , Y _ { C } )$ conditional on the fixed task and runtime, the restart restores the initial state, and the alarm consumes no budget. The success change relative to continuing every live attempt is

$$
G ( q , p , \tau , \phi ) = ( 1 - q ) \tau p - q \phi ( 1 - p ) .\tag{8}
$$

$I f \left( 1 - q \right) \tau + q \phi > 0$ , a positive gain requires

$$
p > \frac { q \phi } { ( 1 - q ) \tau + q \phi } .\tag{9}
$$

Proof. On correctly alarmed failures, the restart changes success from zero to $p ,$ contributing $( 1 - q ) \tau p$ . On falsely alarmed successes, it changes success from one to $p ,$ contributing $- q \phi ( 1 - p )$ . Unalarmed outcomes are unchanged. Rearrangement gives the threshold. □

For $q = 1 / 2 , \tau = 0 . 9$ , and $\phi = 0 . 1$ , the detector’s classification accuracy is 90% for every $p ,$ while $G = 0 . 5 p - 0 . 0 5$ . Thus identical alarm quality gives a five-point loss at $p = 0$ , no change at $p = 0 . 1$ and a 35-point gain at $p = 0 . 8$ . Even perfect failure prediction produces no additional successes when $p = 0$ , although early termination can still save cost. These are analytical controls, not measured monitor performance. They motivate joint reporting of alarm discrimination, residualbudget restartability, and realized utility. A global accuracy or AUROC alone omits the second quantity.

An edit-preserving restart changes the intervention and need not satisfy the independence assumption. For a general alarm, the exact identity is $G = \mathbb { E } [ A ( Y _ { R } - Y _ { C } ) ]$ . Computing it requires outcomes for the specified recovery action. Mixing task populations can also induce dependence, even when fresh draws are independent within each task. We therefore estimate and interpret action values within task and runtime before aggregation.

Corollary 4.2 (Opportunity ceiling for a failure alarm) Under the assumptions of proposition $4 \cdot 1 ,$ any alarm satisfies

$$
G \leq ( 1 - q ) p .\tag{10}
$$

Equality is attained by a perfect alarm, $A = 1 - Y _ { C }$ . The additional value of this clairvoyant alarm over the better constant action is

$$
( 1 - q ) p - ( p - q ) _ { + } = \operatorname* { m i n } \{ ( 1 - q ) p , \ q ( 1 - p ) \} .\tag{11}
$$

Proof. In equation (8), use $\tau \leq 1$ and $\phi \geq 0$ . Subtract the better constant-action gain $( p - q ) _ { + }$ and distinguish $p \geq q$ from $p < q$ □

The ceiling is a value-of-perfect-information benchmark for one specified action pair. It is neither a realizable detector nor an upper bound for state-preserving recovery, multiple restarts, or another model. A plug-in estimate of this ceiling can diagnose an evaluation set with little measured opportunity; it does not establish that the true opportunity is small without sampling uncertainty.

## 4.2 The decision value of a richer progress signal

Let Z be a richer observation and let $D ( Z ) = Q _ { R } ( Z ) - Q _ { C } ( Z ) = - \Delta ( Z )$ denote restart advantage. A coarser representation is $G = g ( Z )$ . Fix the residual budget and population of live decisions. The best policy using G has value

$$
V ( G ) = \operatorname { \mathbb { E } } [ Q _ { C } ( Z ) ] + \operatorname { \mathbb { E } } \left[ ( \operatorname { \mathbb { E } } [ D ( Z ) \mid G ] ) _ { + } \right] ,\tag{12}
$$

where $( x ) _ { + } = \operatorname* { m a x } ( x , 0 )$

Proposition 4.3 (Value lost by compressing a progress signal) The loss from replacing Z by $G = g ( Z )$ is

$$
V ( Z ) - V ( G ) = \frac { 1 } { 2 } \left\{ \mathbb { E } | D ( Z ) | - \mathbb { E } | \mathbb { E } [ D ( Z ) \mid G ] | \right\} \geq 0 .\tag{13}
$$

It vanishes when each coarse observation contains advantages of only one sign, up to zero-probability events. In particular, preserving the optimal action can preserve decision value without reconstructing the original trace.

Proof. Using $x _ { + } = ( x + | x | ) / 2$ in both values cancels the common mean advantage. Conditional Jensen’s inequality proves nonnegativity. Equality in the absolute-value inequality holds exactly when positive and negative advantages do not coexist within a conditional observation class. □

This gives a task-specific criterion for context compression: preserve distinctions that change the preferred action. Token retention, embedding fidelity, and reconstruction quality do not by themselves measure this loss. The identity is a binary-action specialization of classical information comparison, not a new universal information theorem. It also permits imperfect signals to have positive value; exact decision suficiency is an ideal reference, not a prerequisite for practical improvement. If acquiring Z reduces expected utility by κ while otherwise preserving the actionvalue model, its net benefit is $V ( Z ) - V ( G ) - \kappa$ . When acquisition consumes calls or changes the workspace, action values must instead be evaluated at the resulting budget and state.

## 4.3 A finite-sample decision certificate

Let E denote explicit protocol termination. At a fixed t, a trace is eligible only if it is still live:

$$
L _ { t } = \{ T > t , \ E > t \} .\tag{14}
$$

For task $j ,$ the audited comparison is uninterrupted continuation through B versus one clean restart with $B - t$ calls:

$$
q _ { j , t } = \mathbb { P } ( T \leq B \mid L _ { t } , j ) , \qquad p _ { j , t } = \mathbb { P } ( T \leq B - t \mid j ) .\tag{15}
$$

The first quantity is estimated from eligible attempts; the second from all fresh attempts on that task. A later protocol terminal counts as unsuccessful continuation. An earlier terminal is outside the live-decision population. Neither arm extrapolates beyond the observed horizon.

For a fixed family of M task-time comparisons, form two-sided Clopper-Pearson intervals [10] for every $q _ { j , t }$ and $p _ { j , t }$ , each with error probability $\alpha / ( 2 M )$ . Write them as $[ L _ { C } , U _ { C } ]$ and $[ L _ { R } , U _ { R } ]$ An empty eligible sample receives [0, 1]. The decision audit returns

$$
\mathcal { D } _ { j , t } = \left\{ \begin{array} { l l } { \mathrm { r e s t a r t , } } & { L _ { R } > U _ { C } , } \\ { \mathrm { c o n t i n u e , } } & { U _ { R } < L _ { C } , } \\ { \mathrm { a b s t a i n , } } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{16}
$$

Proposition 4.4 (Simultaneous sign certification) For IID attempts within each fixed task and runtime, complete observation through B, and a comparison family fixed independently of the evaluated outcomes, every nonabstaining decision in equation (16) has the correct sign with probability at least $1 - \alpha$

Proof. Each binomial interval has its stated marginal coverage, including the continuation interval conditional on the number of eligible attempts. A union bound over the 2M intervals gives simul taneous coverage. On that event, $p _ { j , t } - q _ { j , t } \in [ L _ { R } - U _ { C } , U _ { R } - L _ { C } ]$ , so either strict separation certifies the sign. Independence between the two estimated arms is unnecessary for the union bound; the samples may overlap. □

This certificate implements a simple form of conservative policy improvement [11]. It is a suficient evidence rule, not an optimal test or an impossibility bound for all estimators. Abstention means that this procedure does not resolve the comparison. Falling back to continuation after abstention does not certify continuation as optimal. Deployment also requires the calibration population to match the future runtime; the guarantee does not cover arbitrary distribution shift or uncorrected repeated checking.

![](images/f9c176fa4b229ceb12f3033f9d85e3f16445f40168e3886078995d465cc31849.jpg)

B Resolution cost of a boundary result  
![](images/25209cfce58fddecc89acdd8ed2ef9910d4ec927714d79a1f3bcf7f0e8283399.jpg)  
Analytical sensitivity only. A: failure TPR 0.90, FPR 0.10; no monitoring cost.  
B: equal-tailed binomial bounds, family error 0.05. Counts refer to independent eligible trials.

Figure 2: Analytical controls for monitor evaluation. Panel A keeps the failure detector’s true-positive and false-positive rates fixed while varying continuation success and restartability. The black curve is the zerogain boundary. Panel B gives the number of zero-success observations required by the specified simultaneous interval procedure to bound a success probability. Neither panel reports newly collected agent performance.

Evidence cost at zero success. For n observations with zero successes, the upper endpoint under this interval allocation is

$$
U _ { 0 } ( n , M ) = 1 - \left( \frac { \alpha } { 4 M } \right) ^ { 1 / n } .\tag{17}
$$

To obtain $U _ { 0 } \leq \epsilon$ requires

$$
n \geq \left\lceil { \frac { \log ( \alpha / ( 4 M ) ) } { \log ( 1 - \epsilon ) } } \right\rceil .\tag{18}
$$

These counts describe this specified binomial procedure. They quantify the evidence needed to bound one success probability, not suficient sample sizes to establish an intervention efect. Conditional continuation additionally requires that these observations reach the decision alive.

## 5 Experimental Method

## 5.1 Design principle

We use a sequential falsification design. Each experiment was motivated by the preceding result, but its task rule, intervention, outcome, and stopping rule were frozen before new outcomes were observed. Earlier data are used only where stated. This structure does not make an adaptive research program equivalent to one preregistered confirmatory study. It does prevent within-stage changes to the tested intervention after seeing its result.

## 5.2 Benchmark and runtime tuple

We use 19 issues from the SymPy repository in SWE-bench Verified [7]. Restricting to one repository removes language and build-system variation, while retaining issue-level diferences in symbolic mathematics, printing, assumptions, matrices, and related subsystems. It also limits external validity, which we treat explicitly in section 7.3.

Table 2: Experimental ladder. Model attempts count actual agent or resolver executions. Patch algebra used existing failed workspaces and no model calls.
<table><tr><td>Stage</td><td>Question</td><td></td><td>Attempts Frozen comparison</td></tr><tr><td>Initial restart</td><td>Does a short cutoff beat one 30-call at- tempt?</td><td></td><td>70 Cutoffs and universal sched- ules</td></tr><tr><td>Prospective transfer</td><td>Does the selected cutoff transfer to un- seen tasks?</td><td></td><td>60 Fixed 16 versus fixed 30</td></tr><tr><td>Horizon extension</td><td>Is completion mass hidden after call 30?</td><td></td><td>20 One 60 versus repeated 30</td></tr><tr><td>Context recovery</td><td>Does fresh context recover protocol col- lapse?</td><td></td><td>40 Baseline versus rejuvenation</td></tr><tr><td>Patch algebra</td><td>Do clean unions of failed patches solve?</td><td></td><td>0 19 frozen composites</td></tr><tr><td>Semantic resolver</td><td>Can one agent synthesize failed hy- potheses?</td><td></td><td>4 Four bounded portfolio runs</td></tr><tr><td>Total</td><td>19 unique tasks, two models</td><td></td><td>194 Plus 19 verifier-only candi- dates</td></tr></table>

An experimental runtime is identified by benchmark task, model endpoint, agent release, provider route, prompt configuration, and sampling policy. We use mini-swe-agent 2.4.6 [32]. Attempts use temperature 0.7, restore the benchmark-specified repository state before each run, execute under WSL2 Bubblewrap isolation without network access, and expose a workspace visible only to the attempt. The initial comparison uses openai/gpt-5.6-luna and deepseek/deepseek-v4-flash-0731. All subsequent model-driven experiments use the former.

The primary budget unit is a model call. Dollar cost and wall time are secondary operational measures. A 30-call attempt need not emit 30 valid environment actions, and an explicit format failure can end it early. We retain the distinction between nominal horizon, first verified completion, and protocol-terminal call.

Formally, let E be the first explicit protocol-terminal call and let H be the configured horizon. The trace records

$$
U = \operatorname* { m i n } \{ T , E , H \} , \qquad J \in \{ { \mathrm { v e r i f i e d } } , { \mathrm { p r o t o c o l ~ t e r m i n a l } } , { \mathrm { r i g h t ~ c e n s o r e d } } \} .\tag{19}
$$

The event label J matters for replay. A protocol terminal can release calls to a new episode; a right-censored live trajectory cannot be treated as known failure beyond H.

## 5.3 External verification and completion time

Each selected task passes two environment gates before collection: the benchmark-provided starting state fails the task evaluator and the reference patch passes it. The evaluator and test patch remain outside the agent sandbox. After every environment action, the controller transfers the candidate dif into a clean evaluator and runs the hidden fail-to-pass tests. Completion time is the first passing state. This prevents a later destructive action or an inaccurate self-declaration from changing the recorded time.

Every successful model attempt was audited for changes to the verifier, test runner, or hidden tests. None modified those assets. Successful patches changed implementation files, with occasional scratch reproductions. The analysis never exposes hidden verifier output, the reference patch, or completion labels to the model.

Table 3: Runtime controls shared across model-driven experiments unless a stage states otherwise.
<table><tr><td>Component</td><td>Frozen value</td></tr><tr><td>Benchmark</td><td>SWE-bench Verified, selected sympy/sympy tasks</td></tr><tr><td>Agent</td><td>mini-swe-agent 2.4.6</td></tr><tr><td>Models</td><td>openai/gpt-5.6-luna; historical comparator deepseek/deepseek-v4-flash-0731</td></tr><tr><td>Sampling</td><td>Temperature 0.7; separate provider draws</td></tr><tr><td>Primary budget</td><td>Model calls, with stage horizons of 12, 30, or 60</td></tr><tr><td>Verification</td><td>External fail-to-pass evaluator after every environment action</td></tr><tr><td>Isolation</td><td>Benchmark-specified task state, network-disabled Bubblewrap, evaluator outside workspace</td></tr><tr><td>Collection dates</td><td>August 31 and September 6, 2026</td></tr></table>

## 5.4 Restart replay

The initial cohort contains five nominally independent 30-call attempts for each of seven tasks and two models, totaling 70 observations. We evaluate a fixed cutof c under total budget 30 by empirical resampling of fresh attempts within task. Write $3 0 = q c + r$ , with $q = \lfloor 3 0 / c \rfloor$ and $0 \leq r < c$ . In the absence of an earlier protocol-terminal event, the plug-in probability is

$$
\widehat { P } _ { j , c } ( 3 0 ) = 1 - \left( 1 - \widehat { F } _ { j } ( c ) \right) ^ { q } \left( 1 - \widehat { F } _ { j } ( r ) \right) .\tag{20}
$$

The exact replay additionally releases unused calls when a recorded protocol-terminal event occurs before the assigned cutof. Macro success averages the resulting probability across tasks. Cutofs range from 4 to 30 in steps of two. The full-horizon cutof is the uninterrupted baseline. For geometric and Luby schedules, the implementation uses Monte Carlo replay with 50,000 episode sequences per task.

Cutof selection is evaluated in sample and by leave-one-task-out selection. Each fold selects the best cutof on the other tasks, breaking ties toward the longer cutof, then evaluates it on the held-out task. No attempt-level split is used because attempts from one issue share task dificulty.

## 5.5 Prospective task cohorts

The first new cohort selects the seven highest-ID unseen SymPy tasks that pass the environment gates. This mechanical cohort produced 35 attempts. Because all were solved early, a second dificulty-enriched cohort targeted unseen issues labeled at least one hour, then filled rejected positions from the highest-ID unseen issues labeled 15 minutes to one hour. Five tasks passed the gates, yielding 25 attempts. The transferred cutof of 16 calls and the 30-call baseline were fixed before either outcome set was observed.

The 60-call extension contains five fresh attempts on each of four stress tasks that were 0/5 at horizon 30. Its primary outcome is any first completion in calls 31 through 60. A secondary analysis separates right-censoring from explicit repeated-format termination.

## 5.6 Context-rejuvenation intervention

Four tasks that exhibited repeated-format termination receive five fresh baseline and five treatment attempts each. Both conditions have a shared 60-call cap. After three consecutive invalid tool responses, treatment preserves the exact repository state, discards the conversation, instantiates a fresh model client, and supplies the original issue plus deterministic git status and dif-stat summaries. It receives no extra calls, hidden feedback, progress score, or model checkpoint. Baseline follows the unmodified harness behavior.

The prospective expansion rule required at least a five-point success gain or two recovered successes, with no more than a 20% mean-cost increase. The rule was frozen before collection.

## 5.7 Failed-branch reuse

The context experiment yielded 38 externally failed, nonempty, production-code-only terminal patches after excluding one verified patch and one empty patch. We deduplicated them by content, enumerated unordered within-task pairs, and performed clean three-way composition against the common benchmark starting state. Candidate generation used no model, verifier output, or gold patch. The ranking was frozen before verification and allowed at most eight candidates per task. Nineteen candidates were evaluated.

The semantic-resolver screen then supplied one clean 12-call Luna attempt per task with the issue and every unique failed patch for that task, each explicitly labeled as a failed hypothesis. The resolver could inspect and edit a clean workspace. The four contexts were frozen before execution. The protocol allowed no retries, and a second stage would open only after at least one verified success.

## 5.8 Retrospective decision-value audit

We reanalyze the 70 initial and 60 prospective 30-call attempts, without additional model calls. These 130 observations form 26 task-model cells with five attempts each. The audit evaluates $t \in \{ 4 , 8 , 1 2 , 1 6 , 2 0 , 2 4 \}$ at total budget B = 30, giving a family of M = 156 comparisons. It uses the live eligibility rule and binomial intervals in section 4.3, with $\alpha = 0 . 0 5$ . Empty live populations are reported as not reached, not as successful abstentions. All attempt identities are deduplicated before analysis.

The decision is a single clean restart using the residual budget, not the repeated-cutof policy used in the earlier replay. We report conditional action values within each task, then weight the change by the observed fraction reaching the decision and average tasks equally. A task oracle chooses the better estimated action separately for each task; a perfect failure alarm additionally knows which live continuations will fail, as in corollary 4.2. Both are retrospective plug-in references, not deployed policies or validated opportunity bounds. In particular, zero observed restart success does not establish zero true restartability.

This audit was designed after inspection of the collected experiments. Its comparison family and analysis are fully specified for reproduction, but the audit is exploratory, not preregistered or a prospective test of the coverage guarantee. It supplies no paired state-conditioned intervention outcomes and does not evaluate a learned prefix monitor.

## 5.9 Uncertainty and reporting

For policy contrasts, percentile bootstrap samples tasks with replacement and then attempts within sampled task. Unless specified, intervals use 10,000 replicates and report the empirical 2.5% and 97.5% quantiles. We report bootstrap sign probabilities as descriptive quantities, not posterior probabilities or frequentist p values. Boundary outcomes are accompanied by an unclustered Newcombe interval as a sensitivity check where useful. All success estimates are absolute percentagepoint diferences.

Table 4: Main empirical results. Intervals resample tasks and attempts unless noted.
<table><tr><td>Comparison</td><td>Observed result</td><td>Interpretation</td></tr><tr><td>Fixed 16 versus 30, Luna training tasks</td><td>+2.29 points [0.00, 6.86]</td><td>Weak in-sample gain</td></tr><tr><td>Luna-selected fixed 16 on DeepSeek</td><td>-42.29 points [-69.14,-17.71]</td><td>Selected scale does not transfer across models</td></tr><tr><td>Fixed 16 versus 30, all 19 Luna tasks</td><td>-0.21 points [-4.21, 2.53]</td><td>Pooled estimate includes initial and prospective tasks</td></tr><tr><td>First completion after call 30</td><td>0/20; one-sided upper bound 13.9%</td><td>No observed useful late tail; rare mass remains possible</td></tr><tr><td>Fresh context versus baseline</td><td>0/20 versus 1/20; mean cost +56.7%</td><td>Protocol label improves, task outcome does not</td></tr><tr><td>Failed-branch reuse</td><td> $0 / 1 9$  composites; 0/4 resolvers</td><td>Textual diversity does not supply verified closure</td></tr></table>

For task j and policy $\pi ,$ let $\theta _ { j , \pi } ( B )$ be the probability of external verification within total budget B. The primary estimand is the task-macro policy value

$$
\Theta _ { \pi } ( B ) = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } \theta _ { j , \pi } ( B ) ,\tag{21}
$$

and a policy contrast is $\Delta _ { \pi , \pi _ { 0 } } = \Theta _ { \pi } ( B ) - \Theta _ { \pi _ { 0 } } ( B )$ . Equal task weighting prevents an issue with more retained attempts from dominating the result. The hierarchical bootstrap targets variability over both sampled tasks and provider draws within task. With only one repository and purposefully selected cohorts, it should not be read as a population interval over all software engineering tasks.

Unsuccessful attempts at a horizon are right-censored: they establish $T > c ,$ not $T = \infty$ [33]. Ofline replay never evaluates a cutof beyond the observed horizon. Explicit protocol termination is a competing event, recorded at its observed call, and may release the residual budget to a new attempt. Treating these terminal events as ordinary censoring would overcharge the episode and obscure the distinction between time-based truncation and fault recovery.

## 6 Results

Table 4 states the principal contrasts before the stage-by-stage analysis. The paper gives priority to frozen prospective comparisons over in-sample maxima.

## 6.1 A restart scale does not transfer across models

The initial 70 attempts already contain two diferent completion-time regimes under the same tasks and harness. Luna solves 29 of 35 attempts, mostly early. DeepSeek solves 24 of 35, but many completions occur after call 16. Per-task outcomes appear in table $5 ;$ the empirical completion distributions and fixed-cutof policy values appear in figure 3.

The best Luna cutof spans calls 16 through 20 and reaches 85.14%, compared with 82.86% for one 30-call attempt. At cutof 16, the estimated change is +2.29 points with a task-hierarchical interval of [0.00, 6.86] and bootstrap positive fraction 0.504. This is weak in-sample evidence.

Applying the same 16-call scale to DeepSeek changes success from 68.57% to 26.29%, a −42.29 point efect with interval [−69.14, −17.71]. DeepSeek’s best cutof is the full 30 calls. The direction of transfer in the other direction is conservative: applying 30 calls to Luna retains 82.86% and gives up only the small in-sample oracle diference.

Table 5: Initial externally verified outcomes. Call statistics are conditional on successful attempts.
<table><tr><td rowspan="2">Task</td><td colspan="2">Luna</td><td colspan="2">DeepSeek</td></tr><tr><td>Pass</td><td>Calls: min/mean/max</td><td>Pass</td><td>Calls: min/mean/max</td></tr><tr><td>23413</td><td>4/5</td><td>5/7.0/9</td><td>4/5</td><td>21/23.8/27</td></tr><tr><td>23824</td><td>5/5</td><td>4/5.2/6</td><td>5/5</td><td>11/12.8/15</td></tr><tr><td>23950</td><td>0/5</td><td>none</td><td>1/5</td><td>29/29.0/29</td></tr><tr><td>24066</td><td>5/5</td><td>5/7.0/11</td><td>3/5</td><td>20/25.0/29</td></tr><tr><td>24213</td><td>5/5</td><td>4/4.6/5</td><td>5/5</td><td>13/19.8/28</td></tr><tr><td>24443</td><td>5/5</td><td>8/12.0/16</td><td>4/5</td><td>19/23.0/30</td></tr><tr><td>24661</td><td>5/5</td><td>6/7.4/9</td><td>2/5</td><td>15/22.5/30</td></tr><tr><td>Total</td><td>29/35</td><td></td><td>24/35</td><td></td></tr></table>

Restart value depends on the model-specific completion scale

![](images/1f84645719b305d1ff1d822c3ff92cbb8477915dce5c8157428e1a3499829677.jpg)

<sup>B</sup> Policy value under a 30-call budget  
![](images/ac17a468716c69addb351e1c34a611b93a4b64609fb50faa9215a19872314672.jpg)  
Initial cohort: 7 tasks, 5 attempts per task and model. Bands are pointwise 95% task-then-attempt bootstrap intervals.  
Figure 3: Model-specific completion scale and restart-policy value on the initial seven tasks. Panel A reports the task-macro probability that one attempt has verified by call c. Panel B reports the exact plug-in success of repeated cutof c under 30 total calls. Shading gives pointwise 95% hierarchical bootstrap intervals. Enlarged points mark the Luna-selected cutof of 16 calls.

Leave-one-task-out selection removes the apparent Luna benefit. It yields 75.54% against 82.86% for the 30-call baseline, a −7.31 point diference. The decisive held-out task is 24443: the other six tasks select cutof 10, while all five successful runs of task 24443 require 8 to 16 calls and repeated cutof 10 has only 48.8% plug-in success. DeepSeek selects 30 calls in every fold and ties its baseline.

Finite geometric schedules reach at most approximately 84.59% on Luna and 22.88% on DeepSeek under the preregistered family. The tested Luby schedules range from 45.72% to 67.40% on Luna and have zero success on DeepSeek because none allocates an episode long enough for an observed completion. This is the scale sensitivity of proposition 3.3, not a contradiction of asymptotic restart theory.

Finding. The same tasks and harness do not determine a portable restart scale. Changing only the model moves useful completion mass beyond the selected cutof and turns a small apparent gain into a large loss.

Table 6: Transfer of the Luna-selected 16-call cutof. The interval resamples tasks and attempts.
<table><tr><td>Cohort</td><td>Tasks / attempts</td><td></td><td>Fixed 16</td><td>Fixed 30</td><td>Difference and 95% interval</td></tr><tr><td>Initial</td><td></td><td>7 / 35</td><td>85.14%</td><td>82.86%</td><td>+2.29 [0.00, 6.86]</td></tr><tr><td>Prospective high ID</td><td></td><td>7 /35</td><td>100.00%</td><td>100.00%</td><td>0.00 [0.00, 0.00]</td></tr><tr><td>Difficulty enriched</td><td></td><td>5 / 25</td><td>0.00%</td><td>4.00%</td><td>-4.00 [-16.00, 0.00]</td></tr><tr><td>Pooled</td><td></td><td>19 / 95</td><td>68.21%</td><td>68.42%</td><td>-0.21 [-4.21, 2.53]</td></tr></table>

## 6.2 The in-sample gain does not survive prospective transfer

The two prospective cohorts add 60 Luna attempts. All 35 attempts in the mechanically selected high-ID cohort pass by call 15, so both policies score 100%. In the dificulty-enriched cohort, only one of 25 attempts passes, at call 23. Fixed 16 therefore scores zero while fixed 30 scores 4%. Results are summarized in table 6.

Across all 19 Luna tasks, fixed 16 scores 68.21% and fixed 30 scores 68.42%. The diference is −0.21 points with interval [−4.21, 2.53] and bootstrap positive fraction 0.317. The pooled in-sample oracle is cutof 24 at 68.84%, only 0.42 points above fixed 30. Leave-one-task-out cutof selection scores 67.37%, a −1.05 point diference from fixed 30.

Under total budget 60, a conservative policy of 15, 15, then a reserved 30-call attempt scores 69.43%. Two 30-call attempts score 70.11%, for a diference of −0.67 points with interval [−2.53, 0.84]. Reserving a full episode avoids catastrophic truncation, but the short probes displace another useful long attempt.

Finding. The positive cutof result does not replicate prospectively. Easy tasks create no headroom, while the only stress-cohort success occurs after the selected cutof.

## 6.3 Policy diferences do not certify individual restart decisions

The retrospective audit separates where a decision occurs, which action has higher estimated value, and whether the estimate resolves its sign. Of 156 task-time cells, 103 contain at least one live attempt; the other 53 are never reached. None of the 103 live comparisons is certified by the simultaneous interval rule. This is a finite-sample limitation of the specified conservative procedure, not evidence that all actions have equal value.

At the previously selected cutof of 16, only 13 task-model cells remain live. Table 7 summarizes the decision opportunities; figure 4 displays conditional advantages and uncertainty. Five initial Luna tasks have completed in every attempt. Task 23413 has one survivor among five. For that survivor, estimated continuation success is zero and fresh success within the remaining 14 calls is 4/5. Its task-macro contribution is therefore

$$
{ \frac { 1 } { 7 } } \cdot { \frac { 1 } { 5 } } \cdot { \frac { 4 } { 5 } } = 0 . 0 2 2 8 5 7 .\tag{22}
$$

The remaining live Luna task, 23950, has no observed success in either arm. Thus the entire 2.29- point initial gain is supplied by one surviving failed attempt and the early successes of other draws on that same issue. The headline efect is not evidence of a broad stagnant-trajectory phenotype.

All six live DeepSeek task cells favor continuation in the plug-in estimates. The prospective high-ID cohort supplies no live decision at all: both policies tie because completion precedes the alarm time. The stress cohort supplies many failures, but no observed fresh completion within 14 calls. This is a failure-rich evaluation set with little measured recovery opportunity under the chosen action. A detector cannot be usefully assessed for completion gains merely by selecting trajectories likely to fail.

Table 7: Decision opportunities at call 16 with total budget 30. Changes are task-macro percentage points relative to continuation. The oracle knows future continuation failure and uses the specified independent clean restart. Its values are plug-in diagnostics, not statistically validated upper bounds. No live cell receives a sign certificate.
<table><tr><td>Cohort</td><td>Attempts</td><td>Live</td><td>Restart all</td><td>Perfect alarm</td></tr><tr><td>Initial Luna</td><td>35</td><td>6</td><td>+2.29</td><td>+2.29</td></tr><tr><td>Initial DeepSeek</td><td>35</td><td>27</td><td>-42.29</td><td>0.00</td></tr><tr><td>Prospective high ID</td><td>35</td><td>0</td><td>0.00</td><td>0.00</td></tr><tr><td>Prospective stress</td><td>25</td><td>25</td><td>-4.00</td><td>0.00</td></tr></table>

The perfect-alarm calculation in corollary 4.2 adds no estimated completion gain beyond continuation in either prospective cohort or the initial DeepSeek cohort at this decision time. This statement is narrowly conditional on the empirical distributions, call 16, and one independent clean restart. It does not bound a richer recovery action, a diferent decision time, or the true opportunity concealed by limited sampling. Even oracle diagnostics require uncertainty assessment.

The lack of individual certificates is compatible with the large cross-model policy loss. The earlier bootstrap measures an aggregate policy contrast across sampled tasks; the audit asks for the sign of each conditional action comparison with simultaneous protection. Five attempts per task, sometimes reduced to one live continuation, cannot provide comparable resolution. For example, equation (18) requires 90 zero-success observations to reduce the specified upper bound to 10% when $M = 1 5 6$ , versus 42 for one decision. These are procedure-specific evidence costs, not lower bounds on every possible learning method.

Finding. Failure prevalence, decision opportunity, and statistical support are diferent quantities. The same audit set can contain many failures, negligible measured recovery opportunity, and insuficient evidence to certify any individual action.

## 6.4 Additional horizon reveals protocol collapse, not late success

The horizon extension tests four tasks that were previously 0/5 at 30 calls. Among 20 fresh 60-call attempts, one passes at call 17. No completion first appears during calls 31 through 60. An unclustered one-sided 95% binomial upper bound is 13.9%. This attempt-level sensitivity calculation does not provide population coverage over heterogeneous tasks; rare late completion and repository-level variation remain unresolved.

Seventeen attempts end after three consecutive invalid tool responses and two exhaust the horizon. On the same four tasks at horizon 30, only one of 20 attempts had ended in repeatedformat failure. The attempt-level Fisher exact comparison of $1 7 / 2 0$ with $1 / 2 0$ gives $p = 4 . 0 6 \times 1 0 ^ { - 7 }$ ; it is descriptive because attempts are clustered within four tasks. Fourteen of the 17 format failures occur after call 30.

At total budget 60, repeated fixed 30 scores 9% by empirical replay, compared with 5% for one 60-call attempt. A policy that allows the full 60 calls but immediately recycles budget after explicit terminal failure also scores 9%. Hence the apparent restart benefit comes entirely from reusing calls after an observable protocol terminal event. Arbitrarily censoring a live trajectory adds nothing.

Decision direction is heterogeneous; every interval includes zero  
![](images/619f1d160eaaddc78d62bd6cab79103e4d9505a26c8a17a68cbf01a98173aef8.jpg)  
One decision at call 16, total budget 30. Simultaneous 95% bounds across 156 cells.  
Dots are retrospective estimates. No state-conditioned monitor or live restart was tested.

Figure 4: Conditional restart advantage in the 13 task-model cells with a live attempt at call 16. Dots are plug-in estimates. Lines are simultaneous diference bounds formed from binomial intervals across all 156 audited comparisons. Positive values favor restarting. Every interval includes zero despite substantial heterogeneity in point estimates.

Finding. More nominal horizon increases exposure to protocol failure but reveals no observed late completion. Recycling budget after an explicit failure is distinct from declaring a still-running trajectory stagnant.

## 6.5 Fresh context changes the terminal label, not success

The context intervention fires in 18 of 20 treated attempts. Baseline solves one of 20; treatment solves none. The task-clustered bootstrap diference is −5 points with empirical range [−20, 0]. Since treatment has no successes, this range is boundary-degenerate and cannot represent a positive treatment event. An unclustered Newcombe sensitivity interval is [−23.6, 15.2] points.

The treatment does eliminate the final RepeatedFormatError label. Every triggered attempt instead consumes its remaining calls, and all 20 treated attempts end at LimitsExceeded. Mean dollar cost rises from \$0.03786 to \$0.05934, a 56.7% increase. Mean wall time rises from 309.5 to 512.8 seconds, a 65.7% increase. Figure 5 displays the failure-mode substitution.

The frozen expansion criterion is not met. Fresh inference context removes the terminal interface symptom but preserves the workspace, including potentially wrong edits. The tested reset does not support conversational contamination as a suficient explanation for recoverable failure. It does not isolate which retained edits or other mechanisms caused the remaining failures.

Fresh context substitutes the failure mode without recovering success

Table 8: Fresh 60-call outcomes on tasks previously unsolved in five 30-call attempts.
<table><tr><td>Task</td><td>Pass</td><td>First pass</td><td>Format fail</td><td>Limit</td><td>Mean cost</td></tr><tr><td>16597</td><td> $0 / 5$ </td><td>none</td><td>4</td><td>1</td><td>$0.05716</td></tr><tr><td>17630</td><td> $0 / 5$ </td><td>none</td><td>5</td><td>0</td><td>$0.03577</td></tr><tr><td>21612</td><td> $1 / 5$ </td><td>call 17</td><td>4</td><td>0</td><td>$0.03126</td></tr><tr><td>21930</td><td> $0 / 5$ </td><td>none</td><td>4</td><td>1</td><td>$0.04996</td></tr><tr><td>Total</td><td>1/20</td><td>no pass after 30</td><td>17</td><td>2</td><td></td></tr></table>

![](images/31bd1cc98fb447f205c20307dfb2c41b1467046d79db638b771ae554d4d9a5d4.jpg)  
Prospective intervention: 20 attempts per condition under a shared 60-call budget.  
Figure 5: Context rejuvenation changes the observed failure mode without improving task success. Pane A shows terminal outcomes, with counts and within-condition percentages. Panel B shows the corresponding increase in mean metered cost and wall time relative to baseline.

## 6.6 Failed branches are diverse but not compositional

The 40-run context experiment yields 38 admissible failed sources and 37 unique patch contents. Of 162 unordered source pairs, 51 form clean, novel three-way composites. The frozen verifier-blind ranking selects 19. None passes. Task 17630 produces no clean novel composite because branches edit the same matrix-code region incompatibly.

One bounded semantic resolver is then run per task. It receives every unique failed patch as an explicitly failed hypothesis and starts from a clean workspace. All four runs use the 12-call cap and fail verification. The patches are not copies: their maximum character-level dif similarity to any supplied parent is 0.563, 0.700, 0.645, and 0.802. The task-level relevant tests pass in counts 76/79, 21/22, 99/100, and 47/52, but the protocol forbids extending the near miss after outcome inspection.

The historical clean-context cohort also has no completion within 12 calls on these tasks, so the resolver screen provides no evidence of improvement and is not treated as a randomized efect estimate. The null result has a more focused interpretation: syntactic novelty and access to multiple failed views are insuficient evidence of independent corrective information.

Table 9: Context-rejuvenation outcomes under the shared 60-call budget.
<table><tr><td>Outcome</td><td>Baseline</td><td>Rejuvenation</td><td>Treatment difference</td></tr><tr><td>Verified success</td><td>5.0%</td><td>0.0%</td><td>-5.0 points</td></tr><tr><td>Repeated-format terminal</td><td>90.0%</td><td>0.0%</td><td>-90.0 points</td></tr><tr><td>Limit terminal</td><td>5.0%</td><td>100.0%</td><td>+95.0 points</td></tr><tr><td>Mean cost</td><td>$0.03786</td><td>$0.05934</td><td>+$0.02148</td></tr><tr><td>Mean wall time</td><td>309.5 s</td><td>512.8 s</td><td>+203.3 s</td></tr></table>

Table 10: Verifier-blind patch composition from failed terminal branches.
<table><tr><td>Task</td><td>Sources</td><td>Unique</td><td>Pairs</td><td>Clean novel</td><td>Evaluated</td><td>Pass</td></tr><tr><td>16597</td><td>10</td><td>9</td><td>45</td><td>35</td><td>8</td><td>0</td></tr><tr><td>17630</td><td>10</td><td>10</td><td>45</td><td>0</td><td>0</td><td>0</td></tr><tr><td>21612</td><td>9</td><td>9</td><td>36</td><td>13</td><td>8</td><td>0</td></tr><tr><td>21930</td><td>9</td><td>9</td><td>36</td><td>3</td><td>3</td><td>0</td></tr><tr><td>Total</td><td>38</td><td>37</td><td>162</td><td>51</td><td>19</td><td>0</td></tr></table>

## 7 Discussion

## 7.1 Contribution and scope

The contribution is an operational account of when progress information can improve a specified agent-control decision. Classical restart optimization already allows informative observations, and contemporary learned monitors demonstrate successful termination and recovery. Our results do not overturn those findings. They connect failure discrimination to residual-budget restartability, distinguish decision-preserving compression from trace reconstruction, and expose the sampling support of per-task action comparisons.

Three levels of evidence must remain separate. The mathematical identities hold under stated assumptions and specialize established decision theory. The retrospective audit diagnoses the col lected traces and supplies a reusable calculation, but does not demonstrate a deployed policy gain. The prospective intervention screens establish outcomes of particular controllers on selected tasks, not universal limits on monitoring or recovery. Combining these levels makes the paper useful only if their inferential boundaries are preserved.

## 7.2 Implications for evaluation and deployment

Measure opportunity before monitor quality. A failure detector should be evaluated where an alternative action can plausibly recover value. A ceiling cohort ofers no live decisions; a hard cohort can ofer many failures but little residual-budget restartability. The perfect-alarm reference in corollary 4.2 is a diagnostic for this distinction. A small plug-in ceiling warrants additional evidence or a diferent intervention, not a conclusion that every progress signal is useless.

Match the signal to the action. A clean restart, conversational reset with mandatory workspace retention, and optional reuse of previous edits define diferent potential outcomes. Their action values cannot be interchanged. FailFast-RestartSmart [5], for example, evaluates a learned monitor with a recovery mechanism not tested here. Its positive results and our negative timeout screens can both hold. A meaningful comparison must fix reset semantics, residual budget, verifier, and monitoring cost.

![](images/a4da72b4e11c2e20ced50c6f578c951c2750bff5655c98f286ff27ff3736ba15.jpg)  
Observed outcomes distinguish protocol changes from verified completion.  
Figure 6: Five bounded tests of restart and recovery. Outcomes identify limitations of the tested interventions, not impossibility results for richer controllers. The task populations and uncertainty difer across stages.

Preserve decision distinctions, not every token. The information-value identity gives a precise connection to context compression. A representation can discard large parts of a trace without losing control value if the discarded distinctions do not reverse the preferred action. Conversely, faithful reconstruction can preserve many irrelevant details. This is a criterion for future evaluation, not evidence that our study has built a superior compressor.

Make abstention explicit. The released audit returns an action only when its specified intervals separate. Otherwise it reports insuficient evidence. This is useful as a calibration check before enabling an automatic restart rule; it is not a claim that continuation is optimal whenever the audit abstains. The present five-run cells are too sparse for this conservative task-specific rule. Hierarchical pooling could improve eficiency, but would introduce assumptions that require heldout validation.

Separate protocol repair from task recovery. Observable termination is not a live stagnation decision. Recycling its unused calls recovers budget in our replay. Replacing its final error label without changing verified completion is a diferent result. The context intervention illustrates why dashboards should report final task outcomes and cost alongside protocol diagnostics. Likewise, failed-patch diversity alone does not establish that a particular composition or synthesis operator can recover a solution. We do not estimate error correlation or identify the causal source of the unsuccessful combinations.

## 7.3 Limitations

All 19 tasks are selected SymPy issues. The cohort construction deliberately mixes a pilot, a mechanical high-ID selection, and dificulty enrichment; pooled rates do not estimate performance over SWE-bench Verified or agent workloads generally. Later interventions use one model and one compact harness. Two models on seven shared tasks demonstrate heterogeneity, not broad model coverage.

Five attempts per task leave substantial uncertainty, particularly after conditioning on live survival. The zero-success treatment and reuse screens do not exclude modest benefits in other populations. The unclustered late-tail calculation is a sensitivity result, not a task-population confidence statement. The retrospective decision family was chosen after the original experiments and is not a prospective validation of the certification rule.

The analysis assumes exchangeable independent provider draws within each fixed task and runtime. Shared routing, temporal drift, or correlated restarts can invalidate that approximation. The binomial certificate does not transfer across arbitrary changes in the task distribution, prompt, harness, provider, or recovery action. It also provides no protection for uncorrected adaptive monitoring.

The binary verifier measures a finite fail-to-pass specification, not every aspect of issue correctness. Base-fail/reference-pass gating and successful-patch audits reduce implementation threats but cannot eliminate specification gaps. Per-action external evaluation supplies completion labels unavailable to a typical production controller. Its computational overhead is not part of the model-call budget; reported model costs are not end-to-end deployment costs.

Finally, we have no matched prefix-conditioned continuation and restart trials for a learned signal. The empirical audit uses task identity and elapsed calls, not rich trace features. It cannot measure the decision value of embeddings, learned monitors, or semantic summaries. The information and opportunity identities describe what such experiments should estimate; they do not substitute for them.

## 7.4 A focused next test

The next decisive experiment should evaluate one existing monitor under the same recovery semantics as a simple baseline, rather than add another timeout family. First, use independent calibration data to identify tasks with both live failures and nontrivial residual-budget recovery. Freeze the monitor, threshold, budget, and clean-reset or edit-reuse rule. On held-out tasks, compare continuation, a calibrated fixed-time rule, and the monitor-triggered intervention, charging monitoring and recovery overhead to the same budget. Report opportunity, calibration, action-value regret, success, and cost separately.

A second model or repository is necessary to test transfer; more draws on the existing four hard tasks cannot establish it. Such a study would convert the current measurement contribution into evidence for a deployable control method. It is deliberately distinct from the present ofline audi and should not be described as completed.

## 8 Conclusion

Failure prediction is useful for agent control only through the value of the action it triggers. For an independent clean restart, that value depends jointly on alarm errors and success within the residual budget. Classical information comparison yields a decision-oriented criterion for retaining or compressing progress observations, while exact binomial intervals expose when a per-task action preference lacks statistical support.

Across 194 model executions and 19 verifier-only composites, simple restart and recovery interventions show strong runtime dependence and no prospective replication of the initial cutof gain. A new audit of 130 existing traces finds 103 live task-time comparisons but no sign certificate under its simultaneous rule. These findings motivate an evaluation standard: declare the intervention, measure recovery opportunity, account for monitoring cost, and report uncertainty at the level where decisions are made. RestartLab provides the accompanying trace analysis and audit implementation; broader prospective validation remains necessary.

## References

[1] Henry Kautz, Eric Horvitz, Yongshao Ruan, Carla Gomes, and Bart Selman. Dynamic restart policies. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 674-681, 2002. URL https://www.cs.cornell.edu/gomes/pdf/2002\_kautz\_aaai\_iaai\_dynamic.pdf.

[2] Matthew Streeter, Daniel Golovin, and Stephen F. Smith. Restart schedules for ensembles of problem instances. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1204-1210, 2007. URL https://cdn.aaai.org/AAAI/2007/AAAI07-191.pdf.

[3] Matthew Streeter. Bayes optimal early stopping policies for black-box optimization. arXiv preprint arXiv:1902.08285, 2019. URL https://arxiv.org/abs/1902.08285.

[4] Yaoqi Guo, Ying Xiao, Jie M. Zhang, Mark Harman, Yiling Lou, Yang Liu, and Zhenpeng Chen. EET: Experience-driven early termination for cost-eficient software engineering agents. arXiv preprint arXiv:2601.05777, 2026. URL https://arxiv.org/abs/2601.05777.

[5] Chenyu Wang, Yunbo Lyu, Junda He, Zhou Yang, Chenxing Zhong, Yaniv Harel, and David Lo. Fail-fast, restart-smart: Early failure prediction and restart for SWE agentic tasks. arXiv preprint arXiv:2608.03222, 2026. URL https://arxiv.org/abs/2608.03222.

[6] David Blackwell. Equivalent comparisons of experiments. The Annals of Mathematical Statistics, 24(2):265-272, 1953. doi: 10.1214/aoms/1177729032.

[7] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/ 2310.06770.

[8] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems, 2024. URL https://arxiv.org/abs/2405.15793.

[9] Michael Luby, Alistair Sinclair, and David Zuckerman. Optimal speedup of Las Vegas algorithms. Information Processing Letters, 47(4):173-180, 1993. doi: 10.1016/0020-0190(93) 90029-9.

[10] C. J. Clopper and E. S. Pearson. The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika, 26(4):404-413, 1934. doi: 10.1093/biomet/26.4.404.

[11] Romain Laroche, Paul Trichelair, and Remi Tachet des Combes. Safe policy improvement with baseline bootstrapping. In Proceedings of the International Conference on Machine Learning, pages 3652-3661, 2019. URL https://proceedings.mlr.press/v97/laroche19a.html.

[12] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2203.11171.

[13] Shunyu Yao, Dian Yu, Jefrey Zhao, Izhak Shafran, Thomas L. Grifiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models.

In Advances in Neural Information Processing Systems, 2023. URL https://arxiv.org/abs/ 2305.10601.

[14] Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. Inference scaling laws: An empirical analysis of compute-optimal inference for problem-solving with language models. arXiv preprint arXiv:2408.00724, 2024. URL https://arxiv.org/abs/2408.00724.

[15] Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou. Mixture-of-agents enhances large language model capabilities. arXiv preprint arXiv:2406.04692, 2024. URL https://arxiv.org/abs/2406.04692.

[16] Thomas G. Dietterich. Ensemble methods in machine learning. In Multiple Classifier Systems, pages 1-15. Springer, 2000. doi: 10.1007/3-540-45014-9\_1.

[17] Gavin Brown, Jeremy Wyatt, Rachel Harris, and Xin Yao. Diversity creation methods: A survey and categorisation. Information Fusion, 6(1):5-20, 2005. doi: 10.1016/j.infus.2004.04. 004.

[18] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegrefe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Sean Welleck, Bodhisattwa Prasad Majumder, Shashank Gupta, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, 2023. URL https://arxiv.org/abs/2303.17651.

[19] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, 2023. URL https://arxiv.org/abs/2303.11366.

[20] Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2310. 01798.

[21] Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. AgentBoard: An analytical evaluation board of multi-turn LLM agents. In Advances in Neural Information Processing Systems, 2024. URL https: //arxiv.org/abs/2401.13178.

[22] Hyundoo Park and Byungho Choi. When do agent loops mistake stagnation for progress? selfevaluation bias and externally grounded verification in long-running autonomous LLM agent loops. arXiv preprint arXiv:2607.25152, 2026. URL https://arxiv.org/abs/2607.25152.

[23] Sahil Shrivastava. Semantic early-stopping for iterative LLM agent loops. arXiv preprint arXiv:2606.27009, 2026. URL https://arxiv.org/abs/2606.27009.

[24] Nicolas Tacheny. Geometric dynamics of agentic loops in large language models. arXiv preprint arXiv:2512.10350, 2026. URL https://arxiv.org/abs/2512.10350.

[25] Qijun Han, Haoqin Tu, Zijun Wang, Haoyue Dai, Yiyang Zhou, Nancy Lau, Alvaro A. Cardenas, Yuhui Xu, Ran Xu, Caiming Xiong, Zeyu Zheng, Huaxiu Yao, Yuyin Zhou, and Cihang Xie. VLAA-GUI: Knowing when to stop, recover, and search, a modular framework for GUI automation. arXiv preprint arXiv:2604.21375, 2026. URL https://arxiv.org/abs/2604. 21375.

[26] Shukai Liu, Jian Yang, Bo Jiang, Yizhi Li, Jinyang Guo, Xianglong Liu, and Bryan Dai. Context as a tool: Context management for long-horizon SWE-agents. arXiv preprint arXiv:2512.22087, 2025. URL https://arxiv.org/abs/2512.22087.

[27] Lu Yi, Runlin Lei, Liuyi Yao, Yuexiang Xie, Yuyang Li, Wenhao Zhang, Zhewei Wei, Yaliang Li, and Jian-Yun Nie. Learning agent-compatible context management for long-horizon tasks. arXiv preprint arXiv:2605.30785, 2026. URL https://arxiv.org/abs/2605.30785.

[28] Xiaochuan Li, Ryan Ming, Meng Chu, Shuai Shao, Rong Jin, and Chenyan Xiong. ACM: Agentic context management for long horizon tasks. arXiv preprint arXiv:2607.23809, 2026. URL https://arxiv.org/abs/2607.23809.

[29] Sayash Kapoor, Benedikt Stroebl, Zachary S. Siegel, Nitya Nadgir, and Arvind Narayanan. AI agents that matter. arXiv preprint arXiv:2407.01502, 2024. URL https://arxiv.org/abs/ 2407.01502.

[30] You Wang, Michael Pradel, and Zhongxin Liu. Are “solved issues” in SWE-bench really solved correctly? an empirical study. arXiv preprint arXiv:2503.15223, 2025. URL https: //arxiv.org/abs/2503.15223.

[31] Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. Planning and acting in partially observable stochastic domains. Artificial Intelligence, 101(1-2):99-134, 1998. doi: 10.1016/S0004-3702(98)00023-X.

[32] SWE-agent contributors. mini-swe-agent: The 100-line AI agent that solves GitHub issues. GitHub repository, version 2.4.6, 2026. URL https://github.com/SWE-agent/ mini-swe-agent.

[33] John D. Kalbfleisch and Ross L. Prentice. The Statistical Analysis of Failure Time Data. Wiley, second edition, 2002. doi: 10.1002/9781118032985.

## A Notation

Table 11: Principal notation.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $X _ { t }$ </td><td>Latent agent and environment state after call t</td></tr><tr><td> $O _ { t } , \mathcal { F } _ { t }$ </td><td>Observation at call t and the observable history filtration</td></tr><tr><td> $T$ </td><td>First call whose resulting state passes the external verifier</td></tr><tr><td> $E , H$ </td><td>Explicit protocol-terminal time and configured attempt horizon</td></tr><tr><td> $B , b$ </td><td>Total policy budget and residual budget</td></tr><tr><td> $F ( t ) , \lambda _ { t }$ </td><td>Completion-time CDF and discrete completion hazard</td></tr><tr><td> $Q _ { a } ( h , b )$ </td><td>Conditional utility of action a after history h with residual budget b</td></tr><tr><td> $\Delta ( h , b )$ </td><td>Continuation advantage over restart</td></tr><tr><td> $G _ { t }$ </td><td>Observable candidate progress representation</td></tr><tr><td> $m _ { \mathrm { e f f } }$ </td><td>Effective number of independent views in a correlated ensemble</td></tr></table>

Table 12: Task IDs by cohort. IDs omit the common sympy\_\_sympy- prefix.
<table><tr><td>Cohort</td><td>Task IDs</td></tr><tr><td>Initial</td><td>23413, 23824, 23950, 24066, 24213, 24443, 24661</td></tr><tr><td>Prospective high ID</td><td>22456, 22714, 22914, 23262, 23534, 24539, 24562</td></tr><tr><td>Difficulty enriched</td><td>16597, 17630, 21612, 21930, 22080</td></tr><tr><td>Horizon 60</td><td>16597, 17630, 21612, 21930</td></tr><tr><td>Context  $\mathrm { A } / \mathrm { B }$ </td><td>16597, 17630, 21612, 21930</td></tr><tr><td>Patch and resolver</td><td>16597, 17630, 21612, 21930</td></tr></table>

## B Task Cohorts and Frozen Decisions

The task sets overlap across stages by design. Counts of 19 unique tasks and 194 model attempts deduplicate task identity but not fresh executions. The total consists of 70 initial attempts, 60 prospective attempts, 20 horizon-extension attempts, 40 context $\mathrm { A } / \mathrm { B }$ attempts, and four semanticresolver attempts. The patch-algebra stage adds 19 external verifier calls on composed candidates but no model attempt.

## C Policy Definitions

Single. Run one attempt up to budget B, stopping at first external pass or explicit terminal failure.

Fixed c. Run independent attempts with per-episode cap c until the total budget is spent. An explicit terminal failure releases unused episode budget.

Geometric. Allocate episode caps in a frozen geometric sequence, truncated by the remaining total budget.

Luby. Allocate episode caps according to a scaled prefix of the universal sequence in Luby et al.   
[9], truncated by the remaining total budget.

Horizon insurance. At budget 60, allocate two 15-call probes followed by one reserved 30-call episode. Compare with two 30-call episodes.

The fixed-policy plug-in estimate treats the five empirical attempts for each task as an IID distribution. For a sequence of episode caps $( c _ { 1 } , \ldots , c _ { k } )$ and empirical CDF $\widehat { F } _ { j }$ , the task success probability under independent resampling is

$$
1 - \prod _ { i = 1 } ^ { k } \Big ( 1 - \widehat { F } _ { j } ( c _ { i } ) \Big ) .\tag{23}
$$

Monte Carlo replay is used in the implementation when terminal failures create episode-dependent residual budgets.

## D Additional Operational Results

Costs are provider-metered values recorded in the traces. They are not estimates of deployment expense. The large cost and latency change in context rejuvenation arises because treatment consumes all 60 calls after replacing an early terminal format failure with a fresh episode.

Table 13: Collection cost and accumulated wall time. Wall time sums parallelizable attempts and is not elapsed study duration.
<table><tr><td>Stage</td><td>Attempts</td><td>Metered cost</td><td>Wall hours</td></tr><tr><td>Initial Luna</td><td>35</td><td>$0.38327</td><td>0.771</td></tr><tr><td>Initial DeepSeek</td><td>35</td><td>$0.38023</td><td>3.644</td></tr><tr><td>Prospective Luna</td><td>60</td><td>$1.14379</td><td>2.596</td></tr><tr><td>Horizon 60</td><td>20</td><td>$0.87077</td><td>2.207</td></tr><tr><td>Context A/B</td><td>40</td><td>$1.94403</td><td>4.568</td></tr><tr><td>Semantic resolver</td><td>4</td><td>$0.05132</td><td>0.145</td></tr><tr><td>Total</td><td>194</td><td>$4.77341</td><td>13.931</td></tr></table>

## E Artifact Manifest

The accompanying analysis package contains the following sanitized artifacts. They exclude prompts, trajectories, local paths, hidden verifier output, reference patches, and credentials.

Table 14: Accompanying analysis artifacts and their role in the study.
<table><tr><td>Artifact cohort</td><td>Scope</td><td>Primary use</td></tr><tr><td>Initial paired-model cohort</td><td>70 attempts, 7 tasks</td><td>Completion curves and restart replay</td></tr><tr><td>Prospective transfer cohorts</td><td>60 attempts, 12 tasks</td><td>Out-of-sample cutoff transfer</td></tr><tr><td>Horizon extension</td><td>20 attempts, 4 tasks</td><td>Late completion and protocol termination</td></tr><tr><td>Context intervention trial</td><td>40 attempts, 4 tasks</td><td>Terminal outcomes, cost, and latency</td></tr><tr><td>Patch-composition screen</td><td>19 verifier-only candidates</td><td>Failed-branch compositionality</td></tr><tr><td>Semantic resolver screen</td><td>4 attempts, 4 tasks</td><td>Semantic aggregation</td></tr><tr><td>Decision audit</td><td>156 derived cells from 130 existing attempts</td><td>Action values, eligibility, and simultaneous bounds</td></tr></table>

The package also contains stage protocols, the frozen composition plan, agent adapters, replay policies, analysis code, command-line tools, and tests. The sanitized artifacts reproduce the reported trace analyses and figures ofline; new agent collection requires a model credential. Derived audit cells reuse earlier attempts and do not increase the count of model executions.

## F Threats to Interpretation

Task selection. The first seven tasks were an operational pilot, the prospective high-ID cohort used a mechanical order, and the stress cohort deliberately enriched dificulty. Pooled rates are descriptive for these tasks, not prevalence estimates for SWE-bench Verified.

Verifier granularity. Verification after every environment action yields precise first-pass timing, but the fail-to-pass suite is still a finite specification. We report verifier success rather than semantic issue resolution and retain the standard benchmark terminology only when naming the dataset.

Adaptive sequence. Later stages were chosen after earlier results. Each stage has its own frozen protocol, but cross-stage narratives can still invite hindsight. The ladder table therefore states the question and comparison at each stage, and null follow-ups are retained rather than omitted.

Multiplicity. The study evaluates several policies and mechanisms. The stage-level bootstrap intervals are not adjusted for simultaneous coverage. Only the decision-audit intervals use the

explicit family allocation in section 4.3. We avoid claiming significance for exploratory maxima, identify in-sample oracles as nondeployable, and give priority to prospective contrasts.

Model identity. Hosted behavior can change under a stable model identifier. Requested model ID and collection date therefore define, but do not fully identify, the runtime used for calibration.

## G Failure-Conditioned Branches: Supporting Examples

Branching is often proposed when continuation is uncertain. Its benefit depends on the epistemic relation among branches, not their textual distance. Conditioning all inputs to an ensemble on terminal failure can also reverse component reliability.

Lemma G.1 (Failure-conditioned component inversion) Let $Y \sim$ Bernoulli(q) indicate that one component of a patch is correct, with $q > 1 / 2$ . Let $Z \sim \mathrm { B e r n o u l l i } ( r )$ indicate that the remaining requirements are correct, independent of Y, and let full success be $S = Y Z$ . Then

$$
\mathbb { P } ( Y = 1 \mid S = 0 ) = \frac { q ( 1 - r ) } { 1 - q r } < \frac { 1 } { 2 }\tag{24}
$$

whenever $r > ( 2 q - 1 ) / q$

Proof. Bayes’ rule gives the displayed conditional probability. Comparing it with $1 / 2$ yields $2 q ( 1 - r ) < 1 - q r$ , which is equivalent to $r > ( 2 q - 1 ) / q$ □

Thus a component that is reliable in the unconditional branch population can become minorityreliable after selecting only failed full patches. The lemma is an existence result, not an estimate of our patch population.

Correlation supplies a second limit. Let $Y _ { 1 } , \dots , Y _ { m }$ be exchangeable correctness indicators with mean $q$ and common pairwise correlation $\rho .$ Then

$$
\mathrm { V a r } \left( { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } Y _ { i } \right) = { \frac { q ( 1 - q ) } { m } } \left[ 1 + ( m - 1 ) \rho \right] .\tag{25}
$$

The variance is that of only

$$
m _ { \mathrm { { e f f } } } = \frac { m } { 1 + ( m - 1 ) \rho }\tag{26}
$$

independent views. If $\rho = 1$ , increasing the number of branches does not reduce uncertainty at all. This standard ensemble efect [16, 17] motivates our distinction between behavioral diversity, such as diferent files or difs, and epistemic independence, meaning error diversity suficient for aggregation. Branches from one model and harness can satisfy the former while failing the latter.