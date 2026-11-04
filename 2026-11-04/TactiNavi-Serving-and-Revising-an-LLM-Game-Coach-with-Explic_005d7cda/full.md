# TactiNavi: Serving and Revising an LLM Game Coach with Explicit Strategy Conditions

Qi Liu<sup>\*1</sup>, Xiaoyang Yuan<sup>\*1,2,3</sup>, Zhuomeng Zhang<sup>1</sup>, Yubin Ruan<sup>1</sup>, Xinyi Mou<sup>1</sup>, Wenjin Wang<sup>1</sup>, Hanying Jiao<sup>1</sup>, Di Wu<sup>1</sup>, Jinke Li<sup>1</sup>, Jingxuan Hong<sup>1</sup>, Ke Feng<sup>1</sup>, Zixun Sun<sup>1</sup>

<sup>1</sup>Tencent <sup>2</sup>Tongji University <sup>3</sup>Shanghai Innovation Institute

## Abstract

A deployed LLM coaching agent answers requests whose correct response depends on player state and game version, so advice that is right for one player can be wrong for the next. We describe TactiNavi, the serving system behind the LLM coach of a commercial openworld game, which makes strategy applicability an explicit artifact shared by offline compilation, online execution, failure diagnosis, and release. An offline compiler turns versioned game evidence and expert guidance into strategy packages with explicit prerequisites, the serving plane evaluates those prerequisites as three-valued conditions over structured player state and records decision traces, and a control plane uses the resulting condition–rule dependencies to validate revisions on held-out traffic before release. In a retrospective evaluation of 1,122 logged coaching turns from three deployment periods, condition-bearing packages raise correctness from 57.9% to 82.0% over atomic facts and reduce strategy misapplication under changed player state from 18.1% to 10.4%, while trace-based claim verification reaches 86.4% precision at a median reply latency of 2.07 s. The deployment also showed that most apparent fixes are generation noise, because an unchanged-system rerun recovers 38.4% of failing turns and only one of 15 shipped revision units passes the release checks retrospectively. We distill these results into lessons on treating applicability as a firstclass serving object, sizing traces for later diagnosis, and building the null model into the release path.

## 1 Introduction

Deployed LLM agents increasingly answer requests whose correctness depends on live state rather than on the request alone. A coaching assistant for a strategic game advises on lineups, resource use, and matchups, and the same advice can be right for one player and wrong for another because players own different characters, have made different progress, and play under different game versions. Serving such a system is therefore not only a retrieval problem. The system must decide, per request, whether a piece of strategic advice actually applies, and it must keep making that decision as the game changes underneath the deployment.

Retrieval and grounded synthesis supply the content that such an assistant needs (Lewis et al., 2020; Sarthi et al., 2024; Shao et al., 2024), and offline compilation makes that content available at request time. Applicability is a separate problem. When the prerequisites of a strategy stay embedded in prose, the reply model has to notice them, decide whether the current state satisfies them, and explain itself when it cannot tell. A lineup that requires a support role makes the difficulty concrete. The requirement is trivial to check against a roster, yet a prose description leaves the check to generation. Explicit conditions move that work out of generation and into a deterministic evaluation over structured state, where a missing field yields unknown instead of a guess.

The same representation also determines what the system can do when a recommendation turns out to be wrong. A player complaint may target an earlier turn, and the disputed claim may rest on flawed strategic knowledge, on a prerequisite that nobody wrote down, on a rule that failed to enforce a valid prerequisite, or on stale player state. Separating these causes requires a record of how the response was produced, and repairing them requires knowing which knowledge and which execution rules depend on each other. A knowledge edit that introduces a new prerequisite is incomplete until some rule can evaluate it, and a rule edit may invalidate the packages that rely on it.

Revision then becomes a measurement problem. A candidate that fixes the reported case may leave the underlying mechanism broken, and a candidate that improves the targeted behavior may damage traffic that nobody thought to check. Generation variability compounds the difficulty, because a rerun of an unchanged system already recovers a substantial share of previously failing interactions. Release decisions therefore need held-out cases drawn from the conditions and rules a revision touches, together with an independent measurement of what the revision actually changed.

We present TactiNavi, a serving system that carries explicit applicability conditions through the whole lifecycle of a coaching agent. Gamespecific information enters through three interfaces, namely versioned knowledge and expert guidance, a schema for player state, and tools for game-mechanic calculations. A Strategic Knowledge Compiler synthesizes evidence-grounded strategy packages offline. An Agent Execution Plane evaluates package conditions against player state and game version, recommends only strategies whose prerequisites hold, offers alternatives or asks for missing information otherwise, and records decision traces that link claims to evidence, condition outcomes, and execution rules. A Diagnosis and Repair Evaluator uses those traces to locate the response that feedback targets, verify its claims, and attribute contradictions to the first unsupported stage. An Evolution Control Plane uses the dependency links between conditions and rules to select affected held-out interactions, construct condition-derived counterexamples, and compare reported gains with independently measured effects before a versioned release.

We instantiate TactiNavi as the coaching assistant of a commercial open-world game with strategic battles and evaluate the deployment retrospectively on 1,122 logged coaching turns from three consecutive periods separated by two production revision rounds. Strategy packages reach 82.0% correctness against 57.9% for atomic facts, and under controlled state changes they reduce strategy misapplication from 18.1% to 10.4%. Claim verification reaches 86.4% precision on player doubts and localizes doubts that refer back to earlier turns, which preceding-turn baselines cannot do. Median reply generation takes 2.07 s, with a 95th percentile of 5.48 s. The revision experiments explain why the control plane is needed. An unchanged system recovers 38.4% of previously failing turns when simply rerun, only one of 15 shipped revision units satisfies the release criteria, and one unit that gains 25.0 points inside its declared scope loses 20.8 points outside it. Section 6 condenses these measurements into lessons for agents that serve state-dependent advice.

Our contributions are:

• Condition-bearing knowledge and an offline compiler. We separate what a strategy says from when it applies, compile evidencegrounded packages with explicit prerequisites, counterexamples, and instructions, and evaluate those prerequisites as three-valued conditions over structured player state and game version.

• A serving plane that records enough to repair itself. Decision traces bind response claims to evidence, condition outcomes, and consuming rules, which supports target-turn localization, claim verification, first-unsupported-stage attribution, and the derivation of dependent knowledge and execution repair targets.

• Dependency-scoped revision control. We select held-out cases from the conditions and rules a revision touches, add conditionderived counterexamples, replay candidates against an unchanged-system baseline, and gate release on validated gains and scopeaware regressions.

## 2 Related Work

Knowledge serving for LLM agents. Retrieval methods give models access to external evidence (Lewis et al., 2020; Guu et al., 2020; Jiang et al., 2023; Asai et al., 2024), and synthesis methods organize that evidence into reusable artifacts (Sarthi et al., 2024; Shao et al., 2024). Agent architectures such as ReAct, CoALA, and Talker-Reasoner structure the loop that consumes such artifacts, combining reasoning, tool calls, and interaction (Yao et al., 2022; Sumers et al., 2023; Christakopoulou et al., 2024). Tool-use work studies how models learn to call external functions and how those calls are planned and executed (Schick et al., 2023; Qin et al., 2024; Kim et al., 2023). State-Grounded Conditioning (SGC) brings structured user state and deterministic rules into the wrappers around a user-facing agent (Liu et al., 2026). TactiNavi builds on that grounding mechanism and adds a compiled representation of strategy applicability, so that a prerequisite becomes an object that is evaluated, traced, and revised rather than a sentence the model must interpret.

Game agents and coaching systems. Gameplaying agents use documentation, execution feedback, reusable skills, and retrieval to plan and act (Wu et al., 2023; Wang et al., 2024, 2023; Hu et al., 2024). Coaching supports decisions that players make themselves. MOBA Coach analyzes match data for tactical review (Horst et al., 2021), video-based coaching identifies moments worth commenting on (Akie et al., 2026), ExIn-COACH couples reinforcement learning with language models for state-aware tutoring (Hua et al., 2026), and observational studies of esports coaching emphasize situational analysis and actionable explanation (Lee et al., 2025). These systems motivate guidance that adapts to circumstances. Our interest is in the serving and maintenance side of that requirement, where the same condition both selects a strategy and scopes a repair.

Failure attribution and agent revision. Agent improvement spans reflection and self-critique (Shinn et al., 2023; Madaan et al., 2023), toolgrounded correction (Gou et al., 2024), and prompt or pipeline optimization (Pryzant et al., 2023; Khattab et al., 2024). The Detection-Aware Attribution Framework (DAAF) formulates failures in terms of editable system attributes that are identified from controlled interventions (Yuan et al., 2026). Who&When attributes failures in multi-agent traces in a single pass (Zhang et al., 2025). At the system level, Ouroboros and ModularRSI improve an agent harness through reviewed changes and recurring deficiencies (Razzhigaev et al., 2026; Wu et al., 2026), and RRSI and GDPevo study how self-improvement can be constrained or evaluated so that gains generalize (Xia et al., 2026; Zhou et al., 2026). Tacti-Navi combines target-turn localization and claim verification with condition–rule dependency tracing, which lets one attribution step name both the knowledge and the execution rule that a repair has to touch.

Release engineering for learned systems. Serving learned components safely has a long literature on hidden technical debt and production readiness (Sculley et al., 2015; Breck et al., 2017). Recent work evaluates deployed agents against production-aligned signals and audits reusable skills with verifiers (Wang et al., 2026; Huang and Huang, 2025), while benchmark studies show how easily evaluation integrity degrades when the evaluator is inside the optimization loop (Atinafu and Cohen, 2026; Thaman, 2026). TactiNavi treats release as a measurement stage with its own held-out traffic, keeps the release scorer outside candidate development, and reports what an unchanged system already achieves so that a candidate gain can be compared against that floor.

## 3 Workload and Requirements

Requests and state. The workload consists of player requests that arrive inside a session, each accompanied by dialogue history and a structured snapshot of player state. The snapshot covers owned characters, resources, progress, and the game version that the session runs against. Roughly three quarters of the coaching turns in our trace ask for strategic guidance such as which lineup to field or how to allocate a limited resource, and the remainder ask about mechanics such as damage calculation or skill interaction. Strategic requests are the ones that need a synthesized strategy rather than a fact lookup.

Knowledge and revision cadence. Two sources change on different clocks. Game knowledge changes with version updates and expert corrections, and the deployed system is revised in rounds that bundle many small changes. In our evaluation window, the system was revised twice over three consecutive two-day periods, producing three versions and 15 revision units that we group by the strategic sub-intent packages they affect. Revisions arrive after the fact. A player complaint can refer to a response several turns earlier, and the engineer who handles it sees the complaint rather than the state that produced the original answer.

Interfaces. Three interfaces carry everything that is specific to the game. The first supplies versioned basic knowledge and expert guidance, which the compiler consumes offline. The second defines the state schema that the execution plane reads, including the fields that conditions are allowed to reference. The third exposes tools for mechanic calculations such as damage multipliers and skill interactions. Everything else in TactiNavi is game-independent, and this boundary is what makes the design portable to another title.

Requirements. Four requirements follow from this workload, and Table 1 maps each to the mechanism that addresses it.

R1, grounded content. Advice must be traceable to evidence for the version in force, because a plausible recommendation drawn from an outdated description is worse than no recommendation for a player who acts on it.

Table 1: Serving requirements and the mechanisms that address them.
<table><tr><td>Req.</td><td>Mechanism</td></tr><tr><td>R1</td><td>Versioned packages with evidence, prerequisites, and applicable versions</td></tr><tr><td>R2</td><td>(Sec. 4.2). Three-valued conditions over state,</td></tr><tr><td>R3</td><td>evaluated before generation (Sec. 4.3). Traces and condition-rule dependencies for attribution (Sec. 4.4).</td></tr><tr><td>R4</td><td>Scoped validation, counterexamples, and release gates (Sec. 4.5).</td></tr></table>

R2, state-dependent applicability. A strategy may be recommended only when its prerequisites hold for the current player, and the system must distinguish a prerequisite that is false from one that cannot be evaluated. Collapsing the second case into either truth value produces confident advice from missing data.

R3, diagnosability. When feedback questions a response, the system must locate the response, decide whether the disputed claim is actually wrong, and name the component that should change. Without a record of the evidence and rules behind the response, every complaint turns into a manual reconstruction.

R4, bounded revision risk. A revision must be assessed on interactions that its conditions and rules can reach, compared against the variation of an unchanged system, and checked for effects outside its declared scope. Revision proposals are cheap to generate and expensive to ship, so the gate has to be stricter than the proposal process.

## 4 System Design

## 4.1 Overview

Figure 1 shows TactiNavi as a compiler and two planes that share one representation, together with an evaluator that connects them. The Strategic Knowledge Compiler runs offline and produces versioned strategy packages. The Agent Execution Plane serves requests, evaluates package conditions against the current snapshot, and records a decision trace for every response. The Diagnosis and Repair Evaluator reads those traces after player feedback arrives and proposes candidate revisions. The Evolution Control Plane validates each candidate on held-out traffic before it enters the serving path. Applicability conditions and the links between conditions and execution rules are the objects that cross all four components, which is why a prerequisite written once can be evaluated online, traced during diagnosis, and used to scope a release decision.

## 4.2 Strategic Knowledge Compiler

Compilation tasks. The compiler builds strategies offline from basic game knowledge and expert guidance. Each task declares a strategic intent, a scope, the versions it applies to, and completion criteria. A knowledge agent gathers evidence from character attributes, skills, counter relations, game rules, and battle statistics. Deterministic tools verify the calculations that the evidence depends on, covering damage multipliers and skill interactions, and a critic tests assumptions and searches for counterexamples. Every resulting claim keeps its evidence, its prerequisites, and the versions under which it holds.

Package structure. Verified claims become versioned strategy packages that carry applicability conditions, counterexamples, and instructional explanations. A condition states a requirement over player state or game version and evaluates to true, false, or unknown when the available information is insufficient to decide. A lineup strategy may require a support role, for instance, and leave the choice of which eligible owned character fills that role to online execution. Instructions explain why a strategy works, how to apply it, and when to abandon it. Consistency checks and expert review qualify a package as a candidate for versioned release.

Dependency index. Each condition is linked to the state fields it reads and to the execution-plane rules that evaluate it. These links turn a knowledge artifact into a unit of change with a known blast radius. A requirement that existing rules can already evaluate is updated inside its package, while a new prerequisite that no rule can evaluate also requires a rule change. The same links, combined with decision traces, identify the recorded interactions that a proposed knowledge or execution change would reach.

## 4.3 Agent Execution Plane

State-grounded conditioning. Building on SGC (Liu et al., 2026), the execution plane, which we call the Coach, conditions replies on intent, state, and interaction history through Perception,

![](images/d9af941b46886410db746d776db470491b2519af92d9b97661780c066b8548e7.jpg)  
Figure 1: Overview of TactiNavi. The Strategic Knowledge Compiler supplies versioned strategy packages for online condition evaluation. The Agent Execution Plane generates guidance and records decision traces, which support feedback-driven diagnosis. Attributed repair targets enter the Evolution Control Plane for independent validation, and accepted revisions update knowledge and execution components for subsequent coaching.

Grounding, and Interaction wrappers. The state interface supplies player observations, owned resources, progress, and game version. Information that the interface cannot supply stays unknown and is never filled in by the reply model.

Request path. The Coach first classifies the request into an intent and sub-intent, then retrieves the packages that match the player’s goal, subintent, and game version. The Grounding wrapper checks ownership and resources, and conditional knowledge rules evaluate the remaining prerequisites while deriving the facts that execution needs, such as which owned characters satisfy a required support role. The three-valued outcomes then select a response mode. Satisfied conditions permit the strategy, violated conditions produce an alternative that explains the unmet requirement, and unknown conditions trigger retrieval or a clarification question. Bounded game-tool calls resolve remaining character comparisons, damage estimates, and matchup constraints, with independent calls issued in parallel when possible (Kim et al., 2023). The resulting decisions and instructions condition the reply model, which writes the response.

A worked request. A player asks which lineup to field. Retrieval returns a package whose condition requires a support-role character, and the grounding rule that checks ownership finds no eligible character in the roster, so the condition evaluates to false and the Coach answers with an alternative that names the missing role instead of recommending the package. Had the roster been absent from the snapshot, the outcome would be unknown and the reply would ask which characters the player owns. Both paths record the same fields, which is what a later complaint about this turn is diagnosed against.

Decision traces. Each interaction produces a trace that links the response to the selected packages and component versions, the player state that was read, the condition identifiers and their outcomes, the supporting evidence, the tool results, and the reply context. Traces are written on the serving path, so their cost has to stay small relative to generation, yet the same fields are what make the system replayable. Controlled replay holds requests, histories, player states, planning decisions, and tool outputs fixed while varying only knowledge and reply-stage rules, which turns a production interaction into a controlled comparison without re-running the rest of the pipeline.

## 4.4 Diagnosis and Repair Evaluator

The evaluator follows the DAAF formulation of failures as editable system attributes (Yuan et al., 2026). For coaching, it combines target-turn localization, claim verification, and knowledge– execution dependencies to name repair targets for an earlier response or for a single claim inside it. It can also conclude that no repair is needed, or leave attribution unresolved when the evidence does not decide the question.

Detection and localization. Player doubts, corrections, frustration, tool failures, and empty retrieval results trigger review, together with a sample of unflagged interactions. The evaluator locates the response that the feedback targets, retrieves its trace, and judges the original recommendation against the evidence that was available when it was produced. Judging an old response against current knowledge would attribute every knowledge update to the response rather than to the knowledge.

Claim verification and evidence chains. The evaluator decomposes the targeted response into atomic claims labeled supported, contradicted, or unknown with respect to versioned knowledge, player resources, and recorded tool outputs. For a contradicted claim it first asks whether the reply misread the context it was given. If the context itself was wrong, it walks the evidence chain through context assembly, routing, tool invocation, and knowledge retrieval to find the first stage that lacks support. Condition identifiers then connect the disputed claim to its prerequisites and to the rules that evaluate them. A flawed prerequisite implicates the package, a valid prerequisite that execution failed to enforce implicates a rule, and inaccurate resources implicate state processing. A verdict record captures the evidence and the implicated attributes, including cases with several targets or none.

Repair effect. An attributed candidate $\delta$ replaces an editable attribute or a dependent set of knowledge and execution attributes. The evaluator replays the current configuration S and the candidate $S ^ { \delta }$ on matched development cases $D _ { \mathrm { s e l } }$ , holding requests, player states, game versions, and tool results fixed. With $Q _ { i } ( S )$ denoting the correctness score of case $i ,$ averaged over repeated generations under a fixed claim-level rubric and executable checks, the paired effect is

$$
\widehat { \Delta } _ { \mathrm { s e l } } ( \delta ) : = \frac { 1 } { | D _ { \mathrm { s e l } } | } \sum _ { i \in D _ { \mathrm { s e l } } } \big [ Q _ { i } ( S ^ { \delta } ) - Q _ { i } ( S ) \big ] .\tag{1}
$$

A gain on $D _ { \mathrm { s e l } }$ is therefore treated as a claim that the control plane has to verify.

## 4.5 Evolution Control Plane

The control plane assesses revisions on held-out cases with a fixed release protocol that is separate from candidate development. It uses the same correctness definition as repair assessment, while validation cases and release records stay independent of candidate selection.

Proposals and dependency closure. A proposal records the triggering traces, the targeted attributes, the intended correction, the declared scope, the component versions, and the development gain. Knowledge edits are prepared through the compiler, and condition–rule dependencies determine which execution changes are required for the edit to take effect. Dependent edits are evaluated together as one versioned candidate, because a knowledge change without its rule change would ship a prerequisite that nothing enforces.

Condition-aware validation. The held-out set $D _ { \mathrm { v a l } }$ combines interactions exposed to the affected conditions or rules, condition-derived counterexamples, and cases for broader regression assessment. For a revised support-role requirement, state variants represent players with an eligible owned character, players without one, and players whose ownership is unknown. The variants preserve unrelated context, and rephrased requests test whether a correction generalizes past the wording that triggered it. Satisfied conditions test that valid strategies remain available, violated conditions test that the Coach still guides the player toward a feasible alternative, and unknown conditions test that incomplete information is treated as unresolved. The validation set therefore mirrors the semantics of the revised condition rather than the surface form of the reported case.

Gain reconciliation. The control plane reexecutes S and $S ^ { \delta }$ on $D _ { \mathrm { v a l } }$ and computes the paired effect ${ \widehat \Delta } _ { \mathrm { v a l } } ( \delta )$ by applying Eq. 1 to the valbidation cases. The gap between reported and validated gain is

$$
\gamma ( \delta ) = \widehat { \Delta } _ { \mathrm { s e l } } ( \delta ) - \widehat { \Delta } _ { \mathrm { v a l } } ( \delta ) .\tag{2}
$$

Decision traces mark the cases that $\delta$ can reach, either because modified knowledge enters their reply context or because a modified rule applies to them. Exposed cases split into those inside and outside the declared repair scope, and the rest stay unexposed, so the validated effect decomposes as

$$
\widehat { \Delta } _ { \mathrm { v a l } } ( \delta ) = \sum _ { g \in \{ \mathrm { i n } , \mathrm { o u t } , 0 \} } \pi _ { g } \widehat { \Delta } _ { g } ( \delta ) ,\tag{3}
$$

Algorithm 1 Dependency-scoped revision valida  
tion   
1: input: configuration $S ,$ , candidate $\delta ,$ develop  
ment cases $D _ { \mathrm { s e l } }$   
2: apply $\delta$ and close over dependent conditions   
and rules   
3: select $D _ { \mathrm { v a l } }$ from exposed cases, counterexam  
ples, and rephrased requests   
4: measure $\hat { \Delta } _ { \mathrm { s e l } }$ on $D _ { \mathrm { s e l } }$ by Eq. 1   
5: measure $\Delta _ { \mathrm { v a l } }$ and its group terms on $D _ { \mathrm { v a l } }$   
b6: checks: in-scope lower bound $> 0 ,$ recov  
ery at least half of development recovery, un  
exposed shift within rerun variation, out-of  
scope loss within tolerance   
7: if any check fails then   
8: return quarantine   
9: end if   
10: if generality check finds a case-specific   
change then   
11: return quarantine or a scoped exception af  
ter review   
12: end if   
13: release $S ^ { \delta }$ with rollback record and predeces  
sor version

where $\pi _ { g }$ is the share of validation cases in group g and $\widehat { \Delta } _ { g }$ is their mean paired effect. The decompobsition distinguishes a scoped improvement from a change that redistributes correctness across traffic, and a large positive $\gamma ( \delta )$ signals that the development gain did not reproduce on independent cases. The recovery criterion below turns this signal into a release condition.

Release gates. A candidate is eligible only if the lower confidence bound of its in-scope gain is positive, recovery on exposed validation failures reaches at least half the recovery observed on development failures, effects on unexposed cases stay within unchanged-system rerun variation, and neither out-of-scope correctness nor answer coverage drops by more than three percentage points. Algorithm 1 states the procedure. Candidates that fail any check stay quarantined with their evidence.

Generality and rollback. The generality check asks whether a repair addresses a reusable condition or mechanism across affected player states and rephrased requests. Corrections tied to specific entities, case identifiers, or the wording of the triggering interaction remain quarantined, while evidence-backed corrections to narrowly applicable mechanics can be accepted as scoped exceptions after expert confirmation. An accepted candidate becomes a release bundle that records its evidence, reported and validated gains, versions, and rollback predecessor. Dependent edits share a release boundary, and a later hard-constraint violation rolls the bundle back to that predecessor (Sculley et al., 2015; Breck et al., 2017).

Coupled edits. When a candidate changes both knowledge K and execution $E ,$ the control plane measures the combined effect and, when feasible, each part separately. The interaction $I \ =$ $\widehat { \Delta } _ { K + E } - \widehat { \Delta } _ { K } - \widehat { \Delta } _ { E }$ captures the departure from b b badditive effects on matched cases. A joint candidate has to pass the gates as a compatible unit, and Section 5.4 reports a measured interaction of this kind.

## 5 Evaluation

We evaluate TactiNavi on recorded player interactions and on controlled replay. The evaluation asks whether condition-bearing knowledge improves correctness, whether traces support diagnosis of real complaints, and whether the release gate separates durable revisions from changes that only look like improvements.

## 5.1 Setup

Deployment and data. We instantiate TactiNavi as the coaching assistant of a commercial openworld game with strategic battles, where players assemble battle lineups from owned characters and ask for guidance on strategies and mechanics. The evaluation uses logged requests, histories, player states, tool outputs, and reply contexts. Player and session identifiers are de-identified before analysis, and the controlled comparisons, condition variants, and planted revisions run as offline replays of frozen traces that never reach players. The serving path and its traces ran in production throughout the window, and the two revision rounds shipped to players. The release checks of Section 4.5 were not part of those rounds, so we apply them afterwards with criteria fixed in advance and ask how many shipped revisions they would have admitted. Periods and replay. Two revision rounds produce three system versions, $S _ { 1 } , S _ { 2 }$ , and $S _ { 3 }$ , over consecutive two-day periods $T _ { 1 } , T _ { 2 }$ , and $T _ { 3 }$ . Both rounds use preceding interactions for development and leave $T _ { 3 }$ sealed. The first round mainly changes reply-stage rules and the second mainly updates game knowledge, and together they comprise 15 revision units grouped by the strategic sub-intent packages they affect. We sample 82 sessions per period, which yields 1,122 coaching turns, 360 of them in $T _ { 3 }$ . Controlled comparisons replay the target reply while holding requests, histories, player states, planning decisions, and tool outputs fixed, and only knowledge and reply-stage rules follow the configuration under test. Each configuration generates two responses per turn, so that reruns of an unchanged system measure generation variation.

Models and measures. A self-hosted DeepSeek-V4-Flash model (DeepSeek-AI, 2026) generates responses and provides judgments over shuffled, configuration-blind outputs against versioned game evidence. A response is correct when it contains no contradicted claims and passes executable checks on owned characters, available skills, counter relations, and resource-dependent abilities. It passes when it is also instructive and averages at least 1.5 across applicable relevance, explanation (Q2), and actionability (Q3) scores, each ranging from 0 to 2.

Annotation and statistics. Game-domain experts annotate 294 turns from $T _ { 1 }$ and T<sub>2</sub>, reaching $\kappa = 0 . 6 9$ on correctness. After calibration on $T _ { 2 }$ the judge agrees with expert correctness labels on 85.7% of 154 held-out $T _ { 1 }$ turns when executable checks are included. The rubric is fixed after calibration and applied unchanged across replayed configurations. We report 95% confidence intervals from session-clustered bootstrap resampling and use exact McNemar tests for stable paired outcomes.

## 5.2 Strategic Knowledge and Applicability

We compare four knowledge configurations on 197 strategic turns from $T _ { 3 }$ that use strategy packages. All four share the player state, tool outputs, reply-stage rules, and response model, so differences come from the knowledge form alone. Atomic facts supplies character and mechanic information. An unstructured guide expresses the content of a package and its rule outputs as prose without explicit condition markers. The strategy package keeps structured applicability conditions, and the cond. rules ablation removes conditional-rule outputs while retaining the rest of the package.

We also construct 108 matched player-state variants from 36 turns whose strategies depend on a lineup-wide resource requirement, making each requirement true, false, or unknown. Guides keep their original text, while packages re-evaluate the affected condition. A violation occurs when advice applies a strategy under a false condition, withholds it under a true condition, or treats an unknown condition as settled.

Table 2: Knowledge configurations on $T _ { 3 }$ strategic turns. Corr., Pass, and Viol. are percentages. Viol. is measured on false and unknown variants. Contr. counts contradicted claims and violations. Q2 and Q3 range from 0 to 2. Tok. counts knowledge tokens.
<table><tr><td>Knowledge</td><td>Corr.</td><td>Contr.</td><td>Q2</td><td>Q3</td><td>Pass</td><td>Tok.</td><td>Viol.</td></tr><tr><td>Atomic facts</td><td>57.9</td><td>0.91</td><td>1.02</td><td>1.21</td><td>41.9</td><td>767</td><td>10.4</td></tr><tr><td>Guide</td><td>82.5</td><td>0.28</td><td>1.27</td><td>1.43</td><td>62.7</td><td>1298</td><td>18.1</td></tr><tr><td>Package</td><td>82.0</td><td>0.28</td><td>1.32</td><td>1.44</td><td>65.2</td><td>1477</td><td>10.4</td></tr><tr><td>— cond. rules</td><td>75.6</td><td>0.43</td><td>1.19</td><td>1.43</td><td>61.4</td><td>1246</td><td>21.5</td></tr></table>

Table 2 shows that the package reaches 82.0% correctness against 57.9% for atomic facts, a 24.1- point gain with a 95% CI of [16.4, 31.7] and $p ~ < ~ 0 . 0 0 1$ , and raises the pass rate from 41.9% to 65.2%. Explanation quality rises from 1.02 to 1.32 and actionability from 1.21 to 1.44. On the 75 turns that use conditional-rule outputs, removing those rules lowers correctness from 84.7% to 69.3% $( p < 0 . 0 0 1 )$ ).

The unstructured guide matches the package on recorded turns, at 82.5%. On the state variants with false or unknown conditions, however, it violates applicability more often than the package, at 18.1% against 10.4%, and removing conditional rules pushes violations to 21.5%. Relative to the guide, the package reduces violations by 7.6 points with a 95% CI of [2.1, 15.4], and relative to the ablation by 11.1 points with a 95% CI of [1.9, 22.8]. The guide comparison rests on 72 variant units from 36 base turns, and the exact McNemar test on stable pairs, with 5 discordant pairs favoring the package and none favoring the guide, gives $p \ = \ 0 . 0 6 2$ , so we read the violation reduction as consistent evidence of an effect rather than as a settled one. The gap is widest when the relevant state is unknown, where violations are 26.4% for the guide and 15.3% for the package. Atomic facts violate as rarely as the package, at 10.4%, because they carry no strategy that could be misapplied, and they pay for this with the lowest correctness on recorded turns. The comparison therefore separates two things that a single correctness number conflates. Synthesis improves what the system says, and explicit conditions determine whether it says it when the strategy applies.

Table 3: Failure diagnosis on expert-adjudicated $T _ { 1 }$ doubts. Prec. and Rec. are percentages for founded doubts. Rej. counts rejected unfounded doubts out of 12. Lag. is delayed-feedback localization (%).
<table><tr><td>Method</td><td>Prec.</td><td>Rec.</td><td>Rej.</td><td>Lag.</td></tr><tr><td>Doubt as failure</td><td>75.0</td><td>100.0</td><td>0/12</td><td>0.0</td></tr><tr><td>Direct feedback</td><td>75.6</td><td>94.4</td><td>1/12</td><td>0.0</td></tr><tr><td>Category routing</td><td>88.0</td><td>61.1</td><td>9/12</td><td>0.0</td></tr><tr><td>Who&amp;When</td><td>77.8</td><td>97.2</td><td>2/12</td><td>0.0</td></tr><tr><td>TactiNavi</td><td>86.4</td><td>52.8</td><td>9/12</td><td>26.7</td></tr><tr><td>— turn localization</td><td>80.8</td><td>58.3</td><td>7/12</td><td>0.0</td></tr></table>

## 5.3 Failure Diagnosis

We evaluate 130 player-doubt turns from 39 sessions in $T _ { 1 }$ , including 48 doubts adjudicated by experts, of which 36 are founded, meaning the disputed response was indeed wrong, and 12 unfounded. Precision and recall are computed over founded doubts, and a method rejects an unfounded doubt when it declines to call the response a failure. Baselines treat doubt as failure, judge the feedback directly, route failures into categories, or apply Who&When trace attribution (Zhang et al., 2025), and we also ablate turn localization. All methods use the same underlying model.

Table 3 reports 86.4% precision and 52.8% recall for TactiNavi, which rejects 9 of 12 unfounded doubts against 1 for direct feedback and 2 for Who&When. Category routing reaches 88.0% precision and 61.1% recall, so the trace-based method gives up a little precision and some recall in exchange for rejecting unfounded complaints and locating the responsible earlier response in 26.7% of delayed-feedback failures, which preceding-turn methods cannot do at all. Removing target-turn localization drops precision to 80.8%. Evidencebased screening and delayed-feedback localization are the two capabilities that the traces add.

## 5.4 Revision Gains and Release Validation

We assess the deployed revisions on the sealed $T _ { 3 }$ period. Replaying $S _ { 1 } , \ S _ { 2 } ,$ , and $S _ { 3 }$ on the same 360 turns shows that the final version does not differ measurably from the first, with a correctness difference of +0.6 percentage points between $S _ { 3 }$ and $S _ { 1 }$ , a 95% CI of [ 2.2, 3.4], and $p = 1 . 0$ . A factorial replay on 26 turns affected by jointly developed first-round revisions separates knowledgeonly updates, which gain +7.7 points, from the knowledge–execution interaction of 9.6 points with a 95% CI of [ 17.9, 2.0]. The combination performs below the sum of its parts, which is why dependent edits are released as a unit.

Table 4: Independent revision assessment on $T _ { 3 }$ . Rec. is recovery among previously failing turns for the unchanged rerun and among failing exposed turns for planted controls (%). Gain is the paired correctness difference in percentage points, evaluated in scope for planted controls. Spec. counts case-specific revisions, and Eligible counts those passing release checks. Unchanged reruns use $S _ { 1 }$ , while planted controls use $S _ { 3 }$
<table><tr><td>Revision</td><td>Units Rec.</td><td></td><td>Gain</td><td>Spec. Eligible</td></tr><tr><td>Unchanged rerun</td><td>一</td><td>38.4</td><td>+0.6 一</td><td>一</td></tr><tr><td>Deployed units</td><td>15</td><td>一</td><td>一 一</td><td>1/15</td></tr><tr><td>Planted, entity</td><td>3</td><td>39.1</td><td>-0.6 3/3</td><td>0/3</td></tr><tr><td>Planted, mechanism</td><td>3</td><td>40.7</td><td>+0.7 0/3</td><td>0/3</td></tr></table>

We then apply the release checks retrospectively to the 15 deployed revision units with criteria fixed in advance. Recovery, the share of previously failing turns that a configuration answers correctly, is the signal a development team would normally read as evidence of a repair, so we also measure it for an unchanged system. To test whether the checks distinguish case-specific corrections from reusable ones, we build six planted controls across three failure classes, namely recommending unavailable characters, misstating counter relations, and making unsupported turn-order claims. Each class receives an entity-specific correction and a mechanism-based correction of similar length. The candidates are inserted independently into the $S _ { 3 }$ reply-stage rules and assessed on $T _ { 3 }$

Only one deployed unit satisfies the release criteria. Thirteen show no gain beyond generation variation. The remaining unit improves in-scope correctness by 25.0 points while reducing out-ofscope correctness by 20.8 points, a regression that targeted evaluation would have missed and that the scope decomposition in Eq. 3 exposes directly.

Table 4 makes the noise floor visible. Rerunning the unchanged system recovers 38.4% of previously failing turns, close to the 39.1% and 40.7% recovery of the planted revisions, yet the planted controls produce paired in-scope gains of only 0.6 and +0.7 points and all six fail gain reconciliation. Recovery on failing turns is therefore not evidence of a repair. The gate requires a positive in-scope gain before release, and that requirement alone removes every planted candidate.

The generality check classifies the three entityspecific revisions as case-specific and the three mechanism-based revisions as general, reaching 85.7% agreement with expert labels over 21 units at $\kappa = 0 . 5 6$ . Mechanism-based controls are classified as general even though they fail gain reconciliation, which shows that the two checks answer different questions. Gain reconciliation asks whether correctness improved, and generality asks whether the correction extends beyond the cases that triggered it.

## 5.5 Cost and Serving Behavior

Reply generation on the 360 $T _ { 3 }$ turns has a median of 2.07 s and a 95th percentile of 5.48 s, with a mean of 2.47 s. The 269 strategy turns are somewhat slower, at 2.42 s median and 5.60 s at the 95th percentile, which is consistent with their longer replies. Condition evaluation runs as rule checks over the structured snapshot and issues no model calls, so the measured latency is dominated by generation rather than by the applicability machinery.

The knowledge representation has a price that is visible in context size. A package costs 1,477 knowledge tokens against 767 for atomic facts and 1,298 for the prose guide, so explicit conditions add 14% over the guide while cutting violations by 7.6 points on state variants. The ablation shows the reverse trade, at 1,246 tokens with violations of 21.5%, because conditional rules are cheaper to omit than to evaluate.

Replay is what keeps validation affordable. Only the target reply is regenerated, with planning decisions and tool outputs held fixed, so a candidate can be compared against the current system on hundreds of held-out turns without re-executing the full pipeline. The same mechanism gives the release gate its control condition, because an unchanged system can be replayed under identical inputs and its variation used as the null.

## 6 Lessons from Deployment

Operating TactiNavi taught us four lessons that we expect to carry over to other LLM agents whose answers depend on live user state and on a knowledge base that changes under them.

Lesson 1: Applicability deserves to be an object. The comparison between the prose guide and the strategy package carries the central design argument. The two forms are indistinguishable on recorded turns, at 82.5% and 82.0%, and they diverge by 7.6 points once the same strategies meet changed player state. Content quality and applicability control are therefore separable engineering goals, and a system that optimizes only the first will look finished while it silently misapplies what it knows. Making conditions explicit costs 14% more context than prose and buys a fallback path for missing state, a target for diagnosis, and a scope for release.

Lesson 2: Traces must be cheap on the serving path and sufficient off it. The trace is the only artifact that survives the request, and it carries the evidence, the condition outcomes, and the rules that produced the response. Because it is written on the serving path, it cannot hold everything. Because diagnosis happens later and often about a different turn, it must hold enough to reconstruct why a claim was made. The 26.7% delayedfeedback localization and the rejection of 9 of 12 unfounded doubts both depend on that record, and neither is available from the conversation alone.

Lesson 3: Independence belongs in the system, not in the process. An unchanged system recovers 38.4% of failing turns, which is close to what the planted revisions recover. Any workflow that accepts a candidate because the reported case now passes will ship changes that do nothing, and 13 of our own 15 shipped units fit that description. Logged correctness stayed at 83.1%, 84.4%, and 84.4% across the three periods even though two revision rounds shipped in between (Appendix C.3). The control plane treats an unchanged rerun as the null and requires a positive in-scope gain before release, which is a strict rule and eliminates all six planted controls. The rule is affordable because replay regenerates only the reply, so validating a candidate on hundreds of held-out turns costs one generation per turn.

Lesson 4: Coupled knowledge and execution need a shared release boundary. Knowledge edits create prerequisites that rules must evaluate, and rules encode assumptions that packages rely on. The measured interaction of 9.6 points on jointly developed changes shows that the two parts cannot be optimized or released in isolation. The dependency closure gives the pair a single boundary, which also makes rollback well defined.

Limitations. We evaluate one commercial game across two revision rounds, so the transfer to games with different mechanics and to longer revision sequences remains open. The state-variant comparison between packages and guides uses 36 base turns, which leaves the violation reduction short of conventional significance. Replay holds player states, planning decisions, and tool outputs fixed, which characterizes revisions to knowledge and reply rules rather than to the whole pipeline. Expert-attributed failures concentrate in reply execution, so attribution performance on other components is not measured here. The automated judge shares the Coach’s underlying model, which makes evaluation sensitive to shared judgment errors. The reported latency covers reply generation and not the complete online workflow.

## 7 Conclusion

We described TactiNavi, the serving system behind a deployed LLM game coach, which treats strategy applicability as an explicit artifact and carries it through compilation, serving, diagnosis, and release. Strategy packages with evidencegrounded content and three-valued conditions raise correctness over atomic facts and reduce misapplication when player state changes, while decision traces make late and ambiguous complaints diagnosable and condition–rule dependencies give a revision a scope. The deployment showed that most shipped revisions did not survive independent measurement, and that a release gate which rejects them costs one generation per held-out turn because the rest of the pipeline is held fixed. We expect the same three ingredients, namely explicit applicability, diagnosis-ready traces, and a replayed null model at release, to matter for other agents that serve state-dependent advice from a changing knowledge base.

## References

Ryogo Akie, Yunhao Tu, Kazuyuki Murase, and Fady Alnajjar. 2026. Game coaching system using 3d cnn and llm. In 2026 IEEE 2nd International Conference on Consumer Technology (ICCT-Pacific), pages 594–597. IEEE.

Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. 2024. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In International conference on learning representations, pages 9112–9141.

Yonas Atinafu and Robin Cohen. 2026. Rewardhackingagents: Benchmarking evaluation integrity for llm ml-engineering agents. In 2026 IEEE 42nd International Conference on Data Engineering Workshops (ICDEW), pages 27–34. IEEE.

Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, and D Sculley. 2017. The ml test score: A rubric for ml production readiness and technical debt reduction. In 2017 IEEE international conference on big data (big data), pages 1123–1132. IEEE.

Konstantina Christakopoulou, Shibl Mourad, and Maja Mataric. 2024. Agents thinking fast and slow:´ A talker-reasoner architecture. arXiv preprint arXiv:2410.08328.

DeepSeek-AI. 2026. Deepseek-v4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. 2024. Critic: Large language models can self-correct with tool-interactive critiquing. In International Conference on Learning Representations, pages 57734– 57811.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. In International conference on machine learning, pages 3929– 3938. PMLR.

Robin Horst, Micha Lanvers, László von Kacsoh, and Ralf Dörner. 2021. Moba coach: Exploring and analyzing multiplayer online battle arena data. In International Symposium on visual computing, pages 197–209. Springer.

Sihao Hu, Tiansheng Huang, and Ling Liu. 2024. Pokellmon: A human-parity agent for pokemon battles with large language models. arXiv preprint arXiv:2402.01118.

Rui Hua, Zhaoyu Huang, Jinhao Lu, Yakun Li, and Na Zhao. 2026. Exincoach: Strategic exploration meets interactive tutoring for context-aware game onboarding. Information Fusion, page 104151.

Ken Huang and Jerry Huang. 2025. Audited skillgraph self-improvement for agentic llms via verifiable rewards, experience synthesis, and continual memory. arXiv preprint arXiv:2512.23760.

Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented generation. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 7969–7992.

Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. 2024. Dspy: Compiling declarative language model calls into state-ofthe-art pipelines. In International Conference on Learning Representations, pages 54928–54958.

Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W Mahoney, Kurt Keutzer, and Amir Gholami. 2023. An llm compiler for parallel function calling. arXiv preprint arXiv:2312.04511.

Hanbyeol Lee, Erica Kleinman, Namsub Kim, Sangbeom Park, Casper Harteveld, and Byungjoo Lee. 2025. Crafting champions: an observation study of esports coaching processes. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, pages 1–20.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive nlp tasks. Advances in neural information processing systems, 33:9459–9474.

Qi Liu, Xiaoyang Yuan, Yubin Ruan, Zhuomeng Zhang, Wenjin Wang, Di Wu, Mingye Xu, Xinyi Mou, Xingxi Yin, Ke Feng, et al. 2026. Stategrounded conditioning: Wrapping user-facing LLM agents where direction depends on live state. arXiv preprint arXiv:2609.27606.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, and 3 others. 2024. Agentbench: Evaluating llms as agents. In International Conference on Learning Representations, pages 52989–53046.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: Nlg evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 2511–2522.

Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. 2024. Agentboard: An analytical evaluation board of multi-turn llm agents. Advances in neural information processing systems, 37:74325–74362.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. Advances in neural information processing systems, 36:46534–46594.

Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2024. Gaia: a benchmark for general ai assistants. In International Conference on Learning Representations, pages 9025–9049.

Reid Pryzant, Dan Iter, Jerry Li, Yin Lee, Chenguang Zhu, and Michael Zeng. 2023. Automatic prompt optimization with “gradient descent” and beam search. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 7957–7968.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2024. Toolllm: Facilitating large language models to master 16000+ real-world apis. In International Conference on Learning Representations, pages 9695–9717.

Anton Razzhigaev, Andrei Gritsaev, Andrei Kaznacheev, Nikita Dragunov, Roman Yampolskiy, and Andrei Kuznetsov. 2026. Ouroboros: A selfdeveloping frontier coding agent with reviewed core evolution. arXiv preprint arXiv:2608.08311.

Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher Manning. 2024. Raptor: Recursive abstractive processing for tree-organized retrieval. In International Conference on Learning Representations, pages 32628– 32649.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539–68551.

David Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-Francois Crespo, and Dan Dennison. 2015. Hidden technical debt in machine learning systems. Advances in neural information processing systems, 28.

Yijia Shao, Yucheng Jiang, Theodore Kanell, Peter Xu, Omar Khattab, and Monica Lam. 2024. Assisting in writing wikipedia-like articles from scratch with large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6252–6278.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652.

Theodore R Sumers, Shunyu Yao, Karthik Narasimhan, and Thomas L Griffiths. 2023. Cognitive architectures for language agents. arXiv preprint arXiv:2309.02427.

Kunvar Thaman. 2026. Reward hacking benchmark: measuring exploits in llm agents with tool use. arXiv preprint arXiv:2605.02964.

Chih Hui Wang, Mengdie Tu, Qianyun Zhang, Wei Wu, Lili Zhou, Mingqi Shen, and Changshuai Wei. 2026. Self-evolving agentic customer support system at linkedin. arXiv preprint arXiv:2608.10224.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291.

Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Ma, and Yitao Liang. 2024. Describe, explain, plan and select: Interactive planning with large language models enables open-world multitask agents. In Proceedings of the 37th International Conference on Neural Information Processing Systems. Red Hook: Curran Associates Inc, pages 34153–34189.

Siwei Wu, Jincheng Ren, Yizhi Li, Haau-Sing Li, Chengran Yang, Xudong Ma, Yuxuan Zhang, Weicheng Gu, Jian Yang, Riza Batista-Navarro, Chuanyi Zhang, Xianglong Liu, Ming Zhou, Bryan Dai, and Chenghua Lin. 2026. Modularrsi: Modular and generalizable recursive harness self-improvement. arXiv preprint arXiv:2609.14857.

Yue Wu, So Yeon Min, Shrimai Prabhumoye, Yonatan Bisk, Russ R Salakhutdinov, Amos Azaria, Tom M Mitchell, and Yuanzhi Li. 2023. Spring: Studying papers and reasoning to play games. Advances in Neural Information Processing Systems, 36:22383– 22687.

Peng Xia, Rujun Han, Zifeng Wang, Yanfei Chen, Yufan Zhuang, Yoonho Lee, Chengsong Huang, Han Yu, Zhongying CuiZhu, Yifei Ming, Huaxiu Yao, Burak Gokturk, Tomas Pfister, and Chen-Yu Lee. 2026. Rrsi: Regularized recursive selfimprovement of agent harnesses. arXiv preprint arXiv:2609.24972.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

Xiaoyang Yuan, Qi Liu, Yubin Ruan, Xinyi Mou, Zhuomeng Zhang, Wenjin Wang, Hanying Jiao, Di Wu, Mingye Xu, Yi Bin, et al. 2026. DAAF: From failure localization to editable system assets in LLM agents. arXiv preprint arXiv:2609.32498.

Shaokun Zhang, Ming Yin, Jieyu Zhang, Jiale Liu, Zhiguang Han, Jingyang Zhang, Beibin Li, Chi Wang, Huazheng Wang, Yiran Chen, and Qingyun Wu. 2025. Which agent causes task failures and when? on automated failure attribution of llm multiagent systems. arXiv preprint arXiv:2505.00212.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena.

Advances in neural information processing systems, 36:46595–46623.

Leijun Zhou, Zhihao Liu, Xiang Qu, Chenxu Liu, Yifei Liu, Yanke Yu, Jingzhe Xu, Xuejun Wu, Buyue Qian, Xi Chen, Yaowei Zheng, and Junhao Hu. 2026. Gdpevo: Evaluating agent self-evolution on real business tasks. arXiv preprint arXiv:2608.03764.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. 2024. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, pages 15585–15606.

## A Component Details

This section describes how applicability conditions support evidence-chain attribution, condition-aware validation, and the assessment of proposed repairs.

## A.1 Evidence-chain attribution

Attribution starts from a contradicted claim and follows the recorded evidence chain from the reply toward its sources. Table 5 lists the stages examined and the repair owner of each. The first stage that lacks support identifies the implicated component. A supported claim requires no repair, and missing evidence leaves attribution unresolved for clarification or expert review. Condition identifiers link a disputed claim to its prerequisites and to the rules that evaluate them, which lets related knowledge and execution targets be assessed together.

## A.2 Condition-aware validation

Validation cases follow the applicability conditions that a revision modifies and the execution rules that consume them. The control plane retrieves held-out interactions involving these dependencies and builds matched counterexamples by changing the relevant player-state fields while preserving the request and unrelated state. For a strategy that requires an eligible role, the variants represent players with an eligible owned character, players without one, and players whose ownership information is unavailable. These states produce true, false, and unknown outcomes, and resource and version requirements are tested across their relevant boundaries. Rephrased requests vary the original wording while preserving intent and constraints.

Expected behavior is established independently of the proposed revision, using versioned game rules and expert-approved strategy conditions. A satisfied condition permits the strategy, a violated condition calls for an alternative that explains the unmet requirement, and an unknown condition requires additional evidence or clarification. Previously successful interactions involving the affected knowledge or rules provide regression cases, so the validation set measures both the targeted correction and its effect on existing valid guidance.

Table 5: Evidence-chain stages used for failure attribution. The first unsupported stage determines the implicated repair owner.
<table><tr><td>Stage</td><td>Unsupported when</td><td>Repair owner</td></tr><tr><td>Reply rendering</td><td>The response misstates or exceeds its reply context</td><td>Reply execution</td></tr><tr><td>Context assembly</td><td>Available relevant evidence is omitted from the reply context</td><td>Reply execution</td></tr><tr><td>Result merging</td><td>A correct result is lost when tool outputs are combined</td><td>Upstream execution</td></tr><tr><td>Routing</td><td>A required tool or knowledge source is not selected</td><td>Upstream execution</td></tr><tr><td>Tool input</td><td>A tool receives incorrect arguments</td><td>Upstream execution</td></tr><tr><td>Tool output</td><td>A tool omits information available from its source</td><td>Upstream execution</td></tr><tr><td>Knowledge source</td><td>Required knowledge is incorrect, missing, or inaccessible</td><td>Knowledge</td></tr></table>

## A.3 Interpreting a failed reconciliation

The release checks separate distinct sources of apparent improvement. A large positive reconciliation gap means that the reported development gain exceeds the effect reproduced on independent cases. Changes on unexposed cases beyond unchanged-system variation call for inspecting sampling effects, scorer sensitivity, and changes outside the declared scope. A decline in out-ofscope correctness identifies a regression, and reduced answer coverage or increased deferral suggests that a candidate avoids difficult recommendations. These signals guide further examination of quarantined proposals.

## A.4 Generality and scoped exceptions

The generality check considers how a repair is expressed and how its effects extend across relevant interactions. A reusable correction specifies a prerequisite, role, or mechanism that applies across player states within its declared scope, while changes tied to entities, case identifiers, or the wording of triggering interactions are tested with rephrased requests and condition-derived counterexamples.

Some game mechanics legitimately apply only to particular characters or circumstances. Such corrections can be registered as scoped exceptions when their evidence and applicability conditions receive expert confirmation, and the release record preserves the accepted scope and supporting evidence. Other case-specific changes remain quarantined.

## B Evaluation Protocol

## B.1 Sampling

We define coaching turns as responses to players’ strategic or knowledge requests. Standalone recommendations, casual conversation, refusals, and safety-filtered interactions are excluded. From each consecutive two-day deployment period we sample 82 sessions at random and keep all coaching turns within them, which yields 1,122 turns split as 397, 365, and 360 across the three periods. Earlier turns stay available as dialogue history when later responses are replayed. Revision decisions and release tolerances were fixed before the sealed period was evaluated.

## B.2 Expert annotation

Game-domain experts annotate interaction histories using player state, retrieved strategy packages, rule outputs, tool results, and versioned game evidence, with system versions and deployment periods hidden. Labels cover correctness, instructiveness, responsible components, acceptable repair targets, and the validity and target of player feedback. Substantive errors include claims contradicted by game mechanics or player resources and omitted interactions that change a recommended decision, while evasive or generic responses are labeled non-instructive. Repair ownership follows Table 5 and may name several targets when more than one change could resolve a failure.

Annotation quality is measured by Cohen’s κ of 0.69 for correctness, 0.84 for instructiveness, 0.64 for repair ownership, and 0.77 for the validity of player doubt.

## B.3 Judge protocol

Evaluation combines response-level judgments, executable checks, and recorded process evidence, following agent evaluation protocols for task completion (Liu et al., 2024; Zhou et al., 2024; Mialon et al., 2024) and intermediate progress (Ma et al., 2024). A fixed self-hosted DeepSeek-V4-Flash judge (DeepSeek-AI, 2026) applies structured criteria (Liu et al., 2023) to responses that are shuffled without configuration labels. Its rubric was selected on a calibration set from the second period and held fixed for later evaluation and for all system comparisons.

Reference evidence includes the recorded reply context, tool outputs, knowledge snapshots, and game data from the relevant period. When the recorded context conflicts with game-data tables, the protocol gives precedence to the recorded context because it preserves the version-specific information supplied during that interaction. Replayed responses are judged against that context and the target period’s rules. Claims inconsistent with the reference evidence are contradicted, including incorrect counter relations and omitted skill interactions that change the conclusion, and outcomes that the available calculations cannot establish are labeled unknown.

Correctness requires no contradicted claims and satisfaction of four executable constraints. Recommended characters must be available in the player’s lineup, instructed skills must belong to the selected characters, stated counter relations must agree with the reference table, and resourcedependent abilities must be supported by the player’s resources. Instructiveness is assessed separately, and evasive answers, generic guidance, and unnecessary clarification receive no credit on the applicable quality dimensions. A response passes when it is correct and instructive and averages at least 1.5 across applicable relevance, explanation, and actionability scores. Requests that ask for a numerical calculation are not penalized for limited strategic explanation when the calculation answers the request.

On the held-out evaluation set, the judge agrees with expert correctness labels on 89.0% of turns $( \kappa ~ = ~ 0 . 4 8 )$ and with instructiveness labels on 86.4% $\begin{array} { r l r } { ( \kappa } & { { } = } & { 0 . 5 6 ) } \end{array}$ Incorrect-response rates are 11.7% for the judge and 12.3% for experts, and error-detection precision and recall are 55.6% and 52.6%. Adding executable checks yields

85.7% correctness agreement $( \kappa ~ = ~ 0 . 4 2 )$ , and the knowledge-form comparisons retain their direction under judge-only correctness labels.

Bias checks follow Zheng et al. (2023). Placing reference evidence before the response leaves correctness agreement at 89.0% $( p ~ = ~ 1 . 0 )$ , and false alarms are more frequent for longer responses, reaching 15.9% in the length analysis, while shorter responses account for more missed errors. Each response is generated twice. Outcomes that are consistent across repetitions are treated as stable for paired tests, and cases with low-confidence judgments or missing evidence remain in the denominator with unresolved status.

## B.4 Knowledge forms and condition variants

The knowledge-form comparison uses 197 strategic turns from 69 sessions in the sealed period, with shared request, player state, tool outputs, reply-stage rules, and response model. Atomic facts supplies relevant versioned character attributes, skills, and counter relations. The unstructured guide rewrites packages, rule outputs, and instructional content as continuous prose, preserving facts, values, and recommendations while removing explicit condition markers and structural headings. Manual review of a sample of rewrites finds no missing facts or conditions, and the few that omit a presentation instruction do not change the comparisons when excluded. The cond. rules ablation removes conditional rule outputs and keeps the remaining package content. Knowledge tokens count reply-prompt tokens beyond the shared context.

For the applicability evaluation we select 36 turns whose strategies depend on a lineup-wide resource and construct three matched player states per turn, in which the requirement is satisfied, violated, or unavailable for evaluation. The static guide keeps its original text while the package recomputes the applicable rule outputs. Each configuration generates two responses per variant, and a separate judgment with the same underlying model assesses violations against conditiondependent behavior specified before evaluation.

## B.5 Replay and revision units

Replay reconstructs a target reply from its recorded request, dialogue history, player state, planning decision, and tool outputs. The configuration under test supplies knowledge content and reply-stage rules, knowledge unavailable in that version is excluded, and logged tool explanations and derived game-data tables remain fixed. Only the target reply is regenerated, with earlier responses preserved as recorded, and a fidelity check verifies that logged rules and retrieved knowledge appear in the snapshot of the corresponding deployed version.

Changes from the two deployed rounds are grouped into 15 revision units by the strategic subintent packages they affect. Shared reply instructions and basic-knowledge changes form common units that cover the relevant coaching turns, and development cases are earlier failures attributed to a unit or judged incorrect by experts. Each unit is assessed retrospectively on the sealed set.

Release eligibility requires a positive 95% lower confidence bound for in-scope correctness gain. Recovery among later exposed failures must reach at least half the recovery observed on development failures, changes on unexposed cases must stay within unchanged-system rerun variation, and neither out-of-scope correctness nor answer coverage may decline by more than three percentage points. Dependent knowledge and execution changes are evaluated as a combined candidate.

For the planted controls we select three failure classes from the second period, each with at least two triggering cases, namely recommending an unavailable character, misstating a counter relation, and asserting a turn-order advantage without supporting evidence. Each class receives an entity-specific correction and a mechanism-based correction of similar length, and the six candidates are inserted separately into the reply-stage rules of the third version. Expected specificity labels were fixed before evaluation, the generality evaluator applies a fixed rubric with a majority decision over three runs, and experts label all 21 planted and deployed units without knowing which candidates were planted.

## B.6 Statistical analysis

We estimate 95% confidence intervals from 10,000 session-level bootstrap resamples that preserve all turns and repeated generations within a sampled session. Exact McNemar tests use paired turns with stable outcomes across both generations. Reported gains are also compared with unchanged-system rerun differences, and Holm correction is applied to prespecified families of multiple comparisons. Reply generation on the sealed period has a median of 2.07 s and a 95th percentile of 5.48 s, and the measurement covers generation rather than the complete online workflow. Reproduction materials include model identifiers, prompts, evaluation rubrics, and revision-unit definitions, subject to restrictions on releasing player interaction records.

## C Additional Results

## C.1 Failure diagnosis

The diagnosis set contains 130 player-doubt turns from 39 sessions in the first period, with 48 doubts adjudicated by experts, of which 36 are founded and 12 unfounded. Doubt as failure treats the immediately preceding turn as erroneous. Direct feedback judges the request, response, and subsequent player utterance. Category routing predicts a failure category and maps it to a component. Who&When (Zhang et al., 2025) applies single-pass trace attribution across planning, toolcall, tool-result, and reply stages. TactiNavi verifies claims and locates the targeted turn, and its ablation considers only the preceding response.

Category routing reaches 88.0% precision and 61.1% recall, against 86.4% and 52.8% for Tacti-Navi. The latter rejects 9 of 12 unfounded doubts and leaves 36.9% of flagged cases unresolved when trace evidence is insufficient. Target-turn localization identifies 26.7% of delayed-feedback failures, which the preceding-turn ablation misses entirely. On separately evaluated repair attribution, ownership macro-F1 is 18.8% and repair-target Hit@1 is 36.7%, and reply execution accounts for 28 of the 30 expert-attributed failures.

## C.2 Condition variants

Table 6 reports violations separately for false, unknown, and true conditions. Unknown conditions produce the largest difference between the static guide and the package, at 26.4% against 15.3%. Removing conditional rules yields a 19.4% violation rate under false conditions. Under true conditions every configuration keeps the strategy in more than 93% of responses. Atomic facts and the package show the same violation rates in this controlled test even though their correctness on recorded strategic turns differs substantially, which is the separation that the knowledge-form comparison of the main paper quantifies.

Table 6: Strategy-condition violation rates (%) by condition outcome on 108 matched player-state variants.
<table><tr><td>Knowledge</td><td>False</td><td>Unknown</td><td>True</td></tr><tr><td>Atomic facts</td><td>5.6</td><td>15.3</td><td>4.2</td></tr><tr><td>Unstructured guide</td><td>9.7</td><td>26.4</td><td>5.6</td></tr><tr><td>Strategy package</td><td>5.6</td><td>15.3</td><td>4.2</td></tr><tr><td>— cond. rules</td><td>19.4</td><td>23.6</td><td>6.9</td></tr></table>

## C.3 Deployed-version comparisons

Replaying the three versions on the same 360 sealed turns, with two generations per configuration, gives a paired correctness difference of +0.6 points from the first to the third version, with a 95% CI of [ 2.2, 3.4] and exact McNemar $p =$ 1.0. On the 72 turns exposed to second-round changes, the third version exceeds the second by 2.8 points with a 95% CI of [ 4.2, 10.3]. Neither comparison establishes a significant improvement.

Logged-response correctness across the three periods is 83.1%, 84.4%, and 84.4%, and an ordered-period analysis gives an odds ratio of 1.05 $( p ~ = ~ 0 . 6 6 )$ . The pass rate rises from 51.4% to 55.8% $( p = 0 . 3 0 )$ , while the non-instructive rate falls from 16.6% to 12.8% $( p ~ = ~ 0 . 2 2 )$ . In a blinded comparison of 60 turns, expert preferences for the third version over the first are 44.0% for correctness, 36.3% for instructional quality, and 38.8% overall. These observational and expert comparisons likewise do not establish a significant improvement across the deployed revisions.

## C.4 Release checks and generality

Retrospective assessment finds that one of 15 deployed units satisfies the eligibility criteria. Thirteen units lack sufficient gain beyond unchangedsystem variation. The remaining unit improves inscope correctness by 25.0 points while reducing out-of-scope correctness by 20.8 points, which exceeds the regression tolerance and shows why both affected and unaffected traffic are measured.

An unchanged rerun of the first version recovers 38.4% of previously failing turns with a paired difference of +0.6 points. The planted entity-specific and mechanism-based controls recover 39.1% and 40.7%, while their mean paired in-scope differences are 0.6 and +0.7 points, so all six fail independent gain requirements. The generality check classifies all three entity-specific controls as case-specific and all three mechanism-based controls as general. Agreement with independent expert labels over all 21 planted and deployed units is 85.7% $( \kappa ~ = ~ 0 . 5 6 )$ , and expert classifications match five of the six prespecified planted labels.

## C.5 Knowledge-execution effects

A factorial replay evaluates first-round knowledge revisions alone, execution-rule revisions alone, and their combination on the same frozen inputs. Among 26 turns exposed to jointly developed changes, knowledge-only updates yield a +7.7- point difference. The interaction term is 9.6 points with a 95% CI of $[ - 1 7 . 9 , - 2 . 0 ]$ , so the combined effect falls below the sum of the separately measured effects on this subset. The result supports joint assessment of dependent edits while keeping their contributions separately measurable.