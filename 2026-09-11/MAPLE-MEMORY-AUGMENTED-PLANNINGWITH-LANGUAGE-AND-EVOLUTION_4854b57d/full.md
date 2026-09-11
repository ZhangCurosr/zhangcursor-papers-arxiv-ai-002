![](images/2611135577b6d045f63a545ee84fd9bca4679ce9f369214c7b11925b3d24bbfa.jpg)

# MAPLE: MEMORY-AUGMENTED PLANNINGWITH LANGUAGE AND EVOLUTION

Research https://github.com/xin8coder/MAPLE

Application https://github.com/xin8coder/MAPLE-harness-core

Live demo https://xin8coder.github.io/MAPLE-harness/

Kesheng Chen Yamin Hu Wenjian Luo Harbin Institute of Technology, Shenzhen, China

## ABSTRACT

Domain practitioners understand their business constraints but may lack operations-research expertise or dedicated support. LLM-based optimization agents translate natural-language requirements into models or solver programs that established optimization tools can execute. This progress makes optimization more accessible, but real-world operations are dynamic: changing demand, resources, and priorities require updates to data, constraints, and objectives. Methods centered on isolated requests offer limited support for rapid adaptation that preserves earlier decisions and reuses useful search results. We introduce MAPLE (Memory-Augmented Planning with Language and Evolution), an agent for maintaining optimization problems through successive natural-language requests. MAPLE combines language-based problem construction with mathematical programming and evolutionary search. It retains the optimization program, accepted plans, earlier updates, and candidate solutions for subsequent requests. We introduce NLDO, a benchmark of 15 trajectories and 180 updates spanning selection, scheduling, rostering, routing, and cloud-resource placement. In the main evaluation, MAPLE completes all trajectories and achieves online scalar quality of 0.951 and a Pareto hypervolume ratio of 0.875. Controlled comparisons further show that maintaining executable state improves update validity and can preserve useful search information across substantial revisions.

![](images/c637e7205feffcbf0333e546a7ebf2714140bc469e978f48b1cb9a4d6d329063.jpg)

<table><tr><td>Natrve capabmities ad evaluateu adaptatlomis Method</td><td>Natural language</td><td>Solver output</td><td>Retained Pareto set</td><td>Live updates</td><td>Solver state reuse</td></tr><tr><td>ReAct</td><td>Yes</td><td>Yes</td><td>No</td><td>Part. 水</td><td>No</td></tr><tr><td>OR-LLM-Agent</td><td>Yes</td><td>Yes</td><td>No</td><td>Part. 水</td><td>No</td></tr><tr><td>ORLM</td><td>Yes</td><td>Yes</td><td>No</td><td>Part. 宗</td><td>No</td></tr><tr><td>OptiMUS</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td><td>No</td></tr><tr><td>OptimAI</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Persistent ReAct</td><td>Yes</td><td>Yes</td><td>No</td><td>Yes*</td><td>Part. *</td></tr><tr><td>MAPLE (ours)</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr></table>

<table><tr><td>Method</td><td>NLP4LP. hard</td><td>BWOR20</td><td>NLDO SS</td><td>NLDO DS</td><td>NLDO SM</td><td>NLDO DM</td></tr><tr><td>ReAct</td><td>57.6%</td><td>70.0%</td><td>0.889 水</td><td>0.333 兴</td><td>0.300</td><td>0.023</td></tr><tr><td>OR-LLM-Agent</td><td>50.8%</td><td>65.0%</td><td>0.778 水</td><td>0.111 兴</td><td>0.440 *</td><td>0.046 *</td></tr><tr><td>ORLM</td><td>23.7%</td><td>45.0%</td><td>1.000 水</td><td>0.120 水</td><td>0.000*</td><td>0.000 *</td></tr><tr><td>OptiMUS</td><td>69.5%</td><td>65.0%</td><td>0.889 水</td><td>0.281 兴</td><td>0.417 冰</td><td>0.084 水</td></tr><tr><td>OptimAI</td><td>61.0%</td><td>55.0%</td><td>水 0.556</td><td>水 0.077</td><td>* 0.377</td><td>茶 0.051</td></tr><tr><td>Persistent ReAct</td><td>62.7%</td><td>70.0%</td><td>0.889 水</td><td>0.501 宗</td><td>0.259*</td><td>0.042*</td></tr><tr><td>MAPLE (ours)</td><td>59.3%</td><td>80.0%</td><td>1.000</td><td>0.951</td><td>0.832</td><td>0.875</td></tr></table>

Figure 1: MAPLE maintains plans through successive requests. The routing example shows problem revision, memory of earlier decisions, and reuse of search results. The tables compare capabilities and performance. An asterisk marks adapted baselines.

## 1 INTRODUCTION

Domain practitioners understand operational requirements but may lack operations research training or specialist support. Production planners, dispatchers, workforce coordinators, and infrastructure operators must translate these requirements into optimization models and maintain the software that solves them. This expertise barrier limits everyday adoption of optimization methods (AhmadiTeshnizi et al., 2024; Zhang et al., 2025). LLM-based optimization agents translate natural-language descriptions into models or solver code and execute them with established optimization packages. Some systems use execution errors to revise the generated model or program (AhmadiTeshnizi et al., 2024; Huang et al., 2025; Thind et al., 2025; Zhang et al., 2025). More recent systems ask users to clarify the planning problem and build optimization models from business data (Xie, 2026). These systems connect domain practitioners to optimization tools.

Real-world operations are dynamic: new orders arrive, resources become unavailable, and priorities change. Model generation alone cannot ensure rapid, consistent adaptation. Keeping a previously discussed customer’s vehicle assignment requires the agent to revise the program, recover that decision, and assess whether earlier candidates remain useful.

Related work addresses parts of this setting from different directions. Multi-turn planning benchmarks study how plans change as new constraints arrive (Oh et al., 2025). Other LLM-based optimization methods address nonlinear problems and multi-objective search (Thind et al., 2025; Schwanke et al., 2026). Dynamic optimization, meanwhile, has long studied population repair, transfer, and restart under changing objectives and constraints (Branke, 2002; Nguyen et al., 2012). We address a practical gap in real-world operations: adapting executable optimization programs to evolving natural-language requirements while preserving accepted decisions and reusing relevant search state. We call this setting live dynamic optimization.

MAPLE maintains an executable optimization program (the Workbench), input data, accepted plans, update records, and candidate solutions. At initialization, Typed Search-Space Scaffolding (TSS) lets the model define decisions and evaluation, while the system supplies search operations and solvers. It supports mathematical programming for linear formulations, evolutionary search for nonlinear or combinatorial objectives, and Pareto search for competing objectives. For subsequent requests, Live Problem Decomposition (LPD) identifies the input data or Workbench functions affected by the revision. Live State Memory (LSM) retrieves information from earlier updates and accepted plans. When a request asks to keep an earlier assignment, the system enforces that assignment during the next search. For evolutionary search, a restart selector reads a summary of the changes and assesses whether earlier candidates remain useful. Fixed checks then select repaired-history initialization (Warm) or fresh initialization (Full). The revised state is committed only after executable and output checks pass.

We evaluate MAPLE on NLDO, a controlled benchmark containing 15 trajectories and 180 naturallanguage updates across selection, scheduling, rostering, routing, and cloud-resource placement. NLDO provides both single- and multi-objective tasks and evaluates trajectory continuity separately from the quality of the resulting plans. In the main evaluation, MAPLE completes all trajectories, reaching online scalar quality of 0.951 and Pareto hypervolume ratio of 0.875. The component and restart controls further examine how executable-state maintenance affects validity, historical grounding, and reuse across revisions.

Our contributions are threefold. First, we formulate live dynamic optimization as optimization under cumulative natural-language revisions and introduce NLDO, with static and dynamic views of single- and multi-objective tasks. Second, we develop MAPLE, an executable-state architecture that integrates problem construction, accepted decisions, historical grounding, and reusable numerical search state. Third, we evaluate complete agent workflows and controlled variants, showing that persistent executable state improves continuity and validity while allowing useful search information to survive substantial revisions.

## 2 TASK AND BENCHMARK

From requests to optimization problems. NLDO evaluates a sequence of planning requests. Each episode begins with a request and input tables $( x _ { 0 } , D _ { 0 } )$ , followed by revisions $u _ { 1 : T }$ . Together they define, at time t, a feasible set $\mathcal { F } _ { t }$ and objective vector $\mathbf { f } _ { t } = ( f _ { t , 1 } , \ldots , f _ { t , k _ { t } } )$ . For $k _ { t } = 1$ , the desired output is one feasible plan with a small objective value. For $k _ { t } > 1$ , it is a set of feasible, mutually nondominated plans that represents the trade-offs. Objectives are converted to minimization for evaluation. The public request distinguishes optimization objectives, hard constraints, and auxiliary diagnostics. Every method receives the initial request, public tables, cumulative public updates, required outputs, and public validation feedback. Each method retains its own dialogue and state through a separate execution history. Section 4 specifies the different history interfaces used in the comparison. MAPLE stores the input tables $D _ { t }$ , optimization program $W _ { t }$ , accepted output $A _ { t }$ , and candidate solutions $P _ { t }$ . These form the state $S _ { t } = \mathbf { \bar { \Phi } } ( D _ { t } , W _ { t } , A _ { t } , P _ { t } )$ ). It also retains the earlier update records $E _ { t }$ . For $t \geq 1$ , the update procedure receives the current request, preceding optimization state, and event history:

$$
S _ { t } = \Phi ( x _ { 0 } , u _ { 1 : t } , S _ { t - 1 } , E _ { t - 1 } ) .
$$

An accepted update appends its event record to $E _ { t - 1 }$ . A rejected revision leaves the preceding committed state and event history unchanged. For Pareto tasks, $A _ { t }$ stores alternative plans and the representative used for historical assignments (Section 3).

Continuity and quality. We evaluate recorded protocol continuity and saved-plan quality separately. Let $a _ { t }$ record whether the original output passes both its output-protocol checks and its recorded validity check. The strict trajectory protocol keeps the prefix $\textstyle r _ { t } = \prod _ { j = 0 } ^ { t } a _ { j }$ . Let $v _ { t }$ indicate whether the saved output contains a feasible plan under the evaluation formulas, and let $q _ { t }$ measure its quality, with $q _ { t } = 0$ when $v _ { t } = 0$ . We report

$$
\mathrm { S o l v e R a t e } = \frac { 1 } { T + 1 } \sum _ { t = 0 } ^ { T } r _ { t } v _ { t } , \qquad \mathrm { O n l i n e Q u a l i t y } = \frac { 1 } { T + 1 } \sum _ { t = 0 } ^ { T } r _ { t } q _ { t } .
$$

Here $q _ { t }$ is normalized scalar objective quality or a hypervolume (HV) ratio for Pareto outputs. Both summaries include the initial state. The first recorded rejection ends the eligible prefix, and all remaining states receive zero, including unattempted states. We separately report output-protocol failures, infeasible plans, and feasibility among attempted updates. Appendix A.5 defines normalization, IGD, and reference construction.

NLDO coverage. NLDO has five families with three sequences each: 15 trajectories, 195 states, and 180 updates. Each sequence contains an initial request and twelve updates. NLDO-SS and NLDO-SM evaluate the single- and multi-objective initial states. NLDO-DS and NLDO-DM evaluate complete trajectories from t00 to t12, including initialization. Update-only analyses explicitly use t01–t12. Numerical seeds repeat the search within each model-generated trajectory. The main episodes retain their initial table schemas. Rostering, routing, and cloud placement include disruptive replacements at t11–t12. Some updates change only diagnostics or positively rescale an objective, preserving Pareto dominance. Here, public data means input data available to the agent. Separate controls test historical bindings and candidate reuse beyond the main trajectories (Appendix A.13).

Appendix A.6 explains how updates affect feasibility, objective ordering, rescaling, and diagnostics.

## 3 METHOD

MAPLE operates in two stages: constructing an initial executable state and revising it as requests arrive. TSS connects the model’s decision declarations to fixed search operations and solvers during construction (Figure 2). For later requests, LPD locates the required edits and LSM resolves references to accepted history. Restart selection determines whether evolutionary search reuses earlier candidates or starts afresh. The fixed runtime checks the program and solver output before committing the updated state (Figure 3).

TSS: bind decision types to problem semantics. The model writes build\_problem(public\_context) to read public tables, construct evaluation data, and declare solver mode, objectives, and decision segments. A segment contains binary, categorical, bounded integer, bounded real, permutation, or assignment decisions; segments compose a typed candidate. evaluate(genome, data) decodes the candidate into a plan and returns objective values and constraint violations.

![](images/ff216b354abe343970bbe64d06c03ad6841af4b9b64357f922a7f97fdce4daa9.jpg)  
Figure 2: Constructing the initial state. The model builds $W _ { 0 }$ from $D _ { 0 } ;$ ; TSS and NSGA-II produce $P _ { 0 }$ and $A _ { 0 }$ . Dashed teal marks model-authored code; solid blue-gray marks fixed execution.

Declaring a type binds its fixed initialization, crossover, mutation, and domain-repair operators. Real vectors use simulated binary crossover and polynomial mutation (Deb et al., 2002). Permutations use order crossover and swap mutation (Davis, 1985), while discrete vectors and assignments use gene-wise parent choice and type-valid mutation (Holland, 1975; Syswerda, 1989).

All four operators preserve the declared representation space $\mathcal { X } ( W _ { t } , D _ { t } )$ ; evaluate() checks problem constraints. The declared formulation selects the solver. For linear and mixed-integer linear problems, the fixed interface builds a model from variables, bounds, objective coefficients, and constraints, then extracts the requested plan from returned variable values (Dantzig, 1963; Nemhauser & Wolsey, 1988). Scalar nonlinear or combinatorial tasks use GA (Holland, 1975), and separate objectives use NSGA-II (Deb et al., 2002). The evolutionary loop evaluates candidates, selects parents, applies variation and repair, and updates the population and Pareto set; stopping uses Workbench search progress.

Routing example (Figure 2). build\_problem() reads the depot, order, vehicle, and policy tables, keeps active orders and available vehicles, and declares two decision segments. An assignment $a _ { i }$ selects each order’s vehicle; a permutation π specifies visit order. Assignment initialization and mutation sample allowed vehicles, while crossover inherits each order’s vehicle from a parent. Permutations use shuffling, order crossover, and swaps, retaining each active order exactly once. Assignment repair corrects invalid vehicle choices.

evaluate() groups orders by a in π order into one depot-to-depot route per used vehicle. It accumulates Euclidean distance, simulates travel, waiting, and service for lateness, and computes emissions from distance, the vehicle emission rate, and carbon multiplier. Capacity excess, late shift return, and missing orders produce constraint violations. It returns routes and three objectives to the fixed NSGA-II loop.

LPD: locate the requested revision. LPD reads the public update, decision declarations, stored events, and accepted result to identify changes to data, decisions, evaluation logic, or historical bindings. The system applies supplied table edits directly. When a change is described only in language, the model translates it into edits to the input tables. The editor receives the requested Workbench functions and preserves the remaining functions. A price change can therefore update the tables alone, while a new constraint may require an evaluation edit.

LSM: bind requests to accepted history. LSM stores events, accepted decisions, and search candidates. For a request to preserve the assignment of a previously mentioned customer, the event identifies the customer and the accepted plan supplies the vehicle. A typed binding query retrieves this relation and checks it against the current decision domain. The runtime enforces the resolved assignment during search. Events are added to the committed history after an update is accepted. Requests about earlier assignments use one saved plan from the preceding accepted state. The full Pareto set is stored separately. In the recorded runs, the saved plan is the first candidate returned by final NSGA-II selection.

![](images/60a7d1f5a20f36c3b75753b0d8476ee03b6b3f58b6624bcc0552e5b6ebe7f4fe.jpg)  
Figure 3: A localized routing update. LPD locates edits and LSM resolves historical references. Compilation and public-test errors trigger bounded repair; invalid search outputs reject the update. Valid outputs commit the state and history.

Restart selector: estimate reuse risk. After compilation and a small input check, the runtime summarizes changes to public data, decision types, and objectives. The model uses this summary and accepted history to assess reuse risk and recommend Warm or Full. Its explanation cites changed fields or declarations, such as replaced active orders, reversed objective preferences, or changed machine capacities and energy costs. Algorithm 1 verifies the cited changes and structural support for each proposed cause.

Algorithm 1 Fixed gate for semantic restart selection   
Require: public-change summary $\Delta _ { t } ,$ model assessment $( r , M , E , v )$   
Ensure: restart action Warm or Full   
1: V ← M ∩ SupportedMechanisms $( \Delta _ { t } )$ ▷ check structural support   
2: C ← E ∩ ChangedFields $\left( \Delta _ { t } \right)$ ▷ verify the cited changes   
3: return Full if $r = { \mathsf { h i g h } }$ and v = true and $V \neq \emptyset$ and $C \neq \varnothing ;$   
otherwise return Warm.

Here r is the assessed reuse risk, M contains the model’s proposed causes of that risk, E identifies the public changes cited in its explanation, and v is true when it recommends initializing the whole population afresh.

SupportedMechanisms $\left( \Delta _ { t } \right)$ checks which of the five allowed causes have matching structural changes. For example, changing the declared decision segments supports replacement of decision choices; changes to multiple resource attributes, such as CPU capacity and energy cost, support a change in the relative usefulness of resources. ChangedFields $\left( \Delta _ { t } \right)$ lists the public data and program declarations that actually changed, including table values and membership, decision segments, objectives, and the two editable functions. Thus V retains supported causes and C retains verified cited changes. Appendix A.2 specifies the five causes and their structural tests.

Warm maps the preceding population and archive to the revised decisions, then uses local typed repair to prioritize feasibility and lower penalized scalar fitness. Repaired candidates fill at most half the population; fresh candidates fill the rest. Full initializes every candidate afresh. Both use the same subsequent solver and stopping rule; NSGA-II compares objectives separately. Warm additionally evaluates candidates during migration and repair.

Validation and commitment. The runtime compiles each edit, checks the required function interfaces, and executes a small input example. Errors in these checks trigger at most B model repair attempts. The fixed solver then runs and checks the required output fields and structure. A failure rejects the revision and preserves the preceding state. A passing result commits the revised data, Workbench, accepted output, and search state, then appends the accepted event. Held-out evaluation scores the saved solution after execution. Appendix A.1 gives the control flow, and Appendix A.3 records the editable surfaces and available choices.

## 4 EXPERIMENTS

We evaluate initial solving, continued plan revision, use of earlier information, and candidate reuse.

Compared agents. We compare MAPLE with ReAct, Persistent ReAct, and four optimizationagent workflows: ORLM, OptiMUS, OR-LLM-Agent, and OptimAI. The NLDO adapters provide cumulative public requests, execution feedback, and each method’s available history, with at most three code-output repairs. Persistent ReAct carries its solver code and accepted plan between requests and can return a Pareto set for each request. Appendix A.8 specifies the workflow implementations and carried state.

Model runs and numerical search. The main comparison uses DeepSeek-V4-Pro (Preview), with API ID deepseek-v4-pro. For each MAPLE episode, one sequence of model outputs supplies the optimization programs and restart choices. We repeat the numerical search with ten random seeds using this same sequence. Each search uses 200 candidates and at most 200 generations. External methods contribute one model trajectory per episode. The independent-model controls in Appendix A.13 vary Workbench generation separately. We set the Workbench repair limit to B = 3. Stopping uses the Workbench’s search progress (Appendix A.9).

In full-sequence comparisons, each reuse strategy continues with the candidates produced by its own earlier searches. In single-update comparisons, the strategies start from MAPLE’s same saved population. Only the initialization choice differs.

Evaluation. Search uses the Workbench evaluator; restart selection reads public history and the change summary. Held-out scores and references are used only offline. We evaluate saved Pareto plans using the revised public formulas and new reference archives (Appendix A.6). HV divides the submitted archive’s hypervolume by the state-specific reference’s and can exceed one. IGD uses the same scales for nonempty feasible archives. Alongside the strict prefix defined in Section 2, we report a mathematical prefix that accepts feasible saved plans regardless of auxiliary fields. Appendices A.5 and A.6 provide normalization and formula details.

## 4.1 Q1: CAN THE AGENT SOLVE THE INITIAL REQUEST?

We first evaluate the initial request before testing later revisions. Table 1 reports execution and reference-objective agreement. MAPLE reaches 59.3% on NLP4LP-hard and 80.0% on BWOR20, with scalar quality of 1.000 and Pareto HV of 0.832 on the NLDO initial states. Appendix A.7 reports tolerances and workflow diagnostics.

## 4.2 Q2: CAN THE AGENT FOLLOW A SEQUENCE OF REVISIONS?

We test whether each agent continues to return accepted, feasible plans as the user changes the request. Acceptance checks the required output fields and structure. Feasibility checks whether the saved plan satisfies the problem constraints. Figure 4 shows per-state quality, and Table 2 summarizes continuity and quality over the same sequences.

<table><tr><td rowspan="2">Method</td><td colspan="2">NLP4LP-hard</td><td colspan="2">BWOR20</td><td colspan="2">NLDO-SS</td><td colspan="2">NLDO-SM</td></tr><tr><td>Exec.</td><td>Obj.</td><td>Exec.</td><td>Obj.</td><td>Valid</td><td>Quality</td><td>Valid</td><td>HV</td></tr><tr><td>MAPLE</td><td>93.2%</td><td>59.3%</td><td>100.0%</td><td>80.0%</td><td>100.0%</td><td>1.000</td><td>100.0%</td><td>0.832</td></tr><tr><td>Persistent ReAct</td><td>98.3%</td><td>62.7%</td><td>100.0%</td><td>70.0%</td><td> $8 8 . 9 \%$ </td><td>0.889</td><td> $5 0 . 0 \% ^ { * }$ </td><td>0.259*</td></tr><tr><td>ReAct</td><td>78.0%</td><td>57.6%</td><td>95.0%</td><td>70.0%</td><td> $8 8 . 9 \%$ </td><td>0.889*</td><td> $3 3 . 3 \% ^ { * }$ </td><td>0.300</td></tr><tr><td>OR-LLM-Agent</td><td>96.6%</td><td>50.8%</td><td>95.0%</td><td>65.0%</td><td> $7 7 . 8 \%$ </td><td> $0 . 7 7 8 ^ { ^ { \ast } }$ </td><td> $6 6 . 7 \% ^ { * }$ </td><td> $\boldsymbol { 0 . 4 4 0 } ^ { * }$ </td></tr><tr><td>ORLM</td><td>61.0%</td><td>23.7%</td><td>55.0%</td><td>45.0%</td><td> $1 0 0 . 0 \% ^ { * }$ </td><td>1.000*</td><td> $0 . 0 \% ^ { * }$ </td><td>0.000</td></tr><tr><td>OptiMUS</td><td>93.2%</td><td>69.5%</td><td>100.0%</td><td>65.0%</td><td> $8 8 . 9 \%$ </td><td> $0 . 8 8 9 ^ { * }$ </td><td> $6 6 . 7 \% ^ { * }$ </td><td> $0 . 4 1 7 ^ { * }$ </td></tr><tr><td>OptimAI</td><td>98.3%</td><td>61.0%</td><td>80.0%</td><td>55.0%</td><td> $5 5 . 6 \%$ </td><td> $0 . 5 5 6 ^ { * }$ </td><td> $6 6 . 7 \% ^ { * }$ </td><td> $0 . 3 7 7 ^ { * }$ </td></tr></table>

Table 1: Static execution and solution quality. Exec.: successful execution. Obj.: referenceobjective agreement over all cases. Valid: accepted and feasible under the recorded output protocol. HV: public-formula ratio, zero for invalid outputs. Appendix A.7 reports supplementary diagnostic checks. <sup>\*</sup> marks task-interface adaptation.

<table><tr><td rowspan="2">Method</td><td colspan="3">Dynamic single-objective</td><td colspan="3">Dynamic multi-objective</td></tr><tr><td>Solve rate</td><td>Strict Quality</td><td>Math. Quality</td><td>Solve rate</td><td>Strict HV</td><td>Math. HV</td></tr><tr><td>MAPLE</td><td>100%</td><td>0.951</td><td>0.951</td><td>100%</td><td>0.875</td><td>0.875</td></tr><tr><td>Persistent ReAct</td><td>51.3%</td><td>0.501</td><td>0.763</td><td>11.5%</td><td>0.042</td><td>0.185</td></tr><tr><td>ReAct</td><td>33.3%</td><td>0.333</td><td>0.376</td><td>2.6%</td><td>0.023</td><td>0.091</td></tr><tr><td>OptiMUS</td><td>28.2%</td><td>0.281</td><td>0.333</td><td>12.8%</td><td>0.084</td><td>0.118</td></tr><tr><td>ORLM</td><td>12.0%</td><td>0.120</td><td>0.298</td><td>0%</td><td>0.000</td><td>0.021</td></tr><tr><td>OptimAI</td><td>7.7%</td><td>0.077</td><td>0.103</td><td>11.5%</td><td>0.051</td><td>0.059</td></tr><tr><td>OR-LLM-Agent</td><td>11.1%</td><td>0.111</td><td>0.171</td><td>12.8%</td><td>0.046</td><td>0.046</td></tr></table>

Table 2: Continuity and quality, t00–t12. Solve rate uses strict prefixes. Math.: mathematical prefix. Pareto scores use public formulas. Both protocols assign zero to missing suffixes. <sup>\*</sup> marks dynamic adaptation.

![](images/8c74d2659e3920dbbe01c9cbb5cefd2d4cbf77e1a829a3f8014730aeec7ed818.jpg)  
Figure 4: Strict-prefix quality through updates. Upper: scalar; lower: Pareto. Columns cover t00–t12. Pareto quality uses public-formula evaluation. A protocol rejection ends the eligible prefix.  
MAPLE completes every trajectory in the main run and reaches strict-prefix online quality of 0.951 on scalar tasks and 0.875 HV on Pareto tasks. Persistent ReAct obtains 0.501 and 0.042, respectively. Seven Persistent ReAct scalar trajectories first fail the output checks because a required reasoning field is missing. One of these outputs also contains an infeasible plan. Accepting mathematically feasible saved plans regardless of auxiliary fields raises Persistent ReAct’s scores to 0.763 and 0.185. MAPLE’s scores are unchanged.

Program-generation cost. We count input and output tokens for program generation, editing, and repair. Across four additional runs per method, these operations use 51.6k tokens per MAPLE trajectory and 72.8k without TSS. The version without TSS reuses saved decisions about which parts to edit and how to update the input tables. Appendix A.13 reports these costs separately.

## 4.3 Q3: WHAT HELPS THE AGENT HANDLE REVISIONS CORRECTLY?

TSS reduces invalid outputs. We compare MAPLE with a version that generates its own candidate representation, evaluation, search operations, and repair functions instead of using TSS. Both versions use the same fixed numerical search loop. Removing TSS lowers feasibility from 100.0% to 80.6% and mean HV from 0.879 to 0.665 over the Pareto updates (Table 3).

<table><tr><td>Method</td><td>Feasible</td><td>HV↑</td><td>IGD</td></tr><tr><td>MAPLE</td><td>100.0%</td><td>0.879±0.014</td><td>0.079</td></tr><tr><td>Always Warm</td><td>100.0%</td><td>0.868±0.016</td><td>0.091</td></tr><tr><td>Always Full</td><td>100.0%</td><td>0.824±0.015</td><td>0.111</td></tr><tr><td>w/o TSS</td><td>80.6%</td><td>0.665±0.004</td><td>0.143</td></tr></table>

Table 3: Scaffolding and sequential reuse. Six NLDO-DM episodes, t01–t12, ten seeds. Public-formula HV: mean ± sample SD across seed means, with invalid states scored zero. IGD uses valid outputs.

The version without TSS returns 140 invalid outputs among 720 update–seed results. Its routing program omits the check that a vehicle serves only one route. Its cloud-placement program fails to reject GPU-incompatible assignments.

Its valid outputs have mean HV of 0.826.

Search operations tailored to the decisions. We change only how the search combines and modifies candidates. The control copies a complete decision block from one parent or generates a new valid block at random. The optimization program, repair procedure, starting population, and restart choice remain fixed. Mean HV falls from 0.774 to 0.618 across one routing task and one cloud-placement task (Table 22). The comparison covers twelve updates and three search seeds under the original evaluation rules and reference sets.

Variation across program generations and language models. We compare three independently generated program sequences per method for one routing task and one cloud-placement task. These include the main sequence, and restart choices remain fixed. Under the original evaluation rules, mean HV is 0.669 for MAPLE and 0.336 without TSS on routing. The corresponding cloud-placement values are 0.728 and 0.513 (Table 23). A separate full-benchmark run uses Kimi k2.7 under the common public-formula evaluation. Its scalar solve rate and quality are 100.0% and 0.937. Its Pareto solve rate and HV are 92.3% and 0.766 (Table 24).

Agreement between predicted and saved edits. We compare LPD’s proposed changes to the input data, decision definitions, and evaluation logic with the changes saved after each update. They agree in 170 of 180 updates. Neither Workbench function changes in 166 updates. Appendix A.13 reports the disagreements.

Using earlier updates and accepted plans. We first check 30 benchmark requests that identify an entity through earlier updates. LSM resolves all 30, including six that require following a chain of three references.

We then test 18 assignment requests across two domains, three information requirements, and three wordings. Six can be answered from earlier update records. Six explicitly name entities and ask to preserve their assignments from the accepted plan. The remaining six require both records. For example, keeping the vehicle assignment of the customer mentioned in a congestion note requires identifying the customer from that note and retrieving its vehicle from the saved plan.

With both records, LSM resolves all 18 requests. Each partial condition resolves only its six directly answerable requests; neither resolves those requiring both records (Table 20a).

Separately, search replay uses saved queries and the original online plans in six cases with three search seeds each. Full LSM and a control given the required assignment directly each produce 18 valid results (Table 20b): the correct assignment is retrieved and enforced, and every returned plan satisfies the constraints.

## 4.4 Q4: WHEN SHOULD SEARCH REUSE PREVIOUS SOLUTIONS?

Restart choices across task revisions. We compare MAPLE with Always Warm and Always Full over six Pareto task sequences. The controls fix the restart action at every update; each strategy carries candidates from its own searches. Mean update HV is 0.879 for MAPLE, 0.868 for Always Warm, and 0.824 for Always Full.

MAPLE and Always Warm both obtain 0.915 on the first ten updates. The last two updates replace substantial parts of the tasks and resources. MAPLE selects Full and obtains 0.696, compared with 0.636 for Always Warm. The gains are concentrated in routing (Figure 5).

For each sequence, we compute the paired difference between MAPLE and Always Warm. The mean gain is 0.010, with an episode-bootstrap 95% interval of [0.001, 0.021]. Five of the six differences are positive (two-sided exact sign test, p = 0.219).

![](images/ccd9ce09cfb21b413f103cb8e92cdb4532321d8f58ab997c289db556e4594c38.jpg)

<table><tr><td>Change</td><td>Routing</td><td>Cloud</td></tr><tr><td>Renamed</td><td>+0.149</td><td>0.000</td></tr><tr><td>entities</td><td></td><td></td></tr><tr><td>Added tasks</td><td>+0.708</td><td>+0.022</td></tr></table>

Chosen action: MAPLE selects Warm; both rules select Full.  
Figure 5: Reuse before disruption, restart after it. Public-formula HV over six episodes: ten-seed means and one-SD bands. Each policy carries its own population; gray marks disruptive updates.  
Table 4: Reuse after renaming or adding tasks. Warm minus Full HV: three paired seeds per case, with separate pooled references under the original evaluation rules. Numerical decisions precede identifier mapping (Appendix A.13).

Earlier candidates remain useful after renaming or adding tasks. Four additional tests rename entities or add low-demand tasks while retaining existing resources. Both actions start from the same saved population with three search seeds. MAPLE selects Warm in all four cases. Adding tasks gives Warm an HV gain over Full of 0.708 in routing and 0.022 in cloud placement (Table 4), using the original evaluation rules and separate pooled references per case.

A rule based on the fraction of changed table cells matches MAPLE on the main Pareto sequences but selects Full in all four added tests. A numerical rule re-evaluates old candidates: using old identifiers selects Full in the renaming tests, while mapping identifiers first makes all six decisions Warm (two tasks, three seeds). Appendix A.13 gives the full comparisons.

## 5 RELATED WORK

Natural-language optimization for domain practitioners. NL4Opt maps language to linear programs (Ramamonjison et al., 2023). Optimization agents generate formulations and solver code, using execution feedback for revision (AhmadiTeshnizi et al., 2024; Zhang et al., 2025; Huang et al., 2025; Thind et al., 2025). MIRROR and ORPilot add problem elicitation and operational data handling (Shi et al., 2026; Xie, 2026). MAPLE supports continued revision as operations change.

Interactive and dynamic language agents. Agents retain dialogue and experience across interactions (Yao et al., 2023), with memory systems retrieving prior information (Packer et al., 2023; Maharana et al., 2024). Flex-TravelPlanner introduces constraints over multiple turns (Oh et al., 2025); Gaia2 tests asynchronous events and actions that change the environment (Froger et al., 2026). MAPLE retains executable optimization state, while NLDO evaluates continuity and solution quality across revisions.

Multi-objective and dynamic numerical optimization. Dynamic optimization uses population repair, transfer, and restart under changing objectives and constraints (Branke, 2002; Nguyen et al., 2012). MOHOLLM uses LLM surrogate models and candidate samplers for hierarchical multiobjective search (Schwanke et al., 2026). MAPLE revises executable problems through language, then runs established solvers such as NSGA-II (Deb et al., 2002).

## 6 CONCLUSION

MAPLE maintains executable optimization state across natural-language revisions. It completes all 15 NLDO trajectories, with online scalar quality of 0.951 and Pareto HV of 0.875. Controls show that TSS reduces invalid outputs and that some historical requests require both update records and accepted plans. Fresh initialization helps after disruptive routing changes; repaired candidates remain useful in renaming and task-addition controls.

Ethics and reproducibility. NLDO uses synthetic data. The supplement provides tasks, prompts, settings, seed-level evidence, and offline scoring with references separate from agent inputs (Appendix A.10).

## REFERENCES

Ali AhmadiTeshnizi, Wenzhi Gao, and Madeleine Udell. OptiMUS: Scalable optimization modeling with (MI)LP solvers and large language models. arXiv preprint arXiv:2402.10172, 2024.

J. E. Beasley. OR-Library: Distributing test problems by electronic mail. Journal of the Operational Research Society, 41(11):1069–1072, 1990. doi: 10.1057/jors.1990.166.

Tolga Bekta¸s and Gilbert Laporte. The pollution-routing problem. Transportation Research Part B: Methodological, 45(8):1232–1250, 2011. doi: 10.1016/j.trb.2011.02.004.

Anton Beloglazov, Jemal Abawajy, and Rajkumar Buyya. Energy-aware resource allocation heuristics for efficient management of data centers for cloud computing. Future Generation Computer Systems, 28(5):755–768, 2012. doi: 10.1016/j.future.2011.04.017.

Paolo Brandimarte. Routing and scheduling in a flexible job shop by tabu search. Annals of Operations Research, 41:157–183, 1993. doi: 10.1007/BF02023073.

Jürgen Branke. Evolutionary Optimization in Dynamic Environments. Springer, 2002.

Sara Ceschia, Nguyen Dang, Patrick De Causmaecker, Stefaan Haspeslagh, and Andrea Schaerf. The second international nurse rostering competition. Annals ofOperations Research, 274:171–186, 2018. doi: 10.1007/s10479-018-2816-0.

George B. Dantzig. Linear Programming and Extensions. Princeton University Press, Princeton, NJ, 1963.

Lawrence Davis. Applying adaptive algorithms to epistatic domains. In Proceedings of the 9th International Joint Conference on Artificial Intelligence, pp. 162–164, 1985.

Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan. A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2):182–197, 2002.

Kalyanmoy Deb, N. Udaya Bhaskara Rao, and S. Karthik. Dynamic multi-objective optimization and decision-making using modified NSGA-II: A case study on hydro-thermal power scheduling. In Evolutionary Multi-Criterion Optimization, volume 4403 of Lecture Notes in Computer Science, pp. 803–817, 2007. doi: 10.1007/978-3-540-70928-2\_60.

Sevgi Erdogan and Elise Miller-Hooks. A green vehicle routing problem. ˘ Transportation Research Part E: Logistics and Transportation Review, 48(1):100–114, 2012.

Romain Froger, Pierre Andrews, Matteo Bettini, Amar Budhiraja, Ricardo Silveira Cabral, Virginie Do, Emilien Garreau, Jean-Baptiste Gaya, Hugo Laurençon, Maxime Lecanu, Kunal Malkan, Dheeraj Mekala, Pierre Ménard, Gerard Moreno-Torres Bertran, Ulyana Piterbarg, Mikhail Plekhanov, Mathieu Rita, Andrey Rusakov, Vladislav Vorotilov, Mengjue Wang, Ian Yu, Amine Benhalloum, Grégoire Mialon, and Thomas Scialom. Gaia2: Benchmarking LLM agents on dynamic and asynchronous environments. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=9gw03JpKK4.

John H. Holland. Adaptation in Natural and Artificial Systems. University of Michigan Press, Ann Arbor, MI, 1975.

Chenyu Huang, Zhengyang Tang, Shixi Hu, Ruoqing Jiang, Xin Zheng, Dongdong Ge, Benyou Wang, and Zizhuo Wang. ORLM: A customizable framework in training large models for automated optimization modeling. Operations Research, 2025. arXiv:2405.17743.

Min Liu, Jinhua Zheng, Junnian Wang, Yuzhen Liu, and Lei Jiang. An adaptive diversity introduction method for dynamic evolutionary multiobjective optimization. In 2014 IEEE Congress on Evolutionary Computation, pp. 3160–3167, 2014. doi: 10.1109/CEC.2014.6900364.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of LLM agents. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pp. 13851–13870, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.747. URL https://aclanthology.org/2024.acl-long.747/.

George L. Nemhauser and Laurence A. Wolsey. Integer and Combinatorial Optimization. Wiley, New York, 1988.

Thanh T. Nguyen, Shengxiang Yang, and Jürgen Branke. Evolutionary dynamic optimization: A survey of the state of the art. Swarm and Evolutionary Computation, 6:1–24, 2012.

Juhyun Oh, Eunsu Kim, and Alice Oh. Flex-TravelPlanner: A benchmark for flexible planning with language agents. arXiv preprint arXiv:2506.04649, 2025. URL https://arxiv.org/abs/2506. 04649.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560, 2023.

Victor Pillac, Michel Gendreau, Christelle Guéret, and Andrés L. Medaglia. A review of dynamic vehicle routing problems. European Journal of Operational Research, 225(1):1–11, 2013. doi: 10.1016/j.ejor.2012.08.015.

Rindranirina Ramamonjison, Timothy T. Yu, Raymond Li, Haley Li, Giuseppe Carenini, Bissan Ghaddar, Shiqi He, Mahdi Mostajabdaveh, Amin Banitalebi-Dehkordi, Zirui Zhou, and Yong Zhang. NL4Opt competition: Formulating optimization problems based on their natural language descriptions. In Proceedings of the NeurIPS 2022 Competition Track, volume 220, pp. 189–203, 2023.

Shaaban Sahmoud and Haluk Rahmi Topcuoglu. Exploiting characterization of dynamism for enhancing dynamic multi-objective evolutionary algorithms. Applied Soft Computing, 85:105783, 2019. doi: 10.1016/j.asoc.2019.105783.

Andrej Schwanke, Lyubomir Ivanov, David Salinas, Frank Hutter, and Arber Zela. Multi-objective hierarchical optimization with large language models. arXiv preprint arXiv:2601.13892, 2026. URL https://arxiv.org/abs/2601.13892.

Yifan Shi, Jialong Shi, Jiayi Wang, Ye Fan, and Jianyong Sun. MIRROR: A multi-agent framework with iterative adaptive revision and hierarchical retrieval for optimization modeling in operations research. arXiv preprint arXiv:2602.03318v1, 2026. URL https://arxiv.org/abs/2602. 03318v1.

Gilbert Syswerda. Uniform crossover in genetic algorithms. In Proceedings ofthe Third International Conference on Genetic Algorithms, pp. 2–9, 1989.

Raghav Thind, Youran Sun, Ling Liang, and Haizhao Yang. OptimAI: Optimization from natural language using LLM-powered AI agents. arXiv preprint arXiv:2504.16918, 2025.

Guangrui Xie. ORPilot: A production-oriented agentic LLM-for-OR tool for optimization modeling. arXiv preprint arXiv:2605.02728, 2026.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023.

Bowen Zhang, Pengcheng Luo, Genke Yang, Boon-Hee Soong, and Chau Yuen. OR-LLM-Agent: Automating modeling and solving of operations research optimization problems with reasoning LLM. arXiv preprint arXiv:2503.10009, 2025.

## A APPENDIX

Appendix guide. The appendix first defines the mechanisms, metrics, and execution setup, then presents supplementary experiments. The Harness walkthrough follows the results. Prompt interfaces and the recorded benchmark requests appear last.

<table><tr><td>Topic</td><td>Go to</td><td>Evidence</td></tr><tr><td>Update execution</td><td>App. A.1-A.3</td><td>annotated algorithms, restart checks, and typed operators</td></tr><tr><td>Metrics and experimental setup</td><td></td><td>App. A.5–A.10 metrics, task definitions, interfaces, and numerical settings</td></tr><tr><td>Supplementary results</td><td>App. A.11- A.15</td><td>per-problem results, visual examples, controls, and sensitivity</td></tr><tr><td>Application walkthrough</td><td>App. A.16</td><td>a complete request-to-update walkthrough</td></tr><tr><td>Recorded prompts and requests</td><td>App. A.17- A.18</td><td>prompt interfaces, 15 initial requests, and 180 updates</td></tr></table>

Table 5: Appendix contents and the corresponding evidence locations.

## A.1 MAPLE INITIALIZATION AND UPDATE PROCEDURE

Algorithm 2 shows how MAPLE builds the initial program and handles each later request. LPD decides whether public data, the decision definition, or the evaluation function must change. TSS checks the two editable Workbench functions. The semantic reuse assessment reads the accepted public history and current change digest. A fixed gate checks structural support for the proposed Warm–Full choice. The fixed optimizer then runs an exact solver, scalar evolutionary search, or Pareto evolutionary search. LSM stores events, the accepted state, and search archives with the public tables and Workbench. For a request about an earlier assignment, LPD identifies the requested relation. A fixed runtime procedure resolves the requested relation from stored events and accepted decisions, then enforces the binding during evaluation.

MAPLE w/o TSS builds its own representation, evaluation, search, and repair functions from the public t00 task. It carries that Workbench, accepted state, and search archives through the sequence. The fixed optimization loop still performs selection, maintains the archive, records metrics, and checks stopping. Table 9 specifies method inputs and carried state.

Algorithm notation. Let $x _ { 0 }$ denote the initial request and $u _ { t }$ the t-th update. $T$ is the number of updates. The repair limit B counts additional model attempts after a failed public check $( B = 3$ in the reported runs). The saved state is ${ \cal S } _ { t } = ( D _ { t } , W _ { t } , A _ { t } , P _ { t } )$ $E _ { t }$ contains the accepted update history. For Pareto tasks, $A _ { t }$ contains an archive and its representative plan. Evolutionary search uses the candidate population $P _ { t }$ . The LP/MILP path does not use a population. A tilde marks tentative tables or functions that have not yet replaced the accepted state. The public-change summary $\Delta _ { t }$ records differences between the accepted and tentative tables, decision declarations, and objectives. A smoke test runs a small example from the available input data. It checks data access, function calls, and output structure before numerical search.

LLM calls are conditional on the requested data translation, function edit, or repair. Restart assessment uses the public update history and change digest before population construction. Held-out feasibility checks and reference scores are computed after execution.

## A.2 CHECKS FOR RESTART RECOMMENDATIONS

The model proposes whether to reuse earlier candidates. The fixed checks determine whether the response meets the conditions for a fresh start.

Parsing the model response. The parser lowercases the risk string and trims its whitespace. It removes unrecognized mechanisms and duplicate list entries. It also strips whitespace and outer slashes from cited field paths. A cited path is retained only if it belongs to the public change summary’s changed-field set. The five allowed identifiers are decision\_support\_replacement, objective\_preference\_reversal, constraint\_regime\_change, resource\_role\_reversal, and distant\_basin\_risk. Their checks are listed below.

Algorithm 2 MAPLE initialization and live update   
Require: initial request $x _ { 0 } ,$ initial public tables $\overline { { D _ { 0 } } } ,$ updates $u _ { 1 : T } ,$ repair limit B   
Ensure: accepted states and event histories $( S _ { t } , E _ { t } )$ , stopping at the first failed state   
1: LLM: construct candidate Workbench $\widetilde { W _ { 0 } }$ from $( x _ { 0 } , D _ { 0 } )$ ▷ declare decisions and their evaluation   
2: Compile and test $\widetilde { W } _ { 0 }$ on a small public example. ▷ check signatures, types, and outputs   
3: On public errors, LLM: repair and repeat these checks, at most B times.   
4: if checks still fail then   
5: return failure. ▷ no initial state is committed   
6: end if   
7: Run the fixed solver; check execution and the public output contract.   
8: if execution or output-contract checks fail then   
9: return failure. ▷ no post-search LLM repair   
10: end if   
11: Commit $S _ { 0 } = ( D _ { 0 } , W _ { 0 } , A _ { 0 } , P _ { 0 } )$ and start $E _ { 0 } .$ ▷ save the accepted result and initial event   
12: for $t = 1 , \dots , T$ do   
13: LLM (LPD): identify edits and bindings required by $u _ { t } .$ ▷ public state and history; no population   
14: Resolve queries from stored events and accepted plans. ▷ e.g., retrieve the customer’s vehicle   
15: Copy $( D _ { t - 1 } , W _ { t - 1 } )$ into $( \widetilde { D } _ { t } , \widetilde { W } _ { t } )$ ▷ keep the committed state until acceptance   
16: Apply supplied table edits; LLM: translate language-only edits if needed.   
17: Edit only the requested Workbench functions (Algorithm 3).   
18: Compile and smoke-test; on public errors, allow at most B LLM repairs.   
19: if a check still fails then   
20: return failure. ▷ retain $S _ { t - 1 }$ and $E _ { t - 1 }$   
21: end if   
22: Validate bindings in the revised decision domain. ▷ check entities and assignments   
23: if the solver uses an evolutionary population then   
24: Compare old and candidate states to form $\Delta _ { t } .$ ▷ changed fields, decision types, objectives   
25: LLM: assess reuse risk from $u _ { t } ,$ public history, and $\Delta _ { t }$   
26: Check the response with Algorithm 1. ▷ select Warm or Full before search   
27: Build the initial population for the selected action. ▷ Warm: at most half repaired; Full: all fresh   
28: end if   
29: Run the fixed solver and check the public output contract. ▷ fixed LP/MILP, GA, or NSGA-II   
30: if execution or output-contract checks fail then   
31: return failure. ▷ retain $S _ { t - 1 }$ and $E _ { t - 1 }$   
32: end if   
33: Commit $S _ { t } = ( \widetilde { D } _ { t } , \widetilde { W } _ { t } , A _ { t } , P _ { t } ) ;$ ; extend $E _ { t - 1 }$ with the accepted event to obtain $E _ { t }$   
34: end for

Algorithm 3 Updating the two TSS Workbench functions   
Require: LPD edit labels, candidate public tables $\widetilde { D } _ { t } ,$ accepted Workbench $W _ { t - 1 }$   
Ensure: checked candidate Workbench $\widetilde { W } _ { t }$ , or a public error for Algorithm 2   
1: ${ \widetilde { W } } _ { t } \gets \mathrm { c o p y } ( W _ { t - 1 } )$ ▷ preserve functions outside the requested edit   
2: if the decision definition or public-data extraction changes then   
3: LLM: edit build\_problem $( )$ ▷ read tables; declare domains, solver, objectives   
4: end if   
5: if evaluation, constraints, decoding, or solution reporting changes then   
6: LLM: edit evaluate(). ▷ decode one genome; compute violations and values   
7: end if   
8: Bind declared types to their fixed operators. ▷ initialization, crossover, mutation, domain repair   
9: Compile; check solver choice, objective order, and required outputs.   
10: Execute a small example using $\widetilde { D } _ { t }$ ▷ exercise data access, decoding, and evaluation   
11: if any public check fails then   
12: return the public error to Algorithm 2. ▷ the outer loop owns the bounded repair budget   
13: end if   
14: return $\widetilde { W } _ { t } .$ ▷ LP/MILP, GA, and NSGA-II routines remain fixed

<table><tr><td>Mechanism</td><td>Required structural change in the public summary</td></tr><tr><td>Decision-support replacement</td><td>Changed typed segments; or active-ID Jaccard overlap at most 0.5; or row-ID overlap at most 0.5 in a table with at least four old and four new rows.</td></tr><tr><td>Objective-preference reversal</td><td>Both a preference path and a policy path change; a changed evaluation function can supply the policy path.</td></tr><tr><td>Constraint-regime change</td><td>Changed typed segments, active-ID overlap at most 0.5, or at least two changed paths ending in a recognized constraint field.</td></tr><tr><td>Resource-role reversal</td><td>At least two changed resource-table paths ending in a recognized capacity, compatibility, emissions, or energy field.</td></tr><tr><td>Distant-basin risk</td><td>At least one of the preceding four structural tests passes.</td></tr></table>

Table 6: Structural support tests used by the fixed restart gate. The model must also report high risk, cast a Full vote, and cite at least one changed path.

Recognized constraint fields are active, available, availability, capacity, cpu, mem, gpu, gpu\_required, max\_shifts, required, eligibility, eligible, and shift\_end. Resource tables are machines, vehicles, nurses, staff, and resources; their recognized fields are capacity, cpu, mem, gpu, emission\_rate, energy\_idle, energy\_per\_cpu, and max\_shifts. Preference paths contain /preferences/ or /preference. Policy paths contain /policy/ or equal workbench/fitness.

The path and mechanism tests are separate. The reason string is logged and does not enter the gate. The prompt requests a JSON boolean for full\_vote. The implementation applies Python boolean conversion. Missing fields have defaults of unknown risk, empty lists, and a false vote. Ordinary parsing or provider errors return Warm. A detected quota-limit error is passed back to the caller.

## A.3 TSS WORKBENCH AND AVAILABLE CHOICES

The TSS Workbench is a reusable optimization program. A small catalog describes the decision types and solvers available to the Workbench-building model, together with the fixed checks and restart actions available to the runtime. The model edits only the decision-definition function and the evaluation function. The first reads public tables, declares typed decisions, chooses the solver, and names the outputs. The second checks one candidate against the public constraints and objectives. Table 7 summarizes this boundary.

How linear problems are sent to the solver. The Workbench specifies variables, bounds, the objective, and linear constraints for the fixed solver. Each variable has a name and a continuous or integer domain. The objective declaration includes coefficients and a direction. Each constraint supplies a coefficient row, a relation, and a right-hand side. The fixed backend builds the coefficient arrays and uses HiGHS through SciPy’s linear or mixed-integer solver. A declared extraction rule maps returned values to selected entities, assignments, or schedules. The runtime validates that output and retains the solver status and objective. This numerical formulation is produced within the same Workbench construction call as the data access code.

Adding a decision type or solver. Each catalog entry specifies its inputs, outputs, use conditions, and possible failures. An entry can call an existing solver or restart policy. Adding a decision type requires implementing its initialization, variation, and repair operators and registering them in the catalog.

TSS rejects a decision space that cannot be expressed by its declared decision types. Solver selection preserves the public objective names and cannot turn a Pareto task into a scalar one. Warm migrates historical candidates into the current decision domains and re-evaluates their feasibility. It deduplicates the repaired seeds and fills the remaining population slots with fresh candidates.

How old candidates are repaired before search. Warm first adapts earlier candidates to the updated decisions and checks their feasibility. These candidates come from the previous population and archive. For multiple objectives, the seed ordering includes objective extremes, knee candidates, and nondominated selection with crowding. Each candidate is first migrated to the current domains, then evaluated. The local search tests type-specific moves and accepts the first improvement in the

## Public-data change: P015-t02 Evaluation-function change: P010-t02

```diff
tables/machines[id=M04] --- Before update
- energy_per_cpu: 0.23 +++ After update
+ energy_per_cpu: 0.285 @@ congestion anchor @@
The remaining public fields and + congestion_anchor_set = set()
+ for oid, o in data["order_by_id"].items():
both editable functions are un- + if abs(o["x"] - 48) < 1e-9 and abs(o["y"] - 11) < 1e-9:
changed. + congestion_anchor_set.add(oid)
+ break # at most one anchor
@@ route traversal @@
+ prev_oid = None # no order before first stop
+ if prev_oid in congestion_anchor_set or oid in congestion_anchor_set:
+ leg_dist *= 1.25
+ prev_oid = oid
@@ return-to-depot leg @@
+ if prev_oid in congestion_anchor_set:
+ leg_dist *= 1.25
Excerpt from the saved patch; unrelated lines are omitted.
```  
Figure 6: Two localized Workbench edits. The left patch changes public data. The right patch changes route evaluation using a coordinate-based congestion anchor and a distance multiplier. Appendix A.6 examines the historical binding and the travel-time calculation required by the public request.

<table><tr><td>Choice</td><td>Available options</td><td>When code must be added</td></tr><tr><td>Decision types</td><td>binary, categorical, integer, real, permutation, assignment, composition</td><td>a new decision type</td></tr><tr><td>Solvers</td><td>LP/MILP, scalar EA, Pareto EA</td><td>a new solver</td></tr><tr><td>Population start</td><td>Warm (repaired history plus fresh candidates) or Full (fresh candidates)</td><td>a new restart action</td></tr><tr><td>Checks</td><td>compilation, output fields, feasibility, and archive checks</td><td>a new safety rule</td></tr></table>

Table 7: Choices exposed by the TSS runtime. The Workbench-building model selects decision types and a solver route. Fixed code supplies checks and operators, while the separate semantic assessment proposes Warm or Full after an update.

pair (⊮[infeasible], $f _ { \mathrm { s c a l a r } } )$ . It stops after a sweep with no improvement or 80 evaluated proposals. The scalar fitness includes the generated penalty, whereas the subsequent NSGA-II loop retains the separate objective vector. After deduplication, seed feasibility is checked again for reporting. The returned seeds enter feasibility-first search, and the remaining slots are sampled afresh.

With population 200 and at most 100 historical candidates, Warm uses up to 8,200 additional evaluation calls before the shared evolutionary loop. The loop itself uses N(g +1) calls for population size N and g completed generations. Stage wall time also includes initialization, repair, and validation.

## A.4 LIMITATIONS AND ETHICS

MAPLE relies on a fixed catalog of decision types, solver routes, and restart actions. Tasks requiring a new decision type or numerical operator need a runtime extension. Typed operators preserve representation membership, but feasibility still depends on the generated evaluation function.

LSM grounds references in retained events and accepted results. The agent needs clarification when a request could refer to several earlier entities or leaves a constraint unspecified. The semantic restart gate similarly depends on the model’s interpretation of the update.

The evolutionary routes optimize within finite budgets. Repairing retained candidates adds work before search, so the choice between reuse and fresh initialization also depends on the available latency.

The current implementation runs generated Python functions in the application process. Operational use requires stronger isolation of files, imports, credentials, and network access. MAPLE is intended for batch or human-supervised re-optimization. High-impact decisions should support inspection, human approval, and rollback.

## A.5 METRIC DEFINITIONS

NLDO evaluates every state with a fixed hidden evaluator. For scalar profiles, let $f _ { t } ( x )$ be the evaluation objective and let $f _ { t } ^ { \star }$ be the exact or best-known reference after sign conversion. The normalized scalar quality is

$$
Q _ { t } ^ { \mathrm { s c a l a r } } = \mathbf { 1 } [ \mathrm { f e a s i b l e } ] \operatorname* { m a x } \left( 0 , 1 - { \frac { \operatorname* { m a x } ( 0 , f _ { t } ( x ) - f _ { t } ^ { \star } ) } { \operatorname* { m a x } ( | f _ { t } ( x ) | , | f _ { t } ^ { \star } | , 1 ) } } \right) .
$$

An infeasible output receives zero. An output that matches the reference receives one. Scalar tables use bounded 0–1 quality. Pareto tables use the raw ratio below. Dynamic tables separate solve rate from online quality using the quantities in Section 2. The recorded indicator $a _ { t }$ includes the original output-protocol and validity decisions, and $r _ { t }$ marks their consecutive accepted prefix. The new mathematical check $v _ { t }$ requires at least one feasible plan in the saved output. Solve rate averages $r _ { t } v _ { t } .$ and online quality averages $r _ { t } q _ { t }$ . An output may therefore be mathematically feasible but ineligible because its original protocol failed or an earlier state was rejected. Unattempted states have $v _ { t } = q _ { t } = 0$ and remain in the denominator. The archive scorer computes mathematical validity, while the trajectory aggregation applies recorded eligibility. For MAPLE, we average quality across ten search seeds at each state, then across states. The reported numerical variation is computed from the seed means. The DS denominator is 9 scalar episodes × 13 sequential states and the DM denominator is 6 Pareto episodes × 13 sequential states. Analyses restricted to updates have denominators 108 and 72, respectively, and are explicitly labelled t01–t12. Online quality is the mean quality over those same states, with infeasible and unsolved states contributing zero.

For multi-objective profiles, the evaluator compares a submitted archive $A _ { t }$ with a withheld reference archive $R _ { t }$ after removing infeasible candidates. For every multi-objective episode and time step, $R _ { t }$ is built from 10 hidden reference runs. Each run uses population 500 and 500 generations. We pool all ten final populations, recheck feasibility, and keep the nondominated set without truncating the merged front. Each time step therefore starts from 5,000 reference candidates before filtering and deduplication. All 78 Pareto-state references are searched afresh under the public objective definitions, including the disruptive updates. This construction is independent of any submitted method. It uses the same public state as the submitted methods but a larger search budget. The reference search has no early stopping and starts independently at each state. We retain each seed’s final population, random seed, completed generation count, and checksums. All objectives are minimized. Each time step uses fixed ideal and reference points derived from its held-out archive. The submitted and reference archives use the same points. For objective $i ,$ let $\ell _ { t , i }$ and $h _ { t , i }$ be its minimum and maximum over $R _ { t }$ . The evaluation reference bound is the larger of its stored bound and

$$
b _ { t , i } = h _ { t , i } + 0 . 5 \operatorname* { m a x } ( h _ { t , i } - \ell _ { t , i } , | h _ { t , i } | , 1 ) .
$$

The implementation adds $1 0 ^ { - 9 }$ before rounding the bound to six decimal places. Both archives use coordinates

$$
\hat { f } _ { t , i } = \mathrm { c l i p } _ { [ 0 , 1 . 5 ] } \left( \frac { f _ { t , i } - \ell _ { t , i } } { \operatorname* { m a x } ( b _ { t , i } - \ell _ { t , i } , 1 0 ^ { - 9 } ) } \right) .
$$

The bounds are shared by every method at that stage and do not change as its search proceeds. An objective value better than the ideal point is clipped to that point. A value worse than the reference bound contributes no volume inside the unit HV box. IGD uses these same clipped coordinates, including the upper cap of 1.5. The benchmark has two or three objectives, for which the runtime computes HV exactly in this normalized box. We report hypervolume ratio

$$
\mathrm { H V R a t i o } _ { t } = \frac { \mathrm { H V } ( A _ { t } ) } { \mathrm { H V } ( R _ { t } ) }
$$

without clipping. Because $R _ { t }$ is a finite best-known set, a feasible archive may slightly exceed a ratio of one. The evaluator removes infeasible candidates before computing HV and IGD. An archive with no feasible candidate receives zero HV ratio. IGD is reported only when an eligible archive contains at least one feasible candidate. We also report inverted generational distance

$$
\mathrm { I G D } _ { t } = \frac { 1 } { | R _ { t } | } \sum _ { y \in R _ { t } } \operatorname* { m i n } _ { x \in A _ { t } } \| \hat { x } - \hat { y } \| _ { 2 } .
$$

We also report ideal gap, for which smaller is better. It measures the distance from $A _ { t }$ to an ideal point placed 10% beyond the best reference value on each objective. Tables report raw HV ratio,

IGD, and ideal gap. The primary saved-plan evaluator rechecks all submitted candidates without imposing a new archive-size cap. Dominated or duplicate points add no HV. Any archive cap used while producing a method’s saved output belongs to that method’s search, not to the pooled reference construction.

## A.6 BENCHMARK AND EVALUATION DETAILS

NLDO has five profiles with three episodes each, spanning established selection, scheduling, rostering, routing, and cloud-placement settings (Beasley, 1990; Brandimarte, 1993; Ceschia et al., 2018; Pillac et al., 2013; Erdogan & Miller-Hooks, 2012; Bekta¸s & Laporte, 2011; Beloglazov et al., 2012).˘ Compact scalar tasks use exact references, larger scalar tasks use best-known solutions, and Pareto tasks use held-out archives.

Effects of the requested updates. We label updates by changes to feasibility, candidate rankings, objective scale, or reported diagnostics. A changed table does not always imply changed optimal decisions. Multiplying one objective by a candidate-independent positive constant preserves Pareto dominance, while changing a coefficient for only one machine or vehicle can reorder candidates. A deadline used only in a reported diagnostic has a different role from a hard return deadline. Historica reference is a separate annotation.

<table><tr><td>View</td><td>Setting</td><td>Objectives</td><td>Episodes</td><td>Time</td><td>Domains</td></tr><tr><td>SS</td><td>Static</td><td>Single-objective</td><td>P001-P009</td><td>t00</td><td>selection/allocation, precedence scheduling, coverage rostering</td></tr><tr><td>SM</td><td>Static</td><td>Multi-objective</td><td>P010-P015</td><td>t00</td><td>ordered service routing, cloud-resource placement</td></tr><tr><td>DS</td><td></td><td>Dynamic Single-objective</td><td>P001-P009 t00-t12</td><td></td><td>selection/allocation, precedence scheduling, coverage rostering</td></tr><tr><td>DM</td><td></td><td>Dynamic Multi-objective</td><td></td><td></td><td>P010–P015 t00–t12 ordered service routing, cloud-resource placement</td></tr></table>

Table 8: The four NLDO views and their problem, time, and domain coverage. Dynamic views include initialization. Update-only analyses exclude t00 and are labelled separately.

![](images/445bbface1fc8d6810a4c076ba7b48935a806204d2b5238046b992bdcbdcfd1d.jpg)  
Figure 7: Five NLDO problem families. Each card pairs a representative MAPLE solution with the public data, update types, and reference used for that family. The tasks span compact exact problems, larger scalar search, and Pareto routing and placement.

Each profile contributes 36 updates. Thirty of the 180 updates require information from earlier requests. Across all updates, 103 first paragraphs have distinct wording. Rostering, routing, and cloud placement contribute six disruptive t11–t12 states each. All table changes retain the initial columns and listed decision entities.

Agent inputs and evaluation annotations. The agent receives the problem description and input tables. Separate annotations identify the intended entities and reference chains for evaluation. Table 9 defines method inputs and retained information.

How saved routes and assignments are scored. We recompute the objectives of saved plans using the formulas stated below.

Routing travel time uses distance divided by speed. Reducing speed by 25% multiplies the affected travel time by $4 / 3 ,$ leaving physical distance unchanged. Emissions sum route distance times vehicle emission rate and the carbon multiplier. The public definition applies congestion only on arrival at the named order, with its identity retained across later edits to that order. Every nonempty route leaves the current depot at time zero and returns by the vehicle’s shift end. Waiting until an order’s ready time and its service duration both count toward elapsed time. Lateness measures service-start time beyond the soft due time. A requested reduction of a soft due target applies to that target directly. It can precede the ready time, in which case the earliest service already incurs lateness. For example, reducing a soft due target of 87 by 50 gives 37. With a ready time of 40 and a revised soft due time of 37, the earliest service incurs three units of lateness.

Cloud energy sums one idle charge per used machine and the assigned CPU demand times that machine’s per-CPU energy rate, scaled by energy price and carbon intensity. The hidden evaluator measures imbalance as

$$
B = 1 0 0 \sum _ { m \in \mathcal { M } _ { \mathrm { a v a i l } } } | u _ { m } - \bar { u } | , \qquad u _ { m } = \frac { \sum _ { j : a ( j ) = m } c _ { j } } { C _ { m } } , \qquad \bar { u } = \frac { \sum _ { m \in \mathcal { M } _ { \mathrm { a v a i l } } } u _ { m } } { | \mathcal { M } _ { \mathrm { a v a i l } } | } .
$$

Here $c _ { j }$ is an active job’s CPU demand and $C _ { m }$ is machine capacity. Available but unused machines contribute zero utilization to the mean.

Execution and scoring versions. The original runs searched with their generated Workbench functions. The primary tables re-score the saved plans with the public formulas stated above. The original hidden routing evaluator multiplies incoming distance, time, and emissions by 1.25 and adds an undeclared load-dependent emission factor. The saved generated patch identifies the affected customer by coordinates and uses a different affected-leg scope (Figure 6). The original hidden cloud evaluator accumulates an energy charge after each assignment, making energy depend on dictionary order. For two jobs with CPU demands 2 and 4, idle energy 5, and per-CPU rate 2, this rule gives 26 or 30 depending on order, while the public formula gives 17 before global scaling. Energy is invariant to the order of keys in the returned assignment map. The original wording allowed variance or absolute deviation. The release now states this absolute-deviation formula explicitly in the initial request and every update. We re-score saved routes and assignments using the public formulas above and construct new common references under those formulas. The main agent comparison, the complete TSS ablation, and the full-sequence restart comparison use the public-formula evaluation. The single-update restart, search-operator, independent-model, and reference-sensitivity controls use the evaluation functions recorded in their original runs. We also checked all saved external Pareto outputs against the formal reference. Persistent ReAct’s old scores used embedded constructive reference sets on 30 feasible recorded rows, nine of which also retained objective vectors inconsistent with re-evaluation of the saved assignment maps. The primary tables recompute objectives from plans and use the common formal reference pool. Across 4,368 Pareto state–seed records, every originally eligible output remains mathematically feasible after re-evaluation. Sixteen additional saved outputs are feasible but outside their recorded strict prefix and receive zero strict-prefix quality.

## A.7 STATIC SOLVING SETUP

The static supplement uses NLP4LP-hard and BWOR20 to measure ordinary one-shot optimization from language. These tasks contain no updates, memory, restart choice, or predecessor population. Each row provides the problem text and public parameters. Each method produces executable Python solver code and follows its published process when available.

Evaluation of initial solutions. We report whether each program runs and whether its returned objective matches the reference. Table 1 reports execution rate and Objective, the fraction of all cases whose returned objective agrees with the reference under the recorded numeric tolerance.

Objective agreement accepts alternative optimal decision vectors. The comparison accepts $| \hat { f } - f ^ { \star } | \leq$ max $\left( \epsilon _ { a } , \epsilon _ { r } | f ^ { \star } | \right)$ , including equality at the threshold. NLP4LP-hard uses $\epsilon _ { a } = \epsilon _ { r } = 1 0 ^ { - 6 } .$ . The BWOR20 manifest overrides these defaults with $\epsilon _ { a } = 0 . 1$ and $\epsilon _ { r } = 0$ . An output without a predicted objective fails the objective check. Status-only acceptance is recorded separately. Both marginal rates are available for all seven methods, with denominators of 59 NLP4LP-hard cases and 20 BWOR20 cases.

Additional diagnostic checks. We separately report compilation, execution, objective agreement, and the original output checks. For case i, let $c _ { i }$ and $e _ { i }$ denote compilation and execution, $o _ { i }$ referenceobjective agreement, and $s _ { i }$ correct reference status on a status-only case. The joint Match diagnostic in Tables 37–38 is

$$
\frac 1 N \sum _ { i = 1 } ^ { N } c _ { i } e _ { i } { \bf 1 } \big [ o _ { i } \mathrm { ~ o r ~ } \big ( \mathrm { s t a t u s - o n l y ~ c a s e ~ } i \mathrm { ~ a n d ~ } s _ { i } \big ) \big ] .
$$

Its denominator includes every benchmark case, including execution failures. The tables also retain compilation, execution, objective agreement, and the original workflow-specific output and namedvariable checks. These diagnostics can overlap. Dashes denote unreported supplementary checks. Joint Match is unreported for ReAct and OptimAI.

Selection of the static benchmark cases. NLP4LP-hard and BWOR20 use fixed subsets of the released datasets. NLP4LP-hard comes from the NLP4LP dataset released with OptiMUS (AhmadiTeshnizi et al., 2024). Our saved snapshot contains 361 instances, including 65 labelled hard. We retain the 59 hard instances with a nonempty reference solution. The other six have no value in the solution field. The reference objectives are read from each retained instance’s supplied solution record. BWOR20 is our fixed 20-case sample from the 82-row BWOR collection distributed with OR-LLM-Agent (Zhang et al., 2025). The manifest samples without replacement using Python’s random generator with seed 20260621, then sorts the selected case identifiers. Its references use the supplied answer field.

## A.8 BASELINE INTERFACES AND EXECUTION

Code and plans retained by Persistent ReAct. Persistent ReAct retains its preceding solver code and accepted answer, then edits the program for the next request. It is our stateful control based on the ReAct pattern. At initialization, the model writes a complete Python solver program using public parameters and solver libraries. At an update it receives the initial public tables, cumulative dialogue, and its carried workspace, then emits the revised program, which applies the requested changes before solving.

The runtime parses, compiles, and executes the program, then reads its printed solution. Compilation, runtime, or solution-format failures return the actual error and available output to the model for at most three code-output repairs. The reasoning and tool-plan fields are auxiliary outputs. Missing auxiliary fields are checked separately and did not trigger code repair in the saved runs.

The workspace retains the complete preceding solver code and a compact final answer. A separate history field contains the preceding plan and cumulative updates. Public execution success with an extracted solution advances this workspace. Nothing is carried at t00. A program may produce multiple candidates and a Pareto set within a stage; the cross-stage workspace retains code and answers, with no separate population or Pareto archive.

## A.9 NUMERICAL SETTINGS AND RELEASED MATERIALS

Table 10 lists the settings used in the experiments. Main model calls share a provider and temperature, with output limits determined by the requested operation. The table separates repair limits from evolutionary-search settings. Stopping rules and budgets were set on pilot material.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Implementation</td><td rowspan="2">Public inputs</td><td rowspan="2">Code retained for the next plan request</td><td rowspan="2">Accepted</td><td rowspan="2">Cross-stage population/ archive</td><td rowspan="2">Repair feedback Compi-</td></tr><tr><td></td></tr><tr><td>MAPLE</td><td>Typed Workbench runtime</td><td>Full history</td><td>Two functions</td><td>Own representative</td><td>Own population and archive</td><td>lation and small- example errors</td></tr><tr><td>w/o TSS</td><td>Untyped Workbench runtime history</td><td>Full</td><td>Full artifact</td><td>Own representative</td><td>Own population and archive</td><td>Compi- lation and small- example errors</td></tr><tr><td>Persistent ReAct executed-code</td><td>Stateful control</td><td>Full history</td><td>Complete solver</td><td>Own compact answer and plan</td><td>None separate from answer</td><td>Compile, runtime, output</td></tr><tr><td>ReAct</td><td>Public-code adapter</td><td>Full history</td><td>In own dialogue, if present</td><td>Own, on continuation</td><td>Own, if returned on continuation</td><td>Compile, runtime, output</td></tr><tr><td>ORLM / OptiMUS / OR-LLM-Agent</td><td>Workflows based on Full the named methods</td><td>history</td><td>In own dialogue, if present</td><td>Own, on continuation</td><td>Own, if returned on continuation</td><td>Compile, runtime, output</td></tr><tr><td>OptimAI</td><td>Paper-based reproduction</td><td>Full history</td><td>In own dialogue, if present</td><td>Own, on continuation</td><td>Own, if returned on continuation</td><td>Compile, runtime, output</td></tr></table>

Table 9: Executed NLDO interfaces. Cumulative inputs include the initial request, public tables, required outputs, and requests through the current stage. Each external workflow uses its own history, with at most three code-output repairs.

<table><tr><td>Setting</td><td>Value</td><td>Applies to</td></tr><tr><td>Language model</td><td>DeepSeek-V4-Pro (Preview), API ID deepseek-v4-pro, temperature 0.0</td><td>all main model calls</td></tr><tr><td>LLM output limits</td><td>1200–18000 tokens</td><td>depend on the requested task</td></tr><tr><td>Workbench edit repair 3 attempts</td><td></td><td>MAPLE Workbench checks</td></tr><tr><td>Workbench check</td><td>compilation and hard-coded-ID check</td><td>public table identifiers only</td></tr><tr><td>External-code repair</td><td>3 attempts</td><td>external methods</td></tr><tr><td>Search budget</td><td>population 200, at most 200 generations, seeds 0-9</td><td>main numerical runs</td></tr><tr><td>Search archives</td><td>at most 500 feasible, unique, nondominated points</td><td>main numerical runs</td></tr><tr><td>restart selector</td><td>semantic choice from stored events and the public change summary</td><td>no trial search or hidden score</td></tr><tr><td>Restart actions</td><td>Warm: up to 100 repaired, then fresh. Full: 200 fresh</td><td>same solver and search budget</td></tr><tr><td>Scalar stopping</td><td>start at g=40, gain  $\epsilon = 5 { \times } 1 0 ^ { - 4 }$  , patience 25</td><td>feasible scalar objective</td></tr><tr><td>Pareto stopping</td><td>g=40, min. archive 8,  $\epsilon = 5 { \times } 1 0 ^ { - 4 }$  , box 0.01</td><td>fixed internal objective box</td></tr><tr><td>Metric record</td><td>every generation, at most 200 saved points</td><td>recording only</td></tr><tr><td>Held-out references</td><td>fixed 500 × 500, 10-seed set per time step</td><td>final metrics only</td></tr><tr><td>Model repeats</td><td>3 Workbench sequences × seeds 0–2</td><td>P010 routing and P015 cloud, fixed restart choices</td></tr></table>

Table 10: Settings for the reported experiments.

Stopping rule. Main evolutionary runs stop after at most 200 generations. From generation 40 onward, a run may stop after 25 generations without sufficient improvement. A scalar run requires a relative improvement above $5 \times 1 0 ^ { - 4 }$ in its best feasible objective. A Pareto run becomes eligible only with at least eight feasible archive members. The run’s objective box is then fixed with a 5% upper margin. The 25-generation count restarts after an internal HV gain above $5 \times 1 0 ^ { - 4 }$ . It also restarts after the front occupies a new normalized cell of width 0.01 or improves the normalized ideal point by more than $5 \times 1 0 ^ { - 4 }$ . Stopping uses these internal Workbench signals. Held-out HV and IGD are computed from saved candidates for final scoring and generation-wise plots.

## A.10 REPRODUCIBILITY MATERIALS

The supplementary materials contain the public requests and tables, prompts, generated Workbench examples, numerical settings, and saved result records. The evaluation package provides versioned scoring code and state-specific references for scoring saved plans. Aggregation combines these scores with the recorded acceptance prefixes. Provider credentials are excluded from the packages.

The evaluation package contains 195 states, scalar references, and 78 Pareto references from ten 500 × 500 searches per state, with reference manifests and file checksums. The primary aggregate uses 4,368 Pareto state–seed records, original acceptance indicators, and the retained static execution/objective summaries. Supplementary control records include P007–P015 matched restarts, 180 update transitions, 30 historical references, state-binding cases, independent P010/P015 model runs, and reference/archive checks. The time-step mapping and reference-objective audit checks agree with the saved records.

Episode-bootstrap intervals use 20,000 resamples (NumPy generator seed 20260905).

The original DeepSeek-V4-Pro (Preview) checkpoint is no longer available for reruns, so supplementary joint-Match measurements were not obtained for ReAct and OptimAI.

## A.11 AGGREGATE AND PER-PROBLEM RESULTS

Checking the information retained by Persistent ReAct. We inspect the saved inputs to verify that each update receives the preceding program and plan from the same run. The reported source is one original, unmerged branch selected by continuity of its carried state. All 141 records include successful final compilation and execution. Six records use repair, with ten repair calls in total and a cap of three per state. All 126 update inputs match the preceding publicly executable stage’s solver code, compact final answer, and public plan. The trace check also verifies cumulative dialogue, unchanged initial public tables, and the absence of another method’s search state. The evidence includes original responses, execution records, predecessor checksums, and per-state acceptance flags. The strict evaluation combines the saved output-protocol checks with mathematical validity for every external method. The mathematical-prefix diagnostic isolates the effect of auxiliary output fields.

Routing and cloud-placement results. Table 11 separates results for the two Pareto domains.
<table><tr><td>Method</td><td>Routing HV</td><td>Cloud HV</td><td>Overall HV</td><td>IGD↓</td></tr><tr><td>MAPLE</td><td>0.768</td><td>0.982</td><td>0.875</td><td>0.082</td></tr><tr><td>Always Warm</td><td>0.749</td><td>0.982</td><td>0.866</td><td>0.093</td></tr><tr><td>Always Full</td><td>0.665</td><td>0.984</td><td>0.825</td><td>0.112</td></tr><tr><td>w/o TSS</td><td>0.394</td><td>0.936</td><td>0.665</td><td>0.143</td></tr></table>

Table 11: Public-formula evaluation of saved Pareto plans. All rows cover t00–t12 and ten numerical seeds. Fixed policies follow their own histories. HV includes zero-valued invalid states. IGD is conditional on valid archives, with 780 states for each full-system policy and 630 for w/o TSS.

MAPLE’s overall HV is 0.875, compared with 0.866 for own-history Warm and 0.825 for own-history Full. The domain breakdown locates the gain: adaptive reuse helps routing, while Full is slightly stronger in cloud placement. The untyped Workbench can find strong cloud plans but loses validity on routing.

The following summaries cover all 15 episodes with population 200, at most 200 generations per state, and 10 seeds.

<table><tr><td>Episode Profile</td><td></td><td>HV mean ± SD</td><td>IGD mean ± SD</td></tr><tr><td>P010</td><td>Ordered service</td><td>0.754±0.076</td><td>0.154±0.051</td></tr><tr><td>P011</td><td>Ordered service</td><td>0.759±0.070</td><td>0.150±0.040</td></tr><tr><td>P012</td><td>Ordered service</td><td>0.790±0.052</td><td>0.157±0.040</td></tr><tr><td>P013</td><td>Cloud resources</td><td>0.986±0.004</td><td>0.011±0.004</td></tr><tr><td>P014</td><td>Cloud resources</td><td>0.972±0.003</td><td>0.013±0.001</td></tr><tr><td>P015</td><td>Cloud resources</td><td>0.989±0.003</td><td>0.010±0.001</td></tr></table>

Table 12: Public-formula Pareto quality by episode. Every row uses 10 seeds and 13 states per seed against the same freshly searched reference union. HV and IGD first average over time steps and then report variation across seeds.
<table><tr><td colspan="3">DS</td><td colspan="4">DM</td></tr><tr><td>Update type</td><td>States</td><td>Online quality</td><td>Update type</td><td>States</td><td>Online quality</td><td>IGD↓</td></tr><tr><td>All updates</td><td>108</td><td>0.947</td><td>All updates</td><td>72</td><td>0.879</td><td>0.079</td></tr><tr><td>Feasibility</td><td>3</td><td>0.721</td><td>Feasibility</td><td>15</td><td>0.722</td><td>0.163</td></tr><tr><td>Objective ordering</td><td>63</td><td>0.942</td><td>Objective ordering</td><td>48</td><td>0.836</td><td>0.104</td></tr><tr><td>Positive rescaling</td><td>0</td><td>一</td><td>Positive rescaling</td><td>12</td><td>0.940</td><td>0.044</td></tr><tr><td>Diagnostic only</td><td>6</td><td>1.000</td><td>Diagnostic only</td><td>12</td><td>0.988</td><td>0.010</td></tr><tr><td>Unchanged</td><td>39</td><td>0.948</td><td>Unchanged</td><td>0</td><td>1</td><td></td></tr><tr><td>Historical reference</td><td>21</td><td>0.996</td><td>Historical reference</td><td>9</td><td>0.894</td><td>0.070</td></tr></table>

Table 13: MAPLE results by audited update semantics, t01–t12. Ten-seed state means use the original scalar evaluation for DS and public formulas for DM. Labels may overlap; historical reference is a separate dimension. Diagnostic-only changes preserve objectives and hard constraints. Solve rate is 100% in every nonempty group; empty groups have unavailable scores.
<table><tr><td>NLDO profile</td><td>Method</td><td>Overall mean</td><td>Dynamic mean</td><td>Gap/HV</td><td>IGD↓</td></tr><tr><td>Selection/allocation</td><td>MAPLE (ours)</td><td>0.957</td><td>0.953</td><td>19.231</td><td>一</td></tr><tr><td>Precedence scheduling</td><td>MAPLE (ours)</td><td>0.965</td><td>0.962</td><td>11.038</td><td>1</td></tr><tr><td>Coverage rostering</td><td>MAPLE (ours)</td><td>0.933</td><td>0.927</td><td>298.920</td><td>一</td></tr><tr><td>Ordered service routing</td><td>MAPLE (ours)</td><td>0.752</td><td>0.757</td><td>0.757</td><td>0.165</td></tr><tr><td>Cloud-resource placement MAPLE (ours)</td><td></td><td>0.805</td><td>0.801</td><td>0.801</td><td>0.129</td></tr></table>

Table 14: Recorded-evaluator results by profile. Overall includes t00. Dynamic covers t01–t12. Scalar rows report objective gap. Pareto rows report HV and IGD. Public-formula results are in Table 12.

Effect of auxiliary output fields. We recompute the eligible prefix while ignoring auxiliary fields and keeping the saved plans and feasibility checks unchanged. Table 16 includes every saved attempt at t01–t12, including attempts after an earlier output-check failure. Persistent ReAct’s initial P004 program runs, but its schedule is infeasible and its output lacks a required reasoning field. Its accepted prefix therefore has length zero. Table 17 lists three feasible rosters rejected because required auxiliary fields were missing. Those omissions did not trigger code repair.

<table><tr><td>Episode Profile</td><td>Mean</td><td>Gap</td><td>HV</td><td></td><td>IGD↓</td></tr><tr><td>P001</td><td>Selection/allocation</td><td>1.000</td><td>0.000</td><td></td><td></td></tr><tr><td>P002</td><td>Selection/allocation</td><td>0.871</td><td>57.692</td><td></td><td></td></tr><tr><td>P003</td><td>Selection/allocation</td><td>1.000</td><td>0.000</td><td></td><td></td></tr><tr><td>P004</td><td>Precedence scheduling</td><td>0.894</td><td>33.115</td><td></td><td></td></tr><tr><td>P005</td><td>Precedence scheduling</td><td>1.000</td><td>0.000</td><td></td><td></td></tr><tr><td>P006</td><td>Precedence scheduling</td><td>1.000</td><td>0.000</td><td></td><td></td></tr><tr><td>P007</td><td>Coverage rostering</td><td>0.930</td><td>452.043</td><td></td><td></td></tr><tr><td>P008</td><td>Coverage rostering</td><td>1.000</td><td>-7.873</td><td></td><td></td></tr><tr><td>P009</td><td>Coverage rostering</td><td>0.868</td><td>452.592</td><td></td><td></td></tr><tr><td>P010</td><td>NLDO-DM, ordered service</td><td>0.747</td><td>0.415</td><td>0.747</td><td>0.165</td></tr><tr><td>P011</td><td>NLDO-DM, ordered service</td><td>0.733</td><td>0.417</td><td>0.733</td><td>0.187</td></tr><tr><td>P012</td><td>NLDO-DM, ordered service</td><td>0.777</td><td>0.407</td><td>0.777</td><td>0.157</td></tr><tr><td>P013</td><td>NLDO-DM, cloud resources</td><td>0.855</td><td>0.277</td><td>0.855</td><td>0.109</td></tr><tr><td>P014</td><td>NLDO-DM, cloud resources</td><td>0.771</td><td>0.322</td><td>0.771</td><td>0.130</td></tr><tr><td>P015</td><td>NLDO-DM, cloud resources</td><td>0.789</td><td>0.326</td><td>0.789</td><td>0.139</td></tr></table>

Table 15: Recorded-evaluator scores by episode. Ten seeds, t00–t12, 200 × 200 maximum budget. Scalar rows give objective gap; Pareto rows give recorded HV, ideal gap, and IGD. Table 12 gives public-formula Pareto scores.
<table><tr><td>Method</td><td>Attempted updates</td><td>Feas.| attempt</td><td>Valid prefx</td><td>Mean accepted update</td><td>DS quality</td><td>DM HV</td></tr><tr><td>Persistent ReAct</td><td>126/180</td><td>96.8%</td><td>58/180</td><td>count 3.87</td><td>0.469</td><td>0.024</td></tr><tr><td>ReAct</td><td>55/180</td><td>83.6%</td><td>31/180</td><td>2.07</td><td>0.287</td><td>0.000</td></tr><tr><td>OptiMUS</td><td>53/180</td><td>83.0%</td><td>31/180</td><td>2.07</td><td>0.231</td><td>0.056</td></tr><tr><td>ORLM</td><td>35/180</td><td>77.1%</td><td>5/180</td><td>0.33</td><td>0.046</td><td>0.000</td></tr><tr><td>OptimAI</td><td>18/180</td><td>61.1%</td><td>9/180</td><td>0.60</td><td>0.037</td><td>0.024</td></tr><tr><td>OR-LLM-Agent</td><td>26/180</td><td>65.4%</td><td>12/180</td><td>0.80</td><td>0.056</td><td>0.013</td></tr></table>

Table 16: Update-only continuity and quality, t01–t12. Feas.|attempt considers attempted updates. Mean accepted update count, E[horizon], counts updates in the strict prefix. Prefix, mean count, and quality assign zero to failed suffixes. DM uses public formulas. MAPLE’s DS quality and DM HV are 0.947 and 0.879. Table 2 includes initialization.
<table><tr><td></td><td>Episode First break Direct cause</td><td></td><td>Observable evidence</td></tr><tr><td>P007</td><td>t03</td><td>Missing output field</td><td>The code compiled and ran, but a required reasoning field was missing. Code-output repair did not check this field.</td></tr><tr><td>P008</td><td>t05</td><td>Missing output field</td><td>The code compiled and ran, but a required reasoning field was missing. Code-output repair did not check this field.</td></tr><tr><td>P009</td><td>t02</td><td>Missing output field</td><td>The code compiled and ran, but a required reasoning field was missing. Code-output repair did not check this field.</td></tr></table>

Table 17: First protocol failures in three Persistent ReAct rostering sequences. Each saved plan is mathematically feasible at the first rejected state, but omits a required reasoning field.

For every external method, the mathematical prefix $\begin{array} { r } { \tilde { r } _ { t } = \prod _ { j = 0 } ^ { t } v _ { j } } \end{array}$ accepts feasible plans regardless of auxiliary fields, retaining the same plans, public formulas, references, and zero missing suffixes. Persistent ReAct continued running after some auxiliary fields were omitted. Under mathematicalprefix scoring, its scalar solve rate is 85.5%, compared with 51.3% under strict scoring. Its Pareto solve rate is 42.3%, compared with 11.5%.

## A.12 CASE STUDY AND VISUAL RESULT ATLAS

Figures 8 and 9 show how accepted plans, Pareto fronts, and search traces change across successive routing and cloud-placement requests.

![](images/e0efde55a9ffb1330cdc1492c4e8e850a022d66a9b5f2aad27513964e6e41c20.jpg)

Figure 8: Ordered-service routing across t01–t12 (NLDO-P010). The upper/lower blocks show t01–t06/t07–t12. Teal diamonds show MAPLE’s ten-seed pooled Pareto set; amber circles show the recorded reference. Route panels show the seed-0 plan, zoomed to active orders; inactive orders are omitted. Front coordinates normalize distance and lateness within each stage. HV uses all three objectives, including emissions.

The third row follows normalized HV during search. Generation 0 is the population just after restart. Curves remain constant after termination. The dashed line marks the mean stopping generation.

![](images/2b56a3cd00456c54bdc2d282ff63f77e4d54a9af3bb145e5bf2af38acdce4432.jpg)  
Figure 9: Cloud-resource placement across t01–t12 (NLDO-P015). Blocks show t01–t06 and t07–t12. Diamonds show MAPLE’s ten-seed pooled energy–load-imbalance fronts; circles show the reference, normalized within each stage. Seed-0 plans show CPU utilization on machines 1–8: teal marks usage, pale bars mark unused capacity, and 0 marks idle machines. Ten-seed HV curves show the initial quality loss and recovery at t11/t12.

Accepted plans with lower solution quality. We inspect accepted plans whose normalized quality decreases after a revision. Accepted outputs pass the runtime’s public checks, but their evaluated quality can still be low. At P002-t10, the scalar objective changes from monetary cost to service time. The retained assignment remains feasible and has lower normalized quality under the revised objective. At cloud update t12, the machine identifiers remain the same, but workloads and resource properties change. A feasible placement can have lower quality under the revised problem.

## A.13 COMPONENT AND RESTART EVIDENCE

Agreement between predicted and saved edits. We compare LPD’s three edit decisions with the changes saved after each update. The saved decisions are those used at seed 0. All ten numerical seeds share the same LLM-generated data and Workbench. The analysis contains 180 language decisions. LPD predicts changes to data, decision definition, and evaluation. All three predictions match the accepted changes in 170 updates. The separate counts are 174, 178, and 178. The six data differences are explicit roster-table replacements at P007–P009 t11–t12. In the other four cases, LPD requests a cautious code edit but the model returns the function unchanged. Whenever a function changes, LPD has marked it for change.

Thirty benchmark updates refer to entities through earlier requests (Table 18). LSM resolves all 30, including six requests that require following three reference links across the update history.
<table><tr><td>Reference links</td><td>Updates</td><td>Entity/field</td><td>Value</td><td>Valid result</td></tr><tr><td>1</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>2</td><td>21</td><td>21</td><td>21</td><td>21</td></tr><tr><td>3</td><td>6</td><td>6</td><td>6</td><td>6</td></tr></table>

Table 18: Reference grounding with stored events. None of the 30 current requests states the target entity ID. Entity/field and value compare the accepted public-data change with the benchmark’s intended update. The intended answer is never shown to the language model.

These requests test whether earlier update records identify the required entities. The following control also tests requests that need assignments from the accepted plan.

Information needed to interpret assignment requests. We vary whether the model receives earlier update records, the accepted plan, both, or neither. We use saved routing states from P010 and cloud-placement states from P015 at t04, t08, and t10. The requests require three kinds of information. One kind identifies the requested assignment through earlier updates. Another names the relevant entities and asks to preserve assignments from the accepted plan. The third uses an earlier update to identify the entities and the accepted plan to retrieve their assignments. Each kind has three equivalent wordings in each domain, giving 18 language requests.

For example, the first routing request assigns the customer identified in a congestion note to the vehicle identified in an emissions update. The second names orders O001, O002, and O003 and preserves their saved vehicle assignments. The third preserves the saved vehicle assignment of the customer identified in the congestion note. Table 19 contains the recorded request texts.

A fifth condition supplies the required assignment through a fixed procedure. The required answers are stored separately for evaluation.

Saved plans used in the language tests and search replay. The original language tests and the search replay use representatives selected by different rules. The language tests read a representative reconstructed from each seed’s saved archive. They select the plan with the lowest recorded scalar fitness and break ties by archive order. The original online run selects its representative by a different rule. It stores the first member of the final NSGA-II population, ordered by nondomination rank, decreasing crowding distance, and violation score, with stable ties.

For the search replay, we use the saved query from the first wording and read assignments from the originally exported online plan. We run six cases with three search seeds under each of the five information conditions. This gives 18 numerical results per condition and 90 results in total. Each search uses a population of 200 and at most 200 generations. The public tables, Workbench, Warm choice, starting population, and search budget are held fixed across conditions. The requested assignment is imposed after the restart population is built.

<table><tr><td></td><td>Profile Information needed Public request q0</td><td></td></tr><tr><td>P010</td><td>Earlier update records</td><td>For the next dispatch window, the customer previously marked as the congestion anchor must be served by the vehicle whose telemetry update changed its emission rate. Treat this as a hard assignment constraint. All other feasibility rules and Pareto objectives stay unchanged.</td></tr><tr><td>P010</td><td>Accepted plan</td><td>For the next dispatch, keep the currently accepted vehicle assignments of orders 0001, 0002, and 0003. Other orders may move, and the original feasibility rules and Pareto objectives remain unchanged.</td></tr><tr><td>P010</td><td>Both records</td><td>Preserve the currently accepted vehicle assignment of the customer referred to as the congestion anchor. Other orders may be reassigned, and the original objectives remain unchanged.</td></tr><tr><td>P015</td><td>Earlier update records</td><td>For the next placement window, assign the first job named when workload group burst-2 was introduced to machine M07. This is a hard assignment constraint. All other feasibility rules and Pareto objectives stay unchanged.</td></tr><tr><td>P015</td><td>Accepted plan</td><td>For the next placement, keep the currently accepted machine assignments of jobs J012, J013, and J014. Other jobs may move, and the original feasibility rules and Pareto objectives remain unchanged.</td></tr><tr><td>P015</td><td>Both records</td><td>Keep the current accepted machine assignment of every job in the workload group called burst-2. Other jobs may move, and the original objectives remain unchanged.</td></tr></table>

Table 19: Requests that use different stored information. The released cases include all three wordings and identifiers for their source states.

Identifying and enforcing the requested assignment. We evaluate assignment identification in the language tests and assignment enforcement in a separate search replay. Full LSM resolves all 18 requests. With only earlier update records, the model resolves the six requests answerable from those records. With only the accepted plan, it resolves the six requests that explicitly name the relevant entities. Neither condition resolves the six requests that need both types of information. With neither record, no request is resolved. The supplied-assignment condition returns the required relation directly.

The numerical test requires the correct assignment to be retrieved and enforced. Every plan in the returned archive must satisfy the problem constraints and preserve that assignment. Missing or incorrect assignments and infeasible archives receive zero quality. Full LSM and the suppliedassignment control each produce 18 valid results and mean HV of 0.825. The earlier-records-only condition produces six valid results and mean HV of 0.261. The accepted-plan-only condition produces six valid results and mean HV of 0.258. Table 20 separates these numerical results from the original language tests.

Comparison of reconstructed and online plans. We compare the requested assignments in the reconstructed representatives with those in the original online plans. The reconstructed representative and the original online plan differ in 12 of the 18 case–seed states. Among the twelve cases that read an accepted plan, the requested assignments differ in five. The differences occur in the requests that name orders and preserve their assignments in P010 (seeds 0 and 1), and in the requests that use both records in P015 (seeds 0–2). The requested assignments agree in the other seven cases. The six requests answered from earlier update records do not read the accepted plan.

Checksums confirm that all 18 starting populations match those in the original control and are shared across its five conditions. In the thirteen cases whose requested assignments are unchanged, the replay reproduces the scored objective sets for all five conditions.

Program construction without TSS. Without TSS, the model defines its own candidate representation and the functions that generate, evaluate, modify, and repair candidates. It retains this program through the sequence. At t00, the model receives the public problem and the required Workbench functions. It receives no TSS decision types, encoding guidance, or type-specific search operators. It also defines data loading and conversion from a candidate to a solution. Under Warm, it can repair only solutions from its own preceding state. LPD still identifies which part of the problem changed.

(a) Identifying the requested assignment: original language tests
<table><tr><td>Information provided</td><td>Uses earlier updates</td><td>Uses the accepted plan</td><td>Uses both records</td></tr><tr><td>Full LSM</td><td>6/6</td><td>6/6</td><td>6/6</td></tr><tr><td>Earlier update records only</td><td>6/6</td><td>0/6</td><td>0/6</td></tr><tr><td>Accepted plan only</td><td>0/6</td><td>6/6</td><td>0/6</td></tr><tr><td>Current request only</td><td>0/6</td><td>0/6</td><td>0/6</td></tr><tr><td>Assignment supplied directly</td><td>6/6</td><td>6/6</td><td>6/6</td></tr></table>

(b) Preserving the assignment during search: replay results
<table><tr><td>Information provided</td><td>Valid search results</td><td>Mean HV</td></tr><tr><td>Full LSM</td><td>18/18</td><td>0.825</td></tr><tr><td>Earlier update records only</td><td>6/18</td><td>0.261</td></tr><tr><td>Accepted plan only</td><td>6/18</td><td>0.258</td></tr><tr><td>Current request only</td><td>0/18</td><td>0.000</td></tr><tr><td>Assignment supplied directly</td><td>18/18</td><td>0.825</td></tr></table>

Table 20: Using stored events and the accepted state. (a) Original grounding results across three wordings, using minimum-recorded-fitness archive representatives. (b) Replay of the saved q0 queries against original online plans: six cases and three numerical seeds per condition, scored by the recorded evaluator. Missing or incorrect assignments and infeasible archives receive zero. In the supplied-assignment condition, a fixed procedure provides the expected relation directly.

LSM still supplies stored events, the method’s own accepted state, and its search archives. The fixed optimization loop still summarizes the change, applies the same restart selector, performs selection, maintains the archive, checks stopping, and formats the output. Compilation, public validation, and the repair limit are unchanged. This control removes the typed representation, type-specific operators, and restricted editing interface together.

One generated Workbench sequence is used per episode. Seeds 0–9 evolve independent populations under that fixed model run. Across P010–P015, the six model runs contain 13 distinct Workbench versions. All 13 saved Workbench versions compile. The failing routing version omits vehicle uniqueness, producing 120 invalid update–seed outputs. The failing cloud version records GPU incompatibility without clearing the feasibility flag, producing 20 invalid outputs. The held-out evaluator assigns zero quality to these outputs.
<table><tr><td>Episode</td><td></td><td>Failed results Observed violation</td><td>Missing check</td></tr><tr><td>P010 (routing)</td><td>120/120</td><td>routes</td><td>one vehicle assigned to two vehicle uniqueness omitted from the evaluation function</td></tr><tr><td>P015 (cloud)</td><td>20/120</td><td>GPU-incompatible placements</td><td>violations recorded without clearing the feasible flag</td></tr></table>

Table 21: Hidden constraint failures of the persistent untyped Workbench. Counts use the hidden evaluator on the final result of every failed update–seed result. The Workbench reports all 140 results as feasible. The hidden evaluator finds a hard-constraint violation, so each result contributes zero to online quality.

Combining and modifying candidates. This test holds the optimization program, starting candidates, repair procedure, and restart choice fixed. Table 22 reports the results. At every P010/P015 update, both methods receive the same MAPLE population, Workbench, restart action, typed repair, and consistency checks. The control replaces search operators designed for permutations, assignments, and values. It instead selects a complete decision segment from one parent or samples a new valid segment at random.

<table><tr><td>Variation</td><td>Solve rate</td><td>P010 HV</td><td>P015 HV</td><td>All HV</td></tr><tr><td>Typed operators</td><td>100.0%</td><td>0.756</td><td>0.792</td><td>0.774</td></tr><tr><td>Generic random sampling</td><td>91.7%</td><td>0.657</td><td>0.579</td><td>0.618</td></tr></table>

Table 22: Comparison of type-specific search operators. Recorded-evaluator values cover t01–t12 with numerical seeds 0–2 under the same 200 × 200 limit and held-out reference. Only crossover and mutation change.

Variation across independently generated programs. We generate three program sequences for MAPLE and its w/o-TSS variant on P010 routing and P015 cloud placement. The three sequences include the original main sequence. Each sequence uses search seeds 0–2 across all twelve updates. The Warm–Full choices, evaluation functions from the original runs, 200 × 200 budget, archive limit of 500, and stopping rule remain fixed. Final scores use the same held-out 500 × 500 × 10 reference. A model run that does not produce a valid Workbench receives zero solve rate and online quality for its requested results.

<table><tr><td>Method</td><td>Model run</td><td>Execution completed</td><td>Solve rate (%)</td><td>Online quality</td><td>Quality if valid</td><td>Numerical SD</td></tr><tr><td>P010 (routing)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MAPLE</td><td>0</td><td>yes</td><td>100.0</td><td>0.756</td><td>0.756</td><td>0.120</td></tr><tr><td></td><td>1</td><td>yes</td><td>100.0</td><td>0.693</td><td>0.693</td><td>0.013</td></tr><tr><td></td><td>2</td><td>yes</td><td>100.0</td><td>0.560</td><td>0.560</td><td>0.038</td></tr><tr><td></td><td>All</td><td>3/3</td><td>100.0 ±0.0</td><td>0.669 ±0.100</td><td>0.669</td><td>0.057</td></tr><tr><td>MAPLE w/o TSS</td><td>0</td><td>yes</td><td>0.0</td><td>0.000</td><td>一</td><td>0.000</td></tr><tr><td></td><td>1</td><td>yes</td><td>100.0</td><td>0.435</td><td>0.435</td><td>0.016</td></tr><tr><td></td><td>2</td><td>yes</td><td>100.0</td><td>0.572</td><td>0.572</td><td>0.029</td></tr><tr><td></td><td>All</td><td>3/3</td><td>66.7 ±57.7</td><td>0.336 ±0.299</td><td>0.503</td><td>0.015</td></tr><tr><td>P015 (cloud)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MAPLE</td><td>0</td><td>yes</td><td>100.0</td><td>0.792</td><td>0.792</td><td>0.022</td></tr><tr><td></td><td>1</td><td>yes</td><td>83.3</td><td>0.708</td><td>0.850</td><td>0.004</td></tr><tr><td></td><td>2</td><td>yes</td><td>83.3</td><td>0.685</td><td>0.822</td><td>0.010</td></tr><tr><td></td><td>All</td><td>3/3</td><td>88.9 ±9.6</td><td>0.728 ±0.056</td><td>0.819</td><td>0.012</td></tr><tr><td>MAPLE w/o TSS</td><td>0</td><td>yes</td><td>83.3</td><td>0.751</td><td>0.901</td><td>0.011</td></tr><tr><td></td><td>1</td><td>yes</td><td>100.0</td><td>0.059</td><td>0.059</td><td>0.001</td></tr><tr><td></td><td>2</td><td>yes</td><td>83.3</td><td>0.729</td><td>0.874</td><td>0.019</td></tr><tr><td></td><td>All</td><td>3/3</td><td>88.9 ±9.6</td><td>0.513 ±0.393</td><td>0.577</td><td>0.011</td></tr></table>

Table 23: Independent model runs on one routing and one cloud episode. Recorded-evaluator control. Each run independently generates a Workbench sequence. Numerical seeds 0–2 then search that sequence over t01–t12. Warm–Full choices remain fixed, so only Workbench construction and maintenance vary. Solve rate uses the held-out validity check. Online quality sets invalid results to zero. The All rows report mean ± sample standard deviation across model runs. Quality if valid includes accepted states only. Numerical SD measures variation across search seeds within one model run.

Across three Workbench runs, MAPLE’s mean online quality is 0.669 in routing and 0.728 in cloud, compared with 0.336 and 0.513 without TSS. All three typed routing runs remain valid. The two additional typed cloud runs fail the held-out check at t11–t12. Table 23 reports model-run variation separately from numerical-seed variation for these two episodes.

Evaluation with a second language model. We repeat the full benchmark with Kimi k2.7 using the same evaluation and search settings. Table 24 reports the results under the common public-formula evaluation protocol.
<table><tr><td rowspan="2">Language model</td><td colspan="2">DS</td><td colspan="2">DM</td></tr><tr><td>Solve rate</td><td>Online quality</td><td>Solve rate</td><td>Online quality</td></tr><tr><td>MAPLE (DeepSeek-V4-Pro Preview)</td><td>100.0%</td><td>0.951</td><td>100.0%</td><td>0.875</td></tr><tr><td>MAPLE (Kimi k2.7)</td><td>100.0%</td><td>0.937</td><td>92.3%</td><td>0.766</td></tr></table>

Table 24: Full-benchmark MAPLE replication across language models. Both rows use the complete MAPLE system with one model run per episode and ten numerical seeds. Public-formula evaluation, search budget, and stopping rule match Table 2.

Model calls and tokens for program construction. We group recorded calls into program construction, identification of requested edits, and input-table updates. The grouping uses each recorded call’s system prompt. Tables 25–26 report means over four additional runs per method on P010/P015. Each run covers twelve updates with three search seeds and fixed restart choices. Code construction includes generation, modification, and repair. Token counts include both input and output. The w/o-TSS runs reuse saved localization decisions and data patches.

<table><tr><td>Method</td><td>Code calls</td><td>Code tokens</td><td>Tokens for identifying requested edits</td><td>Tokens for updating input tables</td></tr><tr><td>MAPLE</td><td>4.0</td><td>51.6k</td><td>68.9k</td><td>135.0k</td></tr><tr><td>MAPLE w/o TSS</td><td>4.8</td><td>72.8k</td><td>Reused</td><td>Reused</td></tr></table>

Table 25: Code construction uses fewer tokens with TSS. Reused denotes update processing supplied from existing records.

Initial code generation averages 19.1k tokens with TSS and 26.0k without it. Code modification averages 32.5k and 46.9k, respectively, including repair calls in both categories. The combined reduction is computed before rounding.

<table><tr><td>Method</td><td>Recorded scope</td><td>Tokens Calls</td><td></td><td>Provider min.</td><td>Wall min.</td></tr><tr><td>MAPLE</td><td>Code, localization, data patches</td><td>255.4k</td><td>27.8</td><td>14.7</td><td>19.9</td></tr><tr><td>w/o TSS</td><td>Code; supplied update processing</td><td>72.8k</td><td>4.8</td><td>6.5</td><td>11.1</td></tr></table>

Table 26: Execution metadata for the same runs as Table 25. Wall time includes numerical search.

Reuse strategies over complete update sequences. Each strategy starts from the same initial state and continues with candidates produced by its own searches. The initial state is the typed Workbench and population at t00. Each strategy proceeds through t01–t12. Always Full keeps the TSS Workbench but draws all 200 candidates anew at every update. The policies use the same saved Workbench at each state: 60 initial execution instances and 720 update execution instances across six episodes and ten numerical seeds.

Across six episodes, the mean paired gain over Always Warm is 0.010. Five episode means are positive. The episode-cluster bootstrap interval is [0.001, 0.021], and the two-sided exact sign test gives p = 0.219. The gain is concentrated in routing. The mean gain over Always Full is 0.054. Table 11 reports the results by domain.

Recorded reuse and restart decisions. We report the initialization action selected before search for each saved update.

Across P007–P015, the selector chooses Warm at t01–t10 for all nine episodes. It chooses Full at t11 for all nine. At t12, it chooses Full for P008 and P010–P015 and Warm for P007/P009. The total is 16 Full and 92 Warm decisions. These decisions are made before numerical optimization and are identical across the 10 numerical seeds. The disruptive updates appear at t11–t12 by design. At P010–P012 t08, the selector chooses Warm and cites the retained decision representation, active entities, and objective declarations. On the main Pareto trajectories, the deterministic table-change rule makes the same Warm–Full choices as the selector.

<table><tr><td>Sequential policy</td><td>Full restarts</td><td>Solve rate</td><td>Online quality</td><td>Generations</td><td>∆ quality vs. APLE</td></tr><tr><td>MAPLE (ours)</td><td>12/72</td><td>100.0%</td><td>0.879</td><td>162.8</td><td></td></tr><tr><td>Always Warm</td><td>0/72</td><td>100.0%</td><td>0.868</td><td>162.8</td><td>-0.010</td></tr><tr><td>Always Full</td><td>72/72</td><td>100.0%</td><td>0.824</td><td>199.0</td><td>-0.054</td></tr></table>

Table 27: Restart policies that follow their own populations on P010–P015. Public-formula evaluation of t01–t12. Warm and Full carry their own results through all updates. Every policy uses the same TSS Workbenches, 200 × 200 maximum budget, 10 seeds, and stopping rule. Generations are taken from the original executions. Paired differences are computed before rounding. Displayed means are rounded separately.

Restart decisions based on changed input cells. We compare MAPLE with a rule that starts afresh when at least half of the counted input cells change. For every table with an active field, it counts the cells in rows whose current active value is true, excluding the id and active fields themselves. This count is the table’s denominator. Rows are matched by their raw public id values. A row with an identifier absent from the preceding table, or previously inactive, contributes all its counted cells to the numerator. For other rows, only unequal cell values count as changed. The rule uses the maximum table fraction, with a zero fraction for an empty denominator. The rule chooses Full when this fraction is at least 0.5 and Warm otherwise. Its choices are fixed before the Warm and Full results are examined. Across P010–P015, the ratio ranges from 0 to 0.060 on ordinary updates and from 0.705 to 1.000 at t11–t12. The rule therefore matches all 12 Full decisions from the restart selector. It obtains the same 0.779 HV under the recorded evaluator and 162.8 mean generations without a model call or a new optimizer run.

Reuse after rescaling, renaming, and task addition. Starting from saved routing and cloudplacement states, we rescale values, rename entities, or add small tasks. Each additional test starts at t05 of P010 or P015. Each affects more than half of the counted table cells. The editable Workbench functions, decision types, and required output remain unchanged, while the number of active decisions grows when tasks are added. A fixed procedure constructs the tables and identifier mappings without model calls.

Rescaling routing parameters. For routing rescaling, depot and order coordinates, order demand, ready/due/service times, vehicle capacity, and shift end are multiplied by 10. Priorities and emission rates stay fixed. The historical Workbench identifies congestion by the original coordinate pair (48, 11) (Figure 6). After coordinate rescaling, that match disappears, so this branch changes the incident’s effect as well as the parameter scale. Its table-change ratio is 0.86.

Rescaling cloud parameters. For cloud rescaling, job CPU and memory demands, machine CPU and memory capacities, and idle energy are multiplied by 10. Job deadlines, latency sensitivities, and priorities are also multiplied by 10 but enter only diagnostics in this Workbench. GPU requirements, availability, per-CPU energy rates, and global energy multipliers stay fixed. For any fixed feasible assignment, utilization and the recorded imbalance B are unchanged, while energy becomes $E ^ { \prime } =$ 10E. Thus feasibility and Pareto dominance are preserved, although scalar Warm repair uses the changed weighting 10E + B. The table-change ratio is 0.71.

Renaming entities. Relabeling consistently renames orders and vehicles, or jobs and machines, without changing their other attributes (ratio 1.00). The table-change rule does not apply the identifier mapping before matching rows, so every renamed row is treated as new even though its non-identifier values are retained.

Adding tasks. Task additions introduce 20 low-demand orders near the depot or 26 small CPU/memory jobs with loose deadlines (ratios 0.53 and 0.52). These additions enlarge the task while retaining the existing resources and earlier demands.

Comparing the restart choices. Warm and Full begin with the same saved candidate population and use three search seeds. They use the original objective definitions and the combined nondominated reference for each additional test. The table-change rule and the numerical rule that evaluates candidates before updating their identifiers choose Full on all six cases. The restart selector chooses Warm on all six and cites the rescaling, renaming, or easily placed additions in its recorded assessment.

Warm obtains mean HV of 0.862 across the six cases, compared with 0.682 for Full. The largest difference is +0.708 for Warm (Table 28). The six cases cover routing and cloud placement. The routing-rescaling case also changes how congestion is applied. P015 recovers almost fully from fresh restarts, so most of the quality difference comes from routing. Warm uses fewer generations in five of the six branches. Table 4 presents the relabeling and task-addition results compactly alongside the disruptive main-sequence updates. The rescaling branches and the pure-scaling numerical control below are supplementary diagnostics.
<table><tr><td>Case</td><td>Warm HV</td><td>Full HV</td><td>Paired ∆ HV  $\mathbf { M e a n } \pm \mathbf { S D }$ </td><td>Min. ΔHV</td><td>Max. ΔHV</td></tr><tr><td>P010 parameter rescaling</td><td>0.716</td><td>0.510</td><td> $+ 0 . 2 0 6 \pm 0 . 4 3 3$ </td><td>-0.294</td><td>+0.478</td></tr><tr><td>P010 ID relabel</td><td>0.752</td><td>0.603</td><td> $+ 0 . 1 4 9 \pm 0 . 1 9 6$ </td><td>-0.060</td><td>+0.329</td></tr><tr><td>P010: Added low-demand orders</td><td>0.769</td><td>0.061</td><td> $+ 0 . 7 0 8 \pm 0 . 2 3 6$ </td><td>+0.442</td><td>+0.892</td></tr><tr><td>P015 parameter rescaling</td><td>0.989</td><td>0.990</td><td> $- 0 . 0 0 2 \pm 0 . 0 0 3$ </td><td>-0.005</td><td>+0.001</td></tr><tr><td>P015 ID relabel</td><td>0.980</td><td>0.980</td><td> $0 . 0 0 0 \pm 0 . 0 1 2$ </td><td>-0.013</td><td>+0.008</td></tr><tr><td>P015: Added small jobs</td><td>0.968</td><td>0.946</td><td> $+ 0 . 0 2 2 \pm 0 . 0 2 1$ </td><td>+0.005</td><td>+0.045</td></tr><tr><td>All six cases</td><td>0.862</td><td>0.682</td><td>+0.181</td><td>一</td><td>一</td></tr></table>

Table 28: Reuse after rescaling, relabeling, and task additions. Six branches start at t05, with three matched seeds per action and 36 executions. For each branch, $\Delta _ { s } = \mathrm { H V } _ { \mathrm { W a r m } , s } - \mathrm { H V } _ { \mathrm { F u l l } , s }$ uses the same incoming population and branch-specific pooled reference. SD is the sample deviation of the three paired differences. Min./max. are observed seed extrema, not confidence bounds. The last row averages the six branch means. Recorded-evaluator, action-matched control. Routing rescaling also removes the coordinate-based congestion match. Rounding follows Table 27.

For each branch, $\begin{array} { r } { \bar { \Delta } = \frac { 1 } { 3 } \sum _ { s = 0 } ^ { 2 } \Delta _ { i } } \end{array}$ <sub>s</sub> and $\begin{array} { r } { s _ { \Delta } = \sqrt { \frac { 1 } { 2 } \sum _ { s = 0 } ^ { 2 } ( \Delta _ { s } - \bar { \Delta } ) ^ { 2 } } } \end{array}$ . The three routing task-addition differences are all positive, ranging from 0.442 to 0.892. Routing rescaling and relabeling have positive means but each includes a negative seed difference. The cloud differences are small, consistent with the similar final quality and earlier Warm stopping discussed above.

Updating candidate identifiers before evaluation. We replace the identifiers in each old candidate before evaluating it against the renamed tables. For relabeling, we compare $f _ { \mathrm { o l d } } ( x )$ with $f _ { \mathrm { n e w } } ( T ( x ) )$ where T applies the public ID mapping to assignment keys, assigned resources, and route permutations. The numerical selector retains its original sample indices and thresholds. All 1,200 incoming candidates across the two profiles and three seeds preserve their objective vectors and feasibility after mapping. The six sampled decisions change from Full to Warm: both objective displacement and feasibility loss become zero.

Reuse under uniform routing rescaling. We keep the congestion rule attached to the same order before and after scaling, then compare Warm and Full. We replace the coordinate lookup with the same public order ID in both pre-update and scaled Workbenches. The incident’s historical effect and all other recorded scoring rules remain fixed. On all 600 incoming candidates, the unscaled rewrite reproduces the historical evaluator, and scaling multiplies each objective by 10 while preserving feasibility. The numerical selector has $r = 0 . 9$ and chooses Full for all three seeds.

Warm has higher HV for every routing seed in these controls. The cloud difference is small: the three paired gains are 0.0054, −0.0023, and 0.0015. In the pure-scaling case, Warm uses 97.7 generations on average and Full uses 200. The generation counts do not include the extra evaluations used to adapt and repair old candidates.

The 18 restart-and-search calls take 170.6 seconds in total, including migration and repair but excluding source-state reconstruction and scoring.

<table><tr><td>Additional control</td><td>Warm HV</td><td>Full HV</td><td>Paired ∆ HV</td></tr><tr><td>P010: Candidates updated to renamed IDs</td><td>0.786</td><td>0.395</td><td> $+ 0 . 3 9 1 \pm 0 . 2 7 8$ </td></tr><tr><td>P015: Candidates updated to renamed IDs</td><td>0.984</td><td>0.982</td><td> $+ 0 . 0 0 2 \pm 0 . 0 0 4$ </td></tr><tr><td>P010 pure scaling</td><td>0.791</td><td>0.466</td><td> $+ 0 . 3 2 5 \pm 0 . 2 9 9$ </td></tr></table>

Table 29: Reuse with consistent representations. Eighteen new searches: three branches, three paired seeds, and two actions, with population 200 and a 200-generation cap. Each branch uses the pooled nondominated reference from its six searches. Differences are Warm minus Full, reported as mean ± sample SD.
<table><tr><td>Control</td><td></td><td>Seed Warm HV</td><td>Full HV</td><td>Δ HV</td><td>Generations</td></tr><tr><td rowspan="2">P010: Candidates updated to renamed IDs</td><td>0</td><td>0.848</td><td>0.310</td><td>+0.538</td><td>75/200</td></tr><tr><td>1</td><td>0.966</td><td>0.402</td><td>+0.564</td><td>70/200</td></tr><tr><td rowspan="4">P015: Candidates updated to renamed IDs</td><td>2</td><td>0.545</td><td>0.474</td><td>+0.071</td><td>158/200</td></tr><tr><td>0</td><td>0.983</td><td>0.977</td><td>+0.005</td><td>176/200</td></tr><tr><td>1</td><td>0.983</td><td>0.986</td><td>-0.002</td><td>200/200</td></tr><tr><td>2</td><td>0.985</td><td>0.984</td><td>+0.002</td><td>200/200</td></tr><tr><td rowspan="2">P010 pure scaling</td><td>0 1</td><td>0.840 0.971</td><td>0.649 0.303</td><td>+0.191</td><td>65/200</td></tr><tr><td>2</td><td>0.562</td><td>0.446</td><td>+0.668</td><td>70/200</td></tr><tr><td></td><td></td><td></td><td></td><td>+0.116</td><td>158/200</td></tr></table>

Table 30: Individual seeds for the offline controls. Generations are Warm/Full. Identical incomingpopulation hashes are checked within each action pair; mapped-ID Full archives also reproduce all six historical Full archives exactly.

Restart actions from a shared candidate population. For each update, both restart actions receive MAPLE’s same saved candidate population. These are the Matched Warm and Matched Full controls. Only MAPLE’s selected result advances the continuing trajectory. All action-matched comparisons use the recorded evaluator with population 200, a 200-generation cap, seeds 0–9, the same stopping rule, and the same held-out $5 0 0 { \times } 5 0 0 { \times } 1 0$ reference for final scoring. The Warm/Full comparison contains 1,080 paired episode–time-step–seed results per method over P007–P015. The numerical selector covers the 720 Pareto results on P010–P015.
<table><tr><td>Policy</td><td>Full restarts</td><td>HV</td><td>Generations</td><td>Gap to best action</td></tr><tr><td>MAPLE</td><td>120/720</td><td>0.779</td><td>162.8</td><td>0.004</td></tr><tr><td>Matched Warm</td><td>0/720</td><td>0.766</td><td>162.7</td><td>0.017</td></tr><tr><td>Numerical selector</td><td>245/720</td><td>0.765</td><td>169.4</td><td>0.018</td></tr><tr><td>Matched Full</td><td>720/720</td><td>0.731</td><td>199.0</td><td>0.053</td></tr><tr><td>Best action after results</td><td>300/720</td><td>0.783</td><td>167.8</td><td>0.000</td></tr></table>

Table 31: Action-matched restart under the recorded evaluator. All policies solve every state.

Re-evaluating old candidates to select a restart action. We evaluate a sample of earlier candidates under the old and revised problems before selecting Warm or Full. This comparison follows a common response to change in dynamic evolutionary optimization (Deb et al., 2007; Liu et al., 2014; Sahmoud & Topcuoglu, 2019). For each seed and update, it samples 20 of the 200 incoming candidates without replacement. It evaluates the same candidates with the old and updated Workbenches. For objective j, it computes

$$
r _ { j } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \frac { | f _ { j } ( x _ { i } , t ) - f _ { j } ( x _ { i } , t - 1 ) | } { \operatorname* { m a x } ( | f _ { j } ( x _ { i } , t ) | , | f _ { j } ( x _ { i } , t - 1 ) | , 1 0 ^ { - 1 2 } ) } , \qquad r = \operatorname* { m a x } _ { j } r _ { j } .
$$

K is the number of retained valid objective pairs. It chooses Full when $r \geq 0 . 5$ . It also chooses Full when at least half of the previously feasible sampled candidates become infeasible, the objective

dimension changes, or no valid objective pair remains. A valid pair contains finite objective vectors of the same nonzero length. If the old evaluation fails, the rule uses the saved fitness. Other unsuccessful pairs are omitted. It chooses Warm otherwise. The maximum over objectives keeps one large change from being averaged away. The relative denominator makes objective units comparable. Sampling is fixed by the numerical seed and time step. The numerical selector evaluates each unchanged incoming genome under both Workbenches before migration or ID mapping. All six relabeling seed cases retain 20 defined objective pairs, but lose feasibility for all sampled candidates and have r = 1. Their Full decisions therefore include representation mismatch. The mapped-candidate control in Appendix A.13 removes this mismatch and changes all six decisions to Warm.
<table><tr><td>Episode</td><td>Full t01–t10</td><td>Full t11–t12</td><td>Numerical HV</td><td>MAPLE HV</td><td>Matched Warm HV</td></tr><tr><td>P010</td><td>23/100</td><td>20/20</td><td>0.729</td><td>0.752</td><td>0.724</td></tr><tr><td>P011</td><td>24/100</td><td>20/20</td><td>0.710</td><td>0.736</td><td>0.712</td></tr><tr><td>P012</td><td>28/100</td><td>20/20</td><td>0.745</td><td>0.785</td><td>0.764</td></tr><tr><td>P013</td><td>10/100</td><td>20/20</td><td>0.855</td><td>0.854</td><td>0.853</td></tr><tr><td>P014</td><td>20/100</td><td>20/20</td><td>0.768</td><td>0.763</td><td>0.761</td></tr><tr><td>P015</td><td>20/100</td><td>20/20</td><td>0.785</td><td>0.785</td><td>0.784</td></tr><tr><td>All</td><td>125/600</td><td>120/120</td><td>0.765</td><td>0.779</td><td>0.766</td></tr></table>

Table 32: Numerical change as a restart signal. Recorded-evaluator, action-matched control over t01–t12. Counts are episode–time-step–seed results. The numerical selector chooses Full on all 120 disruptive update–seed cases and on 125 of the 600 ordinary cases.

Appendix A.15 reports the threshold comparison.
<table><tr><td>Variant</td><td>DS Quality</td><td>DS Gen.</td><td>DM HV</td><td>t11 HV</td><td>t12 HV</td><td>DM Gen.</td></tr><tr><td>MAPLE</td><td>0.927</td><td>66.7</td><td>0.779</td><td>0.561</td><td>0.426</td><td>162.8</td></tr><tr><td>Matched Warm</td><td>0.927</td><td>66.8</td><td>0.766</td><td>0.509</td><td>0.327</td><td>162.7</td></tr><tr><td>Matched Full</td><td>0.919</td><td>71.1</td><td>0.731</td><td>0.561</td><td>0.426</td><td>199.0</td></tr></table>

Table 33: Recorded-evaluator, action-matched control over t01–t12. Restart actions from the same incoming population. All methods solve all 360 DS and 720 DM time-step–seed results. Quality is normalized scalar score for DS and held-out-reference HV ratio for DM. Gen. is the mean generation count. The t11/t12 columns show MAPLE’s 0.053/0.099 gains over Matched Warm.

Table 33 reports stage-end solution quality and the number of evolutionary generations for each restart policy. On P010–P015, MAPLE improves over Matched Warm by 0.013 normalized HV (episode-cluster 95% bootstrap interval [0.004, 0.022]), with the gain concentrated at t11–t12. All six episode means favor MAPLE over Matched Warm (two-sided exact sign test, $p = 0 . 0 3 1 )$ . MAPLE recovers 74% of the improvement obtained by choosing the better action after seeing both outcomes. The mean HV gain over Matched Full is 0.048 (hierarchical paired 95% CI [0.005, 0.097]). MAPLE uses 162.8 generations on average, compared with 199.0 for Matched Full. At t11–t12, where the selector chooses Full for all six Pareto episodes, it matches the corresponding Matched Full controls. On P007–P009, MAPLE and Matched Warm have nearly the same result. Always using Full is slightly worse and uses more generations.

The per-episode breakdown in Table 35 localizes this effect. Across routing episodes P010–P012, the disruptive-update gain over Matched Warm ranges from 0.122 to 0.168 HV. The corresponding cloud gains are 0.002–0.011, consistent with the stronger benefit of fresh initialization in routing.

Routing Family mean: M 0.757, N 0.391, W 0.733, F 0.656
<table><tr><td rowspan="2">Time</td><td colspan="4">P010</td><td colspan="4">P011</td><td colspan="4">P012</td></tr><tr><td>M</td><td>N</td><td>W</td><td>F</td><td>M</td><td>N</td><td>W</td><td>F</td><td>M</td><td>N</td><td>W</td><td>F</td></tr><tr><td>All-t</td><td>0.752</td><td>0.000</td><td>0.724</td><td>0.663</td><td>0.736</td><td>0.327</td><td>0.712</td><td>0.662</td><td>0.785</td><td>0.845</td><td>0.764</td><td>0.643</td></tr><tr><td>t01</td><td>0.746</td><td>0.000</td><td>0.746</td><td>0.710</td><td>0.731</td><td>0.379</td><td>0.731</td><td>0.663</td><td>0.847</td><td>0.995</td><td>0.847</td><td>0.684</td></tr><tr><td>t02</td><td>0.774</td><td>0.000</td><td>0.774</td><td>0.657</td><td>0.776</td><td>0.492</td><td>0.776</td><td>0.726</td><td>0.872</td><td>0.838</td><td>0.872</td><td>0.744</td></tr><tr><td>t03</td><td>0.791</td><td>0.000</td><td>0.791</td><td>0.769</td><td>0.778</td><td>0.440</td><td>0.778</td><td>0.730</td><td>0.815</td><td>0.880</td><td>0.815</td><td>0.642</td></tr><tr><td>t04</td><td>0.778</td><td>0.000</td><td>0.778</td><td>0.696</td><td>0.768</td><td>0.450</td><td>0.768</td><td>0.704</td><td>0.832</td><td>0.931</td><td>0.832</td><td>0.659</td></tr><tr><td>t05</td><td>0.805</td><td>0.000</td><td>0.805</td><td>0.668</td><td>0.797</td><td>0.397</td><td>0.797</td><td>0.692</td><td>0.838</td><td>0.869</td><td>0.838</td><td>0.640</td></tr><tr><td>t06</td><td>0.849</td><td>0.000</td><td>0.849</td><td>0.751</td><td>0.842</td><td>0.241</td><td>0.842</td><td>0.772</td><td>0.872</td><td>0.895</td><td>0.872</td><td>0.723</td></tr><tr><td>t07</td><td>0.850</td><td>0.000</td><td>0.850</td><td>0.733</td><td>0.846</td><td>0.289</td><td>0.846</td><td>0.736</td><td>0.876</td><td>0.940</td><td>0.876</td><td>0.688</td></tr><tr><td>t08</td><td>0.862</td><td>0.000</td><td>0.862</td><td>0.764</td><td>0.784</td><td>0.463</td><td>0.784</td><td>0.629</td><td>0.905</td><td>0.931</td><td>0.905</td><td>0.699</td></tr><tr><td>t09</td><td>0.886</td><td>0.000</td><td>0.886</td><td>0.659</td><td>0.804</td><td>0.532</td><td>0.804</td><td>0.715</td><td>0.891</td><td>0.938</td><td>0.891</td><td>0.706</td></tr><tr><td>t10</td><td>0.910</td><td>0.000</td><td>0.910</td><td>0.785</td><td>0.918</td><td>0.238</td><td>0.918</td><td>0.797</td><td>0.898</td><td>0.797</td><td>0.898</td><td>0.766</td></tr><tr><td>t11</td><td>0.342</td><td>0.000</td><td>0.230</td><td>0.342</td><td>0.239</td><td>0.000</td><td>0.040</td><td>0.239</td><td>0.397</td><td>0.656</td><td>0.395</td><td>0.397</td></tr><tr><td>t12</td><td>0.427</td><td>0.000</td><td>0.203</td><td>0.427</td><td>0.545</td><td>0.000</td><td>0.458</td><td>0.545</td><td>0.370</td><td>0.469</td><td>0.129</td><td>0.370</td></tr></table>

Cloud Family mean: M 0.801, N 0.821, W 0.799, F 0.805
<table><tr><td rowspan="2">Time</td><td colspan="4">P013</td><td colspan="4">P014</td><td colspan="4">P015</td></tr><tr><td>M</td><td>N</td><td>W</td><td>F</td><td>M</td><td>N</td><td>W</td><td>F</td><td>M</td><td>N</td><td>W</td><td>F</td></tr><tr><td>All-t</td><td>0.854</td><td>0.837</td><td>0.853</td><td>0.851</td><td>0.763</td><td>0.870</td><td>0.761</td><td>0.773</td><td>0.785</td><td>0.755</td><td>0.784</td><td>0.791</td></tr><tr><td>t01</td><td>0.915</td><td>0.813</td><td>0.915</td><td>0.873</td><td>0.815</td><td>0.905</td><td>0.815</td><td>0.833</td><td>0.826</td><td>0.883</td><td>0.826</td><td>0.816</td></tr><tr><td>t02</td><td>0.917</td><td>0.901</td><td>0.917</td><td>0.929</td><td>0.743</td><td>0.880</td><td>0.743</td><td>0.763</td><td>0.821</td><td>0.896</td><td>0.821</td><td>0.836</td></tr><tr><td>t03</td><td>0.899</td><td>0.874</td><td>0.899</td><td>0.901</td><td>0.782</td><td>0.881</td><td>0.782</td><td>0.764</td><td>0.846</td><td>0.915</td><td>0.846</td><td>0.859</td></tr><tr><td>t04</td><td>0.879</td><td>0.878</td><td>0.879</td><td>0.898</td><td>0.770</td><td>0.892</td><td>0.770</td><td>0.795</td><td>0.833</td><td>0.900</td><td>0.833</td><td>0.835</td></tr><tr><td>t05</td><td>0.911</td><td>0.902</td><td>0.911</td><td>0.913</td><td>0.823</td><td>0.927</td><td>0.823</td><td>0.840</td><td>0.834</td><td>0.905</td><td>0.834</td><td>0.854</td></tr><tr><td>t06</td><td>0.898</td><td>0.888</td><td>0.898</td><td>0.909</td><td>0.843</td><td>0.927</td><td>0.843</td><td>0.860</td><td>0.846</td><td>0.908</td><td>0.846</td><td>0.859</td></tr><tr><td>t07</td><td>0.924</td><td>0.914</td><td>0.924</td><td>0.924</td><td>0.748</td><td>0.907</td><td>0.748</td><td>0.786</td><td>0.813</td><td>0.900</td><td>0.813</td><td>0.840</td></tr><tr><td>t08</td><td>0.887</td><td>0.880</td><td>0.887</td><td>0.885</td><td>0.749</td><td>0.897</td><td>0.749</td><td>0.781</td><td>0.845</td><td>0.915</td><td>0.845</td><td>0.837</td></tr><tr><td>t09</td><td>0.882</td><td>0.866</td><td>0.882</td><td>0.873</td><td>0.751</td><td>0.921</td><td>0.751</td><td>0.725</td><td>0.834</td><td>0.912</td><td>0.834</td><td>0.831</td></tr><tr><td>t10</td><td>0.891</td><td>0.866</td><td>0.891</td><td>0.870</td><td>0.823</td><td>0.921</td><td>0.823</td><td>0.825</td><td>0.871</td><td>0.920</td><td>0.871</td><td>0.867</td></tr><tr><td>t11</td><td>0.768</td><td>0.780</td><td>0.763</td><td>0.768</td><td>0.762</td><td>0.770</td><td>0.767</td><td>0.762</td><td>0.858</td><td>0.000</td><td>0.857</td><td>0.858</td></tr><tr><td>t12</td><td>0.473</td><td>0.486</td><td>0.475</td><td>0.473</td><td>0.546</td><td>0.617</td><td>0.520</td><td>0.546</td><td>0.197</td><td>0.000</td><td>0.178</td><td>0.197</td></tr></table>

Table 34: Recorded-evaluator results at each Pareto update. M denotes MAPLE and N its independently generated untyped Workbench control. W and F denote Matched Warm and Matched Full, each initialized from MAPLE’s incoming population. Values are ten-seed HV means, with invalid outputs set to zero. All-t averages t01–t12. Family means pool routing or cloud results.

<table><tr><td colspan="4">All t01–t12</td><td colspan="3">Disruptive updates t11–t12</td></tr><tr><td>Episode</td><td>MAPLE</td><td>Matched Warm</td><td>Matched Full</td><td>MAPLE</td><td>Matched Warm</td><td>Matched Full</td></tr><tr><td>P010</td><td>0.752</td><td>0.724</td><td>0.663</td><td>0.385</td><td>0.217</td><td>0.385</td></tr><tr><td>P011</td><td>0.736</td><td>0.712</td><td>0.662</td><td>0.392</td><td>0.249</td><td>0.392</td></tr><tr><td>P012</td><td>0.785</td><td>0.764</td><td>0.643</td><td>0.384</td><td>0.262</td><td>0.384</td></tr><tr><td>P013</td><td>0.854</td><td>0.853</td><td>0.851</td><td>0.621</td><td>0.619</td><td>0.621</td></tr><tr><td>P014</td><td>0.763</td><td>0.761</td><td>0.773</td><td>0.654</td><td>0.644</td><td>0.654</td></tr><tr><td>P015</td><td>0.785</td><td>0.784</td><td>0.791</td><td>0.528</td><td>0.517</td><td>0.528</td></tr></table>

Table 35: Online quality for the six Pareto episodes. Recorded-evaluator, action-matched control. All three policies have 100% solve rate. Each value is the normalized-HV mean over 10 seeds after infeasible results are set to zero. The right block isolates the disruptive t11–t12 updates.

<table><tr><td>Scalar restart policy</td><td>Full restarts</td><td>Online quality</td><td>Generations</td><td>Gap to best action</td></tr><tr><td>MAPLE</td><td>4/36</td><td>0.927</td><td>66.7</td><td>0.016</td></tr><tr><td>Matched Warm</td><td>0/36</td><td>0.927</td><td>66.8</td><td>0.016</td></tr><tr><td>Matched Full</td><td>36/36</td><td>0.919</td><td>71.1</td><td>0.024</td></tr><tr><td>Best action after results</td><td>6/36</td><td>0.943</td><td>66.5</td><td>0.000</td></tr></table>

Table 36: Restart results on P007–P009. Recorded-evaluator, action-matched t01–t12. All policies solve every state. MAPLE and Matched Warm both reach mean scalar quality 0.927. Gap uses the post-hoc best action.

![](images/d6109c69c7d7e9077bda1fe647120759910820435cf914daad183b8646cf3b32.jpg)

Disruptive update. At t11–t12, the routing episode changes orders, depot, vehicles, and operating rules while retaining its representation and objectives. Current action. The restart selector selects Full. It matches the Matched Full control because both use the same action and incoming state, and exceeds Matched Warm under this recorded evaluator.

Without TSS. The untyped Workbench assigns one vehicle to two routes, violating a hard constraint.

Figure 10: Recorded within-search behavior at disruptive routing updates. Ten-seed feasibilitygated HV means, standard-deviation bands, and mean stopping generations use the recorded reference. Figure 5 reports final quality under the fresh common public-formula references.  
Curves remain constant after the marked stopping generation. Initial quality and recovery time differ: at t11–t12, Full can start below Warm and finish above it.  
![](images/d1205ec5fc50044f130e44a94eef7bee49a51f4756dc238fd12be3117b806662.jpg)  
Figure 11: Green-routing component results (NLDO-P010, t01–t12). Recorded-evaluator mean HV over ten seeds, standard-deviation bands, and mean stopping generations (triangles). MAPLE and Matched Warm/Full remain feasible. Without TSS, duplicate vehicle assignment makes every P010 update infeasible with zero quality.

## A.14 STATIC SOLVING RESULTS

Tables 37 and 38 report the supplementary diagnostics defined in Appendix A.7.
<table><tr><td>Method</td><td>Output pass</td><td>Compile</td><td>Run</td><td>Objective</td><td>Named match</td><td>Joint Match</td></tr><tr><td>MAPLE static</td><td>33.9%</td><td>93.2%</td><td>93.2%</td><td>59.3%</td><td>37.3%</td><td>59.3%</td></tr><tr><td>Persistent ReAct</td><td>37.3%</td><td>100.0%</td><td>98.3%</td><td>62.7%</td><td>44.1%</td><td>62.7%</td></tr><tr><td>ReAct</td><td>27.1%</td><td>91.5%</td><td>78.0%</td><td>57.6%</td><td>32.2%</td><td></td></tr><tr><td>OptiMUS workflow</td><td>69.5%</td><td>96.6%</td><td>93.2%</td><td>69.5%</td><td>一</td><td>69.5%</td></tr><tr><td>ORLM workflow</td><td>23.7%</td><td>84.7%</td><td>61.0%</td><td>23.7%</td><td></td><td>23.7%</td></tr><tr><td>OptimAI</td><td>47.5%</td><td>100.0%</td><td>98.3%</td><td>61.0%</td><td>49.2%</td><td></td></tr><tr><td>OR-LLM-Agent workflow</td><td>50.8%</td><td>96.6%</td><td>96.6%</td><td>50.8%</td><td></td><td>50.8%</td></tr></table>

Table 37: NLP4LP-hard diagnostic checks. Output pass records each workflow’s original acceptance flag. Named-variable agreement is reported separately. Joint Match requires execution and reference-objective/status agreement.
<table><tr><td>Method</td><td>Output pass</td><td>Compile</td><td>Run</td><td>Objective</td><td>Named match</td><td>Joint Match</td></tr><tr><td>MAPLE static</td><td>80.0%</td><td>100.0%</td><td>100.0%</td><td>80.0%</td><td>n.a.</td><td>80.0%</td></tr><tr><td>Persistent ReAct</td><td>70.0%</td><td>100.0%</td><td>100.0%</td><td>70.0%</td><td>n.a.</td><td>70.0%</td></tr><tr><td>ReAct</td><td>75.0%</td><td>100.0%</td><td>95.0%</td><td>70.0%</td><td>n.a.</td><td></td></tr><tr><td>OptiMUS workflow</td><td>65.0%</td><td>100.0%</td><td>100.0%</td><td>65.0%</td><td>n.a.</td><td>65.0%</td></tr><tr><td>ORLM workflow</td><td>45.0%</td><td>90.0%</td><td>55.0%</td><td>45.0%</td><td>n.a.</td><td>45.0%</td></tr><tr><td>OptimAI</td><td>60.0%</td><td>80.0%</td><td>80.0%</td><td>55.0%</td><td>n.a.</td><td></td></tr><tr><td>OR-LLM-Agent workflow</td><td>65.0%</td><td>95.0%</td><td>95.0%</td><td>65.0%</td><td>n.a.</td><td>65.0%</td></tr></table>

Table 38: BWOR20 diagnostic checks. Output pass, numeric objective agreement, and joint Match are distinct recorded criteria. Named-variable matching does not apply.

The two static benchmarks expose different weaknesses. On BWOR20, all 20 MAPLE rows compile and run, and 16 match the reference objective. On NLP4LP-hard, 55 of 59 rows compile and run, 35 match the objective, and 20 pass the additional historical output check.

## A.15 REFERENCE AND ARCHIVE SENSITIVITY

Sensitivity to the reference set and evaluation box. We vary the reference set and evaluation box and recompute the reported HV ratios.

Table 39 examines two choices that remain when the Pareto reference is finite. For reference subsampling, each method uses the same fixed subset in each of 20 repeats. The evaluation box from the full reference remains fixed. For box sensitivity, both numerator and denominator are recomputed after expanding the ideal-to-reference span.

Under the recorded evaluator, five w/o-TSS outputs at P012-t01 exceed HV ratio 1. Expanding the HV box reduces the number of ratios above 1 from five to four and then one. The 500-member archive cap is inactive for the MAPLE results in this analysis.

Effect of the Pareto archive limit. We repeat the search with archive limits of 100, 200, and 500 while keeping the other settings fixed.

Caps 100 and 200 truncate the archive in 240 and 162 results, respectively, while cap 500 is inactive. Their mean HV ratios are 0.775, 0.776, and 0.774. The largest change in the mean is 0.0016. Mean IGD is 0.139, 0.139, and 0.138. The smaller limits do change individual fronts. Their mean absolute HV differences from limit 500 are 0.053 and 0.040. They also use 5.3 and 2.8 more generations on average. The mean is stable, but archive size can still matter at an individual time step.

<table><tr><td colspan="4">Reference subsampling: ∆HV ratio</td><td colspan="3">HV-box span: mean ratio</td></tr><tr><td>Method</td><td>75%</td><td>50%</td><td>25%</td><td>1.00×</td><td>1.10×</td><td>1.25×</td></tr><tr><td>MAPLE</td><td>+0.005</td><td>+0.013</td><td>+0.039</td><td>0.779</td><td>0.796</td><td>0.818</td></tr><tr><td>Always Warm</td><td>+0.005</td><td>+0.013</td><td>+0.039</td><td>0.769</td><td>0.786</td><td>0.808</td></tr><tr><td>Always Full</td><td>+0.005</td><td>+0.012</td><td>+0.036</td><td>0.731</td><td>0.752</td><td>0.779</td></tr><tr><td>MAPLE w/o TSS</td><td>+0.004</td><td>+0.010</td><td>+0.028</td><td>0.606</td><td>0.621</td><td>0.640</td></tr></table>

Table 39: Sensitivity to the finite Pareto reference on P010–P015, t01–t12. Recorded-evaluator results. Each method contributes 720 time-step–seed results. The Warm and Full rows follow their own populations through the sequence. Reference subsampling reports the change from the fullreference mean. The box columns recompute both volumes. Method ordering is unchanged in every setting. For MAPLE w/o TSS, the number of results above one changes from 5 to 4 to 1 as the box expands. The maximum changes from 1.020 to 1.014 to 1.008.
<table><tr><td>Cap</td><td>P014 HV</td><td>P015 HV</td><td>Mean HV</td><td>IGD</td><td>Gen.</td><td>Active/results</td></tr><tr><td>100</td><td>0.760</td><td>0.791</td><td>0.775</td><td>0.139</td><td>188.4</td><td>240/240</td></tr><tr><td>200</td><td>0.771</td><td>0.780</td><td>0.776</td><td>0.139</td><td>185.9</td><td>162/240</td></tr><tr><td>500</td><td>0.763</td><td>0.785</td><td>0.774</td><td>0.138</td><td>183.1</td><td>0/240</td></tr></table>

Table 40: Sensitivity to the archive limit on P014/P015 under the recorded evaluator. Each row contains the same 240 update–seed results. The restart choices, 200 × 200 search budget, stopping rule, and held-out reference remain fixed. Only the maximum archive size changes.

Sensitivity to restart thresholds. We vary the numerical-change threshold and identify the interval that leaves the table-change decisions unchanged. The numerical selector uses a change threshold of 0.5 in the reported comparison. For thresholds of 0.25, 0.50, and 0.75, the numerical selector obtains HV ratios of 0.755, 0.765, and 0.767, with Full rates of 48.8%, 34.0%, and 25.0%. All three HV ratios are below MAPLE’s 0.779 under the recorded evaluator. At threshold 0.50 it uses 169.4 generations, compared with 162.7 for Matched Warm and 162.8 for MAPLE. The table-change threshold lies in a ten-fold gap. Ordinary updates score 0–0.060, while disruptive updates score 0.705–1.000. Any value in [0.3, 0.7] therefore makes identical decisions. Tables 40 and 39 separately test the archive limit and Pareto reference.

Sensitivity to congestion and return-deadline checks. We separately recompute scores after changing the congestion scope and enforcing the return deadline. Under the recorded reference, extending congestion to adjacent departure/return legs changes routing HV from 0.7471 to 0.7466 without changing MAPLE’s ordering relative to Warm/Full. Four saved candidate occurrences violate the return deadline. Enforcing this deadline leaves the primary aggregate HV unchanged.

## A.16 USING MAPLE THROUGH AN INTERACTIVE INTERFACE

The interface lets users submit a planning request, inspect the solutions, and revise the same problem. The application is called MAPLE Harness. Each session retains the Workbench, candidate plans, objective traces, and restart actions across updates. Figure 12 shows a CNC scheduling session from the published walkthrough, separate from NLDO.

The interface sends planning requests to a separate service that runs MAPLE through the Model Context Protocol (MCP). Each request returns a job identifier, which the interface uses to display search progress. After the result is accepted, the service saves the updated session. Users can inspect the solutions and export them as CSV or JSON.

![](images/a8291874f0b00efb85cf737057ff8809a3151e56f7b488341b70be1fc9fa2283.jpg)  
Figure 12: A complete Harness interaction from request to accepted update. Recorded steps: (a) scheduling request and public tables; (b) initial traces and eight Pareto solutions; (c) J01’s priority changes from 2.0 to 2.5, followed by a Warm update; (d) revised traces and fifteen Pareto solutions.

## A.17 PROMPT INTERFACES AND EXECUTION RULES

This section summarizes the recorded prompt interfaces, required output fields, and state-transfer rules. The TSS prompts specify two editable functions and the required table format. The other interfaces specify the functions used by search. Field names and response delimiters below retain their recorded spelling. Problem text, public tables, update history, previous output, and repair messages fill the variable fields. Table 9 defines the carried-state interfaces.

![](images/b61fb47c44fa8bb11a1576c92e4338519c239e46015154ce2aa0b245363ed6f1.jpg)

Rules. Do not generate a separate model schema, GA/NSGA code, a large data contract, or unused helper code. Do not hard-code entity ids, resource names, source ids, hidden references, or benchmark-specific constants. Loop over public table rows. Submitted solution identifiers must be scalar public IDs, not full row objects.

Public task definition. If objective\_mode=multi\_objective, the prompt requires solver\_mode="moea" and objective names/order exactly matching the public task definition. Scalarization and literal constant Pareto dimensions are forbidden.

Repair turn. If compilation or smoke execution fails, the next prompt includes the previous public error plus the previous slot contents and asks for corrected Workbench slots.

## MAPLE update localizer

## dynamic classification prompt

System. Classify the dynamic update for a fixed LiveOpt Workbench. Return exactly one JSON object.

Input. Natural-language update, current segment signature, current public-context summary, representative accepted decision, public event/reference ledger, and allowed restart-skill names. The search population and archive are not shown.

Required JSON fields. data\_update, patch\_setup, patch\_fitness, restart\_skill, reason, and state\_binding\_queries.

Decision rules. Use patch\_setup only when search segments or data extraction must change. Use patch\_fitness when objectives, constraints, penalties, decode, or solution reporting must change. If public tables changed but the existing Workbench slots loop over public\_context, prefer data\_update=true and patch\_setup=false.

State binding. For an explicit cross-stage assignment hold, return a typed lock\_assignment, preserve\_demand\_assignments, or preserve\_resource\_assignments query and leave the code-patch flags false unless another part of the request changes the Workbench. Fixed code resolves accepted assignment values and checks them against the current typed segment.

Restart hint. One allowed restart identifier for recording the choice. The active run replaces it with the separately checked semantic Warm–Full selector.

## MAPLE semantic restart selector

## reuse-risk assessment prompt

System. Assess whether repaired solutions near the previous population are likely to remain competitive.

Input. An opaque update id, the public update text, and a fixed summary containing changed public field paths, representative row changes, old/new decision types, and old/new public objective declarations. The selector also keeps the initial request and earlier accepted public updates in its conversation context.

Required JSON. reuse\_risk, change\_mechanisms, full\_vote, evidence\_paths, and reason.   
Evidence paths must exactly match the digest.

Decision rule. Warm already mixes at most 50% repaired history with fresh candidates. Full is reserved for a supported replacement of key decisions, reversal of objective preferences or resource roles, new constraint system, or risk that good solutions move far from the old population. Ordinary changes and uncertainty return Warm.

Boundary. The prompt explicitly forbids trial optimization, benchmark/time-step identity, difficulty labels, search-action advice in the event text, hidden references, HV, and IGD. The gate accepts Full when the assessment reports high reuse risk, casts a Full vote, includes a structurally supported mechanism, and cites at least one changed field path.

## MAPLE public-table updater

## natural language to public table patch

System. Patch public optimization data from a natural-language update. Return exactly one JSON object.

Writable root. Only public\_context["tables"]. Every table is a list[dict]. Singleton tables such as policy, settings, depot, or global parameters are still one-row lists.

Operations. update\_row, append\_row, delete\_row, set\_value, replace\_value, and replace\_table.

Required shape. A compact JSON object with operations, where each operation records op, path, optional match, row/value/values, and a public reason.

Repair turn. If an operation misses a row, has the wrong path, or violates the list-of-dicts ABI, the next prompt includes the error and previous patch JSON and asks for a corrected patch.

## MAPLE TSS Workbench editor

slot-level code patch

System. Patch only the requested Workbench slots. Return code blocks only.

Input. Current Workbench functions, updated public context, public task definition, predicted changes, restart choice selected outside code, and previous validation error if any.

Slot rule. Return only requested slots. If public data extraction/search segments are unaffected, do not edit the decision-definition function. If candidate evaluation is unaffected, do not edit the evaluation function. Restart logic is never implemented inside the slots.

Output rule. Preserve solver mode, objective names, and objective-vector order required by the public task definition. If new public policy values appear, pass them through the data reader and read them in the evaluator. Do not hard-code update constants.

Validation. Returned functions are compiled, run on a small public example, and checked against the public task definition before they are accepted.

## MAPLE w/o TSS

## untyped Workbench construction

System. Build a complete persistent Python optimization artifact from scratch.

Input. The public natural-language problem and normalized public context. No TSS slot, SegmentSpec, candidate schema, encoding hint, or operator strategy is supplied.

Required callbacks. build\_state, initialize, evaluate\_decode, crossover, mutate, and coerce\_repair. The model chooses the representation and implements every callback.

Fixed optimization loop. Selection, nondominated sorting, archive maintenance, early stopping, restart selection, and the evolutionary loop remain fixed and cannot be reimplemented by the generated program.

Update and repair. Later prompts provide the complete accepted artifact, the new public update/context, and any compile or smoke-test error, then request one complete replacement Python block.

## External baseline time-step prompts

## public-stage instructions

User task. Solve the current NLDO stage from public information only. The prompt then gives the initial request, public tables, every public update through the selected time step, the public task definition, and the required solution fields.

State boundary. The saved source contains 127 initial or early rows based on each original method and 135 continuation rows. Continuation receives only that method’s own dialogue, preceding accepted output,

and candidate population. No row receives a LiveOpt population, TSS state, restart decision, hidden change, reference archive, or hidden objective/checker implementation.

Executable output. Every method returns strict JSON containing executable Python solver code and a final solution. A method may also return a candidate archive on Pareto tasks. The runner compiles the code, executes it on the public tables, and parses its JSON output.

Repair. Compile, runtime, or output-schema failures receive at most three code-output repairs. The repair prompt includes public execution feedback but never hidden feasibility or reference quality.

## ReAct+Public Tools

## instruction based on the original method

System. You are a ReAct+Tools optimization agent. Use a Thought -> public tool-plan -> validation pattern, then return executable solver code and the final answer. Return JSON only.

Required fields. thought, react\_trace, tool\_plan, solver\_code, validation\_plan, and final\_answer.

## Persistent ReAct

## stateful public baseline

System. You are a Persistent ReAct+Tools optimization agent. Same Thought → tool-plan → validation pattern and required fields as ReAct+Public Tools, plus: You keep a persistent code workspace across stages. . . Apply the smallest necessary edits for the new public update while keeping parts that are still valid. . . Always return the COMPLETE updated solver code, never a diff. You have no TSS workbench slots, no candidate population or archive, no restart interface, and no LiveOpt artifacts.

## OptiMUS and ORLM

## modeling and code instructions based on the original methods

OptiMUS system. You are an OptiMUS-style optimization code generator. Produce Gurobi solver code and debug it. Return JSON only.

ORLM system. You are an ORLM-style optimization modeler. Produce a mathematical model and COPT solver code. Return JSON only.

Required fields. formulation, solver\_choice, solver\_plan, solver\_code, validation\_plan, and final\_answer. Public execution uses the available local solver route rather than exposing evaluation-only code.

## OptimAI

## multi-role instruction based on the original method

System. You are an OptimAI-style optimization agent. Formulate the task, propose multiple solver-code plans, select a plan, and include a debug/validation plan. Return JSON only.

Required fields. formulation, candidate\_solver\_plans, selected\_plan, solver\_code, debug\_validation\_plan, and final\_answer.

## OR-LLM-Agent

## executed OR-LLM-Agent instruction

System. You are an OR-LLM-Agent-style agent. Produce reasoning/modeling, formulation, code-generation, self-verification, and self-repair artifacts for the public optimization task. Return JSON only.

Required fields. reasoning\_modeling\_trace, formulation, code\_generation\_plan, solver\_code, self\_verification, self\_repair, and final\_answer.

Static adapters use the shared DeepSeek model with COPT for ORLM and Gurobi for OptiMUS and OR-LLM-Agent. For the five non-persistent external NLDO workflows, 127 initial/early rows use method-based prompts and 135 continuation rows expose their own preceding outputs and candidates when available. Sixteen earlier repair rows also contain their own preceding accepted output.

## A.18 RECORDED PUBLIC NLDO REQUESTS AND UPDATES

This section reproduces the business-language requests from the saved executions, before their shared evaluation reminder. The annotations describe how each update changes the evaluated problem. F marks a change to feasibility. O marks objective changes that can alter the ordering of candidate solutions. R marks positive objective rescaling that preserves dominance. D marks changes to reported diagnostics only. U marks an unchanged mathematical problem. M independently marks a historical-reference requirement and can overlap any semantic label. The labels do not enter agent prompts. Table 13 aggregates these audited labels. Exact reproduction uses the original machine-readable requests, public tables, output contracts, and checksums.

## NLDO-P001 (NLDO; COBENCH)

Initial public request. The instance data is provided in attached tables; treat the listed rows and columns as the source of truth. Select a feasible subset or portfolio under fixed capacity. Later updates revise public item values but do not change capacity, item weights, or eligibility fields. The constraints table gives the current objective\_mode. Expose a minimization scalar suitable for the generic solver. Select a feasible portfolio under fixed capacity 34. Items: I01 value 15 weight 7, I02 value 22 weight 3, I03 value 10 weight 8, I04 value 17 weight 4, I05 value 24 weight 9, I06 value 12 weight 5, I07 value 19 weight 10, I08 value 26 weight 6, I09 value 14 weight 2, I10 value 21 weight 7, I11 value 9 weight 3, I12 value 16 weight 8. Maximize value; later natural-language updates may revise public item values but do not change capacity or eligibility.

## Public updates.

t01 O A sponsor review increases item I03’s public value by 6. The capacity and item eligibility rules stay unchanged.

t02 O A risk review lowers item I07’s public value by 5; it remains selectable under the same capacity rule.

t03 O Quality audit increases item I10’s public value by 4 while the feasible portfolio definition remains fixed.

t04 O,M The item praised in the sponsor review receives another value increase of 6; infer which item that was from memory.

t05 O,M The item from the risk review recovers 3 value points, while the sponsor-reviewed item gains 1 more point; resolve both items from memory.

t06 U Operations asks for a checkpoint only: keep the current public objective and hard constraints, revalidate the plan, and change it only if it is no longer feasible.

t07 U No new business event is reported for this stage. Refresh the solution against the current public tables.

t08 U Treat this as a routine review. The model, objective terms, and public rows stay as they are; preserve feasibility under the current state.

t09 U The planning team wants the current public contract revalidated without adding new constraints.

t10 O A late value-model reset changes several public item scores without changing capacity or eligibility. Item I10 receives a value bonus of 10, item I11 receives a value penalty of 8, and item I05 receives a value bonus of 10. Rebuild the portfolio under the same feasible subset definition rather than carrying forward the old value ranking.

t11 U Operations asks for a checkpoint only: keep the current public objective and hard constraints, revalidate the plan, and change it only if it is no longer feasible.

t12 U No new business event is reported for this stage. Refresh the solution against the current public tables.

## NLDO-P002 (NLDO; COBENCH)

Initial public request. The instance data is provided in attached tables; treat the listed rows and columns as the source of truth. Choose open facilities and assign each customer to an open facility using only public facilities, customers, and constraints tables. Respect fixed facility capacity; later updates revise public opening/service costs but do not change capacity, demand, or facility eligibility. The constraints table gives the current objective\_mode.

Expose a minimization scalar suitable for the generic solver. Choose facilities and assign customers to open facilities. Facilities: F1 open cost 19 capacity 22, F2 open cost 23 capacity 26, F3 open cost 27 capacity 30, F4 open cost 18 capacity 18, F5 open cost 22 capacity 22. Customers: C01 demand 3, C02 demand 4, C03 demand 5, C04 demand 6, C05 demand 2, C06 demand 3, C07 demand 4, C08 demand 5, C09 demand 6, C10 demand 2. Minimize opening plus service cost; later public updates may revise costs while capacities, customer demand, and eligibility stay fixed.

## Public updates.

t01 O Facility F2’s opening cost increases by 4 after an energy-price review; capacity and facility eligibility stay unchanged.

t02 O Serving customer C03 from facility F1 becomes 3 cost units cheaper per demand unit after a local contract update.

t03 O Facility F5’s opening cost decreases by 5 because of a regional service credit.

t04 O,M The facility from the regional service credit receives one more opening-cost discount of 2; infer that facility from memory.

t05 O,M The customer-facility pair from the local contract update receives another 1 unit service-cost discount, while the facility from the energy-price review gets 1 more opening-cost penalty; resolve both references from memory.

t06 U Operations asks for a checkpoint only: keep the current public objective and hard constraints, revalidate the plan, and change it only if it is no longer feasible.

t07 U No new business event is reported for this stage. Refresh the solution against the current public tables.

t08 U Treat this as a routine review. The model, objective terms, and public rows stay as they are; preserve feasibility under the current state.

t09 U The planning team wants the current public contract revalidated without adding new constraints.

t10 O A same-day service-level reset flips the scalar objective. The public facility rows, customer rows, demand values, and capacity limits remain unchanged, but stop minimizing monetary opening plus service cost. The new objective is to minimize total service time using the public serve\_time columns, even if that requires a more expensive assignment.

t11 U Operations asks for a checkpoint only: keep the current public objective and hard constraints, revalidate the plan, and change it only if it is no longer feasible.

t12 U No new business event is reported for this stage. Refresh the solution against the current public tables.

## NLDO-P003 (NLDO; COBENCH)

Initial public request. The instance data is provided in attached tables; treat the listed rows and columns as the source of truth. Select a minimum-cost set collection using only public sets, elements, and constraints tables. Cover the fixed active element universe; later updates revise public set costs but do not change coverage requirements or set eligibility. Expose a minimization scalar suitable for the generic solver. Select a minimumcost collection of public sets that covers the fixed active element universe. Elements: E01, E02, E03, E04, E05, E06, E07, E08, E09, E10. Sets: S01 cost 9 covers E02/E05/E07/E08, S02 cost 14 covers E01/E04/E07/E10, S03 cost 6 covers E03/E06/E07/E09, S04 cost 11 covers E02/E05/E07/E08, S05 cost 16 covers E01/E04/E07/E10, S06 cost 8 covers E01/E02/E03/E04/E05/E06/E07/E08/E09/E10, S07 cost 13 covers E02/E05/E07/E08, S08 cost 5 covers E01/E04/E07/E10. Later updates may revise public set costs but do not add/remove coverage requirements or set eligibility.

## Public updates.

t01 O Set S03 receives a public service-quality rebate of 3 cost units; all element coverage requirements stay unchanged.

t02 O Set S06 becomes 2 cost units cheaper because it carries a compliance certificate.

t03 O Provider audit increases set S02’s public cost by 4, but the set remains available and the coverage universe is unchanged.

t04 O,M The set carrying the compliance certificate becomes 3 cost units cheaper; infer which set that was from memory.

t05 O,M The audited set receives a 1 cost-unit correction, and the high-priority rebate set receives another 2 unit discount; resolve both sets from memory.

t06 U Operations asks for a checkpoint only: keep the current public objective and hard constraints, revalidate the plan, and change it only if it is no longer feasible.

t07 U No new business event is reported for this stage. Refresh the solution against the current public tables.

t08 U Treat this as a routine review. The model, objective terms, and public rows stay as they are; preserve feasibility under the current state.

t09 U The planning team wants the current public contract revalidated without adding new constraints.

t10 U Run a maintenance update with the same public contract. Revalidate feasibility and quality under the current public state.

t11 U Operations asks for a checkpoint only: keep the current public objective and hard constraints, revalidate the plan, and change it only if it is no longer feasible.

t12 U No new business event is reported for this stage. Refresh the solution against the current public tables.

## NLDO-P004 (NLDO; FJSP)

Initial public request. The instance data is provided in attached tables; treat the listed rows and columns as the source of truth. Create a flexible job-shop schedule. Assign every operation to one eligible resource, respect fixed job precedence and resource capacity, and use settings.objective\_mode for the current scalar objective. Later updates may revise soft due/priority values or machine cost\_rate but do not change machine availability, operation eligibility, or processing requirements. Plan-change disruption is reported only as a diagnostic and never defines hard feasibility. Schedule flexible job-shop operations with precedence and machine capacity constraints. Minimize weighted tardiness; report disruption only as an auxiliary diagnostic. J1 due 28: J1O1 can run on M3/M1 for 9 minutes, J1O2 can run on M1/M2 for 4 minutes, J1O3 can run on M2/M3 for 6 minutes J2 due 32: J2O1 can run on M1/M2 for 5 minutes, J2O2 can run on M2/M3 for 7 minutes, J2O3 can run on M3/M1 for 9 minutes J3 due 36: J3O1 can run on M2/M3 for 8 minutes, J3O2 can run on M3/M1 for 10 minutes, J3O3 can run on M1/M2 for 5 minutes

## Public updates.

t01 D Machine M1’s public cost\_rate increases by 1. All machines remain available and operation eligibility is unchanged.

t02 O J1 becomes urgent: set priority to 3 and tighten its soft due time by 5 minutes. This changes tardiness cost but not schedule feasibility.

t03 O,M For the job that became urgent in the previous note, increase its priority by 1 again; infer the job from memory.

t04 O,M The same urgent job receives a 3-minute soft due-time relaxation after customer confirmation; infer the job from memory.

t05 D,M The machine from the earliest cost-rate note receives another 0.5 cost-rate increase; infer it from memory and re-optimize the same feasible schedule space.

t06 U Operations asks for a checkpoint only: keep the current public objective and hard constraints, revalidate the plan, and change it only if it is no longer feasible.

t07 U No new business event is reported for this stage. Refresh the solution against the current public tables.

t08 U Treat this as a routine review. The model, objective terms, and public rows stay as they are; preserve feasibility under the current state.

t09 U The planning team wants the current public contract revalidated without adding new constraints.

t10 O A late accounting reset flips the scalar objective. All machines remain available, and due dates remain soft diagnostics. Stop minimizing tardiness; the new objective is to minimize total machine operating cost using each machine’s public cost\_rate. This may prefer cheaper machines even when the schedule finishes later.

t11 U Operations asks for a checkpoint only: keep the current public objective and hard constraints, revalidate the plan, and change it only if it is no longer feasible

t12 U No new business event is reported for this stage. Refresh the solution against the current public tables.

## NLDO-P005 (NLDO; FJSP)

Initial public request. The instance data is provided in attached tables; treat the listed rows and columns as the source of truth. Create a flexible job-shop schedule. Assign every operation to one eligible resource, respect fixed job precedence and resource capacity, and use settings.objective\_mode for the current scalar objective. Later updates may revise soft due/priority values or machine cost\_rate but do not change machine availability, operation eligibility, or processing requirements. Plan-change disruption is reported only as a diagnostic and never defines hard feasibility. Schedule flexible job-shop operations with precedence and machine capacity constraints. Minimize weighted tardiness; report disruption only as an auxiliary diagnostic. J1 due 29: J1O1 can run on M1/M2 for 10 minutes, J1O2 can run on M2/M3 for 5 minutes, J1O3 can run on M3/M1 for 7 minutes J2 due 33: J2O1 can run on M2/M3 for 6 minutes, J2O2 can run on M3/M1 for 8 minutes, J2O3 can run on M1/M2 for 10 minutes J3 due 37: J3O1 can run on M3/M1 for 9 minutes, J3O2 can run on M1/M2 for 4 minutes, J3O3 can run on M2/M3 for 6 minutes

## Public updates.

t01 D Machine M2’s public cost\_rate increases by 2. All machines remain available and operation eligibility is unchanged.

t02 O J2 becomes urgent: set priority to 3 and tighten its soft due time by 5 minutes. This changes tardiness cost but not schedule feasibility.

t03 O,M For the job that became urgent in the previous note, increase its priority by 1 again; infer the job from memory.

t04 O,M The same urgent job receives a 3-minute soft due-time relaxation after customer confirmation; infer the job from memory.

t05 D,M The machine from the earliest cost-rate note receives another 0.5 cost-rate increase; infer it from memory and re-optimize the same feasible schedule space.

t06 U Operations asks for a checkpoint only: keep the current public objective and hard constraints, revalidate the plan, and change it only if it is no longer feasible.

t07 U No new business event is reported for this stage. Refresh the solution against the current public tables.

t08 U Treat this as a routine review. The model, objective terms, and public rows stay as they are; preserve feasibility under the current state.

t09 U The planning team wants the current public contract revalidated without adding new constraints.

t10 U Run a maintenance update with the same public contract. Revalidate feasibility and quality under the current public state.

t11 U Operations asks for a checkpoint only: keep the current public objective and hard constraints, revalidate the plan, and change it only if it is no longer feasible.

t12 U No new business event is reported for this stage. Refresh the solution against the current public tables.

NLDO-P006 (NLDO; FJSP)

Initial public request. The instance data is provided in attached tables; treat the listed rows and columns as the source of truth. Create a flexible job-shop schedule. Assign every operation to one eligible resource, respect fixed job precedence and resource capacity, and use settings.objective\_mode for the current scalar objective. Later updates may revise soft due/priority values or machine cost\_rate but do not change machine availability, operation eligibility, or processing requirements. Plan-change disruption is reported only as a diagnostic and never defines hard feasibility. Schedule flexible job-shop operations with precedence and machine capacity constraints. Minimize weighted tardiness; report disruption only as an auxiliary diagnostic. J1 due 30: J1O1 can run on M2/M3 for 4 minutes, J1O2 can run on M3/M1 for 6 minutes, J1O3 can run on M1/M2 for 8 minutes J2 due 34: J2O1 can run on M3/M1 for 7 minutes, J2O2 can run on M1/M2 for 9 minutes, J2O3 can run on M2/M3 for 4 minutes J3 due 38: J3O1 can run on M1/M2 for 10 minutes, J3O2 can run on M2/M3 for 5 minutes, J3O3 can run on M3/M1 for 7 minutes

## Public updates.

t01 D Machine M3’s public cost\_rate increases by 3. All machines remain available and operation eligibility is unchanged.

t02 O J3 becomes urgent: set priority to 3 and tighten its soft due time by 5 minutes. This changes tardiness cost but not schedule feasibility.

t03 O,M For the job that became urgent in the previous note, increase its priority by 1 again; infer the job from memory.

t04 O,M The same urgent job receives a 3-minute soft due-time relaxation after customer confirmation; infer the job from memory.

t05 D,M The machine from the earliest cost-rate note receives another 0.5 cost-rate increase; infer it from memory and re-optimize the same feasible schedule space.

t06 U Operations asks for a checkpoint only: keep the current public objective and hard constraints, revalidate the plan, and change it only if it is no longer feasible.

t07 U No new business event is reported for this stage. Refresh the solution against the current public tables.

t08 U Treat this as a routine review. The model, objective terms, and public rows stay as they are; preserve feasibility under the current state.

t09 U The planning team wants the current public contract revalidated without adding new constraints.

t10 U Run a maintenance update with the same public contract. Revalidate feasibility and quality under the current public state.

t11 U Operations asks for a checkpoint only: keep the current public objective and hard constraints, revalidate the plan, and change it only if it is no longer feasible.

t12 U No new business event is reported for this stage. Refresh the solution against the current public tables.

## NLDO-P007 (NLDO; INRC2)

Initial public request. The instance data is provided in attached tables; treat the listed rows and columns as the source of truth. Create a staff roster over days and shifts. Respect current public coverage, absences, nurse max-shift capacities, and one assignment per staff member per day. Later updates may replace these public rows as well as soft preferences and penalty weights. Report roster disruption only as a diagnostic. Create a one-week nurse roster for 30 nurses over 7 days and day/evening/night shifts. Satisfy fixed coverage and one shift per nurse per day, then optimize workload fairness and soft preferences; report assignment-change counts only as a diagnostic.

## Public updates.

t01 O Nurse N004 requests D2 off if possible. This is a soft preference only; coverage requirements and availability stay unchanged.

t02 O Preference satisfaction becomes more important: increase the public preference penalty weight by 10.

t03 O Nurse N009 requests D4 off if possible under the same hard coverage contract.

t04 O,M The nurse from the first off-duty preference also requests D5 off if possible; infer the nurse from memory.

t05 O Activate a temporary soft fairness rule: avoid extra evening shifts for nurses who already worked a night shift.

t06 O Fairness becomes more important after a staff survey: increase the public fairness penalty weight by 8.

t07 O Nurse N013 requests D6 off if possible; hard availability and coverage remain unchanged.

t08 O Weekend preference review: increase sequence-penalty weight by 12 and preference-penalty weight by 5 under the same coverage requirements.

t09 O,M The nurse from the latest off-duty preference should also avoid D7 night duty if possible; record it as a soft D7 preference from memory.

t10 O A late policy reset changes only soft rostering priorities: increase fairness weight by 15, reduce preference weight by 5, and keep the evening-after-night sequence rule active. Rebuild the roster under the same fixed coverage and availability constraints.

t11 F,O A hospital-wide staffing reset starts for the next planning week. Replace the current coverage, active-nurse, absence, preference, nurse max-shift, and policy tables by the public regime-A rows supplied with this update. The active nurse IDs and hard coverage remain valid, but off-duty preferences and their public penalty weight are replaced by the supplied current rows.

t12 O The following planning week switches independently to public regime B. Replace the current coverage, active-nurse, absence, preference, nurse max-shift, and policy tables by the supplied regime-B rows. The same nurse IDs and hard coverage remain valid, while the off-duty preference rows and their public penalty weight are replaced by the supplied current rows.

## NLDO-P008 (NLDO; INRC2)

Initial public request. The instance data is provided in attached tables; treat the listed rows and columns as the source of truth. Create a staff roster over days and shifts. Respect current public coverage, absences, nurse max-shift capacities, and one assignment per staff member per day. Later updates may replace these public rows as well as soft preferences and penalty weights. Report roster disruption only as a diagnostic. Create a one-week nurse roster for 34 nurses over 7 days and day/evening/night shifts. Satisfy fixed coverage and one shift per nurse per day, then optimize workload fairness and soft preferences; report assignment-change counts only as a diagnostic.

## Public updates.

t01 O Nurse N007 requests D2 off if possible. This is a soft preference only; coverage requirements and availability stay unchanged.

t02 O Preference satisfaction becomes more important: increase the public preference penalty weight by 10.

t03 O Nurse N012 requests D4 off if possible under the same hard coverage contract.

t04 O,M The nurse from the first off-duty preference also requests D5 off if possible; infer the nurse from memory.

t05 O Activate a temporary soft fairness rule: avoid extra evening shifts for nurses who already worked a night shift.

t06 O Fairness becomes more important after a staff survey: increase the public fairness penalty weight by 8.

t07 O Nurse N018 requests D6 off if possible; hard availability and coverage remain unchanged.

t08 O Weekend preference review: increase sequence-penalty weight by 12 and preference-penalty weight by 5 under the same coverage requirements.

t09 O,M The nurse from the latest off-duty preference should also avoid D7 night duty if possible; record it as a soft D7 preference from memory.

t10 O Preference pressure softens: reduce the public preference penalty weight by 6 while keeping all hard coverage requirements unchanged.

t11 F,O A hospital-wide staffing reset starts for the next planning week. Replace the current coverage, active-nurse, absence, preference, nurse max-shift, and policy tables by the public regime-A rows supplied with this update. The active nurse IDs and hard coverage remain valid, but off-duty preferences and their public penalty weight are replaced by the supplied current rows.

t12 O The following planning week switches independently to public regime B. Replace the current coverage, active-nurse, absence, preference, nurse max-shift, and policy tables by the supplied regime-B rows. The same nurse IDs and hard coverage remain valid, while the off-duty preference rows and their public penalty weight are replaced by the supplied current rows.

## NLDO-P009 (NLDO; INRC2)

Initial public request. The instance data is provided in attached tables; treat the listed rows and columns as the source of truth. Create a staff roster over days and shifts. Respect current public coverage, absences, nurse max-shift capacities, and one assignment per staff member per day. Later updates may replace these public rows as well as soft preferences and penalty weights. Report roster disruption only as a diagnostic. Create a one-week nurse roster for 38 nurses over 7 days and day/evening/night shifts. Satisfy fixed coverage and one shift per nurse per day, then optimize workload fairness and soft preferences; report assignment-change counts only as a diagnostic.

## Public updates.

t01 O Nurse N010 requests D2 off if possible. This is a soft preference only; coverage requirements and availability stay unchanged.

t02 O Preference satisfaction becomes more important: increase the public preference penalty weight by 10.

t03 O Nurse N015 requests D4 off if possible under the same hard coverage contract.

t04 O,M The nurse from the first off-duty preference also requests D5 off if possible; infer the nurse from memory.

t05 O Activate a temporary soft fairness rule: avoid extra evening shifts for nurses who already worked a night shift.

t06 O Fairness becomes more important after a staff survey: increase the public fairness penalty weight by 8.

t07 O Nurse N023 requests D6 off if possible; hard availability and coverage remain unchanged.

t08 O Weekend preference review: increase sequence-penalty weight by 12 and preference-penalty weight by 5 under the same coverage requirements.

t09 O,M The nurse from the latest off-duty preference should also avoid D7 night duty if possible; record it as a soft D7 preference from memory.

t10 O Preference pressure softens: reduce the public preference penalty weight by 6 while keeping all hard coverage requirements unchanged.

t11 F,O A hospital-wide staffing reset starts for the next planning week. Replace the current coverage, active-nurse, absence, preference, nurse max-shift, and policy tables by the public regime-A rows supplied with this update. The active nurse IDs and hard coverage remain valid, but off-duty preferences and their public penalty weight are replaced by the supplied current rows.

t12 O The following planning week switches independently to public regime B. Replace the current coverage, active-nurse, absence, preference, nurse max-shift, and policy tables by the supplied regime-B rows. The same nurse IDs and hard coverage remain valid, while the off-duty preference rows and their public penalty weight are replaced by the supplied current rows.

NLDO-P010 (NLDO; GREEN\_VRP\_MULTIOBJECTIVE)

Initial public request. Plan same-day delivery routes from a single depot for a small urban fleet. Initial depot, vehicle, and order data are provided in public CSV tables. Use the attached depot, vehicle, order, and policy tables as the source of truth. All active orders should be served once under vehicle capacity. Rows with active=false are public candidate orders that are not served until a later natural-language update explicitly activates them. Later updates may replace public order geometry, demand, time windows, priorities, vehicle emissions, depot coordinates, or carbon policy; all current rows remain public. Optimize three main criteria: travel distance, delivery lateness, and emissions using public policy parameters when present. Also report route disruption and priority-customer penalties as auxiliary diagnostics

## Public updates.

t01 O Existing order O006 becomes a same-day priority delivery: increase its public priority by 2 and tighten its soft due target by 18 minutes. The order set, vehicle set, capacities, and active flags stay unchanged.

t02 F,O Traffic around order O004 slows vehicles by 25 percent; remember this customer as the congestion anchor.

t03 O Vehicle V1’s emission\_rate increases by 0.18 after a telemetry update; it remains available with the same capacity.

t04 R Carbon policy is tightened: set the public policy carbon\_multiplier to 1.35 so emissions count 35 percent more heavily in dispatch evaluation.

t05 O,M The customer mentioned in the congestion note now needs a tighter delivery window ending 20 minutes earlier; infer the order from memory.

t06 O An express-service wave temporarily reverses dispatch priorities. Set policy.carbon\_multiplier to 0.75 and tighten the soft due-time targets for O006, O009, and O012 by 45, 50, and 35 minutes respectively. Increase each of those existing orders’ public priority by 1. These due-time targets affect lateness and priority-service quality rather than hard feasibility, so route quality should be reconsidered instead of blindly preserving the previous carbon-oriented plan.

t07 O Existing pharmacy-like order O010 gets a tighter soft due target by 22 minutes and priority increases by 2. The order table remains fixed.

t08 O A carbon-rationing notice takes effect after the express wave. Set policy.carbon\_multiplier to 4.80, increase every vehicle emission\_rate by 0.35, and reduce priorities for O006, O009, and O012 by 1. This is a regime change toward carbon-efficient routing over the same active orders and vehicles, so routes specialized for the previous express wave can be misleading.

t09 O,M The vehicle from the telemetry update receives a 0.10 emission\_rate correction downward; use the vehicle identity from memory.

t10 O A late emergency service wave changes priorities over the same order set. Set policy.carbon\_multiplier to 0.45, subtract 0.20 from every vehicle emission\_rate, and subtract another 0.08 from cleaner vehicle V2. Tighten the soft due-time targets for O001, O002, and O003 by 40, 45, and 50 minutes, and increase their priorities by 2. These due-time targets affect lateness and priority-service quality, not hard feasibility. Use the current public tables as the source of truth.

t11 F,O A network-wide dispatch reset activates public regime A. Replace the active workload by the prelisted cohort A, replace the depot coordinates and every active order’s x, y, demand, ready, due, service, and priority fields by the supplied current rows; also replace every vehicle capacity, shift\_end, availability, emission\_rate, and policy.carbon\_multiplier by the supplied values. Serve every public order row in the active cohort exactly once. The route encoding type and the three objective definitions remain unchanged.

t12 F,O A network-wide dispatch reset activates public regime B. Replace the active workload by the prelisted cohort B, replace the depot coordinates and every active order’s x, y, demand, ready, due, service, and priority fields by the supplied current rows; also replace every vehicle capacity, shift\_end, availability, emission\_rate, and policy.carbon\_multiplier by the supplied values. Serve every public order row in the active cohort exactly once. The route encoding type and the three objective definitions remain unchanged.

## NLDO-P011 (NLDO; GREEN\_VRP\_MULTIOBJECTIVE)

Initial public request. Plan same-day delivery routes from a single depot for a small urban fleet. Initial depot, vehicle, and order data are provided in public CSV tables. Use the attached depot, vehicle, order, and policy tables as the source of truth. All active orders should be served once under vehicle capacity. Rows with active=false are public candidate orders that are not served until a later natural-language update explicitly activates them. Later updates may replace public order geometry, demand, time windows, priorities, vehicle emissions, depot coordinates, or carbon policy; all current rows remain public. Optimize three main criteria: travel distance, delivery lateness, and emissions using public policy parameters when present. Also report route disruption and priority-customer penalties as auxiliary diagnostics.

## Public updates.

t01 O Existing order O007 becomes a same-day priority delivery: increase its public priority by 2 and tighten its soft due target by 18 minutes. The order set, vehicle set, capacities, and active flags stay unchanged.

t02 F,O Traffic around order O005 slows vehicles by 25 percent; remember this customer as the congestion anchor.

t03 O Vehicle V2’s emission\_rate increases by 0.18 after a telemetry update; it remains available with the same capacity.

t04 R Carbon policy is tightened: set the public policy carbon\_multiplier to 1.35 so emissions count 35 percent more heavily in dispatch evaluation.

t05 O,M The customer mentioned in the congestion note now needs a tighter delivery window ending 20 minutes earlier; infer the order from memory.

t06 O An express-service wave temporarily reverses dispatch priorities. Set policy.carbon\_multiplier to 0.75 and tighten the soft due-time targets for O007, O010, and O013 by 45, 50, and 35 minutes respectively. Increase each of those existing orders’ public priority by 1. These due-time targets affect lateness and priority-service quality rather than hard feasibility, so route quality should be reconsidered instead of blindly preserving the previous carbon-oriented plan

t07 O Existing pharmacy-like order O011 gets a tighter soft due target by 22 minutes and priority increases by 2. The order table remains fixed.

t08 O A carbon-rationing notice takes effect after the express wave. Set policy.carbon\_multiplier to 4.80, increase every vehicle emission\_rate by 0.35, and reduce priorities for O007, O010, and O013 by 1. This is a regime change toward carbon-efficient routing over the same active orders and vehicles, so routes specialized for the previous express wave can be misleading.

t09 O,M The vehicle from the telemetry update receives a 0.10 emission\_rate correction downward; use the vehicle identity from memory.

t10 O A late emergency service wave changes priorities over the same order set. Set policy.carbon\_multiplier to 0.45, subtract 0.20 from every vehicle emission\_rate, and subtract another 0.08 from cleaner vehicle V3. Tighten the soft due-time targets for O002, O003, and O004 by 40, 45, and 50 minutes, and increase their priorities by 2. These due-time targets affect lateness and priority-service quality, not hard feasibility. Use the current public tables as the source of truth.

t11 F,O A network-wide dispatch reset activates public regime A. Replace the active workload by the prelisted cohort A, replace the depot coordinates and every active order’s x, y, demand, ready, due, service, and priority fields by the supplied current rows; also replace every vehicle capacity, shift\_end, availability, emission\_rate, and policy.carbon\_multiplier by the supplied values. Serve every public order row in the active cohort exactly once. The route encoding type and the three objective definitions remain unchanged.

t12 F,O A network-wide dispatch reset activates public regime B. Replace the active workload by the prelisted cohort B, replace the depot coordinates and every active order’s x, y, demand, ready, due, service, and priority fields by the supplied current rows; also replace every vehicle capacity, shift\_end, availability, emission\_rate, and policy.carbon\_multiplier by the supplied values. Serve every public order row in the active cohort exactly once. The route encoding type and the three objective definitions remain unchanged.

## NLDO-P012 (NLDO; GREEN\_VRP\_MULTIOBJECTIVE)

Initial public request. Plan same-day delivery routes from a single depot for a small urban fleet. Initial depot, vehicle, and order data are provided in public CSV tables. Use the attached depot, vehicle, order, and policy tables as the source of truth. All active orders should be served once under vehicle capacity. Row with active=false are public candidate orders that are not served until a later natural-language update explicitly activates them. Later updates may replace public order geometry, demand, time windows, priorities, vehicle emissions, depot coordinates, or carbon policy; all current rows remain public. Optimize three main criteria: travel distance, delivery lateness, and emissions using public policy parameters when present. Also report route disruption and priority-customer penalties as auxiliary diagnostics.

## Public updates.

t01 O Existing order O008 becomes a same-day priority delivery: increase its public priority by 2 and tighten its soft due target by 18 minutes. The order set, vehicle set, capacities, and active flags stay unchanged.

t02 F,O Traffic around order O006 slows vehicles by 25 percent; remember this customer as the congestion anchor.

t03 O Vehicle V3’s emission\_rate increases by 0.18 after a telemetry update; it remains available with the same capacity.

t04 R Carbon policy is tightened: set the public policy carbon\_multiplier to 1.35 so emissions count 35 percent more heavily in dispatch evaluation.

t05 O,M The customer mentioned in the congestion note now needs a tighter delivery window ending 20 minutes earlier; infer the order from memory.

t06 O An express-service wave temporarily reverses dispatch priorities. Set policy.carbon\_multiplier to 0.75 and tighten the soft due-time targets for O008, O011, and O014 by 45, 50, and 35 minutes respectively. Increase each of those existing orders’ public priority by 1. These due-time targets affect lateness and priority-service quality rather than hard feasibility, so route quality should be reconsidered instead of blindly preserving the previous carbon-oriented plan.

t07 O Existing pharmacy-like order O012 gets a tighter soft due target by 22 minutes and priority increases by 2. The order table remains fixed.

t08 O A carbon-rationing notice takes effect after the express wave. Set policy.carbon\_multiplier to 4.80, increase every vehicle emission\_rate by 0.35, and reduce priorities for O008, O011, and O014 by 1. This is a regime change toward carbon-efficient routing over the same active orders and vehicles, so routes specialized for the previous express wave can be misleading.

t09 O,M The vehicle from the telemetry update receives a 0.10 emission\_rate correction downward; use the vehicle identity from memory.

t10 O A late emergency service wave changes priorities over the same order set. Set policy.carbon\_multiplier to 0.45, subtract 0.20 from every vehicle emission\_rate, and subtract another 0.08 from cleaner vehicle V2. Tighten the soft due-time targets for O003, O004, and O005 by 40, 45, and 50 minutes, and increase their priorities by 2. These due-time targets affect lateness and priority-service quality, not hard feasibility. Use the current public tables as the source of truth.

t11 F,O A network-wide dispatch reset activates public regime A. Replace the active workload by the prelisted cohort A, replace the depot coordinates and every active order’s x, y, demand, ready, due, service, and priority fields by the supplied current rows; also replace every vehicle capacity, shift\_end, availability, emission\_rate, and policy.carbon\_multiplier by the supplied values. Serve every public order row in the active cohort exactly once. The route encoding type and the three objective definitions remain unchanged.

t12 F,O A network-wide dispatch reset activates public regime B. Replace the active workload by the prelisted cohort B, replace the depot coordinates and every active order’s x, y, demand, ready, due, service, and priority fields by the supplied current rows; also replace every vehicle capacity, shift\_end, availability, emission\_rate, and policy.carbon\_multiplier by the supplied values. Serve every public order row in the active cohort exactly once. The route encoding type and the three objective definitions remain unchanged.

## NLDO-P013 (NLDO; CLOUD\_SCHEDULING\_MULTIOBJECTIVE)

Initial public request. The instance data is provided in attached tables; treat the listed rows and columns as the source of truth. Assign active tasks to available compute resources. Rows with active=false are public candidate jobs that are not scheduled until a later update explicitly activates them. Respect current public CPU, memory, accelerator needs, and machine availability. Optimize energy and imbalance as main objectives and report SLA/latency/migration diagnostics without treating those diagnostics as hard feasibility constraints. Later updates may replace public workload requirements, machine capacity/accelerator profiles, availability, soft SLA, priority, energy, or carbon parameters. Place active cloud jobs onto available machines for the next scheduling window. Machines: M01 cpu 38 mem 80 gpu false, M02 cpu 42 mem 88 gpu true, M03 cpu 46 mem 96 gpu false, M04 cpu 34 mem 72 gpu false, M05 cpu 38 mem 80 gpu true, M06 cpu 42 mem 88 gpu false, M07 cpu 46 mem 96 gpu false, M08 cpu 34 mem 72 gpu true. Jobs: J001 cpu 7 mem 11 deadline 81 priority 2, J002 cpu 3 mem 18 deadline 92 priority 3, J003 cpu 8 mem 7 deadline 103 priority 4, J004 cpu 4 mem 14 deadline 114 priority 1, J005 cpu 9 mem 21 deadline 125 priority 2, J006 cpu 5 mem 10 deadline 136 priority 3, J007 cpu 10 mem 17 deadline 147 priority 4, J008 cpu 6 mem 6 deadline 158 priority 1, J009 cpu 2 mem 13 deadline 169 priority 2, J010 cpu 7 mem 20 deadline 180 priority 3, J011 cpu 3 mem 9 deadline 71 priority 4, J012 cpu 8 mem 16 deadline 82 priority 1, J013 cpu 4 mem 5 deadline 93 priority 2, J014 cpu 9 mem 12 deadline 104 priority 3, J015 cpu 5 mem 19 deadline 115 priority 4, J016 cpu 10 mem 8 deadline 126 priority 1, J017 cpu 6 mem 15 deadline 137 priority 2, J018 cpu 2 mem 4 deadline 148 priority 3, J019 cpu 7 mem 11 deadline 159 priority 4, J020 cpu 3 mem 18 deadline 170 priority 1, J021 cpu 8 mem 7 deadline 181 priority 2, J022 cpu 4 mem 14 deadline 72 priority 3, J023 cpu 9 mem 21 deadline 83 priority 4, J024 cpu 5 mem 10 deadline 94 priority 1, B20\_01 cpu 3 mem 8 deadline 69 priority 4, B20\_02 cpu 4 mem 11 deadline 73 priority 4, B20\_03 cpu 5 mem 14 deadline 77 priority 4, B20\_04 cpu 2 mem 7 deadline 81 priority 4, B20\_05 cpu 3 mem 10 deadline 85 priority 4, B20\_06 cpu 4 mem 13 deadline 89 priority 4, L001 cpu 8 mem 16 deadline 91 priority 2, L002 cpu 13 mem 25 deadline 102 priority 3, L003 cpu 6 mem 34 deadline 113 priority 4, L004 cpu 11 mem 15 deadline 124 priority 1, L005 cpu 4 mem 24 deadline 135 priority 2, L006 cpu 9 mem 33 deadline 146 priority 3, L007 cpu 14 mem 14 deadline 157 priority 4, L008 cpu 7 mem 23 deadline 168 priority 1, L009 cpu 12 mem 32 deadline 179 priority 2, L010 cpu 5 mem 13 deadline 190 priority 3, L011 cpu 10 mem 22 deadline 201 priority 4, L012 cpu 3 mem 31 deadline 212 priority 1, L013 cpu 8 mem 12 deadline 223 priority 2, L014 cpu 13 mem 21 deadline 234 priority 3, L015 cpu 6 mem 30 deadline 85 priority 4, L016 cpu 11 mem 11 deadline 96 priority 1, L017 cpu 4 mem 20 deadline 107 priority 2, L018 cpu 9 mem 29 deadline 118 priority 3, L019 cpu 14 mem 10 deadline 129 priority 4, L020 cpu 7 mem 19 deadline 140 priority 1, L021 cpu 12 mem 28 deadline 151 priority 2, L022 cpu 5 mem 9 deadline 162 priority 3, L023 cpu 10 mem 18 deadline 173 priority 4, L024 cpu 3 mem 27 deadline 184 priority 1, L025 cpu 8 mem 8 deadline 195 priority 2, L026 cpu 13 mem 17 deadline 206 priority 3, L027 cpu 6 mem 26 deadline 217 priority 4, L028 cpu 11 mem 7 deadline 228 priority 1, L029 cpu 4 mem 16 deadline 239 priority 2, L030 cpu 9 mem 25 deadline 90 priority 3, L031 cpu 14 mem 34 deadline 101 priority 4, L032 cpu 7 mem 15 deadline 112 priority 1, L033 cpu 12 mem 24 deadline 123 priority 2, L034 cpu 5 mem 33 deadline 134 priority 3, L035 cpu 10 mem 14 deadline 145 priority 4, L036 cpu 3 mem 23 deadline 156 priority 1, L037 cpu 8 mem 32 deadline 167 priority 2, L038 cpu 13 mem 13 deadline 178 priority 3, L039 cpu 6 mem 22 deadline 189 priority 4, L040 cpu 11 mem 31 deadline 200 priority 1. Jobs with active=false are public candidate jobs that are not scheduled until a later natural-language update explicitly activates them. Respect CPU, memory, GPU needs, and machine availability. Optimize two main criteria: energy use and load imbalance. Also report SLA violations, latency, and migrations as diagnostics; these diagnostics do not define feasibility.

## Public updates.

t01 D A latency-sensitive workload group burst-0 is identified among existing jobs J001, J002, J003, J004, J005. Increase their public priority by 1 and remember this group name for later updates. The job table remains fixed.

t02 O Machine M02’s energy\_per\_cpu increases by 0.055 because of thermal throttling, but its CPU and memory capacity stay unchanged.

t03 D Soft SLA target for job J005 is tightened by 35 minutes after an escalation. This affects latency penalty, not hard feasibility.

t04 R Energy price doubles for the next control window, so energy cost becomes much more important.

t05 D,M The workload burst mentioned at the beginning now gets premium priority 5; infer the affected jobs from memory.

t06 O A carbon-aware operating window starts: machine M08’s energy\_idle increases by 1.6, carbon\_intensity rises to 1.80, and energy\_price is set to 1.25. This changes the energy-versus-balance trade-off without changing the machine set or capacities.

t07 D Existing analytics-like job J018 becomes less urgent: reduce its priority by 1 and extend its soft SLA target by 25 minutes. The active job set remains fixed.

t08 R A balancing directive follows the scarcity window: energy\_price drops to 0.70 while carbon\_intensity remains elevated at 1.35. The plan should now expose better load balance instead of simply minimizing energy.

t09 R Operations relaxes the analytics job J018 by extending its soft SLA target by 30 minutes, and carbon\_intensity is set to 1.10.

t10 O A late reset changes energy and service priorities over the same jobs. Machine M02’s energy\_per\_cpu is reduced by 0.04; machine M05’s energy\_idle increases by 2.2; energy\_price is set to 0.60 and carbon\_intensity to 0.85. Existing jobs J010, J011, J012, J013, J014 receive priority +1 and soft SLA targets tightened by 20 minutes. The objective trade-off has moved, but current feasibility is still judged over the same fixed public resource and job set.

t11 F,O A cluster-wide hardware and workload reset activates public regime A. Activate the prelisted latewindow jobs, replace every machine’s CPU, memory, GPU, availability, idle-energy, and per-CPU

energy fields and replace every active job’s CPU, memory, GPU requirement, deadline, latency sensitivity, and priority by the supplied rows. Job and machine IDs remain unchanged, and the placement encoding and two objective definitions remain unchanged. Also replace energy\_price and carbon\_intensity by the supplied current values.

A cluster-wide hardware and workload reset activates public regime B. Activate the prelisted late window jobs, replace every machine’s CPU, memory, GPU, availability, idle-energy, and per-CPU energy fields and replace every active job’s CPU, memory, GPU requirement, deadline, latency sensitivity, and priority by the supplied rows. Job and machine IDs remain unchanged, and the placement encoding and two objective definitions remain unchanged. Also replace energy\_price and carbon\_intensity by the supplied current values.

NLDO-P014 (NLDO; CLOUD\_SCHEDULING\_MULTIOBJECTIVE)

Initial public request. The instance data is provided in attached tables; treat the listed rows and columns as the source of truth. Assign active tasks to available compute resources. Rows with active=false are public candidate jobs that are not scheduled until a later update explicitly activates them. Respect current public CPU, memory, accelerator needs, and machine availability. Optimize energy and imbalance as main objectives and report SLA/latency/migration diagnostics without treating those diagnostics as hard feasibility constraints. Later updates may replace public workload requirements, machine capacity/accelerator profiles, availability, soft SLA, priority, energy, or carbon parameters. Place active cloud jobs onto available machines for the next scheduling window. Machines: M01 cpu 42 mem 96 gpu false, M02 cpu 46 mem 72 gpu true, M03 cpu 34 mem 80 gpu false, M04 cpu 38 mem 88 gpu false, M05 cpu 42 mem 96 gpu true, M06 cpu 46 mem 72 gpu false, M07 cpu 34 mem 80 gpu false, M08 cpu 38 mem 88 gpu true. Jobs: J001 cpu 8 mem 13 deadline 84 priority 3, J002 cpu 4 mem 20 deadline 95 priority 4, J003 cpu 9 mem 9 deadline 106 priority 1, J004 cpu 5 mem 16 deadline 117 priority 2, J005 cpu 10 mem 5 deadline 128 priority 3, J006 cpu 6 mem 12 deadline 139 priority 4, J007 cpu 2 mem 19 deadline 150 priority 1, J008 cpu 7 mem 8 deadline 161 priority 2, J009 cpu 3 mem 15 deadline 172 priority 3, J010 cpu 8 mem 4 deadline 183 priority 4, J011 cpu 4 mem 11 deadline 74 priority 1, J012 cpu 9 mem 18 deadline 85 priority 2, J013 cpu 5 mem 7 deadline 96 priority 3, J014 cpu 10 mem 14 deadline 107 priority 4, J015 cpu 6 mem 21 deadline 118 priority 1, J016 cpu 2 mem 10 deadline 129 priority 2, J017 cpu 7 mem 17 deadline 140 priority 3, J018 cpu 3 mem 6 deadline 151 priority 4, J019 cpu 8 mem 13 deadline 162 priority 1, J020 cpu 4 mem 20 deadline 173 priority 2, J021 cpu 9 mem 9 deadline 184 priority 3, J022 cpu 5 mem 16 deadline 75 priority 4, J023 cpu 10 mem 5 deadline 86 priority 1, J024 cpu 6 mem 12 deadline 97 priority 2, B21\_01 cpu 4 mem 10 deadline 70 priority 4, B21\_02 cpu 5 mem 13 deadline 74 priority 4, B21\_03 cpu 2 mem 6 deadline 78 priority 4, B21\_04 cpu 3 mem 9 deadline 82 priority 4, B21\_05 cpu 4 mem 12 deadline 86 priority 4, B21\_06 cpu 5 mem 5 deadline 90 priority 4, L001 cpu 9 mem 19 deadline 92 priority 3, L002 cpu 14 mem 28 deadline 103 priority 4, L003 cpu 7 mem 9 deadline 114 priority 1, L004 cpu 12 mem 18 deadline 125 priority 2, L005 cpu 5 mem 27 deadline 136 priority 3, L006 cpu 10 mem 8 deadline 147 priority 4, L007 cpu 3 mem 17 deadline 158 priority 1, L008 cpu 8 mem 26 deadline 169 priority 2, L009 cpu 13 mem 7 deadline 180 priority 3, L010 cpu 6 mem 16 deadline 191 priority 4, L011 cpu 11 mem 25 deadline 202 priority 1, L012 cpu 4 mem 34 deadline 213 priority 2, L013 cpu 9 mem 15 deadline 224 priority 3, L014 cpu 14 mem 24 deadline 235 priority 4, L015 cpu 7 mem 33 deadline 86 priority 1, L016 cpu 12 mem 14 deadline 97 priority 2, L017 cpu 5 mem 23 deadline 108 priority 3, L018 cpu 10 mem 32 deadline 119 priority 4, L019 cpu 3 mem 13 deadline 130 priority 1, L020 cpu 8 mem 22 deadline 141 priority 2, L021 cpu 13 mem 31 deadline 152 priority 3, L022 cpu 6 mem 12 deadline 163 priority 4, L023 cpu 11 mem 21 deadline 174 priority 1, L024 cpu 4 mem 30 deadline 185 priority 2, L025 cpu 9 mem 11 deadline 196 priority 3, L026 cpu 14 mem 20 deadline 207 priority 4, L027 cpu 7 mem 29 deadline 218 priority 1, L028 cpu 12 mem 10 deadline 229 priority 2, L029 cpu 5 mem 19 deadline 80 priority 3, L030 cpu 10 mem 28 deadline 91 priority 4, L031 cpu 3 mem 9 deadline 102 priority 1, L032 cpu 8 mem 18 deadline 113 priority 2, L033 cpu 13 mem 27 deadline 124 priority 3, L034 cpu 6 mem 8 deadline 135 priority 4, L035 cpu 11 mem 17 deadline 146 priority 1, L036 cpu 4 mem 26 deadline 157 priority 2, L037 cpu 9 mem 7 deadline 168 priority 3, L038 cpu 14 mem 16 deadline 179 priority 4, L039 cpu 7 mem 25 deadline 190 priority 1, L040 cpu 12 mem 34 deadline 201 priority 2. Jobs with active=false are public candidate jobs that are not scheduled until a later natural-language update explicitly activates them. Respect CPU, memory, GPU needs, and machine availability. Optimize two main criteria: energy use and load imbalance. Also report SLA violations, latency, and migrations as diagnostics; these diagnostics do not define feasibility. Public updates.

A latency-sensitive workload group burst-1 is identified among existing jobs J002, J003, J004, J005, J006. Increase their public priority by 1 and remember this group name for later updates. The job table remains fixed.

t02 O Machine M03’s energy\_per\_cpu increases by 0.055 because of thermal throttling, but its CPU and memory capacity stay unchanged.

t03 D Soft SLA target for job J006 is tightened by 35 minutes after an escalation. This affects latency penalty, not hard feasibility.

t04 R Energy price doubles for the next control window, so energy cost becomes much more important. t05 D,M The workload burst mentioned at the beginning now gets premium priority 5; infer the affected jobs from memory.

t06 O A carbon-aware operating window starts: machine M07’s energy\_idle increases by 1.6, carbon\_intensity rises to 1.80, and energy\_price is set to 1.25. This changes the energy-versus-balance trade-off without changing the machine set or capacities.

t07 D Existing analytics-like job J019 becomes less urgent: reduce its priority by 1 and extend its soft SLA target by 25 minutes. The active job set remains fixed.

t08 R A balancing directive follows the scarcity window: energy\_price drops to 0.70 while carbon\_intensity remains elevated at 1.35. The plan should now expose better load balance instead of simply minimizing energy.

t09 R Operations relaxes the analytics job J019 by extending its soft SLA target by 30 minutes, and carbon\_intensity is set to 1.10.

t10 O A late reset changes energy and service priorities over the same jobs. Machine M03’s energy\_per\_cpu is reduced by 0.04; machine M06’s energy\_idle increases by 2.2; energy\_price is set to 0.60 and carbon\_intensity to 0.85. Existing jobs J011, J012, J013, J014, J015 receive priority +1 and soft SLA targets tightened by 20 minutes. The objective trade-off has moved, but current feasibility is still judged over the same fixed public resource and job set.

t11 F,O A cluster-wide hardware and workload reset activates public regime A. Activate the prelisted latewindow jobs, replace every machine’s CPU, memory, GPU, availability, idle-energy, and per-CPU energy fields and replace every active job’s CPU, memory, GPU requirement, deadline, latency sensitivity, and priority by the supplied rows. Job and machine IDs remain unchanged, and the placement encoding and two objective definitions remain unchanged. Also replace energy\_price and carbon\_intensity by the supplied current values.

t12 F,O A cluster-wide hardware and workload reset activates public regime B. Activate the prelisted latewindow jobs, replace every machine’s CPU, memory, GPU, availability, idle-energy, and per-CPU energy fields and replace every active job’s CPU, memory, GPU requirement, deadline, latency sensitivity, and priority by the supplied rows. Job and machine IDs remain unchanged, and the placement encoding and two objective definitions remain unchanged. Also replace energy\_price and carbon\_intensity by the supplied current values

## NLDO-P015 (NLDO; CLOUD\_SCHEDULING\_MULTIOBJECTIVE)

Initial public request. The instance data is provided in attached tables; treat the listed rows and columns as the source of truth. Assign active tasks to available compute resources. Rows with active=false are public candidate jobs that are not scheduled until a later update explicitly activates them. Respect current public CPU, memory, accelerator needs, and machine availability. Optimize energy and imbalance as main objectives and report SLA/latency/migration diagnostics without treating those diagnostics as hard feasibility constraints. Later updates may replace public workload requirements, machine capacity/accelerator profiles, availability, soft SLA, priority, energy, or carbon parameters. Place active cloud jobs onto available machines for the next scheduling window. Machines: M01 cpu 46 mem 80 gpu false, M02 cpu 34 mem 88 gpu true, M03 cpu 38 mem 96 gpu false, M04 cpu 42 mem 72 gpu false, M05 cpu 46 mem 80 gpu true, M06 cpu 34 mem 88 gpu false, M07 cpu 38 mem 96 gpu false, M08 cpu 42 mem 72 gpu true. Jobs: J001 cpu 9 mem 15 deadline 87 priority 4, J002 cpu 5 mem 4 deadline 98 priority 1, J003 cpu 10 mem 11 deadline 109 priority 2, J004 cpu 6 mem 18 deadline 120 priority 3, J005 cpu 2 mem 7 deadline 131 priority 4, J006 cpu 7 mem 14 deadline 142 priority 1, J007 cpu 3 mem 21 deadline 153 priority 2, J008 cpu 8 mem 10 deadline 164 priority 3, J009 cpu 4 mem 17 deadline 175 priority 4, J010 cpu 9 mem 6 deadline 186 priority 1, J011 cpu 5 mem 13 deadline 77 priority 2, J012 cpu 10 mem 20 deadline 88 priority 3, J013 cpu 6 mem 9 deadline 99 priority 4, J014 cpu 2 mem 16 deadline 110 priority 1, J015 cpu 7 mem 5 deadline 121 priority 2, J016 cpu 3 mem 12 deadline 132 priority 3, J017 cpu 8 mem 19 deadline 143 priority 4, J018 cpu 4 mem 8 deadline 154 priority 1, J019 cpu 9 mem 15 deadline 165 priority 2, J020 cpu 5 mem 4 deadline 176 priority 3, J021 cpu 10 mem 11 deadline 187 priority 4, J022 cpu 6 mem 18 deadline 78 priority 1, J023 cpu 2 mem 7 deadline 89 priority 2, J024 cpu 7 mem 14 deadline 100 priority 3, B22\_01 cpu 5 mem 12 deadline 71 priority 4, B22\_02 cpu 2 mem 5 deadline 75 priority 4, B22\_03 cpu 3 mem 8 deadline 79 priority 4, B22\_04 cpu 4 mem 11 deadline 83 priority 4, B22\_05 cpu 5 mem 14 deadline 87 priority 4, B22\_06 cpu 2 mem 7 deadline 91 priority 4, L001 cpu 10 mem 22 deadline 93 priority 4,

L002 cpu 3 mem 31 deadline 104 priority 1, L003 cpu 8 mem 12 deadline 115 priority 2, L004 cpu 13 mem 21 deadline 126 priority 3, L005 cpu 6 mem 30 deadline 137 priority 4, L006 cpu 11 mem 11 deadline 148 priority 1, L007 cpu 4 mem 20 deadline 159 priority 2, L008 cpu 9 mem 29 deadline 170 priority 3, L009 cpu 14 mem 10 deadline 181 priority 4, L010 cpu 7 mem 19 deadline 192 priority 1, L011 cpu 12 mem 28 deadline 203 priority 2, L012 cpu 5 mem 9 deadline 214 priority 3, L013 cpu 10 mem 18 deadline 225 priority 4, L014 cpu 3 mem 27 deadline 236 priority 1, L015 cpu 8 mem 8 deadline 87 priority 2, L016 cpu 13 mem 17 deadline 98 priority 3, L017 cpu 6 mem 26 deadline 109 priority 4, L018 cpu 11 mem 7 deadline 120 priority 1, L019 cpu 4 mem 16 deadline 131 priority 2, L020 cpu 9 mem 25 deadline 142 priority 3, L021 cpu 14 mem 34 deadline 153 priority 4, L022 cpu 7 mem 15 deadline 164 priority 1, L023 cpu 12 mem 24 deadline 175 priority 2, L024 cpu 5 mem 33 deadline 186 priority 3, L025 cpu 10 mem 14 deadline 197 priority 4, L026 cpu 3 mem 23 deadline 208 priority 1, L027 cpu 8 mem 32 deadline 219 priority 2, L028 cpu 13 mem 13 deadline 230 priority 3, L029 cpu 6 mem 22 deadline 81 priority 4, L030 cpu 11 mem 31 deadline 92 priority 1, L031 cpu 4 mem 12 deadline 103 priority 2, L032 cpu 9 mem 21 deadline 114 priority 3, L033 cpu 14 mem 30 deadline 125 priority 4, L034 cpu 7 mem 11 deadline 136 priority 1, L035 cpu 12 mem 20 deadline 147 priority 2, L036 cpu 5 mem 29 deadline 158 priority 3, L037 cpu 10 mem 10 deadline 169 priority 4, L038 cpu 3 mem 19 deadline 180 priority 1, L039 cpu 8 mem 28 deadline 191 priority 2, L040 cpu 13 mem 9 deadline 202 priority 3. Jobs with active=false are public candidate jobs that are not scheduled until a later natural-language update explicitly activates them. Respect CPU, memory, GPU needs, and machine availability. Optimize two main criteria: energy use and load imbalance. Also report SLA violations, latency, and migrations as diagnostics; these diagnostics do not define feasibility. Public updates.

t01 D A latency-sensitive workload group burst-2 is identified among existing jobs J003, J004, J005, J006, J007. Increase their public priority by 1 and remember this group name for later updates. The job table remains fixed.

t02 O Machine M04’s energy\_per\_cpu increases by 0.055 because of thermal throttling, but its CPU and memory capacity stay unchanged.

t03 D Soft SLA target for job J007 is tightened by 35 minutes after an escalation. This affects latency penalty, not hard feasibility.

t04 R Energy price doubles for the next control window, so energy cost becomes much more important.

t05 D,M The workload burst mentioned at the beginning now gets premium priority 5; infer the affected jobs from memory.

t06 O A carbon-aware operating window starts: machine M06’s energy\_idle increases by 1.6, carbon\_intensity rises to 1.80, and energy\_price is set to 1.25. This changes the energy-versus-balance trade-off without changing the machine set or capacities.

t07 D Existing analytics-like job J020 becomes less urgent: reduce its priority by 1 and extend its soft SLA target by 25 minutes. The active job set remains fixed.

t08 R A balancing directive follows the scarcity window: energy\_price drops to 0.70 while carbon\_intensity remains elevated at 1.35. The plan should now expose better load balance instead of simply minimizing energy.

t09 R Operations relaxes the analytics job J020 by extending its soft SLA target by 30 minutes, and carbon\_intensity is set to 1.10.

t10 O A late reset changes energy and service priorities over the same jobs. Machine M04’s energy\_per\_cpu is reduced by 0.04; machine M07’s energy\_idle increases by 2.2; energy\_price is set to 0.60 and carbon\_intensity to 0.85. Existing jobs J012, J013, J014, J015, J016 receive priority +1 and soft SLA targets tightened by 20 minutes. The objective trade-off has moved, but current feasibility is still judged over the same fixed public resource and job set.

t11 F,O A cluster-wide hardware and workload reset activates public regime A. Activate the prelisted latewindow jobs, replace every machine’s CPU, memory, GPU, availability, idle-energy, and per-CPU energy fields and replace every active job’s CPU, memory, GPU requirement, deadline, latency sensitivity, and priority by the supplied rows. Job and machine IDs remain unchanged, and the placement encoding and two objective definitions remain unchanged. Also replace energy\_price and carbon\_intensity by the supplied current values.

t12 F,O A cluster-wide hardware and workload reset activates public regime B. Activate the prelisted latewindow jobs, replace every machine’s CPU, memory, GPU, availability, idle-energy, and per-CPU energy fields and replace every active job’s CPU, memory, GPU requirement, deadline, latency sensitivity, and priority by the supplied rows. Job and machine IDs remain unchanged, and the placement encoding and two objective definitions remain unchanged. Also replace energy\_price and carbon\_intensity by the supplied current values.