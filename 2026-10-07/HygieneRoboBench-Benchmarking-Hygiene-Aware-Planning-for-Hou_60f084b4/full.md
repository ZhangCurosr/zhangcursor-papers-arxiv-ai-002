# HygieneRoboBench: Benchmarking Hygiene-Aware Planning for Household Robots

Yurun Chen<sup>1,2</sup>, Josh Qixuan Sun<sup>3</sup>, Jason Qin<sup>4</sup>, Chengtai Li<sup>2,5,7</sup> Tianyi Wang<sup>2,6</sup>, Mark Crowley<sup>3</sup>, Wentao Zhu<sup>2,7</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Eastern Institute of Technology, Ningbo <sup>3</sup>University of Waterloo <sup>4</sup>Stony Brook University <sup>5</sup>University of Science and Technology of China   
<sup>6</sup>Chengdu Institute of Computer Applications, Chinese Academy of Sciences <sup>7</sup>Ningbo Institute of Digital Twin

“Please finish placing the fruit.”

![](images/356ca5d28def96a7dcfd9dab990501575aabc3150960349763a1bae792f3c7f3.jpg)  
Fig. 1. Contact history, unexpected events, and user priorities shape the continuation. (a) Contact with a contaminated tray transfers contamination through the left gripper to a spoon handle. Washing both grippers leaves the spoon contaminated. (b) Unexpected contact with the spoon contaminates the clean right gripper. (c) The robot replans according to user priorities, choosing to wash or replace the right contact pad before resuming fruit arrangement. OmniGibson [1] scenes are staged illustrations of the task-defined hygiene conditions.

Abstract— Contact with contaminated objects can spread hazards through a household robot’s grippers, tools, and shared surfaces, while new contacts can make an existing plan unsafe. Existing benchmarks do not jointly assess how planners identify hygiene risks from contact history and plan safe continuations after new contact events. Planners must do so within time and resource limits while respecting user priorities. We introduce HygieneRoboBench, with 624 instances across 134 task families, to evaluate safe resolution of household tasks from a given execution history. Tasks capture contamination through two grippers and shared objects, treatment costs, and user priorities. We combine controlled history, profile, and event comparisons with independent plan evaluation. These assess safe resolution, cost efficiency under user priorities, and responses to contact events. Evaluation of LLM-based and symbolic planners shows that safely completing a task does not guarantee the lowest execution costs under the user’s priorities. To address this problem, we introduce Hygiene-NSP. It combines LLM-based grounding, contact-history reconstruction, and CP-SAT to jointly plan hygiene treatment and task execution under user priorities. Hygiene-NSP achieves safe resolution and optimal safe resolution rates of 94.4% and 90.4%, respectively. Both rates are higher than those of the evaluated baseline planners on the full dataset. Project page: https://euron-zc.github.io/HygieneRoboBench/.

## I. INTRODUCTION

Hygiene is a basic requirement for household robots because their actions directly affect the food, utensils, and personal items that people use. Cross-contamination can spread harmful microorganisms to food and household surfaces, creating risks of illness [2], [3]. As robots reuse grippers and tools across tasks, contamination acquired during one activity can be carried into later interactions. A robot may therefore complete the requested task while leaving the people it serves exposed to hygiene hazards.

LLMs can interpret household requests and generate task plans [4], [5]. However, this ability does not ensure that their plans address contact-dependent hygiene risks and satisfy user requirements. Planners must determine how contamination spreads through contact, whether cleaning removes the relevant hazards, and whether unexpected events invalidate an existing plan. In Fig. 1, contamination passes from a tray through the left gripper to a spoon. An unexpected handover then contaminates the right gripper, making the original fruit-handling plan unsafe. In this example, the grippers use washable, replaceable contact pads. The robot must wash or replace the contaminated contact pad before handling the fruit. Cleaning after every action consumes time and supplies and may prevent the task from being completed within the available budget. The planner must therefore determine when treatment is necessary and select an appropriate option based on the remaining resources and user priorities. Existing evaluations address execution safety, adaptive planning, and personalization [6], [7], [8]. However, they lack a systematic joint assessment of these hygiene decisions. Such an assessment must distinguish when contact history changes which continuation is optimal and when the same continuation remains optimal. Table I summarizes their coverage.

TABLE I  
HOUSEHOLD PLANNING CAPABILITIES COVERED BY RELATED BENCHMARKS AND STUDIES.
<table><tr><td>Work</td><td></td><td>Safety Contact checks hygiene</td><td>History input</td><td>Plan updates</td><td>Task</td><td>User limits preferences</td></tr><tr><td>SafeAgentBench [9]</td><td>√</td><td>X</td><td>√</td><td>√</td><td>T</td><td>×</td></tr><tr><td>IS-Bench [10]</td><td>√</td><td>√</td><td>√</td><td>√</td><td>T</td><td>×</td></tr><tr><td>VestaBench [11]</td><td></td><td>X</td><td>√</td><td>√</td><td>T/R</td><td>×</td></tr><tr><td>SafeManip [12]</td><td>√</td><td>√</td><td>×</td><td>×</td><td>T/R</td><td>×</td></tr><tr><td>SIMMER [6]</td><td>√</td><td>√</td><td>×</td><td>X</td><td>T/R</td><td>×</td></tr><tr><td>PARTNR [13]</td><td>X</td><td>X</td><td>√</td><td>√</td><td>T/R</td><td>X</td></tr><tr><td>PbP [8]</td><td>×</td><td>×</td><td>√</td><td>×</td><td>T</td><td>√</td></tr><tr><td>MEMENTO [14]</td><td>×</td><td>×</td><td>√</td><td>√</td><td>T</td><td>√</td></tr><tr><td>AdaPlanBench [7]</td><td>√</td><td>X</td><td>√</td><td>√</td><td>T/R</td><td>√</td></tr><tr><td>HygieneRoboBench</td><td>√</td><td>√</td><td>√</td><td>√</td><td>T/R</td><td>√</td></tr></table>

T: action-order or time constraints. R: resource constraints. —: not specified.

We introduce HygieneRoboBench to evaluate household service planning under contact-dependent hygiene risks. It tests whether robots meet user needs despite unexpected events and within time and resource limits. The benchmark contains 624 instances across 134 task families and spans seven household areas. It covers five activity groups: food preparation, dishware and dining, cleaning and laundry, bedroom and storage, and pet and household care. Tasks capture contamination propagation through two grippers and shared objects, resource trade-offs between washing and replacement, and differences in user priorities over time and consumables. We construct paired instances with the same current physical configuration and task conditions but different contact histories. These pairs test whether planners use history to make appropriate choices. Event and profile comparisons further test plan adaptation and adherence to user priorities. Human review and full-history replay checks validate the constructed tasks. Evaluation covers three aspects: safe task resolution, user priorities and execution costs, and responses to histories and events.

Following the language-to-symbolic planning paradigm exemplified by LLM+P [15], we develop Hygiene-NSP, a hygiene-aware neuro-symbolic planner (NSP). It grounds household requests and reconstructs hygiene conditions from contact history. A trusted compiler builds the constraint problem, and CP-SAT [16] jointly schedules treatment and service actions according to user priorities. Optional state projection reduces the planning representation while preserving the information needed for the current continuation.

All seven evaluated planners show lower performance as the benchmark’s structural task difficulty increases. On controlled history pairs, considering cost optimality alongside safe completion changes the relative ranking of planners.

Hygiene-NSP achieves the highest overall rates of safe resolution (94.4%) and optimal safe resolution (90.4%).

Our contributions are:

• Problem and benchmark. We introduce HygieneRoboBench to evaluate planning with contamination propagation through two grippers and shared objects. It tests how robots use contact history and respond to unexpected events while meeting hygiene and user requirements within time and resource limits.

• Controlled construction and evaluation. We establish history, profile, and event comparisons with metrics for safe resolution, user-priority optimality, and decision responses. Human review, independent plan evaluation, and certification by the Reference support task plausibility and reliable evaluation.

• Systematic evaluation and findings. We evaluate seven planners across task difficulties and household activities. Controlled comparisons reveal differences in optimal planning that safe completion alone does not capture. Independent evaluation further identifies execution violations and inefficiencies in resource use and scheduling.

• Companion neuro-symbolic planner. We introduce Hygiene-NSP, which combines language grounding, history-based state reconstruction, and constraint solving to jointly optimize hygiene treatment and task execution under user priorities.

## II. RELATED WORK

Task and safety benchmarks. BEHAVIOR-1K and Robo-Casa provide household activity definitions and manipulation environments [1], [17]. EmbodiedBench, Embodied Agent Interface, and PlanBench assess agents’ ability to interpret tasks, plan actions, and reason about their effects [18], [19], [20]. SafeAgentBench evaluates safe task planning [9]. IS-Bench evaluates interaction risks and their mitigation order [10]. VestaBench studies long-horizon planning with multiple constraints and adversarial settings [11]. Safe-Manip checks temporal safety properties of manipulation traces [12]. SIMMER models contamination and other latent failures in complete cooking plans [6]. HygieneRoboBench tests how past contact changes the safe ways to complete a task under new conditions.

Adaptive and personalized planning. Inner Monologue uses environmental feedback to revise robot plans [21]. Ada-PlanBench tests plan revision as world and user constraints are revealed [7]. PARTNR evaluates human–robot collaboration under temporal and capability constraints [13]. PbP learns user preferences from demonstrations [8]. MEMENTO uses past interactions to interpret personalized requests [14]. HygieneRoboBench tests whether planners follow given user priorities when choosing among plans that satisfy hygiene and service constraints.

Language and constraint-based planning. SayCan grounds action selection in robot affordances [4]. ProgPrompt and Code as Policies generate programmatic plans and policies [5], [22]. LLM+P connects language interpretation to formal planning [15]. VIRF verifies generated plans [23], while ISR-LLM uses validator feedback to refine them [24]. SayCanPay guides cost-effective search with feasibility and payoff [25]. JIT-Memory maintains safety-relevant information across embodied interaction [26]. Finite-trace temporal logic specifies execution properties [27]. Scheduling research addresses shared-resource limits and revises schedules after disruptions [28], [29]. In household hygiene planning, treatment changes both action safety and available resources. Hygiene-NSP uses history-derived hygiene conditions to jointly plan treatment and household actions under user priorities.

![](images/a19c848ed96b8a2872dad2e5757209d949ff1cfcb1abf90bc74972ab5f658e38.jpg)  
Fig. 2. Dataset construction. (a) Household activities and hygiene guidance supply construction inputs. (b) Five shared registries define task, hygiene, event, resource, and user-priority specifications. (c) Episode Builder combines a base task with contact history, user priorities, event information, and applicable shared definitions to construct controlled instances within a task family. The Verifier groups matched-condition checks, Reference validation, independent replay, and human review for dataset admission.

## III. HYGIENEROBOBENCH

Safely completing household tasks while meeting user needs requires planners to account for contact history, user priorities, and event information. Evaluating these interrelated requirements calls for a unified benchmark that assesses plan feasibility and alignment with user priorities.

## A. Task Formulation

Each HygieneRoboBench task instance requires a robot to safely fulfill a household request within time and resource limits, following the user’s priorities among feasible plans.

Planning starts with a natural-language request u and an execution history $L _ { \leq t }$ recording actions, contacts, treatments, and observed events up to decision time t. The robot also receives structured physical observations $x _ { t }$ and its remaining resources $R _ { t }$ , which describe the current setting for completing the request. Together with event information, user priorities, and shared rules, these inputs form the public planning input at time t

$$
{ \mathcal { T } } _ { t } = ( u , x _ { t } , L _ { \leq t } , R _ { t } , E _ { \leq t } , p , K ) ,\tag{1}
$$

where $E _ { \leq t }$ references event information in $L _ { \leq t }$ available to the planner by time t, and $p$ specifies the user’s priority order over execution costs.

All planners receive the same applicable public rules and skill definitions $\kappa ,$ specifying hygiene requirements and action parameters, preconditions, effects, durations, resource use, and costs.

Given this input, the planner uses execution history to determine whether and how the remaining task can be completed safely. The output plan $\pi _ { t }$ covers the remaining task. It specifies each action’s objects, gripper assignment, and start and end times in abstract time units (TU). The planner also reports estimated execution costs. When no plan is returned, the planner reports infeasibility or an undetermined outcome.

Feasible plans safely complete the remaining task under action, time, and resource constraints, keeping executed actions fixed and respecting ongoing processes. Within the feasible set $\mathcal { F } _ { t }$ , the objective is to minimize execution costs in the lexicographic priority order specified by $p .$

## B. Dataset Construction

Fig. 2(a)–(c) shows inputs, shared registries, and controlled task instance construction, respectively. Fig. 3 summarizes the instance distribution by activity, construction design, and difficulty.

a) Inputs: We adapt BEHAVIOR-1K activities into base tasks, preserving their service goals, functional object roles, and necessary action ordering. The tasks span five activity groups and seven household areas. The hygiene-rule registry draws on peer-reviewed studies and public-health guidance [2], [30], [3]. We formalize these principles as contact-transfer and treatment rules with explicit material and procedure requirements.

b) Shared Registries: Five shared registries define task requirements, action specifications, and execution costs for task construction and evaluation.

The task-source registry anchors each adapted goal to its source activity. Given that goal, the hygiene-rule registry defines contamination propagation, protected objects, and treatments. The action–time–resource registry specifies action preconditions, durations, and resource requirements. The event registry describes new contacts and other changes that update these conditions. Together, these definitions provide a common basis for determining which continuations are safe and executable.

![](images/039ca9ec2bfae1e3cd8e8bf926a39f0278ecc42e4b7534ec778b458463ec78f8.jpg)  
Fig. 3. Dataset composition. The 134 task families comprise 624 instances. From outer to inner, the rings show activity groups, construction designs, and instance difficulty. Each ring independently partitions the same 624 instances, with labels indicating instance counts.

Within this feasible set, the cost–profile registry provides a common basis for comparing plans. It derives the executioncost vector $\mathbf { c } ( \pi )$ from the actions and schedule of plan $\pi .$ The vector has ten dimensions: water, cleaning agent, disinfectant, replacement parts, discarded items, solid waste, energy, tool wear, active robot time, and makespan. Each user profile specifies a priority order $p$ over these cost dimensions. For the same task, changing $p$ can change how plans are ranked while leaving action costs fixed. In the fruit example, hygiene and action requirements determine whether washing or replacing the right contact pads is feasible. User priorities determine whether saving replacement parts or saving water takes precedence. Applicable entries from all five registries form the public definitions K, keeping task construction, planning, and evaluation consistent.

We use an LLM to screen candidate action durations and consumable requirements from online sources. Human reviewers check their plausibility and relative magnitudes. We express these parameters in abstract units for consistent plan comparisons. Examples of the shared registries are shown in the supplementary video.

c) Controlled Task Instances: Episode Builder combines a base task with contact history L, user priorities $p ,$ and event information E. It produces a task instance, its public input $\mathcal { T } _ { t }$ , and private assessment information. For the same base task, we combine different $L , p ,$ and $E$ according to the controlled comparison design to form a task family. The middle ring of Fig. 3 reports instance counts by construction type. Crossed-history groups combine relevant and irrelevant contact variations in a $2 \times 2$ design, accounting for 200 instances. Event contrasts include no-event, relevant-event, and irrelevant-event conditions. Of the 24 instances with a prior violation, 10 admit a safe continuation and 14 have no feasible continuation.

For difficulty analysis, we use seven instance-level structural features: request composition, remaining-goal composition, goal-preservation conflicts, temporal coupling, active hazards on multiple carriers, constrained treatment resources, and relevant events known to the planner. These features define three tiers: easy, moderate, and hard. The inner ring of Fig. 3 reports the tier distribution.

Dataset admission combines matched-condition checks, Reference validation of feasibility and costs, independent full-history replay, and human review. Three reviewers assess the entire dataset for task wording, household plausibility, and treatment applicability.

## C. Evaluation Protocol and Metrics

We evaluate submitted continuation plans and judgments of infeasibility. The Evaluator checks submitted plans through full-history replay, independently of the planning solver. It checks whether the submitted plan achieves the goals and satisfies hygiene, timing, and resource constraints. It independently computes execution costs from the plan’s actions and schedule. The Reference provides feasibility and certified optimal costs under the user’s priorities within the declared action and time bounds.

Planners receive only the public input $\mathcal { T } _ { t }$ . Every response, including one without a plan, is assessed against private Reference targets. These targets and evaluation feedback remain hidden from planners.

For each contact-event instance, all planners receive the same updated history and task context. We report SR and OSR separately for relevant and irrelevant events, using the same instances for all planners. These instances have safe prefixes, certified-feasible continuations, and certified reference optima.

a) Safe Resolution: Safe Resolution Rate (SR) measures safe completion or correct rejection of the remaining task from the prescribed decision point:

$$
\mathrm { S R } = \textstyle \frac { 1 } { N } \sum _ { i \in \mathcal { D } } \left[ f _ { i } b _ { i } + ( 1 - f _ { i } ) r _ { i } \right] .\tag{2}
$$

Here, $\mathcal { D }$ is the common comparison set and $\textit { N } = \ | \mathcal { D } |$ The dataset contains 604 instances with certified-feasible continuations and 20 with certified-infeasible continuations. The indicator $f _ { i }$ denotes continuation feasibility, and $r _ { i }$ denotes an explicit infeasibility judgment without a submitted plan. Safe completion is $b _ { i } \ = \ v _ { i } ,$ , where $v _ { i }$ denotes a submitted continuation satisfying the goal, hygiene, timing, resource, and other feasibility conditions in Section III-A. Earlier violations remain recorded but do not by themselves determine continuation feasibility. Undetermined outcomes and incorrect rejections do not count as resolutions.

b) User Priorities and Execution Costs: Under user priorities $p ,$ we compare recomputed plan costs with the certified reference optimum. Optimal Safe Resolution Rate (OSR) requires safe completion at this optimum or correct rejection of a certified-infeasible task. Profile pairs vary only user priorities $( p _ { 1 }$ versus $p _ { 2 } )$ , with history $L ,$ event information $E ,$ and other task conditions fixed. Profile-Pair Success (PS) counts pairs where both instances complete safely at their respective reference-optimal costs.

$$
\begin{array} { l } { \displaystyle \mathrm { O S R } = \frac { 1 } { N } \sum _ { i \in \mathcal { D } } \big [ f _ { i } b _ { i } o _ { i } m _ { i } + ( 1 - f _ { i } ) r _ { i } \big ] , } \\ { \displaystyle \mathrm { P S } = \sum _ { ( i , j ) \in \mathcal { U } } b _ { i } b _ { j } m _ { i } m _ { j } . } \end{array}\tag{3}
$$

Here, $o _ { i }$ indicates a certified reference optimum under instance $i \ ' s$ priorities $p .$ For submitted plan $\pi _ { i } , m _ { i } = 1$ when its recomputed cost $\mathbf { c } ( \pi _ { i } )$ matches that optimum, and zero when no match is established. The set U contains profile pairs whose members both have safe prefixes and certified-feasible continuations. We report PS on general profile pairs and on ranking-flip and ranking-invariant controls. These controls are defined by whether changing user priorities changes the cost ranking of verified alternative plans.

c) History Response: History pairs vary designated contact records in $L$ while keeping event information $E ,$ user priorities $p ,$ and other task conditions fixed. History-pair completion (HC) counts the pairs in which both instances complete safely. $\mathrm { H C ^ { * } }$ additionally requires both to match their respective reference-optimal costs:

$$
\mathrm { H C } = \sum _ { ( i , j ) \in \mathcal { H } } b _ { i } b _ { j } , \quad \mathrm { H C ^ { * } } = \sum _ { ( i , j ) \in \mathcal { H } } b _ { i } b _ { j } m _ { i } m _ { j } .\tag{4}
$$

Both metrics use the same pair set ${ \mathcal { H } } ,$ with each member required to have a safe prefix and a certified-feasible continuation. In this benchmark, relevant-history pairs admit a shared safe continuation but have different certified optimal cost vectors. They therefore require different optimal continuations. Irrelevant-history pairs admit a shared certifiedoptimal continuation under the same task conditions.

## IV. HYGIENE-NSP: A HYGIENE-AWARE NEURO-SYMBOLIC PLANNER

Hygiene-NSP reconstructs hygiene conditions from contact history and jointly plans treatment and service actions under user priorities. These decisions are coupled: a washed gripper can be contaminated again, and cleaning or replacement consumes resources needed for service. Algorithm 1 summarizes one planning call, repeated on the updated public input after an action or an update to event information.

## A. Language Grounding and Constraint Construction

The LLM Grounder reads the request and lists the goals and their types, identifies the objects involved, and records the time requirements. It also includes the supplied user priorities and event information. The recorded observations and resource amounts keep their given values.

The Trusted Compiler converts these goals and the shared rules K into a planning problem $\mathcal { P } _ { t }$ . It checks that object names match available objects, sets the requirements and effects of each action, and updates the affected constraints using event information.

Algorithm 1 Hygiene-NSP continuation planning   
Require: Public input $\mathcal { T } _ { t } ,$ solver time budget B   
Ensure: Plan and modeled costs, or infeasible/undetermined   
1: $\mathscr { P } _ { t } \gets \mathrm { C o m p i l e } _ { \mathcal { K } } ( \mathrm { G r o u n d } ( \mathscr { T } _ { t } ) )$   
2: $S _ { t } \gets \mathrm { R e p l a y } _ { \mathcal { K } } ( \widetilde { L } _ { \leq t } ) ; \bar { S } _ { t } \gets \tilde { S } _ { t }$   
3: if projection is enabled and supported then   
4: $S _ { t } ^ { \mathsf { ^ { \prime } } } \gets \mathrm { P r o j e c t } ( S _ { t } , \mathcal { P } _ { t } )$   
5: if Consistent $( \dot { S } _ { t } ^ { \prime } , S _ { t } )$ then $\bar { S } _ { t } \gets S _ { t } ^ { \prime }$   
6: $\widehat { \mathcal { F } } _ { t } \gets$ JointModel $( \mathcal { P } _ { t } , \bar { S } _ { t } )$   
7: $( q , \pi _ { t } , \mathbf { c } ) \gets$ LexCP-SAT $( \widehat { \mathcal { F } } _ { t } , p , B )$   
8: return $( q , \pi _ { t } , \mathbf { c } )$   
q distinguishes success, proved infeasibility, and undetermined outcomes.

## B. History Replay and State Reconstruction

Physical observations alone need not indicate the hygiene consequences of earlier actions. Moreover, a record that an object was washed is insufficient if later contact can contaminate it again. The planner derives the current hygiene state from the order and effects of past contacts and treatments.

We start with the given initial conditions and follow the recorded actions in $L _ { \leq t }$ in order. The shared rules K specify how contamination from each source spreads and how completed treatments change it. This process, called history replay, gives the hygiene state for the remaining task:

$$
S _ { t } = \mathrm { R e p l a y } _ { \kappa } ( L _ { \leq t } ) .\tag{5}
$$

History replay reconstructs the current hygiene state while preserving past violations and cumulative resource consumption. After another action or a new event, the planner updates the planning problem and replays the full history before planning the continuation.

Optional state projection retains state information and past evidence relevant to the remaining task. A checker validates the retained states, past violations, and cumulative resource consumption against full replay before the smaller state $\bar { S } _ { t }$ is used. With projection disabled or its support conditions unestablished, planning uses the full state, $\bar { S } _ { t } = S _ { t }$

## C. Joint Hygiene and Service Optimization

A necessary hygiene treatment is also an action that consumes time and resources. Fixing a treatment before scheduling the service can exclude alternatives that better satisfy the remaining constraints or user priorities. We therefore choose treatment routes, service actions, and their timing jointly, first enforcing feasibility and then comparing feasible plans by their execution costs.

CP-SAT [16] uses $\mathcal { P } _ { t }$ and $\bar { S } _ { t }$ to choose actions and their start and end times. Only selected actions activate their time intervals and update task and hygiene states according to the rules. Other rules prevent conflicting uses of the same gripper, keep simultaneous resource use within shared capacity, and keep total consumption within the remaining supplies. Starting with $\bar { S } _ { t } ,$ , the model tracks contamination to prevent it from reaching protected objects. These constraints define $\widehat { \mathcal { F } } _ { t } .$ , the feasible set of Hygiene-NSP’s grounded planning model. The benchmark feasible set $\mathcal { F } _ { t }$ follows the task requirements in Section III-A. The independent Evaluator checks submitted plans against those requirements.

A. Overall performance and instance difficulty (%)
<table><tr><td>Planner</td><td colspan="2">Overall (N = 624)</td><td colspan="2">Easy (n = 200)</td><td colspan="2">Moderate (n = 238)</td><td colspan="2">Hard (n = 186)</td></tr><tr><td></td><td>SR</td><td>OSR</td><td>SR</td><td>OSR</td><td>SR</td><td>OSR</td><td>SR</td><td>OSR</td></tr><tr><td>GPT-5.6-Sol</td><td>78.5</td><td>76.9</td><td>87.5</td><td>85.5</td><td>78.6</td><td>77.7</td><td>68.8</td><td>66.7</td></tr><tr><td>GPT-5.6-Luna</td><td>46.6</td><td>37.0</td><td>55.0</td><td>45.5</td><td>52.1</td><td>41.2</td><td>30.6</td><td>22.6</td></tr><tr><td>Gemini 3.6 Flash</td><td>89.7</td><td>86.1</td><td>95.0</td><td>91.5</td><td>90.8</td><td>88.7</td><td>82.8</td><td>76.9</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>90.7</td><td>75.0</td><td>96.0</td><td>83.5</td><td>92.4</td><td>71.8</td><td>82.8</td><td>69.9</td></tr><tr><td>MiniMax-M3</td><td>67.0</td><td>57.2</td><td>74.5</td><td>66.0</td><td>70.2</td><td>58.8</td><td>54.8</td><td>45.7</td></tr><tr><td>Rule-SAT</td><td>30.9</td><td>30.0</td><td>38.0</td><td>37.0</td><td>36.6</td><td>35.3</td><td>16.1</td><td>15.6</td></tr><tr><td>Hygiene-NSP</td><td>94.4</td><td>90.4</td><td>96.5</td><td>96.0</td><td>96.2</td><td>91.6</td><td>89.8</td><td>82.8</td></tr></table>

B. Performance by activity (%)
<table><tr><td></td><td colspan="2">Food preparation  $( n = 1 2 6 )$ </td><td colspan="2">Dishware &amp; dining  $( n = 1 3 8 )$ </td><td colspan="2">Cleaning &amp; laundry  $( n = 1 3 0 )$ </td><td colspan="2">Bedroom &amp; storage  $( n = 1 1 2 )$ </td><td colspan="2">Pet &amp; household care  $( n = 1 1 8 )$ </td></tr><tr><td>Planner</td><td>SR</td><td>OSR</td><td>SR</td><td>OSR</td><td>SR</td><td>OSR</td><td>SR</td><td>OSR</td><td>SR</td><td>OSR</td></tr><tr><td>GPT-5.6-Sol</td><td>76.2</td><td>73.8</td><td>73.9</td><td>69.6</td><td>84.6</td><td>83.8</td><td>75.9</td><td>75.9</td><td>82.2</td><td>82.2</td></tr><tr><td>GPT-5.6-Luna</td><td>38.1</td><td>29.4</td><td>44.9</td><td>33.3</td><td>54.6</td><td>46.9</td><td>34.8</td><td>31.2</td><td>60.2</td><td>44.1</td></tr><tr><td>Gemini 3.6 Flash</td><td>86.5</td><td>85.7</td><td>87.7</td><td>84.1</td><td>92.3</td><td>86.2</td><td>91.1</td><td>88.4</td><td>91.5</td><td>86.4</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>83.3</td><td>65.9</td><td>88.4</td><td>64.5</td><td>91.5</td><td>79.2</td><td>96.4</td><td>86.6</td><td>94.9</td><td>81.4</td></tr><tr><td>MiniMax-M3</td><td>69.0</td><td>60.3</td><td>58.7</td><td>50.7</td><td>75.4</td><td>61.5</td><td>67.9</td><td>63.4</td><td>64.4</td><td>50.8</td></tr><tr><td>Rule-SAT</td><td>38.1</td><td>34.9</td><td>15.9</td><td>15.9</td><td>37.7</td><td>36.9</td><td>37.5</td><td>36.6</td><td>27.1</td><td>27.1</td></tr><tr><td>Hygiene-NSP</td><td>100.0</td><td>100.0</td><td>89.9</td><td>86.2</td><td>86.9</td><td>85.4</td><td>96.4</td><td>82.1</td><td>100.0</td><td>98.3</td></tr></table>

Activity headers give instance counts. Across Tables II, IV, and V, bold and underlining indicate the best and second-best distinct values for each metric.

The solver optimizes costs in priority order $p ,$ fixing each proved optimum before proceeding to the next component:

$$
\begin{array} { r } { \begin{array} { r l } & { \gamma _ { j } = \underset { \pi \in \widehat { \mathcal { F } } _ { t } } { \operatorname* { m i n } } c _ { p _ { j } } ( \pi ) , } \\ & { \mathrm { s u b j e c t ~ t o } \quad c _ { p _ { \ell } } ( \pi ) = \gamma _ { \ell } \quad \mathrm { f o r ~ a l l ~ } \ell < j . } \end{array} } \end{array}\tag{6}
$$

Here, $p _ { j }$ selects the next component of $\mathbf { c } ( \pi )$ , and $\gamma _ { \ell }$ is a previously proved stage optimum. If optimization remains incomplete, the solver retains any feasible plan found without claiming full lexicographic optimality. If no plan is found and infeasibility is not proved, it returns an undetermined outcome. When optimization succeeds, it returns $\pi _ { t }$ and its modeled cost vector for independent evaluation.

## V. EXPERIMENTS

We evaluate overall resolution (RQ1), plan choices under different histories and user priorities (RQ2), and safe, costoptimal continuation after contact events (RQ3). A separate method analysis examines state projection.

## A. Experimental Setup

We evaluated GPT-5.6-Sol, GPT-5.6-Luna, Gemini 3.6 Flash, DeepSeek-V4.1-Flash, and MiniMax-M3 on 624 instances across 134 task families, alongside Rule-SAT and Hygiene-NSP. The language models generate complete plans directly. Rule-SAT constructs constraint problems through rule-based parsing, while Hygiene-NSP uses GPT-5.6-Sol as its language grounder. Both planners use CP-SAT. We used high reasoning settings where available and enabled thinking for MiniMax-M3.

All planners receive the same public input $\mathcal { T } _ { t }$ and applicable rules K, and their outputs are evaluated under the protocol in Section III-C. The main comparison uses full contact history. Event response is assessed on the event subset of this dataset, and state projection is examined separately.

## B. RQ1: How Reliably Do Planners Resolve Requests?

We evaluate safe resolution (SR) and optimal safe resolution (OSR) under user priorities. Both metrics credit correct rejection of infeasible continuations. Table II reports overall results, difficulty tiers, and activity breakdowns.

HygieneRoboBench distinguishes planner performance across task difficulty and household activities. Hygiene-NSP achieves the highest overall SR and OSR, at 94.4% and 90.4%, respectively (Table II-A). SR and OSR decrease from easy to moderate to hard instances for all seven planners. This consistent trend shows that HygieneRoboBench’s difficulty tiers, defined by task structure, distinguish planner performance at different levels of task complexity. Activitylevel results expose differences hidden by the overall ranking: Gemini 3.6 Flash achieves the highest SR and OSR in cleaning and laundry (Table II-B).

HygieneRoboBench identifies execution and hygiene violations that checks of service-goal completion alone can miss. Hygiene-NSP produces no invalid continuations in the full evaluation (Table III). In a leftover-storage task, DeepSeek-V4.1-Flash completes the requested placements. However, it grasps a protected food container with a gripper exposed to peanut residue through a shared handle. Fullhistory replay detects the resulting contamination. On the same instance, Hygiene-NSP replaces the contact insert before moving and placing the objects. Its plan is safe and matches the reference-optimal cost.

FAILURE AND COST OUTCOMES ACROSS 624 INSTANCES PER PLANNER. COUNTS ARE MUTUALLY EXCLUSIVE. THE FIRST THREE COLUMNS FAIL SR. THE LAST PASSES SR BUT NOT OSR.
<table><tr><td></td><td>Other no plan</td><td>Incorrect rejection</td><td>Invalid continuation</td><td>Safe, nonoptimal</td></tr><tr><td>GPT-5.6-Sol</td><td>1</td><td>25</td><td>108</td><td>10</td></tr><tr><td>GPT-5.6-Luna</td><td>27</td><td>112</td><td>194</td><td>60</td></tr><tr><td>Gemini 3.6 Flash</td><td>8</td><td>12</td><td>44</td><td>23</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>3</td><td>7</td><td>48</td><td>98</td></tr><tr><td>MiniMax-M3</td><td>33</td><td>10</td><td>163</td><td>61</td></tr><tr><td>Rule-SAT</td><td>415</td><td>12</td><td>4</td><td>6</td></tr><tr><td>Hygiene-NSP</td><td>22</td><td>13</td><td>0</td><td>25</td></tr></table>

Independent cost evaluation identifies resource and scheduling inefficiencies in safe plans. DeepSeek-V4.1- Flash returns 98 safe, nonoptimal plans (Table III). In 41 of these plans, the first nonoptimal cost dimension under user priorities is water or replacement parts. In a hot-cocoa task, DeepSeek-V4.1-Flash and Hygiene-NSP use the same actions and consumables. DeepSeek-V4.1-Flash places the marshmallows after cooking and pouring the cocoa, taking 19 TU in total. Hygiene-NSP places them during cooking and finishes in 13 TU, matching the reference optimum.

## C. RQ2: Do Plans Adapt to History and User Priorities?

Controlled pairs test whether planners select safe, optimal continuations under different histories and user priorities.

The 62 profile-contrast instances form 31 pairs. Two pairs have prior violations and do not meet the safe-prefix requirement, leaving 24 ranking-flip and five ranking-invariant pairs. The spoon–fruit task family contributes one additional ranking-flip pair.

History-pair evaluation identifies differences in optimal planning that safe completion counts alone do not capture. On relevant-history pairs, Hygiene-NSP achieves the highest HC<sup>∗</sup>, at 156 (Table IV-A). DeepSeek-V4.1-Flash has a higher HC than Gemini 3.6 Flash but a lower HC<sup>∗</sup>. Including optimality therefore reverses their ranking. On irrelevant-history pairs, Hygiene-NSP and DeepSeek-V4.1- Flash both achieve an HC of 86, but their HC<sup>∗</sup> counts are 78 and 60, respectively.

We further evaluated saved planner outputs under history swaps across the 180 relevant-history and 92 irrelevanthistory pairs. We tested both directions and kept submitted actions and schedules unchanged. Across seven planners, none of the 1,699 originally safe, optimal plan replays remained both safe and optimal under relevant-history swaps. All 1,288 irrelevant-history swap evaluations preserved the original safety and optimality outcomes. These results support the validity of the controlled history pairs: relevanthistory changes affect plan safety or optimality, whereas

CONTROLLED HISTORY AND PRIORITY PAIRS. ENTRIES GIVE HC/HC<sup>∗</sup> OR PS COUNTS; n COUNTS ELIGIBLE PAIRS FROM THE CONSTRUCTION GROUPS IN FIG. 3. EVALUATION SETS MAY SHARE INSTANCES.

A. History pairs (HC / HC<sup>∗</sup>)
<table><tr><td>Planner</td><td>Relevant (n = 180)</td><td>Irrelevant (n = 92)</td></tr><tr><td>GPT-5.6-Sol</td><td>126 / 123</td><td>69  / 66</td></tr><tr><td>GPT-5.6-Luna</td><td>57 /  38</td><td>27  /  16</td></tr><tr><td>Gemini 3.6 Flash</td><td>157  /  141</td><td>81  / 76</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>162 / 118</td><td>86 / 60</td></tr><tr><td>MiniMax-M3</td><td>94  /  65</td><td>51  /  33</td></tr><tr><td>Rule-SAT</td><td>74 / 71</td><td>50 /  50</td></tr><tr><td>Hygiene-NSP</td><td>168 / 156</td><td>86  /  78</td></tr></table>

<table><tr><td colspan="4">B. Profile pairs (PS)</td></tr><tr><td>Planner</td><td>Ranking flip (n = 25)</td><td>Ranking invariant (n = 5)</td><td>General (n = 268)</td></tr><tr><td>GPT-5.6-Sol</td><td>20</td><td>3</td><td>184</td></tr><tr><td>GPT-5.6-Luna</td><td>7</td><td>3</td><td>58</td></tr><tr><td>Gemini 3.6 Flash</td><td>23</td><td>5</td><td>218</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>16</td><td>3</td><td>178</td></tr><tr><td>MiniMax-M3</td><td>11</td><td>3</td><td>108</td></tr><tr><td>Rule-SAT</td><td>6</td><td>0</td><td>83</td></tr><tr><td>Hygiene-NSP</td><td>24</td><td>3</td><td>234</td></tr></table>

irrelevant-history changes preserve the evaluated outcomes. The supplementary video presents the swap evaluation procedure and results by planner.

User-priority pairs evaluate safe, optimal planning under different resource preferences. Hygiene-NSP succeeds on 234 of 268 general profile pairs and 24 of 25 ranking-flip pairs, the highest counts in both groups (Table IV-B). In the spoon–fruit task illustrated in Fig. 1, Hygiene-NSP switches between washing and replacement under the two priority profiles, attaining the reference optimum in both cases.

D. RQ3: Can Planners Continue Safely at Optimal Costs After Contact Events?

We evaluate 50 relevant-event and 50 irrelevant-event instances across 25 task families. Table V reports SR and OSR for both conditions.

TABLE V  
CONTACT-EVENT RESULTS. ENTRIES REPORT SR/OSR (%); EACH CONDITION CONTAINS 50 INSTANCES.
<table><tr><td>Planner</td><td>Relevant SR/OSR</td><td>Irrelevant SR/OSR</td></tr><tr><td>GPT-5.6-Sol</td><td>78/74</td><td>74/70</td></tr><tr><td>GPT-5.6-Luna</td><td>40/24</td><td>34/30</td></tr><tr><td>Gemini 3.6 Flash</td><td>92/90</td><td>94/94</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>88/66</td><td>92/86</td></tr><tr><td>MiniMax-M3</td><td>76/62</td><td>62/56</td></tr><tr><td>Rule-SAT</td><td>12/12</td><td>12/12</td></tr><tr><td>Hygiene-NSP</td><td>90/88</td><td>92/92</td></tr></table>

Hygiene-NSP achieves high safe and optimal safe resolution rates under both contact-event conditions. Under relevant and irrelevant events, its SR is 90% and 92%, respectively, while its OSR is 88% and 92%. DeepSeek-V4.1- Flash shows a larger difference between the two conditions, with OSR of 66% and 86%. Gemini 3.6 Flash achieves the highest OSR in both conditions.

Hygiene-NSP and Gemini exhibit different failure types. Across the 100 contact-event instances, Hygiene-NSP’s nine SR failures comprise five undetermined outcomes and four incorrect rejections. All of its submitted plans pass execution checks. Gemini’s seven SR failures comprise four incorrect rejections and three invalid continuations.

## E. Method Analysis: State Projection

We compared full-state (Full) and projected-state (Projected) planning on 59 task families with three paired runs each. The median family-level reduction in local planning time was 20.1%, excluding LLM calls. Across 177 pairs, including 21 full-state fallbacks, plan status and cost agreed in 175 pairs, with no new hygiene violations. In one differing pair, Projected found a safe plan where Full returned undetermined. In the other, it reduced makespan from 17 to 10 TU relative to Full’s feasible plan.

## VI. CONCLUSION AND FUTURE WORK

We introduced HygieneRoboBench, with 624 instances across 134 task families. Controlled construction and independent evaluation assess household service planning under contact history, user priorities, and contact events. All seven planners perform worse as structural task difficulty increases, demonstrating the benchmark’s ability to distinguish performance across task complexity. Controlled pairs further reveal differences in optimal planning that safe completion counts alone do not capture. Hygiene-NSP achieves the highest overall SR and OSR, at 94.4% and 90.4%, respectively, with no invalid submitted continuations.

HygieneRoboBench provides a common basis for evaluating hygiene decisions within complete service tasks, connecting contact-history reasoning with resource trade-offs under user priorities. The current evaluation uses explicit contact and treatment rules. Future work will extend evaluation to physical execution, using measured contact and treatment outcomes to assess these planning decisions.

## REFERENCES

[1] C. Li et al., “BEHAVIOR-1K: A benchmark for embodied AI with 1,000 everyday activities and realistic simulation,” in Proc. 6th Conf. Robot Learn. (CoRL), ser. Proc. Mach. Learn. Res., vol. 205, 2023, pp. 80–93.

[2] T. A. Cogan, J. Slader, S. F. Bloomfield, and T. J. Humphrey, “Achieving hygiene in the domestic kitchen: The effectiveness of commonly used cleaning procedures,” J. Appl. Microbiol., vol. 92, no. 5, pp. 885–892, 2002, doi: 10.1046/j.1365-2672.2002.01598.x.

[3] Centers for Disease Control and Prevention, “When and how to clean and disinfect your home,” CDC, Jan. 31, 2025, accessed: Sep. 14, 2026. [Online]. Available: https://www.cdc.gov/hygiene/about/when-a nd-how-to-clean-and-disinfect-your-home.html

[4] B. Ichter et al., “Do as I can, not as I say: Grounding language in robotic affordances,” in Proc. 6th Conf. Robot Learn. (CoRL), ser. Proc. Mach. Learn. Res., vol. 205, 2023, pp. 287–318.

[5] I. Singh et al., “ProgPrompt: Generating situated robot task plans using large language models,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 11 523–11 530.

[6] X. Lu, R. H. Zhang, and R. Zhang, “SIMMER: Benchmarking latent failures in LLM executable planning with a world model,” in Third Conference on Language Modeling (COLM), 2026. [Online]. Available: https://openreview.net/forum?id=qthVjAZgAm

[7] J. Liu et al., “AdaPlanBench: Evaluating adaptive planning in large language model agents under world and user constraints,” arXiv preprint arXiv:2606.05622, 2026.

[8] M. Xu, X. Yang, W. Liang, C. Zhang, and Y. Zhu, “Learning to plan with personalized preferences,” arXiv preprint arXiv:2502.00858v3, 2025.

[9] S. Yin et al., “SafeAgentBench: A benchmark for safe task planning of embodied LLM agents,” arXiv preprint arXiv:2412.13178, 2024.

[10] X. Lu et al., “IS-Bench: Evaluating interactive safety of VLM-driven embodied agents in daily household tasks,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 42, 2026, pp. 35 680– 35 688, doi: 10.1609/aaai.v40i42.40880.

[11] T. Sadhu, Y. Chen, and A. Pesaranghader, “VestaBench: An embodied benchmark for safe long-horizon planning under multi-constraint and adversarial settings,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, 2025, pp. 2122–2145, doi: 10.18653/v1/2025.emnlp-industry.149.

[12] C. Huang, K. V. Huynh, S. Elbaum, Z. Kira, and L. Feng, “SafeManip: A property-driven benchmark for temporal safety evaluation in robotic manipulation,” arXiv preprint arXiv:2605.12386, 2026.

[13] M. Chang et al., “PARTNR: A benchmark for planning and reasoning in embodied multi-agent tasks,” in International Conference on Learning Representations, 2025.

[14] T. Kwon et al., “Embodied agents meet personalization: Investigating challenges and solutions through the lens of memory utilization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2026.

[15] B. Liu et al., “LLM+P: Empowering large language models with optimal planning proficiency,” 2023, arXiv:2304.11477.

[16] L. Perron, F. Didier, and S. Gay, “The CP-SAT-LP solver,” in Proc. 29th Int. Conf. Principles Pract. Constraint Program. (CP), ser. LIPIcs, vol. 280, 2023, pp. 3:1–3:2, doi: 10.4230/LIPIcs.CP.2023.3.

[17] S. Nasiriany et al., “RoboCasa: Large-scale simulation of everyday tasks for generalist robots,” in Proc. Robot. Sci. Syst. (RSS), 2024, doi: 10.15607/RSS.2024.XX.050.

[18] R. Yang et al., “EmbodiedBench: Comprehensive benchmarking multimodal large language models for vision-driven embodied agents,” in Proc. 42nd Int. Conf. Mach. Learn. (ICML), ser. Proc. Mach. Learn. Res., vol. 267, 2025, pp. 70 576–70 631.

[19] M. Li et al., “Embodied agent interface: Benchmarking LLMs for embodied decision making,” in Adv. Neural Inf. Process. Syst., vol. 37, 2024, pp. 100 428–100 534, doi: 10.52202/079017-3188.

[20] K. Valmeekam, M. Marquez, A. Olmo, S. Sreedharan, and S. Kambhampati, “PlanBench: An extensible benchmark for evaluating large language models on planning and reasoning about change,” in Adv. Neural Inf. Process. Syst., vol. 36, 2023, pp. 38 975–38 987, doi: 10.52202/075280-1693.

[21] W. Huang et al., “Inner monologue: Embodied reasoning through planning with language models,” in Proc. 6th Conf. Robot Learn. (CoRL), ser. Proc. Mach. Learn. Res., vol. 205, 2023, pp. 1769–1782.

[22] J. Liang et al., “Code as policies: Language model programs for embodied control,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2023, pp. 9493–9500, doi: 10.1109/icra48891.2023.10160591.

[23] F. Wu, X. Zheng, Y. Qu, Z. Wang, Z. Feng, and H. Li, “Grounding generative planners in verifiable logic: A hybrid architecture for trustworthy embodied AI,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2026.

[24] Z. Zhou, J. Song, K. Yao, Z. Shu, and L. Ma, “ISR-LLM: Iterative self-refined large language model for long-horizon sequential task planning,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2024, pp. 2081–2088, doi: 10.1109/icra57147.2024.10610065.

[25] R. Hazra, P. Zuidberg Dos Martires, and L. De Raedt, “SayCanPay: Heuristic planning with large language models using learnable domain knowledge,” in Proc. AAAI Conf. Artif. Intell., vol. 38, no. 18, 2024, pp. 20 123–20 133, doi: 10.1609/aaai.v38i18.29991.

[26] B. Sima, L. Wang, X. Lu, K. He, and X. Yang, “Self-evolving just-intime memory for proactive embodied safety,” 2026, arXiv:2607.16247.

[27] G. De Giacomo and M. Y. Vardi, “Linear temporal logic and linear dynamic logic on finite traces,” in Proc. 23rd Int. Joint Conf. Artif. Intell. (IJCAI), 2013, pp. 854–860.

[28] S. Hartmann and D. Briskorn, “An updated survey of variants and extensions of the resource-constrained project scheduling prob-

lem,” Eur. J. Oper. Res., vol. 297, no. 1, pp. 1–14, 2022, doi: 10.1016/j.ejor.2021.05.004.

[29] I. Sabuncuoglu and M. Bayiz, “Analysis of reactive scheduling problems in a job shop environment,” Eur. J. Oper. Res., vol. 126, no. 3, pp. 567–586, 2000, doi: 10.1016/S0377-2217(99)00311-2.

[30] B. Bedford, G. Liggans, L. Williams, and L. Jackson, “Allergen removal and transfer with wiping and cleaning methods used in retail and food service establishments,” J. Food Prot., vol. 83, no. 7, pp. 1248–1260, 2020, doi: 10.4315/JFP-20-025.