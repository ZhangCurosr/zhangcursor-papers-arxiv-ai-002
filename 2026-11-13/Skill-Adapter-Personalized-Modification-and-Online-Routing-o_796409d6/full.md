# Skill Adapter: Personalized Modification and Online Routing of General-Purpose Skills

Qijia Zhuang<sup>1,\*</sup>, Yanyi Gou<sup>1,\*</sup>, Wenbin Ma<sup>2,\*</sup>, Yuxiang Ren<sup>1,†</sup>

<sup>1</sup>Nanjing University <sup>2</sup>Jiangnan University

These authors contributed equally and share first authorship. <sup>†</sup>Corresponding author.

## Abstract

Large language model (LLM) agents use external skills to reuse task procedures and experience with tools. Generalpurpose skills describe shared execution procedures, while specific tool choices, operating practices, and output requirements vary across users. In long-horizon tasks, subgoals, available inputs, and environmental states change as execution progresses. Agents must select currently applicable skills from an expanding skill pool and prepare candidates for subsequent steps.

We propose Skill Adapter, which enables personalized modification and online routing of general-purpose skills. The system distills local adjustments from user requirements and execution feedback and stores them as Adapters linked to general-purpose skills. The routing agent selects generalpurpose skills and applicable Adapters based on the current subtask, execution progress, and environmental feedback, incrementally maintaining a task-level online skill library that the main LLM reads on demand. The system preloads candidates for subsequent nodes and rechecks their applicability once actual inputs are determined. Reinforcement learning optimizes the two types of decisions using modification gains and routing outcomes, respectively.

We evaluate Skill Adapter on ALFWorld, WebShop, and SkillsBench. Compared with SkillRL, it improves task success rates by an average of 11.43 percentage points and reduces mean execution time on all three benchmarks. Skill Adapter retains its performance advantage when the skill pool grows to 1,000 skills.

## Introduction

Large language model (LLM) agents are increasingly capable of planning, invoking tools, and interacting continuously with their environments. Their applications have expanded from single-turn question answering to web interaction, embodied decision-making, and complex workflows (Yao et al. 2023; Schick et al. 2023; Liu et al. 2023; Wu et al. 2023). In these tasks, organizing past experience into reusable skills helps reduce repeated exploration and improve the stability of longhorizon execution (Wang et al. 2023; Zhao et al. 2023; Shinn et al. 2023). Skill pools thus provide an external capability layer that connects task experience to agent execution.

General-purpose skills describe shared procedures for a class of tasks, while specific execution procedures vary with user requirements. For example, when translating an English contract into a Chinese PDF, some users focus on the translated content, while others need to preserve the original pagination, layout, and annotations. Both sets of requirements involve document extraction, translation, and PDF generation, but difer in the fields to extract, tool parameters, and checks to perform.

Personalized modification must express user-specific operating requirements within shared procedures. Online routing must select the general-purpose skills and applicable Adapters for the current subtask according to the execution state. Semantic matching between tasks and skills provides initial candidates, while actual inputs, tool availability, and execution feedback determine whether each candidate can be used at the current step.

Subsequent steps in long-horizon tasks depend on the actual results of preceding operations. For example, contract layout reconstruction requires selecting a procedure based on the extracted layout information, and feedback from checking the generated output may introduce further processing requirements. Routing must therefore update skill provision as execution progresses, reusing valid entries and adding or replacing candidates. Growth of the global skill pool also increases the cost of retrieval and selection.

To address these challenges, we propose Skill Adapter. The framework uses general-purpose skills as a shared foundation and stores user-specific modifications as Adapters linked to their original skills. Each Adapter describes applicability conditions and local adjustments to tools, parameters, operating steps, and output checks. As the task progresses, online routing selects general-purpose skills and their applicable Adapters and organizes the results into a task-level online skill library. The main LLM reads the skills needed for the current node on demand, while the routing agent prepares candidates for subsequent nodes and incrementally updates library entries in response to new inputs and feedback.

We use group relative policy optimization (GRPO) to train the modification and routing policies. The modification policy receives rewards based on changes in task success rate, while the routing policy receives rewards based on task execution outcomes and the quality of skill selection. Our contributions are threefold:

• We introduce a personalized modification mechanism for general-purpose skills that expresses user requirements through local Adapters and preserves their links to original skills and supporting feedback.

![](images/27cf330ff0614fc0e70446d1d25a1b9abf46d7bc750d419845d1349c1dbab572.jpg)  
Figure 1: The influence of user requirements on skill selection and composition. In contract translation, layout-preservation requirements afect document extraction and make layout reconstruction part of the subsequent workflow. Task descriptions and user feedback jointly inform skill adjustments and the selection of execution procedures.

• We introduce an online routing mechanism for long-horizon tasks that selects general-purpose skills and their Adapters according to the execution state, supporting continued execution through incremental updates to the online skill library and preloading for subsequent nodes.

• On ALFWorld, WebShop, and SkillsBench, Skill Adapter achieves higher task success rates than all compared baselines. Compared with SkillRL, it improves success rates by an average of 11.43 percentage points and reduces mean execution time on all three benchmarks. Skill Adapter retains a success-rate advantage in the scaling experiment with 1,000 skills.

## Related Work

## Skill Extraction and Reuse

Voyager generates executable functions from successful trajectories and maintains an expanding skill pool (Wang et al. 2023). ExpeL derives experiential rules from successful and failed trajectories, while Reflexion uses verbal feedback to improve subsequent decisions (Zhao et al. 2023; Shinn et al. 2023). Skill Adapter distills historical execution experience into general-purpose skills and uses associated Adapters to express the local modifications required by diferent users.

## Skill Retrieval and Routing

SkillsBench evaluates the efects of external skills on task execution, while Skills-in-the-Wild studies retrieval and use within large skill collections (Li et al. 2026b; Liu et al. 2026). Graph of Skills uses explicit skill relations to support structured retrieval, SkillDAG organizes dependencies, conflicts, and specialization information through typed relations, and SkillRouter selects skills through retrieval and reranking (Li et al. 2026a; Bai et al. 2026; Zheng et al. 2026).

Skill Adapter focuses on online routing during long-horizon tasks. The system retrieves general-purpose skills based on the current subtask and execution state, then selects applicable Adapters under each skill. Routing results enter a tasklevel online skill library and are updated incrementally as inputs, environmental feedback, and task progress change. Preloading for subsequent nodes and reusing valid entries jointly support the continuous provision of skills.

## Reinforcement Learning and Skill Optimization

Policy optimization methods such as PPO and GRPO support training language models with environmental feedback; GRPO constructs its optimization signal from relative rewards within a group (Schulman et al. 2017; Shao et al. 2024). Among skill-augmented agents, SAGE incorporates skill accumulation into sequential rollouts, while SkillRL constructs a hierarchical skill library from task experience and optimizes the skill-use policy (Wang et al. 2025; Xia et al. 2026). We apply reinforcement learning to personalized modification and online routing. The modification policy receives feedback from task performance before and after skill adjustments, while the routing policy receives feedback from execution outcomes and the quality of skill selection.

## Method

As shown in Figure 2, Skill Adapter implements personalized modification and online routing over a shared pool of generalpurpose skills, using reinforcement learning to optimize both types of decisions. Given a user u and a task $q ,$ the system represents the user’s explicitly stated preferences, operational requirements, and historical feedback through a personalization context $P _ { u } .$ . At step t of a long-horizon task, the available context is

$$
x _ { u , t } = \langle q , P _ { u } , h _ { t } \rangle ,\tag{1}
$$

where $h _ { t }$ contains completed steps, current environment observations, and available intermediate results. The context $P _ { u }$ combines the current request with previously obtained information and is updated as user feedback arrives. Task-specific requirements are used for the current task, while preferences that generalize across tasks are used for subsequent adaptation.

## Personalized Modification

General-Purpose Skills and Execution Evidence The system first constructs a general-purpose skill pool $\mathcal { G }$ by extracting reusable task objectives, operation sequences, tool invocation procedures, and input and output constraints from historical trajectories. Existing skills can also serve directly as the shared foundation. Each general-purpose skill is represented as

$$
s _ { i } = \langle n _ { i } , d _ { i } , c _ { i } , I _ { i } , O _ { i } \rangle ,\tag{2}
$$

where $n _ { i }$ and $d _ { i }$ denote its name and functional description, $c _ { i }$ is its general execution procedure, and $I _ { i }$ and $O _ { i }$ specify input and output constraints. Skills are organized by topic and retain directed composition relations supported by successful trajectories, allowing candidate expansion during routing.

![](images/2e9906614fdc35c1be92307c7e8b408b52291af55614f892d8c7a496d83eb390.jpg)  
Figure 2: The Skill Adapter framework. The system extracts general-purpose skills from historical trajectories and maintains the Adapters associated with each skill based on user requirements and execution feedback. During task execution, the routing agent selects general-purpose skills and applicable Adapters for the current DAG node, incrementally updates the online skill library, and preloads candidates for subsequent nodes. The main LLM reads skills and Adapters as needed, while execution progress and environment feedback inform the next routing round. GRPO optimizes the two policies using the gains from personalized modification and the execution outcomes of online routing, respectively.

The trajectory pool records tasks, personalization contexts, the skills and versions actually invoked, environment observations, and final feedback. A trajectory for user u is represented as

$$
\tau _ { u } = \langle q , P _ { u } , \{ e _ { t } \} _ { t = 1 } ^ { L _ { \tau } } , F , y \rangle ,\tag{3}
$$

where $F$ denotes user feedback or environment diagnostics, and y denotes the task outcome. Each execution step $e _ { t }$ records the general-purpose skill ID, Adapter ID, environment action, and observation. The system distills general procedures that can be reused across tasks from trajectories and generates personalized modifications using execution evidence relevant to the current user’s requirements.

Personalized Adapters For a general-purpose skill $s _ { i } ,$ , an Adapter for user u is defined as

$$
\begin{array} { r l } & { a _ { u , i , k } = \langle i , p _ { u , i , k } , \Delta _ { u , i , k } , \kappa _ { u , i , k } \rangle , } \\ & { \widetilde { s } _ { u , i , k } = \mathrm { A p p l y } ( s _ { i } , \Delta _ { u , i , k } ) . } \end{array}\tag{4}
$$

Here, i identifies the associated general-purpose skill, and k distinguishes the user’s adapted versions for diferent conditions. The term $p _ { u , i , k }$ describes the applicability conditions, $\Delta _ { u , i , k }$ is a local modification to the skill’s execution instructions, and $\kappa _ { u , i , k }$ stores usage records and associated feedback evidence. At execution time, the local modification is applied to the general procedure to obtain the personalized skill $\widetilde { s } _ { u , i , k }$

Local modifications may concern tool selection, parameter settings, the addition or removal of steps, output formats, and verification requirements. For example, to meet a requirement to preserve the layout of a contract, an Adapter can add requirements to extract text positions, page numbers, and reading order to the general-purpose PDF extraction procedure.

Let $\mathcal { A } _ { u }$ denote the set of Adapters for user u. Based on the general-purpose skill, personalization context, and relevant trajectories, the modification policy proposes an action to add, update, or delete an Adapter:

$$
\begin{array} { r l } & { m _ { t } \sim \pi _ { \mathrm { m o d } } ( \cdot \mid s _ { i } , P _ { u } , \mathcal { B } _ { u , i } , \mathcal { A } _ { u } ) , } \\ & { \mathcal { A } _ { u } ^ { \prime } = \mathrm { U p d a t e } ( \mathcal { A } _ { u } , m _ { t } ) . } \end{array}\tag{5}
$$

Here, $B _ { u , i }$ contains historical records relevant to the user’s requirements and skill $s _ { i }$ . The system distills reusable modifications from recurring deviations and applies them according to their applicability conditions. When the user’s requirements change, the corresponding Adapter is updated based on feedback. Adapters and general-purpose skills are stored separately, and the same general-purpose skill can be associated with adapted versions for multiple users.

## Online Routing

Online Selection of Skills and Adapters Online routing continues throughout the execution of a long-horizon task. The main LLM decomposes the task into a directed acyclic graph (DAG) D. The routing agent processes ready nodes according to the current execution progress and prepares candidate skills for subsequent nodes. For task node $q _ { j }$ at step t, the available context is $x _ { u , j , t } = \langle q _ { j } , P _ { u } , h _ { t } \rangle$ . Based on the node’s objective and available intermediate results, the system retrieves a candidate set from the general-purpose skill pool:

$$
\mathcal { R } _ { j , t } = \mathrm { R e t r i e v e } ( q _ { j } , h _ { t } , \mathcal { G } ) .\tag{6}
$$

Retrieval uses topic-based search and semantic matching, followed by candidate expansion along skill composition relations. Each candidate skill is associated with its Adapters. The system filters these Adapters according to the current inputs, tool states, and applicability conditions to construct skill-version pairs:

$$
\begin{array} { r l } { \mathcal { C } _ { u , j , t } = \{ ( s _ { i } , a ) : s _ { i } \in \mathcal { R } _ { j , t } , } & { } \\ { a \in \mathcal { A } _ { u , i } ( x _ { u , j , t } ) \cup \{ \varnothing \} \} . } \end{array}\tag{7}
$$

The symbol $\mathcal { D }$ denotes direct use of the general procedure. Candidate pairs must satisfy skill input requirements, tool availability, and task constraints. The online routing policy selects an ordered sequence of skill-Adapter pairs from the valid candidates:

$$
Z _ { u , j , t } \sim \pi _ { \mathrm { r o u t e } } ( \cdot \mid x _ { u , j , t } , \mathcal { C } _ { u , j , t } ) .\tag{8}
$$

Online routing combines user preferences, historical feedback, and the current execution state to select general-purpose skills and their associated Adapters. When the general procedure satisfies the current requirements, routing can use it directly. As task inputs or environment states change, routing can rematch the skill with an Adapter or select other general-purpose skills and their adapted versions.

Incremental Maintenance of the Online Skill Library Routing results are written to the task-level online skill library $\mathcal { L } _ { u , q , t } .$ Each entry stores a reference to the general-purpose skill, the selected Adapter, applicable nodes, composition relations, and usage guidance. The library is updated as follows:

$$
\begin{array} { r } { \mathcal { L } _ { u , q , t + 1 } = \mathrm { R e f r e s h } ( \mathcal { L } _ { u , q , t } , Z _ { u , j , t } , } \\ { E _ { u , j , t } ^ { \mathrm { c o m p } } , U _ { u , j , t } , x _ { u , j , t } ) . } \end{array}\tag{9}
$$

Here, $E _ { u , j , t } ^ { \mathrm { c o m p } }$ denotes valid input-output connections within the current skill composition, and $\bar { U _ { u , j , t } }$ denotes node-level usage guidance. During initialization, the routing agent creates entries for the first ready nodes in the DAG. The main LLM reads the skills and Adapters required by the current node and executes them, while the routing agent preloads candidates for subsequent nodes.

Based on execution feedback, the system reuses applicable entries, removes or replaces invalid entries, and retrieves skills and reselects Adapters for new requirements. For subsequent nodes that depend on upstream results, the system prepares candidates in advance, then checks applicability conditions and completes selection once the actual inputs are determined. The main LLM returns environment actions, execution observations, and the IDs of the skills and Adapters actually used, providing state information for the next routing round. After the task is completed, its trajectory and feedback support subsequent learning, and user Adapters can be reused in later tasks.

## Reinforcement Learning

Optimization Objective Personalized modification produces skill versions that satisfy specific requirements, while online routing determines the skills and Adapters to use at each step according to the execution state. We use GRPO to train $\pi _ { \mathrm { m o d } }$ and $\pi _ { \mathrm { r o u t e } }$ separately. For the same input, we sample K candidate decisions and compute group-relative advantages from their rewards:

$$
\widehat { A } _ { k } = \frac { R _ { k } - \mathrm { M e a n } ( \{ R _ { l } \} _ { l = 1 } ^ { K } ) } { \mathrm { S t d } ( \{ R _ { l } \} _ { l = 1 } ^ { K } ) + \epsilon } .\tag{10}
$$

Both policies are updated using the following objective:

$$
\begin{array} { c l } { \displaystyle { J ( \theta ) = \mathbb { E } \Bigg [ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \operatorname* { m i n } \Big ( \rho _ { k } ( \theta ) \widehat { A } _ { k } , } } \\ { \displaystyle { \mathrm { c l i p } \big ( \rho _ { k } ( \theta ) , 1 - \epsilon _ { c } , } } \\ { \displaystyle { 1 + \epsilon _ { c } \big ) \widehat { A } _ { k } \Big ) } } \\ { \displaystyle { - \beta _ { \mathrm { K L } } D _ { \mathrm { K L } } \big ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } \big ) \Bigg ] . } } \end{array}\tag{11}
$$

Here, $\rho _ { k } \mathopen { } \mathclose \bgroup \left( \theta \aftergroup \egroup \right)$ is the ratio of the current policy’s probability of a candidate decision to that of the old policy, $\epsilon _ { c }$ is the clipping coeficient, and $\pi _ { \mathrm { r e f } }$ is the reference policy. The main LLM remains frozen during policy training.

Reward for Personalized Modification The modification policy uses a set of trajectories to propose candidate Adapter updates and evaluates their efects on a held-out task set $\mathcal { Q } _ { u } ^ { \mathrm { { e v a l } } }$ The general-purpose skill pool, user conditions, executor, and routing configuration are kept identical before and after modification. The modification reward is defined as the change in task success rate in percentage points:

$$
\begin{array} { r l } & { R ^ { \mathrm { { m o d } } } = 1 0 0 \Big [ \mathrm { S R } \big ( \mathcal { Q } _ { u } ^ { \mathrm { e v a l } } ; \mathcal { G } , \mathcal { A } _ { u } ^ { \prime } \big ) } \\ & { \qquad - \mathrm { S R } \big ( \mathcal { Q } _ { u } ^ { \mathrm { e v a l } } ; \mathcal { G } , \mathcal { A } _ { u } \big ) \Big ] . } \end{array}\tag{12}
$$

The modification policy generates candidates from historical trajectories and evaluates execution outcomes using the requirements and feedback of the held-out tasks. Modifications that increase task success rate receive positive rewards, while those that decrease it receive negative rewards.

Reward for Online Routing After training the modification policy, we train the routing policy with the current generalpurpose skill pool and Adapter set. The routing action $Z _ { u , j , t }$ is provided to the main LLM through the online skill library, and the task outcome $y _ { j }$ is obtained after execution is complete. During training, a reference skill set $S _ { j } ^ { \mathrm { g o l d } }$ measures the quality of general-purpose skill selection. Let

$$
\begin{array} { r } { \begin{array} { l } { S _ { j } = \mathrm { B a s e I D s } ( Z _ { u , j , t } ) , } \\ { m _ { j } = | S _ { j } \cap S _ { j } ^ { \mathrm { g o l d } } | , } \\ { n _ { j } = | S _ { j } | , \qquad k _ { j } = | S _ { j } ^ { \mathrm { g o l d } } | . } \end{array} } \end{array}\tag{13}
$$

The routing reward is

$$
R _ { j } ^ { \mathrm { r o u t e } } = \left\{ { \begin{array} { l l } { \displaystyle { \frac { 2 m _ { j } } { n _ { j } + k _ { j } } } , } & { y _ { j } = \mathrm { s u c c e s s } , } \\ { - 1 , } & { y _ { j } = \mathrm { f a i l u r e } . } \end{array} } \right.\tag{14}
$$

Successful tasks receive the F1 score between the selected general-purpose skills and the reference set, while failed tasks receive a negative reward. When both sets are empty, the F1 score for a successful task is defined as 1. Diferent Adapters share the ID of their parent general-purpose skill, so the efect of version selection enters the reward through the actual task outcome. This objective provides feedback on both task execution and skill selection to learn compact combinations of skills and Adapters suited to the current execution state. At execution time, the routing policy takes the current subtask, execution history, applicable requirements, and available candidates as inputs.

## Experiments

## Experimental Setup

We evaluate Skill Adapter on ALFWorld, WebShop, and SkillsBench to examine how personalized modification and online routing of general-purpose skills afect task success and execution eficiency. We also analyze the efects of skill pool size, the organization of the online skill library, and reinforcement learning for the two policies.

Benchmarks and task conditions. ALFWorld (Shridhar et al. 2021) comprises six types of long-horizon tasks: object placement, cleaning, heating, cooling, examination, and manipulation of multiple objects. WebShop (Yao et al. 2022) requires agents to browse product pages and purchase items that satisfy user-specified attribute constraints. SkillsBench (Li et al. 2026b) evaluates agents’ ability to identify, select, and use external skills. The three test sets contain 420, 1,500, and 210 tasks, respectively, with environment interaction budgets of 50, 15, and 30 steps. Personalization conditions are drawn from the goals, operational constraints, and output requirements explicitly stated in task instructions. Skill construction and policy training use execution experience from the training data.

Baselines. We compare seven configurations. Backbone (ReAct) (Yao et al. 2023) uses the main agent to execute tasks directly. Backbone (all skills) provides the main LLM with the entire skill pool. Backbone + BM25 retrieves skills through text matching (Robertson and Zaragoza 2009). LLM as Selector uses an LLM to select candidate skills. SkillRouter follows a retrieval-and-reranking pipeline (Zheng et al. 2026). Graph of Skills uses explicit skill relationships for structural retrieval (Li et al. 2026a). SkillRL optimizes skill use through reinforcement learning (Xia et al. 2026). Within each benchmark, all methods share the executor, initial general-purpose skill pool, task inputs, and interaction budget.

Metrics. For ALFWorld, we report success rates for the six task types, the overall success rate (SR), and mean task execution time. For WebShop, we report task score, success rate, and mean execution time. For SkillsBench, we report success rate, mean reward, and mean execution time. Execution time includes skill retrieval, modification and routing inference, main-LLM inference, and environment interaction.

Models and skill sources. We use Qwen3-8B (Yang et al. 2025) as the executor for ALFWorld and WebShop, and GPT-5-mini (OpenAI 2025) for SkillsBench. For the first two benchmarks, we cluster training trajectories from AgentTraj-L and use GPT-5.2-Codex to extract general-purpose skills. The initial SkillsBench skill pool is formed by merging the skills provided by the benchmark. The modification and routing policies use Qwen3-8B, with training inputs constructed from ALFWorld training trajectories and skills. The base procedures of general-purpose skills and their Adapters are stored separately, and routing results are supplied to the executor through the task-level online skill library.

Retrieval and training configuration. Candidate retrieval uses BGE-M3 embeddings. We first select three relevant topics and retrieve 20 skills, then expand the candidates along skill composition relationships and cap the candidate set at 30. The routing policy selects at most eight general-purpose skills. Both policies are trained with GRPO, using a group size of 8, a learning rate of $1 \times 1 0 ^ { - 6 }$ , a clipping coeficient of 0.2, a KL coeficient of 0.01, and a global batch size of 64. Each stage comprises 100 optimizer updates. The executor remains frozen during training. We use random seeds 42, 43, and 44, and report the mean results over three runs in the main tables.

## Main Results

Overall Performance Tables 1 and 2 compare performance on the three benchmarks. Skill Adapter achieves the best results on all core metrics. Compared with SkillRL, success rates on ALFWorld, WebShop, and SkillsBench increase from 87.14%, 72.73%, and 21.43% to 95.48%, 81.53%, and 38.57%, respectively, corresponding to improvements of 8.34, 8.80, and 17.14 percentage points.

Skill Adapter achieves the highest success rate on all six ALFWorld task types. Compared with SkillRL, the success rate on Look tasks increases from 69.23% to 89.74%, while gains on the other five task types range from 3.81 to 10.41 percentage points. The WebShop task score increases from 85.20 to 92.67, and the SkillsBench mean reward increases from 25.33 to 34.90.

On all three benchmarks, configurations that select skills achieve higher task success rates than the configuration that provides the main LLM with the entire skill pool.

Online Routing and Execution Eficiency The main LLM reads the skills and Adapters available for the current node from the online skill library, while the routing agent preloads candidates for subsequent nodes along the DAG. When subgoals, intermediate results, or environment feedback change the applicability conditions, routing retains entries that remain applicable and updates the afected skills and their versions.

Compared with SkillRL, Skill Adapter reduces mean execution time from 70.6 to 61.9 seconds on ALFWorld, from 57.8 to 50.7 seconds on WebShop, and from 529.4 to 398.9 seconds on SkillsBench.

<table><tr><td></td><td colspan="8">ALFWorld</td><td colspan="3">WebShop</td></tr><tr><td>Method</td><td>Pick</td><td>Clean</td><td>Heat</td><td>Cool</td><td>Look</td><td>Pick 2</td><td>SR↑</td><td>Time (s) ↓</td><td>Score ↑</td><td>SR↑</td><td>Time (s) ↓</td></tr><tr><td>Backbone (ReAct)</td><td>59.05</td><td>60.49</td><td>39.58</td><td>42.67</td><td>48.72</td><td>56.94</td><td>52.86</td><td>88.6</td><td>46.00</td><td>19.80</td><td>72.4</td></tr><tr><td>Backbone (all skills)</td><td>30.48</td><td>33.33</td><td>20.83</td><td>25.33</td><td>28.21</td><td>23.61</td><td>27.62</td><td>121.5</td><td>24.80</td><td>7.53</td><td>103.7</td></tr><tr><td>Backbone + BM25</td><td>68.57</td><td>61.73</td><td>56.25</td><td>56.00</td><td>53.85</td><td>65.28</td><td>61.67</td><td>81.3</td><td>58.13</td><td>25.27</td><td>66.8</td></tr><tr><td>LLM as Selector</td><td>70.48</td><td>67.90</td><td>60.42</td><td>56.00</td><td>61.54</td><td>68.06</td><td>65.00</td><td>99.7</td><td>61.00</td><td>28.87</td><td>82.5</td></tr><tr><td>SkillRouter</td><td>75.24</td><td>71.60</td><td>66.67</td><td>64.00</td><td>58.97</td><td>73.61</td><td>69.76</td><td>76.8</td><td>68.87</td><td>46.13</td><td>62.4</td></tr><tr><td>Graph of Skills</td><td>73.33</td><td>74.07</td><td>68.75</td><td>72.00</td><td>56.41</td><td>70.83</td><td>70.71</td><td>79.5</td><td>67.93</td><td>46.73</td><td>65.1</td></tr><tr><td>SkillRL</td><td>94.29</td><td>85.19</td><td>85.42</td><td>89.33</td><td>69.23</td><td>87.50</td><td>87.14</td><td>70.6</td><td>85.20</td><td>72.73</td><td>57.8</td></tr><tr><td>Skill Adapter (ours)</td><td>98.10</td><td>95.06</td><td>95.83</td><td>94.67</td><td>89.74</td><td>95.83</td><td>95.48</td><td>61.9</td><td>92.67</td><td>81.53</td><td>50.7</td></tr></table>

Table 1: Performance comparison on ALFWorld and WebShop with Qwen3-8B as the executor. The best results are boldfaced and the second-best results are underlined.

Pick, Clean, Heat, Cool, Look, and Pick 2 denote the six ALFWorld task types. Higher success rates and scores are better; lower mean execution times are better.
<table><tr><td>Method</td><td>SR↑</td><td>Mean reward ↑</td><td>Mean time (s) ↓</td></tr><tr><td>Backbone (ReAct)</td><td>7.14</td><td>9.65</td><td>468.3</td></tr><tr><td>Backbone (all skills)</td><td>9.05</td><td>14.87</td><td>621.4</td></tr><tr><td>Backbone + BM25</td><td>13.33</td><td>17.24</td><td>525.2</td></tr><tr><td>LLM as Selector</td><td>14.76</td><td>18.63</td><td>587.3</td></tr><tr><td>SkillRouter</td><td>17.62</td><td>19.46</td><td>615.0</td></tr><tr><td>Graph of Skills</td><td>18.57</td><td>23.09</td><td>462.9</td></tr><tr><td>SkillRL</td><td>21.43</td><td>25.33</td><td>529.4</td></tr><tr><td>Skill Adapter (ours)</td><td>38.57</td><td>34.90</td><td>398.9</td></tr></table>

Table 2: Performance comparison on SkillsBench with GPT-5-mini as the executor. The best results are boldfaced and the second-best results are underlined.

Scaling the Skill Pool The scaling experiment starts with the same task-relevant skills and gradually adds distractor skills from other task categories. The left panel of Figure 3 reports success rates as the skill pool expands from 50 to 1,000 skills.

Success rates decline for all methods as the skill pool grows. Skill Adapter decreases from approximately 95.5% to 80.4%, while SkillRL decreases from approximately 87.2% to 60.3%. At 1,000 skills, Skill Adapter retains a lead of approximately 20 percentage points. Compared with BM25, Graph of Skills achieves success rates that are 9.04, 21.46, and 5.24 percentage points higher on the three benchmarks, respectively; Skill Adapter further exceeds Graph of Skills by 24.77, 34.80, and 20.00 percentage points.

## Ablation Study

Table 3 analyzes the efects of general-purpose skill representations, the organization of online routing, and training the modification and routing policies.

General-Purpose Skills and Online Routing Organization Adding a general-purpose skill pool to the trajectory pool increases mean success rate from 29.28% to 41.86%, an improvement of 12.58 percentage points. Adding skill composition relationships and usage guidance raises mean success rate to 48.35%. Further adding DAG preloading increases it to 52.75%, a gain of 4.40 percentage points.

![](images/6d3b5dc30bcd3d9b415f56a4dd7f53dc83cb985c3d1c65c43cf6f82a9818e500.jpg)

![](images/6452ade4b3d3089ea70fdb3737ddae19e6f64adaa8b5cb9f366d05f1b1e5e2b1.jpg)  
Figure 3: Skill pool scaling and policy training. Left: success rates at diferent skill pool sizes. Right: changes in the success rates of the two policies over 100 training steps.

<table><tr><td>Group</td><td colspan="4">Variant ALF. ↑ Web. ↑ SB. ↑ Avg. ↑</td></tr><tr><td>Skills</td><td>Trajectory pool only</td><td>52.86</td><td>25.93 9.05</td><td>29.28</td></tr><tr><td>Skills</td><td>+ General-purpose skill pool</td><td>70.24</td><td>41.53 13.81</td><td>41.86</td></tr><tr><td></td><td>Routing + Composition and usage guidance</td><td>76.90</td><td>51.4716.67</td><td>48.35</td></tr><tr><td></td><td>Routing + DAG preloading</td><td>80.71</td><td>57.07 20.48</td><td>52.75</td></tr><tr><td>RL</td><td>Modification policy</td><td>87.86</td><td>68.8023.81</td><td>60.16</td></tr><tr><td>RL</td><td>Routing policy</td><td>90.95</td><td>73.93 30.95</td><td>65.28</td></tr><tr><td>Full</td><td>Skill Adapter</td><td>95.48</td><td>81.53 38.57</td><td>71.86</td></tr></table>

Table 3: Ablation study of general-purpose skills, online routing, and reinforcement learning (RL). ALF., Web., and SB. denote success rates on the three benchmarks, respectively; Avg. is their arithmetic mean.

Training the Modification and Routing Policies The configuration with reinforcement learning for the modification policy achieves a mean success rate of 60.16%, while the configuration with reinforcement learning for the routing policy achieves 65.28%. The full method uses both trained policies and achieves a mean success rate of 71.86%, exceeding training the routing policy alone by 6.58 percentage points.

In the right panel of Figure 3, the success rates of the modification and routing policies increase from 59.5% and 72% to approximately 87% and 95%, respectively.

## Conclusion

Skill Adapter stores users’ specific requirements for tools, operating procedures, and outputs as Adapters linked to general-purpose skills. As a task progresses, the routing agent selects general-purpose skills and applicable Adapters, incrementally maintains an online skill library that the main LLM reads on demand, and preloads candidates for subsequent nodes. GRPO trains the two policies using modification gains and routing outcomes.

On ALFWorld, WebShop, and SkillsBench, Skill Adapter improves success rates over SkillRL by an average of 11.43 percentage points and reduces mean execution time on all three benchmarks. In the skill pool scaling experiment, Skill Adapter achieves a higher success rate than SkillRL in the setting with 1,000 skills.

## References

Ahn, M.; Brohan, A.; Brown, N.; Chebotar, Y.; Cortes, O.; David, B.; Finn, C.; Fu, C.; Gopalakrishnan, K.; Hausman, K.; et al. 2022. Do As I Can, Not As I Say: Grounding Language in Robotic Afordances. arXiv:2204.01691.

Bai, T.; Wan, Z.; Zhou, P.; Yu, X.; Zhao, W.; You, Y.; and Tsang, I. W. 2026. SkillDAG: Self-Evolving Typed Skill Graphs for LLM Skill Selection at Scale. arXiv:2606.03056.

Deng, X.; Gu, Y.; Zheng, B.; Chen, S.; Stevens, S.; Wang, B.; Sun, H.; and Su, Y. 2023. Mind2Web: Towards a Generalist Agent for the Web. arXiv:2306.06070.

Li, D.; Li, Z.; Du, H.; Wu, X.; Gui, S.; Kuang, Y.; and Sun, L. 2026a. Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills. arXiv:2604.05333.

Li, X.; Chen, W.; Liu, Y.; Zheng, S.; Chen, X.; He, Y.; Li, Y.; et al. 2026b. SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks. arXiv:2602.12670.

Liang, Y.; Zhong, R.; Xu, H.; Jiang, C.; Zhong, Y.; Fang, R.; Gu, J.-C.; et al. 2026. SkillNet: Create, Evaluate, and Connect AI Skills. arXiv:2603.04448.

Liu, X.; Yu, H.; Zhang, H.; Xu, Y.; Lei, X.; Lai, H.; Gu, Y.; Ding, H.; Men, K.; Yang, K.; et al. 2023. AgentBench: Evaluating LLMs as Agents. arXiv:2308.03688.

Liu, Y.; Ji, J.; An, L.; Jaakkola, T.; Zhang, Y.; and Chang, S. 2026. How Well Do Agentic Skills Work in the Wild: Benchmarking LLM Skill Usage in Realistic Settings. arXiv:2604.04323.

Lu, Y.; Yu, H.; and Khashabi, D. 2023. GEAR: Augmenting Language Models with Generalizable and Eficient Tool Resolution. arXiv:2307.08775.

OpenAI. 2025. GPT-5 System Card. https://openai.com/ index/gpt-5-system-card/. Accessed July 20, 2026.

Packer, C.; Wooders, S.; Lin, K.; Fang, V.; Patil, S. G.; Stoica, I.; and Gonzalez, J. E. 2023. MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.

Paranjape, B.; Lundberg, S.; Singh, S.; Hajishirzi, H.; Zettlemoyer, L.; and Ribeiro, M. T. 2023. ART: Automatic Multistep Reasoning and Tool-use for Large Language Models. arXiv:2303.09014.

Park, J. S.; O’Brien, J. C.; Cai, C. J.; Morris, M. R.; Liang, P.; and Bernstein, M. S. 2023. Generative Agents: Interactive Simulacra of Human Behavior. In Proceedings ofthe 36th Annual ACM Symposium on User Interface Software and Technology, 1–22.

Patil, S. G.; Zhang, T.; Wang, X.; and Gonzalez, J. E. 2023. Gorilla: Large Language Model Connected with Massive APIs. arXiv:2305.15334.

Qian, C.; Han, C.; Fung, Y. R.; Qin, Y.; Liu, Z.; and Ji, H. 2023. CREATOR: Tool Creation for Disentangling Abstract and Concrete Reasoning of Large Language Models. arXiv:2305.14318.

Qin, Y.; Liang, S.; Ye, Y.; Zhu, K.; Yan, L.; Lu, Y.; Lin, Y.; Cong, X.; Tang, X.; Qian, B.; et al. 2023. ToolLLM: Facilitating Large Language Models to Master 16000+ Realworld APIs. arXiv:2307.16789.

Robertson, S.; and Zaragoza, H. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4): 333–389.

Schick, T.; Dwivedi-Yu, J.; Dessì, R.; Raileanu, R.; Lomeli, M.; Zettlemoyer, L.; Cancedda, N.; and Scialom, T. 2023. Toolformer: Language Models Can Teach Themselves to Use Tools. arXiv:2302.04761.

Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal Policy Optimization Algorithms. arXiv:1707.06347.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; et al. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.

Shen, J.; Zhang, T.; Zhao, X.; and Cheng, H. 2026. Dynamic Skill Lifecycle Management for Agentic Reinforcement Learning. arXiv:2605.10923.

Shinn, N.; Cassano, F.; Berman, E.; Gopinath, A.; Narasimhan, K.; and Yao, S. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning. In Advances in Neural Information Processing Systems.

Shridhar, M.; Yuan, X.; Côté, M.-A.; Bisk, Y.; Trischler, A.; and Hausknecht, M. 2021. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In International Conference on Learning Representations.

Tu, S.; Xu, C.; Zhang, Q.; Zhang, Y.; Lan, X.; Li, L.; Li, D.; and Zhao, D. 2026. Dynamic Dual-Granularity Skill Bank for Agentic RL. arXiv:2603.28716.

Wang, G.; Xie, Y.; Jiang, Y.; Mandlekar, A.; Xiao, C.; Zhu, Y.; Fan, L.; and Anandkumar, A. 2023. Voyager: An Open-Ended Embodied Agent with Large Language Models. arXiv:2305.16291.

Wang, J.; Yan, Q.; Wang, Y.; Tian, Y.; Mishra, S. S.; Xu, Z.; Gandhi, M.; Xu, P.; and Cheong, L. L. 2025. Reinforcement Learning for Self-Improving Agent with Skill Library. arXiv:2512.17102.

Wu, Q.; Bansal, G.; Zhang, J.; Wu, Y.; Li, B.; Zhu, E.; Jiang, L.; Zhang, X.; Zhang, S.; Liu, J.; Awadallah, A. H.; White, R. W.; Burger, D.; and Wang, C. 2023. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. arXiv:2308.08155.

Xia, P.; Chen, J.; Wang, H.; Liu, J.; Zeng, K.; Wang, Y.; Han, S.; et al. 2026. SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning. arXiv:2602.08234.

Xie, T.; Zhang, D.; Chen, J.; Li, X.; Zhao, S.; Cao, R.; Hua, T. J.; Cheng, Z.; Shin, D.; Lei, F.; Liu, Y.; Xu, Y.; Zhou, S.; Savarese, S.; Xiong, C.; Zhong, V.; and Yu, T. 2024. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments. In Advances in Neural Information Processing Systems, volume 37.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; et al. 2025. Qwen3 Technical Report. arXiv:2505.09388.

Yang, J.; Jimenez, C. E.; Wettig, A.; Lieret, K.; Yao, S.; Narasimhan, K.; and Press, O. 2024. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineer-

ing. arXiv:2405.15793.

Yao, S.; Chen, H.; Yang, J.; and Narasimhan, K. 2022. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents. In Advances in Neural Information Processing Systems.

Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; and Cao, Y. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations.

Zhao, A.; Huang, D.; Xu, Q.; Lin, M.; Liu, Y.-J.; and Huang, G. 2023. ExpeL: LLM Agents Are Experiential Learners. arXiv:2308.10144.

Zheng, Y.; Zhang, Z.; Ma, C.; Yu, Y.; Zhu, J.; Dong, B.; and Zhu, H. 2026. SkillRouter: Retrieve-and-Rerank Skill Selection for LLM Agents at Scale. arXiv:2603.22455.

Zhou, S.; Xu, F. F.; Zhu, H.; Zhou, X.; Lo, R.; Sridhar, A.; Cheng, X.; Ou, T.; Bisk, Y.; Fried, D.; Alon, U.; and Neubig, G. 2024. WebArena: A Realistic Web Environment for Building Autonomous Agents. In International Conference on Learning Representations.