# OOPMAS: Object-Oriented Multi-Agent Systems for Query-Level Workflow Generation

Qi Cheng   
Rutgers University   
New Brunswick, NJ   
q.cheng@rutgers.edu   
Yiqun Xie   
University of Maryland   
College Park, MD   
xie@umd.edu   
Shengyu Chen   
University of Pittsburgh   
Pittsburgh, PA   
shc160@pitt.edu   
Xiaowei Jia   
Rutgers University   
New Brunswick, NJ   
xj159@cs.rutgers.edu

Wei Cheng NEC Labs America Princeton, NJ weicheng@nec-labs.com

Haoyu Wang   
NEC Labs America   
Princeton, NJ   
haoyu@nec-labs.com   
Haifeng Chen   
NEC Labs America   
Princeton, NJ   
haifeng@nec-labs.com

## Abstract

Multi-agent systems (MAS) powered by large language models have shown strong performance across code generation, mathematical reasoning, and question answering. However, existing methods for automating MAS design mostly operate at the task level, producing a single fixed workflow per benchmark that is applied uniformly to all queries. This assumption fails under realistic conditions. Query difficulty varies widely within a task, and real-world workloads mix heterogeneous task types. We introduce OOPMAS, a training-free framework that generates both the agent set and the coordination workflow at the granularity of individual queries. Agents are represented as object-oriented class definitions with dedicated roles, tools, and persistent state, and workflows are expressed as executable main functions over these agent objects. A dynamic skill library accumulates structured lessons from execution feedback across optimization rounds, enabling in-context improvement without any gradient updates or fine-tuning. On a mixed-task benchmark of queries spanning code, math, and QA, OOPMAS achieves 89.6% accuracy, outperforming the strongest baseline by 18.1 percentage points. A model-swap study across four LLM backbones shows consistent scaling, reaching 92.4% with the strongest model.

## 1 Introduction

Multi-agent systems (MAS) powered by large language models have emerged as a leading paradigm for solving complex tasks across code generation, mathematical reasoning, question answering, and scientific analysis [Hong et al., 2024, Wu et al., 2023, Wang et al., 2024a, Li et al., 2023] Designing these systems, however, remains a largely manual process [Nori et al., 2023, Wei et al., 2023, Madaan et al., 2023, Wang et al., 2024b]. Each new task requires developers to decide which agents to create, what roles and tools to assign, and how agents should coordinate. This manual effort limits both scalability and adaptability.

![](images/826d4251c7904a497e5b879d0d40f408eeb31a07ac73a2538f263271fe0c81c8.jpg)  
Figure 1: Task-level workflows fail on real-world query streams. Each fixed workflow catches only its native domain; queries from other domains and multi-domain queries (gray) are missed. No single workflow handles the full mixed distribution.

Recent works are moving to automate the generation and optimization of multi-agent workflows [Zhang et al., 2025a, Hu et al., 2025, Zhang et al., 2025b,c]. These methods have shown strong results on individual benchmarks including HumanEval [Chen et al., 2021], GSM8k [Cobbe et al., 2021], MATH [Hendrycks et al., 2021], and HotpotQA [Yang et al., 2018] by optimizing a single agentic workflow for each of the benchmarks. This one workflow per task approach share a common assumption that a single fixed workflow is the optimal solution uniformly to all queries within that benchmark. We argue that this task-level assumption breaks down in realistic deployment settings for two reasons (illustrated in Figure 1). First, within a single task category, the optimal MAS structure varies with query difficulty. An easy coding problem can be solved by a straightforward workflow, while a repository-level coding task benefits from a substantially more complex workflow. A single task-level workflow must commit to one topology and cannot adapt. Second, real-world query streams are heterogeneous mixtures of task types with unpredictable distribution. A workflow optimized for code generation provides no benefit when a math reasoning or question answering query arrives while supporting mixed workloads under the task-level paradigm would require maintaining a separate optimized system per domain plus a routing mechanism. Our argument is supported by the evaluation of these task-level methods on a mixed-task benchmark containing code, math, and QA queries, where the single optimized workflow often performs well on one task but poorly on others.

A second structural limitation for both task-level and query-level methods. Existing approaches search for workflow while holding the agent pool fixed [Zhang et al., 2025b, Hu et al., 2025, Gao et al., 2025]. We argue that constraining the agent set limits the reachable solution space. If no agent in the pool possesses the domain expertise required by a particular query, no workflow in the search space over that pool can produce an optimal solution. The agents themselves must also be generated dynamically.

In this paper, we introduce OOPMAS (Object-Oriented Programming Multi-Agent Systems), a training-free framework that generates both the agent set and the workflow for individual queries. Given a query, a frozen LLM acting as a meta-agent first generates a pool of specialist agents with tailored role prompts, tool bindings, and model configurations, then produces an executable workflow program that orchestrates these agents to solve the query. Agents are represented as object-oriented class definitions, a choice that naturally encapsulates each agent’s role, tools, and persistent state, and that aligns with code patterns prevalent in LLM pretraining data, leading to more reliable generation. The workflow is iteratively refined across multiple optimization rounds using a dynamic skill library that accumulates structured lessons from past successes and failures, enabling in-context improvement without any gradient updates, distillation, or fine-tuning.

We evaluate OOPMAS on two benchmarks. The standard Mixed benchmark draws 120 queries from six standard sources (HumanEval, MBPP, MATH, GSM8K, HotpotQA, DROP); the Hard Mixed benchmark draws 180 competition-level queries (AIME 2025/2026, LiveCodeBench, CodeContests,

GPQA Diamond) with diverse output formats. OOPMAS achieves 89.6% on Mixed (vs. 71.5% for ADAS, +18.1 points) and 90.8% on Hard Mixed (vs. 72.3% for ADAS, +18.5 points). The same gap is preserved despite the much harder problem distribution, showing that per-query generation generalises rather than relying on benchmark-specific memorisation. A model-swap study across four LLM backbones (GPT-4.1 through GPT-5.2) shows that OOPMAS scales consistently with model capability, reaching 92.4% with GPT-5.2 without any system-level changes.

Our contributions are as follows:

1. Problem formulation. We formalise MAS generation as a query-level optimisation problem over a joint agent-and-workflow search space, and argue that both query-level adaptation and dynamic agent generation are necessary for handling heterogeneous task distributions.

2. OOPMAS framework. We propose a training-free, two-phase pipeline consisting of dynamic OOP-style agent generation and iterative workflow optimisation with a structured skill library. The system requires no reinforcement learning, distillation, or fine-tuning and is immediately deployable on arbitrary task distributions.

3. Mixed and Hard Mixed benchmarks. We introduce two benchmarks that test MAS generation under realistic heterogeneous workloads, including a competition-difficulty Hard Mixed benchmark spanning AIME, LiveCodeBench, CodeContests, and GPQA. OOPMAS substantially outperforms existing task-level and query-level baselines across both benchmarks and all domains, with the gap preserved as task difficulty grows.

## 2 Related Work

Multi-agent systems with LLMs. A first wave of LLM multi-agent work hand-designs roles, prompts, and coordination protocols for specific applications. AutoGen [Wu et al., 2023], MetaGPT [Hong et al., 2024], and CAMEL [Li et al., 2023] provide programming abstractions for orchestrating multiple specialised LLM agents and demonstrate strong results on software engineering, research, and embodied tasks. Single-agent prompting techniques such as chain-of-thought [Wei et al., 2023], tree-of-thoughts [Yao et al., 2023a], self-refine [Madaan et al., 2023], multi-persona collaboration [Wang et al., 2024b], ReAct [Yao et al., 2023b], and Voyager [Wang et al., 2023] are special cases of multi-agent design with one or two roles. While effective, these systems require human experts to choose the agent set and the coordination structure for each new task, limiting scalability and adaptability.

Automated MAS generation at the task level. A second wave of work automates parts of the design process. AFlow [Zhang et al., 2025b] formulates workflow optimisation as a search problem over code-represented operator graphs and uses Monte Carlo Tree Search to discover effective workflows from a fixed operator vocabulary. ADAS [Hu et al., 2025] uses a meta-agent that iteratively programmes and archives new agent designs in code, leveraging the Turing-completeness of the search space. GPTSwarm [Zhuge et al.] casts multi-agent construction as an optimisation over computational graphs. MaAS [Zhang et al., 2025a] searches over a multi-agent supernet to identify high-performance subnetworks, and MetaAgent Search [Zhang et al., 2025c] learns a controller that emits the best architecture per benchmark. FlowReasoner [Gao et al., 2025] also operates at this granularity. All of these methods optimise a single workflow per task category and apply it uniformly to every query in that category. We argue (§4) that this assumption breaks under heterogeneous workloads: query difficulty varies within a task and a single workflow cannot serve a mixed query stream that contains code, math, and QA.

Query-level adaptation. Two earlier works move toward query-level adaptation. FlowReasoner [Gao et al., 2025] produces a workflow per query but draws agents from a fixed predefined set; the workflow is therefore restricted to the expressivity of those agents. Self-refine [Madaan et al., 2023] and tree-of-thoughts [Yao et al., 2023a] adapt the inference-time computation per query but use a single underlying agent. OOPMAS combines these two ideas: it generates both the agent set and the workflow per query, and the agents are themselves dynamically synthesised from a meta-agent rather than chosen from a closed library.

Skill libraries and self-improvement. Voyager [Wang et al., 2023] accumulates a library of executable skills across game episodes and reuses them at later episodes. ADAS’s archive plays a similar role at coarse granularity. Our skill library is closer in spirit to Voyager but stores structured failure/fix records rather than executable code, and is consumed by the meta-agent as in-context optimisation memory rather than as a callable library. Crucially, skill construction in OOPMAS is training-free: the skill library improves the meta-agent’s prompt across rounds without any model parameter updates.

![](images/3ec5007222a342cc366fec9da72fe0a966bb18456f6ccd6f4bcf5447b1bf9049.jpg)  
Figure 2: The OOPMAS pipeline. (a) A meta-agent emits a tailored agent set $A _ { q }$ for each query. (b) A second meta-agent emits an executable workflow $W _ { q }$ over $A _ { q } ,$ , conditioned on the skill library K. (c) $W _ { q }$ is run and scored; failures are diagnosed. (d) Each diagnosis distills a reusable skill into K, which is injected into workflow generation in subsequent rounds. Training-free.

Benchmarks for MAS evaluation. Existing MAS-generation methods are typically evaluated on individual benchmarks (HumanEval [Chen et al., 2021], MBPP [Austin et al., 2021], MATH [Hendrycks et al., 2021], GSM8K [Cobbe et al., 2021], HotpotQA [Yang et al., 2018], DROP [Dua et al., 2019]). We argue this single-benchmark evaluation does not test the realistic deployment setting. The Mixed and Hard Mixed benchmarks introduced in this paper include heterogeneous query types and output formats per run, exposing failure modes of task-level methods that single-benchmark evaluation hides. The Hard Mixed benchmark additionally tests competition-level difficulty (AIME, LiveCodeBench [], CodeContests [Li et al., 2022], GPQA [Rein et al., 2023]) that has not yet saturated, providing headroom to distinguish methods.

## 3 Method

In this section, we first formalize the query-level MAS generation problem in section 3.1, then we describe the two core phases of OOPMAS: dynamic agent generation and workflow generation and optimization. Figure 2 summarises the pipeline.

## 3.1 Problem Formulation

Multi-Agent System. We define a multi-agent system (MAS) for a query q as a tuple $S _ { q } =$ $( \mathcal { A } _ { q } , \mathcal { W } _ { q } )$ , where $\mathcal { A } _ { q } = \{ a _ { 1 } , \ldots , a _ { n } \}$ is a set of agents and ${ \mathcal { W } } _ { q }$ is a workflow that coordinates them. Each agent $a _ { i } = ( p _ { i } , { \mathcal { T } } _ { i } , m _ { i } ( h _ { i } ) , s _ { i } )$ is characterized by a role prompt $p _ { i }$ specifying its role and expertise, a tool set $\mathcal { T } _ { i }$ defining its action space, a backbone model $m _ { i }$ with hyperparameter such as temperature $h _ { i }$ , and states $s _ { i }$ that contains conversation history and intermediate results. The workflow ${ \mathcal { W } } _ { q }$ is an executable program over $A _ { q }$

Query-Level Workflow Optimization. Given a query q drawn from an arbitrary distribution Q and a task-specific evaluation function Score(·), the MAS generation problem is to find:

$$
S _ { q } ^ { * } = \arg \operatorname* { m a x } _ { S _ { q } \in S } \mathrm { ~ S c o r e } ( S _ { q } , q ) ,\tag{1}
$$

where $S = \{ ( \mathcal { A } , \mathcal { W } ) : \mathcal { A } \subseteq \mathcal { A } ^ { * }$ , W over $\mathcal { A } \}$ is the space of all valid multi-agent systems and $\ b { A } ^ { * }$ is the space of all possible agent configurations. This is a per-query optimization: the system $S _ { q } ^ { * }$ is generated fresh for each $q ,$ allowing both the agent composition and the workflow topology to adapt to the specific demands of the query.

## 3.2 Dynamic Agent Generation

We represent both agents and workflows as executable Python code: agents follow an object-oriented programming (OOP) convention and the corresponding workflow is a main function that imports agent classes.

OOP Agent Representation. Each agent in OOPMAS is a self-contained Python object whose attributes encode the role prompt $p _ { i } ,$ model configuration $m _ { i } ( h _ { i } )$ , and persistent state $s _ { i } ,$ and whose methods expose the tool set $\mathcal { T } _ { i }$ and predefined tasks. This design lets an agent accumulate conversation history and intermediate results in $s _ { i }$ across multiple interactions inside a single workflow execution, going beyond a single LLM API call. We choose $\mathrm { O O P }$ for three reasons: it naturally encapsulates per-agent state without external bookkeeping; it lets inter-agent interactions be expressed as flexible Python object interactions (passing objects as arguments, reading exposed state, calling each other’s methods) rather than being constrained to a fixed workflow schema; and $\mathrm { O O P }$ code is well represented in LLM pretraining corpora, making workflow code generation more reliable than custom DSLs or graph schemas.

Agent Pool Generation. Given validation queries $\{ q _ { 1 } , \dots , q _ { N } \}$ from a task distribution, the metaagent $\mathcal { M } _ { \mathrm { a g e n t } }$ generates a shared agent pool A<sup>¯</sup> in a single call. The pool is regenerated per task set; unlike methods that rely on a fixed operator set [Zhang et al., 2025b], the roles and expertise in $\bar { A }$ are tailored to the query distribution.

Per-Query Coverage Verification. Although $\bar { A }$ targets common capabilities across queries, individual queries may need specialised expertise absent from the pool. For each $q _ { i }$ we run a coverage check: the system inspects $\bar { A }$ together with $q _ { i } ,$ and if the pool lacks a required capability it generates additional specialist agents and merges them into a query-specific set $\mathcal { A } _ { q _ { i } }$ . This two-tier process balances pool reuse with per-query adaptability.

## 3.3 Workflow Generation and Optimisation

Given the agent set $\mathcal { A } _ { q } , \mathrm { O O P M A S }$ generates a workflow ${ \mathcal { W } } _ { q }$ and iteratively refines it over R optimisation rounds. Rather than training the meta-agent via RL or distillation, we keep the metaagent fixed and pursue cross-query improvement through a dynamic skill library $\kappa$ that accumulates structured experience from execution feedback. The pipeline is therefore training-free: no gradient updates, distillation, or fine-tuning are required, yet workflow generation improves across rounds.

Workflow generation and execution. In each round, $\mathcal { M } _ { \mathrm { w o r k f l o w } }$ receives $q , A _ { q } .$ , and $\kappa ,$ , and generates an executable main function that imports agent classes, calls their interfaces, invokes tool functions, and composes outputs into a final answer. Because the workflow is unrestricted Python code over OOP agent objects, the meta-agent can express arbitrary coordination patterns without being limited to predefined operators. A lightweight static check rejects programmes that are syntactically invalid, lack an entry point, or never call agent.run; the meta-agent regenerates with the error messages, up to K repair attempts. The validated workflow runs in a standalone subprocess and is scored by a task-specific function Score.

Optimisation via the skill library. After each round we classify outcomes into workflow crash, wrong output, or success. For crashes and wrong outputs, the meta-agent receives the failing workflow, the error trace, and $\kappa ,$ , and produces a structured diagnosis (root cause, fix summary, fix reasoning) followed by a corrected workflow that is re-executed. The skill library $\kappa$ is a structured store of FailureRecord and FixRecord entries, organised into four prompt-injectable sections: round-level score trajectories, crash patterns by error type, successful root-cause/fix pairs, and failed fix attempts. The full skill schema and a worked example are given in Appendix C. Algorithm 1 summarises the full pipeline; Phase 2 runs all queries in parallel and accumulates experience across rounds.

Algorithm 1 OOPMAS: Training-Free Query-Level MAS Generation   
Require: Queries $\{ q _ { 1 } , \dots , q _ { N } \}$ , evaluation function Score, rounds R   
Ensure: Optimized workflows $\{ \mathcal { W } _ { q _ { i } } ^ { * } \} _ { i = 1 } ^ { N }$   
1: Phase 1: Agent Generation   
2: $\bar { \mathcal { A } }  \mathcal { M } _ { \mathrm { a g e n t } } \big ( \{ q _ { 1 } , \dotsc , q _ { N } \} \big )$ ▷ Pooled generation (single LLM call)   
3: for each query $q _ { i }$ do   
4: $\mathcal { A } _ { q _ { i } } \gets \dot { \mathbf { C } } \dot { \mathbf { O V E R A G E C H E C K } } ( \bar { \mathcal { A } } , q _ { i } )$ ▷ Augment if needed   
5: end for   
6: Phase 2: Iterative Workflow Optimization   
7: Initialize skill library $\kappa  \emptyset$   
8: for round $r = 1$ to R do   
9: for each query $q _ { i }$ in parallel do   
10: ${ \mathcal { W } } _ { q _ { i } } ^ { 1 \ldots N ^ { \star } } {  } \bar { { \mathcal { M } } } _ { \mathrm { w o r k f l o w } } ( q _ { i } , { \mathcal { A } } _ { q _ { i } } , { \mathcal { K } } )$ ▷ Generate workflow   
11: $s _ { i } ^ { 1 . . . N } \gets \mathrm { S c o r e } ( \mathcal { W } _ { q _ { i } } ^ { 1 . . . N } , q _ { i } )$ ▷ Execute and evaluate   
12: for $w _ { q _ { i } } ^ { j } \in \mathcal { W } _ { q _ { i } } ^ { 1 \dots N }$ do   
13: if $s _ { i } ^ { j } \in \mathfrak { s } _ { i } ^ { 1 \dots N } < \theta$ then   
14: $\dot { \mathcal { W } } _ { q _ { i } } ^ { \prime }  \operatorname { D I A G N O S E } \& \operatorname { F I X } ( \mathcal { W } _ { q _ { i } }$ , error $_ i , \ d \mathcal { K } )$   
15: $s _ { i } ^ { \prime } \gets \mathrm { S c o r e } ( \mathcal { W } _ { q _ { i } } ^ { \prime } , q _ { i } )$   
16: end if   
17: end for   
18: end for   
19: $\kappa  \kappa \cup$ {failures, fixes, scores from round r}   
20: end for

## 4 Experiments

We evaluate OOPMAS on four experimental settings: (i) a mixed-task benchmark of standard tasks (§4.2); (ii) a hard mixed-task benchmark of competition-level tasks (§4.3) to test generalisation as task difficulty increases; (iii) a component ablation that isolates the contributions of the agent pool, skill library, and optimisation memory (§4.4); and (iv) a model-swap study together with a cost–accuracy comparison against baselines (§4.5, §4.6). All experiments use GPT-5 unless stated otherwise.

## 4.1 Experimental Setup

Mixed-Task Benchmark. Prior automated MAS-generation methods evaluate each task type separately: AFlow [Zhang et al., 2025b] and ADAS [Hu et al., 2025] run a separate optimisation per benchmark for HumanEval [Chen et al., 2021], MBPP [Austin et al., 2021], MATH [Hendrycks et al., 2021], GSM8K [Cobbe et al., 2021], HotpotQA [Yang et al., 2018], and DROP [Dua et al., 2019], producing one optimised workflow per benchmark. This task-specific generation does not reflect real-world deployment: in practice, users submit queries from a mixture of task types with unpredictable proportions, and the MAS-generation system must handle them with a single pipeline. To evaluate this setting, we construct a mixed-task benchmark of 120 queries: 40 code (HumanEval, MBPP), 40 math (MATH, GSM8K), and 40 QA (HotpotQA, DROP). All samples are drawn from the validation splits.

Hard Mixed-Task Benchmark. The standard mixed benchmark is now largely saturated by strong frontier LLMs, so accuracy alone no longer separates methods. To stress-test query-level adaptation under realistic difficulty, we additionally construct a hard mixed-task benchmark of 180 queries drawn from competition-level sources: 60 math from AIME 2025/2026, 60 code from LiveCodeBench [Jain et al., 2024] and CodeContests [Li et al., 2022], and 60 QA from GPQA Diamond [Rein et al., 2023]. Each domain has heterogeneous formats (integer answers for AIME, full stdin/stdout programs for CodeContests, four-way multiple choice for GPQA), forcing the MAS-generation system to produce specialised workflows per query rather than a single template.

Baselines. We compare against four automated MAS-generation methods: (1) AFlow [Zhang et al., 2025b], MCTS-based optimisation of code-represented workflows over a fixed operator vocabulary; (2) ADAS [Hu et al., 2025], a meta-agent that iteratively programmes and archives agent designs;

Table 1: Accuracy (%) on the Mixed and Hard Mixed benchmarks. All methods use GPT-5 as the worker model and receive the same mixed batch of queries. Best results in bold.
<table><tr><td></td><td colspan="4">Mixed</td><td colspan="4">Hard Mixed</td></tr><tr><td>Method</td><td>Code</td><td>Math</td><td>QA</td><td>Overall</td><td>Code</td><td>Math</td><td>QA</td><td>Overall</td></tr><tr><td>ADAS</td><td>50.0</td><td>82.5</td><td>82.0</td><td>71.5</td><td></td><td></td><td></td><td>72.3</td></tr><tr><td>MetaAgent</td><td>50.0</td><td>80.0</td><td>80.7</td><td>70.2</td><td>64.8</td><td>45.0</td><td>58.4</td><td>61.8</td></tr><tr><td>MaAS†</td><td>2.5</td><td>77.5</td><td></td><td>40.0</td><td></td><td></td><td></td><td></td></tr><tr><td>AFlow</td><td>90.0</td><td>0.0</td><td>0.0</td><td>30.0</td><td>0.0</td><td>20.0</td><td>36.5</td><td>8.9</td></tr><tr><td>OOPMAS (ours)</td><td>97.5</td><td>85.0</td><td>86.4</td><td>89.6</td><td>74.5</td><td>98.3</td><td>98.3</td><td>90.8</td></tr></table>

(3) MaAS [Zhang et al., 2025a], search over a multi-agent supernet; and (4) MetaAgent [Zhang et al., 2025c]. For all baselines, the system receives the same mixed batch and must produce workflows that handle every query type, matching the setting OOPMAS operates in.

Evaluation Metrics. We use pass@1 for code (HumanEval, MBPP, LiveCodeBench, CodeContests), solve rate for math (MATH, GSM8K, AIME), and F1 for QA (HotpotQA, DROP), and accuracy for multiple-choice (GPQA). We report per-domain accuracy averaged over the sub-benchmarks within each domain, and an Overall column averaged across all queries.

Implementation Details. Unless stated otherwise, all methods use GPT-5 as both the meta-agent and the worker model. OOPMAS runs for R = 5 optimisation rounds on the mixed benchmark and R = 3 on the hard mixed benchmark, with concurrency 30 and at most 7 agents per query. We report the highest score across the two configurations of OOPMAS (with the dynamic skill library on or off); other hyperparameters are reported in Appendix A.

## 4.2 Mixed-Task Benchmark Results

Table 1 reports results on the standard mixed-task benchmark. OOPMAS reaches 89.6% overall accuracy and outperforms every baseline by a wide margin: the best baseline, ADAS, reaches 71.5%, an absolute gap of 18.1 percentage points. The advantage is largest on code generation (+26.0 points over ADAS) and persists across math and QA. Task-level baselines exhibit characteristic failure modes on heterogeneous distributions. AFlow’s MCTS optimisation discovers a single workflow specialised for one domain; that workflow scores 90.0 on code but 0.0 on math and QA, collapsing the overall score to 30.0. MaAS exhibits the same pathology in reverse, achieving reasonable math accuracy but near-zero on code. ADAS and MetaAgent are more balanced because their archives include heterogeneous solutions, but they still under-perform OOPMAS by a wide margin. OOPMAS’s per-query generation tailors both the agent set and the workflow topology to the specific demands of each query, so simple problems receive a single-expert call while harder ones receive multi-agent pipelines with planners, reviewers, and iterative testing loops.

## 4.3 Hard Mixed-Task Benchmark

Hard Mixed (right half of Table 1) is substantially more challenging: it pairs competition-level math (AIME) with stdin/stdout competitive programming (LiveCodeBench, CodeContests) and graduatelevel science MCQ (GPQA). Despite the increased difficulty, OOPMAS reaches 90.8% overall, again leading the strongest baseline (ADAS, 72.3%) by 18.5 percentage points – almost identical to the gap on the easier mixed benchmark. The per-query agent and workflow generation thus generalises to harder problem distributions.

The baseline failure modes on Hard Mixed are informative. AFlow’s single MCTS-discovered work flow assumed HumanEval-style function-completion outputs and is incompatible with stdin/stdout competitive programming, collapsing to 0% on code. MetaAgent’s code accuracy improves slightly (50 → 64.8) but math drops sharply (80 → 45) because the AIME integer-answer format is out of distribution. ADAS is the most consistent baseline (71.5 → 72.3) but still trails by 18.5 points. These patterns reinforce the central argument: a single fixed workflow cannot accommodate heterogeneous output formats and difficulty levels in realistic mixed workloads, while query-level generation handles them naturally. OOPMAS’s edge widens on math and QA, where per-query workflows exploit problem structure (chain-of-specialists for AIME, multi-perspective consensus for ambiguous GPQA items); on code the lead is smaller because failure modes (off-by-one, edge cases) are orthogonal to workflow topology.

![](images/d12d64fb1a7796863f7860419673e94bdaa384ab3bd5368f834820f0c6133f61.jpg)  
Figure 3: Component ablation on Mixed. Removing each component individually leaves overall accuracy within ±3.5 points of Full (dotted red line). The agent pool and per-query generation are the dominant accuracy drivers; skill-library and memory components contribute marginal additional points on the saturated Mixed-validation set.

## 4.4 Component Ablation

We ablate the four post-pool components of OOPMAS on the Mixed validation set and report results in Figure 3. Each ablation removes exactly one component while keeping the others enabled. Full corresponds to the median of four random-seed reruns of the complete system. Removing the optimisation memory (−mem.), skill creation (−skill\_init), skill updates (−skill\_upd), skill selection (−skill\_sel), all three skill components together (−all skills), or per-query coverage verification (−cov.) leaves overall accuracy within a ±3.5-point band of the Full configuration. Code accuracy is the most stable across ablations, ranging from 93.7 to 100.0. The key takeaway is robustness: the OOP agent pool generated for the query distribution already captures most of the headroom on Mixed, and the optimisation components matter most when the validation distribution is small or when test problems differ from validation problems. Crucially, every ablation still beats every baseline in Table 1 by at least 20 points.

## 4.5 Model Swap Study

We re-run OOPMAS and AFlow on the Mixed benchmark with four different worker models – GPT-4.1, GPT-5-nano, GPT-5, and GPT-5.2 – regenerating agents and workflows from scratch each time. Figure 4 shows the per-domain breakdown. OOPMAS consistently outperforms AFlow on GPT-4.1, GPT-5, and GPT-5.2, and scales smoothly with model capability (72.5% → 92.4% from GPT-4.1 to GPT-5.2). The exception is GPT-5-nano, where AFlow’s fixed MCTS workflow (65.4%) outperforms OOPMAS (51.0%). With a weak meta-agent, the per-query agent code is too unreliable; AFlow’s pre-discovered scaffold is more robust because it does not require online code synthesis. As the worker model strengthens, dynamic generation rapidly dominates, exceeding the fixed workflow by ≈ 27 points at GPT-5.2.

## 4.6 Cost–Accuracy Trade-off

Figure 5 plots accuracy against total dollar cost. On Mixed, OOPMAS uses \$35 to reach 89.6% vs. \$15 for ADAS at 71.5% – a ∼ 7× premium for +18 accuracy points. On Hard Mixed, OOPMAS at \$271 reaches 90.8% vs. ADAS at \$25 for 72.3%. The cost premium buys robustness (OOPMAS is the only method that maintains accuracy as difficulty grows) rather than marginal accuracy, and the per-query nature of the cost makes selective deployment easy: route cheap queries to a singleexpert workflow and expensive queries to the full pipeline. The framework remains training-free, so deploying on a new domain only requires changing the validation queries. A per-stage cost decomposition is in Appendix B.

![](images/ca1e323e5edc5cba2ec586a0597a0bc43d93860884fa0fc44e4e9bffd6ac2d54.jpg)

![](images/c6c2759d61a0514bc9658bb98e71ddee53096c433b46e46caf7c8b318ae26b32.jpg)  
Figure 4: Model-swap study on Mixed. OOPMAS scales consistently with the worker model’s capability (51.0 → 92.4 from GPT-5-nano to GPT-5.2). At weak models (GPT-5-nano), AFlow’s fixed MCTS scaffold is more robust because it does not require the meta-agent to author high-quality agent code; at stronger models, dynamic generation dominates.  
Figure 5: Accuracy versus cost on Mixed (left) and Hard Mixed (right). OOPMAS sits in the upper-right of both panels. Costs cover the full pipeline; baseline costs are estimates derived from the configuration used in Table 1.

## 5 Conclusion

## 6 Conclusion

We presented OOPMAS, a training-free framework for query-level multi-agent system generation. Unlike prior automated MAS design methods that optimize a single fixed workflow for an entire task or benchmark, OOPMAS treats each query as its own system-design problem. For every query, it jointly synthesises a tailored set of object-oriented agents and an executable coordination workflow, then refines the workflow over R optimization rounds using execution feedback and a dynamic skill library. This design expands the search space from workflow-only optimization over a fixed agent pool to joint agent-and-workflow generation, allowing the system to adapt both the expertise available to the workflow and the structure of the workflow itself.

Limitation. Several limitations remain. First, OOPMAS depends on a strong meta-agent. When the backbone is weak, the system may fail to synthesize reliable agent classes or executable workflows, and fixed scaffolds such as AFlow can be more robust. Second, the current skill library is run-local: it improves workflow generation within an optimization run but does not yet persist across deployments, domains, or future query streams. Third, per-query generation increases inference cost substantially compared with fixed-workflow baselines, especially on hard problems that require multiple repair attempts. Finally, while Mixed and Hard Mixed cover diverse domains and formats, they remain benchmark-based evaluations; broader deployment settings may introduce tool-use, long-context, safety, privacy, and distribution-shift challenges not fully captured here.

## References

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/2107.03374.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proc. ofNAACL, 2019.

Hongcheng Gao, Yue Liu, Yufei He, Longxu Dou, Chao Du, Zhijie Deng, Bryan Hooi, Min Lin, and Tianyu Pang. Flowreasoner: Reinforcing query-level meta-agents, 2025. URL https: //arxiv.org/abs/2504.15257.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. URL https://arxiv.org/abs/2103.03874.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. MetaGPT: Meta programming for a multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=VtmBAGCN7o.

Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems, 2025. URL https: //arxiv.org/abs/2408.08435.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. LiveCodeBench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for "mind" exploration of large language model society, 2023. URL https://arxiv.org/abs/2303.17760.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code generation with alphacode. Science, 378(6624):1092–1097, December 2022. ISSN 1095-9203. doi: 10.1126/science.abq1158. URL http://dx.doi.org/10.1126/science.abq1158.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback, 2023. URL https://arxiv.org/abs/2303.17651.

Harsha Nori, Yin Tat Lee, Sheng Zhang, Dean Carignan, Richard Edgar, Nicolo Fusi, Nicholas King, Jonathan Larson, Yuanzhi Li, Weishung Liu, Renqian Luo, Scott Mayer McKinney, Robert Osazuwa Ness, Hoifung Poon, Tao Qin, Naoto Usuyama, Chris White, and Eric Horvitz. Can generalist foundation models outcompete special-purpose tuning? case study in medicine, 2023. URL https://arxiv.org/abs/2311.16452.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark, 2023. URL https://arxiv.org/abs/2311.12022.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models, 2023. URL https://arxiv.org/abs/2305.16291.

Yuxia Wang, Minghan Wang, Muhammad Arslan Manzoor, Fei Liu, Georgi Georgiev, Rocktim Jyoti Das, and Preslav Nakov. Factuality of large language models: A survey, 2024a. URL https: //arxiv.org/abs/2402.02420.

Zhenhailong Wang, Shaoguang Mao, Wenshan Wu, Tao Ge, Furu Wei, and Heng Ji. Unleashing the emergent cognitive synergy in large language models: A task-solving agent through multi-persona self-collaboration, 2024b. URL https://arxiv.org/abs/2307.05300.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. URL https://arxiv.org/abs/2201.11903.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, and Chi Wang. Autogen: Enabling next-gen llm applications via multi-agent conversation, 2023. URL https://arxiv.org/abs/2308.08155.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models, 2023a. URL https://arxiv.org/abs/2305.10601.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023b. URL https://arxiv.org/ abs/2210.03629.

Guibin Zhang, Luyang Niu, Junfeng Fang, Kun Wang, Lei Bai, and Xiang Wang. Multi-agent architecture search via agentic supernet, 2025a. URL https://arxiv.org/abs/2502.04180.

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. Aflow: Automating agentic workflow generation, 2025b. URL https://arxiv.org/abs/ 2410.10762.

Yaolun Zhang, Xiaogeng Liu, and Chaowei Xiao. Metaagent: Automatically constructing multi-agent systems based on finite state machines, 2025c. URL https://arxiv.org/abs/2507.22606.

Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, and Jürgen Schmidhuber. Gptswarm: Language agents as optimizable graphs. In Forty-first International Conference on Machine Learning.

## Technical Appendices

## A Reproducibility Details

Hardware and runtime. Experiments were run on a single workstation with 48 CPU threads; all LLM calls are offloaded to the OpenAI API. A typical Mixed run completes in 30–60 minutes; a typical Hard Mixed run takes 1.5–3 hours. Wall-clock is dominated by API latency rather than local compute.

Evaluation specifics. For HumanEval and MBPP we treat a query as solved iff every assertion in the dataset’s test list passes, evaluated in an isolated subprocess. For MATH and GSM8K we use the standard boxed answer extraction. For HotpotQA and DROP we use F1 on the predicted answer. For AIME we extract the final integer in the range [0, 999] and compare to ground truth. For LiveCodeBench and CodeContests we run the predicted programme on every public and private test case and require all to pass. For GPQA we extract the final letter ∈ {A, B, C, D}.

## B Per-Stage Cost Breakdown

OOPMAS Token Usage Across Models (Mixed Benchmark)

![](images/89b8b88baa6691deec93964e55e324fa2f59e7b5b4f97e6cf977648fa39f151c.jpg)  
Figure 6: Token usage by OOPMAS on the Mixed benchmark, decomposed into Optimisation (workflow generation, skill update, repair) and Execution (per-agent inference). Total tokens scale roughly linearly with model size up to GPT-5; GPT-5.2 uses fewer optimisation tokens than GPT-5 because its workflows succeed earlier and require less repair.

Figure 6 reports the total token usage for OOPMAS across four backbone models. The Optimisation phase dominates (workflow generation, repair, skill creation), while Execution (per-agent calls inside the workflow) is small in comparison because most queries terminate within a single agent call. A GPT-5 run on Mixed costs approximately \$35 at current public API pricing, or \$0.30 per query. On Hard Mixed, the per-query cost rises to \$1.50 because problems are longer and require more repair attempts. Within a single Mixed run, the per-stage decomposition is approximately: agent generation 0.3%, skill creation/update 1.0%, skill selection 4.5%, workflow generation 58%, workflow execution 36% (numbers from a representative GPT-5 run; full breakdowns are produced by the codebase under each run’s cost\_summary.json).

## C Skill Library Examples

The skill library is a structured store of FailureRecord and FixRecord entries serialised between rounds. A single skill manifest entry has the schema:

```json
{
"skill_id": "code_generate_test_refine",
"name": "Code: Generate-Test-Refine Loop",
"applicable_domains": ["code"],
"when_to_use": "Code-generation problems with verifiable test cases.",
"strategy": "1. Pass the full task to a code-writer agent.
2. Run the function’s tests via execute_code.
3. On failure, pass error+code to a code-refiner agent.
4. Print the final code between markers.",
"detection_keywords": ["writer.run", "execute_code", "CODE_START"]
}
```

After each optimisation round, the meta-agent reviews up to 30 failure traces and 10 successful fixes from the round and produces an updated manifest. We constrain each skill to a single domain to prevent the workflow generator from importing cross-domain agents into a single workflow (e.g., importing a math agent into a code workflow), which is a major source of contamination in early prototypes.