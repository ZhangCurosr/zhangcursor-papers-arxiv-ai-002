# RITE: EVALUATING END-TO-END RESILIENCE TO SILENT TOOL ERRORS IN E-COMMERCE AGENTIC SYSTEMS

Wenshuo Zhang<sup>1</sup>, Yuquan Lin<sup>1</sup>, Yushi Sun<sup>1</sup>, Yishi Lin<sup>2</sup>, Rui Sheng<sup>1</sup>, Peng He<sup>2</sup>, Han Gao<sup>2</sup>, Jikai Wang<sup>2,†</sup>, Yifeng Rong<sup>2,†</sup>

<sup>1</sup>The Hong Kong University of Science and Technology <sup>2</sup>Tencent wzhangeb@connect.ust.hk, {crazyrong, teddyjkwang}@tencent.com <sup>†</sup>Co-corresponding authors.

## ABSTRACT

Tool-using LLM agents increasingly answer e-commerce queries through multistep tool interactions. The expected output of a tool call can shift between agent development and execution, allowing a plausible but incorrect value to be returned without an explicit failure signal. These silent errors may propagate through subsequent decisions into final answers. To study agents’ resilience across this endto-end process, we formulate Silent-error Resilience Task, an interactive task that evaluates whether they complete the original query after a silent error corrupts an intermediate state. We introduce RITE, a benchmark built from human-authored industrial workflows and production-log-derived inputs. To enable controlled and reproducible evaluation, we use a semantic error taxonomy to inject errors into fixed workflow prefixes, freeze recorded tool behavior, and let agents continue freely. It contains 167 matched units across nine e-commerce workflows, yielding 501 silent-error, correct-value, and explicit-failure instances. Across 10 recent LLMs, silent errors roughly halve task success and are consistently harder than explicit failures. Recognition, use of an available verification path, and integration of the resulting evidence form successive bottlenecks. We also propose a light State Checker as an intervention to test whether resilience can be improved.

## 1 INTRODUCTION

Agentic LLM systems increasingly complete e-commerce tasks through multi-step workflows that invoke multiple external tools and pass their outputs from one step to the next. In industrial deployments, a tool’s runtime output may differ from what the agent was built to expect because underlying business data, tool implementations, or deployed versions can change between development and execution. This build-time–runtime mismatch can remain hidden: a call may complete and conform to the expected API schema while returning a plausible value that is incorrect for the current task. We refer to such plausible-but-wrong intermediate values without an explicit failure signal as silent errors. For example, suppose a user mainly watches local football videos, one of which mentions blockchain-based fan tokens. During agent development, the profile tool returns “local sports”; after an update that overweights “blockchain,” it returns “blockchain” at execution. The new label remains plausible and schema-valid, so no failure is signaled. The absence of an explicit failure signal makes silent errors particularly severe: an agent may have no immediate basis for detecting that the value is incorrect. It may therefore propagate the corrupted value through subsequent tool calls, allowing a single intermediate error to directly alter the final user-facing output.

Existing benchmarks evaluate tool use in nominal settings (Qin et al., 2024; Trivedi et al., 2024; Ma et al., 2024), recovery from explicit or implicit faults injected while agents run (Zhu et al., 2026; Tian et al., 2026; Chen et al., 2026), silent-error detection (Sun et al., 2024; Mazumder & jahan Lia, 2026), or how corrupted intermediate values affect later outputs (Gurram, 2026). In these faultbased settings, each agent runs freely and meets a fault only if and when its own trajectory reaches the perturbed call, so different agents face silent errors at different points and with different histories.

They therefore cannot compare different agents on the same silent error at the same decision point, nor pair that error with correct-value and explicit-failure counterparts at that point.

To address this gap, we formulate Silent-error Resilience Task, in which a plausible but incorrect tool result enters an agent’s execution history without an explicit failure signal. Given the query and toolbox, the agent operates in an agentic loop, repeatedly reasoning, calling tools that may return silent errors, and incorporating their outputs until it produces a final answer. The task asks whether the agent can recognize the error, reacquire reliable evidence, and complete the original query instead of propagating the error into its final answer.

To evaluate this task under controlled conditions, we introduce RITE (Resilience to Incorrect Too Evidence). Each human-authored workflow specifies a pre-orchestrated agentic process, and its execution log records tool calls and observations in the same format as an agent trajectory. We use this log as a fixed trajectory, allowing us to control where a silent error is introduced. We select a semantically important slot in the log and inject a silent error at that point. The trajectory up to and including this slot is provided as the agent’s prefix, while the workflow’s tools and their recorded input–output behavior are frozen to form the evaluation world, so that recorded calls return identical outputs across models and conditions. The agent then continues in an agentic loop, allowing us to compare its behavior while controlling for error type, slot, and environment.

Following this protocol, we first construct a semantic taxonomy covering errors in object or category selection and in attributes or values. We sample inputs from the production logs of one internal platform and use them to run workflows deployed on a separate internal platform. From the resulting traces, we select task-relevant checkpoints and instantiate taxonomy-guided plausible but incorrect values. Each resulting unit contains matched prefixes for the silent-error, correct-value, and explicitfailure conditions, all paired with the same evaluation world. RITE contains 167 matched units from nine e-commerce workflows, yielding 501 instances.

We evaluate 10 recent closed and open-weight LLMs on RITE. Across all models, silent errors reduce task success to roughly half of correct-condition performance and are consistently harder to handle than matched explicit failures. Our analysis traces this weakness through a three-stage chain: agents often fail to recognize that an inherited value may be wrong; recognition does not reliably lead them to refetch its source; and refetching does not guarantee that the corrected value is carried through to final recovery. We further find that most benchmark corruptions would change the downstream e-commerce task output, with damage varying substantially by error type and workflow position. Finally, a lightweight State Checker makes agents notice and refetch more often but yields only limited improvement in final task success. Making agents look is therefore easier than making them repair what they find. In summary, our contributions are:

• Task formulation. We formulate Silent-error Resilience Task as an interactive agentic task for studying resilience to silent errors throughout the query-to-answer process. It evaluates whether an agent can recognize an unreliable tool result, reacquire trustworthy evidence, and still produce a correct final answer rather than propagate the error downstream.

• Benchmark construction. We introduce RITE, which replays a human-authored workflow execution up to a selected point, returns a correct, silently incorrect, or explicitly failed result there, and lets the agent continue freely. Using inputs sampled from one internal platform’s production logs, workflows deployed on another, and a semantic error taxonomy, we construct 167 matched units across nine e-commerce workflows, comprising 501 silenterror, correct-value, and explicit-failure instances.

• Empirical analysis. Across 10 LLMs, silent errors roughly halve task success. We identify recognition, producer refetch, and recovery as successive failure points, characterize effects by error type and position, and propose a lightweight State Checker as an intervention to test whether resilience can be improved.

## 2 RELATED WORK

Tool-use and agent evaluation. Agent evaluation spans three levels. At the tool-call level, Meta-Tool (Huang et al., 2024b) asks whether a tool is needed and which one to select, while ToolBench, BFCL, and API-Bank (Qin et al., 2024; Patil et al., 2025; Li et al., 2023) evaluate call construction and extend to multi-tool, dependent, and stateful multi-turn interactions; ToolACE (Liu et al.,

2025) provides synthesized data for improving these abilities. At the task-outcome level, AppWorld, GAIA, and WebArena (Trivedi et al., 2024; Mialon et al., 2024; Zhou et al., 2024) judge a complete rollout by its final answer or by the state it leaves in the environment, while allowing different valid action paths. At the process level, AgentBoard (Ma et al., 2024) tracks task progress, WorFBench (Qiao et al., 2025) evaluates generated workflow structure, ToolScan (Kokane et al., 2025) diagnoses call-level errors, and process-mining methods (van der Aalst, 2016) compare executed and expected flows. Across these levels, the benchmarks above vary task difficulty and agent behavior but do not systematically manipulate the semantic correctness of intermediate tool values. We instead vary the intermediate business value and ask whether an agent can still complete the task correctly when that value is wrong. To keep the test controlled and reproducible, we provide the recorded workflow execution up to the selected checkpoint as the agent’s initial history, then let it continue freely.

Tool-failure recovery, silent-error detection, and error propagation. Work on unreliable tools can be separated by the stage of the error-handling chain that it evaluates. One line studies recovery from tool failures: agents resume from the environment left by a failed attempt (Tan et al., 2025), rewind to an earlier checkpoint (Zhuang et al., 2026), recover from explicit or implicit faults injected into tool responses while the agent runs (Zhu et al., 2026; Tian et al., 2026; Chen et al., 2026), or diagnose failed trajectories and characterize recovery cost (Vuddanti & Chittiprolu, 2026; Zhu et al., 2025; Zhang et al., 2026). Resumption settings start from uncontrolled natural failures, while injection settings expose an agent to a fault only if and when its own trajectory reaches the perturbed call. A second line studies silent-error detection, asking whether agents identify faulty outputs that trigger no exception (Sun et al., 2024) or silent data-quality faults introduced during controlled replay (Mazumder & jahan Lia, 2026). In both, the faulty response is delivered when the agent’s own run issues the corresponding call. A third line studies error propagation, tracing a schemavalid but semantically wrong argument to the final answer (Gurram, 2026) and measuring how perturbed requests or tool responses alter later calls and task outcomes (Xiong et al., 2025; Wang et al., 2026b). To our knowledge, RITE is the first benchmark to evaluate business-semantic silent errors at controlled decision points: every agent encounters the same plausible but incorrect value after the same recorded execution history, the same point is also instantiated with the correct value and with an explicit failure, and the agent then continues freely to a final answer.

E-commerce and business-domain agents. Work in this domain spans three forms of capability. Domain models test whether a model understands products and users well enough to answer questions (Peng et al., 2024), while recommender agents use that understanding to retrieve, rank, and recommend products (Gao et al., 2023; Huang et al., 2024a). Shopping benchmarks place agents in in teractive stores, where they must search, compare, and purchase while satisfying constraints on products, vouchers, budgets, and sellers (Yao et al., 2023; Wang et al., 2026a). Business-service benchmarks require agents to interact with users over multiple turns, operate on business records through tools, and comply with domain policies (Yao et al., 2025; Wang et al., 2025), while CRMArena-Pro additionally tests confidentiality in enterprise workflows (Huang et al., 2026). These benchmarks do not systematically test what happens when a task-critical business value is silently corrupted. Grounded in production execution logs and human-authored industrial e-commerce workflows, our benchmark instead examines this reliability risk involving tools developed and maintained by mul tiple parties: a call may complete while an intermediate business value is silently wrong.

## 3 PROBLEM FORMULATION

We study whether an agent can remain reliable when a tool silently returns an incorrect result and still complete the original task. We first formulate this end-to-end resilience as Silent-error Resilience Task, then define how RITE evaluates it in this section.

Silent-error Resilience Task. In a standard agentic task, an agent $A$ receives a task query $q$ and repeatedly selects and executes tool calls from a tool set $\tau$ provided by an environment $E$ . The interaction begins without a preloaded execution history, $\mathrm { i . e . , } h _ { 0 } ^ { \mathrm { s t d } } = \emptyset$ . At each step $t ,$ the agent observes the current history $h _ { t }$ and selects an action $a _ { t }$ . The environment executes the action and returns an observation $o _ { t }$ , which is appended to the history:

$$
a _ { t } = A ( q , h _ { t } , \mathcal { T } ) , \qquad o _ { t } = E ( a _ { t } ) , \qquad h _ { t + 1 } = h _ { t } \oplus ( a _ { t } , o _ { t } ) .\tag{1}
$$

The standard action set is $\mathcal { A } _ { \mathrm { 0 } } = \{ \mathrm { c a l l } , \mathrm { r e a s o n } , \mathrm { s t o p } \}$ . The interaction ends when the agent stops or reaches the step budget, producing a final answer.

As illustrated in Figure 1, the standard loop does not guarantee that a returned observation is semantically correct. In deployed systems, changes to business data, tool implementations, or deployed versions can cause runtime results to diverge from the expectations under which the agent was built. A call may still complete and conform to its API contract while returning a result that is incorrect for q. At a tool-call step $j ,$ let $o _ { i } ^ { \star }$ denote the task-

![](images/6ab5c5cecc384f92a021dc9eaa7b848cdcb64b0055b17472126c4b8341d0ae98.jpg)  
Figure 1: Silent errors from build-time–runtime mismatch. Tool-side data or implementation changes can make a runtime result diverge from what the agent was built to expect while signaling no failure.

correct result and ${ \tilde { o } } _ { j }$ the result actually returned by the tool. A silent error occurs when ${ \tilde { o } } _ { j }$ differs from $o _ { j } ^ { \star }$ in task-relevant meaning but remains plausible, schema-valid, and free of an explicit failure signal. Because the ordinary update rule still appends $( a _ { j } , \tilde { o } _ { j } )$ to the history, subsequent actions condition on unreliable evidence and may propagate it into the final answer.

The Silent-error Resilience Task measures whether the agent can still complete q after such a result has entered its execution history. Process measures track whether the agent recognizes the unreliable result, reacquires trustworthy evidence, and integrates that evidence into a correct final answer. The agent retains $\mathcal { A } _ { \mathrm { 0 } }$ and may reason from the current state, revisit or reissue an earlier tool call, seek corroborating evidence, or pursue another continuation.

Benchmark evaluation. For each benchmark unit i, RITE constructs three matched instances $D _ { i } ^ { k } \ : = \ : ( q _ { i } , p _ { i } ^ { k } , c _ { i } )$ for $k \in$ {correct, implicit, explicit}. The instances share the query $q _ { i } ,$ result constraint $c _ { i } .$ , and continuation environment $E _ { i }$ , keeping the user task, required result, and available recovery paths fixed. They differ only in the affected evidence in $p _ { i } ^ { k }$ : the correct condition provides the recorded value as a baseline, the implicit condition provides a silently incorrect value, and the explicit condition provides an overt failure to expose the effect of a visible failure signal.

For each condition, RITE initializes the agent with $h _ { 0 , i } ^ { k } = p _ { i } ^ { k }$ and provides $q _ { i }$ as the task query. Starting from the matched checkpoint controls variation in the preceding execution, while free continuation in $E _ { i }$ with the standard action set $\mathcal { A } _ { \mathrm { 0 } }$ lets the agent choose its own recovery path. The run ends when the agent stops or reaches the step budget. Let $\hat { r } _ { i } ^ { k }$ denote the agent-generated continuation, which ends with a final answer $y _ { i } ^ { k }$ ; the complete trace is $r _ { i } ^ { k } = p _ { i } ^ { k } \oplus \hat { r } _ { i } ^ { k }$

We use three evaluation measures. Noticed records whether the agent explicitly identifies a content or result anomaly and takes a related remedial action. Result Success Rate (RSR) records whether $y _ { i } ^ { k }$ resolves $q _ { i }$ , capturing end-to-end task success. Retention, the ratio of implicit-condition RSR to correct-condition RSR, reports how much baseline performance is preserved under silent errors.

## 4 BENCHMARK CONSTRUCTION AND EVALUATION ENVIRONMENT

Each human-authored workflow specifies a pre-orchestrated agentic process. Its execution produces a controlled trace of tool calls and observations in the same format as an agent rollout. We use this trace to construct matched prefixes that share the preceding history and differ only in selected intermediate evidence; all prefixes are paired with the same continuation environment. This section first describes the data sources and silent-error taxonomy, then explains how each clean trace yields paired benchmark instances $D _ { i } ^ { k }$ and an evaluation environment $E _ { i } ,$ and reports its distribution.

## 4.1 DATA SOURCES AND SILENT-ERROR TAXONOMY

Data sources. Benchmark construction begins with two source collections. The first is the work flow collection W, drawn from human-authored e-commerce workflows already deployed and running on an internal industrial workflow platform. We screen these existing workflows for structural validity and task diversity, yielding nine golden workflows with explicit control flow and deterministic tool dependencies. The second source is the input collection X from a separate internal e-commerce data platform. Candidate production-log records are mapped to workflow inputs and selected only when their executions yield complete traces with task-relevant intervention points; the corresponding traces are retained. Appendix A defines the workflow- and log-selection criteria.

![](images/9f8f74a567c9fa2469ccda5ad3825c3eda97108c0a41cb1a1a08ac8b99e956dd.jpg)  
Figure 2: RITE benchmark construction and rollout pipeline. A workflow $W _ { i }$ and paired log-derived input $x _ { i }$ are used to activate a side-effect-free workflow replica and reconstruct a clean trace $\tau _ { i }$ . A task-relevant slot then yields a shared query and result oracle together with three matched prefixes (correct, explicit, and implicit). The workflow and clean trace define the fixed evaluation environment $E _ { i }$ , in which the evaluated LLM continues from each prefix to produce a complete rollout.

Taxonomy definition. We constructed the silent-error taxonomy through a targeted, literaturegrounded synthesis of adjacent error and failure taxonomies. We selected studies whose source categories could clarify the boundary of a silent state corruption or be operationalized as schema-valid, plausible-but-wrong workflow values. For each relevant source category, two authors independently assessed its observable error signal, workflow locus, and suitability for conversion into a single-slot implicit perturbation. They then reconciled overlapping categories and clarified the resulting boundaries. The final taxonomy contains seven leaf types organized into two semantic groups. Object or category selection errors cover substitution with a same-level category (category sibling drift), an ancestor or descendant (category granularity mismatch), or a similar but incorrect real entity (entity near-duplicate). Attribute or value errors cover changes to an entity’s attribute (profile attribute drift), movement to a neighboring valid numeric range (numeric bucket shift), superficially similar text with a different business meaning (semantic synonym drift), and reuse of a formerly valid but outdated value (temporal staleness). The benchmark instantiates every leaf type except temporal staleness. Although a stale value can be schema-valid and lead to an incorrect answer, recognizing that it is stale requires prior or time-indexed evidence that the agent does not observe. Appendix B provides the complete definitions, category boundaries, and literature grounding.

## 4.2 BENCHMARK INFRASTRUCTURE CONSTRUCTION

The infrastructure has a shared start followed by two sibling construction branches:

$$
( W _ { i } , x _ { i } ) \longmapsto \tau _ { i } , \qquad \tau _ { i } \longmapsto D _ { i } ^ { k } = ( q _ { i } , p _ { i } ^ { k } , c _ { i } ) , \qquad ( W _ { i } , \tau _ { i } ) \longmapsto E _ { i } .
$$

The data branch constructs the paired prefixes initializing an agent, whereas the environment branch independently constructs the world where the agent continues acting (Fig. 2).

Workflow trace reconstruction. To avoid polluting the internal platform data, which may record the workflow execution result, we first post-process every selected workflow into a deployable replica by mocking out reporting and other side-effecting nodes and deploy the modified workflow back onto the platform. Then, a local central controller connects the log and workflow to provide an execution trace. It initially maps the production log record to the start schema of the deployed replica, producing the input $x _ { i }$ . It then invokes the deployed workflow through the platform API with $x _ { i }$ and retrieves the full execution trace of that run, which records every executed node with its arguments, outputs, status, and data dependencies. We retain only executions that complete successfully and contain full information needed to reconstruct the workflow state. The retained $\tau _ { i }$ is therefore the clean execution record shared by the two downstream branches.

Paired prefix construction. Slot selection. We align the workflow graph with $\tau _ { i }$ and obtain an execution graph whose edges carry the values propagated between nodes. We then search this graph along producer $ s _ { i }  \mathrm { c o n s u m e r } $ downstream result: an intermediate slot $s _ { i }$ qualifies when it carries business-relevant content, is consumed downstream, and plausibly affects the task outcome. For each retained slot, we record its producer and consumer and determine the locus at which the intervention is realized, either in the producer observation (tool output) or in a downstream tool argument (tool input). A single execution yields multiple such slots, and each retained slot defines one task through the prefix built around it.

Task construction. This stage converts each retained slot into a task including the task query and corresponding three paired prefixes.

(i) Three conditions at the same slot. We truncate the common trace at the same slot and render the three prefixes $p _ { i } ^ { \mathrm { c o r r e c t } } , p _ { i } ^ { \mathrm { e x p l i c i t } }$ , and $p _ { i } ^ { \mathrm { i m p l i c i t } } \ ( \mathrm { S e c . } \ 3 )$ , with neutral historical rationales and no corruption marker. The correct condition preserves the recorded value. The explicit condition replaces access to that value with an overt failure, such as an empty argument or an error-bearing observation. The implicit condition obtains its value in two steps. First, it selects an applicable error type from the taxonomy and constructs a substitute that is typeand schema-compatible, remains locally plausible, and changes the task-relevant meaning of the slot. Depending on the error type and carrier, candidates come from compatible values observed in other cases or from a type-specific controlled transformation. Second, the selected substitute is written into the slot at its predetermined locus: the producer observation under tool output, or a downstream tool argument under tool input.

(ii) Query and oracle. From the clean trace, we extract the correct value of $s _ { i }$ and construct the result constraint $c _ { i }$ that a correct final answer must satisfy. We then generate or refine $q _ { i }$ so that it is designed to depend on this value: answers grounded in the gold and polluted values should be distinguishable under $c _ { i }$

Verification. Before delivery, each candidate must preserve prefix executability and structural compatibility. Four additional gates then check the intervention itself. (1) Plausibility. The implicit value must remain schema-valid and locally reasonable while differing from the recorded value in task-relevant meaning. (2) Explicit-failure clarity. The explicit condition must expose an unambiguous failure signal. (3) Non-leakage. Neither the query nor the prefix may reveal that an intervention was applied. (4) Discriminability. The query remains solvable, and answers grounded in the gold value and in the polluted value must remain distinguishable under $c _ { i }$ . The surviving artifacts are delivered in the formulation’s schema $D _ { i } ^ { k } = \left( q _ { i } , p _ { i } ^ { k } , c _ { i } \right)$ , with all three conditions sharing the same query and result constraint. Providing every model with the same recorded execution before the affected value deliberately reduces variation in reaching that point, so the matched conditions focus on how agents handle the evidence they receive there.

Evaluation environment construction. In parallel with prefix construction, we extract the callable tool set $\mathcal { T } _ { i }$ from $W _ { i }$ and build one replay world $R _ { i }$ for that workflow by integrating the recorded responses of all collected traces.

$$
( W _ { i } , \tau _ { i } ) \longmapsto E _ { i } ( a _ { t } ) = { \left\{ \begin{array} { l l } { R _ { i } ( a _ { t } ) , } & { a _ { t } \in \mathrm { d o m } ( R _ { i } ) , } \\ { \operatorname { L i v e } _ { { \mathcal { T } } _ { i } } ( a _ { t } ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. }\tag{2}
$$

Under this interface, a covered call returns the recorded response from $R _ { i } ,$ while an unseen valid call is dispatched to the live implementation in $\mathcal { T } _ { i }$ deployed on the internal industrial workflow platform. RITE changes only the condition-specific prefix, not $E _ { i }$ because the replay world is built from the clean trace, re-querying a recorded producer returns the corresponding unpolluted response, giving the agent a direct path to recover reliable evidence and produce a correct final answer. Every instance is therefore recoverable by design, isolating whether agents use an available verification path; faults that persist across re-queries are discussed in Appendix G.

## 4.3 DATASET STATISTICS

The final RITE release contains 167 paired units, each defined by a unique slot–locus pair and expanded into correct, explicit, and implicit conditions. This yields 501 instances, with 167 instances per condition and complete triplets throughout. Nine workflows contribute data, and the release covers six of the seven taxonomy error types because temporal staleness has no retained instance. Of the 167 units, 147 intervene at tool output; the remaining 20 intervene at tool input. Detailed distributions are reported in Appendix F.

## 5 EXPERIMENTS

We evaluate 10 recent LLMs in three groups: two closed frontier models (Claude-Opus-4.8 and GPT-5.5), five open-weight frontier models (GLM-5.2, Kimi-K3, DeepSeek-V4-Pro, Hy3, and Qwen3.5-397B), and three smaller open-weight models (Qwen3.6-35B, Qwen3.5-35B, and Qwen3.5-122B). Each model runs all 501 instances at temperature 0 with a 25-step budget. Rollouts invalidated by environment-side tool anomalies are excluded. The evaluation environment fol lows Section 4.2, and the rollout and judging protocol is detailed in Appendix C, and a human validation of the Result and Noticed labels is reported in Appendix D. Following Section 3, we report condition-specific RSR and Noticed, Retention, and the paired gap: the paired explicit-minusimplicit RSR difference over slots observed in both conditions.

## 5.1 MAIN RESULTS

Table 1 shows that implicit errors substantially degrade task performance across all evaluated models and, in most cases, remain unnoticed. Appendix E provides two end-to-end cases.

Table 1: Results across the three matched conditions, grouped by model class and ordered within each group by implicit-condition Noticed. The Noticed gap is explicit minus implicit. Bold marks the best value in each column.
<table><tr><td></td><td></td><td colspan="3">Noticed</td><td colspan="3">RSR</td><td colspan="2">Resilience</td></tr><tr><td>Category</td><td>Model</td><td>Implicit ↑ Explicit ↑</td><td></td><td>Gap</td><td></td><td></td><td></td><td></td><td>Correct ↑ Implicit ↑ Explicit ↑ Retention ↑ Paired gap</td></tr><tr><td rowspan="2">Closed frontier</td><td>Claude-Opus-4.8</td><td>0.336</td><td>0.579</td><td>0.243</td><td>0.765</td><td>0.403</td><td>0.507</td><td>0.526</td><td>+0.089</td></tr><tr><td>GPT-5.5</td><td>0.280</td><td>0.594</td><td>0.314</td><td>0.716</td><td>0.363</td><td>0.526</td><td>0.507</td><td>+0.185</td></tr><tr><td rowspan="5">Open-weight frontier Kimi-K3</td><td></td><td>0.263</td><td>0.590</td><td>0.327</td><td>0.784</td><td>0.302</td><td>0.555</td><td>0.385</td><td>+0.228</td></tr><tr><td>GLM-5.2</td><td>0.217</td><td>0.487</td><td>0.270</td><td>0.805</td><td>0.277</td><td>0.507</td><td>0.345</td><td>+0.243</td></tr><tr><td>Hy3</td><td>0.215</td><td>0.494</td><td>0.278</td><td>0.806</td><td>0.310</td><td>0.552</td><td>0.385</td><td>+0.236</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.212</td><td>0.468</td><td>0.255</td><td>0.827</td><td>0.327</td><td>0.533</td><td>0.395</td><td>+0.217</td></tr><tr><td>Qwen3.5-397B</td><td>0.210</td><td>0.477</td><td>0.267</td><td>0.809</td><td>0.320</td><td>0.543</td><td>0.396</td><td>+0.220</td></tr><tr><td rowspan="3">Smaller open-weight Qwen3.6-35B</td><td></td><td>0.248</td><td>0.512</td><td>0.264</td><td>0.823</td><td>0.335</td><td>0.531</td><td>0.408</td><td>+0.210</td></tr><tr><td>Qwen3.5-35B</td><td>0.211</td><td>0.512</td><td>0.301</td><td>0.791</td><td>0.340</td><td>0.534</td><td>0.429</td><td>+0.208</td></tr><tr><td>Qwen3.5-122B</td><td>0.208</td><td>0.506</td><td>0.299</td><td>0.784</td><td>0.302</td><td>0.564</td><td>0.385</td><td>+0.265</td></tr></table>

F1: Silent errors substantially degrade task performance across models. Across all ten models, implicit errors reduce RSR from 0.716–0.827 in the correct condition to 0.277–0.403, yielding Retention of only 0.345–0.526 (Table 1). This loss is tied to the absence of an observable failure signal: when the corresponding failure is explicit, RSR rises to 0.507–0.564, producing a positive explicit-minus-implicit paired gap for every model (+0.089 to +0.265; +0.214 pooled). Stronger task performance does not eliminate this vulnerability. Correct-condition RSR spans only 0.111 across models, whereas Retention spans 0.182; closed frontier models have higher mean Retention than open-weight models (0.517 versus 0.391), yet even the best model preserves barely half of its correct-condition performance. Thus, susceptibility to silent errors is widespread rather than confined to models with weak baseline task ability.

F2: Models rarely notice silent errors or refetch their producers. Across models, implicit-condition Noticed ranges only from 0.208 to 0.336 (Table 1), so most silent errors go unrecognized. Table 2 shows that recognition helps but is not sufficient: the latest producer observation matches the gold value in 0.3% of Unnoticed rollouts and 23.3% of Noticed rollouts, while 57.8% of Noticed rollouts still match the polluted value. To examine whether recognition leads to active verification, we track producer refetch, defined as re-invoking the tool that produced the affected value. Among scored Noticed rollouts, RSR is 0.387 without a producer refetch and 0.655 with one. GLM-5.2 illustrates this gap: its Noticed rate is 0.217, but its producer-refetch rate is only 0.071 and its Retention is the lowest at 0.345. Across the eight open-weight models, producer-refetch rate is positively but not significantly associated with Retention $( r = 0 . 4 9 1$ $p = 0 . 2 1 7 )$ . These results identify two bottlenecks: models rarely notice silent errors, and noticing one often does not lead to refetching its source.

Table 2: Latest valid producer observation by Noticed in the implicit condition.
<table><tr><td>Noticed</td><td>n</td><td>Gold Polluted Other</td></tr><tr><td>No</td><td>1198 0.3%</td><td>99.6% 0.1%</td></tr><tr><td>Yes</td><td>377 23.3%</td><td>57.8%18.8%</td></tr></table>

F3: Producer refetch does not ensure recovery. To test whether producer refetch restores the correct evidence, the state judge compares the latest valid producer observation with the gold and polluted values. Among the 147 implicit-condition rollouts with a producer refetch, 87 (59.2%) match the gold value and 60 (40.8%) match neither value; none match the polluted value. Of the 87 matches, 85 are obtained in one re-query and the remaining two require multiple rounds. Among Noticed rollouts with a producer refetch, about one third still fail the task (RSR 0.655; F2).

To understand why refetching does not ensure recovery, we manually inspect the 60 trajectories that did not match value, focusing on unsolved rollouts in which the re-query nonetheless returns usable evidence. After usable evidence is acquired, we identify three recurring patterns. First, reasoning overwrite occurs when a model initially uses recovered evidence but later replaces it with a selfimposed heuristic. In one trajectory, an agent reconstructs the correct timestamps and uses them in the next RPC call, but then redefines the “opening warm-up” boundary and selects the wrong final comment. Second, lossy selection occurs when a model alters recovered candidates while choosing among them. One agent merges two recovered topic clusters despite an instruction to select one, while another replaces the retrieved topic with a plausible but unsupported reformulation. Third, unsupported synthesis occurs when a model uses corrected evidence but adds claims absent from it. Agents incorporate corrected profile or interest labels but add unsupported viewing-duration, interaction, or presentation claims. Across these patterns, usable evidence is obtained, but the correction does not carry through to the final answer.

## 6 DISCUSSION

The main results separate silent-error handling into recognition, use of an available verification path, and integration of the resulting evidence into final recovery. This section asks three follow-up questions: whether the resulting performance loss corresponds to changed e-commerce task outputs (D1), where the vulnerability concentrates (D2), and how far a light check can go (D3).

D1: Do silent errors change e-commerce task outputs? Not every corrupted task-relevant value changes the downstream e-commerce output. We therefore use an LLM judge to identify slots where answering with the gold versus the polluted value would change the business conclusion (Appendix C) and recompute Retention on this subset. Table 3a compares it with the full benchmark, while Table 3b reports the output-change rate by error type.

The gold and polluted values are judged to lead to different task outputs for 146 of 167 slots (87.4%). Depending on the workflow, this means a different product recommendation, live-assistant intervention, topic suggestion, or other task result. The change rate is at least 82.6% for every nonnumeric error type and 65.6% for numeric bucket shifts. In a typical non-changing numeric case, average watch times of 64 and 77 seconds both clear a 60-second threshold and therefore produce the same downstream result. Restricting evaluation to the 146 output-changing slots changes mean Retention only from 0.416 to 0.408. The near-identical Retention shows that F1’s performance loss is not driven by factual differences that leave downstream behavior unchanged; it persists, and is slightly larger, when silent errors alter the e-commerce action or result.

Table 3: E-commerce task-output impact and result robustness.
<table><tr><td>Subset</td><td>Slots</td><td>Share</td><td>Ret.</td></tr><tr><td>All</td><td>167</td><td>100.0%</td><td>0.416</td></tr><tr><td>Output-changing</td><td>146</td><td>87.4%</td><td>0.408</td></tr></table>

(a) Overall output impact.
<table><tr><td>Error type</td><td>Slots</td><td>Change rate</td></tr><tr><td>Cat. granularity</td><td>17</td><td>100.0%</td></tr><tr><td>Cat. sibling</td><td>20</td><td>95.0%</td></tr><tr><td>Profile attribute</td><td>35</td><td>94.3%</td></tr><tr><td>Semantic synonym</td><td>40</td><td>92.5%</td></tr><tr><td>Entity near-duplicate</td><td>23</td><td>82.6%</td></tr><tr><td>Numeric bucket</td><td>32</td><td>65.6%</td></tr></table>

(b) Output-change rate by error type.

D2: Damage concentrates by error type and slot position. Damage varies across both silenterror types and slot positions: it is greatest for category errors (Table 4a) and for slots that occur early in the workflow trace (Table 4b).

Error type. Category errors are the most damaging error types in RITE: category granularity mismatch is hardest, category sibling drift is next, and numeric bucket shift is most noticeable but remains destructive (Table 4a. One possible explanation is that in the benchmark’s recoverable environment, a wrong number or ID is checked against the producer response, whereas a wrong category appears anomalous only in context. For example, identifying “blockchain” as an implausible interest requires knowing the user watches local sports; the category value alone carries no anomaly.

Slot position. Early slots have the highest Noticed rate (0.264) but the lowest implicit-condition RSR (0.309) and Retention (0.399; Table 4b). One plausible explanation is that an early checkpoint yields a shorter prefix with fewer inherited observations, making its corruption easier to notice. At the same time, the agent must complete a longer downstream path, giving the polluted value—or an incomplete recovery—more opportunities to affect later steps.

Table 4: Damage localization. Slots are paired units; Not. and RSR are measured in the implicit condition.
<table><tr><td>Error type</td><td>Slots</td><td>Not.</td><td>RSR</td><td>Ret.</td></tr><tr><td>Category granularity</td><td>17</td><td>0.099</td><td>0.140</td><td>0.180</td></tr><tr><td>Category sibling</td><td>20</td><td>0.121</td><td>0.199</td><td>0.292</td></tr><tr><td>Numeric bucket</td><td>32</td><td>0.342</td><td>0.224</td><td>0.297</td></tr><tr><td>Semantic synonym</td><td>40</td><td>0.231</td><td>0.307</td><td>0.382</td></tr><tr><td>Profile attribute</td><td>35</td><td>0.268</td><td>0.479</td><td>0.562</td></tr><tr><td>Entity near-duplicate</td><td>23</td><td>0.258</td><td>0.507</td><td>0.608</td></tr></table>

(a) Damage by silent-error type.

The positional trend therefore separates detection from final success: from early to late, Noticed falls, while implicitcondition RSR and Retention rise. This pattern makes upstream verification especially important.

<table><tr><td>Slot position</td><td>Slots</td><td>Not.</td><td>RSR</td><td>Ret.</td></tr><tr><td>Early</td><td>128</td><td>0.264</td><td>0.309</td><td>0.399</td></tr><tr><td>Mid</td><td>26</td><td>0.175</td><td>0.384</td><td>0.452</td></tr><tr><td>Late</td><td>13</td><td>0.122</td><td>0.395</td><td>0.470</td></tr></table>

(b) Damage by slot position.

D3: A lightweight intervention for handling silent errors. F2 shows that blind trust in inherited tool results is the dominant failure pattern, suggesting an additional reliability check during execution. We test State Checker, which asks the model to treat every tool result as potentially unreliable and, before using it, conduct an additional step to reflect on whether the result is trustworthy and should be verified. The underlying tools and benchmark instances remain unchanged. Claude-Opus-4.8, DeepSeek-V4-Pro, and GLM-5.2 each rerun half of their slots (Appendix C) with the additional check, paired against the same slots without

it. State Checker increases Noticed and producer-refetch rates, but produces only modest gains in implicit-condition RSR (Figure 3).  
![](images/1d51e1c4c0a4ae246f52626a19c7c13f9a5f6b77b292530f5b0d1ed7b358c62b.jpg)

(b)  
![](images/f2b811e09bc938155c3222e634a93812b355b0b6a7fdd3d6f2de20876ed388b7.jpg)

![](images/2fa7da8eea542e8a3610faf8a870efc8f7062e805300f5ab090d2c68526f5873.jpg)

![](images/3f51189e7b86e6003fa50899ef83029c3314babeed45293424839453c5993312.jpg)  
Figure 3: (a) State Checker adds a reliability check before a tool result is used. (b) Noticed, producer-refetch, and implicit-condition RSR with/without State Checker on paired implicit slots.

State Checker changes behavior in all three models, with larger increases in Noticed and producer refetch than in final RSR. The clearest behavioral shift is in GLM-5.2, whose producer-refetch rate more than triples; by contrast, Claude-Opus-4.8 gains no RSR despite noticing more errors.

State Checker makes evidence-seeking behavior steerable: an explicit reliability check increases recognition and use of the available verification path. The smaller RSR gains locate the remaining bottleneck in evidence integration and final task success, while the surfaced contradictions improve observability for a human or downstream controller. State Checker relies on the unpolluted verification path in RITE’s single-site corruption setting. Persistent or system-wide tool faults may invalidate this path, requiring independent sources, tool substitution, redundancy, or human check.

## 7 CONCLUSION

We study silent errors in agentic tasks, where a plausible but incorrect tool result enters the execution history without an explicit failure signal and propagates through subsequent steps. We formulate a task that measures whether an agent can tolerate this corrupted state and still complete the original task, and introduce RITE, a benchmark built from 167 corruption slots across nine industrial e-commerce workflows grounded in real service logs. Across 10 recent LLMs, silent errors roughly halve task success and usually remain unnoticed; recovery remains incomplete even when models question the affected evidence and re-query its producer. Further analysis shows that these errors often alter downstream e-commerce outputs, with category errors and early-stage corruptions causing the greatest damage. Beyond the benchmark, we provide a simple reliability check that increases error detection and producer refetching more than final task success.

## AUTHOR STATEMENTS

## AI USE STATEMENT

In this work, we used generative AI tools in four roles. First, the authors wrote the initial draft of every manuscript section, after which AI tools were used to polish the language and improve clarity. Second, AI assisted benchmark construction by helping screen candidate workflows and production-log records (Appendix A) and by serving as model components during workflow execution. The authors manually reviewed every workflow selected through this process. Third, GLM-5 serves as the automatic judge of task success and observation state, and labels whether a corruption would change the business conclusion (Appendix C). Fourth, AI assisted with coding. The authors reviewed and validated all AI-assisted code, screening decisions, workflow outputs, and generated benchmark artifacts before use.

## ETHICS STATEMENT

The workflows used in this study were deployed in production, and the logs were generated by their online operation. Before any log-derived data were provided to language models, sensitive identifiers such as user IDs were anonymized; benchmark runs used only anonymized data.

Benchmark data and generated outputs were isolated from production storage: no benchmark inputs, intermediate states, or outputs were reported to or written back to company production databases.

## REPRODUCIBILITY STATEMENT

The paper documents the task, data sources, benchmark construction, evaluation protocol, dataset distribution, and auditable workflow examples needed to understand and assess the reported results.

The code is subject to an internal company review that may take several months. We will release it if approval is granted; otherwise, it will remain unavailable.

## REFERENCES

Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. Why do multi-agent llm systems fail?, 2025. URL https://arxiv. org/abs/2503.13657.

Chaoran Chen, Vy Nguyen, Ziji Zhang, Abhinav Gullapalli, Ziyi Wang, Yuxuan Lu, Dakuo Wang, Jing Huang, Zhou Yu, and Jin Lai. Retry, switch, or abstain? learning strategy-aware tooluse policies via controlled error injection, 2026. URL https://arxiv.org/abs/2608. 11977.

Ahmed K. Elmagarmid, Panagiotis G. Ipeirotis, and Vassilios S. Verykios. Duplicate record detection: A survey. IEEE Trans. on Knowl. and Data Eng., 19(1):1–16, January 2007. ISSN 1041-4347.

Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, and Jiawei Zhang. Chat-rec: Towards interactive and explainable llms-augmented recommender system, 2023. URL https: //arxiv.org/abs/2303.14524.

Bhaskar Gurram. Auditing automated evaluation, error propagation, and runtime mitigation in toolusing language agents, 2026. URL https://arxiv.org/abs/2604.16706.

Kung-Hsiang Huang, Akshara Prabhakar, Onkar Thorat, Divyansh Agarwal, Prafulla Kumar Choubey, Yixin Mao, Silvio Savarese, Caiming Xiong, and Chien-Sheng Wu. CRMArena-pro: Holistic assessment of LLM agents across diverse business scenarios and interactions. Transactions on Machine Learning Research, 2026. ISSN 2835-8856. URL https://openreview. net/forum?id=EPlpe3Fx1x.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1–55, January 2025. ISSN 1558-2868. doi: 10.1145/3703155. URL http://dx.doi.org/10.1145/3703155.

Xu Huang, Jianxun Lian, Yuxuan Lei, Jing Yao, Defu Lian, and Xing Xie. Recommender ai agent: Integrating large language models for interactive recommendations, 2024a. URL https:// arxiv.org/abs/2308.16505.

Yue Huang, Jiawen Shi, Yuan Li, Chenrui Fan, Siyuan Wu, Qihui Zhang, Yixin Liu, Pan Zhou, Yao Wan, Neil Zhenqiang Gong, and Lichao Sun. Metatool benchmark for large language models: Deciding whether to use tools and which to use. In The Twelfth International Conference on Learning Representations, 2024b. URL https://openreview.net/forum?id=R0c2qtalgG.

Won Kim, {Byoung Ju} Choi, {Eui Kyeong} Hong, {Soo Kyung} Kim, and Doheon Lee. A taxonomy of dirty data. Data Mining and Knowledge Discovery, 7(1):81–99, January 2003. ISSN 1384-5810. doi: 10.1023/A:1021564703268.

Shirley Kokane, Ming Zhu, Tulika Manoj Awalgaonkar, Jianguo Zhang, Akshara Prabhakar, Thai Quoc Hoang, Zuxin Liu, Rithesh R N, Liangwei Yang, Weiran Yao, Juntao Tan, Zhiwei Liu, Shelby Heinecke, Huan Wang, Juan Carlos Niebles, Caiming Xiong, and Silvio Savarese. Toolscan: A benchmark for characterizing errors in tool-use LLMs. In ICLR 2025 Workshop on Building Trust in Language Models and Applications, 2025. URL https://openreview. net/forum?id=09tnQgqKuZ.

Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. API-bank: A comprehensive benchmark for tool-augmented LLMs. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 3102–3116, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.187. URL https://aclanthology.org/2023.emnlp-main.187/.

Weiwen Liu, Xu Huang, Xingshan Zeng, xinlong hao, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Zhengying Liu, Yuanqing Yu, Zezhong WANG, Yuxian Wang, Wu Ning, Yutai Hou, Bin Wang, Chuhan Wu, Wang Xinzhi, Yong Liu, Yasheng Wang, Duyu Tang, Dandan Tu, Lifeng Shang, Xin Jiang, Ruiming Tang, Defu Lian, Qun Liu, and Enhong Chen. ToolACE: Winning the points of LLM function calling. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=8EB8k6DdCU.

Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. Agentboard: An analytical evaluation board of multi-turn llm agents, 2024. URL https://arxiv.org/abs/2401.13178.

Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. On faithfulness and factuality in abstractive summarization. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (eds.), Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pp. 1906–1919, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/ 2020.acl-main.173. URL https://aclanthology.org/2020.acl-main.173/.

Aritra Mazumder and Nusrat jahan Lia. Agentcheck: A reproduce-intervene-mitigate workbench for llm agents over mcp, 2026. URL https://arxiv.org/abs/2607.11098.

Gregoire Mialon, Cl´ ementine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia:´ a benchmark for general ai assistants. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 9025–9049, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/ 2024/file/25ae35b5b1738d80f1f03a8713e405ec-Paper-Conference.pdf.

Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=2GmDdhBdDk.

Bo Peng, Xinyi Ling, Ziru Chen, Huan Sun, and Xia Ning. ecellm: Generalizing large language models for e-commerce from large-scale, high-quality instruction data, 2024. URL https: //arxiv.org/abs/2402.08831.

Shuofei Qiao, Runnan Fang, Zhisong Qiu, Xiaobin Wang, Ningyu Zhang, Yong Jiang, Pengjun Xie, Fei Huang, and Huajun Chen. Benchmarking agentic workflow generation, 2025. URL https://arxiv.org/abs/2410.07869.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, dahai li, Zhiyuan Liu, and Maosong Sun. ToolLLM: Facilitating large language models to master 16000+ real-world APIs. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=dHng2O0Jjr.

Yangjun Ruan, Honghua Dong, Andrew Wang, Silviu Pitis, Yongchao Zhou, Jimmy Ba, Yann Dubois, Chris J. Maddison, and Tatsunori Hashimoto. Identifying the risks of lm agents with an lm-emulated sandbox, 2024. URL https://arxiv.org/abs/2309.15817.

Jimin Sun, So Yeon Min, Yingshan Chang, and Yonatan Bisk. Tools fail: Detecting silent errors in faulty tools. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 14272–14289, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.790. URL https://aclanthology.org/2024. emnlp-main.790/.

Shangyin Tan, Kevin Lin, Koushik Sen, and Matei Zaharia. Recovery-bench: Evaluating agentic recovery from mistakes. In NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling, 2025. URL https://openreview.net/ forum?id=8FZRnDgDxq.

Liyan Tang, Tanya Goyal, Alex Fabbri, Philippe Laban, Jiacheng Xu, Semih Yavuz, Wojciech Kryscinski, Justin Rousseau, and Greg Durrett. Understanding factual errors in summarization: Errors, summarizers, datasets, error detectors. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11626–11644, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.650. URL https://aclanthology.org/2023.acl-long.650/.

Yang Tian, Zhengpeng Shi, Yu Zhou, and Bo Zhao. Beyond function calling: Benchmarking toolusing agents under tool-environment unreliability, 2026. URL https://arxiv.org/abs/ 2606.25819.

Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian. AppWorld: A controllable world of apps and people for benchmarking interactive coding agents. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 16022–16076, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.850. URL https://aclanthology.org/2024.acl-long.850/.

Gladys Tyen, Hassan Mansoor, Victor Carbune, Peter Chen, and Tony Mak. Llms cannot find ˘ reasoning errors, but can correct them given the error location, 2024. URL https://arxiv. org/abs/2311.08516.

Wil M. P. van der Aalst. Process mining: data science in action. Springer, Germany, January 2016. ISBN 978-3-662-49850-7. doi: 10.1007/978-3-662-49851-4.

Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, and Thang Luong. FreshLLMs: Refreshing large language models with search engine augmentation. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings of the Association for Computational Linguistics: ACL 2024, pp. 13697–13720, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653

v1/2024.findings-acl.813. URL https://aclanthology.org/2024.findings-acl. 813/.

Sri Vatsa Vuddanti and Satwik Kumar Chittiprolu. Recoverability has a law: The err measure for tool-augmented agents, 2026. URL https://arxiv.org/abs/2601.22352.

Haoxin Wang, Xianhan Peng, Huang Cheng, Yizhe Huang, Ming Gong, Chenghan Yang, Yang Liu, and Jiang Lin. ECom-bench: Can LLM agent resolve real-world E-commerce customer support issues? In Saloni Potdar, Lina Rojas-Barahona, and Sebastien Montella (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 276–284, Suzhou (China), November 2025. Association for Computational Linguistics. ISBN 979-8-89176-333-3. doi: 10.18653/v1/2025.emnlp-industry.19. URL https: //aclanthology.org/2025.emnlp-industry.19/.

Jiangyuan Wang, Kejun Xiao, Qi Sun, Huaipeng Zhao, Tao Luo, Jian Dong Zhang, and Xiaoyi Zeng. Shoppingbench: A real-world intent-grounded shopping benchmark for llm-based agents, 2026a. URL https://arxiv.org/abs/2508.04266.

Ruipeng Wang, Yuxin Chen, Yukai Wang, Chang Wu, Junfeng Fang, Xiaodong Cai, Qi Gu, Hui Su, An Zhang, Xiang Wang, Xunliang Cai, and Tat-Seng Chua. Agentnoisebench: Benchmarking robustness of tool-using llm agents under noisy condition, 2026b. URL https://arxiv. org/abs/2602.11348.

Cailin Winston and Rene Just. A taxonomy of failures in tool-augmented llms. In´ 2025 IEEE/ACM International Conference on Automation of Software Test (AST), pp. 125–135, 2025. doi: 10. 1109/AST66626.2025.00019.

Qian Xiong, Yuekai Huang, Ziyou Jiang, Zhiyuan Chang, Yujia Zheng, Tianhao Li, and Mingyang Li. Butterfly effects in toolchains: A comprehensive analysis of failed parameter filling in LLM tool-agent systems. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 16712–16729, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/2025.findings-emnlp.907. URL https: //aclanthology.org/2025.findings-emnlp.907/.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable realworld web interaction with grounded language agents, 2023. URL https://arxiv.org/ abs/2207.01206.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik R Narasimhan. τ-bench: A benchmark for Tool-Agent-User interaction in real-world domains. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= roNSXZpUDN.

Guibin Zhang, Junhao Wang, Junjie Chen, Wangchunshu Zhou, Kun Wang, and Shuicheng YAN. Agentracer: Who is inducing failure in the LLM agentic systems? In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=l05DseqvuD.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id= oKn9c6ytLx.

Dongsheng Zhu, Xuchen Ma, Yucheng Shen, Xiang Li, Yukun Zhao, Shuaiqiang Wang, Lingyong Yan, and Dawei Yin. When tools fail: Benchmarking dynamic replanning and anomaly recovery in llm agents, 2026. URL https://arxiv.org/abs/2606.05806.

Kunlun Zhu, Zijia Liu, Bingxuan Li, Muxin Tian, Yingxuan Yang, Jiaxun Zhang, Pengrui Han, Qipeng Xie, Fuyang Cui, Weijia Zhang, Xiaoteng Ma, Xiaodong Yu, Gowtham Ramesh, Jialian Wu, Zicheng Liu, Pan Lu, James Zou, and Jiaxuan You. Where llm agents fail and how they can learn from failures, 2025. URL https://arxiv.org/abs/2509.25370.

Yu Zhuang, Kefei Chen, Yitong Duan, Shuxin Zheng, Jian Li, and Xu-Yao Zhang. Agentrewind: Recoverable execution for long-horizon llm agents, 2026. URL https://arxiv.org/abs/ 2608.14380.

## A WORKFLOW AND LOG DATA SOURCES

This appendix documents two data-source stages: selecting the workflow collection W and mapping production logs to workflow inputs that yield eligible traces. It also records the resulting golden workflows. Workflow identifiers are anonymized and stable across the released artifacts.

Workflow selection: structural validity. We inspected e-commerce applications at the DSL level and retained complete deterministic DAGs with explicit control flow. We excluded graphs containing autonomous agentic nodes, atomic retrieval wrappers, short utility graphs, AB-test fragments, dangling references, or incomplete subgraphs. Fixed tool calls, read-only requests, code and LLM transformations, and explicit branch or loop nodes remained eligible.

Workflow selection: task diversity. Because the application library contains multiple versions of many task families, we deduplicated each family, selected a representative complete version, and formed a set spanning different business stages, task objectives, and data dependencies. These two workflow checks yielded nine golden workflows.

Log selection and trace retention. For each selected workflow, we mapped each candidate production-log record to the required start inputs and executed it on a structure-preserving replica. We selected a log only when its execution yielded a complete node-level trace with sufficient process depth and at least one business-semantic intermediate state that was produced upstream, consumed downstream, and locatable at a producer or consumer checkpoint. The corresponding trace was retained with the selected log. For safe trace collection, the replicas retained node identities, graph edges, data references, read-only retrieval, deterministic computation, and model-inference nodes, while writes, reporting calls, locks, memory updates, and external sends were neutralized or guarded.

Table 5 summarizes the selected workflows, their log-derived execution contexts, graph sizes, and resulting final units. Graph size reports the total nodes and statically counted tool- or model-facing interfaces in each collection replica. The interface count is descriptive and precedes trace-level eligibility checks. Afinal unit is one retained slot–locus pair before expansion into the three conditions.

Table 5: The nine golden workflows used by RITE. Log-derived inputs are described by semantic role rather than internal field names. Graph size is reported as total nodes / static tool-model interfaces.
<table><tr><td>ID</td><td>Task abstraction</td><td>Log-derived execution context</td><td>Graph size</td><td>Final units</td></tr><tr><td>wf01</td><td>needs, creator profiles, and</td><td>Product recommendation from user Anonymized viewer, creator, and request context. A task query is constructed from the available profile context</td><td>55 / 14</td><td>11</td></tr><tr><td>wf02</td><td>selection history Live-assistant scheduling and intervention selection</td><td>Anonymized session, creator, task, and configuration context</td><td>58 / 11</td><td>22</td></tr><tr><td>wf04</td><td>Pre-session script generation from creator and audience context</td><td>Anonymized creator identity. Profiles and historical content are retrieved by read-only nodes</td><td>22 / 11</td><td>15</td></tr><tr><td>wf05</td><td>Real-time product-presentation coaching</td><td>Anonymized session, creator, task, and available configuration context</td><td>47 / 12</td><td>17</td></tr><tr><td>wf06</td><td>Cross-domain intent and interest modeling</td><td>An anonymized user identity passing data-availability screening. Behavioral and profile records are retrieved downstream</td><td>52 / 14</td><td>52</td></tr><tr><td>wf07</td><td>Post-session summary over content, interaction, and commerce session-derived time window</td><td>Anonymized session, creator, request, and a</td><td>56 / 10</td><td>15</td></tr><tr><td>wf08</td><td>signals Topic suggestion from historical videos and comments</td><td>Anonymized creator, video or session, and request 21 / 5 context</td><td></td><td>12</td></tr><tr><td>wf09</td><td>Extraction of reusable presentation patterns from strong sessions</td><td>Anonymized session, creator, request, and analysis 37 / 5 window</td><td></td><td>9</td></tr></table>

Continued on next page.

Table 5 continued.
<table><tr><td>ID</td><td>Task abstraction</td><td>Log-derived execution context</td><td>Graph size</td><td>Final units</td></tr><tr><td>wf10</td><td>Periodic perception of the current live-session state</td><td>Anonymized session, creator, task, and available configuration context</td><td>46/9</td><td>14</td></tr><tr><td colspan="3">Total: nine golden workflows</td><td>394 / 91</td><td>167</td></tr></table>

## B SILENT-ERROR TAXONOMY AND LITERATURE EVIDENCE

We synthesize 15 papers spanning agent and tool failures, data quality, and factuality. Two authors independently coded the error phenomena and boundaries described by each source, mapped overlapping concepts to a common vocabulary, and reconciled the mappings. The resulting taxonomy describes the semantic form of a plausible but incorrect intermediate value. Injection location is recorded separately.

## B.1 OPERATIONAL DEFINITIONS

The main taxonomy contains two semantic groups and seven leaves (Table 6). The first group changes which object or category is referenced; the second preserves the object while changing one of its attributes or values. Each leaf requires a schema-valid value whose task-relevant meaning differs from the recorded value.

Table 6: Operational definitions and boundaries of the seven silent-error leaves.
<table><tr><td>ID</td><td>Leaf</td><td>Operational definition</td><td>Primary boundary</td></tr><tr><td>L1.1</td><td>category-sibling-drift</td><td>Replace a category with a nearby category at the same hierarchy level.</td><td>Parent-child substitutions belong to L1.2.</td></tr><tr><td>L1.2</td><td>category-granularity- mismatch</td><td>Replace a category with an ancestor or descendant, making its scope too broad or too narrow.</td><td>Same-level substitutions belong to L1.1.</td></tr><tr><td>L1.3</td><td>entity-near_duplicate</td><td>Replace the intended entity with another real, similar entity from the candidate domain.</td><td>If entity identity is preserved and only an attribute changes, use L2.1.</td></tr><tr><td>L2.1</td><td>profile_attribute_drift</td><td>Preserve entity identity but replace a profile, preference, label, or other attribute with another valid value.</td><td>Whole-entity replacement belongs to L1.3.</td></tr><tr><td>L2.2</td><td>numeric_bucket_shift</td><td>Move a numeric value or interval to a neighboring valid range.</td><td>Malformed, impossible, or plainly out-of-range values are explicit failures.</td></tr><tr><td>L2.3</td><td>semantic_synonym_drift</td><td>Substitute text that appears semantically close but changes the business interpretation used downstream.</td><td>Meaning-preserving paraphrases are not errors.</td></tr><tr><td>L2.4</td><td>temporal_staleness</td><td>Reuse a value that was valid previously but is no longer correct at the task time.</td><td>A contemporaneous neighboring numeric value belongs to L2.2.</td></tr></table>

The injection locus is annotated independently of the semantic leaf and takes one of two values: tool output, where the substitute is written into the producer observation, or tool input, where it is written into a downstream tool argument.

## B.2 LITERATURE GROUNDING

Silentness and exclusion boundaries. Tools Fail (Sun et al., 2024) establishes the central operational criterion: a tool output may be schema-valid and wrong without raising an exception. The Tool-Augmented LLM Failure Taxonomy (Winston & Just, 2025) reinforces the distinction between explicit and silent failures. The LLM Hallucination Survey (Huang et al., 2025) and Where LLM Agents Fail (Zhu et al., 2025) further separate entity, context, profile, memory, and parameter errors, motivating value-level interventions while excluding failures that arise only from the agent’s internal reasoning or memory. Other taxonomies delimit the scope from adjacent phenomena: the Multi-Agent Failure Taxonomy (Cemri et al., 2025) covers behavioral and system failures, τ-bench (Yao et al., 2025) emphasizes domain-policy compliance, CRMArena-Pro (Huang et al., 2026) adds confidentiality failures, and Reasoning Error Localization (Tyen et al., 2024) distinguishes recognizing an error from correcting its downstream effects. Together, these sources define silent error as an incorrect intermediate value rather than a planning, collaboration, policy, confidentiality, or reasoning failure.

Semantic distinctions. Data-quality and factuality research supplies the distinctions used to organize the seven leaves. The Dirty Data Taxonomy (Kim et al., 2003) separates category, value, relation, and stale-data errors, supporting category-level substitutions, attribute and numeric changes, and temporal staleness. Faithfulness versus Factuality (Maynez et al., 2020) frames an error as a contradiction with its source evidence, providing a common criterion across all leaves. The taxonomy of Factual Errors in Summarization (Tang et al., 2023) distinguishes entity, attribute, numeric, and temporal errors, which map naturally to entity near-duplicates, profile-attribute drift, numericbucket shifts, and temporal staleness.

Specialized leaves and loci. Several sources refine particular boundaries. Butterfly Effects in Toolchains (Xiong et al., 2025) shows how incorrect tool arguments propagate downstream, motivating tool input as an intervention locus in addition to tool output. Work on Duplicate Record Detection (Elmagarmid et al., 2007) motivates the distinction between real near-duplicate entities and superficially similar text. FreshLLMs (Vu et al., 2024) grounds temporal staleness in answers that change over time. Finally, LM-Emulated Sandbox Risks (Ruan et al., 2024) helps exclude system and safety failures from the semantic taxonomy.

Taken together, the agent and tool literature defines when an error is silent and what lies outside scope, while the data-quality and factuality literature supplies the semantic forms of the corrupted value. We adapt these distinctions into schema-valid, single-slot workflow interventions; the semantic leaf describes what changed, and the locus records where that change enters the trajectory.

## C EVALUATION PROTOCOL DETAILS

This appendix specifies the rollout, judging prompts, and process measures used in evaluation. All prompts, trajectories, and case-study contents in this paper are English translations.

Evaluation flow. Each rollout follows four stages:

(i) Initialize. For benchmark unit i and condition k ∈ {correct, implicit, explicit}, the query is $q _ { i }$ , the continuation environment is $E _ { i } ,$ and the condition-specific prefix $p _ { i } ^ { k }$ initializes the agent history as $h _ { 0 , i } ^ { k } = p _ { i } ^ { k }$ . The correct prefix preserves the recorded value, the implicit prefix substitutes a silently incorrect value, and the explicit prefix presents an overt failure. The agent receives the task instruction and task inputs, followed by the prefix rendered as a numbered record of thoughts, actions, and observations.

(ii) Execute. The agent follows Eq. 1 for at most 25 steps. At each step, it outputs a thought and a JSON action that either calls a tool with arguments or stops with a final answer. Each condition uses a newly initialized copy of the same workflow-specific tool registry and casespecific replay snapshot. Tool names are normalized through workflow aliases. For each call, $E _ { i }$ first searches for an argument-matched recorded response and returns it if found; repeated unmatched calls may consume the next unconsumed response in recorded order, while calls absent from or exhausted in the snapshot fall back to the live read-only toolbox. Replay hits, misses, live fallbacks, and observation lengths are recorded for audit.

(iii) Judge. GLM-5 at temperature 0 independently applies the state-judge and result-judge prompts below. The same model also judges whether an agent that faithfully uses the gold versus the polluted value would reach a different business conclusion, which defines the outputchanging slots in D1 (Section 6).

(iv) Measure. Producer refetch and Noticed are computed from the trajectory independently of both judge outputs.

Rollout counts and exclusions. Across the ten models, each condition has 1,670 scheduled rollouts; in total, 73 scheduled rollouts produced no valid record. A rollout is excluded if any of its observations contains a tool-execution error message, which indicates an environment-side anomaly; this rule removes 201 rollouts. The retained rollouts number 1,599, 1,575, and 1,562 for the correct, implicit, and explicit conditions. Noticed is computed on all retained rollouts, and RSR on the 1,576, 1,555, and 1,543 retained rollouts that received a result judgment.

Judge prompts. Braced fields denote values filled separately for each benchmark instance.

State judge. The state judge determines whether the latest valid producer observation recovers the affected evidence.

You are a strict evaluator. An intermediate slot may have been corrupted.   
Slot: {slot}; condition: {condition}.   
Gold: {gold value}.   
Polluted: {polluted value}.   
Latest observation after the agent re-invoked the producer:   
{latest observation}.   
Classify the value as gold, polluted, neither, or unknown. Differences in   
formatting do not change semantic equivalence. Return only JSON with fields used and   
reason.

The outputs gold, polluted, and neither correspond to Recovered, Residual, and Neither observation states, respectively; unknown indicates insufficient evidence for attribution.

Result judge. The result judge determines whether the final answer completes the original user task.

You are a strict evaluator. Decide only whether the agent’s final answer solves the user   
task.   
User task: {user task}.   
Correct reference evidence: {correct reference evidence}.   
New observations: {new observations}.   
Final answer: {final answer}.   
Check whether every task-critical claim is supported by either the reference evidence or a   
new observation. A contradiction unsupported by new evidence, an incomplete answer, or   
an error termination is unsolved. Return only JSON with fields solved and reason.

The solved output defines RSR. The state and result judgments are independent: recovering the affected evidence does not by itself guarantee a correct final answer.

Process measures. Producer refetch records whether the agent invokes the producer of the affected value again after that value appears in the prefix. Noticed is true when the reasoning explicitly signals a related content or result anomaly and is followed by a remedial action, such as refetching the producer, consulting a previously unused corroborating tool, or issuing an explicit refetch.

Table 7: Objects evaluated by the process measures and judge outputs. These quantities are complementary rather than successive scoring rules.
<table><tr><td>Measure</td><td>Evaluated object</td><td>Operational interpretation</td></tr><tr><td>Producer refetch</td><td>Post-prefix tool actions</td><td>Whether the agent re-invokes the producer of the affected value, regardless of what the new call returns.</td></tr><tr><td>Noticed</td><td>Reasoning plus remediation</td><td>Whether the agent explicitly identifies a related anomaly and follows it with a remedial action.</td></tr><tr><td>Observation state</td><td>Latest valid producer observation</td><td>Recovered matches gold, Residual matches the polluted value, and Neither matches neither reference value.</td></tr><tr><td>RSR</td><td>Complete final answer</td><td>Whether the answer resolves the original query using the available evidence; it is independent of observation-state attribution.</td></tr></table>

State Checker subset. The State Checker comparison in D3 (Section 6) uses 81 slots. We stratify slots by workflow and error type and take the first half of each stratum (rounded up), ordered by slot identifier. Workflow wf05 is excluded because one of its internal model components was being taken offline.

## D HUMAN VALIDATION OF RESULT AND NOTICED LABELS

This appendix reports an independent human validation of two automatically produced evaluation labels: the LLM-based Result judgment that defines RSR and the trajectory-derived Noticed label.

Recruitment and assignment. We recruited six university participants through a campus forum. The recruitment post described an approximately 50-minute annotation task, and each participant received RMB 50 upon completion. We selected 100 rollouts using stratified sampling: 28 were from the correct condition, 35 from the explicit-error condition, and 37 from the implicit-error condition, with 10 rollouts from each evaluated model. Within each stratum defined by model and condition, rollouts were drawn uniformly at random. Each rollout was independently assigned to three participants, and each participant annotated 50 rollouts, yielding 300 participant–rollout judgments. To balance overlap, each of the 20 possible three-person annotator groups was assigned five rollouts. Participants were instructed to work independently and not discuss individual cases.

Two-stage annotation protocol. For each rollout, the interface presented the user task, the correct reference evidence, the complete agent trajectory, any new tool observations, and the final answer. Annotation then proceeded in two stages. In the blind stage, participants answered two binary questions: (i) whether the final answer solved the user task (Result), and (ii) whether the agent explicitly noticed a data anomaly and subsequently invoked a tool to verify it (Noticed). A Noticed label therefore required both an explicit recognition signal in the reasoning and a later verification action. Once submitted, these labels were locked. In the review stage, the interface revealed the two automatic evaluation labels and their rationales. Participants marked each label as agree, disagree, or unsure, with an optional comment. Figure 4 illustrates the two stages. Because the second-stage responses may be influenced by the revealed rationales, we treat the blind-stage comparison as the primary validation and report review-stage agreement only as complementary evidence.

Agreement results. For the blind-stage analysis, we used the majority vote of the three independent annotations as the human reference label for each rollout. Table 8 reports raw agreement and Cohen’s κ between the human reference and the corresponding automatic label, as well as Fleiss κ among the three human annotators. Cohen’s κ adjusts raw agreement for the agreement expected from the two label distributions. Precision, recall, and F1 treat the human reference as the target and the positive automatic label as the prediction. Review agreement is the fraction of all 300 participant–rollout responses marked agree after the automatic labels were revealed.

Table 8: Human validation of the automatic Result and Noticed labels over 100 rollouts. Each rollout received three independent annotations. Blind-stage statistics provide the primary validation; postreveal agreement is complementary.
<table><tr><td>Label</td><td>Majority agreement</td><td>Cohen&#x27;s κ</td><td>Human Fleiss&#x27;κ</td><td>Precision</td><td>Recall</td><td>F1</td><td>Review agreement</td></tr><tr><td>Result</td><td>76/100 (76%)</td><td>0.318</td><td>0.477</td><td>97.1%</td><td>75.3%</td><td>84.8%</td><td>252/300 (84%)</td></tr><tr><td>Noticed</td><td>87/100 (87%)</td><td>0.739</td><td>0.569</td><td>87.0%</td><td>85.1%</td><td>86.0%</td><td>267/300 (89%)</td></tr></table>

The trajectory-derived Noticed label aligns strongly with the human reference (87/100; Cohen’s κ = 0.739). Result agreement is lower $( 7 6 / 1 0 0 ; \kappa = 0 . 3 1 8 )$ , but its high precision (97.1%) and lower recall (75.3%) indicate a conservative evaluator: positive success judgments are usually supported by the human majority, while some human-accepted rollouts are labeled unsolved.

The asymmetric Result errors help explain its lower κ: the human majority labeled 89/100 rollouts as solved, compared with 69/100 for the result judge, and 22 of the 24 disagreements followed this direction. Inspection of the rationales and comments suggests that participants sometimes accepted broadly responsive answers, whereas the result judge more strictly enforced query-specific completeness and evidence support. This strictness follows the judge’s design: an answer counts as solved only if every task-critical claim is supported by the reference evidence or a new observation, and the same judge and criterion are applied to all three matched conditions. Although human Fleiss κ for Result is 0.477, all three participants agreed on 80/100 rollouts, indicating that disagreement is concentrated in a limited set of borderline cases rather than reflecting widespread annotation instability. After the automatic labels and rationales were revealed, participants agreed with 84% of the Result judgments and 89% of the Noticed labels. Because these responses may be influenced by the revealed rationales, we do not treat them as independent evidence of accuracy; instead, they show that most automatic decisions were acceptable to participants once explained. Taken together, the blind and review stages provide more consistent support for Noticed, while Result disagreement mainly reflects the evaluator’s stricter success criterion.

![](images/cdfc6d992f299cde940ccd24e6c78e10ce8e73e4ee4c89362a72c60b7be94213.jpg)  
(a) Blind annotation. Automatic evaluation labels are unavailable, and the participant independently labels Result and Noticed.

![](images/b84373ab4b965f9b7715c313ff154efb673848fa39bd89480b83f96c26875ded.jpg)  
(b) Post-reveal review. The interface shows each automatic evaluation label and its rationale for agreement assessment.  
Figure 4: English translation of the original Chinese two-stage annotation interface used in the human-validation study. Blind labels were locked before the automatic evaluation labels and rationales were revealed.

## E END-TO-END CASE STUDIES OF SILENT-ERROR HANDLING

This appendix presents two baseline case studies, not State Checker interventions. In Case 1, the agent successfully refetches the corrupted producer but later overwrites the recovered evidence during reasoning. Case 2 follows an Unnoticed implicit error from a plausible category substitution to a wrong final recommendation.

How to read the case boxes. Entries labeled P\* summarize inherited prefix events shown to the agent; entries labeled R\* are continuation steps generated by the evaluated model. Each step lists the agent action, the observed result, and a diagnostic note. Producer refetch records whether the mode re-invokes the producer of the affected value. Observation state describes the latest valid producer observation as Recovered, Residual, or Neither, while final task success separately asks whether the complete answer resolves the user query.

## E.1 CASE 1: PRODUCER REFETCH SUCCEEDS BUT FINAL RECOVERY FAILS

The setup box makes the instance, intervention, and expected evidence explicit before presenting the trajectory.

<table><tr><td>Case 1: Setup</td></tr><tr><td>Model and condition: Qwen3.6-35B; baseline implicit condition</td></tr><tr><td>Workflow: wf09, live-stream assistant-data extraction</td></tr><tr><td>Affected producer: BUILDCOMMENTREQUEST</td></tr><tr><td>User task: Extract parent interactions and assistant speech from the opening warm-up stage, then use that stage&#x27;s end time to determine the sentiment of the final interaction.</td></tr><tr><td>Task input: feed_id=anonymized; start_timestamp=1787916387; end_timestamp=1787920006</td></tr><tr><td>Gold slot value: start_sec=1787916387; end_sec=1787920006</td></tr><tr><td>Implicit-prefix value: start_sec=1787916527; end_sec=1787920146</td></tr><tr><td>Why the error is silent: The shifted interval has the expected schema and plausible timestamps, and the prefix contains no warning or competing value.</td></tr></table>

## Case 1: Prefix and Rollout

<table><tr><td colspan="2">Inherited prefix supplied to the model</td></tr><tr><td colspan="2">P1 Call GETFEEDDASHBOARD.</td></tr><tr><td rowspan="4">P2</td><td>Result: The recorded observation is empty.</td></tr><tr><td>Diagnosis: No useful stage evidence is established before the affected slot.</td></tr><tr><td>Call BUILDCOMMENTREQUEST with the original feed identifier and timestamps.</td></tr><tr><td>Result: The prefix reports start_sec=1787916527 and end_sec=1787920146. Diagnosis: Silent corruption enters the history as a plausible request interval.</td></tr><tr><td colspan="2">Model-generated continuation</td></tr><tr><td rowspan="2"></td><td>R1 Reuse the inherited request in FETCHALLCOMMENTSRPC.</td></tr><tr><td>Result: The RPC returns an argument error because the inherited request omits the feed identifier. Diagnosis: A downstream explicit error appears, but the shifted interval itself is not yet</td></tr><tr><td rowspan="2">R2</td><td>corrected.</td></tr><tr><td>Attribute the failure to the missing identifier and refetch BUILDCOMMENTREQUEST with the task-level timestamps. Result: The tool returns the feed identifier with start_sec=1787916387 and</td></tr><tr><td rowspan="2">R3</td><td>end_sec=1787920006. Diagnosis: Producer refetch succeeds and restores the gold slot value.</td></tr><tr><td>Fetch comments with the corrected request.</td></tr><tr><td rowspan="2"></td><td>Result: The call completes. Boundary records include 1787919998 (“Can Teacher J. dance?"), 1787920002 (“Teacher J."), and 1787920006 (“OK, OK, OK").</td></tr><tr><td>Diagnosis: Correct downstream evidence is acquired through the recovered end time.</td></tr><tr><td rowspan="2">R4</td><td>Construct an ASR request using the corrected interval.</td></tr><tr><td>Result: The request preserves the recovered start and end boundaries. Diagnosis: The model initially carries the repaired evidence into a second task branch.</td></tr><tr><td>Case 2: Setup</td></tr><tr><td></td></tr><tr><td>Model and condition: GLM-5.2; baseline implicit condition</td></tr><tr><td>Workflow: wf06, cross-domain intent and interest modeling Affected evidence: A content category in the user's article-reading history</td></tr></table>

<table><tr><td>R5</td><td>Invoke the ASR RPC. Result: The call completes but returns no speech payload. Diagnosis: Assistant-speech evidence remains unavailable, increasing pressure to use a</td></tr><tr><td>R6</td><td>fallback. Parse the full comment sequence. Result: The parser reports 482 records, including early comments and final records at 1787920002 and 1787920006.</td></tr><tr><td>R7</td><td>Diagnosis: The recovered boundary remains supported by observable comments. Invoke candidate segmentation for the opening stage. Result: The tool returns no candidate segments.</td></tr><tr><td>R8</td><td>Diagnosis: Missing stage annotations trigger fallback reasoning, not evidence that the recovered interval is wrong. Reparse a compact boundary set. Result: Ten boundary comments are returned, again including the final interaction at</td></tr><tr><td>R9</td><td>1787920006. Diagnosis: Correct endpoint evidence remains available immediately before the decisive error. Replace the recovered boundary with a self-defined ten-minute warm-up window. Result: The model selects the earlier “[strong]&quot; reaction at 1787916453 and labels it positive.</td></tr><tr><td>Reported answer</td><td>Diagnosis: Reasoning overwrite: a heuristic supersedes recovered evidence and causes the wrong final interaction.</td></tr><tr><td colspan="2">The model reports “yeah&quot; and “[strong][strong][strong]&quot; as the opening-stage parent interactions, states that assistant speech is unavailable, and identifies “[strong][strong][strong]&quot;as the final inter- action with positive sentiment. The answer is not copied directly from the polluted prefix; it fails after</td></tr></table>

## Case 1: Prefix and Rollout (continued)

## Case 1: Judgment

Producer refetch: Yes. R2 re-invokes BUILDCOMMENTREQUEST.

State judge: Recovered. R2 returns both task-provided timestamps exactly.

Final task success: Unsolved. Later comments exist through 1787920006, but the answer selects an earlier interaction from a heuristic window.

Observation recovery and final task success intentionally evaluate different objects.

State progression: Prefix (plausible corrupted interval) → verification (correct interval) → downstream use (correctly fetched comments) → reasoning (heuristic ten-minute window) → answer (wrong final interaction). The failure occurs after successful evidence reacquisition.

Failure mechanism. The decisive error is a reasoning overwrite. The model obtains and initially uses the correct interval, but later treats missing stage annotations as permission to invent a generic boundary. Producer refetch is therefore necessary but insufficient: recovery also requires preserving the authority of corrected evidence over later derived decisions.

## E.2 CASE 2: AN UNNOTICED CATEGORY ERROR PROPAGATES TO THE ANSWER

The second case shows the complementary failure mode: the model never questions or refetches the affected producer, so the polluted value remains the organizing premise throughout the continuation.

## Case 2: Setup (continued)

User task: Use interests in CATL news and AI side-income content to analyze the profile, generate live-stream recommendation keywords, and choose directly between the finance and pet domains.

Gold category: First level: Finance; second level: Stocks

Implicit-prefix category: First level: Pet; second level: Cat

Unchanged context: Article identifier, timestamp, scene, and read time=115920 remain unchanged; the model sees no competing Finance/Stocks value.

Why the error is silent: Pet/Cat is well formed and locally plausible, and no warning accompanies the article record.

## Case 2: Prefix and Rollout

## Inherited prefix supplied to the model

P-context Receive the normal profile and live- and short-video behavior history. Result: The user is a middle-aged man in a large city; no prior record establishes a pet-domain preference. Diagnosis: Background context does not independently support the later Pet/Cat conclusion.

P-error Receive an article record with category1=Pet, category2=Cat, and read time=115920. Result: The record is schema-valid and carries no error signal. Diagnosis: A Finance/Stocks record is silently replaced by Pet/Cat at the affected slot.

## Model-generated continuation

R1 Treat the inherited category and long reading time as strong evidence of pet interest; request short-video summaries. Result: The only positive pet evidence is the polluted article category. Diagnosis: The corrupted value becomes the initial domain hypothesis.

R2 Interpret comedy, daily-life, and public-speaking summaries as showing no finance or AI interest; classify the user as pet-domain and request recommendation terms. Result: The new summaries contain no direct pet evidence. Diagnosis: Confirmation through absence: non-finance evidence is misread as positive support for Pet/Cat.

R3 Accept the encoder output and stop. Result: The encoder returns the user’s city, pet snacks, canned cat food, pet food, cat litter boxes, cat litter, and related terms. Diagnosis: A category-level error is concretized into product-level recommendations.

## Reported answer

The final answer classifies the user as belonging to the pet domain, cites the Pet/Cat article record with read time=115920 as its primary justification, and recommends pet-food and cat-supply keywords.

## Case 2: Judgment

Noticed: No. No thought questions Pet/Cat or requests verification of the article record.

Producer refetch: No. No continuation step revisits the source of the article category.

Observation state: Residual. The model continues to rely on the inherited Pet/Cat record.

Final task success: Unsolved. The pet-domain conclusion contradicts Finance/Stocks, and no new observation supports Pet/Cat.

State progression: Prefix (Finance replaced by Pet) → adoption (Pet treated as strongest interest) → interpretation (absence of finance videos used as confirmation) → tool output (pet-product keyword generation) → answer (wrong domain recommendation). The polluted value shapes every later interpretation.

Failure mechanism. The important transition occurs in R2. The additional evidence is nonfinancial, not positively pet-related, but the inherited category has already established a pet hypothesis. The model therefore interprets absence of finance content as confirmation rather than insufficient evidence. This is the propagation pattern targeted by the benchmark: a locally plausible error becomes an organizing premise, shapes later evidence interpretation, and determines the recommended action.

## F DATASET DISTRIBUTION

This appendix reports the distribution of the 167 paired units underlying Section 4.3. Each unit is one retained slot–locus pair from a clean workflow execution and expands into three conditions, yielding 501 instances. Because one execution may contribute multiple slots or loci, all counts below refer to paired units rather than workflow runs.

Figure 5 shows the distribution by workflow on the left and by silent-error type on the right. Both panels use the same vertical scale and label every bar with its exact count.

![](images/535acf1e17fc9fcfb3b55f39bd290bb9febf2366078d9c9b80f5af962d461bdc.jpg)

![](images/de614268d1f89915f96d65815ce683b12ab9133d4a95e6df8612b1b13e3f0f6d.jpg)  
Figure 5: Distribution of the 167 paired units. Left: units contributed by each workflow. Right: units per silent-error type; blue bars denote object or category errors and orange bars denote attribute or value errors. Numeric labels give exact counts, and temporal staleness has no retained instance.

All nine workflows are represented: wf06 contributes 52 units, and each remaining workflow contributes between 9 and 22. Of the 167 units, 147 place the intervention at tool output and 20 at tool input.

Across error types, semantic synonym drift, profile attribute drift, numeric bucket shift, entity nearduplicate, category sibling drift, and category granularity mismatch contribute 40, 35, 32, 23, 20, and 17 units, respectively. Temporal staleness has no retained instance.

These counts account for all 167 paired units and provide the denominators for per-workflow and per-type analyses.

## G LIMITATIONS

Several limitations qualify our findings. First, each silent-error instance contains a single corrupted value, and re-querying its producer returns the unpolluted response; RITE therefore measures recovery when a trustworthy verification path is available, and faults that persist across re-queries are left to future work. Second, recorded calls are replayed, whereas valid calls absent from the recordings are served by read-only live tools, so continuations that leave the recorded paths may observe current tool behavior. Third, temporal staleness is defined in the taxonomy but not instantiated. Finally, RSR relies on an LLM judge, whose errors may affect absolute scores.