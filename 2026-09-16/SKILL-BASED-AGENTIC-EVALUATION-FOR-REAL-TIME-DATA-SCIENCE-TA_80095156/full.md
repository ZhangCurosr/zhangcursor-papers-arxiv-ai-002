# SKILL-BASED AGENTIC EVALUATION FOR REAL-TIME DATA SCIENCE TASKS

A PREPRINT

Aniruddha Tamhane, Raghavendra Addanki, Ayushi Aggarwal, Aditya Bansal, Rui Wang, Charles Menguy, Swati Jain Adobe atamhane@adobe.com

## ABSTRACT

We present a framework for evaluating data-science agents on live, continuously updated data using executable ground truth and format-agnostic factoid scoring. Consider this example query: “what were last week’s audience sizes”—the reference answer changes as the underlying data changes, so static references become outdated and standard LLM-as-a-judge pipelines cannot verify responses against a fixed ground truth. Our central contribution, ground-truth-as-code, encodes each expected answer as an executable reference function that recomputes the answer directly from live data at evaluation time, ensuring the reference remains consistent with the system it describes. We combine this with a factoid-level, format-agnostic judge that decomposes both the agent’s response and the computed ground truth into atomic claims and scores precision, recall, and accuracy over them, irrespective of the response format (prose, list, table, HTML, etc.). The approach is applicable to agents whose expected outputs can be expressed as executable data computations. We validate the framework through a human–LLM agreement study on an internally developed machine learning skill deployed in production, using a synthetic database constructed to reproduce production schemas and entity relationships. Relative to a natural-language ground-truth baseline, our method achieves a 29% improvement in the Matthews Correlation Coefficient (MCC)—a class-balanced measure of agreement between expert annotators and LLM-as-a-judge predictions—and a 16% reduction in token consumption per test case, while a self-directed baseline lacking explicit ground truth is anticorrelated with human judgment. Agents that perform multi-source data integration and computation over non-stationary data are routinely deployed in industry; we propose ground-truth-as-code as a practical methodology for their evaluation.

## 1 Introduction

We present a framework for evaluating data-science agents whose outputs depend on live, continuously updated data, combining executable ground truth with format-agnostic factoid scoring. We motivate this design with a concrete example.

Consider a marketing analyst who queries an agent for last week’s best-performing campaign segments. The correct answer depends on the current state of the data warehouse—which rows have been ingested, which schema is active, which tables must be joined—rather than on the answer that was correct at benchmark-authoring time. A frozen reference answer is therefore stale by construction: by the time it is annotated, the underlying data has already changed the correct response.

Agentic AI systems are increasingly entrusted with precisely this class of deep data computation tasks—multisource integration, multi-step analysis, and reasoning over large, continuously updated databases—and now underpin production data science workflows across industry [Nam et al., 2025, Sun et al., 2025a, Chen et al., 2025]. Evaluating such agents is qualitatively harder than standard LLM evaluation: the property that makes them valuable, operating over live and changing organizational data, violates the time-invariance assumption underlying existing benchmarks and LLM-as-a-judge pipelines, leaving the absence of time-varying ground truth a primary open challenge [Chen et al., 2026a] (Section 2).

We address this gap with two ideas, developed in turn below. First, since the reference answer to a live-data query is fundamentally a computation rather than a static fact, we encode it as such: each expected answer is expressed as a runnable Python function—ground-truth-as-code—executed against the live system at evaluation time to recompute the answer from first principles (Section 3.2). Because the reference is recomputed from the same data state observed by the agent, it remains invariant to data drift, and the function signature serves as a schema contract that fails explicitly when an upstream API changes.

Second, an agent may report identical underlying facts as prose, a table, or a bulleted list, and a single holistic LLM-asa-judge score conflates presentation quality with the correctness of the underlying data. We instead have an LLM-based judge decompose both the agent’s response and the computed ground truth into atomic claims (factoids) and score precision, recall, and accuracy over the matched claims (Section 3.5); because the scoring layer never observes the raw response format, the measure is format-agnostic by construction, and hallucinated or omitted claims are directly attributable in the scorecard. The complete pipeline is implemented as a harness skill, inheriting the harness’s native engineering features (Section 3).

We focus on data-science tasks for which the reference answer can be computed from accessible data APIs; extending the approach to settings without such APIs remains outside our current scope.

We validate the framework through a human–LLM agreement study (Section 5) on an internally developed, in-production ML skill, using a synthetic database that reproduces the schema, entity relationships, and advancing temporal structure of production data (Section 4). Relative to a natural-language ground-truth baseline, ground-truth-as-code improves the Matthews Correlation Coefficient (MCC)—which quantifies, robustly to class imbalance, how closely the LLM judge’s PASS/FAIL verdicts align with expert labels—by 29% (0.427 vs. 0.331) and reduces token cost per test case by 16%, while a self-directed baseline lacking explicit ground truth is anti-correlated with human judgment (MCC = −0.379).

We contribute: (i) ground-truth-as-code, a technique for constructing drift-resistant reference answers for evaluation on live data; (ii) a factoid-level, format-agnostic scoring framework that separates factual correctness from presentation; and (iii) a validation study against expert judgments on a production-deployed ML skill, showing higher human agreement and lower token cost than natural-language and self-directed baselines.

## 2 Related Work

Data Science AI Agents. Transformer-based [Vaswani et al., 2017] Large Language Models [Brown et al., 2020, White et al., 2024] mimic human verbal abilities including conversation, step-by-step reasoning [Wei et al., 2022], and planning [Yao et al., 2022]. AI agents extend this paradigm by granting LLMs agency through tools, APIs, and MCPs [Hou et al., 2025], and through static or dynamic networks of sub-agents. Agentic harnesses [Pan et al., 2026] consolidate these capabilities into a standardized substrate that can be specialized via skills [Jiang et al., 2026]—bundles of markdown, scripts, connectors, and tools that augment the harness while reusing its native engineering features (sub-agent orchestration, memory and context management, etc.). Within this paradigm, production data science workloads span data pulling, aggregation, visualization, and forecasting through to extensive reports and proactive recommendations requiring deep analysis planning, model training, inference, and summarization [Nam et al., 2025, Sun et al., 2025a, Chen et al., 2025]. Narrower workflows with strong consistency requirements adopt rigid agentic patterns, with LLMs acting as routers [Guo et al., 2024] or executors of a high-level plan [Hong et al., 2024]; embedding planning in the orchestration loop enables open-ended task solving [Fu et al., 2025, Sun et al., 2025b], and advanced systems such as Nam et al. [2025] explore multiple concurrent analyses, each with independent planning. Chen et al. [2026a] provide a concurrent survey of evaluation tools for data science AI assistants, identifying the lack of time-varying ground truth as a primary open challenge.

Agent Evaluation. Evaluating advanced agentic capabilities requires test-benches of complex tasks annotated for correctness. GAIA [Mialon et al., 2023] pioneered this with a corpus of real-world question–answer pairs demanding complex planning, execution, and information processing. Data-driven exploration and analysis tasks were curated by Nie et al. [2026], Lai et al. [2025], while Egg et al. [2025] targeted multi-turn problem solving with a factoid-level evaluation setting. Chen et al. [2026b] evaluate data science agents across a broad range of open-ended real-world tasks, explicitly noting that exact-match and string-overlap metrics are insufficient for such settings. These works informed our test-bench design; their common limitation is the absence of time- or environment-dependent ground truth.

Execution-Based Ground Truth. A foundational precedent for our ground-truth-as-code technique is the execution accuracy paradigm established in text-to-SQL evaluation. WikiSQL [Zhong et al., 2017] was among the first to evaluate NL-to-code by executing generated SQL and comparing outputs rather than matching query strings, establishing that code correctness is best measured through its observable effect. Spider [Yu et al., 2018] scaled this across 200 complex databases, and BIRD [Li et al., 2023] extended it to real-world enterprise settings requiring external knowledge reasoning. Spider 2.0 [Lei et al., 2024] pushes further into enterprise agentic workflows where even frontier models solve fewer than one in five tasks. We transfer this execution-accuracy philosophy to open-ended data science agent responses on live production data, generalizing beyond SQL to arbitrary Python computations.

LLM-as-a-Judge and Atomic Factoid Evaluation. Agentic responses are inherently unstructured, motivating LLMas-a-judge approaches [Gu et al., 2024, Zheng et al., 2023], though these have known limitations on specialized content [Szymanski et al., 2025]. Agents-as-Judge [Zhuge et al., 2024] extends the idea by granting the judge limited interaction capabilities. Factoid-level evaluation—decomposing free-text into atomic claims and scoring each independently—was established by FActScore [Min et al., 2023], which achieves near-human factual precision measurement on long-form generation. SAFE [Wei et al., 2024] extends this by deploying an LLM sub-agent to verify each atomic fact via search, introducing sub-agent parallelism as a natural implementation pattern. Jafari et al. [2026] and Fan et al. [2025] target factoid-level precision and recall directly; our framework extends these ideas with conditioning on the user query when extracting ground-truth factoids (Eq. 1). Retrieval-augmented generation (RAG) evaluation frameworks occupy an adjacent but architecturally distinct niche. RAGAS [Es et al., 2023] decomposes RAG pipeline quality into four LLM-judged metrics—Faithfulness, Answer Relevancy, Context Precision, and Context Recall—each requiring the retrieved context chunks as an explicit input; the framework is thus inapplicable to agents that produce answers through API calls or code execution rather than document retrieval. ARES [Saad-Falcon et al., 2023] similarly targets RAG pipelines, fine-tuning lightweight LLM judges on synthetic preference data to score Context Relevance, Answer Faithfulness, and Answer Relevance, using prediction-powered inference for statistical confidence intervals. Both systems assume a static document corpus and have no mechanism for ground truth that must be re-executed against a live data system. Our framework extends the factoid-decomposition spirit of these approaches to the agentic data science setting, replacing retrieved-context fidelity with executable-code ground truth evaluated against production data at inference time.

Dynamic and Contamination-Free Evaluation. Static benchmarks face two compounding failure modes. First, ground truth derived from world-state facts degrades as production data drifts: Margatina et al. [2023] demonstrate this with temporal concept drift in language model benchmarks, and Shi et al. [2025] quantify the resulting score inflation for static factuality test sets. Second, training-data contamination of widely-shared benchmarks systematically overstates model capability [Xu et al., 2025]. Live-data evaluation frameworks such as LiveBench [White et al., 2024] and PolyBench [Arora et al., 2026] partially address contamination by continuously refreshing questions from real-world streams, but retain fixed answer verification logic that cannot accommodate changing production schemas. Our ground-truth-as-code approach addresses both failure modes: the Python function is re-executed at evaluation time against live data, making both the answer and its verification inherently drift-resistant.

## 3 Implementation

## 3.1 Framework Overview

The pipeline assesses factual equivalence between the agent’s response and a reference ground truth, conditioned on the user query and on conversational expectations (conciseness, coverage, tone); Figure 1 shows the full flow. Given a skill S under evaluation and an agentic harness $H$ , we build a test-bench $T = \{ \tau _ { i } \} _ { i = 1 } ^ { t }$ of tuples $\tau _ { i } = ( U _ { i } , G _ { i } , \mathcal { E } , \mathcal { C } )$ : the user query $U _ { i }$ , the ground-truth-as-code function $G _ { i }$ , environment variables $\mathcal { E }$ (business context, co-resident skills), and compute variables $\bar { \boldsymbol { \mathcal { C } } }$ (credentials, timeouts, API endpoints). Crucially, the reference ground truth is never precomputed or stored: for each $\tau _ { i }$ the harness runs $S$ and executes $G _ { i }$ live at evaluation time, in parallel and against the same current state of the data system, to collect the response $r _ { i }$ and the freshly computed ground-truth output $g _ { i }$ , then compares them at the level of atomic factual claims.

## 3.2 Ground-Truth-as-Code

Each test case encodes its expected answer not as a static string but as a typed Python function $G _ { i } : { \mathcal { E } } \to { \mathcal { D } }$ , where D maps factoid keys to their expected values. At evaluation time, $\bar { G } _ { i }$ is executed within the compute environment specified by C, which connects to the live data system and pulls real-time data from the cloud to produce $g _ { i } = G _ { i } ( \mathcal { E } )$ . Because g<sub>i</sub> is computed from the same current data state as the response $r _ { i } ,$ the reference resists data drift; and being ordinary code, $G _ { i }$ is versioned with the test-bench in source control, giving the same per-change auditability as production code.

![](images/a0c32498fd5454b48db8454062fb741b337851612514eebeef33564eeaabf49d.jpg)  
Figure 1: Evaluation flow for a single test case $\tau _ { i } = ( U _ { i } , G _ { i } , \mathcal { E } , \mathcal { C } )$ . Both the skill response $r _ { i }$ and the ground-truth output $g _ { i }$ are produced at evaluation time against live data. Factoids are extracted from each, matched semantically, and aggregated into a per-test-case scorecard.

## 3.3 Factoid Decomposition

A factoid is an atomic claim—the smallest unit of information independently verifiable as true or false. Let $\Phi ( \cdot )$ map a string to its set of atomic claims; we apply it asymmetrically to the response and ground-truth:

$$
{ \mathcal { F } } _ { r } = \Phi ( r _ { i } ) , \qquad { \mathcal { F } } _ { g } = \Phi ( g _ { i } \mid U _ { i } ) .\tag{1}
$$

The response operator is unconditioned—a well-behaved agent surfaces only facts relevant to $U _ { i } ,$ so any extraneous claim in $r _ { i }$ is itself a scoreable signal—while the ground-truth operator is conditioned on $U _ { i } ,$ keeping only the factoids in $g _ { i }$ that bear on the query. Extraction isformat-independent: regardless of the surface format of $r _ { i } , \mathcal { F } _ { r }$ contains the same atomic claims. This is the source of the framework’s style agnosticism—the scoring layer never observes the raw response format, only the extracted factoids.

## 3.4 Factoid Matching

Matching is performed by the agentic harness, which attempts a one-to-one mapping between the response factoids ${ \mathcal { F } } _ { r }$ and the ground-truth factoids $\mathcal { F } _ { g }$ (the latter already conditioned on the user query $U _ { i } ) ; \mathcal { F } _ { r g }$ is the resulting set of matched factoids. Each match is a binary decision (no partial credit), so a matched factoid is unambiguously correct and an unmatched one unambiguously wrong, preserving the interpretability of the metrics below.

## 3.5 Metrics

Factoid-level metrics. We define accuracy (A), precision $( P ) _ { \mathrm { { \ell } } }$ , and recall $( R )$ :

$$
\begin{array} { r l } & { A = \frac { \displaystyle \left. \mathcal { F } _ { r g } \right. } { \displaystyle \left. \mathcal { F } _ { g } \right. + \displaystyle \left. \mathcal { F } _ { r } \right. - \displaystyle \left. \mathcal { F } _ { r g } \right. } , } \\ & { P = \frac { \displaystyle \left. \mathcal { F } _ { r g } \right. } { \displaystyle \left. \mathcal { F } _ { r } \right. } , \qquad R = \frac { \displaystyle \left. \mathcal { F } _ { r g } \right. } { \displaystyle \left. \mathcal { F } _ { g } \right. } . } \end{array}\tag{2}
$$

P is the fraction of agent-stated facts that are correct (low P indicates hallucination), R the fraction of expected facts surfaced (low R indicates omission), and A the overall accuracy over factoids. The metrics are interpretable by construction: each scorecard logs $\mathcal { F } _ { r } \backslash \mathcal { F } _ { r g }$ (hallucinated claims) and $\dot { \mathcal { F } } _ { g } \setminus \mathcal { F } _ { r g }$ (omitted facts) verbatim, enabling direct inspection of failure modes.

Qualitative dimensions. Beyond factual overlap, a rubric-guided LLM judge scores three further dimensions: Question Coverage, how completely the explicit sub-questions or action items in $U _ { i }$ are addressed by $\boldsymbol { r } _ { i } ;$ Tone, the politeness and professionalism of $\boldsymbol { r } _ { i } ;$ and Conciseness, the verbosity of $r _ { i }$ relative to the information conveyed. Each rubric anchors its scale to reduce intra-judge variance across test cases and agent versions.

For consistency across all dimensions, every factoid-level and qualitative metric is reported on a common 0–9 scale, obtained by a simple linear rescaling of the underlying [0, 1] score (i.e., multiplying by 9).

Table 1: Vertical-wise summary statistics for the primary 6-month synthetic benchmark bundles. Each bundle contains experience events (EE), customer/patient profiles, and a domain catalog deployed on the stage sandbox.
<table><tr><td>Vertical</td><td>EE Records</td><td>Profiles</td><td>Catalog</td><td>Window</td></tr><tr><td>Retail</td><td>17,825</td><td>10,000</td><td>50</td><td>26wk</td></tr><tr><td>Fin. Services</td><td>49,651</td><td>10,000</td><td>12</td><td>26wk</td></tr><tr><td>Healthcare</td><td>285,854</td><td>10,000</td><td>20</td><td>26wk</td></tr><tr><td>Total</td><td>353,330</td><td>30,000</td><td>82</td><td></td></tr></table>

## 4 Synthetic Data Generation

To validate the framework with known ground truth, we construct a reproducible synthetic database that mirrors the schema, entity relationships, and continuously advancing temporal structure of the production customer data. It is deployed on a live stage sandbox behind the same APIs the agent and ground-truth functions $G _ { i }$ must query, mirroring the query-then-execute pathway of production rather than a static local mock—so moving to production data requires only a credentials swap. A Poisson–DAG engine generates user-behavior events, customer profiles, and a domain catalog across three verticals (retail, financial services, healthcare), each over a rolling 26-week window calibrated to the production system’s event rates, cohort sizes, and catalog cardinalities (Table 1). Heterogeneous domain vocabularies (e.g., “prescription fill” vs. “cart abandonment”) stress the ML-goal identification and dataset-discovery sub-tasks of Section 5.

Profiles use fixed seeds and realistic within-vertical distributions—credit tiers and income bands (financial), insurance types and chronic-condition flags (healthcare), loyalty tiers (retail)—and a realism layer injects column confounds (plausible but semantically misleading column names) to prevent trivially correct schema matches.

Decoy datasets. Because naïve retrieval can succeed by lexical overlap alone, we inject decoys confusable with a correct bundle along a single dimension yet unsuitable for the full ML pipeline: (i) a profile-only snapshot missing the events and catalog needed for training; (ii) an events-only extract that cannot be joined to profiles; (iii) a cross-vertical billing dataset whose event names overlap financial terminology; (iv) a mislabeled legacy dataset claiming a 6-month window but holding only 2 weeks; and (v) a marketing audience export overlapping the retail profile namespace. We further add 2-week-window bundles across five verticals that share schema with their 6-month counterparts but are too short for the propensity tasks, trapping agents that ignore data recency. Satisfying a ground-truth function requires identifying all three datasets of a bundle—events, profiles, and catalog—so any decoy selection incurs a precision penalty.

## 5 Human–LLM Correlation Study

Test-bench. We validate the LLM-as-a-judge component on 55 single-turn test cases targeting two sub-tasks of an in-house data science skill—ML-goal identification and dataset discovery from an abstract user request—backed by the synthetic database of Section 4. Each case pairs a user request tied to a product-level business goal (e.g., “I’d like a loan application propensity model to predict which customers will seek a personal loan next quarter”) with ground-truth code encoding ml\_goal, search\_terms, and expected\_datasets; environment and compute variables are inherited from the test skill. Of the 55 cases, 53 are comparable for the pass/fail analysis.

Experimental setup. In our experiments the agentic harness is Claude Code and the underlying LLM is Claude Sonnet 4.6, used both by the skill under test and by the LLM-based evaluation operators (e.g., factoid extraction and the rubric-guided qualitative judge). The framework is nonetheless model- and harness-agnostic: any sufficiently capable LLM and any agentic harness can be substituted without changes to the test-bench or metrics.

Skill under test. The skill under evaluation is an in-house machine learning skill that parses the abstract user query, discovers the relevant datasets from the live data system, and constructs the input variables and target features for a downstream propensity model—all expressed as executable Python code.

Annotation. After a calibration round, three domain-expert annotators independently rate each response against the ground truth, with live-data access for fact verification, on (a) a 5-point Likert per atomic factoid and (b) a 5-point Likert across three qualitative dimensions (Coverage, Conciseness, Tone). The LLM judge emits an execution score exec\_score $\in [ 0 , 9 ]$ ; human ratings are rescaled to the same range via $( x - 1 ) \times 9 / 4$ 4.

Study quality. Prevalence-adjusted Gwet’s $\mathrm { A C _ { 2 } }$ clears the 0.667 acceptability threshold at both the factoid level (0.931) and the stacked-dimension level (0.883), confirming the annotations are trustworthy (Table 2).

Table 2: Inter-annotator agreement (Gwet’s AC , prevalence-adjusted) at the factoid level and over all dimensions; both clear the 0.667 acceptability threshold.
<table><tr><td>Level</td><td>AC2</td></tr><tr><td>All dimensions (overall)</td><td>0.883</td></tr><tr><td>Factoid-level</td><td>0.931</td></tr></table>

Table 3: Baseline comparison against the three-annotator human reference (53 TCs): PASS/FAIL agreement (MCC), precision, recall, F1, and mean per-test-case token cost.
<table><tr><td>Configuration</td><td>MCC</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>Tokens/TC</td></tr><tr><td>Baseline-1: No GT</td><td>-0.379</td><td>0.238</td><td>0.200</td><td>0.217</td><td>~70k</td></tr><tr><td>Baseline-2: NL GT</td><td>0.331</td><td>0.561</td><td>0.920</td><td>0.697</td><td>29.2k</td></tr><tr><td>GT-as-code (ours)</td><td>0.427</td><td>0.568</td><td>1.000</td><td>0.724</td><td>~24.6k</td></tr></table>

## 5.1 Baseline Comparison

We compare ground-truth-as-code against two baselines, each weakening a different aspect of the reference given to the agentic-harness evaluator (Table 3). We report the Matthews Correlation Coefficient (MCC)—a single-figure correlation score, insensitive to skewed class distributions, capturing the alignment between expert annotations and the LLM judge’s PASS/FAIL predictions—alongside precision, recall, and F1.

Baseline-1 (No GT) provides no explicit ground truth at all. Instead, it repurposes the agentic harness’s native skillcreator skill—a built-in component for authoring and evaluating skills—with a prompt that nudges it to study the skill under test and devise its own criteria for correctness. This baseline isolates the merit of supplying any explicit ground truth to a harness-based LLM evaluator: if a capable harness can reliably infer correctness on its own, an externally provided reference would be redundant.

Baseline-2 (NL GT) supplies the same expected answer as our ground-truth-as-code, but expressed as natural-language instructions rather than runnable code—a textual specification of how to compute the answer and what to expect, which the evaluator must interpret and act on itself. This baseline isolates the merit of generating the ground truth in a structured, executable form (code) as opposed to an unstructured natural-language set of instructions.

The two baselines underperform ground-truth-as-code in distinct, instructive ways (exact figures in Table 3). Baselineconsistently evaluated each response only against the final-answer checks explicitly declared as final checks in the skill under test; it never formulated an alternate pathway to compute the ground truth and thus never accessed the realtime data. Grading the skill against its own asserted behavior leaves its agreement anti-correlated with human judgment, and at the highest token cost, since self-devising a checking plan is token-heavy. Baseline-2 did invoke the correct API function, but over-specified some of its input parameters; the resulting reference carried extra factoids that occasionally confounded the LLM judge, depressing precision while still incurring the token cost of generating a natural-language reference and invoking the skill. Ground-truth-as-code instead recomputes the reference deterministically against live data, attaining the best human agreement (highest MCC and F1, full recall) at the lowest token cost. The most precise and token-efficient ground-truth computation is therefore achieved through the ground-truth-as-code implementation.

## 6 Conclusions

In conclusion, we present a skill-based agentic evaluation framework for data science AI agents operating on real-time data, addressing challenges that static offline benchmarks cannot: temporally dynamic ground truth and format-agnostic fact verification. Ground-truth-as-code makes the reference drift-resistant by executing against live data at evaluation time. Style-agnostic factoid-level metrics separate factual correctness from response format, and the per-factoid scoring is interpretable by construction: developers can read off hallucination rates, omission rates, and failure-clustering by fact type directly from the scorecard. The harness-skill implementation delivers concurrent, logged evaluation that evolves alongside the systems it assesses, and the actionable per-factoid verdicts close the loop from measurement to targeted improvement. In our study, this executable reference agreed with human annotators more closely than natural-language or self-directed baselines while costing fewer tokens, evidencing the practical payoff of ground-truth-as-code.

## Limitations

We discuss the limitations of our study below and, where applicable, propose mitigations for future work.

Self-judging bias. In our experiments the same underlying LLM (Claude Sonnet 4.6) powers both the skill under test and the LLM-based evaluation operators (factoid extraction and the rubric-guided qualitative judge), which in principle exposes the evaluation to self-preference bias, in which a judge favors outputs from its own model family. We argue that this risk is substantially mitigated by our design: the judge does not score free-form responses holistically but is driven by a clearly defined rubric that quantifies correctness against an externally computed, executable ground truth at the level of atomic factoids. Grounding every verdict in rubric-anchored, factoid-level comparisons against ground-truth-as-code leaves little room for stylistic self-preference to influence the score. Nonetheless, a systematic study that varies the judge model independently of the skill model remains valuable future work.

Completeness of the ground-truth code. The correctness guarantees of the framework rest on the assumption that the ground-truth-as-code function $G _ { i }$ generates correct and complete reference data for all sub-queries implied by the user request $U _ { i } .$ . If $\mathrm { \Delta } G _ { i }$ omits a relevant sub-query or computes it incorrectly, the resulting factoid-level metrics will misjudge the agent accordingly. Ensuring the completeness and correctness of the ground-truth code is therefore a precondition for reliable evaluation, and currently rests on expert authoring and review.

Annotation effort. Curating ground-truth-as-code requires more expert annotation effort than a natural-language reference; a systematic method for auto-generating the ground-truth code remains to be explored and has not been studied in this paper.

Higher-capacity models. The efficacy of natural-language ground truth should be re-examined with higher-capacity models such as Claude Opus or OpenAI GPT-5.5, which may interpret and explore their computational environments and follow unstructured instructions more reliably.

Robustness of the token-efficiency claim. The robustness of the token-efficiency claim needs to be tested by running similar experiments on agentic harnesses with different memory and tool-call caching mechanisms.

## Ethics Statement

Annotator participation. All annotators took part on a fully voluntary basis. They were either full-time employees or paid interns, completely independent from the study itself, and none had any stake in its outcome; no compensation was tied to the ratings they provided.

Data privacy. The synthetic database used throughout our evaluation was constructed specifically to prevent leakage of customer privacy of any sort. Absolutely no real-world data was used in either the creation of the synthetic data or the evaluation; all profiles, events, and catalogs are programmatically generated from fixed seeds and contain no personally identifiable information.

Broader impact. Our framework can potentially be used to evaluate other data-reporting systems on factual accuracy, helping to surface and quantify hallucinated facts. To mitigate AI-generated hallucination of facts, the ground truth is encoded as runnable code rather than free-form text, so the reference is computed deterministically rather than generated. Because this code is executed within a secure, sandboxed compute environment, the risk of malicious code being executed by the AI is contained.

## References

Jaehyun Nam, Jinsung Yoon, Jiefeng Chen, and Tomas Pfister. Ds-star: Data science agent via iterative planning and verification. arXiv preprint arXiv:2509.21825, 2025.

Maojun Sun, Ruijian Han, Binyan Jiang, Houduo Qi, Defeng Sun, Yancheng Yuan, and Jian Huang. A survey on large language model-based agents for statistics and data science. arXiv preprint arXiv:2412.14222, 2025a.

Ke Chen, Peiran Wang, Yaoning Yu, Xianyang Zhan, and Haohan Wang. Large language model-based data science agent: A survey. arXiv preprint arXiv:2508.02744, 2025.

Jiahao Chen et al. Measuring data science automation: A survey of evaluation tools for AI assistants and agents. arXiv preprint arXiv:2506.08800, 2026a.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Ben Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Sreemanti Dey, et al. Livebench: A challenging, contamination-limited llm benchmark. arXiv preprint arXiv:2406.19314, 2024.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chainof-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.

Xinyi Hou, Yanjie Zhao, Shenao Wang, and Haoyu Wang. Model context protocol (mcp): Landscape, security threats, and future research directions. ACM Transactions on Software Engineering and Methodology, 2025.

Linyue Pan, Lexiao Zou, Shuo Guo, Jingchen Ni, and Hai-Tao Zheng. Natural-language agent harnesses. arXiv preprint arXiv:2603.25723, 2026.

Yanna Jiang, Delong Li, Haiyu Deng, Baihe Ma, Xu Wang, Qin Wang, and Guangsheng Yu. Sok: Agentic skills–beyond tool use in llm agents. arXiv preprint arXiv:2602.20867, 2026.

Siyuan Guo, Cheng Deng, Ying Wen, Hechang Chen, Yi Chang, and Jun Wang. DS-Agent: Automated data science by empowering large language models with case-based reasoning. arXiv preprint arXiv:2402.17453, 2024.

Sirui Hong, Yizhang Lin, Bang Liu, Bangbang Liu, Binhao Wu, Ceyao Zhang, Chenxing Wei, Danyang Li, Jiaqi Chen, Jiayi Zhang, Jinlin Wang, Li Zhang, Lingyao Zhang, Min Yang, Mingchen Zhuge, Taicheng Guo, Tuo Zhou, Wei Tao, Xiangru Tang, Xiangtao Lu, Xiawu Zheng, Xinbing Liang, Yaying Fei, Yuheng Cheng, Zhibin Gou, Zongze Xu, and Chenglin Wu. Data interpreter: An LLM agent for data science. arXiv preprint arXiv:2402.18679, 2024.

Yanjie Fu, Dongjie Wang, Wangyang Ying, Xinyuan Wang, Xiangliang Zhang, Huan Liu, and Jian Pei. Autonomous data agents: A new opportunity for smart data. arXiv preprint arXiv:2509.18710, 2025.

Zhaoyan Sun, Jiayi Wang, Xinyang Zhao, Jiachi Wang, and Guoliang Li. Data agent: A holistic architecture for orchestrating data+AI ecosystems. arXiv preprint arXiv:2507.01599, 2025b.

Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. GAIA: a benchmark for general AI assistants. arXiv preprint arXiv:2311.12983, 2023.

Fan Nie, Junlin Wang, Harper Hua, Federico Bianchi, Yongchan Kwon, Zhenting Qi, Owen Queen, Shang Zhu, and James Zou. Dsgym: A holistic framework for evaluating and training data science agents. arXiv preprint arXiv:2601.16344, 2026.

Eugenie Lai, Gerardo Vitagliano, Ziyu Zhang, Om Chabra, Sivaprasad Sudhir, Anna Zeng, Anton A. Zabreyko, Chenning Li, Ferdi Kossmann, Jialin Ding, Jun Chen, Markos Markakis, Matthew Russo, Weiyang Wang, Ziniu Wu, Michael J. Cafarella, Lei Cao, Samuel Madden, and Tim Kraska. KramaBench: A benchmark for AI systems on data-to-insight pipelines over data lakes. arXiv preprint arXiv:2506.06541, 2025.

Alex Egg, Martin Iglesias Goyanes, Friso Kingma, Andreu Mora, Leandro von Werra, and Thomas Wolf. DABstep: Data agent benchmark for multi-step reasoning. arXiv preprint arXiv:2506.23719, 2025.

Mingyang Chen et al. DSAEval: Evaluating data science agents on a wide range of real-world data science problems. arXiv preprint arXiv:2601.13591, 2026b.

Victor Zhong, Caiming Xiong, and Richard Socher. Seq2SQL: Generating structured queries from natural language using reinforcement learning. In arXiv preprint arXiv:1709.00103, 2017.

Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 3911–3921, 2018.

Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng, Nan Huo, et al. Can LLM already serve as a database interface? A Big bench for large-scale database grounded text-to-SQLs. In Advances in Neural Information Processing Systems, volume 36, 2023.

Fangyu Lei, Jixuan Chen, Yuxiao Li, Ruisheng Huang, et al. Spider 2.0: Evaluating language models on real-world enterprise text-to-SQL workflows. arXiv preprint arXiv:2411.07763, 2024.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, and Jian Guo. A survey on LLM-as-a-judge. arXiv preprint arXiv:2411.15594, 2024.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan $\mathrm { L i } ,$ Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

Annalisa Szymanski, Noah Ziems, Heather A. Eicher-Miller, Toby Jia-Jun Li, Meng Jiang, and Ronald A. Metoyer. Limitations of the LLM-as-a-judge approach for evaluating LLM outputs in expert knowledge tasks. In Proceedings ofthe 30th International Conference on Intelligent User Interfaces, 2025.

Mingchen Zhuge, Changsheng Zhao, Dylan Ashley, Wenyi Wang, Dmitrii Khizbullin, Yunyang Xiong, Zechun Liu, Ernie Chang, Raghuraman Krishnamoorthi, Yuandong Tian, Yangyang Shi, Vikas Chandra, and Jürgen Schmidhuber. Agent-as-a-judge: Evaluate agents with agents. arXiv preprint arXiv:2410.10934, 2024.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12076–12100, 2023.

Jerry Wei, Chengrun Yang, Xinying Song, Yifeng Lu, Nathan Hu, Dustin Tran, Daiyi Peng, Ruibo Liu, Da Huang, Cosmo Du, and Quoc V Le. Long-form factuality in large language models. arXiv preprint arXiv:2403.18802, 2024.

Nazanin Jafari, James Allan, and Mohit Iyyer. Beyond precision: Importance-aware recall for factuality evaluation in long-form LLM generation. arXiv preprint arXiv:2604.03141, 2026.

Yongqi Fan, Yating Wang, Guandong Wang, Jie Zhai, Jingping Liu, Qi Ye, and Tong Ruan. MinosEval: Distinguishing factoid and non-factoid for tailored open-ended QA evaluation with LLMs. arXiv preprint arXiv:2506.15215, 2025.

Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. RAGAS: Automated evaluation of retrieval augmented generation. arXiv preprint arXiv:2309.15217, 2023.

Jon Saad-Falcon, Omar Khattab, Christopher Potts, and Matei Zaharia. ARES: An automated evaluation framework for retrieval-augmented generation systems. arXiv preprint arXiv:2311.09476, 2023.

Katerina Margatina, Shuai Ou, Nikolaos Aletras, and Ellie Pavlick. Dynamic benchmarking of masked language models on temporal concept drift with multiple views. In Proceedings ofthe 17th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics, pages 2145–2157, 2023.

Tianyi Shi et al. When benchmarks age: Temporal misalignment through large language model factuality evaluation. arXiv preprint arXiv:2510.07238, 2025.

Kun Xu et al. Benchmarking large language models under data contamination: A survey from static to dynamic evaluation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025.

Pratham Arora et al. PolyBench: Benchmarking LLM forecasting and trading capabilities on live prediction market data. arXiv preprint arXiv:2604.14199, 2026.

## Appendix A: Worked Example of a Single Test-Case Evaluation

To make the pipeline of Section 3 concrete, we walk through the evaluation of one real test case end to end. The example is test case $\tau _ { 0 }$ from the v3 data-discovery benchmark (financial vertical, tag = financial-modified), with components $( U _ { 0 } , G _ { 0 } , \mathcal { E } , \mathcal { C } )$ : the user query $U _ { 0 } ,$ the ground-truth-as-code function $G _ { 0 } ,$ and the environment and compute variables $\mathcal { E } ,$ C inherited from the skill under test. All content below is reproduced verbatim from our evaluation artifacts; long text is abridged with $^ { 6 6 } [ . . . ] ^ { , , }$

## .1 Input query $U _ { 0 }$

Our risk committee requested this after reviewing last quarter’s delinquency rates. We need a propensity model toflag credit card accounts showing early signs ofdefault risk—find the relevant AEP datasets, identify the key input variables, the target variable, and define the target mapping function.

## .2 Ground-truth-as-code $G _ { 0 }$

The reference answer is expressed as an executable function invocation, not as literal expected values. The test case stores the call

get\_ground\_truth\_string(idx=0, gt\_keys=[’ml\_goal\_type’, ’target\_variable’,   
’expected\_datasets’, ’input\_variables’, ’target\_mapping\_function’,   
’observation\_window\_days’, ’outcome\_window\_days’, ’failure\_modes’])

which invokes a reference function declared by the test bench. Executing $G _ { 0 }$ at evaluation time produces the computed ground truth $g _ { 0 }$ below. In this benchmark the function retrieves and formats the reference specification for the case; the same code-based interface generalizes to reference functions that recompute the answer from live data (Section 3.2).

## .3 Computed ground truth $g _ { 0 } = G _ { 0 } ( \mathcal { E } )$

The string returned by executing $G _ { 0 }$ specifies:

• ML goal: binary\_classification.

• Target variable: default\_risk\_30d.

• Expected datasets: the Financial 6M bundle (69bb43ac..., 69bb43ad3..., 69bb43ad7...).

• Input variables (with roles): creditTier (feature), incomeBand (confounder), accountBalance (feature); count\_90d aggregations of the events account.login, creditCard.purchase, and billing.paymentReceived; riskBand (confounder); and lastPaymentFailedAt (leakage—must be excluded from training).

• Target mapping function: let $t _ { 0 } = \mathrm { m a x } ( \mathrm { t i m e s t a m p } )$ per user; $y = 1$ if a billing.paymentFailed event occurs in $( t _ { 0 } , t _ { 0 } + 3 0 \mathrm { d } ]$ , else $0 ;$ features use events in $( t _ { 0 } - 9 0 0 , t _ { 0 } ] ;$ ; exclude lastPaymentFailedAt.

• Windows: observation 90 days, outcome 30 days.

## .4 Agent response r<sub>0</sub>

The agent’s response (abridged to the scored sections) selected the base Financial datasets rather than the Financial 6M bundle, and defined the target from a utilization heuristic rather than the billing.paymentFailed event:

Datasets Selected: [Financial] Experience Events 69b33da5..., [Financial] Customer   
Profiles 69b33da6..., [Financial] Account Catalog 69b33da6.... [...]   
Target Variable: is\_default\_risk (binary). [...]   
Target Mapping Function: is\_default\_risk = 1 if utilization\_rate > 0.85 OR   
transactionType == ’missed\_payment’, else 0. [...]

Profile features (creditTier, incomeBand, accountBalance) and the binary-classification goal are correct; the response includes no data-leakage warning.

## .5 Factoid extraction, matching, and metrics

The judge extracts factoids from the response and the computed ground truth (Eq. 1), then matches them (Section 3.4). For this case the judge counts approximately $| \mathcal { F } _ { r } | \approx 1 4 $ response factoids, $| \mathcal { F } _ { g } | \approx 1 2$ query-relevant ground-truth factoids, and $| \mathcal { F } _ { r g } |$ ≈ 5 matched factoids.

Matching is one-to-one and binary (Section 3.4): each response factoid either matches exactly one ground-truth factoid or is left unmatched, with no partial credit. The result partitions the factoids into three groups.

• Matched $( { \mathcal { F } } _ { r g } , \approx 5 ) { \mathrm { : } }$ the binary-classification goal; the input variables creditTier, incomeBand, and accountBalance; and the binary target concept. These are the facts the response and ground truth agree on.

• Ground-truth factoids with no match $( \mathcal { F } _ { g } \setminus \mathcal { F } _ { r g } , \approx 7 )$ — omissions that lower recall: the Financial 6M dataset IDs, the billing.paymentFailed target event, the count\_90d event aggregations (account.login, creditCard.purchase, billing.paymentReceived), the 90-day observation and 30-day outcome windows with $t _ { 0 }$ anchoring, and the lastPaymentFailedAt leakage exclusion.

• Response factoids with no match $( \mathcal { F } _ { r } \setminus \mathcal { F } _ { r g } , \approx 9 )$ — claims that lower precision: the base Financial dataset IDs (wrong bundle), the target defined from utilization\_rate $> \ 0 . 8 5$ , the value transactionType = ’missed\_payment’ (absent from the schema), and additional features proposed but not in the ground truth (creditLimit, interestRate, monthlyFee, primaryAccountType, derived customer age).

Substituting the three counts into Eq. 2:

$$
\begin{array} { r } { P = \frac { 5 } { 1 4 } \approx 0 . 3 6 , \qquad R = \frac { 5 } { 1 2 } \approx 0 . 4 2 , \qquad A = \frac { 5 } { 1 4 + 1 2 - 5 } = \frac { 5 } { 2 1 } \approx 0 . 2 4 . } \end{array}
$$

The judge rescales each fraction to the common 0–9 scale (Section 3.5) by multiplying by 9, yielding the reported scores in Table 4.

Table 4: Judge scores for test case $\tau _ { 0 } \left( 0 \mathrm { - } 9 \right.$ scale). The factoid-level metrics $( A , P , R )$ are the focus of this paper; the remaining dimensions are the rubric-guided qualitative scores.

<table><tr><td>Dimension</td><td>Score</td></tr><tr><td>Precision</td><td>3.2</td></tr><tr><td>Recall</td><td>3.8</td></tr><tr><td>Accuracy</td><td>2.1</td></tr><tr><td>Coverage</td><td>6.0</td></tr><tr><td>Conciseness</td><td>7.5</td></tr><tr><td>Tone</td><td>8.5</td></tr><tr><td>Clarity</td><td>8.0</td></tr><tr><td>Overall</td><td>5.1</td></tr></table>

## .6 Why the scores

Precision is low because several response factoids are incorrect or unsupported: the base Financial dataset IDs instead of the required Financial 6M IDs, a utilization-based target instead of the billing.paymentFailed mechanism, and a transactionType=’missed\_payment’ value that does not exist in the schema. Recall is low because critical ground truth factoids are absent from the response: the correct 6M dataset IDs, the billing.paymentFailed target event, the count\_90d event aggregations, the explicit 90-day/30-day windows, and the lastPaymentFailedAt leakage exclusion. This mirrors the interpretation in Section 3.5: hallucinated or wrong facts depress precision, while omitted facts depress recall, and both are directly attributable in the scorecard (Table 5).

Table 5: Issue categories flagged by the judge for test case $\tau _ { 0 } .$
<table><tr><td>Category</td><td>Description</td></tr><tr><td>Wrong datasets</td><td>Base Financial selected, not Financial 6M</td></tr><tr><td>Wrong target mechanism</td><td>Utilization proxy, not billing.paymentFailed</td></tr><tr><td>Missing event aggregations</td><td>No count_90d event features</td></tr><tr><td>Leakage not flagged</td><td>lastPaymentFailedAt not excluded</td></tr><tr><td>Wrong observation window</td><td>90d/30d windowing not defined</td></tr></table>