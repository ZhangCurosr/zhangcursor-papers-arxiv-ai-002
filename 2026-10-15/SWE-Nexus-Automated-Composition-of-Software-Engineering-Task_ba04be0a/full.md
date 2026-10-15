# SWE-Nexus: Automated Composition of Software Engineering Tasks from Test Execution

Songwei Yu<sup>1,⋆</sup>, Ziqi Miao<sup>2,⋆</sup>, Jingxuan Xu<sup>4,⋆</sup>, Junhao Fan<sup>3</sup>, Haoyang Huang<sup>4</sup> Yuan Xiong<sup>2</sup>, Zihang Xu<sup>2</sup>, Weihao Li<sup>4</sup>, Wenqiang Zhu<sup>4</sup>, Yuechen Luo<sup>3</sup> Xinshun Feng<sup>2</sup>, Yanan Wu<sup>4</sup>, Ken Deng<sup>4,</sup>† <sup>1</sup> National University of Singapore, Singapore <sup>2</sup> Shanghai Artificial Intelligence Laboratory, Shanghai, China <sup>3</sup> Tsinghua University, Beijing, China <sup>4</sup> Independent Researchers

## Abstract

Historical code changes provide convenient task boundaries, but do not identify which production functions selected tests exercise or support composing validated behaviors beyond a single change. We present SWE-Nexus, a fully automated pipeline that constructs repository-level implementation tasks from test execution. Four construction agents select tests, adapt tracing and evaluation protocols to each repository, and generate behavioral specifications. For each selected test file, tracing records an observed execution footprint of production function bodies. A scope is retained only if all expected scoring points pass in the original repository and masking the traced bodies causes at least one failure or error. Compatible scopes are composed deterministically, with shared functions counted once and every constituent test file revalidated under the complete combined mask. The benchmark contains 600 tasks from 105 repositories across six programming languages, including 116 atomic and 484 composite tasks. We evaluate 12 models under three agent scaffolds. The best configuration resolves 37.8% of tasks in one attempt, and scope-weighted scoring measures partial recovery. SWE-Nexus enables task construction from validated behavior beyond individual historical changes.

## 1 Introduction

Repository-level benchmarks commonly derive tasks from historical pull requests (PRs), combining issue descriptions, code changes, tests, and execution environments (Jimenez et al., 2024; Zan et al., 2025; Badertdinov et al., 2026). As agents solve more established instances, fixed collections become less informative about their capabilities (Deng et al., 2026). Public implementations also raise concerns about training data contamination (Badertdinov et al., 2025; Jain et al., 2025a). Informative evaluation and execution-based training therefore require a continuing supply of challenging, verifiable tasks (Pan et al., 2025; Yang et al., 2025; Jain et al., 2025b; Wei et al., 2026). Yet code and tests that appear in the same PR do not reveal which implementation each test exercises. For example, a test may exercise only some modified functions while calling unchanged helpers. Historical patch extraction can also include unrelated tests (Badertdinov et al., 2025). Human review can improve task descriptions and test suites (Deng et al., 2026), while execution tracing directly records the functions each test exercises (Rothermel and Harrold, 1997). A PR or commit captures one historical change, whereas a software requirement may span files and modules.

![](images/d0107901392ff64a9685d1733136189b3c3bc2a03cec5debe093ff4360c0b048.jpg)  
Figure 1: Task construction from historical PRs (a) and test execution (b). Edges show observed test–function associations. Deterministic composition counts shared functions once. Independent checks before and after composition require all expected points to pass originally (✓) and at least one failure or error per file under the complete mask (×).

These challenges motivate a central question:

Can we automatically construct diverse and challenging repository-level tasks beyond individual PR boundaries, with implementation scopes validated through test execution?

We introduce SWE-Nexus, a fully automated, four-agent pipeline that constructs atomic and composite implementation tasks from test execution (Fig. 1). The Selector, Trace, Validation, and Query Agents handle test selection, tracing, protocol adaptation, and specification generation, respectively. The production function bodies recorded for each test file form its observed execution footprint. An independent procedure retains a scope only when all expected scoring points pass in the original repository and masking the recorded bodies causes at least one point to fail or report an error. Each validated scope forms an atomic task. Compatible scopes from the same repository revision and environment are combined deterministically into composite tasks. Composition counts shared functions once and revalidates every constituent test file under the complete combined mask.

The evaluated solver is independent of the construction agents and receives only the behavioral specification and masked repository. Selected scoring tests and construction metadata remain hidden. The benchmark contains 600 tasks from 105 repositories, balanced across six programming languages. We evaluate 12 models under three agent scaffolds using the resolved rate for complete recovery and the scope-weighted score for partial recovery. The best configuration resolves 37.8% of tasks in one attempt. Under Claude Code, mean resolved rates decline from 55.2% on Easy tasks to 7.4% on Hard tasks. Within comparable configurations, resolved attempts generally have lower median counts of tool calls and modified files.

Our contributions are threefold:

• Automated construction. We introduce a fully automated four-agent pipeline that discovers implementation scopes from test execution. Fixed procedures validate each scope and recheck every constituent test file after deterministic composition.

• Compositional benchmark. We construct 600 tasks across six languages: 116 atomic and 484 composite tasks spanning two to nine test files.

• Evaluation. We introduce scope-weighted scoring for partial recovery, evaluate 12 models with three agent scaffolds, and analyze recorded solver activity.

## 2 Related Work

Repository-level evaluation. Defects4J and BugsInPy provide reproducible bugs and tests for controlled testing and debugging (Just et al., 2014; Widyasari et al., 2020). SWE-bench evaluates patches for GitHub issues using repository tests (Jimenez et al., 2024). Multi-SWE-bench broadens language coverage, while SWE-rebench V2 automates repository setup and task-quality assessment (Zan et al., 2025; Badertdinov et al., 2026). SWE-rebench collects fresh tasks for contamination-sensitive evaluation (Badertdinov et al., 2025), and SWE-Bench Pro targets longerhorizon changes with explicit requirements and interfaces (Deng et al., 2026).

Constructing executable tasks. SWE-smith retains model-generated, procedural, and historical modifications that break previously passing tests, then combines bugs within files or modules (Yang et al., 2025). Self-Play SWE-RL trains through iterative bug injection and repair with test patches as specifications (Wei et al., 2026). R2E-Gym synthesizes tests and problem descriptions from commits (Jain et al., 2025b). FeatureBench constructs feature development tasks across multiple PRs by tracing Python tests (Zhou et al., 2026).

Agent scaffolds and training. SWE-agent and OpenHands provide interfaces for inspecting, editing, and executing repository code, whereas Agentless uses a fixed localization, repair, and validation pipeline (Yang et al., 2024; Wang et al., 2025; Xia et al., 2025). SWE-agent’s interface ablations examine how search, editing, and execution feedback affect task resolution and repository navigation (Yang et al., 2024). SWE-Gym and R2E-Gym train agents and verifiers in executable repository environments (Pan et al., 2025; Jain et al., 2025b), where test outcomes supply verifiable behavioral feedback (Le et al., 2022). SWE-RL learns from historical changes using patchsimilarity rewards, while DeepSWE trains coding agents with test-based execution rewards (Wei et al., 2025; Luo et al., 2025).

![](images/d5fc3b48a8df56a3418351c4026f4eea0867977989e1e6993951deb9801e40c6.jpg)  
(a) Task composition

![](images/70ae166006ba63b8fd9b994ff2ea36ab2c650011b744db287dc4355c60588ffe.jpg)  
(b) Difficulty by language

![](images/a407e3ce00fb2715f9d58951d40a48dc59524d8858dd844ea51d5c975eee36c3.jpg)  
(c) Repository coverage

![](images/751a10c7d1b192e2acce6b53445e9cf6b35029bdf5e180bfb3178e602b7054ba.jpg)  
(d) Selected test-file counts

![](images/4c3006a3c9597f0c43eb89be3db4ca12702ecf7fc4793ac34c5e0aabb79067e5.jpg)

![](images/d7e677c9ec20df49495c678c8d48fc2cf3f512ae1c22b408eff048354f759608.jpg)  
Figure 2: Statistics of the 600-task benchmark: (a) task composition and connectivity of composite tasks (connected or disconnected), (b) difficulty by language (100 tasks each), (c) repository coverage, (d) selected test-file counts, (e) changed lines in reference restoration patches (log scale), and (f) scoring points per task (10-point bins). Edges link test files that share observed production functions. Dashed lines mark medians.

## 3 Dataset Overview

Each SWE-Nexus task asks a solver to recover repository functionality from a natural-language specification. The surrounding codebase remains available, with selected production function bodies masked and scoring tests withheld.

## 3.1 Task Definition

Let R denote a repository at a fixed revision with an associated execution environment. A task is defined by selected test files T and a validated implementation scope S of production functions whose bodies are masked. Let R− denote the masked repository. The solver receives a natural-language specification q and a copy of R− with the selected test files withheld. It submits a patch ∆ to production code for evaluation against those tests in a clean environment. Unselected repository tests may remain available during development.

The construction procedure retains the original implementations, selected tests, and fixed evaluation protocol. Restoring the original bodies gives a reference restoration patch for measuring task size. The tests and protocol define a fixed set of expected scoring points P. A scoring point is an evaluation unit with an outcome that can be identified consistently, typically a test case or parameterized instance. When finer outcomes are unavailable, a selected test file serves as one scoring point. Individual assertions are not counted separately. Evaluation tests the specified behavior without requiring the original implementation.

## 3.2 Dataset Characteristics

The benchmark contains 600 tasks from 105 repositories across six languages, with 100 tasks per language: 116 atomic tasks use one test file, and 484 composite tasks combine two to nine (Fig. 2(a)). Linking test files that share observed production functions yields one connected component (connected) or multiple components (disconnected). Behaviors span configuration parsing, application interfaces, and data serialization. One cloud-init task, for example, combines hosts-file operations with resolver configuration.

Tasks have a median of five selected test files and 35 masked production function bodies. Reference restoration patches modify a median of 10 production files and 433 lines, counting additions and deletions. The interquartile ranges are 2–7 selected test files, 242–1,046 changed lines, and 7– 32 scoring points. The full range is 1–202 scoring points, totaling 15,975 task–point pairs across the release. Selected tests and their scoring points may recur across composite tasks.

![](images/4c764f482607a4ba84f913160d47b1cd3c0df693b00c16074e367e31e5084a0c.jpg)  
Figure 3: Construction and evaluation in SWE-Nexus. The Selector, Trace, Validation, and Query Agents select tests, trace execution, adapt protocols, and generate specifications. Independent checks validate scopes before and after composition. The solver receives the specification and masked repository, and the evaluator restores hidden tests and applies the fixed protocol. Passing p<sub>1</sub> and p<sub>2</sub> gives a scope-weighted score of 50% and leaves the task unresolved.

Task scope. We assign Easy, Medium, and Hard tiers using selected test-file count (1–3, 4–6, 7–9) and reference restoration patch size (1–300, 301– 999, ≥ 1,000 changed lines). The final difficulty is the higher tier. Every language includes all three groups (Fig. 2(b) and Appendix C).

## 4 Dataset Construction

SWE-Nexus links repository tests to the production code they exercise. For a fixed revision and execution environment, tracing identifies candidate scopes, and validation retains those whose masking disrupts expected behavior. These scopes form atomic tasks or are composed deterministically into composite tasks. Four construction agents select tests, adapt tracing and evaluation protocols, and generate specifications. Masking, verification, composition, and scoring follow fixed procedures (Fig. 3).

## 4.1 Execution-Grounded Scope Validation

Test assertions specify expected behavior, but do not identify which production code to mask. The Selector Agent chooses files from a prefiltered pool, prioritizing explicit behavioral assertions, stable execution, and focused functional scope. The Trace Agent connects these tests to production code through runtime observation. It uses highquality tracing examples, existing strategies, and repository instructions to adapt tracing to the test runner, process boundaries, and source layout. For each selected file t, it executes the tests and maps observed functions to editable production function bodies. The resulting set $S _ { t }$ is the observed execution footprint for t. We retain these test–function associations for composition and scoring.

To verify that masking the observed scope affects the tested behavior, we replace the bodies in $S _ { t }$ with language-appropriate stubs, producing $R _ { t } ^ { - }$ while preserving declarations, signatures, and surrounding code structure. The Validation Agent adapts the execution plan and result parser to the repository and fixed environment. The resulting protocol specifies how to execute tests, parse their outputs, and assign each expected scoring point a pass, failure, or error status. An independent validation procedure runs this protocol on R and $R _ { t } ^ { - }$ Let $\mathcal { P } _ { t }$ denote the expected scoring points for t, and let $o _ { X } ( p )$ denote the parsed outcome of point $p$ in repository state $X .$ . We retain a scope only if

$$
\begin{array} { r l } & { \forall p \in \mathscr { P } _ { t } , \quad o _ { R } ( p ) = \mathrm { p a s s } , } \\ & { \exists p \in \mathscr { P } _ { t } : \quad o _ { R _ { t } ^ { - } } ( p ) \in \{ \mathrm { f a i l } , \mathrm { e r r o r } \} . } \end{array}\tag{1}
$$

The original run checks test validity, while the masked run checks whether removing the scope causes an observable failure or error. Each validated scope is retained with its evaluation protocol.

## 4.2 Deterministic Scope Composition

One validated scope forms an atomic task with a single selected test file. Combining scopes requires counting shared functions once and ensuring that every constituent test remains relevant. For compatible test files $\tau$ from the same repository revision and execution environment, we take the union $\textstyle S = \bigcup _ { t \in T } S _ { t }$ Compatibility requires that the scopes can be masked and evaluated together. We deduplicate shared functions by their source identities and build the complete masking patch from the clean original source.

The independent validation procedure rechecks every constituent file under the complete combined mask using the fixed execution and parsing rules. For each file $t \in \tau$ , all expected scoring points must pass in the original repository, and at least one must fail or report an error in the masked repository. Every selected file must therefore test behavior affected by the final mask, including files that share functions. Validated composite tasks contain two to nine selected test files and require one implementation to satisfy their combined requirements. This composition stage is deterministic.

## 4.3 Behavioral Specification

The natural-language specification describes the behavior to recover while the selected scoring tests remain hidden. The Query Agent generates the specification $q$ from the original implementations, selected tests, and repository context. Highquality specification examples guide its structure and coverage. The Problem Statement explains the target functionality, its purpose, and its role in the repository. The Requirements describe observable inputs, outputs, boundary cases, defaults, error handling, and side effects. The task package pairs $q$ with the masked repository R− after removing the selected test files. Execution traces, test–function associations, expected outcomes, and scoring configurations remain hidden.

The solver must recover the behavior from the specification and available repository context.

## 4.4 Fixed Evaluation and Scoring

The solver is the model under evaluation and operates independently of the four construction agents. It receives the task package and submits a patch $\Delta$ to production code. In a clean environment, an independent evaluator applies $\Delta$ to $R ^ { - }$ −, restores the selected test files from a trusted copy, and runs the protocol established during task validation. Test execution, output parsing, and scoring criteria are fixed before solving and applied consistently across submissions. A task is resolved only when the evaluation run is valid, every expected scoring point passes, and the patch introduces no additional failures detected by the protocol. The resolved rate is the proportion of tasks satisfying this criterion.

To measure partial recovery, the scope-weighted score weights test outcomes by the source spans of associated production function bodies. Each function $f \in S$ has a recorded body span of $L _ { f }$ source lines. The retained associations map each function to a nonempty set of expected scoring points $A ( f ) \subseteq { \mathcal { P } }$ from tests whose footprints contain that function. Let $b _ { p } ( \Delta ) \ \in \ \{ 0 , 1 \}$ indicate whether point $p$ passes for patch $\Delta$ . We divide each function’s span equally among its associated points and fix the resulting weights before solving:

$$
\begin{array} { c } { { w _ { p } = \displaystyle \sum _ { f \in S : p \in A ( f ) } \frac { L _ { f } } { | A ( f ) | } , } } \\ { { S _ { w } ( \Delta ) = \displaystyle \frac { \sum _ { p \in \mathcal { P } } w _ { p } b _ { p } ( \Delta ) } { \sum _ { p \in \mathcal { P } } w _ { p } } , } } \end{array}\tag{2}
$$

Each function contributes its span once, including functions shared by several tests, so $\textstyle \sum _ { p } w _ { p } = \sum _ { f } L _ { f }$ Appendix E.1 describes the coarser and unit-weight configurations used when finer attribution is unavailable. These configurations are also fixed before solving.

## 5 Evaluation

## 5.1 Evaluation Setup

We evaluate 12 proprietary and open-weight model variants on the frozen 600-task benchmark using Claude Code (Anthropic, 2026a), Cline (Cline, 2026), and OpenCode (OpenCode, 2026). All models have the same budget of up to 200 turns per scaffold invocation and 6,600 seconds per solver attempt. Each submission is evaluated in a clean environment using the hidden tests and protocol fixed during task construction. Overall and language results use one attempt per task (P@1). The Claude Code analysis by task scope also includes four attempts (P@4). Appendix D.1 lists model references and details execution limits and the handling of timeouts and retries.

![](images/c0265a24d8440fc0b760a03036663945a99be3030073b81538ebe2283db405a3.jpg)  
(a) Claude Code

![](images/480c447c252148603e2ef46a9ad2bb0a39f50115f41326bfd7b3c48e17b2acf5.jpg)  
(b) Cline

![](images/23b873c1a3aa1d5e80d8cf340721fa498bc3b59d9177f080e530c92bf60ba19d.jpg)  
(c) OpenCode

Figure 4: Overall P@1 performance across 12 models and three agent scaffolds. Segments connect each model’s resolved rate (circle) and mean scope-weighted score (diamond). All panels share a 0–65% scale. Appendix D reports exact values.
<table><tr><td>Model</td><td colspan="2">Easy (n = 138)</td><td colspan="2">Medium (n = 231)</td><td colspan="2">Hard (n = 231)</td><td colspan="2">Overall (n = 600)</td></tr><tr><td></td><td>P@1</td><td>P@4</td><td>P@1</td><td>P@4</td><td>P@1</td><td>P@4</td><td>P@1</td><td>P@4</td></tr><tr><td colspan="9">Proprietary models</td></tr><tr><td>Claude Opus 4.8</td><td>80.4 / 86.2</td><td>84.1 / 90.6</td><td>32.9 / 53.3</td><td>43.7 / 67.0</td><td>17.3 / 28.4</td><td>22.9 / 39.7</td><td>37.8 / 51.3</td><td>45.0 / 61.9</td></tr><tr><td>Gemini 3.1 Pro</td><td>56.5 / 70.4</td><td>64.5 / 79.5</td><td>14.3 / 46.5</td><td>23.4 / 60.5</td><td>2.2 / 25.7</td><td>8.2 / 37.4</td><td>19.3 / 44.0</td><td>27.0/ 55.9</td></tr><tr><td>GPT-5.3 Codex</td><td>60.9 / 78.7</td><td>71.7 / 86.2</td><td>29.0 / 60.2</td><td>39.8 / 72.4</td><td>10.0 / 32.1</td><td>13.9 /41.2</td><td>29.0 / 53.7</td><td>37.2 / 63.6</td></tr><tr><td>GPT-5.4</td><td>55.1 / 70.2</td><td>62.3 / 77.9</td><td>25.1 / 59.0</td><td>35.1/ 69.6</td><td>11.7 / 40.2</td><td>16.0 / 48.4</td><td>26.8 / 54.4</td><td>34.0 / 63.4</td></tr><tr><td>GPT-5.5</td><td>61.6 / 79.8</td><td>71.0 / 86.0</td><td>35.9 / 66.0</td><td>45.5 / 77.8</td><td>14.7 / 41.5</td><td>16.9/47.1</td><td>33.7 / 59.7</td><td>40.3 / 67.8</td></tr><tr><td>Qwen3.5-Plus</td><td>42.0 / 54.6</td><td>60.9 / 73.0</td><td>6.9 / 42.6</td><td>21.2 / 59.2</td><td>1.3 / 21.8</td><td>2.2 / 29.4</td><td>12.8 / 37.4</td><td>23.0 / 50.9</td></tr><tr><td colspan="9">Open-weight models</td></tr><tr><td>DeepSeek-V3.2</td><td>37.7 / 50.8</td><td>41.3 / 57.1</td><td>5.2 / 28.6</td><td>8.2 / 39.5</td><td>0.0 / 14.2</td><td>0.0 / 19.9</td><td>10.7 / 28.2</td><td>12.7 / 36.0</td></tr><tr><td>DeepSeek-V4-Flash</td><td>53.6 / 67.8</td><td>61.6 / 77.8</td><td>17.3 / 50.7</td><td>27.3 / 63.4</td><td>5.2 / 26.9</td><td>9.5 / 38.6</td><td>21.0 / 45.4</td><td>28.3 / 57.2</td></tr><tr><td>GLM-5</td><td>46.4 / 59.4</td><td>50.7 / 65.4</td><td>12.1 / 38.8</td><td>17.7 / 47.6</td><td>3.9 / 20.2</td><td>4.3 / 26.1</td><td>16.8 / 36.3</td><td>20.2 / 43.4</td></tr><tr><td>GLM-5.2</td><td>66.7 / 81.4</td><td>76.1 / 87.1</td><td>34.6 / 62.9</td><td>45.5 / 75.1</td><td>14.3 / 37.4</td><td>15.6 / 44.5</td><td>34.2 / 57.3</td><td>41.0 / 66.1</td></tr><tr><td>Kimi K2.6</td><td>57.3 / 74.1</td><td>64.5 / 81.2</td><td>21.2 / 52.7</td><td>29.9 / 63.9</td><td>6.1 / 28.6</td><td>10.4 / 37.5</td><td>23.7 / 48.3</td><td>30.3 / 57.7</td></tr><tr><td>MiniMax-M2.7</td><td>44.2 /58.4</td><td>49.3 / 64.4</td><td>10.4 / 38.6</td><td>15.1 / 47.3</td><td>2.6 / 19.9</td><td>3.0/26.2</td><td>15.2/ 35.9</td><td>18.3 / 43.1</td></tr></table>

Table 1: Claude Code performance by task scope. Cells report resolved rate / scope-weighted score (%). P@1 and P@4 use one and four attempts. Bold marks metric maxima, including ties.

We report the resolved rate, the percentage of tasks satisfying the strict criterion in Section 4.4, and the scope-weighted score averaged across tasks and expressed as a percentage. Scoring weights are fixed before solving, following Equation 2 and the configurations in Appendix E.1. At P@4, a task is resolved if any of its four patches resolves it, and its scope-weighted score is the maximum achieved by a single patch.

## 5.2 Main Results

Fig. 4 compares P@1 performance across all 36 combinations of models and scaffolds. Under Claude Code, Claude Opus 4.8 leads in resolved rate at 37.8% (51.3% weighted), whereas GPT-5.5 leads in scope-weighted score at 59.7% (33.7% resolved). The leading model therefore differs for complete and partial recovery.

The model with the highest resolved rate also varies by scaffold. Claude Opus 4.8 leads under Claude Code (37.8%) and OpenCode (29.7%), while GPT-5.5 and GPT-5.4 tie under Cline (31.2%).

## 5.3 Performance by Task Scope

Under Claude Code, both P@1 measures decrease from Easy to Medium to Hard for every model (Table 1 and Fig. 5(a,c)). Across the 12 models, the mean resolved rate falls from 55.2% to 20.4% and 7.4%, while the mean scope-weighted score falls from 69.3% to 50.0% and 28.1%.

![](images/13a624a58cb862bef2066c22a4d4484760e47068c163d09abc1983a33604a52d.jpg)  
(a) Resolved by task scope

![](images/bdea49db34a0514cdb2a84f0b7ce1937b364fd51faeffac80fe89aa10998d2c6.jpg)  
(b) Resolved by language

![](images/08e73e8620cc42c4729b3c3a3a2be25db159236bff8b3369323de51214266524.jpg)  
(c) Weighted by task scope

![](images/af9169bf9dffb14ad922667700d1a63b083b99e6e5324004eba37b0a747755c4.jpg)  
(d) Weighted by language  
Figure 5: Performance by task scope and language. Panels (a,c) show P@1/P@4 means across 12 models under Claude Code. Panels (b,d) show P@1 means across models for each scaffold. Shading and ranges show the middle 50%. Dashed lines mark grand means. JS and TS denote JavaScript and TypeScript.

This ordering persists for every model at P@4. On Hard tasks, four attempts raise the mean resolved rate from 7.4% to 10.2% and the scopeweighted score from 28.1% to 36.3%. Most Hard tasks remain unresolved under this larger attempt budget. The consistent decline across models and attempt budgets supports the distinction between the three difficulty tiers.

## 5.4 Performance across Languages

Under Claude Code (Table 2), Claude Opus 4.8 attains the highest resolved rate on Go, Python, JavaScript, and TypeScript, while GPT-5.5 leads on Java and Kotlin. For scope-weighted score, GPT-5.5 leads four languages, GPT-5.3 Codex leads TypeScript, and Claude Opus 4.8 leads Java.

Across all 36 combinations, mean P@1 resolved rates range from 12.7% on JavaScript to 37.1% on Java, and mean scope-weighted scores range from 33.1% on Go to 52.6% on Java (Fig. 5(b,d)). Averaging across models, Claude Code leads four languages on both measures. OpenCode leads on Java for both metrics and on Kotlin for scope-weighted score, while Cline leads on Kotlin for resolved rate. Appendix D reports results for every model under Cline and OpenCode.

## 6 Analysis

## 6.1 Interpreting Partial Recovery

A FlowFuse patch passes 128 of 141 scoring points and achieves a scope-weighted score of

91.6%. The 13 remaining failures concern getters for stored context, assistant defaults or getters controlled by feature flags, and the format of expert messages (Appendix B). A patch to Jackson’s Kotlin module passes 18 of 20 points and scores 90.4%. Both remain unresolved, but their scopeweighted scores capture substantial partial recovery. For reinforcement learning, this score could provide a graded reward for partial recovery based on the spans of associated function bodies, even when the task remains unresolved. Appendix E.1 gives the scoring details.

## 6.2 Solver Activity in Recorded Trajectories

The corpus contains 7,945 nonempty trajectories from eight models, three agent scaffolds, and six languages. Of these, 7,787 have scorable outcomes. We compare medians for resolved and unresolved attempts within 126 configurations defined by model, scaffold, and language, each with at least five attempts of each outcome.

Fig. 6 shows lower medians for resolved attempts in 122 of 126 configurations for tool calls (96.8%) and in 123 for modified files (97.6%). The median differences (resolved minus unresolved) are −35.5 calls and −4 files, with the same pattern in most configurations under each scaffold. Task difficulty and stopping after completion may partly explain this association.

## 6.3 Case Study: Task Composition

Fig. 7 illustrates a composite task for generating OpenAPI documentation from Kotlin routes and data types. Seven test files combine requirements for authentication, field constraints, examples, complex types, route parsing, schema generation, and nested resources. The tests check generated schemas and the served OpenAPI document against expected JSON snapshots. Composition expands the implementation scope from at most 40 functions in any individual footprint to 94 functions. Shared functions such as addToSpec and fromTypeToSchema connect document assembly with routing and type conversion. All 39 expected scoring points pass originally and fail under the complete mask, affecting every selected file.

<table><tr><td>Model</td><td>Go</td><td>Python</td><td>JavaScript</td><td>TypeScript</td><td>Java</td><td>Kotlin</td></tr><tr><td colspan="7">Proprietary models</td></tr><tr><td>Claude Opus 4.8</td><td>34.0 / 47.1</td><td>40.0 / 52.0</td><td>30.0 / 53.7</td><td>41.0 / 52.5</td><td>51.0 / 64.5</td><td>31.0/ 38.0</td></tr><tr><td>Gemini 3.1 Pro</td><td>16.0 /37.3</td><td>8.0 /37.1</td><td>12.0 / 49.2</td><td>16.0/ 43.9</td><td>35.0 /51.1</td><td>29.0 / 45.2</td></tr><tr><td>GPT-5.3 Codex</td><td>19.0 /36.7</td><td>31.0 /61.1</td><td>16.0 / 59.3</td><td>31.0 / 57.7</td><td>45.0 / 60.3</td><td>32.0 / 46.9</td></tr><tr><td>GPT-5.4</td><td>14.0/ 36.7</td><td>27.0/ 61.6</td><td>19.0 / 63.1</td><td>26.0/ 57.6</td><td>44.0 / 57.8</td><td>31.0 / 49.5</td></tr><tr><td>GPT-5.5</td><td>27.0 / 49.9</td><td>24.0 / 64.9</td><td>18.0 / 64.0</td><td>32.0/ 56.8</td><td>54.0 / 64.1</td><td>47.0 / 58.7</td></tr><tr><td>Qwen3.5-Plus</td><td>10.0/ 28.7</td><td>6.0/ 33.7</td><td>7.0 / 42.6</td><td>18.0 / 48.3</td><td>21.0/ 42.9</td><td>15.0 / 27.8</td></tr><tr><td colspan="7">Open-weight models</td></tr><tr><td>DeepSeek-V3.2</td><td>9.0 / 18.5</td><td>7.0 /26.8</td><td>7.0 / 36.8</td><td>10.0/ 33.8</td><td>19.0/ 34.5</td><td>12.0/18.7</td></tr><tr><td>DeepSeek-V4-Flash</td><td>18.0/ 34.9</td><td>11.0/41.9</td><td>11.0 / 48.2</td><td>16.0 / 46.9</td><td>33.0/ 47.7</td><td>37.0/ 53.0</td></tr><tr><td>GLM-5</td><td>14.0/ 26.4</td><td>11.0/34.2</td><td>12.0 / 42.6</td><td>12.0/ 39.8</td><td>27.0 / 41.0</td><td>25.0/ 34.1</td></tr><tr><td>GLM-5.2</td><td>27.0 / 49.0</td><td>27.0 / 61.3</td><td>22.0/ 61.3</td><td>34.0 / 55.3</td><td>52.0 / 64.0</td><td>43.0 /53.1</td></tr><tr><td>Kimi K2.6</td><td>17.0 /37.0</td><td>20.0 / 47.9</td><td>13.0 / 53.8</td><td>22.0 / 50.1</td><td>40.0 /55.3</td><td>30.0 / 46.0</td></tr><tr><td>MiniMax-M2.7</td><td>12.0 / 25.9</td><td>9.0 / 33.6</td><td>10.0 /41.9</td><td>11.0/39.7</td><td>25.0 / 40.4</td><td>24.0/ 34.1</td></tr></table>

Table 2: Claude Code P@1 performance by language (100 tasks each). Cells report resolved rate / scope-weighted score (%). Bold marks maxima for each language, including ties.

![](images/7c776de4f10c6da56664867aa9e3b40e08437f14b844069151560e56a3aea455.jpg)  
(a) Tool calls

![](images/a711da654dee788e7c52230a5201fabd0eaf499ac93ebf5b037e164fd33a5b70.jpg)  
(b) Modified files  
Figure 6: Recorded activity in resolved and unresolved trajectories. Each point compares group medians for one of 126 configurations defined by model, scaffold, and language: (a) tool calls and (b) modified production files. Dashed diagonals mark equality.

## 7 Conclusion

SWE-Nexus is a fully automated, four-agent pipeline for constructing repository-level tasks from test execution. It validates candidate scopes against original and masked repositories, then rechecks every constituent test file after deterministic composition. Evaluation applies the same hidden tests and fixed protocol across submissions.

The benchmark contains 600 tasks from 105 repositories across six languages. Across 12 models and three agent scaffolds, the best configuration resolves 37.8% of tasks in one attempt. Scopeweighted scoring captures partial recovery with weights fixed from observed test–function associations before solving. Composing validated execution footprints extends tasks beyond individual historical changes to span functions and modules.

## Limitations

SWE-Nexus relies on existing repository tests to identify implementation scopes and validate task behavior. Behaviors and edge cases that these tests do not exercise may be underrepresented in the resulting tasks. Extending the approach to functionality with limited test coverage requires developing and validating additional tests. The benchmark currently covers six programming languages. Broader language coverage requires adapting tracing, source mapping, and result parsing to additional runtimes and test frameworks.

## Ethical Considerations

This benchmark derives implementation tasks from public software repositories. Use and redistribution of repository code, tests, and derived tasks remain subject to the original licenses. During construction and evaluation, code from repositories and agents should run in environments isolated from sensitive hosts and credentials, since tests and setup commands can have side effects. Benchmark results verify selected test outcomes and do not establish security or deployment readiness. The study does not involve human participants.

![](images/fbb18238069a38a1e285402ace37553d147e943c92e60f6c6b6643d67ee03ec4.jpg)  
Figure 7: A composite task for OpenAPI documentation generation in Kompendium, with an abridged description. Seven selected test files cover 94 production functions across 33 source files. Purple circles mark the 36 functions associated with multiple selected test files.

## References

Anthropic. 2026a. Claude Code: Overview. Software documentation. Accessed October 4, 2026.

Anthropic. 2026b. Claude Opus 4.8 system card.

Ibragim Badertdinov, Alexander Golubev, Maksim Nekrashevich, Anton Shevtsov, Simon Karasik, Andrei Andriushchenko, Maria Trofimova, Daria Litvintseva, and Boris Yangel. 2025. SWE-rebench: An automated pipeline for task collection and decontaminated evaluation of software engineering agents. In Advances in Neural Information Processing Systems, volume 38. Curran Associates, Inc.

Ibragim Badertdinov, Maksim Nekrashevich, Anton Shevtsov, and Alexander Golubev. 2026. SWErebench V2: Language-agnostic SWE task collection at scale. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 4820–4843. PMLR.

Cline. 2026. Cline: Overview. Software documentation. Accessed October 4, 2026.

DeepSeek-AI. 2026. DeepSeek-V4-Flash: Model card.

DeepSeek-AI et al. 2025. DeepSeek-V3.2: Pushing the frontier of open large language models. Preprint, arXiv:2512.02556.

Xiang Deng, Jeff Da, Edwin Pan, Yannis Yiming He, Charles Ide, Kanak Garg, Niklas Lauffer, Andrew Park, Chetan Rane, Karmini Sampath, Maya Krishnan, Srivatsa R Kundurthy, Sean M. Hendryx, Zifan

Wang, Chen Bo Calvin Zhang, Noah Jacobson, Bing Liu, and Brad Kenstler. 2026. SWE-Bench Pro: Can AI agents solve long-horizon software engineering tasks? In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 23917–23934. PMLR.

GLM-5 Team et al. 2026. GLM-5: From vibe coding to agentic engineering. Preprint, arXiv:2602.15763.

Google DeepMind. 2026. Gemini 3.1 Pro: Model card.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida I. Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2025a. Live-CodeBench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations.

Naman Jain, Jaskirat Singh, Manish Shetty, Tianjun Zhang, Liang Zheng, Koushik Sen, and Ion Stoica. 2025b. R2E-Gym: Procedural environments and hybrid verifiers for scaling open-weights SWE agents. In Second Conference on Language Modeling.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2024. SWE-bench: Can language models resolve real-world GitHub issues? In The Twelfth International Conference on Learning Representations.

René Just, Darioush Jalali, and Michael D. Ernst. 2014. Defects4J: A database of existing faults to enable controlled testing studies for Java programs. In Proceedings of the 2014 International Symposium on Software Testing and Analysis, pages 437–440.

Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu Hong Hoi. 2022.

CodeRL: Mastering code generation through pretrained models and deep reinforcement learning. In Advances in Neural Information Processing Systems, volume 35, pages 21314–21328. Curran Associates, Inc.

Michael Luo, Naman Jain, Jaskirat Singh, Sijun Tan, Ameen Patel, Qingyang Wu, Alpay Ariyak, Colin Cai, Tarun Venkat, Shang Zhu, Ben Athiwaratkun, Manan Roongta, Ce Zhang, Li Erran Li, Raluca Ada Popa, Koushik Sen, and Ion Stoica. 2025. Deep-SWE: Training a fully open-sourced, state-of-the-art coding agent by scaling RL. Together AI and Agentica research blog.

MiniMax. 2026. MiniMax M2.7: Early echoes of selfevolution.

Moonshot AI. 2026. Kimi K2.6: Advancing opensource coding.

OpenAI. 2026a. GPT-5.3-Codex system card.

OpenAI. 2026b. GPT-5.4 Thinking system card.

OpenAI. 2026c. GPT-5.5 system card.

OpenCode. 2026. OpenCode: Documentation. Software documentation. Accessed October 4, 2026.

Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, and Yizhe Zhang. 2025. Training software engineering agents and verifiers with SWE-Gym. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 47717–47737. PMLR.

Qwen Team. 2026. Qwen3.5-397B-A17B: Model card.

Gregg Rothermel and Mary Jean Harrold. 1997. A safe, efficient regression test selection technique. ACM Transactions on Software Engineering and Methodology, 6(2):173–210.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Daniel Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, and 5 others. 2025. OpenHands: An open platform for AI software developers as generalist agents. In The Thirteenth International Conference on Learning Representations.

Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang, Daniel Fried, Gabriel Synnaeve, Rishabh Singh, and Sida I. Wang. 2025. SWE-RL: Advancing LLM reasoning via reinforcement learning on open software evolution. In Advances in Neural Information Processing Systems, volume 38.

Yuxiang Wei, Zhiqing Sun, Emily McMilin, Jonas Gehring, David W. Zhang, Gabriel Synnaeve, Daniel Fried, Lingming Zhang, and Sida Wang. 2026. Toward training superintelligent software agents through self-play SWE-RL. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 134075–134095. PMLR.

Ratnadira Widyasari, Sheng Qin Sim, Camellia Lok, Haodi Qi, Jack Phan, Qijin Tay, Constance Tan, Fiona Wee, Jodie Ethelda Tan, Yuheng Yieh, Brian Goh, Ferdian Thung, Hong Jin Kang, Thong Hoang, David Lo, and Eng Lieh Ouh. 2020. BugsInPy: A database of existing bugs in Python programs to enable controlled testing and debugging studies. In Proceedings of the 28th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, pages 1556–1560.

Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. 2025. Demystifying LLM-based software engineering agents. Proceedings of the ACM on Software Engineering, 2(FSE):801–824.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems, volume 37, pages 50528–50652. Curran Associates, Inc.

John Yang, Kilian Lieret, Carlos E. Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, Binyuan Hui, Ofir Press, Ludwig Schmidt, and Diyi Yang. 2025. SWE-smith: Scaling data for software engineering agents. In Advances in Neural Information Processing Systems, volume 38. Curran Associates, Inc.

Z.ai. 2026. GLM-5.2: Built for long-horizon tasks.

Daoguang Zan, Zhirong Huang, Wei Liu, Hanwu Chen, Shulin Xin, Linhao Zhang, Qi Liu, Aoyan Li, Lu Chen, Xiaojian Zhong, Siyao Liu, Yongsheng Xiao, Liangqiang Chen, Yuyu Zhang, Jing Su, Tianyu Liu, Rui Long, Ming Ding, and Liang Xiang. 2025. Multi-SWE-bench: A multilingual benchmark for issue resolving. In Advances in Neural Information Processing Systems, volume 38.

Qixing Zhou, Jiacheng Zhang, Haiyang Wang, Rui Hao, Jiahe Wang, Minghao Han, Yuxue Yang, Shuzhe Wu, Feiyang Pan, Lue Fan, Dandan Tu, and Zhaoxiang Zhang. 2026. FeatureBench: Benchmarking agentic coding for complex feature development. In The Fourteenth International Conference on Learning Representations.

## A Execution and Task Construction Details

Repository adaptation. Tracing examples provide instrumentation strategies and repository instructions. The Trace Agent adapts these materials to each selected test, defining its execution command and process scope. Table 3 summarizes the supported mechanisms and their coverage constraints.

Structural masking. The masking procedure locates editable bodies in clean source independently of the locations reported by tracing. We use source identities to deduplicate aliases and shared functions. Language-specific stubs preserve declarations while making missing behavior observable. Adjustments include retaining explicit constructor delegation in Java, adapting unused imports in Go, and preserving necessary type relationships in TypeScript. Structural checks compare the original and masked source and verify that the masking patch applies correctly. Independent runs on both repositories check the acceptance conditions in Equation 1.

Execution protocol. The evaluation protocol specifies environment setup, test execution, report collection, and result parsing for both repository states in clean environments. It also records exit status and test identifiers. The Validation Agent may refine the parser before acceptance, and any subsequent change requires renewed validation. When outcomes are available only for a test file as a whole, validation and evaluation use the same file-level scoring unit. Missing test case reports are not counted as individual failures.

Composition. For two observed execution footprints A and B, overlap is measured by $J ( A , B ) = \ell ( A \cap B ) / \ell ( A \cup B )$ , where ℓ(U) = $\sum _ { f \in U } L _ { f }$ and $L _ { f }$ is the recorded body span of function f in source lines. This ratio weights functions by their body spans. Candidate groups can be selected by union size and overlap. Every constituent test file must pass validation under the complete combined mask. Each scoring point is identified by its source test file and original test identifier to avoid collisions.

Specification and access policies. The Query Agent uses the original code, selected tests, repository context, and specification examples to generate a Problem Statement and Requirements. Function declarations and signatures remain available in the masked repository. Checks for format and literal leakage run before packaging. We remove the original Git object history and withhold construction artifacts from the solver workspace. Task instructions restrict external retrieval of repository source or solution code. Appendix B gives the specification format and recorded examples.

## B Task Specifications and Case Studies

Fig. 8 details the specification and solver inputs introduced in Fig. 3.

The examples use the Problem Statement and Requirements fields and the same execution restrictions. Some prompts also repeat function signatures available in the masked repository. We present atomic and composite tasks in Fig. 9 and recorded solver actions with independent evaluation outcomes in Fig. 10. Appendix F provides detailed specifications adapted from six composite tasks.

## B.1 Concrete Task Forms

Fig. 9 gives one released instance for each task form. An atomic task uses one selected test file and may mask multiple functions. A composite

<table><tr><td>Language</td><td>Representative observation mechanism</td><td>Scope or mapping constraint</td></tr><tr><td>Python</td><td>Profiling callbacks during pytest execution, with source paths and qualified names.</td><td>A call-phase hook does not automatically cover setup, teardown, native extensions, or subpro- cesses.</td></tr><tr><td>Go</td><td>Function-entry instrumentation in source files sup- plied through a build overlay.</td><td>Package compilation and execution can exceed the selected physical test file.</td></tr><tr><td>JavaScript</td><td>Source instrumentation or test-runner transforms in the executing Node.js process.</td><td>Export-only instrumentation and worker coverage depend on the chosen backend.</td></tr><tr><td>TypeScript</td><td>Source-aware transforms integrated with Vite/Vitest or the repository runner.</td><td>Transpilation, expression bodies, and test owner- ship require explicit source mapping.</td></tr><tr><td>Java</td><td>Source probes or runtime bytecode instrumentation with test-framework hooks.</td><td>Constructors, generated methods, and thread boundaries affect observation and localization.</td></tr><tr><td>Kotlin</td><td>Source probes or runtime observation followed by mapping to Kotlin declarations.</td><td>Synthetic wrappers and declarations without inde- pendent source bodies require special handling.</td></tr></table>

Table 3: Representative tracing mechanisms and coverage constraints. The mechanism is adapted to each repository’s test runner and source layout.

![](images/d86450a0173af05a32789151f416a31b9de7b361d306870acfba4ffc08587a7f.jpg)

Figure 8: Specification generation and solver inputs. The Query Agent uses the original implementation and selected tests to describe behavior. The solver receives the specification and masked repository, which retains function declarations and signatures.  
![](images/cdc4793b8db14833d7cf082a3f3886400022867a00660c91d28340f5389e63a7.jpg)

Figure 9: Atomic and composite tasks with their validated implementation scopes.
<table><tr><td>Language Repository</td><td></td><td>Tier</td><td>k Functions / lines Calls Resolved Weighted</td><td></td><td></td><td></td></tr><tr><td>Python</td><td>canonical/cloud-init</td><td>Easy</td><td>2 22 /216</td><td>47</td><td>Yes</td><td>100.0</td></tr><tr><td>JavaScript</td><td>FlowFuse/flowfuse</td><td>Medium 6</td><td>85 / 646</td><td>71</td><td>No</td><td>91.6</td></tr><tr><td></td><td>TypeScript minvws/NL COVID-19 Dashboard</td><td>Easy</td><td>2 4/113</td><td>50</td><td>Yes</td><td>100.0</td></tr><tr><td>Java</td><td>apache/avro</td><td>Hard</td><td>7 632 / 5,198</td><td>89</td><td>No</td><td>0.0</td></tr><tr><td>Kotlin</td><td>FasterXML/Jackson Kotlin</td><td>Medium 6</td><td>49 /553</td><td>192</td><td>No</td><td>90.4</td></tr></table>

Table 4: First-attempt Qwen3.5-Plus cases under Claude Code across five languages. k counts selected test files. Functions / lines gives unique functions and recorded body spans. Weighted is the scope-weighted score (%).

task combines the validated scopes of multiple test files, counting shared functions once.

## B.2 Recorded Execution Cases

Fig. 10 contrasts a fully resolved task with a highscoring patch that leaves several behaviors incomplete. The excerpts include only visible tool actions and recorded results.

## B.3 Cases across Languages

Table 4 reports first-attempt trajectories across five languages with the model and scaffold fixed. The cases cover all three scope tiers and include fully resolved tasks, partial recovery, and zero recovery.

## C Dataset Scope and Statistics

Dataset composition. Tables 5 and 6 summarize the frozen 600-task release by language and scope tier. The numbers of tasks selecting one through nine test files are respectively 116, 56, 59, 61, 61, 67, 63, 55, and 62. We link test files that share observed production functions. Of the 484 composite tasks, 255 have connected graphs with one component, and 229 have disconnected graphs with multiple components. Every constituent test file is validated under the complete combined mask, regardless of graph structure.

![](images/8ce73f909347468cfbc8ee142361d4e8610bb2ded1ec7eea3fbb6dfdcd6cda14.jpg)

![](images/5cd09adb8770491c2a09cefc94dd0254a5fb8f3ba82a1e8caf98e3d848656b1a.jpg)

Figure 10: Recorded tool actions and outcomes from two trajectories. Visible checks guide repair, and independent evaluation determines complete and partial recovery.
<table><tr><td>Language</td><td>Tasks</td><td>Repos. Mean N</td><td>Median N</td></tr><tr><td>Go</td><td>100</td><td>28 14.68</td><td>9</td></tr><tr><td>Python</td><td>100</td><td>18 35.68</td><td>26</td></tr><tr><td>JavaScript</td><td>100</td><td>13 35.66</td><td>21</td></tr><tr><td>TypeScript</td><td>100</td><td>19 31.67</td><td>25</td></tr><tr><td>Java</td><td>100</td><td>13 27.15</td><td>14</td></tr><tr><td>Kotlin</td><td>100</td><td>14 14.91</td><td>11.5</td></tr><tr><td>All</td><td>600</td><td>105</td><td>26.63 17</td></tr></table>

Table 5: Dataset statistics by language. N counts scoring points per task. Repository counts are per language. Repositories and scoring points may recur across tasks.

<table><tr><td>Statistic</td><td>Easy Medium</td><td></td><td>Hard</td><td>All</td></tr><tr><td>Tasks</td><td>138</td><td>231</td><td>231</td><td>600</td></tr><tr><td>Selected test files (median)</td><td>2</td><td>4</td><td>8</td><td>5</td></tr><tr><td>Masked bodies (median)</td><td>11</td><td>30</td><td>119</td><td>35</td></tr><tr><td>Reference-patch files (median)</td><td>3</td><td>9</td><td>20</td><td>10</td></tr><tr><td>Changed lines (median)</td><td>109.5</td><td>418</td><td>1,555</td><td>433</td></tr><tr><td>Scoring points (median)</td><td>5</td><td>16</td><td>28</td><td>17</td></tr></table>

Table 6: Recorded task scope by tier and overall. Masked bodies are production function bodies. File and line counts describe reference restoration patches. Changed lines count additions and deletions and differ from the source spans used for weighting.

Repository and parent reuse. The largest repository contributes 24% of Go tasks, 36% of Python, 23% of JavaScript, 18% of TypeScript, 27% of Java, and 28% of Kotlin. A validated parent test may participate in multiple composites. The maximum reuse in these language sets is 8, 24, 16, 12, 21, and 19 tasks, respectively. Task outcomes can therefore be statistically dependent, and future training and evaluation splits should control shared repositories and parent tests.

## D Evaluation Details

## D.1 Evaluation Configuration

Table 7 lists the evaluated model variants and their reference sources. All configurations use the common execution limits in Table 8. The output limit applies to each model request, and the solver deadline applies to each attempt. The Claude Code and OpenCode versions are 2.1.158 and 1.3.17, respectively.

On timeout, the runner preserves the current production patch for independent evaluation. Infrastructure retries are recorded separately from the solver attempts used for P@1/P@4. Evaluation applies the same hidden tests, parsing rules, and scoring criteria to every submission for a given task.

<table><tr><td>Model</td><td>Reference</td></tr><tr><td>Claude Opus 4.8</td><td>(Anthropic, 2026b)</td></tr><tr><td>Gemini 3.1 Pro</td><td>(Google DeepMind, 2026)</td></tr><tr><td>GPT-5.3 Codex</td><td>(OpenAI, 2026a)</td></tr><tr><td>GPT-5.4</td><td>(OpenAI, 2026b)</td></tr><tr><td>GPT-5.5</td><td>(OpenAI, 2026c)</td></tr><tr><td>Qwen3.5-Plus</td><td>(Qwen Team, 2026)</td></tr><tr><td>DeepSeek-V3.2</td><td>(DeepSeek-AI et al., 2025)</td></tr><tr><td>DeepSeek-V4-Flash (DeepSeek-AI, 2026)</td><td></td></tr><tr><td>GLM-5</td><td>(GLM-5 Team et al., 2026)</td></tr><tr><td>GLM-5.2</td><td>(Z.ai, 2026)</td></tr><tr><td>Kimi K2.6</td><td>(Moonshot AI, 2026)</td></tr><tr><td>MiniMax-M2.7</td><td>(MiniMax, 2026)</td></tr></table>

Table 7: Reference sources for the evaluated model variants.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Agent turns</td><td>200 per scaffold invocation</td></tr><tr><td>Solver time limit</td><td>6,600 seconds per attempt</td></tr><tr><td>Runtime timeout</td><td>7,200 seconds</td></tr><tr><td>Output token limit</td><td>32,000 per model request</td></tr></table>

Table 8: Common execution limits across all evaluated models and agent scaffolds. These limits are fixed before solving.
<table><tr><td>Model</td><td>Claude Code</td><td>Cline</td><td>OpenCode</td></tr><tr><td colspan="4">Proprietary models</td></tr><tr><td>Claude Opus 4.8</td><td>37.8 / 51.3</td><td>26.5 / 43.5</td><td>29.7 / 53.9</td></tr><tr><td>Gemini 3.1 Pro</td><td>19.3 / 44.0</td><td>19.5 / 43.0</td><td>18.2 / 41.4</td></tr><tr><td>GPT-5.3 Codex</td><td>29.0 / 53.7</td><td>27.7 / 52.4</td><td>27.8 / 53.8</td></tr><tr><td>GPT-5.4</td><td>26.8 / 54.4</td><td>31.2 / 55.4</td><td>23.8 / 51.3</td></tr><tr><td>GPT-5.5</td><td>33.7 / 59.7</td><td>31.2 / 58.0</td><td>25.5 / 53.6</td></tr><tr><td>Qwen3.5-Plus</td><td>12.8 / 37.4</td><td>9.3 / 22.1</td><td>13.2/ 33.2</td></tr><tr><td colspan="4">Open-weight models</td></tr><tr><td>DeepSeek-V3.2</td><td>10.7 / 28.2</td><td>13.2 / 30.2</td><td>13.5 /31.9</td></tr><tr><td>DeepSeek-V4-Flash</td><td>21.0 / 45.4</td><td>13.5 / 34.5</td><td>16.5 / 41.0</td></tr><tr><td>GLM-5</td><td>16.8 / 36.3</td><td>14.8 / 32.8</td><td>16.5 / 36.6</td></tr><tr><td>GLM-5.2</td><td>34.2 / 57.3</td><td>30.3 / 54.4</td><td>26.5 / 53.7</td></tr><tr><td>Kimi K2.6</td><td>23.7 / 48.3</td><td>23.2 / 47.2</td><td>22.5 / 47.0</td></tr><tr><td>MiniMax-M2.7</td><td>15.2 / 35.9</td><td>13.3 / 32.1</td><td>14.8 /36.0</td></tr></table>

Table 9: Exact overall P@1 performance on SWE-Nexus (%). Cells report resolved rate / scopeweighted score. Bold marks metric maxima, including ties.

## D.2 Complete Results

Tables 9 and 10 report complete P@1 results, rounded to one decimal. Means are computed from tabulated values over the models or configurations compared.

## E Scoring and Trajectory Analysis

## E.1 Scope-Weighted Scoring

Equation 2 divides each function’s recorded span equally among its associated scoring points, giving $\textstyle \sum _ { p } w _ { p } \ = \sum _ { f } L _ { f }$ . Coarser attribution associates a function with all points in its execution record. A file supplies one point when individual outcomes are unavailable. In the release, 573 tasks assign positive weight to every expected point. All scoring configurations, including coarser associations or unit weights where finer attribution is unavailable, are fixed before solving.

As a synthetic example, suppose a 20-line function is associated with points $p _ { 1 }$ and $p _ { 2 }$ , a 30-line function only with $p _ { 2 }$ , and a 50-line function only with $p _ { 3 }$ . Their point weights are 10, 40, and 50. Passing $p _ { 1 }$ and $p _ { 2 }$ yields an unweighted score of $2 / 3$ and a scope-weighted score of 0.5, but the task remains unresolved.

In recorded attempts, a FlowFuse patch passes 128 of 141 points with a scope-weighted score of 0.915988, and a patch to Jackson’s Kotlin module passes 18 of 20 with a score of 0.904475. Both remain unresolved. Appendix B traces the Flow-Fuse attempt from visible verification to independent evaluation.

## E.2 Trajectory Analysis

Each trajectory is identified by its model, scaffold, task instance, and trajectory hash. Of 7,945 nonempty trajectories, 7,787 are scorable. We compare median activity between resolved and unresolved attempts for each combination of model, scaffold, and language. Of 144 configurations, 126 contain at least five attempts of each outcome and are retained.

Tool call counts measure visible activity, and modified file counts measure the breadth of code changes. We also record whether the solver verifies its edits and where the first edit occurs in the trajectory. Across configurations, the median differences (resolved minus unresolved) are +4.39 and +2.11 percentage points, respectively.

## F Task Examples across Six Languages

Figs. 11–16 present one composite task for each language in SWE-Nexus. We select tasks that combine complementary constraints on a coherent software feature. The specifications preserve their functional goals, boundary cases, and errors, with wording edited for presentation.

<table><tr><td>Model</td><td>Go</td><td>Python</td><td>JavaScript</td><td>TypeScript</td><td>Java</td><td>Kotlin</td></tr><tr><td colspan="7">Cline · Proprietary models</td></tr><tr><td>Claude Opus 4.8</td><td>28.0 / 39.4</td><td>20.0 / 47.2</td><td>21.0/ 38.9</td><td>22.0 / 41.2</td><td>40.0 / 56.2</td><td>28.0 / 38.2</td></tr><tr><td>Gemini 3.1 Pro</td><td>20.0/ 33.5</td><td>11.0/39.1</td><td>13.0 / 44.4</td><td>11.0/ 40.5</td><td>36.0 / 53.9</td><td>26.0 / 46.7</td></tr><tr><td>GPT-5.3 Codex</td><td>16.0 / 32.8</td><td>24.0 / 47.6</td><td>14.0 / 59.1</td><td>24.0 / 53.2</td><td>51.0 /64.2</td><td>37.0 / 57.6</td></tr><tr><td>GPT-5.4</td><td>20.0 / 36.4</td><td>21.0 / 56.3</td><td>25.0 / 64.8</td><td>24.0 / 53.9</td><td>47.0/58.3</td><td>50.0 / 62.4</td></tr><tr><td>GPT-5.5</td><td>27.0 / 43.2</td><td>16.0 / 55.7</td><td>16.0 / 60.5</td><td>29.0 / 60.8</td><td>53.0 / 66.7</td><td>46.0 / 61.4</td></tr><tr><td>Qwen3.5-Plus</td><td>7.0 / 14.5</td><td>4.0 / 18.9</td><td>5.0 / 20.8</td><td>0.0/21.2</td><td>23.0 /31.0</td><td>17.0 /26.1</td></tr><tr><td colspan="7">Cline · Open-weight models</td></tr><tr><td>DeepSeek-V3.2</td><td>9.0/ 19.8</td><td>9.0/ 25.1</td><td>6.0 / 36.6</td><td>15.0 / 39.2</td><td>23.0 / 36.0</td><td>17.0 / 24.7</td></tr><tr><td>DeepSeek-V4-Flash</td><td>10.0 /30.5</td><td>4.0/ 25.7</td><td>5.0 / 38.1</td><td>6.0/ 28.7</td><td>25.0 / 44.2</td><td>31.0/ 39.6</td></tr><tr><td>GLM-5</td><td>11.0/ 24.8</td><td>8.0 / 25.9</td><td>6.0/ 37.7</td><td>12.0 / 35.8</td><td>26.0 / 40.4</td><td>26.0/ 31.9</td></tr><tr><td>GLM-5.2</td><td>27.0 / 42.3</td><td>18.0/ 53.6</td><td>16.0/55.1</td><td>28.0 / 55.9</td><td>50.0 / 64.1</td><td>43.0/ 55.6</td></tr><tr><td>Kimi K2.6</td><td>18.0/ 33.2</td><td>16.0 / 42.9</td><td>13.0/51.0</td><td>19.0 / 46.2</td><td>41.0/ 58.5</td><td>32.0 / 51.6</td></tr><tr><td>MiniMax-M2.7</td><td>10.0 / 24.6</td><td>7.0/ 25.4</td><td>5.0 / 37.3</td><td>10.0 / 34.5</td><td>24.0 / 39.7</td><td>24.0/ 31.4</td></tr><tr><td colspan="7">OpenCode · Proprietary models</td></tr><tr><td>Claude Opus 4.8</td><td>26.0 / 43.6</td><td>27.0 / 55.8</td><td>18.0 / 66.9</td><td>31.0 / 54.9</td><td>43.0/ 57.0</td><td>33.0 / 45.4</td></tr><tr><td>Gemini 3.1 Pro</td><td>16.0/ 33.4</td><td>9.0/34.9</td><td>9.0/ 41.7</td><td>7.0/ 33.6</td><td>39.0 / 56.1</td><td>29.0 / 48.6</td></tr><tr><td>GPT-5.3 Codex</td><td>19.0/ 37.4</td><td>18.0 / 46.0</td><td>13.0 / 59.4</td><td>26.0 / 54.2</td><td>48.0 / 63.7</td><td>43.0 / 62.2</td></tr><tr><td>GPT-5.4</td><td>9.0 / 30.2</td><td>23.0 / 54.6</td><td>13.0 / 59.4</td><td>24.0 / 56.0</td><td>42.0 /57.4</td><td>32.0 / 50.5</td></tr><tr><td>GPT-5.5</td><td>17.0/ 39.8</td><td>15.0 / 48.6</td><td>15.0/ 62.3</td><td>21.0 / 54.5</td><td>50.0 / 64.4</td><td>35.0/ 51.8</td></tr><tr><td>Qwen3.5-Plus</td><td>3.0 / 23.2</td><td>8.0 / 27.5</td><td>7.0/ 37.4</td><td>16.0/34.7</td><td>24.0 / 43.5</td><td>21.0/ 32.6</td></tr><tr><td colspan="7">OpenCode · Open-weight models</td></tr><tr><td>DeepSeek-V3.2</td><td>9.0 / 20.1</td><td>11.0 /28.7</td><td>7.0 / 34.3</td><td>8.0 / 32.6</td><td>22.0 / 41.9</td><td>24.0 / 34.0</td></tr><tr><td>DeepSeek-V4-Flash</td><td>15.0/ 32.0</td><td>8.0 / 27.6</td><td>8.0 / 47.6</td><td>14.0/51.7</td><td>34.0 / 55.3</td><td>20.0 / 32.1</td></tr><tr><td>GLM-5</td><td>13.0/ 26.1</td><td>12.0 / 29.2</td><td>10.0 / 40.8</td><td>12.0 / 41.6</td><td>28.0 / 48.0</td><td>24.0/ 33.7</td></tr><tr><td>GLM-5.2</td><td>20.0 / 40.8</td><td>18.0/ 50.4</td><td>15.0/63.5</td><td>22.0 / 54.6</td><td>48.0 / 62.5</td><td>36.0 / 50.2</td></tr><tr><td>Kimi K2.6</td><td>16.0 / 35.2</td><td>14.0 /39.9</td><td>11.0 /49.7</td><td>15.0 / 42.9</td><td>44.0 / 59.5</td><td>35.0/ 54.7</td></tr><tr><td>MiniMax-M2.7</td><td>11.0/ 25.4</td><td>10.0 / 28.2</td><td>8.0 / 40.2</td><td>10.0 / 41.2</td><td>27.0 / 47.9</td><td>23.0 /33.1</td></tr></table>

Table 10: Cline and OpenCode P@1 performance by language (100 tasks each). Cells report resolved rate / scopeweighted score (%). Bold marks maxima for each language and scaffold, including ties.

Go: Broker lifecycle. Fig. 11 combines six selected test files into one Broker lifecycle. Contract contents, readiness, and cleanup must remain consistent. Reconciliation must avoid unnecessary generation increments, while finalization preserves external topics.

Python: REST API runtime. Fig. 12 combines seven selected test files spanning request parsing and response representations. Form parsing must preserve the underlying Django request, and version selection must agree with hyperlink generation and validation. Distinguishing parser errors from ordinary attribute access adds another constraint across the framework’s request, serializer, and URL components.

JavaScript: Tour lifecycle. Fig. 13 combines nine selected test files covering rendered components, events, steps, and attachment utilities. Together they describe a complete user-facing interaction, including asynchronous showing and later cleanup. A step may render correctly on its first appearance yet mishandle a changed target or overwrite its original tabindex on a later show. The example makes lifecycle consistency and reversible DOM changes part of the same task.

TypeScript: Tab interaction. Fig. 14 combines seven selected test files around Tabs and its supporting utilities. The task connects asynchronous guards, public event ordering, keyboard focus, and reactive rendering. An immediate click event must coexist with a delayed state update, while reordered children must update both header order and indicator position. The example brings component state, event semantics, and supporting utilities under one behavioral specification.

Java: Schema evolution. Fig. 15 combines eight selected test files around a shared schema model. The feature spans construction, compatibility checking, and runtime use: accepting a default in a builder is insufficient if decoding later produces different bytes. Recursive histories and precise incompatibility locations add further constraints. The example illustrates a composite task in which several APIs must implement the same data-model rules consistently.

Kotlin: Multipart integrity. Fig. 16 combines nine selected test files covering object request conversion and multipart integrity. The example links accepted header formats to upload validation, decoded payloads, and digest generation. A matching ETag does not make an incorrectly ordered completion valid, and a correctly decoded body must still yield the required checksum representation. Together, these requirements specify object request handling and multipart integrity with concrete success and error behavior.

## Knative Kafka Broker: reconciliation and finalization

## Problem Statement

Restore the Kafka Broker control plane so creating, updating, and deleting a Broker produces consistent topic state, endpoint addresses, and readiness conditions. Reconciliation must translate Broker configuration into the shared data-plane contract stored in a ConfigMap. It must preserve other Brokers and existing egress entries, support authentication and transport-security settings, and safely clean up resources during finalization.

## Requirements

1. Delivery configuration. Map content modes and backoff policies to contract values, convert ISO-8601 durations to milliseconds with the expected defaults, and apply egress merge precedence. Contract generation increments must follow the existing wraparound behavior.

2. Topic settings. Parse default.topic.partitions, default.topic.replication.factor, and bootstrap.servers from ConfigMap data. Reject missing or invalid required values, normalize bootstrap server lists, retain optional default.topic.config.\* entries, and provide stable topic names.

3. Topic operations. Treat an already-existing topic as a successful creation and an unknown topic or partition as a successful deletion. Presence checks must accept valid topic metadata and reject missing, internal, or partitionless topics with useful errors.

4. Addresses and probes. Produce object paths of the form /namespace/name and service DNS names of the form name.namespace.svc. Build HTTP/HTTPS addressables with the appropriate audience and CA certificate data. Probe adapters must delegate to their supplied functions.

5. Broker reconciliation. Create or fetch the contract ConfigMap, handling empty or unreadable data. Resolve topic settings, reconcile managed or external topics, and insert or update exactly one Broker entry.

6. Semantic updates. Preserve unrelated resources and the Broker’s existing egress entries. Increment contract generation only when contract content changes. If receiver or dispatcher pod annotations are stale, update their volume-generation annotations to the persisted generation even when the contract itself is unchanged.

7. Readiness and status. Set or clear the appropriate conditions and annotations for configuration resolution, data-plane availability, topic readiness, and probe results. Record the resolved dead-letter URI when applicable.

8. Feature settings. Honor custom topic templates and manage authentication-secret references and finalizers. With OpenID Connect (OIDC) enabled, propagate audiences and apply authorization fallback policies. Project ready event policies, including CESQL filters, into ingress policies. Expose HTTP, HTTPS, or both according to the transport-encryption mode.

9. Finalization. Remove the Broker entry, persist the updated contract, and synchronize pod generation annotations. Delete managed topics while retaining external topics, and clean up secret finalizers. Tolerate missing ConfigMaps, topics, or secrets and configuration failures specific to teardown. Requeue when probe results or transient Kafka or secret conditions require another attempt.

Figure 11: An adapted Go composite task from knative-extensions/eventing-kafka-broker.

<table><tr><td colspan="2">Django REST framework: requests, representations, and versioned URLs</td></tr><tr><td colspan="2">Problem Statement</td></tr><tr><td colspan="2">Restore the runtime behavior that connects incoming Django requests to REST framework parsing, serializer fields, response negotiation, and versioned URLs. Applications must receive consistent values and error details across JSON, form, and multipart inputs. Renderer selection, settings reloads, and hyperlink generation must respect the active request context, including API versions and nested URL namespaces. Requirements</td></tr><tr><td>1. Bound serializer fields. Support serializer[field_name] for primitive, JSON, and nested fields. Bound objects must expose values, errors, and fully qualified names, and remain navigable when parent errors are not mappings. Form conversion must turn scalar None/False into empty strings while retaining submitted text for invalid JSON. 2. Request parsing. Wrap a Django HttpRequest, reject an already wrapped REST framework</td><td>5. Content negotiation. Parse Accept headers and select the best renderer/media-type match, including wildcards and parameters. Format-suffix filtering must raise Http404 when no renderer matches. Media-type helpers must preserve matching rules, precedence, and string representations. 6. Settings. Support string, list, and tuple import coercion through perform_import, with</td></tr><tr><td>request, and expose query_params as GET. Use the selected parser for empty GET/HEAD requests, forms, multipart uploads, and other bodies. Copy parsed form POST/FILES data to the underlying request so repeated form processing is safe. Reprocessing a consumed non-form stream must retain Django's raw-post-data exception. 3. Attribute errors. Re-raise parser and</td><td>informative import errors. Reject removed settings, cache resolved values, and refresh local and global api_settings after a Django settings override. Emit rest_framework.W001 when PAGE_SIZE is set without DEFAULT_PAGINATION_CLASS. 7. Format suffixes. Generate suffix-aware patterns for path() and re_path(), preserving order and default keyword arguments. Traverse nested include() trees with mixed path and regex</td></tr><tr><td>authenticator attribute errors as WrappedAttributeError, including access through user, data, hasattr, and authentication helpers. Preserve normal attribute proxying and user/auth setters on the underlying request. 4. Error details. Recursively convert translated strings, lists, tuples, dictionaries, and serializer return containers to ErrorDetail. Preserve message/code structure and consistent equality, hashing, and string representations. Throttling errors must round wait times for messages and expose wait metadata. Bad-request and server-error handlers must return JSON with status 400 and 500.</td><td>patterns. Support required suffixes and allowed-format lists. 8. Version selection. Resolve versions from Accept headers, URL paths, namespaces, hostnames, or query parameters. Apply allowed-version and default-version rules, including an explicitly allowed None, and raise the scheme's API exception for invalid versions. Scheme-specific reversing must apply the corresponding version to the URL. 9. Reversing and hyperlinks. Public reverse() must use the active versioning scheme with ordinary reversing as a fallback and preserve query overrides. Hyperlinked relational fields</td></tr><tr><td colspan="2">Shepherd: rendering and running interactive tour steps</td></tr><tr><td colspan="2">Problem Statement Restore Shepherd's step rendering and lifecycle so applications can configure interactive tours, update their</td></tr><tr><td colspan="2">contents, and move reliably between steps. A step must coordinate its components, event handlers, attachment target, and positioning with the surrounding Tour. Showing, hiding, or destroying a step must manage accessibility attributes and temporary DOM changes without losing the target's original state. Requirements</td></tr><tr><td>1. Buttons and cancel controls. Accept literal or callback option values and apply custom attrs, converting numeric and boolean values to strings. Shepherd's own type, class, disabled state, ARIA label, tabindex, and click handler must take precedence over conflicting custom attributes. 2. Content composition. Render a header, text area,</td><td>6. Deferred showing. Await an asynchronous beforeShowPromise before showing the step. Resolve attachment selectors and callbacks when needed for that step, and resolve them again on later shows. Do not evaluate every target eagerly at tour creation. Resolved-attachment lookup must remain safe before the first show.</td></tr><tr><td>or footer only when its options call for one. Headers may contain a title, cancel icon, or both. Footers contain one button per configured entry and none for an empty or omitted list. Text may be an HTML string, callback result, or HTMLElement. 3. Events. Evented. on must register handlers, and trigger must forward arguments and preserve</td><td>7. Accessibility and highlights. Tooltip/dialog markup must include the expected aria-labelledby and aria-describedby links. Apply enabled, target, and highlight classes to the appropriate elements when shown. Hiding or destroying the step must remove these classes and restore prior state.</td></tr><tr><td>their context. off must remove all handlers for an event or a specific handler. A once handler must be removed after its first invocation. 4. Lifecycle. cancel() and complete() must call the Tour and emit the matching step event. hide() must emit before-hide and hide, hide any modal overlay, and remove target styling. destroy() must emit destroy and remove</td><td>8. Focus restoration. Preserve the attachment target's original tabindex across repeated show/hide cycles. If it originally had no attribute, remove the temporary one. If it had a value, restore that value rather than an intermediate runtime value. 9. Advancing. bindAdvance must support</td></tr><tr><td>tooltip and component DOM state. 5. Option updates. Merge defaults, including nested floating UI options. updateStepOptions</td><td>selector-scoped listeners and delegated body listeners. Advance only when the step is open and the configured event/selector conditions match. Unregister listeners when the step is destroyed.</td></tr><tr><td>must retain unchanged options and refresh rendered text, titles, classes, and buttons when their values change.</td><td>10. Attachment and positioning. Resolve string selectors and callback targets, binding callbacks to the step. Resolve extra-highlight selector lists to matching elements, and center steps with absent or incomplete attachment options. Preserve an explicit floating UI strategy and place user middleware after Shepherd's default middleware.</td></tr></table>

Figure 12: An adapted Python composite task from encode/django-rest-framework.

Figure 13: An adapted JavaScript composite task from shipshapecode/shepherd.

![](images/248fa45b3224a0d122fc0cd547809858c7a18ed3b4abf36495b91c3fd66241a1.jpg)  
Figure 14: An adapted TypeScript composite task from element-plus/element-plus.

![](images/0c019230923144a0e8ebcd00a538d065f54b3db23ada390b0673be75e8eef602.jpg)  
Figure 15: An adapted Java composite task from apache/avro.

![](images/a5afc41745be44bd0e93ec15c594ca2868cf93a65766c07e69610367c68fac3d.jpg)  
Figure 16: An adapted Kotlin composite task from adobe/S3Mock.