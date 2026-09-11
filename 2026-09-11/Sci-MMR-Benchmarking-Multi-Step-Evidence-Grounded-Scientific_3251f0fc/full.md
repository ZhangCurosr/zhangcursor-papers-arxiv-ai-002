# Sci-MMR: Benchmarking Multi-Step Evidence-Grounded Scientific Reasoning in Multimodal Agents

Jiaqiang Li<sup>\*</sup>, Yajie Yang<sup>\*</sup>, Zhiheng Xi<sup>\*</sup>, Jiadong Chen, Enyu Zhou, Senjie Jin, Yang Nan, Jiazheng Zhang, Han Wang, Yanxin Li, Dingwei Zhu, Bicheng Deng, Yuhui Wang, Xiang Zheng, Qi Zhang, Lei Bai, Xingjun Ma<sup>†</sup>, Tao Gui<sup>†</sup>

Fudan NLP Group corresponding.author@fudan.edu.cn

Autonomous research agents are increasingly expected to search the literature, analyze experimental evidence, and generate scientific hypotheses. These capabilities require multi-step evidencegrounded reasoning that progressively acquires, integrates, and verifies evidence before reaching a conclusion. Existing multimodal benchmarks, however, largely evaluate final-answer accuracy, leaving open whether predictions are actually supported by traceable scientific evidence. We introduce Sci-MMR, a benchmark for multi-step evidence-grounded scientific reasoning built on structured argument graphs linking scientific claims, citation-grounded knowledge, visual evidence, and supporting regions. Sci-MMR comprises 235 multi-hop reasoning tasks spanning four scientific disciplines, with an average of nine figure panels per task. Evaluating eight frontier multimodal models, we find that answer accuracy consistently exceeds complete-evidence recovery rate by more than 20%, revealing a substantial gap that answer-only evaluation is structurally unable to capture. Through controlled interventions, we identify two fundamental bottlenecks. First, evidence acquisition: models struggle to extract complete structured evidence from scientific figures, accounting for 57.2% of failures. While cropping tools yield modest gains (+4.5 points), providing gold evidence improves accuracy by up to 37.0 points, indicating dificulty in assembling complete multi-region evidence. Second, evidence integration: models struggle to translate available evidence into correct conclusions, accounting for 31.8% of failures, while even with gold evidence the strongest model achieves only 69.1% accuracy on the hardest tasks. These findings indicate that current answercentric benchmarks substantially overestimate the evidence-grounded reasoning capabilities of multimodal research agents.

## 1. Introduction

Scientific reasoning rarely relies on isolated facts. Instead, researchers progressively acquire, integrate, and verify evidence from textual descriptions, visual observations, and experimental results before reaching a scientific conclusion [Pramanick et al., 2024, Li et al., 2024]. As autonomous research agents become increasingly capable of literature analysis, experimental interpretation, and hypothesis generation [Lu et al., 2024, Schmidgall et al., 2025, Gottweis et al.,

2025], multi-step evidence-grounded reasoning becomes a fundamental capability. Unlike conventional multimodal question answering, these agents must actively acquire evidence distributed across figures, experiments, and prior scientific knowledge before synthesizing reliable conclusions. Existing multimodal evaluations, however, primarily measure final-answer correctness [Pramanick et al., 2024, Wang et al., 2025, Zhao et al., 2026], leaving unanswered whether models can actually acquire and organize the evidence needed to justify their predictions.

To address this gap, we introduce Sci-MMR, a benchmark for evaluating multi-step evidencegrounded scientific reasoning in multimodal agents. Built from peer-reviewed scientific publications, Sci-MMR represents each task as a structured argument graph linking scientific claims, citation-grounded knowledge, visual evidence, and supporting image regions, enabling finegrained evaluation of both evidence acquisition and evidence integration (Figure 1). The benchmark comprises 235 multi-hop reasoning tasks spanning four scientific disciplines and 35 domains, with each task requiring evidence aggregation across an average of nine figure panels rather than a single figure or table.

Evaluating eight frontier multimodal models under four reasoning settings—Caption-only, Direct Visual Reasoning, Evidence-Hint Reasoning, and Agentic Tool-Use—reveals a consistent gap between answer accuracy and evidence coverage: models frequently reach correct conclusions while recovering only partial supporting evidence. Through controlled interventions, we identify two fundamental bottlenecks. The first is evidence acquisition. Models often identify relevant visual regions but fail to assemble the complete evidence distributed across multiple figures. Under Direct Visual Reasoning, evidence access and grounding account for 57.2% of errors. Models locate at least one relevant region in 62.1% of runs, but recover the complete evidence set in only 12.7%, showing that the challenge lies not in finding any relevant region but in assembling the complete set that multi-panel tasks require. Equipping models with visual cropping tools raises accuracy by 4.5 points, confirming that active localization does help, though its gains remain modest compared to gains of 19.1–37.0% from directly supplying gold evidence statements – indicating that most of the remaining dificulty lies in achieving complete, multi-region localization, not in extracting evidence once a region is found.

The second bottleneck is evidence integration. models struggle to translate available evidence into correct conclusions, accounting for 31.8% of failures, while even with gold evidence the strongest model achieves only 69.1% accuracy on the hardest tasks. These findings show that evidence-grounded reasoning remains a distinct challenge beyond evidence acquisition, even when the required evidence is fully available.

![](images/c0f36e6fe3701bce9932e0cfe6b5026416ca111d94e0eac3269811f5f83d932a.jpg)  
Figure 1 | Illustration of a Sci-MMR task. A scientific reasoning task requiring evidence acquisition across multiple figure panels, intermediateclaim reasoning, knowledge retrieval, and premise auditing to reconcile experimental kinetics with calculated energy landscapes.

Our main contributions are summarized as follows:

• We introduce Sci-MMR, the first benchmark for multi-step evidence-grounded scientific reasoning. Built from peer-reviewed publications, it comprises 235 multi-hop reasoning tasks spanning four scientific disciplines and 35 domains, grounded in structured argument graphs connecting scientific claims, citation-grounded knowledge, visual evidence, and supporting image regions.

• We develop a four-setting evaluation protocol that disentangles failures in evidence acquisition, evidence integration, and finalanswer reasoning, enabling fine-grained diagnosis of multimodal scientific reasoning.

• Our analysis of 8 frontier multimodal models reveals two fundamental bottlenecks: 1) they struggle to transform complex scientific figures into complete structured evidence; and 2) evidence integration remains challenging even when gold evidence is provided, indicating that evidence-grounded reasoning extends well beyond visual perception.

## 2. Related Work

Synthesizing Multi-Hop Reasoning Tasks. Existing scientific multimodal benchmarks evaluate figure-grounded question answering, claim verification, and paper-level understanding, but typically treat reasoning as recovering a final answer rather than reconstructing the evidence supporting a scientific claim [Pramanick et al., 2024, Wang et al., 2025, Ansari et al., 2026, Zhao et al., 2026]. Existing multi-hop task construction methods compose reasoning steps within a single image [Wang et al., 2026, Tran et al., 2025], sample paths from knowledge or content graphs [Ning et al., 2026, Du et al., 2026, Sung et al., 2026], or synthesize retrieval trajectories through graph expansion and information obfuscation [Li et al., 2025]. In contrast, Sci-MMR reconstructs evidence dependencies directly from peer-reviewed papers, grounding each reasoning step in citationsupported evidence and visual regions.

Beyond Answer-Centric Evaluation. Recent work has moved beyond answer-only evaluation by assessing intermediate reasoning and agent behaviors. PhysicsArena [Dai et al., 2025] evaluates variable identification, process formulation, and solution derivation, while VDR-Bench [Zeng et al., 2026] measures intermediate entity recovery alongside answer correctness. Research-agent benchmarks further introduce expert-authored rubrics, process–report consistency, capability-aware evaluators, taskspecificjudging, and multimodal evidence-fidelity checks [Sharma et al., 2025, Ye et al., 2026, Ben-Avraham et al., 2026, Ai et al., 2026, Huang et al., 2026]. These approaches reveal intermediate failures beyond final-answer accuracy but do not evaluate whether models acquire the complete evidence supporting a scientific conclusion. Sci-MMR instead disentangles evidence acquisition from evidence integration by evaluating evidence coverage and evidence-to-claim reasoning as complementary signals.

## 3. Sci-MMR and Evaluation

Scientific reasoning is inherently evidence-driven: conclusions are established by integrating multiple evidence items distributed across text, fig-

<table><tr><td>Benchmark</td><td></td><td>Disc. Img. Hop Agentic Process</td><td></td><td></td></tr><tr><td>SPIQA [Pramanick et al., 2024]</td><td>1</td><td>10.3 1.0</td><td>x</td><td>x</td></tr><tr><td>SciVer [Wang et al., 2025]</td><td>1</td><td>1.5 1.0</td><td>x</td><td>x</td></tr><tr><td>PhysicsArena [Dai et al., 2025]</td><td>1</td><td>1.0 1.0</td><td>x</td><td>√</td></tr><tr><td>PaperMind [Zhao et al., 2026]</td><td>7</td><td>1.0 1.0</td><td>√</td><td>x</td></tr><tr><td>MMDR [Huang et al., 2026]</td><td></td><td>2.8 1.0</td><td>√</td><td>√</td></tr><tr><td>CRIT [Sung et al., 2026]</td><td></td><td>6.5 2.7</td><td>x</td><td>x</td></tr><tr><td>MC-Search [Ning et al., 2026]</td><td></td><td>1.0 3.8</td><td>√</td><td>√</td></tr><tr><td>VDR [Zeng et al., 2026]</td><td></td><td>1.0 1.0</td><td>√</td><td>√</td></tr><tr><td>Sci-MMR (Ours)</td><td>4</td><td>9.0 9.9</td><td>√</td><td>√</td></tr></table>

Table 1 | Comparison with existing benchmarks. Disc. denotes the number of disciplines (− indicates open-domain coverage). Img. and Hop denote per-task averages. Agentic: agent-based execution; Process: process-level evaluation.

ures, and prior knowledge. Existing scientific multimodal benchmarks, however, primarily evaluate final answers without explicitly modeling the evidence dependencies underlying a scientific claim. Even benchmarks with multi-hop reasoning typically construct reasoning paths synthetically through graph sampling or retrieval expansion, rather than recovering the evidential argument presented in the original paper.

Sci-MMR addresses this limitation by reconstructing the evidence dependencies directly from peer-reviewed scientific publications. Each task is represented as an Argument Graph that links scientific claims, evidence statements, citationgrounded knowledge, and supporting visual regions. This representation enables fine-grained evaluation of both evidence acquisition—whether a model retrieves the required evidence—and evidence integration—whether it correctly connects the acquired evidence to the target scientific claim.

## 3.1. Argument Graph Formulation

Scientific arguments are inherently hierarchical: visual evidence supports observations, which are recursively composed into increasingly abstract scientific claims [Teufel et al., 1999, Lauscher et al., 2018, Moser and Mercer, 2020]. We represent this structure as an Argument Graph, a directed acyclic graph:

$$
\mathcal { G } = ( V , E ) ,
$$

where nodes � denote information at diferent abstraction levels and edges � encode support relationships.

![](images/b14c664eaa69fd1b33f4416244b98603b1390a44223061cad189803dd8e82b54.jpg)  
Figure 2 | Overview of Sci-MMR. Scientific papers are grounded in visual evidence and represented as hierarchical Argument Graphs. Sampled argument subgraphs drive task generation and quality control, yielding 235 tasks spanning four disciplines and 35 domains, with an average of 9 figure panels per task. Model responses are evaluated using answer accuracy, evidence coverage, and claim coverage.

Nodes. The node set is partitioned into four disjoint types:

$$
V = V _ { \nu } \cup V _ { e } \cup V _ { k } \cup V _ { c } .
$$

• $V _ { \nu } \mathbf { \dot { \mathbf { \cdot } } }$ Visual source nodes denote figure panels, tables, and other visual elements containing experimental information.

• $V _ { e } i$ Evidence nodes represent evidence grounded in visual sources and link scientific statements to their supporting visual elements.

• $V _ { k }$ : Knowledge nodes represent external scientific knowledge, methodological definitions, or prior findings grounded in the paper’s cited references.

$V _ { c } { \mathrm { : } }$ Claim nodes represent hierarchical scientific claims, from intermediate conclusions supported by evidence to the final scientific conclusion.

Edges. Each edge $( u , \nu ) \in E$ denotes that node � directly supports node �. Support relations are restricted to:

$$
E \subseteq \left( V _ { \nu } \times V _ { e } \right) \cup \ \left( V _ { e } \times V _ { c } \right) \cup \ \left( V _ { k } \times V _ { c } \right) \cup \ \left( V _ { c } \times V _ { c } \right) ,
$$

where $V _ { \nu } \times V _ { e }$ grounds evidence in visual regions, $V _ { e } \times V _ { c }$ links evidence to claims, $V _ { k } \times V _ { c }$ captures dependencies on citation-grounded scientific knowledge, and $V _ { c } \times V _ { c }$ composes lower-level claims into higher-level ones. Since $\mathcal { G }$ is a DAG, every claim is supported by an acyclic chain of evidence and intermediate claims terminating at the task conclusion.

Root and task structure. Each graph has a unique root claim $c ^ { * } \in V _ { c } ,$ corresponding to the task’s final scientific conclusion. Solving a task requires recovering the relevant evidence and knowledge nodes and integrating them through the argument graph to infer $c ^ { * }$ . We characterize each task by its hop count, Hop = |�|, graph depth, defined as the longest directed path terminating at $c ^ { * }$ , and panel count, $\left| V _ { \nu } \right|$ , the number of distinct visual sources.

## 3.2. Benchmark Construction and Composition

We adopt a human-in-the-loop pipeline in which automated tools perform scalable extraction and graph construction, while domain experts intervene only where scientific judgment is required: evidence annotation and final verification.

Graph construction. We collect 800 papers published between 2020 and 2026 from leading venues including Nature, Science, Cell, and PNAS; the final benchmark is dominated by 2026 papers (Appendix A.1). For each paper, MinerU [Wang et al., 2024] extracts figures, tables, and captions. An LLM constructs an initial Argument Graph linking visual sources, evidence, scientific knowledge, and claims, retrieving scientific knowledge from the paper’s cited references when required. Domain experts then localize each evidence node to its supporting visual region, rewrite it as a conservative, directly observable statement, and verify the complete argument graph.

Task generation. From each verified Argument Graph, we sample a argument subgraph suficient to derive a target claim. The sampled graph specifies the required observations, intermediate claims, and scientific knowledge, from which an LLM generates a natural multi-hop question without revealing the target conclusion. Reference answers are generated independently from the complete argument graph, presenting grounded observations, intermediate reasoning, and the final conclusion in dependency order with explicit figure and table attributions.

Automated quality control. Candidate tasks first pass through two automated filters. Deterministic validation removes 54% of tasks whose argument graph depth falls below a minimum threshold. The remaining tasks are evaluated using an eightdimensional LLM-based quality rubric, filtering a further 17%; failed tasks are revised and reevaluated rather than discarded. The complete rubric is detailed in Supplementary Section B.

Human verification. Automated filtering can only screen for structural and surface-level quality; it cannot verify whether a task’s argument is scientifically sound. Domain experts therefore conduct a final manual review of every surviving task, confirming that each evidence statement is faithfully grounded in its visual region and that the reasoning chain genuinely supports the target claim. This is the last and only stage capable of catching tasks that pass automated checks yet fail scientifically.

This pipeline yields Sci-MMR, comprising 235 multi-hop reasoning tasks spanning 4 scientific disciplines and 35 domains, with an average of 9 figure panels per task. Figure 3 summarizes the benchmark composition and task complexity, while Supplementary Secs. A.2–A.3 provide additional dataset statistics.

![](images/6cca56f21709d181adbc3eb9f31c8b744ee24b2f07a442a8ca76f412aabbd389.jpg)

![](images/576a988436539d56529c854abb8ec1b0954d5a9c8f881b024e397d9348aeb020.jpg)

![](images/f364f16b1c1590e146215f10c496758bed0cef42ebf3538b00077e328ce69b61.jpg)  
Figure 3 | Sci-MMR composition and task complexity. Discipline–domain distribution and distributions of graph edges and figure panels per task.

## 3.3. Evaluation Metrics

Scientific reasoning proceeds from visual evidence to intermediate claims and ultimately to a scientific conclusion. Accordingly, Sci-MMR evaluates three complementary dimensions: (1) Answer Accuracy, (2) Evidence Coverage, and (3) Claim Coverage, measuring whether models recover the complete reasoning chain.

(1) Answer Accuracy. For a task $t \in \mathcal { T }$ , let $r _ { t }$ denote the model response and $y _ { t }$ the reference conclusion. The task-level score is

$$
A _ { t } = \operatorname* { m a t c h } _ { \operatorname { a n s } } ( r _ { t } , y _ { t } ) ,\tag{1}
$$

where match $\mathsf { l } _ { \mathrm { a n s } } ( r _ { t } , y _ { t } ) = 1$ if the final conclusion expressed in $r _ { t }$ is semantically consistent with $y _ { t } ,$ and 0 otherwise. Overall Answer Accuracy is the dataset-level average:

$$
\mathsf { A c c . } = \frac { 1 } { | \mathscr { T } | } \sum _ { t \in \mathscr { T } } A _ { t } .\tag{2}
$$

(2) Evidence Coverage. For a task � with reference evidence nodes $V _ { e , t }$ and response $r _ { t }$ , the task-level score is

$$
E _ { t } = \frac { 1 } { | V _ { e , t } | } \sum _ { e _ { i } \in V _ { e , t } } m ( e _ { i } , r _ { t } ) ,\tag{3}
$$

where $m ( e _ { i } , r _ { t } ) = 1$ if the response substantively expresses the scientific proposition represented

Agentic Tool-Use
<table><tr><td rowspan="2">Model</td><td colspan="2">Easy (n = 62)</td><td colspan="2">Medium (n = 92)</td><td colspan="2"></td><td colspan="2">Hard (n = 81)</td><td colspan="3">Overall (n = 235)</td></tr><tr><td>Acc. E-Cov. C-Cov.</td><td></td><td></td><td>Acc. E-Cov. C-Cov.</td><td></td><td></td><td>Acc. E-Cov. C-Cov.</td><td></td><td></td><td>Acc. E-Cov. C-Cov.</td><td></td></tr><tr><td></td><td>Direct Visual Reasoning</td><td></td><td colspan="7"></td></tr><tr><td></td><td></td><td>66.0</td><td>56.3</td><td>81.5</td><td>60.7 46.4</td><td>29.6</td><td>50.3</td><td>36.5</td><td>67.7</td><td>58.5</td><td>46.1</td></tr><tr><td>GPT-5.5 Claude Opus 4.8</td><td>96.8 95.2</td><td>75.7</td><td>64.6</td><td>66.3</td><td>54.8 51.4</td><td>11.1</td><td>51.8</td><td>39.7</td><td>54.9</td><td>59.3</td><td>51.5</td></tr><tr><td>Kimi K2.7</td><td>93.5</td><td>59.1</td><td>39.7</td><td>56.5</td><td>51.1 42.1</td><td>11.1</td><td>44.6</td><td>27.0</td><td>50.6</td><td>50.9</td><td>36.7</td></tr><tr><td>Gemini 3.1 Pro</td><td>90.3</td><td>58.0</td><td>43.1</td><td>58.7</td><td>42.1</td><td>40.6</td><td>9.9 35.0</td><td>34.6</td><td>50.2</td><td>43.8</td><td>39.4</td></tr><tr><td>MiniMax-M3</td><td>93.5</td><td>77.8</td><td>57.9</td><td>48.9</td><td>60.8</td><td>50.6</td><td>4.9 49.9</td><td>47.5</td><td>45.5</td><td>61.5</td><td>51.7</td></tr><tr><td>GLM-5V Turbo</td><td>83.9</td><td>58.9</td><td>40.6</td><td>41.3</td><td>45.6</td><td>32.3</td><td>8.6 41.3</td><td>29.9</td><td>41.3</td><td>47.6</td><td>33.9</td></tr><tr><td>Qwen3.7 Plus</td><td>71.0</td><td>57.9</td><td>43.9</td><td>28.3</td><td>40.3</td><td>31.3</td><td>4.9 40.7</td><td>30.4</td><td>31.5</td><td>45.1</td><td>34.6</td></tr><tr><td>Intern-S2</td><td>71.0</td><td>57.5</td><td>38.8</td><td>16.3</td><td>38.1</td><td>33.6</td><td>4.9 33.9</td><td>21.1</td><td>26.8</td><td>41.8</td><td>31.2</td></tr><tr><td>Average</td><td>86.9</td><td>63.9</td><td>48.1</td><td>49.7</td><td>49.2</td><td>41.0</td><td>10.6</td><td>43.4 33.3</td><td>46.1</td><td>51.1</td><td>40.6</td></tr></table>

<table><tr><td>GPT-5.5</td><td>100.0</td><td>73.8</td><td>63.4</td><td>84.8</td><td>62.6</td><td>51.9</td><td>29.6</td><td>57.2</td><td>46.3</td><td>69.8</td><td>63.7</td><td>53.4</td></tr><tr><td>Claude Opus 4.8</td><td>93.5</td><td>79.5</td><td>68.3</td><td>72.8</td><td>67.4</td><td>52.0</td><td>19.8</td><td>59.7</td><td>43.4</td><td>60.0</td><td>67.9</td><td>54.0</td></tr><tr><td>Kimi K2.7</td><td>93.5</td><td>67.9</td><td>48.8</td><td>63.0</td><td>54.0</td><td>42.8</td><td>9.9</td><td>48.0</td><td>39.7</td><td>52.8</td><td>55.6</td><td>43.5</td></tr><tr><td>Gemini 3.1 Pro</td><td>88.7</td><td>68.2</td><td>49.4</td><td>54.3</td><td>55.0</td><td>56.7</td><td>22.2</td><td>48.3</td><td>43.4</td><td>52.3</td><td>56.2</td><td>50.5</td></tr><tr><td>MiniMax-M3</td><td>83.9</td><td>80.3</td><td>67.1</td><td>58.7</td><td>70.3</td><td>51.0</td><td>16.0</td><td>54.6</td><td>46.3</td><td>50.6</td><td>67.5</td><td>54.1</td></tr><tr><td>GLM-5V Turbo</td><td>82.3</td><td>68.9</td><td>47.8</td><td>46.7</td><td>56.3</td><td>42.3</td><td>12.3</td><td>49.9</td><td>43.6</td><td>44.3</td><td>57.4</td><td>44.3</td></tr><tr><td>Qwen3.7 Plus</td><td>71.0</td><td>65.4</td><td>49.0</td><td>44.6</td><td>53.4</td><td>36.0</td><td>13.6</td><td>48.2</td><td>39.0</td><td>40.9</td><td>54.8</td><td>40.6</td></tr><tr><td>Intern-S2</td><td>71.0</td><td>44.2</td><td>26.9</td><td>28.3</td><td>28.3</td><td>17.1</td><td>12.3</td><td>28.7</td><td>24.5</td><td>34.0</td><td>32.6</td><td>22.1</td></tr><tr><td>Average</td><td>85.5</td><td>68.5</td><td>52.6</td><td>56.7</td><td>55.9</td><td>43.7</td><td>17.0</td><td>49.3</td><td>40.8</td><td>50.6</td><td>57.0</td><td>45.3</td></tr></table>

![](images/681935c0b920c8e7bab440ed579e53ff1771c33cc5132907fd2e9a8a72f9b629.jpg)  
Table 2 | Direct Visual Reasoning and Agentic Tool-Use performance by task dificulty and overall. Scores are percentages; overall results are unweighted averages. Bold and underlined values indicate the best and second-best performance, respectively. The lower panel compares overall Answer Accuracy (Acc.), Evidence Coverage (E-Cov.), and Claim Coverage (C-Cov.); hollow and filled markers denote Direct Visual Reasoning and Agentic Tool-Use, respectively.

by $e _ { i } ,$ and 0 otherwise. Overall Evidence Coverage is

$$
\mathsf { E } \mathrm { - } \mathsf { C o v . } = \frac { 1 } { | \mathscr { T } | } \sum _ { t \in \mathscr { T } } E _ { t } .\tag{4}
$$

This metric evaluates evidence acquisition, i.e., whether models correctly transform visual evidence into grounded observations.

(3) Claim Coverage. Let $V _ { c , t } ^ { \mathrm { i n t } } = V _ { c , t } \ \backslash \ \{ c _ { t } ^ { * } \}$ denote the reference intermediate-claim nodes for task �, excluding the final root claim. The task-level score is

$$
C _ { t } = \frac { 1 } { | V _ { c , t } ^ { \mathrm { i n t } } | } \sum _ { c _ { i } \in V _ { c , t } ^ { \mathrm { i n t } } } m ( c _ { i } , r _ { t } ) ,\tag{5}
$$

defined only for tasks with $\lvert V _ { c , t } ^ { \mathrm { i n t } } \rvert ~ > ~ 0$ . Overall Claim Coverage is the average over this subset:

$$
\begin{array} { r l } & { \mathrm { C - C o v . } = \frac { 1 } { | \mathcal { T } ^ { \prime } | } \displaystyle \sum _ { t \in \mathcal { T } ^ { \prime } } C _ { t } , } \\ & { \mathcal { T } ^ { \prime } = \{ t \in \mathcal { T } : | V _ { c , t } ^ { \mathrm { i n t } } | > 0 \} . } \end{array}\tag{6}
$$

This metric evaluates evidence integration by measuring whether recovered evidence is composed into the intermediate claims required to support the final conclusion.

## 4. Experiments

We structure our experiments as a progressively constrained diagnostic evaluation that isolates failures in evidence acquisition, evidence integration, and final-answer reasoning.

Evaluation Models. We evaluate eight frontier multimodal models on Sci-MMR: five proprietary models—GPT-5.5 [OpenAI, 2026], Claude Opus 4.8 [Anthropic, 2026], Gemini 3.1 Pro [Google DeepMind, 2026a], GLM-5V-Turbo [Team et al., 2026], and Qwen3.7- Plus [Qwen Team, 2026]—and three open-weight models: Kimi K2.7 Code [Moonshot AI, 2026], MiniMax M3 [MiniMax, 2026], and Intern-S2- Preview-FP8 [InternLM, 2026].

Evaluation Settings. We evaluate all models under four progressively constrained settings that isolate diferent stages of evidence-grounded scientific reasoning.

• Caption-only. Models receive only figure captions, measuring the extent to which tasks can be solved from textual context alone.

• Direct Visual Reasoning. Models receive the original question and scientific figures, requiring them to acquire visual evidence and derive the final conclusion autonomously.

• Evidence-Hint Reasoning. Models additionally receive expert-annotated evidence statements, isolating evidence integration by removing the need for visual evidence acquisition.

• Agentic Tool-Use. Models are equipped with visual tools for cropping and magnifying regions of interest while preserving the original context, evaluating whether interactive evidence acquisition improves scientific reasoning.

Metrics. We evaluate free-form responses using three complementary metrics. Answer Accuracy (Acc.) measures whether the model reaches the correct scientific conclusion. Evidence Coverage (E-Cov.) measures whether the response recovers the required visual evidence. Claim Coverage (C-Cov.) measures whether the response recovers the intermediate claims connecting evidence to the final conclusion. Together, these metrics distinguish answer correctness from evidence acquisition and evidence integration. All responses are evaluated by Gemini 3.5 Flash [Google Deep-Mind, 2026b] using a unified rubric for answer correctness and graph coverage. Human agreement and cross-judge validation are reported in Supplementary Sec. C.3.

## 4.1. Main Results

Table 2 reports Answer Accuracy, Evidence Coverage, and Claim Coverage under Direct Visual Reasoning and Agentic Tool-Use across three task dificulty tiers.

Direct Visual Reasoning. GPT-5.5 achieves the highest overall accuracy at 67.7%, followed by Claude Opus 4.8 at 54.9% and Kimi K2.7 at 50.6%, while Intern-S2 ranks last at 26.8%. Performance deteriorates sharply with task dificulty for every model, but the extent of degradation varies substantially. GPT-5.5 drops from 96.8% on easy tasks to 29.6% on hard tasks, yet remains the strongest model on the hardest subset. In contrast, MiniMax-M3 declines from 93.5% to 4.9%, exhibiting the largest degradation of any model. These results suggest that strong performance on easier tasks does not reliably translate to complex multi-hop scientific reasoning, where deeper evidence integration becomes the dominant bottleneck.

Agentic Tool-Use. Equipping models with visual tools consistently improves performance, although the gains remain modest. Claude Opus 4.8 achieves the largest improvement at 5.1 points, whereas Gemini 3.1 Pro and Kimi K2.7 improve by only 2.1 points each. GPT-5.5 remains the strongest model overall at 69.8% and across all three dificulty tiers, indicating that interactive evidence acquisition alone does not close the gap on complex scientific reasoning.

Evidence and Claim Coverage. Table 2 also reports Evidence Coverage and Claim Coverage alongside Answer Accuracy. Across both evaluation settings, the three metrics exhibit diferent rankings, indicating that answer correctness, evidence acquisition, and evidence integration capture distinct aspects of scientific reasoning. A representative example is MiniMax-M3, which achieves the highest overall Evidence Coverage of any model under Direct Visual Reasoning at 61.5% and the highest Claim Coverage at 51.7%, yet ranks only fifth in Answer Accuracy at 45.5%, trailing GPT-5.5, Claude Opus 4.8, Kimi K2.7, and Gemini 3.1 Pro. Similar dissociations are observed within individual models, as detailed in Supplementary Section D.3, demonstrating that higher evidence or claim coverage does not necessarily translate into correct scientific conclusions.

![](images/9b42ea2fde503fc50213261382c5ef3771f6d5f7fa3a07db85b0c42956024d8e.jpg)  
Figure 4 | Accuracy gains from Direct Visual Reasoning to Evidence-Hint Reasoning by model (a) and dificulty (b).

Simple-task accuracy, tool-assisted gains, and evidence coverage each expose a limitation in current evaluation, but none pinpoint its cause. This leaves a deeper question open: do frontier multimodal agents genuinely perform multi-step, evidence-grounded scientific reasoning, or merely reach correct answers through other means? To this end, we propose three research questions targeting successive stages of the reasoning pipeline. RQ1 asks whether evidence availability alone is suficient for reliable reasoning. RQ2 asks whether autonomous evidence acquisition is the primary bottleneck. RQ3 asks specifically where the remaining errors originate. Together, RQ1–RQ3 trace the full pipeline from evidence acquisition to grounded reasoning, moving from establishing that a gap exists to locating its source and characterizing its precise manifestation.

RQ1: Is Evidence Availability Suficient for Reliable Scientific Reasoning? Figure 4 shows that supplying gold evidence statements under Evidence-Hint Reasoning raises accuracy for every model, but the gains fall well short of closing the gap to reliable performance. Averaged across models, accuracy rises by 27.4 points, from 46.1% to 73.5%, yet mean accuracy on hard tasks reaches only 54.8%, still far from ceiling. Models that scored lowest under Direct Visual Reasoning tend to gain the most: Intern-S2, the weakest model at 26.8%, gains 33.6 points, while GPT-5.5, the strongest at 67.7%, gains only 19.1 points, consistent with acquisition rather than reasoning capacity being their binding constraint. The pattern also holds across dificulty tiers: hard tasks gain the most at 44.1 points, compared to 27.2 points on medium tasks and only 5.8 points on easy tasks, indicating that evidence availability disproportionately helps on harder tasks without eliminating their dificulty. Evidence availability is therefore necessary but not suficient: even when the complete evidence set is handed to a model, integrating it into a correct scientific conclusion remains a substantial, unresolved challenge.

a Required-region coverage distribution  
![](images/65acb351cafc2aed1f7a2d94d53899d827d29335f44f33a98c8fe5db333a334f.jpg)

![](images/eb0134a280eadd3ab0510e21ddb5e5127ede6cd27e062279d1bd7d3748b3f0a9.jpg)  
Figure 5 | Crop localization and recovery at IoU = 0.5. (a) Required-region coverage among cropcalled runs. (b) Shares of initially incorrect, low-E-Cov. runs with increased E-Cov. or a rescued answer, grouped by annotated regions hit; bands show cross-model IQRs.

RQ2: Is Autonomous Evidence Acquisition the Primary Bottleneck? Figure 5 examines whether models can autonomously locate and recover the evidence required for a task. Among crop-called runs, models hit at least one annotated region in 62.1% of cases, but the distribution of coverage is heavily skewed: 37.9% of runs hit no required region at all, while only 12.7% achieve complete coverage, yielding a mean coverage of just 34.1%. This confirms that acquisition failure is widespread and severe. However, acquisition alone does not fully explain the accuracy gap. In panel (b), 46.4%–64.5% of runs improve in E-Cov., with the strongest recovery in the upper-hit bins, while answer rescue rises from 20.0% to 38.7%. Greater region access is therefore associated with better recovery. However, localization alone remains insuficient: even with at least four hits, only 38.7% of answers are rescued. This gap indicates that locating relevant evidence does not guarantee its correct extraction and integration, motivating the error analysis in RQ3. We additionally assess sensitivity to the IoU criterion using thresholds of 0.3 and 0.7 around the default of 0.5 (Supplementary Sec. F.4).

RQ3: When Reasoning Fails, What Specifically Goes Wrong? RQ2 shows that autonomous evidence acquisition is a dominant but incomplete explanation for the accuracy–coverage gap, leaving open what accounts for the remaining errors. Figure 6 addresses this by decomposing all incorrect Direct Visual Reasoning outputs into six failure subtypes using a cross-setting routing procedure. Evidence-related failures account for 57.2% of errors overall, dominated by access and localization failures at 40.1%, with gold underuse at 14.0% and visual extraction errors at 3.1% contributing smaller shares. Reasoning-related failures account for a further 31.8%, split between interpretation failures at 27.5% and final-answer mismatches at 4.2%; the remaining 11.0% fall outside these categories. This breakdown is consistent across nearly all eight evaluated models: access and localization failures remain the largest single subtype for every model, ranging from 37% for Claude Opus 4.8 to 48% for Kimi K2.7, while interpretation failures form the second-largest category for most models. Taken together with RQ2, this fine-grained attribution confirms that current multimodal models fail along two largely distinct axes – acquiring complete evidence from raw visuals, and correctly interpreting evidence once retrieved – rather than along a single, dominant failure mode.

## 5. Limitations

Our Sci-MMR provides a large-scale benchmark for evaluating evidence-grounded scientific reasoning and enables systematic analysis of evidence acquisition and evidence integration. Nevertheless, several limitations should be acknowledged. First, an Argument Graph represents one plausible reconstruction of a paper’s evidential structure rather than a unique ground truth. Evidence granularity, intermediate claims, and support relations may admit alternative yet valid interpretations. Although our LLM-assisted pipeline incorporates expert verification, some degree of construction subjectivity is unavoidable. Second, Evidence Coverage and Claim Coverage evaluate the evidence and claims explicitly expressed in the final response rather than the model’s internal reasoning process. In addition, although our evaluation demonstrates strong cross-judge agreement, LLM-based assessment may still be afected by semantic ambiguity, response style, and evaluator bias [Zheng et al., 2023, Zeng et al., 2024, Chen et al., 2025].

![](images/c9017c7a0e42f0e3833080e932eff66bade39f139a2436a60447e148c4b923dd.jpg)  
Figure 6 | Error patterns among incorrect Direct Visual Reasoning outputs. The left panel shows the overall distribution, and the right panel shows the distribution for each model. Bubble size indicates the percentage of a model’s errors.

## 6. Conclusion

We introduced Sci-MMR, a benchmark of 235 tasks for evaluating multi-step, evidencegrounded scientific reasoning. Its argument graphs connect visual sources, evidence, citationgrounded scientific knowledge, and claims, enabling evaluation beyond answer accuracy alone. Across eight frontier multimodal models, supplying gold evidence substantially improves performance, while equipping models with visual cropping tools yields only modest gains. Errors persist even when evidence is explicitly provided, revealing challenges in both evidence acquisition and evidence integration. These findings show that answer accuracy alone can overestimate scientific reasoning reliability, motivating evaluation that traces the complete, structured evidence behind a conclusion.

Looking ahead, a natural next step is to extend Sci-MMR from within-paper argument graphs to cross-paper argument modeling, enabling evaluation of whether systems can connect complementary or conflicting evidence and claims across scientific articles.

## References

Kuangshi Ai, Haichao Miao, Kaiyuan Tang, Nathaniel Gorski, Jianxin Sun, Guoxi Liu, Helgi I. Ingólfsson, David Lenz, Hanqi Guo, Hongfeng Yu, Teja Leburu, Michael Molash, Bei Wang, Tom Peterka, Chaoli Wang, and Shusen Liu. Scivisagentbench: A benchmark for evaluating scientific data analysis and visualization agents. CoRR, abs/2603.29139, 2026. doi: 10.48550/ARXIV.2603.29139. URL https://doi.org/10.48550/arXiv.2603.29139.

Abolfazl Ansari, Delvin Ce Zhang, Zhuoyang Zou, Wenpeng Yin, and Dongwon Lee. M2- verify: A large-scale multidomain benchmark for checking multimodal claim consistency. CoRR, abs/2604.01306, 2026. doi: 10.485 50/ARXIV.2604.01306. URL https://doi.org/ 10.48550/arXiv.2604.01306.

Anthropic. Claude Opus 4.8 system card, May 2026. URL https://www.anthropic.com/clau de-opus-4-8-system-card. Accessed: 2026-07- 27.

Elad Ben-Avraham, Changhao Li, Ron Dorfman, Roy Ganz, Oren Nuriel, Amir Dudai, Aviad Aberdam, Noah Flynn, Elman Mansimov, Aditya Kalyanpur, and Ron Litman. DREAM: deep research evaluation with agentic metrics. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens, editors, Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2026, San Diego, California, United States, July 2-7, 2026, pages 9879–9904. Association for Computational Linguistics, 2026. URL https://aclanthology.org/2026.acl-long. 448/.

Zhi-Yuan Chen, Hao Wang, Xinyu Zhang, Enrui Hu, and Yankai Lin. Beyond the surface: Measuring self-preference in LLM judgments. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, EMNLP 2025, Suzhou, China,

November 4-9, 2025, pages 1653–1672. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.EMNLP-MAIN.86. URL https://doi.org/10.18653/v1/2025.emnlp-m ain.86.

Song Dai, Yibo Yan, Jiamin Su, Dongfang Zihao, Yubo Gao, Yonghua Hei, Jungang Li, Junyan Zhang, Sicheng Tao, Zhuoran Gao, and Xuming Hu. Physicsarena: The first multimodal physics reasoning benchmark exploring variable, process, and solution dimensions. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Findings of the Association for Computational Linguistics: EMNLP 2025, Suzhou, China, November 4-9, 2025, pages 17290–17316. Association for Computational Linguistics, 2025. doi: 10.186 53/V1/2025.FINDINGS-EMNLP.937. URL https://doi.org/10.18653/v1/2025.finding s-emnlp.937.

Yifan Du, Zikang Liu, Jinbiao Peng, Jie Wu, Junyi Li, Jinyang Li, Wayne Xin Zhao, and Ji-Rong Wen. Towards long-horizon agentic multimodal search. CoRR, abs/2604.12890, 2026. doi: 10.48550/ARXIV.2604.12890. URL https: //doi.org/10.48550/arXiv.2604.12890.

Google DeepMind. Gemini 3.1 Pro model card, February 2026a. URL https://deepmind.g oogle/models/model-cards/gemini-3-1-pro. Accessed: 2026-07-27.

Google DeepMind. Gemini 3.5 Flash model card, May 2026b. URL https://deepmind.googl e/models/model-cards/gemini-3-5-flash/. Accessed: 2026-07-27.

Juraj Gottweis, Wei-Hung Weng, Alexander N. Daryin, Tao Tu, Anil Palepu, Petar Sirkovic, Artiom Myaskovsky, Felix Weissenberger, Keran Rong, Ryutaro Tanno, Khaled Saab, Dan Popovici, Jacob Blum, Fan Zhang, Katherine Chou, Avinatan Hassidim, Burak Gokturk, Amin Vahdat, Pushmeet Kohli, Yossi Matias, Andrew Carroll, Kavita Kulkarni, Nenad Tomasev, Yuan Guan, Vikram Dhillon, Eeshit Dhaval Vaishnav, Byron Lee, Tiago R. D. Costa, José R. Penadés, Gary Peltz, Yunhan Xu, Annalisa Pawlosky, Alan Karthikesalingam, and Vivek

Natarajan. Towards an AI co-scientist. CoRR, abs/2502.18864, 2025. doi: 10.48550/ARXIV .2502.18864. URL https://doi.org/10.48550 /arXiv.2502.18864.

Peizhou Huang, Zixuan Zhong, Zhongwei Wan, Donghao Zhou, Samiul Alam, Xin Wang, Zexin Li, Zhihao Dou, Li Zhu, Jing Xiong, Chaofan Tao, Yan Xu, Dimitrios Dimitriadis, Tuo Zhang, and Mi Zhang. Mmdeepresearch-bench: A benchmark for multimodal deep research agents. CoRR, abs/2601.12346, 2026. doi: 10.48550/ARXIV.2601.12346. URL https: //doi.org/10.48550/arXiv.2601.12346.

InternLM. Intern-S2-Preview-FP8 model card, 2026. URL https://huggingface.co/internlm/ Intern-S2-Preview-FP8. Accessed: 2026-07-27.

Anne Lauscher, Goran Glavas, and Simone Paolo Ponzetto. An argument-annotated corpus of scientific publications. In Noam Slonim and Ranit Aharonov, editors, Proceedings of the 5th Workshop on Argument Mining, ArgMining@EMNLP 2018, Brussels, Belgium, November 1, 2018, pages 40–46. Association for Computational Linguistics, 2018. doi: 10.18653/V1/W18-5 206. URL https://doi.org/10.18653/v1/w1 8-5206.

Chuhan Li, Ziyao Shangguan, Yilun Zhao, Deyuan Li, Yixin Liu, and Arman Cohan. M3sciqa: A multi-modal multi-document scientific QA benchmark for evaluating foundation models. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Findings of the Association for Computational Linguistics: EMNLP 2024, Miami, Florida, USA, November 12-16, 2024, volume EMNLP 2024 of Findings of ACL, pages 15419–15446. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.F INDINGS-EMNLP.904. URL https://doi.org/ 10.18653/v1/2024.findings-emnlp.904.

Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, and Jingren Zhou. Websailor: Navigating super-human reasoning for web

agent. CoRR, abs/2507.02592, 2025. doi: 10.48550/ARXIV.2507.02592. URL https: //doi.org/10.48550/arXiv.2507.02592.

Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob N. Foerster, Jef Clune, and David Ha. The AI scientist: Towards fully automated open-ended scientific discovery. CoRR, abs/2408.06292, 2024. doi: 10.48550/ARXIV.2408.06292. URL https://doi.org/10.48550/arXiv.2408.06292.

MiniMax. MiniMax M3, June 2026. URL https:// www.minimax.io/models/text/m3. Accessed: 2026-07-27.

Moonshot AI. Kimi K2.7 Code model card, 2026. URL https://huggingface.co/moonshotai/Ki mi-K2.7-Code. Accessed: 2026-07-27.

Eli Moser and Robert E. Mercer. Use of claim graphing and argumentation schemes in biomedical literature: A manual approach to analysis. In Elena Cabrio and Serena Villata, editors, Proceedings ofthe 7th Workshop on Argument Mining, ArgMining 2020, Barcelona, Spain (Online), December 13, 2020, pages 88– 99. Association for Computational Linguistics, 2020. URL https://aclanthology.org/2020.ar gmining-1.10/.

Xuying Ning, Dongqi Fu, Tianxin Wei, Mengting Ai, Jiaru Zou, Ting-Wei Li, Hanghang Tong, Yada Zhu, Hendrik F. Hamann, and Jingrui He. Mc-search: Evaluating and enhancing multimodal agentic search with structured long reasoning chains. CoRR, abs/2603.00873, 2026. doi: 10.48550/ARXIV.2603.00873. URL https://doi.org/10.48550/arXiv.2603.00873.

OpenAI. GPT-5.5 system card, April 2026. URL https://openai.com/index/gpt-5-5-system-c ard/. Accessed: 2026-07-27.

Shraman Pramanick, Rama Chellappa, and Subhashini Venugopalan. SPIQA: A dataset for multimodal question answering on scientific papers. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,

BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper\_files/paper /2024/hash/d74033a247989e8f6f3bf9e0c96 29fb5-Abstract-Datasets\_and\_Benchmarks\_T rack.html.

Qwen Team. Qwen3.7-Plus: Multimodal agent intelligence, June 2026. URL https://qwen.ai/ blog?id=qwen3.7-plus. Accessed: 2026-07-27.

Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Michael Moor, Zicheng Liu, and Emad Barsoum. Agent laboratory: Using LLM agents as research assistants. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Findings of the Association for Computational Linguistics: EMNLP 2025, Suzhou, China, November 4-9, 2025, pages 5977–6043. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.FINDI NGS-EMNLP.320. URL https://doi.org/10.1 8653/v1/2025.findings-emnlp.320.

Manasi Sharma, Chen Bo Calvin Zhang, Chaithanya Bandi, Clinton Wang, Ankit Aich, Huy Nghiem, Tahseen Rabbani, Ye Htet, Brian Jang, Sumana Basu, Aishwarya Balwani, Denis Peskof, Marcos Ayestaran, Sean M. Hendryx, Brad Kenstler, and Bing Liu. Researchrubrics: A benchmark of prompts and rubrics for evaluating deep research agents. CoRR, abs/2511.07685, 2025. doi: 10.48550/ARXI V.2511.07685. URL https://doi.org/10.48550/arXiv.2511.07685.

Junyoung Sung, Seungwoo Lyu, Minjun Kim, Sumin An, Arsha Nagrani, and Paul Hongsuck Seo. CRIT: graph-based automatic data synthesis to enhance cross-modal multi-hop reasoning. CoRR, abs/2604.01634, 2026. doi: 10.48550/ARXIV.2604.01634. URL https: //doi.org/10.48550/arXiv.2604.01634.

V Team, Wenyi Hong, Xiaotao Gu, Ziyang Pan, Zhen Yang, Yuting Wang, Yue Wang, Yuanchang Yue, Yu Wang, Yanling Wang, Yan Wang, Xijun Liu, Wenmeng Yu, Weihan Wang, Wei Li, Shuaiqi Duan, Sheng Yang, Ruiliang Lv, Mingdao Liu, Lihang Pan, Ke Ning, Junhui Ji, Jinjiang Wang, Jing Chen, Jiazheng Xu, Jiale

Zhu, Jiale Cheng, Ji Qi, Guobing Gan, Guo Wang, Cong Yao, Zijun Dou, Zihao Zhou, Zihan Wang, Zhiqi Ge, Zhijie Li, Zhenyu Hou, Zhao Xue, Zehui Wang, Zehan Qi, Zehai He, Yutao Zhang, Yusen Liu, Yukuo Cen, Yuchen Li, Yuan Wang, Yu Yang, Yongbin Liu, Yijian Lu, Yifan Xu, Yanzi Wang, Yanxiao Zhao, Yanfeng Wang, Yadong Xue, Yabo Xu, Xinyu Zhang, Xinyu Liu, Xiao Liu, Wenyi Zhao, Wenkai Li, Tianyu Tong, Tianshu Zhang, Shudan Zhang, Shengdong Yan, Qinkai Zheng, Mingde Xu, Licheng Bao, lat Long long, Jiaxing Xu, Jiaxin Fan, Jiawen Qian, Jiali Chen, Jiahui Lin, Jiadai Sun, Haozhi Zheng, Haoran Wang, Haochen Li, Hanyu Lai, Han Xu, Fan Yang, Dan Zhang, Da Yin, Chuangxin Zhao, Chengcheng Wu, Boyan Shi, Bowen Lv, Bowei Jia, Bo Li, Bin Chen, Baoxu Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, and Jie Tang. Glm-5v-turbo: Toward a native foundation model for multimodal agents, 2026. URL https://arxiv.org/abs/2604.26752.

Simone Teufel, Jean Carletta, and Marc Moens. An annotation scheme for discourse-level argumentation in research articles. In EACL 1999, 9th Conference of the European Chapter of the Association for Computational Linguistics, June 8-12, 1999, University of Bergen, Bergen, Norway, pages 110–117. The Association for Computer Linguistics, 1999. URL https://aclanthology.org/E99-1015/.

Duong T. Tran, Trung-Kien Tran, Manfred Hauswirth, and Danh Le Phuoc. Reasonvqa: A multi-hop reasoning benchmark with structural knowledge for visual question answering. In IEEE/CVF International Conference on Computer Vision, ICCV 2025, Honolulu, HI, USA, October 19-25, 2025, pages 18793–18803. IEEE, 2025. doi: 10.1109/ICCV51701.2025.01746. URL https://doi.org/10.1109/ICCV51701.2025.0 1746.

Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Li, Botian Shi, Yu Qiao, Dahua Lin, and Conghui He. Mineru: An open-source solution for precise document content extraction. CoRR, abs/2409.18839,

2024. doi: 10.48550/ARXIV.2409.18839. URL https://doi.org/10.48550/arXiv.2409.18839.

Chengye Wang, Yifei Shen, Zexi Kuang, Arman Cohan, and Yilun Zhao. Sciver: Evaluating foundation models for multimodal scientific claim verification. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 8562–8579. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.ACL-L ONG.420. URL https://doi.org/10.18653/v1/ 2025.acl-long.420.

Shenzhi Wang, Shixuan Liu, Jingren Zhou, Chang Gao, Xiong-Hui Chen, Binghai Wang, An Yang, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Hopchain: Multi-hop data synthesis for generalizable vision-language reasoning. CoRR, abs/2603.17024, 2026. doi: 10.48550/ARXIV .2603.17024. URL https://doi.org/10.48550 /arXiv.2603.17024.

Fangda Ye, Yuxin Hu, Pengxiang Zhu, Yibo Li, Ziqi Jin, Yao Xiao, Yibo Wang, Lei Wang, Zhen Zhang, Lu Wang, Yue Deng, Bin Wang, Yifan Zhang, Liangcai Su, Xinyu Wang, He Zhao, Chen Wei, Qiang Ren, Bryan Hooi, An Bo, Shuicheng Yan, and Lidong Bing. Miroeval: Benchmarking multimodal deep research agents in process and outcome. CoRR, abs/2603.28407, 2026. doi: 10.48550/ARXIV .2603.28407. URL https://doi.org/10.48550 /arXiv.2603.28407.

Yu Zeng, Wenxuan Huang, Zhen Fang, Shuang Chen, Yufan Shen, Yishuo Cai, Xiaoman Wang, Zhenfei Yin, Lin Chen, Zehui Chen, Shiting Huang, Yiming Zhao, Xu Tang, Yao Hu, Philip Torr, Wanli Ouyang, and Shaosheng Cao. Vision-deepresearch benchmark: Rethinking visual and textual search for multimodal large language models. CoRR, abs/2602.02185, 2026. doi: 10.48550/ARXIV.2602.02185. URL https://doi.org/10.48550/arXiv.2602.02185.

Zhiyuan Zeng, Jiatong Yu, Tianyu Gao, Yu Meng, Tanya Goyal, and Danqi Chen. Evaluating

large language models at evaluating instruction following. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. Open-Review.net, 2024. URL https://openreview.n et/forum?id=tr0KidwPLc.

Yanjun Zhao, Tianxin Wei, Jiaru Zou, Xuying Ning, Yuanchen Bei, Lingjie Chen, Simmi Rana, Wendy H. Yang, Hanghang Tong, and Jingrui He. PAPERMIND: benchmarking agentic reasoning and critique over scientific papers in multimodal llms. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens, editors, Findings of the Association for Computational Linguistics, ACL 2026, San Diego, California, United States, July 2-7, 2026, pages 10457–10474. Association for Computational Linguistics, 2026. URL https://aclanthology.o rg/2026.findings-acl.508/.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files/paper/202 3/hash/91f18a1287b398d378ef22505bf41 832-Abstract-Datasets\_and\_Benchmarks.htm l.

## Appendix

## A. Benchmark Composition and Audit

The Sci-MMR benchmark contains 235 questions spanning biology, chemistry, computer science, and physics. We first report the source-pool and benchmark composition, then characterize the structural and visual demands of individual questions, provide an audited graph example, and document expert verification.

## A.1. Temporal, Disciplinary, and Domain Coverage

The construction pool contains 800 deduplicated papers: 50 from the Nature family, 40 from the Science family, 30 from the Cell family, 60 from PNAS, and 620 from other sources. MinerU parsing succeeds for 700 papers, argument graphs are available for 475, and 300 papers contribute candidate tasks. The task-level validation flow follows the two automated rates reported in the main text: deterministic validation removes 54% of the candidate-task inventory, and the subsequent rubric stage filters a further 17%. Failed candidates are revised and re-evaluated. The final expert-review cohort contains 260 task IDs; 25 are rejected and 235 are retained in the benchmark. The final 235 tasks come from 170 papers; 119 papers contribute one task, 39 contribute two, 10 contribute three, and two contribute four.

For temporal coverage, we record each final task’s source paper by its latest release or version year, which may difer from its original publication year. The 235 benchmark questions are not uniformly distributed across 2020–2026: the respective counts are 2, 2, 3, 16, 46, 38, and 128. Thus, 128 questions (54.5%) use papers whose latest release or version is dated 2026.

Table 3 summarizes the task-level construction flow; paper-level counts are upstream inventory counts.

Table 4 reports the complete discipline– primary-domain partition using the intrinsic sample taxonomy. Physics contributes 88 questions (37.4%), followed by computer science with 63 (26.8%), biology with 50 (21.3%), and chemistry with 34 (14.5%). Within biology, the largest domains are microbial cell biology and physiology (12), microbial genomics and metagenomics (10), and clinical microbiology and infectious disease (6). Chemistry is led by electrochemistry and electrocatalysis (9), followed by organic synthesis and mechanism and quantum and computational chemistry (5 each). Computer science is led by machine learning and reasoning (22), language models and multimodal systems (14), and AI agents and tool-use systems (11). Physics is led by condensed matter and materials physics (23), astrophysics and cosmology (14), and fluids and plasma physics (11). The remaining domains contain between one and nine questions each. The resulting mix emphasizes physics and computer science while maintaining broad topical coverage: the largest primary domain contains 23 questions, or 9.8% of the benchmark.

## A.2. Structural and Visual Complexity

Figure 7 places related quantities next to one another: dependency complexity occupies the top row, while the lower rows form columns for graph size, evidence composition, and claim/visual load. Layer width is the maximum number of nodes at any dependency level, depth is the length of the longest dependency chain, and branching is the maximum number of direct dependencies from one node. These distributions are compact but nontrivial, with median layer width four, depth three, and branching three. The deterministic structural check requires a minimum dependency depth of 3. In the final 235-task benchmark, the depth counts are 125, 84, 25, and 1 at depths 3, 4, 5, and 6, respectively. The graph-size column gives medians of seven nodes and eight edges. The evidence column shows a median of three evidence nodes per question, while 500 of the 570 cited figure/table uses (87.7%) support no more than two evidence nodes. The claim/visual column shows a median of one claim node and five manually annotated subfigures per question; the subfigure count is more heterogeneous, with a mean of 8.97 and a maximum of 104. Thus, most local visual references have focused evidential roles even though a complete question may combine several references within a larger reasoning graph.

## A.3. Audited Argument-Graph Example

Figures 8 and 9 provide an audited example of the graph representation used by the benchmark. The example complements the aggregate distributions above with a concrete view of a higher-complexity graph and its source traceability. The complete graph contains 24 nodes and 32 directed relations at dependency depth five: one final claim and six intermediate claims are supported by 13 evidence nodes, each of which is grounded in one of four source figures. The claim hierarchy separates structural and electronic properties, electrochemical performance, and the proposed storage mechanism, while retaining the individual evidence values and source-panel identifiers.

<table><tr><td>Stage or event</td><td>Reported outcome</td><td>Identifier-preserving disposition and reason</td></tr><tr><td>Deduplicated paper pool</td><td>800 papers</td><td>Upstream source inventory; no task IDs yet.</td></tr><tr><td>MinerU parsing available</td><td>700 papers</td><td>Papers retained for downstream graph construction.</td></tr><tr><td>Argument graphs available</td><td>475 papers</td><td>Papers with a usable graph.</td></tr><tr><td>Candidate-task generation</td><td>300 papers</td><td>Candidate tasks receive an immutable identifier before validation.</td></tr><tr><td>Deterministic validation Rubric and hard checks</td><td>54% removed</td><td>Depth &lt; 3 or invalid graph; surviving IDs retain their identity. Failed candidates are revised and re-evaluated. The 66 below-threshold</td></tr><tr><td></td><td>17% further filtered</td><td>flags, 140 rewrite triggers, and 130 accepted rewrites are overlapping event counts.</td></tr><tr><td>Expert review</td><td>260 tasks</td><td>Two independent experts check each surviving ID; revisions retain the task ID through adjudication.</td></tr><tr><td>Released benchmark</td><td>235 tasks</td><td>25 expert-review rejections are terminal dispositions; the retained IDs form the released benchmark.</td></tr></table>

Table 3 | Task construction and quality-control flow. Stage rates use the single candidate-task denominator reported in the main text. Immutable IDs persist through rewrites and adjudication; the 54% and 17% entries are rates, while rewrite flags are overlapping events.
<table><tr><td>Discipline</td><td>Primary domain</td></tr><tr><td>Biology (50)</td><td>microbial cell biology &amp; physiology (12) microbial genomics &amp; metagenomics (10) clinical microbiology &amp; infectious disease (6) microbial ecology &amp; microbiome (5) plant genomics &amp; crop biology (4) computational biology &amp; bioinformatics (3) public health microbiology &amp; epidemiology (3)</td></tr><tr><td>Chemistry (34)</td><td>structural biology &amp; cryo-electron mi- croscopy (2) biotechnology &amp; applied microbiology (1) electrochemistry &amp; electrocatalysis (9) organic synthesis &amp; mechanism (5) quantum &amp; computational chemistry (5) molecular modeling &amp; cheminformatics (4)</td></tr><tr><td>(63)</td><td>polymer &amp; macromolecular chemistry (4) reaction informatics &amp; synthesis planning (4) toxicology &amp; environmental chemistry (2) catalysis &amp; energy conversion (1) Computer science machine learning &amp; reasoning (22) language models &amp; multimodal systems (14)</td></tr><tr><td>Physics (88)</td><td>AI agents &amp; tool-use systems (11) code generation &amp; software engineering (6) computer vision &amp; visual computing (6) embodied AI &amp; robotics (3) retrieval &amp; search systems (1) condensed matter &amp; materials physics (23) astrophysics &amp; cosmology (14) fluids &amp; plasma physics (11) climate &amp; atmospheric physics (9) geophysics &amp; seismology (9)</td></tr></table>

Table 4 | Question counts for all 35 primary domains. Counts in parentheses sum to the discipline totals shown in the column headers.

Figure 9 expands the source side of the same graph. Instead of shrinking each composite source figure into a thumbnail, it shows evidence-specific crops labeled by both graph node and source panel. This view allows the numerical observations, curves, spectra, and computed structures to remain legible while preserving the E#–F# mapping used by the graph.

In the audited example, arrows are read from a conclusion to the evidence or lower-level claims that support it, and from an observation to the source panel in which it appears. For conceptual exposition, the main paper presents the same relations in the opposite inferential order—from visual evidence to claims. The two descriptions therefore encode the same dependency structure; only the direction in which the reader follows the arrows difers.

## A.4. Expert Verification Protocol

Six experts verify the benchmark during construction. Four are discipline-matched reviewers with research backgrounds in biology, chemistry, computer science, or physics; two are senior reviewers with experience in scientific-figure interpretation, benchmark construction, or crossdisciplinary review. All have at least master’s-level training in a relevant field. Every task is independently checked by one discipline-matched expert and one senior cross-reviewer. A second senior reviewer, who did not participate in the initial review of that task, adjudicates disagreements.

The review covers the question, supporting region boxes, evidence statements, claim and

## Benchmark core sample characteristics

Median Mean P25–P75

Teal top row: dependency complexity; lower columns: graph size, evidence, and claim/visual load All panels except h are question-level; h summarizes cited uses. Node counts include basis/reference nodes; claim counts include the final claim

![](images/0bca72a5f4e601206aa548bbadd53926c8fb7a6638b93ebf33b401ac9922143d.jpg)

b Maximum dependency depth  
![](images/f39e4a3941527900fed72ad55c9279460b4359b0ffbd1fb6fe5183202ff050cf.jpg)

c Maximum dependency branching  
![](images/881116f982ffe8e924b991a9856c5eba980c43daca0eb3616b09dcb2b76fdb2d.jpg)

d Reasoning-graph nodes  
![](images/6df4cd62c2419e6a83e1d72482eebe2519d8f0474473295c8d3fdb0dc950391c.jpg)

![](images/de14afec87ce332a07fc1b91fb7a4f9e00e4eac02327fba25515654d372494b8.jpg)

f Claim nodes (including final claim)  
![](images/dc400847952fb026934893370b25824551e4b4050c7490713604d6a87eba0b99.jpg)

![](images/3a4f54537fb8540ad06e7c1c5ae778a906200277de79ea75a1eb79346bea47db.jpg)

h Evidence nodes per cited figure/table  
![](images/ae3d79b813a9e2aab0dcae6f0e01dd033d961370b4ee215aade2f7dd9561bbba.jpg)

i Manually annotated subfigures  
![](images/e28ecf97d8946e054b55883f4c219ff1729f7007604e1a74b123ca45bf869031.jpg)  
Figure 7 | Core structural and visual characteristics of the benchmark. All panels except (h) summarize 235 questions; panel (h) summarizes 570 cited figure/table uses. The teal top row, panels (a)–(c), shows dependency width, depth, and branching. In the lower two rows, the blue column, panels (d) and (g), shows reasoning-graph nodes and edges; the green column, panels (e) and (h), shows question-level evidence nodes and evidence nodes per cited use; and the pink column, panels (f) and (i), shows claim nodes and manually annotated subfigures. Solid and dashed lines mark the median and mean, respectively, and cream spans show the interquartile range.

knowledge nodes, graph edges, reference final answer, overall answerability, and answer leakage. Discipline experts identify supporting regions and rewrite evidence as conservative propositions directly observable from the visual material. Crossreviewers independently issue an accept-or-revise decision after inspecting the original visual, panel identity, and evidence coverage without seeing the first reviewer’s decision. Region annotations are directly accepted when they refer to the same semantic region and reach IoU≥ 0.7; otherwise a third reviewer adjudicates and, when needed, redraws the region. Evidence review checks entities, conditions, directions, comparisons, values, and units, and rejects unsupported inference or answer leakage. Graph review verifies node provenance and meaning, each support relation, connectivity and acyclicity, and the complete evidence → intermediate-claim → root-claim path. Initial, first-review, second-review, adjudication, and final-consensus versions are retained.

![](images/8285d2e525dc6bdb95b8ea64e55003a260a45b427bbc8b7cc21b4b5889ddbee0.jpg)  
b Enlarged claim–evidence graph

![](images/623ffa9646b759048140fb11e36b31016a10602ba75045e84a331e3d6081852a.jpg)

![](images/43762e815ade591385a6e17a89ef633c74df2cb18cddb91fae79bcef539b5975.jpg)

![](images/1a32cc9e8dfc74b784d15d0d881f4d4411e1c67429c25dc8280f8f5a63ead7d1.jpg)  
Figure 8 | Audited full argument graph for one benchmark question. Panel (a) summarizes the final claim and its structural, performance, and mechanism branches. Panel (b) expands all intermediate claims and evidence nodes. Arrows link each conclusion to the evidence or lower-level claim that supports it, and each F# label identifies the source panel for an evidence node. The graph contains seven claim nodes, 13 evidence nodes, four source-figure nodes, and 32 relations. The source claim’s use of “stable” is retained in the claim text, while the separately evidenced propositions are represented by explicit evidence nodes.

All 260 handof tasks receive expert review, including 25 tasks that are rejected at this stage. Among the 235 retained tasks, 145 are directly accepted, 69 require minor revision, and 21 require major revision or redrawing; 24 tasks require third-reviewer adjudication. Ninety retained tasks require at least one object-level modification, while 145 pass without modification. Table 5 reports the complete task-level disposition together with the object-level decisions; the adjudication column overlaps the disposition columns. Table 6 reports agreement between the two initial reviewers before adjudication.

![](images/f7f0a302998f624ab2f072404159a4820cdf7c21585adbcc2d5757ede3985af9.jpg)  
Figure 9 | Evidence grounding for the audited argument graph. F1 grounds the electronic-structure evidence in E1; F2 grounds the room- and low-temperature performance evidence in E2–E7; F3 grounds the kinetic and spectroscopic evidence in E8–E11; and F4 grounds the computed storage mechanism in E12–E13. Each crop preserves the original source content and is labeled with the corresponding evidence node and source-panel location. Cropping is used only to improve legibility; source identity and panel-level traceability remain unchanged.

## B. Question Quality Control

The construction process begins with deterministic (rule-based) validation followed by an eightdimensional language-model quality assessment. In the single candidate-task denominator used in the main text, deterministic validation removes 54%, and the rubric stage filters a further 17%. The depth check uses a minimum dependency depth of 3 together with graph-structure validity. Rubric failures, high-risk explicit checks, or failures on critical dimensions trigger conditional rewrites; 66 below-threshold flags, 140 rewrite triggers, and 130 accepted rewrites are overlapping event counts rather than a partition of tasks. After automated selection, 260 tasks enter expert review; 25 are rejected and 235 form the final benchmark. Table 3 records the stage rates, terminal outcomes, and rewrite events.

The language-model assessment evaluates the generated question stem rather than a model response. Its purpose is to verify that a blind multihop question points to the intended visual materials, makes the evidence-to-claim path recoverable and genuinely necessary, states a clear scientific target, and does not reveal or presuppose the answer. Table 7 gives the complete rubric in operational terms.

The overall passing threshold is 85. The three critical dimensions are anchor coverage, reasoning dependency, and non-leakage; a normalized score below 0.5 on any of them triggers revision even when the weighted total is near the threshold. Structural validation first checks dependency depth, graph validity, and the required structural elements. Question-level hard checks then flag high-risk defects that can be recognized directly, including exam-essay or proof-task phrasing, exposed solution procedures, over-generic stems, answer leakage, and answer-shaping formulations. A candidate is conditionally revised when a high-risk check fires, the language model recommends revision, the overall score is below 85, or a critical dimension falls below 0.5. Revision preserves the selected visual anchors, intended answer format, and multi-hop dependency while applying the smallest targeted wording change.

This construction-time rubric is distinct from the graph-aligned evaluation protocol below: it scores whether a question is suitable for inclusion, whereas Answer Accuracy, E-Cov., and C-Cov. score model responses to an accepted task.

## C. Evaluation Protocol

## C.1. Evaluation Settings

We evaluate 235 questions with eight models under Caption-only, Direct Visual Reasoning, Evidence-Hint Reasoning, and Agentic Tool-Use, yielding 1,880 planned model–question observations per setting. Caption-only provides the question and complete source-paper figure/table captions, without image pixels, evidence hints, or paper body text. Direct Visual Reasoning provides the question and original visual references. Evidence-Hint Reasoning retains those inputs and supplies the annotated evidence text. Agentic Tool-Use preserves the original visual context and enables the model to crop and magnify selected regions. Answer accuracy always uses the planned denominator: missing evaluable answers count as incorrect. The numbers of available answers are 1,876/1,880 for Caption-only, 1,866/1,880 for Direct Visual Reasoning, 1,868/1,880 for Evidence-Hint Reasoning, and 1,873/1,880 for Agentic Tool-Use.

Dificulty is fixed before comparing models or evaluation settings. Each question receives a dificulty score from a weighted combination of graphstructural characteristics and visual-evidence demands, computed from the benchmark annotations rather than from model responses. The resulting strata contain 81 hard, 92 medium, and 62 easy questions, and the same fixed labels are used for every model and setting.

## C.2. Coverage Metrics

Evidence coverage (E-Cov.) measures the fraction of required evidence nodes explicitly covered by the answer, and claim coverage (C-Cov.) analogously measures coverage of applicable intermediate-claim nodes. E-Cov. has 1,880 valid observations per setting. C-Cov. is defined for questions with a required intermediate claim and has 880 valid observations per setting. Together, the two metrics operationalize supportpath reconstruction through observable answer

<table><tr><td>Object</td><td>Reviewed</td><td>Direct accept</td><td>Minor</td><td>Major/redraw</td><td>Rejected</td><td>Adjudicated</td></tr><tr><td>Benchmark tasks</td><td>260</td><td>145</td><td>69</td><td>21</td><td>25</td><td>24</td></tr><tr><td>Questions</td><td>235</td><td>180</td><td>45</td><td>10</td><td>1</td><td>14</td></tr><tr><td>Region boxes</td><td>2,107</td><td>1,790</td><td>242</td><td>75</td><td>一</td><td>105</td></tr><tr><td>Evidence statements</td><td>897</td><td>673</td><td>180</td><td>44</td><td></td><td>81</td></tr><tr><td>Claim/knowledge nodes</td><td>649</td><td>520</td><td>98</td><td>31</td><td>一</td><td>52</td></tr><tr><td>Graph edges</td><td>2,336</td><td>1,986</td><td>268</td><td>82</td><td>一</td><td>128</td></tr><tr><td>Reference final answers</td><td>235</td><td>211</td><td>19</td><td>5</td><td>一</td><td>7</td></tr></table>

Table 5 | Construction-stage expert review. The task row covers all 260 reviewed tasks: the 235 retained tasks are partitioned into direct acceptance, minor revision, and major revision/redrawing, while 25 are rejected. Rejected tasks are represented by their terminal task outcome; the remaining rows summarize final objects from retained tasks. Adjudication is an overlapping count.
<table><tr><td>Decision unit</td><td>n</td><td>Exact agreement</td><td>Cohen&#x27;s κ</td></tr><tr><td>Region accept/revise</td><td>2,107</td><td>91%</td><td>0.80</td></tr><tr><td>Evidence-statement decision</td><td>897</td><td>87%</td><td>0.77</td></tr><tr><td>Claim/knowledge-node decision</td><td>649</td><td>87%</td><td>0.77</td></tr><tr><td>Graph-edge decision</td><td>2,336</td><td>91%</td><td>0.83</td></tr><tr><td>Question accept/revise</td><td>235</td><td>91%</td><td>0.83</td></tr><tr><td>Final-answer accept/revise</td><td>235</td><td>96%</td><td>0.87</td></tr></table>

Table 6 | Independent-reviewer agreement before construction-stage adjudication.

text.

Answer correctness is judged through four binary atoms: whether the response gives a final conclusion, matches the gold final claim, preserves any applicable direction/polarity/comparison, and avoids a materially conflicting conclusion. The task-level Accuracy label is true only when all applicable atoms are labeled “yes”; atoms marked “not applicable” are excluded, and a “no” on any applicable atom makes the response incorrect. Evidence and claim coverage are computed from independent binary node-match judgments. Missing model answers count as incorrect for Accuracy and as unmatched for applicable evidence and claim nodes.

## C.3. Human and Cross-Judge Audit

We conduct a human audit of 384 responses using a balanced stratified sample: four responses are sampled from each of the 8 models × 4 settings × 3 fixed graph-structure dificulty strata. This $8 \times 4 \times 3 \times 4 = 3 8 4$ calculation is the denominator used in the audit tables. Four independent annotators with relevant master’s-level scientific backgrounds participate, and each response is assigned to three annotators through a balanced rotation of the four possible three-person groups. A fifth, senior adjudicator with a doctorate and relevant research experience resolves every disputed atomic label after consulting the source paper and argument graph when necessary. Interannotator agreement is computed from the independent labels before adjudication; the adjudicator produces final human-consensus labels for comparison with Gemini 3.5 Flash.

Table 9 reports agreement at the correctnessatom and graph-node levels. Three-annotator unanimous agreement ranges from 86% to 96% for correctness atoms, is 83% over 1,253 evidencenode decisions, and is 86% over 273 intermediateclaim decisions. Against adjudicated human consensus, Gemini reaches 92% exact agreement on the derived final Accuracy label, 85% on evidencenode matches, and 87% on intermediate-claim matches. Table 8 further shows close aggregate agreement for Accuracy, E-Cov., and C-Cov.

Figure 10 summarizes a multi-judge audit on shared records. Mean exact agreement over the four answer-correctness atoms is 90.8% (atom range 84.0–100.0%; 25–29 comparable groups). Exact agreement is 83.3% for evidence coverage over 120 groups and 87.0% for claim coverage over 23 groups. In the pairwise audit, Gemini 3.5 Flash agrees with the four peer judges on 81.4– 100.0% of shared correctness atoms (� = 24–60) and 76.3–88.2% of shared evidence-node labels (� = 18–69). The claim-node estimates use 3– 12 shared labels, with fewer than 10 for three of the four peers. Accordingly, the cross-judge consistency conclusion is anchored in the more extensively supported correctness and evidencenode comparisons.

<table><tr><td>Dimension</td><td></td><td>Wt. What is evaluated</td><td>Illustrative failure</td></tr><tr><td>Anchor coverage</td><td>12</td><td>Whether the stem depends on the com- A required figure is omitted, or the plete set of selected figures and tables question asks the respondent to use without introducing unselected material. an additional figure that is outside</td><td>the task.</td></tr><tr><td>Anchor localization8</td><td></td><td>Whether the wording lets the respondent The stem says only “the figures&quot; even reliably identify the intended figures, ta- though the supplied materials con- bles, or material range. A collective refer- tain several unrelated groups. ence is acceptable when it is unambigu- ous.</td><td></td></tr><tr><td>erage</td><td></td><td>Evidence-path cov- 14 Whether the stem supplies enough neu-&quot;What do these figures show?&quot;pro- tral scientific context to recover the in- vides no indication of which objects, tended observation → interpretation → conditions, or relationship should or- synthesis path, without spelling out the ganize the evidence.</td><td></td></tr><tr><td>dency</td><td></td><td>hidden reasoning steps. Reasoning depen- 14 Whether answering requires combining One plotted value or one panel di- local observations with at least one in- rectly answers the question, so no terpretive or synthesis step, rather than cross-evidence inference is needed. performing a single lookup or stating only</td><td></td></tr><tr><td>Target specificity</td><td>10</td><td>a final conclusion. Whether the scientific object, condition, &quot;How well does the method work?&quot; readout, comparison, or task context is leaves the dataset, metric, compari- precise enough to define the intended un- son, and operating condition unspec- known without borrowing wording from ified.</td><td></td></tr><tr><td>Stem naturalness</td><td>22</td><td>the answer. Whether the stem is clear, readable, and A long instruction enumerates every phrased like a peer&#x27;s scientific question figure, metric, and reasoning step rather than an exam prompt, procedural before asking for an “overall conclu- scaffold, checklist, or dense stack of tech- sion.&quot;</td><td></td></tr><tr><td>Non-leakage</td><td>15</td><td>nical nouns. Whether the stem withholds result direc- &quot;Why does method A outperform tion, rankings, trends, relation structure, method B?&quot; discloses the compari- intermediate conclusions, and the focal son outcome that the respondent is claim.</td><td>meant to infer.</td></tr><tr><td>Answer shaping</td><td>5</td><td>Whether the stem avoids treating a target “How do these changes prove mecha- explanation or evaluative relation as al- nism X?&quot; presupposes both the mech- ready established and thereby forcing the anism and the evidential relation to response toward a predetermined conclu- it. sion.</td><td></td></tr></table>

Table 7 | Eight-dimensional quality rubric for generated blind multi-hop questions. The weights sum to 100 and produce a normalized overall score on a 0–100 scale. The last column illustrates common failure modes.
<table><tr><td>Metric</td><td>n</td><td>Human</td><td>Gemini</td><td>MAE</td><td>Spearman ρ</td></tr><tr><td>Accuracy</td><td>384</td><td>52%</td><td>52%</td><td></td><td></td></tr><tr><td>E-Cov.</td><td>384</td><td>54%</td><td>54%</td><td>0.08</td><td>0.85</td></tr><tr><td>C-Cov.</td><td>192</td><td>45%</td><td>45%</td><td>0.10</td><td>0.82</td></tr></table>

Table 8 | Metric-level validation against adjudicated human consensus. Scores are means over the audited responses.

<table><tr><td rowspan="2">Decision unit</td><td colspan="3">Independent human annotators</td><td colspan="3">Human consensus vs. Gemini</td></tr><tr><td>n</td><td>Exact</td><td>Fleiss&#x27; κ</td><td>n</td><td>Exact</td><td>Cohen&#x27;s κ</td></tr><tr><td>Gives final conclusion</td><td>384</td><td>96%</td><td>0.92</td><td>384</td><td>96%</td><td>0.81</td></tr><tr><td>Matches gold final claim</td><td>384</td><td>91%</td><td>0.80</td><td>384</td><td>91%</td><td>0.84</td></tr><tr><td>Preserves direction/polarity/comparison</td><td>200</td><td>86%</td><td>0.81</td><td>200</td><td>91%</td><td>0.82</td></tr><tr><td>No materially conflicting conclusion</td><td>384</td><td>91%</td><td>0.82</td><td>384</td><td>92%</td><td>0.81</td></tr><tr><td>Evidence-node match</td><td>1,253</td><td>83%</td><td>0.77</td><td>1,253</td><td>85%</td><td>0.80</td></tr><tr><td>Intermediate-claim-node match</td><td>273</td><td>86%</td><td>0.76</td><td>273</td><td>87%</td><td>0.82</td></tr><tr><td>Derived final Accuracy label</td><td>1</td><td></td><td>1</td><td>384</td><td>92%</td><td>0.80</td></tr></table>

Table 9 | Human inter-annotator agreement before adjudication and Gemini 3.5 Flash agreement with adjudicated human consensus. Human exact agreement requires all three assigned annotators to agree.

## Cross-judge agreement supports retained metrics and Gemini judging Metric consistency and atom-level agreement on shared multi-judge audit records

![](images/5b50336389b152b028fd8034a4662bf56e8021b52baad9c5042fcf990ed02c15.jpg)

![](images/c63966ced75da158782b240aed87f9469ba99c03afedf40de173677ce630b2f9.jpg)  
ACC denotes the final metric; agreement is audited on its four correctness-judgment atoms. Source: repository audit dated 2026-05-22.  
Figure 10 | Cross-judge consistency. (a) Exact agreement for correctness, E-Cov., and C-Cov. (b) Pairwise exact agreement between Gemini 3.5 Flash and four peer judges.

## D. Baseline Reasoning and Evidence Recovery

## D.1. Caption-Only Diagnostic

Caption-only measures how much of each task can be solved from source-paper captions without visual pixels. Table 10 reports the complete modellevel results.

Pooled Caption-only accuracy is 38.7%, below the 46.1% achieved under Direct Visual Reasoning, with E-Cov. and C-Cov. of 26.5% and 26.8%. Accuracy decreases for six models; MiniMax-M3 gains 0.5 points and Intern-S2 gains 9.8 points. Captions therefore provide useful semantic cues for some tasks but do not replace the original visuals overall.

## D.2. Discipline and Dificulty Strata

Table 11 reports the complete Direct Visual Reasoning breakdown by discipline and overall; the main results table gives the complementary fixed graph-annotation strata.

b Accuracy at Low vs. High E-Cov.
<table><tr><td>Model</td><td>Answered</td><td>Acc.</td><td>∆ Acc.</td><td>E-Cov.</td><td>C-Cov.</td></tr><tr><td>GPT-5.5</td><td>235/235</td><td>49.4</td><td>-18.3</td><td>28.7</td><td>31.6</td></tr><tr><td>Claude Opus 4.8</td><td>231/235</td><td>42.1</td><td>-12.8</td><td>27.8</td><td>29.0</td></tr><tr><td>Kimi K2.7</td><td>235/235</td><td>48.5</td><td>-2.1</td><td>31.1</td><td>33.3</td></tr><tr><td>Gemini 3.1 Pro</td><td>235/235</td><td>25.1</td><td>-25.1</td><td>18.6</td><td>16.4</td></tr><tr><td>MiniMax-M3</td><td>235/235</td><td>46.0</td><td>+0.5</td><td>34.5</td><td>38.1</td></tr><tr><td>GLM-5V Turbo</td><td>235/235</td><td>37.0</td><td>-4.3</td><td>27.2</td><td>25.2</td></tr><tr><td>Qwen3.7 Plus</td><td>235/235</td><td>25.1</td><td>-6.4</td><td>20.3</td><td>16.1</td></tr><tr><td>Intern-S2</td><td>235/235</td><td>36.6</td><td>+9.8</td><td>24.2</td><td>24.8</td></tr><tr><td>Pooled</td><td>1876/1880</td><td>38.7</td><td>-7.4</td><td>26.5</td><td>26.8</td></tr></table>

Table 10 | Caption-only diagnostic results. Inputs contain the question and complete source-paper figure/table captions, without image pixels, evidence hints, or paper body text. Accuracy changes are relative to Direct Visual Reasoning.
<table><tr><td>Model</td><td>Bio.</td><td>Chem.</td><td>Comp.</td><td>Phys.</td><td>Overall</td></tr><tr><td>GPT-5.5</td><td>54.0</td><td>76.5</td><td>73.0</td><td>68.2</td><td>67.7</td></tr><tr><td>Claude Opus 4.8</td><td>48.0</td><td>76.5</td><td>61.9</td><td>45.5</td><td>54.9</td></tr><tr><td>Kimi K2.7</td><td>46.0</td><td>55.9</td><td>58.7</td><td>45.5</td><td>50.6</td></tr><tr><td>Gemini 3.1 Pro</td><td>46.0</td><td>61.8</td><td>49.2</td><td>48.9</td><td>50.2</td></tr><tr><td>MiniMax-M3</td><td>46.0</td><td>58.8</td><td>46.0</td><td>39.8</td><td>45.5</td></tr><tr><td>GLM-5V Turbo</td><td>46.0</td><td>50.0</td><td>36.5</td><td>38.6</td><td>41.3</td></tr><tr><td>Qwen3.7 Plus</td><td>40.0</td><td>41.2</td><td>33.3</td><td>21.6</td><td>31.5</td></tr><tr><td>Intern-S2</td><td>30.0</td><td>32.4</td><td>23.8</td><td>25.0</td><td>26.8</td></tr><tr><td>Pooled</td><td>44.5</td><td>56.6</td><td>47.8</td><td>41.6</td><td>46.1</td></tr></table>

Table 11 | Direct Visual Reasoning accuracy (%) by scientific discipline and overall. Dificultystratified headline results are reported in the main results table using fixed graph-annotation strata.

Pooled accuracy is highest for chemistry (56.6%) and lowest for physics (41.6%) in this analysis set. The discipline comparison is descriptive; the main results table reports the corresponding fixed dificulty-stratum comparison.

## D.3. Coverage and Correctness

Figure 11 summarizes the relationship between coverage and correctness under Direct Visual Reasoning. Panel (a) reports pooled accuracy and C-Cov. across E-Cov. bins, together with the range of model-level accuracy. Panel (b) compares each model’s accuracy at low E-Cov. (< 0.3) and high E-Cov. (≥ 0.8).

Table 12 provides a stricter lower-tail check, defining low E-Cov. as below 0.2 while retaining high E-Cov. at 0.8 or above. The localization and error-routing analyses use 0.3 as their declared low-coverage threshold.

The low–high comparison holds within every model. Pooled accuracy is 25.1% among the 486 observations with E-Cov.< 0.2, compared with 62.6% among the 554 observations with E-Cov.≥ 0.8, a 37.5-point diference. This descriptive comparison establishes a consistent association between output evidence coverage and correctness; the following intervention analyses then examine performance under direct evidence access.

![](images/7fc9dff3b739766318021c0ea77ede17ae1e7103168e2ed2ad19b2d2a7d9abe5.jpg)

![](images/7d526ac8f142bd6f096a240730cf7ae10fe797a46814c45cc7cd2dc385916def.jpg)  
Figure 11 | Coverage diagnostics under Direct Visual Reasoning. (a) Pooled accuracy and C-Cov. by E-Cov. bin, with model ranges shaded. (b) Permodel accuracy at low (< 0.3) and high (≥ 0.8) E-Cov.

MiniMax-M3 illustrates why coverage complements accuracy. Under Direct Visual Reasoning it has the highest model-level E-Cov. (61.5%) and C-Cov. (51.7%), together with 45.5% accuracy. Coverage records whether annotated evidence and intermediate claims appear in the output, while accuracy captures whether that information is integrated and mapped to the correct final option. The MiniMax-M3 result therefore identifies a model whose outputs frequently express the intended support chain while achieving moderate final-answer accuracy.

## E. Evidence-Access Interventions

## E.1. Evidence Hint Gains

Table 13 gives model-level overall results for Evidence-Hint Reasoning, together with paired rescue and loss counts; the main results table gives the complementary fixed graph-annotation

<table><tr><td>Model</td><td>Low n</td><td>Low Acc.</td><td>High n</td><td>High Acc.</td><td>Δ</td></tr><tr><td>GPT-5.5</td><td>47</td><td>51.1</td><td>90</td><td>73.3</td><td>+22.3</td></tr><tr><td>Claude Opus 4.8</td><td>46</td><td>28.3</td><td>85</td><td>68.2</td><td>+40.0</td></tr><tr><td>Kimi K2.7</td><td>72</td><td>31.9</td><td>78</td><td>61.5</td><td>+29.6</td></tr><tr><td>Gemini 3.1 Pro</td><td>68</td><td>27.9</td><td>47</td><td>68.1</td><td>+40.1</td></tr><tr><td>MiniMax-M3</td><td>38</td><td>15.8</td><td>99</td><td>60.6</td><td>+44.8</td></tr><tr><td>GLM-5V Turbo</td><td>63</td><td>25.4</td><td>57</td><td>59.6</td><td>+34.3</td></tr><tr><td>Qwen3.7 Plus</td><td>73</td><td>16.4</td><td>51</td><td>52.9</td><td>+36.5</td></tr><tr><td>Intern-S2</td><td>79</td><td>11.4</td><td>47</td><td>46.8</td><td>+35.4</td></tr><tr><td>Pooled</td><td>486</td><td>25.1</td><td>554</td><td>62.6</td><td>+37.5</td></tr></table>

Table 12 | Within-model Direct Visual Reasoning accuracy at low E-Cov. (< 0.2) and high E-Cov. (≥ 0.8); Δ is high minus low in percentage points.

Evidence-Hint Reasoning improves pooled accuracy and C-Cov. across the benchmark, while the main table gives the corresponding fixedstratum view. C-Cov. increases for all eight models, with model-level changes between +15.8 and +33.1 points.

Paired transitions separate gross accuracy changes from individual outcome reversals. Across the 1,880 planned pairs, Evidence-Hint Reasoning rescues 597 initially wrong Direct Visual Reasoning answers and loses 82 initially correct answers, for a net gain of 515 correct observations (+27.4 points). Every model has substantially more rescues than losses. These gains quantify the combined efect of the evidence-explicit intervention, including greater evidence salience and reduced visual transcription and disambiguation burden.

## E.2. Agentic Tool-Use Changes

Model-level changes. Table 14 compares Agentic Tool-Use with Direct Visual Reasoning for accuracy, E-Cov., C-Cov., and paired answer transitions.

Agentic Tool-Use produces a smaller and less uniform intervention than Evidence Hint. It yields 260 rescues and 175 losses, for a net gain of 85 correct observations (+4.5 points). Seven models improve both E-Cov. and C-Cov., but Intern-S2 is the exception: its accuracy rises from 26.8% to 34.0% while E-Cov. falls from 41.8% to 32.6% and C-Cov. falls from 31.2% to 22.1%.

Accuracy and coverage summarize diferent properties of the 235 outputs. Accuracy counts correct final answers, whereas E-Cov. and C-Cov. average the fractions of annotated nodes expressed in each response. Because these metrics aggregate separate response properties, their changes can be distributed diferently across question pairs. Agentic Tool-Use can therefore increase the total number of correct answers while reducing the average amount of annotated support stated in the outputs. For Intern-S2, the supported conclusion is an output-level divergence: higher final-answer accuracy accompanies lower explicit evidence and claim coverage.

## E.3. Localization and Observed Rescue

Table 15 focuses on 299 observations that are initially wrong, have Direct Visual Reasoning E-Cov.< 0.3, and invoke the crop tool.

<table><tr><td>Hits</td><td>n</td><td>E-Cov. improved</td><td>Answer rescued</td></tr><tr><td>0</td><td>105</td><td>51 (48.6%)</td><td>21 (20.0%)</td></tr><tr><td>1</td><td>84</td><td>39 (46.4%)</td><td>22 (26.2%)</td></tr><tr><td>2-3</td><td>79</td><td>44 (55.7%)</td><td>23 (29.1%)</td></tr><tr><td>≥ 4</td><td>31</td><td>20 (64.5%)</td><td>12 (38.7%)</td></tr></table>

Table 15 | Agentic Tool-Use outcomes by annotated-region hit count for initially incorrect, low-coverage observations. A hit requires samereference IoU≥ 0.5.

The answer-rescue rate increases from 20.0% with no annotated-region hit to 38.7% with at least four hits. The shares of runs with increased E-Cov. are 48.6%, 46.4%, 55.7%, and 64.5% across the four hit bins, respectively, with the two highest values in the upper-hit bins. Together, these descriptive results associate greater annotatedregion access with stronger evidence recovery and answer rescue.

A key-region hit requires a crop-tool box and gold region on the same reference to reach IoU≥ 0.5. This definition makes hit count a conservative geometric measure: crops can contain useful legends, axes, captions, neighboring structure, or semantically relevant but loosely aligned regions below the strict threshold. Hit count therefore records geometric access, while E-Cov. records the evidence expressed from that access.

<table><tr><td>Model</td><td>Direct Acc.</td><td>Hint Acc. (∆)</td><td>ΔC-Cov.</td><td>Rescue</td><td>Loss</td></tr><tr><td>GPT-5.5</td><td>67.7</td><td> $8 6 . 8 \ ( + 1 9 . 2 )$ </td><td>+33.1</td><td>53</td><td>8</td></tr><tr><td>Claude Opus 4.8</td><td>54.9</td><td> $7 8 . 3 \ ( + 2 3 . 4 )$ </td><td>+16.5</td><td>60</td><td>5</td></tr><tr><td>Kimi K2.7</td><td>50.6</td><td> $7 6 . 6 \ : ( + 2 6 . 0 )$ </td><td>+26.5</td><td>74</td><td>13</td></tr><tr><td>Gemini 3.1 Pro</td><td>50.2</td><td> $7 0 . 3 \ ( + 2 0 . 0 )$ </td><td>+15.8</td><td>59</td><td>12</td></tr><tr><td>MiniMax-M3</td><td>45.5</td><td> $7 5 . 3 \ : ( + 2 9 . 8 )$ </td><td>+20.5</td><td>84</td><td>14</td></tr><tr><td>GLM-5V Turbo</td><td>41.3</td><td> $7 1 . 5 \ : ( + 3 0 . 2 )$ </td><td>+20.8</td><td>85</td><td>14</td></tr><tr><td>Qwen3.7 Plus</td><td>31.5</td><td> $6 8 . 5 \ : ( + 3 7 . 0 )$ </td><td>+27.3</td><td>92</td><td>5</td></tr><tr><td>Intern-S2</td><td>26.8</td><td>60.4 (+33.6)</td><td>+22.5</td><td>90</td><td>11</td></tr><tr><td>Pooled</td><td>46.1</td><td>73.5 (+27.4)</td><td>+22.9</td><td>597</td><td>82</td></tr></table>

Table 13 | Overall Evidence-Hint Reasoning diagnostics by model; parentheses show accuracy changes from Direct Visual Reasoning. Dificulty-stratified results are reported in the main results table.
<table><tr><td>Model</td><td>Tool-Use Acc. (∆)</td><td>Tool-Use E-Cov. (∆)</td><td>Tool-Use C-Cov. (∆)</td><td>Rescue</td><td>Loss</td></tr><tr><td>GPT-5.5</td><td> $6 9 . 8 \ ( + 2 . 1 ) $ </td><td> $6 3 . 7 \ ( + 5 . 2 )$ </td><td>53.4 (+7.3)</td><td>20</td><td>15</td></tr><tr><td>Claude Opus 4.8</td><td> $6 0 . 0 \ ( + 5 . 1 )$ </td><td> $6 7 . 9 \ ( + 8 . 6 )$ </td><td>54.0 (+2.5)</td><td>30</td><td>18</td></tr><tr><td>Kimi K2.7</td><td> $5 2 . 8 \ ( + 2 . 2 )$ </td><td> $5 5 . 6 \ ( + 4 . 7 )$ </td><td>43.5 (+6.8)</td><td>29</td><td>24</td></tr><tr><td>Gemini 3.1 Pro</td><td> $5 2 . 3 \ ( + 2 . 1 )$ </td><td> $5 6 . 2 \ ( + 1 2 . 3 )$ </td><td>50.5 (+11.1)</td><td>23</td><td>18</td></tr><tr><td>MiniMax-M3</td><td> $5 0 . 6 \ : ( + 5 . 1 )$ </td><td> $6 7 . 5 \ ( + 6 . 0 )$ </td><td>54.1 (+2.4)</td><td>37</td><td>25</td></tr><tr><td>GLM-5V Turbo</td><td> $4 4 . 3 \ ( + 3 . 0 )$ </td><td> $5 7 . 4 \ : ( + 9 . 8 )$ </td><td>44.3 (+10.4)</td><td>37</td><td>30</td></tr><tr><td>Qwen3.7 Plus</td><td> $4 0 . 9 \ ( + 9 . 4 )$ </td><td> $5 4 . 8 \ ( + 9 . 7 )$ </td><td>40.6 (+6.0)</td><td>47</td><td>25</td></tr><tr><td>Intern-S2</td><td> $3 4 . 0 \ ( + 7 . 2 ) $ </td><td> $3 2 . 6 \ ( - 9 . 2 ) $ </td><td>22.1 (-9.0)</td><td>37</td><td>20</td></tr><tr><td>Pooled</td><td> $5 0 . 6 \ ( + 4 . 5 )$ </td><td>57.0 (+5.9)</td><td>45.3 (+4.7)</td><td>260</td><td>175</td></tr></table>

Table 14 | Agentic Tool-Use diagnostics by model; parentheses show changes from Direct Visual Reasoning.

## F. Error Attribution and Qualitative Analysis

## F.1. Routing Rules

The analysis unit is one paired model–question observation among the 1,014 incorrect Direct Visual Reasoning answers. Let $E _ { D }$ and $C _ { D }$ denote its Direct Visual Reasoning E-Cov. and $\mathrm { C } { \mathrm { - } } \mathrm { C o v . }$ , and let � and � denote final-answer correctness under Evidence-Hint Reasoning and Agentic Tool-Use. A coverage value is low below 0.3 and high at or above 0.8.

The Direct Visual Reasoning group � is assigned as follows. It is claim/interpretation insuficient when $E _ { D } \ge 0 . 8$ and $C _ { D }$ is either inapplicable or below 0.3. It is final answer still wrong when $E _ { D } \geq$ 0.8 and applicable $C _ { D } \geq 0 . 8$ . All remaining cases, including observations with intermediate E-Cov. or C-Cov., are assigned to evidence insuficient.

Localization under Agentic Tool-Use is no call, no hit, partial hit, or all hit; a hit requires samereference IoU≥ 0.5, and all hit requires every annotated key region to be hit. For a wrong response with high E-Cov., we use the final-answer-residual label only when applicable C-Cov. is also high; otherwise, including when C-Cov. is inapplicable, we use the interpretation-residual label. The label “Agentic Tool-Use rescue, no qualifying hit” records either the absence of a crop meeting this annotated-region hit criterion or the absence of an observed crop call; output outcomes separately capture useful information available outside strict overlap.

We apply the following ordered rules so that the eight labels are mutually exclusive and exhaustive. First, we inspect the Direct Visual Reasoning response. If its E-Cov. is at least 0.8 while C-Cov. is either inapplicable or below 0.3, we label the case “claim/interpretation insuficient.” If both E-Cov. and applicable C-Cov. are at least 0.8, we label it “final answer still wrong.” All other Direct Visual Reasoning errors are classified as “evidence insufficient.” This preserves the earliest informative trace before using outcomes from later settings.

For the remaining cases, we compare correctness under Evidence-Hint Reasoning and Agentic Tool-Use. When the hint response is wrong but the tool-use response is correct, the case is a “crosssetting Agentic Tool-Use rescue.” When both are wrong, a hint E-Cov. below 0.8 is labeled “Evidence Hint not fully reflected”; otherwise, high evidence coverage is separated into an interpretation residual or a final-answer residual according to claim coverage. When both intervention responses are correct, a tool run with no call or no qualifying region hit is recorded as an “Agentic Tool-Use rescue, no qualifying hit,” while a run with a qualifying hit is assigned to access/localization. Finally, when the hint response is correct but the tool-use response is wrong, incomplete localization is assigned to access/localization; after all annotated regions are hit, Tool-Use E-Cov. below 0.3 is “visual extraction: low,” E-Cov. from 0.3 to below 0.8 is “visual extraction: partial,” and higher E-Cov. is separated into an interpretation or final-answer residual by claim coverage.

## F.2. Model-Level Subtype Composition

Figure 12 shows that access/localization is the largest individual subtype for every model, while the balance between Evidence Hint uptake, downstream interpretation, and audit categories varies. The overall partition contains 407 access/localization errors, 142 Evidence-Hint-notreflected errors, 14 low-extraction errors, 17 partial-extraction errors, 279 interpretation residuals, 43 final-answer residuals, 43 cross-setting Agentic Tool-Use rescues, and 69 Agentic Tool-Use rescues without a qualifying hit. These sum exactly to 1,014.

The main text reports six broader subtypes by combining the two extraction labels and the two audit labels. The resulting counts are 407 access/localization errors (40.1%), 142 Evidence-Hint-not-reflected (goldunderuse) errors (14.0%), 31 visual-extraction errors (3.1%), 279 interpretation residuals (27.5%), 43 final-answer residuals (4.2%), and 112 audit/other cases (11.0%). Thus, the three evidencerelated groups total 57.2% of errors, the two downstream-reasoning groups total 31.8%, and the audit/other group accounts for the remaining 11.0%. The eight-way breakdown below retains the finer distinctions used for sensitivity analysis and representative cases.

## F.3. Interpreting Zero-Count Subtypes

Zero entries in Figure 12 report observed support under the router’s joint criteria. The two visual-extraction labels activate when Evidence-

Hint Reasoning is correct, Agentic Tool-Use is wrong, and all annotated regions are hit, so their branch-specific support is substantially smaller than the 76–172 Direct Visual Reasoning errors contributed by each model. GPT-5.5’s Hint-notreflected observations provide a complementary example: its ten cases with both Evidence-Hint Reasoning and Agentic Tool-Use wrong all have Hint E-Cov. of at least 0.8 and are assigned to downstream residuals. The zero entries therefore summarize empirical subtype prevalence under the declared routing rules and branch eligibility.

## F.4. Threshold Sensitivity

The error-routing sensitivity analysis reruns the fine-grained router at IoU thresholds of 0.3 and 0.7, bracketing the default threshold of 0.5. Table 16 reports aggregate subtype counts, while Table 17 lists the nonzero label transitions between adjacent thresholds. Of the 1,014 routed observations, 54 labels change from IoU 0.3 to 0.5 and 37 change from 0.5 to 0.7. The access/localization subtype remains the largest at every threshold (405, 407, and 400 observations, respectively), and the two downstream subtypes together remain stable (325, 322, and 319). Stricter overlap primarily moves successful Agentic Tool-Use cases into the no-qualifying-hit audit category and moves complete-localization extraction cases back into access/localization.

## F.5. Representative Error-Pattern Cards

Figures 13, 14, 15, and 16 present one audited illustrative case for each subtype. Every card follows the same reading order: cross-setting correctness and coverage/localization values, the source or actual crop views, the model–gold contrast, and the routing implication. The subtype counts and shares quantify prevalence over all 1,014 Direct Visual Reasoning errors, and each selected example provides a concrete instance of its routing rule.

Access and Evidence Hint uptake. Figure 13 contrasts two evidence-acquisition patterns. In Case 1, Agentic Tool-Use inspects 4 of 14 annotated regions and misses the cross-reference identities needed to interpret the central panel. In Case 2, the supplied Evidence Hint is partially reflected (E-Cov. 50.0%): the answer uses one equality while omitting the output-change and guard-routing constraints. The pair separates incomplete visual access from incomplete uptake of already supplied evidence.

![](images/34597e0d8363ff9f310e45fbc05551528feed832cda4f7e5b5bd764f8cf2c385.jpg)

Figure 12 | Per-model composition of the eight mutually exclusive error subtypes at IoU 0.5. Each horizontal row is normalized over that model’s incorrect Direct Visual Reasoning outputs, while the accompanying count gives its error denominator. Colors retain the same subtype mapping across models; subtype assignment follows the ordered routing rules described in the text.
<table><tr><td>Error subtype</td><td>IoU 0.3</td><td>IoU 0.5</td><td>IoU 0.7</td></tr><tr><td>Access / localization</td><td>405</td><td>407</td><td>400</td></tr><tr><td>Evidence Hint not fully reflected</td><td>142</td><td>142</td><td>142</td></tr><tr><td>Visual extraction: low</td><td>26</td><td>14</td><td>7</td></tr><tr><td>Visual extraction: partial</td><td>30</td><td>17</td><td>12</td></tr><tr><td>Interpretation residual</td><td>281</td><td>279</td><td>276</td></tr><tr><td>Final-answer residual</td><td>44</td><td>43</td><td>43</td></tr><tr><td>Cross-setting rescue</td><td>43</td><td>43</td><td>43</td></tr><tr><td>Rescue / no qualifying hit</td><td>43</td><td>69</td><td>91</td></tr><tr><td>Total</td><td>1,014</td><td>1,014</td><td>1,014</td></tr></table>

Table 16 | Error-routing sensitivity analysis: error-subtype counts under alternative same-reference IoU thresholds. Every column contains the same 1,014 initially incorrect Direct Visual Reasoning observations.

Extraction after complete localization. Figure 14 restricts attention to two examples with the same operational localization status: both hit every annotated region, Evidence Hint answers correctly, and Agentic Tool-Use remains wrong. Case 3 covers 14.3% of the required evidence and substitutes a diferent geophysical mechanism, whereas Case 4 reaches 33.3% and extracts part of the scientific role of the two scalar-field scenarios. The contrast operationalizes the low and partial visual-extraction branches and separates geometric access from evidence recovery.

Downstream interpretation and final selection. Figure 15 distinguishes two errors in outputs that explicitly cover all annotated evidence. Case 5 has complete Tool-Use E-Cov. but zero C-Cov.; the answer notices a velocity cue without forming the required joint claim about interactions and curvature. Case 6 already covers both the required evidence and intermediate claims under Direct Visual Reasoning, yet selects the wrong fault plane during final aggregation. The pair therefore separates missing claim expression from an incorrect final-answer selection in the observed outputs.

![](images/deff1b73e33e940609c79d72a2c57ed1bd768d9027137297df12ba8c3c9a99e7.jpg)  
Figure 13 | Audited evidence-acquisition examples. The upper card (Case 1) is routed to access/localization because Agentic Tool-Use reaches 4 of 14 annotated regions and omits two cross-reference checks. The lower card (Case 2) is routed to Evidence Hint not fully reflected because its answer covers part of the supplied constraints (Hint E-Cov. 50.0%). Each card reports the cross-setting outcomes, routing measurements, inspected visual evidence, model–gold contrast, and diagnostic implication.

![](images/0aa1eb1fe9810e2aae609eea763a4a0390d3f1981942f9784b0f39c3d4c1154c.jpg)  
Figure 14 | Audited visual-extraction examples after complete annotated-region access. The upper card (Case 3) has 7/7 region hits and 14.3% Tool-Use E-Cov., yielding the low-extraction label. The lower card (Case 4) has 3/3 hits and 33.3% Tool-Use E-Cov., yielding the partial-extraction label. The model–gold contrasts distinguish complete geometric access from recovery of scientific meaning.

<table><tr><td>Threshold change</td><td>Previous label</td><td>New label</td><td>n</td></tr><tr><td> $0 . 3  0 . 5$ </td><td>Access / localization</td><td>Rescue / no qualifying hit</td><td>26</td></tr><tr><td> $0 . 3  0 . 5$ </td><td>Visual extraction: partial</td><td>Access / localization</td><td>13</td></tr><tr><td> $0 . 3  0 . 5$ </td><td>Visual extraction: low</td><td>Access / localization</td><td>12</td></tr><tr><td> $0 . 3  0 . 5$ </td><td>Interpretation residual</td><td>Access / localization</td><td>2</td></tr><tr><td> $0 . 3  0 . 5$ </td><td>Final-answer residual</td><td>Access / localization</td><td>1</td></tr><tr><td> $0 . 5  0 . 7$ </td><td>Access / localization</td><td>Rescue / no qualifying hit</td><td>22</td></tr><tr><td> $0 . 5  0 . 7$ </td><td>Visual extraction: low</td><td>Access / localization</td><td>7</td></tr><tr><td> $0 . 5  0 . 7$ </td><td>Visual extraction: partial</td><td>Access / localization</td><td>5</td></tr><tr><td> $0 . 5  0 . 7$ </td><td>Interpretation residual</td><td>Access / localization</td><td>3</td></tr><tr><td></td><td></td><td>Changed at 0.3 → 0.5</td><td>54</td></tr><tr><td></td><td></td><td>Changed at  $\mathbf { 0 . 5  0 . 7 }$ </td><td>37</td></tr></table>

Table 17 | Error-routing label transitions between adjacent IoU thresholds; unlisted observations are unchanged.

## Cross-setting and localization-proxy audits.

Figure 16 records two audit categories at the boundaries of the intervention and localization metrics. Case 7 is a cross-setting reversal in which Agentic Tool-Use combines all three required comparison branches and answers correctly after an incorrect Evidence Hint output. Case 8 answers correctly with zero strict IoU@0.5 hits; its crops contain semantically useful obstacle structure, with maximum IoU 0.395 and area recall 0.533. Together, the cases preserve the distinction between Evidence Hint outcomes and oracle behavior, and between strict region overlap and useful visual access.

![](images/c22c7e6222614b814cfbe8b4320b25a69aaebf90e19dd3383b356a8263e8dd8f.jpg)  
Figure 15 | Audited downstream-reasoning examples. The upper card (Case 5) has 100% Tool-Use E-Cov. and 0% Tool-Use C-Cov., placing it in the interpretation-residual category. The lower card (Case 6) has 100% Direct Visual Reasoning E-Cov. and C-Cov. and selects a diferent final fault plane, placing it in the final-answer-residual category.

![](images/9d0ac4afb428d6fbc286dec9837acdc24ea9fa1093f4fd73f0eedb2ffb942824.jpg)  
Figure 16 | Audited intervention and metric-boundary examples. The upper card (Case 7) is a crosssetting reversal: Agentic Tool-Use is correct after covering the three comparison branches following an incorrect Evidence Hint output. The lower card (Case 8) is an Agentic Tool-Use rescue with zero strict IoU@0.5 hits and semantically relevant structure in the crops. These labels characterize observed cross-setting reversals and geometric-proxy boundaries.

## G. Prompt Catalogue

The catalogue reports the task-defining instructions, dynamic input fields, and downstreamconsumed output contracts used by the benchmark. Repeated discipline-specific wording, long pedagogical examples, expanded sample payloads, and serialization-equivalent historical protocol variants are summarized rather than reproduced four times. Black-tabbed boxes are native searchable LaTeX listings; long boxes continue automatically across columns or pages.

## G.1. Benchmark Construction

## Argument Graph and Visual Grounding

## C-G1 Argument-Graph Construction

[SCOPE]   
n=235; shared by all four prompt profiles; source: figure\_qa/   
prompts/\*/graph.py

You are an expert scientific reader. Build an argument graph for ONE paper.

## ## Node roles

g observation \*from this paper\* that is used to support a claim. Make the ‘label‘ self-contained and specific enough to stand on its own. It does \*\*not\*\* need to be limited to one sentence

: use one to a few sentences when needed to preserve the assay , comparison, direction, and key quantitative detail, but keep one node focused on one observation rather than mixing unrelated findings.

\- \*\*claim\*\* labels should also be informative rather than compressed. They do \*\*not\*\* need to be limited to one sentence : use one to a few sentences when needed to state the conclusion precisely, but keep one node focused on one claim rather than bundling several independent conclusions together.

\- \*\*figure\_ref\*\*: one cited paper figure used as a proof anchor. Treat multiple sub-panels from the same overall figure as the same ‘figure\_ref‘ unless the paper clearly cites them as separate proof anchors.

\- \*\*table\_ref\*\*: one cited paper table used as a proof anchor. Treat one overall table as one ‘table\_ref‘ unless the paper clearly separates table sections into distinct proof anchors. - \*\*established\_basis\*\*: non-figure proof the paper relies on as \*\*widely accepted\*\* (textbook fact, standard method, prior consensus, or clearly framed "it is known that ..."). Use a short ‘label‘ quoting or paraphrasing what is taken for granted. Use ‘attributes.citation‘ when the text cites references for that basis.

## ## Article-source attribution

\- Every ‘claim‘ and ‘evidence‘ node must contain a non-empty ‘ source\_attributions‘ array that identifies where its information appears in the supplied paper text.

\- Each source attribution must contain ‘section\_title‘ (the nearest section/subsection heading, or null when unavailable) and ‘source\_excerpt‘ (a short verbatim passage sufficient to verify the node label).

\- Use multiple source-attribution items only when a node genuinely synthesizes non-contiguous passages. Keep each excerpt minimal and exact; do not paraphrase, invent wording,

\- ‘figure\_ref‘, ‘table\_ref‘, and ‘established\_basis‘ nodes use an empty ‘source\_attributions‘ array because their provenance is represented by their citation/label contract.

## ## Mandatory proof for every evidence

\- \*\*Every ‘evidence‘ node MUST have ‘proven\_by‘ anchors, including exactly one outgoing edge to either ‘figure\_ref‘ or ‘table\_ref‘.\*\*

\- An ‘evidence‘ node may also have additional outgoing ‘proven \_by‘ edges to ‘established\_basis‘ when background knowledge is needed to interpret that same ref-grounded observation. - Direction: ‘source‘ = evidence id, ‘target‘ = proof id, ‘ type‘ = ‘proven\_by‘.

\- Claims are linked by ‘supported\_by‘: \*\*source\*\* is always the ‘claim‘ being supported; \*\*target\*\* is the supporting node , usually ‘evidence‘, but \*\*may also be another ‘claim‘ or an ‘established\_basis‘\*\* when the paper chains conclusions or explicitly invokes accepted prior knowledge. \*\*Do not\*\* force everything into a single-hop triple when the narrative is multi-step.

\- ‘figure\_ref‘ and ‘table\_ref‘ nodes are proof anchors only: they may be the target of ‘proven\_by‘, but they must never be the target of ‘supported\_by‘.

## ## Chain arguments (multi-hop is allowed; mildly prefer mixed support when clearly warranted)

\- If the text naturally presents stepwise reasoning (broader conclusion -> sub-result / premise), \*\*slightly prefer\*\* preserving the intermediate ‘claim‘ node(s) and multiple ‘ supported\_by‘ edges instead of collapsing everything into one hop.

\- When a higher-level claim is justified by both (a) direct observations and (b) an intermediate conclusion, represent both supports explicitly (‘claim\_high -> evidence‘ and ‘claim\_ high -> claim\_mid‘).

\- Do not over-fragment: if the paper states a direct support relation without a meaningful intermediate step, keep it single-hop.

\- Typical path shapes include ‘claim\_main ->supported\_by-> claim\_sub ->supported\_by-> evidence ->proven\_by-> figure\_ref/ table\_ref‘, or ‘claim\_high ->supported\_by-> evidence‘ plus ‘ claim\_high ->supported\_by-> claim\_mid‘.

\- A ‘claim‘ that only serves as a step may have \*\*both\*\* outgoing ‘supported\_by‘ (to evidence and/or sub-claims) and incoming ‘supported\_by‘ (from a broader claim).

## ## established\_basis usage

\- If the paper cites prior work as the reason an observation holds, model that observation as ‘evidence‘ and add ‘proven\_by ‘ to both the single ‘figure\_ref‘/‘table\_ref‘ anchor and the needed ‘established\_basis‘ node(s).

\- If the paper directly invokes accepted prior knowledge as part of a higher-level inference, you may link ‘claim -> supported\_by-> established\_basis‘.

\- Do not output floating ‘established\_basis‘ nodes. Every basis must be used by at least one ‘evidence‘ via ‘proven\_by‘ or by at least one ‘claim‘ via ‘supported\_by‘.

\- Experimental setup, assay condition, grouping scheme, or measurement protocol from this paper is \*\*not\*\* ‘established\_ basis‘; do not invent them as separate basis nodes.

## ## First-round ref policy

\- This is the first round. Do not try to bind refs to real image files or final reference ids.

\- Use temporary ids such as ‘graph\_ref\_1‘, ‘graph\_ref\_2‘, graph\_ref\_3‘.

\- Each ‘figure\_ref‘ / ‘table\_ref‘ label must be detailed enough for later semantic matching: state what the figure/

\- Keep ‘attributes.citation‘ when the text names a figure/ table citation such as ‘Fig. 2B‘ or ‘Table 3‘.

\- One ref may support multiple distinct evidence nodes if the paper reads multiple separate observations from the same overall figure/table.

\- Do not split one overall figure into multiple ref nodes just because different panels are mentioned, unless the text clearly treats them as separate proof anchors.

## ## Scale

\- Expect \*\*more nodes\*\* when many figures are discussed; \*\* roughly 20--80 nodes\*\* is normal for figure-heavy papers. Preserve distinct proof anchors, but do not create extra ref nodes for every panel crop.

## ## Output

\- \*\*Exactly one\*\* JSON object, no markdown fences, no commentary.

\- List \*\*every\*\* ‘supported\_by‘ and ‘proven\_by‘ in the toplevel \*\*‘edges‘ array only\*\*. Do \*\*not\*\* nest ‘proven\_by‘ (or any edges) inside ‘nodes‘.

\- Unique ‘id‘ per node (ASCII snake\_case or short alphanumeric ).

\- Use English ‘label‘ when the paper is English; otherwise keep the original language.

\- Include every listed JSON key.

\- ‘attributes‘ must always be an object with key ‘citation‘; ‘ ll‘ h i i i i il bl

Allowed node ‘type‘ values: claim, evidence, figure\_ref, table\_ref, established\_basis

Allowed edge ‘type‘ values: supported\_by, proven\_by

```ini
[SCOPE]
C-G2 n=235; C-G3 conditional n=68
[C-G2 -- BATCH TEXT LINKING]
You batch-link graph-side figure/table references to real
layout candidates for one paper.
Task:
- You will receive all graph-side refs from the extracted
argument graph.
- You will also receive all real figure/table candidates
extracted from the paper layout.
- For each graph ref, choose the single best matching
candidate id, or null when text alone is not sufficient.
```

## Minimal sufficient support (per claim and per evidence)   
- For each \*\*claim\*\* ‘C‘, treat its outgoing ‘supported\_by‘   
edges as a \*\*conjunction\*\* of premises: list only targets the   
text actually uses together to establish ‘C‘. \*\*Avoid\*\*   
parallel edges that duplicate the same reasoning; prefer   
distinct, non-overlapping premises.   
- For each \*\*evidence\*\* ‘E‘, each ‘proven\_by‘ target must   
anchor what the sentence genuinely uses as proof in the paper.   
- Each evidence must have exactly one ref anchor (‘figure\_ref‘   
or ‘table\_ref‘). If the paper combines two refs, split that   
into multiple evidence nodes plus an intermediate claim when   
needed.   
- One ref may be reused by multiple evidence nodes when the   
paper draws multiple distinct observations from the same   
figure or table.   
- Do not attach figure/table/basis nodes that are not   
substantively used for that observation.   
- Evidence nodes should be concise but not underspecified:   
include the measured object, contrast/group, and direction of   
effect when the text gives them. Multi-sentence labels are   
acceptable when needed for fidelity.   
- When quantitative detail is explicit and central (counts,   
percentages, fold-changes, significance, named genes/methods),   
prefer keeping it in the node ‘label‘ and/or ‘attributes‘   
rather than omitting it.   
‘figure\_ref‘ / ‘table\_ref‘ fields for this round: only output   
‘id‘, ‘type‘, ‘label‘, and ‘attributes.citation‘. Use   
temporary ids and keep the label detailed enough for later   
matching.   
‘established\_basis‘ fields: ‘label‘ (required), ‘attributes.   
citation‘ nullable for in-text ref tokens.   
[USER TEMPLATE -- DYNAMIC FIELDS]   
Article id: <ARTICLE\_ID>   
Figure/table candidate list: <REFERENCE\_INVENTORY>   
Main paper text: <ARTICLE\_TEXT>   
[OUTPUT CONTRACT]   
{   
"nodes": [{   
"id" " UNIQUE ID "   
"type": "<claim|evidence|figure\_ref|table\_ref|established\_   
basis>",   
"label": "<SELF\_CONTAINED\_LABEL>",   
"attributes": {"citation": "<CITATION\_OR\_NULL>"},   
"source\_attributions": [{   
"section\_title": "<NEAREST\_HEADING\_OR\_NULL>",   
"source\_excerpt": "<SHORT\_VERBATIM\_PASSAGE>"   
}]   
}],   
"edges": [{"source":"<ID>","target":"<ID>","type":"<   
supported\_by|proven\_by>"}]   
}   
Claim/evidence nodes require source\_attributions. Reference   
and established-basis nodes use an empty source\_attributions   
array.

## C-G2/3 Reference Linking and Conditional Visual Review

```prolog
Rules:
- Match by caption semantics first.
- ‘graph_ref_type‘ and ‘ref_type‘ must agree.
- Use citation strings such as ‘Fig. 2B‘ or ‘Table 3‘ only as
weak hints, never as the primary key.
- Do not match a main-text figure to an ‘Extended Data Figure
‘, or the reverse, unless the citation family explicitly
agrees.
- Prefer candidates whose caption meaning matches both the
graph ref label and its neighboring evidence/claim context.
- When text-only matching is genuinely uncertain, set ‘needs_
visual_disambiguation=true‘ and provide ‘alternative_candidate
_ids‘.
- Do not force a choice when multiple candidates remain
plausible from text alone.
```

```jsonl
Return exactly one JSON object with field:
- ‘matches‘: array of objects containing
- ‘graph_ref_id‘
‘matched_candidate_id‘
- ‘confidence‘
- ‘alternative_candidate_ids‘
- ‘needs_visual_disambiguation‘
Final output contract:
{"matches":[{"graph_ref_id":"<input graph ref id>","matched_
candidate_id":"<best candidate id or null>","confidence":"<
high|medium|low|null>","alternative_candidate_ids":["<
candidate id>"],"needs visual disambiguation":false}l}
[INPUT / OUTPUT]
Input: graph_refs[] plus extracted figure/table candidates[]
and neighboring evidence/claim labels.
Output: {"matches":[{
"graph_ref_id":"<INPUT_ID>",
"matched_candidate_id":"<BEST_ID_OR_NULL>",
"confidence":"<high|medium|low|null>",
"alternative_candidate_ids":["<ID>"],
"needs_visual_disambiguation":false
}]}
[C-G3 -- CONDITIONAL VISUAL REVIEW]
You visually disambiguate one graph-side figure/table
reference against a short candidate list.
Task:
- Read the graph ref label, its citation hint, and its
neighboring evidence/claim labels.
- Compare them against the provided candidate captions.
- Inspect the attached candidate preview images in the same
order as ‘review_candidates‘.
- Return the single best-matching candidate id, or null when
the match remains genuinely uncertain.
Rules:
- Match graph semantics to both caption meaning and visual
layout/content.
- Use the attached images only to break ambiguities that
remained after text matching.
- Treat ‘Figure N‘ and ‘Extended Data Figure N‘ as different
figure families unless the citation family explicitly matches.
- Do not force a choice if the candidates still cannot be
distinguished confidently.
Return exactly one JSON object with:
- ‘graph_ref_id‘
- ‘matched_candidate_id‘
Final output contract:
{"graph_ref_id":"<input graph ref id>","matched_candidate_id
":"<best candidate id or null>"}
[INPUT / OUTPUT]
Input: one unresolved graph_ref, a short candidate list, and
candidate preview images.
Output: {"graph_ref_id":"<INPUT_ID>","matched_candidate_id":"<
BEST_ID_OR_NULL>"}
```

## C-G4 Visual-Evidence Rewrite

[SCOPE]   
shared contract; BIO 50, CHEM 34, CS 63, PHY 88   
[S S O ]   
You rewrite paper-text evidence into figure/table-observable   
evidence for a figure-grounded benchmark.   
You will receive one resolved reference and the evidence nodes   
currently linked to it. Inspect the attached reference image/   
table preview.   
Task for each evidence item:   
1. Confirm the best reference id from the provided reference   
context. In this first implementation there is usually one   
resolved ref; still return ‘selected\_ref\_id‘.   
2. Rewrite the original evidence into ‘visual\_evidence\_text‘:   
a statement that can be directly observed from this figure/   
table using visible panels, axes, legends, labels, annotations   
, table values, or caption-defined setup.   
3. Remove mechanisms, causal explanations, author conclusions,   
clinical/biological interpretations, and paper-text-only   
claims that are not directly visible in the selected reference   
4. If only part of the evidence is visible, keep only the   
directly visible part.   
5. If no directly observable support is found, leave ‘visual\_   
evidence\_text‘ empty or extremely conservative.

Write one bridge question that links local observations to the   
next supported conclusion.

```json
[OUTPUT CONTRACT]
{
"step":"<INTERMEDIATE_CLAIM_STEP>",
"derived_node_id":"<CLAIM_NODE_ID>",
"bridge_question":"<OBSERVATION_TO_INTERPRETATION_QUESTION>"
}
```

You will receive:   
- article background;   
- the canonical program-side data for one evidence step;   
- the image/table directly tied to that evidence step.

Rules:   
- Do not invent numeric values, groups, markers, genes, panels   
, trends, or labels not visible in the image/table or caption  
defined setup.   
- Caption/setup terms may be used to name conditions or   
measurements, but caption conclusions must not be copied as   
observations unless they are visibly supported.   
- Prefer concrete visual language: higher/lower, increase/   
decrease, overlap, enrichment, localization, panel/axis/legend   
/table-value references.   
- Avoid words that overclaim beyond the reference: proves,   
demonstrates mechanism, causes, confirms, therapeutic   
potential, due to, therefore.   
- Return exactly one JSON object with field ‘items‘ and one   
item for every input evidence id.   
[USER TEMPLATE -- DYNAMIC FIELDS]   
reference: id, type, label, citation, caption, and non-result   
context   
evidence\_items[]: evidence\_id, original\_evidence\_text, parent\_   
claim\_labels, candidate\_ref\_ids, and article\_source\_   
attributions   
attachment: <RESOLVED\_FIGURE\_OR\_TABLE\_IMAGE>   
[OUTPUT CONTRACT]   
{"items":[{   
"evidence\_id":"<INPUT\_EVIDENCE\_ID>",   
"selected\_ref\_id":"<REFERENCE\_ID\_OR\_NULL>",   
"visual\_evidence\_text":"<DIRECTLY\_OBSERVABLE\_EVIDENCE\_OR\_   
EMPTY>"   
}]}   
[DISCIPLINE PROFILE ADAPTATIONT   
The task and output contract are shared across all four   
profiles. Only terminology and examples change:   
- Biology: biological objects, compartments, conditions,   
assays, and response patterns.   
- Chemistry: compounds, materials, reaction conditions,   
spectra, morphology, and readouts.   
- Computer Science: methods, datasets, metrics, benchmarks,   
ablations, and failure modes.   
- Physics: physical systems, observables, parameters, regimes,   
spectra, and measurement channels.

## Question and Gold-Answer Construction

## C-Q1 Evidence-Dialogue Planning

Task: use the current image/table, the current evidence statement, and the abstract task family to propose multiple observation questions about directly readable phenomena. These questions should target the evidence encoded by the current statement, but they must ask about observable facts rather than restating the conclusion itself.

Requirements:   
- Focus on what should be observed, compared, or measured from   
the current material; do not restate a higher-level   
conclusion.   
- Prefer quantitative comparisons when the material supports   
them: who is higher, by how much, how much earlier, which row/   
column is larger, whether a curve collapses or rebounds, and   
similar directly readable facts.   
- The question may mention figure/table/panel/location cues   
when useful.   
- If a figure/table is mentioned, use the question-time alias   
labels supplied by the user JSON; figure and table numbering   
restart from 1 inside the current QA task.   
- Keep every sample local, observable, and non-conclusive; do   
not reveal the final claim in advance.   
- Return all requested samples in one JSON object.   
- Every sample must target the same evidence step, but the   
samples should be meaningfully different in angle or emphasis   
rather than trivial paraphrases.   
- Do not ask for the already-compressed conclusion as the   
direct answer.   
- Return one shared ‘key\_region‘ field for the whole evidence   
step, not separate boxes per sample.

Return one JSON object only, with no markdown. Fields:   
- ‘step‘   
- ‘derived\_node\_id‘   
- ‘key\_region‘   
- ‘samples‘, each containing only ‘observation\_question‘   
[USER TEMPLATE -- DYNAMIC FIELDS]   
article background; question-time reference aliases; one   
canonical evidence step; article source attributions;   
requested sample count; attached figure/table   
[OUTPUT CONTRACT]   
{   
"step":"<OBSERVATION\_STEP>",   
"derived\_node\_id":"<EVIDENCE\_NODE\_ID>",   
"key\_region":[x\_min,y\_min,x\_max,y\_max],   
"samples":[{"observation\_question":"<LOCAL\_VISUAL\_QUESTION   
>"}]   
}   
[DISCIPLINE PROFILE ADAPTATION]   
The task and output contract are shared across all four   
profiles. Only terminology and examples change:   
- Biology: biological objects, compartments, conditions,   
assays, and response patterns.   
- Chemistry: compounds, materials, reaction conditions,   
spectra, morphology, and readouts.   
- Computer Science: methods, datasets, metrics, benchmarks,   
ablations, and failure modes.   
- Physics: physical systems, observables, parameters, regimes,   
spectra, and measurement channels.

## C-Q2 Transition-Bridge Planning

stage observed in n=87 benchmark samples

## You will receive:

\- Biology: biological objects, compartments, conditions,

\- Chemistry: compounds, materials, reaction conditions, spectra, morphology, and readouts.

\- Computer Science: methods, datasets, metrics, benchmarks, ablations, and failure modes.

\- Physics: physical systems, observables, parameters, regimes, spectra, and measurement channels.

## C-Q3 Blind-Question Generation

## [SCOPE]

stage observed in n=235 benchmark samples

## [SYSTEM PROMPT]

Write a blind question from slim material hints.

The user JSON will provide article background, selected anchors, final\_claim\_text, neutral observation focuses, optional bridge questions, bridge mode, and a minimal output contract. Use those inputs to write only the question stem. Do not output the answer, explanation, or reasoning chain.

Use final\_claim\_text as the guarded target conclusion: it is provided only to prevent the question from drifting away from the intended final claim. The stem must make the respondent inspect the selected figures/tables, establish local observations, and reach that target through a recoverable observation-to-interpretation path grounded in the supplied observation focuses and bridge questions. Those inputs are latent guidance only: the stem does not need to spell out the bridge structure in advance. Do not turn final\_claim\_text into the stem or into answer-guiding shells such as "what overall conclusion follows", "which explanation is best supported", " which interpretation is most consistent with", or "what, if anything, does ...".

Task: write one difficult blind question. The ideal respondent cannot see the article text and only has access to the relevant figures/tables, so they must inspect the materials, establish local observations, connect them to at least one local inference, and then synthesize the overall answer.

## Question-design requirements:

\- Write one clear biology research question that sounds like a peer’s real scientific question, not a synthesis scaffold or a material-by-material checklist.

\- Keep one main scientific unknown in view. You may name one or two high-level biological objects, response contexts, or readout families when needed, but do not recover specificity by stacking categories.

\- Prefer direct question forms that name the main biological unknown, relationship, or consistency condition to be resolved . If the question stays recoverable without opening on figure/ table labels, prefer object-first, mechanism-first, or consistency-check phrasing over anchor-first openings such as "Across Figures ...", "In Figure ...", or "From Figures ...". Evidence-to-interpretation wording is fine when it stays natural, but do not default to fixed anchor-led openers or generic synthesis shells such as "what do the selected materials suggest, show, or imply".

\- Use discipline-appropriate biological terms only when they clarify the main object, response, compartment, condition, or assay context; do not add terms just to sound technical.

\- Make it clear that the answer depends on the provided materials. You do not need to enumerate every selected figure/ table in the stem. If the question remains unambiguous without

coverage hints. The stem should still imply an observations -> interpretation -> synthesis dependency, but it does not need to explicitly map or preview each bridge step; it is enough that a respondent can naturally recover that path from the materials. In ‘bridge\_mode = bridge\_free‘, it must require a local synthesis without inventing a fake intermediate claim.

\- Default to figure/table-level anchors. Narrow to panel, timepoint, cell population, or assay slice only when the broader anchor would be misleading or genuinely ambiguous. - Keep detailed row/column/value checks out of the stem. The stem may be one sentence or multiple sentences; do not force everything into one sentence when that would make the wording denser or harder to parse. Use as many sentences as needed to keep the question natural and readable without turning it into a checklist.

## Blind-question constraints:

\- Figure/table/panel/location cues and professional terminology are allowed when they help precision, but do not add detail that effectively gives away the target observation.

\- Use explicit nouns such as "this interpretation", "this response pattern", "this relationship", or the relevant biological object when a bare pronoun or phrase such as "that account" would be ambiguous.

\- If a figure/table is mentioned, use the question-time alias labels supplied by the user JSON. Mention specific labels only when they help precision; otherwise a clear collective

\- Do not directly restate or closely paraphrase final\_claim\_ text or any intermediate claim, and do not expose their key relation structure, directional conclusion, winner shape, or supporting results in advance.

\- Do not wrap the stem in answer-guiding or meta shells such as "what overall conclusion follows", "what overall picture emerges", "which explanation is best supported", "which interpretation is most consistent with", or "what, if anything , does ...".

\- Keep the stem at the level of a real scientific uncertainty, not an instruction about how to solve the QA. - Mild non-directional guidance is acceptable, but do not reveal result direction, rankings, trend outcomes, intermediate conclusions, or final claim wording.

## Return one JSON object only, with no markdown:

{"question":"<blind multihop question>"}

Do not output answers, reasoning, hops, or any extra explanation.

## [USER TEMPLATE -- DYNAMIC FIELDS]

article background; selected figure/table aliases; guarded final\_claim\_text; observation focuses; optional bridge questions; bridge\_mode

## [OUTPUT CONTRACT]

{"question":"<BLIND\_MULTIHOP\_QUESTION>"}

Do not output the answer, reasoning steps, involved-reference bookkeeping, or response-requirement bookkeeping.

## [DISCIPLINE PROFILE ADAPTATION]

The task and output contract are shared across all four profiles. Only terminology and examples change:

\- Biology: biological objects, compartments, conditions, assays, and response patterns.

\- Chemistry: compounds, materials, reaction conditions, spectra, morphology, and readouts.

\- Computer Science: methods, datasets, metrics, benchmarks, ablations, and failure modes.

\- Physics: physical systems, observables, parameters, regimes, spectra, and measurement channels.

## C-Q4/5 Question Quality Control and Conditional Rewrite

## [SCOPE]

QC n=234; conditional rewrite n=140

[C-Q5 -- QUESTION QUALITY CONTROL]

Judge whether the blind question is natural, specific, visually grounded, non-leaking, and recoverable from the selected evidence path.

## Score only the dimensions used to decide revision:

\- naturalness and clarity;

\- target specificity;

\- anchor localization and selected-material coverage;

\- evidence-path recoverability;

\- non-leakage and answer-shaping;

\- absence of checklist, procedural, or noun-stacked phrasing.

Return JSON containing the dimension scores, failed dimensions , and concise repair advice. Do not answer the scientific question.

[C-Q4 -- CONDITIONAL REWRITE]

Rewrite a blind question into a clearer synthesis question.

You will receive:

\- the current question stem;

\- focused failure feedback from the rubric and hard checks;

\- the selected anchors whose material scope must remain unchanged;

\- at most one recent failed candidate summary.

Task: make the smallest useful fix that turns the stem into a clearer, more natural synthesis question while preserving the same anchors, answer contract, and multihop reasoning demand. Do not perform unrelated enhancements.

\- Do not add new scientific claims, new anchors, or new readout categories that were not needed to fix the feedback.

\- Avoid repeating the recent failed candidate phrasing, especially procedural templates, vague takeaway wording, and list-like anchor scaffolds.

\- In biology tasks, prioritize these fixes in order when needed:

1. remove information stacking

2. remove answer-direction cueing

3. remove checklist/procedural phrasing

4. make unnecessarily abstract framing more direct

5. replace ambiguous pronoun references with explicit nouns 5. replace ambiguous pronoun references with explicit nouns

6. keep only the minimum biological specificity needed to keep the target clear

\- Keep the question blind: do not reveal the answer, local outcomes, rankings, or trend directions.

\- Keep the selected-material dependency clear, but do not force explicit mention of every selected figure/table. Use specific labels only when they genuinely improve disambiguation or local localization.

\- Do not enumerate every evidence unit, row, column, metric, or difference check in the stem.

\- Do not add detail that effectively gives away the target observation.

\- The rewrite should sound like a peer asking a real biology question after reading the figures, not like a benchmark instruction.

\- Preserve a natural evidence-to-interpretation question when it is already clear and non-leaky; rewrite only when the wording sounds templated, compressed, or unnaturally heavy.

\- Use at most one or two biological handles that truly help specificity; do not recover specificity by listing every readout family again.

\- Prefer a direct key-point question when it makes the stem clearer, but do not collapse it into a vague "What do these figures show?" form.

\- Replace unclear references such as "that account" with explicit wording such as "this interpretation", "this response

\- Preserve the need for at least one local interpretive jump before the final synthesis, but do not spell out the reasoning protocol or add extra bridge scaffolding just to make the path more explicit.

\- If ‘bridge\_mode‘ is ‘bridge\_free‘, preserve a local synthesis step before the overall answer without introducing a fake intermediate-claim wording.

\- Preserve the selected-material scope while changing only the question wording.

\- Do not use vague broad anchor phrases that hide selected references.

\- Do not directly state or closely paraphrase the hidden final /intermediate claim wording, relation structure, or directional conclusion.

\- Do not use exam-essay phrasing such as "How fully is the claim that..." or "to what extent is the claim that...". - Do not use proof-task phrasing or meta-reasoning protocol

\- Do not repair leakage by switching into shells such as "what overall conclusion follows", "which explanation is best supported", "which interpretation is most consistent with", or supported", "which interpretation is most consistent with", or "what, if anything, does ...".

\- Do not fix naturalness by merely shortening the stem into something too vague; shorter is not automatically better.

\- If the current stem compresses too much information into one sentence, it is acceptable to split it into multiple

sentences when that improves readability and still sounds like one natural research question.

Return one JSON object only, with no markdown:

{"question":"<rewritten blind multihop question>"}

[INPUT / OUTPUT]

Input: current question, failed QC dimensions, repair advice,

selected anchors, and the unchanged answer contract.

Output: {"question":"<MINIMALLY\_REWRITTEN\_BLIND\_QUESTION>"}

[DISCIPLINE PROFILE ADAPTATION]

The task and output contract are shared across all four

profiles. Only terminology and examples change:

\- Biology: biological objects, compartments, conditions, assays, and response patterns.

\- Chemistry: compounds, materials, reaction conditions, spectra, morphology, and readouts.

\- Computer Science: methods, datasets, metrics, benchmarks, ablations, and failure modes.

\- Physics: physical systems, observables, parameters, regimes, spectra, and measurement channels.

## C-Q6 Gold-Answer Generation

stage observed in n=235 benchmark samples

You are writing the standard answer for an academic reading assessment.

\- the program-extracted preferred claim;

\- the program-provided canonical trajectory, which is the gold dependency chain and must not be altered in node identity or order.

Task: use the canonical trajectory and execution plan to write one clear, checkable standard answer.

\- The final answer must resolve to the claim identified by ‘ preferred\_claim\_id‘.

\- Attribute figure/table observations to the corresponding task aliases and make the final evidence-to-claim connection explicit.

Return one JSON object only, with no markdown:

\- Rewriting or replacing ‘derived\_node\_id‘ values from the canonical trajectory.

question; selected figure/table context; canonical evidence-to -claim steps; established-basis context; node\_source\_ attributions for evidence and claims

Use article-source excerpts to verify scientific meaning, but write the answer for a respondent who sees the selected visual materials rather than the source article text.

## G.2. Reported Inference Settings

## R1 LLM Reasoning

[SCOPE]

eight models x 235 samples; audited records n=1880

[SYSTEM PROMPT]

You are a careful scientific figure QA model. Answer concisely

[USER TEMPLATE]

Answer the question using the provided figures/tables.

Question: <QUESTION>

[MULTIMODAL CONTENT PARTS]

Figure 1:

<FIGURE\_1\_PIXELS>

Table 1:

<TABLE\_1\_PIXELS>

## R2 Evidence Hint

[SCOPE]

eight models x 235 samples; audited records n=1880

[SYSTEM PROMPT]

You are a careful scientific figure QA model. Answer concisely

[USER TEMPLATE]

Answer the question using the provided figures/tables.

<QUESTION>

## Evidence hints:

Evidence <STEP\_ID> (<SOURCE\_REFERENCE\_LABELS>): <DERIVED\_

EVIDENCE\_STATEMENT>

\- <ADDITIONAL\_EVIDENCE\_HINTS\_AS\_NEEDED>

Use the evidence hints to focus on relevant visual evidence, but verify them against the provided figures/tables before answering.

## [MULTIMODAL CONTENT PARTS]

Figure 1:

<FIGURE\_1\_PIXELS>

Table 1:

<TABLE\_1\_PIXELS>

## R3 Agentic Tool-Use

## [SCOPE]

canonical protocol plus serialization/continuation variants

[CANONICAL FUNCTION-TOOL PROTOCOL]

You are a crop-only scientific figure/table QA agent.

You must answer using the provided figure/table pixels and crop operations only.

On every turn, call exactly one available tool. Every tool call must include ‘reasoning‘ explaining your thinking.

## Image visibility:

\- Images are attached directly in each model turn, alongside the Available images list.

\- Original figure/table images are visible from the first turn

\- crop\_image creates a new listed crop image; that crop is attached directly in later turns.

\- Keep key visual facts in reasoning and, when available, image captions so later steps stay grounded.

## Available actions:

## 1. crop\_image

Crop a specific region of a provided figure/table.

Crop only original figure/table images, not crop outputs such as figure1\_crop1.

The bbox must be normalized [x1, y1, x2, y2] with values in [0, 1].

Coordinates use the source image’s top-left origin: (0, 0) is

the top-left, x increases to the right, and y increases downward.

Do not use crop\_image for references that are not listed in the task.

{"action\_type":"crop\_image","reasoning":"inspect the answerrelevant legend and axis labels more closely","image\_name":" figure1","bbox":[0.3,0.4,0.5,0.6]}

## 2. finish

observations are sufficient to answer the user question, and additional crops are unlikely to change the core conclusion or add an important qualification.

The final answer should explain the key figure/table observations first, then any external context that materially changes or qualifies the interpretation, then reason through intermediate interpretations step by step before giving the synthesized conclusion. When relevant, also include what new knowledge the exploration uncovered bevond the literal

, or future research directions suggested by the figure/table. Do not mention hidden gold answers or internal task fields. Schema:

{"action\_type":"finish","reasoning":"explain why the available visual evidence is sufficient to answer","text":"final answer ..."}

## Decision policy:

\- Images are attached directly in every turn; inspect the

attached figure/table pixels before choosing an action.

\- Use crop\_image when closer visual/table inspection is needed

\- Do not use search, fetch\_page, fetch\_image, or update\_image\_ caption; they are unavailable in this profile.

\- Do not rely on external web evidence, page text, or hidden metadata.

\- Avoid repeated crops of the same area unless the previous crop was unreadable.

\- Finish once the remaining uncertainty would only add minor detail or confidence.

## Output format:

Call exactly one tool matching one of the schemas above.

## [AVAILABLE TOOLS]

crop\_image(image\_name, bbox, reasoning): inspect an answerrelevant region using normalized [x1,y1,x2,y2] coordinates. finish(text, reasoning): submit the final answer when further crops are unlikely to change or materially qualify it.

## [USER TEMPLATE]

Available images: <ORIGINAL\_FIGURE\_AND\_TABLE\_NAMES> Inline pixels: <ORIGINAL\_IMAGES\_AND\_ONE\_TURN\_CROPS>

## [ CO O OCO A A S]

The reported trajectories include the dominant function-tool protocol, a small recovered-shard function variant, and a ReAct-JSON serialization variant. They expose the same croponly evidence-access policy. Conditional continuation messages only enforce remaining-turn and invalid-action constraints; they do not add scientific evidence.

## G.3. Evaluation

## E1 Evidence and Intermediate-Claim Coverage

## [SCOPE]

produces E-Cov. and C-Cov.; source: figure\_qa\_eval/core/ graders.py

## [SYSTEM PROMPT]

You are a scientific QA graph-structure judge. Your task is

NOT to score overall answer quality. You only judge whether the agent’s answer substantively covers the specified evidence nodes and intermediate claims.

## RULES:

1. You are judging structural coverage, not overall quality or correctness.

2. Allow semantic equivalence -- the answer does not need to repeat node labels verbatim. If the answer’s content supports the core scientific meaning of a node, mark it as matched.

3. A node is matched only when the answer substantively

engages with its scientific point. Vague or tangential mentions do not count.

4. Only output node IDs that appear in the input. Never invent new node IDs.

5. Do NOT output coverage scores -- only node-level yes/no judgments.

6. Treat article source attributions only as provenance

context for interpreting the intended node meaning. The answer need not quote an excerpt or name its section to receive a yes judgment.

## COMPACT OUTPUT RULES:

1. Output exactly one JSON object and nothing else.

2. Consider the full input carefully before assigning labels.

3. Do not restate or quote the question, answer, hints,

4. Use only the keys shown in the schema example.

5. Each judgment entry should contain only the schema-required keys.

## OUTPUT FORMAT: You must output exactly one JSON object with this schema:

{"evidence\_node\_judgments": [{"node\_id": "id1", "raw\_label": " yes"}, ...], "intermediate\_claim\_judgments": [{"node\_id": "id3 ", "raw\_label": "yes"}, ...]}

All node IDs must only come from the input. raw\_label must be "yes" or "no". Output raw JSON without markdown code fences.

## [USER TEMPLATE]

"question": "<QUESTION>",

"answer\_text": "<MODEL\_ANSWER>",

"required\_evidence\_nodes": [

"node\_id": "<EVIDENCE\_NODE\_ID>",

"label": "<EVIDENCE\_NODE\_LABEL>",

"citation": "<CITATION\_OR\_NULL>",

"original\_label": "<ORIGINAL\_EVIDENCE\_LABEL>",

"article\_source\_attributions": [

"section\_title": "<SOURCE\_SECTION\_OR\_NULL>",

"source\_excerpt": "<SHORT\_VERBATIM\_EVIDENCE\_PASSAGE   
>"   
}   
]   
}   
],   
"required\_intermediate\_claims": [   
{   
"node\_id": "<INTERMEDIATE\_CLAIM\_NODE\_ID>",   
"label": "<INTERMEDIATE\_CLAIM\_LABEL>",   
"article\_source\_attributions": [   
{   
"section\_title": "<SOURCE\_SECTION\_OR\_NULL>",   
"source\_excerpt": "<SHORT\_VERBATIM\_CLAIM\_PASSAGE>"   
}   
1   
]   
}

## E2 Final-Answer Correctness

```jsonl
[SCOPE]
produces Accuracy; source: figure_qa_eval/core/graders.py
[SYSTEM PROMPT]
You are a scientific QA correctness judge. Your ONLY task is
to judge whether the agent’s final answer is semantically
correct compared to the gold standard answer.
CRITICAL: The agent does NOT need to explicitly state evidence
, intermediate claims, or reasoning steps. As long as the
final conclusion is semantically correct, the answer is
correct.
The gold standard answer is the canonical reference and may be
more specific than a compressed final claim, so compare
against what the standard answer actually says.
Judge these sub-items (binary: yes/no/null):
1. gives_final_conclusion_to_question (binary): Does the
answer explicitly give a final conclusion that addresses the
main question?
2. matches_gold_final_claim (binary, legacy key name): Is the
final conclusion semantically consistent with the gold
standard answer?
3. preserves_direction_polarity_comparison (binary): When the
gold standard answer contains direction/polarity/comparison,
does the answer preserve it? Use null if the gold standard
answer does not involve direction/polarity/comparison.
4. does_not_make_materially_conflicting_conclusion (binary):
Does the answer NOT make any conclusion that materially
conflicts with the gold standard answer?
Scoring: binary: "yes"=1.0, "no"=0.0, "null"=not applicable.
COMPACT OUTPUT RULES:
1. Output exactly one JSON object and nothing else.
2. Consider the full input carefully before assigning labels.
3. Do not restate or quote the question, answer, hints,
evidence, or input.
4. Use only the keys shown in the schema example.
5. Each judgment entry should contain only the schema-required
keys.
OUTPUT FORMAT:
{"item_scores": {"gives_final_conclusion_to_question": {"raw_
label": "yes"}, "matches_gold_final_claim": {"raw_label": "yes
"}, "preserves_direction_polarity_comparison": {"raw_label": "
yes"}, "does_not_make_materially_conflicting_conclusion": {"
raw_label": "yes"}}}
Output raw JSON without markdown code fences.
[USER TEMPLATE]
{
"question": "<QUESTION>",
"gold_standard_answer": "<GOLD_STANDARD_ANSWER>",
"answer_text": "<MODEL_ANSWER>"
}
```