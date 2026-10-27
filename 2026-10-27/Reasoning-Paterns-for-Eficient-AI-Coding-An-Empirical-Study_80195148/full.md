# Reasoning Paterns for Eficient AI Coding: An Empirical Study

ZHE YIN, Shanghai Jiao Tong University, China

YULING SHI, Shanghai Jiao Tong University, China

XIAODONG GU, Shanghai Jiao Tong University, China

BEIJUN SHEN<sup>∗</sup>, Shanghai Jiao Tong University, China

Large Reasoning Models (LRMs) generate long reasoning traces that expose intermediate inference behavior ofering opportunities to study and control reasoning in AI coding. However, the underlying patterns in these traces and their potential for analyzing and guiding coding models remain poorly understood, limiting systematic characterization and inference-time control of LRM reasoning. We conduct an empirical study of reasoning patterns across four representative coding tasks. We transform reasoning traces into structured action sequences and mine recurring fragments using within-task support, cross-task contrasts, and valid– invalid support contrasts. We characterize four general reasoning patterns by coding agents: design-to implementation for code generation, state tracing for execution reasoning, evidence–fix–recheck for debugging, and semantic mapping for translation. We further identify positive and negative patterns associated with valid and invalid reasoning traces, respectively. Positive patterns reflect verification-driven convergence, whereas negative patterns reflect repeated design revision, reflection, and knowledge recall without closure. To demonstrate their practical utility, we evaluate three pattern-based applications: task identification, patternguided prompting, and pattern-aware early stopping. Features capturing general reasoning patterns enable accurate task identification from reasoning traces without accessing task descriptions or final answers. Prompting guided by positive and negative patterns improves the overall accuracy of downstream tasks, while negative pattern-aware early stopping substantially reduces API-reported token usage with only a modest decrease in pass@1. All three applications also generalize to previously unseen models, showing that reasoning patterns can support both the analysis and control of LRM behavior in AI coding.

CCS Concepts: • Software and its engineering → Software development techniques; • Computing methodologies → Artificial intelligence.

Additional Key Words and Phrases: Reasoning Patterns, AI Coding, Reasoning Trace Analysis, Reasoning Pattern Mining, Inference-Time Optimization

## ACM Reference Format:

Zhe Yin, Yuling Shi, Xiaodong Gu, and Beijun Shen. 2026. Reasoning Patterns for Eficient AI Coding: An Empirical Study . 1, 1 (October 2026), 21 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

## 1 Introduction

Large reasoning models (LRMs) are rapidly reshaping AI coding [35, 36]. Unlike conventional code language models that generate an answer directly, LRMs produce a sequence of intermediate

Authors’ Contact Information: Zhe Yin, Shanghai Jiao Tong University, Shanghai, China, yin\_zhe@sjtu.edu.cn; Yuling Shi, ShanghaiJiao Tong University, Shanghai, China, yuling.shi@sjtu.edu.cn; Xiaodong Gu, ShanghaiJiao Tong University, Shang hai, China, xiaodong.gu@sjtu.edu.cn; Beijun Shen, Shanghai Jiao Tong University, Shanghai, China, bjshen@sjtu.edu.cn.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.   
© 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.   
ACM XXXX-XXXX/2026/10-ART   
https://doi.org/10.1145/nnnnnnn.nnnnnnn

reasoning steps before arriving at a solution. This capability is particularly valuable for coding tasks, where solving a problem often involves intermediate decisions, such as decomposing requirements and designing an algorithm for code generation or tracking program states during execution reasoning. The reasoning process is therefore exposed as part of the model’s output rather than remaining implicit.

The availability of these reasoning traces has led to a growing body of research on their role in code-related tasks [6, 29]. Existing studies show that intermediate reasoning can improve task performance, but also reveal substantial ineficiencies. Reasoning traces may contain incorrect assumptions, redundant exploration, repeated revisions, and prolonged deliberation that adds inference cost without improving the final result [4, 12, 18, 39]. These observations have motivated the use of reasoning traces for purposes beyond generating an answer, including self-correction [20, 22], evaluation [26, 30], and inference-time control [9, 10].

However, most existing studies examine reasoning from the perspective of individual tasks or specific behaviors. They focus on final task performance [3, 15, 16], prompting strategies [13, 27, 29], or particular reasoning behaviors in tasks such as code generation [12, 18, 31, 39]. This leaves a more basic question largely unanswered: do reasoning traces contain recurring structures that characterize how LRMs solve coding tasks? If such structures exist, it is also unclear which of them are associated with successful or unsuccessful reasoning, and whether they can be used to improve the eficiency and reliability of inference. Addressing these questions requires looking beyond individual reasoning traces and examining them as sequences of recurring reasoning actions.

To address these questions, we conduct an empirical study of reasoning patterns in AI coding. We collect reasoning traces generated by LRMs across four representative coding tasks: code generation, code execution reasoning, program debugging, and code translation. We transform these traces into structured reasoning action sequences using a unified action taxonomy. This representation allows us to analyze reasoning in terms of observable actions rather than the surface form of generated text. We then examine reasoning patterns at two levels. General patterns capture recurring task-level reasoning structures, whereas positive and negative patterns capture reasoning behaviors at the action level that distinguish valid from invalid trajectories. Finally, we investigate whether the discovered patterns can be leveraged for inference-time analysis and control of LRM behavior.

Specifically, we investigate the following research questions:

## • RQ1 (General Patterns): What task-level reasoning patterns emerge across diferent coding tasks?

Diferent coding tasks impose distinct reasoning demands, yet it remains unclear whether these diferences are reflected in recurring task-level structures. To characterize such structures, we represent reasoning traces as action sequences using a unified action taxonomy and mine recurring action fragments across the four coding tasks. We then examine the resulting patterns to identify the reasoning strategies that characterize each task.

## • RQ2 (Positive and Negative Patterns): What action-level reasoning patterns distinguish valid from invalid reasoning trajectories?

General patterns capture diferences across coding tasks, but they do not reveal the local reasoning behaviors underlying trace validity. We therefore compare judge-labeled valid and invalid reasoning traces and mine recurring action fragments that characterize the two groups. We assess trace validity using an ensemble LLM-as-a-judge protocol and identify positive patterns prevalent in the valid-labeled group and negative patterns prevalent in the invalid-labeled group.

• RQ3 (Analysis and Control): Can the discovered reasoning patterns support diagnostic analysis and inference-time control of LRM behavior?

![](images/db9eeb56f5225653fa26b9c00de3d6627b3fa12c91cece1c4f52081fd9a74401.jpg)  
Fig. 1. Overview of the Empirical Study.

We further examine how the discovered patterns can be used. Specifically, we evaluate a diagnostic task-identification probe and two inference-time controls. First, we use generalpattern features to identify coding tasks from reasoning traces. Second, we augment prompts with guidance derived from positive and negative patterns to steer reasoning behavior. Third, we use negative pattern-aware early stopping to terminate inference when such patterns are detected and quantify the resulting accuracy–token trade-of.

Our study yields three main findings. At the task level, coding tasks exhibit distinct reasoning structures: code generation follows a design-to-implementation flow, code execution reasoning centers on state tracing, debugging follows an evidence–fix–recheck loop, and code translation centers on semantic mapping. At the action level, judge-labeled valid and invalid trajectories exhibit distinct positive and negative patterns. Valid traces show verification-driven convergence, whereas invalid traces exhibit recurring behaviors such as repeated redesign, reflection, and knowledge recall without resolution. Beyond characterization, the discovered patterns support a diagnostic probe and two inference-time controls across the evaluated task–model settings. General-pattern features support diagnostic task identification, pattern-guided prompting yields higher mean pass@1, and negative pattern-aware early stopping reduces API-reported token usage with a corresponding decrease in pass@1. Each evaluation also includes previously unseen models.

In summary, this paper makes the following contributions:

• We establish a unified reasoning-action representation for cross-task analysis of LRM reasoning traces and conduct an empirical study across four representative AI coding tasks.

• We characterize reasoning in AI coding at two levels, identifying general patterns at the task level and positive and negative patterns at the action level from judge-labeled valid and invalid reasoning trajectories, respectively.

• We demonstrate the utility ofthese patterns for AI coding through a diagnostic task-identification probe and two inference-time controls: pattern-guided prompting and negative pattern-aware early stopping.

## 2 Methodology

Figure 1 presents an overview of our empirical study. First, we collect reasoning traces generated by two LRMs across four representative AI coding tasks and assess the validity of each trace. Second, we convert each trace into a sequence of reasoning actions using a unified reasoning-action taxonomy and quantify the resulting behaviors through size and structural metrics. Third, we mine recurring patterns from the resulting action traces, including task-level general patterns for RQ1 and action-level positive and negative patterns for RQ2. Finally, in RQ3, we evaluate how the discovered patterns can be used for AI coding through a diagnostic task-identification probe and two inference-time controls: prompting guided by positive and negative patterns, and early stopping guided by negative patterns.

## 2.1 Study Setup

2.1.1 Tasks and Datasets. To study reasoning patterns in AI coding across diverse problemsolving settings, we consider four representative coding tasks covering program synthesis, comprehension, maintenance, and transformation:

• Code generation requires a model to synthesize executable programs from natural-language specifications. We use LiveCodeBench v4–v6 [15], comprising 442 instances.

• Code execution reasoning requires a model to infer the output of a program through static reasoning rather than actual execution. We use the input–output prediction subset of Code-Sense [21], comprising 382 instances.

• Program debugging requires a model to localize and repair defects in buggy programs. We use DebugBench [25], comprising 500 instances.

• Code translation requires a model to transform programs across programming languages while preserving their semantics. We use ClassEval-T [32], comprising 376 instances.

All four tasks provide objective correctness oracles, which allow us to analyze both reasoning traces and task outcomes under a unified experimental setting.

2.1.2 Models. For RQ1 and RQ2, we use two publicly available LRMs, DeepSeek-R1-0528 (R1-0528) and its Qwen3-8B distilled variant, DeepSeek-R1-0528-Qwen3-8B (R1-Qwen3-8B) [6]. Both models use identical prompts and decoding configurations, and we collect their complete reasoning traces and final responses for analysis. This pair holds the task and collection conditions fixed while exposing a within-lineage scale and distillation contrast; agreement between them is therefore interpreted as within-lineage consistency rather than independent model-family replication. For RQ3, we additionally evaluate two previously unseen models, DeepSeek-V4-Flash (DS-V4-F) [8] and GLM-5.1 [37], to test transfer of fixed pattern-derived guidance. DS-V4-F provides an additional model within the DeepSeek family, whereas GLM-5.1 provides one cross-family evaluation.

## 2.2 Reasoning Trace Processing

2.2.1 Reasoning Trace Collection. For each task instance, we instantiate the benchmarkprovided prompt template with the corresponding instance-specific input, such as a problem description, starter code, buggy program, or source-language snippet. We submit each instantiated prompt independently to both models in thinking mode under a unified decoding configuration (temperature = 0.6, top-� = 0.95).

We extract the content enclosed by the <think> tags as the reasoning trace. To ensure completeness, we regenerate any response with an empty reasoning trace or lacking the information required for annotation or task evaluation, until a complete trace and task output are obtained.

After excluding incomplete, unparsable, or unlabeled responses, the resulting discovery dataset contains 3,376 reasoning traces across the four tasks and two models. Specifically, we retain 442/440 traces for code generation, 377/378 for code execution reasoning, 499/496 for program debugging, and 372/372 for code translation on R1-0528/R1-Qwen3-8B, respectively.

2.2.2 Reasoning Trace Validity Assessment. To distinguish valid from invalid reasoning behaviors in RQ2, we assess the validity of each reasoning trace using an ensemble LLM-as-a-judge protocol [17, 40]. Trace-level validity is distinct from task-level functional correctness: a trace may lead to a correct final answer while containing logical inconsistencies, unsupported transitions, or unnecessary reasoning detours. We consider a reasoning trace valid if its reasoning is internally consistent and progresses coherently toward the final solution.

We employ three judge models, DeepSeek-V3.2 [7], Gemini-3-Flash [11], and GLM-5.1 [37]. We use a unified evaluation prompt that provides the same validity criteria and requires each judge to independently classify a trace as valid or invalid and report a confidence score in the range [0, 1]. To reduce the efect of stochastic judgment variation, we query each judge three times for each trace, yielding nine judgments per trace.

We aggregate the nine judgments using confidence-weighted voting. For a candidate label $\ell ,$ its normalized support is defined as:

$$
S ( \ell ) = \frac { \sum _ { m \in { \cal { M } } } \sum _ { r = 1 } ^ { 3 } c _ { m , r } \cdot 1 \left[ y _ { m , r } = \ell \right] } { \sum _ { m \in { \cal { M } } } \sum _ { r = 1 } ^ { 3 } c _ { m , r } } .\tag{1}
$$

We assign the label with the larger normalized support and use its support as the aggregate confidence. Traces with an aggregate confidence above 0.8 are automatically assigned the corresponding label; the remaining traces are adjudicated independently by two human annotators. The resulting labels define the valid and invalid reasoning-trace groups used in RQ2. The final dataset contains 2,452 valid and 924 invalid traces. R1-0528 contributes 1,439 valid and 251 invalid traces, while R1-Qwen3-8B contributes 1,013 valid and 673 invalid traces.

2.2.3 Reasoning Action Taxonomy and Annotation. Reasoning traces are free-form naturallanguage text, making systematic comparison across tasks and models dificult. To enable structured analysis, we represent each reasoning trace as a sequence of fine-grained reasoning actions, where a reasoning action is defined as a contiguous text span expressing a distinct reasoning behavior.

Drawing on prior studies of LLM reasoning [12, 20, 22, 35], we define a taxonomy of eight reasoning actions that captures the major stages of code reasoning, ranging from task understanding and solution construction to verification and reflection. To strengthen its validity, the taxonomy is further inspected by two researchers with over five years of experience in code intelligence systems, who independently evaluated the coverage, granularity, and ambiguity of the action definitions and iteratively refined them to resolve overlaps. The resulting taxonomy provides a task-independent representation of reasoning behaviors across diferent coding tasks. Table 1 summarizes the eight reasoning actions.

Using this taxonomy, we automatically annotate each reasoning trace with DeepSeek-V3.2 [7] and Gemini-2.5 Pro [24]. Specifically, each reasoning trace is segmented into contiguous spans, and each span is assigned exactly one reasoning action. To ensure consistent annotations, we require the models to generate structured JSON outputs containing the assigned reasoning actions, corresponding text spans, and their ordering. The annotation prompt specifies (1) the definitions of the eight reasoning actions with representative examples, (2) segmentation guidelines covering boundary criteria, granularity, and special-case handling, (3) priority rules for resolving ambiguous cases, and (4) the required JSON output schema. The complete annotation prompt is included in our replication package.

Table 1. Taxonomy of Reasoning Actions
<table><tr><td>Abbr.</td><td>Reasoning Action</td><td>Description</td></tr><tr><td>PU</td><td>Problem understanding</td><td>Understanding and clarifying the problem requirements.</td></tr><tr><td>SD</td><td>Solution design</td><td>Designing the high-level solution approach and strategy.</td></tr><tr><td>IP</td><td>Implementation planning</td><td>Planning the detailed implementation steps and logic.</td></tr><tr><td>CC</td><td>Code construction</td><td>Writing and refining the actual code.</td></tr><tr><td>VV</td><td>Verification and validation</td><td>Testing, debugging, and validating the solution.</td></tr><tr><td>KR</td><td>Knowledge retrieval</td><td>Retrieving and applying relevant knowledge.</td></tr><tr><td>MR</td><td>Meta-reasoning</td><td>Reflecting on and monitoring the reasoning process itself.</td></tr><tr><td>AUX</td><td>Auxiliary</td><td>Auxiliary statements that support the reasoning flow.</td></tr></table>

To assess annotation reliability, two human annotators independently re-annotated a randomly sampled subset of reasoning traces. Their annotations achieved substantial agreement with the LLM-generated annotations (Cohen’s $\kappa = 0 . 8 1 6 )$ [5], indicating that the proposed taxonomy and annotation protocol produce reliable reasoning-action annotations.

2.2.4 Action Trace Construction. The annotated reasoning actions are converted into an action trace, defined as an ordered sequence of reasoning actions:

$$
\tau = ( s _ { 1 } , s _ { 2 } , \ldots , s _ { n } ) ,\tag{2}
$$

where each $s _ { i } \in C = \{ P U , S D , I P , C C , V V , K R , M R , A U X \}$ denotes one reasoning action from the taxonomy. We use the sequence in Eq. (2) as the canonical representation for all subsequent analyses.

To emphasize transitions between reasoning stages while reducing the influence of repeated elaboration within the same action, we further derive a collapsed action trace �¯ by merging adjacent identical actions. For example,

$$
( S D , S D , K R , S D , V V , V V ) \twoheadrightarrow ( S D , K R , S D , V V ) .
$$

2.2.5 Action Trace Measurement. We characterize each reasoning action trace using two groups of metrics: size metrics, which quantify the length and output volume of a reasoning trace, and structural metrics, which capture the distribution and transition dynamics of reasoning actions.

Size metrics quantify the length of the reasoning trace at both the action and token levels. We report the raw action-trace length (n) and collapsed length (m), with their ratio (n/m) measuring the degree of consecutive action repetition. We additionally compute the number of reasoning tokens generated by the model using a consistent tokenizer across all comparisons. These tokenizer-based counts characterize generated reasoning volume and are distinct from API-reported token usage.

Structural metrics characterize the composition of reasoning actions and their transition dynamics. For each action �, its allocation is defined as the fraction of action spans in a trace assigned to �. To characterize reasoning flow, we construct a transition matrix $P ,$ where $P _ { a b }$ denotes the probability of transitioning from action � to action �. We estimate $P _ { a b }$ by counting adjacent $a  b$ transitions in collapsed traces and normalizing by the total number of outgoing transitions from �. We then quantify transition diversity using the stationary-distribution-weighted average of the row-wise transition entropies:

$$
H ( P ) = \sum _ { a } \pi _ { a } \left( - \sum _ { b } P _ { a b } \log _ { 2 } P _ { a b } \right) ,\tag{3}
$$

where � denotes the stationary distribution of �. Higher entropy indicates greater diversity in the actions that can follow a given action, reflecting a less concentrated transition structure.

## 2.3 Reasoning Patern Mining

This section describes how we mine reasoning patterns from the resulting reasoning action traces and characterize them at two granularities: general patterns at the task level and positive and negative patterns at the action level, corresponding to RQ1 and RQ2, respectively.

2.3.1 Reasoning Patern Definition. A reasoning pattern is a recurring action-fragment signature identified from a collection of LRM reasoning action traces. Given a trace set $\mathcal { T }$ , the support of a signature � is the fraction of traces in T whose collapsed action trace contains �:

$$
\operatorname { s u p p } ( s , \mathcal { T } ) = \frac { | t \in \mathcal { T } : s \subseteq t | } { | \mathcal { T } | } .\tag{4}
$$

A signature is considered a candidate pattern when its support exceeds a predefined minimum support threshold. We treat reasoning patterns as observable regularities in action traces rather than as direct evidence of the model’s latent cognitive mechanisms.

We distinguish two types of reasoning patterns according to their analysis granularity. General patterns capture task-level reasoning structures through ordering relations among actions. We denote $A \Rightarrow B$ when action � precedes action � in a trace, allowing arbitrary intermediate actions between them. Thus, this representation captures coarse-grained ordering relations without requiring the actions to be adjacent. Positive and negative patterns, in contrast, capture local reasoning structures through adjacent action transitions. We denote $A  B$ when � and � occur consecutively, and $A  B$ when � and � repeatedly alternate over adjacent positions, forming a short cyclic loop. These representations provide the basis for identifying task-level regularities and distinguishing local patterns prevalent in valid and invalid reasoning trajectories, respectively.

2.3.2 Mining General Paterns. General patterns are mined to identify recurring action structures that distinguish one coding task from the others. For each task, we extract candidate actionfragment signatures from its collapsed reasoning traces. We consider two forms of signatures: contiguous action �-grams with lengths of 4–5 and non-contiguous ordered action paths. For example, $P U \Rightarrow S D \Rightarrow I P \Rightarrow C C$ denotes an ordered pattern in which the four actions occur in the specified order, while allowing arbitrary intermediate actions.

We then assess the task specificity of each candidate signature using its within-task support and its support diference from the other tasks. A signature is retained as a general pattern for a task only if its support within that task exceeds the minimum support threshold and its support advantage over the other tasks exceeds a predefined margin. The resulting general patterns capture recurring task-level reasoning structures and are subsequently used for task identification in RQ3.

2.3.3 Mining Positive and Negative Paterns. Positive and negative patterns are mined to identify recurring action-level structures that distinguish valid reasoning traces from invalid ones. We examine the two trace groups from complementary perspectives: aggregate action allocation and local action transitions. The former characterizes how reasoning actions are distributed across a trace, whereas the latter captures recurring local structures in the action sequence.

The first perspective focuses on action allocation. We define the closure share as the combined proportion of action spans labeled verification and validation (�� ), code construction (��), or auxiliary (�� �). This aggregate measure captures the extent to which a trace allocates reasoning toward convergence-oriented actions. Here, �� � retains its taxonomy definition of auxiliary statements and does not specifically denote answer finalization. Conversely, the drift share is the combined proportion of spans labeled solution design, meta-reasoning, or knowledge retrieval, capturing allocation toward actions that revisit, extend, or redirect the reasoning process. We summarize the relative balance between the two using the closure-minus-drift score, defined as the closure share minus the drift share. Positive values indicate greater allocation toward closureoriented actions, whereas negative values indicate greater allocation toward drift-oriented actions.

The second perspective focuses on local structural patterns. From collapsed action traces, we extract local structural patterns represented by adjacent action transitions (� → �) and short alternating loops $( A  B )$ . Unlike the task-level ordered paths used for general patterns, these structures capture local reasoning dynamics that occur over neighboring actions. These patterns capture recurring action-level structures that are not reflected in aggregate allocation measures. Whereas closure and drift shares characterize the overall allocation of reasoning actions, local patterns identify the specific action transitions and short sequences through which these allocation tendencies manifest.

For local patterns, we identify group-enriched candidates based on both support and support contrast. A candidate pattern must exceed a minimum support threshold in its enriched group and exhibit a minimum support diference relative to the comparison group. Patterns more prevalent in valid traces are labeled positive, whereas those more prevalent in invalid traces are labeled negative. We assess the statistical significance of the resulting group diferences using two-sided Fisher’s exact tests with Benjamini–Hochberg correction [2].

## 2.4 Leveraging Reasoning Paterns

2.4.1 Coding Task Identification. We train a supervised probe on features extracted from reasoning action traces to identify the coding task performed by an LRM. The features operationalize the general patterns identified in RQ1 by encoding how a trace allocates actions, transitions between them, and distributes them across reasoning stages. We construct 57 features in seven groups: (1) eight action-allocation features; (2) two trace-size features; (3) twelve transition-probability features; (4) three repetition features; (5) twenty-four position-aware action-allocation features; (6) four action-efort features; and (7) four aggregate features. The probe uses only reasoning-trace features and has no access to task descriptions, final answers, or external metadata.

2.4.2 Patern-Guided Prompting. We augment the original task prompts with guidance derived from the positive and negative patterns identified in RQ2. Positive patterns are presented as desirable reasoning structures to encourage during inference, whereas negative patterns are formulated as constraints that discourage recurring undesirable transitions or loops, such as ��→��→�� and ��→��→��. The original task instruction is kept unchanged, and the pattern-derived guidance is appended as an additional inference-time signal, allowing us to examine whether empirically identified reasoning patterns can influence downstream coding behavior without modifying the underlying model.

2.4.3 Negative Patern-Aware Early Stopping. We further use negative patterns as online signals to terminate unproductive reasoning before the model completes its original generation. Because full-trace LLM annotation is unsuitable for online intervention, we employ a lightweight rule-based action mapper separate from the LLM-based annotation used for pattern discovery in RQ1 and RQ2. The mapper processes streamed reasoning\_content, segments it into micro-units, and assigns each unit one of eight fine-grained actions and one of four coarse states (Explore, Build, Verify, or Wrap) using lexical cues. Consecutive units are grouped into segments based on coarse-state changes, fine-action transitions, and lexical novelty, and each segment receives its majority action label. Adjacent segments with identical labels are collapsed before matching task-specific motifs. The mapper operates locally without additional model or API calls.

At each completed segment, the stopping policy combines negative motifs with recent exploration, lexical novelty, and build/verify history. A loop trigger requires sustained drift and insuficient closure evidence, subject to task-specific minimum-progress guards, while a separate closure trigger requires strong closure evidence and answer or wrap indicators. A repeated-text detector provides an additional safeguard against degenerate output. When a stopping condition is met, generation is terminated and the original task with the generated reasoning prefix is submitted to the same model in a second request for finalization. The request uses answer-only instructions and disables thinking where supported; any reasoning generated during finalization remains included in the API-reported token usage.

Table 2. General Reasoning Paterns Identified across Four Coding Tasks
<table><tr><td rowspan="2">Task</td><td rowspan="2">General Pattern</td><td rowspan="2">Representative Signature</td><td colspan="2">Support (within-task vs. cross-task)</td></tr><tr><td>R1-0528</td><td>R1-Qwen3-8B</td></tr><tr><td>Generation</td><td>Design-to-implementation</td><td> $P U \Rightarrow S D \Rightarrow I P \Rightarrow V V$ </td><td> $9 1 . 4 \% \ \mathrm { v s . } \ 1 7 . 0 \%$ </td><td>88.0% vs. 22.0%</td></tr><tr><td>Execution Reasoning</td><td>State tracing</td><td> $P U \Rightarrow \{ V V , M R \} ^ { * }$ </td><td> $4 5 . 9 \% \mathrm { v s . } 2 . 2 \%$ </td><td>53.5% vs. 3.4%</td></tr><tr><td>Debugging</td><td>Evidence-fix-recheck</td><td> $\{ S D , I P \} \Rightarrow C C \Rightarrow V V$ </td><td> $6 8 . 1 \% \ \mathbf { v s . 2 1 . 5 \% }$ </td><td> $7 0 . 8 \% \mathbf { v s . 2 0 . 4 \% }$ </td></tr><tr><td>Translation</td><td>Semantic mapping-to-implementation</td><td> $P U \Rightarrow S D \Rightarrow I P \Rightarrow C C$ </td><td> $3 6 . 5 \% \ \mathrm { v s . } \ 9 . 5 \%$ </td><td> $3 0 . 1 \% \ \mathrm { v s . } \ 1 6 . 2 \%$ </td></tr></table>

## 3 RQ1: General Paterns

## 3.1 Setup

RQ1 characterizes task-level reasoning behaviors using the general pattern mining procedure described in Section 2.3.2. We analyze 3,376 collapsed reasoning action traces generated by two LRMs across four coding tasks. A general pattern is retained if its within-task support exceeds 0.05 and its support diference from the other tasks exceeds 0.03. These thresholds were calibrated by two experts through empirical inspection of preliminary results.

## 3.2 Results

We identify one task-specific general reasoning pattern for each coding task. Table 2 reports representative signatures and within- versus cross-task support; the large gaps (e.g., 91.4% vs. 17.0% for code generation and 45.9% vs. 2.2% for execution reasoning on R1-0528) confirm their task specificity.

Code Generation Follows a Design-to-Implementation Pattern. Both models progress from solution design through implementation planning to verification. R1-0528 interleaves design, planning, and verification $( P U \to S D \to V V \to S D \to I P \to V V )$ , with $I P  V V$ dominant; R1-Qwen3-8B follows the same structure while incorporating additional meta-reasoning $( P U $ $S D  M R  S D  V V )$

Code Execution Reasoning Follows a State-Tracing Pattern. Both models simulate program states and verify intermediate results. R1-0528 follows a compact trajectory of state tracing and output verification $( P U \to V V \to M R )$ with the lowest transition entropy across the four tasks (1.690), whereas R1-Qwen3-8B exhibits a longer verification–reflection cycle $( P U  V V  M R $ ��), indicating more iterative transitions within the same structure.

Program Debugging Follows an Evidence-Fix-Recheck Pattern. Both models alternate between fault validation, code modification, and re-verification. R1-0528 enters verification early and repeatedly alternates �� and �� $( P U \to V V \to I P \to V V \to C C \to V V )$ , with frequent $P U \to V V$ and �� → �� transitions; R1-Qwen3-8B follows the same cycle but revisits �� more often $( P U  S D  V V  S D  I P  C C  V V  C C )$

Code Translation Follows a Semantic Mapping Pattern. Both models establish source–target correspondences before implementation. R1-0528 transitions directly from semantic mapping to implementation and validation $( P U  S D  I P  V V  I P )$ , whereas R1-Qwen3-8B revisits �� more often $( P U  S D  I P  S D  I P )$ ; the shared signature $P U \Rightarrow S D \Rightarrow I P \Rightarrow C C$ captures semantic reconstruction followed by code construction.

Table 3. Length and Action Allocation of Valid and Invalid Reasoning Traces
<table><tr><td rowspan="2">Metric</td><td colspan="2">R1-0528</td><td colspan="2">R1-Qwen3-8B</td></tr><tr><td>Valid</td><td>Invalid</td><td>Valid</td><td>Invalid</td></tr><tr><td>Reasoning tokens</td><td>3,690</td><td>6,609</td><td>4,330</td><td>6,991</td></tr><tr><td>Raw action length</td><td>18.58</td><td>24.96</td><td>22.64</td><td>32.21</td></tr><tr><td>Collapsed action length</td><td>11.78</td><td>14.91</td><td>15.97</td><td>21.99</td></tr><tr><td>Closure share</td><td>0.414</td><td>0.241</td><td>0.330</td><td>0.219</td></tr><tr><td>Drift share</td><td>0.247</td><td>0.391</td><td>0.313</td><td>0.464</td></tr><tr><td>Closure-Drift</td><td>+0.167</td><td>-0.150</td><td>+0.017</td><td>-0.245</td></tr></table>

Shared Reasoning Scafold. Despite diferent objectives, all tasks share an understand-then-act scafold. �� appears in more than 95% of traces, after which traces diverge into task-specific sequences involving ��, ��, ��, and ��. �� is prominent in generation, execution reasoning, and debugging, while �� is prominent in generation and translation. The raw-to-collapsed length ratio remains stable (1.3–1.7), indicating comparable consecutive repetition. Thus, task-specific structure emerges primarily after the shared problem-understanding stage.

Cross-Model Comparison. The two R1 models exhibit highly consistent task-level structures but difer in realization: R1-Qwen3-8B generates longer traces than R1-0528 (+37.1% for code generation, +133.1% for execution reasoning, +21.8% for debugging, and +7.6% for translation) and exhibits equal or higher transition entropy, particularly for execution reasoning and debugging. Thus, the models share task-level structures, while R1-Qwen3-8B tends to realize them through longer and more dispersed trajectories.

Finding 1. Each coding task exhibits a distinct general reasoning pattern that reflects its underlying problem-solving strategy. Code generation follows a design-to-implementation pattern, code execution reasoning relies on state tracing, program debugging follows an evidence-fix-recheck cycle, and code translation is driven by semantic mapping followed by implementation.

## 4 RQ2: Positive and Negative Paterns

## 4.1 Setup

RQ2 characterizes action-level reasoning diferences between valid and invalid reasoning traces using the positive and negative pattern mining procedure described in Section 2.3.3. We analyze 2,452 valid and 924 invalid traces generated by the two primary R1 models across four coding tasks. A pattern is retained when its support in the enriched group exceeds 0.08 and its support diference from the comparison group exceeds 0.06. These thresholds were calibrated by two domain experts through empirical inspection of preliminary results.

## 4.2 Results

We compare valid and invalid traces through aggregate action allocation and local transition patterns, revealing fine-grained diferences in action dynamics.

Invalid Traces Shift Action Allocation Toward Drift-oriented Actions. Table 3 summarizes reasoning-token counts, action-trace lengths, and action allocation across the four oracle-backed tasks. Invalid traces are longer than valid traces in reasoning tokens, raw action length, and collapsed action length, indicating greater reasoning volume and more action steps.

Table 4. Positive and Negative Paterns Identified across Four Coding Tasks. Gray-shaded cells denote negative paterns.
<table><tr><td>Task</td><td>Model</td><td>Positive/Negative Pattern</td><td>Representative Signature</td></tr><tr><td rowspan="6">Generation</td><td rowspan="3">R1-0528</td><td>Plan-verify-close</td><td>IP→VV→AUX</td></tr><tr><td>Plan-verify iteration</td><td>IP→VV→IP→VV→IP</td></tr><tr><td>Knowledge-verify-reflect detour</td><td>KR→VV→MR</td></tr><tr><td rowspan="3"></td><td>Design-knowledge detour</td><td>SD→KR→IP</td></tr><tr><td>Verify-plan-reverify</td><td>VV→IP→VV</td></tr><tr><td>Extended verify-plan</td><td>VV→IP→VV→IP</td></tr><tr><td rowspan="3"></td><td rowspan="2"></td><td>Knowledge-design loop Reflection-knowledge loop</td><td>KR↔SD MR↔KR</td></tr><tr><td>State-trace verification</td><td>PU→VV→MR→VV</td></tr><tr><td rowspan="2">R1-0528</td><td>Understand-reopening reflection</td><td>MR→PU→MR</td></tr><tr><td rowspan="2">R1-Qwen3-8B</td><td>State-trace verification</td><td>PU→VV→MR→VV</td></tr><tr><td></td><td>Reflection-design reopening</td><td>MR→SD→{PU,VV}</td></tr><tr><td rowspan="5">Debugging</td><td rowspan="2">R1-0528</td><td>Change-verify-close</td><td>CC→VV→AUX</td></tr><tr><td>Verify-design-reflect reopening</td><td>VV→SD→MR</td></tr><tr><td rowspan="2">R1-Qwen3-8B</td><td>Design-reflect detour</td><td>SD→MR→IP</td></tr><tr><td>Verify-plan-reverify</td><td>VV→IP→VV</td></tr><tr><td rowspan="2"></td><td>Knowledge-design reopening</td><td>SD→KR→SD</td></tr><tr><td>Design-plan-construct</td><td>SD→IP→CC</td></tr><tr><td rowspan="4">Translation</td><td rowspan="2">R1-0528</td><td>Design-reflect reopening</td><td>SD→MR→SD→MR→SD</td></tr><tr><td>Verify-plan-construct detour</td><td>VV→IP→CC</td></tr><tr><td rowspan="2">R1-Qwen3-8B</td><td>Understand-design-plan-reflect</td><td>PU→SD→IP→MR</td></tr><tr><td>Design-reflect replanning Understand-design reopening</td><td>SD→MR→SD→IP</td></tr></table>

The two groups also difer substantially in action allocation. Invalid traces exhibit lower closure shares $( \mathrm { R 1 - 0 5 2 8 : ~ 0 . 4 1 4 } ~  ~ 0 . 2 4 1 ; \mathrm { ~ R 1 - Q w e n 3 - 8 B : ~ 0 . 3 3 0 ~  ~ 0 . 2 1 9 } )$ and higher drift shares (R1- $0 5 2 8 \colon 0 . 2 4 7  0 . 3 9 1 ; \mathrm { R } 1 \mathrm { - } \mathrm { Q w e n } 3 \mathrm { - } 8 \mathrm { B } ; 0 . 3 1 3  0 . 4 6 4 )$ . This shift corresponds to less allocation to verification, code construction, and auxiliary actions and greater allocation to solution design, meta-reasoning, and knowledge retrieval.

The closure-minus-drift score further summarizes this diference. Its value changes from positive to negative between valid and invalid traces for both models (R1-0528: +0.167 → −0.150; R1- Qwen3-8B: +0.017 → −0.245), indicating that invalid traces allocate relatively more reasoning to drift-oriented than closure-oriented actions.

Positive and Negative Patterns Distinguish Convergent Iteration from Repeated Reopening. We next examine whether the diferences in action allocation are reflected in local reasoning structures. Table 4 and Figure 2 summarize the local positive and negative patterns and their support contrasts across task–model settings. Valid and invalid traces exhibit distinct local transition patterns. For example, for R1-Qwen3-8B on code generation, the positive pattern $V V  I P  V V$ occurs in 46.6% of valid traces, compared with 19.6% of invalid traces. In contrast, the negative pattern �� ↔ �� occurs in 56.4% of invalid traces but only 15.9% of valid traces. A similar contrast appears in debugging for R1-0528: �� → �� → ��� occurs in 21.7% of valid traces versus 4.0% of invalid traces, whereas �� → �� → �� occurs in 20.0% of invalid traces versus 2.7% of valid traces.

![](images/82dde4b9c326151525e1604c8f49c1526aafa1f69e98124e62d7ec84bde6e7de.jpg)  
Fig. 2. Support Comparison between Positive and Negative Paterns.

These patterns indicate two qualitatively diferent forms of iteration. Positive patterns return to verification after planning or implementation, such as $V V  I P  V V$ , or proceed from implementation to verification and auxiliary completion, such as $C C \to V V \to A U X$ . Negative patterns, in contrast, more often repeat transitions among design, reflection, and knowledge retrieval, including cycles such as �� ↔ ��. Thus, the distinction between valid and invalid traces is reflected not only in overall action allocation but also in how actions are locally sequenced.

We assess representative support contrasts using two-sided Fisher’s exact tests with Benjamini– Hochberg correction; complete support, count, �-value, and �-value results are provided in the replication package. The association with validity depends on the actions and transition structure, not iteration frequency alone.

Cross-model Consistency with Task-dependent Strength. Finally, we examine the consistency of the observed validity contrasts across models and coding tasks. The distinction between closure-oriented and drift-oriented reasoning is observed in both primary R1 models, suggesting that the identified diferences are not specific to a single model within this model family. The same direction of contrast is also reflected in the local patterns: valid traces more frequently contain verification-centered transitions, whereas invalid traces more frequently contain repeated transitions involving design revision, reflection, and knowledge retrieval.

The magnitude of these diferences varies across settings. For R1-Qwen3-8B, invalid traces exhibit higher drift shares and more frequent cycles involving ��, ��, and ��. This diference is also reflected in trace length. For code generation, invalid traces contain substantially more reasoning tokens than valid traces for both R1-0528 (17,163 vs. 7,102) and R1-Qwen3-8B (14,925 vs. 8,598), with a stronger concentration of drift-oriented actions. Code execution reasoning and debugging exhibit similar but more structured contrasts, whereas the diferences are smaller and less symmetric for code translation.

Finding 2. Valid and invalid reasoning traces difer in trace length, action allocation, and local reasoning patterns. Positive patterns in valid traces predominantly reflect verification-centered iteration, whereas negative patterns in invalid traces more often involve repeated cycles of design revision, reflection, and knowledge retrieval.

![](images/aef1b220443902a423217599370ccd49aeb7bf4a237e9adf41af7940e3df7cd7.jpg)  
Fig. 3. Task Identification Accuracy Using General-Patern Features. Error bars denote the standard deviation across cross-validation folds, and the dashed line indicates the 25% random baseline. Results for DS-V4-F and GLM-5.1 represent cross-model transfer using classifiers trained on R1-0528 traces

## 5 RQ3: Leveraging Reasoning Paterns

To examine how the discovered reasoning patterns can be leveraged at inference time, we evaluate three applications covering task identification, pattern-guided prompting, and pattern-aware early stopping. Specifically, we investigate:

• RQ3.1 Can general reasoning patterns support reliable identification of diferent AI coding tasks? (Section 5.1)

• RQ3.2 How does prompting guided by positive and negative patterns afect coding performance? (Section 5.2)

• RQ3.3 How does negative pattern-aware early stopping afect coding accuracy and token usage? (Section 5.3)

## 5.1 Task Identification Using General-Patern Features

Setup. Following the task-identification probe described in Section 2.4, RQ3.1 examines whether coding tasks can be identified using general-pattern features derived solely from action traces. The evaluation dataset contains 3,376 reasoning traces generated by two models across four coding tasks. We evaluate two standard classifiers, logistic regression and random forest, using 5-fold stratified cross-validation. We report classification accuracy as the mean and standard deviation across folds; for the four-class setting, random guessing yields an accuracy of 25%.

To evaluate cross-model transfer, we train the classifiers on R1-0528 traces and directly apply them to reasoning traces generated by the two previously unseen models, DS-V4-F and GLM-5.1. The target-model traces are used only for testing, with no retraining or model-specific calibration. We extract the same 57 action-trace features and predict the same four task labels.

Results. Figure 3 presents task-identification accuracy using the 57 general-pattern features extracted from reasoning action traces.

1) Direct Results. Logistic regression achieves accuracies of $0 . 8 6 1 \pm 0 . 0 3 2$ for R1-0528 and $0 . 7 7 0 \pm$ 0.019 for R1-Qwen3-8B, while random forest achieves $0 . 8 6 4 \pm 0 . 0 2 7$ and $0 . 7 7 5 \pm 0 . 0 2 2$ , respectively. Both classifiers substantially exceed the 25% random baseline for both models, showing that general-pattern features provide strong signals for distinguishing the four coding tasks. Because the classifiers use only features derived from reasoning action traces, these results show that task identity is reflected in the structural organization of LRM reasoning.

2) Transfer Results. The classifiers trained on R1-0528 also transfer to the two previously unseen models without retraining. On DS-V4-F, logistic regression and random forest achieve accuracies of $0 . 8 3 4 \pm 0 . 0 1 2$ and $0 . 8 2 9 \pm 0 . 0 1 1$ , respectively. On GLM-5.1, they achieve $0 . 8 4 5 \pm 0 . 0 1 4$ and $0 . 8 3 8 \pm 0 . 0 1 3$ . All four results substantially exceed the 25% random baseline, showing that the task-discriminative signals captured by the general-pattern features transfer to target-model traces without model-specific adaptation.

3) Feature Importance Analysis. To further examine which aspects of the reasoning traces contribute to task identification, we analyze feature importance using mutual information. For the two primary R1 models, the most informative features primarily capture the allocation of reasoning efort across core actions, including the design ratio and the distributions of (PU), (SD), (IP), and (VV). These results suggest that action allocation provides a particularly informative signal for distinguishing coding tasks.

Finding 3.1. General-pattern features support accurate coding-task identification, with actionallocation features providing the most informative signals. Classifiers trained on R1-0528 traces also transfer to DS-V4-F and GLM-5.1, achieving 0.829–0.845 accuracy without target-model retraining.

## 5.2 Patern-Guided Prompting

Setup. RQ3.2 evaluates whether positive and negative reasoning patterns can be used as inferencetime guidance for coding tasks. We compare four prompting conditions. Default uses the original task instruction without additional reasoning guidance. Positive-only augments the instruction with guidance derived solely from positive patterns mined from valid traces, whereas Negative-only augments it with constraints derived solely from negative patterns mined from invalid traces. PN guidance combines both positive-pattern guidance and negative-pattern constraints. All four conditions use the same task inputs and prompting settings, difering only in the reasoning guidance provided.

For the direct setting, we apply each primary R1 model’s own pattern inventory to the corresponding model across the four coding tasks with objective correctness oracles. To evaluate cross-model transfer, we apply the pattern inventories mined from the R1 models to two previously unseen models from diferent model families, DeepSeek-V4-Flash (DS-V4-F) [8] and GLM-5.1 [37]. This design allows us to examine both within-model efectiveness and transferability of pattern-guided prompting.

We evaluate coding performance using pass@1 [3]. To reduce variance from stochastic decoding, each condition is evaluated in three independent runs, and we report the mean pass@1 across runs. We additionally report the mean number of generated output tokens per instance, measured in thousands, to characterize the efect of prompting on output length.

Results. Table 5 summarizes the efects of positive- and negative-pattern guidance on coding performance across the four tasks.

1) Direct Results. PN guidance increases mean pass@1 on all four tasks for both R1-0528 and R1-Qwen3-8B. The largest gains are observed on debugging for R1-0528 (+5.80 percentage points) and on code execution reasoning and code translation for R1-Qwen3-8B (+5.50 and +5.05 points, respectively). The changes in performance are accompanied by varying changes in generated-output token usage. For example, on debugging, R1-0528 improves by 5.80 percentage points while mean generated-output tokens increase from 4.5K to 6.6K. On code generation, R1-Qwen3-8B improves by 0.67 points while generated-output tokens decrease slightly from 17.9K to 17.3K. These counts exclude the additional tokens introduced by the guidance prompt.

Table 5. Patern-Guided Prompting Results and Cross-Model Transfer. Direct applies each model’s own patern inventory, whereas Transferred applies patern inventories mined from the R1 models to the previously unseen models DS-V4-F and GLM-5.1. Tok.: mean generated-output tokens per instance (thousands).
<table><tr><td rowspan="3">Task</td><td rowspan="3">Method</td><td colspan="4">Direct</td><td colspan="4">Transferred</td></tr><tr><td colspan="2">R1-0528</td><td colspan="2">R1-Qwen3-8B</td><td colspan="2">DS-V4-F</td><td colspan="2">GLM-5.1</td></tr><tr><td>P@1↑</td><td>Tok.↓</td><td>P@1↑</td><td>Tok.↓</td><td>P@1↑</td><td>Tok.↓</td><td>P@1↑</td><td>Tok.↓</td></tr><tr><td rowspan="4">Generation</td><td>Default</td><td>72.46</td><td>14.9</td><td>56.21</td><td>17.9</td><td>80.78</td><td>13.0</td><td>90.29</td><td>16.5</td></tr><tr><td>Positive-only</td><td>73.14</td><td>15.2</td><td>55.86</td><td>18.3</td><td>84.36</td><td>13.5</td><td>90.06</td><td>13.2</td></tr><tr><td>Negative-only</td><td>72.08</td><td>14.6</td><td>56.39</td><td>17.6</td><td>79.84</td><td>12.5</td><td>90.74</td><td>13.0</td></tr><tr><td>PN guidance</td><td>74.49</td><td>15.6</td><td>56.88</td><td>17.3</td><td>87.80</td><td>12.2</td><td>91.17</td><td>12.9</td></tr><tr><td rowspan="4">Execution Reasoning</td><td>Default</td><td>55.76</td><td>1.7</td><td>44.24</td><td>3.4</td><td>56.81</td><td>2.1</td><td>59.84</td><td>1.6</td></tr><tr><td>Positive-only</td><td>57.16</td><td>2.4</td><td>48.17</td><td>4.4</td><td>58.21</td><td>2.6</td><td>60.32</td><td>1.6</td></tr><tr><td>Negative-only</td><td>56.63</td><td>2.1</td><td>46.86</td><td>3.9</td><td>57.54</td><td>2.3</td><td>60.41</td><td>1.5</td></tr><tr><td>PN guidance</td><td>58.37</td><td>2.9</td><td>49.74</td><td>4.7</td><td>59.27</td><td>2.8</td><td>60.73</td><td>1.5</td></tr><tr><td rowspan="4">Debugging</td><td>Default</td><td>88.80</td><td>4.5</td><td>61.20</td><td>7.7</td><td>91.40</td><td>2.4</td><td>93.00</td><td>2.8</td></tr><tr><td>Positive-only</td><td>91.20</td><td>5.8</td><td>61.54</td><td>7.8</td><td>90.92</td><td>2.5</td><td>93.02</td><td>2.6</td></tr><tr><td>Negative-only</td><td>90.47</td><td>5.1</td><td>61.08</td><td>7.6</td><td>92.18</td><td>2.2</td><td>93.11</td><td>2.6</td></tr><tr><td>PN guidance</td><td>94.60</td><td>6.6</td><td>61.72</td><td>7.9</td><td>92.80</td><td>2.3</td><td>93.20</td><td>2.5</td></tr><tr><td rowspan="4">Translation</td><td>Default</td><td>55.08</td><td>5.7</td><td>39.09</td><td>4.2</td><td>55.05</td><td>2.1</td><td>56.97</td><td>2.5</td></tr><tr><td>Positive-only</td><td>56.81</td><td>6.5</td><td>42.51</td><td>4.4</td><td>54.73</td><td>2.2</td><td>58.03</td><td>2.4</td></tr><tr><td>Negative-only</td><td>57.44</td><td>6.2</td><td>43.27</td><td>4.3</td><td>56.73</td><td>1.8</td><td>58.52</td><td>2.4</td></tr><tr><td>PN guidance</td><td>58.78</td><td>7.3</td><td>44.14</td><td>4.5</td><td>57.44</td><td>1.9</td><td>58.75</td><td>2.3</td></tr></table>

2) Ablation of Positive and Negative Guidance. We next compare the four prompting conditions. Across the sixteen task–model settings, neither Positive-only nor Negative-only consistently improves over Default when used alone. Positive-only reduces pass@1 for R1-Qwen3-8B on code generation (55.86 vs. 56.21) and for DS-V4-F on code translation (54.73 vs. 55.05). Negative-only likewise reduces pass@1 for DS-V4-F on code generation (79.84 vs. 80.78) and for R1-Qwen3-8B on debugging (61.08 vs. 61.20). In contrast, PN guidance achieves the highest pass@1 among the four conditions in every task–model setting and exceeds Default in all sixteen settings.

3) Transfer Results. We further evaluate whether pattern inventories mined from the R1 models transfer to previously unseen models. On DS-V4-F, PN guidance improves pass@1 on all four tasks, with the largest gain on code generation (+7.02 points). Generated-output token usage increases for code execution reasoning but decreases for the other three tasks. On GLM-5.1, pass@1 increases on all four tasks by 0.20–1.78 percentage points, while generated-output token usage decreases across all four tasks. For example, on code generation, pass@1 increases from 90.29 to 91.17 (+0.88 points), while mean generated-output tokens decrease from 16.5K to 12.9K.

Finding 3.2. PN guidance improves mean pass@1 across all evaluated task–model settings, including two previously unseen models. It also achieves higher pass@1 than the corresponding Positive-only and Negative-only variants in all sixteen settings.

## 5.3 Negative Patern-Aware Early Stopping

Setup. RQ3.3 evaluates whether negative-pattern signals can support online early stopping to reduce token usage while preserving coding accuracy. We compare two settings: Full Reasoning, which allows the model to complete its reasoning trace before producing the final answer, and

Table 6. Negative Patern-Aware Early Stopping Results across Models. Tok.: API-reported token usage (thousands); stopped runs include finalizer output.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Model</td><td colspan="2">Full Reasoning</td><td colspan="4">Pattern-Aware Stop</td></tr><tr><td>P@1↑</td><td>Tok.↓</td><td>P@1↑</td><td>Tok.↓</td><td>∆P@1 (pp)</td><td>∆Tok. (%)</td></tr><tr><td rowspan="3">Generation</td><td>R1-Qwen3-8B</td><td>56.21</td><td>17.89K</td><td>52.14</td><td>10.64K</td><td>-4.07</td><td>-40.56%</td></tr><tr><td>DS-V4-F</td><td>80.78</td><td>13.00K</td><td>73.36</td><td>8.43K</td><td>-7.42</td><td>-35.13%</td></tr><tr><td>GLM-5.1</td><td>90.29</td><td>16.54K</td><td>88.06</td><td>7.85K</td><td>-2.23</td><td>-52.52%</td></tr><tr><td rowspan="3">Execution Reasoning</td><td>R1-Qwen3-8B</td><td>44.24</td><td>3.43K</td><td>40.58</td><td>2.10K</td><td>-3.66</td><td>-38.75%</td></tr><tr><td>DS-V4-F</td><td>56.81</td><td>2.13K</td><td>51.57</td><td>0.85K</td><td>-5.24</td><td>-59.78%</td></tr><tr><td>GLM-5.1</td><td>59.84</td><td>1.64K</td><td>58.63</td><td>0.52K</td><td>-1.21</td><td>-68.53%</td></tr><tr><td rowspan="3">Debugging</td><td>R1-Qwen3-8B</td><td>61.20</td><td>7.65K</td><td>54.40</td><td>4.18K</td><td>-6.80</td><td>-45.32%</td></tr><tr><td>DS-V4-F</td><td>91.40</td><td>2.42K</td><td>86.60</td><td>1.69K</td><td>-4.80</td><td>-30.16%</td></tr><tr><td>GLM-5.1</td><td>93.00</td><td>2.77K</td><td>91.20</td><td>1.74K</td><td>-1.80</td><td>-37.16%</td></tr><tr><td rowspan="3">Translation</td><td>R1-Qwen3-8B</td><td>39.09</td><td>4.18K</td><td>37.64</td><td>2.33K</td><td>-1.45</td><td>-44.22%</td></tr><tr><td>DS-V4-F</td><td>55.05</td><td>2.14K</td><td>50.79</td><td>1.23K</td><td>-4.26</td><td>-42.65%</td></tr><tr><td>GLM-5.1</td><td>56.97</td><td>2.53K</td><td>55.48</td><td>1.51K</td><td>-1.49</td><td>-40.56%</td></tr></table>

Pattern-Aware Stop, which monitors the streaming reasoning trace and triggers answer finalization when a stopping condition is met.

Because Pattern-Aware Stop requires streaming reasoning outputs, we evaluate it on R1-Qwen3- 8B, DS-V4-F, and GLM-5.1, excluding R1-0528 because its interface does not provide streaming reasoning outputs. We directly apply the negative patterns mined from R1-Qwen3-8B to the same model and transfer them to DS-V4-F and GLM-5.1 without model-specific adaptation.

We run each setting three times and report mean pass@1 and mean API-reported token usage per instance. For stopped runs, token usage includes the finalizer request, which resubmits the original task with the generated reasoning prefix. The online monitor is rule-based and requires no additional model or annotation-model calls.

To quantify monitoring overhead, we replay 40 stored R1-Qwen3-8B traces comprising 9,019 chunks of 25 characters each. The monitor processes each chunk in 0.13 ms at the median and 0.30 ms at the 95th percentile, with cumulative CPU time of 19 ms and 142 ms per trace, respectively. This microbenchmark excludes model inference, network, and finalizer latency.

Results. Table 6 summarizes the efects of negative pattern-aware early stopping across the four coding tasks.

1) Direct Results. For R1-Qwen3-8B, API-reported token usage decreases by 38.75%–45.32%, accompanied by a 1.45–6.80 percentage-point decrease in pass@1. Code translation shows the smallest accuracy decrease (1.45 points) with a 44.22% reduction in token usage, whereas code generation and debugging show larger accuracy decreases.

2) Transfer Results. Applying the negative patterns mined from R1-Qwen3-8B to the two unseen models also reduces API-reported token usage. The reductions range from 30.16% to 59.78% for DS-V4-F and from 37.16% to 68.53% for GLM-5.1, with corresponding pass@1 decreases of 4.26–7.42 and 1.21–2.23 percentage points, respectively. Thus, the same negative-pattern signals reduce token usage across all three models, although the associated accuracy changes vary by task and model.

Finding 3.3. Negative pattern-aware early stopping reduces API-reported token usage by 30–69% across the three evaluated models, with a corresponding 1.21–7.42 percentage-point decrease in pass@1 across task–model settings.

## 6 Discussion

## 6.1 Understanding Reasoning Structures

The patterns suggest that reasoning structures arise from how a task constrains uncertainty and when the model can commit to an intermediate result. All four tasks begin with problem understanding because the model must first establish what is being transformed, executed, repaired, or produced. They then diverge because their uncertainty is resolved diferently: generation and translation require constructing a representation before it can be checked, whereas execution and debugging require maintaining an externally constrained state and repeatedly testing that state. The resulting task-specific structures therefore reflect diferent control demands, rather than superficial diferences in trace length or vocabulary.

The validity contrasts reveal what happens when this control process loses its anchor. Verification is not merely another action; it provides a checkpoint that can confirm or reject the current interpretation and thereby allow the trace to move forward. Design, reflection, or knowledge retrieval can be useful when they reduce a concrete uncertainty, but repeated transitions among them without renewed verification reopen decisions that have not been resolved. This explains why longer reasoning is not a reliable sign of better reasoning: additional steps help only when they convert uncertainty into a checked commitment. Positive and negative patterns consequently distinguish two forms of iteration, one that closes a loop through evidence and one that perpetuates exploration without closure.

This interpretation also explains the cross-model results. Models can realize the same task-level strategy with diferent amounts of repetition and dispersion because the underlying task constraints remain stable while the eficiency of the control process varies. The structural representation separates these two factors: general patterns describe the task-conditioned workflow, while local transitions expose whether the workflow is being stabilized by verification or repeatedly reopened. Reasoning structure is therefore a process-level signal of how a model manages uncertainty and commits to intermediate states, which is why it can support task identification and inference-time intervention without treating every additional reasoning token as useful progress.

## 6.2 Future Research Directions

Evaluation of Reasoning Models. Reasoning traces provide behavioral signals not captured by outcome-based metrics such as pass@1 or trace length. Future evaluation frameworks could incorporate action allocation and transition structure to provide a finer-grained characterization of reasoning behavior.

Structure-Aware Inference Control. The discovered patterns can serve as structural signals for inference-time monitoring and intervention. Beyond early stopping, such signals could support adaptive reasoning budgets, targeted verification, or re-planning when recurring drift patterns are detected.

Learning with Reasoning Structure. Beyond inference-time guidance, positive and negative patterns could provide process-level supervision for training. Training objectives could reward desirable action transitions and discourage recurring negative patterns, moving from externally guided reasoning toward models that learn structured reasoning behavior.

## 7 Threats to Validity

Construct Validity. We abstract free-form reasoning traces into discrete reasoning actions to enable systematic comparison, which may omit fine-grained semantic distinctions. We mitigate this risk by grounding the taxonomy in prior work, refining it through expert inspection, and validating automatic annotations against human re-labeling (Cohen’s $\kappa = 0 . 8 1 6 )$ . Nevertheless, alternative taxonomies, segmentation granularities, or collapsing rules may yield diferent patterns.

Internal Validity. Trace-validity labels are derived from an ensemble ofLLM judges, with human adjudication for low-confidence cases. Confidence-weighted voting reduces judgment variance, but subtle reasoning errors may remain undetected. Task-level functional correctness is evaluated independently using objective oracles and reported as pass@1. The prompting and early-stopping results therefore provide intervention evidence but do not establish causal efects attributable to individual mined patterns.

For early stopping, the online rule-based mapper is distinct from the ofline LLM-based annotators used for pattern discovery. Their agreement on partial reasoning prefixes has not been independently evaluated, so mapping errors may afect motif detection and stopping decisions. The stopping policy also includes a repeated-text trigger that operates independently of action motifs.

External Validity. We study four coding tasks and four model configurations, which may not capture the diversity of software engineering tasks or model architectures. R1-0528 and its Qwen3-8B distillation share a model lineage, while DS-V4-F is from the same model family; GLM-5.1 provides one cross-family evaluation. Moreover, GLM-5.1 also participates in the validity-judge ensemble for the primary traces, so its intervention results do not constitute a fully independent replication of the discovery-to-intervention pipeline. Broader evaluation across tasks, programming languages, and model families is needed to assess the generality of the findings.

## 8 Related Work

Recent LRMs incorporate extended reasoning traces as a core component of inference [6]. In code-related settings, prior work shows that reasoning-before-coding is not uniformly beneficial and depends on how reasoning is structured relative to code generation [18]. Other studies explore leveraging external or structured signals to improve reasoning, such as mining discussion data into training chains [34] or validating intermediate steps via search-based refinement [28]. These eforts demonstrate the utility of explicit reasoning for code generation, but primarily focus on improving outputs or training signals rather than characterizing recurring reasoning structures across tasks.

Prior work has examined reasoning quality and failure modes. Studies find that reasoning quality does not always correspond to final correctness: flawed reasoning can still lead to correct outputs, while correct outputs can also arise from flawed reasoning [39]. Similarly, surface-level properties such as reasoning length or step count provide weak evidence of correctness [31]. Recent studies further categorize reasoning behaviors and failure patterns, including hallucination, overthinking, and erroneous backtracking [1, 12, 23]. While these studies move beyond outcome-based evaluation, they often focus on individual tasks or rely on coarse-grained or manually defined categories. How reasoning structures recur across coding tasks, and how such structures distinguish valid from invalid reasoning traces, therefore remains less explored.

A separate line of work uses reasoning traces for inference-time control. SEER adapts reasoning strategies during generation [10], while studies of overthinking motivate methods for removing redundant reasoning steps [4, 14, 38]. Other early-stopping approaches use intermediate signals, such as answer convergence or confidence, to terminate generation earlier [19, 33]. These approaches primarily target decoding eficiency or reasoning control, but do not derive the control signals from recurring reasoning structures identified through empirical analysis of traces.

In contrast, we treat reasoning traces as behavioral data and abstract them into reasoning action traces to study recurring reasoning structures across coding tasks. This allows us to examine reasoning patterns at both the task and action levels, with particular attention to their recurrence across tasks and their association with trace validity.

## 9 Conclusion

In this paper, we present an empirical study of reasoning patterns in AI coding. By abstracting reasoning traces into action sequences, we identify general patterns that capture task-specific reasoning strategies and positive and negative patterns that characterize valid and invalid reasoning behaviors at the action level across four coding tasks. We further demonstrate the utility of these patterns for inference-time analysis and control through task identification, pattern-guided prompting, and negative pattern-aware early stopping. These results show that reasoning patterns can provide a structured basis for analyzing LRM reasoning and controlling inference-time behavior in AI coding.

## Data Availability

All code and data used in this study are publicly available at: https://anonymous.4open.science/r/ FSE\_2027. The repository includes: (i) the collected reasoning traces and annotated datasets across the four tasks (code generation, code execution reasoning, debugging, and translation); (ii) the code for pattern discovery, pattern-guided prompting and pattern-aware early stopping (RQ1–RQ3); and (iii) all experimental results and evaluation scripts.

## References

[1] Mohammad Abdollahi, Khandaker Rifah Tasnia, Soumit Kanti Saha, Jinqiu Yang, Song Wang, and Hadi Hemmati. 2025. Demystifying errors in LLM reasoning traces: An empirical study of code execution simulation. ACM Transactions on Software Engineering and Methodology (2025).

[2] Yoav Benjamini and Yosef Hochberg. 1995. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal statistical society: series B (Methodological) 57, 1 (1995), 289–300.

[3] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374 (2021).

[4] Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. [n. d.]. Do NOT Think That Much for 2+3=? On the Overthinking of Long Reasoning Models. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025 (Proceedings ofMachine Learning Research). PMLR / OpenReview.net.

[5] Jacob Cohen. 1960. A coeficient of agreement for nominal scales. Educational and psychological measurement 20, 1 (1960), 37–46.

[6] DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. CoRR abs/2501.12948 (2025).

[7] DeepSeek-AI. 2025. DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models. CoRR abs/2512.02556 (2025).

[8] DeepSeek-AI. 2026. DeepSeek-V4: Towards Highly Eficient Million-Token Context Intelligence. Technical Report. DeepSeek-AI. https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek\_V4.pdf

[9] Pengfei Gao and Chao Peng. [n. d.]. More with Less: An Empirical Study of Turn-Control Strategies for Eficient Coding Agents. In 2026 IEEE/ACM 48th International Conference on Software Engineering (ICSE ’26), April 12-18, 2026 Rio de Janeiro, Brazil.

[10] Shuzheng Gao, Chaozheng Wang, Cuiyun Gao, and Michael R. Lyu. 2025. SEER: Enhancing Chain-of-Thought Code Generation through Self-Exploring Deep Reasoning. CoRR abs/2510.17130 (2025).

[11] Google DeepMind. 2025. Gemini 3 Flash Model Card. https://deepmind.google/models/model-cards/gemini-3-flash/. Accessed: 2026-05-28.

[12] Kevin Halim, Sin G. Teo, Ruitao Feng, Zhenpeng Chen, Yang Gu, Chong Wang, and Yang Liu. 2025. A Study on Thinking Patterns of Large Reasoning Models in Code Generation. CoRR abs/2509.13758 (2025).

[13] Dong Huang, Qingwen Bu, Yuhao Qing, and Heming Cui. 2023. Codecot: Tackling code syntax errors in CoT reasoning for code generation. CoRR abs/2308.08784 (2023).

[14] Kerui Huang, Shuhan Liu, Xing Hu, Tongtong Xu, Lingfeng Bao, and Xin Xia. 2025. Reasoning Eficiently Through Adaptive Chain-of-Thought Compression: A Self-Optimizing Framework. CoRR abs/2509.14093 (2025).

[15] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. [n. d.]. LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for

Code. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025.

[16] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R. Narasimhan. [n. d.]. SWE-bench: Can Language Models Resolve Real-world Github Issues?. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

[17] Qintong Li, Leyang Cui, Lingpeng Kong, and Wei Bi. 2025. Exploring the Reliability of Large Language Models as Customized Evaluators for Diverse NLP Tasks. In Proceedings ofthe 31st International Conference on Computational Linguistics. Association for Computational Linguistics, Abu Dhabi, UAE, 10325–10344.

[18] Renbiao Liu, Anqi Li, Chaoding Yang, Hui Sun, and Ming Li. [n. d.]. Revisiting Chain-of-Thought in Code Generation: Do Language Models Need to Learn Reasoning before Coding?. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025. PMLR / OpenReview.net.

[19] Xin Liu and Lu Wang. [n. d.]. Answer Convergence as a Signal for Early Stopping in Reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, EMNLP 2025. Association for Computationa Linguistics, 17896–17907. https://aclanthology.org/2025.emnlp-main.904

[20] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegrefe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. [n. d.]. Self-Refine: Iterative Refinement with Self-Feedback. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

[21] Monoshi Kumar Roy, Simin Chen, Benjamin Steenhoek, Jinjun Peng, Gail E. Kaiser, Baishakhi Ray, and Wei Le. [n. d.]. CodeSense: a Real-World Benchmark and Dataset for Code Semantic Reasoning. In The Fourteenth International Conference on Learning Representations, ICLR 2026. OpenReview.net. https://openreview.net/forum?id=ehXVDJm0PS

[22] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. [n. d.]. Reflexion: Language Agents with Verbal Reinforcement Learning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

[23] Zhongxiang Sun, Qipeng Wang, Haoyu Wang, Xiao Zhang, and Jun Xu. 2025. Detection and Mitigation of Hallucination in Large Reasoning Models: A Mechanistic Perspective. CoRR abs/2505.12886 (2025).

[24] Gemini Team. 2025. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. CoRR abs/2507.06261 (2025).

[25] Runchu Tian, Yining Ye, Yujia Qin, Xin Cong, Yankai Lin, Yinxu Pan, Yesai Wu, Haotian Hui, Weichuan Liu, Zhiyuan Liu, and Maosong Sun. [n. d.]. DebugBench: Evaluating Debugging Capability of Large Language Models. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024. 4173–4198.

[26] Xinchen Wang, Ruida Hu, Pengfei Gao, Chao Peng, and Cuiyun Gao. [n. d.]. An Agent-based Evaluation Framework for Complex Code Generation. In 40th IEEE/ACM International Conference on Automated Software Engineering, ASE 2025, Seoul, Korea, Republic of, November 16-20, 2025. IEEE, 2427–2439.

[27] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. [n. d.]. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

[28] Yibo Wang, Zhihao Peng, Ying Wang, Zhao Wei, Hai Yu, and Zhiliang Zhu. [n. d.]. MCTS-Refined CoT: High-Quality Fine-Tuning Data for LLM-Based Repository Issue Resolution. In 40th IEEE/ACM International Conference on Automated Software Engineering, ASE 2025, Seoul, Korea, Republic of, November 16-20, 2025. IEEE, 1844–1855.

[29] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. [n. d.]. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

[30] Qiujie Xie, Qingqiu Li, Zhuohao Yu, Yuejie Zhang, Yue Zhang, and Linyi Yang. [n. d.]. An Empirical Analysis of Uncertainty in Large Language Model Evaluations. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

[31] Haoran Xue, Gias Uddin, and Song Wang. [n. d.]. An Empirical Study of Reasoning Steps in Thinking Code LLMs. In 3rd ACM International Conference on AI-powered Software, AIware 2026. https://openreview.net/forum?id=2WgHyQZ2iO

[32] Pengyu Xue, Linhao Wu, Zhen Yang, Chengyi Wang, Xiang Li, Yuxiang Zhang, Jia Li, Ruikai Jin, Yifei Pei, Zhaoyan Shen, Xiran Lyu, and Jacky Wai Keung. 2025. ClassEval-T: Evaluating Large Language Models in Class-Level Code Translation. Proc. ACM Softw. Eng. 2, ISSTA (2025), 1421–1444.

[33] Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Minghui Chen, Zheng Lin, and Weiping Wang. [n. d.]. Dynamic Early Exit in Reasoning Models. In The Fourteenth International Conference on Learning Representations, ICLR 2026. https://arxiv.org/abs/2504.15895

[34] Chengran Yang, Zhensu Sun, Hong Jin Kang, Jieke Shi, and David Lo. [n. d.]. Think Like Human Developers: Harnessing Community Knowledge for Structured Code Reasoning. In 2026 IEEE/ACM 48th International Conference on Software Engineering (ICSE ’26), April 12-18, 2026, Rio de Janeiro, Brazil.

[35] Dayu Yang, Tianyang Liu, Daoan Zhang, Antoine Simoulin, Xiaoyi Liu, Yuwei Cao, Zhaopu Teng, Xin Qian, Grey Yang, Jiebo Luo, et al. 2025. Code to think, think to code: A survey on code-enhanced reasoning and reasoning-driven code intelligence in llms. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. 2586–2616.

[36] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. [n. d.]. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

[38] Wenhao Zeng, Yaoning Wang, Chao Hu, Yuling Shi, Chengcheng Wan, Hongyu Zhang, and Xiaodong Gu. 2025. Pruning the Unsurprising: Eficient Code Reasoning via First-Token Surprisal. CoRR abs/2508.05988 (2025).

[37] Z.ai. 2026. GLM-5.1 Model Card. https://huggingface.co/zai-org/GLM-5.1. Accessed: 2026-06-15.

[39] Binquan Zhang, Li Zhang, Zhiwen Luo, Yuxin Du, Fang Liu, Song Wang, and Lin Shi. 2026. Are They All Good? Evaluating the Quality of CoTs in LLM-based Code Generation. IEEE Transactions on Software Engineering (2026), 1928–1943. doi:10.1109/TSE.2026.3676295

[40] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. [n. d.]. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.