# ReproBridge: Building Reproducibility into ML Research Artifacts

Yubo Wang<sup>1,2</sup>, Rui Meng<sup>2</sup>, Jaehyun Nam<sup>2</sup>, Tomas Pfister<sup>2</sup> and Jinsung Yoon<sup>2</sup> <sup>1</sup>University of Waterloo, <sup>2</sup>Google

Reproducing machine learning experiments requires reconstructing procedures from information scattered across papers, released code, and supporting artifacts. For coding agents, reproduction success can therefore depend on their ability to fill in missing details, rather than on the released artifacts alone. We ask whether preserving recovered procedures and decisions improves subsequent independent reproduction. We define a standardized artifact design comprising a refined codebase, environment setup, experiment launchers, and a reproduction guide, and introduce ReproBridge, an agent workflow for constructing these refined reproducible artifacts. ReproBridge-Bench provides 1,029 tasks from 387 experiments across 36 papers, anchored by five human-verified reference scores per task. Agents attempt tasks in fresh sessions with raw code or refined artifacts; a separate Auditor checks execution evidence. Across four evaluated models, refined artifacts raise mean reproduction success from 76.4% to 92.6% and reduce downstream API spending by 19.7% on average. The cross-model success-rate spread narrows from 21.6 to 3.9 percentage points, with all four models achieving mean success rates above 90%. Removing any component reduces success in the ablation study, with the largest losses from code and environment removal. These results suggest that preserving execution knowledge helps even models with lower raw-code success reproduce experiments more reliably and at lower API cost.<sup>1</sup>

## 1. Introduction

Reproducing a machine learning experiment often requires recovering details that its paper and released code leave implicit. A reproducer may resolve incompatible dependencies, identify the intended configuration, repair a data loader, and work out how run outputs become a reported score. Even a successful attempt can leave these decisions buried in conversations or shell histories, forcing the next reproducer to repeat the investigation. Code release is an essential starting point (Pineau et al., 2021); preserving the procedure recovered during execution is a further step toward reusable research artifacts (Stodden et al., 2016).

Coding-agent benchmarks expose this reconstruction burden through unresolved dependencies and misidentified outputs (Siegel et al., 2024), missing implementation details (Xiang et al., 2025), and plans that never become completed experiments (Starace et al., 2025). Consulting cited papers can recover some missing details (Zhao et al., 2026b), but successful execution may still depend on knowledge a capable agent supplies implicitly. How much does reproduction success depend on released artifacts, and how much on the reproducer’s ability to compensate for missing details?

Here, reproduction means independently re-executing machine-learning experiments using authorreleased materials, targeting numerically compatible measurements. Research artifacts comprise the code, data, checkpoints, environment specifications, and documentation supporting a study. Our hypothesis is that preserving the procedures and decisions recovered during reproduction can improve computational reproducibility. Dependency resolution can become an installation procedure; experiment discovery, a dedicated launcher; and a correction, a documented choice with its justification and afected tasks. We study this hypothesis through three contributions.

![](images/fa61897b1aab5eea49c6f2c9fd703b98de2e0cec728f8d03afd336420e5e0ace.jpg)  
Figure 1 | Preserving reproduction work for reuse. (a) ReproBridge turns a paper and raw codebase into a refined reproducible artifact (RRA) with code, environment setup, launchers, and a guide. (b,c) Hollow blue (Raw) and filled green (Refined) circles show mean success and Reproducer API spending over three evaluations; dashed lines connect the two conditions for each model. Gain is the success diference (percentage points); cost reduction is relative to Raw. These procedures improve success and reduce reconstruction efort.

Artifact design. We standardize a refined reproducible artifact with four components: a refined codebase, environment setup, experiment launchers, and a reproduction guide (Figure 1; Section 3.2). Together they preserve executable procedures and the reasons for their settings and repairs. Component removals test the contribution of each part, while an ordinary-outputs control tests reuse of code, scripts, and logs without the standardized handof.

Construction workflow. ReproBridge constructs these artifacts from a paper and its oficial repository (Section 3.1). Humans define the workflow, delivery standards, and protocols; agents specify experiments, prepare and execute them, and preserve the resulting procedures through review and packaging. Subsequent reuse is evaluated in fresh Reproducer sessions that receive either the raw codebase or the refined artifact.

Benchmark and empirical evaluation. ReproBridge-Bench provides 1,029 tasks from 387 experiments across 36 papers, with five human-verified reference scores per task (Section 2). Numerical compatibility is assessed against observed executions and their variability; a separate Auditor checks execution evidence because matching numbers alone can conceal substituted measurements. Across four coding agents, refined artifacts improve success by 5.5 to 23.2 pp and reduce downstream API spending by 16.8% to 23.7%. Robustness analyses, component ablations, case studies, and human validation of the integrity checks examine the scope and mechanisms of these gains (Section 4).

## 2. Problem formulation and ReproBridge-Bench

## 2.1. Execution deviations and reference variation

A task specifies the data, model configuration, training or evaluation procedure, and reporting statistic to reproduce. The paper’s stated setup takes precedence over conflicting code. Let � be the expected value of that statistic when the specified experiment is faithfully executed. The published score is one observation, not an error-free value of �.

For an attempt producing a measurement �, we write

$$
Y = \mu + D + \varepsilon .
$$

Here � is the shift in expectation from execution choices departing from the specified experiment, and � is zero-mean variation conditional on them. This working model draws on the distinction between systematic and random efects in measurement (Joint Committee for Guides in Metrology, 2020; Taylor and Kuyatt, 1994). Stochastic training can vary between runs (Nagarajan et al., 2018). Independent repetitions can reduce uncertainty about the mean of a fixed procedure (Bouthillier et al., 2021), but do not remove a persistent shift �. ReproBridge aims to reduce |�| by preserving procedures and decisions for faithful execution, while � remains.

We use five human-verified reference observations of the same statistic per task. Assuming faithful and independent reference executions (� = 0), their mean estimates �, and their spread characterizes �. A prediction interval accommodates uncertainty in that estimated center and variation in a new observation � (National Institute of Standards and Technology, n.d.). Section 2.3 describes collection and verification. Numerical compatibility is combined with the execution-integrity checks in Section 4.1: a nearby number alone does not show that the intended experiment was performed.

## 2.2. Selection and units of analysis

The collection targets main experimental results with accessible oficial implementations. The sampling manifest starts with 742 ICML 2026 Spotlight papers, identifies 241 with code and nonpurely-theoretical content, and randomly samples 50. Excluding six for artifact or code availability, four for compute requirements, and four for closed-model access yields 36 retained test papers. Figure 2 summarizes their research areas and paired reproduction gains. ReproBridge and evaluation workflows were developed on ten papers, five from NeurIPS 2026 and five from ICLR 2026, scored under the same task, reference, and audit conventions and reported separately in Appendix D.7.

A task is one own-method measurement in a main table, identified by dataset, metric, reporting statistic, and location; selected main-table variants are retained. Baselines, stand-alone ablations, figures, and appendix-only results are excluded. An experiment can emit several task scores. The 36 papers yield 1,463 curated tasks from 502 experiments. Table 1 reports the full verification distribution: 1,029 tasks (70.3%) have numerically compatible reference results and form the downstream evaluation set, spanning 387 experiments. The remaining 434 tasks distinguish numerical disagreement from execution and resource barriers.

## 2.3. Reference collection and task verification

The reference protocol specifies five human reproducers on separate servers, starting from the same pinned raw commit with separate execution directories and coding-agent sessions. They must not reuse another participant’s generated scripts or measurements, refined artifacts, or downstream reproduction traces. Coding-agent assistance is permitted, but humans inspect the scripts, configurations, edits, and measurement extraction actually used. Before aggregation, the five reproducers’ execution setups were cross-verified against the paper’s specified setup and the common task definition. Each reference observation measures the specified task statistic: a multiseed mean or best-of-� statistic is one observation, not interchangeable with single-seed scores.

Verifiers must fix unspecified repetition and seed choices before inspecting scores, and must not tune procedures or repeat valid measurements until they approach the published value. Diagnostic runs and corrected reruns are recorded separately with reasons for replacement; numerical disagreement alone does not justify discarding a valid execution. Appendix A.3 gives the complete verification and evidence-retention protocol.

![](images/5b2916803c0a50e27900aede0b7921149b57b3f124e2d676165591236948e7e4.jpg)

![](images/4e0ec4f2d55ca48393df1b02debdcba5509c1b37a4ae8400c0aab046d6e70555.jpg)  
Figure 2 | Research coverage and benchmark composition. (a) Eight ICML-adapted topic families and 22 subdomains cover 36 papers; sector angles encode paper counts. Outer marks show 144 paper–model pairs from the first evaluation with Codex auditing. Each paper’s four clockwise marks follow Sonnet $4 . 6 ,$ Sonnet 5, Opus 5, and Gemini 3.8 Flash. Green bars encode RRA-minus-Raw gains from the blue zero ring; dashed rings mark 25, 50, and 100 pp. Red/grey ticks indicate decreases/ties without magnitude. Legend counts are pairs; positive bins are (�, �]. (b) Paper selection. (c) Procedure and hardware are separate partitions of the same 1,029 verified tasks.

Numerical compatibility. Following Section 2.1, the five observations estimate $\mu$ and reference variation �. For task $j ,$ let $p _ { j }$ be the published score and $x _ { j 1 } , \dotsc , x _ { j 5 }$ the reference observations, with sample mean $\bar { x } _ { j }$ and sample standard deviation $s _ { j } . \mathsf { A }$ reported range is represented by its midpoint. The compatibility interval is

$$
I _ { j } ^ { ( q ) } = [ \bar { x } _ { j } - r _ { j } ^ { ( q ) } , \bar { x } _ { j } + r _ { j } ^ { ( q ) } ] , \qquad r _ { j } ^ { ( q ) } = \operatorname * { m a x } \left\{ t _ { ( 1 + q ) / 2 , 4 } s _ { j } \sqrt { 1 + \textstyle { \frac { 1 } { 5 } } } , \ 0 . 0 1 | \bar { x } _ { j } | \right\} .\tag{1}
$$

The default is $q = 0 . 9 5$ . Without the explicit 1% tolerance floor, this is a Student-� prediction interval for a new observation of the same statistic under an independent, approximately normal working model (National Institute of Standards and Technology, n.d.). It accommodates both uncertainty in the reference mean and variation in the new observation. Thresholds are fixed across raw and refined-artifact conditions; $q = 0 . 8 0$ supplies the stricter scoring check in Section 4.3.

Reference variability and task eligibility. Categories 1 and 2 both require $p _ { j } \in I _ { j } ^ { ( . 9 5 ) }$ , and difer only in relative reference variability. We define

$$
V _ { j } = 1 0 0 \frac { \mathrm { m a x } _ { i } x _ { j i } - \mathrm { m i n } _ { i } x _ { j i } } { | p _ { j } | } \quad ( p _ { j } \neq 0 ) ; \qquad V _ { j } = + \infty \quad ( p _ { j } = 0 ) .\tag{2}
$$

Stable tasks satisfy $V _ { j } \leq 1 0 $ , meaning that the reference range is at most 10% of the absolute published score; higher-variability tasks exceed this threshold. Both categories enter the downstream

Table 1 | Reference verification outcomes. A task is one selected own-method measurement from a paper’s main tables. Counts and shares cover all 1,463 curated tasks from 36 papers. Categories 1 and 2 have verified reference measurements compatible with the published score and difer at the 10% relative-variation threshold; together they supply the 1,029 downstream tasks. Categories 3–5 distinguish numerical disagreement, execution blockers, and resource/access limits.
<table><tr><td>Cat.</td><td>Verification outcome</td><td>Interpretation</td><td>Tasks</td><td>Share (%)</td></tr><tr><td>1</td><td>Reproduced: stable</td><td>Compatible reference, low variability</td><td>701</td><td>47.9</td></tr><tr><td>2</td><td>Reproduced: variable</td><td>Compatible reference, higher variability</td><td>328</td><td>22.4</td></tr><tr><td>3</td><td>Numerical mismatch</td><td>Executed, but numerically incompatible</td><td>183</td><td>12.5</td></tr><tr><td>4</td><td>Execution blocked</td><td>Missing artifacts, dependencies, or code defects</td><td>186</td><td>12.7</td></tr><tr><td>5</td><td>Resource/access limit</td><td>Compute capacity or restricted access</td><td>65</td><td>4.4</td></tr><tr><td colspan="3">Downstream evaluation (Categories 1 + 2)</td><td>1,029</td><td>70.3</td></tr><tr><td colspan="4">Total curated tasks</td><td>1,463 100.0</td></tr></table>

evaluation. We pool them for the main comparison and examine them separately in Section 4.3.   
Appendix A.4 gives the uncertainty derivation and additional details of the numerical rules.

## 2.4. Scope and information access

The evaluated set contains 631 full-procedure and 398 evaluation-only tasks (730 GPU, 299 CPU-only). Evaluating an explicitly permitted oficial checkpoint does not require repeating its training. Evaluation alone cannot establish completion of a training task.

All eight downstream conditions use the same task categories and numerical thresholds, applied outside the Reproducer session. Published targets remain visible in the paper and task specification; the Auditor checks that submitted values come from new executions. Session instructions withhold the reference information, alternative repository condition, and other agents’ run directories. An allowed checkpoint may serve as input; a shipped score file never counts as a measurement.

The benchmark includes 586 lower-is-better and 443 higher-is-better measurements, all assessed by the same two-sided compatibility rule. This evaluates agreement with the specified experimental result: a score beyond either bound remains numerically incompatible, including one that improves on the published value. Reference submissions must also link each score to the starting code commit, executed scripts and edits, environment and hardware, configuration and seeds, raw outputs, and score-extraction or aggregation procedure, making the reported statistic traceable to a concrete execution. Appendix A gives selection records, verification protocols, and statistical rules.

## 3. ReproBridge: artifact design and construction

ReproBridge turns reproduction work into an artifact that a later researcher or agent can use without access to the construction conversation. The handof comprises refined code, environment setup, experiment launchers, and a reproduction guide. Its construction workflow preserves the evidence behind execution choices as well as the executable steps themselves.

## 3.1. Construction workflow

Humans define the workflow, delivery standards, and protocols; agents perform all construction, review, and packaging. Figure 3 organizes this work into three stages: Specify fixes the intended experiment before execution; Execute establishes a working procedure and records measurements, repairs, and failures; and Preserve packages that procedure and its rationale for reuse. Table 2 specifies each role’s inputs and persistent outputs.

![](images/ddb2fb1132bf63e952bb051d458025452e645394653e598f5cb94cd0521fed67.jpg)  
Figure 3 | ReproBridge artifact construction. Six roles form three stages: Specify (Intake, Planner), Execute (Builder, construction Reproducer), and Preserve (Distiller, Packager). Footers list stage outputs. Solid arrows show construction; dashed links show independent review and revision. The final handof contains refined code, environment setup, launchers, and a guide. Downstream reproduction and auditing are separate evaluation steps.

Table 2 | Construction skills: inputs, responsibility, and persistent outputs. A checkpoint reviewer is reused at diferent stages in separate contexts. The construction Reproducer is not the fresh downstream session used to evaluate reuse.
<table><tr><td>Role</td><td>Input and responsibility</td><td>Output and check</td></tr><tr><td>Intake</td><td>Paper, pinned raw code, selected tasks; read and map the inputs</td><td>Input manifest, experiment groups, artifact availability, initial ambiguities</td></tr><tr><td>Planner</td><td>Task mapping and ambiguities; specify the measurement before execution</td><td>Task specifications with source references, recorded choices, execution plans, draft</td></tr><tr><td>Builder</td><td>Dependency and artifact requirements; construct a working environment</td><td>launchers Locked dependencies, setup and data scripts, container definition, smoke-test</td></tr><tr><td>Reproducer</td><td>Plans and verified setup; execute and record justified changes</td><td>evidence File-backed measurements, code changes, repetition records, explicit failure reasons</td></tr><tr><td>Distiller</td><td>Execution evidence, repairs, and decisions; synthesize the handoff</td><td>Reproduction guide, change record, aggregation and variability guidance</td></tr><tr><td>Packager</td><td>Code, setup, launchers, and documents; assemble a portable deliverable</td><td>Refined artifact and clean-shell verification evidence</td></tr><tr><td>Reviewer</td><td>Plans, execution records, or completed artifact; challenge unsupported claims</td><td>Localized findings at planning, reproduction, and final handoff checkpoints</td></tr></table>

Persistent evidence connects the stages. A phase controller sequences the roles and checks required outputs between stages. Before scored execution, task specifications record the data, model, procedure, reporting statistic, source references, and unresolved choices. This makes execution decisions explicit before their results are known. When paper and code conflict, agents follow the paper’s setup and flag the discrepancy (Section 6). Each measurement must link to an actual output file; blocked tasks retain an explicit failure reason, and diagnostic runs remain separate from scored repetitions. The guide and change record are then grounded in these stored specifications, execution records, and justified repairs.

Independent review tests the evidence and the handof. A Reviewer inspects plans, execution records, and the completed artifact in contexts separate from the roles being reviewed. It returns localized findings to the responsible stage for revision and rechecking. Delivery also requires testing the supplied entry points from a clean shell, including an inexpensive real repetition and checks on output structure and score extraction. The delivered environment must match the one used for construction measurements. Human reference information is withheld throughout construction. Appendix B gives the full role specifications and delivery checks.

## 3.2. Four components preserve execution knowledge

A later reproducer needs to know what code to run, how to prepare its environment, how to obtain the requested measurement, and why particular choices were made. The four-component refined reproducible artifact (RRA in figures and tables) addresses these needs. The refined codebase repairs execution defects and supplies missing experiment support. Environment setup specifies compatible dependencies and data or checkpoint acquisition. Experiment launchers connect tasks to configurations, execution scope, outputs, and measurement extraction, distinguishing scored runs from smoke tests. The reproduction guide, including its change record, explains settings, paper–code discrepancies, reporting statistics, and resource or numerical sensitivities. Section 4.5 tests the contribution of each component through removal experiments.

The guide connects instructions to their rationale. Each ambiguity entry records its source location, the gap, the adopted resolution and rationale, sensitivity, and afected tasks. The change record distinguishes execution fixes, correctness fixes, and convenience changes, while retaining pitfalls and open choices. Repetition guidance preserves the specified reporting unit and characterizes variation � without selecting favorable outcomes. Figure 4 illustrates how SCIQL’s guide links measurement decisions and failure checks to executable support; Appendix B.2 expands its twelve-section structure.

![](images/de1ed0bc9b9e74b3cdacad3cc8a74a4844773bc5f95f38fbdda85406ff54bc77.jpg)  
Figure 4 | A refined artifact for SCIQL. Left: selected guide entries specify the six-criterion, five-seed protocol, unweighted aggregation, and checks for silently empty datasets. Right: an alias applied before unpickling preserves episode classes; CPU PyTorch avoids a cuDNN conflict with GPU JAX; launchers connect configurations to execution, output collection, and aggregation. The four components preserve executable steps and the reasons for them.

## 4. Experiments

## 4.1. Evaluation protocol and setup

Opus 5 constructs the refined artifacts for the main experiment. Fresh Reproducer sessions receive the paper, selected tasks, and either raw code or its refined artifact. Construction conversations, hidden reference observations, and the alternative condition are withheld. Sessions submit new measurements and execution evidence; we compare success, token usage, and API spending.

Sonnet 4.6, Sonnet 5, Opus 5, and Gemini 3.8 Flash (Gemini 3.8 in figures and tables) attempt all 1,029 tasks from 36 test papers, with one fresh session per paper–model–condition per evaluation. Both raw-code and refined-artifact conditions use a 24-hour wall-clock limit per paper. GPU experiments use eight H100-80GB GPUs. Codex is the primary Auditor. Harnesses difer across vendors, so cross-model comparisons are descriptive. Paper-wise, stratified, and task-level diagnostics use the first evaluation. Appendix C details the harness, information boundaries, and auditing.

Execution-integrity audit. An independent Auditor checks the evidence chain for each reported result, tracing the value to its output location, execution record, and generating script. It uses ledger timestamps to check execution and extraction order, and available script and output hashes to verify that the inspected files match the recorded versions. The Auditor cross-checks executed code, logs, and extracted values to locate copied measurements, altered outputs, or substituted computations.

After mechanical screening identifies leads, a Finder inspects executed code and records and proposes localized violations. A separate Refuter tries to overturn each proposed invalidation by checking the evidence, its efect on the measurement, and the task specification. Unsubstantiated findings remain uncertain. If the evidence remains contested, a panel examines its support, the measurement efect, and alternative explanations. The Auditor operates after reproduction and provides no feedback to the Reproducer.

Invalidation criteria and task success. The five eligible violation classes distinguish copying a value without measuring it, substituting the measurement instrument, altering an operand used by the intended formula, choosing the reported outcome conditional on observed values, and computing the wrong statistic. Invalidation requires an eligible violation supported by localized evidence, a material efect on the measurement, and a failed refutation. A numerical mismatch, crashed run, hardware diference, or suspicion without a localized artifact does not automatically meet these conditions. Paper–code discrepancies are assessed against the same criteria.

A task succeeds when its result lies in the fixed compatibility interval (Equation 1) and is not invalidated by the primary Auditor; missing or incompatible results fail. Uncertain audit findings do not count as invalidations. Section 4.6 checks the sensitivity of this scoring rule to the Auditor and validates audit judgments against human review.

## 4.2. Refined artifacts improve success and reduce downstream cost

Refined artifacts improve success and lower downstream cost for every tested model (Table 3). Gains range from 5.5 pp for Opus 5, the strongest raw reproducer, to 23.2 pp for Gemini, with 6.3% to 19.8% fewer tokens and 16.8% to 23.7% lower API spending, including failed attempts. Crossmodel summaries weight models equally; API savings are relative to each model’s raw-code cost.

The cross-model success-rate spread narrows from 21.6 to 3.9 pp. This suggests that, given suficient execution information, Reproducers with lower raw-code success can achieve high success. This supports our hypothesis that preserving execution knowledge reduces the need to infer missing details. Section 4.3 examines paper-level heterogeneity and scoring sensitivity.

External benchmark. We compare handofs on 45 CORE-Bench Hard capsules (Siegel et al., 2024); Sonnet 5 constructs the refined artifacts and VERITAS’s patched codebase and fix log (Liu et al., 2026a) (Table 4). Refined artifacts achieve the highest success, reduce median duration from 33 to 14 minutes, and lower spending per successful capsule from \$8.43 to \$4.34. VERITAS lies between raw and refined handofs; the latter matches or exceeds raw success in each area.

Table 3 | Main results on 36 test papers and 1,029 tasks. Raw uses the released codebase; RRA uses the refined reproducible artifact. Entries report mean ± standard deviation over three evaluations (SD in subscripts). Success requires numerical compatibility and no confirmed audit violation. Gain is RRA minus Raw in percentage points; Red. is 100(Raw − RRA)/Raw, computed from the means. Tokens (B: billions) and API USD sum Reproducer sessions, including failed attempts; construction, auditing, and GPU costs are excluded.
<table><tr><td rowspan="2"></td><td colspan="3">Success (%)</td><td colspan="3">Tokens (B)</td><td colspan="3">API (USD)</td></tr><tr><td>Model</td><td>Raw RRA</td><td>Gain</td><td>Raw</td><td>RRA</td><td>Red.</td><td>Raw</td><td>RRA</td><td>Red.</td></tr><tr><td>Sonnet 4.6</td><td>72.3±1.1</td><td>91.6±3.4</td><td>+19.3</td><td> $8 . 4 8 _ { \pm 0 . 3 8 }$ </td><td>6.80±0.76</td><td>19.8</td><td>3,165±142</td><td>2,512±286</td><td>20.6</td></tr><tr><td>Sonnet 5</td><td>76.5±2.8</td><td>93.3±1.4</td><td>+16.8</td><td>4.98±0.55</td><td>4.52±0.24</td><td>9.2</td><td>1,726±175</td><td>1,317±102</td><td>23.7</td></tr><tr><td>Opus 5</td><td> $8 9 . 2 _ { \pm 1 . 2 }$ </td><td> $9 4 . 7 _ { \pm 2 . 2 }$ </td><td>+5.5</td><td> $3 . 0 1 _ { \pm 0 . 1 6 }$ </td><td> $2 . 8 2 _ { \pm 0 . 2 9 }$ </td><td>6.3</td><td> $^ { 3 , 4 8 0 _ { \pm 1 6 5 } }$ </td><td> $2 , 8 6 5 _ { \pm 3 1 2 }$ </td><td>17.7</td></tr><tr><td>Gemini 3.8</td><td> $6 7 . 6 _ { \pm 1 . 9 }$ </td><td> $9 0 . 8 _ { \pm 1 . 1 }$ </td><td>+23.2</td><td> $2 . 2 4 _ { \pm 0 . 2 6 }$ </td><td> $1 . 8 3 _ { \pm 0 . 0 9 }$ </td><td>18.3</td><td> $3 9 2 _ { \pm 4 1 }$ </td><td> $3 2 6 _ { \pm 1 6 }$ </td><td>16.8</td></tr></table>

Table 4 | Artifact reuse on CORE-Bench Hard. Sonnet 4.5 reproduces 45 capsules from raw code, VERITAS’s patched code and fix log, or a refined artifact (RRA); Sonnet 5 constructs both handofs. A capsule succeeds only if all its questions satisfy the benchmark’s reference intervals (79 questions overall). (a) Overall capsule success, Reproducer API USD per successful capsule, and median Reproducer session duration. (b) Success within computer (CS), medical (Med.), and social (Soc.) sciences; parentheses give capsule counts.  
(a) Overall
<table><tr><td>Handoff</td><td>Success (%) USD/success</td><td>Time (min)</td></tr><tr><td>Raw</td><td>62.2 8.43</td><td>33.0</td></tr><tr><td>VERITAS</td><td>68.9 6.78</td><td>24.5</td></tr><tr><td>RRA</td><td>73.3 4.34</td><td>14.4</td></tr></table>

(b) Success by area (%)
<table><tr><td colspan="2">Handoff CS(18) Med. (13) Soc. (14)</td></tr><tr><td>Raw 50.0</td><td>76.9 64.3</td></tr><tr><td>VERITAS 61.1</td><td>84.6 64.3</td></tr><tr><td>RRA 66.7</td><td>84.6 71.4</td></tr></table>

## 4.3. Gains persist across papers and scoring thresholds

The main comparison pools task outcomes, so papers with more selected measurements contribute more weight. We examine whether gains persist when papers receive equal weight, when numerical scoring is stricter, and within each reference-variability group. These diagnostics use the first evaluation; Table 3 reports means over three evaluations.

Paper-level heterogeneity. Figure 5 compares raw and refined-artifact success within each of the 36 papers. Paper-macro rates average these per-paper percentages with equal weight. All four models improve under this aggregation, with gains of 6.7 to 20.7 pp (Table 5a). Success improves or ties on 30 to 33 papers per model, while three to six papers regress. Thus the aggregate gains extend across papers, although refinement does not improve every paper. Section 4.4 examines remaining failures and task-level reversals.

Stricter numerical compatibility. Replacing the default 95% interval with the 80% interval in Equation 1 preserves the selected tasks and the primary Auditor’s judgments. Refined artifacts still improve success for every model, with gains of 10.3 to 39.1 pp (Table 5b). The positive gains therefore persist under a tighter numerical criterion.

![](images/b10e1b1a3b9a604ba16ffd7b991de5681f1620e2fe6139a7884080dbc5d679d8.jpg)  
Figure 5 | Paper-wise reproduction success in the first evaluation. Each panel contains one point per retained paper, comparing the percentage of the same tasks that succeed with Raw (horizontal axis) and the refined artifact, RRA (vertical axis), under primary Codex scoring. Marker area is proportional to task count. Green points above the equality line improve, blue points on the line tie, and red points below it regress. Overlapping observations retain their exact positions.

Table 5 | Gains persist with equal paper weights and stricter scoring. First-evaluation results under primary Codex scoring. (a) Paper-macro success gives equal weight to all 36 papers with the default 95% interval; I/T/W counts improved, tied, and worse papers. (b) Task-level success uses the 80% interval on the same 1,029 tasks, with category labels fixed. Raw and RRA are success percentages; Gain is RRA minus Raw in percentage points, calculated before rounding.
<table><tr><td colspan="4">(a) Paper-macro, 95% interval</td><td colspan="4">(b) Task-level, 80% interval</td></tr><tr><td>Model</td><td>Raw</td><td>RRA</td><td>Gain</td><td>I/T/W</td><td>Raw</td><td>RRA</td><td>Gain</td></tr><tr><td>Sonnet 4.6</td><td>69.9</td><td>89.7</td><td>+19.8</td><td>25/8/3</td><td>61.3</td><td>88.2</td><td>+26.9</td></tr><tr><td>Sonnet 5</td><td>75.1</td><td>90.6</td><td>+15.5</td><td>23/9/4</td><td>63.9</td><td>90.5</td><td>+26.5</td></tr><tr><td>Opus 5</td><td>87.5</td><td>94.2</td><td>+6.7</td><td>16/15/5</td><td>82.0</td><td>92.3</td><td>+10.3</td></tr><tr><td>Gemini 3.8 Flash</td><td>67.2</td><td>87.9</td><td>+20.7</td><td>25/5/6</td><td>47.7</td><td>86.8</td><td>+39.1</td></tr></table>

Reference variability. Both reference-variability groups improve across all four models, with larger point-estimate gains in Cat. 1: 8.0 to 23.3 pp versus 2.1 to 22.3 pp in Cat. 2. The groups difer in task composition and interval width, so this contrast is descriptive. Appendix D provides the full stratified results.

## 4.4. Preserved procedures support recovery and reduce reconstruction

![](images/f12c4b63d89eb0e10b809bff7cfb0c8722338e4891a9c38570a8dd7e89b6eb47.jpg)

![](images/50dd3a9c4ba3be7b585d95a36f5a4b170e3f60396aef55dd5c0f626b7fb029f1.jpg)  
Figure 6 | Recovering failures and shortening preparation. First-evaluation results across four models and 1,029 tasks per model. Left: positive bars count Raw failures becoming RRA successes, grouped by the Raw outcome: numerical mismatch, execution/resource barrier (Not run), or audit-invalid numerical match. Negative bars count Raw successes lost with RRA; labels give recovery and reversal totals. Right: median time to first scored execution across paper sessions (minutes); hollow and filled markers denote Raw and RRA, respectively.

Case studies and qualitative analyses use the first evaluation. Across paired task–model outcomes, 813 raw failures become successes, against 136 reversals (Figure 6). Recoveries include 348 numerical mismatches, 210 unexecuted tasks due to execution or resource barriers (Not run), and 255 audit-invalid numerical matches. These records suggest two complementary contributions: completing ambiguous or incomplete measurement specifications and preserving executable procedures for reuse.

Completing and clarifying measurement specifications. In Mixture of Concept Bottleneck Experts (De Santis et al., 2026), the reduction of multiple tree-edit distances is unspecified, and the released code lacks a collector for Table 1. Sonnet 5’s raw attempt sums the four distances, reporting 60 rather than 15. The refined artifact resolves this ambiguity by specifying the per-expression mean and supplying the collector; the later attempt reports 15. SCIQL’s released code similarly leaves the mapping from run outputs to table entries incomplete; the refined artifact specifies the output field and aggregation across criteria and seeds. Such clarification and experiment-to-result mapping together represent about 35% of the case-review attribution distribution.

Recovered configurations become executable: HELIX. For HELIX (Zhang et al., 2026a), Opus 5 restores the paper’s architecture on raw code but retains the shipped batch size and learning rate. The refined artifact identifies a repository revision matching that architecture, records it as the source of the adopted settings, and exposes them through a launcher. A later attempt produces numerically compatible MAEs for both illustrated tasks. The guide records the evidence for the choice, and the launcher makes it reusable (Appendix D.5).

A compatible environment transfers across sessions: WBMM. For six COCO tasks in WBMM (Song et al., 2026), Sonnet 4.6’s raw attempt fails to install a compatible vision-library stack for H100. The refined artifact preserves the dependency versions, build prerequisite, and source-build route for the target architecture. The later session executes all six tasks without a code patch; its box and mask AP values agree numerically with the reported targets. This case shows how an environment repair can become a reusable installation procedure (Appendix D.5).

Fixing bugs in the released code. In Efective Model Pruning (Wang et al., 2026), the released loader silently labels all 10,000 TinyImageNet validation images as one class. During construction, ReproBridge adds a script organizing images by their true labels and a launcher check requiring prepared data. In paired Gemini 3.8 Flash runs, ResNet-18 validation loss changes from 8.898 on raw code to 2.288 with the refined artifact, close to the paper’s 2.303. Model–paper sessions changing code downstream fall from 113/144 to 54/144, and median time to first scored execution decreases for every model (Figure 6, right). Together with lower token use, this indicates less reconstruction.

Explicit procedures also help the model that built them. Clearer code organization and explicit execution decisions can reduce repeated search and context demands, even for the model that built the artifact. In ETTFS (Che et al., 2026), the raw Opus 5 attempt searches 6,377 lines of Python for measurement code absent from the release. The refined artifact organizes the added measurement code into a dedicated profiling module, a launcher that sequences training and profiling, and documented output fields. A fresh Opus 5 session follows this structure and completes a previously blocked measurement.

Remaining failures. Failures shared by both handofs mainly reflect paper–code setup mismatches, typographical errors in the paper, or experimental variability placing measurements outside the compatibility interval. Refinement can also preserve or introduce incorrect experimental choices.

The 136 reversals from raw success to refined failure comprise 84 numerical mismatches, 31 resource/access failures, 11 audit-invalid matches, and 10 execution blockers. AuxiliaryMCMC and LiME account for 50 and 14 reversals, respectively, or 47.1% combined. The LiME case (Kowsher et al., 2026) illustrates an attempt that fails to complete training and evaluation; AuxiliaryMCMC (Yuan and Wang, 2026) illustrates numerical fragility, hardware sensitivity, and protocol deviations under time pressure (Appendix D.3).

## 4.5. Each artifact component contributes to reuse

Ordinary outputs are Opus 5’s raw-code execution directory with code, scripts, and logs retained and result files removed. They carry no standardized guide.

Component-removal protocol. Refinedcode removal restores the original source; environment removal deletes the setup route; launcher removal deletes the experiment entry points. For these three conditions, descriptions of the removed component are also deleted from the remaining documentation and scripts. The guide ablation removes the guide itself. The change record belongs to the documentation component.

Table 6 | Handof and component ablations. Sonnet 5, 287 tasks, 12 papers, one evaluation. Loss is the drop from full RRA (pp); removals follow the adjacent protocol.
<table><tr><td>Handoff Success (%) Loss (pp)</td></tr><tr><td>Raw code 73.9</td></tr><tr><td>Ordinary outputs 81.2</td></tr><tr><td>Full RRA 93.0</td></tr><tr><td>Full RRA without one component:</td></tr><tr><td>Refined codebase 87.8 5.2</td></tr><tr><td>Environment setup 88.5 4.5</td></tr><tr><td>Reproduction guide 89.2 3.8</td></tr><tr><td>Experiment launchers 89.9 3.1</td></tr></table>

A source file and an explanation of how to reconstruct it can carry overlapping knowledge. Deleting only the file leaves a route for the downstream agent to restore the same functionality. Removing both the component and its descriptions tests the loss of that information while leaving the other components available; the resulting losses are conditional on those retained materials.

The full artifact improves on ordinary outputs. The complete artifact outperforms ordinary outputs by 11.8 pp (Table 6). Ordinary outputs also outperform raw code, demonstrating value in reusing execution work. Both handofs come from separate upstream runs with comparable resource use. The comparison evaluates protocol-guided construction and its resulting handof.

Code and environment removal cause the largest losses, showing the need for repaired implementations and compatible dependencies. Removing the guide or launchers also reduces success when code and environment remain available: an executable codebase can still leave settings, measurement rules, and entry points implicit. Retained components can partly compensate for a removal (Appendix D.4). These losses favor pairing executable support with explicit guidance on measuring results.

## 4.6. Gains persist across Auditors despite imperfect detection

Sensitivity to the Auditor. Codex and Claude inspect the same numerically matching outcomes in the first evaluation (Table 7). Refined artifacts improve success under either Auditor for every model. Diferences in invalidations change the absolute success rates, while the direction of the raw-to-refined comparison is preserved.

Validation against human review. Two human reviewers cross-verified all 409 selected tasks from eight randomly selected papers under both Raw and RRA conditions, totaling 818 outcomes.

Table 7 | Sensitivity to the Auditor in a single evaluation. Codex and Claude inspect the same numerically matching outcomes for each model and handof. Invalid / matched reports the Auditor’s invalidations and the number of numerical matches; Success is 100(matched − invalid)/1,029. Raw denotes the released codebase and RRA the refined artifact.
<table><tr><td rowspan="2">Reproducer Input</td><td rowspan="2"></td><td colspan="2">Codex</td><td colspan="2">Claude</td></tr><tr><td>Invalid / matched</td><td>Success (%)</td><td>Invalid / matched</td><td>Success (%)</td></tr><tr><td>Sonnet 4.6</td><td>Raw</td><td>63/805</td><td>72.1</td><td>41/805</td><td>74.2</td></tr><tr><td>Sonnet 4.6</td><td>RRA</td><td>23/968</td><td>91.8</td><td>6/968</td><td>93.5</td></tr><tr><td>Sonnet 5</td><td>Raw</td><td>51/838</td><td>76.5</td><td>4/838</td><td>81.0</td></tr><tr><td>Sonnet 5</td><td>RRA</td><td>0/962</td><td>93.5</td><td>0/962</td><td>93.5</td></tr><tr><td>Opus 5</td><td>Raw</td><td>43/961</td><td>89.2</td><td>0/961</td><td>93.4</td></tr><tr><td>Opus 5</td><td>RRA</td><td>10/991</td><td>95.3</td><td>0/991</td><td>96.3</td></tr><tr><td>Gemini 3.8</td><td>Raw</td><td>173/869</td><td>67.6</td><td>119/869</td><td>72.9</td></tr><tr><td>Gemini 3.8</td><td>RRA</td><td>6/938</td><td>90.6</td><td>3/938</td><td>90.9</td></tr></table>

Review includes outcomes not flagged by either Auditor. Each reviewer traces the measurement to primary execution evidence, records an initial judgment, and cross-checks the other’s assessment before they agree on the final label. Selection is at the paper level, and task outcomes can share execution evidence. Appendix C.4 gives the complete protocol.
<table><tr><td></td><td>Raw·Codex</td><td></td><td>Raw·Claude</td><td></td><td></td><td>RRA·Codex</td><td>RRA·Claude</td><td></td></tr><tr><td>TP -FN</td><td>26</td><td>7</td><td>20</td><td>13</td><td>5</td><td>1</td><td>4</td><td>2</td></tr><tr><td>FP - TN</td><td>3</td><td>373</td><td>4</td><td>372</td><td>1</td><td>402</td><td>0</td><td>403</td></tr></table>

Figure 7 | Audit judgments against human verification. Two humans cross-verify 409 tasks from eight papers under both Raw and refined-artifact (RRA) conditions (818 outcomes). Each 409-outcome matrix has human invalid/valid rows and Auditor flag/pass columns. TP and FN are detected and missed invalid outcomes; FP and TN are flagged and accepted valid outcomes. Both Auditors are scored against the same human labels.

Against the shared human labels (Figure 7), Codex achieves 88.6% precision and 79.5% recall, versus Claude’s 85.7% and 61.5%. Each Auditor falsely flags four of 779 valid outcomes (0.5%). Codex detects seven more invalid outcomes at the same false-positive count. The shared labels identify 39 invalid outcomes: 33 with Raw and six with RRA. Codex misses eight of these outcomes, and Claude misses 15.

In a WBMM timing case, reports replace measured epoch durations with full-dataset projections, including 32.5 seconds reported as 206 seconds; files are edited after process exit. These projections do not constitute the requested measurements. The measured durations also satisfy the numerical intervals, making execution evidence decisive for detecting the substitution. Before invalidation, the Refuter checks the measurement efect and whether permitted protocol choices explain the apparent violation. Appendix C presents the human-review protocol and additional audit cases.

## 4.7. Repeated reuse can ofset construction API cost

Constructing the 36 artifacts costs \$5,026 in API usage (\$139.61 per paper). At the mean downstream costs in Table 3, break-even requires 8, 13, and 9 reuse rounds for Sonnet 4.6, Sonnet 5, and Opus 5, respectively. Each round repeats all 36 papers; the scenario charges the full construction cost to one model, holds prices and savings fixed, and excludes GPU, auditing, and maintenance costs. The corresponding API savings are \$653, \$409, and \$615 per complete reuse round, respectively. Higher success accompanies these savings, so reuse improves reproduction while progressively recovering the construction investment. Appendix E details the accounting rates and break-even calculation.

## 5. Related work

Scientific trust and computational reproducibility. Scientific trust depends on collective scrutiny and accessible evidence (Munafò et al., 2017; Nosek et al., 2015; Oreskes, 2019); AIassisted research retains these requirements (Messeri and Crockett, 2024). Following ACM terminology, reproducibility means independent execution of authors’ artifacts, rather than independent reimplementation or bitwise-identical replay (Association for Computing Machinery, 2020; Peng, 2011). Reproducibility Enhancement Principles and the NeurIPS reproducibility program promote explicit dependencies, code release, and reporting checklists (Pineau et al., 2021; Stodden et al., 2016).

Research reproduction benchmarks. PaperBench assesses reconstruction through expert rubrics covering code, execution, and results (Starace et al., 2025); SciReplicate-Bench also studies implementation reconstruction (Xiang et al., 2025). REPRO-Bench and ReplicationBench extend assessment to social science and astrophysics (Hu et al., 2025; Ye et al., 2025). CORE-Bench tests execution of released Code Ocean capsules and questions about their outputs (Siegel et al., 2024). ReproBridge-Bench evaluates released artifacts through binary task success, combining intervals based on variability across five human-verified executions with an independent integrity audit. Tasks include GPU training and evaluation. Table 4 compares handofs under CORE-Bench’s scoring rule.

Automated reproduction and reusable experience. PaperCoder constructs repositories through planning and code generation (Seo et al., 2026). AutoReproduce recovers implicit details from cited papers and evaluates execution fidelity (Zhao et al., 2026b). VERITAS executes research procedures and produces patched code, fix records, and claim assessments (Liu et al., 2026a); our external comparison tests reuse of its handof. Reflexion retains verbal feedback, while Voyager accumulates executable skills (Shinn et al., 2023; Wang et al., 2023). The Compiler for Agent-Native Research Artifacts (ARA) (Liu et al., 2026b) packages papers and repositories with expert rubrics in its PaperBench study; a model judge scores reproduction against those rubrics. We construct runnable four-component handofs without evaluation rubrics or human reference scores, and assess their value through four component removals and an ordinary-outputs control.

## 6. Discussion and limitations

The four component contracts make reproduction experience reusable, and ReproBridge-Bench provides verified reference scores for evaluation. We study their roles together; the delivery protocol guides what ReproBridge recovers and preserves during construction. In the formulation of Section 2.1, the aim is to reduce avoidable deviations � while documenting uncertainty about the intended experiment.

## 6.1. Use before submission and by later reproducers

The primary intended use is for authors to apply ReproBridge as a reproducibility check before submitting a paper and releasing its artifacts. Executing the paper’s specified setup can expose missing details, execution barriers, and paper–code inconsistencies, giving authors concrete issues to inspect and correct before submission. Third-party reproducers can likewise preserve their work in the same format to lower the cost of later attempts. A human reader can use the setup route and launchers to execute experiments, then inspect the guide and change record to assess the settings and repairs. Our experiments measure reuse by coding agents.

## 6.2. Paper–code misalignment

When the paper and released code conflict, ReproBridge strictly follows the paper’s setup and flags the discrepancy with source locations and afected tasks. For example, the Manifold-Aware Perturbations guide identifies a learning rate of $1 0 ^ { - 4 }$ in the paper versus $1 0 ^ { - 3 }$ in the released code. Such records help authors check consistency between their manuscript and implementation during the pre-submission review described above.

A separate agent-assisted static analysis records 238 in-scope contradiction findings touching 669 distinct selected tasks. A contradiction requires an explicit paper statement and conflicting implemented behavior; absent implementations and choices on which the paper is silent are recorded separately. These findings are possible sources of execution deviation �, but their efects on measured performance require controlled comparisons. They are neither human-validated error rates nor explanations of the measured gains. The analysis is separate from primary success scoring, and its records were not supplied to evaluated Reproducers. Appendix D.6 documents the scope, categories, and evidence.

## 6.3. Limitations

The evaluation conditions on retained papers, fixed reference regions, and the tested agents; it is not a conference-wide reproduction rate. Five reference observations are a protocol choice, not a guarantee of adequate uncertainty estimation. Numerical compatibility does not certify scientific validity or complete fidelity to the stated experiment, and refined artifacts can preserve incorrect assumptions. Automated audits also miss some violations, as quantified against human verification.

The complete artifact intervention includes upstream investigation, repair, review, and packaging. Ordinary outputs and refined artifacts are produced in separate runs with comparable upstream resource use. This comparison evaluates the construction workflow and resulting handof together; it does not isolate the efect of reorganizing identical outputs. Component ablations measure losses conditional on the remaining materials. Case annotations and variability strata describe patterns rather than isolate causal efects.

API savings concern downstream Reproducer sessions. Construction, independent verification, GPU execution, maintenance, and human review have separate costs; reuse estimates assume fixed prices and savings. Human time savings and comprehension require a separate study.

## 7. Conclusion

ReproBridge standardizes recovered reproduction procedures and decisions into four reusable components: a refined codebase, environment setup, experiment launchers, and a reproduction guide. ReproBridge-Bench provides verified reference scores to assess their value. Reproduction success depends jointly on artifact quality and reproducer ability. All four evaluated models achieve mean reproduction success above 90% with refined artifacts, including those with lower raw-code success. The cross-model success-rate spread and downstream API spending both decrease. Removing code or environment setup causes the largest ablation losses; removing guides or launchers also reduces success with the other components retained.

Future work should evaluate reuse by human researchers and measure the full costs of constructing, verifying, and maintaining artifacts across repeated attempts.

## AI use statement

Coding agents constructed the refined artifacts, performed downstream reproductions, and produced automated integrity audits as part of the experiments (Sections 3 and 4.1). They also assisted human reference reproducers with setup, debugging, and script preparation under human verification (Appendix A.3). Agent-assisted reviews annotated recovered outcomes and paper–code discrepancies; these annotations are descriptive and do not establish causal efects or human-validated error rates (Appendices D.2 and D.6).

We used generative AI to draft and edit text, write and check analysis scripts, and prepare figure designs that were redrawn as editable diagrams. The authors defined the research question, hypothesis, benchmark, workflow, and evaluation protocols; reviewed all AI-assisted work; and checked the reported numbers and statistical plots against the recorded results and frozen analysis inputs. We did not use generative AI to generate synthetic data. The authors take responsibility for the final text, claims, and artifacts.

## References

S. Agrawal, T. H. Nguyen, and D. Kiela. BlitzRank: Principled zero-shot ranking agents with tournament graphs. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/agrawal26d.html. ICML 2026 Spotlight.

R. Angell, R. Singhal, Z. Horvitz, Z. Yu, R. Ranganath, K. McKeown, and H. He. Estimating tail risks in language model output distributions. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/angell26a.html. ICML 2026 Spotlight.

Association for Computing Machinery. Artifact review and badging, 2020. URL https://www.acm.org/pu blications/policies/artifact-review-and-badging-current. Version 1.1 terminology; accessed November 4, 2026.

X. Bouthillier, P. Delaunay, M. Bronzi, A. Trofimov, B. Nichyporuk, J. Szeto, N. Mohammadi Sepahvand, E. Raf, K. Madan, V. Voleti, S. Ebrahimi Kahou, V. Michalski, T. Arbel, C. Pal, G. Varoquaux, and P. Vincent. Accounting for variance in machine learning benchmarks. In Proceedings ofMachine Learning and Systems, volume 3, 2021. URL https://proceedings.mlsys.org/paper\_files/paper/2021/hash/0184b 0cd3cfb185989f858a1d9f5c1eb-Abstract.html.

M. Brockschmidt, M. Schröder, and S. Feuerriegel. SurvDif: a difusion model for generating synthetic data in survival analysis. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/brockschmidt26a.html. ICML 2026 Spotlight.

K. Che, Z. Ma, Y. Huang, P. Xue, L. Yuan, W. Fang, and Y. Tian. Eficiently training Time-to-First-Spike spiking neural networks from scratch. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/che26a.html. ICML 2026 Spotlight.

Y. Chen, Y. Liu, X. Xu, P. Zhang, M. Beyer, M. Rapp, J. Zhu, and J. Chen. TetraJet-v2: accurate NVFP4 training for large language models with oscillation suppression and outlier control. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v3 06/chen26cl.html. ICML 2026 Spotlight.

H. Chi, Q. Wu, Z. Zhou, J. Light, E. Dodwell, and Y. Ma. Unifying and optimizing data values for selection via sequential Decision-Making. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/chi26d.html. ICML 2026 Spotlight.

F. De Santis, G. Ciravegna, G. De Felice, A. Casanova, F. Giannini, M. Diligenti, J. Schneider, D. Giordano, M. Espinosa Zarlenga, and P. Barbiero. Mixture of concept bottleneck experts. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/d e-santis26a.html. ICML 2026 Spotlight.

S. Ding, H. Gao, P. Liu, X. Tian, and Y. Zhao. Disentangling latent risk pathways via bayesian hypergraph inference. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/ding26c.html. ICML 2026 Spotlight.

F. Erata, O. Paradise, T. Typaldos, T. Antonopoulos, T. Nguyen, S. Goldwasser, and R. Piskac. Learning randomized reductions. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/erata26a.html. ICML 2026 Spotlight.

B. Ferrere, N. Bousquet, F. Gamboa, J.-M. Loubes, and J. Muré. Exact functional ANOVA decomposition for categorical inputs models. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/ferrere26a.html. ICML 2026 Spotlight.

J. J. Gonzalez Ortiz, A. Gupta, C. Rinard, and D. Blalock. FlashOptim: Optimizers for memory-eficient training. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/gonzalez-ortiz26a.html. ICML 2026 Spotlight.

Z. He, H. Yang, Z. Qin, and Y. R. Fung. ClinTutor-R1: advancing scalable and robust One-to-Many alignment in clinical socratic education. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/he26as.html. ICML 2026 Spotlight.

Z. Hong, Q. Zhang, J. Sun, Z. Shang, M. Kong, X. Wang, Y. Shu, and Z. Dai. MASPOB: Bandit-Based prompt optimization for Multi-Agent systems with graph neural networks. In Proceedings ofthe 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/hong26l.html. ICML 2026 Spotlight.

S. Howard, N. Nüsken, and J. Pidstrigach. Control consistency losses for difusion bridges. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.pr ess/v306/howard26b.html. ICML 2026 Spotlight.

C. Hu, L. Zhang, Y. Lim, A. Wadhwani, A. Peters, and D. Kang. REPRO-Bench: Can agentic AI systems assess the reproducibility of social science research? In Findings of the Association for Computational Linguistics: ACL 2025, pages 23616–23626. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.findings-acl.1210. URL https://aclanthology.org/2025.findings-acl.1210/.

S. Huber, H. Unger, G. Schäfer, and J. Rehrl. Chebyshev policies and the mountain car problem: Reinforcement learning for Low-Dimensional control tasks. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/huber26a.html. ICML 2026 Spotlight.

Joint Committee for Guides in Metrology. Guide to the expression of uncertainty in measurement—part 6: Developing and using measurement models. Technical Report JCGM GUM-6:2020, BIPM, 2020. URL https://www.bipm.org/documents/20126/2071204/JCGM\_GUM\_6\_2020.pdf.

K. Keegan and L. Ruthotto. Manifold-Aware perturbations for constrained generative modeling. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://procee dings.mlr.press/v306/keegan26a.html. ICML 2026 Spotlight.

S. Kiyani, S. Noorani, G. J. Pappas, and H. Hassani. When to trust the cheap check: Weak and strong verification for reasoning. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/kiyani26a.html. ICML 2026 Spotlight.

M. Kowsher, H. Mansoor, N. J. Prottasha, O. Garibay, V. Zhu, Z. Ji, and C. Chen. LiME: lightweight mixture of experts for eficient multimodal multi-task learning. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/kowsher26a.html. ICML 2026 Spotlight.

B. La Rosa and L. H. Gilpin. Guaranteed optimal compositional explanations for neurons. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.pres s/v306/la-rosa26a.html. ICML 2026 Spotlight.

L. Li, Y. Wang, J. Yan, W. Zhang, J. Deng, H. Sun, Z. Han, and Y. Gong. From text to forecasts: Bridging modality gap with temporal evolution semantic space. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026a. URL https://proceedings.mlr.press/v306/li26hc.html. ICML 2026 Spotlight.

P. Li, X. Tang, S. Chen, Y. Cheng, R. Metoyer, T. Hua, and N. V. Chawla. Adaptive testing for LLM evaluation: A psychometric alternative to static benchmarks. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026b. URL https://proceedings.mlr.press/v306/li26fx.html. ICML 2026 Spotlight.

X. Li, D. Liu, and K. Kawaguchi. Initialization is half the battle: Generating diverse images from a guidance potential posterior. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026c. URL https://proceedings.mlr.press/v306/li26de.html. ICML 2026 Spotlight.

Z. Li, H. Hu, S. H. Lim, X. Li, F. Gao, E. Diao, Z. Ding, M. Vazirgiannis, and H. Boström. A kinetic energy perspective of flow matching. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026d. URL https://proceedings.mlr.press/v306/li26bk.html. ICML 2026 Spotlight.

H. Liu, F. A. Tjiaranata, and C. Tan. VERITAS: Towards a general-purpose replication tool for scientific research. arXiv preprint arXiv:2607.02931, 2026a. URL https://arxiv.org/abs/2607.02931.

J. Liu, J. Pei, J. Huang, C. Si, A. Qu, X. Tang, R. Lu, L. Chen, X. Bai, H. Zheng, C. Chen, Z. Chen, H. Ye, Y. Fu, Z. He, Z. Jin, Z. Zhang, S. Sun, M. Harmon, J. D. Wang, J. Zeng, J. Sun, M. Wu, B. Zhou, C. You, S. Lu, Y. Qiu, F. Lai, Y. Yuan, Y. Li, J. Hong, R. Zhu, B. Chen, A. Pentland, A. Chen, M. Chowdhury, and Z. Zhang. The last human-written paper: Agent-native research artifacts. arXiv preprint arXiv:2604.24658, 2026b. URL https://arxiv.org/abs/2604.24658v3. Version 3, revised May 19, 2026.

Y. Liu, Y. Zhao, Z. Xie, Q. Ye, J. Jiao, Y. Hu, S. Cao, and Y. Liu. Balancing understanding and generation in discrete difusion models. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026c. URL https://proceedings.mlr.press/v306/liu26gi.html. ICML 2026 Spotlight.

Z. Liu, J. Liu, H. Chen, J. Yu, Z. Guo, C. Hou, C. Gu, X. Mi, R. Zhang, K. Wu, Z. Che, J. Tang, P.-A. Heng, and S. Zhang. LaST<sub>0</sub>: latent Spatio-Temporal Chain-of-Thought for robotic Vision-Language-Action model. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026d. URL https://proceedings.mlr.press/v306/liu26ch.html. ICML 2026 Spotlight.

Y. Lu, C. Du, Q. Shi, H. Chen, J. Peng, L. Jiang, S. Zhao, and H. He. Mind-Omni: a unified Multi-Task framework for Brain-Vision-Language modeling via discrete difusion. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/l u26d.html. ICML 2026 Spotlight.

V. Macko and V. Boža. MACKO: Sparse matrix-vector multiplication for low sparsity. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v3 06/macko26a.html. ICML 2026 Spotlight.

T. Martens, L. Devos, L. Cascioli, W. Meert, H. Blockeel, and J. Davis. OC-space: a unifying perspective on verification of tree ensembles. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/martens26a.html. ICML 2026 Spotlight.

L. Messeri and M. J. Crockett. Artificial intelligence and illusions of understanding in scientific research. Nature, 627:49–58, 2024. doi: 10.1038/s41586-024-07146-0. URL https://www.nature.com/articles/ s41586-024-07146-0.

M. R. Munafò, B. A. Nosek, D. V. M. Bishop, K. S. Button, C. D. Chambers, N. Percie du Sert, U. Simonsohn, E.-J. Wagenmakers, J. J. Ware, and J. P. A. Ioannidis. A manifesto for reproducible science. Nature Human Behaviour, 1:0021, 2017. doi: 10.1038/s41562-016-0021. URL https://www.nature.com/artic les/s41562-016-0021.

P. Nagarajan, G. Warnell, and P. Stone. Deterministic implementations for reproducibility in deep reinforcement learning. arXiv preprint arXiv:1809.05676, 2018. URL https://arxiv.org/abs/1809.05676.

National Institute of Standards and Technology. Dataplot reference manual: Prediction limits. NIST Statistical Engineering Division, n.d. URL https://www.itl.nist.gov/div898/software/dataplot/refman1/ auxillar/predlimi.htm. Accessed November 4, 2026.

NIST/SEMATECH. e-handbook of statistical methods: Confidence limits for the mean. Section 1.3.5.2, n.d. URL https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm. Accessed November 4, 2026.

B. A. Nosek, G. Alter, G. C. Banks, D. Borsboom, S. D. Bowman, S. J. Breckler, S. Buck, C. D. Chambers, G. Chin, G. Christensen, M. Contestabile, A. Dafoe, E. Eich, J. Freese, R. Glennerster, D. Gorof, D. P. Green, B. Hesse, M. Humphreys, J. Ishiyama, D. Karlan, A. Kraut, A. Lupia, P. Mabry, T. A. Madon, N. Malhotra, E. Mayo-Wilson, M. McNutt, E. Miguel, E. L. Paluck, U. Simonsohn, C. Soderberg, B. A. Spellman, J. Turitto, G. VandenBos, S. Vazire, E. J. Wagenmakers, R. Wilson, and T. Yarkoni. Promoting an open research culture. Science, 348(6242):1422–1425, 2015. doi: 10.1126/science.aab2374. URL https://doi.org/10.1126/science.aab2374.

N. Oreskes. Why Trust Science? Princeton University Press, 2019. ISBN 9780691179001. URL https: //press.princeton.edu/books/hardcover/9780691179001/why-trust-science.

A. Paulus, I. Kulikov, B. Amos, R. Munos, I. Evtimov, K. Chaudhuri, and A. Zharmagambetov. Safety alignment of LMs via non-cooperative games. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/paulus26b.html. ICML 2026 Spotlight.

R. D. Peng. Reproducible research in computational science. Science, 334(6060):1226–1227, 2011. doi: 10.1126/science.1213847. URL https://doi.org/10.1126/science.1213847.

M. Petitbois, R. Portelas, and S. Lamprier. Ofline reinforcement learning of High-Quality behaviors under robust style alignment. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/petitbois26a.html. ICML 2026 Spotlight.

N. Pfaf, T. Cohn, S. Zakharov, R. Cory, and R. Tedrake. SceneSmith: agentic generation of Simulation-Ready indoor scenes. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/pfaf26a.html. ICML 2026 Spotlight.

J. Pineau, P. Vincent-Lamarre, K. Sinha, V. Larivière, A. Beygelzimer, F. d’Alché Buc, E. Fox, and H. Larochelle. Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program). Journal of Machine Learning Research, 22(164):1–20, 2021. URL https://jmlr.org/papers/v22/20-303.html.

Z. Qiu, L. Liu, A. Weller, H. Shi, and W. Liu. POET-X: memory-eficient LLM training by scaling orthogonal transformation. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/qiu26o.html. ICML 2026 Spotlight.

M. Seo, J. Baek, S. Lee, and S. J. Hwang. Paper2Code: Automating code generation from scientific papers in machine learning. In International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/41b6674c28a9b93ec8d22a53ca25bc3b-A bstract-Conference.html.

N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao. Reflexion: Language agents with verbal reinforcement learning. arXiv preprint arXiv:2303.11366, 2023. URL https://arxiv.org/abs/ 2303.11366.

Z. S. Siegel, S. Kapoor, N. Nadgir, B. Stroebl, and A. Narayanan. CORE-Bench: Fostering the credibility of published research through a computational reproducibility agent benchmark. arXiv preprint arXiv:2409.11363, 2024. URL https://arxiv.org/abs/2409.11363v2. Version 2, revised June 22, 2026.

M. Słupiński and P. Lipiński. RED-HDP-HMM: Observation-Dependent durations for bayesian nonparametric sequential models. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/slupinski26a.html. ICML 2026 Spotlight.

W. Song, W. Zhou, R. Wang, J. Yu, T. Kurihara, J. Xu, and S. Zhan. WBMM: windowed batch matrix multiplication for eficient large receptive field convolution. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/song26ae.html. ICML 2026 Spotlight.

G. Starace, O. Jafe, D. Sherburn, J. Aung, J. S. Chan, L. Maksin, R. Dias, E. Mays, B. Kinsella, W. Thompson, J. Heidecke, A. Glaese, and T. Patwardhan. PaperBench: Evaluating AI’s ability to replicate AI research. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 56843–56873. PMLR, 2025. URL https://proceedings.mlr.press/v267 /starace25a.html.

V. Stodden, M. McNutt, D. H. Bailey, E. Deelman, Y. Gil, B. Hanson, M. A. Heroux, J. P. A. Ioannidis, and M. Taufer. Enhancing reproducibility for computational methods. Science, 354(6317):1240–1241, 2016. doi: 10.1126/science.aah6168. URL https://doi.org/10.1126/science.aah6168.

Y. Sun, H. Bao, W. Wang, Z. Peng, L. Dong, S. Huang, Y. Chang, J. Wang, and F. Wei. Multimodal latent language modeling with Next-Token difusion. In Proceedings ofthe 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/sun26c.html. ICML 2026 Spotlight.

B. N. Taylor and C. E. Kuyatt. Guidelines for evaluating and expressing the uncertainty of NIST measurement results. Technical Note 1297, National Institute of Standards and Technology, 1994. URL https: //doi.org/10.6028/NIST.tn.1297.

M. Tusoni, G. Masi, A. Coletta, A. Glielmo, V. Arrigoni, and N. Bartolini. Robust causal discovery in Real-World time series with Power-Laws. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/tusoni26a.html. ICML 2026 Spotlight.

G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023. URL https://arxiv.org/abs/2305.16291.

Y. Wang, D. P. Guralnik, S. Akbari, and W. E. Dixon. Efective model pruning: Measuring the redundancy of model components. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/wang26bp.html. ICML 2026 Spotlight.

H. Wen, C. Zhang, H. He, H. Hang, and M. Lei. Progressive graph structure adjustment for homophily shift adaptation. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/wen26j.html. ICML 2026 Spotlight.

Y. Xiang, H. Yan, S. Ouyang, L. Gui, and Y. He. SciReplicate-Bench: Benchmarking LLMs in agentdriven algorithmic reproduction from research papers. arXiv preprint arXiv:2504.00255, 2025. URL https://arxiv.org/abs/2504.00255.

T. Xie, H. Luo, H. Tang, Y. Hu, J. K. Liu, Q. Ren, Y. Wang, W. X. Zhao, R. Yan, B. Su, C. Luo, and B. Guo. Controlled LLM training on spectral sphere. In Proceedings ofthe 43rd International Conference on Machine Learning (ICML), 2026a. URL https://proceedings.mlr.press/v306/xie26r.html. ICML 2026 Spotlight.

Z. Xie, Y. Wei, H. Cao, C. Zhao, C. Deng, J. Li, D. Dai, H. Gao, M. Xu, K. Yu, L. Zhao, S. Zhou, Z. Xu, Z. Zhang, W. Zeng, S. Hu, Y. Wang, J. Yuan, L. Wang, and W. Liang. mHC: Manifold-Constrained Hyper-Connections. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026b. URL https://proceedings.mlr.press/v306/xie26aa.html. ICML 2026 Spotlight.

R. Xu, Y. Chen, J. Chen, and S. Xie. Information flow reveals when to trust language models. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.pr ess/v306/xu26c.html. ICML 2026 Spotlight.

S. Xue, T. Xie, T. Hu, Z. Feng, J. Sun, K. Kawaguchi, Z. Li, and Z.-M. Ma. Any-Order GPT as masked difusion model: Decoupling formulation and architecture. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/xue26k.html. ICML 2026 Spotlight.

C. Ye, S. Yuan, S. Cooray, S. Dillmann, I. L. V. Roque, D. Baron, P. Frank, S. Martin-Alvarez, N. Koblischke, F. J. Qu, D. Yang, R. Wechsler, and I. Ciuca. ReplicationBench: Can AI agents replicate astrophysics research papers? arXiv preprint arXiv:2510.24591, 2025. URL https://arxiv.org/abs/2510.24591.

W. Yuan and G. Wang. Markov chain monte carlo without evaluating the target: an auxiliary variable approach. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/yuan26j.html. ICML 2026 Spotlight.

G. Zabërgja, R. Kamel, A. Kadra, C. M. M. Frey, and J. Grabocka. End-to-End compression for tabular foundation models. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/zabergja26a.html. ICML 2026 Spotlight.

B. Zeng, H. Li, S. Song, Y. Wang, Z. Wang, Z. He, X. Wang, and Z. Lin. PonderLM-2: pretraining LLM with latent thoughts in continuous space. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/zeng26l.html. ICML 2026 Spotlight.

F. Zhang, W. Du, H. Zhang, K. Yu, and S. Qu. HELIX: hybrid encoding with learnable identity and crossdimensional synthesis for time series imputation. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026a. URL https://proceedings.mlr.press/v306/zhang26ak.html. ICML 2026 Spotlight.

K. Zhang, W. Qin, Y. Teng, J. Yu, Y. Ma, J. Lin, L. Sun, J. Zheng, and J. Yu. Towards Sub-Second molecular docking as a structural primitive: A quantized consistency difusion framework. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026b. URL https://proceedings.mlr.press/ v306/zhang26ew.html. ICML 2026 Spotlight.

S. Zhao, F. Liu, X. Zhang, H. Chen, T. Han, J. Gong, R. Tao, P. Xiao, X. Gu, and L. Bai. Transforming weather data from pixel to latent space. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026a. URL https://proceedings.mlr.press/v306/zhao26ad.html. ICML 2026 Spotlight.

X. Zhao, Z. Sang, Y. Li, Q. Shi, W. Zhao, S. Wang, D. Zhang, X. Han, Z. Liu, and M. Sun. AutoReproduce: Automatic AI experiment reproduction with paper lineage. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 21920–21942. Association for Computational Linguistics, 2026b. doi: 10.18653/v1/2026.acl-long.1001. URL https://aclantholo gy.org/2026.acl-long.1001/.

G. Zhou, Y. Luo, Z. He, Z. Jin, X. Ge, W. Shu, D. Meng, and X. Qiu. A distributional view for visual mechanistic interpretability: KL-Minimal Soft-Constraint principle. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/zhou26ai.html. ICML 2026 Spotlight.

S. Zou, S. Wang, Z. Zhu, J. Li, C. Dong, V. Sassone, and H. Wu. SlaClip: gradient norm slacks can be indicator for adaptive clipping in DP-SGD. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://proceedings.mlr.press/v306/zou26n.html. ICML 2026 Spotlight.

## A. Benchmark and reference protocol

This appendix covers selection, reference verification, statistical rules, and paper inventories.

## A.1. Task selection and reporting units

Extraction and reporting units. Each retained task has a paper identifier, task identifier, source experiment identifier, paper-table locator, reported score, metric direction, task type, and compute type. The selected set contains 730 GPU tasks and 299 CPU-only tasks; 586 have lower-is-better metrics and 443 higher-is-better metrics. Absolute interval membership is used in both cases: outperforming the target in one direction does not automatically satisfy numerical compatibility. Of the 36 papers, 28 are classified as GPU papers and eight as CPU-only papers.

Sampling and exclusions. The supplied sampling manifest records 742 Spotlight papers, a mechanically filtered pool of 241 papers with code and non-purely-theoretical content, a random sample of 50, and 36 retained papers. Its primary exclusion categories are artifact/code availability (six), compute requirements (four), and closed-model dependence (four). The compute criteria concern papers for which more than half of the main experiments require over 100 H100 GPUhours per experiment or more than one eight-H100 node; the API criterion concerns more than half of the main experiments. These are selection criteria, not measured universal execution budgets. Tables 9 and 10 provide the retained and excluded inventories.

## A.2. Primary research-area assignments

Figure 2 uses eight topic families adapted from the ICML 2026 research topics, further divided into 22 subdomains.<sup>2</sup> We merge general machine learning with theory, and machine learning systems with optimization; the other six directions remain separate. Each paper receives one editorial assignment by its primary contribution; these are not its original conference submission labels. Generative models fall under deep learning; interpretability, verification, safety, privacy, and causality fall under trustworthy ML. For cross-cutting work, tabular compression is assigned to systems and optimization, conditioned difusion bridges to probabilistic methods, and brain modeling and survival-data generation to application-driven ML. Table 8 gives the complete primary-area mapping.

Table 8 | Paper-level assignments to the eight ICML-adapted topic families in Figure 2. IDs refer to the paper inventory in Table 9. Each paper contributes one count; no task-level weighting is applied.
<table><tr><td>Primary research area</td><td>Paper IDs</td><td>Papers</td></tr><tr><td>Deep learning</td><td>35, 71, 84, 95, 96, 116, 129, 145</td><td>8</td></tr><tr><td>Trustworthy ML</td><td>7, 20, 24, 90, 124, 135, 169, 187</td><td>8</td></tr><tr><td>General ML &amp; theory</td><td>1, 40, 66, 149, 223, 230</td><td>6</td></tr><tr><td>ML systems &amp; optimization</td><td>81, 113, 121, 136, 141, 238</td><td>6</td></tr><tr><td>Reinforcement learning &amp; robotics</td><td>2,58,134</td><td>3</td></tr><tr><td>Probabilistic methods</td><td>50,112</td><td>2</td></tr><tr><td>Application-driven ML</td><td>126, 198</td><td>2</td></tr><tr><td>Evaluation</td><td>69</td><td>1</td></tr></table>

Development and test. Five NeurIPS 2026 papers and five ICLR 2026 papers supported development of construction, reproduction, and audit workflows. They follow the same task extraction, reference, and integrity-check conventions as test papers. Their results are therefore comparable under the same scoring definition, but they are not held out from workflow development. Results and costs are reported separately in Table 3 for the 36-paper test set and Appendix D.7 for development.

## A.3. Human reference-verification protocol

For reference collection, a human verifier is a participant responsible for both reproducing an experiment and checking its execution evidence. The following requirements apply to each of the five participants in Section 2.3; this role is distinct from human validation of the downstream Auditors.

Common starting point and independent execution. Each verifier must use a separate server and a fresh checkout of the same pinned raw-repository commit, together with the paper, common task specification, and authorized supporting artifacts. Execution directories and coding-agent sessions must remain separate. Verifiers must not reuse another participant’s generated scripts or measurements, constructed refined artifacts, or downstream reproduction traces. Published scores remain visible in the paper but cannot substitute for newly computed measurements.

Experimental fidelity and documented repairs. Before aggregating the reference measurements, the five verifiers’ executed scripts, configurations, and reporting procedures were crossverified to confirm that the executions followed the paper’s specified setup and the common task definition. Verification must check the task’s dataset and split, preprocessing, model configuration, training procedure, and evaluation metric against that setup, including required hardware and numerical precision, and inspect the released implementation for consistency. A training task cannot be replaced by checkpoint evaluation; an oficial checkpoint is permitted when the task explicitly specifies evaluation-only reproduction. Necessary environment or implementation fixes must retain the intended experiment, with their rationale and diferences from the starting commit recorded. Material ambiguities must be documented rather than resolved by selecting whichever interpretation best matches the published score.

Human responsibility for agent-assisted work. Coding agents may assist with setup, debugging, and script preparation. The human verifier must inspect the scripts and configuration actually executed, including agent-generated edits, data loading, checkpoint selection, and metric extraction. Verification must connect the reported value to the computation that produced it, rather than rely on an agent’s summary, a successful exit code, or numerical proximity alone. A measurement whose execution fidelity cannot be established must not be accepted as a reference score.

Consistent reporting units and repetitions. Each participant must produce one complete observation of the specified reporting statistic per task. For a multi-seed mean, this includes all required seed-level runs and their prescribed aggregation; for best-of-�, it includes all � candidates and the specified selection rule. Constituent runs must not be pooled across participants to create the five reference observations. Seed choices, repetition counts, and stopping rules must follow the task protocol; unspecified choices must be documented and fixed before inspecting the resulting scores. Prescribed fixed seeds do not remove the requirement for separate executions.

No score-driven selection or substitution. Verifiers must not tune procedures or repeat valid measurements until they approach the paper’s score. Diagnostic runs and corrected reruns must be distinguished from reference measurements, with the reason for replacement recorded; numerical disagreement alone is not grounds for discarding a valid execution. Failed or resource-blocked attempts must retain their failure reason, without substituting cached scores, another participant’s result, or an imputed value. Numerical compatibility is assessed after execution verification using the common rules in Section 2.3.

Evidence retention and information separation. Each submitted observation must identify its task, verifier, execution, starting commit, executed scripts and edits, environment and hardware, configuration and seeds, raw outputs, and score-extraction or aggregation procedure. The verification record must state what was checked and any unresolved deviations. Reference measurements and verification guidance must remain separate from ReproBridge and downstream Reproducer inputs. These requirements support execution fidelity; the statistical assumptions for combining the five observations are specified separately in Appendix A.4.

## A.4. Error model and reference rules

We expand the formulation in Section 2.1 and give additional details of the compatibility and category rules defined in Section 2.3.

## A.4.1. Execution error and reference uncertainty

Let $\pi _ { j }$ specify the intended data, configuration, execution conditions, and reporting statistic for task $j .$ Its target is $\mu _ { j } = \mathbb { E } _ { \pi _ { j } } [ Y _ { j } ]$ , the expected result under that procedure. A paper score $p _ { j }$ has realized discrepancy $p _ { j } - \mu _ { j } ;$ neither its magnitude nor its cause is determined by the published value alone. If the reporting unit is a multi-seed mean or best-of-� statistic, $Y _ { j }$ denotes that entire statistic.

For attempts yielding a computed result, let � denote the realized execution procedure, including choices reconstructed from the supplied information. Expectations describe repeated attempts for a fixed task, agent, and artifact condition. Suppressing these indices, define

$$
m ( Z ) = \mathbb { E } [ Y \mid Z ] , \qquad D = m ( Z ) - \mu , \qquad \varepsilon = Y - m ( Z ) .\tag{A.1}
$$

Then $\mathbb { E } [ \varepsilon \mid Z ] = 0$ and $\mathbb { E } [ D \varepsilon ] = 0$ . With finite second moments, conditioning on the procedure gives

$$
\begin{array} { r l } & { \mathbb { E } [ ( Y - \mu ) ^ { 2 } ] = \mathbb { E } [ ( m ( Z ) - \mu ) ^ { 2 } ] + \mathbb { E } [ \mathrm { V a r } ( Y \mid Z ) ] } \\ & { \qquad = ( \mathbb { E } D ) ^ { 2 } + \mathrm { V a r } ( D ) + \mathbb { E } [ \varepsilon ^ { 2 } ] . } \end{array}\tag{A.2}
$$

The first two terms capture persistent mean shifts and variation in those shifts across reconstructed procedures. The last captures variation conditional on the procedure. This distinction is analogous to between-condition and within-condition efects in measurement models (Joint Committee for Guides in Metrology, 2020). For a faithful fixed procedure, $D = 0 ;$ conversely, $D = 0$ alone does not prove fidelity, since diferent procedures can have the same expected score. Nor must the residual variance be equal across raw and refined-artifact conditions. Independent repetitions of an unchanged procedure reduce the variance of its mean, not a persistent shift in its expected value.

What five reference observations estimate. Under a faithful reference model, $X _ { j i }$ are independent observations of the same reporting statistic under comparable execution conditions, with mean $\mu _ { j }$ and variance $\sigma _ { j , \mathrm { r e f } } ^ { 2 } .$ . Let ${ \bar { X } } _ { j }$ be their sample mean and $s _ { j }$ the observed sample standard deviation. Thus $\mathbb { E } [ \bar { X } _ { j } ] = \mu _ { j }$ and $\mathrm { V a r } ( \bar { X } _ { j } ) = \sigma _ { j , \mathrm { r e f } } ^ { 2 } / 5$ . Approximate normality motivates the Student-� construction. A confidence interval for $\mu _ { j }$ has half-width $t _ { 0 . 9 7 5 , 4 } s _ { j } / \sqrt { 5 }$ (NIST/SEMATECH, n.d.); a prediction interval for a new observation instead has half-width $t _ { 0 . 9 7 5 , 4 } s _ { j } \sqrt { 1 + 1 / 5 }$ (National Institute of Standards and Technology, n.d.). The latter includes both reference-center uncertainty and the new observation’s variability. Equation 1 additionally imposes the explicit 1% tolerance floor.

If a downstream measurement � is independent of these unbiased reference observations, then

$$
{ \mathbb E } [ ( Y - \bar { X } _ { j } ) ^ { 2 } ] = { \mathbb E } [ ( Y - \mu _ { j } ) ^ { 2 } ] + \frac { \sigma _ { j , \mathrm { r e f } } ^ { 2 } } { 5 } .\tag{A.3}
$$

Squared distance to the reference mean therefore also contains reference-estimation uncertainty. The paper uses the decomposition to motivate knowledge preservation, not to report empirically identified bias or variance components.

## A.4.2. Additional details of the numerical rules

Section 2.3 defines the compatibility interval (Equation 1) and reference-variation measure (Equation 2). The � multipliers for $q = 0 . 9 5$ and $q = 0 . 8 0$ are approximately 2.78 and 1.53; calculations retain the stored precision. The 1% tolerance floor is active on 197 retained tasks. There is no additional near-zero rule for � : all 21 retained zero-target tasks belong to the highervariability group. A compatibility interval can contain the reported score even when the five observations do not bracket it.

## A.5. Retained and excluded paper inventories

Table 9 | Retained papers and successful task counts from a single evaluation, raw / RRA, under the primary Codex scoring rule. The four columns are Sonnet 4.6 (S4.6), Sonnet 5 (S5), Opus 5 (O5), and Gemini 3.8 Flash (G3.8). IDs are dataset identifiers, not ranks.
<table><tr><td>ID Paper</td><td></td><td>Tasks</td><td>S4.6</td><td>S5</td><td>05</td><td>G3.8</td></tr><tr><td>1</td><td>(Li et al., 2026a)</td><td>12</td><td>9/9</td><td>9/9</td><td>11/9</td><td>12/9</td></tr><tr><td>2</td><td>(Petitbois et al., 2026)</td><td>15</td><td>2/15</td><td>15 / 15</td><td>15 / 15</td><td>10 / 15</td></tr><tr><td>7</td><td>(Ferrere et al., 2026)</td><td>82</td><td>75/76</td><td>75/81</td><td>80 / 77</td><td>62/75</td></tr><tr><td>20</td><td>(Zhou et al., 2026)</td><td>12</td><td>10 /12</td><td>2/12</td><td>9/12</td><td>11/9</td></tr><tr><td>24</td><td>(Angell et al., 2026)</td><td>3</td><td>3/3</td><td>3/2</td><td>3/3</td><td>3/2</td></tr><tr><td>35</td><td>(Li et al., 2026c)</td><td>40</td><td>9/40</td><td>28 /40</td><td>40/40</td><td>16/40</td></tr><tr><td>40</td><td>(Zhang et al., 2026a)</td><td>9</td><td>6/9</td><td>8／9</td><td>4/9</td><td>5/6</td></tr><tr><td>50</td><td>(Howard et al., 2026)</td><td>18</td><td>15 /13</td><td>14/16</td><td>15/16</td><td>11/16</td></tr><tr><td>58</td><td>(Huber et al., 2026)</td><td>22</td><td>20/20</td><td>18 / 21</td><td>20 /21</td><td>22/22</td></tr><tr><td>66</td><td>(Erata et al., 2026)</td><td>38</td><td>18 /35</td><td>16/32</td><td>24/35</td><td>9/33</td></tr><tr><td>69</td><td>(Li et al., 2026b)</td><td>82</td><td>71／82</td><td>55/82</td><td>76/82</td><td>50/75</td></tr><tr><td>71</td><td>(Liu et al., 2026c)</td><td>21</td><td>21/21</td><td>18 / 21</td><td>21/21</td><td>15 / 21</td></tr><tr><td>81</td><td>(Gonzalez Ortiz et al., 2026)</td><td>30</td><td>8/30</td><td>9/30</td><td>30/30</td><td>0/29</td></tr><tr><td>84</td><td>(Kowsher et al., 2026)</td><td>19</td><td>4/0</td><td>8/4</td><td>17 / 16</td><td>4/0</td></tr><tr><td>90</td><td>(La Rosa and Gilpin, 2026)</td><td>21</td><td>0/6</td><td>8/14</td><td>14/20</td><td>3/10</td></tr><tr><td>95</td><td>(Keegan and Ruthotto, 2026)</td><td>60</td><td>54/54</td><td>54/55</td><td>55／55</td><td>54/54</td></tr><tr><td>96</td><td>(Xue et al., 2026)</td><td>26</td><td>22／25</td><td>23/23</td><td>25/24</td><td>18 / 21</td></tr><tr><td>112</td><td>(Yuan and Wang, 2026)</td><td>19</td><td>19 / 7</td><td>15 / 0</td><td>18/ 0</td><td>12 /12</td></tr><tr><td>113</td><td>(Macko and Boža, 2026)</td><td>15</td><td>15 ／ 15</td><td>15 ／ 15</td><td>15 / 15</td><td>12 /15</td></tr><tr><td>116</td><td>(Che et al., 2026)</td><td>11</td><td>6/11</td><td>4/11</td><td>6/11</td><td>7/11</td></tr><tr><td>121</td><td>(Wang et al., 2026)</td><td>46</td><td>46/46</td><td>46/46</td><td>46/46</td><td>42／46</td></tr><tr><td>124</td><td>(Martens et al., 2026)</td><td>15</td><td>5/13</td><td>6/14</td><td>11/13</td><td>10/ 11</td></tr><tr><td>126</td><td>(Lu et al., 2026)</td><td>53</td><td>43/52</td><td>42/53</td><td>46/53</td><td>53／46</td></tr><tr><td>129</td><td>(Li et al., 2026d)</td><td>20</td><td>15 /19</td><td>20 / 19</td><td>18 /20</td><td>20 /18</td></tr><tr><td>134</td><td>(Liu et al., 2026d)</td><td>16</td><td>13 /16</td><td>12/16</td><td>13/16</td><td>8/16</td></tr><tr><td>135</td><td>(De Santis et al., 2026)</td><td>22</td><td>16/22</td><td>17 /22</td><td>22/22</td><td>10 /22</td></tr><tr><td>136</td><td>(Zabërgja et al., 2026)</td><td>45</td><td>41／44</td><td>41／44</td><td>42/44</td><td>32/44</td></tr><tr><td>141</td><td>(Qiu et al., 2026)</td><td>57</td><td>49／57</td><td>52/57</td><td>57/57</td><td>52／55</td></tr><tr><td>145</td><td>(Zeng et al., 2026)</td><td>38</td><td>37/38</td><td>38/38</td><td>38/38</td><td>37/38</td></tr><tr><td>149</td><td>(Wen et al., 2026)</td><td>15</td><td>7/15</td><td>12/15</td><td>15/15</td><td>15 / 15</td></tr></table>

<table><tr><td>ID Paper</td><td></td><td>Tasks</td><td>S4.6</td><td>S5</td><td>05 G3.8</td></tr><tr><td>169</td><td>(Tusoni et al., 2026)</td><td>23 21/23</td><td> $2 3 / 2 3$ </td><td> $2 3 / 2 3$ </td><td>21/23</td></tr><tr><td>187</td><td>(Zou et al., 2026)</td><td>30  $9 / 2 4$ </td><td> $9 / 3 0$ </td><td> $1 2 / 3 0$ </td><td>9/30</td></tr><tr><td>198</td><td>(Brockschmidt et al., 2026)</td><td>14</td><td> $1 2 / 1 4$   $1 4 / 1 4$ </td><td> $1 1 / 1 4$ </td><td>13 / 14</td></tr><tr><td>223</td><td>(Chi et al., 2026)</td><td>13  $9 / 1 3$ </td><td> $9 / 1 3$ </td><td> $1 3 / 1 3$ </td><td>13 / 13</td></tr><tr><td>230</td><td>(Kiyani et al., 2026)</td><td>9</td><td>9/9</td><td>9/9 9/9</td><td>0/9</td></tr><tr><td>238</td><td>(Song et al., 2026)</td><td>58 23 /57</td><td>40／57</td><td>44／57</td><td>25/57</td></tr></table>

Table 10 | Excluded papers in the supplied 50-paper manifest. C1: code/artifact availability; C2: node capacity; C3: per-experiment compute; C4: closed-model API dependence. The first listed code is the primary reason. Selection criteria are described in Appendix A.
<table><tr><td>ID</td><td>Paper Criteria</td></tr><tr><td>133</td><td>(Ding et al., 2026) C1</td></tr><tr><td>156</td><td>(Słupiński and Lipiński, 2026) C1</td></tr><tr><td>44</td><td>(Xu et al., 2026) C1</td></tr><tr><td>27 (Sun et al., 2026)</td><td>C1/C3</td></tr><tr><td>218</td><td>(Zhao et al., 2026a) C1/C3</td></tr><tr><td>219</td><td>(Zhang et al., 2026b) C1</td></tr><tr><td>235</td><td>(Xie et al., 2026b) C3/C2/C1</td></tr><tr><td>179 (Paulus et al., 2026)</td><td>C3/C2/C4</td></tr><tr><td>15</td><td>(Xie et al., 2026a) C3</td></tr><tr><td>201</td><td>(Chen et al., 2026) C3</td></tr><tr><td>101</td><td>(Agrawal et al., 2026) C4</td></tr><tr><td>39</td><td>(Hong et al., 2026) C4</td></tr><tr><td>43</td><td>(He et al., 2026) C4</td></tr><tr><td>177 (Pfaff et al., 2026)</td><td>C4</td></tr></table>

## B. Artifact design and construction

This appendix expands the construction contracts in Section 3.1 and Table 2 with full role instructions and the artifact’s document structure.

## B.1. Construction-skill specifications

ReproBridge implements artifact construction through role-specific skills, each with instructions, reference protocols, tools, and required outputs. The specifications below expand the construction contracts and coordination described in Section 3.1 and Table 2. Separate downstream Reproducer and Auditor workflows evaluate the artifacts; human reference collection and human audit validation belong to benchmark development and evaluation, respectively.

Intake. Read the complete paper, including appendices, before planning. Preserve the selected task list and pinned raw commit. Map each task to its table entry, survey the implementation, check required artifacts, and record missing details. Do not silently substitute an unavailable repository revision or redefine a task to fit the available code.

Planner. For each task, specify the measurement, data, procedure, aggregation, and stated repetitions, with references to supporting paper passages or code locations. Group tasks that share an execution. Record the adopted resolution of each ambiguity before the dependent run; prepare an executable plan and launcher. The Planner does not execute experiments. When paper and code conflict, strictly follow the paper’s stated setup, flag the conflicting code and afected tasks, and record the required implementation corrections before the dependent run.

Builder. Resolve and record dependencies, provide reproducible acquisition of required data and checkpoints, and test the resulting environment. The environment described by the deliverable should be the one used for construction measurements, not an untested reconstruction assembled afterward. Environment work must not silently change the experimental plan.

Construction Reproducer. Execute planned experiments in the prepared environment. Link values to actual output files and record commands, resources, and failures. Distinguish diagnostic runs from scored repetitions; justify changes and repetition extensions. Do not silently edit a frozen plan or select favorable repetitions in place of the specified reporting statistic. Every task needs a supported measurement or an explicit, evidenced reason why none was produced.

Distiller. Write the guide and change record from stored task specifications, execution evidence, repairs, and choices. Explain the reporting statistic, observed variability, resource requirements, pitfalls, and uncovered scope. Preserve uncertainty where the record does not resolve it. Historical measurements belong in separate result records, not in instructions substituting for new measurements.

Packager. Assemble the code, environment, launchers, and documentation actually used. Check portable paths and verify the handof from a clean shell, including an inexpensive real repetition. Compare the output structure and extraction route with the record. Do not invent replacement values or treat a successful smoke test as completion of a scientific experiment.

Reviewer. Check the plan before scored execution, inspect evidence and recorded changes during reproduction, and test the final handof as an unfamiliar user. Inspect the underlying files and outputs directly instead of accepting another role’s summary. Produce localized findings with the responsible role; do not repair and approve the same work in one context.

Alignment extension. The Aligner skill extracts paper-side requirements before inspecting their implementation, then records contradictions separately from missing code and unspecified details. It does not execute experiments, edit plans, or infer a measured performance efect from a plausible mechanism. Appendix D.6 describes the separate analysis.

## B.2. Four-component handof and document structure

The normalized refined artifact has seven top-level entries:

<table><tr><td>Path</td><td>Role</td></tr><tr><td>src/</td><td>Derived source code</td></tr><tr><td>Dockerfile</td><td>Preferred container setup</td></tr><tr><td>setup/</td><td>Environment lock, installation, and data scripts</td></tr><tr><td>experiments/</td><td>Launchers and shared path setup</td></tr><tr><td>reproduction_guideline.md</td><td>Reproduction guide</td></tr><tr><td>change_log.md</td><td>Change record and open choices</td></tr><tr><td>verify.sh</td><td>Package verification entry point</td></tr></table>

```markdown
reproduction_guideline.md 12 sections
# Reproducing Table 1 of SCIQL
## What this is
- Table 1: style alignment for CBC, SCBC, and SCIQL.
- Released Circle2D and HalfCheetah datasets; code commit `5389c6e`.
## Hardware
- One GPU per repetition; this guide used NVIDIA H100 80 GB.
## Software
- Python 3.10; JAX 0.5.2; NumPy 2.2.3; PyTorch 2.5.1+cpu.
- JAX runs on the GPU; PyTorch is used for logging.
## Environment setup
```bash
bash setup/01_environment.sh
source .venv/bin/activate
DATA_ROOT=/data bash setup/02_data.sh
## Artifacts
- `datasets.zip`: SHA-256 checked; 1,000 episodes per Circle2D dataset.
- No released checkpoints; each experiment trains from scratch.
## Experiments
G15: SCIQL × circle2d-inplace-v0; one seed, all six style criteria.
`bash
DATA_ROOT=/data OUTPUT_ROOT=/outputs bash experiments/run_G15.sh 0
## Repetitions and variance
- Run index = seed; the paper reports a mean over five seeds.
- Use distinct seeds and retain every run, including low-scoring runs.
## Outputs and metrics
`$OUTPUT_ROOT/G15/run0/metrics.json`, key: `alignment_x100`.
- Unweighted mean over labels, then criteria, multiplied by 100.
## Paths
`DATA_ROOT`: datasets. `OUTPUT_ROOT`: new measurements and logs.fi
`src/datasets/traj2d` links into `$DATA_ROOT/episodes/traj2d`.
## Ambiguities and unspeci ed details
A1: use unweighted aggregation; checked against Appendix Table 3.
A3: released episodes refer to the old package name `style_bench`.
- Adopt aliases to `sciql.core.data` in `src/tools/sciql_launch.py`;
retain the released episode classes.
## Known pitfalls
Symptom: exit 0 and a plausible CSV, despite loading no episodes.
Cause: import errors are caught and skipped by the loader.
Fix: use the compatibility launcher and keep `--min-episodes 1000`.
- Keep smoke outputs separate from full runs (`run<i>.smoke/`).
## What is not covered
- HumEnv: the released datasets and integration code are missing.
- Experiments outside Table 1.
```  
Figure 8 | Inside a reproduction guide. Condensed SCIQL entries retain all twelve section headings in their original order. The document records hardware and software prerequisites, data acquisition, launch commands, repetitions, output fields and aggregation rules, path conventions, ambiguities, known pitfalls, and unsupported scope. The example links a renamed-package issue to a compatibility launcher and a minimum-episode check, preserving both an executable route and the reason for it. Commands and output keys specify how to produce and extract new measurements.

The archive additionally separates reproduction-result pages and paper–code alignment records.   
The manuscript’s “change record” refers to change\_log.md.

The normalized guide has twelve sections: purpose, hardware, software, environment setup, artifacts, experiments, repetitions and variance, expected output structure, paths, ambiguities and unspecified details, pitfalls, and uncovered scope. Its environment section distinguishes Docker from installation scripts. The ambiguity table records ID, type, location, gap, adopted resolution and reason, sensitivity, and afected tasks. The change record uses six sections: changes required to run, changes required for correctness, convenience changes, deliberately unchanged behavior, pitfalls, and open choices.

Across the 36 packaged repositories, normalized documents contain 932 ambiguity-table entries.   
The corpus contains 482 launcher scripts, including those outside the retained scoring scope.

Figure 8 expands SCIQL’s twelve-section guide with representative instructions and decisions.

## C. Evaluation protocol and audit validation

The evaluation uses fresh downstream sessions and a separate integrity audit. The following settings, information boundaries, and validation protocol concern this evaluation, not artifact construction.

Test-set diagnostic statistics here and in Appendix D report a single evaluation; Table 3 and the cost summary in Table 21 use means over three evaluations.

## C.1. Model configurations and information boundaries

Table 11 | Model identifiers and harness settings for main-experiment construction and downstream evaluation. “Max” and “high” are tool-specific efort settings, not a shared compute scale.
<table><tr><td>Role / label</td><td>Model identifier</td><td>Harness</td><td>Effort</td></tr><tr><td>Construction</td><td>claude-opus-5[1m]</td><td>Claude Code</td><td>max</td></tr><tr><td>Sonnet 4.6</td><td>claude-sonnet-4-6[1m]</td><td>Claude Code</td><td>max</td></tr><tr><td>Sonnet 5</td><td>claude-sonnet-5[1m]</td><td>Claude Code</td><td>max</td></tr><tr><td>Opus 5</td><td>claude-opus-5[1m]</td><td>Claude Code</td><td>max</td></tr><tr><td>Gemini 3.8 Flash</td><td>gemini-3.8-flash</td><td>Gemini CLI</td><td>high</td></tr><tr><td>Primary Auditor</td><td>gpt-5.6-sol</td><td>Codex CLI</td><td>high</td></tr><tr><td>Comparison Auditor</td><td>claude-opus-5[1m]</td><td>Claude Code</td><td>max</td></tr></table>

Visibility. Construction permits the paper, raw repository, required artifacts, and the agents’ own execution feedback, while excluding human reference guidance. Reproducers receive the paper and published targets, the selected task specification, and their assigned repository. Reference categories and hidden score fields are removed from session input; the external evaluator retains the exported thresholds. Instructions prohibit using other runs or the alternative repository and require submitted measurements to originate in the current run. Some guides describe historical result directions or magnitudes. The Auditor checks whether submitted measurements come from new executions.

Execution records. The downstream Reproducer workflow stores execution IDs, output files and locators, task evidence, and code diferences. Where recorded, script/file hashes and ordering checks provide additional provenance evidence. Resource accounting distinguishes wrappedexecution device-hours from total reserved-lane device-hours. Both raw-code and refined-artifact conditions use the same 24-hour wall-clock limit per paper. This session limit applies to downstream evaluation; the experiment-level compute criterion governs benchmark eligibility.

## C.2. Independent reproduction skill

A fresh downstream Reproducer session uses the repro-run execution skill. It receives the assigned repository, paper, and selected tasks, then plans experiment scripts, prepares permitted inputs, schedules executions, extracts values, and reports code changes and evidence locators. The session retains responsibility for the final evidence, including work delegated by the harness.

Programmatic support freezes inputs, schedules work under the resource limits, links values to files, and checks evidence consistency. Scientific assessment combines these records with inspection of the paper and executed code.

## C.3. Additional audit coordination

Section 4.1 defines the Finder–Refuter procedure and invalidation criteria; Table 7 compares the two Auditors on the same numerically matching outcomes. Codex supplies the primary success labels, while Claude supplies a sensitivity comparison. For either Auditor, success is the number of numerical matches minus invalidations, divided by 1,029 tasks.

Unresolved screening leads require recorded dispositions, and final task-level verdicts are assembled from structured findings. Separate sessions and withheld prior verdicts reduce direct self-validation and cross-auditor anchoring. The human verification protocol below assesses both Auditors against the same evidence requirements.

## C.4. Two-person human verification protocol

Two human auditors cross-verified selected Reproducer outcomes. The protocol below defines this review’s evidence requirements and cross-checking responsibilities. It evaluates existing execution records; human reference collection (Appendix A.3) generates numerical reference observations.

Review scope and unit. Both reviewers must examine each selected task–condition outcome, including outcomes not flagged by either automated Auditor. Reviewing only automated disagreements or invalid flags would miss violations accepted by both systems. Shared executions may support several judgments, but each alleged violation must be linked to the afected task scores; a problem in one experiment does not automatically invalidate every result in the paper.

Trace the measurement to primary evidence. Each reviewer must connect the submitted score to the executed command, scripts and configuration, input artifacts, raw outputs, and extraction or aggregation procedure. The supplied code and subsequent changes must be inspected to establish the computation underlying the scored result. Review must also check for copied values, substituted measurements, result-dependent selection, incorrect reductions, and altered execution records. Automated explanations can identify claims to inspect, but are not evidence by themselves.

Apply the same invalidation criteria. Reviewers must apply the invalidation criteria in Section 4.1, considering permitted reporting rules and environment or resource efects before confirming a violation. Judgments concern the executed procedure and its evidence rather than inferred intent.

Cross-check judgments and resolve disagreements. Each reviewer must record a judgment and its supporting evidence before the pair compares assessments. Each must then check the other’s task mapping, supporting files and outputs, and interpretation, including counterevidence. Disagreements must be resolved by returning to the execution record and documenting why a judgment is retained or revised, not by treating either automated Auditor as a tie-breaker. Both reviewers must endorse the final human label and its rationale; an unresolved case must not be presented as an agreed judgment.

Retain the review record and compare verdicts. The record must retain task and condition identifiers, the associated execution, both initial judgments, evidence locators, any disagreement and its resolution, and the final cross-verified label. Correctness of an automated verdict must be evaluated against this human label, separately from whether its explanation identifies the correct mechanism. Count each task–condition outcome once when computing detection statistics, even when several outcomes share execution evidence.

## C.4.1. Human-verified subset

Eight papers are randomly selected from the test set for two-person cross-verification: Exact-FANOVA (82 tasks), Learning Randomized Reductions (38), ATLAS (82), MindOmni (53), POET-X (57), SlaClip (30), Weak-and-Strong Verification (9), and WBMM (58). Review covers all 409 tasks from these papers in both conditions: 409 raw-codebase and 409 refined-artifact outcomes, totaling 818 task–condition outcomes. It identifies 39 invalid outcomes: 33 with raw code and six with refined artifacts. The remaining 779 outcomes are valid under this review, comprising 376 raw-codebase and 403 refined-artifact outcomes. Invalid is the positive class, and uncertain audit judgments are negative. Precision divides correct invalid detections by all invalid predictions; recall divides them by all human-confirmed invalid outcomes. Random selection occurs at the paper level; task outcomes within a paper can share execution evidence and should not be treated as independent draws.

Table 12 | Audit predictions checked against human verification on 409 outcomes per condition (818 overall). Flagged = TP + FP; TP and FP are correct and incorrect invalid flags, and FN counts missed invalid outcomes. Human-confirmed invalid counts are 33 for Raw and six for RRA. Precision = TP/(TP + FP); recall = TP/(TP + FN).
<table><tr><td>Scope</td><td>Auditor</td><td>Flagged</td><td>TP</td><td>FP</td><td>FN</td><td>Precision (%)</td><td>Recall (%)</td></tr><tr><td>All</td><td>Codex</td><td>35</td><td>31</td><td>4</td><td>8</td><td>88.6</td><td>79.5</td></tr><tr><td>All</td><td>Claude</td><td>28</td><td>24</td><td>4</td><td>15</td><td>85.7</td><td>61.5</td></tr><tr><td>Raw</td><td>Codex</td><td>29</td><td>26</td><td>3</td><td>7</td><td>89.7</td><td>78.8</td></tr><tr><td>Raw</td><td>Claude</td><td>24</td><td>20</td><td>4</td><td>13</td><td>83.3</td><td>60.6</td></tr><tr><td>RRA</td><td>Codex</td><td>6</td><td>5</td><td>1</td><td>1</td><td>83.3</td><td>83.3</td></tr><tr><td>RRA</td><td>Claude</td><td>4</td><td>4</td><td>0</td><td>2</td><td>100.0</td><td>66.7</td></tr></table>

Codex misses eight of 39 human-confirmed invalid outcomes (20.5%); Claude misses 15 (38.5%). Each incorrectly flags four of 779 human-verified valid outcomes (0.5%). Codex detects a larger fraction of invalid outcomes in each condition, while Claude makes no false positive on the refined artifact subset (Table 12).

## C.5. Qualitative audit cases

These cases illustrate measurement validity and evidence integrity through execution records.

Timing reports versus executed measurements: WBMM. For refined-artifact tasks T028, T033, and T038, execution records show epoch times of approximately 32.5, 53, and 92 seconds.

Submitted JSON and local logs instead contain 206, 335, and 579 seconds, obtained by projecting subset timings to a 2,502-iteration epoch. Reviewed records show file edits 8–17 minutes after process exit and discrepancies with captured stdout. The case review notes that measured durations also satisfy the numerical intervals; the issue is reporting projections as measurements, not missing a target.

Authentic scores do not authenticate the accompanying record. In Weak-and-Strong Verification, the case review finds genuine computed scores alongside edited audit records, including a FAIL-to-PASS change and backdated file timestamps. The case demonstrates why a valid numerical computation does not settle whether the submitted evidence faithfully represents the execution.

## D. Extended results and case evidence

## D.1. Reference-variability strata

![](images/bab12a557d02671d91aa2998dac8501ee79d2cfcfc3ab7afd907707f332e839b.jpg)  
Figure 9 | Gains across reference-variability levels. Blue circles show stable references (Cat. 1, 701 tasks); amber squares show higher-variability references (Cat. 2, 328 tasks), separated by the 10% relativevariation threshold. Markers and right-hand labels give refined-artifact (RRA) minus Raw success in percentage points in the first evaluation.

Table 13 | Success by reference variability in the first evaluation. Stable (Cat. 1, � = 701) and highervariability (Cat. $2 , n = 3 2 8 )$ tasks use the same fixed compatibility thresholds in Raw and refined-artifact (RRA) conditions. Gain is RRA minus Raw in percentage points, calculated before rounding the displayed rates.
<table><tr><td>Model</td><td>Cat.</td><td>Raw (%)</td><td>RRA (%)</td><td>Gain (pp)</td></tr><tr><td>Sonnet 4.6</td><td>1</td><td>71.3</td><td>93.6</td><td>+22.3</td></tr><tr><td>Sonnet 4.6</td><td>2</td><td>73.8</td><td>88.1</td><td>+14.3</td></tr><tr><td>Sonnet 5</td><td>1</td><td>75.6</td><td>95.4</td><td>+19.8</td></tr><tr><td>Sonnet 5</td><td>2</td><td>78.4</td><td>89.3</td><td>+11.0</td></tr><tr><td>Opus 5</td><td>1</td><td>88.9</td><td>96.9</td><td>+8.0</td></tr><tr><td>Opus 5</td><td>2</td><td>89.9</td><td>92.1</td><td>+2.1</td></tr><tr><td>Gemini 3.8 Flash</td><td>1</td><td>68.3</td><td>91.6</td><td>+23.3</td></tr><tr><td>Gemini 3.8 Flash</td><td>2</td><td>66.2</td><td>88.4</td><td>+22.3</td></tr></table>

Both groups improve for every model, with larger gains in Cat. 1 for each model (Figure 9). Gains span 8.0 to 23.3 pp on stable tasks and 2.1 to 22.3 pp on higher-variability tasks. Wider compatibility intervals can accommodate larger numerical diferences, and the task composition difers between strata; the contrast between groups is therefore descriptive.

## D.2. Outcome transitions and execution diagnostics

Table 14 | First-evaluation success counts and transitions, with 1,029 tasks per model. Recovery: Raw failure to RRA success; reversal: Raw success to RRA failure. Net is recoveries minus reversals.
<table><tr><td>Model</td><td>Raw</td><td></td><td>RRA Recoveries Reversals</td><td></td><td>Net</td></tr><tr><td>Sonnet 4.6</td><td>742</td><td>945</td><td>234</td><td>31</td><td>203</td></tr><tr><td>Sonnet 5</td><td>787</td><td>962</td><td>208</td><td>33</td><td>175</td></tr><tr><td>Opus 5</td><td>918</td><td>981</td><td>100</td><td>37</td><td>63</td></tr><tr><td>Gemini 3.8 Flash</td><td>696</td><td>932</td><td>271</td><td>35</td><td>236</td></tr><tr><td>Total</td><td>3,143</td><td>3,820</td><td>813</td><td>136</td><td>677</td></tr></table>

Table 15 | Downstream repair and time-to-execution diagnostics. Patch counts are papers with a Reproducer code dif out of 36, not independent defects. Time is the median to the first scored execution, not end-to-end runtime.
<table><tr><td colspan="2"></td><td colspan="2">Papers with patches First scored execution (h)</td></tr><tr><td>Model</td><td>Raw</td><td>RRA Raw</td><td>RRA</td></tr><tr><td>Sonnet 4.6</td><td>26</td><td>11</td><td>0.50 0.27</td></tr><tr><td>Sonnet 5</td><td>30</td><td>14</td><td>0.90 0.53</td></tr><tr><td>Opus 5</td><td>24</td><td>16</td><td>0.54 0.32</td></tr><tr><td>Gemini 3.8 Flash</td><td>33</td><td>13</td><td>0.73 0.26</td></tr></table>

An agent-assisted case review assigns primary explanations across 101 model–paper pairs (Table 16). Analysis agents inspect paired runs and refined-artifact documents; a consolidation agent reopens selected examples, and an independent reviewing agent challenges them. The categories describe explanations grounded in execution records. Their counts allocate the reviewed shares proportionally across recoveries; they are not independent causal measurements.

Protocol covers clarified settings and reporting rules; Launcher covers entry points and experiment-to-result mapping. Env covers com-

Table 16 | Explanations for recovered outcomes. Estimated counts allocate case-review shares across 813 recoveries, with rounding that preserves the total.
<table><tr><td>Explanation</td><td>Est. outcomes</td><td>Share (%)</td></tr><tr><td>Protocol</td><td>210</td><td>25.8</td></tr><tr><td>Launcher</td><td>76</td><td>9.4</td></tr><tr><td>Env</td><td>87</td><td>10.7</td></tr><tr><td>Code</td><td>99</td><td>12.2</td></tr><tr><td>Artifact</td><td>79</td><td>9.7</td></tr><tr><td>Integrity</td><td>217</td><td>26.7</td></tr><tr><td>Budget</td><td>37</td><td>4.5</td></tr><tr><td>Other</td><td>8</td><td>1.0</td></tr></table>

patible dependencies, and Code covers repairs or missing implementations. Artifact covers data/checkpoint acquisition and paths. Integrity concerns replacing measurement shortcuts with valid execution; audit-invalid names a scored outcome. Budget covers resource planning and scheduling; Other covers remaining explanations.

## D.3. Failure modes and concentrated regressions

Failures shared by both handofs. Qualitative review of task–model pairs that fail with both raw code and refined artifacts identifies paper–code setup mismatches, typographical errors in the paper, and substantial experimental variability. A usable artifact can still preserve a setting that conflicts with the intended experiment or cannot resolve an erroneous paper statement. Even under a fixed procedure, a new observation can fall outside the reference compatibility interval.

Reversals after refinement. A reversal is a task–model pair that succeeds with raw code but fails with the refined artifact; it difers from a net decrease over an entire paper. Table 17 partitions all 136 reversals by their refined-artifact outcome. Counts span four models, so a task can appear in several pairs. AuxiliaryMCMC contributes 50, Manifold-Aware Perturbations 19, LiME 14, and ExactFANOVA 10. These four papers account for 93/136 reversals (68.4%). A paper can contain both recovered and lost tasks: Gemini’s AuxiliaryMCMC total remains 12 successes in each condition despite five reversals ofset by five recoveries.

AuxiliaryMCMC: numerical fragility interacts with a finite budget. The Opus 5 refined-artifact Gaussian sweep for AuxiliaryMCMC (Yuan and Wang, 2026) targets 100 rounds at each of three acceptance rates. Its measured workload exceeds the available clock at the chosen concurrency, and the agent executes only 55 rounds per rate. The primary audit invalidates numerical matches afected by this altered measurement proce-

Table 17 | Failure states after a reversal. Counts and shares of all 136 Raw-success-to-RRA-failure outcomes across four models.
<table><tr><td>RRA outcome</td><td>Reversals</td><td>Share (%)</td></tr><tr><td>Numerical mismatch</td><td>84</td><td>61.8</td></tr><tr><td>Execution blocked</td><td>10</td><td>7.4</td></tr><tr><td>Resource/access limit</td><td>31</td><td>22.8</td></tr><tr><td>Audit-invalid match</td><td>11</td><td>8.1</td></tr></table>

dure. Separately, one Poisson-MALA chain remains constant with zero acceptance. Its undefined efective sample size propagates through the evaluator’s ordinary mean, leaving three measurements unavailable. The Reproducer declines to redraw only the failed chain, which would select a repetition based on its outcome. Efective sample size per second also varies with host load and concurrency.

The raw attempt also runs fewer than 100 rounds in part of its sweep. These records illustrate runtime constraints, numerical edge cases, and deviations from the prescribed measurement procedure.

LiME: setup success is not experiment completion. In Sonnet 5’s refined-artifact attempt on LiME (Kowsher et al., 2026), data staging and environment preparation take approximately 83 minutes, but the full schedule includes 5,949 training steps followed by evaluation over 47 splits. The attempted run’s observed throughput projects 7 to 92 wall-clock hours per variant, competing within the paper’s 24-hour allocation. Model-construction parameter counts are available before training finishes, whereas accuracy measurements require completed training and evaluation. The session reaches the former but not the latter. Across models, 13 of LiME’s 14 reversals are resource/access failures and one is a numerical mismatch.

Interpreting residual numerical mismatches. The 84 numerical reversals span several possible mechanisms: stochastic training, timing conditions, changed execution choices, and unresolved paper–code disagreements. Construction may also introduce an incorrect setting, metric interpretation, or repair that later agents inherit. Diagnosing a particular reversal requires tracing its executed configuration and measurement procedure. The Manifold-Aware Perturbations guide, for example, flags a learning-rate discrepancy between the paper and released code for further inspection (Appendix D.5).

## D.4. Component ablations and reconstruction efort

We randomly select 12 papers from the 36-paper test set for the Sonnet 5 ablation. The same 287 tasks are evaluated across all seven conditions in Table 6; Table 18 lists the selected papers and their experiment and task counts.

For ordinary outputs, Opus 5 attempts reproduction with the raw codebase. Its execution directory, with result files removed but code, scripts, and logs retained, becomes the input codebase for a fresh Sonnet 5 session; no standardized documentation is added. This run and ReproBridge construction are separate upstream attempts with comparable resource use. ReproBridge receives the four-component delivery protocol from the start; the comparison evaluates the two workflows’ handofs.

Table 18 | Randomly selected papers for component ablations. IDs match Table 9. Tasks are scored measurements associated with the listed experiments; CPU-only tasks are a subset of Tasks. The same 12 papers and 287 tasks are used in every condition.
<table><tr><td>ID</td><td>Paper</td><td>Experiments</td><td>Tasks</td><td>CPU-only tasks</td></tr><tr><td>1</td><td>TESS</td><td>4</td><td>12</td><td>0</td></tr><tr><td>20</td><td>EnergyDPS</td><td>3</td><td>12</td><td>6</td></tr><tr><td>50</td><td>ControlConsistencyBridges</td><td>13</td><td>18</td><td>0</td></tr><tr><td>112</td><td>AuxiliaryMCMC</td><td>2</td><td>19</td><td>19</td></tr><tr><td>129</td><td>KPEKTS</td><td>5</td><td>20</td><td>0</td></tr><tr><td>141</td><td>POET-X</td><td>57</td><td>57</td><td>0</td></tr><tr><td>149</td><td>PSAHS</td><td>15</td><td>15</td><td>0</td></tr><tr><td>169</td><td>PLACy</td><td>13</td><td>23</td><td>23</td></tr><tr><td>187</td><td>SlaClip</td><td>30</td><td>30</td><td>0</td></tr><tr><td>198</td><td>SurvDiff</td><td>3</td><td>14</td><td>0</td></tr><tr><td>230</td><td>Weak-and-Strong Verification</td><td>1</td><td>9</td><td>9</td></tr><tr><td>238</td><td>WBMM</td><td>9</td><td>58</td><td>0</td></tr><tr><td colspan="2">Total</td><td>155</td><td>287</td><td>57</td></tr></table>

Section 4.5 defines the component removals and explains why descriptions of removed components are also deleted. The following case shows the reconstruction work that can return after removing the prepared environment.

WBMM without the prepared environment. For WBMM (Song et al., 2026), the environment ablation retains experiment code and commands but removes the compatible H100 dependency route. Sonnet 5 reconstructs a Python 3.10 / PyTorch 2.0 / CUDA 11.8 stack with compatible vision libraries. A supplied binary lacks an operator’s H100 kernels, requiring a source rebuild before COCO evaluation. The agent also fixes an installation-tool compatibility issue and moves data of a quota-constrained shared filesystem. No paper-source patch is required, yet substantial environment work returns downstream. The trace produces measurements for all 58 selected tasks.

Timing measurements also depend on GPU concurrency and storage behavior. In this run, reuse of a smaller local image pool changes cache behavior alongside the reconstructed environment.

## D.5. Worked reproduction examples

These examples expand the paired-run case evidence in Section 4.4, retaining exact configurations, task identifiers, and measurements. They illustrate information preserved in the refined artifact and corresponding recorded outcomes.

Recovering omitted configuration: HELIX. For HELIX (Zhang et al., 2026a), Opus 5 restores the paper’s architecture on raw code but retains a shipped batch size of 32 and learning rate of approximately $6 . 2 9 \times 1 0 ^ { - 4 }$ . The guide’s ambiguity entry A6 identifies revision 9760b347, whose architecture matches the paper configuration, as the source for batch size 8 and learning rate $5 . 7 5 \times 1 0 ^ { - 4 }$ . The –paper\_config launcher makes this choice executable. The recorded MAE changes from 0.15265 to 0.12986 for T004 and from 0.50527 to 0.41160 for T006; the respective compatibility intervals are [0.12524, 0.13448] and [0.38442, 0.43878]. The artifact records the repository history and architectural match as the rationale for adopting these settings.

An environment route that transfers: WBMM. For six COCO tasks (T063–T068) in WBMM (Song et al., 2026), Sonnet 4.6’s raw attempt fails to install a compatible mmcv-full stack for H100. The refined artifact specifies matching PyTorch/MMDetection dependencies, the Ninja build prerequisite, and a source build for the target architecture. The downstream refined-artifact session makes no code patch and executes all six tasks. Its two box AP values, 51.61 and 52.78, agree numerically with reported targets 51.6 and 52.8; mask AP is 44.78/45.64 against 44.8/45.6. The artifact preserves an executable dependency route for later reuse.

Fixing validation labels: Efective Model Pruning. The released TinyImageNet loader uses ImageFolder on a validation directory containing only an images subdirectory. It therefore assigns all 10,000 validation images to one class. During construction, ReproBridge adds a preparation script that places images into class directories using val\_annotations.txt, verifies 200 classes and 10,000 images, and writes a preparation receipt required by the launcher. In Gemini 3.8 Flash’s paired runs, the ResNet-18 validation loss is 8.898 with raw code and 2.288 with the refined artifact, against the paper’s 2.303. This records a concrete repair and its downstream reuse.

Organizing measurement code for reuse: ETTFS. The raw Opus 5 attempt searches the released ETTFS (Che et al., 2026) source for the metrics in Tables 2 and 3, finds no producing implementation, and leaves the inference-time measurement unexecuted. During construction, ReproBridge adds Code/profile\_ttfs.py, documents the metric definitions and output fields, and supplies a launcher that trains a checkpoint before profiling it. The later Opus 5 session uses this explicit route and reports 2.0652 inference time steps against the paper’s 2.03. The artifact localizes code and resolved decisions that would otherwise require renewed investigation.

A measurement is not just a model execution: Mixture of Concept Bottleneck Experts. For four-expert settings in Mixture of Concept Bottleneck Experts (De Santis et al., 2026), the paper does not specify how multiple expression distances should be reduced, and the repository provides no Table 1 collector. The raw Sonnet 5 attempt constructs a matching between learned and reference expressions but reports the sum of tree-edit distances. For two tasks this produces 60 and 188, whereas the per-expression means are 15 and 47. The refined artifact resolves the ambiguity by documenting the denominator and implementing the corresponding collector; those downstream values are 15 and 47. The case locates the discrepancy in the underspecified reporting statistic.

Preserving the experimental aggregation: SCIQL. The released SCIQL (Petitbois et al., 2026) code produces run outputs but does not assemble the Table 1 statistic, leaving the mapping from logged fields and experiment settings to reported values incomplete. The raw Sonnet 4.6 attempt uses one default criterion, one seed, and an incorrect output column for a table statistic that averages across six criteria and five seeds. The refined artifact’s launcher and collector make this mapping explicit by specifying the sweep and the correct alignment field. One representative measurement changes from 19.70 to 55.66 against a paper value of 58.6. Criterion selection, output mapping, and repetition change together in this case, illustrating preservation of the complete measurement protocol.

An optimizer release is not a complete experiment: FlashOptim. FlashOptim (Gonzalez Ortiz et al., 2026) supplies optimizer code but not the GPT-2 training and evaluation harness for the selected table. Gemini’s raw attempt leaves 18 GPT-2 outcomes unexecuted. The refined artifact adds a training/evaluation harness, data acquisition, output mapping, and launchers; all 18 pass in its recorded downstream attempt. The construction report lists 6,346 added lines of experiment support.

A localized repair: POET-X. The defect occurs in examples/peft\_pretraining/traini ng\_utils.py, line 222. Full batches call collate\_fn, whereas the trailing partial batch is returned as a list. The evaluator calls batch.items(), so it fails when the final batch is reached. The patch changes the tail to yield collate\_fn(batch). The record traces the sole caller to validation and describes CPU checks covering empty, partial, complete, and over-full batch counts, without changing the example count. This example illustrates how a later reproducer can recover both the fix and its experimental scope.

Flagging a setup discrepancy: Manifold-Aware Perturbations. The guide’s ambiguity entry A1 contrasts the paper’s stated learning rate of 10<sup>−4</sup> with the released driver, training scripts, and 171 committed manifests using 10<sup>−3</sup>, and marks the discrepancy as high sensitivity. This gives authors a concrete inconsistency to inspect and resolve before submission.

## D.6. Separate alignment analysis

A separate static audit compares selected paper results with their raw implementations. It distinguishes implemented behavior that contradicts the paper from absent implementations and details on which the paper is silent. Among 293 recorded findings, 238 are in scope for contradiction analysis and touch 669 distinct tasks. High-severity labels occur on 73 findings touching 482 tasks; 25 papers have at least one such finding. Task counts are deduplicated; a task may have multiple severity labels.

Agents produced these findings through static inspection of papers, code, and committed evidence. The analysis is separate from primary success scoring and covers the selected evaluation tasks. The evidence includes the learning-rate case above and a difusion-objective case in which the code squares a norm described as an unsquared distance, changing its gradient. These cases illustrate that numerical compatibility and code execution can coexist with a paper–implementation discrepancy.

From an apparent gap to a localized finding. The audit first extracts the paper-side requirement independently of the code-side inspection. It then identifies the implementation in that role, such as the function computing the loss or selecting the evaluation population. A contradiction requires an explicit paper statement and conflicting implemented behavior, with source locations and afected task IDs. If no implementation occupies the role, the finding is a missing implementation; if the paper specifies no relevant choice, it is underspecification rather than a contradiction. This separates 44 absent-implementation records and 11 paper-silent records from the 238 in-scope findings.

Table 19 | Static paper–code contradiction findings by recorded category. Scope is determined from evidence, not category names. Distinct tasks and papers are deduplicated within a row but can overlap between rows; these columns must not be summed. Counts describe agent-reviewed records.
<table><tr><td>Category</td><td>Findings</td><td>Papers</td><td>Distinct tasks</td></tr><tr><td>Hyperparameters</td><td>41</td><td>26</td><td>287</td></tr><tr><td>Objective</td><td>41</td><td>22</td><td>400</td></tr><tr><td>Data</td><td>40</td><td>20</td><td>298</td></tr><tr><td>Architecture</td><td>33</td><td>18</td><td>209</td></tr><tr><td>Evaluation procedure</td><td>30</td><td>20</td><td>261</td></tr><tr><td>Training procedure</td><td>22</td><td>16</td><td>176</td></tr><tr><td>Metric</td><td>21</td><td>13</td><td>112</td></tr><tr><td>Baseline implementation</td><td>5</td><td>3</td><td>4</td></tr><tr><td>Divergent implementation choices</td><td>3</td><td>3</td><td>15</td></tr><tr><td>Scale / hardware</td><td>2</td><td>2</td><td>25</td></tr></table>

What these discrepancies mean. A learning-rate disagreement is concrete even without retraining: one can point to the paper’s $1 0 ^ { - 4 }$ and the code’s $1 0 ^ { - 3 }$ . But the resulting change in final accuracy cannot be inferred from that factor alone. Similarly, replacing a distance by its square changes the objective and gradient weighting; its empirical impact requires a controlled comparison. The workflow records the contradiction and its scope while following the paper’s specified setup. These records can guide authors’ reproducibility checks before submission. In the terms of Section 2.1, the discrepancies are possible sources of $D ,$ not estimates of its magnitude.

Severity is also an agent judgment. The record set assigns 73 findings a high label, of which 63 use an argued magnitude-based route and 10 concern inability to instantiate the stated setup. These labels help prioritize inspection. The alignment records were not supplied to the evaluated Reproducers.

## D.7. Development-set results

We use Sonnet 5 and Gemini 3.8 Flash as Reproducers on the ten development papers: five from NeurIPS 2026 and five from ICLR 2026. Table 20 reports a single evaluation of reproduction success and downstream API spending under the same scoring definition as the test set. These papers informed workflow development, and their results are kept separate from the test-set aggregates.

Table 20 | Development-set results from a single evaluation. Success and downstream Reproducer API spending on the ten papers used for workflow development, reported separately from test results. RRA denotes refined reproducible artifacts; gain is RRA minus Raw in percentage points, and API reduction is 100(Raw − RRA)/Raw. Construction, auditing, and GPU costs are excluded.
<table><tr><td colspan="4">Success rate (%)</td><td colspan="3">API USD</td></tr><tr><td>Reproducer</td><td>Raw</td><td>RRA</td><td>Gain (pp)</td><td>Raw</td><td>RRA</td><td>Reduction (%)</td></tr><tr><td>Sonnet 5</td><td>75.6</td><td>92.2</td><td>+16.6</td><td>1,883</td><td>1,396</td><td>25.9</td></tr><tr><td>Gemini 3.8 Flash</td><td>59.7</td><td>91.2</td><td>+31.5</td><td>478</td><td>389</td><td>18.6</td></tr></table>

## E. Cost accounting

Table 21 | Three-evaluation mean paired Reproducer costs from Table 3, raw / RRA. Each model uses the same papers in both arms. Tokens include recorded input, output, cache writes/reads, and separately recorded thinking; USD uses frozen accounting rates. Construction, audit, and GPU costs are excluded. Reductions use the means.
<table><tr><td>Model</td><td></td><td>Tokens (B) Reduction (%)</td><td>API USD Reduction (%)</td></tr><tr><td>Sonnet 4.6</td><td> $8 . 4 8 / 6 . 8 0$ </td><td>19.8</td><td> $3 , 1 6 5 / 2 , 5 1 2$  20.6</td></tr><tr><td>Sonnet 5</td><td> $4 . 9 8 / 4 . 5 2$ </td><td>9.21</td><td> $\cdot , 7 2 6 / 1 , 3 1 7$  23.7</td></tr><tr><td>Opus 5</td><td> $3 . 0 1 / 2 . 8 2$ </td><td>6.3</td><td> $3 , 4 8 0 / 2 , 8 6 5$  17.7</td></tr><tr><td>Gemini 3.8 Flash</td><td> $2 . 2 4 / 1 . 8 3$ </td><td>18.3</td><td> $3 9 2 / 3 2 6$  16.8</td></tr></table>

Table 21 reports three-evaluation mean Reproducer costs, summing paper sessions within each evaluation. Claude tokens comprise input, output (including thinking), cache writes, and cache reads. Gemini totals include its separately recorded thinking field. The fixed per-milliontoken accounting rates, ordered as input/output/cache-write/cache-read, are (3, 15, 3.75, 0.30) for Sonnet 4.6, (2, 10, 2.50, 0.20) for Sonnet 5, and (5, 25, 6.25, 0.50) for Opus 5. Gemini rates are \$0.75 for input, \$3.75 for output plus thinking, and \$0.075 for cache reads; recorded cache writes are zero.

Paper-session costs cover failed attempts and work beyond the 1,029 retained scoring tasks.

## E.1. Construction investment and reuse

Constructing the 36 artifacts costs \$5,026 in API usage, averaging \$139.61 per paper.

Sources of construction efort. The workflow provides a qualitative explanation of API expenditure. Specification requires reading the paper and code to resolve experiment settings and measurement rules. Execution support requires environment repair, missing implementation, and diagnosis through successive runs; each revision can require further code and log inspection. Review and packaging then check the evidence and preserve the resulting procedures. The WBMM dependency route and HELIX configuration recovery in Section 4.4, together with the FlashOptim training/evaluation harness in Appendix D.5, illustrate these kinds of work. GPU runtime is a separate expense: it afects API spending through associated model calls for setup, inspection, and debugging, not through elapsed GPU time itself.

Model-specific reuse. For downstream model � in {Sonnet 4.6, Sonnet 5, Opus 5}, let $c _ { m } ^ { \mathrm { r a w } }$ and $c _ { m } ^ { \mathrm { r e f i n e d } }$ be the mean paired aggregate API costs in Table 3, and let $C _ { \mathrm { b u i l d } } = \$ 5,026$ be the full construction expense. One reuse round repeats the 36-paper workload represented by those costs. With stationary per-round spending, raw reuse costs $n c _ { m } ^ { \mathrm { r a w } }$ , while refined-artifact reuse costs $C _ { \mathrm { b u i l d } } + n c _ { m } ^ { \mathrm { r e f i n e d } }$ . The API break-even count is

$$
n _ { m } ^ { * } = \left\lceil \frac { C _ { \mathrm { b u i l d } } } { c _ { m } ^ { \mathrm { r a w } } - c _ { m } ^ { \mathrm { r e f i n e d } } } \right\rceil \quad \mathrm { w h e n } c _ { m } ^ { \mathrm { r a w } } > c _ { m } ^ { \mathrm { r e f i n e d } } .
$$

Each model-specific scenario bears the full construction cost. For positive construction cost and nonpositive savings, there is no finite API crossover.

For example, Sonnet 5 costs a mean of \$1,726 per raw round and \$1,317 per refined-artifact round. The diference is \$409; \$5,026 divided by this saving is 12.29 rounds, hence 13 complete rounds to break even. The corresponding continuous crossover points are 7.70 rounds for Sonnet 4.6 and 8.17 for Opus 5, giving the 8 and 9 complete rounds reported in Section 4.7.

These scenarios repeat the complete 36-paper workload at fixed prices, caching, artifact contents, and agent behavior. Costs and savings are aggregated over paper sessions, including shared executions that produce multiple task scores. The crossover accounts for construction and downstream Reproducer API spending; independent auditing, GPU execution, maintenance, and unrecorded cache-storage fees are excluded. Human reference collection is a separate benchmark-development cost.