# Validation-Gated ML Pipeline Synthesis with Expert-Grounded Self-Evolving Skills

Zixiao Zhao   
School of Computer Science   
University of Auckland   
Auckland, New Zealand   
zixiao.zhao@auckland.ac.nz

Jing Sun School of Computer Science University of Auckland Auckland, New Zealand jing.sun@auckland.ac.nz

Zhe Hou   
School of Information and   
Communication Technology   
Griffith University   
Brisbane, Australia   
z.hou@griffith.edu.au

Qian Liu School of Computer Science University of Auckland Auckland, New Zealand liu.qian@auckland.ac.nz

Mengze Li School of Computer Science University of Auckland Auckland, New Zealand mengze.li@auckland.ac.nz

Xin Wang

DSA Thrust

The Hong Kong University of

Science and Technology

(Guangzhou)

Guangzhou, China

xwang496@connect.hkust-gz.edu.cn

Cheng-Hao Cai   
Metasics, China.   
Xiamen, China   
chenghao.cai@metasics.cn

Gillian Dobbie School of Computer Science University of Auckland Auckland, New Zealand g.dobbie@auckland.ac.nz

Zijian Zhang   
School of Cyberspace   
Science and Technology   
Beijing Institute of Technology   
Beijing, China   
zhangzijian@bit.edu.cn   
Jin Song Dong   
Department of Computer Science   
National University of Singapore   
Singapore   
dcsdjs@nus.edu.sg

Abstract—Recent advances in large language models have enabled agentic systems that automate machine learning pipeline construction from minimal user input. However, existing agentic AutoML methods either depend on rigid templates that limit adaptability, target narrow task domains, or rely on freeform code generation that lacks reliability guarantees, leaving intermediate errors to propagate silently and degrade the final model. We present EvoML, an agentic framework for endto-end ML pipeline synthesis built on expert-grounded skills that evolve across runs, executed through a validation-gated harness. The framework decomposes pipeline construction into four phases coordinated by three agents: an Orchestrator that enforces validation gates, advancing each phase only when its outputs pass structural, execution, and semantic checks, which precludes error propagation and keeps every artifact auditable; an Executor that runs expert-grounded skills, grounded in the nnU-Net workflow and pairing reusable templates with online adaptation; and a Curator that performs post-run evolution, consolidating the scripts and guidance recovered during a run into the skill library so it improves across datasets. We evaluate EvoML on eight datasets spanning six sophisticated, real-world ML task types, against agentic AutoML systems and commandline coding tools. EvoML achieves full automation and the best final model quality on all eight, spanning segmentation (up to 0.94 Dice), classification (0.99 accuracy), reconstruction (0.95 SSIM), and denoising (0.79 normalized PSNR), while transferring to broader ML tasks.

Index Terms—Automated Machine Learning, Multi-Agent Framework, Program Synthesis

## I. INTRODUCTION

Automated Machine Learning (AutoML) aims to reduce the manual effort of building ML solutions by automating stages such as data preparation, model selection, hyperparameter optimisation, and pipeline construction [1], [2]. Traditional systems search over predefined model families, operators, and hyperparameter spaces [3]–[5], but their fixed search spaces and hand-designed primitives limit flexibility for complex workflows, domain-specific processing, or end-to-end executable code [6]–[8]. Large language models (LLMs) and agentic coding systems offer a new opportunity to interpret high-level intent, generate executable code, invoke tools, and iteratively refine solutions [9]. Building on these capabilities, recent work applies LLMs to individual ML pipeline stages, including data and feature extraction [10]–[12], model selection and training [13], [14], and hyperparameter optimisation [15], [16], offering more flexible, context-aware automation than fixed search spaces.

This potential has further motivated agentic AutoML systems that aim to coordinate multiple ML stages within a single end-to-end workflow. Recent systems [17]–[19] demonstrate the potential of LLM-based agents for automating ML workflows, but several limitations remain when applied to robust end-to-end pipeline construction. First, they often require users to supply template code, dataset-specific features, or detailed solution descriptions beyond minimal problem input, making automation sensitive to the quality of human guidance. Second, execution is not consistently governed by effective validation gates that verify whether each stage has produced executable, reusable artifacts before subsequent stages proceed; as a result, failures can propagate through the workflow, leading to incomplete artifacts or requiring manual intervention to diagnose and correct. Third, pipeline construction is commonly driven by search over predefined operators, retrieved knowledge, or online examples, which can limit robustness when tasks demand domain-specific procedures or tight coordination across dependent stages. Finally, failures observed during execution are rarely converted into persistent improvements of the underlying procedures, leaving the system unable to accumulate operational experience across runs.

To address these limitations, we propose EvoML, a validation-gated ML pipeline synthesis framework with expertgrounded, self-evolving skills. EvoML decomposes end-to-end pipeline synthesis into three coordinated GPT-5.5 [20] agents. The Orchestrator governs stage-by-stage progression through the ML workflow, enforcing validation-gated transitions, advancing only when expected artifacts and validation signals satisfy predefined criteria, such as successful execution, required file generation, valid data format, or task-relevant metric constraints, and triggering retries. The Executor performs each stage by invoking reusable, expert-grounded skills, each of which encodes a standard operating procedure for a specific ML stage, including data preparation, model configuration, training, hyperparameter optimisation, and code integration. The Curator analyses validation and retry traces after each run, identifies skills that incur repeated failures or excessive retries, and evolves those skills for future executions, allowing the framework to continuously refine its execution capabilities. By separating workflow control, skill-based execution, and post-run skill evolution into distinct agent responsibilities, the framework supports autonomous pipeline synthesis from minimal user input while maintaining auditable validation, bounded failure recovery, and continuous improvement.

This paper makes the following contributions.

• We present a fully automated agentic system for ML pipeline synthesis that autonomously executes, monitors, and refines pipelines from minimal user input, without relying on task-specific templates or human guidance.

• We introduce a unified validation-gated execution harness in which stage progression is governed by artifactand metric-based checks, enabling robust transitions, bounded failure recovery, and auditable execution traces.

• We couple expert-grounded skills with post-run evolution to support reliable and continuously improving ML pipeline synthesis, consolidating successful recoveries from validation and retry traces into persistent skill updates across runs.

The remainder of this paper is organised as follows: Section II reviews previous methods on LLM-based program synthesis and agentic AutoML. Section III presents the EvoML framework, detailing its three-agent design, expert-grounded skill library, validation-gated execution, and post-run skill evolution. Section IV reports the experimental evaluation across a range of ML tasks, including ablation studies of the core components. Section V provides an architectural comparison with the baselines and discusses cost-efficiency and threats to validity. Section VI concludes with a summary and future research directions.

## II. BACKGROUND

LLM-Based Program Synthesis Frameworks. Program synthesis, the automatic generation of programs from highlevel specifications, is a long-standing software-engineering task [21]; early formal, deductive approaches were limited in scalability and generality [22]. Large language models trained on code have since enabled the generation of syntactically correct, functionally plausible code from natural language [23]– [26], and agentic coding frameworks extend single-shot generation toward iterative, tool-augmented workflows [27], [28]. Multi-agent systems such as MetaGPT [29], ChatDev [30], SWE-agent [31], and MaCTG [32] decompose complex tasks across collaborating agents, and assistants such as GitHub Copilot [33], Cursor [34], Codex [35], and Claude Code [36] bring these capabilities into practical use. These generalpurpose frameworks, however, are not designed for the structural and procedural demands of ML pipeline construction: they lack domain-specific workflow coordination and depend on human-supplied guidance, such as task descriptions, template code, or annotated examples, limiting their applicability to autonomous end-to-end synthesis.

Agentic AutoML Methods. Applying agentic coding to ML pipeline synthesis raises challenges of stage coordination, domain-specific procedures, and execution reliability. Early efforts addressed individual stages in isolation: CAAFE [37] automates feature engineering and DS-Agent [38] applies case-based reasoning to data processing and training, but both rely on structured human input and leave cross-stage coordination unaddressed. AutoML-GPT [39] and SELA [40] attempt end-to-end automation through LLM-generated pipelines and tree search, yet still depend on detailed human input and lack validation-gated progression or failure recovery. The most capable end-to-end systems are AutoML-Agent [17], which coordinates multiple agents but requires user-supplied task templates; M3Builder [19], which performs strongly on medical imaging but is constrained to specific dataset types and tasks; and MLZero [18], which runs without manual input via documentation retrieval and iterative debugging but validates only execution success and retains memory within a single task. Across all of them, a common gap persists: none converts execution-time failures into persistent improvements, leaving the system unable to accumulate experience across runs.

## III. METHODOLOGY

EvoML synthesises a complete, executable ML pipeline from a dataset and a minimal task description, without taskspecific templates or human guidance. We follow a designand-evaluate process in which the limitations identified in Sections I–II are translated into three design requirements for reliable agentic ML pipeline synthesis: preventing error propagation across stages, grounding execution in reliable domain procedures, and retaining successful recovery experience across runs. These requirements motivate EvoML’s three core mechanisms: validation-gated orchestration, expert-grounded skill execution, and post-run skill evolution.

Accordingly, Section III first presents the overall system architecture and its three constituent agents, then introduces the expert-grounded skill library, describes the validation-gated execution that governs phase progression, and finally presents the post-run evolution mechanism that refines the skill library across runs.

## A. System Overview

Figure 1 illustrates how the proposed framework addresses the ML pipeline synthesis task through three dedicated agents: (1) the Orchestrator, which governs synthesis progression through validation gates; (2) the Executor, which runs expertgrounded skill groups from the skill library for each phase; and (3) the Curator, which operates post-run, auditing and evolving the skill library. For an ML pipeline synthesis task, the input consists of a dataset and a minimal task description specifying the task type and data path. Given this, the system autonomously generates a complete, executable ML pipeline covering all stages from data preparation to model training and evaluation. EvoML organises the synthesis via these three coordinated agents across four main phases.

At runtime, the Orchestrator interprets the user request and drives the four phases in turn, invoking the Executor with the corresponding skill group at each: Task Understanding (dataset preparation and fingerprint extraction), Planning (experiment planning and preprocessing), Training (HPO based on Optuna [41], promotion, and full training), and Post-Train Processing (evaluation aggregation and artifact export). After each phase the Executor returns its artifacts and logs for gate evaluation; on failure the Orchestrator returns the failure context for a bounded retry, and on success it advances. Once the run concludes, the Curator inspects the accumulated evidence, identifies underperforming skills, and proposes verified updates to the library for future runs.

This separation of concerns gives each agent an isolated context and clear accountability at each phase boundary: the Orchestrator validates independently of the code that produced the artifacts, the Executor focuses solely on running skills, and the Curator analyses the complete execution evidence postrun rather than reacting with partial context mid-run. The four-phase structure mirrors the natural decomposition of an ML pipeline, with validation boundaries that localise failures before downstream computation is committed.

## B. Expert-Grounded Skill Library Design

1) Skill Structure and Expert Grounding: Following the Agent Skills format [42], each skill in EvoML is a selfcontained module comprising a Markdown specification, a set of Python script templates, and a reference document that encodes the relevant domain knowledge. The Markdown file encodes the stage objective, expected inputs, the usage of the accompanying Python templates, and a self-validation instruction for the Executor to verify successful execution. The reference document, predominantly an output-format specification such as the expected dataset format, grounds each skill’s execution in a verifiable contract. Together, all skills form a shared library from which the Executor is prompted to invoke the appropriate skill at each stage of the current workflow.

Reliable end-to-end synthesis requires more than free-form code generation; it demands a principled procedure. We therefore ground EvoML’s skills in nnU-Net [43], a selfconfiguring framework that achieves strong performance not through elaborate architectures but by analysing the dataset and adapting preprocessing, model, and training configuration to its characteristics. This data-driven, configurationfirst approach is not specific to any one task type, which makes it a sound basis for general pipeline synthesis. This grounding also resolves a fundamental tension in agentic synthesis: pure LLM generation is flexible but structurally variable and unreliable across runs, whereas fixed templates are consistent but cannot capture task-specific requirements. Deriving templates from nnU-Net’s well-defined procedures supplies structural consistency, while LLM-generated adaptations retain the flexibility each task demands. nnU-Net itself, however, requires manual deployment and expert configuration per task and lacks automated validation; EvoML resolves these limitations by encoding its workflow as agent-executable skills, automating configuration while adding validation-gated execution and cross-run evolution that nnU-Net lacks.

2) Skill Library Composition: The library comprises nine skills spanning the four phases. The Dataset preparation skill converts a raw dataset into the standardised nnU-Net format, automatically selecting, cloning, and adapting the appropriate converter for the input layout. The Fingerprint extraction skill then computes a compact statistical summary of the dataset, including image shape, voxel spacing, and intensity distribution, which conditions all subsequent configuration decisions. From this fingerprint, the Experiment planning skill derives the model and training configuration, including network topology, patch size, and batch size, while the Preprocessing skill crops, resamples to the target spacing, and normalises intensities to produce training-ready arrays. The Optuna HPO and HPO promotion skills are introduced beyond nnU-Net’s native workflow to improve final performance. The HPO skill runs hyperparameter optimisation trials on a fixed tuning fold, sampling learning rate, weight decay, and earlystopping configurations, while the promotion skill compares the best trial against the baseline configuration and promotes it only when the performance gain exceeds a predefined threshold. The Full training skill then executes the standard nnU-Net five-fold cross-validation under the promoted configuration, augmented with an early-stopping mechanism that halts training once validation performance converges, and the Evaluation aggregation skill consolidates the cross-validation predictions, determines postprocessing, and computes the final metrics. Finally, the Artifact export skill packages the executed stage scripts into a standalone, runnable pipeline. These nine skills constitute a fixed, complete decomposition of the ML pipeline; evolution therefore refines existing skills rather than inserting or deleting them.

![](images/ec45ebf33930c3879812b9355f8fea6746652a73534a0662ee757eecb68e021f.jpg)  
Fig. 1: Overview of the proposed validation-gated agentic ML pipeline synthesis framework. The Orchestrator progresses through four phases and invokes the corresponding Executor skill group in sequence: Task Understanding (skills 1–2), Planning (3–4), Training (5–7), and Post-train Processing (8–9). After each phase, generated artifacts and execution evidence are returned for gated validation; passed phases advance, while failed phases trigger recovery. The Curator operates after the run to evolve the skill library from accumulated execution evidence.

## C. Validation-Gated Pipeline Execution

The Orchestrator and Executor interact in a structured invoke-validate-advance loop within a unified validation-gated harness that enforces phase-level contracts, manages artifact dependencies, and provides bounded failure recovery across four phases from Task Understanding to Post-Train Processing.

1) Phase Execution: At each phase $p _ { k }$ , the Orchestrator invokes the Executor to run the skill group $\Phi _ { k }$ , which takes the artifact set $\mathcal { A } _ { k - 1 }$ produced by the preceding phase as input and produces a new artifact set $\mathcal { A } _ { k }$ according to:

$$
\mathcal { A } _ { k } = \Phi _ { k } ( \mathcal { A } _ { k - 1 } , \mathcal { C } _ { k } )\tag{1}
$$

where $\mathcal { C } _ { k }$ denotes the task context at phase $p _ { k }$ , including the task type, run state, and any phase-specific configuration inherited from earlier phases. Concretely, running $\Phi _ { k }$ executes each constituent skill sequentially.

For each skill $\phi \in \Phi _ { k }$ , the Executor selects the appropriate Python template $\tau _ { k } ~ \in ~ \phi$ based on the task context $\mathcal { C } _ { k }$ Where the template covers the required configuration, it is executed directly with task-specific parameters substituted; where adaptation is required, the Executor invokes Codex [35] as an MCP tool [44], passing the template and task context to generate a task-specific adaptation. The Executor then executes the adapted script and verifies it against the self-validation criteria $v _ { k }$ specified in the skill’s Markdown, catching any runtime errors or tracebacks and re-invoking Codex with the error context to iteratively refine the adaptation until the script executes successfully. The Executor returns $\mathcal { A } _ { k }$ to the Orchestrator only when $v _ { k } ( \mathcal { A } _ { k } ) = 1$ , establishing self-validation as a necessary correctness condition that prevents structurally invalid outputs from reaching the phase-level validation gate and incurring unnecessary recovery overhead.

2) Validation Gates and Recovery: Each gate verifies that the phase has produced the artifacts the next phase depends on: a formatted dataset and fingerprint in Task Understanding, an experiment plan and preprocessed data in Planning, the optimised configuration and trained weights in Training, and the aggregated metrics and exported pipeline in Post-Train Processing. At every phase, the Orchestrator applies the same three categories of check: structural (expected artifacts and formats), execution (process completion and exit status), and semantic (task-appropriateness of the artifacts). What differs between phases is not the checks but the reference they are evaluated against, which specifies the outputs a given phase should produce and remains largely stable across runs.

Structural and execution checks are performed programmatically, matching the produced artifacts against concrete expectations: the files that must exist, their formats and attributes, and each step’s exit conditions. The semantic category admits no fixed rule: whether an artifact suits the task is a matter of convention, so the Orchestrator, as an LLM, evaluates it against the domain knowledge encoded in the phase’s reference, for example applying clipping to CT rather than the z-score normalisation appropriate to MRI.

Formally, the gate evaluates the phase output $\mathcal { A } _ { k }$ against the phase’s criteria $\Gamma _ { k } = \Gamma _ { k } ^ { \mathrm { s t r } } \cup \Gamma _ { k } ^ { \mathrm { e x e } } \cup \Gamma _ { k } ^ { \mathrm { s e m } }$ , collecting the structural, execution, and semantic criteria, and passes only if all are satisfied,

$$
g _ { k } ( { \mathcal { A } } _ { k } ) = \bigwedge _ { \gamma \in \Gamma _ { k } } { \mathcal { H } } _ { \gamma } ( { \mathcal { A } } _ { k } )  \{ 0 , 1 \} \times { \mathcal { L } } _ { k } ,\tag{2}
$$

where ${ / k } _ { \gamma } ( { \mathcal { A } } _ { k } ) = 1 \mathrm { i f } \ A _ { k }$ satisfies criterion $\gamma$ and 0 otherwise. For each failed criterion, the Orchestrator records its category $c _ { i }$ and a natural-language summary $s _ { i } ,$ producing the structured log

$$
\mathcal { L } _ { k } = \{ ( c _ { i } , s _ { i } ) | \mathcal { k } _ { \gamma _ { i } } ( \mathcal { A } _ { k } ) = 0 \} ,\tag{3}
$$

returned to the Executor as $\mathcal { F } _ { k } = ( 0 , \mathcal { L } _ { k } )$ for targeted recovery rather than full re-execution. A phase advances only when its gate passes; it retries while $r _ { k } < R _ { \operatorname* { m a x } }$ . If the retry limit is reached, the Orchestrator invokes a minimal-safe fallback that performs only the essential operations required by the phase contract without further optimisation. The fallback is evaluated by the same gate, and the workflow terminates if it also fails. In our experiments, normal recovery always succeeded before this fallback was required. Because each phase executes only on artifacts that have passed all upstream gates, error propagation is structurally precluded, and every committed artifact is auditable through the criteria it satisfied.

Figure 2 illustrates an example cycle of this gating-recovery mechanism in the Planning phase. Taking the formatted dataset and the extracted fingerprint $\boldsymbol { A } _ { 1 }$ as input, the Orchestrator invokes the Executor with the Planning skills. The Executor runs the experiment planning and preprocessing skills and selfvalidates that the scripts execute successfully, producing the experiment plan and preprocessed data as $\boldsymbol { A } _ { 2 }$ . The Orchestrator then evaluates $\boldsymbol { A } _ { 2 }$ at the validation gate: although the structural and execution checks pass, the semantic check detects that a CT dataset has been normalised with an MRI-style Zscore rather than the required clipping, a correctness error that script-level self-validation cannot catch. The resulting failure context is returned to the Executor for a targeted retry, and only once all checks pass does the phase advance to Training.

## D. Post-Run Skill Evolution

While the Orchestrator’s recovery resolves failures within a run, its effects are yet transient: the adapted scripts it yields are not retained, and the guidance that misjudged the case remains unrefined. A skill encountering a similar condition must therefore rediscover the same resolution through retries, a pattern that recurs whenever such conditions arise. The Curator addresses this by operating post-run, transforming accumulated execution evidence into durable refinements of the skill library. In-run recovery and post-run evolution are therefore driven by a common signal: the validation and retry traces emitted at each gate. The harness persists these traces during execution, so the Curator’s evolution draws on the same auditable evidence that governs in-run control.

![](images/fa599963b00ea52539e8193f0f481bc362a2434a7b4496b60a16b45f989e0cde.jpg)  
Fig. 2: An example validation-gated recovery cycle in the Planning phase.

1) Evidence Aggregation and Brittleness Identification: After a run concludes, the Curator aggregates the traces that the harness persisted for each phase into a per-skill evidence record. For a skill $\phi ,$ this record collects the retry count it incurred, the validation logs of the checks it failed, and the adapted script $\boldsymbol { \tau } _ { \phi } ^ { \star }$ that the Executor produced to resolve the difficult case during recovery:

$$
\mathcal { E } _ { \phi } = \left( r _ { \phi } , \ \{ \mathcal { L } _ { k } \mid \phi \in \Phi _ { k } \} , \ \tau _ { \phi } ^ { \star } \right) , \quad \quad r _ { \phi } = \sum _ { k : \phi \in \Phi _ { k } } r _ { k } .\tag{4}
$$

Each skill is first screened by its accumulated retry count $r _ { \phi } .$ , which indicates where the standing guidance may not sufficiently cover the encountered task conditions. The skill with the highest retry count is selected for further inspection:

$$
\phi ^ { \star } = \arg \operatorname* { m a x } _ { \phi } r _ { \phi } .\tag{5}
$$

The Curator then analyses its complete evidence record, including failed validation conditions, the recovery sequence, and the successfully adapted script, to identify the underlying deficiency and determine whether and how the skill should be updated.

As shown in Figure 3, a 3D reconstruction dataset supplied in HDF5 format forces the Dataset preparation skill into repeated recovery, as the Executor initially assumes the NIfTI layout used for segmentation and adapts a new HDF5 preparation script only after retrying. The Curator aggregates the traces of this run into the skill’s evidence record $\mathcal { E } _ { \phi } ,$ , capturing its elevated retry count, the validation logs of the failed format checks, and the adapted HDF5 script that ultimately succeeded. Brittleness identification across all skills then identifies Dataset preparation as the brittleness maximiser $\phi ^ { \star }$ , marking it as the candidate for patching.

![](images/f3161d7ff79b6689affcb672c05da1503091f478cf7674a26e8f7953c1fa6973.jpg)  
Fig. 3: The Curator’s post-run skill evolution loop after a 3D reconstruction task with HDF5 dataset format.

2) Patch Proposal and Validation: For the flagged skill $\phi ^ { \star }$ , the Curator analyses its evidence record $\mathcal { E } _ { \phi ^ { \star } }$ to diagnose the deficiency underlying the retries, and proposes a patch δ comprising two complementary parts. 1) The adapted script $\boldsymbol { \tau } _ { \phi ^ { \star } } ^ { \star }$ that the Executor produced to resolve the difficult case during the run, recorded in the evidence but not yet part of the library, is consolidated into the skill as a new reusable template; and 2) the skill’s durable knowledge, its Markdown specification or reference document, is extended to describe the condition and when that template applies. Consolidating both the recovered script and the guidance that selects it transfers the handling of the condition from reactive recovery to proactive capability, generalising it to subsequent encounters rather than remedying only the instance that elicited it.

Before it is committed, the patch is validated by reexecuting the affected step on the precise input $\mathcal { A } ^ { \star }$ that triggered the retries, and is accepted only if the re-execution satisfies the corresponding gate:

$$
\phi ^ { \star }  \{ \phi ^ { \star } \cup \delta \quad { \mathrm { i f ~ } } g ( \Phi ( A ^ { \star } ; \phi ^ { \star } \cup \delta ) ) = 1  \qquad\tag{6}
$$

Validation requires the patched skill to pass the gate on its first execution without retries, indicating that the guidance now covers the condition rather than leaving it to reactive recovery. A verified patch is committed additively, as $\phi ^ { \star } \cup \delta ,$ extending the skill while preserving previously covered cases, so evolution maintains coverage monotonically. Over successive runs, the library thus covers more of the encountered task distribution.

Figure 3 depicts a representative instance for the Dataset preparation skill: the Curator consolidates the recovered HDF5 preparation script into the skill’s templates and extends its specification with a data-type routing clause that recognises HDF5 inputs up front, then validates the patch by re-executing preparation on the same dataset. Upon passing the gate, the patch is committed to the evolved skill library so that the recovered handling is applied proactively thereafter.

## IV. EVALUATION

We evaluate the proposed framework through a series of experiments designed to assess its end-to-end effectiveness, the contribution of its core design components, and its generalisability. Specifically, we investigate the following research questions:

RQ1: How does EvoML compare against existing agentic AutoML systems and command-line coding tools on endto-end ML pipeline synthesis?

RQ2: To what extent does validation-gated execution contribute to pipeline automation and model performance? RQ3: How do expert grounding and post-run skill evolution contribute to pipeline synthesis performance?

RQ4: How well does the framework generalise beyond the targeted benchmarks to broader ML tasks?

These RQs follow the design of EvoML and progressively examine its effectiveness. RQ1 evaluates the complete framework against existing agentic AutoML systems and coding tools. RQ2 isolates the contribution of validation-gated execution, while RQ3 examines the effects of expert grounding and post-run skill evolution. RQ4 then evaluates whether the framework extends beyond the primary medical-imaging setting to broader ML tasks.

## A. Benchmarking Datasets

We evaluate primarily on medical imaging, a demanding testbed for ML pipeline synthesis. Existing ML-engineering benchmarks such as MLE-bench [45] centre on competitionstyle tasks in which the challenge is maximising performance through model selection and tuning, rather than constructing a correct pipeline. Medical imaging instead combines heterogeneous formats, 2D and 3D volumes, and heavy modality-specific preprocessing, all under strict correctness constraints, yet a runnable pipeline can still be silently wrong. We select eight medical datasets spanning six task types: BTCV [46]: supervised 3D segmentation of 13 organs in CT (50 cases; 512 × 512 × 123), with few, imbalanced samples. ACDC [47]: supervised 3D cardiac segmentation in MRI $( 2 1 5 \times 2 2 8 \times 1 0 )$ , with pronounced voxel anisotropy. FLARE21 [48]: supervised 3D segmentation of 4 abdominal organs in CT (361 cases; $5 1 2 \times 5 1 2 \times 1 2 6 )$ . FLARE22 [49]:

semi-supervised 3D segmentation of 13 abdominal organs in CT (50 labelled, 2000 unlabelled; $5 1 2 \times 5 1 2 \times 1 9 6 )$ , where label scarcity and semi-supervised training policy are the central difficulty. Promise12MSBench [50]: supervised 2D binary prostate segmentation in MRI $( 5 1 2 \times 5 1 2 )$ , adapted from PROMISE12 [51]. OrganMNIST3D [52]: supervised 3D classification of 11 organ categories in CT (1,742 samples; $6 4 \times 6 4 \times 6 4 )$ , the least demanding task. FastMRI [53]: supervised 3D reconstruction from undersampled k-space (60 cases; 288 $\times 6 4 0 \times 1 6 \times 8 3 )$ , demanding for its large HDF5 footprint and k-space pairing. IXI [54]: supervised 3D denoising of brain MRI (581 T1 volumes; $2 5 6 \times 2 5 6 \times 1 5 0 )$ , requiring fine structure to be retained while suppressing noise.

## B. Evaluation Metrics

We evaluate each system along two dimensions: its ability to autonomously construct an executable end-to-end ML pipeline, and the predictive quality of the resulting model. Accordingly, we report End-to-End Automation Capability (EAC) and Final Model Quality (FMQ).

End-to-End Automation Capability (EAC). EAC measures how autonomously a method produces a complete, executable pipeline, penalised per step by the human effort required. We track five milestones: (1) dataset preparation, (2) model design, (3) training, (4) hyperparameter optimisation, and (5) pipeline integration. For milestone k in run $r ,$ let $c _ { r , k } \in \{ 0 , 1 \}$ indicate completion with a valid artifact, and $h _ { r , k }$ the number of interventions at that milestone. The permilestone score is:

$$
s _ { r , k } = c _ { r , k } \cdot \left( 1 - \operatorname* { m i n } \left( \frac { h _ { r , k } } { 3 } , 1 \right) \right) ,\tag{7}
$$

where the intervention budget of three distinguishes minor recoverable failures from cases requiring substantial manual guidance. EAC is the mean across milestones:

$$
\mathrm { E A C } _ { r } = \frac { 1 } { 5 } \sum _ { k = 1 } ^ { 5 } s _ { r , k } .\tag{8}
$$

Each milestone contributes only when completed with a valid artifact, and its contribution decreases with the number of interventions it required.

Final Model Quality (FMQ). FMQ measures the predictive performance of the final model. For segmentation datasets, given predicted foreground mask $\hat { Y } _ { r , i }$ and ground-truth mask $Y _ { i }$ for test case $i \in T$ , we use the Dice score [55], [56]:

$$
\mathrm { F M Q } _ { \mathrm { s e g } , r } = \mathrm { D i c e } _ { r } = \frac { 1 } { | T | } \sum _ { i \in T } \frac { 2 | \hat { Y } _ { r , i } \cap Y _ { i } | } { | \hat { Y } _ { r , i } | + | Y _ { i } | } .\tag{9}
$$

For classification datasets, we use accuracy:

$$
\mathrm { F M Q } _ { \mathrm { c l s } , r } = \mathrm { A c c } _ { r } = \frac { 1 } { | T | } \sum _ { i \in T } \mathbb { I } ( \hat { y } _ { r , i } = y _ { i } ) ,\tag{10}
$$

where $\hat { y } _ { r , i }$ and $y _ { i }$ denote the predicted and ground-truth labels, respectively.

For the reconstruction task, we use the Structural Similarity Index Measure (SSIM) [57]:

$$
\mathrm { F M Q } _ { \mathrm { r e c } , r } = \mathrm { S S I M } _ { r } = \frac { 1 } { | T | } \sum _ { i \in T } \frac { ( 2 \mu _ { p } \mu _ { q } + c _ { 1 } ) ( 2 \sigma _ { p q } + c _ { 2 } ) } { ( \mu _ { p } ^ { 2 } + \mu _ { q } ^ { 2 } + c _ { 1 } ) ( \sigma _ { p } ^ { 2 } + \sigma _ { q } ^ { 2 } + c _ { 2 } ) } ,\tag{11}
$$

where $p = \hat { Y } _ { r , i } , q = Y _ { i } , \mu _ { p } , \mu _ { q }$ and $\sigma _ { p } ^ { 2 } , \sigma _ { q } ^ { 2 }$ denote the local means and variances, $\sigma _ { p q }$ the local covariance, and $c _ { 1 } , \ : c _ { 2 }$ are numerical stability constants.

For denoising, we use the Peak Signal-to-Noise Ratio (PSNR):

$$
\mathrm { F M Q } _ { \mathrm { d e n } , r } = \mathrm { P S N R } _ { r } = \frac { 1 } { | T | } \sum _ { i \in T } 1 0 \log _ { 1 0 } \left( \frac { L ^ { 2 } } { \mathrm { M S E } ( \hat { Y } _ { r , i } , Y _ { i } ) } \right) ,\tag{12}
$$

where L is the maximum possible pixel intensity value and $\begin{array} { r } { \mathrm { M S E } ( \hat { Y } _ { r , i } , Y _ { i } ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } ( \hat { Y } _ { r , i , j } - \hat { Y _ { i , j } } ) ^ { 2 } } \end{array}$

## C. Experiment Settings

1) Baseline Selection and Configuration: We compare against two categories of baselines: three agentic AutoML systems targeting end-to-end ML pipeline construction, AutoML-Agent [17], MLZero [18], and M3Builder [19]; and two widely adopted command-line coding tools, Codex [35] (GPT-5.5 Medium) and Claude Code [36] (Opus 4.7 Medium). To ensure a fair comparison, we apply the same task requirements to all baselines. For the agentic AutoML systems, we preserve each system’s native prompting conventions so that it operates under the input format for which it was designed, and additionally supply the skeleton and template files that certain baselines require, following the setup specified in their respective implementations. For the command-line coding tools, we provide the same task description as EvoML and guide them through the same four-phase workflow, ensuring that all methods address an equivalent synthesis task.

2) Environment and Protocol: All experiments are conducted on an Ubuntu 22.04 server equipped with an AMD Ryzen Threadripper PRO 3995WX CPU (64 cores), 252GB of RAM, and two NVIDIA RTX A6000 GPUs (48GB VRAM each). We provide all methods in a unified Python environment pre-installed with ML and medical imaging libraries and common dependencies, and each method may install additional dependencies as needed during pipeline execution to ensure a fair comparison. To account for the stochasticity of LLMbased generation, each method is executed three times per dataset, and we report the mean ± std of EAC and FMQ.

## D. Experiment Results

1) Comparison with Baseline Methods: Figure 4 compares EvoML against the five baselines across the eight datasets. On automation (EAC), EvoML is the only method that reaches 1.0 on every dataset, as its validation-gated harness resolves stage failures within the retry budget without human intervention. The baselines reflect their design assumptions. M3Builder handles supervised 3D segmentation reliably but falls short on the semi-supervised and non-segmentation tasks, where its EAC drops to 0.4667 on FLARE22 and FastMRI. MLZero, relying on documentation retrieval and iterative trial rather than a fixed workflow, completes pipelines unevenly (0.5111 to 0.7778) as its exploratory loops stall on harder tasks. AutoML-Agent, lifted by the code skeleton we supply, reaches near 1.0 on the easier segmentation and classification tasks but fails to adapt to harder scenarios, dropping to 0.7778 on BTCV and 0.7555 on IXI. The command-line tools are the most autonomous baselines, recovering from runtime errors by monitoring the terminal, yet they require intervention on the reconstruction and denoising tasks, where failures arise from modelling design rather than execution.

![](images/5f50c47511df92dc1d74a9e313ff5b7af0cb29c44cf8dfab0e064fd5ee6d28a6.jpg)

![](images/f63c05de7c60659235077cb61d3139c5a243b7efd4249a850f62f01b506bad26.jpg)  
Fig. 4: Comparison of EvoML against agentic AutoML systems and command-line coding tools across the benchmarking datasets: EvoML attains full automation and the best FMQ on every dataset, with markedly lower variance than the baselines.

On model performance (FMQ), EvoML attains the best result on every dataset, owing to its expert-grounded skills and the systematic HPO and promotion the baselines lack. The margin is largest on the harder tasks, reaching 0.9498 on FastMRI against the next-best 0.8579 and 0.7859 on IXI against 0.7215. The baselines degrade in ways that trace to their design: the command-line tools stay competitive on easier tasks but fall away on the hardest, with Codex dropping to 0.6167 on BTCV and 0.5309 on IXI, while the dedicated AutoML systems collapse more sharply, AutoML-Agent to 0.3837 on ACDC and M3Builder to 0.1979 on FastMRI.

We take FastMRI as a representative case, since reconstruction exposes the widest range of quality and allows clear rightand-wrong design choices. Table I compares each method’s input–target pairing strategy, model design, and loss function selection, and the results track how faithfully each handles the reconstruction-specific requirements. The command-line tools adopt a sound model and loss function but shortcut the data pairing, synthesising the input from the target rather than the measured k-space, which simplifies the task and yields competitive but lower SSIM. MLZero makes sounder choices, reconstructing via backprojection with an L1-trained

TABLE I: FastMRI reconstruction pipeline design choices across methods, with resulting mean FMQ (SSIM).
<table><tr><td>Method</td><td>Input Pairing</td><td>Model</td><td>Loss / Output</td><td>FMQ</td></tr><tr><td>Codex</td><td>Synthesised from target</td><td>U-Net3D</td><td>L1+SSIM, 1-ch</td><td>0.8579</td></tr><tr><td>Claude Code</td><td>Synthesised from target</td><td>U-Net3D</td><td>L1+SSIM, 1-ch</td><td>0.8521</td></tr><tr><td>MLZero</td><td>Backprojection → temptv</td><td>U-Net2D</td><td>L1, 1-ch</td><td>0.6953</td></tr><tr><td>AutoML-Agent</td><td>Zero-filled → temptv</td><td>U-Net2D</td><td>SSIM, 4-ch sigmoid</td><td>0.6583</td></tr><tr><td>M3Builder</td><td>Zero-filled → temptv</td><td>U-Net3DTiny</td><td>L1+SSIM, 4-ch sigmoid</td><td>0.1979</td></tr><tr><td>EvoML</td><td>Undersampled → full k-space RSS</td><td>VarNet3D</td><td>L1+SSIM, 1-ch</td><td>0.9498</td></tr></table>

2D U-Net. AutoML-Agent and M<sup>3</sup>Builder fare the worst, importing segmentation defaults unsuited to reconstruction: both pair a coarsely resized, zero-filled reconstruction with a mismatched target and retain a multi-channel sigmoid output suited to label maps rather than intensity regression, with M<sup>3</sup>Builder collapsing to 0.1979 under the most aggressive resizing. EvoML follows the correct protocol, undersampling the fully sampled k-space relative to the full RSS target, using a reconstruction-specific VarNet3D and an L1–SSIM objective, achieving the highest FMQ. Final quality thus depends less on model capacity than on whether the pipeline respects the task structure, which is what EvoML’s expert grounding enforces.

EvoML is also markedly more consistent, with FMQ standard deviation below 0.013 on every dataset, whereas the baselines fluctuate widely (AutoML-Agent ±0.2127 on IXI, M3Builder ±0.1317 on FLARE22). This stability stems from the expert-grounded skills, which anchor every run in the same proven nnU-Net procedure rather than re-deriving the pipeline, while evolution reinforces it by settling the correct task-specific handling into the skills so quality-determining choices are applied reliably across runs.

Answer to RQ1: EvoML outperforms all five baselines, achieving full automation and the best final model quality on every dataset with markedly lower variance. This stems from three complementary strengths: expert-grounded skills produce consistently strong models, post-run evolution stabilises quality across runs, and the validation-gated harness sustains full autonomy.

2) Ablation Study: We conduct two ablations to isolate the contributions of EvoML’s core components: the validationgated execution harness and the expert-grounded, self-evolving skill design. Each ablation removes or varies one component while holding the rest of the framework fixed, and is evaluated under the same protocol as the main comparison.

![](images/ec1b65b0305643e72b2f45c381796f06cbbf4499922f8ec668ebd5e06108ceea.jpg)  
Fig. 5: Effect of validation gating: without gates, EAC and FMQ both decline, sharply on the harder reconstruction, denoising, and semi-supervised tasks.

Validation Gating. To isolate the contribution of validationgated execution, we compare full EvoML against a variant in which the gates are removed, so stages progress unconditionally and failures are neither detected nor automatically recovered. Figure 5 reports both metrics across the eight datasets. The removal of validation gates degrades automation wherever failures arise: a faulty artifact can no longer be detected and repaired in place, so pipeline completion reverts to manual intervention, and EAC declines from 1.0 to as low as 0.5333 on IXI and 0.6000 on FastMRI. This degradation scales with task difficulty, leaving the simpler OrganMNIST, FLARE21, and ACDC largely unaffected while the more demanding reconstruction, denoising, and semi-supervised tasks deteriorate substantially. Model quality declines in parallel, though for a more subtle reason. The errors intercepted by the gates are frequently not outright failures but silent inconsistencies, such as the application of an MRI-style normalisation to CT data, which execute without error yet corrupt the training process. Left unchecked, such errors propagate into the model and manifest only as degraded output: FMQ falls most markedly on the hardest tasks, declining from 0.9199 to 0.7422 on FLARE22 and from 0.9498 to 0.7116 on FastMRI, while the simplest task, OrganMNIST, remains nearly unaffected. The gates thus safeguard both autonomy, by recovering failures automatically, and quality, by arresting silent errors before they reach the model.

Answer to RQ2: Validation gating underpins both automation and quality: by detecting and repairing faulty artifacts in place, it sustains autonomy; by intercepting silent inconsistencies, it prevents the error propagation that would otherwise degrade model performance.

Skill Expert-Grounding and Evolution. We examine the skill design along two axes: the contribution of expert grounding, and the contribution of post-run evolution.

![](images/badb3e03dfc624c04caebf792b62e50b748f3ed5505a678aad88a31c9b5c6dcd.jpg)  
Fig. 6: Effect of expert grounding: without the nnU-Net grounding, both automation and quality fall and variance roughly doubles.

Expert-Grounding. We compare full EvoML against a variant that keeps the skill structure, harness, and evolution but removes the nnU-Net grounding, replacing dataset preparation, fingerprinting, planning, preprocessing, training, and evaluation with generic, executor-designed equivalents. Lacking a proven procedure to follow, the executor re-derives each step from scratch, so its choices vary from run to run and often miss the dataset-specific handling the grounded skills encode. As Figure 6 shows, these ad hoc steps fail more often, lowering EAC to as low as 0.6667 on FLARE22, and yield weaker models, with FMQ dropping from 0.9199 to 0.7150 on FLARE22 and 0.9498 to 0.7967 on FastMRI; run-to-run variance also roughly doubles, since each run reconstructs the pipeline independently. Expert grounding therefore raises quality, sustains autonomy, and reduces variance.

![](images/7efb1e1cc2a14955d414a14f6146d76b70bbdb3d30f9249b1d80ef544a8f5712.jpg)

(a) Synthesis time  
![](images/a462be11087ebd7843ceec1dcbfe5d92c8632b3fa9c8e4875a52ac662d572165.jpg)  
(b) Token consumption  
Fig. 7: Effect of post-run evolution: evolution reduces synthesis time and token consumption across the dataset sequence.

Skill Evolution. We isolate post-run evolution by disabling the Curator: the Executor still adapts templates online, but neither the adapted scripts nor the guidance refinements are retained, so every dataset runs against the original library. Both configurations run over an identical, fixed dataset order, from segmentation (BTCV, ACDC, FLARE21, the semi-supervised FLARE22, Promise12) through classification (OrganMNIST) to the hardest reconstruction and denoising tasks (FastMRI, IXI). Only the evolving configuration accumulates experience across this sequence. Results are shown in Figure 7. Evolution reduces total synthesis time by roughly 41% (from 490 to 290 minutes) and token consumption by roughly 30% (from 71 to 50 million), as later datasets can reuse previously recovered scripts and guidance instead of rediscovering them through repeated adaptation. Beyond efficiency, we assess evolution using three process-level criteria: first-pass skill success, whether a skill completes without recovery; validationtriggered retries, the number of recovery attempts caused by failed gates; and cross-task reuse, whether adaptations retained from earlier datasets improve first-pass execution on later tasks. Excluding BTCV, where no prior evolution is available, first-pass skill success increases from 36/63 (57.1%) to 50/63 (79.4%), while validation-triggered retries decrease from 27 to 13 (−51.9%). Cross-task reuse is also visible on later datasets: Promise12 and OrganMNIST complete all nine skill executions on the first pass, while IXI improves from 3/9 to 9/9 after accumulated evolution. The two configurations coincide on BTCV, confirming that these gains arise only after experience has accumulated across runs. These process-level criteria complement FMQ by capturing changes in execution behaviour across runs, allowing us to distinguish retained procedural improvement from changes in final predictive performance.

Answer to RQ3: Expert grounding and post-run evolution contribute in complementary ways. Expert grounding improves model quality, autonomy, and consistency, while evolution reduces repeated recovery, improves first-pass execution, and lowers synthesis time and token consumption as experience accumulates across tasks.

## E. Generalisation

To test whether EvoML generalises beyond nnU-Net’s medical grounding domain, we evaluate it on three general ML tasks: natural image classification (CIFAR-10 [58]), natural image segmentation (VOC2012 [59]), and text classification (AG News [60]), against all five baselines. These general tasks are comparatively simple and follow well-established pipelines, so all methods achieve near-perfect automation, with EAC close to 1.0 throughout; Table II therefore reports FMQ, the dimension on which they differ.

TABLE II: Generalisation to three general ML tasks: FMQ $( \mathrm { m e a n } \pm \mathrm { s t d } )$
<table><tr><td>Method</td><td>CIFAR-10</td><td>VOC2012</td><td>AG News</td></tr><tr><td>M3Builder</td><td>0.9216 ±0.0049</td><td>0.6097 ±0.0123</td><td>0.9113 ±0.0044</td></tr><tr><td>AutoML-Agent</td><td>0.9419 ±0.0027</td><td>0.7328 ±0.0084</td><td>0.9103 ±0.0056</td></tr><tr><td>MLZero</td><td>0.9466 ±0.0030</td><td>0.7117 ±0.0068</td><td>0.9441 ±0.0033</td></tr><tr><td>Codex</td><td>0.9424 ±0.0042</td><td>0.7036 ±0.0088</td><td>0.9207 ±0.0061</td></tr><tr><td>Claude Code</td><td> $0 . 9 3 9 1 \pm \substack { 0 . 0 0 3 2 }$ </td><td>0.7086 ±0.0065</td><td>0.9457 ±0.0039</td></tr><tr><td>EvoML</td><td> $\mathbf { 0 . 9 6 7 4 } \pm \mathbf { 0 . 0 0 2 6 }$ </td><td> $\mathbf { 0 . 7 4 0 1 } \pm \mathbf { 0 . 0 0 5 8 }$ </td><td>0.9548 ±0.0030</td></tr></table>

EvoML achieves the best FMQ across all three tasks, confirming that its framework generalises beyond medical imaging. The expert-grounded skills are not rigidly tied to the medical setting: the Executor adapts them to each new domain and post-run evolution refines them further, so the grounded procedures continue to benefit general tasks. The margins are narrower here than on the medical benchmarks largely because these tasks are simpler and more standardised, leaving less room to distinguish methods, yet EvoML’s combination of adapted expert skills, validation-gated execution, and evolution still yields the strongest results on every task.

Answer to RQ4: EvoML extends beyond the targeted benchmarks to general ML tasks, achieving the best FMQ on all three with full automation. Its expert-grounded skills adapt to each task and, with validation-gated execution and evolution, continue to outperform every baseline.

## V. DISCUSSION

## A. Architectural Comparison with Baselines

Table III contrasts EvoML with the baselines on the architectural dimension that distinguishes it. M3Builder and AutoML-Agent require a user-supplied skeleton or templates, whereas MLZero and the command-line tools run from minimal input but lack expert grounding. On validation, the baselines check only execution success (MLZero, command-line tools), file and shape structure (M3Builder), or conformance to user requirements assessed over pseudo-execution rather than real runs (AutoML-Agent); none verify task-level correctness, so a runnable but task-inappropriate pipeline passes unflagged. Furthermore, no baselines retain experience once a run completes. EvoML alone combines minimal-input operation, adaptive expert grounding, task-aware validation, and cross-run evolution, the combination that yields its automation, quality, and consistency.

TABLE III: Architectural comparison of EvoML against agentic AutoML systems and command-line coding tools.
<table><tr><td>Method</td><td>Manual-Artifact- Free Input</td><td>Expert Grounding</td><td>Stage-Gated Validation</td><td>Cross-run Evolution</td></tr><tr><td>M³Builder</td><td>x (DataCard required)</td><td>√ (Fixed templates)</td><td>x (Shallow output check)</td><td>x</td></tr><tr><td>AutoML-Agent</td><td>x (Skeleton required)</td><td>x</td><td>x (Validation on pseudo execution)</td><td>x</td></tr><tr><td>MLZero</td><td>√</td><td>x (Document retrieval)</td><td>x (Execution only)</td><td>x</td></tr><tr><td>Codex</td><td>√</td><td>x</td><td>x (Execution only)</td><td>x</td></tr><tr><td>Claude Code</td><td>√</td><td>x</td><td>x (Execution only)</td><td>x</td></tr><tr><td>EvoML</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## B. Cost Against Human Expert Development

TABLE IV: Cost-efficiency comparison against expert leaderboard solutions.
<table><tr><td>Dataset</td><td>Human Leaderboard</td><td>EvoML</td></tr><tr><td>FLARE22</td><td>0.910</td><td>0.920</td></tr><tr><td>ACDC</td><td>0.936</td><td>0.919</td></tr><tr><td>BTCV</td><td>0.921</td><td>0.837</td></tr><tr><td>FLARE21</td><td>0.909</td><td>0.935</td></tr><tr><td>Promise12</td><td>0.919</td><td>0.920</td></tr><tr><td>OrganMNIST3D</td><td>0.996</td><td>0.992</td></tr><tr><td>FastMRI</td><td>0.963</td><td>0.950</td></tr><tr><td>Time</td><td>8-40 h</td><td>20-45 min</td></tr><tr><td>Cost</td><td>$800-$6,000</td><td>$10-$35</td></tr></table>

We compare EvoML against expert-engineered leaderboard solutions for the evaluated datasets. As shown in Table IV, EvoML surpasses the expert reference on FLARE21, FLARE22, and Promise12, remains close on OrganMNIST3D and FastMRI, but performs lower on ACDC and BTCV. Overall, EvoML achieves competitive model quality while substantially reducing synthesis effort and cost. The comparison underscores EvoML’s cost-efficiency: developing a leaderboard-level solution typically requires on the order of 8 to 40 hours of expert effort depending on task difficulty, excluding unattended model training, corresponding to roughly \$800 to \$6,000 in labour cost, whereas EvoML synthesises a pipeline in about 20–45 minutes at a generation cost of \$10 to \$35. It thus trades expensive, scarce expert time for cheap, fully automated synthesis while remaining competitive in quality.

## C. Threats to Validity

1. Internal validity: EvoML depends on the underlying foundation model. To assess this effect, we rerun the full framework with Claude Opus 4.7 on BTCV, FLARE22, and FastMRI. The resulting FMQ differs from GPT-5.5 by less than 0.008, suggesting that the main findings are not specific to one backbone. Since Opus 4.7 costs roughly 1.7× as much per pipeline, we use GPT-5.5 as the default backbone for comparable quality at a lower synthesis cost. 2. Construct validity: EvoML incurs longer model-training time because its nnU-Net grounding includes thorough preprocessing, augmentation, and five-fold cross-validation. On a typical supervised segmentation task, the baselines train in roughly 10 hours, EvoML in 60, and nnU-Net in 300. We consider this additional computation a trade-off for the stronger automation and model quality enabled by the grounded workflow. 3. External validity: We focus primarily on medical imaging because its pipelines rely on domain-specific processing and modelling choices, where mistakes can remain executable yet still degrade final model quality. We additionally evaluate three general ML tasks as initial evidence beyond this setting. For broader SE tasks, the same principles remain applicable: workflows can be decomposed into stages, recurring procedures can be encoded as skills, intermediate artifacts can be validated before downstream execution, and successful recoveries can be retained across tasks. Extending EvoML therefore mainly requires adapting the skill content and validation criteria.

## VI. CONCLUSION

We presented EvoML, an agentic framework for end-toend ML pipeline synthesis built on expert-grounded skills, validation-gated execution, and post-run skill evolution. By advancing each phase only when its outputs satisfy structural, execution, and semantic checks, EvoML limits error propagation, while post-run evolution consolidates successful recoveries into persistent skill updates that improve execution across subsequent tasks. Across eight medical-imaging datasets and three general ML tasks, EvoML achieved full automation and the strongest, most consistent model quality among the compared systems, while substantially reducing synthesis effort relative to expert development. Our additional evolution analysis further shows improved first-pass skill success, fewer validation-triggered retries, and cross-task reuse of retained adaptations. Future work will extend EvoML to broader ML and software-engineering tasks and investigate finer-grained validation throughout the pipeline to further improve model quality and reliability.

## VII. DATA AVAILABILITY

The artifacts that support the results discussed in this paper are available at: https://anonymous.4open.science/r/ EvoML-624B.

[1] Q. Yao, M. Wang, Y. Chen, W. Dai, Y.-F. Li, W.-W. Tu, Q. Yang, and Y. Yu, “Taking human out of learning applications: A survey on automated machine learning,” arXiv preprint arXiv:1810.13306, vol. 31, 2018.

[2] X. He, K. Zhao, and X. Chu, “Automl: A survey of the state-of-the-art,” Knowledge-based systems, vol. 212, p. 106622, 2021.

[3] M. Feurer, A. Klein, K. Eggensperger, J. Springenberg, M. Blum, and F. Hutter, “Efficient and robust automated machine learning,” Advances in neural information processing systems, vol. 28, 2015.

[4] R. S. Olson and J. H. Moore, “Tpot: A tree-based pipeline optimization tool for automating machine learning,” in Workshop on automatic machine learning. PMLR, 2016, pp. 66–74.

[5] I. Drori, Y. Krishnamurthy, R. Rampin, R. DE PAULA LOURENCO, J. Piazentin Ono, K. Cho, C. Silva, and J. Freire, “Alphad3m: Machine learning pipeline synthesis,” in ICML AutoML Workshop, 2021.

[6] R. Barbudo, S. Ventura, and J. R. Romero, “Eight years of automl: categorisation, review and trends,” Knowledge and Information Systems, vol. 65, no. 12, pp. 5097–5149, 2023.

[7] A. Alsharef, K. Aggarwal, Sonia, M. Kumar, and A. Mishra, “Review of ml and automl solutions to forecast time-series data,” Archives of computational methods in engineering, vol. 29, no. 7, pp. 5297–5311, 2022.

[8] M. Bahri, F. Salutari, A. Putina, and M. Sozio, “Automl: state of the art with a focus on anomaly detection, challenges, and research directions,” International Journal of Data Science and Analytics, vol. 14, no. 2, pp. 113–126, 2022.

[9] Y. Gu, H. You, J. Cao, M. Yu, H. Fan, and S. Qian, “Large language models for constructing and optimizing machine learning workflows: A survey,” ACM Trans. Softw. Eng. Methodol., Oct. 2025, just Accepted. [Online]. Available: https://doi.org/10.1145/3773084

[10] S. Hong, Y. Lin, B. Liu, B. Liu, B. Wu, C. Zhang, D. Li, J. Chen, J. Zhang, J. Wang et al., “Data interpreter: An llm agent for data science,” in Findings of the Association for Computational Linguistics: ACL 2025, 2025, pp. 19 796–19 821.

[11] D. Li, Z. Tan, and H. Liu, “Exploring large language models for feature selection: A data-centric perspective,” ACM SIGKDD Explorations Newsletter, vol. 26, no. 2, pp. 44–53, 2025.

[12] S. Malberg, E. Mosca, and G. Groh, “Felix: Automatic and interpretable feature engineering using llms,” in Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, 2024, pp. 230–246.

[13] Y. Shen, K. Song, X. Tan, D. Li, W. Lu, and Y. Zhuang, “Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face,” Advances in Neural Information Processing Systems, vol. 36, pp. 38 154–38 180, 2023.

[14] H. Li, H. Jiang, T. Zhang, Z. Yu, A. Yin, H. Cheng, S. Fu, Y. Zhang, and W. He, “Traineragent: Customizable and efficient model training through llm-powered multi-agent system,” arXiv preprint arXiv:2311.06622, 2023.

[15] T. Liu, N. Astorga, N. Seedat, and M. van der Schaar, “Large language models to enhance bayesian optimization,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=OOxotBmGol

[16] M. R. Zhang, N. Desai, J. Bae, J. Lorraine, and J. Ba, “Using large language models for hyperparameter optimization,” in NeurIPS 2023 Foundation Models for Decision Making Workshop, 2023. [Online]. Available: https://openreview.net/forum?id=n6z9v6Uyv0

[17] P. Trirat, W. Jeong, and S. J. Hwang, “AutoML-agent: A multi-agent LLM framework for full-pipeline AutoML,” in Proceedings of the 42nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaff, and J. Zhu, Eds., vol. 267. PMLR, 13–19 Jul 2025, pp. 60 099–60 146. [Online]. Available: https://proceedings.mlr.press/v267/trirat25a.html

[18] H. Fang, B. Han, N. Erickson, X. Zhang, S. Zhou, A. Dagar, J. Zhang, A. C. Turkmen, T. Hu, H. Rangwala et al., “Mlzero: A multi-agent system for end-to-end machine learning automation,” Advances in Neural Information Processing Systems, vol. 38, pp. 69 001–69 070, 2026.

[19] J. Feng, Q. Zheng, C. Wu, Z. Zhao, Y. Zhang, Y. Wang, and W. Xie, “M 3 builder: A multi-agent system for automated machine learning in medical imaging,” in International Workshop on Agentic AI for Medicine. Springer, 2025, pp. 115–124.

[20] OpenAI, “Introducing gpt-5.5.” [Online]. Available: https://openai.com/ index/introducing-gpt-5-5/

[21] Z. Manna and R. J. Waldinger, “Toward automatic program synthesis,” Communications of the ACM, vol. 14, no. 3, pp. 151–165, 1971.

[22] S. Gulwani, O. Polozov, and R. Singh, “Program synthesis,” Foundations and Trends in Programming Languages, vol. 4, no. 1-2, pp. 1–119, 2017.

[23] J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le et al., “Program synthesis with large language models,” arXiv preprint arXiv:2108.07732, 2021.

[24] E. Nijkamp, B. Pang, H. Hayashi, L. Tu, H. Wang, Y. Zhou, S. Savarese, and C. Xiong, “Codegen: An open large language model for code with multi-turn program synthesis,” ICLR, 2023.

[25] N. Jain, S. Vaidyanath, A. Iyer, N. Natarajan, S. Parthasarathy, S. Rajamani, and R. Sharma, “Jigsaw: Large language models meet program synthesis,” in Proceedings of the 44th International Conference on Software Engineering, 2022, pp. 1219–1231.

[26] J. Jiang, F. Wang, J. Shen, S. Kim, and S. Kim, “A survey on large language models for code generation,” ACM Transactions on Software Engineering and Methodology, vol. 35, no. 2, pp. 1–72, 2026.

[27] H. Wang, J. Gong, H. Zhang, J. Xu, and Z. Wang, “Ai agentic programming: A survey of techniques, challenges, and opportunities,” arXiv preprint arXiv:2508.11126, 2025.

[28] R. Sapkota, K. I. Roumeliotis, and M. Karkee, “Vibe coding vs. agentic coding: Fundamentals and practical implications of agentic ai,” arXiv preprint arXiv:2505.19443, 2025.

[29] S. Hong, M. Zhuge, J. Chen, X. Zheng, Y. Cheng, J. Wang, C. Zhang, S. Yau, Z. Lin, L. Zhou et al., “Metagpt: Meta programming for a multi-agent collaborative framework,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 23 247–23 275.

[30] C. Qian, W. Liu, H. Liu, N. Chen, Y. Dang, J. Li, C. Yang, W. Chen, Y. Su, X. Cong et al., “Chatdev: Communicative agents for software development,” in Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 1: Long papers), 2024, pp. 15 174–15 186.

[31] J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, and O. Press, “Swe-agent: Agent-computer interfaces enable automated software engineering,” Advances in Neural Information Processing Systems, vol. 37, pp. 50 528–50 652, 2024.

[32] Z. Zhao, J. Sun, Z. Hou, Z. Wei, C.-H. Cai, M. Qiao, and J. S. Dong, “Mactg: Multi-agent collaborative thought graph for automatic programming,” arXiv preprint arXiv:2410.19245, 2024.

[33] “Github copilot · your ai pair programmer,” 1 2026. [Online]. Available: https://github.com/features/copilot

[34] C. Team, “Cursor: The best coding agent,” 5 2026. [Online]. Available: https://cursor.com/

[35] OpenAI, “Introducing codex.” [Online]. Available: https://openai.com/ index/introducing-codex/

[36] Anthropic, “Claude code — anthropic’s agentic coding system.” [Online]. Available: https://www.anthropic.com/product/claude-code

[37] N. Hollmann, S. Muller, and F. Hutter, “Large language models for¨ automated data science: Introducing caafe for context-aware automated feature engineering,” Advances in Neural Information Processing Systems, vol. 36, pp. 44 753–44 775, 2023.

[38] S. Guo, C. Deng, Y. Wen, H. Chen, Y. Chang, and J. Wang, “DSagent: Automated data science by empowering large language models with case-based reasoning,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235. PMLR, 2024, pp. 16 813–16 848.

[39] S. Zhang, C. Gong, L. Wu, X. Liu, and M. Zhou, “Automl-gpt: Automatic machine learning with gpt,” arXiv preprint arXiv:2305.02499, 2023.

[40] Y. Chi, Y. Lin, S. Hong, D. Pan, Y. Fei, G. Mei, B. Liu, T. Pang, J. Kwok, C. Zhang et al., “Sela: Tree-search enhanced llm agents for automated machine learning,” arXiv preprint arXiv:2410.17238, 2024.

[41] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, “Optuna: A next-generation hyperparameter optimization framework,” in The 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2019, pp. 2623–2631.

[42] Anthropic, “Equipping agents for the real world with agent skills,” October 2025. [Online]. Available: https://www.anthropic.com/ engineering/equipping-agents-for-the-real-world-with-agent-skills

[43] F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnu-net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature methods, vol. 18, no. 2, pp. 203–211, 2021.

[44] Anthropic, “Introducing the model context protocol,” November 2024. [Online]. Available: https://www.anthropic.com/news/ model-context-protocol

[45] J. S. Chan, N. Chowdhury, O. Jaffe, J. Aung, D. Sherburn, E. Mays, G. Starace, K. Liu, L. Maksin, T. Patwardhan et al., “Mle-bench: Evaluating machine learning agents on machine learning engineering,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 50 466–50 494.

[46] B. Landman, Z. Xu, J. Igelsias, M. Styner, T. Langerak, and A. Klein, “Miccai multi-atlas labeling beyond the cranial vault–workshop and challenge,” in Proc. MICCAI multi-atlas labeling beyond cranial vault—workshop challenge, vol. 5. Munich, Germany, 2015, p. 12.

[47] O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, X. Yang, P.-A. Heng, I. Cetin, K. Lekadir, O. Camara, M. A. G. Ballester et al., “Deep learning techniques for automatic mri cardiac multi-structures segmentation and diagnosis: is the problem solved?” IEEE transactions on medical imaging, vol. 37, no. 11, pp. 2514–2525, 2018.

[48] J. Ma, Y. Zhang, S. Gu, X. An, Z. Wang, C. Ge, C. Wang, F. Zhang, Y. Wang, Y. Xu et al., “Fast and low-gpu-memory abdomen ct organ segmentation: the flare challenge,” Medical Image Analysis, vol. 82, p. 102616, 2022.

[49] J. Ma, Y. Zhang, S. Gu, C. Ge, S. Mae, A. Young, C. Zhu, X. Yang, K. Meng, Z. Huang et al., “Unleashing the strengths of unlabelled data in deep learning-assisted pan-cancer abdominal organ quantification: the flare22 challenge,” The Lancet Digital Health, vol. 6, no. 11, pp. e815– e826, 2024.

[50] Z. Kus¸ and M. Aydin, “Medsegbench: A comprehensive benchmark for medical image segmentation in diverse data modalities,” Scientific Data, vol. 11, no. 1, p. 1283, 2024.

[51] G. Litjens, R. Toth, W. Van De Ven, C. Hoeks, S. Kerkstra, B. Van Ginneken, G. Vincent, G. Guillard, N. Birbeck, J. Zhang et al., “Evaluation of prostate segmentation algorithms for mri: the promise12 challenge,” Medical image analysis, vol. 18, no. 2, pp. 359–373, 2014.

[52] J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister, and B. Ni, “Medmnist v2-a large-scale lightweight benchmark for 2d and 3d biomedical image classification,” Scientific Data, vol. 10, no. 1, p. 41, 2023.

[53] J. Zbontar, F. Knoll, A. Sriram, T. Murrell, Z. Huang, M. J. Muckley, A. Defazio, R. Stern, P. Johnson, M. Bruno et al., “fastmri: An open dataset and benchmarks for accelerated mri,” arXiv preprint arXiv:1811.08839, 2018.

[54] B. I. A. Group, “Ixi dataset.” [Online]. Available: https: //brain-development.org/ixi-dataset/

[55] A. P. Zijdenbos, B. M. Dawant, R. A. Margolin, and A. C. Palmer, “Morphometric analysis of white matter lesions in mr images: method and validation,” IEEE transactions on medical imaging, vol. 13, no. 4, pp. 716–724, 1994.

[56] J. Bertels, T. Eelbode, M. Berman, D. Vandermeulen, F. Maes, R. Bisschops, and M. B. Blaschko, “Optimizing the dice score and jaccard index for medical image segmentation: Theory and practice,” in International conference on medical image computing and computer-assisted intervention. Springer, 2019, pp. 92–100.

[57] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.

[58] A. Krizhevsky, V. Nair, G. Hinton et al., “The cifar-10 dataset,” 2014.

[59] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, “The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results,” http://www.pascalnetwork.org/challenges/VOC/voc2012/workshop/index.html.

[60] “Ag’s corpus of news articles.” [Online]. Available: http://groups.di. unipi.it/<sup>∼</sup>gulli/AG corpus of news articles.html