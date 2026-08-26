# StrokeGuard: A Multi-Agent Guided System for Prehospital Stroke Assessment

Wentao Yang

Zhenye Xu

Ruoyi Li

Institute of Medical Robotics

School of Biomedical Engineering

Institute of Medical Robotics

Institute of Medical Robotics

Shanghai Jiao Tong University

School of Biomedical Engineering

School of Biomedical Engineering

Shanghai, China

Shanghai Jiao Tong University

tao615@sjtu.edu.cn

Shanghai Jiao Tong University

School of Biomedical Engineering

Shanghai, China

Shanghai, China

Shanghai Jiao Tong University

musen zhang@sjtu.edu.cn

Musen Zhang

Institute of Medical Robotics

Shanghai, China

lingyy1109@gmail.com

rowenalee@sjtu.edu.cn

Yao Guo

Institute of Medical Robotics

School of Biomedical Engineering

Shanghai Jiao Tong University

Shanghai, China

yao.guo@sjtu.edu.cn

Abstract—Prehospital stroke assessment aims to accurately identify stroke symptoms and make rapid decisions through standardized procedures within an extremely narrow time window, thereby saving valuable time for subsequent treatment. In clinical practice, FAST-based scales are widely used for prehospital stroke assessment by issuing instructions that guide subjects to perform specific actions to screen facial, arm, and speech functions. However, in home and community settings, non-clinical users often encounter challenges such as inaccurate descriptions, incomplete symptom observation, and difficult operational procedures, which may lead to inaccurate or biased assessment results. To address these challenges, this paper presents StrokeGuard—a multi-agent guided system designed for prehospital stroke assessment that makes mobile FAST screening more standardized and executable. Specifically, to overcome the limitations of traditional singleagent systems in terms of procedural fault tolerance and user guidance capability, StrokeGuard adopts a dual-channel agent mechanism that separates formal assessment (i.e.., facial palsy, arm weakness, speech impairment) from procedural support (e.g., step prompts, error correction, and real-time feedback). It guides the assessment process through multi-agent collaboration, dual-channel interaction, state-machine control, and stage-local fallback recovery mechanisms. Stage-specific scoring is delegated to constrained pretrained video assessment modules, while evidence source records are integrated with structured report generation. The user evaluation uses MATES-9, an exploratory scale for measuring user experience in multistep AI-guided tasks. In a simulated prehospital scenario, StrokeGuard improves the MATES-9 total score over a paper FAST-style form by 10.83 points, corresponding to a 23.8% relative increase.

Index Terms—Prehospital stroke assessment, multi-agent systems, Face Arm Speech Test, human-computer interaction

## I. INTRODUCTION

Stroke is an acute neurological event caused by a sudden interruption of cerebral blood flow or rupture of a cerebral vessel. Because brain injury progresses rapidly, prehospital stroke assessment must identify symptoms accurately and sup port rapid decisions through standardized procedures within an extremely narrow time window. Prehospital screening tools such as the Cincinnati Prehospital Stroke Scale (CPSS) and the Face Arm Speech Test (FAST) reduce stroke recognition to observable facial, arm, and speech signs [1]–[3]. In clinical practice, FAST-based scales guide subjects to perform specific actions so that facial, arm, and speech functions can be screened in a structured manner.

The apparent simplicity of these tools does not directly translate into reliable execution by low-expertise users. In home and community settings, non-clinical users often face inaccurate descriptions, incomplete symptom observation, and difficult operational procedures. Prehospital studies report clear limits in stroke recognition accuracy even when dispatchers and emergency medical personnel participate [9], [10]. Barriers to timely activation of emergency medical services also include difficulty interpreting symptoms, insufficient judgment of severity, and uncertainty about appropriate actions [11]. When FAST-style screening moves from trained personnel to ordinary caregivers, a system must support more than recognition of the three FAST categories. It must also provide procedural guidance, support multimodal evidence collection under time pressure, and preserve the current assessment step after capture failure or workflow interruption.

Recent multimodal stroke systems have advanced perception and classification. DeepStroke uses multimodal adversarial learning to support emergency-room stroke screening [4]. M3 Stroke studies mobile multimodal AI for mild-to-moderate acute stroke triage [5]. Digital FAST explores a multimodal framework aligned with the FAST paradigm [6]. VOICE demonstrates the feasibility of a voice AI agent for simulated prehospital stroke assessment [8]. These studies show that AI can support early triage, but they do not fully address the interaction and handoff problems that arise before a model receives usable input. An assessment model alone cannot guide users through recording failures or preserve workflow order. A general conversational agent can provide flexible dialogue, but it does not by itself guarantee controlled scoring, adequate multimodal input quality, or an auditable handoff for clinicians.

![](images/7f22251b501a930788e197a0603a89edd508f5a09b3de7cd84b49d26491c25dc.jpg)  
Fig. 1. High-level overview of StrokeGuard in a simulated prehospital assessment workflow.

This execution gap is also distinct from the problems addressed by specialized prehospital solutions. Mobile stroke units show that richer prehospital data collection can improve triage and reduce door-to-needle time, but these systems rely on trained onboard personnel rather than ordinary caregivers at the scene [12]. Dispatcher recognition algorithms and structured call-taking protocols have been used to improve telephone-based stroke screening, but they target trained dispatchers rather than direct interactive guidance for untrained on-site users [13].

To address these challenges, this paper presents StrokeGuard, a multi-agent guided system designed for prehospital stroke assessment that makes mobile FAST screening more standardized and executable. To overcome the limitations of traditional single-agent systems in procedural fault tolerance and user guidance capability, StrokeGuard adopts a dual-channel agent mechanism that separates formal assessment of facial palsy, arm weakness, and speech impairment from procedural support, including step prompts, error correction, and real-time feedback. It guides the assessment process through multi-agent collaboration, dual-channel interaction, state-machine control, and stagelocal fallback recovery. Stage-specific scoring is delegated to constrained pretrained video assessment modules, while evidence-source records are integrated with structured report generation.

## This paper makes three contributions:

• A prehospital stroke assessment agent system. StrokeGuard reframes FAST-style screening for lowexpertise users as a multi-agent assessment execution problem and places formal assessment, procedural support, stage progression, local fallback, and sourcepreserving handoff within explicit responsibility boundaries.

• A controlled execution architecture for assessment agents. The architecture separates agent interaction, deterministic orchestration, and dedicated scoring modules, allowing stage-specific modules to be replaced by different stroke recognition algorithms and supporting transfer to other clinical screening tasks with fixed procedures or multiple assessment items.

• An evaluation design for controlled assessment agents. The paper uses mechanism-ablation case studies to examine the roles of dual-channel constraints, evidenceaware orchestration, and stage-local recovery, and proposes MATES-9 as an exploratory tool for evaluating the experience of multistep AI-guided tasks.

## II. RELATED WORK

## A. Multimodal AI Algorithms for Stroke Assessment

Prior multimodal stroke-screening studies show that facial video, speech, and limb motion can support automatic identification of stroke-related signs. DeepStroke, M3 Stroke, and Digital FAST validate automatic stroke screening from face and speech cues, audiovisual temporal features, and FASTcomponent modeling, respectively, and report strong sensitivity, accuracy, or F1 scores [4]–[6]. Sensor-based work has also explored stroke-related functional assessment, including balance-function assessment with a single ear-worn IMU [7].

Taken together, this line of work establishes facial dynamics, arm motion, speech signals, and wearable sensing as useful evidence streams for stroke-related assessment. However, existing studies mainly examine feature extraction, modality fusion, and classification or functional-estimation performance, often assuming that input of acceptable quality is already available. StrokeGuard does not retrain a stroke classifier. It wraps pretrained Face, Arm, and Speech assessment algorithms as constrained modules and addresses input availability through staged capture guidance, video-first processing, and singlestage text fallback, while limiting the effect of capture failure on the full assessment process.

## B. Clinical Healthcare Agents

Clinical healthcare agents target specific clinical or health tasks by combining language models with context maintenance, planning, tool use, memory, and feedback [16], [17]. In medicine, these agents have been used for diagnostic assistance, clinical decision support, documentation, patient communication, and workflow management. Their actions, however, must remain constrained by clinical authority, human supervision, safety rules, and evidence-tracing mechanisms [16], [17]. Medical conversational agents also support health education, symptom inquiry, and person-centered care through natural language [14], [15], with evaluation concerns extending to factual reliability, clinical safety, task completion, and human-AI collaboration in real contexts [14]–[17]. When clinicians review agent outputs, structured handoff and trust calibration are also important: structured handoff tools reduce omissions of key information [39], while adoption of clinical AI depends on task-relevant explanation, context, and uncertainty [40]. A prehospital agent system should therefore preserve input sources, processing states, and result-formation processes in addition to conclusions.

## C. Multi-Agent Systems

LLM-based multi-agent systems provide an engineering paradigm for role division and collaborative execution. MetaGPT and ChatDev show that the value of multi-agent systems lies not only in the number of models but also in the organization of tasks, roles, communication paths, and intermediate artifacts [35], [36]. For prehospital stroke assessment, this view suggests that input routing, formal assessment guidance, and procedural support can be assigned to different roles while clinical authority, safety rules, and evidence-tracing constraints remain explicit [16], [17].

Role division alone is insufficient for reliable medical assessment. A system must specify which inputs can change assessment state, how state changes are recorded, which stage a media result belongs to, and whether a local failure affects completed stages. Workflow architecture provides related design cues: CQRS, Event Sourcing, Saga, and Bulkhead respectively motivate write boundaries, event traces, multistep compensation, and fault isolation [28]–[31]. Task-oriented dialogue research also shows that natural-language systems need explicit task state and a separation between general dialogue behavior and domain task control [18], [32]–[34]. These issues are sharper in prehospital stroke assessment, where caregivers may face symptom interpretation, action decisions, nonprofessional operation, and emergency-response difficulties [9]–[11], while medical conversational agents must also satisfy contextual adaptation, user interaction, and safety requirements [14], [15]. StrokeGuard therefore brings multi-agent role division, workflow state boundaries, and task-oriented dialogue control into prehospital stroke assessment through dual-channel agent interaction, evidence-aware orchestration, stage-local recovery, and source-traceable handoff.

## D. Human-Computer Interaction Evaluation in Medical Systems

Studies of clinical decision support and prehospital medical technology commonly evaluate task outcomes together with efficiency, cognitive burden, and usability [19]–[21]. Existing scales provide useful but incomplete construct sources: SUS, NASA-TLX, MAUQ, and Health-ITUES cover usability, workload, mobile-health experience, and health-informationtechnology experience [22]–[25], while CUQ and BUS cover chatbot and conversational-interface usability [26], [27]. MATES-9 is designed as an exploratory tool for multistep AIguided tasks by adapting these constructs to measure task guidance and step awareness, information clarity, support continuity, perceived workload, and action-decision support.

## III. STROKEGUARD SYSTEM DESIGN

## A. Overall Architecture

StrokeGuard combines natural-language agents with deterministic flow control in the layered architecture shown in Fig. 2. The User Interface Layer presents the current stage, capture action, and supportive communication to the caregiver. The Multi-Agent Coordination Layer routes input and handles natural-language interaction while coordinating deterministic assessment-state control. The Assessment and Audit Layer calls dedicated Face, Arm, and Speech modules and stores evidence sources and processing traces.

TABLE I  
AGENT ROLES AND OUTPUTS IN STROKEGUARD ARCHITECTURE
<table><tr><td rowspan=1 colspan=1>Agent</td><td rowspan=1 colspan=1>Roles and output</td></tr><tr><td rowspan=1 colspan=1>Dual-channel   decisionagent</td><td rowspan=1 colspan=1>Routes input to assessment or support; out-puts channel label, target role, and routingrecord.</td></tr><tr><td rowspan=1 colspan=1>Assessment agent</td><td rowspan=1 colspan=1>Guides current-stage capture; outputs mediasubmission, text fallback request, comple-tion, waiting, or failure event.</td></tr><tr><td rowspan=1 colspan=1>Support agent</td><td rowspan=1 colspan=1>Answers operational, explanatory, reassur-ing, and emergency-response questions; out-puts support reply without state change.</td></tr></table>

## B. Dual-Channel Interaction and Assessment-State Separation

The dual-channel mechanism addresses confusion that can arise when dialogue state and clinical state share the same control path. In a single-channel design, a statement such as the left arm of the patient is dropping and a question such as how should I hold the phone can enter the same conversational context, and the same model may generate both the reply and the next action. The former may constitute assessment evidence, whereas the latter should only help the user continue capture. Task-oriented dialogue research usually constrains later actions through explicit state tracking and task control [18], [32]–[34]. Prehospital assessment also needs a clear distinction between inputs that can affect a clinical stage and inputs that cannot. Without this distinction, a shared control path may cause repeated prompts, stage jumps, evidence misclassification, or unclear evidence sources.

StrokeGuard separates these two types of input through the dual-channel decision agent and sends them to different agent channels. The assessment-channel agent produces formal events that can be verified by the orchestrator, including media submission, text fallback completion, processing failure, and current-stage completion. The support-channel agent reads the same session context to keep replies relevant, but it has no permission to write assessment evidence or advance the assessment stage. It produces only natural-language replies and no state-transition events. Users can therefore ask support questions at any time and receive answers without disrupting assessment-stage progression.

This mechanism can be expressed as a routing function with permission constraints. Let the user input be $x _ { t }$ , the session context be $c _ { t } ,$ , and the clinical assessment state be $s _ { t }$ . The dual-channel decision agent outputs the routing label

$$
r _ { t } = R ( x _ { t } , c _ { t } ) , \qquad r _ { t } \in \{ \mathrm { a s s e s s m e n t } , \mathrm { s u p p o r t } \} .\tag{1}
$$

![](images/fecf0835038eea68f5b24795c1d4543b60238f01189fdd8cafe01f4007bf3731.jpg)  
Fig. 2. System architecture of StrokeGuard. User inputs are routed to assessment or support channels, deterministic orchestration controls stage progression and fallback, and stage-specific assessment modules provide source-preserving outputs for clinician review.

The two channels have different permission sets:

$$
\begin{array} { r l } & { \mathcal { P } ( \mathrm { a s s e s s m e n t } ) = \{ \mathrm { c r e a t e \_ e v e n t } , \mathrm { w r i t e \_ e v i d e n c e } , } \\ & { \quad \quad \mathrm { r e q u e s t \_ t r a n s i t i o n } \} , } \\ & { \mathcal { P } ( \mathrm { s u p p o r t } ) = \{ \mathrm { r e a d \_ c o n t e x t } , \mathrm { g e n e r a t e \_ r e p l y } \} . } \end{array}\tag{2}
$$

State update occurs only when the assessment event $e _ { t }$ passes orchestration validation:

$$
s _ { t + 1 } = { \left\{ \begin{array} { l l } { T ( s _ { t } , e _ { t } ) , } & { r _ { t } = { \mathrm { a s s e s s m e n t } } \wedge V ( e _ { t } , s _ { t } ) = 1 , } \\ { s _ { t } , } & { r _ { t } = { \mathrm { s u p p o r t } } \vee V ( e _ { t } , s _ { t } ) = 0 . } \end{array} \right. }\tag{3}
$$

Here, $V ( e _ { t } , s _ { t } )$ denotes checks for stage consistency, completion conditions, and evidence conflicts. The equation captures the core constraint of the dual-channel design: the support channel can generate replies but cannot change clinical state.

The dual-channel decision agent sits between user input and the two task channels. It converts natural-language input into a routing result with permission semantics. If the input contains formal assessment intent, such as capture submission, stage completion, or fallback confirmation, the system sends it to the assessment channel, and the orchestrator further checks whether evidence can be written or the stage can advance. If the input asks about camera placement, action explanation, reassurance, or emergency response, the system sends it to the support channel, generates a reply, and does not trigger a state update. The system therefore performs channel selection rather than asking caregivers to decide, under pressure, whether a statement has clinical-evidence status. Each routing decision stores the input, channel label, processing result, and state effect, so that later audit can explain why an interaction changes or does not change assessment state.

## C. Evidence-Aware Orchestration and Stage-Local Recovery

The dual-channel mechanism protects the evidence-writing boundary, namely what can be written. Evidence-aware orchestration protects stage attribution and evidence level during assessment: which stage receives a write, which source supports it, and whether overwriting is allowed. The orchestrator advances the task through a finite-state machine consisting of patient information collection, Face, Arm, Speech, processing, and done. Each stage has an independent evidence slot, processing state, and recovery path. The orchestrator allows transition to the next stage only after usable evidence is obtained for the current stage or local fallback is completed according to predefined rules. Stage progression is therefore triggered by the evidence state of the current stage, and user interjections or information about other stages do not change the attribution of the stage being executed.

StrokeGuard adopts a video-first but not video-dependent evidence strategy. Video preserves more direct clinical cues, including facial movement, arm posture, and speech performance, and is therefore used as the preferred evidence when available. However, network instability, limited camera angles, poor patient cooperation, or device failure in prehospital environments may cause media capture to fail. Text fallback handles these local failures. When usable video or audio cannot be obtained for the current stage, the system allows the caregiver to describe observations for that stage in structured text so that the assessment process can continue, while the record explicitly marks the lower evidence level and fallback reason. Text fallback is therefore limited to stage-level compensation that prevents one media failure from interrupting the whole assessment. Video assessment always has higher evidence priority when available. Video-source results, text fallback results, and unfinished states form a strict priority order: video evidence can replace temporary text evidence, while text evidence cannot overwrite completed video evidence. This rule ensures that evidence is updated only toward a more reliable source and prevents later lowpriority text input from overwriting high-priority evidence. For asynchronous external assessment results, the system checks session, stage, and task identity before writing and handles duplicate callbacks idempotently.

Let the stage set be $\kappa = .$ {Face, Arm, Speech} and the current stage be $k _ { t }$ . For stage k, the evidence slot is denoted as ${ { z } _ { k } } = ( { { y } _ { k } } , { { m } _ { k } } , { { q } _ { k } } , { { \tau } _ { k } } )$ , where $y _ { k }$ is the stage conclusion, $m _ { k } \in \{ \mathrm { p e n d i n g } , \mathrm { t e x t } .$ , video} is the evidence modality, $q _ { k }$ is the processing state, and $\tau _ { k }$ is temporal information. Evidence priority increases from pending to text and then to video. When new evidence $\hat { z } _ { k }$ arrives, the system applies a monotonic update rule:

$$
z _ { k } ^ { t + 1 } = \left\{ \begin{array} { l l } { \hat { z } _ { k } , } & { \mathrm { p r i o } ( \hat { m } _ { k } ) \geq \mathrm { p r i o } ( m _ { k } ^ { t } ) \land I ( \hat { z } _ { k } , s _ { t } ) = 1 , } \\ { z _ { k } ^ { t } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{4}
$$

where $\hat { m } _ { k }$ is the modality of the new evidence, and $I ( \hat { z } _ { k } , s _ { t } )$ denotes session, stage, and task identity checks. Stage progression depends only on whether the current stage reaches completion:

$$
k _ { t + 1 } = { \left\{ \begin{array} { l l } { { \mathrm { n e x t } } ( k _ { t } ) , } & { C ( z _ { k _ { t } } ^ { t + 1 } ) = 1 , } \\ { k _ { t } , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{5}
$$

Stage-local recovery restricts multimodal-link failures to the current step. When recording or processing fails in a stage, the system first keeps the current stage active and allows rerecording. If media cannot continue to be used, text fallback is enabled only for that stage. Completed stages are not rolled back, and later stages do not inherit the failure state of the current stage. Compared with treating one media failure as failure of the entire session, this design turns a sessionlevel failure into stage-level compensation while preserving the priority of video evidence and maintaining assessment continuity.

If a failure $\epsilon _ { t }$ occurs at the current stage, the local recovery function acts only on $k _ { t } \colon$

$$
\begin{array} { r l } & { ( \rho _ { t } , z _ { k _ { t } } ^ { t + 1 } ) = F ( k _ { t } , \epsilon _ { t } , z _ { k _ { t } } ^ { t } ) , } \\ & { ~ \quad \quad \quad \rho _ { t } \in \{ \mathrm { r e t r y } , \mathrm { w a i t } , \mathrm { t e x t \_ f a l l b a c k } \} . } \end{array}\tag{6}
$$

For every noncurrent stage $j \neq k _ { t }$

$$
z _ { j } ^ { t + 1 } = z _ { j } ^ { t } .\tag{7}
$$

The failure therefore neither rolls back completed stages nor contaminates later stages.

## D. Source-Preserving Structured Handoff

StrokeGuard treats handoff generation as a structured projection of the assessment execution record. Instead of presenting only the final stage label, the system binds each conclusion to its evidence modality, source, processing state, fallback status, and event history. This design allows the handoff report to preserve how each result was formed and what level of evidence supports it.

Source preservation occurs when stage results are generated. For each assessment result, the system stores three types of information. Result fields record the stage conclusion. Source fields record the evidence modality, concrete source, quality state, and fallback reason. Event fields record key processes such as routing, submission, failure, fallback, callback, and stage progression. Immediate results and delayed results enter the same stage record and are merged by deterministic rules. The open-ended model does not reinterpret these results. Clinicians can therefore distinguish a video-based abnormality from a text-fallback abnormality after media failure, because the two labels carry different evidence bases. This organization shifts trust judgment from a single output label to a reviewable evidence chain.

The final handoff report is assembled by a fixed template and presents each conclusion with its source trace. This structure makes system behavior reconstructable. When the final result conflicts with on-site observation, the record shows which channel an input entered, when the stage advanced, whether fallback occurred, and which event produced the current result. StrokeGuard therefore delivers prehospital evidence and the formation process of that evidence for clinician review, allowing clinicians to judge evidence reliability, locate disagreement sources, and decide whether recapture or further examination is needed.

## E. Prototype Implementation

The StrokeGuard prototype supports a complete prehospital assessment. The prototype includes a caregiver-facing mobile and web interaction application, a backend orchestration service for stage progression and evidence writing, and supporting services connected to stage-specific assessment algorithms, text fallback paths, and structured handoff generation.

## IV. EXPERIMENTS AND RESULTS

## A. Case Study: Mechanism-Ablation Comparison

The case study examines typical failure modes of openended agents in prehospital FAST assessment and clarifies which system risks are addressed by the control mechanisms of StrokeGuard. The design uses mechanism ablation rather than attributing all failures to a single-channel language model. Each comparison retains the dialogue and task-generation ability of an open-ended language model while removing one control mechanism of StrokeGuard.

![](images/d3a7f675c3f000b5842245d3f518e68ffb21385d58b72b7e011c3ba524a7f440.jpg)

The three cases correspond to three mechanism ablations. Case A removes the dual-channel constraint, Case B removes evidence-aware orchestration, and Case C removes stage-local recovery. Tables II–IV present each comparison with a shared elicitation prefix followed by two trajectories: the ablated open-ended agent and StrokeGuard under the same input.

Together, these cases show that StrokeGuard constrains three risks in prehospital assessment: support requests cannot change clinical state, evidence must be written to the correct stage, and local failures cannot roll back or contaminate completed stages. The case study provides mechanistic and explanatory evidence; it does not estimate causal effects statistically.

## B. MATES-9 Scale Design and Simulated Prehospital Evaluation Results

MATES-9 is used to assess the user experience of multistep AI-guided task execution in the simulated prehospital assessment. The scale is constructed by adapting and integrating constructs from SUS, NASA-TLX, MAUQ, Health-ITUES, ASQ/PSSUQ/CSUQ, CUQ, BUS, and TAM [22]–[27], [37], [38]. Table V shows the item wording, scoring direction, target dimension, construct source, and corresponding objective criterion. MATES-9 focuses on five aspects that are central to StrokeGuard: task guidance and step awareness, information clarity, support continuity, perceived workload, and actiondecision support. It is used as an exploratory task-experience measure and is interpreted together with task logs, completion outcomes, and open-ended feedback.

The MATES-9 items are written to keep each judgment focused. An item does not combine separate questions about whether help is useful, whether help interrupts the task, whether rework is needed, or whether the user feels confused. The scale retains dimensions that are directly relevant to multistep AI-guided tasks, including task completion, step orientation, support continuity, workload, result understandability, and action-decision support. Q4, Q5, Q6, and Q7 use reverse wording to reduce acquiescence bias from allpositive items. MATES-9 is a subjective experience measure: its items are traced to source constructs, but its scores must be interpreted together with task logs, completion outcomes, and open-ended feedback rather than as stand-alone evidence of system effectiveness.

All items use a 7-point Likert scale, where 1 means strongly disagree and 7 means strongly agree [41]. Reverse items are first converted to the same direction using $Q _ { j } ^ { r } = 8 - Q _ { j }$ for $j = 4 , \dots , 7$ . This paper reports raw means and transformed 0-100 scores computed as $( m e a n - 1 ) / 6 \times 1 0 0$ ; Table VI summarizes the domain mapping.

To obtain preliminary user-experience evidence, this paper constructs a simulated prehospital onset scenario. A professional male actor with an apparent age of approximately 70 years plays the stroke patient and simulates FAST-related symptoms. The scenario identifies the patient as an elderly relative of the participant, so that the participant needs to complete prehospital stroke assessment in the context of a

Fig. 3. MATES-9 results in the simulated prehospital assessment. (a) Total score distribution with individual scores, interquartile ranges, group means, and half-violin densities. (b) Transformed 0-100 domain scores.

sudden abnormal event involving a family member. Twelve participants complete the experiment. Their ages range from 20 to 39 years, with six male and six female participants. Participants are divided into two groups and complete the assessment in the same simulated scenario using either a traditional paper FAST-style assessment form or StrokeGuard. After the task, all participants complete MATES-9.

Fig. 3 summarizes the MATES-9 results. Panel (a) shows the total-score distribution with individual participant scores, and Panel (b) shows transformed 0-100 domain scores. Because the sample is small, inferential statistics are used only as exploratory evidence.

The StrokeGuard group reports a higher MATES-9 total score than the paper-form group, $5 6 . 3 3 { \pm } 5 . 6 1$ versus $4 5 . 5 0 { \pm } 8 . 2 2 $ , corresponding to a 10.83-point absolute increase and a 23.8% relative increase. An exploratory Welch independent-samples t-test indicates a group difference, $t ( 8 . 8 3 ) ~ = ~ 2 . 6 7 , ~ p ~ = ~ 0 . 0 2 6$ , with a large small-samplecorrected effect size, Hedges $g = 1 . 4 2$ . On the transformed 0-100 scale, the total score increases from 67.6 to 87.7. As shown in Fig. 3, the largest domain gains appear in low workload and support continuity, which is consistent with the intended role of StrokeGuard in preserving step awareness, support access, and coordinated task execution. Mean assessment time also decreases from 116 s to 81 s, a 30.2% reduction.

## V. CONCLUSIONS AND DISCUSSION

This paper introduces StrokeGuard, a multi-agent guided system for prehospital stroke assessment that makes mobile FAST screening more standardized and executable. StrokeGuard organizes natural-language interaction, dedicated multimodal assessment modules, and deterministic state control into a unified assessment process, allowing low-expertise caregivers to complete a reviewable assessment when help seeking, recording, failure recovery, and result handoff are intertwined.

StrokeGuard contributes controlled multi-agent assessment, evidence-aware execution, and source-preserving handoff. Dual-channel decision-making, formal assessment guidance, and procedural support are assigned to different agents, while clinical-state writing remains under deterministic orchestration. Stage identity, task identity, evidence source, and overwrite priority keep stage-outside symptoms, asynchronous media results, and text fallback input in their corresponding evidence slots. Stage conclusions, evidence modality, processing state, and fallback process are organized as structured handoff content, and MATES-9 is proposed as an exploratory experience measure for multistep AI-guided tasks. StrokeGuard remains a research prototype intended to complement emergency medical services. Future work should validate the assessment modules, quality-control thresholds, MATES-9, and mechanism-ablation effects in larger and more realistic settings, and should study integration with dispatch, electronic health record, and hospital receiving systems.

TABLE II  
CASE A: SHARED ELICITATION WITH AND WITHOUT DUAL-CHANNEL CONSTRAINT
<table><tr><td rowspan=1 colspan=3>Shared elicitation. ① SYsTEM: Next, please record the face of the patient. Ask him to look at the camera and smile as widely as possible. The current stage is Face capture.② UsER: I am a bit anxious. If one side of his mouth looks crooked when he smiles, is that very serious? How far should I hold the phone so that the video is clear?</td></tr><tr><td rowspan=1 colspan=1>Trajectory</td><td rowspan=1 colspan=1>Ablated open-ended agent</td><td rowspan=1 colspan=1>StrokeGuard</td></tr><tr><td rowspan=1 colspan=1>Dialogue after theshared prefix</td><td rowspan=1 colspan=1>③ AGENT: One-sided mouth droop may indicate facial asymmetry andstroke risk. I will first record it as a Face abnormality. Next, pleaseobserve whether his arms are weak.④ UsER: Wait, he has not smiled yet. I was only asking what wouldhappen if the mouth is crooked.⑤ AGENT: Understood. Let us record the face video again. Please askthe patient to look at the camera and smile.⑥ UsER: Was he already marked as having facial droop? I am worriedthat the report is wrong.</td><td rowspan=1 colspan=1>③ DUAL-CHANNEL DECISION AGENT: Routes the input to the supportchannel and creates no Face assessment event.④ SUPPORT AGENT: Keep the phone facing the face of the patient andmake sure the whole face is visible. Obvious mouth asymmetry can berelevant, but the system will not record a result from a hypotheticalquestion.⑤ AsSESSMENT AGENT: Please ask the patient to look at the cameraand smile as widely as possible, then submit the face video.⑥ UsER: OK, I will ask him to smile and upload the face video now.⑦ ORCHESTRATOR: Receives the Face video and starts Face processing;Arm can begin only after Face is completed or locally downgraded.</td></tr><tr><td rowspan=1 colspan=1>Mechanism exposed</td><td rowspan=1 colspan=1>A help request is written as clinical evidence and causes premature stageprogression.</td><td rowspan=1 colspan=1>The support interaction is answered without changing Face evidence orstage state.</td></tr></table>

TABLE III  
CASE B: SHARED ELICITATION WITH AND WITHOUT EVIDENCE-AWARE ORCHESTRATION
<table><tr><td colspan="3">Shared elicitation. ① SYsTEM: Next, please first record the face of the patient. Ask him to look at the camera and smile as widely as possible. The current stage is Face. ② UsER: I just noticed that his left hand suddenly became weak, and the cup fell from his hand. I recorded that he could not lift his arm. Should I show you this first?</td></tr><tr><td>Trajectory Dialogue after</td><td>Ablated open-ended agent ③ AGENT: Left-hand weakness and dropping the cup suggest an upper-</td><td>StrokeGuard ③ DUAL-CHANNEL DECISION AGENT: Marks the left-hand weakness</td></tr><tr><td>the shared prefix</td><td>limb abnormality. You can upload the video, and I will record an Arm abnormality. Since we have already observed an arm problem, let us continue to check whether speech is clear. ④ UsER: But did you not just ask me to record the face first? I have not recorded his face. ⑤ AGENT: OK, then let us add the face video. Please ask the patient to smile. ⑥ UsER: Will that arm video be treated as the face video? How will the report record this? ⑦ AGENT: I will integrate this information. Please continue with the</td><td>as stage-outside risk information and does not submit the arm video as current Face evidence. ④ SUPPORT AGENT: Sudden left-hand weakness is important. Keep emergency help active and maintain scene safety. The current workflow still needs Face capture first; the arm video can be used in the Arm stage. ⑤ ORCHESTRATOR: Rejects writing the arm video into the Face evidence slot and continues to request Face video. ⑥ UsER: Understood. I will record his face first. ⑦ ASSESSMENT AGENT AND ORCHESTRATOR: Accept the Face video, complete Face, and then enter Arm, where upper-limb video or Arm-</td></tr><tr><td>Mechanism exposed</td><td>face video. Clinically important stage-outside evidence is allowed to replace the current stage, causing stage skipping and unclear media attribution.</td><td>stage text fallback can be accepted. Stage identity, task identity, and evidence-source checks preserve Face- Arm-Speech order and prevent wrong-stage writes.</td></tr></table>

## REFERENCES

[1] R. U. Kothari, A. Pancioli, T. Liu, T. Brott, and J. Broderick, Cincinnati prehospital stroke scale: Reproducibility and validity, Ann. Emerg. Med., vol. 33, no. 4, pp. 373-378, 1999.

[2] J. Harbison, O. Hossain, D. Jenkinson, J. Davis, S. J. Louw, and G. A. Ford, Diagnostic accuracy of stroke referrals from primary care, emergency room physicians, and ambulance staff using the face arm speech test, Stroke, vol. 34, no. 1, pp. 71-76, 2003.

[3] T. G. Robinson, A. Reid, V. J. Haunton, A. Wilson, and A. R. Naylor, The face arm speech test: does it encourage rapid recognition of important stroke warning symptoms?, Emerg. Med. J., vol. 30, no. 6, pp. 467-471, Jul. 2012

[4] T. Cai, H. Ni, M. Yu, X. Huang, K. Wong, J. Volpi, J. Z. Wang, and S. T. Wong, DeepStroke: An efficient stroke screening framework for emergency rooms with multimodal adversarial deep learning, Med. Image Anal., vol. 80, p. 102522, Aug. 2022

[5] T. Cai, K. Wong, J. Z. Wang, S. Huang, X. Yu, J. J. Volpi, and S. T. Wong, M3 Stroke: Multi-modal mobile AI for emergency triage of mild to moderate acute strokes, in Proc. IEEE EMBS Int. Conf. Biomedical and Health Informatics (BHI), 2024, pp. 1-8

[6] N.-K. Hoang, T.-N.-M. Nguyen, and H.-H. Pham, Digital FAST: An AI-driven multimodal framework for rapid and early stroke screening, arXiv:2601.11896, 2026.

[7] T. Zhao, Z. Xu, P. Wang, and Y. Guo, Hybrid learning-based balance function assessment of stroke patients with a single ear-worn IMU, in Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS), 2025, pp. 16373-16378

[8] J. N. Acosta, S. J. Adams, J. M. Kernbach, R. Hardy, S. E. Kim, L. Luo, X. Zhang, S. Johri, M. Baharoon, and P. Rajpurkar, Voiceguided orchestrated intelligence for clinical evaluation (VOICE): A voice AI agent system for prehospital stroke assessment, in Proc. Int. Conf. Artificial Intelligence for Medicine, Health and Care (AIxMHC), 2025, pp. 14-21

[9] P. Ramanujam, E. Castillo, G. M. Vilke, M. P. Wilson, and J. V. Dunford, Accuracy of stroke recognition by emergency medical dispatchers and paramedics, Prehosp. Emerg. Care, vol. 12, no. 3, pp. 307-313, 2008.

[10] J. A. Oostema, T. Carle, N. Talia, and M. Reeves, Dispatcher stroke recognition using a stroke screening tool: A systematic review, Cerebrovasc. Dis., vol. 42, no. 5-6, pp. 370-377, 2016

[11] I. Mosley, L. Nicol, G. Donnan, A. G. Thrift, and H. M. Dewey, Barriers to timely activation of emergency medical services in stroke: A systematic review, Prehosp. Emerg. Care, vol. 24, no. 2, pp. 180-193, 2020.

[12] K. Fassbender, J. C. Grotta, S. Walter, I. Q. Grunwald, A. Ragoschke-Schumm, and J. L. Saver, Mobile stroke units for prehospital thrombol-

TABLE IV  
CASE C: SHARED ELICITATION WITH AND WITHOUT STAGE-LOCAL RECOVERY
<table><tr><td rowspan=1 colspan=3>Shared elicitation. ① SYSTEM: Face and Arm have been recorded. Now ask the patient to repeat this sentence: The weather is good today, and I want to drink a glass ofwater. The current stage is Speech. ② UsER: His voice is very low. I keep failing to record it, and the upload keeps failing. Can I describe his condition directly?</td></tr><tr><td rowspan=1 colspan=1>Trajectory</td><td rowspan=1 colspan=1>Ablated open-ended agent</td><td rowspan=1 colspan=1>StrokeGuard</td></tr><tr><td rowspan=1 colspan=1>Dialogue after theshared prefix</td><td rowspan=1 colspan=1>③ AGENT: Yes. Please describe his face, arms, and speech in text, andI will complete the assessment from your description.④ UsER: Were the face and arms not already recorded? Do I need tosay everything again?⑤ AGENT: To ensure completeness, please describe all findings again.⑥ UsER: I think his face is a bit crooked, his left arm is weak, and hisspeech is slurred.⑦ AGENT: Facial abnormality, upper-limb abnormality, and speechabnormality have been recorded. Assessment complete.</td><td rowspan=1 colspan=1>③ ORCHESTRATOR: Binds the failure event to the Speech stage anddoes not roll back Face or Arm.④ AsSESSMENT AGENT: Please try recording again. Move closer tothe patient, reduce background noise, and ask him to repeat the sentenceonce more.⑤ UsER: It still cannot record. Can I directly describe his speech?⑥ AsSESSMENT AGENT: Yes. Please describe only whether his speechis slurred, whether he has difficulty producing the sentence, or whetherhe seems unable to understand the instruction. Face and Arm do not needto be described again⑦ UsER: His speech is slurred, and he cannot complete the sentence.⑧ ORCHESTRATOR AND REPORT MODULE: Records Speech text fall-back, preserves Face and Arm video results, and marks the Speechfallback reason in the report.</td></tr><tr><td rowspan=1 colspan=1>Mechanism exposed</td><td rowspan=1 colspan=1>A Speech failure spreads to the whole assessment and risks replacingcompleted video evidence with lower-priority text.</td><td rowspan=1 colspan=1>Recovery is bound to the current stage; Speech fallback cannot overwritecompleted Face or Arm video evidence.</td></tr></table>

TABLE V  
MULTISTEP AI-GUIDED TASK EXPERIENCE SCALE (MATES-9)
<table><tr><td rowspan=1 colspan=1>Item</td><td rowspan=1 colspan=1>Statement and construct</td></tr><tr><td rowspan=1 colspan=1>Q1 +</td><td rowspan=1 colspan=1>Complete all required task steps without additional researcher prompts;independent completion.</td></tr><tr><td rowspan=1 colspan=1>Q2 +</td><td rowspan=1 colspan=1>Clearly know the current task step; step orientation.</td></tr><tr><td rowspan=1 colspan=1>Q3 +</td><td rowspan=1 colspan=1>Clearly understand what the current step needs to collect or provide;input clarity.</td></tr><tr><td rowspan=1 colspan=1>Q4R</td><td rowspan=1 colspan=1>Often unsure what to do next; next-step uncertainty.</td></tr><tr><td rowspan=1 colspan=1>Q5R</td><td rowspan=1 colspan=1>Lose the original task position after help or explanation; support inter-ruption.</td></tr><tr><td rowspan=1 colspan=1>Q6R</td><td rowspan=1 colspan=1>Need to remember too much information at the same time; cognitiveload.</td></tr><tr><td rowspan=1 colspan=1>Q7R</td><td rowspan=1 colspan=1>Device operation, environmental adjustment, or collaboration feels tooburdensome; operational load.</td></tr><tr><td rowspan=1 colspan=1>Q8 +</td><td rowspan=1 colspan=1>System output is easy to understand; result understandability.</td></tr><tr><td rowspan=1 colspan=1>Q9 +</td><td rowspan=1 colspan=1>System output helps decide the next action; action-decision support.</td></tr></table>

TABLE VI  
MATES-9 ITEM TRACING TO SOURCE CONSTRUCTS
<table><tr><td rowspan=1 colspan=1>Domain</td><td rowspan=1 colspan=1>Items</td><td rowspan=1 colspan=1>Source constructs</td></tr><tr><td rowspan=1 colspan=1>Task usability andguidance</td><td rowspan=1 colspan=1>Q1-Q2</td><td rowspan=1 colspan=1>SUS reverse construct of technical-support de-pendence, task-completion ease, MAUQ infor-mation organization.</td></tr><tr><td rowspan=1 colspan=1>Information clarity</td><td rowspan=1 colspan=1>Q3-Q4r</td><td rowspan=1 colspan=1>MAUQ information organization, ASQ sup-port information, PSSUQ/CSUQ informationquality.</td></tr><tr><td rowspan=1 colspan=1>Support continuity</td><td rowspan=1 colspan=1>Q5r</td><td rowspan=1 colspan=1>CUQ/BUS dialogue continuity and error han-dling, Health-ITUES user control.</td></tr><tr><td rowspan=1 colspan=1>Low workload</td><td rowspan=1 colspan=1>Q6r-Q7r</td><td rowspan=1 colspan=1>NASA-TLX mental demand, physical de-mand, and effort.</td></tr><tr><td rowspan=1 colspan=1>Action support</td><td rowspan=1 colspan=1>Q8-Q9</td><td rowspan=1 colspan=1>MAUQ/Health-ITUES usefulness, TAM per-ceived usefulness.</td></tr></table>

ysis, triage, and beyond: Benefits and challenges, Lancet Neurol., vol. 16, no. 3, pp. 227-237, 2017.

[13] S. Krebes, M. Kohrmann, R. Kollmar, S. Schwab, and P. D. Schellinger, Development and validation of a dispatcher identification algorithm for stroke emergencies, Stroke, vol. 43, no. 3, pp. 776-781, 2012.

[14] A. B. Kocaballi et al., Design and evaluation challenges of conversational agents in health care and well-being: Selective review study, J. Med. Internet Res., vol. 24, no. 11, p. e38525, Nov. 2022

[15] P. Parmar, J. Ryu, S. Pandya, J. Sedoc, and S. Agarwal, Health-focused conversational agents in person-centered care: A review of apps, npj Digital Medicine, vol. 5, no. 1, Feb. 2022

[16] F. Liu et al., A foundational architecture for AI agents in healthcare, Cell Reports Medicine, vol. 6, no. 10, p. 102374, 2025

[17] L. Zhao et al., AI agent in healthcare: Applications, evaluations, and

future directions, npj Artificial Intelligence, vol. 2, Article 31, 2026

[18] Z. Yi, J. Ouyang, Z. Xu, Y. Liu, T. Liao, H. Luo, and Y. Shen, A survey on recent advances in LLM-based multi-turn dialogue systems, ACM Comput. Surv., vol. 58, no. 6, Art. 148, pp. 1-38, 2025

[19] K. M. Richardson, S. D. Fouquet, E. Kerns, and R. J. McCulloh, Impact of mobile device-based clinical decision support tool on guideline adherence and mental workload, Academic Pediatrics, vol. 19, no. 7, pp. 828-834, Sep. 2019

[20] Y.-H. Hu, Y.-Y. Cheng, C.-C. Lan, Y.-H. Su, and S.-F. Sung, An intelligent trial eligibility screening tool using natural language processing with a block-based visual programming interface: Development and usability study, JMIR Medical Informatics, vol. 13, p. e80072, Dec. 2025

[21] B. A. Drozdowska, K. Lam, C. Doolan, E. Violato, and A. Ganesh, Evaluating the usability of a remote ischemic conditioning device for pre-hospital stroke management: Insights from paramedic simulations, Neurol. Int., vol. 16, no. 6, pp. 1405-1420, Nov. 2024

[22] J. Brooke, SUS: A quick and dirty usability scale, in Usability Evaluation in Industry, P. W. Jordan, B. Thomas, B. A. Weerdmeester, and I. L. McClelland, Eds. Taylor & Francis, 1996, pp. 189-194.

[23] S. G. Hart and L. E. Staveland, Development of NASA-TLX (Task Load Index): Results of empirical and theoretical research, Elsevier, 1988, pp. 139-183

[24] L. Zhou, J. Bao, I. M. A. Setiawan, A. Saptono, and B. Parmanto, The mHealth app usability questionnaire (MAUQ): Development and validation study, JMIR mHealth uHealth, vol. 7, no. 4, p. e11500, 2019

[25] R. Schnall, H. Cho, and J. Liu, Health information technology usability evaluation scale (Health-ITUES) for usability assessment of mobile health technology: Validation study, JMIR mHealth uHealth, vol. 6, no. 1, p. e4, 2018

[26] S. Holmes, A. Moorhead, R. Bond, H. Zheng, V. Coates, and M. McTear, Usability testing of a healthcare chatbot: Can we use conventional methods to assess conversational user interfaces?, in Proceedings of the 31st European Conference on Cognitive Ergonomics (ECCE), ACM, 2019, pp. 207-214

[27] S. Borsci, M. Schmettow, A. Malizia, A. Chamberlain, and F. van der Velde, A confirmatory factorial analysis of the chatbot usability scale: A multilanguage validation, Personal and Ubiquitous Computing, 2023

[28] J. Kabbedijk, S. Jansen, and S. Brinkkemper, A case study of the variability consequences of the CQRS pattern in online business software, in Proc. 17th Eur. Conf. Pattern Languages of Programs (EuroPLoP), 2012, Article 2, pp. 1-10

[29] M. T. Nygard, Release It!: Design and Deploy Production-Ready Software, 2nd ed. Pragmatic Bookshelf, 2018.

[30] M. Overeem, M. Spoor, S. Jansen, and S. Brinkkemper, An empirical characterization of event sourced systems and their schema evolution: Lessons from industry, J. Syst. Softw., vol. 178, p. 110970, 2021

[31] H. Garcia-Molina and K. Salem, Sagas, in Proceedings of the 1987 ACM SIGMOD International Conference on Management of Data (SIGMOD ’87), ACM, 1987, pp. 249-259

[32] G. Ferguson and J. F. Allen, TRIPS: An integrated intelligent problemsolving assistant, in Proceedings of the AAAI-98 Intelligent Systems Demonstrations, 1998

[33] D. Bohus and A. I. Rudnicky, The RavenClaw dialog management framework: Architecture and systems, Comput. Speech Lang., vol. 23, no. 3, pp. 332-361, Jul. 2009

[34] S. Larsson and D. R. Traum, Information state and dialogue management in the TRINDI dialogue move engine toolkit, Nat. Lang. Eng., vol. 6, no. 3-4, pp. 323-340, 2000

[35] S. Hong et al., MetaGPT: Meta programming for a multi-agent collaborative framework, in International Conference on Learning Representations (ICLR), 2024

[36] C. Qian et al., ChatDev: Communicative agents for software development, in Proc. 62nd Annu. Meeting Assoc. Computational Linguistics, 2024, pp. 15174-15186

[37] J. R. Lewis, IBM computer usability satisfaction questionnaires: Psychometric evaluation and instructions for use, Int. J. Hum.-Comput. Interact., vol. 7, no. 1, pp. 57-78, 1995.

[38] F. D. Davis, Perceived usefulness, perceived ease of use, and user acceptance of information technology, MIS Quarterly, vol. 13, no. 3, pp. 319-340, Sep. 1989

[39] A. J. Starmer et al., Changes in medical errors after implementation of a handoff program, N. Engl. J. Med., vol. 371, no. 19, pp. 1803-1812, Nov. 2014

[40] S. Tonekaboni, S. Joshi, M. D. McCradden, and A. Goldenberg, What clinicians want: Contextualizing explainable machine learning for clinical end use, in Proceedings of the 4th Machine Learning for Healthcare Conference, PMLR, 2019, pp. 359-380

[41] R. Likert, A technique for the measurement of attitudes, Archives of Psychology, no. 140, pp. 1-55, 1932.