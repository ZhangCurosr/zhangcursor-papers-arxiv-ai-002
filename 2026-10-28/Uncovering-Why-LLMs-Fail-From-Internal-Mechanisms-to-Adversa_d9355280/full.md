# Uncovering Why LLMs Fail: From Internal Mechanisms to Adversarial Reasoning Trajectories

Zirui Ding<sup>1,2</sup>, Xinxin Fan<sup>2,3</sup>, Hao Yuan<sup>1,4</sup>, Shaoye Luo<sup>2,3</sup>, Men Niu<sup>1,2</sup>, Yunfeng Lu<sup>5</sup>

<sup>1</sup>Institute of Computing Technology, Chinese Academy of Sciences

<sup>2</sup>University of Chinese Academy of Sciences

<sup>3</sup>State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences <sup>4</sup>ShanghaiTech University <sup>5</sup>Beihang University

## Abstract

Large language models (LLMs) are usually evaluated by finaloutput correctness or safety, yet an endpoint label does not reveal how a failure develops throughout the reasoning process. We therefore ask whether ordinary reasoning failures and adversarially induced failures can be characterized using the same internal coordinates. Towards this end, we in this paper study a white-box diagnostic framework CauTrace to disclose the root-cause of why LLM fails from three fundamental inspections: Functional States, Propagation of Representation, Temporal-State Coordination and Reasoning Trajectory. First, seven functional states are creatively defined to represent the support and constraint processing in LLMs. Second, the propagation of representation change measures how the changes among the seven states persist across subsequent Transformer blocks. Third, temporal-state coordination and trajectory topology characterize when the commitment and verification occur relative to the support, and how the dominant state sequence evolves over the full inference process. Finally, we evaluate CauTrace on both clean reasoning failures and paired adversarial perturbations. Across Qwen and LLaMA-family models, the failed traces show weaker Attention–MLP agreement on support-state updates and less verification activity relative to commitment, while the exact representation-level signatures and timing shifts vary by architecture. Held-out repair and steering tests demonstrate consistent operational signals that were in line with our claims for certain models and specific settings, with aligned steering producing the predicted directional changes in process measures. Overall, CauTrace provides a unified diagnostic framework for analyzing failure dynamics through changes in support, commitment, and verification, which would shed light on the in-depth research on LLM security.

## 1 Introduction

Hallucinations and harmful generations can make large language models (LLMs) appear certain while producing unsupported facts, fabricated evidence, or invalid reasoning, as documented by recent hallucination and factuality surveys [13,43]. In ordinary use, such outputs can propagate misinformation and mislead downstream decisions. In security-sensitive settings, HarmBench [28] and JailbreakBench [6] standardize the measurement of whether manipulated prompts turn safetyaligned models from refusal to harmful compliance. The cost is therefore not limited to a wrong answer: a fluent but ungrounded explanation can also make the failure difficult for users or automated detectors to identify [22, 27].

Research on trustworthy and safe LLMs has identified several non-exclusive sources of these failures. TruthfulQA shows that models can reproduce popular misconceptions learned from their training distribution [23]. Lost-in-themiddle analyses [24] and distraction tests [38] show that task-relevant evidence can be overlooked when it is embedded in long or irrelevant context. Calibration studies examine whether models can estimate the correctness of their own answers [18, 47], whereas SelfCheckGPT detects unsupported passages through disagreement among sampled responses [27]. Reasoning traces introduce another gap: biasing features can change an answer without being acknowledged in the generated chain of thought [41], and intervention-based studies find that textual rationales do not always mediate the final prediction [20,36]. In safety-aligned models, adversarial prompts can redirect intermediate hidden-state processing [51], exploit safety signals concentrated in the chat-template region [21], alter refusal-related representations [2], or hijack attention from safety-relevant template tokens [4]. These findings sug gest that hallucination and unsafe compliance can involve interacting failures of evidence use, reasoning coordination, and internal control rather than one isolated defect.

Nevertheless, LLM trustworthiness and safety are still commonly evaluated from endpoint outputs, and an endpoint label alone does not reveal how the result was produced through successive Transformer layers and generation steps [43,51]. Prior studies show that the same endpoint failure label can arise from heterogeneous failure patterns: task-relevant premises may be overlooked when evidence is embedded in long or distracting contexts [24, 38]; parametric falsehoods or unsupported associations may dominate a response [23]; multi-hop reasoning may omit a required inference or fail to integrate the final supporting fact [11, 36]; and explicit task constraints may be violated [17]. Moreover, answer commitment may be driven by shortcut features rather than by the stated support, with the reported reasoning merely rationalizing that choice [41]; a later self-check may then fail to repair the answer or may even degrade it [12]. The same ambiguity arises in safety evaluation: an unsafe completion establishes that a safeguard failed, but not which part of the internal generation process was reorganized by the adversarial prompt [6,51]. Consequently, endpoint evaluation alone leaves a fundamental diagnostic question unanswered: how can failures be traced to changes within the model’s internal generation pipeline?

Existing research therefore provides four important but partial analytic views. Endpoint benchmarks measure factuality through learned misconceptions in TruthfulQA [23], generated and human-annotated hallucinations in HaluEval [22], or standardized harmful-behavior evaluation in HarmBench [28] and JailbreakBench [6]. Calibration and consistency methods estimate whether an answer is unreliable [18, 27]; chain of-thought studies test whether textual rationales faithfully mediate predictions [36, 41]; and mechanistic methods decode hidden representations [3, 10] or localize safety-relevant directions [2]. These views are rarely connected by a common process-level unit. Endpoint scores can merge distinct failure routes, textual rationales need not expose the operative computation, and a localized representation does not by itself explain when support, commitment, and verification become miscoordinated over an entire generation. A unified analysis therefore needs to connect where a change persists, when functional roles switch or lose coordination, and which whole-trace route produces the endpoint.

In this work, our goal is therefore not to propose another endpoint detector or a stronger attack, but to ask a nontrivial question: Can non-adversarial reasoning failure and adversarially-induced reasoning failure be described with a common set of reasons? Surrounding this question, we treat a well-formed generation as a reasoning trajectory in which task-relevant support and constraint are established before answer commitment becomes dominant, and later verification remains able to confirm or revise the candidate answers. Failures can therefore appear as weak support continuity, shifted commitment/verification timing, frequent state switching, or low verification activity relative to commitment, which offers an operational and quantitative criterion for analyzing reasoning failures.

Generally, the final LLM-output answers whether the response is incorrect, unsupported, or unsafe, but cannot locate which part of the internal process changed; the same output label can also hide different state-transition routes. Our proposed CauTrace first maps each generation to seven functional-state coordinates describing support/constraint processing, answer commitment, and verification. It then analyzes three complementary properties: whether representation changes persist across subsequent Transformer blocks, whether the timing of support, commitment, and verification remains coordinated, and which trajectory topology emerges over the full generation trace. In a nutshell, three main contributions are involved:

• Reasoning-Process Formulation. We creatively define seven radical functional-state coordinates to abstract the entire reasoning process, by which the support/constraint processing, answer commitment, and verification can be formulated appropriately.

• Three-Level Failure-Diagnostic Analytics. We construct the propagation of representation mechanism across Transformer blocks, temporal coordination among functional states, and whole-trace trajectory topology, by which the fine-grained representation changes can be captured microscopically.

• Empirical Verification. Multi-facet experimentoriented evidences across the popular Qwen and LLaMA-family models demonstrate that: i) the failed traces differ from successful traces in support-state update agreement and relative verification activity; ii) that even the same output label can follow different reasoning trajectory chains; and iii) that harmful-benignpaired adversarial flips are accompanied by larger internal-state change.

## 2 Related Work

## 2.1 Reliability and Reasoning Analytics

Hallucination analytics first requires an operational definition of failure. Surveys distinguish factuality, faithfulness to a source, evidence support, and task correctness as related but non-equivalent targets [13,15,43,50]. At the benchmark level, TruthfulQA tests whether models reproduce common misconceptions learned from human text [23], whereas HaluEval combines generated and human-annotated samples to evaluate hallucination recognition across general queries and taskspecific settings [22]. FActScore moves from response-level labels to claim-level analysis by decomposing long-form generations into atomic facts and checking each fact against a knowledge source [30]. Related task analytics isolate other endpoint failure dimensions: FollowBench separates finegrained instruction constraints [17], and NovelHopQA varies reasoning depth and context length to identify missed finalhop integration and long-range drift [11]. These benchmarks improve diagnostic granularity at the output level, but the same failed label or unsupported claim can still be produced through different internal routes.

A complementary line examines whether the model itself provides signals of unreliability. Kadavath et al. [18] derive probability-based estimates of answer correctness and knowledge confidence from model outputs [18]; Yin et al. [47] evaluate self-knowledge across answerable and unanswerable questions [47]; and SelfCheckGPT [27] uses disagreement among stochastic generations as a black-box signal of unsupported content [27]. Verification is not automatically corrective, however, intrinsic self-correction without external feedback can fail to improve reasoning and can even reduce accuracy [12]. These methods provide uncertainty or consistency scores, but they do not identify which stage of evidence use, answer formation, or checking produced the score.

Reasoning analytics uses generated intermediate text as a more detailed behavioral trace. Scratchpads externalize intermediate computation [33]; chain-of-thought and zero-shot reasoning prompts elicit multi-step rationales [19, 44]; and self-consistency aggregates answers across multiple sampled chains [42]. Faithfulness research separates the plausibility of a rationale from whether it actually mediates the prediction [14]. Biasing features can systematically change answers while remaining absent from the stated rationale [41]; rationale perturbation and truncation tests reveal task- and modeldependent faithfulness [20]; and causal mediation analysis finds that models do not always rely on their own intermediate steps when producing the final answer [36]. Consequently, textual reasoning traces provide valuable behavioral evidence but cannot be treated as direct records of internal computation.

## 2.2 Adversarial-Prompt and Safety Analytics

Security analytics first characterizes attacks and their behavioral endpoints. HarmBench standardizes harmful behaviors, red-teaming methods, and refusal classifiers within one evaluation protocol [28]; JailbreakBench defines a reproducible threat model and artifact repository for comparing jailbreak attacks and defenses [6]; and StrongREJECT grades both harmfulness and response informativeness so that empty or non-responsive outputs are not counted as successful jail breaks [40]. Many-shot jailbreaking demonstrates that long contexts containing many demonstrations can progressively erode refusal behavior [1]. Corpus-based studies characterize in-the-wild jailbreak prompts [37] and analyze transferable semantic attack patterns [48], whereas AutoDAN automates the search for stealthy jailbreak instructions [25]. Prompt-injection work separately formalizes attacker objectives and evaluation settings [26], including optimization attacks against LLM-as-a-judge systems [39]. This line establishes whether an attack succeeds and under which threat model, but its principal analytic unit remains the prompt– response outcome.

A second line analyzes the internal organization of safety behavior. Representation Engineering extracts populationlevel activation directions associated with behavioral concepts [52]. Arditi et al. identify a direction that mediates refusal and validate it through activation addition and ablation [2]; subsequent studies describe refusal using concept cones [45] or multiple approximately orthogonal safety dimensions [35]. Intermediate-state analysis compares aligned and jailbroken generations across layers [51]. More localized studies show that safety decisions can be anchored in the chat-template region [21] and that universal suffixes redirect attention away from safety-relevant template tokens [4]. Activated-concept analysis has also been used for jailbreak defense [49], while PromptLocate localizes prompt-injection influence within the model pipeline [16]. These methods reveal safety-relevant representations or information-flow vulnerabilities, but most are organized around a particular refusal mechanism, attack family, or localized component rather than a task-general generation process.

## 2.3 Mechanistic and Representation Analytics

White-box analytics supplies the primitives for reading and intervening on model internals. Contrast-consistent learning extracts latent truth-related structure without labeled answers [5], while the Tuned Lens decodes layer-wise predictive distributions and exposes how token predictions evolve through the residual stream [3]. Patchscopes generalizes projection and intervention methods by transplanting hidden representations into a target prompt so that the model can express their information in natural language [10]. TransformerLens [31] provides reusable instrumentation for caching, decomposing, and patching activations [31]. Together, these methods determine what information is decodable at a layer or how an intervention changes a selected behavior.

Mechanistic studies then seek smaller computational units. Feed-forward layers can operate as key–value memories that promote factual concepts [9]; induction-head analyses identify attention circuits that implement repeated-pattern completion [34]; and causal tracing with ROME localizes and edits factual associations [29]. Automated Circuit Discovery systematizes activation-patching workflows by searching for the edges that form a behavior-specific circuit [7]. Such methods provide component- or circuit-level explanations, but their units are usually selected for one behavior and do not by themselves describe how functional roles coordinate over an entire generation trace.

Summary & Motivation. Existing analytics offer increasingly precise endpoint scores, attack measurements, hiddenrepresentation decoders, and localized safety mechanisms, but rarely connect these resolutions into one generation-level account of ordinary and adversarial failures. CauTrace addresses this gap by organizing a white-box evidence chain around seven functional-state/process coordinates and jointly analyzing Propagation of Representation, Temporal-State Coordination, and Reasoning Trajectory, rather than proposing another endpoint benchmark, probe, or refusal direction.

![](images/cfe263ef35c1de46c5cd03dd8dc7106ab9800fe72dfc26bf6225d61d7919ac84.jpg)  
Figure 1: CauTrace overview. CauTrace maps clean/adversarial traces to seven functional-state coordinates and analyzes (1) Propagation of Representation across Transformer blocks; (2) Temporal-state Coordination among support, commitment, and verification; and (3) Reasoning Trajectory over the full generation trace.

## 3 Methodology

Fig. 1 sketches the overview framework, the propose Cau-Trace treats each generation as one hidden-state trajectory and uses the seven functional-state coordinates as a reasoning chain to maps the generated trace. To unveil the inherent reasoning process, we inspect three-level linked analytics: Propagation of Representation. It examines whether a local change is retained or canceled/absorbed alongside Transformer’s Attention and MLP updates; ii) Temporal-State Coordination. It is used to reconstruct how the same coordinates switch and align dynamically over the generation time, including whether support and constraint precede answer commitment and whether the later verification can revise it back; and iii) Reasoning Trajectory. It integrates the local and temporal measurements into a whole-trace route for further observation and judgement on the output trajectory.Together, these three levels provide a unified view of the reasoning process across different analytical resolutions.

The above three-level analytics form a measurement hierarchy rather than three independent feature inspections. Representation analytics aims to identify where a deviation remains available to later computation; temporal-state analysis identifies when state-links, commitment, and verification become miscoordinated; and trajectory chains specify which reasoning chain exactly lead to the final results. Therefore, in view of these three-level analytics, they are hierarchical and mutually complementary to calibrate the reasoning trace.

## 3.1 Functional-State Coordinates

Inspired by decompositions of problem-solving into representation, knowledge access, operation, monitoring, and answer confirmation [8, 32], we define seven functional-state/process coordinates for LLM reasoning. They address three questions: what supports or constrains the answer? when does the model commit to a candidate answer? and can the later computation verify or revise that commitment? Table 1 explicitly maps each state to one of the three research questions (Que. I-III): PB, KR, RC, OC, and FRT represent support or constraint formation; AC describes answer selection; and SV indicates post-commitment verification and revision.

Why These Seven States. Three considerations motivate this seven-state decomposition: i) Coverage of Roles of LLM Reasoning. The seven-state decomposition covers recurring functional roles in reasoning, including binding, evidence access and composition, operational computation, rule tracking, commitment, and checking.; ii) Operational Separability. Our defined seven states enables to adequately decouple the entangled roles under task-grounded weak supervision; and iii) Failure-Type Diagnostic Correction. The seven states can validly distinguish the categories of LLM failures, also verified by the subsequent experiments. We here do not claim that these seven states are a unique or minimal cognitive decomposition. The first five states can overlap and need not occur in a fixed order. They provide a general task decomposition at a suitable granularity that can be used to deeply inspect the hidden-state reasoning process in a microscopic

manner.

Let PB, KR, RC, OC, and FRT respectively denote the support/constraint family,

$$
\begin{array} { r } { S ^ { + } = \{ P B , K R , R C , O C , F R T \} . } \end{array}\tag{1}
$$

The family captures premise binding, knowledge retrieval, reasoning composition, operational computation, and rule tracking. The defined $A C$ can convert these signals into a candidate answer; SV can evaluate whether the candidate answer remains supported or revisable. Hence, the process failure is defined by abnormal coordination rather than by a missing state. A probe score $p _ { s } ( t )$ measures the evidence for an operational role in the hidden representation; it does not indicate the entry into a discrete reasoning stage. Cau-Trace therefore allows overlapping state scores and ostensibly "non-sequential" transitions, despite different categories of LLM failures are subject to specified reasoning trajectories. To make these functional coordinates concrete and clear, we provide a simple reasoning example below.

## State Walkthrough Example

Question. Can a person born in 2000 legally vote in the United States in 2017?   
Target Output. No   
State Interpretation.

1. Premise Binding: bind the birth year (2000), decision year (2017), and jurisdiction (United States).

2. Knowledge Retrieval: recall the voting-age rule that a person must be at least 18.

3. Reasoning Composition: combine the age rule with the question conditions.

4. Operational Computation: compute that the person is 17 in 2017.

5. Formal Rule Tracking: keep the answer in the required Yes/No decision space.

6. Answer Commitment: select No once support is sufficient.

7. Self-Verification: check that the age calculation and the voting rule are consistent with No.

## 3.2 Propagation of Representation

The propagation of representation aims to analyze whether a local change in support state signals is canceled/absorbed or persists across later Transformer updates. That is, the Attention–MLP update agreement measures whether the two sublayers move support-state in the same direction; the valuespectrum statistics measure changes in the value-projected representation of the state probes; and the alignment with the Answer Commitment (AC) probe tests whether the dominant value-space direction becomes more commitment-related.

Representation-Level Deviation Score. Let $t \in \{ 1 , \ldots , T \}$ denote a generation step, $l \in \mathcal { L }$ an analyzed Transformer layer, $\mathcal { S }$ the seven-state set and $s \in S$ a specific state. Let $p _ { s , t , l } ^ { \mathrm { i n } } , p _ { s , t , l } ^ { \mathrm { a t t n } }$ and $p _ { s , t , l } ^ { \mathrm { o u t } }$ represent the state-coordinate scores at the block input, after the Attention sublayer, and after the MLP sublayer, respectively. We define

$$
\Delta _ { s , t , l } ^ { \mathrm { a t t n } } = p _ { s , t , l } ^ { \mathrm { a t t n } } - p _ { s , t , l } ^ { \mathrm { i n } } ,\tag{2}
$$

$$
\Delta _ { s , t , l } ^ { \mathrm { m l p } } = p _ { s , t , l } ^ { \mathrm { o u t } } - p _ { s , t , l } ^ { \mathrm { a t t n } } ,\tag{3}
$$

$$
\begin{array} { r } { \boldsymbol { A } _ { s , t , l } = \mathrm { s i g n } \left( \Delta _ { s , t , l } ^ { \mathrm { a t t n } } \right) \mathrm { s i g n } \left( \Delta _ { s , t , l } ^ { \mathrm { m l p } } \right) , } \end{array}\tag{4}
$$

where $\Delta _ { s , t , l } ^ { \mathrm { a t t n } }$ and $\Delta _ { s , t , l } ^ { \mathrm { m l p } }$ are Attention-side and MLP-side changes in the corresponding state-coordinate score. $A _ { s , t , l } \in$ $\{ - 1 , 0 , 1 \}$ records whether the two sublayers move the state in the same, opposite, or zero direction. We average $A _ { s , t , l }$ over support states and analyzed layers to obtain $\bar { A } _ { \mathrm { s u p } } ( t )$ . A smaller $\bar { A } _ { \mathrm { s u p } }$ indicates lower agreement between Attention and MLP updates on support-related state scores.

For value-projected Attention, we analyze the projected representations rather than the parameter matrix. Let $h _ { j , l }$ be the normalized residual representation of token j before Attention in layer $l ,$ and $W _ { l , h } ^ { V }$ the value-projection matrix of head h. Thus, the head-level value vector can be defined as

$$
\nu _ { j , l , h } ^ { V } = \widetilde { h } _ { j , l } W _ { l , h } ^ { V } \in \mathbb { R } ^ { d _ { \nu } } ,\tag{5}
$$

where $d _ { \nu }$ is the value-head dimension. At generation step t, let $I _ { t }$ denote the token-layer-head units used for spectral analysis. We stack the corresponding value vectors into

$$
\begin{array} { r } { V _ { t } = \left[ ( \nu _ { j , l , h } ^ { V } ) ^ { \top } \right] _ { ( j , l , h ) \in I _ { t } } \in \mathbb { R } ^ { m \times d _ { \nu } } , } \end{array}\tag{6}
$$

where $m = | I _ { t } |$ . The local second-moment matrix is $\begin{array} { r } { \Sigma _ { t } ^ { V } = } \end{array}$ $\left( V _ { t } \right) ^ { \top } V _ { t } / m$ . Let $\lambda _ { 1 , t } ^ { V } \geq \lambda _ { 2 , t } ^ { V } \geq \cdots$ denote the eigenvalues of $\Sigma _ { t } ^ { V }$ , sorted in non-increasing order. We respectively define the leading spectral concentration, normalized spectrum, and step-to-step spectral change as

$$
\boldsymbol { \rho } _ { t } ^ { V } = \frac { \lambda _ { 1 , t } ^ { V } } { \sum _ { j } \lambda _ { j , t } ^ { V } } ,\tag{7}
$$

$$
\pi _ { t , k } ^ { V } = \frac { \lambda _ { k , t } ^ { V } } { \sum _ { j } \lambda _ { j , t } ^ { V } } ,\tag{8}
$$

$$
D _ { \mathrm { K L } } ^ { V } ( t ) = D _ { \mathrm { K L } } \big ( \pi _ { t } ^ { V } \| \pi _ { t - 1 } ^ { V } \big ) ,\tag{9}
$$

where $\rho _ { t } ^ { V }$ is the leading spectral concentration, $\pi _ { t , k } ^ { V }$ is the normalized spectral mass, and $D _ { \mathrm { K L } } ^ { V } ( t )$ is the step-to-step spectral change. Let $u _ { \mathrm { a c } } ^ { V }$ represent the answer-commitment probe direction in the same value space and $\nu _ { 1 , t } ^ { V }$ the leading eigenvector, thus their alignment degree can be defined as

$$
\Gamma _ { \mathrm { a c } , V } ( t ) = \left| \cos \left( u _ { \mathrm { a c } } ^ { V } , \nu _ { 1 , t } ^ { V } \right) \right| .\tag{10}
$$

The absolute cosine removes the arbitrary sign of the eigenvector. $\Gamma _ { \mathrm { a c } , V } ( t )$ therefore measures alignment degre between

Table 1: Seven functional-state coordinates organized by three diagnostic questions.
<table><tr><td>Seven States</td><td>Functional Roles</td><td>Questions</td></tr><tr><td>Premise Binding (PB)</td><td>Bind entities, conditions, and prompt constraints</td><td>Que.I</td></tr><tr><td>Knowledge Retrieval (KR)</td><td>Activate relevant factual or commonsense knowledge</td><td>Que. I</td></tr><tr><td>Reasoning Composition (RC)</td><td>Compose evidence across hops or intermediate claims</td><td>Que. I</td></tr><tr><td>Operational Computation (OC)</td><td>Execute numerical or symbolic operations</td><td>Que. I</td></tr><tr><td>Formal Rule Tracking (FRT)</td><td>Maintain task rules, labels, and output constraints</td><td>Que. I</td></tr><tr><td>Answer Commitment (AC)</td><td>Select and stabilize a candidate answer</td><td>Que. II</td></tr><tr><td>Self-Verification (SV)</td><td>Check, revise, or reject the committed candidate</td><td>Que. III</td></tr></table>

the dominant value direction and the AC probe direction. We combine the four indicators into

$$
\begin{array} { r } { R _ { \mathrm { s t r } } ( t ) = \mathsf { \beta } _ { 1 } [ \ S _ { 1 } - \bar { A } _ { \mathrm { s u p } } ( t ) ] _ { \hat { + } } + \mathsf { \beta } _ { 2 } [ \ S _ { 2 } - \mathsf { p } _ { t } ^ { V } ] _ { \hat { + } } } \\ { + \mathsf { \beta } _ { 3 } [ D _ { \mathrm { K L } } ^ { V } ( t ) - \ S _ { 3 } ] _ { \hat { + } } + \mathsf { \beta } _ { 4 } [ \Gamma _ { \mathrm { a c } , V } ( t ) - \ S _ { 4 } ] _ { \hat { + } } } \end{array}\tag{11}
$$

where $[ \cdot ] _ { \hat { + } } = \operatorname* { m a x } ( \cdot , 0 )$ . Thresholds are calibrated on correct reasoning traces: lower-tail thresholds are used for support-state agreement/concentration quantities, and uppertail thresholds for change/alignment quantities. Each component is normalized by its interquartile range. We use equal weights $( \beta _ { k } = 1 / 4 )$ by default and evaluate weight sensitivity in Appendix B.1. $R _ { \mathrm { s t r } }$ is a composite representation-level deviation score instead of one causal factor.

Diagnostic Claim I – Representation Propagation. A failure is not defined by one isolated activation difference. Rather, support-related changes can survive the Attention–MLP update sequence, remain visible in later value-projected representations, and become aligned with AC instead of being canceled by subsequent blocks.

## 3.3 Temporal-State Coordination

At this level analytics, the seven states are treated as "nodes" in a sophisticated stepwise-switching chain. This part is not a comparison of seven isolated score magnitudes, but to reconstruct the intrinsic correlation by which the support and constraint states switch into one another, the point at which the chain enters AC, and whether the subsequent SV activity returns to support formation or revises the committed candidate. We therefore analyze both the state-transition correlation— including repeated resets, skipped support links, and commitment/verification cycles—and their temporal-sate coordination, such as ordering, onset, persistence, and relative timing.

We operationalize the stepwise-switching reasoning chain with four complementary measurements: activation onset identifies the first sustained appearance of a state; activityweighted mean time gives the center of mass of its score curve, dominant-state transitions record the directed links and switching frequency, and relative verification activity measures whether the verification remains active in proportion to commitment. Taken together, these statistics can appropriately describe the ordering, persistence, and revision opportunity of the reasoning chain.

Let $p _ { s } ( t )$ be the causal-pooled probe score of state s at generation step t, and let θ<sub>s</sub> denote state-specific activation threshold. The thresholds are calibrated on the clean-sample reasoning split and fixed before held-out or adversarial evaluation. The reasoning onset of s is set as

$$
\tau _ { s } = \operatorname* { m i n } \{ t | p _ { s } ( t ) \geq \Theta _ { s } , p _ { s } ( t + 1 ) \geq \Theta _ { s } \} .\tag{12}
$$

The two-step condition suppresses the transient-score spikes. The dominant state sequence is $c _ { t } = \mathrm { a r g } \operatorname* { m a x } _ { s } p _ { s } ( t )$ ; and its switch count and aggregate support signal are defined as

$$
\Omega = \sum _ { i = 2 } ^ { T } \mathbf { 1 } [ c _ { i } \neq c _ { i - 1 } ] ,\tag{13}
$$

$$
S ( t ) = \frac { 1 } { | S ^ { + } | } \sum _ { s \in S ^ { + } } p _ { s } ( t ) .\tag{14}
$$

For any nonnegative signal $\boldsymbol { u } = \left( u _ { 1 } , \ldots , u _ { T } \right)$ , we define its center of mass as

$$
\mathbf { C o M } ( u ) = \frac { \sum _ { t = 1 } ^ { T } t u _ { t } } { \sum _ { t = 1 } ^ { T } u _ { t } + \mathfrak { E } } ,\tag{15}
$$

where $\varepsilon > 0$ is a small stabilizer. Let $p _ { \mathrm { c o m } } ( t ) = p _ { A C } ( t )$ and $p _ { \mathrm { v e r } } ( t ) = p _ { S V } ( t )$ . The aggregate-support onset $\tau _ { \mathrm { s u p } }$ is computed by applying the same two-step rule to $S ( t )$ with threshold $\theta _ { \mathrm { s u p } } .$ , calibrated like θ<sub>s</sub>. We quantify commitment/support timing, verification/commitment timing, and relative verification activity by

$$
g _ { c s } = \mathrm { C o M } ( p _ { \mathrm { c o m } } ) - \mathrm { C o M } ( S ) ,\tag{16}
$$

$$
g _ { \nu c } = \mathrm { C o M } ( p _ { \mathrm { v e r } } ) - \mathrm { C o M } ( p _ { \mathrm { c o m } } ) ,\tag{17}
$$

$$
R _ { \nu c } = \frac { A _ { \mathrm { v e r } } } { A _ { \mathrm { c o m } } + \varepsilon } , \quad A _ { x } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } x _ { t } ,\tag{18}
$$

where $g _ { c s } < 0$ is the early-commitment direction, while larger positive $g _ { \nu c }$ indicates later verification relative to commitment. $R _ { \nu c }$ is the verification-to-commitment activity ratio. Correspondingly, we can define the aggregate reasoning processdeviation score as

$$
\begin{array} { c } { { R _ { \mathrm { p r o c } } = \alpha _ { 1 } { \bf 1 } [ \tau _ { \mathrm { c o m } } < \tau _ { \mathrm { s u p } } ] + \alpha _ { 2 } [ - g _ { c s } ] _ { + } + \alpha _ { 3 } \displaystyle \frac { \Omega } { T } } } \\ { { + \alpha _ { 4 } [ g _ { \nu c } ] _ { + } + \alpha _ { 5 } [ 1 - R _ { \nu c } ] _ { + } , } } \end{array}\tag{19}
$$

where $\tau _ { \mathrm { c o m } }$ is the onset of AC.Continuous risk terms are normalized to [0,1] using the 5th and 95th percentiles of correct reasoning traces and then clipped. We choose percentile-based scaling rather than min-max normalization to reduce sensitivity to rare outliers while preserving the typical variation of successful reasoning processes.The default weights are uniform $( \alpha _ { k } = 1 / 5 )$ . The score above combines early-commitment, support–commitment timing inversion, frequent dominantstate switching, late verification, and low verification activity relative to commitment.

Diagnostic Claim II – Temporal-State Coordination. Reliability of LLM reasoning depends on the coordination of the state chain, not on the magnitude of any one singlestate. Failed traces reorganize the handoffs among support-/constraint states, AC, and SV; the onset shifts, temporal gaps, and switching patterns remain model- and failure typedependent.

## 3.4 Reasoning Trajectory

To draw the reasoning state microscopically, we summarize each generation as a reasoning trajectory-feature vector:

$$
\begin{array} { r } { \Phi ( x _ { 1 : T } ) = ( \mathfrak {tau } _ { \mathrm { s u p } } , \mathfrak { \tau } _ { \mathrm { c o m } } , \mathfrak { \tau } _ { \mathrm { v e r } } , \Omega , g _ { c s } , g _ { \nu c } , R _ { \nu c } , L _ { \mathrm { r u l e } } , \bar { S } , T ) , } \end{array}\tag{20}
$$

where $x _ { 1 : T }$ is the sequence of state-coordinate scores, $\tau _ { \mathrm { v e r } } =$ $\tau _ { S V } , L _ { \mathrm { r u l e } }$ is the duration of weak rule/operation tracking, $\bar { S } = A _ { S }$ denotes the mean support strength, and T stands for the generation length. The feature groups correspond to support $( \tau _ { \mathrm { s u p } } , \bar { S } )$ , commitment ordering $( \tau _ { \mathrm { c o m } } , g _ { c s } )$ , verification $( \tau _ { \mathrm { v e r } } , g _ { \nu c } , R _ { \nu c } )$ , state switching (Ω), constraint preservation $\left( L _ { \mathrm { r u l e } } \right)$ , and continuation length (T).

These features characterize complementary aspects of trajectory organization, including support formation $( \tau _ { \mathrm { s u p } } , \bar { S } )$ commitment coordination $( \tau _ { \mathrm { c o m } } , g _ { c s } )$ , verification behavior $( \tau _ { \mathrm { v e r } } , g _ { \nu c } , R _ { \nu c } )$ , transition stability (Ω), constraint preservation $( L _ { \mathrm { r u l e } } )$ , and continuation dynamics (T).

Based on these trajectory coordinates, a well-organized reasoning process is expected to satisfy three general organization principles. First, support-related states should provide sufficient evidence before answer commitment, indicating that candidate selection is grounded in accumulated support rather than premature commitment. Second, functional-state transitions should remain coordinated, with limited unnecessary oscillation or repeated revisits among unrelated states. Third, post-commitment verification should preserve the ability to confirm or revise the current candidate, enabling later computation to correct earlier deviations.

These principles define the expected organization of reasoning trajectories and provide a reference for identifying those abnormal transition patterns. In Section 4.5, we empirically investigate how failed traces depart from these organization principles and characterize recurring trajectory patterns in the proposed feature space.

Diagnostic Claim III – Trajectory Heterogeneity. An endpoint label does not identify a unique failure route. Trajectory Topology preserves differences in support continuity, commitment ordering, rule preservation, state recurrence, verification effectiveness, and continuation length, thereby separating multiple reasoning-process organizations within the same output category.

## 4 Experiment Evaluation

## 4.1 Configuration

Procedure. CauTrace is evaluated as an offline diagnostic framework. For each generation, we (1) collect hidden states and module representations without changing decoding, (2) apply fixed probes to obtain probe score $\{ p _ { s } ( t ) \} _ { s , t } , ( 3 )$ compute the representation-level, temporal-state, and trajectory diagnostics from Section 3, and (4) compare successful and failed generation traces, matched prompt pairs, and model families under fixed development-set calibration. The resulting output is a process-level diagnostic profile rather than a new endpoint classifier.

The following experiments are conducted surrounding four research questions that correspond to the aforementioned three diagnostic claims:

• RQ1 – Categories of LLM Failures: How do failed and successful traces differ in propagation of representation and in the coordination of support, answer commitment, and self-verification?

• RQ2 – Specificity of Reasoning Trajectory: Can the samples with the same output label be distinguished by their reasoning trajectory topology?

• RQ3 – Verification of Adversarial Mechanism: When a harmful prompt-perturbation causes a behavioral flip, are associated the internal changes larger than that in nonflipped pairs and broadly consistent with the targeted or downstream functional states in theory?

• RQ4 – Generalization of Methodology: Are the diagnostic analytics stable to aggregation choices, and transferable across different model families?

## 4.2 Threat Model

For RQ3, we consider a prompt-channel adversary who controls the user prompt and task context. The attacker may introduce misleading premises, retrieval decoys, reordered reasoning cues, incorrect intermediate computations, rulepriority overrides, forced answer commitments, or instructions that suppress verification. An attack succeeds when a cleansuccessful reasoning instance becomes incorrect, unsupported, or task-inconsistent, or when a safe refusal becomes unsafe or partially unsafe compliance. The task objective and evaluation target remain unchanged, enabling pairwise comparison between each adversarial trace and its clean counterpart.

The adversary cannot modify model weights, decoding code, sampling parameters, trace collection, the evaluator, or CauTrace pipeline, and has no access to hidden states or diagnostic outputs during prompt construction. We exclude training-time poisoning, weight editing, tool or retrievalsystem compromise, and deployment-side exfiltration. Whitebox access is reserved for the post-generation analyst, who inspects hidden states, module outputs, layers, heads, and Q/K/V/O after generation; it is not part of the attack model.

## 4.3 Representation-Level: Support-Change Persists Across Updates

To test whether local support-state changes survive later Transformer updates (Que. I and Claim I), we compare Correct and Full Hallucination traces from the nine-dataset clean reliability benchmark across Qwen2.5-1.5B/3B/7B/14B. Table 2 measures whether Attention and MLP move the support coordinates in agreeing directions, while Fig. 2 evaluates the value-projected Attention representation through spectral concentration, step-to-step spectral change, and alignment with the AC probe direction. These measurements are computed from the same held-out analysis-track traces and combine probe-dependent indicators with partly probe-independent value-space statistics.

Support-Related Representation Changes Remain Visible Across Later Transformer Updates, Despite Exact Carrier Is Not Uniform Across Scales. As shown in Table 2, the Attention–MLP update agreement decreases from Correct to Full Hallucination for Qwen2.5-14B (−0.560 → −0.600), 7B $( - 0 . 5 6 8 \to - 0 . 6 2 5 )$ , and 3B $( - 0 . 5 0 0  - 0 . 9 5 0 )$ ; the 3B gap is the largest by −0.450. The 1.5B result moves in the opposite direction $( - 0 . 7 0 0 \to - 0 . 5 0 0 )$ , possibly because the substantially smaller model scale leads to different internal representation dynamics and update coordination patterns.This observation suggests that $\bar { A } _ { \mathrm { s u p } }$ should not be interpreted as a scale-invariant failure classifier, but rather as a mechanistic indicator of support-related propagation changes.

The value-space panels show why the conclusion is nevertheless about propagation rather than one statistic. Throughout Fig. 2(a), the failed traces retain a much lower but nonzero leading value-spectrum concentration than correct traces; Fig. 2(b) shows the repeated failure-associated spectral change spikes, most visibly near the middle and final 20% of generation; and Fig. 2(c) shows failed traces becoming sharply aligned with the AC direction only near the endpoint, whereas correct traces remain commitment-aligned across a broader portion of the trace. The three panels therefore clearly connect a changed support representation to later commitment organization: the deviation is not erased after its local update, but remains available and is reorganized near commitment.

Table 2: Representation-level indicators. Lower $\bar { A } _ { \mathrm { s u p } }$ denotes lower Attention and MLP updates to support states.
<table><tr><td>Model</td><td>Label</td><td> $\Delta _ { s u p } ^ { a t t n }$ </td><td> $\Delta _ { s u p } ^ { m l p }$ </td><td> $\bar { A } _ { \mathrm { s u p } }$ </td></tr><tr><td>Qwen2.5-14B</td><td>Correct</td><td>0.076</td><td>-0.015</td><td>-0.560</td></tr><tr><td>Qwen2.5-14B</td><td>Full Hall.</td><td>0.125</td><td>-0.160</td><td>-0.600</td></tr><tr><td>Qwen2.5-7B</td><td>Correct</td><td>-0.009</td><td>0.009</td><td>-0.568</td></tr><tr><td>Qwen2.5-7B</td><td>Full Hall.</td><td>0.019</td><td>-0.056</td><td>-0.625</td></tr><tr><td>Qwen2.5-3B</td><td>Correct</td><td>0.023</td><td>0.029</td><td>-0.500</td></tr><tr><td>Qwen2.5-3B</td><td>Full Hall.</td><td>0.041</td><td>-0.050</td><td>-0.950</td></tr><tr><td>Qwen2.5-1.5B</td><td>Correct</td><td>0.010</td><td>0.039</td><td>-0.700</td></tr><tr><td>Qwen2.5-1.5B</td><td>Full Hall.</td><td>0.025</td><td>-0.060</td><td>-0.500</td></tr></table>

Together with the cross-scale exception in Table 2, this supports Claim I as a propagated diagnostic association, not a uniquely localized circuit.

## 4.4 Temporal-State Level: Coordination of Failure-Reorganizing

To verify Claim II, we compare Correct and Full Hallucination traces from the same analysis track at two resolutions. Fig. 3 reports the aggregate reasoning process-deviation score and its components across Qwen2.5 scales; Table 3 tests the relative placement and activity of support, AC, and SV; and Fig. 4 shows which dominant-state handoffs gain or lose probability. The combination calibrate the coordination rather than treating any one single-state score as a failure label. We compute q-values from two-sided unpaired permutation tests with Benjamini–Hochberg correction across all 15 comparisons.

Failures Consistently Show Insufficient Verification After Commitment, While Their Timing Relationship Varies Across Models. Across models, failed traces exhibit a consistently lower verification-to-commitment ratio $R _ { \nu c }$ , indicating that verification activity becomes insufficient relative to commitment after the answer direction is formed. Pooled across models, $R _ { \nu c }$ decreases from 4.172 to 1.065 $( \Delta = - 3 . 1 0 7$ $q < 0 . 0 0 1 )$ as shown in Table 3. This reduction is significant across all Qwen2.5 scales, with failure–correct differences of −7.783, −5.292, −1.192, and −1.752 for 1.5B, 3B, 7B, and 14B, respectively. However, the exact temporal relationship between verification and commitment is not identical across models: after FDR correction, the verification–commitment gap is not significant for 1.5B and 7B. Therefore, Claim II is supported by a consistent reduction in verification coverage after commitment, rather than by assuming a universal timing reversal across all models.

Fig. 3(a) provides the aggregate view: Full Hallucination has a higher aggregate reasoning process-deviation score $R _ { \mathrm { p r o c } }$ bar than Correct at every model size, with the largest visible separation at 3B, while the overlaid verification-onset indicator varies substantially across scales. The decomposition in Fig. 3(b) further shows that the aggregate change is not a uniform rise in all five components: early-commitment and late-verification contributions increase, whereas other components change only slightly or in the opposite direction. This is precisely the coordination account in Claim II—different component combinations can produce failure, but inadequate checking relative to commitment remains the common relation confirmed by Table 3.

![](images/74e79511612fa9d88cd704ce48e3b05f7ceccf5a7b48e3555c00756ff1f60aa3.jpg)  
(a) Value-projected spectral concentration $\rho _ { t } ^ { V }$

![](images/75ddcce672bd768a2b8faefaeeed33ae7c40acb288d05fd7905c9bd0d88b5b4c.jpg)  
(b) Value-spectrum change $D _ { \mathrm { K L } } ^ { V } ( t )$

![](images/1a23cf558d9c93bd3439207fff3cee9b778d2f021ff939c73ac3622e86508cc1.jpg)  
(c) Commitment alignment $\Gamma _ { \mathrm { a c } , V } ( t )$  
Figure 2: Representation-level diagnostics. Panel (a) reports value-projected Attention spectral concentration $\rho _ { t } ^ { V }$ ; Panel (b) reports step-to-step change in the normalized value spectrum, $D _ { \mathrm { K L } } ^ { V } ( t )$ ; Panel (c) reports alignment $\Gamma _ { \mathrm { a c } , V } ( t )$ between the dominant value-space direction and the AC probe direction

The transition matrices identify the links underlying that aggregate shift. Relative to correct traces, Fig. 4(c) shows that failed traces lose probability on support-preserving handoffs, such as KR→FRT (−0.11) and PB→RC (−0.08), while gaining probability on KR→AC (+0.08), SV→AC (+0.06), and RC→KR (+0.06). These selective shifts in Fig. 4(c) mean that failure does not merely reduce all state activity: support construction is rerouted toward commitment or reset, and verification is more likely to return to the committed candidate. This discrete link evidence explains how the lower $R _ { \nu c }$ in Table 3 appears as a reorganization of the state chain.

## 4.5 Trajectory-Topology Level: One Output Label Contains Multiple Routes

To validate Claim III, we summarize the same held-out Full Hallucination traces with the trajectory features in Eq. (20) and compare the five failure-prone regions defined in Table 16 in Appendix B.4. Fig. 5 tests whether the regions occupy distinguishable neighborhoods in the full feature space, Fig. 6 exposes their time-domain organization, and Fig. 7 makes the corresponding state-order motifs explicit.

Full Hallucination Exhibits Topologically Heterogeneity: Some Routes Are Sharply Distinguishable, While Others Overlap because They Share Only Part of Support–Commitment–Verification Organization. In Fig. 5(a), the first two projection axes explain only 15.5% + 12.7% = 28.2% of trajectory-feature variance, yet Early-Commitment and Verification without Overwrite occupy visibly different neighborhoods. The row-normalized matrix in Fig. 5(b) quantifies this asymmetry: Early-Commitment and Verification without Overwrite have diagonal proportions of 0.99 and 0.84, whereas Fragmented Support and Rule Failure have only 0.09 and 0.25 on the diagonal and are often mapped to Verification without Overwrite. The result validates Claim III without overstating separability: the endpoint category contains multiple routes, but the routes are not rigidly-delineated, but overlapping-existing regions.

The time-domain panels explain what the projection preserves. Normal Convergence in Fig. 6(a) maintains strong support through the first half of generation and then transfers activity to verification before the trace ends. Early Commitment in Fig. 6(b) instead begins with substantial commitment activity and returns to a strong commitment near the endpoint; Fragmented Support in Fig. 6(c) retains high but uneven support while verification stays weak; and Rule Failure in Fig. 6(d) shows support decaying while commitment grows without a comparably strong verification response. The last two panels separate two superficially similar long failures: Verification without Overwrite develops substantial late verification while commitment persists in Fig. 6(e), whereas Verbose Unsupported Continuation loses support after midgeneration and ends in a renewed commitment spike with little remaining verification in Fig. 6(f). These panel-specific differ ences interpret the temporal meaning of reasoning trajectory, i.e. the same Full Hallucination endpoint can result from premature selection, disrupted support, weak rule preservation, ineffective checking, or unsupported continuation.

Fig. 7 translates those curve-shapes into state-order motifs. The Early-Commitment examples enter AC before later RC/KR re-entry; The Fragmented Support examples repeat KR/RC resets; and the Verification without Overwrite examples explicitly return from SV to AC. The Verbose Unsupported Continuation examples are the longest and most recurrent in the figure $( T = 1 0 , R = 7$ and $T = 1 1 , R = 8 )$ consisting of repeated commitment/support re-entry without proportional verification. The above analytics validate Claim III by showing that trajectory types encode families of routes rather than merely one token-level chain per output label.

Table 3: Timing comparisons between Correct and Full Hallucination traces. Each cell reports Correct → Full Hallucination with ∆ = Full Hallucination−Correct. For $_ { g _ { c s } }$ and $g _ { \nu c } ,$ , the measured quantity is a shift in relative temporal position; negative $_ { g _ { c s } }$ denotes the early-commitment risk direction. Lower $R _ { \nu c }$ indicates less verification activity relative to answer commitment.
<table><tr><td>Model</td><td> $g _ { c s } \colon \mathbf { s u p p o r t - c o m m i t m e n t ~ g a p }$  negative = early commitment; shift large shift / sign flip = disrupted ratio = change</td><td> $_ { g _ { \nu c } : }$  verification-commitment gap checking</td><td> $R _ { \nu c } \colon$  verification/commitment lower ratio = less relative veri- fication activity</td></tr><tr><td>All models</td><td> $1 4 . 0 2 5  0 . 4 8 9$   $\Delta = - 1 3 . 5 3 6 , q < 0 . 0 0 1$ </td><td> $- 0 . 2 7 4  1 6 . 3 1 1$   $\Delta = 1 6 . 5 8 5 , q < 0 . 0 0 1$ </td><td> $4 . 1 7 2  1 . 0 6 5$   $\Delta = - 3 . 1 0 7 , q < 0 . 0 0 1$ </td></tr><tr><td>Qwen2.5-1.5B</td><td> $1 4 . 1 8 2  - 1 . 4 3 2$   $\Delta = \mathbf { - 1 5 . 6 1 4 } , q = 0 . 0 3 4$ </td><td>not significant after FDR correction</td><td> $9 . 0 6 4  1 . 2 8 1$   $\Delta = - 7 . 7 8 3 , q = 0 . 0 1 1$ </td></tr><tr><td>Qwen2.5-3B</td><td> $2 4 . 2 8 3  3 . 1 2 5$   $\Delta = - 2 1 . 1 5 9 , q < 0 . 0 0 1$ </td><td> $4 . 9 5 1  2 0 . 6 1 6$   $\Delta = 1 5 . 6 6 4 , q = 0 . 0 1 1$ </td><td> $6 . 4 7 3  1 . 1 8 1$   $\Delta = - 5 . 2 9 2 , q < 0 . 0 0 1$ </td></tr><tr><td>Qwen2.5-7B</td><td> $3 . 7 5 0 \to - 0 . 8 5 7$ </td><td>not significant after FDR correction</td><td> $1 . 7 8 4  0 . 5 9 2$   $\Delta = \mathbf { - 1 . 1 9 } 2 , q = 0 . 0 1 3$ </td></tr><tr><td>Qwen2.5-14B</td><td> $\Delta = . 4 . 6 0 7 , q = 0 . 0 4 2$   $9 . 3 1 9  0 . 3 3 3$   $\Delta = - 8 . 9 8 6 , q < 0 . 0 0 1$ </td><td> $- 3 . 9 4 0  2 1 . 4 3 7$   $\Delta = 2 5 . 3 7 6 , q < 0 . 0 0 1$ </td><td> $2 . 4 7 4  0 . 7 2 2$   $\Delta = - 1 . 7 5 2 , q = 0 . 0 1 8$ </td></tr></table>

## 4.6 Adversarial Test: Behavioral Flips Coincide with Larger Internal Changes

To test whether the adversarial prompt-induced failures disturb the same reasoning-process organization as clean failures (RQ3), we compare each adversarial trace with a matched clean trace under a fixed task objective. Table 4 reports 500 reasoning-interference pairs and 300 AdvBench/Harm-Bench safety pairs per model; Table 5 then compares state-coordinate, cascade, onset, and Q/K/V/O changes for flipped and non-flipped pairs, here a flip means that a clean correct/evidence-grounded answer becomes erroneous or unsupported, or that a clean safe refusal becomes unsafe or partially unsafe compliance. The two tables therefore separate endpoint vulnerability from the internal route associated with a successful adversarial perturbation.

![](images/be3d10930762c3ad3c1a7e2fcbee7715b617401afc9e581d3340593702e2d20f.jpg)  
(a) Composite process-deviation score across model sizes

Behavioral Flips Are Accompanied by Broad Changes in Same Functional-State Coordinates. In the reasoninginterference rows of Table 4, PFR ranges from 1.8% to 6.2% and conditional PFR from 10.0% to 26.1%; in the safety rows, PFR is higher at 12.7%–22.3% and conditional PFR at 14.7%–30.5%. These paired flips establish that the prompt intervention can change the endpoint under a matched objective. However, TAR@1 is only 15.2%–16.6% versus the $1 / 7 = 1 4 . 3 \%$ chance level, and TAR@3 is 41.8%–49.8% versus $3 / 7 = 4 2 . 9 \%$ , as shown in Table 4, and Appendices A.4 and A.5 provide the paired construction, target-state taxonomy, and metric definitions. We leverage the working mechanism to test related reasoning-process changes in theory, while treating exact target-state identification as unsupported.

The flip-associated columns in Table 5 provide the processlevel support for the claim. Reasoning-interference flips show their strongest contrasts in OC for Qwen2.5-1.5B (d = $- 3 . 1 8 0 )$ and 7B $( d = - 3 . 1 3 4 )$ , and in FRT onset for 3B $( d = 2 . 9 8 0 )$ , all in Table 5. These are support/constraint coordinates from Claim II, thus their disruption explains why a harmful-benign-paired prompt change can reroute the chain before the endpoint flips. Furthermore, safety flips follow a different route: the strongest state/cascade indicators are organized around AC, including peak-layer shifts of $d = 0 . 6 4 2$ for 3B and $d = 0 . 6 3 6$ for 7B. This contrast supports a shared coordinate system, not a shared single route: attacks primarily disturb operation/rule support, whereas successful attacks are more closely associated with commitment organization.

![](images/41d0e32718d4dc21626a5d08eda9e92adb2aa021e82848c1f1ffaa869f7ad0af.jpg)  
(b) Process-deviation score components  
Figure 3: Temporal-state coordination diagnostics: Panel (a) compares $R _ { \mathrm { p r o c } }$ and the verification-onset indicator $I _ { \mathrm { v e r } } ;$ Panel (b) decomposes $R _ { \mathrm { p r o c } }$ into early commitment onset, support delay, oscillation, late verification, and low verification coverage.

3verification/overwrite: Commit/Verify transitions  
![](images/d9972fd899a720b80ab5ab6cd6cc24a5c10a6a2aa8d273d1de560d2ef3d0d568.jpg)  
(a) Correct trajectories

![](images/33352330181603570b24dbab26493206f3425695c614f43d5743cbdb04c95e7e.jpg)

(b) Failed trajectories  
![](images/d2b4756998cb58f77adb8efebdb083cded6e745639bd7328b0ee595de443f2cf.jpg)  
1support construction: Bind/Retrieve/Reason transitions2commitment persistence: transitions into Commit  
(c) Failure-minus-correct shift

Figure 4: State-transition matrices for correct and failed trajectories: Panels (a) and (b) report transition probabilities for correct and failed generations, respectively; Panel (c) reports the failure-minus-correct difference.  
![](images/a88026765bf7b749a4148c5fe23ca0f1e48fc666bfd2457fb15b90031aed8631.jpg)  
(a) Trajectory-feature projection

![](images/9fed5c83c894b6a67a9060ee28b532e6e3efc88c6542c23989558a2807b2703e.jpg)  
(b) Type-prediction confusion matrix  
Figure 5: Trajectory topology in the trajectory-feature space: Panel (a) visualizes trajectory-level features with approximate type neighborhoods; Panel (b) gives the row-normalized confusion matrix of a lightweight rule-derived trajectory predictor constructed directly from the diagnostic definitions in Eq. (20).

Table 4: Paired adversarial test results. BSR denotes clean-side success, AFR is adversarial failure rate, PFR is pair flip rate, and conditional PFR measures flips among clean-successful pairs. Top-k Target Accuracy Rate (TAR) is reported for process reasoning perturbations, where each perturbation defines an explicit seven-state target.
<table><tr><td>Dataset</td><td>Model</td><td>Pairs</td><td>BSR</td><td>AFR</td><td>PFR</td><td>Cond.PFR</td><td>TAR@1</td><td>TAR@3</td></tr><tr><td rowspan="3">Reasoning Interfered</td><td>Qwen2.5-1.5B</td><td>500</td><td>23.8%</td><td>34.0%</td><td>6.2%</td><td>26.1%</td><td>16.6%</td><td>49.8%</td></tr><tr><td>Qwen2.5-3B</td><td>500</td><td>18.0%</td><td>14.0%</td><td>1.8%</td><td>10.0%</td><td>15.2%</td><td>41.8%</td></tr><tr><td>Qwen2.5-7B</td><td>500</td><td>26.6%</td><td>21.0%</td><td>4.8%</td><td>18.0%</td><td>16.6%</td><td>48.4%</td></tr><tr><td rowspan="3">AdvBench/HarmBench</td><td>Qwen2.5-1.5B</td><td>300</td><td>73.3%</td><td>37.0%</td><td>22.3%</td><td>30.5%</td><td>一</td><td></td></tr><tr><td>Qwen2.5-3B</td><td>300</td><td>69.3%</td><td>33.3%</td><td>14.7%</td><td>21.2%</td><td>一</td><td>一</td></tr><tr><td>Qwen2.5-7B</td><td>300</td><td>86.0%</td><td>18.0%</td><td>12.7%</td><td>14.7%</td><td></td><td></td></tr></table>

![](images/e82f3e96818da317d07961270d064ebfa10e38137e0597d477c38359a21abfd9.jpg)  
forward transition revisit / backward jump start end

![](images/6bf3a83f9017a5971f7914fbbc642b92a2eba0da2866b7b8eee3f3b02e79bdca.jpg)  
(a) Normal Convergence

![](images/7d1cdff59282d5f94a7f5eb7287ea444de072bb35fa3438d6a4584728662cb42.jpg)  
(b) Early-Commitment

![](images/1ac1b5a5498674a6ede106584f2308155145109d67cc04b8644c22d48b178328.jpg)  
(c) Fragmented Support

![](images/d746717465bca18ce25e8951456fd06876bb6a3cb15688ece8b3237e1f5dffc8.jpg)  
(d) Rule Failure

![](images/4eb2b8cff2ec7171866a17bcfa1f59f9997ee230594b75f12979d7aabb58dc0c.jpg)  
(e) Verification without Overwrite

![](images/f92d95cfac57fb92a47832ccecafd2a2a7575eb20cec2b2ba738e73a7273e331.jpg)  
(f) Verbose Unsupported Continuation  
Figure 6: Time-domain state-score trajectories for one normal and five failure-prone trajectory types. The curves show how each trajectory pattern differs in support formation, answer commitment, and self-verification over the generation process.

![](images/5a9b7301685b6090128a7c4f5379deb9a2394ab5367506926e4c6b3c20c23b44.jpg)

![](images/bb7de880f4a320e6e527a61c0fed4521d92446b5571c23fe1bf4e015fa308898.jpg)

![](images/e08fd482690151efa20aea2662c9ccaa0df602d298e889c656d5e668b6671da7.jpg)

![](images/cbd29ff514d4d972463ade246b499ead24531ec90e80bf129a4805547c1931c8.jpg)  
Figure 7: Representative state-transition chains for the five failure-prone trajectory types.

Table 5: Internal changes under paired adversarial perturbations.
<table><tr><td>Dataset</td><td>Model</td><td>Dominant State Shift</td><td>Strongest Flip-Associated Signal</td><td>Strongest Paired QKV Sig- nal</td></tr><tr><td rowspan="3">Reasoning Interfered</td><td>Qwen2.5-1.5B</td><td>Premise Binding —0.133</td><td>OC cascade (d = −3.180)</td><td>K pairwise cosine +0.003  $( d _ { z } = 0 . 3 5 2 )$  Q rank -0.038</td></tr><tr><td>Qwen2.5-3B</td><td>Reasoning +0.115</td><td>composition Formal Rule Tracking on- set delay (d = 2.980)</td><td>stable  $( d _ { z } = - 0 . 9 7 5 )$ </td></tr><tr><td>Qwen2.5-7B</td><td>Reasoning -0.058</td><td>Composition OC state (d = −3.134)</td><td>V-0 norm correlation  $+ 0 . 0 6 3 ( d _ { z } = 0 . 3 1 5 )$ </td></tr><tr><td rowspan="3">AdvBench /HarmBench</td><td>Qwen2.5-1.5B</td><td>Answer +0.057</td><td>commitment</td><td>Commitment cascade 0 strength (d = 0.232) Commitment peak-layer</td></tr><tr><td>Qwen2.5-3B</td><td>Answer -0.001</td><td>commitment shift (d = 0.642)</td><td> $- 0 . 0 7 8 ( d _ { z } = - 1 . 2 8 9 )$  Q pairwise-cosine disper- sion  $+ 0 . 0 0 6 ( d _ { z } = 0 . 4 8 5 )$ </td></tr><tr><td>Qwen2.5-7B</td><td>Answer +0.002</td><td>commitment Commitment peak-layer shift (d = 0.636)</td><td>O row-norm dispersion  $- 0 . 0 3 1 ( d _ { z } = - 0 . 6 6 1 )$ </td></tr></table>

The Q/K/V/O column in Table 5 links those process-level changes back to Claim I. The largest paired representation effects are the O consecutive-cosine shift for 1.5B safety pairs $( d _ { z } = - 1 . 2 8 9 )$ and the Q stable-rank shift for 3B reasoning pairs $( d _ { z } = - 0 . 9 7 5 )$ , as reported in Table 5. Thus, the successful adversarial perturbations are visible in both reasoning-process coordinates and fine-grained Transformer representations, while the changing Q/K/V/O carrier again argues against one universal circuit. Held-out checks in Tables 17 and 18 show the largest observed 1.5B reasoning reduction with unified repair (76.0% to 71.0%) and predicted 7B aligned-steering shifts in risk, rule, and verification. The adversarial evidence supports adversarial perturbation-linked, intervention-sensitive changes in the three-level analytics.

## 4.7 Generalization

To assess the generalizability of our proposed analytics framework, beyond Qwen2.5, we also apply the Correct/Full Hallucination analysis to LLaMA-3.2-1B/3B, LLaMA-3.1-8B, and Qwen1.5-MoE and Table 13 reports failure-minus-correct differences in the representation-level indicators and in the verification activity relative to commitment.

The Coordination Result Transfers Across Model Families More Consistently Than Any Individual

Representation-Level Carrier. All four models show lower support-state update agreement for failed traces, with $\Delta \bar { A } _ { \mathrm { s u p } }$ ranging from −0.012 to −0.052 as shown in Table 13. They also show lower "verification-to-commitment" activity, with $\Delta R _ { \nu c }$ ranging from −1.102 to −15.519. These shared directions support the central relations as stated in Claims I-II. By contrast, $\Delta \rho _ { V }$ and $\Delta \Gamma _ { \mathrm { a c } , V }$ change sign across models, and $\Delta D _ { K L } ^ { V }$ is positive for the three dense LLaMA models but approximately zero for Qwen1.5-MoE. The cross-family results therefore support representation-propagating differences and weaker relative verification at the diagnostic level while indicating that the measurable carrier depends on architecture.

## 5 Conclusion

We have quantitatively verified the proposed three diagnostic claims through three-level hierarchical analytics. Our work fills a reasoning process-level gap for LLM reliability and safety inspections: final-output labels identify failures but not how they emerge during the entire process of generation. By organizing the hidden-state traces into seven functionalstate coordinates, our proposed CauTrace connects the representation changes across Transformer blocks, temporal-state coordination among support/commitment/verification, and trajectory topology over the full reasoning trajectory chain. Across Qwen and LLaMA-family models, our experiments show lower support-state update agreement and lower verification activity relative to commitment in failed traces, different trajectory topologies among samples with the same output label, and larger internal changes under successful adversarial perturbations. The evidence supports a unified diagnostic account of failure dynamics and provides a foundation for more fine-grained causal localization. We believe that our work can provide a foundation for future research on LLM safety and inspire further efforts toward safer and more reliable LLMs.

## Ethics Considerations

In developing CauTrace, we considered the ethical implications of combining white-box diagnostics with adversarial evaluation of large language models. The work proposes an offline analytical framework rather than an attack system: adversarial prompts are introduced only as controlled stress tests of whether ordinary and induced failures exhibit related internal process changes. All experiments use locally hosted open-weight models and public benchmarks; they involve no human participants, private user data, unauthorized access, or attacks on deployed services.

Stakeholders and Potential Impact. The principal stakeholders are LLM users, individuals and communities affected by unsupported or harmful outputs, model developers, AI reliability and safety researchers, and benchmark maintainers. CauTrace may benefit these groups by clarifying how evidence support, answer commitment, rule tracking, and selfverification become miscoordinated before an unreliable or unsafe endpoint. Such diagnostics can inform future monitoring and mitigation, but they should not be interpreted as a per-generation safety guarantee or as a substitute for deployment-specific evaluation.

Dual-Use Risks and Responsible Disclosure. The same internal patterns that help diagnose failure could be misused to design prompts that encourage premature commitment, weaken rule tracking, or suppress verification. We limit this risk in both experimental scope and artifact release. The threat model is restricted to prompt-side perturbations against locally hosted models, and the adversary has no access to model weights, hidden states, diagnostic outputs, evaluators, or deployed systems during prompt construction. We will not release turnkey jailbreak pipelines, operational instructions for bypassing safeguards, unredacted harmful-prompt collections, or harmful model completions. If future use of CauTrace reveals a deployment-specific vulnerability, we will coordinate disclosure with the relevant model or system provider before releasing operational details.

Handling of Harmful Content. The paired safety evaluation may expose models to harmful prompts or produce unsafe responses. We reduce researcher exposure through automated safety judges, predefined output labels, and structured filtering, with manual inspection limited to debugging and validation. Harmful instructions are never executed, and potentially unsafe prompts or generations are neither reproduced in the paper nor redistributed in the artifact. Reported findings use aggregate statistics and sanitized identifiers wherever possible.

Balancing Potential Benefits and Harms. CauTrace is intended to improve failure analysis by linking endpoint behavior to Propagation of Representation, Temporal-State Coordination, and Reasoning Trajectory. This evidence can help researchers distinguish failures that share an output label but require different monitoring or mitigation strategies. Although the resulting diagnostic knowledge retains some dual-use potential, the controlled evaluation setting, bounded threat model, and restricted release policy substantially reduce operational misuse. We therefore judge the expected reliability and safety benefits to outweigh the residual risks under these safeguards.

## Open Science

Reproducibility is a central goal of this work. We will provide an anonymous artifact containing the materials required to reproduce the reported diagnostic analyses while applying the safety restrictions described above.

Data and Model Use. The experiments use publicly available reasoning and safety benchmarks, including the cleanreasoning datasets documented in Appendix A.2 and the AdvBench/HarmBench evaluation sets, together with publicly released Qwen and LLaMA-family model checkpoints. These resources are used only for controlled evaluation and will remain subject to the licenses and usage conditions of their original providers. We collect no new human-subject data or personally identifiable information. Where redistribution of raw or harmful content is inappropriate, the artifact will provide source references, dataset manifests, version records, and sanitized instance identifiers rather than repackaging that content.

Code and Artifacts. The artifact will include the Cau-Trace implementation and environment specification; model and dataset version records; scripts for trace collection, functional-state/process-coordinate scoring, probe training, weak-label construction, evaluation, statistical testing, and figure generation; fixed data splits and configuration files; and sanitized metadata and aggregate outputs for the matched clean/adversarial analysis. It will exclude unsafe model completions, unredacted harmful prompts, turnkey jailbreak pipelines, and operational instructions for circumventing safeguards. The review artifact will be available at https://anonymous.4open.science/r/artifact-B43D; following review, the released version will be preserved in a persistent archival repository.

## References

[1] Cem Anil, Esin Durmus, Nina Panickssery, Mrinank Sharma, Joe Benton, Sandipan Kundu, Joshua Batson, Meg Tong, Jesse Mu, Daniel Ford, et al. Many-shot jailbreaking. In Advances in Neural Information Processing Systems, volume 37, 2024.

[2] Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. Refusal in language models is mediated by a single direction. In Advances in Neural Information Processing Systems, volume 37, 2024.

[3] Nora Belrose, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith, Danny Halawi, Stella Biderman, and Jacob Steinhardt. Eliciting latent predictions from transformers with the tuned lens. arXiv preprint arXiv:2303.08112, 2023.

[4] Matan Ben-Tov, Mor Geva, and Mahmood Sharif. Universal jailbreak suffixes are strong attention hijackers. Transactions of the Association for Computational Linguistics, 14:1028–1050, 2026. doi:10.1162/TACL.a. 695.

[5] Collin Burns, Haotian Ye, Dan Klein, and Jacob Stein hardt. Discovering latent knowledge in language models without supervision. In International Conference on Learning Representations, 2023.

[6] Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramèr, et al. Jailbreakbench: An open robustness benchmark for jailbreaking large language models. In Advances in Neural Information Processing Systems, volume 37, 2024.

[7] Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. In Advances in Neural Information Processing Systems, volume 36, 2023.

[8] Janet E Davidson and Robert J Sternberg. The psychol ogy of problem solving. Cambridge university press, 2003.

[9] Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 5484–5495, 2021.

[10] Asma Ghandeharioun, Avi Caciularu, Adam Pearce, Lucas Dixon, and Mor Geva. Patchscopes: A unifying framework for inspecting hidden representations of language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 15466– 15490. PMLR, 2024.

[11] Abhay Gupta, Kevin Zhu, Vasu Sharma, Sean O’Brien, and Michael Lu. NovelHopQA: Diagnosing multi-hop reasoning failures in long narrative contexts. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 26134– 26151. Association for Computational Linguistics, 2025. doi:10.18653/v1/2025.emnlp-main.1328.

[12] Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. In International Conference on Learning Representations, 2024.

[13] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1–55, 2025.

[14] Alon Jacovi and Yoav Goldberg. Towards faithfully interpretable nlp systems: How should we define and evaluate faithfulness? In Proceedings ofthe 58th annual meeting of the association for computational linguistics, pages 4198–4205, 2020.

[15] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM computing surveys, 55(12):1– 38, 2023.

[16] Yuqi Jia, Yupei Liu, Zedian Shao, Jinyuan Jia, and Neil Zhenqiang Gong. Promptlocate: Localizing prompt injection attacks. In 2026 IEEE Symposium on Security and Privacy (SP). IEEE, 2026. doi:10.1109/SP63933. 2026.00105.

[17] Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin Jiang, Qun Liu, and Wei Wang. Followbench: A multi-level fine-grained constraints following benchmark for large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4667–4688, 2024.

[18] Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

[19] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.

[20] Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, et al. Measuring faithfulness in chain-ofthought reasoning. arXiv preprint arXiv:2307.13702, 2023.

[21] Chak Tou Leong, Qingyu Yin, Jian Wang, and Wenjie Li. Why safeguarded ships run aground? aligned large language models’ safety mechanisms tend to be anchored in the template region. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15212–15229. Association for Computational Linguistics, 2025. doi:10.18653/v1/2025.acl-long.738.

[22] Junyi Li, Xiaoxue Cheng, Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. HaluEval: A large-scale hallucina tion evaluation benchmark for large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6449– 6464. Association for Computational Linguistics, 2023. doi:10.18653/v1/2023.emnlp-main.397.

[23] Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings ofthe 60th annual meeting ofthe association for computational linguistics (volume 1: long papers), pages 3214–3252, 2022.

[24] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173, 2024. doi: 10.1162/tacl\_a\_00638.

[25] Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. Autodan: Generating stealthy jailbreak prompts on aligned large language models. In International Conference on Learning Representations, 2024.

[26] Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, and Neil Zhenqiang Gong. Formalizing and benchmarking prompt injection attacks and defenses. In 33rd USENIX Security Symposium (USENIX Security 24), pages 1831– 1847. USENIX Association, 2024.

[27] Potsawee Manakul, Adian Liusie, and Mark Gales. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 9004–9017, 2023.

[28] Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, and Dan Hendrycks. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal. In Proceedings of the International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 35181–35224, 2024.

[29] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt. Advances in neural information processing systems, 35:17359–17372, 2022.

[30] Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12076–12100. Association for Computational Linguistics, 2023. doi:10.18653/v1/2023.emnlp-main. 741.

[31] Neel Nanda and Joseph Bloom. Transformerlens. https://github.com/TransformerLensOrg/ TransformerLens, 2022.

[32] Allen Newell, Herbert Alexander Simon, et al. Human problem solving, volume 104. Prentice-hall Englewood Cliffs, NJ, 1972.

[33] Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. arXiv preprint arXiv:2112.00114, 2021.

[34] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. Incontext learning and induction heads. arXiv preprint arXiv:2209.11895, 2022.

[35] Wenbo Pan, Zhichao Liu, Qiguang Chen, Xiangyang Zhou, Yu Haining, and Xiaohua Jia. The hidden dimen sions of llm alignment: A multi-dimensional analysis of orthogonal safety directions. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 47697–47716, 2025.

[36] Debjit Paul, Robert West, Antoine Bosselut, and Boi Faltings. Making reasoning matter: Measuring and improving faithfulness of chain-of-thought reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 15012–15032, 2024.

[37] Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, and Yang Zhang. Do anything now: Characterizing and evaluating in-the-wild jailbreak prompts on large language models. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security. ACM, 2024.

[38] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi, Nathanael Schärli, and Denny Zhou. Large language models can be easily distracted by irrelevant context. In International Conference on Machine Learning, pages 31210–31227. PMLR, 2023.

[39] Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, and Neil Zhenqiang Gong. Optimization-based prompt injection attack to llm-asa-judge. In Proceedings ofthe 2024 on ACM SIGSAC Conference on Computer and Communications Security, pages 660–674. ACM, 2024. doi:10.1145/3658644. 3690291.

[40] Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel, Justin Svegliato, Scott Emmons, Olivia Watkins, et al. A strongreject for empty jailbreaks. In Advances in Neural Information Processing Systems, volume 37, 2024.

[41] Miles Turpin, Julian Michael, Ethan Perez, and Samuel Bowman. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. Advances in Neural Information Processing Systems, 36:74952–74965, 2023.

[42] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations, 2023.

[43] Yuxia Wang, Minghan Wang, Muhammad Arslan Manzoor, Fei Liu, Georgi Nenkov Georgiev, Rocktim Jyoti Das, and Preslav Nakov. Factuality of large language models: A survey. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 19519–19529, 2024.

[44] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[45] Tom Wollschläger, Jannes Elstner, Simon Geisler, Vin cent Cohen-Addad, Stephan Günnemann, and Johannes Gasteiger. The geometry of refusal in large language models: Concept cones and representational independence. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 66945–66970, 2025.

[46] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chen-

gen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[47] Zhangyue Yin, Qiushi Sun, Qipeng Guo, Jiawen Wu, Xipeng Qiu, and Xuan-Jing Huang. Do large language models know what they don’t know? In Findings of the associationfor Computational Linguistics: ACL 2023, pages 8653–8665, 2023.

[48] Zhiyuan Yu, Xiaogeng Liu, Shunning Liang, Zach Cameron, Chaowei Xiao, and Ning Zhang. Don’t listen to me: Understanding and exploring jailbreak prompts of large language models. In 33rd USENIX Security Symposium (USENIX Security 24), pages 4675–4692. USENIX Association, 2024.

[49] Shenyi Zhang, Yuchen Zhai, Keyan Guo, Hongxin Hu, Shengnan Guo, Zheng Fang, Lingchen Zhao, Chao Shen, Cong Wang, and Qian Wang. {JBShield}: Defending large language models from jailbreak attacks through activated concept analysis and manipulation. In 34th USENIX Security Symposium (USENIX Security 25), pages 8215–8234, 2025.

[50] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, et al. Siren’s song in the ai ocean: A survey on hallucination in large language models. Computational Linguistics, 51(4):1373–1418, 2025.

[51] Zhenhong Zhou, Haiyang Yu, Xinghua Zhang, Rongwu Xu, Fei Huang, and Yongbin Li. How alignment and jailbreak work: Explain llm safety through intermediate hidden states. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 2461–2488. Association for Computational Linguistics, 2024.

[52] Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405, 2023.

## A Experimental Protocol and Details

This appendix provides the experimental details and supplemental results needed to interpret the main evaluation, including benchmark composition, output evaluation, probe training, paired-adversary construction, and Trajectory Topology evidence.

## A.1 Evaluation Configuration

Each sample is evaluated under concise and analysis prompt tracks. Unless otherwise stated, process-level results use the analysis track, which requests two to four short reasoning steps followed by a constrained final-answer line. The evaluator assigns one of five task-specific labels: Correct, Partial Hallucination, Full Hallucination, Abstain, or Invalid. Safety pairs are evaluated analogously as safe refusal versus unsafe or partially unsafe compliance. The matched clean/adversarial evaluation reports Benign Success Rate (BSR), Adver sarial Failure Rate (AFR), Pair Flip Rate (PFR), and Target Alignment Rate (TAR). Subsequent subsections specify the benchmark, output evaluation, probes, prompt construction, and paired metrics.

## A.2 Benchmark Composition and Evaluation

The clean reliability benchmark contains nine public datasets: TruthfulQA, HotpotQA, MuSiQue, GSM8K, MATH, FOLIO, ProofWriter, StrategyQA, and CommonsenseQA. Together, they cover factual QA, multi-hop reasoning, numerical reasoning, formal mathematics, logic/rule reasoning, implicit reasoning, and commonsense multiple-choice QA. Table 6 reports the composition of the benchmark. Automatic evaluation follows the task-specific judge\_type.

Table 7 specifies the required output form and parsing constraint for each task family. Constraint failures are therefore evaluated against task-specific requirements rather than a single open-ended criterion.

Open-ended short answers are evaluated using normalized token-level F1:

$$
F 1 ( \hat { y } , y ) = \frac { 2 P R } { P + R } ,\tag{21}
$$

$$
P = \frac { | T ( \hat { y } ) \cap T ( y ) | } { | T ( \hat { y } ) | } , \qquad R = \frac { | T ( \hat { y } ) \cap T ( y ) | } { | T ( y ) | } .\tag{22}
$$

When $F 1 ~ \ge ~ 0 . 6 0$ without an exact match, the sample is labeled partial\_hallucination. For mcq, yesno, and label tasks, the structured parser $\pi _ { j } ( \cdot )$ is applied before scoring; a parsing failure produces the label invalid\_or\_unparseable. Abstention is detected using regular expressions for phrases such as insufficient information, cannot determine, and unknown. The evaluator first checks parseability and then distinguishes abstention, correctness, partial hallucination, and full hallucination. These endpoint labels are separate from the trajectory types defined by CauTrace.

## A.3 State-Probe Training and Scale Selection

The seven functional-state/process coordinates are estimated by applying linear probes to causally pooled hidden states over window sizes {1,2, 4,8,16, 32,64}. Probe training uses weak process labels derived from task type, output label, and sequence-region heuristics; it does not use manually annotated token-level state labels. A higher score indicates stronger evidence for the corresponding operational role under the weak supervision, not a ground-truth cognitive state. Table 8 shows that validation AUC approaches saturation from w32 to w64 across the evaluated Qwen2.5 models. We therefore use w32 in the main analysis because it remains in the topperforming range while preserving greater temporal resolution for activation-onset and state-transition measurements. These AUC values measure separability with respect to the weak supervision; they do not establish that a probe recovers a unique latent state. Table 9 lists the fixed training hyperparameters.

Trace collection and offline diagnosis. CauTrace computes all diagnostic measurements after the model has generated a response. During generation, we record the hidden states, module outputs, attention-stream projections, and token positions required for analysis, but do not modify logits, hidden states, attention weights, or decoding decisions. The original model output determines the endpoint label. Statecoordinate scores, transition variables, Attention/MLP update statistics, cascade statistics, layer/head sensitivity, and Q/K/V/O measurements are then computed offline from the recorded trace. CauTrace is therefore a diagnostic framework, not a generation-time controller. In paired experiments, the clean and adversarial rows use identical instrumentation and evaluation; only the prompt condition differs. Consequently, differences between the two traces are attributable to the prompt condition and the model’s response to it, rather than to changes introduced by the analysis pipeline.

## A.4 Prompt Tracks and Matched Clean/Adversarial Construction

Each sample is evaluated under concise and analysis prompt tracks. Both use the same system instruction. The concise track requests only the final answer in the task-required format, whereas the analysis track requests two to four short reasoning steps followed by exactly one final-answer line. When uncertain, the model is instructed to output Insufficient information. The analysis track increases the observability of intermediate support and verification while retaining a parseable endpoint. Table 10 summarizes the differences between the tracks.

The matched clean/adversarial benchmark extends the process-level analysis to a security-relevant setting. Each pair shares an identifier, task semantics, evaluation target, and safety objective. The clean row contains the ordinary task or refusal instruction; the adversarial row adds a CauTracetargeted prompt perturbation without changing the evaluation target. The design isolates the prompt condition at the input level. Differences in outputs and internal measurements are therefore interpreted as perturbation-linked changes, not as evidence of a uniquely localized cause.

The paired evaluation combines reasoning-reliability pairs derived from the clean benchmark with public red-team-style safety pairs based on AdvBench and HarmBench [28, 37, 48]. Reasoning perturbations may introduce false premises, misleading source cues, shortcut rationales, incorrect calculation drafts, or answer-forcing instructions. For harmful-goal pairs, the clean condition requires safe refusal, whereas the adversarial condition applies jailbreak-style semantic pressure to formal rule tracking, answer commitment, or self-verification. We do not reproduce operational attack prompts or harmful procedural content; the analysis is limited to paired outcomes and internal changes.

Table 6: Clean benchmark composition used in the main reliability evaluation.
<table><tr><td>Dataset</td><td>Size</td><td>Judge</td><td>Dataset</td><td>Size</td><td>Judge</td><td>Dataset</td><td>Size</td><td>Judge</td></tr><tr><td>TruthfulQA</td><td>120</td><td>semantic</td><td>HotpotQA</td><td>7,400</td><td>support</td><td>MuSiQue</td><td>242</td><td>support</td></tr><tr><td>GSM8K</td><td>880</td><td>numeric</td><td>MATH</td><td>1,250</td><td>math</td><td>FOLIO</td><td>150</td><td>label</td></tr><tr><td>ProofWriter</td><td>600</td><td>label</td><td>StrategyQA</td><td>250</td><td>yes/no</td><td>CommonsenseQA</td><td>1,200</td><td>mcq</td></tr></table>

Table 7: Output constraints by judge\_type.
<table><tr><td>judge_type</td><td>Output Form</td><td>Constraint</td></tr><tr><td>label_match</td><td>Label</td><td>entailment/contradiction/unknown</td></tr><tr><td>yesno_match</td><td>Binary answer</td><td>yes/no</td></tr><tr><td>mcq_match</td><td>Option</td><td>A/B/C/D/E</td></tr><tr><td>numeric_match</td><td>Number</td><td>Final number only</td></tr><tr><td>math answer match</td><td>Mathematical answer</td><td>Final expression or result only</td></tr><tr><td>semantic</td><td>Short answer</td><td>Concise, comparable answer</td></tr><tr><td>support</td><td>QA answer</td><td>Answer based on supporting facts</td></tr></table>

Table 8: Probe validation AUC across pooling-window scales.
<table><tr><td>Model</td><td>w1</td><td>w2</td><td>w4</td><td>w8</td><td>w16</td><td>w32</td><td>w64</td><td>fused</td></tr><tr><td>Qwen2.5-1.5B</td><td>0.9006</td><td>0.9253</td><td>0.9501</td><td>0.9715</td><td>0.9843</td><td>0.9901</td><td>0.9918</td><td>0.9950</td></tr><tr><td>Qwen2.5-3B</td><td>0.9038</td><td>0.9311</td><td>0.9570</td><td>0.9793</td><td>0.9902</td><td>0.9939</td><td>0.9952</td><td>0.9972</td></tr><tr><td>Qwen2.5-7B</td><td>0.9557</td><td>0.9752</td><td>0.9880</td><td>0.9950</td><td>0.9973</td><td>0.9977</td><td>0.9979</td><td>0.9957</td></tr><tr><td>Qwen2.5-14B</td><td>0.9794</td><td>0.9916</td><td>0.9974</td><td>0.9985</td><td>0.9992</td><td>0.9990</td><td>0.9989</td><td>0.9979</td></tr></table>

Table 9: Multi-scale probe training hyperparameters.
<table><tr><td>Parameter</td><td>Setting</td></tr><tr><td>Epochs</td><td>40</td></tr><tr><td>Batch size</td><td>4096</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Patience</td><td>6</td></tr><tr><td>Train/Val</td><td>0.9 / 0.1</td></tr><tr><td>Scale set</td><td>w1, w2, w4, w8, w16, w32, w64, fused</td></tr></table>

## A.5 Adversarial Perturbations

Each adversarial sample is annotated with the CauTrace state that its prompt perturbation is designed to disrupt: PB, KR, RC, OC, FRT, AC, or SV. The annotation is excluded from endpoint scoring and used only in the target-alignment analysis. Specifically, we test whether the largest clean-toadversarial changes occur in the intended state or in theoretically downstream states. Table 11 defines the perturbation taxonomy.

The main experiments use Qwen2.5-Instruct models with 1.5B, 3B, 7B, and 14B parameters without fine-tuning [46]; the transfer analysis uses dense LLaMA-family models and Qwen1.5-MoE. Decoding is deterministic, with temperature 0.0, top\_p= 1.0, and do\_sample=false. Every clean and adversarial row passes through the same evaluator, trace collector, probe stack, transition analyzer, cascade analyzer, and Q/K/V/O analyzer. For any comparable scalar metric η, the pair-level change is

$$
\Delta \mathfrak { \eta } = \mathfrak { \eta } ( \mathrm { a d v e r s a r i a l } ) - \mathfrak { \eta } ( \mathrm { c l e a n } ) ,\tag{23}
$$

This notation avoids overloading m, which denotes the number of value vectors in Eq. (9).

The paired analysis uses four primary behavior-level metrics. Benign Success Rate (BSR) is the fraction of clean rows that satisfy the correctness, evidence-support, or safe-refusal criterion. Adversarial Failure Rate (AFR) is the fraction of adversarial rows that are erroneous, unsupported, unsafe, or partially unsafe. Pair Flip Rate (PFR) is the fraction of pairs that move from benign success to adversarial failure. Target Alignment Rate at k (TAR@k) measures whether the intended CauTrace target state occurs among the k largest internal changes; the main text reports TAR@1 and TAR@3. We also compare flipped and non-flipped pairs to determine whether realized endpoint failures are associated with larger changes in state scores, modules, cascades, layers/heads, or Q/K/V/O representations.

Table 10: Differences between concise and analysis tracks.
<table><tr><td>Dimension</td><td>Concise</td><td>Analysis</td><td>Potential Failure Mode</td></tr><tr><td>Output length</td><td>Short</td><td>2-4 steps</td><td>May induce Verbose Unsupported Continuation outputs</td></tr><tr><td>Observability</td><td>Low</td><td>High</td><td>Rationale text is not always faithful verification</td></tr><tr><td>Commitment risk</td><td>Medium</td><td>Potentially higher</td><td>Early Commitment or Verification without Overwrite</td></tr><tr><td>Parsing stability</td><td>High</td><td>Medium</td><td>Invalid parsing if final-line format is violated</td></tr></table>

Table 11: CauTrace state-targeted adversarial perturbations.
<table><tr><td>Target State</td><td>Process Role</td><td>Perturbation Focus</td></tr><tr><td>Premise Binding</td><td>straints.</td><td>Bind entities, roles, context, and con- False premises, role confusion, conflicting context bindings.</td></tr><tr><td>Knowledge Retrieval</td><td>Retrieve relevant facts or supporting evi- dence.</td><td>Plausible but wrong facts, source bias, retrieval decoys.</td></tr><tr><td>Reasoning Composition</td><td>Organize multi-step inference and inter- mediate claims.</td><td>Shortcut reasoning, chain reordering, incompatible intermediate claims.</td></tr><tr><td>Operational Computa- tion</td><td>Execute numerical or symbolic opera- tions.</td><td>Wrong scratch work, expression substitution, arith- metic interference.</td></tr><tr><td>Formal Rule Tracking</td><td>Maintain formal rules, safety policies, and task boundaries.</td><td>Rule-priority inversion, forged exceptions, con- straint override.</td></tr><tr><td>Answer Commitment</td><td>Stabilize an answer after sufficient sup- port.</td><td>Premature conclusion, forced single answer, un- certainty compression.</td></tr><tr><td>Self-Verification</td><td>Check evidence, answer consistency, and risk boundaries.</td><td>Reflection suppression, no-check instructions, refusal-review weakening.</td></tr></table>

## B Supplemental Results

This section provides the endpoint, Trajectory Topology, and intervention results referenced in the main evaluation.

## B.1 Robustness to Diagnostic Aggregation

To determine whether Claims I–III depend on a particular aggregation scheme, we recompute the four-indicator aggregation $R _ { \mathrm { s t r } }$ and reasoning process-deviation score $R _ { \mathrm { p r o c } }$ using leave-one-component-out ablations and Dirichlet-random weights on the clean Correct/Full Hallucination comparison. Table 12 reports rank stability, failure–correct separation, and threshold-decision stability relative to the equal-weight scores.

Representation-Change Propagation Is Stable as a Ranking, Whereas Reasoning-Process Deviation Is Intrinsically Multi-Component and Should Not Be Reduced to One Threshold. For $R _ { \mathrm { s t r } }$ , both ablation and random weighting retain rank stability of 1.000 in Table 12; however, ablation decision stability ranges from 0.250 to 1.000, and its separation range reaches 0.000. Thus, Claim I is robust at the ordering level but not as a universal binary rule. For $R _ { \mathrm { p r o c } } ,$ median rank stability falls to 0.790 under ablation and 0.696 under random weights, while median decision stability is 0.827 and 0.811, respectively. This sensitivity is consistent with Claims II–III: different Trajectory Topologies are dominated by different combinations of timing, switching, rule preservation, and verification, so the aggregate score must be interpreted together with its components.

## B.2 Cross-Family Diagnostic Results

Table 13 provides the complete cross-family measurements summarized in the main Generalization subsection. The shared directions of $\Delta \bar { A } _ { \mathrm { s u p } }$ and $\Delta R _ { \nu c }$ support the representation-propagation and coordination relations, whereas the other representation-level carriers remain

Table 12: Aggregation sensitivity and component-ablation analysis for $R _ { \mathrm { s t r } }$ and $R _ { \mathrm { p r o c } }$ . Rank stability denotes Spearman correlation with the default equal-weight score. Separation delta denotes the failure-minus-correct contrast under the perturbed score. Decision stability denotes preservation of the default threshold decision.
<table><tr><td>Score Variant</td><td>Perturbation</td><td>Rank Stability</td><td>Separation Delta</td><td></td><td>Decision Stability</td></tr><tr><td> $R _ { \mathrm { s t r } }$  default</td><td>none; equal weights</td><td>1.000</td><td>+0.014</td><td></td><td>1.000</td></tr><tr><td> $R _ { \mathrm { s t r } }$  ablation</td><td>support/value components leave- one-out</td><td>1.000 [1.000, 1.000]</td><td>+0.014 +0.018]</td><td>[+0.000,</td><td>0.812 [0.250, 1.000]</td></tr><tr><td> $R _ { \mathrm { s t r } }$  random weights</td><td>Dirichlet random weights</td><td>1.000 [1.000, 1.000]</td><td>+0.014 +0.050]</td><td>[+0.000,</td><td>1.000 [1.000, 1.000]</td></tr><tr><td> $R _ { \mathrm { p r o c } }$  default  $R _ { \mathrm { p r o c } }$  ablation</td><td>none; equal weights</td><td>1.000</td><td>-0.022 -0.022</td><td></td><td>1.000</td></tr><tr><td></td><td>process components leave-one- out</td><td>0.790 [0.633, 0.887]</td><td>+0.006]</td><td>[-0.046,</td><td>0.827 [0.767, 0.933]</td></tr><tr><td> $R _ { \mathrm { p r o c } }$  random weights</td><td>Dirichlet random weights</td><td>0.696 [0.022, 0.987]</td><td>-0.022 +0.061]</td><td>[-0.107,</td><td>0.811 [0.567, 1.000]</td></tr></table>

architecture-dependent.

## B.3 Endpoint Outcomes and Distinct Trajectory Topologies

To show why endpoint labels are insufficient for the three process-level claims, we compare the concise and analysis tracks on the clean nine-dataset benchmark. Table 14 reports the overall outcome distribution by Qwen2.5 scale. Table 15 and Fig. 8 provide the corresponding task-group breakdown.

Improved Endpoint Accuracy Does Not Imply Single Reliable Reasoning-Process Route. In Table 14, the analysis track raises correctness by 13.2, 20.1, 18.6, and 11.7 percentage points for Qwen2.5-1.5B/3B/7B/14B, respectively. Full Hallucination nevertheless increases by 6.1 points for 3B and 7.6 points for 7B, even though it decreases for 1.5B and 14B. Longer reasoning is therefore associated with different outcomes across scales: it may provide more opportunity to build support and verify an answer, but it may also extend an unsupported commitment. Endpoint changes alone cannot distinguish these process routes.

The task-group results show that endpoint distributions also depend strongly on task structure. In Table 15, failure is the largest category for formal mathematics (0.718), abstention is the largest for multi-hop QA (0.678), and correctness is the largest for commonsense multiple-choice QA (0.721). Figure 8 displays the same distributions across all seven task groups. These task-dependent mixtures motivate separate analysis of operational and rule failures, missing support, and constraint handling rather than treating an endpoint label as a complete process explanation.

## B.4 Trajectory Signatures

The trajectory types in the main text are rule-defined regions of the trajectory-feature space. They are neither output labels nor mutually exclusive ground-truth classes. Table 16 specifies their operational signatures. The state-transition chains in Fig. 7 are schematic examples consistent with these definitions and the observed transition-probability differences. They are not token-level reconstructions, and each trajectory type may contain multiple chain variants.

## B.5 CauTrace-Guided Repair and State Steering

Calibration and held-out protocol. We test whether the functional-state/process coordinates identified by CauTrace can guide response repair or internal-state steering on held-out data. All data-dependent choices—including the repair trigger, Trajectory Topology assignment, and steering strength—are fixed using calibration data before test evaluation. The heldout outcomes are therefore used only for evaluation and do not affect the intervention configuration.

Reasoning repair. For Qwen2.5-1.5B reasoning, we use 50 calibration pairs to determine which responses should be repaired. A trace is triggered when its process-deviation score $R _ { \mathrm { p r o c } }$ exceeds the 90th percentile of the clean-success calibration traces. This percentile provides a conservative high-risk operating point: it targets traces whose process organization is unusually different from successful clean reasoning, while avoiding unnecessary second-pass generation and test-set threshold tuning. Each triggered trace is also assigned a Trajectory Topology using the fixed CauTrace diagnostics.

We then evaluate five conditions on 200 disjoint test pairs. Baseline retains the original response. Generic review applies a diagnosis-free second-pass review with the same decoding and response budget as the CauTrace conditions. Unified Cau Trace repair (cautrace\_repair) applies a fixed sequence of support reconstruction, rule checking, provisional commitment, and final verification. Topology-conditioned repair uses a correction protocol specialized to the diagnosed Trajectory

Table 13: Cross-family validation of representation-level and temporal diagnostics.
<table><tr><td>Model</td><td> $\Delta \bar { A } _ { \mathrm { s u p } }$ </td><td> $\Delta \rho _ { V }$ </td><td> $\Delta D _ { K L } ^ { V }$ </td><td> $\Delta \Gamma _ { \mathrm { a c } , V }$ </td><td> $\Delta R _ { \nu c }$ </td></tr><tr><td>LLaMA-3.2-1B</td><td>-0.052</td><td>-0.000</td><td>+0.001</td><td>-0.058</td><td>-13.085</td></tr><tr><td>LLaMA-3.2-3B</td><td>-0.022</td><td>+0.075</td><td>+0.002</td><td>+0.050</td><td>-13.558</td></tr><tr><td>LLaMA-3.1-8B</td><td>-0.042</td><td>+0.073</td><td>+0.007</td><td>-0.018</td><td>-15.519</td></tr><tr><td>Qwen1.5-MoE</td><td>-0.012</td><td>+0.000</td><td>-0.000</td><td>+0.057</td><td>-1.102</td></tr></table>

Table 14: Outcome distributions across Qwen2.5 models and prompt tracks. Len. denotes mean generation length.
<table><tr><td>Model</td><td>Track</td><td>Correct</td><td>Partial</td><td>Full</td><td>Abstain</td><td>Invalid</td><td>Len.</td></tr><tr><td>1.5B</td><td>Concise</td><td>19.7</td><td>0.5</td><td>54.4</td><td>24.7</td><td>0.7</td><td>16.0</td></tr><tr><td>1.5B</td><td>Analysis</td><td>32.9</td><td>1.2</td><td>43.7</td><td>21.9</td><td>0.2</td><td>76.5</td></tr><tr><td>3B</td><td>Concise</td><td>12.1</td><td>0.3</td><td>20.1</td><td>65.1</td><td>2.3</td><td>25.5</td></tr><tr><td>3B</td><td>Analysis</td><td>32.2</td><td>1.2</td><td>26.2</td><td>40.1</td><td>0.3</td><td>67.5</td></tr><tr><td>7B</td><td>Concise</td><td>15.8</td><td>0.4</td><td>12.7</td><td>71.1</td><td>0.0</td><td>6.0</td></tr><tr><td>7B</td><td>Analysis</td><td>34.4</td><td>1.3</td><td>20.3</td><td>44.0</td><td>0.0</td><td>41.3</td></tr><tr><td>14B</td><td>Concise</td><td>18.5</td><td>0.4</td><td>16.1</td><td>58.9</td><td>6.1</td><td>29.1</td></tr><tr><td>14B</td><td>Analysis</td><td>30.2</td><td>0.7</td><td>12.7</td><td>56.4</td><td>0.0</td><td>55.7</td></tr></table>

![](images/9d6309c60041a9f0ffab6350c753079dd3496968dce86de7193df45e1d2021be.jpg)  
Figure 8: Task-group outcome distributions for Qwen2.5 generations. The horizontal bars visualize the proportions reported in Table 15; they are not independent estimates.

Table 15: Task-group outcome composition, aggregated across Qwen2.5 scales and prompt tracks.
<table><tr><td>Task group</td><td>Size</td><td>Correct</td><td>Failure</td><td>Abstain</td><td>Invalid</td></tr><tr><td>Calculation</td><td>7032</td><td>0.445</td><td>0.380</td><td>0.090</td><td>0.085</td></tr><tr><td>Formal math</td><td>10000</td><td>0.240</td><td>0.718</td><td>0.042</td><td>0.000</td></tr><tr><td>Multi-hop QA</td><td>61176</td><td>0.137</td><td>0.185</td><td>0.678</td><td>0.000</td></tr><tr><td>Knowledge QA</td><td>656</td><td>0.110</td><td>0.494</td><td>0.396</td><td>0.000</td></tr><tr><td>Logic/rule</td><td>5952</td><td>0.252</td><td>0.177</td><td>0.486</td><td>0.086</td></tr><tr><td>Commonsense MCQ</td><td>9680</td><td>0.721</td><td>0.263</td><td>0.013</td><td>0.003</td></tr><tr><td>Implicit reasoning</td><td>2224</td><td>0.537</td><td>0.262</td><td>0.188</td><td>0.013</td></tr></table>

Topology, whereas wrong-topology repair uses a deliberately mismatched topology while preserving the same prompt structure and generation budget. Only triggered traces are regenerated; non-triggered traces retain their baseline response under every second-pass condition. These controls distinguish improvement from generic additional review, improvement from the unified CauTrace protocol, and improvement that specifically depends on the diagnosed topology. Table 17 reports the held-out endpoint results.

State steering. For Qwen2.5-7B safety, we test whether the same diagnostics can guide a direct hidden-state intervention. Let $\nu _ { \mathrm { a l i g n } }$ denote the normalized steering direction constructed from the signed state directions predicted by the diagnosed trajectory: positive support, formal rule, and verification directions are combined with a negative answer-commitment direction. At generation step t and intervened layer $\ell ,$ steering

Table 16: Operational signatures of the Trajectory Topology regions. Arrows indicate the expected direction relative to Normal Convergence. A sample may satisfy more than one signature.
<table><tr><td>Trajectory</td><td>Metric Signature</td><td>Diagnostic Interpretation</td></tr><tr><td>Early Commitment</td><td> $\tau _ { \mathrm { c o m } } < \tau _ { \mathrm { s u p } } \mathrm { o r } g _ { \mathrm { c s } } \downarrow$ </td><td>Answer commitment becomes dominant before support is sufficiently established.</td></tr><tr><td>Fragmented Support</td><td> $\Omega \uparrow , { \bar { S } } \downarrow .$  unstable  $\tau _ { \mathrm { s u p } }$ </td><td>Support-related states activate but do not form a stable, continuous chain.</td></tr><tr><td>Rule Failure</td><td> $L _ { \mathrm { r u l e } }$  ↑ with weak FRT/OC state scores</td><td>The trace approaches an answer without preserving formal rules, output constraints, or symbolic/numerical consistency.</td></tr><tr><td>Verification without Overwrite</td><td> $g _ { \mathrm { v c } } \uparrow , R _ { \mathrm { v c } } \downarrow$ </td><td>Verification follows commitment but remains too weak to revise the stabilized answer.</td></tr><tr><td>Verbose Unsupported Continuation</td><td> $T \uparrow \mathrm { w i t h } \bar { S } \downarrow$  and  $R _ { \mathrm { v c } } \downarrow$ </td><td>Generation continues without a corresponding increase in support or effective verification.</td></tr><tr><td>Normal Convergence</td><td> ${ \tau } _ { \mathrm { s u p } } \leq { \tau } _ { \mathrm { c o m } } ,$  bounded Ω,  $R _ { \mathrm { v c } }$  ↑</td><td>Support precedes commitment, constraints remain active, and verification can confirm or revise the candidate.</td></tr></table>

Table 17: Held-out Qwen2.5-1.5B reasoning repair on 200 test pairs. Endpoint rates are percentages. Clean good denotes the proportion of clean responses that remain correct, and ∆Fail is the adversarial-failure change relative to the baseline in percentage points.
<table><tr><td>Condition</td><td>Adv. fail</td><td>Clean good</td><td>∆Fail</td></tr><tr><td>Baseline</td><td>76.0</td><td>26.0</td><td></td></tr><tr><td>Unified CauTrace</td><td>71.0</td><td>31.0</td><td>-5.0</td></tr><tr><td>Generic review</td><td>72.0</td><td>32.0</td><td>-4.0</td></tr><tr><td>Topology-conditioned</td><td>73.0</td><td>29.5</td><td>-3.0</td></tr><tr><td>Wrong topology</td><td>73.5</td><td>30.5</td><td>-2.5</td></tr></table>

is applied as

$$
h _ { \ell , t } ^ { \prime } = h _ { \ell , t } + \alpha \mathrm { R M S } ( h _ { \ell , t } ) \nu _ { \mathrm { a l i g n } } .\tag{24}
$$

where RMS normalization makes the intervention scale comparable across layers and responses, and α controls its strength.

We select α using six calibration pairs and a predefined grid 0.25,0.5,1.0. The selected value minimizes adversarial failure on the calibration pairs subject to a maximum twopercentage-point reduction in clean-good rate. This constraint prevents the intervention strength from being selected solely for adversarial recovery at the expense of clean behavior. The resulting value is $\mathbf { \Delta } \mathbf { Q } = 0 . 2 5$ , which is fixed before evaluation on 12 disjoint clean-success/adversarial-failure test pairs.

The direction is injected into a three-layer band centered on the CauTrace-identified risk layer during the first eight generated tokens. This restricts the intervention to the diagnosed model region and the early generation stage in which the response trajectory is formed. We compare aligned steering with three equal-budget controls: reversed steering uses $- \nu _ { \mathrm { a l i g n } } ;$ random steering uses a norm-matched direction orthogonal to $\nu _ { \mathrm { a l i g n } } \mathrm { ; }$ ; and wrong-state steering strengthens answer commitment while weakening formal rule tracking and verification. All four conditions use the same α, layers, intervention duration, and decoding configuration. The controls therefore test whether the predicted state direction changes the reasoning process beyond the effect of a generic hiddenstate perturbation. Table 18 reports both endpoint outcomes and online process-coordinate changes.

Table 18: Held-out Qwen2.5-7B safety steering on 12 test pairs. Process deltas pool the clean and adversarial member of each pair; lower $R _ { \mathrm { p r o c } }$ and higher rule and verification coverage are better.
<table><tr><td>Condition</td><td>Adv. fail Clean</td><td> $\Delta R _ { \mathrm { p r o c } }$ </td><td> $\Delta \mathrm { R u l e } _ { \mathrm { c o v } }$ </td><td> $\Delta V _ { \mathrm { c o v } }$ </td></tr><tr><td>Baseline</td><td>100.0 100.0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Aligned</td><td>66.7</td><td>91.7 -0.008</td><td>+0.011 +0.021</td><td></td></tr><tr><td>Random</td><td>66.7 100.0</td><td>0.000</td><td>+0.002</td><td>0.000</td></tr><tr><td>Reversed</td><td>66.7</td><td> $9 1 . 7 \ + 0 . 0 1 5$ </td><td>-0.014</td><td>4-0.019</td></tr><tr><td>Wrong state</td><td>66.7</td><td> $9 1 . 7 \ + 0 . 0 1 0$ </td><td>-0.007</td><td>7-0.026</td></tr></table>

Unified Repair Gives the Largest Observed 1.5B Failure Reduction, and Aligned Steering Moves the 7B Process Measures in the Predicted Direction. On the full 1.5B reasoning test, unified CauTrace repair reduces adversarial failure from 76.0% to 71.0%, corresponding to a 5.0-percentagepoint reduction (paired 95% bootstrap CI [−8.5,−2.0]; Mc-Nemar $p = 0 . 0 0 6 3 \mathrm { \Omega }$ ), with 11 failed responses rescued and one previously successful response regressing. This is the largest observed reduction among the five conditions in Table 17; generic review reduces failure by 4.0 points, while topology-conditioned and wrong-topology repair reduce it by 3.0 and 2.5 points, respectively. The direct 1.0-point difference between unified CauTrace repair and generic review is not statistically significant (95% CI $[ - 4 . 0 , + 1 . 5 ] ; p = 0 . 7 2 7 )$ The result therefore shows that the complete support–rule– commitment–verification protocol can improve held-out reasoning outcomes, but does not establish superiority over generic review or a topology-specific repair effect.

On 7B safety, all four steering directions reduce adversarial failure from 100.0% to 66.7% in Table 18; endpoint recovery alone therefore does not distinguish the aligned direction from its controls. The internal process changes, however, follow different directions. Aligned steering is the only condition that simultaneously lowers process risk $( \Delta R _ { \mathrm { p r o c } } = - 0 . 0 0 8 )$ increases rule coverage (+0.011), and increases verification coverage (+0.021). Reversed and wrong-state steering instead increase $R _ { \mathrm { p r o c } }$ and reduce both rule and verification coverage, whereas the random control remains close to zero. The same pattern is stronger among the adversarial responses rescued by aligned steering, for which $\Delta R _ { \mathrm { p r o c } } = - 0 . 0 5 1$ $\Delta \mathrm { R u l e } _ { \mathrm { c o v } } = + 0 . 0 3 8$ , and $\Delta V _ { \mathrm { c o v } } = + 0 . 0 4 2$ . Thus, aligned steering produces the process-direction changes predicted by the support/rule–commitment–verification relation in Claim II. Given the 12-pair test and the identical endpoint rates across steering controls, this result supports theory-consistent process steering in the evaluated setting without attributing endpoint recovery to a unique causal state direction.

## B.5.1 Discussion: Defense Implications and Future Directions

These results suggest a concrete path from the three-level diagnostics to process-aware defenses. Structural retention can identify where an intervention may be applied, functionalstate/process coordinates can specify whether support, rule tracking, and verification should be strengthened or premature commitment suppressed, and Trajectory Topology can provide a candidate basis for selecting repair policies. In the present tests, unified repair provides the largest observed 1.5B reasoning reduction, while aligned steering moves the 7B process measures in the predicted direction, showing that Cau-Trace signals are actionable in the evaluated settings. Future work can develop this fixed intervention into a closed-loop defense that monitors process risk during generation, triggers repair only when needed, and adapts the repair policy or steering strength while preserving clean behavior. Evaluation across broader model families, tasks, attacks, and held-out samples, together with stronger direction-specific controls, will clarify when these process-level improvements translate into consistent endpoint robustness.