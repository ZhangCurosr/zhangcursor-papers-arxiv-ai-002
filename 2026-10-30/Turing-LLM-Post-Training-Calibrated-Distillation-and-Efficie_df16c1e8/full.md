# Turing LLM Post-Training: Calibrated Distillation and Efficient Dynamic-MoE Execution

Foundation Model Team, XPeng Inc.

## Abstract

We study post-training for dynamically routed mixture-of-experts (MoE) models, focusing on integrating independently optimized specialists while keeping consolidation efficient. Our approach combines calibrated multi-teacher on-policy distillation (C-MOPD), which matches each selected teacher on a local Top-K support with teacher-wise loss normalization and clipping, with an execution path that uses post-capacity bounded-width dispatch, finalroute replay, and asynchronous rollout–training overlap. On Turing-20B-A2B, the selected model achieves a four-domain normalized integration score of 0.9988 and retains at least 90.16% of the measured specialist gain in every domain. At matched $K = 3 2 ,$ the full calibration objective scores 0.9962, compared with 0.9724 without calibration and 0.9834 without clipping. At a matched 8K response cap, the model exceeds both shared SFT and Qwen3-8B on all four reasoning-enabled mathematics and code benchmarks (unweighted mean: 85.92, versus 80.33 for Qwen3-8B and 74.21 for SFT), and improves over SFT on ten of twelve broader evaluation metrics. In a PPU-based profile, the combined execution configuration reduces consolidation step time from 398.7 to 139.3 seconds, an observed 2.86× speedup over serial execution. These results demonstrate strong specialist integration alongside more efficient consolidation, while reflecting the complete training and execution framework rather than any single component.

Keywords: LLM Post-Training, Mixture of Experts, On-Policy Distillation, Reinforcement Learning

![](images/53020c21268a295a61cf0cf77139a480fbd9e6932df1f7cd1daacdd7905fc9b0.jpg)  
Figure 1: The Turing LLM post-training framework. Shared supervised initialization provides the starting point for domain-specialist RL. C-MOPD consolidates the frozen specialists into one student through teacher-local Top-K matching and loss-scale calibration on student-generated prefixes. Bounded-width dispatch, route repla following R3, and asynchronous overlap support efficient dynamic-MoE execution in the consolidation workflow.

## 1 Introduction

Modern LLM post-training increasingly relies on capability-specific optimization pipelines. Different capabilities may call for different data, feedback signals, training environments, and optimization recipes, allowing each pipeline to develop a strong specialist. Deploying these specialists separately can increase model-storage and serving overhead, motivating their consolidation into a single model with broader task coverage. We refer to the challenge of combining their complementary capabilities while retaining specialist gains as capability integration.

Existing integration routes make different compromises. Mixed-domain reinforcement learning (RL) couples heterogeneous domains to a shared sampling schedule and optimization trajectory, while sequential RL can overwrite capabilities acquired in earlier stages. Off-policy distillation from fixed teacher trajectories can suffer from exposure mismatch, while parameter-space merging combines specialist checkpoints through weight-space operations. MOPD [1] offers a specializethen-consolidate alternative: domain RL teachers are developed independently and then distilled into a student on the student’s own rollouts.

Two practical challenges arise in our setting. First, token-level distillation losses can vary substantially in scale across teachers, complicating the use of shared optimization settings and fixed teacher weights. Second, the rollout–actor loop for a dynamically routed mixture-of-experts (MoE) student must handle variable-width expert dispatch and cross-engine synchronization while preserving route consistency between generation and training.

We study both challenges in the post-training of Turing-20B-A2B, a 20B-parameter dynamic MoE with approximately 2B active parameters per token [2]. A shared SFT checkpoint initializes RL specialists for mathematics, code, instruction following, and agentic tool use. We then consolidate the frozen specialists into one student with calibrated multi-teacher on-policy distillation (C-MOPD). Following MOPD, task routing selects one teacher for each sample, and supervision is evaluated on student-generated prefixes. C-MOPD applies teacher-wise loss-scale normalization and token-level clipping to teacher-local Top-K reverse-KL matching [3], reducing sensitivity to teacher-specific loss scales while retaining explicit teacher weights.

The consolidated student achieves a normalized four-domain integration score of 0.9988. With K = 16 and $c _ { \operatorname* { m a x } } = 5 ,$ , the selected model retains at least 90.16% of the improvement achieved by the corresponding specialist over the shared SFT baseline in each domain. The final model produced by the specialist-training and consolidation pipeline also improves over the shared SFT baseline on all four mathematics and code benchmarks evaluated in reasoning mode and on ten of twelve metrics in the broader evaluation suite.

To improve consolidation efficiency, we combine asynchronous rollout–actor overlap with one-step policy lag, bounded-width dispatch after capacity filtering, and final-route replay following R3 [4]. With the post-capacity dispatch width capped at 32, the combined configuration reduces the profiled consolidation step time from 398.7 to 139.3 seconds, yielding a 2.86× speedup over serial execution.

Our contributions are:

• Specialist capability integration. We instantiate an end-to-end specialize-then-consolidate pipeline for a dynamic MoE, connecting a shared SFT initialization, four verifier-driven RL specialists, and a single C-MOPD student.

• Calibrated multi-teacher distillation. We introduce C-MOPD, which combines teacher-local Top-K reverse-KL matching with teacher-wise loss-scale normalization and token-level clipping under fixed teacher weights. In matched ablations at K = 32, the full objective achieves a normalized integration score of 0.9962, compared with 0.9724 when normalization and clipping are both removed and 0.9834 when clipping alone is removed.

• Efficient dynamic-MoE execution. We adapt the consolidation workflow through bounded-width dispatch after capacity filtering, final-route replay, and asynchronous rollout–actor overlap to improve execution efficiency while maintaining cross-engine route consistency.

section 2 describes C-MOPD and the dynamic-MoE execution path. section 3 presents capabilityintegration results, matched ablations, and execution profiles. The appendices provide training recipes, complete results, diagnostic analyses, and reproducibility details.

Table 1: Prompt-form mixtures and reward components for specialist training. Think and Non-think denote reasoning-enabled and direct-response prompts; percentages give their sampling proportions within each domain.
<table><tr><td>Task</td><td>Think</td><td>Non-think</td><td>Verification and reward</td></tr><tr><td>Code</td><td>80%</td><td>20%</td><td>Extraction, signature validity, compilation, and unit-test pass rate; auxiliary terms are gated by executable validity.</td></tr><tr><td>Math</td><td>80%</td><td>20%</td><td>Final-answer correctness and normal completion, with penalties for truncation and repeated reasoning.</td></tr><tr><td>IF</td><td>20%</td><td>80%</td><td>Constraint satisfaction or option correctness, with a preference for concise valid outputs.</td></tr><tr><td></td><td>Agent 100%</td><td>0%</td><td>Tool-call format, argument correctness, task completion, and process feedback from a 20B process reward model.</td></tr></table>

## 2 Method

## 2.1 Framework Overview

Our framework comprises three stages (Figure 1): shared supervised fine-tuning, domain-specific reinforcement learning, and multi-teacher on-policy distillation. Starting from a common SFT checkpoint, we train four specialists for mathematics, code, instruction following, and agentic tool use. C-MOPD then consolidates the frozen specialists into a single student using student-generated trajectories. We use specialist to denote a complete domain teacher and expert to denote an internal MoE module. Task-level teacher selection and token-level expert routing operate at different levels of the framework.

The execution workflow builds on the base model’s capacity-constrained Quantile Routing [2]. Postcapacity dispatch-width bounding limits the number of expert assignments represented and executed for each token. Final-route replay following R3 [4] preserves these discrete assignments during actor computation, while asynchronous overlap runs rollout generation concurrently with actor updates.

## 2.2 Shared Initialization and Specialist Construction

The shared SFT stage uses approximately 5.28M examples spanning mathematics, code, tool use, instruction following, long context, and dialogue. The same model supports two prompt-selected response modes, following Qwen’s convention [5]: reasoning-enabled prompts end the assistant prefix with an open <think> marker, so generation proceeds in the reasoning block until </think>; directresponse prompts instead insert an empty <think></think> pair before the answer. The data recipe includes teacher correction, capability-boundary filtering, and protocol-randomized multiple-choice construction; Appendix A describes these procedures and the data mixture.

Starting from this checkpoint, we train four domain specialists with GSPO, using 16 rollouts per prompt [6]. Each domain uses a task-specific prompt mixture and reward design, as summarized in Table 1. After training, we freeze the four specialists and use them as teachers for C-MOPD consolidation.

## 2.3 C-MOPD: Calibrated Multi-Teacher On-Policy Distillation

## 2.3.1 Task-Conditioned On-Policy Supervision

Following MOPD [1], a task router selects one frozen domain teacher for each prompt $x _ { j }$ ,

$$
d _ { j } = r ( x _ { j } ) .\tag{1}
$$

In the synchronous formulation, the student samples a trajectory $y _ { j } \sim \pi _ { \theta } ( \cdot \mid x _ { j } )$ . At each valid response position t, teacher $d _ { j }$ is evaluated on the student-generated prefix $( x _ { j } , y _ { j , < t } )$ . Let $p _ { \theta , j , t }$ and $q _ { d _ { j } , j , t }$ denote the student and selected-teacher next-token distributions on that prefix. When AsyncOverlap is enabled, trajectories are generated using the preceding student policy version, introducing a one-step policy lag (subsection 2.4).

## 2.3.2 Teacher-Local Reverse KL

For each valid response position, define the selected teacher’s local support

$$
S _ { j , t } = \mathrm { T o p } _ { K } \left( q _ { d _ { j } , j , t } \right) .\tag{2}
$$

We independently renormalize the student and teacher distributions within this support,

$$
\bar { p } _ { \theta , j , t } ^ { S } ( v ) = \frac { p _ { \theta , j , t } ( v ) } { \sum _ { u \in S _ { j , t } } p _ { \theta , j , t } ( u ) } , \qquad \bar { q } _ { d _ { j } , j , t } ^ { S } ( v ) = \frac { q _ { d _ { j } , j , t } ( v ) } { \sum _ { u \in S _ { j , t } } q _ { d _ { j } , j , t } ( u ) } ,\tag{3}
$$

and compute the token-level reverse KL divergence between the locally normalized distributions,

$$
\ell _ { j , t } = { \cal D } _ { \mathrm { K L } } \Big ( \bar { p } _ { \theta , j , t } ^ { S } \Big | \Big | \bar { q } _ { d _ { j } , j , t } ^ { S } \Big ) .\tag{4}
$$

This local-support objective follows prior on-policy distillation work [3]. It matches relative probabili ties among the teacher’s K most probable tokens. Because both distributions are renormalized within $S _ { j , t }$ , the loss does not directly constrain the student’s total probability mass outside this support. We compare the full objective with its calibration ablations at a fixed support size of ${ \mathrm { K } } = 3 2$ to assess the effects of loss-scale normalization and token-level clipping under this setting. We also evaluate different support sizes with the full objective at a clipping threshold of 5. The selected model uses K = 16, which achieves the highest observed macro integration score among the tested support sizes.

## 2.3.3 Teacher-Wise Loss-Scale Calibration

Token-level distillation losses may differ in scale across teachers. C-MOPD applies teacher-wise normalization and token-level clipping to reduce sensitivity to these differences.

Let $B _ { \mu }$ be a microbatch and $m _ { j , t } \in \{ 0 , 1 \}$ mark valid response tokens. For each teacher represented in the microbatch, we compute the standard deviation of its token losses,

$$
\sigma _ { d } ^ { ( \mu ) } = \mathrm { S t d } \{ \ell _ { j , t } : j \in \mathcal { B } _ { \mu } , d _ { j } = d , m _ { j , t } = 1 \} .\tag{5}
$$

The scale statistic $\sigma _ { d } ^ { ( \mu ) }$ is detached from the computation graph. With a scale floor of 0.1, the calibrated token loss is

$$
\widehat { \ell } _ { j , t } = \operatorname* { m i n } \left( c _ { \operatorname* { m a x } } , \frac { \ell _ { j , t } } { \operatorname* { m a x } \left( \sigma _ { d _ { j } } ^ { ( \mu ) } , 0 . 1 \right) } \right) ,\tag{6}
$$

where $c _ { \operatorname* { m a x } } = 5$ in the selected model. Scaling without mean centering preserves non-negativity. With the scale detached, tokens whose normalized loss exceeds $c _ { \mathrm { m a x } }$ before clipping contribute zero direct gradient through the clipped term.

With fixed non-negative teacher weights $w _ { d }$ , the microbatch objective is

$$
\mathcal { L } _ { \mu } ( \boldsymbol { \theta } ) = \frac { \sum _ { j \in \mathcal { B } _ { \mu } } \sum _ { t } m _ { j , t } w _ { d _ { j } } \widehat { \ell } _ { j , t } } { \sum _ { j \in \mathcal { B } _ { \mu } } \sum _ { t } m _ { j , t } } .\tag{7}
$$

The objective averages weighted losses over all valid response tokens (token\_mean). Domain sampling determines how often each teacher is selected, while $w _ { d }$ multiplies that teacher’s token losses. We fix $w _ { \mathrm { I F } } = 0 . 3$ and $w _ { d } = 1 . 0$ for Math, Code, and Agent throughout consolidation. The weighting configuration is discussed in Appendix B.

At each valid response position, the teacher supplies its Top-K token identifiers and the corresponding log probabilities. The retained teacher outputs and local divergence computation scale as $O ( \bar { T } K )$ for response length T. The student forward pass still computes full-vocabulary logits.

## 2.4 Efficient Execution for Dynamic MoE

The execution path applies capacity-constrained Quantile Routing, bounds the resulting dispatch width, and replays the final expert assignments during actor computation. These mechanisms are illustrated in Figure 2. Asynchronous overlap schedules rollout generation and actor updates concurrently.

## (a) Bounded-width dynamic routing for RL

![](images/55f3ec61b94ba0906c6f4f7672d76e0abc02662e7181fe3ff7b409d342fec47c.jpg)

## (b) Final-route replay and asynchronous overlap

![](images/e0c973e7757ae614ed8e3218070a71ec231209b83712f8f6104950da3d52ee2a.jpg)

![](images/46b108aa556d2585f8b0905e58bdd9363017872a53516941fccaf5a14b2accc9.jpg)  
Idealized steady-state timing; overhead omitted.

![](images/7d22f7870f720ae754d15bf215385838451a7eeff2513191c7c6795967631ff4.jpg)  
Figure 2: Dynamic-MoE execution during consolidation. (a) RW32 bounds post-capacity dispatch width at 32, preserving non-overflow expert assignments and retaining the highest-scoring 32 assignments on overflow. (b) R3-style final-route replay reuses the executed expert assignments and recomputes routing weights under the current actor parameters. AsyncOverlap overlaps rollout with actor updates using a one-step policy lag. (c) Base-model Quantile Routing selects a variable number of experts per token, followed by per-expert capacity filtering.

## 2.4.1 Capacity-Constrained Quantile Routing

TuringLLM’s Quantile Routing uses expert-specific score thresholds to select a token-adaptive number of experts under a controlled average budget [2]. For a routing window W, per-expert capacity filtering retains at most $C ( W ) = \mathsf { \bar { \rho } } \alpha \mathsf { | } W \mathsf { | } \bar { k } / E ^ { - }$ ⌉ highest-scoring assignments among the selected candidates, where α is the capacity factor, <sup>¯</sup>k the target average expert count, and E the number of experts. We retain this base-model routing rule throughout post-training (Figure 2c).

## 2.4.2 Post-Capacity Dispatch-Width Bounding

The default rollout representation reserves E = 256 expert slots per token. In the measured routing profile, the 99th percentile of active expert counts is approximately 19, and 0.21% of token routes exceed 32 experts. Based on this profile, we set $W _ { \mathrm { r o u t e } } = 3 2$ after capacity filtering. Let A<sup>cap</sup> denote the post-capacity assignment map, $s _ { t , e }$ the routing score, and $\mathcal { E } _ { t } ^ { \mathrm { c a p } } \overset { \cdot } { = } \{ e : \boldsymbol { A } _ { t , e } ^ { \mathrm { c a p } } = 1 \}$ }. The executed expert set is

$$
\mathcal { E } _ { t } ^ { \mathrm { r o l l } } = \left\{ \begin{array} { l l } { \mathcal { E } _ { t } ^ { \mathrm { c a p } } , } & { | \mathcal { E } _ { t } ^ { \mathrm { c a p } } | \leq W _ { \mathrm { r o u t e } } , } \\ { \mathrm { T o p K } _ { e \in \mathcal { E } _ { t } ^ { \mathrm { c a p } } } ( s _ { t , e } , W _ { \mathrm { r o u t e } } ) , } & { | \mathcal { E } _ { t } ^ { \mathrm { c a p } } | > W _ { \mathrm { r o u t e } } . } \end{array} \right.\tag{8}
$$

RW32 preserves post-capacity expert assignments for routes containing at most 32 experts and retains the highest-scoring 32 assignments for wider routes (Figure 2a). This reduces the width-dependent route representation to one eighth of its default size. Here, W denotes expert dispatch width; K in C-MOPD denotes the teacher-local vocabulary support size.

## 2.4.3 Final-Route Replay and Asynchronous Overlap

Capacity filtering depends on the tokens in each execution window. Because rollout and actor windows may differ, independently repeating this filtering can produce different expert assignments. Following R3 [4], we replay the final assignment map $A ^ { \mathrm { { r o } \overline { { 1 } } } }$ obtained after capacity filtering and width bounding. The actor reuses these discrete assignments and recomputes continuous routing weights under its current parameters,

$$
A ^ { \mathrm { a c t o r } } = A ^ { \mathrm { r o l l } } , \qquad p _ { t , e } ^ { \mathrm { a c t o r } } = A _ { t , e } ^ { \mathrm { r o l l } } \sigma ( z _ { t , e } ^ { \theta } ) \gamma ,\tag{9}
$$

where $\gamma$ is an optional routing scale.

AsyncOverlap runs rollout generation concurrently with actor updates (Figure 2b). Trajectories are generated by the preceding student policy version and evaluated by the current actor under the replayed expert assignments, introducing a one-step policy lag.

## 3 Experiments

## 3.1 Experimental Setup

Turing-20B-A2B has 20B total parameters, approximately 2B active parameters per token, and a native 128K context window [2]. Post-training retains its architecture and dynamic Quantile Routing. Unless stated otherwise, C-MOPD denotes the selected $K = 1 6 , c _ { \mathrm { m a x } } = 5$ checkpoint. Comparisons with the shared SFT checkpoint evaluate the complete specialist-training and consolidation pipeline; the calibration components are assessed in subsection 3.3.

We use a common generative evaluation pipeline based on OpenCompass task definitions [7]. The suite covers knowledge, reasoning, mathematics, code, verifiable instruction following, and function calling [8–19]. All Qwen3-8B reference scores in this section were reproduced using the same pipeline as the corresponding Turing checkpoints. Prompt mode, decoding settings, and responselength limits are matched within each comparison unless stated otherwise.

Direct-response prompts insert an empty <think></think> pair, while reasoning-enabled prompts leave the reasoning channel open. BFCL v4 uses reasoning-enabled evaluation and is labeled “Think” in the tables. Training budgets and data mixtures are reported in Appendix A; benchmark prompting and scoring protocols are summarized in Appendix C.

Differences in benchmark scores are expressed in percentage points; normalized integration scores and their differences are dimensionless. Bold marks the best reported value within each comparison group.

## 3.2 Multi-Domain Capability Integration

Following MOPD [1], we measure the fraction of each specialist’s improvement over SFT recovered by the consolidated student. For each checkpoint, $s _ { d }$ is the unweighted mean of benchmark scores within domain $d ,$ using the same benchmark composition for the student and its SFT and specialist references. We normalize this domain mean using the shared SFT checkpoint and the corresponding specialist, then average the recovery ratios across $n = 4$ domains with equal weights:

$$
\widetilde s _ { d } = \frac { s _ { d } - s _ { d } ^ { \mathrm { S F T } } } { s _ { d } ^ { \mathrm { t e a c h e r } } - s _ { d } ^ { \mathrm { S F T } } } , \qquad \mathrm { I n t e g r a t i o n } = \frac 1 n \sum _ { d = 1 } ^ { n } \widetilde s _ { d } .\tag{10}
$$

Math includes GSM8K and MATH in direct-response mode, and GSM8K and MATH-500 in reasoning mode. Code includes HumanEval, MBPP, and MBPP+ in both modes. IF uses IFEval, and Agent uses BFCL v4 in reasoning mode. The SFT and specialist references map to 0 and 1, respectively; normalized values are not clipped.

Table 2: Normalized specialist-gain recovery. The SFT and specialist rows define the normalization references; Macro is the unweighted mean of the four domain ratios.
<table><tr><td>Reference / model</td><td>Math</td><td>Code</td><td>IF</td><td>Agent</td><td>Macro</td></tr><tr><td>SFT reference</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td>Specialist references</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td> $\mathrm { C - M O P D } \left( K = 1 6 , \mathrm { c _ { m a x } = 5 } \right)$ </td><td>0.9407</td><td>1.0093</td><td>0.9016</td><td>1.1435</td><td>0.9988</td></tr></table>

The selected model achieves a macro integration score of 0.9988, with all four domain recovery ratios at or above 0.9016. Math and IF retain 94.07% and 90.16% of their specialists’ improvements over SFT, while the Code and Agent domain means exceed their specialist references. Table 3 reports stage-wise benchmark scores, and Appendix D provides the complete raw results underlying these normalized ratios.

Table 3: Stage-wise scores on the benchmarks directly associated with each specialist. All rows use direct-response evaluation except BFCL v4, which uses reasoning-enabled evaluation.
<table><tr><td>Metric</td><td>SFT</td><td>Specialist</td><td>C-MOPD</td><td>∆ vs. SFT</td><td>∆ vs. specialist</td></tr><tr><td>GSM8K</td><td>90.07</td><td>91.96</td><td>91.58</td><td>+1.51</td><td>-0.38</td></tr><tr><td>MATH</td><td>65.86</td><td>74.50</td><td>74.10</td><td>+8.24</td><td>-0.40</td></tr><tr><td>MBPP</td><td>69.31</td><td>72.75</td><td>73.28</td><td>+3.97</td><td>+0.53</td></tr><tr><td>HumanEval</td><td>79.88</td><td>86.59</td><td>84.15</td><td>+4.27</td><td>-2.44</td></tr><tr><td>IFEval</td><td>58.41</td><td>80.96</td><td>78.74</td><td>+20.33</td><td>-2.22</td></tr><tr><td>BFCL v4 (Think)</td><td>20.54</td><td>32.04</td><td>33.69</td><td>+13.15</td><td>+1.65</td></tr></table>

## 3.3 Calibration and Hyperparameter Ablations

## 3.3.1 Loss-Scale Normalization and Clipping

We compare three calibration variants at a fixed teacher-local support size of $K = 3 2$ (Table 4). The full objective achieves a macro integration score of 0.9962, compared with 0.9724 without normalization and clipping and 0.9834 with normalization but without clipping. Relative to both removal variants, the full objective scores higher in Code, IF, and Agent and lower in Math.

Table 4: Calibration ablations at $K = 3 2$ , with sampled-token MOPD as a separate method reference. No calibration removes normalization and clipping; No clipping retains normalization.
<table><tr><td>Objective</td><td>Variant</td><td>Math</td><td>Code</td><td>IF</td><td>Agent</td><td>Macro</td></tr><tr><td>Sampled-token MOPD</td><td></td><td>0.7747</td><td>0.7830</td><td>0.9836</td><td>1.1417</td><td>0.9208</td></tr><tr><td rowspan="3">Top-K (K = 32)</td><td>No calibration</td><td>0.8938</td><td>0.9142</td><td>0.9756</td><td>1.1061</td><td>0.9724</td></tr><tr><td>No clipping</td><td>0.8736</td><td>0.9356</td><td>0.9592</td><td>1.1652</td><td>0.9834</td></tr><tr><td>Full C-MOPD</td><td>0.8298</td><td>0.9514</td><td>0.9836</td><td>1.2200</td><td>0.9962</td></tr></table>

Sampled-token MOPD is included as a separate method reference; it differs in both the matching objective and the calibration configuration.

## 3.3.2 Support Size and Clipping Threshold

At $c _ { \mathrm { m a x } } = 5 , K = 1 6$ achieves the highest observed macro score of 0.9988, closely followed by $K = 3 2$ at 0.9962 (Table 5). Compared with K = 32, K = 16 also yields a higher minimum domain recovery ratio, 0.9016 versus 0.8298, while halving the retained teacher entries per token. At K = 32, $c _ { \operatorname* { m a x } } = 5$ achieves the highest macro score among the tested thresholds, compared with 0.9181 for $c _ { \operatorname* { m a x } } = 2$ and 0.8829 for $c _ { \operatorname* { m a x } } = 1 0$

Table 5: Normalized integration scores for two sweeps: support size K with $c _ { \operatorname* { m a x } } = 5$ fixed, and clipping threshold $c _ { \mathrm { m a x } }$ with $K = 3 2$ fixed.
<table><tr><td>Sweep</td><td>Value</td><td>Math</td><td>Code</td><td>IF</td><td>Agent</td><td>Macro</td></tr><tr><td rowspan="4">Support size</td><td> $K = 8$ </td><td>0.7846</td><td>0.9661</td><td>0.9264</td><td>1.1870</td><td>0.9660</td></tr><tr><td> $K = 1 6$ </td><td>0.9407</td><td>1.0093</td><td>0.9016</td><td>1.1435</td><td>0.9988</td></tr><tr><td> $K = 3 2$ </td><td>0.8298</td><td>0.9514</td><td>0.9836</td><td>1.2200</td><td>0.9962</td></tr><tr><td> $K = 6 4$ </td><td>0.8250</td><td>0.8226</td><td>0.9016</td><td>1.1809</td><td>0.9325</td></tr><tr><td rowspan="3">Clipping threshold</td><td> $c _ { \operatorname* { m a x } } = 2$ </td><td>0.7949</td><td>0.9403</td><td>0.8772</td><td>1.0600</td><td>0.9181</td></tr><tr><td> $c _ { \operatorname* { m a x } } = 5$ </td><td>0.8298</td><td>0.9514</td><td>0.9836</td><td>1.2200</td><td>0.9962</td></tr><tr><td> $c _ { \operatorname* { m a x } } = 1 0$ </td><td>0.7833</td><td>0.7651</td><td>0.8772</td><td>1.1061</td><td>0.8829</td></tr></table>

## 3.4 Broader Capability Evaluation

Relative to SFT, C-MOPD improves 10 of the 12 reported metrics (Table 6). The largest gains are on IFEval (+20.33 points), BFCL v4 (+13.15), and MATH (+8.24). CMMLU remains nearly unchanged (−0.01), while GPQA Diamond decreases by 5.05 points, from 40.40 to 35.35.

Table 6: Broader benchmark results. The first eleven rows use direct-response evaluation; BFCL v4 uses reasoning-enabled evaluation. ∆ denotes the change from Turing SFT.
<table><tr><td>Metric</td><td>Qwen3-8B</td><td>Turing SFT</td><td>C-MOPD</td><td>∆ vs. SFT</td></tr><tr><td colspan="5">General Knowledge and Reasoning Tasks</td></tr><tr><td>MMLU</td><td>74.23</td><td>76.59</td><td>78.10</td><td>+1.51</td></tr><tr><td>CMMLU</td><td>76.81</td><td>83.49</td><td>83.48</td><td>-0.01</td></tr><tr><td>C-Eval</td><td>67.74</td><td>80.48</td><td>81.07</td><td>+0.59</td></tr><tr><td>MMLU-Pro</td><td>25.71</td><td>45.75</td><td>48.53</td><td>+2.78</td></tr><tr><td>BBH</td><td>80.47</td><td>72.10</td><td>74.25</td><td>+2.15</td></tr><tr><td>GPQA Diamond</td><td>37.88</td><td>40.40</td><td>35.35</td><td>-5.05</td></tr><tr><td colspan="5">Mathematics Tasks</td></tr><tr><td>GSM8K</td><td>88.32</td><td>90.07</td><td>91.58</td><td>+1.51</td></tr><tr><td>MATH</td><td>79.24</td><td>65.86</td><td>74.10</td><td>+8.24</td></tr><tr><td colspan="5">Code Tasks</td></tr><tr><td>MBPP</td><td>75.13</td><td>69.31</td><td>73.28</td><td>+3.97</td></tr><tr><td>HumanEval</td><td>80.49</td><td>79.88</td><td>84.15</td><td>+4.27</td></tr><tr><td colspan="5">Instruction-Following Task</td></tr><tr><td>IFEval</td><td>83.18</td><td>58.41</td><td>78.74</td><td>+20.33</td></tr><tr><td colspan="5">Agent Task</td></tr><tr><td>BFCL v4 (Think)</td><td>38.64</td><td>20.54</td><td>33.69</td><td>+13.15</td></tr></table>

With reasoning enabled and an 8K response cap, C-MOPD improves over SFT on all four mathematics and code benchmarks (Table 7). Its unweighted mean score is 85.92, compared with 74.21 for SFT and 80.33 for Qwen3-8B.

Table 7: Reasoning-enabled mathematics and code evaluation with an 8K response cap for all models.
<table><tr><td>Metric</td><td>Qwen3-8B</td><td>Turing SFT</td><td>C-MOPD</td><td>∆ vs. SFT</td><td>∆ vs. Qwen3</td></tr><tr><td>GSM8K</td><td>92.72</td><td>92.95</td><td>95.68</td><td>+2.73</td><td>+2.96</td></tr><tr><td>MATH-500</td><td>80.80</td><td>73.00</td><td>82.40</td><td>+9.40</td><td>+1.60</td></tr><tr><td>MBPP</td><td>62.43</td><td>55.29</td><td>77.78</td><td>+22.49</td><td>+15.35</td></tr><tr><td>HumanEval</td><td>85.37</td><td>75.61</td><td>87.80</td><td>+12.19</td><td>+2.43</td></tr></table>

## 3.4.1 Output-Length Budgets

Table 8 reports output-cap hit rates, defined as the percentage of responses reaching the configured generation limit. At 8K, C-MOPD has fewer cap hits than Qwen3-8B on all four tasks, alongside the higher accuracies in Table 7. Across 2,382 responses, 22 C-MOPD outputs (0.92%) reach the cap, compared with 197 (8.27%) for Qwen3-8B.

Increasing the Qwen3-8B cap to 32K raises its unweighted mean accuracy from 80.33 to 84.64, compared with 85.92 for C-MOPD at 8K. On MATH-500, Qwen3-8B at 32K scores 95.00, exceeding C-MOPD’s 82.40 at 8K.

Table 8: Output-cap hit rates under reasoning-enabled prompts and Qwen3-8B accuracy at 32K. A cap hit is a response that reaches the configured token limit. Corresponding 8K accuracies are reported in Table 7. Bold marks the lowest cap-hit rate for each task.
<table><tr><td></td><td colspan="3">Cap hits (%)</td><td>Accuracy (%)</td></tr><tr><td>Metric</td><td>C-MOPD 8K</td><td>Qwen3 8K</td><td>Qwen3 32K</td><td>Qwen3 32K</td></tr><tr><td>GSM8K</td><td>0.08</td><td>1.74</td><td>0.30</td><td>92.95</td></tr><tr><td>MATH-500</td><td>3.60</td><td>19.00</td><td>2.60</td><td>95.00</td></tr><tr><td>HumanEval</td><td>0.00</td><td>12.20</td><td>10.37</td><td>86.59</td></tr><tr><td>MBPP</td><td>0.75</td><td>14.79</td><td>9.52</td><td>64.02</td></tr></table>

## 3.5 Dynamic-MoE Consolidation Efficiency

We compare serial Consolidate, one-step AsyncOverlap, and AsyncOverlapRW32, which combines asynchronous scheduling with a post-capacity expert-dispatch width limit of 32. Measurements cover the consolidation stage and are averaged over a common profiling interval (Table 9).

The combined configuration reduces step time from 398.7 to 139.3 seconds, yielding an observed 2.86× speedup. AsyncOverlap alone reduces trainer waiting but increases rollout time, resulting in a step time of 330.1 seconds. RW32 is profiled only with asynchronous overlap.

Table 9: Average consolidation-system measurements over the common profiling interval.
<table><tr><td>Metric</td><td>Consolidate</td><td>AsyncOverlap</td><td>AsyncOverlapRW32</td></tr><tr><td>Step time (s)</td><td>398.7</td><td>330.1</td><td>139.3</td></tr><tr><td>Trainer wait (s)</td><td>265.0</td><td>195.8</td><td>18.3</td></tr><tr><td>Trainer compute (s)</td><td>133.7</td><td>134.3</td><td>121.0</td></tr><tr><td>Rollout time (s)</td><td>242.9</td><td>312.1</td><td>93.9</td></tr><tr><td>Rollout throughput (tok/GPU/s)</td><td>183.9</td><td>143.1</td><td>478.4</td></tr><tr><td>Actor throughput (tok/s)</td><td>26,966</td><td>26,451</td><td>31,165</td></tr><tr><td>Actor TFLOPS</td><td>7.51</td><td>7.37</td><td>8.68</td></tr></table>

![](images/ded08b84d65583cc19c3982b472f6e67899f48aae2d0ba3e12457b3814ba71a1.jpg)  
Figure 3: Consolidation-system profile. (a) Mean step time decomposed into trainer compute and waiting. (b) Rollout throughput in tokens per GPU per second. Summary values compare AsyncOverlapRW32 with serial Consolidate. Rollout generation may overlap trainer phases.

Code benchmark scores at each configuration’s selected checkpoint are reported in Appendix E. These checkpoints occur at different training steps, so they provide a descriptive comparison of execution-checkpoint quality.

## 4 Conclusion

We presented an end-to-end post-training framework that consolidates four independently trained domain specialists into a single dynamic-MoE student. C-MOPD calibrates multi-teacher on-policy distillation through teacher-wise loss-scale normalization and token-level clipping. The selected model achieves a normalized four-domain integration score of 0.9988 and retains at least 90.16% of the corresponding specialist’s measured improvement over the shared SFT baseline in every domain. At a fixed support size of K = 32, the full objective achieves a higher macro integration score than variants with calibration components removed. The complete specialist-training and consolidation pipeline also improves over SFT on all four reasoning-enabled mathematics and code benchmarks and on ten of twelve metrics in the broader evaluation suite.

On PPU hardware, the execution workflow combines asynchronous rollout–training overlap, boundedwidth expert dispatch after capacity filtering, and final-route replay. This combined configuration reduces the profiled average consolidation step time from 398.7 to 139.3 seconds, yielding an observed 2.86× speedup over serial execution.

## 5 Contributors

We sincerely thank every member of the team for their dedication and contributions to the Turing posttraining project. This work brings together data construction, supervised fine-tuning, domain-specific reinforcement learning, calibrated multi-teacher on-policy distillation, MoE route replay and systems optimization, evaluation, and technical-report preparation. The results reflect a close collaboration across algorithm research, data and reward engineering, distributed training infrastructure, and model evaluation.

Advisors: Hang Zhang, Honggou Yang, Xianming Liu

Project Lead: Qiman Wu

Contributors: Yue He<sup>\*</sup>, Jiahui Hu<sup>\*</sup>, Shaman Tang, Hua Zhou, Menglin Li, Yizhao Wang, Yuheng Zhang, Da Zhu, Zhou Wang, Anhua Liu, Hanlin Chen, Mingyuan Wang, Rui Xin, Yuhua Wei, Pengfei Diao, Shuang Su, Minghao Li, Haojie Yang, Siqi Liu, Kai Yang, Weichao Huang.

<sup>\*</sup>Core contribution.

## References

[1] Wenhan Ma, Jianyu Wei, Liang Zhao, et al. MOPD: Multi-teacher on-policy distillation for capability integration in llm post-training, 2026. URL https://arxiv.org/abs/2606.30406.

[2] Yuheng Zhang, Yizhao Wang, Da Zhu, et al. TuringLLM: Efficiently scaling foundation models toward physical ai, 2026. URL https://arxiv.org/abs/2608.30567.

[3] Yuqian Fu, Haohuan Huang, Kaiwen Jiang, et al. Revisiting on-policy distillation: Empirical failure modes and simple fixes, 2026. URL https://arxiv.org/abs/2603.25562.

[4] Wenhan Ma, Hailin Zhang, Liang Zhao, et al. Stabilizing moe reinforcement learning by aligning training and inference routers, 2025. URL https://arxiv.org/abs/2510.11370.

[5] Qwen Team. Qwen3 quickstart: Switching between thinking and non-thinking modes. Qwen Documentation, 2025. URL https://qwen.readthedocs.io/en/latest/getting\_started/quickstart.html. Accessed 2026-09-03.

[6] Chujie Zheng, Shixuan Liu, Mingze Li, et al. Group sequence policy optimization, 2025. URL https://arxiv.org/abs/2507.18071.

[7] Maosong Cao, Kai Chen, Haodong Duan, et al. OpenCompass: A universal evaluation platform for large language models, 2026. URL https://arxiv.org/abs/2605.19276.

[8] Dan Hendrycks, Collin Burns, Steven Basart, et al. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021.

[9] Haonan Li, Yixuan Zhang, Fajri Koto, et al. CMMLU: Measuring massive multitask language understanding in chinese, 2024.

[10] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, et al. C-Eval: A multi-level multi-discipline chinese evaluation suite for foundation models. In Advances in Neural Information Processing Systems, 2023.

[11] Yubo Wang, Xueguang Ma, Ge Zhang, et al. MMLU-Pro: A more robust and challenging multi-task language understanding benchmark, 2024.

[12] David Rein, Betty Li Hou, Asa Cooper Stickland, et al. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

[13] Mirac Suzgun, Nathan Scales, Nathanael Scharli, et al. Challenging BIG-Bench tasks and whether chain-of-thought can solve them. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, 2023.

[14] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, et al. Training verifiers to solve math word problems, 2021.

[15] Dan Hendrycks, Collin Burns, Saurav Kadavath, et al. Measuring mathematical problem solving with the MATH dataset. In Advances in Neural Information Processing Systems, 2021.

[16] Mark Chen, Jerry Tworek, Heewoo Jun, et al. Evaluating large language models trained on code, 2021.

[17] Jacob Austin, Augustus Odena, Maxwell Nye, et al. Program synthesis with large language models, 2021.

[18] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, et al. Instruction-following evaluation for large language models, 2023.

[19] Fanjia Yan, Huanzhi Mao, Charlie Cheng-Jie Ji, et al. Berkeley function-calling leaderboard. Gorilla Project, University of California, Berkeley, 2024. URL https://gorilla.cs.berkeley.edu/leaderboard.html.

[20] Alibaba Cloud. Qwen3.5-Plus. Alibaba Cloud Model Studio Documentation, 2026. URL https://help.aliyun.com/en/model-studio/qwen3-5-plus. Accessed: 2026-09-03.

[21] DeepSeek-AI. DeepSeek-V4-Flash. DeepSeek API Documentation, 2026. URL https://api-docs.deepseek.com/news/news260424/. Accessed: 2026-09-03.

## Appendix

## A Training Recipe Details

Table 10 summarizes the training budgets for shared SFT, specialist RL, and C-MOPD consolidation.

Table 10: Post-training stages and data budgets. Pool sizes are rounded. RL and C-MOPD batch sizes distinguish sampled prompts from rollouts per prompt; epochs denote approximate passes through the corresponding data pool.
<table><tr><td>Stage</td><td>Steps</td><td>Batch / step</td><td>Training pool</td><td>Epochs</td></tr><tr><td>SFT</td><td>10,000</td><td>512 packed rows</td><td>5.27M samples</td><td>~1.0</td></tr><tr><td>Code RL</td><td>240</td><td>128 prompts × 16 rollouts</td><td>16K prompts</td><td>~2.0</td></tr><tr><td>IF RL</td><td>480</td><td>128 prompts × 16 rollouts</td><td>16.2K prompts</td><td>~4.0</td></tr><tr><td>Math RL</td><td>550</td><td>128 prompts × 16 rollouts</td><td>35.5K prompts</td><td>~2.0</td></tr><tr><td>Agent RL</td><td>512</td><td>64 prompts × 16 rollouts</td><td>8.5K prompts</td><td>~4.0</td></tr><tr><td>C-MOPD</td><td>400</td><td>128 prompts × 1 rollout</td><td>51.5K prompts</td><td>~1.0</td></tr></table>

## A.1 Supervised Data and Response Modes

The shared SFT mixture contains approximately 5.27M examples (Table 11). Qwen3.5-Plus and DeepSeek-V4-Flash provide mathematics and code supervision [20, 21]. For selected source prompts, the current Turing checkpoint generates ten candidate responses. Incorrect responses are retained for teacher correction, which produces corrected SFT targets. Reasoning-enabled targets preserve the reasoning trace, while direct-response targets retain only the final response and use an empty <think></think> assistant prefix. Across the full mixture, approximately 53% of examples use the reasoning-enabled form and 47% use the direct-response form. Within mathematics and code distillation, the reasoning-enabled share is 85%.

Table 11: Composition of the SFT mixture. Counts are rounded to two decimal places in millions; shares use unrounded counts.
<table><tr><td>Data category</td><td>Examples</td><td>Share</td></tr><tr><td>Mathematics and code distillation</td><td>2.69M</td><td>51.0%</td></tr><tr><td>Agent tool use</td><td>1.51M</td><td>28.7%</td></tr><tr><td>General and instruction following (including ICL)</td><td>0.67M</td><td>12.7%</td></tr><tr><td>Long context and dialogue</td><td>0.40M</td><td>7.6%</td></tr><tr><td>Total</td><td>5.27M</td><td>100%</td></tr></table>

## A.2 Capability-Boundary Filtering

During iterative data construction, the current SFT checkpoint generates $n \geq 4$ responses at temperature 1.0 for each prompt x. A task verifier assigns $r _ { i } = 1$ to a correct response and $r _ { i } = 0$ otherwise. The empirical pass rate is

$$
{ \widehat { p } } ( x ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } r _ { i } .\tag{11}
$$

We retain prompts satisfying

$$
{ \frac { 2 } { n } } \leq { \widehat { p } } ( x ) \leq { \frac { n - 2 } { n } } .\tag{12}
$$

This offline filter retains prompts with at least two verified successes and at least two failures among the sampled responses, selecting prompts with mixed observed outcomes under the current checkpoint.

## A.3 Multiple-Choice Answer-Format Randomization

For a multiple-choice item $( q , C , \ell )$ , let q denote the question, C the candidate list, and ℓ the correct candidate in the original order. We sample a prompt template τ, an answer format $f ,$ and a candidate permutation ϖ. The target map $g _ { f , \varpi } ( \ell )$ accounts for the selected format and shuffled candidate order. A deterministic parser $E _ { f }$ returns ⊥ when the response violates the format. The joint correctnessand-format reward is

$$
r _ { \mathrm { m c q } } ( y ) = { \bf 1 } [ E _ { f } ( y ) \neq \bot ] { \bf 1 } [ E _ { f } ( y ) = g _ { f , \varpi } ( \ell ) ] .\tag{13}
$$

The format library contains 29 English and 29 Chinese variants, including alphabetic, numeric, Chinese-ordinal, and short-text forms. The reward is one only when the response follows the selected format and identifies the correct option.

## A.4 C-MOPD Domain Sampling and Loss Configuration

The consolidation pool contains 51,500 prompts: 15,000 for Math, 10,000 for Code, 16,500 for IF, and 10,000 for Agent. These counts describe the consolidation pool; the specialist RL pool sizes are listed separately in Table 10. The domain counts define the prompt-sampling distribution

$$
( \rho _ { \mathrm { M a t h } } , \rho _ { \mathrm { C o d e } } , \rho _ { \mathrm { I F } } , \rho _ { \mathrm { A g e n t } } ) \approx ( 0 . 2 9 1 , 0 . 1 9 4 , 0 . 3 2 0 , 0 . 1 9 4 ) _ { \cdot }
$$

The prompt-domain sampling probabilities and teacher-loss multipliers remain fixed throughout the 400 consolidation steps, including 20 warmup steps and 380 subsequent steps. Valid response-token losses use the token\_mean reduction in Equation 7. The fixed teacher-loss multipliers, applied after normalization and clipping, are given in Appendix B. The selected C-MOPD configuration uses $K = 1 6 ,$ , a standard-deviation floor of 0.1, and $c _ { \operatorname* { m a x } } = 5 ;$ the ablations vary K and $c _ { \mathrm { m a x } }$ as specified in subsection 3.3.

## B Fixed Teacher Weights and Specialist Diagnostics

The teacher-loss multipliers $w _ { d }$ in Equation 7 are fixed before consolidation as $\begin{array} { c c l } { ( w _ { \mathrm { M a t h } } , w _ { \mathrm { C o d e } } , w _ { \mathrm { I F } } , w _ { \mathrm { A g e n t } } ) } & { = } & { ( 1 . 0 , 1 . 0 , 0 . 3 , 1 . 0 ) } \end{array}$ and remain unchanged throughout training. Figure 4 summarizes specialist RL behavior. These curves provide diagnostic context; the multipliers are fixed hyperparameters rather than values derived from the plotted statistics.

![](images/5341e88b6754f426e168f4ae177f9e9670fddda0cbca07761ec0128c680347c1.jpg)  
Figure 4: Specialist RL diagnostics over the displayed training intervals. Normalized verifier reward tracks task-specific rollout quality; the effective-prompt count records groups with non-degenerate relative-advantage signals; response length is measured in generated tokens; KL is measured against each specialist’s RL reference policy rather than the C-MOPD student. Light lines show per-step values and dark lines show TensorBoard EMA curves $( s = 0 . 9 )$ .

Reward and effective-prompt counts describe training behavior within each domain’s verifier and sampling setup. The effective-prompt counts also depend on the per-step prompt budgets, which differ between Agent and the other domains (Table 10). Mean response length records tokens generated during specialist RL.

During consolidation, $\rho _ { d }$ controls prompt-domain sampling and $w _ { d }$ multiplies the corresponding calibrated token losses. The scale statistic $\sigma _ { d } ^ { ( \mu ) }$ rescales token losses, and the threshold $c _ { \mathrm { m a x } }$ caps their normalized values. Under token\_mean, a domain’s contribution also depends on its valid student-response token count. Scale calibration acts on loss values and does not normalize the number of contributing tokens.

## C Evaluation Protocol

Table 12 summarizes the benchmark splits, prompting settings, and scoring metrics. Few-shot and CoT settings describe benchmark prompting, while the direct-response and Think labels specify the model’s response mode (section 3).

Table 12: Benchmark splits, prompting settings, and scoring metrics used in the reported evaluations.
<table><tr><td>Benchmark</td><td>Split</td><td>Prompting</td><td>Metric</td></tr><tr><td>MMLU / CMMLU</td><td>Test</td><td>5-shot</td><td>Acc.</td></tr><tr><td>C-Eval</td><td>Validation</td><td>5-shot</td><td>Acc.</td></tr><tr><td>MMLU-Pro</td><td>Test</td><td>5-shot CoT</td><td>Acc.</td></tr><tr><td>GPQA Diamond</td><td>Diamond</td><td>5-shot CoT</td><td>Acc.</td></tr><tr><td>BBH</td><td>All 27 tasks</td><td>3-shot CoT</td><td>EM</td></tr><tr><td>GSM8K</td><td>Test</td><td>4-shot CoT</td><td>EM</td></tr><tr><td>MATH</td><td>Test (5,000 examples)</td><td>4-shot</td><td>EM</td></tr><tr><td>MATH-500</td><td>Test (500 examples)</td><td>0-shot</td><td>EM</td></tr><tr><td>HumanEval / MBPP</td><td>Standard benchmark splits</td><td>0-shot</td><td>Execution</td></tr><tr><td>IFEval</td><td>Official set (541 prompts)</td><td>0-shot</td><td>Prompt-level strict Acc.</td></tr><tr><td>BFCL v4</td><td>Official suite</td><td>Benchmark-specific think mode</td><td>Official score</td></tr></table>

The direct-response mathematics and code comparisons use the same task-specific caps for C-MOPD and Qwen3: 512 tokens for GSM8K, 2,048 for MATH, and 8,192 for HumanEval and MBPP. The dedicated reasoning-enabled comparison uses an 8K cap; Qwen3 at 32K is reported as a larger-budget reference.

## C.1 Base-Model Reference

Table 13 reports base-model results separately because the base and post-trained checkpoints use different prompting and response formats. This reference uses GPQA Main, whereas the post-training comparison in Table 6 uses GPQA Diamond.

Table 13: Base-model results under the base-model prompting setup. GPQA uses the Main subset.
<table><tr><td>Metric</td><td>Qwen3-8B Base</td><td>Turing-20B-A2B Pretrain</td></tr><tr><td>MMLU</td><td>78.92</td><td>79.83</td></tr><tr><td>CMMLU</td><td>81.28</td><td>84.17</td></tr><tr><td>C-Eval</td><td>83.06</td><td>81.76</td></tr><tr><td>MMLU-Pro</td><td>55.35</td><td>55.17</td></tr><tr><td>BBH</td><td>73.67</td><td>73.61</td></tr><tr><td>GPQA Main</td><td>34.82</td><td>37.05</td></tr><tr><td>GSM8K</td><td>90.07</td><td>74.53</td></tr><tr><td>MATH</td><td>56.12</td><td>62.20</td></tr></table>

## D Full Raw C-MOPD Results

Table 14 provides the raw scores underlying the normalized integration analysis. Sampled-token MOPD serves as a separate method reference. The three Top-K calibration variants use $K = 3 2$ while the selected C-MOPD model uses K = 16 and $c _ { \operatorname* { m a x } } = 5$ . The teacher reference in each row

is the corresponding domain specialist. Unmarked rows use direct-response evaluation, and rows marked Think use reasoning-enabled evaluation.

Table 15 reports the support-size and clipping-threshold sweeps. Domain scores and the macro integration score are calculated from these raw results using Equation 10.

Table 14: Raw benchmark scores (%) for the reference checkpoints, sampled-token MOPD, three calibration variants at K = 32, and the selected C-MOPD model. Teacher denotes the corresponding domain specialist. Bold marks the highest score in each row.
<table><tr><td></td><td colspan="2">References</td><td>Sampled</td><td colspan="3">Objective controls at  $K = 3 2$ </td><td>Selected</td></tr><tr><td>Metric</td><td>SFT</td><td>Teacher</td><td>MOPD</td><td>No calibration</td><td>No clipping</td><td>Full</td><td>K = 16 Full</td></tr><tr><td colspan="8">Mathematics Tasks</td></tr><tr><td>GSM8K</td><td>90.07</td><td>91.96</td><td>90.45</td><td>91.81</td><td>91.13</td><td>91.66</td><td>91.58</td></tr><tr><td>MATH</td><td>65.86</td><td>74.50</td><td>70.32</td><td>73.28</td><td>73.14</td><td>73.34</td><td>74.10</td></tr><tr><td>GSM8K (Think)</td><td>92.95</td><td>95.68</td><td>95.53</td><td>95.38</td><td>95.53</td><td>95.38</td><td>95.68</td></tr><tr><td>MATH-500 (Think)</td><td>73.00</td><td>83.00</td><td>83.60</td><td>82.20</td><td>82.40</td><td>80.80</td><td>82.40</td></tr><tr><td colspan="8">Code Tasks</td></tr><tr><td>HumanEval</td><td>79.88</td><td>86.59</td><td>80.49</td><td>85.98</td><td>85.37</td><td>82.93</td><td>84.15</td></tr><tr><td>MBPP</td><td>69.31</td><td>72.75</td><td>69.84</td><td>72.22</td><td>73.81</td><td>73.81</td><td>73.28</td></tr><tr><td>MBPP+</td><td>57.41</td><td>62.17</td><td>61.64</td><td>63.23</td><td>62.96</td><td>63.23</td><td>64.02</td></tr><tr><td>HumanEval (Think)</td><td>75.61</td><td>86.59</td><td>85.37</td><td>85.98</td><td>85.37</td><td>87.80</td><td>87.80</td></tr><tr><td>MBPP (Think)</td><td>55.29</td><td>78.04</td><td>76.46</td><td>75.66</td><td>76.46</td><td>76.46</td><td>77.78</td></tr><tr><td>MBPP+ (Think)</td><td>48.68</td><td>66.67</td><td>64.55</td><td>64.02</td><td>64.55</td><td>65.34</td><td>66.40</td></tr><tr><td colspan="8">Instruction-Following Task</td></tr><tr><td>IFEval</td><td>58.41</td><td>80.96</td><td>80.59</td><td>80.41</td><td>80.04</td><td>80.59</td><td>78.74</td></tr><tr><td colspan="8">Agent Task</td></tr><tr><td>BFCL v4 (Think)</td><td>20.54</td><td>32.04</td><td>33.67</td><td>33.26</td><td>33.94</td><td>34.57</td><td>33.69</td></tr></table>

Table 15: Raw benchmark scores (%) for the support-size and clipping-threshold sweeps. Bold marks the highest score within each row and sweep block.
<table><tr><td></td><td colspan="4"> $\mathbf { S u p p o r t s i z e } , c _ { \mathrm { m a x } } = 5$ </td><td colspan="3">Clipping,  $K = 3 2$ </td></tr><tr><td>Metric</td><td> $K = 8$ </td><td> $K = 1 6$ </td><td> $K = 3 2$ </td><td> $K = 6 4$ </td><td> $c _ { \operatorname* { m a x } } = 2$ </td><td> $c _ { \operatorname* { m a x } } = 5$ </td><td> $c _ { \operatorname* { m a x } } = 1 0$ </td></tr><tr><td colspan="8">Mathematics Tasks</td></tr><tr><td>GSM8K</td><td>90.75</td><td>91.58</td><td>91.66</td><td>91.21</td><td>90.30</td><td>91.66</td><td>90.90</td></tr><tr><td>MATH</td><td>73.90</td><td>74.10</td><td>73.34</td><td>73.22</td><td>73.22</td><td>73.34</td><td>73.30</td></tr><tr><td>GSM8K (Think)</td><td>95.68</td><td>95.68</td><td>95.38</td><td>94.84</td><td>95.45</td><td>95.38</td><td>95.30</td></tr><tr><td>MATH-500 (Think)</td><td>79.80</td><td>82.40</td><td>80.80</td><td>81.80</td><td>81.40</td><td>80.80</td><td>80.60</td></tr><tr><td colspan="8">Code Tasks</td></tr><tr><td>HumanEval</td><td>82.93</td><td>84.15</td><td>82.93</td><td>81.10</td><td>84.76</td><td>82.93</td><td>82.93</td></tr><tr><td>MBPP</td><td>75.13</td><td>73.28</td><td>73.81</td><td>73.54</td><td>72.22</td><td>73.81</td><td>70.37</td></tr><tr><td>MBPP+</td><td>62.70</td><td>64.02</td><td>63.23</td><td>61.90</td><td>62.17</td><td>63.23</td><td>60.58</td></tr><tr><td>HumanEval (Think)</td><td>87.20</td><td>87.80</td><td>87.80</td><td>84.76</td><td>88.41</td><td>87.80</td><td>85.98</td></tr><tr><td>MBPP (Think)</td><td>77.25</td><td>77.78</td><td>76.46</td><td>75.40</td><td>75.93</td><td>76.46</td><td>74.34</td></tr><tr><td>MBPP+ (Think)</td><td>65.34</td><td>66.40</td><td>65.34</td><td>64.29</td><td>65.34</td><td>65.34</td><td>62.96</td></tr><tr><td colspan="8">Instruction-Following Task</td></tr><tr><td>IFEval</td><td>79.30</td><td>78.74</td><td>80.59</td><td>78.74</td><td>78.19</td><td>80.59</td><td>78.19</td></tr><tr><td colspan="8">Agent Task</td></tr><tr><td>BFCL v4 (Think)</td><td>34.19</td><td>33.69</td><td>34.57</td><td>34.12</td><td>32.73</td><td>34.57</td><td>33.26</td></tr></table>

## E Execution Configuration Quality

Table 16 reports timing and code benchmark scores for the PPU execution configurations, providing a run-level view of their quality–efficiency trade-off. The code scores characterize the model states evaluated for each configuration; because these states come from different training steps, score differences should not be attributed to the execution configuration alone. Common-interval system averages are reported in Table 9.

Table 16: Timing and code benchmark scores for the PPU execution configurations.
<table><tr><td>Configuration</td><td>Step time (s)</td><td></td><td></td><td>Wait (%) Speedup HumanEval MBPP MBPP+</td><td></td><td></td></tr><tr><td>Consolidate</td><td>381.4</td><td>65.4</td><td>1.0×</td><td>86.59</td><td>78.04</td><td>66.67</td></tr><tr><td>AsyncOverlap</td><td>335.3</td><td>59.8</td><td>1.1×</td><td>84.76</td><td>76.46</td><td>65.34</td></tr><tr><td>AsyncOverlapRW32</td><td>137.2</td><td>12.4</td><td>2.8×</td><td>87.20</td><td>76.19</td><td>65.87</td></tr></table>