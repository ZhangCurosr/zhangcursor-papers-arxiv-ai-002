# Routing by Reasoning Need: Trajectory-Aware Decoding Control for Diffusion Vision-Language Models

Yixiang Liu<sup>1</sup>\*, Zhongxing Xu<sup>2</sup>\*, Zhonghua Wang<sup>2</sup>\*, Xiaoying Tang<sup>1†</sup>, <sup>1</sup>Southern University of Science and Technology, <sup>2</sup>Monash University Correspondence: tangxy@sustech.edu.cn

## Abstract

Diffusion vision-language models generate answers through iterative refinement, exposing intermediate answer trajectories that can be in spected and controlled at inference time. How ever, this controllability creates a reasoningneed mismatch, where a universal generation length is applied to questions with different reasoning demands. Visually closed ques tions may be harmed by continued refinement after a stable answer has formed, whereas reasoning-sensitive questions may be harmed by premature commitment. We formulate this problem as reasoning-budget mismatch and study it in LLaDA-V. Rather than choosing a universal generation length, our training-free controller routes each example to early commitment, baseline preservation, or reasoningsupportive decoding using trajectory signals from answer closure, commitment evidence, and representation revision pressure, without using ground-truth answers. Across answerfocused, mixed-reasoning, and CoT-sensitive benchmarks, routed control improves robustness over fixed long decoding, pure short decoding, and single-rule interventions. The gains are not explained by shorter outputs alone. Answerclosed examples often benefit from commitment, whereas CoT-sensitive examples require preserving or supporting intermediate reasoning. Taken together, these results suggest diffusion VLM decoding should route inferencetime control by the state suggested by the observed trajectory instead of relying on a universal decoding length.

## 1 Introduction

Diffusion vision-language models (VLMs) change what can be controlled during decoding. In left-toright autoregressive generation, the model commits to a token sequence one position at a time. After a token is emitted, later computation can condition on it but cannot directly revise it. Diffusion language models instead generate through iterative denoising or masked refinement, and recent diffusion VLMs extend this design to multimodal instruction following (Nie et al., 2026; You et al., 2026). This makes decoding a sequence of observable revisions over a partially specified answer rather than a single irreversible trace. Because these intermediate answer trajectories are visible at inference time, they can indicate whether a candidate answer has stabilized, whether the trajectory provides sufficient grounding-and-format evidence for commitment, and whether the model is still changing the representation that will support the final answer. For a visually closed VQA example, such as many compact diagnostic questions in MME (Fu et al., 2025), the answer may appear early and remain consistent across refinements (Li et al., 2026b; Xu et al., 2026). By contrast, for geometry, chemistry, or multi-hop reasoning examples, such as MMMUstyle expert questions (Yue et al., 2024), the same early surface answer may be unreliable because the model has not yet organized the relevant relations. As shown in the top row of Figure 1, answer closure can occur early for compact visual questions, while reasoning-intensive examples may require later refinement before a reliable answer emerges. Diffusion decoding therefore exposes both the final answer and the revision process that leads to it.

![](images/eda2d168092eb4ee723f05854b835e01c672dae603ec568f3af9cb8e2b976059.jpg)  
Figure 1: Motivation for reasoning-need matched decoding control. The top row shows answer-closure timing under the fixed 128-step decoding budget across benchmark regimes. Gray regions indicate decoding intervals where no valid answer has converged within the 128- step trajectory. The bottom row illustrates three representative trajectory outcomes, including over-refinement after closure, matched control, and premature commitment before closure. These patterns motivate routing the decoding action by the state suggested by the observed trajectory instead of applying a universal generation length.

This controllability also creates risk under a fixed decoding budget (Li et al., 2026a). In our experiments, the default fixed-budget reference uses a 128-step refinement trajectory, which assigns every sample the same amount of decoding computation even when questions differ sharply in their reasoning demands. Visually closed questions may be harmed by continued refinement after a stable answer has formed, while reasoning-sensitive questions may be harmed by premature commitment. We call this failure a reasoning-budget mismatch. The mismatch is not simply that some outputs should be shorter and others longer. It is that different samples require different inference-time control actions because the uncertainty left after early refinement has different meanings. In answer-closed cases, the suitable action is early commitment. In unresolved cases, where intervention is not justified, the safer action is to preserve the fixed-budget baseline. In CoT-sensitive cases, where the final answer should remain supported by intermediate reasoning, the suitable action is reasoning-supportive decoding rather than premature commitment.

Figure 1 bottom illustrates this mismatch. Some trajectories reach a stable answer early and are later degraded by additional refinement. Some trajectories remain stable under continued decoding and should be preserved. Others do not yet contain a reliable answer or support-bearing rationale, so early commitment would truncate the reasoning process. These cases suggest that the central question is not whether diffusion decoding should be globally short or long, but which control action is appropriate for the current trajectory.

This paper studies reasoning-budget mismatch in LLaDA-V. Rather than choosing a universal generation length, we propose a training-free inferencetime controller that routes each example to one of three actions: early commitment, baseline preservation, or reasoning-supportive decoding. The controller uses trajectory signals computed without ground truth from answer closure, commitment evidence, and representation revision pressure. These signals do not estimate answer correctness or define a ground-truth semantic label of the example. Instead, we use reasoning need as an operational decoding concept: it denotes the control action suggested by the observed diffusion trajectory. This distinction explains why fixed long decoding, pure short decoding, and single-rule interventions can each help in some regimes but fail in others. Diffusion VLM decoding should route inference-time control by the state suggested by the observed trajectory rather than relying on a universal decoding length.

Our contributions are threefold.

• We identify reasoning-budget mismatch in diffusion VLM decoding, where the same refinement budget can over-refine answer-closed examples while under-supporting reasoningsensitive examples.

• We propose a training-free routed decoding controller for LLaDA-V, which selects among early commitment, baseline preservation, and reasoning-supportive decoding using trajectory signals computed without ground-truth answers.

• We evaluate the controller across answerfocused, mixed-reasoning, and CoT-sensitive settings, showing that routing inference-time control actions is more robust than applying a single global length rule.

## 2 Related Work

Diffusion VLMs and adaptive decoding. Diffusion language models generate text through iterative denoising or masked refinement rather than left-to-right token commitment (Nie et al., 2026). LLaDA-V extends this formulation to multimodal instruction following and visual reasoning, making intermediate states observable during visionlanguage generation (You et al., 2026). This process provides a natural interface for inference-time control, since the decoder can inspect partial predictions, uncertainty patterns, and refinement dynamics before the final answer is fixed. Recent adaptive decoding methods exploit such internal signals by adjusting denoising steps, block granularity, token representations, or reasoning modes (Wu et al., 2026; Lu et al., 2026; Yang et al., 2025;

Xu et al., 2026; Li et al., 2026a). These methods show that fixed decoding schedules are often sub optimal, but their objectives are usually efficiency, uncertainty reduction, or hallucination mitigation. Our work studies a different mismatch. The appropriate control action depends on the sample’s trajectory state. Some samples are already answer closed and benefit from early commitment, some remain unresolved, and others require reasoning supportive decoding rather than shorter decoding. Reasoning length, grounding, and CoT sensitivity. Longer reasoning is not uniformly beneficial. In multimodal settings, extended reasoning can improve task solving while weakening visual grounding, and reasoning-induced hallucination should be distinguished from perception-induced errors (Xu et al., 2025; Dongre et al., 2025). CoT prompting is task-conditional. It helps when intermediate reasoning supports the decision, but can be unnecessary or harmful when the answer is already recoverable from the input (Sprague et al., 2025; Cheng et al., 2025; Balasubramanian et al., 2025). This suggests that reasoning should be instanceconditional rather than globally enabled or globally suppressed. The relevant decision is not whether to use more or less reasoning for all samples, but whether the current trajectory still requires structured reasoning support. Grounded reasoning methods show that when reasoning is necessary, the bet ter remedy is often to anchor intermediate steps in visual evidence rather than truncate the chain (Man et al., 2025). Our distinction between answerclosed and CoT-sensitive cases follows this view. Early commitment is useful when further refine ment mainly adds drift, while preserved or struc tured reasoning is useful when the decision still depends on unresolved visual-textual evidence.

Decoding-time control and its boundary. Training-free decoding-time methods modify generation without retraining. Representative approaches include visual contrastive decoding, over-trust penalties, layer-contrastive decoding, self-corrective decoding, and memory-based visual refinement (Leng et al., 2024; Huang et al., 2024; Chuang et al., 2024; Huo et al., 2025; Zou et al., 2025). Entropy-aware decoding further shows that token-level uncertainty can guide mode switching and visual-anchor injection during uncertain reasoning states (Xu et al., 2025). These methods demonstrate that internal generation signals can support effective test-time intervention. Our results suggest a boundary for single-rule interventions. Pure short decoding, fixed long decoding, and revision-based control each help in some regimes and fail in others. We therefore treat internal signals not as universal hallucination or factuality detectors, but as evidence for choosing among control actions. This reframes diffusion VLM decoding as a regime-conditioned decision problem, where the system must decide whether to commit early, preserve the baseline path, or apply reasoning-supportive decoding.

## 3 Method

Figure 2 gives an overview of our reasoningneed routed decoding framework. The controller observes a fixed-budget diffusion trajectory and extracts three signals from intermediate refinement states, including answer closure, grounding-and-format evidence from format and visual-engagement guards, and representation revision pressure. These signals do not use gold labels or estimate correctness directly. They operationalize the state suggested by the observed trajectory as the safest inference-time action under the observed trajectory, routing each example to early commitment, baseline preservation, or reasoningsupportive decoding instead of using a universal short or long budget.

## 3.1 Problem Setup

We study multimodal question answering with a frozen diffusion VLM. Each input is x = $( x _ { v } , x _ { q } , c , I )$ , where $x _ { v }$ is the image, $x _ { q }$ is the question, c denotes optional answer choices, and I denotes an optional reasoning instruction. The model returns an answer aˆ and, when requested, a rationale rˆ.

Let $\boldsymbol { B } = ( T , L , M )$ denote the decoding budget, where T is the number of denoising steps, L is the generation length, and M is the block length. The fixed-budget reference uses the same budget for every sample. At denoising step t, the diffusion state is ${ z } _ { t } = ( y _ { t } , H _ { t } , P _ { t } )$ , where $y _ { t }$ is the token state, $H _ { t }$ denotes hidden states, and $P _ { t }$ denotes token distributions. We write the fixed-budget trajectory and parsed output as

$$
\begin{array} { r l } & { \quad z _ { t + 1 } = F _ { \theta } ( z _ { t } , x , B ) , } \\ & { \tau _ { B } ( x ) = ( z _ { 0 } , \ldots , z _ { T } ) , } \\ & { o _ { B } ( x ) = \psi ( y _ { T } ) = ( \hat { a } _ { B } , \hat { r } _ { B } ) . } \end{array}\tag{1}
$$

Here θ is frozen and ψ is the benchmark parser. Our controller does not update model weights, train a

![](images/590102c575074ba3a78d0cddd2793ab47b5f4093258aae9adf5e7b308420deb8.jpg)  
Figure 2: Overview of reasoning-need routed decoding control.

verifier, or use gold labels. It only selects an action from the decoding trajectory.

## 3.2 Representative Trajectory Patterns

Figure 3 illustrates three answer-token trajectories that motivate our routing policy. The plots show when output positions are decoded, where top-1 predictions change, and whether the answer token remains correct or drifts. The examples correspond to three decoding states: early answer closure with later drift, late convergence that should be preserved, and reasoning-heavy instability that requires additional support. These patterns motivate selecting among early commitment, baseline preservation, and reasoning-supportive decoding according to the sample’s reasoning need, rather than using one fixed decoding length.

## 3.3 Routing Evidence

The router uses three trajectory signals: answer closure, commitment evidence, and representation revision pressure.

Answer closure. Answer closure measures whether the parsed answer has stabilized. Let $\psi _ { A } ( y _ { t } )$ return the parsed answer from state $y _ { t } .$ , or ⊥ if no valid answer is available. For a routing window $\mathcal { W } .$ , let $\hat { a } _ { \mathcal { W } }$ be the most frequent valid parsed answer in that window, with $\hat { a } _ { \mathcal { W } } = \perp$ if no valid answer appears. We define

$$
C _ { A } ( x ) = \mathcal { k } [ \hat { a } _ { \mathcal { W } } \neq \bot ] \frac { 1 } { | \mathcal { W } | } \sum _ { t \in \mathcal { W } } \mathcal { k } [ \psi _ { A } ( y _ { t } ) = \hat { a } _ { \mathcal { W } } ] .\tag{2}
$$

High $C _ { A }$ means that the same parseable answer repeatedly appears; it does not imply correctness.

visual-format closure. We use visual engagement as a proxy for whether the trajectory attends to image tokens, not whether the answer is true. The proxy aggregates parser validity, visual-attention evidence, prefix-dominance risk, and parsed-answer flip risk:

$$
\begin{array} { r l } & { \mathbf { s } _ { V } ( x ) = \Big ( C _ { \mathrm { f m t } } ( x ) , C _ { \mathrm { v i s } } ( x ) , } \\ & { \qquad R _ { \mathrm { p r e f i x } } ( x ) , R _ { \mathrm { f i p } } ( x ) \Big ) , } \\ & { C _ { V } ( x ) = g _ { V } ( \mathbf { s } _ { V } ( x ) ) \in [ 0 , 1 ] . } \end{array}\tag{3}
$$

Here $C _ { \mathrm { { f m t } } }$ checks answer-format validity, $C _ { \mathrm { v i s } }$ is computed from visual-attention mass over imagetoken positions, and $R _ { \mathrm { p r e f i x } }$ and $R _ { \mathrm { { f l i p } } }$ capture prefix-dominance and answer-instability risks. The fixed aggregation function $g _ { V }$ provides only routing evidence and does not verify whether the answer is visually true.

Representation revision pressure. Representation revision pressure measures whether early and final layer distributions still disagree. Let $p _ { e , t , j }$ and $p _ { f , t , j }$ be token distributions from an early layer and the final layer at position $j .$ For route positions $\Omega _ { t } .$ we compute

$$
\begin{array} { c } { \displaystyle \boldsymbol { J } _ { t } = \frac { 1 } { \left| \Omega _ { t } \right| } \sum _ { j \in \Omega _ { t } } \operatorname { J S } ( p _ { e , t , j } \left| \right| p _ { f , t , j } ) , } \\ { \displaystyle \rho ( x ) = \mathrm { A g g } _ { t \in \mathcal { W } } \boldsymbol { J } _ { t } . } \end{array}\tag{4}
$$

High $\rho ( x )$ indicates that the model is still revising its internal distribution. It is a routing signal, not a hallucination detector or correctness score.

## 3.4 Routing Policy

The controller chooses among three actions, including COMMITROUTE, PRESERVEROUTE, and REA-SONROUTE. Let $K ( x ) \in \{ 0 , 1 \}$ be the fixed CoTsensitive guard, set from the prompt, task format, or disclosed route configuration. Let $G _ { \mathrm { o n } } ( x ) \in \{ 0 , 1 \}$ be the fixed route-on guard for the reported operating point. With thresholds fixed before evaluation, define

$$
\begin{array} { c l l } { { U _ { R } ( x ) \equiv \rho ( x ) \geq \beta _ { \rho } \vee C _ { A } ( x ) < \beta _ { A } , } } \\ { { U _ { E } ( x ) \equiv C _ { A } ( x ) \geq \alpha _ { A } \wedge C _ { V } ( x ) \geq \alpha _ { V } } } \\ { { \wedge \rho ( x ) \leq \alpha _ { \rho } . } } \end{array}\tag{5}
$$

We use $\beta _ { A } \leq \alpha _ { A }$ and $\alpha _ { \rho } \leq \beta _ { \rho } .$ , so intermediate cases fall back to preservation. The reason and commit conditions are as follows.

$$
\begin{array} { l } { { \mathcal { R } ( x ) = K ( x ) G _ { \mathrm { o n } } ( x ) \mathbf { 1 } [ U _ { R } ( x ) ] , } } \\ { { \mathcal { E } ( x ) = \mathbf { 1 } [ U _ { E } ( x ) ] \chi _ { \mathrm { c o m m i t } } ( x ) . } } \end{array}\tag{6}
$$

Here $\chi _ { \mathrm { c o m m i t } } ( x )$ denotes fixed hard guards, such as parser availability, operating-point constraints, or route-specific exclusion rules. Signals already summarized by $C _ { V } ( x )$ are not counted again in χ<sub>commit</sub>.

The routing policy is

$$
\Pi ( x ) = \left\{ { \begin{array} { l l } { \mathsf { r e a s o n } , } & { \mathcal { R } ( x ) = 1 , } \\ { \mathsf { c o m m i t } , } & { \mathcal { R } ( x ) = 0 \wedge \mathcal { E } ( x ) = 1 , } \\ { \mathsf { p r e s e r v e } , } & { \mathrm { o t h e r w i s e } . } \end{array} } \right.\tag{7}
$$

The reason condition is evaluated before commitment because answer stability alone is insufficient in CoT-sensitive interfaces. When $\mathcal { R } ( x ) = 1$ , the controller avoids early commitment and applies the disclosed reasoning-supportive configuration. When neither reason nor commit is justified, the controller preserves the fixed-budget output.

## 3.5 Routing Actions

For REASONROUTE, the controller uses a fixed reasoning-supportive decoding configuration. Let $D _ { \theta } ( x ; B , \lambda )$ denote the parsed output obtained by running the frozen model under budget B and inference-time configuration λ. Then

$$
o _ { r } ( x ) = D _ { \theta } ( x ; B , \lambda _ { \mathrm { o n } } ) ,\tag{8}
$$

where $\lambda _ { \mathrm { o n } }$ is fixed and disclosed for the reported operating point.

For COMMITROUTE, the controller commits to the first locally stable parsed answer when the enabled closure guards fire. If an online commit step is used, let

$$
t _ { c } = \operatorname* { m i n } \{ t \mid \mathcal { E } _ { t } ( x ) = 1 \} ,\tag{9}
$$

where $\mathcal { E } _ { t } ( x )$ applies the commit predicate in $\operatorname { E q }$ . 6 to the local routing window ending at step t. Let $\hat { a } _ { c } = \psi _ { A } ( y _ { t _ { c } } )$ . The committed output is

$$
o _ { c } ( x ) = A _ { c } ( \hat { a } _ { c } , \tau _ { B } , o _ { B } ) ,\tag{10}
$$

where $A _ { c }$ is the fixed commit implementation for the reported row. In the main online setting, $A _ { c }$ fixes the answer field to the first stable parsed answer. Trace composition or source selection variants are marked as offline analyses.

PRESERVEROUTE returns the fixed-budget output unchanged:

$$
o _ { p } ( x ) = o _ { B } ( x ) .\tag{11}
$$

The final output is

$$
o ( x ) = \left\{ { \begin{array} { l l } { o _ { c } ( x ) , } & { \Pi ( x ) = { \mathrm { c o m m i t } } , } \\ { o _ { p } ( x ) , } & { \Pi ( x ) = { \mathrm { p r e s e r v e } } , } \\ { o _ { r } ( x ) , } & { \Pi ( x ) = { \mathrm { r e a s o n } } . } \end{array} } \right.\tag{12}
$$

## 3.6 Routed Decoding Algorithm

Algorithm 1 summarizes the routing procedure. The algorithm computes trajectory evidence, applies the route conditions in Eq. 6, and returns the corresponding routed output. The routine ObtainRoutingTrace denotes the trace used for routing; rows that require cached candidate traces are marked separately as offline analyses and are not used for latency claims.

## 4 Experiments

## 4.1 Experimental Setup

We evaluate LLaDA-V under answer-focused, mixed-reasoning, and CoT or support-sensitive settings. MME (Fu et al., 2025), MMMU (Yue et al., 2024), and MMStar (Chen et al., 2024) use compact final-answer interfaces, but differ in reasoning demands. MME contains many compact visual diagnostic questions, while MMMU and MMStar include more expert-level, fine-grained, or vision-indispensable cases. ScienceQA-IMG (Lu et al., 2022), A-OKVQA (An et al., 2024), and MME-CoT (Jiang et al., 2025) are used as support-sensitive settings. ScienceQA-IMG provides annotated explanations, A-OKVQA requires image-grounded commonsense and world knowledge, and MME-CoT directly evaluates multimodal chain-of-thought quality.

![](images/e1bd21c5952772cbf563d2f97cadc52eacd7c4abfffa89ed6bb8cd46fdb28320.jpg)

![](images/561f91687cc68e08092648e6b81753b26b7a13a6aa2fddd62caa08ffadc94aa9.jpg)

![](images/efbeb12425ab5d484e0658599c52f1fffa63df4b2990175f0d9ee853922d1a2c.jpg)  
Figure 3: Representative answer-token trajectory patterns that motivate reasoning-need routing. Each panel shows top-1 token changes, decoded positions, and answer-token correctness across diffusion decoding steps. (a) An early closure case where the correct answer appears early but later refinement drifts to an incorrect answer, motivating early commitment. (b) A late-converging case where the answer becomes reliable only after sufficient refinement, motivating baseline preservation when early commitment is not justified. (c) A reasoning-heavy unstable case where the answer trajectory remains unsettled under simple length control, motivating reasoning-supportive decoding rather than a universal short or long budget.

Algorithm 1 Reasoning-Need Routed Decoding   
Control   
Require: input $\boldsymbol { x } = ( x _ { v } , x _ { q } , c , I )$ , budget B, frozen model   
$D _ { \theta }$   
Ensure: routed output o   
1: τ<sub>B</sub>, o<sub>B</sub> ← ObtainRoutingTrace (x, B)   
2: $C _ { A } \gets \mathrm { A }$ nswerClosure(τ<sub>B</sub>)   
3: C<sub>V</sub> ← GroundingFormatClosure(τ<sub>B</sub>, x)   
4: ρ ← RevisionPressure(τ<sub>B</sub>)   
5: K ← CoTSensitiveGuard(x)   
6: $G _ { \mathrm { o n } } $ RouteOnGuard(x, τ<sub>B</sub>)   
7: χ<sub>commit</sub> ← CommitGuard(x, τ<sub>B</sub>)   
8: Compute $U _ { R } , U _ { E } , \mathcal { R } , \mathcal { E }$ using Eqs. 5–6   
9: $\mathbf { i f } \ \mathcal { R } ( x ) = 1$ then   
10: return $D _ { \theta } ( x ; B , \lambda _ { \mathrm { o n } } )$   
11: else i $\begin{array} { r } { \left[ \mathcal { E } ( x ) \right. = \mathrm { \ i } } \end{array}$ then   
12: $t _ { c } \gets$ FirstCommitStep(τ<sub>B</sub>)   
13: $\hat { a } _ { c } \gets \psi _ { A } ( y _ { t _ { c } } )$   
14: return $A _ { c } \big ( \hat { a } _ { c } , \tau _ { B } , o _ { B } \big )$   
15: else   
16: return $O _ { B }$   
17: end if

All methods use the same frozen LLaDA-V model and the original benchmark parser. We compare fixed-budget baselines with different generation budgets, including Len2, Len32, Len64, and Len128, where Len128 is the default fixed-budget reference. We also compare visual and contrastive decoding controls, adaptive-budget controls, and our routed controller when benchmark-matched artifacts are available. Final-answer accuracy is the primary metric for benchmark comparison. For auxiliary analysis, we report decoding budget statistics in Appendix and CoT rationale diagnostics on the ScienceQA-IMG active-control subset. These diagnostics use reference-based text metrics and GPT-5.5 judge scores, but are not treated as goldstandard human faithfulness evidence.

## 4.2 Controlled Results Across Regimes

Table 1 summarizes controlled comparisons across answer-focused and support-sensitive settings. The results support three observations.

First, fixed long decoding is not consistently optimal in answer-focused settings. On MME, routed control reaches 79.96, improving over Len128 by 6.55 points and slightly exceeding Len2 at 79.56. On MMMU, it reaches 49.44, improving over Len128 by 7.00 points and exceeding Len2 at 48.89. These gains suggest that routed control is not simply applying a global short budget. Instead, it commits when the trajectory is low-risk and preserves the baseline when early commitment is not justified. On MMStar, routed control reaches 50.77, slightly above Len128 at 49.90, which further supports sample-wise action selection in compact-answer multimodal settings.

Second, final-answer closure can appear even in rationale-bearing benchmarks, but this should not be interpreted as absence of reasoning. On ScienceQA-IMG, routed control reaches 88.60, improving over Len128 by 13.29 points and over Len2 by 4.32 points. Because ScienceQA-IMG provides explanations but Table 1 reports final-answer accuracy, this result shows that final-answer scoring can exhibit early answer closure within a CoT-capable benchmark. It does not imply that the underlying task lacks reasoning structure.

![](images/6c12a9b4103ee45708343c185386a87f763bcf7756579496c0e77f150210e3fc.jpg)

<table><tr><td></td><td colspan="3">Answer-Focused Evaluation</td><td colspan="3">CoT-Sensitive Evaluation</td></tr><tr><td>Method</td><td>MME</td><td>MMMU</td><td>MMStar</td><td>SQA- IMG</td><td>A-OKVQA</td><td>MME- CoT</td></tr><tr><td colspan="7">Global budget and task baselines</td></tr><tr><td>Len2</td><td>79.56</td><td>48.89</td><td>46.88</td><td>84.28</td><td>78.75</td><td>48.72</td></tr><tr><td>Len32</td><td>73.81</td><td>28.78</td><td>25.00</td><td>53.00</td><td>82.50</td><td>49.57</td></tr><tr><td>Len64</td><td>74.60</td><td>36.11</td><td>17.19</td><td>76.65</td><td>82.50</td><td>49.57</td></tr><tr><td>Len128</td><td>73.41</td><td>42.44</td><td>49.90</td><td>75.31</td><td>83.75</td><td>49.57</td></tr><tr><td colspan="7">Visual and contrastive controls</td></tr><tr><td>VCD</td><td>73.41</td><td>47.33</td><td>32.81</td><td>80.80</td><td>83.75</td><td>49.29</td></tr><tr><td>SID</td><td>68.65</td><td>47.78</td><td>34.38</td><td>81.00</td><td>82.50</td><td>50.43</td></tr><tr><td>MEMVR</td><td></td><td>48.56</td><td>35.94</td><td></td><td></td><td>19.37</td></tr><tr><td colspan="7">Adaptive-budget controls</td></tr><tr><td>Fast-dLLM</td><td>72.33</td><td>42.89</td><td>34.38</td><td>75.61</td><td>83.75</td><td>49.57</td></tr><tr><td>AdaBlock</td><td>71.40</td><td>42.33</td><td>35.94</td><td>74.81</td><td>80.00</td><td>49.29</td></tr><tr><td>DAEDAL-lite</td><td>72.03</td><td>42.44</td><td>32.81</td><td>75.90</td><td>83.75</td><td>49.57</td></tr><tr><td>dLLM-Var</td><td>69.29</td><td>40.89</td><td>20.31</td><td>70.20</td><td>85.00</td><td>48.15</td></tr><tr><td colspan="7">Reasoning-need routed control</td></tr><tr><td>Ours</td><td>79.96</td><td>49.44</td><td>50.77</td><td>88.60</td><td>88.75</td><td>52.42</td></tr></table>

Table 1: Controlled results across answer-focused and CoT-sensitive evaluation settings. Scores are accuracies. Missing entries indicate that no benchmark-matched artifact was available.

Third, pure short decoding is not CoT-safe. On MME-CoT, Len2 falls to 48.72, below Len128 at 49.57, while routed control reaches 52.42, suggesting that premature commitment can remove useful reasoning structure. On targeted A-OKVQA closure, routed control reaches 88.75, outperforming Len128 at 83.75 and the strongest non-routed baseline at 85.00. These patterns support trajectoryaware routing over a universal length policy: short decoding helps stable answers, long decoding preserves reasoning capacity, and routing selects the safer action.

## 4.3 Length-Policy Stress Test

A central question is whether the improvement comes only from making outputs shorter. The results do not support this explanation. Table 2 compares representative settings where short, long, and routed decoding behave differently. On MME, Len2 is already strong, suggesting that many compact visual questions reach answer closure early. Routed control still slightly improves over Len2 while avoiding a global short policy. On ScienceQA-IMG, routed control improves over both Len2 and Len128, indicating that final-answer gains are not explained by uniformly shortening the output. On MME-CoT, Len2 is worse than Len128, while routed control improves over both.

![](images/f8b9e39996bc6729e27d47a9aff7b287b101059feeddba77f308554793aad630.jpg)

![](images/311ff11cf10b87cad8da0302e5c17e1d2643dded52acefc66292cdce4e5258cc.jpg)  
Figure 4: Auxiliary CoT rationale-quality diagnostics on the ScienceQA-IMG active-control subset. The axes include BLEU-4, ROUGE-L, SBERT similarity, factual support, and answer support. The two judge dimensions correspond to Reference-Grounded Factuality and Rationale-Answer Alignment. The left panel compares visual and contrastive controls, while the right panel compares adaptive-budget controls. Both panels use the same zero-based per-axis normalization for visualization. These diagnostics are auxiliary and should not be interpreted as evidence of causal rationale faithfulness.

This shows that early commitment is unsafe when reasoning support is required.

Table 3 gives a complementary route-level diagnostic. If routed control simply behaved like short decoding, most examples would be routed to commitment across all settings. Instead, the allocation changes with the evaluation interface and trajectory state. MME is dominated by commitment, which matches its compact answer-focused interface. In

<table><tr><td>Setting</td><td>Len2</td><td>Len128</td><td>Ours</td><td>Steps</td></tr><tr><td>MME</td><td>79.56</td><td>73.41</td><td>79.96</td><td>128.00</td></tr><tr><td>SQA-IMG</td><td>84.28</td><td>75.31</td><td>88.60</td><td>118.09</td></tr><tr><td>MME-CoT</td><td>48.72</td><td>49.57</td><td>52.42</td><td>102.09</td></tr></table>

Table 2: Length-policy stress test. Scores are accuracies, and Steps reports the average decoding steps of routed control. The pattern shows that routed control is not equivalent to uniformly shortening the output.
<table><tr><td>Setting</td><td>Commit</td><td>Preserve</td><td>Reason</td></tr><tr><td>MME</td><td>75.99</td><td>24.01</td><td>0.00</td></tr><tr><td>SQA-IMG†</td><td>0.00</td><td>88.75</td><td>11.25</td></tr><tr><td>MME-CoT</td><td>0.00</td><td>71.79</td><td>28.21</td></tr></table>

Table 3: Diagnostic route distribution across evaluation settings. Values are percentages. <sup>†</sup>ScienceQA-IMG uses the CoT diagnostic route interface.

ScienceQA-IMG and MME-CoT diagnostic CoT settings, premature commitment is suppressed, and the controller mainly preserves the baseline or applies reasoning-supportive decoding.

Together, the two diagnostics clarify the role of routing. Short decoding helps when the answer is already closed, but it can remove useful intermediate structure in CoT-sensitive settings. Long decoding preserves reasoning capacity, but it can continue refining samples whose answer has already stabilized. Routed control improves robustness by selecting the safer action under the observed trajectory, rather than by committing to one global length policy.

## 4.4 Auxiliary CoT Diagnostics on ScienceQA-IMG

Final-answer accuracy does not show whether a generated reasoning trace resembles the reference explanation or supports the predicted answer. We therefore add an auxiliary CoT diagnostic on the ScienceQA-IMG active-control subset. For automatic evaluation, we compare each generated CoT trace with the ScienceQA reference explanation and report BLEU-4, ROUGE-L, and SBERT similarity. BLEU-1 is reported in Appendix as a supplementary overlap metric. For judge-based diagnostics, GPT-5.5 scores two text-only support dimensions on a 0–10 scale, factual support and answer support. The judge sees the question, reference explanation, predicted answer, and generated reasoning trace, but these scores are used only as auxiliary diagnostics.

Figure 4 and Table 6 show that routed control improves reference-based explanation metrics over the fixed long-budget baseline. BLEU-4 increases from 0.105 to 0.124, ROUGE-L from 0.268 to 0.296, and SBERT similarity from 0.697 to 0.720. The paired tests are significant for all three metrics, with $p ~ = ~ 1 . 0 3 { \times } 1 0 ^ { - 5 }$ for BLEU-4, $p \ =$ $8 . 5 3 \times 1 0 ^ { - 6 }$ for ROUGE-L, and $p = 3 . 4 1 \times 1 0 ^ { - 3 }$ for SBERT similarity. The judge scores provide complementary evidence that routed control improves answer support while remaining competitive on factual support.

These results are auxiliary rather than definitive evidence of reasoning faithfulness. They show that routed control can improve actively controlled reasoning traces on a rationale-bearing benchmark, but they do not establish that the generated rationales faithfully explain the model’s predictions.

## 5 Conclusion

Diffusion VLMs expose intermediate answer trajectories before the final output is fixed. This makes fixed-budget decoding a poor fit for questions with different closure times. Some examples already contain a stable, visually supported answer and can be harmed by continued refinement. Others require additional refinement, so early commitment can remove a useful intermediate structure.

We study this problem as reasoning-budget mismatch and address it with a parameter-frozen routed controller. The controller chooses among early commitment, fixed-budget preservation, and reasoning-supportive decoding from trajectory evidence rather than from a global length rule. Across answer-focused, mixed-reasoning, and CoTsensitive settings, this routing view avoids the main failure modes of both pure short decoding and fixed long decoding. The broader implication is that diffusion VLM decoding should be treated as trajectory-aware control, not as universal generation-length selection.

## Limitations

This study has several limitations. First, the controller is evaluated on LLaDA-V, and additional diffusion VLMs are needed to establish modellevel generality. Second, the routing signals are label-free proxies rather than correctness estimators. In particular, visual-attention evidence measures image-token engagement, not whether the answer is visually true. Third, some operating points use benchmark-specific guards or reasoningsupportive configurations; although these are fixed before evaluation, future work should study more unified threshold selection and calibration. Fourth, our CoT diagnostics combine reference-based automatic metrics with judge-based support scores, which should not be interpreted as causal evidence of rationale faithfulness. Finally, while routed control improves robustness, it is not primarily an acceleration method in the reported setting, since some routes retain long decoding budgets to preserve reasoning support.

## Acknowledgments

This study was supported by the National Natural Science Foundation of China (T2422012); the National Key Research and Development Program of China (2023YFC2415400); the Guangdong Basic and Applied Basic Research (2024B1515020088); the Shenzhen Science and Technology Program (ZDYJ20251211121037006); the High Level of Special Funds (G030230001, G03034K003); the Guangdong S&T Program (2025B1111080001); the SUSTech Fang Keng Faculty Award.

## References

Wenbin An, Feng Tian, Jiahao Nie, Wenkai Shi, Haonan Lin, Yan Chen, QianYing Wang, Yaqiang Wu, Guang Dai, and Ping Chen. 2024. Knowledge acquisition disentanglement for knowledge-based visual question answering with large language models. arXiv preprint arXiv:2407.15346.

Sriram Balasubramanian, Samyadeep Basu, and Soheil Feizi. 2025. A closer look at bias and chain-ofthought faithfulness of large (vision) language models. In EMNLP (Findings), pages 13406–13439.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. 2024. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087.

Jiahao Cheng, Tiancheng Su, Jia Yuan, Guoxiu He, Jiawei Liu, Xinqi Tao, Jingwen Xie, and Huaxia Li. 2025. Chain-of-thought prompting obscures hallucination cues in large language models: An empirical evaluation. arXiv preprint arXiv:2506.17088.

Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James R Glass, and Pengcheng He. 2024. Dola: Decoding by contrasting layers improves factuality in large language models. In International Conference on Learning Representations, volume 2024, pages 54158–54183.

Vardhan Dongre, Chi Gui, Shubham Garg, Hooshang Nayyeri, Gokhan Tur, Dilek Hakkani-Tür, and Vikram S Adve. 2025. Mirage: A benchmark for multimodal information-seeking and reasoning in agricultural expert-guided conversations. arXiv preprint arXiv:2506.20100.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji, Caifeng Shan, and Ran He. 2025. Mme: A comprehensive evaluation benchmark for multimodal large language models. Advances in Neural Information Processing Systems, 38.

Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. 2024. Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13418–13427. IEEE.

Fushuo Huo, Wenchao Xu, Zhong Zhang, Haozhao Wang, Zhicheng Chen, and Peilin Zhao. 2025. Selfintrospective decoding: Alleviating hallucinations for large vision-language models. In International Conference on Learning Representations, volume 2025, pages 24272–24295.

Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanwei Li, Yu Qi, Xinyan Chen, Liuhui Wang, Jianhan Jin, Claire Guo, Shen Yan, Bo Zhang, Chaoyou Fu, Peng Gao, and Hongsheng Li. 2025. MME-cot: Benchmarking chain-of-thought in large multimodal models for reasoning quality, robustness, and efficiency. In International Conference on Machine Learning.

Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. 2024. Mitigating object hallucinations in large visionlanguage models through visual contrastive decoding. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13872– 13882. IEEE.

Jinsong Li, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jiaqi Wang, and Dahua Lin. 2026a. Beyond fixed: Training-free variable-length denoising for diffusion large language models. In International Conference on Learning Representations, volume 2026, pages 91715–91731.

Pengxiang Li, Yefan Zhou, Dilxat Muhtar, Lu Yin, Shilin Yan, Li Shen, Yi Liang, Soroush Vosoughi, and Shiwei Liu. 2026b. Diffusion language model knows the answer before it decodes. In The Fourteenth International Conference on Learning Representations.

Guanxi Lu, Hao Chen, Yuto Karashima, Zhican Wang, Daichi Fujiki, and Hongxiang Fan. 2026. Adablockdllm: Semantic-aware diffusion llm inference via adaptive block size. In International Conference

on Learning Representations, volume 2026, pages 140018–140033.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in neural information processing systems, 35:2507–2521.

Yunze Man, De-An Huang, Guilin Liu, Shiwei Sheng, Shilong Liu, Liang-Yan Gui, Jan Kautz, Yu-Xiong Wang, and Zhiding Yu. 2025. Argus: Vision-centric reasoning with grounded chain-of-thought. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14268–14280. IEEE.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. 2026. Large language diffusion models. Advances in Neural Information Processing Systems, 38:50608–50646.

Zayne Sprague, Fangcong Yin, Juan Rodriguez, Dongwei Jiang, Manya Wadhwa, Prasann Singhal, Xinyu Zhao, Xi Ye, Kyle Mahowald, and Greg Durrett. 2025. To cot or not to cot? chain-of-thought helps mainly on math and symbolic reasoning. In International Conference on Learning Representations, volume 2025, pages 94118–94162.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. 2026. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. In International Conference on Learning Representations, volume 2026, pages 57027–57051.

Zhongxing Xu, Chengzhi Liu, Qingyue Wei, Juncheng Wu, James Zou, Xin Wang, Yuyin Zhou, and Sheng Liu. 2025. More thinking, less seeing? assessing amplified hallucination in multimodal reasoning models. Advances in Neural Information Processing Systems, 38:82878–82905.

Zhongxing Xu, Zhonghua Wang, Zhe Qian, Dachuan Shi, Feilong Tang, Ming Hu, Shiyan Su, Xiaocheng Zou, Wei Feng, Dwarikanath Mahapatra, Yifan Peng, Minquan Lin, and Zongyuan Ge. 2026. Thinking in uncertainty: Mitigating hallucinations in mlrms with latent entropy-aware decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11064–11075.

Yicun Yang, Cong Wang, Shaobo Wang, Zichen Wen, Biqing Qi, Hanlin Xu, and Linfeng Zhang. 2025. Diffusion llm with native variable generation lengths: Let [eos] lead the way. arXiv preprint arXiv:2510.24605.

Zebin You, Shen Nie, Xiaolu Zhang, JUN ZHOU, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. 2026. Llada-v: Large language diffusion models with visual instruction tuning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10093–10105.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, et al. 2024. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567.

Xin Zou, Yizhou Wang, Yibo Yan, Yuanhuiyi Lyu, Kening Zheng, Sirui Huang, Junkai Chen, Peijie Jiang, Jia Liu, Chang Tang, and Xuming Hu. 2025. Look twice before you answer: Memory-space visual retracing for hallucination mitigation in multimodal large language models. In Forty-second International Conference on Machine Learning.

## A Appendix

## A.1 Available Diagnostic Comparisons

Table 5 reports action-level diagnostic comparisons for settings where corresponding artifacts are available. These rows are not intended as a full component ablation. They compare the full routed setting with available preserve-only, commit-only, separated-source, and guarded-routing variants.

The comparisons show two patterns. First, the full router improves over preserve-only baselines across all reported settings. Second, the full router is not equivalent to a commit-only policy. Commit proxy is competitive on answer-focused settings, but it underperforms the full router and drops on MME-CoT, where premature commitment can remove reasoning support. The ScienceQA-IMG CoT separated-source setting reaches the same answer accuracy as the full routed setting, suggesting that answer accuracy alone does not capture the effect of source selection in this diagnostic protocol. On MME-CoT, the guarded route improves over the unguarded route variant, indicating that the final guard helps reduce harmful route-on cases. We treat these rows as diagnostic action comparisons, while full component isolation is left to future controlled ablations.

## A.2 Routing Guards and Operating Points

K(x) is an interface-level CoT-sensitive guard. It is enabled when the benchmark prompt requests a rationale, the evaluation interface scores reasoning support, or a disclosed route configuration marks the input as CoT-sensitive. $G _ { \mathrm { o n } } ( x )$ is the fixed route-on guard for the reported operating point. χ<sub>commit</sub>(x) contains hard exclusions such as parser unavailability, operating-point constraints, or routespecific exclusion rules.

For MME-CoT, $\lambda _ { \mathrm { o n } }$ corresponds to the answerfirst CoT prompt with jsd\_adaptive\_throttle and fixed route-on thresholds. When Route v3 is reported, it should be interpreted as a disclosed MME-CoT operating point rather than a benchmark-independent routing rule. Rows that use cached fixed-budget and route-on candidate outputs are marked as offline trace-composition analyses and are not used for online latency claims.

## A.3 Budget and Efficiency Analysis

Table 4 reports the budget controls. Fixed128 / Long128 decoding is computationally expensive across MME, ScienceQA-IMG, MME-CoT, and

<table><tr><td>Benchmark</td><td>Method Steps</td><td>Time</td><td>Len.</td></tr><tr><td colspan="4">Answer-closed / mixed settings</td></tr><tr><td>MME</td><td>Len128 128.00</td><td>22.53s</td><td>47.0</td></tr><tr><td>MME MME</td><td>Len64 64.00 Len32</td><td>10.60s 4.85s</td><td>23.43 11.40</td></tr><tr><td>MME Len2</td><td>32.00 2.00</td><td>0.28s 22.77s</td><td>1.0 40.01</td></tr><tr><td colspan="4">MME Ours 128.00 CoT-sensitive setting</td></tr><tr><td>MME-CoT MME-CoT MME-CoT</td><td>Len128 Len64 Len32</td><td>95.91 25.52s 40.05 10.41s 28.09 4.97s</td><td>89.42 45.05 23.55</td></tr><tr><td>MME-CoT MME-CoT ScienceQA-IMG</td><td>Len2 Ours Len128</td><td>2.00 0.35s 102.09 25.67s 24.6s</td><td>1.46 89.26 99.94</td></tr><tr><td>ScienceQA-IMG ScienceQA-IMG ScienceQA-IMG ScienceQA-IMG</td><td>Len64 Len32 Len2 Ours 118.09</td><td>101.47 64.00 32.00 2.00 0.31s</td><td>11.10s 23.57 5.64s 11.15 1.00</td></tr></table>

Table 4: Budget and efficiency controls. The table reports decoding budget rather than answer accuracy. Entries marked with are approximate estimates from measured budget trends when full audit latency or output length was unavailable. Len2 is inexpensive for answerclosed settings, whereas the MME-CoT route retains a long reasoning budget, indicating that its gain is not an output-shortening effect.

MMMU. Short2 is much cheaper and is often strong in answer-closed settings, reaching 79.56 on MME, 84.28 on ScienceQA-IMG, and 48.89 on MMMU with a much smaller decoding budget. This confirms that many answer-closed examples do not need the full fixed refinement path.

However, Short2 is not CoT-safe: on MME-CoT it reaches 48.72, below the Fixed128 CoT baseline at 49.57. Intermediate budgets also do not consistently solve the mismatch. Short32 and Fixed64 are below Short2 on MME and ScienceQA-IMG, and they do not improve over Fixed128 on MME-CoT. Route v3’s MME-CoT gain is not explained by shorter outputs alone because its average steps and latency are comparable to the long CoT baseline. The method is therefore not simply reducing length; it selectively allocates or structures reasoning. These results motivate selective budget allocation rather than uniform budget reduction.

## A.4 Shared Decoding Settings

All reported rows use the frozen LLaDA-V model with the benchmark image input and task prompt assembled by the evaluation scripts. We do not update model weights or train a verifier. Unless a method-specific control requires a different field, decoding uses deterministic sampling with temperature 0, low-confidence remasking, and the benchmark parser used by the corresponding evaluation.

<table><tr><td>Variant</td><td>MME</td><td>MMMU</td><td>SQA-CoT</td><td>MME-CoT</td></tr><tr><td>Full router</td><td>79.96</td><td>49.44</td><td>88.60</td><td>52.42</td></tr><tr><td>Preserve only</td><td>73.41</td><td>42.44</td><td>84.18</td><td>49.57</td></tr><tr><td>Commit proxy</td><td>79.56</td><td>48.89</td><td>NA</td><td>48.72</td></tr><tr><td>Separated source</td><td>NA</td><td>NA</td><td>88.60</td><td>NA</td></tr><tr><td>Unguarded route</td><td>NA</td><td>NA</td><td>NA</td><td>51.57</td></tr></table>

Table 5: Available diagnostic comparisons for routed decoding. Entries are accuracies. Preserve only denotes the fixed-budget or same-setting joint CoT baseline. Commit proxy uses the shortest benchmark-matched answer proxy. Separated source is reported only for ScienceQA-IMG CoT. Unguarded route is reported only for MME-CoT.
<table><tr><td>Metric</td><td>Baseline</td><td>Method</td><td>Delta</td><td>p</td></tr><tr><td>BLEU-1</td><td>0.269</td><td>0.280</td><td>+0.011</td><td> $1 . 0 9 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>BLEU-4</td><td>0.105</td><td>0.124</td><td>+0.018</td><td> $1 . 0 3 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>ROUGE-L</td><td>0.268</td><td>0.296</td><td>+0.028</td><td> $8 . 5 3 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>SBERT sim</td><td>0.697</td><td>0.720</td><td>+0.023</td><td> $3 . 4 1 \times 1 0 ^ { - 3 }$ </td></tr></table>

Table 6: Automatic ScienceQA-IMG CoT explanation metrics on the active-control subset. BLEU-4, ROUGE-L, and SBERT similarity are used in Figure 4; BLEU-1 is reported as a supplementary overlap metric.

Fixed-budget rows keep the same model, prompt, and parser while changing the diffusion budget. Len2, Len32, Len64, and Len128 use generation lengths matched to the corresponding number of denoising steps. Adaptive-budget controls keep the same model and parser, while using their methodspecific decoding rules.

## A.5 Routing Configuration Disclosure

Table 8 summarizes the routing configuration used in each evaluation setting. All routing rules are fixed before evaluation and do not use gold labels. For answer-focused settings, the controller mainly decides whether the trajectory is safe to commit or should be preserved. For CoT-sensitive settings, the reason route can be enabled when the interface requires support-bearing reasoning and the observed trajectory remains unresolved.

Rows that use cached candidate outputs are used only for diagnostic composition and are not used for online latency claims. This preserves the same benchmark prompt, parser, model weights, and branch-level decoding configuration, while separating diagnostic route analysis from online efficiency evaluation.

<table><tr><td>Field</td><td>Setting</td></tr><tr><td>Model</td><td>Frozen LLaDA-V</td></tr><tr><td>Temperature</td><td>0.0</td></tr><tr><td>Remasking</td><td>Low confidence</td></tr><tr><td>Default budget</td><td>128 steps, 128 generation length</td></tr><tr><td>Short budgets</td><td>2, 32, and 64 steps</td></tr><tr><td>Parser</td><td>Original benchmark parser</td></tr><tr><td>Trace collection</td><td>Enabled for routed settings</td></tr></table>

Table 7: Shared decoding settings used in the reported experiments.

<table><tr><td>Setting</td><td>Commit</td><td>Preserve</td><td>Reason</td></tr><tr><td>ScienceQA-IMG</td><td>enabled</td><td>enabled</td><td>setting dependent</td></tr><tr><td>MME</td><td>enabled</td><td>enabled</td><td>disabled</td></tr><tr><td>MMMU</td><td>enabled</td><td>enabled</td><td>enabled</td></tr><tr><td>MME-CoT</td><td>disabled</td><td>enabled</td><td>enabled</td></tr></table>

Table 8: Routing actions enabled in each evaluation setting. ScienceQA-IMG uses a CoT diagnostic route interface for the rationale-quality analysis.

## A.6 LLM-Judge Prompt for ScienceQA-IMG CoT Diagnostics

We use a text-only LLM judge as an auxiliary diagnostic for ScienceQA-IMG CoT outputs. The judge does not inspect the image and does not see method names. It receives the question, answer choices, reference explanation, predicted answer, and generated rationale. The scores are used only to assess local support properties of the generated rationale and are not treated as evidence of causal rationale faithfulness.

You are evaluating a generated rationale for a ScienceQA-IMG multiple-choice example. This is a text-only evaluation. You cannot inspect the image. Use only the question, answer choices, reference explanation, predicted answer, and generated rationale. Do not infer unseen image contents. Do not use method names. Return JSON only.

Scoring guide: 10 = accurate, specific, coherent, and well   
supported. 8 = mostly supported with minor   
omissions or harmless imprecision. 6 = partially   
supported but generic or incomplete. 4 = weak   
support, notable unsupported claims, or loose   
connection. 2 = mostly unsupported, contradictory,   
repetitive, or off-topic. 0 = empty, invalid,   
nonsensical, or supports a different answer.   
Question: {question}   
Answer choices: {choices}   
Reference explanation: {reference\_explanation}   
Predicted answer: {predicted\_answer}   
Generated rationale: {generated\_rationale}   
Return exactly:   
{   
"factual\_support": integer from 0 to 10,   
"answer\_support": integer from 0 to 10,   
"short\_reason": "one short sentence"   
}

Judge calls use deterministic decoding with temperature 0. The judge outputs are parsed as JSON. We cache judge results by example and method to avoid repeated calls. We do not request or store hidden chain-of-thought from the judge.

## A.7 Illustrative ScienceQA-IMG non-preserve CoT examples

Illustrative ScienceQA-IMG non-preserve CoT examples (Case1–Case3). Each panel compares Len128 with Ours on the same example id and reports automatic explanation metrics and text-only judge scores. These diagnostics support rationalequality analysis but do not prove faithfulness.

## A.8 Illustrative MME answer-token drift cases under fixed 128-step decoding.

Illustrative MME answer-token drift cases (Case4 and Case 5) under fixed 128-step decoding. Green marks the correct answer-token interval, red marks the drifted answer-token interval, blue marks decoded positions, and orange marks top-1 changes. These examples diagnose answer-slot drift and do not constitute additional benchmark results.

## scienceqa\_full\_01985: ScienceQA-IMG CoT Example

Selected from the non-preserve subset where Ours improves over Len128; illustrative rationale-quality diagnostic.

![](images/2e8f92e1b92f5f8e45eee8d4d6977cd81d6208fa3ca2d73b39153962e21fc983.jpg)

## Question

Which month is the hottest on average in Adelaide? A. June, July, and August B. April and May C. January and February

Choices / Gold

Gold: C (January and February)

Reference explanation To describe the average temperature trends in Adelaide, look at the graph. Choice "Feb" is incorrect. Choice "Apr" is incorrect. Choice "May" is incorrect. Choice "Jun" is incorrect. Choice "Jul" is incorrect. Choice "Aug" is incorrect. Choice "Jan" is incorrect. The average temperatures in January and February are just above 20°C. These months have the highest average temperatures of all of the months. So, they are hotter than the other months.

![](images/69a128f07f9be46c0d93a73fd0f9c0ccabbc823ffa28fa7c1d62fff467b6140f.jpg)  
Figure 5: Case1

![](images/079246ee19b173a020dd53fd9975a6b47b887eada5672e416680c544f9cb5314.jpg)

# scienceqa\_full\_03285: ScienceQA-IMG CoT Example

Selected from the non-preserve subset where Ours improves over Len128; illustrative rationale-quality diagnostic.

![](images/bb07124056e9b422b321b2851500186f365cd85575f65a7f32df740ecbdadf91.jpg)

Question   
Which month has the lowest average precipitation in London? A. November B. July C. February   
Choices / Gold   
A. November B. July C. February   
Gold: B (July)   
Reference explanation   
To describe the average precipitation trends in London, look at the graph. Choice "Feb" is incorrect. Choice "Jul" is incorrect. Choice "Nov" is incorrect. July has an average monthly precipitation of about 45 millimeters. This is lower than in any other month. So, July has the lowest average precipitation.

## scienceqa\_full\_01410: ScienceQA-IMG CoT Example

Selected from the non-preserve subset where Ours improves over Len128; illustrative rationale-quality diagnostic.

Question

Does this passage describe the weather or the climate? A. climate B. weather

![](images/356f6d4d6d59674115e5465cf83e0b66b7b007cbb2de6086ef22ba7e93c33958.jpg)

Choices / Gold

A. climate B. weather

Gold: B (weather)

Reference explanation

Read the passage carefully. Madison, the capital of Wisconsin, is located in the southern part of the state. For two weeks in December, the temperature never rose above 20°F. The underlined part of the passage tells you what the temperature in Madison was like in December. This passage describes the atmosphere at a certain place and time. So, this passage describes the weather.

## Len128 Baseline

Pred: B | correct=True

BLEU-4 4.3 ROUGE-L 21.7 SBERT 39.1

Factual 25 Answer 50 Halluc. safety 0 Avg 25

To determine whether the passage describes the weather or the climate, I will analyze the visual cues in the image and compare them to the options. The image shows a large building with a dome, surrounded by snow-covered trees and ground The sky is overcast, suggesting a cold, snowy environment. The presence of snow and the overcast sky indicate a specific condition at a particular time, which is characteristic of weather. Climate, on the other hand, refers to long-term averages and patterns. Since the image depicts a specific moment with snow and overcast skies, it describes the weather. B

Judge note: It selects the correct answer, but the explanation invents image details instead of using the passage-based evidence in the reference.

##

Pred: B | correct=True BLEU-4 21.8 ROUGE-L 34.8 SBERT 82.9 Factual 100 Answer 100 Halluc. safety 100 Avg 100

Evidence 1: The passage mentions that the temperature in Madison, Wisconsin, never rose above 20°F for two weeks in December. This is a specific description of the atmosphere at a particular time and place. Evidence 2: The passage describes a short-term event, which is characteristic of weather.

Judge note: It correctly identifies the passage as describing short-term atmospheric conditions at a specific place and time.

## mme\_0881: Answer-Token Drift Case

Illustrative appendix example; not an additional benchmark result.

![](images/ef4555ba0e053fe49bb3ff8846f047f7395e27e9994b3949d8b9c68b138ec833.jpg)

## Question

Are there two people appear in this image? Please answer yes or no.

Gold: No

Fixed128 final: Yes (correct=False)

First wrong-after-correct step: 71; stable wrong from step: 71.

Read the heatmap as answer-slot behavior over diffusion steps: green denotes the correct answer token interval, red denotes the drifted answer-token interval, and blue marks positions decoded at each step.

The trace observes mannequins but then treats the presence of two mannequin figures as support for two people, creating an answer-type mismatch. Answer drift: correct answer persists through step 70, then the final answer stabilizes as Yes from step 71.

Attention diagnostic: Final answer token \`Yesattends more to generated prefix than visual tokens on average (0.468 vs 0.173); late-layer prefix/visual ratio = 1.08x.

## mme\_0061: Answer-Token Drift Case

## Question

Illustrative appendix example; not an additional benchmark result.

![](images/d8a65dbf52bfd7b78c64364c6c86b5a24487713b34ba3a75ad11b570b2db7b9d.jpg)

Is this artwork created by morel, jean-baptiste? Please answer yes or no.

First wrong-after-correct step: 69; stable wrong from step: 69.

Gold: No

![](images/f85c5fd2fd956aa73c9756dde322a62438eb5eb0a36635b1543b956d256c0944.jpg)

Read the heatmap as answer-slot behavior over

![](images/272d8243c6f2cadeaa0dc0f8a87398c447bd8758c11f2b4558e073f00a5d001b.jpg)

Fixed128 final: Yes (correct=False)

diffusion steps: green denotes the correct answer token interval, red denotes the drifted answer-token interval. and blue marks positions decoded at each step and blue marks positions decoded at each step.

## Diagnostic error mechanism

The trace over-extends from generic art-style language to an unsupported attribution to Jean-Baptiste Morel, adding painter-biography content not grounded by the image. Answer drift: correct answer persists through step 68, then the final answer stabilizes as Yes from step 69.

Attention diagnostic: Final answer token \`Yesattends more to generated prefix than visual tokens on average (0.550 vs 0.125); late-layer prefix/visual ratio = 2.68x.

Figure 8: Case4  
![](images/d3aab802816cfcc58601a9d999602ecab07afe6b6cdc4f54006f172fbc202c2c.jpg)

![](images/04bf018bcca2fc9e5d9ffa1343a0ec3bfd535e8c4fe1e36e2044729b06c255be.jpg)

Figure 9: Case5