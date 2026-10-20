# Not Every Rollout Needs Distillation: Correctness-Filtered On-Policy Self-Distillation

Harrison (Zihao) Zhao<sup>1</sup>, Lei Huang<sup>1</sup>, Fei Liu<sup>1</sup>, Wen Shi<sup>1</sup>, Zijian Jin<sup>1</sup>

<sup>1</sup>Meta

On-policy self-distillation (OPSD) supervises student-generated trajectories with a referenceconditioned version of the same base model. We ask whether every trajectory supplies distinct supervision that warrants its distillation cost. Gradient measurements on Qwen3-1.7B at initializa tion reveal that correct and incorrect responses have similar individual gradient magnitudes, but incorrect responses have approximately three times the coherent gradient share. Their estimated population-mean directions also largely overlap. A signal–noise model fitted at a small group size tracks alignment at larger group sizes without refitting. Motivated by these measurements, we introduce correctness-filtered OPSD (C-OPSD), which generates one response per problem and distills only a subset of incorrect responses. Removing unselected rows before both distillation forwards avoids their teacher/student forward and student backward work. Across Qwen3-1.7B, 4B, and 8B on AIME24, AIME25, and HMMT25, C-OPSD achieves on-par performance with full OPSD. In a measured micro-batch-size-four implementation, executed rows decrease by approximately 68%, and with retained rows packed into full micro-batches, training takes about 40% less time while rollout generation remains unchanged. Together, the measurements and training results connect outcome-conditioned supervision redundancy to more compute-eficient distillation.

∞Meta

## 1 Introduction

On-policy distillation trains a language model on trajectories sampled from its own policy, using a teacher’s token-level distributions to supervise the prefixes the student actually visits (Agarwal et al., 2023). On-policy self-distillation (OPSD) obtains this supervision without a separately trained, larger teacher: the student sees a problem, while a teacher instantiated from the same base model additionally sees a reference solution (Zhao et al., 2026). The reference supplies privileged information during training but is not required at inference. This creates a simple route to improving reasoning, but leaves an allocation question open: how much of this privileged supervision must actually be computed?

Standard OPSD applies its distillation objective to every sampled trajectory, irrespective of whether the student reached the correct final answer. At the training pass rates in our experiments, roughly one-third of the rollouts are already correct. Each nevertheless incurs a student forward, a teacher forward, and student backward computation. Correctness does not make this supervision trivially unnecessary: a correct answer need not imply a correct derivation, and the teacher may still improve the student’s intermediate distributions. The relevant question is empirical: can that supervision be omitted without sacrificing the improvement obtained by full OPSD?

Gradient measurements provide a motivation for selective distillation: incorrect-response gradients are more coherent, while the two outcome populations largely share a mean direction.

Motivated by this observation, we introduce correctness-filtered on-policy self-distillation (C-OPSD). The method generates one response per training problem, checks its final answer, and distills only a subset of incorrect responses. Unselected rows are physically removed before both distillation forwards, reducing teacher/student forward and student backward work without changing rollout generation or the token-level distillation target.

Across Qwen3-1.7B, 4B, and 8B on AIME24, AIME25, and HMMT25, C-OPSD achieves on-par performance while substantially reducing distillation computation. In a measured micro-batch-size-four implementation, executed rows decrease by approximately 68% for each teacher/student forward and student backward. Our experiments evaluate both the performance–computation trade-of and the role of outcome-based supervision allocation.

SCOPE and I-SDPO assign diferent training objectives according to rollout outcomes (Zheng et al., 2026) (Zhang et al., 2026). We instead study whether substantial supervision can be removed under a standalone OPSD objective while retaining performance, and investigate the gradient structure that makes this possible. This distinction motivates both our compute-aware implementation and our analysis of redundancy between outcome-conditioned gradient populations.

Our contributions are threefold:

• Measurements of outcome-conditioned supervision redundancy. We characterize gradient coherence and shared directions across correct and incorrect responses to explain opportunities for selective distillation.

• A measurement-motivated filtering method. C-OPSD retains a subset of incorrect trajectories and removes unselected rows before both distillation forwards to avoid their computation.

• Performance and computation evaluation. Experiments at three model scales show on-par performance with substantially less distillation work; allocation controls connect the supervision choice to observed training outcomes.

## 2 Background

## 2.1 On-policy and privileged-context distillation

In on-policy distillation, a student generates a response and a teacher provides token-level supervision along that response’s prefixes. Unlike training exclusively on teacher-generated sequences, this places supervision on states visited by the student, including those arising from its own mistakes (Agarwal et al., 2023). The sampled tokens determine the prefixes at which distributions are compared; they are not themselves treated as ground-truth answers to imitate.

On-policy self-distillation (OPSD) obtains the teacher’s informational advantage from privileged context rather than a separately trained, larger model (Zhao et al., 2026). Let (x, r) be a training problem and its reference solution, and let $y = ( y _ { 1 } , . . . , y _ { \mathrm { T } } )$ be a student-generated response. At response position $t ,$ the student and teacher distributions are

$$
\begin{array} { l } { p _ { \theta , t } = \pi _ { \theta } ( \cdot \mid x , y _ { < t } ) , } \\ { q _ { \bar { \theta } , t } = \pi _ { \bar { \theta } } ( \cdot \mid x , r , y _ { < t } ) . } \end{array}\tag{1}
$$

Both models are evaluated on the same student-response prefix, but only the teacher sees the reference solution. The notation suppresses the respective prompt templates. In our fixed-teacher configuration, the student trains a LoRA adapter over a frozen base model, while the teacher uses that base model with the adapter disabled. Teacher weights therefore remain fixed as the student adapter changes. The teacher supplies distributions through forward evaluation; it does not generate a second response, and no gradient is propagated through its outputs.

## 2.2 The OPSD training objective

For a micro-batch B of sampled responses, let $m _ { \mathrm { i t } }$ indicate a valid response-token position, excluding prompt and padding positions. Writing $D _ { \tau }$ for the implementation’s clipped token-level divergence, the empirical OPSD objective is

$$
\mathcal { L } _ { \mathrm { O P S D } } ( \theta ; \mathcal { B } ) = \frac { \displaystyle \sum _ { i \in \mathcal { B } } \sum _ { t } m _ { i t } D _ { \tau } \big ( p _ { \theta , i t } , \mathrm { s g } [ q _ { \bar { \theta } , i t } ] \big ) } { \displaystyle \operatorname* { m a x } \left( 1 , \sum _ { i \in \mathcal { B } } \sum _ { t } m _ { i t } \right) } .\tag{2}
$$

where sg denotes stop-gradient. Sampled responses are treated as fixed when diferentiating this loss: the update diferentiates the student distributions, not the discrete sampling procedure. We retain $D _ { \tau }$ as a named operator here because our intervention concerns trajectory selection, not the choice of distillation divergence.

Equation (2) averages over valid tokens within the micro-batch, rather than assigning equal total weight to every response. Responses of diferent lengths can consequently contribute diferent numbers of loss terms. The standard mask does not depend on final-answer correctness: correct and incorrect responses are supervised through the same token-level objective.

## 2.3 Outcome correctness

For each sampled response, let $c _ { \mathrm { i } } \in \{ 0 , 1 \}$ be the outcome label returned by a symbolic answer checker using the training reference answer. This label becomes available after generation and does not require evaluating the privileged teacher. It is a property of the sampled response, not an intrinsic binary dificulty label for the problem: diferent attempts at the same problem may receive diferent outcomes. The checker evaluates the final answer rather than the intermediate reasoning steps.

Using this label to choose which trajectories receive distillation is distinct from using a binary reward in a policy-gradient objective. The supervision on a retained trajectory still comes from the teacher’s conditional distributions. In particular, distilling an incorrect trajectory does not mean maximizing the likelihood of its incorrect final answer.

## 2.4 Where the computation is spent

An OPSD step contains three computational components: student rollout generation, student and teacher evaluation on the sampled trajectories, and backward computation through the student’s loss. The teacher has no backward pass. Generation and subsequent distillation evaluation are separate costs.

A loss mask selects positions in the objective, whereas physical row selection removes trajectories from model evaluation. Selecting rows before the distillation forwards saves both forward evaluation and subsequent student backward work. We track supervision retention, executed forward/backward workload, and end-toend runtime separately; C-OPSD reduces the distillation workload while retaining the baseline generation procedure.

## 3 Related Work

## 3.1 On-policy and privileged-context self-distillation

On-policy distillation supervises prefixes generated by the student, addressing the distribution mismatch between teacher-generated training sequences and student behavior at inference (Agarwal et al., 2023). OPSD obtains teacher supervision from the same base model augmented with a reference solution (Zhao et al., 2026). SDPO similarly converts additional context into dense self-distillation targets, using environment feedback or successful student attempts to inform the self-teacher (Hübotter et al., 2026). Its teacher template also reevaluates successful attempts using the attempt itself as the correct-solution context. These methods establish privileged-context supervision; C-OPSD keeps the OPSD teacher and token-level target fixed and studies which generated trajectories need that supervision computed.

## 3.2 Correctness-dependent supervision

SCOPE separates correct and incorrect rollouts, applies a student-side likelihood objective to correct responses, and uses teacher-guided distillation on incorrect responses, with perplexity-based weighting on both branches (Zheng et al., 2026). I-SDPO routes groups of responses for the same question: all-incorrect groups with available ground truth use privileged self-distillation, whereas groups containing any success use group-relative reinforcement learning, including their incorrect responses (Zhang et al., 2026). C-OPSD instead samples one rollout per question per sampling round, removes correct trajectories without assigning a replacement likelihood or reinforcement-learning loss, and retains only a subset of incorrect trajectories for the original OPSD objective.

Negative Sample Reinforcement provides a related precedent outside distillation: learning from negatively rewarded responses alone can be efective without explicit positive reinforcement of correct responses (Zhu et al., 2025). Its update suppresses incorrect responses and redistributes probability according to the policy’s existing preferences. C-OPSD instead matches privileged teacher distributions at the prefixes of retained incorrect responses. The shared motivation is selective use of outcome-labeled experience; the supervisory signal and update are diferent.

## 3.3 Selecting problems, trajectories, and tokens

PACED estimates each problem’s student pass rate from multiple samples and weights distillation toward intermediate competence, using a default kernel proportional to p(1 − p) (Xu et al., 2026a). Most experiments estimate these weights once and keep them fixed; refresh schedules are also studied. This problem-level curriculum difers from filtering the outcome of a single newly sampled trajectory. A failed attempt on a usually solvable question remains eligible for C-OPSD, and the selection decision requires neither a multi-rollout pass-rate estimate nor teacher logits.

Two further distinctions are worth stating, because Equation (8) resembles PACED’s kernel in form. That kernel is a function of each problem’s estimated pass rate and concentrates weight at intermediate competence; Equation (8) is a function of the batch-level pass rate and describes what fraction of a batch survives, not which dificulty is preferred. The populations they select are in fact near opposite ends of the dificulty range: the mean ofline pass rate of C-OPSD’s retained set is 0.09 (Table 9), against the intermediate values PACED targets. PACED also reweights the loss, so every trajectory still passes through both forwards, whereas C-OPSD removes rows before them and converts the selection into computation saved.

At token level, TIP combines student entropy and teacher–student divergence to select informative positions, including confident predictions on which the teacher disagrees (Xu et al., 2026b). TRACE uses privileged span annotations to route self-distillation to selected correct-key or error spans and combines it with GRPO (Wang et al., 2026). TRACE also anneals the distillation coeficient to zero and then skips teacher forwards, providing a direct precedent for reducing teacher computation as well as selective supervision. C-OPSD uses a diferent intervention: whole trajectories are selected before both distillation forwards, removing discarded rows from student forward, teacher forward, and student backward computation without introducing span annotation or a replacement RL objective. Autoregressive generation is unchanged.

Hou et al. study the number and choice of distinct training problems (Hou et al., 2026). Their eight-hardproblem experiment reaches an average score of 53.6 versus 53.7 for the 17K-problem baseline. The selected problems are reused across training updates, with eight responses per prompt in the default setup; this is not an eight-rollout training budget. Their analysis links hard-problem benefits to longer reasoning trajectories. C-OPSD addresses a complementary eficiency axis: reducing distillation computation on already-generated online trajectories rather than reducing the number of distinct training problems.

## 3.4 Gradient geometry and learning efficiency

Gradient-coherence analyses characterize agreement among example-level updates (Chatterjee, 2020) (Chatterjee and Zielinski, 2020), while gradient-noise scale connects stochastic-gradient variability to useful batch size (McCandlish et al., 2018). PACED applies a related signal–noise perspective to distillation: individual gradients on mastered problems need not vanish, but their dispersion across problems can yield low aggregate signal-to-noise ratio (Xu et al., 2026a). Its empirical diagnostic bins problems by estimated pass rate and measures forward-KL gradients on fixed teacher-generated solutions.

![](images/9d3ba4837e838d7e52b6714c44da09adc3f939bfaf09abefcbc7544b42c6b917.jpg)  
(a) Coherence differs

Our measurements instead partition student-generated OPSD trajectories by observed correctness and jointly examine individual gradient magnitudes, within-population coherence, and overlap between the correct and incorrect population-mean directions. The additional cross-population comparison asks whether the more coherent incorrect trajectories also preserve the direction supplied by correct ones. We further test a fitted signal–noise model at larger group sizes without refitting and connect these measurements to a whole-trajectory compute-removal intervention. Routed OPD studies another level of the update, transporting probability mass between student-excess and teacher-deficit vocabulary alternatives at a given state (Xia et al., 2026). C-OPSD leaves that token-level distillation target unchanged and changes which trajectories are evaluated and diferentiated.

## 4 Outcome-Conditioned Gradient Geometry

We investigate which student-generated trajectories provide distinct distillation signal. The analysis separates three properties: individual gradient magnitude, coherence within each outcome population, and overlap between population-mean directions. These measurements motivate the allocation rule in Section 5; the training experiments in Section 6 evaluate its performance and computational consequences. Figure 1 summarizes the two main geometric findings examined in this section.

![](images/4772fd404424d6d4150f444d0a1daf7c1dc28bc26ad508465175b1d6c78ee6b1.jpg)  
(b) Mean direction barely shifts  
Figure 1 Outcome-conditioned gradient measurements. Qwen3-1.7B at initialization, using 3,000 trajectories and baseweight gradients. (a) Fitted coherent shares are 6.0% and 18.6%, despite similar individual gradient magnitudes. (b) The response-weighted mixture $\mu _ { \mathrm { f u l l } } = \mathrm { p } \mu _ { \mathrm { C } } + ( 1 - \mathrm { p } ) \mu _ { \mathrm { I } }$ , with $\mathrm { p = 0 . 2 9 1 }$ , is approximately $2 . 8 7 ^ { \circ }$ from the incorrect mean direction. Vectors use measured mean magnitudes and the noise-corrected mean-direction cosine; the tip inset is an equal-axis 3× enlargement. These angles describe estimated population means. The following subsections give the measurement procedure and derivation.

## 4.1 Measurement setup

We collect 3,000 trajectories from the Qwen3-1.7B training distribution at initialization: 873 correct and 2,127 incorrect responses, for a 29.1% pass rate. Sampling uses non-thinking responses, a 1,024-token generation limit, temperature 1.1, top-p 0.95, and top-k 20.

Correct and incorrect responses also come from almost disjoint parts of the dificulty range (Table 1): measured against an ofline eight-rollout pass-rate probe, 62% of correct responses fall on problems the base model solves at least seven times in eight, and 69% of incorrect responses on problems it never solves.

Table 1 Ofline pass rate p of the problems behind the measured trajectories, computed on the 2,027 of 3,000 trajectories whose problems appear in the probe.
<table><tr><td>Population</td><td>Trajectories</td><td>Mean po</td><td> ${ \tt p } _ { \mathrm { 0 } } = { \tt 0 }$ </td><td> ${ \pmb \ p } _ { 0 } \ge { \pmb 0 } . { \pmb 8 } 7 { \pmb 5 }$ </td></tr><tr><td>All</td><td>2,027</td><td>0.34</td><td>45.8%</td><td>21.5%</td></tr><tr><td>Correct</td><td>680</td><td>0.80</td><td>0.0%</td><td>62.1%</td></tr><tr><td>Incorrect</td><td>1,347</td><td>0.11</td><td>68.9%</td><td>1.0%</td></tr></table>

Within the middle band $0 . 1 2 5 \le \mathrm { p } _ { 0 } < 0 . 8 7 5$ (258 correct and 406 incorrect trajectories), incorrect responses remain more coherent, with a coherent share of 19.3% against 10.9% for correct responses at group size four. We do not select on dificulty directly, because estimating p<sub>0</sub> requires several rollouts per problem, whereas correctness requires a single answer check.

For each trajectory, we recompute the OPSD loss using a teacher context containing the reference solution and collect gradients with respect to the base-weight matrices targeted by LoRA. This measurement space avoids an initialization-specific degeneracy in adapter coordinates: when the B factor is initialized to zero, the gradient of the A factor also vanishes. Base-weight gradients let us compare the update directions that diferent trajectories induce on the same set of model weights.

We partition trajectories by final-answer correctness and construct four disjoint groups in each trial: two correct groups and two incorrect groups. After summing the gradients within each group, we compute cosine similarities between the two correct groups, between the two incorrect groups, and across outcomes.

The group size n is the number of trajectories aggregated for analysis, not the number of rollouts generated per problem. Within-population alignment measures how consistently trajectories recover a shared direction; cross-population alignment measures whether the two populations supply overlapping directions.

## 4.2 Individual gradient magnitude does not distinguish the populations

Individual gradient magnitudes are 1.538 for correct trajectories and 1.469 for incorrect trajectories. They are similar, and correct trajectories do not have smaller gradients. Their lower signal eficiency therefore cannot be explained simply by claiming that a correct answer makes the distillation gradient negligible.

The distinction therefore requires looking beyond per-trajectory magnitude. We next examine how much gradient energy survives aggregation along a common direction.

## 4.3 A signal–noise model of gradient coherence

Individual gradient magnitude does not fully characterize the learning signal. Large gradients can cancel when they point in diferent directions, whereas aligned gradients preserve a stronger common signal after aggregation.

For either outcome-conditioned population, write a per-trajectory gradient as

$$
g = \mu + \epsilon , \qquad \mathbb { E } [ \epsilon ] = 0 , \qquad \sigma ^ { 2 } = \mathbb { E } \| \epsilon \| ^ { 2 } .
$$

Here, $\mu$ is the population-mean gradient and ϵ is a trajectory’s deviation from that mean. Noise refers to across-trajectory gradient variability, rather than incorrect answers or erroneous labels.

For the sum $\mathrm { G } _ { \mathrm { n } }$ of n independently sampled gradients,

$$
\mathbb { E } \| G _ { n } \| ^ { 2 } = n ^ { 2 } \| \mu \| ^ { 2 } + n \sigma ^ { 2 } .\tag{3}
$$

The shared-direction energy grows quadratically with n, while noise energy grows linearly. The common direction therefore becomes clearer as more trajectories are aggregated. When group norms concentrate suficiently, the expected cosine between two independent groups from the same population is approximately

$$
c _ { n } \approx \frac { \| \boldsymbol { \mu } \| ^ { 2 } } { \| \boldsymbol { \mu } \| ^ { 2 } + \sigma ^ { 2 } / n } .\tag{4}
$$

This relation connects measurable group alignment to the signal–noise structure of individual trajectories. We define the coherent share as

$$
q = \frac { \| \mu \| ^ { 2 } } { \| \mu \| ^ { 2 } + \sigma ^ { 2 } } \approx \frac { c _ { n } } { n - ( n - 1 ) c _ { n } } .
$$

Fitting the model to measurements at group size four gives the parameters in Table 2.

Table 2 Signal–noise parameters fitted at group size four.
<table><tr><td>Population</td><td>Within-population cosine</td><td>Mean-gradient magnitude  $\| \mu \|$ </td><td>Noise scale σ</td><td>Coherent share q</td></tr><tr><td>Correct</td><td>0.2042</td><td>0.3776</td><td>1.4911</td><td>6.0%</td></tr><tr><td>Incorrect</td><td>0.4778</td><td>0.6339</td><td>1.3253</td><td>18.6%</td></tr></table>

The incorrect population has approximately three times the coherent share of the correct population. The distinction is therefore not simply how large individual gradients are, but how much of their energy survives aggregation along a common direction.

The diference emerges after aggregation. Incorrect trajectories have a 1.68-fold larger mean-gradient magnitude and a lower noise scale. Defining per-trajectory signal-to-noise ratio as

$$
\mathrm { S N R _ { 1 } } = { \frac { \lVert { \boldsymbol { \mu } } \rVert } { \sigma } } ,
$$

the incorrect population has approximately 1.89 times the SNR of the correct population.

Under independent sampling from a fixed population, the SNR of n aggregated trajectories scales as $\sqrt { \mathrm { n } }$ Correct trajectories would therefore require approximately $1 . 8 9 ^ { 2 } \approx 3 . 6$ times as many samples to attain the same mean-estimation SNR. This ratio describes sample efficiency for estimating the mean gradient.

This gives our first central finding: correct trajectories produce substantial gradients, but those gradients are more dispersed; incorrect trajectories recover a stable common direction with fewer samples. This diference motivates prioritizing incorrect trajectories when distillation computation is limited.

## 4.4 Testing predictions at different group sizes

If this decomposition captures the gradient structure, parameters estimated from small groups should also predict alignment in larger groups. We fix the parameters fitted at group size four and compare the resulting predictions with measurements at other group sizes, without refitting.

Table 3 Predicted and measured within-population cosine at diferent group sizes.
<table><tr><td>Group size</td><td>Correct: predicted</td><td>Correct: measured</td><td>Incorrect: predicted</td><td>Incorrect: measured</td></tr><tr><td>16</td><td>0.506</td><td>0.521</td><td>0.785</td><td>0.790</td></tr><tr><td>64</td><td>0.804</td><td>0.800</td><td>0.936</td><td>0.940</td></tr><tr><td>218</td><td>0.933</td><td>0.934</td><td>0.980</td><td>0.982</td></tr></table>

(a) Alignment across group sizes

![](images/ffa9a7cddf95d748a305bdd17647913490fb8f56a4d0c7260db98275d938a7bc.jpg)  
○ / □ fit at m = 4; filled markers: larger groups  
(b) Prediction error

![](images/9dfebd7178060ec039b354b0904633de2d3cfaab4608079309e05e3052e3c720.jpg)

Figure 2 Measured alignment and fixed-model predictions. Markers show within-population gradient cosine; curves use the parameters fitted at group size four, without refitting at larger sizes. Hollow markers identify the fit group size, and filled markers show measurements at sizes 16, 64, and 218. The right panel shows measured minus predicted cosine. Group size counts trajectories aggregated for analysis, not rollouts per problem. All measurements are from Qwen3-1.7B at initialization.

Predictions closely track measurements across aggregation scales. The model captures both the increase in alignment with group size and the diferent rates at which the populations approach their mean directions: incorrect trajectories yield a stable direction in smaller groups, whereas correct trajectories require more samples to average out variability.

Large-group measurements provide a second cross-check. At group size 218, estimated mean-gradient magnitudes are approximately 0.375 and 0.633 for correct and incorrect trajectories, close to 0.378 and 0.634 from the small-group fit. These results support using the same signal–noise parameters to describe gradient behavior across aggregation scales.

## 4.5 Stability during training

We repeat them on trajectories sampled from a C-OPSD checkpoint (1.7B) at steps 25, 50, and 100, using the same protocol at group size 16.

Table 4 Coherent share and noise-corrected cross-population cosine during C-OPSD training, group size 16.
<table><tr><td>Checkpoint</td><td>Correct q</td><td>Incorrect q</td><td>Ratio</td><td>Corrected cross cosine</td></tr><tr><td>Initialization</td><td>6.4%</td><td>19.1%</td><td>2.99</td><td>0.975</td></tr><tr><td>Step 25</td><td>8.3%</td><td>25.9%</td><td>3.12</td><td>0.980</td></tr><tr><td>Step 50</td><td>11.3%</td><td>33.2%</td><td>2.94</td><td>0.941</td></tr><tr><td>Step 100</td><td>11.5%</td><td>33.9%</td><td>2.95</td><td>0.943</td></tr></table>

Both populations become more coherent as training proceeds, but the ratio between them stays close to three, and the two mean directions remain closely aligned. The two findings that motivate C-OPSD hold throughout the regime in which it operates, not only at initialization.

## 4.6 Shared mean directions and the effect of omitting correct responses

Greater coherence alone does not explain why correct trajectories can be omitted. If they supply a distinct, important direction, dropping them could remove complementary supervision despite their higher noise. We therefore compare the two population-mean directions.

At group size 218, the measured cross-population cosine is 0.9267. Finite groups still contain sampling variability, so this value reflects both the diference between the mean directions and within-population noise. Using each population’s internal alignment, a directional-noise approximation gives

$$
\cos ( \mu _ { \mathrm { c o r r e c t } } , \mu _ { \mathrm { i n c o r r e c t } } ) \approx \frac { 0 . 9 2 6 7 } { \sqrt { 0 . 9 3 4 3 \times 0 . 9 8 2 4 } } \approx 0 . 9 6 7 .\tag{5}
$$

The corresponding angle is approximately 14.7<sup>◦</sup>. Applying the same correction at group sizes 16, 64, and 218 gives 0.974, 0.967, and 0.967, respectively, indicating a stable estimate of mean-direction overlap across aggregation scales.

Decomposing the correct population’s mean into components parallel and orthogonal to the incorrect mean gives approximate magnitudes

$$
\begin{array} { r } { \| \mu _ { \mathrm { c o r r e c t , \parallel } } \| \approx 0 . 3 6 5 3 , \qquad \| \mu _ { \mathrm { c o r r e c t , \perp } } \| \approx 0 . 0 9 5 8 . } \end{array}
$$

The correct mean therefore lies predominantly along a direction already supplied by incorrect trajectories, with a smaller orthogonal component.

This gives our second central finding: correct trajectories largely reinforce an existing direction rather than supplying a separate one. Together with the coherence analysis, this suggests that the reason to omit correct trajectories is not that their gradients are harmful, but that they provide noisier, substantially overlapping supervision.

To quantify the efect of removing the correct population from the response-weighted mean, let p = 873/3,000 = 0.291 and let θ be the estimated angle between the two population means. The mixture is

$$
\mu _ { \mathrm { f u l l } } = p \mu _ { \mathrm { C } } + ( 1 - p ) \mu _ { \mathrm { I } } .
$$

Using $\| \mu _ { \mathrm { C } } \| = 0 . 3 7 7 6 , \| \mu _ { \mathrm { I } } \| = 0 . 6 3 3 9$ , and the unrounded noise-corrected cosine from Equation (5), the angle between this mixture and the incorrect mean is

$$
\varphi = \arctan \left( \frac { p \| \mu _ { \mathrm { C } } \| \sin \theta } { ( 1 - p ) \| \mu _ { \mathrm { I } } \| + p \| \mu _ { \mathrm { C } } \| \cos \theta } \right) \approx 2 . 8 7 ^ { \circ } .
$$

The correct population is less frequent and has a smaller mean-gradient magnitude, limiting its contribution to the mixture’s orthogonal component. The resulting mixture direction deviates from the incorrect mean by approximately $2 . 8 7 ^ { \circ }$ , with a normalized orthogonal projection residual of approximately 5.0%. Figure 1 illustrates this response-weighted geometry.

Together, greater incorrect-gradient coherence and overlapping mean directions motivate allocating less distillation work to correct responses. C-OPSD operationalizes this allocation by retaining a subset of incorrect responses, and the training experiments test whether the resulting reduction in supervision preserves performance.

## 5 Method

The gradient measurements motivate reducing supervision on correct responses. C-OPSD implements this idea using outcome verification after generation and physical row removal before distillation, as shown in Figure 3.

![](images/c0a792d952495024376fae175e400566d6f29fe636e94d1fe09167b79da71445.jpg)  
Same generation · smaller tensors in both forwards and student backward  
Figure 3 One rollout per problem and physically compacted distillation batches. Full OPSD processes all generated trajectories. C-OPSD checks correctness, retains s = min(k, n − k) incorrect responses, and removes unselected rows before both distillation forwards. The example $\mathbf { n } = 4 , \mathbf { k } = 1$ produces one-row tensors rather than four-row tensors with masked entries. The reference-conditioned teacher is frozen; only the student is updated. Generation is unchanged. Empty selections execute one fully masked fallback row.

## 5.1 Outcome verification after generation

Consider a micro-batch of n distinct training problems, each with one student-generated response. We retain the same student rollout procedure and privileged reference context as the OPSD baseline. Once generation has finished, an answer extractor and symbolic checker compare each response with its training reference answer, producing $\mathrm { { c } _ { i } \in \{ 0 , 1 \} }$ . The implementation uses the same math\_verify-based verification path described for evaluation, allowing equivalent forms such as $1 / 2$ and 0.5 to agree. This decision uses training labels, not evaluation-benchmark answers.

Verification operates directly on the generated answer and training reference, enabling selection before either distillation forward without a second generation or teacher-scoring pass.

## 5.2 Correctness-filtered retention

Let $\mathrm { C = \{ i : c _ { i } = 1 \} }$ and $\mathrm { W = \{ i : c _ { i } = 0 \} }$ , with $\mathrm { k } = | \mathrm { C } |$ and $| \mathrm { W } | = \mathrm { n - k }$ . C-OPSD retains the first $\mathrm { ~ s ~ } =$ min(k, $\mathrm { ~ n ~ - ~ k ~ } )$ indices in W under the existing micro-batch order. Thus all retained trajectories are incorrect, but not every incorrect trajectory is retained. We denote the resulting index set by S.

$$
s = \operatorname* { m i n } ( k , n - k ) , \quad \quad S = \operatorname { f r s t } _ { s } ( W ) .\tag{6}
$$

The dependence on k defines the retained budget used in the selection study. Because s is capped by the number of available incorrect responses, C-OPSD is not always exactly count-matched to a correct-only arm that retains all k correct responses. It is exactly count-matched, on the same sampled batch, to a random control selecting s responses regardless of correctness. Across separate on-policy training runs, sampled responses and hence realized counts can diverge.

For k = 0 or k = n, the rule selects no trajectory. For n = 1 it always selects none, so this particular budget rule requires larger micro-batches to provide training signal. The method’s selection behavior depends on micro-batch size, not only on the total optimizer-step batch size.

## 5.3 Retained objective and row removal

For a retained set S, the filtered objective is the same token divergence as in Equation (2), normalized over the valid tokens of retained trajectories:

$$
\mathcal { L } _ { \mathrm { C - O P S D } } ( \theta ; \mathcal { B } , S ) = \frac { \displaystyle \sum _ { i \in S } \sum _ { t } m _ { i t } D _ { \tau } \left( p _ { \theta , i t } , \mathrm { s g } \big [ q _ { \bar { \theta } , i t } \big ] \right) } { \displaystyle \operatorname* { m a x } \left( 1 , \sum _ { i \in S } \sum _ { t } m _ { i t } \right) } .\tag{7}
$$

Equation (7) averages the distillation loss over retained response tokens, with the selection treated as fixed during diferentiation. For fixed sampled trajectories and per-token model outputs, it is equivalent to masking all unselected rows in the full-shaped batch: both formulations use the same retained-token numerator and denominator.

To realize computational savings, we index-select the retained rows from student input IDs and attention masks, teacher input IDs and attention masks, and labels before either distillation forward. Derived tensors are formed after selection. Prompt-length values that are scalars are not row-indexed. For an empty selection, the implementation keeps one row with all response labels masked so that the loss remains zero without passing a zero-row batch to unsupported kernels.

## 5.4 Algorithm and controls

Algorithm 1. C-OPSD with physical row removal.

1. Generate one response per problem with the current student policy; retain the generated tokens and the training references.

2. Extract and verify final answers, construct C and W, and compute s = min(|C|, |W|).

3. Select the first s incorrect rows under the existing batch order.

4. If $\mathrm { \ s > 0 }$ , index-select the same rows from all row-aligned student, teacher, and label tensors. If $\mathrm { s } = 0$ , keep one fully masked fallback row.

5. Evaluate the student and reference-conditioned teacher on the selected prefixes; keep teacher outputs detached.

6. Compute the retained-token loss, backpropagate through the student, and follow the baseline’s accumulation and optimizer-step procedure.

Table 5 Supervision-allocation strategies.
<table><tr><td>Arm</td><td>Retained trajectories</td><td>Count for the same batch</td><td>Role</td></tr><tr><td>Full OPSD</td><td>All trajectories</td><td>n</td><td>Performance reference</td></tr><tr><td>Correct-only</td><td>All correct trajectories</td><td>k</td><td>Opposite outcome allocation</td></tr><tr><td>C-OPSD</td><td>First s incorrect trajectories s = min(k, n − k)</td><td></td><td>Proposed filtered objective</td></tr><tr><td>Random selection</td><td>Outcome-blind subset of all s trajectories</td><td></td><td>Count-matched selection control</td></tr><tr><td>All-incorrect</td><td>All incorrect trajectories</td><td>n − k</td><td>Higher-retention comparison</td></tr></table>

## 5.5 Retention as a function of micro-batch size

For an illustrative model with independent, identically distributed correctness labels of success probability p, K follows Binomial(n, p). The expected loss-retention fraction is ${ \bf r } _ { \mathrm { n } } ( { \mathrm { p } } ) = \mathrm { E } [ \operatorname* { m i n } ( \mathrm { K } , \mathrm { n } - \mathrm { K } ) ] / \mathrm { n }$ . For $\mathrm { n = 4 }$ direct enumeration gives:

$$
r _ { 4 } ( p ) = p ( 1 - p ) ( 1 + p - p ^ { 2 } ) .\tag{8}
$$

The maximum is 0.3125 at p = 0.5. At p = 0.33, the expected loss-retention fraction is approximately 0.270. Empty selections occur with probability $\mathrm { { p } ^ { n } + ( 1 - p ) ^ { n } }$ . Under the one-row fallback, the expected executed-row fraction is $\mathrm { { r } _ { n } ( p ) + [ p ^ { n } + ( 1 - p ) ^ { n } ] / n }$ . For $\mathrm { n = 4 }$ and $\mathrm { p = 0 . 3 3 }$ this is approximately 0.323, or 1.29 physical rows per micro-batch. For ${ \mathrm { n } } = 2 .$ the same implementation always executes one row: its physical retention is 50%, even though its expected loss retention is p(1 − p).

The rule has no free parameter, but it does have a domain. Equation (8) makes $\mathrm { \Delta r _ { 4 } ~ a }$ function of the pass rate: it peaks at 31.3% near $\mathrm { p } = 0 . 5$ and is flat over $\mathrm { p } \in [ 0 . 3 , 0 . 5 ]$ , which covers the 29–35% pass rates observed in our runs. Outside that band it falls away on both sides, to approximately 22% at p = 0.25, 18.6% at $\mathrm { p = 0 . 2 }$ and 9.8% at $\mathrm { p = 0 . 1 }$ . Because the pass rate shifts as the student trains, from 33% to 29% over 100 steps at 1.7B, the budget also adapts without a schedule. A fixed-rate subsampling rule would be insensitive to this; min(k, n − k) trades that robustness for the absence of a tuned retention hyperparameter, and at our pass rates the two retain near-identical sets on the same sampled batches. We therefore state the rule as intended for pass rates in roughly [0.25, 0.7], and recommend a fixed retention fraction outside that range.

## 6 Experimental Results

## 6.1 Models, training, and evaluation

We evaluate Qwen3 models (Qwen Team, 2025) at 1.7B, 4B, and 8B parameters using the OPSD training split named siyanzhao/Openthoughts\_math\_30k\_opsd. Our ofline dificulty probe covers 29,427 problems. We use rank-64 LoRA (Hu et al., 2021) with scaling α = 128, targeting attention q, k, v, and o projections and MLP gate, up, and down projections. The reference-conditioned teacher uses the frozen base with the adapter disabled.

Table 6 Training and evaluation settings.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Optimizer-step rollout count</td><td>32 distinct problems; one response per problem</td></tr><tr><td>Student rollout mode</td><td>Non-thinking; vLLM (Kwon et al., 2023) colocated generation</td></tr><tr><td>Maximum rollout length</td><td>1,024 tokens</td></tr><tr><td>Rollout sampling</td><td>Temperature 1.1; top-p 0.95; top-k 20</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 6 }$  ; constant schedule</td></tr><tr><td>Token-divergence clip parameter</td><td>0.05 for 1.7B/4B; 0.06 for 8B</td></tr><tr><td>Main checkpoint selection set</td><td>Steps 25, 50, 75, and 100</td></tr><tr><td>Training runs per arm</td><td>Five (main comparison and ablation); three (SFT, GRPO)</td></tr><tr><td>Evaluation</td><td>Thinking mode; maximum 38,912 new tokens; temperature 1.0; top-p 0.95</td></tr></table>

Table 7 Batch configurations across experiments.
<table><tr><td>Result block</td><td>Per-device batch</td><td>Gradient accumulation</td><td>Ranks</td><td>Total rollouts/step</td></tr><tr><td>All experiments</td><td>4</td><td>8</td><td>1</td><td>32</td></tr></table>

For each benchmark problem, Avg@M averages correctness over M sampled responses and then averages over problems, measuring sampled-answer accuracy. All reported results use Avg@48. Each of the three benchmark grids contains 30 problems, yielding 1,440 sampled responses per benchmark. The main comparison and the outcome-allocation ablation use five independent training runs per arm; SFT and GRPO use three. Rollout sampling is not seeded, so repeated runs of one configuration diverge on their own; training jobs also vary the trainer seed, which changes the data order. Reported scores are means over runs.

The main table uses the best checkpoint for each benchmark among steps 25, 50, 75, and 100; its aggregate combines those benchmark-specific selections. We also report per-benchmark accuracy at each of these checkpoints using the same runs. The main selection uses only the first 100 steps, including for 1.7B runs that continued to step 200.

## 6.2 Main comparison: on-par performance with less distillation

Table 8 reports all five arms at the three model scales. It contains two kinds of comparison. Both OPSD arms lead the untrained base model and the SFT and GRPO baselines on every benchmark at every scale, by margins that are large relative to the run-to-run standard deviations in the SD column. Between the two OPSD arms the aggregate diferences (C-OPSD minus OPSD) are −0.19, +0.23, and +0.09 percentage points at 1.7B, 4B, and 8B: inconsistent in sign and of the same size as that variation, with 95% confidence intervals of [−0.67, 0.28], [−0.35, 0.82], and [−0.24, 0.42]. At a fixed final checkpoint (step 100), C-OPSD minus OPSD is +0.41, +1.02, and +0.10. C-OPSD is therefore on par with full OPSD, and retains the whole of its advantage over the baselines, while computing the distillation loss on roughly a quarter of the generated trajectories.

Table 8 Best checkpoint per benchmark, Avg@48. Scores are percentages, averaged over training runs: five for the OPSD arms, three for SFT and GRPO, one for the untrained base. SD is the standard deviation of the per-run aggregate. A rule separates the baselines from the two OPSD arms within each model size. Retention is the fraction of generated responses distilled. Executed-row measurements are presented separately in Section 7.1.
<table><tr><td>Model</td><td>Method</td><td>AIME24</td><td>AIME25</td><td>HMMT25</td><td>Mean</td><td>SD</td><td>Retention</td></tr><tr><td>1.7B</td><td>Base</td><td>49.10</td><td>33.89</td><td>23.26</td><td>35.42</td><td></td><td></td></tr><tr><td></td><td>SFT</td><td>50.42</td><td>38.59</td><td>24.44</td><td>37.82</td><td>0.19</td><td>100%</td></tr><tr><td></td><td>GRPO</td><td>49.26</td><td>36.16</td><td>23.66</td><td>36.36</td><td>0.42</td><td>100%</td></tr><tr><td></td><td>OPSD</td><td>56.19</td><td>41.85</td><td>28.74</td><td>42.26</td><td>0.29</td><td>100%</td></tr><tr><td></td><td>C-OPSD</td><td>56.18</td><td>41.50</td><td>28.51</td><td>42.06</td><td>0.34</td><td>26%</td></tr><tr><td>4B</td><td>Base</td><td>74.38</td><td>65.35</td><td>43.75</td><td>61.16</td><td>—</td><td></td></tr><tr><td></td><td>SFT</td><td>72.78</td><td>66.76</td><td>43.12</td><td>60.89</td><td>0.32</td><td>100%</td></tr><tr><td></td><td>GRPO</td><td>74.35</td><td>66.48</td><td>43.84</td><td>61.56</td><td>0.30</td><td>100%</td></tr><tr><td></td><td>OPSD</td><td>74.96</td><td>67.90</td><td>45.03</td><td>62.63</td><td>0.47</td><td>100%</td></tr><tr><td></td><td>C-OPSD</td><td>75.54</td><td>68.47</td><td>44.57</td><td>62.86</td><td>0.18</td><td>26%</td></tr><tr><td>8B</td><td>Base</td><td>76.81</td><td>69.38</td><td>45.28</td><td>63.82</td><td></td><td></td></tr><tr><td></td><td>SFT</td><td>75.05</td><td>68.43</td><td>44.95</td><td>62.81</td><td>0.30</td><td>100%</td></tr><tr><td></td><td>GRPO</td><td>76.64</td><td>69.91</td><td>45.67</td><td>64.07</td><td>0.09</td><td>100%</td></tr><tr><td></td><td>OPSD</td><td>77.50</td><td>70.72</td><td>46.65</td><td>64.96</td><td>0.20</td><td>100%</td></tr><tr><td></td><td>C-OPSD</td><td>77.43</td><td>70.67</td><td>47.06</td><td>65.05</td><td>0.24</td><td>26%</td></tr></table>

Appendix A (Table A1) provides the per-dataset, per-checkpoint breakdown for the two OPSD arms.

## 6.3 Outcome-allocation ablation

We next compare which trajectories receive a restricted budget. The 1.7B ablation uses the same protocol as the main comparison – best checkpoint per benchmark, Avg@48 – with five training runs per arm. Measured retention is 26% for random selection and C-OPSD, so the random control is closely count-matched; correct-only retains 33% and all-incorrect 68%. Table 9 reports the result.

The pattern is consistent across the four allocations. Correct-only is by far the weakest, 1.71 points below C-OPSD: spending the budget on trajectories the student already solved is the one choice that clearly costs accuracy. Random selection sits between the two, 0.48 below C-OPSD; it already directs 66% of its budget to incorrect responses, since those are the majority of generated trajectories, so restricting the remainder to incorrect responses moves the score only a little further. All-incorrect, which keeps 68% of rollouts rather than 26%, scores level with C-OPSD and with full OPSD, so the gain comes from excluding correct trajectories rather than from retaining more of the incorrect ones.

Table 9 Outcome allocation: best checkpoint per benchmark, Avg@48, mean over five training runs. Ofline p<sub>0</sub> estimates the retained problems’ pass rates from a separate eight-rollout probe. Full OPSD, at 100% retention, scores 42.26 under the same protocol.
<table><tr><td>Arm</td><td>Retention</td><td>Incorrect among retained</td><td>Mean po</td><td>Mean accuracy (SD)</td></tr><tr><td>Correct-only</td><td>33%</td><td>0%</td><td>0.83</td><td>40.35 (0.20)</td></tr><tr><td>Random selection</td><td>26%</td><td>66%</td><td>0.42</td><td>41.58 (0.31)</td></tr><tr><td>C-OPSD</td><td>26%</td><td>100%</td><td>0.09</td><td>42.06 (0.34)</td></tr><tr><td>All-incorrect</td><td>68%</td><td>100%</td><td>0.09</td><td>42.17 (0.38)</td></tr></table>

The retained populations also difer in problem dificulty: their mean ofline pass rates are 0.83, 0.42, and 0.09 for correct-only, random, and C-OPSD, respectively. The ablation compares these outcome-conditioned populations, with response correctness and problem dificulty varying together.

## 6.4 Gradient geometry and supervision allocation

Across the three allocation strategies, the estimated shared-direction SNR follows the same ordering as measured accuracy: C-OPSD > Random selection > Correct-only. Table 10 compares the SNR estimates, computed from the fitted gradient parameters and retained trajectory counts, with the best-checkpoint, Avg@48 ablation results.

Table 10 Estimated SNR and measured ablation results for three count-matched supervision-allocation strategies.
<table><tr><td>Strategy</td><td>Expected retained trajectories/step</td><td>Shared-direction signal</td><td>Noise</td><td>Estimated SNR</td><td>Mean accuracy</td></tr><tr><td>Correct-only</td><td>10.56</td><td>3.86</td><td>4.85</td><td>0.80</td><td>40.35</td></tr><tr><td>Random selection 8.32</td><td></td><td>4.51</td><td>3.99</td><td>1.13</td><td>41.58</td></tr><tr><td>C-OPSD</td><td>8.32</td><td>5.27</td><td>3.82</td><td>1.38</td><td>42.06</td></tr></table>

The estimated SNR ordering is consistent with the observed accuracy ordering: C-OPSD exceeds random selection, which exceeds correct-only selection. C-OPSD and random selection retain nearly the same number of trajectories. Correct-only selection retains more trajectories yet has lower estimated SNR and lower measured accuracy.

The comparison requires similar retained counts, because aggregated SNR grows with the square root of that count. Full OPSD retains all 32 and scores 2.29 against C-OPSD’s 1.38, yet the two are on par in Table 8. Table 10 therefore speaks to which trajectories are kept, not how many.

Together with the main comparison, these results connect the outcome-conditioned geometry to a practical allocation of distillation work. The geometry motivates the method; the training experiments show that C-OPSD achieves on-par performance with substantially less forward and backward computation.

## 7 Computational Efficiency

## 7.1 Measured forward/backward reduction

C-OPSD converts trajectory filtering into actual computation savings by removing unselected rows before both distillation forwards. In the measured micro-batch-size-four implementation, each student forward, teacher forward, and student backward processes an average of 1.29 rows instead of 4, a reduction of approximately 68%. The count includes the fully masked fallback row executed for an empty selection. The retained-token objective is defined in Section 5.3.

Table 11 Executed distillation rows at micro-batch size four.
<table><tr><td>Distillation operation</td><td>Full OPSD rows</td><td>C-OPSD mean rows</td><td>Executed-row reduction</td></tr><tr><td>Student forward</td><td>4</td><td>1.29</td><td>≈68%</td></tr><tr><td>Teacher forward</td><td>4</td><td>1.29</td><td>≈68%</td></tr><tr><td>Student backward</td><td>4</td><td>1.29</td><td>≈68%</td></tr></table>

All four student trajectories are generated before selection. The reduction therefore applies to distillation evaluation and backward work; Section 7.2 incorporates the unchanged generation cost into the total-step FLOP model.

## 7.2 Estimated total-step FLOP savings

For an illustrative model of dense parameter-matrix work, let N be the non-embedding parameter count, P the student prompt length, C the completion length, $\mathrm { L } _ { \mathrm { r e f } }$ the extra teacher-reference length, and R the number of physical rows used by distillation in an optimizer step. We estimate these components as follows:

$$
\begin{array} { r l } & { F _ { \mathrm { g e n e r a t i o n } } \approx 3 2 ( P + C ) \times 2 N , } \\ & { F _ { \mathrm { s t u d e n t } } \approx R ( P + C ) \times 6 N , } \\ & { F _ { \mathrm { t e a c h e r } } \approx R ( P + L _ { \mathrm { r e f } } + C ) \times 2 N . } \end{array}\tag{9}
$$

The student term assumes a frozen-base LoRA implementation with activation-gradient work and one checkpoint-recomputation forward, rather than full base-weight gradient computation. For this estimate, we use $\mathrm { P } \approx 1 5 0 , \mathrm { C } = 9 7 8 , \mathrm { L } _ { \mathrm { r e f } } = 6 1 6$ , and change R from 32 to 10.32 for the n = 4 implementation. With these shared assumptions, the fractional saving is independent of N and is approximately 55.5% of the modeled total step FLOPs.

Table 12 Modeled FLOPs per optimizer step (TFLOPs), using the shared n = 4 configuration and lengths in Equation (9).
<table><tr><td>Model scale</td><td>Generation</td><td>Student f+b</td><td>Teacher f</td><td>Full total</td><td>Filtered total</td><td>Modeled saving</td></tr><tr><td>1.7B</td><td>101.7</td><td>305.2</td><td>157.3</td><td>564.1</td><td>250.9</td><td>55.5%</td></tr><tr><td>4B</td><td>262.3</td><td>786.8</td><td>405.5</td><td>1454.6</td><td>646.8</td><td>55.5%</td></tr><tr><td>8B</td><td>501.5</td><td>1504.3</td><td>775.3</td><td>2781.1</td><td>1236.6</td><td>55.5%</td></tr></table>

The total-step FLOP estimate includes unchanged rollout generation and reduced distillation forward/backward work.

## 7.3 Measured wall-clock time

We measure training time on a single GB300 GPU with colocated vLLM and 32 rollouts per optimizer step (per-device batch 4, gradient accumulation 8). For both methods, all rollouts of a step are generated in one vLLM call, and to turn the row reduction into time, C-OPSD packs the retained rows of a step into full micro-batches of four. Both changes are mathematically equivalent to the original update, identical up to floating-point rounding, which we verified numerically (Appendix B). Table 13 reports the wall-clock time of a full 100-step training loop, including checkpoint saves every 25 steps, and the median time per step over steps 2–12.

Table 13 Measured wall-clock time on a single GB300 GPU.
<table><tr><td colspan="4">100-step training (min)</td><td colspan="2">Per step (s)</td></tr><tr><td>Model</td><td>OPSD</td><td>C-OPSD</td><td>Saving</td><td>OPSD</td><td>C-OPSD</td></tr><tr><td>1.7B</td><td>14.2</td><td>8.4</td><td>41%</td><td>7.86</td><td>4.79</td></tr><tr><td>4B</td><td>19.8</td><td>11.8</td><td>40%</td><td>12.36</td><td>6.95</td></tr><tr><td>8B</td><td>22.6</td><td>13.7</td><td>40%</td><td>12.98</td><td>8.05</td></tr></table>

## 8 Conclusion

Outcome-conditioned gradient measurements reveal a form of supervision redundancy in OPSD. Correct and incorrect responses have similar individual gradient magnitudes, but incorrect responses have greater coherence, while the two population-mean directions largely overlap. A signal–noise model fitted at a small group size tracks alignment at larger sizes, and the measured population proportions and mean magnitudes explain why omitting the correct population changes the estimated response-weighted mean direction only slightly.

These findings motivate C-OPSD, which generates one response per problem, retains a subset of incorrect responses, and removes unselected rows before both distillation forwards. Across Qwen3-1.7B, 4B, and 8B on AIME24, AIME25, and HMMT25, C-OPSD achieves on-par performance with full OPSD. In a measured micro-batch-size-four implementation, executed rows decrease from 4 to 1.29 per teacher/student forward and student backward, a reduction of approximately 68%, and packing the retained rows into full micro-batches turns this into about 40% less training time. The method preserves the token-level distillation target while concentrating computation on a smaller set of trajectories. Together, the measurements and training results demonstrate the practical value of allocating distillation work according to outcome-conditioned gradient structure.

## References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. arXiv:2306.13649; 2023.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, Aditya Grover. Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models. arXiv:2601.18734; 2026.

Binbin Zheng, Xing Ma, Yiheng Liang, Jingqing Ruan, Xiaoliang Fu, Kepeng Lin, Benchang Zhu, Ke Zeng, Xunliang Cai. SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting. arXiv:2604.10688; 2026.

Yubo Zhang, Xinhong Ma, Zezhong Tan, Ziqiang Dong. I-SDPO: Instance-Level Adaptive Self-Distillation Policy Optimization. arXiv:2608.12957; 2026.

Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, Alborz Geramifard. TIP: Token Importance in On-Policy Distillation. arXiv:2604.14084; 2026b.

Zhinan Hou, Jiaqi Zhang, Xunliang Cai, Keyou You. What Matters in On-Policy Distillation? A Perspective on Data Eficiency and Data Selection. arXiv:2609.05198; 2026.

Satrajit Chatterjee. Coherent Gradients: An Approach to Understanding Generalization in Gradient Descent-Based Optimization. arXiv:2002.10657; 2020.

Satrajit Chatterjee, Piotr Zielinski. Making Coherence Out of Nothing At All: Measuring the Evolution of Gradient Alignment. arXiv:2008.01217; 2020.

Sam McCandlish, Jared Kaplan, Dario Amodei, OpenAI Dota Team. An Empirical Model of Large-Batch Training. arXiv:1812.06162; 2018.

Tianle Xia, Lingxiang Hu, Yiding Sun, Linfang Shang, Ming Xu, Lan Xu, Ning Zheng, Wei Xu, Jie Jiang. Distillation as Probability Transport: Routed On-Policy Distillation. arXiv:2609.08337; 2026.

Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685; 2021.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. Eficient Memory Management for Large Language Model Serving with PagedAttention. arXiv:2309.06180; 2023.

Qwen Team. Qwen3 Technical Report. arXiv:2505.09388; 2025.

Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang. PACED: Distillation and On-Policy Self-Distillation at the Frontier of Student Competence. arXiv:2603.11178v3; 2026a.

Jiaxuan Wang, Xuan Ouyang, Zhiyu Chen, Yulan Hu, Zheng Pan, Xin Li, Lan-Zhe Guo. TRACE: Distilling Where It Matters via Token-Routed Self On-Policy Alignment. arXiv:2605.10194v1; 2026.

Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, Andreas Krause. Reinforcement Learning via Self-Distillation. arXiv:2601.20802v2; 2026.

Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, Yu Meng. The Surprising Efectiveness of Negative Reinforcement in LLM Reasoning. Advances in Neural Information Processing Systems (NeurIPS); 2025. arXiv:2506.01347.

## A Per-Dataset Checkpoint Results

Table A1 reports accuracy on each benchmark at steps 25, 50, 75, and 100 for the same runs, averaged over the five training runs per arm. This breakdown shows checkpoint-level performance on AIME24, AIME25, and HMMT25 alongside the best-checkpoint comparison in Table 8.

Table A1 Per-dataset accuracy (Avg@48) at each evaluation checkpoint, averaged over the five training runs per arm used in Table 8. Bold indicates the higher score between OPSD and C-OPSD within each model, dataset, and checkpoint.
<table><tr><td>Model</td><td>Dataset</td><td>Method</td><td>Step 25</td><td>Step 50</td><td>Step 75</td><td>Step 100</td></tr><tr><td>1.7B</td><td>AIME24</td><td>OPSD</td><td>53.33</td><td>54.76</td><td>56.00</td><td>55.10</td></tr><tr><td>1.7B</td><td>AIME24</td><td>C-OPSD</td><td>52.64</td><td>54.03</td><td>56.04</td><td>55.14</td></tr><tr><td>1.7B</td><td>AIME25</td><td>OPSD</td><td>39.60</td><td>41.32</td><td>41.01</td><td>39.75</td></tr><tr><td>1.7B</td><td>AIME25</td><td>C-OPSD</td><td>39.03</td><td>40.40</td><td>41.38</td><td>40.38</td></tr><tr><td>1.7B</td><td>HMMT25</td><td>OPSD</td><td>24.57</td><td>27.08</td><td>28.74</td><td>27.89</td></tr><tr><td>1.7B</td><td>HMMT25</td><td>C-OPSD</td><td>24.43</td><td>25.42</td><td>27.49</td><td>28.46</td></tr><tr><td>4B</td><td>AIME24</td><td>OPSD</td><td>74.03</td><td>74.93</td><td>74.53</td><td>73.72</td></tr><tr><td>4B</td><td>AIME24</td><td>C-OPSD</td><td>74.56</td><td>74.85</td><td>75.49</td><td>75.03</td></tr><tr><td>4B</td><td>AIME25</td><td>OPSD</td><td>67.11</td><td>67.42</td><td>67.64</td><td>66.14</td></tr><tr><td>4B</td><td>AIME25</td><td>C-OPSD</td><td>66.57</td><td>67.51</td><td>68.04</td><td>67.54</td></tr><tr><td>4B</td><td>HMMT25</td><td>OPSD</td><td>42.71</td><td>43.99</td><td>44.90</td><td>43.78</td></tr><tr><td>4B</td><td>HMMT25</td><td>C-OPSD</td><td>42.60</td><td>42.83</td><td>44.17</td><td>44.14</td></tr><tr><td>8B</td><td>AIME24</td><td>OPSD</td><td>76.15</td><td>77.22</td><td>77.18</td><td>76.92</td></tr><tr><td>8B</td><td>AIME24</td><td>C-OPSD</td><td>76.96</td><td>76.56</td><td>76.81</td><td>77.38</td></tr><tr><td>8B</td><td>AIME25</td><td>OPSD</td><td>69.11</td><td>70.08</td><td>69.96</td><td>70.46</td></tr><tr><td>8B</td><td>AIME25</td><td>C-OPSD</td><td>69.72</td><td>69.42</td><td>69.67</td><td>70.15</td></tr><tr><td>8B</td><td>HMMT25</td><td>OPSD</td><td>45.43</td><td>45.89</td><td>46.36</td><td>46.47</td></tr><tr><td>8B</td><td>HMMT25</td><td>C-OPSD</td><td>45.68</td><td>45.71</td><td>46.42</td><td>46.61</td></tr></table>

## B Implementation Details

Divergence. With teacher distribution q and student distribution p at a response position, both computed from logits divided by the sampling temperature 1.1,

$$
D _ { \tau } ( p , q ) = \sum _ { v } \operatorname* { m i n } \bigl ( q _ { v } \log ( q _ { v } / p _ { v } ) , \tau \bigr ) ,
$$

the forward KL with each vocabulary entry’s contribution clipped at τ (0.05 for 1.7B and 4B, 0.06 for 8B), as in the OPSD implementation. The loss averages D over valid response tokens.

Row packing. Micro-batches $1 , \ldots , k - 1$ of an accumulation window apply the selection rule and pass their retained rows on. The last micro-batch packs all retained rows into micro-batches of four, left-padded to a common prompt width. Each token is weighted by $1 / N _ { j }$ , where $N _ { j }$ is the number of retained tokens in its original micro-batch j, so the accumulated gradient is the same as without packing. A micro-batch with no retained rows contributes nothing.

Baselines. SFT and GRPO follow the setup of the OPSD paper (Zhao et al., 2026), with the same 100-step budget and checkpoint selection as the OPSD arms.

Gradient measurements. The parameters in Table 2 average 64 trials at group size 4; the out-of-sample measurements average 8, 6, and 2 trials at group sizes 16, 64, and 218; each training-stability measurement averages 8 trials at group size 16. Every trial draws four disjoint groups.