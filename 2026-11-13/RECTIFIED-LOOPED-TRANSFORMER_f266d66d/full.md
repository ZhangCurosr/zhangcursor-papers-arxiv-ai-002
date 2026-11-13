# RECTIFIED LOOPED TRANSFORMER

Musheng Li Jiawei Zhang<sup>†</sup> Ziyuan Liu Yuantao Gu<sup>‡</sup>

Tsinghua University

lms21@mails.tsinghua.edu.cn

## ABSTRACT

Looped Transformers increase effective depth by repeatedly applying a shared block, offering a parameter-efficient approach to language modeling. However, most existing works directly reuse each loop’s output as the next’s input, which can lead to hidden-state scale growth and perturbation amplification as recurrence deepens. We propose Rectified Looped Transformer (RLT), which redesigns feedback between loops using an interpolation schedule inspired by flow-based modeling. At each loop, RLT forms the next input by linearly interpolating between the current state and its prediction. Such interpolation distributes state changes across loops, encouraging a smoother trajectory from the initial input to the final output. Theoretical analysis proves that, under appropriate assumptions, RLT keeps state magnitudes uniformly bounded across loops and reduces the worstcase perturbation amplification. We evaluate RLT on language modeling corpora and downstream benchmarks. At matched training and inference depths, RLT consistently achieves lower perplexity than a standard looped Transformer, while exhibiting greater robustness to input perturbations and variance in inference depth. Consistent with the theoretical results, latent trajectory analysis shows that RLT maintains more stable hidden-state norms across loops, which also facilitates extrapolation-based decoding strategies. Notably, RLT achieves these gains without requiring additional trainable parameters or extra recurrent block evaluations, thus providing a simple yet more stable feedback mechanism for looped Transformers.

## 1 INTRODUCTION

Improving language model capabilities while controlling parameter count is an important challenge in model scaling. Looped Transformers address this challenge by repeatedly applying a shared stack of layers, increasing effective computational depth without increasing the number of physical parameters. This makes recurrence depth an additional axis for scaling language models. Early work (Saunshi et al., 2025) establishes the reasoning potential of looped Transformers through controlled experiments and theoretical analysis. Huginn (Geiping et al., 2025) demonstrates their feasibility at scale, while Ouro (Zhu et al., 2025) further scales pretraining to trillions of tokens, achieving performance competitive with larger standard LLMs and gains on tasks requiring multi-hop reasoning. Together, these advances establish latent recurrent computation as a promising route to more capable, parameter-efficient language models.

A key design choice in recurrent computation is how each loop’s output becomes the next loop’s input. GRUs (Cho et al., 2014) use interpolation between existing and candidate states for effective sequence modeling (Chung et al., 2014); Thinking with Looped Flows (Suleymanzade et al., 2026) couples answer-space flow updates with recurrence to improve structured reasoning. In the directfeedback formulation (hereafter referred to as Standard Looped Transformer, SLT) commonly used in current looped language models, including Ouro (Zhu et al., 2025), Loopie (Gao et al., 2026), and LoopFormer (Jeddi et al., 2026), the network output fully replaces the current input, without retaining it as a separate component of the feedback update. From a dynamical systems perspective, this applies the entire change from the current state to the network prediction at every loop, coupling prediction with state advancement. As the shared network is repeatedly applied, these changes can accumulate into hidden-state scale growth and perturbation amplification. STARS (Yang et al., 2026)

![](images/625221a45c11a5dd05c25562c67cb36362f53b6ee19e382a2631bc7b15fe0de7.jpg)  
Figure 1: Overview of Rectified Looped Transformer (RLT). (a) SLT and RLT share the same looped architecture and uniform all-step supervision. RLT replaces direct feedback with interpolation between the current input and the predicted endpoint. (b) Schematic latent dynamics under matched and mismatched loop counts, illustrating more uniform state advancement and more stable final predictions in RLT, while SLT may fall short or drift beyond the target region.

links performance degradation at larger recurrent depths to unstable latent dynamics; our SLT likewise exhibits sensitivity to loop count, accompanied by pronounced changes in state scale. This raises a concrete question: can retaining the current input state and controlling its advance toward each new prediction yield more effective and stable recurrent computation?

To address this gap, we propose Rectified Looped Transformer (RLT; Figure 1(a)), which organizes recurrent feedback through linear interpolation inspired by rectified flow (Liu et al., 2023). RLT treats the network output at each loop as an endpoint prediction and combines it with the current input state to form the next input. The interpolation weight on the endpoint prediction increases as the remaining number of loops decreases. Intuitively, each loop re-estimates the endpoint and distributes the remaining displacement over the remaining steps. This provides a geometric intuition of approximately linear progress toward a shared target region, while allowing the endpoint estimate to evolve across loops (Figure 1(b)). The design separates prediction from state advancement and explicitly organizes how network updates accumulate across loops. Our theory establishes depthuniform bounds on feedback-state and output magnitudes under time-dependent residual control. Even when the network can amplify perturbations, RLT reduces the worst-case amplification bound for initial perturbations from exponential to polynomial in depth under a uniform sensitivity bound.

Empirically, with identical uniform all-step supervision, RLT consistently achieves lower perplexity than the direct-feedback SLT when training and inference depths are matched. Besides, experiments show greater robustness to perturbations and changes in inference depth, including inference with both fewer and more loops than used during training. Trajectory analyses reveal differences in scale accumulation and prediction evolution, while improved scale stability also benefits contrastive decoding and guidance. Notably, these gains require only elementwise interpolation, with no additional trainable feedback parameters or recurrent-block forward passes, highlighting the efficiency of RLT’s feedback design.

Our main contributions are as follows:

• We introduce Rectified Looped Transformer (RLT), which interprets each loop’s output as an endpoint prediction and organizes latent-state feedback through a remaining-time interpolation schedule.

• We theoretically analyze state and prediction magnitudes and perturbation amplification, establishing depth-uniform magnitude bounds under time-dependent residual control and polynomial bounds on initial-perturbation amplification under a uniform Lipschitz condition.

• We empirically demonstrate improvements in language modeling and robustness to changes in loop count and numerical perturbations, supported by latent-trajectory analyses and extrapolation-based decoding experiments.

## 2 RELATED WORK

Looped Transformers for Language Modeling. Looped Transformers repeatedly apply a shared stack of layers, increasing effective computational depth without adding physical parameters. Early architectures such as Universal Transformers (Dehghani et al., 2019) and ALBERT (Lan et al., 2020) established parameter sharing across depth, while subsequent theoretical and empirical studies demonstrated its potential for algorithm execution (Giannou et al., 2023) and reasoning (Saunshi et al., 2025). Recently, Huginn (Geiping et al., 2025), Ouro (Zhu et al., 2025), and Loopie (Gao et al., 2026) have extended this paradigm to large-scale language modeling, achieving performance competitive with larger standard LLMs on commonsense and reasoning benchmarks while using fewer physical parameters. Existing efforts seek to improve the performance and efficiency of looped language models through sparse experts (Gao et al., 2026; Chen et al., 2026; Lee et al., 2026), depth-specific adapters (Bae et al., 2025a), and adaptive allocation of recurrent computation (Bae et al., 2025b; Fu et al., 2026; You et al., 2026), etc. In this work, we focus on state feedback and propose RLT to improve language modeling and recurrent stability.

State Feedback and Recurrent Stability. State feedback determines how each loop’s output becomes the next input. The most common practice in looped Transformers is direct feedback, which reuses the output as the next recurrent state, as in Ouro (Zhu et al., 2025), Loopie (Gao et al., 2026), and LoopFormer (Jeddi et al., 2026). Recent work has begun to explore feedback control by scaling the residual update between the current state and the network output. ELR (Ladhak et al., 2026) seeks to preserve one-loop performance while benefiting from additional loops, using learned gates to scale residual updates; TAPS (Wang et al., 2026) aims to improve recurrent inference efficiency by adaptively adjusting update magnitudes according to persistent progress and fluctuation estimated from the trajectory. Recent studies such as STARS (Yang et al., 2026) also reveal that looped Transformers can become unstable as recurrence deepens. To stabilize these dynamics, Fixed-Point Reasoners (Movahedi et al., 2026) use damped feedback to suppress oscillation and promote convergence toward fixed points. Our RLT uses a predefined remaining-time interpolation schedule, with theoretical analysis establishing tighter control over state magnitudes and perturbation amplification than direct feedback.

Flow-based Modeling and Iterative Reasoning. Flow-based generative modeling (Lipman et al., 2023) connects individual state updates to continuous transport between distributions through learned ODEs. Rectified Flow (Liu et al., 2023) uses linear interpolation between source and target samples as training paths, encouraging straighter flows that support sampling with fewer steps. Recent work combines looped computation with flow-based generation. In structured reasoning tasks such as Sudoku and maze solving, Thinking with Looped Flows (Suleymanzade et al., 2026) couples a noise-to-answer flow with recurrent hidden states and local denoising supervision; Unrolled Flow Models (Izermine et al., 2026) use a DiT (Peebles & Xie, 2023) backbone to learn flow-based answer generation through their own latent rollouts with a terminal prediction loss. In contrast, our RLT applies flow-inspired interpolation to state feedback in autoregressive language models, combining each loop’s output with its input state to form the next input.

## 3 RECTIFIED LOOPED TRANSFORMER

## 3.1 PRELIMINARIES: LOOPED COMPUTATION AND STATE FEEDBACK

As illustrated in Figure 1(a), a looped Transformer repeatedly applies a shared stack of layers to the input sequence’s representation. We focus on the recurrent computation for predicting the next token from a fixed input context. The initial input state at the prediction position is $\mathbf { x } _ { 0 } \in \mathbb { R } ^ { d }$ , formed

from token and positional embeddings, where d is the hidden size. We associate $K \geq 1$ loops with a uniform time grid on [0, 1]:

$$
\Delta t = \frac { 1 } { K } , \qquad t _ { k } = \frac { k } { K } , \qquad k = 0 , \ldots , K - 1 .\tag{1}
$$

The shared Transformer stack $\mathbf { F } _ { \theta }$ performs one loop of computation and receives time and step-size conditioning through AdaLN-style modulation (Peebles & Xie, 2023). For brevity, we suppress its step-size argument and conditioning on the current prefix representations below. At loop $k , \mathbf { x } _ { k }$ denotes the input state and $\mathbf { h } _ { k + 1 }$ the hidden output. A feedback coefficient $\eta _ { k }$ determines their contribution to the next input:

$$
\mathbf h _ { k + 1 } = \mathbf F _ { \theta } ( \mathbf x _ { k } , t _ { k } ) , \qquad \mathbf x _ { k + 1 } = ( 1 - \eta _ { k } ) \mathbf x _ { k } + \eta _ { k } \mathbf h _ { k + 1 } , \qquad 0 \le \eta _ { k } \le 1 .\tag{2}
$$

Standard Looped Transformer (SLT) uses direct feedback, setting $\eta _ { k } = 1$ and directly feeding the output $\mathbf { h } _ { k + 1 }$ back as the next input state. Parameters are shared across loops and held fixed when varying inference depth.

The shared readout $\mathcal { D }$ acts on each hidden output:

$$
\boldsymbol { \mathcal { D } } ( \mathbf { h } ) = \mathbf { W } \operatorname { R M S N o r m } ( \mathbf { h } ) , \qquad \boldsymbol { \ell _ { k } } = \operatorname { C E } ( \boldsymbol { \mathcal { D } } ( \mathbf { h } _ { k } ) , \mathbf { y } ) ,\tag{3}
$$

where W is the vocabulary projection, y denotes the next-token target, and CE is the cross-entropy loss. In this work, the looped Transformer receives uniform all-step supervision with training loss $K ^ { - 1 } \sum _ { k = 1 } ^ { K } \ell _ { k }$ . Inference decodes the final hidden output h $^ { 1 } K$

## 3.2 FLOW-INSPIRED STATE FEEDBACK

Inspired by linear interpolation in rectified flow (Liu et al., 2023), we propose Rectified Looped Transformer (RLT), which treats each loop’s network output as an endpoint prediction and separates prediction from state advancement. At time $t _ { k }$ , the displacement from the current state to the predicted endpoint is $\mathbf { h } _ { k + 1 } - \mathbf { x } _ { k }$ , and the remaining time is $1 - t _ { k }$ . Advancing by one time step gives

$$
\mathbf { x } _ { k + 1 } = \mathbf { x } _ { k } + \Delta t { \frac { \mathbf { h } _ { k + 1 } - \mathbf { x } _ { k } } { 1 - t _ { k } } } .
$$

The feedback weight is therefore the fraction of the remaining time covered by one step:

$$
\eta _ { k } = \frac { \Delta t } { 1 - t _ { k } } = \frac { 1 } { K - k } , \qquad k = 0 , \dots , K - 1 .\tag{4}
$$

The weight increases from $\eta _ { 0 } = 1 / K \mathrm { t o } \eta _ { K - 1 } = 1$ , forming a smoothly interpolated trajectory from $\mathbf { x } _ { \mathrm { 0 } }$ to the final prediction ${ \bf x } _ { K } = { \bf h } _ { K }$ . Intuitively, each loop re-estimates the endpoint and distributes the remaining displacement over the remaining loops. When endpoint estimates agree, the rule yields uniformly spaced states along the line from the initial state to the endpoint (Figure 1(b)).

We next establish stability guarantees for RLT by bounding state and output norms and initialperturbation amplification as loop depth increases.

## 3.3 THEORETICAL ANALYSIS OF STABILITY

We establish depth-uniform bounds on state and hidden-output norms and polynomial bounds on initial-perturbation amplification for the RLT schedule. Throughout, $\| \cdot \|$ and $\langle \cdot , \cdot \rangle$ denote the Euclidean norm and inner product. Constants in depth-uniform bounds, including those hidden by $\lesssim$ are independent of K. Theorem 1 below bounds state norms under general feedback schedules.

Theorem 1. Suppose there exist $M \geq 0$ and $0 < \alpha \leq 1$ such that, for all $0 \leq t < 1$

$$
\operatorname* { s u p } _ { \mathbf { x } \in \mathbb { R } ^ { d } } \left\| \mathbf { F } _ { \theta } ( \mathbf { x } , t ) - \mathbf { x } \right\| \leq M ( 1 - t ) ^ { \alpha } .\tag{5}
$$

Then thefeedback rule in Eq. (2) satisfies

$$
\| \mathbf { x } _ { k } \| \leq \| \mathbf { x } _ { 0 } \| + M \sum _ { j = 0 } ^ { k - 1 } \eta _ { j } ( 1 - t _ { j } ) ^ { \alpha } , \qquad 1 \leq k \leq K .\tag{6}
$$

Remark. For $\alpha = 1$ , Eq. (5) is equivalent to $\begin{array} { r } { \operatorname* { s u p } _ { \mathbf { x } , t } \| \mathbf { v } _ { \theta } ( \mathbf { x } , t ) \| \leq M } \end{array}$ for the flow-inspired velocity field

$$
\mathbf { v } _ { \theta } ( \mathbf { x } , t ) = \frac { \mathbf { F } _ { \theta } ( \mathbf { x } , t ) - \mathbf { x } } { 1 - t } .\tag{7}
$$

For $0 < \alpha < 1$ , the condition allows $\begin{array} { r } { \operatorname* { s u p } _ { \mathbf x } \| \mathbf v _ { \theta } ( \mathbf x , t ) \| \le M ( 1 - t ) ^ { \alpha - 1 } } \end{array}$ , relaxing uniform velocity boundedness to permit integrable growth near $t = 1$

In practice, we observe pronounced norm growth under SLT’s direct feedback, while RLT maintains more stable norms (Section 4.2). The following two corollaries characterize the different depth dependence of these feedback rules.

Corollary 1.1. Under the assumptions of Theorem 1, SLT’s direct feedback $( \eta _ { k } \equiv 1 )$ gives

$$
\left\| \mathbf { x } _ { K } \right\| \leq \left\| \mathbf { x } _ { 0 } \right\| + M \underbrace { K ^ { - \alpha } \sum _ { j = 1 } ^ { K } j ^ { \alpha } } _ { \longrightarrow \infty } .\tag{8}
$$

Corollary 1.2. Under the assumptions of Theorem 1, the RLT schedule in Eq. (4) gives

$$
\operatorname* { m a x } \left\{ \operatorname* { m a x } _ { 0 \leq k \leq K } \| \mathbf { x } _ { k } \| , \operatorname* { m a x } _ { 1 \leq k \leq K } \| \mathbf { h } _ { k } \| \right\} \leq \underbrace { \| \mathbf { x } _ { 0 } \| + \frac { M } { \alpha } } _ { c o n s t a n t i n K } .\tag{9}
$$

These corollaries show that SLT’s direct feedback permits unbounded state-norm growth with depth (Appendix A.1), whereas the RLT schedule keeps both input-state and hidden-output norms uniformly bounded as loop depth increases. Under an additional radial condition on the residual, we obtain the following stronger bound.

Corollary 1.3. Under the assumptions of Theorem 1 and the RLT schedule, suppose $\left. \mathbf { x } _ { k } , \mathbf { h } _ { k + 1 } - \right.$ $\mathbf { x } _ { k } \rangle \le 0 \dot { f } o r 0 \le k < K$ . Then

$$
\operatorname* { m a x } _ { 0 \leq k \leq K } \| { \bf x } _ { k } \| ^ { 2 } \leq \| { \bf x } _ { 0 } \| ^ { 2 } + M ^ { 2 } \underbrace { K ^ { - 2 \alpha } \sum _ { j = 1 } ^ { K } j ^ { 2 \alpha - 2 } } _ { \longrightarrow 0 ~ ( K  \infty ) } .\tag{10}
$$

The radial condition makes the first-order contribution to squared norm growth nonpositive, while the accumulated squared update magnitudes vanish with depth. Proofs of Theorem 1 and its corollaries are provided in Appendix A.1.

We next examine how perturbations to the initial state $\mathbf { x } _ { \mathrm { 0 } }$ propagate across loops. The following theorem bounds the separation between two trajectories under a Lipschitz condition on the shared network.

Theorem 2. Suppose $\mathbf { F } _ { \theta } ( \mathbf { x } , t _ { k } )$ is $L _ { k }$ -Lipschitz in x, with $L _ { k } \ge 0 . \ { \cal I } f { \bf x } _ { k }$ and $\widetilde { \mathbf { x } } _ { k }$ follow Eq. (2) with the same contextual conditioning andfeedback schedule, then

$$
\| \mathbf { x } _ { k } - \widetilde { \mathbf { x } } _ { k } \| \leq \prod _ { j = 0 } ^ { k - 1 } ( 1 - \eta _ { j } + \eta _ { j } L _ { j } ) \| \mathbf { x } _ { 0 } - \widetilde { \mathbf { x } } _ { 0 } \| , \qquad 1 \leq k \leq K .\tag{11}
$$

Theorem 2 bounds trajectory separation through a product of per-loop factors, relating the difference between terminal states to the initial perturbation. A uniform Lipschitz bound $L \ > \ 1$ allows perturbations to be amplified across loops. The following corollary compares the resulting depth dependence under the feedback rules of RLT and SLT.

Corollary 2.1. Under the assumptions of Theorem $^ { 2 , }$ suppose $L _ { k } \leq L$ uniformly in k and $K ,$ with $L > 1$ . Then SLT’s directfeedback $( \eta _ { k } \equiv 1 )$ gives

$$
\begin{array} { r } { \| \mathbf { x } _ { K } - \widetilde { \mathbf { x } } _ { K } \| \lesssim L ^ { K } \| \mathbf { x } _ { 0 } - \widetilde { \mathbf { x } } _ { 0 } \| . } \end{array}\tag{12}
$$

For the RLT schedule,

$$
\begin{array} { r } { \| \mathbf { x } _ { K } - \widetilde { \mathbf { x } } _ { K } \| \lesssim K ^ { L - 1 } \| \mathbf { x } _ { 0 } - \widetilde { \mathbf { x } } _ { 0 } \| . } \end{array}\tag{13}
$$

Corollary 2.1 shows that the RLT schedule reduces the depth dependence of the worst-case amplification bound from exponential to polynomial, providing tighter control over sensitivity to initial perturbations. Since ${ \bf x } _ { K } = { \bf h } _ { K }$ , it also controls the sensitivity of RLT’s final hidden output. Proofs of Theorem 2 and Corollary 2.1 are provided in Appendix A.2.

For a time-independent nonexpansive network, recurrent computation can also be viewed as fixedpoint iteration (Movahedi et al., 2026). RLT then follows the classical Krasnosel’skii–Mann iteration (Cominetti et al., 2014), yielding the following fixed-point residual bound.

Proposition 3. Suppose $\mathbf { F } _ { \theta } ( \mathbf { x } , t ) = \mathbf { F } _ { \theta } ( \mathbf { x } )$ is nonexpansive on $\mathbb { R } ^ { d }$ and $\operatorname { F i x } ( \mathbf { F } _ { \theta } ) : = \{ \mathbf { x } \in \mathbb { R } ^ { d }$ $\mathbf { F } _ { \theta } ( \mathbf { \bar { x } } ) = \mathbf { x } \} \neq \emptyset$ . Then,for the RLT schedule and $K > 2 ,$

$$
\| \mathbf { F } _ { \theta } ( \mathbf { x } _ { K } ) - \mathbf { x } _ { K } \| \lesssim \frac { \mathrm { d i s t } ( \mathbf { x } _ { 0 } , \mathrm { F i x } ( \mathbf { F } _ { \theta } ) ) } { \sqrt { \log K } } ,\tag{14}
$$

where dist denotes distancefrom a point to a set.

Proposition 3 follows from Cominetti et al. (2014, Proposition 11); a brief proof appears in Appendix A.3. Together, these results connect RLT’s remaining-time feedback rule to controlled state scales, reduced perturbation amplification bounds, and fixed-point residual decay. These properties provide theoretical support for the stability of RLT’s recurrent computation.

## 4 EXPERIMENTS

We compare SLT and RLT in language modeling, recurrent stability, and latent-state dynamics.   
Matched architectures and training settings allow us to examine the effect of state feedback.

Following LoopFormer (Jeddi et al., 2026), we adopt a GPT-2-style looped Transformer conditioned on time and step size (Figure 1(a)). Each loop maps its input state to a hidden output through the shared Transformer stack. SLT feeds this hidden output back as the next loop’s input state, whereas RLT constructs the next input by interpolating between the current input state and the hidden output using a remaining-time schedule. These feedback rules are formalized in Sections 3.1–3.2. Our default configuration is 3 ⊗ 8: three shared Transformer layers applied over eight loops, with 278M physical parameters. We denote the training and inference loop counts by $K _ { \mathrm { t r a i n } }$ and $K _ { \mathrm { t e s t } }$ respectively. Unless otherwise specified, inference follows the uniform time grid defined in Eq. (1), with $t _ { k } = k / K _ { \mathrm { t e s t } }$ and $\Delta t = 1 / K _ { \mathrm { t e s t } }$

Both models are trained from scratch on the deduplicated version of The Pile (Gao et al., 2020), with the training data comprising 6.88B tokens and uniform all-step next-token prediction supervision. We train for 14,000 optimizer steps with a peak learning rate of $6 \times 1 0 ^ { \dot { - } 4 }$ and a global batch size of 480 sequences of 1,024 tokens. We measure perplexity on The Pile (Gao et al., 2020), FineWeb-Edu (FWE) (Penedo et al., 2024), and OpenWebText (OWT) (Gokaslan & Cohen, 2019), and evaluate zero-shot accuracy following lm-eval harness (Gao et al., 2023). Our downstream evaluation covers ten well-established benchmarks spanning commonsense reasoning, reading comprehension, scientific question answering, and discourse completion: COPA (Roemmele et al., 2011), HellaSwag (HS) (Zellers et al., 2019), LAMBADA (LB) (Paperno et al., 2016), OpenBookQA (OBQA) (Mihaylov et al., 2018), PIQA (Bisk et al., 2020), RACE (Lai et al., 2017), SciQ (Welbl et al., 2017), ARC (Clark et al., 2018), Social IQA (SIQA) (Sap et al., 2019), and WinoGrande (WG) (Sakaguchi et al., 2021). We report accuracy or normalized accuracy as appropriate, averaging ARC-Easy and ARC-Challenge before computing the ten-task mean.

Further details on the model architecture, training hyperparameters, and datasets are provided in Appendices C, D, and E, respectively.

## 4.1 LANGUAGE MODELING PERFORMANCE

We first compare SLT and RLT at matched training and inference depths, $K _ { \mathrm { t r a i n } } = K _ { \mathrm { t e s t } } = K \in$ {2, 4, 8}. At each depth, both models use three shared Transformer layers, identical training settings, and uniform all-step supervision. Table 1 shows that RLT consistently achieves lower perplexity across all three depths and all three evaluation corpora. At $K = 4$ , it reduces PPL by 4.2%, 5.5%, and 5.2% on Pile, FWE, and OWT, respectively. This advantage also holds at $K = 2$ and $K = 8$

Table 1: Language modeling and downstream performance at matched depth $K = K _ { \mathrm { t r a i n } } = K _ { \mathrm { t e s t } }$ Structure denotes shared layers $\otimes$ loops. Accuracy is reported in percentages; Avg Acc is the unweighted mean across ten benchmarks. Bold marks the better result within each depth group.
<table><tr><td rowspan="2">Variants Structure</td><td rowspan="2"></td><td>Perplexity ↓</td><td colspan="2"></td><td colspan="7">Accuracy  $( \% ) \uparrow$ </td></tr><tr><td>Pile FWE OWT</td><td>COPA</td><td>HS</td><td>LB</td><td>OBQA PIQA RACE</td><td></td><td>SciQ</td><td></td><td>ARC SIQA WG</td><td>|Avg Acc</td></tr><tr><td colspan="10"> $\pmb { K } = \mathbf { 8 }$ </td></tr><tr><td>SLT</td><td>388</td><td>11.0827.5825.39</td><td></td><td>64.00 36.50 33.00</td><td>29.50</td><td>62.50</td><td>28.57</td><td></td><td></td><td>66.50 30.22 36.9255.91</td><td>44.36</td></tr><tr><td>RLT</td><td>388</td><td>10.82 26.74 24.61</td><td></td><td>62.0036.50 34.00</td><td>28.00</td><td>60.87</td><td>29.52</td><td></td><td></td><td>70.50 31.3138.4644.09</td><td>43.53</td></tr><tr><td colspan="10"></td></tr><tr><td>SLT</td><td></td><td>11.7629.7627.30</td><td></td><td></td><td> $K = 4$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RLT</td><td>3⊗4 3⊗4</td><td>11.27 28.14 25.88</td><td></td><td>64.00 35.50 26.00 68.00 35.50 31.00</td><td>27.50 29.00</td><td>59.78 60.33</td><td>26.67 23.81</td><td></td><td></td><td>66.5032.3837.4448.03 65.50 33.0134.8751.97</td><td>42.38 43.30</td></tr><tr><td colspan="10"></td></tr><tr><td>SLT</td><td></td><td></td><td></td><td></td><td>K = 2</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RLT</td><td>3⊗2 3⊗2</td><td>12.4932.2429.52 12.44 32.04 29.33</td><td></td><td>64.0035.50 23.50 60.00 35.00 28.50</td><td>27.50 25.50</td><td>59.24 60.87</td><td>20.95 22.86</td><td></td><td></td><td>65.50 31.94 39.4948.03 63.00 29.81 39.49 48.03</td><td>41.56 41.31</td></tr></table>

![](images/b6344883dabae978e0cebd2fa76173fc545edf049a5083c40fbc7f363075c1b3.jpg)

![](images/0ade31d6e97324dba4fa1845dcb3397536edb9d60cccbb7c4ce77981e31cd8f0.jpg)  
Figure 2: Cross-loop hidden-state scales relative to the first-loop output $\mathbf { h } _ { 1 }$ (left) and the initial embedding input $\mathbf { x } _ { \mathrm { 0 } }$ (right), with $K _ { \mathrm { t r a i n } } = K _ { \mathrm { t e s t } } = 8$

On downstream benchmarks, the two methods achieve comparable average accuracy, with taskspecific differences across depths. At K = 4, RLT increases the ten-task average from 42.38% to 43.30%. LAMBADA accuracy improves at all three depths, with gains of $5$ percentage points at both $K = 2$ and $K = 4$ . These results show that reorganizing state feedback consistently improves language modeling while maintaining competitive downstream performance. We next examine how this feedback design affects the numerical behavior of recurrent computation.

## 4.2 RECURRENT STABILITY AND ROBUSTNESS

We first examine hidden-state scales and responses to initial-input and recurrent perturbations to study RLT’s recurrent stability. We then assess the practical effects of these properties through contrastive decoding and guidance, followed by an evaluation of performance under quantization.

Hidden-state scale. We measure the scale of each loop’s output $\mathbf { h } _ { i }$ before the final RMS normalization. Figure 2 compares the token-wise RMS ratios relative to the first-loop output $\mathbf { h } _ { 1 }$ and the initial embedding input $\mathbf { x } _ { \mathrm { 0 } }$ . SLT exhibits pronounced scale accumulation over the early loops: by loop eight, the mean ratio relative to $\mathbf { h } _ { 1 }$ reaches 5.08, compared with 0.65 for RLT. The comparison relative to $\mathbf { x } _ { \mathrm { 0 } }$ likewise shows substantial scale accumulation in SLT, while RLT maintains a more controlled scale across loops. These observations are consistent with the output-magnitude bound in Corollary 1.2. Measurement details are provided in Appendix F.1.

![](images/85a2ed1863d8f250f87d300ab3575dac3e6749425ca9ed6c111cee702896bdb3.jpg)  
Figure 3: Final-output displacement after injecting Gaussian noise into the initial embedding input $\mathbf { x } _ { \mathrm { 0 } }$ . Distances are averaged over tokens and measured before RMS normalization.

![](images/be869ffbad1a500868a9b3c3f7d24ddc754eac2a1f17933a2629f01df36764e7.jpg)  
Figure 4: Performance under recurrent perturbations applied to h before feedback (top) or x after feedback (bottom). ∆CE and ∆Acc are changes relative to each model’s clean baseline.

Robustness to perturbations. We first inject RMS-scaled Gaussian noise into the initial embedding input x and measure the resulting displacement of the final hidden output. Across all three corpora and all tested noise levels, RLT exhibits smaller absolute output displacements (Figure 3), in line with the initial-perturbation analysis in Theorem 2.

We then inject noise at every loop, either into the block output h before feedback or into the state x after feedback. As shown in Figure 4, RLT better preserves predictive performance overall under both schemes, with larger benefits for output perturbations. At $\rho = 0 . 5$ , output perturbations increase Pile cross-entropy by 0.61 nats/token for RLT versus 2.86 for SLT. The dependence on injection location reflects the role of interpolation: output noise passes through the feedback coefficient, whereas noise added after feedback enters the next loop directly. See Appendix F.2 for experimental details.

Loop-wise contrastive decoding. Contrastive decoding (Li et al., 2023) exploits differences between stronger and weaker predictions to improve downstream task performance; related methods in looped Transformers (Yu et al., 2026; Liu et al., 2026) construct these contrasts from different recurrent depths. We apply hidden-state extrapolation before the final readout, using either native intermediate outputs $\mathbf { h } _ { i }$ or predictions $\hat { \mathbf { h } } _ { i }$ from an auxiliary MLP branch. The branch predicts the complete block output from its first layer’s representation and is trained with relative mean squared error (see Appendix F.3 for details).

![](images/436cd333ec9aa81ff0d12e352448ff7c01fe4c3d6dfbd27952fbf7bfd5d24343.jpg)  
Figure 5: Loop-wise contrastive decoding with native outputs h<sub>i</sub> (top) or MLP predictions h<sup>ˆ</sup><sub>i</sub> (bottom). References retain their original scale (left) or are RMS-aligned to h<sub>K</sub> (right). Color intensity indicates extrapolation strength λ; horizontal dashed lines mark unguided baselines.  
Figure 6: Guidance applied over different loop intervals for RLT and SLT. Colors indicate the guided loops, and dashed lines mark unguided baselines.

Using references from the first three loops, RLT obtains consistently larger accuracy gains than SLT across both reference types and RMS-alignment settings. As shown in Figure 5, RLT benefits more from early references without RMS alignment, whereas SLT shows little or no improvement. Aligning reference magnitudes to the final output makes SLT’s early references more effective: with the first-loop reference and $\lambda = 0 . 5 .$ , accuracy increases from 42.79% to 43.35%. RLT also benefits from MLP references, reaching 44.45% from a 42.73% baseline with the third-loop reference at $\lambda = 0 . 5$ without alignment. These findings highlight how RLT’s more stable representation scales support the effective use of intermediate predictions and improve contrastive decoding.

Guidance during recurrence. We further examine guidance within the recurrent computation, drawing on autoguidance with a weaker version of the model (Karras et al., 2024). At selected loops, we extrapolate the block output away from the RMSaligned MLP prediction and feed the result into subsequent computation. See Appendix F.4 for details. Guiding loops 3–5 improves RLT at every tested strength shown in Figure 6, reaching a gain of 0.76 percentage points at $\lambda ~ = ~ 0 . 3$ In SLT, guidance over loops 1–3 or 1–7 causes substantial degradation as strength increases. RLT can therefore benefit from interventions during recurrence, with the middle loops providing an effective guidance interval.

![](images/fc309722c9690d5f89bdcb6c857dfd83a107902960ac3ac60289c3e2b496e936.jpg)  
Figure 7: Perplexity under quantization at different inference depths. Rows and columns correspond to quantization settings and evaluation corpora, respectively.

<table><tr><td rowspan=2 colspan=1>Model</td><td rowspan=1 colspan=1>Perplexity ↓</td><td rowspan=1 colspan=1>Acc (%) ↑</td></tr><tr><td rowspan=1 colspan=1>Pile FWE OWT</td><td rowspan=1 colspan=1>Avg Acc</td></tr><tr><td rowspan=1 colspan=2> $K _ { \mathrm { t e s t } } = 4$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SLT</td><td rowspan=1 colspan=1>13.01 33.62 30.87</td><td rowspan=1 colspan=1>42.95</td></tr><tr><td rowspan=1 colspan=1>RLT</td><td rowspan=1 colspan=1>11.76 29.56 27.06</td><td rowspan=1 colspan=1>41.96</td></tr><tr><td rowspan=1 colspan=3> $K _ { \mathrm { t e s t } } = 2$ </td></tr><tr><td rowspan=1 colspan=1>SLT</td><td rowspan=1 colspan=1>21.60 69.18 62.04</td><td rowspan=1 colspan=1>39.24</td></tr><tr><td rowspan=1 colspan=1>RLT</td><td rowspan=1 colspan=1>13.85 35.82 32.81</td><td rowspan=1 colspan=1>40.76</td></tr></table>

Table 2: Fewer-loop inference with $K _ { \mathrm { t r a i n } } = 8 .$ Bold marks the better result at each depth; Avg Acc averages ten benchmarks.  
![](images/f8b62140f38440f55b4497eebdacecb7226318d6d1b57476707ba4769ac57169.jpg)  
Figure 8: Performance across inference depths for models trained with $K _ { \mathrm { t r a i n } } = 8 .$ . The dashed line marks the training depth, and the shaded region indicates larger inference budgets.

Robustness under quantization. Finally, we evaluate W8A16, W4A16, and W8A8 quantization at inference depths of 2, 4, and 8 loops. Figure 7 shows that RLT retains its perplexity advantage across all three quantization settings, inference budgets, and evaluation corpora. This extends the performance advantage to quantized computation, including reduced-loop inference. Quantization configurations and implementation details are provided in Appendix F.5.

## 4.3 LATENT DYNAMICS ACROSS INFERENCE DEPTHS

We first compare SLT and RLT across inference depths, then use PCA visualizations to examine the latent dynamics underlying their performance differences. Both models are trained with eight loops and evaluated at smaller and larger inference budgets.

Adaptation to inference depth. We first reduce the inference budget while keeping the trained weights fixed. Table 2 shows that RLT achieves lower perplexity on all three corpora at lower budgets. At $K _ { \mathrm { t e s t } } = 2 .$ , it reduces PPL by 35.9%, 48.2%, and 47.1% on Pile, FWE, and OWT, respectively, while increasing average downstream accuracy from 39.24% to 40.76%. These results demonstrate that RLT better preserves language modeling quality when fewer recurrent steps are available.

We then extend the sweep to inference depths between 2 and 32 (Figure 8). The results show that RLT achieves lower Pile perplexity at every evaluated depth and exhibits a flatter performance profile around the training depth. Beyond eight loops, SLT’s perplexity rises rapidly and its downstream accuracy declines, whereas RLT tolerates the increased depth more effectively. At $K _ { \mathrm { t e s t } } = 3 2$ , RLT achieves a PPL of 17.72 versus 57.72 for SLT and an average accuracy of 42.88% versus 38.05%. RLT thus supports a broader range of inference budgets, with improved performance retention both below and above the training depth. Evaluation details are provided in Appendix G.1.

Hidden-state trajectories. To examine the latent dynamics behind these performance differences, we use 3D PCA to visualize the final token’s hidden outputs $\mathbf { h } _ { 1 } , \ldots , \mathbf { h } _ { K }$ in three examples from randomly sampled OpenWebText windows (Figure 9). Each panel overlays inference budgets $K _ { \mathrm { t e s t } } \in$ {4, 6, 8, 12, 24} in a common projection, with the matched-depth endpoint at $K _ { \mathrm { t e s t } } = \overline { { K } } _ { \mathrm { t r a i n } } = 8$ highlighted as a reference. Trajectory extraction, PCA construction, and visualizations of additional windows are provided in Appendix G.2.

Figure 9 clearly shows that RLT dynamically adjusts its state-update step sizes using time information as the inference budget changes, while keeping its final hidden outputs concentrated around the matched-depth endpoint. Several RLT trajectories near the training depth also exhibit more uniform step-to-step changes than their SLT counterparts. In SLT, fewer loops fall short of the matcheddepth endpoint, whereas additional loops drift beyond it and substantially expand the trajectory. For example, the terminal PC1 coordinate in Window 2 increases from 31.2 at $K _ { \mathrm { t e s t } } = 4$ to 62.5 at $K _ { \mathrm { t e s t } } = 8$ and 388.3 at $K _ { \mathrm { t e s t } } = 2 4$ , reaching over six times the matched-depth value. These contrasting behaviors agree with the schematic dynamics in Figure 1(b) and connect RLT’s greater tolerance of inference-depth changes with more controlled representation dynamics. They also align with RLT’s remaining-time feedback schedule, which adjusts the amount of state advancement to the chosen loop budget.

![](images/bf4148bbd7b8faf9633636ae5ba1728825beaecb8c7e0d73a44be9eb58db23b0.jpg)  
Figure 9: 3D PCA projections of final-token hidden outputs h for three OpenWebText windows. SLT and RLT are on the top and bottom rows, respectively. Colors indicate the inference budget; stars and squares mark h<sub>1</sub> and $\mathbf { h } _ { K }$ . The outlined green square marks the matched-depth endpoint.

## 5 CONCLUSION

We introduced Rectified Looped Transformer (RLT), which improves recurrent computation through a flow-inspired feedback rule without additional trainable feedback parameters or recurrent-block forward passes. RLT interpolates between the current input state and each loop’s endpoint prediction according to the remaining time, separating prediction from state advancement. Our analysis establishes depth-uniform state and output bounds under time-dependent residual control and reduces worst-case initial-perturbation amplification bounds from exponential to polynomial in depth unde a uniform Lipschitz condition. Experiments demonstrate consistently lower perplexity at matched depths, greater robustness to perturbations and inference-depth changes, and more controlled latent dynamics than direct feedback. Improved scale stability also supports more effective contrastive decoding and guidance. Our findings highlight state feedback as an effective design choice for jointly improving language modeling and the stability of latent recurrent computation.

## REFERENCES

Sangmin Bae, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Seungyeon Kim, and Tal Schuster. Relaxed recursive Transformers: Effective parameter sharing with layer-wise LoRA. In International Conference on Learning Representations, 2025a.

Sangmin Bae, Yujin Kim, Reza Bayat, Sungnyun Kim, Jiyoun Ha, Tal Schuster, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Aaron Courville, and Se-Young Yun. Mixture-of-recursions: Learning dynamic recursive depths for adaptive token-level computation. In Advances in Neural Information Processing Systems, volume 38, pp. 96572–96617, 2025b. doi: 10.52202/085713-3229.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: Reasoning about physical commonsense in natural language. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 34, pp. 7432–7439, 2020. doi: 10.1609/aaai.v34i05.6239.

Wenkai Chen, Tianshu Li, Wenyong Huang, Yichun Yin, Lifeng Shang, and Chengwei Qin. LoopMoE: Unifying iterative computation with mixture-of-experts for language modeling. arXiv preprint arXiv:2606.04438, 2026.

Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger¨ Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder–decoder for statistical machine translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1724–1734. Association for Computational Linguistics, 2014. doi: 10.3115/v1/D14-1179.

Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Roberto Cominetti, Jose A. Soto, and Jos´ e Vaisman. On the rate of convergence of Krasnosel’ski´ ˘ı– Mann iterations and their connection with sums of Bernoullis. Israel Journal ofMathematics, 199 (2):757–772, 2014. doi: 10.1007/s11856-013-0045-4.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal Transformers. In International Conference on Learning Representations, 2019.

Tianyu Fu, Yichen You, Zekai Chen, Guohao Dai, Huazhong Yang, and Yu Wang. Think-at-hard: Dynamic looped Transformers for improved reasoning. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pp. 32129–32154. PMLR, 2026.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The Pile: An 800GB dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, December 2023.

Zitian Gao, Yilong Chen, Yihao Xiao, Xinyu Yang, Ran Tao, Joey Zhou, and Bryan Dai. Loop the Loopies! arXiv preprint arXiv:2607.16051, 2026.

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. In Advances in Neural Information Processing Systems, volume 38, pp. 41340–41391, 2025. doi: 10.52202/085713-1380.

Angeliki Giannou, Shashank Rajput, Jy-Yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papailiopoulos. Looped Transformers as programmable computers. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 11398–11442. PMLR, 2023.

Aaron Gokaslan and Vanya Cohen. OpenWebText corpus, 2019.

Faissal Izermine, Hanru Bai, Oscar Davis, and T. Konstantin Rusch. Unrolled flow models for reasoning. arXiv preprint arXiv:2610.09759, 2026.

Ahmadreza Jeddi, Marco Ciccone, and Babak Taati. LoopFormer: Elastic-depth looped transformers for latent reasoning via shortcut modulation. In The Fourteenth International Conference on Learning Representations, 2026.

Tero Karras, Miika Aittala, Tuomas Kynka¨anniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine.¨ Guiding a diffusion model with a bad version of itself. In Advances in Neural Information Processing Systems, volume 37, pp. 52996–53021. Curran Associates, Inc., 2024. doi: 10.52202/ 079017-1679.

Faisal Ladhak, Shantanu Acharya, Santiago Akle Serano, Simeng Sun, Boris Ginsburg, and Krishna C. Puvvada. One model to loop them all: Elastic latent recurrence for language models. arXiv preprint arXiv:2610.12582, 2026.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. RACE: Large-scale ReAding comprehension dataset from examinations. In Proceedings ofthe 2017 Conference on Empirical Methods in Natural Language Processing, pp. 785–794. Association for Computational Linguistics, 2017. doi: 10.18653/v1/D17-1082.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. ALBERT: A lite BERT for self-supervised learning of language representations. In International Conference on Learning Representations, 2020.

Ryan Lee, Jacob Biloki, Edward J. Hu, and Jonathan May. Sparse layers are critical to scaling looped language models. arXiv preprint arXiv:2605.09165, 2026.

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive decoding: Open-ended text generation as optimization. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12286–12312. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.acl-long.687.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

Weihao Liu, Huangjie Zheng, Tianrong Chen, Rohit Dilip, Richard He Bai, Yizhu Jiao, Yuyang Wang, and Ruixiang Zhang. Decoding looped Transformers better for (almost) free. arXiv preprint arXiv:2610.02185, 2026.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2381–2391. Association for Computational Linguistics, 2018. doi: 10.18653/v1/D18-1260.

Sajad Movahedi, Vera Milovanovic, Shlomo Libo Feigin, Alexander Theus, Thomas Hofmann,´ Valentina Boeva, T. Konstantin Rusch, and Antonio Orvieto. Fixed-point reasoners: Stable and adaptive deep looped Transformers. arXiv preprint arXiv:2606.18206, 2026.

Denis Paperno, German Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi,´ Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. The LAMBADA dataset:´ Word prediction requiring a broad discourse context. In Proceedings ofthe 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1525–1534. Association for Computational Linguistics, 2016. doi: 10.18653/v1/P16-1144.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Guilherme Penedo, Hynek Kydl´ıcek, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colinˇ Raffel, Leandro von Werra, and Thomas Wolf. The FineWeb datasets: Decanting the web for the finest text data at scale. In Advances in Neural Information Processing Systems, volume 37, pp. 30811–30849. Curran Associates, Inc., 2024. doi: 10.52202/079017-0970.

Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S. Gordon. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In AAAI Spring Symposium on Logical Formalizations ofCommonsense Reasoning, 2011.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An adversarial winograd schema challenge at scale. Communications ofthe ACM, 64(9):99–106, 2021. doi: 10.1145/3474381.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social IQA: Commonsense reasoning about social interactions. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 4463–4473. Association for Computational Linguistics, 2019. doi: 10.18653/v1/D19-1454.

Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J. Reddi. Reasoning with latent thoughts: On the power of looped transformers. In The Thirteenth International Conference on Learning Representations, 2025.

Ayhan Suleymanzade, Chanhyuk Lee, Floor Eijkelboom, Nicholas M. Boffi, <sup>˙</sup>Ismail <sup>˙</sup>Ilkan Ceylan, and Jinwoo Kim. Thinking with looped flows. arXiv preprint arXiv:2609.11801, 2026.

Boyuan Wang, Chengyao Yu, Jiaxi Ren, Hongxin Wei, Bingyi Jing, and Yuxin Tao. Scheduling recursive reasoning in looped Transformers. arXiv preprint arXiv:2609.36653, 2026.

Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Proceedings of the 3rd Workshop on Noisy User-generated Text, pp. 94–106. Association for Computational Linguistics, 2017. doi: 10.18653/v1/W17-4413.

Xiao-Wen Yang, Ziyu Han, Xi-Hua Zhang, Wen-Da Wei, Jie-Jing Shao, Lan-Zhe Guo, and Yu-Feng Li. Stabilizing recurrent dynamics for test-time scalable latent reasoning in looped language models. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pp. 144670–144682. PMLR, 2026.

Yichen You, Tianyu Fu, Aosong Feng, Xingtai Lv, Xuefei Ning, Ning Ding, and Yu Wang. Improving test-time scaling with adaptive looped Transformers. arXiv preprint arXiv:2609.35748, 2026.

Byeongho Yu, Junhyuk So, and Eunhyeok Park. LoopCD: Loop-wise contrastive decoding for improving reasoning in looped language models. In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing, pp. 31185–31199. Association for Computational Linguistics, 2026.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800. Association for Computational Linguistics, 2019. doi: 10.18653/v1/P19-1472.

Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, Lu Li, Jiajun Shi, Kaijing Ma, Shanda Li, Taylor Kergan, Andrew Smith, Xingwei Qu, Mude Hui, Bohong Wu, Qiyang Min, Hongzhi Huang, Xun Zhou, Wei Ye, Jiaheng Liu, Jian Yang, Yunfeng Shi, Chenghua Lin, Enduo Zhao, Tianle Cai, Ge Zhang, Wenhao Huang, Yoshua Bengio, and Jason Eshraghian. Scaling latent reasoning via looped language models. arXiv preprint arXiv:2510.25741, 2025.

## A PROOFS OF THEOREMS AND PROPOSITIONS

## A.1 PROOF OF THEOREM 1 AND COROLLARIES 1.1–1.3

Proof. Applying the triangle inequality gives

$$
\begin{array} { r } { \| \mathbf { x } _ { k + 1 } \| \leq \| \mathbf { x } _ { k } \| + \eta _ { k } \| \mathbf { h } _ { k + 1 } - \mathbf { x } _ { k } \| \leq \| \mathbf { x } _ { k } \| + M \eta _ { k } ( 1 - t _ { k } ) ^ { \alpha } . } \end{array}\tag{15}
$$

Summing Eq. (15) with respect to k proves Eq. (6). Our schedule (Eq. (4)) gives

$$
\| \mathbf { x } _ { k } \| \leq \| \mathbf { x } _ { 0 } \| + M \Delta t \sum _ { j = 0 } ^ { k - 1 } ( 1 - t _ { j } ) ^ { \alpha - 1 } \leq \| \mathbf { x } _ { 0 } \| + \frac { M } { \alpha } \big [ 1 - ( 1 - t _ { k } ) ^ { \alpha } \big ] .\tag{16}
$$

For $k < K$ , this further yields

$$
\left\| \mathbf { h } _ { k + 1 } \right\| \leq \left\| \mathbf { x } _ { k } \right\| + M ( 1 - t _ { k } ) ^ { \alpha } \leq \left\| \mathbf { x } _ { 0 } \right\| + { \frac { M } { \alpha } } - M \left( { \frac { 1 } { \alpha } } - 1 \right) \left( 1 - t _ { k } \right) ^ { \alpha } \leq \left\| \mathbf { x } _ { 0 } \right\| + { \frac { M } { \alpha } } .\tag{17}
$$

Thus Eq. (9) follows.

Setting $\eta _ { k } \equiv 1$ for direct feedback in Eq. (6) gives Eq. (8). We further note that the bound in Eq. (8) can be attained. Specifically, when $\mathbf { x } _ { 0 } = 0$ and $\mathbf { F } _ { \theta } ( \mathbf { x } , t ) = \mathbf { x } + M ( 1 - t ) ^ { \alpha } \mathbf { u }$ , where $\| \mathbf { u } \| = 1$ , then for $M > 0$

$$
\left\| \mathbf { x } _ { K } \right\| = M K ^ { - \alpha } \sum _ { j = 1 } ^ { K } j ^ { \alpha } \sim { \frac { M K } { \alpha + 1 } } .\tag{18}
$$

Finally, under the radial condition in Corollary 1.3, we have

$$
\begin{array} { r } { \left\| \mathbf { x } _ { k + 1 } \right\| ^ { 2 } - \left\| \mathbf { x } _ { k } \right\| ^ { 2 } = 2 \eta _ { k } \left. \mathbf { x } _ { k } , \mathbf { h } _ { k + 1 } - \mathbf { x } _ { k } \right. + \eta _ { k } ^ { 2 } \left\| \mathbf { h } _ { k + 1 } - \mathbf { x } _ { k } \right\| ^ { 2 } \leq M ^ { 2 } \eta _ { k } ^ { 2 } ( 1 - t _ { k } ) ^ { 2 \alpha } . } \end{array}\tag{19}
$$

Summing and noting that $j = K - k _ { \mathrm { { } } }$

$$
\operatorname* { m a x } _ { 0 \leq k \leq K } \left\| \mathbf { x } _ { k } \right\| ^ { 2 } \leq \left\| \mathbf { x } _ { 0 } \right\| ^ { 2 } + M ^ { 2 } \sum _ { k = 0 } ^ { K - 1 } \eta _ { k } ^ { 2 } ( 1 - t _ { k } ) ^ { 2 \alpha } = \left\| \mathbf { x } _ { 0 } \right\| ^ { 2 } + M ^ { 2 } K ^ { - 2 \alpha } \sum _ { j = 1 } ^ { K } j ^ { 2 \alpha - 2 } .\tag{20}
$$

The additive term is $\mathcal { O } ( K ^ { - 2 \alpha } )$ for $0 < \alpha < 1 / 2 , { \mathcal { O } } ( ( \log K ) / K )$ for $\alpha = 1 / 2 ,$ , and $\mathcal { O } ( K ^ { - 1 } )$ for $1 / 2 < \alpha \leq 1 . \mathrm { A t } \alpha = 1$ , the sum equals K, giving a vanishing bound as $\mathcal { O } ( M ^ { 2 } / K )$ □

## A.2 PROOF OF THEOREM 2 AND COROLLARY 2.1

Proof. For two trajectories with the same coefficients,

$$
\begin{array} { r l } & { \| \mathbf { x } _ { k + 1 } - \widetilde \mathbf { x } _ { k + 1 } \| \leq \left( 1 - \eta _ { k } \right) \| \mathbf { x } _ { k } - \widetilde \mathbf { x } _ { k } \| + \eta _ { k } \| \mathbf { F } _ { \theta } ( \mathbf { x } _ { k } , t _ { k } ) - \mathbf { F } _ { \theta } ( \widetilde \mathbf { x } _ { k } , t _ { k } ) \| } \\ & { \qquad \leq \left( 1 - \eta _ { k } + \eta _ { k } L _ { k } \right) \| \mathbf { x } _ { k } - \widetilde \mathbf { x } _ { k } \| . } \end{array}\tag{21}
$$

Eq. (11) follows by taking the product over k. Setting $\eta _ { k } \equiv 1$ gives the direct-feedback bounds.

For our schedule (Eq. (4)) and $L _ { k } \leq L ,$

$$
\prod _ { k = 0 } ^ { K - 1 } ( 1 - \eta _ { k } + \eta _ { k } L _ { k } ) \leq \prod _ { j = 1 } ^ { K } \left( 1 + { \frac { L - 1 } { j } } \right) .\tag{22}
$$

Note that log $\begin{array} { r } { ( 1 + a ) \le a , \sum _ { i = 1 } ^ { K } j ^ { - 1 } \le 1 + \log K } \end{array}$ and $L > 1$ , thus

$$
\prod _ { j = 1 } ^ { K } \left( 1 + \frac { L - 1 } { j } \right) \leq \exp \left( ( L - 1 ) \sum _ { j = 1 } ^ { K } \frac { 1 } { j } \right) \leq e ^ { L - 1 } K ^ { L - 1 } .
$$

This proves Eq. (13).

## A.3 PROOF OF PROPOSITION 3

Proof. Using the result of the Krasnosel’skii–Mann method in (Cominetti et al., 2014, Proposition 11) with the infimum taken over all fixed points, we have

$$
\| \mathbf { F } _ { \theta } ( \mathbf { x } _ { K } ) - \mathbf { x } _ { K } \| \leq \frac { \mathrm { d i s t } ( \mathbf { x } _ { 0 } , \mathrm { F i x } ( \mathbf { F } _ { \theta } ) ) } { \sqrt { \sum _ { k = 0 } ^ { K - 1 } \eta _ { k } ( 1 - \eta _ { k } ) } } .\tag{23}
$$

Here dist $\begin{array} { r } { ( \mathbf { x } _ { 0 } , \mathrm { F i x } ( \mathbf { F } _ { \theta } ) ) = \operatorname* { i n f } _ { \mathbf { x } ^ { \star } \in \mathrm { F i x } ( \mathbf { F } _ { \theta } ) } \| \mathbf { x } _ { 0 } - \mathbf { x } ^ { \star } \| } \end{array}$ . For schedule in Eq. (4) and $K \geq 2 ,$

$$
\sum _ { k = 0 } ^ { K - 1 } \eta _ { k } ( 1 - \eta _ { k } ) = \sum _ { j = 2 } ^ { K } \frac { j - 1 } { j ^ { 2 } } \ge \frac { 1 } { 2 } \sum _ { j = 2 } ^ { K } \frac { 1 } { j } \ge \frac { 1 } { 2 } \log \frac { K + 1 } { 2 } \ge \frac { 1 } { 4 } \log K .\tag{24}
$$

Substitution proves Eq. (14).

## B LOCAL SPECTRAL ANALYSIS

Here we further state the local stability consequence for interpolated feedback in Eq. (2) following the damping analysis of Movahedi et al. (2026, Appendix A.3).

Proposition B.1. Fix t, and suppose $\mathbf { F } _ { \theta } ( \cdot , t )$ is continuously differentiable near a fixed point $\mathbf { x } ^ { \star }$ If every eigenvalue λ of $\cdot \partial _ { \mathbf { x } } \mathbf { F } _ { \theta } ( \mathbf { x } ^ { \star } , t )$ satisfies Re $\lambda < 1$ , then there exists $\eta \in ( 0 , 1 ]$ such that the iteration

$$
\mathbf { x } _ { k + 1 } = ( 1 - \eta ) \mathbf { x } _ { k } + \eta \mathbf { F } _ { \theta } ( \mathbf { x } _ { k } , t )\tag{25}
$$

is locally asymptotically stable at $\mathbf { x } ^ { \star }$

Proof. By the eigenvalue mapping in Movahedi et al. (2026, Appendix A.3), interpolation maps each eigenvalue λ of $\bar { \partial } _ { \mathbf { x } } \mathbf { F } _ { \theta } ( \mathbf { x } ^ { \star } , t ) \bar { \mathrm { t o } } \bar { \mu } = 1 + \eta ( \lambda - 1 )$ ). Since all real parts of λ are below one, we can always choose

$$
0 < \eta < \operatorname* { m i n } _ { \lambda } \frac { 2 ( 1 - \operatorname { R e } \lambda ) } { | \lambda - 1 | ^ { 2 } } .\tag{26}
$$

Thus all mapped eigenvalues satisfy $| \mu | < 1$ , yielding local asymptotic stability.

Proposition B.1 shows that interpolation can stabilize a fixed point even when the original Jacobian has eigenvalues outside the unit disk. The interpolation coefficient controls how far each eigenvalue moves from 1 toward its original value, allowing local stability whenever all real parts lie below one.

## C MODEL ARCHITECTURE

As illustrated in Figure 1(a), we adopt the LoopFormer (Jeddi et al., 2026) architecture, with token embeddings and learned absolute positional embeddings forming the initial state $\mathbf { x } _ { \mathrm { 0 } }$ . The shared stack contains three Transformer layers, each comprising causal multi-head self-attention (MHSA) and a two-layer feed-forward network (FFN) with GELU activations. Time and step-size conditioning uses AdaLN-style modulation (Peebles & Xie, 2023) with RMSNorm: the conditioning vector controls the scales of normalized features and the gates on the attention and feed-forward residual branches.

Time and step-size embeddings. We encode time and step size with sinusoidal features. For a scalar $\tau ,$ define

$$
\omega _ { j } = 1 0 0 0 0 ^ { - 2 j / d _ { \mathrm { f } } } , \qquad j = 0 , \dots , d _ { \mathrm { f } } / 2 - 1 , \qquad d _ { \mathrm { f } } = 2 5 6 ,\tag{27}
$$

$$
\begin{array} { r } { \mathbf { s } ( \tau ) = \mathrm { c o n c a t } \left( \left( \cos ( \tau \omega _ { j } ) \right) _ { j = 0 } ^ { d _ { \mathrm { f } } / 2 - 1 } , \left( \sin ( \tau \omega _ { j } ) \right) _ { j = 0 } ^ { d _ { \mathrm { f } } / 2 - 1 } \right) . } \end{array}\tag{28}
$$

Two separate MLPs then produce the conditioning vector:

$$
\mathbf { c } _ { k } = \mathrm { M L P } _ { t } ( \mathbf { s } ( t _ { k } ) ) + \mathrm { M L P } _ { \Delta t } ( \mathbf { s } ( \Delta t ) ) \in \mathbb { R } ^ { d } .\tag{29}
$$

Each MLP has dimensions $2 5 6 \to 2 0 4 8 \to 2 0 4 8$ , with a SiLU activation between linear layers. The conditioning vector is shared across token positions and supplied to each layer of the recurrent stack.

Table 3: Default model configuration.
<table><tr><td>Configuration</td><td>Setting</td></tr><tr><td>Number of shared Transformer layers</td><td>3</td></tr><tr><td>Default training / inference loops  $\left( K _ { \mathrm { t r a i n } } / K _ { \mathrm { t e s t } } \right)$ </td><td>8/8</td></tr><tr><td>Hidden dimension d</td><td>2,048</td></tr><tr><td>Attention heads / head dimension</td><td>32 / 64</td></tr><tr><td>Feed-forward intermediate dimension</td><td>5,120</td></tr><tr><td>Feed-forward activation</td><td>GELU</td></tr><tr><td>Maximum context length</td><td>1,024</td></tr><tr><td>Tokenizer</td><td>GPT-2 BPE</td></tr><tr><td>Positional encoding</td><td>Learned absolute embeddings</td></tr><tr><td>Dropout</td><td>0</td></tr><tr><td>RMSNorm €</td><td>2-23</td></tr><tr><td>Physical parameters</td><td>278M</td></tr></table>

Recurrent computation and feedback. Consistent with Section 3.1, let K denote the loop count for the current computation, taking $K = K _ { \mathrm { t r a i n } }$ during training and $K = K _ { \mathrm { t e s t } }$ during inference. On the uniform grid in Eq. (1), the recurrent computation is

$$
\begin{array} { r l } & { \mathbf { h } _ { k + 1 } = \mathbf { F } _ { \theta } ( \mathbf { x } _ { k } , t _ { k } ) , } \\ & { \mathbf { x } _ { k + 1 } = ( 1 - \eta _ { k } ) \mathbf { x } _ { k } + \eta _ { k } \mathbf { h } _ { k + 1 } , \qquad k = 0 , \ldots , K - 1 , } \end{array}\tag{30}
$$

with feedback coefficients

$$
\eta _ { k } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { S L T } , } \\ { \displaystyle \frac { \Delta t } { 1 - t _ { k } } = \frac { 1 } { K - k } , } & { \mathrm { R L T } . } \end{array} \right.\tag{31}
$$

Changing $K _ { \mathrm { t e s t } }$ updates the time grid, step-size conditioning, and RLT feedback coefficients accordingly.

Each hidden output is decoded using the same readout:

$$
\begin{array} { r } { \mathcal { D } ( { \bf h } ) = { \bf W } \mathrm { \ R M S N o r m } ( { \bf h } ) . } \end{array}\tag{32}
$$

Here, W is the weight matrix of lm head, shared with the input token embedding matrix.

## D TRAINING AND IMPLEMENTATION DETAILS

SLT and RLT use the same initial weights, training examples, and data order. Each model is trained with a fixed $K _ { \mathrm { t r a i n } }$ and uniform all-step next-token prediction supervision. Using the per-loop loss defined in Eq. (3), the training objective is

$$
\ell _ { k } = \mathrm { C E } ( \mathcal { D } ( { \mathbf h } _ { k } ) , { \mathbf y } ) , \qquad \mathcal { L } = \frac { 1 } { K _ { \mathrm { t r a i n } } } \sum _ { k = 1 } ^ { K _ { \mathrm { t r a i n } } } \ell _ { k } .\tag{33}
$$

Cross-entropy is averaged over tokens and sequences within each batch.

We use AdamW with linear learning-rate warmup followed by cosine decay. Weight decay is applied to matrix-valued parameters, excluding biases and normalization parameters.

Training uses four NVIDIA A100 GPUs with distributed data parallelism and gradient accumulation. Each GPU processes microbatches of six sequences and accumulates gradients over 20 microbatches per optimizer update. We use BF16 mixed precision with cross-entropy computed in FP32.

## E DATA AND EVALUATION PROTOCOL

Training data. The Pile (Gao et al., 2020) combines 22 text sources covering web content, books, academic writing, code, and other domains. We use its publicly available deduplicated version, tokenized with the GPT-2 tokenizer and separated by end-of-text tokens. We reserve 2,636 complete documents, approximately 12.5M tokens, for internal validation and exclude training windows whose inputs or targets intersect these documents. SLT and RLT use the same training split containing 6.88B tokens, with identical training examples and data order.

Table 4: Training hyperparameters shared by SLT and RLT.
<table><tr><td>Hyperparameter</td><td>Setting</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Peak learning rate</td><td> $6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Minimum learning rate</td><td> $6 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Warmup steps</td><td>4,000</td></tr><tr><td>Optimizer steps</td><td>14,000</td></tr><tr><td>Adam  $( \beta _ { 1 } , \beta _ { 2 } )$ </td><td>(0.9,0.95)</td></tr><tr><td>Weight decay</td><td>0.2</td></tr><tr><td>Gradient norm clipping</td><td>1.0</td></tr><tr><td>Global Batch Size</td><td>480 sequences</td></tr><tr><td>Sequence length</td><td>1,024 tokens</td></tr><tr><td>Total training tokens</td><td>6.88B</td></tr></table>

Perplexity evaluation. For The Pile (Gao et al., 2020), we use its released test split. For FineWeb-Edu (Penedo et al., 2024), we use 8,192 nonempty documents from the first shard of the sample-350BT configuration. For OpenWebText (Gokaslan & Cohen, 2019), we use a validation set comprising 4,007 documents. Each evaluation corpus is tokenized and divided into 1,024-token windows.

To control evaluation cost across loop counts and experimental conditions, we use fixed 10% evaluation subsets of the available token windows from each corpus, shared across all models and settings. Table 5 summarizes their sizes.

Table 5: Corpora used for perplexity evaluation.
<table><tr><td>Corpus</td><td>Available windows</td><td>Evaluation windows</td><td>Evaluation tokens</td></tr><tr><td>The Pile</td><td>364,548</td><td>36,455</td><td>37.33M</td></tr><tr><td>FineWeb-Edu</td><td>8,437</td><td>844</td><td>0.864M</td></tr><tr><td>OpenWebText</td><td>4,331</td><td>433</td><td>0.443M</td></tr><tr><td>Total</td><td>377,316</td><td>37,732</td><td>38.64M</td></tr></table>

Perplexity is computed as

$$
\mathrm { P P L } = \exp \left( - \frac { 1 } { N _ { \mathrm { e v a l } } } \sum _ { n = 1 } ^ { N _ { \mathrm { e v a l } } } \log p _ { \theta } ( y _ { n } \mid \mathcal { C } _ { n } ) \right) ,\tag{34}
$$

where $N _ { \mathrm { e v a l } }$ is the total number of evaluated target tokens, $y _ { n }$ is a target token, and ${ \mathcal { C } } _ { n }$ is its preceding context within the corresponding evaluation window.

Downstream tasks and sample sizes. By default, our downstream evaluation uses a fixed subset of 1,866 examples across ten benchmarks, with ARC evaluated on its Easy and Challenge subsets. Table 6 reports the original and evaluation sample counts. These subsets reduce evaluation cost while making sample counts more balanced across benchmarks. All models use the same selected examples within each experimental setting.

Prompt formats and scoring. Evaluation follows the task templates and scoring conventions of lm-eval harness (Gao et al., 2023). Multiple-choice tasks select the candidate with the highest conditional log-likelihood; normalized accuracy divides each candidate’s score by its character length. LAMBADA uses the OpenAI version and requires the target word’s tokens to match greedy predictions. SciQ prompts include the supporting passage, while RACE uses the high-school subset

Table 6: Downstream evaluation sample sizes and metrics.
<table><tr><td rowspan=1 colspan=1>Benchmark</td><td rowspan=1 colspan=1>Original size</td><td rowspan=1 colspan=1>Evaluation size</td><td rowspan=1 colspan=1>Metric</td></tr><tr><td rowspan=1 colspan=1>COPA</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>Acc</td></tr><tr><td rowspan=1 colspan=1>HellaSwag</td><td rowspan=1 colspan=1>10,042</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1> $\operatorname { A c c } _ { \operatorname { n o r m } }$ </td></tr><tr><td rowspan=1 colspan=1>LAMBADA</td><td rowspan=1 colspan=1>5,153</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>Acc</td></tr><tr><td rowspan=1 colspan=1>OpenBookQA</td><td rowspan=1 colspan=1>500</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1> $\operatorname { A c c } _ { \operatorname { n o r m } }$ </td></tr><tr><td rowspan=1 colspan=1>PIQA</td><td rowspan=1 colspan=1>1,838</td><td rowspan=1 colspan=1>184</td><td rowspan=1 colspan=1> $\operatorname { A c c } _ { \operatorname { n o r m } }$ </td></tr><tr><td rowspan=1 colspan=1>RACE-high</td><td rowspan=1 colspan=1>1,045</td><td rowspan=1 colspan=1>105</td><td rowspan=1 colspan=1>Acc</td></tr><tr><td rowspan=1 colspan=1>SciQ</td><td rowspan=1 colspan=1>1,000</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1> $\operatorname { A c c } _ { \operatorname { n o r m } }$ </td></tr><tr><td rowspan=1 colspan=1>ARC-Easy</td><td rowspan=1 colspan=1>2,376</td><td rowspan=1 colspan=1>238</td><td rowspan=1 colspan=1> $\operatorname { A c c } _ { \operatorname { n o r m } }$ </td></tr><tr><td rowspan=1 colspan=1>ARC-Challenge</td><td rowspan=1 colspan=1>1,172</td><td rowspan=1 colspan=2>117               $\operatorname { A c c } _ { \operatorname { n o r m } }$ </td></tr><tr><td rowspan=1 colspan=1>Social IQA</td><td rowspan=1 colspan=1>1,954</td><td rowspan=1 colspan=2>195              $\operatorname { A c c }$ </td></tr><tr><td rowspan=1 colspan=1>WinoGrande</td><td rowspan=1 colspan=3>1,267                  127              $\operatorname { A c c }$ </td></tr></table>

and includes preceding question–answer pairs from the same passage before scoring the final question.   
Contexts exceeding the model’s input limit are truncated from the left.

We report the task-specific metric listed in Table 6. The ARC score is the arithmetic mean of the ARC-Easy and ARC-Challenge scores, and the overall downstream score is the unweighted mean across the ten benchmarks.

## F RECURRENT STABILITY EXPERIMENTS

All experiments use the frozen models obtained after 14,000 training steps. Unless otherwise specified, $K _ { \mathrm { t r a i n } } = K _ { \mathrm { t e s t } } = 8 .$ , with the uniform time grid in Eq. (1). We write $K = K _ { \mathrm { t e s t } }$ below and define the RMS of a token representation $\mathbf { v } \in \mathbb { R } ^ { d }$ as

$$
\mathrm { R M S } ( \mathbf { v } ) = \sqrt { \frac { 1 } { d } \sum _ { j = 1 } ^ { d } v _ { j } ^ { 2 } } .\tag{35}
$$

## F.1 HIDDEN-STATE SCALE MEASUREMENT

We measure hidden-state scales on 1,200 fixed windows of 1,024 tokens from the internal validation split of The Pile described in Appendix E, yielding 1,228,800 token representations per loop. For each loop i, we extract the complete shared-block output $\mathbf { h } _ { i }$ before feedback and the final RMS normalization. We compute two token-wise ratios, $\mathrm { R M } \bar { \mathrm { S } } ( \mathbf { h } _ { i } ) / \mathrm { R M S } ( \mathbf { h } _ { 1 } )$ and $\mathrm { R M S } ( \mathbf { h } _ { i } ) / \mathrm { R M S } ( \mathbf { x } _ { 0 } )$ using the corresponding first-loop output $\mathbf { h } _ { 1 }$ or initial embedding input $\mathbf { x } _ { \mathrm { 0 } }$ from the same token and model as the reference. We then aggregate each ratio across tokens, reporting the mean, median, and 10th–90th percentiles at each loop.

## F.2 PERTURBATION PROTOCOLS

We inject Gaussian noise with magnitude proportional to the RMS of the perturbed representation:

$$
\begin{array} { r } { \mathcal { N } _ { \rho } ( { \mathbf v } ) = { \mathbf v } + \rho { { \mathrm { R M S } } } ( { \mathbf v } ) { \mathbf z } , { \mathbf z } \sim \mathcal { N } ( 0 , { \mathbf I } _ { d } ) , } \end{array}\tag{36}
$$

where $\rho$ controls the noise level and scaling is applied separately to each token.

Initial-input perturbations. We perturb the initial embedding input $\mathbf { x } _ { \mathrm { 0 } }$ , which combines token and positional embeddings, before the first loop. We evaluate $\rho \in \{ 0 . 0 3 , 0 . 1 , 0 . 3 , 0 . 5 , 1 , 3 , 5 , 1 0 \}$ on the three perplexity evaluation corpora described in Appendix E. For each input, we compare the clean and perturbed final hidden outputs before the final RMS normalization, using the token-averaged Euclidean displacement

$$
D _ { \mathrm { o u t } } = \mathrm { m e a n } _ { \mathrm { t o k e n s } } \left\| \mathbf { h } _ { K } ^ { \mathrm { n o i s y } } - \mathbf { h } _ { K } ^ { \mathrm { c l e a n } } \right\| _ { 2 } .\tag{37}
$$

Recurrent perturbations. We inject noise at every loop, either into the complete block output h<sub>i</sub> before feedback or into the state $\mathbf { x } _ { i }$ after feedback. Each injection uses fresh Gaussian noise scaled by the pre-injection RMS of the corresponding state on the current perturbed trajectory. Perturbations are also applied at the final loop, and the resulting vector is passed through the final normalization and language-model head for evaluation. We report changes in token-averaged cross-entropy and ten-task mean accuracy relative to each model’s clean baseline, using the evaluation sets in Appendix E.

Both perturbation experiments use three paired noise realizations across SLT and RLT. We first aggregate the evaluation results separately for each realization, then report their mean and sample standard deviation.

## F.3 AUXILIARY MLP AND FINAL-OUTPUT EXTRAPOLATION FOR CONTRASTIVE DECODING

Auxiliary prediction branch. We train a separate auxiliary MLP for each frozen backbone and share its parameters across loops. Its input $\mathbf { u } _ { i }$ is the output of the first physical Transformer layer at loop i, and its target h<sub>i</sub> is the output of the complete three-layer block at the same loop. The branch is a bias-free SwiGLU MLP:

$$
[ \mathbf { a } _ { i } , \mathbf { b } _ { i } ] = \mathbf { W } _ { \mathrm { u p } } \mathbf { u } _ { i } , \qquad \hat { \mathbf { h } } _ { i } = \mathbf { W } _ { \mathrm { d o w n } } \big ( \mathrm { S i L U } ( \mathbf { a } _ { i } ) \odot \mathbf { b } _ { i } \big ) , \qquad \mathbf { a } _ { i } , \mathbf { b } _ { i } \in \mathbb { R } ^ { d } ,\tag{38}
$$

with $d = 2 { , } 0 4 8$ . The backbone remains frozen while the MLP is trained on its native trajectories using relative mean squared error:

$$
\mathcal { L } _ { \mathrm { M L P } } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \mathbb { E } _ { \mathrm { v a l i d \ t o k e n s } } \left[ \frac { \| \hat { \mathbf { h } } _ { i } - \mathrm { s g } ( \mathbf { h } _ { i } ) \| _ { 2 } ^ { 2 } } { \| \mathrm { s g } ( \mathbf { h } _ { i } ) \| _ { 2 } ^ { 2 } + 1 0 ^ { - 8 } } \right] ,\tag{39}
$$

where sg denotes stop-gradient.

The MLP is trained on the Pile training split for 2,000 steps, corresponding to approximately 0.98B tokens, with a global batch size of 480 and a sequence length of 1,024. We use AdamW with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = \mathrm { { \bar { 0 } } } . 9 5$ , weight decay 0.01, and gradient clipping at 1.0. The learning rate increases to $3 \times 1 0 ^ { - 4 }$ over 100 warmup steps and then follows cosine decay to $3 \times 1 0 ^ { - 5 }$ . We use the final MLP checkpoint and freeze both the backbone and the MLP during evaluation.

Final-output extrapolation for contrastive decoding. After computing the unguided trajectory, we extrapolate the final output from either a native intermediate output or an MLP prediction:

$$
\mathbf { h } _ { \mathrm { e x t } } = \mathbf { h } _ { K } + \lambda ( \mathbf { h } _ { K } - \mathbf { r } _ { i } ) , \qquad \mathbf { r } _ { i } \in \{ \mathbf { h } _ { i } , { \hat { \mathbf { h } } } _ { i } \} .\tag{40}
$$

For RMS-aligned references, we replace $\mathbf { r } _ { i }$ with

$$
\mathcal { A } ( { \bf r } _ { i } ; { \bf h } _ { K } ) = { \bf r } _ { i } \frac { \mathrm { R M S } ( { \bf h } _ { K } ) } { \mathrm { m a x } \{ \mathrm { R M S } ( { \bf r } _ { i } ) , 1 0 ^ { - 1 2 } \} } ,\tag{41}
$$

computed separately for each token. We evaluate reference loops $i \in \{ 1 , \ldots , 8 \}$ and strengths $\lambda \in \mathbf { \bar { \{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 \} } }$ , with $\lambda = 0$ as the baseline. The extrapolated output is passed through the original final RMS normalization and language-model head. This intervention occurs only at the final readout and is not fed back into the recurrent computation.

Loop-wise contrastive decoding is evaluated on the ten downstream benchmarks. Prompt templates, task metrics, and the ten-task averaging procedure follow Appendix E.

## F.4 GUIDANCE DURING RECURRENCE

We use the auxiliary MLP from Appendix F.3 to guide selected loops. At an activated loop i, both $\mathbf { h } _ { i }$ and $\hat { \mathbf { h } } _ { i }$ are computed from the current guided trajectory. We RMS-align the MLP prediction to the current block output and form

$$
\mathbf { g } _ { i } = \mathbf { h } _ { i } + \lambda \big [ \mathbf { h } _ { i } - \boldsymbol { \mathcal { A } } ( \hat { \mathbf { h } } _ { i } ; \mathbf { h } _ { i } ) \big ] .\tag{42}
$$

At inactive loops, $\mathbf { g } _ { i } = \mathbf { h } _ { i }$ . Following the feedback rules in Sections 3.1–3.2, the next input is

$$
\begin{array} { r } { \mathbf { x } _ { i } = ( 1 - \eta _ { i - 1 } ) \mathbf { x } _ { i - 1 } + \eta _ { i - 1 } \mathbf { g } _ { i } , \qquad \eta _ { i - 1 } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { S L T } , } \\ { 1 / ( K - i + 1 ) , } & { \mathrm { R L T } . } \end{array} \right. } \end{array}\tag{43}
$$

We evaluate guidance over loops $1 \mathrm { - 3 , 3 \mathrm { - 5 , 5 \mathrm { - 7 } , } }$ and 1–7. The final loop is unguided, and its output is decoded through the readout. Evaluation uses the downstream tasks described in Appendix E.

## F.5 QUANTIZATION CONFIGURATIONS

We apply symmetric round-to-nearest quantization to the attention QKV and output projections and the two feed-forward linear layers per shared Transformer layer, covering twelve linear layers in total. Token and positional embeddings, the language-model head, conditioning modules, normalization layers, and feedback operations retain their original precision. Table 7 summarizes the configurations.

Table 7: Quantization configurations for the shared Transformer blocks. Weight groups are formed along the input dimension within each output channel.
<table><tr><td>Configuration</td><td>Weight quantization</td><td>Activation format</td></tr><tr><td>W8A16</td><td>INT8, per output channel</td><td>BF16</td></tr><tr><td>W4A16</td><td>INT4, groups of 128</td><td>BF16</td></tr><tr><td>W8A8</td><td>INT8, per output channel</td><td>Dynamic per-token INT8</td></tr></table>

Quantization scales are determined by the maximum absolute value within each channel or group, using integer ranges [−127, 127] for INT8 and [−7, 7] for INT4. W8A8 reuses the W8A16 quantized weights and dynamically quantizes the input to each target linear layer separately for each token. We simulate quantization through quantize–dequantize operations, with the dequantized tensors used in BF16 matrix multiplications. The same quantized weights are reused across loops and inference depths. Quantization is applied without calibration or retraining.

## G EVALUATION AND TRAJECTORY ANALYSIS ACROSS INFERENCE DEPTHS

## G.1 INFERENCE-DEPTH EVALUATION

We evaluate the frozen SLT and RLT checkpoints trained with $K _ { \mathrm { t r a i n } } = 8$ at different inference depths. For each budget $K = K _ { \mathrm { t e s t } }$ , we reconstruct the time grid with uniform step sizes according to Eq. (1) and update the step-size conditioning accordingly. Predictions are obtained from $\mathbf { h } _ { K }$ through the final RMS normalization and shared language-model head.

## G.2 TRAJECTORY EXTRACTION AND PCA VISUALIZATION

Trajectory extraction. We record recurrent trajectories on 16 fixed, nonoverlapping OpenWebText validation windows of 1,024 tokens, shared across models and budgets. Each trajectory contains the final token’s full-block outputs $\mathbf { h } _ { 1 } , \ldots , \mathbf { h } _ { K }$ before feedback and final RMS normalization.

Shared projections across budgets. For each model and window, we jointly fit a weighted PCA basis over an inference-budget set $\kappa _ { \mathrm { { f i t } } }$ . Each budget receives equal total weight, with each hidden state weighted by

$$
w _ { K , i } = \frac { 1 } { | \mathcal { K } _ { \mathrm { f i t } } | K } .\tag{44}
$$

We center the pooled hidden states by their weighted mean and retain the first three principal components in P. All budgets within a panel use this basis.

For visualization, we translate coordinates relative to the first-loop output at the matched depth:

$$
{ \bf z } _ { i } ^ { ( K ) } = { \bf P } ^ { \top } \left( { \bf h } _ { i } ^ { ( K ) } - { \bf h } _ { 1 } ^ { ( 8 ) } \right) .\tag{45}
$$

The matched-depth endpoint is $\mathbf { h } _ { 8 } ^ { ( 8 ) }$

Additional windows. Figure 10 shows three additional OpenWebText windows using the same trajectory extraction and projection protocol as the main-text visualization. Across these examples, RLT dynamically adjusts its state-update step sizes, keeping its final hidden outputs close to the matched-depth reference as the inference budget changes. In SLT, smaller budgets stop short of the matched-depth endpoint, whereas larger budgets extend beyond it and substantially expand the trajectory along the dominant principal direction. These observations further support the claim that time-dependent state feedback enables RLT to adapt its recurrent updates to different inference budgets while maintaining output stability.

![](images/87189ede27edea2ba9068c66ad99f6505114a433085f5f4150164d730875f24f.jpg)  
Figure 10: 3D PCA projections of final-token hidden outputs for three additional OpenWebText windows. Each row corresponds to one window; SLT and RLT occupy the left and right columns, respectively. Colors indicate the inference budget $K = K _ { \mathrm { t e s t } } ;$ stars and squares mark $\mathbf { h } _ { 1 }$ and $\mathbf { h } _ { K }$ The outlined green square marks the matched-depth endpoint.