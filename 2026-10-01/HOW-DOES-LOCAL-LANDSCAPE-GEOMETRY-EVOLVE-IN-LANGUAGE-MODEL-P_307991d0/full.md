# HOW DOES LOCAL LANDSCAPE GEOMETRY EVOLVE IN LANGUAGE MODEL PRE-TRAINING?

Zhanpeng Zhou<sup>∗1</sup>, Yuhan Sun<sup>∗1</sup>, Bingrui Li<sup>2</sup>, Jinbo Wang<sup>3</sup>, Huaijin Wu<sup>1</sup>, Lei Wu<sup>3</sup>, Junchi Yan<sup>1</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Tsinghua University <sup>3</sup> Peking University

## ABSTRACT

The scale and expense of pre-training language models make efficient hyperparameter tuning essential, yet a principled guidance is still missing. In this work, we analyze language model pre-training dynamics from a local landscape geometry perspective. Our study reveals two distinct phases. In Phase I, sharpness of the local landscape is initially high, leading to instability and loss plateaus under large learning rates (LRs). The landscape shifts from sharp to flatter regions early in training. This dynamic explains the necessity of LR warmup and further suggests that larger peak LRs require proportionally longer warmup periods. In Phase II, the local landscape is governed by the gradient noise scale. Our theory identifies a depth flatness trade-off: high noise from smaller batches widens the loss basin, whereas reduced noise from larger batches deepens it. This theory motivates a dynamic batch-size (BS) scheduler that begins with a small BS and increases it late in training. Together, we provide a unified view of loss landscape evolution, which translates into actionable tuning strategies for large-scale pre-training.

## 1 INTRODUCTION

Training language models efficiently requires carefully tuned hyperparameters, yet a principled guidance for tuning remains unclear. While practitioners often rely on grid search, these approaches are costly and unreliable at scale. Recent research (Foret et al., 2021; Cohen et al., 2021; Gilmer et al., 2022) has highlighted that the geometry of the local loss landscape offers fundamental insights into optimization, revealing how factors such as sharpness<sup>1</sup> (Keskar et al., 2017; Zhang et al., 2017; Jiang et al., 2020) shape training dynamics. Leveraging insights from the local loss landscape presents a promising path toward principled hyperparameter tuning for language model pre-training.

Several pioneering works have already attempted to study language models from the local loss landscape perspective. Zhang et al. (2024a); Wang et al. (2025) identified blockwise sharpness patterns in language models through Hessian-based analyses. Wen et al. (2024) introduced the “river-valley” landscape to explain the effectiveness of Warmup-Stable-Decay (WSD) schedules (Hu et al., 2024). Peng et al. (2024); Chen et al. (2025) further visualized the loss landscapes of finetuned language models, offering geometric insights into the safety alignment.

To this end, we pose the central questions of this paper:

1. How does the local landscape geometry evolve in language model pre-training?

2. What implications does this evolution have for principled hyperparameter tuning?

Our contributions. In this work, we present the systematic study of the evolution of local landscape geometry during language model pre-training. As illustrated in Figure 1, our analysis reveals two distinct phases, each with implications for hyperparameter tuning.

• Phase I: From Sharp to Flat Landscapes. In Phase I, we observe that the model shifts from sharper regions of the loss landscape toward flatter ones, contrary to the progressive sharpening phenomenon in prior works (Cohen et al., 2021; Song & Yun, 2023; Cohen et al., 2025). Lyapunov stability analysis in Section 4 shows that the maximum stable learning rate (LR) is inversely proportional to sharpness. Since sharpness is extremely high early in pre-training, using large peak LRs without sufficient warmup leads to instabilities, such as loss spikes and plateaus (see Figure 2).

![](images/c832ff3c17ffed665f034ac73419889dfbe57cea36264c0a4d845d994860fb01.jpg)

![](images/8705c09b97174775fe474ae2a07a70e470633f19830e0e98f93f93a6a58635b2.jpg)  
Figure 1: The evolution of local loss landscape throughout pre-training. We train LLaMA-2 models with 170M parameters using different BSs (0.49M and 7.8M), and visualize the one-dimensional loss landscape at iterate $\theta _ { t }$ along a random direction δ, i.e., plot $L ( \theta _ { t } + \alpha \delta )$ vs. the perturbation coefficient α. The landscapes are shown across different training iterations t. Phase I. The landscapes gradually widen for both training runs. Phase II. Training with smaller BS produces wider landscapes than training with larger BS.

Implications. The sharp-to-flat transition explains the necessity of LR warmup: LR should remain small until sharpness has sufficiently decayed, preventing training instabilities. This further provides a tuning recipe: within a reasonable range, larger peak LRs require proportionally longer warmup, to safely navigate the sharpest stage of training.

• Phase II: Basin Selection Governed by Noise Scale. In Phase II, the local loss landscape is largely governed by the noise scale during training, with batch size (BS) serving as its primary controller. Our analysis shows that smaller BS widens the loss basin, while larger BS deepens it. Theoretically, we analyze a continuous setup of preconditioned SGD, which uncovers a depth flatness trade-off: reduced gradient noise tends to minimize the loss, leading to deeper minima; whereas increased noise tends to regularize the sharpness of landscape, moving toward wider ones.

Implications. The trade-off, together with the ramping-time experiment in Figure 6, motivates a BS scheduling strategy: begins with a small BS and ramps it until the late phase of training. Our scheduling ensures steady loss reduction with few token consumption. Moreover, since the noise scale is proportional to $\eta / B$ in our theory, we predict that BS ramping and LR decay reduce the noise scale in similar ways and thus yield comparable performance (see Figure 8).

In summary, our work provides a two-phase picture of pre-training: an early sharp-to-flat transition that necessitates LR warmup, and a late noise-driven regime that motivates BS scheduling. This unified view advances our understanding of pre-training dynamics.

## 2 RELATED WORKS

Local landscape geometry evolution. Understanding how local landscape geometry, particularly sharpness, evolves during training has drawn attention before the success of large language models. Wu et al. (2018); Cohen et al. (2022); Song & Yun (2023); Cohen et al. (2025) showed that initially gradient descent (GD) tends to move from flatter to sharper regions of the landscape. In addition, Jastrz˛ebski et al. (2019); Jastrzebski et al. (2020) argued that in SGD, sharpness also changes monotonically but either increase or decrease depending on the setting. In the later phase, however, sharpness is largely governed by the properties of the optimizer (Zhou et al., 2025). One notable example is that the stochastic noise of SGD and its variants implicitly biases training toward flat minima (Wu et al., 2018; Zhu et al., 2019; Xie et al., 2021; Wu et al., 2022). Yet, these findings are largely restricted to small networks; In comparison, our work presents a systematic study of how local landscape geometry evolves in large-scale language model pre-training.

Large-scale pre-training: learning rate warmup. Learning rate warmup, first introduced in largebatch ResNet (He et al., 2016; Goyal et al., 2017) and Transformer training (Vaswani et al., 2017), is now standard in large-scale pre-training (Shoeybi et al., 2019; Zhang et al., 2022; Hu et al., 2024). Its mechanism, however, remains poorly understood. Gotmare et al. (2019) showed that warmup prevents excessively large early parameter updates; Bergsma et al. (2025) attributed the early updates to bias reduction. Kosson et al. (2024) showed in language model pre-training that warmup mitigates momentum bias correction and correlated gradients that drive unstable representation shifts. Yet no unified explanation exists. In comparison, our work views warmup from a geometric perspective, suggesting that larger peak LRs require proportionally longer warmup.

Large-scale pre-training: batch size schedules. Batch size is another critical hyperparameter in large-scale pre-training, controlling the trade-off between time efficiency and data efficiency. Prior works (McCandlish et al., 2018; Kaplan et al., 2020; Gray et al., 2023; 2024; Zhang et al., 2025) have focused on the critical batch size (CBS), the point where further increasing BS yields diminishing returns. However, CBS is usually considered as constant throughout training, and less attention has been given to BS scheduling. Early works on adaptive sampling proposed gradually increasing BS to balance efficiency and noise reduction (De et al., 2017; Lau et al., 2024b;a; 2025; Ostroukhov et al., 2024). Advanced language models (Brown et al., 2020; Touvron et al., 2023; Liu et al., 2024; Li et al., 2025) employed stage-wise BS schedules, but without systematic analysis. In contrast, our work connects BS scheduling to the evolving local loss landscape, providing a principled way for when to increase the BS.

## 3 PRELIMINARIES

Basic notations. We use bold lowercase letters $( \mathbf { e . g . } , \pmb { x } = ( x _ { i } ) )$ to denote vectors and bold uppercase letters $( \mathbf { e . g . , A } = ( a _ { i j } ) )$ ) to denote matrices. For a matrix A, let $\| \mathbf { A } \| _ { 2 } , \| \mathbf { A } \| _ { F }$ , and $\operatorname { T r } ( \mathbf { A } )$ denote its spectral norm, Frobenius norm and trace, respectively. The Hadamard product is denoted by ⊙.

Theoretical setup. Our theory focuses on the preconditioned stochastic gradient descent (PSGD). We consider a model with parameters $\pmb { \theta } \in \mathbb { R } ^ { p }$ and a training set of n examples. Let $L _ { i } ( \pmb \theta )$ be the fitting error evaluated at the i-th example and $\begin{array} { r } { L ( \pmb { \theta } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { \bar { \ } } L _ { i } ( \pmb { \theta } ) } \end{array}$ be the empirical risk. We analyze the preconditioned SGD with a fixed positive-definite<sup>2</sup> preconditioner M ≻ 0. At iteration $k ,$ the update rule gives:

$$
\pmb { \theta } _ { k + 1 } = \pmb { \theta } _ { k } - \eta \mathbf { M } ( \nabla L ( \pmb { \theta } _ { k } ) + \pmb { \xi } _ { k } ) ,\tag{1}
$$

where $\eta > 0$ is the LR and $\{ \pmb { \xi } _ { k } \}$ are i.i.d. random noise vectors with

$$
\mathbb { E } [ \pmb { \xi } _ { k } ] = \mathbf { 0 } , \quad \mathbb { E } [ \pmb { \xi } _ { k } \pmb { \xi } _ { k } ^ { \top } ] = \pmb { \Sigma } ( \pmb { \theta } _ { k } ) / B .\tag{2}
$$

Note that $\begin{array} { r } { \Sigma ( \pmb { \theta } _ { k } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \nabla L _ { i } ( \pmb { \theta } _ { k } ) \nabla L _ { i } ( \pmb { \theta } _ { k } ) ^ { \top } - \nabla L ( \pmb { \theta } _ { k } ) \nabla L ( \pmb { \theta } _ { k } ) ^ { \top } } \end{array}$ is the gradient covariance at $\theta _ { k } .$ , and B denotes the BS. During the late phase of training, the model remains close to some global minimum $\pmb { \theta } ^ { \star }$ and the loss can be approximated quadratically:

$$
L ( \pmb \theta ) = L ( \pmb \theta ^ { \star } ) + \frac { 1 } { 2 } ( \pmb \theta - \pmb \theta ^ { \star } ) ^ { \top } \mathbf { H } ( \pmb \theta ^ { \star } ) ( \pmb \theta - \pmb \theta ^ { \star } ) , \quad \mathbf { H } ( \pmb \theta ^ { \star } ) : = \nabla ^ { 2 } L ( \pmb \theta ^ { \star } ) \succ 0 .\tag{3}
$$

Similar formulations have been widely used in dynamical stability analyses (Wu et al., 2018; Cohen et al., 2021; Zhou et al., 2025) and theoretical advances on BS scaling (McCandlish et al., 2018).

Experimental setup. Our experiments are mainly conducted on LLaMA-2 architecture (Touvron et al., 2023) models with 93M and 170M parameters. Training is performed on the FineWeb-Edu dataset (Penedo et al., 2024), with sufficient training budgets ranging from 50 to 1000 tokens-perparameter $( \mathrm { T P P } ) ^ { 3 }$ and a context length of 1024. We adopt AdamW (Kingma & Ba, 2014) with hyperparameters $\beta _ { 1 } = 0 . 9 5 , \beta _ { 2 } = 0 . \bar { 9 } 5$ , and weight decay 0.1, together with gradient clipping at 1.0 for stability. The evaluation is conducted on a held-out validation split of approximately 50M tokens. More experiments on larger scales, other architectures, and optimizers are deferred to Section C.

Our experiments vary the LRs and BSs. In Section 4, we primarily study the role of LR and warmup length, fixing BS at 7.8M. In Section 5, we focus on the effect of BS, with LR fixed at $2 ^ { - 1 0 }$ . To decouple BS ramping from LR decay, we adopt a warmup-stable schedule: after linear warmup to the peak value, the LR remains constant (similar to WSD (Hu et al., 2024), but without decay phase).

![](images/8b75675b9a6f66f29705798e313b0af3e1e3ebabdcfaec87c66c23539947e370.jpg)

![](images/f2ba2f4cd3ac0eb86ecd9502e330c3b02810b9783aae4ded15871d415371c390.jpg)

![](images/57871bb30096e395dbaf27a83e86cebbb9e3237e00a7e1d00c3dbb10306de12c.jpg)  
Figure 2: Loss spikes and plateaus in Phase I. We train a series of $\mathtt { L I a M A - } 2$ models with 93M and 170M parameters. We adopt a warmup-stable schedule, where the warmup length is shortened to 16 iterations and the peak LR is varied, $\eta \in \{ 2 ^ { - 1 1 } , 2 ^ { - 1 0 } , 2 ^ { - 9 } , 2 ^ { - 8 } , 2 ^ { - 7 } \}$ . (Left). LR schedule: η vs. training iteration t. (Middle, right). Training loss curves for different model sizes: $L ( \pmb \theta _ { t } )$ vs. training iteration t. The vertical dashed line marks the end of the warmup phase.

## 4 PHASE I: FROM SHARP TO FLAT LANDSCAPES

In this section, we provide evidence that, during Phase I, the local landscape of language models evolves from sharp regions toward flatter ones. We first observe that training with large LRs and insufficient warmup often leads to instability and early loss plateaus. Via Lyapunov stability analysis, we attribute these behaviors to the sharp-to-flat dynamics. This finding explains the necessity of LR warmup and suggests that larger peak LRs require proportionally longer warmup.

Motivating observations: instability and loss plateaus early in training. The loss curves are typically smooth initially; the model escapes from random initialization and the loss decreases rapidly. Yet, surprisingly, when the warmup length is extremely shortened, we consistently observe loss spikes and plateaus near the end of the warmup phase.

To demonstrate this, we train models of different sizes with a fixed warmup length of 16 iterations while varying the peak LR. As shown in Figure 2, a loss plateau reliably appears around the end of the warmup phase across all settings. Additionally, larger LRs produce higher spikes, which mark a characteristic feature of early training instability. Given these results, two natural questions arise:

Q1. Why does shortened warmup induce training instability?

Q2. Why do spikes and plateaus occur only at the very beginning of training?

To shed light on these questions, we analyze the dynamics of PSGD via Lyapunov stability analysis.

Lyapunov stability analysis: sharpness matters. Let $\theta _ { k } , \tilde { \theta } _ { k }$ be two nearby trajectories, and define their difference as $\boldsymbol { e } _ { k } : = \tilde { \boldsymbol { \theta } } _ { k } - \boldsymbol { \theta } _ { k }$ . When the noise term $\boldsymbol { \xi }$ is set to zero, the evolution of $e _ { k }$ satisfies:

$$
\begin{array} { r } { \pmb { e } _ { k + 1 } = \pmb { e } _ { k } - \eta \mathbf { M } \big ( \nabla L ( \pmb { \theta } _ { k } + \pmb { e } _ { k } ) - \nabla L ( \pmb { \theta } _ { k } ) \big ) \overset { \mathrm { ( L i n e a r i z a t i o n ) } } { = } ( \mathbf { I } - \eta \mathbf { M } \mathbf { H } ( \pmb { \theta } _ { k } ) ) \pmb { e } _ { k } , } \end{array}\tag{4}
$$

The dynamics in Equation (4) describe the local sensitivity of the iteration: if matrix $( \mathbf { I } - \eta \mathbf { M } \mathbf { H } ( \pmb { \theta } _ { k } ) )$ repeatedly expands $e _ { k }$ , small perturbations grow exponentially and the iterates are linearly unstable. Intuitively, the LR η interacts directly with the curvature of the landscape: if η is too large relative to the sharpest direction, the update rule amplifies perturbations and leads to loss spikes. The following lemma formalizes this stability condition for preconditioned GD.

Lemma 4.1 (Stability condition for preconditioned GD). Define the preconditioned curvature matrix $\mathbf { S } ( \pmb { \theta } _ { k } ) : = \mathbf { M } ^ { 1 / 2 } \mathbf { H } ( \mathbf { \bar { \theta } } _ { k } ) \mathbf { M } ^ { 1 / 2 }$ , and let $\{ \lambda \} _ { i = 1 } ^ { p }$ be the eigenvalues of ${ \bf S } ( \pmb { \theta } _ { k } )$ . The linear system in Equation (4) is asymptotically stable (i.e., lim $\boldsymbol { \mathbf { \ell } } _ { k  \infty } \boldsymbol { e } _ { k } = \boldsymbol { \mathbf { 0 } } )$ ifη satisfies $\begin{array} { r } { 0 < \eta < \frac { 2 } { \lambda _ { \mathrm { m a x } } ( \mathbf { S } ( \theta _ { k } ) } , \forall k \geq 0 } \end{array}$

Lemma 4.1 shows that the Lyapunov stability is governed by the largest eigenvalue of S. If the curvature along the sharpest direction is too large, only a sufficiently small LR can prevent divergence.

We next characterize the one-step loss change as η approaches the stability boundary $2 / \lambda _ { \operatorname* { m a x } } ( \mathbf { S } _ { k } )$

Lemma 4.2 (One-step loss change). Let $\delta _ { k } : = \theta _ { k + 1 } - \theta _ { k }$ . Suppose that along the segment $\{ \pmb \theta _ { k } + \alpha \pmb \delta _ { k } : \alpha \in [ 0 , 1 ] \}$ , we have $\bar { 0 } \leq \lambda _ { \operatorname* { m i n } } ( \mathbf { S } ( \pmb \theta _ { k } + \alpha \pmb \delta _ { k } ) ) \leq \lambda _ { \operatorname* { m a x } } ( \bar { \mathbf { S } } ( \pmb \theta _ { k } + \alpha \pmb \delta _ { k } ) ) \leq \Lambda _ { k }$ . Then,

$$
L ( \pmb \theta _ { k + 1 } ) - L ( \pmb \theta _ { k } ) \leq - \eta ( 1 - \frac { 1 } { 2 } \eta \Lambda _ { k } ) ( \nabla L ( \pmb \theta _ { k } ) ) ^ { \top } \mathbf { M } \nabla L ( \pmb \theta _ { k } ) .
$$

![](images/e6e5517086353fdb3407bf8c3721a68049b93697821ab2ae1662423f8a48e24f.jpg)  
Figure 3: Early pre-training shifts iterates from sharp to flat regions. We visualize the local landscape geometry evolution of training runs in Figure 2. For each model size, we select the training run with LR $2 ^ { - 1 0 }$ . (Top). Evolution of the top eigenvalues of the Hessian across iterations: $\lambda _ { i } ( \mathbf { H } ( \pmb \theta _ { t } ) )$ vs. iteration t. (Bottom). One-dimensional loss landscape along a random perturbation direction: the perturbed loss $L ( \theta _ { t } + \alpha \delta )$ vs. perturbation coefficient α, shown across early training iterations t.

In particular, $i f \eta \uparrow 2 / \Lambda _ { k }$ , the guaranteed decrease per step $| ( L ( \pmb \theta _ { k } ) - L ( \pmb \theta _ { k + 1 } ) ) / \eta  0 |$

Lemma 4.2 states that when η is close to $2 / \Lambda _ { k }$ , each update yields only a marginal decrease in loss. Together with Lemma 4.1, it is clear that training near the stability boundary naturally leads to characteristic loss spikes and plateaus.

Importantly, the stability boundary is determined by the sharpness of the loss landscape. To further address Q1-2, we analyze how sharpness evolves during the early phase.

The early dynamics: from sharp to flat landscapes. We study how the local landscape geometry, particularly the sharpness, evolves for training runs in Figure 2. Specifically, we track the evolution of the top eigenvalues of the Hessian<sup>4</sup> $\mathbf { H } ( \pmb \theta _ { t } )$ . For some early checkpoints $\theta _ { t }$ , we also visualize the onedimensional loss landscape along a random direction by plotting the function $\mathcal { L } ( \alpha ) : = L ( \pmb \theta _ { t } + \alpha \pmb \delta )$ with $\mathbf { \delta } \delta \sim \mathcal { N } ( 0 , \mathbf { I } )$ . Li et al. (2018) showed that these random-direction visualizations reliably capture intrinsic properties of the loss landscape properties, such as sharpness. To ensure fair comparison across iterations, we fix the same random vector δ for all $\theta _ { t }$

In Figure 3 (top), the largest eigenvalues of the Hessian $\mathbf { H } ( \pmb \theta _ { t } )$ start at high values<sup>5</sup> and then decrease sharply. In Figure 3 (bottom), the loss landscape along a random direction progressively widens as training proceeds, confirming that the model shifts from sharp to flat regions.

A Tuning Recipe: Larger Peak LR, Longer Warmup. We have seen that training stability depends on sharpness: when the landscape is shar, only a sufficiently small LR can keep training stable; and pre-training initially traverses from sharp landscapes to flatter ones. Now let us return to Q1 and Q2:

A1. If the warmup phase is shortened, the LR rises too quickly while the model is still in sharp regions, leading to loss spikes and plateaus.

A2. As training progresses, the landscape becomesflatter and the same LR no longer threatens stability, which explains why instability is confined to the very beginning.

Therefore, in practice, we need a sufficiently long warmup phase to keep the LR small until sharpness has decayed, thereby preventing loss spikes and plateaus. This further suggests a practical tuning recipe: the larger the peak LR, the longer the warmup should $b e ,$ , ensuring iterates safely transition into flatter landscapes.

To validate this, we train models with varied peak LRs η and warmup lengths $T _ { w }$ (in iterations). In Figure 4, within a LR range of $2 ^ { - 8 } \tan 2 ^ { - 1 1 }$ , larger peak LRs require proportionally longer warmup to achieve the optimal validation loss $L ( \theta _ { \mathrm { b e s t } } )$ ). However, this proportionality does not hold universally. When $\eta = 2 ^ { - 7 }$ , the optimal warmup length remains $2 ^ { 1 0 }$ iterations, the same as for $\eta = 2 ^ { - 8 }$ . Thus, the proportional relationship applies only within a certain range.

## 5 PHASE II: LOCAL LANDSCAPE GOVERNED BY NOISE SCALE

In this section, we turn to the local landscape geometry in late phase. We observe that BS plays a central role: training with a large BS tends to find a deeper basin of the landscape, whereas a small BS favors a wider basin. Theoretically, we prove that this tradeoff between widen or deepen is governed by the noise scale. Building on this, we propose a BS scheduler for the data-limited regime: use small BS early and ramp the BS late, which consumes fewer tokens to achieve the same loss.

The effect of BS: local landscapes late in training. We conduct experiments to systematically investigate the role of BS in shaping the local loss landscape during the late phase of training. Specifically, we train models with different BSs for $\bar { T ^ { \mathrm { ~ } } } = 2 0 , 4 8 \bar { 0 }$ iterations. Figure 5 (top left) shows the validation loss curves for each run. Larger BS consistently leads to lower terminal loss and faster convergence in term of iterations<sup>6</sup>. We then visualize the loss landscape around the final iterate $\pmb { \theta } _ { T }$ . In Figure 5 (top right), it is clear that small BS produces flatter basins, whereas large BS yields deeper ones. Furthermore, Figure 5 (bottom) compares the landscape evolution of runs with $B = \bar { 0 . 4 9 } \mathrm { { M } }$ and $B = 7 . 8 \mathbf { M }$ indicating that in the late phase, larger BS tends to deepen the basin, while smaller BS shifts toward wider basins.

![](images/c00e0e4b5c0acb29d11ddc9d78d2e930a2865891ed6378ec8cda573b82c6ffa1.jpg)  
Figure 4: Larger peak LR, longer warmup. We plot the best validation loss $L ( \theta _ { \mathrm { b s t } } )$ vs. warmup $T _ { w }$ for different η. Optimal $T _ { w }$ is highlighted with a star.

Despite these results, two key questions remain:

Q3. Why is there a trade-off between widening and deepening the basin?

Q4. Which factor underlying the hyperparameter BS governs this trade-off?

To delve into Q3-4, we revisit the stochastic differential equation (SDE) in Jastrz˛ebski et al. (2017).

Widen or deepen: noise scale governs basin selection. Following Jastrz˛ebski et al. (2017), we take the continuous-time limit of Equation (1). Suppose that the noise covariance satisfies $\begin{array} { r } { \frac { \eta } { R } \mathbf { M } \pmb { \Sigma } ( \pmb { \theta } ^ { \star } ) \mathbf { M } ^ { \top } = 2 \tau \mathbf { M } + \mathcal { O } ( \eta ) } \end{array}$ for some temperature $\tau > 0 . \operatorname { A s } \eta \to 0$ , the scaled discrete process $\boldsymbol { \widetilde { \theta } } _ { \lfloor t / \eta \rfloor }$ converges weakly to the Itô SDE:

$$
d \pmb { \theta } _ { t } = - \mathbf { M } \nabla L ( \pmb { \theta } _ { t } ) d t + \sqrt { 2 \tau } \mathbf { M } ^ { 1 / 2 } d W _ { t }\tag{5}
$$

where $W _ { t }$ is standard Brownian motion and the noise scale τ is proportional to $\eta / B$

Building on Equation (5) and the local quadratic model in Equation (3), we establish the trade-off between deepening and widening the loss basin.

Theorem 5.1 (Depth-flatness trade-off). Let the empirical risk $L ( \theta )$ admit multiple local minima $\{ \pmb { \theta } _ { i } ^ { \star } \} _ { i = 1 } ^ { m }$ with Hessians $\mathbf { H } ( \pmb { \theta } _ { i } ^ { \star } ) \succ 0$ . Under the SDE in Equation (5) with temperature $\tau ,$ , the stationary probability that training resides in basin i is given by:

$$
P _ { \tau } ( b a s i n i ) = \frac { \exp ( - F _ { i } ( \tau ) / \tau ) } { \sum _ { j } \exp ( - F _ { j } ( \tau ) / \tau ) } , \quad F _ { i } ( \tau ) : = \ : L ( \pmb { \theta } _ { i } ^ { \star } ) \ : + \ : \frac { \tau } { 2 } \log \operatorname* { d e t } \mathbf { H } ( \pmb { \theta } _ { i } ^ { \star } ) .
$$

![](images/5cfaf947a2ffd40231295c1bedce0087bf3cfd28fa93fa60fe2137b0e964174e.jpg)

![](images/41101936f472a4c772b1eeeaeda6b76f51499269860f8f24cf14b8ed4cc84305.jpg)

![](images/ec8e116d00f47779dd8ca0e82545ef2e5731f675cdca0a6e8d8074dd32c3965b.jpg)

![](images/2872e7a93cb88405c6199cc1305fd30424a89e8a43f40683dd9c7cec632ac892.jpg)  
Figure 5: Large BS deepens the basin, small BS widens the basin. We train a series of LLaMA-2 models (170M) for $T = 2 0 { , } 4 8 0$ iterations, using BSs $B \in \{ 0 . 4 9 \mathbf { M } , 0 . 9 8 \mathbf { M } , 1 . 9 \mathbf { M } , 3 . 9 \mathbf { M } , 7 . 8 \mathbf { M } \}$ (Top left). Validation loss curves for different BSs: $L ( \pmb \theta _ { t } )$ vs. training iteration t. (Top right). One-dimensional loss landscapes at the final iterates $\pmb { \theta } _ { T }$ along a random perturbation direction: perturbed loss $L ( \pmb { \theta } _ { T } + \alpha \pmb { \delta } )$ vs. perturbation coefficient $\alpha ,$ , visualized across different BSs. (Bottom). One-dimensional loss landscape: the perturbed loss $L ( \theta _ { t } + \alpha \delta )$ vs. perturbation coefficient α, shown across late training iterations t for $\bar { B = 0 . 4 9 \mathrm { M } }$ and $B = 7 . 8 \mathbf { M }$

Theorem 5.1 states that the basin selection is controlled by the free energy function $F ( \tau ) = L ( \pmb \theta ^ { \star } ) +$ $\begin{array} { l } { { \frac { \tau } { 2 } } } \end{array}$ log det $\mathbf { H } ( \pmb \theta ^ { \star } )$ . In early training, the loss term $L ( \theta ^ { \star } )$ dominates, so the model primarily seeks regions of lower loss. In later training, $L ( \theta ^ { \star } )$ is comparable to the flatness penalty log det $\mathbf { H } ( \pmb \theta ^ { \star } )$ , and basin selection becomes increasingly sensitive to the noise scale $\tau \propto \eta / \bar { B }$

Efficient pre-training: a BS scheduler in data-limited regime. Turning back to Q3 and Q4, the trade-off arises because basin selection balances loss minimization against curvature regularization (A3), with the governing factor being the noise scale τ (A4). Since the primary objective of pretraining is to minimize the training loss<sup>8</sup>, this balance naturally favors largest BS available (small τ). In practice, however, data availability is limited, and excessively large BS substantially increase data consumption<sup>9</sup>. Thus, scheduling BS in pre-training is crucial, particularly in the data-limited regime.

• BS scheduler: design principle I. Inspired by our theory, loss reduction dominates early in training, during which large BS yields limited benefit. This suggests the first design principle.

## Design principle I. Start the training process with a small BS before increasing it later.

Related ideas were noted by Li et al. (2025); Merrill et al. (2025), often referred to as BS warmup. However, a key difference in our design lies in when the BS should be increased. Surprisingly, we find that ramping BS later in training yields consistently greater performance.

• BS scheduler: design principle II. To study this, we train models with different BS schedulers while keeping total training iterations fixed. In Figure 6 (left), all runs begin with an initial BS of 0.49M and ramp up to either 4× or 2× that value at different training iterations. Remarkably, all loss curves eventually collapse onto the same trajectory, regardless of when the BS ramping occurs. Note that when measured at the same training iteration, ramping the BS earlier results in higher data consumption. This indicates that early BS ramping offers no efficiency advantage, achieving the same loss but consuming more data.

We next evaluate BS schedules under a fixed token budget. Specifically, we consider a two-stage BS-ramping scheduler characterized by ramp times $T _ { 1 }$ and $T _ { 2 }$ . To isolate the effect of each stage, we vary either $T _ { 1 }$ or $T _ { 2 }$ while keeping the other fixed. See Figure 6 (middle) for an illustration. In

![](images/66a693d67b5730df6bd560aa934a0b6b000ae69c470710fa9cd1f295acd8c955.jpg)

![](images/5f7bd376b965a635231290c0184ded688214b4e5e6df8f7b76ba545da2b76999.jpg)  
Tokens (B)

![](images/480bc9e27daf625220fd38b3bf1d3bf0aca71448cb92595ec089a67dad47c966.jpg)  
Figure 6: (Left) Collapse of loss curves under different BS schedules. Validation loss curves for training with different BS scheduling. In all runs, BS starts at 0.49M. For blue curves, BS is ramped up to 4× its initial value; for red curves, BS is ramped to $2 \times$ . The ramping times, $T _ { 2 \times }$ or $T _ { 4 \times }$ , are varied across different positions. (Middle, Right) Ramping BS is more efficient late in training. We evaluate a two-stage BS-ramping schedule with ramp times $T _ { 1 }$ and $T _ { 2 }$ . For the red curves, we fix $T _ { 2 } = 1 0 \mathbf { B }$ and vary $T _ { 1 } ;$ for the blue curves, we fix $T _ { 1 } = 1 0 \mathbf { B }$ and vary $T _ { 2 }$ . (Middle). Illustration of BS schedulers. (Right). Final validation loss vs. the relative ramping time, i.e., $( T _ { 1 } ) / T , ( T _ { 2 } - T _ { 1 } ) / T \in [ 0 , 0 . 5 ]$ , where $T$ denotes the total training tokens.

![](images/3e7b14d4138310f2a435c3ef8519cb84d16941b57ca6ab0f8629fe965805055d.jpg)

![](images/6b4e413e1fa0ead6c7d58587e3b4cf4003e59259234616795830534b7ad4d239.jpg)  
Tokens (B)

![](images/4854901dd65551836a20991e0b45f542084b1a90612d8ef691535514fcd2228b.jpg)  
Figure 7: BS scheduling improves data efficiency. We train $\mathrm { L L a M A } { - 2 }$ models with 93M and 170M parameters, using a BS schedule that starts at 0.49M and increases by 4× at each ramp. Models are trained with 1, 2, or 3 ramping steps, while models without ramping serve as the baseline. Vertical gray dashed lines indicate ramping positions. (Left, Middle). The validation curves for each run. (Right). Comparison between training with BS ramping to 7.8M and training with a fixed 7.8M BS.

Figure 6 (right), a clear trend emerges: BS ramping is most effective when applied late in training (i.e., with $T _ { 1 }$ and $T _ { 2 }$ large), whereas ramping too early consistently harms final performance.

Together, since BS ramping ultimately leads all runs onto the same trajectory, delaying it allows maximal progress (lower loss) along that trajectory under a data-limited budget. This behavior also aligns with our theory. In the late phase of training, the flatness penalty becomes comparable to the loss term, and BS ramping sharply reduces the noise scale, driving rapid convergence toward deeper minima. This consistency between theory and practice leads to our second principle.

## Design principle II. Ramp the batch size late in training—when loss reduction becomes marginal.

To further validate our design principle, we train models using a BS schedule that starts at 0.49M and ramps by 4× whenever loss minimization slows. In Figure 7 (left, middle), models with 1, 2 or 3 BS ramping steps achieves significant lower validation loss. While additional ramping steps provide diminishing returns, each step still offers a measurable improvement. Moreover, Figure $\bar { 7 }$ (right) highlights the data-efficiency of the BS scheduling: ramping the BS up to 7.8M achieves nearly the same final validation loss as training with a fixed 7.8M BS, but requires only about $\textstyle { \frac { 1 } { 4 } }$ of the tokens (i.e., a ∼ 4× speedup). These results confirm that our BS scheduling design preserves the benefits of large BS while substantially reducing data consumption.

Comparison with McCandlish et al. (2018); Merrill et al. (2025). McCandlish et al. (2018) linked BS scaling to the gradient noise and introduced the notion of CBS. Merrill et al. (2025) explored the BS scheduling (BS warmup), doubling BS once the CBS exceeds the current BS. We differs from CBS-related works: without estimating CBS on the fly, we propose to ramp the BS late in training.

![](images/a2cde583ba63337b2178d6c3c11d6f02cef1baa87675669850099f82102c6bdb.jpg)

The Interplay of BS Ramping & LR Decay  
![](images/286090e825eb4e5cd0869fbc8d2c6aa99551207131ef2bd6496cda85e07f8f38.jpg)  
Figure 8: (Left) BS ramping performs similarly to LR decay. Validation loss curves for training with either BS ramping or LR decay. For BS ramping, BS increases to 16× its initial value; for LR decay, the LR drops to 1/16 of its initial value. Each method applies a single step at varying positions. (Right) Interplay between BS ramping and LR decay. We evaluate four different scheduling strategies. At the 10B tokens, (a) LR drops to $1 / 4$ of its initial value; (b) LR drops to 1/16; (c) BS ramps to 16×. (d) LR drops to 1/4 and BS ramps to 4×. In all runs (both left and right), BS starts at 0.49M and LR begins at $2 ^ { - 1 0 }$ (after linear warmup).

## 6 MORE DISCUSSIONS: LR DECAY AND BS RAMPING

We have excluded LR decay in our experiments to isolate the effect of BS ramping. Yet, recall that the noise scale τ is proportional to $\eta / { \dot { B } } .$ . Our theory suggests that decaying the LR and ramping the BS both reduce the noise scale, and thus may have similar effects on basin selection.

Comparing BS ramping with LR decay. We first train models using either BS ramping or LR decay. Both methods apply a one-time step change: BS ramping multiplies the BS by 16 at $T _ { \mathrm { r a m p } } ,$ , while LR decay divides the LR by 16 at $T _ { \mathrm { d e c a y } }$ . We align $T _ { \mathrm { r a m p } }$ and $T _ { \mathrm { d e c a y } }$ so that the changes occur at the same positions. In Figure 8 (left), BS ramping and LR decay produce remarkably similar validation loss curves across all change positions, consistent with the idea that both reduce the noise scale in comparable ways.

Interacting BS ramping with LR decay. Furthermore, we study the combined effect of using both BS ramping and LR decay. Specifically, we decay the LR by 4× and simultaneously ramp the BS by 4× at 10B tokens. We compare this hybrid schedule with three baselines: at the same point, we (a) drops the LR by 4×, (b) drops the LR by 16× and (c) ramps the BS by 16×. We denote the hybrid schedule by (d). In Figure 8 (right), three of the schedules (b c d) produce nearly identical loss curves. Crucially, these three configurations yield the same noise scale $\dot { \eta / B }$ . In contrast, schedule (a) results in a noticeably different trajectory.

In summary, our results confirm our theoretical prediction: training dynamics in the late phase are governed by the noise scale $\eta / B .$ . LR decay reduce the noise scale in the same manner as BS ramping, and any scheduler that preserves $\eta / B$ will exhibit nearly identical behavior.

## 7 CONCLUSION AND LIMITATIONS

In conclusion, we present a study of how local landscape geometry evolves during language model pre-training. Our analysis reveals two phases: an early sharp-to-flat transition and a late noisegoverned regime. Phase I explains the necessity of LR warmup, suggesting that larger peak LRs require proportionally longer warmup lengths. Phase II motivates a BS scheduling that starts with small BS and increases the BS late in training. Limitations. The current theory primarily relies on strong assumptions, such as infinite-small LR in SDE. A natural future direction is to generalize the theory to more realistic settings. Additionally, the current theory cannot fully explain the collapse of loss curves under different BS schedules. Understanding the learning dynamics under different BS schedules remains an open question.

## REFERENCES

Shane Bergsma, Nolan Simran Dey, Gurpreet Gosal, Gavia Gray, Daria Soboleva, and Joel Hestness. Straight to zero: Why linearly decaying the learning rate to zero works best for LLMs. In The Thirteenth International Conference on Learning Representations, 2025. 3

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 3

Huanran Chen, Yinpeng Dong, Zeming Wei, Yao Huang, Yichi Zhang, Hang Su, and Jun Zhu. Understanding pre-training and fine-tuning from loss landscape perspectives. arXiv preprint arXiv:2505.17646, 2025. 1

Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, et al. Symbolic discovery of optimization algorithms. Advances in Neural Information Processing Systems, 36, 2024. 20

Jeremy Cohen, Alex Damian, Ameet Talwalkar, J Zico Kolter, and Jason D. Lee. Understanding optimization in deep learning with central flows. In The Thirteenth International Conference on Learning Representations, 2025. 1, 2

Jeremy M Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. International Conference on Learning Representations, 2021. 1, 3, 5

Jeremy M Cohen, Behrooz Ghorbani, Shankar Krishnan, Naman Agarwal, Sourabh Medapati, Michal Badura, Daniel Suo, David Cardoze, Zachary Nado, George E Dahl, et al. Adaptive gradient methods at the edge of stability. arXiv preprint arXiv:2207.14484, 2022. 2

Soham De, Abhay Yadav, David Jacobs, and Tom Goldstein. Automated inference with adaptive batches. In Artificial Intelligence and Statistics, pp. 1504–1513. PMLR, 2017. 3

John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal ofMachine Learning Research, 12(61):2121–2159, 2011. 3

Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization for efficiently improving generalization. In International Conference on Learning Representations, 2021. 1

Justin Gilmer, Behrooz Ghorbani, Ankush Garg, Sneha Kudugunta, Behnam Neyshabur, David Cardoze, George Edward Dahl, Zachary Nado, and Orhan Firat. A loss curvature perspective on training instabilities of deep learning models. In International Conference on Learning Representations, 2022. 1

Akhilesh Gotmare, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. A closer look at deep learning heuristics: Learning rate restarts, warmup and distillation. In International Conference on Learning Representations, 2019. 3

Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017. 2

Gavia Gray, Anshul Samar, and Joel Hestness. Efficient and approximate per-example gradient norms for gradient noise scale. In Workshop on Advancing Neural Network Training: Computational Efficiency, Scalability, and Resource Optimization (WANT@ NeurIPS 2023), 2023. 3

Gavia Gray, Shane Bergsma, Joel Hestness, et al. Normalization layer per-example gradients are sufficient to predict gradient noise scale in transformers. Advances in Neural Information Processing Systems, 37:93510–93539, 2024. 3

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016. 2

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. 3

Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024. 1, 3

Stanisław Jastrz˛ebski, Zachary Kenton, Devansh Arpit, Nicolas Ballas, Asja Fischer, Yoshua Bengio, and Amos Storkey. Three factors influencing minima in sgd. arXiv preprint arXiv:1711.04623, 2017. 6

Stanislaw Jastrzebski, Maciej Szymczak, Stanislav Fort, Devansh Arpit, Jacek Tabor, Kyunghyun Cho\*, and Krzysztof Geras\*. The break-even point on optimization trajectories of deep neural networks. In International Conference on Learning Representations, 2020. 2

Stanisław Jastrz˛ebski, Zachary Kenton, Nicolas Ballas, Asja Fischer, Yoshua Bengio, and Amost Storkey. On the relation between the sharpest directions of DNN loss and the SGD step length. In International Conference on Learning Representations, 2019. 2

Yiding Jiang, Behnam Neyshabur, Hossein Mobahi, Dilip Krishnan, and Samy Bengio. Fantastic generalization measures and where to find them. In International Conference on Learning Representations, 2020. 1

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. 3

Andrej Karpathy. NanoGPT. https://github.com/karpathy/nanoGPT, 2022. 20

Jordan Keller et al. Muon optimizer. https://github.com/KellerJordan/Muon?tab= readme-ov-file, 2024. 20

Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. In International Conference on Learning Representations, 2017. 1

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 3, 20

Atli Kosson, Bettina Messmer, and Martin Jaggi. Analyzing & reducing the need for learning rate warmup in GPT training. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 3

Tim Tsz-Kit Lau, Weijian Li, Chenwei Xu, Han Liu, and Mladen Kolar. Communicationefficient adaptive batch size strategies for distributed local gradient methods. arXiv preprint arXiv:2406.13936, 2024a. 3

Tim Tsz-Kit Lau, Han Liu, and Mladen Kolar. Adadagrad: Adaptive batch size schemes for adaptive gradient methods. arXiv preprint arXiv:2402.11215, 2024b. 3

Tim Tsz-Kit Lau, Weijian Li, Chenwei Xu, Han Liu, and Mladen Kolar. Adaptive batch size schedules for distributed training of language models with data and model parallelism. In Proceedings of Conference on Parsimony and Learning, 2025. 3

Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, Chang Liu, Cheng Zhu, Chunhao Zhang, Congchao Guo, Da Chen, Dong Li, et al. Minimax-01: Scaling foundation models with lightning attention. arXiv preprint arXiv:2501.08313, 2025. 3, 7

Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. Visualizing the loss landscape of neural nets. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. 5

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 3

Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota Team. An empirical model of large-batch training. arXiv preprint arXiv:1812.06162, 2018. 3, 8

William Merrill, Shane Arora, Dirk Groeneveld, and Hannaneh Hajishirzi. Critical batch size revisited: A simple empirical approach to large-batch language model training. arXiv preprint arXiv:2505.23971, 2025. 7, 8

Petr Ostroukhov, Aigerim Zhumabayeva, Chulu Xiang, Alexander Gasnikov, Martin Takác, and ˇ Dmitry Kamzolov. Adabatchgrad: Combining adaptive batch size and adaptive step size. arXiv preprint arXiv:2402.05264, 2024. 3

Guilherme Penedo, Hynek Kydlícek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colinˇ Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. 3, 20

ShengYun Peng, Pin-Yu Chen, Matthew Daniel Hull, and Duen Horng Chau. Navigating the safety landscape: Measuring risks in finetuning large language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 1

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. 20

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019. 3

Minhak Song and Chulhee Yun. Trajectory alignment: understanding the edge of stability phenomenon via bifurcation theory. arXiv preprint arXiv:2307.04204, 2023. 1, 2

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 20

Tijmen Tieleman and Geoffrey Hinton. Rmsprop: Divide the gradient by a running average of its recent magnitude. coursera: Neural networks for machine learning. COURSERA Neural Networks Mach. Learn, 17:6, 2012. 3

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 3, 20

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, volume 30, 2017. 2

Jinbo Wang, Mingze Wang, Zhanpeng Zhou, Junchi Yan, Weinan E, and Lei Wu. The sharpness disparity principle in transformers for accelerating language model pre-training. In Forty-second International Conference on Machine Learning, 2025. 1

Kaiyue Wen, Zhiyuan Li, Jason Wang, David Hall, Percy Liang, and Tengyu Ma. Understanding warmup-stable-decay learning rates: A river valley loss landscape perspective. arXiv preprint arXiv:2410.05192, 2024. 1

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38–45, Online, October 2020. Association for Computational Linguistics. 20

Lei Wu, Chao Ma, and Weinan E. How SGD selects the global minima in over-parameterized learning: A dynamical stability perspective. Advances in Neural Information Processing Systems, 31:8279–8288, 2018. 2, 3

Lei Wu, Mingze Wang, and Weijie J Su. The alignment property of SGD noise and how it helps select flat minima: A stability analysis. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. 2

Zeke Xie, Issei Sato, and Masashi Sugiyama. A diffusion theory for deep learning dynamics: Stochastic gradient descent exponentially favors flat minima. In International Conference on Learning Representations, 2021. 2

Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In International Conference on Learning Representations, 2017. 1

Hanlin Zhang, Depen Morwani, Nikhil Vyas, Jingfeng Wu, Difan Zou, Udaya Ghai, Dean Foster, and Sham Kakade. How does critical batch size scale in pre-training? International Conference on Learning Representations, 2025. 3

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022. 3

Yushun Zhang, Congliang Chen, Tian Ding, Ziniu Li, Ruoyu Sun, and Zhi-Quan Luo. Why transformers need adam: A hessian perspective. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024a. 1

Yushun Zhang, Congliang Chen, Ziniu Li, Tian Ding, Chenwei Wu, Yinyu Ye, Zhi-Quan Luo, and Ruoyu Sun. Adam-mini: Use fewer learning rates to gain more. arXiv preprint arXiv:2406.16793, 2024b. 20

Zhanpeng Zhou, Mingze Wang, Yuchen Mao, Bingrui Li, and Junchi Yan. Sharpness-aware minimization efficiently selects flatter minima late in training. In The Thirteenth International Conference on Learning Representations, 2025. 2, 3

Zhanxing Zhu, Jingfeng Wu, Bing Yu, Lei Wu, and Jinwen Ma. The anisotropic noise in stochastic gradient descent: Its behavior of escaping from sharp minima and regularization effects. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 7654–7663. PMLR, 09–15 Jun 2019. 2

## A TERMINOLOGIES

<table><tr><td>Terminology</td><td>General Meaning</td><td>Usage in This Paper</td></tr><tr><td>Sharpness</td><td>A measure of curvature in the loss landscape, often characterized via the Hessian. Different works may define it differently.</td><td>We define sharpness as the curvature along the sharpest direction of the loss landscape. Mathematically, it is presented as the largest eigenvalue of the Hessian  $\bar { \lambda } _ { \operatorname* { m a x } } \mathbf { \bar { ( H ( \pmb \theta _ { t } ) ) } } )$  or of the preconditioned curvature matrix  $\lambda _ { \operatorname* { m a x } } ( \mathbf { S } ( \pmb { \theta } _ { t } ) )$ </td></tr><tr><td>mum</td><td>Flat/sharp mini- A minimum is a point where the gradient vanishes  $\overline { { \nabla } } L ( \pmb \theta ) = 0$  and the loss does not decrease in a small neighborhood. A sharp min- imum has large curvature; a flat minimum has small curvature.</td><td>We use these terms sparingly and fol- low the standard definitions from the sharpness/flat-minima literature.</td></tr><tr><td>Wide/deep basin</td><td>A loss basin is a region of the landscape surrounding a minimum. A wide basin rises loss slowly in most directions, whereas a deep basin has a significantly lower min- imum value compared to its sur- roundings.</td><td>We use these terms to establish the depth-flatness trade-off: large noise scales tend to find wide basins, while small noise scales tend to find deeper regions with lower loss.</td></tr></table>

Table 1: Terminology and usage in this paper.

## B MISSING PROOF

## B.1 PHASE I: LYAPUNOV STABILITY ANALYSIS

Lemma B.1 (Stability Condition for Preconditioned GD). Define the preconditioned curvature matrix $\mathbf { S } ( \pmb { \theta } _ { k } ) : = \mathbf { M } ^ { 1 / 2 } \mathbf { H } ( \pmb { \theta } _ { k } ) \mathbf { M } ^ { 1 / 2 }$ , and let $\{ \lambda \} _ { i = 1 } ^ { p }$ be the eigenvalues of $\mathbf { S } ( \pmb { \theta } _ { k } )$ . The linear system in Equation (4) is asymptotically stable (i.e., lim $\boldsymbol { 1 } _ { k } \to _ { \infty } \boldsymbol { e } _ { k } = \mathbf { 0 } )$ if η satisfies

$$
0 < \eta < \frac { 2 } { \lambda _ { \mathrm { m a x } } ( \mathbf { S } ( \pmb { \theta } _ { k } ) } , \forall k \geq 0 .\tag{6}
$$

Proof. Since $\begin{array} { r } { \pmb { e } _ { k + 1 } = ( \mathbf { I } - \eta \mathbf { M } \mathbf { H } _ { k } ) \pmb { e } _ { k } } \end{array}$ , the linear system is asymptotically stable if all eigenvalues of $\mathbf { I } - \dot { \eta } \mathbf { M } \mathbf { H } _ { k }$ have magnitude less than 1. Note that:

$$
\mathbf { I } - \eta \mathbf { M } \mathbf { H } _ { k } = \mathbf { M } ^ { 1 / 2 } ( \mathbf { I } - \eta \mathbf { S } _ { k } ) \mathbf { M } ^ { - 1 / 2 } ,\tag{7}
$$

so the eigenvalues are $\mathrm { 1 } - \eta \lambda _ { j } ( \mathbf { S } _ { k } )$ . The stability condition $| 1 - \eta \lambda _ { j } | < 1$ for all $j$ is equivalent to:

$$
0 < \eta < \frac { 2 } { \lambda _ { \mathrm { m a x } } ( \mathbf { S } _ { k } ) } .\tag{8}
$$

□

Lemma B.2 (Exact one-step loss change). Define:

$$
\begin{array} { r l } & { { \mathbf S } ( \pmb \theta ) : = { \mathbf M } ^ { 1 / 2 } { \mathbf H } ( \pmb \theta ) { \mathbf M } ^ { 1 / 2 } , } \\ & { ~ { \pmb g } _ { k } : = { \mathbf M } ^ { 1 / 2 } \nabla L ( \pmb \theta _ { k } ) , } \\ & { ~ \pmb \delta _ { k } : = \pmb \theta _ { k + 1 } - \pmb \theta _ { k } = - \eta { \mathbf M } \nabla L ( \pmb \theta _ { k } ) . } \end{array}
$$

Then the true loss change can be written exactly as

$$
L ( \pmb \theta _ { k + 1 } ) - L ( \pmb \theta _ { k } ) = - \eta \| \nabla L ( \pmb \theta _ { k } ) \| _ { \mathbf { M } } ^ { 2 } + \eta ^ { 2 } \int _ { 0 } ^ { 1 } ( 1 - t ) \pmb \theta _ { k } ^ { \top } \mathbf { S } ( \pmb \theta _ { k } + t \delta _ { k } ) g _ { k } d t ,\tag{9}
$$

where $\| \nabla L ( \pmb \theta _ { k } ) \| _ { \mathbf { M } } ^ { 2 } : = \big ( \nabla L ( \pmb \theta _ { k } ) \big ) ^ { \top } \mathbf { M } \nabla L ( \pmb \theta _ { k } ) .$

Proof. Let $\delta _ { k } : = \pmb { \theta } _ { k + 1 } - \pmb { \theta } _ { k } = - \eta \mathbf { M } \nabla L ( \pmb { \theta } _ { k } )$ and define the scalar function

$$
\begin{array} { r } { \phi ( t ) : = L ( \theta _ { k } + t \delta _ { k } ) , \quad t \in [ 0 , 1 ] . } \end{array}
$$

Then

$$
L ( \pmb \theta _ { k + 1 } ) - L ( \pmb \theta _ { k } ) = \phi ( 1 ) - \phi ( 0 ) .
$$

Compute the derivatives:

$$
\begin{array} { l } { \boldsymbol { \phi } ^ { \prime } ( t ) = \left( \nabla L ( \pmb { \theta } _ { k } + t \delta _ { k } ) \right) ^ { \top } \delta _ { k } , } \\ { \boldsymbol { \phi } ^ { \prime \prime } ( t ) = \delta _ { k } ^ { \top } \mathbf { H } ( \pmb { \theta } _ { k } + t \delta _ { k } ) \delta _ { k } . } \end{array}
$$

By Taylor’s theorem with integral remainder:

$$
\phi ( 1 ) - \phi ( 0 ) = \phi ^ { \prime } ( 0 ) + \int _ { 0 } ^ { 1 } ( 1 - t ) \phi ^ { \prime \prime } ( t ) d t .
$$

Now evaluate at $t = 0$

$$
\phi ^ { \prime } ( 0 ) = \left( \nabla L ( \pmb { \theta } _ { k } ) \right) ^ { \top } \pmb { \delta } _ { k } = - \eta ( \nabla L ( \pmb { \theta } _ { k } ) ) ^ { \top } \mathbf { M } \nabla L ( \pmb { \theta } _ { k } ) = - \eta \| \nabla L ( \pmb { \theta } _ { k } ) \| _ { \mathbf { M } } ^ { 2 } .
$$

For the second derivative term:

$$
\phi ^ { \prime \prime } ( t ) = \delta _ { k } ^ { \top } \mathbf { H } ( \theta _ { k } + t \delta _ { k } ) \delta _ { k } = \eta ^ { 2 } \pmb { g } _ { k } ^ { \top } \mathbf { S } ( \theta _ { k } + t \delta _ { k } ) \pmb { g } _ { k } ,
$$

since $\delta _ { k } = - \eta \mathbf { M } \nabla L ( \pmb { \theta } _ { k } )$ and $\pmb { g } _ { k } = \mathbf { M } ^ { 1 / 2 } \nabla L ( \pmb { \theta } _ { k } )$ , and thus

$$
\begin{array} { r } { \pmb { \delta } _ { k } ^ { \top } \mathbf { H } ( \cdot ) \pmb { \delta } _ { k } = \eta ^ { 2 } \pmb { g } _ { k } ^ { \top } \mathbf { S } ( \cdot ) \pmb { g } _ { k } . } \end{array}
$$

Substituting both terms yields the result.

Lemma B.3 (One-step Loss Change). Let $\delta _ { k } : = \theta _ { k + 1 } - \theta _ { k }$ . Suppose that along the segment $\{ \pmb \theta _ { k } + \alpha \pmb \delta _ { k } : \alpha \in [ 0 , 1 ] \}$ , we have $0 \leq \lambda _ { \operatorname* { m i n } } ( \mathbf { S } ( \pmb \theta _ { k } + \alpha \pmb \delta _ { k } ) ) \leq \lambda _ { \operatorname* { m a x } } ( \mathbf { S } ( \pmb \theta _ { k } + \alpha \pmb \delta _ { k } ) ) \leq \Lambda _ { k }$ . Then,

$$
L ( \pmb \theta _ { k + 1 } ) - L ( \pmb \theta _ { k } ) \leq - \eta ( 1 - \frac { 1 } { 2 } \eta \Lambda _ { k } ) ( \nabla L ( \pmb \theta _ { k } ) ) ^ { \top } \mathbf { M } \nabla L ( \pmb \theta _ { k } ) .
$$

In particular, ${ i f \eta \leq 2 / \Lambda _ { k } , }$ , each update is guaranteed to non-increasing in loss, i.e., $L ( \theta _ { k + 1 } ) \leq$ $L ( \theta _ { k } )$ . Instead, $i f \eta \uparrow 2 / \Lambda _ { k }$ , the guaranteed decrease per step $| ( L ( \pmb \theta _ { k } ) - L ( \pmb \theta _ { k + 1 } ) ) / \eta  0 |$

Proof. From Lemma B.2, we have

$$
g _ { k } ^ { \top } \mathbf { S } ( \pmb { \theta } _ { k } + t \pmb { \delta } _ { k } ) \pmb { g } _ { k } \leq \Lambda _ { k } \lVert \pmb { g } _ { k } \rVert ^ { 2 }
$$

for all t, since $\mathbf { S } ( \cdot )$ is symmetric. Therefore,

$$
\begin{array} { r l r } {  { L ( \pmb { \theta } _ { k + 1 } ) - L ( \pmb { \theta } _ { k } ) \leq - \eta \| \nabla L ( \pmb { \theta } _ { k } ) \| _ { \mathbf { M } } ^ { 2 } + \eta ^ { 2 } \Lambda _ { k } \| \pmb { g } _ { k } \| ^ { 2 } \int _ { 0 } ^ { 1 } ( 1 - t ) d t } } \\ & { } & { = - \eta \| \nabla L ( \pmb { \theta } _ { k } ) \| _ { \mathbf { M } } ^ { 2 } + \frac { 1 } { 2 } \eta ^ { 2 } \Lambda _ { k } \| \pmb { g } _ { k } \| ^ { 2 } . } \end{array}
$$

Note that $\| \pmb { g } _ { k } \| ^ { 2 } = \| \nabla L ( \pmb { \theta } _ { k } ) \| _ { \mathbf { M } } ^ { 2 }$ , yielding the result.

## B.2 PHASE II: SDE ANALYSIS

## B.2.1 DISCRETE-TIME SOLUTION

Lemma B.4 (Eigenbasis Decomposition). Let $\mathbf { S } : = \mathbf { M } ^ { 1 / 2 } \mathbf { H } ( \pmb \theta ^ { \star } ) \mathbf { M } ^ { 1 / 2 }$ with eigendecomposition ${ \bf S } = { \bf Q } { \bf \Lambda } { \bf Q } ^ { \top } , { \bf \Lambda } { \bf \Lambda } = d i a g ( \lambda _ { 1 } , \ldots , \lambda _ { d } )$ . Define $\mathbf { G } : = \mathbf { Q } ^ { \top } \mathbf { M } ^ { 1 / 2 } \Sigma ( \pmb { \theta } ^ { \star } ) \mathbf { M } ^ { 1 / 2 } \mathbf { Q } / B$ . In coordinates $\pmb { w } _ { k } : = \mathbf { Q } ^ { \top } \mathbf { M } ^ { - 1 / 2 } \pmb { e } _ { k }$ , the recursion gives:

$$
\pmb { w } _ { k + 1 } = ( \mathbf { I } - \eta \pmb { \Lambda } ) \pmb { w } _ { k } + \eta \pmb { \zeta } _ { k } , \quad \mathbb { E } [ \zeta _ { k } \pmb { \zeta } _ { k } ^ { \top } ] = \mathbf { G }
$$

The stationary covariance $\Sigma _ { w }$ has diagonal elements:

$$
\left( \pmb { \Sigma } _ { w } \right) _ { j j } = \frac { \eta ^ { 2 } \mathbf { G } _ { j j } } { 1 - \left( 1 - \eta \lambda _ { j } \right) ^ { 2 } } = \frac { \eta \mathbf { G } _ { j j } } { 2 \lambda _ { j } - \eta \lambda _ { j } ^ { 2 } }\tag{10}
$$

Proof. First, we verify that $\mathbf { S } = \mathbf { M } ^ { 1 / 2 } \mathbf { H } ( \pmb { \theta } ^ { \star } ) \mathbf { M } ^ { 1 / 2 }$ can be eigendecomposed. Since both M and $\mathbf { H } ( \pmb \theta ^ { \star } )$ ) are positive definite matrices, S is also positive definite matrix. By the spectral theorem, S admits the eigendecomposition $\mathbf { S } = \mathbf { Q } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { Q } ^ { \top }$ , where Q is orthogonal and $\bar { \mathbf { \Lambda } } = \mathrm { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { d } )$ with $\lambda _ { i } > 0$

Starting from $\boldsymbol { e } _ { k + 1 } = \mathbf { A } \boldsymbol { e } _ { k } + \eta \mathbf { M } \xi _ { k }$ with $\mathbf { A } = \mathbf { I } - \eta \mathbf { M } \mathbf { H } ( \theta ^ { \star } )$ ), we change variables to $\scriptstyle w _ { k } \ =$ $\mathbf { Q } ^ { \top } \mathbf { M } ^ { - 1 / 2 } e _ { k }$

$$
\begin{array} { r l } & { \mathbf { \boldsymbol { w } } _ { k + 1 } = \mathbf { \boldsymbol { \mathbf { Q } } } ^ { \top } \mathbf { \boldsymbol { M } } ^ { - 1 / 2 } \boldsymbol { e } _ { k + 1 } = \mathbf { \boldsymbol { \mathbf { Q } } } ^ { \top } \mathbf { \boldsymbol { M } } ^ { - 1 / 2 } ( \mathbf { \boldsymbol { A } } \boldsymbol { e } _ { k } + \eta \mathbf { \boldsymbol { M } } \boldsymbol { \xi } _ { k } ) } \\ & { \qquad = \mathbf { \boldsymbol { \mathbf { Q } } } ^ { \top } \mathbf { \boldsymbol { M } } ^ { - 1 / 2 } ( \mathbf { \boldsymbol { I } } - \eta \mathbf { \boldsymbol { M } } \mathbf { \boldsymbol { H } } ( \boldsymbol { \theta } ^ { \star } ) ) \boldsymbol { e } _ { k } + \eta \mathbf { \boldsymbol { Q } } ^ { \top } \mathbf { \boldsymbol { M } } ^ { 1 / 2 } \boldsymbol { \xi } _ { k } } \\ & { \qquad = \mathbf { \boldsymbol { \mathbf { Q } } } ^ { \top } \mathbf { \boldsymbol { M } } ^ { - 1 / 2 } \boldsymbol { e } _ { k } - \eta \mathbf { \boldsymbol { \mathbf { Q } } } ^ { \top } \mathbf { \boldsymbol { M } } ^ { 1 / 2 } \mathbf { \boldsymbol { H } } ( \boldsymbol { \theta } ^ { \star } ) \boldsymbol { e } _ { k } + \eta \mathbf { \boldsymbol { Q } } ^ { \top } \mathbf { \boldsymbol { M } } ^ { 1 / 2 } \boldsymbol { \xi } _ { k } } \\ & { \qquad = \boldsymbol { w } _ { k } - \eta \mathbf { \boldsymbol { \mathbf { Q } } } ^ { \top } \mathbf { \boldsymbol { M } } ^ { 1 / 2 } \mathbf { \boldsymbol { H } } ( \boldsymbol { \theta } ^ { \star } ) \mathbf { \boldsymbol { M } } ^ { 1 / 2 } \mathbf { \boldsymbol { Q } } \boldsymbol { w } _ { k } + \eta \mathbf { \boldsymbol { Q } } ^ { \top } \mathbf { \boldsymbol { M } } ^ { 1 / 2 } \boldsymbol { \xi } _ { k } } \\ &  \qquad = \boldsymbol { w } _ { k } - \eta \mathbf { \boldsymbol { \mathbf { Q } } } ^ { \top } \mathbf { \boldsymbol { S } } \mathbf { \boldsymbol { Q } } \boldsymbol { w } _ { k } + \eta \mathbf  \boldsymbol  Q  \end{array}
$$

Defining $\zeta _ { k } : = \mathbf { Q } ^ { \top } \mathbf { M } ^ { 1 / 2 } \xi _ { k }$ , we get:

$$
\mathbb { E } [ \zeta _ { k } \zeta _ { k } ^ { \top } ] = { \mathbf { Q } } ^ { \top } { \mathbf { M } } ^ { 1 / 2 } \mathbb { E } [ \xi _ { k } \xi _ { k } ^ { \top } ] { \mathbf { M } } ^ { 1 / 2 } { \mathbf { Q } } = { \mathbf { Q } } ^ { \top } { \mathbf { M } } ^ { 1 / 2 } { \boldsymbol { \Sigma } } ( \theta ^ { \star } ) { \mathbf { M } } ^ { 1 / 2 } { \mathbf { Q } } / B = : { \mathbf { G } } ^ { \top } { \mathbf { M } } ^ { 1 / 2 } ,
$$

As matrix $\left( \mathbf { I } - \eta \pmb { \Lambda } \right)$ is diagonal, the recursion now decouples into independent scalar equations for each component j:

$$
( \pmb { w } _ { k + 1 } ) _ { j } = ( 1 - \eta \lambda _ { j } ) ( \pmb { w } _ { k } ) _ { j } + \eta ( \zeta _ { k } ) _ { j } .
$$

For each component $j ,$ the stationary variance satisfies:

$$
( \pmb { \Sigma } _ { w } ) _ { j j } = ( 1 - \eta \lambda _ { j } ) ^ { 2 } ( \pmb { \Sigma } _ { w } ) _ { j j } + \eta ^ { 2 } \mathbf { G } _ { j j }\tag{11}
$$

Solving for $( \pmb { \Sigma } _ { w } ) _ { j j } \colon$

$$
\begin{array} { c } { { ( { \pmb { \Sigma } } _ { w } ) _ { j j } = \displaystyle \frac { \eta ^ { 2 } { \bf G } _ { j j } } { 1 - ( 1 - \eta \lambda _ { j } ) ^ { 2 } } = \displaystyle \frac { \eta ^ { 2 } { \bf G } _ { j j } } { 1 - ( 1 - 2 \eta \lambda _ { j } + \eta ^ { 2 } \lambda _ { j } ^ { 2 } ) } } } \\ { { = \displaystyle \frac { \eta ^ { 2 } { \bf G } _ { j j } } { 2 \eta \lambda _ { j } - \eta ^ { 2 } \lambda _ { j } ^ { 2 } } = \displaystyle \frac { \eta { \bf G } _ { j j } } { 2 \lambda _ { j } - \eta \lambda _ { j } ^ { 2 } } } } \end{array}
$$

## B.2.2 CONTINUOUS-TIME LIMIT

We now take the continuous-time limit $( \eta  0 )$ to derive a simpler universal theory. The exact solution for the variance in the eigenbasis from Lemma B.4, i.e., $( \pmb { \Sigma } _ { w } ) _ { j j } = \eta \mathbf { G } _ { j j } / ( 2 \lambda _ { j } - \eta \lambda _ { j } ^ { 2 } )$ guides the necessary scaling for the continuous-time limit. Because $( \Sigma _ { w } ) _ { j j }$ converges to a finite non-zero value as $\eta  0$ , the numerator $\eta \mathbf { G } _ { j j }$ must remain finite. This suggests defining a quantity τ such that for each mode $j \colon$

$$
\eta \mathbf { G } _ { j j }  2 \tau \quad \mathrm { a s } \quad \eta  0 .
$$

We strengthen this to :

$$
\eta { \bf G }  2 \tau { \bf I } \mathrm { a s } \eta  0 .
$$

Recalling that $\mathbf { G } = \mathbf { Q } ^ { \top } \mathbf { M } ^ { 1 / 2 } \Sigma ( \pmb { \theta } ^ { \star } ) \mathbf { M } ^ { 1 / 2 } \mathbf { Q } / B$ , this condition in the original coordinate system translates to the required scaling for the noise covariance:

$$
\frac { \eta } { B } \mathbf { M } \Sigma ( \pmb { \theta } ^ { \star } ) \mathbf { M } ^ { \top }  2 \tau \mathbf { M } .
$$

Proposition B.1 (Convergence to SDE). Consider the scaled discrete process ${ \pmb \theta } _ { \lfloor t / \eta \rfloor } ~ a s ~ \eta  0 \mathrm { ~ }$ Suppose the noise covariance satisfies

$$
\frac { \eta } { B } \mathbf { M } \pmb { \Sigma } ( \pmb { \theta } ^ { \star } ) \mathbf { M } ^ { \top } = 2 \tau \mathbf { M } + O ( \eta ) ,\tag{12}
$$

for some temperature $\tau > 0 .$ . Then the process converges weakly to the Itô SDE:

$$
d \pmb { \theta } _ { t } = - \mathbf { M } \nabla L ( \pmb { \theta } _ { t } ) d t + \sqrt { 2 \tau } \mathbf { M } ^ { 1 / 2 } d W _ { t }\tag{13}
$$

where $W _ { t }$ is standard Brownian motion.

Proof. Consider the discrete preconditioned SGD update:

$$
\pmb { \theta } _ { k + 1 } = \pmb { \theta } _ { k } - \eta \mathbf { M } ( \nabla L ( \pmb { \theta } _ { k } ) + \pmb { \xi } _ { k } ) ,
$$

Define the scaled process $\pmb { \theta } ^ { ( \eta ) } ( t ) = \pmb { \theta } _ { \lfloor t / \eta \rfloor }$ . The increment $\Delta \pmb { \theta } _ { k } = \pmb { \theta } _ { k + 1 } - \pmb { \theta } _ { k }$ satisfies:

$$
\mathbb { E } [ \Delta \pmb { \theta } _ { k } \ | \ \pmb { \theta } _ { k } = \pmb { \theta } ] = - \eta \mathbf { M } \nabla L ( \pmb { \theta } ) ,
$$

$$
\mathbf { C o v } ( \Delta \pmb { \theta } _ { k } \mid \pmb { \theta } _ { k } = \pmb { \theta } ) = \frac { \eta ^ { 2 } } { B } \mathbf { M } \pmb { \Sigma } ( \pmb { \theta } ^ { \star } ) \mathbf { M } ^ { \top } .
$$

Given the scaling condition Equation (12), the covariance is ${ \cal { O } } ( \eta )$

The generator $\mathcal { L } ^ { ( \eta ) }$ of the discrete process for a smooth function $f$ is:

$$
\mathscr { L } ^ { ( \eta ) } \pmb { f } ( \pmb { \theta } ) = \frac { 1 } { \eta } \mathbb { E } [ \pmb { f } ( \pmb { \theta } _ { k + 1 } ) - \pmb { f } ( \pmb { \theta } _ { k } ) \ | \ \pmb { \theta } _ { k } = \pmb { \theta } ] .
$$

Using a Taylor expansion and taking conditional expectation:

$$
\mathbb { E } [ \pmb { f } ( \pmb { \theta } _ { k + 1 } ) - \pmb { f } ( \pmb { \theta } _ { k } ) \mid \pmb { \theta } ] = - \eta \nabla \pmb { f } ( \pmb { \theta } ) ^ { \top } \mathbf { M } \nabla L ( \pmb { \theta } ) + \frac { 1 } { 2 } \mathbb { E } [ ( \Delta \pmb { \theta } ) ^ { \top } \nabla ^ { 2 } \pmb { f } ( \pmb { \theta } ) \Delta \pmb { \theta } ] + O ( \eta ^ { 3 / 2 } ) .
$$

For the second term, with $\Delta \pmb { \theta } = - \eta M ( \nabla L ( \pmb { \theta } ) + \pmb { \xi } _ { k } )$

$$
\mathbb { E } [ ( \Delta \pmb { \theta } ) ^ { \top } \nabla ^ { 2 } \pmb { f } ( \pmb { \theta } ) \Delta \pmb { \theta } ] = \eta ^ { 2 } \mathbb { E } [ ( \nabla L ( \pmb { \theta } ) + \pmb { \xi } _ { k } ) ^ { \top } \mathbf { M } ^ { \top } \nabla ^ { 2 } \pmb { f } ( \pmb { \theta } ) \mathbf { M } ( \nabla L ( \pmb { \theta } ) + \pmb { \xi } _ { k } ) ]
$$

$$
\begin{array} { l } { { \displaystyle = \eta ^ { 2 } \mathbb { E } [ \xi _ { k } ^ { \top } \mathbf { M } ^ { \top } \nabla ^ { 2 } f ( \theta ) \mathbf { M } \xi _ { k } ] + O ( \eta ^ { 2 } ) } } \\ { { \displaystyle = \eta ^ { 2 } \mathrm { T r } \big ( \mathbf { M } ^ { \top } \nabla ^ { 2 } f ( \theta ) \mathbf { M } \mathbb { E } [ \xi _ { k } \xi _ { k } ^ { \top } ] ) + O ( \eta ^ { 2 } ) } } \\ { { \displaystyle = \frac { \eta ^ { 2 } } { B } \mathrm { T r } \big ( \mathbf { M } ^ { \top } \nabla ^ { 2 } f ( \theta ) \mathbf { M } \Sigma ( \theta ^ { \star } ) \big ) + O ( \eta ^ { 2 } ) } } \\ { { \displaystyle = \frac { \eta ^ { 2 } } { B } \mathrm { T r } \big ( \mathbf { M } \Sigma ( \theta ^ { \star } ) \mathbf { M } ^ { \top } \nabla ^ { 2 } f ( \theta ) \big ) + O ( \eta ^ { 2 } ) } } \end{array}
$$

where we used $\mathbb { E } [ \pmb { \xi } ^ { \top } \mathbf { A } \pmb { \xi } ] = \operatorname { T r } ( A \mathbb { E } [ \pmb { \xi } \pmb { \xi } ^ { \top } ] )$ and trace cyclicity $\operatorname { T r } ( \mathbf { A } \mathbf { B } \mathbf { C } ) = \operatorname { T r } ( \mathbf { C } \mathbf { A } \mathbf { B } )$ .

Therefore:

$$
\frac { 1 } { 2 } \mathbb { E } [ ( \Delta \pmb { \theta } ) ^ { \top } \nabla ^ { 2 } \pmb { f } ( \pmb { \theta } ) \Delta \pmb { \theta } ] = \frac { \eta ^ { 2 } } { 2 B } \mathrm { T r } ( \mathbf { M } \Sigma ( \pmb { \theta } ^ { \star } ) \mathbf { M } ^ { \top } \nabla ^ { 2 } \pmb { f } ( \pmb { \theta } ) ) + O ( \eta ^ { 2 } ) .
$$

Using the scaling condition Equation (12), we have :

$$
\frac { \eta ^ { 2 } } { 2 B } \mathrm { T r } ( \mathbf { M } \Sigma ( \theta ^ { \star } ) \mathbf { M } ^ { \top } \nabla ^ { 2 } f ( \theta ) ) = \frac { \eta } { 2 } \mathrm { T r } \left( 2 \tau \mathbf { M } \nabla ^ { 2 } f ( \theta ) \right) + O ( \eta ^ { 2 } ) = \eta \tau \mathrm { T r } ( \mathbf { M } \nabla ^ { 2 } f ( \theta ) ) + O ( \eta ^ { 2 } ) .
$$

Thus,

$$
\begin{array} { r } { \begin{array} { r } { \mathcal { L } ^ { ( \eta ) } \pmb { f } ( \pmb { \theta } ) = - \nabla \pmb { f } ( \pmb { \theta } ) ^ { \top } \mathbf { M } \nabla L ( \pmb { \theta } ) + \tau \mathrm { T r } ( \mathbf { M } \nabla ^ { 2 } \pmb { f } ( \pmb { \theta } ) ) + O ( \eta ) . } \end{array} } \end{array}
$$

As $\eta \to 0 , \mathcal { L } ^ { ( \eta ) } f ( \pmb { \theta } )$ converges to:

$$
\mathcal { L } \pmb { f } ( \pmb { \theta } ) = - \nabla \pmb { f } ( \pmb { \theta } ) ^ { \top } \mathbf { M } \nabla L ( \pmb { \theta } ) + \tau \mathrm { T r } ( \mathbf { M } \nabla ^ { 2 } \pmb { f } ( \pmb { \theta } ) ) ,
$$

which is the generator of the Itô SDE:

$$
d \pmb { \theta } _ { t } = - \mathbf { M } \nabla L ( \pmb { \theta } _ { t } ) d t + \sqrt { 2 \tau } \mathbf { M } ^ { 1 / 2 } d W _ { t } .
$$

By the weak convergence theory $( \mathrm { e . g . }$ , via the martingale problem or generator convergence), the process $\theta ^ { ( \eta ) } ( t )$ converges weakly to the solution of this SDE. □

Proposition B.2 (Gibbs Stationary Distribution). The SDE in Equation (13) has stationary distribution:

$$
p _ { \infty } ( \pmb \theta ) \propto \mathrm { e x p } ( - L ( \pmb \theta ) / \tau )\tag{14}
$$

Proof. The generator of the SDE (5) is $\mathcal { L } f = - \mathbf { M } \nabla L \cdot \nabla f + \tau \mathbf { t r } ( \mathbf { M } \nabla ^ { 2 } f )$ . The Fokker-Planck equation for the probability density $p ( t , \pmb \theta )$ is:

$$
\partial _ { t } p = \mathcal { L } ^ { * } p = \nabla \cdot \left( \mathbf { M } \nabla L p \right) + \tau \nabla \cdot \left( \mathbf { M } \nabla p \right)
$$

where $\mathcal { L } ^ { \ast }$ is the adjoint operator. Setting $\partial _ { t } p = 0$ for stationarity:

$$
\begin{array} { r l } & { 0 = \nabla \cdot \left( \mathbf { M } \nabla L p _ { \infty } \right) + \tau \nabla \cdot \left( \mathbf { M } \nabla p _ { \infty } \right) } \\ & { \quad = \nabla \cdot \left( \mathbf { M } \nabla L p _ { \infty } + \tau \mathbf { M } \nabla p _ { \infty } \right) } \end{array}
$$

This implies the current $J = \mathbf { M } \nabla L p _ { \infty } + \tau \mathbf { M } \nabla p _ { \infty }$ has zero divergence. For a potential-driven system, we require $J = \mathbf { 0 }$

$$
\begin{array} { c } { { \bf M } \nabla L p _ { \infty } + \tau { \bf M } \nabla p _ { \infty } = { \bf 0 } \qquad } \\ { \nabla L p _ { \infty } + \tau \nabla p _ { \infty } = { \bf 0 } \quad ( \mathrm { s i n c e ~ } { \bf M } \succ 0 ) } \\ { \displaystyle \frac { \nabla p _ { \infty } } { p _ { \infty } } = - \frac { \nabla L } { \tau } \qquad } \end{array}
$$

Integrating: log $p _ { \infty } = - L / \tau + \mathrm { c o n s t }$ , which gives Equation (14).

Theorem B.1 (Free Energy Minimization). Let the empirical risk $L ( \theta )$ admit multiple local minima $\{ \pmb { \theta } _ { i } ^ { \star } \} _ { i = 1 } ^ { m }$ with Hessians $\bf \\\ddot { H } ( \pmb { \theta } _ { \it i } ^ { \star } ) \succ \ 0$ . Under the SDE in Equation (13) with temperature $\tau ,$ , the stationary probability that training resides in basin i is given by:

$$
P _ { \tau } ( b a s i n i ) = \frac { \exp ( - F _ { i } ( \tau ) / \tau ) } { \sum _ { j } \exp ( - F _ { j } ( \tau ) / \tau ) } , \quad F _ { i } ( \tau ) : = L ( \pmb { \theta } _ { i } ^ { \star } ) + \frac { \tau } { 2 } \log \operatorname* { d e t } \mathbf { H } ( \pmb { \theta } _ { i } ^ { \star } ) .\tag{15}
$$

Proof. From the Gibbs distribution Equation (14), the probability mass in basin i is:

$$
P _ { \tau } ( \mathbf { b a s i n } i ) = \frac { \int _ { B _ { i } } e ^ { - L ( \pmb \theta ) / \tau } d \pmb \theta } { \int _ { \mathbb { R } ^ { d } } e ^ { - L ( \pmb \theta ) / \tau } d \pmb \theta }
$$

where $B _ { i }$ is the basin of attraction around minimum $\pmb { \theta } _ { i } ^ { \star }$

For the numerator, using the quadratic approximation $\begin{array} { r } { L ( \pmb \theta ) = L ( \pmb \theta _ { i } ^ { \star } ) + \frac { 1 } { 2 } ( \pmb \theta - \pmb \theta _ { i } ^ { \star } ) ^ { \top } \mathbf H ( \pmb \theta _ { i } ^ { \star } ) ( \pmb \theta - \pmb \theta _ { i } ^ { \star } ) } \end{array}$ in basin i we get:

$$
\begin{array} { r l r } & { } & { \displaystyle \int _ { B _ { i } } e ^ { - L ( \theta ) / \tau } d \theta = \int _ { \mathbb { R } ^ { d } } \exp \left( - \frac { L ( \theta _ { i } ^ { \star } ) } { \tau } - \frac { 1 } { 2 \tau } ( \theta - \theta _ { i } ^ { \star } ) ^ { \top } { \bf H } ( \theta _ { i } ^ { \star } ) ( \theta - \theta _ { i } ^ { \star } ) \right) d \theta } \\ & { } & { = e ^ { - L ( \theta _ { i } ^ { \star } ) / \tau } \displaystyle \int _ { \mathbb { R } ^ { d } } \exp \left( - \frac { 1 } { 2 \tau } ( \theta - \theta _ { i } ^ { \star } ) ^ { \top } { \bf H } ( \theta _ { i } ^ { \star } ) ( \theta - \theta _ { i } ^ { \star } ) \right) d \theta } \end{array}
$$

The integral is a multivariate Gaussian with covariance $\tau \mathbf { H } ( \pmb { \theta } _ { i } ^ { \star } ) ^ { - 1 }$ . Using the standard formula for Gaussian integrals:

$$
\int _ { \mathbb { R } ^ { d } } \exp \left( - { \frac { 1 } { 2 } } y ^ { \top } \Sigma ^ { - 1 } y \right) d y = ( 2 \pi ) ^ { d / 2 } ( \operatorname* { d e t } \Sigma ) ^ { 1 / 2 }
$$

With $\pmb { \Sigma } = \tau \mathbf { H } ( \pmb { \theta } _ { i } ^ { \star } ) ^ { - 1 }$ , we have det $\pmb { \Sigma } = \tau ^ { d } ( \operatorname* { d e t } \mathbf { H } ( \pmb { \theta } _ { i } ^ { \star } ) ) ^ { - 1 }$ and $\Sigma ^ { - 1 } = \tau ^ { - 1 } \mathbf { H } ( \pmb { \theta } _ { i } ^ { \star } )$

$$
\begin{array} { l } { \displaystyle \int _ { \mathbb { R } ^ { d } } \exp \left( - \frac { 1 } { 2 \tau } ( \pmb { \theta } - \pmb { \theta } _ { i } ^ { \star } ) ^ { \top } \mathbf { H } ( \pmb { \theta } _ { i } ^ { \star } ) ( \pmb { \theta } - \pmb { \theta } _ { i } ^ { \star } ) \right) d \pmb { \theta } = ( 2 \pi ) ^ { d / 2 } ( \tau ^ { d } ( \operatorname* { d e t } \mathbf { H } ( \pmb { \theta } _ { i } ^ { \star } ) ) ^ { - 1 } ) ^ { 1 / 2 } } \\ { \displaystyle \qquad = ( 2 \pi \tau ) ^ { d / 2 } ( \operatorname* { d e t } \mathbf { H } ( \pmb { \theta } _ { i } ^ { \star } ) ) ^ { - 1 / 2 } } \end{array}
$$

Therefore:

$$
\begin{array} { r l } & { \displaystyle \int _ { B _ { i } } e ^ { - L ( \theta ) / \tau } d \theta = e ^ { - L ( \theta _ { i } ^ { \star } ) / \tau } ( 2 \pi \tau ) ^ { d / 2 } ( \operatorname* { d e t } \mathbf { H } ( \theta _ { i } ^ { \star } ) ) ^ { - 1 / 2 } } \\ & { \quad \quad \quad = ( 2 \pi \tau ) ^ { d / 2 } \exp \left( - L ( \theta _ { i } ^ { \star } ) / \tau - \frac { 1 } { 2 } \log \operatorname* { d e t } \mathbf { H } ( \theta _ { i } ^ { \star } ) \right) } \\ & { \quad \quad = ( 2 \pi \tau ) ^ { d / 2 } \exp \left( - \frac { 1 } { \tau } \left( L ( \theta _ { i } ^ { \star } ) + \frac { \tau } { 2 } \log \operatorname* { d e t } \mathbf { H } ( \theta _ { i } ^ { \star } ) \right) \right) } \\ & { \quad \quad = ( 2 \pi \tau ) ^ { d / 2 } \exp ( - F _ { i } ( \tau ) / \tau ) } \end{array}
$$

Similarly, the total partition function is:

$$
\begin{array} { c } { { \displaystyle Z ( \tau ) = \int _ { \mathbb { R } ^ { d } } e ^ { - L ( \pmb \theta ) / \tau } d \pmb \theta = d \sum _ { j = 1 } ^ { m } \int _ { B _ { j } } e ^ { - L ( \pmb \theta ) / \tau } d \pmb \theta } } \\ { { = ( 2 \pi \tau ) ^ { d / 2 } \sum _ { j = 1 } ^ { m } \exp ( - F _ { j } ( \tau ) / \tau ) } } \end{array}
$$

Therefore:

$$
P _ { \tau } ( \mathtt { b a s i n } i ) = \frac { ( 2 \pi \tau ) ^ { d / 2 } \exp ( - F _ { i } ( \tau ) / \tau ) } { ( 2 \pi \tau ) ^ { d / 2 } \sum _ { j } \exp ( - F _ { j } ( \tau ) / \tau ) } = \frac { \exp ( - F _ { i } ( \tau ) / \tau ) } { \sum _ { j } \exp ( - F _ { j } ( \tau ) / \tau ) }
$$

This completes the proof of the free energy formula Equation (15).

## C EXPERIMENTAL SETUPS AND MORE RESULTS

## C.1 EXPERIMENTAL SETUPS

Models. We utilize two popular classes of LLM models for our pre-training experiments:

• GPT-2. We use GPT-2 (small) model (Radford et al., 2019), implemented via the nanoGPT code base (Karpathy, 2022). Following nanoGPT, the model employs Gaussian Error Linear Unit (GELU) activations and standard Layer Normalization (LayerNorm). Detailed model configurations are provided in Table 2.

• LLaMA. LLaMA (Touvron et al., 2023) is another popular decoder-only Transformer architecture, incorporating Rotary Positional Encoding (RoPE) (Su et al., 2024), Swish-Gated Linear Unit (SwiGLU), and Root mean square layer normalization (RMSNorm). For implementation, we utilize the LLaMA code from HuggingFace Transformers Library (Wolf et al., 2020). Additional model configurations are detailed in Table 2.

Datasets. Training is performed on the FineWeb-Edu dataset (Penedo et al., 2024). We adopt the a subset randomly sampled from the whole dataset of around 100B GPT-2 tokens. The same dataset has been widely used in literature on LLM pre-training.

Optimizers. To generalize our findings across different optimizers, we choose:

• AdamW. AdamW (Kingma & Ba, 2014) is adopted with hyperparameters $\beta _ { 1 } = 0 . 9 5 , \beta _ { 2 } =$ 0.95, and weight decay 0.1.

• Muon. Muon (Keller et al., 2024) is used with momentum of 0.95 and weight decay 0.1.

• Adam-mini. The hyperparameter of Adam-mini (Zhang et al., 2024b) is the same as AdamW.

• Lion. Lion (Chen et al., 2024) is used with hyperparameters $\beta _ { 1 } = 0 . 9 5 , \beta _ { 2 } = 0 . 9 8 .$ . The LR of Lion η is divided by 10× compared with the LR of AdamW in the same experiments, and the weight decay λ is ramped up to 10× to keep the effective LR λη = 0.1.

All these optimizers are used with gradient clipping at 1.0 for stability.

Table 2: Model configurations.
<table><tr><td>Acronym</td><td>Size</td><td> $d _ { \mathrm { m o d e l } }$ </td><td> $d _ { \mathrm { F F } }$ </td><td>n_head</td><td>depth</td></tr><tr><td>GPT-2 (small)</td><td>124M</td><td>768</td><td>3072</td><td>12</td><td>12</td></tr><tr><td>LLaMA (93M)</td><td>93M</td><td>512</td><td>2048</td><td>16</td><td>8</td></tr><tr><td>LLaMA (170M)</td><td>170M</td><td>768</td><td>3072</td><td>12</td><td>8</td></tr><tr><td>LLaMA (270M)</td><td>270M</td><td>1024</td><td>4096</td><td>16</td><td>8</td></tr><tr><td>LLaMA (530M)</td><td>530M</td><td>1536</td><td>6144</td><td>24</td><td>8</td></tr></table>

## C.2 MORE RESULTS UNDER VARIOUS SETUPS.

In this section, we extend our findings to other architectures, optimization algorithms, and larger training scales. Due to computational constraints, we primarily focus on validating the sharp-to-flat early dynamics and the proposed BS scheduling principle across these settings. We also explore the warmup–tuning recipe on additional architectures.

Extension to GPT-2 Architectures. See Figures 9 and 10 for details.

Extension to Other Optimizers. See Figure 11 for Adam-mini, and see Figure 12 for Lion.

Extension to Larger Models. See Figure 13 for models with 270M and 530M parameters.

## C.3 ABLATION STUDIES ON EARLY INSTABILITIES.

In this section, we conduct ablation studies on the root cause of instabilities, such as loss spikes and plateaus, observed in early training.

![](images/b05eba680dd8f1a4e4d0f6fc84acf062a1e75389c8b0b50047a1f0d0c9586ad6.jpg)

![](images/06796dc5834ebb3c68757aaf6c1c0b011e98916f56c9a83f324dc0745dd03da4.jpg)  
Tokens (B)

Figure 9: Extensive to GPT-2 architectures. (Left). Early sharp-to-flat dynamics. Evolution of the top eigenvalues of the Hessian across iterations: $\lambda _ { i } ( \mathbf { H } ( \pmb \theta _ { t } ) )$ vs. iteration t. (Right). BS scheduling improves data efficiency. BS scheduling improves data efficiency. We use a BS schedule that starts at 0.49M and increases by 4× at each ramp. Models are trained with 1, 2, or 3 ramping steps, while models without ramping serve as the baseline. Vertical gray dashed lines indicate ramping positions.  
![](images/f52f35e55a93d06938dc7dba89930e439b0127341111d638699a6fa456cfcfa7.jpg)  
Figure 10: Extensive to GPT-2 architectures. Larger Peak LR, Longer Warmup. We train a series of $\mathrm { G P T } { - 2 }$ models with 100 TPP. We vary the peak LRs η and warmup lengths $T _ { w }$ . We plot the best validation loss $L ( \theta _ { \mathrm { b s t } } )$ vs. $T _ { w }$ for different $\eta .$ The optimal $T _ { w }$ is highlighted with a star.

![](images/bc0a4c62434876ba71cc2bb19e2f1cc2c85128831d74ba42070082c4c024ef8e.jpg)

![](images/cd86d639d840abe143d069eb9ba630f5571299f87db55236956bab406a09fdf0.jpg)  
Tokens (B)  
Figure 11: Extensive to Adam-mini optimizer. (Left). Early sharp-to-flat dynamics. Evolution of the top eigenvalues of the Hessian across iterations: $\lambda _ { i } ( \mathbf { H } ( \pmb \theta _ { t } ) )$ vs. iteration t. (Right). BS scheduling improves data efficiency. BS scheduling improves data efficiency. We use a BS schedule that starts at 0.49M and increases by 4× at each ramp. Models are trained with 1, 2, or 3 ramping steps, while models without ramping serve as the baseline. Vertical gray dashed lines indicate ramping positions.

Is it the unstable optimizer? To disentangle optimizer-induced instability from landscape-induced instability, we repeated the experiments using Muon, a substantially more stable optimizer than AdamW. In Figure 14, the loss spikes and plateaus consistently occurs under Muon when warmup is shortened or the peak LR is increased. This rules out the possibility that the behavior stems from AdamW’s startup issues.

![](images/4d1638a12f56bea25489f489519e4aa0a7c92b508d3234ad0eeddebc684e1ddf.jpg)

![](images/185770e86a3e9b05567f83651b4dd449a74344b5889cd7f0ea6ee85a7cc2e209.jpg)

Figure 12: Extensive to Lion optimizer. (Left). Early sharp-to-flat dynamics. Evolution of the top eigenvalues of the Hessian across iterations: $\lambda _ { i } ( { \bf H } ( \pmb \theta _ { t } ) )$ vs. iteration t. (Right). BS scheduling improves data efficiency. BS scheduling improves data efficiency. We use a BS schedule that starts at 0.49M and increases by 4× at each ramp. Models are trained with 1, 2, or 3 ramping steps, while models without ramping serve as the baseline. Vertical gray dashed lines indicate ramping positions.  
![](images/1ed27ad082c8b11362d15374ebc7f25616378f10362095b13b3c3aa0a335e0a7.jpg)

![](images/294db19181e80b7f6214ef53310c39ebf775d2a177736e4c4b612a7cfba851fa.jpg)

Figure 13: Extensive to Larger Scale. (Left). Early sharp-to-flat dynamics. Evolution of the top eigenvalues of the Hessian across iterations: $\lambda _ { i } ( \mathbf { H } ( \pmb \theta _ { t } ) )$ vs. iteration t. (Right). BS scheduling improves data efficiency. BS scheduling improves data efficiency. We use a BS schedule that starts at 0.49M and increases by 4× at each ramp. Models are trained with 1, 2, or 3 ramping steps, while models without ramping serve as the baseline. Vertical gray dashed lines indicate ramping positions.  
![](images/d37ad59af3ab7221b05388f622e9783afd39c6d85729e6e70d56398e0bd1c2f3.jpg)

![](images/a8a253c0ac71a5e1f593c1748a451e870e32c25528b8d521a8c7e6d99ae47fbc.jpg)  
Figure 14: Muon consistently shows loss spikes and plateaus early in training. We train a series of LLaMA-2 models with 170M parameters. We adopt a warmup-stable schedule, where the warmup length is shortened to 16 iterations and the peak LR is varied, $\eta \in \{ 2 ^ { - 1 1 } , 2 ^ { - 1 0 } , 2 ^ { - 9 } , 2 ^ { - 8 } , 2 ^ { - 7 } \}$ (Left). LR schedule: $\eta _ { t }$ vs. training iteration t. (Middle, Right). Training loss curves for different model sizes: $L ( \pmb \theta _ { t } )$ vs. training iteration t. The vertical dashed line marks the end of the warmup phase.

What if we use longer warmup? We vary only the warmup length while fixing the peak LR at $2 ^ { - 7 }$ In Figure 15 (left), shorter warmup lengths lead to higher possibilities of loss spikes This behavior is consistent with the sharp-to-flat dynamics. Early in training, the model resides in sharper regions of the landscape, where only sufficiently small LRs ensure stable updates. Therefore, warmup is needed to gradually increase the LR until the trajectory enters flatter regions that can tolerate larger LRs.

What if zero warmup and small BS? We also conduct experiments with no warmup and small batch size. In Figure 15 (right), the loss spikes become even more significant. This aligns with our explanation: with no warmup, the LR jumps immediately to a large value while sharpness is still extremely high, causing a spike almost at initialization. More importantly, small BS does not replace warmup, and the instability still appears because of the sharpness.

![](images/9ecf4da45b479f39430779ddc3645206de8762dd254defec380650234f83e35c.jpg)

![](images/1135595e9ab0aeee23418f55c56c023c54763d69d867c3b36f52e861af34286c.jpg)  
Figure 15: (Left.) Shorter warmup, more loss spikes. We train a series of $\mathtt { L L a M A - } 2$ models with 170M parameters. We adopt a warmup-stable schedule, where the warmup length varies from $\{ 2 ^ { 5 } , 2 ^ { 6 } , 2 ^ { 7 } , \dot { 2 } ^ { 8 } , 2 ^ { 9 } , 2 ^ { 1 0 } \}$ iterations and the peak LR is fixed $\eta \ : = \ : 2 ^ { - 7 }$ . (Right). Zero warmup and small BS leads to larger loss spikes. We train a series of $\mathtt { L I a M A - } 2$ models with 170M parameters. We adopt a constant LR schedule, where no warmup and the peak LR is varied, $\eta \in \{ 2 ^ { - 1 1 } , 2 ^ { - 1 0 } , 2 ^ { - 9 } , 2 ^ { - 8 } , 2 ^ { - 7 } \}$ . We also use the 0.49M BS.