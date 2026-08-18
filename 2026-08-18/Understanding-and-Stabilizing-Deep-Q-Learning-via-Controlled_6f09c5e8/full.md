# Understanding and Stabilizing Deep Q-Learning via Controlled Bootstrapping and Regulated Value Dynamics

Bozhou Chen, Yongyi Wang, Hanyu Liu, Xionghui Yang, and Wenxin Li

Abstract—Deep Q-learning (DQL) has achieved remarkable empirical success in reinforcement learning, yet its training process remains notoriously unstable. Existing studies often attribute instability to isolated factors such as overestimation bias or representation learning issues, lacking a unified understanding of how different sources of instability interact during recursive value estimation. In this work, we provide a systematic analysis of instability in deep Q-learning from three complementary perspectives: operator-level bias in Bellman bootstrapping, estimator-level sensitivity of greedy action selection to regression noise, and parameter-dynamics imbalance under aggressive data reuse. We identify a reward-triggered self-reinforcing trap and characteristic parameter spike dynamics, then derive stabilization principles for controlled bootstrapping, ensemble quantile estimation, and spike-based parameter regulation. Experiments on Atari-100K and Procgen demonstrate competitive performance and improved training stability.

Index Terms—Deep reinforcement learning, Deep Q-learning, Bellman bootstrapping, Ensemble learning, Distributional reinforcement learning, Training stability, Parameter regulation

## 1 INTRODUCTION

D <sup>EEP</sup> <sup>Q-learning</sup> <sup>has</sup> <sup>become</sup> <sup>a</sup> <sup>cornerstone</sup> <sup>method</sup> <sup>in</sup>deep reinforcement learning (RL), widely applied in deep reinforcement learning (RL), widely applied in domains ranging from video games to robotics. Its appeal lies in learning effective policies directly from high-dimensional observations via neural network approximation.

A large body of prior work has studied instability in value learning from specific perspectives. Classical analyses focus on overestimation bias induced by the max operator in Bellman updates. Standard Q-learning applies the maximization operator to noisy value estimates, which leads to systematically optimistic targets. Double Q-learning [1] and its deep variant Double DQN [2] mitigate this issue by decoupling action selection from evaluation. More recent methods further reduce bias or variance through ensemble estimation or distributional value modeling.

However, empirical evidence suggests that instability in deep Q-learning cannot be fully explained by maximization bias alone. In practice, value learning exhibits complex feedback effects involving bootstrap targets, stochastic regression, and evolving network representations. These mechanisms interact with each other through the recursive structure of value updates, yet they are typically studied in isolation.

In this work, we revisit instability in deep Q-learning from a unified perspective and identify three interacting mechanisms that jointly shape learning dynamics:

Operator-level instability. Recursive Bellman bootstrapping may introduce structural bias in target construction. Beyond classical overestimation, we observe a previously overlooked feedback mechanism, which we term the selfreinforcing trap. In reward-bearing transitions, value amplification induced by positive rewards interacts with actionconditioned representation generalization. As a result, the same action may be repeatedly selected in bootstrap targets, creating a feedback loop that continuously amplifies its value.

Estimator-level instability. Deep Q-learning is fundamentally a regression problem in which the value function is learned through stochastic approximation. Even when estimation noise is unbiased, greedy action selection depends on relative value differences. When action gaps are small, regression noise may alter greedy decisions, which subsequently changes the distribution of collected experience. In this way, estimation noise propagates through the control loop and may destabilize learning dynamics.

Parameter-dynamics instability. We further observe that neural networks used for value approximation may gradually develop imbalanced parameter distributions during training, particularly under high replay ratios. In such regimes, a small subset of parameters may grow disproportionately large while the majority remain relatively unchanged. This phenomenon is closely related to the loss of network plasticity [3], [4], [5]. To better characterize this effect, we introduce a simple diagnostic statistic termed the spike ratio, which measures the concentration of extreme parameter magnitudes within each layer and serves as a practical indicator of parameter imbalance.

These observations suggest that instability in deep Qlearning arises from interacting feedback mechanisms across target construction, value estimation, and parameter evolution, requiring coordinated control rather than isolated heuristics. Based on this analysis, we derive several practical stabilization principles for value learning. These include controlling recursive amplification in bootstrap targets, reducing decision variance in value regression, and maintaining parameter plasticity during prolonged training. We instantiate these principles in a practical learning framework that integrates controlled bootstrapping, ensemble quantile regression, and parameter regulation.

Extensive experiments on Atari-100K and Procgen demonstrate that the resulting algorithm achieves strong performance and improved training robustness. More importantly, the empirical results support the proposed instability analysis and illustrate how the identified mechanisms influence learning dynamics in practice.

Our main contributions are summarized as follows:

• We provide a systematic analysis of instability in deep Q-learning, identifying three interacting mechanisms: operator-level bias in bootstrapping, estimator-level decision sensitivity to regression noise, and parameterdynamics imbalance during training.

• We uncover a previously overlooked feedback mechanism termed the self-reinforcing trap, where rewarddriven value amplification interacts with representation generalization to produce biased bootstrap dynamics.

• We reveal the role of parameter spike dynamics in the loss of network plasticity and introduce the spike ratio as a practical statistic for monitoring training stability.

• Based on these insights, we derive stabilization principles for value learning and instantiate them in a practical algorithm that achieves competitive performance on Atari-100K and Procgen benchmarks.

The remainder of this paper is as follows. Section 2 reviews related work on bootstrapping bias, distributional learning, ensemble estimation, etc. Section 3 presents the instability analysis from operator-level, estimator-level, and parameter-dynamics perspectives. Section 4 introduces stabilization principles derived from this analysis, and Section 5 presents their algorithmic instantiation. Section 6 provides experimental evaluation and mechanism studies. Finally, Section 7 concludes the paper and discusses future directions.

## 2 RELATED WORK

## 2.1 Overestimation and Stable Bootstrapping Targets

The maximization operator in the Bellman update introduces systematic overestimation when applied to noisy value predictions. Thrun and Schwartz [6] first identified this bias in the tabular setting, and under function approximation such bias may accumulate through recursive bootstrapping. Double Q-learning [1] mitigates this issue by decoupling action selection from evaluation using two estimators, and Double DQN [2] extends this idea to deep RL using online and target networks. Subsequent works further refine bootstrap target construction. Averaged-DQN [7] reduces variance by averaging past networks, while Maxmin DQN [8] selects the minimum among an ensemble of critics to control optimism. In continuous control, TD3 [9] adopts clipped double Qlearning, taking the minimum of two critics as a conservative target.

Ensemble-based strategies provide additional bias control. REM [10] enforces Bellman consistency on random convex combinations of Q-heads, and REDQ [11] leverages larger ensembles with randomized subsampling to stabilize training under high update-to-data ratios.

These approaches primarily stabilize value learning at the level of bootstrap target construction by reducing maximization bias or target variance. However, they mainly address the bias properties of the Bellman target itself, while other factors affecting the dynamics of recursive value learning remain less explored.

## 2.2 Distributional Reinforcement Learning and Ensemble Estimation

Distributional RL models, such as C51 [12], the full return distribution rather than only its expectation, capturing richer information about uncertainty in value estimation. C51 introduced a categorical representation over a fixed support and demonstrated improved stability on Atari. QR-DQN [13] replaced categorical projections with quantile regression, while IQN [14] parameterized the entire quantile function to allow flexible distributional modeling. In continuous control, TQC [15] combines distributional critics with an ensemble of Q-functions and truncates the highest quantile atoms to suppress overestimation, providing a softer alternative to scalar minimum operators.

Ensemble techniques improve both value stability and exploration. Bootstrapped DQN [16] trained multiple Qheads with bootstrapped data partitions, enabling temporally consistent deep exploration. REM [10] performs ensemble aggregation across critics to obtain more accurate value estimates. SUNRISE [17] introduced uncertainty-weighted Bellman backups, down-weighting unreliable targets to reduce error propagation. SPQR [18] explicitly addressed ensemble collapse by injecting structured noise to preserve independent estimation behavior.

These methods improve robustness at the estimation level by modeling return distributions or aggregating multiple value predictors. However, they are primarily motivated by improving prediction accuracy or uncertainty estimation, while the role of regression noise in affecting decision and interaction dynamics remains less explicitly analyzed.

## 2.3 Value Learning in Data-Efficient Regimes

Recent progress in data-efficient reinforcement learning has significantly improved performance under limited interaction budgets, such as the Atari-100K setting. A central trend in this regime is the integration of strong representation learning objectives into value-based methods.

CURL [19] introduced contrastive representation learning for pixel-based RL, improving sample efficiency through latent consistency. SPR [20] and SR-SPR [21] further leveraged self-predictive representations by enforcing temporal consistency in latent space. DrQ [22] demonstrated that simple data augmentation substantially improves data efficiency and generalization. More recent methods such as BBF [23], SGF [24], Drama [25], IRIS [26], and STORM [27] combined stronger regularization, auxiliary objectives, and architectural refinements to further push data-efficient performance.

In addition to representation learning, high replay ratios and aggressive update-to-data strategies have been widely adopted in data-limited regimes. DER [28] and BBF [23] showed that increasing gradient updates per environment interaction can substantially improve sample efficiency. However, heavy data reuse may also amplify bootstrap errors and increase sensitivity to estimation noise, particularly under recursive value updates.

These methods substantially improve sample efficiency through enhanced representation learning and increased data reuse. However, instability still fundamentally originates from recursive bootstrapping dynamics.

## 2.4 Optimization Dynamics and Network Plasticity

Beyond bias in bootstrap targets, a growing body of work has investigated instability in deep reinforcement learning from the perspective of optimization dynamics and network behavior during training. Early stabilization techniques primarily aim to mitigate the feedback effects introduced by bootstrapped updates. Target networks [29] provide a slowly updated objective to reduce oscillations in value estimates. Additional techniques such as the Huber loss [29] limit the influence of extreme TD errors, and safe backup operators including Retrace(λ) truncate or down-weight multi-step returns to prevent uncontrolled value propagation.

Subsequent studies examined instability from the perspective of optimization under function approximation. Spectral normalization [30] constrains the Lipschitz constant of $\mathrm { Q } \mathrm { - }$ networks to stabilize value prediction. Creus Castanyer et al. [31] identified unstable gradient flow in deep Q-networks under non-stationary data and proposed architectural and optimization modifications. Related analyses [32], [33] showed that off-policy algorithms remain sensitive to estimation errors amplified through bootstrapped updates.

Another line of work focuses on the loss of network plasticity during prolonged reinforcement learning training. Nikishin et al. [3] identified the primacy bias, where agents overfit early interaction data and struggle to adapt under high replay ratios. Sokar et al. [4] reported the dormant neuron phenomenon, observing that a large fraction of neurons become inactive during training. Lyle et al. [5] further connected this loss of plasticity to rank collapse in learned representations, demonstrating that common architectural and optimization choices can accelerate representational degradation.

To mitigate plasticity loss, several intervention strategies have been proposed. Periodic parameter resets [3] aim to counteract primacy bias but may discard useful structure. Shrink-and-Perturb [34] restores gradient flow by shrinking parameters toward zero while injecting noise, and ReDo [4] selectively reinitializes dormant neurons based on activation statistics to recycle unused network capacity.

Together, these studies highlight that instability in deep reinforcement learning is closely related to optimization dynamics and the evolution of network representations during prolonged bootstrapped training.

## 2.5 World Models and Model-Based Reinforcement Learning

Recent years have seen rapid advances in model-based reinforcement learning, particularly in data-efficient settings such as Atari-100K. These methods improve sample efficiency by learning an explicit environment model and performing planning in latent space. IRIS [26] introduced a Transformerbased world model operating on discrete VQ-VAE latents, achieving strong sample efficiency without explicit lookahead search. DreamerV3 [35] scaled recurrent latent world models into a unified algorithm spanning over 150 tasks, emphasizing normalization and architectural stability to ensure cross-domain robustness. STORM [27] combined Transformer architectures with stochastic latent variables to enhance modeling flexibility, while DIAMOND [36] leveraged diffusion-based generation to better preserve visual details that discrete latent models may discard. DART [37] and Drama [25] further explored trajectory-aware and statespace modeling architectures, continuing the trend toward increasingly expressive and stable world models. These methods address sample efficiency through model learning, while our work focuses on the stability of value learning itself.

## 3 ANALYSIS OF INSTABILITY MECHANISMS

Although instability in deep Q-learning has long been recognized, existing explanations typically focus on individual phenomena such as maximization bias or representation learning effects. In practice, however, value learning forms a recursive feedback system in which multiple components interact through the Bellman update and environment interaction loop. Instability therefore often arises not from a single source, but from the coupling of several mechanisms that influence the dynamics of recursive value estimation.

In this section, we analyze instability in deep Q-learning from three complementary perspectives corresponding to different stages of the value learning process. First, at the operator level, recursive Bellman bootstrapping may introduce structural bias in target construction, leading to feedback amplification effects beyond classical maximization bias. Second, at the estimator level, the stochastic nature of value regression may perturb greedy action selection, causing regression noise to propagate through the control loop and alter the data distribution encountered during training. Third, at the parameter-dynamics level, neural network parameters may gradually evolve toward imbalanced configurations under aggressive data reuse, reducing the network’s ability to adapt to changing value targets.

These mechanisms interact through the recursive structure of value learning. Amplified bootstrap targets influence value regression, noisy estimation alters control decisions and experience collection, and evolving parameter configurations further shape the learning dynamics. Together, these feedback pathways create complex instability patterns that are difficult to explain from any single perspective. The following subsections examine these mechanisms in detail and provide empirical evidence illustrating their roles in deep Q-learning instability.

## 3.1 Operator-Level Instability: Unstable Recursive Bootstrapping

Operator-level instability arises from structural biases in the Bellman backup. Standard Q-learning constructs targets via

$$
r + \gamma \operatorname * { m a x } _ { a ^ { \prime } } Q _ { \bar { \theta } } ( s ^ { \prime } , a ^ { \prime } ) .\tag{1}
$$

Because the maximum is taken over noisy estimates,

$$
\mathbb { E } \Big [ \operatorname* { m a x } _ { a } \hat { Q } ( s , a ) \Big ] \ \geq \ \operatorname* { m a x } _ { a } \mathbb { E } \Big [ \hat { Q } ( s , a ) \Big ]\tag{2}
$$

![](images/83b6ae8c1918e29c308f0f24741f16b6fcff3d1146180fce3850821f853c8483.jpg)  
Fig. 1: Reward-triggered self-reinforcing bias in Atari Alien. The dashed blue curve (left axis) shows the recent 10-episode mean return during training, while the solid red curve (right axis) reports the minibatch statistic Count $( a = \arg \operatorname* { m a x } _ { a ^ { \prime } } Q ( s ^ { \prime } , a ^ { \hat { \prime } } ) )$ , the number of sampled transitions whose greedy bootstrap action at the next state coincides with the original action. The action space contains 18 actions and the minibatch size is 32. The consistently elevated counts indicate a strong coupling between a and the bootstrap action $a ^ { \prime * }$ , providing empirical evidence for the self-reinforcing trap discussed in this section.

leading to classical overestimation bias.

In deep reinforcement learning, an additional source of overestimation arises from function approximation and representation generalization. In discrete control, the actionvalue function is commonly parameterized as

$$
Q _ { \theta } ( s , a ) = f _ { \theta _ { 1 } } ( \phi _ { \theta _ { 2 } } ( s ) , a ) ,\tag{3}
$$

where $\phi _ { \theta _ { 2 } } ( \cdot )$ denotes a shared state encoder and $f _ { \theta _ { 1 } } ( \cdot , a )$ is an action-conditioned output head. Under this architecture, generalization primarily occurs across states for the same action, while value estimates for different actions are modeled by separate heads and are therefore not directly coupled by representation generalization.

As a result, for a fixed action $^ { a , }$ the corresponding value estimates satisfy $Q _ { \theta } ( s , a ) \ \approx \ Q _ { \theta } ( s ^ { \prime } , a )$ , whereas no such coupling is induced between $Q _ { \theta } ( s , a )$ and $Q _ { \theta } ( s ^ { \prime } , a ^ { \prime } )$ for $a ^ { \prime } \neq a .$ . This action-conditioned generalization improves sample efficiency, but it also creates an implicit pathway through which bootstrapped targets can propagate across nearby states for the same action.

A particularly problematic case arises for reward-bearing transitions. Consider a replayed transition $( s , a , r , s ^ { \prime } )$ with $r > 0 .$ . Due to action-conditioned generalization, the update of $Q ( s , a )$ driven by the positive reward also affects the estimate of $Q ( s ^ { \prime } , a )$ , since the two states induce similar representations and share the same action head. As a result, after a number of updates, the value $Q ( s ^ { \prime } , a )$ tends to become relatively large compared to other actions at state $s ^ { \prime } .$

When computing the bootstrapped target, the max operator selects the action $a ^ { \prime * } \in \arg \tilde { \operatorname* { m a x } } _ { a ^ { \prime } } \bar { Q ^ { } } ( s ^ { \prime } , a ^ { \prime } )$ . Because $Q ( s ^ { \prime } , a )$ has already been amplified through representation generalization induced by the positive reward $r > 0 ,$ , the maximization step is biased toward selecting the same action, i.e., $\smash { a ^ { \prime } { } ^ { * } = a }$ . This bias is not caused by the max operator alone, but the interaction between reward-driven value amplification and action-conditioned generalization. As illustrated in Fig. 1, we empirically examine this phenomenon in the Atari Alien environment by recording the number of transitions for which $a = \arg \operatorname* { m a x } _ { a ^ { \prime } } Q ( s ^ { \prime } , \breve { a ^ { \prime } } )$ . The results show a pronounced bias toward selecting the previous action, indicating a strong correlation between $a ^ { \prime * }$ and a.

Once $\boldsymbol { a } ^ { \prime } { } ^ { * } = \boldsymbol { a }$ is selected, the resulting target $r + \gamma Q ( s ^ { \prime } , a )$ further increases $Q ( s , a )$ . This creates a positive feedback loop in which the reward-induced increase of $Q ( s , a )$ propagates to $Q ( s ^ { \prime } , a )$ , and the maximization step repeatedly selects the same action. Importantly, this mechanism is specific to transitions with $r > 0$ . For transitions with zero reward, no persistent upward shift is introduced, and the max operator does not exhibit the same systematic preference for reselecting the original action.

Notably, the magnitude of this self-reinforcing amplification admits an upper bound in an idealized repeated-replay scenario. Consider repeatedly replaying the same rewardbearing transition $( s , a , r , s ^ { \prime } )$ with $r > 0 ,$ , and suppose the maximization step persistently reselects the same action so that $\boldsymbol { a } ^ { \prime } { } ^ { * } = \boldsymbol { a }$ holds throughout training, as indicated by our empirical observation. In this case, the bootstrapped update reduces to a one-dimensional fixed-point iteration

$$
Q _ { k + 1 } ( s , a ) \ : = \ : r + \gamma Q _ { k } ( s ^ { \prime } , a ) .\tag{4}
$$

Under the self-reinforcing coupling described above, $Q _ { k } ( s ^ { \prime } , a )$ is repeatedly driven upward in tandem with $Q _ { k } ( s , a )$ , and the iteration effectively behaves as

$$
Q _ { k + 1 } ( s , a ) \approx r + \gamma Q _ { k } ( s , a ) ,\tag{5}
$$

whose unique fixed point is

$$
Q ( s , a ) \ \to \ { \frac { r } { 1 - \gamma } } .\tag{6}
$$

Therefore, even though the feedback loop can systematically inflate Q-values, its amplification under repeated replay of a single transition is bounded and converges to a finite limit determined by the reward scale and discount factor.

## 3.2 Estimator-Level Instability: Interaction Drift Induced by Regression Noise

Even when the Bellman operator is unbiased and target construction is structurally stable, deep Q-learning remains sensitive to estimation noise during action selection. Fundamentally, deep Q-learning is a stochastic regression problem: the value function is optimized to minimize a squarederror objective with bootstrapped targets. Although such regression may be unbiased in expectation, control decisions rely on the maximization of noisy value estimates.

Suppose the learned action-value function satisfies

$$
Q _ { \theta } ( s , a ) = Q ^ { * } ( s , a ) + \varepsilon _ { a } ,\tag{7}
$$

where $\varepsilon _ { a }$ is a zero-mean estimation error with $\mathbb { E } [ \varepsilon _ { a } ] = 0$ . The greedy policy selects

$$
a _ { \theta } ^ { * } ( s ) = \arg \operatorname* { m a x } _ { a } Q _ { \theta } ( s , a ) .\tag{8}
$$

Let $\Delta ( s )$ denote the action gap between the optimal action $a ^ { * }$ and the second-best alternative:

$$
\Delta ( s ) = Q ^ { * } ( s , a ^ { * } ) - \operatorname* { m a x } _ { a \neq a ^ { * } } Q ^ { * } ( s , a ) .\tag{9}
$$

The probability of selecting a suboptimal action is therefore

$$
\operatorname* { P r } ( a _ { \theta } ^ { * } ( s ) \neq a ^ { * } ) = \operatorname* { P r } ( \varepsilon _ { a ^ { \prime } } - \varepsilon _ { a ^ { * } } > \Delta ( s ) ) ,\tag{10}
$$

for some $a ^ { \prime } \neq a ^ { * }$

Thus, even unbiased estimation noise can induce unstable action selection when the action gap is small. While this does not introduce systematic value bias, it directly affects the quality of decisions made during interaction with the environment. Because the behavior policy determines the distribution of collected data, noisy action selection may lead the agent to generate lower-return trajectories. Learning slows not due to inefficient reuse of samples, but because the interaction policy itself produces less informative experience. In this sense, estimator-level instability degrades control reliability and indirectly limits performance improvement.

These observations motivate the need for variancereduction techniques in value regression, so that action selection remains stable under stochastic estimation noise.

## 3.3 Parameter-Dynamics Instability: Layer-wise Parameter Imbalance and Plasticity Loss

Beyond operator-level bias and estimator-level decision sensitivity, instability may also arise from the evolution of parameters under non-stationary sampling.

In reinforcement learning, the data distribution is determined by the behavior policy, which continuously changes throughout training. As a result, the representation learned by the network must remain adaptable to shifting state-action distributions. However, repeated optimization on a limited or slowly evolving data distribution can progressively bias parameter updates toward frequently observed patterns.

This effect exists in general, but becomes increasingly pronounced as the replay ratio grows. When past experiences are reused many times before sufficient new interactions are collected, the influence of the current data distribution is amplified. Over time, this can drive certain weights to dominate their layers, while the remaining weights receive comparatively weaker updates. The resulting parameter distribution becomes increasingly skewed.

To quantify this imbalance, we introduce a diagnostic statistic termed the spike ratio. For a given layer ℓ with parameters $\theta _ { \ell } ,$ we define

$$
\mathrm { S p i k e R a t i o } _ { \ell } = \frac { \| \theta _ { \ell } \| _ { \infty } } { \mathrm { Q u a n t i l e } _ { 0 . 9 9 } ( | \theta _ { \ell } | ) } .\tag{11}
$$

It measures the dominance of extreme parameter values relative to the bulk of the distribution. An increasing spike ratio indicates that a small subset of weights grows disproportionately large, revealing an imbalanced representation.

Fig. 9 illustrates the evolution of SpikeRatio under different replay ratios (rr = 1, 2, 4). While spike growth is observable even at lower replay ratios, it becomes more pronounced as the replay ratio increases, particularly in postencoder MLP layers. This pattern indicates that replay amplification systematically accelerates parameter imbalance.

Parameter-dynamics instability does not necessarily lead to numerical divergence of value estimates. Instead, it reflects a gradual loss of representational plasticity. As the data distribution shifts with policy improvement, a skewed parameter configuration may adapt more slowly, thereby constraining long-term performance improvement. These observations underscore the importance of preserving balanced parameter evolution throughout training.

![](images/2fe5eaf2a1ef51bbb233229bde1bfd8954fb413e82cabd7e74395e794f2f7a46.jpg)  
Fig. 2: Layer-wise spike ratio evolution during training under different replay ratios. Each row corresponds to a network layer, and color intensity indicates spike magnitude. Higher replay ratios lead to more pronounced spike growth, particularly in post-encoder MLP layers. Further details are provided in the supplementary materials.

## 4 STABILIZATION PRINCIPLES FOR VALUE LEARN-ING

The analysis in Section 3 reveals that instability in deep Qlearning arises from three interacting mechanisms: operatorlevel bias in recursive bootstrapping, estimator-level sensitivity of greedy action selection to regression noise, and parameter-dynamics imbalance under non-stationary sampling. These mechanisms interact through the recursive structure of value learning. Instability may emerge when bootstrap targets amplify value estimates, when noisy regression alters control decisions, or when network parameters gradually lose plasticity during prolonged training. Stabilizing value learning therefore requires coordinated regulation of these components rather than isolated heuristics.

Based on the preceding analysis, we derive three stabilization principles for deep Q-learning:

• Controlled bootstrapping. Recursive Bellman updates must be regulated to prevent feedback amplification during target construction.

• Variance-aware value estimation. Reducing estimation variance improves the reliability of greedy action selection and stabilizes interaction dynamics.

• Parameter-dynamics regulation. Maintaining balanced parameter evolution preserves network plasticity under prolonged replay.

In the following, we describe practical mechanisms that instantiate these principles. Together they form a controlled optimization procedure for stabilizing value learning.

## 4.1 Controlled Bootstrapping

Section 3.1 shows that operator-level instability arises not only from classical maximization bias, but also from a previously overlooked reward-triggered self-reinforcing pathway in recursive bootstrapping.

## 4.1.1 Cross-Model Decoupling for Action Selection and Evaluation

The overestimation phenomenon in Eq. (2) arises from using the same noisy estimator for both action selection and value

evaluation. Double DQN alleviates this issue by decoupling these roles using online and target networks.

We generalize this idea to an ensemble setting. Let

$$
Q = \{ \hat { Q } _ { 1 } , . . . , \hat { Q } _ { m } \}\tag{12}
$$

denote an ensemble of value functions. For each model ${ \hat { Q } } _ { i } ,$ we select the bootstrap action using another model $\hat { Q } _ { j } \left( j \neq i \right)$ while the value is evaluated using $\hat { Q } _ { i }$ :

$$
y _ { i } = r + \gamma \hat { Q } _ { i } \left( s ^ { \prime } , \arg \operatorname* { m a x } _ { a ^ { \prime } } \hat { Q } _ { j } ( s ^ { \prime } , a ^ { \prime } ) \right) .\tag{13}
$$

This structurally eliminates selection-evaluation coupling. When the estimation noises of $\hat { Q } _ { i }$ and $\hat { Q } _ { j }$ are weakly correlated as encouraged by independent initialization and separate bootstrap targets, the selection noise and evaluation noise are effectively decoupled, which substantially reduces the upward bias induced by the max operator.

## 4.1.2 Action-Decoupling for Reward-Bearing Transitions

The instability analysis in Section 3.1 further reveals a feedback pathway termed the self-reinforcing trap. When a reward-bearing transition $( s , a , r , s ^ { \prime } )$ is repeatedly replayed, representation generalization may increase both $Q ( s , a )$ and $\hat { Q ( s ^ { \prime } , a ) }$ simultaneously, making action a likely to be selected again during the bootstrap step.

To break this feedback loop, we impose a constraint on reward-bearing transitions:

$$
a _ { i } ^ { \prime } \neq a \quad { \mathrm { i f ~ } } r > 0 .\tag{14}
$$

This constraint prevents the bootstrap target from repeatedly selecting the same action that produced the reward, thereby disrupting the amplification pathway identified in the instability analysis.

## 4.1.3 Bounded Bellman Updates

Even with decoupled bootstrapping, recursive updates may gradually increase the numerical scale of value estimates. To prevent uncontrolled growth, we impose an upper bound on target values derived from the discounted return along greedy trajectories.

Given a greedy trajectory $\tau ,$ we define a trajectory-level bound

$$
B ( \tau ) = \operatorname* { m a x } _ { t } G _ { t } ,\tag{15}
$$

where $G _ { t }$ denotes the discounted return. This bound provides a natural constraint on the numerical scale of value estimates during training. Further implementation details are provided in the supplementary material.

## 4.2 Variance-Aware Value Estimation

Section 3.2 shows that even unbiased regression noise may alter greedy action selection when action gaps are small. This effect may destabilize interaction dynamics.

To reduce estimation variance, we adopt ensemble quantile regression. Following QR-DQN, we model the return distribution using K quantile estimates

$$
Z _ { \theta } ( s , a ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \delta _ { z _ { \theta } ^ { ( i ) } ( s , a ) } .\tag{16}
$$

The scalar value estimate is

$$
Q _ { \theta } ( s , a ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } z _ { \theta } ^ { ( i ) } ( s , a ) .\tag{17}
$$

Given bootstrap action $a ^ { \prime * } ,$ the target quantiles are

$$
y ^ { ( j ) } = r + \gamma z _ { \bar { \theta } } ^ { ( j ) } ( s ^ { \prime } , a ^ { \prime * } ) ,\tag{18}
$$

and parameters are optimized via the quantile regression loss

$$
\mathcal { L } _ { \mathrm { Q R } } ( \theta ) = \frac { 1 } { K ^ { 2 } } \sum _ { i = 1 } ^ { K } \sum _ { j = 1 } ^ { K } \rho _ { \tau _ { i } } ^ { \kappa } \left( y ^ { ( j ) } - z _ { \theta } ^ { ( i ) } ( s , a ) \right) .\tag{19}
$$

To further reduce variance, we maintain an ensemble of quantile networks

$$
\{ Z _ { \theta _ { 1 } } , \ldots , Z _ { \theta _ { m } } \} .\tag{20}
$$

The ensemble-averaged value used for action selection is

$$
\bar { Q } ( s , a ) = \frac { 1 } { m } \sum _ { \ell = 1 } ^ { m } \frac { 1 } { K } \sum _ { i = 1 } ^ { K } z _ { \theta _ { \ell } } ^ { ( i ) } ( s , a ) .\tag{21}
$$

Ensemble aggregation reduces both intra-model and intermodel estimation variance, improving the reliability of greedy action selection.

## 4.3 Parameter-Dynamics Regulation

Section 3.3 reveals that high replay ratios may gradually distort parameter distributions, leading to reduced network plasticity. To monitor this phenomenon, we introduce the spike ratio, defined as Eq. (11). This statistic measures the dominance of extreme parameter values relative to the bulk of the distribution. When the spike ratio of a layer exceeds a predefined threshold, corrective intervention is triggered by resetting the affected parameters. This mechanism restores distributional balance and helps preserve long-term adaptability of the value network.

## 5 ALGORITHM INSTANTIATION

The mechanisms described above provide practical implementations of the stabilization principles derived from the instability analysis. Together they form a controlled optimization procedure for value learning.

Algorithm 1 summarizes the overall training process. At each interaction step (Lines 1–8), the agent observes the current state $s _ { t }$ (Line 2) and selects an action using ϵ-greedy exploration (Lines 3–7). With probability ϵ, a random action is sampled uniformly (Line 4). Otherwise, the agent acts greedily w.r.t. the ensemble-averaged value $\bar { Q } ( s , a )$ defined in Eq. (21) (Line 6). The resulting transition $\left( { { s _ { t } } , { a _ { t } } , { r _ { t } } , { s _ { t + 1 } } } \right)$ is then stored in the replay buffer D (Line 8). For learning, we sample a prioritized minibatch $\{ ( s ^ { ( b ) } , a ^ { ( b ) } , r ^ { ( b ) } , s ^ { \prime ( b ) } ) \} _ { b = 1 } ^ { B }$ from $\bar { \mathcal { D } }$ (Line 9). Next, we determine bootstrap actions for each ensemble member using its own target network (Lines 10–17). Specifically, for each member $i ,$ we first form the scalar action values from its quantile outputs (Line 11, Eq. (17)) and then apply the reward-bearing decoupling rule (Lines 12–15): for any sampled transition b with $\stackrel { \cdot } { r } { } ^ { ( b ) } > 0$ , we mask the self-reinforcing candidate by setting $Q _ { \theta _ { i } ^ { - } } ( s ^ { \prime ( b ) } , a ^ { ( b ) } ) \ = \ - \infty ,$ , which enforces $a _ { i } ^ { * } { \left( s ^ { \prime ( b ) } \right) } ^ { \prime } \neq a ^ { ( b ) }$ during maximization. We then compute the greedy bootstrap action $a _ { i } ^ { * } \big ( s ^ { \prime ( b ) } \big ) = \arg \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } } Q _ { \theta _ { \ast } ^ { - } } ^ { \ast } \big ( s ^ { \prime ( b ) } , \tilde { a } ^ { \prime } \big )$ (Line 16). To decouple action selection from value evaluation across models, we sample a random permutation π over ensemble indices (Line 18). During the subsequent update (Lines 19–

Algorithm 1: Bootstrapping Control and Ensemble   
Quantile Regression   
Input: Replay buffer $\mathcal { D } ;$ ensemble size m; number of   
quantiles $K ;$ Online networks $\{ Q _ { i } ( \cdot ; \theta _ { i } ) \} _ { i = 1 } ^ { m }$   
and target networks $\{ Q _ { i } ( \cdot ; \theta _ { i } ^ { - } ) \} _ { i = 1 } ^ { m } .$ ; Quantile   
fractions $\begin{array} { r } { \{ \tau _ { j } = \frac { j - 0 . 5 } { K } \} _ { j = 1 } ^ { K } ; } \end{array}$ exploration rate ϵ   
Output: Updated ensemble parameters $\{ \theta _ { i } \} _ { i = 1 } ^ { m }$   
1 for t = 1 to $T$ do   
$/ /$ Environment interaction   
2 Observe state $s _ { t }$   
3 if rand $1 ( ) < \epsilon$ then   
4 Select $a _ { t } \sim$ Uniform(A)   
5 else   
6 Select greedy action according to   
ensemble-averaged value(Eq. (21)):   
$a _ { t } = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \bar { Q } ( s _ { t } , a )$   
7 end   
8 Execute $a _ { t } ,$ observe $( r _ { t } , s _ { t + 1 } )$ , and store   
$\left( { { s _ { t } } , { a _ { t } } , { r _ { t } } , { s _ { t + 1 } } } \right)$ in D   
// Prioritized experience replay   
9 Sample a mini-batch $\{ ( s ^ { ( b ) } , a ^ { ( b ) } , r ^ { ( b ) } , \bar { s ^ { \prime ( b ) } } ) \} _ { b = 1 } ^ { B }$   
from D   
$/ /$ Bootstrap action selection   
10 for $i = 1$ to m do   
11 Aggregate distributional value estimation   
according to Eq. (17)   
12 Decoupling reward-bearing transitions:   
13 if $r ^ { ( b ) } > 0$ then   
14 $Q _ { \theta _ { i } ^ { - } } ( s ^ { \prime ( b ) } , a ^ { ( b ) } ) = - \infty$   
15 end   
16 Select bootstrap action:   
$a _ { i } ^ { * } = \arg \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } } Q _ { \theta _ { i } ^ { - } } ( s ^ { \prime } , a ^ { \prime } )$   
17 end   
18 Sample a random permutation π over $\{ 1 , \ldots , m \}$   
19 for $\dot { i } = 1$ to m do   
// Cross-model bootstrapping   
20 for $j ^ { \prime } = 1$ to K do   
21 Compute target quantiles:   
$y _ { i } ^ { ( b , \hat { j } ^ { \prime } ) } = r ^ { ( b ) } + \stackrel { \sim } { \gamma } z _ { i , j ^ { \prime } } \left( s ^ { \prime ( b ) } , a _ { \pi ( i ) } ^ { * } ; \theta _ { i } ^ { - } \right)$   
22 end   
23 Compute quantile regression loss and update   
parameters according to Eq. (19)   
24 end   
25 Periodically update target networks $\theta _ { i } ^ { - }  \theta _ { i }$   
26 Periodically monitor layer-wise spike ratios   
according to Eq. (11)   
27 if Layer $\ell ^ { \prime } \mathrm { s }$ spike ratio is greater than threshold   
then   
28 Reset layer $\ell ^ { \prime } { \mathbf { s } }$ parameters   
29 end   
30 end

24), member i evaluates the bootstrap action selected by another member $\pi ( i )$ . Concretely, for each quantile index $j ^ { \prime } \in \{ 1 , \ldots , K \}$ , we construct the target quantile $y _ { i } ^ { ( b , j ^ { \prime } ) } = r ^ { ( b ) } + \gamma z _ { i , j ^ { \prime } } \bar { ( s ^ { \prime ( b ) } , a _ { \pi ( i ) } ^ { * } ( s ^ { \prime ( b ) } ) ; \theta _ { i } ^ { - } ) }$ (Line 21), i.e., the action comes from the permuted selector while the quantile evaluation uses member i’s target network. Member i is then updated by minimizing the quantile regression objective in Eq. (19) (Line 23). Finally, target networks are periodically synchronized (Line 25), and we monitor layer-wise spike ratios (Line 26, Eq. (11)) to detect over-adaptation. If a layer’s spike ratio exceeds a predefined threshold, we reset that layer’s parameters to restore training stability (Lines 27–29).

## 6 EXPERIMENTS

Experiments evaluate the proposed analysis and the derived stabilization principles. First, we evaluate whether the algorithm instantiated from the proposed principles achieves competitive performance on standard reinforcement learning benchmarks. Then, we analyze how the proposed stabilization principles influence learning behavior through controlled ablations.

We begin with Atari-100K [38](Section 6.1), a widely adopted benchmark for data-efficient reinforcement learning. Due to its limited interaction budget and high replay utilization, Atari-100K provides a challenging setting where bootstrapping instability and value overestimation are particularly pronounced. This benchmark allows us to directly assess whether the proposed stabilization mechanisms improve learning reliability under constrained data conditions.

To further evaluate robustness in discrete control, we consider Procgen [39](Section 6.2), which features procedurally generated environments with diverse visual appearances and dynamics. Unlike fixed-layout benchmarks, Procgen tests representation robustness and generalization across unseen levels, enabling us to examine whether improved value stability translates into stronger cross-distribution performance.

In addition to benchmark comparisons, we perform a series of diagnostic analyses to investigate the mechanisms identified in Section 3 and the stabilization principles introduced in Section 4.

## 6.1 Atari-100K: Learning Stability under Data-Efficient Regimes

We evaluate our method on the Atari-100K benchmark, which consists of 26 Atari 2600 games under a strict interaction budget of 100k environment steps. This corresponds to 400k frames with a frame skip of 4. Under this limited data regime, agents must repeatedly reuse collected transitions, resulting in high replay ratios.

Our implementation follows the standard Atari preprocessing pipeline [29]. Observations are resized and stacked following common practice. We adopt a convolutional ResNet-style backbone without normalization layers, consistent with our method design. Results are averaged over multiple random seeds. Detailed hyperparameter configurations are provided in supplementary materials.

We compare against a diverse set of representative methods evaluated under the Atari-100K protocol. Random and human scores are reported as reference points. DER [28] incorporates auxiliary supervised objectives into value learning to improve sample efficiency. DrQ [22] enhances data efficiency via image augmentation applied directly to the value function. IRIS [26], STORM [27], DreamerV3 [35], DIAMOND [36], DART [37], and Drama [25] are model-based or hybrid approaches that leverage latent dynamics modeling for improved planning and sample efficiency. REM [10] reduces overestimation through ensemble-based value aggregation. SGF [24] focuses on stabilizing gradient propagation during representation learning. BBF [23] combines aggressive replay, regularization, and parameter resetting to achieve strong performance under limited data. These baselines collectively cover value-based, model-based, and hybrid paradigms for data-efficient reinforcement learning.

TABLE 1: Aggregate performance on Atari-100K across 26 games. All metrics are computed using human-normalized scores (HNS). Mean, Median, and Interquartile Mean (IQM) are reported following standard evaluation practice. #Human denotes the number of games surpassing human-level performance, and #Best indicates the number of per-game best results. Higher is better for all metrics.
<table><tr><td>Algorithm</td><td>#Human</td><td>#Best</td><td>Mean</td><td>Median</td><td>IQM</td></tr><tr><td>DER</td><td>2</td><td>0</td><td>0.350</td><td>0.189</td><td>0.183</td></tr><tr><td>DrQ</td><td>3</td><td>0</td><td>0.465</td><td>0.313</td><td>0.280</td></tr><tr><td>IRIS</td><td>9</td><td>0</td><td>1.046</td><td>0.289</td><td>0.501</td></tr><tr><td>REM</td><td>12</td><td>1</td><td>1.222</td><td>0.280</td><td>0.673</td></tr><tr><td>STORM</td><td>10</td><td>5</td><td>1.266</td><td>0.580</td><td>0.636</td></tr><tr><td>DreamerV3</td><td>9</td><td>1</td><td>1.120</td><td>0.466</td><td>0.490</td></tr><tr><td>DIAMOND</td><td>11</td><td>0</td><td>1.459</td><td>0.373</td><td>0.641</td></tr><tr><td>DART</td><td>9</td><td>0</td><td>1.022</td><td>0.790</td><td>0.575</td></tr><tr><td>SGF</td><td>6</td><td>0</td><td>0.884</td><td>0.152</td><td>0.287</td></tr><tr><td>Drama</td><td>8</td><td>1</td><td>1.049</td><td>0.270</td><td>0.367</td></tr><tr><td>BBF</td><td>12</td><td>8</td><td>2.247</td><td>0.917</td><td>1.045</td></tr><tr><td>Ours</td><td>14</td><td>10</td><td>1.799</td><td>1.045</td><td>1.070</td></tr></table>

Performance is measured using the Human-Normalized Score (HNS). We report aggregate statistics across 26 games, including the mean HNS, median HNS, and Interquartile Mean (IQM), which provides a robust estimate that is less sensitive to extreme outliers. We additionally report the number of games surpassing human-level performance and the number of per-game best results.

Aggregate results are presented in Table 1. Our method achieves the highest IQM and Median HNS among all compared approaches. We also obtain the largest number of games exceeding human-level performance and the highest number of per-game best results. These results indicate strong overall performance under the stringent 100k interaction constraint. For completeness and transparency, full perenvironment scores are reported in supplementary materials.

## 6.2 Procgen: Representation Robustness and Generalization

We further evaluate our method on the Procgen benchmark [39], which emphasizes representation robustness and cross-level generalization. Unlike Atari-100K, where training and evaluation share identical game layouts, Procgen generates diverse levels procedurally and allows strict separation between training and test environments.

TABLE 2: Procgen (easy difficulty, 200 training levels). Scores are normalized using min-max normalization as defined in the original Procgen paper.
<table><tr><td></td><td>Test Mean</td><td>Test Median</td><td>Test IQM</td></tr><tr><td>PPO (Impala CNN)</td><td>0.33</td><td>0.38</td><td>0.27</td></tr><tr><td>Ours</td><td>0.40</td><td>0.40</td><td>0.39</td></tr></table>

We adopt the generalization protocol, where the agent is trained on a fixed set of 200 procedurally generated levels and evaluated on an unseen set of test levels. This setting explicitly measures cross-level generalization ability rather than memorization of specific layouts. All agents interact with environments for a total of 25M environment steps. During evaluation, exploration is disabled and performance is averaged over multiple test levels. Detailed hyperparameter configurations are provided in supplementary materials.

We compare against the official PPO implementation reported in the Procgen paper [39]. PPO is widely adopted as a strong on-policy baseline for Procgen and has been carefully tuned for this benchmark. Using PPO as the primary baseline allows us to assess whether improved value stability translates into stronger generalization under procedural diversity.

For each game, we report the average test return over unseen levels. To summarize performance across environments, we additionally report: the mean score, the median score, and the interquartile mean (IQM), following recent evaluation practice for robust aggregate comparison. Higher values indicate better generalization performance.

Table 2 presents aggregate statistics. Our method consistently outperforms the baseline on the majority of environments, achieving higher average scores. The improvement is particularly pronounced in environments with high visual and structural variability, suggesting that stabilizing recursive value learning benefits representation robustness under procedural diversity. Detailed results and additional visualization are provided in the supplementary materials.

Per-environment performance reveals a structured pattern rather than uniform improvement.

Substantial gains are observed in environments such as BigFish, Dodgeball, and StarPilot, which provide relatively dense reward signals and require stable action-value propagation. In these settings, recursive bootstrapping occurs frequently, and over-amplified value estimates can easily destabilize training. By controlling bootstrap bias and reducing estimation variance, our method produces smoother value propagation and more reliable action selection, leading to significant performance improvements.

In contrast, environments such as Chaser, Heist, and Maze remain challenging. These tasks feature sparse rewards and long-horizon credit assignment, where performance is heavily influenced by exploration efficiency and global planning capability. Since our method primarily targets value instability rather than exploration enhancement, its advantages are less pronounced in such environments.

## 6.3 Mechanism Analysis and Sensitivity Study

Beyond benchmark results, we conduct additional experiments to analyze the behavior of the proposed method.

![](images/31cca45d1680881a537c9a100abed60b24bf5cd157991f00601c7e6e32a77c8b.jpg)  
Fig. 3: Training performance under different ensemble sizes $N _ { e }$ on four Atari-100K environments. Curves show the recent 10-episode mean return; shaded regions denote 95% CIs.

![](images/047847fee77d17c479d1b0000d9c4ed9dab75e85c361d225bb25ad5a7c0126aa.jpg)  
Fig. 4: Normalized final performance as a function of the ensemble size. Results are normalized within each environment.

Specifically, we examine the effects of ensemble size, the Action-Decoupling Constraint, parameter-level spike dynamics, and replay ratio. These studies aim to provide further insight into the factors influencing training stability and performance.

## 6.3.1 Effect ofEnsemble Size

We investigate the influence of the number of ensemble networks $N _ { e }$ on training performance. Experiments are conducted on four representative Atari environments: Alien, Amidar, BankHeist, and Breakout. We evaluate $N _ { e } \in \{ 2 , 4 , 8 , 1 6 \}$ while keeping all other training settings fixed. Each configu ration is repeated with multiple random seeds.

Fig. 3 shows the evolution of the recent 10-episode mean return during training. Increasing the ensemble size generally improves performance and training stability across most environments. In particular, larger ensembles lead to faster performance growth and higher final returns in Alien, Amidar, and BankHeist. Breakout exhibits a less monotonic trend, indicating that the effect of ensemble size may vary depending on environment dynamics.

![](images/629450131bab947580b4d3636b96b1330b3d6103b721a4b1521acd8d7d989768.jpg)  
Fig. 5: Comparison of training dynamics with and without the Action-Decoupling Constraint. For each environment, the left axis shows the recent 10-episode mean return, and the right axis shows the average Q-value. Solid lines correspond to the constrained version, while dashed lines denote the unconstrained variant.

To further summarize the overall trend, Fig. 4 reports normalized final performance as a function of $N _ { e }$ . Performance improves consistently as the ensemble size increases from 2 to 16 in most cases, suggesting that larger ensembles contribute positively to learning stability.

## 6.3.2 Effect of the Action-Decoupling Constraint

We evaluate the impact of the Action-Decoupling Constraint applied to reward-bearing transitions by comparing the full method with a variant in which the constraint is removed during target computation. All other training settings are kept identical. Experiments are conducted on Alien, Amidar, BankHeist, and Breakout.

Fig. 5 presents the evolution of the recent 10-episode mean return (left axis) and the average Q-value (right axis). On Alien, the constrained version yields a clear and consistent performance improvement over the unconstrained variant. In contrast, on Amidar, BankHeist, and Breakout, the performance differences are less pronounced, and in some cases the unconstrained version performs comparably or slightly better.

A key distinction lies in reward density. Alien provides relatively dense and frequent rewards, where excessive selfreinforcing updates may amplify overestimated values and destabilize value learning. The Action-Decoupling Constraint mitigates this amplification, leading to more stable performance. In environments with sparser rewards, moderate self-reinforcing effects can help strengthen weak learning signals, which may explain the smaller performance gap in those tasks. In such cases, weaker signal amplification can also be compensated by longer training or increased data reuse. Further analysis under higher replay ratios is provided later in this section.

## 6.3.3 Parameter-Level Spike Dynamics and Reset Mechanism

We first analyze parameter-level spike dynamics during training. For each network layer, we monitor the spike ratio (Eq. (11)) over training progress, defined as the ratio between the maximum absolute parameter value and a high-percentile statistic within the same layer. This metric provides a proxy for detecting abnormal parameter amplification.

![](images/2eb03141c703ac8c1f502e1bc527a54ace83e49fd522961e1be786fa1787a0d7.jpg)  
Fig. 6: Effect of parameter resetting on training performance. Solid lines denote the version with parameter reset, while dashed lines indicate the variant without resetting. The right axis shows the frequency of reset events during training.

Fig. 9 visualizes the spike ratio evolution under different replay ratios $( \mathrm { r r } = 1 , 2 , \bar { 4 } )$ . Two consistent patterns emerge. First, spike ratios are significantly higher in the MLP layers following the encoder, particularly in the projection head. In contrast, convolutional encoder layers remain comparatively stable. Second, increasing the replay ratio leads to systematically higher spike ratios across layers. This suggests that aggressive data reuse amplifies parameter imbalance and increases the risk of unstable value growth.

We further investigate the role of the parameter reset mechanism. Specifically, we compare training performance with and without parameter resetting, and record the frequency of reset events during training. As shown in Fig. 6, parameter resets occur more frequently in early training stages and under environments where value growth is more volatile. When parameter resetting is enabled, its impact varies across environments. Among the four evaluated tasks, a clear improvement is observed on Alien, while the effect on the other environments is comparatively modest. This suggests a trade-off: although resetting can restore network plasticity, it may also introduce irreversible performance loss by disrupting useful parameter structures that have already formed during training. As learning progresses, the model may gradually specialize toward informative features, and aggressively preserving plasticity can therefore interfere with stable decision making. Figure 2 further shows that, although plasticity tends to decrease during training, it can partially recover as new interaction data are incorporated, rather than collapsing permanently. Empirically, higher replay ratios require stricter and more frequent parameter resetting to maintain stable training dynamics.

## 6.3.4 Sensitivity to Replay Ratio

We further evaluate the sensitivity of the proposed method to replay ratio (rr), which controls the degree of data reuse during training. Experiments are conducted on Alien, Amidar, BankHeist, and Breakout with rr $\in \{ 1 , 2 , 4 , 8 \}$ , while keeping all other hyperparameters fixed.

![](images/09a0eaa1311cb6badb2b8e81c160d1a530597e79c39f888cdc3fa5b019c62137.jpg)  
Fig. 7: Training performance under different replay ratios on four Atari-100K environments. Curves show the recent 10-episode mean return; shaded regions denote 95% CIs.

![](images/b9f19a316981d4be6bb493d2e62e707fbed411b7e52f5344df9ee3ba127b5942.jpg)  
Fig. 8: Normalized final performance as a function of replay ratio. Results are normalized within each environment.

Fig. 7 presents training curves under different replay ratios. The impact of data reuse varies across environments. In Alien and Amidar, moderate replay ratios $( \mathrm { e . g . , r r = 4 } )$ lead to improved performance compared to lower reuse settings. However, excessively large replay ratios (e.g., rr = 8) may cause performance degradation in some environments. In contrast, BankHeist benefits consistently from higher replay ratios, suggesting that additional data reuse can help amplify weak learning signals in sparser reward settings. Breakout exhibits a different trend, where performance peaks at lower replay ratios and decreases as data reuse becomes aggressive.

Fig. 8 summarizes the normalized final performance as a function of replay ratio. The results reveal a non-monotonic relationship between replay ratio and final performance. Across multiple environments, performance improves as rr increases from low to moderate values, but degrades when rr becomes excessively large. This suggests that while additional data reuse enhances sample efficiency, there exists a task-dependent threshold beyond which further reuse no longer benefits learning.

A possible explanation relates to the interaction between aggressive data reuse and parameter spike dynamics. As observed in the previous section, higher replay ratios systematically amplify spike ratios, especially in post-encoder layers. Since spike monitoring and parameter resetting are performed periodically to avoid excessive computational overhead, very large replay ratios may lead to rapid overamplification within a single monitoring interval. In such cases, the network may temporarily overfit recent data before the next reset step is triggered, slowing down effective learning progress. Consequently, excessively high replay ratios can reduce training efficiency.

These findings indicate that replay ratio must be carefully balanced with stability mechanisms, and that optimal data reuse is inherently environment-dependent.

## 7 CONCLUSION AND FUTURE WORK

This work presents a unified perspective on instability in deep Q-learning by examining the dynamics of recursive value learning. We show that instability arises from the interaction of three mechanisms: operator-level bias in Bellman bootstrapping, estimator-level sensitivity of greedy decisions to regression noise, and parameter-dynamics imbalance under aggressive data reuse.

Our analysis reveals two characteristic behaviors that have received limited attention in prior studies. First, we identify a reward-triggered self-reinforcing trap in recursive bootstrapping, where value amplification interacts with representation generalization to repeatedly reinforce certain actions in bootstrap targets. Second, we uncover parameter spike dynamics associated with the gradual loss of network plasticity during training, and introduce the spike ratio as a practical diagnostic indicator for monitoring this effect.

Based on these insights, we derive a set of stabilization principles that regulate bootstrap target construction, reduce decision variance in value estimation, and maintain balanced parameter evolution during training. We instantiate these principles in a practical learning algorithm integrating controlled bootstrapping, ensemble quantile estimation, and spike-based parameter regulation. Experiments on Atari-100K and Procgen demonstrate that the resulting method achieves competitive performance while improving training stability.

Beyond empirical results, our findings highlight the importance of analyzing reinforcement learning through training dynamics. The interaction between bootstrapping, stochastic estimation, and parameter evolution suggests that stability in value learning is fundamentally a systemslevel property rather than the result of isolated algorithmic components. Future work may explore adaptive stabilization strategies, automated coordination of hyperparameters such as replay ratio and ensemble size, and extensions of the proposed analysis to actor-critic methods and continuouscontrol settings.

## REFERENCES

[1] H. Hasselt, “Double q-learning,” Advances in neural information processing systems, vol. 23, 2010.

[2] H. Van Hasselt, A. Guez, and D. Silver, “Deep reinforcement learning with double q-learning,” in Proceedings of the AAAI conference on artificial intelligence, vol. 30, 2016.

[3] E. Nikishin, M. Schwarzer, P. D’Oro, P.-L. Bacon, and A. Courville, “The primacy bias in deep reinforcement learning,” in International conference on machine learning. PMLR, 2022, pp. 16 828–16 847.

[4] G. Sokar, R. Agarwal, P. S. Castro, and U. Evci, “The dormant neuron phenomenon in deep reinforcement learning,” in International Conference on Machine Learning. PMLR, 2023, pp. 32 145–32 168.

[5] C. Lyle, M. Rowland, G. Ostrovski, and W. Dabney, “Understanding plasticity in neural networks,” in International Conference on Machine Learning. PMLR, 2023, pp. 23 190–23 211.

[6] S. Thrun and A. Schwartz, “Issues in using function approximation for reinforcement learning,” in Proceedings ofthe Fourth Connectionist Models Summer School. Lawrence Erlbaum Associates, 1993, pp. 255–263.

[7] O. Anschel, N. Baram, and N. Shimkin, “Averaged-dqn: Variance reduction and stabilization for deep reinforcement learning,” in International conference on machine learning. PMLR, 2017, pp. 176– 185.

[8] Q. Lan, Y. Pan, A. Fyshe, and M. White, “Maxmin q-learning: Controlling the estimation bias of q-learning,” arXiv preprint arXiv:2002.06487, 2020.

[9] S. Fujimoto, H. Hoof, and D. Meger, “Addressing function approximation error in actor-critic methods,” in International conference on machine learning. PMLR, 2018, pp. 1587–1596.

[10] R. Agarwal, D. Schuurmans, and M. Norouzi, “An optimistic perspective on offline reinforcement learning,” in International conference on machine learning. PMLR, 2020, pp. 104–114.

[11] X. Chen, C. Wang, Z. Zhou, and K. Ross, “Randomized ensembled double q-learning: Learning fast without a model,” in 9th International Conference on Learning Representations (ICLR 2021), 2021, arXiv:2101.05982.

[12] M. G. Bellemare, W. Dabney, and R. Munos, “A distributional perspective on reinforcement learning,” in International conference on machine learning. PMLR, 2017, pp. 449–458.

[13] W. Dabney, M. Rowland, M. G. Bellemare, and R. Munos, “Distributional reinforcement learning with quantile regression,” arXiv preprint arXiv:1710.10044, 2017.

[14] W. Dabney, G. Ostrovski, D. Silver, and R. Munos, “Implicit quantile networks for distributional reinforcement learning,” in International conference on machine learning. PMLR, 2018, pp. 1096–1105.

[15] A. Kuznetsov, P. Shvechikov, A. Grishin, and D. Vetrov, “Controlling overestimation bias with truncated mixture of continuous distributional quantile critics,” in International Conference on Machine Learning. PMLR, 2020, pp. 5556–5566.

[16] I. Osband, C. Blundell, A. Pritzel, and B. Van Roy, “Deep exploration via bootstrapped dqn,” Advances in neural information processing systems, vol. 29, 2016.

[17] K. Lee, M. Laskin, A. Srinivas, and P. Abbeel, “Sunrise: A simple unified framework for ensemble learning in deep reinforcement learning,” in International conference on machine learning. PMLR, 2021, pp. 6131–6141.

[18] D. Lee, S. Han, T. Cho, and J. Lee, “Spqr: controlling q-ensemble independence with spiked random model for reinforcement learning,” Advances in Neural Information Processing Systems, vol. 36, pp. 65 224–65 251, 2023.

[19] M. Laskin, A. Srinivas, and P. Abbeel, “Curl: Contrastive unsupervised representations for reinforcement learning,” in International conference on machine learning. PMLR, 2020, pp. 5639–5650.

[20] M. Schwarzer, A. Anand, R. Goel, R. D. Hjelm, A. Courville, and P. Bachman, “Data-efficient reinforcement learning with selfpredictive representations,” arXiv preprint arXiv:2007.05929, 2020.

[21] P. D’Oro, M. Schwarzer, E. Nikishin, P.-L. Bacon, M. G. Bellemare, and A. Courville, “Sample-efficient reinforcement learning by breaking the replay ratio barrier,” in Deep Reinforcement Learning Workshop NeurIPS 2022, 2022.

[22] D. Yarats, I. Kostrikov, and R. Fergus, “Image augmentation is all you need: Regularizing deep reinforcement learning from pixels,” in International conference on learning representations, 2021.

[23] M. Schwarzer, J. S. O. Ceron, A. Courville, M. G. Bellemare, R. Agarwal, and P. S. Castro, “Bigger, better, faster: Human-level atari with human-level efficiency,” in International Conference on Machine Learning. PMLR, 2023, pp. 30 365–30 380.

[24] J. Robine, M. Höftmann, and S. Harmeling, “Simple, good, fast: Self-supervised world models free of baggage,” arXiv preprint arXiv:2506.02612, 2025.

[25] W. Wang, I. Dusparic, Y. Shi, K. Zhang, and V. Cahill, “Drama: Mamba-enabled model-based reinforcement learning is sample and parameter efficient,” arXiv preprint arXiv:2410.08893, 2024.

[26] V. Micheli, E. Alonso, and F. Fleuret, “Transformers are sampleefficient world models,” arXiv preprint arXiv:2209.00588, 2022.

[27] W. Zhang, G. Wang, J. Sun, Y. Yuan, and G. Huang, “Storm: Efficient stochastic transformer based world models for reinforcement learning,” Advances in Neural Information Processing Systems, vol. 36, pp. 27 147–27 166, 2023.

[28] H. P. Van Hasselt, M. Hessel, and J. Aslanides, “When to use parametric models in reinforcement learning?” Advances in Neural Information Processing Systems, vol. 32, 2019.

[29] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski et al., “Human-level control through deep reinforcement learning,” nature, vol. 518, no. 7540, pp. 529–533, 2015.

[30] F. Gogianu, T. Berariu, M. C. Rosca, C. Clopath, L. Busoniu, and R. Pascanu, “Spectral normalisation for deep reinforcement learning: an optimisation perspective,” in International Conference on Machine Learning. PMLR, 2021, pp. 3734–3744.

[31] R. C. Castanyer, J. Obando-Ceron, L. Li, P.-L. Bacon, G. Berseth, A. Courville, and P. S. Castro, “Stable gradients for stable learning at scale in deep reinforcement learning,” arXiv preprint arXiv:2506.15544, 2025.

[32] J. Markowitz, J. Silverberg, and G. L. Collins, “Avoiding value estimation error in off-policy deep reinforcement learning,” in I Can’t Believe It’s Not Better Workshop: Failure Modes of Sequential Decision-Making in Practice (RLC 2024), 2024.

[33] M. Nauman, M. Bortkiewicz, P. Miło´s, T. Trzci ´nski, M. Ostaszewski, and M. Cygan, “Overestimation, overfitting, and plasticity in actorcritic: the bitter lesson of reinforcement learning,” arXiv preprint arXiv:2403.00514, 2024.

[34] J. T. Ash and R. P. Adams, “On warm-starting neural network training,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 3884–3894.

[35] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, “Mastering diverse domains through world models,” arXiv preprint arXiv:2301.04104, 2023.

[36] E. Alonso, A. Jelley, V. Micheli, A. Kanervisto, A. J. Storkey, T. Pearce, and F. Fleuret, “Diffusion for world modeling: Visual details matter in atari,” Advances in Neural Information Processing Systems, vol. 37, pp. 58 757–58 791, 2024.

[37] P. Agarwal, S. Andrews, and S. E. Kahou, “Learning to play atari in a world of tokens,” arXiv preprint arXiv:2406.01361, 2024.

[38] J. S. O. Ceron and P. S. Castro, “Revisiting rainbow: Promoting more insightful and inclusive deep reinforcement learning research,” in International Conference on Machine Learning. PMLR, 2021, pp. 1373–1383.

[39] K. Cobbe, C. Hesse, J. Hilton, and J. Schulman, “Leveraging procedural generation to benchmark reinforcement learning,” in International conference on machine learning. PMLR, 2020, pp. 2048– 2056.

![](images/e9b9af6ed3050bc98c8300934937b905694b90400c5063a15305a31ba2c1018d.jpg)  
Fig. 9: Layer-wise spike ratio evolution during training under different replay ratios. Each row corresponds to a network layer, and color intensity indicates spike magnitude.

This section provides detailed experimental settings and implementation details for the spike ratio analysis presented in Fig. 9(Fig. 2 of the main paper). The purpose of this experiment is to examine how replay ratio evolves over the course of learning.

To monitor parameter dynamics during training, we periodically compute the spike ratio. For a network layer ℓ with parameter tensor $\theta _ { \ell } ,$ , the spike ratio is defined as

$$
\mathrm { S p i k e R a t i o } _ { \ell } = \frac { \| \theta _ { \ell } \| _ { \infty } } { \mathrm { Q u a n t i l e } _ { 0 . 9 9 } ( | \theta _ { \ell } | ) } .\tag{22}
$$

The experiment follows the standard Atari-100K evaluation protocol described in the main paper. Training uses the same optimization settings as the main algorithm described in this paper. Unless otherwise specified, all hyperparameters remain identical to the main experiments. The replay ratio rr controls the number of gradient updates per environment step. To analyze the effect of data reuse, we conduct experiments with three different settings, $r r \in \{ 1 , 2 , \hat { 4 } \}$

The value function is implemented as an ensemble quantile network. Each ensemble member consists of a convolutional encoder followed by a linear projection head that outputs quantile estimates for all actions.

The encoder adopts a lightweight ResNet-style architecture. Specifically, the input observation $\boldsymbol { x } \in \mathbb { R } ^ { 4 \times 8 4 \times 8 4 }$ is processed by a sequence of convolutional layers and residual blocks to produce a latent feature vector.

Table 3 lists all trainable layers for which spike statistics are monitored during training. Since the model contains multiple ensemble members with identical architectures, spike ratios are first computed independently for each ensemble network and then averaged to obtain the aggregated statistics used in Figure 9.

## APPENDIX B

## BOUNDED BELLMAN UPDATE

This bound is derived from the structure of the discounted return and reflects the maximum achievable value under the sampled reward scale.

Consider a trajectory generated by greedily following the current value function. Starting from $( s _ { 0 } , a _ { 0 } )$ with $a _ { 0 } ~ \in$ arg $\operatorname { m a x } _ { a } Q ( s _ { 0 } , a )$ , we iteratively select

$$
a _ { t } \in \arg \operatorname* { m a x } _ { a } Q ( s _ { t } , a ) ,
$$

and obtain a greedy trajectory

$$
\mathcal { T } = \{ ( s _ { 0 } , a _ { 0 } , r _ { 0 } ) , ( s _ { 1 } , a _ { 1 } , r _ { 1 } ) , \dots , ( s _ { T } , a _ { T } , r _ { T } ) \} ,
$$

which terminates at time step $T .$

Along this greedy trajectory, the one-step greedy backup satisfies

$$
r _ { t } + \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q ( s _ { t + 1 } , a ^ { \prime } ) = r _ { t } + \gamma Q ( s _ { t + 1 } , a _ { t + 1 } ) ,\tag{23}
$$

TABLE 3: Network layers monitored for spike ratio statistics in Figure 9. The spike ratio is computed for each layer every 1000 environment steps to analyze how parameter imbalance evolves during training. The encoder’s channel width is controlled by a scale factor s (with $s = 4$ s = in our experiments).
<table><tr><td>Layer Group</td><td>Layer Name</td><td>Input Shape</td><td>Weight Shape</td></tr><tr><td>Stem</td><td>stem.0</td><td> $4 \times 8 4 \times 8 4$ </td><td> $( 8 s ) \times 4 \times 3 \times 3$ </td></tr><tr><td rowspan="4">Stage1</td><td>11.0.conv1</td><td> $( 8 s ) \times 8 4 \times 8 4$ </td><td> $( 8 s ) \times ( 8 s ) \times 3 \times 3$ </td></tr><tr><td>11.0.conv2</td><td>(8s)  $\times 8 4 \times 8 4$ </td><td> $( 8 s ) \times ( 8 s ) \times 3 \times 3$ </td></tr><tr><td>11.1.conv1</td><td> $( 8 s ) \times 8 4 \times 8 4$ </td><td> $( 8 s ) \times ( 8 s ) \times 3 \times 3$ </td></tr><tr><td>11.1.conv2</td><td> $( 8 s ) \times 8 4 \times 8 4$ </td><td> $( 8 s ) \times ( 8 s ) \times 3 \times 3$ </td></tr><tr><td rowspan="4">Stage2</td><td>12.0.conv1</td><td> $( 8 s ) \times 8 4 \times 8 4$ </td><td> $( 1 6 s ) \times ( 8 s ) \times 3 \times 3$ </td></tr><tr><td>12.0.conv2</td><td> $( 1 6 s ) \times 4 2 \times 4 2$ </td><td> $( 1 6 s ) \times ( 1 6 s ) \times 3 \times 3$ </td></tr><tr><td>12.1.conv1</td><td> $\dot { ( 1 6 s ) } \times 4 2 \times 4 2$ </td><td> $( 1 6 s ) \times ( 1 6 s ) \times 3 \times 3$ </td></tr><tr><td>12.1.conv2</td><td> $( 1 6 s ) \times 4 2 \times 4 2$ </td><td> $( 1 6 s ) \times ( 1 6 s ) \times 3 \times 3$ </td></tr><tr><td rowspan="4">Stage3</td><td>l3.0.conv1</td><td> $( 1 6 s ) \times 4 2 \times 4 2$ </td><td> $( 3 2 s ) \times ( 1 6 s ) \times 3 \times 3$ </td></tr><tr><td>13.0.conv2</td><td> $( 3 2 s ) \times 2 1 \times 2 1$ </td><td> $( 3 2 s ) \times ( 3 2 s ) \times 3 \times 3$ </td></tr><tr><td>l3.1.conv1</td><td> $( 3 2 s ) \times 2 1 \times 2 1$ </td><td> $( 3 2 s ) \times ( 3 2 s ) \times 3 \times 3$ </td></tr><tr><td>13.1.conv2</td><td> $( 3 2 s ) \times 2 1 \times 2 1$ </td><td> $( 3 2 s ) \times ( 3 2 s ) \times 3 \times 3$ </td></tr><tr><td rowspan="4">Stage4</td><td>14.0.conv1</td><td> $( 3 2 s ) \times 2 1 \times 2 1$ </td><td> $( 6 4 s ) \times ( 3 2 s ) \times 3 \times 3$ </td></tr><tr><td>14.0.conv2</td><td> $( 6 4 s ) \times 1 1 \times 1 1$ </td><td> $( 6 4 s ) \times ( 6 4 s ) \times 3 \times 3$ </td></tr><tr><td>14.1.conv1</td><td> $( 6 4 s ) \times 1 1 \times 1 1$ </td><td> $( 6 4 s ) \times ( 6 4 s ) \times 3 \times 3$ </td></tr><tr><td>14.1.conv2</td><td>(64s)  $\times 1 1 \times 1 1$ </td><td> $( 6 4 s ) \times ( 6 4 s ) \times 3 \times 3$ </td></tr><tr><td>Projection Head</td><td>head.1</td><td> $( 6 4 s ) \times 1 1 \times 1 1$ </td><td> $( 1 2 8 s ) \times ( 6 4 s \cdot 1 1 \cdot 1 1 )$ </td></tr><tr><td>Value Head</td><td>quantile head</td><td>128s</td><td> $( | \mathcal { A } | K ) \times ( 1 2 8 s )$ </td></tr></table>

because $a _ { t + 1 } \in$ arg max<sub>a</sub> $Q ( s _ { t + 1 } , a )$ by construction. Therefore, by recursively substituting the greedy action at each subsequent state, we obtain

$$
\begin{array} { r l } & { Q ( s _ { t } , a _ { t } ) = r _ { t } + \gamma \underset { a ^ { \prime } } { \mathrm { m a x } } Q ( s _ { t + 1 } , a ^ { \prime } ) } \\ & { \qquad = r _ { t } + \gamma Q \big ( s _ { t + 1 } , a _ { t + 1 } \big ) } \\ & { \qquad = r _ { t } + \gamma \Big ( r _ { t + 1 } + \gamma \underset { a ^ { \prime } } { \mathrm { m a x } } Q \big ( s _ { t + 2 } , a ^ { \prime } \big ) \Big ) } \\ & { \qquad = r _ { t } + \gamma r _ { t + 1 } + \gamma ^ { 2 } Q \big ( s _ { t + 2 } , a _ { t + 2 } \big ) } \\ & { \qquad \vdots } \\ & { \qquad = r _ { t } + \gamma r _ { t + 1 } + \cdot \cdot + \gamma ^ { T - t } Q \big ( s _ { T } , a _ { T } \big ) . } \end{array}\tag{24}
$$

If the episode terminates at $T$ with no future continuation (or equivalently $Q ( s _ { T } , a _ { T } ) = r _ { T }$ under terminal dynamics), the recursion closes and yields

$$
Q ( s _ { t } , a _ { t } ) = \sum _ { k = t } ^ { T } \gamma ^ { k - t } r _ { k } \ \triangleq \ G _ { t } .\tag{25}
$$

For a given greedy trajectory $\tau ,$ we define the trajectory-level value bound as

$$
B ( \mathcal { T } ) = \operatorname* { m a x } _ { 0 \leq t \leq T } G _ { t } .\tag{26}
$$

This greedy-rollout identity provides a natural way to use $G _ { t }$ as a trajectory-wise bound for the numerical scale of $Q ( s _ { t } , a _ { t } )$ under the current greedy policy.

## APPENDIX C

## ATARI EXPERIMENT DETAILS

This section provides additional implementation details for the Atari experiments reported in the main paper. The experiments follow the Atari-100K evaluation protocol with a total interaction budget of 100k environment steps.

## C.1 Environment Setup

All experiments are conducted using the Atari Learning Environment through Gymnasium. Observations follow the standard preprocessing pipeline implemented by the Stable-Baselines3 Atari wrapper.

Specifically, raw frames are converted to grayscale, resized to $8 4 \times 8 4 ,$ and stacked over four consecutive frames to form the input state representation. Frame skipping is set to 4, which corresponds to 400k environment frames under the Atari-100K setting.

The environment wrapper additionally applies a random number of no-op actions at the beginning of each episode (up to 30 steps). Episode termination follows the standard Atari convention, where life loss is treated as terminal during training.

Rewards are not clipped to $[ - 1 , 1 ]$ during environment interaction. However, the sign of the reward is stored in the replay buffer for training updates.

## C.2 Replay Buffer and Sampling

Experience replay uses a prioritized replay buffer with a capacity of 100, 000 transitions.

Transitions are sampled according to prioritized sampling with exponent $\alpha = 0 . 6$ . Importance sampling weights are applied during optimization using exponent $\beta = 0 . 4$

For each sampled transition, observations are normalized to [0, 1] by dividing pixel values by 255.

## C.3 Network Architecture

As Table 3, the value function is implemented as an ensemble quantile network. Each ensemble member consists of a convolutional encoder followed by a linear projection head.

The encoder adopts a lightweight ResNet-style architecture. The input observation has shape (4, 84, 84) and is processed through a sequence of convolutional residual blocks with progressive downsampling:

$$
{ 8 4 \times 8 4 }  { 8 4 } \times 8 4  4 2 \times 4 2  2 1 \times 2 1  1 1 \times 1 1 .
$$

The number of channels is controlled by a scale factor $s = 4$ . After the convolutional encoder, features are flattened and projected to a fully connected layer of size 512s.

The final layer outputs K quantile values for each action. With $N _ { e }$ ensemble members and K quantile atoms, the network produces a tensor of shape

$$
( B , N _ { e } , K , | { \mathcal { A } } | ) ,
$$

where B denotes the batch size.

## C.4 Training Procedure

The agent is trained for 100, 000 environment steps. During training, the agent interacts with the environment using an ϵ-greedy policy.

The exploration parameter ϵ is linearly annealed from 1.0 to 0.01 over the first 2000 steps.

Learning begins after 2000 transitions have been collected. At each environment step, rr gradient updates are performed, where rr denotes the replay ratio.

Optimization uses the Adam optimizer with learning rate $1 \times 1 0 ^ { - 4 }$

Target networks are updated using Polyak averaging with coefficient $\tau = 0 . 0 0 5$

Gradients are clipped with a maximum norm of 10.

## C.5 Distributional Value Learning

The value function is trained using quantile regression with the quantile Huber loss. The number of quantile atoms is $K = 5 1$

Quantile fractions follow the standard QR-DQN formulation

$$
\tau _ { i } = \frac { i + 0 . 5 } { K } , \quad i = 0 , \dots , K - 1 .
$$

For each update, the target distribution is constructed using the target network and the greedy action selected according to the mean value across quantiles.

## C.6 Ensemble Learning

The proposed method employs an ensemble of $N _ { e } = 1 6$ value networks. Each ensemble member independently estimates quantile value distributions for all actions.

During action selection, quantile values are averaged across both the quantile dimension and the ensemble dimension to produce the final action-value estimates.

## C.7 Parameter Reset Mechanism

$$
\theta ,
$$

$$
\operatorname { S p i k e R a t i o } = { \frac { \operatorname* { m a x } | \theta | } { \operatorname { Q u a n t i l e } _ { 0 . 9 9 } ( | \theta | ) } } .
$$

If the spike ratio of a layer exceeds the threshold, the parameters of that layer are reinitialized.

The optimizer states associated with the reset parameters are also cleared, and the target network is synchronized with the updated parameters.

## C.8 Hardware and Implementation

All experiments were conducted on a single NVIDIA RTX 5090 GPU. The implementation is written in PyTorch and runs on a Linux environment with CUDA acceleration. All reported results are obtained using the same hardware configuration.

## C.9 Hyperparameters

The hyperparameters used in the Atari experiments are summarized in Table 4.

TABLE 4: Hyperparameter configuration used for Atari-100K experiments.
<table><tr><td colspan="2">Training Configuration (Atari-100K)</td></tr><tr><td>Total environment steps</td><td>100,000</td></tr><tr><td>Discount factor γ</td><td>0.99</td></tr><tr><td>Learning rate</td><td>1 × 10−4 (Adam)</td></tr><tr><td>Batch size</td><td>32</td></tr><tr><td>Replay buffer size</td><td>100,000</td></tr><tr><td>Start learning after steps</td><td>2,000</td></tr><tr><td>Training frequency</td><td>1 update per step</td></tr><tr><td>Soft target update rate τ</td><td>0.005</td></tr><tr><td>Replay ratio Gradient clipping</td><td>4</td></tr><tr><td>Parameter reset threshold</td><td>10.0 6.0</td></tr><tr><td></td><td></td></tr><tr><td colspan="2">Exploration</td></tr><tr><td>Initial €</td><td>1.0</td></tr><tr><td>Final €</td><td>0.01</td></tr><tr><td>€ decay steps</td><td>2,001</td></tr><tr><td colspan="2">Replay Buffer</td></tr><tr><td>Prioritization exponent α</td><td>0.6</td></tr><tr><td>Importance sampling exponent β Priority epsilon</td><td>0.4 1 × 10−6</td></tr><tr><td></td><td>Distributional / Ensemble Settings</td></tr><tr><td colspan="2"></td></tr><tr><td>Number of quantiles</td><td>51 16</td></tr><tr><td>Number of ensemble heads Quantile Huber parameter κ</td><td>1</td></tr><tr><td></td><td></td></tr><tr><td colspan="2">Observation Processing</td></tr><tr><td>Frame stack</td><td>4</td></tr><tr><td>Frame skip Observation resolution</td><td>4</td></tr><tr><td></td><td>84 × 84</td></tr></table>

## C.10 Experimental Results

Detailed performance comparisons on the Atari benchmark are provided in Table 5.

## APPENDIX D

## PROCGEN EXPERIMENT DETAILS

This section provides additional implementation details for the Procgen experiments reported in the main paper. All experiments follow the standard Procgen generalization protocol.

## D.1 Environment Setup

We evaluate our method on the Procgen benchmark under the generalization mode. In this setting, the agent is trained on a fixed set of procedurally generated levels and evaluated on unseen levels.

Specifically, training environments use the first 200 levels of each game (start\_level = 0, num\_levels = 200), while evaluation is performed on disjoint levels starting from level 200 (start\_level = 200, num\_levels = 0), which corresponds to an infinite stream of unseen levels.

All environments are run in easy difficulty mode. Observations are RGB images with resolution 64 × 64.

Following common practice in prior work, we use a vectorized environment with 128 parallel instances during training to improve data throughput.

<sub>K</sub> <sub>performance</sub> <sub>comparison</sub> <sub>acr</sub><sup>oss</sup> <sup>environments.</sup> <sup>The</sup> <sup>best</sup> <sup>score</sup> <sup>in</sup> <sup>each</sup> <sup>row</sup> <sup>is</sup>
<table><tr><td>Env</td><td>Random</td><td>Human</td><td>DER</td><td>DrQ</td><td>IRIS</td><td>REM STORM</td><td>DreamerV3</td><td></td><td>DIAMOND</td><td>DART</td><td>SGF</td><td>Drama</td><td>BBF</td><td>Ours</td></tr><tr><td>Alien</td><td>227.8</td><td>7127.7</td><td>802.3</td><td>865.2</td><td>420.0</td><td>607.2</td><td>983.6</td><td>959.0</td><td>744.1</td><td>962.0</td><td>518.8</td><td>820</td><td>1173.2</td><td>1340.0</td></tr><tr><td>Amidar</td><td>5.8</td><td>1719.5</td><td>125.9</td><td>137.8</td><td>143.0</td><td>95.3</td><td>204.8</td><td>139.0</td><td>225.8</td><td>125.7</td><td>62.7</td><td>131</td><td>244.6</td><td>293.6</td></tr><tr><td>Assault</td><td>222.4</td><td>742.0</td><td>561.5</td><td>579.6</td><td>1524.4</td><td>1764.2</td><td>801.0</td><td>706.0</td><td>1526.4</td><td>1316.0</td><td>850.1 539</td><td></td><td>2098.5 2348.3</td><td></td></tr><tr><td>Asterix</td><td>210.0</td><td>8503.3</td><td>535.4</td><td>763.6</td><td>853.6</td><td>1637.5</td><td>1028.0</td><td>932.0</td><td>3698.5</td><td>956.2</td><td>802.5</td><td>1632</td><td>3946.1 2430.0</td><td></td></tr><tr><td>BankHeist</td><td>14.2</td><td>753.1</td><td>185.5</td><td>232.9</td><td>53.1</td><td>19.2</td><td>641.2</td><td>649.0</td><td>19.7</td><td>629.7</td><td>58.7</td><td>137</td><td>732.9 772.0</td><td></td></tr><tr><td>BattleZone</td><td>2360.0</td><td>37187.5</td><td>8977.0</td><td>10165.3</td><td>13074.0</td><td>11826.0</td><td>13540.0</td><td>12250.0</td><td>4702.0</td><td>15325.0</td><td>3747.0</td><td>10860</td><td>24459.8 18600.0</td><td></td></tr><tr><td>Boxing</td><td>0.1</td><td>12.1</td><td>-0.3</td><td>9.0</td><td>70.1</td><td>87.5</td><td>79.7</td><td>78.0</td><td>86.9</td><td>83.0</td><td>83.4</td><td>78</td><td>85.8 25.1</td><td></td></tr><tr><td>Breakout</td><td>1.7</td><td>30.5</td><td>9.2</td><td>19.8</td><td>83.7</td><td>90.7</td><td>15.9</td><td>31.0</td><td>132.5</td><td>41.9</td><td>50.7</td><td>370.6</td><td>233.0</td><td></td></tr><tr><td>ChopperCommand</td><td>811.0</td><td>7387.8</td><td>925.9</td><td>844.6</td><td>1565.0</td><td>2561.2</td><td>1888.0</td><td>420.0</td><td>1369.8</td><td>1263.8</td><td>1775.4 1642</td><td>7549.3</td><td>2740.0</td><td></td></tr><tr><td>CrazyClimber</td><td>10780.5</td><td>35829.4</td><td>34508.6</td><td>21539.0</td><td>59324.2</td><td>76547.6</td><td>66776.0</td><td>97190.0</td><td>99167.8</td><td>34070.6</td><td>15751.3 83931</td><td>58431.8</td><td>84700.0</td><td></td></tr><tr><td>DemonAttack</td><td>152.1</td><td>1971.0</td><td>627.6</td><td>1321.5</td><td>2034.4</td><td>5738.6</td><td>164.6</td><td>303.0</td><td>288.1</td><td>2452.3</td><td>2809.5 201</td><td>13341.4 25.5</td><td>7192.5</td><td></td></tr><tr><td>Freeway</td><td>0.0</td><td>29.6</td><td>20.9</td><td>20.3</td><td>31.1</td><td>32.3</td><td>33.5</td><td>0.0</td><td>33.3</td><td>32.2</td><td>11.9</td><td></td><td>33.3</td><td></td></tr><tr><td>Frostbite</td><td>65.2</td><td>4334.7</td><td>871.0</td><td>1014.2</td><td>259.1</td><td>240.5</td><td>1316.0</td><td>909.0</td><td>274.1</td><td>346.8</td><td>265.6 785</td><td>2384.8</td><td>2886.0</td><td></td></tr><tr><td>Gopher</td><td>257.6</td><td>2412.5</td><td>467.0</td><td>621.6</td><td>2236.1</td><td>5452.4</td><td>8239.6</td><td>3730.0</td><td>5897.9</td><td>1980.5</td><td>416.4 2757</td><td>1331.2</td><td>2346.0</td><td></td></tr><tr><td>Hero</td><td>1027.0</td><td>30826.4</td><td>6226.0</td><td>4167.9</td><td>7037.4</td><td>6484.8</td><td>11044.3</td><td>11161.0</td><td>5621.8</td><td>4927.0</td><td>1522.9 7946</td><td>7818.6</td><td>13436.0</td><td></td></tr><tr><td>Jamesbond</td><td>29.0</td><td>302.8</td><td>275.7</td><td>349.1</td><td>462.7</td><td>391.2</td><td>509.0</td><td>445.0</td><td>427.4</td><td>353.1</td><td>280.9 372</td><td>1129.6</td><td>510.0</td><td></td></tr><tr><td>Kangaroo</td><td>52.0</td><td>3035.0</td><td>581.7</td><td>1088.4</td><td>838.2</td><td>467.6</td><td>4208.0</td><td>4098.0</td><td>5382.2</td><td>2380.0</td><td>271.2</td><td>1384 6614.7</td><td>7520.0</td><td></td></tr><tr><td>Krull</td><td>1598.0</td><td>2665.5</td><td>3256.9</td><td>4402.1</td><td>6616.4</td><td>4017.7</td><td>8412.6</td><td>7782.0</td><td>8610.1</td><td>7658.3</td><td>7813.7</td><td>9693 8223.4</td><td>7851.0</td><td></td></tr><tr><td>KungFuMaster</td><td>258.5</td><td>22736.3</td><td>6580.1</td><td>11467.4</td><td>21759.8</td><td>25172.2</td><td>26182.0</td><td>21420.0</td><td>18713.6</td><td>23744.3</td><td>20169.8 23920</td><td>18991.7</td><td>24190.0</td><td></td></tr><tr><td>MsPacman</td><td>307.3</td><td>6951.6</td><td>1187.4</td><td>1218.1</td><td>999.1</td><td>962.5</td><td>2673.5</td><td>1327.0</td><td>1958.2</td><td>1132.7</td><td>1356.8 2270</td><td></td><td>2008.3</td><td>1986.0</td></tr><tr><td>Pong PrivateEye</td><td>-20.7 24.9</td><td>14.6 69571.3</td><td>-9.7</td><td>-9.1</td><td>14.6</td><td>18.0</td><td>11.3</td><td>18.0</td><td>20.4</td><td>17.2</td><td>12.6</td><td></td><td></td><td>21.0</td></tr><tr><td>Qbert</td><td>163.9</td><td>13455.0</td><td>72.8 1773.5</td><td>3.5 1810.7</td><td>100.0</td><td>99.6</td><td>7781.0</td><td>882.0</td><td>114.3</td><td>765.7</td><td>405.5</td><td></td><td></td><td>100.0</td></tr><tr><td>RoadRunner</td><td>11.5</td><td>7845.0</td><td>11843.4</td><td></td><td>745.7</td><td>743.0</td><td>4522.5</td><td>3405.0</td><td>4499.3</td><td>750.9</td><td>685.0</td><td>796</td><td>4447.1</td><td>4985.0</td></tr><tr><td>Seaquest</td><td>68.4</td><td>42054.7</td><td>304.6</td><td>11211.4 352.3</td><td>9614.6</td><td>14060.2</td><td>17564.0</td><td>15565.0</td><td>20673.2</td><td>7772.5</td><td>8164.2</td><td>14020</td><td>33426.8</td><td>32760.0</td></tr><tr><td>UpNDown</td><td>533.4</td><td>11693.2</td><td>3075.0</td><td>4324.5</td><td>661.3 3546.2</td><td>1036.7 3757.6</td><td>525.2</td><td>618.0</td><td>551.2</td><td>895.8 3954.5</td><td>476.8 7745.0</td><td>497</td><td>1232.5</td><td>1198.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>7985.0</td><td>9234.0</td><td>3856.3</td><td></td><td></td><td>7387</td><td>12101.7</td><td>33994.0</td></tr><tr><td># &gt; Human # Best</td><td></td><td>0</td><td>20</td><td></td><td>9</td><td>12</td><td>10</td><td>91</td><td>11</td><td>9</td><td>60</td><td>81</td><td>12 8</td><td>14</td></tr><tr><td></td><td>00</td><td>0</td><td></td><td>30 0.465</td><td>0 1.046</td><td>1 1.222</td><td>5 1.266</td><td>1.120</td><td>0 1.459</td><td>0 1.022</td><td>0.884 1.049</td><td></td><td>2.247</td><td>10 1.799</td></tr><tr><td>Mean HNS Median HNS</td><td>0.000 0.000</td><td>1.000 1.000</td><td>0.350 0.189</td></table>

## D.2 Observation Processing

Unlike Atari, Procgen observations are already provided as RGB images with fixed resolution. Therefore no grayscale conversion or resizing is required.

Observations are represented as tensors of shape (3, 64, 64) corresponding to RGB channels. Pixel values are normalized to [0, 1] before being fed into the neural network.

Frame stacking is not used in our experiments since Procgen environments are fully observable.

## D.3 Network Architecture

The value function is implemented as the same ensemble quantile network used in the Atari experiments, but with an encoder adapted to the $6 \bar { 4 } \times 6 4$ Procgen observations.

The encoder follows a lightweight residual architecture. Starting from a $6 4 \times 6 4$ input image, the spatial resolution is progressively reduced through strided residual blocks:

$$
6 4 \times 6 4  3 2 \times 3 2  1 6 \times 1 6  8 \times 8  4 \times 4 .
$$

The resulting feature map is flattened and projected to a 512-dimensional feature vector. The final layer outputs K quantile values for each action.

With N<sub>e</sub> ensemble members and K quantiles, the network produces a tensor of shape

$$
( B , N _ { e } , K , | { \mathcal { A } } | ) ,
$$

where B denotes the batch size.

## D.4 Training Procedure

Agents are trained for a total of 25 million environment steps.

Exploration follows an ϵ-greedy strategy. The exploration rate is linearly annealed from 1.0 to 0.05 over the first 5 million environment steps.

Training begins after an initial replay buffer warm-up period. During training, the agent performs gradient updates using mini-batches sampled from the replay buffer.

The optimization objective is the quantile regression loss. The Adam optimizer is used for parameter updates.

## D.5 Replay

The replay buffer stores up to 100, 000 transitions. Mini-batches of size 256 are sampled uniformly during training.

## D.6 Ensemble Distributional Value Learning

Our method employs an ensemble of $N _ { e } = 1 6$ value networks. Each ensemble member independently estimates the quantile value distribution for each action.

During action selection, the predicted quantiles are averaged across both the ensemble dimension and the quantile dimension to produce the final action-value estimates.

## D.7 Hardware and Implementation

All Procgen experiments are implemented in PyTorch and executed on a single NVIDIA RTX 5090 GPU. The implementation uses vectorized environments with CUDA acceleration to improve training throughput.

The hyperparameters used for the Procgen experiments are summarized in Table 6.

TABLE 6: Hyperparameters for Procgen (generalization mode, 200 training levels).
<table><tr><td>Total environment steps</td><td>25M</td></tr><tr><td>Training levels</td><td>200</td></tr><tr><td>Evaluation levels</td><td>Unseen procedural levels</td></tr><tr><td>Parallel environments</td><td>128</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Discount factor γ</td><td>0.99</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td>1e-4</td></tr><tr><td>Soft target update rate τ</td><td>0.005</td></tr><tr><td>Ensemble size Ne Quantile number K</td><td>16</td></tr><tr><td>Initial exploration €</td><td>51</td></tr><tr><td>Final exploration €</td><td>1.0</td></tr><tr><td>€ decay steps</td><td>0.05 5M</td></tr><tr><td></td><td></td></tr></table>

TABLE 7: Per-environment raw test performance on Procgen (easy, 200 training levels).
<table><tr><td>Environment</td><td>PPO (Impala CNN)</td><td>Ours</td></tr><tr><td>BigFish</td><td>3.0</td><td>18.1</td></tr><tr><td>BossFight</td><td>8.3</td><td>8.1</td></tr><tr><td>CaveFlyer</td><td>5.6</td><td>7.0</td></tr><tr><td>Chaser</td><td>5.5</td><td>1.9</td></tr><tr><td>Climber</td><td>6.1</td><td>7.5</td></tr><tr><td>CoinRun</td><td>8.8</td><td>6.0</td></tr><tr><td>Dodgeball</td><td>2.2</td><td>16.0</td></tr><tr><td>FruitBot</td><td>27.0</td><td>25.7</td></tr><tr><td>Heist</td><td>2.2</td><td>1.0</td></tr><tr><td>Jumper</td><td>5.8</td><td>6.0</td></tr><tr><td>Leaper</td><td>4.4</td><td>6.0</td></tr><tr><td>Maze</td><td>5.7</td><td>2.0</td></tr><tr><td>Miner</td><td>9.0</td><td>6.1</td></tr><tr><td>Ninja</td><td>5.8</td><td>6.0</td></tr><tr><td>Plunder</td><td>5.4</td><td>5.5</td></tr><tr><td>StarPilot</td><td>25.0</td><td>54.9</td></tr></table>

## D.8 Detailed Per-Environment Results

To provide a more comprehensive view of performance across different tasks, we report the raw test scores for each Procgen environment in Table 7 under the generalization protocol.

In addition to the final evaluation scores, we also visualize the training dynamics for each environment. Figure 10 shows the training performance curves throughout the entire learning process. Each subplot corresponds to a different Procgen environment, and the horizontal axis represents the number of environment interaction steps.

![](images/6489e5c42231fad94d9a1e61266ddf16c4817836e272c3b1e75871bc506d7e00.jpg)  
Fig. 10: Per-environment train performance during training on Procgen.