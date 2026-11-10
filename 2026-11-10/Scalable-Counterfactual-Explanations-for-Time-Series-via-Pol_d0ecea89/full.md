# Scalable Counterfactual Explanations for Time Series via Policy Learning

Peiyu Li\*, Pouya Hosseinzadeh†, Soukaïna Filali Boubrahimi†, Shah Muhammad Hamdi†

\* McComish Department of Electrical Engineering and Computer Science,

South Dakota State University, Brookings, SD, USA

Email:peiyu.li@sdstate.edu

† Department of Computer Science,

Utah State University, Logan, UT, USA

Email: {pouya.hosseinzadeh, soukaina.boubrahimi, s.hamdi}@usu.edu

Abstract—Generating counterfactual (CF) explanations for time series is challenging due to complex temporal dependencies and the computational inefficiency of traditional instance-specific optimization. We propose a policy learning framework for counterfactual explanation in time series, trained as a one-step contextual-bandit policy-learning problem. By training an actorcritic agent in the latent space of a pre-trained autoencoder, our method learns a transferable policy that captures global structural regularities. Unlike existing methods that rely on iterative optimization for each new input, the learned policy enables direct generation of valid and plausible counterfactuals through a single forward pass, avoiding per-instance optimization at inference time. Experiments on 13 UCR benchmark datasets show that our framework achieves high validity and strong plausibility across multiple anomaly detection metrics. Overall, by decoupling counterfactual generation from per-instance optimization, the proposed method provides an efficient and scalable alternative to existing approaches for time series counterfactual explanations.

Index Terms—Explainable Artificial Intelligence, Counterfactual Explanation, Policy Learning, Time Series

## I. INTRODUCTION

Machine-learning models are increasingly used to classify time series in high-stakes domains such as healthcare monitoring, financial risk assessment, and environmental forecasting [1]–[3]. Although deep models achieve strong predictive performance, their black-box nature limits transparency and trust [4]. Counterfactual explanations (CFEs) address this limitation by identifying plausible changes to an input that would produce a desired prediction [5]. Such explanations show how the model's decision responds to changes in the input rather than merely identifying features associated with the original prediction.

Generating meaningful CFEs for time series is particularly challenging. Unlike tabular features, time-series values form temporally dependent trajectories whose trends, continuity, and recurring patterns must be preserved. Pointwise or segmentlevel changes can therefore introduce abrupt discontinuities or unrealistic shapes [6], [7]. Optimization-based approaches seek to improve validity and temporal coherence through combinations of prediction, distance, and structural constraints [8]–[11]. However, most existing approaches remain instancebased: they perform a new iterative search for every test input, whether in the input space or a learned latent space.

Consequently, their computation grows with both the number of instances and the cost of the optimization procedure, while knowledge gained from one search is not reused for subsequent inputs. This makes it difficult to jointly achieve validity, plausibility, and scalability.

To address this limitation, we formulate counterfactual generation as a one-step reinforcement-learning problem, or contextual bandit. A pre-trained autoencoder maps each input x to a compact latent representation z. Given z, the classifier's original prediction $y _ { M }$ , and a desired target class $y _ { T }$ , an actor produces a counterfactual latent representation $z _ { C F } ,$ which the decoder maps back to the input space. The black-box classifier then provides an immediate binary reward indicating whether the decoded candidate is assigned to yT. A critic learns a differentiable approximation of this reward and guides the actor, while an $L _ { 1 } + L _ { 2 }$ proximity penalty limits deviations from the original input. By learning from interactions collected across many training instances, the actor acquires a reusable generation policy instead of restarting an optimization procedure for each input. Once trained, it generates counterfactuals in batches through a single actor forward pass followed by decoding. The framework requires only classifier predictions and thus does not require gradient access to the black-box model.

We evaluate the proposed framework on 13 datasets from the UCR Time Series Classification Archive and compare it with optimization-based, saliency-guided, and latent-space baselines. The results show high and stable validity together with consistently strong plausibility under Isolation Forest, Local Outlier Factor, and One-Class SVM scores. The method produces larger aggregate $L _ { 1 }$ and $L _ { 2 }$ distances than localized input-space methods, but remains competitive in $L _ { \infty }$ and avoids extreme pointwise changes. Runtime results further show that the benefit of the learned policy increases with the number of test instances, making it particularly suitable for large-scale counterfactual generation.

The main contributions of this work are as follows:

• We formulate time-series counterfactual generation as a one-step contextual-bandit problem that learns a transferable policy and eliminates per-instance optimization at inference time.

![](images/12f209ab4cd405bb716108b423ff67dc8e92b712df9f9bbe02fcad9146b479bc.jpg)  
Fig. 1: Comparison of instance-based counterfactual generation methods and the proposed policy-learning framework.

• We develop a black-box actor-critic framework that generates counterfactuals in an autoencoder latent space and combines reward maximization with input-space proximity regularization.

• We evaluate validity, proximity, plausibility, ablation behavior, and runtime on 13 UCR datasets, demonstrating reliable generation and increasing scalability benefits as the test set grows.

## II. RELATED WORK

Counterfactual explanations (CFEs) have received increasing attention because of their ability to provide intuitive and actionable insights into machine learning predictions. By identifying minimal changes that alter a model's decision, CFEs help users understand model behavior and explore alternative outcomes. Such explanations are especially valuable in highstakes applications, where transparency and interpretability are as important as accuracy.

Early efforts in time series CFEs relied on instance-based heuristics, producing counterfactuals by perturbing each instance individually, often through segment replacement. For example, NG [6] and MG-CF [7] construct counterfactuals by altering or replacing salient subsequences drawn from the data (e.g., CAM-highlighted regions, class motifs). Their main strength is interpretability, since the modifications are grounded in real observed patterns that domain experts can readily relate to. In addition, because they reuse real subsequences, these methods often produce results that appear plausible at the local level. However, they depend heavily on the quality of saliency or motif mining and can struggle with validity (failing to consistently flip the label) and global coherence (as a swapped segment may not integrate smoothly with the rest of the sequence), which reduces their robustness across datasets.

Building on this idea, optimization-based approaches formulate counterfactual generation as an optimization problem for each input instance. The classic formulation by [5], originally introduced for tabular data, has since been widely adopted as a baseline in time series studies [7], [9], [12]. Wachter typically minimizes a loss that balances prediction validity (ensuring the label flips) with proximity to the original instance, often measured by L1 or L2 distance. While this formulation is simple and intuitive, proximity constraints alone cannot guarantee temporal plausibility, and naïve pointwise perturbations often disrupt sequential dynamics, leading to unrealistic counterfactuals. To address this limitation, constrained optimization variants have been proposed. For example, SG-CF [9] and TeRCE [13] restrict perturbations to subsequences defined by shapelets or temporal rules, while TimeX [8] enforces temporal coherence through barycenter averaging. These constraints improve local interpretability and coherence but introduce additional rigidity, as they rely on predefined structures that may not capture global dependencies, and they still suffer from the inefficiency of per-instance optimization.

A related direction uses saliency-guided masks to localize perturbations and improve interpretability. Methods such as CELS [14] and Info-CELS [15] learn perturbation masks guided by gradient information and the nearest unlike neighbor, enabling sparse and interpretable modifications without the need for predefined structures or rules. These approaches produce intuitive visual explanations by highlighting learned saliency maps. However, a key limitation is their reliance on the nearest unlike neighbor: counterfactuals are generated by interpolating between the original instance and this neighbor, with the interpolation rate controlled by the saliency map. If the nearest unlike neighbor exhibits temporal shifts or misalignments, the resulting counterfactuals can still be implausible, even if the label flips successfully.

Latent-space perturbation has recently emerged as a promising direction for improving plausibility. Glacier [11] perturbs representations in both the input space and autoencoderderived latent space, guided by saliency explainers such as LimeSegment [16]. The saliency vectors serve as constraints that determine which time steps should be perturbed, helping to localize modifications and preserve temporal structure. While this approach improves temporal localization, its performance remains highly sensitive to the quality of the saliency explainer. If the saliency vectors highlight noisy or irrelevant regions, the generated counterfactuals may still lack plausibility and robustness.

All of the methods discussed above are fundamentally instance-based: counterfactuals are generated independently for each test input via iterative replacement or optimization in either the raw feature space or a learned latent space. This design limits scalability and generalization, as the search must be repeated from scratch for every new instance. A small number of recent studies have explored reinforcement learning-based counterfactual generation in tabular domains, where agents learn action policies to apply minimal perturbations that flip model predictions [17]-[19]. These works demonstrate the promise of experience-driven policy learning but rely on the assumption of independent, semantically meaningful features. Unlike tabular data, time series consist of temporally dependent trajectories, and treating each time step as an independent feature leads to temporally incoherent perturbations and an impractically large action space, resulting in unrealistic counterfactuals and poor scalability. Motivated by this gap, we revisit the RL perspective for time series classification. Instead of per-instance optimization or perturbing raw time steps, our approach learns a transferable policy in a structured latent space that captures temporal dependencies, enabling temporally coherent counterfactuals and scalable, single-pass inference on unseen inputs.

## III. PRELIMINARY

## A. Counterfactual Explanation for Time Series Classification

We consider a dataset of N time series $\mathcal { X } = \{ x _ { 1 } , \ldots , x _ { N } \}$ where each $x _ { i } \in \mathbb { R } ^ { T }$ is a sequence of length $T .$ Let $M : \mathbb { R } ^ { T } $ $\mathcal { V }$ be a black-box classifier that maps each time series to one of $C$ class labels in Y. For an input instance x, the model prediction is $y _ { M } = M ( x )$ . Given a target label $y _ { T } \neq y _ { M }$ , the goal of counterfactual explanation is to find a perturbed time series $x _ { C F }$ such that

$$
\begin{array} { r } { x _ { C F } = x + \delta , \quad \delta \in \mathbb { R } ^ { T } , \quad M ( x _ { C F } ) = y _ { T } . } \end{array}\tag{1}
$$

The perturbation δ describes the changes needed to alter the model prediction from $y _ { M }$ to the desired target label $y _ { T }$

## IV. METHODOLOGY

Instead of relying on instance-based optimization, where each counterfactual must be obtained by perturbing the input through the same optimization process for every instance, we employ an actor-critic reinforcement learning framework to learn a policy that generates counterfactuals efficiently. Once trained, the policy produces counterfactuals in batches through a single forward pass, offering greater scalability than perinstance optimization.

Formally, we formulate counterfactual generation as a onestep Markov decision process (contextual bandit):

$$
\begin{array} { r } { \mathcal { M } = ( \mathcal { S } , A , R ) , \quad s \in \mathcal { S } , \ a \in \mathcal { A } , \ R = R ( s , a ) , } \end{array}\tag{2}
$$

where S is the state space, with $s = ( z , y _ { M } , y _ { T } )$ representing the latent embedding z, current prediction $y _ { M } ,$ and target class $y _ { T } ; \ A$ is the action space, where $a \ = \ z _ { C F }$ represents the modified latent embedding produced by the actor; and R is the immediate reward function: $R ( s , a ) = 1$ if $M ( \operatorname* { d e c } ( a ) ) = y _ { T }$ and 0 otherwise.

Note that there is no transition function P or next state $s ^ { \prime } { : }$ each counterfactual generation constitutes a single decision, after which the episode terminates. This differs from traditional multi-step RL, where agents navigate through sequential states. The environment in our formulation is the interaction between the time series representation and the classifier M. Given a state s, the actor generates an action $^ { a , }$ which is decoded and evaluated by M to produce an immediate binary reward. Unlike sequential RL problems with long-term credit assignment, our one-step formulation provides direct feedback after each decision.

To handle the high dimensionality and temporal structure of time series, we first train an autoencoder to obtain a lowerdimensional latent representation $z = \operatorname { e n c } ( x )$ . Operating in this compact latent space significantly reduces the effective action dimensionality for the RL agent, avoiding the need to perturb every time step of the raw sequence. Counterfactual generation is then performed in this latent space, where perturbations are smoother and more structured. The actor network $\mu _ { \theta }$ takes the latent state z as input and outputs a perturbed embedding $z _ { C F }$ corresponding to a counterfactual candidate. This representation is decoded back to the input space, $\tilde { x } _ { C F } ~ = ~ \operatorname * { d e c } ( z _ { C F } )$ , which is then evaluated by the classifier to determine the reward.

We adopt a simplified deterministic actor-critic framework inspired by Deep Deterministic Policy Gradient (DDPG) [20]. The critic $Q _ { \phi } ( s , a )$ estimates the value of applying action a in state s and is trained by regressing directly to the observed (immediate) reward signal:

$$
L _ { \mathrm { c r i t i c } } = \mathbb { E } \Big [ \big ( Q _ { \phi } ( s _ { t } , a _ { t } ) - R _ { t } \big ) ^ { 2 } \Big ] .\tag{3}
$$

Critic in One-Step Settings. In standard multi-step DDPG, the critic estimates long-term cumulative rewards: $Q ( s , a ) =$ $\mathbb { E } [ R _ { t } + \gamma R _ { t + 1 } + \gamma ^ { 2 } R _ { t + 2 } + \cdot \cdot \cdot ]$ . However, in our one-step formulation, the critic simplifies to predicting immediate reward expectations:

$$
Q ( s , a ) \approx \mathbb { E } [ R \mid s , a ] \quad { \mathrm { w h e r e ~ } } R \in \{ 0 , 1 \} .\tag{4}
$$

The critic loss becomes a direct regression to observed binary rewards without bootstrapping from future values. This differs from standard DDPG's temporal-difference target: $Q _ { \mathrm { t a r g e t } } =$ $R + \gamma Q ^ { \prime } ( s ^ { \prime } , a ^ { \prime } )$ . We have no $s ^ { \prime }$ (next state) or $\gamma$ (discount factor).

Despite this simplification, the critic plays three essential roles in our framework. First, it reduces variance by learning a smooth function $Q ( s , a )$ that approximates the expected reward, providing more stable gradients than the binary flipreward signal. Second, it enables off-policy learning through experience replay, allowing stored $( s , a , R )$ tuples to be reused efficiently even as the actor policy evolves. Finally, it offers a differentiable surrogate for the non-differentiable classifier and reward, allowing the actor to be optimized via gradients of $Q ( s , \mu _ { \theta } ( s ) )$ 1

![](images/20c379ab7103099894306895faba12d20f734f5d97a2fa63004142eef9704cc2.jpg)  
Fig. 2: The overview of the proposed framework.

The actor is trained to maximize the critic's estimate of value through the deterministic policy gradient:

$$
L _ { \mathrm { m a x } } = - \mathbb { E } \big [ Q _ { \phi } ( s _ { t } , \mu _ { \theta } ( s _ { t } ) ) \big ] .\tag{5}
$$

To promote similarity between the counterfactual and the original input, we add a proximity regularizer that combines $L _ { 1 }$ and $L _ { 2 }$ distances:

$$
L _ { \mathrm { p r o x } } = \beta \| \tilde { x } _ { C F } - x \| _ { 1 } + \| \tilde { x } _ { C F } - x \| _ { 2 } ^ { 2 } .\tag{6}
$$

The final actor objective is then

$$
L _ { \mathrm { a c t o r } } ^ { \mathrm { t o t a l } } = L _ { \mathrm { m a x } } + \lambda _ { \mathrm { p } } L _ { \mathrm { p r o x } } .\tag{7}
$$

During training, exploration is encouraged by injecting noise into the actor's proposed perturbed embedding, gradually transitioning from uniform random actions to noisy perturbations around the actor's output. Through this combination, the actor learns to generate counterfactuals that maximize validity while preserving proximity, and the decoder ensures that the resulting sequences remain realistic and temporally coherent

a) Training Workflow.: Algorithm 1 and Figure 2 summarizes the training pipeline of our policy learning framework for counterfactual generation in time series, which is conducted on the training dataset. The procedure begins by loading a pre-trained encoder-decoder pair (line 1), which defines a structured latent space for applying perturbations. The actor and critic networks are initialized along with a replay buffer, and a reward function f is defined to convert classifier predictions relative to the target label into a reward signal (lines 2–4).

At each training step (lines 5–12), an input time series x and target label $y _ { T }$ are sampled. The input is encoded into a latent representation z, and the actor generates a candidate counterfactual latent $z _ { C F }$ conditioned on $\left( z , y _ { M } , y _ { T } \right)$ Gaussian noise is added for exploration, and the perturbed latent $\tilde { z } _ { C F }$ is clipped before being decoded into a candidate $\tilde { x } _ { C F }$ . The black-box classifier M evaluates $\tilde { x } _ { C F }$ to produce a reward $R \ = \ f ( M ( \tilde { x } _ { C F } ) , y _ { T } )$ , and the interaction tuple $( z , y _ { M } , y _ { T } , \tilde { z } _ { C F } , R )$ is stored in the replay buffer.

Every U steps, the framework performs a gradient update (lines 13–19). First, a minibatch from the replay buffer is used to update the critic by minimizing the squared error between its predicted value and the observed reward, thereby improving its approximation of the reward function. Next, the actor is updated with a combined objective: a policy loss, which encourages the actor to maximize the critic's Q-value and thus generate valid counterfactuals, and a proximity loss, which penalizes large deviations from the original input. The overall actor loss $L _ { \mathrm { m a x } } + \lambda _ { p } L _ { \mathrm { p r o x } }$ balances validity with proximity.

Through repeated interactions with the black-box classifier and joint optimization of both actor and critic, the actor progressively learns a policy for generating counterfactuals that achieve the desired label change while remaining close to the original time series.

b) Inference Workflow.: As illustrated in Figure 2, after training, counterfactual generation is performed by applying the learned policy directly to each test instance. Given an input time series x, the encoder first maps it to a latent representation $z = \operatorname { e n c } ( x )$ , and the black-box classifier provides the original prediction $y _ { M } = M ( x )$ . Conditioned on $\left( z , y _ { M } , y _ { T } \right)$ , where yT is the desired target label, the trained actor outputs a counterfactual latent representation $z _ { C F } = \mu ( z , y _ { M } , y _ { T } )$ . The decoder then maps this representation back to the input space, producing the counterfactual sequence $x _ { C F } ~ = ~ \operatorname* { d e c } ( z _ { C F } )$ Since this process requires only one actor forward pass followed by decoding, inference avoids per-instance optimization and supports efficient batch generation of counterfactual ex-

L: sequence length.

Algorithm 1 Training procedure   
Inputs: M: black-box model; proximity loss weight $\lambda _ { p } ;$ total   
steps $T ;$ update interval $U ;$ batch size $| B | ;$ exploration std. $\sigma ;$   
mixing coefficient $\beta { = } 0 . 5$   
Outputs: Trained actor network $\mu$ for counterfactual genera  
tion   
Note: Each training step generates one counterfactual experi  
ence $( s , a , R )$ without sequential state transitions.   
1: Load pre-trained encoder enc and decoder dec   
2: Initialize actor $\mu ( \cdot ; \theta _ { \mu } )$ and critic $Q ( \cdot ; \theta _ { Q } )$   
3: Initialize replay buffer $\mathcal { D }$   
4: Define reward function $f ( \cdot , \cdot )$   
5: for t = 1 to $T$ do T training steps; each generates   
ONE counterfactual experience   
6: Sample input time series x and target class $y _ { T }$   
7: $y _ { M }  M ( x ) ; \quad z  e n c ( x )$   
8: $z _ { C F } \gets \mu ( z , y _ { M } , y _ { T } ; \theta _ { \mu } )$   
9: $\tilde { z } _ { C F } \gets \mathrm { c l i p } ( z _ { C F } + \varepsilon , - 1 , 1 ) , ~ \varepsilon \sim \mathcal { N } ( 0 , 0 . 1 )$   
10: $\tilde { x } _ { C F } \gets d e c ( \tilde { z } _ { C F } )$   
11: $R  f ( M ( \tilde { x } _ { C F } ) , y _ { T } )$ Immediate binary reward: 1   
if flip, 0 otherwise   
12: Store $( z , y _ { M } , y _ { T } , \tilde { z } _ { C F } , R )$ in the replay buffer D   
No next state—episode ends   
13: if t mod $U = 0$ then   
14: Sample batch $\boldsymbol { B }$ of size $B$ from D   
15: Update critic by one-step gradient descent using:   
$\nabla _ { \theta _ { Q } } \ \frac { 1 } { | B | } { \sum _ { B } \left( Q ( z , y _ { M } , y _ { T } , \tilde { z } _ { C F } ) - R \right) } ^ { 2 }$   
16: Recompute zCF $ \mu ( z , y _ { M } , y _ { T } ; \theta _ { \mu } )$ ,xCF ←   
dec(zcF)   
$\begin{array} { r } { L _ { \mathrm { m a x } } = - \frac { 1 } { | B | } \displaystyle \sum _ { B } Q ( z , y _ { M } , y _ { T } , z _ { C F } ) } \end{array}$   
$L _ { \mathrm { p r o x } } = \frac { 1 } { | B | } \sum _ { B } \left( \beta \| x - x _ { C F } \| _ { 1 } + \| x - x _ { C F } \| _ { 2 } ^ { 2 } \right)$ with   
17: Update actor by one-step gradient descent using:   
$\nabla _ { \theta _ { \mu } } \left( L _ { \mathrm { m a x } } + \lambda _ { p } L _ { \mathrm { p r o x } } \right)$   
18: end if   
19: end for

$$
\beta = 0 . 5 .
$$

planations.

## V. EXPERIMENTS

## A. Experimental Setup

Datasets. We evaluate our approach on a diverse collection of publicly available univariate time series datasets drawn from the University of California, Riverside (UCR) Time Series Classification Archive [21]. The UCR archive is a widely used benchmark suite for time series classification and has been extensively adopted to assess both traditional and deep learning models. To ensure that our findings are representative across different domains and levels of complexity, we select

TABLE I: Datasets Metadata
<table><tr><td rowspan=1 colspan=1>ID</td><td rowspan=1 colspan=1>Dataset Name</td><td rowspan=1 colspan=1>L</td><td rowspan=1 colspan=1>DS train size</td><td rowspan=1 colspan=1>DS test size</td><td rowspan=1 colspan=1>Type</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Chinatown</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>343</td><td rowspan=1 colspan=1>TRAFFIC</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Coffee</td><td rowspan=1 colspan=1>286</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>SPECTRO</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>ECG200</td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>ECG</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>FordA</td><td rowspan=1 colspan=1>500</td><td rowspan=1 colspan=1>3601</td><td rowspan=1 colspan=1>1320</td><td rowspan=1 colspan=1>SENSOR</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>FordB</td><td rowspan=1 colspan=1>500</td><td rowspan=1 colspan=1>3636</td><td rowspan=1 colspan=1>810</td><td rowspan=1 colspan=1>SENSOR</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>FreezerRegularTrain</td><td rowspan=1 colspan=1>301</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>2850</td><td rowspan=1 colspan=1>DEVICE</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>GunPoint</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>HAR</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>GunPointAgeSpan</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>135</td><td rowspan=1 colspan=1>316</td><td rowspan=1 colspan=1>HAR</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>GunPointMaleVersusFemale</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>135</td><td rowspan=1 colspan=1>316</td><td rowspan=1 colspan=1>HAR</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>GunPointOldVersusYoung</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>135</td><td rowspan=1 colspan=1>316</td><td rowspan=1 colspan=1>HAR</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>HandOutlines</td><td rowspan=1 colspan=1>2709</td><td rowspan=1 colspan=1>1000</td><td rowspan=1 colspan=1>370</td><td rowspan=1 colspan=1>IMAGE</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>TwoLeadECG</td><td rowspan=1 colspan=1>82</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>1139</td><td rowspan=1 colspan=1>ECG</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>Wafer</td><td rowspan=1 colspan=1>152</td><td rowspan=1 colspan=1>1000</td><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>SENSOR</td></tr></table>

13 datasets spanning application areas such as spectro, ECG, human activity recognition (HAR), sensor, traffic, etc. Table I provides detailed statistics for all datasets, including sequence length, number of classes, and train-test splits. These datasets have been shown to yield strong performance with state-of-theart deep classifiers [22], making them suitable for evaluating the quality of counterfactual explanations.

The chosen datasets also vary in size, covering scenarios from small-scale problems with fewer than 100 training instances to medium-scale collections with 100-250 training samples, and to larger-scale datasets containing thousands of examples. This diversity allows us to systematically investigate how our method scales with training set size. Importantly, the datasets are partitioned following the standard UCR setup, where training and testing sets are evenly sampled across classes to avoid class imbalance issues.

Baselines. We select four representative instance-based methods covering the main counterfactual-generation strategies.

Wachter [5] directly optimizes counterfactuals in the raw input space using validity and proximity losses, providing a basic comparison with optimization without policy learning.

TimeX [8] adds Dynamic Barycenter Averaging to perinstance optimization, representing methods with explicit temporal-plausibility constraints.

Info-CELS [15] learns a continuous saliency map to interpolate between an input and its nearest unlike neighbor, representing sparse and localized perturbations.

Glacier-AE [11] performs per-instance counterfactual optimization in an autoencoder latent space without learning a generation policy, allowing comparison with our latent-space policy-learning framework.

Evaluation Metrics. Following the evaluation practices commonly adopted in the time-series counterfactual explanation literature [23], we assess different methods along three standard metrics: (1) Validity Metric (Flip Label Rate): This metric measures how often the generated counterfactuals successfully change the model's original prediction. Formally, it is defined as:

$$
f l i p \_ r a t e = \frac { n u m \_ f l i p p e d } { n u m \ t e s t s a m p l e } ,\tag{8}
$$

where num\_flipped is the number of counterfactuals that flip the predicted label, and num\_testsample is the total number of test inputs. A higher flip rate (closer to 1) indicates better validity, as more counterfactuals satisfy the label change requirement.

(2) Proximity Metrics $( L _ { 1 } , \ L _ { 2 } ,$ and $L _ { \infty } ) .$ Proximity evaluates how close each counterfactual is to the original input. We report three common distance-based metrics: (a) the $L _ { 1 }$ Distance (Manhattan distance) measures the total absolute change across all time steps; (b) the $L _ { 2 }$ Distance (Euclidean distance): Emphasizes larger differences by squaring deviations; (c) the $L _ { \infty }$ Distance (Chebyshev distance): Captures the maximum absolute deviation.

(2) Plausibility Metrics. (IF, LOF, OC\_SVM) We evaluate plausibility by comparing the generated counterfactuals against the training dataset and computing how “out-of-distribution" they appear under three standard unsupervised anomaly detectors: (a) Isolation Forest (IF) [24]; (b) Local Outlier Factor (LOF) [25]; (c) One-Class SVM (OC\_SVM) [26]. Each detector is trained only on the original training data, which defines the normal data manifold. We then feed the generated counterfactuals to these detectors and measure their outlier scores. Counterfactuals that receive lower outlier scores are considered more plausible, since they lie closer to the distribution of real time series in the training set.

## B. Implementation Details

We train the framework for 50,000 steps using datasetspecific batch sizes ranging from 16 to 256 and set $\lambda _ { p } = 1$ For each dataset, we evaluate autoencoders with latent dimensions of {8, 16, 32, 64} and select the model with the lowest reconstruction error. The actor and critic are implemented as multilayer perceptrons with two 256-unit hidden layers, layer normalization, and ReLU activations. The actor uses a tanh output layer to generate bounded latent representations, while the critic outputs a scalar value. The critic minimizes the squared error between predicted values and observed rewards, whereas the actor maximizes the critic output with proximity regularization. Gaussian noise is added to the actor output for exploration. The black-box classifier and pre-trained autoencoder remain fixed during policy training. Training metrics and intermediate samples are logged using Weights & Biases every 100 steps. The source code provides the complete architecture, dataset-specific parameters, and training configurations.¹

## C. Experimental Results

Validity (Flip Rate): Figure 3 compares the validity of counterfactuals across methods, measured by flip rate. The violin plot shows the distribution of flip rates across datasets. (The black horizontal line and red dot represent the median and mean across datasets, respectively. This is kept consistent across all violin plots in the paper.) Our policy learning framework consistently achieves strong performance, with validity concentrated in the 0.8–1.0 range and a high mean and median (0.95), reflecting reliable performance across datasets. This indicates that nearly all generated counterfactuals successfully flip the classifier's decision, a critical property for actionable explanations. TimeX and InfoCELS often reach near-perfect validity (≈1.0), but their performance is less reliable, dropping below 0.7 on certain datasets. Glacier is even more fragile: while it can achieve high validity on some datasets, it collapses to zero on others. These fluctuations reduce the overall robustness of the baselines. By contrast, our proposed method maintains a compact distribution, whereas the wider spread of InfoCELS, TimeX, and Glacier reveals susceptibility to dataset-specific failures. Taken together, these results highlight that our proposed method provides a stable and generalizable mechanism for generating valid counterfactuals in diverse time series domains.

![](images/66c5ca4c11b888858a493e4d9228e3f8b82f744d5506ec2be8db5bb103feb943.jpg)  
Fig. 3: Validity evaluation measured by flip rate, where higher values indicate more valid counterfactuals.

Proximity (L1, L2, $L _ { \infty } ) { \ : } :$ Figure 4 reports the proximity of generated counterfactuals, where lower values indicate smaller perturbations from the original inputs. For L1 and L2 distances, InfoCELS, TimeX, and Wachter achieve the lowest values. InfoCELS attains the best proximity by perturbing only the most salient time steps, while TimeX and Wachter maintain relatively low distances through explicit distancebased penalties in their optimization objectives. By contrast, our method and Glacier operate in latent space rather than directly perturbing the input, which leads to higher L1 and L2 values after decoding, as modifications are distributed more broadly across the sequence. Although our method employs an elastic distance penalty within the actor network to constrain perturbations, the decoding process through the autoencoder inevitably spreads changes across multiple time steps, yielding larger aggregate distances.

For $L _ { \infty } .$ which captures the largest pointwise deviation, our method ranks behind Wachter but demonstrates competitive performance with TimeX and InfoCELS, while clearly outperforming Glacier. This suggests that although our method produces broader perturbations—reflected in its higher $L _ { 1 }$ and $L _ { 2 }$ distances—it effectively avoids extreme spikes at individual time steps, leading to smoother and more controlled modifications. This behavior stems from the method's design: the autoencoder embedding promotes globally distributed adjustments rather than highly localized changes. As a result, our method yields counterfactuals that are less proximate in aggregate distance but remain better aligned with the underlying data manifold.

![](images/4af58d436099301166e7ebd5bb3faa62e4ac4e88f124fbb392dee4f0561f5f97.jpg)

![](images/1d71f31a4f73cbfbf24b6328ddf136082d1644e7d5b6229796c59746ea12133c.jpg)

![](images/dd4f2e642f744dc8d68090592df1858add856a57ce167eda9e9125a90ccea889.jpg)  
Fig. 4: Proximity evaluation using three distance metrics. Lower values indicate counterfactuals closer to the original inputs.

![](images/fb60c28cca9caf43724c0a59e1b8bd16a5b060ab91a49c27e9cdf9b19819563e.jpg)

![](images/4feed7882dc9340be3f5db9c1a8d7ac2687471b7d570c3676d29e4ed70bdf786.jpg)

![](images/a4aea9992ff0c5a312ec9d638cbba339c8724ea246f4f4dd5bae6ea11f529bc8.jpg)  
Fig. 5: Plausibility evaluation using three outlier detection methods. Lower scores indicate more realistic counterfactuals.

Plausibility (IF, LOF, OC\_SVM): Figure 5 shows that our method maintains consistently low outlier scores across all three detectors. Under IF, RL remains within the 0–0.35 range, with both mean and median around 0.15, while the baselines span the full 0-1 interval with higher variance and frequent outliers. For LOF, our method stays below 0.3 across all datasets, achieving the lowest mean and median values (close to 0) and performing competitively with Wachter, TimeX, and InfoCELS, whereas Glacier again reaches up to 1.0. Under OC\_SVM, our method clearly outperforms all baselines, keeping scores consistently below 0.1 with mean and median near zero, while Wachter and TimeX extend toward 0.4 and Glacier fluctuates widely between 0–1.

These results demonstrate that the counterfactuals generated by our method align more closely with the data manifold, benefiting from latent-space policy learning that encourages globally coherent perturbations rather than localized and potentially unrealistic changes. By operating in the latent space, our method captures structural regularities of the data, which translates into counterfactuals that are not only valid but also plausible under multiple outlier-detection metrics.

Taken together with the proximity results in Figure 4, these findings reveal a clear validity-proximity-plausibility tradeoff. Input-space baselines such as Wachter, TimeX, and Info-CELS can achieve smaller aggregate $L _ { 1 }$ and $L _ { 2 }$ distances by making sparse or localized modifications, but these lower distances do not necessarily imply higher realism, as the resulting counterfactuals may still deviate from the data distribution. In contrast, our latent-space policy tends to produce broader perturbations after decoding, which can increase aggregate normbased distances but helps preserve global temporal coherence and manifold consistency. Overall, the evidence highlights our proposed policy learning framework as a method that balances validity, proximity, and plausibility: it may sacrifice some raw input-space proximity, but it produces counterfactuals that are more stable, distributionally plausible, and effective for model explanation.

## D. Efficiency Analysis and Running-Time Comparison

To compare the computational efficiency of different counterfactual generation approaches, we analyze the runtime of our proposed framework relative to mainstream instancebased optimization baseline methods. A key distinction is that baseline methods do not include a separate “training" phase for counterfactual generation—they optimize each test instance independently—whereas our proposed approach consists of both a training stage (policy learning) and a fast inference stage (single forward pass). To provide a fair comparison, we report:

• Training time (RL only),

• Inference time (all methods), and • Total end-to-end time (RL training + RL inference vs. inference-only baselines).

(c) InfoCELS running time  
![](images/2a98858190d3d016b991c204cd78532b88430390e64b8e881c62657ea5f9c224.jpg)  
(a) Dataset sizes across all UCR datasets

![](images/642bd187332479a787117a5a8e701582cffac4a2ff5b32922b36f9b591659350.jpg)  
(b) RL-CF (Ours) running time

![](images/5b8706a346e699ac1ef808178f738b2c148419e57afc12b46177fd9131cf00e9.jpg)

![](images/97597a0e58388f189b45d4046e4ce7b331850f6765d4511e7ab43be908856f94.jpg)  
(d) Glacier running time Total Runtime per Dataset

![](images/b4e2b40fb2bd24ba55c178b1560c6fa5b3ce73d1437be88bcbce154c77bb0c48.jpg)  
(e) Wachter running time

![](images/e7fbdc4ffd35ebc1b7091f0536733ad8c85b138b1dec199d0527b58057e0b53f.jpg)  
(f) TimeX running time  
Fig. 6: Running-time comparison of our proposed method and baseline counterfactual explanation methods across all UCR datasets. Subfigure (a) shows the dataset sizes. Subfigure (b) presents the running time of our proposed method. Subfigures (c)–(f) report the runtime of InfoCELS, Glacier, Wachter, and TimeX, respectively. All results were obtained using the same hardware environment.

![](images/81bff057bfd1ebc660a43ede6c6ec9249d3ab20844069d84a9adbe2a8cefb9fc.jpg)  
Fig. 7: Ablation study comparing the full RL method with experience replay against the variant without replay.

Figure 6 presents the runtime comparison across datasets of increasing size (ordered from smallest to largest). Sorting datasets by size makes it easier to observe how methods scale as the number of test samples grows. Because all instancebased methods require iterative optimization for every input, their runtime increases roughly linearly with dataset size. In contrast, the RL method is trained once on the dataset, and its training cost is independent of the number of test instances. During inference, each counterfactual is produced by a single actor forward pass, which takes only milliseconds, so the overall inference time grows linearly but very slowly with the size of the test set.

a) Small Datasets.: On small datasets (e.g., Coffee, ECG200,GunPoint), instance-based methods such as Wachter, TimeX, InfoCELS, and Glacier have low total runtime simply because there are very few test samples. In these cases, the one-time training cost of our method becomes the main part of the total runtime, so the baselines may look faster. This is expected, since the benefit of training an RL policy only shows up when the test set is larger.

b) Medium-Sized Datasets.: As we move to mediumsized datasets, the difference becomes more pronounced. Instance-based methods require optimization loops for an increasing number of samples, which causes their inference time to grow rapidly. In contrast, our method is trained once and its training cost stays within a reasonable range for all datasets. During inference, our method generates counterfactuals with a single forward pass, so inference time only grows linearly in the number of test instances and stays within seconds. As a result, the total runtime of our method becomes competitive with—and often lower than—the baseline methods on medium-sized datasets.

c) Large Datasets.: For the largest datasets (e.g., FordA, Wafer, FreezerRegularTrain), the scalability benefits of Our method becomes most significant. Baseline methods must run thousands of optimization loops, causing their runtime to increase sharply. Wachter, TimeX, InfoCELS, and Glacier all show substantial slowdowns because each counterfactual requires repeated gradient-based updates on every instance. In contrast, our method introduces no per-instance optimization: once trained, generating a counterfactual only requires an actor forward pass followed by decoding. Although RL inference still grows linearly with the number of test samples, each forward pass takes only milliseconds, making the overall inference cost very small. As a result, our method achieves the lowest total runtime among all approaches on large datasets. These results highlight that our method is especially advantageous for large-scale counterfactual generation scenarios.

## E. Ablation Study

To evaluate the contribution of experience replay in our counterfactual generation framework, we conduct an ablation study comparing the full method (RL agent with the replay buffer) with a variant that removes the replay buffer (RL w/o Replay). The goal of this ablation is to examine how replay affects the performance of generated counterfactuals across three key metrics: validity, proximity, and plausibility.

![](images/737a7dc582e9cff44151dd9de427f848b35735962889df27bfd0cce102337a6d.jpg)  
Fig. 8: Qualitative Case Studies on the GunPoint Dataset

From Figure 7, we observe that experience replay contributes differently across the three evaluation metrics, with varying levels of impact on validity, proximity, and plausibility. Importantly, both proximity and plausibility are computed only on valid counterfactuals, ensuring that these results directly reflect the quality of CFs that successfully flip the label.

For validity, replay helps produce a tighter and higher fliprate range of [0.8, 1.0], whereas the no-replay variant has a wider range of [0.5, 0.99]. Although the median validity without replay is slightly higher and the mean slightly lower than with replay, this indicates that not all datasets require replay to maintain high validity of counterfactuals generated; some datasets could generate high validity counterfactuals even without it. For proximity, all three metrics $( L _ { 1 } , \ L _ { 2 } ,$ and $L _ { \infty } )$ show only minor differences between the replay and no-replay variants. The values are generally comparable across both settings, with no large or systematic shift. The replay variant exhibits slightly lower mean and median values across datasets, but the overall effect is modest. This indicates that replay does not strongly influence how close the counterfactuals remain to the original inputs. The largest performance difference appears in the plausibility metrics (IF, LOF, and OC-SVM). Even though the no-replay variant produces valid counterfactuals (i.e., they successfully flip the label), these valid CFs exhibit noticeably higher OOD scores and substantially larger variance, indicating that the perturbations are more likely to deviate from the data manifold. In comparison, the replay variant consistently produces lower and more stable plausibility values. These results collectively show that experience replay plays an important role in generating valid counterfactuals that are also plausible and manifoldconsistent.

## F. Case Study: Qualitative Comparison of Counterfactual Explanations

To complement the quantitative evaluation, we conduct qualitative case studies on the UCR GunPoint dataset—to visually assess the temporal coherence and structural plausibility of counterfactuals generated by different methods. In particular, we plot all valid counterfactuals under each baseline and compare them with our proposed RL-generated counterfactuals. The corresponding visualizations are shown in Figure 8).

As we can see, even instances from the same class exhibit substantial variation in magnitude and slope, making counterfactual generation more challenging. Most baselines, especially Wachter, TimeX, and Glacier, tend to introduce abrupt local distortions, irregular fluctuations, and unstable peak shapes. In contrast, our method better preserves the global rise-fall motion pattern and produces comparatively smoother and more coherent counterfactuals. Although minor irregular distortions still remain, partly due to the AE-based backbone, they are less severe than those observed in the baseline methods, leading to improved global shape consistency under the dataset's inherent variability.

## VI. CONCLUSION

In this paper, we proposed a scalable policy learning framework for counterfactual explanation in time series classification. The key idea is to move counterfactual generation from per-instance test-time optimization to an experience-driven learning process, where an actor learns a transferable policy for producing counterfactual latent representations. We formulate this process as a one-step reinforcement learning problem and operate in the latent space of an autoencoder, enabling the model to generate globally coherent perturbations while avoiding direct optimization over raw time steps. Across 13 UCR benchmark datasets, our method achieves high validity and strong plausibility under multiple anomaly-detection-based metrics. The runtime analysis further demonstrates that, once trained, the learned policy generates counterfactuals with only a single forward pass, making it particularly effective for largescale counterfactual generation. Overall, our findings highlight policy learning as a promising direction for efficient and reusable time series explanation. Future work will investigate more expressive reward functions, tighter temporal constraints, and broader evaluation on multivariate time series domains.

## REFERENCES

[1] H. W. Loh, C. P. Ooi, S. Seoni, P. D. Barua, F. Molinari, and U. R. Acharya, “Application of explainable artificial intelligence for healthcare: A systematic review of the last decade (2011-2022)," Computer methods and programs in biomedicine, vol. 226, p. 107161, 2022.

[2] S. Bharati, M. R. H. Mondal, and P. Podder, "A review on explainable artificial intelligence for healthcare: Why, how, and when?" IEEE Transactions on Artificial Intelligence, vol. 5, no. 4, pp. 1429–1442, 2023.

[3] J. Černevičienė and A. Kabašinskas, “Explainable artificial intelligence (xai) in finance: a systematic literature review,"Artificial Intelligence Review, vol. 57, no. 8, p. 216, 2024.

[4] R. Dwivedi, D. Dave, H. Naik, S. Singhal, R. Omer, P. Patel, B. Qian, Z. Wen, T. Shah, G. Morgan et al., "Explainable ai (xai): Core ideas, techniques, and solutions," ACM Computing Surveys, vol. 55, no. 9, pp. 1–33, 2023.

[5] S. Wachter, B. Mittelstadt, and C. Russell, "Counterfactual explanations without opening the black box: Automated decisions and the gdpr, Harv. JL & Tech., vol. 31, p. 841, 2017.

[6] E. Delaney, D. Greene, and M. T. Keane, "Instance-based counterfactual explanations for time series classification," in International Conference on Case-Based Reasoning. Springer, 2021, pp. 32–47.

[7] P. Li, S. F. Boubrahimi, and S. M. Hamd, "Motif-guided time series counterfactual explanations," arXiv preprint arXiv:2211.04411, 2022.

[8] S. Filali Boubrahimi and S. M. Hamdi, "On the mining of time series data counterfactual explanations using barycenters," in Proceedings of the 31st ACM International Conference on Information & Knowledge Management, 2022, pp. 3943–3947.

[9] P. Li, O. Bahri, S. F. Boubrahimi, and S. M. Hamdi, "Sg-cf: Shapeletguided counterfactual explanation for time series classification," in 2022 IEEE International Conference on Big Data (Big Data). IEEE, 2022, pp. 1564–1569.

[10] P. Hosseinzadeh, P. Li, O. Bahri, S. F. Boubrahimi, and S. M. Hamdi, “Acts: Adaptive counterfactual explanations for time series data using barycenters," in 2024 IEEE International Conference on Big Data (BigData). IEEE, 2024, pp. 1327–1332.

[11] Z. Wang, I. Samsten, I. Miliou, R. Mochaourab, and P. Papapetrou, “Glacier: guided locally constrained counterfactual explanations for time series classification," Machine Learning, pp. 1–31, 2024.

[12] O. Bahri, S. F. Boubrahimi, and S. M. Hamdi, "Shapelet-based counterfactual explanations for multivariate time series," arXiv preprint arXiv:2208.10462, 2022.

[13] O. Bahri, P. Li, S. F. Boubrahimi, and S. M. Hamdi, "Temporal rule-based counterfactual explanations for multivariate time series," in 2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA). IEEE, 2022, pp. 1244–1249.

[14] P. Li, O. Bahri, S. F. Boubrahimi, and S. M. Hamdi, "Cels: Counterfactual explanations for time series data via learned saliency maps," in 2023 IEEE International Conference on Big Data (BigData). IEEE, 2023, pp. 718–727.

[15] P. Li, O. Bahri, P. Hosseinzadeh, S. F. Boubrahimi, and S. M. Hamdi, “Info-cels: Informative saliency map-guided counterfactual explanation for time series classification," Electronics, vol. 14, no. 7, p. 1311, 2025.

[16] T. Sivill and P. Flach, “Limesegment: Meaningful, realistic time series explanations," in International Conference on Artificial Intelligence and Statistics. PMLR, 2022, pp. 3418–3433.

[17] R.-F. Samoilescu, A. Van Looveren, and J. Klaise, "Model-agnostic and scalable counterfactual explanations via reinforcement learning," arXiv preprint arXiv:2106.02597, 2021.

[18] Z. Chen, F. Silvestri, J. Wang, H. Zhu, H. Ahn, and G. Tolomei, "Relax: Reinforcement learning agent explainer for arbitrary predictive models," in Proceedings of the 31st ACM international conference on information & knowledge management, 2022, pp. 252–261.

[19] T. M. Nguyen, T. P. Quinn, T. Nguyen, and T. Tran, "Counterfactual explanation with multi-agent reinforcement learning for drug target prediction," arXiv preprint arXiv:2103.12983, 2021.

[20] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, “Continuous control with deep reinforcement learning," arXiv preprint arXiv:1509.02971, 2015.

[21] H. A. Dau, A. Bagnall, K. Kamgar, C.-C. M. Yeh, Y. Zhu, S. Gharghabi, C. A. Ratanamahatana, and E. Keogh, “The ucr time series archive," IEEE/CAA Journal of Automatica Sinica, vol. 6, no. 6, pp. 1293–1305, 2019.

[22] H. Ismail Fawaz, G. Forestier, J. Weber, L. Idoumghar, and P.-A. Muller, “Deep learning for time series classification: a review," Data mining and knowledge discovery, vol. 33, no. 4, pp. 917–963, 2019.

[23] P. Li, O. Bahri, S. F. Boubrahimi, and S. M. Hamdi, "Counterfactual explanation bake-off: A review and experimental evaluation for time series classification," Machine Learning, vol. 115, no. 5, p. 114, 2026.

[24] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in 2008 eighth ieee international conference on data mining. IEEE, 2008, pp. 413–422.

[25] M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, "Lof: identifying density-based local outliers," in Proceedings of the 2000 ACM SIGMOD international conference on Management of data, 2000, pp. 93–104.

[26] B. Schölkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, and R. C. Williamson, “Estimating the support of a high-dimensional distribution," Neural computation, vol. 13, no. 7, pp. 1443–1471, 2001.