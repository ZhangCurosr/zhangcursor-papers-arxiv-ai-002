# LUCID DREAMING FOR WORLD MODELS: LEARNING TODOUBT IMAGINATION AND DECIDE BY TRUST

Ziqi Wen<sup>1,2</sup> Ting Xu<sup>1,2</sup> Lianyu Wang<sup>1,2</sup> Xian Lin<sup>1,2</sup> Yanda Meng<sup>3</sup> Huazhu Fu<sup>4</sup> Meng Wang<sup>1,2,B</sup> Ching-Yu Cheng<sup>1,2,5,6</sup>

<sup>1</sup>Centre for Innovation and Precision Eye Health, Yong Loo Lin School of Medicine, National University of Singapore, Singapore 119228, Singapore <sup>2</sup>Department of Ophthalmology, Yong Loo Lin School of Medicine, National University of Singapore, Singapore 119228, Singapore <sup>3</sup>Bioengineering Program, Biomedical Sciences Division (BioMed), King Abdullah University of Science and Technology (KAUST), Thuwal, Saudi Arabia <sup>4</sup>Institute of Advanced Intelligence and Computing (IAIC), Agency for Science, Technology and Research (A\*STAR), Singapore 138632, Singapore <sup>5</sup>Singapore Eye Research Institute, Singapore National Eye Centre, Singapore 169856, Singapore <sup>6</sup>Ophthalmology & Visual Sciences Academic Clinical Program (EYE ACP), Duke-NUS Medical School, Singapore 169856, Singapore

![](images/aa88023b64ad3e544ec69f32d359cd2f05a46ac26f0338322de88351f27057de.jpg)  
Figure 1: Our world model with doubt. The doubt is read from the same head that generates its imagination, with no added parameter. The dark line takes one real fall as an example: the model reports its doubt eight steps before the fall and stays above its alarm line throughout it. We show that it reports all twenty consecutive real falls in the same way, every one before the fall begins. The usual readout, the entropy of the prediction, reports none of them.

## ABSTRACT

World models enable agents to learn and plan in imagination, but predictions beyond their experience can become unreliable and mislead decisions. Existing uncertainty estimates derived from predictions can remain overconfident on unfamiliar state-action pairs. We propose the Lucid World Model (LucidWM), which learns doubt from experience and propagates trust through imagination. By integrating Subjective Logic into categorical latent transitions, LucidWM distinguishes predicted outcomes from their evidential support and assigns each transition a degree of doubt. The complement of this doubt defines transition-level trust, which accumulates multiplicatively along imagined trajectories to reweight returns for policy learning and guide action selection. Uncertainty estimation requires no additional parameters or forward passes. Evaluated on four base world models against seventeen uncertainty readouts, LucidWM detects environmental changes and signals uncertainty during action-corrupted rollouts. In a controlled navigation case study, acting on trust reduces the number of steps required to reach the goal from 362 to 190. Fifteen demonstration videos show how LucidWM doubts its dreams and acts on that doubt. Videos are available at https://lucidwm.github.io.

## 1 Introduction

A world model is learned from experience and predicts how the world responds to an agent’s actions. With it, the agent can dream: it feeds each predicted state back into the model and imagines future trajectories without further interaction with the environment (Hafner et al., 2025). These trajectories support behaviour learning and planning. Such imagination is especially valuable when real trials are costly (Wu et al., 2022), unsafe (Gao et al., 2024) or impossible to repeat under identical conditions (Xu et al., 2026). Yet it is often needed to evaluate actions and states beyond the agent’s experience, precisely where its world model is least reliable (Yu et al., 2020; Kidambi et al., 2020). Without sufficient experiential support, the model must extrapolate, and prediction errors can compound as each imagined state becomes the input to the next transition (Janner et al., 2019; Asadi et al., 2018).

An unreliable dream is dangerous when believed. The agent may learn behaviours that exploit model errors and plan towards states that are implausible in the real environment (Kurutach et al., 2018). Existing methods use uncertainty estimates to propagate predictive uncertainty or penalise unreliable model-generated transitions (Chua et al., 2018; Yu et al., 2020). Common signals include the entropy or variance of a predictive distribution and disagreement among ensemble members (Sekar et al., 2020). However, a sharp prediction or agreement among models does not, by itself, establish that a queried transition is supported by experience. In our evaluations, these signals can remain insensitive to unfamiliar inputs or even become more confident as prediction reliability deteriorates (Fig. 1, Table 1).

We trace these failures to two distinct limitations. (i) Doubt concerns support for the input, not just the shape of the output. A state and an action may each be familiar while their combination is unsupported by observed transitions. The model can still produce a sharp prediction (Hein et al., 2019), so a readout of predictive spread may report confidence despite limited evidence for that transition (Fig. 2). (ii) Trust depends on the rollout, not just the current step. Each imagined state inherits uncertainty from the transitions that produced it. A prediction should therefore be trusted only to the extent that its preceding trajectory is supported, back to the observed starting state. A pointwise uncertainty score alone does not capture this dependence (Berger et al., 2026).

We propose to learn uncertainty from the evidence that experience provides for a transition, rather than infer it solely from the resulting prediction. The world model can then dream lucidly: just as a lucid dreamer knows a dream from waking life, it distinguishes what it pre-

![](images/06d7687acd22445360f1c826455f41d281751f986d1a968bfda6bccfa2f7fe6d.jpg)  
Figure 2: Same prediction, different experience. Top: a scene and an action never experienced together are predicted as sharply as a pair that was (real frames: at a fall’s onset, and 30 steps before it). Bottom: ours reads the Dirichlet behind the input, whose strength and doubt separate the two; the entropy of the output cannot. Bars: each readout over its own alarm line.

dicts from the evidence supporting that prediction. We use the existing transition-head logits to parameterise a Dirichlet distribution over categorical next-state probabilities. Its mean gives the prediction, its concentration above the prior represents learned evidential strength, and the prior’s share of the total concentration defines the step’s doubt (Jøsang, 2016). Our training objective ties evidence to observed transitions and discourages unsupported evidence. During imagination, we accumulate the complements of these doubts multiplicatively into trajectory trust. This trust governs how strongly the agent relies on imagined continuations when learning and deciding.

We realise this as the Lucid World Model (LucidWM), which integrates doubt and trust into three stages (Fig. 3). (1) Learning doubt from experience. We formulate the categorical transition head as a Subjective Logic (SL) opinion, with evidence measured above the Dirichlet prior, and update the posterior by fusing evidence from observation. (2) Carrying trust through imagination. We incorporate trust into the standard λ-return, reducing reliance on uncertain imagined continuations. Setting trust to one recovers the standard return. (3) Deciding by trust. Before acting, we evaluate the doubt associated with candidate actions and score their imagined futures using trust. LucidWM is designed for world models with categorical latent states and requires no additional learnable parameters or forward passes for uncertainty estimation.

![](images/661b153e6d1d4455bda8cebf3abc5bfd0f7d31f336f64224c4b9727d55ca64c0.jpg)

![](images/caeaef3df42d5473c48072b5799112c576c209c63f51b5a56eb8de151169a080.jpg)  
Figure 3: Doubt is learned from experience and carried as trust. (a) A standard world model; (b) LucidWM, changes in grey and blue, doubt in orange. 1) The head’s evidence e yields a doubt u in place of the fixed $\varepsilon ;$ the discipline $\mathcal { L } _ { u }$ lets evidence recede toward the vacuous opinion where the dynamics loss does not hold it up. 2) Doubt sets each imagined step’s trust $\tau _ { k } ,$ compounded into $C _ { k } ; R ^ { \tau }$ keeps $\lambda \tau _ { k }$ of each step where $R ^ { \lambda }$ keeps λ. 3) A course whose trust collapses is vetoed.

Our contributions are threefold. (i) We identify how common predictive readouts can miss unsupported transitions and fail to capture cumulative uncertainty along imagined trajectories. (ii) We propose LucidWM, integrating evidential doubt and cumulative trust into world model learning, imagination and decision. (iii) Across four world model backbones and seventeen uncertainty readouts, our results demonstrate improved detection of environment changes after observation and rollout drift before decision making. In the reported fall and goal-reaching experiments, LucidWM raises an alarm before each of 20 evaluated falls and reaches the goal in 190 steps, compared with 362 for the base model.

## 2 Preliminaries and Related Work

World models. A world model (Ha & Schmidhuber, 2018; Hafner et al., 2019) has two parts, an encoder that compresses each observation $x _ { t }$ into a latent state $s _ { t } .$ , and a transition that predicts the next state from $s _ { t }$ and the action $a _ { t } .$ . Most recent models make the latent discrete, whether the transition is recurrent or a transformer (Hafner et al., 2025; Morihira et al., 2026; Burchi & Timofte, 2025; Zhang et al., 2026). Its stochastic part is G categorical variables of K classes, and the transition predicts each, $z _ { t + 1 }$ , with a categorical head of K logits $\ell ( s _ { t } , a _ { t } )$ , a softmax with a small uniform mass mixed in so that no class is ever ruled out (Hafner et al., 2025):

$$
p ( \boldsymbol { z } _ { t + 1 } \mid \boldsymbol { s } _ { t } , \boldsymbol { a } _ { t } ) = ( 1 - \varepsilon ) \mathrm { s o f t m a x } \big ( \ell ( \boldsymbol { s } _ { t } , \boldsymbol { a } _ { t } ) \big ) + \varepsilon / K ,\tag{1}
$$

where ε is a fixed constant, the same at every state. In training, a second head, the posterior $q ( z _ { t + 1 } \mid$ $s _ { t } , a _ { t } , x _ { t + 1 } )$ , reads the observation that actually arrived, and the transition head is trained to match it. Imagination samples from $p$ for H steps, feeding the sampled state back as the next input, with a reward head supplying $r _ { t } ;$ an actor π and a critic $v ,$ which estimates the value of a state, are then trained on the imagined trajectory with the λ-return (Hafner et al., 2025; Micheli et al., 2023):

$$
{ \cal R } _ { t } ^ { \lambda } \ = \ r _ { t } + \gamma \big [ ( 1 - \lambda ) v ( s _ { t + 1 } ) + \lambda { \cal R } _ { t + 1 } ^ { \lambda } \big ] , \qquad { \cal R } _ { H } ^ { \lambda } = v ( s _ { H } ) ,\tag{2}
$$

where $\gamma$ is the discount and the fixed λ sets, at every step, how far the return continues into the imagined future rather than falling back on the critic (we omit Dreamer $\mathbf { V } 3 ^ { \circ } \mathbf { s }$ continuation flag).

Subjective Logic. SL (Jøsang, 2016) extends probability by recording how much evidence a belief rests on, so that not knowing is told apart from knowing that an outcome is unlikely. A belief over $K$ outcomes, an opinion,

is a Dirichlet over the outcome probabilities whose concentration adds the evidence $e _ { i } \geq 0$ observed for each outcome i to a prior of weight W spread by a base rate, uniform throughout this paper. With $S = \sum _ { i } e _ { i }$ the total evidence and $\hat { p } = e / \cal S$ its direction, the concentration, the uncertainty mass and the mean are:

$$
\alpha _ { i } = e _ { i } + W / K , \qquad u = W / ( W + S ) , \qquad P _ { i } = \alpha _ { i } / ( W + S ) = ( 1 - u ) \hat { p } _ { i } + u / K .\tag{3}
$$

The mean P is the probability the opinion offers to a decision, and the uncertainty mass $u ,$ the prior’s share of the Dirichlet’s strength $W + { \dot { S } }$ , is one before any evidence and falls as evidence accumulates. We observe that P has exactly the form of Eq. (1), with the fixed ε replaced by u.

Uncertainty in world models. Current world models draw their uncertainty from three sources. The first is the prediction of a single model, read as the entropy or variance of the predicted distribution, its top probability, or the likelihood it gave to the observation that then arrived (Malik et al., 2019; Savov et al., 2025). The second is a head fitted on the model’s frozen features, scoring a state by the distance of its features from those seen in training (Lee et al., 2018) or by how poorly a trained network predicts a fixed random network’s output on them (Burda et al., 2019); it sees the state, not the action. The third is disagreement among several models trained on the same data (Lakshminarayanan et al., 2017), used as an exploration reward in Plan2Explore (Sekar et al., 2020), as a signal to halt imagination in MOReL (Kidambi et al., 2020), and to weight, mask or truncate imagined rollouts (Buckman et al., 2018; Pan et al., 2020; Frauenknecht et al., 2024). All three are read off what the model produces. We read uncertainty from the evidence behind each imagined step instead. An evidential head reads such evidence for a single prediction (Sensoy et al., 2018; Amini et al., 2020); a world model’s prediction is its next input, so its doubt must be carried along the rollout, fused with the observation that follows, and allowed to change what the agent learns.

## 3 Method

A standard world model pins two constants: its head, Eq. (1), mixes the same ε into every prediction, and its return, Eq. (2), keeps the same share λ of every imagined step. We unfold both into quantities read from the head’s own logits, the doubt of a transition and the trust in a step, and follow them through learning (Sec. 3.1), imagination (Sec. 3.2) and decision (Sec. 3.3), as Fig. 3 summarises.

## 3.1 Learning doubt from experience

Reading doubt from the transition head. The transition head already computes what we need: its logits, read as the evidence of an opinion. For each categorical variable of the latent, the head’s K logits are read as e = softplus $\ell ( s _ { t } , a _ { t } )$ , a nonnegative amount of evidence per class. By Eq. (3), their total S is the experience the head brings to the scene and action it is asked about, $\hat { p } = e / S$ is its direction, and $u = \bar { \operatorname* { m a x } } \{ W / ( W + S ) , \varepsilon \}$ is the doubt of the transition, the share of the prediction that no experience supports, floored at the standard head’s ε so that the lucid head never trusts a transition more than the standard head does. The prediction is the direction of the evidence; the doubt is its amount. Our head spreads that share uniformly over the classes:

$$
\mathrm { S T A N D A R D ~ H E A D } p ( z _ { t + 1 } \mid s _ { t } , a _ { t } ) = ( 1 - \varepsilon ) \mathrm { s o f t m a x } \big ( \ell ( s _ { t } , a _ { t } ) \big ) + \varepsilon / K ,\tag{1}
$$

$$
\begin{array} { r l } { \mathrm { O U R \ L U C I D \ H E A D } p ( z _ { t + 1 } \mid s _ { t } , a _ { t } ) = \left( 1 - u ( s _ { t } , a _ { t } ) \right) \widehat { p } ( z _ { t + 1 } \mid s _ { t } , a _ { t } ) + u ( s _ { t } , a _ { t } ) / K . } \end{array}\tag{4}
$$

Proposition 1 (Standard heads are constant-doubt opinions). Any head of the form Eq. (1) is an instance of Eq. (4) with direction softmax(ℓ) and doubt pinned at $u \equiv \varepsilon ;$ equivalently, through Eq. (3), it asserts the same evidence total $S \equiv W ( 1 \overrightharpoon { 1 - \varepsilon } ) / \varepsilon$ for every transition. For example, with Dreamer $\mathrm { V } 3 ^ { \circ } \mathrm { s } \varepsilon = 0 . 0 1$ and $W = 2 , S = 1 9 8 ( \mathrm { A p p . C . 1 } )$

A standard world model thus assigns every transition, including those it has never experienced, the same amount of evidence. Ours lets the count vary with the transition: prediction and doubt become separate quantities, $\hat { p }$ can be peaked while S is small, and only S sets u. The head predicts as before and reports its doubt at every imagined step from the same forward pass. Read alone, S is only a name for a total; what makes it track experience is how the head is trained, to which we turn next.

Binding evidence to experience. Evidence should grow with experience and with nothing else. The dynamics loss, which trains the transition head toward the posterior, does not enforce this: it decides how the evidence for a transition is allocated across classes, and has zero gradient in u over a whole interval of doubts (App. C.3), so the total evidence is free. We close the gap with a regulariser, the evidence discipline: a pull toward the vacuous opinion, the opinion with no evidence and $u = 1$ . Evidential classifiers use such a pull to shed evidence the labels do not support (Sensoy et al., 2018); here its job is different: with the fit blind to the total, the discipline alone sets it:

$$
\mathcal { L } _ { u } = \beta _ { u } \sum _ { g } \mathrm { K L } \Big ( \mathrm { D i r } \big ( e ^ { g } ( s _ { t } , a _ { t } ) + ( W / K ) \mathbf { 1 } \big ) \Big \| \mathrm { D i r } \big ( ( W / K ) \mathbf { 1 } \big ) \Big ) ,\tag{5}
$$

where $g$ runs over the $G$ categorical variables and $\beta _ { u }$ weights the term. The dynamics loss now holds evidence up only on the scene–action pairs the agent has taken, and the discipline lets it recede wherever nothing holds it up. On the training distribution the head settles at the most doubtful opinion consistent with the fit (App. C.4); beyond it the doubt rises, as Sec. 4.3 shows.

Fusing observation into prediction. What the model observes should reduce its doubt and never raise it. In a standard world model the posterior is a separate network that reads $( s _ { t } , a _ { t } , x _ { t + 1 } )$ and replaces the prediction outright: what the prediction knew is discarded, and what the observation adds is never counted. Our posterior instead fuses the observation into the prediction with a fusion operator ⊕ of SL (Han et al., 2023; Xu et al., 2024); we use cumulative fusion, which adds evidence (Jøsang, 2016) (App. D.1). A linear layer, which replaces the posterior network, reads $x _ { t + 1 }$ alone and reports observation evidence $e _ { \mathrm { o b s } } ( x _ { t + 1 } )$ ; this keeps the two sources independent, so no evidence is counted twice:

$$
e _ { \mathrm { p o s t } } ( s _ { t } , a _ { t } , x _ { t + 1 } ) = e ( s _ { t } , a _ { t } ) \oplus e _ { \mathrm { o b s } } ( x _ { t + 1 } ) .\tag{6}
$$

The posterior is the mean of the fused opinion, Eq. (3) with $e _ { \mathrm { p o s t } }$ , from which $z _ { t + 1 }$ is drawn in place of $q .$ Fusion never raises the doubt, and brings two more properties: a vacuous observation leaves the state unchanged, and an informative one moves the state toward the observation by exactly the share of doubt it removes. It is also where experience enters the head: the dynamics loss pulls the prediction toward this posterior, whose sharpness bounds the doubt the prediction may keep (App. C.3), so evidence accrues where observations have sharpened the posterior.

## 3.2 Carrying trust through imagination

Carrying doubt forward as trust. The trust in an imagined step is inherited: each imagined state is the model’s own prediction, so a step can be trusted no further than its input. One might carry it in the state, by deducing each imagined opinion from the doubtful one before it; but that recursion settles after one step at a level set by the current step alone, and the lineage is lost (App. D.2). We carry it in the return instead, and show that it then compounds as the trust discounting of SL (App. D.3). The standard return already holds a trust of a kind: at every step it keeps a fixed share λ of the imagined future and hands the rest to the critic. We let that share follow the doubt. Let $\bar { u } _ { t + 1 }$ be the doubt of the transition $( s _ { t } , a _ { t } )$ , averaged over the $G$ variables, and $\tau _ { t + 1 } = ( 1 - \bar { u } _ { t + 1 } ) / ( 1 - \varepsilon )$ its trust, equal to one when the doubt sits on its floor ε; the share kept beyond step t becomes $\lambda \tau _ { t + 1 } \colon$

$$
\mathrm { S T A N D A R D ~ R E T U R N } R _ { t } ^ { \lambda } = r _ { t } + \gamma \big [ ( 1 - \lambda ) v ( s _ { t + 1 } ) + \lambda R _ { t + 1 } ^ { \lambda } \big ] ,\tag{2}
$$

$$
\begin{array} { r } { \mathrm { O U R \ L U C I D \ R E T U R N } \quad R _ { t } ^ { \tau } = r _ { t } + \gamma \big [ ( 1 - \lambda \tau _ { t + 1 } ) v ( s _ { t + 1 } ) + \lambda \tau _ { t + 1 } \ R _ { t + 1 } ^ { \tau } \big ] , } \end{array}\tag{7}
$$

where $R _ { t } ^ { \tau }$ is our trust-weighted return, with the same terminal value $R _ { H } ^ { \tau } = v ( s _ { H } )$ . A step with high doubt passes less of the imagined future to the return and more to the critic’s value at that step.

Corollary 1. The λ-return Eq. (2) is Eq. (7) with constant trust, $\tau \equiv 1$ . With Prop. 1, a standard world model is our lucid world model with both constants pinned.

Compounding trust along the rollout. Trust compounds along the rollout. For a rollout of H steps from a real state $s _ { 0 }$ , write $\begin{array} { r } { C _ { n } = \bar { \prod } _ { k = 1 } ^ { n } \tau _ { k } } \end{array}$ , with $C _ { 0 } = 1$ , for the trust carried to its n-th imagined step. Unrolled from $s _ { 0 }$ , the return shows what this product does:

$$
R _ { 0 } ^ { \tau } = \sum _ { n = 1 } ^ { H - 1 } \left( \lambda ^ { n - 1 } C _ { n - 1 } - \lambda ^ { n } C _ { n } \right) R ^ { ( n ) } ~ + ~ \lambda ^ { H - 1 } C _ { H - 1 } R ^ { ( H ) } ,\tag{8}
$$

where $\begin{array} { r } { R ^ { ( n ) } = \sum _ { k = 0 } ^ { n - 1 } \gamma ^ { k } r _ { k } + \gamma ^ { n } v ( s _ { n } ) } \end{array}$ is the n-step return. The weights are nonnegative and sum to one, and the weight of the n-step return is the share lost between steps $n - 1$ and n: one doubtful step lowers the weight of every return beyond it. With $\tau \equiv 1$ the weights are the geometric weights $( 1 - \lambda ) \lambda ^ { n - 1 }$ of Eq. (2) (App. D.4). How far imagination counts is thus set by trust, not by a constant: the return reaches as deep as the model’s experience reaches, and falls back on the critic beyond it.

## 3.3 Deciding by trust

Deciding what to learn from. We train the actor and the critic on $R ^ { \tau }$ instead of $R ^ { \lambda }$ , so trust decides how much of each imagined step the agent learns from. For this to be safe, reweighting must not change what the critic converges to, and it should make model errors hurt less. Both hold:

Theorem 1 (Contraction and bias bound). For any fixed doubt profile: (i) the critic’s update under Eq. (7) is a γ-contraction to ${ \hat { V } } ^ { \pi }$ , the value of π in the imagined model, as under Eq. (2); (ii) with the critic at the true value $V ^ { \pi }$ and $\delta _ { j }$ the model’s one-step error at imagined depth j (App. D.4):

$$
\left\| \mathbb { E } [ R _ { 0 } ^ { \tau } ] - V ^ { \pi } \right\| _ { \infty } \leq \ \sum _ { j = 0 } ^ { H - 1 } \gamma ^ { j } \lambda ^ { j } C _ { j } \delta _ { j } .\tag{9}
$$

(i) holds because the weights of Eq. (8) are set by the head, not by the value being learned: trust changes how fast the critic gets there, not where. (ii) weights each depth’s model error by $\lambda ^ { j } C _ { j } ^ { \bar { . } }$ , the share of the return that reaches that depth, so the bound tightens wherever doubt precedes error.

Deciding what to do. Doubt is a function of $( s _ { t } , a )$ , so it is known for any candidate action before the action is taken (App. D.6). We score a candidate future by the trust-weighted sum of its rewards:

$$
J ( a _ { 0 : H - 1 } ) = \sum _ { k = 0 } ^ { H - 1 } \gamma ^ { k } C _ { k } r _ { k } + \gamma ^ { H } C _ { H } v ( s _ { H } ) .\tag{10}
$$

J is Eq. (7) with λ = 1 and the fallback set to zero: untrusted imagined reward contributes nothing, whereas in learning that share went to the critic, so a course the model cannot vouch for cannot win on the critic’s word. The simplest use of J is a veto: when trust in the agent’s course collapses, every reward beyond that point drops out of J, so the agent withdraws from the course and chooses again (Sec. 4.6). Together, Eqs. (4), (7) and (10) apply one rule three times: keep the share of a prediction that experience supports, and hand the rest to a fallback, the uniform distribution, the critic, or zero.

## 4 Experiments

## 4.1 Experimental protocol

Bases. Lucid dreaming applies to any world model with a categorical transition head, and we test it on four bases that span the current designs, in four independent code bases, with the same readout and the same protocol throughout. DreamerV3 (Hafner et al., 2025) is a recurrent state-space model with a pixel decoder; R2-Dreamer (Morihira et al., 2026) is recurrent but learns its representation without reconstruction; EMERALD (Burchi & Timofte, 2025) predicts its latent tokens with a masked transformer; OC-STORM (Zhang et al., 2026) is a transformer world model, used here in its visual-only branch. Between them they cover recurrent and transformer dynamics, reconstruction-based and reconstruction-free representations, and flat and spatial latent states.

Environments. We use three environment families to produce situations the model has not experienced. In DeepMind Control (DMC) continuous control (walker, cheetah, pendulum, finger, cartpole), where the dynamics are smooth, we corrupt the actions fed to imagination; in a first-person ViZDoom maze, navigated from 64 × 64 pixels with discrete actions, we train on one map and then alter it, opening a sealed door or repainting the walls; in Crafter, a procedurally generated open world, we corrupt the actions too and let the agent explore and meet caves, lava and terrain it has never seen. Together they cover continuous and discrete actions, proprioceptive and pixel inputs, and situations we inject, build in, or let the agent meet on its own (App. F).

Opponents. We compare against seventeen readouts, eleven in each of the two tests, in the direction their authors define. Free readouts of the same pass: entropy, maximum probability, the KL between posterior and prediction, reconstruction and one-step error, and our own formula on the unmodified base (base in Table 1); fitted heads on frozen features: RND on the latent and on the encoder embedding, an evidential head, Mahalanobis and k-NN distances, a latent ensemble; multiple forwards: MC dropout, Laplace, snapshot, selfand deep ensembles. Before decision, every fitted head and ensemble receives the same categorical latent our readout uses, so that what is compared is the readout, not how much the head is allowed to see. Appendix E.4 defines each readout and its cost.

Table 1: (A) After observation: AUROC of the readout, frames of the opened room against familiar ones; 0.5 is chance. (B) Before decision: lift $\rho _ { 1 } / \rho _ { 0 }$ , the readout under corrupted actions over the readout under true ones; 1 is no response. Under each column its cost relative to the base (+c: heads with c times its parameters; m fwd: m forward passes). Means over five seeds; bold is the best in a row; grey numbers move the wrong way, falling as the model leaves its experience; a dash (–) is a readout the base cannot provide. Each test uses eleven of the seventeen readouts; reconstruction and one-step error need the arriving frame, so appear in (A) only.
<table><tr><td>(A) After observation</td><td></td><td colspan="2"></td><td colspan="4">free readouts</td><td colspan="4">fitted heads</td><td colspan="2">multiple forwards</td></tr><tr><td></td><td></td><td>ours</td><td>base entropy</td><td></td><td>KL</td><td>recon.</td><td>1-step</td><td>RND</td><td>RND-e evid.</td><td></td><td>latent</td><td>self</td><td>deep</td></tr><tr><td>base</td><td>task</td><td>1×</td><td>1×</td><td>1×</td><td>1×</td><td>1×</td><td>1×</td><td>+.18</td><td>+.38</td><td>+.12</td><td>+.62</td><td>3 fwd</td><td>3×</td></tr><tr><td>DreamerV3</td><td></td><td>0.77</td><td>0.24</td><td>0.31</td><td>0.24</td><td>0.06</td><td>0.10</td><td>0.39</td><td>0.27</td><td>0.48</td><td>0.33</td><td>0.21</td><td>0.21</td></tr><tr><td>R2-Dreamer</td><td>maze</td><td>0.83</td><td>0.37</td><td>0.36</td><td>0.52</td><td></td><td>0.50</td><td>0.45</td><td>0.66</td><td>0.30</td><td>0.33</td><td>0.41</td><td>一</td></tr><tr><td>EMERALD</td><td></td><td>0.96</td><td>0.72</td><td>0.40</td><td>0.27</td><td>0.05</td><td>0.11</td><td>0.22</td><td>0.20</td><td>0.85</td><td>0.83</td><td>0.26</td><td>一</td></tr><tr><td>OC-STORM</td><td></td><td>0.82</td><td>0.75</td><td>0.75</td><td>0.37</td><td>0.09</td><td>0.16</td><td>0.53</td><td>0.46</td><td>0.57</td><td>0.59</td><td>0.24</td><td>一</td></tr><tr><td colspan="2">(B) Before decision</td><td></td><td colspan="3">free readouts</td><td colspan="4">fitted heads</td><td colspan="4">multiple forwards</td></tr><tr><td></td><td></td><td>ours</td><td>base</td><td>e entropy</td><td>max p.</td><td>. Mahal.</td><td>kNN RND</td><td></td><td>evid.</td><td></td><td>latent MC drop Laplace</td><td></td><td>snap.</td></tr><tr><td>base</td><td>task</td><td>1X</td><td>1×</td><td>1×</td><td>1×</td><td>+.04</td><td>+2.4</td><td>+.13</td><td>+.10</td><td>+.52</td><td>20 fwd</td><td>20 fwd</td><td>4 fwd</td></tr><tr><td>DreamerV3</td><td>walker</td><td>1.30</td><td>0.85</td><td>1.03</td><td>1.00</td><td>1.06</td><td>1.07</td><td>0.80</td><td>1.00</td><td>1.04</td><td>1.02</td><td>1.01</td><td>0.93</td></tr><tr><td></td><td>cheetah</td><td>2.33</td><td>0.70</td><td>1.14</td><td>1.14</td><td>1.10</td><td>0.95</td><td>0.93</td><td>1.00</td><td>1.04</td><td>1.04</td><td>0.88</td><td>0.94</td></tr><tr><td></td><td>pendulum</td><td>1.33</td><td>1.00</td><td>1.03</td><td>1.04</td><td>1.08</td><td>1.02</td><td>0.96</td><td>1.00</td><td>0.91</td><td>1.07</td><td>0.84</td><td>0.95</td></tr><tr><td></td><td>Crafter</td><td>1.49</td><td>0.84</td><td>1.02</td><td>0.97</td><td>0.88</td><td>0.93</td><td>1.02</td><td>1.00</td><td>1.01</td><td>0.99</td><td>0.92</td><td>0.95</td></tr><tr><td>R2-Dreamer walker</td><td></td><td>3.11</td><td>1.19</td><td>1.00</td><td>1.01</td><td>2.45</td><td>1.14</td><td>1.36</td><td>0.99</td><td>1.33</td><td>1.31</td><td>1.37</td><td>一</td></tr><tr><td></td><td>cheetah</td><td>2.76</td><td>0.99</td><td>1.01</td><td>1.02</td><td>1.17</td><td>1.01</td><td>1.18</td><td>1.00</td><td>1.04</td><td>1.21</td><td>1.01</td><td>一</td></tr><tr><td></td><td>pendulum</td><td>1.53</td><td>1.02</td><td>0.88</td><td>0.87</td><td>1.01</td><td>0.94</td><td>1.11</td><td>0.99</td><td>0.96</td><td>0.96</td><td>1.11</td><td>一</td></tr><tr><td></td><td>Crafter</td><td>1.15</td><td>0.81</td><td>1.02</td><td>1.00</td><td>1.05</td><td>1.02</td><td>1.05</td><td>1.00</td><td>1.04</td><td>1.00</td><td>1.14</td><td>1.06</td></tr><tr><td>EMERALD</td><td>walker</td><td>3.13</td><td>1.10</td><td>1.55</td><td>1.67</td><td>2.08</td><td>1.42</td><td>1.51</td><td>1.00</td><td>1.53</td><td>0.05</td><td>0.06</td><td>0.16</td></tr><tr><td></td><td>cheetah</td><td>3.67</td><td>1.04</td><td>1.42</td><td>1.57</td><td>2.58</td><td>1.56</td><td>1.42</td><td>1.00</td><td>1.72</td><td>0.28</td><td>0.29</td><td>0.25</td></tr><tr><td></td><td>pendulum</td><td>2.06</td><td>1.03</td><td>1.02</td><td>1.03</td><td>1.39</td><td>1.80</td><td>1.87</td><td>1.00</td><td>1.34</td><td>0.62</td><td>0.66</td><td>0.94</td></tr><tr><td></td><td>Crafter</td><td>1.34</td><td>1.01</td><td>1.16</td><td>1.16</td><td>0.89</td><td>0.89</td><td>1.01</td><td>1.00</td><td>1.01</td><td>0.97</td><td>0.97</td><td>1.00</td></tr><tr><td>OC-STORM walker</td><td></td><td>5.62</td><td>1.16</td><td>2.74</td><td>2.14</td><td>1.08</td><td>0.93</td><td>2.48</td><td>1.00</td><td>4.18</td><td>4.37</td><td>1.08</td><td>一</td></tr><tr><td></td><td>cheetah</td><td>4.30</td><td>1.06</td><td>2.21</td><td>2.08</td><td>1.12</td><td>0.98</td><td>1.47</td><td>1.03</td><td>2.18</td><td>2.95</td><td>0.69</td><td>一</td></tr><tr><td></td><td>pendulum</td><td>2.29</td><td>1.03</td><td>1.27</td><td>1.14</td><td>0.86</td><td>0.89</td><td>0.98</td><td>1.00</td><td>0.86</td><td>1.01</td><td>0.80</td><td>一</td></tr><tr><td></td><td>Crafter</td><td>2.59</td><td>1.04</td><td>2.08</td><td>1.79</td><td>0.82</td><td>0.92</td><td>0.85</td><td>1.01</td><td>1.35</td><td>0.61</td><td>0.55</td><td>0.80</td></tr></table>

## 4.2 Main results

(A) After observation. This test asks whether the doubt read after an observation tells that the world has changed. We open a sealed door in the maze and drive the same routes on the old and the new map; a readout is scored by its AUROC (Table 1). On every base the LucidWM readout recognises the new room. Most readouts of the unmodified base point the wrong way: seen through familiar walls, the new room yields a sharp prediction, so they read it as more familiar than the rest of the maze, and the fitted heads and ensembles, at up to three times the cost, are no more reliable.

(B) Before decision. This test asks whether the doubt read before a decision tells that an imaginedfuture has left the model’s experience. Imagination starts from real states, and its actions are replaced by random ones with growing probability; we report the lift $\rho _ { 1 } / \rho _ { 0 }$ , the readout under fully corrupted actions over that under the true ones, so 1 is no response and a readout passes when it rises above it. The LucidWM readout rises and leads on every row, while the same formula on the unmodified base moves little or the wrong way: fed nonsense, a model without doubt barely notices (App. F.5).

## 4.3 Doubt rises beyond experience

Imagination leaving experience. This test asks whether doubt rises when imagination leaves what the model has experienced. Fig. 4 projects the visited states onto a plane, ringed at two standard deviations, and starts 48 randomaction rollouts from real states in the cloud, forty steps each, colouring every step by doubt; below, each rollout’s doubt and the median are drawn against the 95th percentile on experienced states. The rollouts of both models drift equally far out of the cloud, to twice the distance of any experienced state from its neighbours. Only LucidWM’s doubt rises as they leave: within ten steps its median crosses the 95th percentile, and by the end nine in ten sit above it. The base’s doubt never moves: its imagination has left everything it knows, and it is as confident as ever.

![](images/7c625b1d6399c664fd57bb35e90b47119ad0025b1408deabf240a14ff3f1a48c.jpg)  
95th percentile of doubt on experienced states (both panels)  
Figure 4: Ours doubts the departure; base does not. Top: experienced states and 48 random-action rollouts coloured by doubt. Bottom: doubt along the rollouts.

Observation leaving experience. This test asks whether the doubt reads more

than the surface oftheframes. We alter the maze in two ways (Fig. 5): the walls of one room are repainted, which changes every pixel and nothing else; and a wall is opened onto a room that was sealed off during training, which changes the layout and shows the agent no texture it has not seen. For the new room we replay the same actions on the training map and read the difference. In the repainted room the lucid doubt rises far above its alarm line again and again; at the new room, it rises each time the agent turns to the opening and falls back as soon as it turns away. Under one alarm rule, set on familiar frames, the LucidWM readout raises an alarm in both rooms and the base’s readout in neither. The doubt reads the structure beneath the pixels: every frame at the opening is made of familiar walls, and the doubt rises all the same.

(a) New paint  
![](images/9cbade6c345ee79c1d438033476c1a5bb67d3586247e84332f6f4aee96212489.jpg)

(b) New layout  
![](images/1b766fdd0af612328719a54ff605c6fbf576fa5c2042a256fa3aa7c58b81ae44.jpg)  
Figure 5: The alarm fires where the world changed. (a) A drive into the repainted room. (b) A patrol that twice faces the newly opened room. Doubt in each model’s own units; dashed line: alarm line; sand: frames in the changed region, where the gold-outlined pictures were taken.

## 4.4 Doubt falls as experience grows

Experience accumulating. As the model sees more of the world, its doubt should drain, and drain where its predictions improve. We take one stretch of experience and, at six check points of one run (three in Fig. 6, all in App. F.7), imagine the same future from it and compare it with what happened. At one step the doubt halves, falling at every checkpoint, and the error falls nearly 25-fold, though the two are measured independently: the error by comparing pixels, the doubt from the transition’s evidence alone. Step 24 lies past the point where the dynamics turn chaotic, so no amount of training makes the prediction right; there the doubt stays the highest of the steps shown at every checkpoint.

![](images/3fbb36230be85c5478f7485b0eef1fe7ca8e42619ae231bda523170b242a84ab.jpg)  
Figure 6: Learning more, doubting less. Rows: checkpoints of one run; columns: imagined steps. Tiles: pixel error, dark right, bright wrong. Shaded: chaotic horizon, never learned.

## 4.5 Ablating trust

Trust as a filter. If trust means what it should, the imagined steps it trusts least should be the ones most wrong. We test this: from 480 real states per run we imagine 15 steps, rank every imagined step by a readout, drop the worst fifth, and measure how far the prediction error of the rest falls, on three DMC tasks. Deep steps are worse on average, so depth alone is a good filter, and over all depths it matches both trust and entropy (Fig. 7a, pale bars). To see what a readout knows beyond depth, we rank within each depth (solid bars): depth alone drops to zero, entropy to almost nothing on two of three tasks, while trust keeps a gain of its own on all three, up to 23%. Over all depths, compounded trust removes 32% of the error against 19% for the doubt of one step; within each

![](images/493dedc6ba1e924aace2299dc086bb36c74e60662e38f4077a7de3e155321c66.jpg)

(b)  
![](images/4562dc984a957c6bc145025c3d9c7aaf100a808c71b6b2f4e8683f36dbbb9967.jpg)  
Figure 7: Trust tells which imagined steps to keep. Error removed by dropping the worst fifth of steps: (a) three DMC tasks, (b) cheetah from pixels. Pale: over all depths; solid: within each depth; whiskers: bootstrap 95% intervals.

depth, the posterior’s doubt trails both (Fig. 7b): compounding carries the depth of a dream, the doubt of each transition its drift, so trust reads both how far the dream has gone and how far it has drifted (App. F.8–F.10).

## 4.6 Acting on trust

A repainted dead end. We run the same weights and seed twice through a dead end whose deep half was repainted after training: every wall is where it was, so a policy that learned the map walks in as before. Once the agent ignores its doubt (base); once it acts on trust with the veto of Sec. 3.3 (ours). The veto is one rule: when doubt holds above the alarm line for three frames, trust in the course collapses, the agent turns, and its policy resumes. For the first 24 steps the two are one agent. At step 24 the veto fires; ours turns, leaves the dead end and reaches the goal, an armour, at step 190, while base circles the room until step 262 and arrives at step 362 (Fig. 8): the same policy, deciding 238 steps earlier and arriving in half the steps. Over its whole walk the entropy of base crosses its own line for a single frame, so the same rule would never have fired. On both altered maps, 143 of 145 vetoes fired inside a changed room, though the agent is never told where the changes are (App. F.13).

![](images/cde9c3edb8f630679821b5105ab73246651e465821c36d227f4179b722e39d87.jpg)

![](images/cc61202c4c7eadf10c9d2002c54cd93940a0417c43a1bda8ad13bf83b4335018.jpg)  
Figure 8: One veto, half the steps. Top: steps to the armour. Middle: both routes from the repainted dead end (sand) to the armour (star), and the two views at step 183, seven steps before ours arrives. Bottom: doubt along ours and the entropy of base, each on its own p<sub>99</sub> alarm line; circles: where ours turns and base leaves.

## 5 Conclusion

In this work, we identified that existing uncertainty estimates for world models are read from the prediction, and therefore stay confident on exactly what the model has never experienced. To address this, we proposed the Lucid World Model (LucidWM), which learns uncertainty from experience: SL is built into the latent state transition, so that every predicted state carries a doubt, doubt compounds along imagination into trust, and trust weights both what the agent learns from it and what it decides to do. Experiments on four world-model bases and against seventeen readouts show that the doubt recognises a changed world after observation and a drifting rollout before decision, where prior readouts barely move or move the wrong way, and that acting on trust leads to safer choices and faster learning (App. F.11, F.12). A standard world model is the special case with both constants pinned; unfolding them adds no parameter and no forward pass.

## References

Milton Abramowitz and Irene A. Stegun. Handbook ofMathematical Functions with Formulas, Graphs, and Mathematical Tables, volume 55 of Applied Mathematics Series. National Bureau of Standards, 1964.

Alexander Amini, Wilko Schwarting, Ava Soleimany, and Daniela Rus. Deep evidential regression. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

Kavosh Asadi, Dipendra Misra, and Michael L. Littman. Lipschitz continuity in model-based reinforcement learning. In International Conference on Machine Learning (ICML), 2018.

Viktor Bengs, Eyke Hullermeier, and Willem Waegeman. Pitfalls of epistemic uncertainty quantification ¨ through loss minimisation. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Julia Berger, Bernd Frauenknecht, Sebastian Trimpe, and Bastian Leibe. Biased dreams: Limitations to epistemic uncertainty quantification in latent dynamics models. Reinforcement Learning Journal, 7, 2026.

Jacob Buckman, Danijar Hafner, George Tucker, Eugene Brevdo, and Honglak Lee. Sample-efficient reinforcement learning with stochastic ensemble value expansion. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

Maxime Burchi and Radu Timofte. Accurate and efficient world modeling with masked latent transformers. In International Conference on Machine Learning (ICML), 2025.

Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. In International Conference on Learning Representations (ICLR), 2019.

Bertrand Charpentier, Daniel Zugner, and Stephan G¨ unnemann. Posterior network: Uncertainty estimation¨ without OOD samples via density-based pseudo-counts. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine. Deep reinforcement learning in a handful of trials using probabilistic dynamics models. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, and Philipp Hennig. Laplace redux – effortless Bayesian deep learning. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Bernd Frauenknecht, Artur Eisele, Devdutt Subhasish, Friedrich Solowjow, and Sebastian Trimpe. Trust the model where it trusts itself – model-based actor-critic with uncertainty-aware rollout adaption. In International Conference on Machine Learning (ICML), 2024.

Yarin Gal and Zoubin Ghahramani. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In International Conference on Machine Learning (ICML), 2016.

Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

David Ha and Jurgen Schmidhuber. Recurrent world models facilitate policy evolution. In¨ Advances in Neural Information Processing Systems (NeurIPS), 2018.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In International Conference on Machine Learning (ICML), 2019.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 640:647–653, 2025.

Zongbo Han, Changqing Zhang, Huazhu Fu, and Joey Tianyi Zhou. Trusted multi-view classification with dynamic evidential fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2): 2551–2566, 2023.

Matthias Hein, Maksym Andriushchenko, and Julian Bitterwolf. Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu, John E. Hopcroft, and Kilian Q. Weinberger. Snapshot ensembles: Train 1, get M for free. In International Conference on Learning Representations (ICLR), 2017.

Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. When to trust your model: Model-based policy optimization. In Advances in Neural Information Processing Systems (NeurIPS), 2019.

Audun Jøsang. Subjective Logic: A Formalism for Reasoning Under Uncertainty. Springer, 2016.

Mira Jurgens, Nis Meinert, Viktor Bengs, Eyke H ¨ ullermeier, and Willem Waegeman. Is epistemic uncertainty¨ faithfully represented by evidential deep learning methods? In International Conference on Machine Learning (ICML), 2024.

Rahul Kidambi, Aravind Rajeswaran, Praneeth Netrapalli, and Thorsten Joachims. MOReL: Model-based offline reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

Thanard Kurutach, Ignasi Clavera, Yan Duan, Aviv Tamar, and Pieter Abbeel. Model-ensemble trust-region policy optimization. In International Conference on Learning Representations (ICLR), 2018.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting outof-distribution samples and adversarial attacks. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

Ali Malik, Volodymyr Kuleshov, Jiaming Song, Danny Nemer, Harlan Seymour, and Stefano Ermon. Calibrated model-based deep reinforcement learning. In International Conference on Machine Learning (ICML), 2019.

Andrey Malinin and Mark Gales. Predictive uncertainty estimation via prior networks. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

Vincent Micheli, Eloi Alonso, and Franc¸ois Fleuret. Transformers are sample-efficient world models. In International Conference on Learning Representations (ICLR), 2023.

Naoki Morihira, Amal Nahar, Kartik Bharadwaj, Yasuhiro Kato, Akinobu Hayashi, and Tatsuya Harada. R2-Dreamer: Redundancy-reduced world models without decoders or augmentation. In International Conference on Learning Representations (ICLR), 2026.

Feiyang Pan, Jia He, Dandan Tu, and Qing He. Trust the model when it is confident: Masked model-based actor-critic. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

Nedko Savov, Naser Kazemi, Mohammad Mahdi, Danda Pani Paudel, Xi Wang, and Luc Van Gool. Exploration-driven generative interactive environments. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Ramanan Sekar, Oleh Rybkin, Kostas Daniilidis, Pieter Abbeel, Danijar Hafner, and Deepak Pathak. Planning to explore via self-supervised world models. In International Conference on Machine Learning (ICML), 2020.

Murat Sensoy, Lance Kaplan, and Melih Kandemir. Evidential deep learning to quantify classification uncertainty. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

Yiyou Sun, Yifei Ming, Xiaojin Zhu, and Yixuan Li. Out-of-distribution detection with deep nearest neighbors. In International Conference on Machine Learning (ICML), 2022.

Philipp Wu, Alejandro Escontrela, Danijar Hafner, Ken Goldberg, and Pieter Abbeel. DayDreamer: World models for physical robot learning. In Conference on Robot Learning (CoRL), 2022.

Cai Xu, Jiajun Si, Ziyu Guan, Wei Zhao, Yue Wu, and Xiyue Gao. Reliable conflictive multi-view learning. In AAAI Conference on Artificial Intelligence (AAAI), 2024.

Qianyi Xu, Gousia Habib, Feng Wu, Dilruk Perera, and Mengling Feng. medDreamer: Model-based reinforcement learning with latent imagination on complex EHRs for clinical decision support. In ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2026.

Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Zou, Sergey Levine, Chelsea Finn, and Tengyu Ma. MOPO: Model-based offline policy optimization. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

Weipu Zhang, Adam Jelley, Trevor McInroe, Amos Storkey, and Gang Wang. Object-centric world models from few-shot annotations for sample-efficient reinforcement learning. In International Conference on Learning Representations (ICLR), 2026.

APPENDIX PAGE   
OVERVIEW   
A NOTATION 14   
B WHAT LUCIDWM CHANGES 14   
THEORY   
C WHY THE DOUBT READS EXPERIENCE 15   
C.1 The doubt measures the evidence total 15   
C.2 In a table, the doubt counts experience 16   
C.3 The prediction and the fit cannot see the evidence total 16   
C.4 The discipline makes the doubt count experience 17   
C.5 The doubt sets the expected error 18   
D WHY TRUST IS CARRIED IN THE RETURN 18   
D.1 Observation never raises the doubt 19   
D.2 In the state, the lineage is lost 19   
D.3 In the return, the doubt compounds 20   
D.4 The lucid return keeps the fixed point and discounts model error 20   
D.5 When trust tightens the bound 22   
D.6 Doubt ranks actions before acting 22   
EXPERIMENTS   
E EXPERIMENTAL DETAILS 22   
E.1 Protocols 22   
E.2 Four bases 23   
E.3 Every experiment at a glance 23   
E.4 Seventeen readouts and their cost 24   
F FURTHER RESULTS 24   
F.1 The alarm fires in the new room 24   
F.2 The alarm fires before the fall 25   
F.3 The alarm fires at nightfall 26   
F.4 The alarm fires on every base 26   
F.5 The worse the actions, the higher the doubt 26   
F.6 The alarm fires at any line, any checkpoint 27   
F.7 The doubt drains as the model learns 27   
F.8 Trust fades faster where the dream drifts 28   
F.9 Trust keeps its edge across ten runs 28   
F.10 Hyperparameter analysis 29   
F.11 Choosing futures by trust cuts deaths 29   
F.12 LucidWM learns Crafter faster than its base 30   
F.13 Vetoes fire where the world changed 30

## A Notation

Table 2: Symbols of Secs. 2–4. Per variable: defined for each of the G categorical variables, whose index g is dropped when clear. The last column gives the value that recovers the standard world model.
<table><tr><td>symbol</td><td>meaning</td><td>defined</td><td>standard</td></tr><tr><td colspan="4">World model and imagination</td></tr><tr><td> $x _ { t } , \ s _ { t } , \ a _ { t }$ </td><td>observation, latent state, action at time t</td><td>Sec. 2</td><td></td></tr><tr><td> $G , K$ </td><td>number of categorical variables of the state; classes per variable</td><td>Sec. 2</td><td></td></tr><tr><td> $z _ { t + 1 }$ </td><td>one categorical variable of the next state</td><td>Eq. (1)</td><td></td></tr><tr><td> $\ell ( s _ { t } , a _ { t } )$ </td><td>the transition head&#x27;s K logits (per variable)</td><td>Eq. (1)</td><td></td></tr><tr><td> $p ( z _ { t + 1 } \mid s _ { t } , a _ { t } )$ </td><td>the head&#x27;s prediction</td><td>Eqs. (1), (4)</td><td></td></tr><tr><td> $q ( \cdot \mid s _ { t } , a _ { t } , x _ { t + 1 } )$ </td><td>posterior of a standard world model</td><td>Sec. 2</td><td></td></tr><tr><td>ε</td><td>uniform share mixed into the head (0.01 in DreamerV3); floor Eq. (1) of the doubt</td><td></td><td></td></tr><tr><td> $r _ { t } , \gamma , H$ </td><td>imagined reward, discount, imagination horizon</td><td>Eq. (2)</td><td></td></tr><tr><td> $\pi , v$ </td><td>actor, critic</td><td>Eq. (2)</td><td></td></tr><tr><td> $\lambda$ </td><td>share of the imagined future kept at each step</td><td>Eq. (2)</td><td></td></tr><tr><td> $R _ { t } ^ { \lambda }$ </td><td>λ-return</td><td>Eq. (2)</td><td></td></tr><tr><td colspan="4">Opinion of the transition head (per variable)</td></tr><tr><td> $e , \ e _ { i }$ </td><td>evidence softplus l; its entry for class i</td><td>Eq. (3), Sec. 3.1</td><td></td></tr><tr><td> $W$ </td><td>prior weight of the opinion</td><td>Eq. (3)</td><td></td></tr><tr><td> $S$ </td><td>total evidence  $\textstyle \sum _ { i } e _ { i }$ </td><td>Eq. (3)</td><td> $W ( 1 - \varepsilon ) / \varepsilon$ </td></tr><tr><td> $\hat { p }$ </td><td>direction of the evidence,  $e / S$ </td><td>Eq. (3)</td><td>softmax(l)</td></tr><tr><td> $\alpha _ { i } , \textit { P }$ </td><td>Dirichlet concentration  $e _ { i } + W / K ;$  mean of the opinion</td><td>Eq. (3)</td><td></td></tr><tr><td>u</td><td>doubt: uncertainty mass  $W / ( W + S )$  , floored at ε</td><td>Sec.3.1</td><td>ε</td></tr><tr><td> $\mathcal { L } _ { u } , ~ \beta _ { u }$ </td><td>evidence discipline and its weight</td><td>Eq. (5)</td><td> $\beta _ { u } = 0$ </td></tr><tr><td> $e _ { \mathrm { o b s } } ( x _ { t + 1 } )$ </td><td>evidence read from the observation alone</td><td>Eq. (6)</td><td></td></tr><tr><td> $\oplus , \ e _ { \mathrm { p o s t } }$ </td><td>fusion of opinions (here cumulative, which adds evidence); posterior evidence</td><td>Eq. (6)</td><td></td></tr><tr><td colspan="4">Trust along imagination and decision</td></tr><tr><td> $\bar { u } _ { t + 1 }$ </td><td>doubt of the transition  $\left( { { s _ { t } } , { a _ { t } } } \right)$  , averaged over the G variables Sec. 3.2</td><td></td><td>ε</td></tr><tr><td> $\tau _ { t + 1 }$ </td><td>trust of that transition.  $( 1 - \bar { u } _ { t + 1 } ) / ( 1 - \varepsilon )$ </td><td>Sec. 3.2</td><td>1</td></tr><tr><td> $R _ { t } ^ { \tau }$ </td><td>trust-weighted return</td><td>Eq.(7)</td><td> $R _ { t } ^ { \lambda }$ </td></tr><tr><td> $C _ { n }$ </td><td>trust carried to imagined step n,  $\begin{array} { r } { \prod _ { k = 1 } ^ { n } \tau _ { k } , C _ { 0 } = 1 } \end{array}$ </td><td>Sec. 3.2</td><td>1</td></tr><tr><td> ${ \cal R } ^ { ( n ) }$ </td><td>n-step return from the real state so</td><td>Eq. (8)</td><td></td></tr><tr><td> $\hat { V } ^ { \pi } , V ^ { \pi }$ </td><td>value of π in the imagined model; in the environment</td><td>Thm. 1</td><td></td></tr><tr><td> $\delta _ { j }$ </td><td>the model&#x27;s one-step error at imagined depth j</td><td>Thm. 1</td><td></td></tr><tr><td> $J \big ( a _ { 0 : H - 1 } \big )$ </td><td>trust-weighted score of a candidate course of actions</td><td>Eq. (10)</td><td>C ≡ 1</td></tr><tr><td> $\rho _ { 1 } / \rho _ { 0 }$ </td><td>lift: readout under corrupted actions over readout under true ones</td><td>Sec. 4.2</td><td></td></tr></table>

Symbols used inside a single proof in Apps. C–D are defined where they appear.

## B What LucidWM changes

A standard categorical world model pins two constants: its head mixes the same ε into every prediction, Eq. (1), and its return keeps the same share λ of every imagined step, Eq. (2). LucidWM unfolds both into quantities read from the head’s own logits and applies one rule at every stage: keep the share of a prediction that experience supports, and hand the rest to a fallback. Table 3 lists every change; the statements behind each row are Prop. 1, Prop. 4, Prop. 6, Thm. 1 and Prop. 10. No parameter and no forward pass is added: the posterior is the fused evidence of Eq. (6) in place of a second network, and the readouts u, τ and C are functions of quantities the model already computes.

Table 3: What LucidWM changes in a categorical world model. Every readout is a function of the head’s logits; the standard model is recovered by $u \equiv \varepsilon$ and $\tau \equiv 1$
<table><tr><td>stage</td><td>standard</td><td>LucidWM</td><td>fallback</td></tr><tr><td>prediction, Eq. (4) training, Eq. (5)</td><td> $( 1 - \varepsilon ) \operatorname { s o f t m a x } ( \ell ) + \varepsilon / K$  fit only</td><td> $( 1 - u ) \hat { p } + u / K , u \mathrm { f r o m } S$  fit and evidence discipline e ⊕ eobs</td><td>uniform  $1 / K$  the vacuous opinion none: doubt never rises</td></tr></table>

## C Why the doubt reads experience

Section 3.1 reads the total evidence behind a prediction as the experience the head brings to a scene and action. This appendix gives the grounds in five steps (Table 4), with one example pair, in green boxes, followed through all of them. Every result treats the evidence of each scene–action pair on its own and its target as fixed; App. C.4 ends with what carries over to a network with shared parameters.

Table 4: The argument of App. $\mathrm { C } ,$ one result per step.
<table><tr><td>step</td><td>result</td><td>where</td></tr><tr><td>measure</td><td>the doubt  $u = W / ( W + S )$  sees the evidence only through its total S</td><td>Eq. (3)</td></tr><tr><td>ideal</td><td>in a table updated by Bayes&#x27; rule, S is the joint count; an untaken pair has  $u = 1$ </td><td>Prop. 2</td></tr><tr><td>obstacle</td><td>neither a readout of the prediction nor the fit can see S</td><td>Prop. 3</td></tr><tr><td>remedy</td><td>the discipline keeps the most doubtful opinion with its mean;  $u = 1$  if untaken</td><td>Prop. 4</td></tr><tr><td>payoff</td><td>the doubt sets the expected error of the prediction</td><td>Prop.5</td></tr></table>

## C.1 The doubt measures the evidence total

Notation. An opinion over $K \geq 2$ classes with evidence $e \geq 0$ is the Dirichlet $\omega \sim \operatorname { D i r } ( \alpha )$ of Eq. (3), with

$$
S = \sum _ { i } e _ { i } , \qquad \alpha = e + { \frac { W } { K } } { \bf 1 } , \qquad \alpha _ { 0 } = W + S , \qquad u = { \frac { W } { \alpha _ { 0 } } } ,
$$

$$
\hat { p } = \frac { e } { S } , \qquad P = \mathbb { E } [ \omega ] = \frac { \alpha } { \alpha _ { 0 } } = ( 1 - u ) \hat { p } + \frac { u } { K } .
$$

A target q is the distribution the fit pulls the prediction toward (in training, the posterior); the fit is $\mathrm { K L } ( q \parallel P )$ ; a readout is any quantity computed from P. We say count for the number N of transitions taken from a pair and evidence total for S.

The doubt is the share of the mean P held by the prior: the uncertainty mass of SL (Jøsang, 2016), whose belief masses $e _ { i } / \alpha _ { 0 }$ make up the rest. It depends on the evidence only through S, is one at $\bar { S } = 0$ and falls strictly as S grows. By conjugacy, observed counts add to $\alpha ,$ so evidence adds; this is the cumulative fusion of App. D.1.

Proof of Prop. 1. A head of the form Eq. (1) is Eq. (4) with $\hat { p } = \operatorname { s o f t m a x } ( \ell )$ and u = ε, that is, the opinion with the single total $S _ { \varepsilon } = W ( 1 - \varepsilon ) / \bar { \varepsilon }$ at every input. With the customary $W = 2$ (Jøsang, 2016) and Dreamer $\mathrm { V } \bar { 3 ^ { \circ } \mathrm { s } } \varepsilon = 0 . 0 1 , S _ { \varepsilon } = \mathrm { i } 9 8$

Example. Take $K = 3$ and $W = 2 ,$ and a pair taken $N = 8$ times whose next latent fell in the three classes 6, 2 and 0 times. Then $e = ( 6 , 2 , 0 ) , \mathbf { \bar { \it S } } = 8$ and $u = 0 . 2$ : the prior holds a fifth of the prediction $P \approx ( 0 . 6 7 , 0 . 2 7 , 0 . 0 7 )$

## C.2 In a table, the doubt counts experience

Proposition 2 (Doubt counts joint experience). Give each scene–action pair $( s , a )$ its own opinion ${ \omega ( \tilde { s , a ) } \sim \mathrm { D i r } ( ( W / K ) \mathbf { 1 } ) }$ over one categorical variable of the next latent, independent across pairs, and update it by Bayes’ rule on the transitions experienced from it. After $N ( s , a , \bar { z ^ { \prime } } )$ transitions $( s , a ) \to z ^ { \prime }$ for each $z ^ { \prime } .$

$$
e ( s , a ) = N ( s , a , \cdot ) , \qquad u ( s , a ) = { \frac { W } { W + N ( s , a ) } } , \qquad N ( s , a ) = \sum _ { z ^ { \prime } } N ( s , a , z ^ { \prime } ) .\tag{11}
$$

The doubt depends on experience only through the joint count $N ( s , a )$ , and is one for a pair never taken, however often its scene and its action were seen apart.

Proof. The priors are independent across pairs and the likelihood $\begin{array} { r } { \prod _ { ( s , a ) } \prod _ { z ^ { \prime } } \omega _ { z ^ { \prime } } ( s , a ) ^ { N ( s , a , z ^ { \prime } ) } } \end{array}$ factorises over pairs. By conjugacy the posterior of $( s , a )$ is $\mathrm { D i r } \big ( N ( s , a , \cdot ) + ( W / K ) \mathbf { 1 } \big )$ , so $e ( s , a ) = N ( s , a , \cdot )$ $S = N ( s , a )$ and $u = W / ( W + N ( s , a ) )$ ; no count of another pair enters.

Example. The evidence $e = ( 6 , 2 , 0 )$ is exactly the counts; a pair never taken has $e = 0$ and $u = 1$

In words. A familiar scene and a familiar action that were never taken together, the case of Fig. 2, therefore have doubt one. A network is not a table: it shares one function $e ( s _ { t } , a _ { t } )$ across pairs and is trained by a fit, which, as the next step shows, cannot recover the count.

## C.3 The prediction and the fit cannot see the evidence total

Many opinions share one prediction (Fig. 9a).

(a)  
![](images/9bd9ccbc77505bc981230e431081bb7db007b1e2d597db9fd9364c6588d8b4e3.jpg)

(b)  
![](images/330e7e9d11baec0b64c65cdac20f240ffe189dd8563d95d3cd19dad0511d0e90.jpg)

![](images/222dbf8ee828b01ce93c4c9152a590e1dbc79585870b0a8f60738962105d6807.jpg)  
Figure 9: One prediction, many evidence totals, for the example pair. (a) Dirichlets with the same mean P (dot) and totals $S = 8 ,$ 18 and 198. (b) Change from $S = 8$ along this family: the fit and every readout stay constant (Prop. 3), while the discipline grows with the total (Lemma 1); among opinions with this mean it prefers the smallest total, $S = 8 ,$ , the count. (c) The expected error falls with the total (Prop. 5); at $S = 1 9 8$ the total a standard head asserts, it is eighteen times smaller than at the count.

Proposition 3 (The prediction and the fit are blind to the total). Fix a prediction $P$ with full support and let $m = K$ min<sub>i</sub> $P _ { i }$

(i) For every $u \in ( 0 , m ]$ , the opinion Dir $\left( ( W / u ) P \right)$ has mean $P$ and doubt $u ,$ and it is the only one; no opinion with mean $P$ has doubt above $m$

(ii) Every readout, and the fit to any fixed target $q ,$ take the same value on all of these opinions.

Proof. An opinion with doubt u has $\alpha _ { 0 } = W / u ,$ so its mean is $P$ exactly when $\alpha = ( W / u ) P$ , that is, when

$$
e = { \frac { W } { u } } P - { \frac { W } { K } } { \bf 1 } ,
$$

which is nonnegative exactly when $u \leq m$ . Readouts and the fit depend on the opinion only through $P .$

Example. Here $m = 0 . 2 .$ , and the opinions with $S = 8 ,$ 18 and 198 all have mean $P \left( { \mathrm { F i g . 9 a } } \right)$ , so every readout and every fit takes one value on them $\left( \mathrm { F i g . 9 b } \right)$

In words. The readouts of Table 1 that are read off the prediction therefore cannot tell the unexperienced pair of Fig. 2 from an experienced pair with the same sharp prediction. The fit cannot set the total either: its gradient in u vanishes over the whole interval $( 0 , m ]$ . This is the non-identifiability of second-order learners whose loss sees only the predictive mean (Jurgens et al.¨ , 2024, Thm. 3.2), while losses averaged over the second-order distribution collapse it to a point mass (Bengs et al., 2022, Thm. 1). Something other than the fit must set the total.

## C.4 The discipline makes the doubt count experience

The evidence discipline sets the total. We model training in the form of Prop. 2: the discipline takes the place of the prior, charged once on every pair, and the fit the place of the likelihood, charged once per experienced transition. The discipline pulls every pair toward the vacuous opinion; Prior Networks pull toward a flat Dirichlet, and only on inputs out of distribution (Malinin & Gales, 2018). A pair with N experienced transitions and target q is then trained by

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { N } ( e ) = N \operatorname { K L } \bigl ( q \parallel P \bigr ) + \beta _ { u } D ( \alpha ) , } \\ { \displaystyle D ( \alpha ) = \operatorname { K L } \bigl ( \operatorname { D i r } ( \alpha ) \parallel \operatorname { D i r } ( \frac { W } { K } { \mathbf 1 } ) \bigr ) = \log \frac { \Gamma ( \alpha _ { 0 } ) \Gamma ( \frac { W } { K } ) ^ { K } } { \Gamma ( W ) \prod _ { i } \Gamma ( \alpha _ { i } ) } + \sum _ { i } e _ { i } \bigl [ \psi ( \alpha _ { i } ) - \psi ( \alpha _ { 0 } ) \bigr ] , } \end{array}
$$

with ψ the digamma function. By Prop. 3 an opinion is fixed by its mean $P$ and its strength $\alpha _ { 0 } = W + S \geq$ $t _ { P } : = W / ( K \operatorname* { m i n } _ { i } P _ { i } )$ , and the fit depends on , and the fit depends on $\bar { P }$ alone. The discipline grows with the total: ith the total

Lemma 1 (The discipline grows with the evidence). (i) At a fixed mean $P , D$ is strictly increasing in $\alpha _ { 0 }$   
on $[ t _ { P } , \infty )$   
(ii) $\begin{array} { r } { \mathbf { \bar { \cal D } } \ge 0 , } \end{array}$ , with equality only at $e = 0 .$   
Proof. Idea. At a fixed mean, the derivative of $D$ in the total is a positively weighted sum of $\xi ( \alpha _ { i } ) - \xi ( \alpha _ { 0 } )$   
for a decreasing ξ.   
Differentiating gives $\partial D / \partial \alpha _ { i } = e _ { i } \psi ^ { \prime } ( \alpha _ { i } ) - S \psi ^ { \prime } ( \alpha _ { 0 } )$ . The function $\xi ( y ) = y \psi ^ { \prime } ( y )$ is strictly decreasing,   
since integrating $\begin{array} { r } { \psi ^ { \prime } ( y ) = \int _ { 0 } ^ { \infty } t e ^ { - y t } ( 1 - e ^ { - t } ) ^ { - 1 } d t } \end{array}$ (Abramowitz & Stegun, 1964, Eq. 6.4.1) by parts gives   
$\begin{array} { r } { \xi ( y ) = 1 + \int _ { 0 } ^ { \infty } e ^ { - y t } h ^ { \prime } ( t ) \dot { { \boldsymbol { \omega } } } } \end{array}$ t with $h ( t ) = t / ( 1 - e ^ { - t } )$ increasing, so also $\xi > 1 ;$ and every $\alpha _ { i } < \alpha _ { 0 }$ . Hence   
$\frac { d D } { d \alpha _ { 0 } } \Big | _ { \alpha = \alpha _ { 0 } P } = \frac { 1 } { \alpha _ { 0 } } \sum _ { i } e _ { i } \big [ \xi ( \alpha _ { i } ) - \xi ( \alpha _ { 0 } ) \big ] > 0$   
wherever e $\neq 0 .$ . (ii) is Gibbs’ inequality.   
Proposition 4 (The discipline sets the evidence total). Let q have full support and $\beta _ { u } > 0 . ~ \mathcal { L } _ { N }$ has a   
minimiser, and every minimiser has $u ^ { * } = K \operatorname* { m i n } _ { i } P _ { i } ^ { * } \colon$ it is the most doubtful opinion with its own mean.   
Moreover:   
(i) without experience, $N = 0 ,$ , the doubt is $u ^ { * } = 1 ;$   
(ii) with experience, $u ^ { * } \geq K \operatorname* { m i n } _ { i } q _ { i } ;$ for $q$ not uniform, $u ^ { * } - K$ mi $\mathfrak { 1 } _ { i } q _ { i } = \Theta ( \beta _ { u } / N )$ as $\beta _ { u } / N \to 0 .$   
Proof. Idea. The fit sees only the mean and the discipline grows with the total, so the minimiser takes the   
least total its mean allows; comparing it with the opinion that fits $q$ exactly then bounds the excess. Let   
$F ( P ) = \mathrm { K L } ( q \parallel P ) , D _ { \mathrm { m i n } } ( P ) \stackrel { . } { = } D ( \stackrel { . } { t _ { P } } P )$ and $\kappa = \dot { \beta } _ { u } / N .$   
Reduce to the mean. Every opinion with mean $P$ has $\alpha _ { 0 } \geq t _ { P } ,$ so by Lemma 1(i)   
$\begin{array} { r } { \mathcal { L } _ { N } ( e ) \ : \geq \ : \Phi _ { N } ( P ) : = N F ( P ) + \beta _ { u } D _ { \mathrm { m i n } } ( P ) , } \end{array}$   
with equality exactly at $\alpha _ { 0 } = t _ { P } . \mathrm { ~ A ~ }$ minimiser therefore has the smallest total its mean allows: its least   
likely class holds no evidence, and $u ^ { * } = K \operatorname* { m i n } _ { i } P _ { i } ^ { * } . \Phi _ { N }$ is continuous and, as q has full support, grows   
without bound as min $P _ { i } \to 0 ,$ so a minimiser exists.   
(i). At $N = 0$ the loss is $\beta _ { u } D$ , which vanishes only at $e = 0$ (Lemma 1(ii)).   
(ii), upper bound. The opinion with mean q and doubt K min q (Prop. 3) has loss $\beta _ { u } D _ { \mathrm { m i n } } ( q ) .$ so $F ( P ^ { * } ) \leq$   
$\kappa [ D _ { \mathrm { m i n } } ( q ) \mathring { \ } _ { - } ^ { { \cal - } } D _ { \mathrm { m i n } } ( P ^ { \mathring { \ast } } ) ] \leq \kappa D _ { \mathrm { m i n } } ( q )$ , and $P ^ { * } \to q$ as $\kappa  0$ . Near $q , D _ { \mathrm { m i n } } \mathrm { ~ i }$ s Lipschitz and $\dot { F } \geq$   
$\frac { 1 } { 2 } \bar { \| } P ^ { * } - \dot { q } \| _ { 1 } ^ { 2 }$ (Pinsker), so $\| P ^ { * } - q \| _ { 1 }$ and hence $\boldsymbol { u } ^ { * } - \boldsymbol { K }$ min $\begin{array} { r } { \mathrm { { } } _ { i } q _ { i } \le K \| P ^ { * } - q \| _ { 1 } } \end{array}$ are $O ( \kappa )$   
(ii), lower bound. Let $N > 0$ and $q$ be non-uniform (for uniform $q , e ^ { * } = 0$ and $u ^ { * } = 1 = K$ min<sub>i</sub> $q _ { i } ) ;$ then   
$S ^ { * } > 0$ . At the class j left without evidence, raising $e _ { j }$ cannot lower the loss, the first inequality below;

multiplying it by $\alpha _ { 0 } ^ { * } / N$ gives the second:

$$
N \Big ( \frac { 1 } { \alpha _ { 0 } ^ { * } } - \frac { K q _ { j } } { W } \Big ) - \beta _ { u } S ^ { * } \psi ^ { \prime } ( \alpha _ { 0 } ^ { * } ) \geq 0 \Longrightarrow 1 - \frac { K q _ { j } } { u ^ { * } } \geq \kappa S ^ { * } \xi ( \alpha _ { 0 } ^ { * } ) > \kappa S ^ { * } ,
$$

using $\xi > 1$ (proof of Lemma 1). Hence $u ^ { * } = K P _ { i } ^ { * } > K q _ { j } / ( 1 { - } \kappa S ^ { * } ) \geq K$ min<sub>i</sub> $q _ { i } \left( 1 + \kappa S ^ { * } \right)$ . By the upper bound $u ^ { * } \to K$ min q , so $K \operatorname* { m i n } _ { i } q _ { i } S ^ { * }  W ( \bar { 1 } - K \operatorname* { m i n } _ { i } q _ { i } ) > 0$ and the excess is at least of order κ.

Example. With $\beta _ { u } = 0 . \mathrm { { \ i } }$ 1 the discipline settles the pair at $u ^ { * }$ ≈ 0.24 against the count’s 0.2, and returns $u ^ { * } = 1$ for a pair never taken.

In words. When the target is the posterior mean of Prop. 2 after N transitions, K min $q _ { i } \ge W / ( W + N )$ the doubt of Eq. (11), with equality when some class was never observed. By (ii), the discipline then never leaves an experienced pair more confident than the count, and errs toward doubt: by an excess of order $\beta _ { u } / N$ and by more where every class was observed. Regularised evidential learners take, among equal fits, the one the regulariser prefers (Jurgens et al.¨ , 2024, Sec. 3.4), and their uncertainty is then set by the regularisation weight (Bengs et al., 2022; Jurgens et al.¨ , 2024); here the weight sets only an excess of order $\beta _ { u } \mathbf { \bar { / } } N$ above a level fixed by the target.

With a network, the evidence of every pair is one function $e ( s _ { t } , a _ { t } )$ of shared parameters. Posterior Networks keep the count of Prop. 2 in such a function with a learned density, setting the evidence of a class to its count times the density at the input (Charpentier et al., 2020); the lucid head needs no density model. The fit rewards a direction that generalises across pairs but never rewards evidence away from the transitions that demand it, so through the shared parameters the discipline lowers the total wherever no transition holds it up. Sec. 4 measures this on four bases: after observation and before decision (Table 1), along a rollout that leaves the experienced states (Fig. 4), and as experience accumulates (Fig. 6).

## C.5 The doubt sets the expected error

Proposition 5 (Doubt sets the expected error of the prediction). Let $\omega \sim \operatorname { D i r } ( \alpha )$ be the model’s opinion on the distribution of a next latent variable, with mean P and doubt u. Then

$$
\mathbb { E } \| \omega - P \| _ { 2 } ^ { 2 } = \left( 1 - \| P \| _ { 2 } ^ { 2 } \right) \frac { u } { W + u } ,\tag{12}
$$

the share $u / ( W + u )$ of the largest expected error, $1 - \| P \| _ { 2 } ^ { 2 }$ , that any belief with mean P can carry. For a fixed prediction it rises strictly with the doubt; a standard head gives every transition with that prediction the same share.

Proof. A Dirichlet has $\mathrm { V a r } [ \omega _ { i } ] = P _ { i } ( 1 - P _ { i } ) / ( \alpha _ { 0 } + 1 )$ , so $\mathbb { E } \Vert \omega - P \Vert _ { 2 } ^ { 2 } = ( 1 - \Vert P \Vert _ { 2 } ^ { 2 } ) / ( \alpha _ { 0 } + 1 )$ , and $\alpha _ { 0 } + 1 = ( W + u ) / u$ . Any belief with mean P has $\begin{array} { r } { \mathbb { E } \| \omega - \dot { P } \| _ { 2 } ^ { 2 } = \mathbb { E } \| \bar { \omega } \| _ { 2 } ^ { 2 } - \| P \| _ { 2 } ^ { 2 } \leq 1 - \| P \| _ { 2 } ^ { 2 } } \end{array}$ , with equality for the belief on the vertices. For fixed $P , 1 - \| \mathbf { \ddot { } } P \| _ { 2 } ^ { 2 } > \ddot { 0 }$ and $u / ( W + u )$ rises with u; a standard head has $\alpha _ { 0 } = W / \varepsilon$ everywhere.

Example. Here $1 - \| P \| _ { 2 } ^ { 2 } = 0 . 4 8 \colon$ the expected error is 0.044 at the count, $S = 8 ,$ , but 0.0024 at $S = 1 9 8 ,$ eighteen times smaller for the same prediction (Fig. 9c).

The error is taken under the model’s own opinion; Fig. 6 tests whether that opinion is calibrated, and App. D.5 says what calibration buys in the return.

## D Why trust is carried in the return

Section 3.2 carries the doubt of each imagined step as trust in the return, and Sec. 3.3 learns and decides by that trust. Carried through the imagined state instead, the doubt of a step would be lost one step later; carried in the return, it compounds, keeps the critic’s fixed point and discounts every model error that follows a doubtful step. Table 5 lays out the argument; Apps. D.1–D.5 each give an example in a green box, and Apps. D.2–D.5 follow one imagined rollout.

Notation. Symbols are those of Table 2. A rollout starts at a real state s<sub>0</sub> and takes H imagined steps; step k has the doubt $\bar { u } _ { k } \ge \varepsilon$ of the transition $\left( s _ { k - 1 } , a _ { k - 1 } \right)$ and the trust $\tau _ { k } \in [ 0 , 1 ]$ , and $C _ { 0 } = 1$ . We take $\lambda \in ( 0 , 1 ]$ and $\gamma \in ( 0 , 1 )$ , and write $M _ { n } = \lambda ^ { n } C _ { n }$ for the share of the return that reaches step n. A doubt profile isfixed when $\bar { u } _ { 1 } , \dots , \bar { u } _ { H }$ are the same on every imagined rollout. Prop. 6 uses the doubt before the floor, $u = W / ( W + S )$ , as in $\mathsf { A p p . C }$

Table 5: The argument of App. D, step by step.
<table><tr><td>step</td><td>result</td><td>where</td></tr><tr><td>start</td><td>at the real state, fusing in an observation never raises the doubt</td><td>Prop. 6</td></tr><tr><td>alternative</td><td>carried in the state, the doubt of a step is lost one step later</td><td>Prop. 7</td></tr><tr><td>choice</td><td>carried in the return, trust compounds, as SL trust discounting does</td><td>Prop. 8</td></tr><tr><td>safety</td><td>the lucid return keeps the critic&#x27;s fixed point</td><td>Thm. 1</td></tr><tr><td>payoff</td><td>errors that follow a doubtful step are discounted, where it pays</td><td>Prop.9</td></tr><tr><td>action</td><td>doubt ranks candidate actions before any is taken</td><td>Prop. 10</td></tr></table>

## D.1 Observation never raises the doubt

Proposition 6 (Fusion never raises the doubt). Fuse a prediction (evidence e, doubt u) with an observation   
(evidence $e _ { \mathrm { o b s } } ,$ , total $S _ { \mathrm { o b s } } ,$ doubt $u _ { \mathrm { o b s } } = W / ( W + S _ { \mathrm { o b s } } ^ { \mathrm { ^ { - } } } ) )$ by cumulative fusion (Jøsang, 2016, Sec. 12.3),   
$e _ { \mathrm { p o s t } } = e + e _ { \mathrm { o b s } }$ , and let $u _ { \mathrm { p o s t } }$ and $P _ { \mathrm { p o s t } }$ be the doubt and mean of the result. Then   
(i) u<sub>post</sub> ≤ min $( u , u _ { \mathrm { o b s } } )$ , and $1 / u _ { \mathrm { p o s t } } \stackrel { . } { = } 1 / u + S _ { \mathrm { o b s } } / W ;$   
(ii) $\dot { P } _ { \mathrm { p o s t } } = \left( 1 - \theta \right) P + \theta \hat { p } _ { \mathrm { o b s } } ,$ , with $\hat { p } _ { \mathrm { o b s } } = e _ { \mathrm { o b s } } / S _ { \mathrm { o b s } }$ and $\theta = 1 - u _ { \mathrm { p o s t } } / u$ the share of the doubt the   
observation removes; a vacuous observation, $S _ { \mathrm { o b s } } = 0 ,$ , has $\theta = 0$ and leaves the prediction unchanged.   
Proof. The total $S + S _ { \mathrm { { o b s } } }$ is at least S and at least $S _ { \mathrm { o b s } }$ , and the doubt falls with the total; $W / u _ { \mathrm { p o s t } } = W +$   
$S + S _ { \mathrm { o b s } } = W / u + S _ { \mathrm { o b s } } , \mathrm { F o r ~ ( i i ) } , \alpha _ { \mathrm { p o s t } } = \alpha + \epsilon _ { \mathrm { o b s } } , \mathrm { s o } P _ { \mathrm { p o s t } } = \left( \left( W + S \right) P + S _ { \mathrm { o b s } } \hat { p } _ { \mathrm { o b s } } \right) / \left( W + S + S _ { \mathrm { o b s } } \right)$   
and $S _ { \mathrm { o b s } } / ( W + S + S _ { \mathrm { o b s } } ) = 1 - u _ { \mathrm { p o s t } } / u .$

Example. Fuse the example pair of App. $\mathrm { ~ C ~ } , e = ( 6 , 2 , 0 )$ and $u = 0 . 2$ , with an observation $e _ { \mathrm { o b s } } = ( 4 , 0 , 0 )$ $u _ { \mathrm { o b s } } = 0 . 3 3$ . Then $e _ { \mathrm { p o s t } } = ( 1 0 , 2 , 0 )$ and $u _ { \mathrm { p o s t } } = 0 . 1 4$ , below both. The observation removes $\theta = 0 . 2 9$ of the doubt and moves $P \approx ( \mathrm { 0 . 6 7 , 0 . 2 7 , 0 . 0 7 } )$ 29% of the way to $( 1 , 0 , 0 )$ , to $P _ { \mathrm { p o s t } } \approx ( 0 . 7 6 , 0 . 1 9 , 0 . 0 5 )$

These identities are specific to cumulative fusion. The reduced Dempster rule of evidential multi-view learning also never raises the uncertainty mass (Han et al., 2023, Prop. 3.3), while averaging fusion, which averages the evidence of the views (Xu et al., 2024), can raise it above the smaller of the two.

## D.2 In the state, the lineage is lost

One could carry the doubt in the imagined state, deducing the opinion on each imagined state from the opinion on the state before, as the deduction of SL does (Jøsang, 2016, Ch. 9).

Proposition 7 (Deduction forgets the lineage). Deduce the opinion on each imagined state from the opinion on the one before, which has doubt $\tilde { u } _ { k - 1 }$ , with $\tilde { u } _ { 0 } = 0$ at the real state. Take the doubt of the step’s conditionals as $\bar { u } _ { k } ,$ , the only doubt the head reports. Then

$$
\tilde { u } _ { k } = \bar { u } _ { k } + \sigma _ { k } \tilde { u } _ { k - 1 } = \sum _ { j = 1 } ^ { k } \Big ( \prod _ { l = j + 1 } ^ { k } \sigma _ { l } \Big ) \bar { u } _ { j } , \qquad \sigma _ { k } = u _ { k } ^ { \circ } - \bar { u } _ { k } ,
$$

where $u _ { k } ^ { \circ }$ is the doubt that deduction assigns to state k when state $k - 1$ is vacuous. So $\tilde { u } _ { 1 } ~ = ~ \bar { u } _ { 1 }$ $| \tilde { u } _ { k } - \bar { u } _ { k } ^ { \cdot \cdot } | = | \sigma _ { k } | \tilde { u } _ { k - 1 } \leq | \sigma _ { k } |$ at every depth, and the doubt of an earlier step j reaches depth k only through the factor $\textstyle \prod _ { l = j + 1 } ^ { k } \sigma _ { l } :$ where $| \sigma |$ is small, a doubtful step is forgotten one step later.

Proof. Deduction gives the deduced opinion the doubt $\begin{array} { r } { u _ { Y \parallel X } = u _ { X } u _ { Y \parallel \hat { X } } + \sum _ { i } u _ { Y \mid x _ { i } } b _ { X } ( x _ { i } ) } \end{array}$ (Jøsang, 2016, Eq. (9.75)), where $u _ { X }$ and $b _ { X }$ are the doubt and belief masses of the antecedent and $u _ { Y \parallel \hat { X } }$ is the doubt deduced from a vacuous antecedent. With $u _ { X } = \tilde { u } _ { k - 1 } , u _ { Y \parallel \hat { X } } = u _ { k } ^ { \circ } , u _ { Y | x _ { i } } = \bar { u } _ { k }$ and $\begin{array} { r } { \sum _ { i } b _ { X } ( \ddot { x } _ { i } ) = 1 - \tilde { u } _ { k - 1 } } \end{array}$ this is $\tilde { u } _ { k } = u _ { k } ^ { \circ } \tilde { u } _ { k - 1 } + \bar { u } _ { k } \left( 1 - \tilde { u } _ { k - 1 } \right)$ , the first form. Unrolling from $\tilde { u } _ { 0 } = 0$ gives the second, and $\tilde { u } _ { k - 1 } \leq 1$ gives the bound.

Example. Take a rollout of $H = 8$ imagined steps that leaves experience at steps 3 and 4, $\bar { u } _ { 3 } = \bar { u } _ { 4 } = 0 . 4 0 .$ and is familiar elsewhere, $\bar { u } _ { k } = 0 . 0 1$ ; with $\varepsilon = 0 . 0 1$ the trusts are 0.61 at steps 3 and 4 and 1 elsewhere. With $| \sigma _ { k } | \le 0 . 0 1$ , deduction gives states 3 and 4 the doubt 0.40 and states 5 to 8 the doubt 0.01, each to within 0.005: states 5 to 8 are trusted again (Fig. 10a).

In words. On a trained model on cheetah run, $\sigma = 0 . 0 0 9$ at the median, and exact deduction gives the same doubt at depth fifteen as at depth one, where compounding would have raised it. The recursion settles after one step at a level set by the current step, and the lineage of the rollout is lost.

## D.3 In the return, the doubt compounds

Carried in the return instead, trust compounds along the rollout.

Proposition 8 (Carried trust is the trust discounting of SL). Read the k-th imagined step as an advisor in a referral chain that starts at the real state $s _ { 0 } ,$ and its trust $\tau _ { k }$ as the projected probability of the binomial opinion that this step is reliable. Then the carried trust $\begin{array} { r } { C _ { n } = \prod _ { k = 1 } ^ { n } \hat { \tau } _ { k } } \end{array}$ is the projected probability that the transitive trust discounting of SL (Jøsang, 2016, Sec. 14.3.4, Def. 14.7) assigns to the chain of the first n steps, and an opinion with belief masses b reported at depth n is discounted to belief $C _ { n } b$ and uncertainty $1 { \stackrel { - } { - } } C _ { n } \sum _ { i } b _ { i }$ . With $\tau \equiv 1$ the chain is fully trusted and nothing is discounted.

Proof. Transitive discounting multiplies the projected probabilities of the referral edges along the path (Jøsang, 2016, Eq. (14.13)) and scales the belief of the final opinion by that product (Jøsang, 2016, Def. 14.7). With edge k at $\tau _ { k }$ the product is $C _ { n }$ . Only the projected probabilities enter, so the result does not depend on how each $\tau _ { k }$ is split into belief, disbelief and uncertainty.

Example. In the return, the rollout carries the trust $C _ { n } = 1$ up to step 2, 0.61 at step 3 and 0.37 from step 4 on: states 5 to 8 inherit the doubt of steps 3 and 4, which deduction dropped (Fig. 10a).

In words. Deduction through the state and discounting along the chain are the two ways SL chains opinions. The first forgets a doubtful step once it is passed, the second carries it to every step built on it (Fig. 10a); this is why LucidWM carries doubt as trust in the return.

(a)  
![](images/cd9a97794e55c663de6de1873a39a2fcdeaf0119804ba7cb81903ffed0b3bf35.jpg)

(b)  
![](images/a15719ded7e5ad65720cd6860d71c0d6eb4813089023697e5af8f42ac5a88de1.jpg)  
Figure 10: Trust in the state loses the lineage; trust in the return compounds, for the example rollout. (a) Trust at each imagined step n: deduced through the state, $( 1 - \tilde { u } _ { n } ) / ( 1 - \varepsilon )$ with $\sigma = 0 . 0 0 9$ (Prop. 7), and carried in the return, $C _ { n } \left( \mathrm { P r o p . ~ } 8 \right)$ . (b) Weights of the one- to eight-step returns under the λ-return and the lucid return (Lemma 2).

## D.4 The lucid return keeps the fixed point and discounts model error

The lucid return mixes the n-step returns; Lemma 2 gives the mixture, and Thm. 1 what it does to learning.

Lemma 2 (Unrolled lucid return). For every doubt profile, Eq. (7) unrolls to Eq. (8): the n-step return ${ \cal R } ^ { ( n ) }$ has weight $w _ { n } = M _ { n - 1 } - M _ { n }$ for $n < H$ and $w _ { H } = M _ { H - 1 }$ . The weights are nonnegative and sum to one, and the returns longer than n steps carry total weight $M _ { n }$ . With $\tau \equiv 1 , w _ { n } = ( 1 \bar { - } \lambda ) \lambda ^ { n - 1 }$ $w _ { H } = \lambda ^ { H - 1 }$ and $R _ { t } ^ { \tau } = R _ { t } ^ { \lambda }$ (Cor. 1).

Proof. Since $R _ { H } ^ { \tau } = v ( s _ { H } )$ , the last step reads $R _ { H - 1 } ^ { \tau } = r _ { H - 1 } + \gamma v ( s _ { H } )$ whatever $\tau _ { H }$ . Unrolling Eq. (7) from $t = 0$

$$
R _ { 0 } ^ { \tau } = \sum _ { k = 0 } ^ { H - 1 } \gamma ^ { k } M _ { k } r _ { k } + \sum _ { n = 1 } ^ { H - 1 } \gamma ^ { n } M _ { n - 1 } ( 1 - \lambda \tau _ { n } ) v ( s _ { n } ) + \gamma ^ { H } M _ { H - 1 } v ( s _ { H } ) .
$$

In $\textstyle \sum _ { n } w _ { n } R ^ { ( n ) }$ the reward $r _ { k }$ collects $\begin{array} { r } { \sum _ { n > k } w _ { n } = M _ { k } } \end{array}$ and $v ( s _ { n } )$ collects $w _ { n } ,$ and $M _ { n - 1 } ( 1 - \lambda \tau _ { n } ) =$ $M _ { n - 1 } ^ { \cdots } - M _ { n } ,$ so the two agree. The weights are nonnegative as $\lambda \tau _ { n } \leq 1$ , telescope to $M _ { 0 } = 1$ , and those beyond n sum to $M _ { n } . \mathrm { { \bf ~ W i t h } } \tau \equiv 1 , M _ { n } \stackrel { \_ } { = } \lambda ^ { n }$

Example. With $\lambda = 0 . 9 5$ , the weight of the eight-step return falls from 0.70 under the λ-return to 0.26, and the three- and four-step returns, which hand over to the critic at steps 3 and 4, gain 0.34 and 0.18 (Fig. 10b).

Theorem 1 (Contraction and bias bound, restated and extended). For any fixed doubt profile: (i) the critic’s update under Eq. (7) is a γ-contraction to ${ \hat { V } } ^ { \pi }$ , the value of π in the imagined model, as under Eq. (2); (ii) with the critic at the true value $V ^ { \pi }$ and $\delta _ { j } = \operatorname* { s u p } \lvert ( { \hat { T } } - T ) V ^ { \pi } \rvert$ over the states reached after $j$ imagined steps, where $\hat { T }$ and $T$ are the Bellman operators of π in the model and in the environment,

$$
\left\| \mathbb { E } [ R _ { 0 } ^ { \tau } ] - V ^ { \pi } \right\| _ { \infty } \leq \ \sum _ { j = 0 } ^ { H - 1 } \gamma ^ { j } \lambda ^ { j } C _ { j } \delta _ { j } ;
$$

(iii) when the trust varies with the state–action pair, (i) holds as it stands, and at each real state $s _ { 0 }$ , with $\eta = ( { \hat { T } } - T ) V ^ { \pi }$ the model’s one-step error,

$$
\big | \mathbb { E } [ R _ { 0 } ^ { \tau } \mid s _ { 0 } ] - V ^ { \pi } ( s _ { 0 } ) \big | \ \leq \ \sum _ { j = 0 } ^ { H - 1 } \gamma ^ { j } \lambda ^ { j } \mathbb { E } \big [ C _ { j } | \eta ( s _ { j } ) | \big | \ s _ { 0 } \big ] .
$$

Proof. Idea. By Lemma 2 the target is a mixture of n-step returns with weights set by the head; each part then follows from one backward recursion.

(i). Let $( \mathcal { R } v ) ( s _ { 0 } ) = \mathbb { E } [ R _ { 0 } ^ { \tau } \mid s _ { 0 } ]$ , over imagined rollouts from $s _ { 0 }$ under $\pi$ and the model. By Lemma 2, $\begin{array} { r } { R _ { 0 } ^ { \tau } = \sum _ { k = 0 } ^ { H - 1 } \gamma ^ { k } M _ { k } r _ { k } + \sum _ { n = 1 } ^ { H } \gamma ^ { n } w _ { n } v ( s _ { n } ) } \end{array}$ , so

$$
| \mathcal { R } v - \mathcal { R } v ^ { \prime } | ( s _ { 0 } ) \leq \mathbb { E } \sum _ { n = 1 } ^ { H } \gamma ^ { n } w _ { n } | v - v ^ { \prime } | ( s _ { n } ) \leq \gamma \| v - v ^ { \prime } \| _ { \infty } ,
$$

as $\gamma ^ { n } \leq \gamma$ and the weights sum to one. For the fixed point, Eq. (7) reads $R _ { t } ^ { \tau } = r _ { t } + \gamma v ( s _ { t + 1 } ) +$ $\gamma \lambda \tau _ { t + 1 } \big ( R _ { t + 1 } ^ { \tau } - v ( s _ { t + 1 } ) \big )$ . Let $v = { \hat { V } } ^ { \pi }$ and $\zeta _ { t } = \mathbb { E } [ R _ { t } ^ { \tau } \mid s _ { t } ] - { \hat { V } } ^ { \pi } ( s _ { t } )$ , with $\zeta _ { H } = 0 $ . As $r _ { t } + \gamma \hat { V } ^ { \pi } ( s _ { t + 1 } )$ has mean $\hat { V } ^ { \pi } ( s _ { t } )$ under the model, $\zeta _ { t } = \gamma \lambda \tau _ { t + 1 } \mathbb { E } [ \zeta _ { t + 1 } \mid s _ { t } ]$ , so every $\zeta _ { t }$ vanishes: ${ \hat { V } } ^ { \pi }$ is a fixed point, and by the contraction the only one.

(ii). Let $v = V ^ { \pi }$ and $\zeta _ { t } = \mathbb { E } [ R _ { t } ^ { \tau } \mid s _ { t } ] - V ^ { \pi } ( s _ { t } )$ . Now $r _ { t } + \gamma V ^ { \pi } ( s _ { t + 1 } )$ has mean $( { \hat { T } } V ^ { \pi } ) ( s _ { t } ) = V ^ { \pi } ( s _ { t } ) +$ $\eta ( s _ { t } )$ , since $T V ^ { \pi } = V ^ { \pi }$ . So $\zeta _ { t } = \eta ( s _ { t } ) \substack { + \gamma \lambda \tau _ { t + 1 } \mathbb { E } [ \zeta _ { t + 1 } \mid s _ { t } ] } ,$ , which unrolls to $\begin{array} { r } { \zeta _ { 0 } = \sum _ { j = 0 } ^ { H - 1 } \gamma ^ { j } M _ { j } \mathbb { E } [ \eta ( s _ { j } ) } \end{array}$ | $s _ { 0 } ]$ , and $\left| \mathbb { E } [ \eta ( s _ { j } ) \mid s _ { 0 } ] \right| \leq \delta _ { j }$

(iii). Both parts use only that the trust of a step is known at the state and action it leaves. The weights of Lemma 2 stay nonnegative and sum to one on every rollout, so (i) holds; with $\tau _ { t + 1 }$ <sub>1</sub> inside the expectation, (ii) unrolls to $\begin{array} { r } { \zeta _ { 0 } = \sum _ { j } \gamma ^ { j } \lambda ^ { j } \mathbb { E } [ C _ { j } \eta ( s _ { j } ) \mid s _ { 0 } ] } \end{array}$ , which gives the bound.

## D.5 When trust tightens the bound

With a critic off the true value, trust also sets how much of the return rests on the critic.

Proposition 9 (When lowering trust tightens the bound). Let the critic err by $\Delta _ { v } = \| v - V ^ { \pi } \| _ { \infty }$ . For any fixed doubt profile:

(i) the bias is bounded by

$$
\left. \mathbb { E } [ R _ { 0 } ^ { \tau } ] - V ^ { \pi } \right. _ { \infty } \leq \gamma \Delta _ { v } + \delta _ { 0 } + \sum _ { j = 1 } ^ { H - 1 } \gamma ^ { j } M _ { j } \big ( \delta _ { j } - ( 1 - \gamma ) \Delta _ { v } \big ) ,
$$

and lowering the trust $\tau _ { k }$ of one step, $1 \leq k \leq H - 1$ , with $C _ { k } > 0$ , lowers this bound exactly when the mean of $\delta _ { j }$ over $j \geq k$ , weighted by $\gamma ^ { j } M _ { j }$ , exceeds $( 1 - \gamma ) \Delta _ { v } ;$   
(ii) with the critic at the true value, $\Delta _ { v } = 0$ , this is the bound $\begin{array} { r } { B ( C ) = \sum _ { j = 0 } ^ { H - 1 } \gamma ^ { j } \lambda ^ { j } C _ { j } \delta _ { j } } \end{array}$ of Eq. (9), and against the standard return, $\begin{array} { r } { \tau \equiv 1 , B ( \mathbf { 1 } ) - B ( C ) = \sum _ { j = 0 } ^ { H - 1 } \gamma ^ { j } \lambda ^ { j } ( 1 - C _ { j } ) \delta _ { j } \geq 0 ; } \end{array}$ the error at each depth is discounted by the trust lost before it.

Proof. (i) By Lemma 2, replacing $V ^ { \pi }$ by v changes $R _ { 0 } ^ { \tau }$ ${ \begin{array} { r } { \sum _ { n = 1 } ^ { H } { \gamma ^ { n } w _ { n } \left( v \mathrm { ~ - ~ } V ^ { \pi } \right) \left( s _ { n } \right) } } \end{array} }$ , at most $\textstyle \Delta _ { v } \sum _ { n = 1 } ^ { H } \gamma ^ { n } w _ { n }$ in size, and summation by parts gives $\begin{array} { r } { \sum _ { n = 1 } ^ { H } \gamma ^ { n } w _ { n } = \gamma - ( 1 - \gamma ) \sum _ { j = 1 } ^ { H - 1 } \gamma ^ { j } M _ { j } } \end{array}$ . Adding Thm. 1(ii) gives the bound. Every $M _ { j }$ with $j \geq k$ is proportional to $\tau _ { k }$ and no other term depends on it, so the bound is linear in $\tau _ { k }$ , whose slope has the sign of $\begin{array} { r } { \sum _ { j \geq k } \gamma ^ { j } M _ { j } \big ( \delta _ { j } - ( 1 - \gamma ) \Delta _ { v } \big ) } \end{array}$ . (ii) Subtract the two sums; as $\bar { u } \ge \varepsilon , \tau \le 1$ , so $C _ { j } \leq 1$ , and $\delta _ { j } \geq 0$

Example. In the rollout, errors at depths 0 to 2, up to the first doubtful transition, keep their factors $M _ { j } = 1 , 0 . 9 5 , 0 . 9 0$ in the bound. Errors from depth 3 on, which follow the doubtful steps, enter with $M _ { 3 } = 0 . 5 2$ down to $M _ { 7 } = 0 . 2 6$ , where the λ-return has λ<sup>j</sup>, 0.86 down to 0.70.

In words. Every $M _ { j }$ with $j \geq k$ carries $\tau _ { k }$ , so one doubtful step discounts every model error that follows it and none before it; a smaller λ would discount every depth alike. As the doubt rises where the rollout leaves experience (Sec. 4.3), these are the errors made after it has left what the model knows. Prop. 9(i) says where the discount pays when the critic errs: where the model’s one-step error beyond a step exceeds $( 1 - \dot { \gamma } ) \Delta _ { v } .$ , the critic’s error per step. Which imagined steps the trust actually removes is measured in Sec. 4.5.

## D.6 Doubt ranks actions before acting

Proposition 10 (Doubt before acting). The doubt $u ( s _ { t } , a )$ is computed by the head from $( s _ { t } , a )$ alone, so it is known for every candidate action before any is taken. Above the floor ε $, u ( s _ { t } , a ) > \dot { u } ( s _ { t } , \dot { a } ^ { \prime } )$ exactly when $S ( s _ { t } , a ) < \dot { S ( \boldsymbol { s } _ { t } , a ^ { \prime } ) }$ , for every prior weight $W > 0$

Proof. The evidence $e = \mathfrak { s }$ oftplus $\ell ( s _ { t } , a )$ needs no observation of the outcome, and $u = W / ( W + S )$ falls strictly with S for every $W > 0$

What acting on trust changes in the agent’s decisions is measured in Sec. 4.6.

## E Experimental details

LucidWM is compared with seventeen readouts on four bases and three environment families (Table 7). This appendix gives the bases, readouts and protocols; $\mathbf { A p p . }$ . F gives the results beyond the main text.

## E.1 Protocols

Maze. The agent trains on a ViZDoom map with one room sealed off. The opened map unseals it; the repainted map keeps every wall in place and changes the paint of one room. The repainted dead end of Sec. 4.6 keeps every wall and repaints the deep half of one dead end. A walk is a scripted route that reads the agent’s position only, never the model, so that every readout sees the same frames. Learning to doubt leaves the task intact: every trained agent, with LucidWM and without, reaches the goal in all 100 test episodes.

Corrupted actions. From real start states the model imagines forward while each action is replaced, with a set probability, by a random one drawn uniformly from the action space. The lift of a readout is its value under fully corrupted actions over its value under the true ones.

Alarm line. Each readout is set against its own alarm line, taken on familiar frames of the same run; Figs. 11, 13, 14 and 15 show each readout in alarm units, 0 at its familiar level and 1 at its line.

## E.2 Four bases

Table 6 lists the four bases.

Table 6: The four bases; $\varepsilon = 0 . 0 1$ and $\lambda = 0 . 9 5$ on all. $G \times K \colon$ : categorical variables and classes of the latent state; W: prior weight; H: imagination horizon.
<table><tr><td>base</td><td>dynamics</td><td> $G \times K$ </td><td>W</td><td>H</td></tr><tr><td>DreamerV3</td><td>recurrent, pixel decoder</td><td>32 × 32</td><td>2</td><td>15</td></tr><tr><td>R2-Dreamer</td><td>recurrent, no reconstruction</td><td> $3 2 \times 1 6$ </td><td>1</td><td>15</td></tr><tr><td>EMERALD</td><td>masked transformer, spatial</td><td> $4 \times 4 \mathrm { c e l l s  o f 3 2 \times 3 2 }$ </td><td>2</td><td>15</td></tr><tr><td>OC-STORM</td><td>transformer</td><td> $3 2 \times 3 2$ </td><td>2</td><td>16</td></tr></table>

## E.3 Every experiment at a glance

Table 7: Every experiment in the paper and what it finds. DMC in Table 1(B): walker, cheetah, pendulum.
<table><tr><td>experiment</td><td>bases</td><td>environments</td><td>result</td></tr><tr><td colspan="4">main text</td></tr><tr><td>after observation, Tab. 1(A)</td><td>all four</td><td>maze</td><td>first on every base</td></tr><tr><td>before decision, Tab. 1(B)</td><td>all four</td><td>DMC, Crafter</td><td>first on all 16 rows</td></tr><tr><td>leaving experience, Fig. 4</td><td>DreamerV3</td><td>walker</td><td>9 in 10 over the line; base flat</td></tr><tr><td>two changed rooms, Fig. 5</td><td>DreamerV3</td><td>maze</td><td>alarm in both; base in neither</td></tr><tr><td>learning, Fig. 6</td><td>DreamerV3</td><td>cheetah</td><td>doubt falls with the error</td></tr><tr><td>trust as a filter, Fig. 7</td><td>DreamerV3</td><td></td><td>walker, cheetah, finger gain within depth on all three, up to 23%</td></tr><tr><td>acting on trust, Fig. 8</td><td>DreamerV3</td><td>maze</td><td>goal at step 190; base at 362</td></tr><tr><td colspan="4">appendix</td></tr><tr><td>new room, Fig. 11</td><td>DreamerV3</td><td>maze</td><td>25/29 walks; base 0/29</td></tr><tr><td>one pass vs. three models, Fig. 12</td><td>DreamerV3</td><td>maze</td><td>8/10 walks; ensembles 0/10</td></tr><tr><td>falls, Fig. 13</td><td>DreamerV3</td><td>cheetah</td><td>20/20 before onset; entropy 0/20</td></tr><tr><td>nightfall, Fig. 14</td><td>DreamerV3</td><td>Crafter</td><td>peak 2.6× its line; base &lt; 0.5×</td></tr><tr><td>every base, Fig. 15</td><td>all four</td><td>maze</td><td>alarm on all four bases</td></tr><tr><td>corruption ladder, Fig. 16</td><td>OC-STORM,</td><td>walker, cheetah,</td><td>over fivefold; beats 20-pass readouts</td></tr><tr><td>any line, any checkpoint,</td><td>EMERALD DreamerV3</td><td>Crafter maze</td><td>28/28 rules, 38/38 checkpoints</td></tr><tr><td>Fig. 17 six checkpoints, Fig. 18</td><td>DreamerV3</td><td>cheetah</td><td>doubt halved; error 24.6× lower</td></tr><tr><td>dream and reality, Fig. 19</td><td>DreamerV3</td><td>walker, Crafter</td><td>trust 0.16 against 0.42 when calm</td></tr><tr><td>trust over ten runs, Fig. 20(a)</td><td>DreamerV3</td><td>four DMC tasks</td><td>11% within depth, first of six</td></tr><tr><td>hyperparameters, Fig. 20(b, c)</td><td>DreamerV3</td><td>cheetah, walker</td><td>whole rollout best; return flat</td></tr><tr><td>choosing by trust, Fig. 21(a)</td><td>DreamerV3</td><td>Crafter</td><td>deaths/1k steps 3.80 against 4.08</td></tr><tr><td>learning Crafter, Fig. 21(b, c)</td><td>DreamerV3</td><td>Crafter</td><td>score 15.6% against 9.0%</td></tr><tr><td>where vetoes fire, Fig. 22</td><td>DreamerV3</td><td>maze</td><td>143/145 in a changed room</td></tr></table>

total: 210 compared cells in Table 1 · 4 bases · 3 environment families, 7 tasks · 17 readouts

## E.4 Seventeen readouts and their cost

Every readout is read on the same frames and imagined futures, and oriented so that higher means less familiar (Table 8). The fitted heads and multiple forwards are RND (Burda et al., 2019), an evidential head (Sensoy et al., 2018), latent disagreement (Sekar et al., 2020), the Mahalanobis distance (Lee et al., 2018), deep nearest neighbours (Sun et al., 2022), deep ensembles (Lakshminarayanan et al., 2017), MC dropout (Gal & Ghahramani, 2016), a Laplace posterior (Daxberger et al., 2021) and snapshot ensembles (Huang et al., 2017); self, three samples of one model, is our control for deep.

Table 8: The readouts compared in Table 1, as defined on DreamerV3; the other bases read them from their own features. Cost as in Table 1, for tests (A) / (B).
<table><tr><td>family</td><td>readout</td><td>what it reads</td><td>cost (A / B)</td></tr><tr><td>free</td><td>base</td><td>our doubt, read from the logits of the unmodified base</td><td>1×</td></tr><tr><td></td><td>entropy</td><td>entropy of the latent distribution</td><td>1×</td></tr><tr><td></td><td>max p.</td><td>one minus its largest probability</td><td>1×</td></tr><tr><td></td><td>KL</td><td>divergence of the posterior from the prediction</td><td>1×</td></tr><tr><td></td><td>recon.</td><td>reconstruction error of the arriving frame</td><td>1×</td></tr><tr><td></td><td>1-step</td><td>error of the arriving frame decoded from the prediction</td><td>1×</td></tr><tr><td>fitted heads</td><td>RND</td><td>error against a fixed random network, on the model state</td><td>+.18/+.13</td></tr><tr><td></td><td>RND-e</td><td>the same, on the encoder embedding</td><td>+.38</td></tr><tr><td></td><td>evid.</td><td>vacuity of an evidential head trained on the model state</td><td>+.12/+.10</td></tr><tr><td></td><td>latent</td><td>disagreement of five transition heads</td><td>+.62 / +.52</td></tr><tr><td></td><td>Mahal.</td><td>Mahalanobis distance to the training states</td><td>+.04</td></tr><tr><td></td><td>kNN</td><td>distance to the fiftieth nearest training state</td><td>+2.4</td></tr><tr><td>multiple forwards</td><td>self</td><td>variance of three sampled predictions of one model</td><td>3 fwd</td></tr><tr><td></td><td>deep</td><td>disagreement of three independently trained models</td><td>3×</td></tr><tr><td></td><td>MC drop</td><td>variance over twenty dropout masks on the transition head</td><td>20 fwd</td></tr><tr><td></td><td>Laplace</td><td>variance over twenty weight samples of a Laplace posterior</td><td>20 fwd</td></tr><tr><td></td><td>snap.</td><td>disagreement of four earlier snapshots of the model</td><td>4 fwd</td></tr></table>

## F Further results

## F.1 The alarm fires in the new room

![](images/32bffa7fef5ec80a6456bf9a48aa17cfa126812804e7f2b16971db72deace2c4.jpg)  
Figure 11: Into the new room, and into it again. (a) One walk: views outside, inside (gold), back and inside again, and the doubt of Ours and Base in alarm units; sand: in the room; red: alarms. (b) Walks on which each readout raises its alarm inside the room.

The room sealed in training shows familiar walls in a layout the agent has never experienced. The doubt raises its alarm inside it on 25 of 29 walks and the base’s readout on none (Fig. 11); along one walk, it peaks each time the agent enters.

![](images/7d1c7b9b93b5fb8943273116b39fd1f5e5da91ef896c15d43be9d353d5d108c5.jpg)

![](images/5e5574516c449f345cc50ac051d525701f5f90aeeb16bcc735d6bfe547264ac2.jpg)  
Figure 12: One pass against three models. (a) AUROC of the new room against relative parameters, as in Table 1(A); grey band: below chance, where a readout reads the new room as more familiar than the maze. (b) Walks through the opened door on which each readout raises its alarm, with its cost; top: walk 10 at −14, 0 and +25 steps from entry, gold where Ours raises its alarm, joined to its dot.

One pass does what three models cannot (Fig. 12): at the cost of one forward pass the doubt ranks the new room above the maze, while the base’s latent and deep ensembles, at up to three times the cost, read it as more familiar; on ten further walks through the door, the doubt raises its alarm on eight, the deep and self-ensembles and the entropy on none.

## F.2 The alarm fires before the fall

![](images/4b884d4047716c1d1fc660df58036d53c62584f46fc0282ba8b8241a7039785b.jpg)

(b) Alarmed  
![](images/bbe559ea3354be65c482f91815f8fb8adcf5387dc7453415b4743912ac1a5637.jpg)  
Figure 13: Twenty falls, twenty alarms before the fall. (a) Doubt (blue) and the entropy of the same prediction (grey) around the onset of each of twenty consecutive falls, in alarm units; dashed: alarm line; red: onset of the doubt’s alarm that holds into the fall; sand: the fall. (b) Falls whose alarm has fired by each step. Top: fall 5, the fall of Fig. 1, at −20, −6, 0 and +5 steps; gold: step −6, where its alarm holds; the doubt first crosses the line at −8, which Fig. 1 counts as its lead.

The doubt warns of each of twenty consecutive cheetah falls before it begins (Fig. 13): its alarm fires a median of four steps ahead, while the entropy of the same prediction raises none.

## F.3 The alarm fires at nightfall

![](images/4aafa47e182f1ccd055bb0c57998f9883f94cd99baabd2355a76bbc7a6a60d66.jpg)  
Figure 14: Nightfall. Views from fifteen steps before nightfall to thirty-five after (gold: at the alarm), above the doubt of Ours and Base in alarm units; sand: the night; strip: the fading light.

Nightfall darkens Crafter step by step. The doubt climbs over its line within twenty-five steps and peaks above twice it, while the base’s readout never reaches half of its own (Fig. 14).

## F.4 The alarm fires on every base

![](images/5531e3624c0dae98ed279ad87e3c878707356cd7b12ea617873399ef6a289fb7.jpg)  
(a) DreamerV3

![](images/248aaffad448957eb1db6d80f254061408cbe0b966f5a33184e96e2ed7d7b714.jpg)

![](images/e5bfb94677b60face3ff156dfa5c4d9783aae3dab4b19c1d450b971385d2c523.jpg)  
(b) R2-Dreamer

![](images/d503f1e1c52294fc3ee04b98b85dc06512943aa224f8603eff401c60562d187b.jpg)

![](images/b473448ebe24964a6fc07b667b8a245dc8672c56c6d479137710a78474553c6a.jpg)

(c) EMERALD  
![](images/8b8e0c05538612fe43e1502541f95c0bff3cf6933295d591ca0a485ecd92d361.jpg)  
steps from entry

(d) OC-STORM  
![](images/7c1e0c90ea166b036a7c15273ffa9c50df9bb2147fbe2a13b9f4087b5f287756.jpg)  
Figure 15: Four bases, four alarms. Doubt of Ours and Base on each base as the agent approaches the new room, in alarm units; sand: the room in view; red: the first alarm of Ours. Top: the route plotted in (b) and (d) at −14, 0 and +11 steps from entry, gold at the alarm of (b), joined to its dot; (a) and (c) pass the same door.

On all four bases the doubt crosses its line at the new room (Fig. 15).

## F.5 The worse the actions, the higher the doubt

The more actions are corrupted, the higher the doubt climbs (Fig. 16). On OC-STORM its median over starts rises over fivefold on walker and over twofold on Crafter, while the base’s readout barely moves. On EMERALD walker and cheetah the doubt rises over threefold from one pass, where MC dropout, Laplace and a snapshot ensemble, at up to twenty passes, fall instead.

![](images/7880747e13f9be1f52cf2e13ad20741f79b51f8f12d17274e5da9205e64f0687.jpg)

![](images/9ec17eefe27b98a4a909a7e6693ffb5a630d71bba1de5c9a5e67a6a0070e92cb.jpg)  
Figure 16: The worse the actions, the higher the doubt. Lift against the share of actions corrupted; thin line at one: no response. (a, b) OC-STORM, median over 300 starts (band: interquartile range of Ours); insets: imagination under true and under corrupted actions. (c, d) EMERALD against MC dropout, Laplace and a snapshot ensemble, on a log scale; insets: a real frame of each task.

## F.6 The alarm fires at any line, any checkpoint

![](images/969643e375cdfbd022606946dc53c6b281c83aa083f46a4b687eb3420ea19330.jpg)  
Figure 17: Any alarm line, any checkpoint. (a) Rules, placement of the line (rows) by frames required above it (columns), under which each readout raises its alarm in the repainted room. (b) AUROC of the opened room at each of 38 checkpoints of training; dashed: chance. Top: each change seen from one viewpoint, as trained and as changed.

The alarm does not hinge on where its line is drawn or on when the model is read (Fig. 17). In the repainted room the doubt raises its alarm under all 28 rules, four placements of the line by seven required durations, and the base’s readout under only 4. Across 38 checkpoints of training, from 0.1M to 3.8M frames, the doubt ranks the frames of the opened room above the familiar ones at every checkpoint, and the base’s readout ranks them below chance at every one.

## F.7 The doubt drains as the model learns

At one step, and averaged over all 24 imagined steps, the doubt falls at every checkpoint of the run of Sec. 4.4 (Fig. 18): from 0.194 at 48k environment steps to 0.093 at 180k, and from 0.181 to 0.096. Over the same span the error falls 24.6-fold at one step and 3.9-fold on average, while the entropy of the same prediction, averaged likewise, rises from 0.79 to 1.01.

![](images/1b7b5a42d514162c8fc4010d16fd9ee097d2961c2bba375ede2b773de26da26b.jpg)  
Figure 18: Six checkpoints, one future. As Fig. 6, at all six checkpoints of the run. Right: error, doubt and the entropy of the same prediction, each averaged over the 24 steps, relative to the first checkpoint.

## F.8 Trust fades faster where the dream drifts

Fed one real action sequence, the world and the imagination part ways (Fig. 19). On walker the imagined walker falls while the real one walks on, and the pixel error between the two futures grows almost fivefold. On Crafter lava appears in the real world from the second step and never in the dream, and the trust of this rollout drains far faster than that of rollouts started in calm scenes: 0.16 against 0.42 at step 6.

(a) Walker  
![](images/5125a0eec77d448107a7eb256ed27c9bbca33ab2a49db354d50596c9c670ab73.jpg)

(b) Crafter  
![](images/131b91e797f2cb0f20de2c0ce6bf65e11bf8944be28260b8c3080e1f4e54b60b.jpg)

![](images/7e0deb3b2968e204ff1066346f6d8a97ea4c1be92a4604703568f73410e5b29a.jpg)

![](images/c14c8d27968c18ae6ca146c360761529f6c9002edbbf8f8271c178974da7512d.jpg)  
Figure 19: Trust fades faster where the dream drifts. Real (top) and imagined (bottom) futures under one action sequence. (a) Walker; gold: the imagined walker has fallen; the pixel error between the two futures. (b) Crafter; gold: lava in the real world; trust of this rollout against the median over thirty calm starts.

## F.9 Trust keeps its edge across ten runs

Within each depth, averaged over ten runs across walker, cheetah, finger and cartpole, trust removes 11.0% of the error, more than any other readout of Sec. 4.5 (Fig. 20a): trust compounded over two steps removes

9.3%, the doubt of one step 6.8%, the doubt of the posterior 4.2%, entropy 3.6% and depth alone 0.0%. Trust removes error within depth on each of the ten runs.

(a) Ten runs, within depth

![](images/e7fd59b95de2c0bc0a7c7f5731f96f13a6f1f25511bee279ad24772b4ae4466b.jpg)

![](images/43ef850c53f238b951be3954a0c49be564f16ef2154847d259cf9cfd9ffdf24a.jpg)

![](images/6619efef8f3b08f28ac99aba2eb652997a774b94405ed1fdd6b0fa733f49a837.jpg)

![](images/7268b9c67def6d41b40fcc1570ab11cce1960f770143defa077e3716867f7e72.jpg)

(b) Steps carried  
![](images/763770bd30d1685bf9d5a634cd1a821f8babe2436fa909767c2e33c5a21ecf03.jpg)

![](images/925e86de7cf900dd02bc5ac78c7ebbf19a2c6a5b679e0ae75d6de82753118f34.jpg)

![](images/384c91943daa7d9aff81984836d4098a8634f326ef9c82ae5981e255f0c7b7e7.jpg)  
Figure 20: Trust across ten runs, and two hyperparameters. (a) Error removed by dropping the worst fifth of imagined steps, ranked within each depth as in the solid bars of Fig. 7, averaged over ten runs across walker, cheetah, finger and cartpole; top: the four tasks. (b) The same, ranked over all depths on cheetah from pixels as in Fig. 7b, with trust compounded over the last w steps; w = 15: the whole rollout, as in LucidWM. (c) Return on walker against the discipline weight $\beta _ { u }$ . Whiskers: bootstrap 95% intervals.

## F.10 Hyperparameter analysis

LucidWM carries trust through the whole rollout, and no shorter window w does better (Fig. 20b): on cheetah from pixels, the task of Fig. 7b, the error removed grows with every longer window, from 18.7% at w = 1, the doubt of one step, to 31.5% at all 15 steps. The weight $\beta _ { u }$ of the evidence discipline leaves the return in place (Fig. 20c): on walker, from $3 \times 1 0 ^ { - 4 } \mathrm { t o 1 0 ^ { - 2 } }$ , it stays between 924 and 949.

## F.11 Choosing futures by trust cuts deaths

(a) Choosing by trust  
![](images/ee0de722b5eb7940f1b4f00cd7d4b8933ca4fb6c7f7ad55dc473580caa4a6aa2.jpg)  
(b) Learning Crafter

![](images/b12ed63d0232a85fc03c291ea1c15d5509bb182a7d4f84eb280ecd36e5c8987d.jpg)  
(c) Score

![](images/438a5b26d561d34c19c84986becef6932ef88eea8ad7576c595c1defd815ca14.jpg)

![](images/9b821fa0bc8c264a3c5d999deaa936a0884172506cc7f76eb0d4342934db7709.jpg)  
Figure 21: Fewer deaths, faster learning. (a) One agent choosing among imagined futures scored with and without trust; median life: per seed, averaged over 50 seeds. (b) Evaluation return over training; dots and lines: where each first reaches 8. (c) The official Crafter score of the 22 achievements over the last 100 evaluation episodes; whiskers: bootstrap 95% intervals over episodes; tiles: from the scored episodes, Ours placing a furnace and Base a table.

Used beyond the veto to choose among whole futures, trust keeps the agent alive longer (Fig. 21a). At every step in Crafter, LucidWM on DreamerV3 imagines eight futures of 15 steps with its own actor and takes the first action of the future that scores highest. With trust, a future is scored by a variant of J (Eq. (10)) that keeps λ, weighting step k by $\lambda ^ { k } C _ { k }$ , the share of the lucid return that reaches it; without trust, by $\lambda ^ { k }$ alone. Both scores see the same futures, and trust changes the choice at 43% of steps. Over 50 seeds of 1,000 steps, trust cuts deaths from 4.08 to 3.80 per 1,000 steps and lengthens the median life from 236 to 253 steps (paired Wilcoxon, $p = 0 . 0 4 1$ and 0.0016).

## F.12 LucidWM learns Crafter faster than its base

The agent of App. F.11 learns Crafter faster than its base and scores higher (Fig. 21b, c). After 0.35M steps it reaches the evaluation return of 8 that its base first reaches after 0.96M, and over the last 100 evaluation episodes of training its Crafter score is 15.6% against 9.0%, with 11.0 achievements unlocked per episode against 8.9.

## F.13 Vetoes fire where the world changed

![](images/330a7c5581bdfc50c943ba02e79d1ff7037afcd2ca40a760cd325949df4cc59a.jpg)

![](images/f2c62741fe643256fed4532e881d9d2dc582626fa666c3adf887be8737f698e7.jpg)  
Figure 22: Vetoes fire where the world changed. Where every veto fired on (a) the opened map and (b) the repainted dead end; sand: the changed rooms; rings: vetoes outside them. Insets: the agent’s view at a veto, gold and joined to its dot; in (b), also the repainted room at step 8. The veto at step 24 in (b) is that of Fig. 8.

The veto of Sec. 4.6 fires where the world has changed (Fig. 22). On the two altered maps, 143 of its 145 vetoes fall inside a changed room: all 127 in the opened room and 16 of 18 in the repainted dead end.