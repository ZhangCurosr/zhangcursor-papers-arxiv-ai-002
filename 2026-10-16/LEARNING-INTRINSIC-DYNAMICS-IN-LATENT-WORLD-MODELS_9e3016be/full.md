# LEARNING INTRINSIC DYNAMICS IN LATENT WORLD MODELS

Jiamin Shi<sup>1</sup> Yuanhong Zhang<sup>1</sup> Zhiyang Liu<sup>2</sup> Jiawei Sun<sup>2</sup> Jingwen Fu<sup>3</sup>

<sup>1</sup>Xi’an Jiaotong University <sup>2</sup>National University of Singapore

<sup>3</sup>Beijing Zhongguancun Academy

## ABSTRACT

Learning intrinsic dynamics—the relationship between actions and their consequences—is essential for world models. However, visual variations unrelated to the underlying control process can obscure this relationship in learned representations. We investigate the connection between action predictability, defined as the ability to reconstruct an action from consecutive latent states, and the preservation of intrinsic dynamics. Our central hypothesis is that encouraging action predictability discourages representationsfrom discarding the distinctions between different actions and their consequences. We provide two complementary forms of evidence for this connection. Theoretically, we derive a positive lower bound on squared inverse prediction error when the encoder merges distinguishable outcomes of different actions from the same state and appearance, and show that forward prediction can still be exact on those transitions. Empirically, controlled appearance-diversity comparisons, supervision interventions, and latent probes support the role of action prediction in preserving control-relevant transition information. We instantiate action predictability with Concat IDM, which predicts actions from concatenated latent states and updates the shared encoder using action labels already available in offline trajectories, without changing the test-time planner. Across three independent training runs on Push-T with 24 appearances, Concat IDM increases planning success from 9.1% to 77.2%; its paired improvement is 52.9 percentage points larger than that observed with a single appearance. Together, these findings support action predictability as an inductive bias toward intrinsic dynamics and demonstrate the practical benefits of Concat IDM for latent planning under visual diversity.

## 1 INTRODUCTION

![](images/55d137f894aa8ae674e575ed8f3490a3af41f935118711dfd52c07e8dc3ca25a.jpg)  
(a) Same dynamics, different appearances

![](images/1f1f76911ecee2fdb028f045ce9f02c3b4c8247b1e9621b5d6083ab09ac4c52c.jpg)  
(b) Larger gains under visual diversity  
Figure 1: Visual diversity amplifies the benefit of action prediction. (a) Schematic of appearance variation with physical trajectories held fixed. (b) Under this controlled variation on Push-T, the gain of Concat IDM over Plain JEPA grows from 15.3 pp at K=1 to 68.1 pp at K=24.

An action-conditioned latent world model should capture how the underlying system responds to actions. We refer to this action-conditioned transition structure as intrinsic dynamics. This relationship concerns the underlying control process rather than the visual appearance through which it is observed. However, images contain variations unrelated to that process: object color, background, lighting, and rendering style can change observations without changing the physical transition. We refer to these variations as external factors (Figure 1a). Learning from visually diverse observations therefore presents a challenge: the representation must retain action-dependent transition information even as the appearance of the same underlying process varies.

When the encoder and predictor are trained jointly, the encoder also defines the prediction targets. Accurate latent prediction does not, by itself, ensure that the representation preserves distinctions in how the underlying system responds to different actions. We therefore investigate whether action predictability can help preserve these distinctions. We define action predictability as the ability to reconstruct the executed action $a _ { t }$ from consecutive latent states $z _ { t }$ and $z _ { t + 1 }$ . Predicting actions from transitions is an established inverse-dynamics objective in representation learning and latent planning (Agrawal et al., 2016; Pathak et al., 2017; Paster et al., 2021; Sobal et al., 2025). Our focus is on understanding how this requirement can help preserve the relationship between actions and their consequences when external factors vary.

Our central hypothesis is that encouraging action predictability helps representations distinguish the consequences ofdifferent actions. Suppose two different actions taken from the same state lead to different outcomes. If the encoder maps both outcomes to the same latent state, the latent transition no longer tells us which action was taken. Training the model to predict the action therefore encourages the encoder to keep these outcomes distinct.

We provide two complementary forms of evidence connecting action predictability to learning intrinsic dynamics. Theoretically, if the encoder maps distinguishable outcomes of different actions from the same state and appearance to the same latent state, squared inverse prediction error has a positive lower bound, while forward prediction can remain exact (Proposition 3.1). Action prediction thus penalizes the loss of action–consequence distinctions that forward prediction may overlook. Empirically, controlled comparisons show larger planning gains from action prediction under greater appearance diversity, with physical trajectories held fixed. Supervision interventions link these gains to encoder updates from correctly paired action labels, while frozen-encoder probes show that actions are decoded more accurately from transitions than from either endpoint alone. Together, these findings support action predictability as an inductive bias toward intrinsic dynamics, rather than a guarantee of complete dynamics recovery.

We instantiate action predictability with Concat IDM, which predicts $a _ { t }$ from concatenated latent states $[ z _ { t } ; z _ { t + 1 } ]$ . Its inverse-dynamics loss updates the shared encoder, so that action supervision directly constrains the representation used by the world model. The method uses action labels already available in offline trajectories and leaves the test-time planner unchanged. We implement Concat IDM on the LeWorldModel backbone (Maes et al., 2026) and compare it against Plain JEPA, which uses the same backbone without the inverse-dynamics objective. Across three independent training runs on Push-T with 24 appearances, Concat IDM increases planning success from 9.1% to 77.2%. Its paired improvement over Plain JEPA is 52.9 percentage points larger than that observed with a single appearance (Figure 1b). Because the physical trajectory pool is held fixed, this comparison shows that the benefit grows sharply with visual diversity in this controlled setting. These results demonstrate the practical value of Concat IDM while complementing the theoretical and empirical evidence for action predictability as an inductive bias toward intrinsic dynamics.

The central contribution of this paper is to clarify the connection between action predictability and intrinsic dynamics in latent world models. Specifically, we (1) hypothesize that encouraging action predictability helps preserve distinctions between the consequences of different actions that action-conditioned forward prediction alone may overlook; (2) provide theoretical and empirical evidence supporting this hypothesis; and (3) instantiate this principle with Concat IDM and evaluate its effects on learned representations and downstream planning.

## 2 RELATED WORK

Learning latent dynamics. Latent world models support online planning in PlaNet, behavior learning through imagination in Dreamer, and task-oriented predictive control in TD-MPC (Hafner et al., 2019; 2020; Hansen et al., 2022). Self-predictive representations also use multi-step latent prediction as an auxiliary objective for visual RL (Schwarzer et al., 2021). Joint-embedding predictive architectures learn prediction targets in representation space without pixel reconstruction (Assran et al., 2023; Bardes et al., 2024); V-JEPA 2 extends video pretraining to action-conditioned robotic planning (Assran et al., 2025). DINO-WM predicts frozen DINOv2 features (Zhou et al., 2025), whereas PLDM and LeWorldModel jointly learn the encoder and latent dynamics (Sobal et al., 2025; Maes et al., 2026). PLDM already combines inverse dynamics with VICReg (Bardes et al., 2022); LeWorldModel combines next-embedding prediction with SIGReg (Balestriero & LeCun, 2025). Concurrent Temporal-Distance JEPA addresses the mismatch between short-horizon representation learning and multi-step planning costs (Bai & Xiong, 2026). We study a different axis: how visual diversity over fixed physical trajectories affects jointly learned representations and the benefit of inverse supervision.

Inverse dynamics and action predictability. Inverse dynamics provides supervision for representations of physical interaction and control (Agrawal et al., 2016; Pathak et al., 2017; Paster et al., 2021). Multistep inverse models underlie provable RL with exogenous distractors (Efroni et al., 2022), control-endogenous state discovery in AC-State (Lamb et al., 2023), and offline agent-centric representation learning in ACRO (Islam et al., 2023). ACDF identifies failures of multistep inverse prediction alone and combines it with forward modeling (Levine et al., 2024). Our analysis isolates how inverse prediction penalizes the merging of distinguishable action consequences. Concurrent Delta-JEPA predicts actions from latent displacements (Zhang et al., 2026); our matched ∆z-IDM comparison changes only the inverse input, holding the backbone, regularizer, data, and planner fixed. Concurrent ACID uses inverse dynamics at decision time to score imagined transitions (Seo et al., 2026). In contrast, our inverse loss constrains the shared encoder during training; the inverse head is discarded for planning.

External visual factors. Domain randomization and adversarial domain learning address visual variation (Tobin et al., 2017; Ganin et al., 2016). CURL learns contrastive representations from augmented views (Laskin et al., 2020a), while RAD and DrQ improve pixel-based control through data augmentation (Laskin et al., 2020b; Kostrikov et al., 2020). Bisimulation-based approaches preserve control-relevant state equivalence (Zhang et al., 2021); Toso et al. (2026) apply this principle over pretrained visual features for latent world-model planning under background changes and distractors. Concurrent ACPC diagnoses predictive consistency between matched clean and perturbed histories under the same actions (An et al., 2026). We study action–consequence distinctions during joint encoder–predictor training by varying visual appearances over fixed offline trajectories, without an explicit appearance-invariance objective or matched-view supervision.

## 3 THEORY AND METHOD

## 3.1 INTRINSIC DYNAMICS AND EXTERNAL FACTORS

Let $s _ { t }$ denote the physical state, a<sub>t</sub> the executed action, and c the factors that affect only the visual observation. For the controlled environments considered here,

$$
o _ { t } = R ( s _ { t } , c ) , \qquad s _ { t + 1 } = F ( s _ { t } , a _ { t } ) .\tag{1}
$$

The action-conditioned transition structure induced by F is the intrinsic dynamics; the variables in c are external factors because they alter the rendering without altering the physical transition.

We construct appearance-diverse datasets by rendering a fixed pool of control trajectories with a finite palette of size K. Each trajectory receives one rendering, and matched renderings of the same trajectory are not provided during training. The physical state and action sequences are unchanged as K varies; increasing K redistributes the same control data across more visual realizations. We also vary the episode-level assignment while holding the physical trajectory pool and palette fixed (Section 4.3).

![](images/0b211c70b647e4fb417b66848858c1f59da7e605aa5ba0ca6dce06621fba5f6b.jpg)  
Figure 2: Action-supervised latent world modeling and planning. $E _ { \theta } \colon$ visual encoder; $P _ { \phi } \colon$ latent predictor; $I _ { \psi } \colon$ inverse dynamics head. (a) CEM ranks predicted rollouts by terminal latent goal cost. (b) Concat IDM predicts actions from concatenated observed latents, supervising the shared encoder during training.

## 3.2 ACTION PREDICTABILITY AND TRANSITION DISTINCTIONS

Action predictability constrains the information retained between consecutive states: their encodings should remain predictive of the executed action. To characterize this constraint, consider a fixed encoder $E _ { \theta }$ and an inverse decoder g that predicts actions in $\mathbb { R } ^ { d _ { a } }$ . Fix a physical state s and appearance c in Eq. (1). Let $o = R ( s , c )$ and $o _ { i } ^ { \mp } = R ( F ( s , a _ { i } ) , c )$ for distinct actions $a _ { 1 } , a _ { 2 }$ . Assume $o _ { 1 } ^ { + } \neq o _ { 2 } ^ { + }$ , so the two outcomes are distinguishable before encoding. Consider a distribution assigning probabilities $p \in ( 0 , 1 )$ and $1 - p$ to these two transitions, and define

$$
u _ { i } = [ E _ { \theta } ( o ) ; E _ { \theta } ( o _ { i } ^ { + } ) ] , \qquad \mathcal { R } _ { \mathrm { p a i r } } = \frac { p \| g ( u _ { 1 } ) - a _ { 1 } \| _ { 2 } ^ { 2 } + ( 1 - p ) \| g ( u _ { 2 } ) - a _ { 2 } \| _ { 2 } ^ { 2 } } { d _ { a } } .\tag{2}
$$

Proposition 3.1 (Penalty for merging distinguishable outcomes). Under the conditions above, if $E _ { \theta } ( \mathbf { \bar { \omega } } _ { 1 } ^ { + } ) = E _ { \theta } ( o _ { 2 } ^ { + } )$ , then every inverse decoder satisfies

$$
\mathcal { R } _ { \mathrm { p a i r } } \geq \frac { p ( 1 - p ) } { d _ { a } } \| a _ { 1 } - a _ { 2 } \| _ { 2 } ^ { 2 } > 0 .\tag{3}
$$

The bound is attained by predicting $\bar { a } = p a _ { 1 } + ( 1 - p ) a _ { 2 }$ at the shared latent input. For this pair alone, a forward predictor can nevertheless attain zero latent prediction error by outputting the common successor encodingfor both actions.

Proposition 3.1 isolates an information-loss mode of latent prediction: distinct physical outcomes can share an exactly predictable latent target, whereas inverse supervision penalizes this merger. The result holds at a fixed appearance; Section 4.3 tests empirically how the benefit of action supervision changes as appearance diversity varies. To see the bound, let b be the decoder output at the shared latent input and write $\bar { a } = p a _ { 1 } + ( 1 - p ) a _ { 2 }$ . Completing the square gives $p \| b - \dot { a _ { 1 } } \| _ { 2 } ^ { 2 } + ( 1 - p ) \| b - a _ { 2 } \| _ { 2 } ^ { 2 } =$ $\| b - \bar { a } \| _ { 2 } ^ { 2 } + p ( \bar { 1 } - p ) \| \dot { a } _ { 1 } - \bar { a _ { 2 } } \| _ { 2 } ^ { 2 }$

What the inverse objective preserves. The pairwise result extends to an arbitrary transition distribution with finite second action moments. Write $X = ( O _ { t } , O _ { t + 1 } ) , U = [ E _ { \theta } ( O _ { t } ) ; E _ { \theta } ( O _ { t + 1 } ) ]$ and let $\mu _ { X } = \operatorname { \mathbb { E } } [ A \mid X ]$ and $\mu _ { U } = \mathbb { E } [ A \mid U ]$ . If $\mathcal { R } _ { X } ^ { * }$ and $\mathcal { R } _ { { I J } } ^ { * }$ are the minimum coordinate-averaged squared action errors over unrestricted decoders of X and $\breve { U } .$ , respectively, then

$$
\mathcal { R } _ { U } ^ { * } - \mathcal { R } _ { X } ^ { * } = \frac { 1 } { { { d _ { a } } } } \mathbb { E } \big [ \| \mu _ { X } - \mu _ { U } \| _ { 2 } ^ { 2 } \big ] .\tag{4}
$$

Thus, encoding incurs an additional inverse-prediction penalty whenever it removes distinctions needed to recover the conditional action mean. The decomposition separates ambiguity already present in the observations from information lost through the encoder. Forward prediction rewards predictable latent targets, whereas inverse prediction makes those targets retain information about the actions that produced them. This characterization motivates examining both action decoding and forward rollout consistency: the two diagnostics test different properties of the representation used by the planner.

## 3.3 CONCAT IDM AND TRAINING OBJECTIVE

We use an action-conditioned JEPA with a shared visual encoder and a latent predictor. Figure 2 summarizes the planning pipeline and action-supervised training. Our implementation follows LeWorldModel (Maes et al., 2026). The encoder produces $z _ { t } = E _ { \theta } ( o _ { t } )$ ), and the predictor forecasts

$$
\hat { z } _ { t + 1 } = P _ { \phi } ( z _ { \leq t } , a _ { \leq t } ) ,\tag{5}
$$

where the observation history is limited to three frames. Here and in the training losses below, $a _ { t }$ concatenates the primitive actions between consecutive sampled observations, standardized coordinatewise using the offline dataset mean and standard deviation. The next-embedding prediction loss is

$$
\mathcal { L } _ { \mathrm { p r e d } } = \frac { 1 } { N _ { \mathrm { p r e d } } d _ { z } } \sum _ { t } \left. \hat { z } _ { t + 1 } - z _ { t + 1 } \right. _ { 2 } ^ { 2 } ,\tag{6}
$$

where $N _ { \mathrm { p r e d } }$ counts the predicted transitions and $d _ { z }$ is the embedding dimension. As in LeWorld-Model, the same encoder produces context and target embeddings, and gradients pass through both branches; there is no target network or stop-gradient in this loss. We use SIGReg (Balestriero & LeCun, 2025) to discourage representational collapse by regularizing random one-dimensional projections of the encoded latents toward a standard Gaussian distribution. We denote this regularizer by $\mathcal { L } _ { \mathrm { S I G } }$ and use 17 knots and 1024 projections.

We instantiate action predictability with an inverse dynamics head $I _ { \psi } .$ . Concatenation makes both endpoint representations available to the decoder without restricting it to their displacement. Our main variant, Concat IDM, predicts

$$
\hat { a } _ { t } = I _ { \psi } \big ( [ z _ { t } ; z _ { t + 1 } ] \big ) , \qquad \mathcal { L } _ { \mathrm { i d m } } = \frac { 1 } { N _ { \mathrm { i d m } } d _ { a } } \sum _ { t } \| \hat { a } _ { t } - a _ { t } \| _ { 2 } ^ { 2 } ,\tag{7}
$$

where $N _ { \mathrm { i d m } }$ counts adjacent latent pairs and $d _ { a }$ is the action-target dimension. Both squared losses are averaged over latent pairs and target coordinates. The complete objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { p r e d } } + \beta \mathcal { L } _ { \mathrm { S I G } } + \lambda \mathcal { L } _ { \mathrm { i d m } } , } \end{array}\tag{8}
$$

with SIGReg weight $\beta { = } 0 . 0 9$ and IDM weight $\lambda { = } 1$ in the main experiments. SIGReg regularizes the marginal latent distribution, while the inverse loss supervises action-dependent relations between endpoint encodings. The inverse loss updates the IDM head and propagates through both endpoint encodings to the shared encoder; it does not update the forward predictor directly.

$\Delta z { \mathbf { - } } { \mathbf { I D M } } .$ To compare input parameterizations, we replace the concatenated endpoints by their displacement and train $I _ { \psi } ( z _ { t + 1 } - z _ { t } )$ while keeping the remaining model and objective fixed.

Intervention ablations. (i) the stop-gradient intervention replaces both IDM inputs with detached latents, allowing the inverse head to train while preventing its objective from updating the encoder. (ii) the shuffled-target intervention permutes action labels within each batch, preserving the architecture while breaking their correspondence with the transitions. Together, these interventions test whether the gain depends on representation updates induced by correctly paired action supervision.

## 3.4 LATENT PLANNING

At test time, the encoder and predictor are frozen. Given a goal observation $o _ { g } ,$ we compute $z _ { g } = E _ { \theta } ( o _ { g } )$ . Starting from the encoded observation history, the predictor rolls each candidate action sequence $A = \left( a _ { t } , \ldots , a _ { t + H - 1 } \right)$ forward autoregressively to obtain $\hat { z } _ { t + H }$ . The main planner evaluates the terminal prediction in the full latent space:

$$
J ( A ) = \left\| \hat { z } _ { t + H } - z _ { g } \right\| _ { 2 } ^ { 2 } .\tag{9}
$$

CEM iteratively updates a distribution over action sequences to minimize J. The agent executes the first action block from the selected sequence, receives a new observation, and replans in closed loop. CEM uses 300 samples, 30 optimization steps, 30 elites, planning and receding horizons of 5, and an action block of 5 primitive steps.

Table 1: Planning gains across appearance conditions. Gain is Concat IDM minus Plain JEPA, computed from the displayed success means in percentage points (pp). Bold marks the higher success mean, including ties, and positive gains.
<table><tr><td rowspan="2">Task</td><td colspan="3">Single appearance  $( K { = } 1 )$ </td><td colspan="4">Multiple appearances</td></tr><tr><td>Plain JEPA</td><td>Concat IDM</td><td>Gain</td><td> $K$ </td><td>Plain JEPA</td><td>Concat IDM</td><td>Gain</td></tr><tr><td>Push-T</td><td> $7 2 . 2 { \pm } 7 . 9 $ </td><td> ${ \bf 8 7 . 5 \pm 4 . 4 }$ </td><td>+15.3</td><td>24</td><td> $9 . 1 { \pm } 2 . 7 $ </td><td> ${ \bf 7 7 . 2 \pm 0 . 2 }$ </td><td>+68.1</td></tr><tr><td>TwoRoom</td><td> $8 8 . 5 { \pm } 1 . 3 $ </td><td> ${ \bf 9 1 . 1 \pm 1 . 0 }$ </td><td>+2.6</td><td>36</td><td>53.2±1.4</td><td> ${ \bf 9 7 . 3 \pm 3 . 2 }$ </td><td>+44.1</td></tr><tr><td>Cube</td><td> $6 0 . 9 { \pm } 1 . 2 $ </td><td> ${ \bf 7 1 . 1 \pm 2 . 2 }$ </td><td> ${ \bf + 1 0 . 2 }$ </td><td>24</td><td>60.7±1.2</td><td> ${ \bf 7 5 . 4 2 1 . 3 }$ </td><td> $\mathbf { + 1 4 . 7 }$ </td></tr><tr><td>Reacher</td><td> $\mathbf { 1 0 0 . 0 { \pm } 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 { \pm } 0 . 0 }$ </td><td>0.0</td><td>24</td><td> $\mathbf { 1 0 0 . 0 { \dot { \pm } } 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 { \dot { \pm } } 0 . 0 }$ </td><td>0.0</td></tr></table>

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP AND METRICS

Our experiments ask whether action supervision improves latent planning, how its benefit depends on visual diversity, and what action information the learned representations retain.

Protocol. Primary planning comparisons use three independently trained models per condition, each evaluated on 500 episodes in 10 runs of 50 episodes. Paired methods share the physical trajectory pool, evaluation tasks, and planner. The Push-T diversity comparison pairs training runs across $K { = } 1$ and $K { = } 2 4 ;$ the other tasks compare three-model means at K=1 and their task-specific high-diversity settings. Evaluation starts and goals are sampled from the offline trajectory pool used for world-model learning, with current and goal observations sharing an appearance from the training palette. Separate evaluations change the training appearance assignment or use a disjoint test palette. The backbone uses a ViT-Tiny encoder with 192-dimensional embeddings and a six-layer predictor; the IDM head has hidden width 512. Unless noted, training uses AdamW for 10 epochs, learning rate $2 \times 1 0 ^ { - 5 }$ weight decay $1 0 ^ { - 3 }$ , and batch size 128.

Metrics. Our primary metric is closed-loop success. Unless specified in the table, aggregate results are mean ± sample SD across three independently trained models per condition or appearance assignment. Action-decoding $R ^ { 2 }$ measures accessible action information; predicted-versus-realized latent-cost agreement measures rollout-ranking consistency. Normalized final error is $\mathrm { N F E = }$ $d ( s _ { T } , g ) / ( d ( \bar { s } _ { 0 } , g ) + 1 0 ^ { - 8 } )$ , where $s _ { T }$ is the recorded state at first success or the end of the evaluation budget. Best progress is $\begin{array} { r } { 1 - \operatorname* { m i n } _ { 0 \leq t \leq T } d ( s _ { t } , g ) / ( d ( s _ { 0 } , g ) + 1 0 ^ { - 8 } ) } \end{array}$ . Push-T uses the Euclidean distance of the raw state vector (agent and block positions, block orientation, and agent velocity); TwoRoom uses agent-to-target Euclidean distance. These auxiliary metrics are compared within each task.

## 4.2 PERFORMANCE ACROSS CONTROL TASKS

Planning gains across tasks. Concat IDM improves planning on Push-T, TwoRoom, and Cube without changing the test-time planner (Table 1). Its advantage is larger under high visual diversity on Push-T and TwoRoom: gains rise from 15.3 to 68.1 percentage points and from 2.6 to 44.1 points, respectively. On Cube, Plain JEPA remains near 61% while Concat IDM gains 10.2 points at $K { = } 1$ and 14.7 at $K { = } 2 4$ . Reacher is saturated at 100% for both methods.

The continuous-error results complement these success gains. On Push-T at $K { = } 2 4$ , NFE falls from 1.353±0.108 to $0 . 6 5 2 { \pm } 0 . 0 3 \bar { 2 }$ , while best progress rises from $0 . 4 8 1 { \pm } 0 . 0 4 6$ to $0 . 6 7 4 { \scriptstyle \pm 0 . 0 0 7 }$ TwoRoom shows the same direction: NFE falls from 0.798±0.015 to $0 . 4 5 7 { \pm } 0 . 0 1 9$ and best progress rises from 0.387±0.008 to 0.568±0.010. These within-task comparisons indicate improvement in both the closest approach to the goal and the final recorded state, rather than only a change in how often a success threshold is crossed.

Inverse-dynamics designs. Compared with $\Delta z { \mathrm { - I D M } }$ , Concat IDM has higher mean success on Push-T (77.2±0.2% versus 65.8±14.7%) and the same mean on TwoRoom (97.3±3.2% versus 97.3±2.4%). Against PLDM, Concat IDM gains 24.0 percentage points and wins all three matched

Table 2: Planning success (%) and normalized final error (NFE) for inverse-dynamics designs and regularizers on Push-T (K=24).  
(a) IDM-based alternatives
<table><tr><td>Method</td><td>Success (%) ↑</td><td>NFE↓</td></tr><tr><td>Plain JEPA</td><td> $9 . 1 { \pm } 2 . 7 $ </td><td> $1 . 3 5 3 { \pm } 0 . 1 0 8$ </td></tr><tr><td>PLDM</td><td> $5 3 . 2 { \pm } 1 1 . 8 $ </td><td> $0 . 7 8 3 { \pm } 0 . 1 0 3$ </td></tr><tr><td> $\Delta z { \mathrm { - I D M } }$ </td><td> $6 5 . 8 { \pm } 1 4 . 7 \ $ </td><td> $0 . 7 7 7 { \scriptstyle \pm 0 . 1 7 7 }$ </td></tr><tr><td>Concat IDM</td><td> ${ \bf 7 7 . 2 \pm 0 . 2 }$ </td><td> $0 . 6 5 2 { \pm } 0 . 0 3 2$ </td></tr></table>

![](images/471f1509c4a71e5decd65b490b11bf817fe324ef1562197e4b0f6f1e5be31631.jpg)

<table><tr><td colspan="5">(b) Regularizer control</td></tr><tr><td rowspan="2">Reg.</td><td colspan="2">No IDM</td><td colspan="2">Concat IDM Gain</td></tr><tr><td>Success ↑</td><td>NFE ↓ Success ↑</td><td>NFE↓</td><td>(pp)</td></tr><tr><td>SIGReg</td><td> $9 . 1 { \pm } 2 . 7 $ </td><td>1.353±0.108  $7 7 . 2 { \pm } 0 . 2 $ </td><td>0.652±0.032</td><td> $+ 6 8 . 1$ </td></tr><tr><td> $\mathrm { V I C R e g }$ </td><td> $5 . 2 { \pm } 3 . 0 $ </td><td>1.039±0.059  $2 6 . 0 { \pm } 7 . 1 $ </td><td>0.890±0.050+20.8</td><td></td></tr></table>

Separate intervention cohort<sub>Figure 3: Action supervision</sub> through the encoder. Blocking encoder gradients or shuffling action labels reduces Push-T success relative to a matched Concat IDM control (K=24).

Push-T training pairs (Table 2a). This cross-design comparison retains PLDM’s own optimization configuration.

Dependence on regularization. The benefit of inverse supervision also appears with VI-CReg (Bardes et al., 2022). In a control that replaces SIGReg with VICReg and keeps the backbone, data, training schedule, and planner fixed, adding Concat IDM increases success from 5.2% to 26.0% and reduces NFE from 1.039 to 0.890 (Table 2b).

Action supervision through the encoder. We isolate the contribution of the inverse loss by blocking its encoder gradients or disrupting action–transition correspondence (Figure 3). Detaching the IDM inputs reduces success from the matched Concat IDM control of 75.5±2.8% to 21.1±17.2%; shuffling action targets reduces it to 9.4±3.7%. Both interventions also worsen NFE. These reductions support the role of correctly paired action supervision in shaping representations useful for planning.

## 4.3 ROLE OF APPEARANCE DIVERSITY

A larger gain at higher visual diversity. On Push-T, we render the same physical trajectory pool under $K { = } 1$ and $K { = } 2 4$ , matching training blocks across conditions. The high-minus-low difference in the Concat-minus-Plain gain is +52.9±7.1 percentage points, positive in all three blocks (Figure 1b).

Robustness across appearance assignments. Concat IDM maintains high success when the mapping from physical trajectories to appearances changes. We train three Plain/Concat pairs under each of three independent episode-to-appearance assignments, keeping the trajectory pool, $K { = } 2 4$ palette, and canonical evaluation tasks and renderings fixed. All nine Concat IDM models achieve 74.2–78.4% success, close to the canonical 77.2±0.2%, while Plain JEPA spans 5.6–61.2%. Concat IDM improves success and reduces NFE in all nine matched pairs (Figure 4a,b).

Changing the assignment tests a different source of variation from changing K. The palette size and physical transitions remain fixed, but the grouping of trajectories that share an appearance changes. Keeping the evaluation rendering fixed then tests whether the learned planner depends on a particular training association between appearance and behavior. The narrow Concat IDM success range across these assignments contrasts with the broad Plain JEPA range, suggesting that action supervision reduces sensitivity to this association within the observed palette.

## 4.4 ACTION INFORMATION AND PLANNING COSTS

Action information in latent transitions. We train fresh MLP probes on frozen encoders to predict actions from $z _ { t } , z _ { t + 1 } , \Delta z , \mathrm { o r } \left[ z _ { t } ; z _ { t + 1 } \right]$ , using a common trajectory-level training, validation, and test split within the world-model data pool. For concatenated endpoints, test $R ^ { 2 }$ is 0.799±0.001 with Concat IDM versus 0.251±0.209 with Plain JEPA (Figure 5a). Concat IDM supports displacement decoding at 0.846±0.002, compared with 0.254±0.003 from its better individual endpoint. The $\Delta z { \bf - I D M }$ encoder attains a similar displacement score (0.850±0.002), despite lower mean Push-T planning success (Table 2a). The probes thus establish improved action accessibility in both representations, motivating a separate assessment of candidate-cost ordering.

Ordering candidate plans. We compare each predicted terminal cost with the realized latent cost obtained by executing the same candidate and encoding its outcome with the same model. This diagnostic measures whether imagined and realized outcomes induce consistent candidate rankings in the learned latent space. At horizon H=5, Concat IDM achieves Spearman correlations of 0.395– 0.411, compared with 0.153–0.170 for Plain JEPA (Figure 5b). Together, the probes and ranking comparisons show more accessible action information and more consistent latent rollouts.

Rank correlation summarizes the entire candidate set, whereas CEM acts on its low-cost subset. We therefore also examine selection regret and overlap between the predicted and realized top-10% candidates. At H=5, Concat IDM has normalized selection regret of 0.961–1.067, compared with 1.664–3.107 for Plain JEPA; top-10% overlap increases from 0.147–0.170 to 0.263–0.277. Both measures support the ranking improvement at the part of the candidate set used for action selection. Because realized costs use each model’s own encoder, these comparisons assess rollout consistency within a representation; closed-loop success separately evaluates whether optimizing that representation achieves the physical task. The similar displacement-probe scores of the two IDM variants further show why action decodability alone cannot determine their planning performance.

## 4.5 SENSITIVITY AND SCOPE

IDM weight and training progress. At the default λ=1, Concat IDM improves success from 9.1% to 77.2% and reduces NFE from 1.353 to 0.652 (Table 3a). The additional weight settings yield success rates of 71.0% at λ=0.1 and 85.8% at λ=10. Across the three training runs, the advantage emerges early: at epoch 3, Concat IDM reaches 63.1% success versus 10.6% for Plain JEPA, with lower NFE at every reported epoch (Table 3b).

The training curves distinguish early gains from sustained improvement. Plain JEPA’s mean success peaks at epoch 5 among the reported checkpoints and then falls, while Concat IDM continues to improve through epoch 10. Over the same interval, their NFE values move in opposite directions.

![](images/b0e38e57e3a9b34ef642c8ab6fe015d4be4fc9099baf96743a5c0aa9963a5278.jpg)

![](images/233365c877e787a948f2169ebb2c2069c70eb065a1835653ccfbcc0d4b9c119c.jpg)  
(b) Final-state error  
Figure 4: Robustness to training appearance assignments on Push-T (K=24). (a) Planning success. (b) Normalized final error. Bars show means with sample SDs; dots denote trained models and lines connect matched runs.

![](images/3ef970e8043f2da99903c30bd3629a0b1ffb9665e6d68649abce0eead26dbfbb.jpg)  
t agreement(a) Frozen-encoder action probes

![](images/76007ee6ab4e80921473ea3a76a57d6a3aace2da9483eab8acf78e94c55ea180.jpg)  
(b) Latent-cost agreement  
1.0Figure 5: Action information and latent rollout consistency on Push-T. (a) Frozen-encoder action-<sup>ρ</sup>probe test $R ^ { 2 }$ ; Pair denotes $[ z _ { t } ; z _ { t + 1 } ] .$ . (b) Spearman correlation between predicted and realized latent <sup>a</sup>costs. Thin lines show individual models; thick lines show means.

0.0Table 3: IDM weight and training progress on Push-T $( K { = } 2 4 )$ . In (a), $n { = } 3$ for $\lambda \in \{ 0 , 1 \}$ and $n { = } 1$ 1 3 5<sub>otherwise.</sub> <sub>In</sub> <sub>(b),</sub> <sub>each</sub> <sub>epoch</sub> <sub>uses</sub> <sub>the</sub> <sub>same</sub> <sub>three</sub> <sub>training</sub> <sub>runs</sub> <sub>per</sub> <sub>method.</sub> <sub>Success</sub> <sub>is</sub> <sub>in</sub> <sub>percent;</sub> R<sub>NFE is lower-is-better.</sub>  
(a) IDM loss weight  
(b) Training progress
<table><tr><td>λ</td><td>Success ↑ NFE↓</td></tr><tr><td>0</td><td>9.1±2.7 1.353±0.108</td></tr><tr><td>0.1</td><td>71.0 0.713</td></tr><tr><td>1 77.2±0.2 10</td><td>0.652±0.032 85.8 0.610</td></tr></table>

<table><tr><td></td><td colspan="2">Plain JEPA</td><td colspan="2">Concat IDM</td></tr><tr><td>Epoch</td><td>Success ↑</td><td>NFE ↓ Success ↑</td><td></td><td>NFE↓</td></tr><tr><td>1</td><td></td><td>4.3±1.4 1.616±0.041 10.6±1.6</td><td></td><td> $0 . 9 1 4 { \pm } 0 . 0 9 0$ </td></tr><tr><td>3</td><td> $1 0 . 6 { \pm } 0 . 9 $ </td><td>1.287±0.088</td><td> $6 3 . 1 { \pm } 4 . 0 $ </td><td> $0 . 6 9 4 { \pm } 0 . 0 2 8$ </td></tr><tr><td>5</td><td> $2 4 . 4 { \pm } 1 1 . 4 $ </td><td>1.039±0.225</td><td> $7 0 . 8 { \pm } 1 . 6 $ </td><td> $0 . 6 7 8 { \scriptstyle \pm 0 . 0 1 5 }$ </td></tr><tr><td>10</td><td> $9 . 1 { \pm } 2 . 7 $ </td><td>1.353±0.108</td><td> $7 7 . 2 { \pm } 0 . 2 $ </td><td> $0 . 6 5 2 { \pm } 0 . 0 3 2$ </td></tr></table>

Action supervision therefore changes the trajectory of planning performance during training, not only the final checkpoint comparison. The lower NFE already visible at epoch 1 also indicates progress toward the goal before either model achieves high success.

Scope and limitations. The principal finding concerns planning with seen appearances and matching current and goal renderings. Appearance remains decodable, and success on a disjoint palette is 4.2% for Plain JEPA and 13.4% for Concat IDM, leaving substantial room for transfer to unseen appearances. Directly measuring action-outcome merging in trained models would further connect the proposition to planning behavior.

## 5 CONCLUSION

Action prediction penalizes the merging of distinguishable action outcomes that forward latent prediction can leave unpenalized. Under controlled appearance variation over fixed physical trajectories, Concat IDM improves planning, with larger gains at high visual diversity on Push-T and TwoRoom and consistent benefits across independent appearance assignments. Encoder interventions, action probes, and candidate-cost comparisons link these gains to correctly paired action supervision, accessible transition information, and improved latent rollout consistency. These findings motivate supervising action–consequence distinctions alongside forward prediction when learning planning representations from visually diverse observations.

In this work, we used generative AI tools for drafting and editing prose, assisting with the development and checking of the theoretical analysis, and supporting code development and experiment management. We have reviewed all AI-assisted text and code; all scientific claims, experimental protocols, numbers, and analyses were verified by the authors. We take responsibility for the fina content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

Code, configuration files, and evaluation scripts are planned for public release. We report closed-loop success together with normalized final error. Continuous-error metrics use episodes with finite recorded errors and initial distance at least 10<sup>−6</sup>; success includes all episodes. Every task, method, and appearance-set size in Table 1 uses three independently trained models, each evaluated on 500 episodes. The additional Push-T assignment study uses three training repetitions per method under each of three episode assignments on the same 500 tasks as the canonical models. Unless otherwise specified, sensitivity analyses, diversity sweeps, and additional controls use the same training and evaluation protocol.

## REFERENCES

Pulkit Agrawal, Ashvin Nair, Pieter Abbeel, Jitendra Malik, and Sergey Levine. Learning to poke by poking: Experiential learning of intuitive physics. In Advances in Neural Information Processing Systems (NeurIPS), 2016.

Guo An, Zijing Wu, Honghua Dong, Yuhao Yan, Zixuan Gui, Haochong Chen, Shanzhao Ruan, Xiang Wang, Yurong Ling, and Qi Tian. Diagnosing JEPA world models with action-conditioned predictive consistency, 2026.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025. URL https://arxiv.org/abs/2506.09985.

Jiaxin Bai and Jiaxuan Xiong. Temporal-distance JEPA: Plan-aware representation learning for latent world model predictive control, 2026.

Randall Balestriero and Yann LeCun. LeJEPA: Provable and scalable self-supervised learning without the heuristics. arXiv preprint arXiv:2511.08544, 2025. URL https://arxiv.org/abs/ 2511.08544.

Adrien Bardes, Jean Ponce, and Yann LeCun. VICReg: Variance-invariance-covariance regularization for self-supervised learning. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=xm6YD62D1Ub.

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. Transactions on Machine Learning Research, 2024. URL https://openreview. net/forum?id=QaCCuDfBk2. V-JEPA.

Yonathan Efroni, Dipendra Misra, Akshay Krishnamurthy, Alekh Agarwal, and John Langford. Provable RL with exogenous distractors via multistep inverse dynamics. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id= RQLLzMCefQu.

Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. Domain-adversarial training of neural networks. Journal ofMachine Learning Research, 17(59):1–35, 2016.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 2555–2565, 2019. URL https://proceedings.mlr.press/ v97/hafner19a.html.

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=S1lOTC4tDS.

Nicklas A. Hansen, Hao Su, and Xiaolong Wang. Temporal difference learning for model predictive control. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 8387–8406, 2022. URL https:// proceedings.mlr.press/v162/hansen22a.html.

Riashat Islam, Manan Tomar, Alex Lamb, Yonathan Efroni, Hongyu Zang, Aniket Rajiv Didolkar, Dipendra Misra, Xin Li, Harm van Seijen, Remi Tachet des Combes, and John Langford. Principled offline RL in the presence of rich exogenous information. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 14390–14421, 2023. URL https://proceedings.mlr.press/v202/islam23a. html.

Ilya Kostrikov, Denis Yarats, and Rob Fergus. Image augmentation is all you need: Regularizing deep reinforcement learning from pixels. arXiv preprint arXiv:2004.13649, 2020. URL https: //arxiv.org/abs/2004.13649.

Alex Lamb, Riashat Islam, Yonathan Efroni, Aniket Rajiv Didolkar, Dipendra Misra, Dylan J. Foster, Lekan P. Molu, Rajan Chari, Akshay Krishnamurthy, and John Langford. Guaranteed discovery of control-endogenous latent states with multi-step inverse models. Transactions on Machine Learning Research, 2023. URL https://openreview.net/forum?id=99J6lWgXCl.

Michael Laskin, Aravind Srinivas, and Pieter Abbeel. CURL: Contrastive unsupervised representations for reinforcement learning. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings ofMachine Learning Research, pp. 5639–5650, 2020a. URL https://proceedings.mlr.press/v119/laskin20a.html.

Misha Laskin, Kimin Lee, Adam Stooke, Lerrel Pinto, Pieter Abbeel, and Aravind Srinivas. Reinforcement learning with augmented data. In Advances in Neural Information Processing Systems, volume 33, 2020b. URL https://proceedings.neurips.cc/paper/2020/hash/ e615c82aba461681ade82da2da38004a-Abstract.html.

Alexander Levine, Peter Stone, and Amy Zhang. Multistep inverse is not all you need. Reinforcement Learning Journal, 2:884–925, 2024. URL https://rlj.cs.umass.edu/2024/papers/ Paper117.html.

Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. LeWorldModel: Stable end-to-end joint-embedding predictive architecture from pixels, 2026.

Keiran Paster, Sheila A. McIlraith, and Jimmy Ba. Planning from pixels using inverse dynamics models. In International Conference on Learning Representations (ICLR), 2021.

Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In Proceedings of the International Conference on Machine Learning (ICML), 2017.

Max Schwarzer, Ankesh Anand, Rishab Goel, R. Devon Hjelm, Aaron Courville, and Philip Bachman. Data-efficient reinforcement learning with self-predictive representations. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum? id=uCQfPZwRaUu.

Gawon Seo, Dongwon Kim, and Suha Kwak. ACID: Action consistency via inverse dynamics for planning with world models, 2026.

Vlad Sobal, Wancong Zhang, Kyunghyun Cho, Randall Balestriero, Tim G. J. Rudner, and Yann LeCun. Learning from reward-free offline data: A case for planning with latent dynamics models. arXiv preprint arXiv:2502.14819, 2025. PLDM.

Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017.

Leonardo F. Toso, Davit Shadunts, Yunyang Lu, Nihal Sharma, Donglin Zhan, Nam H. Nguyen, and James Anderson. Learning invariant visual representations for planning with joint-embedding predictive world models, 2026.

Amy Zhang, Rowan McAllister, Roberto Calandra, Yarin Gal, and Sergey Levine. Learning invariant representations for reinforcement learning without reconstruction. In International Conference on Learning Representations (ICLR), 2021.

Zhenghao Zhang, Yuanxiang Wang, Zhenyu Guan, Yujia Yang, Bingkang Shi, Tianyu Zong, Hongzhu Yi, Guoqing Chao, Xingchen Chen, Tiankun Yang, Chenxi Bao, Tao Yu, Jingjing Zhou, and Jungang Xu. Delta-JEPA: Learning action-sensitive world models via latent difference decoding, 2026.

Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. DINO-WM: World models on pre-trained visual features enable zero-shot planning. In Proceedings of the International Conference on Machine Learning (ICML), 2025.