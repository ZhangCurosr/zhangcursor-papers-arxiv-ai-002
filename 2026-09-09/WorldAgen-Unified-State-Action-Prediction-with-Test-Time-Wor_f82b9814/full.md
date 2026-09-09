# WorldAgen: Unified State-Action Prediction with Test-Time World Model Training

Chi Wan<sup>1\*</sup>, Kangrui Wang<sup>1\*†</sup>, Yuan Si<sup>1</sup>, Pingyue Zhang<sup>1</sup>, Manling Li<sup>1</sup>

<sup>1</sup>Northwestern University

https://worldagen.github.io

## Abstract

How can vision-language-action (VLA) models adapt to new environments where world dynamics shift? While recent research has combined world modeling and action prediction to improve VLA performance, existing methods largely rely on pretraining on static datasets, without mechanisms for active adaptation at deployment time. As a result, these models often fail to generalize when deployed in unseen scenarios with novel object configurations or dynamics. We present WorldAgen, a unified framework that jointly learns world modeling and action prediction while enabling Test-Time Training (TTT) to adapt to new environments. WorldAgen employs a shared Transformer backbone with two heads: (1) a world model head that predicts future states from past state-action trajectories, and (2) an agent model head that predicts actions conditioned on task instructions. We design a Mixed Unidirectional Attention Mask to separate these two models. During test time, WorldAgen samples exploratory actions, collects ground-truth state transitions, and performs lightweight TTT updates to refine its world model. This adaptation improves the model’s understanding of the environment and leads to more accurate action predictions. Experiments on the CALVIN and LIBERO benchmarks demonstrate that our baseline model achieves comparable, and in some cases superior, performance to current state-of-the-art approaches. Moreover, with TTT on a small number of samples, our method surpasses existing state-of-the-art models, highlighting the effectiveness of adapting world models at inference time.

## Introduction

Vision-language-action (VLA) models have emerged as a powerful and popular paradigm for robotic manipulation (Ma et al. 2025; Din et al. 2025), enabling agents to follow natural language instructions and act directly from raw visual observations. Recent work has explored joint stateaction prediction techniques (Tian et al. 2024; Guo et al. 2024; Liu et al. 2025; Xian et al. 2023; Wolf et al. 2025), allowing models not only to predict the next actions but also to anticipate how their actions will transform the environment. This joint formulation has improved data efficiency and scene understanding, leading to stronger performance across manipulation benchmarks.

However, we argue that current methods remain fundamentally limited by training on static datasets (Kim, Finn, and Liang 2025; Li et al. 2025; Kim et al. 2024). Once pretrained, these models lack mechanisms to adapt their internal representations of world dynamics when deployed in novel environments. In realistic settings, distribution shifts such as new object layouts, lighting conditions, or physical properties are inevitable, and static world modeling fails to capture these variations (Wu et al. 2023b; Lanier et al. 2025). This leads to a key question: How can we enable VLA models to actively adapt their understanding of the environment during test time?

As shown in Figure 1, we address this question with WorldAgen, a unified framework that combines joint stateaction prediction with a novel TTT strategy:

• Joint State-Action Modeling. WorldAgen uses a shared Transformer backbone with two heads. The world model head predicts future states from historical state-action trajectories, while the policy head predicts actions conditioned on task instructions. By training these tasks jointly, WorldAgen aligns scene understanding with action prediction.

• Test-Time Training for Scene Adaptation. During test time, WorldAgen executes exploratory actions to collect ground-truth state transitions. It then performs lightweight, LoRA-based TTT updates to refine its world model. These updates improve the internal representation of the environment, indirectly boosting the agent’s ability to generate effective actions in novel scenarios, as depicted in Figure 1(b).

Our work reframes world modeling from a passive pretraining objective into an active test time adaptation mechanism, bridging the gap between offline training and real-world generalization.

In sum, we make the following contributions:

• Unified joint state-action prediction architecture. We present a unified Transformer-based structure that integrates world modeling and action prediction, enabling shared representations and work separately via a Mixed Unidirectional Attention Mask.

(a) WorldAgen Architecture  
![](images/446888dc23c56ab9c461cb56de11c3b90b53ec90f846367e014c36080b96ce8d.jpg)

(b) TTT Pipeline  
![](images/ab51149b2a0d77d00f681d54f5a491bd21b37f5c360e41cf53713aa6442300ef.jpg)

![](images/55fdfd9733b79545f01cd70da6b946cd8a235b844e6cb7260d9781713df4f87c.jpg)

(c) TTT Performance Gain  
![](images/a8df7753c65c526b1ea7ff746dbff5946fc8a0a489c0bec6a409f92d4bda6b47.jpg)  
Figure 1: Overview of WorldAgen. (a) WorldAgen Architecture. WorldAgen unifies an agent policy head for task-conditioned action prediction and a world modeling head for task-agnostic state prediction within a single Transformer backbone. At trajectory unit i, the agent receives the task instruction $G _ { i }$ , the observation chunk $O _ { i } ,$ and the robot state chunk $S _ { i }$ , the action placeholder $A _ { p }$ and observation placeholder $O _ { p }$ . Then it predicts the action chunk ${ \hat { A } } _ { i }$ . After that, the world model predicts the future observation chunk $\hat { O } _ { i }$ to refine the shared representation of environment dynamics. (b) Two-stage TTT pipeline. Stage 1 performs random exploration in the new environment to collect unlabeled trajectories. Stage 2 adapts the world model using LoRA-based fine-tuning on the observation loss $\mathcal { L } _ { o } ,$ improving environment modeling without modifying the agent policy. (c) TTT Performance. We tested WorldAgen with TTT strategy on CALVIN and LIBERO benchmark, which show great performance gain.

• Active test-time adaptation for VLA models. We introduce a TTT paradigm that transforms world modeling from a static pretraining into an active, test time adaptation mechanism, allowing VLA models to refine their scene understanding on the fly.

• Empirical validation across challenging benchmarks. Our baseline achieves performance comparable to, or even better than, state-of-the-art methods. Furthermore, by fine-tuning the world modeling component with only a small number of samples at test time, our method attains state-of-the-art results on both the CALVIN and LIBERO benchmarks.

We believe these results demonstrate that continuous adaptation during test time, rather than merely scaling model size or pretraining data, is a key ingredient for robust and generalizable robotic manipulation.

## Method

## Task Formulation

We frame our task as a Partially Observable Markov Decision Process (POMDP). Each rollout trajectory is divided into a sequence of trajectory units indexed by i. A trajectory unit is defined as $\bar { U _ { i } } \stackrel { \cdot } { = } ( \bar { G _ { i } } , O _ { i } , S _ { i } , A _ { i } , A _ { p } , \stackrel { \cdot } { O _ { p } } )$ , where $G _ { i }$ denotes the task instruction, and $O _ { i } , S _ { i } ,$ and ${ \bar { A } } _ { i }$ denote the observation, robot state, and action chunks, respectively, accompanied by an action placeholder $A _ { p }$ and an observation placeholder $O _ { p } .$

Each chunk $( \mathrm { i . e . , } O _ { i } , S _ { i } ,$ and $A _ { i } )$ consists of a sequence of time-indexed elements, including observations, states, or actions extracted from the raw trajectory τ. Each element $( \mathrm { i } . \mathrm { e } . , o _ { t } , s _ { t } , a _ { t } )$ is further tokenized into a sequence of tokens that serves as the actual model input.

In unit i, the model first predicts an action chunk ${ \hat { A } } _ { i }$ conditioned on the instruction $G _ { i }$ , the observation chunk $O _ { i } .$ and the state chunk $S _ { i }$ . It then predicts the next observation chunk conditioned on $( O _ { i } , \bar { S } _ { i } , A _ { i } )$ during training or $( O _ { i } , S _ { i } , \hat { A } _ { i } )$ during inference<sup>1</sup>. Executing the predicted actions advances the environment and produces the next unit $U _ { i + 1 } .$ . A summary of the key notations used in the WorldAgen framework is provided in Table 12 in the Appendix.

## Joint State–Action Prediction

WorldAgen contains two predictive components: a taskconditioned agent model and a task-agnostic world model. Both models share a single Transformer backbone, enabling a unified representation across tasks, as illustrated in Figure 1(a).

We maintain two types of histories:

Agent history:

$$
h _ { i } ^ { a } = ( G _ { i } , O _ { x } , S _ { x } , \hat { A } _ { x } ) _ { x = i - k } ^ { i - 1 } ,
$$

which contains the task instruction $G _ { i }$ along with the observation, robot state, and predicted action chunks from the previous k segments. This history is used by the agent model for task-aware action prediction.

World history:

$$
h _ { i } ^ { w } = ( O _ { x } , S _ { x } , \hat { A } _ { x } ) _ { x = i - k } ^ { i - 1 } ,
$$

which includes the observation, robot state, and predicted action chunks from the previous k segments, and is used by the world model to predict task-agnostic environment $\mathrm { { d y } \mathrm { { - } } }$ namics.

We define the two models at trajectory segment $U _ { i }$ as follows:

Agent Model. The agent model predicts the next action chunk ${ \hat { A } } _ { i }$ conditioned on the current observation chunk $O _ { i } ,$ robot state chunk $S _ { i }$ , task instruction $G _ { i }$ , and agent history $h _ { i } ^ { a }$

$$
p _ { a } \biggl ( \hat { A } _ { i } \ : \vert \ : G _ { i } , O _ { i } , S _ { i } , h _ { i } ^ { a } \biggl ) .
$$

World Model. The world model predicts the next observation chunk $\hat { O } _ { i }$ conditioned on the current observation chunk $O _ { i } ,$ robot state chunk $S _ { i } ,$ , predicted action chunk ${ \hat { A } } _ { i } ,$ and world history $h _ { i } ^ { w }$

$$
p _ { w } \Big ( \hat { O } _ { i } \vert O _ { i } , S _ { i } , \hat { A } _ { i } , h _ { i } ^ { w } \Big ) .
$$

During training, we adopt teacher forcing and replace all predicted actions ${ \hat { A } } _ { i }$ with the ground-truth actions $A _ { i }$

## Trajectory Splitting and Action Chunking

Predicting multiple actions at once has been shown to improve the execution efficiency of real-world robots (Intelligence et al. 2025). However, consecutive observation frames often contain highly redundant visual information, and predicting long observation sequences slows down both training and deployment. To balance these factors, WorldAgen supports different action chunk lengths m and observations chunk length n. We further propose a robust trajectory splitting and action chunking scheme that flexibly adapts to different experimental configurations.

Concretely, given a rollout ${ \boldsymbol { \tau } } = \{ ( o _ { t } , s _ { t } , a _ { t } ) \} _ { t = 0 } ^ { T - 1 }$ with $T$ timesteps, we transform it into a sequence of trajectory units $\{ U _ { i } \} _ { i = 0 } ^ { I - 1 }$ with I units. The initial unit $U _ { 0 }$ aggregates the first n timesteps of the rollout and is paired with the first action chunk $A _ { 0 } = A _ { 0  n - 1 } = [ a _ { 0 } , a _ { 1 } , \dotsc , a _ { n - 1 } ] $ . For units $U _ { i }$ with $i > 0$ , we uniformly sample n timesteps to construct the current observation chunk $O _ { i } .$ , state chunk $S _ { i }$ , and associated action chunk $A _ { i } { \mathrm { : } }$

$$
\begin{array} { r l } & { { \mathcal O } _ { i } = { \mathcal O } _ { n + ( i - 1 ) m  n + i m - 1 } ^ { n } } \\ & { \phantom { { \mathcal O } _ { n + ( i - 1 ) m } } = \bigl [ o _ { n + ( i - 1 ) m } , o _ { n + ( i - 1 ) m + \lfloor \frac { 1 } { n } m \rfloor } , \cdots , o _ { n + ( i - 1 ) m + \lfloor \frac { n - 1 } { n } m \rfloor } \bigr ] , } \\ & { { S } _ { i } = { S } _ { n + ( i - 1 ) m  n + i m - 1 } ^ { n } } \\ & { \phantom { { O } _ { n + ( i - 1 ) m } } = \bigl [ s _ { n + ( i - 1 ) m } , s _ { n + ( i - 1 ) m + \lfloor \frac { 1 } { n } m \rfloor } , } \\ & { \phantom { { O } _ { n + ( i - 1 ) m + \lfloor \frac { n - 1 } { n } m \rfloor } } \cdots , s _ { n + ( i - 1 ) m + \lfloor \frac { n - 1 } { n } m \rfloor } \bigr ] , } \\ & { \phantom { { O } _ { n + ( i - 1 ) m + \lfloor \frac { n - 1 } { n } m \rfloor } } A _ { i } = A _ { n + ( i - 1 ) m - n + i m - 1 } } \\ & { \phantom { { O } _ { n + ( i - 1 ) m + \lfloor \frac { n - 1 } { n } m \rfloor } } = \bigl [ a _ { n + ( i - 1 ) m + \lfloor \frac { 1 } { n } m + 1 } , \dots , a _ { n + i m - 1 } \bigr ] . } \end{array}
$$

This procedure keeps each observation chunk fixed to length $n ,$ reducing redundant visual information and improving runtime efficiency.

![](images/a46c75bc4412bb78e87337d88114de6495ec369447e5cda007ebb04f6f7414d8.jpg)

![](images/bc8cd2bd82535c6b903870f39659f83ecca607f357897148f43425aac99762e5.jpg)  
Figure 2: Trajectory Splitting, Chunking, and Two-Step Inference. Example with observation chunk length $n = 2$ and action chunk length $m = 3 .$ . (a) Trajectory Splitting and Chunking: a raw trajectory is divided into trajectory units $U _ { i }$ , where each unit contains an observation chunk $O _ { i }$ and a robot state chunk $S _ { i }$ obtained via uniform subsampling from m consecutive observations and states, and an action chunk $A _ { i }$ obtained by grouping m actions. (b) Two-Step Inference: the agent model first initializes a zero action placeholder $A _ { p }$ and then replaces it with the predicted action chunk ${ \hat { A } } _ { i }$ . The world model subsequentl predicts the next observation chunk $\hat { O } _ { i }$ , conditioned on the updated trajectory unit.

Figure 2 (a) illustrates an example with $ { n _ { \mathrm { ~  ~ } } } = \mathrm { ~  { ~ 2 ~ } }$ and $m = 3 .$ Algorithm of trajectory splitting and action chunking for variable length of action, observation and robot state chunk is shown in Algorithm 1 in the Appendix, where UNIFORMSUBSAMPLE(a, b, n) uniformly samples n distinct indices from the integer interval $\{ a , \ldots , b \}$ without replacement and returns them in ascending order, which we then use to construct the observation and state chunks.

## Mixed Unidirectional Attention Mask

To ensure that the world modeling and agent policy tasks are trained jointly without information leakage, we introduce a mixed unidirectional attention masking mechanism as shown in Figure 3. This mechanism integrates two complementary components:

Local mask: ensures that the current observation chunk $O _ { i }$ , the robot proprioception chunk $S _ { i }$ , the task instruction $G _ { i } ,$ , and the action placeholder $A _ { p }$ cannot access the action chunk $A _ { i }$ . This prevents information leakage within a single timestep.

Global mask: guarantees that the world model head remains invisible to the task instruction $G _ { i }$ across all time steps. Additionally, the global mask enforces mutual invisibility among the task instruction and observations to avoid cross-modal leakage.

Meanwhile, the general structure conforms to strictly causal prediction, such that both the world model and policy heads cannot access any future tokens.

## Pretraining

We pretrain WorldAgen using large-scale trajectory data collected from robot demonstrations. We split trajectories using our proposed robust splitting method. During pretraining, we jointly optimize the agent model and world model using teacher forcing:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { a } + \lambda \mathcal { L } _ { o } , } \end{array}
$$

where ${ \mathcal { L } } _ { a }$ is the cross-entropy loss for predicting future action chunks, $\scriptstyle { \mathcal { L } } _ { o }$ is the reconstruction loss for predicting future observation chunks, and λ is a weighting factor balancing the two objectives.

It is worth noting that during training, we feed the groundtruth action chunk (teacher forcing), whereas during testing, the model performs autoregressive prediction by feeding the previously predicted action chunk at each timestep.

## Two-Step Inference

At test time, WorldAgen runs in a two-step inference loop over trajectory units as shown in Figure 2 (b). For unit i, we first perform the agent model step. We construct the input sequence with a zero-initialized action placeholder $A _ { p }$ and feed it to the agent model, which predicts the current action chunk ${ \hat { A } } _ { i } .$ The predicted actions then overwrite the placeholder $A _ { p } .$

Next, we perform the world model step. Conditioned on the task instruction $G _ { i } ,$ the current observation chunk $O _ { i }$ and the updated action chunk $A _ { i }$ (same as ${ \hat { A } } _ { i } )$ , the world model predicts the next observation chunk $\hat { O } _ { i }$ . During testing, these two steps are applied alternately to roll out the whole trajectory.

![](images/9c983d5d3d3a90f03cd339907e29c32ada97e8a17274c2989537869604721412.jpg)  
Figure 3: Proposed Mixed Unidirectional Attention Mask: combines local and global masking. For each trajectory unit, WorldAgen applies a local mask that prevents the action chunk $A _ { p }$ from attending to the corresponding action placeholder ${ \bar { A _ { i } } } .$ A global mask is further used to construct a taskagnostic world model head, so that the observation placeholder $O _ { p }$ will not attend to the task instruction $G _ { i }$ . General structure conforms to strictly causal prediction.

## Test-Time Training

While joint pretraining improves world understanding, distribution shifts in novel environments can still degrade performance. To address this, we introduce a two-stage Test-Time Training strategy that adapts the world model online, refining its understanding of environment dynamics and indirectly improving downstream action prediction. The whole pipeline is given in Figure 1 (b).

Stage 1: Random Exploration and Data Collection. At deployment, the agent first performs free exploratory rollouts in the target environment. During this phase, it records trajectories of $( O _ { i } , S _ { i } , A _ { i } )$ . To ensure that adaptation focuses on environment dynamics rather than task-specific instructions, all collected trajectories are relabeled with a generic “no-lang” token in place of language instructions.

Stage 2: World Model Adaptation. Using the collected trajectories, we adapt only the shared backbone with LoRAbased parameter-efficient fine-tuning, while keeping the policy head frozen. The update rule is:

$$
\theta _ { w } ^ { \prime }  \theta _ { w } - \eta \nabla _ { \theta _ { w } } \mathcal { L } _ { o } ,
$$

where $\theta _ { w }$ are the parameters of the world model head and η is the learning rate.

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>T1  T2  T3  T4  T5</td><td rowspan=1 colspan=1>Avg. Len. ↑</td></tr><tr><td rowspan=1 colspan=1>RoboFlamingo</td><td rowspan=1 colspan=1>82.461.9 46.633.1 23.5</td><td rowspan=1 colspan=1>2.47</td></tr><tr><td rowspan=1 colspan=1>SuSIE</td><td rowspan=1 colspan=1>87.069.0 49.038.0 26.0</td><td rowspan=1 colspan=1>2.69</td></tr><tr><td rowspan=1 colspan=1>GR-1</td><td rowspan=1 colspan=1>85.4 71.2 59.6 49.7 40.1</td><td rowspan=1 colspan=1>3.06</td></tr><tr><td rowspan=3 colspan=1>3D Diffuser ActorCLOVERSeer</td><td rowspan=2 colspan=1>92.2 78.7 63.9 51.2 41.296.0 83.5 70.8 57.5 45.4</td><td rowspan=1 colspan=1>3.27</td></tr><tr><td rowspan=1 colspan=1>3.53</td></tr><tr><td rowspan=1 colspan=1>93.0 82.4 72.3 62.6 53.3</td><td rowspan=1 colspan=1>3.64</td></tr><tr><td rowspan=1 colspan=1>Seer-Large</td><td rowspan=1 colspan=1>92.784.6 76.1 68.960.3</td><td rowspan=1 colspan=1>3.83</td></tr><tr><td rowspan=1 colspan=1>WorldAgenWorldAgen-TTT</td><td rowspan=1 colspan=1>96.387.7 76.867.359.196.6 88.5 78.5 68.7 60.5</td><td rowspan=1 colspan=1>3.873.93</td></tr></table>

Table 1: Performance comparison on the CALVIN benchmark. We report the success rate (%) for completing 5 consecutive tasks and the average sequence length (ASL).

This targeted update improves world understanding ability and enhances the shared representations, which are in turn leveraged by the agent model for action prediction.

By decoupling adaptation from the task goal and restricting it to the world model, our TTT procedure enables robust scene adaptation without requiring additional taskspecific annotations, paving the way for scalable testtime adaptation in VLA models.

## Experiments

## Datasets

CALVIN. CALVIN (Mees et al. 2022) is an open-source simulated benchmark designed for learning long-horizon language-conditioned robot manipulation tasks. The benchmark requires agents to solve complex manipulation tasks by understanding a series of unconstrained language instructions in sequence.

LIBERO. LIBERO (Liu et al. 2023a) is a comprehensive benchmark for lifelong learning in robot manipulation that emphasizes knowledge transfer across diverse tasks.

## Implementation Details

Test-Time Training TTT represents a crucial component of WorldAgen that enables adaptive performance improvements during inference. In the TTT phase, we apply LoRA fine-tuning to the backbone of Qwen3. Specifically, we apply LoRA to the attention projection layers (q proj, k proj, v proj, o proj) and the MLP projection layers (gate proj, up proj, down proj) of Qwen3.

The test-time training procedure, including sampling and LoRA adaptation, can be executed on an RTX 4090 GPU and requires approximately 8 minutes per task for CALVIN and 2 minutes per scene for LIBERO. More details can be found in Appendix.

## Results

Our experimental results demonstrate the effectiveness of WorldAgen across both CALVIN and LIBERO benchmarks. CALVIN As shown in Table 1, we compare WorldAgen with recent VLA baselines, including RoboFlamingo (Li et al. 2023), SuSIE (Black et al. 2023), GR-1 (Wu et al.

<table><tr><td>Method</td><td>Avg. Success ↑</td><td>Put soup and box in basket</td><td>Put box and butter in basket</td><td>Turn on stove and put pot</td><td>Put bowl in drawer and close it</td></tr><tr><td>MT-ACT</td><td>41.0</td><td>30.0</td><td>50.0</td><td>75.0</td><td>85.0</td></tr><tr><td>MVP</td><td>68.2</td><td>83.3</td><td>90.0</td><td>80.0</td><td>88.3</td></tr><tr><td>MPI</td><td>77.3</td><td>66.6</td><td>86.6</td><td>96.6</td><td>95.0</td></tr><tr><td>OpenVLA</td><td>54.0</td><td>35.0</td><td>95.0</td><td>65.0</td><td>45.0</td></tr><tr><td>Seer</td><td>78.7</td><td>80.0</td><td>90.0</td><td>91.7</td><td>81.7</td></tr><tr><td>WorldAgen WorldAgen-TTT</td><td>75.5 79.0</td><td>70.0 85.0</td><td>75.0 75.0</td><td>95.0 95.0</td><td>100 100</td></tr><tr><td>Put mugs on</td><td>Pick book</td><td>Put mug on</td><td>Put soup</td><td></td><td>Put mug in</td></tr><tr><td>left and right plates</td><td>and place it in back</td><td>plate and put pudding to right</td><td>and sauče in basket</td><td>Put both pots on stove</td><td>microwave and close it</td></tr><tr><td>20.0</td><td>75.0</td><td>0.0</td><td>0.0</td><td>10.0</td><td>65.0</td></tr><tr><td>46.7</td><td>63.3</td><td>45.0</td><td>78.3</td><td>60.0</td><td>46.7</td></tr><tr><td>83.3</td><td>83.3</td><td>56.6</td><td>86.6</td><td>40.0</td><td>78.3</td></tr><tr><td>40.0</td><td>80.0</td><td>60.0</td><td>45.0</td><td>20.0</td><td>55.0</td></tr><tr><td>85.0</td><td>65.0</td><td>86.7</td><td>88.3</td><td>51.7</td><td>66.7</td></tr><tr><td>85.0</td><td>90.0</td><td>60.0</td><td>100</td><td>45.0</td><td>35.0</td></tr><tr><td>90.0</td><td>90.0</td><td>50.0</td><td>100</td><td>45.0</td><td>60.0</td></tr></table>

Table 2: Performance comparison on LIBERO benchmark. We report the success rate (%) across different manipulation tasks.

2023a), 3D Diffuser Actor (Ke, Gkanatsios, and Fragkiadaki 2024), and CLOVER (Bu et al. 2024). WorldAgen follows a GR-1-style GPT video architecture, but introduces the Mixed Unidirectional Attention Mask to separate task instruction from world modeling and employs a lightweight TTT strategy to enhance world-modeling capability. Our method achieves consistent performance improvements across five consecutive tasks, indicating that strengthening world modeling via TTT improves environment understanding and boosts long-horizon manipulation performance.

LIBERO As shown in Table 2, we further evaluate WorldAgen on the LIBERO benchmark against multi-task VLA baselines such as MT-ACT (Bharadhwaj et al. 2024), MVP (Xiao et al. 2022), MPI (Zeng et al. 2024), Open-VLA (Kim et al. 2024) and Seer (Tian et al. 2024). WorldAgen attains the best or on-par performance across most LIBERO subsets, further supporting that test-time adaptation of the world model leads to stronger environment understanding and improved robustness to distribution shifts.

We observe improvements or maintained performance across almost every task compared to the baseline. These consistent gains across diverse manipulation scenarios mirror our findings on CALVIN, demonstrating robustness and generalization ability of our TTT approach across different benchmarks and scenario dynamics.

## Ablation Study

To better understand the key components contributing to performance, we conduct ablation studies on (1) the contribution of world modeling to policy learning, (2) the effect of

LoRA parameterization during TTT, (3) the impact of the amount of randomly sampled data used for TTT, and (4) transfer from simulator to the real robot. We further analyze the impact of trajectory splitting and preprocessing, image and action chunk lengths, and different Transformer backbones, and compare LoRA-based TTT with full fine-tuning while visualizing performance before and after TTT. All ablation and additional experimental results are provided in the Appendix.

<table><tr><td>Dataset</td><td>World Modeling</td><td>Avg. Success ↑</td></tr><tr><td rowspan="2">CALVIN</td><td>X</td><td>2.96</td></tr><tr><td>√</td><td>3.87</td></tr><tr><td rowspan="2">LIBERO</td><td>X</td><td>46.5%</td></tr><tr><td>√</td><td>75.5%</td></tr></table>

Table 3: Ablation study on world modeling. We compare models with and without image prediction to evaluate the contribution of world modeling to agent policy learning.

World Modeling Ability To validate the contribution of world modeling to policy learning, we compare models with and without image prediction capability by including or removing image-prediction tokens in the input as shown in Table 3. World modeling consistently improves performance: on CALVIN, the average success length increases from 2.96 to 3.87 (+30.7%), and on LIBERO, the success rate rises from 46.5% to 78.0% (+67.7%). These results indicate that world modeling yields richer representations of environment dynamics, leading to more effective action prediction

<table><tr><td>LoRA Rank</td><td>Avg. Len. ↑</td></tr><tr><td>16</td><td>3.928</td></tr><tr><td>32</td><td>3.918</td></tr><tr><td>64</td><td>3.918</td></tr><tr><td>128</td><td>3.930</td></tr><tr><td>256</td><td>3.923</td></tr></table>

Table 4: Ablation study on LoRA rank during TTT in CALVIN dataset. All other parameters are kept constant.

LoRA Configuration We study the effect of LoRA rank on TTT performance while fixing the learning rate and training data size as shown in Table 4. Performance only varies within 0.02 across ranks, indicating that under fixed hyperparameters, TTT performance is largely insensitive to the LoRA rank. We therefore use a LoRA rank of 128 in our main experiments, which slightly outperforms other settings while remaining computationally efficient.

<table><tr><td>Test Time Training Data</td><td>Avg. Lens. ↑</td></tr><tr><td>6</td><td>3.871</td></tr><tr><td>90</td><td>3.922</td></tr><tr><td>204</td><td>3.928</td></tr><tr><td>340</td><td>3.917</td></tr></table>

Table 5: Ablation study on TTT data sampling volume. We vary the amount of test-time training data, defined as the product of number of samples and repeat times.

TTT Data Sampling Volume We study how the amount of test-time training data affects performance as shown in Table 5. As the number of TTT samples increases from 6 to 204, the average success score improves from 3.871 to 3.928, indicating that more diverse exploration data benefits world-model adaptation. However, further increasing the data to 340 slightly reduces performance to 3.917, suggesting diminishing returns and mild overfitting to image generation rather than action prediction. These results indicate that moderately increasing TTT data improves performance, but excessive data can be counterproductive, highlighting the needfor an appropriate TTT data budget.

<table><tr><td></td><td>w/o TTT</td><td>w/o Noise</td><td>w/ Noise</td></tr><tr><td>CALVIN</td><td>3.87</td><td>3.93</td><td>3.90</td></tr><tr><td>LIBERO</td><td>75.5</td><td>79.0</td><td>78.0</td></tr></table>

Table 6: Performance comparison under Gaussian noise perturbation.

Simulator to Real World We add Gaussian noise (std = 0.1) to the observations to simulate real-world conditions, such as camera distortion and sensor noise, as shown in Table 6.

This experiment indicates that even under noisy conditions, our method consistently outperforms the baselines, demonstrating its robustness and strong adaptability to real-world uncertainty.

## Related Work

Vision-Language-Action Models. Vision-Language-Action (VLA) models unify perception, language, and control for robotic manipulation (Sapkota et al. 2025; Din et al. 2025). Early systems such as RT-1 and RT-2 established the effectiveness of large-scale transformer policies, while PaLM-E and LLaVA further incorporated multimodal grounding (Brohan et al. 2023; Zitkovich et al. 2023; Driess et al. 2023; Liu et al. 2023b). However, most VLA models directly predict actions without explicitly modeling world dynamics, limiting generalization to novel environments (Zhang et al. 2025).

World Models. World models learn predictive dynamics to support planning and decision-making (Sutton 1990; Hafner et al. 2025). Neural approaches such as World Models and Dreamer improve sample efficiency and longhorizon reasoning (Ha and Schmidhuber 2018; Hafner et al. 2022, 2025). In robotics, however, these models are typically used for state prediction or model-based planning and remain largely decoupled from language-grounded action generation (Sakagami et al. 2023; Wang et al. 2026).

Test-Time Training. Test-time training adapts models to distribution shifts during inference via self-supervised objectives (Sun et al. 2020; Ma 2024). While TTT has been explored in both vision and NLP (Ye et al. 2023), robotic adaptation has mainly focused on perception modules or lowlevel policies (Zhao, Queralta, and Westerlund 2020). Our work instead incorporates TTT into a unified VLA framework by adapting the world model itself during deployment.

## Conclusion

We introduced WorldAgen, a unified vision–language–action framework that jointly learns world modeling and action prediction through a shared Transformer backbone. WorldAgen integrates a world model head that predicts future states and a policy head that predicts task-conditioned actions. At test time, the model performs lightweight Test-Time Training (TTT) using short exploratory rollouts, turning world modeling into an online adaptation signal. This enables the agent to quickly adjust to new environments and leads to consistent performance gains on CALVIN and LIBERO, demonstrating that adaptive VLA systems can benefit substantially from continual test-time interaction.

## Limitations and Future Work

Our study evaluates only a single model size and architecture, and focuses test-time training exclusively on the worldmodel component. Future work includes exploring larger and more diverse model architectures, as well as developing joint test-time-training methods that improve both action prediction and world modeling.

## References

Bharadhwaj, H.; Vakil, J.; Sharma, M.; Gupta, A.; Tulsiani, S.; and Kumar, V. 2024. Roboagent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking. In 2024 IEEE International Conference on Robotics and Automation (ICRA), 4788–4795. IEEE.

Black, K.; Nakamoto, M.; Atreya, P.; Walke, H.; Finn, C.; Kumar, A.; and Levine, S. 2023. Zero-shot robotic manipulation with pretrained image-editing diffusion models. arXiv preprint arXiv:2310.10639.

Brohan, A.; Brown, N.; Carbajal, J.; Chebotar, Y.; Dabis, J.; Finn, C.; Gopalakrishnan, K.; Hausman, K.; Herzog, A.; Hsu, J.; Ibarz, J.; Ichter, B.; Irpan, A.; Jackson, T.; Jesmonth, S.; Joshi, N. J.; Julian, R.; Kalashnikov, D.; Kuang, Y.; Leal, I.; Lee, K.-H.; Levine, S.; Lu, Y.; Malla, U.; Manjunath, D.; Mordatch, I.; Nachum, O.; Parada, C.; Peralta, J.; Perez, E.; Pertsch, K.; Quiambao, J.; Rao, K.; Ryoo, M.; Salazar, G.; Sanketi, P.; Sayed, K.; Singh, J.; Sontakke, S.; Stone, A.; Tan, C.; Tran, H.; Vanhoucke, V.; Vega, S.; Vuong, Q.; Xia, F.; Xiao, T.; Xu, P.; Xu, S.; Yu, T.; and Zitkovich, B. 2023. RT-1: Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.

Bu, Q.; Zeng, J.; Chen, L.; Yang, Y.; Zhou, G.; Yan, J.; Luo, P.; Cui, H.; Ma, Y.; and Li, H. 2024. Closed-loop visuomotor control with generative expectation for robotic manipulation. Advances in Neural Information Processing Systems, 37: 139002–139029.

Din, M. U.; Akram, W.; Saoud, L. S.; Rosell, J.; and Hussain, I. 2025. Vision Language Action Models in Robotic Manipulation: A Systematic Review. arXiv:2507.10672.

Driess, D.; Xia, F.; Sajjadi, M. S.; Lynch, C.; Chowdhery, A.; Wahid, A.; Tompson, J.; Vuong, Q.; Yu, T.; Huang, W.; et al. 2023. PaLM-E: An Embodied Multimodal Language Model. In International Conference on Machine Learning (ICML), 8469–8488. PMLR.

Guo, Y.; Hu, Y.; Zhang, J.; Wang, Y.-J.; Chen, X.; Lu, C.; and Chen, J. 2024. Prediction with action: Visual policy learning via joint denoising process. Advances in Neural Information Processing Systems, 37: 112386–112410.

Ha, D.; and Schmidhuber, J. 2018. World Models. CoRR.

Hafner, D.; Lillicrap, T.; Norouzi, M.; and Ba, J. 2022. Mastering Atari with Discrete World Models. arXiv:2010.02193.

Hafner, D.; Pasukonis, J.; Ba, J.; and Lillicrap, T. 2025. Mastering diverse control tasks through world models. Nature, 640(8059): 647–653.

He, K.; Chen, X.; Xie, S.; Li, Y.; Dollar, P.; and Girshick,´ R. 2022. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 16000–16009.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations (ICLR).

Intelligence, P.; Black, K.; Brown, N.; Darpinian, J.; Dhabalia, K.; Driess, D.; Esmail, A.; Equi, M.; Finn, C.;

Fusai, N.; et al. 2025. π<sub>0.5</sub>: a Vision-Language-Action Model with Open-World Generalization. arXiv preprint arXiv:2504.16054.

Jaegle, A.; Gimeno, F.; Brock, A.; Vinyals, O.; Zisserman, A.; and Carreira, J. 2021. Perceiver: General perception with iterative attention. In International conference on machine learning, 4651–4664. PMLR.

Ke, T.-W.; Gkanatsios, N.; and Fragkiadaki, K. 2024. 3d diffuser actor: Policy diffusion with 3d scene representations. arXiv preprint arXiv:2402.10885.

Kim, M. J.; Finn, C.; and Liang, P. 2025. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. arXiv:2502.19645.

Kim, M. J.; Pertsch, K.; Karamcheti, S.; Xiao, T.; Balakrishna, A.; Nair, S.; Rafailov, R.; Foster, E.; Lam, G.; Sanketi, P.; Vuong, Q.; Kollar, T.; Burchfiel, B.; Tedrake, R.; Sadigh, D.; Levine, S.; Liang, P.; and Finn, C. 2024. Open-VLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.

Kojima, Y.; Xu, J.; Zou, X.; and Wang, X. 2025. LoRA-TTT: Low-Rank Test-Time Training for Vision-Language Models. arXiv:2502.02069.

Lanier, J.; Kim, K.; Karamzade, A.; Liu, Y.; Sinha, A.; He, K.; Corsi, D.; and Fox, R. 2025. Adapting World Models with Latent-State Dynamics Residuals. arXiv:2504.02252.

Li, P.; Wu, Y.; Xi, Z.; Li, W.; Huang, Y.; Zhang, Z.; Chen, Y.; Wang, J.; Zhu, S.-C.; Liu, T.; and Huang, S. 2025. ControlVLA: Few-shot Object-centric Adaptation for Pre-trained Vision-Language-Action Models. arXiv:2506.16211.

Li, X.; Liu, M.; Zhang, H.; Yu, C.; Xu, J.; Wu, H.; Cheang, C.; Jing, Y.; Zhang, W.; Liu, H.; et al. 2023. Visionlanguage foundation models as effective robot imitators. arXiv preprint arXiv:2311.01378.

Liu, B.; Zhu, Y.; Gao, C.; Feng, Y.; Liu, Q.; Zhu, Y.; and Stone, P. 2023a. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36: 44776–44791.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023b. Visual instruction tuning. Advances in neural information processing systems, 36: 34892–34916.

Liu, S.; Wu, L.; Li, B.; Tan, H.; Chen, H.; Wang, Z.; Xu, K.; Su, H.; and Zhu, J. 2025. RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation. arXiv:2410.07864.

Ma, J. 2024. Improved self-training for test-time adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 23701–23710.

Ma, Y.; Song, Z.; Zhuang, Y.; Hao, J.; and King, I. 2025. A Survey on Vision-Language-Action Models for Embodied AI. arXiv:2405.14093.

Mees, O.; Hermann, L.; Rosete-Beas, E.; and Burgard, W. 2022. CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks. IEEE Robotics and Automation Letters (RA-L), 7(3): 7327–7334.

Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.; Sutskever, I.; et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8): 9.

Sakagami, R.; Lay, F. S.; Domel, A.; Schuster, M. J.; Albu-¨ Schaffer, A.; and Stulp, F. 2023. Robotic world mod-¨ els—conceptualization, review, and engineering best practices. Frontiers in Robotics and AI, 10: 1253049.

Sapkota, R.; Cao, Y.; Roumeliotis, K. I.; and Karkee, M. 2025. Vision-Language-Action (VLA) Models: Concepts, Progress, Applications and Challenges. arXiv:2505.04769.

Sun, Y.; Wang, X.; Liu, Z.; Miller, J.; Efros, A. A.; and Hardt, M. 2020. Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. arXiv:1909.13231.

Sutton, R. S. 1990. Integrated architectures for learning, planning, and reacting based on approximating dynamic programming. In Machine learning proceedings 1990, 216– 224. Elsevier.

Tian, Y.; Yang, S.; Zeng, J.; Wang, P.; Lin, D.; Dong, H.; and Pang, J. 2024. Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation. arXiv:2412.15109.

Wang, K.; Zhang, P.; Wang, Z.; Gao, Y.; Li, L.; Wang, Q.; Chen, H.; Lu, Y.; Yang, Z.; Wang, L.; et al. 2026. Vagen: Reinforcing world model reasoning for multi-turn vlm agents. Advances in Neural Information Processing Systems, 38: 172871–172933.

Wolf, R.; Shi, Y.; Liu, S.; and Rayyes, R. 2025. Diffusion Models for Robotic Manipulation: A Survey. arXiv:2504.08438.

Wu, H.; Jing, Y.; Cheang, C.; Chen, G.; Xu, J.; Li, X.; Liu, M.; Li, H.; and Kong, T. 2023a. Unleashing large-scale video generative pre-training for visual robot manipulation. arXiv preprint arXiv:2312.13139.

Wu, P.; Escontrela, A.; Hafner, D.; Abbeel, P.; and Goldberg, K. 2023b. Daydreamer: World models for physical robot learning. In Conference on robot learning, 2226–2240. PMLR.

Xian, Z.; Gkanatsios, N.; Gervet, T.; Ke, T.-W.; and Fragkiadaki, K. 2023. ChainedDiffuser: Unifying Trajectory Diffusion and Keypose Prediction for Robotic Manipulation. In Conference on Robot Learning (CoRL), 2323–2339. PMLR.

Xiao, T.; Radosavovic, I.; Darrell, T.; and Malik, J. 2022. Masked visual pre-training for motor control. arXiv preprint arXiv:2203.06173.

J.; Lin, J.; Dang, K.; Bao, K.; Yang, K.; Yu, L.; Deng, L.;

Li, M.; Xue, M.; Li, M.; Zhang, P.; Wang, P.; Zhu, Q.; Men,

R.; Gao, R.; Liu, S.; Luo, S.; Li, T.; Tang, T.; Yin, W.; Ren,

X.; Wang, X.; Zhang, X.; Ren, X.; Fan, Y.; Su, Y.; Zhang,

Y.; Zhang, Y.; Wan, Y.; Liu, Y.; Wang, Z.; Cui, Z.; Zhang,

Z.; Zhou, Z.; and Qiu, Z. 2025. Qwen3 Technical Report. arXiv:2505.09388.

Ye, H.; Ding, Y.; Li, J.; and Ng, H. T. 2023. Robust Question Answering against Distribution Shifts with Test-Time Adaptation: An Empirical Study. arXiv:2302.04618.

Zeng, J.; Bu, Q.; Wang, B.; Xia, W.; Chen, L.; Dong, H.; Song, H.; Wang, D.; Hu, D.; Luo, P.; et al. 2024. Learning manipulation by predicting interaction. arXiv preprint arXiv:2406.00439.

Zhang, J.; Wu, S.; Luo, X.; Wu, H.; Gao, L.; Shen, H. T.; and Song, J. 2025. InSpire: Vision-Language-Action Models with Intrinsic Spatial Reasoning. arXiv:2505.13888.

Zhao, W.; Queralta, J. P.; and Westerlund, T. 2020. Sim-toreal transfer in deep reinforcement learning for robotics: a survey. In 2020 IEEE symposium series on computational intelligence (SSCI), 737–744. IEEE.

Zitkovich, B.; Yu, T.; Xu, S.; Xu, P.; Xiao, T.; Xia, F.; Wu, J.; Wohlhart, P.; Welker, S.; Wahid, A.; et al. 2023. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, 2165– 2183. PMLR.

## Appendix

This appendix provides comprehensive supplementary materials for our WorldAgen framework. We present detailed implementation specifications including benchmark descriptions, model architecture components, and trajectory processing methods for variable-length chunks. Additionally, we report extensive experimental results examining the effects of different chunk configurations on baseline performance, and comprehensive TTT analyses comparing LoRA (Hu et al. 2022) versus full fine-tuning approaches across various parameter settings and data sizes.

## Implementation Details

## Benchmark

LIBERO Benchmark (Liu et al. 2023a) is a comprehensive benchmark for lifelong learning in robot manipulation that consists of four distinct task suites designed to evaluate different aspects of knowledge transfer. The benchmark includes LIBERO-Spatial (10 tasks focusing on spatial relationship transfer), LIBERO-Object (10 tasks emphasizing object-centric knowledge transfer), LIBERO-Goal (10 tasks targeting procedural knowledge generalization), and LIBERO-100 (100 tasks with highly entangled knowledge requirements). Each task suite is accompanied by high-quality human teleoperation demonstrations to support sample-efficient learning.

We use LIBERO-100, which is the most challenging and comprehensive subset, containing 100 diverse manipulation tasks that require the transfer of mixed declarative and procedural knowledge. This suite is further divided into LIBERO-90, consisting of 90 short-horizon tasks used for pretraining policies, and LIBERO-10 (also referred to as LIBERO-Long), which contains 10 long-horizon tasks specifically selected for evaluating downstream lifelong learning performance.

CALVIN Benchmark (Mees et al. 2022) is an opensource simulated benchmark designed for learning longhorizon language-conditioned robot manipulation tasks. The dataset contains 34 manipulation tasks that are more complex than existing vision-and-language datasets in terms of sequence length, action space, and language complexity. The environment features a Franka Emika Panda robot with a parallel-jaw gripper operating in a desktop workspace containing various interactive objects, including a sliding door, drawer, colored blocks, LED, and light bulb that can be manipulated according to unconstrained natural language instructions.

CALVIN provides four different environments (A, B, C, D) that vary in desk colors and object configurations, enabling evaluation across different visual contexts and supporting flexible specification of sensor suites. The benchmark’s evaluation protocol requires agents to solve sequences of up to five consecutive tasks by understanding and executing a series of language instructions, such as open the drawer, pick up the blue block, rotate the block, push the block into the drawer, open the sliding door. This sequential task completion paradigm makes CALVIN particularly challenging for assessing the generalization capabilities and long-horizon reasoning of language-conditioned manipulation policies.

## Model Architecture

Leveraging the flexible architecture of Transformers, we achieve unified processing of image, language, action, and robot state modalities, supporting variable-length image chunks and action chunks for both input and output.

Encoder In the encoder component, we employ a MAEpretrained ViT-B (He et al. 2022) as the visual encoder, utilizing dual-view RGB inputs from static and wrist cameras. Similar to Seer (Tian et al. 2024), we incorporate a Perceiver Resampler (Jaegle et al. 2021) to reduce the number of image tokens, thereby decreasing computational load and improving efficiency. The Perceiver Resampler is an attentionbased feature compression module that uses a set of learnable query tokens to extract the most important information from a large number of image features. Through this approach, it compresses the originally large number of image tokens into a fixed number of compact representations, preserving essential visual information while significantly reducing the computational complexity of subsequent network layers.

For language processing, we utilize the text encoder from CLIP ViT-B/32. For robot state and action, we employ MLPs to project them into the same embedding space. Both robot state and action are represented as 7-dimensional vectors, where the first 6 dimensions encode the arm state representing the end effector’s position and orientation, and the last dimension represents the gripper state indicating the open/close status. We use separate MLP layers to process arm state and gripper state independently, then concatenate the results.

We initialize predicted tokens as zero-filled tensors with the same dimensions as the target actions and images to be predicted, and insert them after the input data at each time step.

Decoder In the decoder, we index the predicted tokens based on the input and slice out the image and action tokens. For image tokens, we employ the Vision Transformer encoder architecture (He et al. 2022) for decoding, followed by a linear layer to predict pixel patches. For the action decoder, similarly, we use an MLP to reduce the action-corresponding vectors to 7 dimensions. For the arm state and gripper state components, we employ different linear layers for decoding. For the gripper state, we apply binary thresholding at 0.5 to represent the gripper’s open/close status (0 for closed, 1 for open).

Backbone Qwen3 (Yang et al. 2025) represents the latest advancement in the Qwen large language model family, comprising both dense and Mixture-of-Experts (MoE) architectures with parameter scales ranging from 0.6 to 235 billion. Utilizing Qwen3 as a backbone architecture offers significant advantages for large language model applications, primarily through its innovative parameter efficiency where Qwen3 dense base models achieve performance comparable to much larger Qwen2.5 models while using fewer parameters, with Qwen3-4B matching the performance of Qwen2.5-72B-Instruct. The MoE variants provide exceptional computational efficiency, as Qwen3-MoE base models deliver similar performance to Qwen2.5 dense base models while utilizing only 10% of the active parameters, resulting in significant savings in both training and inference costs.

4.0  
Avg. Len. on CALVIN Benchmark 3.8 3.89 X  
3.82  
3.6  
X  
3.4 3.50  
3.45  
3.2  
<sub>o</sub><sup>n</sup> 3.0  
2.8  
2.6  
2.66  
2.4  
2 3 4 5 6  
Number of Chunks  
Figure 4: This figure shows the performance of the WorldAgen model on CALVIN as the number of chunks changes.

## Baseline Results

In this section, we present more results of our baseline on CALVIN benchmark, including pretraining, different length and number of chunks and different backbone.

## Pretraining

Given the powerful modeling capability and flexibility of Qwen3 (Yang et al. 2025), we adopt it as the network backbone for WorldAgen. Following the configuration established in (Tian et al. 2024), we configure the transformer architecture with 12 transformer heads and 24 transformer layers, resulting in a total network size of 370M parameters with 120M trainable parameters. Model pretraining is performed on a 4×H100 GPU server, taking approximately 60 hours for CALVIN and 5 hours for LIBERO.

## Image and Action Chunk Lengths

WorldAgen offers flexible input-output interfaces, accommodating image and action chunks of variable lengths. We evaluate the effect of different image and action chunk lengths on model performance. The results are shown in Table 7.

Therefore, on CALVIN benchmark, we employ an image chunk length of 1, action chunk length of 5, and trajectory length of 16. For LIBERO, we use an image chunk length of 1, action chunk length of 3, and trajectory length of 7.

We first fix the image chunk length to 1, as adjacent image frames contain redundant information and predicting more images increases computational overhead, thereby reducing model efficiency. By varying the action chunk length, we observe that longer action sequences lead to degraded performance. We attribute this to error accumulation in action predictions, where mistakes in early actions compound over longer sequences. However, excessively short action chunk lengths also result in performance degradation, as the network learns less information from the dataset.

Algorithm 1 Trajectory splitting with variable-length ac  
tion, observation, and robot state units.   
Require: Rollout $\tau = \{ ( o _ { t } , s _ { t } , a _ { t } ) \} _ { t = 0 } ^ { T - 1 }$ , observation chunk   
length $n ,$ action chunk length m   
1: $u  \emptyset$   
2: for $i = 0$ to $I - 1$ do   
3: $\mathbf { i f } \ i = 0$ then   
4: ${ \mathcal { T } } ^ { \mathrm { o b s } } \gets \{ 0 , 1 , \ldots , n - 1 \}$   
5: $O _ { 0 \to n - 1 } ^ { n }  \{ o _ { t } \mid t \in \mathbb { Z } ^ { \mathrm { { o b s } } } \}$   
6: $S _ { 0 \to n - 1 } ^ { n }  \{ s _ { t } \mid t \in \mathcal { T } ^ { \mathrm { o b s } } \}$   
7: $A _ { 0 \to n - 1 }  \{ a _ { t } \mid t \in \mathcal { T } ^ { \mathrm { o b s } } \}$   
8: $O _ { i } \left. O _ { 0 \right. n - 1 } ^ { n }$   
9: $S _ { i } \gets S _ { 0 \to n - 1 } ^ { n }$   
10: $A _ { i } \left. A _ { 0 \right. n - 1 }$   
11: $A _ { i } ^ { p }  \mathbf { 0 } ^ { n }$ // action placeholder (length n)   
12: $O _ { i } ^ { \bar { p } }  \mathbf { 0 } ^ { n } \qquad / /$ observation placeholder (length n)   
13: else   
14: $a  n + ( i - 1 ) m$   
15: $b \gets n + i m - 1$   
16: ${ \mathcal { T } } ^ { \mathrm { o b s } } \gets \mathrm { U N I F O R M S U B S A M P L E } ( a , b , . n )$   
17: $O _ { n + ( i - 1 ) m \right. n + i m - 1 } ^ { n } \left. \{ o _ { t } \mid t \in \mathbb { Z } ^ { \mathrm { o b s } } \}$   
18: $S _ { n + ( i - 1 ) m \right. n + i m - 1 } ^ { n } \left. \{ s _ { t } \mid t \in \mathbb { Z } ^ { \mathrm { o b s } } \}$   
19: $A _ { n + ( i - 1 ) m \right. n + i m - 1 } \left. \{ a _ { t } \mid n + ( i - 1 ) m \leq t \leq$   
$n + i m - 1 \}$   
20: $O _ { i } \left. O _ { n + ( i - 1 ) m \right. n + i m - 1 } ^ { n }$   
21: $S _ { i } \left. S _ { n + ( i - 1 ) m \right. n + i m - 1 } ^ { n }$   
22: $A _ { i } \left. A _ { n + ( i - 1 ) m \right. n + i m - 1 }$   
23: $A _ { i } ^ { p } \gets \mathbf { 0 } ^ { m }$ // action placeholder (length m)   
24: $O _ { i } ^ { p }  \mathbf { 0 } ^ { n }$ // observation placeholder (length n)   
25: end if   
26: $U _ { i } \gets \left( G , O _ { i } , S _ { i } , A _ { i } , A _ { i } ^ { p } , O _ { i } ^ { p } \right)$   
27: $\mathcal { U }  \mathcal { U } \cup \{ U _ { i } \}$   
28: end for   
29: return U

On LIBERO, we obtain consistent results, with performance deteriorating as action chunk length increases. Based on these findings, we adopt action chunk lengths of 5 for CALVIN and 3 for LIBERO in our main experiments.

## Number of Chunks

To investigate the effect of the number of chunks, we fix the image chunk length to 1 and the action chunk length to 3, conducting comprehensive experiments with the number of chunks varying from 2 to 6. As illustrated in Figure 4, we observe the same pattern as in our experiments with image chunk lengths and action chunk lengths.

A larger number of chunks require the model to predict more action steps, which leads to accumulation ofprediction errors. And the decreasing rate becomes larger when the number of chunks increases. However, an excessively small number of chunks limits the model’s ability to learn the relationship between world modeling and actions during training.

![](images/25aeb42856bc7ed9d2672eebf43e564161d1d4ee6735f8f8fa83e01f6f643f9e.jpg)  
Figure 5: In the comparison between the trajectories before and after TTT, the red regions indicate failures within the trajectory, while the green regions represent successful portions.

<table><tr><td>Dataset</td><td>I. Chunk</td><td>A. Chunk</td><td>Avg. Success ↑</td></tr><tr><td rowspan="7">CALVIN</td><td>1</td><td>3</td><td>3.82</td></tr><tr><td>1</td><td>5</td><td>3.87</td></tr><tr><td>1</td><td>7</td><td>3.43</td></tr><tr><td>1</td><td>9</td><td>3.16</td></tr><tr><td>3</td><td>5</td><td>3.30</td></tr><tr><td>5</td><td>5</td><td>1.78</td></tr><tr><td>1</td><td>3</td><td>78.0%</td></tr><tr><td rowspan="2">LIBERO</td><td>1</td><td>5</td><td>74.5%</td></tr><tr><td>1</td><td>7</td><td>47.0%</td></tr></table>

Table 7: Ablation study on image and action chunk lengths across CALVIN and LIBERO benchmarks. I. Chunk and A. Chunk represent image chunk length and action chunk length, respectively

According to our results, setting the number of chunks to 3 achieves the best performance.

We apply this finding to the configuration with image chunk length and action chunk length equal to 1 and 5, thereby achieving state-of-the-art results. This demonstrates

Both the scalability and robustness of our baseline approach, validating that the optimal chunk configuration principles discovered through systematic experimentation can be effectively transferred to different parameter settings to achieve superior performance.

## Test-Time Training Ablation Study TTT Details

For CALVIN, since the test scenarios are relatively uniform within each environment, we do not need to perform TTT for every individual sample. Instead, we conduct adaptation by sampling only on the first sample’s scenario. We select 34 text instructions that have appeared in the training set as guidance and perform random exploration for 60 frames. We then uniformly sample 6 times within these 60 frames for each task, which leads to 34 × 6 = 204 trajectories in total. Each trajectories have the same length as the pretraining dataset. Following (Kojima et al. 2025), we use single-epoch single-step optimization with LoRA rank set to 128, employing the AdamW optimizer with a learning rate of 0.005 and weight decay of 0.01.

For LIBERO, the evaluation involves 10 different test scenarios in the LIBERO-10 dataset, necessitating a different TTT strategy. We perform scene-level TTT sampling in each individual test scenario to adapt to the specific environmental conditions. Similar to CALVIN, we uniformly sample for 6 times with 6 different random seed for 60 frames exploration, which generates 6 × 6 samples per scenario. Also we set LoRA rank to 64 and employ single-epoch single-step optimization using the AdamW optimizer with a learning rate of 0.005 and weight decay of 0.01.

## LoRA vs. Full Fine-tuning

We conducted a comparative study between LoRA and Full Fine-Tuning (FFT) on the CALVIN dataset. In our experiments, we set the LoRA rank to 128. Both LoRA finetuning and FFT used identical hyperparameters: learning rate $l r ~ = ~ 0 . 0 0 0 5$ , Adam optimizer, and weight decay of 0.01.

<table><tr><td>Method</td><td>Avg. Len.</td></tr><tr><td>WorldAgen</td><td>3.87</td></tr><tr><td>WorldAgen + LoRA</td><td>3.93</td></tr><tr><td> $\mathrm { \ w o r l d A g e n + F F T }$ </td><td>3.85</td></tr></table>

Table 8: Comparison of test-time training with LoRA and FFT on CALVIN dataset. Avg. Len. represents the average sequence length.

As shown in Table 8, our results reveal that FFT does not improve model performance and actually leads to a decline in performance. We attribute this phenomenon to the fact that

FFT causes the network to overfit to the test scenarios, whereas LoRA enables the model to acquire scenariospecific features while preserving the original scene perception capabilities. Therefore, LoRA demonstrates superior performance by striking a better balance between adaptation to new scenarios and retention of pretrained knowledge.

## Scalability and Robustness

To investigate the scalability and robustness of our method, we select three parameters crucial to TTT: number of chunks (N), image chunk length (n), and action chunk length (m). We conducted TTT experiments across different combinations of N, n, and m values.

We choose three representative settings with different configurations to demonstrate the generalizability of our approach across various parameter combinations.

<table><tr><td>N</td><td>n</td><td>m</td><td>LoRA Rank</td><td>Data Size</td><td>Before TTT After TTT</td></tr><tr><td>3</td><td>1</td><td>3</td><td>64</td><td>170</td><td>3.89 3.90</td></tr><tr><td>5</td><td>1</td><td>3</td><td>256</td><td>204</td><td>3.50 3.67</td></tr><tr><td>3</td><td>1</td><td>5</td><td>128</td><td>204</td><td>3.87 3.93</td></tr></table>

Table 9: TTT experiments under different number of chunks (N), image chunk length (n), and action chunk length (m) configurations.

As demonstrated in the Table 9, TTT consistently improves performance across different settings using only around 200 samples. This validates

The scalability and robustness ofour TTT approach and it can effectively adapt to various architectural configurations while maintaining consistent improvement patterns.

## Test-Time Training Data Size

Furthermore, during the grid search process for the experimental configuration with $n = 1 , m = 3 , N = 5 .$ , as illustrated in the Figure 6, we discovered that TTT exhibits linear performance improvement with small amounts of data. However, when the data volume becomes excessive, we observe a phenomenon where TTT performance degrades.

![](images/21ccd9dcc74a4abb281c71a672193d6df7dac8731953d4eace44a5e62111ee52.jpg)  
Figure 6: This figure shows that as the amount of data increases, the TTT effect gradually increases.

These experiments were conducted with LoRA rank = 256 and learning rate = 0.0005.

We attribute this phenomenon to the fact that excessive training samples cause LoRA to overfit on the world modeling component, making the model more biased towards predicting world observations rather than executing correct actions. This finding highlights

The importance of carefully balancing the amount of test-time training data to achieve optimal performance without compromising the model’s action execution capabilities.

## Comparison Between Backbones

<table><tr><td>Backbone</td><td>CALVIN</td><td>LIBERO-10</td></tr><tr><td>GPT2</td><td>3.32</td><td>75.0</td></tr><tr><td>Qwen3</td><td>3.87</td><td>75.5</td></tr></table>

Table 10: Performance comparison on different backbones.

We incorporate the GPT-2 (Radford et al. 2019) backbone to examine the performance gap across different language backbones. Compared with GPT-2, Qwen3 demonstrates stronger representation capacity and better alignment with multimodal instructions, leading to consistently higher performance, as shown in Table 10.

## LIBERO Subsets

<table><tr><td>Spatial</td><td>Object</td><td>Goal</td><td>LIBERO-10</td></tr><tr><td>93.0</td><td>79.5</td><td>91.0</td><td>79.0</td></tr></table>

Table 11: Performance across different subsets in LIBERO.

We further include additional subsets of the LIBERO benchmark to comprehensively evaluate the performance of WorldAgen. The results demonstrate that our method maintains stable and consistent performance across various subsets, indicating strong generalization ability and robustness to task diversity, as shown in Table 11.

## Comparison Visualization

We visualize the comparison before and after applying TTT. As shown in the figure 5, the red regions represent the failed portions of the trajectory, while the green regions denote the successful units. The selected task sequence is rotate blue block right, move slider right, lift red block slider, and place in slider. It can be observed that before applying TTT, the robot only completed the rotate blue block right task and then entered a halted state. After incorporating TTT, however, the model demonstrates a significantly enhanced ability to model the environment, thereby successfully accomplishing a series of tasks.

Table 12: Summary of important notations used in the WorldAgen framework.
<table><tr><td>Notation</td><td>Symbol</td><td>Description</td></tr><tr><td>Action</td><td> $a _ { t }$ </td><td>Robot action at time step t in the raw rollout.</td></tr><tr><td>Observation</td><td> $o _ { t }$ </td><td>Environmental observation at time step t in the raw rollout.</td></tr><tr><td>Robot State</td><td> $s _ { t }$ </td><td>Robot proprioceptive state at time step t in the raw rollout.</td></tr><tr><td>Rollout Trajectory</td><td> $\tau$ </td><td>A trajectory with T time steps:  ${ \boldsymbol { \tau } } = \{ ( o _ { t } , s _ { t } , a _ { t } ) \} _ { t = 0 } ^ { T - 1 }$ </td></tr><tr><td>Task Instruction</td><td> $G _ { i }$ </td><td>Language instruction specifying the task in unit ¿.</td></tr><tr><td>Action Chunk Length</td><td> $m$ </td><td>Number of actions in each action chunk  ${ \overline { { A _ { i } } } } .$ </td></tr><tr><td>Observation / Robot State Chunk Length</td><td> $n$ </td><td>Number of observations and robot proprioceptive states in each chunk  $O _ { i }$  and  $S _ { i } .$ </td></tr><tr><td>Trajectory Unit</td><td> $\overline { { U _ { i } } }$ </td><td>The i-th trajectory unit:  $U _ { i } = ( G _ { i } , O _ { i } , S _ { i } , A _ { i } , A _ { p } , O _ { p } )$ </td></tr><tr><td rowspan="2">Action Chunk</td><td> $A _ { i }$ </td><td>Action chunk in unit i:</td></tr><tr><td></td><td></td></tr><tr><td rowspan="4">Observation Chunk</td><td rowspan="4"> $O _ { i }$ </td><td> $\begin{array} { l } { A _ { i } = A _ { n + ( i - 1 ) m  n + i m - 1 } } \\ { \quad = [ a _ { n + ( i - 1 ) m } , a _ { n + ( i - 1 ) m + 1 } , \ldots , a _ { n + i m - 1 } ] . } \end{array}$ </td></tr><tr><td>Observation chunk in unit i:</td></tr><tr><td> $O _ { i } = O _ { n + ( i - 1 ) m  n + i m - 1 } ^ { n }$ </td></tr><tr><td> $= \left[ o _ { n + ( i - 1 ) m } , o _ { n + ( i - 1 ) m + \lfloor \frac { 1 } { n } m \rfloor } , \ldots , o _ { n + ( i - 1 ) m + \lfloor \frac { n - 1 } { n } m \rfloor } \right] .$ </td></tr><tr><td rowspan="4">Robot State Chunk</td><td rowspan="4"> $S _ { i }$ </td><td>Robot state chunk in unit i:</td></tr><tr><td></td></tr><tr><td> $\begin{array} { r l } & { S _ { i } = S _ { n + ( i - 1 ) m  n + i m - 1 } ^ { n } } \\ & { \quad = [ s _ { n + ( i - 1 ) m } , s _ { n + ( i - 1 ) m + \lfloor \frac { 1 } { n } m \rfloor } , \dots , s _ { n + ( i - 1 ) m + \lfloor \frac { n - 1 } { n } m \rfloor } ] . } \end{array}$ </td></tr><tr><td></td></tr><tr><td></td><td> $\overline { { A _ { p } } }$ </td><td>Placeholder action tokens.</td></tr><tr><td>Observation Placeholder</td><td> $O _ { p }$ </td><td>Placeholder observation tokens.</td></tr><tr><td>Predicted Action Chunk</td><td> ${ \hat { A } } _ { i }$ </td><td>Predicted action chunk:  $\hat { A } _ { i } = \left[ \hat { a } _ { n + ( i - 1 ) m } , \hat { a } _ { n + ( i - 1 ) m + 1 } , . . . , \hat { a } _ { n + i m - 1 } \right] .$ </td></tr><tr><td rowspan="5">Predicted Observation Chunk</td><td rowspan="5"> $\hat { O } _ { i }$ </td><td></td></tr><tr><td>Predicted next observation chunk:</td></tr><tr><td></td></tr><tr><td> $\begin{array} { r l } & { \hat { O } _ { i } = \hat { O } _ { n + i m  n + ( i + 1 ) m - 1 } ^ { n } } \\ & { \quad \quad = [ \hat { o } _ { n + i m } , \hat { o } _ { n + i m + \lfloor \frac { 1 } { n } m \rfloor } , \dots , \hat { o } _ { n + i m + \lfloor \frac { n - 1 } { n } m \rfloor } ] . } \end{array}$ </td></tr><tr><td></td></tr></table>