# Reusing Multimodal Agents: Efficient Evaluation of Sound Designs in Games

Thai Nguyen Van<sup>1,∗</sup>, Ibrahim Khan<sup>2</sup>, Mury Fajar Dewantoro<sup>3</sup>,

Pratch Suntichaikul<sup>3</sup>, Hai Van Pham<sup>4</sup>, Ruck Thawonmas<sup>2</sup>

<sup>1</sup>Graduate School of Information Science and Engineering, Ritsumeikan University, Japan

<sup>2</sup>College of Information Science and Engineering, Ritsumeikan University, Japan

<sup>3</sup>Independent Researcher

<sup>4</sup>School of Information and Communications Technology, Hanoi University of Science and Technology, Vietnam

Abstract—This paper introduces a novel framework for efficiently evaluating in-game sound designs using multimodal learning and reinforcement learning. Sound design significantly influences gameplay and accessibility; however, objective evaluation methods, defined here as automated assessments based on agent performance rather than human ratings, often require retraining agents from scratch for each sound design. Our approach first pre-trains a multimodal encoder that encodes visual and auditory information into a shared latent space. A reinforcement learning agent is then trained using this latent representation. We fine-tune the multimodal encoder for each sound design while keeping the trained policy fixed, allowing us to isolate the effects of sound changes on agent behavior. To evaluate the proposed method, we conducted experiments on games from multiple genres, comparing the performance of blind agents (agents using only sound input) under various sound designs. Experimental results suggest that this approach can detect differences in agent performance caused by degraded or altered sound design. Moreover, we significantly reduce the computational cost compared to retraining the agent from scratch. Experimental results suggest that this approach provides a practical and scalable solution for evaluating in-game sound designs. The source code, data and supplementary document are available <sup>1</sup>.

Index Terms—Sound design evaluation, Reinforcement Learning, Multimodal Learning, Representation Learning

## I. INTRODUCTION

Sound plays a critical role in games, not only enriching the overall player experience but also conveying vital information for perception, navigation, and interaction [1], [2]. Carefully crafted audio cues draw players into the game world, significantly elevating their enjoyment through heightened immersion [3]. While the broader scope of game accessibility includes other sensory modalities, such as haptic technologies [4] to support deaf players, for those with visual impairments, sound is often the primary or sole channel through which gameplay is experienced [5]–[7]. As a result, sound design becomes not merely an aesthetic choice but a fundamental component of accessible gameplay. While its aesthetic and subjective qualities remain essential to the player experience, in this work, we focus on the complementary challenge of objectively assessing how informative a sound design is for gameplay.

In recent years, game AI research has begun to explore agents that operate based on audio input alone, leading to new challenges in evaluating the effectiveness of different sound designs, not just from a human-centered perspective, but also through the lens of AI perception and learning [8]–[10]. To address these challenges, prior studies have employed in-game audio for representation learning [11], [12], reinforcement learning [8], [13], and sound evaluation [9], [14]. However, they are hindered by a major limitation: each new sound design typically requires training a deep reinforcement learning (RL) agent from scratch. This retraining process is computationally intensive and time-consuming, making it impractical for iterative sound design testing and comparison. Moreover, retraining introduces stochasticity into the evaluation, complicating efforts to assess the impact of specific sound modifications fairly.

To address this gap, we propose a novel two-phase evalua  
tion framework that decouples sound feature extraction from   
policy training. In the first phase, an encoder is trained to   
extract informative auditory features. In the second phase,   
only the sound-specific portion of this encoder is fine-tuned   
for each new design, while the policy network remains fixed.   
This separation allows new sound designs to be evaluated more   
efficiently, without retraining the entire AI agent from scratch. This study’s contributions are as follows:

• Conceptual shift from retraining to adaptation: We propose a novel evaluation paradigm that treats sound design assessment as a representation learning problem rather than a standard reinforcement learning task. By decoupling sound design sensitivity from the policy network, our framework eliminates the need for expensive “from scratch” retraining for each design iteration.

• Validation of representation sensitivity: We demonstrate that fine-tuning a multimodal encoder, rather than the entire agent, is sufficient to capture the nuances of sound design quality. We show that this approach preserves agent performance under equivalent designs while remaining highly sensitive to design degradations.

• Efficiency-performance correlation: We provide an empirical analysis of the relationship between latent space efficiency and agent behavior. Specifically, we show how the information-theoretic properties of the representation space correlate directly with the agent’s ability to utilize sound cues for decision-making.

## II. RELATED WORK

## A. Sound design evaluation

Several studies have investigated the evaluation of sound design in games [9], [14], [15]. One notable example is the DareFightingICE Sound Design Competition [15], which introduced both a sound design track and an AI evaluation track. In this competition, a deep RL blind AI agent [9], which plays games using only sound, was employed as an objective evaluator, measuring how effectively different sound designs supported gameplay performance. In addition, Khan et al. [14] proposed an evaluation framework using both blind agents and human players for sound design evaluation in a first-person shooter game. While these studies demonstrate the feasibility of agent-based sound design evaluation, our work investigates this problem from a different perspective by leveraging multimodal representation learning and applying it to alternative game environments.

## B. Representation learning in reinforcement learning

Representation learning has been widely explored in RL to address the challenges of high-dimensional sensory input, such as images or audio [16], [17]. Autoencoder-based approaches, including variational autoencoders, have shown that compact latent spaces can improve the sample efficiency and transferability of RL agents. More recent studies have introduced contrastive objectives to further enhance the robustness of learned representations, for example, through predictive coding [18] or cross-modal alignment [19]–[21]. Nevertheless, state representation has been explored in offline visual RL to isolate taskrelevant factors and reduce the impact of spurious correlations [22], [23]. These studies demonstrate that the combination of autoencoding and contrastive methods provides effective representations for downstream RL tasks.

These representation learning techniques are applied distinctly across RL paradigms. In model-based approaches, latent representations are primarily learned to capture environment dynamics and support efficient planning, as demonstrated in methods like World Models [17], PlaNet [16], and the Dreamer series [24]–[26]. Conversely, in model-free approaches, representations are typically shaped by auxiliary objectives to improve the feature quality for a policy or value function. Our work builds on these principles, employing a contrastive learning framework to develop representations from sound and vision for the purpose of evaluating game sound design.

## C. Multimodal learning

Multimodal learning seeks to combine information from different sensory inputs into a unified representation that captures cross-modal relationships [27]. More recent approaches, including contrastive methods such as CLIP [28], have shown that aligning modalities in a shared latent space can yield robust and transferable representations. In RL, multimodal representation methods such as CoRAL [21] and M2CURL [20] have been explored for integrating vision with proprioceptive or tactile signals. Not only RL, multimodal learning has also been explored in other tasks such as classification [29] and self-supervised learning [30].

Fusion strategies in multimodal learning vary from early fusion of raw features to late fusion at the decision level [31]–[33], with shared latent representations emerging as a particularly effective approach for exploiting cross-modal relationships. Compared to generative-based methods, fusion methods can be more resilient to imperfections in any single data modality [34]. If one modality is noisy or incomplete, correlated information from another modality (e.g., the visual channel) can help recover suppressed features, resulting in a more robust and reliable representation [35]. Despite this progress, relatively little attention has been devoted to game audio, even though sound is a critical modality for accessibility and immersion in games. Motivated by these advances, our work applies multimodal representation learning to align vision and sound in games, enabling systematic evaluation of sound designs through their effect on agent performance.

## III. METHODOLOGY

In this work, the task of sound design evaluation is formulated as measuring the performance of an AI agent when it relies on audio cues to act in a game. An effective sound design should provide informative cues that enable the agent to achieve higher performance, while degraded or ambiguous designs are expected to reduce performance. We therefore use agent performance as an objective and reproducible metric for comparing different sound designs.

For each game, we adopted a two-step training procedure consisting of pretraining and fine-tuning. In the pretraining phase, we first trained a multimodal encoder using a baseline sound design together with visual input to learn a joint latent representation. A policy network was then trained to take this latent representation as input and output actions for the agent. Algorithm 1 and Algorithm 2 show the details of pretraining and training policy network, respectively.

The multimodal encoder learns a joint latent representation from both visual and audio information using the baseline sound design. At each game step, three types of observations can be derived from the environment: (i) the image observation - the raw game frame; (ii) the audio observation - a raw representation of game audio; and (iii) the audio-image - a fused representation that combines the audio and image observations into a single multimodal feature vector. The fusion of audio-image is the input to the policy network during policy training, while in the blind evaluation, only the audio observation is provided.

In the fine-tuning stage, only the audio-specific components were updated, while the remaining parts of the multimodal encoder and the policy network were kept frozen. This design allowed us to adapt the representation to new sound design without retraining the policy, thus saving computational cost and ensuring consistency across sound designs. To evaluate performance, we performed zero-shot policy transfer, where the pretrained policy was directly reused with the output of the finetuned encoder. Finally, we measured performance in a blind setting. In the blind setting, the visual modality, which was present during training, is removed exclusively during testing to assess how effectively the agent can perform tasks using only auditory cues. Algorithm 3 shows the details of finetuning process. Although it uses the same objective function as Algorithm 1, it updates only the audio-related components, keeping all other network components frozen. Figure 1 shows our method pipeline and evaluation method and the overall architecture of the agent is shown in Figure 2.

![](images/73b86dfe3798552966f10d1181adfb7c844c07f23140940922d7e9b8a0ed9088.jpg)  
Fig. 1. Our method pipeline and agent evaluation setting. Only audio is used in the blind setting, while both images and audio are used for the standard setting.

![](images/3b4dbef92209b418174baf877f7e4fe6dde5678c3a1af9a7940d6b073c44fd9f.jpg)

Fig. 2. Blind Agent Architecture. Audio input is processed by a multimodal encoder to extract latent features, which the RL policy uses to output an action.  
Algorithm 1 Stage 1: Pretrain Multimodal Encoders   
Require: Dataset $\mathcal { D } \ = \ \{ ( x ^ { \mathrm { i m g } } , x ^ { \mathrm { a u d } } ) \}$ from the baseline   
sound design; batch size $B ;$ temperature $\tau$   
1: Initialize encoders $E _ { \mathrm { i m g } } , E _ { \mathrm { a u d } } , E _ { \mathrm { f u s e } }$ and common head C   
2: while not converged do   
3: Sample $\{ ( x _ { b } ^ { \mathrm { { i m g } } } , x _ { b } ^ { \mathrm { { a u d } } } ) \} _ { b = 1 } ^ { B } \sim \mathcal { D }$   
4: for $b \gets 1$ to $B$ do   
5: $h _ { b } ^ { \mathrm { i m g } }  E _ { \mathrm { i m g } } ( x _ { b } ^ { \mathrm { i m g } } ) ; \quad h _ { b } ^ { \mathrm { a u d } }  E _ { \mathrm { a u d } } ( x _ { b } ^ { \mathrm { a u d } } )$   
6: $z _ { b } ^ { \mathrm { i m g } }  C ( h _ { b } ^ { \mathrm { i m g } } ) ; \ : \ : \ : z _ { b } ^ { \mathrm { a u d } }  C ( h _ { b } ^ { \mathrm { a u d } } )$   
7: $z _ { b } ^ { \mathrm { f u s e } }  C ( E _ { \mathrm { f u s e } } ( x _ { b } ^ { \mathrm { i m g } } \oplus x _ { b } ^ { \mathrm { a u d } } ) )$   
8: end for   
9: Compute GMC loss   
$\mathcal { L } _ { \mathrm { G M C } } ( \{ { \dot { z } } _ { b } ^ { \mathrm { i m g } } \} _ { b = 1 } ^ { B } , \{ z _ { b } ^ { \mathrm { a u d } } \} _ { b = 1 } ^ { B } , \{ z _ { b } ^ { \mathrm { f u s e } } \} _ { b = 1 } ^ { B } ; \tau )$   
10: Update the parameters of $\{ E _ { \mathrm { i m g } } , E _ { \mathrm { a u d } } , E _ { \mathrm { f u s e } } , C \}$ by   
descending $\boldsymbol { \nabla } \mathcal { L } _ { \mathrm { G M C } }$   
11: end while   
12: return $E _ { \mathrm { i m g } } , E _ { \mathrm { a u d } } , E _ { \mathrm { f u s e } } , C$

## A. Multimodal Encoder

We used the geometric multimodal contrast (GMC) multimodal encoder introduced in [10] to learn latent representations of the visual and audio modalities because it is lightweight and robust to missing modality. For completeness, we provide a brief summary of the model, adapted from the original work. An illustration of the multimodal encoder is shown in Figure 3. The GMC multimodal encoder comprises the following components:

Algorithm 2 Stage 2: Train Policy on Latent (use the Fusion   
encoder when both modalities are present)   
Require: Frozen $E _ { \mathrm { i m g } } , E _ { \mathrm { a u d } } , E _ { \mathrm { f u s e } } ,$ and $\overline { { C ; } }$ RL algorithm   
(e.g., PPO)   
1: Initialize policy $\pi _ { \phi }$ (and value net $V _ { \psi }$ if used)   
2: while RL not converged do   
3: Collect rollouts $\{ ( o _ { t } ^ { \mathrm { i m g } } , o _ { t } ^ { \mathrm { a u d } } , r _ { t } ) \} _ { t = 1 } ^ { T } ,$ where   
$\{ ( o _ { t } ^ { \mathrm { i m g } } , o _ { t } ^ { \mathrm { a u d } } \}$ are visual observation and audio observation   
at timestep t, respectively.   
4: for $t \gets 1$ to $T$ do   
5: $z _ { t } \gets C ( E _ { \mathrm { f u s e } } ( o _ { t } ^ { \mathrm { i m g } } \oplus o _ { t } ^ { \mathrm { a u d } } ) )$   
6: Sample action $a _ { t } \sim \pi _ { \phi } ( \cdot \mid z _ { t } )$   
7: $A  A \cup \{ a _ { t } \}$   
8: $Z \gets Z \cup \{ z _ { t } \}$   
9: $R \gets R \cup \{ r _ { t } \}$   
10: end for   
11: Update $\phi$ (and ψ) with the chosen RL objective   
function $O ( A , Z , R )$   
12: end while   
13: return trained policy $\pi _ { \phi }$ (and value net $V _ { \psi }$ if used)

Algorithm 3 Stage 3: Finetune Audio (and Fusion) for New   
Sound Design; Policy Reuse   
Require: New dataset $\mathcal { D } ^ { \prime } = \{ ( x ^ { \mathrm { i m g } } , x _ { \mathrm { n e w } } ^ { \mathrm { a u d } } ) \}$ ; frozen $E _ { \mathrm { i m g } } , C ,$   
and both the image features block and MLP block of   
$E _ { \mathrm { f u s e } } ;$ trainable $E _ { \mathrm { a u d } }$ and the audio features block of   
$E _ { \mathrm { f u s e } }$   
1: while not converged do   
2: Sample $\{ ( x _ { b } ^ { \mathrm { { i m g } } } , x _ { b } ^ { \mathrm { { a u d } } } ) \} _ { b = 1 } ^ { B } \sim \mathcal { D } ^ { \prime }$   
3: for $b \gets 1$ to B do   
4: $z _ { b } ^ { \mathrm { i m g } }  C ( E _ { \mathrm { i m g } } ( x _ { b } ^ { \mathrm { i m g } } ) )$   
5: $z _ { b } ^ { \mathrm { { \bar { a u d } } \mathrm { { - n e w } } } }  C ( \mathrm { { \bar { E } } _ { a u d } } ( x _ { b } ^ { \mathrm { { a u d } } } ) )$   
6: $z _ { b } ^ { \mathrm { \bar { f } u s e \mathrm { - } n e w } }  C ( E _ { \mathrm { f u s e } } ( x _ { b } ^ { \mathrm { \bar { l } m g } } \oplus x _ { b } ^ { \mathrm { a u d } } ) )$   
7: end for   
8: Compute $\mathrm { G M C }$ loss   
$\mathcal { L } _ { \mathrm { G M C } } \big ( \{ z _ { b } ^ { \mathrm { i m g } } \} _ { b = 1 } ^ { B } , \{ z _ { b } ^ { \mathrm { a u d \_ n e w } } \} _ { b = 1 } ^ { B } , \{ z _ { b } ^ { \mathrm { f u s e \_ n e w } } \} _ { b = 1 } ^ { B } ; \tau \big )$   
9: Update only $E _ { \mathrm { a u d } }$ and the audio features block of   
$E _ { \mathrm { f u s e } }$ by descending ∇L<sub>GMC</sub>   
10: end while   
11: return $E _ { \mathrm { i m g } } , E _ { \mathrm { a u d } } , E _ { \mathrm { f u s e } } , C$

![](images/d478a3eb009c9bf0fe7de29906670b5380b8635bb7ce0100e2ca992dbd3c4672.jpg)  
Fig. 3. Multimodal encoder architecture. Audio, Image, and fused modalities are processed by dedicated feature extractors and MLPs before converging into a shared Common MLP module.

• Modality-specific encoders that process unimodal inputs and fused representations of the visual and audio modalities, producing intermediate features. There are three encoders: image encoder, audio encoder, and fusion encoder.

• A shared projection head, which is a Multilayer Perceptron (MLP), mapping the intermediate features into the latent space.

• To train the multimodal encoder, a multimodal contrastive loss function $(  { \mathcal { L } } _ { \mathrm { G M C } } )$ that aligns the geometric structure of modality-specific representations is used. The definition of this loss function is given later.

1) Training Objective: Let $s i m ( u , v )$ be the cosine similarity between vectors u and $v ,$ and let $\tau \in \mathsf { \Gamma } ( 0 , \infty )$ be the temperature. The similarity between two representations $z _ { p } ^ { i }$ and $z _ { q } ^ { j }$ is defined as below:

$$
s _ { ( p , q ) } ( i , j ) = e x p ( s i m ( z _ { p } ^ { i } , z _ { q } ^ { j } ) / \tau ) ,\tag{1}
$$

where $z _ { p } ^ { i }$ and $z _ { q } ^ { j }$ correspond to the ith and jth samples from a batch $\boldsymbol { B } ^ { 2 } = \{ \boldsymbol { z } _ { 1 : M } ^ { i } \} _ { i = 1 } ^ { B }$ which is a batch of B complete representations. For a given modality m, $( z _ { m } ^ { i } , z _ { 1 : M } ^ { i } )$ and $\left( z _ { 1 : M } ^ { i } , z _ { m } ^ { i } \right)$ are positive pairs for $i = 1 , \ldots , B$ and the remaining pairs are negative. Therefore

$$
\Omega _ { ( p , q ) } ( i ) = \sum _ { i \neq j } s _ { p , p } ( i , j ) + \sum _ { i \neq j } s _ { p , q } ( i , j )\tag{2}
$$

is the sum of similarities among negative pairs corresponding to the positive pair $( z _ { p } ^ { i } , z _ { q } ^ { i } )$ , where a positive pair consists of modality representations from the same sample, and a negative pair consists of representations from different samples. The contrastive loss for the positive pairs $( z _ { m } ^ { i } , z _ { 1 : M } ^ { i } )$ and $\left( z _ { 1 : M } ^ { i } , z _ { m } ^ { i } \right)$ is defined as

$$
\begin{array} { r } { l _ { m } ( i ) = - \log \frac { s _ { m , 1 : M } ( i , i ) } { \Omega _ { m , 1 : M } ( i ) } - \log \frac { s _ { 1 : M , m } ( i , i ) } { \Omega _ { 1 : M , m } ( i ) } . } \end{array}\tag{3}
$$

Finally, the loss function for the batch $\boldsymbol { B }$ is defined as follows.

$$
\mathcal { L } _ { \mathrm { G M C } } ( \boldsymbol { \mathcal { B } } ) = \sum _ { m = 1 } ^ { M } \sum _ { i = 1 } ^ { B } l _ { m } ( i ) .\tag{4}
$$

## IV. EXPERIMENTS

We evaluated our framework in three games: PENDULUM, HYPERHOT, and VizDoom, which represent different gameplay genres. In these games, the reward function does not explicitly model perceptual sound quality beyond agent task performance. Rather, the quality of a sound design is defined by its informativeness; designs with rich cues facilitate better decision-making, while degraded designs convey limited or ambiguous information. For each game, we prepared four sound designs. Design 1 serves as the baseline and is used for multimodal pretraining. Design 2 is semantically equivalent to the baseline and is used in fine-tuning to confirm whether the agent performs consistently under equivalent conditions. Design 3 is deliberately degraded with less informative cues to evaluate the agent’s ability to detect quality loss. Finally, Design 4 introduces further degradation compared to Design 3. The details of their conditions for each game are described in the following subsections.

## A. PENDULUM

In the PENDULUM game, the task is to swing the pendulum upward and keep it balanced for as long as possible. We used a modified version of the OpenAI Gym environment [11] that provides audio feedback to the agent. Figure 4 shows an example of the pendulum environment with its corresponding sound designs.

1) Baseline Sound Design: For the baseline sound design (Design 1), we adopted the configuration introduced in [11]. In this setup, the tip of the pendulum continuously emits a base frequency $f _ { 0 } ,$ , which is received by a set of sound receivers $\{ \rho _ { 1 } , \ldots , \rho _ { S } \}$ . At each timestep, the perceived frequency $f _ { i } ^ { \prime }$ at receiver $\rho _ { i }$ is determined by the Doppler effect, which models the change in observed frequency as the relative distance between the sound source and receiver decreases or increases. The resulting frequency $f _ { i } ^ { \prime }$ is given by:

$$
f _ { i } ^ { \prime } = \left( { \frac { c + { \dot { \pmb { \rho } } } _ { i } \cdot ( { \bf e } - { \pmb { \rho } } _ { i } ) } { c - { \dot { \bf e } } \cdot ( { \pmb { \rho } } _ { i } - { \bf e } ) } } \right) f _ { 0 } ,
$$

where c is the speed of sound, the overdot represents the velocity of its vector, and e is the position of the sound emitter,

![](images/9fce8f75fec8e73487214277ff2e5b449f11e591a684ba173bb8e3d9fea52fe8.jpg)

![](images/27edb34d9d6d5ab4ff9975813e839bcbb8b038adc9e1892fa1a2a4fa5680479d.jpg)  
Fig. 4. PENDULUM sound designs.  
(b) Design 2

![](images/bf5ff6f3814097aae5da936bd25cca577b3171f684a31c465fb46750ccf6346c.jpg)  
(c) Design 3

![](images/8b2578b47f9bf5c9e33d2a8bf4ebbcab25a3cb78db1d069ed30479c8c0c9fe5c.jpg)  
(d) Design 4  
(a) Design 1: three listeners positioned at the bottom left and right, and middle top.  
(b) Design 2: same as (a) but the three listeners are moved to the top left and right, and bottom middle.  
(c) Design 3: same as (b) but regions where the sound cue is indistinguishable are shaded.  
(d) Design 4: same as (b) but regions where the sound cue is indistinguishable are shaded.

$\rho _ { i }$ and e being two-dimension vectors. The amplitude $a _ { i }$ heard by receiver $p _ { i }$ follows the inverse square law

$$
a _ { i } = \frac { K } { \| \mathbf { e } - \pmb { \rho } _ { i } \| ^ { 2 } } ,
$$

where $K$ is a scaling constant. In the PENDULUM game, the angle θ of the pendulum is defined as the angular displacement from the upright vertical position. More specifically

$\theta = 0$ : The pendulum is pointing straight up (the goal state).

• θ = 180 (or −180) degree: The pendulum is pointing straight down.

• Positive values: Represent counter-clockwise rotation.

• Negative values: Represent clockwise rotation.

2) Finetuning sound designs: Based on the baseline sound design, we created three variants. The first variant (Design 2) rotates the listeners’ locations while keeping all other settings unchanged. The second variant (Design 3) uses the same listener positions as Design 2 but restricts the sound to a specific angular range of the pendulum. Specifically, when the pendulum’s angle lies in (45, 135), the emitted sound’s frequency is fixed, as if the pendulum were at 90 degree. Design 4 introduces an additional constraint, when the angle lies in (-45, -135), the sound’s frequency is fixed as if the pendulum were at -90 degree.

## B. HYPERHOT

HYPERHOT game is a novel top-down shooter game inspired by the SPACE INVADERS Atari game and has been used in previous studies [11], [12]. The objective is to eliminate enemies positioned above while dodging incoming fire by moving horizontally. A sample screenshot of HYPERHOT is provided in Figure 5. The game employs a reward system where the agent gains points for successful actions, defined as follows:

$$
r = { \left\{ \begin{array} { l l } { 1 0 } & { { \mathrm { i f ~ a l l ~ e n e m i e s ~ a r e ~ k i l l e d , ~ } } i . e . , { \mathrm { ~ w i n } } } \\ { - 1 } & { { \mathrm { i f ~ p l a y e r ~ i s ~ k i l l e d ~ o r ~ t i m e ~ i s ~ u p , ~ } } i . e . , { \mathrm { ~ l o s e } } } \\ { 0 } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }
$$

1) Baseline sound design: For the baseline sound design (Design 1), we followed the configuration described in [11]. In this setup, environmental sound cues are generated by multiple emitters $e _ { i } { \mathrm { . } }$ , each producing a fixed frequency $f _ { 0 } ^ { ( i ) }$

• Enemy units on the left $( e _ { 0 } )$ and right $( e _ { 1 } )$ emit sound with frequencies $f _ { 0 } ^ { ( 0 ) }$ and $f _ { 0 } ^ { ( 1 ) }$ , respectively.

• Enemy projectiles $( e _ { 2 } )$ emit sound with frequency $f _ { 0 } ^ { ( 2 ) }$

• The agent’s projectiles (e<sub>3</sub>) emit sound with frequency $f _ { 0 } ^ { ( 3 ) }$

These sound cues are detected by a set of S receivers, $\{ \rho _ { 1 } , . . . , \rho _ { S } \}$ . At each frame, the contribution of every emitter within the audible region is calculated based on its distance to each receiver. The sound wave generated by emitter $e _ { i }$ and perceived at receiver $\rho _ { j }$ , denoted $a ^ { ( i ) }$ , is modeled as:

$$
a ^ { ( i ) } = a _ { 0 } ^ { ( i ) } \exp \left( - \delta \| \mathbf { e } _ { i } - \mathbf { p } _ { j } \| ^ { 2 } \right) ,
$$

where $\delta$ is a scaling constant, and $\mathbf { e } _ { i }$ and $\mathbf { p } _ { j }$ denote the positions of emitter $e _ { i }$ and receiver $\rho _ { j } ,$ , respectively.

2) Fine-tuning Sound Designs: We created three sound designs to fine-tune the encoder from the baseline. Design 2 was semantically equivalent to Design 1. However, listeners locations were changed. For Design 3, we made the sound of enemies and bullets the same, with unchanged amplitude at varying distances, and used only one listener. As a result, Design 3 was less informative than both Design 1 and Design 2. Design 4 further restricts the auditory field by limiting sound emission to only two enemies and their bullets, while preserving the indistinguishability of enemy and bullet sounds and the constant amplitude characteristics of Design 3, but utilizing two listeners.

## C. VizDoom

VizDoom [36] is an AI research platform for games based on Doom, a first-person shooter game. It supports the development of game AI agents that play the game using visual and audio buffers, making it suitable for research in deep reinforcement learning. There have been various studies about reinforcement learning conducted, where VizDoom serves as the environment [13], [37]–[39].

(a) Design 1: four listeners (white dots) receive distinct sound cues from enemies (yellow) and bullets (green/blue).  
![](images/0f1d00e71c083e06c049f71fe68007e921946a6e0f061c6e586cc751939fb266.jpg)  
(a) Design 1  
Fig. 5. HYPERHOT sound designs.

![](images/fb5a2b285b2324a8088f97d44636ccf063c2c805a65d64bca6eafc7a147ee3ea.jpg)  
(b) Design 2

![](images/9d0e961fad8135e6bd7817958ee756f3b1c7d0bf01eb87e9fda6a7398daf344d.jpg)  
(c) Design 3

![](images/1ee2043abfd2be9ccfd3e3da8848ffbe4a628cf5833f43289e8361289c5be468.jpg)  
(d) Design 4  
(b) Design 2: same as (a), but listener positions are changed.  
(c) Design 3: enemies and bullets emit the same pitch, and only one listener remains at the bottom left.  
(d) Design 4: only two enemies and two bullets emit sound, enemies and bullets emit the same pitch, and two listeners remain at the bottom left and bottom right.

In this study, we utilized the Basic scenario<sup>2</sup> provided by the VizDoom team. The environment consists of a rectangular map with uniform gray walls, ceiling, and floor. The agent is initialized at the center of one of the longer walls, while a red circular monster is spawned at a random location along the opposite wall. The agent’s action space is restricted to lateral movement (left/right) and shooting. A single successful shot eliminates the monster, and each episode terminates either upon the monster’s elimination or when a predefined time limit is reached. In this scenario, the agent is rewarded for killing the monster, with the following reward function:

$$
r = { \left\{ \begin{array} { l l } { 1 0 6 } & { { \mathrm { f o r ~ k i l l i n g ~ t h e ~ m o n s t e r } } } \\ { - 5 } & { { \mathrm { f o r ~ e v e r y ~ s h o t } } } \\ { 1 } & { { \mathrm { f o r ~ e v e r y ~ t i c ~ t h e ~ a g e n t ~ i s ~ a l i v e } } } \end{array} \right. }
$$

1) Baseline sound design: For the baseline sound design, we adopted the enhanced configuration proposed by [14] (Design 1). This design introduces an auditory cue to indicate the presence of an enemy, as well as a feedback sound that is triggered when the monster is successfully hit, thereby confirming the effectiveness of the action.

2) Fine-tuning Sound Designs: We evaluated three alternative sound designs alongside the baseline. While Design 2 is semantically equivalent to Design 1, it employs a quadratic scaling for the monster’s volume relative to its distance from the agent, whereas Design 1 utilizes linear scaling. Consequently, these designs provide identical information through different delivery methods. Design 3 introduces a spatial constraint, triggering sound only when the enemy appears on the left side of the player’s camera. Finally, Design 4 represents the default sound design of the Basic scenario, where there are no sound cues indicating the presence of the enemy. In terms of informational density, Design 4 is semantically weaker than Design 3, which in turn is weaker than both Design 1 and Design 2.

## V. RESULTS AND DISCUSSIONS

## A. Evaluation Setup and Hyperparameters

Across all three games, we adopted a consistent evaluation procedure. The multimodal encoder was first pretrained with the baseline sound design and visual input, and a policy network was trained on the resulting latent representation. For each new sound design, the multimodal encoder was finetuned and the policy network was kept intact. The finetuned multimodal encoder then provided input to the trained policy, enabling zero-shot transfer without retraining the agent. Agent performance was evaluated in blind mode using sound-only input, and results were reported as both gameplay performance and computational cost.

For all games, the baseline agent trained from scratch was implemented by replacing the multimodal encoder with a single audio encoder. The hyperparameter and encoder architecture configurations were then adapted for each game. To ensure fair benchmarking and prevent manual tuning bias, we adopted established hyperparameters from prior literature. Specifically, the multimodal encoder training and the Pendulum RL configurations follow [10]. For Hyperhot and ViZDoom, the RL hyperparameters match those validated by [12] and [14], respectively. For the environment, we used a single machine with an A100 NVIDIA GPU. Table T1 in the supplementary document summarizes the hyperparameters used to train the multimodal encoder for the three games. Tables T2, T3, and T4 in the supplementary document present the hyperparameters used to train the policy network and to perform full training from scratch in each game. Figures F1, F2 and F3 in the supplementary document show the detailed architecture of the multimodal encoder in PENDULUM, HY-PERHOT, and VizDoom, respectively.

## B. Effectiveness of Multimodal Encoder Finetuning

Table II presents the agent’s performance across Design 2, Design 3, and Design 4 in the three games. In general, finetuning the multimodal encoder leads to distinct performance improvements. The primary exception occurs with Design 4 in ViZDoom. Because this specific design removes crucial audio cues regarding the enemy’s location, the encoder cannot extract actionable spatial features, effectively reducing the agent’s behavior to random exploration.

![](images/287df990dcbe71680989d38a763b036b081155112ce89d14215b12626592f59c.jpg)  
(a) Design 1

![](images/9fbbe764ec1322187d100b41bc57f769d5135b1875fa259a992ed88dbc43493c.jpg)  
(b) Design 2

![](images/ac72e341e0322db3db6de15c52aa1efdf4fabf28e4e720c5e71856e02b6fe20c.jpg)  
(c) Design 3  
Fig. 6. VizDoom basic scenario sound designs. The listener is shown as a green dot at the bottom. (a) Design 1: enemy emits a sound cue whose volume decreases linearly with distance.

![](images/a61e323daa2fc963d1156935208acae38d363e97e22d2e195c7683ca02a33748.jpg)  
(d) Design 4  
(b) Design 2: same as (a) but the emitted volume decreases quadratically with distance.  
(c) Design 3: the player can only hear the enemy’s sound if it is on the left side of the player’s view.  
(d) Design 4: there is no sound cues indicating the presence of enemy.

Additionally, we evaluated our approach against two baseline methods: training an RL agent from scratch under the blind setting for each sound design, and training a new policy network using the fine-tuned multimodal encoder. Our method achieved comparable performance in Design 2, which shares strong similarities with Design 1. However, performance degraded in Designs 3 and 4. This degradation happens because fine-tuning the multimodal encoder induced a shift in the latent representation space, misaligning it with the frozen policy network’s expected inputs. Crucially, this process did not destroy the latent space; when a new policy network was trained on the fine-tuned encoder, the agent’s performance consistently improved. This indicates that future research should focus on latent space alignment techniques or incorporate limited policy fine-tuning to accommodate these representational shifts.

## C. Agent Performance Across Sound Designs

The comparative results for all games are summarized in Tables I, II, and III. Due to the non-normal distribution of results, we employed the Mann-Whitney U-test and correlation coefficient effect size to evaluate the statistical significance of the difference between two sound designs. As shown in Table III, because all tests share an identical sample size, we calculated the simple average of the effect sizes for each method. In the following subsections, we analyzed these results for each game in detail.

1) PENDULUM: We evaluated the average reward for each setting over 100 episodes in 10 seeded runs. As shown in Table I, the agent’s performance with Design 2 was comparable to that with the baseline (Design 1). Furthermore, Table II showed that our agent exhibited the same performance trend as both the blind-setting RL agent and the agent trained with a policy network atop the fine-tuned encoder. In all three cases, performance decreased sequentially across Design 2, Design 3, and Design 4.

2) HYPERHOT: Due to the sparse reward structure in the HYPERHOT game, we evaluated performance by reporting the average number of successful enemy hits per game. For each setting, we conducted 100 episodes across 10 seeded runs. As shown in Table I, the agent’s performance with Design 2 was statistically indistinguishable from that of the baseline (Design 1). This finding supports our hypothesis that Design 2 conveyed equivalent information to that of the baseline, resulting in comparable agent performance across the two designs. As shown in Table III, although our agent did not show a statistically significant difference in performance between Design 3 and Design 4 $( p = 0 . 2 9 )$ , it distinguished between Design 2 and Design 3 more pronouncedly than the other approaches. Furthermore, the agent trained by our approach produced the same overall performance trend as both the blind-setting RL agent and the two-stage fine-tuning method. Across all three approaches, the performance consistently decreased from Design 2, to Design 3, and Design 4.

TABLE I  
COMPARISON OF FINETUNED AGENT PERFORMANCE BETWEEN THE BASELINE (DESIGN 1) AND DESIGN 2 ACROSS THREE GAMES. STATISTICAL SIGNIFICANCE IS EVALUATED USING THE MANN-WHITNEY U TEST, AND THE 95% CONFIDENCE INTERVALS (CI) FOR THE DIFFERENCE IN MEDIANS ARE CALCULATED USING A TWO-SAMPLE INDEPENDENT PERCENTILE BOOTSTRAP METHOD (NUMBER OF RESAMPLES: 9999).
<table><tr><td>Game</td><td>Design 1</td><td>Design 2</td><td>p-value</td><td>95% CI</td></tr><tr><td>Pendulum</td><td> $- 1 . 0 2 0 \pm 0 . 0 3 3$ </td><td> $- 1 . 0 2 8 \pm 0 . 0 4 8$ </td><td>0.273</td><td>[-0.016, 0.044]</td></tr><tr><td>Hyperhot</td><td> $1 . 7 2 8 \pm 0 . 1 8 3$ </td><td> $1 . 7 8 0 \pm 0 . 1 2 4$ </td><td>0.364</td><td>[-0.299, 0.160]</td></tr><tr><td>ViZDoom</td><td> $0 . 7 9 8 \pm 0 . 0 1 4$ </td><td> $0 . 8 1 0 \pm 0 . 0 1 1$ </td><td>0.076</td><td>[-0.027, 0.006]</td></tr></table>

3) VizDoom: In VizDoom, we evaluated the average reward for each setting over 100 episodes across 10 seeded runs. As shown in Table I, the agent’s performance with Design 2 was comparable to that with the baseline (Design 1). Furthermore, the agent trained by our approach exhibited the same performance trend as both the blind-setting RL agent and the agent trained with a policy network atop the fine-tuned encoder, as shown in Table II. In all three of these cases, the performance decreased sequentially across Design 2, Design 3, and Design 4.

Overall, our agent successfully distinguished the differences between sound designs across all three games, aligning with the performance of the other methods. As shown in Table III, the agent trained by our approach achieved the highest average effect size (0.837) in Pendulum and comparable results in ViZDoom. In addition, as shown in Table II, compared to the two baseline methods, our agent performed comparably on Design 2, though its performance declined on Design 3 and Design 4. Nevertheless, it could still distinguish between these sound designs just as effectively as the baselines. We attribute this to a practical trade-off between computational resources and overall performance.

TABLE II  
COMPARISON OF AGENT PERFORMANCE ACROSS DESIGNS 2, 3, AND 4 IN DIFFERENT GAMES. NO FINE-TUNING: THE MULTIMODAL ENCODER IS TRAINED ONLY ON DESIGN 1 AND IS NOT FINE-TUNED.
<table><tr><td>Game</td><td>Method</td><td>Design 2</td><td>Design 3</td><td>Design 4</td></tr><tr><td rowspan="4">PENDULUM</td><td>Ours</td><td> $- 1 . 0 2 8 \pm 0 . 0 4 8$ </td><td> $- 1 . 3 8 2 \pm 0 . 0 4 3$ </td><td> $- 2 . 4 6 4 \pm 0 . 1 0 9$ </td></tr><tr><td>No fine-tuning</td><td> $- 5 . 8 9 7 \pm 0 . 2 9 8$ </td><td> $- 5 . 8 0 7 \pm 0 . 0 5 7$ </td><td> $- 5 . 4 3 7 \pm 0 . 0 6 5$ </td></tr><tr><td>RL training under blind setting</td><td> $- 1 . 0 0 5 \pm 0 . 0 9 1$ </td><td> $- 2 . 2 3 6 \pm 0 . 0 6 8$ </td><td> $- 2 . 7 1 9 \pm 0 . 2 4 4$ </td></tr><tr><td>Finetuned + Policy Training</td><td> $- 1 . 0 3 0 \pm 0 . 0 9 0$ </td><td> $- 1 . 1 9 8 \pm 0 . 0 4 7$ </td><td> $- 2 . 2 9 8 \pm 0 . 1 5 2$ </td></tr><tr><td rowspan="4">HYPERHOT</td><td>Ours</td><td> $1 . 7 8 0 \pm 0 . 1 2 4$ </td><td> $\overline { { 0 . 9 5 3 \pm 0 . 0 8 8 } }$ </td><td> $\overline { { 0 . 9 0 4 \pm 0 . 0 8 1 } }$ </td></tr><tr><td>No fine-tuning</td><td> $1 . 7 0 1 \pm 0 . 1 5 7$ </td><td> $0 . 8 7 7 \pm 0 . 0 9 1$ </td><td> $0 . 8 7 7 \pm 0 . 0 9 1$ </td></tr><tr><td>RL training under blind setting</td><td> $1 . 9 3 \pm 0 . 1 0 7$ </td><td> $1 . 7 9 \pm 0 . 1 5 7$ </td><td> $1 . 0 1 3 \pm 0 . 0 7 4$ </td></tr><tr><td>Finetuned + Policy Training</td><td> $1 . 8 5 2 \pm 0 . 1 2 4$ </td><td> $1 . 6 9 7 \pm 0 . 1 2 8$ </td><td> $1 . 3 7 6 \pm 0 . 0 7 2$ </td></tr><tr><td rowspan="4">VizDoom</td><td>Ours</td><td> $\overline { { 0 . 8 1 0 \pm 0 . 0 1 1 } }$ </td><td> $- 0 . 4 5 8 \pm 0 . 2 1 1$ </td><td> $- 2 . 5 7 2 \pm 0 . 1 9 0$ </td></tr><tr><td>No fine-tuning</td><td> $0 . 8 0 5 \pm 0 . 0 0 7$ </td><td> $- 1 . 0 4 7 \pm 0 . 2 1 3$ </td><td> $- 2 . 5 5 2 \pm 0 . 2 5 8$ </td></tr><tr><td>RL training under blind setting</td><td> $0 . 8 0 0 \pm 0 . 0 1 3$ </td><td> $0 . 7 4 3 \pm 0 . 0 1 9$ </td><td> $- 1 . 2 4 9 \pm 0 . 2 7 3$ </td></tr><tr><td>Finetuned + Policy Training</td><td> $0 . 8 0 3 \pm 0 . 0 0 8$ </td><td> $- 0 . 1 0 5 \pm 0 . 2 0 3$ </td><td> $- 1 . 9 3 9 \pm 0 . 1 7 6$ </td></tr></table>

TABLE III

PAIRWISE PERFORMANCE COMPARISONS (p-VALUES/EFFECT SIZE) USING MANN-WHITNEY U TEST OF SOUND DESIGNS. D2: DESIGN 2, D3: DESIGN 3, D4: DESIGN 4.
<table><tr><td>Game</td><td>Method</td><td>D2 vs D3</td><td>D3 vs D4</td><td>D2 vs D4</td><td>Average Effect size</td></tr><tr><td rowspan="3">PENDULUM</td><td>Ours</td><td> $\overline { { 1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4 } }$ </td><td> $\overline { { 1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4 } }$ </td><td> $\overline { { 1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4 } }$ </td><td>0.84</td></tr><tr><td>RL training under blind setting</td><td> $1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4$ </td><td> $3 . 3 \times 1 0 ^ { - 4 } / 0 . 8 0$ </td><td> $1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4$ </td><td>0.83</td></tr><tr><td>Finetuned + Policy Training</td><td> $4 . 4 \times 1 0 ^ { - 4 } / 0 . 7 9$ </td><td> $1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4$ </td><td> $1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4$ </td><td>0.82</td></tr><tr><td rowspan="3">HYPERHOT</td><td>Ours</td><td> $\overline { { 1 . 7 \times 1 0 ^ { - 4 } / 0 . 8 4 } }$ </td><td> $\overline { { 0 . 2 9 / 0 . 2 4 } }$ </td><td> $\overline { { 1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4 } }$ </td><td>0.64</td></tr><tr><td>RL training under blind setting</td><td> $4 . 5 \times 1 0 ^ { - 2 } / 0 . 4 4$ </td><td> $1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4$ </td><td> $1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4$ </td><td>0.70</td></tr><tr><td>Finetuned + Policy Training</td><td> $1 . 9 \times 1 0 ^ { - 2 } / 0 . 5 2$ </td><td> $2 . 4 \times 1 0 ^ { - 4 } / 0 . 8 2$ </td><td> $1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4$ </td><td>0.73</td></tr><tr><td rowspan="3">VizDoom</td><td>Ours</td><td> $\overline { { 1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4 } }$ </td><td> $\overline { { 1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4 } }$ </td><td> $\overline { { 1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4 } }$ </td><td>0.84</td></tr><tr><td>RL training under blind setting</td><td> $1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4$ </td><td> $1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4$ </td><td> $1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4$ </td><td>0.84</td></tr><tr><td>Finetuned + Policy Training</td><td> $1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4$ </td><td> $1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4$ </td><td> $1 . 8 \times 1 0 ^ { - 4 } / 0 . 8 4$ </td><td>0.84</td></tr></table>

## D. Computational Resource Comparison

In addition to agent performance, we evaluated the efficiency of our method using hardware-agnostic metrics for parameter, computational efficiency, and sample efficiency, while excluding hardware-dependent measures such as wallclock time.

1) Parameter Efficiency and Sample Efficiency: To compare the two approaches, we evaluated both parameter efficiency and sample efficiency. Parameter efficiency was quantified by the number of trainable parameters, which reflected the storage requirements and the memory usage of the optimizer. Sample efficiency was measured by the number of training samples consumed by each approach, providing a direct indication of computational cost.

2) Computational Efficiency: We analyzed computational efficiency by reporting the total training cost in terms of floating-point operations (FLOPs) per sound design. For each method, the cost was estimated by first measuring the FLOPs per training step using torch.utils.flop\_counter, and then multiplying this value by the total number of training steps S.

$$
C o s t _ { t o t a l } = F L O P s _ { s t e p } \times S ,
$$

where $F L O P s _ { s t e p }$ is backward F LOP s per step. We excluded the computational cost of inference as it was negligible compared to the training cost, and we also excluded the policy network training cost because it was performed only once on the baseline sound design. For each new sound design, only the multimodal encoder was fine-tuned. For the approach involving a policy network trained atop the fine-tuned encoder, the total computational overhead is the aggregate of the multimodal fine-tuning phase and the subsequent reinforcement learning stage.

Tables IV showed that our approach was substantially more sample-efficient and required fewer trainable parameters than the others. For PENDULUM, the backward FLOPs were higher in our method. Although the common encoder and the MLP block connecting the audio features block and the image features block of the fusion encoder were frozen, their output gradients still had to be computed to propagate the loss signal back to the trainable audio encoder and the trainable audio features block of the fusion encoder. In addition, only in the PENDULUM setting did the output dimension of the image features block (14,400) exceed that of the audio features block (50), whereas in the other games these dimensions were aligned. This further amplified the backward computational load in PENDULUM. As a result, the backward pass incurred additional cost, even though no parameters were updated within the frozen modules.

By comparison, the RL agent trained under the blind setting required the agent to learn by trial-and-error through interaction with the environment, and each sample must be generated via simulation. This process was computationally expensive and often provides a limited learning signal, particularly in tasks with sparse rewards. Furthermore, methods that train a policy network on top of a fine-tuned multimodal encoder incur the costs of both fine-tuning and policy training. In contrast, our method required only a much smaller data set to train the multi-modal representation, thus reducing both the sample requirements and the overall computational cost.

TABLE IV  
COMPARISON OF TRAINING COST IN PENDULUM, HYPERHOT AND VIZDOOM.
<table><tr><td>Game</td><td>Method</td><td>Samples</td><td>Trainable Parameters</td><td>Backward FLOPs</td><td>Training Steps</td><td>Total FLOPs</td></tr><tr><td rowspan="4">PENDULUM</td><td>Ours</td><td>20,000</td><td>9,664</td><td>0.25 GFLOP</td><td>70,499</td><td>17 TFLOP</td></tr><tr><td>RL training under blind setting</td><td>150,000</td><td>340,268</td><td>0.19 GFLOP</td><td>150,000</td><td>29 TFLOP</td></tr><tr><td>Finetuned + Policy Training</td><td>20,000 +</td><td>9,664 +</td><td>0.25 GFLOP +</td><td>70,499 +</td><td>17 TFLOP +</td></tr><tr><td></td><td>150,000</td><td>275,972</td><td>0.36 GFLOP</td><td>150,000</td><td>54 TFLOP</td></tr><tr><td rowspan="3">HYPERHOT</td><td>Ours RL training under blind setting</td><td>32,000 2,000,000</td><td>9,140,288 9,248,220</td><td>2.58 GFLOP 4.8 GFLOP</td><td>56,250</td><td>145 TFLOP 9632 TFLOP</td></tr><tr><td></td><td>32,000 +</td><td>9,248,220 +</td><td>2.58 GFLOP +</td><td>2,000,000 56,250 +</td><td>145 TFLOP +</td></tr><tr><td>Finetuned + Policy Training</td><td>2,000,000</td><td>48,140</td><td>0.024 GFLOP</td><td>2,000,000</td><td>48 TFLOP</td></tr><tr><td rowspan="4">VizDoom</td><td>Ours</td><td>32,000</td><td>245,312</td><td>0.24 GFLOP</td><td>56,250</td><td>13.5 TFLOP</td></tr><tr><td>RL training under blind setting</td><td> $5 \times 1 0 ^ { 8 }$ </td><td>1,684,741</td><td>2.3 GFLOP</td><td> $5 \times 1 0 ^ { 8 }$ </td><td> $1 . 1 5 \times 1 0 ^ { 6 } \ \mathrm { T F L O P }$ </td></tr><tr><td>Finetuned + Policy Training</td><td> ${ 3 2 , 0 0 0 + }$ </td><td>245,312 +</td><td>0.24 GFLOP +</td><td> $^ { 5 6 , 2 5 0 + }$ </td><td> $1 3 . 5 \mathrm { \ T F L O P + }$ </td></tr><tr><td></td><td> $5 \times 1 0 ^ { 8 }$ </td><td>853,509</td><td>18.95 GFLOP</td><td> $5 \times 1 0 ^ { 8 }$ </td><td> $9 . 4 7 \times 1 0 ^ { 6 } \ \mathrm { T F L O P }$ </td></tr></table>

TABLE V

LATENT REPRESENTATION SIMILARITIES ACROSS SOUND DESIGNS IN PENDULUM, HYPERHOT AND VIZDOOM.
<table><tr><td>Game</td><td>Sound Design</td><td>Image-Sound</td><td>Image-Joint</td><td>Sound-Joint</td><td>Average</td></tr><tr><td rowspan="4">PENDULUM</td><td>Design 1</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>Design 2</td><td>0.99</td><td>1.00</td><td>0.99</td><td>0.99</td></tr><tr><td>Design 3</td><td>0.83</td><td>0.98</td><td>0.87</td><td>0.89</td></tr><tr><td>Design 4</td><td>0.80</td><td>0.98</td><td>0.85</td><td>0.88</td></tr><tr><td rowspan="4">HYPERHOT</td><td>Design 1</td><td>0.68</td><td>0.44</td><td>0.52</td><td>0.55</td></tr><tr><td>Design 2</td><td>0.64</td><td>0.45</td><td>0.50</td><td>0.53</td></tr><tr><td>Design 3</td><td>0.15</td><td>0.14</td><td>0.18</td><td>0.16</td></tr><tr><td>Design 4</td><td>0.12</td><td>0.14</td><td>0.17</td><td>0.14</td></tr><tr><td rowspan="4">VizDoom</td><td>Design 1</td><td>0.49</td><td>0.53</td><td>0.97</td><td>0.66</td></tr><tr><td>Design 2</td><td>0.52</td><td>0.62</td><td>0.93</td><td>0.69</td></tr><tr><td>Design 3</td><td>0.33</td><td>0.53</td><td>0.85</td><td>0.57</td></tr><tr><td>Design 4</td><td>0.08</td><td>0.44</td><td>0.70</td><td>0.41</td></tr></table>

## E. Impact of Latent Representation Adaptation on Reinforcement Learning Policy

Table V presents the average cosine similarity of the latent modality representations across the three games evaluated in our experiments. Each modality was first processed by a modality-specific encoder and then projected into a shared latent space via a common encoder. In all the games, Design 1 and Design 2 showed comparable modality similarities, both in theory and in practice. Since they were constructed to preserve a similar structure, their empirical similarity scores align perfectly with our initial design hypotheses.

In PENDULUM, we observed an almost perfect alignment between sound and image, with similarity scores approaching 1.0 for both Design 1 and Design 2. This indicated that the auditory cues in such settings directly captured the underlying game state. However, in more complex environments like HYPERHOT and VizDoom, the similarity values decreased substantially, reflecting the increased difficulty of aligning audio and visual modalities when the dynamics and sound effects are more intricate. While we observe that cross-modality similarity naturally decreases as game scenarios become more complex, this does not preclude the framework’s real-world applicability but shifts how the framework is deployed. In highly complex, modern game environments, purely zero-shot transfer with a strictly frozen policy may reach a performance ceiling. However, our framework can serve as a foundation for a semi-automated evaluation pipeline. By allowing for limited, few-shot finetuning of the policy network on top of the adapted multimodal encoder, the agent can rapidly align with complex new audio cues. To validate this hypothesis, we conducted an ablation study by fine-tuning the policy network for only 20% of the maximum RL training steps. As shown in Table VI, this limited adaptation consistently improved agent performance across all sound designs and games, confirming the efficacy of few-shot policy recovery. Crucially, this approach remains more computationally efficient than the standard baseline of training an RL agent completely from scratch for every sound design iteration. Therefore, the proposed method provides a highly scalable solution for iterative audio quality assurance in realistic game development cycles.

TABLE VI  
PERFORMANCE OF FINE-TUNED POLICY AGENTS USING DESIGN 3 ANDDESIGN 4 ACROSS THREE GAMES. VALUES IN PARENTHESES DENOTE THEPERFORMANCE IMPROVEMENT RELATIVE TO BASELINE AGENTS WHEREONLY THE MULTIMODAL ENCODER WAS FINE-TUNED
<table><tr><td>Game</td><td>Design 3</td><td>Design 4</td></tr><tr><td>Pendulum</td><td>-1.081 (+0.301)</td><td>-1.893 (+0.571)</td></tr><tr><td>Hyperhot</td><td>1.538 (+0.585)</td><td>1.073 (+0.169)</td></tr><tr><td>VizDoom</td><td>0.272 (+0.73)</td><td>-1.631 (+0.941)</td></tr></table>

In contrast, Design 3 and Design 4 represented a poorer design choice compared to Designs 1 and 2, and this was consistently reflected in the marked decline of modality similarity scores across all three games. This reduction in alignment was consistently mirrored by degraded agent performance, showing that the pretrained policy could not fully interpret the altered sound cues without fine-tuning. Taken together, these findings demonstrate that, across three different games, the expected similarity between Design 1 and Design 2 was confirmed both empirically in modality alignment and in the comparable performance of agents. While further validation is required, these results suggest a potential link between crossmodal similarity scores and agent robustness in unseen sound conditions.

## F. Limitations

Our approach has several limitations that point to directions for future research. First and most importantly, our framework evaluates the functional informativeness of sound rather than its perceptual quality or emotional impact. While our AIbased metrics provide an objective measure of sound utility, they do not capture the aesthetic or subjective experience of a human listener. Consequently, this framework is intended to complement, rather than replace, user-centered studies that assess the emotional resonance and artistic value of sound design.

Second, although current evaluation environments are relatively simplified, the framework’s modular architecture is designed to scale to complex, realistic games. By replacing lightweight encoders with advanced audio foundation models and incorporating 3D spatial features, the system can be adapted for high-fidelity titles and partial observability. Third, the transition from from-scratch training to fine-tuning introduces a latent-shift mismatch between the updated encoder and the fixed policy network. Future work should investigate limited policy adaptation or fixed-space constraints to mitigate this. Furthermore, the granular contribution of individual sound cues remains an open question that warrants deeper investigation.

Beyond its theoretical contributions, this framework offers immediate practical utility for game developers. It serves as an objective, automated quality assurance tool that can be integrated into early-stage development pipelines to identify uninformative audio cues. By providing a scalable proxy for sound utility, designers can rapidly iterate on audio-visual balance before committing to expensive and time-consuming human playtesting.

## VI. CONCLUSION

This study introduced a framework for evaluating in-game sound designs by combining multimodal representation learning with RL. In our approach, only the multimodal encoder was fine-tuned for each new sound design, while the policy network remained fixed. Experimental results showed that the framework provides a highly effective and efficient alternative to training a new agent from scratch for distinguishing differences between sound designs, substantially lowering computational cost and thereby improving scalability. Furthermore, the observed correlation between latent representation efficiency and agent performance highlighted the potential of this approach and points to future research on advancing multimodal learning techniques for sound design evaluation in complex game environments.

## REFERENCES

[1] K. Tuuri, O. Koskela, J. Vahlo, and H. Tissari, “Identifying the impact of game music both within and beyond gameplay,” in Entertainment Computing – ICEC 2021, J. Baalsrud Hauge, J. C. S. Cardoso, L. Roque, and P. A. Gonzalez-Calero, Eds. Cham: Springer International Publishing, 2021, pp. 411–418.

[2] F. Andersen, Danny, C. L. King, and A. A. S. Gunawan, “Audio influence on game atmosphere during various game events,” Procedia Computer Science, vol. 179, pp. 222–231, 2021, 5th International Conference on Computer Science and Computational Intelligence 2020. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S1877050921000016

[3] K. Worrall, “Final fantasy vii remake: Music redesign for evolved expectations across console generations,” Journal of Sound and Music in Games, vol. 5, no. 4, pp. 34–57, 10 2024. [Online]. Available: https://doi.org/10.1525/jsmg.2024.5.4.34

[4] F. Ganis, A. Gulli, F. Fontana, and S. Serafin, “The role of haptics in training and games for hearing-impaired individuals: A systematic review,” Multimodal Technologies and Interaction, vol. 8, no. 1, 2024. [Online]. Available: https://www.mdpi.com/2414-4088/8/1/1

[5] B. Yuan, E. Folmer, and F. C. Harris Jr, “Game accessibility: a survey,” Universal Access in the information Society, vol. 10, no. 1, pp. 81–100, 2011.

[6] N. Mohd Norowi, H. Azman, and N. Wahiza Abdul Wahat, “Apple swipe: A mobile game apps for visually impaired users using binaural sounds,” in Proceedings of the Asian CHI Symposium 2021, ser. Asian CHI ’21. New York, NY, USA: Association for Computing Machinery, 2021, p. 196–201. [Online]. Available: https://doi.org/10.1145/3429360.3468209

[7] A. Adkins, K. Kohm, R. Zhang, and N. Gustafson, “Lost in spaze: An audio maze game for the visually impaired,” in Extended Abstracts of the 2020 CHI Conference on Human Factors in Computing Systems, ser. CHI EA ’20. New York, NY, USA: Association for Computing Machinery, 2020, p. 1–6. [Online]. Available: https://doi.org/10.1145/3334480.3381660

[8] R. D. Gaina and M. Stephenson, ““Did You Hear That?” Learning to Play Video Games from Audio Cues,” in 2019 IEEE Conference on Games (CoG), 2019, pp. 1–4.

[9] T. Van Nguyen, X. Dai, I. Khan, R. Thawonmas, and H. V. Pham, “A deep reinforcement learning blind ai in darefightingice,” in 2022 IEEE Conference on Games (CoG), 2022, pp. 632–637.

[10] P. Poklukar, M. Vasco, H. Yin, F. S. Melo, A. Paiva, and D. Kragic, “Geometric multimodal contrastive representation learning,” in Proceedings of the 39th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, Eds., vol. 162. PMLR, 17–23 Jul 2022, pp. 17 782–17 800. [Online]. Available: https://proceedings.mlr.press/v162/poklukar22a.html

[11] R. Silva, M. Vasco, F. S. Melo, A. Paiva, and M. Veloso, “Playing games in the dark: An approach for cross-modality transfer in reinforcement learning,” in Proceedings of the 19th International Conference on Autonomous Agents and MultiAgent Systems, ser. AAMAS ’20. Richland, SC: International Foundation for Autonomous Agents and Multiagent Systems, 2020, p. 1260–1268.

[12] M. Vasco, H. Yin, F. S. Melo, and A. Paiva, “How to sense the world: Leveraging hierarchy in multimodal perception for robust reinforcement learning agents,” in Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems, ser. AAMAS ’22. Richland, SC: International Foundation for Autonomous Agents and Multiagent Systems, 2022, p. 1301–1309.

[13] S. Hegde, A. Kanervisto, and A. Petrenko, “Agents that listen: Highthroughput reinforcement learning with multiple sensory systems,” in 2021 IEEE Conference on Games (CoG). IEEE Press, 2021, p. 1–5. [Online]. Available: https://doi.org/10.1109/CoG52621.2021.9619096

[14] I. Khan, T. Van Nguyen, M. C. Gursesli, and R. Thawonmas, “Sonic doom: Enhanced sound design and accessibility in a first-person shooter game,” in 2025 IEEE Conference on Games (CoG), 2025, pp. 1–8.

[15] I. Khan, T. V. Nguyen, X. Dai, and R. Thawonmas, “Darefightingice competition: A fighting game sound design and ai competition,” in 2022 IEEE Conference on Games (CoG). IEEE Press, 2022, p. 478–485. [Online]. Available: https://doi.org/10.1109/CoG51982.2022.9893624

[16] D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and J. Davidson, “Learning latent dynamics for planning from pixels,” in International conference on machine learning. PMLR, 2019, pp. 2555– 2565.

[17] D. Ha and J. Schmidhuber, “World models,” arXiv preprint arXiv:1803.10122, vol. 2, no. 3, 2018.

[18] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.

[19] A. Stooke, K. Lee, P. Abbeel, and M. Laskin, “Decoupling representation learning from reinforcement learning,” in International conference on machine learning. PMLR, 2021, pp. 9870–9879.

[20] F. Lygerakis, V. Dave, and E. Rueckert, “M2curl: sample-efficient multimodal reinforcement learning via self-supervised representation learning for robotic manipulation,” in 2024 21st International Conference on Ubiquitous Robots (UR). IEEE, 2024, pp. 490–497.

[21] P. Becker, S. Mossburger, F. Otto, and G. Neumann, “Combining reconstruction and contrastive methods for multimodal representations in RL,” Reinforcement Learning Conference (RLC), 2024.

[22] Y. Zhang, K. Chen, and Y. Liu, “Causal representation learning in offline visual reinforcement learning,” Knowledge-Based Systems, vol. 320, p. 113565, 2025. [Online]. Available: https://www.sciencedirect. com/science/article/pii/S0950705125006112

[23] H. Cao, F. Feng, M. Fang, S. Dong, T. Yang, J. Huo, and Y. Gao, “Towards empowerment gain through causal structure learning in modelbased reinforcement learning,” in The Thirteenth International Conference on Learning Representations, 2025.

[24] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to control: Learning behaviors by latent imagination,” in International Conference on Learning Representations, 2020. [Online]. Available: https://openreview.net/forum?id=S1lOTC4tDS

[25] D. Hafner, T. P. Lillicrap, M. Norouzi, and J. Ba, “Mastering Atari with Discrete World Models,” in International Conference on Learning Representations, 2021. [Online]. Available: https: //openreview.net/forum?id=0oabwyZbOu

[26] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, “Mastering diverse control tasks through world models,” Nature, pp. 1–7, 2025.

[27] J. Ngiam, A. Khosla, M. Kim, J. Nam, H. Lee, A. Y. Ng et al., “Multimodal deep learning.” in ICML, vol. 11, 2011, pp. 689–696.

[28] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[29] L. Li, G. Chen, H. Wang, B. Li, B. Wang, Z. Yi, and C. Zhao, “VHTformer: A joint query perception method for visual-haptic-textual information based on Transformer,” Applied Soft Computing, vol. 181, p. 113529, 2025. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1568494625008403

[30] J. Zou, Z. Wan, F. Wang, S. Ye, and S. Liu, “The self supervised multimodal semantic transmission mechanism for complex network environments,” Scientific Reports, vol. 15, no. 1, p. 29899, 2025.

[31] Y.-H. H. Tsai, P. P. Liang, A. Z. 0001, L.-P. Morency, and R. Salakhutdinov, “Learning Factorized Multimodal Representations,” in 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. [Online]. Available: https://openreview.net/forum?id=rygqqsA9KX

[32] Y.-H. H. Tsai, S. Bai, P. P. Liang, J. Z. Kolter, L.-P. Morency, and R. Salakhutdinov, “Multimodal transformer for unaligned multimodal language sequences,” in Proceedings of the conference. Association for computational linguistics. Meeting, vol. 2019, 2019, p. 6558.

[33] P. P. Liang, Y. Lyu, X. Fan, Z. Wu, Y. Cheng, J. Wu, L. Chen, P. Wu, M. A. Lee, Y. Zhu et al., “Multibench: Multiscale benchmarks for multimodal representation learning,” Advances in neural information processing systems, vol. 2021, no. DB1, p. 1, 2021.

[34] G. Barnum, S. Talukder, and Y. Yue, “On the benefits of early fusion in multimodal representation learning,” arXiv preprint arXiv:2011.07191, 2020.

[35] G. Sterpu, C. Saam, and N. Harte, “Attention-based audio-visual fusion for robust automatic speech recognition,” in Proceedings of the 20th ACM International Conference on Multimodal Interaction, ser. ICMI ’18. New York, NY, USA: Association for Computing Machinery, 2018, p. 111–115. [Online]. Available: https://doi.org/10.1145/3242969. 3243014

[36] M. Kempka, M. Wydmuch, G. Runc, J. Toczek, and W. Jaskowski,´ “ViZDoom: A Doom-based AI research platform for visual reinforcement learning,” in IEEE Conference on Computational Intelligence and Games. Santorini, Greece: IEEE, Sep 2016, pp. 341–348, the Best Paper Award.

[37] M. Wydmuch, M. Kempka, and W. Jaskowski, “ViZDoom Competitions:´ Playing Doom from Pixels,” IEEE Transactions on Games, vol. 11, no. 3, pp. 248–259, 2019, the 2022 IEEE Transactions on Games Outstanding Paper Award.

[38] Y. Xu, “Deep reinforcement learning and imitation learning based on VizDoom,” in Proceedings of the 2022 6th International Conference on Electronic Information Technology and Computer Engineering, ser. EITCE ’22. New York, NY, USA: Association for Computing Machinery, 2023, p. 1700–1706. [Online]. Available: https://doi.org/10.1145/3573428.3573729

[39] A. Zakharenkov and I. Makarov, “Deep Reinforcement Learning with DQN vs. PPO in VizDoom,” in 2021 IEEE 21st International Symposium on Computational Intelligence and Informatics (CINTI), 2021, pp. 000 131–000 136.