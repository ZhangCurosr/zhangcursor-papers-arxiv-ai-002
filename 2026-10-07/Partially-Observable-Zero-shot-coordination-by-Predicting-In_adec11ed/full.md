# Partially Observable Zero-shot coordination by Predicting Intention of Partner

Jinnyeong Yang<sup>∗</sup> Yuhwan Jeong<sup>∗</sup> Hoyong Kwon<sup>∗</sup> Minseok Kim<sup>†</sup> Jihun Kim<sup>†</sup> Kuk-Jin Yoon

KAIST, Visual Intelligence Lab

{jinnyeong6118, jeongyh98, kwonhoyong3, alstjrx1x1, jihun1998, kjyoon}@kaist.ac.kr

## Abstract

Zero-shot coordination in embodied settings requires acting while the partner is intermittently out of view, leaving existing methods with ambiguous partner representations and uncertainty over hidden partner states. We propose Predicting Intention of Partner (PIP) to jointly address these challenges. PIP uses a Jointview VAE to distill richer training-time evidence from the union of both agents local observations into a partner representation available from local observations alone. Partner-state Belief networks further infer the partner’s hidden location and behavioral tendencies from the ego agent’s interaction history. We evaluate PIP in Burrito-PO, Overcooked-PO, and a Melting Pot substrate, together with a human evaluation in Burrito-PO. PIP attains the highest mean performance among the compared methods across all three benchmarks. Human evaluation and diagnostic analyses further support coordination with unseen partners and the contributions of both components under partner occlusion.

## 1 Introduction

As AI agents and robots are increasingly deployed in real-world interactive systems, human-AI collaboration has become a central goal of artificial intelligence. In such settings, an agent must coordinate with diverse unseen humans or partner agents that were never encountered during training. This requirement has motivated the study of zero-shot coordination (ZSC) [18, 25, 28], whose central difficulty is the shift of partner distribution between training and deployment. Agents trained through self-play [15, 40] often converge to arbitrary conventions and fail when paired with partners that follow different ones. To mitigate this, prior work has expanded the diversity of the partner population used during training [5, 43, 50]. Representing the space of possible partners with a finite set of policies, however, is inherently limited. Generative prior-based methods [6, 23, 26] have therefore emerged as a promising alternative. By modeling a continuous latent space of partner behaviors from coordination trajectories, they expose the target policy to partners beyond the observed population, and have achieved remarkable results in fully observable environments such as Overcooked [4].

In practical collaboration, however, limited sensing and the movement of agents rarely afford full observability over either the task state or the partner [34, 35]. Under such partial observability (PO), coordination cues are confined to the ego agent’s observable region, as illustrated in Fig. 1. The agent must coordinate with unseen partners from incomplete information about partner behavior and task state. Within ZSC, partial observability has been studied primarily in the informational sense exemplified by Hanabi [2], where the hidden information is prescribed by the game rules. In such settings the partner’s actions remain visible, and what is concealed is a fixed set of variables. Embodied collaboration instead induces spatial and dynamic partial observability.

What the ego agent perceives changes over time with its restricted field of view and with the movement of both agents. The information asymmetry here is therefore induced by viewpoint. Settings of this kind have only recently been taken up as ZSC problems, with existing studies focusing on environment design [10] or symmetry [31]. Yet the failure modes induced by viewpoint-based partial observability itself remain underexplored.

(b) Partially observable  
![](images/316ed6bc8cd827188a6e3792ce66e3d9f6feab94cb59cd42484ebe10da8b2c14.jpg)

![](images/0ff6cd2c0f85b8eec833f7a1b9971586f4760518c53793b7010ab171c05288a4.jpg)  
Figure 1: Illustration of coordination under dynamic and spatial partial observability in Burrito-PO. (a) Under full observation, the partner state is directly available. (b) Under a restricted local view, the partner may leave the visible region, creating ambiguity over its location and behavior.

In this work, we study ZSC under dynamic and spatial partial observability. We identify two key challenges that arise when VAE-based generative ZSC is transferred to partially observable environments. First, partial observability degrades the partner diversity captured in the generator’s latent space. Because informative segments of the partner’s motion are occluded outside the field of view, the encoder receives distinct partner trajectories as similar local observations. Behavior patterns that are separable under richer observations thus collapse into overlapping latent regions, restricting the coverage of the modeled partner space. To address this, we introduce a Joint-view VAE. A teacher encoder receives the union of the two agents’ local observations, and its posterior is distilled into a student encoder that sees only the ego-centric observation. This preserves behaviorally meaningful variation under limited input and retains diversity in the generated partners. Here, the joint view denotes additional partial-view evidence available only during VAE training, not access to the unobservable global state. The resulting generator therefore remains deployment-compatible.

Second, partial observability restricts the evidence on which the target policy bases its decisions. Once the partner leaves the ego agent’s observable region, essential information such as its current location, recent action, and intended subtask is lost [8, 13, 21, 33, 37]. Since coordination decisions depend heavily on where the partner is and what it will do next, the policy must infer hidden partner-related information from its own observation history. Inspired by belief-based modeling in partially observable reinforcement learning [17, 19, 52], we introduce Partner-state Belief networks. A Location Belief estimates the position of the partner while it is out of view, and a Behavioral Belief predicts what the partner is likely to do. Together they allow the policy to compensate for uncertainty about the partner and to act in a belief-aware manner. Importantly, we follow a decentralized PO protocol during policy execution. Logged partner information is used only as auxiliary supervision for the belief networks at the PPO update stage, and is never provided as input to the policy. At deployment, the policy therefore acts solely from its own observations and its self-estimated beliefs.

Integrating these two components, we propose Predicting Intention of Partner (PIP), a framework for zero-shot coordination under dynamic and spatial partial observability. The Joint-view VAE improves the diversity of generated partners, and the Partner-state Belief enables belief-aware decision making. PIP thus jointly addresses partner generation and policy learning from incomplete observations. We evaluate PIP on Overcooked-PO, Burrito-PO, and Melting Pot [1, 4, 23], benchmarks that combine dynamic cooperative tasks with spatial partial observability. PIP achieves strong coordination performance with both unseen partners and humans, and further analyses demonstrate that each component operates as intended and contributes to robust coordination. Our main contributions are as follows:

• To the best of our knowledge, our work is the first to address the failure modes that dynamic spatial partial observability induces in both partner generation and policy learning in ZSC.

• We propose PIP, which combines a Joint-view VAE for diverse partner generation with Partner-state Belief networks to handle information asymmetry in spatial environment.

• Extensive experiments including human study on the Burrito-PO, Overcooked-PO, and Melting Pot benchmarks demonstrate the effectiveness of PIP.

## 2 Related works

Zero-Shot Coordination and Partner Diversity. Zero-shot coordination requires agents to collaborate with previously unseen partners, including independently trained agents and humans [15, 41]. Other-Play [15] and Any-Play [27] reduce reliance on arbitrary coordination conventions. CEC [18] promotes coordination between agents trained in different environments. Another major direction is population-based training, which exposes a target policy to diverse partners and conventions. Existing methods construct partner populations or explicitly optimize their diversity [28, 38, 43, 50]. GAMMA [26], TALENTS [23], and GOAT [6] use generative latent spaces to produce diverse or challenging partners. E3T [48] avoids a pretrained population by constructing partners from mixtures of the ego and random policies while jointly learning partner action prediction. Collectively, these approaches improve generalization by exposing the target policy to diverse behaviors and coordination conventions during training.

Zero-Shot Coordination under Partial Observability. PO-ZSC includes different settings depending on which information is unobserved and how it becomes unavailable. A representative line of work considers environments such as Hanabi [2], where private or asymmetric information is defined by the game structure. In Hanabi, agents have access to different card information, and actions serve both task execution and communication. Prior studies combine compatibility with novel partners and belief-based inference of hidden information [8, 15, 28, 30]. ERS [31] instead broadens the symmetry class to discourage incompatible conventions in Dec-POMDPs. These studies primarily address information asymmetry imposed by game rules. In spatial coordination environments, however, observability changes with agent positions and movements. OvercookedV2 [10] introduces restricted fields of view and asymmetric task information and evaluates existing ZSC methods under spatial partial observability. However, existing methods handle the resulting intermittent loss of partner ob servations only implicitly. To the best of our knowledge, prior work that jointly addresses incomplete partner representations and explicitly infers the positions and actions of currently unobserved partners remains limited.

## 3 Preliminaries

Two-Player Partially Observable Markov Decision Process. A two-player partially observable Markov decision process extends the standard two-player Markov decision process to settings with incomplete information. Formally, the environment is defined as $\begin{array} { r l } { \mathcal { M } } & { { } = } \end{array}$ $( \mathcal { S } , \mathcal { A } ^ { 0 } , \breve { \mathcal { A } } ^ { 1 } , \mathcal { O } ^ { 0 } , \mathcal { O } ^ { 1 } , P , \breve { R , } \Omega , \gamma )$ , where S denotes the global state space, and A<sup>i</sup> and $\mathcal { O } ^ { i }$ denote the action and observation spaces of agent $i \in \{ 0 , 1 \}$ , respectively. $P ( s ^ { \prime } \mid s , a ^ { 0 } , a ^ { 1 } )$ is the transition dynamics, $R ( s , a ^ { 0 } , a ^ { 1 } , s ^ { \prime } )$ is the shared reward function, $\bar { \Omega } ( o ^ { 0 } , o ^ { \mathrm { i } } \mid s )$ denotes the observation function, and $\gamma \in \left[ 0 , 1 \right)$ is the discount factor. At each timestep $t ,$ both agents receive local observations $( o _ { t } ^ { 0 } , o _ { t } ^ { 1 } ) \sim \Omega ( \cdot | \ s _ { t } )$ rather than the full global state $s _ { t } .$ , and each agent selects an action $a _ { t } ^ { i } \sim \pi _ { i } ( \cdot \mid h _ { t } ^ { i } )$ conditioned on its local action-observation history $h _ { t } ^ { i } = ( o _ { 0 } ^ { i } , a _ { 0 } ^ { i } , \ldots , o _ { t } ^ { i } )$ . The joint transition then follows $s _ { t + 1 } \sim P ( \cdot \mid s _ { t } , a _ { t } ^ { 0 } , a _ { t } ^ { 1 } )$ The objective is to learn policies $( \pi _ { 0 } , \pi _ { 1 } )$ that maximize the expected discounted return $\begin{array} { r l } & { \underset { \pi _ { 0 } , \pi _ { 1 } } { \operatorname* { m a x } } \mathbb { E } \left[ \sum _ { t = 0 } ^ { H } \gamma ^ { t } R ( s _ { t } , a _ { t } ^ { 0 } , a _ { t } ^ { 1 } , s _ { t + 1 } ) \right] } \end{array}$ . In this work, the partner’s strategy is not directly observed and must be inferred from local interaction history alone.

Generative Latent-Variable ZSC. Recent generative latent-variable approaches to ZSC [6, 26] represent a cooperation strategy as a latent variable $z \in { \mathcal { Z } }$ in a continuous latent space. These methods first collect partner trajectory data, such as trajectories generated by MEP [50]-trained policies. They then learn a generative representation of partner behaviors to model diverse coordination strategies. Let $\tau = \stackrel { \smile } { ( } o _ { 0 } ^ { p } , a _ { 0 } ^ { p } , \ldots , \stackrel { \bullet } { o } _ { T } ^ { p } , a _ { T } ^ { p } )$ denote a trajectory generated by a partner policy, and let $h _ { t } ^ { p } = \mathbf { \bar { \Phi } } ( o _ { 0 } ^ { p } , a _ { 0 } ^ { p } , \ldots , o _ { t } ^ { p } )$ be the partner’s history up to time t. An encoder $q _ { \phi } ( z \mid \bar { \tau } )$ maps the trajectory to a posterior over latent strategies, where z captures high-level behavioral factors such as convention, preference, or execution style. Conditioned on $z ,$ a decoder p<sub>θ</sub> $\left( a _ { t } ^ { p } \ | \ z , h _ { t } ^ { p } \right)$ predicts the partner’s next action, thereby defining a latent-conditioned partner policy $\dot { \mu _ { z } } \dot { ( } a _ { t } ^ { p } \mid h _ { t } ^ { p } \rangle \dot { \bf ~ : } = p _ { \theta } ( a _ { t } ^ { p } \mid z , \dot { h } _ { t } ^ { p } )$ . The generative model is trained with a variational objective

$$
\mathcal { L } _ { \mathrm { g e n } } ( \theta , \phi ) = \mathbb { E } _ { \tau \sim \mathcal { D } } \biggl [ \mathbb { E } _ { z \sim q _ { \phi } ( z | \tau ) } \bigl [ \sum _ { t = 0 } ^ { T } \log p _ { \theta } ( a _ { t } ^ { p } \mid z , h _ { t } ^ { p } ) \bigr ] - \beta D _ { \mathrm { K L } } ( q _ { \phi } ( z \mid \tau ) \parallel p ( z ) ) \biggr ] ,\tag{1}
$$

where $p ( z )$ is typically chosen as a simple prior such as a standard Gaussian. After training, latentconditioned partner policies are instantiated by sampling z and using the learned decoder.

Partial-Observation Protocol. We follow the centralized training with decentralized execution (CTDE) paradigm for multi-agent reinforcement learning. To consider a more realistic partial observability setting, the policy is trained from PO rollouts collected by the agents, without using unobserved global state as policy input. During PPO training, logged partner-state information from these rollouts is used only as auxiliary supervision for the belief heads, and is not provided to the actor as input or used during execution. Thus, throughout decentralized execution, the policy is conditioned only on the agent’s own ego-centric observation history.

## 4 Proposed method

Our method, PIP, consists of two components for PO-ZSC. The Joint-view VAE improves localview partner-behavior modeling with richer training-time evidence. The Partner-state Belief policy maintains beliefs over the partner’s behavior and coarse location. Together, they address partner generation and policy learning under hidden partner states.

## 4.1 Joint-view VAE

Building on the generative latent-variable ZSC formulation, we consider the PO case where the behavior encoder receives only an agent-centered observation-action sequence. Under strict partial observability, an agent’s local view may omit key interaction cues, such as the partner’s location.

When these cues are missing, the same local trajectory can be consistent with multiple partner behaviors. This ambiguity makes it difficult for the encoder to infer a clear latent strategy from partial observations. As a result, distinct interaction modes can be mapped to nearby or overlapping regions in the latent space, making the learned decoder less effective as a behavior generator. To address this issue, we propose a Joint-view VAE (JV-VAE), which uses a teacher posterior from richer evidence to guide a student posterior conditioned on the agent-centered partial observation.

Let $i \in \{ 0 , 1 \}$ } denote the agent whose behavior is being encoded, and let $\bar { i } = 1 - i$ denote the other agent. The teacher is given a wider view of the interaction by combining the two agents’ local observations, while the student is kept under the same input constraint as the vanilla VAE. We define

$$
u _ { t } ^ { i } = \mathrm { U n i o n } ( o _ { t } ^ { i } , o _ { t } ^ { \bar { i } } ) , \qquad x _ { T , t } ^ { i } = \mathrm { C o n c a t } ( u _ { t } ^ { i } , o _ { t } ^ { i } ) , \qquad x _ { S , t } ^ { i } = o _ { t } ^ { i } .\tag{2}
$$

The teacher-side input $x _ { T , \ l } ^ { i }$ uses both the union view and $o _ { t } ^ { i } .$ , whereas the student-side input $\boldsymbol { x } _ { S , t } ^ { i }$ uses only $o _ { t } ^ { i }$ . The union view contains cells visible to either agent, while cells invisible to both agents remain masked. Thus, the teacher’s joint input remains partial rather than a full observation and does not use the unobserved global state.

This joint evidence helps the teacher posterior distinguish behavior modes that are ambiguous from the student input alone. Matching this posterior encourages the student to preserve meaningful diversity in the latent space while keeping the PO input interface unchanged.

For each branch $\bullet \in \{ T , S \}$ , let $\tau _ { \bullet } ^ { i } = \{ ( x _ { \bullet , t } ^ { i } , a _ { t } ^ { i } ) \} _ { t = 1 } ^ { H }$ denote the input-action trajectory used to infer the latent variable. The encoder is recurrent over this trajectory, so the latent is inferred from a temporal summary rather than from a single frame. We parameterize the posterior as

$$
q _ { \bullet } ( z \mid \tau _ { \bullet } ^ { i } ) = \mathcal { N } \big ( \mu _ { \bullet } ( \tau _ { \bullet } ^ { i } ) , \mathrm { d i a g } ( \sigma _ { \bullet } ^ { 2 } ( \tau _ { \bullet } ^ { i } ) ) \big ) .\tag{3}
$$

Although downstream target policy training does not use the encoder, the posterior still shapes how behavior trajectories are organized in the latent space during VAE training. In this sense, posterior distillation improves the learned behavior representation by encouraging trajectories with different interaction modes to be better separated in the latent space.

Given these branch-specific posteriors, training follows a two-stage distillation procedure. We first train the teacher as a VAE on the richer partial-view input. We then freeze the teacher and train the student with its own negative ELBO together with a posterior distillation term,

$$
{ \mathcal { L } } _ { S } = { \mathcal { L } } _ { S } ^ { \mathrm { V A E } } + \lambda _ { \mathrm { K D } } { \mathcal { L } } _ { \mathrm { K D } } , \qquad { \mathcal { L } } _ { \mathrm { K D } } = \mathbb { E } _ { i \in \{ 0 , 1 \} } \left[ \mathrm { K L } \bigl ( q _ { T } ( z \mid \tau _ { T } ^ { i } ) \parallel q _ { S } ( z \mid \tau _ { S } ^ { i } ) \bigr ) \right] .\tag{4}
$$

Here, $\mathcal { L } _ { S } ^ { \mathrm { V A E } }$ denotes the student’s negative ELBO.

In subsequent policy learning, we use the trained student decoder as a deployment-compatible partner generator for target-policy training.

## 4.2 Partner-State Belief Policy

Although the Joint-view VAE improves the generative partner model, it alone does not resolve the target agent’s decision-making problem under partial observability. At deployment, the agent must select ego actions from its ego-centric history, where the partner’s current location, intention, and action tendency may be hidden. While the recurrent actor can implicitly summarize the interaction history, we use a separately supervised belief state to explicitly retain partner-related information that may become unavailable in the current observation. We therefore augment the target policy with a Partner-state Belief module updated from the agent’s local observation history during execution.

The main design principle is to decouple the action state from the belief state used for partner-state estimation. Let $o _ { t }$ denote the ego-centric partial observation. The actor updates its action state $h _ { t } ^ { \mathrm { a c t } } = f _ { \mathrm { a c t } } ( e _ { t } , h _ { t - 1 } ^ { \mathrm { a c t } } )$ and uses it to select the ego action, where $e _ { t }$ is the encoded feature of $o _ { t }$ . In parallel, the belief module updates a separate belief state $h _ { t } ^ { \mathrm { b e l } } = f _ { \mathrm { b e l } } ( \tilde { e } _ { t } , h _ { t - 1 } ^ { \mathrm { b e l } } )$ , where $\tilde { e } _ { t }$ is obtained by applying the same observation encoder to a masked observation $\tilde { o } _ { t } .$ . Here, ${ \tilde { o } } _ { t }$ is constructed by masking the agent-orientation channel in $o _ { t }$ . This masking removes a direct orientation shortcut and encourages the belief module to infer partner state from the remaining partial observation history.

Behavioral Belief. The first auxiliary objective learns the Behavioral Belief over future partner actions. Motivated by theory-of-mind accounts [32, 36, 47] that distinguish a partner’s stable behavioral character from their situation-specific reaction, we separate Behavioral Belief into reaction and tendency components. We use two prediction heads, $\ell _ { t } ^ { \mathrm { r e c } } = \psi _ { \mathrm { r e c } } ( h _ { t } ^ { \mathrm { b e l } } )$ and $\ell _ { t } ^ { \mathrm { t e n } } = \psi _ { \mathrm { t e n } } ( h _ { t } ^ { \mathrm { b e l } } )$ where rec and ten denote reaction and tendency, respectively, and $\psi _ { \mathrm { r e c } }$ and $\psi _ { \mathrm { t e n } }$ map the belief state to behavior logits. Given the partner action sequence $a _ { t } ^ { p }$ , the corresponding soft targets are

$$
q _ { t } ^ { \mathrm { r e c } } = \mathcal { N } \left( \sum _ { k = 1 } ^ { K _ { \mathrm { o f f } } - 1 } \gamma _ { \mathrm { r e c } } ^ { k - 1 } \mathbf { 1 } [ a _ { t + k } ^ { p } ] \right) , \qquad q _ { t } ^ { \mathrm { t e n } } = \mathcal { N } \left( \sum _ { k = K _ { \mathrm { o f f } } } ^ { T - t } \gamma _ { \mathrm { t e n } } ^ { k - K _ { \mathrm { o f f } } } \mathbf { 1 } [ a _ { t + k } ^ { p } ] \right) .\tag{5}
$$

Here, $\mathcal { N } ( \cdot )$ normalizes a nonnegative vector to the probability simplex. The reaction target emphasizes short-term partner actions, whereas the tendency target captures more persistent behavioral patterns. Both heads are trained with soft cross-entropy losses $\mathcal { L } _ { \mathrm { r e c } }$ and $\mathcal { L } _ { \mathrm { t e n } }$

Location Belief. The second auxiliary objective learns the Location Belief. For each offset $\Delta \in \mathcal { D }$ where D includes the current offset $\Delta = 0$ and multiple future offsets, the location head predicts $\ell _ { t . \Delta } ^ { \mathrm { l o c } } = \psi _ { \mathrm { l o c } } ( h _ { t } ^ { \mathrm { b e l } } , e _ { \Delta } )$ , where $e _ { \Delta }$ is a learned offset embedding. The target $\tilde { b } _ { t , \Delta }$ is a soft distribution over coarse cells that reflects both the partner’s future position and the uncertainty caused by partial observability. We construct it by mixing two coarse-cell distributions, a Gaussian future-cell target $g _ { t , \Delta } ^ { \mathrm { c e l l } }$ and a reachability prior $g _ { t , \Delta } ^ { \mathrm { r e a c h } }$ from the last seen partner location, using a coefficient $\alpha \in [ 0 , 1 ]$

$$
\tilde { b } _ { t , \Delta } = \alpha g _ { t , \Delta } ^ { \mathrm { c e l l } } + ( 1 - \alpha ) g _ { t , \Delta } ^ { \mathrm { r e a c h } } .\tag{6}
$$

The future-cell target is centered at the true future partner cell and masked to physically reachable cells. The reachability prior assigns probability mass to coarse cells whose distance from the last seen partner cell is no larger than the elapsed time since the partner was last observed. If the partner has not been observed for more than $H _ { \mathrm { m a x } }$ steps, we replace the reachability prior with a uniform spatial prior. This mixture avoids treating the hidden partner location as fully observed when the partner has been occluded. The location head is trained with a normalized weighted KL loss, where $w _ { \Delta }$ controls the relative contribution of each offset,

$$
\mathcal { L } _ { \mathrm { l o c } } = \frac { \sum _ { \Delta } w _ { \Delta } \mathrm { K L } \Big ( \mathrm { s o f t m a x } ( \ell _ { t , \Delta } ^ { \mathrm { l o c } } ) \| \tilde { b } _ { t , \Delta } + \epsilon \Big ) } { \sum _ { \Delta } w _ { \Delta } } .\tag{7}
$$

Within each offset, samples in which the partner is unobserved receive larger weights, since Location Belief is most useful under partial observability.

Table 1: Evaluation with 12 held-out behavior-preference partners unseen during training. The best result in each layout is shown in bold. Mean ± standard error.
<table><tr><td>Method</td><td>Open</td><td>Hallway</td><td>FC</td><td>Ring</td></tr><tr><td>FCP</td><td> $1 4 1 . 6 \pm 9 . 6$ </td><td> $5 7 . 1 \pm 2 . 7$ </td><td> $2 1 . 9 \pm 6 . 0$ </td><td> $4 8 . 3 \pm 6 . 1$ </td></tr><tr><td>MEP</td><td> $1 1 5 . 9 \pm 1 0 . 3$ </td><td> $5 8 . 2 \pm 2 . 6 $ </td><td> $1 7 . 6 \pm 5 . 8$ </td><td> $5 1 . 1 \pm 5 . 4$ </td></tr><tr><td>E3T</td><td> $5 5 . 3 \pm { 3 . 6 }$ </td><td> $4 4 . 5 \pm 5 . 7$ </td><td> $2 1 . 0 \pm 4 . 1$ </td><td> $2 7 . 5 \pm 3 . 1$ </td></tr><tr><td>ERS</td><td> $1 0 1 . 0 \pm 7 . 8$ </td><td> $3 6 . 6 \pm 4 . 5$ </td><td> $2 4 . 0 \pm 4 . 3$ </td><td> $4 8 . 9 \pm 5 . 5$ </td></tr><tr><td>GAMMA</td><td> $1 6 3 . 3 \pm 1 0 . 6$ </td><td> $5 7 . 7 \pm 3 . 3$ </td><td> $1 9 . 8 \pm 5 . 3$ </td><td> $4 8 . 5 \pm 5 . 6$ </td></tr><tr><td>GOAT</td><td> $1 6 5 . 5 \pm 1 0 . 6$ </td><td> $6 1 . 5 \pm 5 . 1$ </td><td> $1 9 . 0 \pm 5 . 3$ </td><td> $7 7 . 3 \pm 5 . 9$ </td></tr><tr><td>PIP (Ours)</td><td> ${ \bf 2 4 9 . 0 \pm 1 2 . 2 }$ </td><td> ${ \bf 7 8 . 1 \pm 5 . 3 }$ </td><td> ${ \bf 2 5 . 4 \pm 8 . 9 }$ </td><td> ${ \bf 1 2 5 . 8 \pm 5 . 6 }$ </td></tr></table>

The Partner-state Belief is then passed to the actor through a stable additive interface. We first project the belief state as $r _ { t } = \dot { \phi } _ { \mathrm { b e l } } ( \mathrm { s t o p g r a d } ( h _ { t } ^ { \mathrm { b e l } } ) )$ ), where $\phi _ { \mathrm { b e l } }$ is a projection layer. The stopgradient operation prevents PPO gradients from directly reshaping the belief state and keeps the belief representation primarily governed by the Behavioral Belief and Location Belief objectives. We also compute a normalized uncertainty feature from the entropy of the current Location Belief, $u _ { t } = H ( \mathrm { s o f t m a x } ( \ell _ { t , 0 } ^ { \mathrm { l o c } } ) ) / \log N _ { \mathrm { c e l l } }$ , and define the actor-side belief message as $m _ { t } ^ { \mathrm { b e l } } = [ r _ { t } , u _ { t } ]$

The actor receives this belief message as an additive update to its action state, $\Delta h _ { t } ^ { \mathrm { a c t } } = W _ { \mathrm { b e l } } m _ { t } ^ { \mathrm { b e l } } +$ $c _ { \mathrm { b e l } } .$ , where $W _ { \mathrm { b e l } }$ and $c _ { \mathrm { b e l } }$ denote the linear projection and bias of the belief-injection layer, respectively. The actor then uses $\bar { h } _ { t } ^ { \mathrm { a c t } } = h _ { t } ^ { \mathrm { a c t } } + \Delta \bar { h } _ { t } ^ { \mathrm { a c t } }$ for action selection. The parameters $W _ { \mathrm { b e l } }$ and $c _ { \mathrm { b e l } }$ are initialized to zero, so the initial policy is identical to the recurrent baseline.

We train the belief-augmented actor with PPO from environment rewards and supervise the belief module with training-only partner-state targets. We add a lightweight alignment regularizer $\mathcal { L } _ { \mathrm { a u x } }$ on the belief-to-actor interface, blocking gradients to the belief RNN. The combined loss is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t r a i n } } = \mathcal { L } _ { \mathrm { P P O } } \big ( \pi ( \cdot \mid \bar { h } _ { t } ^ { \mathrm { a c t } } ) \big ) + \lambda _ { \mathrm { b e h a v i o r } } ( \alpha _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } + \alpha _ { \mathrm { t e n } } \mathcal { L } _ { \mathrm { t e n } } ) + \lambda _ { \mathrm { l o c } } \mathcal { L } _ { \mathrm { l o c } } + \lambda _ { \mathrm { a u x } } \mathcal { L } _ { \mathrm { a u x } } . } \end{array}\tag{8}
$$

Since the injected belief message is detached from the belief state, PPO gradients do not directly update the belief RNN. The belief RNN is trained by the Behavioral Belief and Location Belief losses, while PPO learns how to use the injected belief message for reward maximization.

## 5 Experiments

We independently instantiate and train all methods for every benchmark environment, including each layout. For every environment, we separately train the partner population, construct the trajectory dataset, train the generative partner model, and optimize the target policy. No policy, VAE, or partner population is transferred across environments. Across all benchmarks, deployed policies and evaluation partners operate only from local observations. The joint-view teacher is used solely during VAE training and uses the union of both agents’ local observations, while the distilled student and all deployed policies remain restricted to local observations.

We compare PIP with the generative-partner baselines GAMMA [26] and GOAT [6] in all three benchmarks. For Burrito and Overcooked, we restrict fully observable environments to egocentric local views. Burrito-PO additionally includes FCP [43], MEP [50], E3T [48], and ERS [31]. Performance is measured using the task reward obtained with partners excluded from target-policy training. Environment-specific architectures, hyperparameters, partner construction procedures, and additional implementation details are provided in the appendices.

## 5.1 Burrito-PO

Burrito-PO is our primary benchmark for zero-shot coordination under ego-centric partial observability. It adapts the cooperative Burrito environment [23] by restricting each agent to a $7 \times 7$ window centered on itself and masking cells outside this window. We consider four layouts, Open, Hallway, Forced-Coord (FC), and Ring. Following prior work [23, 45], we construct 24 candidate partners per layout and select 12 behaviorally diverse partners for evaluation. For each layout, every selected partner is evaluated over 10 episodes. We report mean performance and standard error.

Table 2: Performance in two additional coordination environments, Overcooked-PO and Melting Pot, respectively.  
Table 3: Ablation study in the Open layout. Behavioral and Location denote the Behavioral Belief and Location Belief modules, respectively.
<table><tr><td>Method</td><td>Overcooked-PO</td><td>Melting Pot</td><td></td><td>JV-VAE</td><td>Behavioral</td><td>Location</td><td>Reward</td></tr><tr><td>GAMMA</td><td> $3 4 9 . 6 \pm 5 . 1 4$ </td><td> $1 7 1 . 3 \pm 5 . 2$ </td><td>(a)</td><td></td><td></td><td></td><td> $1 6 3 . 3 \pm 1 0 . 6$ </td></tr><tr><td>GOAT</td><td> $3 2 5 . 9 \pm 5 . 7 3$ </td><td> $1 5 3 . 1 \pm 5 . 2$ </td><td>(b)</td><td>√</td><td></td><td></td><td> $1 8 1 . 6 \pm 9 . 1$ </td></tr><tr><td>PIP (Ours)</td><td> $\mathbf { 4 0 2 . 3 4 } \pm 3 . 8 2$ </td><td> $2 1 3 . 5 \pm 1 1 . 2$ </td><td>(c)</td><td>√</td><td>√</td><td></td><td> $2 1 1 . 0 \pm 8 . 5$ </td></tr><tr><td></td><td></td><td></td><td>(d)</td><td>√</td><td></td><td>√</td><td> $2 1 3 . 2 \pm 5 . 2$ </td></tr><tr><td></td><td></td><td></td><td>(e)</td><td></td><td>√</td><td>√</td><td> $1 8 3 . 1 \pm 1 2 . 3$ </td></tr><tr><td></td><td></td><td></td><td>(f)</td><td>√</td><td>√</td><td>√</td><td> $2 4 9 . 0 \pm 1 2 . 2$ </td></tr></table>

Table 1 reports ZSC performance against held-out partners under partial observability. PIP achieves the best performance across all four layouts. The relative improvement is particularly large on Ring, where the movable region forms a narrow loop-like route. The partner’s future movement is strongly constrained by this structure, allowing the belief module to anticipate the partner’s hidden position and movement direction more reliably from local observations. This result suggests that Partner-state Belief is particularly beneficial when the partner is frequently outside the ego agent’s field of view and the policy must infer hidden partner states to anticipate coordination opportunities.

## 5.2 Overcooked-PO

Overcooked-PO is a local-observation variant of Overcooked [4] that retains the original transition dynamics, six primitive actions, task objective, and reward structure. We evaluate on CounterCircuit using a 5 × 5 ego-centered local observation. Consequently, the partner and its current subtask may become unobservable when the agents move across different regions of the kitchen. All methods are evaluated with the same 12 behavior-preference partners.

As shown in Tab. 2, PIP achieves the best performance among the compared methods on CounterCircuit. This result suggests that PIP remains effective in another 5 × 5 grid-based cooking environment under local observations.

## 5.3 Melting Pot

We further evaluate PIP in Melting Pot’s [1] coop\_mining substrate, instantiated as a two-player configuration. This benchmark differs from the cooking environments through its native $8 \bar { 8 } \times 8 8$ ego-centric RGB observation, eight primitive actions, and cooperative mining objective. Each target policy is evaluated with three official bot partners that are excluded from target-policy training.

As shown in Tab. 2, PIP achieves the best performance among the compared methods in Melting Pot. Together with the results on Burrito-PO and Overcooked-PO, this finding suggests that PIP remains effective across observation representations ranging from structured local grids to native ego-centric RGB inputs, as well as with primitive control and a cooperative objective distinct from the cooking tasks. However, the use of official bots precludes direct cross-benchmark comparisons.

## 5.4 Human Evaluation

We recruit 66 participants and ask each participant to interact with three anonymized partner agents corresponding to GAMMA, GOAT, and PIP in Burrito-PO. The agent order is randomized across participants. After each interaction, participants rate coordination quality via a survey. After all three interactions, they rank each agent by ease of cooperation.

As shown in Fig. 2 (a) and (b), participants attain the highest sparse reward with PIP and rank PIP first most often among the three agents. These results indicate that PIP improves task performance while also being preferred as a coordination partner. Figure 2 (c) visualizes the survey results across the six categories. PIP receives the highest score in every category, suggesting that participants perceived PIP as more effective, predictable, adaptable, and desirable to cooperate with under partial observability. Additional analyses are provided in the appendices.

![](images/2bde56afd4e4d83e57e21bec26050d644f617b2e415144bd036b9224ad019859.jpg)  
(a) Reward

![](images/45aec0b463c412138381f191f5ddf07f441c18488053af2c720e91ec5c7bc5cf.jpg)  
(b) Rank distribution

![](images/11d0c3189e3f5a4baa233b1940f537c8697b107a4ab016397a5be6c7649f0d9a.jpg)

Figure 2: Human evaluation in the Burrito-PO environment. Participants interacted with GAMMA, GOAT, and our method. Results report (a) sparse reward, (b) cooperation preference ranking, and (c) post-interaction survey scores on coordination quality.  
![](images/2498d3ba3c609b7d9775231d7343369fdc86ac48287699eb43e9e9d31c4649a8.jpg)  
Figure 3: Policy classification results.

![](images/dd361960ec35d128dfc47294b9ad01ac1aa0c85ebfee9c7091dc430c7ff61496.jpg)

![](images/69219bfcd37c51817652654252dcc4c3e8e1c794edec8ba76deaca778a241b46.jpg)  
Figure 4: Aggregate posterior results.  
(b) Distribution-level

## 6 Additional Analysis

## 6.1 Component Ablation

We conduct an ablation study to isolate the contribution of each component in our framework, as summarized in Tab. 3. Row (a) corresponds to the GAMMA baseline, whose VAE is trained only on partial-observation trajectories. Comparing rows (a) and (b), replacing this VAE with our student VAE distilled from the joint-view teacher VAE improves performance, suggesting that distillation mitigates representation degradation under partial observability and provides more useful latent partner representations for generated-partner training. A similar improvement is observed from rows (e) to (f), further supporting the benefit of the distilled VAE when Partner-state Belief is used.

Rows (c) and (d) add Behavioral Belief and Location Belief, respectively, on top of the distilled VAE, and both improve over row (b). This shows that each belief component contributes to ZSC by estimating partner-related information that may be missing from the coordinator’s local observation. Moreover, row (e) improves over row (a) even without the distilled VAE, indicating that Partner-state Belief is not only effective under our distilled-VAE training setup but also provides an independent benefit. Finally, row (f) achieves the best performance by combining the distilled VAE and Partnerstate Belief. These results show that the components play complementary roles. The distilled VAE improves generated-partner training, while the two belief components provide distinct partner-state estimates, leading to the strongest zero-shot coordination performance when used together. Additional ablations controlling for the increased parameter count and examining the effect of the supervision loss are provided in the appendices.

## 6.2 Joint-view VAE Representation Analysis

During ZSC inference, the student receives only partner-side inputs, while the teacher is unavailable. Therefore, the student’s latent z must satisfy two requirements simultaneously. It must be informative enough to preserve partner identity, and it must remain in-distribution under the prior $p ( z )$ from which test-time samples are drawn. The Fig. 3 evaluates the first requirement using four standard clustering metrics computed on encoder latents, namely Acc [12], NMI [3, 7, 42], ARI [16, 24], and the silhouette coefficient [39, 49]. The distilled student VAE consistently outperforms both the fixed-σ and heteroscedastic baselines, with especially large gains on NMI and ARI. The Fig. 4 evaluates coverage [29, 44] using log $q _ { \mathrm { a g g } } ( z )$ on prior samples $z \sim p ( z )$ and the squared MMD [11, 51] between $q _ { \mathrm { a g g } }$ and $p ( z )$ . PIP achieves both the highest coverage and the lowest $\mathbf { M M D ^ { 2 } }$ These results show that joint-view distillation produces a partner-discriminative latent representation while reducing prior holes that would place test-time samples outside the decoder’s support. Additional metric details and t-SNE visualizations are provided in the appendices.

![](images/1bfac0bac0adcbc1cdf8787d93d9fd727820845970319fd5964a1042b0c00b08.jpg)  
Figure 5: Action-distribution difference between π and $\pi _ { \mathrm { l e s i o n } }$ across TSLS buckets.

![](images/c992768e7eef9b417e1f5350e9f56a938d56f85a83dd6b354bf14ef434b6f3fc.jpg)  
Figure 6: Episode sparse reward for PIP and its belief-lesion policy across partner occlusion.

## 6.3 Partner-state Belief under Occlusion

Effect of Partner-state Belief on Policy Actions. We test whether the belief message $\Delta h _ { t } ^ { \mathrm { a c t } }$ changes policy actions by comparing π<sub>PIP</sub> with a lesion policy that zeros the message while preserving the observation and recurrent state. Figure 5 shows clear differences in their action distributions across various Time Since Last Seen (TSLS) buckets, the number of steps since the partner left the visible range. The difference is small at TSLS 0, increases after the partner leaves the field of view, and decreases at large TSLS as the hidden state becomes less informative. This suggests that Partner-state Belief most strongly affects actions when inferred partner information remains useful.

Coordination Benefit under Occlusion. We analyze how Partner-state Belief benefits coordination across different partner occlusion levels. We sort the held-out BP policies by the fraction of time their partner is outside the ego agent’s field of view when paired with PIP and divide them into four equally sized groups. As shown in Fig. 6, PIP consistently outperforms the lesion policy that zero-pads the belief message. The gap is largest for the group whose partner is occluded most frequently. This result shows that Partner-state Belief improves zero-shot coordination most when partner information cannot be obtained directly from the current observation. Additional analyses of Partner-state Belief are provided in the appendices.

## 7 Conclusion

We study generative ZSC under local-view partial observability and propose PIP, which combines a Joint-view VAE for deployment-compatible partner representations with Partner-state Belief networks for hidden-state inference. PIP achieves the highest mean performance across Burrito-PO, Overcooked-PO, and a Melting Pot substrate, with human evaluation and diagnostic analyses further supporting its design. These results demonstrate the importance of partner representations and belief-aware decision making for ZSC under spatially restricted observations.

## References

[1] J. P. Agapiou, A. S. Vezhnevets, E. A. Duéñez-Guzmán, J. Matyas, Y. Mao, P. Sunehag, R. Köster, U. Madhushani, K. Kopparapu, R. Comanescu, et al. Melting pot 2.0. arXiv preprint arXiv:2211.13746, 2022.

[2] N. Bard, J. N. Foerster, S. Chandar, N. Burch, M. Lanctot, H. F. Song, E. Parisotto, V. Dumoulin, S. Moitra, E. Hughes, I. Dunning, S. Mourad, H. Larochelle, M. G. Bellemare, and M. Bowling. The hanabi challenge: A new frontier for ai research. Artificial Intelligence, 280:103216, 2020. doi: 10.1016/j.artint.2019.103216.

[3] M. Caron, I. Misra, J. Mairal, P. Goyal, P. Bojanowski, and A. Joulin. Unsupervised learning of visual features by contrasting cluster assignments. Advances in neural information processing systems, 33: 9912–9924, 2020.

[4] M. Carroll, R. Shah, M. K. Ho, T. Griffiths, S. Seshia, P. Abbeel, and A. Dragan. On the utility of learning about humans for human-ai coordination. Advances in neural information processing systems, 32, 2019.

[5] R. Charakorn, P. Manoonpong, and N. Dilokthanakul. Generating diverse cooperative agents by learning incompatible policies. In The Eleventh International Conference on Learning Representations, 2023.

[6] P. R. Chaudhary, Y. Liang, D. Chen, S. S. Du, and N. Jaques. Improving human-AI coordination through online adversarial training and generative models. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=AeehNfbHqD.

[7] Z. Dang, C. Deng, X. Yang, K. Wei, and H. Huang. Nearest neighbor matching for deep clustering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13693–13702, 2021.

[8] J. Foerster, F. Song, E. Hughes, N. Burch, I. Dunning, S. Whiteson, M. Botvinick, and M. Bowling. Bayesian action decoder for deep multi-agent reinforcement learning. In International Conference on Machine Learning, pages 1942–1951. PMLR, 2019.

[9] M. Friedman. The use of ranks to avoid the assumption of normality implicit in the analysis of variance. Journal ofthe american statistical association, 32(200):675–701, 1937.

[10] T. Gessler, T. Dizdarevic, A. Calinescu, B. Ellis, A. Lupu, and J. N. Foerster. Overcookedv2: Rethinking overcooked for zero-shot coordination. arXiv preprint arXiv:2503.17821, 2025.

[11] A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola. A kernel two-sample test. The journal ofmachine learning research, 13(1):723–773, 2012.

[12] A. Grover, M. Al-Shedivat, J. Gupta, Y. Burda, and H. Edwards. Learning policy representations in multiagent systems. In International conference on machine learning, pages 1802–1811. PMLR, 2018.

[13] P. Gu, M. Zhao, J. Hao, and B. An. Online ad hoc teamwork under partial observability. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id= 18Ys0-PzyPI.

[14] S. Holm. A simple sequentially rejective multiple test procedure. Scandinavian journal of statistics, pages 65–70, 1979.

[15] H. Hu, A. Lerer, A. Peysakhovich, and J. Foerster. “other-play” for zero-shot coordination. In International conference on machine learning, pages 4399–4410. PMLR, 2020.

[16] L. Hubert and P. Arabie. Comparing partitions. Journal of classification, 2(1):193–218, 1985.

[17] M. Igl, L. Zintgraf, T. A. Le, F. Wood, and S. Whiteson. Deep variational reinforcement learning for pomdps. In International conference on machine learning, pages 2117–2126. PMLR, 2018.

[18] K. Jha, W. Carvalho, Y. Liang, S. S. Du, M. Kleiman-Weiner, and N. Jaques. Cross-environment cooperation enables zero-shot multi-agent coordination. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=zBBYsVGKuB.

[19] L. P. Kaelbling, M. L. Littman, and A. R. Cassandra. Planning and acting in partially observable stochastic domains. Artificial intelligence, 101(1-2):99–134, 1998.

[20] D. S. Kerby. The simple difference formula: An approach to teaching nonparametric correlation. Comprehensive Psychology, 3:11–IT, 2014.

[21] J. Kolb and K. M. Feigh. Inferring belief states in partially-observable human-robot teams. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7115–7122. IEEE, 2024.

[22] J. Z. Leibo, E. A. Dueñez-Guzman, A. Vezhnevets, J. P. Agapiou, P. Sunehag, R. Koster, J. Matyas, C. Beattie, I. Mordatch, and T. Graepel. Scalable evaluation of multi-agent reinforcement learning with melting pot. In Proceedings ofthe 38th International Conference on Machine Learning, pages 6187–6199. PMLR, 2021.

[23] B. Li, S. Shi, L. Romero, H. Li, Y. Xie, W. Kim, S. Nikolaidis, M. Lewis, K. Sycara, and S. Stepputtis. Adaptively coordinating with novel partners via learned latent strategies. arXiv preprint arXiv:2511.12754, 2025.

[24] Y. Li, P. Hu, Z. Liu, D. Peng, J. T. Zhou, and X. Peng. Contrastive clustering. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 8547–8555, 2021.

[25] Y. Li, S. Zhang, J. Sun, Y. Du, Y. Wen, X. Wang, and W. Pan. Cooperative open-ended learning framework for zero-shot coordination. In International Conference on Machine Learning, pages 20470–20484. PMLR, 2023.

[26] Y. Liang, D. Chen, A. Gupta, S. S. Du, and N. Jaques. Learning to cooperate with humans using generative agents. Advances in Neural Information Processing Systems, 37:60061–60087, 2024.

[27] K. Lucas and R. E. Allen. Any-play: An intrinsic augmentation for zero-shot coordination. In Proceedings ofthe 21st International Conference on Autonomous Agents and Multiagent Systems, pages 853–861, 2022.

[28] A. Lupu, B. Cui, H. Hu, and J. Foerster. Trajectory diversity for zero-shot coordination. In International conference on machine learning, pages 7204–7213. PMLR, 2021.

[29] A. Makhzani, J. Shlens, N. Jaitly, I. Goodfellow, and B. Frey. Adversarial autoencoders. arXiv preprint arXiv:1511.05644, 2015.

[30] D. Muglich, L. M. Zintgraf, C. A. S. De Witt, S. Whiteson, and J. Foerster. Generalized beliefs for cooperative ai. In International Conference on Machine Learning, pages 16062–16082. PMLR, 2022.

[31] D. Muglich, J. Forkel, E. van der Pol, and J. N. Foerster. Expected return symmetries. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=wFg0shwoRe.

[32] D. Nguyen, P. Nguyen, H. Le, K. Do, S. Venkatesh, and T. Tran. Learning theory of mind via dynamic traits attribution. In Proceedings ofthe 21st International Conference on Autonomous Agents and Multiagent Systems, AAMAS ’22, page 954–962, Richland, SC, 2022. International Foundation for Autonomous Agents and Multiagent Systems. ISBN 9781450392136.

[33] D. Nguyen, H. Le, K. Do, S. Venkatesh, and T. Tran. Social motivation for modelling other agents under partial observability in decentralised training. In IJCAI, pages 4082–4090, 2023.

[34] S. Omidshafiei, J. Pazis, C. Amato, J. P. How, and J. Vian. Deep decentralized multi-task multi-agent reinforcement learning under partial observability. In International conference on machine learning, pages 2681–2690. PMLR, 2017.

[35] S. C. Ong, S. W. Png, D. Hsu, and W. S. Lee. Planning under uncertainty for robotic tasks with mixed observability. The International Journal of Robotics Research, 29(8):1053–1068, 2010.

[36] N. Rabinowitz, F. Perbet, F. Song, C. Zhang, S. A. Eslami, and M. Botvinick. Machine theory of mind. In International conference on machine learning, pages 4218–4227. PMLR, 2018.

[37] A. Rahman, I. Carlucho, N. Höpner, and S. V. Albrecht. A general learning framework for open ad hoc teamwork using graph-based policy learning. Journal ofMachine Learning Research, 24(298):1–74, 2023.

[38] A. Rahman, E. Fosong, I. Carlucho, and S. V. Albrecht. Generating teammates for training robust ad hoc teamwork agents via best-response diversity. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=l5BzfQhROl.

[39] P. J. Rousseeuw. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal ofcomputational and applied mathematics, 20:53–65, 1987.

[40] B. Sarkar, A. Shih, and D. Sadigh. Diverse conventions for human-AI collaboration. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum? id=MljeRycu9s.

[41] P. Stone, G. Kaminka, S. Kraus, and J. Rosenschein. Ad hoc autonomous agent teams: Collaboration without pre-coordination. In Proceedings of the AAAI conference on artificial intelligence, volume 24, pages 1504–1509, 2010.

[42] A. Strehl and J. Ghosh. Cluster ensembles—a knowledge reuse framework for combining multiple partitions. Journal ofmachine learning research, 3(Dec):583–617, 2002.

[43] D. Strouse, K. McKee, M. Botvinick, E. Hughes, and R. Everett. Collaborating with humans without human data. Advances in neural information processing systems, 34:14502–14515, 2021.

[44] J. Tomczak and M. Welling. Vae with a vampprior. In International conference on artificial intelligence and statistics, pages 1214–1223. PMLR, 2018.

[45] X. Wang, S. Zhang, W. Zhang, W. Dong, J. Chen, Y. Wen, and W. Zhang. Zsc-eval: An evaluation toolkit and benchmark for multi-agent zero-shot coordination. Advances in Neural Information Processing Systems, 37:47344–47377, 2024.

[46] F. Wilcoxon. Individual comparisons by ranking methods. In Breakthroughs in statistics: Methodology and distribution, pages 196–202. Springer, 1992.

[47] S. A. Wu, R. E. Wang, J. A. Evans, J. B. Tenenbaum, D. C. Parkes, and M. Kleiman-Weiner. Too many cooks: bayesian inference for coordinating multi-agent collaboration. Topics in Cognitive Science, 13(2): 414–432, 2021.

[48] X. Yan, J. Guo, X. Lou, J. Wang, H. Zhang, and Y. Du. An efficient end-to-end training approach for zero-shot human-ai coordination. Advances in neural information processing systems, 36:2636–2658, 2023.

[49] B. Yang, X. Fu, N. D. Sidiropoulos, and M. Hong. Towards k-means-friendly spaces: Simultaneous deep learning and clustering. In international conference on machine learning, pages 3861–3870. PMLR, 2017.

[50] R. Zhao, J. Song, Y. Yuan, H. Hu, Y. Gao, Y. Wu, Z. Sun, and W. Yang. Maximum entropy populationbased training for zero-shot human-ai coordination. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 6145–6153, 2023.

[51] S. Zhao, J. Song, and S. Ermon. Infovae: Balancing learning and inference in variational autoencoders. In Proceedings ofthe aaai conference on artificial intelligence, volume 33, pages 5885–5892, 2019.

[52] L. Zintgraf, K. Shiarlis, M. Igl, S. Schulze, Y. Gal, K. Hofmann, and S. Whiteson. Varibad: A very good method for bayes-adaptive deep rl via meta-learning. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=Hkl9JlBYvr.

<table><tr><td>Preference</td><td>Reward Shape</td></tr><tr><td>Plating Ingredients Washing Plates Delivering Dishes</td><td>-30, 20 -30, 20 -30, 20</td></tr><tr><td>Chopping Ingredients Potting Rice Grilling Meat/Mushroom</td><td>-30, 20 -30, 20</td></tr><tr><td>Taking Mushroom From Dispenser</td><td>-30, 20 -15, 10</td></tr><tr><td>Taking Rice From Dispenser Taking Meat From Dispenser</td><td>-15, 10 -15, 10</td></tr></table>

Table B.1: Event-based BP features and corresponding reward design in Burrito-PO.

## A Limitations

Although the deployed policy uses only local observations, the belief module is trained with auxiliary supervision for future partner actions and partner positions. Such labels are available in simulated training environments but may be difficult or costly to obtain in real-world decentralized settings. Future work should therefore investigate weakly supervised or self-supervised belief learning methods that reduce reliance on explicit partner-state labels.

## B Evaluation Protocol

## B.1 Construction of BP Agent Candidates

According to the ZSC-Eval [45] framework, a well-designed population of behavior preference (BP) agents, constructed with event-based reward functions, can provide behaviorally diverse partners for zero-shot coordination evaluation. Following this idea, we construct, for each domain, a candidate pool of held-out BP agents whose sole purpose is to supply diverse evaluation partners that are never seen during target policy training.

Burrito-PO. We follow the partner construction protocol used in TALENTS [23] and define nine tunable behavior preferences, grouped into three categories of three binary preference dimensions each. Each dimension assigns either an encouraged (a positive weight) or a discouraged (a negative weight) preference to one behavior, and all remaining shaping terms are held fixed and shared across the pool. Enumerating all binary combinations within one category yields 2<sup>3</sup> = 8 agents, and the three categories together give 24 BP candidate agents per layout. The preference categories, shaping terms, and corresponding weights are summarized in Tab. B.1.

Overcooked-PO. Overcooked is the domain for which ZSC-Eval originally defines its event set, so we adopt that specification directly rather than designing our own. A BP agent is parameterized by a weight vector over the canonical shaped-event counters, together with one additional entry scaling the sparse task reward. A small number of entries are allowed to take encouraging or discouraging values while the rest are held fixed, and we admit only vectors expressing at most three non-zero preferences, so that each agent exhibits a few interpretable tendencies rather than an arbitrary mixture. From the resulting set of admissible weight vectors we draw 60 to form the BP candidate pool for this domain. The preference dimensions and their value sets are summarized in Tab. B.2.

BP Training phase. In both domains, to train the BP agents themselves, the corresponding preference terms are added as auxiliary shaping rewards on top of the sparse environmental reward. Each BP agent is trained together with its corresponding best-response agent, producing partner policies with distinct behavioral tendencies. Once trained, all BP agents are frozen and are used exclusively for selecting evaluation partners and evaluating the target policy. No BP agent, checkpoint, or reward weight is exposed to any stage of target policy training.

<table><tr><td>Events</td><td>Weights</td></tr><tr><td>Put an onion / dish / soup / tomato onto the counter</td><td>0</td></tr><tr><td>Pickup an onion / dish / soup / tomato from the counter</td><td>0</td></tr><tr><td>Pickup an onion from the onion dispenser Pickup an tomato from the tomato dispenser Pickup a dish from the dish dispenser Pickup a soup Place an ingredient into the pot</td><td>-20,0,10 -20,0,10 -20,0,10 -20,0,5,10 -20,0,3,10</td></tr></table>

Table B.2: Event-based BP features and corresponding reward design in Overcooked-PO

## B.2 Selection of Evaluation BP Agents

From the BP candidate agents, we select 12 evaluation partners for each layout following the BRdiversity-based selection procedure of ZSC-Eval [45]. The goal of this selection step is to obtain a compact evaluation set that preserves behavioral diversity among the candidate partners.

For each candidate BP agent, let $\pi _ { i } ^ { \mathrm { B P } }$ denote the candidate partner and $\pi _ { i } ^ { \mathrm { B R } }$ denote its corresponding best-response agent, where $i \in \mathsf { \Gamma } \{ 1 , \ldots , N \}$ . Here, $N \stackrel { - } { = } 2 4$ for Burrito-PO and $N \mathrm { ~ = ~ } 6 0$ for Overcooked-PO. We roll out each pair and summarize the behavior of $\pi _ { i } ^ { \mathrm { B R } }$ using event-based statistics. Let E be the set of environment events, and let $c _ { i , e } ^ { ( r ) }$ denote the number of times event $e \in { \mathcal { E } }$ occurs for the corresponding best-response agent in rollout r. The behavior feature of candidate i is defined as the average event-count vector

$$
\phi _ { i } ( e ) = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } c _ { i , e } ^ { ( r ) } , \qquad e \in \mathcal { E } ,
$$

where R is the number of rollouts used for estimating the event statistics and set to 5.

We then normalize each event dimension across the candidate pool:

$$
\tilde { \phi } _ { i } ( e ) = \frac { \phi _ { i } ( e ) } { \operatorname* { m a x } _ { j \in \{ 1 , \dots , N \} } \phi _ { j } ( e ) + \epsilon } ,
$$

where ϵ is a small constant for numerical stability. Stacking the normalized feature vectors gives the matrix

$$
\tilde { \Phi } = \left[ \begin{array} { c } { { \tilde { \phi } _ { 1 } ^ { \top } } } \\ { { \tilde { \phi } _ { 2 } ^ { \top } } } \\ { { \vdots } } \\ { { \tilde { \phi } _ { N } ^ { \top } } } \end{array} \right] .
$$

We compute the behavioral similarity matrix as

$$
K = \tilde { \Phi } \tilde { \Phi } ^ { \top } .
$$

Finally, we select a subset $S ^ { * }$ of $M = 1 2$ evaluation partners by maximizing the log-determinant of the corresponding principal submatrix:

$$
S ^ { * } = \arg \operatorname* { m a x } _ { S \subset \{ 1 , \ldots , N \} , \vert S \vert = M } \log \operatorname* { d e t } K _ { S } ,
$$

where $K _ { S }$ denotes the principal submatrix of K indexed by S. Since a larger determinant indicates that the selected behavior feature vectors span a more diverse set of directions, this criterion favors evaluation partners associated with diverse best-response behaviors. The BP agents corresponding to the selected indices $S ^ { * }$ are fixed as the final held-out evaluation partners for the layout.

![](images/dd7d520cde995fa54f9b37d5148b68b164e5705ae1e9e7156158d9443c7152c2.jpg)  
Figure C.1: Partner-visibility ratio (%) by view size and layout, measured from Self-Play rollouts using the MEP algorithm.

## B.3 Zero-shot Evaluation Protocol

Burrito-PO and Overcooked-PO. We evaluate the zero-shot coordination performance of the trained coordinator by pairing it with the selected held-out BP partners. The evaluation partner set consists of 12 BP agents selected from 24 candidates for Burrito-PO and 60 candidates for Overcooked-PO for each layout. For each selected BP partner, we run the coordinator-partner pair under both role assignments by swapping their spawn positions. Thus, each partner induces two ordered evaluation pairings.

For each ordered pairing, we collect five episodes, resulting in $1 2 \times 2 \times 5 = 1 2 0$ episodes per evaluation. Equivalently, the coordinator is evaluated for 10 episodes with each selected BP partner, matching the evaluation protocol described in the main paper. All main results and analyses are reported using this selected evaluation set.

We include role-swapped evaluation because partial observability and ego-centric inputs make the two agent roles potentially asymmetric. Although the sparse reward is shared between agents, the policy input representation and dense shaping signals are agent-specific during training. Evaluating the coordinator under both role assignments therefore provides a more complete estimate of its zero-shot coordination performance.

Melting Pot. Since Melting Pot provides official bot partners rather than BP agents, we follow a separate protocol for this benchmark. The target policy is paired with each of the three official bots (cooperator, defector, and mixed) on the original $2 { \bar { 7 } } \times 2 { \bar { 7 } }$ coop\_mining, and we collect 100 episodes per bot, resulting in 300 evaluation episodes in total. In each episode, we measure the episodic reward obtained by the target policy, and we report its mean and standard error.

## C Implementation Details

## C.1 Window Size Selection

Zero-shot coordination (ZSC) inherently requires agents to collaborate with a partner. When the partner is almost never observable, the task effectively degenerates into single-agent play, eliminating meaningful coordination signals. Conversely, when the partner is almost always visible, the setting becomes nearly indistinguishable from full observability. This makes it difficult to evaluate zero-shot coordination performance under partial observability. To determine an appropriate PO window size for Burrito-PO, we first empirically measured the partner-visibility ratio for different view sizes across all Burrito-PO layouts. Specifically, we performed self-play rollouts using SP agents trained in Stage 1 of MEP, running 5 episodes per policy (40 episodes per layout in total), and recorded partner visibility at each timestep for each view size. The results are summarized in Fig. C.1.

For view size $K = 3 \times 3$ , the visibility ratio in the Forced-Coord (FC) layout is $0 \%$ , making it overly restrictive as a coordination task. While $K = 5 \times 5$ achieves moderate visibility in other layouts, it remains insufficient in FC to reliably evaluate PO-ZSC performance. In contrast, for $K = 9 \times 9$ and $K = 1 1 \times 1 1$ , visibility ratios are consistently high across all layouts. As a result, the distinction between full and partial observability becomes blurred, making it difficult to isolate zero-shot coordination performance under partial observability. Based on this analysis, we select $K = 7 \times 7$ as the default window size for Burrito-PO, as it maintains a non-trivial level of visibility across all layouts while preserving sufficient occlusion. Under this setting, agents must meaningfully infer their partners while still receiving intermittent observations, forming an appropriate regime for evaluating ZSC under partial observability. We follow a similar procedure for the remaining benchmarks. This yields $K = 5 \times 5$ for Overcooked-PO. For Melting Pot, we adopt the official egocentric view range (5 cells to each side, 1 cell behind, and 9 cells ahead of the agent, with each cell rendered at $8 \times 8$ pixels) which we found to satisfy the criterion above, exhibiting neither degenerate near-zero nor near-full partner visibility across substrates. We therefore keep it unchanged and use these settings throughout our experiments.

## C.2 Hyperparameter Selection

Unless otherwise specified, hyperparameters are adopted from the corresponding baseline methods whenever possible, and the remaining hyperparameters are selected via grid search over a predefined search space, taking computational budget into consideration.

## C.3 Reward Function

In the Burrito environment, the one-step reward consists of a shared sparse reward and an agentspecific dense shaping reward. After each joint action, the reward for agent i is computed as

$$
r _ { i } ^ { t } = r _ { \mathrm { s p a r s e } } ^ { t } + \eta _ { t } r _ { \mathrm { d e n s e } , i } ^ { t } ,\tag{C.1}
$$

where $r _ { \mathrm { s p a r s e } } ^ { t }$ is the cooperative sparse reward shared by both agents, $r _ { \mathrm { d e n s e } , i } ^ { t }$ is the dense shaping reward caused by agent i at timestep $t ,$ and $\eta _ { t }$ is the reward-shaping coefficient.

Sparse reward. The sparse reward is generated only when a dish is delivered. If no delivery occurs, the sparse reward is zero. When a delivery occurs, the delivery reward is divided equally between the two agents, so both agents receive the same sparse reward regardless of which agent performed the final delivery action:

$$
r _ { \mathrm { s p a r s e } } ^ { t } = \left\{ \begin{array} { l l } { \frac { 1 } { 2 } R _ { \mathrm { d e l } } ^ { t } , } & { \mathrm { i f ~ a ~ d i s h ~ i s ~ d e l i v e r e d ~ a t ~ t i m e s t e p } \ : t , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{C.2}
$$

The delivery reward depends on the delivered dish type. If the dish matches the first active order, it is treated as an in-order delivery. If it matches another order in the order list, it is treated as an out-of-order delivery. Otherwise, it is treated as a random dish delivery. The delivery reward is computed as

$$
\begin{array} { r } { R _ { \mathrm { d e l } } ^ { t } = \left\{ \begin{array} { l l } { 2 0 + 2 0 n _ { \mathrm { i n g } } + m c _ { \mathrm { t i p } } , } & { \mathrm { i n - o r d e r ~ d e l i v e r y } , } \\ { 2 0 + 2 0 n _ { \mathrm { i n g } } + m c _ { \mathrm { t i p } } , } & { \mathrm { o u t - o f - o r d e r ~ d e l i v e r y } , } \\ { R _ { \mathrm { r a n d o m } } , } & { \mathrm { r a n d o m ~ d i s h ~ d e l i v e r y } , } \end{array} \right. } \end{array}\tag{C.3}
$$

where $n _ { \mathrm { i n g } }$ is the number of ingredients, m is the consecutive-order multiplier, $c _ { \mathrm { t i p } }$ is a deadlinebased $\mathrm { t i p } .$ , and $R _ { \mathrm { r a n d o m } }$ is the fixed reward for random dish delivery. In-order deliveries increase the consecutive-order count, whereas out-of-order deliveries reset it.

Dense shaping reward. The dense reward provides immediate feedback for useful intermediate task progress, such as cooking ingredients, assembling food, serving dishes, washing dishes, and resolving hazards. Unlike the sparse reward, the dense reward is not shared. Each agent receives dense reward only for shaping events caused by its own action at the current timestep.

The dense shaping term is used as a training aid. The coefficient $\eta _ { t }$ is annealed toward zero during training, so dense rewards help exploration and credit assignment early in training, while the final policy is optimized primarily toward the sparse cooperative delivery objective.

## C.4 VAE Dataset Construction

## C.4.1 Burrito-PO.

Following the VAE data-collection protocol of Burrito [23], we adapt the dataset construction to our training population and collect trajectories at the level of ordered policy pairs. For VAE training, we use 24 MEP policies as the training population and enumerate all ordered pairs by taking the Cartesian product of this population with itself. This produces $2 4 \times 2 4 = 5 7 6$ ordered policy pairs, including self-pairs. We collect three episodes for each ordered pair, resulting in 1728 trajectories, which are used to train the generative partner model.

## C.4.2 Overcooked-PO.

Following the VAE data-collection protocol of GAMMA [26], we adapt the dataset construction to our setting and collect trajectories using our training population. For VAE training, we use 24 MEP policies as the training population and collect 20k trajectories, which are used to train the generative partner model.

## C.4.3 Melting Pot.

Because the original Melting Pot work [22] does not provide a separate protocol for constructing a VAE training dataset, we follow the Burrito-PO protocol and adapt it to the Melting Pot setting. For VAE training, we use 24 MEP policies as the training population and enumerate all ordered pairs by taking the Cartesian product of this population with itself. This produces $2 4 \times 2 4 = 5 7 6$ ordered policy pairs, including self-pairs. We collect four episodes for each ordered pair, resulting in 2304 trajectories, which are used to train the generative partner model.

## C.5 Baselines

FCP [43] builds a frozen partner pool by collecting init / mid / final checkpoints from several self-play seeds, and trains a best-response (BR) ego policy by uniformly sampling partners from this pool. Our only strict-PO modification is the observation regime. The original method was evaluated under full observability, whereas in our setting both the Stage 1 self-play training (8 seeds) and the Stage 2 BR training are run with ego-centric partial observations using view radius 3, corresponding to a $7 \times 7$ local observation window. Thus, neither the ego nor any frozen partner observes the other agent’s state unless it falls within the local field of view. The algorithm itself (skill-stratified pool construction, uniform partner sampling, PPO-based BR training) is left unchanged.

MEP [50] builds a frozen partner pool by collecting init / mid / final checkpoints from several self-play seeds with a population entropy bonus, and trains a best-response (BR) ego policy using prioritized sampling that biases sampling toward low-performing partners. Our only strict-PO modification is the observation regime. The original method was evaluated under full observability, whereas in our setting both the Stage 1 self-play training (8 seeds) and the Stage 2 BR training are run with ego-centric partial observations. Thus, neither the ego nor any frozen partner observes the other agent’s state unless it falls within the local field of view. The algorithm itself (self-play with a population entropy bonus, prioritized partner sampling, PPO-based BR training) is left unchanged.

E3T [48] avoids any external partner pool by setting the partner side to a mixture of an ego self-copy and a random policy, $\pi _ { p } = \epsilon \pi _ { r } + ( 1 - \epsilon ) \pi _ { e }$ , and jointly trains, end-to-end, a context encoder consuming a length-k history of (state, partner-action) pairs together with a head that predicts the partner’s next action distribution; the ego policy is conditioned on the current state and the predicted partner action distribution and trained by PPO. Our strict-PO setting introduces two modifications. First, every state input on both the encoder side and the ego-policy side is replaced with the ego’s partial observation, so that no module ever sees the environment state directly. Second, the partneraction sequence fed to the encoder is FOV-masked: steps where the partner was inside the ego’s field of view keep the true action token, while steps where the partner was outside are replaced by a learnable ⟨UNK⟩ embedding, since the original lossless-encoding assumption no longer holds when the partner leaves the ego’s view. All other components (mixture partner, encoder/ego separate optimization, single-stage end-to-end) are kept as in the original.

<table><tr><td>Hyperparameter</td><td>Burrito-PO</td><td>Overcooked-PO</td><td>Melting Pot</td></tr><tr><td>CNN kernels CNN channels hidden layer size recurrent layer size activation function weight decay</td><td>[3, 3], [3, 3], [3, 3]</td><td>[3, 3], [3, 3], [3, 3] [32, 64, 32] [64] 64 ReLU</td><td>[8, 8], [4, 4], [3, 3]</td></tr><tr><td>environment steps parallel environments</td><td>60M 64</td><td>0 100M 200</td><td>200M 128</td></tr><tr><td>episode length PPO batch size</td><td>2400  $2 \times 2 4 0 0 \times 6 4$ </td><td>400  $2 \times 4 0 0 \times 2 0 0$ </td><td>1000  $2 \times 1 0 0 0 \times 1 2 8$ </td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>mini batch size</td><td> $2 \times 2 4 0 0 \times 6 4 \div 1 0$ </td><td> $2 \times 4 0 0 \times 2 0 0$ </td><td> $2 \times 1 0 0 0 \times 1 2 8 \div 1 6$ </td></tr><tr><td>PPO epoch</td><td></td><td></td><td></td></tr><tr><td>PPO learning rate</td><td></td><td>15</td><td></td></tr><tr><td></td><td></td><td>0.0005</td><td></td></tr><tr><td>GAE λ</td><td></td><td></td><td></td></tr><tr><td>discounting factor γ</td><td></td><td>0.95</td><td></td></tr><tr><td></td><td></td><td>0.99</td><td></td></tr></table>

Table C.1: Hyperparameters for policy models in each environment.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Behavioral Belief</td><td></td></tr><tr><td>λbehavior</td><td>0.1</td></tr><tr><td> $K _ { \mathrm { o f f } }$ </td><td>10</td></tr><tr><td> $\gamma _ { \mathrm { r e c } }$ </td><td>0.90 0.97</td></tr><tr><td> $\gamma _ { \mathrm { t e n } }$ </td><td>1.0</td></tr><tr><td> $\alpha _ { \mathrm { { r e c } } }$   $\alpha _ { \mathrm { t e n } }$ </td><td>0.3</td></tr><tr><td>Location Belief</td><td></td></tr><tr><td> $\lambda _ { \mathrm { l o c a t i o n } }$ </td><td>1.0</td></tr><tr><td>α</td><td>0.4</td></tr><tr><td>Gaussian target sigma</td><td>1.5</td></tr><tr><td></td><td>50</td></tr><tr><td> $H _ { \mathrm { m a x } }$ </td><td></td></tr><tr><td> $\Delta$ </td><td>[0, 1, 3, 5, 10, 20]</td></tr><tr><td> $w _ { \Delta }$ </td><td>[1.0, 0.5, 0.7, 1.0, 1.0, 0.8]</td></tr><tr><td> $\lambda _ { \mathrm { a u x } }$ </td><td>0.1</td></tr></table>

Table C.2: Partner-state Belief hyperparameters for target policy models.

ERS [31] learns expected-return symmetries, policy transformations σ composed of an observationside input map and an action-side output map that approximately preserve the expected return, and trains a cooperator against symmetry-transformed partners. In our setting, we train $\sigma _ { i j }$ for all 56 ordered pairs of eight strict-PO self-play base policies by pairing the frozen base<sub>i</sub> with σ ◦ base<sub>j</sub> and maximizing the cross-play return with PPO, where only σ and a fresh critic are trainable. We retain the top $l = 8$ symmetries ranked by cross-play return, compose a 16-policy pool from the base policies and their symmetry-transformed counterparts, and train the cooperator with PPO against uniformly sampled pool partners. We reduce the population size and pair budget relative to the original setting for computational reasons while keeping the symmetry architecture and training procedure unchanged.

GAMMA [26] pretrains a conditional VAE that generates partner behaviors, and trains the cooperator with PPO by sampling a fresh latent $z \sim \mathcal { N } ( 0 , \bar { I } )$ at the start of each episode and rolling out against the decoder conditioned on z. Our strict-PO setting introduces two modifications. First, while the original VAE encoder consumes full joint trajectories τ and the decoder relies on simulator-side state information, our VAE is retrained so that both the encoder and the decoder take only the partner’s PO as input; the partner generator is therefore closed under partial observability, and the partner behaviors produced by the decoder remain within the action distribution of strict-PO policies. Second, although the original work places no explicit restriction on the cooperator’s critic, we disable the centralized value function, so that both the actor and the critic of the cooperator operate solely on its own PO, i.e. in IPPO mode. The remaining structure (per-episode z sampling, BR training of the cooperator against the z-conditioned partner via PPO) is identical to the original.

<table><tr><td>Metric</td><td>Baseline</td><td>PIP (Ours)</td><td>Ratio</td></tr><tr><td>Total parameters</td><td>463K</td><td>591K</td><td>1.28×</td></tr><tr><td>Actor rollout inference time</td><td>1.13 ms</td><td>2.05 ms</td><td>1.81×</td></tr><tr><td>Actor rollout FLOPs</td><td>0.381G</td><td>0.764G</td><td>2.01×</td></tr></table>

Table C.3: Representative computational costs at the training rollout batch size of 64. Baseline corresponds to GAMMA / GOAT, whose target policies share an identical architecture.

<table><tr><td>hyperparameter</td><td>Stage 1 (teacher pretrain)</td><td colspan="2">Stage 2 (student distill)</td></tr><tr><td>trained module encoder input</td><td colspan="2">teacher (enc + dec)</td><td>student (enc + dec), teacher frozen</td></tr><tr><td></td><td colspan="2">(01 ∪ O2) ⊕ Opartner</td><td>Opartner</td></tr><tr><td>λKD</td><td colspan="2"></td><td>10.0</td></tr><tr><td>epoch</td><td colspan="2">500</td><td>500</td></tr><tr><td colspan="4">shared across stages</td></tr><tr><td>hyperparameter</td><td>Burrito-PO</td><td>Overcooked-PO</td><td>Melting Pot</td></tr><tr><td>CNN kernels CNN channels</td><td>[3, 3], [3, 3], [3, 3]</td><td>[3, 3], [3, 3], [3, 3]</td><td>[8, 8], [4, 4], [3, 3]</td></tr><tr><td>hidden layer size</td><td></td><td>[32, 64, 32] [256]</td><td></td></tr><tr><td>recurrent layer size</td><td></td><td>256</td><td></td></tr><tr><td>activation function</td><td></td><td>ReLU</td><td></td></tr><tr><td>weight decay</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0.0001</td><td></td></tr><tr><td>episode length</td><td>2400</td><td>400</td><td>1000</td></tr><tr><td>chunk length</td><td>200</td><td>100</td><td>100</td></tr><tr><td>chunks per episode</td><td>6</td><td>1</td><td>5</td></tr><tr><td>batch size</td><td></td><td>256</td><td></td></tr><tr><td>learning rate</td><td></td><td>0.001</td><td></td></tr><tr><td>KL penalty coefficient β</td><td></td><td>0 → 0.1</td><td></td></tr><tr><td>latent variable dimension</td><td></td><td>16</td><td></td></tr></table>

Table C.4: Hyperparameters for the Joint-view VAE.

GOAT [6] reuses the same frozen VAE decoder as a partner generator but replaces the fixed prior $z \sim \mathcal { N } ( 0 , I )$ with an adversary policy $\pi _ { A }$ that maps $z { \bar { \sim } } { \mathcal { N } } ( 0 , { \bar { I } } ) \to z ^ { \prime } = \pi _ { A } { \bar { ( z ) } }$ so as to maximize the regret Regret $( \pi _ { P } , \pi _ { C } ) = \mathbb { E } [ V ( \pi _ { P } , \pi _ { P } ) - V ( \pi _ { P } , \pi _ { C } ) ] = J _ { \mathrm { S P } } - J _ { \mathrm { X P } }$ . Each episode therefore consists of a two-phase rollout, a partner-vs-partner self-play rollout that estimates $J _ { \mathrm { S P } }$ and a partnervs-cooperator cross-play rollout that estimates J , with the cooperator and the adversary forming the two sides of a minimax game. Our strict-PO setting inherits the two GAMMA modifications and adds nothing else: the partner-generator VAE is trained to consume only the partner’s partial observation, and the cooperator is trained as IPPO. Since the adversary π<sub>A</sub> operates purely on the latent z and never receives any observation as input, it does not introduce any new PO-violating surface; the two-phase rollout structure and the regret objective are kept exactly as in the original method.

## C.6 Policy and VAE model details

We use a convolutional recurrent backbone for all policy models, where observations are encoded by a three-layer CNN, followed by a fully connected layer and a recurrent layer for partial observability. The MEP partners are trained with PPO using this backbone. The policy backbones and PPO hyperparameters for Burrito-PO, Overcooked-PO, and Melting Pot are summarized in Tab. C.1.

The target policy differs from the MEP models in two aspects. First, it is trained with simulated partner data generated from the pretrained partner model, rather than through the original MEP population training procedure. Second, our target coordinator additionally includes the Partnerstate Belief branch described in the main paper. Except for this additional branch, the base policy backbone and PPO optimization hyperparameters are kept identical to those of the MEP partners. The hyperparameters used for the Partner-state Belief branch during target policy training are summarized in Tab. C.2. The reverse KL direction discourages probability mass in low-feasibility regions without requiring the prediction to reproduce the full spread of the smoothed target.

For the generative partner model, we train the Joint-view VAE in two stages. In the first stage, the teacher VAE is trained using a richer partial-view input constructed from the union of both agents’ partial observations and the partner-side observation. In the second stage, the teacher is frozen and the student VAE is trained from the partner-side partial observation with an additional posterior distillation loss. We use KL(q<sub>T</sub>|q<sub>S</sub>) rather than the reverse direction so that the student posterior covers all modes of the more informative teacher posterior, preserving the diversity that the teacher captures from the joint view. Both stages share the same backbone architecture, optimizer settings, latent dimension, and KL annealing schedule. The VAE hyperparameters are provided in Tab. C.4.

## D Computational Resources

We report representative computational costs in Tab. C.3, measured at the training rollout batch size of 64. Compared with the baseline average of GAMMA and GOAT, our method increases total parameters from 463K to 591K, with the additional parameters concentrated in the actor while the critic size remains unchanged. Actor rollout inference time increases from 1.13 ms to 2.05 ms, and actor rollout computation increases from 0.381G to 0.764G FLOPs. This corresponds to an actor rollout inference rate of roughly 500 Hz at this batch size, suggesting that the inference overhead is unlikely to preclude real-time execution in this setting.

## E Environment Layouts

## E.1 Burrito-PO

We evaluate all methods on four Burrito-PO layouts, shown in Fig. E.1, which are adapted from Burrito [23]. The layouts differ in workspace openness, movement bottlenecks, object placement, and the spatial separation between task-relevant stations, thereby inducing different partial-observation challenges. The Forced-Coord, Open, Ring, and Hallway layouts correspond to panels (a)–(d) in Fig. E.1, respectively.

Forced-Coord (FC) consists of two narrow vertical workspaces separated by central obstacles and counters. Each agent has access to different nearby task objects, and the usable movement paths are constrained by narrow passages. This layout explicitly enforces spatial separation between agents, making the full recipe infeasible for a single agent under the same task constraints. Therefore, we do not report a single-play shortest-path completion step count for this layout. The layout instead emphasizes role division and handoff-style coordination, while the spatial separation makes it difficult for one agent to continuously observe the other.

Open provides a large central workspace with relatively few movement constraints. Ingredient dispensers, pots, plates, and delivery-related objects are placed around the boundary of the open area. Since agents can move more freely and visual contact is easier to recover, this layout represents the least structurally constrained setting among the four maps.

Ring contains a central blocked region surrounded by a loop-shaped traversable corridor. Task objects are distributed around the outside of the loop, so agents must coordinate while moving along constrained circular routes. The loop structure limits possible movement directions and often causes agents to disappear behind the central obstacle under ego-centric partial observation.

Hallway has a larger and more irregular workspace with task stations distributed across separated regions. The layout includes longer navigation paths and multiple object clusters, requiring agents to coordinate across delayed and partially occluded information. Compared with the other maps, Hallway introduces more complex spatial organization and stronger demands on navigation and partner-state inference.

![](images/4f1c20113cdd36bbd23e626ad0429c8c65e7133441ccfa6460cfc87904337a93.jpg)  
(a) Forced-Coord

![](images/84a54dd8861ad85be265af6180812ff09d37a6723caeda80ffbc1d00e6f020b6.jpg)  
(b) Open

![](images/6da68e194b6e35db4576ec33863e232140717f34f4b1dddd51000f444d591b86.jpg)  
(c) Ring

![](images/d440be86ef4b14dda498a183f92cbcf1ef0f57d6d98bd441b74855ec08858758.jpg)  
(d) Hallway  
Figure E.1: Burrito environment layouts used in our experiments. Each layout induces different coordination patterns under ego-centric partial observability.

![](images/88391ba72ba4031fe25c0d310947cbfbbeea47ece46c205cee1ca5829abea3c6.jpg)  
Figure E.2: The Overcooked environment layout used in our experiments, counter circuit.

The single-play shortest-path completion steps further quantify the structural differences among the layouts where a solo completion path is feasible. Open requires 137 steps, Ring requires 156 steps, and Hallway requires 172 steps to complete the recipe under a shortest-path solo planner. This increase reflects that the layouts require progressively longer navigation paths, more distributed object access, and more complex task execution even before multi-agent coordination is considered. Thus, the layouts differ not only in visual shape but also in the amount of movement and planning required to complete the underlying cooking task.

These layout variations allow us to evaluate coordination under different forms of partial observability, ranging from forced spatial separation and relatively open interaction spaces to loop-like movement and spatially distributed task structures.

![](images/6bbb52cabc4a66ef2bdee7c7f9ed783e39699a311142c9a09cd31a86f00d6970.jpg)  
(a) coop\_mining\_small (16 × 16)

![](images/58bf97ba100be0b5eec5d885d2632f5a437127fecb31fa629b5075352ebc4e8c.jpg)  
(b) coop\_mining (27 × 27)  
Figure E.3: The Melting Pot coop mining maps used in our experiments. Grey walls occlude the agents’ view, iron ore (grey) gives +1 when mined alone, whereas gold ore (yellow) gives +8 to each agent only if both mine the same ore within a 3-step window.

## E.2 Overcooked-PO

We evaluate on the counter\_circuit layout, a 5 × 9 ring shaped kitchen whose central counter block leaves only a one tile wide circular corridor. The two cooks cannot pass each other and lose sight of one another whenever they are on opposite sides of the ring. We chose this layout because it contains a tomato dispenser in addition to the onion dispenser. All five layouts of the original ZSC benchmark, cramped\_room, asymmetric\_advantages, coordination\_ring, forced\_coordination, and the onion only counter\_circuit\_1\_order, expose onions only and admit a single fixed recipe. As a result, partner uncertainty is reduced to a purely spatial problem. We instead use the counter\_circuit, which preserves the same ring topology while adding tomatoes. In this layout, all three recipes, onion tomato tomato, onion onion tomato, and onion tomato, require both ingredients. An agent must therefore infer which recipe its partner is pursuing from the pot contents and from which dispenser the partner visits. This makes the layout a benchmark for both ingredient level and spatial partner inference under partial observability.

## E.3 Melting Pot

In coop\_mining, iron ore can be mined individually for a reward of +1, whereas gold ore yields +8 to each agent only when both agents mine the same ore within a three step window. Obtaining the high value reward therefore requires successful rendezvous with a partner who is typically outside the agent’s egocentric field of view. Our pipeline trains an MEP partner population, a partner generating VAE, a KD teacher student VAE, and finally the target policy. Each stage requires hundreds of millions of environment steps. For our method, training on the original $2 { \bar { 7 } } \times 2 { \bar { 7 } }$ map (global map size 216 × 216) results in out-of-memory errors on an NVIDIA RTX A6000 GPU, making the computational and memory requirements prohibitively expensive. We therefore train all stages on a reduced 16×16 variant, coop\_mining\_small, with a global map size of 128×128, which preserves the original dynamics and reward rules. Since the policy input remains a fixed 88 $\times 8 8 \times 3$ egocentric RGB observation and the action space is identical across map sizes, the learned policies transfer without modification. We evaluate them in a zero shot manner on the original $2 7 \times 2 7$ coop\_mining map against the official Melting Pot bots, cooperator, defector, and mixed.

## E.4 Discussion of Environments

Our environment selection preserves comparability with existing partner-based evaluations while testing whether PIP transfers across partially observable settings. Burrito-PO retains the Burrito task and partner protocol, whereas Overcooked-PO follows the partner-based evaluation structures used in GAMMA and GOAT while modifying the observation condition. This design limits confounding changes in task mechanics and partner construction, allowing the effect of intermittent partner visibility to be examined more directly. Burrito-PO provides the primary setting for partner-state inference, while Overcooked-PO tests whether the same approach remains effective under a different task structure and held-out-partner protocol. In both environments, explicit predictions of unobserved partner locations and actions allow us to assess partner-state inference accuracy alongside coordination performance.

![](images/8a87d3f8853ce74d522cf3f3ac59d96b7dd601a03d934686e6cfe64d526fbafe.jpg)  
Figure F.1: t-SNE visualization of episode-level posterior means from each encoder on held-out trajectories from 12 BP partners (144 episodes per partner; perplexity 50, PCA initialization). Each point corresponds to one episode and is colored by the partner policy that generated it, with the same color used for the same partner across panels. Compared with (a) Fixed σ and (b) Het σ, (c) Ours produces more clearly separated partner-specific latent clusters.

OvercookedV2 [10] also supports partial observations, but its original evaluation uses cross-play among independently trained agents and studies coordination under asymmetric task information and protocol formation. Adopting it would therefore change the coordination scenario and evaluation protocol in addition to the observation condition. We use Overcooked-PO to preserve continuity with existing partner-based evaluations, rather than because partner-state inference cannot be studied in OvercookedV2.

Melting Pot [1] serves a complementary role by evaluating PIP under a different observation interface, task dynamics, and partner construction procedure. We therefore interpret its results as evidence of transfer across settings rather than directly comparing the magnitude of performance improvements with those observed in Burrito-PO and Overcooked-PO.

## F Additional Analysis

## F.1 Latent-Space Inspection via t-SNE

To complement the clustering metrics in the main paper, we qualitatively visualize the latent space of each encoder using t-SNE. For each held-out trajectory collected with one of the 12 held-out BP partners, we extract the episode-level posterior mean µ, project the embeddings to two dimensions with t-SNE (perplexity 50, PCA initialization), and color each point by the BP partner policy that generated the trajectory. The same color denotes the same held-out BP partner across panels, and all encoders are evaluated on the same held-out BP trajectories under the same local $7 \times 7$ observation setting.

As shown in Fig. F.1, the fixed-σ baseline produces a broad latent region with heavily intermixed partner identities. The heteroscedastic baseline forms somewhat more structured regions, but many partners remain overlapped. In contrast, the KD-distilled student yields substantially more separated partner-specific clusters, even though it uses the same local observation input at test time as the baselines. This visualization is consistent with the higher NMI and ARI reported in the main paper, suggesting that posterior distillation transfers behaviorally meaningful partner information from the richer teacher view to the partial-observation student.

## F.2 Partner-state Belief Prediction Analyses

We analyze whether the proposed belief-aware policy forms meaningful Partner-state Belief, as shown in Fig. F.2. The belief branch is designed to capture two complementary aspects of the partner: where the partner is or will be, and what the partner is likely to do.

![](images/69907c60095882cae6bf26ceb75706973e3eaad4df4b1ad014b7cbbe36341f96.jpg)

![](images/2ab2f103f3dc1103f7aeb1eba7b993fd3140a793e4e28d05ee3308712141b4c3.jpg)

![](images/20967f04f07207dcadbeb918c4327ae72b1cd661efaac5daf60d485894ca604a.jpg)  
Figure F.2: Partner-state Belief analysis. (a) Current Location Belief accuracy by Time Since Last Seen (TSLS). (b) ∆-step future Location Belief accuracy. (c) ∆-step future Behavioral Belief accuracy.

<table><tr><td>Method</td><td>Score</td></tr><tr><td>GAMMA</td><td> $1 4 3 . 3 \pm 1 1 . 2$ </td></tr><tr><td>GOAT</td><td> $1 0 3 . 0 \pm 8 . 6$ </td></tr><tr><td>PIP</td><td> ${ \bf 2 5 6 . 2 \pm 1 5 . 1 }$ </td></tr></table>

Table F.1: Full-observation coordination performance on the Burrito Open layout with 12 held-out behavior-preference partners.

We first evaluate the Location Belief. Since exact cell-level localization under partial observability is unnecessarily strict, the model predicts a coarse location bin, where each bin corresponds to $\phantom { - } 1 2 \times 2$ region of the map. In panel (a) of Fig. F.2, we measure current coarse location prediction accuracy when the partner is outside the ego agent’s field of view, grouped by Time Since Last Seen (TSLS), the number of steps since the partner left the visible range. To focus on practically meaningful cases, we evaluate prediction when the partner is moving. The proposed Location Belief consistently outperforms statistical baselines, including (i) probability spreading over time-dependent reachable regions (Reachable probability), (ii) movement-direction-based extrapolation (Dead-reckoning), (iii) empirical action statistics from the data (Empirical statistics), and (iv) random selection from the reachable region (Walkable random). The gap is especially clear in the small-TSLS region, where the partner has just disappeared from view. This indicates that our model forms an informative belief about the likely region of the currently invisible partner, rather than relying only on simple reachability statistics. Moreover, panel (b) shows that the model predicts the partner’s ∆-step future location more accurately than the baselines, suggesting that the Location Belief captures both current hidden location and near-future movement.

We next evaluate the Behavioral Belief, which consists of two temporal components: the reactive belief for near-future behavior and the tendency belief for broader action tendency. As shown in panel (c) of Fig. F.2, the reactive belief achieves high accuracy at short horizons and gradually degrades as the prediction horizon increases, matching its role of capturing near-future actions. In contrast, the tendency belief remains more stable across timesteps, suggesting that it captures the partner’s overall action distribution rather than a specific future action. Taken together, these results show that the belief branch forms Partner-state Belief as intended: the Location Belief captures where the partner is and will be, while the Behavioral Belief captures both near-future actions and longer-term tendencies.

## F.3 Full-Observation Evaluation

To assess how closely our partial observability training pipeline approaches the performance attainable under full observability, we conduct a controlled experiment in which the entire pipeline is instead trained in a fully observable setting. Comparing the resulting performance with that of our proposed method allows us to quantify the performance gap caused by partial observability and evaluate how effectively our approach mitigates it. Specifically, partner dataset collection, VAE training, and target policy training are all performed under full observability. During evaluation, the fully observable-trained target policy acts with full observations, while the held-out BP partners receive only partial observations, identical to those used in the main experiments. We deliberately evaluate against the same held-out BP partners as in the main partial observability experiments, since BP agents differ substantially in task proficiency depending on their behavior preference. Reusing the identical evaluation partners eliminates partner-specific performance differences as a confounding factor, allowing the comparison to isolate the effect of the full-observability setting. The results are summarized in Table F.1.

<table><tr><td>Method</td><td>Cap. Sup.</td><td></td><td>Reward</td></tr><tr><td>GAMMA</td><td>No</td><td>No</td><td> $1 6 3 . 3 \pm 1 0 . 6$ </td></tr><tr><td rowspan="3">+ Capacity control + Supervision control</td><td>Yes</td><td>No</td><td> $1 4 5 . 8 \pm 5 . 7$ </td></tr><tr><td>No</td><td>Yes</td><td> $8 8 . 7 \pm 1 0 . 2 $ </td></tr><tr><td>Yes</td><td>Yes</td><td> $9 6 . 0 \pm 7 . 1$ </td></tr><tr><td>PIP (Ours)</td><td>Yes</td><td>Yes</td><td> ${ \bf 2 4 9 . 0 \pm 1 2 . 2 }$ </td></tr></table>

Table F.2: Capacity and supervision controls on Burrito-PO Open. Cap. denotes capacity approximately matched to PIP, and Sup. denotes direct auxiliary supervision using privileged partner-state labels.

The proposed method exhibits no significant difference between training under full observability and partial observability, showing only a marginal improvement of +7.2. This result suggests that, in this setting, the additional information available under FO does not translate into a measurable performance gain. In contrast, both baselines perform worse despite having access to the informational advantage of full observability during training. GAMMA shows a decrease of −20.00, GOAT shows a decrease of −62.47.

Two distinct factors are involved in this experiment: the information available to the coordinator itself, and the behavior distribution of the partners it is trained with. Regarding the former, the Open layout is the least occluded of the four maps, so the information removed by the PO setting is comparatively limited. This may partially account for the small difference observed for our method. The latter factor, however, operates independently of how often the coordinator sees its partner. Partners acting under partial observability may search or wait when uncertain about the other agent, and thus follow a behavior distribution that differs from that of fully observable partners. An FO-trained coordinator is never exposed to such behaviors during training, so pairing it with PO evaluation partners introduces a partner-side distribution shift even when the coordinator’s own observations are complete. We hypothesize that GOAT’s pronounced degradation primarily reflects this shift, while identifying why the adversarially trained generator is particularly sensitive to it is left for future work.

## F.4 Capacity and Supervision Controls

To evaluate whether increased policy capacity or simple auxiliary partner-state prediction can reproduce PIP’s gains, we construct three controls based on GAMMA. The capacity control enlarges GAMMA’s policy GRU to approximately match PIP’s trainable parameter count. The supervision control attaches cross-entropy heads to GAMMA’s shared policy representation to predict future coarse partner locations and the partner’s current action. The joint control combines both modifications. Unlike these controls, PIP uses dedicated belief modules that predict future locations and discounted future-action distributions and conditions the policy on their representations. Because their action targets differ, the supervision controls do not exactly match PIP’s belief supervision. Results are reported in Tab. F.2.

None of the evaluated controls reproduces PIP’s performance. The capacity control achieves a lower mean reward than GAMMA, while the supervision and joint controls achieve substantially lower mean rewards. These results indicate that approximately matching PIP’s parameter count or adding the evaluated partner-state prediction heads to GAMMA is insufficient to account for PIP’s gain under this setting.

## F.5 Robustness to Observation Corruption

We test how our method handles degraded observations by evaluating the converged PIP coordinator without further training in Burrito-PO Open layout. At test time, we corrupt its observation using one of two stochastic processes. The sparse process is a blackout in which the entire observation is masked with probability r at each timestep, modelling intermittent sensor loss. The noisy process is a per-channel shuffle in which each observation channel is independently corrupted with probability $r ,$ and the values of each corrupted channel are spatially permuted within the field of view. This preserves the per-channel value distribution while destroying spatial correspondence, modelling per-modality scrambling. Thus, r denotes the probability of masking an entire observation under sparse blackout and the independent probability of shuffling each channel under noisy corruption. We sweep r from 0 to 0.2 and report the mean return against the held-out pool of M=12 partners (Mean ± S.E.; Fig. F.3). PIP degrades gradually under both corruption processes. At r=0.2, the entire observation is masked at 20% of timesteps under sparse blackout, whereas 20% of observation channels are shuffled on average under noisy corruption. PIP retains returns of 207.1 and 187.8 under these respective conditions. Across the full range of r, PIP remains above GAMMA under the uncorrupted observation condition $( 1 6 3 . 3 \pm 1 0 . 6 $ dashed line), showing that its performance advantage persists under substantial test-time observation corruption.

![](images/e11b2f3bbf281a56a6cf71c2249967e590787327263cfbb39998086ba6e67053.jpg)  
Figure F.3: Return of PIP against the M=12 partners as the corruption rate r increases from 0 to 0.2 (mean ± s.e.), under sparse (blackout) and noisy (per-channel shuffle) observation corruption. The dashed line is GAMMA with clean observations (r = 0).

## G Human Evaluation

## G.1 Participant Consent and Data Protection

As mentioned in the main paper, we voluntarily recruited participants for the experiment and provided them with sufficient time to understand the study procedure, potential risks, and data collection policy before obtaining their consent. As shown in Fig. G.1, the consent form described the purpose of the human-AI coordination study, the overall experimental procedure, the types of anonymous data collected, and the purpose of data collection. The form also clarified that no personal identifiers, IP addresses, device information, location information, or other tracking identifiers were collected. To minimize privacy risks, each participant was assigned a random anonymous ID, and all collected data were used only for statistical analysis and research reporting. The study involved no physical or psychological risks beyond ordinary computer use, and participants were informed that their participation was voluntary and that they could withdraw from the study at any time.

## G.2 Gameplay Interface

Before the main user study, participants completed an interactive tutorial to familiarize themselves with the game rules, controls, object interactions, and burrito-making procedure. As shown in the left panel of Fig. G.2, the tutorial interface displayed the current step, task instruction, relevant object icon, and a yellow “Continue” button. Participants advanced each step by clicking the button or pressing Enter, and the game timer was paused while instructions were displayed to avoid time pressure during reading. The tutorial guided participants through the essential gameplay mechanics, including movement, order-card interpretation, ingredient collection, chopping, cooking, plating, serving, fire handling, disposal of burned food, and dish washing, as summarized in Tab. G.1. After completing the guided steps, participants were given a 60-second free-play period to further practice the controls and gameplay loop before the main survey, helping ensure a fairer comparison across AI partners.

![](images/203c82ea3806f7fd43809b767f0f719388837c2e441ee608a39a4e71390180fd.jpg)  
Figure G.1: Participant consent form for the Burrito human-AI coordination study, outlining the study purpose, procedure, anonymous data collection, exclusion of personal identifiers, data protection policy, potential risks, voluntary participation, and withdrawal conditions.

![](images/84817061b96123422a93870fcc6f11c354c5d967a719db680ba5d7b48d8eb94f.jpg)

![](images/fb480e1f6704a44503d5cee6be0344dcd0ac0f520c49d2fc37c9d21769430fdc.jpg)  
Figure G.2: Tutorial and survey interfaces. The left panel shows step-by-step instructions, object icons, and the ego-centric partial-observation view. The right panel shows the main-survey gameplay interface, where participants cooperate with an anonymized AI partner under ego-centric partial observability.

<table><tr><td>#</td><td>Step</td><td>Instruction</td></tr><tr><td>1</td><td>Movement</td><td>Use the arrow keys to move in all four directions. Participants were informed that the game only resumes after pressing Enter or the</td></tr><tr><td>2</td><td>Read menu</td><td>yellow “Continue&quot; button. Inspect the order cards at the bottom of the screen. Each card indicates the requested burrito type and the remaining ticks before</td></tr><tr><td>3</td><td>Pick meat</td><td>the order expires. Orders can be delivered in any sequence. Move to the meat dispenser and press Space to pick up a piece of meat.</td></tr><tr><td>4</td><td>Chop meat</td><td>Place the meat on the cutting board, press Space repeatedly to chop</td></tr><tr><td>5</td><td>Grill meat</td><td>it, and pick up the resulting chopped meat. Place the chopped meat on the grill to cook it.</td></tr><tr><td>6</td><td>Boil rice</td><td>Pick up rice from the rice dispenser and place it in the pot to boil.</td></tr><tr><td>7</td><td>Assemble steak burrito</td><td>Pick up a clean plate and stack a tortilla, cooked meat, and boiled rice on the plate in any order. Participants were informed that ingredients can be added directly to a held plate using Space.</td></tr><tr><td>8</td><td>Serve steak burrito</td><td>Carry the completed steak burrito to the delivery location and press Space to deliver it for score points.</td></tr><tr><td>9</td><td>Pick mushroom</td><td>Move to the mushroom dispenser and press Space to pick up a mushroom.</td></tr><tr><td>10</td><td>Chop mushroom</td><td>Place the mushroom on the cutting board, press Space repeatedly to</td></tr><tr><td>11</td><td>Grill mushroom</td><td>chop it, and pick up the chopped mushroom. Place the chopped mushroom on the grill to fry it.</td></tr><tr><td>12</td><td>Boil rice again</td><td>Prepare another portion of rice by placing rice from the dispenser into the pot.</td></tr><tr><td>13</td><td>Assemble mushroom bur- rito</td><td>Stack a tortilla, fried mushroom, and boiled rice on a clean plate in any order to make a mushroom burrito.</td></tr><tr><td>14</td><td>Serve mushroom burrito</td><td>Carry the completed mushroom burrito to the delivery location and press Space to deliver it.</td></tr><tr><td>15</td><td>Burn demonstration</td><td>Place rice in the pot and wait until it burns into charcoal, demon- strating the consequence of overcooking.</td></tr><tr><td>16</td><td>Extinguish fire</td><td>Pick up the fire extinguisher, face the burning station, and press Space to put out the fire.</td></tr><tr><td>17</td><td>Trash charcoal</td><td>Put down the fire extinguisher, pick up the charcoal, and dispose of</td></tr><tr><td>18</td><td>Wash plate</td><td>it in the trash bin. Pick up a dirty plate, bring it to the sink, and press Space several</td></tr><tr><td>19</td><td>Free play</td><td>times to wash it clean. After all instructions were completed, participants freely played the game for 60 seconds before proceeding to the main survey.</td></tr></table>

Table G.1: Tutorial steps provided before the human-AI coordination survey. The tutorial familiarized participants with the game controls, object interactions, burrito recipes, and failure-recovery mechanics.

After the tutorial, participants played the main survey episodes with AI partners using the gameplay interface shown in the right panel of Fig. G.2. Each participant interacted with three different AI policies, GAMMA, GOAT, and PIP (Ours), in a randomized order, and played one 120-second (1200-timestep) episode with each policy. The interface displayed the anonymous participant ID, episode information, assigned AI partner, participant role and color, current score, remaining time, active order cards, plate count, and environment tick. The actual policy identity corresponding to the assigned AI partner, i.e., whether it was GAMMA, GOAT, or PIP (Ours), was not revealed to participants during gameplay. The survey gameplay followed an ego-centric partial-observation setting: participants could observe only a limited region centered on their own character, while areas outside the visible region were masked in black and updated as the participant moved. Thus, participants evaluated each anonymized AI partner under incomplete information about both the partner and the full task state.

## G.3 Post-Game Evaluation and Preference Survey

After each gameplay episode, participants completed a post-episode evaluation survey to assess the AI partner they had just played with. As shown in the left panel of Fig. G.3, the survey used

![](images/6932113b5cab9bec7349d01254d0c6eec55350af2a2ca037fab1830fb87cee37.jpg)  
Figure G.3: Post-game evaluation materials. The left panel shows the post-episode evaluation survey assessing each AI partner after gameplay. The right panel shows the final preference ranking of the three AI partners based on collaboration experience.

<table><tr><td>Metric</td><td>GAMMA</td><td>GOAT</td><td>PIP (Ours)</td><td>vs. GAMMA</td><td>vs. GOAT</td></tr><tr><td>Sparse reward mean ± SE</td><td>355.7 ± 19.7</td><td>370.4 ± 18.4</td><td>476.0 ± 22.7</td><td>+33.8%</td><td>+28.5%</td></tr><tr><td>Top-1 preference</td><td>10/66, 15.2%</td><td>8/66, 12.1%</td><td>48/66, 72.7%</td><td>+57.5 pts.</td><td>+60.6 pts.</td></tr></table>

Table G.2: Human evaluation performance and preference summary (N = 66 participants).

Likert-scale questions to measure participants’ subjective evaluation of the agent, including its coordination quality, predictability, adaptability, role complementarity, subjective experience, and overall teamwork. After completing all gameplay episodes with the three AI partners, participants also provided a final preference ranking. As shown in the right panel of Fig. G.3, this ranking asked participants to order the three AI partners according to their overall collaboration experience.

## G.4 Human Evaluation Details

We provide additional quantitative analysis of the human evaluation results. Each participant interacted with all three anonymized agents, GAMMA, GOAT, and PIP (Ours), under the same ego-centric partial-observation setting. We therefore treat the human evaluation as a within-subject comparison and analyze task performance, preference rankings, and post-episode survey scores.

Survey Score Construction. Each post-episode survey item was rated on a 1–7 Likert scale. Before computing category-level scores, negatively worded items were reverse-coded so that higher values consistently indicate better perceived partner quality. Each category score is then computed as the mean of its constituent items after this transformation.

Task Performance and Preference Ranking. Table G.2 summarizes task-level sparse reward and final preference ranking. PIP achieves the highest sparse reward among the three agents, with a mean reward of 476.0 compared with 355.7 for GAMMA and 370.4 for GOAT. This corresponds to a

<table><tr><td rowspan="2">Outcome</td><td colspan="2">Friedman test</td><td colspan="2">PIP vs. GAMMA</td><td colspan="2">PIP vs. GOAT</td></tr><tr><td> $\chi ^ { 2 } ( 2 )$ </td><td>p</td><td>p</td><td>r</td><td>p</td><td>r</td></tr><tr><td>Sparse reward</td><td>34.95</td><td> $2 . 5 7 \times 1 0 ^ { - 8 }$ </td><td> $2 . 4 3 \times 1 0 ^ { - 7 }$ </td><td>+0.74</td><td> $3 . 1 6 \times 1 0 ^ { - 6 }$ </td><td>+0.66</td></tr><tr><td>Preference rank</td><td>37.85</td><td> $6 . 0 4 \times 1 0 ^ { - 9 }$ </td><td> $4 . 8 0 \times 1 0 ^ { - 6 }$ </td><td>+0.62</td><td> $1 . 1 6 \times 1 0 ^ { - 6 }$ </td><td>+0.67</td></tr></table>

Table G.3: Within-subject statistical tests for task performance and preference ranking.

<table><tr><td>Category</td><td>GAMMA</td><td>GOAT</td><td>PIP</td><td>vs. GAMMA</td><td>vs. GOAT</td></tr><tr><td>Coordination</td><td>3.68</td><td>3.26</td><td>5.01</td><td>+36.1%</td><td>+53.7%</td></tr><tr><td>Predictability</td><td>3.33</td><td>3.11</td><td>4.83</td><td>+45.0%</td><td>+55.3%</td></tr><tr><td>Adaptability</td><td>3.29</td><td>3.11</td><td>4.89</td><td>+48.6%</td><td>+57.2%</td></tr><tr><td>Role Complementarity</td><td>3.09</td><td>2.80</td><td>4.35</td><td>+40.8%</td><td>+55.4%</td></tr><tr><td>Subjective Experience</td><td>3.33</td><td>3.13</td><td>5.02</td><td>+50.8%</td><td>+60.4%</td></tr><tr><td>Overall</td><td>3.45</td><td>3.18</td><td>5.27</td><td>+52.8%</td><td>+65.7%</td></tr></table>

Table G.4: Category-level survey scores and relative improvements.

<table><tr><td rowspan="2">Category</td><td colspan="2">Friedman test</td><td colspan="2">PIP vs. GAMMA</td><td colspan="2">PIP vs. GOAT</td></tr><tr><td> $\chi ^ { 2 } ( 2 )$ </td><td>p</td><td>p</td><td>r</td><td>p</td><td>r</td></tr><tr><td>Coordination</td><td>35.46</td><td> $1 . 9 9 \times 1 0 ^ { - 8 }$ </td><td> $2 . 1 4 \times 1 0 ^ { - 5 }$ </td><td>+0.62</td><td> $1 . 4 2 \times 1 0 ^ { - 7 }$ </td><td>+0.76</td></tr><tr><td>Predictability</td><td>39.07</td><td> $3 . 2 8 \times 1 0 ^ { - 9 }$ </td><td> $6 . 6 4 \times 1 0 ^ { - 7 }$ </td><td>+0.72</td><td> $4 . 3 1 \times 1 0 ^ { - 8 }$ </td><td>+0.82</td></tr><tr><td>Adaptability</td><td>33.95</td><td> $4 . 2 5 \times 1 0 ^ { - 8 }$ </td><td> $1 . 1 4 \times 1 0 ^ { - 5 }$ </td><td>+0.66</td><td> $1 . 2 3 \times 1 0 ^ { - 6 }$ </td><td>+0.72</td></tr><tr><td>Role Complementarity</td><td>28.31</td><td> $7 . 1 2 \times 1 0 ^ { - 7 }$ </td><td> $7 . 1 0 \times 1 0 ^ { - 5 }$ </td><td>+0.60</td><td> $3 . 0 1 \times 1 0 ^ { - 6 }$ </td><td>+0.73</td></tr><tr><td>Subjective Experience</td><td>32.57</td><td> $8 . 4 7 \times 1 0 ^ { - 8 }$ </td><td> $1 . 2 0 \times 1 0 ^ { - 6 }$ </td><td>+0.73</td><td> $5 . 8 6 \times 1 0 ^ { - 7 }$ </td><td>+0.76</td></tr><tr><td>Overall</td><td>42.90</td><td> $4 . 8 3 \times 1 0 ^ { - 1 0 }$ </td><td> $2 . 7 0 \times 1 0 ^ { - 7 }$ </td><td>+0.79</td><td> $1 . 8 5 \times 1 0 ^ { - 7 }$ </td><td>+0.78</td></tr></table>

Table G.5: Supporting within-subject statistical tests for post-episode survey scores. Wilcoxon pvalues are uncorrected, and all PIP-vs-baseline comparisons remain significant after Holm–Bonferroni correction.

33.8% improvement over GAMMA and a 28.5% improvement over GOAT. PIP is also ranked as the most preferred partner by 48 out of 66 participants, corresponding to 72.7% of all participants. In comparison, GAMMA and GOAT are selected as the top-ranked partner by 15.2% and 12.1% of participants, respectively. The full preference distribution is shown in panel (b) of the humanevaluation figure in the main paper.

Statistical Analysis. Because each participant evaluated all three agents, we use within-subject nonparametric tests and report p-values rather than relying on the visual overlap of standard-error bars. The significance markers shown in panel (a) of the human-evaluation figure in the main paper correspond to the paired Wilcoxon signed-rank tests reported here. We first apply Friedman tests [9] across the three agents, followed by paired Wilcoxon signed-rank tests [46] comparing PIP against each baseline, and report matched-pairs rank-biserial correlation r as the effect size. As summarized in Tab. G.3, the Friedman tests show significant differences across agents for both sparse reward and preference rank. The paired Wilcoxon tests show that PIP significantly outperforms both GAMMA and GOAT on both metrics, with large effect sizes $( r \geq 0 . 6 2 )$ . Reported paired Wilcoxon signed-rank test values are uncorrected p-values, and all paired comparisons remain significant after Holm-Bonferroni correction across the four PIP-vs-baseline tests in Tab. G.3. For preference rank, the sign of r is oriented so that $r > 0$ indicates that PIP achieves a lower, better rank than the baseline.

Survey Score Decomposition. Table G.4 reports the category-level scores and relative improvements of human evaluation. PIP receives the highest score in every category. The largest relative gains appear in Overall, Subjective Experience, Predictability, and Adaptability, suggesting that participants perceived PIP not only as more task-effective, but also as easier to understand, more responsive to their behavior, and more desirable to cooperate with under partial observability.

As a supporting analysis, we also test each survey category using within-subject nonparametric tests and report matched-pairs rank-biserial correlation [20] r as the effect size, where $| r | \geq 0 . 5$ is conventionally considered a large effect. As summarized in Tab. G.5, Friedman tests indicate significant differences among the three agents for all six survey categories. Paired Wilcoxon signedrank tests show that PIP significantly outperforms both baselines in every category, with large effect sizes throughout $( r \geq 0 . 6 0 )$ . Reported paired Wilcoxon p-values are uncorrected, and these paired survey comparisons remain significant after Holm-Bonferroni correction [14] across the 12 PIP-vs-baseline survey comparisons in Tab. G.5.

Overall, the human evaluation suggests that the proposed agent improves both objective and subjective aspects of human-AI coordination. Participants achieved higher sparse reward with PIP, selected it as the most preferred partner substantially more often, and gave it higher ratings across all survey dimensions. These results indicate that the proposed framework improves not only tasklevel coordination, but also perceived predictability, adaptability, role complementarity, and overall cooperation quality under ego-centric partial observability.