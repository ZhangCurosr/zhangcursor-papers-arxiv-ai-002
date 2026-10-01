# GAME-GUIDED SKILL DISCOVERY THROUGH SELF-PLAY FOR PLAYABLE AGENT CONTROL

Seungeun Rho<sup>1</sup> Jeonghwan Kim<sup>1</sup> Xue Bin Peng<sup>2</sup> Sehoon Ha<sup>1</sup>

<sup>1</sup>Georgia Institute of Technology

<sup>2</sup>Simon Fraser University and NVIDIA

{srho31,jkim3662,sehoonha}@gatech.edu, xbpeng@sfu.ca

## ABSTRACT

We present Game-Guided Skill Discovery (GGSD), a framework that uses selfplay in games to discover motor skills that are directly playable by humans. Playable skills provide a compact abstraction for controlling embodied agents through a small set of learned behaviors rather than low-level actions. To be effective, these skills should be semantically distinct, interpretable, and expressive; properties that existing unsupervised skill-discovery methods often fail to achieve simultaneously. GGSD achieves these desiderata by grounding skill discovery in competitive gameplay. A hierarchical agent competes against its past selves, with a high-level policy selecting from a small discrete skill set and a skill-conditioned low-level policy learning the corresponding behaviors. After training, a human can replace the high-level policy and directly control the agent through the same discrete skills. Despite the small number of high-level actions, skill transitions give rise to emergent combo behaviors, expanding expressivity beyond individual primitives. Across Ant, Franka-arm, and Unitree G1 environments, we show that GGSD produces human-playable skills that humans can compose to solve unseen tasks, such as Maze and CubePush, without additional training. An interactive demo is available at https://ggsd-demo.github.io.

![](images/475d0cf98ee262fbe60c336e8acd2890400763a176c1ab7f865e95fc20a0577a.jpg)  
Figure 1: Different game rules guide agents toward different playable skills. Shown are gameplay snapshots of agents learned with GGSD.

## 1 INTRODUCTION

Learning a compact repertoire of reusable motor skills is a long-standing goal in skill discovery. Such skills can provide useful abstractions not only for downstream policy learning, but also for direct human control: if complex behaviors are organized into a small set of intuitive skills, a human can select and compose them without specifying low-level actions. For this interface to be effective, the discovered skills should be semantically distinct, interpretable, scalable to high-degree-offreedom agents, and sufficiently expressive despite a compact action space.

Existing unsupervised skill-discovery methods do not naturally guarantee these properties. Most approaches optimize objectives based on mutual information (MI) (Gregor et al., 2016; Sharma et al., 2019; Eysenbach et al., 2019; Kwon, 2020; Laskin et al., 2022) or Wasserstein dependency measures (WDM) (Park et al., 2021; 2024), encouraging different skills to induce distinguishable state distributions or trajectories. While this can produce interpretable behaviors in simple environments, increasing agent complexity introduces many ways for skills to differ without being behaviorally meaningful. As a result, distinctiveness alone can yield skills that are easy to distinguish but difficult to interpret or reuse.

We argue that competitive games provide a natural source of structure for discovering such skills, as illustrated across diverse embodiments in Figure 1. A simple game rule specifies what constitutes success while leaving open how success should be achieved. Through self-play, evolving opponents continually expose the agent to new strategic and physical situations, encouraging the emergence of useful behaviors without requiring users to specify individual skills a priori. Self-play has long been shown to induce emergent behaviors, from superhuman strategies in competitive games Silver et al. (2016; 2018); Vinyals et al. (2019); Baker et al. (2020); Oh et al. (2021) to structured motor behaviors in physically embodied agents Bansal et al. (2018); Haarnoja et al. (2024); Jansonnie et al. (2024). We ask whether this same mechanism can be used to discover a compact repertoire of skills that is directly playable by humans.

Based on this idea, we introduce Game-Guided Skill Discovery (GGSD), a framework that uses self-play in simple 1v1 competitive games to discover playable motor skills. Each agent is controlled by a hierarchical policy. A high-level policy selects from a small set of discrete skills, while a skill-conditioned low-level policy maps the selected skill to motor actions. The game objective encourages behaviors that are useful for winning, while a mutual-information objective encourages different skill codes to acquire distinct behavioral semantics.

After training, we remove the high-level policy and map each skill to a human input, such as a keyboard button, allowing users to directly control the agent without training a new task-specific controller. We intentionally keep the number of skills small (only five or six) in all environments to maintain a simple interface for human play. Despite this compact action vocabulary, the controller gains additional expressivity through skill transitions: executing one skill can place the agent in a state from which another skill produces a qualitatively different behavior, giving rise to emergent combo behaviors. Similar to button combinations in commercial games, these transitions allow a small set of discrete skills to support behaviors richer than the individual primitives alone.

As a result, GGSD discovers semantically diverse and human-interpretable skills even for highdegree-of-freedom embodiments such as humanoids. These skills are also directly reusable beyond the games in which they are learned. Without any additional policy training, humans can compose them to solve previously unseen downstream tasks, including complex locomotion in Maze and object interaction in CubePush.

The core contributions of our work are as follows:

• We introduce GGSD, a skill-discovery framework that uses self-play in games as lightweight guidance for learning human-playable motor skills.

• We show that GGSD discovers semantically distinct and human-interpretable skills that scale to high-degree-of-freedom agents. Despite using only a small discrete skill set, transitions between skills give rise to emergent combo behaviors that substantially expand the controller’s expressivity.

• We demonstrate GGSD across Ant, Franka Arm, and Unitree G1 environments, and show that humans can directly compose the learned skills to solve previously unseen locomotion and object-interaction tasks without additional policy training.

## 2 RELATED WORKS

## 2.1 SELF-PLAY FOR MOTOR SKILL LEARNING

Fictitious play provides a classical mechanism for stabilizing self-play by repeatedly learning best responses to the empirical average of opponents’ historical strategies (Brown, 1951; Robinson, 1951). Fictitious self-play extends this idea to extensive-form games (Heinrich et al., 2015), while Neural Fictitious Self-Play (NFSP) approximates best responses with deep reinforcement learning and historical average strategies with supervised learning (Heinrich & Silver, 2016). These methods primarily aim to learn equilibrium strategies. Similarly, our method trains against a distribution of historical policies. Competing against a diverse pool of past selves encourages the agent to develop reusable skills that remain useful across a wide range of opponents rather than specializing to a particular strategy.

Consistent with this intuition, self-play and competitive interaction have been shown to induce complex motor behaviors in physically simulated agents. Bansal et al. (2018) demonstrated the emergence of behaviors such as running, blocking, tackling, and kicking through multi-agent competition, while later works extended competitive learning to bipedal soccer (Haarnoja et al., 2024), robotic manipulation (Jansonnie et al., 2024), and hierarchical multi-drone volleyball (Zhang et al., 2025). Won et al. (Won et al., 2021) studied high-DoF humanoids in boxing and fencing, but learned the underlying motor skills from reference motions before training competitive strategies.

In contrast, we use self-play itself as guidance for discovering the low-level skill repertoire. Rather than learning a monolithic game-playing policy or relying on predefined/reference-based motor skills, our method explicitly organizes behaviors induced by competition into discrete, diverse, and reusable skills that can be directly controlled by humans.

## 2.2 GUIDANCE IN SKILL DISCOVERY

Several works introduce external guidance to address a key limitation of unsupervised skill discovery, as optimizing solely for distinctiveness or state coverage can produce behaviors that are diverse but semantically meaningless. Language-Guided Skill Discovery (Rho et al., 2025b) uses LLMgenerated state descriptions to encourage semantically distinct skills. However, obtaining language descriptions throughout the explored state space becomes increasingly impractical as agent dimensionality grows. DoDont (Kim et al., 2024) uses desirable and undesirable demonstrations, while Reference Grounded Skill Discovery (Rho et al., 2026) uses reference motions to guide the learned skill repertoire. While effective for high-DoF agents, the discovered repertoire is largely shaped by the provided motion dataset, and preparing sufficiently diverse reference motions can itself be costly. In contrast, guidance from simple game rules is lightweight. A game’s rules can be specified in only a few lines of code, while self-play autonomously discovers how to succeed under those rules. This allows the agent to discover novel behaviors beyond explicitly provided examples, while remaining scalable to high-DoF systems.

## 3 PRELIMINARIES

Markov Decision Process. We first consider a standard Markov decision process (MDP), defined by $\boldsymbol { \mathcal { M } } = ( \boldsymbol { \mathcal { S } } , \boldsymbol { \mathcal { A } } , \boldsymbol { P } , \boldsymbol { r } , \boldsymbol { \gamma } )$ , where S and A denote the state and action spaces, $P ( s ^ { \prime } \mid s , a )$ is the transition probability, $r ( s , a )$ is the reward function, and $\gamma \in [ 0 , 1 )$ is the discount factor. A policy $\pi ( a \mid s )$ is trained to maximize the expected discounted return

$$
J ( \pi ) = \mathbb { E } _ { \pi , P } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t } , a _ { t } ) \right] .\tag{1}
$$

Two-player games. We consider a two-player game between an agent and an opponent. The game state is given by $s = ( s ^ { \mathrm { m e } } , s ^ { \mathrm { f o e } } )$ , where $s ^ { \mathrm { { \dot { m e } } } }$ denotes the agent’s own state, and $s ^ { \mathrm { f o e } }$ denotes the state of the opponent. Similarly, the two players take actions $a ^ { \mathrm { m e } } \in \mathcal { A } ^ { \mathrm { m e } }$ and $a ^ { \mathrm { f o e } } \in \mathcal { A } ^ { \mathrm { f o e } }$ . The game dynamics are described by

$$
P _ { \mathrm { g a m e } } \left( s ^ { \prime } \mid s , a ^ { \mathrm { m e } } , a ^ { \mathrm { f o e } } \right) ,\tag{2}
$$

and the controlled agent receives a game reward $r ^ { \mathrm { m e } } ( s , a ^ { \mathrm { m e } } , a ^ { \mathrm { f o e } } )$

When the opponent follows a fixed policy $\pi _ { \mathrm { f o e } } ( a ^ { \mathrm { f o e } } \mid s )$ over continuous actions, its action can be marginalized into the environment dynamics. In particular, we define the induced transition probability

$$
\tilde { P } _ { \pi _ { \mathrm { f o e } } } ( s ^ { \prime } \mid s , a ^ { \mathrm { m e } } ) = \int _ { A ^ { \mathrm { f o e } } } \pi _ { \mathrm { f o e } } ( a ^ { \mathrm { f o e } } \mid s ) P _ { \mathrm { g a m e } } \left( s ^ { \prime } \mid s , a ^ { \mathrm { m e } } , a ^ { \mathrm { f o e } } \right) d a ^ { \mathrm { f o e } } ,\tag{3}
$$

with the corresponding expected reward

$$
\tilde { r } _ { \pi _ { \mathrm { f o e } } } ( s , a ^ { \mathrm { m e } } ) = \int _ { \mathcal { A } ^ { \mathrm { f o e } } } \pi _ { \mathrm { f o e } } ( a ^ { \mathrm { f o e } } \mid s ) r ^ { \mathrm { m e } } ( s , a ^ { \mathrm { m e } } , a ^ { \mathrm { f o e } } ) d a ^ { \mathrm { f o e } } .\tag{4}
$$

![](images/c904b9f36e1c5de45ce29b98b03e9f35c5f05a5479ed4d3948fb099ea20704da.jpg)  
Figure 2: Overview of GGSD. Top: We jointly learn a hierarchical policy through self-play against opponents sampled from a pool of past checkpoints. Bottom: After training, a human can replace the high-level policy and directly control the agent across diverse scenarios.

Therefore, for a fixed opponent policy, the two-player game reduces to an ordinary single-agent MDP

$$
\mathcal { M } _ { \pi _ { \mathrm { f o e } } } = \left( { \cal S } , \mathcal { A } ^ { \mathrm { m e } } , \tilde { P } _ { \pi _ { \mathrm { f o e } } } , \tilde { r } _ { \pi _ { \mathrm { f o e } } } , \gamma \right) .\tag{5}
$$

From the perspective of the controlled agent, training against a fixed opponent is thus equivalent to standard reinforcement learning in an environment whose dynamics implicitly include the opponent’s behavior. Self-play can be viewed as repeatedly changing this MDP as the opponent policy evolves over the course of training.

## 4 GAME-GUIDED SKILL DISCOVERY

GGSD converts game rules into human-playable skills through self-play. We first describe our selfplay procedure, then introduce hierarchical skill learning, and finally explain how the learned skills are made directly playable by humans.

## 4.1 SELF-PLAY WITH A POOL OF PAST SELVES

A straightforward form of self-play trains against a copy of the current policy. However, as the policy is updated, the opponent changes simultaneously, resulting in a continuously moving training objective. Instead, as illustrated in Figure $^ { 2 , }$ we maintain a pool of historical policy checkpoints and sample a fixed opponent at the beginning of each episode.

Recall from Sec. 3 that a fixed opponent policy $\pi _ { \mathrm { f o e } }$ induces a single-agent MDP $\mathcal { M } _ { \pi _ { \mathrm { f o e } } }$ . We maintain an opponent pool

$$
\Pi _ { K } = \{ \pi ^ { ( 1 ) } , \ldots , \pi ^ { ( K ) } \} ,\tag{6}
$$

where $\pi ^ { ( K ) }$ denotes the most recent checkpoint. When the pool size K is greater than 1, we sample an opponent according to

$$
\rho _ { K } ( k ) = \left\{ \begin{array} { l l } { { p , } } & { { k = K , } } \\ { { \displaystyle \frac { 1 - p } { K - 1 } , } } & { { 1 \leq k < K , } } \end{array} \right.\tag{7}
$$

and keep it fixed throughout the episode. Thus, the latest checkpoint is sampled with probability $p ,$ while the remaining probability is distributed uniformly across earlier checkpoints.

This results in an opponent distribution that is piecewise stationary rather than continuously changing, while still maintaining competitive pressure from recent opponents. The parameter $p$ controls the balance between these two effects: larger values place greater emphasis on the latest opponent, while smaller values increase diversity from past strategies. We use $p \in [ 0 . 6 , 0 . 8 ]$ in all experiments.

Our procedure is inspired by fictitious play (Brown, 1951; Robinson, 1951) and its self-play variants (Heinrich et al., 2015; Heinrich & Silver, 2016). While we do not implement exact fictitious play, we follow its central principle of training against a distribution of past strategies, with additional emphasis on recent opponents.

## 4.2 HIERARCHICAL SKILL LEARNING

Jointly learning strategies and motor skills. GGSD jointly learns the high- and low-level policies during self-play. The high-level policy determines which skill to execute as part of the current game strategy, while the low-level policy determines how that skill is physically realized. Joint training allows the two levels to co-evolve: improved strategies expose the agent to new situations that call for new low-level skills, while an expanding skill repertoire enables more sophisticated strategies.

This joint training is implemented through a temporal hierarchy. Let $s _ { t } ^ { \mathrm { a l l } } = ( s _ { t } ^ { \mathrm { m e } } , s _ { t } ^ { \mathrm { f o e } } )$ denote the full game state, and let $\tau _ { j } = j k$ denote the j-th high-level decision time.

The high-level policy selects a discrete skill

$$
z _ { j } \sim \pi _ { \mathrm { h i g h } } \left( z \mid s _ { \tau _ { j } } ^ { \mathrm { a l l } } \right) ,\tag{8}
$$

which is held fixed for the following k environment steps. Within this interval, the low-level policy produces a motor action at every step,

$$
a _ { t } \sim \pi _ { \mathrm { l o w } } \left( a \mid s _ { t } ^ { \mathrm { p r o p r i o } } , z _ { j } \right) , \qquad \tau _ { j } \leq t < \tau _ { j } + k ,\tag{9}
$$

where $s _ { t } ^ { \mathrm { p r o p r i o } }$ contains only the controlled agent’s proprioceptive observations.

Hierarchical policy optimization. During training, we retain the sampled skill variables and optimize the joint distribution over the augmented trajectory. For one skill interval, its policy-dependent probability factorizes as

$$
p _ { \theta } \left( z _ { j } , a _ { \tau _ { j } : \tau _ { j } + k - 1 } \mid s _ { \tau _ { j } : \tau _ { j } + k - 1 } \right) = \pi _ { \mathrm { h i g h } } \left( z _ { j } \mid s _ { \tau _ { j } } ^ { \mathrm { a l l } } \right) \prod _ { t = \tau _ { j } } ^ { \tau _ { j } + k - 1 } \pi _ { \mathrm { l o w } } \left( a _ { t } \mid s _ { t } ^ { \mathrm { p r o p r i o } } , z _ { j } \right) .\tag{10}
$$

Thus, the joint log likelihood decomposes into high- and low-level terms, allowing the two policies to be optimized at their respective temporal resolutions. We use separate PPO objectives for both levels, following prior work on joint hierarchical policy optimization (Li et al., 2020).

Each level maintains its own value function and reward. The high-level policy is optimized only for the game objective, since it is responsible for strategic skill selection. The low-level policy additionally receives the mutual-information reward $r _ { t } ^ { \mathrm { M I } }$ defined in Eq. 15, which encourages different skill codes to acquire distinct motor behaviors:

$$
r _ { t } ^ { \mathrm { h i g h } } = r _ { t } ^ { \mathrm { g a m e } } , \qquad r _ { t } ^ { \mathrm { l o w } } = r _ { t } ^ { \mathrm { g a m e } } + \lambda _ { \mathrm { M I } } r _ { t } ^ { \mathrm { M I } } .\tag{11}
$$

We view the game objective as providing the behavioral guidance that shapes which skills emerge, while MI primarily associates these behaviors with distinct skill codes; see Appendix B for further discussion.

The low-level critic $V _ { \mathrm { l o w } } ( s _ { t } ^ { \mathrm { a l l } } , z _ { j } )$ operates at every environment step. The high-level critic operates only at skill-selection steps, and is trained using target values calculated according to:

$$
y _ { j } ^ { \mathrm { h i g h } } = \sum _ { i = 0 } ^ { k - 1 } \gamma ^ { i } r _ { \tau _ { j } + i } ^ { \mathrm { g a m e } } + \gamma ^ { k } V _ { \mathrm { h i g h } } \left( s _ { \tau _ { j + 1 } } ^ { \mathrm { a l l } } \right) .\tag{12}
$$

Thus, the high-level policy uses an effective discount factor of $\gamma ^ { k }$ between consecutive skill decisions, while the low-level policy uses the original discount factor $\gamma$ at every environment step. We use $k = 1 0$ in all experiments.

## 4.3 PLAYABLE LEARNED SKILLS

Having described how the two levels are jointly trained, we now introduce the design choices that allow the learned low-level skills directly playable by humans.

Discrete and diverse skills. We use a discrete skill variable $z \in { 1 , \dots , M } ,$ , represented as a onehot vector, with $M = 5$ or 6 depending on the environment. This compact skill set provides a simple interface for human control, while state-dependent skill transitions enable emergent combo behaviors (Sec. 5.3).

To prevent the high-level policy from collapsing to a small subset of skills, we regularize it with categorical entropy:

$$
\mathcal { H } _ { \mathrm { h i g h } } = \mathbb { E } _ { s ^ { \mathrm { a l l } } } \left[ \mathcal { H } \left( \pi _ { \mathrm { h i g h } } ( \cdot  { \left| \begin{array} { l } { s ^ { \mathrm { a l l } } } \end{array} \right) \right) } \right] .\tag{13}
$$

This encourages the policy to leverage the full discrete skill set during gameplay.

Mutual-information maximization. High-level entropy encourages diverse skill selection, but does not ensure that different codes correspond to distinct behaviors. We therefore maximize the mutual information between the selected skill Z and the resulting state S. A discriminator $q _ { \phi } ( z \mid s )$ gives the variational lower bound

$$
\begin{array} { r } { I ( Z ; S ) \ge \mathcal { H } ( Z ) + \mathbb { E } _ { p ( z , s ) } \left[ \log q _ { \phi } ( z \mid s ) \right] . } \end{array}\tag{14}
$$

Although the skill distribution is induced by the high-level policy, it is fixed with respect to $\theta _ { \mathrm { l o w } }$ during the low-level update. The discriminator therefore provides the intrinsic reward

$$
\begin{array} { r } { r _ { t } ^ { \mathrm { M I } } = \log q _ { \phi } ( z _ { j } \mid s _ { t } ) . } \end{array}\tag{15}
$$

Together, high-level entropy encourages the use of multiple skills, while the MI objective encourages different skill codes to induce distinguishable behaviors.

Opponent-agnostic low-level control. For direct human control, each skill should retain consistent motor semantics across different game situations. To encourage this, we restrict both the low-level policy and the discriminator to proprioceptive observations, while allowing the high-level policy to observe the full game state:

$$
\pi _ { \mathrm { h i g h } } \left( z \mid s ^ { \mathrm { a l l } } \right) , \qquad \pi _ { \mathrm { l o w } } \left( a \mid s ^ { \mathrm { p r o p r i o } } , z \right) , \qquad q _ { \phi } \left( z \mid s ^ { \mathrm { p r o p r i o } } \right) .\tag{16}
$$

This places opponent-dependent strategic reasoning in the high-level controller and prevents the discriminator from distinguishing skills based on opponent states, encouraging low-level skills to retain consistent motor semantics that are easier for humans to control.

Putting these components together, the two policies optimize the following objective:

$$
\begin{array} { r l } & { \theta _ { \mathrm { h i g h } } ^ { * } = \underset { \theta _ { \mathrm { h i g h } } } { \arg \operatorname* { m i n } } \mathcal { L } _ { \mathrm { h i g h } } ^ { \mathrm { P P O } } \left( r ^ { \mathrm { g a m e } } \right) - \lambda _ { \mathrm { H } } \mathcal { H } _ { \mathrm { h i g h } } , } \\ & { \theta _ { \mathrm { l o w } } ^ { * } = \underset { \theta _ { \mathrm { l o w } } } { \arg \operatorname* { m i n } } \mathcal { L } _ { \mathrm { l o w } } ^ { \mathrm { P P O } } \left( r ^ { \mathrm { g a m e } } + \lambda _ { \mathrm { M I } } r ^ { \mathrm { M I } } \right) . } \end{array}\tag{17}
$$

Pseudocode for GGSD is provided in Algorithm 1, and the full hyperparameters in Appendix D.

## 5 EXPERIMENTS

## 5.1 IMPLEMENTATION DETAILS

Game Rules. GGSD transforms a given game rule into a set of playable motor skills. We consider four games with distinct objectives and embodiments. (1) In AntSumo, an agent wins by pushing its opponent out of the arena, while falling results in a loss. (2) In AntFencing, an agent wins by touching the opponent’s root body with one of its front legs; falling or leaving the arena results in a loss. (3) In FrankaAirHockey, two Franka robot arms compete to strike a puck into the opponent’s goal. (4) In G1Boxing, two humanoid robots compete by striking the opponent’s head or torso, with the objective of inflicting more damage than the opponent. Detailed reward terms, observations, and environment configurations for each game are provided in the Appendix C.

![](images/bf25ac55a21a049a757d9db5cb60e43eec28023cb63dd7b8d592b7806c3004d0.jpg)  
win + draw---- win rate-- draw rate

Figure 3: Win and draw rates of the final policy against itself and its previous checkpoints.  
![](images/8626d86d98d1044bf0178756f3ef283613687b8d0eabc47a329d2494642816bb.jpg)  
Figure 4: Skill usage ratios, showing no collapse.

Self-Play. Sampling a separate opponent policy for each of the 4,096 environments is prohibitively expensive in GPU memory, so we use grouped opponent sampling. We divide the 4,096 parallel environments into 10 groups, each of which samples one opponent policy shared by all environments in the group. Thus, training requires only the current policy and 10 frozen opponent policies to reside on the GPU simultaneously. We add the current policy to the opponent pool every 2,000 updates and resample each group’s opponent every 200 updates.

## 5.2 GGSD LEARNS TO PLAY GAMES WELL

We first examine whether our self-play procedure is stable and whether it produces agents that can effectively play the given games. Figure 3 reports the performance of the final policy, trained for 70k updates, against checkpoints from 2k to 70k updates. For each of three independent training runs, we evaluate each checkpoint pair over 1,000 games and average the results.

Self-play produces increasingly competitive agents. The final policy achieves high win rates against policies from early stages of training. As the opponent checkpoint becomes more recent, the win rate decreases while the draw rate increases, indicating that the policies become increasingly competitive. Importantly, the combined win-and-draw rate remains above 50% against every evaluated checkpoint. For G1Boxing, the final policy achieves over 80% win rate against every past checkpoint except itself at 70k, suggesting that training has not yet fully saturated. As training approaches a plateau, stronger recent opponents should lead to a more gradual decline in win rate.

The learned agent also performs strongly against human players. Four users each played five games against the final 70k checkpoint on AntSumo, achieving only two human wins (10% human win rate). Overall, these results suggest that training improves the policy against the opponent pool as a whole, rather than overfitting to a particular opponent checkpoint.

Figure 4 further shows how frequently each learned skill is selected during gameplay. Across all games, the high-level policy does not collapse to a single skill; instead, it consistently makes use of multiple skills. This indicates that the discovered skill set remains behaviorally relevant during competitive play.

## 5.3 QUALITATIVE ANALYSIS OF LEARNED SKILLS

We next examine what motor behaviors emerge from GGSD. Figure 5 visualizes all skills learned in each game. For visualization, we fix a single skill at the beginning of an episode, execute it continuously for five seconds, and overlay snapshots of the resulting motion.

GGSD discovers highly interpretable skills. The learned skills exhibit clear and readily interpretable behavioral semantics. For AntSumo, the agents discover a variety of turning, locomotion, and pushing behaviors. In G1Boxing, the humanoid learns basic locomotion primitives as well as task-specific behaviors such as punching and guarding. For example, one skill raises the arms in front of the upper body, resembling a defensive guard against incoming punches.

These qualitative patterns are consistent across random seeds. Although different seeds do not necessarily produce identical skill sets, they repeatedly recover behaviors that are important for successful gameplay, such as locomotion and striking. This suggests that the game objective provides a consistent behavioral structure while still allowing multiple solutions to emerge.

![](images/89fcbfcc08b104302f8821970272c6ca284a931040437a5ec694ffbfe2af09d0.jpg)  
Figure 5: Learned motor skills (top) and combo behaviors through skill transitions (bottom).

![](images/133bf512f4dba32ce2507d45392ad27a3130e05e98a56c013d9d327e34dda84b.jpg)

![](images/a1bb68cd0db23aa80419e6e7c4a7cd758308efb01b2de89c1b9e380ac966e86c.jpg)

![](images/80c7d1b72d5de6b42848ce9f30930b765451803a683cc609a4085a454232e12d.jpg)  
Figure 6: Semantic diversity of learned skills across different embodiments. Higher is better.

Skill transitions produce emergent combo behaviors. A particularly interesting phenomenon is that meaningful behaviors can emerge not only from individual skills but also from transitions between them. This is especially apparent in FrankaAirHockey. Individual Franka skills are relatively static: when executed in isolation, each skill moves the end effector toward a particular region of the table and then remains there. However, transitions between these discrete skills generate rapid and purposeful movements that are used to strike the puck.

A similar effect appears in G1Boxing. For example, transitioning from skill 5 to skill 2 produces a backward lean used to evade or absorb punches, whereas the reverse transition produces a strong forward punch by rapidly shifting momentum. These examples show that discrete skills act as compositional building blocks whose transitions produce richer behaviors than the individual skills alone.

State-dependent skill semantics. Emergent combos reveal an interesting tension between semantic consistency and compositional expressivity: the effect of a skill can depend on the state induced by preceding skills. Nevertheless, the resulting controllers remain playable, suggesting that consistency need not hold globally. Instead, each skill can retain predictable semantics within a small number of state regimes, while transitions between regimes enable qualitatively different behaviors. This expands the expressivity of a compact skill set without increasing the number of human inputs, although too many regimes could make the interface difficult to interpret. We discuss this further in Appendix A.

![](images/73c26fb003e96ea2e8d1670dbbe84bcc21a4f09871d33f109b243acf185fa411.jpg)  
Figure 7: Successful human-play trajectories on unseen tasks. Red dots indicate cube in AntCubePush, and robot in Maze.

Table 1: Human play results.
<table><tr><td>Task</td><td></td><td>Succ. Time(s)</td></tr><tr><td>AntMaze 100%</td><td></td><td>82.7</td></tr><tr><td>AntCube</td><td>100%</td><td>62.2</td></tr><tr><td>G1Maze</td><td>84%</td><td>22.0</td></tr></table>

## 5.4 QUANTITATIVE ANALYSIS AGAINST BASELINES

We quantitatively evaluate how effectively game-based guidance promotes semantically distinct skills. Distinctiveness alone is insufficient. Two skills may be distinct while both lack clear semantic meaning.

We therefore adapt the language distance metric of Rho et al. (2025b) to a vision-language setting. For each skill, we sample rollout videos at 5 fps, use Gemini-3.6-Flash to generate a natural-language behavior description, and compute pairwise distances between skill descriptions using Gemini-Embedding-2. For each training run, we average these pairwise distances to obtain a semantic-diversity score, and report the mean across three runs. Results are averaged over three seeds, with prompting and evaluation details provided in the Appendix E. We compare against DIAYN, DADS (Sharma et al., 2019), and METRA (Park et al., 2024). Since METRA uses a continuous skill space, we train it with a 5-dimensional skill and evaluate five skills corresponding to the one-hot basis vectors.

GGSD learns semantically distinct skills. As shown in Figure 6, GGSD achieves the highest semantic diversity across all embodiments. It is approximately 2× higher than the strongest baseline on Ant and G1, and 28% higher on Franka. These results indicate that game-based guidance promotes not only diverse motions, but semantically distinct behaviors.

## 5.5 HUMAN PLAY ON UNSEEN DOWNSTREAM TASKS

Finally, we evaluate whether skills learned through gameplay can be directly reused by humans on unseen tasks. We keep the learned low-level policies fixed and replace the high-level policy with human input, without additional training.

Using the policy learned from AntSumo, we evaluate AntMaze and AntCubePush; we also evaluate the G1Boxing policy on G1Maze. In all tasks, users observe a top-down view and select among the learned skills in real time.

GGSD skills are directly playable by humans. We conducted a human evaluation with five participants, including two authors, each performing five trials per task.<sup>1</sup> Before evaluation, participants read brief control tips and practiced for at most five minutes. Table 1 reports the success rate and average completion time across all participants and trials. The average success rate is at least 84% on all three tasks. Figure 7 shows representative trajectories from the user study. AntCubePush requires object manipulation and the maze tasks introduce wall contacts, neither of which is present in the training games. Moreover, all policies are trained in Isaac Lab (Mittal et al., 2025) but evaluated in MuJoCo (Todorov et al., 2012) web. Despite these task and simulator shifts, humans can compose a small set of GGSD skills to reliably navigate and manipulate objects without additional policy optimization.

## 6 CONCLUSION

We investigated whether self-play in 1v1 competitive games can produce reusable motor skills that are directly playable by humans. The learned skills support interactive gameplay and can be reused by humans without additional training to solve unseen downstream tasks. Our results suggest that games provide a simple form of guidance for discovering diverse, reusable, and human-playable skills.

Limitations and future work. A central limitation of GGSD is that the learned skill repertoire is bounded by the skills required by the game. Simple games may demand only a narrow set of behaviors, limiting transfer to downstream tasks. For example, a policy trained on G1Boxing is unlikely to acquire dexterous manipulation skills. Scaling GGSD to games that require broader and more diverse motor capabilities may enable more generic skill repertoires that transfer across a wider range of downstream tasks, which we view as an important direction for future work.

## REFERENCES

Bowen Baker, Ingmar Kanitscheider, Todor Markov, Yi Wu, Glenn Powell, Bob McGrew, and Igor Mordatch. Emergent tool use from multi-agent autocurricula. In International Conference on Learning Representations, 2020.

Trapit Bansal, Jakub Pachocki, Szymon Sidor, Ilya Sutskever, and Igor Mordatch. Emergent complexity via multi-agent competition. In International Conference on Learning Representations, 2018.

George W. Brown. Iterative solution of games by fictitious play. In T. C. Koopmans (ed.), Activity Analysis ofProduction and Allocation. Wiley, New York, 1951.

Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, and Sergey Levine. Diversity is all you need: Learning skills without a reward function. In International Conference on Learning Representations, 2019.

Karol Gregor, Danilo Jimenez Rezende, and Daan Wierstra. Variational intrinsic control. arXiv preprint arXiv:1611.07507, 2016.

Tuomas Haarnoja, Ben Moran, Guy Lever, Sandy H. Huang, Dhruva Tirumala, Jan Humplik, Markus Wulfmeier, Saran Tunyasuvunakool, Noah Y. Siegel, Roland Hafner, Michael Bloesch, Kristian Hartikainen, Arunkumar Byravan, Leonard Hasenclever, Yuval Tassa, Fereshteh Sadeghi, Nathan Batchelor, Federico Casarini, Stefano Saliceti, Charles Game, Neil Sreendra, Kushal Patel, Marlon Gwira, Andrea Huber, Nicole Hurley, Francesco Nori, Raia Hadsell, and Nicolas Heess. Learning agile soccer skills for a bipedal robot with deep reinforcement learning. Science Robotics, 9(89):eadi8022, 2024. doi: 10.1126/scirobotics.adi8022.

Johannes Heinrich and David Silver. Deep reinforcement learning from self-play in imperfectinformation games. arXiv preprint arXiv:1603.01121, 2016.

Johannes Heinrich, Marc Lanctot, and David Silver. Fictitious self-play in extensive-form games. In Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings ofMachine Learning Research, pp. 805–813. PMLR, 2015.

Paul Jansonnie, Bingbing Wu, Julien Perez, and Jan Peters. Unsupervised skill discovery for robotic manipulation through automatic task generation. In 2024 IEEE-RAS 23rd International Conference on Humanoid Robots (Humanoids), pp. 926–933. IEEE, 2024. doi: 10.1109/ Humanoids58906.2024.10769879.

Hyunseung Kim, Byungkun Lee, Hojoon Lee, Dongyoon Hwang, Donghu Kim, and Jaegul Choo. Do’s and don’ts: Learning desirable skills with instruction videos. Advances in Neural Information Processing Systems, 37:47741–47766, 2024.

Taehwan Kwon. Variational intrinsic control revisited. arXiv preprint arXiv:2010.03281, 2020.

Michael Laskin, Hao Liu, Xue Bin Peng, Denis Yarats, Aravind Rajeswaran, and Pieter Abbeel. Unsupervised reinforcement learning with contrastive intrinsic control. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=9HBbWAsZxFt.

Alexander Li, Carlos Florensa, Ignasi Clavera, and Pieter Abbeel. Sub-policy adaptation for hierarchical reinforcement learning. In International Conference on Learning Representations, 2020.

Mayank Mittal, Pascal Roth, James Tigue, Antoine Richard, Octi Zhang, Peter Du, Antonio Serrano-Muñoz, Xinjie Yao, René Zurbrügg, Nikita Rudin, Lukasz Wawrzyniak, Milad Rakhsha, Alain Denzler, Eric Heiden, Ales Borovicka, Ossama Ahmed, Iretiayo Akinola, Abrar Anwar, Mark T. Carlson, Ji Yuan Feng, Animesh Garg, Renato Gasoto, Lionel Gulich, Yijie Guo, M. Gussert, Alex Hansen, Mihir Kulkarni, Chenran Li, Wei Liu, Viktor Makoviychuk, Grzegorz Malczyk, Hammad Mazhar, Masoud Moghani, Adithyavairavan Murali, Michael Noseworthy, Alexander Poddubny, Nathan Ratliff, Welf Rehberg, Clemens Schwarke, Ritvik Singh, James Latham Smith, Bingjie Tang, Ruchik Thaker, Matthew Trepte, Karl Van Wyk, Fangzhou Yu, Alex Millane, Vikram Ramasamy, Remo Steiner, Sangeeta Subramanian, Clemens Volk, CY Chen, Neel Jawale, Ashwin Varghese Kuruttukulam, Michael A. Lin, Ajay Mandlekar, Karsten Patzwaldt, John Welsh, Huihua Zhao, Fatima Anes, Jean-Francois Lafleche, Nicolas Moënne-Loccoz, Soowan Park, Rob Stepinski, Dirk Van Gelder, Chris Amevor, Jan Carius, Jumyung Chang, Anka He Chen, Pablo de Heras Ciechomski, Gilles Daviet, Mohammad Mohajerani, Julia von Muralt, Viktor Reutskyy, Michael Sauter, Simon Schirm, Eric L. Shi, Pierre Terdiman, Kenny Vilella, Tobias Widmer, Gordon Yeoman, Tiffany Chen, Sergey Grizan, Cathy Li, Lotus Li, Connor Smith, Rafael Wiltz, Kostas Alexis, Yan Chang, David Chu, Linxi "Jim" Fan, Farbod Farshidian, Ankur Handa, Spencer Huang, Marco Hutter, Yashraj Narang, Soha Pouya, Shiwei Sheng, Yuke Zhu, Miles Macklin, Adam Moravanszky, Philipp Reist, Yunrong Guo, David Hoeller, and Gavriel State. Isaac lab: A gpu-accelerated simulation framework for multi-modal robot learning. arXiv preprint arXiv:2511.04831, 2025. URL https://arxiv.org/abs/2511.04831.

Inseok Oh, Seungeun Rho, Sangbin Moon, Seongho Son, Hyoil Lee, and Jinyun Chung. Creating pro-level ai for a real-time fighting game using deep reinforcement learning. IEEE Transactions on Games, 14(2):212–220, 2021.

Seohong Park, Jongwook Choi, Jaekyeom Kim, Honglak Lee, and Gunhee Kim. Lipschitzconstrained unsupervised skill discovery. In International Conference on Learning Representations, 2021.

Seohong Park, Oleh Rybkin, and Sergey Levine. Metra: Scalable unsupervised rl with metric-aware abstraction. In International Conference on Learning Representations, 2024.

Xue Bin Peng, Yunrong Guo, Lina Halper, Sergey Levine, and Sanja Fidler. Ase: Large-scale reusable adversarial skill embeddings for physically simulated characters. ACM Transactions on Graphics, 41(4), 2022.

Seungeun Rho, Kartik Garg, Morgan Byrd, and Sehoon Ha. Unsupervised skill discovery as exploration for learning agile locomotion. In Conference on Robot Learning, pp. 2678–2694. PMLR, 2025a.

Seungeun Rho, Laura Smith, Tianyu Li, Sergey Levine, Xue Bin Peng, and Sehoon Ha. Language guided skill discovery. In The Thirteenth International Conference on Learning Representations, 2025b.

Seungeun Rho, Aaron Trinh, Danfei Xu, and Sehoon Ha. Reference grounded skill discovery. In The Fourteenth International Conference on Learning Representations, 2026.

Julia Robinson. An iterative method of solving a game. Annals of Mathematics, 54(2):296–301, 1951. doi: 10.2307/1969530.

Archit Sharma, Shixiang Gu, Sergey Levine, Vikash Kumar, and Karol Hausman. Dynamics-aware unsupervised discovery of skills. arXiv preprint arXiv:1907.01657, 2019.

David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. Mastering the game of go with deep neural networks and tree search. Nature, 529(7587):484–489, 2016. doi: 10.1038/ nature16961.

David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, and Demis Hassabis. A general reinforcement learning algorithm that masters chess, shogi, and go through self-play. Science, 362(6419):1140–1144, 2018. doi: 10.1126/science. aar6404.

Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pp. 5026–5033. IEEE, 2012.

Oriol Vinyals, Igor Babuschkin, Wojciech M. Czarnecki, Michaël Mathieu, Andrew Dudzik, Junyoung Chung, David H. Choi, Richard Powell, Timo Ewalds, Petko Georgiev, Junhyuk Oh, Dan Horgan, Manuel Kroiss, Ivo Danihelka, Aja Huang, Laurent Sifre, Trevor Cai, John P. Agapiou, Max Jaderberg, Alexander S. Vezhnevets, Rémi Leblond, Tobias Pohlen, Valentin Dalibard, David Budden, Yury Sulsky, James Molloy, Tom L. Paine, Caglar Gulcehre, Ziyu Wang, Tobias Pfaff, Yuhuai Wu, Roman Ring, Dani Yogatama, Dario Wünsch, Katrina McKinney, Oliver Smith, Tom Schaul, Timothy Lillicrap, Koray Kavukcuoglu, Demis Hassabis, Chris Apps, and David Silver. Grandmaster level in starcraft ii using multi-agent reinforcement learning. Nature, 575(7782): 350–354, 2019. doi: 10.1038/s41586-019-1724-z.

Jungdam Won, Deepak Gopinath, and Jessica Hodgins. Control strategies for physically simulated characters performing two-player competitive sports. ACM Transactions on Graphics (TOG), 40 (4):1–11, 2021.

Ruize Zhang, Sirui Xiang, Zelai Xu, Feng Gao, Shilong Ji, Wenhao Tang, Wenbo Ding, Chao Yu, and Yu Wang. Mastering multi-drone volleyball through hierarchical co-self-play reinforcement learning. In Proceedings of The 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pp. 5278–5300. PMLR, 2025.

Algorithm 1 Game-Guided Skill Discovery (GGSD)   
Require: π<sub>high</sub>, π<sub>low</sub>, discriminator $q _ { \phi } ,$ , opponent pool $\Pi _ { K }$ , latest-opponent probability p, skill du  
ration $k , \breve { \lambda } _ { \mathrm { M I } } , \lambda _ { \mathrm { H } }$   
1: for each training iteration do   
2: Construct the opponent distribution $\rho _ { K }$ as in Sec. 4.1   
3: Initialize rollout buffers $\mathcal { D } _ { \mathrm { h i g h } }$ and $\mathcal { D } _ { \mathrm { l o w } }$   
4: for each rollout environment or environment group do   
5: Sample opponent index $i \sim \rho _ { K }$ and fix $\pi _ { \mathrm { f o e } }  \pi ^ { ( i ) }$   
6: for each high-level decision step t do   
7: Sample $z \sim \pi _ { \mathrm { h i g h } } ( \cdot \mid s _ { t } ^ { \mathrm { a l l } } )$   
8: Execute z for up to k environment steps using $\pi _ { \mathrm { l o w } } ( \cdot  { \mid } s ^ { \mathrm { p r o p r i o } } , z )$   
9: Compute per-step MI and low-level rewards according to Eqs. 15 and 17   
10: Store low-level transitions in $\mathcal { D } _ { \mathrm { l o w } }$   
11: Accumulate the discounted game reward over the executed skill interval and store   
the resulting high-level transition in $\mathcal { D } _ { \mathrm { h i g h } }$   
12: end for   
13: end for   
14: Update $q _ { \phi }$ on $\mathcal { D } _ { \mathrm { l o w } }$ by maximizing $\mathbb { E } [ \log q _ { \phi } ( z \mid s ^ { \mathrm { p r o p r i o } } ) ]$   
15: Update $\pi _ { \mathrm { l o w } }$ and $V _ { \mathrm { l o w } }$ with PPO using the low-level reward in Eq. 17   
16: Update $\pi _ { \mathrm { h i g h } }$ and $V _ { \mathrm { h i g h } }$ with PPO using the high-level target in Eq. 12 and entropy regular  
ization   
17: if checkpoint interval is reached then   
18: Add the current hierarchical policy to the opponent pool and increment K   
19: end if   
20: end for

## A STATE-DEPENDENT SKILL SEMANTICS AND EMERGENT COMBOS

Our experiments reveal a tension between semantic consistency and compositional expressivity. For direct human control, each discrete skill should have a predictable meaning. Pressing the same button should generally produce the same type of behavior. At the same time, we intentionally keep the skill set small to preserve a simple interface, which limits the behaviors that individual skills can represent.

Skill transitions provide a natural way to increase expressivity without adding more buttons. Because the low-level policy is state-conditioned, $\pi _ { \mathrm { l o w } } ( a \mid s , z )$ , the behavior induced by a skill z can depend on the state from which it is executed. One skill may move the agent into a particular region of the state space, from which another skill produces a qualitatively different behavior. This gives rise to combo behaviors that do not appear when either skill is executed in isolation.

This does not necessarily require abandoning semantic consistency. Instead, the state space can be viewed as containing a small number of behaviorally distinct regimes,

$$
\begin{array} { r } { S = S _ { 1 } \cup \cdots \cup S _ { R } , } \end{array}\tag{18}
$$

within which each skill remains relatively predictable. Skill semantics can therefore be locally consistent within a regime while changing after the agent transitions to another regime. This resembles a compact finite-state controller, where transitions between regimes allow the same small set of skill inputs to realize a broader behavioral repertoire.

G1Boxing provides a concrete example. The backward-leaning pose forms a distinctive regime relative to ordinary locomotion and boxing states. From regular standing states, the learned skills produce behaviors such as walking, guarding, and punching. After entering the backward-leaning state, however, a subsequent skill can exploit the stored posture and momentum to produce a sub stantially stronger forward punch.

This interpretation is compatible with our mutual-information objective. The discriminator encourages different skill codes to induce distinguishable behaviors, but does not require a skill to produce exactly the same motion from every state. Thus, state-dependent effects can coexist with distinguishable semantics as long as the resulting behaviors remain sufficiently predictable and separable.

There is nevertheless a limit to this mechanism. If the state space were fragmented into too many context-dependent regimes, the same button could acquire too many meanings, making the controller difficult to understand. Our results instead suggest a useful middle ground: a small number of state-dependent regimes can increase compositional expressivity while retaining a compact and in terpretable interface. Understanding this trade-off between semantic consistency, compositionality, and human playability is an interesting direction for future work.

## B AN ALTERNATIVE VIEW OF MUTUAL INFORMATION IN SKILL DISCOVERY

Mutual information (MI) is often treated as a central objective for skill discovery. Here, we offer a complementary interpretation: rather than viewing MI as the primary mechanism that creates meaningful behaviors, we view it mainly as an association mechanism that assigns distinct behavior to latent skill codes.

Once different skills induce behaviors that are sufficiently distinguishable for Z to be inferred from the resulting behavior, the MI objective is largely satisfied. It does not by itself require those behaviors to be semantically meaningful, useful, natural, or even substantially different. Small but easily detectable behavioral differences may be sufficient. From this perspective, MI more naturally answers ‘which behavior should belong to which latent code?” than ‘which behaviors should be discovered?” The latter requires an additional source of behavioral structure. Below, we examine five skill-discovery approaches that use MI as a central component and ask what additional source of behavioral structure each method provides. As we will see, the nature of this additional guidance largely determines the kinds of skills that ultimately emerge.

DIAYN: task-agnostic exploration. DIAYN (Eysenbach et al., 2019) combines MI maximiza tion with maximum-entropy RL. Entropy encourages broad, task-agnostic exploration, while MI partitions the resulting behavioral variation across latent codes. In simple agents, this may yield recognizable skills such as different locomotion directions. In high-dimensional agents, however, many easily distinguishable behaviors need not correspond to coherent or useful motor skills. Under this view, DIAYN relies primarily on unguided exploration to generate behavioral variation and MI to organize it.

ASE: imitation guidance. ASE (Peng et al., 2022) combines an MI-based skill objective with adversarial imitation from motion data. The imitation objective constrains learning toward natural human motion, while MI organizes this behavioral space into a controllable latent representation. Thus, imitation largely determines learned skills, whereas MI helps make them controllable through z.

RGSD: structured latent guidance. Reference-Grounded Skill Discovery (RGSD) (Rho et al., 2026) provides behavioral structure by grounding the latent representation in reference motions. Once this representation is structured, a DIAYN-style reward can encourage the policy to visit states associated with a particular reference, effectively turning an MI-form objective into an imitationlike signal. This illustrates how the behavioral effect of MI depends strongly on the representation surrounding it.

SDAX: locomotion-task guidance. SDAX (Rho et al., 2025a) combines skill discovery with explicit task rewards for challenging locomotion behaviors such as leaping, climbing, and obstacle traversal. Task rewards direct exploration toward useful regions of behavior space, while the skill objective encourages different latent codes to capture distinct solutions. Skill discovery therefore acts as strategic exploration rather than diversity for its own sake.

GGSD: competitive-game guidance. GGSD follows the same broad perspective but obtains behavioral structure from competitive self-play. To defeat evolving opponents, the agent must discover effective ways to move, evade, strike, block, reposition, and interact with the environment. Because these behaviors emerge in service of winning, the resulting distinctions are naturally tied to strategically meaningful actions rather than arbitrary behavioral variation. Skills such as punching, guarding, turning, or striking a puck emerge because they are useful for winning, not because they are explicitly specified.

MI then organizes these behaviors into a compact discrete skill set. The game objective determines what kinds of behaviors are useful to discover, while MI encourages different latent codes to represent distinguishable motor behaviors. Without the game objective, MI has no preference for semantically meaningful behaviors over arbitrary distinguishable motion; without MI, self-play need not organize effective behaviors into individually controllable skills.

This perspective suggests a common decomposition across these methods:

$$
\underbrace { \mathrm { b e h a v i o r a l ~ g u i d a n c e } } _ { \mathrm { w h a t } \ : \mathrm { b e h a v i o r s ~ e m e r g e } } \quad + \quad \underbrace { \mathrm { \bf M I - b a s e d ~ a s s o c i a t i o n } } _ { \mathrm { w h i c h } \ : z \ : \mathrm { r e p r e s e n t s ~ t h e m } } .\tag{19}
$$

Under this view, the central contribution of GGSD is not a new mechanism for maximizing mutual information, but a new source of guidance for skill discovery: competitive games provide a lightweight objective from which strategically useful and semantically meaningful motor behaviors can emerge, while MI organizes them into the discrete skill vocabulary required for direct human control.

## C ENVIRONMENT DETAILS

All environments are symmetric two-player games between “me” and “foe”. Both agents receive observations with identical layouts under swapped roles, allowing a single policy to control either side. Following the notation in the main text, the high-level policy and critics receive the full game state $s ^ { \mathrm { a l l } }$ , while the low-level actor receives only the controlled agent’s proprioceptive observation $s ^ { \mathrm { p r o p r i o } }$ . Thus, s<sup>proprio</sup> forms a subset of $s ^ { \mathrm { a l l } }$

Components marked with ✓ are included in s<sup>proprio</sup> and are therefore available to the low-level actor. The remaining components are available only through s<sup>all</sup>. Unless noted otherwise, positions are expressed relative to the per-environment origin and velocities in the agent’s body frame.

The discriminator receives the same proprioceptive information as the low-level actor, except that the previous-action dimensions are removed in AntSumo and G1Boxing.

Table 2: Observation-space dimensions.
<table><tr><td>Environment</td><td> $s ^ { \mathrm { p r o p r i o } }$ </td><td> $s ^ { \mathrm { a l l } }$ </td><td># Skills</td></tr><tr><td>Ant Sumo</td><td>35</td><td>91</td><td>5</td></tr><tr><td>Ant Fencing</td><td>35</td><td>91</td><td>5</td></tr><tr><td>Franka Hockey</td><td>23</td><td>61</td><td>5</td></tr><tr><td>G1 Boxing</td><td>86</td><td>194</td><td>6</td></tr></table>

## C.1 ANT SUMO

Two quadruped ants compete in an 8 m $\times 8$ m square arena; a player loses when it is pushed out of the arena. Observation and reward are presented in Table 3 and 4, respectively.

Table 3: Ant Sumo observation space. Components marked with ✓ are included in $s ^ { \mathrm { p r o p r i o } } ;$ all components together form $s ^ { \mathrm { a l l } }$
<table><tr><td>Group</td><td>Component</td><td>Dim.</td><td> $s ^ { \mathrm { p r o p r i o } }$ </td></tr><tr><td rowspan="9">Me</td><td>Base height z</td><td>1</td><td>√</td></tr><tr><td>Base linear velocity (body frame)</td><td>3</td><td>V</td></tr><tr><td>Base angular velocity (body frame)</td><td>3</td><td>√</td></tr><tr><td>Base yaw</td><td>1</td><td>√</td></tr><tr><td>Projected gravity (body frame)</td><td>3</td><td>V</td></tr><tr><td>Joint positions (normalized)</td><td>8</td><td>√</td></tr><tr><td>Joint velocities (×0.2)</td><td>8</td><td>√</td></tr><tr><td>Previous action</td><td>8</td><td>V</td></tr><tr><td>Global position  $( x , y )$ </td><td>2</td><td></td></tr><tr><td>Me (global)</td><td>Base roll</td><td>1</td><td></td></tr><tr><td>Relative</td><td>Foe position relative to me (world frame, xyz)</td><td>3</td><td></td></tr><tr><td rowspan="9">Foe</td><td>Global position  $( x , y )$ </td><td>2</td><td></td></tr><tr><td>Base height z</td><td>1</td><td></td></tr><tr><td>Base linear velocity (body frame)</td><td>3</td><td></td></tr><tr><td>Base angular velocity (body frame)</td><td>3</td><td></td></tr><tr><td>Base yaw and roll</td><td>2</td><td></td></tr><tr><td>Projected gravity (body frame)</td><td>3</td><td></td></tr><tr><td>Joint positions (normalized)</td><td>8</td><td></td></tr><tr><td>Joint velocities (×0.2)</td><td>8</td><td></td></tr><tr><td>Previous action</td><td>8</td><td></td></tr><tr><td rowspan="6">Strategic</td><td>Planar distance to foe</td><td>1</td><td></td></tr><tr><td>Foe relative position  $( x , y )$  in me&#x27;s body frame</td><td>2</td><td></td></tr><tr><td>Foe relative velocity  $( x , y )$  in me&#x27;s body frame</td><td>2</td><td></td></tr><tr><td>Normalized distances to the four arena walls</td><td>4</td><td></td></tr><tr><td>Relative heading (sin ∆ψ, cos ∆ψ)</td><td>2</td><td></td></tr><tr><td>Normalized episode progress  $t / T$ </td><td>1</td><td></td></tr><tr><td></td><td>Total</td><td></td><td></td></tr><tr><td></td><td>Total</td><td>91 35</td><td></td></tr></table>

Table 4: Reward terms for Ant Sumo.
<table><tr><td>Term</td><td>Description</td><td>Expression Weight</td><td></td></tr><tr><td>Approach</td><td>Decrease in distance to the opponent</td><td> $d _ { t - 1 } - d _ { t }$ </td><td>2.0</td></tr><tr><td>Push</td><td>Decrease in the opponent&#x27;s distance to the nearest edge</td><td> $w _ { t - 1 } - w _ { t }$ </td><td>2.0</td></tr><tr><td>Win</td><td>Opponent leaves the arena or falls (once, terminal)</td><td> $\mathbf { 1 } [ \mathrm { w i n } ]$ </td><td>+20</td></tr><tr><td>Lose</td><td>Opponent wins</td><td>1[lose]</td><td>-20</td></tr><tr><td>Timeout</td><td>Draw at 10 s (once, terminal)</td><td>1[draw]</td><td>-3</td></tr><tr><td colspan="2">Alive, escape, upright, action, joint-velocity</td><td></td><td>0</td></tr></table>

## C.2 ANT FENCING

Ant Fencing uses the same observation space as Ant Sumo (Table 3). The game uses the same arena and agent configuration, but a player additionally loses when the opponent’s front foot touches its torso with sufficient contact force. The observation space is identical to that of Ant Sumo. Reward terms are presented in Table 5.

Table 5: Reward terms for Ant Fencing. The reward is identical to Ant Sumo except that the push term is disabled. Arena $8 \times 8 \mathrm { m }$ , 10 s episodes, random initial heading.
<table><tr><td>Term</td><td>Description</td><td>Expression Weight</td><td></td></tr><tr><td>Approach</td><td>Decrease in distance to the opponent</td><td> $d _ { t - 1 } - d _ { t }$ </td><td>2.0</td></tr><tr><td>Win</td><td>Front foot hits the opponent&#x27;s torso  $( \| F \| > 1 \mathrm { N } )$  , or opponent leaves the arena / falls</td><td> $\mathbf { 1 } [ \mathrm { w i n } ]$ </td><td>+20</td></tr><tr><td>Lose</td><td>Opponent&#x27;s front foot hits the agent&#x27;s torso, or agent leaves the arena / falls</td><td>1[lose]</td><td>-20</td></tr><tr><td>Timeout</td><td>Draw at 10 s</td><td>1[timeout]</td><td>-3</td></tr></table>

## C.3 FRANKA AIR HOCKEY

Two 7-DoF Franka arms face each other across an air-hockey table and attempt to shoot a puck into the opponent’s goal; the first player to score two goals wins the match. All positions and velocities are expressed in the agent’s own table frame, whose +x axis points toward the opponent’s goal. For the second player, the world x and y axes are negated so that both players observe the game from the same viewpoint. Joint velocities are scaled by 0.1. The end-effector (EE) yaw is measured relative to the agent’s facing direction and encoded as (sin ψ, cos ψ). The low-level actor observes only the controlled arm state, while the puck, opponent, and match state are included only in $s ^ { \mathrm { a l l } }$

Table 6: Franka Air Hockey observation space. Components marked with $\checkmark$ are included in $s ^ { \mathrm { p r o p r i o } }$ all components together form $s ^ { \mathrm { a l l } }$
<table><tr><td>Group</td><td>Component</td><td>Dim.</td><td> $s ^ { \mathrm { p r o p r i o } }$ </td></tr><tr><td rowspan="6">Me</td><td>Arm joint positions</td><td>7</td><td>√</td></tr><tr><td>Arm joint velocities (×0.1)</td><td>7</td><td>√</td></tr><tr><td>EE position (table frame)</td><td>3</td><td>√</td></tr><tr><td>EE linear velocity (table frame)</td><td>3</td><td>√</td></tr><tr><td>EE yaw (sin ψ, cos ψ)</td><td>2</td><td>√</td></tr><tr><td>EE yaw rate</td><td>1</td><td>√</td></tr><tr><td rowspan="6">Puck</td><td>Puck position (table frame)</td><td>3</td><td></td></tr><tr><td>Puck linear velocity (table frame)</td><td>3</td><td></td></tr><tr><td>Puck angular velocity (table frame)</td><td>3</td><td></td></tr><tr><td>Puck position relative to me&#x27;s EE</td><td>3</td><td></td></tr><tr><td>Foe goal center relative to puck (x, y)</td><td>2</td><td></td></tr><tr><td>Foe EE position (table frame)</td><td>3</td><td></td></tr><tr><td rowspan="3">Foe</td><td>Foe EE linear velocity (table frame)</td><td>3</td><td></td></tr><tr><td>Foe arm joint positions</td><td></td><td></td></tr><tr><td>Foe arm joint velocities (×0.1)</td><td>7 7</td><td></td></tr><tr><td rowspan="4">Match</td><td></td><td></td><td></td></tr><tr><td>Me score (normalized by score-to-win)</td><td>1</td><td></td></tr><tr><td>Foe score (normalized by score-to-win)</td><td>1</td><td></td></tr><tr><td>Round progress Normalized match progress  $t / T$ </td><td>1</td><td></td></tr><tr><td></td><td> $\mathbf { T o t a l } \ s ^ { \mathrm { a l l } }$ </td><td>1 61</td><td></td></tr></table>

Table 7: Reward terms for Franka Air Hockey. A match lasts 12 s and is won by the first player to score two goals; if the puck rests for 3 s inside (0.5 s outside) the workspace, the match ends with the current score. $x ^ { p }$ is the puck position along the axis toward the opponent’s goal.
<table><tr><td>Term</td><td>Description</td><td>Expression</td><td>Weight</td></tr><tr><td>Puck progress</td><td>Forward puck displacement toward the opponent&#x27;s goal (backward motion ignored)</td><td> $\operatorname* { m a x } ( 0 , x _ { t } ^ { p } - x _ { t - 1 } ^ { p } )$ </td><td>10.0</td></tr><tr><td>Goal</td><td>Agent scores (once per goal)</td><td>1[goal]</td><td>+40</td></tr><tr><td>Concede</td><td>Opponent scores (once per goal)</td><td>1[concede]</td><td>-40</td></tr><tr><td>Match win</td><td>Two goals first, or leading at timeout / idle-puck termination</td><td>1[match win]</td><td>+80</td></tr><tr><td>Match lose</td><td>Mirror of match win</td><td>1 [match lose]</td><td>-80</td></tr><tr><td>Draw</td><td>Tied at termination</td><td>1[draw]</td><td>0</td></tr><tr><td colspan="4">Action, workspace penalties</td></tr></table>

## C.4 G1 BOXING

Two Unitree G1 humanoids with 23 actuated joints compete in a $1 0 \mathrm { m } \times 1 0$ m arena. Successful strikes to the opponent’s torso or head reduce its hit points (HP), and the player with more remaining HP at timeout wins.

Table 8: G1 Boxing observation space. Components marked with ✓ are included in $s ^ { \mathrm { p r o p r i o } }$ ; all components together form $s ^ { \mathrm { a l l } }$
<table><tr><td>Group</td><td>Component</td><td>Dim.</td><td> $s ^ { \mathrm { p r o p r i o } }$ </td></tr><tr><td rowspan="10">Me</td><td>Base yaw</td><td>1</td><td>√</td></tr><tr><td>Base height z</td><td>1</td><td>√</td></tr><tr><td>Base linear velocity (body frame)</td><td>3</td><td>√</td></tr><tr><td>Base angular velocity (body frame)</td><td>3</td><td>V</td></tr><tr><td>Projected gravity (body frame)</td><td>3</td><td>√</td></tr><tr><td>Glove positions (heading frame,  $2 \times 3 )$ </td><td>6</td><td>V</td></tr><tr><td>Joint positions</td><td>23</td><td>√</td></tr><tr><td>Joint velocities (×0.2)</td><td>23</td><td>√</td></tr><tr><td>Previous action</td><td>23</td><td>√</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Me (global)</td><td>Global position (x, y)</td><td>2</td><td></td></tr><tr><td rowspan="2">Relative Foe gloves</td><td>Foe position relative to me (heading frame, xyz)</td><td>3</td><td></td></tr><tr><td>Foe glove positions (me&#x27;s torso frame,  $2 \times 3 )$  Foe glove velocities (me&#x27;s torso frame, 2 × 3)</td><td>6 6</td><td></td></tr><tr><td rowspan="9"></td><td>Global position (x, y)</td><td>2</td><td></td></tr><tr><td>Base yaw</td><td>1</td><td></td></tr><tr><td>Base height z</td><td>1</td><td></td></tr><tr><td>Base linear velocity (body frame)</td><td>3</td><td></td></tr><tr><td>Base angular velocity (body frame)</td><td>3</td><td></td></tr><tr><td>Projected gravity (body frame)</td><td>3</td><td></td></tr><tr><td>Glove positions (heading frame, 2 × 3)</td><td>6</td><td></td></tr><tr><td>Joint positions</td><td>23</td><td></td></tr><tr><td>Joint velocities (×0.2)</td><td>23</td><td></td></tr><tr><td></td><td>Previous action</td><td>23</td></tr><tr><td rowspan="3">Match</td><td>Me HP (normalized by initial HP)</td><td>1</td><td></td></tr><tr><td>Foe HP (normalized by initial HP)</td><td>1</td><td></td></tr><tr><td>Normalized time remaining  $1 - t / T$ </td><td>1</td><td></td></tr><tr><td></td><td> $\mathbf { T o t a l } \ s ^ { \mathrm { a l l } }$ </td><td></td><td></td></tr><tr><td></td><td>Total</td><td>194 86</td><td></td></tr></table>

Table 9: Reward terms for G1 Boxing, adapted from Won et al. Won et al. (2021). Arena $1 0 \times 1 0 \mathrm { m }$ 15 s episodes, initial HP $H _ { 0 } = 1 0 ^ { 5 }$ per player. Punch damage is $1 5 0 \bar { v } _ { \parallel } ^ { 2 }$ , where $\bar { v } _ { \perp }$ is the glove’s closing velocity normal to the struck face clipped at 12 m/s (head hits ${ \bar { \times 1 . 5 } } ) ;$ it drains the victim’s HP. The outcome is decided by HP alone; a fall or arena exit voids the round. uˆ is the unit vector toward the opponent, $\mathbf { f } _ { \mathrm { t o r s o } } , \mathbf { f } _ { \mathrm { p e l v i s } }$ are the forward axes of the torso and pelvis, $\mathbf { v } _ { \perp }$ is the root XY velocity rotated by 90<sup>◦</sup> (speed clipped at 1 m/s), q˜ is the joint position normalised to $[ - 1 , 1 ]$ within its soft limits, and $F _ { i j }$ are self-contact forces.
<table><tr><td>Term</td><td>Description</td><td>Expression</td><td>Weight</td></tr><tr><td>Punch</td><td>HP drained from the opponent minus HP lost</td><td> $( \Delta H _ { t } ^ { \mathrm { o p p } } - \Delta H _ { t } ^ { \mathrm { m e } } ) / H _ { 0 }$ </td><td>125</td></tr><tr><td>Facing</td><td>Torso and pelvis facing the opponent, in [—2, 2]</td><td> $\mathbf { f } _ { \mathrm { t o r s o } } { \cdot } \hat { \mathbf { u } } + \mathbf { f } _ { \mathrm { p e l v i s } } { \cdot } \hat { \mathbf { u } }$ </td><td>0.03</td></tr><tr><td>Facing velocity</td><td>Penalises moving sideways relative to the body&#x27;s facing</td><td> $- \big ( | \mathbf { f } _ { \mathrm { t o r s o } } \cdot \mathbf { v } _ { \perp } | + | \mathbf { f } _ { \mathrm { p e l v i s } } \cdot \mathbf { v } _ { \perp } | \big )$ </td><td>0.15</td></tr><tr><td>Approach</td><td>Decrease in distance to the opponent; zero when  $d _ { t } \leq 1  { \mathrm { m } }$ </td><td> $d _ { t - 1 } - d _ { t }$ </td><td>3.0</td></tr><tr><td>Joint velocity</td><td>Squared joint velocity, clipped at ±30 rad/s</td><td> $- \textstyle \sum _ { j } \mathrm { c l i p } ( \dot { q } _ { j } ) ^ { 2 }$ </td><td> $3 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Joint limit</td><td>Excess beyond 80 % of the soft joint range,</td><td> $\begin{array} { r } { - \sum _ { j } \mathrm { c l i p } \big ( \operatorname* { m a x } ( | \tilde { q } _ { j } | - 0 . 8 , 0 ) \big ) ^ { 2 } } \end{array}$ </td><td>0.5</td></tr><tr><td>Action rate</td><td>clipped at 0.3 Squared change of the action</td><td> $- \| \mathbf { a } _ { t } - \mathbf { a } _ { t - 1 } \| ^ { 2 }$ </td><td>0.005</td></tr><tr><td>Self-contact</td><td>Sum of self-collision forces, clipped at 50 N per pair</td><td> $- \sum _ { ( i , j ) } \mathrm { c l i p } ( \Vert F _ { i j } \Vert )$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Win / Lose</td><td>Opponent&#x27;s HP reaches zero first, or higher HP at</td><td> $\mathbf { 1 } [ \mathrm { w i n } ] / \mathbf { 1 } [ \mathrm { l o s e } ]$ </td><td> $+ 3 0 / - 3 0$ </td></tr><tr><td>Draw</td><td>timeout (mirror for lose) Simultaneous HP-out or equal HP at timeout</td><td>1[draw]</td><td>-5</td></tr><tr><td>Arena out</td><td>Agent falls (torso below 0.65 m) or leaves the arena</td><td>1 [arena out]</td><td>-3</td></tr></table>

## D HYPERAPARAMETERS

We provide the hyperparameters used in our experiments.

Table 10: Training hyper-parameters of the hierarchical self-play policies.
<table><tr><td>Hyper-parameter</td><td>Ant Sumo</td><td>Ant Fencing</td><td>Franka Hockey</td><td>G1 Boxing</td></tr><tr><td>PPO</td><td></td><td></td><td></td><td></td></tr><tr><td>Parallel environments</td><td></td><td>4096</td><td></td><td></td></tr><tr><td>Rollout length (steps / env)</td><td>32</td><td>32</td><td>24</td><td>32</td></tr><tr><td>Training iterations</td><td></td><td></td><td>70000</td><td></td></tr><tr><td>Learning epochs / mini-batches</td><td></td><td></td><td></td><td></td></tr><tr><td>Learning rate</td><td></td><td></td><td> $\begin{array} { c } { { 5 / 4 } } \\ { { 1 \times 1 0 ^ { - 4 } } } \end{array}$ </td><td></td></tr><tr><td>Discount γ</td><td>0.99</td><td>0.99</td><td>0.995</td><td>0.99</td></tr><tr><td>GAE λ</td><td></td><td></td><td>0.95</td><td></td></tr><tr><td>Clip ratio €</td><td></td><td></td><td>0.2</td><td></td></tr><tr><td>Value loss coefficient (clipped)</td><td></td><td></td><td>1.0</td><td></td></tr><tr><td>Max gradient norm</td><td></td><td></td><td>1.0</td><td></td></tr><tr><td>Entropy coefficient, low-level</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td></td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Entropy coefficient, high-level</td><td>0.02</td><td>0.02</td><td> $^ { 1 } _ { 0 . 0 2 } $ </td><td>0.05</td></tr><tr><td>Initial action std, low-level</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.2</td></tr><tr><td>Action clipping</td><td>10</td><td>10</td><td>10</td><td>1</td></tr><tr><td>Control frequency (Hz)</td><td>60</td><td>60</td><td>60</td><td>30</td></tr><tr><td>Seed</td><td></td><td></td><td>42</td><td></td></tr><tr><td>Hierarchy and skill discovery</td><td></td><td></td><td></td><td></td></tr><tr><td>Number of skills K</td><td>5</td><td>5</td><td>5</td><td>6</td></tr><tr><td>Skill duration</td><td></td><td></td><td>10</td><td></td></tr><tr><td>MI reward scale β</td><td>0.02</td><td>0.02</td><td>0.005</td><td>0.05</td></tr><tr><td>Discriminator learning rate</td><td></td><td></td><td> $1 \times 1 0 ^ { - 4 }$ </td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Self-play opponent pool</td><td></td><td></td><td></td><td></td></tr><tr><td>Warm-up iterations (mirror self-play)</td><td>0</td><td>0</td><td>2000</td><td>2000</td></tr><tr><td>Opponent snapshot interval (iterations)</td><td>2000</td><td>2000</td><td>2000</td><td>1000</td></tr><tr><td>Opponent resampling interval (iterations)</td><td></td><td></td><td>200</td><td></td></tr><tr><td>Opponent groups</td><td>10</td><td>10</td><td>4</td><td>10</td></tr><tr><td>Fraction of groups using the latest policy</td><td>0.6</td><td>0.6</td><td>0.6</td><td>0.3</td></tr></table>

Network architecture. All networks are multilayer perceptrons with ELU activations. For the Ant Sumo, Ant Fencing, and G1 Boxing tasks, the high-level policy has three hidden layers of [512, 512, 256] units and outputs a categorical distribution over the K skills; the low-level policy and both critics (one for each level) have four hidden layers of [512, 512, 256, 128] units, and the skill discriminator has two hidden layers of 256 units. For Franka Hockey, the high-level policy, the low-level policy, the critics, and the discriminator all use four hidden layers of [512, 512, 512, 256] units. The low-level policy receives only the agent’s own proprioceptive state together with the one-hot skill and outputs the mean of a Gaussian over joint actions with a learned state-independent standard deviation; for G1 Boxing the mean is additionally bounded with a tanh.

## E VLM-BASED SKILL DIVERSITY METRIC

Pipeline. For every trained policy we record one clip per skill (Ant: 5, Franka: 5, G1: 6 skills; for METRA a skill is the one-hot unit vector along each latent axis). Each clip is a single-environment rollout that starts from a reset state and runs for 5 s, rendered at $1 9 2 0 \times 1 0 8 0$ . The clip is captioned by a vision-language model (VLM), the caption is embedded with a text-embedding model, and the diversity score of a policy is the mean pairwise cosine distance

$$
D = { \frac { 1 } { \binom { K } { 2 } } } \sum _ { i < j } { \bigl ( } 1 - \cos ( \mathbf { e } _ { i } , \mathbf { e } _ { j } ) { \bigr ) }
$$

over the K skill embeddings $\mathbf { e } _ { i }$ (L2-normalized). The prompt asks the VLM to answer no semantic meaning when a clip shows no interpretable behavior; every pair containing such a clip is assigned distance 0, so uninterpretable skills do not add diversity. Scores are averaged over 3 training seeds (mean ± standard deviation).

Models and settings. Captioning uses gemini-3.6-flash (temperature 0.1, at most 1024 output tokens). The video is sampled by the API at 5 fps over a fixed segment of the clip: Ant 1–4 s, Franka 2–4 s, G1 0–5 s. The segment skips the initial settling phase after the reset where the robot has not yet started to move. Captions are embedded with gemini-embedding-001 (task type SEMANTIC\_SIMILARITY, 3072 dimensions). The marker no semantic meaning is matched case-insensitively as a substring, since the model occasionally wraps it in quotes or a full sentence.

Prompts. One prompt is used per robot; the same prompt is applied to every method on that robot.

• Ant. “This is a rendered clip of a simulated robot. Describe the semantic meaning of the robot’s behavior in one sentence. Ifyou cannot think ofa meaningful downstream taskfor which this behavior could be useful, say ‘no semantic meaning’.”

• Franka. “This is a rendered clip of a simulated robot arm. Describe which region of the table surface the robot arm moves over. Ifthe robot arm moves toofar awayfrom the table, or would otherwise be unlikely to meaningfully interact with objects on the table, say ‘no semantic meaning’.”

• G1. “This is a rendered clip of a humanoid robot with a sphere attached to each hand. Describe the semantic meaning ofthe robot’s behavior in one sentence. Ifthe behavior has no clear semantic meaning, respond with exactly ‘no semantic meaning’.”

Example captions. Ant (ours): “The quadruped robot performs a self-righting maneuver to flip itself over and stand back up on its feet.”, “The robot is rotating in place.”, “The robot is crawling forward.” Franka (ours): “The robot arm remains stationary over the bottom-right region of the table surface.” G1 (ours): “The humanoid robot is walking backwards.”