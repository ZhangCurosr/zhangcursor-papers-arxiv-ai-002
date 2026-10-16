# Learning Rewards in Strategic Games

Antoine Bergerault

Cyrille Kone

Giorgia Ramponi

ALPI Lab, University of Zurich {bergerault,kone,ramponi}@ifi.uzh.ch

## Abstract

Understanding the objectives that drive agents’ behavior is crucial in strategic environments, where multiple agents interact and adapt to one another. This challenge arises in domains such as cybersecurity, autonomous driving, and online markets, where agents’ underlying incentives are typically unknown and must be inferred from their observed behavior. Inverse reinforcement learning (IRL) provides a natural framework for this problem, but extending IRL to strategic multi-agent settings introduces a fundamental challenge: a reward that explains observed behavior does not necessarily induce optimal strategies when used for decision-making. In this work, we study this problem through the lens of strategic identifiability, and show that learning from a fixed policy pair can be insuficient to strategically identify rewards, independent of the number of expert samples K. Motivated by this, we introduce an active IRL algorithm for two-player repeated Stackelberg games, in which the leader adaptively selects policies to elicit informative responses about the follower’s reward. Combining maximum-likelihood estimation with experimental design, we derive finite-sample guarantees on the Stackelberg sub-optimality induced by the learned reward, matching up to polylogarithmic factors a new minimax learning rate of K<sup>−1/2</sup>. Finally, we evaluate our approach in a realistic anti-poaching setting, demonstrating the benefits of active exploration for reward recovery and downstream strategic performance.<sup>1</sup>

## 1 Introduction

Strategic decision-making often requires reasoning about agents whose objectives are unknown. A defender may need to anticipate the incentives of an attacker, an autonomous vehicle the preferences of other drivers, or a wildlife-protection agency the objectives of poachers. In such settings, agents’ behavior provides an indirect signal about their underlying incentives. Recovering these objectives is valuable not only for explaining observed behavior, but, crucially, for predicting how agents may respond to new policies or interventions.

Inverse reinforcement learning (IRL) (Ng et al., 2000; Abbeel & Ng, 2004) provides a natural framework for inferring latent reward functions from the behavior of approximately optimal agents. While IRL has been extensively studied in single-agent decision-making, strategic interactions fundamentally change the problem. An agent’s observed behavior depends not only on its own reward, but also on the behavior of the agents it interacts with. Consequently, a reward that perfectly explains behavior under one interaction may fail to predict how the same agent responds when its opponent changes strategy. This raises a fundamental question for IRL in games:

when does observed behavior contain enough information about an agent’s reward to support reliable strategic decision-making?

Rather than requiring recovery of the ground-truth reward itself, we argue that reward learning should preserve the strategic consequences that matter at deployment. We study this question through the lens of strategic identifiability, relating ambiguity in the inferred reward to the sub-optimality incurred when using it for downstream strategic decision-making. We illustrate this objective in Figure 1 for a wildlife protection application, where an error in reward estimation from a ranger patrol can incur highly sub-optimal behavior at deployment.

Poacher reward estimation

![](images/9cdccce23dff2882849384312f453cf159e5d29c4f75942f949e390e6c78a9bb.jpg)  
Ground truth

![](images/1e33653e28210a001547ae1b91edc0a53f6b96e960309ea23ee65fb22aeae1d0.jpg)  
Reward error

Task 1: prioritize north  
![](images/6737079be37157072eaa85ae75f173decac2f375865410e1b4177565ad5c8632.jpg)  
Optimal behavior

![](images/8198a7ac42df63c1886df345ae14898e217ddc4770f24118c2883a9fe1463d76.jpg)  
Patrol coverage  
Deployment  
Reward

Task 2: prioritize south  
![](images/7a129fb0f8483c6d41a453d80858c0b05307a84d1f72a0e0a5f73259ea2e202d.jpg)  
Wildlife Ranger base  
Optimal behavior

![](images/7bd56f07752832fabbbeeecbdefc95133f1e333888d61b357a4be83526b1690c.jpg)  
Deployment  
Predicted poacher

Figure 1: Illustration of the impact of a poacher reward error on downstream tasks in a wildlife preservation setting. At deployment, a small error in reward estimation can lead to highly sub-optimal behavior from a rational patrol.

A key dificulty is that passive observations may provide insuficient information for this objective. Observing an agent repeatedly against a fixed policy only reveals its preferences under that particular interaction. Distinct reward functions may therefore generate indistinguishable behavior while inducing substantially diferent responses to alternative policies. As we show in this work, this can prevent low strategic sub-optimality even with infinitely many observations. This limitation suggests a diferent approach: rather than passively observing strategic behavior, can we actively choose interactions that reveal the reward information needed for downstream decision-making?

We answer this question in repeated Stackelberg games, where a leader can adaptively select policies and observe the responses of a bounded-rational follower. This setting naturally allows the leader to trade of exploiting its current estimate of the follower’s reward with selecting interactions that are informative about the follower’s underlying objectives. We develop an active IRL approach that combines maximum-likelihood reward estimation with experimental design to progressively resolve strategically relevant reward ambiguity, and we provide finite-sample guarantees on the strategic performance induced by the learned reward.

Contributions We summarize our main contributions as follows:

• We formalize the objective of IRL in games in terms of the strategic performance induced by the learned reward. We show that observations generated by a fixed policy pair can be insuficient to identify a strategically useful reward, even with infinitely many samples (Lemma 2), while on hard instances any adaptive sampling strategy must sufer a Stackelberg sub-optimality lower bounded by $\Omega ( K ^ { - 1 / 2 } )$ after K observations (Theorem 1).

• Motivated by this impossibility, we introduce an active IRL framework for repeated Stackelberg normal-form games, in which the leader adaptively selects policies to elicit informative responses from a bounded-rational follower with linear rewards. Our algorithm combines maximum-likelihood reward estimation with experimental design and achieves a high-probability Stackelberg sub-optimality of order $\tilde { \mathcal { O } } ( d / \sqrt { K } )$ after K interactions where d is the dimensionality of the follower features (Theorem 3). This matches the optimal dependence on K up to polylogarithmic factors.

• We then evaluate our approach on synthetic normal-form games and a realistic wildlife-protection environment. Our experiments demonstrate the benefits of actively selecting interactions for reward recovery and downstream strategic performance.

Related work Our work connects inverse reinforcement learning and learning in games. In single-agent settings, IRL has long studied the ambiguity inherent in recovering rewards from demonstrations (Ng et al., 2000; Abbeel & Ng, 2004). In multi-agent settings, recent work has focused on whether policies learned from demonstrations remain robust when interacting with strategic agents, typically through notions of exploitability (Tang et al., 2024; Freihaut et al., 2026a; Bergerault et al., 2026a).

Emerging work on IRL in games has instead characterized sets of rewards compatible with observed strategic behavior (Freihaut & Ramponi, 2026; Barbara et al., 2026). We take a complementary perspective: rather than only characterizing which rewards are consistent with observed behavior, we ask which interactions provide the information needed to identify a strategically useful reward, and how such interactions can be actively selected with finite-sample guarantees.

A more extensive discussion on previous work can be found in Appendix A.

## 2 Preliminaries and formal setup

Notation For the rest of the paper, we will use the following notations. For any measurable space $x , \Delta ( \mathcal { X } )$ denotes the probability simplex over X. Moreover, we define the Shannon entropy $\begin{array} { r } { \mathcal { H } _ { \mathcal { X } } : \Delta ( \mathcal { X } )  \mathbb { R } \mathrm { ~ a s ~ } \mathcal { H } _ { \mathcal { X } } ( P ) = - \int \log ( P ( x ) ) \mathrm { d } P ( x ) } \end{array}$ , and we will use the simplified notation H when X is clear from context. For any positive definite matrix $P \in \mathbb { R } ^ { n \times n }$ and any $x \in \mathbb { R } ^ { n }$ , we denote its norm with respect to P as $\| x \| _ { P } = { \sqrt { x ^ { \top } P x } }$ . Finally, for any positive $N \in \mathbb { Z } _ { + }$ , we use the notation $[ N ] : = \{ 1 , \dots , N \}$ , and for any finite set S we denote by |S| its cardinality.

We define a two-player normal-form game as a tuple $\mathcal { M } : = ( A , B , r _ { 1 } , r _ { 2 } )$ where $A , B$ represent the finite action spaces of the two players, and for $i \in \{ 1 , 2 \}$ , we denote by $r _ { i } : \mathcal { A } \times \mathcal { B }  \mathbb { R }$ , $r _ { i } \in \mathcal { R } _ { i } \subseteq \mathbb { R } ^ { A \times B }$ the individual rewards. We denote by $\mu \in \Delta ( { \mathcal { A } } )$ and $\nu \in \Delta ( B )$ the policies of the two players, such that joint actions are sampled from the product distribution $\mu \otimes \nu \in \Delta ( \mathcal { A } \times \mathcal { B } )$

In this environment, player i aims to maximize the objective $J _ { r _ { i } } ( \mu , \nu ) = \mathbb { E } _ { ( a , b ) \sim ( \mu , \nu ) } \left[ r _ { i } ( a , b ) \right]$ such that given $\mu ,$ player 2 plays a best-response $\nu _ { \mu } ^ { * } ( r _ { 2 } ) \in \arg \operatorname* { m a x } _ { \tilde { \nu } } J _ { r _ { 2 } } ( \mu , \tilde { \nu } )$ , similarly for the µ-player.

## 2.1 Multi-agent learning objectives

Several metrics can be used to represent the sub-optimality of learning players in games.

Adapted from the single-agent learning literature (Cesa-Bianchi et al., 2006; Orabona, 2019), the regret measures the opportunity cost of each agent in the environment induced by the other player.

Definition 1 (Regret). We define the (external) regret of the players as follows:

$$
\mathrm { R e g } _ { r } ^ { 1 } ( \mu , \nu ) : = \operatorname* { m a x } _ { \mu ^ { * } } J _ { r _ { 1 } } ( \mu ^ { * } , \nu ) - J _ { r _ { 1 } } ( \mu , \nu ) , \quad \mathrm { R e g } _ { r } ^ { 2 } ( \mu , \nu ) : = \operatorname* { m a x } _ { \nu ^ { + } } J _ { r _ { 2 } } ( \mu , \nu ^ { * } ) - J _ { r _ { 2 } } ( \mu , \nu ) .
$$

Following Ramponi et al. (2023); Tang et al. (2024), recent multi-agent imitation learning works often aggregate these regrets $\mathrm { a s ^ { 2 } }$ NashGap $\operatorname { \lrcorner } ( \mu , \nu ) = \operatorname* { m a x } \left\{ \operatorname { R e g } _ { r } ^ { 1 } ( \mu , \nu ) , \operatorname { R e g } _ { r } ^ { 2 } ( \mu , \nu ) \right\}$ such that each pair $( \mu , \nu )$ satisfying Nash $\mathrm { G a p } _ { r } ( \mu , \nu ) \leq \epsilon$ is an ϵ-Nash equilibrium (Nash Jr, 1950).

This is appealing for two main reasons. First, when ofline methods are used as a bootstrap to further policy optimization (Macuglia et al., 2025; Wagenmaker et al., 2025) seeking a Nash equilibrium. Second, as a deployment guarantee with a fixed opponent, i.e. how much a player could improve by deviating when the other players are deployed without adapting to these deviations.

However, when learning and deploying in the presence of adaptive strategic players, regret notions are oblivious, as they ignore how other agents will react to a change of policy (Mertikopoulos, 2026). To better measure this strategic coupling between the agents, we therefore take the perspective of a Stackelberg setting (Sessa et al., 2020; Bai et al., 2021; Chen et al., 2025).

Under this view, a leader starts by committing to a policy $\mu ,$ based on which a follower decides on its own policy ν. Given $\mu ,$ , we assume that the follower is optimal by choosing $\nu = \nu _ { \mu } ^ { * } ( r _ { 2 } ) ^ { 3 }$ , i.e. the follower best-responds to $\mu$ with respect to its reward $r _ { 2 }$ . The goal of the leader is then to determine a policy $\mu ^ { \ast } \in$ arg max $\dot { \pi } _ { r } ^ { \mathrm { l e a d e r } } ( \tilde { \mu } )$ maximizing its expected rewards against an optimal follower:

$$
J _ { r } ^ { \mathrm { l e a d e r } } ( \mu ) : = J _ { r _ { 1 } } ( \mu , \nu _ { \mu } ^ { * } ( r _ { 2 } ) ) .
$$

From this formulation, we define our measure of sub-optimality as follows.

Definition 2 (Stackelberg sub-optimality). Given a leader policy $\mu ,$ we define its Stackelberg sub-optimality with respect to the reward r as:

$$
{ \mathrm { S u b O p t } } _ { r } ( \mu ) : = \operatorname* { m a x } _ { \mu ^ { * } } J _ { r } ^ { \mathrm { l e a d e r } } ( \mu ^ { * } ) - J _ { r } ^ { \mathrm { l e a d e r } } ( \mu ) .
$$

Compared to $\log _ { r } ^ { 1 } ( \mu , \nu )$ , this objective no longer measures the learner’s sub-optimality against a fixed $\nu ,$ but rather how the µ-player can improve when facing an adaptive follower.

Given K episodes, we state the goal of an inverse reinforcement learning algorithm as finding an approximate reward $\hat { r } _ { 2 }$ such that $\mathrm { S u b O p t } _ { r } ( \hat { \mu } )$ is minimal for $\hat { \mu } \in \arg \operatorname* { m a x } _ { \tilde { \mu } } J _ { ( r _ { 1 } , \hat { r } _ { 2 } ) } ^ { \mathrm { l e a d e r } } ( \tilde { \mu } )$ . In other words, an optimal leader for $\hat { r } _ { 2 }$ should be close to optimal under the true reward $r _ { 2 }$

## 2.2 Modeling assumptions

For finite action spaces, $\mathcal { R } _ { 2 }$ is in general the entire set of tabular rewards. However, in order to better model the reward function, we make the following popular modeling assumption.

Assumption 1 (Linear rewards). There exists a known feature map $\phi : \mathcal { A } \times \mathcal { B }  \mathbb { R } ^ { d } , S > 0$ and a parameter $\omega \in \Omega : = \Bigl \{ \tilde { \omega } \in \mathbb { R } ^ { d } : \| \tilde { \omega } \| _ { 2 } \leq S \Bigr \}$ for some $d \in \mathbb { Z } _ { + }$ such that:

$$
\boldsymbol { r } _ { 2 } ( \boldsymbol { a } , \boldsymbol { b } ) = \langle \phi ( \boldsymbol { a } , \boldsymbol { b } ) , \omega \rangle , \qquad \forall ( \boldsymbol { a } , \boldsymbol { b } ) \in \mathcal { A } \times \mathcal { B } .
$$

Additionally, we assume the features are normalized, i.e. $\operatorname* { m a x } _ { a , b } \lVert \phi ( a , b ) \rVert _ { 2 } \leq 1 . ^ { 4 }$

The problem of IRL therefore reduces to estimating the parameter $\omega$ from observed interactions. This precise structure of $r _ { 2 }$ is interpretable (the feature map is known) and allows our algorithm to optimize the information on the reward function via a Fisher information matrix on its coeficients.

As a consequence, we model our estimate $\hat { r } _ { 2 }$ as a linear map with features $\phi$ and parameter $\hat { \omega } _ { ; }$ and will use $\mathrm { S u b O p t } _ { \omega }$ and $\mathrm { S u b O p t } _ { \hat { \omega } }$ to denote SubOpt and $\mathrm { S u b O p t } _ { ( r _ { 1 } , \hat { r } _ { 2 } ) }$ , respectively.

Remark 1. Any reward can be linearly parametrized with features being the canonical basis $\operatorname { o f } \mathbb { R } ^ { | \boldsymbol { A } | \cdot | \boldsymbol { B } | }$ and coeficients $\omega _ { a b } = r _ { 2 } ( a , b )$ , therefore satisfying Assumption 1 with $d = | { \mathcal { A } } | \cdot | B |$ In general, d can be much smaller with a better choice of features.

Compared to the tabular setting, Assumption 1 therefore allows to learn at a faster statistical rate, as a single interaction can carry more information on the underlying reward.

Additionally, we assume an entropy regularized follower. This common assumption (Fu et al., 2017; Geist et al., 2019) is motivated by bounded rationality in human decision-making and policies.

Assumption 2 (Bounded rationality of the ν-player). The ν-player is entropy-regularized with inverse temperature parameter $\eta > 0 , \mathrm { i . e . } \ \nu _ { \mu } ^ { * } ( \tilde { r } ) = \mathrm { a r g } \operatorname* { m a x } _ { \tilde { \nu } } J _ { \tilde { r } } ( \mu , \tilde { \nu } ) + \eta ^ { - 1 } \mathcal { H } ( \tilde { \nu } )$

To make the setting learnable, we ignore the trivial case where the ν-player is completely independent of its reward<sup>5</sup>.

Assumption 3 (Non-degeneracy). There exists $\mu \in \Delta ( { \mathcal { A } } )$ and two linear rewards $r _ { 2 } , \tilde { r } _ { 2 } \in \mathcal { R } _ { 2 }$ with feature map $\phi$ such that $\nu _ { \mu } ^ { * } ( r _ { 2 } ) \neq \nu _ { \mu } ^ { * } ( \tilde { r } _ { 2 } )$

For notational convenience, we also define the bound on the µ-player rewards as follows.   
Note that $R _ { \mathrm { m a x } }$ is always well-defined since the action spaces are finite.

Definition 3. We denote $\begin{array} { r } { R _ { \operatorname* { m a x } } : = \| r _ { 1 } \| _ { \infty } , \mathrm { ~ i . e . ~ } R _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { a \in \mathcal A , \ b \in \mathcal B } \left| r _ { 1 } ( a , b ) \right| } \end{array}$

In the next section, we introduce the problem of reward identifiability and its geometric implications for learning from a fixed policy pair. Moving to the statistical perspective, we further derive a lower bound on the sub-optimality of any adaptive algorithm in the finite sample regime. Then, in Section 4, we derive a principled adaptive algorithm (Algorithm 1) for Stackelberg sub-optimality minimization in MA-IRL that matches our lower bound up to poly-logarithmic factors.

## 3 On reward identifiability in games

The goal of inverse reinforcement learning is to leverage demonstrations to extract a reward function inducing similar behaviors, and identifiability measures the ability of recovering such reward. The strongest notion aims to extract the exact reward $\hat { r } _ { 2 } = r _ { 2 }$ , whereas in multi-agent applications it would be enough to preserve strategic behaviors defined from a given learning objective (see Section 2.1).

In order to learn reward information, the approach relies on the assumption of some (approximately) optimal players. IRL for the ν-player therefore aims to find a function $\hat { r } _ { 2 }$ that belongs to the feasible reward set $\mathcal { R } _ { 2 } ^ { \mathrm { f e a s } } ( \mu , \nu _ { \mu } ^ { * } ) : = \left. \tilde { r } _ { 2 } \in \mathcal { R } _ { 2 } : \nu _ { \mu } ^ { * } = \nu _ { \mu } ^ { * } ( \tilde { r } _ { 2 } ) \right.$ for any sampled policy pair $( \mu , \nu _ { \mu } ^ { * } )$ , where sub-optimality can be measured using diferent learning objectives.

In the entropy-regularization setting, where the follower is boundedly-rational by trading-of between rewards and entropy, the relation between the best-response and the reward of the ν-player can be made explicit. In fact, a classical result (see e.g. Geist et al. (2019)) is the following:

Lemma 1. Under Assumption 2, the unique best-response $\nu _ { \mu } ^ { * }$ to the policy µ is such that:

$$
\nu _ { \mu } ^ { * } ( b ) = \frac { \exp \left( \eta \sum _ { a \in \mathcal { A } } \mu ( a ) r _ { 2 } ( a , b ) \right) } { \sum _ { b ^ { \prime } \in \mathcal { B } } \exp \left( \eta \sum _ { a \in \mathcal { A } } \mu ( a ) r _ { 2 } ( a , b ^ { \prime } ) \right) } , \qquad \forall b \in \mathcal { B } .\tag{1}
$$

Hence, the average reward $r _ { 2 } ^ { \mu } ( b ) : = \langle \mu , r _ { 2 } ( \cdot , b ) \rangle$ such that $\nu _ { \mu } ^ { * } = \operatorname { s o f t m a x } ( \eta \cdot r _ { 2 } ^ { \mu } )$ can be learned by inverting this multinomial logistic model, but only up to constant shifts (Freihaut & Ramponi, 2026).

Under Assumption 1 of a linear follower reward, this relation can be simplified. Specifically, letting $\begin{array} { r } { \psi _ { \mu } ( b ) = \sum _ { a \in \mathcal { A } } \phi ( a , b ) \mu ( a ) } \end{array}$ be the average feature vector of the ν-player in the environment induced by $\mu ,$ Liao et al. (2025) notes that $\omega$ is solution to the following linear system:

$$
\begin{array} { r } { \underbrace { \left[ \begin{array} { c } { \psi _ { \mu } ( b _ { 2 } ) ^ { \top } - \psi _ { \mu } ( b _ { 1 } ) ^ { \top } } \\ { \vdots } \\ { \psi _ { \mu } ( b _ { \vert B \vert } ) ^ { \top } - \psi _ { \mu } ( b _ { 1 } ) ^ { \top } } \end{array} \right] } _ { = : \Phi ( \mu ) \in \mathbb { R } ^ { ( \vert B \vert - 1 ) \times d } } \omega = \eta ^ { - 1 } \left[ \begin{array} { c } { \log \left( \nu _ { \mu } ^ { * } ( b _ { 2 } ) / \nu _ { \mu } ^ { * } ( b _ { 1 } ) \right) } \\ { \vdots } \\ { \log \left( \nu _ { \mu } ^ { * } ( b _ { \vert B \vert } ) / \nu _ { \mu } ^ { * } ( b _ { 1 } ) \right) } \end{array} \right] . } \end{array}\tag{2}
$$

From this relation, we can study diferent notions of identifiability for the reward function.

Recovering the exact reward function. As noticed by Liao et al. (2025); Rolland et al. (2022); Freihaut & Ramponi (2026), the linear system in Equation (2) is invertible if and only if rank $( \Phi ( \mu ) ) = d .$ . From basic linear algebra, one can already conclude that the reward cannot be recovered if $| B | - 1 < d$ . Intuitively, this holds because there always exists a non-zero $\omega _ { 0 } \in \mathbb { R } ^ { d }$ such that $\psi _ { \mu } ( b ) ^ { \intercal } \omega _ { 0 }$ is constant over all $b ,$ hence the constant shift ambiguity is not removed by the linear structure of Assumption 1.

When the rank condition holds, previous work such as Liao et al. (2025) learn the ordinary least squares solution of the system, estimating the target vector from samples. The final error bound on $\| \boldsymbol { \omega } - \boldsymbol { \hat { \omega } } \| _ { 2 }$ relies on concentration results scaling with the inverse minimum probability of actions, of order $| B | e ^ { 2 \eta S }$ in the worst-case. Crucially, this bound is in the standard Euclidean geometry and not weighted by the directions spanned by the features. If the rank condition is not satisfied, such a bound might never vanish and the standard workaround is to study convergence in Hausdorf distance between the true and recovered feasible sets (Metelli et al., 2021; Liao et al., 2025; Freihaut & Ramponi, 2026; Barbara et al., 2026).

In this work, we weaken this goal of exact identifiability and focus on learning a reward $\hat { r } _ { 2 }$ (approximately) preserving the strategic behaviors of the agents<sup>6</sup>.

Strategic identifiability of the reward. Toward this goal, we focus on learning a reward inducing (approximately) the same best-response followers as the true reward. Indeed, as we assume the ν-player to best-respond during training, we also assume this capability at deployment, and therefore take the Stackelberg sub-optimality $\operatorname { S u b O p t } _ { \ i }$ instead of $\mathrm { R e g } _ { r } ^ { 1 }$ as the most natural strategic measure for the learned policy.

Learning from a fixed policy pair, exact reward identifiability is impossible when the linear system of Equation (2) is not invertible. We extend below this observation to strategic identifiability, where no choice of $\mu$ allows to arbitrarily minimize sub-optimality. We report the proof in Appendix C.

Lemma 2. Fix $\eta > 0$ . There exists a set $( r _ { 1 } , \mathcal { A } , B , \phi )$ such that for any reference policy $\mu _ { r e f }$ and any number $o f$ samples $K$ , no algorithm only sampling from $( \mu _ { r e f } , \nu _ { \mu _ { r e f } } ^ { * } )$ can guarantee that $\mathrm { S u b O p t } _ { r } ( { \hat { \mu } } ) \leq c _ { \eta }$ with probability greater than $1 / 2 . \ c _ { \eta } > 0$ only depends on $\eta$

The above lemma is a geometric artifact built on the same degeneracy as exact nonidentifiability. This degeneracy makes it impossible to learn information that could help distinguish between two, strategically unequal coeficients, even in the infinite-data regime. Observations from a passive leader can therefore be insuficient to recover strategy-preserving rewards.

This motivates the need for active interaction strategies where the learner deliberately selects diferent leader policies to probe the follower in useful directions. In particular, these adaptively chosen directions should be both informative and strategically important for the reward coeficients. Under this framing, learning rewards remains constrained by the following statistical lower bound.

Theorem 1 (Minimax lower bound). Let $d \geq 3 2$ ， $\begin{array} { r } { \epsilon = \frac { 1 } { 2 \eta } \sqrt { d / K } \le S } \end{array}$ and $\boldsymbol \omega ^ { ( i ) } : = \varepsilon \boldsymbol e _ { i }$ for $i \in [ d ]$ . There exists a game such that for every algorithm producing an output $\hat { \mu }$ after $K \geq 4 d$ interactions,

$$
\operatorname* { m a x } _ { i \in [ d ] } \mathbb { E } ^ { ( i ) } \left[ \mathrm { S u b O p t } _ { \omega ^ { ( i ) } } ( \hat { \mu } ) \right] \ \geq \ \frac { 1 } { 4 4 } R _ { \operatorname* { m a x } } \sqrt { \frac { d } { K } } ,
$$

where $\mathbb { E } ^ { ( i ) }$ is for the expectation under the trajectory law $\mathbb { P } ^ { ( i ) }$ induced by $\boldsymbol { \omega } ^ { ( i ) }$

In the next section, we show how to achieve such a learning rate up to poly-logarithmic factors.

## 4 Active Stackelberg IRL in normal-form games

In many applications, indeed, one can actively interact with the environment over time. Examples include cybersecurity settings where defenders adapt to strategic attackers and online platforms interacting with self-interested users. In those cases, the learner is not restricted to passively observing a fixed behavior, but can deliberately choose policies inducing diverse environments for the ν-player.

This allows the learner to elicit informative responses from the follower, related in spirit to the strategy of Rolland et al. (2022), who modify the dynamics of a single-agent setting to make the reward exactly identifiable from expert demonstrations. Similarly, in our setting, varying the leader policies helps reduce estimation uncertainty in strategically relevant directions of the follower’s reward function.

Specifically, we consider a leader who interacts repeatedly with a (boundedly-)rational follower, for a total of N rounds. At each round $\tau \in [ N ]$ , the leader commits to a (potentially mixed) strategy $\mu _ { \tau } \in \Delta ( \mathcal { A } )$ , and observes a sample $b _ { \tau } \sim \nu _ { \tau }$ from the follower’s unique bestresponse $\nu _ { \tau } : = \nu _ { \mu _ { \tau } } ^ { \ast } ( r _ { 2 } )$ . After the N iterations, the goal is to return an estimate $\hat { \omega }$ such as to minimize $\mathrm { S u b O p t } _ { \omega } ( \hat { \mu } )$ where $\hat { \mu }$ is optimal under the estimated reward $\hat { r } _ { 2 } ( \cdot , \cdot ) = \langle \phi ( \cdot , \cdot ) , \hat { \omega } \rangle$ i.e. Su $\mathrm { b } \mathrm { O p t } _ { \hat { \omega } } ( \hat { \mu } ) = 0$

Denoting the average features $\begin{array} { r } { \psi _ { \mu _ { \tau } } ( b ) = \sum _ { a } \mu _ { \tau } ( a ) \phi ( a , b ) } \end{array}$ , Lemma 1 gives for all $b \in B$

$$
\nu _ { \tau } ( b ) = \frac { 1 } { Z _ { \mu _ { \tau } } ( \omega ) } \exp \left( \eta \langle \psi _ { \mu _ { \tau } } ( b ) , \omega \rangle \right) , \qquad Z _ { \mu _ { \tau } } ( \tilde { \omega } ) : = \sum _ { b ^ { \prime } \in B } \exp \left( \eta \langle \psi _ { \mu _ { \tau } } ( b ^ { \prime } ) , \tilde { \omega } \rangle \right) ,\tag{3}
$$

from which one can derive likelihood values for the $\omega$ . Our approach is based on this idea.

## 4.1 Combining optimal design and maximum likelihood estimation

For any history of n pairs $\mathcal { D } = \{ ( b _ { \tau } , \mu _ { \tau } ) \} _ { \tau = 1 } ^ { n }$ , let us denote the log-likelihood of $\tilde { \omega } \in \Omega$ as

$$
\mathcal { L } ( \tilde { \omega } | \mathcal { D } ) : = \sum _ { \tau = 1 } ^ { n } l ( \tilde { \omega } | b _ { \tau } , \mu _ { \tau } ) , \qquad l ( \tilde { \omega } | b _ { \tau } , \mu _ { \tau } ) : = \log { \mathbb { P } ( b _ { \tau } | \omega = \tilde { \omega } , \mu _ { \tau } ) } ,
$$

with the instantaneous log-likelihood $l ( \tilde { \omega } | b _ { \tau } , \mu _ { \tau } ) = \eta \langle \psi _ { \mu _ { \tau } } ( b _ { \tau } ) , \tilde { \omega } \rangle - \log Z _ { \mu _ { \tau } } ( \tilde { \omega } )$ . With the exponential family property of Equation (3), the Fisher information matrix of $\mu _ { \tau }$ also simplifies<sup>7</sup>:

$$
\begin{array} { r } { \mathcal { I } _ { \mu _ { \tau } } ( \tilde { \omega } ) : = - \nabla ^ { 2 } l ( \tilde { \omega } | \mu _ { \tau } ) = \eta ^ { 2 } \mathrm { C o v } _ { b \sim \mathbb { P } ( \cdot | \tilde { \omega } , \mu _ { k } ) } \left[ \psi _ { \mu _ { k } } ( b ) \right] , } \end{array}
$$

such that $\begin{array} { r } { - \nabla ^ { 2 } \mathcal { L } ( \tilde { \omega } | \mathcal { D } ) = \sum _ { \tau = 1 } ^ { n } \mathcal { T } _ { \mu _ { \tau } } ( \tilde { \omega } ) } \end{array}$ is the cumulated Fisher information of $\mathcal { D }$

Following these remarks, we derive an algorithm reducing uncertainty in the Fisher information geometry, by using this relation on the sub-optimality, whose proof is deferred to Appendix F.1.

Lemma 3 (Reward transfer lemma). Let $\mu ^ { * }$ and $\hat { \mu }$ be optimal leaders under ω and $\hat { \omega }$ . Then,

$$
\mathrm { S u b O p t } _ { \omega } ( \hat { \mu } ) \leq 2 R _ { m a x } \cdot \operatorname* { m a x } _ { \tilde { \mu } \in \{ \mu ^ { * } , \hat { \mu } \} } \chi \left( \eta \cdot d _ { \tilde { \mu } } ( \omega , \hat { \omega } ) \right) \cdot \| \omega - \hat { \omega } \| _ { \mathcal { T } _ { \tilde { \mu } } ( \hat { \omega } ) } ,
$$

$$
w i t h  \ \chi ( x ) : = 2 x ^ { - 1 } \left( e ^ { x / 2 } - 1 \right) f o r \ x > 0 \ a n d \ d _ { \tilde { \mu } } ( \omega , \hat { \omega } ) = \operatorname* { m a x } _ { b , d } \langle \psi _ { \tilde { \mu } } ( b ) - \psi _ { \tilde { \mu } } ( d ) , \omega - \hat { \omega } \rangle .
$$

Essentially, Lemma 3 shows how the sub-optimality aligns with an estimation problem in the Fisher information geometry. This alignment depends on $d _ { \tilde { \mu } } ( \omega , \hat { \omega } )$ , which represents the span semi-norm of the diference between logits induced by ω and $\hat { \omega }$

We optimize this upper bound by combining three ingredients, presented in the proof outline below.

Proof roadmap

Analyzing the second term, for any $\lambda > 0$ and ${ \mathcal { D } } \in ( { \mathcal { B } } \times \Delta ( { \mathcal { A } } ) ) ^ { n }$ , we change the measure as:

$$
\begin{array} { r } { \| \omega - \hat { \omega } \| _ { \mathcal { T } _ { \widetilde { \mu } } ( \hat { \omega } ) } ^ { 2 } \leq \underbrace { \| \omega - \hat { \omega } \| _ { G ^ { \lambda } ( \hat { \omega } ) } ^ { 2 } } _ { \mathrm { e s t i m a t i o n } } \cdot \underbrace { \mathrm { T r a c e } \left( \left[ G ^ { \lambda } ( \hat { \omega } ) \right] ^ { - 1 } \mathcal { T } _ { \widetilde { \mu } } ( \hat { \omega } ) \right) } _ { \mathrm { F i s h e r ~ u n c e r t a i n t y } } , } \end{array}
$$

for the regularized $G ^ { \lambda } ( \tilde { \omega } | \mathcal { D } ) : = \lambda I - \nabla ^ { 2 } \mathcal { L } ( \tilde { \omega } | \mathcal { D } )$

We optimize these terms with Maximum Likelihood Estimators (MLE) and optimal design.

(i) Estimation. Specifically, for $\begin{array} { r } { \{ \hat { \omega } _ { \tau } : = \arg \operatorname* { m a x } _ { \tilde { \omega } \in \Omega } \mathcal { L } ( \tilde { \omega } | \mathcal { D } _ { 1 : \tau } ) \} ^ { \otimes } } \end{array}$ the constrained MLE sequence, the estimation is controlled using the following confidence result from Lee et al. (2024).

Theorem 2 (Informal version of Theorem 5). With probability at least $1 - \delta ,$

$$
\| \omega - \hat { \omega } _ { \tau } \| _ { \Gamma _ { \tau } ^ { \lambda } ( \hat { \omega } _ { \tau } ) } ^ { 2 } \leq \mathcal { O } \left( \lambda S ^ { 2 } + \log \delta ^ { - 1 } + d \log \left( \operatorname* { m a x } \left\{ e , d ^ { - 1 } \tau \eta S \right\} \right) \right) , \quad \forall \tau \in [ n ] ,
$$

where we define $\begin{array} { r } { \Gamma _ { \tau } ^ { \lambda } ( \tilde { \omega } ) : = \int _ { 0 } ^ { 1 } ( 1 - s ) G ^ { \lambda } ( \tilde { \omega } + s ( \omega - \tilde { \omega } ) | \mathcal { D } _ { 1 : \tau } ) } \end{array}$ ds and fix $\delta \in ( 0 , 1 )$

(ii) Design. This holds independently of the policies in D, that we therefore choose to minimize the Fisher uncertainty, or equivalently maximize the Fisher coverage and select:

$$
\mu _ { \tau } \in \underset { \mu \in \Delta ( \mathcal { A } ) } { \arg \operatorname* { m a x } } \mathrm { T r a c e } \left( \left[ G ^ { \lambda } ( \hat { \omega } _ { k - 1 } \vert \mathcal { D } _ { 1 : \tau } ) \right] ^ { - 1 } \mathcal { T } _ { \mu } ( \hat { \omega } _ { \tau - 1 } ) \right) ,
$$

i.e. greedily optimize the information gain at each step.

(iii) Self-concordance and warm-up. We combine everything together by deriving new self-concordance properties for Fisher information matrices and gains expressed in terms of $d _ { \tilde { \mu } }$ , which can be of independent interest.

To force self-concordance into a harmless regime, we adapt a pure exploration warm-up strategy by generalizing Kiefer-Wolfowitz results to our setting where the leader only has partial control over the feature coverage. Specifically, we collect $n _ { \mathrm { w a r m } }$ samples from allocation computed ofline, and subsequently perform $K : = N - n _ { \mathrm { w a r m } }$ adaptive policy selections with the cumulative data:

$$
\mathscr { D } _ { k } : = \mathscr { D } _ { 1 : n _ { \mathrm { w a r m } } + k } = \mathscr { D } _ { \mathrm { w a r m } } \cup \{ \left( b _ { \tau ^ { \prime } } ^ { \mathrm { o n } } , \mu _ { \tau ^ { \prime } } ^ { \mathrm { o n } } \right) \} _ { \tau ^ { \prime } = 1 } ^ { k } , \qquad \forall k \in [ K ] ,
$$

where $\{ ( b _ { \tau ^ { \prime } } ^ { \mathrm { o n } } , \mu _ { \tau ^ { \prime } } ^ { \mathrm { o n } } ) \} _ { \tau ^ { \prime } = 1 } ^ { k }$ are collected online and ${ \mathcal { D } } _ { \mathrm { w a r m } } : = { \mathcal { D } } _ { 1 : n _ { \mathrm { w a r m } } }$ is from the warm-up phase.

This warm-up guarantees that all MLE terms $\{ \hat { \omega } _ { k } ^ { \mathrm { o n } } : = \hat { \omega } _ { n _ { \mathrm { w a r m } } + k } \} k \le K$ satisfy ma $\mathrm { \Large ~ c } _ { \mu } d _ { \mu } ( \omega , \hat { \omega } _ { k } ^ { \mathrm { o n } } ) \leq$ $1 / \eta$ simultaneously with high probability. This also bounds the first term of Lemma 3 and concludes the analysis.

## 4.2 An active algorithm with warm-up

As mentioned in the proof outline and in contrast to standard exploration that would maximize a bonus of the form $\| \phi ( \cdot , \cdot ) \| _ { ( \lambda I + \Lambda _ { k } ) ^ { - 1 } }$ with $\begin{array} { r } { \Lambda _ { k } : = \sum _ { \tau = 1 } ^ { k } \left\lceil \psi _ { \mu _ { \tau } } ( b _ { \tau } ) \psi _ { \mu _ { \tau } } ( b _ { \tau } ) ^ { \top } \right\rceil } \end{array}$ (e.g. LSVI-UCB (Jin et al., 2020)), we design our objective around minimizing uncertainty in strategically-relevant directions, measured by the Fisher information (Lemma 3). We summarize our approach in Algorithm 1.

After an initial burn-in phase with $n _ { \mathrm { w a r m } } \approx \tilde { \mathcal { O } } \left( \eta S e ^ { 2 \eta S } | \mathcal { B } | d ^ { 2 } \log \delta ^ { - 1 } \right)$ , we derive the following finite-sample Stackelberg sub-optimality guarantees.

Algorithm 1 Active Stackelberg IRL for two-player normal-form games   
1: Input: horizon $K \geq 0 .$ confidence δ, burn-in budget $n _ { \mathrm { w a r m } } \geq 0 .$ regularizer $\lambda \geq 0 ,$   
tolerance ϵ.   
2: Warm-up phase (simplified, see the full procedure as Algorithm 2 in Appendix E)   
3: Solve the following optimal design up to a factor $1 + \epsilon ,$ using proxy $\tilde { \omega } = 0 :$   
$w \in$ arg min max Trace $\Big ( W ( \tilde { w } ) ^ { \dagger } \mathcal { T } _ { \delta _ { a } } ( 0 ) \Big )$ , s.t. Im $\mathcal { T } _ { \delta _ { a } } ( 0 ) \subseteq$ Im $W ( \tilde { w } )$ ∀a $\in { \mathcal { A } } .$   
$\tilde { w } \in \Delta ( \mathcal { A } )$ a∈A   
where $\delta _ { a }$ is the pure policy on a and $W ( \tilde { w } ) : = \mathbb { E } _ { a ^ { \prime } \sim \tilde { w } } [ \mathcal { T } _ { \delta _ { a ^ { \prime } } } ( 0 ) ]$   
4: Round w to an allocation of $n _ { \mathrm { w a r m } }$ policies and collect ${ \mathcal { D } } _ { \mathrm { w a r m } } .$   
5: Adaptive online phase   
6: $\begin{array} { r } { \mathcal { D } _ { 0 } : = \mathcal { D } _ { \mathrm { w a r m } } , \hat { \omega } _ { 0 } ^ { \mathrm { o n } } \gets \arg \operatorname* { m a x } _ { \tilde { \omega } \in \Omega } \mathcal { L } ( \tilde { \omega } | \mathcal { D } _ { \mathrm { w a r m } } ) } \end{array}$ or arbitrary in Ω if $\mathcal { D } _ { \mathrm { w a r m } } = \emptyset .$   
7: for $k = 1 , 2 , \ldots , K$ do   
8: $\begin{array} { r } { \mu _ { k } ^ { \mathrm { o n } } \in \mathrm { a r g } \operatorname* { m a x } _ { \mu \in \Delta ( \mathcal { A } ) } \mathrm { T r a c e } \left( \left[ G ^ { \lambda } ( \hat { \omega } _ { k - 1 } ^ { \mathrm { o n } } | \mathcal { D } _ { k - 1 } ) \right] ^ { - 1 } \mathcal { T } _ { \mu } ( \hat { \omega } _ { k - 1 } ^ { \mathrm { o n } } ) \right) ^ { \mathrm { 9 } } } \end{array}$   
9: $\begin{array} { r } { b _ { k } \sim \nu _ { \mu _ { k } ^ { \mathrm { o n } } } ^ { * } ( \omega ) ; \quad \mathcal { D } _ { k }  \mathcal { D } _ { k - 1 } \cup \{ ( b _ { k } , \mu _ { k } ^ { \mathrm { o n } } ) \} ; \quad \hat { \omega } _ { k } ^ { \mathrm { o n } }  \arg \operatorname* { m a x } _ { \tilde { \omega } \in \Omega } ^ { ' } \mathcal { L } ( \tilde { \omega } | \mathcal { D } _ { k } ) } \end{array}$   
10: end for   
11: Output: $\hat { \omega } : = \hat { \omega } _ { K } ^ { \mathrm { o n } }$

Theorem 3 (Informal). Let ωˆ the result ofAlgorithm 1 with $n _ { w a r m } \approx \tilde { \mathcal { O } } \left( \eta S e ^ { 2 \eta S } | \mathcal { B } | d ^ { 2 } \log \delta ^ { - 1 } \right)$ $K \geq 1$ and $\lambda = 0$ . Then, with probability at least $1 - \delta ,$ , for any µˆ such that $\mathrm { S u b O p t } _ { \hat { \omega } } ( \hat { \mu } ) = 0 \dot { : }$

$$
\mathrm { S u b O p t } _ { \omega } ( \hat { \mu } ) \leq \tilde { \mathcal { O } } \left( \frac { R _ { m a x } \sqrt { d \left( d \eta S + \log { \delta ^ { - 1 } + d } \right) } } { \sqrt { K } } \right) ,
$$

where $\tilde { \mathcal { O } }$ hides polylog terms in log $\delta ^ { - 1 }$ , K and environment parameters.

The proof follows the following thread: after an initial warm-up phase, we ensure that every online MLE is suficiently localized near the true parameter ω. Given this, we proceed by a greedy online phase using the running estimators in place of the true unknown reward coeficient. Finally, we conclude with a classical elliptical potential bound argument. The formal version of Theorem 3 and its complete proof are in Appendix F.2.

Remark 2. Our main result assumes an initial warm-up phase for $n _ { \mathrm { w a r m } }$ exponential in the parameters η and S. This corresponds to the usual κ-dependency of the various previous works on preference-based learning and bandits with the logistic function (Faury et al., 2020; Abeille et al., 2021; Schlaginhaufen et al., 2026; Bergerault et al., 2026b).

While Theorem 3 is given for a specific $n _ { \mathrm { w a r m } }$ and $\lambda = 0$ , we note that in general and in practice, our algorithm can be run with smaller warm-up steps (or no warm-up at all), requiring $\lambda > 0$ for proper matrix inversion in the policy selection.

Lastly, by a standard reduction argument, this PAC result also yields the following:

$$
\mathbb { E } \left[ \mathrm { S u b O p t } _ { \omega } ( \hat { \mu } ) \right] \leq \tilde { \mathcal { O } } \left( R _ { \operatorname* { m a x } } \sqrt { d / K } \right) ,
$$

where the expectation is taken with respect to the follower best-responses.

This matches the lower bound in Theorem 1 up to polylogarithmic factors for suficiently large K.

## 5 Numerical validation

We empirically validate our algorithm on a realistic anti-poaching task from the GUARD benchmark (Krever et al., 2026). We simulate both active methods where the controllable leader interacts with a best-response follower and fixed-expert methods. These fixedexpert baselines rely on the strong assumption of a known Stackelberg or quantal-response equilibrium oracle that we sample from.

(a) Active methods  
![](images/1eb7cab394ce11d017048d9a1670ed36a0edab713fb2998797d0ec944e31ef5b.jpg)

(b) Fixed-expert methods  
![](images/c0e373c10dc2f9d5201df59c2c2f80433ddc3d39c3322a37b6dcf052ab2bb2cf.jpg)  
Number of observed poacher actions  
Figure 2: Mean SubOpt with the number of poacher actions. 95% CIs over 50 seeds.

In Figure 2, we compare our algorithm Active IRL with two active MLE baselines where a policy is sampled randomly at each step (Online random) or once at the beginning of training then fixed (Fixed random). Additionally, we compare to previous (fixed-expert) works with Zero-sum OLS<sup>10</sup> (derived from Liao et al. (2025)), Follower-only OLS and the version of MaxEntIRL in our setting as Fixed-expert MLE (derived from Ziebart et al. (2008)).

As shown, Active IRL approximately matches or improves over all the baselines, despite the structural advantage of fixed-expert methods. In terms of transfer of the learned rewards to multiple leader objectives, we provide some illustrative results in Figure 3 below for arbitrary downstream tasks<sup>11</sup>, comparing our algorithm with the second-to-best algorithm (Fixed-expert MLE).

![](images/731f631534cd4586c34b2610fdfc4b651f850e3c0e03edbb1d726d03f6094043.jpg)  
Figure 3: Patrol coverage and associated mean SubOpt across downstream tasks. Trained on 1,600 poacher actions. Darker map cells indicate greater coverage; the triangle marks the ranger base.

We provide additional details, experiments and ablations in Appendix H.

## 6 Conclusion

We study IRL for interactive agents in strategic games as the problem of learning reward functions that (approximately) preserve optimality of agents when deployed among strategic players. In two-player normal-form games under linearity and entropy-regularization assumptions, we develop an active algorithm that provably minimizes Stackelberg sub-optimality at a minimax optimal rate, ignoring polylogarithmic factors. We then empirically validate our algorithm on a realistic anti-poaching game and a set of synthetic environments.

Future directions A current open problem is principled learning in dynamic games. Specifically, open challenges include the derivation of exploration strategies over states and the computation of reward estimators from trajectories (beyond strong myopic assumptions of previous works).

## References

Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári. Improved algorithms for linear stochastic bandits. Advances in neural information processing systems, 24, 2011.

Pieter Abbeel and Andrew Y Ng. Apprenticeship learning via inverse reinforcement learning. In Proceedings of the twenty-first international conference on Machine learning, pp. 1, 2004.

Marc Abeille, Louis Faury, and Clément Calauzènes. Instance-wise minimax-optimal algorithms for logistic bandits. In International Conference on Artificial Intelligence and Statistics, pp. 3691–3699. PMLR, 2021.

Zeyuan Allen-Zhu, Yuanzhi Li, Aarti Singh, and Yining Wang. Near-optimal discrete optimization for experimental design: A regret minimization approach. Mathematical Programming, 186(1):439–478, 2021.

Zhiyu An and Wan Du. Diml: Diferentiable inverse mechanism learning from behaviors of multi-agent learning trajectories. arXiv preprint arXiv:2601.17678, 2026.

Francis Bach. Self-concordant analysis for logistic regression. Electronic Journal of Statistics, 4:384 – 414, 2010. doi: 10.1214/09-EJS521.

Yu Bai, Chi Jin, Huan Wang, and Caiming Xiong. Sample-eficient learning of stackelberg equilibria in general-sum games. In Proceedings of the 35th International Conference on Neural Information Processing Systems, NIPS ’21. Curran Associates Inc., 2021. ISBN 9781713845393.

Annalisa Barbara, Riccardo Poiani, Martino Bernasconi, and Andrea Celli. Optimal rates for feasible payof set estimation in games. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=qV7yFzdK1R.

Antoine Bergerault, Volkan Cevher, and Negar Mehr. Matching multiple experts: On the exploitability of multi-agent imitation learning. In The Fourteenth International Conference on Learning Representations, 2026a. URL https://openreview.net/forum? id=B9qYvPTbTN.

Antoine Bergerault, Anna Maria Maddux, Andreas Schlaginhaufen, and Maryam Kamgarpour. Preference-based distributed welfare maximization: A game-theoretic approach. In ICML 2026 Workshop on New Frontiers in Game-Theoretic Learning (NExT-Game), 2026b. URL https://openreview.net/forum?id=CAS7aInVos.

Haoyang Cao, Samuel Cohen, and Lukasz Szpruch. Identifiability in inverse reinforcement learning. Advances in Neural Information Processing Systems, 34:12362–12373, 2021.

Nicolo Cesa-Bianchi, Gábor Lugosi, et al. Prediction, learning, and games, volume 1. Cambridge university press Cambridge, 2006.

Jianguo Chen, Jinlong Lei, Biqiang Mu, Yiguang Hong, and Hongsheng Qi. Active inverse methods in stackelberg games with bounded rationality. arXiv preprint arXiv:2510.15582, 2025.

Siyu Chen, Mengdi Wang, and Zhuoran Yang. Actions speak what you want: Provably sample-eficient reinforcement learning of the quantal stackelberg equilibrium from strategic feedbacks. arXiv preprint arXiv:2307.14085, 2023.

Louis Faury, Marc Abeille, Clément Calauzènes, and Olivier Fercoq. Improved optimistic algorithms for logistic bandits. In International Conference on Machine Learning, pp. 3052–3060. PMLR, 2020.

Till Freihaut and Giorgia Ramponi. On feasible rewards in multi-agent inverse reinforcement learning. Advances in Neural Information Processing Systems, 38:114230–114284, 2026.

Till Freihaut, Luca Viano, Volkan Cevher, Matthieu Geist, and Giorgia Ramponi. Learning equilibria from data: Provably eficient multi-agent imitation learning. Advances in Neural Information Processing Systems, 38:97744–97794, 2026a.

Till Freihaut, Luca Viano, Emanuele Nevali, Volkan Cevher, Matthieu Geist, and Giorgia Ramponi. Rate optimal learning of equilibria from data. In The 29th International Conference on Artificial Intelligence and Statistics, 2026b.

Justin Fu, Katie Luo, and Sergey Levine. Learning robust rewards with adversarial inverse reinforcement learning. arXiv preprint arXiv:1710.11248, 2017.

Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized markov decision processes. In International conference on machine learning, pp. 2160–2169. PMLR, 2019.

Denizalp Goktas, Amy Greenwald, Sadie Zhao, Alec Koppel, and Sumitra Ganesh. Eficient inverse multiagent learning. arXiv preprint arXiv:2502.14160, 2025.

Martin Grötschel, László Lovász, and Alexander Schrijver. Geometric Algorithms and Combinatorial Optimization. Springer, 1988. URL http://eudml.org/doc/204222.

Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably eficient reinforcement learning with linear function approximation. In Conference on learning theory, pp. 2137– 2143. PMLR, 2020.

Jack Kiefer and Jacob Wolfowitz. The equivalence of two extremum problems. Canadian Journal of Mathematics, 12:363–366, 1960.

Noah Krever, Jakub Cerny, Moïse Blanchard, and Christian Kroer. Guard: Constructing realistic two-player matrix and security games for benchmarking game-theoretic algorithms. Advances in Neural Information Processing Systems, 38, 2026.

Joongkyu Lee and Min-Hwan Oh. Improved online confidence bounds for multinomial logistic bandits. In International Conference on Machine Learning, pp. 33576–33615. PMLR, 2025.

Jungyhun Lee, Se-Young Yun, and Kwang-Sung Jun. A unified confidence sequence for generalized linear models, with applications to bandits. Advances in Neural Information Processing Systems, 37:124640–124685, 2024.

Junyi Liao, Zihan Zhu, Ethan X Fang, Zhuoran Yang, and Vahid Tarokh. Decoding rewards in competitive games: Inverse game theory with entropy regularization. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaf, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 37610–37622. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/ liao25i.html.

David Lindner, Andreas Krause, and Giorgia Ramponi. Active exploration for inverse reinforcement learning. Advances in Neural Information Processing Systems, 35:5843– 5853, 2022.

Qinghua Liu, Yuanhao Wang, and Chi Jin. Learning markov games with adversarial opponents: Eficient algorithms and fundamental limits. In International Conference on Machine Learning, pp. 14036–14053. PMLR, 2022.

Maël Macuglia, Paul Friedrich, and Giorgia Ramponi. Fine-tuning behavioral cloning policies with preference-based reinforcement learning. arXiv preprint arXiv:2509.26605, 2025.

Negar Mehr, Mingyu Wang, Maulik Bhatt, and Mac Schwager. Maximum-entropy multiagent dynamic games: Forward and inverse solutions. IEEE transactions on robotics, 39 (3):1801–1815, 2023.

Panayotis Mertikopoulos. Regret, equilibrium, and learning in games: A guided tour. arXiv preprint arXiv:2608.09389, 2026.

Alberto Maria Metelli, Giorgia Ramponi, Alessandro Concetti, and Marcello Restelli. Provably eficient learning of transferable rewards. In International Conference on Machine Learning, pp. 7665–7676. PMLR, 2021.

John F Nash Jr. Equilibrium points in n-person games. Proceedings of the national academy of sciences, 36(1):48–49, 1950.

Andrew Y Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In Icml, volume 99, pp. 278–287. Citeseer, 1999.

Andrew Y Ng, Stuart Russell, et al. Algorithms for inverse reinforcement learning. In Icml, volume 1, pp. 2, 2000.

Francesco Orabona. A modern introduction to online learning. arXiv preprint arXiv:1912.13213, 2019.

Giorgia Ramponi, Pavel Kolev, Olivier Pietquin, Niao He, Mathieu Laurière, and Matthieu Geist. On imitation in mean-field games. Advances in Neural Information Processing Systems, 36:40426–40437, 2023.

Paul Rolland, Luca Viano, Norman Schürhof, Boris Nikolov, and Volkan Cevher. Identifiability and generalizability from multiple experts in inverse reinforcement learning. Advances in Neural Information Processing Systems, 35:550–564, 2022.

Andreas Schlaginhaufen and Maryam Kamgarpour. Towards the transferability of rewards recovered via regularized inverse reinforcement learning. Advances in Neural Information Processing Systems, 37:21461–21501, 2024.

Andreas Schlaginhaufen, Reda Ouhamma, and Maryam Kamgarpour. Eficient preferencebased reinforcement learning: Randomized exploration meets experimental design. Advances in Neural Information Processing Systems, 38:12146–12183, 2026.

Pier Giuseppe Sessa, Ilija Bogunovic, Maryam Kamgarpour, and Andreas Krause. Learning to play sequential games versus unknown opponents. Advances in neural information processing systems, 33:8971–8981, 2020.

Jingwu Tang, Gokul Swamy, Fei Fang, and Zhiwei S Wu. Multi-agent imitation learning: Value is easy, regret is hard. Advances in Neural Information Processing Systems, 37: 27790–27816, 2024.

Andrew Wagenmaker, Perry Dong, Raymond Tsao, Chelsea Finn, and Sergey Levine. Posterior behavioral cloning: Pretraining bc policies for eficient rl finetuning. arXiv preprint arXiv:2512.16911, 2025.

Jibang Wu, Weiran Shen, Fei Fang, and Haifeng Xu. Inverse game theory for stackelberg games: the blessing of bounded rationality. Advances in Neural Information Processing Systems, 35:32186–32198, 2022.

Renbo Zhao and Robert M Freund. Analysis of the frank–wolfe method for convex composite optimization involving a logarithmically-homogeneous barrier: R. zhao, rm freund. Mathematical programming, 199(1):123–163, 2023.

Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, Anind K Dey, et al. Maximum entropy inverse reinforcement learning. In Aaai, volume 8, pp. 1433–1438. Chicago, IL, USA, 2008.

## Appendix

The Appendix contains the proofs of the diferent results, as well as additional information.

## Table of Contents

A Previous work 18   
B Notations and Symbols 20   
B.1 Additional notations for the appendix 20   
B.2 Summary of the notations 24   
Hardness results for strategic identifiability of rewards 25   
C.1 Hardness of learning from a fixed policy 25   
C.2 A statistical lower bound for adaptive algorithms 27   
D Self-concordance properties 29   
D.1 Self-concordance of the log-likelihood in the span semi-norm . 29   
D.2 Change of coeficients in the Fisher information 32   
E Warm-up analysis 34   
E.1 A grouped Kiefer-Wolfowitz theory 36   
E.2 Solving for the warm-up allocation 38   
E.3 Rounding and certificate . 40   
F Analysis of the main algorithm 44   
F.1 Reward transfer lemma 44   
F.2 Finite-sample Stackelberg sub-optimality guarantee 45   
G Useful results 48   
H Additional experiments 51   
H.1 Lobéké environment from the GUARD benchmark 52   
H.2 Hyperparameter selection and sensitivity analysis . 56   
H.3 Diagnosis in synthetic environments 57

## A Previous work

We review the literature most relevant to our problem, with a particular focus on reward identifiability and inverse reinforcement learning.

Reward identifiability Strictly convex regularization of the objective, such as entropy, ensures the uniqueness of the optimal policies to any given set of rewards. Nevertheless, inverse RL remains ill-defined as rewards are invariant to potential shaping (Ng et al., 1999). This identifiability issue was studied in the seminal work of Cao et al. (2021), later refined by Rolland et al. (2022) and Schlaginhaufen & Kamgarpour (2024). More recently, Freihaut & Ramponi (2026) extends those results to games, showing that even the nature (e.g. collaborative or competitive) of a game cannot be determined from the feasible reward sets of an expert.

This line of work has motivated inverse RL works like Metelli et al. (2021); Lindner et al. (2022), that minimize the distance between the true and estimated feasible reward sets. Metelli et al. (2021) assumes access to a generative model, an assumption later removed by Lindner et al. (2022). However, these works do not provide a method for extracting a good reward or task representation from the interactions. This is the target of more traditional methods, inspired by the single-agent case.

Inverse reinforcement learning in games When considering multiple interacting agents, inverse RL aims to recover the individual rewards explaining their behaviors. To this end, Mehr et al. (2023) empirically extend the classical MaxEntIRL framework of Ziebart et al. (2008) to games, learning from demonstrations of a Nash equilibrium. Similarly, Liao et al. (2025) study reward recovery from samples of a fixed expert in two-player zero-sum game expert under coverability assumptions. Later, Barbara et al. (2026) derives sample complexity bounds for recovering the feasible reward set by sampling a fixed expert, evaluated in the Hausdorf distance. Under the strong assumption of oracle access to the loss gradient, Goktas et al. (2025) perform principled inverse multi-agent RL with parameter-dependent payofs. All these settings assume access to reliable equilibrium demonstrations, i.e. that the game has been solved a-priori by a set of players we can learn from (e.g. humans). This is restrictive in many applications, particularly in non-cooperative games where a player must learn to act against an adversary with unknown incentives. This motivates our setting, in which a player learns to behave strategically in arbitrary downstream tasks, playing against a best-responding agent.

Closer to this setup, Chen et al. (2023) extract the reward function from observing the follower in a two-player Stackelberg scenario. By controlling a prescription announced by the leader, they are able to bound the suboptimality of the optimal leader of the estimated environment. In contrast, we assume that the leader directly commits to a policy (as opposed to a prescription), and actively select leader strategies via optimal design to bound the ofline suboptimality objective. This has been studied by Chen et al. (2025), where the convergence of their algorithm is studied in the asymptotic regime. Contrary to their work, we derive an algorithm with sample complexity analysis and a matching minimax lower bound. Previously, Wu et al. (2022) also studied bounded rationality in the context of

Stackelberg Games. In particular, the authors show that observations are more informative from a QRE than a Nash equilibrium. This means that bounded rationality helps to be more robust against an irrational opponent. Under finite action spaces, this makes it possible to construct algorithms with exponential dependencies on neither the number of leader actions nor the number of follower actions.

Learning against unknown or adversarial opponents. Several works study learning in multi-agent environments with unknown or adaptive opponents, where non-stationarity and strategic interactions make learning particularly challenging. In particular, Sessa et al. (2020) consider learning efective strategies against unknown opponents, while more recent approaches use demonstrations to improve learning in strategic settings, as in DIML (An & Du, 2026). Other works study robustness against adversaries and worst-case behaviors during training (Liu et al., 2022). In contrast, we focus on reward learning under strategic interactions rather than solely on policy optimization or robustness.

Additionally, we expand below on the recent developments around adjacent tasks, such as multi-agent imitation learning, and contrast with the scope and goals of the present paper.

Imitation learning in games Recent imitation learning works in games measure suboptimality of the learned behavior by the notion of Nash gap, initially introduced by Ramponi et al. (2023). In general Markov games, Tang et al. (2024) showed how harder this objective is compared to the common value gap extension from single-agent problems. Later, these remarks have been extended to more precise characterizations for imitation learning by Freihaut et al. (2026a;b); Bergerault et al. (2026a). In the non-interactive case, these works show that the hardness of equilibria extraction can be reduced to notions of concentrability and continuity at equilibrium. Further, Freihaut et al. (2026a;b) show how to optimize such Nash gap regrets in the interactive setting by forcing the exploration of the state space to cover uncertain deviations. This is done by controlling all the agents sequentially. We argue that such control over the agents is not realistic in many scenarios, where agents are not collaborative in learning, e.g. in real world environments when learning from interactions with (partially) adversarial players. To this end, we adopt a Stackelberg objective to measure the optimality of extracted policies, evaluating a leader when deployed with other strategic players.

On reward identifiability Furthermore, imitation learning extracts a policy from (approximately) optimal agents corresponding to fixed incentives. Modeling agents’ behavior for a fixed task without describing their intents prevents transferring to new environments or learner tasks, limiting adaptation of the learner at deployment. Fortunately, observing (approximately) optimal actions from strategic players allows to extract signals about their individual rewards. However, the complete rewards can in general not be recovered, even with infinite data.

This ambiguity on the reward function has first been explored in the single-agent case, with the analysis of potential-based transformations or potential shaping originating from Ng et al. (1999). In the normal-form game setting, these transformations reduce to constant shifts, whose invariance to optimal policies carry over to the entropy-regularized setting (i.e.

softmax policies remain invariant to constant shifts of the rewards). The picture extends to games, where each individual reward is contained in an individual feasible reward set. In this scenario, even the nature of the game (e.g. cooperative, zero-sum) can sometimes not be identifiable from the feasible sets alone (Freihaut & Ramponi, 2026).

To mitigate this issue, one could consider learning in diferent environments, as pointed out by Rolland et al. (2022); Freihaut & Ramponi (2026). A second possibility is to weaken the exact identifiability objective, which aims to exactly learn the reward function, in particular by minimizing the mismatch with the ground truth in the Euclidean norm (Liao et al., 2025). Our work applies both propositions, where the learner agent induces diferent environments to the soft-optimal follower, and where only the strategic identifiability of its reward is considered.

Empirical inverse reinforcement learning In the single-agent case, the seminal work of Ziebart et al. (2008) develops MaxEntIRL, an algorithm recovering the reward matching the maximum-entropy solution given by the demonstrations. The proposed method operates under linear rewards. Their method essentially boils down to gradient descent on the reward parameters, where gradients, given by expected features, are approximated from rollouts of the policy in the environment.

MaxEntIRL has later been empirically extended to games by Mehr et al. (2023), and the theoretical guarantees of such extensions in terms of strategic exploitability are yet to be understood. Furthermore, one caveat of this method is the requirement that all players are controllable, which can be unreasonable in various applications. In contrast, we develop in this work a principled algorithm with PAC guarantees under a finite number of samples. In our two-player setting, only one agent is assumed controllable, while the non-controllable agent (the follower) is best-responding to the learner (the leader policy).

## B Notations and Symbols

## B.1 Additional notations for the appendix

For the simplicity of the analysis, we adopt throughout the appendix additional notations not introduced in the main content. We summarize all introduced notations at the end of this section.

Already, let us define the policy spaces as $\Pi _ { \mu } : = \Delta ( { \mathcal { A } } )$ and $\Pi _ { \nu } : = \Delta ( B )$ for the $\mu -$ and ν-player, respectively.

## Features and feature spaces

We recall from the main content that without loss of generality, the known feature map $\phi : \mathcal { A } \times \mathcal { B }  \mathbb { R } ^ { d }$ is such that $\| \phi ( a , b ) \| _ { 2 } \leq 1$ for all $a , b \in { \mathcal { A } } \times B$

For every $a \in { \mathcal { A } } , b \in B _ { \cdot }$ and any policies $\mu \in \Pi _ { \mu } , \nu \in \Pi _ { \nu } .$ , we define the average features as:

$$
\psi _ { \mu } ( b ) : = \sum _ { a ^ { \prime } \in A } \mu ( a ^ { \prime } ) \phi ( a ^ { \prime } , b ) , \qquad \psi _ { \nu } ( a ) : = \sum _ { b ^ { \prime } \in B } \nu ( b ^ { \prime } ) \phi ( a , b ^ { \prime } ) ,
$$

where the notation makes it clear from context if we take the average over A or B.

These averaged features are used to define the uniform-centered features $g _ { \mu }$ spanning the space $\mathcal { G } _ { 1 }$ :

$$
g _ { \mu } ( b ) : = \psi _ { \mu } ( b ) - \frac { 1 } { | \mathcal { B } | } \sum _ { b ^ { \prime } \in \mathcal { B } } \psi _ { \mu } ( b ^ { \prime } ) , \qquad \mathcal { G } _ { 1 } : = \mathrm { s p a n } \{ g _ { \mu } ( b ) \} _ { \mu , b } .
$$

For any $\mu \in \Pi _ { \mu } , \nu \in \Pi _ { \nu }$ , let us define the following expected features:

$$
\overline { { \psi } } _ { \mu } ( \tilde { \omega } ) = \sum _ { b \in \mathcal { B } } \mathbb { P } ( b | \tilde { \omega } , \mu ) \psi _ { \mu } ( b ) \qquad \mathrm { a n d } \qquad \overline { { \psi } } _ { \nu } ( \tilde { \omega } ) = \sum _ { a \in \mathcal { A } } \mathbb { P } ( a | \tilde { \omega } , \nu ) \psi _ { \nu } ( a ) ,
$$

where we can therefore recognize $g _ { \mu } ( b ) = \psi _ { \mu } ( b ) - \overline { { \psi } } _ { \mu } ( 0 )$ , and similarly for $g _ { \nu }$

From the definition of the features, d represents the ambient dimension. We also define the following two terms:

$$
d _ { 1 } ^ { * } : = \dim \operatorname { s p a n } \{ \psi _ { \mu } ( b ) \} _ { \mu , b } , \qquad d _ { 1 } ^ { \diamond } : = \dim \mathcal { G } _ { 1 } ,
$$

where in general, $d \mathcal { \hat { 1 } } \leq d _ { 1 } ^ { * } \leq d .$

Using these notations, Assumption 3 is equivalent to assuming $d _ { 1 } ^ { \circ } \geq 1$

To simplify notations, we will use $d ^ { \circ } , d ^ { * }$ and $\mathcal { G }$ to refer to $d _ { 1 } ^ { \circ } , d _ { 1 } ^ { \ast }$ and $\mathcal { G } _ { 1 }$ , respectively. This is because for our main result, the analysis only depends on these quantities for the leader player, i.e. player 1.

## Measuring distances between logits

Our analysis builds on how well the coeficients are estimated throughout the iterations of the algorithm. This is, in part, measured by the diferences in induced logits for the best-response player. Intuitively, to allow for arbitrary constant shifts (it doesn’t change the softmax), we project away this dimension and express our results with the following span semi-norm.

Definition 4 (Span semi-norm). For any $n \in \mathbb { Z } _ { + }$ and $\begin{array} { r } { v \in \mathbb { R } ^ { n } , \| v \| _ { \mathrm { s p a n } } = \operatorname* { m a x } _ { i } v _ { i } - \operatorname* { m i n } _ { j } v _ { j } } \end{array}$

In particular, for all $\mu \in \Pi _ { \mu } , \omega _ { 1 } , \omega _ { 2 } \in \Omega$

$$
\| \langle \psi _ { \mu } ( \cdot ) , \omega _ { 1 } - \omega _ { 2 } \rangle \| _ { \mathrm { s p a n } } = \operatorname* { m a x } _ { b _ { 1 } , b _ { 2 } \in \mathcal B } \langle \psi _ { \mu } ( b _ { 1 } ) - \psi _ { \mu } ( b _ { 2 } ) , \omega _ { 1 } - \omega _ { 2 } \rangle .
$$

This latter quantity will be key to our analysis. We therefore define the following two functions:

$$
d _ { \mu } ( \omega _ { 1 } , \omega _ { 2 } ) : = \| \langle \psi _ { \mu } ( \cdot ) , \omega _ { 1 } - \omega _ { 2 } \rangle \| _ { \mathrm { s p a n } } , \qquad d _ { * } ( \omega _ { 1 } , \omega _ { 2 } ) : = \operatorname* { m a x } _ { \tilde { \mu } } d _ { \tilde { \mu } } ( \omega _ { 1 } , \omega _ { 2 } ) ,
$$

where by Cauchy-Schwarz and the triangle inequality: $d _ { * } ( \omega _ { 1 } , \omega _ { 2 } ) \leq 2 \left\| \omega _ { 1 } - \omega _ { 2 } \right\| _ { 2 } \leq 4 S .$

## Linear algebra on the identifiable subspace

As mentioned before, the observable space has dimension $d ^ { \circ } \leq d ,$ and the inequality can be strict. Any interaction can only reveal information about the projection of the reward coeficients in that space, and this is suficient for any reward representation in terms of strategic identifiability. We therefore define this subspace more precisely here, and in particular appropriate operators over this subspace.

This allows for simpler notations and proofs, but the following is bookkeeping rather than deep machinery. Natural operations resolve cleanly.

Let $P _ { \mathcal { G } }$ denote the orthogonal projector onto ${ \mathcal { G } } ,$ and define

$$
\begin{array} { r } { \mathbb { S } ( \mathcal { G } ) : = \left\{ A \in \mathbb { S } ^ { d } : A = P \mathcal { G } A P \mathcal { G } \right\} , \qquad \mathbb { S } _ { + } ( \mathcal { G } ) : = \mathbb { S } ( \mathcal { G } ) \cap \mathbb { S } _ { + } ^ { d } , } \end{array}
$$

so that $\mathbb { S } _ { + } ( \mathcal { G } )$ is the set of PSD ambient operators supported on $\mathcal { G }$

We further define the set of positive definite operators on $\mathcal { G }$ as

$$
\mathbb { S } _ { + + } ( \mathcal { G } ) : = \left\{ A \in \mathbb { S } _ { + } ( \mathcal { G } ) : x ^ { \top } A x > 0 , \ \forall x \in \mathcal { G } \setminus \{ 0 \} \right\} ,
$$

precisely the set of PSD matrices such that Im $A = { \mathcal { G } }$ , so that $A$ can cover the information of any pair $( \mu , \omega )$ . We keep $\omega$ in $\mathbb { R } ^ { d }$ for the simplicity of the analysis and algorithms, so that Images and kernels are always understood as subspaces of the ambient space $\mathbb { R } ^ { d }$

Now fix $U \in \mathbb { R } ^ { d \times d ^ { \circ } }$ such that the columns of $U$ form an orthonormal basis of $\mathcal { G }$ . The definitions below are invariant to the specific choice of $U$

For $A \in \mathbb { S } _ { + } ( \mathcal { G } )$ , we define its determinant in $\mathcal { G }$ and use the extended-real convention as:

$$
\begin{array} { r } { \operatorname* { d e t } _ { \mathcal { G } } ( A ) : = \operatorname* { d e t } ( U ^ { \top } A U ) , \qquad \log \operatorname* { d e t } _ { \mathcal { G } } ( A ) = - \infty \quad \mathrm { i f } \quad A \not \in \mathbb { S } _ { + + } ( \mathcal { G } ) . } \end{array}
$$

This A also implies the following extended dual semi-norm for any $u \in { \mathcal { G } } \colon$

$$
\| u \| _ { A , * } = \left\{ { \begin{array} { l l } { \sqrt { u ^ { \top } A ^ { \dagger } u } , } & { u \in \operatorname { I m } A , } \\ { + \infty , } & { u \not \in \operatorname { I m } A , } \end{array} } \right.
$$

where $A ^ { \dagger }$ denotes the ordinary Moore-Penrose pseudo-inverse.

In particular, when $A \in \mathbb { S } _ { + + } ( \mathcal { G } )$ , we recover the standard semi-norm in $A ^ { \dagger }$

$$
\| u \| _ { A , * } ^ { 2 } = \| u \| _ { A ^ { \dagger } } ^ { 2 } = u ^ { \top } A ^ { \dagger } u , \qquad A ^ { \dagger } = U ( U ^ { \top } A U ) ^ { - 1 } U ^ { \top } .
$$

For any $M \in \mathbb { S } _ { + } ( \mathcal G )$ with eigen value-vector pairs $\{ \lambda _ { i } , u _ { i } \} _ { i }$ , we extend the trace as:

Trace $\left( A ^ { \dagger } M \right) = \sum _ { i : \lambda _ { i } > 0 } \lambda _ { i } \| u _ { i } \| _ { A , * } ^ { 2 } ,$ such that Trace $\left( A ^ { \dagger } M \right) = + \infty$ if Im $M \not \subseteq$ Im A. (4)

All the newly introduced operators behave similarly to the usual operators. We show such some properties below.

Lemma 4 (Linear algebra on $\mathcal { G } )$ . Let A, $B \in \mathbb { S } _ { + } ( \mathcal { G } ) , u \in \mathcal { G }$ , and $\boldsymbol { x } \in \mathbb { R } ^ { d }$ . Then the following properties hold.

1. Generalized Cauchy-Schwarz inequality: $| \langle u , x \rangle | \leq \| x \| _ { A } \| u \| _ { A , * }$

2. If $A \succeq c B$ for some $c > 0$ , then $\| u \| _ { A , * } ^ { 2 } \leq c ^ { - 1 } \| u \| _ { B , * } ^ { 2 }$

Moreover, $i f A , B \in \mathbb { S } _ { + + } ( { \mathcal { G } } )$ , then $A ^ { \dagger } \preceq c ^ { - 1 } B ^ { \dagger }$

3. If $A \in \mathbb { S } _ { + + } ( \mathcal { G } )$ and $H \in \mathbb { S } ( \mathcal { G } )$ , then

$$
{ \frac { \mathrm { d } } { \mathrm { d } t } } \log \operatorname* { d e t } _ { \mathcal { G } } ( A + t H ) { \bigg | } _ { t = 0 } = \mathrm { T r a c e } ( A ^ { \dagger } H ) , \qquad \mathrm { T r a c e } ( A ^ { \dagger } A ) = d ^ { \diamond } .
$$

Proof. Define the following terms:

$$
\bar { A } : = U ^ { \top } A U , \qquad \bar { B } : = U ^ { \top } B U , \qquad \bar { u } : = U ^ { \top } u , \qquad \bar { x } : = U ^ { \top } x .
$$

The claims then reduce to the corresponding standard facts for $\bar { A } , \bar { B } \in \mathbb { S } _ { + } ^ { d ^ { \circ } }$

## Coverage and information matrices

For any $\omega _ { 1 } , \omega _ { 2 } \in \Omega , \tilde { \mu } \in \Pi _ { \mu }$ , we introduce the Fisher information in its integral form as

$$
\Gamma _ { \tilde { \mu } } ( \omega _ { 1 } , \omega _ { 2 } ) : = \int _ { 0 } ^ { 1 } \mathcal { T } _ { \tilde { \mu } } ( \omega _ { 1 } + s ( \omega _ { 2 } - \omega _ { 1 } ) ) \mathrm { d } s ,
$$

where we recall the Fisher information matrix:

$$
\begin{array} { r } { \mathcal { T } _ { \tilde { \mu } } ( \tilde { \omega } ) = \eta ^ { 2 } \mathrm { C o v } _ { b \sim \nu _ { \tilde { \mu } } ^ { * } ( \tilde { \omega } ) } \left[ \psi _ { \tilde { \mu } } ( b ) \right] = \eta ^ { 2 } \mathbb { E } _ { b \sim \nu _ { \tilde { \mu } } ^ { * } } \left[ \left( \psi _ { \tilde { \mu } } ( b ) - \overline { { \psi } } _ { \tilde { \mu } } ( \tilde { \omega } ) \right) ^ { \otimes 2 } \right] . } \end{array}
$$

This motivates the following definitions of feature coverage.

For any policy $\mu \in \Pi _ { \mu }$ , we define its uniform coverage as the matrix:

$$
M _ { \mu } : = \frac 1 { | \mathcal B | } \sum _ { b \in \mathcal B } g _ { \mu } ( b ) g _ { \mu } ( b ) ^ { \top } \succeq 0 ,
$$

such that the following defines the average uniform coverage for any distribution $w \in \Delta ( \Pi _ { \mu } )$

$$
V ( w ) : = \int _ { \Pi _ { \mu } } M _ { \mu } w ( \mathrm { d } \mu ) \in \mathbb { S } _ { + } ( \mathcal { G } ) .
$$

We additionally call w admissible if $V ( w ) \in \mathbb { S } _ { + + } ( \mathcal { G } )$ , i.e. Im $V ( w ) = \mathcal { G }$ , and denote by $\mathcal { W } _ { \mathcal { G } }$ the set of admissible designs.

As span $\mathsf { U } _ { \mu }$ Im $M _ { \mu } = \mathcal G$ , this implies

$$
w \in { \mathcal { W } } _ { \mathcal { G } } \quad \Longleftrightarrow \quad { \underset { \mu } { \operatorname* { m a x } } } { \mathrm { T r a c e } } \left( V ( w ) ^ { \dagger } M _ { \mu } \right) < \infty \quad \Longleftrightarrow \quad \log \mathrm { d e t } _ { \mathcal { G } } V ( w ) > - \infty .
$$

## Additional notations for the online phase

We introduce additional notations to simplify the indexation of policies and likelihoods during the online phase. Specifically, we define

$$
N _ { k } : = n _ { \mathrm { w a r m } } + k , \quad \mathcal { L } _ { k } ( \cdot ) : = \mathcal { L } ( \cdot | \mathcal { D } _ { k } ) , \quad G _ { k } ^ { \lambda } ( \cdot ) : = G ^ { \lambda } ( \cdot | \mathcal { D } _ { k } ) , \qquad \forall k \in [ K ] .
$$

## B.2 Summary of the notations

General notation
<table><tr><td> $\Delta ( \mathcal { X } )$ </td><td>Probability simplex over a measurable space  $\mathcal { X }$ </td></tr><tr><td> $\mathcal { H } _ { \mathcal { X } } ( P )$ </td><td>Shannon entropy; abbreviated  $\mathcal { H }$  when  $\mathcal { X }$  is clear</td></tr><tr><td> $\| { \boldsymbol { x } } \| _ { P }$ </td><td>Mahalanobis (semi-)norm  $\sqrt { x ^ { \top } P x } , P \succeq 0$ </td></tr><tr><td> $x ^ { \otimes 2 }$ </td><td>Outer product  $x x ^ { \top }$ </td></tr><tr><td> $D _ { \mathrm { K L } } ( p \vert \vert q )$ </td><td>Kullback-Leibler divergence</td></tr><tr><td>Im  $M ,$  span, rank</td><td>Range, linear span, rank</td></tr><tr><td> $M ^ { \dagger }$ </td><td>Moore-Penrose pseudo-inverse of a PSD matrix M</td></tr><tr><td> $P _ { \mathcal { G } }$ </td><td>Orthogonal projector onto  $\mathcal { G }$ </td></tr><tr><td> $\mathbb { S } _ { + } ( \mathcal { G } )$ </td><td>PSD ambient operators supported on  $\mathcal { G }$ </td></tr><tr><td> $\mathbb { S } _ { + + } ( \mathcal { G } )$ </td><td>Positive definite ambient operators supported on  $\mathcal { G }$ </td></tr><tr><td> $\| u \| _ { A , * }$ </td><td>Dual norm in  $\mathcal { G }$ </td></tr><tr><td> $\operatorname* { d e t } g ( A )$ </td><td>Determinant of the restriction of  $A$  to  $\mathcal { G }$ </td></tr><tr><td> $\ell ( x )$ </td><td>Logistic function  $( 1 + e ^ { - x } ) ^ { - 1 }$ </td></tr><tr><td> $\lVert v \rVert _ { \mathrm { s p a n } }$ </td><td>Span seminorm max  $\dot { \mathbf { \rho } } _ { i } v _ { i } - \operatorname* { m i n } _ { j } v _ { j }$ </td></tr></table>

<table><tr><td> $A , B$ </td><td>Finite action spaces of leader and follower</td></tr><tr><td> $\mu , \nu$ </td><td>Policies of the leader and the follower</td></tr><tr><td> $\eta > 0$ </td><td>Inverse temperature (bounded rationality)</td></tr><tr><td> $\nu _ { \mu } ^ { * } ( \tilde { \omega } )$ </td><td>Follower&#x27;s regularised best response, Equation (1)</td></tr><tr><td> ${ \dot { J } } _ { r } ^ { \mathrm { l e a d e r } } ( \mu )$ </td><td>Leader value  $J _ { r ^ { 1 } } ( \mu , \nu _ { \mu } ^ { * } ( r ^ { 2 } ) )$ </td></tr><tr><td> $\mathrm { S u b O p t } _ { r } ( \mu )$ </td><td>Stackelberg sub-optimality</td></tr><tr><td> $\mathrm { N a s h G a p } _ { r } ( \mu , \nu )$ </td><td>Maximal regret over the two players</td></tr><tr><td> $\Pi _ { \mu } , \Pi _ { \nu }$ </td><td>Leader and follower policy spaces</td></tr></table>

<table><tr><td> $\phi ( a , b )$ </td><td>Known feature map,  $\mathrm { m a x } _ { a , b } \| \phi ( a , b ) \| _ { 2 } \leq 1$ </td></tr><tr><td> $\omega , \hat { \omega } , \hat { \omega } _ { k }$ </td><td>True  $/$  final  $/$  round-k reward coefficients</td></tr><tr><td> $S , \Omega$ </td><td>Bound  $\| \omega \| _ { 2 } \leq S$  and the ball  $\{ \tilde { \omega } : \| \tilde { \omega } \| _ { 2 } \leq S \}$ </td></tr><tr><td> $R _ { \mathrm { m a x } }$ </td><td>Upper bound  $\| r _ { 1 } \| _ { \infty }$ </td></tr><tr><td> $\psi _ { \mu } ( b )$ </td><td>Averaged follower feature  $\textstyle \sum _ { a } \mu ( a ) \phi ( a , b )$ </td></tr><tr><td> $\overline { { \psi } } _ { \mu } ( \tilde { \omega } )$ </td><td>Expected feature  $\mathbb { E } _ { b \sim \nu _ { \mu } ^ { * } ( \tilde { \omega } ) } \psi _ { \mu } ( b )$ </td></tr></table>

<table><tr><td> $g _ { \mu } ( b )$ </td><td>Uniform-centred feature  $\begin{array} { r } { \psi _ { \mu } ( b ) - \frac { 1 } { | B | } \sum _ { b ^ { \prime } } \psi _ { \mu } ( b ^ { \prime } ) } \end{array}$ </td></tr><tr><td> $\mathcal { G }$ </td><td>span  $\{ g _ { \mu } ( b ) \} _ { \mu , b } = \operatorname { s p a n } \{ \psi _ { \mu } ( b ) - \psi _ { \mu } ( \dot { d } ) \} _ { \mu , b , d }$ </td></tr><tr><td> $d ^ { \circ }$ </td><td>dim  $\mathcal { G } \colon$  the reward directions the follower can distinguish</td></tr><tr><td> $d ^ { * }$ </td><td>dim span  $\{ \psi _ { \mu } ( b ) \} _ { \mu , b } ,$  SO  $d ^ { \circ } \leq d ^ { * } \leq d$ </td></tr></table>

## Information and estimation

<table><tr><td> $N , n _ { \mathrm { w a r m } } , K$ </td><td>Total, warm-up, and online interaction counts, with  $K =$   $N - n _ { \mathrm { w a r m } }$ </td></tr><tr><td> $N _ { k }$ </td><td>Total sample count  $n _ { \mathrm { w a r m } } + k$  at online checkpoint k</td></tr><tr><td> $\mathcal { D } , \mathcal { D } _ { \mathrm { w a r m } } , \mathcal { D } _ { k }$ </td><td>Full, warm-up, and online historical data</td></tr><tr><td> $l , \mathcal { L } , \mathcal { L } _ { k }$ </td><td>Instantaneous, cumulative and online cumulative log-likelihood</td></tr><tr><td> $\mathcal { T } _ { \boldsymbol { \mu } } ( \widetilde { \omega } )$ </td><td>Fisher information  $\eta ^ { 2 } \mathrm { C o v } _ { b \sim \nu _ { \mu } ^ { * } ( \tilde { \omega } ) } [ \psi _ { \mu } ( b ) ]$ </td></tr><tr><td> $G ^ { \lambda } , G _ { k } ^ { \lambda }$ </td><td>Fisher and online cumulative regularized Fisher information</td></tr><tr><td> $\Gamma _ { \tau } ^ { \lambda } ( \tilde { \omega } )$ </td><td>Residual (integrated) information, Theorem 5</td></tr><tr><td> $\beta _ { \tau } ( \delta )$ </td><td>MLE confidence radius, Theorem  $5$ </td></tr><tr><td> $d _ { \mu } ( \omega _ { 1 } , \omega _ { 2 } )$ </td><td>Span seminorm  $\| \langle \psi _ { \mu } ( \cdot ) , \omega _ { 1 } - \omega _ { 2 } \rangle |$  |span</td></tr><tr><td> $d _ { * } ( \omega _ { 1 } , \omega _ { 2 } )$ </td><td>maxµ  $d _ { \tilde { \mu } } ( \omega _ { 1 } , \omega _ { 2 } )$  ; bounded by 4S</td></tr><tr><td> $\chi ( x )$ </td><td> $2 ( e ^ { x / 2 } - 1 ) / x$  for  $x > 0 , \chi ( 0 ) = 1$  , increasing</td></tr><tr><td>κ</td><td>Coverage constant  $2 e ^ { 2 \eta S } / \eta ^ { 2 } ,$  Lemma 14</td></tr></table>

## Experimental design

<table><tr><td> $M _ { \mu }$ </td><td>Grouped arm  $\begin{array} { r } { \frac { 1 } { | \mathcal { B } | } \sum _ { b } g _ { \mu } ( b ) g _ { \mu } ( b ) ^ { \top } \succeq 0 } \end{array}$ </td></tr><tr><td> $V ( w )$ </td><td>Design uniform covariance  $\begin{array} { r } { \int _ { \Pi _ { \mu } } M _ { \mu } w ( \mathrm { d } \mu ) } \end{array}$ </td></tr><tr><td>admissible,  $\mathcal { W } g$ </td><td>w with Im  $V ( w ) = \mathcal { G } ; \mathcal { W } _ { \mathcal { G } }$  is the set of such designs</td></tr><tr><td> $w _ { D }$ </td><td>D-optimal design arg maxw log detg V(w)</td></tr><tr><td>€</td><td>Design  $/$  rounding tolerance</td></tr><tr><td> $n _ { \mathrm { w a r m } } , \mathcal { D } _ { \mathrm { w a r m } }$ </td><td>Warm-up budget and its sample set</td></tr><tr><td> $V _ { \mathrm { w a r m } }$ </td><td>Uniform allocation coverage matrix</td></tr></table>

## C Hardness results for strategic identifiability of rewards

We provide in this section the proofs of the impossibility and lower bound results present in the main text, i.e. Lemma 2 and Theorem 1.

## C.1 Hardness of learning from a fixed policy

We start by proving the hardness of identifying rewards that maintain the optimal leader by querying a single pair of policies.

Lemma 2. Fix $\eta > 0$ . There exists a set $( r _ { 1 } , \mathcal { A } , B , \phi )$ such that for any reference policy $\mu _ { r e f }$ and any number $o f$ samples $K$ , no algorithm only sampling from $( \mu _ { r e f } , \nu _ { \mu _ { r e f } } ^ { * } )$ can guarantee that $\mathrm { S u b O p t } _ { r } ( { \hat { \mu } } ) \leq c _ { \eta }$ with probability greater than $1 / 2 . \ c _ { \eta } > 0$ only depends on η.

Proof. Let $\mathcal { M } _ { + } , \mathcal { M } _ { - }$ be two Stackelberg games with $A = \{ a _ { 1 } , a _ { 2 } \} , B = \{ b _ { 1 } , b _ { 2 } \}$ , and the tabular feature map $\phi ( a _ { i } , b _ { j } ) = e _ { i j } \in \mathbb { R } ^ { 4 }$

Fix an arbitrary leader policy $\mu _ { \mathrm { r e f } }$ and let us write $p _ { \mathrm { r e f } } = \mu _ { \mathrm { r e f } } ( a _ { 1 } )$ and $1 - p _ { \mathrm { r e f } } = \mu _ { \mathrm { r e f } } ( a _ { 2 } )$

Define two follower rewards $r _ { 2 } ^ { + }$ and $r _ { 2 } ^ { - }$ by setting

$$
s _ { 1 } : = 1 - p _ { \mathrm { r e f } } , \qquad s _ { 2 } : = - p _ { \mathrm { r e f } } ,
$$

and

$$
r _ { 2 } ^ { + } ( a _ { i } , b _ { 1 } ) = \frac { 1 } { 2 } + \frac { s _ { i } } { 2 } , \qquad r _ { 2 } ^ { + } ( a _ { i } , b _ { 2 } ) = \frac { 1 } { 2 } - \frac { s _ { i } } { 2 } ,
$$

while

$$
r _ { 2 } ^ { - } ( a _ { i } , b _ { 1 } ) = \frac { 1 } { 2 } - \frac { s _ { i } } { 2 } , \qquad r _ { 2 } ^ { - } ( a _ { i } , b _ { 2 } ) = \frac { 1 } { 2 } + \frac { s _ { i } } { 2 } .
$$

All rewards lie in [0, 1], and since the features are tabular, both $r _ { 2 } ^ { + }$ and $r _ { 2 } ^ { - }$ are linear in $\phi .$ For any $q \in [ 0 , 1 ]$ , we denote by $\mu _ { q }$ the leader policy such that $\mu _ { q } ( a _ { 1 } ) = q$ . Under the policy $\mu _ { q } ,$ the averaged reward diferences are as follows:

• In $\mathcal { M } _ { + }$

$$
\sum _ { a } \mu _ { q } ( a ) \left( r _ { 2 } ^ { + } ( a , b _ { 1 } ) - r _ { 2 } ^ { + } ( a , b _ { 2 } ) \right) = q - p _ { \mathrm { r e f } }
$$

• In M<sub>−</sub>:

$$
\sum _ { a } \mu _ { q } ( a ) \left( r _ { 2 } ^ { - } ( a , b _ { 1 } ) - r _ { 2 } ^ { - } ( a , b _ { 2 } ) \right) = p _ { \mathrm { r e f } } - q
$$

Therefore we have the following entropy-regularized best-responses,

$$
\nu _ { q , + } ^ { * } ( b _ { 1 } ) = \ell ( \eta ( q - p _ { \mathrm { r e f } } ) ) , \qquad \nu _ { q , - } ^ { * } ( b _ { 1 } ) = \ell ( \eta ( p _ { \mathrm { r e f } } - q ) ) ,
$$

where ℓ is the standard logistic function $\ell ( x ) = ( 1 + \exp ( - x ) ) ^ { - 1 }$

At the queried policy $q = p _ { \mathrm { r e f } }$ , both games induce

$$
\nu _ { p _ { \mathrm { r e f } } , + } ^ { * } ( b _ { 1 } ) = \nu _ { p _ { \mathrm { r e f } } , - } ^ { * } ( b _ { 1 } ) = \frac { 1 } { 2 } .
$$

Hence the full observation law is identical in the two games, for every number of samples $K$ Now choose the leader reward $r _ { 1 } ( a , b ) = \mathbf { 1 } \{ b = b _ { 1 } \}$ such that the leader values are:

$$
J _ { r ^ { \pm } } ^ { \mathrm { l e a d e r } } ( \mu _ { q } ) = \nu _ { q , \pm } ^ { * } ( b _ { 1 } ) = \ell ( \pm \eta ( q - p _ { \mathrm { r e f } } ) ) .
$$

As ℓ is strictly increasing, the optimal leaders of $\mathcal { M } _ { + }$ and M<sub>−</sub> are therefore $\mu _ { 1 }$ and $\mu _ { 0 } .$ respectively, with corresponding optimal values $\ell ( \eta ( 1 - p _ { \mathrm { r e f } } ) )$ and $\ell ( \eta p _ { \mathrm { r e f } } )$ . Using this fact, for any q,

$$
\begin{array} { r l r } { \mathrm { S u b O p t } _ { r ^ { + } } ( \mu _ { q } ) + \mathrm { S u b O p t } _ { r ^ { - } } ( \mu _ { q } ) = \ell ( \eta ( 1 - p _ { \mathrm { r e f } } ) ) - \ell ( \eta ( q - p _ { \mathrm { r e f } } ) ) } & { } & \\ { + \ell ( \eta p _ { \mathrm { r e f } } ) - \ell ( - \eta ( q - p _ { \mathrm { r e f } } ) ) } & { } & \\ { = \ell ( \eta ( 1 - p _ { \mathrm { r e f } } ) ) + \ell ( \eta p _ { \mathrm { r e f } } ) - 1 } & { } & \\ { \geq \ell ( \eta ) - \displaystyle \frac { 1 } { 2 } , } & { } & \end{array}
$$

where the last inequality follows because $p \mapsto \ell ( \eta p ) + \ell ( \eta ( 1 - p ) ) - 1$ is symmetric and minimized at $p = 0 \ \mathrm { o r } \ p = 1$

Hence at least one of SubOpt (q) and SubOpt (q) is larger than $\begin{array} { r } { \frac { 1 } { 2 } \left( \ell ( \eta ) - \frac { 1 } { 2 } \right) } \end{array}$ , therefore strictly larger than $\begin{array} { r } { c _ { \eta } : = \frac { 1 } { 3 } \left( \ell ( \eta ) - \frac { 1 } { 2 } \right) } \end{array}$

Since the two games induce exactly the same observation distribution at the only queried policy, any algorithm has the same output distribution under the two games. Let

$$
\begin{array} { r } { \mathcal E _ { + } : = \{ \mathrm { S u b O p t } _ { r _ { + } } ( \hat { q } ) \leq c _ { \eta } \} , \qquad \mathcal E _ { - } : = \{ \mathrm { S u b O p t } _ { r _ { - } } ( \hat { q } ) \leq c _ { \eta } \} , } \end{array}
$$

with $\mathcal { E } _ { + } \cap \mathcal { E } _ { - } = \emptyset$ from the previous remark.

Since $\hat { q }$ has the same law in the two games,

$$
2 \cdot \operatorname* { m i n } \{ \mathbb { P } _ { + } ( \mathcal { E } _ { + } ) , \mathbb { P } _ { - } ( \mathcal { E } _ { - } ) \} \leq \mathbb { P } _ { \mathcal { M } _ { + } } ( \mathcal { E } _ { + } ) + \mathbb { P } _ { \mathcal { M } _ { - } } ( \mathcal { E } _ { - } ) \leq 1 .
$$

So no algorithm observing only samples from the single policy $\mu _ { \mathrm { r e f } }$ can guarantee Stackelberg suboptimality at most $c _ { \eta }$ with probability larger than $1 / 2$ uniformly over the two games.

## C.2 A statistical lower bound for adaptive algorithms

The usual strategy for a lower bound is to exhibit a problem instance where no algorithm can quickly distinguish between edge cases. We define such a hard instance as follows.

Definition 5 (Hard instance). Let $\mathcal { A } = [ d ] , \mathcal { B } = \{ b _ { 1 } , b _ { 2 } \} , R _ { \mathrm { m a x } } \in \mathbb { R } _ { + }$ , and

$$
\phi ( a _ { i } , b _ { 1 } ) = 0 , \qquad \phi ( a _ { i } , b _ { 2 } ) = e _ { i } , \qquad r _ { 1 } ( a , b ) = R _ { \mathrm { m a x } } \cdot { \bf 1 } \{ b = b _ { 2 } \} .
$$

We identify $\mu \in \Delta _ { d }$ and write $\ell ( x ) : = ( 1 + e ^ { - x } ) ^ { - 1 }$ for the logistic function.

For simplicity of the analysis, we first state some properties of this instance.

Proposition 1 (Properties of the hard instance). Reusing the notations of the hard instance (Definition 5), for every leader $\mu \in \Delta _ { d }$ and $\tilde { \omega } \in \Omega$ :

• Structural simplifications

$$
\begin{array} { r l r } & { } & { \psi _ { \mu } ( b _ { 1 } ) = 0 , \qquad \psi _ { \mu } ( b _ { 2 } ) = \mu , \qquad \| \psi _ { \mu } ( b ) \| _ { 2 } \le 1 , ~ } \\ & { } & { \mathcal { G } = \mathbb { R } ^ { d } , \qquad d ^ { \diamond } = d , \qquad d _ { \mu } ( 0 , \varepsilon e _ { i } ) = \varepsilon \mu _ { i } ~ ( \varepsilon \ge 0 ) . } \end{array}
$$

• Implied best-response

$$
\nu _ { \mu } ^ { * } ( \tilde { \omega } ) ( b _ { 2 } ) = p _ { \mu } ( \tilde { \omega } ) .
$$

• Performance

$$
J _ { \tilde { \omega } } ^ { \mathrm { l e a d e r } } ( \mu ) = R _ { \mathrm { m a x } } \cdot p _ { \mu } ( \tilde { \omega } ) , \qquad \mathrm { S u b O p t } _ { \tilde { \omega } } ( \mu ) = R _ { \mathrm { m a x } } \left[ \ell \left( \eta \operatorname* { m a x } _ { j } \tilde { \omega } _ { j } \right) - p _ { \mu } ( \tilde { \omega } ) \right] .
$$

where we denote $p _ { \mu } ( \tilde { \omega } ) : = \ell ( \eta \langle \mu , \tilde { \omega } \rangle )$ ).

Proof. We only prove some properties, the rest directly follows from definition.

Implied best-response

Note that since $| B | = 2$ , the best-response is the following logistic law:

$$
\nu _ { \mu } ^ { * } ( \tilde { \omega } ) ( b _ { 2 } ) = \frac { \exp \left( \eta \langle \mu , \tilde { \omega } \rangle \right) } { 1 + \exp \left( \eta \langle \mu , \tilde { \omega } \rangle \right) } = \ell \left( \eta \langle \mu , \tilde { \omega } \rangle \right) = p _ { \mu } ( \tilde { \omega } ) .\tag{5}
$$

Performance

Given Equation (5), since ℓ is increasing and $\mu \mapsto \langle \mu , \tilde { \omega } \rangle$ is linear on the simplex,

$$
\operatorname* { m a x } _ { \mu } J _ { \tilde { \omega } } ^ { \mathrm { l e a d e r } } ( \mu ) = R _ { \mathrm { m a x } } \cdot \ell \left( \eta \operatorname* { m a x } _ { j } \tilde { \omega } _ { j } \right) .
$$

This implies the $\mathrm { S u b O p t } _ { \tilde { \omega } } ( \mu )$ equality.

We can now state the lower bound as follows.

Theorem 1 (Minimax lower bound). Let $\begin{array} { r } { d \ge 3 2 , \epsilon = \frac { 1 } { 2 \eta } \sqrt { d / K } \le S } \end{array}$ and $\boldsymbol \omega ^ { ( i ) } : = \varepsilon \boldsymbol e _ { i }$ for $i \in [ d ]$ . There exists a game such that for every algorithm producing an output $\hat { \mu }$ after $K \geq 4 d$ interactions,

$$
\operatorname* { m a x } _ { i \in [ d ] } \mathbb { E } ^ { ( i ) } \left[ \mathrm { S u b O p t } _ { \omega ^ { ( i ) } } ( { \hat { \mu } } ) \right] ~ \geq ~ \frac { 1 } { 4 4 } R _ { \operatorname* { m a x } } \sqrt { \frac { d } { K } } ,
$$

where $\mathbb { E } ^ { ( i ) }$ is for the expectation under the trajectory law $\mathbb { P } ^ { ( i ) }$ induced by $\boldsymbol { \omega } ^ { ( i ) }$

Proof. Consider the hard instance of Definition 5 and note that in this environment, $d = d ^ { \circ }$ Let $\mathbb { P } ^ { 0 }$ denote the law of the trajectory under the null hypothesis $\omega ^ { ( 0 ) } : = 0$ . We use two budget constraints under $\mathbb { P } ^ { 0 }$ . First, since $\| \mu _ { k } \| _ { 2 } \leq \| \mu _ { k } \| _ { 1 } = 1$ 2

$$
\mathbb { E } ^ { 0 } \left[ \sum _ { k \le K } \| \mu _ { k } \| _ { 2 } ^ { 2 } \right] = \sum _ { i = 1 } ^ { d } \mathbb { E } ^ { 0 } \left[ \sum _ { k \le K } \mu _ { k , i } ^ { 2 } \right] \le K ,\tag{6}
$$

which corresponds to a bound on the collected Fisher information<sup>12</sup>.

Second, by Markov’s inequality and $\textstyle \sum _ { i } { \hat { \mu } } _ { i } = 1$ 2

$$
\sum _ { i = 1 } ^ { d } \mathbb { P } ^ { 0 } \left( \hat { \mu } _ { i } \geq \frac { 1 } { 2 } \right) \leq 2 \sum _ { i = 1 } ^ { d } \mathbb { E } ^ { 0 } \left[ \hat { \mu } _ { i } \right] = 2 .\tag{7}
$$

By Equation (6) at most $d / 4$ indices satisfy $\begin{array} { r } { \mathbb { E } ^ { 0 } [ \sum _ { k } \mu _ { k , i } ^ { 2 } ] > 4 K / d , } \end{array}$ , and by Equation (7) at most $d / 4$ indices satisfy $\mathbb { P } ^ { 0 } ( \hat { \mu } _ { i } \geq 1 / 2 ) > 8 / d .$ . Since $d / 4 + d / 4 < d /$ , there exists $i ^ { \star } \in [ d ]$ with

$$
\mathbb { E } ^ { 0 } \left[ \sum _ { k \le K } \mu _ { k , i \cdot } ^ { 2 } \right] \le \frac { 4 K } { d } \qquad \mathrm { a n d } \qquad \mathbb { P } ^ { 0 } \left( \hat { \mu } _ { i ^ { \star } } \ge \frac { 1 } { 2 } \right) \le \frac { 8 } { d } .\tag{8}
$$

Indistinguishability. By the decomposition of the KL (Lemma 22) and the KL of the softmax (Lemma 23), we have:

$$
\begin{array} { r l r } { D _ { \mathrm { K L } } ( \mathbb { P } ^ { 0 }  \mathbb { P } ^ { ( i ^ { * } ) } ) \leq \displaystyle \frac { \eta ^ { 2 } } { 8 } \sum _ { k \leq K } \mathbb { E } ^ { 0 } [ d _ { \mu _ { k } } ( 0 , \varepsilon e _ { i ^ { * } } ) ^ { 2 } ] \qquad } & { \mathrm { ( L e m m a s ~ 2 2 , ~ 2 3 ) } } \\ { \displaystyle = \frac { \eta ^ { 2 } \varepsilon ^ { 2 } } { 8 } \mathbb { E } ^ { 0 } [ \sum _ { k \leq K } \mu _ { k , i ^ { * } } ^ { 2 } ] } & { \mathrm { ( P r o p o s i t i o n ~ 1 ) } } \\ { \leq \displaystyle \frac { \eta ^ { 2 } \varepsilon ^ { 2 } K } { 2 d } } & { \mathrm { ( E q u a t i o n ~ ( 8 ) ) } } \\ { \displaystyle = \frac { 1 } { 8 } , } \end{array}
$$

where the last step uses $\eta ^ { 2 } \varepsilon ^ { 2 } ~ = ~ d / ( 4 K )$ Pinsker’s inequality gives $\mathrm { T V } ( \mathbb { P } ^ { 0 } , \mathbb { P } ^ { ( i ^ { \star } ) } ) \ \leq$ $\sqrt { D _ { \mathrm { K L } } / 2 } \le 1 / 4$ , so with Equation (8) and $d \geq 3 2$

$$
\begin{array} { r } { \mathbb { P } ^ { ( i ^ { \star } ) } \left( \widehat { \mu } _ { i ^ { \star } } \geq \frac { 1 } { 2 } \right) \leq \displaystyle \frac { 8 } { d } + \frac { 1 } { 4 } \leq \frac { 1 } { 2 } , \qquad \mathrm { h e n c e } \qquad \mathbb { P } ^ { ( i ^ { \star } ) } \left( \widehat { \mu } _ { i ^ { \star } } < \frac { 1 } { 2 } \right) \geq \frac { 1 } { 2 } . } \end{array}
$$

Performance implication. From Proposition 1 we have Sub $\operatorname { O p t } _ { \omega ( i ^ { \star } ) } ( \mu ) = R _ { \operatorname* { m a x } } \left[ \ell ( \eta \varepsilon ) - \ell ( \eta \varepsilon \mu _ { i ^ { \star } } ) \right]$ where we note that since $K \geq 4 d$

$$
\eta \varepsilon = \frac { 1 } { 2 } \sqrt { d / K } \leq \frac { 1 } { 4 } .
$$

Therefore, on the event $\{ \hat { \mu } _ { i ^ { \star } } < 1 / 2 \}$ we have,

$$
\begin{array} { l } { \mathrm { S u b O p t } _ { \omega ( \ast ^ { \ast } ) } ( \hat { \mu } ) \overset { ( i ) } { \geq } R _ { \operatorname* { m a x } } \left[ \ell ( \eta \varepsilon ) - \ell \left( \frac { \eta \varepsilon } { 2 } \right) \right] } \\ { \overset { ( i i ) } { \geq } \frac { R _ { \operatorname* { m a x } } \eta \varepsilon } { 2 } \underset { | x | \leq 1 } { \operatorname* { i n f } } \ \ell ^ { \prime } ( x ) } \\ { = \frac { R _ { \operatorname* { m a x } } \eta \varepsilon } { 2 } \cdot \frac { e } { ( 1 + e ) ^ { 2 } } } \\ { \qquad \geq \frac { R _ { \operatorname* { m a x } } \eta \varepsilon } { 1 1 } . } \end{array}
$$

where (i) and (ii) come from the monotonicity of ℓ and the mean value theorem, respectively. Combining with the indistinguishability bound above we get:

$$
\mathbb { E } ^ { ( i ^ { \star } ) } \left[ \mathrm { S u b O p t } _ { \omega ^ { ( i ^ { \star } ) } } ( { \hat { \mu } } ) \right] \geq { \frac { 1 } { 2 } } \cdot { \frac { R _ { \operatorname* { m a x } } \eta \varepsilon } { 1 1 } } = { \frac { R _ { \operatorname* { m a x } } \eta \varepsilon } { 2 2 } } = { \frac { 1 } { 4 4 } } R _ { \operatorname* { m a x } } { \sqrt { \frac { d } { K } } } ,
$$

where $d = d ^ { \circ }$ (Proposition 1).

## D Self-concordance properties

## D.1 Self-concordance of the log-likelihood in the span semi-norm

In this section, we derive self-concordance properties for the log-likelihood function, inspired by Lee & Oh (2025) but expressed in the span semi-norm.

Extending the standard self-concordance property (Bach, 2010), we define a self-concordancelike function in this semi-norm as follows.

Definition 6 (Self-concordance in the span semi-norm). A function $f : \mathbb { R } ^ { m }  \mathbb { R }$ is an M-self-concordant-like function in the span semi-norm if:

$$
| \phi ^ { \prime \prime \prime } ( s ) | \leq M \| b \| _ { \operatorname { s p a n } } \phi ^ { \prime \prime } ( s ) , \qquad \forall s \in \mathbb { R } ,
$$

where $\phi ( s ) : = f ( a + s b )$ for any $a , b \in \mathbb { R } ^ { m }$

Recall the instantaneous log-likelihood:

$$
l ( \tilde { \omega } | b _ { \tau } , \mu _ { \tau } ) = \eta \langle \psi _ { \mu _ { \tau } } ( b _ { \tau } ) , \tilde { \omega } \rangle - \log Z _ { \mu _ { \tau } } ( \tilde { \omega } ) .
$$

Now, let $- f _ { \tau }$ be a reparameterization with logits $z \in \mathbb { R } ^ { | B | }$

$$
f _ { \tau } ( z ) = \log \left( \sum _ { b ^ { \prime } } e ^ { \eta z _ { b ^ { \prime } } } \right) - \eta z _ { b _ { \tau } } ,
$$

and note that $f _ { \tau }$ is invariant to constant shifts: $f _ { \tau } ( z + a \cdot 1 ) = f _ { \tau } ( z )$ for all $z \in \mathbb { R } ^ { | B | } , \ a \in \mathbb { R }$ Then, we have the following Lemma.

Lemma 5. The function $f _ { \tau }$ is η-self-concordant-like in the span semi-norm.

Proof. For any $a , b \in \mathbb { R } ^ { | B | }$ , let $\phi ( s ) : = f _ { \tau } ( a + s b )$ and $p _ { i } ( s ) : = \left. e ^ { \eta ( a _ { i } + s b _ { i } ) } \right/ \sum _ { j } e ^ { \eta ( a _ { j } + s b _ { j } ) }$ for every $i \in [ | B | ]$ . Further, let $X _ { s }$ be the discrete random variable such that for all $i ,$ $\mathbb { P } ( X _ { s } = i ) = p _ { i } ( s )$

Then,

$$
\phi ^ { \prime \prime } ( s ) = \eta ^ { 2 } \mathrm { V a r } ( b _ { X _ { s } } ) , \qquad \phi ^ { \prime \prime \prime } ( s ) = \eta ^ { 3 } \mathbb { E } \left[ ( b _ { X _ { s } } - \mathbb { E } \left[ b _ { X _ { s } } \right] ) ^ { 3 } \right] .
$$

By Hölder’s inequality we then conclude:

$$
| \phi ^ { \prime \prime \prime } ( s ) | \leq \eta \| b - \mathbb { E } [ b _ { X _ { s } } ] \| _ { \infty } \phi ^ { \prime \prime } ( s ) \leq \eta \| b \| _ { \mathrm { s p a n } } \phi ^ { \prime \prime } ( s ) ,
$$

and therefore $f _ { \tau }$ is η-self-concordant-like.

Now similarly to Proposition B.4 in Lee $\& \mathrm { ~ O h ~ }$ (2025), we can derive the following.

Lemma 6. For any $\tau \in [ N ]$ and any $x , u _ { 1 } , u _ { 2 } \in \mathbb { R } ^ { | B | }$ , we have:

$$
\begin{array} { r l } & { \left| D ^ { 3 } f _ { \tau } ( x ) [ u _ { 1 } , u _ { 2 } , u _ { 2 } ] \right| \leq \eta \| u _ { 1 } \| _ { s p a n } \| u _ { 2 } \| _ { \nabla ^ { 2 } f _ { \tau } ( x ) } ^ { 2 } , } \end{array}
$$

where we define $D ^ { 3 } f _ { \tau } ( x ) [ u _ { 1 } , u _ { 2 } , u _ { 3 } ] : = \langle D ^ { 3 } f _ { \tau } ( x ) [ u _ { 1 } ] u _ { 2 } , u _ { 3 } \rangle$

Proof. Fix $x , u _ { 1 } , u _ { 2 } \in \mathbb { R } ^ { | B | }$ and define for every $s \in \mathbb { R }$ and $i \in [ | B | ]$ the function $p _ { i } ( s ) : =$ $e ^ { \eta ( x _ { i } + s u _ { 1 , i } ) } / \sum _ { j } e ^ { \eta ( x _ { j } + s u _ { 1 , j } ) }$

Additionally, let $X _ { s }$ a discrete random variable such that $\mathbb { P } ( X _ { s } = i ) = p _ { i } ( s )$ and note that for any $g : \mathbb { R }  \mathbb { R }$

$$
\frac { d } { d s } \mathbb { E } [ g ( X _ { s } ) ] = \eta \mathbb { E } [ g ( X _ { s } ) \cdot ( ( u _ { 1 } ) _ { X _ { s } } - \mathbb { E } [ ( u _ { 1 } ) _ { X _ { s } } ] ) ] .
$$

Hence,

$$
\begin{array} { l } { \displaystyle { D ^ { 3 } f _ { \tau } ( x ) [ u _ { 1 } , u _ { 2 } , u _ { 2 } ] = \eta ^ { 2 } \left. \frac { d } { d s } \mathrm { V a r } ( ( u _ { 2 } ) _ { X _ { s } } ) \right. _ { s = 0 } } } \\ { \displaystyle { \quad \quad = \eta ^ { 2 } \left. \frac { d } { d s } \left( \mathbb { E } [ ( u _ { 2 } ) _ { X _ { s } } ^ { 2 } ] - \mathbb { E } [ ( u _ { 2 } ) _ { X _ { s } } ] ^ { 2 } \right) \right. _ { s = 0 } } } \\ { \displaystyle { \quad \quad = \eta ^ { 3 } \mathbb { E } \left[ ( ( u _ { 2 } ) _ { X _ { 0 } } - \mathbb { E } [ ( u _ { 2 } ) _ { X _ { 0 } } ] ) ^ { 2 } ( ( u _ { 1 } ) _ { X _ { 0 } } - \mathbb { E } [ ( u _ { 1 } ) _ { X _ { 0 } } ] ) \right] . } } \end{array}
$$

Therefore by Hölder’s inequality,

$$
\left| D ^ { 3 } f _ { \tau } ( x ) [ u _ { 1 } , u _ { 2 } , u _ { 2 } ] \right| \leq \eta ^ { 3 } \| u _ { 1 } \| _ { \mathrm { s p a n } } \mathrm { V a r } \left( \left( u _ { 2 } \right) _ { X _ { 0 } } \right) = \eta \| u _ { 1 } \| _ { \mathrm { s p a n } } \| u _ { 2 } \| _ { \nabla ^ { 2 } f _ { \tau } ( x ) } ^ { 2 } .
$$

Finally, we derive the following two lemmas (variants of Propositions B.5 and B.6 of Lee & Oh (2025)), that we will use throughout the analysis of both algorithms.

Lemma 7. For all $\tau \in [ N ]$ and any $z _ { 1 } , z _ { 2 } \in \mathbb { R } ^ { | B | }$

$$
e ^ { - \eta \| z _ { 1 } - z _ { 2 } \| _ { s p a n } } \nabla ^ { 2 } f _ { \tau } ( z _ { 1 } ) \preceq \nabla ^ { 2 } f _ { \tau } ( z _ { 2 } ) \preceq e ^ { \eta \| z _ { 1 } - z _ { 2 } \| _ { s p a n } } \nabla ^ { 2 } f _ { \tau } ( z _ { 1 } ) .
$$

Proof. Fix $z _ { 1 } , z _ { 2 } , u \in \mathbb { R } ^ { | B | } , \tau \in [ N ]$ . Then for any $s \in [ 0 , 1 ]$ , let $g ( s ) : = u ^ { \top } \nabla ^ { 2 } f _ { \tau } \left( z _ { 1 } + s ( z _ { 2 } - z _ { 1 } ) \right)$ u such that $g ( 0 ) = \| u \| _ { \nabla ^ { 2 } f _ { \tau } ( z _ { 1 } ) } ^ { 2 }$ and $g ( 1 ) = \| u \| _ { \nabla ^ { 2 } f _ { \tau } ( z _ { 2 } ) } ^ { 2 } .$

Then from Lemma $6 .$

$$
| g ^ { \prime } ( s ) | \leq \eta \| z _ { 2 } - z _ { 1 } \| _ { \mathrm { s p a n } } g ( s ) .
$$

Using $\nabla ^ { 2 } f _ { \tau } ( z ) \succeq 0$ for all $z ,$ we then integrate as follows:

$$
\int _ { 0 } ^ { 1 } - \eta \| z _ { 2 } - z _ { 1 } \| _ { \mathrm { s p a n } } \mathrm { d } s \leq \int _ { 0 } ^ { 1 } \frac { \mathrm { d } \log g ( s ) } { \mathrm { d } s } \mathrm { d } s \leq \int _ { 0 } ^ { 1 } \eta \| z _ { 2 } - z _ { 1 } \| _ { \mathrm { s p a n } } \mathrm { d } s ,
$$

hence,

$$
e ^ { - \eta \| z _ { 2 } - z _ { 1 } \| _ { \mathrm { s p a n } } } \leq \frac { g ( 1 ) } { g ( 0 ) } \leq e ^ { \eta \| z _ { 2 } - z _ { 1 } \| _ { \mathrm { s p a n } } } ,
$$

from which we conclude.

Lemma 8. For any $\tau \in [ N ] , \ u , z _ { 1 } , z _ { 2 } \in \mathbb { R } ^ { | \mathcal { B } | }$

$$
u ^ { \top } \left( \int _ { 0 } ^ { 1 } ( 1 - s ) \nabla ^ { 2 } f _ { \tau } ( z _ { 1 } + s ( z _ { 2 } - z _ { 1 } ) ) d s \right) u \geq { \frac { 1 } { 2 + \eta \| z _ { 2 } - z _ { 1 } \| _ { s p a n } } } u ^ { \top } \nabla ^ { 2 } f _ { \tau } ( z _ { 1 } ) u .
$$

Proof. $\operatorname { I f } \| z _ { 2 } - z _ { 1 } \| _ { \mathrm { s p a n } } = 0$ , the relation holds directly with equality. Hence we assume below $\| z _ { 2 } - z _ { 1 } \| _ { \mathrm { s p a n } } > 0$

Applying Lemma $^ 7$ we get

$$
\begin{array} { r l } & { u ^ { \top } \left( \displaystyle \int _ { 0 } ^ { 1 } ( 1 - s ) \nabla ^ { 2 } f _ { \tau } ( z _ { 1 } + s ( z _ { 2 } - z _ { 1 } ) ) d s \right) u \geq u ^ { \top } \left( \displaystyle \int _ { 0 } ^ { 1 } ( 1 - s ) e ^ { - \eta \| s ( z _ { 2 } - z _ { 1 } ) \| _ { \mathrm { s p a n } } } \nabla ^ { 2 } f _ { \tau } ( z _ { 1 } ) d s \right) u } \\ & { \qquad \geq u ^ { \top } \nabla ^ { 2 } f _ { \tau } ( z _ { 1 } ) u \left( \displaystyle \frac { 1 } { \eta \| z _ { 2 } - z _ { 1 } \| _ { \mathrm { s p a n } } } + \displaystyle \frac { e ^ { - \eta \| z _ { 2 } - z _ { 1 } \| _ { \mathrm { s p a n } } } - 1 } { ( \eta \| z _ { 2 } - z _ { 1 } \| _ { \mathrm { s p a n } } ) ^ { 2 } } \right) } \\ & { \qquad \geq u ^ { \top } \nabla ^ { 2 } f _ { \tau } ( z _ { 1 } ) u \left( \displaystyle \frac { 1 } { 2 + \eta \| z _ { 2 } - z _ { 1 } \| _ { \mathrm { s p a n } } } \right) , } \end{array}
$$

where the third inequality uses the relation $\textstyle { \frac { 1 } { x } } \left( 1 + { \frac { e ^ { - x } - 1 } { x } } \right) \geq { \frac { 1 } { 2 + x } }$ for all $x > 0$

## D.2 Change of coeficients in the Fisher information

Noting that $- \nabla ^ { 2 } l ( \tilde { \omega } | b , \mu _ { \tau } ) = \mathcal { T } _ { \mu _ { \tau } } ( \tilde { \omega } )$ for any $b \in B$ and $\tau \in [ N ]$ , we apply the previous results to bound the Fisher information matrix on changes of coeficients.

Specifically, the previous results are expressed in the logit space via the function $f _ { \tau }$ . We turn back to the coeficient space by using the following linear lift:

$$
z _ { \mu } ( { \tilde { \omega } } ) : = \Psi _ { \mu } { \tilde { \omega } } \in \mathbb { R } ^ { | \mathcal { B } | } , \qquad \Psi _ { \mu } : = \left[ \psi _ { \mu } ( b ) ^ { \top } \right] _ { b \in \mathcal { B } } \in \mathbb { R } ^ { | \mathcal { B } | \times d } ,\tag{9}
$$

where $z _ { \mu } ( \omega )$ represents the logits associated to the policy $\mu$ and the coeficients $\tilde { \omega }$

Explicitly,

$$
- l ( \tilde { \omega } | b _ { \tau } , \mu _ { \tau } ) = \log \left( \sum _ { b ^ { \prime } \in \mathcal { B } } e ^ { \eta \langle \psi _ { \mu _ { \tau } } ( b ^ { \prime } ) , \tilde { \omega } \rangle } \right) - \eta \langle \psi _ { \mu _ { \tau } } ( b _ { \tau } ) , \tilde { \omega } \rangle = f _ { \tau } ( z _ { \mu _ { \tau } } ( \tilde { \omega } ) ) ,\tag{10}
$$

and logit diferences can be measured by $d _ { \mu } ( \omega _ { 1 } , \omega _ { 2 } ) = \lVert z _ { \mu } ( \omega _ { 1 } ) - z _ { \mu } ( \omega _ { 2 } ) \rVert _ { \mathrm { s p a n } } \mathrm { . }$

Further, note that for any $\tau , \tau ^ { \prime }$ , the function $f _ { \tau } , f _ { \tau ^ { \prime } }$ only difer by a linear term. Therefore, the hessian $H ( z ) : = \nabla ^ { 2 } f _ { \tau } ( z )$ doesn’t depend on τ . Hence, using Equation (10), we have for any $\mu { : }$

$$
\mathcal { T } _ { \mu } ( \omega ) = \Psi _ { \mu } ^ { \intercal } H \left( z _ { \mu } ( \omega ) \right) \Psi _ { \mu } .\tag{11}
$$

We can now state useful results following Appendix D.1 and these remarks.

Lemma 9. For any $\omega _ { 1 } , \omega _ { 2 } \in \Omega$ and $\mu _ { ; }$ ,

$$
e ^ { - \eta d _ { \mu } ( \omega _ { 1 } , \omega _ { 2 } ) } \mathscr { T } _ { \mu } ( \omega _ { 1 } ) \preceq \mathscr { T } _ { \mu } ( \omega _ { 2 } ) \preceq e ^ { \eta d _ { \mu } ( \omega _ { 1 } , \omega _ { 2 } ) } \mathscr { T } _ { \mu } ( \omega _ { 1 } ) .
$$

Proof. By Lemma $^ { 7 , }$

$$
e ^ { - \eta d _ { \mu } ( \omega _ { 1 } , \omega _ { 2 } ) } H ( z _ { \mu } ( \omega _ { 1 } ) ) \preceq H ( z _ { \mu } ( \omega _ { 2 } ) ) \preceq e ^ { \eta d _ { \mu } ( \omega _ { 1 } , \omega _ { 2 } ) } H ( z _ { \mu } ( \omega _ { 1 } ) ) .
$$

Now note that for any $A \preceq B$ , we have $\Psi _ { \mu } ^ { \top } A \Psi _ { \mu } \preceq \Psi _ { \mu } ^ { \top } B \Psi _ { \mu }$ , since for any $v \in \mathbb { R } ^ { d }$

$$
v ^ { \top } \left( \Psi _ { \mu } ^ { \top } A \Psi _ { \mu } - \Psi _ { \mu } ^ { \top } B \Psi _ { \mu } \right) v = \left( \Psi _ { \mu } v \right) ^ { \top } \left( A - B \right) ( \Psi _ { \mu } v ) \leq 0 .
$$

We therefore conclude using Equation (11).

Lemma 10. For any $\omega _ { 1 } , \omega _ { 2 } \in \Omega$ and $\mu ,$ denote $a : = \eta \cdot d _ { \mu } ( \omega _ { 1 } , \omega _ { 2 } )$ . Then,

$$
\frac { 1 } { 2 + a } \mathcal { T } _ { \mu } ( \omega _ { 1 } ) \preceq \int _ { 0 } ^ { 1 } ( 1 - s ) \mathcal { T } _ { \mu } ( \omega _ { 1 } + s ( \omega _ { 2 } - \omega _ { 1 } ) ) d s \preceq \frac { e ^ { a } - a - 1 } { a ^ { 2 } } \mathcal { T } _ { \mu } ( \omega _ { 1 } ) .
$$

Proof. Write $\omega _ { s } : = \omega _ { 1 } + s ( \omega _ { 2 } - \omega _ { 1 } )$ and $z _ { i } : = z _ { \mu } ( \omega _ { i } )$ , so that $z _ { \mu } ( \omega _ { s } ) = z _ { 1 } + s ( z _ { 2 } - z _ { 1 } )$ by linearity of the lift (Equation (9)).

Lower bound. Fix $v \in \mathbb { R } ^ { d }$ and apply Lemma $8$ to $u : = \Psi _ { \mu } v$ . Using Equation (11),

$$
\begin{array} { r l } { v ^ { \top } \left( \displaystyle \int _ { 0 } ^ { 1 } ( 1 - s ) \mathbb { Z } _ { \mu } ( \omega _ { s } ) d s \right) v = u ^ { \top } \left( \displaystyle \int _ { 0 } ^ { 1 } ( 1 - s ) H ( z _ { 1 } + s ( z _ { 2 } - z _ { 1 } ) ) d s \right) u } & { } \\ { \displaystyle \geq \frac { 1 } { 2 + a } u ^ { \top } H ( z _ { 1 } ) u } & { } \\ { \displaystyle } & { = \frac { 1 } { 2 + a } v ^ { \top } \mathbb { Z } _ { \mu } ( \omega _ { 1 } ) v . } \end{array}
$$

Upper bound. Since $\omega _ { s } - \omega _ { 1 } = s ( \omega _ { 2 } - \omega _ { 1 } )$ and $\eta d _ { \mu } ( \omega _ { s } , \omega _ { 1 } ) = s a$ for every $s \in [ 0 , 1 ]$ , Lemma 9 thus gives $\overline { { \mathcal { T } _ { \mu } ( } } \omega _ { s } ) \preceq e ^ { s a } \mathcal { T } _ { \mu } ( \omega _ { 1 } )$ . Integrating,

$$
\int _ { 0 } ^ { 1 } ( 1 - s ) \mathbb { Z } _ { \mu } ( \omega _ { s } ) \mathrm { d } s \preceq \left( \int _ { 0 } ^ { 1 } ( 1 - s ) e ^ { s a } \mathrm { d } s \right) \mathbb { Z } _ { \mu } ( \omega _ { 1 } ) = \frac { e ^ { a } - a - 1 } { a ^ { 2 } } \mathbb { Z } _ { \mu } ( \omega _ { 1 } ) .
$$

Both constants extend continuously to $a = 0$ , where they equal $\begin{array} { r } { \int _ { 0 } ^ { 1 } ( 1 - s ) d s = 1 / 2 } \end{array}$ □

Lemma 11. For any $\omega _ { 1 } , \omega _ { 2 } \in \Omega$ and $\mu ,$ , denote $a : = \eta \cdot d _ { \mu } ( \omega _ { 1 } , \omega _ { 2 } )$ . Then,

$$
\frac { 1 - e ^ { - a } } { a } \mathcal { T } _ { \mu } ( \omega _ { 1 } ) \preceq \Gamma _ { \mu } ( \omega _ { 1 } , \omega _ { 2 } ) \preceq \frac { e ^ { a } - 1 } { a } \mathcal { T } _ { \mu } ( \omega _ { 1 } ) ,
$$

with $T _ { \mu } ( \omega _ { 1 } ) \preceq \Gamma _ { \mu } ( \omega _ { 1 } , \omega _ { 2 } ) \preceq { \mathcal { T } } _ { \mu } ( \omega _ { 1 } )$ when $a = 0$

Proof. The proof is identical to the proof of Lemma 9, integrating $\mathcal { T } _ { \mu } ( \omega _ { s } )$ directly without the $( 1 - s )$ factor. □

Lemma 12 (Best-response stability). For every $\omega _ { 1 } , \omega _ { 2 }$ and $\mu _ { ; }$

$$
\left\| \nu _ { \mu } ^ { * } ( \omega _ { 1 } ) - \nu _ { \mu } ^ { * } ( \omega _ { 2 } ) \right\| _ { 1 } \leq \int _ { 0 } ^ { 1 } \| \omega _ { 1 } - \omega _ { 2 } \| _ { \mathcal { T } _ { \mu } ( \omega _ { s } ) } d s \leq \chi \left( \eta \cdot d _ { \mu } ( \omega _ { 1 } , \omega _ { 2 } ) \right) \cdot \| \omega _ { 1 } - \omega _ { 2 } \| _ { \mathcal { T } _ { \mu } ( \omega _ { 2 } ) } ,
$$

with $\begin{array} { r } { \chi ( x ) : = \frac { 2 ( e ^ { x / 2 } - 1 ) } { x } } \end{array}$ (increasing function) and $\omega _ { s } : = \omega _ { 2 } + s ( \omega _ { 1 } - \omega _ { 2 } )$

Proof. Write $a : = \eta \cdot d _ { \mu } ( \omega _ { 1 } , \omega _ { 2 } ) , p _ { s } : = \nu _ { \mu } ^ { \ast } ( \omega _ { s } ) \in \Delta ( \mathcal { B } )$ , and let $h ( b ) : = \langle \psi _ { \mu } ( b ) , \omega _ { 1 } - \omega _ { 2 } \rangle$ Note: since B is finite and $\nu _ { \mu } ^ { * }$ is a softmax, $s \mapsto p _ { s }$ is $C ^ { 1 }$ and $p _ { s } ( b ) > 0$ for every $b \in B$ By Equation (3) and $\mathrm { d } \omega _ { s } = \left( \omega _ { 1 } - \omega _ { 2 } \right)$ ds, for every $b \in B$ ，

$$
\frac { \mathrm { d } } { \mathrm { d } s } \log p _ { s } ( b ) = \eta \left( h ( b ) - \mathbb { E } _ { p _ { s } } \left[ h \right] \right) \Leftrightarrow \frac { \mathrm { d } } { \mathrm { d } s } p _ { s } ( b ) = \eta p _ { s } ( b ) \left( h ( b ) - \mathbb { E } _ { p _ { s } } [ h ] \right) .
$$

Taking the $\ell _ { 1 }$ norm and applying Jensen’s inequality to the concave map $x \mapsto { \sqrt { x } } .$

$$
\begin{array} { r l } & { \left. \frac { \mathrm { d } } { \mathrm { d } s } p _ { s } \right. _ { 1 } = \eta \mathbb { E } _ { p _ { s } } \left[ | h - \mathbb { E } _ { p _ { s } } [ h ] | \right] } \\ & { \qquad \leq \eta \sqrt { \mathrm { V a r } _ { p _ { s } } ( h ) } } \\ & { \qquad = \sqrt { ( \omega _ { 1 } - \omega _ { 2 } ) ^ { \top } \eta ^ { 2 } \mathrm { C o v } _ { b \sim p _ { s } } \left[ \psi _ { \mu } ( b ) \right] ( \omega _ { 1 } - \omega _ { 2 } ) } } \\ & { \qquad = \Vert \omega _ { 1 } - \omega _ { 2 } \Vert _ { \mathcal { I } _ { \mu } ( \omega _ { s } ) } . } \end{array}
$$

(Jensen’s inequality)

(exponential family)

The fundamental theorem of calculus and the triangle inequality then give

$$
\left\| { \nu _ { \mu } ^ { * } } ( \omega _ { 1 } ) - { \nu _ { \mu } ^ { * } } ( \omega _ { 2 } ) \right\| _ { 1 } = \left\| \int _ { 0 } ^ { 1 } { \frac { \mathrm { d } } { \mathrm { d } s } } p _ { s } \mathrm { d } s \right\| _ { 1 } \leq \int _ { 0 } ^ { 1 } \left\| { \frac { \mathrm { d } } { \mathrm { d } s } } p _ { s } \right\| _ { 1 } \mathrm { d } s \leq \int _ { 0 } ^ { 1 } \| \omega _ { 1 } - \omega _ { 2 } \| _ { \mathcal { T } _ { \mu } ( \omega _ { s } ) } \mathrm { d } s .
$$

Now, since $d _ { \mu }$ is induced by a semi-norm and $\omega _ { s } - \omega _ { 2 } = s ( \omega _ { 1 } - \omega _ { 2 } )$ , we have $\eta \cdot d _ { \mu } ( \omega _ { s } , \omega _ { 2 } ) = s a$ By Lemma 9, this implies ${ \mathcal { T } } _ { \mu } ( \omega _ { s } ) \preceq e ^ { s a } { \mathcal { T } } _ { \mu } ( \omega _ { 2 } )$ . Taking square roots and integrating,

$$
\int _ { 0 } ^ { 1 } \| \omega _ { 1 } - \omega _ { 2 } \| _ { \mathcal { I } _ { \mu } ( \omega _ { s } ) } \mathrm { d } s \leq \left( \int _ { 0 } ^ { 1 } e ^ { s a / 2 } \mathrm { d } s \right) \| \omega _ { 1 } - \omega _ { 2 } \| _ { \mathcal { I } _ { \mu } ( \omega _ { 2 } ) } = \chi ( a ) \cdot \| \omega _ { 1 } - \omega _ { 2 } \| _ { \mathcal { I } _ { \mu } ( \omega _ { 2 } ) } .
$$

Lemma 13 (Fisher uncertainty transfer). For any $\omega _ { 1 } , \omega _ { 2 } , \lambda , \mu$ and allocation $\{ \mu _ { \tau } \} _ { \tau \leq k }$

$$
\mathrm { T r a c e } ( G ^ { - 1 } ( \omega _ { 1 } ) \mathbb { Z } _ { \mu } ( \omega _ { 1 } ) ) \leq e ^ { 2 \eta d _ { * } ( \omega _ { 1 } , \omega _ { 2 } ) } \mathrm { T r a c e } ( G ^ { - 1 } ( \omega _ { 2 } ) \mathbb { Z } _ { \mu } ( \omega _ { 2 } ) ) ,
$$

where $\begin{array} { r } { G ( \tilde { \omega } ) = \lambda I + \sum _ { \tau = 1 } ^ { k } \mathcal { T } _ { \mu _ { \tau } } ( \tilde { \omega } ) } \end{array}$

Proof. Set $a _ { * } : = \eta \cdot d _ { * } ( \omega _ { 1 } , \omega _ { 2 } ) \geq 0$

Now noting that $d _ { \mu _ { \tau } } \leq d _ { * }$ for all τ , apply Lemma 9 and get:

$$
G ( \omega _ { 2 } ) = \lambda I + \sum _ { \tau = 1 } ^ { k } \bar { Z } _ { \mu _ { \tau } } ( \omega _ { 2 } ) \preceq \lambda I + e ^ { a _ { * } } \sum _ { \tau = 1 } ^ { k } \bar { Z } _ { \mu _ { \tau } } ( \omega _ { 1 } ) \preceq e ^ { a _ { * } } G ( \omega _ { 1 } ) ,
$$

where we also noted that $\lambda I \preceq e ^ { a _ { * } } \lambda I$ since $a _ { * } \geq 0$

Inverting on both sides we get:

$$
G ^ { - 1 } ( \omega _ { 2 } ) \succeq e ^ { - a _ { * } } G ^ { - 1 } ( \omega _ { 1 } ) \Leftrightarrow G ^ { - 1 } ( \omega _ { 1 } ) \preceq e ^ { a _ { * } } G ^ { - 1 } ( \omega _ { 2 } ) .\tag{12}
$$

We then use twice the fact that

$$
A _ { 1 } \preceq A _ { 2 } , C \succeq 0 \implies \mathrm { T r a c e } ( A _ { 1 } C ) \leq \mathrm { T r a c e } ( A _ { 2 } C ) , \qquad \forall A _ { 1 } , A _ { 2 } , C ,
$$

which follows from Trace $( ( A _ { 2 } - A _ { 1 } ) C ) = \mathrm { T r a c e } \left( C ^ { 1 / 2 } ( A _ { 2 } - A _ { 1 } ) C ^ { 1 / 2 } \right) \geq 0 .$

Specifically,

$$
\begin{array} { r l r } { \mathrm { T r a c e } \left( G ^ { - 1 } ( \omega _ { 1 } ) \mathcal { T } _ { \mu } ( \omega _ { 1 } ) \right) \leq e ^ { a _ { * } } \mathrm { T r a c e } \left( G ^ { - 1 } ( \omega _ { 2 } ) \mathcal { T } _ { \mu } ( \omega _ { 1 } ) \right) } & { \qquad } & { \mathrm { ( E q u a t i o n ~ ( 1 2 ) ) } } \\ { \leq e ^ { 2 a _ { * } } \mathrm { T r a c e } \left( G ^ { - 1 } ( \omega _ { 2 } ) \mathcal { T } _ { \mu } ( \omega _ { 2 } ) \right) } & { \qquad } & { \mathrm { ( L e m m a ~ 9 ) . } } \end{array}
$$

## E Warm-up analysis

An ofline warm-up allows to directly optimize a set of exploratory policies to reduce the maximum uncertainty. However, during this ofline phase we cannot collect information about the unknown reward coeficients, which prevents the algorithm to query the best-responding player in the optimal directions.

We therefore perform optimal design assuming a uniform best-responding agents $( \mathrm { i } . \mathrm { e } . \ \omega =$ $0 \in \Omega )$ , and pay for this choice with a global self-concordance introducing an exponential term κ in the burn-in phase for exploration. This exponential dependency is then removed in the online phase, which relies on a running estimate for $\omega$ (namely, the maximum likelihood estimator).

Formally, this price is given by the following lemma.

Lemma 14 (Design curvature). For $n \geq 1$ and any allocation $\{ \mu _ { \tau } \} _ { \tau \leq n }$ , define the uniformcentered design $\begin{array} { r } { V _ { n } : = \sum _ { \tau = 1 } ^ { n } M _ { \mu } } \end{array}$ and $\Gamma _ { n } ^ { 0 }$ as in Theorem 5 with $\lambda = 0$ . Then for any $\tilde { \omega } \in \Omega$

$$
G ^ { 0 } \left( \tilde { \omega } | \{ \mu _ { \tau } \} _ { \tau \leq n } \right) \succeq 2 \kappa ^ { - 1 } V _ { n } \quad a n d \quad \Gamma _ { n } ^ { 0 } ( \tilde { \omega } ) \succeq \kappa ^ { - 1 } V _ { n } ,
$$

for $\kappa : = 2 \eta ^ { - 2 } e ^ { 2 \eta S }$

Hence for any $v \in \mathbb { R } ^ { d } , \| v \| _ { V _ { n } } ^ { 2 } \leq \kappa \| v \| _ { \Gamma _ { n } ^ { 0 } ( \tilde { \omega } ) } ^ { 2 }$

Proof. Denote by $p _ { \mathrm { m i n } } \geq e ^ { - 2 \eta S } / | B |$ a lower bound on the minimal probability of the softmax distribution with normalized features and coeficient $\tilde { \omega } \in \Omega$ . Hence, for any $\tau \in [ n ]$ ，

$$
\begin{array} { r l } { \operatorname { C o v } _ { b \sim \mathbb { P } ( \cdot | \tilde { \omega } , \mu _ { \tau } ) } \left[ \psi _ { \mu _ { \tau } } ( b ) \right] \succeq p _ { \operatorname* { m i n } } \displaystyle \sum _ { b \in \mathcal { B } } \left( \psi _ { \mu _ { \tau } } ( b ) - \overline { { \psi } } _ { \mu _ { \tau } } ( \tilde { \omega } ) \right) ^ { \otimes 2 } } & { } \\ { \succeq | \mathcal { B } | p _ { \operatorname* { m i n } } \cdot M _ { \mu _ { \tau } } , } \end{array}
$$

where the second step re-centers at the uniform mean $\begin{array} { r } { ( \sum _ { b } ( x _ { b } - c ) ^ { \otimes 2 } \succeq \sum _ { b } ( x _ { b } - \overline { { x } } ) ^ { \otimes 2 } } \end{array}$ for any c).

Now fix $s \in [ 0 , 1 ]$ , write $\omega _ { s } : = \tilde { \omega } + s ( \omega - \tilde { \omega } ) \in \Omega$ , and use the previous inequality to show<sup>13</sup>:

$$
\begin{array} { r l } {  { G ^ { 0 } ( \omega _ { s } | \{ \mu _ { \tau } \} _ { \tau \leq n } ) = \eta ^ { 2 } \sum _ { \tau = 1 } ^ { n } \mathrm { C o v } _ { b \sim \mathbb { P } ( \cdot | \omega _ { s } , \mu _ { \tau } ) } [ \psi _ { \mu _ { \tau } } ( b ) ] } } \\ & { \leq \eta ^ { 2 } | \mathcal { B } | p _ { \operatorname* { m i n } } \sum _ { \tau = 1 } ^ { n } M _ { \mu _ { \tau } } } \\ & { \leq \eta ^ { 2 } e ^ { - 2 \eta S } \cdot V _ { n } . } \end{array}
$$

In particular, for $s = 0$ , we have

$$
G ^ { 0 } \left( \omega _ { 0 } | \{ \mu _ { \tau } \} _ { \tau \leq n } \right) = G ^ { 0 } \left( \tilde { \omega } | \{ \mu _ { \tau } \} _ { \tau \leq n } \right) \succeq \eta ^ { 2 } e ^ { - 2 \eta S } \cdot V _ { n } = 2 \kappa ^ { - 1 } V _ { n } ,
$$

with $\begin{array} { r } { \kappa : = \frac { 2 e ^ { 2 \eta S } } { \eta ^ { 2 } } } \end{array}$

Integrating the general term over [0, 1] also gives:

$$
\Gamma _ { n } ^ { 0 } ( \tilde { \omega } ) = \int _ { 0 } ^ { 1 } ( 1 - s ) G ^ { 0 } \left( \omega _ { s } | \{ \mu _ { \tau } \} _ { \tau \le n } \right) \mathrm { d } s \succeq \frac { \eta ^ { 2 } e ^ { - 2 \eta S } } { 2 } V _ { n } = \kappa ^ { - 1 } V _ { n } .
$$

For simplicity, all results in this section will be stated for the µ-player. It is straightforward to extend them to the ν-player by a change of notation.

Algorithm 2 Warm-up procedure (WARM)   
1: Input: tolerance ϵ, budget $n _ { \mathrm { w a r m } } .$   
2: Compute warm-up design $( o f f i n e )$ ▷ See Section E.2   
3: Approximately solve the following grouped design problem:   
$\operatorname * { a r g m i n } _ { \tilde { w } \in \Delta ( \Pi _ { \mu } ) } \operatorname* { m a x } _ { \mu } \operatorname* { m a x } _ { b \in \mathcal { B } } \| g _ { \mu } ( b ) \| _ { V ( \tilde { w } ) , * } ^ { 2 } ,$   
and obtain $w _ { \approx }$ an $( 1 + \epsilon ) | B |$ approximation.   
4: This requires an iterative algorithm for optimal design, presented in Algorithm 3.   
5: Rounding procedure ▷ See Section E.3   
6: Apply the rounding procedure of Definition $^ 7$ to $w _ { \approx }$ and obtain the allocation   
$\mathbf { v } = [ v _ { 1 } , . . . , v _ { s } ] \in \mathbb { N } ^ { s }$ such that $1 ^ { \top } \mathbf { v } = n _ { \mathrm { w a r m } } ,$   
7: where $s \leq d ^ { \circ } ( d ^ { \circ } + 1 ) / 2 + 1 .$   
8: Sample and initialize   
9: Sample $\begin{array} { r l } { \mathcal { D } _ { \mathrm { w a r m } } ; ~ } & { { } \mathcal { L } _ { 0 } ( \tilde { \omega } ) \gets \mathcal { L } ( \tilde { \omega } | \mathcal { D } _ { \mathrm { w a r m } } ) ; \quad \hat { \omega } _ { 0 } \gets \arg \operatorname* { m a x } _ { \tilde { \omega } \in \Omega } \mathcal { L } _ { 0 } ( \tilde { \omega } ) . } \end{array}$   
10: Output: $\mathcal { D } _ { \mathrm { w a r m } } , \mathcal { L } _ { 0 } ( \tilde { \omega } )$ and $\hat { \omega } _ { 0 } .$

We first start by introducing the precise version of the warm-up phase in Algorithm 1 from the main text. We will then prove its guarantees and usefulness for our main IRL algorithm (Algorithm 1).

The overall idea is the following: suppose we have a strategy $w \in \Delta ( \Pi _ { \mu } )$ such that ma $\mathrm { x } _ { \mu , b } \| g _ { \mu } ( b ) \| _ { V ( w ) , * } ^ { 2 } \leq c$ for some constant c. Then, for any $\begin{array} { r } { N \in \mathbb { N } , \operatorname* { m a x } _ { \mu , b } \lVert g _ { \mu } ( \dot { b } ) \rVert _ { N \cdot V ( w ) , * } ^ { 2 } \leq } \end{array}$ $c / N$ . The norm can therefore be arbitrarily small as we increase N.

As w is a probability measure over policies, N · w can be seen as a fractional allocation, and an exact allocation when all masses $N \cdot w ( \{ \mu \} )$ are integers. Instead of stating it as an assumption, we apply a rounding technique inspired by Allen-Zhu et al. (2021) in order to obtain a warm-up allocation v with $1 ^ { \top } v = n _ { \mathrm { w a r m } }$ for some integer $n _ { \mathrm { w a r m } }$ . This allocation determines a sequence $\{ \mu _ { \tau } ^ { \mathrm { w a r m } } \} _ { \tau \leq n _ { \mathrm { w a r m } } }$ that will be used to collect data (Algorithm 2, line 9). Before explaining how we solve the warm-up optimization problem and obtain the allocation via rounding, let us start by extending Kiefer-Wolfowitz theorems for optimal design to our setting.

## E.1 A grouped Kiefer-Wolfowitz theory

The classical Kiefer-Wolfowitz theorem (Kiefer & Wolfowitz, 1960) covers rank-one arms (e.g. see how it is used in Allen-Zhu et al. (2021) in the proof of Lemma 1), and therefore doesn’t directly apply to our setting. In our case, we select a leader policy µ and observe (in expectation) a covariance matrix. Hence, a "grouped" update as the expectation of an outer product of averaged features. We thus need a grouped version of KW theory, that we give here in two steps: the trace value (Lemma 15) and the pointwise value needed by the warm-up (Corollary 1).

Lemma 15 (Grouped Kiefer-Wolfowitz). We have,

$$
\operatorname* { m i n } _ { w \in \Delta ( \Pi _ { \mu } ) } \operatorname* { m a x } _ { \mu } \mathrm { ~ T r a c e } \left( V ( w ) ^ { \dagger } M _ { \mu } \right) = d ^ { \diamondsuit } ,
$$

the minimum being attained at any $w _ { D } \in \mathrm { a r g } \operatorname* { m a x } _ { \tilde { w } }$ log det<sub>G</sub> $V ( \tilde { w } ) \subseteq \mathcal { W } _ { \mathcal { G } }$

Proof. First of all, note that $w _ { D }$ must be admissible, since otherwise the objective is $+ \infty$ (Equation (4)) and can be made finite by any admissible distribution, e.g. w<sub>0</sub> the uniform distribution over any basis of $\mathcal { G }$ . Therefore,

$$
\operatorname { r a n k } V ( w _ { D } ) = d ^ { \diamondsuit } , \qquad \operatorname { T r a c e } \left( V ( w _ { D } ) ^ { \dagger } V ( w _ { D } ) \right) = d ^ { \diamondsuit } .
$$

Also note that log det<sub>G</sub> $V ( w _ { D } ) > - \infty$ , and $w _ { D }$ is well defined: $w \mapsto V ( w )$ has compact range and log det<sub>G</sub> is upper semi-continuous on it.

Now, to prove the equality, we will prove the upper and lower bounds separately.

Upper bound

For any $\mu \in \Pi _ { \mu }$ and $t \in [ 0 , 1 )$ , note that $( 1 - t ) w _ { D } + t \delta _ { \mu } \in \mathcal { W } _ { \mathcal { G } }$ and:

$$
V ( ( 1 - t ) w _ { D } + t \delta _ { \mu } ) = ( 1 - t ) V ( w _ { D } ) + t M _ { \mu } .
$$

From the first-order optimality condition, and given $V ( w _ { D } ) \in \mathbb { S } _ { + + } ( \mathcal { G } )$ and $M _ { \mu } \in \mathbb { S } _ { + } ( \mathcal G )$ 2

$$
\begin{array} { l } { { 0 \geq \displaystyle \frac { d } { d t } \log \operatorname* { d e t } _ { \mathcal { G } } \left[ V ( ( 1 - t ) w _ { D } + t \delta _ { \mu } ) \right] \big | _ { t = 0 + } } } \\ { { \ } } \\ { { \ } } \\ { { \displaystyle \ = \frac { d } { d t } \log \operatorname* { d e t } _ { \mathcal { G } } \left[ ( 1 - t ) V ( w _ { D } ) + t M _ { \mu } \right] \big | _ { t = 0 + } } } \\ { { \ } } \\ { { \displaystyle = \mathrm { T r a c e } \left( V ( w _ { D } ) ^ { \dagger } \left( M _ { \mu } - V ( w _ { D } ) \right) \right) } } \\ { { \ } } \\ { { \displaystyle = \mathrm { T r a c e } \left( V ( w _ { D } ) ^ { \dagger } M _ { \mu } \right) - d ^ { \diamond } . } } \end{array}
$$

Hence, $\operatorname* { m a x } _ { \mu }$ Trace $\left( V ( w _ { D } ) ^ { \dagger } M _ { \mu } \right) \leq d ^ { \circ }$

Lower bound

For any $\tilde { w } \in \mathcal { W } _ { \mathcal { G } }$ , we have:

$$
\begin{array} { l } { \displaystyle \mathop { \operatorname* { m a x } } _ { \mu } \mathrm { T r a c e } \left( V ( \tilde { w } ) ^ { \dagger } M _ { \mu } \right) \geq \int _ { \Pi _ { \mu } } \mathrm { T r a c e } \left( V ( \tilde { w } ) ^ { \dagger } M _ { \mu } \right) \tilde { w } ( \mathrm { d } \mu ) } \\ { \displaystyle = \mathrm { T r a c e } \left( V ( \tilde { w } ) ^ { \dagger } V ( \tilde { w } ) \right) } \\ { \displaystyle = d ^ { \circ } . } \end{array}
$$

This implies the lower bound and concludes the proof.

One direct consequence of Lemma 15 is that it allows us to directly bound the norm of the uniformly-centered features.

Corollary 1 (Pointwise grouped Kiefer-Wolfowitz). Using the same notation,

$$
\operatorname* { m i n } _ { w \in \Delta ( \Pi _ { \mu } ) } \operatorname* { m a x } _ { \mu , b } \| g _ { \mu } ( b ) \| _ { V ( w ) , * } ^ { 2 } \leq | B | d ^ { \circ } .
$$

Proof. Take $w _ { D }$ the D-optimal design of Lemma 15. Then, for any $\mu , b \colon$

$$
\begin{array} { r l } {  { \big \| g _ { \mu } ( b ) \big \| _ { V ( w _ { D } ) , * } ^ { 2 } \leq \sum _ { b ^ { \prime } \in B } \big \| g _ { \mu } ( b ^ { \prime } ) \big \| _ { V ( w _ { D } ) , * } ^ { 2 } } } \\ & { = \operatorname { T r a c e } ( V ( w _ { D } ) ^ { \dagger } \sum _ { b ^ { \prime } \in B } g _ { \mu } ( b ^ { \prime } ) g _ { \mu } ( b ^ { \prime } ) ^ { \top } ) } \\ & { = | B | \operatorname { T r a c e } ( V ( w _ { D } ) ^ { \dagger } M _ { \mu } ) } \\ & { \leq | B | d ^ { \circ } . } \end{array}
$$

Hence the upper bound is valid for the minimizer.

## E.2 Solving for the warm-up allocation

The warm-up needs an approximate optimal design w with ma $\begin{array} { r } { \mathrm { x } _ { \mu , b } \| g _ { \mu } ( b ) \| _ { V ( w ) , * } ^ { 2 } \leq ( 1 + \epsilon ) | B | d ^ { \circ } } \end{array}$ By the proof of Corollary 1 it sufices to compute a design $w _ { D }$ whose grouped trace certificate is at most $( 1 + \epsilon ) d ^ { \diamond }$

We first note that it is suficient for $w _ { D }$ to assign weights to pure policies, i.e. the optimization can be carried over $\Pi _ { \mu } = \Delta ( \mathcal { A } )$ directly instead of the more complex $\Delta ( \Pi _ { \mu } ) = \Delta ( \Delta \mathcal { A } )$ . This is shown below.

Lemma 16 (Reduction to pure leader actions). Let $\delta _ { a }$ denote the pure policy concentrated on action $a \in { \mathcal { A } }$ . Then, for every $V \in \mathbb { S } _ { + + } ( \mathcal G )$

$$
\operatorname * { m a x } _ { \mu \in \Pi _ { \mu } } ~ \mathrm { T r a c e } \left( V ^ { \dagger } M _ { \mu } \right) = \operatorname * { m a x } _ { a \in \mathcal { A } } ~ \mathrm { T r a c e } \left( V ^ { \dagger } M _ { \delta _ { a } } \right) ,
$$

and,

$$
\operatorname* { m a x } _ { \mu \in \Pi _ { \mu } , b \in \mathcal { B } } \| g _ { \mu } ( b ) \| _ { V , * } ^ { 2 } = \operatorname* { m a x } _ { a \in A , b \in \mathcal { B } } \| g _ { \delta _ { a } } ( b ) \| _ { V , * } ^ { 2 } .
$$

Proof. Note that every policy $\mu \in \Pi _ { \mu }$ is a mixture of pure policies:

$$
\mu = \sum _ { a \in \mathcal { A } } \mu ( a ) \delta _ { a } \qquad \mathrm { s u c h ~ t h a t } \qquad g _ { \mu } ( b ) = \sum _ { a \in \mathcal { A } } \mu ( a ) g _ { \delta _ { a } } ( b ) = \langle \mu , \Phi _ { b } \rangle ,
$$

with $\Phi _ { b } = [ g _ { \delta _ { a } } ( b ) ] _ { a \in \mathcal { A } }$

Therefore, since $g _ { \mu } ( b ) \in \mathcal { G }$

$$
\Vert g _ { \mu } ( b ) \Vert _ { V , * } ^ { 2 } = \mu ^ { \top } U _ { b } \mu , \quad \mathrm { T r a c e } ( V ^ { \dagger } M _ { \mu } ) = \frac { 1 } { | \mathcal { B } | } \sum _ { b \in \mathcal { B } } \Vert g _ { \mu } ( b ) \Vert _ { V , * } ^ { 2 } = \mu ^ { \top } \left( \frac { 1 } { | \mathcal { B } | } \sum _ { b \in \mathcal { B } } U _ { b } \right) \mu ,
$$

Algorithm 3 Frank–Wolfe for optimal design   
1: Input: features $\{ \phi ( a , b ) \} _ { a \in \mathcal { A } , b \in \mathcal { B } } ,$ tolerance $\epsilon > 0 .$   
2: Admissible initialization   
3: Pick $d ^ { \circ }$ pairs $( a _ { i } , b _ { i } )$ with $\{ g _ { \delta _ { a _ { i } } } ( b _ { i } ) \} _ { i \leq d ^ { \circ } }$ a basis of ${ \mathcal { G } } .$   
4: Let $w _ { 0 }$ be uniform on the distinct ${ { a } _ { i } } .$   
5: for t = 0, 1, 2, . . . do   
Linear oracle (enumeration over pure actions, Lemma $1 6 )$   
7: V<sub>t</sub> ← V (w<sub>t</sub>); h<sub>t</sub>(a) ← Trace $\left( V _ { t } ^ { \dagger } M _ { \delta _ { a } } \right)$ for a ∈ A; a<sub>t</sub> ∈ arg max $h _ { t } ( a )$   
8: Duality gap and stopping test   
9: $\mathrm { g a p } _ { t }  h _ { t } ( a _ { t } ) - d ^ { \circ } ;$ if $\mathrm { g a p } _ { t } \leq \epsilon d ^ { \circ }$ then return $w _ { t } .$   
10: Update   
11: γ<sub>t</sub> ← arg max $\gamma \in [ 0 , 1 ]$ log det<sub>G</sub> $\Big [ ( 1 - \gamma ) V _ { t } + \gamma M _ { \delta _ { a _ { t } } } \Big ] ; ~ w _ { t + 1 } \gets ( 1 - \gamma _ { t } ) w _ { t } + \gamma _ { t } \delta _ { a _ { t } } .$   
12: end for

with $U _ { b } : = \Phi _ { b } ^ { \top } V ^ { \dagger } \Phi _ { b } \succeq 0$

This concludes the proof as convex functions on a polytope attain their maximum at a vertex. □

This was the last key needed for building a Frank-Wolfe iterative algorithm to solve for the policy distribution by only optimizing over pure policies.

Specifically, for any running uniform coverage matrix $V ( w )$ with admissible $w ,$ we can compute the directional derivative for any $w ^ { \prime } \in \Delta ( \Pi _ { \mu } )$ as

$$
k ( w , w ^ { \prime } ) : = \left. { \frac { \mathrm { d } } { \mathrm { d } t } } \log \operatorname* { d e t } _ { \mathcal { G } } V ( w + t ( w ^ { \prime } - w ) ) \right| _ { t = 0 ^ { + } } = \int _ { \Pi _ { \mu } } h _ { w } ( \mu ) w ^ { \prime } ( \mathrm { d } \mu ) - d ^ { \circ } ,
$$

where $h _ { w } ( \mu ) : = \operatorname { T r a c e } \left( V ( w ) ^ { \dagger } M _ { \mu } \right)$ . By linearity, we can concentrate the mass of any optimal $w ^ { \prime }$ on a single policy $\mu ^ { + } \in \Pi _ { \mu }$ , while by Lemma 16, this $\mu ^ { + }$ can be taken as a pure policy. These two remarks reduce the optimization of $w ^ { \prime }$ to an optimization over $\mathcal { A } \mathrm { : }$

$$
d ^ { \diamond } + \operatorname* { m a x } _ { w ^ { \prime } \in \Delta ( \Pi _ { \mu } ) } \ k ( w , w ^ { \prime } ) = \operatorname* { m a x } _ { \mu ^ { + } \in \Pi _ { \mu } } \ h _ { w } ( \mu ^ { + } ) = \operatorname* { m a x } _ { a \in \mathcal { A } } \ h _ { w } ( \delta _ { a } ) .
$$

With these simplifications, the algorithm is presented in Algorithm 3. Note that it is purely ofline and costs $O ( | { \mathcal { A } } | | B | d ^ { 2 } )$ computations per iteration. Additionally, it terminates in finite time for every $\epsilon > 0$ by standard Frank-Wolfe analysis (see e.g. Zhao & Freund (2023)).

Running Algorithm 3 provides the following guarantee.

Proposition 2 (Certified output). IfAlgorithm 3 returns $w _ { t }$ , then ma ${ \zeta } _ { \mu }$ Trace $\left( V ( w _ { t } ) ^ { \dagger } M _ { \mu } \right) \leq$ $( 1 + \epsilon ) d ^ { \circ }$ and consequently

$$
\operatorname* { m a x } _ { \mu , b } \Vert g _ { \mu } ( b ) \Vert _ { V ( w _ { t } ) , * } ^ { 2 } \leq ( 1 + \epsilon ) \left| \mathcal { B } \right| d ^ { \circ } .
$$

Proof. The stopping test gives

$$
h _ { t } ( a _ { t } ) = \operatorname* { m a x } _ { a } ~ \operatorname { T r a c e } ( V ( w _ { t } ) ^ { \dagger } M _ { \delta _ { a } } ) \leq ( 1 + \epsilon ) d ^ { \diamond } ,
$$

which extends to all $\mu \in \Pi _ { \mu }$ by Lemma 16.

The result then follows exactly as in the proof of Corollary 1, replacing the bound $d ^ { \circ }$ by $( 1 + \epsilon ) d ^ { \circ }$ □

## E.3 Rounding and certificate

We now focus on turning the distribution over policies into an allocation strategy. Precisely, after obtaining a certified output from the Frank-Wolfe algorithm (Algorithm 3) with tolerance ϵ (Proposition 2), we apply a support reduction mechanism (Lemma 17) followed by integer rounding to turn the distribution over policies into an allocation (Definition 7).

Lemma 17 (Carathéodory support reduction). Let $w ^ { \star }$ be an output of Algorithm 3 satisfying $\begin{array} { r } { \operatorname* { m a x } _ { \mu , b } \| g _ { \mu } ( b ) \| _ { V ( w ^ { \star } ) ^ { \dagger } } ^ { 2 } = \rho < \infty } \end{array}$ . Then, there exists a design $\boldsymbol { w } _ { 0 } ^ { * }$ of support $s \leq d ^ { \circ } ( d ^ { \circ } + 1 ) / 2 + 1$ such that

$$
\operatorname* { m a x } _ { \mu , b } \bigl \| g _ { \mu } ( b ) \bigr \| _ { V ( w _ { 0 } ^ { \star } ) , * } ^ { 2 } = \rho ,
$$

and $\boldsymbol { w } _ { 0 } ^ { * }$ can be computed eficiently from $w ^ { * }$

Proof. Set V := conv $\{ M _ { \mu } : \mu \in \Pi _ { \mu } \}$ the convex hull spanned by uniform coverage matrices. Note that these matrices live in a convex cone of ambient dimension

$$
D : = \dim { ( \mathbb { S } ( \mathcal { G } ) ) } = \dim \left( \mathbb { S } ^ { d ^ { \circ } } \right) = d ^ { \circ } ( d ^ { \circ } + 1 ) / 2 ,
$$

which follows from the isomorphism $\mathbb { S } ( \mathcal { G } ) \cong \mathbb { S } ^ { d ^ { \circ } }$

V is compact and convex as $\{ M _ { \mu } \} _ { \mu }$ is the continuous image of the compact $\Pi _ { \mu } = \Delta ( \mathcal { A } )$

Therefore, by Carathéodory’s theorem, for some $s \leq D + 1$ , there exists a set of policies $\mu _ { 1 } , \ldots , \mu _ { s }$ with associated matrices $M _ { \mu _ { 1 } } , \ldots , M _ { \mu _ { s } }$ , such that:

$$
\int _ { \Pi _ { \mu } } M _ { \mu } w ^ { * } ( \mathrm { d } \mu ) = \sum _ { i = 1 } ^ { s } \alpha _ { i } M _ { \mu _ { i } } , \quad \mathrm { f o r ~ s o m e ~ } \alpha \in \Delta _ { s - 1 } .
$$

In particular, there exists a design $\boldsymbol { w } _ { 0 } ^ { * }$ of support s such that $V ( w ^ { * } ) = V ( w _ { 0 } ^ { * } )$

Now, note that by definition in the Frank-Wolfe procedure:

$$
| \mathrm { s u p p } ( w _ { t + 1 } ) | \leq | \mathrm { s u p p } ( w _ { t } ) | + 1 ,
$$

thus $| \mathrm { s u p p } ( w ^ { * } ) | \leq d ^ { \circ } + T$ , where $T < \infty$ is the stopping time of Algorithm 3.

The coeficient $w ^ { * }$ has finite support, and therefore we can apply the standard polynomial Carathéodory reduction (Grötschel et al., 1988) to recover some valid $\boldsymbol { w } _ { 0 } ^ { * }$

This concludes the proof.

This support reduction allows for the following simple rounding scheme.

Definition 7 (Warm-up allocation). Let $w _ { 0 } ^ { * } = \{ w _ { 0 , i } ^ { * } \} _ { i \leq s }$ be the support-reduced version of the output of Algorithm 3, obtained by Lemma 17. Define $o \in \mathbb { N } ^ { s }$ the order of $w _ { 0 } ^ { \ast }$ breaking ties by index such that:

$$
w _ { 0 , o _ { 1 } } ^ { \star } \geq \cdot \cdot \cdot \geq w _ { 0 , o _ { s } } ^ { \star } .
$$

Given $n _ { \mathrm { { w a r m } } } \geq s ,$ the warm-up allocation is the vector $v \in \mathbb { N } ^ { s }$ defined by

$$
v _ { o _ { i } } : = \underbrace { \left[ \left( n _ { \mathrm { w a r m } } - s \right) w _ { 0 , o _ { i } } ^ { \star } \right] } _ { \mathrm { b a s e } _ { o _ { i } } } + \mathbf { 1 } \left\{ i \leq \Delta \right\} , \qquad \Delta : = n _ { \mathrm { w a r m } } - \sum _ { j \leq s } \mathrm { b a s e } _ { j } .
$$

That is, the residual samples are given to the heaviest support points and $1 ^ { \top } v = n _ { \mathrm { w a r m } }$

Remark 3. We do not argue that this is the most eficient rounding procedure, but this is simple enough for our algorithm.

Definition 8 (Allocation uniform coverage). Let v be the warm-up allocation with arbitrary enumeration $\{ \mu _ { \tau } ^ { \mathrm { w a r m } } \} _ { \tau \le n _ { \mathrm { w a r m } } }$ . We denote the allocation uniform coverage by:

$$
V _ { \mathrm { w a r m } } : = \sum _ { \tau = 1 } ^ { n _ { \mathrm { w a r m } } } M _ { \mu _ { \tau } ^ { \mathrm { w a r m } } } = \frac { 1 } { | \mathcal { B } | } \sum _ { \tau = 1 } ^ { n _ { \mathrm { w a r m } } } \sum _ { b \in \mathcal { B } } g _ { \mu _ { \tau } ^ { \mathrm { w a r m } } } ( b ) g _ { \mu _ { \tau } ^ { \mathrm { w a r m } } } ( b ) ^ { \top } .
$$

Note that $V _ { \mathrm { w a r m } }$ is not normalized, so that its dual norm shrinks when $n _ { \mathrm { w a r m } }$ grows. We make that precise in the lemma below.

Before introducing the lemma, let us define the following term:

$$
\begin{array} { r } { C _ { \mathrm { r o u n d i n g } } : = \epsilon ^ { - 1 } ( 1 + \epsilon ) s \leq ( 2 \epsilon ) ^ { - 1 } ( 1 + \epsilon ) ( d ^ { \diamond } ( d ^ { \diamond } + 1 ) + 2 ) . } \end{array}
$$

Lemma 18 (Warm-up schedule trace guarantee). Let $V _ { w a r m }$ denote the allocation uniform coverage. Then, $i f n _ { w a r m } \geq C _ { r o u n d i n g } ,$

$$
\operatorname* { m a x } _ { \mu } \mathrm { T r a c e } \left( V _ { w a r m } ^ { \dagger } M _ { \mu } \right) \leq \frac { ( 1 + \epsilon ) ^ { 2 } d ^ { \diamond } } { n _ { w a r m } } \leq \frac { ( 1 + 3 \epsilon ) d ^ { \diamond } } { n _ { w a r m } } ,
$$

the last step since $\epsilon \leq 1$

Proof. By Proposition $^ { 2 , }$ ma $\mathrm { x } _ { \mu }$ Trace $\left( V ( w ^ { * } ) ^ { \dagger } M _ { \mu } \right) \leq \rho$ with $\rho : = ( 1 + \epsilon ) d ^ { \circ }$ . Apply the support-reduction of Lemma 17 and the rounding of Definition 7 to obtain the density and allocation $\boldsymbol { w } _ { 0 } ^ { * }$ and $v .$

Now note that the rounding is such that,

$$
v _ { i } \geq ( n _ { \mathrm { w a r m } } - s ) w _ { 0 , i } ^ { * } , \qquad \forall i \in [ s ] ,
$$

and therefore

$$
V _ { \mathrm { w a r m } } = \sum _ { \tau = 1 } ^ { n _ { \mathrm { w a r m } } } M _ { \mu _ { \tau } ^ { \mathrm { w a r m } } } = \sum _ { i \leq s } v _ { i } M _ { \mu _ { i } } \succeq ( n _ { \mathrm { w a r m } } - s ) V ( w _ { 0 } ^ { * } ) ,
$$

with $\{ \mu _ { i } \} _ { i \leq i }$ <sub>s</sub> the policies in the support of the allocation.

Hence, for any $\mu$ (elementary property in ${ \mathcal { G } } _ { : }$ see Lemma 4),

$$
\mathrm { T r a c e } \left( V _ { \mathrm { w a r m } } ^ { \dagger } M _ { \mu } \right) \leq \frac { \mathrm { T r a c e } \left( V ( w _ { 0 } ^ { * } ) ^ { \dagger } M _ { \mu } \right) } { n _ { \mathrm { w a r m } } - s } = \frac { \mathrm { T r a c e } \left( V ( w ^ { * } ) ^ { \dagger } M _ { \mu } \right) } { n _ { \mathrm { w a r m } } - s } \leq \frac { \rho } { n _ { \mathrm { w a r m } } - s } ,
$$

where the equality is by Lemma 17.

Replacing $\rho = ( 1 + \epsilon ) d ^ { \circ }$ and using the condition $n _ { \mathrm { w a r m } } \geq C _ { \mathrm { r o u n d i n g } }$

$$
n _ { \mathrm { w a r m } } \ge \frac { 1 + \epsilon } { \epsilon } s \Leftrightarrow \epsilon ( n _ { \mathrm { w a r m } } - s ) \ge s \Leftrightarrow ( 1 + \epsilon ) ( n _ { \mathrm { w a r m } } - s ) \ge n _ { \mathrm { w a r m } } ,
$$

we conclude:

$$
\frac { \rho } { n _ { \mathrm { w a r m } } - s } = \frac { ( 1 + \epsilon ) d ^ { \diamond } } { n _ { \mathrm { w a r m } } - s } \leq \frac { ( 1 + \epsilon ) ^ { 2 } d ^ { \diamond } } { n _ { \mathrm { w a r m } } } \leq \frac { ( 1 + 3 \epsilon ) d ^ { \diamond } } { n _ { \mathrm { w a r m } } } ,
$$

the last step using $( 1 + \epsilon ) ^ { 2 } = 1 + 2 \epsilon + \epsilon ^ { 2 } \leq 1 + 3 \epsilon \mathrm { ~ f o r ~ } \epsilon \leq 1 .$

Corollary 2 (Warm-up schedule norm guarantee). Let $V _ { w a r m }$ denote the allocation uniform coverage. Then, if $\ R _ { w a r m } \geq C _ { r o u n d i n g . }$

$$
\operatorname* { m a x } _ { \mu , b } \| g _ { \mu } ( b ) \| _ { V _ { w a r m } , * } ^ { 2 } \leq \frac { ( 1 + 3 \epsilon ) | \mathcal { B } | d ^ { \diamond } } { n _ { w a r m } } .
$$

Proof. Using a similar argument to the proof of Corollary 1, for any $\mu , b ;$

$$
\begin{array} { r l } & { \| g _ { \mu } ( b ) \| _ { V _ { \mathrm { w a r m } } , * } ^ { 2 } = g _ { \mu } ( b ) ^ { \top } V _ { \mathrm { w a r m } } ^ { \dagger } g _ { \mu } ( b ) } \\ & { \qquad \leq \displaystyle \sum _ { b ^ { \prime } \in B } g _ { \mu } ( b ^ { \prime } ) ^ { \top } V _ { \mathrm { w a r m } } ^ { \dagger } g _ { \mu } ( b ^ { \prime } ) } \\ & { \qquad = | \mathcal { B } | \operatorname { T r a c e } ( V _ { \mathrm { w a r m } } ^ { \dagger } M _ { \mu } ) } \\ & { \qquad \leq \frac { \left( 1 + 3 \epsilon \right) | \mathcal { B } | d ^ { \epsilon } } { n _ { \mathrm { w a r m } } } , } \end{array}
$$

where the last inequality follows from Lemma 18.

Finally, we can combine these results with the maximum likelihood estimation results to provide a certificate over sequence of MLE estimators.

This lemma depends on a new constant $C _ { \mathrm { c e r t i f i c a t e } }$ that we define as follows:

$$
\begin{array} { r l } & { C _ { \mathrm { c e r t i f i c a t e : } } = 8 \kappa ( 1 + 3 \epsilon ) \eta ^ { 2 } | \mathcal { B } | d ^ { \diamond } d \left( \log _ { + } \left( \frac { 4 e \eta S K } { \delta d } \right) + \log \left( 4 \kappa ( 1 + 3 \epsilon ) \eta ^ { 2 } | \mathcal { B } | d ^ { \diamond } d \right) \right) } \\ & { \qquad \leq \mathcal { O } \left( \kappa \eta ^ { 2 } | \mathcal { B } | d ^ { \diamond } d \log \left( \frac { \kappa \eta S | \mathcal { B } | d ^ { \diamond } K } { \delta } \right) \right) } \\ & { \qquad \leq \tilde { \mathcal { O } } \left( e ^ { 2 \eta S } | \mathcal { B } | d ^ { \diamond } d \log \left( \frac { e ^ { \eta S } } { \delta } \right) \right) , } \end{array}
$$

where $\kappa = 2 e ^ { 2 \eta S } / \eta ^ { 2 }$

Lemma 19 (Warm-up certificate). Let δ small enough and set $n _ { w a r m } \geq$ max $\{ C _ { r o u n d i n g } , C _ { c e r t i f i c a t e } \}$ Then, with probability at least $1 - \delta$ , the warm-up ensures $d _ { * } ( \omega , \hat { \omega } _ { k } ^ { o n } ) \leq 1 / \eta f o r a n y 0 \leq k \leq K$

Proof. For any $\mu , k ,$ , we can decompose using the generalized Cauchy-Schwarz (Lemma 4):

$$
d _ { \mu } ( \omega , \hat { \omega } _ { k } ^ { \mathrm { o n } } ) \leq \| \omega - \hat { \omega } _ { k } ^ { \mathrm { o n } } \| _ { V _ { \mathrm { w a r m } } } \cdot \operatorname* { m a x } _ { b , d } \| \psi _ { \mu } ( b ) - \psi _ { \mu } ( d ) \| _ { V _ { \mathrm { w a r m } } , * } ,
$$

where $V _ { \mathrm { w a r m } }$ is the uniform allocation coverage (Definition 7).

For the first term, we combine the design curvature (Lemma 14) with the confidence sets (Theorem 5).

Using $\kappa = 2 e ^ { 2 \eta S } / \eta ^ { 2 }$

$$
\begin{array} { r l } & { \| \omega - \hat { \omega } _ { k } ^ { \mathrm { o n } } \| _ { V _ { \mathrm { w a r m } } } ^ { 2 } \leq \kappa \| \omega - \hat { \omega } _ { k } ^ { \mathrm { o n } } \| _ { \Gamma _ { n _ { \mathrm { w a r m } } } ^ { 0 } ( \hat { \omega } _ { k } ^ { \mathrm { o n } } ) } ^ { 2 } } \\ & { \qquad \leq \kappa \| \omega - \hat { \omega } _ { k } ^ { \mathrm { o n } } \| _ { \Gamma _ { n _ { \mathrm { w a r m } } + k } ^ { 0 } ( \hat { \omega } _ { k } ^ { \mathrm { o n } } ) } ^ { 2 } } \\ & { \qquad \leq \kappa \left( \log \delta ^ { - 1 } + d \log _ { + } \left( 4 e d ^ { - 1 } \eta S ( n _ { \mathrm { w a r m } } + K - 1 ) \right) \right) } \\ & { \qquad : = L ( \delta ) , } \end{array}
$$

where we define log $_ { + } ( x ) : = \log { ( \operatorname* { m a x } { \{ e , x \} } ) }$

The analysis of the second term comes from the optimal design and the warm-up schedule. Specifically, using $n _ { \mathrm { w a r m } } \geq C _ { \mathrm { r o u n d i n g } }$ we apply Lemma $2$ and get:

$$
\operatorname* { m a x } _ { \mu , b } \lVert g _ { \mu } ( b ) \rVert _ { V _ { \mathrm { w a r m } } , * } \leq \sqrt { ( 1 + 3 \epsilon ) | \mathcal { B } | d ^ { \circ } / n _ { \mathrm { w a r m } } } .
$$

Since $\psi _ { \mu } ( b ) - \psi _ { \mu } ( d ) = g _ { \mu } ( b ) - g _ { \mu } ( d )$ , we get from the triangle inequality that for all $\mu \colon$

$$
\operatorname* { m a x } _ { b , d } \| \psi _ { \mu } ( b ) - \psi _ { \mu } ( d ) \| _ { V _ { \mathrm { v a r m } } , * } = \operatorname* { m a x } _ { b , d } \| g _ { \mu } ( b ) - g _ { \mu } ( d ) \| _ { V _ { \mathrm { v a r m } } , * } \leq 2 \cdot \operatorname* { m a x } _ { b } \| g _ { \mu } ( b ) \| _ { V _ { \mathrm { v a r m } } , * } .
$$

Hence, max $\begin{array} { r } { _ { b , d } \| \psi _ { \mu } ( b ) - \psi _ { \mu } ( d ) \| _ { V _ { \mathrm { w a r m } } , * } \leq 2 \cdot \sqrt { \frac { ( 1 + 3 \epsilon ) | \mathcal { B } | d ^ { \diamond } } { n _ { \mathrm { w a r m } } } } . } \end{array}$

Combining the two terms, we conclude for every $k \in [ 0 , K ]$ 」，

$$
d _ { * } ( \omega , \hat { \omega } _ { k } ^ { \mathrm { o n } } ) \leq \sqrt { L ( \delta ) } \cdot 2 \sqrt { \frac { ( 1 + 3 \epsilon ) | \mathcal { B } | d ^ { \circ } } { n _ { \mathrm { w a r m } } } } = 2 \sqrt { \frac { ( 1 + 3 \epsilon ) | \mathcal { B } | d ^ { \circ } L ( \delta ) } { n _ { \mathrm { w a r m } } } } .
$$

Hence with high-probability (the event holds for all k without needing a union bound),

$$
d _ { * } ( \omega , \hat { \omega } _ { k } ^ { \mathrm { o n } } ) \leq 1 / \eta , \qquad \mathrm { i f } \ n _ { \mathrm { w a r m } } \geq 4 ( 1 + 3 \epsilon ) \eta ^ { 2 } L ( \delta ) | \mathcal { B } | d ^ { \diamond } .
$$

Now let us expand and simplify $L ( \delta )$ as follows (using $n _ { \mathrm { w a r m } } , K \geq 1 )$

$$
\begin{array} { r l } & { L ( \delta ) = \kappa \left( \log \left( \displaystyle \frac { 1 } { \delta } \right) + d \log _ { + } \left( \displaystyle \frac { 4 e \eta S ( n _ { \mathrm { w a r m } } + K - 1 ) } { d } \right) \right) } \\ & { \quad \quad \le \kappa d \left( \log _ { + } \left( \displaystyle \frac { 4 e \eta S K } { \delta d } \right) + \log \left( n _ { \mathrm { w a r m } } \right) \right) . } \end{array}
$$

Therefore, a suficient condition is $n _ { \mathrm { w a r m } } \geq a ( b + \log n _ { \mathrm { w a r m } } )$ with:

$$
a = 4 \kappa ( 1 + 3 \epsilon ) \eta ^ { 2 } | \mathcal { B } | d ^ { \diamond } d \qquad \mathrm { a n d } \qquad b = \log _ { + } \left( \frac { 4 e \eta S K } { \delta d } \right) .
$$

Applying Lemma 25 we obtain the following suficient condition:

$$
\begin{array} { l } { { n _ { \mathrm { w a r m } } \geq 2 a ( b + \log a ) } } \\ { { \qquad = 8 \kappa ( 1 + 3 \epsilon ) \eta ^ { 2 } | \mathcal { B } | d ^ { \diamond } d \left( \log _ { + } \left( \frac { 4 e \eta S K } { \delta d } \right) + \log \left( 4 \kappa ( 1 + 3 \epsilon ) \eta ^ { 2 } | \mathcal { B } | d ^ { \diamond } d \right) \right) } } \end{array}
$$

$$
\mathrm { i f ~ } \delta < 1 6 \kappa ( 1 + 3 \epsilon ) \eta ^ { 3 } S | B | d ^ { \circ } K .
$$

We can therefore conclude from the condition $n _ { \mathrm { w a r m } } \geq C _ { \mathrm { c e r t i f i c a t e } } .$

## F Analysis of the main algorithm

Following the proof roadmap in the main text, we analyze the algorithm by first applying the Stackelberg reward transfer lemma (Lemma 20), and then combine the following three ingredients.

i) A confidence sequence for finite-sample estimation We can bound the coeficient distance terms by relating to high probability confidence sets of Generalized Linear Models (GLMs).

Specifically, following (Lee et al., 2024; Schlaginhaufen et al., 2026), we can derive a bound of the following form (Theorem 5):

$$
\mathbb { P } \left( \left\{ \| \omega - \hat { \omega } _ { \tau } \| _ { \Gamma _ { \tau } ^ { \lambda } ( \hat { \omega } _ { \tau } ) } \leq \beta _ { \tau } ( \delta ) , \ \forall \tau \in [ N ] \right\} \right) \geq 1 - \delta ,
$$

for some $\beta _ { \tau } ^ { 2 } ( \delta ) \le \mathcal { O } \left( \lambda S ^ { 2 } + \log \delta ^ { - 1 } + d \log ( \operatorname* { m a x } \left\{ e , \eta \tau S d ^ { - 1 } \right\} \right)$

This allows, with one good event, to control the concentration of the MLE estimators in both the warm-up and online phases.

ii) Minimizing uncertainty in the Fisher information geometry and the elliptical potential lemma We bound Fisher uncertainty by a greedy selection rule maximizing Fisher information gain. With a self-concordance reparameterization bound (see point iii) below), we apply a variant of the standard elliptical potential lemma (Abbasi-Yadkori et al., 2011) and show how it vanishes at a rate $\approx \sqrt { \log ( K ) / K }$

iii) Self-concordance The instantaneous log-likelihood as a function of the logits is self-concordant-like in the span semi-norm. This allows to control the reparameterization of Fisher information matrices. Precisely, an optimal schedule for the leader policies depends on knowing the true coeficient ω. As this is unknown, we instead replace it in the optimization objective by $\tilde { \omega } = 0$ in the warm-up phase, and by the running estimate $\tilde { \omega } = \hat { \omega } _ { k } ^ { \mathrm { o n } }$ during the online phase. More information on the self-concordance properties in Appendix D.

Crucially, this last tool introduces exponential dependency in the induced distance between the proxy $\tilde { \omega }$ and $\omega .$ . This is the reason for the warm-up phase, which removes this dependency in the final bound by using a burn-in budget. As mentioned, in this ofline part we use an arbitrary proxy $\tilde { \omega } = 0$ and rely on global self-concordance. Hence, our ofline warm-up would always produce an exponential sample complexity guarantee. We solve this problem by using a subsequent online phase in the improved self-concordance regime induced by the warm-up certificate (Lemma 19).

## F.1 Reward transfer lemma

Lemma 20 (Stackelberg reward transfer). Let $\mu ^ { * }$ and $\hat { \mu }$ be optimal leaders under $\omega$ and $\hat { \omega } _ { i }$ respectively. Then,

$$
\mathrm { S u b O p t } _ { \omega } ( \hat { \mu } ) \leq 2 R _ { m a x } \cdot \operatorname* { m a x } _ { \tilde { \mu } \in \{ \mu ^ { * } , \hat { \mu } \} } \chi \left( \eta \cdot d _ { \tilde { \mu } } ( \omega , \hat { \omega } ) \right) \cdot \| \omega - \hat { \omega } \| _ { \mathcal { T } _ { \tilde { \mu } } ( \hat { \omega } ) } ,
$$

with $\begin{array} { r } { \chi ( x ) : = \frac { 2 ( e ^ { x / 2 } - 1 ) } { x } } \end{array}$

Proof.

$$
\begin{array} { r l } & { \mathrm { S u b O p t } _ { \omega } ( \hat { \mu } ) = J _ { \omega } ^ { \mathrm { l e a d e r } } ( \mu ^ { * } ) - J _ { \omega } ^ { \mathrm { l e a d e r } } ( \hat { \mu } ) } \\ & { \qquad \leq \Big [ J _ { \omega } ^ { \mathrm { l e a d e r } } ( \mu ^ { * } ) - J _ { \hat { \omega } } ^ { \mathrm { l e a d e r } } ( \mu ^ { * } ) \Big ] + \Big [ J _ { \hat { \omega } } ^ { \mathrm { l e a d e r } } ( \mu ^ { * } ) - J _ { \hat { \omega } } ^ { \mathrm { l e a d e r } } ( \hat { \mu } ) \Big ] } \\ & { \qquad + \Big [ J _ { \hat { \omega } } ^ { \mathrm { l e a d e r } } ( \hat { \mu } ) - J _ { \omega } ^ { \mathrm { l e a d e r } } ( \hat { \mu } ) \Big ] } \\ & { \qquad \leq \mathrm { S u b O p t } _ { \hat { \omega } } ( \hat { \mu } ) + 2 \cdot \underset { \tilde { \mu } \in \{ \hat { \mu } , \mu ^ { * } \} } { \operatorname* { m a x } } \Big \vert J _ { \omega } ^ { \mathrm { l e a d e r } } ( \tilde { \mu } ) - J _ { \hat { \omega } } ^ { \mathrm { l e a d e r } } ( \tilde { \mu } ) \Big \vert } \\ & { \qquad \leq 2 R _ { \operatorname* { m a x } } \cdot \underset { \tilde { \mu } \in \{ \hat { \mu } , \mu ^ { * } \} } { \operatorname* { m a x } } \Big \Vert \nu _ { \tilde { \mu } } ^ { * } ( \omega ) - \nu _ { \tilde { \mu } } ^ { * } ( \hat { \omega } ) \Big \Vert _ { 1 } , } \end{array}
$$

where the last inequality uses the optimality of $\hat { \mu }$ under $\hat { \omega }$

Then applying Lemma 12 gives the final result.

Remark 4. The upper bound of Lemma 20 vanishes exactly when $\hat { r } _ { 2 }$ is feasible at both evaluated leaders, i.e.

$$
\operatorname* { m a x } _ { \widetilde { \mu } \in \{ \mu ^ { * } , \widehat { \mu } \} } \| \omega - \widehat { \omega } \| _ { { \mathcal L } _ { \widetilde { \mu } } ( \triangle ) } = 0 \quad \Longleftrightarrow \quad \widehat { r } _ { 2 } \in { \mathcal R } _ { 2 } ^ { \mathrm { f e a s } } ( \mu ^ { * } ) \cap { \mathcal R } _ { 2 } ^ { \mathrm { f e a s } } ( \widehat { \mu } ) ,
$$

for $\mathcal { R } _ { 2 } ^ { \mathrm { f e a s } } ( \mu ) = \Big \{ \tilde { r } _ { 2 } : \nu _ { \mu } ^ { * } ( \tilde { r } _ { 2 } ) = \nu _ { \mu } ^ { * } ( r _ { 2 } ) \Big \} .$

Using our specific choice of leader policies in Algorithm 1, we are able to bound the two terms of the upper bound separately. This leads to our main result.

## F.2 Finite-sample Stackelberg sub-optimality guarantee

Before deriving the proof, let us quickly note about the approximation given in the main paper for the burn-in phase.

Placing us in the same regime of Lemma 19, we will require $n _ { \mathrm { w a r m } } \geq \operatorname* { m a x } \{ C _ { \mathrm { r o u n d i n g } } , C _ { \mathrm { c e r t i f i c a t e } } \}$ Two things are in order:

1. Typically, $C _ { \mathrm { r o u n d i n g } } \ll C _ { \mathrm { c e r t i f i c a t 6 } }$ <sub>e</sub>.

2. Since for all features we have $d ^ { \circ } \leq d ,$ , we can further bound

$$
C _ { \mathrm { c e r t i f i c a t e } } \leq \tilde { \mathcal { O } } \left( e ^ { 2 \eta S } | \mathcal { B } | d ^ { \circ } d \log \left( \frac { e ^ { \eta S } } { \delta } \right) \right) \leq \tilde { \mathcal { O } } \left( \eta S e ^ { 2 \eta S } | \mathcal { B } | d ^ { 2 } \log \frac { 1 } { \delta } \right) .
$$

This motivates our approximation $n _ { \mathrm { w a r m } } \approx \tilde { \mathcal { O } } \left( \eta S e ^ { 2 \eta S } | \mathcal { B } | d ^ { 2 } \log \delta ^ { - 1 } \right)$ for the informal statement, only using terms introduced in the main text.

Theorem 4. Set $n _ { w a r m } = \mathrm { m a x } \{ C _ { r o u n d i n g } , C _ { c e r t i f i c a t e } \} , \lambda = 0$ and let $\hat { \omega }$ be the result of Algorithm 1 after $K \geq 1$ online iterations. Then, with probability at least $1 - \delta _ { \cdot }$ , for any leader $\hat { \mu }$ such that $\mathrm { S u b O p t } _ { \hat { \omega } } ( \hat { \mu } ) = 0$

$$
\mathrm { S u b O p t } _ { \omega } ( \hat { \mu } ) \leq \tilde { \mathcal { O } } \left( \frac { R _ { m a x } \sqrt { d ^ { \circ } \left( d \eta S + \log { \delta ^ { - 1 } + d } \right) } } { \sqrt { K } } \right) ,
$$

where $\tilde { \mathcal { O } }$ hides polylog terms in log $\delta ^ { - 1 }$ , K and environment parameters.

Proof. In the following, we assume the certificate over the entire sequence of maximum likelihood estimators, which occurs with probability at least $1 - \delta ,$ , assuming a suficient burn-in phase budget (Lemma 19).

Therefore, combining the reward transfer lemma (Lemma 20) with the warm-up result (Lemma 19), the bound reduces to the coeficient approximation in the Fisher information norm:

$$
\mathrm { S u b O p t } _ { \omega } ( \hat { \mu } ) \leq 2 \chi \left( 1 \right) R _ { \operatorname* { m a x } } \cdot \operatorname* { m a x } _ { \hat { \mu } \in \{ \mu ^ { * } , \hat { \mu } \} } \| \omega - \hat { \omega } \| _ { { \mathcal L } _ { \hat { \mu } } ( \hat { \omega } ) } ,
$$

where $\chi ( 1 ) = 2 ( \sqrt { e } - 1 ) \approx 1 . 2 9 7 \mathrm { a n d } \hat { \omega } = \hat { \omega } _ { K } ^ { \mathrm { o n } }$

Now, recall that for any k and $\lambda ,$ the curvature of the log-likelihood corresponds to the cumulative information collected up to online iteration $k \colon$

$$
- \nabla ^ { 2 } \mathcal { L } _ { k } ( \tilde { \omega } ) = \eta ^ { 2 } \sum _ { \tau = 1 } ^ { N _ { k } } \mathbb { E } _ { b \sim \mathbb { P } ( \cdot | \tilde { \omega } , \mu _ { \tau } ) } \left[ \left( \psi _ { \mu _ { \tau } } ( b ) - \overline { { \psi } } _ { \mu _ { \tau } } ( \tilde { \omega } ) \right) ^ { \otimes 2 } \right] = \sum _ { \tau = 1 } ^ { N _ { k } } \mathcal { T } _ { \mu _ { \tau } } ( \tilde { \omega } ) ,
$$

and in its regularized form $G _ { k } ^ { \lambda } ( \tilde { \omega } ) : = \lambda I - \nabla ^ { 2 } \mathcal { L } _ { k } ( \tilde { \omega } )$

By Lemma 18, the warm-up uniform allocation must be admissible. Hence, for all $k \in [ K ]$ we have $G _ { k } ^ { 0 } ( \tilde { \omega } ) \in \mathbb { S } _ { + + } ( \mathcal { G } )$ , i.e. the warm-up phase guarantees that the unregularized total information matrix covers the space $\mathcal { G }$ for all $k .$ . Therefore, setting $\lambda = 0$ , we simplify notations by denoting $G _ { k } : = G _ { k } ^ { 0 }$

For any $\tilde { \mu } \in \{ \mu ^ { * } , \hat { \mu } \}$ , under the certificate,

$$
\begin{array} { r l r } { \| \omega - \hat { \omega } \| _ { \mathcal { T } _ { \widetilde { \mu } } ( \hat { \omega } ) } ^ { 2 } = ( \omega - \hat { \omega } ) ^ { \top } \mathbb { Z } _ { \widetilde { \mu } } ( \hat { \omega } ) ( \omega - \hat { \omega } ) } \\ { \leq \| \omega - \hat { \omega } \| _ { G _ { K } ( \hat { \omega } ) } ^ { 2 } \cdot \operatorname { T r a c e } \left( G _ { K } ^ { \dag } ( \hat { \omega } ) \mathbb { Z } _ { \widetilde { \mu } } ( \hat { \omega } ) \right) } & { \qquad } & { \mathrm { ( L e m m a ~ 2 1 ~ i n ~ \mathcal { G } ) } } \\ { \leq 3 \| \omega - \hat { \omega } \| _ { \Gamma _ { K } ( \hat { \omega } ) } ^ { 2 } \cdot \operatorname { T r a c e } \left( G _ { K } ^ { \dag } ( \hat { \omega } ) \mathbb { Z } _ { \widetilde { \mu } } ( \hat { \omega } ) \right) } & { \qquad } & { \mathrm { ( L e m m a ~ 1 0 , w a r m - u p ) } } \\ { \leq 3 \beta _ { N _ { K } } ^ { 2 } ( \delta ) \cdot \operatorname { T r a c e } \left( G _ { K } ^ { \dag } ( \hat { \omega } ) \mathbb { Z } _ { \widetilde { \mu } } ( \hat { \omega } ) \right) . } & { \qquad } & { \mathrm { ( T h e o r e m ~ 5 ) } } \end{array}
$$

This last term is controlled by our greedy design.

Indeed, first notice that:

$$
\begin{array} { r l r } { \mathrm { T r a c e } \left( G _ { K } ^ { \dagger } ( \hat { \omega } ) Z _ { \hat { \mu } } ( \hat { \omega } ) \right) } & \\ & { \leq \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathrm { T r a c e } \left( G _ { k - 1 } ^ { \dagger } ( \hat { \omega } ) T _ { \hat { \mu } } ( \hat { \omega } ) \right) } & { \quad \mathrm { ( n o n - i n e r e s s i n g ~ s e q u e n c e ) } } \\ & { \leq \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } c ^ { 2 n H _ { i } ( \hat { \omega } ; \hat { \omega } _ { k - 1 } ) } \mathrm { T r a c e } \left( G _ { k - 1 } ^ { \dagger } ( \hat { \omega } _ { k - 1 } ^ { \mathrm { a m } } ) T _ { \hat { \mu } } ( \hat { \omega } _ { k - 1 } ^ { \mathrm { a m } } ) \right) } & { \quad \mathrm { ( s e l f - c o n c o r d a n c e ) } } \\ & { \leq \displaystyle \frac { e ^ { 2 n \phi _ { i } ( \hat { \omega } ; \omega ) } } { K } \sum _ { k = 1 } ^ { K } e ^ { 2 n H _ { i } ( \omega ; \hat { \omega } _ { k - 1 } ^ { \mathrm { a m } } ) } \mathrm { T r a c e } \left( G _ { k - 1 } ^ { \dagger } ( \hat { \omega } _ { k - 1 } ^ { \mathrm { a m } } ) Z _ { \hat { \mu } } ( \hat { \omega } _ { k - 1 } ^ { \mathrm { a m } } ) \right) } & { \quad \mathrm { ( t r i a n g l e ~ i n e q u a l i t y ) } } \\ & { \leq \displaystyle \frac { e ^ { 4 } } { K } \sum _ { k = 1 } ^ { K } \mathrm { T r a c e } \left( G _ { k - 1 } ^ { \dagger } ( \hat { \omega } _ { k - 1 } ^ { \mathrm { a m } } ) Z _ { \hat { \mu } } ( \hat { \omega } _ { k - 1 } ^ { \mathrm { a m } } ) \right) . } & { \quad \mathrm { ( c e r t i f i c a t c ) } } \end{array}
$$

Then, from our selection rule we get:

$$
\begin{array} { r l } { \mathrm { T r a c e } ( G _ { K } ^ { \dagger } ( \hat { \omega } ) \mathcal { T } _ { \bar { \mu } } ( \hat { \omega } ) ) \leq \displaystyle \frac { e ^ { 4 } } { K } \sum _ { k = 1 } ^ { K } \mathrm { T r a c e } ( G _ { k - 1 } ^ { \dagger } ( \hat { \omega } _ { k - 1 } ^ { \mathrm { o n } } ) \mathcal { T } _ { \mu _ { k } ^ { \mathrm { o n } } } ( \hat { \omega } _ { k - 1 } ^ { \mathrm { o n } } ) ) } & { \qquad \mathrm { ( s e l e c t i o n ~ r u l e ) } } \\ { \displaystyle \leq \displaystyle \frac { e ^ { 6 } } { K } \sum _ { k = 1 } ^ { K } \mathrm { T r a c e } ( G _ { k - 1 } ^ { \dagger } ( \omega ) \mathcal { T } _ { \mu _ { k } ^ { \mathrm { o n } } } ( \omega ) ) } & { \qquad \mathrm { ( c e r t i f i c a t e ) } } \end{array}
$$

We will then apply the elliptical potential lemma (see classical isotropic version as Lemma 24) in the information geometry (Corollary 3), leveraging the warm-up phase.

Specifically, let $U _ { 0 } ( \cdot ) : = G ^ { 0 } \left( \cdot | \mathcal { D } _ { \mathrm { w a r m } } \right)$ be the warm-up cumulative Fisher information. From the warm-up guarantees, the information uncertainty is bounded as follows:

$$
\begin{array} { r l r } { \underset { \mu \in \Pi _ { \mu } } { \operatorname* { s u p } } \operatorname { T r a c e } \left( U _ { 0 } ( \omega ) ^ { \dagger } \mathcal { T } _ { \mu } ( \omega ) \right) \leq \frac { \kappa } { 2 } \cdot \underset { \mu \in \Pi _ { \mu } } { \operatorname* { s u p } } \operatorname { T r a c e } \left( V _ { \mathrm { w a r m } } ^ { \dagger } \mathcal { T } _ { \mu } ( \omega ) \right) } & { \qquad \mathrm { ( L e m m a ~ 1 4 ) } } \\ { \leq \frac { \kappa \eta ^ { 2 } | \mathcal { B } | } { 2 } \cdot \underset { \mu \in \Pi _ { \mu } } { \operatorname* { s u p } } \operatorname { T r a c e } \left( V _ { \mathrm { w a r m } } ^ { \dagger } M _ { \mu } \right) } & { \qquad \mathrm { ( s e c o n d ~ m o m e n t ) } } \\ { \leq \frac { \kappa \eta ^ { 2 } | \mathcal { B } | } { 2 } \cdot \frac { ( 1 + 3 \epsilon ) d ^ { \diamond } } { n _ { \mathrm { w a r m } } } . } & { \qquad \mathrm { ( L e m m a ~ 1 8 ) } } \end{array}
$$

We define this last term as $\begin{array} { r } { q _ { \mathrm { w a r m } } : = \frac { ( 1 + 3 \epsilon ) \kappa d ^ { \circ } \eta ^ { 2 } | \mathcal { B } | } { 2 n _ { \mathrm { w a r m } } } \leq \frac { 1 } { 1 6 d } \leq 1 } \end{array}$ , where the simplification is from the condition $n _ { \mathrm { w a r m } } \geq C _ { \mathrm { c e r t i f i c a t e } } .$

We can then apply the elliptical potential lemma in G (Corollary 3) with $U _ { 0 } : = U _ { 0 } ( \omega )$ and $A _ { k } = \mathcal { T } _ { \mu _ { k } ^ { \mathrm { o n } } } ( \omega )$ and obtain

$$
\mathrm { T r a c e } \left( G _ { K } ^ { \dag } ( \hat { \omega } ) \mathcal { T } _ { \tilde { \mu } } ( \hat { \omega } ) \right) \leq \frac { 2 d ^ { \diamond } e ^ { 6 } \log \left( 1 + \frac { q _ { \mathrm { w a r m } } K } { d ^ { \diamond } } \right) } { K } \leq \frac { 2 d ^ { \diamond } e ^ { 6 } \log \left( 1 + \frac { K } { 1 6 d ^ { \diamond } d } \right) } { K } .
$$

To simplify notations, set $\begin{array} { r } { C : = \sqrt { d ^ { \circ } \log \left( 1 + \frac { K } { 1 6 d ^ { \circ } d } \right) } } \end{array}$ and use $\log _ { + } ( x ) : = \log \left( \operatorname* { m a x } \{ e , x \} \right)$ Then, putting everything together, we get with probability at least $1 - \delta$

$$
\begin{array} { r l } { \mathrm { S i n S p t } _ { \omega } ( \hat { \rho } _ { i } ) \leq 2 \chi ( 1 ) R _ { \mathrm { a m s } } \sqrt { \lvert \partial _ { \Omega \times \delta } ^ { \Omega } ( \delta ) \cdot \mathrm { T a s e } ( G _ { \kappa } ^ { \mathrm { T } } ( \delta ) \mathcal { T } _ { \mu } ( \hat { \omega } ) ) } } & { } \\ { \leq \frac { 9 6 \chi ^ { 2 } C R _ { \mathrm { m s } } } { \sqrt { 2 K } } \sqrt { \lvert \mathrm { l o g } \delta ^ { - 1 } + d \mathrm { l o g } _ { 1 } ( \frac { 4 \epsilon \eta \mathcal { N } _ { \delta } \xi } { d } ) } } & { } \\ { \leq \frac { 9 6 \chi ^ { 2 } C R _ { \mathrm { m s } } } { \sqrt { 2 K } } \sqrt { \lvert \mathrm { l o g } \delta ^ { - 1 } + d \mathrm { l o g } ( \frac { 4 \epsilon \eta \mathcal { N } _ { \mathrm { m s } \omega } S } { d } ) + d \mathrm { l o g } _ { + } ( \frac { 4 \epsilon \eta K \delta } { d } ) } } & { } \\ { \leq \bar { \mathcal { O } } ( \frac { C R _ { \mathrm { m s } } } { \sqrt { K } } \sqrt { \lvert \mathrm { l o g } \delta ^ { - 1 } + d \mathrm { l o g } ( c ^ { 2 \mu \delta } \log ( \frac { \epsilon \mathcal { N } \delta } { \delta } ) ) + d ) } } & { } \\ { \leq \bar { \mathcal { O } } ( \frac { R _ { \mathrm { m s } } \sqrt { \bar { \phi } ^ { \mathrm { o p } } ( d \eta \mathcal { N } + \mathrm { l o g } \delta ^ { - 1 } + d \mathrm { l o g } \log ( \frac { \epsilon \mathcal { N } \delta } { \delta } ) ) } } { \sqrt { K } } ) } & { } \\  \leq \bar { \mathcal { O } } ( \frac  R _ { \mathrm { m a x } } \sqrt  \bar { \phi } ^ { \mathrm { o p } } ( d \eta \mathcal { N } + \mathrm { l o g } \delta ^ { - 1 } + d \mathrm { l o g } \log ( \frac  \epsilon \ \end{array}
$$

## G Useful results

Theorem 5 (Corollary of Theorem 3.1 in Lee et al. (2024)). Fix $\lambda \geq 0$ and $\delta \in ( 0 , 1 )$ . Then, with probability at least $1 - \delta$ ，

$$
\Vert \omega - \hat { \omega } _ { \tau } \Vert _ { \Gamma _ { \tau } ^ { \lambda } ( \hat { \omega } _ { \tau } ) } ^ { 2 } \leq \beta _ { \tau } ^ { 2 } ( \delta ) : = 2 \lambda S ^ { 2 } + \log \frac { 1 } { \delta } + d \log \left( \operatorname* { m a x } \left\{ e , \frac { 4 ( \tau - 1 ) e \eta S } { d } \right\} \right) ,
$$

for all $\tau \in [ N ]$ , where we define $\begin{array} { r } { \Gamma _ { \tau } ^ { \lambda } ( \tilde { \omega } ) : = \int _ { 0 } ^ { 1 } ( 1 - s ) G ^ { \lambda } \left( \tilde { \omega } + s ( \omega - \tilde { \omega } ) | \mathcal { D } _ { 1 : \tau } \right) \mathrm { d } s } \end{array}$

Proof. Combining a first-order Taylor expansion and the optimality of $\hat { \omega } _ { \tau }$ with respect to $\mathcal { L } \left( \cdot | \mathcal { D } _ { 1 : \tau } \right)$

$$
\begin{array} { r l } & { \mathcal { L } \left( \omega \middle | \mathcal { D } _ { 1 : \tau } \right) - \mathcal { L } \left( \hat { \omega } _ { \tau } \middle | \mathcal { D } _ { 1 : \tau } \right) = \left. \nabla \mathcal { L } \left( \hat { \omega } _ { \tau } \middle | \mathcal { D } _ { 1 : \tau } \right) , \omega - \hat { \omega } _ { \tau } \right. } \\ & { \qquad + \left( \omega - \hat { \omega } _ { \tau } \right) ^ { \top } \left( \displaystyle \int _ { 0 } ^ { 1 } ( 1 - s ) \nabla ^ { 2 } \mathcal { L } \left( \hat { \omega } _ { \tau } + s ( \omega - \hat { \omega } _ { \tau } ) \middle | \mathcal { D } _ { 1 : \tau } \right) \mathrm { d } s \right) \left( \omega - \hat { \omega } _ { \tau } \right) } \\ & { \qquad \leq \displaystyle \frac { \lambda } { 2 } \| \omega - \hat { \omega } _ { \tau } \| _ { 2 } ^ { 2 } - \| \omega - \hat { \omega } _ { \tau } \| _ { \Gamma _ { \tau } ^ { \lambda } ( \hat { \omega } _ { \tau } ) } ^ { 2 } } \\ & { \qquad \leq 2 \lambda S ^ { 2 } - \| \omega - \hat { \omega } _ { \tau } \| _ { \Gamma _ { \tau } ^ { \lambda } ( \hat { \omega } _ { \tau } ) } ^ { 2 } . } \end{array}
$$

Now, note that $\mathcal { L } \left( \cdot | \mathcal { D } _ { 1 : \tau } \right)$ is $2 \eta ( \tau - 1 )$ -Lipschitz by Rademacher’s theorem.

We can then apply Theorem 3.1 in Lee et al. (2024) and conclude.

Lemma 21 (Trace splitting). Let $x \in \mathbb { R } ^ { d } , A \succeq 0$ , and $B \succ 0$ . Then

$$
\left\| x \right\| _ { A } ^ { 2 } \leq \operatorname { T r a c e } \left( B ^ { - 1 } A \right) \cdot \left\| x \right\| _ { B } ^ { 2 } .
$$

Proof. Set $y : = B ^ { 1 / 2 } x .$ , so that $\| x \| _ { B } ^ { 2 } = \| y \| _ { 2 } ^ { 2 }$ and

$$
\begin{array} { r } { \| x \| _ { A } ^ { 2 } = x ^ { \top } A x = y ^ { \top } \left( B ^ { - 1 / 2 } A B ^ { - 1 / 2 } \right) y \leq \lambda _ { \operatorname* { m a x } } \Big ( B ^ { - 1 / 2 } A B ^ { - 1 / 2 } \Big ) \| y \| _ { 2 } ^ { 2 } . } \end{array}
$$

Now since $B \succ 0$ , we note that:

$$
\lambda _ { \operatorname* { m a x } } \Bigl ( B ^ { - 1 / 2 } A B ^ { - 1 / 2 } \Bigr ) \leq \mathrm { T r a c e } \Bigl ( B ^ { - 1 / 2 } A B ^ { - 1 / 2 } \Bigr ) = \mathrm { T r a c e } \Bigl ( B ^ { - 1 } A \Bigr ) ,
$$

which gives the claim.

Lemma 22 (Divergence decomposition). Let $K \in \mathbb { Z } _ { > 0 }$ and let $Q ^ { ( 1 ) } , Q ^ { ( 2 ) }$ be two kernels from X to a common measurable space such that $Q ^ { ( 1 ) } ( \cdot | x ) \ll Q ^ { ( 2 ) } ( \cdot | x )$ for every $x \in \mathcal { X }$ Define a common kernel family $\{ \pi _ { k } \} _ { k \le K }$ inducing

$$
\mathbb { P } ^ { ( j ) } = \mathrm { L a w } \left( \tau _ { K } : = \left( x _ { 1 } , y _ { 1 } , . . . , x _ { K } , y _ { K } \right) \right) , \qquad y _ { k } \sim Q ^ { ( j ) } ( \cdot | x _ { k } ) , \qquad \forall j \in \{ 1 , 2 \} ,
$$

with $x _ { k } \sim \pi _ { k } ( \cdot \mid \tau _ { < k } )$ for the sub-trajectory $\tau _ { < k } : = ( x _ { 1 } , y _ { 1 } , \dots , x _ { k - 1 } , y _ { k - 1 } )$ Then,

$$
D _ { \mathrm { K L } } ( \mathbb { P } ^ { ( 1 ) } \| \mathbb { P } ^ { ( 2 ) } ) = \sum _ { k = 1 } ^ { K } \mathbb { E } ^ { ( 1 ) } [ D _ { \mathrm { K L } } ( Q ^ { ( 1 ) } ( \cdot | x _ { k } ) \| Q ^ { ( 2 ) } ( \cdot | x _ { k } ) ) ] .
$$

Proof. By construction, the trajectory laws factorize as:

$$
\mathbb { P } ^ { ( j ) } ( \mathrm { d } \tau _ { K } ) = \prod _ { k = 1 } ^ { K } \pi _ { k } ( \mathrm { d } x _ { k } \mid \tau _ { < k } ) Q ^ { ( j ) } ( \mathrm { d } y _ { k } \mid x _ { k } ) ,
$$

so that $\mathbb { P } ^ { ( 1 ) } \ll \mathbb { P } ^ { ( 2 ) }$

Taking the likelihood ratio, the common kernels cancel:

$$
\log \frac { \mathrm { d } \mathbb { P } ^ { ( 1 ) } } { \mathrm { d } \mathbb { P } ^ { ( 2 ) } } ( \tau _ { K } ) = \sum _ { k = 1 } ^ { K } \log \frac { \mathrm { d } Q ^ { ( 1 ) } ( \cdot \mid x _ { k } ) } { \mathrm { d } Q ^ { ( 2 ) } ( \cdot \mid x _ { k } ) } ( y _ { k } ) , \qquad \mathbb { P } ^ { ( 1 ) } \mathrm { - a . s . }
$$

Taking expectations on both sides and using the tower property and substitution law for conditional expectations give the final result. □

Lemma 23 (KL divergence of softmax policies). Let $v , v ^ { \prime } \in \mathbb { R } ^ { d }$ and $\pi = \operatorname { s o f t m a x } ( v ) , \pi ^ { \prime } =$ softmax(v<sup>′</sup>) the two corresponding softmax policies with inverse temperature η. Then,

$$
D _ { \mathrm { K L } } ( \pi | | \pi ^ { \prime } ) \leq \frac { \eta ^ { 2 } } { 8 } \| v - v ^ { \prime } \| _ { s p a n } ^ { 2 }
$$

where $\begin{array} { r } { \| x \| _ { s p a n } = \operatorname* { m a x } _ { a } x _ { a } - \operatorname* { m i n } _ { a } x _ { a } } \end{array}$

Proof. Define the partition terms $Z , Z ^ { \prime }$ such that for any $1 \leq i \leq d ,$

$$
\pi _ { i } = e ^ { \eta v _ { i } } / Z \qquad \mathrm { a n d } \qquad \pi _ { i } ^ { \prime } = e ^ { \eta v _ { i } ^ { \prime } } / Z ^ { \prime } .
$$

Using the notation $\Delta _ { i } = v _ { i } ^ { \prime } - v _ { i }$ , notice that:

$$
Z ^ { \prime } = \sum _ { i } e ^ { \eta v _ { i } ^ { \prime } } = \sum _ { i } e ^ { \eta v _ { i } } \cdot e ^ { \eta \Delta _ { i } } = Z \mathbb { E } _ { \boldsymbol { \pi } } \left[ e ^ { \eta \Delta _ { i } } \right] .
$$

Therefore,

$$
\begin{array} { r l } {  { D _ { \mathrm { { K L } } } ( \pi | | \pi ^ { \prime } ) = \sum _ { i } \pi _ { i } \log ( \pi _ { i } / \pi _ { i } ^ { \prime } ) } } \\ & { = \sum _ { i } \pi _ { i } \cdot [ \eta v _ { i } - \log Z - \eta v _ { i } ^ { \prime } + \log Z ^ { \prime } ] } \\ & { = \log Z ^ { \prime } - \log Z - \eta \sum _ { i } \pi _ { i } \cdot \Delta _ { i } } \\ & { = \log \Big ( \mathbb { E } _ { \pi } [ e ^ { \eta \Delta _ { i } } ] \Big ) - \eta \mathbb { E } _ { \pi } [ \Delta _ { i } ] } \\ & { = \log \Big ( \mathbb { E } _ { \pi } [ e ^ { \eta ( \Delta _ { i } - \mathbb { E } _ { \pi } [ \Delta _ { i } ] ) } ] \Big ) } \\ & { \le \frac { \eta ^ { 2 } \big \| \Delta \big \| _ { \operatorname * { p a m } } ^ { 2 } } { 8 } } \end{array}
$$

where the last line holds by Hoefding’s lemma.

Lemma 24 (Elliptical Potential Lemma). Fix $K \geq 1 , m \geq 1 , \rho > 0$ and some sequence $A _ { 1 } , \dots , A _ { K } \in \mathbb { S } _ { + } ^ { m }$ such that Trace $( A _ { k } ) \le L$ for all $k \in [ K ]$ . Define for all $k \in [ K ]$

$$
U _ { k } : = U _ { 0 } + \sum _ { \tau = 1 } ^ { k } A _ { \tau } , \qquad \mathrm { f o r ~ a ~ g i v e n } \quad U _ { 0 } \succeq \rho I _ { m } .
$$

If $\rho \geq L$ , then

$$
\sum _ { k = 1 } ^ { K } \operatorname { T r a c e } \left( U _ { k - 1 } ^ { - 1 } A _ { k } \right) \leq 2 \log { \frac { \operatorname* { d e t } U _ { K } } { \operatorname* { d e t } U _ { 0 } } } \leq 2 m \log \left( 1 + { \frac { K L } { m \rho } } \right) .
$$

Proof. Define $M _ { k } : = U _ { k - 1 } ^ { - 1 / 2 } A _ { k } U _ { k - 1 } ^ { - 1 / 2 } \succeq 0$ such that

$$
\lambda _ { \operatorname* { m a x } } ( M _ { k } ) \leq \operatorname { T r a c e } ( M _ { k } ) = \operatorname { T r a c e } \left( U _ { k - 1 } ^ { - 1 } A _ { k } \right) \leq { \frac { \operatorname { T r a c e } ( A _ { k } ) } { \lambda _ { \operatorname* { m i n } } ( U _ { k - 1 } ) } } \leq { \frac { L } { \rho } } \leq 1 .
$$

Therefore, using the relation log $( 1 + x ) \geq x / 2$ for every $x \in [ 0 , 1 ]$

$$
\log \left( { \frac { \operatorname* { d e t } U _ { k } } { \operatorname* { d e t } U _ { k - 1 } } } \right) = \log \operatorname* { d e t } ( I + M _ { k } ) = \sum _ { i = 1 } ^ { m } \log ( 1 + \lambda _ { i } ( M _ { k } ) ) \geq { \frac { 1 } { 2 } } \operatorname { T r a c e } ( M _ { k } ) .
$$

Hence, by a telescopic argument,

$$
\sum _ { k = 1 } ^ { K } { \mathrm { T r a c e } } \left( U _ { k - 1 } ^ { - 1 } A _ { k } \right) = \sum _ { k = 1 } ^ { K } { \mathrm { T r a c e } } ( M _ { k } ) \leq 2 \log \left( { \frac { \operatorname* { d e t } U _ { K } } { \operatorname* { d e t } U _ { 0 } } } \right) ,
$$

which gives the first inequality.

The second inequality follows from the AM-GM inequality on the eigenvalues of $\begin{array} { r l } { U _ { K } . \quad } & { { } \boxed { \begin{array} { r l r l } \end{array} } } \end{array}$

Corollary 3 (Whitened Elliptical Potential Lemma on $\mathcal { G } )$ . Fix $K \ge 1 , q > 0 , U _ { 0 } \in \mathbb { S } _ { + + } ( \mathcal { G } )$ and some sequence $A _ { 1 } , \dots , A _ { K } \in \mathbb { S } _ { + } ( { \mathcal { G } } )$ such that for all $k \in [ K ]$

$$
\operatorname { T r a c e } \left( U _ { 0 } ^ { \dagger } A _ { k } \right) \leq q \quad a n d \quad U _ { k } : = U _ { 0 } + \sum _ { \tau = 1 } ^ { k } A _ { \tau } .
$$

If $q \leq 1$ , then

$$
\sum _ { k = 1 } ^ { K } \operatorname { T r a c e } \left( U _ { k - 1 } ^ { \dagger } A _ { k } \right) \leq 2 \log { \frac { \operatorname* { d e t } _ { \mathcal { G } } U _ { K } } { \operatorname* { d e t } _ { \mathcal { G } } U _ { 0 } } } \leq 2 d ^ { \diamond } \log \left( 1 + { \frac { q K } { d ^ { \diamond } } } \right) .
$$

Proof. Let $Q \in \mathbb { R } ^ { d \times d ^ { \circ } }$ have orthonormal columns spanning ${ \mathcal { G } } _ { : }$ , and write the projections of $U _ { k }$ and $A _ { k }$ via $Q$ as:

$$
\bar { U } _ { k } : = Q ^ { \top } U _ { k } Q \in \mathbb { S } _ { + + } ^ { d ^ { \diamond } } , \qquad \bar { A } _ { k } : = Q ^ { \top } A _ { k } Q \in \mathbb { S } _ { + } ^ { d ^ { \diamond } } .
$$

Define the following matrices, whitened by the initial $U _ { 0 } { \mathrm { : } }$

$$
\tilde { U } _ { k } : = \bar { U } _ { 0 } ^ { - 1 / 2 } \bar { U } _ { k } \bar { U } _ { 0 } ^ { - 1 / 2 } , \qquad \tilde { A } _ { k } : = \bar { U } _ { 0 } ^ { - 1 / 2 } \bar { A } _ { k } \bar { U } _ { 0 } ^ { - 1 / 2 } ,
$$

such that ${ \tilde { U } } _ { 0 } = I _ { d ^ { \circ } }$ and Trace $( \tilde { A } _ { k } ) = \mathrm { T r a c e } ( U _ { 0 } ^ { \dagger } A _ { k } ) \leq q .$

We can then apply Lemma 24 with $m = d ^ { \circ } , \rho = 1$ , and $L = q$ and obtain, given $q \leq \rho \colon$

$$
\sum _ { k = 1 } ^ { K } \operatorname { T r a c e } \left( \tilde { U } _ { k - 1 } ^ { - 1 } \tilde { A } _ { k } \right) \leq 2 \log \frac { \operatorname* { d e t } \tilde { U } _ { K } } { \operatorname* { d e t } \tilde { U } _ { 0 } } \leq 2 d ^ { \diamond } \log \left( 1 + \frac { q K } { d ^ { \diamond } } \right) .
$$

We then conclude by the following equalities:

$$
\operatorname { T r a c e } ( \tilde { U } _ { k - 1 } ^ { - 1 } \tilde { A } _ { k } ) = \operatorname { T r a c e } ( U _ { k - 1 } ^ { \dagger } A _ { k } ) , \qquad \operatorname { d e t } \tilde { U } _ { K } = \frac { \operatorname * { d e t } _ { \mathcal { G } } U _ { K } } { \operatorname * { d e t } _ { \mathcal { G } } U _ { 0 } } .
$$

Lemma 25. Let a, $, b \in \mathbb { R } _ { + }$ such that $a e ^ { b } > e$ . Then we have $n \geq a ( b + \log n ) i f n \geq$ $2 a ( b + \log a )$

Proof. In general for every positive integer n, some algebraic manipulations give:

$$
\begin{array} { r l } & { n \geq a ( b + \log n ) \Leftrightarrow n \geq a b + a \log n } \\ & { \qquad \Leftrightarrow \frac { n } { a } - \log n \geq b } \\ & { \qquad \Leftrightarrow \log n - \frac { n } { a } \leq - b } \\ & { \qquad \Leftrightarrow n e ^ { - \frac { n } { a } } \leq e ^ { - b } } \\ & { \qquad \Leftrightarrow \left( - \frac { n } { a } \right) e ^ { - \frac { n } { a } } \geq - \frac { 1 } { a } e ^ { - b } } \end{array}
$$

which gives the suficient condition

$$
n \geq - a \cdot W _ { - 1 } \left( - { \frac { 1 } { a } } e ^ { - b } \right) = - 2 a ^ { \prime } W _ { - 1 } \left( - { \frac { 1 } { 2 a ^ { \prime } { \sqrt { b ^ { \prime } } } } } \right) ,
$$

where we define $a ^ { \prime } : = { \frac { a } { 2 } }$ and $b ^ { \prime } : = e ^ { 2 b }$ $W _ { - 1 }$ represents the lower branch of the Lambert W function.

Applying the simplification of Metelli et al. (2021, Lemma B.8) we get:

$$
- 2 a ^ { \prime } W _ { - 1 } \left( - \frac { 1 } { 2 a ^ { \prime } \sqrt { b ^ { \prime } } } \right) \leq 4 a ^ { \prime } \log ( 2 a ^ { \prime } \sqrt { b ^ { \prime } } ) = 2 a \log ( a e ^ { b } ) = 2 a ( \log a + b ) ,
$$

which works under the assumption $2 a ^ { \prime } { \sqrt { b ^ { \prime } } } = a e ^ { b } > e$

Hence, given $2 a ^ { \prime } { \sqrt { b ^ { \prime } } } = a e ^ { b } > e .$ , we can conclude:

$$
n \geq 2 a ( b + \log a ) \implies n \geq a ( b + \log n ) .
$$

## H Additional experiments

We provide below additional experiments and ablations, complementing Section 5 in the main text. Unless stated otherwise, the plots report the mean and two-sided 95% confidence interval across independent seeds.

We study the following algorithms.

Active algorithms. These methods are implemented with and without an initial warm-up phase (using Algorithm 2), and are operating under the assumption of a follower best-response oracle, given a leader policy that is actively selected.

• Active IRL (ours): An implementation of Algorithm 1.

• Fixed random: An MLE-based reward extraction similar to Algorithm 1, where the leader policy is randomly sampled then kept fixed throughout all online iterations.

• Online random: A similar MLE-based algorithm using a new randomly sampled leader policy at each online iteration.

Fixed-expert algorithms. These methods assume access to a fixed expert, from which they can collect samples. The expert is defined with respect to a fixed (but unknown) leader reward, which is also in contrast to active algorithms.

• Fixed-expert MLE: Uses the same constrained MLE as the active methods, but fixing the policy as a known Stackelberg optimal leader.

• Follower-only OLS: Recovering the ordinary least-square solution of the empirical version of the linear system of Equation 2. This assumes access to an entropyregularized joint-QRE expert oracle.

• Zero-sum OLS: Similar to Follower-only OLS and following the analysis of Liao et al. (2025), leverages the fact that the game is known to be zero-sum and taking the ordinary least squares solution of the stacked system formed by the linear equality constraint of Equation 2 and its leader counter-part.

For each algorithm, we term target SubOpt (respectively target Fisher) the Stackelberg sub-optimality of the policy obtained by optimizing the leader on the learned reward (respectively the norm in the Fisher information geometry described by the reward transfer lemma, Lemma 3), for the true leader reward used during training.

Similarly, the averaged transfer SubOpt and averaged transfer Fisher describe the same quantities, but averaged over ten selected leader rewards. These ten downstream tasks have been chosen from a pool of 500 randomly chosen reward candidates, by greedily maximizing diversity of payof matrices. The pool is disjoint from the tasks used for hyperparameter selection. Averaged transfer SubOpt and averaged transfer Fisher are used as proxies for measuring the strategic-transferability of rewards.

## H.1 Lobéké environment from the GUARD benchmark

We provide additional experiments using the Lobéké National Park environment from the GUARD benchmark (Krever et al., 2026), in the same setup as in the main text. We describe it below.

Task The environment models an anti-poaching scenario in the Lobéké National Park (Cameroon), designed from a dataset of recorded elephant movements. This classical green security game involves patrols (leader player) to protect wildlife against poachers (follower player). Both players distribute in the park and the followers are rewarded based on the area they occupy (as they can poach more animals) and the presence of patrols (as patrols are limiting their capacity to poach). On the other hand, the patrols can be incentivized in minimizing the rewards for the poachers, but also in trading-of this objective with the covering of prioritized areas.

Model (illustrated in Figure 4) The game is model as a $5 \times 5$ grid comprised of 23 accessible locations. A ranger action action $a \in { \mathcal { A } }$ is one of the 23 feasible ten-step patrol schedules generated by GUARD, while the follower response $b \in B$ is the choice of one of the 23 targets. At a given time step t, the leader first chooses a distribution over patrol schedules, to which the follower reacts by choosing a location distribution, modeled as the entropy-regularized best-response to the patrol policy. Taking the role of the leader and observing sampled actions $b _ { t } \sim \nu _ { \mu _ { t } } ^ { * }$ , the learner can therefore learn information about the follower’s reward and, in the active case, adapt its subsequent patrol policies.

Additionally, Zero-sum OLS also observes a sampled expert leader action from its known expert follower. Therefore, this algorithm observes twice as many samples as the other method (one additional leader action for every follower action).

![](images/2202d9bab0418845bb21d5a3f20a9ef5edf2f5338918e06d4f43ad1e7fc191b9.jpg)  
Figure 4: Rewards, patrols, and poacher behavior in the Lobéké environment. The 23 accessible poaching targets are embedded in a $5 \times 5$ park grid, darker cells indicate larger concentration, and the triangle marks the ranger base.

For every pair $a , b \in \mathcal { A } \times \mathcal { B }$ , let us denote by v<sub>b</sub> the poacher’s value for target b and $c _ { a , b } \in [ 0 , 1 ]$ the fraction of target b covered by the patrol at a. From this, we define the following security-game types of features:

$$
\phi _ { \mathrm { r a w } } ( a , b ) : = ( c _ { a , b } , \ v _ { b } , \ c _ { a , b } v _ { b } ) , \qquad \phi ( \cdot , \cdot ) : = \phi _ { \mathrm { r a w } } ( \cdot , \cdot ) / \operatorname* { m a x } _ { a , b } \lVert \phi _ { \mathrm { r a w } } ( a , b ) \rVert _ { 2 } .
$$

From this, we define the true reward as:

$$
r _ { 2 } ( a , b ) : = \langle \phi ( a , b ) , \omega ^ { * } \rangle , \qquad \omega ^ { * } : = ( 0 , 1 , - 2 / 3 ) ,
$$

where $\omega ^ { * }$ is unknown to the algorithms and extracted from the generated GUARD matrix. We define during training the target leader objective as $r _ { 1 } : = - r _ { 2 }$ to make the game zero-sum and be compatible with all the baselines. In particular, this allows for the Zero-sum OLS method extracted from Liao et al. (2025). In general, this is not imposed, and we will measure transfer behaviors to multiple leader rewards. Throughout, we fix $\eta \approx 2 . 1 6 5$

We parametrize the transfer objectives of the patrols with a target-priority vector $z \in [ - 1 , 1 ] ^ { 2 3 }$ For each $z , \ r _ { 1 } ^ { z }$ trades-of between covering the same cells as the poachers and covering prioritized cells:

$$
r _ { 1 } ^ { z } ( a , b ) = ( 2 c _ { a b } - 1 ) z _ { b } .
$$

To ease interpretation, we will report metrics computed from the normalized objective $\begin{array} { r } { \overline { { r } } _ { 1 } ( \cdot , \cdot ) = ( r _ { 1 } ( \cdot , \cdot ) - \operatorname* { m i n } _ { a , b } r _ { 1 } ( a , b ) ) / ( \operatorname* { m a x } _ { a , b } r _ { 1 } ( a , b ) - \operatorname* { m i n } _ { a , b } r _ { 1 } ( a , b ) ) } \end{array}$ , such that $R _ { \mathrm { m a x } } = 1$ and to simplify the interpretation<sup>14</sup>.

Sampling priority vectors uniformly as random from [−1, 1]<sup>23</sup>, we generated 500 candidate transfer objectives, before flattening their induced reward matrices. From those we select 10 rewards maximizing diversity in payof vectors. These are used to define our transfer metrics as averaged over these candidate objectives.

In the following, we provide a completion of the results present in Section 5. For the diferent experiments, we summarize the tuning procedure in Appendix 8.

## H.1.1 Completed comparison of the algorithms: target and averaged transfer

We evaluate how the diferent algorithms perform for the target task and show how they transfer to multiple downstream tasks, without additional interactions in the environment.

![](images/a177ef248f55d08225d98bf8b4621e9e50227e6b9718bf83615768c192fae89c.jpg)  
Figure 5: Target and transfer diagnostics as a function of the number of observed poacher actions. We evaluate the algorithms across 50 distinct seeds.

As can be seen on Figure 5, our algorithm Active IRL presents a real advantage in terms of transfer to multiple downstream tasks compared to fixed-expert methods, and outperforms

the other active method baselines. For the target task, the gap with the Fixed-expert MLE algorithm is significantly reduced, but our algorithm remains competitive without assuming access to a fixed-expert oracle.

## H.1.2 Completed comparison of the algorithms: behaviors at diferent transfer dificulties

To deepen the understanding of transfer abilities of the diferent algorithms, we evaluate them on a broader spectrum of tasks

First, we draw a set of 180 priority vectors independently at random, before normalizing their induced reward matrices to [0, 1] using a positive afine transformation.

Second, we measure a notion of task dificulty through the sub-optimality of the source optimal leader policy:

$$
D _ { \mathrm { g a p } } ( z , \mathrm { s r c } ) = \mathrm { S u b O p t } _ { z } ( \mu _ { \mathrm { s r c } } ^ { \star } ) , \qquad \mu _ { \mathrm { s r c } } ^ { \star } \in \arg \operatorname* { m a x } _ { \tilde { \mu } } J _ { \mathrm { s r c } } ^ { \mathrm { l e a d e r } } ( \tilde { \mu } ) .
$$

Finally, we construct dificulty buckets and measure the average performance of the diferent algorithms for each of those buckets. Each bucket comprises exactly 60 tasks.

The results can be found in the heatmaps of Figure 6 below.

General data-collection assumptions

Fixed-expert assumptions

(a) Active IRL (warm-up)  
![](images/de7a2d6d2abd90ba154d2fc8072f3d77f1bf50711c38e37a5badfe9725dd0fd0.jpg)

(b) Active IRL (no warm-up)  
![](images/a2629d388ffe5bdf3803b488ff4c5a6bb36e6e3ff9c43a00380bfa270a252f8d.jpg)  
(d) Fixed random (warm-up)

(c) Fixed-expert MLE  
![](images/e3a1c96ce6df27ae804b63ee0c681af6916b8ef1e1a8e8f61a23a2728cfe8a81.jpg)  
(f) Zero-sum OLS

(e) Fixed random (no warm-up)  
![](images/14a515cb26f0dcddd6101011b1f245924b3490ccd31a14ed1bcd1e5d3f83dcae.jpg)

![](images/99a66fed1f6d36929da9d1ddcf0f6c62c89166f4ca1e55671f958277a718388c.jpg)

![](images/5dec75851782215447383fac86c19eabc25ec0a74f21dc094b2d922eeefc77e8.jpg)

(g) Online random (warm-up)  
![](images/f2ace153f22ad7ac7b4c3209ec69833436a60432e1a0b9d7dc9f04602e6e5c95.jpg)

(h) Online random (no warm-up)  
(i) Follower-only OLS  
![](images/975af6c3fbb8ab6d2236ad018ea6b3c095595af76de2a0dfda9de53c773ab417.jpg)  
Number of observed poacher actions

![](images/ea850e442648a5662154abf2a0d9206c2b78d77b8428f75f72d69c48de92f208.jpg)

![](images/f083a602c9f8765b855c59a4b75c26de62326f5b4b7e574729088f87c7471fbe.jpg)  
Mean transfer SubOpt(ˆµ)  
Figure 6: Transfer SubOpt by the downstream value gap of the true sourceoptimal leader policy. Cells average results for tasks in their respective bucket, each forming exactly $1 / 3$ of the tasks. Averaged over 50 seeds.

## H.1.3 Completed comparison of the algorithms: illustrative transfer tasks

We now extend the Figure 3 from the main document by reporting in Figure 7 the results for the same downstream tasks and for all the algorithms, with and without initial warm-up.

![](images/019259245536eed07e633e552e5a49f1acad04dd402b881dcbd5f4ae88c3d2e3.jpg)  
Ground truth Active IRL, Fixed random, Online random (Active) Fixed-expert MLE, Zero-sum OLS, Follower-only OLS (Fixed)

Figure 7: Illustrative patrol policies for diverse tasks. Algorithms with logit-response and fixed-expert assumptions. Illustrations of the learned policies for diferent leader rewards. Averaged over 50 seeds, reporting mean target and final SubOpt. w/ is for the performance when using warm-up.

These illustrations are mostly for intuition, we do not argue that they represent neither typical realistic leader tasks nor that they highlight the worst-case leader rewards for the algorithms.

## H.2 Hyperparameter selection and sensitivity analysis

We present in Figure 8 a sweep over pairs of information multiplier $\lambda / \eta ^ { 2 }$ and warm-up budget, to describe the sensitivity of Active IRL to both parameters. We compute each transfer SubOpt mean over 100 fixed and randomly selected tasks, each task performance averaged over 32 seeds.

![](images/11bfa848b9a4be596fc34465c857ac2acb7059d4ea36c8a6c8800b4faea05f59.jpg)  
Figure 8: Joint selection of the warm-up budget and information multiplier for Active IRL after 1,600 observed poacher actions. Means computed over 100 fixed tuning tasks and 32 paired seeds. We use an MLE ridge of $1 0 ^ { - 6 }$ for speed and stability (same as in the evaluation pipeline).

We use this to select hyperparameters for our experiments. Specifically, we select the minimum mean transfer SubOpt. The tasks and corresponding random seeds are only used for tuning and not shared with the evaluation pipeline.

As can be seen in the heatmap, the performances are similar across the diferent configurations, and sensitive to the exact seeds and tasks used. Among those configurations, we therefore choose the best-performing configuration with positive warm-up and without warm-up $( n _ { \mathrm { w a r m } } = 5 0 , \lambda / \eta ^ { 2 } = 0 . 0 3$ and $n _ { \mathrm { w a r m } } = 0 , \lambda / \eta ^ { 2 } = 0 . 1 )$ . This allows to understand the impact of the warm-up, which is minimal in the Lobéké task.

## H.3 Diagnosis in synthetic environments

## H.3.1 Active selection in sparse-information games

We now highlight the benefits of active policy selection by designing a game where the Fisher information is very sparse, i.e. most of the possible leader actions are not informative for learning ω. To this end, we set up a game with m leader actions, such that only the first leader action afects the follower’s reward.

Specifically, the game is defined as $\mathscr { A } = [ m ] , \mathscr { B } = [ 2 ] , \omega = 1 , \eta = 2$ and $r _ { 2 } ^ { \omega } ( a , b ) = \omega \cdot { \bf 1 } \{ a =$ $1 , b = 2 \}$ , where only the leader action $a = 1$ is informative. All other leader actions provide no information about $\omega .$ . The leader rewards are $r _ { 1 } ( 1 , b ) = { \bf 1 } \{ b = 2 \}$ and $r _ { 1 } ( a , b ) = 3 / 4$ for $a > 1$

For this experiment, we report results for Active IRL without warm-up to isolate the online policy-selection behavior of our algorithm.

As can be seen in Figure 9, our algorithm Active IRL is able to quickly learn ω and thus minimizing the sub-optimality of the computed policy $\hat { \mu }$ . In comparison, the two baselines are not able to find useful information eficiently. As expected, the performance discrepancy is increasing with the number of leader actions m.

(a) Target SubOpt at m = 16  
![](images/c4814259615f6d24d607ff4785cb556a1f19a43aeca9456c373aaa3aae8dd903.jpg)  
Number of observed follower actions

(b) Target Fisher error at m = 16  
![](images/bbf65f668745f35615e5018177dc4fed761d9f815dd23e7ce39f0f8ee8892bb9.jpg)  
Number of observed follower actions  
(d) Target Fisher error after 200 observations

(c) Target SubOpt after 200 observations  
![](images/15c9dc0473fd75450ccd32049caac5708621331eee15e1d421a43e5e9a0ffb8c.jpg)

![](images/352d9cc1fee320343804231c26fc3c8ab2d314e5b30141ed6dfe7f4a1563607e.jpg)  
Figure 9: Target metrics in the sparse-information game. Averaged over 100 seeds.

## H.3.2 Active IRL on the lower-bound instance

We recall the hard instance as follows.

Definition 5 (Hard instance). Let $\mathcal { A } = [ d ] , \mathcal { B } = \{ b _ { 1 } , b _ { 2 } \} , R _ { \mathrm { m a x } } \in \mathbb { R } _ { + }$ , and

$$
\phi ( a _ { i } , b _ { 1 } ) = 0 , \qquad \phi ( a _ { i } , b _ { 2 } ) = e _ { i } , \qquad r _ { 1 } ( a , b ) = R _ { \mathrm { m a x } } \cdot { \bf 1 } \{ b = b _ { 2 } \} .
$$

We identify $\mu \in \Delta _ { d }$ and write $\ell ( x ) : = ( 1 + e ^ { - x } ) ^ { - 1 }$ for the logistic function.

From this hard instance, we were able to derive the following lower bound.

Theorem 1 (Minimax lower bound). Let d ≥ 32, $\begin{array} { r } { \epsilon = \frac { 1 } { 2 \eta } \sqrt { d / K } \le S } \end{array}$ and $\boldsymbol \omega ^ { ( i ) } : = \varepsilon e _ { i }$ for $i \in [ d ]$ . There exists a game such that for every algorithm producing an output $\hat { \mu }$ after $K \geq 4 d$ interactions,

$$
\operatorname* { m a x } _ { i \in [ d ] } \mathbb { E } ^ { ( i ) } \left[ \mathrm { S u b O p t } _ { \omega ^ { ( i ) } } ( \hat { \mu } ) \right] \ \geq \ \frac { 1 } { 4 4 } R _ { \operatorname* { m a x } } \sqrt { \frac { d } { K } } ,
$$

where $\mathbb { E } ^ { ( i ) }$ is for the expectation under the trajectory law $\mathbb { P } ^ { ( i ) }$ induced by $\boldsymbol { \omega } ^ { ( i ) }$

This minimax result lower bounds the worst-case SubOpt over all learning algorithms. As we cannot enumerate and run all such possible algorithms, we instead report below the performance of our algorithm Active IRL for the hard instance. This is more illustrative than a direct validation of the lower bound (see the full proof in Appendix C), but it does not invalidate the bound with similar rates, hence working as a sanity check.

In our experiment below, we simplify the instance by selecting $\eta = 1 , R _ { \mathrm { m a x } } = 1$ and $S = 1$ To ease the interpretation, we disable warm-up and set $\lambda = \eta ^ { 2 }$ , therefore only keeping the online part of Algorithm 1. Finally, we directly use the available closed-form leader policy for evaluation.

![](images/c224250c365c38468c5ac503e0e1d1cf20362f473f220de83e015cd6fe644531.jpg)  
Figure 10: Empirical scaling on the minimax hard instance. We use 16 independent seeds per coordinate and report the maximum sub-optimality mean over all coordinates.