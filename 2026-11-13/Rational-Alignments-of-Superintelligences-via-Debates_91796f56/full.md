# Rational Alignments of Superintelligences via Debates

Thanawat Sornwanee Stanford University

November 13, 2026

## Abstract

Artificial superintelligences (ASI) is artificial intelligences with greater capability. However, as human principal has to use it, we have to ensure that the agent will perform in the interest of the human. We investigate the framework of ASI-assisted question answering where ASI agents know the correct answer but the human does not. We provide some example of mechanism for rational alignment, where the agent utility is unchanged, but the interaction between the agent and the human principal is controlled to be in a certain fashion. We show specifically that a debate framework where the agents can further reason and behave in a subgame perfect equilibrium manner will lead to a more predictable result, and in some case will be favorable.

## 1 Introduction

Artificial Superintelligence (ASI) is one of the speculated developments of Artificial Intelligence (AI), where the system has a suficiently high capacity and can attain superhuman performance in several domains. [BB17] The prospect of ASI seems to be inevitable given the current rapid development in Large Language Models (LLMs) as well as theirs emergent abilities in both natural language understanding and reasoning. [KYY<sup>+</sup>24]

Since modern usage of AI tools such as LLM often involves generating suggestion, facilitating decision making, or even implementing some automated tasks, the human principal then needs to ensure that the AI agent action will be made to improve the human principal utility. The problem can then happen when the objective function the AI agent tries to optimize for is diferent from the utility function of the human principal. We denote this problem as a “misalignment". In the current development of AI, the limited capacity of AI may make the regret negligible, but it is possible to envision that a future development of ASI will make the regret from misalignment to be more severe an that it is important for us to prepare some alignment technique for the future.

The applications of AI as well as potential application of ASI seems to be quite broad, so the misalignment as well as the alignment may be varied to diferent setting. In this paper, we will focus on the case of ASI-assisted question answering framework. This is the setting where ASI has costless access to the underlying state but the human does not. The ASI agent then has to convince the human principal of the underlying state.

## 2 ASI-Assisted Question Answering

We are interested in the framework where there is an underlying state of the world $\theta \sim \Pi \in \Delta ( \Theta )$ where the set of every possible state Θ is, for simplicity, finite. The realization of θ is not observed

by the user, but the user is tasked to answer

$$
\hat { \theta } _ { \mathrm { h u m a n } } \in \Theta
$$

to be as close to the reality.

We consider the case when the utility of the human is

$$
U _ { \mathrm { h u m a n } } = \mathbf { 1 } _ { \hat { \theta } _ { \mathrm { h u m a n } } = \theta } ,
$$

meaning that the human wants to maximize the probability of the answer being exactly correct. Although this may not be the use case for most agentic system, one can imagine that the guess $\widehat { \theta } _ { \mathrm { h u m a n } }$ will be used consequentially by the human in other decision making task downstream, and we use the reduced form to be robust to diferent downstream task.

We assume a certain rationality in the human. Specifically, given a subjective posterior belief (after interaction with the agentic system) $\hat { \mu } \in \Delta ( \Theta )$ and a feasible choice set $\hat { \Theta } \subseteq \Theta$ as specified by the mechanism, the human will choose

$$
\hat { \theta } _ { \mathrm { h u m a n } } \in \operatorname * { a r g m a x } _ { \theta ^ { \prime } \in \hat { \Theta } } \hat { \mu } \left( \{ \theta ^ { \prime } \} \right) .
$$

The simple baseline system is when there exists only a single ASI agent. The ASI agent is capable fo observing the underlying state $\theta .$ The ASI agent is then queried by the human about the realization of $\theta ,$ and gets to give the answer

$$
\hat { \theta } \in \Theta ,
$$

as well as providing some “evidence", such as argument and justification, in order to persuade the human to think that the state $\hat { \theta }$ is more likely. Such evidence will increase the subjective probability of the event that $\theta = { \hat { \theta } }$ . We consider a simple subjective belief update as a Bayesian with selection neglect.

## 2.1 Subjective Belief

The irrationality will come in as “selection neglect" where the update of the posterior is done in a subjective manner ignoring where the evidence comes from. This makes the posterior update independent from the equilibrium strategy specifying the evidence selection conditioned on the observed evidence.

We represents the belief, which is assumed to be full support, by its corresponding energy $\hat { s } \in \mathbb { R } ^ { \Theta }$ such that

$$
\hat { \mu } \left( \{ \theta ^ { \prime } \} \right) = \frac { e ^ { s _ { \theta ^ { \prime } } } } { \sum _ { \theta ^ { \prime \prime } \in \Theta } e ^ { s _ { \theta ^ { \prime \prime } } } }
$$

for all $\theta ^ { \prime } \in \Theta$

We assume that the ASI can always find an evidence in support of its claim $\hat { \theta } \in \Theta$ . By doing so, this will makes $s _ { \hat { \theta } }$ becomes

$$
s _ { \hat { \theta } } + \left( 1 + \lambda \mathbf { 1 } _ { \hat { \theta } = \theta } \right) ,
$$

while keeping $s _ { \theta ^ { \prime } }$ for $\theta ^ { \prime } \neq \hat { \theta }$ the same value. Note that this is equivalent to selection neglect with independence evidence assumption.

The parameter $\lambda \in$ R specifies the plausibility of the true state, determining how easy or hard it is for the ASI agent to motivate the true answer θ to the human. Note that we can also think of it as a random variable that is dependent on θ. In such case, it is not hard to imagine the case when $\lambda < 0$ given that some certain state $\theta \in \Theta$ may not have a lot of evidence to be supported. For notational simplicity, we assume that λ is a constant.

## 2.2 Why Naive ASI-Assited Question Answering is Bad

The naive framework will be when there is no restriction on what the human can choose, so $\hat { \Theta } = \Theta$ In this case, we will see that, if λ is not suficiently big and θ is realized to be such that $\Pi ( \{ \theta \} )$ is small, then the ASI agent may not want to motivate the correct answer just for the human to choose the apriori most probable answer instead. The ASI agent may want to lie and choose the most probable state directly instead.

Specifically, we assume that the utility of the ASI agent is

$$
U = I + \epsilon U _ { \mathrm { h u m a n } } ,
$$

where I is the random indicator whether the human has chosen the answer given by such agent, and $\epsilon \in \mathbb { R }$ determine the level of altruism in the ASI agent. If $\epsilon = 0$ , the ASI agent does not have a preference over the utility of the human, and, if $\epsilon < 0$ , the ASI prefers human to get an incorrect answer. Similarly, we can essentially make ϵ a random variable dependent on the state θ to represent the case where the ASI agent may have bias for a certain state. However, we assume that ϵ is a constant for notational simplicity.

We assume that the agent uses the equilibrium strategy, so it is maximizing its utility $U$

Thus, we have the following corollary.

Corollary 1. If mi $1 \theta ^ { \prime } \in \Theta$ ln $\Pi ( \{ \theta \} ) + 1 + \lambda <$ max<sub>θ</sub>′<sub>∈Θ</sub> ln $\Pi ( \{ \theta ^ { \prime } \} )$ , and $\epsilon < 1$ , then there does not exist an equilibrium where $\hat { \theta } _ { h u m a n } = \theta$ almost surely.

Proof. This is because, even when the agent is truthful and asnwers ${ \hat { \theta } } = \theta$ , the subjective belief score of the correct answer in the event when the correct answer is the least likely event will just be min $\iota _ { \theta ^ { \prime } \in \Theta } \ln \Pi ( \{ \theta \} ) + 1 + \lambda$ □

This suggests that we require some alignment. Specifically, we will see that this will be done by changing specifying the controllable and observable environment. Below, we are considering two natural examples and their behaviors.

Example 1 (Single Choice). If we first does not allow the choice sets of the human agent to be the entire set Θ but force it to be $\{ \hat { \theta } \}$ , then we can clearly see that we remove the incentive force for the ASI agent to choose an easier answer, which is the one with higher prior. This will make the ASI agent only cares about the utility of human. Still, we will only have good equilibrium only when $\epsilon \geq 0$

In the case where the agent has $\epsilon < 0$ , we will have that every equilibrium will lead to always incorrect $\hat { \theta } _ { h u m a n }$ as long as the possible states $| \Theta | \ge 2$

Example 2 (Long Sequence). We can still allow the choices set to be $\Theta ,$ but now increase the interaction to be $T > > 1$ timesteps. Thus, as long as $\begin{array} { r } { T > \frac { \operatorname* { m a x } _ { \theta ^ { \prime } \in \Theta } \ln \Pi ( \{ \theta ^ { \prime } \} ) - \operatorname* { m i n } _ { \theta ^ { \prime } \in \Theta } \ln \Pi ( \{ \theta ^ { \prime } \} ) } { 1 + \lambda } } \end{array}$ , the ASI can always convince the human principal of the correct answer. Still, such capability does not imply that the ASI behavior will follow. In fact, $i f \epsilon < 0$ , and there is an incorrect state with high prior, the ASI agent can spend the T interactions in increasing the human principal subjective belief in this incorrect answer.

We will see that, in both examples, the bad scenario is when $\epsilon < 0 .$ . However, this is not universal and is attributed to that there is only 1 ASI agent in the system, thereby having no competition. In the next section, we will show that having competition “may" help with the system.

## 3 Debate System

We introduce a debate system into the question-answering framework. Instead of interacting with only a single ASI agent, the human will get to interact with $n \in \mathbb { N } { \mathrm { ~ A S I } }$ agents in a structured manner.

The first phase of the debate is the answer commitment. Each agent $i \in \{ 1 , 2 , 3 , . . . , n \}$ gets to see the answers of the previous agents and answers

$$
\hat { \theta } _ { i } \in \Theta .
$$

The feasible choice set for the human is

$$
\begin{array} { r } { \hat { \Theta } = \left\{ \hat { \theta } _ { i } : i \in \left\{ 1 , 2 , \dots , n \right\} \right\} . } \end{array}
$$

$\operatorname { I f } \left| { \hat { \Theta } } \right| = 1$ , then the answer is unanimous, and there is no debate. The human then has to choose $\widehat { \theta } _ { \mathrm { h u m a n } }$ to be the only element in $\big | \hat { \Theta } \big |$ . The selection of agent is then done uniformly at random.

If $\left| \hat { \Theta } \right| > 1$ , then the second phase, which is the debate phase, is triggered. In each round, each agent $i \in \{ 1 , 2 , \ldots , n \}$ gets to present a piece of evidence in support of its claim $\hat { \theta } _ { i }$ . There are $T \in \mathbb { N }$ rounds in total. Afterwards, the human principal gets to choose $\widehat { \theta } _ { \mathrm { h u m a n } }$ from $\hat { \Theta }$ . If many agents has $\hat { \theta } _ { i } = \hat { \theta } _ { \mathrm { h u m a n } }$ , then the human selection of agent is done uniformly at random among them.

We assume that solution concept is subgame perfect equilibrium, and we assume that the ASI agent will behave according to subgame perfect equilibrium.<sup>1</sup>

Theorem 1. If $\lambda > 0$ and $\epsilon \ < \ 0$ , then under regularities $| \lambda | < { \frac { 1 } { n } } , | \epsilon | < { \frac { 1 } { n ^ { 2 } } } , | \Theta | = 2$ , and $\begin{array} { r } { T > \frac { \operatorname* { m a x } _ { \theta ^ { \prime } } \ln \Pi ( \{ \theta ^ { \prime } \} ) - \operatorname* { m i n } _ { \theta ^ { \prime } } \ln \Pi ( \{ \theta ^ { \prime } \} ) } { \vert \lambda \vert } } \end{array}$ , then we have that

• $I f ^ { \prime }$ n is even, then, in equilibrium, every ASI agent chooses

$$
\widehat { \theta } _ { i } = \theta .
$$

• $I f$ n is odd, then, in equilibrium, every ASI agent chooses

$$
\hat { \theta } _ { i } \neq \theta .
$$

Proof. We denote A as the correct answer and B as the incorrect answer.

We first consider the case where there exists even number of agents. Since $\lambda > 0$ is small, and there are even agents, the answer A will be selected by the human principal if and only if the total number of agents selecting A is $\textstyle \geq { \frac { n } { 2 } } \in \mathbb { N }$

We will show this by that in the backward induction step, by denoting the current number of agents so far who has chosen the incorrect answer B as $b \in  { \mathbb { N } } _ { 0 }$ , then the only subgame perfect equilibrium action is to choose to propose the correct answer A whenever $b \leq \frac { n } { 2 }$ . This essentially means that, if by choosing the incorrect answer cannot immediately ensure the winning of $B { \mathrm { , } }$ which requires ${ \frac { n } { 2 } } + 1$ , then the agent will not do it in equilibrium.

If $\textstyle b \geq { \frac { n } { 2 } } + 1$ , then it is obvious that choosing to propose A is suboptimal, since A will never win. It is then better to choose B and share the probability of getting selected with other agents.

For the case, of $b \leq \frac { n } { 2 }$ , the proof can be easily seen by mathematical induction. As the base case, we consider the last agent. Unless $\begin{array} { r } { b = \frac { n } { 2 } } \end{array}$ , choosing B will place the last agent in a minority side. In the case where $\begin{array} { r } { b = \frac { n } { 2 } } \end{array}$ , choosing B will make B wins but with $\frac { n } { 2 }$ other agents to share the reward with, while choosing A will reduce such number to $\frac { n } { 2 } - 1$ . Thus, it is also strictly better to choose A. The induction case proceeds similarly. If $\begin{array} { r } { \mathrm { ~ : ~ } b \geq \frac { n } { 2 } + 1 } \end{array}$ , then B has already won, so choosing B is better. Otherwise, every subsequent agent will choose A, so choosing A is optimal.

Note that the solution from backward induction does include the of-equilibrium path action. We can use this rationality requirement to show that the first agent will choose A. The subsequent agents will also follow in choosing A.

In the case where n is odd, there is no need for tie-breaking, making λ irrelevant. We similarly, have that, by denoting a as total number of agents who has proposed A so far, the equilibrium action for the current agent is to choose B if and only if $\textstyle a \leq { \frac { n - 1 } { 2 } }$ . The induction process is similar.

We also have a converse result

Theorem 2. If $\lambda ~ < ~ 0$ and $\epsilon > 0$ , then under regularities $| \lambda | < { \frac { 1 } { n } } , | \epsilon | < { \frac { 1 } { n ^ { 2 } } } , | \Theta | = 2$ , and $\begin{array} { r } { T > \frac { \operatorname* { m a x } _ { \theta ^ { \prime } } \ln \Pi ( \{ \theta ^ { \prime } \} ) - \operatorname* { m i n } _ { \theta ^ { \prime } } \ln \Pi ( \{ \theta ^ { \prime } \} ) } { \vert \lambda \vert } } \end{array}$ , then we have that

$I f n$ is odd, then, in equilibrium, every ASI agent chooses

$$
\widehat { \theta } _ { i } = \theta .
$$

$I f n$ is even, then, in equilibrium, every ASI agent chooses

$$
\hat { \theta } _ { i } \neq \theta .
$$

Proof. The proof is the same as that of the theorem 1.

These two theorems suggest that the nature of ASI agents and the problem of interest is important for the design of the debate in how many agents we should deploy.

## 4 Connection to AI Debate Literature

The paper [ICA18] introduces the notion of AI debates. It briefly mentions the zero-sum aspect of the game but does not formalize nor analyze the game. [KC19] attempts to formalize into a game but with reduced form result where the designer can choose the utility function of the agents.

Other papers require the observation of the unobserved state for its game formulation [BCIP25], or creates a fictitious game merely for analysis and training.[BCIP24, CCFH24, BCIM<sup>+</sup>26] In this paper, we formally analyze the behavior of the game created by a debate system where the agents are superintelligences.

## References

[BB17] Anthony M Barrett and Seth D Baum. A model of pathways to artificial superintelligence catastrophe for risk and decision analysis. Journal of Experimental & Theoretical Artificial Intelligence, 29(2):397–414, 2017.

[BCIM<sup>+</sup>26] Jonah Brown-Cohen, Geofrey Irving, Simon C Marshall, Ilan Newman, Georgios Piliouras, and Mario Szegedy. Debate is eficient with your time. arXiv preprint arXiv:2602.08630, 2026.

[BCIP24] Jonah Brown-Cohen, Geofrey Irving, and Georgios Piliouras. Scalable ai safety via doubly-eficient debate. In International Conference on Machine Learning, pages 4585– 4602. PMLR, 2024.

[BCIP25] Jonah Brown-Cohen, Geofrey Irving, and Georgios Piliouras. Avoiding obfuscation with prover-estimator debate. arXiv preprint arXiv:2506.13609, 2025.

[CCFH24] Xinyi Chen, Angelica Chen, Dean Foster, and Elad Hazan. Playing large games with oracles and ai debate. In Agentic Markets Workshop at ICML 2024, 2024.

[ICA18] Geofrey Irving, Paul Christiano, and Dario Amodei. Ai safety via debate. arXiv preprint arXiv:1805.00899, 2018.

[KC19] Vojtěch Kovařík and Ryan Carey. (when) is truth-telling favored in ai debate? arXiv preprint arXiv:1911.04266, 2019.

[KYY<sup>+</sup>24] HyunJin Kim, Xiaoyuan Yi, Jing Yao, Jianxun Lian, Muhua Huang, Shitong Duan, JinYeong Bak, and Xing Xie. The road to artificial superintelligence: A comprehensive survey of superalignment. arXiv preprint arXiv:2412.16468, 2024.