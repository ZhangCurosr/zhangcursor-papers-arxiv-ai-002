# Reward Objectives for Robust (PO)MDPs: A Law-Preserving Equivalence with (Partially Observable) Games

Durgam Latha   
Indian Institute of Technology Bombay   
Mumbai, India   
Ðorde Žikeli¯ c´   
Nanyang Technological University   
Singapore, Singapore   
Dion Reji   
Indian Institute of Technology Bombay   
Mumbai, India   
Shankaranarayanan Krishna   
Indian Institute of Technology Bombay   
Mumbai, India

## Abstract

Robust partially observable Markov decision processes model agents that act under partial observability while adversarial environment resolves uncertainty in the transition probabilities. Reward objectives such as discounted-sum, long-run average and cumulative reward under finite and infinite horizons allow quantifying the utility of an agent. We consider the problem of solving RPOMDPs under all four reward objectives, where the goal is to decide whether the value of a risk measure with respect to a reward objective exceeds a given threshold. We consider different risk measures, such as (1) expected value, (2) mean-variance (MV), (3) conditional value-at-risk (CVaR), and (4) entropic risk (ERisk). Our results can be summarized as follows. First, we formalize the class of law-invariant risk measures, that subsume all four risk measures above. Second, we propose a unifiedframeworkfor solving RPOMDPs with discounted-sum, long-run average and cumulative reward objectives under any law-invariant risk measure, by reducing the problem to solving partially observable stochastic games (POSGs). Our unified framework applies to (s, a)-rectangular RPOMDPs with polytopic uncertainty sets, which we show to be equivalent to POSGs, for each of the four reward objectives and any law-invariant risk measure. We prove equivalence by designing logspace reductions between the RPOMDP and POSG models, that map strategies to strategies and preserve the distribution of the reward objective, up to a constant rescaling factor. Third, as a corollary, we derive new computational complexity results for R(PO)MDPs and POSGs. In particular, we show that solving (s, a)-rectangular RMDPs is as hard as solving simple stochastic games.

## 1 Introduction

Partially observable Markov decision processes (POMDPs) are the standard model of an agent acting under partial observability in the presence of stochastic uncertainty [23]. The key limitation of POMDPs, however, is that their transition probabilities are assumed to be fully known. This is a strong assumption in the settings where transition probabilities are e.g. estimated from data. Robust MDPs (RMDPs) [22, 34, 51] and, more generally, robust POMDPs (RPOMDPs) [37, 6], account for this by replacing fully known transition probabilities with an uncertainty set of transition probabilities, with the adversarial environment being able to choose transition probabilities from this uncertainty set. Recent years have seen significant interest in solving R(PO)MDPs with reward objectives, such as discounted-sum reward [35, 22, 24, 47, 17, 20, 2] or long-run average reward objectives [49, 19, 18, 50, 9, 32]. The goal in solving an RPOMDP is to compute a strategy that maximizes the expected value of the reward objective. Existing works largely focused on de signing efficient algorithms for solving RMDPs. Solving RPOMDPs, which also account for partial observability, has been considered in [6, 37, 7, 46, 33, 13, 16, 25].

These works all focus on computing a strategy that maximizes the expected value (E) of the reward objective. However, while providing valuable information on the average-case performance of the strategy, expected value alone does not consider the variability of possible outcomes. Two strategies that achieve the same expected reward objective value may carry very different risks. This is why expected value is sometimes replaced by other risk measures such as mean-variance [30], conditional value-at-risk (CVaR) [42, 10], or entropic risk [21, 3]. However, computing strategies that optimize these risk measures has so far mostly been studied for MDPs, but not for robust models such as RMDPs or RPOMDPs. The goal of this work is to address this gap.

We present a unified framework for solving RPOMDPs with reward objectives and different risk measures, by reducing the problem to solving partially observable stochastic games (POSGs). Our framework applies to discounted-sum, long-run average and total reward under finite and infinite horizons objectives, and under any law-invariant risk measure. A risk measure is said to be lawinvariant, if it only depends on the probability distribution of the value of the reward objective over RPOMDP runs. We show that common risk measures such as (1) expected value (E), (2) mean-variance (MV), (3) conditional value-at-risk (CVaR), and (4) entropic risk (ERisk) are all law-invariant. Hence, our unified framework applies to all these risk measures.

We consider (s, a)-rectangular RPOMDPs with polytopic uncertainty sets, which we show to be equivalent to POSGs. For each of the four reward objectives and under any law-invariant risk measure, we design two logspace reductions: one from RPOMDPs to POSGs, and one from POSGs to RPOMDPs. Our reductions map strategies to strategies and are law preserving, meaning that the image of every strategy achieves the same distribution of the reward objective value (up to the reward objective being normalized by a constant rescaling factor), thus mapping optimal strategies to optimal strategies under any law-invariant risk measure. Hence, we can reduce the problem of solving an RPOMDP to solving a POSG under the same reward objective and risk measure and vice-versa. In the case of RMDPs (with full observability), our reductions establish equivalence with turn-based stochastic games (TBSGs) [44, 11].

Building upon our equivalence result, we study the threshold problems for law-invariant risk measures such as E, MV, CVaR, and ERisk with rational parameters, and for the finite and infinite horizon total, long-run average and discounted-sum reward objectives. We consider the non-strict and strict threshold problems: given a model (RPOMDP/RMDP/POSG/TBSG), a risk measure Ψ, a reward objective X, and a rational t, decide whether the agent in the RPOMDP/RMDP (or the maximizing player in POSG/TBSG) can guarantee a value of Ψ on the reward objective X that is greater than t (strict threshold), or greater or equal than t (non-strict threshold). Table 1 summarizes our novel complexity results. Under partial observation, both variants are PSPACE-hard and decidable for finite horizon total reward, and undecidable for infinite horizon total reward and for long-run average reward objectives. For discounted-sum rewards, values can be approximated to any precision, which makes the strict problem recursively enumerable; it is nevertheless undecidable for E and CVaR, while for discounted mean-variance and entropic risk the decidability of the threshold problems remains open. Under full observation, both variants are SSG-hard for RMDPs with total or long-run average rewards for all four risk measures, and for discounted-sum rewards for the ex pected value. For the expected discounted-sum and the long-run average reward, the SSG bound is tight, since both problems lie in NP ∩ coNP. Here, SSG stands for the complexity of solving simple stochastic games, which is known to be in NP ∩ coNP [11] but the question of whether there exists a polynomial algorithm is an open problem.

Our contributions can be summarized as follows:

1. Law-invariant risk measures. We formalize law-invariant risk measures for reward objectives in RPOMDPs. These include (1) expected value (E), (2) mean-variance (MV), (3) conditional value-at-risk (CVaR), and (4) entropic risk (ERisk).

2. Equivalence ofRPOMDPs and POSGs. We present a unifiedframeworkfor solving RPOMDPs with reward objectives and law-invariant risk measures, by reducing the problem to solving

POSGs. Our framework applies to $( s , a )$ -rectangular polytopic RPOMDPs with discountedsum, long-run average, total and finite-horizon reward objectives. It is based on our novel equivalence result between RPOMDPs and POSGs obtained via logspace reductions in both directions. In the case of RMDPs (with full observability), we obtain equivalence with TB-SGs.

3. Uniform complexity landscape. We derive new computational complexity results and provide a complexity landscape for RPOMDPs (and POSGs) under reward objectives and lawinvariant risk measures.

## 2 Preliminaries

We write $\mathbb { N } = \{ 0 , 1 , 2 , \dots \} , \mathbb { Q } .$ and R for the sets of natural, rational, and real numbers. For a finite set Y, we use $\varDelta ( Y )$ to denote the set of probability distributions over $Y$ , and Dirac(y) the distribution with all probability mass on $y \in Y$ . We write $\overline { { \mathbb { R } } } = [ - \infty , + \infty ]$ and $\mathcal { D } ( \overline { { \mathbb { R } } } )$ for the set of Borel probability measures on ${ \overline { { \mathbb { R } } } } ,$ which we call laws. For a random variable $\dot { X } \colon \dot { \Omega }  \overline { { \mathbb { R } } }$ on a probability space $( \varOmega , \mathcal { F } , \mathbb { P } )$ and a Borel set $B \subseteq { \overline { { \mathbb { R } } } }$ , let $X ^ { - 1 } ( B ) = \{ \omega \in \Omega : X ( \omega ) \in B \}$ be the set of outcomes that X maps into $B .$ . The law of X is the measure ${ \mathcal { L } } _ { X } ( B ) = \mathbb { P } ( X ^ { - 1 } ( B ) ) =$ $\mathbb { P } ( X \in B )$ . For $c > 0$ and a law $\mu ,$ we write $( S _ { c } ) _ { \# } \mu$ for the law of $c X$ when X has law $\mu , \mathrm { i . e . }$ $( S _ { c } ) _ { \# } \mu ( B ) = \mu ( \{ x : c x \in B \} )$ for every Borel set B.

## 2.1 Robust POMDPs

RPOMDPs. A robust POMDP (RPOMDP) is a tuple

$$
\mathcal { M } = ( S , A , \mathcal { P } , s _ { 0 } , \mathcal { O } _ { a } , \mathcal { O } _ { \mathrm { e } } , \mathcal { Z } _ { a } , \mathcal { Z } _ { \mathrm { e } } , r ) ,
$$

where S and A are finite sets of states and actions, $s _ { 0 } \in S$ is the initial state, and $\mathcal { O } _ { \mathfrak { a } }$ and $\mathcal { O } _ { \mathtt { e } }$ are finite sets of observations of the agent and of the environment, with observation functions $\mathcal { Z } _ { \mathfrak { a } } \colon S \to \mathcal { O } _ { \mathfrak { a } }$ and $\mathcal { Z } _ { \mathsf { e } } \colon S \to \mathcal { O } _ { \mathsf { e } }$ . The reward function is $r \colon S \times A \times S \to \mathbb { Q }$ . The uncertainty set is given by ${ \mathcal { P } } \subseteq ( S \times A \to A ( S ) )$

A POMDP is an RPOMDP in which $\mathcal { P }$ is a singleton, specifying for each state-action pair in $S \times A$ a single probability distribution over successor states. An RMDP is an RPOMDP in which $\mathcal { Z } _ { \mathfrak { a } }$ and $\mathcal { Z } _ { \mathrm { e } }$ are injective, $\mathrm { i . e . } .$ , every observation corresponds to a unique state. An $M D P$ is an RPOMDP that satisfies both conditions above.

Assumptions. As is standard in the algorithmic study of robust Markov models [22, 51, 9, 28], we consider $( s , a )$ -rectangular polytopic uncertainty sets of the form $\begin{array} { r } { \mathcal { P } = \prod _ { ( s , a ) \in S \times A } \mathcal { P } ( s , a ) } \end{array}$ , where the environment chooses transition probabilities independently across state-action pairs, with each ${ \mathcal { P } } ( s , a ) \subseteq { \varDelta } ( S )$ being a polytope with a finite vertex set $\begin{array} { r } { V _ { s , a } . \mathcal { P } ( s , a ) = \{ \sum _ { i = 1 } ^ { | V _ { s , a } | } \alpha _ { i } v _ { i } \ | \ \alpha _ { i } \ \in } \end{array}$ $[ 0 , 1 ] , \sum _ { i } \alpha _ { i } \ = \ 1 , v _ { i } \ \in \ V _ { s , a } \}$ , each vertex $\textit { v } \in \textit { V } _ { s , a }$ has the form $( x _ { 1 } , \ldots , x _ { | S | } )$ with $x _ { i } \geq$ 0 and $\textstyle \sum _ { i } x _ { i } = 1$ . We assume that the uncertainty set $\mathcal { P }$ is given in the vertex set representation.

Semantics. The RPOMDP starts in $s _ { 0 }$ and proceeds in steps $t = 0 , 1 , \ldots$ . to generate a random infinite play. If at time t the RPOMDP is in state $s _ { t }$ , the agent receives the observation $\mathcal { Z } _ { \mathsf { a } } ( s _ { t } )$ and chooses an action $a _ { t } \in A$ . The environment receives the observation $\mathcal { Z } _ { \mathsf { e } } ( s _ { t } )$ and action $a _ { t }$ and selects a probabilistic transition function $P _ { t } \in { \mathcal { P } } ;$ ; the next state $s _ { t + 1 }$ is drawn from $P _ { t } ( s _ { t } , a _ { t } )$ , and the transition earns the reward $r _ { t } = r ( s _ { t } , a _ { t } , s _ { t + 1 } )$ . This follows the RPOMDP game semantics of [6] for dynamic uncertainty, in which the environment may choose a new assignment in every step.

The semantics are formalized via agent and environment strategies. A history is a finite sequence $h _ { t } = s _ { 0 } , a _ { 0 } , s _ { 1 } , \ldots , s _ { t } \in ( S \times A ) ^ { \star }$ . An agent observation history for history $h _ { t }$ is a sequence $h = o _ { 0 } ^ { a } o _ { 1 } ^ { a } \cdot \cdot \cdot o _ { t } ^ { a } \in { \mathcal { O } } _ { \mathsf { a } } ^ { + }$ of observations of the agent such that $o _ { t } ^ { a } = \mathcal { Z } _ { \mathsf { a } } ( s _ { t } )$ , and let $\mathcal { H } _ { \sf a }$ denote the set of agent observation histories. Then, an agent strategy is a map $\sigma \colon { \mathcal { H } } _ { \mathfrak { a } } \to \varDelta ( A )$ . Likewise, an environment observation history of $h _ { t }$ is a sequence $h = o _ { 0 } ^ { e } o _ { 1 } ^ { e } \cdot \cdot \cdot o _ { t } ^ { e } \in { \mathcal { O } } _ { \mathrm { e } } ^ { \ + }$ such that $o _ { t } ^ { e } = \mathcal { Z } _ { \mathsf { e } } ( s _ { t } )$ of observations of the environment, and let $\mathcal { H } _ { \mathrm { e } }$ denote the set of environment observation histories. Then, an environment strategy is a map π : $\mathcal { H } _ { \mathrm { e } } \times A  \mathcal { P }$ . We write $\Sigma _ { \mathcal { M } }$ and $\pi _ { \mathcal { M } }$ for the sets of agent and environment strategies. A play is an infinite sequence $\rho = s _ { 0 } a _ { 0 } s _ { 1 } a _ { 1 } \cdot \cdot \cdot$ , and its reward sequence is $( r _ { t } ) _ { t \geq 0 }$ with $r _ { t } = r ( s _ { t } , a _ { t } , s _ { t + 1 } )$ . A strategy profile $( \sigma , \pi )$ induces a unique probability measure $\mathbb { P } _ { \mathcal { M } } ^ { \sigma , \dot { \pi } }$ on plays [41] under which a prefix $s _ { 0 } a _ { 0 } \cdots s _ { t }$ is extended by $a _ { t } s _ { t + 1 }$ with probability

$$
\sigma ( h _ { t } ^ { \mathsf { a } } ) ( a _ { t } ) \cdot \pi ( h _ { t } ^ { \mathsf { e } } ) ( s _ { t } , a _ { t } ) ( s _ { t + 1 } ) ,
$$

where $h _ { t } ^ { \mathsf { a } } = \mathcal { Z } _ { \mathsf { a } } ( s _ { 0 } ) \cdot \cdot \cdot \mathcal { Z } _ { \mathsf { a } } ( s _ { t } )$ is the history of the agent and $h _ { t } ^ { \mathsf { e } } = \mathcal { Z } _ { \mathsf { e } } ( s _ { 0 } ) \cdot \cdot \cdot \mathcal { Z } _ { \mathsf { e } } ( s _ { t } )$ is the history of the environment.

Without loss of generality, we adopt a standard assumption for RPOMDPs (and, later, for POSGs) that every action in the action alphabet of a decision maker is enabled at every state. We call an action effective at a state if the construction explicitly specifies its transition there, and an effective edge is a transition of positive probability under an effective action. In figures, we draw only effective actions and edges. Every omitted state-action pair is implicitly completed by a transition to an absorbing sink that is worst possible for the decision maker who chose the action.

## 2.2 Partially Observable Stochastic Games

POSGs. A turn-based partially observable stochastic game (POSG) between the players max and min is a tuple

$$
\mathcal { G } = ( S _ { \sf m a x } , S _ { \sf m i n } , A _ { \sf m a x } , A _ { \sf m i n } , \delta , s _ { 0 } , \mathcal { O } _ { \sf m a x } , \mathcal { O } _ { \sf m i n } , \mathcal { Z } _ { \sf m a x } , \mathcal { Z } _ { \sf m i n } , r )
$$

with disjoint finite sets $S _ { \mathrm { m a x } } , S _ { \mathrm { m i n } }$ of max- and min-states $( S = S _ { \sf m a x } \cup S _ { \sf m i n } )$ , an initial state $s _ { 0 } ~ \in$ $S _ { \sf m a x }$ , action sets $A _ { \sf m a x } , A _ { \sf m i n }$ with $A = A _ { \sf m a x } \cup A _ { \sf m i n } ,$ , a transition function $\delta : ( S _ { \sf m a x } \times A _ { \sf m a x } ) \cup$ $( S _ { \sf m i n } \times A _ { \sf m i n } ) \ :  \ : \Delta ( S )$ , observation functions $\mathcal { Z } _ { \sf m a x } \colon S \  \ \mathcal { O } _ { \sf m a x }$ and $\mathcal { Z } _ { \sf m i n } \colon S \  \ \mathcal { O } _ { \sf m i n } .$ , and a reward function $r \colon S \times ( A _ { \sf m a x } \cup A _ { \sf m i n } ) \times S  \mathbb { Q } .$ The game is alternating-control (an ac-POSG) if $\delta ( s , a ) \ \in \ \varDelta ( S _ { \mathsf { m i n } } )$ for all $s \in \ S _ { \mathsf { m a x } }$ and $\delta ( s , a ) \in \varDelta ( S _ { \mathsf { m a x } } )$ for all $s \in \ S _ { \operatorname* { m i n } }$ . It is fully observable (a turn-based stochastic game (TBSG)) if $\mathcal { Z } _ { \mathrm { m a x } }$ and ${ \mathcal { Z } } _ { \operatorname* { m i n } }$ are injective, i.e., every observation corresponds to a unique state.

A play is an infinite sequence $s _ { 0 } a _ { 0 } s _ { 1 } a _ { 1 } \cdots$ and its reward sequence is $( r _ { t } ) _ { t \geq 0 }$ with $r _ { t } = r ( s _ { t } , a _ { t } , s _ { t + 1 } )$ A history is a finite sequence $h _ { t } = s _ { 0 } , a _ { 0 } , s _ { 1 } , \cdot \cdot \cdot , s _ { t } \in ( S \times A ) ^ { \star }$ . A max-history of $h _ { t }$ is a sequence $h = o _ { 0 } ^ { 1 } \cdot \cdot \cdot o _ { t } ^ { 1 } \in \mathcal { O } _ { \mathsf { m a x } } \overline { { \cdot } }$ of observations of max such that $s _ { t }$ is a max-state and $o _ { k } ^ { 1 } = \mathcal { Z } _ { \operatorname* { m a x } } ( s _ { k } ) ,$ and let $H _ { m a x }$ denote the set of max histories. A max-strategy is a map $\sigma \colon H _ { m a x } \to \varDelta ( A _ { \mathsf { m a x } } )$ . A min-history of $h _ { t }$ is a sequence $h = o _ { 0 } ^ { 2 } \cdot \cdot \cdot o _ { t } ^ { 2 } \in \mathcal { O } _ { \sf m i n } { } ^ { + }$ of observations of min such that $s _ { t }$ is a min-state and $o _ { k } ^ { 2 } = \mathcal { Z } _ { \operatorname* { m i n } } ( s _ { k } )$ , and let $H _ { m i n }$ denote the set of min histories. A min-strategy is a map $\begin{array} { r } { \pi \colon H _ { m i n } \to \prod _ { s \in S _ { \mathsf { m i n } } } \varDelta ( A _ { \mathsf { m i n } } ) } \end{array}$ , which assigns to every min-history h a prescription $\pi ( h )$ , i.e., a probability distribution over actions $\pi ( h ) ( s ) \in \varDelta ( A _ { \sf m i n } )$ for every min-state s. Then, if the current state is $s ,$ min samples the next action according to $\pi ( h ) ( s )$ . We write $\Sigma _ { \mathcal { G } }$ and $\varPi _ { \mathcal { G } }$ for the sets of max- and min-strategies. A profile $( \sigma , \pi )$ induces a unique probability measure $\mathbb { P } _ { \mathcal { G } } ^ { \sigma , \pi }$ on plays under which a prefix $s _ { 0 } a _ { 0 } \cdots s _ { k }$ is extended by $a _ { k } s _ { k + 1 }$ with probability

$$
\left\{ \begin{array} { l l } { \sigma ( h _ { k } ^ { \operatorname* { m a x } } ) ( a _ { k } ) \cdot \delta ( s _ { k } , a _ { k } ) ( s _ { k + 1 } ) , } & { \mathrm { i f ~ } s _ { k } \in S _ { \operatorname* { m a x } } , } \\ { \pi ( h _ { k } ^ { \operatorname* { m i n } } ) ( s _ { k } ) ( a _ { k } ) \cdot \delta ( s _ { k } , a _ { k } ) ( s _ { k + 1 } ) , } & { \mathrm { i f ~ } s _ { k } \in S _ { \operatorname* { m i n } } , } \end{array} \right.
$$

where $h _ { k } ^ { \operatorname* { m a x } } = \mathcal { Z } _ { \operatorname* { m a x } } ( s _ { 0 } ) \cdot \cdot \cdot \mathcal { Z } _ { \operatorname* { m a x } } ( s _ { k } ) , h _ { k } ^ { \operatorname* { m i n } } = \mathcal { Z } _ { \operatorname* { m i n } } ( s _ { 0 } ) \cdot \cdot \cdot \mathcal { Z } _ { \operatorname* { m i n } } ( s _ { k } )$ are observation histories of max and min, respectively.

Remark 1 (Definition of min-strategies). We let a min-strategy assign a probability distribution over actions to every min-state, so that min may play differently in states with the same observation. This is done in order to match the RPOMDP game semantics of [6], where the environment chooses a probabilistic transition function for each state-action pair.

While this may appear to give additional power to the min-player as opposed to the classical POSGs [6, 8] in which min-strategies are of type $\pi \colon H _ { m i n } \to \varDelta ( A _ { \mathsf { m i n } } )$ , we show in Appendix A that a POSG with prescriptions as min-strategies can be compiled into an equivalent classical POSG with minstrategies of type $\pi \colon H _ { m i n } \to \varDelta ( A _ { \mathsf { m i n } } )$ , by a constant scaling of reward objective values.

## 2.3 Reward Objectives

We now define the reward objectives in RPOMDPs and POSGs that we consider in this work. Consider a play $\rho = s _ { 0 } a _ { 0 } s _ { 1 } a _ { 1 } \cdot \cdot \cdot$ in an RPOMDP or a POSG, and let $( r _ { t } ) _ { t \geq 0 }$ with $r _ { t } = r ( s _ { t } , a _ { t } , s _ { t + 1 } )$ be the reward sequence of the play. Note that, since our RPOMDPs and POSGs consist of finitely many states and actions, the reward function r is bounded and all rewards lie in some $[ - R , R ]$ . We define the following four reward objectives:

$$
\begin{array} { r l } & { \mathsf { T o t } _ { N } = \sum _ { t < N } r _ { t } , \qquad \quad \mathsf { T o t } = \sum _ { t \ge 0 } r _ { t } , } \\ & { \mathsf { D i s c } _ { \gamma } = \sum _ { t \ge 0 } \gamma ^ { t } r _ { t } , \qquad \quad \mathsf { L i m A v g } = \operatorname* { l i m i n f } _ { n \to \infty } \frac { 1 } { n } \sum _ { t < n } r _ { t } , } \end{array}
$$

where $N \in \mathbb { N }$ is a horizon and $\gamma \in \mathsf { \Gamma } ( 0 , 1 )$ is a rational discount factor. We refer to Tot<sub>N</sub> as the total reward until time N, Tot as the total reward, Disc<sub>γ</sub> as the discounted-sum reward, and LimAvg as the long-run average reward (or mean-payoff) of the play. The total reward Tot is restricted to nonnegative rewards in $[ 0 , R ]$ so that its value is well defined, whereas the other three reward objectives are defined for all rewards in $[ - R , R ]$

We note that for long-run average rewards, while our presentation considers the lim inf variant, our equivalence proof for RPOMDPs and POSGs is analogous for the lim sup variant and the same computational complexity results hold.

## 3 Law-Invariant Risk Measures

In this work, we study the problem of solving RPOMDPs under the reward objectives defined above, where the goal is to calculate the value of a risk measure of the reward objective. To that end, we first formally define a class of risk measures that we consider. We then define the problem of solving RPOMDPs (and, analogously, POSGs) under these risk measures ofthe above reward objectives.

Let Ω be the set of plays in an RPOMDP M or a POSG G, and let $X : \varOmega  \mathbb R$ be a reward objective which is either the total reward until time N $( X = \mathsf { T o t } _ { N } )$ , the total reward $( X = \mathsf { T o t } )$ the discounted-sum reward $( X = \mathsf { D i s c } _ { \gamma } )$ , or the long-run average reward $( X = \mathsf { L i m A v g } )$ . For a fixed strategy profile $( \sigma , \pi )$ , X induces a random variable defined by the reward objective value of a random play. We denote by $\mathcal { L } _ { X } ^ { \sigma , \pi }$ the law of X under $( \sigma , \pi )$ , i.e. the probability distribution over $\overline { { \mathbb { R } } } = [ - \infty , + \infty ]$ induced by the value of X.

Definition 1 (Law-invariant risk measure). Let M be an RPOMDP and X be a reward objective. A law-invariant risk measure is a function $\varPsi \colon \mathcal { D } ( \overline { { \mathbb { R } } } ) \to \overline { { \mathbb { R } } }$ which to every law over R assigns a value in R. The value of M with respect to X and Ψ is defined via

$$
{ \mathsf { V a l } } _ { \mathcal { M } } ( \varPsi , X ) = \operatorname* { s u p } _ { \sigma \in \Sigma _ { \mathcal { M } } } \operatorname* { i n f } _ { \pi \in \pi _ { \mathcal { M } } } \varPsi ( \mathcal { L } _ { X } ^ { \sigma , \pi } ) .
$$

For a POSG G and a reward objective X, the value $\mathsf { V a l } _ { \mathcal { G } } ( \varPsi , X )$ of $\mathcal { G }$ with respect to X and Ψ is defined analogously, with max maximizing and min minimizing.

Intuitively, law invariance means that the value of the reward objective under a risk measure only depends on the probability distribution of the reward objective value that a random play induces.

Thus, any two strategy profiles that induce the same probability distribution of reward objective values will have the same risk measure. This is thus the standard notion of law invariance of risk measures [27, 15], applied to the reward objective values.

Examples of law-invariant risk measures. Many risk measures commonly studied in Markov models depend only on the probability distribution of the reward objective value, and so are law invariant:

• Expected value. $\mathbb { E } \colon \mathcal { D } ( \overline { { \mathbb { R } } } )  \overline { { \mathbb { R } } }$ is defined via $\begin{array} { r } { \mathbb { E } ( \mu ) = \int x \mathrm { d } \mu ( x ) } \end{array}$ for all $\mu \in { \mathcal { D } } ( { \overline { { \mathbb { R } } } } )$

• Mean-variance. Given $\lambda \geq 0 , \mathsf { M V } _ { \lambda } \colon { \mathcal { D } } ( { \overline { { \mathbb { R } } } } ) $ R is defined via $\mathsf { M V } _ { \lambda } ( \mu ) = \mathbb E ( \mu ) - \lambda \mathsf { V a r } ( \mu )$ where $\begin{array} { r } { \mathsf { V a r } ( \mu ) = \int ( x - \mathbb { E } ( \mu ) ) ^ { 2 } \mathrm { d } \mu ( x ) } \end{array}$ is the variance, for all $\mu \in { \mathcal { D } } ( \mathbb { R } )$

• Conditional value-at-risk. Given $\alpha \in ( 0 , 1 ) , \mathsf { C V a R } _ { \alpha } \colon { \mathcal { D } } ( { \overline { { \mathbb { R } } } } ) \to$ R is defined via ${ \mathsf { C V a R } } _ { \alpha } ( \mu ) =$ $\begin{array} { r l } { \operatorname* { s u p } _ { \eta \in \mathbb { R } } \ S \eta - \frac { 1 } { \alpha } \int ( \eta - x ) ^ { + } \mathrm { d } \mu ( x ) \big \} } \end{array}$ , where $y ^ { + } = \operatorname* { m a x } \{ y , 0 \}$ , for all $\mu \in { \mathcal { D } } ( { \overline { { \mathbb { R } } } } )$ .

• Entropic risk. Given $\theta \neq 0 , \mathsf { E R i s k } _ { \theta } \colon { \mathcal { D } } ( { \overline { { \mathbb { R } } } } ) \to { \overline { { \mathbb { R } } } }$ is defined via $\begin{array} { r } { \mathsf { E R i s k } _ { \theta } ( \mu ) = \frac { 1 } { \theta } \log \int e ^ { \theta x } \mathrm { d } \mu ( x ) \} } \end{array}$ for all $\mu \in { \mathcal { D } } ( { \overline { { \mathbb { R } } } } )$ .

All four risk measures are defined in terms of integrals of different functions with respect to the probability distribution $\mu .$ Hence, they are all of type $\varPsi \colon \mathcal { D } ( \overline { { \mathbb { R } } } ) \to \overline { { \mathbb { R } } }$ , and thus are law invariant. While most works on solving R(PO)MDPs focus on the expected value E, we consider E as well as any other law-invariant risk measure.

Remark 2. For laws $\mu$ that may have unbounded support, such as law of the total reward objective Tot, we use the same formulas for $\mathbb { E } , \mathsf { C V a R } _ { \alpha }$ , and $\mathsf { E R i s k } _ { \theta }$ with $\theta < 0$ , with the conventions $e ^ { - \infty } = 0$ and log $0 = - \infty$ (so that values may be +∞), and we use ${ \mathsf { M V } } _ { \lambda }$ only for returns with finite second moments.

Threshold problems. We now define the threshold problems for RPOMDPs and POSGs with respect to reward objectives and law-invariant risk measures that we consider in this work.

Threshold problems

Given a model M (an RPOMDP or a POSG), a reward objective X, a law-invariant risk measure Ψ (with rational parameters), and a rational threshold c, we consider the following:

• Threshold problem: Decide if $\mathsf { V a l } _ { \mathcal { M } } ( \varPsi , X ) \geq c .$

• Strict threshold problem: Decide $\mathrm { i f } \mathsf { V a l } _ { \mathcal { M } } ( \varPsi , X ) > c .$

## 4 Equivalence of RPOMDPs and POSGs

We now show that RPOMDPs and ac-POSGs are equivalent under the four reward objectives defined above, and any law-invariant risk measure. We prove these by constructing reductions in two directions, one from RPOMDPs to ac-POSGs (theforward construction) and one from ac-POSGs to RPOMDPs (the backward construction). The key property of both reductions is law preservation: each strategy in one model is mapped to a strategy in the other model that induces the same law of the reward objective value, up to rescaling by a constant factor which depends on the reward objective. All proofs are deferred to the Appendix, see the Overview of the Appendix for the pointer to each proof.

![](images/a3f25799400283bdd869e7ab7a4d43724f9ba5aa43eaa86278c4f34be39606a7.jpg)  
Figure 1: The forward construction. (a) An RPOMDP with states $s _ { 1 } , s _ { 2 } , s _ { 3 } ;$ action b is available only in $s _ { 1 }$ . The drawn distributions are generic points of the uncertainty polytopes, $\mathrm { e . g . , } \left( p _ { 1 } , q _ { 1 } , r _ { 1 } \right) \in$ $\mathcal { P } ( s _ { 1 } , a ) = \mathrm { c o n v } \{ v _ { 1 1 } , v _ { 1 2 } , v _ { 1 3 } \} , ( 0 , q _ { 2 } , r _ { 2 } ) \in \mathcal { P } ( s _ { 2 } , a ) = \mathrm { c o n v } \{ v _ { 2 1 } , v _ { 2 2 } \}$ , and $\mathcal { P } ( s _ { 3 } , a ) = \{ v _ { 3 1 } \}$ with $v _ { 3 1 } = ( 0 , q _ { 3 } , r _ { 3 } )$ . (b) The ac-POSG ${ \mathfrak { G } } ( { \mathcal { M } } )$ . Circles are max-states and rectangles are minstates $( s , a )$ , whose actions are the vertices of ${ \mathcal { P } } ( s , a )$ . By randomizing over $v _ { 1 1 } , v _ { 1 2 } , v _ { 1 3 }$ with weights $( \alpha _ { 1 } , \alpha _ { 2 } , \alpha _ { 3 } )$ , min realizes the point $\alpha _ { 1 } v _ { 1 1 } + \alpha _ { 2 } v _ { 1 2 } + \alpha _ { 3 } v _ { 1 3 }$ of $\mathcal { P } ( s _ { 1 } , a )$ . For readability, dashed outcome edges are drawn for one vertex per min-state only.

## 4.1 From RPOMDPs to ac-POSGs

The key idea behind the forward reduction is as follows. Each step of an RPOMDP is split into two steps of the POSG: first the max player chooses the agent’s action, then the min player chooses the environment’s distribution. Since $\mathcal { P } ( s , a )$ is a polytope, choosing a point in $\mathcal { P } ( s , a )$ amounts to randomizing over its vertices, so the actions of the min player are the vertices of $\mathcal { P } ( s , a )$ . The following definition formalizes this construction.

Definition 2 (Forward construction). Let

$$
\mathcal { M } = ( S , A , \mathcal { P } , s _ { 0 } , \mathcal { O } _ { a } , \mathcal { O } _ { \mathrm { e } } , \mathcal { Z } _ { \mathrm { a } } , \mathcal { Z } _ { \mathrm { e } } , r )
$$

be an RPOMDP with $\mathcal { P } ( s , a ) = \operatorname { c o n v } ( V _ { s , a } )$ for all $s \in S$ and $a \in A$ . We define an ac-POSG ${ \mathfrak { G } } ( { \mathcal { M } } )$ as a tuple

$$
( S _ { \operatorname* { m a x } } , S _ { \operatorname* { m i n } } , A _ { \operatorname* { m a x } } , A _ { \operatorname* { m i n } } , \delta , s _ { 0 } , \mathcal { O } _ { \operatorname* { m a x } } , \mathcal { O } _ { \operatorname* { m i n } } , \mathcal { Z } _ { \operatorname* { m a x } } , \mathcal { Z } _ { \operatorname* { m i n } } , r ^ { \prime } )
$$

with the following components:

• States and actions. The max-states and actions are the states and actions of M, i.e., $S _ { \sf m a x } = S$ and $A _ { \sf m a x } = A$ . The min-states are the state-action pairs $( s , a )$ with $s \in S$ and $a \in A .$ , and min-actions are vertices of uncertainty polytopes $V _ { s , a } ;$ thus $\begin{array} { r } { A _ { \operatorname* { m i n } } = \bigcup _ { s , a } V _ { s , a } } \end{array}$ . The initial state is $s _ { 0 }$

• Transitions. For each $s \in S _ { \mathsf { m a x } }$ and $a \in A _ { \sf m a x }$ , we have $\delta ( s , a ) = { \sf D i r a c } ( ( s , a ) )$ . For each $( s , a ) \in S _ { \operatorname* { m i n } }$ and $v \in V _ { s , a }$ , we have $\delta ( ( s , a ) , v ) = v$

• Observations. $\mathcal { O } _ { \sf m a x } = \mathcal { O } _ { a }$ and $\mathcal { O } _ { \sf m i n } = \mathcal { O } _ { \sf e }$ , with $\mathcal { Z } _ { \sf m a x } ( s ) = \mathcal { Z } _ { \sf m a x } ( ( s , a ) ) = \mathcal { Z } _ { \sf a } ( s )$ and $\mathcal { Z } _ { \sf m i n } ( s ) = \mathcal { Z } _ { \sf m i n } ( ( s , a ) ) = \mathcal { Z } _ { \sf e } ( s )$

• Rewards. r<sup>′</sup>(s, a, (s, a)) = 0 and r<sup>′</sup>((s, a), v, s<sup>′</sup>) = r(s, a, s<sup>′</sup>).

In both the max-state s and the min-state $( s , a )$ , both players receive the same observations as the agent and the environment receive in state s in the RPOMDP M, respectively. Strategies of the two models correspond as follows:

• An agent strategy σ in the RPOMDP becomes a max-strategy that plays $\sigma ( o _ { 0 } o _ { 1 } \cdot \cdot \cdot o _ { t } )$ on observation history $O _ { 0 } O _ { 0 } O _ { 1 } O _ { 1 } \cdot \cdot \cdot \ O _ { t - 1 } O _ { t - 1 } O _ { t }$ (in ${ \mathfrak { G } } ( { \mathcal { M } } )$ , the max-observations in max- and min-states are the same so each observation appears twice).

• An environment strategy π becomes a min-strategy as follows. In RPOMDP M, after the environment history $h \ = \ o _ { 0 } \cdots o _ { t }$ , the environment selects for every pair $( s , a )$ a point $p = \pi ( h ) ( s , a )$ of ${ \mathcal { P } } ( s , a ) = \operatorname { c o n v } ( V _ { s , a } )$ . We write this point as a convex combination $\begin{array} { r } { p = \sum _ { v \in V _ { s , a } } \lambda _ { v } } \end{array}$ v of the vertices. Then, in the POSG ${ \mathfrak { G } } ( { \mathcal { M } } )$ , we let min, after the min-history $O _ { 0 } O _ { 0 } O _ { 1 } O _ { 1 } \cdot \cdot \cdot O _ { t } O _ { t }$ , play in the min-state $( s , a )$ the mixed action that chooses each vertex v with probability $\lambda _ { v }$ . The next state is sampled from $\begin{array} { r } { \sum _ { v } \lambda _ { v } v = p , } \end{array}$ exactly as in $\mathcal { M }$

The following lemma shows that this strategy mapping is law preserving, up to rescaling by a constant factor. Recall the definition of $( S _ { c } ) _ { \# }$ from Section 2.

Lemma 1 (Forward law preservation). Let ${ \mathcal G } = { \mathfrak G } ( { \mathcal M } )$ , and associate with each reward objective X of M a reward objective $X ^ { \mathcal { G } }$ of G and a constant $c > 0$ as follows:

$$
\begin{array} { c c c c c } { X : } & { \mathsf { T o t } _ { N } } & { \mathsf { T o t } } & { \mathsf { L i m A v g } } & { \mathsf { D i s c } _ { \beta ^ { 2 } } } \\ { X ^ { \mathcal { G } } : } & { \mathsf { T o t } _ { 2 N } } & { \mathsf { T o t } } & { \mathsf { L i m A v g } } & { \mathsf { D i s c } _ { \beta } } \\ { c : } & { 1 } & { 1 } & { 1 / 2 } & { \beta } \end{array}
$$

where $\beta \in ( 0 , 1 )$ is assumed to be rational. Then, there is a bijection $\sigma \mapsto \sigma ^ { \mathcal { G } }$ from agent strategies of M to max-strategies of G such that, for every agent strategy σ,

$$
\begin{array} { r } { \big \{ \mathcal L _ { X ^ { \mathcal G } } ^ { \sigma ^ { \mathcal G } , \pi ^ { \prime } } : \pi ^ { \prime } \in \pi _ { \mathcal G } \big \} = \big \{ ( S _ { c } ) _ { \# } \mathcal L _ { X } ^ { \sigma , \pi } : \pi \in \pi _ { \mathcal M } \big \} . } \end{array}
$$

More precisely, each min-strategy $\pi ^ { \prime }$ induces an environment strategy π with $\mathcal { L } _ { X ^ { \mathcal { G } } } ^ { \sigma ^ { \mathcal { G } } , \pi ^ { \prime } } = ( S _ { c } ) _ { \# } \mathcal { L } _ { X } ^ { \sigma , \pi }$ for all σ.

The idea of the proof is that erasing the min-states and vertex choices from a play of G yields a play of $\mathcal { M } ,$ and that under corresponding strategies this projection maps $\mathbb { P } _ { \mathcal { G } } ^ { \sigma ^ { \mathcal { G } } , \pi ^ { \prime } }$ to $\mathbb { P } _ { \mathcal { M } } ^ { \sigma , \pi }$ ; the reward sequence of the game play is that of the projected play, with a zero inserted before every reward. The constants c then follow from a direct computation, e.g., $\begin{array} { r } { \sum _ { i } \beta ^ { 2 i + 1 } r _ { i } = \beta \sum _ { i } ( \beta ^ { 2 } ) ^ { i } r _ { i } } \end{array}$ for the discounted return. Thus the law of the return is preserved up to the factor c. As c depends only on the return (for $\mathsf { D i s c } _ { \beta ^ { 2 } } , \mathsf { o n } \beta )$ , and neither on the strategies nor on the law functional, the robust values agree up to the same rescaling (Theorem 1).

## 4.2 From ac-POSGs to RPOMDPs

In the other direction, the environment in the RPOMDP takes the role of the min-player in the ac-POSG. Two points need care. First, in the ac-POSG, min moves in a state t of its own, after observing $\mathcal { Z } _ { \mathrm { m i n } } ( t )$ . In contrast, in the RPOMDP, the environment moves in the same step as the agent, with no intermediate state, and hence no observation, in between. To address this discrepancy, in our reduction we turn every min-state t into a state $d _ { t }$ of the RPOMDP in which the agent has a single dummy action $\top$ and the environment, after observing $\mathcal { Z } _ { \mathrm { m i n } } ( t )$ , chooses the next distribution from the polytope spanned by min’s options. Second, the reward of a min-transition may depend on min’s action, whereas the reward of an RPOMDP transition depends only on the agent state, the agent’s action, and the transition’s target state. We therefore extend every max-state s to a pair $( s , b )$ whose tag b records the min action that led to $s ,$ so that the transition from $d _ { t }$ to $( s ^ { \prime } , b )$ can earn the reward $r ( t , b , s ^ { \prime } )$ . Neither player observes the tags.

Definition 3 (Backward construction). Let

$$
\mathcal { G } = ( S _ { \sf m a x } , S _ { \sf m i n } , A _ { \sf m a x } , A _ { \sf m i n } , \delta , s _ { 0 } , \mathcal { O } _ { \sf m a x } , \mathcal { O } _ { \sf m i n } , \mathcal { Z } _ { \sf m a x } , \mathcal { Z } _ { \sf m i n } , r )
$$

be an ac-POSG, and let ⊤ $\notin A _ { \sf m a x } \cup A _ { \sf m i n }$ be a fresh symbol. The RPOMDP $\Re ( \mathcal { G } )$ is the tuple

$$
( S ^ { \prime } , A ^ { \prime } , \mathcal { P } , ( s _ { 0 } , \top ) , \mathcal { O } _ { \operatorname* { m a x } } \times \{ 0 , 1 \} , \mathcal { O } _ { \operatorname* { m i n } } , \mathcal { Z } _ { { \mathsf { a } } } , \mathcal { Z } _ { { \mathsf { e } } } , r ^ { \prime } )
$$

with the following components.

• States. $S ^ { \prime }$ consists of initial state $( s _ { 0 } , \top )$ , a state $d _ { t }$ for every $t \in S _ { \operatorname* { m i n } }$ and of the tagged states $( s , b )$ for $s \in S _ { \sf m a x }$ and $b \in A _ { \operatorname* { m i n } }$ , where the tag b is the min action that led to s.

• Actions. We set $A ^ { \prime } = A _ { \sf m a x } \cup \{ \top \}$ . For each tagged state $( s , b )$ , the set of effective actions is A, and for each state $d _ { t }$ , the only effective action is ⊤.

• Uncertainty. In tagged states, the environment has no choice:

$$
\mathcal { P } ( ( s _ { 0 } , \top ) , a ) = \{ \hat { \delta } ( s _ { 0 } , a ) \} , \qquad \hat { \delta } ( s _ { 0 } , a ) ( d _ { t } ) = \delta ( s _ { 0 } , a ) ( t )
$$

$$
\begin{array} { r } { \mathcal { P } ( ( s , b ) , a ) = \{ \hat { \delta } ( s , a ) \} , \qquad \hat { \delta } ( s , a ) ( d _ { t } ) = \delta ( s , a ) ( t ) . } \end{array}
$$

In the states $d _ { t } ,$ it chooses from the polytope

$$
\mathcal { P } ( d _ { t } , \top ) = \mathrm { c o n v } \{ \delta _ { t , b } ^ { \prime } : b \in A ( t ) \} ,
$$

where $\delta _ { t , b } ^ { \prime } ( ( s ^ { \prime } , b ^ { \prime } ) ) = \delta ( t , b ) ( s ^ { \prime } )$ if $\boldsymbol { b } ^ { \prime } = \boldsymbol { b }$ and 0 otherwise.

• Observations. For the agent, $\mathcal { Z } _ { \sf d } ( ( s , b ) ) = ( \mathcal { Z } _ { \sf m a x } ( s ) , 0 )$ and $\mathcal { Z } _ { \sf a } ( d _ { t } ) = ( \mathcal { Z } _ { \sf m a x } ( t ) , 1 )$ ; for the environment, $\mathcal { Z } _ { \sf e } ( ( s , b ) ) = \mathcal { Z } _ { \sf m i n } ( s )$ and $\mathcal { Z } _ { \sf e } ( d _ { t } ) = \mathcal { Z } _ { \sf m i n } ( t )$

• Rewards. $r ^ { \prime } ( ( s , b ) , a , d _ { t } ) = r ( s , a , t )$ and $r ^ { \prime } ( d _ { t } , \top , ( s ^ { \prime } , b ) ) = r ( t , b , s ^ { \prime } )$ ; all other rewards are 0.

Since the distributions $\delta _ { t , b } ^ { \prime }$ have pairwise disjoint supports, a point of $\mathcal { P } ( d _ { t } , \top )$ determines the mixed action of min that it represents. The additional bit in the agent’s observations tells the agent whose turn it is, so that it can distinguish the tagged max-states $( s , b )$ , where it chooses among the effective actions $A ( s )$ , from the states $d _ { t }$ , where it has only the effective action ⊤. The bit gives the agent no new information about ${ \mathcal { G } } .$ , because turns alternate: it is 0 after an odd number of observations and 1 after an even number.

Lemma 2 (Backward law preservation). Let ${ \mathcal { M } } = \Re ( { \mathcal { G } } )$ . There is a bijection between the plays of G and those of M under which corresponding plays have identical reward sequences, and a bijection $\sigma \mapsto \sigma ^ { \mathcal { M } }$ from max-strategies to agent strategies such that, for every max-strategy σ and every reward objective X (with the same horizon or discountfactor on both sides),

$$
\{ \mathcal { L } _ { X } ^ { \sigma ^ { \mathcal { M } } , \pi ^ { \prime } } : \pi ^ { \prime } \in \pi _ { \mathcal { M } } \} = \{ \mathcal { L } _ { X } ^ { \sigma , \pi } : \pi \in \Pi \varsigma \} .
$$

Since both constructions preserve achievable return laws, they preserve the value of every law functional which is our main result.

## 4.3 The umbrella theorem

Theorem 1 (Umbrella theorem). Let $\varPsi \colon \mathcal { D } ( \overline { { \mathbb { R } } } ) \to \overline { { \mathbb { R } } }$ be a law-invariant risk measure.

(a) For every RPOMDP M and reward objective X, with $X ^ { \mathcal { G } }$ and c as in Lemma 1,

$$
\mathsf { V a l } _ { \mathfrak { G } ( \mathcal { M } ) } ( \varPsi , X ^ { \mathcal { G } } ) = \mathsf { V a l } _ { \mathcal { M } } ( \varPsi \circ ( S _ { c } ) _ { \# } , X ) .
$$

(b) For every ac-POSG G and reward objective X,

$$
\mathsf { V a l } _ { \mathfrak { R } ( \mathcal { G } ) } ( \varPsi , X ) = \mathsf { V a l } _ { \mathcal { G } } ( \varPsi , X ) .
$$

![](images/d497bc8c373bbbc0a99aa2875425f9d2a996c13c6f3c5307fdcbaf929a2448e3.jpg)  
Figure 2: The backward construction. (a) An ac-POSG G with max-states $s _ { 1 } , s _ { 2 }$ (circles) and minstates $t _ { 1 } , t _ { 2 }$ (rectangles); min can play x in $t _ { 1 }$ and x or y in t . (b) The RPOMDP $\Re ( \mathcal { G } )$ ; dotted boxes are uncertainty sets. The sets $U _ { 3 } = \mathcal { P } ( ( s _ { 1 } , \tau ) , a ) , U _ { 4 } = \mathcal { P } ( ( s _ { 1 } , \tau ) , b )$ , and $U _ { 5 } = \mathcal { P } ( ( s _ { 2 } , \tau ) , a )$ are the singletons $\{ \hat { \delta } ( s _ { 1 } , a ) \} , \{ \hat { \delta } ( s _ { 1 } , b ) \}$ , and $\{ \hat { \delta } ( s _ { 2 } , a ) \}$ for every tag τ , and $U _ { 2 } = \mathcal { P } ( d _ { t _ { 1 } } , \top ) = \{ \delta _ { t _ { 1 } , x } ^ { \prime } \}$ is a singleton as well. The set $U _ { 1 } = \mathcal { P } ( d _ { t _ { 2 } } , \top ) = \mathrm { c o n v } \{ \delta _ { t _ { 2 } , x } ^ { \prime } , \delta _ { t _ { 2 } , y } ^ { \prime } \}$ is a segment: choosing a point on it is choosing a mixed action of min in $t _ { 2 } .$ , and the tag of the next state records which action was realized.

## (c) Both constructions are computable in logarithmic space.

Consequently, for every law-invariant risk measure and each of the four reward objectives, the threshold problems (strict and non-strict) for ac-POSGs reduce in logarithmic space to those for RPOMDPs with the same risk measure and reward objective. Conversely, the threshold problems for Ψ and X on RPOMDPs reduce in logarithmic space to those for Ψ $\mathrm { ~ o ~ } ( S _ { 1 / c } ) _ { \# }$ and $X ^ { \mathcal { G } }$ on ac-POSGs, with the same threshold. For instance, $c = 1$ for finite-horizon total return, hence the risk measure is unchanged. Likewise, for discounted-sum rewards, the direction applies to RPOMDPs whose discount factor is the square $\beta ^ { 2 }$ of a rational $\beta ,$ and the resulting ac-POSG has discount factor $\beta$ with $c = \beta .$

Proof. For a model and a strategy σ of the maximizing player, let $\mathbb { L } ( \sigma )$ be the set of laws of the reward objective that the opponent can induce against σ; then $\mathsf { V a l } ( \varPsi , X ) = \mathsf { s u p } _ { \sigma } \operatorname* { i n f } _ { \mu \in \mathbb { L } ( \sigma ) } \varPsi ( \mu )$ By Lemma $1 , \sigma \mapsto \sigma ^ { \mathcal { G } }$ is a bijection with $\mathbb { L } _ { \mathcal { G } } ( \sigma ^ { \mathcal { G } } ) = ( S _ { c } ) _ { \# } \mathbb { L } _ { \mathcal { M } } ( \sigma )$ , so

$$
\operatorname* { i n f } _ { \mu \in \mathbb { L } _ { \mathcal { G } } ( \sigma ^ { \mathcal { G } } ) } \varPsi ( \mu ) = \operatorname* { i n f } _ { \mu \in \mathbb { L } _ { \mathcal { M } } ( \sigma ) } \varPsi ( ( S _ { c } ) _ { \# } \mu ) ,
$$

and taking the supremum over σ yields (a). Part (b) follows in the same way from Lemma 2. Part (c) is shown in the appendix. For the threshold problems, (b) gives $\mathsf { V a l } _ { \mathcal { G } } ( \varPsi , X ) = \mathsf { V a l } _ { \mathfrak { R } ( \mathcal { G } ) } ( \varPsi , X )$ Moreover, (a), applied to the law functional $\varPsi \circ ( S _ { 1 / c } ) _ { \# }$ , gives

$$
\mathsf { V a l } _ { \mathcal { M } } ( \varPsi , X ) = \mathsf { V a l } _ { \mathfrak { G } ( \mathcal { M } ) } ( \varPsi \circ ( S _ { 1 / c } ) _ { \# } , X ^ { \mathcal { G } } ) ,
$$

because $( S _ { 1 / c } ) _ { \# } ( S _ { c } ) _ { \# } \mu = \mu$ for every law $\mu .$ In both cases, the threshold is unchanged. A detailed proof of all three parts can be found in the appendix. □

Corollary 1. In the setting of Theorem $I ( a ) ,$ let $\mathcal { G } = \mathfrak { G } ( \mathcal { M } )$ . For all $\lambda \ge 0 , \alpha \in ( 0 , 1 )$ , and $\theta \neq 0 ,$

$$
\mathsf { V a l } _ { \mathcal { G } } ( \mathbb { E } , X ^ { \mathcal { G } } ) = c \mathsf { V a l } _ { \mathcal { M } } ( \mathbb { E } , X ) ,
$$

$$
\mathsf { V a l } _ { \mathcal { G } } ( \mathsf { M V } _ { \lambda } , X ^ { \mathcal { G } } ) = c \mathsf { V a l } _ { \mathcal { M } } ( \mathsf { M V } _ { c \lambda } , X ) ,
$$

$$
\mathsf { V a l } _ { \mathcal { G } } ( \mathsf { C V a R } _ { \alpha } , X ^ { \mathcal { G } } ) = c \mathsf { V a l } _ { \mathcal { M } } ( \mathsf { C V a R } _ { \alpha } , X ) ,
$$

$$
\mathsf { V a l } _ { \mathcal { G } } ( \mathsf { E R i s k } _ { \theta } , X ^ { \mathcal { G } } ) = c \mathsf { V a l } _ { \mathcal { M } } ( \mathsf { E R i s k } _ { c \theta } , X ) .
$$

For the backward construction, all values coincide with unchanged parameters.

Table 1: Partial observation: we obtain lower bounds for POMDPs which apply also to RPOMDPs and ac-POSGs; upper bounds (decidability and approximation) hold for RPOMDPs and ac-POSGs. “undec.” means undecidable for both threshold variants, “undec.<sup>∗</sup>” for the strict variant only, “approx.” that values can be approximated to any precision. Full observation: polytopic (s, a)- rectangular RMDPs; the RMDP lower bounds also hold for TBSGs. <sup>‡</sup>Under polynomial-time Turing reductions. New results are in green.
<table><tr><td>E</td><td>Objective</td><td>Finite horizon (Thm. 2, Prop. 1)</td><td>Total Tot (Thm. 3)</td><td>Average LimAvg (Thm. 3)</td><td>Discounted Discγy (Thm. 4, Prop. 1)</td></tr><tr><td>Part tobs.</td><td>MVλ  $\mathsf { C V a R } _ { \alpha }$  ERiskθ</td><td>PSPACE-complete [38], decidable PSPACE-hard, decidable PSPACE-hard, decidable PSPACE-hard, decidable</td><td>undec.* [29] undec. undec. undec.</td><td>undec.* [29] undec. undec. undec.</td><td>undec.* [29]; approx. open for λ &gt; 0; approx. undec.*; approx. open; approx.</td></tr><tr><td>u b s.</td><td>Objective (return) E (discounted) E (total, average) MVλ (total, average) CVaRα (total, average)</td><td>MDP P [41] P [41] total: EXPTIME-hard, in NEXPTIME [40] average, non-strict: P [26]</td><td>SSG-hard‡ (Thm. 6), in NP ∩ coNP [9] SSG-hard (Thm. 5); average: in NP ∩ coNP [9]</td><td>RMDP SSG-hard (Thm. 5)</td><td></td></tr></table>

## 5 The Complexity Landscape

By Theorem 1, every complexity bound for the threshold problem for each of the reward objectives and any law-invariant risk measure transfers between RPOMDPs and ac-POSGs (for average and discounted returns, with the rescaling by the constant c of Lemma 1). We now use this result to derive new complexity bounds for R(PO)MDPs and ac-POSGs under reward objectives; Table 1 summarizes them. All numbers in the input (probabilities, rewards, discount factors, thresholds, and parameters) are rationals encoded in binary, so that every instance has a finite description. Our lower bounds share one proof pattern, which we first isolate in terms of the law-invariant risk measure Ψ in Section 5.1.

## 5.1 Bernoulli profiles

Our hardness results are reductions from decision problems about the probability of an event in a simpler model: whether a POMDP strategy avoids a bad event with probability 1 over a finite horizon [38] (Theorem $2 ) ;$ whether some POMDP strategy reaches a target with probability above b or all strategies stay below a, a gap problem derived from probabilistic automata [12] (Theorem 3); and whether a simple stochastic game reaches its target with probability above <sup>1</sup><sub>2</sub> [11] (Theorem 5). In each reduction we turn the instance into a reward model whose return equals $v > 0$ if the event occurs and 0 otherwise. For the finite-horizon total reward we pay v once at the last step if the event has occurred; for the total reward we pay v once when the absorbing target is entered; for the long-run average we pay v in every step spent in the target. This does not work for discounted returns, where the return also depends on when the event occurs, so discounted-sum rewards are treated separately.

A strategy under which the event has probability p then yields the return law $\mathsf { B e r } _ { v } ( p ) \ = \ ( 1 \ -$ $p ) { \mathsf { D i r a c } } ( 0 ) + p { \mathsf { D i r a c } } ( v )$ for $p \in [ 0 , 1 ]$ . A law-invariant risk measure Ψ sees only this law, so the value of the strategy is $\varPsi ( \mathsf { B e r } _ { v } ( p ) )$ , a function of p alone. If this function is nondecreasing in $p ,$ the best strategy is the one that maximizes p. A threshold for Ψ then becomes a threshold on the optimal probability, which is the question the source problem asks. We therefore only need to know how $\varPsi$ behaves on the laws $\mathsf { B e r } _ { v } ( p )$ , which motivates the following notion.

The Bernoulli profile of a law-invariant risk measure Ψ is the function $\beta _ { v } ^ { \varPsi } ( p ) \ = \ \varPsi ( \mathsf { B e r } _ { v } ( p ) )$ for $v > 0 { \mathrm { ~ a n d } } p \in [ 0 , 1 ]$ . For the four risk measures that we consider (computations from the definitions

in Section 3 are provided in Appendix C.2),

$$
\begin{array} { c } { \beta _ { v } ^ { \mathbb { E } } ( p ) = p v \qquad \beta _ { v } ^ { \mathrm { C V a R } _ { \alpha } } ( p ) = v \operatorname* { m a x } \bigl \{ 0 , 1 - ( 1 - p ) / \alpha \bigr \} } \\ { \beta _ { v } ^ { \mathrm { M V } _ { \lambda } } ( p ) = p v - \lambda v ^ { 2 } p ( 1 - p ) \qquad \beta _ { v } ^ { \mathrm { E R i s k } _ { \theta } } ( p ) = \frac { 1 } { \theta } \log \bigl ( 1 - p + p e ^ { \theta v } \bigr ) } \end{array}
$$

Definition 4 (Monotone and strict at the top Bernoulli profiles). A law-invariant risk measure $\varPsi$ has a monotone Bernoulli profile if, for some rational $v > 0 , \beta _ { v } ^ { \varPsi }$ is nondecreasing on [0, 1] and not constant on $( 0 , 1 )$ . The profile is strict at the top if, in addition, $\beta _ { v } ^ { \varPsi } ( p ) < \beta _ { v } ^ { \varPsi } ( 1 )$ for all $p < 1$ and $\beta _ { v } ^ { \varPsi } ( 1 )$ is rational.

The expectation, $\mathsf { C V a R } _ { \alpha }$ , and ERisk<sub>θ</sub> have monotone Bernoulli profiles that are strict at the top for every $v > 0 . \mathsf { M V } _ { \lambda }$ has monotone Bernoulli profiles that is strict at the top for $v = 1 / ( 1 + \lambda )$ , since then λv < 1.(Appendix C.2)

## 5.2 Partial observation

Lower bounds. Lower bounds for POMDPs apply to RPOMDPs directly, since a POMDP is an RPOMDP with singleton uncertainty sets, and they apply to ac-POSGs by our equivalence in Theorem 1.

Theorem 2 (Finite horizon total reward). If the Bernoulli profile of a law-invariant risk measure Ψ is strict at the top, then the finite-horizon total reward threshold problem for Ψ (both strict and non-strict) is PSPACE-hard, already for POMDPs; hence it is PSPACE-hard for RPOMDPs and ac-POSGs.

Proofsketch. [38] show that it is PSPACE-hard to decide whether a POMDP with nonnegative costs has a strategy of expected total cost 0 over the first $N \leq | S |$ steps, i.e., one that avoids all pairs with positive cost with probability 1. Their result concerns the expected cost, and it is a question about the probability $p ^ { \sigma }$ of avoiding the costly pairs. We show that, for every law-invariant risk measure Ψ whose Bernoulli profile is strict at the top, the threshold problem for $\varPsi$ encodes exactly this question. To this end, we add a step counter and collect reward v exactly when step N is completed without a costly pair, so that the total return has law $\mathsf { B e r } _ { v } ( p ^ { \sigma } )$ , where $p ^ { \sigma }$ is the success probability of σ. By compactness, $\operatorname* { m a x } _ { \sigma } p ^ { \sigma }$ exists, so the value is $\beta _ { v } ^ { \varPsi } ( \operatorname* { m a x } _ { \sigma } p ^ { \sigma } )$ . Strictness at the top gives $\beta _ { v } ^ { \varPsi } ( p ) \ < \ \beta _ { v } ^ { \varPsi } ( 1 )$ for $p \ < \ 1$ , so the value is at least the rational threshold $\beta _ { v } ^ { \varPsi } ( 1 )$ if and only if some strategy succeeds with probability 1. Hence the threshold problem for $\varPsi$ is PSPACE-hard for POMDPs, by a reduction computable in logarithmic space. The result transfers to RPOMDPs, of which POMDPs are a special case, and to ac-POSGs by the forward construction of Section 4, which is computable in logarithmic space and preserves the laws of the finite-horizon total return (with the horizon doubled and no rescaling of the rewards). Thus, the hardness of [38] for expected cost is lifted to all risk objectives with a strict Bernoulli profile, and to both RPOMDPs and ac-POSGs.

For infinite horizon total reward and long-run average reward, we use the undecidability of probabilistic automata with a gap [12]. Lemma C.2 in Appendix C.4 translates the result to POMDPs: for rationals $0 < a < b < 1$ , one computes from a two-counter machine a POMDP with a single observation and an absorbing target $T$ such that some strategy reaches $T$ with probability greater than b if the machine halts, while every strategy, randomized or not, reaches $T$ with probability at most a otherwise.

Theorem 3 (Total and long-run average reward). Ifa law-invariant risk measure Ψ has a monotone Bernoulli profile, then there is a rational c such that, for $X \in \{ \mathsf { T o t } , \mathsf { L i m A v g } \}$ , both $\mathsf { V a l } ( \varPsi , X ) \geq c$ and ${ \mathsf { V a l } } ( \varPsi , X ) > c$ are undecidable for POMDPs with a single observation; hence the threshold problemsfor Ψ are undecidablefor RPOMDPs and ac-POSGs.

Proofsketch. Choose rationals $0 { < } a { < } b { < } 1$ with $\beta _ { v } ^ { \varPsi } ( a ) < \beta _ { v } ^ { \varPsi } ( b )$ and a rational c strictly between these values; they depend only on $\varPsi .$ Given an instance of the probabilistic automaton gap problem for rationals $a$ and $b ,$ in our proof we translate the automaton into a POMDP which collects reward v upon entering $T$ (for Tot), or in every step spent in $T$ (for LimAvg). Then, the value of the reward objective in the POMDP under strategy σ has law $\mathsf { B e r } _ { v } ( p ^ { \sigma } )$ , where $p ^ { \sigma }$ is the probability of reaching $T$ under $\sigma .$ . If the machine halts, some σ has $p ^ { \sigma } > b ,$ , so the value is at least $\beta _ { v } ^ { \varPsi } ( b ) > c .$ Otherwise $\boldsymbol { p } ^ { \sigma } \le a$ for all $\sigma ,$ so the value is at most $\beta _ { v } ^ { \varPsi } ( a ) \ < \ c$ . Hence both threshold problems with the fixed threshold c reduce from the halting problem of two-counter machines. The result transfers to RPOMDPs and by our equivalence result to ac-POSGs. □

For discounted-sum rewards, the Bernoulli technique fails, since the return depends on the distribution of the hitting time of $T .$ . For CVaR, however, we show that a nonnegative return has $\mathsf { C V a R } _ { \alpha } > 0$ if and only if it is positive with probability greater than $1 - \alpha$ . Together with Lemma C.2, this gives undecidability for discounted-sum rewards and CVaR risk measure.

Theorem 4 (Discounted-sum reward with CVaR). For all rationals $\alpha , \gamma \in ( 0 , 1 )$ , it is undecidable whether $\mathsf { V a l } ( \mathsf { C V a R } _ { \alpha } , \mathsf { D i s c } _ { \gamma } ) > 0 ,$ , already for POMDPs with a single observation. Hence the strict threshold problem for discounted-sum rewards with CVaR risk measure is undecidable for RPOMDPs and ac-POSGs, with any rational discount factor.

Proofsketch. Take the POMDP of Lemma C.2 with $a < 1 - \alpha < b ,$ and collect reward 1 upon entering $T$ . For every discount factor the return is nonnegative, and it is positive exactly on plays reaching $T .$ Hence, by Lemma C.3, a strategy has positive $\mathsf { C V a R } _ { \alpha }$ if and only if it reaches $T$ with probability greater than $1 - \alpha$ . If the machine halts, the value is positive; otherwise every strategy reaches $T$ with probability at most $a < 1 - \alpha$ and the value is $0 .$ . This transfers to RPOMDPs, and to by our equivalence result to ac-POSGs. □

Decidability and approximability. The above results establish lower bounds and undecidability of problems. We now present results concerning decidability and approximability. We show that, for finite horizon total rewards and all four risk measures that we consider, the problem is decidable. Then, we use this results to show that, for discounted-sum rewards and all four risk measures, the value can be approximated up to an arbitrary precision.

Proposition 1 (Decidability and approximation). Let $\varPsi$ be one of the four risk measures $\mathbb { E } , \mathsf { M V } _ { \lambda }$ $\mathsf { C V a R } _ { \alpha } ,$ or ERisk<sub>θ</sub>, with rational parameters.

(a) The finite-horizon total reward threshold problem under any of the four risk measures $\varPsi$ is decidablefor RPOMDPs and ac-POSGs.

(b) For discounted-sum rewards under any of the four risk measures $\varPsi ,$ , the value can be approximated up to any precision; in particular, the strict threshold problem is recursively enumerable.

Proofsketch. (a) With a finite horizon, the threshold question is a first-order sentence over the reals of the form “there exist agent parameters such that for all environment parameters a polynomial inequality holds” (CVaR is a maximum of polynomials), which is decidable [48]. For entropic risk, all constants $e ^ { \theta x }$ are integer powers of one transcendental number $e ^ { 1 / D }$ [4]; we replace it by a variable and compare it with the finitely many algebraic boundary points of the resulting semialgebraic set. (b) Truncating after $N$ steps changes the discounted return by at most $\epsilon _ { N } = \gamma ^ { N }$ max $| \boldsymbol { r } | / ( 1 - \gamma )$ on every play. All four functionals are Lipschitz under such uniform perturbations, so the value changes by $O ( \epsilon _ { N } )$ , and the truncated value is computable by (a). □

## 5.3 Full observation

For the special case of full observability and RMDPs, our equivalence result shows that $( s , a ) \cdot$ rectangular polytopic RMDPs are equivalent to turn-based stochastic games (TBSGs), for all four reward objectives and under any law-invariant risk measure. Our novel complexity results for RMDPs concern lower bounds, which to the best of our knowledge present the first lower complexity bounds for RMDPs. In particular, we show that a number of problems are SSG-hard. The SSG problem is concerned with solving simple stochastic games (SSGs) and deciding whether the value of a vertex exceeds $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ . It is known that SSG is in NP ∩ coNP [11], but no polynomial-time algorithm is known. The proofs can be found in the appendix.

Theorem 5 (Total and long-run average reward). Let Ψ be one of the four risk measures E, ${ \mathsf { M V } } _ { \lambda }$ $( \lambda \geq 0 ) , { \mathsf { C V a R } } _ { \alpha } ( \alpha \in ( 0 , 1 ) )$ , or ERisk<sub>θ</sub> $( \theta \neq 0 )$ , and let X be Tot or LimAvg. Both threshold problems for Ψ and X are SSG-hard for RMDPs under polynomial-time many-one reductions; for E, ${ \mathsf { M V } } _ { \lambda } ,$ , and ${ \mathsf { C V a R } } _ { \alpha } ,$ even under logspace reductions.

Theorem 6 (Discounted-sum reward). Let Ψ be the expected value risk measure $\mathbb { E } ,$ and let X be $\mathsf { D i s c } _ { \gamma }$ with a rational discountfactor γ. Both threshold problemsfor Ψ and X are SSG-hard under polynomial-time Turing reductions.

## 6 Related Work

The study of robust MDPs goes back to [22, 34]. Efficient algorithms for solving RMDPs under expected discounted-sum rewards were proposed in [35, 22, 51, 24, 47, 17, 20], and under expected long-run average rewards in [49, 19, 18, 50, 9, 32, 2]. Robust POMDPs were introduced by [37], and [6] proposed game semantics under partial observability.

The relationship between RPOMDPs and POSGs is not a novel observation. [9] showed that RMDPs under expected discounted-sum and long-run average rewards can be reduced to TBSGs, and [6] showed that RPOMDPs can be reduced to POSGs. Most recently, [28] showed that RPOMDPs and POSGs are equivalent (i.e. inter-reducible) under omega-regular objectives, a class of logical objectives including reachability, safety, and linear temporal logic objectives, but which does not consider reward objectives. The novelty of our work is that RPOMDPs and POSGs are equivalent for reward objectives and every law-invariant risk measure. Our reductions are law preserving: they map strategies to strategies that induce the same probability distribution of the reward objective, up to a normalizing constant factor that depends only on the type of the reward objective.

Law invariance is a basic property of risk measures [27, 15]. Variance-based criteria in MDPs are studied in [45, 14, 30, 40]; CVaR in [42, 10, 26, 39, 31]; and entropic risk in [21]. For TBSGs, they are studied in [3]. Time-consistent dynamic risk measures are developed in [43].The works [36, 10] relate risk sensitivity to robustness in MDPs, however they do not propose complexity results or algorithms for solving RMDPs with reward objectives under these risk measures.

## 7 Conclusion

We showed that robust POMDPs and alternating-control POSGs are equivalent in a strong, distributional sense. We designed logspace reductions between the RPOMDP and ac-POSG models, that map strategies to strategies and preserve the distribution of the reward objective, up to a constant factor, for any of our four reward objectives and any law-invariant risk measure. This allows us to derive a number of new computational complexity results for R(PO)MDPs.

Several questions remain open: the exact complexity of finite-horizon total reward RPOMDPs beyond PSPACE-hardness; the decidability of the non-strict threshold problems for discounted E and

CVaR<sub>α</sub>, and of both threshold problems for discounted mean-variance and entropic risk, under partial observation; and upper bounds for mean-variance and CVaR in robust MDPs, where only SSGhardness is known. It would also be interesting to extend the equivalence beyond (s, a)-rectangular uncertainty.

## References

[1] Daniel Andersson and Peter Bro Miltersen. The complexity of solving stochastic games on graphs. In International Symposium on Algorithms and Computation, pages 112–121. Springer, 2009.

[2] Ali Asadi, Krishnendu Chatterjee, Ehsan Kafshdar Goharshady, Mehrdad Karrabi, Alipasha Montaseri, and Carlo Pagano. Strongly polynomial time complexity of policy iteration for l∞ robust mdps. CoRR, abs/2601.23229, 2026. URL https://doi.org/10.48550/ arXiv.2601.23229.

[3] Christel Baier, Krishnendu Chatterjee, Tobias Meggendorfer, and Jakob Piribauer. Entropic risk for turn-based stochastic games. Information and Computation, 301:105214, 2024.

[4] Alan Baker. Transcendental number theory. Cambridge university press, 2022.

[5] Patrick Billingsley. Probability and Measure. New York, 3rd edition, 1995.

[6] Eline M. Bovy, Marnix Suilen, Sebastian Junges, and Nils Jansen. Imprecise probabilities meet partial observability: Game semantics for robust POMDPs. In Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI), pages 6697–6706, 2024.

[7] Mahmoud El Chamie and Hala Mostafa. Robust action selection in partially observable markov decision processes with model uncertainty. In 57th IEEE Conference on Decision and Control, CDC 2018, Miami, FL, USA, December 17-19, 2018, pages 5586–5591, 2018. URL https: //doi.org/10.1109/CDC.2018.8619468.

[8] Krishnendu Chatterjee, Laurent Doyen, and Thomas A. Henzinger. A survey of partialobservation stochastic parity games. Formal Methods Syst. Des., 43(2):268–284, 2013. URL https://doi.org/10.1007/s10703-012-0164-2.

[9] Krishnendu Chatterjee, Ehsan Kafshdar Goharshady, Mehrdad Karrabi, Petr Novotný, and Ðorde Žikeli¯ c. Solving long-run average reward robust MDPs via stochastic games. In´ Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI), pages 6707–6715, 2024.

[10] Yinlam Chow, Aviv Tamar, Shie Mannor, and Marco Pavone. Risk-sensitive and robust decision-making: a cvar optimization approach. Advances in neural information processing systems, 28, 2015.

[11] Anne Condon. The complexity of stochastic games. Information and Computation, 96(2): 203–224, 1992.

[12] Anne Condon and Richard J. Lipton. On the complexity of space bounded interactive proofs. In FOCS, volume 89, pages 462–467, 1989.

[13] Murat Cubuktepe, Nils Jansen, Sebastian Junges, Ahmadreza Marandi, Marnix Suilen, and Ufuk Topcu. Robust finite-state controllers for uncertain pomdps. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021, pages 11792–11800, 2021. URL https://doi.org/10.1609/aaai.v35i13.17401.

[14] Jerzy A. Filar, Lodewijk C. M. Kallenberg, and Huey-Miin Lee. Variance-penalized Markov decision processes. Mathematics of Operations Research, 14(1):147–161, 1989.

[15] Hans Föllmer and Alexander Schied. Stochastic Finance: An Introduction in Discrete Time. De Gruyter, 4th edition, 2016.

[16] Maris F. L. Galesloot, Roman Andriushchenko, Milan Ceska, Sebastian Junges, and Nils Jansen. Robust finite-memory policy gradients for hidden-model pomdps. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI 2025, Montreal, Canada, August 16-22, 2025, pages 8518–8526, 2025. URL https://doi.org/ 10.24963/ijcai.2025/947.

[17] Mohammad Ghavamzadeh, Marek Petrik, and Yinlam Chow. Safe policy improvement by minimizing robust baseline regret. In Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain, pages 2298–2306, 2016. URL https://proceedings.neurips.cc/paper/2016/hash/ 9a3d458322d70046f63dfd8b0153ece4-Abstract.html.

[18] Vineet Goyal and Julien Grand-Clément. Robust markov decision processes: Beyond rectangularity. Math. Oper. Res., 48(1):203–226, 2023. URL https://doi.org/10.1287/ moor.2022.1259.

[19] Julien Grand-Clément and Marek Petrik. Reducing blackwell and average optimality to discounted mdps via the blackwell discount factor. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 52628– 52647. Curran Associates, Inc., 2023. doi: 10.52202/075280-2292. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ a4e720ce31ccd8ba747d8863e1580fa8-Paper-Conference.pdf.

[20] Chin Pang Ho, Marek Petrik, and Wolfram Wiesemann. Partial policy iteration for l1-robust markov decision processes. J. Mach. Learn. Res., 22:275:1–275:46, 2021. URL https: //jmlr.org/papers/v22/20-445.html.

[21] Ronald A. Howard and James E. Matheson. Risk-sensitive Markov decision processes. Management Science, 18(7):356–369, 1972.

[22] Garud N. Iyengar. Robust dynamic programming. Mathematics of Operations Research, 30 (2):257–280, 2005.

[23] Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. Planning and acting in partially observable stochastic domains. Artificial Intelligence, 101(1–2):99–134, 1998.

[24] David L. Kaufman and Andrew J. Schaefer. Robust modified policy iteration. INFORMS J. Comput., 25(3):396–410, 2013. doi: 10.1287/IJOC.1120.0509. URL https://doi.org/ 10.1287/ijoc.1120.0509.

[25] Merlijn Krale, Eline M. Bovy, Maris F. L. Galesloot, Thiago D. Simão, and Nils Jansen. On evaluating policies for robust POMDPs. In Eighteenth European Workshop on Reinforcement Learning, 2025. URL https://openreview.net/forum?id=Q1qM9iF3iH.

[26] Jan Kˇretínský and Tobias Meggendorfer. Conditional value-at-risk for reachability and mean payoff in Markov decision processes. In Proceedings of the 33rd Annual ACM/IEEE Symposium on Logic in Computer Science (LICS), pages 609–618. ACM, 2018.

[27] Shigeo Kusuoka. On law invariant coherent risk measures. In Advances in Mathematical Economics, pages 83–95. Springer, Tokyo, Japan, 2001.

[28] Durgam Latha, Dion Reji, S. Akshay, Ðorde Žikeli ¯ c, and Shankaranarayanan Krishna. Solving ´ robust POMDPs with ω-regular objectives via partially observable stochastic games. arXiv preprint arXiv:2608.24986, 2026.

[29] Omid Madani, Steve Hanks, and Anne Condon. On the undecidability of probabilistic planning and related stochastic optimization problems. Artificial Intelligence, 147(1–2):5–34, 2003.

[30] Shie Mannor and John N. Tsitsiklis. Algorithmic aspects of mean–variance optimization in Markov decision processes. European Journal of Operational Research, 231(3):645–653, 2013.

[31] Tobias Meggendorfer. Risk-aware stochastic shortest path. In Proceedings of the AAAI Con ference on Artificial Intelligence, volume 36, pages 9858–9867, 2022.

[32] Tobias Meggendorfer, Maximilian Weininger, and Patrick Wienhöft. Solving robust markov decision processes: Generic, reliable, efficient. In Thirty-Ninth AAAI Conference on Artificial Intelligence, Thirty-Seventh Conference on Innovative Applications of Artificial Intelligence, Fifteenth Symposium on Educational Advances in Artificial Intelligence, AAAI 2025, Philadelphia, PA, USA, February 25 - March 4, 2025, pages 26631–26641, 2025. URL https://doi.org/10.1609/aaai.v39i25.34865.

[33] Hideaki Nakao, Ruiwei Jiang, and Siqian Shen. Distributionally robust partially observable markov decision process with moment-based ambiguity. SIAM J. Optim., 31(1):461–488, 2021. URL https://doi.org/10.1137/19M1268410.

[34] Arnab Nilim and Laurent El Ghaoui. Robust control of Markov decision processes with uncertain transition matrices. Operations Research, 53(5):780–798, 2005.

[35] Arnab Nilim and Laurent El Ghaoui. Robustness in markov decision problems with uncertain transition matrices. In Advances in Neural Information Processing Systems 16 [Neural Information Processing Systems, NIPS 2003, December 8-13, 2003, Vancouver and Whistler, British Columbia, Canada], pages 839– 846, 2003. URL https://proceedings.neurips.cc/paper/2003/hash/ 300891a62162b960cf02ce3827bb363c-Abstract.html.

[36] Takayuki Osogami. Robustness and risk-sensitivity in markov decision processes. Advances in neural information processing systems, 25, 2012.

[37] Takayuki Osogami. Robust partially observable markov decision process. In International Conference on Machine Learning, pages 106–115. PMLR, 2015.

[38] Christos H. Papadimitriou and John N. Tsitsiklis. The complexity of Markov decision processes. Mathematics of Operations Research, 12(3):441–450, 1987.

[39] Jakob Piribauer and Christel Baier. On Skolem-hardness and saturation points in Markov decision processes. In Proceedings of the 47th International Colloquium on Automata, Languages, and Programming (ICALP), volume 168 of LIPIcs, pages 138:1–138:17. Schloss Dagstuhl – Leibniz-Zentrum für Informatik, 2020.

[40] Jakob Piribauer, Ocan Sankur, and Christel Baier. The variance-penalized stochastic shortest path problem. In Proceedings of the 49th International Colloquium on Automata, Languages, and Programming (ICALP), volume 229 of LIPIcs, pages 129:1–129:19. Schloss Dagstuhl – Leibniz-Zentrum für Informatik, 2022.

[41] Martin L Puterman. Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.

[42] R. Tyrrell Rockafellar and Stanislav Uryasev. Optimization of conditional value-at-risk. Journal ofRisk, 2:21–42, 2000.

[43] Andrzej Ruszczynski. Risk-averse dynamic programming for Markov decision processes.´ Mathematical Programming, 125(2):235–261, 2010.

[44] Lloyd S. Shapley. Stochastic games. Proceedings of the National Academy of Sciences, 39 (10):1095–1100, 1953.

[45] Matthew J. Sobel. The variance of discounted Markov decision processes. Journal ofApplied Probability, 19(4):794–802, 1982.

[46] Marnix Suilen, Nils Jansen, Murat Cubuktepe, and Ufuk Topcu. Robust policy synthesis for uncertain pomdps via convex optimization. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI 2020, pages 4113–4120, 2020. URL https://doi.org/10.24963/ijcai.2020/569.

[47] Aviv Tamar, Shie Mannor, and Huan Xu. Scaling up robust mdps using function approximation. In Proceedings ofthe 31th International Conference on Machine Learning, ICML 2014, Beijing, China, 21-26 June 2014, pages 181–189, 2014. URL http://proceedings. mlr.press/v32/tamar14.html.

[48] Alfred Tarski. A decision method for elementary algebra and geometry. In Quantifier elimination and cylindrical algebraic decomposition, pages 24–84. Springer, 1998.

[49] Ambuj Tewari and Peter L. Bartlett. Bounded parameter markov decision processes with average reward criterion. In Learning Theory, 20th Annual Conference on Learning Theory, COLT 2007, San Diego, CA, USA, June 13-15, 2007, Proceedings, pages 263–277, 2007. URL https://doi.org/10.1007/978-3-540-72927-3\_20.

[50] Yue Wang, Alvaro Velasquez, George K. Atia, Ashley Prater-Bennette, and Shaofeng Zou. Robust average-reward markov decision processes. In Thirty-Seventh AAAI Conference on Artificial Intelligence, AAAI 2023, Thirty-Fifth Conference on Innovative Applications ofArtificial Intelligence, IAAI 2023, Thirteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2023, Washington, DC, USA, February 7-14, 2023, pages 15215–15223, 2023. URL https://doi.org/10.1609/aaai.v37i12.26775.

[51] Wolfram Wiesemann, Daniel Kuhn, and Berç Rustem. Robust Markov decision processes. Mathematics ofOperations Research, 38(1):153–183, 2013.

## Overview of the Appendix

Compiling prescriptions into ordinary strategies p. 20 Two compilations of games with min-prescriptions into ordinary games: the chain gadget for Tot<sub>N</sub>, Tot, and LimAvg (Lemma A.1), and the vector-action compilation for discounted returns (Proposition A.1). Examples A.1 and A.2 show why the chain does not preserve Disc<sub>γ</sub> while vector actions do.

## B. Proofs for Section 4

B.1 Forward construction p. 31 Lemma 1: from an RPOMDP to an ac-POSG, with a bijection of strategies and equal return laws up to the factor c; consequences for ordinary POSGs (Corollary B.1 and its discounted version).

B.2 Backward construction p. 33 Lemma 2: from an ac-POSG to an RPOMDP, with tagged states and dummy states; the return laws are preserved exactly.

B.3 The umbrella theorem p. 35 Theorem 1: equality of values for every law-invariant risk measure, and the logarithmic-space bound.

B.4 Scaling identities p. 36 How E, MV<sub>λ</sub>, CVaR<sub>α</sub>, and ERisk<sub>θ</sub> react to multiplying the return by c (Equation (3)), used for Corollary 1.

## C. Proofs for Section 5

Lower bounds, undecidability, and decidability results for the reward objectives.

C.1 Auxiliary facts p. 37 Monotone maps commute with inf and sup (F1); memoryless deterministic saddle points persist on restricted strategy classes (F2).

C.2 Derivation of the profiles p. 37 The four Bernoulli profiles computed from the definitions, and Lemma C.1: they are monotone and strict at the top.

C.3 Finite horizon p. 38 Theorem 2: PSPACE-hardness, from [38], by adding a step counter and rewarding survival.

C.4 The gap lemma and total and average returns p. 39 Lemma C.2: gap POMDPs from two-counter machines; Theorem 3: undecidability of Tot and LimAvg with a fixed rational threshold.

C.5 Discounted CVaR p. 41 Lemma C.3: positivity of CVaR; Theorem 4: undecidability of the strict threshold problem.

C.6 Decidability and approximation p. 43 Proposition 1: finite-horizon thresholds are decidable via the first-order theory of the reals; discounted values are approximated by truncation.

C.7 SSG-hardness for total and average returns p. 47 Theorem 5: stopping games, the threshold shift for CVaR, rewards that make every return law Bernoulli, the backward construction, and the thresholds.

C.8 SSG-hardness for expected discounted reward p. 50 Theorem 6: the source problem for discounted TBSGs, the RMDP built from a game, its value computed by a verification argument, and the composition of the reductions.

## Appendix

Recall that at the end of section 2.1, we assume all actions are available at all the states of POSG. In this appendix, we present reductions from one model to the another (RPOMDP to POSG and vice-versa), as well as POSGs with prescriptions to ordinary POSGs. In some of these proofs, while going from one model to the other, the set of actions are expanded, and in some transitions we use only the newly defined actions. Indeed, as seen in the main paper, the remaining actions are not “missing”, one can always complete the transitions by directing the transitions on the remaining actions to a sink which is losing for the owner of the said state. The actions that are depicted on transitions from state s (prior to completion) are the effective ones, in the same spirit as [28].

To ease our notations in proofs, we define $A ( s )$ to denote the set of effective actions at s.

## A Compiling prescriptions into ordinary strategies

Let G be an ac-POSG, with min-strategies as in Section 2.2 (prescriptions). A min-strategy is ordinary if $\pi ( h ) ( t ) = \pi ( h ) ( t ^ { \prime } )$ whenever $\mathcal { Z } _ { \sf m i n } ( t ) = \mathcal { Z } _ { \sf m i n } ( t ^ { \prime } )$ (so π is a map $H _ { m i n }  \Delta ( A _ { \operatorname* { m i n } } ) )$ , and max-strategies are always ordinary. Given G with prescriptions, we build a game $\mathcal { G } ^ { \prime }$ (the compiled game) in which min has ordinary strategies only, and relate the two games. For an ac-POSG H with observation functions $\mathcal { Z } _ { \mathfrak { m a x } } { } ^ { \mathcal { H } }$ and $\mathcal { Z _ { \mathrm { m i n } } } ^ { \mathrm { ~ \tiny ~ \mathcal { H } ~ } }$ and a prefix $u = s _ { 0 } a _ { 0 } s _ { 1 } \cdot \cdot \cdot s _ { k }$ of H we write $h _ { \mathcal { H } } ^ { \mathsf { m a x } } ( u ) = \mathcal { Z } _ { \mathsf { m a x } } { } ^ { \mathcal { H } } ( s _ { 0 } ) \cdot \cdot \cdot \mathcal { Z } _ { \mathsf { m a x } } { } ^ { \mathcal { H } } ( s _ { k } )$ and $h _ { \mathcal { H } } ^ { \mathrm { m i n } } ( u ) = \mathcal { Z } _ { \mathrm { m i n } } { } ^ { \mathcal { H } } ( s _ { 0 } ) \cdot \cdot \cdot \mathcal { Z } _ { \mathrm { m i n } } { } ^ { \mathcal { H } } ( s _ { k } )$ for the observation histories of the two players, as in Section 2.2.

The compiled game. For $o \in \mathcal { O } _ { \operatorname* { m i n } }$ , let $C _ { o } = \{ t \in S _ { \sf m i n } : \mathcal { Z } _ { \sf m i n } ( t ) = o \} = \{ t _ { 1 } ^ { o } , \dots , t _ { m _ { o } } ^ { o } \}$ be a fixed enumeration of the class of $^ { O , }$ and let pos $( t _ { i } ^ { o } ) = i .$ . Let $k = \mathrm { m a x } _ { o } m _ { c }$ be the size of the largest class. For $1 \leq i \leq k$ let $A _ { o } ( i ) = A ( t _ { i } ^ { o } ) { \mathrm { i f ~ } } i \leq m _ { o }$ and $A _ { o } ( i ) = A ( t _ { 1 } ^ { o } )$ if $m _ { o } < i \leq k$ . The idea is to replace every min-state t by a chain of k slots, in which min chooses an action at every slot, but only the choice at the slot pos(t) is the real one. Since min observes the slot but not t, a strategy of min in the chain can only depend on the class $C _ { o }$ and the slot $i ,$ and the choice at slot i plays the role of the prescription for $t _ { i } ^ { o }$

Let $\mathcal { G } \ = \ ( S _ { \mathrm { m a x } } , S _ { \mathrm { m i n } } , A _ { \mathrm { m a x } } , A _ { \mathrm { m i n } } , \delta , s _ { 0 } , \mathcal { O } _ { \mathrm { m a x } } , \mathcal { O } _ { \mathrm { m i n } } , \mathcal { Z } _ { \mathrm { m a x } } , \mathcal { Z } _ { \mathrm { m i n } } , r )$ be an ac-POSG and let $\perp \notin$ $A _ { \sf m a x } \cup A _ { \sf m i n }$ and $\star , \circledcirc$ be fresh symbols. For $t \in S _ { \operatorname* { m i n } } .$ , write $o = \mathcal { Z } _ { \operatorname* { m i n } } ( t )$ and $p = \mathrm { p o s } ( t )$ . The compiled game is the ac-POSG $\mathcal { G } ^ { \prime } = ( S _ { \mathrm { m a x } } ^ { \prime } , S _ { \mathrm { m i n } } ^ { \prime } , A _ { \mathrm { m a x } } ^ { \prime } , A _ { \mathrm { m i n } } ^ { \prime } , \delta ^ { \prime } , s _ { 0 } , O _ { \mathrm { m a x } } ^ { \prime } , O _ { \mathrm { m i n } } ^ { \prime } , \mathcal { Z } _ { \mathrm { m a x } } ^ { \prime } , \mathcal { Z } _ { \mathrm { m i n } } ^ { \prime } , r ^ { \prime } )$ defined as follows.

• States. $S _ { \sf m a x } ^ { \prime } = S _ { \sf m a x } \cup D$ and $S _ { \sf m i n } ^ { \prime } = X$ , where

$$
\begin{array} { r l } & { X = \{ x _ { t , i } : t \in S _ { \mathsf { m i n } } , 1 \leq i \leq \mathsf { p o s } ( t ) \} \cup \{ x _ { t , b , i } : t \in S _ { \mathsf { m i n } } , \mathsf { p o s } ( t ) < i \leq k , b \in A ( t ) \} , } \\ & { D = \{ d _ { t , i } : t \in S _ { \mathsf { m i n } } , 1 \leq i < \mathsf { p o s } ( t ) \} \cup \{ d _ { t , b , i } : t \in S _ { \mathsf { m i n } } , \mathsf { p o s } ( t ) \leq i < k , b \in A ( t ) \} . } \end{array}
$$

Here, $A ( t )$ denotes set of effective actions of state t in $\mathcal { G }$ and $A ^ { \prime } ( t )$ denotes set of effective actions of state t in $\mathcal { G } ^ { \prime } .$ . The state $x _ { t , i }$ is the slot i of the chain of t before the real choice has been made (the choice at $i = \mathrm { p o s } ( t )$ is the real one), and $x _ { t , b , i }$ is the slot i after the real choice b has been made and recorded in the state. We call i the slot index of the states $x _ { t , i } , x _ { t , b , i } , d _ { t , i }$ and $d _ { t , b , i }$

• Actions. $A _ { \mathsf { m a x } } ^ { \prime } = A _ { \mathsf { m a x } } \cup \{ \perp \}$ and $A _ { \operatorname* { m i n } } ^ { \prime } = A _ { \operatorname* { m i n } }$ . The available actions are $A ^ { \prime } ( s ) = A ( s )$ for $s \in S _ { \sf m a x } , A ^ { \prime } ( d ) = \{ \bot \}$ for $d \in D$ , and $A ^ { \prime } ( x ) = A _ { o } ( i )$ for every $x \in X$ of the form $\boldsymbol { x } _ { t , i }$ or $x _ { t , b , i } ,$ where $o = \mathcal { Z } _ { \operatorname* { m i n } } ( t )$

• Transitions and rewards. Every transition has reward 0, except for those marked with a reward. Let $t \in S _ { \mathsf { m i n } } , o = \mathcal { Z } _ { \mathsf { m i n } } ( t )$ , and $p = \operatorname { p o s } ( t )$

(E) Entry. For $s \in S _ { \mathsf { m a x } }$ and $\iota \in A ( s ) \colon \delta ^ { \prime } ( s , a ) ( x _ { t , 1 } ) = \delta ( s , a ) ( t )$ , with reward $r ^ { \prime } ( s , a , x _ { t , 1 } ) =$ $r ( s , a , t )$

(C1) For $1 \leq i < p$ and $b \in A _ { o } ( i ) \colon \delta ^ { \prime } ( x _ { t , i } , b ) = \mathsf { D i r a c } ( d _ { t , i } )$ and $\delta ^ { \prime } ( d _ { t , i } , \perp ) = \mathsf { D i r a c } ( x _ { t , i + 1 } )$

(C2) I ${ \mathrm { i } } p < k { \mathrm { : } }$ for $b \in A ( t )$ , the real choice leads to $\delta ^ { \prime } ( x _ { t , p } , b ) = \mathsf { D i r a c } ( d _ { t , b , p } )$ . For $p < i < k$ and $b ^ { \prime } \in A ( t ) , b \in A _ { o } ( i ) \colon \delta ^ { \prime } ( x _ { t , b ^ { \prime } , i } , b ) = \mathsf { D i r a c } ( d _ { t , b ^ { \prime } , i } )$ . For $p \leq i < k \colon \delta ^ { \prime } ( d _ { t , b ^ { \prime } , i } , \perp ) =$ Dirac $\left( x _ { t , b ^ { \prime } , i + 1 } \right)$

(X) Exit. ${ \mathrm { ~ I f ~ } } p \ = \ k { : }$ for $b \in A ( t )$ and $s ^ { \prime } \in S _ { \sf m a x } , \delta ^ { \prime } ( x _ { t , k } , b ) = \delta ( t , b )$ with reward $r ^ { \prime } ( x _ { t , k } , b , s ^ { \prime } ) = r ( t , b , s ^ { \prime } )$ . If $p \ <$ k: for $b ^ { \prime } \in A ( t )$ and $b \in A _ { o } ( k ) , \delta ^ { \prime } ( x _ { t , b ^ { \prime } , k } , b ) =$ $\delta ( t , b ^ { \prime } )$ with reward $r ^ { \prime } ( x _ { t , b ^ { \prime } , k } , b , s ^ { \prime } ) = r ( t , b ^ { \prime } , s ^ { \prime } )$

• Observations. $O _ { \mathsf { m a x } } ^ { \prime } = \mathcal { O } _ { \mathsf { m a x } } \cup \left( \mathcal { O } _ { \mathsf { m a x } } \times \{ \star \} \right)$ and $O _ { \mathsf { m i n } } ^ { \prime } = \mathcal { O } _ { \mathsf { m i n } } \cup ( \mathcal { O } _ { \mathsf { m i n } } \times [ k ] ) \cup ( \mathcal { O } _ { \mathsf { m i n } } \times$ $[ k ] \times \{ \odot \} )$ where $[ k ] = \{ 1 , 2 , \cdots , k \}$ . For $s \in S _ { \sf m a x } , { \mathcal Z } _ { \sf m a x } { } ^ { \prime } ( s ) = { \mathcal Z } _ { \sf m a x } ( s )$ and $ { \mathcal { Z } _ { \sf m i n } } ^ { \prime } { ( s ) } =$ $\mathcal { Z } _ { \mathrm { m i n } } ( s )$ . For all chain states of t (that ${ \mathrm { i s } } ,$ all states in $X \cup D$ with the first index $t ) .$ , with $o = \mathcal { Z } _ { \operatorname* { m i n } } ( t )$ and slot index i:

$$
\begin{array} { r } { \mathcal { Z } _ { \mathrm { m a x } } { ' } = ( \mathcal { Z } _ { \mathrm { m a x } } ( t ) , \star ) , \qquad \mathcal { Z } _ { \mathrm { m i n } } { ' } ( x _ { \cdot , i } ) = ( o , i ) , \qquad \mathcal { Z } _ { \mathrm { m i n } } { ' } ( d _ { \cdot , i } ) = ( o , i , \odot ) . } \end{array}
$$

Properties of $\mathcal { G } ^ { \prime }$ . Every state of $S _ { \mathrm { m a x } } ^ { \prime }$ moves to a state of $S _ { \mathrm { m i n } } ^ { \prime }$ and vice versa, so $\mathcal { G } ^ { \prime }$ is an ac-POSG. A transition from min-state t to $s ^ { \prime }$ on action b in $\mathcal { G }$ with reward $\boldsymbol { r } ( t , b , s ^ { \prime } )$ corresponds in $\mathcal { G } ^ { \prime }$ to the path

$$
x _ { t , 1 } \to y _ { 1 } \to x _ { \cdot , 2 } \to y _ { 2 } \to \cdot \cdot \cdot \to y _ { k - 1 } \to x _ { \cdot , k } \to s ^ { \prime }
$$

with $y _ { i } \in D$ , which has k states of $X , k - 1$ states of $D ,$ and $2 k - 1$ transitions, the last of which earns the reward $\boldsymbol { r } ( t , b , s ^ { \prime } )$ . Together with the entry transition, a round of $\mathcal { G }$ (a max-transition and a min-transition) takes 2k transitions in $\mathcal { G } ^ { \prime }$ . The slots reveal nothing: the observations of both players along a chain do not depend on t within $C _ { o } .$ , and the chain has a fixed length.

Size. The game $\mathcal { G } ^ { \prime }$ has polynomial size in G. Indeed, let $A _ { \mathrm { m i n } } ^ { \# } = \operatorname* { m a x } _ { t \in S _ { \mathrm { m i n } } } | A ( t )$ |. For every $t \in S _ { \operatorname* { m i n } }$ , the chain of t has at most $k + ( k - 1 ) + 2 ( k - 1 ) A _ { \operatorname* { m i n } } ^ { \# } \leq 2 k ( 1 + A _ { \operatorname* { m i n } } ^ { \# } )$ states, so

$$
| S ^ { \prime } | \leq | S _ { \mathrm { m a x } } | + 2 k | S _ { \mathrm { m i n } } | ( 1 + A _ { \mathrm { m i n } } ^ { \# } ) ,
$$

and $k \leq | S _ { \mathrm { m i n } } |$ . The action sets are $A _ { \sf m a x } \cup \{ \bot \}$ and $A _ { \mathrm { m i n } }$ . Every state of $S ^ { \prime }$ has at most max $\{ \vert A _ { \sf m a x } \vert , \vert A _ { \sf m i n } \vert \}$ available actions, and every transition distribution is either a Dirac distribution or a copy of a distribution of $\mathcal { G } \mathrm { : }$ : at the entry, $\delta ^ { \prime } ( s , a )$ is a copy of $\delta ( s , a )$ in which every target $t \in S _ { \operatorname* { m i n } }$ is renamed to $x _ { t , 1 }$ , and at the exit, $\delta ^ { \prime } ( x , b )$ is a copy of $\delta ( t , b ^ { \prime } )$ with unchanged targets, defined at a chain state x of t instead of at t (so the source is renamed). The observations are pairs or triples of an observation of $\mathcal { G }$ and a number up to $k ,$ and the rewards are rewards of $\mathcal { G } \mathrm { ~ o r ~ } 0$ . Hence the encoding of $\mathcal { G } ^ { \prime }$ is polynomial in the encoding of $\mathcal { G } .$

Returns. Let $\rho ^ { \prime }$ be a play of $\mathcal { G } ^ { \prime }$ . It consists of rounds: after the real state $s _ { j }$ and an action $a _ { j } .$ the play traverses the chain of a state $t _ { j }$ , whose real choice is $b _ { j }$ , and moves to $s _ { j + 1 }$ . Let $\operatorname { p r } ( \rho ^ { \prime } ) =$ $s _ { 0 } a _ { 0 } t _ { 0 } b _ { 0 } s _ { 1 } a _ { 1 } t _ { 1 } b _ { 1 } \cdot \cdot \cdot$ be the play of $\mathcal { G }$ obtained by erasing the chain states and keeping the action $b _ { j }$ chosen at the slot $\mathrm { p o s } ( t _ { j } )$ . The same definition applies to a prefix v of $\mathcal { G } ^ { \prime }$ that ends in a state of $S _ { \mathrm { { m a x } } } ,$ which consists of complete rounds, and gives a prefix $\mathrm { p r } ( v )$ of $\mathcal { G }$ that ends in a state of $S _ { \sf m a x }$ . Let $( r _ { n } ) _ { n \geq 0 }$ be the reward sequence of $\operatorname { p r } ( \rho ^ { \prime } )$ , so that $r _ { 2 j } = r ( s _ { j } , a _ { j } , t _ { j } )$ and $r _ { 2 j + 1 } = r ( t _ { j } , b _ { j } , s _ { j + 1 } )$ By the definition of $\mathcal { G } ^ { \prime }$ , the n-th transition of $\vert \rho ^ { \prime }$ earns $r _ { 2 j } \mathrm { i f } n = 2 k j$ , earns $r _ { 2 j + 1 } \mathrm { i f } n = 2 k j + 2 k - 1$ and earns 0 otherwise, so the reward sequence of $\rho ^ { \prime }$ is

$$
r _ { 0 } , \boldsymbol { 0 } ^ { 2 k - 2 } , r _ { 1 } , r _ { 2 } , \boldsymbol { 0 } ^ { 2 k - 2 } , r _ { 3 } , \ldots
$$

For a return X of G, let $X ^ { \prime }$ be the return of $\mathcal { G } ^ { \prime }$ and c the scaling factor given by

$$
\frac { X \enspace \enspace \middle | \enspace \mathrm { T o t } _ { N } \enspace \mathrm { T o t } \enspace \mathrm { L i m A v g } } { X ^ { \prime } \enspace \middle | \enspace \mathrm { T o t } \enspace N ^ { \prime } \enspace \mathrm { T o t } \enspace \mathrm { L i m A v g } } \qquad N ^ { \prime } = \left\{ \begin{array} { l l } { k N } & { \mathrm { i f } \enspace N \mathrm { ~ i s ~ e v e n } , } \\ { k ( N - 1 ) + 1 } & { \mathrm { i f } \enspace N \mathrm { ~ i s ~ o d d } . } \end{array} \right.
$$

The factor $c$ is defined by the requirement $X ^ { \prime } ( \rho ^ { \prime } ) = c \cdot X ( \mathrm { p r } ( \rho ^ { \prime } ) )$ for every play $\rho ^ { \prime }$ of $\mathcal { G } ^ { \prime }$ (proved in the proof of Lemma A.1 below). For ${ \mathsf { T o t } } _ { N }$ and Tot, the plays $\rho ^ { \prime }$ and $\operatorname { p r } ( \rho ^ { \prime } )$ contain the same rewards, in the same order, and $N ^ { \prime }$ is the number of transitions of $\overline { { \rho ^ { \prime } } }$ that contain the first N rewards of $\operatorname { p r } ( \rho ^ { \prime } )$ (each full round has 2k transitions, and an odd N has one extra max-transition, which is a single transition of $\mathcal { G } ^ { \prime } )$ ), so $c = 1$ . For LimAvg, the rewards are also the same, but 2 rewards of $\mathcal { G }$ are spread over 2k transitions of $\mathcal { G } ^ { \prime }$ , so the average per transition of $\mathcal { G } ^ { \prime }$ is $1 / k$ times the average per transition of $\mathcal { G } _ { : }$ , and $c = 1 / k$

Lemma A.1 (Compilation). Let $X \in \{ \mathsf { T o t } _ { N } , \mathsf { T o t } , \mathsf { L i m A v g } \}$ and let $X ^ { \prime }$ and c be as above. There is a bijection $\sigma \mapsto \sigma ^ { \prime }$ from $\Sigma _ { \mathcal { G } }$ to $\Sigma _ { \mathcal { G } }$ ′ and a map dec from the ordinary min-strategies of $\cdot { \mathcal { G } } ^ { \prime }$ onto the prescriptions $\pi _ { \mathcal { G } }$ , such thatfor all $\sigma \in \Sigma _ { \mathcal { G } }$ and all ordinary $\pi ^ { \prime } ,$

$$
\begin{array} { r } { \mathcal { L } _ { X ^ { \prime } } ^ { \sigma ^ { \prime } , \pi ^ { \prime } } = ( S _ { c } ) _ { \# } \mathcal { L } _ { X } ^ { \sigma , \operatorname* { d e c } ( \pi ^ { \prime } ) } . } \end{array}
$$

Recall, $( S _ { c } ) _ { \# }$ from section 2. Consequently, for every law functional $\varPsi ,$ the value of $\because G ^ { \prime }$ with ordinary strategies satisfies $\mathsf { V a l } _ { \mathcal { G } ^ { \prime } } ( \varPsi , X ^ { \prime } ) = \mathsf { V a l } _ { \mathcal { G } } ( \varPsi \circ ( S _ { c } ) _ { \# } , X )$ .

Proof. Histories. Let W be the set offinite observation histories $w = o _ { 0 } o _ { 1 } \cdot \cdot \cdot o _ { 2 n } ( n \geq 0 )$ of $\mathcal { G }$ that end in a max-state, so that the letters with even index belong to max-states and those with odd index belong to min-states. For $o \in \mathcal { O } _ { \operatorname* { m i n } }$ , let $\xi _ { o } = ( o , 1 ) ( o , 1 , \odot ) ( o , 2 ) ( o , 2 , \odot ) \cdots ( o , k - 1 , \odot ) ( o , k )$ be the word of the min-observations of a chain, which has length $2 k - 1$ and depends on o only, and let $\xi _ { \overline { { o } } } ^ { \leq i }$ be its prefix that ends in $( o , i )$ . For $w \in W$ , let $\Theta ^ { \mathsf { m a x } } ( w )$ and $\Theta ^ { \mathrm { m i n } } ( w )$ be the finite words obtained from w by replacing every letter o with odd index by the block $( o , \dot { \star } ) ^ { 2 \dot { k } - 1 }$ and by the block $\xi _ { o } ,$ , respectively. Both maps are injective, because all blocks have the same length $2 k - 1$ . Let u be a prefix of G that ends in $s \in S _ { \sf m a x }$ , and let v be a prefix of $\mathcal { G } ^ { \prime }$ with $\mathrm { p r } ( v ) = u$ that ends in s (here pr erases the chain states of complete rounds, as above). Then

$$
h _ { \mathcal { G } ^ { \prime } } ^ { \mathsf { m a x } } ( v ) = \varTheta ^ { \mathsf { m a x } } ( h _ { \mathcal { G } } ^ { \mathsf { m a x } } ( u ) ) , \qquad h _ { \mathcal { G } ^ { \prime } } ^ { \mathsf { m i n } } ( v ) = \varTheta ^ { \mathsf { m i n } } ( h _ { \mathcal { G } } ^ { \mathsf { m i n } } ( u ) ) .
$$

Suppose now that v is extended by the entry transition to $x _ { t , 1 }$ and by the following chain up to the slot i, where $o = \mathcal { Z } _ { \operatorname* { m i n } } ( t )$ . The map $\Theta ^ { \mathrm { m i n } }$ is applied to the history $h _ { \mathcal { G } } ^ { \mathsf { m i n } } ( u )$ before the visit of $t ,$ which ends in the observation of the max-state $s ,$ and the block $\xi _ { \bar { o } } ^ { \leq i }$ of the current visit is appended. Hence the min-history of $\mathcal { G } ^ { \prime }$ at the slot i is $\theta ^ { \mathsf { m i n } } ( h _ { \mathcal { G } } ^ { \mathsf { m i n } } ( u ) ) \xi _ { \bar { \sigma } } ^ { \leq i }$ , which depends on $h _ { \mathcal { G } } ^ { \mathsf { m i n } } ( u )$ , o and i, but not on $t \in C _ { o }$ . The max-histories of $\mathcal { G } ^ { \prime }$ at the states of $S _ { \sf m a x }$ are exactly the words $\Theta ^ { \mathsf { m a x } } ( w )$ , and the symbol ⋆ separates them from the histories at d-states, where max has the single action ⊥.

Example $( k \ : = \ : 2 )$ . Let $u ~ = ~ s _ { 0 } a _ { 0 } t _ { 0 } b _ { 0 } s _ { 1 }$ . Its observation histories are $h _ { \mathcal { G } } ^ { \sf m i n } ( u ) ~ = ~ o _ { 0 } o _ { 1 } o _ { 2 }$ and $h _ { \mathcal { G } } ^ { \sf m a x } ( u ) = q _ { 0 } q _ { 1 } q _ { 2 }$ , where $o _ { 0 } = \mathcal { Z } _ { \mathrm { m i n } } ( s _ { 0 } ) , o _ { 1 } = \mathcal { Z } _ { \mathrm { m i n } } ( t _ { 0 } ) , o _ { 2 } = \mathcal { Z } _ { \mathrm { m i n } } ( s _ { 1 } ) \mathrm { a n d } q _ { 0 } = \mathcal { Z } _ { \mathrm { m a x } } ( s _ { 0 } )$ $\bar { q _ { 1 } } = \mathcal { Z } _ { \mathfrak { m a x } } ( t _ { 0 } ) , q _ { 2 } = \mathcal { Z } _ { \mathfrak { m a x } } ( s _ { 1 } )$ . The letters $o _ { 0 } , o _ { 2 }$ and $q _ { 0 } , q _ { 2 }$ are observations of the max-states $s _ { 0 }$ and $s _ { 1 }$ (even indices), so $\Theta ^ { \mathrm { m a x } }$ and $\Theta ^ { \mathrm { m i n } }$ copy them unchanged. Only the letters $o _ { 1 }$ and $q _ { 1 }$ of the min-state $t _ { 0 }$ (odd index) are replaced by blocks of $2 k - 1 = 3$ letters. Let the next state be t with ${ \mathcal { Z } } _ { \operatorname* { m i n } } ( t ) = o$ . In ${ \mathcal { G } } .$ min has the history $O _ { 0 } O _ { 1 } O _ { 2 } O$ at t. In $\mathcal { G } ^ { \prime }$ , after the round of $t _ { 0 }$ and the entry transition to $x _ { t , 1 }$ , we have

$$
\begin{array} { r l } & { h _ { \mathcal { G } ^ { \prime } } ^ { \mathsf { m a x } } ( v ) = \theta ^ { \mathsf { m a x } } ( q _ { 0 } q _ { 1 } q _ { 2 } ) = q _ { 0 } ( q _ { 1 } , \star ) ( q _ { 1 } , \star ) ( q _ { 1 } , \star ) q _ { 2 } , } \\ & { h _ { \mathcal { G } ^ { \prime } } ^ { \mathsf { m i n } } ( v ) = \theta ^ { \mathsf { m i n } } ( o _ { 0 } o _ { 1 } o _ { 2 } ) = o _ { 0 } ( o _ { 1 } , 1 ) ( o _ { 1 } , 1 , \odot ) ( o _ { 1 } , 2 ) o _ { 2 } , } \end{array}
$$

and the min-histories at the slots 1 and 2 of the chain of t are $\Theta ^ { \mathsf { m i n } } \bigl ( o _ { 0 } o _ { 1 } o _ { 2 } \bigr ) ( o , 1 ) \mathrm { a n d } \Theta ^ { \mathsf { m i n } } \bigl ( o _ { 0 } o _ { 1 } o _ { 2 } \bigr ) ( o , 1 ) \bigl ( o , 1 , \odot \bigr ) \bigl ( o , 2 \bigr )$ respectively.

Strategies. Let $\sigma \in \Sigma _ { \mathcal { G } }$ . In $\mathcal { G } ^ { \prime }$ , max moves at the states of $S _ { \sf m a x }$ and at the states of D. At a maxhistory $\Theta ^ { \mathsf { m a x } } ( w )$ ends in a state of $S _ { \mathrm { { m a x } } }$ , we let $\sigma ^ { \prime } ( \theta ^ { \mathsf { m a x } } ( w ) ) = \sigma ( w )$ . At a max-history h ending in a state $d \in D$ the only available action is ⊥, so there is no choice, and $\sigma ^ { \prime } ( h ) = \mathsf { D i r a c } ( \perp )$ . Since $\theta ^ { \mathrm { { m a x } } }$ is injective, $\sigma ^ { \prime }$ is well defined, and σ can be recovered from $\sigma ^ { \prime }$ by $\sigma ( w ) = \sigma ^ { \prime } ( \Theta ^ { \mathsf { m a x } } ( w ) )$ Conversely, every max-strategy of $\mathcal { G } ^ { \prime }$ plays ⊥ at the histories of states of $D _ { : }$ , so it is determined by its values at the histories $\Theta ^ { \mathsf { m a x } } ( w )$ . Hence $\sigma \mapsto \sigma ^ { \prime }$ is a bijection between $\Sigma _ { \mathcal { G } }$ and $\Sigma _ { \mathcal { G } ^ { \prime } }$

For an ordinary min-strategy $\pi ^ { \prime }$ of $\mathcal { G } ^ { \prime }$ , define the prescription dec $( \pi ^ { \prime } )$ by

$$
\operatorname * { d e c } ( \pi ^ { \prime } ) ( w ^ { - } o ) ( t _ { i } ^ { o } ) = \pi ^ { \prime } \big ( \Theta ^ { \mathrm { m i n } } ( w ^ { - } ) \xi _ { o } ^ { \le i } \big ) \qquad ( 1 \le i \le m _ { o } ) ,
$$

where $w ^ { - } o$ is a min-history of $\mathcal { G }$ that ends in o (for states $t \in S _ { \operatorname* { m i n } }$ with $\mathcal { Z } _ { \sf m i n } ( t ) \neq o ,$ , the value dec $( \pi ^ { \prime } ) ( w ^ { - } o ) ( t )$ is never used and is chosen arbitrarily). This is a mixed action in $\begin{array} { r } { \varDelta ( A _ { o } ( i ) ) = } \end{array}$ $\varDelta ( A ( t _ { i } ^ { o } ) )$ ). Conversely, for a prescription π let $\operatorname { e n c } ( \pi )$ be the ordinary strategy of $\mathcal { G } ^ { \prime }$ with

$$
\begin{array} { r } { \operatorname { e n c } ( \pi ) \left( \theta ^ { \mathsf { m i n } } ( w ^ { - } ) \xi _ { o } ^ { \le i } \right) = \pi ( w ^ { - } o ) ( t _ { i } ^ { o } ) \quad ( 1 \le i \le m _ { o } ) , } \end{array}
$$

and an arbitrary element of $\varDelta ( A _ { o } ( i ) )$ for $i ~ > ~ m _ { o }$ (the choice at such a slot is never real). It is ordinary, because its argument depends on the observation word only, and it is legal, because $A _ { o } ( i ) = A ( t _ { i } ^ { o } )$ . By definition, $\operatorname { l e c } ( \operatorname { e n c } ( \pi ) ) = \pi$ , so dec is onto $\pi _ { \mathcal { G } }$ . Conversely, $\operatorname { e n c } ( \operatorname* { d e c } ( \pi ^ { \prime } ) )$ agrees with $\pi ^ { \prime }$ at all slots $i \le m _ { o }$ and can differ from $\pi ^ { \prime }$ only at slots $i > m _ { o }$ , where the choice is never real. We identify strategies that differ only at histories that cannot occur, and ordinary min-strategies of $\mathcal { G } ^ { \prime }$ that differ only at such slots.

Measures. Fix $\sigma \in \Sigma _ { \mathcal { G } }$ and an ordinary $\pi ^ { \prime }$ of $\mathcal { G } ^ { \prime }$ , let $\pi = \operatorname* { d e c } ( \pi ^ { \prime } )$ , and write $P = \mathbb { P } _ { \mathcal { G } } ^ { \sigma , \pi }$ and $P ^ { \prime } = \mathbb { P } _ { \mathcal { G } ^ { \prime } } ^ { \sigma ^ { \prime } , \pi ^ { \prime } }$ . For a prefix $u = s _ { 0 } a _ { 0 } t _ { 0 } b _ { 0 } s _ { 1 } \cdot \cdot \cdot s _ { j }$ of $\mathcal { G }$ that ends in a state of $S _ { \mathrm { { m a x } } }$ (that is, u consists of $j$ rounds), let

$$
V ( u ) = \{ v : v { \mathrm { ~ i s ~ a ~ p r e f i x ~ o f ~ } } \mathcal { G } ^ { \prime } { \mathrm { ~ t h a t ~ e n d s ~ i n ~ a ~ s t a t e ~ o f ~ } } S _ { \operatorname* { m a x } } { \mathrm { ~ a n d ~ p r } } ( v ) = u \} .
$$

The set $V ( u )$ is finite, and every $v \in V ( u )$ consists of $j$ rounds of $\mathcal { G } ^ { \prime }$ , so it has exactly $2 k j$ transitions. Hence no element of $V ( u )$ is a proper prefix of another, and the cylinders ${ \mathsf { C y l } } ( v )$ of $\mathcal { G } ^ { \prime }$ for $v \in V ( u )$ are pairwise disjoint. We first compare the probabilities of cylinders: we show that for every prefix u of $\mathcal { G }$ that ends in a state of $S _ { \mathrm { { m a x } } }$

$$
P \big ( \mathsf { C y l } ( u ) \big ) = \sum _ { v \in V ( u ) } P ^ { \prime } \big ( \mathsf { C y l } ( v ) \big ) .\tag{∗}
$$

The proof is by induction on the number $j$ of rounds of $u .$

Base case. For $u = s _ { 0 }$ we have $V ( u ) = \{ s _ { 0 } \}$ , and both sides of (∗) are 1.

Induction step. Let u end in $s \in S _ { \mathrm { \overline { { m a x } } } }$ , and let $u ^ { + } = u a t b s ^ { \prime }$ be an extension of u by one round of $\mathcal { G } \colon$ max plays $a \in A ( s )$ , the game moves to $t \in S _ { \operatorname* { m i n } }$ , min plays $b \in A ( t )$ , and the game moves to $s ^ { \prime } \in S _ { \mathrm { m a x } }$ . Let $o = \mathcal { Z } _ { \mathrm { m i n } } ( t ) , p = \mathrm { p o s } ( t ) ( \mathrm { s o } t = t _ { p } ^ { o } )$ , and $w ^ { - } = h _ { \mathcal { G } } ^ { \sf m i n } ( u )$ , so that min has the history $w ^ { - } o$ at t. By the definition of $P _ { \mathrm { : } }$

$$
P \big ( \mathrm { C y } 1 ( u ^ { + } ) \big ) = P \big ( \mathrm { C y } 1 ( u ) \big ) \cdot f , \qquad f = \sigma \big ( h _ { \mathcal { G } } ^ { \mathsf { m a x } } ( u ) \big ) ( a ) \delta ( s , a ) ( t ) \pi ( w ^ { - } o ) ( t ) ( b ) \delta ( t , b ) ( s ^ { \prime } ) .
$$

Let $B = \{ \bar { b } = ( b _ { 1 } , \dots , b _ { k } ) \in A _ { o } ( 1 ) \times \dots \times A _ { o } ( k ) : b _ { p } = b \}$ be the set of tuples of slot actions whose real component is b. For $v \in V ( u )$ and $\bar { b } \in B$ , let

$$
v _ { \bar { b } } = v a x _ { t , 1 } b _ { 1 } y _ { 1 } \bot y _ { 2 } b _ { 2 } \cdot \cdot \cdot y _ { k - 1 } \bot x _ { k } b _ { k } s ^ { \prime } ,
$$

where the chain states $y _ { i } ~ \in ~ D$ and $x _ { i } ~ \in ~ X$ are determined by $t , b ,$ , and the position i, by the definition of $\mathcal { G } ^ { \prime }$ . Then $v _ { \bar { b } } \in V ( u ^ { + } )$ , and the map $( v , \bar { b } ) \mapsto v _ { \bar { b } }$ from $V ( u ) \times B$ to $V ( u ^ { + } )$ is a bijection. Indeed, it is injective because v¯ contains v and the actions $b _ { 1 } , \ldots , b _ { k }$ . For surjectivity, let $v ^ { \prime } \in V ( u ^ { + } )$ . Truncating $v ^ { \prime }$ after its first j rounds gives a prefix v with $\mathrm { p r } ( v ) = u$ , that $\mathrm { i s } , v \in V ( u )$ and the last round of $v ^ { \prime }$ traverses the chain of some state t<sup>˜</sup> with real choice <sup>˜</sup>b and ends in $s ^ { \prime } .$ . Since $\mathrm { p r } ( v ^ { \prime } ) = u a \tilde { t } \tilde { b } s ^ { \prime }$ must equal $u ^ { + }$ , we get $\tilde { t } = t$ and ${ \tilde { b } } = b$ . Let <sup>¯</sup>b be the tuple of the actions chosen at the k slots in this round. Then $\bar { b } \in B$ , because its component at the slot $p = \mathrm { p o s } ( t )$ is the real choice $b ,$ and $v ^ { \prime } = v _ { b } ^ { - }$

We compute $P ^ { \prime } ( \mathsf C \mathsf y | ( v _ { \bar { b } } ) )$ by the definition of $P ^ { \prime }$ , as $P ^ { \prime } ( \mathsf { C y l } ( v ) )$ ) times the product of the probabilities of the transitions of the last round. The entry transition has probability $\sigma ^ { \prime } ( h _ { \mathcal { C } ^ { \prime } } ^ { \sf m a x } ( v ) ) ( a ) \delta ^ { \prime } ( s , a ) ( x _ { t , 1 } ) =$ $\sigma ( h _ { \mathcal { G } } ^ { \mathsf { m a x } } ( u ) ) ( a ) \delta ( s , a ) ( t )$ , because $h _ { \mathcal { G } ^ { \prime } } ^ { \mathsf { m a x } } ( v ) = \Theta ^ { \mathsf { m a x } } ( h _ { \mathcal { G } } ^ { \mathsf { m a x } } ( u ) )$ and $\sigma ^ { \prime } ( \theta ^ { \mathsf { m a x } } ( w ) ) = \sigma ( w )$ ). At the slot $i ,$ min plays $b _ { i }$ with probability $\pi ^ { \prime } ( \breve { \Theta } ^ { \mathrm { m i n } } ( w ^ { - } ) \xi _ { \sigma } ^ { \leq i } ) ( b _ { i } ) ^ { * }$ , since this is its history there. All other transitions of the chain are deterministic: max plays ⊥ at the d-states with probability 1 and the slot transitions are Dirac distributions. The exit transition has probability $\delta ^ { \prime } ( x _ { k } , b _ { k } ) ( s ^ { \prime } ) = \delta ( t , b ) ( s ^ { \prime } )$

which does not depend on $b _ { k }$ if $p \ < \ k$ (the real choice b is recorded in the state), and equals $\delta ( t , b _ { p } ) ( s ^ { \prime } ) = \delta ( t , b ) ( s ^ { \prime } ) { \mathrm { i f } } p = k$ . Hence

$$
P ^ { \prime } \big ( \mathsf { C y l } ( v _ { \bar { b } } ) \big ) = P ^ { \prime } \big ( \mathsf { C y l } ( v ) \big ) \cdot \sigma \big ( h _ { \mathcal { G } } ^ { \mathsf { m a x } } ( u ) \big ) ( a ) \delta ( s , a ) ( t ) \cdot \prod _ { i = 1 } ^ { k } \pi ^ { \prime } \big ( \theta ^ { \mathsf { m i n } } ( w ^ { - } ) \xi _ { \sigma } ^ { \le i } \big ) ( b _ { i } ) \cdot \delta ( t , b ) ( s ^ { \prime } ) .
$$

We sum over $\bar { b } \in \ B$ . The product over i splits as $\pi ^ { \prime } ( \theta ^ { \mathsf { m i n } } ( w ^ { - } ) \xi _ { o } ^ { \le p } ) ( b )$ for $i \ = \ p ,$ , which is $\pi ( w ^ { - } o ) ( t ) ( b )$ by the definition of dec, times the factors for $i \neq p .$ For each $i \neq p , \pi ^ { \prime } ( \Theta ^ { \mathsf { m i n } } ( w ^ { - } ) \xi _ { \sigma } ^ { \le i } )$ is a probability distribution on $A _ { o } ( i )$ , so summing over $b _ { i } \in A _ { o } ( i )$ gives 1. Therefore

$$
\sum _ { \bar { b } \in B } P ^ { \prime } \bigl ( \mathsf { C y l } ( v _ { \bar { b } } ) \bigr ) = P ^ { \prime } \bigl ( \mathsf { C y l } ( v ) \bigr ) \cdot f .
$$

Using the bijection above, the induction hypothesis $( * )$ for $u ,$ and the formula for $P ( \mathsf { C y l } ( u ^ { + } ) )$ ,

$$
\sum _ { v ^ { \prime } \in V ( u ^ { + } ) } P ^ { \prime } \big ( \operatorname { C y I } ( v ^ { \prime } ) \big ) = \sum _ { v \in V ( u ) } \sum _ { b \in B } P ^ { \prime } \big ( \operatorname { C y I } ( v _ { b } ) \big ) = f \cdot \sum _ { v \in V ( u ) } P ^ { \prime } \big ( \operatorname { C y I } ( v ) \big ) = f \cdot P \big ( \operatorname { C y I } ( u ) \big ) = P \big ( \operatorname { C y I } ( u ^ { + } ) \big ) .
$$

This proves $( * )$

From $( * )$ to the equality of measures. Every infinite play $\rho ^ { \prime }$ of $\mathcal { G } ^ { \prime }$ consists of complete rounds (entry, chain, exit), so $\operatorname { p r } ( \rho ^ { \prime } )$ is a well-defined play of ${ \mathcal { G } } ,$ , and pr is defined on all plays of $\mathcal { G } ^ { \prime }$ . Fix a finite prefix u of $\mathcal { G }$ that ends in $S _ { \sf m a x }$ and has $j$ rounds. A play $\rho ^ { \prime }$ satisfies $\mathrm { p r } ( \rho ^ { \prime } ) \in \mathsf { C y l } ( u )$ if and only if the prefix v of $\rho ^ { \prime }$ consisting of its first $j$ rounds satisfies $\operatorname { p r } ( v ) = u .$ , that is, $v \in V ( u )$ . Hence

$$
\operatorname { p r } ^ { - 1 } \bigl ( \mathsf { C y l } ( u ) \bigr ) = \bigcup _ { v \in V ( u ) } \mathsf { C y l } ( v ) .
$$

The union is disjoint because all $v \in V ( u )$ have the same length $( 2 k j$ transitions), so two distinct such prefixes are incomparable and their cylinders do not intersect. It is finite because $\mathcal { G } ^ { \prime }$ is finite, so $V ( u )$ is finite. Therefore, by (∗),

$$
P ^ { \prime } \bigl ( \mathrm { p r } ^ { - 1 } ( \mathsf { C y } | ( u ) ) \bigr ) = \sum _ { v \in V ( u ) } P ^ { \prime } \bigl ( \mathsf { C y } | ( v ) \bigr ) = P \bigl ( \mathsf { C y } | ( u ) \bigr ) .
$$

This holds for every cylinder ${ \mathsf { C y l } } ( u )$ of $\mathcal { G }$ (the cylinder of a prefix that ends in $S _ { \mathrm { m i n } }$ is the disjoint union of the cylinders of its one-step extensions, so it is covered as well).

In particular, $\mathrm { p r }$ is measurable. The cylinders of G generate its σ-algebra of plays, and the preimage of each generator is a finite union of cylinders of $\mathcal { G } ^ { \prime }$ , hence measurable. So the push-forward $\mathrm { p r } _ { \# } P ^ { \prime }$ defined by $\mathrm { p r } _ { \# } P ^ { \prime } ( E ) = P ^ { \prime } ( \mathrm { p r } ^ { - 1 } ( E ) )$ , is a probability measure on the plays of ${ \mathcal { G } } .$ . Hence, $\mathrm { p r } _ { \# } P ^ { \prime }$ and $P$ agree on all cylinders. Any two cylinders are either nested or disjoint, so the cylinders together with the empty set are closed under finite intersections, and they generate the σ-algebra of plays. By the uniqueness theorem for probability measures (two probability measures that agree on an intersection-closed generator agree on the generated σ-algebra [5, Thm. 3.3]), $\mathrm { p r } _ { \# } P ^ { \prime } = P$

Returns. We show $X ^ { \prime } ( \rho ^ { \prime } ) = c \cdot X ( \mathrm { p r } ( \rho ^ { \prime } ) )$ on every play $\rho ^ { \prime }$ of $\mathcal { G } ^ { \prime }$ . Let $\left( r _ { n } \right)$ be the reward sequence of $\operatorname { p r } ( \rho ^ { \prime } )$ . The rewards r and $r _ { 2 j }$ $r _ { 2 j + 1 }$ are earned at the transitions with indices $2 k j$ and $2 k j + 2 k - 1$ of $\rho ^ { \prime }$ . For Tot the two sums coincide. For ${ \mathsf { T o t } } _ { N }$ , the first $N ^ { \prime }$ transitions of $\rho ^ { \prime }$ contain exactly the rewards $r _ { 0 } , . . . , r _ { N - 1 } . \mathrm { ~ i f ~ } N = 2 n$ , the last of them, $r _ { 2 n - 1 }$ , is earned at the transition $2 k n - 1$ and $N ^ { \prime } = k N = 2 k n ; { \mathrm { i f ~ } } N = 2 n + 1$ , the last one, $r _ { 2 n } .$ , is earned at the transition 2kn, and $N ^ { \prime } = k ( N - 1 ) + 1 = 2 k n + 1$

For LimAvg, let $\rho ^ { \prime }$ be a play of $\mathcal { G } ^ { \prime }$ and $\rho = \mathrm { p r } ( \rho ^ { \prime } )$ the corresponding play of ${ \mathcal { G } } ,$ with rewards $r _ { 0 } , r _ { 1 } , \ldots$ (all transitions of G that carry a reward are indexed from 0). Let $R = \operatorname* { m a x } | r | .$ , which is finite since the game is finite. Write $\begin{array} { r } { S _ { m } = \sum _ { i < m } r _ { i } } \end{array}$ and $A _ { m } = S _ { m } / m$ for $m \geq 1$ , so that $\ L \mathsf { i m A v g } ( \rho ) \ = \ \operatorname* { l i m } \operatorname* { i n f } _ { m } A _ { m }$ (the case of lim sup is identical). In $\mathcal { G } ^ { \prime }$ the reward $r _ { 2 j }$ is earned at transition $2 k j .$ , the reward $r _ { 2 j + 1 }$ at transition $2 k j + 2 k - 1$ , and all other transitions have reward 0. Let $B _ { n }$ be the average of the first n rewards of $\rho ^ { \prime } .$ , that is, of the rewards earned at the transitions $0 , \ldots , n - 1$ . Then $\mathsf { L i m A v g } ( \rho ^ { \prime } ) = \operatorname* { l i m } \operatorname* { i n f } _ { n } B _ { n }$ , and we claim

$$
\begin{array} { r } { \operatorname* { l i m } _ { n } \operatorname* { i n f } B _ { n } = \frac { 1 } { k } \operatorname* { l i m } _ { m } \operatorname* { i n f } A _ { m } , \qquad \operatorname* { l i m } _ { n } \operatorname* { s u p } B _ { n } = \frac { 1 } { k } \operatorname* { l i m } _ { m } \operatorname* { s u p } A _ { m } . } \end{array}
$$

Step $\boldsymbol { l } \colon \boldsymbol { B } _ { n }$ is close to ${ \scriptstyle { \frac { 1 } { k } } } A _ { 2 j } .$ . Every $n \geq 2 k$ lies in $[ 2 k j , 2 k ( j + 1 ) )$ for a unique $j = j ( n ) \geq 1$ namely $j ( n ) = \lfloor n / ( 2 k ) \rfloor$ ⌋, and we distinguish two cases.

$C a s e n = 2 k j .$ . The first $2 k j$ transitions of $\overline { { \rho ^ { \prime } } }$ are j complete rounds. They contain the rewards $r _ { 0 } , \ldots , r _ { 2 j - 1 }$ (the rewards $r _ { 2 i }$ and $r _ { 2 i + 1 }$ of the round $i < j$ are earned at the transitions 2ki and $2 k i + 2 k - 1$ , both below $2 k j )$ and no others. Hence $\begin{array} { r } { B _ { 2 k j } = S _ { 2 j } / ( 2 k j ) = \frac { 1 } { k } A _ { 2 j } } \end{array}$

• Case $2 k j < n < 2 k ( j + 1 )$ . The reward $r _ { 2 j }$ is earned at the transition $2 k j \le n - 1$ , so it is among the first n rewards. The reward $r _ { 2 j + 1 }$ is earned at the transition $2 k j + 2 k - 1 \geq n .$ so it is not. Together with the rewards of the first j rounds, the sum of the first n rewards is therefore $S _ { 2 j } + r _ { 2 j }$ , and $B _ { n } = ( S _ { 2 j } + r _ { 2 j } ) / n$ . Recall that $| r _ { 2 j } | \le R , | S _ { 2 j } | \le 2 j R$ , and $2 k j < n < 2 k ( j + 1 )$ . Put $\varDelta = n - 2 k j , \mathrm { s o } 0 < \varDelta < 2 k$ . Subtracting and regrouping gives

$$
B _ { n } - { \textstyle \frac { 1 } { k } } A _ { 2 j } = \frac { S _ { 2 j } + r _ { 2 j } } { n } - \frac { S _ { 2 j } } { 2 k j } = \frac { r _ { 2 j } } { n } + S _ { 2 j } \left( \frac { 1 } { n } - \frac { 1 } { 2 k j } \right) = \frac { r _ { 2 j } } { n } - S _ { 2 j } \cdot \frac { A } { 2 k j n } .
$$

We bound the two terms separately.

$$
\begin{array} { r } { \Big | \displaystyle \frac { r _ { 2 j } } { n } \Big | \le \frac { R } { n } < \frac { R } { 2 k j } \qquad \mathrm { s i n c e } \ | r _ { 2 j } | \le R \mathrm { ~ a n d ~ } n > 2 k j , \ } \\ { \Big | S _ { 2 j } \cdot \displaystyle \frac { \Delta } { 2 k j n } \Big | \le 2 j R \cdot \displaystyle \frac { \Delta } { 2 k j n } = \frac { R \Delta } { k n } < \frac { 2 R } { n } < \frac { R } { k j } \quad \mathrm { s i n c e } \ | S _ { 2 j } | \le 2 j R , \ \Delta < 2 k , \mathrm { ~ a n d ~ } n > 2 k j . } \end{array}
$$

Adding the two bounds,

$$
\left| B _ { n } - \textstyle { \frac { 1 } { k } } A _ { 2 j } \right| < \frac { R } { 2 k j } + \frac { R } { k j } = \frac { 3 R } { 2 k j } .
$$

In both cases $\begin{array} { r } { | B _ { n } - \frac { 1 } { k } A _ { 2 j ( n ) } | \ \le \ \frac { 3 R } { 2 k j ( n ) } } \end{array}$ , and the right-hand side tends to 0 as $n  \infty$ , because $j ( n )  \infty$ . Hence the sequences $( B _ { n } ) _ { n }$ and $\textstyle { \Bigl ( } { \frac { 1 } { k } } A _ { 2 j ( n ) } { \Bigr ) } ,$ have the same lim inf and the same lim sup.

Step 2: from $( A _ { 2 j ( n ) } ) _ { n } \ t o \ ( A _ { 2 j } ) _ { j }$ . The map $n \mapsto j ( n )$ is non-decreasing and takes every value $j \geq 1$ exactly 2k times. So $\left( A _ { 2 j ( n ) } \right) _ { n }$ is the sequence $\left( A _ { 2 j } \right) _ { j \geq 1 }$ with every term repeated 2k times, which does not change its lim inf or lim sup. Together with Step 1, lim i $\begin{array} { r } { \operatorname { n f } _ { n } B _ { n } = { \frac { 1 } { k } } \operatorname* { l i m } \operatorname* { i n f } _ { j } A _ { 2 j } } \end{array}$ and lim sup $\begin{array} { r } { \mathfrak { d } _ { n } B = \frac { 1 } { k } } \end{array}$ lim sup<sub>j</sub> A<sub>2j</sub> .

Step 3: from even indices to all indices. For every $m \geq 1$

$$
A _ { m + 1 } - A _ { m } = \frac { S _ { m } + r _ { m } } { m + 1 } - \frac { S _ { m } } { m } = \frac { r _ { m } } { m + 1 } - \frac { S _ { m } } { m ( m + 1 ) } , \qquad \mathrm { s o } \qquad | A _ { m + 1 } - A _ { m } | \leq \frac { R } { m + 1 } + \frac { m R } { m ( m + 1 ) } = \frac { 2 R } { m + 1 } ,
$$

using $| S _ { m } | \leq m R$ . Thus every $A _ { m }$ with m odd is within $2 R / m$ of the even-indexed term $A _ { m - 1 }$ and $2 R / m  0$ . Consequently the sequences $( A _ { m } ) _ { m }$ and $( A _ { 2 \lfloor m / 2 \rfloor } ) _ { m }$ have the same lim inf and lim sup, and by the repetition argument of Step 2 so do $( A _ { 2 \lfloor m / 2 \rfloor } ) _ { m }$ and $( A _ { 2 j } ) _ { j }$ . Therefore lim inf<sub>j</sub> $A _ { 2 j } = \operatorname* { l i m } \operatorname* { i n f } _ { m } A _ { m }$ and lim $\begin{array} { r } { \operatorname* { s u p } _ { j } A _ { 2 j } = \operatorname* { l i m } \operatorname* { s u p } _ { m } A _ { m } } \end{array}$

Combining Steps 2 and 3 gives $\begin{array} { r } { \mathsf { L i m A v g } ( \rho ^ { \prime } ) = \frac { 1 } { k } \mathsf { L i m A v g } ( \rho ) } \end{array}$ (for lim inf, and likewise for lim sup), which is the scaling $\begin{array} { r } { c = \frac { 1 } { k } } \end{array}$ for LimAvg.

Conclusion. Let σ be a strategy of max in ${ \mathcal { G } } .$ and let $\sigma ^ { \prime }$ be its image under the bijection $\sigma \mapsto \sigma ^ { \prime }$ defined above: $\sigma ^ { \prime } ( \theta ^ { \mathsf { m a x } } ( w ) ) = \sigma ( w )$ at the histories of prefixes of $\mathcal { G } ^ { \prime }$ that end in $S _ { \sf m a x }$ (and $\sigma ^ { \prime } =$

Dirac(⊥) at the histories of prefixes that end in D). Let $\pi ^ { \prime }$ be a strategy of min in $\mathcal { G } ^ { \prime }$ , and let $\ker ( \pi ^ { \prime } )$ be the strategy of min in $\mathcal { G }$ that $\pi ^ { \prime }$ induces.

Write $P = P _ { \mathcal { G } } ^ { \sigma , \mathrm { d e c } ( \pi ^ { \prime } ) }$ and $P ^ { \prime } = P _ { c ^ { \prime } } ^ { \sigma ^ { \prime } , \pi ^ { \prime } }$ . Recall that the law of a return is the distribution of the return under the play measure, $\mathcal { L } _ { X } ^ { \sigma , \pi ^ { 5 } } \mathrm { = } P ^ { \sigma , \pi } \circ X ^ { - 1 }$ , and that $( S _ { c } ) _ { \# } ( \mu ) = \mu \circ ( x \mapsto c x ) ^ { - 1 }$ is the push-forward of µ under multiplication by c. We have $X ^ { \prime } = c { \cdot } X$ ◦ pr, that is, $X ^ { \prime } ( \rho ^ { \prime } ) = c X ( \mathrm { p r } ( \rho ^ { \prime } ) )$ for every play $\rho ^ { \prime }$ of $\mathcal { G } ^ { \prime }$ , with $\begin{array} { r } { c > 0 ( c = 1 \mathrm { o r } c = \frac { 1 } { k } ) } \end{array}$ . Let $E \subseteq \mathbb { R }$ be a Borel set. Then

$$
\begin{array} { r l r l } { \mathcal { L } _ { X ^ { \prime } } ^ { \sigma ^ { \prime } , \pi ^ { \prime } } ( E ) = P ^ { \prime } \big ( \{ \rho ^ { \prime } : c X ( \mathrm { p r } ( \rho ^ { \prime } ) ) \in E \} \big ) } & { \mathrm { ~ d e f i n i t i o n ~ o f ~ t h e ~ l a w ~ } } \\ { = P ^ { \prime } \big ( \mathrm { p r } ^ { - 1 } \big ( X ^ { - 1 } ( c ^ { - 1 } E ) \big ) \big ) } & { \mathrm { ~ w h e r e ~ } c ^ { - 1 } E = \{ x : c x \in E \} } \\ { = ( \mathrm { p r } _ { \# } P ^ { \prime } ) \big ( X ^ { - 1 } ( c ^ { - 1 } E ) \big ) } & { \mathrm { ~ d e f i n i t i o n ~ o f ~ t h e ~ p u s h \ ' ~ f o r w a r d ~ p r _ \# ~ } P ^ { \prime } } \\ { = P \big ( X ^ { - 1 } ( c ^ { - 1 } E ) \big ) } & { \mathrm { ~ p r _ \# ~ } P ^ { \prime } = P \mathrm { ~ ( f r o m ~ ( * ) ) ~ } } \\ { = \mathcal { L } _ { X } ^ { \sigma , \mathrm { d e t } ( \pi ^ { \prime } ) } ( c ^ { - 1 } E ) } & { \mathrm { ~ d e f i n i t i o n ~ o f ~ t h e ~ l a w ~ } } \\ { = ( S _ { c } ) _ { \# } \big ( \mathcal { L } _ { X } ^ { \sigma , \mathrm { d e t } ( \pi ^ { \prime } ) } \big ) ( E ) } & { \mathrm { ~ s i n c e ~ } ( S _ { c } ) _ { \# } ( \mu ) ( E ) = \mu \big ( \{ x : c x \in E \} \big ) = \mu ( c ^ { - 1 } E ) \mathrm { ~ f o r ~ e v e r y ~ } \mu . } \end{array}
$$

The sets in the first rows are measurable because pr and X are measurable and $c ^ { - 1 } E$ is Borel. Since E was arbitrary, the first and last entries give

$$
\begin{array} { r } { \mathcal { L } _ { X ^ { \prime } } ^ { \sigma ^ { \prime } , \pi ^ { \prime } } = ( S _ { c } ) _ { \# } \mathcal { L } _ { X } ^ { \sigma , \operatorname* { d e c } ( \pi ^ { \prime } ) } . } \end{array}
$$

Since $\sigma \mapsto \sigma ^ { \prime }$ is a bijection and dec is onto $\pi _ { \mathcal { G } }$ , the sets of laws $\{ \mathcal { L } _ { X ^ { \prime } } ^ { \sigma ^ { \prime } , \pi ^ { \prime } } \}$ and $( S _ { c } ) \# \{ { \mathcal { L } } _ { X } ^ { \sigma , \pi } \}$ coincide for every $\sigma ,$ and the statement on values follows. □

Discounted returns. Lemma A.1 covers ${ \mathsf { T o t } } _ { N }$ , Tot and LimAvg because these returns do not depend on exactly when a reward is earned, apart from the horizon, which we adjust, and a constant factor for the average. A discounted return does: the reward of the t-th transition has weight $\gamma ^ { t }$ The chain gadget delays the exit reward of every min-visit and makes one round take 2k instead of 2 transitions. To preserve discounted returns for every $\gamma ,$ we need a compilation that keeps the timing of G. The idea is to give min the whole prescription as a single action: the available actions at all states of an observation class are the tuples that contain one action for each state of the class, and the state that is actually visited uses its own component. This costs an exponential blow-up of the action sets, but changes neither the states nor the timing.

Proposition A.1 (Discounted returns). Let G be an ac-POSG with min-prescriptions and let $\gamma \in$ (0, 1) be rational. There is an ordinary ac- $P O S G { \mathcal { G } } ^ { \times }$ with the same states, the same timing and the same rewards on corresponding plays, such thatfor every max-strategy $\sigma ,$

$$
\{ \mathcal { L } _ { \mathsf { D i s c } _ { \gamma } } ^ { \sigma , \pi ^ { \prime } } : \pi ^ { \prime } o r d i n a r y i n \mathcal { G } ^ { \times } \} = \{ \mathcal { L } _ { \mathsf { D i s c } _ { \gamma } } ^ { \sigma , \pi } : \pi \in \pi _ { \mathcal { G } } \} .
$$

Hence $\mathsf { V a l } _ { \mathcal { G } \times } ( \varPsi , \mathsf { D i s c } _ { \gamma } ) = \mathsf { V a l } _ { \mathcal { G } } ( \varPsi , \mathsf { D i s c } _ { \gamma } )$ for all law functionals Ψ, with the same discountfactor γ and no rescaling. The action sets $o f { \mathcal { G } } ^ { \times }$ may be exponentially larger than those $o f { \mathcal { G } }$

Proof. Idea. A prescription assigns an independent mixed action to every state of an observation class, but only the action of the state that is actually visited is used. An ordinary strategy cannot name the state, but it can choose a whole tuple of actions, one for each state of the class. The visited state then reads off its own component. Playing the product of the mixed actions on tuples reproduces the prescription, and the marginals of any distribution on tuples form a prescription. Nothing else in the game changes, so the timing and the rewards are the same and every return, in particular $\mathsf { D i s c } _ { \gamma } .$ , is preserved.

Step 1: the game $\mathcal { G } ^ { \times }$ . Let $C _ { o } = \{ t _ { 1 } , \dots , t _ { m } \} \subseteq S _ { \mathsf { m i n } }$ be an observation class of min. In $\mathcal { G } ^ { \times }$ , every state of $C _ { o }$ gets the same set of actions, namely the tuples

$$
A _ { o } ^ { \times } = A ( t _ { 1 } ) \times \cdots \times A ( t _ { m } ) .
$$

For a tuple $\beta = ( \beta _ { 1 } , \ldots , \beta _ { m } ) \in A _ { o } ^ { \times }$ , the state $t _ { i }$ uses only the i-th component:

$$
\delta ^ { \times } ( t _ { i } , \beta ) = \delta ( t _ { i } , \beta _ { i } ) , \qquad r ^ { \times } ( t _ { i } , \beta , s ^ { \prime } ) = r ( t _ { i } , \beta _ { i } , s ^ { \prime } ) .
$$

All other components of $\beta$ are ignored at $t _ { i }$ . States, observations, the max-states with their actions, the transitions of max and their rewards are copied from ${ \mathcal { G } } .$ All states of $C _ { o }$ now have the same actions and $\mathcal { G } ^ { \times }$ is an ac-POSG. The observation functions are unchanged, so the observation histories of both players are the same words in $\mathcal { G }$ and in $\mathcal { G } ^ { \times }$ . In particular, observation histories do not contain actions, so the tuples chosen by min do not appear in them. Hence max-strategies of the two games are the same objects, and σ can be used in both games.

Step 2: strategies. Let w be a min-history that ends in the observation o. A prescription $\pi \in \pi _ { \mathcal { G } }$ assigns to w a mixed action $\pi ( w ) ( t _ { i } ) \in \varDelta ( A ( t _ { i } ) )$ for every $t _ { i } \in C _ { o } .$ . An ordinary strategy $\pi ^ { \prime }$ $\mathcal { G } ^ { \times }$ assigns to w a single distribution $\pi ^ { \prime } ( w ) \in \varDelta ( A _ { o } ^ { \times } )$ ) on tuples. We relate the two as follows.

• From tuples to prescriptions. Let $\operatorname* { m a r } ( \pi ^ { \prime } )$ be the prescription with

$$
\operatorname* { m a r } ( \pi ^ { \prime } ) ( w ) ( t _ { i } ) ( b ) = \sum _ { \substack { \beta \in A _ { \sigma } ^ { \times } : \beta _ { i } = b } } \pi ^ { \prime } ( w ) ( \beta ) \qquad ( b \in A ( t _ { i } ) ) ,
$$

that is, ma $\cdot ( \pi ^ { \prime } ) ( w ) ( t _ { i } )$ is the i-th marginal of $\pi ^ { \prime } ( w )$ . It is a distribution on $A ( t _ { i } )$ , so $\operatorname* { m a r } ( \pi ^ { \prime } ) \in$ $\pi _ { \mathcal { G } }$

• From prescriptions to tuples. For $\pi \in \pi _ { \mathcal { G } }$ , let $\pi ^ { \times }$ be the ordinary strategy that plays the product distribution

$$
\pi ^ { \times } ( w ) ( \beta ) = \prod _ { i = 1 } ^ { m } \pi ( w ) ( t _ { i } ) ( \beta _ { i } ) .
$$

Its i-th marginal is $\pi ( w ) ( t _ { i } )$ , since the other factors sum to 1. Hence mar $( \pi ^ { \times } ) = \pi$

Thus mar maps the ordinary strategies of $\mathcal { G } ^ { \times }$ onto $\pi _ { \mathcal { G } }$ . It is in general not injective: different distributions on tuples, for instance correlated ones, can have the same marginals. This does no harm, because in each step only one coordinate of the tuple is used, and a fresh tuple is drawn in the next step.

Step 3: the projection of plays. A play of $\mathcal { G } ^ { \times }$ is $\rho ^ { \times } = s _ { 0 } c _ { 0 } s _ { 1 } c _ { 1 } \cdot \cdot \cdot$ , where $c _ { j }$ is an action of max if $s _ { j } \in S _ { \mathsf { m a x } } ,$ and a tuple if $s _ { j } \in S _ { \operatorname* { m i n } }$ . Let $\operatorname { p r } ( \rho ^ { \times } )$ be the play of $\mathcal { G }$ in which every tuple $c _ { j } = \beta$ chosen at $s _ { j } = t _ { i }$ is replaced by its component $\beta _ { i }$ . This is a play of ${ \mathcal { G } } .$ , because $\delta ^ { \times } ( t _ { i } , \beta ) =$ $\delta ( t _ { i } , \beta _ { i } )$ . The map pr is measurable and keeps the sequence of states, so for every prefix u of $\mathcal { G } _ { : }$ the set $\mathrm { p r } ^ { - 1 } ( \mathsf C \mathsf y | ( u ) )$ ) is the disjoint union of the cylinders $\mathsf { C y l } ( u ^ { \times } )$ over all prefixes $u ^ { \times }$ of $\mathcal { G } ^ { \times }$ with $\mathrm { p r } ( u ^ { \times } ) = u$

Step 4: the measures agree. Fix σ and an ordinary $\pi ^ { \prime }$ of $\mathcal { G } ^ { \times }$ , and write $\pi = \operatorname* { m a r } ( \pi ^ { \prime } )$ . We show

$$
\mathbb { P } _ { \mathcal { G } ^ { \times } } ^ { \sigma , \pi ^ { \prime } } \bigl ( \mathrm { p r } ^ { - 1 } ( \mathsf { C y l } ( u ) ) \bigr ) = \mathbb { P } _ { \mathcal { G } } ^ { \sigma , \pi } \bigl ( \mathsf { C y l } ( u ) \bigr ) \quad \mathrm { f o r e v e r y p r e f i x ~ } u \mathrm { ~ o f ~ } \mathcal { G } ,\tag{∗}
$$

by induction on the length of u. For $u = s _ { 0 }$ both sides are 1. For the induction step, let u be a prefix that ends in a state s, let w be the observation-history of the player who moves from s after u (it is the same in both games, since it only depends on the states of u), and let $u b s ^ { \prime }$ be an extension of $u .$

$s \in S _ { \mathsf { m a x } }$ . The extension has the same probability $\sigma ( w ) ( b ) \delta ( s , b ) ( s ^ { \prime } )$ in both games, because max, its actions and its transitions are not changed. Since this factor does not depend on how u was reached, (∗) for u gives (∗) for u $b s ^ { \prime }$

$s = t _ { i } \in S _ { \operatorname* { m i n } }$ . Let $u ^ { \times }$ be a prefix of $\mathcal { G } ^ { \times }$ with $\mathrm { p r } ( u ^ { \times } ) = u$ . The cylinders of $\mathcal { G } ^ { \times }$ that project to u $\textit { b s } ^ { \prime }$ are those of $u ^ { \times } \beta s ^ { \prime }$ with $\beta _ { i } = b ;$ , and

$$
\mathbb { P } _ { \mathcal { G } ^ { \times } } \left( \mathsf { C y l } ( u ^ { \times } \beta s ^ { \prime } ) \right) = \mathbb { P } _ { \mathcal { G } ^ { \times } } \left( \mathsf { C y l } ( u ^ { \times } ) \right) \cdot \pi ^ { \prime } ( w ) ( \beta ) \cdot \delta ( t _ { i } , b ) ( s ^ { \prime } ) .
$$

We sum over all $\beta$ with $\beta _ { i } = b$ . By the definition of mar, the sum of the factors $\pi ^ { \prime } ( w ) ( \beta )$ is $\pi ( w ) ( t _ { i } ) ( b )$ . This value depends only on w and $t _ { i }$ , and not on $u ^ { \times }$ . Summing then over all $u ^ { \times }$ with $\mathrm { p r } ( u ^ { \times } ) = u$ gives

$$
\mathbb { P } _ { \mathcal { G } \times } \left( \mathrm { p r } ^ { - 1 } ( \mathsf { C y } | ( u b s ^ { \prime } ) ) \right) = \mathbb { P } _ { \mathcal { G } \times } \left( \mathrm { p r } ^ { - 1 } ( \mathsf { C y } | ( u ) ) \right) \cdot \pi ( w ) ( t _ { i } ) ( b ) \delta ( t _ { i } , b ) ( s ^ { \prime } ) ,
$$

and by (∗) for u and the definition of $\mathbb { P } _ { \mathcal { G } } .$ , the right-hand side equals $\mathbb { P } _ { \mathcal { G } } ^ { \sigma , \pi } ( \mathsf { C y l } ( u b s ^ { \prime } ) )$

We now pass from cylinders to all measurable sets of plays. Since pr is measurable (Step 3), $\mu _ { 1 } =$ $\mathrm { p r } _ { \# } \mathbb { P } _ { \mathcal { G } ^ { \times } } ^ { \sigma , \pi ^ { \prime } }$ , defined by $\mu _ { 1 } ( B ) = \mathbb { P } _ { \mathcal { G } \times } ^ { \sigma , \pi ^ { \prime } } ( \mathrm { p r } ^ { - 1 } ( B ) )$ , is a probability measure on the plays of $\mathcal { G }$ . Let $\mu _ { 2 } = \mathbb { P } _ { \mathcal { G } } ^ { \sigma , \operatorname* { m a r } ( \pi ^ { \prime } ) }$ , also a probability measure on the plays of ${ \mathcal { G } } .$ . By the definition of $\mu _ { 1 }$ , equation (∗) says exactly that

$$
\mu _ { 1 } ( { \mathsf { C y l } } ( u ) ) = \mu _ { 2 } ( { \mathsf { C y l } } ( u ) ) \quad { \mathrm { f o r ~ e v e r y ~ p r e f i x ~ } } u { \mathrm { ~ o f ~ } } { \mathcal { G } } .
$$

Two cylinder sets are either disjoint or nested, so their intersection is again a cylinder set or empty: the cylinders form a π-system. It generates the σ-algebra on plays, which is by definition the one generated by the cylinders. The family of sets on which two probability measures agree is closed under complements and countable disjoint unions, so it is a λ-system. By Dynkin’s π-λ theorem, it contains the σ-algebra generated by any π-system it contains. Hence $\mu _ { 1 } = \mu _ { 2 }$ , that is,

$$
\mathrm { p r } _ { \# } \mathbb { P } _ { \mathcal { G } ^ { \times } } ^ { \sigma , \pi ^ { \prime } } = \mathbb { P } _ { \mathcal { G } } ^ { \sigma , \operatorname* { m a r } ( \pi ^ { \prime } ) } .
$$

Step $5 \colon$ the returns agree. By the choice of $r ^ { \times }$ , every transition of $\rho ^ { \times }$ earns exactly the reward of the corresponding transition of $\operatorname { p r } ( \rho ^ { \times } )$ , and both plays have their transitions at the same positions. So both plays have the same reward sequence $( r _ { t } ) _ { t \geq 0 }$ , and for every return X that is a function of the reward sequence, in particular for

$$
X = \mathsf { D i s c } _ { \gamma } = \sum _ { t \geq 0 } \gamma ^ { t } r _ { t }
$$

we have $X ( \rho ^ { \times } ) = X ( \mathrm { p r } ( \rho ^ { \times } ) )$ for every play. We stress that no discount factor is changed: the t-th reward carries the weight $\gamma ^ { t }$ in both games, because $\mathcal { G } ^ { \times }$ has the same timing. Combining this with Step 4,

$$
\begin{array} { r } { \mathcal { L } _ { X } ^ { \sigma , \pi ^ { \prime } } = \mathbb { P } _ { g ^ { \times } } ^ { \sigma , \pi ^ { \prime } } \circ ( X \circ \mathrm { p r } ) ^ { - 1 } = \left( \mathrm { p r } _ { \# } \mathbb { P } _ { g ^ { \times } } ^ { \sigma , \pi ^ { \prime } } \right) \circ X ^ { - 1 } = \mathbb { P } _ { \mathcal { G } } ^ { \sigma , \operatorname* { m a r } ( \pi ^ { \prime } ) } \circ X ^ { - 1 } = \mathcal { L } _ { X } ^ { \sigma , \operatorname* { m a r } ( \pi ^ { \prime } ) } . } \end{array}
$$

Conclusion. For every max-strategy $\sigma ,$ Step 5 shows that every law of an ordinary strategy $\pi ^ { \prime }$ of $\mathcal { G } ^ { \times }$ is a law of the prescription mar $( \pi ^ { \prime } )$ of $\mathcal { G } .$ . Conversely, Step 2 shows that every prescription π is $\operatorname* { m a r } ( \pi ^ { \times } )$ , so its law $\mathcal { L } _ { X } ^ { \sigma , \pi } = \mathcal { L } _ { X } ^ { \sigma , \pi }$ is a law of $\mathcal { G } ^ { \times }$ under an ordinary strategy. Hence the two sets of laws coincide. The value of a game is determined by these sets (it is the same expression in $\varPsi$ of the laws, whatever the order of the optimization over $\sigma$ and over the strategy of min), and thus $\mathsf { V a l } _ { \mathcal { G } \times } ( \varPsi , X ) = \mathsf { V a l } _ { \mathcal { G } } ( \varPsi , X )$ for all $\varPsi$ . Finally, the only blow-up is in the action sets: $\begin{array} { r } { | A _ { o } ^ { \times } | = \prod _ { i \leq m } \left| A ( t _ { i } ) \right. } \end{array}$ for a class of m states. □

Example A.1 (Chain compilation, and why it does not preserve $\mathsf { D i s c } _ { \gamma } )$ . Let $\mathcal { G }$ be the game of Figure 3(a). From $s _ { 0 } .$ , the game moves to $t _ { 1 }$ or t<sub>2</sub> with probability $\frac { 1 } { 2 }$ each. The two min-states have the same observation $^ { O , }$ so they form one class $C _ { o }$ with $k = 2$ . In $t _ { 1 }$ the action r earns 1 and l earns 0, and in $t _ { 2 }$ the rewards are the other way round. The play then stays in the sink z.

Why prescriptions matter. With a prescription, min chooses an action for each state of $C _ { o }$ separately. The prescription $( r , l )$ , which plays $r$ in $t _ { 1 }$ and l in $t _ { 2 } ,$ earns 1 with probability 1. An ordinary strategy cannot do this: the histories at $t _ { 1 }$ and $t _ { 2 }$ look the same, so it plays r with the same probability $p$ in both states, and it earns 1 with probability $\textstyle { \frac { 1 } { 2 } } p + { \frac { 1 } { 2 } } ( 1 - p ) = { \frac { 1 } { 2 } }$ for every $p .$ In general, a prescription with $\pi ( t _ { 1 } ) ( r ) = p _ { 1 }$ and $\pi ( t _ { 2 } ) ( r ) = p _ { 2 }$ earns 1 with probability ${ \textstyle \frac { 1 } { 2 } } p _ { 1 } + { \textstyle \frac { 1 } { 2 } } ( 1 - p _ { 2 } )$ , which takes every value in [0, 1]. The compilation must therefore let an ordinary strategy of the new game reproduce all such prescriptions.

![](images/bec1b7b5d355fe6454ece7ec4a3a98793f525d59f33a17a68d994fbd93dcfa80.jpg)  
(a) G with prescriptions

![](images/b4def18ac56de19567f32e15cde88964e5c5a61a8f790ddbaddf07701e2d006b.jpg)  
(b) The chain game $\mathcal { G } ^ { \prime } \left( k = 2 \right)$

![](images/a966ca0707098885bf88f098e2b1bedaf78a411afa113d9f76447aa9ca4482e3.jpg)  
(c) Position and discount weight of the rewards $r _ { 0 } , \ldots , r _ { 3 }$ in $\mathcal { G }$ and in $\mathcal { G } ^ { \prime }$  
Figure 3: Example A.1. (a) A game $\mathcal { G }$ with two min-states with the same observation. (b) Its chain compilation for $k = 2$ . Both players observe the same along the two chains. The shaded rectangles are the slots where the real choice of min is made, the choice at the other slot is ignored (or recorded in the state). (c) The i-th reward of $\mathcal { G }$ is earned at the transition $2 k j$ or $2 k j + 2 k - 1$ of $\mathcal { G } ^ { \prime } \left( i = 2 j \right.$ or $2 j + 1 )$

The chain. Figure 3(b) shows $\mathcal { G } ^ { \prime }$ . From $t _ { 1 }$ and from $t _ { 2 } ,$ , the game continues in a chain of $2 k - 1 = 3$ transitions, in which min moves at two slots. In the chain of $t _ { 1 }$ , the first slot is the real choice, so min plays the real action $b \in \{ l , r \}$ there, and the state $d _ { t _ { 1 } , b , 1 }$ records it; the second slot is ignored. In the chain of $t _ { 2 } .$ , the second slot is the real choice and the first is ignored. Both chains look the same to min: it observes $( o , 1 )$ , then $( o , 1 , \circledcirc )$ , then $( o , 2 )$ , whichever state was visited. An ordinary strategy of $\mathcal { G } ^ { \prime }$ is therefore a pair of mixed actions, one for each slot. It realizes the prescription that uses the first mixed action in $t _ { 1 }$ and the second in $t _ { 2 }$ . For instance, playing r at the first slot and l at the second realizes $( r , l )$

Strategies $o f { \mathcal { G } } ^ { \prime }$ and prescriptions coincide. Let w be a min-history of $\mathcal { G }$ that ends in the observation o. When the play enters the chain, the histories of min in $\mathcal { G } ^ { \prime }$ continue as $w \cdot ( o , 1 ) \cdot ( o , 1 , \odot ) \cdot ( o , 2 )$ whichever of $t _ { 1 } , t _ { 2 }$ was visited, because the observations along the two chains are identical. An ordinary strategy $\pi ^ { \prime }$ of $\mathcal { G } ^ { \prime }$ therefore fixes, after w, one mixed action $\mu _ { 1 }$ for the first slot and one mixed action $\mu _ { 2 }$ for the second slot, and the second cannot depend on the action played at the first, since strategies only see observations. In the chain of $t _ { i }$ , the action of the i-th slot is the real action of min in $t _ { i }$ and the other slot is ignored. Hence $\pi ^ { \prime }$ induces the prescription $\pi ( w ) ( t _ { i } ) = \mu _ { i }$ for $i = 1 , 2$ . Conversely, given a prescription π, play $\mu _ { i } = \pi ( \boldsymbol { w } ) ( t _ { i } )$ at the i-th slot. In the chain of $t _ { i } ,$ the real action is then drawn from π $\mathbf { \nabla } \cdot ( w ) ( t _ { i } )$ at the i-th slot, exactly as in G. So every prescription arises from some ordinary strategy of $\mathcal { G } ^ { \prime }$ (for general k, from k mixed actions, one per slot).

![](images/b91e1223952f9162b26176bb23bfaebb02e77ba4a750f629b1c69923b5e3b8b3.jpg)  
(a) The game ${ \mathcal { G } } ^ { \times }$

![](images/1dbb799ec4ff32ed6cf6aeed89ae2b8ff43338b381707e3b11d71d02f79c0cd4.jpg)  
(b) An ordinary strategy of ${ \mathcal { G } } ^ { \times }$

![](images/f03a81d5c927457f1f103040912f490d654bbfa54cd36145e60ee2090cf849d1.jpg)  
(c) Every reward is earned at the same transition in G and in ${ \mathcal { G } } ^ { \times }$ , so the discount weights $\gamma ^ { i }$ agree.

Figure 4: Example A.2. (a) The game $\mathcal { G } ^ { \times }$ for the game $\mathcal { G }$ of Figure $3 ( { \mathrm { a } } ) { \mathrm { : } }$ both states of $C _ { o }$ have the four actions $( \beta _ { 1 } , \beta _ { 2 } ) \in \{ l , r \} ^ { 2 }$ , and $t _ { i }$ uses only $\beta _ { i }$ (an edge labelled $( l , * )$ stands for the two tuples with $\beta _ { 1 } ~ = ~ l )$ . (b) The ordinary strategy $\pi ^ { \prime }$ that plays the product of $\begin{array} { r } { \pi ( t _ { 1 } ) = ( \frac { 1 } { 4 } , \frac { 3 } { 4 } ) } \end{array}$ and $\pi ( t _ { 2 } ) = ( \frac { 1 } { 2 } , \frac { 1 } { 2 } )$ on tuples; its marginals are the prescription π. (c) The timing is unchanged.

The laws agree. The objectives of corresponding plays of $\mathcal { G } ^ { \prime }$ and $\mathcal { G }$ are related by $X ^ { \prime } = c \cdot X$ with c as in Lemma A.1, since both plays have the same rewards in the same order, and the strategies correspond as above. Hence the sets of laws of $\mathcal { G } ^ { \prime }$ and $\mathcal { G }$ correspond under $( S _ { c } ) _ { \# }$

Discounted returns. For $\mathsf { T o t } _ { N ^ { \prime } }$ , Tot, and LimAvg, only the order of the rewards matters, so the identity $X ^ { \prime } = c \cdot X$ on corresponding plays holds. For discounted returns, the position of a reward matters. A round of G takes 2 transitions, while a round of $\mathcal { G } ^ { \prime }$ takes $2 k = 4$ , and Figure 3(c) shows where the first four rewards of a play land: $r _ { 0 } , r _ { 1 } , r _ { 2 } , r _ { 3 }$ are earned at the transitions $0 , 3 , 4 , 7$ of $\mathcal { G } ^ { \prime }$ instead of $0 , 1 , 2 , 3$ . With unchanged rewards, the discounted return of a play of $\mathcal { G } ^ { \prime }$ is $r _ { 0 } + \gamma ^ { \prime 3 } r _ { 1 } +$ $\gamma ^ { \prime 4 } r _ { 2 } + \gamma ^ { \prime 7 } r _ { 3 } + \cdot \cdot \cdot$ , while that of the corresponding play of $\mathcal { G }$ is $r _ { 0 } + \gamma r _ { 1 } + \gamma ^ { 2 } r _ { 2 } + \gamma ^ { 3 } r _ { 3 } + \cdot \cdot \cdot$ . These cannot be proportional for all reward sequences: comparing the weights of $r _ { 0 }$ forces the constant to be 1, and then $r _ { 1 }$ and $r _ { 2 }$ would need $\gamma ^ { \prime 3 } = \gamma$ and $\gamma ^ { \prime 4 } = \gamma ^ { 2 }$ simultaneously, which is impossible for $\gamma < 1$ . So no discount factor $\gamma ^ { \prime }$ and no constant c make the chain gadget preserve Disc<sub>γ</sub> with the same rewards.

Example A.2 (Vector actions preserve Disc ). Take again the game $\mathcal { G }$ of Figure 3(a). The game $\mathcal { G } ^ { \times }$ of Proposition A.1 has the same states, transitions and rewards as $\mathcal { G } _ { : }$ but both states of $C _ { o }$ have the four actions $\beta = ( \beta _ { 1 } , \beta _ { 2 } ) \in \{ l , r \} ^ { 2 }$ (Figure 4(a)). The state $t _ { i }$ plays $\beta _ { i }$ and ignores the other component. For the prescription π with $\textstyle \pi ( t _ { 1 } ) = ( { \frac { 1 } { 4 } } , { \frac { 3 } { 4 } } )$ and $\pi ( t _ { 2 } ) = ( \frac { 1 } { 2 } , \frac { 1 } { 2 } )$ , the ordinary strategy $\pi ^ { \times }$ plays the product distribution on tuples (Figure 4(b)), and its marginals are $\pi ( t _ { 1 } )$ $\pi ( t _ { 2 } )$

Same laws in $\mathcal { G }$ and $\mathcal { G } ^ { \times }$ . A play of $\mathcal { G }$ has two steps before it reaches the sink $z ,$ where all further rewards are 0. Counting steps from 0, step 0 is the move $s _ { 0 }  t _ { i } .$ , and step 1 is the move of min from $t _ { i } ~ \mathrm { t o } ~ z .$ The only nonzero reward is the reward 1 of that second step, so $\mathsf { D i s c } _ { \gamma }$ takes only the values 0 and $\gamma ^ { 1 } = \gamma$ , and the law of $\mathsf { D i s c } _ { \gamma }$ is determined by the probability that min earns 1.

In $\mathcal { G }$ under the prescription π, with $\textstyle \pi ( t _ { 1 } ) = ( { \frac { 1 } { 4 } } , { \frac { 3 } { 4 } } )$ and $\pi ( t _ { 2 } ) = ( \frac { 1 } { 2 } , \frac { 1 } { 2 } )$ , min earns 1 in $t _ { 1 }$ by playing r (probability $\frac { 3 } { 4 } )$ and in $t _ { 2 }$ by playing l (probability $\textstyle { \frac { 1 } { 2 } } )$ . Each state is reached with probability $\frac { \mathrm { i } } { 2 } .$ , so min earns 1 with probability $\begin{array} { r } { \frac { 1 } { 2 } \cdot \frac { 3 } { 4 } + \frac { 1 } { 2 } \cdot \frac { 1 } { 2 } = \frac { 5 } { 8 } } \end{array}$

In $\mathcal { G } ^ { \times }$ under the product strategy $\pi ^ { \times } = \pi ( t _ { 1 } ) \otimes \pi ( t _ { 2 } )$ , the tuples $( l , l ) , ( l , r ) , ( r , l ) , ( r , r )$ have probabilities $\textstyle { \frac { 1 } { 8 } } , { \frac { 1 } { 8 } } , { \frac { 3 } { 8 } } , { \frac { 3 } { 8 } }$ . The state $t _ { 1 }$ uses the first component, which is r with probability $\textstyle { \frac { 3 } { 8 } } + { \frac { 3 } { 8 } } = { \frac { 3 } { 4 } }$ and $t _ { 2 }$ uses the second, which is l with probability $\textstyle { \frac { 1 } { 8 } } + { \frac { 3 } { 8 } } = { \frac { 1 } { 2 } }$ . So min earns 1 with probability $\frac { 5 } { 8 }$ as well.

In both games, the earning happens in step 1, so the timing is unchanged, and $\mathsf { D i s c } _ { \gamma }$ has the law $\begin{array} { r } { \frac { 3 } { 8 } \mathsf { D i r a c } ( 0 ) + \frac { 5 } { 8 } \mathsf { D i r a c } ( \gamma ) } \end{array}$ for every $\gamma \in ( 0 , 1 )$ , with no rescaling (Figure 4(c)).

The price of $\mathcal { G } ^ { \times }$ is its action sets, whose size is $\textstyle \prod _ { i } \left| A ( t _ { i } ) \right|$ for a class of k states. The chain gadget above has polynomial size, but it changes the timing and therefore preserves Tot<sub>N</sub>, Tot and LimAvg only. Both constructions show that ordinary ac-POSGs and ac-POSGs with prescriptions are equally expressive for all four returns.

## B Proofs for Section 4

Throughout, the set of plays of a model carries the σ-algebra generated by the cylinder sets $\mathsf { C y l } ( u )$ of finite prefixes u. Let C be the family of cylinder sets, that is, sets of plays with a prescribed finite prefix. The intersection of two cylinder sets is again a cylinder set or empty, so $\mathcal { C }$ is closed under finite intersections (a π-system). It generates the σ-algebra on plays. By the uniqueness theorem for probability measures (a consequence of Dynkin’s $\pi - \lambda$ theorem), two probability measures that agree on $\mathcal { C }$ are therefore equal. For a measurable map $f$ and a measure P, we write $f _ { \# } \mathbb { P } = \mathbb { P } \mathfrak { c }$ ◦ $f ^ { - 1 }$ . Strategies only matter on histories that can occur and at which the player moves; we identify strategies that differ only on other histories, and we call an entry of an assignment used if it can influence the play (for the environment of an RPOMDP, this is the entry $\left( { { s _ { t } } , { a _ { t } } } \right)$ of the assignment chosen in step t).

## B.1 Proof of Lemma 1

Let $\mathcal { G } = \mathfrak { G } ( \mathcal { M } )$ . Every play of $\mathcal { G }$ has the form

$$
\rho ^ { \prime } = s _ { 0 } a _ { 0 } \left( s _ { 0 } , a _ { 0 } \right) v _ { 0 } s _ { 1 } a _ { 1 } \left( s _ { 1 } , a _ { 1 } \right) v _ { 1 } s _ { 2 } \cdot \cdot \cdot \qquad \left( a _ { i } \in A ( s _ { i } ) , v _ { i } \in V _ { s _ { i } , a _ { i } } \right) ,
$$

and $\mathsf { p r o j } ( \rho ^ { \prime } ) = s _ { 0 } a _ { 0 } s _ { 1 } a _ { 1 } \cdot \cdot \cdot$ denotes the play of $\mathcal { M }$ obtained by erasing the min-states and the vertex choices.

Histories. Let proj be the map that erases the min-states and the vertex choices, defined on prefixes and on plays of G, so that it sends a play $\rho ^ { \prime }$ of $\mathcal { G }$ to the play $\mathsf { p r o j } ( \rho ^ { \prime } ) = s _ { 0 } a _ { 0 } s _ { 1 } a _ { 1 } \cdot \cdot \cdot$ of $\mathcal { M }$ . For a prefix $u = s _ { 0 } a _ { 0 } \cdot \cdot \cdot s _ { t }$ of $\mathcal { M } .$ , let

$$
h ^ { \mathsf { a } } ( u ) = { \mathcal Z } _ { \mathsf { a } } ( s _ { 0 } ) \cdot \cdot \cdot { \mathcal Z } _ { \mathsf { a } } ( s _ { t } ) \quad \mathrm { a n d } \quad h ^ { \mathsf { e } } ( u ) = { \mathcal Z } _ { \mathsf { e } } ( s _ { 0 } ) \cdot \cdot \cdot { \mathcal Z } _ { \mathsf { e } } ( s _ { t } ) .
$$

For a word $w = o _ { 0 } \cdot \cdot \cdot o _ { t } , \operatorname { l e t } \eta ( w ) = o _ { 0 } o _ { 0 } o _ { 1 } o _ { 1 } \cdot \cdot \cdot o _ { t } o _ { t }$ be the word in which every letter is doubled, and let $\eta ^ { - } ( w ) = o _ { 0 } o _ { 0 } \cdot \cdot \cdot o _ { t - 1 } o _ { t - 1 } o _ { t }$ be the word in which every letter except the last one is doubled. The agent observes $\mathcal { Z } _ { \mathsf { a } } ( s _ { i } )$ both in $s _ { i }$ and in $( s _ { i } , a _ { i } )$ , and the environment observes $\mathcal { Z } _ { \mathsf { e } } ( s _ { i } )$ at both, so for every prefix $u ^ { \prime }$ of G with $\mathsf { p r o j } ( u ^ { \prime } ) = u$ the histories are as follows. If $u ^ { \prime }$ ends in the max-state $s _ { t } ,$ the max-history is $\eta ^ { - } ( h ^ { \mathsf { a } } ( u ) )$ ). If $u ^ { \prime }$ is extended to the min-state $\left( { { s _ { t } } , { a _ { t } } } \right)$ , the min-history is $\eta ( h ^ { \mathsf { e } } ( u ) )$ . The maps η and $\eta ^ { - }$ are injective, and neither history depends on the vertex choices.

Strategy maps. For $\sigma \in \Sigma _ { \mathcal { M } }$ , let $\sigma ^ { \mathcal { G } } ( \eta ^ { - } ( h ) ) = \sigma ( h )$ ; at all other observation histories, max does not move. This is a bijection between agent strategies and max-strategies. For an observation history $h ,$ write last(h) for its last observation. An environment strategy π assigns to every history h and every pair $( s , a )$ with $\mathcal { Z } _ { \sf e } ( s ) = \log ( h )$ and $a \in A ( s )$ a distribution $\pi ( h ) ( s , a ) \in \mathcal { P } ( s , a )$

For a min-strategy $\pi ^ { \prime } .$ , let $T ( \pi ^ { \prime } )$ be the environment strategy with

$$
\varGamma ( \pi ^ { \prime } ) ( h ) ( s , a ) = \sum _ { v \in V _ { s , a } } \pi ^ { \prime } ( \eta ( h ) ) ( ( s , a ) ) ( v ) \cdot v
$$

for all h and all $( s , a )$ with $\mathcal { Z } _ { \sf e } ( s ) = \mathrm { l a s t } ( h )$ and $a \in A ( s )$ . This is a convex combination of the vertices of $\mathcal { P } ( s , a )$ , so $\varGamma ( \pi ^ { \prime } ) ( h ) ( s , a ) \in \mathcal { P } ( s , a )$

Conversely, since ${ \mathcal { P } } ( s , a ) = \operatorname { c o n v } ( V _ { s , a } )$ , we fix for every pair $( s , a )$ and every $p \ \in \ { \mathcal { P } } ( s , a )$ a distribution $\mu _ { s , a } ( p )$ over $V _ { s , a }$ with

$\begin{array} { r } { \sum _ { v } \mu _ { s , a } ( p ) ( v ) \cdot v = p . } \end{array}$ . Given an environment strategy π, we define the min-strategy $\pi ^ { \prime }$ by

$$
\pi ^ { \prime } ( \eta ( h ) ) ( ( s , a ) ) = \mu _ { s , a } ( \pi ( h ) ( s , a ) )
$$

for all h and all $( s , a )$ with $\mathcal { Z } _ { \sf e } ( s ) = \mathrm { l a s t } ( h )$ . This is well defined because η is injective and the min-history at $( s , a )$ is $\eta ( h ^ { \mathsf { e } } ( u ) )$ , whose last letter is $\mathcal { Z } _ { \mathtt { e } } ( s )$ . Then, for all such h, s, and a,

$$
\varGamma ( \pi ^ { \prime } ) ( h ) ( s , a ) = \sum _ { v \in V _ { s , a } } \mu _ { s , a } \bigl ( \pi ( h ) ( s , a ) \bigr ) ( v ) \cdot v = \pi ( h ) ( s , a ) ,
$$

so $T ( \pi ^ { \prime } ) = \pi .$

Projection of measures Let ${ \mathsf { p r o j } } _ { \# } \mu ( B ) = \mu ( { \mathsf { p r o j } } ^ { - 1 } ( B ) )$ , where $\mathsf { p r o j } ^ { - 1 } ( B ) = \{ \rho ^ { \prime } \mid \mathsf { p r o j } ( \rho ^ { \prime } ) \in$ $B \}$ is the set of plays of $\mathcal { G }$ that project into B, for a measure $\mu$ on plays of G and a measurable set B of plays of $\mathcal { M }$ . We show $\mathsf { p r o j } _ { \# } \mathbb { P } _ { \mathcal { G } } ^ { \sigma ^ { \mathcal { G } } , \pi ^ { \prime } } = \mathbb { P } _ { \mathcal { M } } ^ { \sigma , \Gamma ( \pi ^ { \prime } ) }$ for all σ and $\pi ^ { \prime }$ . Cylinder sets form a π-system that generates the σ-algebra on plays, so two probability measures that agree on all cylinder sets are equal. It therefore suffices to show $\mathbb { P } _ { \mathcal { M } } ^ { \sigma , T ( \pi ^ { \prime } ) } ( \mathsf { C y l } ( u ) ) = \mathbb { P } _ { \mathcal { G } } ^ { \sigma ^ { \mathcal { G } } , \pi ^ { \prime } } ( \mathsf { p r o j } ^ { - 1 } ( \mathsf { C y l } ( u ) ) )$ for every prefix u of $\mathcal { M } _ { : }$ , which we prove by induction on the length of $u = s _ { 0 } a _ { 0 } \cdot \cdot \cdot s _ { t }$ . For $u = s _ { 0 } ,$ , both sides are 1. For the induction step, consider $\boldsymbol { u } \boldsymbol { a } \boldsymbol { s } ^ { \prime }$ . The set $\mathsf { p r o j } ^ { - 1 } ( \mathsf { C y l } ( u a s ^ { \prime } ) )$ is the disjoint union of the cylinders $\mathsf { C y l } \left( u ^ { \prime } a \left( s _ { t } , a \right) v s ^ { \prime } \right)$ over all prefixes $u ^ { \prime }$ of $\mathcal { G }$ with $\mathsf { p r o j } ( u ^ { \prime } ) = u$ and all $v \in V _ { s _ { t } , a }$ , and

$$
\mathbb { P } _ { \mathcal { G } } \left( \operatorname { C y } \lfloor \left( u ^ { \prime } a \left( s _ { t } , a \right) v s ^ { \prime } \right) \right) = \mathbb { P } _ { \mathcal { G } } ( \operatorname { C y } \lfloor \left( u ^ { \prime } \right) ) \cdot \sigma ( h ^ { a } ( u ) ) ( a ) \cdot \pi ^ { \prime } ( \eta ( h ^ { \sf e } ( u ) ) ) ( ( s _ { t } , a ) ) ( v ) \cdot v ( s ^ { \prime } )
$$

Summing over v turns the last two factors into $p ( s ^ { \prime } )$ for the point

$$
p = T ( \pi ^ { \prime } ) ( h ^ { \mathsf { e } } ( u ) ) ( s _ { t } , a ) \in { \mathcal { P } } ( s _ { t } , a ) ,
$$

and summing over $u ^ { \prime }$ replaces $\mathbb { P } _ { \mathcal { G } } ( \mathsf { C y l } ( u ^ { \prime } ) )$ by $\mathbb { P } _ { \mathcal { G } } ( \mathsf { p r o j } ^ { - 1 } ( \mathsf { C y l } ( u ) ) )$ , which equals $\mathbb { P } _ { \mathcal { M } } ( \mathsf { C y l } ( u ) )$ ) by the induction hypothesis. The result is exactly $\mathbb { P } _ { \mathcal { M } } ( \mathsf { C y l } ( u a s ^ { \prime } ) )$

Returns. Let $( r _ { i } ) _ { i \geq 0 }$ be the reward sequence of proj $( \rho ^ { \prime } )$ . The reward sequence of $\bar { \rho } ^ { \prime }$ is $( 0 , r _ { 0 } , 0 , r _ { 1 } , \dots )$ because max-transitions earn 0 and the min-transition $( ( s _ { i } , a _ { i } ) , v _ { i } , s _ { i + 1 } )$ earns $r ( s _ { i } , a _ { i } , s _ { i + 1 } )$ . Hence, on every play,

$$
\begin{array} { r l } & { \mathsf { T o t } _ { 2 N } ( \rho ^ { \prime } ) = \mathsf { T o t } _ { N } ( \mathsf { p r o j } ( \rho ^ { \prime } ) ) , \qquad \mathsf { T o t } ( \rho ^ { \prime } ) = \mathsf { T o t } ( \mathsf { p r o j } ( \rho ^ { \prime } ) ) , } \\ & { \mathsf { D i s c } _ { \beta } ( \rho ^ { \prime } ) = \sum _ { i } \beta ^ { 2 i + 1 } r _ { i } = \beta \mathsf { D i s c } _ { \beta ^ { 2 } } ( \mathsf { p r o j } ( \rho ^ { \prime } ) ) . } \end{array}
$$

For the average, let $\rho ^ { \prime }$ be a play of G, let $\rho = \mathsf { p r o j } ( \rho ^ { \prime } )$ with rewards $r _ { 0 } , r _ { 1 } , \ldots$ , and let $R = \operatorname* { m a x } | r | .$ Write $\begin{array} { r } { A _ { m } = \frac { 1 } { m } \sum _ { i < m } r _ { i } } \end{array}$ and let $B _ { n }$ be the average of the first n rewards of $\rho ^ { \prime }$ . Since the rewards of $\rho ^ { \prime }$ are $0 , r _ { 0 } , 0 , r _ { 1 } , \ldots ,$ the first 2m rewards sum to $\begin{array} { r } { \sum _ { i < m } r _ { i } = m A _ { m } . } \end{array}$ , and the $( 2 m + 1 )$ -st reward is 0. Hence

$$
B _ { 2 m } = { \frac { m A _ { m } } { 2 m } } = { \frac { 1 } { 2 } } A _ { m } \quad { \mathrm { a n d } } \quad B _ { 2 m + 1 } = { \frac { m A _ { m } } { 2 m + 1 } } = { \frac { 2 m } { 2 m + 1 } } B _ { 2 m } .
$$

$$
\begin{array} { r } {  \operatorname { A s } | A _ { m } |  R , \mathrm { w e ~ g e t } | B _ { 2 m + 1 } - B _ { 2 m } | = \frac { | B _ { 2 m } | } { 2 m + 1 } \leq \frac { R } { 2 ( 2 m + 1 ) }  0 . } \end{array}
$$

It follows that the sequences $( B _ { n } ) _ { n }$ and $( B _ { 2 m } ) _ { m }$ have the same limit points. Indeed, every subsequence of $( B _ { n } ) _ { n }$ converging to some L contains infinitely many even indices or infinitely many odd indices; replacing each odd index $2 m + 1$ by 2m changes the values by a vanishing amount, so L is also a limit point of $( B _ { 2 m } ) _ { m }$ , and the converse inclusion is trivial. Therefore

$$
\operatorname* { l i m } _ { n } \operatorname* { i n f } B _ { n } = \operatorname* { l i m } _ { m } \operatorname* { i n f } B _ { 2 m } = { \frac { 1 } { 2 } } \operatorname* { l i m } _ { m } \operatorname* { i n f } A _ { m }
$$

and likewise for lim sup. Thus $X ^ { \mathcal { G } } ( \rho ^ { \prime } ) = c \cdot X ( \mathsf { p r o j } ( \rho ^ { \prime } ) )$ for $\begin{array} { r } { c = { \frac { 1 } { 2 } } } \end{array}$ , whichever of lim inf or lim sup defines LimAvg. For total and discounted return the same identity $X ^ { \mathcal { G } } = c \cdot ( X \circ \mathsf { p r o j } )$ holds with the c of Lemma 1, by the corresponding computation.

Since $X ^ { \mathcal { G } } = c \cdot ( X \circ \mathsf { p r o j } )$ holds on every play of ${ \mathcal { G } } ,$ for every measurable set B of reals we have

$$
\begin{array} { r l } & { \mathcal L _ { X ^ { \mathcal G } } ^ { \sigma ^ { \mathcal G } , \pi ^ { \prime } } ( B ) = \mathbb P _ { \mathcal G } \big ( \{ \rho ^ { \prime } : c \cdot X ( \mathsf { p r o j } ( \rho ^ { \prime } ) ) \in B \} \big ) } \\ & { \qquad = \mathbb P _ { \mathcal G } \big ( \mathsf { p r o j } ^ { - 1 } \big ( X ^ { - 1 } ( S _ { c } ^ { - 1 } ( B ) ) \big ) \big ) } \\ & { \qquad = \big ( \mathsf { p r o j } _ { \# } \mathbb P _ { \mathcal G } \big ) \big ( X ^ { - 1 } ( S _ { c } ^ { - 1 } ( B ) ) \big ) } \\ & { \qquad = \mathbb P _ { \mathcal M } \big ( X ^ { - 1 } ( S _ { c } ^ { - 1 } ( B ) ) \big ) } \\ & { \qquad = \mathcal L _ { X } ^ { \sigma , T ( \pi ^ { \prime } ) } \big ( S _ { c } ^ { - 1 } ( B ) \big ) = ( S _ { c } ) _ { \# } \mathcal L _ { X } ^ { \sigma , T ( \pi ^ { \prime } ) } ( B ) , } \end{array}
$$

where $S _ { c } ^ { - 1 } ( B ) = \{ x : c x \in B \}$ , the fourth equality is the projection of measures shown above (with $\sigma ^ { \mathcal { G } }$ and $T ( \pi ^ { \prime } )$ in the roles of the strategies), and the last equality is the definition of $( S _ { c } ) _ { \# }$ Hence $\mathcal { L } _ { X ^ { \mathcal { G } } } ^ { \sigma ^ { \mathcal { G } } , \pi ^ { \prime } } = ( S _ { c } ) _ { \# } \mathcal { L } _ { X } ^ { \sigma , \Gamma ( \pi ^ { \prime } ) }$ . Since every environment strategy is of the form $\boldsymbol { { \cal T } } ( \pi ^ { \prime } )$ , the two sets of laws in the lemma coincide. □

Corollary B.1 (RPOMDPs and ordinary POSGs). Let $X ~ \in ~ \{ \mathsf { T o t } _ { N } , \mathsf { T o t } , \mathsf { L i m A v g } \}$ . For every RPOMDP M there is an ordinary ac-POSG $\mathcal { G } ^ { \prime } ,$ , ofpolynomial size, and a return $X ^ { \prime } o f { \mathcal { G } } ^ { \prime } $ , such that $\mathsf { V a l } _ { \mathcal { G } ^ { \prime } } ( \varPsi , X ^ { \prime } ) = \mathsf { V a l } _ { \mathcal { M } } ( \varPsi \circ ( S _ { c } ) _ { \# } , X )$ for all law functionals $\varPsi ,$ , where $\mathcal { G } ^ { \prime } = ( \mathfrak { G } ( \mathcal { M } ) ) ^ { \prime }$ and $c = c _ { 1 } c _ { 2 }$ with $c _ { 1 } = 1$ (for Tot<sub>N</sub>, Tot) or <sup>1</sup> (for LimAvg) from Lemma $^ { l , }$ and $c _ { 2 } = 1 , 1 , 1 / k _ { . }$ from Lemma A.1. Here k is the maximal number ofstates $( s , a )$ with $\mathcal { Z } _ { \mathtt { e } } ( s )$ equal.

Proof. Compose Lemmas 1 and A.1, using $( S _ { c _ { 2 } } ) _ { \# } \circ ( S _ { c _ { 1 } } ) _ { \# } = ( S _ { c _ { 1 } c _ { 2 } } ) _ { \# }$ and the fact that minobservations in ${ \mathfrak { G } } ( { \mathcal { M } } )$ depend on $\mathcal { Z } _ { \mathtt { e } } ( s )$ only. □

Corollary B.2 (RPOMDPs and ordinary POSGs, discounted). For every RPOMDP M and rational $\beta \in \mathsf { \Gamma } ( 0 , 1 )$ there is an ordinary ac-POSG $\mathcal { G } ^ { \prime } = ( \mathfrak { G } ( \mathcal { M } ) ) ^ { \times }$ with $\mathsf { V a l } _ { \mathcal { G } ^ { \prime } } ( \varPsi , \mathsf { D i s c } _ { \beta } ) \ : = \ : \mathsf { V a l } _ { \mathcal { M } } ( \varPsi \circ$ $( S _ { \beta } ) _ { \# } , \mathsf { D i s c } _ { \beta ^ { 2 } } )$ for all lawfunctionals Ψ.

Proof. Combine Lemma 1 (with $X = \mathsf { D i s c } _ { \beta ^ { 2 } } , X ^ { \mathcal { G } } = \mathsf { D i s c } _ { \beta } \mathrm { a n d } c = \beta )$ with Proposition A.1, which introduces no further scaling. □

## B.2 Proof of Lemma 2

Let ${ \mathcal { M } } = \Re ( { \mathcal { G } } )$ . A play of the ac-POSG G has the form $\rho = s _ { 0 } a _ { 0 } t _ { 0 } b _ { 0 } s _ { 1 } a _ { 1 } t _ { 1 } b _ { 1 } \cdot \cdot \cdot$ · with $s _ { i } \in$ $S _ { \mathrm { m a x } }$ and $t _ { i } \in S _ { \operatorname* { m i n } }$ . Let

$$
A ( \rho ) = ( s _ { 0 } , \top ) { a } _ { 0 } d _ { t _ { 0 } } \top ( s _ { 1 } , b _ { 0 } ) { a } _ { 1 } d _ { t _ { 1 } } \top ( s _ { 2 } , b _ { 1 } ) \cdot \cdot \cdot .
$$

Every play of $\mathcal { M }$ has this form, because tagged states lead to states $d _ { t }$ and vice versa, and Λ is a bijection whose inverse reads $b _ { i }$ off the tag of the state $( s _ { i + 1 } , b _ { i } )$ . By the definition of the rewards of M, the transition $( s _ { i } , b _ { i - 1 } ) \ \xrightarrow { a _ { i } } \ d _ { t _ { i } }$ earns $r ( s _ { i } , a _ { i } , t _ { i } )$ and the transition $d _ { t _ { i } } \to ( s _ { i + 1 } , b _ { i } )$ earns $r ( t _ { i } , b _ { i } , s _ { i + 1 } )$ , so $\rho$ and $\varLambda ( \rho )$ have identical reward sequences.

Observation Histories and strategies. $\operatorname { A t } s _ { k } .$ , the max-history is

$$
\mathcal { Z } _ { \sf m a x } ( s _ { 0 } ) \mathcal { Z } _ { \sf m a x } ( t _ { 0 } ) \mathcal { Z } _ { \sf m a x } ( s _ { 1 } ) \cdot \cdot \cdot \mathcal { Z } _ { \sf m a x } ( s _ { k } ) ,
$$

while the agent history at $\left( s _ { k } , b _ { k - 1 } \right)$ is

$$
( \mathcal { Z } _ { \sf m a x } ( s _ { 0 } ) , 0 ) ( \mathcal { Z } _ { \sf m a x } ( t _ { 0 } ) , 1 ) \cdot \cdot \cdot ( \mathcal { Z } _ { \sf m a x } ( s _ { k } ) , 0 ) .
$$

Attaching the alternating bits $0 , 1 , 0 , 1 , \ldots$ . is a bijection $\vartheta$ between max observation histories and the agent observation histories that can occur. The agent observation histories that end in an observation with bit 1 are exactly those at states $d _ { t }$ , where the only action is $\top$ . Hence $\sigma ^ { \mathcal { M } } ( \vartheta ( h ) ) = \sigma ( h )$ for observation histories $h$ of odd length, and $\sigma ^ { \mathcal { M } } = \mathsf { D i r a c } ( \top )$ at observation histories that end with bit 1, defines a bijection between max-strategies and agent strategies. The environment of $\mathcal { M }$ observes exactly the sequence $\mathcal { Z } _ { \mathrm { m i n } } ( s _ { 0 } ) \mathcal { Z } _ { \mathrm { m i n } } ( t _ { 0 } ) \mathcal { Z } _ { \mathrm { m i n } } ( s _ { 1 } ) \cdot \cdot$ · that min observes in ${ \mathcal { G } } ,$ so in $d _ { t _ { k } }$ its observation history coincides with the min-history in $t _ { k }$

For a min-strategy $\pi ,$ let $\varOmega ( \pi )$ be the environment strategy whose assignment after observationhistory h has the entries

$$
\Omega ( \pi ) ( h ) ( d _ { t } , \top ) = \sum _ { b \in A ( t ) } \pi ( h ) ( t ) ( b ) \cdot \delta _ { t , b } ^ { \prime } \qquad ( t \in S _ { \operatorname* { m i n } } ) ;
$$

the entries at tagged states are the unique elements of the singleton uncertainty sets. Conversely, for an environment strategy $\pi ^ { \prime \prime }$ of $\mathcal { M }$ , let

$$
\Xi ( \pi ^ { \prime \prime } ) ( h ) ( t ) ( b ) = \sum _ { s ^ { \prime } } \pi ^ { \prime \prime } ( h ) ( d _ { t } , \top ) ( ( s ^ { \prime } , b ) )
$$

be the weight that the point chosen at $d _ { t }$ puts on states with tag b. Recall that $\delta _ { t . b } ^ { \prime } ( ( s ^ { \prime } , b ^ { \prime } ) ) =$ $\delta ( t , b ) ( s ^ { \prime } )$ if $\boldsymbol { b } ^ { \prime } = \boldsymbol { b }$ and 0 otherwise, and that the uncertainty set at $( d _ { t } , \top )$ is conv $\{ \delta _ { t , b } ^ { \prime } : b \in A ( t ) \}$ The supports of the $\delta _ { t , b } ^ { \prime }$ lie in the pairwise disjoint sets $\{ ( s ^ { \prime } , b ) : s ^ { \prime } \}$ , which is the property used below.

Claim. Let $\begin{array} { r } { p = \sum _ { b ^ { \prime } } \lambda _ { b ^ { \prime } } \delta _ { t , b ^ { \prime } } ^ { \prime } } \end{array}$ with $\lambda$ a distribution over $A ( t )$ . Then $\begin{array} { r } { \sum _ { s ^ { \prime } } p ( ( s ^ { \prime } , b ) ) = \lambda _ { b } } \end{array}$ for every $b \in A ( t )$

Indeed, $p ( ( s ^ { \prime } , b ) ) = \lambda _ { b } \delta ( t , b ) ( s ^ { \prime } )$ , because for $\boldsymbol { b } ^ { \prime } \ne b$ the distribution $\delta _ { t , b ^ { \prime } } ^ { \prime }$ gives $( s ^ { \prime } , b )$ weight 0. Summing over $s ^ { \prime }$ and using $\begin{array} { r } { \sum _ { s ^ { \prime } } \delta ( t , b ) ( s ^ { \prime } ) = 1 \mathrm { g i v e s } \lambda _ { b } } \end{array}$

$\varOmega \circ \varXi = \mathrm { i d }$ . Let $p = \pi ^ { \prime \prime } ( h ) ( d _ { t } , \top )$ . It lies in the convex hull, so $p = \sum _ { b } \lambda _ { b } \delta _ { t , b } ^ { \prime }$ for some distribution λ over $A ( t )$ . By the claim, $\Xi ( \pi ^ { \prime \prime } ) ( h ) ( t ) ( b ) = \lambda _ { b }$ . In particular, $\Xi ( \pi ^ { \prime \prime } ) ( h ) ( t )$ is a distribution over $A ( t )$ , and since the environment history in $d _ { t }$ coincides with the min-history in $t , \Xi ( \pi ^ { \prime \prime } )$ is a minstrategy. Moreover,

$$
\Omega ( \Xi ( \pi ^ { \prime \prime } ) ) ( h ) ( d _ { t } , \top ) = \sum _ { b } \Xi ( \pi ^ { \prime \prime } ) ( h ) ( t ) ( b ) \cdot \delta _ { t , b } ^ { \prime } = \sum _ { b } \lambda _ { b } \delta _ { t , b } ^ { \prime } = p = \pi ^ { \prime \prime } ( h ) ( d _ { t } , \top ) .
$$

The entries at tagged states are the unique elements of singleton sets, so they agree. Hence $\varOmega ( \Xi ( \pi ^ { \prime \prime } ) ) =$ $\pi ^ { \prime \prime }$

$\Xi \circ \varOmega = \mathrm { i d }$ . Let $\pi$ be a min-strategy. The point $\begin{array} { r } { \Omega ( \pi ) ( h ) ( { d _ { t } } , \top ) = \sum _ { b ^ { \prime } } \pi ( h ) ( t ) ( b ^ { \prime } ) \delta _ { t , b ^ { \prime } } ^ { \prime } } \end{array}$ lies in the uncertainty set, so $\varOmega ( \pi )$ is an environment strategy. By the claim with $\lambda = \pi ( h ) ( t )$

$$
\Xi ( \varOmega ( \pi ) ) ( h ) ( t ) ( b ) = \sum _ { s ^ { \prime } } \varOmega ( \pi ) ( h ) ( d _ { t } , \top ) ( ( s ^ { \prime } , b ) ) = \pi ( h ) ( t ) ( b )
$$

for all $h , t ,$ and b. Hence $\Xi ( \varOmega ( \pi ) ) = \pi$

Measures. We show $\mathbb { P } _ { \mathcal { M } } ^ { \sigma ^ { \mathcal { M } } , \Omega ( \pi ) } ( \mathsf { C y l } ( A ( u ) ) ) = \mathbb { P } _ { \mathcal { G } } ^ { \sigma , \pi } ( \mathsf { C y l } ( u ) )$ for every prefix u of ${ \mathcal { G } } .$ by induction on the length of $u .$ The base case is trivial. If u ends in a max-state $s _ { k }$ and is extended by a and t, the factor in $\mathcal { G }$ is $\sigma ( h ) ( a ) \delta ( s _ { k } , a ) ( t )$ , and in M it is $\sigma ^ { \mathcal { M } } ( \vartheta ( h ) ) ( a ) \widehat { \delta } ( s _ { k } , a ) ( d _ { t } )$ , which is the same. If u ends in a min-state t and is extended by b and $s ^ { \prime } .$ , the factor in $\mathcal { G }$ is $\pi ( h ) ( t ) ( b ) \delta ( t , b ) ( s ^ { \prime } )$ In $\mathcal { M }$ , the play passes through $d _ { t }$ , where the agent plays ⊤ with probability 1 (its only action, as $\sigma ^ { \mathcal { M } } = \mathsf { D i r a c } ( \top )$ at histories ending with bit 1), and the environment then moves to the tagged state $( s ^ { \prime } , b )$ with probability

$$
\begin{array} { l } { { \varOmega ( \pi ) ( h ) ( d _ { t } , \top ) ( ( s ^ { \prime } , b ) ) = \displaystyle \sum _ { b ^ { \prime } } \pi ( h ) ( t ) ( b ^ { \prime } ) \delta _ { t , b ^ { \prime } } ^ { \prime } ( ( s ^ { \prime } , b ) ) } } \\ { { { } ~ = \pi ( h ) ( t ) ( b ) \delta ( t , b ) ( s ^ { \prime } ) , } } \end{array}
$$

since $\delta _ { t , b ^ { \prime } } ^ { \prime } ( ( s ^ { \prime } , b ) ) = 0$ for $\boldsymbol { b } ^ { \prime } \neq \boldsymbol { b }$ . The factors agree.

Thus the identity holds in both cases, which completes the induction. Extend Λ to prefixes in the obvious way. Every prefix of M is $\varLambda ( u )$ for a unique prefix u of ${ \mathcal { G } } .$ , since Λ is a bijection on plays whose inverse reads $b _ { i }$ off the tag of $( s _ { i + 1 } , b _ { i } )$ , and $\varLambda ( \mathsf { C y l } ( u ) ) = \mathsf { C y l } ( \varLambda ( u ) )$ . The induction therefore gives, for every cylinder ${ \mathsf { C y l } } ( v )$ of M,

$$
\begin{array} { r l } & { A _ { \# } \mathbb { P } _ { \mathcal { G } } ^ { \sigma , \pi } ( \mathsf { C y l } ( v ) ) = \mathbb { P } _ { \mathcal { G } } ^ { \sigma , \pi } \big ( \varLambda ^ { - 1 } ( \mathsf { C y l } ( v ) ) \big ) = \mathbb { P } _ { \mathcal { G } } ^ { \sigma , \pi } ( \mathsf { C y l } ( \varLambda ^ { - 1 } v ) ) = \mathbb { P } _ { \mathcal { M } } ^ { \sigma ^ { M } , \varOmega ( \pi ) } ( \mathsf { C y l } ( v ) ) . } \end{array}
$$

Cylinder sets form a π-system that generates the σ-algebra on play $^ { \prime \mathrm { { S } , } }$ so two probability measures that agree on all cylinder sets are equal. Hence $\Lambda _ { \# } \mathbb { P } _ { \mathcal { G } } ^ { \sigma , \pi } = \mathbb { P } _ { \mathcal { M } } ^ { \sigma ^ { \mathcal { M } } , \Omega ( \pi ) }$

Every return X is a function of the reward sequence, and Λ preserves reward sequences, so $X \circ A =$ X and

$$
\mathscr { L } _ { X } ^ { \sigma ^ { M } , \Omega ( \pi ) } ( B ) = \mathbb { P } _ { M } ^ { \sigma ^ { M } , \Omega ( \pi ) } ( X ^ { - 1 } ( B ) ) = \mathbb { P } _ { g } ^ { \sigma , \pi } ( A ^ { - 1 } X ^ { - 1 } ( B ) ) = \mathbb { P } _ { g } ^ { \sigma , \pi } ( X ^ { - 1 } ( B ) ) = \mathscr { L } _ { X } ^ { \sigma , \pi } ( B ) .
$$

So every law in $\mathcal { G }$ is a law in $\mathcal { M } .$ . Conversely, let $( \sigma ^ { \prime } , \pi ^ { \prime \prime } )$ be a pair of strategies of $\mathcal { M } .$ . Then $\sigma ^ { \prime } = \sigma ^ { \dot { M } }$ for a max-strategy σ (the map $\sigma \mapsto \sigma ^ { \mathcal { M } }$ is a bijection), and $\pi ^ { \prime \prime } = \varOmega ( \varXi ( \pi ^ { \prime \prime } ) )$ , so the pair gives the same law as $( \sigma , \Xi ( \pi ^ { \prime \prime } ) )$ ) in G. Hence the two sets of laws coincide. □

## B.3 Proof of Theorem 1

Notation. For a model, a strategy $\sigma$ of the maximizing player, and a strategy π of the opponent, let $\mathcal { L } _ { X } ^ { \sigma , \pi }$ be the law of the return $X$ . Let $\mathbb { L } ( \sigma ) = \{ \mathcal { L } _ { X } ^ { \sigma , \pi }$ : π a strategy of the opponent}. This set is nonempty, and by definition of the robust value,

$$
\mathsf { V a l } ( \varPsi , X ) = \operatorname* { s u p } _ { \sigma } \operatorname* { i n f } _ { \pi } \varPsi ( \mathcal { L } _ { X } ^ { \sigma , \pi } ) = \operatorname* { s u p } _ { \sigma } \operatorname* { i n f } _ { \mu \in \mathbb { L } ( \sigma ) } \varPsi ( \mu ) ,\tag{1}
$$

because the infimum over $\pi$ of a function of $\mathcal { L } _ { X } ^ { \sigma , \pi }$ is the infimum of that function over the set of all values $\mathcal { L } _ { X } ^ { \sigma , \pi }$ takes. All suprema and infima are taken in ${ \overline { { \mathbb { R } } } } .$ , where they always exist.

A fact about images. Let $f \colon A  B$ be a map between sets and g : $B \to \overline { { \mathbb { R } } } \mathrm { ~ a ~ }$ function. Then

$$
\operatorname* { i n f } _ { y \in f ( A ) } g ( y ) = \operatorname* { i n f } _ { x \in A } g ( f ( x ) ) .\tag{2}
$$

Indeed, every $g ( f ( x ) )$ with $x \in A$ is a value of g on $f ( A )$ , which gives $\because :$ ; and every $y \in f ( A )$ is $f ( x )$ for some $x \in A$ , so $g ( y ) = g ( f ( x ) ) \geq \operatorname* { i n f } _ { x ^ { \prime } \in A } g ( f ( x ^ { \prime } ) )$ , which gives $\stackrel { 6 6 } { \leq } \leq \stackrel { 9 } { 2 }$ . We apply it with $f = ( S _ { c } ) _ { \# }$ , restricted to the set $A = \mathbb { L } _ { \mathcal { M } } ( \sigma )$ of laws, and $g = \varPsi$ . It uses nothing about $\varPsi$

Part (a). Let ${ \mathcal G } = { \mathfrak G } ( { \mathcal M } )$ . By Lemma 1, the map $\sigma \mapsto \sigma ^ { \mathcal { G } }$ is a bijection from the strategies of the agent in $\mathcal { M }$ onto the strategies of max in $\mathcal { G } _ { : }$ , and for every such $\sigma ,$

$$
\mathbb { L } _ { \mathcal { G } } ( \sigma ^ { \mathcal { G } } ) = \{ ( S _ { c } ) _ { \# } \nu : \nu \in \mathbb { L } _ { \mathcal { M } } ( \sigma ) \} .
$$

Here $( S _ { c } ) _ { \# } \nu$ is the law of $c Y$ when Y has law ν. By (2) with $A = \mathbb { L } _ { \mathcal { M } } ( \sigma ) , f = ( S _ { c } ) _ { \# }$ , and $g = \varPsi$

$$
\operatorname* { i n f } _ { \mu \in \mathbb { L } _ { \mathcal { G } } ( \sigma ^ { \mathcal { G } } ) } \varPsi ( \mu ) = \operatorname* { i n f } _ { \nu \in \mathbb { L } _ { \mathcal { M } } ( \sigma ) } \varPsi ( ( S _ { c } ) _ { \# } \nu ) = \operatorname* { i n f } _ { \nu \in \mathbb { L } _ { \mathcal { M } } ( \sigma ) } ( \varPsi \circ ( S _ { c } ) _ { \# } ) ( \nu ) .
$$

The right-hand side is the inner term of (1) for the law functional $\varPsi \circ ( S _ { c } ) _ { \# }$ in $\mathcal { M } .$ . Since $\sigma \mapsto \sigma ^ { \mathcal { G } }$ is a bijection, the supremum over all strategies $\sigma ^ { \mathcal { G } }$ of max in $\mathcal { G }$ equals the supremum over all strategies σ of the agent in $\mathcal { M }$ . Hence

$$
\mathsf { V a l } _ { \mathcal { G } } ( \varPsi , X ^ { \mathcal { G } } ) = \operatorname* { s u p } _ { \sigma } \operatorname* { i n f } _ { \mu \in \mathrm { L } _ { \mathcal { G } } ( \sigma ^ { \mathcal { G } } ) } \varPsi ( \mu ) = \operatorname* { s u p } _ { \sigma } \operatorname* { i n f } _ { \nu \in \mathrm { L } _ { \mathcal { M } } ( \sigma ) } ( \varPsi \circ ( S _ { c } ) _ { \# } ) ( \nu ) = \mathsf { V a l } _ { \mathcal { M } } ( \varPsi \circ ( S _ { c } ) _ { \# } , X ) .
$$

Part (b). Let ${ \mathcal { M } } = \Re ( { \mathcal { G } } )$ . By Lemma $2 , \sigma \mapsto \sigma ^ { \mathcal { M } }$ is a bijection from the strategies of max in $\mathcal { G }$ onto the strategies of the agent in $\mathcal { M } .$ , and $\mathbb { L } _ { \mathcal { M } } ( \sigma ^ { \mathcal { M } } ) = \mathbb { L } _ { \mathcal { G } } ( \sigma )$ , without rescaling. The computation of part (a) with $c = 1$ (that is, with $( S _ { 1 } ) _ { \# }$ the identity) gives $\mathsf { V a l } _ { \mathfrak { R } ( \mathcal { G } ) } ( \varPsi , X ) = \mathsf { V a l } _ { \mathcal { G } } ( \varPsi , X )$

Threshold problems. Let t be a rational and let ▷◁ $\in \{ \geq , > \}$

• From ac-POSGs to RPOMDPs: map $( { \mathcal { G } } , { \varPsi } , X , t )$ to (R(G), Ψ, X, t). By (b), $\mathsf { V a l } _ { \mathfrak { R } ( \mathcal { G } ) } ( \varPsi , X )$ ▷◁ t if and only if $\mathsf { V a l } _ { \mathcal { G } } ( \varPsi , X )$ ▷◁ t.

• From RPOMDPs to ac-POSGs: map $( \mathcal { M } , \varPsi , X , t )$ to $( \mathfrak { G } ( \mathcal { M } ) , \varPsi \circ ( S _ { 1 / c } ) _ { \# } , X ^ { \mathcal { G } } , t )$ . The functional $\varPsi \circ ( S _ { 1 / c } ) _ { \# }$ is a law functional, since $( S _ { 1 / c } ) _ { \# }$ maps laws to laws. Applying (a) to it gives

$$
\mathsf { V a l } _ { \mathfrak { S } ( M ) } ( \varPsi \circ ( S _ { 1 / c } ) _ { \# } , X ^ { \mathcal { G } } ) = \mathsf { V a l } _ { \mathcal { M } } ( \varPsi \circ ( S _ { 1 / c } ) _ { \# } \circ ( S _ { c } ) _ { \# } , X ) = \mathsf { V a l } _ { \mathcal { M } } ( \varPsi , X ) ,
$$

because $( S _ { 1 / c } ) _ { \# } ( S _ { c } ) _ { \# } \nu = \nu$ for every law ν (here $c > 0 )$ . So the answer to the threshold question is the same, and the threshold t is unchanged.

Both maps are computable in logarithmic space by part (c), since the law functional and the threshold are copied or, for $\varPsi \circ ( S _ { 1 / c } ) _ { \# }$ in the standard cases, changed by the parameter rescaling of Corollary 1.

Part (c). We use the standard explicit encoding: lists of states and actions, the vertex lists $V _ { s , a }$ (or the transition distributions), observation tables, and rewards, with all numbers in binary. For the forward construction, a logspace transducer outputs the states s and pairs $( s , a )$ , labels each vertex by its position in the input, copies the coordinates of every vertex verbatim, copies the observation tables, and copies $r ( s , a , s ^ { \prime } )$ to every min-transition $( ( s , a ) , v , s ^ { \prime } )$ . For the backward construction, the transducer outputs the tagged states $( s , b )$ with $\delta ( t , b ) ( s ) > 0$ for some t (found by a scan of the input) together with $( s _ { 0 } , \top )$ , the states $d _ { t } ,$ the singleton uncertainty sets (copies of $\delta ( s , a )$ with renamed targets), the vertices $\delta _ { t , b } ^ { \prime }$ (copies of $\delta ( t , b )$ with tagged targets), the observation tables (with the bits for the agent), and the rewards. Each output symbol depends on a constant number of positions in the input, which can be stored with $O ( \log n )$ bits, and the output size is polynomial. Since logspace reductions compose, the threshold problems are logspace-interreducible in the sense stated in Theorem 1. □

## B.4 Scaling identities for corollary 1

We will use how these functionals react to rescaling: for every $c > 0$

$$
\begin{array} { r l r } & { } & { \mathbb { E } ( ( S _ { c } ) _ { \# } \mu ) = c \mathbb { E } ( \mu ) , \qquad \mathsf { M V } _ { \lambda } ( ( S _ { c } ) _ { \# } \mu ) = c \mathsf { M V } _ { c \lambda } ( \mu ) , } \\ & { } & { \mathsf { C V a R } _ { \alpha } ( ( S _ { c } ) _ { \# } \mu ) = c \mathsf { C V a R } _ { \alpha } ( \mu ) , \quad \mathsf { E R i s k } _ { \theta } ( ( S _ { c } ) _ { \# } \mu ) = c \mathsf { E R i s k } _ { c \theta } ( \mu ) . } \end{array}\tag{3}
$$

Let X have law $\mu$ and let $c > 0 .$ . Clearly $\mathbb { E } ( c X ) = c \mathbb { E } ( X )$ , and $\mathbb { E } ( c X ) - \lambda \mathsf { V a r } ( c X ) = c ( \mathbb { E } ( X ) -$ $c \lambda \mathsf { V a r } ( X ) \big )$ . Substituting $\eta = c \eta ^ { \prime }$ gives

$$
\begin{array} { r } { \mathsf { C V a R } _ { \alpha } ( c X ) = \underset { \eta ^ { \prime } } { \operatorname* { s u p } } \Bigl \{ c \eta ^ { \prime } - \frac { 1 } { \alpha } \mathbb { E } [ ( c \eta ^ { \prime } - c X ) ^ { + } ] \Bigr \} = c \mathsf { C V a R } _ { \alpha } ( X ) , } \end{array}
$$

and

$$
\begin{array} { r } { \mathsf { E R i s k } _ { \theta } ( c X ) = \frac { 1 } { \theta } \log \mathbb { E } [ e ^ { ( c \theta ) X } ] = c \cdot \frac { 1 } { c \theta } \log \mathbb { E } [ e ^ { ( c \theta ) X } ] = c \mathsf { E R i s k } _ { c \theta } ( X ) . } \end{array}
$$

## C Proofs for Section 5

## C.1 Auxiliary facts

(F1) Monotone maps. Let $\beta \colon [ 0 , 1 ] \to { \overline { { \mathbb { R } } } }$ be nondecreasing and continuous, and let $x _ { \sigma , \pi } \in [ 0 , 1 ]$ Then $\beta ( \operatorname { i n f } B ) = \operatorname { i n f } \beta ( B )$ and $\beta ( \operatorname* { s u p } B ) = \operatorname* { s u p } \beta ( B )$ for every nonempty $B \subseteq [ 0 , 1 ]$ ], and therefore $\begin{array} { r } { \operatorname* { s u p } _ { \sigma } \operatorname* { i n f } _ { \pi } \beta ( x _ { \sigma , \pi } ) = \beta ( \operatorname* { s u p } _ { \sigma } \operatorname* { i n f } _ { \pi } x _ { \sigma , \pi } ) } \end{array}$

(F2) Memoryless deterministic saddle points. Let $f$ be an objective on the profiles of a fully informed game (in which players may use arbitrary history-dependent strategies), and let $\sigma ^ { * } , \pi ^ { * }$ be memoryless deterministic (MD) strategies with inf $\begin{array} { r } { { \tiny \tau } ( \sigma ^ { * } , \pi ) = \operatorname* { s u p } _ { \sigma } f ( \sigma , \pi ^ { * } ) = v , } \end{array}$ , where σ and $\pi$ range over all history-dependent strategies. If $\Sigma ^ { \prime } \ni \sigma ^ { * }$ and $\pi ^ { \prime } \ni \pi ^ { * }$ are classes of history-dependent strategies, then

$$
\operatorname* { s u p } _ { \sigma \in \Sigma ^ { \prime } } \operatorname* { i n f } _ { \pi \in \pi ^ { \prime } } f ( \sigma , \pi ) = \operatorname* { i n f } _ { \pi \in \pi ^ { \prime } } \operatorname* { s u p } _ { \sigma \in \Sigma ^ { \prime } } f ( \sigma , \pi ) = v .
$$

Indeed, $\operatorname* { i n f } _ { \pi \in \pi ^ { \prime } } f ( \sigma ^ { * } , \pi ) \geq v ,$ since the infimum over the smaller class $\varPi ^ { \prime }$ is at least the infimum over all strategies. Moreover, for every $\sigma \in \Sigma ^ { \prime }$ , we have $\begin{array} { r } { \operatorname* { i n f } _ { \pi \in \pi ^ { \prime } } f ( \sigma , \pi ) \le f ( \sigma , \pi ^ { * } ) \le } \end{array}$ v. Hence $\begin{array} { r } { \operatorname* { s u p } _ { \sigma \in { \Sigma ^ { \prime } } } \operatorname* { i n f } _ { \pi \in \Pi ^ { \prime } } f ( \sigma , \pi ) = v . } \end{array}$ . Symmetrically, $\operatorname* { s u p } _ { \sigma \in { \Sigma ^ { \prime } } } f ( \sigma , \pi ) \geq f ( \sigma ^ { * } , \pi ) \geq v$ for every $\pi \in \pi ^ { \prime }$ , and $\operatorname* { s u p } _ { \sigma \in { \Sigma ^ { \prime } } } f ( \sigma , \pi ^ { * } ) \leq v ,$ so $\begin{array} { r } { \operatorname* { i n f } _ { \pi \in \pi ^ { \prime } } \operatorname* { s u p } _ { \sigma \in \Sigma ^ { \prime } } f ( \sigma , \pi ) = v . } \end{array}$

We apply (F2) to TBSGs and to games ${ \mathfrak { G } } ( { \mathcal { M } } )$ obtained from RMDPs. In both, the strategies of Section 2 are special history-dependent strategies. They include all MD strategies, because each player can base its choice on the current state: max observes it, and min may use a stationary assignment. Objectives with MD saddle points include reachability [11], reward objectives [44], and the entropic risk of the total reward with nonnegative rewards [3]. For these objectives, the values under our conventions coincide with the classical values.

## C.2 Derivation of the profiles

Let $\mu = \mathsf { B e r } _ { v } ( p )$ , so that $\begin{array} { r } { \int f \mathrm { d } \mu = ( 1 - p ) f ( 0 ) + p f ( v ) } \end{array}$ for every function $f .$ Then:

$\mathbb { E } ( \mu ) = p v$ and $\textstyle \int x ^ { 2 } \mathrm { d } \mu = p v ^ { 2 }$ , so $\mathsf { V a r } ( \mu ) = p v ^ { 2 } - p ^ { 2 } v ^ { 2 } = v ^ { 2 } p ( 1 - p )$ . Hence $\beta _ { v } ^ { \mathsf { M V } _ { \lambda } } ( p ) =$ $p v - \lambda v ^ { 2 } p ( 1 - p )$

$$
\begin{array} { r } { \bullet \int e ^ { \theta x } \mathrm { d } \mu = 1 - p + p e ^ { \theta v } , \mathsf { s o } \beta _ { v } ^ { \mathsf { E R i s k } _ { \theta } } ( p ) = \frac { 1 } { \theta } \log ( 1 - p + p e ^ { \theta v } ) . } \end{array}
$$

• For $\mathsf { C V a R } _ { \alpha }$ , the function inside the supremum is

$$
\begin{array} { r } { g ( \eta ) = \eta - \frac { 1 } { \alpha } \left[ ( 1 - p ) \eta ^ { + } + p ( \eta - v ) ^ { + } \right] . } \end{array}
$$

It is concave and piecewise linear with breakpoints at 0 and v. Its slope is $1 > 0 \mathrm { o n } \left( - \infty , 0 \right)$ it is linear on $[ 0 , v ]$ , and its slope is $\textstyle 1 - { \frac { 1 } { \alpha } } < 0 \operatorname { o n } \left( v , \infty \right)$ because $\alpha < 1$ . So the supremum is attained at 0 or at $v ,$ that is,

$$
\begin{array} { r } { \beta _ { v } ^ { \mathrm { C V a R } _ { \alpha } } ( p ) = \operatorname* { m a x } \{ g ( 0 ) , g ( v ) \} = \operatorname* { m a x } \{ 0 , \ v \left( 1 - \frac { 1 - p } { \alpha } \right) \} . } \end{array}
$$

Lemma C.1. E, $\mathsf { C V a R } _ { \alpha } ( \alpha \in ( 0 , 1 ) )$ , and ERisk<sub>θ</sub> $( \theta \neq 0 )$ have monotone Bernoulli profiles that are strict at the top, for every rational $v > 0 . \mathsf { M V } _ { \lambda } ( \lambda \geq 0$ rational) has such a profile for $v = 1 / ( 1 + \lambda )$ .

Proof. Fix a rational $v > 0 ,$ . For each of the four functionals Ψ we have $\beta _ { v } ^ { \varPsi } ( 1 ) = \varPsi ( \mathsf { D i r a c } ( v ) ) = v .$ which is rational. It remains to show that $\beta _ { v } ^ { \varPsi }$ is nondecreasing, not constant on $( 0 , 1 )$ , and satisfies $\beta _ { v } ^ { \varPsi } ( p ) <$ v for $p < 1$

• E: $\beta ( p ) = p v$ is strictly increasing because $v > 0$

• ERisk : the function $h ( p ) = 1 - p + p e ^ { \theta v }$ is affine with slope $e ^ { \theta v } - 1$ , which is positive if $\theta > 0$ and negative if $\theta < 0$ . So log h is strictly increasing if $\theta > 0$ and strictly decreasing if $\theta < 0$ . In both cases $\begin{array} { r } { \beta ( p ) = \frac { 1 } { \theta } \log h ( p ) } \end{array}$ is strictly increasing, since $\frac { 1 } { \theta }$ has the sign of θ.

$\mathsf { C V a R } _ { \alpha } \colon \beta ( p ) = v \operatorname* { m a x } \{ 0 , 1 - ( 1 - p ) / \alpha \}$ is nondecreasing, because $1 - ( 1 - p ) / \alpha$ is increasing in p. It equals 0 for $p \leq 1 - \alpha$ and is positive for $p > 1 - \alpha$ , and $1 - \alpha \in ( 0 , 1 )$ , so β is not constant on (0, 1). For $p < 1$ we have $( 1 - p ) / \alpha > 0$ , hence $\beta ( p ) < v = \beta ( 1 )$

${ \mathsf { M V } } _ { \lambda }$ , with $v = 1 / ( 1 + \lambda )$ : then $\begin{array} { r } { \lambda v = \frac { \lambda } { 1 + \lambda } < 1 } \end{array}$ . The derivative of $\beta ( p ) = p v - \lambda v ^ { 2 } p ( 1 - p )$ is $\beta ^ { \prime } ( p ) = v - \lambda v ^ { 2 } ( 1 - 2 p ) \geq v - \lambda v ^ { 2 } = v ( 1 - \lambda v ) > 0$ on [0, 1], since $1 - 2 p \leq 1$ . So $\beta$ is strictly increasing.

Whenever $\beta$ is strictly increasing, it is nonconstant on (0, 1) and $\beta ( p ) < \beta ( 1 )$ for $p < 1$ , so the profile is strict at the top.

## C.3 Proof of Theorem 2

[38] show that the following problem is PSPACE-hard: given a POMDP M with nonnegative costs $c ( s , a )$ and a horizon $N \leq | S |$ , decide whether some strategy has expected total cost 0 over the first N steps. Since the costs are nonnegative, the expected total cost of a strategy is 0 if and only if the strategy plays, with probability 0, a forbidden pair $( s , a )$ with $c ( s , a ) > 0$ during the first N steps. So the problem asks, equivalently, for a set B of forbidden state-action pairs (here $B = \{ ( s , a )$ : $c ( s , a ) > 0 \} )$ , whether some strategy avoids B during the first N steps with probability 1.

Given such an instance and the value v from Definition 4, we construct a POMDP M<sup>′</sup> with the states $( s , k )$ for $s \in S$ and $k \in \{ 0 , \ldots , N - 1 \}$ }, where k counts the steps, and two absorbing states ⊥ and ⊤, which indicate failure and success. The initial state is $( s _ { 0 } , 0 )$ , the observations are $\mathcal { Z } _ { \mathsf { a } } ( ( s , k ) ) =$ $( \mathcal { Z } _ { \sf a } ( s ) , k ) , \mathcal { Z } _ { \sf a } ( \bot ) = \bot$ , and $\mathcal { Z } _ { \mathfrak { a } } ( \top ) = \top$ , and the available actions are $A ( ( s , k ) ) = A ( s )$ , with a single action in ⊥ and in ⊤. The transitions are as follows:

• from $( s , k )$ , a forbidden action a, i.e., one with $( s , a ) \in B$ , leads to ⊥ with probability 1;

• from $( s , k )$ with $k < N - 1$ , an allowed action a leads to $( s ^ { \prime } , k + 1 )$ with the probability with which a leads from s to $s ^ { \prime }$ in M;

• from $( s , N - 1 )$ , an allowed action leads to $\top$ with probability 1.

The reward function $r ^ { \prime }$ of $\mathcal { M } ^ { \prime }$ , which is defined on transitions $( x , a , y )$ like every reward function in this paper, assigns the reward v to every transition $( ( s , N - 1 ) , a , \top )$ that enters ⊤, and the reward 0 to all other transitions. Thus the N-th transition of a play earns v if the first N steps avoid B, and every other transition earns 0.

Strategies of M and $\mathcal { M } ^ { \prime }$ correspond: the observations of M<sup>′</sup> additionally reveal the step counter $k ,$ which the agent knows anyway, as it is the length of its history minus one. Under corresponding strategies, the plays of $\mathcal { M } ^ { \prime }$ that reach ⊤ correspond to the plays of $\mathcal { M }$ that avoid B during the first $N$ steps. Hence the return ${ \mathsf { T o t } } _ { N }$ of $\mathcal { M } ^ { \prime }$ equals v on these plays and 0 on all others, so it has law $\mathsf { B e r } _ { v } ( p ^ { \sigma } )$ , where $p ^ { \sigma }$ is the probability that σ avoids B during the first N steps.

We claim that $\operatorname* { m a x } _ { \sigma } p ^ { \sigma }$ exists. The probability $p ^ { \sigma }$ depends only on the distributions $\sigma ( w ) \in \varDelta ( A )$ at the finitely many histories w of length at most $N .$ , and it is a polynomial in the probabilities $\sigma ( w ) ( a )$ : it is a sum, over the finitely many prefixes of length $N$ that avoid B, of products of fixed transition probabilities and of factors $\sigma ( w ) ( a )$ . The set of these parameters is a product of finitely many simplices, which is compact, and a polynomial is continuous. So $p ^ { \sigma }$ attains its maximum $p ^ { * } = \operatorname* { m a x } _ { \sigma } p ^ { \sigma }$

Since the environment of a POMDP has no choice and $\beta _ { v } ^ { \varPsi }$ is nondecreasing, $\mathsf { V a l } _ { \mathcal { M } ^ { \prime } } ( \varPsi , \mathsf { T o t } _ { N } ) =$ $\begin{array} { r } { \operatorname* { s u p } _ { \sigma } \beta _ { v } ^ { \varPsi } ( p ^ { \sigma } ) = \beta _ { v } ^ { \varPsi } ( p ^ { \ast } ) } \end{array}$ : indeed $\beta _ { v } ^ { \varPsi } ( p ^ { \sigma } ) \leq \beta _ { v } ^ { \varPsi } ( p ^ { \ast } )$ for every $\sigma ,$ with equality for a strategy that attains $p ^ { * }$ . As the profile is strict at the top, $\beta _ { v } ^ { \varPsi } ( p ^ { \ast } ) \geq \beta _ { v } ^ { \varPsi } ( 1 )$ if and only ${ \mathrm { i f ~ } } p ^ { * } = 1 , { \mathrm { i . e . } }$ , if and only if the answer to the original instance is positive. Here $\beta _ { v } ^ { \varPsi } ( 1 )$ ) is a rational number, so this is a threshold test. The construction is computable in logarithmic space. Since every POMDP is an RPOMDP, the problem is PSPACE-hard for RPOMDPs. For ac-POSGs, Theorem 1 applies with rescaling factor 1 for finite-horizon total returns (and horizon 2N in the game), so the problem is PSPACE-hard for ac-POSGs as well.

## C.4 The gap lemma and the proof of Theorem 3

Lemma C.2. For all rationals $0 \textless a \textless b \textless 1$ , there is a computable map from two-counter machines C to $P O M D P s \mathcal { M } c$ with a single observation and an absorbing target set T such that: if C halts, then $\mathbb { P } ^ { \sigma } ( \odot T ) > b$ for some pure strategy σ; $i f { \mathcal { C } }$ does not halt, then $\mathbb { P } ^ { \sigma } ( \diamond T ) \le a$ for all strategies σ.

Proof. The construction. [12] (see also the account by [29]) construct from C, a rational $\epsilon > 0$ , and an integer $K \geq 1$ a probabilistic finite automaton A with a finite alphabet $\Sigma ,$ , a finite set of states $Q _ { \ l }$ an initial state $q _ { 0 }$ , an absorbing accepting state $q _ { \mathrm { a c c } }$ , and an absorbing rejecting state $q _ { \mathrm { r e j } }$ , such that, writing Pr(w accepted) for the probability that A is in $q _ { \mathrm { a c c } }$ after reading the word w from $q _ { 0 } \mathrm { : }$

• if C does not halt, then Pr(w accepted) $\leq \epsilon$ for every word $w \in \Sigma ^ { * }$ ;

• if C halts, then there is a word w with li $\mathrm { n } _ { n  \infty } \mathrm { P r } ( w ^ { n } { \mathrm { a c c e p t e d } } ) = 1 - 2 ^ { - K }$

In particular, $q _ { 0 } \neq q _ { \mathrm { a c c } } ,$ , since otherwise the empty word would be accepted with probability $1 > \epsilon$ for non-halting machines. We use these properties of ${ \mathcal { A } } ,$ and construct a POMDP $\mathcal { M } _ { \mathcal { C } }$ from A and then perform the analysis of arbitrary (randomized) strategies of this POMDP as below. Given the rationals $a < b ,$ choose $\epsilon = a$ and an integer $K$ with $1 - 2 ^ { - K } > b .$ . Let $\mathcal { M } _ { \mathcal { C } }$ be the POMDP with the state set $Q .$ , the initial state $q _ { 0 }$ , the action set $\varSigma$ at every state, and the transition probabilities $\delta ( q , \alpha ) ( q ^ { \prime } )$ equal to the probability that A moves from q to $q ^ { \prime }$ when it reads the letter α. All states have the same observation, so the POMDP has a single observation. Let $T = \{ q _ { \mathrm { a c c } } \}$ ; it is absorbing in $\mathcal { M } _ { \mathcal { C } }$ because $q _ { \mathrm { a c c } }$ is absorbing in $\mathcal { A }$

Strategies of $\mathcal { M } _ { \mathcal { C } }$ . A strategy depends only on the history of observations. Since all states have the same observation, the history of length i is determined by i. A strategy $\sigma$ is therefore a sequence $( \sigma _ { i } ) _ { i \geq 0 }$ of distributions over $\Sigma ,$ , where $\sigma _ { i }$ is used at step i. It is pure if every $\sigma _ { i }$ is a Dirac distribution, that is, σ is an infinite word. In particular, the letters played by σ do not depend on the states visited.

Reaching $T$ within n steps. Let $E _ { n }$ be the event that the state after n steps lies in $T .$ Since $T$ is absorbing, once a play is in $T$ it stays there, so $E _ { n }$ is the event of reaching $T$ within the first n steps. Hence $E _ { 1 } \subseteq E _ { 2 } \subseteq \cdots$ · and $\begin{array} { r } { \bigcup _ { n } E _ { n } = \diamond T } \end{array}$

Fix a strategy $\sigma = ( \sigma _ { i } ) _ { i \geq 0 }$ and $n \geq 1$ . The event $E _ { n }$ depends only on the first n steps, so $\mathbb { P } ^ { \sigma } ( E _ { n } )$ is the sum of the probabilities of the finitely many prefixes $q _ { 0 } w _ { 0 } q _ { 1 } w _ { 1 } \cdot \cdot \cdot w _ { n - 1 } q _ { n }$ with $q _ { n } \in T$ . At step $i ,$ the strategy plays the letter $w _ { i }$ with probability $\sigma _ { i } ( w _ { i } )$ , independently of the previous states and letters, because it has no information other than the step number i; then the state moves from $q _ { i }$ to $q _ { i + 1 }$ with probability $\delta ( q _ { i } , w _ { i } ) ( q _ { i + 1 } )$ . So such a prefix has probability

$$
\prod _ { i < n } \sigma _ { i } ( w _ { i } ) \delta ( q _ { i } , w _ { i } ) ( q _ { i + 1 } ) = \Bigl ( \prod _ { i < n } \sigma _ { i } ( w _ { i } ) \Bigr ) \cdot \prod _ { i < n } \delta ( q _ { i } , w _ { i } ) ( q _ { i + 1 } ) .
$$

Group the prefixes by the word $w = w _ { 0 } \cdot \cdot \cdot w _ { n - 1 } \in \Sigma ^ { n }$ that they contain. The prefixes that contain a fixed word w are those with an arbitrary sequence of states $q _ { 1 } , \ldots , q _ { n }$ with $q _ { n } ~ \in ~ T$ , that is, $q _ { n } = q _ { \mathrm { a c c } }$ . Their total probability is

$$
\sum _ { \substack { q _ { 1 } , \ldots , q _ { n } \in Q } } \left( \prod _ { i < n } \sigma _ { i } ( w _ { i } ) \right) \prod _ { i < n } \delta ( q _ { i } , w _ { i } ) ( q _ { i + 1 } ) = \left( \prod _ { i < n } \sigma _ { i } ( w _ { i } ) \right) \cdot \sum _ { \substack { q _ { 1 } , \ldots , q _ { n } \in Q } } \prod _ { i < n } \delta ( q _ { i } , w _ { i } ) ( q _ { i + 1 } ) ,
$$

where we have taken the factor $\Pi _ { i < n } \sigma _ { i } ( w _ { i } )$ out of the sum, because it does not depend on the states. The remaining sum is the probability that A is in $q _ { \mathrm { a c c } }$ after reading w, that is, Pr(w accepted). To see this, let $\mu _ { k }$ be the distribution over Q of the state of A after reading the first k letters of w. Then $\mu _ { 0 } = \mathsf { D i r a c } ( q _ { 0 } )$ and $\begin{array} { r } { \mu _ { k + 1 } ( q ^ { \prime } ) = \sum _ { q } \mu _ { k } ( q ) \delta ( q , w _ { k } ) ( q ^ { \prime } ) } \end{array}$ , and unfolding this recursion gives

$$
\mu _ { n } ( q _ { \mathrm { a c c } } ) = \sum _ { \substack { q _ { 1 , \dots , q _ { n } } \in Q i < n } } \prod _ { i < n } \delta ( q _ { i } , w _ { i } ) ( q _ { i + 1 } ) ,
$$

which is exactly the remaining sum. Summing over all words $w \in \Sigma ^ { n }$ , we obtain

$$
\mathbb { P } ^ { \sigma } ( E _ { n } ) = \sum _ { w \in \Sigma ^ { n } } \left( \prod _ { i < n } \sigma _ { i } ( w _ { i } ) \right) { \mathrm { P r } } ( w { \mathrm { ~ a c c e p t e d } } ) .\tag{†}
$$

The weights $\begin{array} { r } { \lambda ( w ) = \prod _ { i < n } \sigma _ { i } ( w _ { i } ) } \end{array}$ , for $w \in \Sigma ^ { n }$ , are nonnegative and sum to 1:

$$
\sum _ { w \in \Sigma ^ { n } } \lambda ( w ) = \prod _ { i < n } \sum _ { \alpha \in \Sigma } \sigma _ { i } ( \alpha ) = 1 .
$$

So the right-hand side of (†) is a weighted average of the numbers Pr(w accepted), $w \in \Sigma ^ { n }$ , and therefore

$$
\operatorname* { m i n } _ { w \in \Sigma ^ { n } } \operatorname* { P r } ( w { \mathrm { ~ a c c e p t e d } } ) \leq \mathbb { P } ^ { \sigma } ( E _ { n } ) \leq \operatorname* { m a x } _ { w \in \Sigma ^ { n } } \operatorname* { P r } ( w { \mathrm { ~ a c c e p t e d } } ) .
$$

For a pure strategy, λ is the Dirac distribution on the first n letters of the infinite word $\sigma ,$ and (†) says that $\mathbb { P } ^ { \sigma } ( E _ { n } )$ is the acceptance probability of this prefix.

If C halts. Let w be as above. Since lim $\scriptstyle 1 _ { k \to \infty } \operatorname* { P r } ( w ^ { k }$ accepted) = 1 − 2<sup>−K</sup> > b, there is k with $\mathrm { P r } ( w ^ { k }$ accepted) > b. Let $L = k | w |$ be the length of $w ^ { k }$ , and let σ be the pure strategy that plays the letters of $w ^ { k }$ in its first L steps and an arbitrary fixed letter afterwards. For $i < L$ , the distribution $\sigma _ { i }$ is the Dirac distribution on the i-th letter of $w ^ { \dot { k } }$ . Hence, in (†) for $n = L$ , the weight $\Pi _ { i < L } \sigma _ { i } ( u _ { i } )$ of a word $u \in \Sigma ^ { L }$ is 1 if $u = w ^ { k }$ and 0 otherwise, so (†) reduces to $\mathbb { P } ^ { \sigma } ( E _ { L } ) = \operatorname* { P r } ( w ^ { k }$ accepted) > b. Since $E _ { L } \subseteq \langle \gamma T$ , we get $\mathbb { P } ^ { \sigma } ( \langle \gamma T ) > b$

If C does not halt. Let σ be an arbitrary strategy, randomized or not, and let $n \geq 1$ . Every word is accepted with probability at most $\epsilon = a ,$ , so by (†),

$$
\mathbb { P } ^ { \sigma } ( E _ { n } ) \leq \operatorname* { m a x } _ { w \in \Sigma ^ { n } } \operatorname* { P r } ( w { \mathrm { ~ a c c e p t e d } } ) \leq a .
$$

Since the events $E _ { n }$ increase to $\diamond T .$ , we get $\begin{array} { r } { \mathbb { P } ^ { \sigma } ( \diamond T ) = \operatorname* { l i m } _ { n \to \infty } \mathbb { P } ^ { \sigma } ( E _ { n } ) \leq a . } \end{array}$

ProofofTheorem 3. Choice of the threshold. Let v be as in Definition 4 and $\beta = \beta _ { v } ^ { \varPsi }$ . Since $\beta$ is nondecreasing and not constant on $( 0 , 1 )$ , there are $a ^ { \prime } < b ^ { \prime }$ in $( 0 , 1 )$ with $\beta ( a ^ { \prime } ) < \beta ( b ^ { \prime } )$ . Choose rationals a and b with $0 < a \le a ^ { \prime }$ and $b ^ { \prime } \leq b < 1$ ; then, since $\beta$ is nondecreasing,

$$
\beta ( a ) \leq \beta ( a ^ { \prime } ) < \beta ( b ^ { \prime } ) \leq \beta ( b ) .
$$

Choose a rational c with $\beta ( a ) < c < \beta ( b )$ . The numbers $v , a , b ,$ , and c depend only on $\varPsi$ and not on any machine.

The POMDPs. Given a two-counter machine C, let $\mathcal { M } _ { \mathcal { C } }$ and T be as in Lemma C.2 for these a and b. We define the rewards for the two kinds of returns:

• for Tot, the reward is v on the transitions that enter $T$ (that ${ \mathrm { i s } } ,$ from a state outside $T$ to a state in T) and 0 on all other transitions; since T is absorbing, at most one transition of a play enters $T ;$

• for LimAvg, the reward is v on the transitions inside $T$ (from a state in $T$ to a state in T) and 0 on all other transitions.

The rewards are nonnegative, as required for Tot. In both cases, $X = v \cdot \mathbf { 1 } [ \diamond T ]$ on every play. For Tot, this holds because the play collects v once if it reaches T and 0 otherwise. For LimAvg, if the play reaches T at time $\tau ,$ then all transitions after τ lie inside T and earn $v ,$ so the averages converge to $v ;$ if the play never reaches T, all rewards are 0.

The value. For a strategy σ of the agent, let $p ^ { \sigma } = \mathbb { P } ^ { \sigma } ( \diamond T )$ . Then X has the law $\mathsf { B e r } _ { v } ( p ^ { \sigma } )$ , and, since the environment of a POMDP has no choice,

$$
\mathsf { V a l } ( \varPsi , X ) = \operatorname* { s u p } _ { \sigma } \varPsi \bigl ( \mathsf { B e r } _ { v } ( p ^ { \sigma } ) \bigr ) = \operatorname* { s u p } _ { \sigma } \beta ( p ^ { \sigma } ) .
$$

The two cases.

• If C halts, some strategy σ has $p ^ { \sigma } > b$ by Lemma C.2. As $\beta$ is nondecreasing, $\mathsf { V a l } \geq \beta ( p ^ { \sigma } ) \geq$ $\beta ( b ) > c$

• If C does not halt, then $p ^ { \sigma } \leq a$ for every strategy σ, randomized ones included, by Lemma C.2. $\operatorname { A s } \beta$ is nondecreasing, $\beta ( p ^ { \sigma } ) \leq \beta ( a )$ for all σ, so $\mathsf { V a l } \leq \beta ( a ) < c .$

Hence $\mathsf { V a l } > c$ (and therefore also ${ \mathsf { V a l } } \geq c )$ holds if and only if C halts. Since $\mathcal { C } \mapsto \mathcal { M } _ { \mathcal { C } }$ is computable, the halting problem for two-counter machines reduces to both the threshold problem and the strict threshold problem for POMDPs with a single observation, with the fixed rational threshold $c .$ Both are therefore undecidable. They are also undecidable for RPOMDPs, of which POMDPs are a special case.

ac-POSGs. We apply the forward construction of Definition 2 to $\mathcal { M } _ { \mathcal { C } }$ , regarded as an RPOMDP. For Tot, the rescaling factor of Lemma 1 is 1, so the game has the same achievable return laws as $\mathcal { M } _ { \mathcal { C } }$ , and the same threshold c separates the two cases. For LimAvg, Lemma 1 halves the average return. We therefore use the reward 2v, instead of v, on the transitions inside $T$ before applying the construction. Then the average return of the game has the law $\mathsf { B e r } _ { v } ( p ^ { \sigma } )$ again, and the same argument applies. Hence both threshold problems are also undecidable for ac-POSGs. □

## C.5 Proof of Theorem 4

Lemma C.3. For every nonnegative random variable $Z$ and every $\alpha \in ( 0 , 1 ) \colon { \mathsf { C V a R } } _ { \alpha } ( Z ) > 0$ if and only $i f \mathbb { P } ( Z > 0 ) > 1 - \alpha$

Proof. Let $Z \ge 0$ and $\alpha \in ( 0 , 1 )$ , let $m = \mathbb { P } ( Z = 0 )$ , and let

$$
\begin{array} { r } { g ( \eta ) = \eta - \frac { 1 } { \alpha } \mathbb { E } \big [ ( \eta - Z ) ^ { + } \big ] , \qquad \mathrm { s o ~ t h a t } \qquad \mathbb { C } \mathsf { V a R } _ { \alpha } ( Z ) = \operatorname* { s u p } _ { \eta \in \mathbb { R } } g ( \eta ) . } \end{array}
$$

Since $Z \ge 0$ , we have $0 \leq ( \eta - Z ) ^ { + } \leq \eta ^ { + }$ , so the expectation is finite and $g$ is well defined. Because $Z \geq 0$ we also have $\mathbb { P } ( Z > 0 ) = 1 - m$ , so the claim is equivalent to

$$
\mathrm { C V a R } _ { \alpha } ( Z ) > 0 \iff m < \alpha .
$$

The values $\eta \le 0 . \mathrm { ~ H ~ } \eta \le 0$ , then $\eta - Z \leq 0 .$ , so $( \eta - Z ) ^ { + } = 0$ and $g ( \eta ) = \eta \leq 0$ . In particular $g ( 0 ) = 0 , \mathtt { s o C V a R } _ { \alpha } ( Z ) \geq 0$ always, and positivity can only come from some $\eta > 0$

Case $m \geq \alpha$ . Let $\eta > 0$ . Writing $1 = \mathbf { 1 } [ Z = 0 ] + \mathbf { 1 } [ Z > 0 ]$ , we have

$$
( \eta - Z ) ^ { + } = ( \eta - Z ) ^ { + } \mathbf { 1 } [ Z = 0 ] + ( \eta - Z ) ^ { + } \mathbf { 1 } [ Z > 0 ] = \eta \mathbf { 1 } [ Z = 0 ] + ( \eta - Z ) ^ { + } \mathbf { 1 } [ Z > 0 ] \geq \eta \mathbf { 1 } [ Z = 0 ] ,
$$

where the second equality holds because $( \eta - 0 ) ^ { + } = \eta ( \mathrm { a s } \eta > 0 )$ , and the inequality holds because $( \eta - Z ) ^ { + } \mathbf { 1 } [ Z > 0 ] \ge 0$ . Taking expectations gives

$$
\begin{array} { r } { \mathbb { E } \big [ ( \eta - Z ) ^ { + } \big ] \ge \eta \mathbb { P } ( Z = 0 ) = \eta m \ge \eta \alpha , } \end{array}
$$

where the last inequality uses $m \geq \alpha$

Hence

$$
\begin{array} { r } { g ( \eta ) \leq \eta - \frac { 1 } { \alpha } \eta \alpha = 0 . } \end{array}
$$

Together with the values $\eta \leq 0 ,$ , we get $g \le 0$ on R, and $g ( 0 ) = 0$ , so $\mathsf { C V a R } _ { \alpha } ( Z ) = \operatorname* { s u p } _ { \eta } g ( \eta ) = 0$ Thus $\mathsf { C V a R } _ { \alpha } ( Z ) > 0 \mathrm { f a i l s } .$

Case $m < \alpha$ . Let $\eta > 0$ . Since $Z \geq 0 ,$ , we have $0 \leq ( \eta - Z ) ^ { + } \leq \eta .$ , and $( \eta - Z ) ^ { + } = 0 { \mathrm { i f } } Z \geq \eta$ Hence $( \eta - Z ) ^ { + } \leq \eta \cdot \mathbf { 1 } [ Z < \eta ]$ , so $\mathbb { E } [ ( \eta - Z ) ^ { + } ] \le \eta \mathbb { P } ( Z < \eta )$ and

$$
\begin{array} { r } { g ( \eta ) \geq \eta - \frac { 1 } { \alpha } \eta \mathbb { P } ( Z < \eta ) = \eta \Big ( 1 - \frac { \mathbb { P } ( Z < \eta ) } { \alpha } \Big ) . } \end{array}
$$

$\mathrm { A s } \ \eta \downarrow 0$ , the events $\{ Z < \eta \}$ decrease to $\bigcap _ { n > 0 } \{ Z < \eta \} = \{ Z \leq 0 \} = \{ Z = 0 \}$ , where the last equality holds because $Z \ge 0$ . By continuity of the probability measure from above, $\mathbb { P } ( Z < \eta ) $ $m < \alpha .$ . So there is $\eta > 0$ with $\mathbb { P } ( Z < \eta ) < \alpha ,$ , and for this η the right-hand side above is positive. Hence $\mathrm { C V a R } _ { \alpha } ( Z ) \geq g ( \eta ) > 0$

The two cases cover all possibilities, so $\mathsf { C V a R } _ { \alpha } ( Z ) > 0$ if and only if $m < \alpha$ , that is, if and only if $\mathbb { P } ( Z > 0 ) = 1 - m > 1 - \alpha$ □

ProofofTheorem 4. Parameters. Let $\alpha \in ( 0 , 1 )$ be given. Choose rationals $a , b \in ( 0 , 1 )$ with $a < 1 - \alpha < b$ (for example $\begin{array} { r } { a = \frac { 1 - \alpha } { 2 } } \end{array}$ and $\begin{array} { r } { b = \frac { 2 - \alpha } { 2 } ) } \end{array}$ . For a two-counter machine C, let $\mathcal { M } _ { \mathcal { C } }$ and $T$ be as in Lemma C.2 for these a and b. Recall that T is absorbing and that the initial state of $\mathcal { M } _ { \mathcal { C } }$ is not in T. Give $\mathcal { M } _ { \mathcal { C } }$ the reward 1 on the transitions from a state outside T into $T$ and 0 on all other transitions. Since T is absorbing, at most one transition of a play earns the reward 1. For a strategy σ, write $p ^ { \sigma } = \mathbb { P } ^ { \sigma } ( \diamond T )$

Step 1: the sign of the discounted return. Let $\gamma \in ( 0 , 1 )$ be any discount factor, and let $\rho$ be a play of $\mathcal { M } _ { \mathcal { C } } . \mathrm { ~ H ~ } \rho$ enters $T$ by its τ-th transition, $\tau \geq 1$ , then the reward 1 is earned at that transition and nowhere else, so Disc<sub>γ</sub> $\begin{array} { r } { \mathbf { \rho } ( \rho ) = \gamma ^ { \tau - 1 } > 0 . } \end{array}$ If $\rho$ never enters T, then all rewards of $\rho$ are 0 and $\mathsf { D i s c } _ { \gamma } ( \rho ) = 0$ . Since the initial state is not in T, a play enters T if and only if it reaches T. Hence, for every discount factor $\gamma \in ( 0 , 1 )$

$$
\mathsf { D i s c } _ { \gamma } ( \rho ) \geq 0 \mathrm { f o r e v e r y p l a y } \rho , \qquad \mathrm { a n d } \qquad \{ \rho : \mathsf { D i s c } _ { \gamma } ( \rho ) > 0 \} = \diamondsuit T ,\tag{1}
$$

where $\diamond T$ denotes the set of plays that visit $T$

Step 2: the key equivalencefor POMDPs. In a POMDP the environment has no choice, so $\mathsf { V a l } _ { \mathcal { M } _ { c } } ( \mathsf { C V a R } _ { \alpha } , \mathsf { D i s c } _ { \gamma } ) =$ s $\begin{array} { r } { \operatorname * { l p } _ { \sigma } \mathsf { C V a R } _ { \alpha } \bigl ( \mathcal { L } _ { \mathsf { D i s c } _ { \sim } } ^ { \sigma } \bigr ) } \end{array}$ . By (1), Disc is a nonnegative random variable with $\mathbb P ^ { \sigma } ( \mathsf { D i s c } _ { \gamma } > 0 ) = p ^ { \sigma }$ so Lemma C.3 gives

$$
\mathrm { C V a R } _ { \alpha } \big ( \mathcal L _ { \sf D i s c _ { \gamma } } ^ { \sigma } \big ) > 0 \iff p ^ { \sigma } > 1 - \alpha .\tag{2}
$$

Since a supremum of real numbers is positive if and only if one of the numbers is positive,

$$
\mathsf { V a l } _ { \mathcal { M } _ { \mathfrak { c } } } ( \mathsf { C V a R } _ { \alpha } , \mathsf { D i s c } _ { \gamma } ) > 0 \iff \exists \sigma : p ^ { \sigma } > 1 - \alpha .\tag{3}
$$

Step 3: the two cases of the gap. If C halts, Lemma C.2 gives a strategy σ with $p ^ { \sigma } > b > 1 - \alpha _ { \mathrm { { \ O } } }$ , so the right-hand side of (3) holds. If C does not halt, then $p ^ { \sigma } \leq a < 1 - \alpha$ for every strategy σ, so the right-hand side of (3) fails. Hence

$$
\mathsf { V a l } _ { \mathcal { M } _ { \mathcal { C } } } ( \mathsf { C V a R } _ { \alpha } , \mathsf { D i s c } _ { \gamma } ) > 0 \iff \mathcal { C } \mathrm { ~ h a l t s } .
$$

The map ${ \mathcal { C } } \mapsto { \mathcal { M } } _ { \mathcal { C } }$ is computable, and the halting problem of two-counter machines is undecidable. Therefore it is undecidable whether $\mathsf { V a l } ( \mathsf { C V a R } _ { \alpha } , \mathsf { D i s c } _ { \gamma } ) > 0$ for POMDPs with a single observation, and hence for RPOMDPs, of which POMDPs are the special case with singleton uncertainty sets.

Step 4: ac-POSGs. Let $\beta \in ( 0 , 1 )$ be a rational discount factor of the game, and let $\mathcal { G } = \mathfrak { G } ( \mathcal { M } _ { \mathcal { C } } )$ where $\mathcal { M } _ { \mathcal { C } }$ is regarded as an RPOMDP whose uncertainty sets are singletons. Take the discount factor $\gamma = \beta ^ { 2 }$ , which is rational, in Steps 1 to 3. By Lemma 1 with $X = \mathsf { D i s c } _ { \beta ^ { 2 } } , X ^ { \mathcal { G } } = \mathsf { D i s c } _ { \beta }$ , and $c = \beta ,$ , there is a bijection $\sigma \mapsto \sigma ^ { \mathcal { G } }$ from the agent strategies of $\mathcal { M } _ { \mathcal { C } }$ to the max-strategies of $\mathcal { G }$ such that, for every σ,

$$
\{ \mathcal { L } _ { \mathrm { D i s c } _ { \beta } } ^ { \sigma ^ { \mathcal { G } } , \pi ^ { \prime } } : \pi ^ { \prime } \in \pi _ { \mathcal { G } } \} = \big \{ ( S _ { \beta } ) _ { \# } \mathcal { L } _ { \mathrm { D i s c } _ { \beta ^ { 2 } } } ^ { \sigma , \pi } : \pi \in \pi _ { \mathcal { M } _ { c } } \big \} .
$$

Since $\mathsf { C V a R } _ { \alpha } ( ( S _ { \beta } ) _ { \# } \mu ) = \beta \mathsf { C V a R } _ { \alpha } ( \mu )$ by (3) in B.4, taking the infimum of $\mathsf { C V a R } _ { \alpha }$ over both sets and then the supremum over σ gives

$$
\mathsf { V a l } _ { \mathcal { G } } ( \mathsf { C V a R } _ { \alpha } , \mathsf { D i s c } _ { \beta } ) = \beta \cdot \mathsf { V a l } _ { \mathcal { M } _ { \mathcal { C } } } ( \mathsf { C V a R } _ { \alpha } , \mathsf { D i s c } _ { \beta ^ { 2 } } ) .
$$

$\mathbf { A } \mathbf { s } \ \beta > 0$ , the left-hand side is positive if and only if $\mathsf { V a l } _ { \mathcal { M } _ { c } } ( \mathsf { C V a R } _ { \alpha } , \mathsf { D i s c } _ { \beta ^ { 2 } } ) > 0$ , which holds if and only if C halts, by Step 3 applied with $\gamma = \beta ^ { 2 }$ . Since ${ \mathfrak { G } } ( { \mathcal { M } } _ { { \mathcal { C } } } )$ is computable from C (as the composition of the computable map ${ \mathcal { C } } \mapsto { \mathcal { M } } _ { \mathcal { C } }$ and the construction $\mathfrak { G } ( \cdot ) )$ ), the strict threshold problem for discounted CVaR is undecidable for ac-POSGs with every rational discount factor $\beta \in$ (0, 1).

## C.6 Proof of Proposition 1

(a) Let M be an RPOMDP whose uncertainty sets $\mathcal { P } ( s , a )$ are convex polytopes with rational vertex sets $V _ { s , a } ,$ , let N be the horizon, and let all rewards, discount factors, risk parameters $( \lambda , \alpha , \theta )$ and the threshold c be rational. The agent maximises and the environment minimises. Write ▷ for either $\geq$ or >. We show that ${ \mathsf { V a l } } \triangleright c$ is decidable. For $\operatorname { a c - P O S G s } .$ , the claim follows from Theorem 1(b), or directly by the same argument (see the end of the proof).

Step 1: the strategy spaces are finite products of simplices. Strategies depend only on observation histories, and only histories of length less than N matter. Let H be the finite set of observation histories of the agent of length less than $N$ , and G the finite set of histories of length less than N available to the environment. A strategy of the agent is a family $x = \left( x _ { h , a } \right)$ of reals with $x _ { h , a } \geq 0$ and $\textstyle \sum _ { a } x _ { h , a } = 1$ for each $h \in H , \mathrm { ~ A ~ }$ strategy of the environment selects, for each $g \in G$ and each pair $( s , a )$ , a point of $\begin{array} { r } { \mathcal { P } ( s , a ) = \mathrm { c o n v } ( V _ { s , a } ) } \end{array}$ . Every such point has the form $\textstyle \sum _ { v } y _ { g , s , a , v }$ v with $y _ { g , s , a , v } \geq 0$ and $\begin{array} { r } { \sum _ { v } y _ { g , s , a , v } = 1 } \end{array}$ , and conversely each such family $y$ defines a point. (Randomising over points of a convex set yields the same transition law as the mean point, so nothing is lost.) Let X and $Y$ be the sets of all such x and y. They are compact, being finite products of simplices.

Step 2: probabilities ofprefixes are polynomials. Let $u = s _ { 0 } a _ { 0 } s _ { 1 } \cdot \cdot \cdot a _ { N - 1 } s _ { N }$ be a play prefix of length N, let $h _ { i } \in H$ and $g _ { i } \in G$ be the histories of the agent and of the environment along u before step i (they are determined by u), and let $\mu _ { 0 }$ be the initial distribution. Then

$$
p _ { u } ( x , y ) \ : = \ : \mathrm { P r } ( \mathsf { C y l } ( u ) ) \ : = \ : \mu _ { 0 } ( s _ { 0 } ) \prod _ { i < N } x _ { h _ { i } , a _ { i } } \cdot \sum _ { v \in V _ { s _ { i } , a _ { i } } } y _ { g _ { i } , s _ { i } , a _ { i } , v } v ( s _ { i + 1 } ) ,
$$

a polynomial of degree at most $2 N$ in the variables $x , y$ with rational coefficients. The return is a function of the prefix, which takes a rational value $x _ { u }$ on ${ \mathsf { C y l } } ( u )$ . Let V be the (finite) set of these values. The law of the return under $\begin{array} { r } { ( x , y ) \mathrm { i s } \sum _ { u } p _ { u } ( x , y ) \delta _ { x _ { u } } } \end{array}$

Step 3: the value ofthe profile is a continuousfunction $f ( x , y )$ . Write $p _ { u } = p _ { u } ( x , y )$

• E: $\begin{array} { r } { f = \sum _ { u } p _ { u } x _ { u } } \end{array}$

• MV<sub>λ</sub>: f = P<sub>u</sub> p<sub>u</sub> x<sub>u</sub> − λ P<sub>u</sub> p<sub>u</sub> x<sup>2</sup><sub>u</sub> − (P<sub>u</sub> p<sub>u</sub> x<sub>u</sub>)<sup>2</sup>.

$\mathtt { C V a R } _ { \alpha } \colon f = \operatorname* { m a x } _ { z \in \mathcal { V } } g _ { z }$ with $\begin{array} { r } { g _ { z } = z - \frac { 1 } { \alpha } \sum _ { u } p _ { u } ( z - x _ { u } ) ^ { + } . } \end{array}$

$\begin{array} { r } { \mathsf { E R i s k } _ { \theta } \left( \theta \neq 0 \right) : f = \frac { 1 } { \theta } \log S \mathrm { w i t h } S = \sum _ { u } p _ { u } e ^ { \theta x _ { u } } } \end{array}$

In the first two cases $f$ is a polynomial in the $p _ { u }$ . For $\mathsf { C V a R } _ { \alpha }$ we use the Rockafellar-Uryasev formula $\begin{array} { r } { \mathsf { C V a R } _ { \alpha } = \operatorname* { s u p } _ { \eta \in \mathbb { R } } \varphi ( \eta ) } \end{array}$ , where

$$
\varphi ( \eta ) = \eta - { \textstyle \frac { 1 } { \alpha } } \sum _ { u } p _ { u } ( \eta - x _ { u } ) ^ { + } ,
$$

and show that the supremum is attained at a point of V. Let $z _ { 1 } < \cdots < z _ { m }$ be the elements of V, so that every $x _ { u }$ is one of the $z _ { j }$ . There are three cases for η.

$\eta \leq z _ { 1 }$ . Then $\eta \leq x _ { u }$ for all $u ,$ so every term $( \eta - x _ { u } ) ^ { + }$ is 0 and $\varphi ( \eta ) = \eta \leq z _ { 1 } = \varphi ( z _ { 1 } )$

$\eta \geq z _ { m }$ . Then $( \eta - x _ { u } ) ^ { + } = \eta - x _ { u }$ for all u, so $\begin{array} { r } { \varphi ( \eta ) = \left( 1 - \frac { 1 } { \alpha } \right) \eta + \frac { 1 } { \alpha } \sum _ { u } p _ { u } x _ { u } . \mathrm { A s } \alpha \leq 1 } \end{array}$ the coefficient of η is at most 0, hence $\varphi ( \eta ) \leq \varphi ( z _ { m } )$

$z _ { j } \ \leq \ \eta \ \leq \ z _ { j + 1 }$ . No $x _ { u }$ lies strictly between $z _ { j }$ and $z _ { j + 1 }$ , so for each u either $x _ { u } ~ \le ~ z _ { j }$ and $( \eta - x _ { u } ) ^ { + } ~ = ~ \eta - x _ { u }$ on the whole interval, or $x _ { u } ~ \ge ~ z _ { j + 1 }$ and $( \eta - x _ { u } ) ^ { + } = 0$ on the whole interval. Hence $\varphi ( \eta ) = a + b \eta$ on $[ z _ { j } , z _ { j + 1 } ]$ for some constants $a , b$ (namely $\begin{array} { r } { b = 1 - \frac { 1 } { \alpha } \sum _ { u : x _ { u } \leq z _ { i } } p _ { u } ) } \end{array}$ . Write $\eta = ( 1 - t ) z _ { j } + t z _ { j + 1 }$ with $t \in [ 0 , 1 ]$ . Then $\varphi ( \eta ) =$ $( 1 - t ) \varphi ( z _ { j } ) + t \varphi ( \bar { z } _ { j + 1 } ) \leq \operatorname* { m a x } \{ \varphi ( z _ { j } ) , \varphi ( z _ { j + 1 } ) \}$ , since a weighted average of two numbers is at most the larger of them.

In every case $\varphi ( \eta ) \leq \operatorname* { m a x } _ { j } \varphi ( z _ { j } )$ , and $\operatorname* { m a x } _ { j } \varphi ( z _ { j } ) \leq \operatorname* { s u p } \varphi ,$ so ${ \mathsf { C V a R } } _ { \alpha } \ = \ \operatorname* { m a x } _ { z \in \mathcal { V } } \varphi ( z ) \ = \ $ $\operatorname* { m a x } _ { z \in \mathcal { V } } g _ { z }$ . Here $( z - x _ { u } ) ^ { + }$ is a rational constant because $z , x _ { u } \in \mathcal { V } _ { : }$ so each $g _ { z }$ is a polynomial in the $p _ { u }$ . For ERisk , $S \geq \operatorname* { m i n } _ { u } e ^ { \theta x _ { u } } > 0$ , so $f$ is well defined and continuous. In all cases $f$ is continuous on $X \times Y$

Step 4: quantifier form. Since Y is compact and $f$ is continuous, $\begin{array} { r } { g ( x ) \ = \ \operatorname* { m i n } _ { y \in Y } f ( x , y ) } \end{array}$ is attained and continuous in $x ,$ and since $X$ is compact, $\operatorname* { m a x } _ { x \in X } g ( x )$ is attained. Hence $\mathsf { V a l } =$ $\mathrm { m a x } _ { x } \mathrm { m i n } _ { y } f ( x , y )$ , and

$$
{ \mathsf { V a l } } \triangleright c \iff \exists x \in X \forall y \in Y : f ( x , y ) \triangleright c ,
$$

for both choices of ▷. (For $> : \operatorname* { m a x } _ { x } g ( x ) > c$ iff some x has min $\iota _ { y } f ( x , y ) > c$ iff some x has $f ( x , y ) > c$ for all $y ,$ , because the minimum is attained.)

Step $5 \colon \mathbb { E } , \mathsf { M V } _ { \lambda } , \mathsf { C V a R } _ { \alpha }$ . The right-hand side of the equivalence is a sentence of the first-order theory of the reals. The memberships $x \in X$ and $y \in Y$ are conjunctions of polynomial (in)equalities with rational coefficients, and $f ( x , y ) \triangleright c$ is a polynomial inequality in the first two cases and a finite disjunction $\mathsf { V } _ { z \in \mathcal { V } } g _ { z } ( x , y ) \vartriangleright \cdot$ c of polynomial inequalities in the third. The theory is decidable by Tarski’s quantifier elimination [48].

Step 6: ERisk<sub>θ</sub> with rational θ. Tarski’s theorem covers only formulas built from polynomials, whereas ERisk<sub>θ</sub> involves exp and log. The first-order theory of the reals with the exponential function is only known to be decidable under Schanuel’s conjecture, so we cannot apply it to the exponential sentence directly. Instead we observe that, for rational data, the exponential enters only through one fixed real number, and we reduce to a polynomial sentence in which this number appears as a single parameter.

Reduction to one parameter. Since log is increasing, <sup>1</sup> log S ▷ c is equivalent to $S \triangleright e ^ { \theta c } { \mathrm { i f ~ } } \theta > 0$ and to the same inequality with the direction reversed if $\theta < 0$ . The numbers θx with $x \in \mathcal { V } \cup \{ c \}$ are rational, so there is a positive integer $D$ with $\textstyle { \theta x } \in { \frac { 1 } { D } } \mathbb { Z }$ for all of them. Put $\tau = e ^ { 1 / D } > 1$ . Then $e ^ { { \theta } x _ { u } } = \tau ^ { m _ { u } }$ and $e ^ { \theta c } = \tau ^ { m _ { c } }$ for integers $m _ { u } , m _ { c }$ , and the condition reads

$$
\sum _ { u } p _ { u } ( x , y ) \tau ^ { m _ { u } } \ \triangleright \ \tau ^ { m _ { c } }
$$

(direction reversed if $\theta < 0 )$ . Choose $M \in \mathbb { N }$ with $m _ { u } + M \ge 0$ for all u and $m _ { c } + M \ge 0$ , and multiply by $\tau ^ { M } > 0$ to remove negative exponents. Let $\varphi ( z )$ be the formula

$$
\exists x \in X \forall y \in Y : \ \sum _ { u } p _ { u } ( x , y ) z ^ { m _ { u } + M } \triangleright z ^ { m _ { c } + M }
$$

(with the direction reversed if $\theta < 0 )$ , with one free variable z. It is a first-order formula over the reals with rational coefficients, and by Step 4, Val $\vartriangleright c$ holds if and only if $\varphi ( \tau )$ holds. So it remains to decide $\varphi ( \tau )$

Why $\varphi ( \tau )$ is decidable. The idea is that $\varphi ( z )$ is a polynomial sentence, so the set of z satisfying it has a simple shape, and τ lies strictly inside one piece of that shape. We proceed in three parts.

(i) The shape of $Z = \{ z > 0 : \varphi ( z ) \}$ . By Tarski’s quantifier elimination [48], we can compute a quantifier-free formula $\psi ( z )$ equivalent to $\varphi ( z )$ over the reals. It is a Boolean combination of atoms $p _ { j } ( z ) = 0$ and $p _ { j } ( z ) > 0$ for finitely many polynomials $p _ { j }$ with rational coefficients. Let $0 = e _ { 0 } < e _ { 1 } < \dots < e _ { k }$ be 0 together with all positive real roots of $p _ { j }$ which is not the zero polynomial; these are algebraic numbers and can be computed. Let $J$ be an open interval between two consecutive points of $\{ e _ { 0 } , \ldots , e _ { k } , \infty \}$ . Then no $p _ { j }$ has a root in $J .$ If some $p _ { j }$ took values of both signs on $J ,$ say $p _ { j } ( a ) > 0 > p _ { j } ( b )$ with $a , b \in J$ , then by the intermediate value theorem $p _ { j }$ would have a root between a and b, hence in J, which is impossible. So every $p _ { j }$ has constant sign on $^ { J , }$ every atom has the same truth value on J, and so does $\psi .$

Fix an interval J as above. Every atom $p _ { j } ( z ) = 0 \mathrm { o r } p _ { j } ( z ) > 0$ has the same truth value at all $z \in J \colon$ this is clear if $p _ { j } \equiv 0 { : }$ , and otherwise $p _ { j }$ has constant sign on J and no root there, so $p _ { j } ( z ) = 0$ is false throughout J and $p _ { j } ( z ) > 0$ is either true throughout J or false throughout J. The formula ψ is built from these atoms using $\neg , \land$ and $\vee ,$ and these connectives preserve the property of having the same truth value at all points of J (by induction on the structure of $\psi ,$ , the truth value of a negation, conjunction or disjunction is determined by the truth values of its parts, which are constant on J). Hence ψ has the same truth value at all points of J. Since $\psi$ is equivalent to $\varphi$ and $J \subseteq ( 0 , \infty )$ either $\varphi ( z )$ holds for all $z \in J .$ , that is, $J \subseteq Z , \operatorname { o r } \varphi ( z )$ fails for all $z \in J$ , that is, $J \cap Z = \emptyset .$

Hence either $J \subseteq Z$ or $J \cap Z = \emptyset$

(ii) τ lies strictly inside one interval J. The number $\tau = e ^ { 1 / D }$ is not algebraic: otherwise $e = \tau ^ { D }$ would be algebraic, contradicting the transcendence of e [4]. In particular $\tau \neq e _ { i }$ for every $i ,$ so τ belongs to exactly one of the open intervals J from (i), and $\tau \in Z$ if and only if that whole interval J is contained in $Z .$

(iii) Finding the interval. We need not know τ exactly, only which interval contains it. For $n =$ $1 , 2 , \ldots$ , compute a rational interval $I _ { n } \subseteq ( 1 , \infty )$ of length at most $2 ^ { - n }$ that contains τ (for instance from the power series of $e ^ { 1 / D }$ with an explicit bound on the remainder), and refine rational isolating intervals of $e _ { 1 } , \ldots , e _ { k }$ to length at most $2 ^ { - n }$ . Stop at the first n for which $I _ { n }$ is disjoint from all of these isolating intervals. This happens after finitely many steps, because by (ii) τ has positive distance from each $e _ { i } .$ At that point no $e _ { i }$ lies in $I _ { n } ,$ so $I _ { n }$ is contained in a single interval $^ { J , }$ namely the one that contains τ. Let $q$ be the left endpoint of $I _ { n \cdot } \ : \mathbf { B y } \ : ( \mathrm { i } ) , \tau \in Z \ : \mathrm { i f }$ and only i ${ \bf { \dot { q } } } \in Z$ , that is, if and only if the sentence $\varphi ( q )$ holds. This sentence has only rational constants, so it is decided by Tarski’s theorem.

(For entropic risk in a rational base, as in [3], the constants $b ^ { \theta x }$ are real algebraic, and Tarski’s theorem applies directly.)

Conclusion. The decision procedure for ${ \mathsf { V a l } } { \mathsf { D c } }$ with $\mathsf { E R i s k } _ { \theta }$ is therefore as follows. Compute $D ,$ , the exponents $m _ { u } , m _ { c }$ and $M ,$ and the formula $\varphi ( z )$ . Compute the quantifier-free formula $\psi ( z )$ and the algebraic numbers $e _ { 1 } < \cdots < e _ { k }$ . Compute the rational number $q$ as in (iii). Decide the sentence $\varphi ( q )$ by Tarski’s theorem and output its answer. This is correct because

$$
\mathsf { V a l } \triangleright c \iff \varphi ( \tau ) \iff \tau \in Z \iff q \in Z \iff \varphi ( q ) ,
$$

where the first equivalence is Step 4 together with the reduction above, the third holds because $\tau$ and q lie in the same interval J and $J \subseteq Z { \mathrm { ~ o r ~ } } J \cap Z = \emptyset { \mathrm { ~ b y ~ ( i ) } }$ ), and the others are the definitions of $Z .$ Every step is effective, and the procedure terminates because τ is not one of the $e _ { i }$ by (ii). Hence ${ \mathsf { V a l } } \triangleright c$ is decidable for $\mathsf { E R i s k } _ { \theta }$

Step $7 \colon a c { - } P O S G s$ . The same argument works directly for an ac-POSG with a finite horizon. Apply the claim for RPOMDPs to the game obtained from the equivalence of Theorem 1(b).

(b) We proceed in five steps. Let $R = \operatorname* { m a x } | r |$ be the largest absolute reward and $M = R / ( 1 - \gamma )$ Step 1: truncation. Let $\begin{array} { r } { X _ { N } = \sum _ { t < N } \gamma ^ { t } r _ { t } } \end{array}$ be the discounted reward of the first N steps. On every play,

$$
\left| \mathsf { D i s c } _ { \gamma } - X _ { N } \right| = \left| \sum _ { t \ge N } \gamma ^ { t } r _ { t } \right| \le R \sum _ { t \ge N } \gamma ^ { t } = \frac { \gamma ^ { N } R } { 1 - \gamma } = : \epsilon _ { N } ,
$$

and both Disc<sub>γ</sub> and $X _ { N }$ take values in $[ - M , M ]$

Step $2 \cdot$ the functionals are Lipschitz. Let $Z , Z ^ { \prime }$ be random variables on the same probability space with values in $[ - M , M ]$ and $| Z - Z ^ { \prime } | \le \epsilon$ everywhere; we will apply this to $Z = \mathsf { D i s c } _ { \gamma }$ and $Z ^ { \prime } = X _ { N }$ under the same profile, with $\epsilon = \epsilon _ { N }$ . We show $| \varPsi ( Z ) - \varPsi ( Z ^ { \prime } ) | \leq K \epsilon$ (writing $\varPsi ( Z )$ for the value of $\varPsi$ at the law of $Z )$ , with $K = 1$ for E, $\mathsf { C V a R } _ { \alpha }$ , and ERisk<sub>θ</sub>, and $K = 1 + 4 \lambda M$ for ${ \mathsf { M V } } _ { \lambda }$

$$
\bullet \ \lvert \mathbb { E } Z - \mathbb { E } Z ^ { \prime } \rvert \leq \mathbb { E } \lvert Z - Z ^ { \prime } \rvert \leq \epsilon .
$$

$\mathsf { C V a R } _ { \alpha }$ and ERisk are monotone: $Z \leq Z ^ { \prime }$ everywhere implies $\varPsi ( Z ) \leq \varPsi ( Z ^ { \prime } )$ . For $\mathsf { C V a R } _ { \alpha }$ this holds because $( \eta - Z ) ^ { + } \geq ( \eta - Z ^ { \prime } ) ^ { + }$ for every η, so the expression under the supremum in its definition is smaller for $Z .$ . For ERisk , this holds because $\scriptstyle { \frac { 1 } { \theta } } \log \mathbb { E } [ e ^ { \theta Z } ]$ increases with $Z \colon$ for $\theta > 0$ , both $z \mapsto e ^ { \theta z }$ and $y \mapsto { \frac { 1 } { \theta } } \log y$ are increasing, and for $\theta < 0 .$ , both are decreasing. They are also translation equivariant: $\varPsi ( Z + m ) = \varPsi ( Z ) + m$ for every constant $m ,$ as one sees by substituting $\eta = \eta ^ { \prime } + m$ in the definition of ${ \mathsf { C V a R } } _ { \alpha } ,$ and from $\mathbb { E } [ e ^ { \theta ( Z + m ) } ] =$ $e ^ { \theta m } \mathbb { E } [ e ^ { \theta Z } ]$ for $\mathsf { E R i s k } _ { \theta }$ . Since $Z ^ { \prime } - \epsilon \le Z \le Z ^ { \prime } + \epsilon$ , we get $\varPsi ( Z ^ { \prime } ) - \epsilon \leq \varPsi ( Z ) \leq \varPsi ( Z ^ { \prime } ) + \epsilon$

• For the variance, $| \mathsf { V a r } Z - \mathsf { V a r } Z ^ { \prime } | \leq | \mathbb { E } Z ^ { 2 } - \mathbb { E } Z ^ { \prime 2 } | + | ( \mathbb { E } Z ) ^ { 2 } - ( \mathbb { E } Z ^ { \prime } ) ^ { 2 } |$ . Since $| Z ^ { 2 } - Z ^ { \prime 2 } | =$ $| Z - Z ^ { \prime } | \cdot | Z + Z ^ { \prime } | \le 2 M \epsilon$ , the first term is at most 2M ϵ; likewise, $| ( \mathbb { E } Z ) ^ { 2 } - ( \mathbb { E } Z ^ { \prime } ) ^ { 2 } | =$ $| \mathbb { E } Z - \mathbb { E } Z ^ { \prime } | \cdot | \mathbb { E } Z + \mathbb { E } Z ^ { \prime } | \le 2 M \epsilon$ . Hence $| \mathsf { M V } _ { \lambda } ( Z ) - \mathsf { M V } _ { \lambda } ( Z ^ { \prime } ) | \le \epsilon + 4 \lambda M \epsilon$

Step 3: the values are close. By Step 2, $| \varPsi ( \mathcal { L } _ { \sf D i s c _ { \alpha } } ^ { \sigma , \pi } ) - \varPsi ( \mathcal { L } _ { X _ { N } } ^ { \sigma , \pi } ) | \leq K \epsilon _ { N }$ for every profile $( \sigma , \pi )$ If two functions on profiles differ by at most δ everywhere, then so do their infima over π and the suprema of these over σ. Hence $| \mathsf { V a l } ( \varPsi , \mathsf { D i s c } _ { \gamma } ) - \mathsf { V a l } ( \varPsi , X _ { N } ) | \leq K \epsilon _ { N }$

Step 4: the truncated values are computable. Let $\begin{array} { r } { X _ { N } = \sum _ { t < N } \gamma ^ { t } r _ { t } } \end{array}$ be the discounted return truncated after N steps, where $\gamma$ is rational and $r _ { t }$ is the reward of the t-th transition. It depends only on the prefix of length N of the play, and on each such prefix it takes a rational value. The decision procedure of (a) uses the return only through these values $x _ { u }$ (Step 2 of (a)), so it applies to $X _ { N }$ unchanged, even though the reward earned by a transition now depends on the time step. Hence, for every rational $q ,$ we can decide whether $\mathsf { V a l } ( \varPsi , X _ { N } ) \geq q$

Moreover, the value lies in a rational interval $[ l _ { 0 } , u _ { 0 } ]$ that we can compute. Let M be the largest absolute reward and $\begin{array} { r } { R = M \sum _ { t < N } \gamma ^ { t } } \end{array}$ , so that $| X _ { N } | \le R$ on every play. For E, $\mathsf { C V a R } _ { \alpha }$ and ERisk<sub>θ</sub>, the value lies between the smallest and the largest value of the return, so $[ l _ { 0 } , u _ { 0 } ] = [ - R , R ]$ works. For ${ \mathsf { M V } } _ { \lambda }$ , the variance is at most $R ^ { 2 }$ , so $[ l _ { 0 } , u _ { 0 } ] = [ - R - \lambda R ^ { 2 } , R ]$ works.

Now fix a precision $\delta > 0$ . We run bisection. Maintain rationals $l \le \mathsf { V a l } ( \varPsi , X _ { N } ) \le u$ , starting with $l = l _ { 0 } { \mathrm { ~ a n d ~ } } u = u _ { 0 }$ . In each round let $\begin{array} { r } { m = \frac { l + u } { 2 } } \end{array}$ and use (a) to decide whether $\mathsf { V a l } ( \varPsi , X _ { N } ) \geq m$ . If so, set $l : = m$ , and otherwise set $u : = m$ . The invariant is preserved, and the length $u - l$ halves in each round. After $\lceil \log _ { 2 } ( ( u _ { 0 } - l _ { 0 } ) / \delta ) \rceil$ rounds, $u - l \leq \delta$ , so l approximates $\mathsf { V a l } ( \varPsi , X _ { N } )$ to within $\delta .$

Step $5 \colon$ approximation and semi-decidability. Given $\delta > 0 .$ , choose N with $K \epsilon _ { N } \le \delta / 2$ , which is possible since $\epsilon _ { N }  0$ , and compute $\mathsf { V a l } ( \varPsi , X _ { N } )$ up to precision $\delta / 2$ by Step 4; by Step 3, the result is within $\delta$ of $\mathsf { V a l } ( \varPsi , \mathsf { D i s c } _ { \gamma } )$ . For the strict threshold problem, we claim that $\mathsf { V a l } ( \varPsi , \mathsf { D i s c } _ { \gamma } ) > c$ holds if and only if $\mathsf { V a l } ( \varPsi , X _ { N } ) > c + K \epsilon _ { N }$ for some N. If $\mathsf { V a l } ( \varPsi , X _ { N } ) > c + K \epsilon _ { N }$ , then $\mathsf { V a l } ( \varPsi , \mathsf { D i s c } _ { \gamma } ) \geq \mathsf { V a l } ( \varPsi , X _ { N } ) - K \epsilon _ { N } > c$ by Step 3. Conversely, if $\mathsf { V a l } ( \varPsi , \mathsf { D i s c } _ { \gamma } ) > c ,$ , choose $N$ with $2 K \epsilon _ { N } < \mathsf { V a l } ( \varPsi , \mathsf { D i s c } _ { \gamma } ) - c ;$ then $\mathsf { V a l } ( \varPsi , X _ { N } ) \geq \mathsf { V a l } ( \varPsi , \mathsf { D i s c } _ { \gamma } ) - K \epsilon _ { N } > c + K \epsilon _ { N }$ . Since $K$ and $\epsilon _ { N }$ are rational, each condition $\mathsf { V a l } ( \varPsi , X _ { N } ) > c + K \epsilon _ { N }$ is a strict threshold problem with a rational threshold, which is decidable by Step 4. Checking these conditions for $N = 1 , 2 , . . .$ . and accepting as soon as one holds is a procedure that halts exactly when $\mathsf { V a l } ( \varPsi , \mathsf { D i s c } _ { \gamma } ) > c ,$ so the strict threshold problem is recursively enumerable.

ac-POSGs. Steps 3 to 5 use the model only through part (a), and through the bound $| \mathsf { D i s c } _ { \gamma } - X _ { N } | \leq$ $K \epsilon _ { N }$ on every play, which does not depend on the model. Part (a) holds for ac-POSGs (Step 7 of its proof), so the same argument gives the approximation and the semi-decidability of the strict threshold problem for ac-POSGs, with the same $K , \epsilon _ { N }$ and the same discount factor $\gamma .$

## C.7 Proof of Theorem 5

We reduce the SSG problem to both threshold problems for Ψ and X on RMDPs. We first view the SSG as a TBSG, shift its threshold (only for CVaR), and add rewards; we then translate the TBSG into an RMDP, determine the value of the RMDP, and finally compute a rational threshold.

The SSG as a TBSG. An SSG [11] has max-, min-, and random vertices, a 0-sink, and a 1-sink; max wants to reach the 1-sink, and min wants to prevent this. We view it as a TBSG in the sense of section 2.2: a random vertex becomes a max-state with a single action that moves to each of its two successors with probability $\textstyle { \frac { 1 } { 2 } }$ , and the 1-sink becomes an absorbing target state $T ,$ and the 0-sink becomes an absorbing non-target state. Thus max wins exactly on the plays that reach $T .$

By a transformation of [11], which replaces every edge by a gadget of polynomially many random vertices, we may assume that the game is stopping, i.e., that it reaches a sink with probability 1 under all strategies. Then the value $x ^ { * }$ of the initial vertex, i.e., the probability of reaching $T$ that max can guarantee, is a rational number with denominator at most $4 ^ { n }$ , where n is the number of vertices [11]. The task is to decide whether $\begin{array} { r } { x ^ { * } > \frac { 1 } { 2 } } \end{array}$

Threshold shift. The CVaR profile $\beta _ { v } ^ { \mathsf { C V a R } _ { \alpha } } ( p ) = v \operatorname* { m a x } \{ 0 , 1 - ( 1 - p ) / \alpha \}$ is 0 for all $p \leq 1 - \alpha$ so it cannot distinguish small probabilities. For $\varPsi = \mathsf { C V a R } _ { \alpha }$ , we therefore add a new initial vertex, a max-state with a single action that moves to $T$ with probability $\mu = 1 - \alpha$ and to the old initial vertex otherwise; the game remains stopping. Let $x ^ { \prime }$ be the value of the new initial vertex. Then $x ^ { \prime } = \mu + ( 1 - \mu ) x ^ { * } \in [ 1 - \alpha , 1 ]$ , which lies in the range where the CVaR profile is strictly increasing. As x<sup>′</sup> is a strictly increasing affine function of $x ^ { * }$ , we have $x ^ { * } > \frac { 1 } { 2 }$ if and only if $x ^ { \prime } > q$ for $q = \mu + ( 1 - \mu ) { \textstyle { \frac { 1 } { 2 } } } = 1 - { \frac { \alpha } { 2 } }$ . For the other objectives, let $\mu = 0 , x ^ { \prime } = x ^ { * }$ , and $\begin{array} { r } { q = \frac { 1 } { 2 } } \end{array}$

Rewards. Let $v = 1 / ( 1 + \lambda )$ for ${ \mathsf { M V } } _ { \lambda }$ and $v = 1$ otherwise. For Tot, we redirect every transition into T to a fresh state g that moves to an absorbing sink $z ;$ every transition into g earns v, and all other transitions earn 0. This ensures that the reward v is collected exactly once on plays that reach the target, since the absorbing state T itself would be entered again at every step.

For LimAvg, every transition into a state of the absorbing set $T$ earns $v ,$ and all other transitions earn 0. In both cases, rewards depend only on the target state of a transition. On every play that reaches T (for Tot: that reaches $g )$ , the return X equals v: for Tot, the reward v is collected exactly once, and for LimAvg, all rewards from some point on equal v. On all other plays, all rewards are 0, so $X = 0$ . Hence every profile $( \sigma , \pi )$ induces the law Ber $, ( p ^ { \sigma , \pi } )$ , where $p ^ { \sigma , \pi }$ is the probability of reaching T.

Here, the TBSG constructed need not be alternating, and for this we use the following variant of the backward construction 3. The construction normally records the action of min in the state it leads to; this record only serves to transfer rewards that depend on the action of min. In our case, all rewards depend only on the target state of a transition, so we omit the record and keep the states of the TBSG $\mathcal { G } \colon$ the agent plays its actions in the max-states, where the environment has no choice, and the dummy action $\top$ in every min-state $t ,$ where the environment chooses from conv $\{ \delta ( t , b ) : b \in A ( t ) \}$ ; the transition from t to $s ^ { \prime }$ earns the reward of the target state $s ^ { \prime }$

As both players observe the states of the TBSG, the result is an RMDP M with the same achievable return laws as $\mathcal { G } ;$ we explain why.

Why M and G have the same achievable return laws. The states, the actions of the agent in the max-states, and the rewards of $\mathcal { M }$ and $\mathcal { G }$ coincide, and the agent observes the current state in both. The two models differ only in the min-states $t { : }$ in ${ \mathcal { G } } _ { : }$ , min chooses a mixed action $\lambda$ on $A ( t )$ , and the next state is distributed according to $\sum _ { b } \lambda ( b ) \delta ( t , b )$ ; in $\mathcal { M } ,$ the environment chooses a point $\mu \in \mathrm { c o n v } \{ \delta ( t , b ) : b \in A ( t ) \}$ directly, and the next state is distributed according to $\mu .$ . Every $\lambda$ yields such a point, and every such point arises from some $\lambda .$ Hence a strategy of min in $\mathcal { G }$ induces a strategy of the environment in $\mathcal { M }$ (map each history to the point obtained from the mixed action), and conversely a strategy of the environment induces a strategy of min (map each history to a fixed decomposition $\lambda$ of the selected point, chosen once for each point). In both directions the strategy of the agent is unchanged, and the probability of every finite play prefix is the product of the initial probability, the probabilities of the agent’s actions, and the next-state probabilities, which are the same in both models. Hence the two profiles induce the same probability measure on plays. Since the rewards depend only on the states, they induce the same law of the return, so M and $\mathcal { G }$ have the same achievable return laws.

From the TBSG to an RMDP. Applying this variant to the shifted TBSG above, which need not be alternating, yields an RMDP M with the same achievable return laws as the shifted TBSG. Together with the rewards defined above, every profile of $\mathcal { M }$ therefore induces the law $\mathsf { B e r } _ { v } ( p )$ , where $p$ is the probability of reaching $T$ under the corresponding profile $( \sigma , \pi )$ of the shifted TBSG; we write $p ^ { \sigma , \pi }$ for it.

The value of M. Write $\beta = \beta _ { v } ^ { \Psi }$ . We compute $\mathsf { V a l } _ { \mathcal { M } } ( \varPsi , X )$ in four steps.

1. Reduce to a function $o f p .$ . By the definition of the profile $\beta$ and of the value,

$$
\mathsf { V a l } _ { \mathcal { M } } ( \varPsi , X ) = \operatorname* { s u p } _ { \sigma } \operatorname* { i n f } _ { \pi } \varPsi \bigl ( \mathsf { B e r } _ { v } ( p ^ { \sigma , \pi } ) \bigr ) = \operatorname* { s u p } _ { \sigma } \operatorname* { i n f } _ { \pi } \beta ( p ^ { \sigma , \pi } ) ,
$$

where $\sigma$ and π range over the strategies of the TBSG.

2. $\beta$ is continuous and nondecreasing. For $\mathbb { E } , \mathsf { C V a R } _ { \alpha }$ and ERisk this is the content of Lemma C.1 For ${ \mathsf { M V } } _ { \lambda }$ it holds because $\lambda v < 1 \colon$ the derivative of $\beta _ { v } ^ { \mathsf { M V } _ { \lambda } }$ is $v - \lambda v ^ { 2 } ( 1 - 2 p ) \geq v ( 1 - \lambda v ) > 0$

3. Move $\beta$ outside. By (F1), a continuous nondecreasing function commutes with suprema and infima. Applying this to the infimum over π for each fixed $\sigma$ , and then to the supremum over $\sigma ,$ we get

$$
{ \mathsf { V a l } } _ { \mathcal { M } } ( \varPsi , X ) = \beta \Bigl ( \operatorname* { s u p } _ { \sigma } \operatorname* { i n f } _ { \pi } p ^ { \sigma , \pi } \Bigr ) .
$$

4. Identify the argument $o f \beta .$ . The quantity $\operatorname { s u p } _ { \sigma }$ in $\mathrm { f } _ { \pi } p ^ { \sigma , \pi }$ is the value of the reachability game for our classes of strategies. In the stopping game, both players have optimal MD strategies [11], and these belong to our classes, since both players observe the current state of the TBSG. By (F2), the value for our classes therefore equals the classical value of the game, that ${ \mathrm { i s } } ,$ the value $x ^ { \prime }$ of its initial vertex in the (shifted) TBSG. Hence $\mathsf { V a l } _ { \mathcal { M } } ( \varPsi , X ) = \beta ( x ^ { \prime } )$

Thresholds. By the previous step, $\mathsf { V a l } _ { \mathcal { M } } ( \varPsi , X ) = \beta ( x ^ { \prime } )$ , where $x ^ { \prime } \geq \mu$ is the value of the shifted game, and we know that $x ^ { * } > \frac { 1 } { 2 }$ if and only i $\ : x ^ { \prime } > q$ . It remains to translate the question whether $x ^ { \prime } > q$ into threshold questions about $\beta ( x ^ { \prime } )$

$( i ) \ A$ gap for the value of the SSG. The value $x ^ { * }$ is a rational number $p / d$ with $d \leq 4 ^ { n } [ 1 1 ]$ . If $\begin{array} { r } { x ^ { * } > \frac { 1 } { 2 } } \end{array}$ , then $2 p > d ,$ , so $2 p - d \geq 1$ as an integer, and hence $x ^ { * } - { \frac { 1 } { 2 } } = ( 2 p - d ) / ( 2 d ) \geq 1 / ( 2 \cdot 4 ^ { n } )$ Thus

$$
\begin{array} { r } { x ^ { * } > \frac { 1 } { 2 } \iff x ^ { * } \geq \frac { 1 } { 2 } + \frac { 1 } { 2 \cdot 4 ^ { n } } . } \end{array}
$$

(ii) The gapfor x<sup>′</sup>. Recall that $x ^ { \prime } = \mu + ( 1 - \mu ) x ^ { * }$ and $q = \mu + ( 1 - \mu ) \frac { 1 } { 2 }$ . The map $x \mapsto \mu + ( 1 - \mu ) x$ is strictly increasing, since $\mu < 1$ , and it sends $\textstyle { \frac { 1 } { 2 } }$ to q and $\textstyle { \frac { 1 } { 2 } } + { \frac { 1 } { 2 \cdot 4 ^ { n } } }$ to $q + g$ with $g = ( 1 - \mu ) / ( 2 \cdot 4 ^ { n } )$ Applying it to both sides of the equivalence in (i), we get

$$
x ^ { \prime } > q \iff x ^ { \prime } \geq q + g .
$$

Moreover, $\mu \leq q < q + g \leq 1$ , because $\begin{array} { r } { \frac { 1 } { 2 } + \frac { 1 } { 2 \cdot 4 ^ { n } } \leq 1 } \end{array}$ . So $q , q + g ,$ , and $x ^ { \prime }$ all lie in $[ \mu , 1 ]$ , which is the interval on which $\beta$ is strictly increasing (it is $[ 0 , 1 ]$ for $\mathbb { E } , \mathsf { M V } _ { \lambda }$ , and $\mathsf { E R i s k } _ { \theta }$ since $\mu = 0$ there, and $[ 1 - \alpha , 1 ]$ for $\mathsf { C V a R } _ { \alpha } )$

(iii) Thresholds for E, ${ \mathsf { M V } } _ { \lambda }$ , and $\mathsf { C V a R } _ { \alpha }$ . Since $\beta$ is strictly increasing on $[ \mu , 1 ] , a > b$ holds if and only if $\beta ( a ) > \beta ( b )$ , and $a \geq b$ holds if and only $\mathrm { i f } \ \beta ( a ) \geq \beta ( b )$ , for $a , b \in [ \mu , 1 ]$ . Together with (ii) and $\mathsf { V a l } _ { \mathcal { M } } ( \varPsi , X ) = \beta ( x ^ { \prime } )$ this gives

$$
x ^ { * } > { \frac { 1 } { 2 } } \iff x ^ { \prime } > q \iff { \mathsf { V a l } } _ { { \mathcal { M } } } ( \psi , X ) > \beta ( q ) \iff { \mathsf { V a l } } _ { { \mathcal { M } } } ( \psi , X ) \geq \beta ( q + g ) .
$$

For $\mathbb { E } , \mathsf { M V } _ { \lambda }$ , and ${ \mathsf { C V a R } } _ { \alpha } ,$ the profile $\beta$ is a polynomial, respectively $\beta ( p ) = p v , \beta ( p ) = p v -$ $\lambda v ^ { 2 } p ( 1 - p )$ , and $\beta ( p ) = v ( 1 - ( 1 - p ) / \alpha ) \mathrm { o n } [ 1 - \alpha , 1 ]$ , with rational coefficients. The numbers q and g are rationals of bit size $O ( n )$ , so $\beta ( q )$ and $\beta ( q + g )$ are rationals of polynomial size, obtained by a constant number of arithmetic operations. Hence the instances “is $\mathsf { V a l } _ { \mathcal { M } } ( \varPsi , X ) > \beta ( q ) ? ^ { , , }$ of the strict threshold problem and “is $\mathsf { V a l } _ { \mathcal { M } } ( \varPsi , X ) \geq \beta ( q + g ) \ ? ^ { , }$ of the non-strict one are computed in polynomial time, and each is a yes-instance if and only if $\begin{array} { r } { x ^ { * } > \frac { 1 } { 2 } } \end{array}$

(iv) Thresholds $f o r \mathsf { E R i s k } _ { \theta }$ . Here $\begin{array} { r } { \beta ( p ) = \frac { 1 } { \theta } \log ( 1 - p + p e ^ { \theta v } ) } \end{array}$ , and the numbers $\beta ( q )$ and $\beta ( q + g )$ are in general irrational, while the threshold of a threshold problem must be rational. We therefore choose a rational c between them. The derivative

$$
\beta ^ { \prime } ( p ) = { \frac { e ^ { \theta v } - 1 } { \theta \left( 1 - p + p e ^ { \theta v } \right) } }
$$

is positive on $[ 0 , 1 ]$ , since the numerator and θ have the same sign and $1 - p + p e ^ { \theta v }$ is a convex combination of 1 and $e ^ { \theta v }$ . Therefore $\beta ^ { \prime } ( p ) \geq C : = | e ^ { \theta v } - 1 | / ( | \theta | \operatorname* { m a x } \{ 1 , e ^ { \theta v } \} ) > 0 .$ , a constant that depends only on $\theta$ and $v ,$ and fix a rational $0 < C ^ { \prime } \leq C$ . By the mean value theorem, $\beta ( q +$ $g ) - \beta ( q ) \geq C ^ { \prime } g .$ . Compute rationals $b _ { 1 } , b _ { 2 }$ with $| b _ { 1 } - \beta ( q ) | \le C ^ { \prime } g / 4$ and $| b _ { 2 } - \beta ( q + g ) | \le$ $C ^ { \prime } g / 4$ , which takes polynomial time since $g \geq 2 ^ { - O ( n ) }$ requires only $O ( n )$ bits of precision, and let $c = ( b _ { 1 } + b _ { 2 } ) / 2$ . Let $m = ( \beta ( q ) + \beta ( q + g ) ) / 2$ . Then $| c - m | \le C ^ { \prime } g / 4$ , whereas $m - \beta ( q ) =$ $\beta ( q + g ) - m \geq C ^ { \prime } g / 2 .$ , so

$$
\beta ( q ) < c < \beta ( q + g ) .
$$

Now if $x ^ { * } > \frac { 1 } { 2 }$ , then $x ^ { \prime } \geq q + g$ by (ii), so $\begin{array} { r } { \mathsf { V a l } _ { \mathcal { M } } ( \varPsi , X ) = \beta ( x ^ { \prime } ) \ge \beta ( q + g ) > c . \mathrm { ~ I f ~ } x ^ { * } \le \frac { 1 } { 2 } , } \end{array}$ 2 then $x ^ { \prime } \leq q , \bar { \mathsf { s o } } \mathsf { V a l } _ { \mathcal { M } } ( \varPsi , X ) = \beta ( x ^ { \prime } ) \leq \beta ( q ) < c .$ . Hence both “is $\mathsf { V a l } _ { \mathcal { M } } ( \varPsi , X ) > c \tau$ and ${ } ^ { 6 6 } \mathrm { \dot { 1 } s }$ $\mathsf { V a l } _ { \mathcal { M } } ( \varPsi , X ) \geq c ? ^ { \ast }$ are yes-instances if and only if $\begin{array} { r } { x ^ { * } > \frac { 1 } { 2 } } \end{array}$

In all cases, both threshold problems for M decide whether $\begin{array} { r } { x ^ { * } > \frac { 1 } { 2 } } \end{array}$ , with a threshold that is computed in polynomial time.

Complexity of the reduction. All steps run in polynomial time. For $\mathbb { E } , \mathsf { M V } _ { \lambda }$ , and $\mathsf { C V a R } _ { \alpha } .$ they even run in logarithmic space: the transformation into a stopping game, the threshold shift, and the new reward states are local modifications of the game; the construction of the RMDP from the shifted TBSG is computable in logarithmic space, like the backward construction (Theorem 1(c)); and the thresholds are obtained by a constant number of additions and multiplications on rationals of polynomial size. For $\mathsf { E R i s k } _ { \theta }$ , all steps except the computation of the threshold c are as above; the computation of c approximates logarithms and exponentials, which is in polynomial time. □

## C.8 Proofs of Theorem 6

Proof. The source problem. A discounted TBSG is a tuple $G = ( S _ { \sf m a x } , S _ { \sf m i n } , A , \delta , r , s _ { 0 } , \gamma )$ with disjoint finite sets of states $S _ { \mathsf { m a x } } , S _ { \mathsf { m i n } } \left( S = S _ { \mathsf { m a x } } \cup S _ { \mathsf { m i n } } \right)$ , finite action set A and let $A ( x )$ denote the set of effective actions at state x, transitions $\delta ( x , a ) \in { \varDelta ( S ) }$ , rewards $r ( x , a , y ) \in \mathbb { Q } .$ , an initial state $s _ { 0 } .$ , and a rational discount factor $\gamma \in ( 0 , 1 )$ given in binary. The owner of state x chooses the action in x, and the play collects $\textstyle \sum _ { n \geq 0 } \gamma ^ { n } r _ { n }$ . The players need not alternate. For $V \in \mathbb { R } ^ { S } , x \in S$ and $a \in A ( x )$ , let

$$
f _ { x , a } ( V ) = \sum _ { y } \delta ( x , a ) ( y ) \bigl ( r ( x , a , y ) + \gamma V ( y ) \bigr ) ,
$$

and let $\mathcal { G } \colon \mathbb { R } ^ { S }  \mathbb { R } ^ { S }$ be given by ${ \mathcal { G } } ( V ) ( x ) = \operatorname* { m a x } _ { a \in A ( x ) } f _ { x , a } ( V )$ for x $\in \ S _ { \mathsf { m a x } }$ and ${ \mathcal G } ( V ) ( x ) =$ $\mathrm { m i n } _ { a \in A ( x ) } f _ { x , a } ( V )$ for $x \in S _ { \operatorname* { m i n } }$ . Since $\delta ( x , a )$ is a distribution, $| f _ { x , a } ( V ) - f _ { x , a } ( V ^ { \prime } ) | \leq \gamma \| V -$ $V ^ { \prime } \| _ { \infty }$ . Moreover, for real numbers $u _ { i } , v _ { i }$ we have $\begin{array} { r } { | \operatorname* { m a x } _ { i } u _ { i } - \operatorname* { m a x } _ { i } v _ { i } | \leq \operatorname* { m a x } _ { i } | u _ { i } - v _ { i } | } \end{array}$ , and the same for min. Hence $\| \mathcal { G } ( V ) - \mathcal { G } ( V ^ { \prime } ) \| _ { \infty } \leq \gamma \| V - V ^ { \prime } \| _ { \infty } ,$ , so $\mathcal { G }$ is a contraction of $( \mathbb { R } ^ { S } , \| \cdot \| _ { \infty } )$ and, by Banach’s fixed-point theorem, has a unique fixed point W (Shapley’s equations). By [44], $W ( s _ { 0 } )$ is the value of the game $G ,$ and we write $\mathsf { V a l } _ { G } = W ( s _ { 0 } )$ . The threshold problemfor TBSGs asks, given $G$ and a rational c, whether $\mathsf { V a l } _ { G } \geq c$

[1] study discounted games of this kind (in their terminology, discounted Gillette games, where the reward depends only on the state and the action), with discount factor, rewards, and probabilities in binary. They prove that solving such a game, that is, computing optimal pure positional strategies, is polynomial-time equivalent to solving simple stochastic games, and they note that solving a game is polynomial-time Turing equivalent to comparing its value with a given number. Hence SSGs reduce to the threshold problem for discounted TBSGs as follows: deciding whether an SSG has value above $\textstyle { \frac { 1 } { 2 } }$ reduces to solving the SSG, which reduces to solving a discounted game, which reduces under polynomial-time Turing reductions to threshold queries for discounted games. In their games, the reward depends only on the state and the action. Such a game is a discounted TBSG in our sense, with $r ( x , a , y ) : = r ( x , a )$ for all successors y, and it has the same value.

Step 1: the RMDP. Given $G ,$ we build an RMDP $\mathcal { M } _ { G }$ with the same discount factor $\gamma ,$ in which the choice of min becomes a choice of the environment. Its states are $Q = \{ ( x , \tau ) : x \in S , \tau \in$ $\{ \bot \} \cup A _ { \sf m i n } \}$ , where $\begin{array} { r } { A _ { \operatorname* { m i n } } = \bigcup _ { t \in S _ { \operatorname* { m i n } } } A ( t ) } \end{array}$ , and the tag τ records the action of min in the preceding move, or ⊥ if that move was not made by min. The initial state is $( s _ { 0 } , \perp )$ ). Both observation functions are injective (we take $\mathcal { O } _ { \mathsf { a } } = \mathcal { O } _ { \mathsf { e } } = Q$ and $\mathcal { Z } _ { \mathfrak { a } } = \mathcal { Z } _ { \mathfrak { e } } = \mathrm { i d } )$ , so $\mathcal { M } _ { G }$ is an RMDP, and the agent and the environment see the whole state, including the tag. The agent therefore knows the last action of min, which max does not see in $G .$ . This does no harm, as Step 2 computes the value of $\mathcal { M } _ { G }$ directly: the optimal agent strategy ignores the tag, and the bound for the optimal environment strategy holds against all agent strategies.

• For $x \in S _ { \mathrm { { m a x } } }$ and every tag τ, the agent has the actions $A ( x )$ , and $\mathcal { P } ( ( x , \tau ) , a ) = \{ \hat { \delta } _ { x , a } \}$ with $\hat { \delta } _ { x , a } ( ( y , \perp ) ) = \delta ( x , a ) ( y )$ . The transition from $( x , \tau )$ under a to $( y , \perp )$ earns $r ( x , a , y )$ .

• For $t \in S _ { \operatorname* { m i n } }$ and every tag τ , the agent has the single action $\top$ , and

$$
\mathcal { P } ( ( t , \tau ) , \top ) = \mathrm { c o n v } \{ \hat { \delta } _ { t , b } : b \in A ( t ) \} , \qquad \hat { \delta } _ { t , b } ( ( y , b ) ) = \delta ( t , b ) ( y ) .
$$

The transition from $( t , \tau )$ to $( y , b )$ earns $r ( t , b , y )$ . This is well defined because the target determines b.

Every polytope has at most $| A ( t ) |$ vertices, so $\mathcal { M } _ { G }$ has polynomial size, and it is computable in logarithmic space, since every entry is a local copy of an entry of G. A point of $\mathcal { P } ( ( t , \tau ) , \top )$ is a mixture of actions of min, and the tag of the successor $( y , b )$ records which action b was drawn. The tag is needed for the reward: the transition earns $r ( t , b , y )$ , which depends on $b ,$ and in an RMDP the reward of a transition may depend only on its source state, its action, and its target state. With the tag in the target, the reward is paid on the same transition, with no extra step, which would change the timing and hence the discounting.

Step 2: the value $o f { \mathcal { M } } _ { G }$ . We show $\mathsf { V a l } _ { \mathcal { M } _ { G } } ( \mathbb { E } , \mathsf { D i s c } _ { \gamma } ) = W ( s _ { 0 } )$ . Since $\varPsi = \mathbb { E } , \mathsf { V a l } _ { \mathcal { M } _ { G } } ( \mathbb { E } , \mathsf { D i s c } _ { \gamma } ) =$ sup<sub>σ</sub> inf<sub>π</sub> $\mathbb { E } ^ { \sigma , \pi } [ \mathsf { D i s c } _ { \gamma } ]$ . Let $U ( x , \tau ) = W ( x )$ and $R = \operatorname* { m a x } | r |$ . Fix an agent strategy σ and an environment strategy $\pi _ { \mathrm { : } }$ let $x _ { 0 } a _ { 0 } x _ { 1 } a _ { 1 } \cdot \cdot$ · be the random play, let $r _ { n }$ be the reward of its n-th transition, let ${ \mathcal { F } } _ { n }$ be generated by the prefix $x _ { 0 } a _ { 0 } \cdots x _ { n }$ , and let

$$
Z _ { n } = \sum _ { k < n } \gamma ^ { k } r _ { k } + \gamma ^ { n } U ( x _ { n } ) .
$$

Given ${ \mathcal { F } } _ { n } ,$ the action $a _ { n }$ is drawn from a distribution $\mu _ { n }$ on $A ( x _ { n } )$ determined by σ; after $a _ { n } = a _ { \mathrm { { ; } } }$ the environment has fixed a point $p _ { n } ^ { a } \in \mathcal { P } ( x _ { n } , a )$ determined by π, and $x _ { n + 1 } \sim p _ { n } ^ { a }$ . Let $\varPhi _ { n } ( a ) =$ $\begin{array} { r } { \sum _ { z } p _ { n } ^ { a } ( z ) \big ( \hat { r } ( x _ { n } , a , z ) + \gamma U ( z ) \big ) } \end{array}$ , where rˆ is the reward function of $\mathcal { M } _ { G }$ . Since $Z _ { n + 1 } - Z _ { n } \ =$ $\gamma ^ { n } \left( r _ { n } + \gamma U ( x _ { n + 1 } ) - U ( x _ { n } ) \right)$ and $x _ { n }$ is ${ \mathcal { F } } _ { n }$ -measurable, it suffices to average $r _ { n } + \gamma U ( x _ { n + 1 } )$ given ${ \mathcal { F } } _ { n } \mathrm { { : } }$ first over $x _ { n + 1 } \sim p _ { n } ^ { a }$ for fixed $a _ { n } = a ,$ which gives $\varPhi _ { n } ( a )$ , then over $a _ { n } \sim \mu _ { n }$ . Hence

$$
\mathbb { E } [ Z _ { n + 1 } - Z _ { n } \mid { \mathcal F } _ { n } ] = \gamma ^ { n } { \Big ( } \sum _ { a } \mu _ { n } ( a ) \varPhi _ { n } ( a ) - U ( x _ { n } ) { \Big ) } .\tag{1}
$$

We evaluate $\varPhi _ { n }$ in the two kinds of states.

$x _ { n } = ( x , \tau )$ with $x \in S _ { \sf m a x } \colon p _ { n } ^ { a } = \widehat { \delta } _ { x , a } , \mathrm { s o } \ \varPhi _ { n } ( a ) = f _ { x , a } ( W )$ . This is $\leq W ( x )$ for every $^ { a , }$ because $W = \mathcal { G } ( W )$ , and it equals $W ( x )$ for an action $a ^ { * } ( x )$ attaining the maximum.

$x _ { n } = ( t , \tau )$ with $t \in S _ { \operatorname* { m i n } }$ : there is only the action ⊤, and $\begin{array} { r } { p _ { n } = \sum _ { b } \lambda _ { b } \widehat \delta _ { t , b } } \end{array}$ for some distribution λ on $A ( t )$ . Since the successor $( y , b )$ has $U = W ( y )$ and the transition earns $r ( t , b , y )$ we get $\begin{array} { r } { \varPhi _ { n } = \sum _ { b } \lambda _ { b } f _ { t , b } ( W ) } \end{array}$ . This is ≥ min<sub>b</sub> $f _ { t , b } ( W ) = W ( t )$ , and it equals $W ( t )$ if λ is concentrated on an action $b ^ { * } ( t )$ attaining the minimum.

By the tower property, $\operatorname { \mathbb { E } } [ Z _ { n + 1 } - Z _ { n } ] = \operatorname { \mathbb { E } } [ \operatorname { \mathbb { E } } [ Z _ { n + 1 } - Z _ { n } \mid { \mathcal { F } } _ { n } ] ]$ , and $Z _ { 0 } = U ( x _ { 0 } ) = W ( s _ { 0 } )$ is deterministic. Moreover, $Z _ { n } \to \mathsf { D i s c } _ { \harpoonright }$ on every play: the partial sums converge to $\mathsf { D i s c } _ { \gamma }$ because $| r _ { k } | \le R$ and $\scriptstyle \sum _ { k } \gamma ^ { k } R < \infty$ , and $| \gamma ^ { n } U ( x _ { n } ) | \leq \gamma ^ { n } \| W \| _ { \infty }  0$ , where $\| W \| _ { \infty }$ is finite because S is finite. Also $| Z _ { n } | \le R / ( 1 - \gamma ) + \| W \| _ { \infty }$ for all n. Hence, by dominated convergence, $\mathbb { E } [ Z _ { n } ] $ $\mathbb { E } [ \mathsf { D i s c } _ { \gamma } ]$ . We now use (1) in the two directions.

Lower bound. Let $\sigma ^ { * }$ play $a ^ { * } ( x )$ in every state $( x , \tau )$ with $x \in S _ { \mathrm { { m a x } } }$ . Fix any environment strategy π. At states of $S _ { \sf m a x }$ the right-hand side of (1) is $0 \left( \operatorname { a s } \phi _ { n } ( a ^ { * } ( x ) ) = W ( x ) \right)$ , and at states of $S _ { \mathrm { m i n } }$ it $\mathrm { i s } \geq 0$ (as $\phi _ { n } \geq W ( t ) )$ . So $\mathbb { E } [ Z _ { n + 1 } - Z _ { n } \mid { \mathcal { F } } _ { n } ] \geq 0$ , hence $\mathbb { i } [ Z _ { n + 1 } ] \geq \mathbb { E } [ Z _ { n } ]$ , and by induction $\mathbb { E } [ Z _ { n } ] \ge Z _ { 0 } = W ( s _ { 0 } )$ for all n. Letting $n  \infty$ gives $\mathbb { E } ^ { \sigma ^ { * } , \pi } [ \mathsf { D i s c } _ { \gamma } ] \geq W ( s _ { 0 } )$ . Since $\pi$ was arbitrary, in $\begin{array} { r } { \mathrm { f } _ { \pi } \mathbb { E } ^ { \sigma ^ { * } , \pi } [ \mathsf { D i s c } _ { \gamma } ] \geq W ( s _ { 0 } ) } \end{array}$ , and therefore $\mathsf { V a l } _ { \mathcal { M } _ { G } } \geq W ( s _ { 0 } )$

Upper bound. Let $\pi ^ { * }$ be the environment strategy that always picks the vertex $\hat { \delta } _ { t , b ^ { * } ( t ) } \mathrm { i n } \left( t , \tau \right)$ . Fix any agent strategy $\sigma ,$ randomized and history-dependent. At states of $S _ { \sf m a x }$ the right-hand side of (1) is $\leq 0$ (as $\varPhi _ { n } ( a ) \leq W ( x )$ for every $^ { a , }$ hence also for the average over $\mu _ { n } ) _ { : }$ , and at states of $S _ { \mathrm { m i n } }$ it is 0 (as $\varPhi _ { n } = W ( t )$ for the chosen vertex). So $\mathbb { E } [ Z _ { n + 1 } ] \leq \mathbb { E } [ Z _ { n } ]$ , hence $\mathbb { E } [ Z _ { n } ] \le Z _ { 0 } = W ( s _ { 0 } )$ for all $n ,$ and in the limit $\mathbb { E } ^ { \sigma , \pi ^ { * } } [ \mathsf { D i s c } _ { \gamma } ] \le W ( s _ { 0 } )$ . Therefore $\begin{array} { r } { \operatorname* { i n f } _ { \pi } \operatorname { \mathbb { E } } ^ { \mathcal { \bar { \sigma } } , \pi } [ \mathsf { D i s c } _ { \gamma } ] ^ { \cdot } \leq \operatorname { \bar { \mathbb { E } } } ^ { \sigma , \pi ^ { * } } [ \mathsf { D i s c } _ { \gamma } ] \leq W ( s _ { 0 } ) } \end{array}$ for every σ, and $\mathsf { V a l } _ { \mathcal { M } _ { G } } \leq W ( s _ { 0 } )$

Hence $\mathsf { V a l } _ { \mathcal { M } _ { G } } ( \mathbb { E } , \mathsf { D i s c } _ { \gamma } ) = W ( s _ { 0 } ) = \mathsf { V a l } _ { G } .$

Step 3: the reduction. By Step 2, $\mathsf { V a l } _ { G } \geq \mathrm { ~ } c$ if and only if $\mathsf { V a l } _ { \mathcal { M } _ { G } } ( \mathbb { E } , \mathsf { D i s c } _ { \gamma } ) \geq c ,$ and $\mathcal { M } _ { G }$ is a polytopic RMDP with the same binary discount factor, computable in logarithmic space from G. Given a polynomial-time Turing reduction from SSGs to the threshold problem for discounted TBSGs, we replace each oracle query $( G , c )$ by the query $( \mathcal { M } _ { G } , c )$ to the threshold problem for discounted RMDPs. This is a polynomial-time Turing reduction from SSGs to the latter problem. The same holds for the strict threshold problem, because the values coincide: any reduction from SSGs to a variant of the TBSG threshold problem yields a reduction to the same variant for RMDPs.