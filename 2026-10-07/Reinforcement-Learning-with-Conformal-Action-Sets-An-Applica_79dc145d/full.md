# Reinforcement Learning with Conformal Action Sets: An Application to Sequential Recommendation

Wenwen Si University of Pennsylvania

## Abstract

Sequential recommenders typically use a fixed slate size even though the number of useful alternatives changes within a session. We propose Reinforcement Learning with Calibrated Pruning (RLCP), which adapts the retained action set using critic scores and an online threshold. The threshold is updated from binary feedback indicating whether the set contains an action in a proxy target. We prove a deterministic bound on the observed proxy miss rate along adaptive trajectories. To quantify the efect of pruning on reward, we derive an exact decomposition of value loss into filtering and selection losses. Under explicit proxy and critic approximation conditions, this decomposition yields a finite session reward bound that also accounts for imperfect selection and set truncation, without requiring the learning parameters to converge. Experiments on KuaiRand-Pure and MovieLens 1M compare two RLCP implementations with four RL baselines. In each of the 19 configurations, at least one RLCP variant achieves the highest catalog diversity, reaching 1.11× to 5.21× that of the strongest baseline, with competitive session depth and no larger retained sets.

## 1 INTRODUCTION

Sequential recommendation aims to improve system utility over an entire user session, rather than opti mize each interaction in isolation in a long-term deployment. Since a recommendation can afect both the current response and the user’s subsequent interests, reinforcement learning (RL) provides a natural framework for this objective (Afsar et al., 2022; Chen et al., 2021). Within this framework, many RL recommenders present a slate with a fixed predetermined number of items at every request (Zhao et al., 2018;

Honghao Wei Washington State University

Ie et al., 2019; Zhao et al., 2023). However, a fixed cardinality does not reflect how many useful alternatives are available and cannot account for changes in their quality. For example, a small set may sufice when preferences are clear, whereas a broader set can be valuable when several alternatives remain plausible. This mismatch motivates a diferent design criterion: retain a compact set while controlling the risk of excluding all valuable actions.

Achieving this objective requires more than replacing the top M ranking rule with a threshold. First, the optimal long-term values used to define valuable actions are unknown. Although a learned critic can rank items, its score gaps are not themselves calibrated measures of retention. Second, filtering changes the available actions and, consequently, the states visited later in the session. As a result, a threshold calibrated under one policy need not satisfy a population retention requirement under another. Third, even when a valuable item is retained, the downstream actor or user choice mechanism may consistently select a poor alternative. Retention and selection must therefore be distinguished: keeping a valuable item does not by itself ensure a good decision. Together, these challenges call for an analysis that connects action set construction to sequential decision quality.

To address these challenges, we introduce Reinforcement Learning with Calibrated Pruning (RLCP). In RLCP, a critic scores candidate items, an online threshold determines which items are retained, and a downstream selector chooses from the retained set. To calibrate the threshold, we update it using a binary signal indicating whether the raw set contains an action in a proxy target. Following adaptive conformal inference (Gibbs and Candes, 2021; Gibbs and Cand\`es, 2024; Angelopoulos et al., 2024), this update gives a deterministic bound on the observed proxy miss frequency even as the critic, policy, and visited states change. The guarantee therefore concerns observable proxy retention, rather than retention defined by the unknown optimal value function.

![](images/9c18aa7d4352116f2b9820515ecf7e24e818f96071dda043fa6cbb2c3f795449.jpg)  
Figure 1: Performance across the 19 experimental configurations. The axes are normalized, and set size is inverted so that larger values are preferable on every axis. RLCP improves catalog diversity while maintaining competitive session depth under the same set size cap as the baselines. The full detailed results and configurations are in Appendix F.

To account for how pruning afects future decisions, we evaluate expected set size under the state distribution of the policy that uses the retained actions. We prove that this policy’s value loss equals the discounted accumulation of two terms: the loss caused by filtering and the loss incurred by selecting among the retained actions. Under a condition relating proxy scores to optimal action values, calibration bounds the filtering loss, while critic accuracy and the downstream selection rule bound the selection loss. Combining these bounds gives a finite session reward guarantee, includ ing the additional loss from imposing a maximum set size. The result allows the critic, policy, and threshold to change throughout the session and does not require their parameters to converge. Experiments in 19 simulator configurations show that RLCP can reach a broader item catalog while maintaining competitive session depth under the same set size cap as the baselines, as summarized in Figure 1. The main contribu tions are summarized below.

## Main contributions.

1. We develop RLCP to construct action sets through learned scores and online retention calibration. A bilevel formulation evaluates set size and retention under the policy induced by the retained actions, accounting for the efect of filtering on subsequent state visitation.

2. We establish a deterministic proxy miss bound and an exact value decomposition. Together, they give conditional bounds on discounted value and finite session reward that quantify the efects of proxy error, critic error, selection error, and set truncation.

3. We evaluate two RLCP implementations against four RL baselines on KuaiRand-Pure and Movie-Lens 1M. At least one variant achieves the highest catalog diversity in every configuration, with com petitive session depth and mean retained sets no larger than the corresponding baseline slates.

The conformal component builds on existing online calibration methods; our focus is how calibrated retention interacts with action filtering and sequential value. Appendix A places this contribution in the context of conformal recommendation and decisionmaking.

## 2 PROBLEM FORMULATION

We consider a finite discounted MDP ${ \mathcal { M } } = ( S , A _ { : }$ $P , r , \gamma , \mu _ { 0 } )$ , where S and A are finite nonempty state and action spaces, $P$ is the transition kernel, $r ( s , a ) \in$ [0, 1] is the expected reward, $\gamma ~ \in ~ [ 0 , 1 )$ is the discount factor, and $\mu _ { 0 }$ is the initial state distribution. At time $t ,$ the learner observes $s _ { t } ,$ selects $a _ { t } .$ , receives $R _ { t } \in [ 0 , 1 ]$ , and observes $s _ { t + 1 }$ . Let $\mathcal { H } _ { t }$ contain the trajectory and all algorithmic information available before $a _ { t }$ is sampled. The environment satisfies

$$
\begin{array} { r l } & { \operatorname* { P r } ( s _ { t + 1 } = s ^ { \prime } \mid \mathcal { H } _ { t } , a _ { t } = a ) = P ( s ^ { \prime } \mid s _ { t } , a ) , } \\ & { \quad \quad \quad \mathbb { E } [ R _ { t } \mid \mathcal { H } _ { t } , a _ { t } = a ] = r ( s _ { t } , a ) , } \end{array}\tag{1}
$$

The learner may use the full history to update its policy, while the reward and transition laws depend only on the current state and action. Let Π be the set of stationary Markov policies. For $\pi \in \Pi$ , define the state value and state-action value functions as

$$
V ^ { \pi } ( s ) = \mathbb { E } _ { s } ^ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } R _ { t } \right] , Q ^ { \pi } ( s , a ) = \mathbb { E } _ { s , a \sim \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } R _ { t } \right] .\tag{2}
$$

For a state distribution $\nu ,$ write $V ^ { \pi } ( \nu ) = \mathbb { E } _ { s \sim \nu } [ V ^ { \pi } ( s ) ]$ The optimal values are $V ^ { \ast } ( s ) \ =$ ma $\mathfrak { x } _ { \pi \in \Pi } V ^ { \pi } ( s )$ and $\begin{array} { r c l } { Q ^ { * } ( s , a ) } & { = } & { \operatorname* { m a x } _ { \pi \in \Pi } Q ^ { \pi } ( s , a ) } \end{array}$ , with $V ^ { \ast } ( s ) \ =$ $\operatorname* { m a x } _ { a } Q ^ { * } ( s , a )$ and $V ^ { * } ( \nu ) = \mathbb { E } _ { s \sim \nu } [ V ^ { * } ( s ) ]$ . We evaluate action sets under the normalized discounted state occupancy:

$$
d ^ { \pi } ( s ) = ( 1 - \gamma ) \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \operatorname* { P r } ^ { \pi } ( s _ { t } = s \mid s _ { 0 } \sim \mu _ { 0 } ) .\tag{3}
$$

For a value tolerance $\varepsilon \geq 0$ , and $\alpha \in ( 0 , 1 )$ is the target miss rate. Define the oracle target action set as

$$
\mathcal { A } _ { \varepsilon } ^ { * } ( s ) = \{ a \in \mathcal { A } : Q ^ { * } ( s , a ) \geq V ^ { * } ( s ) - \varepsilon \} .\tag{4}
$$

An action belongs to this set if its optimal continuation value is within ε of the best available value. This target specifies which actions should be retained, but it is unknown to the learner.

![](images/b98af5aea3d4a6af280799168c4785962ee45906e4232b00e7c2c14a54f4f1a0.jpg)  
Figure 2: RLCP framework. At state $s _ { k }$ , critic scores $g _ { \theta _ { k } } ( s _ { k } , a )$ and threshold $\tau _ { k }$ define the raw set $C _ { \theta _ { k } , \tau _ { k } } ( s _ { k } )$ A minimum score fallback ensures nonempty execution. The resulting set constrains the downstream actor in RLCP or supplies the simulator in RLCP Single. Proxy miss feedback updates the threshold more frequently than the critic is updated; this schedule is an implementation choice.

## 2.1 Action Filtering and Execution

Since the oracle target $\mathcal { A } _ { \varepsilon } ^ { \ast } ( s )$ depends on the unknown $Q ^ { * }$ , the learner cannot use it directly to filter actions. Instead, it uses a learned score $g _ { \theta } ( s , a ) \ \in \ \mathbb { R }$ , with smaller scores preferred. For a threshold $\tau \in \mathbb { R }$ , the raw set consists of all actions whose scores do not exceed the threshold:

$$
C _ { \theta , \tau } ( s ) = \{ a : g _ { \theta } ( s , a ) \leq \tau \} .\tag{5}
$$

When no action passes the threshold, the policy still needs an action to execute. We therefore fix an order on $\mathcal { A }$ and use the action with the smallest score as a fallback:

$$
a _ { \theta } ^ { \mathrm { \tiny { m i n } } } ( s ) = \mathrm { f i r s t ~ a c t i o n ~ i n ~ } \underset { a } { \arg \operatorname* { m i n } } g _ { \theta } ( s , a ) .\tag{6}
$$

The resulting execution set is always nonempty:

$$
\overline { { C } } _ { \theta , \tau } ( s ) = \left\{ \begin{array} { l l } { C _ { \theta , \tau } ( s ) , } & { C _ { \theta , \tau } ( s ) \neq \emptyset , } \\ { \{ a _ { \theta } ^ { \mathrm { m i n } } ( s ) \} , } & { C _ { \theta , \tau } ( s ) = \emptyset . } \end{array} \right.\tag{7}
$$

The execution set determines which actions the downstream policy may select. To express this restriction, for any set map D satisfying $\emptyset \neq D ( s ) \subseteq A$ at every state, let $\Pi ( D ) = \{ \pi \in \Pi : \operatorname { s u p p } \pi ( \cdot \mid s ) \subseteq D ( s ) , \ \forall s \in$ $S \}$ denote the stationary policies supported on D. For the construction above, we therefore consider a downstream policy $\pi \in \Pi ( \overline { { C } } _ { \theta , \tau } )$

In recommendation, $D ( s )$ represents an intermediate candidate set and need not coincide with the displayed slate. The item MDP applies when an item is selected before the current feedback is generated and, conditional on the current state and selected item, reward and transition no longer depend on the remaining collection. Appendix B.1 formalizes this requirement and explains when the collection itself must instead be modeled as the action.

With execution specified, we next evaluate whether the threshold rule retains a valuable action. This evaluation uses the raw set $C _ { \theta , \tau } ( s )$ : retention succeeds when that set contains at least one action in the oracle target $\boldsymbol { \mathcal { A } } _ { \varepsilon } ^ { \ast } ( s )$ . The fallback serves a diferent purpose, namely, ensuring that execution remains possible. Thus an empty raw set counts as a retention failure even if the fallback action belongs to the oracle target, because no target action passed the threshold. Whether retention succeeds depends on the state. We therefore evaluate it under the normalized discounted occupancy $d ^ { \pi }$ of the downstream policy π. For a target miss rate $\alpha \in ( 0 , 1 )$ , the desired retention condition is

$$
\operatorname* { P r } _ { s \sim d ^ { \pi } } ( C _ { \theta , \tau } ( s ) \cap \mathcal { A } _ { \varepsilon } ^ { * } ( s ) \neq \varnothing ) \ge 1 - \alpha .\tag{8}
$$

The condition allows the states where every oracle target action is removed to carry at most α of the discounted occupancy mass. The policy generating this occupancy is restricted by the execution set, so changing the threshold may change both the retained actions and the states on which retention is evaluated.

Let $\begin{array} { r } { m _ { \theta } ( s ) = \operatorname* { m i n } _ { a \in \mathcal { A } _ { \varepsilon } ^ { \ast } ( s ) } g _ { \theta } ( s , a ) } \end{array}$ be the smallest score among oracle target actions. Retention succeeds exactly when this score passes the threshold:

$$
C _ { \theta , \tau } ( s ) \cap { \mathcal { A } } _ { \varepsilon } ^ { * } ( s ) \neq \emptyset \quad \Longleftrightarrow \quad m _ { \theta } ( s ) \leq \tau .\tag{9}
$$

This gives a scalar condition for threshold selection. The oracle margin is unavailable because its target depends on $Q ^ { * }$ ; RLCP will instead calibrate a proxy margin obtained from observable feedback.

## 2.2 Policy-Dependent Action-Set Objective

A threshold determines which actions remain available throughout the session. The set size objective must therefore account for the policy that uses those actions. Fix a learned score $g _ { \theta }$ and write $D _ { \theta , \tau } = \overline { { C } } _ { \theta , \tau }$ . The best policy supported on this set has value

$$
V _ { \theta , \tau } ^ { * } ( \mu _ { 0 } ) = \operatorname* { m a x } _ { \pi \in \Pi ( D _ { \theta , \tau } ) } V ^ { \pi } ( \mu _ { 0 } ) .\tag{10}
$$

We denote the selected optimal response by $\pi _ { \theta , \tau } ^ { * }$ . A fixed lexicographic rule on optimal occupancies, together with a fixed action at states of zero occupancy, resolves ties; Appendix C gives the construction. We then minimize expected set size subject to retention under this response:

Problem 2.1 (Occupancy-dependent threshold selection). For the score $g _ { \theta }$ , consider

$$
\begin{array} { r l } { \underset { \tau \in \mathbb { R } } { \mathrm { m i n } } } & { \Phi _ { \theta } ( \tau ) : = \mathbb { E } _ { s \sim d ^ { \pi _ { \theta , \tau } ^ { * } } } [ | D _ { \theta , \tau } ( s ) | ] } \\ { s . t . } & { \underset { s \sim d ^ { \pi _ { \theta , \tau } ^ { * } } } { \mathrm { P r } } ( C _ { \theta , \tau } ( s ) \cap \mathcal { A } _ { \varepsilon } ^ { * } ( s ) \neq \varnothing ) \ge 1 - \alpha , } \\ & { \pi _ { \theta , \tau } ^ { * } \in \underset { \pi \in \Pi ( D _ { \theta , \tau } ) } { \mathrm { a r g m a x } } V ^ { \pi } ( \mu _ { 0 } ) . } \end{array}\tag{11}
$$

The inner problem determines how the retained actions are used, and the outer problem evaluates set size and retention under the resulting state occupancy. Consequently, a larger threshold may induce a diferent policy and change which states receive weight in the objective. The smallest feasible threshold need not minimize expected set size, unlike threshold selection under a fixed evaluation distribution.

The score is fixed only when defining this benchmark. Allowing arbitrary scores in the outer optimization would admit the uninformative oracle solution of assigning the smallest score to one optimal action at each state and returning singleton sets. RLCP instead learns scores by value estimation and updates the threshold from proxy feedback. We do not assume that these coupled updates solve (11); the following guarantees apply while the score and policy continue to change.

## 3 MAIN RESULTS

In this section, we formally present our main results. We first bound the proxy miss frequency under the online threshold update. We then derive a value identity that separates the loss from filtering from the loss of selecting an action within the retained set. Combining the two results gives a reward bound for a policy that adapts during a finite session. Appendix C establishes the population benchmark, and the detailed proofs are deferred to Appendix D.

## 3.1 Retention Calibration During Learning

When both the score and policy are fixed, the small est threshold satisfying the retention requirement is the $( 1 - \alpha )$ quantile of the oracle margin (Proposition C.3). During learning, however, updates to the score and policy can change the margin distribution, so a threshold calibrated at one stage need not remain valid later. To account for these changes, we adjust the threshold using observed misses rather than assuming a fixed margin distribution. A miss increases the threshold to make filtering less restrictive, whereas a hit decreases it. Specifically, we use the online calibration update (Gibbs and Candes, 2021; Angelopoulos et al., 2024)

$$
e _ { t } = { \bf 1 } \{ m _ { t } > \tau _ { t } \} , \qquad \tau _ { t + 1 } = \tau _ { t } + \eta _ { t } ( e _ { t } - \alpha ) .\tag{12}
$$

Here $m _ { t }$ is a finite margin, $e _ { t }$ is its miss indicator, and $\eta _ { t } > 0$ is a nonincreasing step size. A miss raises the threshold by $\eta _ { t } ( 1 - \alpha )$ , making filtering less restrictive; a hit lowers it by $\eta _ { t } \alpha$ . Theorem 3.1 bounds the resulting miss frequency for any realized margin sequence.

Theorem 3.1 (Pathwise miss-rate tracking). For any adaptive sequence of finite margins, define

$$
W _ { T } = \operatorname * { m a x } \{ \tau _ { 1 } , m _ { 1 } , \ldots , m _ { T } \} - \operatorname * { m i n } \{ \tau _ { 1 } , m _ { 1 } , \ldots , m _ { T } \} .
$$

Then, for every $T \geq 1$ ，

(13)

$$
\left| \frac { 1 } { T } \sum _ { t = 1 } ^ { T } e _ { t } - \alpha \right| \leq \frac { W _ { T } + \eta _ { 1 } } { \eta _ { T } T } .\tag{14}
$$

If sup $_ T \ : W _ { T } \leq \theta < \infty$ and $\eta _ { T } T ~ \to ~ \infty$ , then $\begin{array} { r } { T ^ { - 1 } \sum _ { t = 1 } ^ { T } e _ { t } \to \alpha . ~ I f \eta _ { t } \equiv \eta } \end{array}$ , the always-valid finitetime form is

$$
\left| \frac { 1 } { T } \sum _ { t = 1 } ^ { T } e _ { t } - \alpha \right| \leq \frac { W _ { T } + \eta } { \eta T } ,\tag{15}
$$

and it is at most $( W + \eta ) / ( \eta T )$ under the uniformwidth condition above.

With bounded margin width and a constant step size, Theorem 3.1 gives an $O ( 1 / T )$ deviation of the observed miss frequency from α. The argument is deterministic: it requires neither independent margins nor a stationary margin distribution. It controls the average along the realized sequence, rather than retention at every state or convergence to a fixed population quantile. To apply the update without observing $Q ^ { * }$ , RLCP uses a finite nonempty proxy target $\widetilde { \mathcal { A } } _ { \varepsilon , t } ( s _ { t } )$ . Its margin and miss indicator are

$$
\widetilde { m } _ { t } = \operatorname* { m i n } _ { a \in \widetilde { \mathcal { A } } _ { \varepsilon , t } ( s _ { t } ) } { g _ { \theta _ { t } } ( s _ { t } , a ) } , \qquad \widetilde { e } _ { t } = \mathbf { 1 } \{ \widetilde { m } _ { t } > \tau _ { t } \} .\tag{16}
$$

Setting $m _ { t } = \widetilde { m } _ { t }$ in Theorem 3.1 controls the observed proxy miss rate while the critic and actor change. The relation between this proxy target and actions with high optimal value determines what the guarantee implies for reward.

A cap on set size can introduce misses even when the raw set retains a proxy target action. Let $C _ { t } ^ { K }$ keep the K actions with the lowest scores in the raw set, or all actions if fewer than K are available. A capped miss is either a raw miss or a raw hit for which truncation removes every proxy target action. Calibration controls the former; the latter depends on the ranking and the cap. Appendix C.3 bounds the combined miss rate and characterizes feasibility under a fixed distribution.

## 3.2 From Retention to Long-Horizon Value

Retaining a valuable action is not enough if the policy does not select it. Consider a single state with two actions giving rewards one and zero. Retaining both actions gives perfect retention, but a policy that always chooses the second receives zero reward (Proposition C.4). To account for this distinction, for a nonempty set map D and policy $\pi \in \Pi ( D )$ , define

$$
\begin{array} { l } { { \displaystyle \Delta _ { D } ( s ) = V ^ { * } ( s ) - \operatorname* { m a x } _ { a \in D ( s ) } Q ^ { * } ( s , a ) , } } \\ { { \displaystyle \zeta _ { \pi , D } ( s ) = \operatorname* { m a x } _ { a \in D ( s ) } Q ^ { * } ( s , a ) - \sum _ { a \in D ( s ) } \pi ( a \mid s ) Q ^ { * } ( s , a ) . } } \end{array}\tag{17}
$$

The filtering loss $\Delta _ { D }$ is the gap between the best available action and the best retained action. The selection loss $\zeta _ { \pi , D }$ is the remaining gap between that retained action and the policy’s expected choice. The next iden tity shows how both losses accumulate under the states visited by the policy.

Theorem 3.2 (Filtering–selection value identity). For every set map D satisfying $D ( s ) \neq \emptyset$ for every state and every stationary $\pi \in \Pi ( D )$ ,

$$
V ^ { * } ( \mu _ { 0 } ) - V ^ { \pi } ( \mu _ { 0 } ) = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d ^ { \pi } } [ \Delta _ { D } ( s ) + \zeta _ { \pi , D } ( s ) ] .\tag{18}
$$

Theorem 3.2 identifies where retention enters the control problem: it bounds the filtering term, while the downstream policy determines the selection term. Both losses use the same unrestricted optimal continuation values $Q ^ { * }$ . A policy optimal under future action restrictions can therefore still have positive selection loss; this term is not just an error in optimizing the actor.

For proxy retention to control filtering loss, the proxy must identify actions with small optimal value gaps. Suppose the proxy target consists of near maximizers of a score ${ \cal \widetilde { Q } } ,$ and its pairwise gaps approximate those of $Q ^ { * }$ up to a scale $\lambda > 0$ and error $\delta _ { \mathrm { p r o x } }$ (Assumption C.7). Proposition C.8 gives

$$
\widetilde { \mathcal { A } } _ { \varepsilon } ( s ) \subseteq \mathcal { A } _ { \lambda \varepsilon + \delta _ { \mathrm { p r o x } } } ^ { * } ( s ) .
$$

Thus a proxy hit retains an action whose optimal value gap is at most $\lambda \varepsilon + \delta _ { \mathrm { p r o x } }$ For selection, a critic error bounded by $\delta _ { \mathrm { c r i t } }$ relative to $Q ^ { * }$ and a maximization error $\delta _ { \mathrm { s e l } } ( s )$ within the retained set give $\zeta _ { \pi , D } ( s ) \leq 2 \delta _ { \mathrm { c r i t } } + \delta _ { \mathrm { s e l } } ( s )$ . Substituting these bounds into the identity yields Corollary C.10. The approximation conditions are needed in addition to calibration: a critic trained to evaluate the current policy need not approximate the unrestricted $Q ^ { * }$

## 3.3 Adaptive Finite Session Guarantee

The preceding identity evaluates a stationary policy. During learning, however, the policy may change after every interaction. We handle these updates by applying the same loss decomposition at each stage and using the calibration bound along the resulting tra jectory. Consider an undiscounted item MDP with H stages, rewards in [0, 1], and optimal values $V _ { h } ^ { * } , Q _ { h } ^ { * }$ Define

$$
\begin{array} { r l r } & { } & { \Delta _ { H } ^ { \mathrm { m a x } } = \underset { h , s , a } { \mathrm { m a x } } [ V _ { h } ^ { * } ( s ) - Q _ { h } ^ { * } ( s , a ) ] , } \\ & { } & { \varepsilon _ { H } = \mathrm { m i n } \{ \lambda \varepsilon + \delta _ { \mathrm { p r o x } } , \Delta _ { H } ^ { \mathrm { m a x } } \} . } \end{array}
$$

Here $\Delta _ { H } ^ { \mathrm { m a x } }$ bounds the loss of any action, while $\varepsilon _ { H }$ bounds the loss of a retained proxy target under the stagewise proxy fidelity condition. Let $\overline { { \delta } } _ { \mathrm { s e l } , H }$ be the average expected selection error measured by the critic within the retained set, and let $\overline { { h } } _ { H } ^ { K }$ be the average probability that a raw proxy hit is lost under a cap of K actions. Appendix C.6 states the full information and approximation conditions.

Theorem 3.3 (Finite-session proxy-to-value guarantee). Under Assumption C.12, let $\eta _ { 1 } \geq \cdot \cdot \cdot \geq \eta _ { H } > 0$ and suppose the realized proxy-margin width is at most W almost surely. For $\begin{array} { r } { V _ { 1 } ^ { \mathrm { \hat { R L C P } } } ( \mu _ { 0 } ) = \mathbb E [ \sum _ { h = 1 } ^ { H } R _ { h } ] } \end{array}$ ，

$$
\begin{array} { r l } & { V _ { 1 } ^ { * } ( \mu _ { 0 } ) - V _ { 1 } ^ { \mathrm { R L C P } } ( \mu _ { 0 } ) } \\ & { \quad \le H \Biggl [ ( 1 - \alpha ) \varepsilon _ { H } + \alpha \Delta _ { H } ^ { \mathrm { m a x } } + 2 \delta _ { \mathrm { c r i t } } + \overline { { \delta } } _ { \mathrm { s e l } , H } } \\ & { \qquad + \left( \Delta _ { H } ^ { \mathrm { m a x } } - \varepsilon _ { H } \right) \left( \frac { W + \eta _ { 1 } } { \eta _ { H } H } + \overline { { h } } _ { H } ^ { K } \right) \Biggr ] . } \end{array}\tag{19}
$$

Without a size cap, take $K = | A |$ and $\overline { { h } } _ { H } ^ { K } = 0$

The first two terms bound filtering loss on proxy hits and misses at the target rate α. The critic and selection errors account for the action chosen from the execution set. The final term adds the deviation from the target miss rate and the hits removed by the cap. All of these quantities are evaluated along the adaptive trajectory, so the bound does not require convergence of the critic, actor, or threshold. Its size still depends on the stated approximation errors.

Algorithm 1 RLCP with Online Action-Set Calibra  
tion   
Input: target miss rate $\alpha ;$ interaction budget $T ;$   
threshold τ ; nonincreasing η $_ t > 0 ;$ cap $K \geq 1 ;$ critic   
$\widehat { Q } _ { \theta _ { 1 } } \colon$ selector $\pi _ { \phi _ { 1 } } ;$ proxy-target verifier.   
1 for $t = 1 , \dots , \dot { T }$ do   
2 Observe s<sub>t</sub> and compute $g _ { \theta _ { t } } ( s _ { t } , \cdot )$ using (20).   
3 Form $C _ { t } = \{ a : g _ { \theta _ { t } } ( s _ { t } , a ) \leq \bar { \tau _ { t } } \} .$   
4 Keep the min $\{ K , | C _ { t } | \}$ lowest-score actions to ob  
tain $\hat { C _ { t } ^ { K } }$   
5 Set $\mathbf { \bar { \boldsymbol { D } } } _ { t } = \boldsymbol { C } _ { t } ^ { K }$ if nonempty; otherwise set $D _ { t } \ =$   
$\{ a _ { \theta _ { t } } ^ { \operatorname* { m i n } } ( s _ { t } ) \}$   
6 Define the proxy target for the current state and   
obtain the raw miss $\widetilde { e } _ { t } = \mathbf { 1 } \{ C _ { t } \cap \widetilde { A } _ { \varepsilon , t } ( s _ { t } ) = \varnothing \}$   
7 Sample $a _ { t } \sim \pi _ { \phi _ { t } } ( \cdot \mid s _ { t } , \tilde { D _ { t } } ) ;$ observe $R _ { t } , s _ { t + 1 }$   
8 Update $\tau _ { t + 1 } = \tau _ { t } + \eta _ { t } ( \widetilde { e } _ { t } - \alpha ) .$   
9 Update the critic and selector from interaction   
data.   
10 end for

The reward determines what this bound measures. When reward indicates a nonterminal state and departure follows the item dynamics, it bounds truncated session depth. A bound for click or like reward does not imply the same depth guarantee, and a departure rule that depends on the entire collection requires a collection action model. Appendix B.1 discusses this distinction for recommendation.

## 4 RLCP ALGORITHM

RLCP constructs the retained set from a critic $\widehat { Q } _ { \theta _ { t } }$ and an online threshold, as shown in Figure 2. At time $t ,$ the score of an action is its estimated value gap from the best candidate:

$$
g _ { \theta _ { t } } ( s , a ) = \operatorname* { m a x } _ { b } \widehat { Q } _ { \theta _ { t } } ( s , b ) - \widehat { Q } _ { \theta _ { t } } ( s , a ) .\tag{20}
$$

Adding the same constant to all critic values at a state leaves these scores unchanged. Thresholding retains actions whose estimated gaps do not exceed $\tau _ { t }$ If the raw set exceeds the cap, we keep the actions with the smallest gaps; if it is empty, the fallback supplies a critic maximizer. The downstream selector then chooses from this execution set. Algorithm 1 gives the complete procedure.

Calibration feedback. The threshold update uses a binary signal indicating whether the raw set intersects the proxy target. This signal evaluates retention before the cap and fallback; neither operation changes the miss used for calibration. A verifier supplies the signal without revealing $Q ^ { * }$ , and our simulator implements the verifier through its response model. For Theorem 3.3, the target must be defined before the current action is sampled, although the label may be revealed later. A target defined from the current realized responses can still be calibrated pathwise, but it does not satisfy this information condition. Feedback on the executed item alone generally cannot identify the required set event, so deployment requires a suitable verifier or counterfactual response model.

Learning and selection. Interaction data update the critic and the selector, while raw proxy misses update the threshold. In our implementation, calibration is updated more frequently than the critic. The pathwise guarantee uses the unprojected update in (12); replacing raw misses with capped misses is a diferent procedure.

The selector also determines whether pruning changes the executed action. The critic gap score, cap, and fallback all preserve a global critic maximizer. An exactly greedy selector using the same critic and tie rule therefore chooses the same action with or without pruning (Proposition C.13). RLCP instead regulates the alternatives available to a stochastic actor, slate constructor, or user choice mechanism. The experiments compare two such implementations.

## 5 EXPERIMENTS

We evaluate RLCP on the whole-session sequential recommendation task provided by the KuaiSim simulator (Zhao et al., 2023). This protocol emulates online policy interaction through full-session rollouts, in which KuaiSim generates user feedback, computes the corresponding rewards, and determines when the user leaves the session. Our experiments are designed to answer the following questions:

• Q1: Can RLCP improve long-term recommendation performance under simulator-based sequential interaction?

• Q2: Can RLCP produce reliable admissible action sets that contain high-quality recommendations while reducing the efective action space?

## 5.1 Datasets and Simulator

We evaluate RLCP in the KuaiSim whole-session setting on KuaiRand-Pure (Gao et al., 2022), the dataset used to construct the original simulator, and on the MovieLens 1M migration demonstrated by KuaiSim (Harper and Konstan, 2015; Zhao et al., 2023). KuaiRand-Pure comes from Kuaishou logs in which uniformly sampled videos were inserted into ordinary feeds. KuaiSim contains six positive behaviors—click, long view, like, comment, follow, and forward—together with dislike and session departure. MovieLens 1M contains about 1M timestamped ratings. Following the KuaiSim migration, ratings above

3 are mapped to positive feedback (like), ratings at most 3 to negative feedback (hate), interactions are ordered chronologically, and day-level groups define sessions. Table 1 reports statistics for both datasets.

Table 1: Statistics of the datasets used.
<table><tr><td>Dataset</td><td></td><td>Users Items</td><td>Interactions</td><td>Sessions</td><td>Density</td></tr><tr><td>KuaiRand -Pure</td><td>27,077 7,551</td><td></td><td>1,436,609</td><td>246,738</td><td>0.70%</td></tr><tr><td>ML-1M</td><td>6,4003,706</td><td></td><td>1,000,208</td><td>16,629</td><td>4.22%</td></tr></table>

We modify KuaiSim’s leave module to make session length depend on the responses elicited by a recommendation. At each request, the recommender observes a user profile and recent interaction history, presents an item collection, and receives feedback from the immediate response model. We use is click and is like as reward signals. In our simulator inspection, click responses varied little across users, while like responses were more sensitive to user preferences.

At step t, the leave module evaluates the admissible set $C _ { \theta , \tau } ( s _ { t } )$ for RLCP and the displayed slate for a baseline with fixed slate size. Patience decreases by p if the collection receives at least one positive response and by $q > p$ otherwise. The session ends when remaining patience falls below 1. We test $( p , q ) = ( 1 , 2 )$ and (0.4, 2), so a failure to elicit a positive response incurs the larger decrement in both settings.

## 5.2 Baselines and Metrics

We compare RLCP with four whole-session sequential recommendation baselines provided by KuaiSim (Zhao et al., 2023): A2C (Mnih et al., 2016), DDPG (Lillicrap et al., 2016), TD3 (Fujimoto et al., 2018), and HAC (Liu et al., 2023); detailed descriptions are provided in Appendix E.1.

We report session depth, average and total reward, catalog diversity, intra-list diversity (ILD), and average admissible-set size, following KuaiSim (Zhao et al. 2023). Depth is the primary session-level utility metric; catalog diversity measures the number of distinct items exposed across active user requests, ILD measures dissimilarity within a recommendation set, and set size measures filtering eficiency. Formal definitions are provided in Appendix E.2.

## 5.3 Method Implementation

We instantiate RLCP with two variants that share the conformal set construction, online calibration objective, and coverage target. The realized state $s _ { t }$ concatenates the user profile with the full within-session history of rollout actions $( a _ { 0 } , a _ { 1 } , \dotsc , a _ { t - 1 } )$ , so the critic and rollout policy condition on all previously executed actions. The variants difer only in how the admissible set becomes a rollout action. Full RLCP follows Algorithm 1: $C _ { \theta , \tau } ( s )$ constrains a downstream slate actor trained through sequential rollouts, while $\tau$ is updated online to maintain the target miss rate for near-optimal items. Critic/score updates occur on the slow timescale, and slate-actor rollouts and threshold calibration on the fast timescale.

RLCP-Single instead feeds the admissible set directly into the environment and uses the simulator’s first rewarded item as the rollout action, imitating real-world user behavior. This simplifies rollout while preserving the online calibration objective. Both variants target 1 − α coverage of ε-optimal items, apply a minimumset-size rescue when required (usually 1), and use the same conformal mask logic to bound set size.

Environment-Side Proxy. To operationalize the ε- optimal action set in (4), we use the Bernoulli success probability $\hat { p } _ { b } ( s _ { t } , a )$ from the simulator’s immediateresponse model and set $\widetilde Q ( s _ { t } , a ) = \hat { p } _ { b } ( s _ { t } , a )$ , which induces

$$
\widetilde { \mathcal { A } } _ { \varepsilon } ( s _ { t } ) = \left\{ a : \widetilde { Q } ( s _ { t } , a ) \geq \operatorname* { m a x } _ { j } \widetilde { Q } ( s _ { t } , j ) - \varepsilon \right\}
$$

as in (33). This conditional one-step expected reward is biased for the long-horizon $Q ^ { * }$ in (4). The numerical value of $\widetilde { Q }$ is never exposed to the actor or critic; the simulator supplies only binary feedback indicating whether $C _ { \theta _ { t } , \tau _ { t } } ( s _ { t } ) \cap \widetilde { \mathcal { A } } _ { \varepsilon } ( s _ { t } )$ is nonempty. This emulates positive user feedback when a recommendation set contains a satisfactory item.

Results. Across all 19 configurations, at least one RLCP variant achieves the highest catalog diversity without requiring a larger retained set. Within each configuration, the RLCP cap equals the baseline slate size M, and all methods share the simulator, reward signal, and patience setting. Appendix Tables 2 through 9 provide the full numerical results. These comparisons show broader catalog exposure with competitive session depth under matched set size budgets.

As shown in Figure 3, RLCP-Single provides the strongest catalog exposure on KuaiRand-Pure. It achieves the highest catalog diversity in all ten configurations, reaching 1.48× to 4.56× that of the strongest baseline. Its mean retained set is smaller than M in nine configurations, with a maximum reduction of 25.5%, and equals M in the remaining configuration. It also attains or ties the highest session depth in six configurations and stays within 0.4 interactions of the best method in the others. Full RLCP achieves the highest reported ILD of 1.00 throughout, but RLCP-Single generally provides a better balance between catalog exposure and session utility.

As shown in Figure 4, the exposure advantage persists

![](images/4309707cfb2a8814ab60cb8c5b60f803a8036791a05035dcab53bd93aa593c00.jpg)  
(a) Is click, p, q = (1, 2).

![](images/e8299236f8c55bdf19fa04c1925335e4bd436ca436dc3830c2d0ed9ff5764b93.jpg)  
(b) Is click, p, q = (0.4, 2).

![](images/b078ef9c71c5cd50c5a646a5908b378f1b68d199362033f552c78418c012b02d.jpg)  
(c) Is like, p, q = (1, 2).

![](images/ec2f9ec4cf3af9f594aa8ed3d54fb920089a44272b2675069b6a547f18a5634c.jpg)  
(d) Is like, p, q = (0.4, 2).

Figure 3: KuaiRand-Pure radar summaries for the four reward and patience configurations reported in Tables 2– 5. Each panel averages over the baseline slate sizes in the corresponding table and shows depth, catalog diversity, intra-list diversity, and inverted set size, with larger radius indicating better performance. RLCP-Single gives the strongest catalog-diversity gains on KuaiRand-Pure while using equal or smaller admissible sets, whereas the full RLCP consistently attains the highest intra-list diversity.

![](images/4421784580be624e65bc9ce371266ee8b59c88addd4a277d3ec97820c52fc573.jpg)  
(a) Is click, p, q = (1, 2).

![](images/066fe7f02128054606de0a64efab97083de51875ffce956040e4f35cf8456602.jpg)  
(b) Is click, p, q = (0.4, 2).

![](images/0a5580fff3601eaff5a0d5c622951ae8cf0e697b53f0c79d3c471fe98fc3b15b.jpg)  
(c) Is like, p, q = (1, 2).

![](images/5f92e1eb43242e76f41dc15d836b99c4cadbd47508ff001a89256c65cac9dec8.jpg)  
(d) Is like, p, q = (0.4, 2).  
Figure 4: ML-1M radar summaries for the four reward and patience configurations reported in Tables 6–9. Each panel averages over the baseline slate sizes in the corresponding table and uses the same normalized axes as Figure 3. On the denser ML-1M data, the full RLCP is more competitive in terms of depth and catalog diversity while maintaining maximal intra-list diversity, and the RLCP variants retain their overall diversity advantage over the fixed-slate baselines.

on ML-1M, where full RLCP becomes more competitive. It leads catalog diversity in five configurations, exceeds the strongest baseline in eight, and attains or ties the highest session depth in seven. It also maintains ILD 1.00 throughout and reduces mean set size below M in four configurations. RLCP-Single leads catalog diversity in the remaining four, giving the better variant in each configuration 1.11× to 5.21× the catalog diversity of the strongest baseline. Under the longer patience setting, this range is 3.17× to 5.21×, although several methods reach the depth ceiling of 48. Thus the main benefit is broader exposure with competitive depth, rather than uniformly higher reward or a single variant that dominates every setting.

tion. RLCP decouples critic-induced item scores from the calibrated threshold, producing state-dependent, variable-size sets rather than fixed-cardinality top-M slates. The threshold provides pathwise control of the realized average miss rate, while two-timescale learning jointly updates the score model, calibration threshold, and rollout policy. Across KuaiRand-Pure and ML-1M, RLCP remains competitive with fixed-slate baselines in session depth while using sets of equal or smaller size and improving diversity. An RLCP variant achieves the highest catalog diversity in every configu ration, and full RLCP consistently achieves the highest within-slate diversity. These results support conformal policy learning for retaining near-optimal recommendations, controlling set size, and broadening exposure without fixing slate cardinality in advance.

## 6 CONCLUSION

We introduced RLCP, a conformal policy-learning framework for whole-session sequential recommenda-

## References

Afsar, M. M., Crump, T., and Far, B. (2022). Reinforcement learning based recommender systems: A survey. ACM Computing Surveys, 55(7):1–38.

Angelopoulos, A. N., Barber, R., and Bates, S. (2024). Online conformal prediction with decaying step sizes. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 1616– 1630. PMLR.

Angelopoulos, A. N. and Bates, S. (2023). Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning, 16(4):494–591.

Angelopoulos, A. N., Krauth, K., Bates, S., Wang, Y., and Jordan, M. I. (2023). Recommendation systems with distribution-free reliability guarantees. In Proceedings of the Twelfth Symposium on Conformal and Probabilistic Prediction with Applications, volume 204 of Proceedings of Machine Learning Research, pages 175–193. PMLR.

Bastani, O., Gupta, V., Jung, C., Noarov, G., Ramalingam, R., and Roth, A. (2022). Practical adversarial multivalid conformal prediction. In Advances in Neural Information Processing Systems, volume 35, pages 29362–29373.

Chen, X., Yao, L., McAuley, J., Zhou, G., and Wang, X. (2021). A survey of deep reinforcement learning in recommender systems: A systematic review and future directions. arXiv preprint arXiv:2109.03540.

De Toni, G., Purificato, E., Gomez, E., Passerini, A., Lepri, B., and Consonni, C. (2025). You don’t bring me flowers: Mitigating unwanted exposure in recommender systems. In Proceedings of the Nineteenth ACM Conference on Recommender Systems, pages 492–502. Association for Computing Machinery.

Dixit, A., Lindemann, L., Wei, S. X., Cleaveland, M., Pappas, G. J., and Burdick, J. W. (2023). Adaptive conformal prediction for motion planning among dynamic agents. In Proceedings of the 5th Annual Learning for Dynamics and Control Conference, volume 211 of Proceedings of Machine Learning Research, pages 300–314. PMLR.

Fujimoto, S., van Hoof, H., and Meger, D. (2018). Addressing function approximation error in actorcritic methods. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 1587–1596. PMLR.

Gao, C., Li, S., Zhang, Y., Chen, J., Li, B., Lei, W., Jiang, P., and He, X. (2022). KuaiRand: An unbiased sequential recommendation dataset with randomly exposed videos.

Gibbs, I. and Candes, E. (2021). Adaptive conformal inference under distribution shift. In Advances in Neural Information Processing Systems, volume 34, pages 1660–1672.

Gibbs, I. and Cand\`es, E. J. (2024). Conformal inference for online prediction with arbitrary distribution shifts. Journal of Machine Learning Research, 25(162):1–36.

Harper, F. M. and Konstan, J. A. (2015). The movielens datasets: History and context. ACM Transactions on Interactive Intelligent Systems, 5(4):19:1– 19:19.

Ie, E., Jain, V., Wang, J., Narvekar, S., Agarwal, R., Wu, R., Cheng, H.-T., Chandra, T., and Boutilier, C. (2019). SlateQ: A tractable decomposition for reinforcement learning with recommendation sets. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, pages 2592–2599. International Joint Conferences on Artificial Intelligence Organization.

Kagita, V. R., Pujari, A. K., Padmanabhan, V., and Kumar, V. (2022). Inductive conformal recommender system. Knowledge-Based Systems, 250:109108.

Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., and Wierstra, D. (2016). Continuous control with deep reinforcement learning. In International Conference on Learning Representations.

Liu, S., Cai, Q., Sun, B., Wang, Y., Jiang, J., Zheng, D., Jiang, P., Gai, K., Zhao, X., and Zhang, Y. (2023). Exploration and regularization of the latent action space in recommendation. In Proceedings of the ACM Web Conference 2023, pages 833–844. Association for Computing Machinery.

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., and Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1928–1937. PMLR.

Si, W., Jang, S., Lee, I., and Bastani, O. (2025). Conformal constrained policy optimization for cost-efective LLM agents. arXiv preprint arXiv:2511.11828.

Vovk, V., Gammerman, A., and Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.

Wang, C., Wang, F., Guo, R., Liang, Y., and Yu, P. S. (2024). Confidence-aware fine-tuning of sequential recommendation systems via conformal prediction.

Wang, X., Liu, S., Cai, Q., Li, X., Hu, L., Li, H., and Xie, G. (2025). Value function decomposition in markov recommendation process. In Proceedings of the ACM Web Conference 2025, pages 379–390. Association for Computing Machinery.

Zhang, Y., Shi, C., and Luo, S. (2023). Conformal of-policy prediction. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 2751–2768. PMLR.

Zhao, K., Liu, S., Cai, Q., Zhao, X., Liu, Z., Zheng, D., Jiang, P., and Gai, K. (2023). KuaiSim: A comprehensive simulator for recommender systems. In Advances in Neural Information Processing Systems, volume 36, pages 44880–44897.

Zhao, X., Xia, L., Zhang, L., Ding, Z., Yin, D., and Tang, J. (2018). Deep reinforcement learning for page-wise recommendations. In Proceedings of the 12th ACM conference on recommender systems, pages 95–103.

# Supplementary Materials

## A RELATED WORK

RLCP draws on sequential control, conformal recommendation, and online calibration. The distinction across these areas is what is calibrated and how that quantity is connected to the subsequent decision. We organize the comparison around this distinction rather than treating the combination of RL and calibration as new.

RL for Sequential Recommendation. RL recommender systems optimize utility across a sequence of interactions rather than only immediate feedback (Afsar et al., 2022; Chen et al., 2021; Zhao et al., 2018). Within this framework, SlateQ makes slate optimization tractable by decomposing slate values under a user choice model (Ie et al., 2019), while HAC learns continuous slate representations and a mapping from latent actions to item lists (Liu et al., 2023). KuaiSim supplies whole session evaluation protocols and actor and critic baselines (Zhao et al., 2023), and work on value estimation further distinguishes policy randomness from user response randomness (Wang et al., 2025). These methods address how to evaluate or select recommendations. RLCP addresses a complementary question: which candidate actions should remain available to that selection rule, and how does their retention afect sequential value? Its item model does not cover every reward or transition rule that depends on a slate.

Conformal Recommendation. Conformal methods provide a way to construct sets around a base prediction model while controlling a statistical error criterion (Vovk et al., 2005; Angelopoulos and Bates, 2023). In recommendation, inductive conformal methods attach conformal p-values to candidate items (Kagita et al., 2022), and distribution free recommendation sets calibrate ranking models against the false discovery rate (FDR) (Angelopoulos et al., 2023). Without explicitly optimizing long-term returns, CPFT incorporates conformal objectives into the fine tuning of sequential recommenders (Wang et al., 2024), while conformal risk control has been used to limit unwanted recommendations (De Toni et al., 2025). These approaches motivate adapting set size to a calibrated target. However, a recommendation target need not identify actions with high continuation value under an interacting policy. RLCP therefore distinguishes proxy retention from oracle retention and states the approximation condition needed to connect them.

Online Calibration and Decision Making. Because RLCP changes the policy and the visited state distribution during learning, its calibration argument uses methods for changing data sequences. Adaptive conformal inference controls long-run empirical miscoverage under distribution changes (Gibbs and Candes, 2021; Gibbs and Cand\`es, 2024). Multivalid methods provide richer guarantees across groups (Bastani et al., 2022), while methods with decaying step sizes combine guarantees for arbitrary sequences with population quantile consistency under stable sampling (Angelopoulos et al., 2024). RLCP uses this established scalar update principle. Theorem 3.1 makes its dependence on the realized action margin range explicit and supplies the calibration term used in the value analysis. Online miscoverage tracking itself is not presented as a new principle.

Conformal methods also support motion planning (Dixit et al., 2023), return prediction for a policy diferent from the data collection policy (Zhang et al., 2023), and cost aware orchestration of LLM agents (Si et al., 2025). In particular, CCPO jointly trains a policy and an adaptive conformal threshold, so RLCP is not distinguished by that broad combination. The distinction lies in the action filtering formulation and the analysis connecting retention to value. A proxy hit must first be related to an oracle hit, and the retained action must then be used by the selector. The value decomposition and finite session bound describe these two steps without assuming convergence of the coupled updates.

## B ADDITIONAL MODELING DETAILS

The main formulation treats an executed item as the MDP action. We now explain how this abstraction relates to a recommendation pipeline, where retrieval and collection based response rules introduce additional conditions.

## B.1 Sequential Recommendation Interpretation

The first requirement concerns the state: it must contain the information needed to predict both immediate feedback and the next user state. A representative state is

$$
s _ { t } = ( x _ { u } , h _ { t } ^ { a } , h _ { t } ^ { y } , \ell _ { t } , c _ { t } ) ,
$$

Here $x _ { u }$ is a user profile, $h _ { t } ^ { a }$ and $h _ { t } ^ { y }$ are action and response histories, $\ell _ { t }$ is remaining patience, and $c _ { t }$ is request context. An absorbing state represents user departure. The finite state model is an analytical idealization; in a large system, a learned representation must approximate the same predictive information.

The second requirement concerns retrieval. The finite action space can represent an eligible item pool, but pruning within that pool cannot recover an action already removed by retrieval. To separate these two losses, fix a state distribution $\nu ,$ let $\mathcal { A } _ { \mathrm { r e t } } ( s ) \subseteq \mathcal { A }$ be a finite nonempty retrieved pool, and let $C ( s ) \subseteq A _ { \mathrm { r e t } } ( s )$ be the raw set retained from it. Fix $\varepsilon _ { \mathrm { r e t } } , \varepsilon _ { \mathrm { f i l t } } \ge 0$ and $\rho \in [ 0 , 1 )$ , and define

$$
\begin{array} { r l } & { G _ { \mathrm { r e t } } ( s ) = \left\{ \underset { a \in \mathcal { A } _ { \mathrm { r e t } } ( s ) } { \operatorname* { m a x } } Q ^ { * } ( s , a ) \ge V ^ { * } ( s ) - \varepsilon _ { \mathrm { r e t } } \right\} , } \\ & { G _ { \mathrm { f l t } } ( s ) = \bigl \{ \exists a \in C ( s ) \mathrm { ~ s u c h ~ t h a t ~ } } \\ & { \qquad Q ^ { * } ( s , a ) \ge \underset { b \in \mathcal { A } _ { \mathrm { r e t } } ( s ) } { \operatorname* { m a x } } Q ^ { * } ( s , b ) - \varepsilon _ { \mathrm { f l t } } \bigr \} . } \end{array}
$$

If $\mathrm { P r } _ { s \sim \nu } ( G _ { \mathrm { r e t } } ^ { c } ) \leq \rho$ and $\operatorname* { P r } _ { s \sim \nu } ( G _ { \mathrm { f i l t } } ^ { c } \mid G _ { \mathrm { r e t } } ) \leq \alpha$ , then on $G _ { \mathrm { r e t } } \cap G _ { \mathrm { f i l t } }$ the retained action has global optimality gap at most $\varepsilon _ { \mathrm { r e t } } + \varepsilon _ { \mathrm { f i l t } }$ . The law of total probability gives

$$
\operatorname* { P r } _ { s \sim \nu } \bigl ( C ( s ) \cap \mathcal { A } _ { \varepsilon _ { \mathrm { r e t } } + \varepsilon _ { \mathrm { f i l t } } } ^ { * } ( s ) = \emptyset \bigr ) \le \rho + ( 1 - \rho ) \alpha .
$$

Thus the value tolerances add, while the miss probability accounts first for retrieval failure and then for filtering failure conditional on retrieval success. To retain the global tolerance ε, the two tolerances must sum to at most ε. Alternatively, filtering can be calibrated directly against $\mathcal { A } _ { \varepsilon } ^ { \ast } ( s ) \cap \mathcal { A } _ { \mathrm { r e t } } ( s )$ conditional on that intersection being nonempty.

The final requirement concerns how a retained collection becomes an executed item. Let $\kappa ( { a } \mid s , D )$ be a stationary Markov choice kernel supported on $D ,$ sampled before the current reward and next state are generated. This kernel may combine a downstream actor, slate construction, and user choice. For a stationary set map $D ,$ the induced item policy is

$$
\pi _ { D , \kappa } ( a \mid s ) = \kappa ( a \mid s , D ( s ) ) .\tag{21}
$$

For this policy to reproduce the recommendation process, the selected item must account for the dependence of the current reward and next state on the collection. The following assumption states that requirement conditiona on all information available before selection.

Assumption B.1 (Pre-response item-mediated dynamics). There is an item-level reward and transition kernel $K ( \mathrm { d } \boldsymbol { r } , \boldsymbol { s } ^ { \prime } \mid \boldsymbol { s } , \boldsymbol { a } )$ such that, under the pipeline law Pr<sub>D</sub> induced by every stationary set map D with $D ( s ) \neq \emptyset$ for all s, the following factorization holds conditional on the full pre-action history $\mathcal { H } _ { t } .$

$$
\begin{array} { r l } & { \operatorname* { P r } _ { D } \big ( a _ { t } = a , R _ { t } \in \mathrm { d } r , s _ { t + 1 } = s ^ { \prime } \mid \mathcal { H } _ { t } \big ) } \\ & { \quad \quad = \kappa ( a \mid s _ { t } , D ( s _ { t } ) ) K ( \mathrm { d } r , s ^ { \prime } \mid s _ { t } , a ) . } \end{array}\tag{22}
$$

The kernel agrees with the item-level MDP primitives:

$$
\begin{array} { c } { { \displaystyle { \int K ( \mathrm { d } r , s ^ { \prime } \mid s , a ) = P ( s ^ { \prime } \mid s , a ) , } } } \\ { { \displaystyle { \sum _ { s ^ { \prime } } \int r K ( \mathrm { d } r , s ^ { \prime } \mid s , a ) = r ( s , a ) . } } } \end{array}\tag{23}
$$

Thus, conditional on the state and executed item, the reward and next-state distribution do not depend further on the admissible set or displayed slate.

Proposition B.2 (Recommendation pipeline reduction). Under Assumption B.1, $\pi _ { D , \kappa } \in \Pi ( D )$ , and the recommendation pipeline has the same reward and state-transition law as the item-level $M D P$ controlled $b y \ \pi _ { D , \kappa }$ Hence all restricted-control and value results below apply to the induced policy.

Proof. For every state, $\kappa ( \cdot \mid s , D ( s ) )$ is a probability distribution supported on $D ( s )$ , so (21) defines a stationary policy in $\Pi ( D )$ . Conditional on the full pre-action history, factorization (22) gives the same one-step joint law of the action, reward, and next state as this item-level policy. Starting from the same initial distribution and applying this equality inductively establishes equality of the complete trajectory laws. □

The order in Assumption B.1 matters. For example, a rule that observes the current response vector and then selects the first rewarded item does not satisfy this order: conditioning on the selected item changes its response law because selection used that response. The process can instead be modeled with the retained set or ordered slate as the action and an environment kernel that jointly generates responses, the selected item, reward, and the next state. However, the item value results do not automatically transfer to this diferent MDP. The pathwise calibration result still concerns the realized proxy misses, and Proposition C.11 separately covers a leave rule based on any positive response in the collection.

In summary, the item abstraction applies when an item selected before feedback determines the next interaction. When position, competition, or exposure to unchosen items also changes reward or transition, the collection must instead be the primitive action. In either model, a value bound concerns the specified reward: a click or like reward does not by itself establish a depth bound without an appropriate relation to departure.

## C ADDITIONAL SUPPORTING RESULTS

The supporting results follow the same order as the main argument. We first establish the population benchmark and its characterization under a fixed policy. We then account for a cap on set size and state the conditions that transfer proxy retention and approximate selection to value.

## C.1 Restricted Control and Existence

Before optimizing a threshold, we must ensure that every nonempty execution set supports a well defined control problem. For a set map D satisfying $D ( s ) \neq \emptyset$ for every state, define

$$
( \mathcal { T } _ { D } v ) ( s ) = \operatorname* { m a x } _ { a \in D ( s ) } \left[ r ( s , a ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) v ( s ^ { \prime } ) \right] .\tag{24}
$$

This is the Bellman operator with maximization restricted to retained actions. The next result establishes its optimal value and an occupancy representation, which is also needed to select a unique response for the outer objective.

Theorem C.1 (Restricted control is well defined). For every set map D satisfying $D ( s ) \neq \emptyset$ for all $s , \ \mathcal { T } _ { D }$ is a γ contraction and has a unique fixed point $V _ { D } ^ { * }$ . A stationary deterministic policy in $\Pi ( D )$ attains this value. The lower problem also admits a normalized discounted occupancy-measure $L P$ whose optimum is $( 1 - \gamma ) V _ { D } ^ { * } ( \mu _ { 0 } )$ , with attainment, zero duality gap, and a deterministic lexicographic rule that selects one optimal occupancy.

Once that response is fixed, the outer problem reduces to comparing the finitely many set maps induced by the score thresholds.

Proposition C.2 (Fixed-score outer existence). For every fixed θ, Problem (11) is feasible and attains its minimum over the thresholds induced by that score.

A threshold retaining every action is feasible, so the outer comparison is nonempty. These results establish the benchmark for a given score. They do not assert that critic or actor training finds its global solution.

## C.2 Threshold Characterization for a Fixed Policy

The threshold problem becomes simpler when the evaluation policy is held fixed as well. Fix θ and a stationary policy π, and define the distribution and quantile of the oracle margin by

$$
\begin{array} { r l } & { F _ { \theta , \pi } ( x ) = \underset { s \sim d ^ { \pi } } { \operatorname* { P r } } ( m _ { \theta } ( s ) \leq x ) , } \\ & { F _ { \theta , \pi } ^ { - 1 } ( u ) = \operatorname* { i n f } \{ x : F _ { \theta , \pi } ( x ) \geq u \} . } \end{array}\tag{25}
$$

Proposition C.3 (Smallest fixed-policy threshold). For fixed $( \theta , \pi )$ , the smallest threshold satisfying (8) is

$$
\tau _ { \mathrm { m i n } } ( \theta , \pi ) = F _ { \theta , \pi } ^ { - 1 } ( 1 - \alpha ) .\tag{26}
$$

Among thresholds evaluated under the same policy, it also minimizes $\mathbb { E } _ { d ^ { \pi } } [ | \overline { { C } } _ { \theta , \tau } ( s ) | ]$

The quantile characterization follows from two facts: a hit is a margin threshold event, and the retained sets grow with the threshold under the same score. It therefore yields a smallest valid threshold and a minimum expected execution set size under a fixed occupancy. The population benchmark remains coupled because its occupancy changes with the downstream response.

## C.3 Maximum Set Size

The calibration result controls the raw set, but execution may impose a maximum size. Suppose the cap keeps at most K actions with the smallest scores, using a fixed action order to resolve ties. Write the capped raw set as $C _ { t } ^ { K }$ , and distinguish all capped misses from the hits removed specifically by truncation:

$$
\begin{array} { r l } & { e _ { t } ^ { K } = \mathbf { 1 } \{ C _ { t } ^ { K } \cap \widetilde { \mathcal { A } } _ { \varepsilon , t } ( s _ { t } ) = \emptyset \} , } \\ & { h _ { t } ^ { K } = \mathbf { 1 } \{ C _ { t } \cap \widetilde { \mathcal { A } } _ { \varepsilon , t } ( s _ { t } ) \neq \emptyset , \ C _ { t } ^ { K } \cap \widetilde { \mathcal { A } } _ { \varepsilon , t } ( s _ { t } ) = \emptyset \} . } \end{array}\tag{27}
$$

Then $e _ { t } ^ { K } = \widetilde { e } _ { t } + h _ { t } ^ { K }$ , because a capped miss is either already a raw miss or is created by truncation. Hence, when the threshold update uses the raw miss $\widetilde { e } _ { t }$

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } e _ { t } ^ { K } \le \alpha + \frac { \widetilde W _ { T } + \eta _ { 1 } } { \eta _ { T } T } + \frac { 1 } { T } \sum _ { t = 1 } ^ { T } h _ { t } ^ { K } ,\tag{28}
$$

Here $\widetilde { W } _ { T }$ is (13) for the proxy margins. The last term is not an error in tracking the raw miss rate. It records whether the ranking and available capacity preserve the proxy actions that passed the threshold.

This distinction also determines whether the retention target is feasible after capping. In the finite state and action model, fix the score, proxy target, and state distribution. A finite threshold can meet the target after capping if and only if

$$
\operatorname* { P r } { \binom { \mathrm { s o m e ~ p r o x y ~ t a r g e t ~ a c t i o n ~ r a n k s } } { \mathrm { ~ a m o n g ~ t h e ~ f i r s t ~ } } } \geq 1 - \alpha .\tag{29}
$$

Necessity follows because increasing the threshold cannot move a target action into the first K ranks. For suficiency, a threshold at least as large as every score makes the capped set equal to the global top K at each state. Thus a failed capacity condition cannot be repaired by calibration alone. For the same reason, an update using capped misses does not generally satisfy the scalar margin proof: that proof requires a target of rank at most K, so that a suficiently large finite threshold can produce a hit.

## C.4 Additional Value Results

We next connect retained actions to value. The first example shows why the selection loss cannot be omitted, even when the retention condition is perfect.

Proposition C.4 (Coverage alone is insuficient). There is a one-state, two-action discounted MDP in which the retained set always contains an optimal action, but a supported policy has value gap $( 1 - \gamma ) ^ { - 1 }$

To quantify the filtering component separately, define the largest oracle action gap and the target tolerance capped at that gap:

$$
\Delta _ { \mathrm { m a x } } = \operatorname* { m a x } _ { s , a } [ V ^ { * } ( s ) - Q ^ { * } ( s , a ) ] ,\tag{30}
$$

$$
\varepsilon _ { 0 } = \operatorname* { m i n } \{ \varepsilon , \Delta _ { \operatorname* { m a x } } \} .
$$

Corollary C.5 (Coverage bound). Let $C ( s ) \subseteq D ( s )$ for every state, assume $D ( s ) \neq \emptyset$ for every state, and let $\pi \in \Pi ( D )$ satisfy

$$
\operatorname* { P r } _ { s \sim d ^ { \pi } } ( C ( s ) \cap \mathcal { A } _ { \varepsilon } ^ { \ast } ( s ) \neq \varnothing ) \ge 1 - \alpha .\tag{31}
$$

With ζ<sub>π,D</sub> = E<sub>d</sub>π [ζ<sub>π,D</sub>(s)],

$$
V ^ { \ast } ( \mu _ { 0 } ) - V ^ { \pi } ( \mu _ { 0 } ) \leq \frac { ( 1 - \alpha ) \varepsilon _ { 0 } + \alpha \Delta _ { \operatorname* { m a x } } + \overline { { \zeta } } _ { \pi , D } } { 1 - \gamma } .\tag{32}
$$

On a hit, the filtering loss is at most the target tolerance; on a miss, it is at most the largest action gap. The corollary averages these two cases and retains the separate selection loss from Theorem 3.2. The factor $( 1 - \gamma ) ^ { - 1 }$ then accounts for accumulation along the discounted session.

Because the population benchmark imposes retention under its own downstream occupancy, the same bound applies directly to every feasible threshold in that benchmark.

Corollary C.6 (Bilevel specialization). For the fixed score $\theta ,$ every feasible threshold τ in Problem (11) satisfies Corollary C.5 with $C = C _ { \theta , \tau } , D = D _ { \theta , \tau }$ , and $\pi = \pi _ { \theta , \tau } ^ { * }$

## C.5 Practical Proxy, Critic, and Recommendation Errors

The preceding corollary assumes oracle retention. To apply it using observable feedback, we first replace the oracle target by the near maximizers of a finite proxy score $\widetilde { Q } : { \cal S } \times { \cal A }  \mathbb { R }$

$$
\widetilde { A } _ { \varepsilon } ( s ) = \{ a : \widetilde { Q } ( s , a ) \geq \operatorname* { m a x } _ { b } \widetilde { Q } ( s , b ) - \varepsilon \} .\tag{33}
$$

The proxy target is nonempty because it contains every maximizer of the finite score. Its membership depends only on diferences between actions at the same state, so absolute agreement with $Q ^ { * }$ is stronger than needed. This matters when, for example, the proxy is a response probability rather than a cumulative value. The following condition instead compares pairwise gaps after allowing for scale.

Assumption C.7 (Proxy gap fidelity). There exist $\lambda > 0$ and $\delta _ { \mathrm { p r o x } } \geq 0$ such that, for every state and pair of actions,

$$
\begin{array} { r } { \left| Q ^ { * } ( s , a ) - Q ^ { * } ( s , b ) - \lambda \big ( \widetilde { Q } ( s , a ) - \widetilde { Q } ( s , b ) \big ) \right| \leq \delta _ { \mathrm { p r o x } } . } \end{array}\tag{34}
$$

Under this condition, a small gap according to the proxy also implies a controlled gap according to the oracle. Proposition C.8 (Proxy-to-oracle target transfer). Under Assumption $C . 7 ,$

$$
\widetilde { \mathcal { A } } _ { \varepsilon } ( s ) \subseteq \mathcal { A } _ { \lambda \varepsilon + \delta _ { \mathrm { p r o x } } } ^ { * } ( s ) \qquad \forall s .\tag{35}
$$

Consequently, a hit of the proxy target is a hit of the enlarged oracle target under any state distribution.

The condition has a concrete interpretation for the response proxy used in recommendation. An immediate response can difer from optimal action value because the action also changes the continuation state. The next result separates that continuation efect from response model error.

Proposition C.9 (Immediate-response proxy). Suppose $\tilde { Q } ( s , a ) = r ( s , a )$ is the expected immediate response. Define

$$
\delta _ { \mathrm { d y n } } = \gamma \operatorname* { m a x } _ { s , a , b } \left. \sum _ { s ^ { \prime } } [ P ( s ^ { \prime } \mid s , a ) - P ( s ^ { \prime } \mid s , b ) ] V ^ { \ast } ( s ^ { \prime } ) \right. .\tag{36}
$$

Then Assumption C.7 holds with $\lambda \ : = \ : 1$ and $\delta _ { \mathrm { p r o x } } = \delta _ { \mathrm { d y n } }$ . More generally, suppose ${ \widetilde Q } ( s , a ) = { \widehat p } ( s , a )$ and $\| \widehat { p } - r \| _ { \infty } \leq \delta _ { \mathrm { r e s p } } .$ Then the assumption holds with $\lambda = 1$ and $\delta _ { \mathrm { p r o x } } = \delta _ { \mathrm { d y n } } + 2 \delta _ { \mathrm { r e s p } }$ . Using the convention $\begin{array} { r } { \mathrm { T V } ( P _ { 1 } , P _ { 2 } ) = \frac { 1 } { 2 } \dot { \sum } _ { s ^ { \prime } } | P _ { 1 } ( s ^ { \prime } ) - P _ { 2 } ( s ^ { \prime } ) | } \end{array}$ , we also have

$$
\delta _ { \mathrm { d y n } } \leq \frac { \gamma } { 1 - \gamma } \operatorname* { m a x } _ { s , a , b } \mathrm { T V } ( P ( \cdot \mid s , a ) , P ( \cdot \mid s , b ) ) .\tag{37}
$$

Thus an immediate response score approximates optimal item gaps when actions have similar continuation efects and the response model is accurate. The term $\delta _ { \mathrm { d y n } }$ records the first requirement, while $2 \delta _ { \mathrm { r e s p } }$ records the second. A proxy equal to the simulator’s exact Bernoulli parameter has zero response model error within that simulated MDP, but it can still have a large continuation error. Proxy retention is therefore not automatically retention of actions with high long term value.

Having related proxy hits to oracle hits, we now account for approximate selection. The following bound combines proxy fidelity with a critic error relative to the optimal value and an error in maximizing that critic within the retained set.

Corollary C.10 (Combined practical errors). Let $C ( s ) \subseteq D ( s )$ for every state, assume $D ( s ) \neq \emptyset$ for every state, and let $\pi \in \Pi ( D )$ . Suppose

$$
\operatorname* { P r } _ { s \sim d ^ { \pi } } \Bigl ( C ( s ) \cap \widetilde { \mathcal { A } } _ { \varepsilon } ( s ) \neq \varnothing \Bigr ) \ge 1 - \alpha ,\tag{38}
$$

Assumption C.7 holds, $\lVert \widehat { Q } - Q ^ { * } \rVert _ { \infty } \leq \delta _ { \mathrm { c r i t } }$ , and

$$
\operatorname* { m a x } _ { a \in D ( s ) } { \widehat Q } ( s , a ) - \sum _ { a \in D ( s ) } \pi ( a \mid s ) { \widehat Q } ( s , a ) \leq \delta _ { \mathrm { s e l } } ( s ) .\tag{39}
$$

With $\overline { { \delta } } _ { \mathrm { s e l } } = \mathbb { E } _ { d ^ { \pi } } [ \delta _ { \mathrm { s e l } } ( s ) ]$ and

$$
\varepsilon _ { \mathrm { r e c } } = \operatorname* { m i n } \{ \lambda \varepsilon + \delta _ { \mathrm { p r o x } } , \Delta _ { \mathrm { m a x } } \} ,\tag{40}
$$

we have

$$
\begin{array} { r } { V ^ { \ast } ( \mu _ { 0 } ) - V ^ { \pi } ( \mu _ { 0 } ) \leq \displaystyle \frac { 1 } { 1 - \gamma } \Big [ ( 1 - \alpha ) \varepsilon _ { \mathrm { r e c } } + \alpha \Delta _ { \mathrm { m a x } } } \\ { + 2 \delta _ { \mathrm { c r i t } } + \overline { { \delta } } _ { \mathrm { s e l } } \Big ] . } \end{array}\tag{41}
$$

The bound identifies what each component must supply: calibration controls proxy misses, proxy fidelity bounds retained action quality, and critic accuracy and selection quality control the remaining decision loss. These are conditional requirements. In particular, a critic that evaluates the current policy or a restricted policy class need not have small error relative to the unrestricted $Q ^ { * }$

For the simulator, it is also useful to study immediate patience changes without claiming a complete session value guarantee. The next result distinguishes a rule that tests the entire collection for a positive response from a rule that tests only an item selected before the response.

Proposition C.11 (Response retention under two patience mechanisms). Suppose that, conditional on state s, a jointly distributed potential-response vector $( Y _ { a } ) _ { a \in \mathcal { A } }$ is defined and that $D ( s )$ is nonempty and determined before this vector is sampled. Let

$$
p ( s , a ) = \operatorname* { P r } ( Y _ { a } = 1 \mid s ) , \qquad p ^ { * } ( s ) = \operatorname* { m a x } _ { a } p ( s , a ) .
$$

For a collection-based response rule, define

$$
p _ { D } ^ { \mathrm { s e t } } ( s ) = \operatorname* { P r } \left( \bigcup _ { a \in D ( s ) } \left\{ Y _ { a } = 1 \right\} \Bigg | s \right) .
$$

For a pre-response item selector $\kappa ,$ assume that its internal randomization is conditionally independent of the potential-response vector given s and $D ( s )$ , and define

$$
p _ { D , \kappa } ^ { \mathrm { i t e m } } ( s ) = \sum _ { a \in D ( s ) } \kappa ( a \mid s , D ( s ) ) p ( s , a )
$$

and

$$
\delta _ { \mathrm { s e l } } ^ { p } ( s ) = \operatorname* { m a x } _ { a \in D ( s ) } p ( s , a ) - p _ { D , \kappa } ^ { \mathrm { i t e m } } ( s ) .
$$

Suppose that, under a state distribution $\nu ,$ the set $D ( s )$ contains an action satisfying $p ( s , a ) \geq p ^ { * } ( s ) - \varepsilon$ with probability at least $1 - \alpha$ . Let

$$
\varepsilon _ { p } = \operatorname* { m i n } \{ \varepsilon , 1 \} , \qquad \bar { \delta } _ { \mathrm { s e l } } ^ { p } = \mathbb { E } _ { \nu } [ \delta _ { \mathrm { s e l } } ^ { p } ( s ) ] .
$$

Then the collection-based rule satisfies

$$
\mathbb { E } _ { \nu } [ 1 - p _ { D } ^ { \mathrm { s e t } } ( s ) ] \leq \mathbb { E } _ { \nu } [ 1 - p ^ { * } ( s ) ] + ( 1 - \alpha ) \varepsilon _ { p } + \alpha .\tag{42}
$$

No independence among the item responses is required. The pre-response item selector satisfies

$$
\begin{array} { r } { \mathbb { E } _ { \nu } [ 1 - p _ { D , \kappa } ^ { \mathrm { i t e m } } ( s ) ] \leq \mathbb { E } _ { \nu } [ 1 - p ^ { * } ( s ) ] + ( 1 - \alpha ) \varepsilon _ { p } + \alpha + \bar { \delta } _ { \mathrm { s e l } } ^ { p } . } \end{array}\tag{43}
$$

If patience decreases by $c _ { + }$ after a positive response and by $c _ { - } > c _ { + }$ otherwise, the excess expected one-step patience decrease relative to the best single item is at most

$$
( c _ { - } - c _ { + } ) \big [ ( 1 - \alpha ) \varepsilon _ { p } + \alpha \big ]\tag{44}
$$

for the collection-based rule, and at most

$$
( c _ { - } - c _ { + } ) \big [ ( 1 - \alpha ) \varepsilon _ { p } + \alpha + \overline { { \delta } } _ { \mathrm { s e l } } ^ { p } \big ]\tag{45}
$$

for the pre-response item selector.

The diference between the bounds again comes from selection. A positive response anywhere in the collection is at least as likely as a positive response from its best item, whereas a selector may choose a worse item and incur the additional selection term. Both conclusions concern one interaction. Extending either to session depth requires the full patience evolution, state transition, and departure model.

## C.6 Full Conditions for the Finite Session Bound

The discounted results above concern stationary policies. Theorem 3.3 instead allows the policy and score to change within a session. To make that extension precise, the following assumption groups the information available before selection, the calibration and execution rules, and the approximation errors. It imposes no convergence requirement on the parameters that generate these quantities.

Assumption C.12 (Finite-session information and approximation errors). Consider an undiscounted episodic MDP with finite nonempty state and action spaces S and ${ \mathcal { A } } ,$ initial state $s _ { 1 } \sim \mu _ { 0 }$ , horizon H, rewards $R _ { h } \in [ 0 , 1 ]$ 2 optimal functions $V _ { h } ^ { * } , Q _ { h } ^ { * }$ , and terminal value $V _ { H + 1 } ^ { * } = 0$ . Let $\mathcal { H } _ { h }$ be the sigma field containing the trajectory and all algorithmic information available immediately before the action at stage h is sampled. In particular, $s _ { h }$ is $\mathcal { H } _ { h }$ measurable. Let $r _ { h } : S \times \mathcal { A }  [ 0 , 1 ]$ and $P _ { h } ( \cdot \mid s , a )$ be the controlled stage reward mean and transition kernel, so that

$$
\begin{array} { r l } & { \mathbb { E } [ R _ { h } \mid \mathcal { H } _ { h } , a _ { h } ] = r _ { h } ( s _ { h } , a _ { h } ) , } \\ & { \quad \mathrm { P r } ( s _ { h + 1 } = s ^ { \prime } \mid \mathcal { H } _ { h } , a _ { h } ) } \\ & { \qquad = P _ { h } ( s ^ { \prime } \mid s _ { h } , a _ { h } ) . } \end{array}
$$

Let

$$
\Delta _ { H } ^ { \operatorname* { m a x } } = \operatorname* { m a x } _ { 1 \leq h \leq H } \operatorname* { m a x } _ { s , a } \big [ V _ { h } ^ { * } ( s ) - Q _ { h } ^ { * } ( s , a ) \big ] .
$$

At every stage $h ,$ suppose that the threshold $\tau _ { h }$ and the action-indexed quantities

$$
\begin{array} { c c c } { { g _ { h } ( a ) = g _ { \theta _ { h } } ( s _ { h } , a ) , } } & { { { } } } & { { \widetilde { Q } _ { h } ( s _ { h } , a ) , } } \\ { { { } } } & { { { } } } & { { { } } } \\ { { \widehat { Q } _ { h } ( s _ { h } , a ) , } } & { { { } } } & { { a \in { \mathcal A } , } } \end{array}
$$

are finite and $\mathcal { H } _ { h }$ measurable. Define the stage-dependent proxy target, proxy margin, raw set, and raw proxy miss $b y$

$$
\widetilde { \mathcal { A } } _ { \varepsilon , h } ( s _ { h } ) = \left\{ a \in \mathcal { A } : \widetilde { Q } _ { h } ( s _ { h } , a ) \geq \operatorname* { m a x } _ { b \in \mathcal { A } } \widetilde { Q } _ { h } ( s _ { h } , b ) - \varepsilon \right\} ,
$$

$$
\widetilde { m } _ { h } = \operatorname* { m i n } _ { a \in \widetilde { A } _ { \varepsilon , h } ( s _ { h } ) } { g _ { h } ( a ) } ,
$$

$$
C _ { h } = \{ a \in { \mathcal { A } } : g _ { h } ( a ) \leq \tau _ { h } \} ,
$$

$$
\widetilde { e } _ { h } = \mathbf { 1 } \{ \widetilde { m } _ { h } > \tau _ { h } \} = \mathbf { 1 } \{ C _ { h } \cap \widetilde { A } _ { \varepsilon , h } ( s _ { h } ) = \emptyset \} .
$$

Let the threshold follow

$$
\tau _ { h + 1 } = \tau _ { h } + \eta _ { h } ( \widetilde { e } _ { h } - \alpha ) , \qquad h = 1 , \ldots , H ,\tag{46}
$$

where $\eta _ { 1 } \ge \eta _ { 2 } \ge \cdots \ge \eta _ { H } > 0$ is deterministic.

Fix $K \in \{ 1 , \ldots , | A | \}$ . Using the fixed action order to break ties, let $C _ { h } ^ { K }$ contain the min $\{ K , | C _ { h } | \}$ elements of $C _ { h }$ having the smallest scores $g _ { h }$ . Let $a _ { h } ^ { \mathrm { m i n } }$ be the first action in arg min $\iota _ { a \in \mathcal { A } } g _ { h } ( a )$ and define

$$
D _ { h } = \left\{ { \begin{array} { l l } { C _ { h } ^ { K } , } & { C _ { h } ^ { K } \neq \emptyset , } \\ { \{ a _ { h } ^ { \operatorname* { m i n } } \} , } & { C _ { h } ^ { K } = \emptyset . } \end{array} } \right.
$$

The tie-breaking rule makes $C _ { h } ^ { K }$ and $D _ { h } \ \mathcal { H } _ { h }$ measurable. Conditional on $\mathcal { H } _ { h }$ , the action is sampled from an $\mathcal { H } _ { h }$ measurable probability kernel supported on $D _ { h }$ , before the current reward and next state are generated. In particular,

$$
\operatorname* { P r } ( a _ { h } \in D _ { h } \mid \mathcal { H } _ { h } ) = 1 .
$$

Define

$$
\begin{array} { r l } & { e _ { h } ^ { K } = \mathbf { 1 } \{ C _ { h } ^ { K } \cap \widetilde { \mathcal { A } } _ { \varepsilon , h } ( s _ { h } ) = \emptyset \} , } \\ & { h _ { h } ^ { K } = \mathbf { 1 } \{ C _ { h } \cap \widetilde { \mathcal { A } } _ { \varepsilon , h } ( s _ { h } ) \not = \emptyset , \ C _ { h } ^ { K } \cap \widetilde { \mathcal { A } } _ { \varepsilon , h } ( s _ { h } ) = \emptyset \} , } \\ & { \widetilde { W } _ { H } = \operatorname* { m a x } \{ \tau _ { 1 } , \widetilde { m } _ { 1 } , \dots , \widetilde { m } _ { H } \} - \operatorname* { m i n } \{ \tau _ { 1 } , \widetilde { m } _ { 1 } , \dots , \widetilde { m } _ { H } \} . } \end{array}
$$

Assume that $\widetilde { W } _ { H } \leq W$ almost surely for a deterministic finite constant W. Also assume that there exist $\lambda > 0$ $\delta _ { \mathrm { p r o x } } \geq 0$ , and $\delta _ { \mathrm { c r i t } } \geq 0$ such that, almost surely, for every h and every $a , b \in A$

$$
\begin{array} { r l r } {  { \Big | Q _ { h } ^ { * } \big ( s _ { h } , a \big ) - Q _ { h } ^ { * } \big ( s _ { h } , b \big ) - \lambda \big ( \widetilde { Q } _ { h } \big ( s _ { h } , a \big ) } } \\ & { } & { \qquad - \widetilde { Q } _ { h } \big ( s _ { h } , b \big ) \big ) \Big | \le \delta _ { \mathrm { p r o x } } . } \end{array}\tag{47}
$$

$$
\operatorname* { m a x } _ { a \in \mathcal { A } } \left| \widehat { Q } _ { h } ( s _ { h } , a ) - Q _ { h } ^ { * } ( s _ { h } , a ) \right| \leq \delta _ { \mathrm { c r i t } } .\tag{48}
$$

Finally, suppose that $\delta _ { \mathrm { s e l } , h }$ is a nonnegative, integrable, $\mathcal { H } _ { h }$ measurable random variable and that, almost surely,

$$
\operatorname* { m a x } _ { a \in D _ { h } } \widehat { Q } _ { h } ( s _ { h } , a ) - \mathbb { E } \Big [ \widehat { Q } _ { h } ( s _ { h } , a _ { h } ) \mid \mathcal { H } _ { h } \Big ] \leq \delta _ { \mathrm { s e l } , h } .\tag{49}
$$

Define

$$
\varepsilon _ { H } = \operatorname* { m i n } \{ \lambda \varepsilon + \delta _ { \mathrm { p r o x } } , \Delta _ { H } ^ { \mathrm { m a x } } \} ,
$$

$$
\overline { { \delta } } _ { \mathrm { s e l } , H } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \mathbb { E } [ \delta _ { \mathrm { s e l } , h } ] ,
$$

$$
\overline { { h } } _ { H } ^ { K } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \mathbb { E } [ h _ { h } ^ { K } ] .
$$

Taking $K = | { \mathcal { A } } |$ represents the case without a cap and gives $\overline { { h } } _ { H } ^ { K } = 0$

The information condition permits the critic and selector to depend on past observations. It excludes using the current action’s realized response to define the target or execution set for that same action. The approximation conditions then apply at the visited states, with critic error measured relative to unrestricted $Q _ { h } ^ { * }$ rather than the current policy value. Thus the theorem explains how these errors afect performance; it does not establish that an arbitrary learning update makes them small.

## C.7 Greedy Preservation and Conditional Policy Recovery

The previous results allow arbitrary supported selectors. For a selector greedy with respect to the scoring critic, however, a simpler boundary case applies: the critic gap score always gives a minimum score to a critic maximizer. To describe its consequences, fix finite critics $\widehat { Q } _ { t } ,$ thresholds $\tau _ { t } ,$ and caps $K _ { t } \geq 1$ . Construct $D _ { t } ( s )$

by thresholding the critic gap, keeping at most $K _ { t }$ actions with the smallest scores, and applying the minimum score fallback when needed. Use the same fixed action order for all ties.

The remaining performance error can then be expressed using the critic’s pairwise error and the selector’s gap within the retained set. For a stationary policy $\pi _ { t }$ supported on $D _ { t } .$ , define

$$
\xi _ { t } ( s , a ) = \widehat { Q } _ { t } ( s , a ) - Q ^ { * } ( s , a ) ,
$$

$$
\beta _ { t } = \operatorname* { m a x } _ { s , a , b } | \xi _ { t } ( s , a ) - \xi _ { t } ( s , b ) | ,
$$

$$
\sigma _ { t } ( s ) = \operatorname* { m a x } _ { a \in D _ { t } ( s ) } \widehat { Q } _ { t } ( s , a ) - \sum _ { a \in D _ { t } ( s ) } \pi _ { t } ( a \mid s ) \widehat { Q } _ { t } ( s , a ) ,
$$

$$
\overline { { \sigma } } _ { t } = \mathbb { E } _ { s \sim d ^ { \pi _ { t } } } [ \sigma _ { t } ( s ) ] .
$$

If any strictly suboptimal state–action pair exists, let

$$
\Delta _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { \substack { s , a : Q ^ { * } ( s , a ) < V ^ { * } ( s ) } } [ V ^ { * } ( s ) - Q ^ { * } ( s , a ) ] > 0 .
$$

If all actions are optimal, set $\Delta _ { \operatorname* { m i n } } = + \infty$

Proposition C.13 (Greedy preservation and conditional recovery). Every operational set $D _ { t } ( s )$ contains the first global maximizer $o f \widehat { Q } _ { t } ( s , \cdot )$ . Hence a greedy selector with the same tie-breaking rule returns the same action with or without filtering. Moreover,

$$
V ^ { * } ( \mu _ { 0 } ) - V ^ { \pi _ { t } } ( \mu _ { 0 } ) \leq \frac { \overline { { \sigma } } _ { t } + \beta _ { t } } { 1 - \gamma } , \qquad \beta _ { t } \leq 2 \| \widehat { Q } _ { t } - Q ^ { * } \| _ { \infty } .
$$

$I f \beta _ { t } < \Delta _ { \operatorname* { m i n } }$ , every global critic maximizer is truly optimal. A policy greedy within $D _ { t }$ is then optimal. Consequently, $\beta _ { t } + \overline { { \sigma } } _ { t }  0$ implies value convergence; for exact greedy selection, $\beta _ { t } \to 0$ implies eventual optimality in a finite MDP. These are conditional implications, not convergence guarantees for the learning updates.

This result does not replace the calibration analysis. It identifies a case in which filtering leaves the chosen action unchanged, so any benefit from controlling the available alternatives must arise through a diferent downstream selection mechanism. The recovery implications remain conditional on critic and selector errors.

## D COMPLETE PROOFS

The proofs follow the dependencies in the main analysis. Bellman identities first establish the reference values. Restricted control and occupancy arguments then justify the population benchmark. The calibration proof controls the miss sequence, while the value proofs convert misses and selection errors into reward loss. Finally, the greedy preservation argument identifies the boundary case in which filtering does not change the selected action.

## D.1 Bellman Identities and Value Bounds

The following identities connect the trajectory definitions of value to the Bellman operators used in the later proofs. The bounded reward assumption then supplies the largest possible action and policy gaps.

Lemma D.1 (Bellman identities). For every stationary policy π,

$$
\begin{array} { c l } { { } } & { { Q ^ { \pi } ( s , a ) = r ( s , a ) + \gamma \displaystyle \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) V ^ { \pi } ( s ^ { \prime } ) , } } \\ { { } } & { { } } \\ { { { \displaystyle V ^ { \pi } ( s ) = \sum _ { a } \pi ( a \mid s ) Q ^ { \pi } ( s , a ) . } } } \end{array}\tag{50}
$$

The optimal values satisfy

$$
Q ^ { * } ( s , a ) = r ( s , a ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) V ^ { * } ( s ^ { \prime } ) ,
$$

$$
V ^ { * } ( s ) = \operatorname* { m a x } _ { a } Q ^ { * } ( s , a ) .\tag{51}
$$

Any stationary policy supported on arg max ${ } _ { a } Q ^ { * } ( s , a )$ at every state is optimal.

Proof. We prove the policy identities first. Fix a stationary policy $\pi ,$ a state $s ,$ and an action a. By the forced-action definition,

$$
Q ^ { \pi } ( s , a ) = \mathbb { E } _ { s } ^ { a \triangleright \pi } \left[ R _ { 0 } + \sum _ { t = 1 } ^ { \infty } \gamma ^ { t } R _ { t } \right] .
$$

The controlled reward model gives

$$
\mathbb { E } _ { s } ^ { a \triangleright \pi } [ R _ { 0 } ] = r ( s , a ) .
$$

For the remaining terms, condition on the next state and shift the time index:

$$
\begin{array} { r l } { \displaystyle \mathbb { E } _ { s } ^ { a \scriptscriptstyle { \triangleright } \pi } \left[ \sum _ { t = 1 } ^ { \infty } \gamma ^ { t } R _ { t } \right] = \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) \mathbb { E } _ { s ^ { \prime } } ^ { \pi } \left[ \sum _ { j = 0 } ^ { \infty } \gamma ^ { j } R _ { j } \right] } & { } \\ { \displaystyle } & { = \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) V ^ { \pi } ( s ^ { \prime } ) . } \end{array}
$$

The first equality uses (1) and the fact that the policy follows π from time one onward. Hence

$$
Q ^ { \pi } ( s , a ) = r ( s , a ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) V ^ { \pi } ( s ^ { \prime } ) .
$$

Under $\pi ,$ the time-zero action has distribution $\pi ( \cdot \mid s )$ . The trajectory law is therefore the corresponding mixture of the forced-action laws:

$$
\begin{array} { l } { { { \cal V } ^ { \pi } ( s ) = \displaystyle \sum _ { a } \pi ( a \mid s ) \mathbb { E } _ { s } ^ { a \triangleright \pi } \left[ \displaystyle \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } R _ { t } \right] } } \\ { { { } = \displaystyle \sum _ { a } \pi ( a \mid s ) Q ^ { \pi } ( s , a ) . } } \end{array}
$$

This establishes (50).

We next establish the optimal identities. Define

$$
( \mathcal { T } v ) ( s ) = \operatorname* { m a x } _ { a \in \mathcal { A } } \left[ r ( s , a ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) v ( s ^ { \prime } ) \right] .
$$

For a stationary policy $\pi ,$ also define its Bellman operator by

$$
( \mathcal T ^ { \pi } v ) ( s ) = \sum _ { a } \pi ( a \mid s ) \left[ r ( s , a ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) v ( s ^ { \prime } ) \right] .
$$

For arbitrary bounded functions v, w,

$$
\begin{array} { l } { | ( { \mathcal T } v ) ( s ) - ( { \mathcal T } w ) ( s ) | \leq \displaystyle \operatorname* { m a x } _ { a } \left| \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) [ v ( s ^ { \prime } ) - w ( s ^ { \prime } ) ] \right| } \\ { \displaystyle \qquad \leq \gamma \| v - w \| _ { \infty } . } \end{array}
$$

Taking the maximum over s $\mathrm { g i }$ ves

$$
\| T v - T w \| _ { \infty } \leq \gamma \| v - w \| _ { \infty } .
$$

Thus $\tau$ has a unique fixed point.

Let $\overline { V }$ denote the unique fixed point of $\tau .$ Set $v _ { 0 } \equiv 0$ and $v _ { H } = \mathcal { T } ^ { H } 0$ . Contraction of $\tau$ gives

$$
v _ { H } \longrightarrow \overline { { { V } } } \qquad \mathrm { i n ~ t h e ~ s u p ~ n o r m } .
$$

The iterates also have the standard finite-horizon interpretation: by induction, $v _ { H } ( s )$ is the largest expected discounted reward collected during the first H decision stages.

We now identify $\overline { V }$ with the value $V ^ { * }$ defined as the maximum over stationary policies, without assuming in advance that a stationary optimal policy exists. Fix any stationary policy π. For every function v,

$$
\tau ^ { \pi } v \leq \tau _ { v }
$$

componentwise because an average over actions cannot exceed their maximum. Both operators are monotone, so induction from the zero function gives

$$
( T ^ { \pi } ) ^ { H } 0 \leq { \cal T } ^ { H } 0 = v _ { H } \qquad \mathrm { f o r ~ e v e r y ~ } H .
$$

The policy operator $\mathcal { T } ^ { \pi }$ is a contraction, so its iterates converge to $V ^ { \pi }$ . Letting $H \to \infty$ yields

$$
V ^ { \pi } \leq { \overline { { V } } } .
$$

Since this holds for every stationary policy,

$$
V ^ { * } \leq { \overline { { V } } } .
$$

For the reverse inequality, choose a stationary deterministic policy $\overline { { \pi } }$ satisfying

$$
\overline { { \pi } } ( s ) \in \mathop { \arg \operatorname* { m a x } } _ { a \in \mathcal { A } } \left[ r ( s , a ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) \overline { { V } } ( s ^ { \prime } ) \right]
$$

at every state. Such a choice exists because A is finite. Since ${ \overline { { V } } } = { \mathcal { T } } { \overline { { V } } }$ , the greedy construction gives

$$
{ \mathcal { T } } ^ { \overline { { \pi } } } { \overline { { V } } } = { \mathcal { T } } { \overline { { V } } } = { \overline { { V } } } .
$$

The operator $\mathcal T ^ { \overline { { \pi } } }$ has the unique fixed point $V ^ { \overline { { \pi } } } .$ , and therefore

$$
V ^ { \overline { { \pi } } } = { \overline { { V } } } .
$$

Because π is stationary, $\overline { { \pi } }$

$$
V ^ { * } \geq V ^ { \overline { { \pi } } } = \overline { { V } } .
$$

Combining both inequalities proves

$$
V ^ { * } = \overline { { V } } = \mathcal { T } V ^ { * } .
$$

This argument simultaneously proves existence of a stationary deterministic optimal policy.

Define

$$
\overline { { { Q } } } ( s , a ) = r ( s , a ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) V ^ { * } ( s ^ { \prime } ) .
$$

It remains to connect this Bellman action value to the trajectory definition $Q ^ { * } ( s , a ) \ : = \ : \operatorname* { m a x } _ { \pi } Q ^ { \pi } ( s , a )$ . We establish the two inequalities separately. For every stationary policy π, the policy value satisfies $V ^ { \pi } \leq V ^ { * }$ componentwise. Therefore

$$
\begin{array} { l } { { \displaystyle Q ^ { \pi } ( s , a ) = r ( s , a ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) V ^ { \pi } ( s ^ { \prime } ) } } \\ { { \mathrm { } } } \\ { { \displaystyle \qquad \leq r ( s , a ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) V ^ { * } ( s ^ { \prime } ) } } \\ { { \mathrm { } } } \\ { { \displaystyle = \overline { { { Q } } } ( s , a ) . } } \end{array}
$$

Taking the maximum over π gives

$$
Q ^ { \ast } ( s , a ) \leq \overline { { { Q } } } ( s , a ) .
$$

For the reverse inequality, choose a stationary policy $\pi ^ { * }$ whose support at every state is contained in

$$
\underset { b } { \arg \operatorname* { m a x } } \left[ r ( s , b ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , b ) V ^ { * } ( s ^ { \prime } ) \right] .
$$

Such a policy exists because the action set is finite. Its policy Bellman operator satisfies

$$
{ \mathcal { T } } ^ { \pi ^ { * } } V ^ { * } = { \mathcal { T } } V ^ { * } = V ^ { * } .
$$

Since $\tau ^ { \pi ^ { * } }$ is a contraction with unique fixed point $V ^ { \pi ^ { * } }$ , we have

$$
V ^ { \pi ^ { * } } = V ^ { * } .
$$

Consequently,

$$
{ Q ^ { \pi } } ^ { * } ( s , a ) = r ( s , a ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) V ^ { * } ( s ^ { \prime } ) = \overline { { { Q } } } ( s , a ) .
$$

Because $Q ^ { * }$ is the maximum over policies,

$$
Q ^ { * } ( s , a ) \geq Q ^ { \pi ^ { * } } ( s , a ) = \overline { { { Q } } } ( s , a ) .
$$

Combining the two inequalities gives

$$
Q ^ { * } ( s , a ) = r ( s , a ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) V ^ { * } ( s ^ { \prime } ) .
$$

Substituting this identity into the fixed-point equation for $V ^ { * }$ yields

$$
V ^ { * } ( s ) = \operatorname* { m a x } _ { a } Q ^ { * } ( s , a ) .
$$

Finally, any stationary policy whose support is contained in arg ma $\mathfrak { c } _ { a } Q ^ { * } ( s , a )$ satisfies

$$
{ \cal T } ^ { \pi } V ^ { * } = V ^ { * } .
$$

The unique fixed point of $\tau ^ { \pi }$ is $V ^ { \pi }$ , so $V ^ { \pi } = V ^ { * }$ . Such a policy is therefore optimal.

Lemma D.2 (Value bounds). For every $\pi \in \Pi$ , state $s ,$ and action $a ,$

$$
0 \leq V ^ { \pi } ( s ) , Q ^ { \pi } ( s , a ) , V ^ { * } ( s ) , Q ^ { * } ( s , a ) \leq \frac { 1 } { 1 - \gamma } .
$$

Consequently, $0 \leq \Delta _ { \operatorname* { m a x } } \leq ( 1 - \gamma ) ^ { - 1 }$

Proof. For every sample path,

$$
0 \leq \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } R _ { t } \leq \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } = \frac { 1 } { 1 - \gamma } .
$$

Taking expectations under the corresponding controlled trajectory laws preserves both inequalities, proving the bounds for $V ^ { \pi }$ and $Q ^ { \pi }$ . Taking maxima over policies gives the same bounds for $V ^ { * }$ and $Q ^ { * }$ . Finally, $V ^ { * } ( s ) = \operatorname* { m a x } _ { b } Q ^ { * } ( s , b ) \geq Q ^ { * } ( s , a )$ , so each diference in (30) is nonnegative and at most $( 1 - \gamma ) ^ { - 1 }$ □

## D.2 Proofs for the Bilevel Formulation

We first establish a supported optimal policy and its occupancy representation. The occupancy formulation then provides a deterministic choice among optimal responses, so that the outer objective has a definite value for every threshold.

Proof of Theorem C.1. Fix a set map D satisfying $D ( s ) \neq \emptyset$ for every state. We will prove that control over this set is equivalent to maximizing reward over supported discounted occupancies. For reference, define

$$
\operatorname* { m a x } _ { \rho \in \mathcal { D } ( D ) } \sum _ { s , a } \rho ( s , a ) r ( s , a ) ,\tag{52}
$$

where

$$
\begin{array} { c l } { \displaystyle \mathcal { D } ( D ) = \{ \rho \geq 0 : \sum _ { a } \rho ( s , a ) = ( 1 - \gamma ) \mu _ { 0 } ( s ) } \\ { \displaystyle + \gamma \sum _ { s ^ { \prime } , a ^ { \prime } } P ( s \mid s ^ { \prime } , a ^ { \prime } ) \rho ( s ^ { \prime } , a ^ { \prime } ) , \ \forall s , } \\ { \displaystyle \rho ( s , a ) = 0 \ \mathrm { f o r } \ a \not \in D ( s ) \} . } \end{array}\tag{53}
$$

The flow constraints record discounted state visitation and rule out actions outside $D .$ . The corresponding dual places an upper bound on reward plus continuation value for every retained action:

$$
\begin{array} { r l r } { \underset { v } { \mathrm { m i n } } } & { ( 1 - \gamma ) \displaystyle \sum _ { s } \mu _ { 0 } ( s ) v ( s ) } & \\ { \mathrm { s u b j e c t ~ t o } } & { v ( s ) \geq r ( s , a ) + \gamma \displaystyle \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) v ( s ^ { \prime } ) , } \\ & { } & { \forall s , ~ a \in D ( s ) . } \end{array}\tag{54}
$$

For each state s and action $a \in D ( s )$ , define

$$
h _ { a } ( v ) = r ( s , a ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) v ( s ^ { \prime } ) .
$$

For any two finite families of real numbers $\{ x _ { a } \}$ and $\{ y _ { a } \}$ 2

$$
\left| \operatorname* { m a x } _ { a } x _ { a } - \operatorname* { m a x } _ { a } y _ { a } \right| \leq \operatorname* { m a x } _ { a } | x _ { a } - y _ { a } | .
$$

Applying this inequality with $x _ { a } = h _ { a } ( v )$ and $y _ { a } = h _ { a } ( w )$ gives

$$
\begin{array} { r l } { \lceil \mathcal T _ { D } v ( s ) - \mathcal T _ { D } w ( s ) \rceil \leq \underset { a \in D ( s ) } { \operatorname* { m a x } } \left| \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) [ v ( s ^ { \prime } ) - w ( s ^ { \prime } ) ] \right| } & { } \\ { \quad } & { \leq \gamma \underset { a \in D ( s ) } { \operatorname* { m a x } } \underset { s ^ { \prime } } { \sum } P ( s ^ { \prime } \mid s , a ) | v ( s ^ { \prime } ) - w ( s ^ { \prime } ) | } \\ { \quad } & { \leq \gamma \| v - w \| _ { \infty } \underset { a \in D ( s ) } { \operatorname* { m a x } } \underset { s ^ { \prime } } { \sum } P ( s ^ { \prime } \mid s , a ) } \\ { \quad } & { = \gamma \| v - w \| _ { \infty } . } \end{array}
$$

Taking the maximum over s proves

$$
\begin{array} { r } { \| T _ { D } v - T _ { D } w \| _ { \infty } \leq \gamma \| v - w \| _ { \infty } . } \end{array}
$$

Because $\gamma < 1$ , the Banach fixed-point theorem gives a unique fixed point, denoted $V _ { D } ^ { * }$

For each state s, the set $D ( s )$ is finite and nonempty, so there exists

$$
a _ { D } ^ { * } ( s ) \in \arg \operatorname* { m a x } _ { a \in D ( s ) } \left[ r ( s , a ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) V _ { D } ^ { * } ( s ^ { \prime } ) \right] .
$$

Let $\pi _ { D } ^ { * }$ select $a _ { D } ^ { * } ( s )$ with probability one. Define the policy Bellman operator

$$
( \mathcal T ^ { \pi } v ) ( s ) = \sum _ { a } \pi ( a \mid s ) \left[ r ( s , a ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) v ( s ^ { \prime } ) \right] .
$$

By the choice of $a _ { D } ^ { * } ( s )$ ，

$$
\begin{array} { r } { \mathcal { T } ^ { \pi _ { D } ^ { * } } V _ { D } ^ { * } = \mathcal { T } _ { D } V _ { D } ^ { * } = V _ { D } ^ { * } . } \end{array}
$$

The operator $\tau ^ { \pi _ { D } ^ { * } }$ is $\mathrm { ~ a ~ } \gamma$ contraction by the same calculation as above, and its unique fixed point is $V ^ { \pi _ { D } ^ { * } }$ . Hence

$$
V ^ { \pi _ { D } ^ { * } } = V _ { D } ^ { * } .
$$

Now fix any $\pi \in \Pi ( D )$ . For every v and every state,

$$
\begin{array} { r l } & { ( T ^ { \pi } v ) ( s ) = \displaystyle \sum _ { a \in D ( s ) } \pi ( a \mid s ) \left[ r ( s , a ) + \gamma \displaystyle \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) v ( s ^ { \prime } ) \right] } \\ & { \quad \quad \quad \le \displaystyle \operatorname* { m a x } _ { a \in D ( s ) } \left[ r ( s , a ) + \gamma \displaystyle \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) v ( s ^ { \prime } ) \right] } \\ & { \quad \quad = ( \mathcal T _ { D } v ) ( s ) . } \end{array}
$$

Both operators are monotone. Starting from the zero function and applying the preceding inequality inductively yields

$$
( T ^ { \pi } ) ^ { n } 0 \leq { \cal T } _ { { D } } ^ { n } 0 \qquad \mathrm { f o r ~ e v e r y ~ } n .
$$

The left side converges to $V ^ { \pi }$ and the right side converges to $V _ { D } ^ { * }$ because both operators are contractions. Therefore

$$
V ^ { \pi } \leq V _ { D } ^ { * }
$$

componentwise. This proves that $\pi _ { D } ^ { * }$ is optimal over $\Pi ( D )$

For $\pi \in \Pi ( D )$ , define

$$
\rho ^ { \pi } ( s , a ) = ( 1 - \gamma ) \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \operatorname* { P r } ^ { \pi } ( s _ { t } = s , a _ { t } = a \mid s _ { 0 } \sim \mu _ { 0 } ) .
$$

Every component is nonnegative. If a $\notin D ( s )$ , then $\pi ( a \mid s ) = 0$ , and hence $\operatorname* { P r } ^ { \pi } ( s _ { t } = s , a _ { t } = a ) = 0$ for every $t ;$ therefore $\rho ^ { \pi } ( s , a ) = 0$

Let

$$
d ^ { \pi } ( s ) = \sum _ { a } \rho ^ { \pi } ( s , a ) .
$$

Using the law of total probability and the Markov property,

$$
\begin{array} { r l } & { d ^ { x } ( s ) = ( 1 - \gamma ) \displaystyle \sum _ { i = 0 } ^ { \infty } \gamma ^ { i } \bar { \mathbf { P } } \overline { { \mathbf { r } } } ( s _ { t } = s ) } \\ & { \qquad = ( 1 - \gamma ) \mu _ { 0 } ( s ) + ( 1 - \gamma ) \displaystyle \sum _ { i = 0 } ^ { \infty } \gamma ^ { i } \bar { \mathbf { P } } \Gamma ( s _ { t } = s ) } \\ & { \qquad = ( 1 - \gamma ) \mu _ { 0 } ( s ) + \gamma ( 1 - \gamma ) \displaystyle \sum _ { j = 0 } ^ { \infty } \gamma ^ { i } \bar { \mathbf { P } } \Gamma ( s _ { j + 1 } = s ) } \\ & { \qquad = ( 1 - \gamma ) \mu _ { 0 } ( s ) } \\ & { \qquad + \gamma ( 1 - \gamma ) \displaystyle \sum _ { j = 0 } ^ { \infty } \gamma ^ { j } \displaystyle \sum _ { s \in \partial \mathcal { S } } \bar { \mathbf { P } } \bar { \mathbf { r } } ( s _ { j } = s ^ { \prime } , a _ { j } = a ^ { \prime } ) P ( s \mid s ^ { \prime } , a ^ { \prime } ) } \\ & { \qquad = ( 1 - \gamma ) \mu _ { 0 } ( s ) + \gamma \displaystyle \sum _ { s \in \partial \mathcal { S } } P ( s \mid s ^ { \prime } , a ^ { \prime } ) \rho ^ { \prime } ( s ^ { \prime } , a ^ { \prime } ) . } \end{array}
$$

Thus $\rho ^ { \pi } \in \mathcal { D } ( D )$

Let $\rho \in \mathcal { D } ( D )$ and define

$$
d ( s ) = \sum _ { a } \rho ( s , a ) .
$$

If $d ( s ) > 0 ;$ , set

$$
\pi _ { \rho } ( a \mid s ) = { \frac { \rho ( s , a ) } { d ( s ) } } .
$$

If $d ( s ) = 0$ , let $\pi _ { \rho } ( \cdot \mid s )$ place probability one on the first action in $D ( s )$ under the fixed global action order. This action exists because $D ( s ) \neq \emptyset$ . The confinement constraints imply that $\pi _ { \rho } ( a \mid s ) = 0$ for every a $\notin D ( s )$ so $\pi _ { \rho } \in \Pi ( D )$

Define

$$
P ^ { \pi _ { \rho } } ( s ^ { \prime } \mid s ) = \sum _ { a } \pi _ { \rho } ( a \mid s ) P ( s ^ { \prime } \mid s , a ) .
$$

We claim that the flow constraints for $\rho$ can be written as

$$
d = ( 1 - \gamma ) \mu _ { 0 } + \gamma ( P ^ { \pi _ { \rho } } ) ^ { \top } d .
$$

Indeed, if $d ( s ^ { \prime } ) > 0 ;$ then $d ( s ^ { \prime } ) \pi _ { \rho } ( a \mid s ^ { \prime } ) = \rho ( s ^ { \prime } , a )$ . If $d ( s ^ { \prime } ) = 0$ , nonnegativity and $\begin{array} { r } { \sum _ { a } \rho ( s ^ { \prime } , a ) = 0 } \end{array}$ imply $\rho ( s ^ { \prime } , a ) = 0 = d ( s ^ { \prime } ) \pi _ { \rho } ( a \mid s ^ { \prime } )$ for every action. Consequently,

$$
\begin{array} { c } { { \displaystyle \sum _ { s ^ { \prime } , a ^ { \prime } } P ( s \mid s ^ { \prime } , a ^ { \prime } ) \rho ( s ^ { \prime } , a ^ { \prime } ) = \sum _ { s ^ { \prime } } d ( s ^ { \prime } ) \sum _ { a ^ { \prime } } \pi _ { \rho } ( a ^ { \prime } \mid s ^ { \prime } ) P ( s \mid s ^ { \prime } , a ^ { \prime } ) } } \\ { { = \displaystyle \sum _ { s ^ { \prime } } P ^ { \pi _ { \rho } } ( s \mid s ^ { \prime } ) d ( s ^ { \prime } ) , } } \end{array}
$$

which proves the vector equation.

Because $P ^ { \pi _ { \rho } }$ is row stochastic, $\| ( P ^ { \pi _ { \rho } } ) ^ { \top } x \| _ { 1 } \leq \| x \|$ for every vector x. Hence

$$
\begin{array} { r } { \| \gamma ( P ^ { \pi _ { \rho } } ) ^ { \top } \| _ { 1  1 } \leq \gamma < 1 , } \end{array}
$$

and the Neumann series

$$
\left[ I - \gamma ( P ^ { \pi _ { \rho } } ) ^ { \top } \right] ^ { - 1 } = \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \left[ ( P ^ { \pi _ { \rho } } ) ^ { \top } \right] ^ { t }
$$

converges. Thus the state-flow equation has the unique solution

$$
d = ( 1 - \gamma ) \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \left[ ( P ^ { \pi _ { \rho } } ) ^ { \top } \right] ^ { t } \mu _ { 0 } ,
$$

which is exactly the normalized discounted state occupancy of $\pi _ { \rho } .$ Multiplying this state occupancy by $\pi _ { \rho } ( \boldsymbol { a } \mid s )$ shows that the state-action occupancy of $\pi _ { \rho }$ is

$$
d ( s ) \pi _ { \rho } ( a \mid s ) = \rho ( s , a )
$$

at every state and action. Therefore every point in $\mathcal { D } ( D )$ is generated by a policy in $\Pi ( D )$ .

For any $\pi \in \Pi ( D )$

$$
\begin{array} { r l } & { ~ \displaystyle \sum _ { s , a } \rho ^ { \pi } ( s , a ) r ( s , a ) = ( 1 - \gamma ) \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \sum _ { s , a } \mathbb { P } \mathbf { r } ( s _ { t } = s , a _ { t } = a ) r ( s , a ) } \\ & { ~ = ( 1 - \gamma ) \displaystyle \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \mathbb { E } ^ { \pi } [ r ( s _ { t } , a _ { t } ) ] } \\ & { ~ = ( 1 - \gamma ) \displaystyle \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \mathbb { E } ^ { \pi } [ R _ { t } ] } \\ & { ~ = ( 1 - \gamma ) V ^ { \pi } ( \mu _ { 0 } ) . } \end{array}
$$

The third equality uses $\mathbb { E } [ R _ { t } \ | \ s _ { t } , a _ { t } ] = r ( s _ { t } , a _ { t } )$ and the tower property. Steps $3$ and 4 give a correspondence between supported policies and feasible occupancies, so maximizing the LP is equivalent to maximizing $V ^ { \pi } ( \mu _ { 0 } )$ over Π(D). Its optimal value is therefore $( 1 - \gamma ) V _ { D } ^ { * } ( \mu _ { 0 } )$

Summing the flow constraints over all states yields

$$
\begin{array} { c l } { { \displaystyle \sum _ { s , a } \rho ( s , a ) = ( 1 - \gamma ) \sum _ { s } \mu _ { 0 } ( s ) } } \\ { { + \displaystyle \gamma \sum _ { s ^ { \prime } , a ^ { \prime } } \rho ( s ^ { \prime } , a ^ { \prime } ) \sum _ { s } P ( s \mid s ^ { \prime } , a ^ { \prime } ) } } \end{array}
$$

$$
= ( 1 - \gamma ) + \gamma \sum _ { s ^ { \prime } , a ^ { \prime } } \rho ( s ^ { \prime } , a ^ { \prime } ) .
$$

Therefore

$$
\sum _ { s , a } \rho ( s , a ) = 1 .
$$

The set $\mathcal { D } ( D )$ is defined by finitely many linear equalities, linear zero constraints, and nonnegativity constraints, so it is closed. The preceding identity and nonnegativity make it bounded. It is nonempty because any deterministic policy supported on $D$ has a feasible occupancy. Hence $\mathcal { D } ( D )$ is compact, and the continuous linear objective attains its maximum.

It is enough to regard $\rho ( s , a )$ for $a \in D ( s )$ as the primal variables. Associate an unrestricted dual variable $v ( s )$ with each flow equality. The coeficient of a primal variable $\rho ( x , a )$ in the equality indexed by s is

$$
1 \{ s = x \} - \gamma P ( s \mid x , a ) .
$$

The corresponding dual inequality is

$$
\sum _ { s } v ( s ) [ \mathbf { 1 } \{ s = x \} - \gamma P ( s \mid x , a ) ] \ge r ( x , a ) ,
$$

which is equivalent to

$$
v ( x ) \geq r ( x , a ) + \gamma \sum _ { s } P ( s \mid x , a ) v ( s ) .
$$

This gives the dual in (54). The primal is feasible and bounded. The dual is feasible: the constant vector

$$
v ( s ) = { \frac { 1 } { 1 - \gamma } }
$$

satisfies, for every admissible action,

$$
r ( s , a ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) v ( s ^ { \prime } ) \leq 1 + \frac { \gamma } { 1 - \gamma } = \frac { 1 } { 1 - \gamma } = v ( s ) .
$$

The finite-dimensional linear-programming strong-duality theorem therefore gives equality of the primal and dual optimal values.

Let $\mathcal { O } _ { D }$ be the set of primal optimizers. It is a nonempty closed subset of the compact set $\mathcal { D } ( D )$ and is therefore compact. Order the $n = | S | | A |$ coordinates as $1 , \ldots , n .$ Set $\mathcal { O } _ { D } ^ { ( 0 ) } = \mathcal { O } _ { D }$ . For $j = 1 , \dots , n _ { \colon }$ , define

$$
m _ { j } = \operatorname* { m i n } _ { \rho \in { \mathcal { O } _ { D } ^ { ( j - 1 ) } } } \rho _ { j } , \qquad { \mathcal { O } _ { D } ^ { ( j ) } } = \{ \rho \in { \mathcal { O } _ { D } ^ { ( j - 1 ) } } : \rho _ { j } = m _ { j } \} .
$$

Each minimum is attained because $\mathcal { O } _ { D } ^ { ( j - 1 ) }$ is nonempty and compact, so each $\mathcal { O } _ { D } ^ { ( j ) }$ is again nonempty and compact. At the last step all coordinates are fixed, so ${ \mathcal { O } } _ { D } ^ { ( n ) }$ contains exactly one vector. This vector is the lexicographically smallest optimal occupancy and is denoted $\rho _ { D } ^ { \mathrm { l e x } }$

Finally, reconstruct a policy from $\rho _ { D } ^ { \mathrm { l e x } }$ , using the first action in $D ( s )$ at every state with zero occupancy. The reconstructed policy has occupancy $\rho _ { D } ^ { \mathrm { l e x } }$ and attains the restricted optimal value. This supplies the unique response convention needed to evaluate the outer objective. □

Proof of Proposition C.2. Fix θ. Because $s$ and A are finite, the collection of score values

$$
\{ g _ { \theta } ( s , a ) : s \in S , a \in A \}
$$

is finite. As τ varies, a raw set changes only when τ crosses one of these values. The fallback action is also fixed by θ and the tie-breaking order. Hence only finitely many complete raw–operational designs $G = ( C _ { \theta , \tau } , D _ { \theta , \tau } )$ can be induced by thresholds.

For any induced design, the lower occupancy LP depends only on D. The lexicographic construction in Theorem C.1 therefore fixes one optimal occupancy and makes both the objective and coverage constraint functions of G alone. The feasible induced designs form a finite set, so the objective attains its minimum whenever this set is nonempty.

It remains to prove nonemptiness. Since every score is finite, choose

$$
\tau _ { \mathrm { a l l } } \geq \operatorname* { m a x } _ { s , a } g _ { \theta } ( s , a ) .
$$

Then $C _ { \theta , \tau _ { \mathrm { a l l } } } ( s ) = D _ { \theta , \tau _ { \mathrm { a l l } } } ( s ) = \mathcal { A }$ for every state. The oracle target is nonempty, so the retention probability equals one. Thus $\tau _ { \mathrm { a l l } }$ is feasible, and a minimizing threshold exists. □

## D.3 Proof of the Quantile Characterization

The argument has two parts. The margin equivalence identifies the feasible thresholds, and nesting of the execution sets establishes size minimality under the same fixed occupancy.

Proof of Proposition C.3. Fix θ and π. For every state $s ,$

$$
\begin{array} { r l } & { C _ { \theta , \tau } ( s ) \cap \varLambda \mathscr { A } _ { \varepsilon } ^ { \ast } ( s ) \neq \emptyset } \\ & { \qquad \Longleftrightarrow \mathrm { t h e r e ~ e x i s t s ~ } a \in \mathcal { A } _ { \varepsilon } ^ { \ast } ( s ) \mathrm { ~ w i t h ~ } g _ { \theta } ( s , a ) \leq \tau } \\ & { \qquad \Longleftrightarrow \underset { a \in \mathcal { A } _ { \varepsilon } ^ { \ast } ( s ) } { \operatorname* { m i n } } g _ { \theta } ( s , a ) \leq \tau } \\ & { \qquad \Longleftrightarrow m _ { \theta } ( s ) \leq \tau . } \end{array}
$$

The minimum is well-defined because $A _ { \varepsilon } ^ { * } ( s )$ is finite and nonempty. Taking probability under $s \sim d ^ { \pi }$ gives

$$
\operatorname* { P r } _ { s \sim d ^ { \pi } } \left( C _ { \theta , \tau } ( s ) \cap \mathcal { A } _ { \varepsilon } ^ { \ast } ( s ) \neq \varnothing \right) = F _ { \theta , \pi } ( \tau ) .
$$

Thus τ satisfies the coverage requirement if and only if

$$
F _ { \theta , \pi } ( \tau ) \geq 1 - \alpha .
$$

Let

$$
q = F _ { \theta , \pi } ^ { - 1 } ( 1 - \alpha ) = \operatorname* { i n f } \{ x : F _ { \theta , \pi } ( x ) \geq 1 - \alpha \} .
$$

Because $s$ is finite, the random variable $m _ { \theta } ( s )$ has finite support under $d ^ { \pi }$ . Hence its CDF is a right-continuous step function and the set

$$
\{ x : F _ { \theta , \pi } ( x ) \geq 1 - \alpha \}
$$

has a smallest element. That element is $q ,$ so $F _ { \theta , \pi } ( q ) \geq 1 - \alpha$ . If $\tau < q$ , then by the definition of the infimum and the fact that $q$ is the smallest feasible support point,

$$
F _ { \theta , \pi } ( \tau ) < 1 - \alpha .
$$

If $\tau \geq q .$ monotonicity of the CDF gives

$$
F _ { \theta , \pi } ( \tau ) \geq F _ { \theta , \pi } ( q ) \geq 1 - \alpha .
$$

Therefore the feasible thresholds are exactly the interval $[ q , \infty )$ , and the smallest feasible threshold is $q =$ $\tau _ { \mathrm { m i n } } ( \theta , \pi )$

It remains to prove the set-size minimality under the same fixed policy. Let $\tau _ { 1 } \leq \tau _ { 2 }$ . The raw sets are nested:

$$
C _ { \theta , \tau _ { 1 } } ( s ) \subseteq C _ { \theta , \tau _ { 2 } } ( s ) \qquad \mathrm { f o r ~ e v e r y ~ } s .
$$

We now verify that the operational sets are also nested.

If $C _ { \theta , \tau _ { 1 } } ( s ) \neq \varnothing$ , then both operational sets equal their raw sets and

$$
\overline { { C } } \theta , \tau _ { 1 } ( s ) = C _ { \theta , \tau _ { 1 } } ( s ) \subseteq C _ { \theta , \tau _ { 2 } } ( s ) = \overline { { C } } \theta , \tau _ { 2 } ( s ) .
$$

If $C _ { \theta , \tau _ { 1 } } ( s ) = \varnothing$ and $C _ { \theta , \tau _ { 2 } } ( s ) = \varnothing$ , then both operational sets equal the same singleton $\{ a _ { \theta } ^ { \mathrm { m i n } } ( s ) \}$ . Finally, suppose $C _ { \theta , \tau _ { 1 } } ( s ) = \varnothing$ but $C _ { \theta , \tau _ { 2 } } ( s ) \neq \emptyset$ . Nonemptiness of the latter implies

$$
\tau _ { 2 } \geq \operatorname* { m i n } _ { a } g _ { \theta } ( s , a ) = g _ { \theta } ( s , a _ { \theta } ^ { \mathrm { m i n } } ( s ) ) ,
$$

so $a _ { \theta } ^ { \mathrm { m i n } } ( s ) \in C _ { \theta , \tau _ { 2 } } ( s )$ . Hence

$$
\overline { { C } } \theta , \tau _ { 1 } ( s ) = \{ a _ { \theta } ^ { \mathrm { m i n } } ( s ) \} \subseteq C _ { \theta , \tau _ { 2 } } ( s ) = \overline { { C } } \theta , \tau _ { 2 } ( s ) .
$$

Thus

$$
| \overline { { C } } _ { \theta , \tau _ { 1 } } ( s ) | \leq | \overline { { C } } _ { \theta , \tau _ { 2 } } ( s ) |
$$

for every state. Averaging with respect to the fixed distribution $d ^ { \pi }$ preserves the inequality. Since every coveragevalid threshold is at least $\tau _ { \mathrm { m i n } } ( \theta , \pi )$ , that threshold minimizes the expected operational-set size among all thresholds evaluated under the same fixed policy. □

## D.4 Proof of Pathwise Calibration

The update cannot move arbitrarily far beyond the realized margin range: outside that range, its direction points back toward it. We first formalize this interval bound and then express the cumulative calibration error in terms of threshold increments.

Proof of Theorem 3.1. Fix an arbitrary horizon $T \geq 1$ and define

$$
\begin{array} { l } { { \underline { { m } } _ { T } = \operatorname* { m i n } \{ \tau _ { 1 } , m _ { 1 } , . . . , m _ { T } \} , } } \\ { { \overline { { m } } _ { T } = \operatorname* { m a x } \{ \tau _ { 1 } , m _ { 1 } , . . . , m _ { T } \} . } } \end{array}
$$

Then $W _ { T } = { \overline { { m } } } _ { T } - \underline { { m } } _ { T }$ . Define

$$
\tau _ { T } ^ { - } = { \underline { { { m _ { T } } } } } - \alpha \eta _ { 1 } , \qquad \tau _ { T } ^ { + } = \overline { { { m } } } _ { T } + ( 1 - \alpha ) \eta _ { 1 } .
$$

We first prove by induction that

$$
\tau _ { t } \in [ \tau _ { T } ^ { - } , \tau _ { T } ^ { + } ] \qquad \mathrm { f o r ~ e v e r y ~ } 1 \leq t \leq T + 1 .\tag{55}
$$

The base case holds because $\tau _ { 1 } \in [ \underline { { m _ { T } } } , \overline { { m } } _ { T } ]$ . Assume the claim holds for some $t \leq T$ . Since $\eta _ { t } \leq \eta _ { 1 }$ , there are three cases.

If $\tau _ { t } < \underline { { m } } _ { T }$ , then $m _ { t } \ge \underline { { m _ { T } } } > \tau _ { t } .$ , so $e _ { t } = 1$ and

$$
\tau _ { t + 1 } = \tau _ { t } + ( 1 - \alpha ) \eta _ { t } .
$$

The update is nonnegative, so $\tau _ { t + 1 } \geq \tau _ { t } \geq \tau _ { T } ^ { - }$ . Also,

$$
\begin{array} { r l } & { \tau _ { t + 1 } < \underline { { m } } _ { T } + ( 1 - \alpha ) \eta _ { t } } \\ & { \qquad \leq \overline { { m } } _ { T } + ( 1 - \alpha ) \eta _ { 1 } = \tau _ { T } ^ { + } . } \end{array}
$$

If $\begin{array} { r } { m _ { T } \leq \tau _ { t } \leq \overline { { m } } _ { T } } \end{array}$ , then $e _ { t } \in \{ 0 , 1 \}$ and

$$
\tau _ { t } - \alpha \eta _ { t } \leq \tau _ { t + 1 } \leq \tau _ { t } + ( 1 - \alpha ) \eta _ { t } .
$$

Using $\eta _ { t } \leq \eta _ { 1 }$ and the bounds on $\tau _ { t }$ yields $\tau _ { T } ^ { - } \leq \tau _ { t + 1 } \leq \tau _ { T } ^ { + }$

If $\tau _ { t } > \overline { { m } } _ { T }$ , then $m _ { t } \le \overline { { m } } _ { T } < \tau _ { t } .$ , so $e _ { t } = 0$ and

$$
\tau _ { t + 1 } = \tau _ { t } - \alpha \eta _ { t } .
$$

The update is nonpositive, so $\tau _ { t + 1 } \leq \tau _ { t } \leq \tau _ { T } ^ { + }$ . Moreover,

$$
\begin{array} { r l } & { \tau _ { t + 1 } > \overline { { m } } _ { T } - \alpha \eta _ { t } } \\ & { \qquad \geq \underline { { m } } _ { T } - \alpha \eta _ { 1 } = \tau _ { T } ^ { - } . } \end{array}
$$

This completes the induction. The invariant interval has width

$$
\begin{array} { r l } { \tau _ { T } ^ { + } - \tau _ { T } ^ { - } = ( \overline { { m } } _ { T } - \underline { { m } } _ { T } ) + \eta _ { 1 } } & { { } } \\ { = W _ { T } + \eta _ { 1 } . } \end{array}\tag{56}
$$

The interval bound now controls the accumulated misses. Rearranging (12) gives

$$
e _ { t } - \alpha = \frac { \tau _ { t + 1 } - \tau _ { t } } { \eta _ { t } } .
$$

Let $c _ { t } = 1 / \eta _ { t }$ . Because $\eta _ { t }$ is positive and nonincreasing, $c _ { t }$ is positive and nondecreasing. Therefore

$$
\begin{array} { l } { S _ { T } = \displaystyle \sum _ { t = 1 } ^ { T } ( e _ { t } - \alpha ) } \\ { \displaystyle \quad = \sum _ { t = 1 } ^ { T } c _ { t } ( \tau _ { t + 1 } - \tau _ { t } ) } \\ { \displaystyle \quad = c _ { T } \tau _ { T + 1 } - c _ { 1 } \tau _ { 1 } - \sum _ { t = 2 } ^ { T } ( c _ { t } - c _ { t - 1 } ) \tau _ { t } . } \end{array}
$$

Define

$$
\omega _ { 1 } = \frac { c _ { 1 } } { c _ { T } } , \qquad \omega _ { t } = \frac { c _ { t } - c _ { t - 1 } } { c _ { T } } \quad ( 2 \leq t \leq T ) .
$$

Each $\omega _ { t } \geq 0$ , and

$$
\sum _ { t = 1 } ^ { T } \omega _ { t } = \frac { c _ { 1 } + \sum _ { t = 2 } ^ { T } ( c _ { t } - c _ { t - 1 } ) } { c _ { T } } = 1 .
$$

Hence

$$
S _ { T } = c _ { T } \left( \tau _ { T + 1 } - \sum _ { t = 1 } ^ { T } \omega _ { t } \tau _ { t } \right) .
$$

The weighted sum is a convex combination of $\tau _ { 1 } , \dots , \tau _ { T }$ . By (55), both this convex combination and $\tau _ { T + 1 }$ lie in $[ \tau _ { T } ^ { - } , \tau _ { T } ^ { + } ]$ . Thus, using (56),

$$
| S _ { T } | \leq c _ { T } ( \tau _ { T } ^ { + } - \tau _ { T } ^ { - } ) = \frac { W _ { T } + \eta _ { 1 } } { \eta _ { T } } .
$$

Dividing by $T$ proves (14). Since the argument used only the update and the realized margin interval, the bound holds for an arbitrary adaptive sequence. The convergence condition and the constant step consequence follow directly from its right hand side. □

## D.5 Proofs for the Value Results

We begin with the counterexample separating retention from selection, then derive the exact value identity. The subsequent proofs bound its filtering and selection terms in turn, before combining them along an adaptive finite session.

Proof of Proposition $C . 4 .$ . Consider one state $s ,$ two actions $a _ { g } , a _ { b }$ , deterministic self-transitions, and rewards

$$
r ( s , a _ { g } ) = 1 , \qquad r ( s , a _ { b } ) = 0 .
$$

The optimal policy always chooses $a _ { g } ,$ , so

$$
V ^ { * } ( s ) = \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } = \frac { 1 } { 1 - \gamma } .
$$

The optimal action values are

$$
\begin{array} { l } { \displaystyle { Q ^ { * } ( s , a _ { g } ) = 1 + \gamma V ^ { * } ( s ) = \frac { 1 } { 1 - \gamma } , } } \\ { \displaystyle { Q ^ { * } ( s , a _ { b } ) = \gamma V ^ { * } ( s ) = \frac { \gamma } { 1 - \gamma } . } } \end{array}
$$

Therefore $A _ { 0 } ^ { * } ( s ) = \{ a _ { g } \}$ . Let $D ( s ) = \{ a _ { g } , a _ { b } \}$ , so the set contains an optimal action with probability one. Let $\pi _ { b }$ always choose $a _ { b }$ . This policy belongs to $\Pi ( D )$ and receives zero reward at every time, hence $V ^ { \pi _ { b } } ( s ) = 0$ . Its value gap is

$$
V ^ { \ast } ( s ) - V ^ { \pi _ { b } } ( s ) = \frac { 1 } { 1 - \gamma } .
$$

Lemma D.2 shows that no value gap can be larger, so the gap is maximal.

Proof of Theorem 3.2. We first verify that both terms are losses, then express their sum as the optimal Bellman residual of the supported policy. Since $D ( s )$ is nonempty and $V ^ { * } ( s ) = \operatorname* { m a x } _ { a \in \mathcal { A } } Q ^ { * } ( s , a )$

$$
\operatorname* { m a x } _ { a \in D ( s ) } Q ^ { * } ( s , a ) \leq V ^ { * } ( s ) ,
$$

so $\Delta _ { D } ( s ) \geq 0$ . Also, an average cannot exceed the maximum of the averaged numbers:

$$
\sum _ { a \in D ( s ) } \pi ( a \mid s ) Q ^ { \ast } ( s , a ) \leq \operatorname* { m a x } _ { a \in D ( s ) } Q ^ { \ast } ( s , a ) ,
$$

so $\zeta _ { \pi , D } ( s ) \geq 0 .$

Define the policy-averaged reward and transition kernel

$$
\begin{array} { c } { { r ^ { \pi } ( s ) = \displaystyle \sum _ { a } \pi ( a \mid s ) r ( s , a ) , } } \\ { { { \cal P } ^ { \pi } ( s ^ { \prime } \mid s ) = \displaystyle \sum _ { a } \pi ( a \mid s ) { \cal P } ( s ^ { \prime } \mid s , a ) . } } \end{array}
$$

For any function $f$ on $s ,$ let

$$
( P ^ { \pi } f ) ( s ) = \sum _ { s ^ { \prime } } P ^ { \pi } ( s ^ { \prime } \mid s ) f ( s ^ { \prime } ) .
$$

To relate these local gaps to the total value loss, define the optimal Bellman residual of $\pi$ by

$$
b _ { \pi } ( s ) = V ^ { * } ( s ) - \sum _ { a } \pi ( a \mid s ) Q ^ { * } ( s , a ) .
$$

Using (51),

$$
\begin{array} { l } { \displaystyle \sum _ { a } \pi ( a \mid s ) Q ^ { * } ( s , a ) } \\ { = \displaystyle \sum _ { a } \pi ( a \mid s ) r ( s , a ) } \\ { + \displaystyle \gamma \displaystyle \sum _ { s ^ { \prime } } \left[ \displaystyle \sum _ { a } \pi ( a \mid s ) P ( s ^ { \prime } \mid s , a ) \right] V ^ { * } ( s ^ { \prime } ) } \\ { = r ^ { \pi } ( s ) + \gamma ( P ^ { \pi } V ^ { * } ) ( s ) . } \end{array}
$$

Lemma D.1 gives $\begin{array} { r } { V ^ { \pi } = r ^ { \pi } + \gamma P ^ { \pi } V ^ { \pi } } \end{array}$ . Subtracting this identity from $V ^ { * }$ yields, state by state,

$$
\begin{array} { r l } & { V ^ { * } - V ^ { \pi } = V ^ { * } - r ^ { \pi } - \gamma P ^ { \pi } V ^ { \pi } } \\ & { \qquad = [ V ^ { * } - r ^ { \pi } - \gamma P ^ { \pi } V ^ { * } ] + \gamma P ^ { \pi } ( V ^ { * } - V ^ { \pi } ) } \\ & { \qquad = b _ { \pi } + \gamma P ^ { \pi } ( V ^ { * } - V ^ { \pi } ) . } \end{array}
$$

Iterating this equality N times gives

$$
\begin{array} { r l } {  { V ^ { * } - V ^ { \pi } = \sum _ { t = 0 } ^ { N - 1 } \gamma ^ { t } ( P ^ { \pi } ) ^ { t } b _ { \pi } } } \\ & { + \gamma ^ { N } ( P ^ { \pi } ) ^ { N } ( V ^ { * } - V ^ { \pi } ) . } \end{array}
$$

A Markov operator is nonexpansive in the sup norm. Lemma D.2 therefore gives

$$
\left\| \gamma ^ { N } ( P ^ { \pi } ) ^ { N } ( V ^ { * } - V ^ { \pi } ) \right\| _ { \infty } \leq \frac { \gamma ^ { N } } { 1 - \gamma } \longrightarrow 0 .
$$

Consequently,

$$
V ^ { * } - V ^ { \pi } = \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } ( P ^ { \pi } ) ^ { t } b _ { \pi } .
$$

Average both sides over $s _ { 0 } \sim \mu _ { 0 }$ . The definition (3) implies

$$
\begin{array} { l l l } { { \displaystyle V ^ { * } ( \mu _ { 0 } ) - V ^ { \pi } ( \mu _ { 0 } ) = \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \mathbb { E } ^ { \pi } [ b _ { \pi } ( s _ { t } ) ] } } \\ { { \displaystyle \qquad = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d ^ { \pi } } [ b _ { \pi } ( s ) ] . } } \end{array}
$$

Because $\pi \in \Pi ( D )$ 2

$$
\begin{array} { r l } & { b _ { \pi } ( s ) = V ^ { * } ( s ) - \displaystyle \sum _ { a \in D ( s ) } \pi ( a \mid s ) Q ^ { * } ( s , a ) } \\ & { \qquad = V ^ { * } ( s ) - \displaystyle \operatorname* { m a x } _ { a \in D ( s ) } Q ^ { * } ( s , a ) } \\ & { \qquad + \displaystyle \operatorname* { m a x } _ { a \in D ( s ) } Q ^ { * } ( s , a ) - \displaystyle \sum _ { a \in D ( s ) } \pi ( a \mid s ) Q ^ { * } ( s , a ) } \\ & { \qquad = \Delta _ { D } ( s ) + \zeta _ { \pi , D } ( s ) . } \end{array}
$$

Substitution proves (18). The decomposition is exact because it splits the same Bellman residual, rather than bounding filtering and selection through separate trajectory comparisons. □

Proof of Corollary C.5. Define the hit set

$$
H = \{ s : C ( s ) \cap \mathcal { A } _ { \varepsilon } ^ { * } ( s ) \neq \emptyset \}
$$

and its miss probability under the policy occupancy:

$$
\alpha _ { \pi } = \operatorname* { P r } _ { s \sim d ^ { \pi } } ( s \notin H ) .
$$

Assumption (31) gives

$$
\alpha _ { \pi } \leq \alpha .
$$

Fix a state $s \in H$ . By definition of H, there exists at least one action

$$
a _ { H } ( s ) \in C ( s ) \cap { \mathcal { A } } _ { \varepsilon } ^ { * } ( s ) .
$$

Since $C ( s ) \subseteq D ( s )$

$$
a _ { H } ( s ) \in D ( s ) .
$$

Therefore

$$
\operatorname* { m a x } _ { a \in D ( s ) } Q ^ { * } ( s , a ) \ge Q ^ { * } ( s , a _ { H } ( s ) )
$$

$$
\geq V ^ { \ast } ( s ) - \varepsilon ,
$$

where the second inequality follows from $a _ { H } ( s ) \in \mathcal { A } _ { \varepsilon } ^ { * } ( s )$ . Rearranging gives

$$
\Delta _ { D } ( s ) = V ^ { * } ( s ) - \operatorname* { m a x } _ { a \in D ( s ) } Q ^ { * } ( s , a ) \leq \varepsilon .
$$

By definition of $\Delta _ { \mathrm { m a x } }$ , every nonempty set $D ( s )$ also satisfies

$$
0 \leq \Delta _ { D } ( s ) \leq \Delta _ { \operatorname* { m a x } } .
$$

Hence, on the hit set,

$$
\Delta _ { D } ( s ) \leq \operatorname* { m i n } \{ \varepsilon , \Delta _ { \operatorname* { m a x } } \} = \varepsilon _ { 0 } ,
$$

while on the miss set,

$$
\Delta _ { D } ( s ) \leq \Delta _ { \operatorname* { m a x } } .
$$

Split the occupancy expectation over H and its complement:

$$
\begin{array} { r l } & { \mathbb { E } _ { s \sim d ^ { \pi } } [ \Delta _ { D } ( s ) ] = \mathbb { E } [ \Delta _ { D } ( s ) \mathbf { 1 } \{ s \in H \} ] } \\ & { \phantom { \Delta _ { \theta } } + \mathbb { E } [ \Delta _ { D } ( s ) \mathbf { 1 } \{ s \notin H \} ] } \\ & { \phantom { \Delta _ { \theta } } \le \varepsilon _ { 0 } \operatorname* { P r } ( s \in H ) + \Delta _ { \operatorname* { m a x } } \operatorname* { P r } ( s \notin H ) } \\ & { \phantom { \Delta _ { \theta } } = ( 1 - \alpha _ { \pi } ) \varepsilon _ { 0 } + \alpha _ { \pi } \Delta _ { \operatorname* { m a x } } } \\ & { \phantom { \Delta _ { \theta } } = \varepsilon _ { 0 } + \alpha _ { \pi } ( \Delta _ { \operatorname* { m a x } } - \varepsilon _ { 0 } ) . } \end{array}
$$

Since $\Delta _ { \mathrm { m a x } } - \varepsilon _ { 0 } \geq 0$ and $\alpha _ { \pi } \leq \alpha$

$$
\begin{array} { r l } & { \mathbb { E } _ { d ^ { \pi } } [ \Delta _ { D } ( s ) ] \leq \varepsilon _ { 0 } + \alpha ( \Delta _ { \operatorname* { m a x } } - \varepsilon _ { 0 } ) } \\ & { \qquad = ( 1 - \alpha ) \varepsilon _ { 0 } + \alpha \Delta _ { \operatorname* { m a x } } . } \end{array}
$$

The exact decomposition in Theorem 3.2 gives

$$
V ^ { * } ( \mu _ { 0 } ) - V ^ { \pi } ( \mu _ { 0 } ) = \frac { \mathbb { E } _ { d ^ { \pi } } [ \Delta _ { D } ( s ) ] + \overline { { \zeta } } _ { \pi , D } } { 1 - \gamma } .
$$

Substituting the preceding bound proves (32).

Proof of Corollary C.6. Feasibility of $( \theta , \tau )$ in Problem (11) gives

$$
\operatorname* { P r } _ { s \sim d ^ { \pi _ { \theta , \tau } ^ { * } } } \left( C _ { \theta , \tau } ( s ) \cap \mathcal { A } _ { \varepsilon } ^ { * } ( s ) \neq \varnothing \right) \ge 1 - \alpha .
$$

By construction,

$$
C _ { \theta , \tau } ( s ) \subseteq D _ { \theta , \tau } ( s )
$$

and $D _ { \theta , \tau } ( s )$ is nonempty for every state. The lower-level policy satisfies

$$
\pi _ { \theta , \tau } ^ { * } \in \Pi ( D _ { \theta , \tau } ) .
$$

Thus every assumption of Corollary C.5 holds after setting

$$
C = C _ { \boldsymbol { \theta } , \tau } , \qquad D = D _ { \boldsymbol { \theta } , \tau } , \qquad \pi = \pi _ { \boldsymbol { \theta } , \tau } ^ { * } .
$$

The stated bound follows directly.

The occupancy bound concerns retention averaged under a specified policy. Under the stronger condition that every state retains a target action, we can instead compare the restricted and unrestricted optimal values at every initial state.

Corollary D.3 (Pointwise retention). $I f D ( s ) \cap A _ { \varepsilon } ^ { * } ( s ) \neq \varnothing$ for every state, then

$$
0 \leq V ^ { * } ( s ) - V _ { D } ^ { * } ( s ) \leq \frac { \varepsilon _ { 0 } } { 1 - \gamma } \qquad \forall s .\tag{57}
$$

Proof of Corollary D.3. For every state s, choose

$$
a _ { Q } ( s ) \in \arg \operatorname* { m a x } _ { a \in D ( s ) } Q ^ { * } ( s , a )
$$

using a fixed tie-breaking rule, and let $\pi _ { Q }$ be the deterministic policy that selects $a _ { Q } ( s )$ . Because $a _ { Q } ( s ) \in D ( s )$

$$
\pi _ { Q } \in \Pi ( D ) .
$$

The pointwise retention assumption gives an action $a _ { H } ( s ) \in D ( s ) \cap \mathcal { A } _ { \varepsilon } ^ { * } ( s )$ at every state. Therefore

$$
\begin{array} { r l } { \underset { a \in D ( s ) } { \operatorname* { m a x } } Q ^ { * } ( s , a ) \geq Q ^ { * } ( s , a _ { H } ( s ) ) } & { } \\ & { } \\ & { \geq V ^ { * } ( s ) - \varepsilon . } \end{array}
$$

Together with $\Delta _ { D } ( s ) \leq \Delta _ { \mathrm { m a x } }$ , this yields

$$
\Delta _ { D } ( s ) = V ^ { * } ( s ) - \operatorname* { m a x } _ { a \in D ( s ) } Q ^ { * } ( s , a ) \leq \varepsilon _ { 0 } \qquad { \mathrm { f o r ~ e v e r y ~ } } s .
$$

For the policy $\pi _ { Q }$ , the optimal Bellman residual is

$$
\begin{array} { r l } & { b _ { \pi _ { Q } } ( s ) = V ^ { * } ( s ) - \displaystyle \sum _ { a } \pi _ { Q } ( a \mid s ) Q ^ { * } ( s , a ) } \\ & { \quad \quad \quad = V ^ { * } ( s ) - Q ^ { * } ( s , a _ { Q } ( s ) ) } \\ & { \quad \quad = \Delta _ { D } ( s ) } \\ & { \quad \quad \quad \leq { \varepsilon _ { 0 } } . } \end{array}
$$

The residual identity derived in the proof of Theorem 3.2 gives

$$
V ^ { * } - V ^ { \pi _ { Q } } = b _ { \pi _ { Q } } + \gamma P ^ { \pi _ { Q } } ( V ^ { * } - V ^ { \pi _ { Q } } ) .
$$

Taking sup norms, using $\| b _ { \pi _ { Q } } \| _ { \infty } \leq \varepsilon _ { 0 }$ , and using that a Markov operator is nonexpansive in the sup norm, we obtain

$$
\begin{array} { r } { \| V ^ { * } - V ^ { \pi _ { Q } } \| _ { \infty } \leq \varepsilon _ { 0 } + \gamma \| V ^ { * } - V ^ { \pi _ { Q } } \| _ { \infty } . } \end{array}
$$

Rearranging gives

$$
\| V ^ { * } - V ^ { \pi _ { Q } } \| _ { \infty } \leq \frac { \varepsilon _ { 0 } } { 1 - \gamma } .
$$

By Theorem C.1, the restricted optimal value dominates every supported policy componentwise:

$$
V _ { D } ^ { * } \geq V ^ { \pi _ { Q } } .
$$

Also $V _ { D } ^ { * } \leq V ^ { * }$ because the restricted policy class is a subset of the unrestricted class. Hence, for every state,

$$
0 \leq V ^ { \ast } ( s ) - V _ { D } ^ { \ast } ( s ) \leq V ^ { \ast } ( s ) - V ^ { \pi _ { Q } } ( s ) \leq \frac { \varepsilon _ { 0 } } { 1 - \gamma } .
$$

This proves (57).

We now replace oracle retention by proxy retention. The key step is to compare any proxy target action with an oracle maximizer through the pairwise fidelity condition.

Proof of Proposition C.8. Fix a state s, an action $a \in \widetilde { \mathcal { A } } _ { \varepsilon } ( s )$ , and $a ^ { \ast } ( s ) \in$ arg max $Q ^ { * } ( s , b )$ . Proxy nearoptimality implies

$$
\begin{array} { l l l } { \widetilde { Q } ( s , a ^ { * } ( s ) ) - \widetilde { Q } ( s , a ) \le \displaystyle \operatorname* { m a x } _ { b } \widetilde { Q } ( s , b ) - \widetilde { Q } ( s , a ) } \\ { \le \varepsilon . } \end{array}
$$

Apply Assumption C.7 to the pair $( a ^ { * } ( s ) , a )$ :

$$
\begin{array} { r l } & { V ^ { * } ( s ) - Q ^ { * } ( s , a ) = Q ^ { * } ( s , a ^ { * } ( s ) ) - Q ^ { * } ( s , a ) } \\ & { \phantom { \quad \quad } \leq \lambda [ \widetilde { Q } ( s , a ^ { * } ( s ) ) - \widetilde { Q } ( s , a ) ] + \delta _ { \mathrm { p r o x } } } \\ & { \phantom { \quad \quad } \leq \lambda \varepsilon + \delta _ { \mathrm { p r o x } } . } \end{array}
$$

Thus $a \in \mathcal { A } _ { \lambda \varepsilon + \delta _ { \mathrm { p r o x } } } ^ { * } ( s )$ , proving (35). If a set $C ( s )$ intersects the proxy target, an action in that intersection also belongs to the enlarged oracle target. This proves the stated hit-event implication under any state distribution. □

Proof of Proposition C.9. The optimal Bellman identity gives, for every state and action pair,

$$
\begin{array} { l } { { \displaystyle Q ^ { * } ( s , a ) - Q ^ { * } ( s , b ) - [ r ( s , a ) - r ( s , b ) ] } } \\ { { \displaystyle ~ = \gamma \sum _ { s ^ { \prime } } [ P ( s ^ { \prime } \mid s , a ) - P ( s ^ { \prime } \mid s , b ) ] V ^ { * } ( s ^ { \prime } ) . } } \end{array}
$$

Taking absolute values and then the maximum proves the pairwise condition with $\lambda = 1$ and $\delta _ { \mathrm { p r o x } } = \delta _ { \mathrm { d y n } }$

For a learned response proxy ${ \widehat { p } } ,$ add and subtract the immediate-reward diference:

$$
\begin{array} { r l } & { | Q ^ { * } ( s , a ) - Q ^ { * } ( s , b ) - [ \widehat { p } ( s , a ) - \widehat { p } ( s , b ) ] | } \\ & { \quad \leq | Q ^ { * } ( s , a ) - Q ^ { * } ( s , b ) - [ r ( s , a ) - r ( s , b ) ] | } \\ & { \quad \quad + | r ( s , a ) - \widehat { p } ( s , a ) | + | r ( s , b ) - \widehat { p } ( s , b ) | } \\ & { \quad \leq \delta _ { \mathrm { d y n } } + 2 \delta _ { \mathrm { r e s p } } . } \end{array}
$$

This proves the stated extension.

By Lemma D.2, the range of $V ^ { * }$ is at most $( 1 - \gamma ) ^ { - 1 }$ . For any two probability distributions $P _ { 1 } , P _ { 2 }$ and any function f with range at most $B _ { : }$

$$
| \mathbb { E } _ { P _ { 1 } } f - \mathbb { E } _ { P _ { 2 } } f | \le B \mathrm { ~ T V } ( P _ { 1 } , P _ { 2 } ) .
$$

Apply this inequality with $f = V ^ { * }$ and $B = ( 1 - \gamma ) ^ { - 1 }$ , then maximize over $( s , a , b )$ and multiply by $\gamma$ to obtain (37). □

Proxy fidelity controls the filtering term. The next consequence controls the selection term by transferring an approximate critic maximization guarantee to the true optimal action values.

Corollary D.4 (Approximate within-set selection). Assume $D ( s ) \neq \emptyset f o r$ every state and let $\pi \in \Pi ( D )$ . Suppose $\lVert \widehat { Q } - Q ^ { * } \rVert _ { \infty } \leq \delta _ { \mathrm { c r i t } }$ and the within-set condition (39) holds. Define $\overline { { \delta } } _ { \mathrm { s e l } } = \mathbb { E } _ { s \sim d ^ { \pi } } [ \delta _ { \mathrm { s e l } } ( s ) ]$ . Then

$$
\zeta _ { \pi , D } ( s ) \leq 2 \delta _ { \mathrm { c r i t } } + \delta _ { \mathrm { s e l } } ( s ) .\tag{58}
$$

If (31) also holds, then

$$
V ^ { \ast } ( \mu _ { 0 } ) - V ^ { \pi } ( \mu _ { 0 } ) \leq \frac { ( 1 - \alpha ) \varepsilon _ { 0 } + \alpha \Delta _ { \operatorname* { m a x } } + 2 \delta _ { \operatorname { c r i t } } + \bar { \delta } _ { \operatorname { s e l } } } { 1 - \gamma } .\tag{59}
$$

Proof of Corollary $D . 4 .$ Fix a state s and abbreviate $D _ { s } = D ( s )$ . The uniform critic error implies, for every a,

$$
Q ^ { * } ( s , a ) \leq \widehat { Q } ( s , a ) + \delta _ { \mathrm { c r i t } }
$$

and

$$
Q ^ { * } ( s , a ) \geq \widehat { Q } ( s , a ) - \delta _ { \mathrm { c r i t } } .
$$

Taking a maximum in the first inequality gives

$$
\operatorname* { m a x } _ { a \in D _ { s } } Q ^ { * } ( s , a ) \leq \operatorname* { m a x } _ { a \in D _ { s } } \widehat { Q } ( s , a ) + \delta _ { \mathrm { c r i t } } .
$$

Averaging the second inequality under $\pi ( \cdot \mid s )$ gives

$$
\begin{array} { r l r } & { } & { \displaystyle \sum _ { a \in D _ { s } } \pi ( a \mid s ) Q ^ { * } ( s , a ) \geq \sum _ { a \in D _ { s } } \pi ( a \mid s ) [ \widehat { Q } ( s , a ) - \delta _ { \mathrm { c r i t } } ] } \\ & { } & { \quad \quad = \displaystyle \sum _ { a \in D _ { s } } \pi ( a \mid s ) \widehat { Q } ( s , a ) - \delta _ { \mathrm { c r i t } } , } \end{array}
$$

because $\textstyle \sum _ { a \in D _ { s } } \pi ( a \mid s ) = 1$ . Subtracting the second bound from the first yields

$$
\begin{array} { r l } & { \zeta _ { \pi , D } ( s ) = \displaystyle \operatorname* { m a x } _ { a \in D _ { s } } Q ^ { * } ( s , a ) - \sum _ { a \in D _ { s } } \pi ( a \mid s ) Q ^ { * } ( s , a ) } \\ & { \qquad \le \displaystyle \operatorname* { m a x } _ { a \in D _ { s } } \widehat { Q } ( s , a ) - \sum _ { a \in D _ { s } } \pi ( a \mid s ) \widehat { Q } ( s , a ) + 2 \delta _ { \mathrm { c r i t } } } \\ & { \qquad \le \delta _ { \mathrm { s e l } } ( s ) + 2 \delta _ { \mathrm { c r i t } } . } \end{array}
$$

This proves (58).

Average the statewise inequality under $d ^ { \pi }$ :

$$
\begin{array} { r l } & { \overline { { \zeta } } _ { \pi , D } = \mathbb { E } _ { s \sim d ^ { \pi } } [ \zeta _ { \pi , D } ( s ) ] } \\ & { \qquad \leq 2 \delta _ { \mathrm { c r i t } } + \mathbb { E } _ { s \sim d ^ { \pi } } [ \delta _ { \mathrm { s e l } } ( s ) ] } \\ & { \qquad = 2 \delta _ { \mathrm { c r i t } } + \overline { { \delta } } _ { \mathrm { s e l } } . } \end{array}
$$

Substituting this bound into Corollary C.5 proves (59).

The combined bound now follows by applying these two transfers in sequence: proxy hits give an enlarged oracle target, and critic and selector errors bound the remaining selection loss.

Proof of Corollary C.10. Proposition C.8 gives, for every state,

$$
\widetilde { \mathcal { A } } _ { \varepsilon } ( s ) \subseteq \mathcal { A } _ { \lambda \varepsilon + \delta _ { \mathrm { p r o x } } } ^ { * } ( s ) .
$$

Therefore the proxy coverage assumption (38) implies

$$
\operatorname* { P r } _ { s \sim d ^ { \pi } } \Big ( C ( s ) \cap \mathcal { A } _ { \lambda \varepsilon + \delta _ { \operatorname { p r o x } } } ^ { \ast } ( s ) \neq \varnothing \Big ) \ge 1 - \alpha .
$$

Apply Corollary C.5 with the tolerance

$$
\varepsilon ^ { \prime } = \lambda \varepsilon + \delta _ { \mathrm { p r o x } } .
$$

The truncated hit-state loss in that corollary is

$$
\operatorname* { m i n } \{ \varepsilon ^ { \prime } , \Delta _ { \mathrm { m a x } } \} = \varepsilon _ { \mathrm { r e c } } .
$$

Thus

$$
V ^ { \ast } ( \mu _ { 0 } ) - V ^ { \pi } ( \mu _ { 0 } ) \leq \frac { ( 1 - \alpha ) \varepsilon _ { \mathrm { r e c } } + \alpha \Delta _ { \mathrm { m a x } } + \overline { { \zeta } } _ { \pi , D } } { 1 - \gamma } .
$$

Corollary D.4 gives

$$
\begin{array} { r } { \bar { \zeta } _ { \pi , D } \leq 2 \delta _ { \mathrm { c r i t } } + \bar { \delta } _ { \mathrm { s e l } } . } \end{array}
$$

Substitution proves (41).

For patience, we use the same retention and selection distinction at a single interaction. The collection response is controlled by event inclusion, whereas the response of a selected item also includes the choice error.

Proof of Proposition C.11. Let

$$
H _ { p } = \left\{ s : D ( s ) \mathrm { ~ c o n t a i n s ~ a n ~ a c t i o n ~ } a \mathrm { ~ w i t h ~ } p ( s , a ) \ge p ^ { * } ( s ) - \varepsilon \right\} .
$$

Let $\alpha _ { p } = \operatorname* { P r } _ { s \sim \nu } ( s \notin H _ { p } ) \leq \alpha$

For every action $a \in D ( s )$ , the event $\{ Y _ { a } = 1 \}$ is contained in the union defining $p _ { D } ^ { \mathrm { s e t } } ( s )$ . Hence

$$
p _ { D } ^ { \mathrm { s e t } } ( s ) \geq \operatorname* { m a x } _ { a \in D ( s ) } p ( s , a )
$$

without any independence assumption. On $H _ { p } ,$ , this gives

$$
p ^ { * } ( s ) - p _ { D } ^ { \mathrm { s e t } } ( s ) \leq \varepsilon _ { p } .
$$

Outside $H _ { p } ,$ , the same diference is at most one. Therefore

$$
\begin{array} { r l } & { \mathbb { E } _ { \boldsymbol \nu } [ 1 - p _ { D } ^ { \mathrm { s e t } } ( s ) ] = \mathbb { E } _ { \boldsymbol \nu } [ 1 - p ^ { * } ( s ) ] + \mathbb { E } _ { \boldsymbol \nu } [ p ^ { * } ( s ) - p _ { D } ^ { \mathrm { s e t } } ( s ) ] } \\ & { \qquad \leq \mathbb { E } _ { \boldsymbol \nu } [ 1 - p ^ { * } ( s ) ] + ( 1 - \alpha _ { p } ) \varepsilon _ { p } + \alpha _ { p } } \\ & { \qquad \leq \mathbb { E } _ { \boldsymbol \nu } [ 1 - p ^ { * } ( s ) ] + ( 1 - \alpha ) \varepsilon _ { p } + \alpha , } \end{array}
$$

where the last inequality uses $\varepsilon _ { p } \leq 1$

For the pre-response item selector,

$$
p ^ { * } ( s ) - p _ { D , \kappa } ^ { \mathrm { i t e m } } ( s ) = p ^ { * } ( s ) - \operatorname* { m a x } _ { a \in D ( s ) } p ( s , a ) + \delta _ { \mathrm { s e l } } ^ { p } ( s ) .
$$

The first term is at most $\varepsilon _ { p }$ on $H _ { p }$ and at most one outside $H _ { p }$ . Taking expectations gives

$$
\begin{array} { r } { \mathbb { E } _ { \nu } [ p ^ { * } ( s ) - p _ { D , \kappa } ^ { \mathrm { i t e m } } ( s ) ] \leq ( 1 - \alpha ) \varepsilon _ { p } + \alpha + \overline { { \delta } } _ { \mathrm { s e l } } ^ { p } , } \end{array}
$$

which proves (43).

For either mechanism, denote its positive-response probability by $p _ { \mathrm { m e c h } } ( s )$ . Its expected patience decrease is

$$
c _ { + } p _ { \mathrm { m e c h } } ( s ) + c _ { - } [ 1 - p _ { \mathrm { m e c h } } ( s ) ] = c _ { + } + ( c _ { - } - c _ { + } ) [ 1 - p _ { \mathrm { m e c h } } ( s ) ] .
$$

The best single item replaces $p _ { \mathrm { m e c h } } ( s )$ by $p ^ { * } ( s )$ . Subtracting and applying the corresponding no-response bound proves (44) and (45). □

Proof of Theorem 3.3. The proof combines a finite horizon performance identity with separate bounds for filtering and selection. Unlike the stationary argument, it conditions on the information available at each stage, so the policy and critic may change during the session.

By the finite-horizon Bellman equation and the controlled Markov property,

$$
\begin{array} { r l } & { ~ Q _ { h } ^ { * } ( s _ { h } , a _ { h } ) = r _ { h } ( s _ { h } , a _ { h } ) + \mathbb { E } \big [ V _ { h + 1 } ^ { * } ( s _ { h + 1 } ) \mid \mathcal { H } _ { h } , a _ { h } \big ] , } \\ & { \mathbb { E } [ R _ { h } \mid \mathcal { H } _ { h } , a _ { h } ] = r _ { h } ( s _ { h } , a _ { h } ) . } \end{array}
$$

The tower property therefore gives

$$
\mathbb { E } [ V _ { h } ^ { * } ( s _ { h } ) - Q _ { h } ^ { * } ( s _ { h } , a _ { h } ) ] = \mathbb { E } \big [ V _ { h } ^ { * } ( s _ { h } ) - R _ { h } - V _ { h + 1 } ^ { * } ( s _ { h + 1 } ) \big ] .
$$

Summing from $h = 1$ to H, using $s _ { 1 } \sim \mu _ { 0 }$ and $V _ { H + 1 } ^ { * } = 0$ , yields

$$
V _ { 1 } ^ { * } ( \mu _ { 0 } ) - V _ { 1 } ^ { \mathrm { R L C P } } ( \mu _ { 0 } ) = \sum _ { h = 1 } ^ { H } \mathbb { E } [ V _ { h } ^ { * } ( s _ { h } ) - Q _ { h } ^ { * } ( s _ { h } , a _ { h } ) ] .\tag{60}
$$

For each stage, split the optimal action gap into the loss from the execution set and the loss from the selected action:

$$
\Delta _ { h } = V _ { h } ^ { * } ( s _ { h } ) - \operatorname* { m a x } _ { a \in D _ { h } } Q _ { h } ^ { * } ( s _ { h } , a ) ,
$$

$$
\zeta _ { h } = \operatorname* { m a x } _ { a \in D _ { h } } Q _ { h } ^ { * } ( s _ { h } , a ) - Q _ { h } ^ { * } ( s _ { h } , a _ { h } ) .
$$

Because $D _ { h }$ is nonempty and $a _ { h } \in D _ { h }$ almost surely, both gaps are nonnegative, $\Delta _ { h } \leq \Delta _ { H } ^ { \mathrm { m a x } }$ , and

$$
V _ { h } ^ { * } ( s _ { h } ) - Q _ { h } ^ { * } ( s _ { h } , a _ { h } ) = \Delta _ { h } + \zeta _ { h } .
$$

Suppose first that $e _ { h } ^ { K } = 0$ . Choose $b _ { h } \in C _ { h } ^ { K } \cap \widetilde { { \mathcal A } } _ { \varepsilon , h } ( s _ { h } )$ and $a _ { h } ^ { * } \in \arg \operatorname* { m a x } _ { a \in \mathcal { A } } Q _ { h } ^ { * } \big ( s _ { h } , a \big )$ . Membership of $b _ { h }$ in the proxy target implies

$$
\widetilde { Q } _ { h } ( s _ { h } , a _ { h } ^ { * } ) - \widetilde { Q } _ { h } ( s _ { h } , b _ { h } ) \leq \operatorname* { m a x } _ { a \in \mathcal { A } } \widetilde { Q } _ { h } ( s _ { h } , a ) - \widetilde { Q } _ { h } ( s _ { h } , b _ { h } ) \leq \varepsilon .
$$

Applying (47) to $( a _ { h } ^ { * } , b _ { h } )$ gives

$$
\begin{array} { r l } & { V _ { h } ^ { * } ( s _ { h } ) - Q _ { h } ^ { * } ( s _ { h } , b _ { h } ) = Q _ { h } ^ { * } ( s _ { h } , a _ { h } ^ { * } ) - Q _ { h } ^ { * } ( s _ { h } , b _ { h } ) } \\ & { \phantom { \frac { 1 } { 1 } } \leq \lambda \big ( \widetilde { Q } _ { h } ( s _ { h } , a _ { h } ^ { * } ) - \widetilde { Q } _ { h } ( s _ { h } , b _ { h } ) \big ) + \delta _ { \mathrm { p r o x } } } \\ & { \phantom { \frac { 1 } { 1 } } \leq \lambda \varepsilon + \delta _ { \mathrm { p r o x } } . } \end{array}
$$

Since $b _ { h } \in C _ { h } ^ { K } \subseteq D _ { h }$ , this proves $\Delta _ { h } \leq \varepsilon _ { H }$ on $\{ e _ { h } ^ { K } = 0 \}$ . On $\{ e _ { h } ^ { K } = 1 \}$ , the universal bound $\Delta _ { h } \leq \Delta _ { H } ^ { \mathrm { m a x } }$ applies. Hence, pathwise,

$$
\begin{array} { r } { \Delta _ { h } \leq \varepsilon _ { H } + ( \Delta _ { H } ^ { \operatorname* { m a x } } - \varepsilon _ { H } ) e _ { h } ^ { K } . } \end{array}\tag{61}
$$

By definition, a capped miss is either a raw miss or a raw hit removed by the cap, and these events are disjoint. Thus

$$
e _ { h } ^ { K } = \widetilde { e } _ { h } + h _ { h } ^ { K } .
$$

The recursion (46) has exactly the form required by Theorem 3.1. Since the step sizes are positive and nonincreasing and $\widetilde { W } _ { H } \leq W$ , that theorem gives, almost surely,

$$
\frac { 1 } { H } \sum _ { h = 1 } ^ { H } \widetilde { e } _ { h } \le \alpha + \frac { W + \eta _ { 1 } } { \eta _ { H } H } .
$$

Consequently,

$$
\frac { 1 } { H } \sum _ { h = 1 } ^ { H } e _ { h } ^ { K } \le \alpha + \frac { W + \eta _ { 1 } } { \eta _ { H } H } + \frac { 1 } { H } \sum _ { h = 1 } ^ { H } h _ { h } ^ { K } .
$$

Average (61) over stages and then take expectations. This yields

$$
\begin{array} { r l r } {  { \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \mathbb { E } [ \Delta _ { h } ] \le ( 1 - \alpha ) \varepsilon _ { H } + \alpha \Delta _ { H } ^ { \operatorname* { m a x } } } } \\ & { } & \\ & { } & { \qquad + ( \Delta _ { H } ^ { \operatorname* { m a x } } - \varepsilon _ { H } ) ( \displaystyle \frac { W + \eta _ { 1 } } { \eta _ { H } H } + \overline { { h } } _ { H } ^ { K } ) . } \end{array}\tag{62}
$$

The remaining term is the gap between the best retained action and the action actually selected. Since $D _ { h }$ and ${ \widehat { Q } } _ { h }$ are $\mathcal { H } _ { h }$ measurable, the critic condition implies

$$
\begin{array} { r l } & { \underset { a \in D _ { h } } { \operatorname* { m a x } } Q _ { h } ^ { * } ( s _ { h } , a ) \leq \underset { a \in D _ { h } } { \operatorname* { m a x } } \widehat { Q } _ { h } ( s _ { h } , a ) + \delta _ { \mathrm { c r i t } } , } \\ & { \mathbb { E } \big [ Q _ { h } ^ { * } ( s _ { h } , a _ { h } ) \mid \mathcal { H } _ { h } \big ] \geq \mathbb { E } \Big [ \widehat { Q } _ { h } ( s _ { h } , a _ { h } ) \mid \mathcal { H } _ { h } \Big ] - \delta _ { \mathrm { c r i t } } . } \end{array}
$$

Subtracting the second inequality from the first and using (49) gives

$$
\mathbb { E } [ \zeta _ { h } \mid \mathcal { H } _ { h } ] \le 2 \delta _ { \mathrm { c r i t } } + \delta _ { \mathrm { s e l } , h } .
$$

The assumed integrability permits taking expectations. Averaging over stages gives

$$
\frac { 1 } { H } \sum _ { h = 1 } ^ { H } \mathbb { E } [ \zeta _ { h } ] \le 2 \delta _ { \mathrm { c r i t } } + \overline { { \delta } } _ { \mathrm { s e l } , H } .\tag{63}
$$

Finally, the filtering and selection bounds control the two components of each gap in (60). Substituting (62) and (63), using $\Delta _ { h } + \zeta _ { h }$ , and multiplying by H gives (19). No step requires the policy, critic, or threshold to converge; the argument uses their stated stagewise information and error conditions. □

## D.6 Proof of Greedy Preservation and Conditional Recovery

We first show that thresholding, capping, and fallback all preserve the same critic maximizer. We then use that preservation to express policy loss through pairwise critic error and the remaining selection gap.

Proof of Proposition C.13. Fix t and $s ,$ and let $\widehat { \boldsymbol { a } } _ { t } ( \boldsymbol { s } )$ be the first maximizer of $\widehat { Q } _ { t } ( s , \cdot )$ under the fixed action order. Every score $g _ { t } { \big ( } s , a { \big ) }$ is nonnegative, and $g _ { t } ( s , \widehat { a } _ { t } ( s ) ) = 0 . \ \mathrm { { I f } } \ \tau _ { t } < 0$ , the raw set is empty and the minimum score fallback is exactly $\widehat { \boldsymbol { a } } _ { t } ( \boldsymbol { s } )$ . If $\tau _ { t } \geq 0$ , the raw set contains every critic maximizer. Its first action in increasing score order is $\widehat { \boldsymbol { a } } _ { t } ( \boldsymbol { s } )$ , so every cap with $K _ { t } \geq 1$ retains this action. In both cases,

$$
\widehat { a } _ { t } ( s ) \in D _ { t } ( s ) , \qquad \operatorname* { m a x } _ { a \in D _ { t } ( s ) } \widehat { Q } _ { t } ( s , a ) = \operatorname* { m a x } _ { a } \widehat { Q } _ { t } ( s , a ) .
$$

Thus the execution set preserves the global critic maximizer selected by the fixed action order. A greedy selector using that order consequently returns the same item with or without filtering.

Choose any $a ^ { * } ( s ) \in \arg \operatorname* { m a x } _ { a } Q ^ { * } ( s , a )$ . Substituting ${ Q } ^ { * } = \widehat { { Q } } _ { t } - { \xi } _ { t }$ into the optimal Bellman residual gives

$$
\begin{array} { l } { { b _ { \pi _ { t } } ( s ) = Q ^ { * } ( s , a ^ { * } ( s ) ) - \displaystyle \sum _ { a } \pi _ { t } ( a \mid s ) Q ^ { * } ( s , a ) } } \\ { { \displaystyle \quad \quad = { \widehat Q } _ { t } ( s , a ^ { * } ( s ) ) - \displaystyle \sum _ { a } \pi _ { t } ( a \mid s ) { \widehat Q } _ { t } ( s , a ) } } \\ { { \displaystyle \quad \quad + \displaystyle \sum _ { a } \pi _ { t } ( a \mid s ) \big [ \xi _ { t } ( s , a ) - \xi _ { t } ( s , a ^ { * } ( s ) ) \big ] } . } \end{array}
$$

For the first diference, replace $\widehat { Q } _ { t } ( s , a ^ { * } ( s ) )$ by its maximum over all actions, which equals its maximum over $D _ { t } ( s )$ by the preceding argument. This upper bounds that diference by $\sigma _ { t } ( s )$ . Every bracket in the final sum is at most $\beta _ { t }$ , and the policy weights are nonnegative and sum to one. Consequently,

$$
0 \leq b _ { \pi _ { t } } ( s ) \leq \sigma _ { t } ( s ) + \beta _ { t } .
$$

The residual identity in the proof of Theorem 3.2 now gives

$$
\begin{array} { r l r } {  { V ^ { * } ( \mu _ { 0 } ) - V ^ { \pi _ { t } } ( \mu _ { 0 } ) = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d ^ { \pi _ { t } } } [ b _ { \pi _ { t } } ( s ) ] } } \\ & { } & { \leq \frac { \overline { { \sigma } } _ { t } + \beta _ { t } } { 1 - \gamma } . } \end{array}
$$

For every state and action pair, the triangle inequality gives

$$
| \xi _ { t } ( s , a ) - \xi _ { t } ( s , b ) | \leq | \xi _ { t } ( s , a ) | + | \xi _ { t } ( s , b ) | \leq 2 \| \widehat { Q } _ { t } - Q ^ { * } \| _ { \infty } .
$$

Maximizing shows that $\beta _ { t } \leq 2 \| \widehat { Q } _ { t } - Q ^ { * } \| _ { \infty }$

Suppose a strictly suboptimal state and action pair exists. Finiteness makes the minimum of its positive gaps strictly positive, so $\Delta _ { \operatorname* { m i n } } > 0$ is well defined. Let $\widehat { a } \in$ arg ma $\mathfrak { c } _ { a } \widehat { Q } _ { t } ( s , a )$ . Then

$$
\begin{array} { r l r } & { } & { V ^ { * } ( s ) - Q ^ { * } ( s , \widehat { a } ) = \widehat { Q } _ { t } ( s , a ^ { * } ( s ) ) - \widehat { Q } _ { t } ( s , \widehat { a } ) } \\ & { } & { ~ + \xi _ { t } ( s , \widehat { a } ) - \xi _ { t } ( s , a ^ { * } ( s ) ) } \\ & { } & { ~ \leq \beta _ { t } , ~ } \end{array}
$$

because the first diference is nonpositive. If ba were suboptimal and $\beta _ { t } < \Delta _ { \operatorname* { m i n } }$ , this would imply

$$
\Delta _ { \operatorname* { m i n } } \leq V ^ { * } ( s ) - Q ^ { * } ( s , \widehat { a } ) \leq \beta _ { t } < \Delta _ { \operatorname* { m i n } } ,
$$

a contradiction. This proves ranking consistency for every state. Since $D _ { t } ( s )$ contains a global critic maximizer, every critic maximizer within $D _ { t } ( s )$ is also a global critic maximizer. A policy supported on these actions is therefore supported on the true optimal actions. Lemma D.1 proves its optimality. If no strictly suboptimal pair exists, every policy is supported on optimal actions, and the same lemma applies directly.

Finally, the quantitative value convergence statement follows by taking limits in the value bound above. For exact greedy selection, $\beta _ { t } \to 0$ implies $\beta _ { t } < \Delta _ { \operatorname* { m i n } }$ at every suficiently large index when a positive gap exists. At those indices the policy is optimal. A unique optimal action at each state forces the action distribution to be the same point mass at all such indices. If several actions are optimal, support on those actions imposes no requirement that the distribution converge. None of these steps uses convergence of a parameter vector or of the threshold. □

## E EXPERIMENT DETAILS

This appendix describes the comparison methods and the reported metrics. The metric definitions distinguish utility, catalog exposure, and retained set size so that the numerical comparisons can be interpreted separately from the calibration guarantee.

## E.1 Baseline Details

• A2C: an advantage actor–critic method that updates the policy using advantage-weighted policy gradients and trains the critic to estimate the state-value function (Mnih et al., 2016).

• DDPG: an of-policy deterministic actor–critic algorithm designed for continuous action spaces, which is applied to continuous action representations for slate recommendation (Lillicrap et al., 2016).

• TD3: an extension of DDPG that mitigates value overestimation and improves training stability through clipped double Q-learning, delayed policy updates, and target policy smoothing (Fujimoto et al., 2018).

• HAC: a hyper-actor–critic method that represents recommendation slates in a latent action space and decomposes slate generation into hyper-action inference and efect-action selection (Liu et al., 2023).

## E.2 Evaluation Metrics

The metrics below describe complete simulated sessions. Let N be the number of evaluated sessions, $T _ { n }$ the length of session n, and $R _ { n , t }$ its reward at request t. Let $L _ { n , t }$ denote the item collection used by the evaluation code at that request. A baseline with fixed slate size has $| L _ { n , t } | = M$ , while the reported RLCP set size uses its retained execution collection. Catalog diversity and ILD require the evaluated collection to be defined consistently across methods.

Depth. We first measure how long the interaction continues. Depth is the mean number of recommendation steps before departure:

$$
\mathrm { { D e p t h } } = { \frac { 1 } { N } } \sum _ { n = 1 } ^ { N } T _ { n } .
$$

A larger value means that the simulated sessions contain more interactions; it does not specify how much reward those interactions produce.

Average Reward. To measure the reward per interaction while giving equal weight to each session, we first average within sessions and then across sessions:

$$
{ \mathrm { A v g R e w a r d } } = { \frac { 1 } { N } } \sum _ { n = 1 } ^ { N } { \frac { 1 } { T _ { n } } } \sum _ { t = 1 } ^ { T _ { n } } R _ { n , t } .
$$

Total Reward. In contrast, total reward records the cumulative reward of a session before averaging across sessions:

$$
\mathrm { T o t a l R e w a r d } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sum _ { t = 1 } ^ { T _ { n } } R _ { n , t } .
$$

Diversity. To measure exposure across users, catalog diversity counts distinct items in the B active requests of an evaluation batch and averages this count over $H _ { \mathrm { e v a l } }$ evaluation steps, following KuaiSim:

$$
\mathrm { D i v e r s i t y } = \frac { 1 } { H _ { \mathrm { e v a l } } } \sum _ { h = 1 } ^ { H _ { \mathrm { e v a l } } } \left| \bigcup _ { j = 1 } ^ { B } L _ { j , h } \right| .
$$

The tables call this metric Diversity. It counts exposed items and is distinct from the proxy retention probability controlled by calibration.

Intra List Diversity. Catalog diversity does not describe how similar the items within a request are. ILD measures that separate property through mean pairwise dissimilarity within each evaluated collection. Write $M _ { n , t } = | L _ { n , t } |$ and let $e _ { a }$ be the embedding of item a. Then

$$
\begin{array} { r l r } {  { \mathrm { I L D } = \frac { 1 } { \sum _ { n = 1 } ^ { N } T _ { n } } \sum _ { n = 1 } ^ { N } \sum _ { t = 1 } ^ { T _ { n } } \frac { 2 } { M _ { n , t } ( M _ { n , t } - 1 ) } } } \\ & { } & { \times \sum _ { 1 \le j < k \le M _ { n , t } } d \big ( e _ { ( L _ { n , t } ) _ { j } } , e _ { ( L _ { n , t } ) _ { k } } \big ) . } \end{array}
$$

Here $d ( \cdot , \cdot )$ is the item dissimilarity function and $M _ { n , t }$ is the number of items in the evaluated collection at request t of session n. For $M _ { n , t } = 1$ , the corresponding diversity term is defined as zero.

Admissible Set Size. Finally, the average admissible set size records how many items remain after filtering, averaged over evaluation requests:

$$
{ \mathrm { S e t S i z e } } = { \frac { 1 } { \sum _ { n = 1 } ^ { N } T _ { n } } } \sum _ { n = 1 } ^ { N } \sum _ { t = 1 } ^ { T _ { n } } | L _ { n , t } | .
$$

A smaller value indicates that fewer alternatives are retained. It is not a measure of total computation, since it excludes scoring the candidate pool and obtaining the proxy label.

## F FULL RESULTS

The following tables report the complete measurements underlying Section 5. Each configuration fixes the dataset, reward signal, patience parameters, and baseline slate size M, with the matched cap $K = M$ for RLCP. Comparisons therefore concern the same maximum set size within a configuration. Boldface identifies the highest depth and catalog diversity and the lowest mean set size, including ties. These markings compare point estimates and do not imply statistical significance.

Table 2: KuaiRand-Pure results under patience setting $\left( p , q \right) = \left( 1 , 2 \right)$ with is click as the reward signal for displayed slate sizes $M \in \{ 3 , 4 , 5 \}$ . The final seven columns report behavior rates; set size is reported for RLCP variants when available.
<table><tr><td>Method</td><td>Depth reward reward Diversity ILD</td><td>Avg.</td><td>Total</td><td></td><td>Set</td><td>size Click Comment Follow Forward Hate Like view</td><td></td><td></td><td></td><td></td><td>Long</td></tr><tr><td colspan="10">Slate size  $M = 3$ </td></tr><tr><td>DDPG</td><td>18.80</td><td>0.62</td><td>13.46</td><td>25.00 0.99 3.00</td><td></td><td>0.62</td><td>0.61</td><td>0.54</td><td></td><td>0.54 0.55 0.58</td><td>0.64</td></tr><tr><td>TD3</td><td>19.20</td><td>0.65</td><td>13.24</td><td>54.50 0.99 3.00</td><td></td><td>0.65</td><td>0.62</td><td>0.52</td><td></td><td>0.540.52 0.58</td><td>0.63</td></tr><tr><td>A2C</td><td>19.60</td><td>0.66</td><td>13.32</td><td>3.00 0.99 3.00</td><td></td><td>0.66</td><td>0.61</td><td>0.58</td><td></td><td>0.56 0.54 0.60</td><td>0.62</td></tr><tr><td>HAC</td><td>19.10</td><td>0.67</td><td>13.47</td><td>4.70 0.98</td><td>3.00</td><td>0.67</td><td>0.62</td><td>0.57</td><td>0.55</td><td>0.54 0.56</td><td>0.68</td></tr><tr><td>RLCP-Single</td><td>19.30</td><td>0.64</td><td>13.20</td><td>92.40 0.99 2.91</td><td></td><td>0.64</td><td>0.57</td><td>0.55</td><td>0.55</td><td>0.52 0.59</td><td>0.65</td></tr><tr><td>RLCP</td><td>19.40</td><td>0.61</td><td>12.77</td><td>28.50 1.003.00</td><td></td><td>0.61</td><td>0.58</td><td>0.56</td><td>0.55</td><td>0.54 0.57</td><td>0.59</td></tr><tr><td colspan="10">Slate size  $M = 4$ </td><td></td></tr><tr><td>DDPG</td><td>19.60</td><td>0.66</td><td>13.55</td><td>19.50 0.99 4.00</td><td></td><td>0.66</td><td>0.61</td><td>0.58</td><td></td><td>0.58 0.52 0.58</td><td>0.65</td></tr><tr><td>TD3</td><td>19.70</td><td>0.61</td><td>13.49</td><td>84.40 0.99 4.00</td><td></td><td>0.61</td><td>0.61</td><td>0.55</td><td>0.52</td><td>0.54 0.56</td><td>0.61</td></tr><tr><td>A2C</td><td>19.90</td><td>0.67</td><td>13.35</td><td>15.500.994.00</td><td></td><td>0.67</td><td>0.65</td><td>0.59</td><td>0.57</td><td>0.55 0.63</td><td>0.66</td></tr><tr><td>HAC</td><td>19.60</td><td>0.67</td><td>13.62</td><td>6.60 0.99 4.00</td><td></td><td>0.67</td><td>0.59</td><td>0.58</td><td>0.54</td><td>0.50 0.61</td><td>0.65</td></tr><tr><td>RLCP-Single</td><td>19.90</td><td>0.64</td><td>13.52</td><td>126.40 0.99 4.00</td><td></td><td>0.64</td><td>0.61</td><td>0.59</td><td>0.55</td><td>0.54 0.60</td><td>0.64</td></tr><tr><td>RLCP</td><td>19.90</td><td>0.59</td><td>13.10</td><td>50.50 1.00 4.00</td><td></td><td>0.59</td><td>0.58</td><td>0.52</td><td>0.51</td><td>0.52 0.56</td><td>0.59</td></tr><tr><td colspan="10">Slate size  $M = 5$ </td></tr><tr><td>DDPG</td><td>19.90</td><td>0.68</td><td>13.67</td><td>17.50 0.99 5.00</td><td></td><td>0.68</td><td>0.60</td><td>0.56</td><td>0.56</td><td>0.54 0.57</td><td>0.65</td></tr><tr><td>TD3</td><td>19.90</td><td>0.64</td><td>13.67</td><td>47.20 0.99</td><td>5.00</td><td>0.64</td><td>0.58</td><td>0.54</td><td>0.53</td><td>0.52 0.58</td><td>0.63</td></tr><tr><td>A2C</td><td>20.00</td><td>0.65</td><td>13.35</td><td>24.80 0.99</td><td>5.00</td><td>0.65</td><td>0.61</td><td>0.58</td><td>0.54</td><td>0.53 0.60</td><td>0.63</td></tr><tr><td>HAC</td><td>20.00</td><td>0.66</td><td>13.99</td><td>10.00 0.99</td><td>5.00</td><td>0.66</td><td>0.60</td><td>0.58</td><td>0.55</td><td>0.53 0.59</td><td>0.65</td></tr><tr><td>RLCP-Single</td><td>20.00</td><td>0.66</td><td>13.93</td><td>154.10 0.99 4.91</td><td></td><td>0.66</td><td>0.56</td><td>0.53</td><td>0.53</td><td>0.55 0.55</td><td>0.64</td></tr><tr><td>RLCP</td><td>20.00</td><td>0.63</td><td>13.03</td><td>36.40 1.005.00</td><td></td><td>0.63</td><td>0.56</td><td>0.56</td><td>0.53</td><td>0.54 0.56</td><td>0.63</td></tr></table>

Table 3: KuaiRand-Pure results under patience setting $( p , q ) = ( 0 . 4 , 2 )$ with is click as the reward signal for displayed slate sizes $M \in \{ 4 , 5 \}$ . The final seven columns report behavior rates; set size is reported for RLCP variants when available.
<table><tr><td>Method</td><td></td><td>Avg.</td><td>Total</td><td>Depth reward reward Diversity ILD</td><td>Set size</td><td></td><td>Click Comment Follow Forward Hate Like view</td><td></td><td></td><td></td><td></td><td>Long</td></tr><tr><td>Slate size</td><td> $M = 4$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DDPG</td><td>46.80</td><td>0.70</td><td>32.83</td><td>14.400.99</td><td>4.00</td><td>0.70</td><td>0.62</td><td>0.59</td><td></td><td>0.57 0.52 0.60</td><td></td><td>0.68</td></tr><tr><td>TD3</td><td>48.00</td><td>0.69</td><td>33.46</td><td>25.90 0.99</td><td>4.00</td><td>0.69</td><td>0.64</td><td>0.59</td><td>0.60</td><td>0.53 0.61</td><td></td><td>0.70</td></tr><tr><td>A2C</td><td>46.00</td><td>0.64</td><td>31.35</td><td>4.10 0.98</td><td>4.00</td><td>0.64</td><td>0.60</td><td>0.57</td><td>0.54</td><td>0.55 0.57</td><td></td><td>0.63</td></tr><tr><td>HAC</td><td>48.00</td><td>0.68</td><td>32.95</td><td>7.10 0.99</td><td>4.00</td><td>0.68</td><td>0.64</td><td>0.58</td><td>0.57</td><td>0.53 0.61</td><td></td><td>0.69</td></tr><tr><td>RLCP-Single</td><td>47.60</td><td>0.68</td><td>32.92</td><td>118.10 0.99</td><td>3.84</td><td>0.68</td><td>0.62</td><td>0.57</td><td>0.54</td><td>0.55 0.60</td><td></td><td>0.65</td></tr><tr><td>RLCP</td><td>47.20</td><td>0.62</td><td>31.91</td><td>41.601.00</td><td>3.91</td><td>0.63</td><td>0.57</td><td>0.58</td><td>0.55</td><td>0.54 0.59</td><td></td><td>0.62</td></tr><tr><td>Slate size</td><td> $M = 5$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DDPG</td><td>48.00</td><td>0.68</td><td>33.63</td><td>20.000.98</td><td>5.00</td><td>0.68</td><td>0.62</td><td>0.60</td><td>0.58</td><td>0.52 0.61</td><td></td><td>0.66</td></tr><tr><td>TD3</td><td>48.00</td><td>0.68</td><td>33.74</td><td>37.90 0.99</td><td>5.00</td><td>0.68</td><td>0.64</td><td>0.58</td><td>0.57</td><td></td><td>0.54 0.60</td><td>0.69</td></tr><tr><td>A2C</td><td>48.00</td><td>0.63</td><td>32.49</td><td>5.00 0.98</td><td>5.00</td><td>0.63</td><td>0.57</td><td>0.57</td><td>0.54</td><td>0.53 0.58</td><td></td><td>0.64</td></tr><tr><td>HAC</td><td>48.00</td><td>0.59</td><td>31.45</td><td>5.40 0.98</td><td>5.00</td><td>0.59</td><td>0.57</td><td>0.55</td><td>0.51</td><td>0.53 0.57</td><td></td><td>0.60</td></tr><tr><td>RLCP-Single</td><td>48.00</td><td>0.69</td><td>33.51</td><td>149.90 0.99</td><td>4.94</td><td>0.69</td><td>0.64</td><td>0.58</td><td>0.58</td><td>0.56 0.60</td><td></td><td>0.69</td></tr><tr><td>RLCP</td><td>48.00</td><td>0.61</td><td>31.35</td><td>77.001.00</td><td>5.00</td><td>0.61</td><td>0.60</td><td>0.55</td><td>0.55</td><td>0.54 0.60</td><td></td><td>0.58</td></tr></table>

Table 4: KuaiRand-Pure results under patience setting $\left( p , q \right) = \left( 1 , 2 \right)$ with is like as the reward signal for displayed slate sizes $M \in \{ 4 , 5 , 6 \}$ . The final seven columns report behavior rates; set size is reported for RLCP variants when available.
<table><tr><td>Method</td><td></td><td>Avg.</td><td>Total</td><td>Depth reward reward Diversity ILD</td><td>Set size</td><td>Click Comment Follow Forward Hate Like view</td><td></td><td></td><td></td><td></td><td></td><td>Long</td></tr><tr><td>Slate size</td><td> $M = 4$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DDPG</td><td>19.80</td><td>0.61</td><td>12.79</td><td>24.000.99</td><td>4.00</td><td>0.66</td><td>0.60</td><td>0.58</td><td></td><td>0.55 0.54 0.61</td><td></td><td>0.65</td></tr><tr><td>TD3</td><td>19.80</td><td>0.60</td><td>12.69</td><td>51.80 0.99</td><td>4.00</td><td>0.62</td><td>0.60</td><td>0.59</td><td>0.57</td><td></td><td>0.56 0.60</td><td>0.62</td></tr><tr><td>A2C</td><td>19.70</td><td>0.60</td><td>12.83</td><td>4.000.98</td><td>4.00</td><td>0.62</td><td>0.60</td><td>0.59</td><td>0.55</td><td>0.53 0.60</td><td></td><td>0.59</td></tr><tr><td>HAC</td><td>20.00</td><td>0.58</td><td>12.84</td><td>8.60 0.99</td><td>4.00</td><td>0.64</td><td>0.59</td><td>0.56</td><td>0.57</td><td>0.53 0.58</td><td></td><td>0.66</td></tr><tr><td>RLCP-Single</td><td>19.80</td><td>0.61</td><td>12.85</td><td>119.70 0.99</td><td>3.91</td><td>0.63</td><td>0.59</td><td>0.57</td><td>0.57</td><td>0.54 0.61</td><td></td><td>0.63</td></tr><tr><td>RLCP</td><td>19.70</td><td>0.59</td><td>12.27</td><td>58.601.00</td><td>4.00</td><td>0.64</td><td>0.59</td><td>0.58</td><td>0.54</td><td>0.54 0.59</td><td></td><td>0.64</td></tr><tr><td colspan="2">Slate size  $M = 5$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DDPG</td><td>20.00</td><td>0.58</td><td>12.88</td><td>21.800.99</td><td>5.00</td><td>0.62</td><td>0.57</td><td>0.56</td><td>0.54</td><td>0.54 0.58</td><td></td><td>0.58</td></tr><tr><td>TD3</td><td>19.90</td><td>0.58</td><td>13.18</td><td>72.300.99</td><td>5.00</td><td>0.61</td><td>0.59</td><td>0.54</td><td>0.55</td><td></td><td>0.52 0.58</td><td>0.63</td></tr><tr><td>A2C</td><td>19.80</td><td>0.59</td><td>12.87</td><td>5.00 0.98</td><td>5.00</td><td>0.62</td><td>0.60</td><td>0.57</td><td>0.54</td><td>0.55 0.59</td><td></td><td>0.62</td></tr><tr><td>HAC</td><td>20.00</td><td>0.57</td><td>13.06</td><td>10.20 0.99</td><td>5.00</td><td>0.65</td><td>0.59</td><td>0.58</td><td>0.56</td><td>0.54 0.57</td><td></td><td>0.63</td></tr><tr><td>RLCP-Single</td><td>20.00</td><td>0.58</td><td>12.83</td><td>149.60 0.99</td><td>4.78</td><td>0.64</td><td>0.60</td><td>0.57</td><td>0.53</td><td>0.52 0.58</td><td></td><td>0.65</td></tr><tr><td>RLCP</td><td>20.00</td><td>0.56</td><td>11.98</td><td>64.401.00</td><td>5.00</td><td>0.62</td><td>0.57</td><td>0.57</td><td>0.53</td><td>0.54 0.56</td><td></td><td>0.62</td></tr><tr><td colspan="2">Slate size M = 6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DDPG</td><td>20.00</td><td>0.57</td><td>12.62</td><td>25.50 0.99</td><td>6.00</td><td>0.59</td><td>0.59</td><td>0.55</td><td>0.55</td><td>0.53 0.57</td><td></td><td>0.57</td></tr><tr><td>TD3</td><td>20.00</td><td>0.59</td><td>12.51</td><td>60.70 0.99</td><td>6.00</td><td>0.65</td><td>0.61</td><td>0.55</td><td>0.55</td><td>0.55 0.59</td><td></td><td>0.63</td></tr><tr><td>A2C</td><td>20.00</td><td>0.62</td><td>12.74</td><td>6.00 0.99</td><td>6.00</td><td>0.66</td><td>0.62</td><td>0.58</td><td>0.54</td><td>0.53 0.62</td><td></td><td>0.64</td></tr><tr><td>HAC</td><td>20.00</td><td>0.62</td><td>12.76</td><td>9.10 0.99</td><td>6.00</td><td>0.63</td><td>0.59</td><td>0.57</td><td>0.56</td><td>0.55 0.62</td><td></td><td>0.61</td></tr><tr><td>RLCP-Single</td><td>19.90</td><td>0.60</td><td>12.83</td><td>155.80 0.99</td><td>4.47</td><td>0.66</td><td>0.61</td><td>0.57</td><td>0.55</td><td>0.54 0.60</td><td></td><td>0.64</td></tr><tr><td>RLCP</td><td>20.00</td><td>0.58</td><td>11.46</td><td>100.20 1.006.00</td><td></td><td>0.60</td><td>0.56</td><td>0.55</td><td>0.53</td><td>0.51 0.58</td><td></td><td>0.59</td></tr></table>

Table 5: KuaiRand-Pure results under patience setting $( p , q ) = ( 0 . 4 , 2 )$ with is like as the reward signal for displayed slate sizes $M \in \{ 5 , 6 \}$ . The final seven columns report behavior rates; set size is reported for RLCP variants when available.
<table><tr><td>Method</td><td></td><td>Avg.</td><td>Total</td><td>Depth reward reward Diversity ILD</td><td>Set size</td><td>Click Comment Follow Forward Hate Like view</td><td></td><td></td><td></td><td></td><td></td><td>Long</td></tr><tr><td colspan="10">Slate size  $M = 5$ </td><td></td><td></td><td></td></tr><tr><td>DDPG</td><td>47.60</td><td>0.57</td><td>31.62</td><td>35.20 0.99</td><td>5.00</td><td>0.65</td><td>0.60</td><td>0.57</td><td></td><td>0.55 0.57 0.57</td><td></td><td>0.67</td></tr><tr><td>TD3</td><td>48.00</td><td>0.60</td><td>31.93</td><td>53.90 0.99</td><td>5.00</td><td>0.69</td><td>0.64</td><td>0.57</td><td>0.58</td><td>0.56 0.60</td><td></td><td>0.71</td></tr><tr><td>A2C</td><td>45.60</td><td>0.60</td><td>30.79</td><td>5.00 0.98</td><td>5.00</td><td>0.64</td><td>0.62</td><td>0.58</td><td>0.56</td><td>0.55 0.60</td><td></td><td>0.66</td></tr><tr><td>HAC</td><td>47.20</td><td>0.61</td><td>29.78</td><td>5.10 0.98</td><td>5.00</td><td>0.63</td><td>0.59</td><td>0.59</td><td>0.52</td><td></td><td>0.54 0.61</td><td>0.63</td></tr><tr><td>RLCP-Single</td><td>48.00</td><td>0.62</td><td>31.09</td><td>151.70 0.99</td><td>4.84</td><td>0.70</td><td>0.63</td><td>0.61</td><td>0.57</td><td></td><td>0.55 0.62</td><td>0.70</td></tr><tr><td>RLCP</td><td>47.70</td><td>0.56</td><td>29.20</td><td>72.201.00</td><td>4.97</td><td>0.60</td><td>0.58</td><td>0.56</td><td>0.53</td><td>0.53 0.57</td><td></td><td>0.58</td></tr><tr><td colspan="10">Slate size  $M = 6$ </td><td></td><td></td><td></td></tr><tr><td>DDPG</td><td>48.00</td><td>0.61</td><td>30.86</td><td>58.300.99</td><td>6.00</td><td>0.67</td><td>0.62</td><td>0.59</td><td>0.56</td><td>0.52 0.61</td><td></td><td>0.67</td></tr><tr><td>TD3</td><td>46.80</td><td>0.60</td><td>31.75</td><td>124.700.99</td><td>6.00</td><td>0.71</td><td>0.64</td><td>0.60</td><td>0.58</td><td>0.52 0.60</td><td></td><td>0.70</td></tr><tr><td>A2C</td><td>48.00</td><td>0.63</td><td>30.11</td><td>6.000.98</td><td>6.00</td><td>0.70</td><td>0.63</td><td>0.59</td><td>0.58</td><td></td><td>0.56 0.63</td><td>0.69</td></tr><tr><td>HAC</td><td>47.90</td><td>0.59</td><td>30.54</td><td>7.50 0.98</td><td>6.00</td><td>0.65</td><td>0.60</td><td>0.58</td><td>0.53</td><td></td><td>0.52 0.59</td><td>0.64</td></tr><tr><td>RLCP-Single</td><td>48.00</td><td>0.62</td><td>31.22</td><td>184.200.99</td><td>5.72</td><td>0.70</td><td>0.65</td><td>0.61</td><td>0.58</td><td></td><td>0.53 0.62</td><td>0.69</td></tr><tr><td>RLCP</td><td>48.00</td><td>0.59</td><td>28.52</td><td>85.50 1.00 6.00</td><td></td><td>0.61</td><td>0.55</td><td>0.55</td><td>0.54</td><td></td><td>0.53 0.60</td><td>0.57</td></tr></table>

Table 6: ML-1M results under patience setting $( p , q ) = ( 1 , 2 )$ with is click as the reward signal for displayed slate sizes $M \in \{ 4 , 5 \}$ . The final three columns report behavior rates; set size is reported for RLCP variants when available.
<table><tr><td></td><td></td><td> $\operatorname { A v g . }$ </td><td> $\mathrm { T o t a l }$ </td><td></td><td></td><td> $\mathbf { S e t }$ </td><td></td><td></td><td></td></tr><tr><td>Method</td><td></td><td></td><td></td><td>Depth reward reward Diversity</td><td>ILD</td><td>size</td><td>Click Like Star</td><td></td><td></td></tr><tr><td colspan="10">Slate size  $M = 4$ </td></tr><tr><td>DDPG</td><td>19.90</td><td>0.69</td><td>13.95</td><td>23.90</td><td>0.99</td><td>4.00</td><td>0.69</td><td>0.69</td><td>0.70</td></tr><tr><td>TD3</td><td>19.90</td><td>0.73</td><td>14.90</td><td>15.90</td><td>0.99</td><td>4.00</td><td>0.73</td><td>0.72</td><td>0.70</td></tr><tr><td>A2C</td><td>19.20</td><td>0.58</td><td>11.97</td><td>46.30</td><td>0.99</td><td>4.00</td><td>0.58</td><td>0.61</td><td>0.62</td></tr><tr><td>HAC</td><td>20.00</td><td>0.68</td><td>14.30</td><td>4.80</td><td>0.99</td><td>4.00</td><td>0.68</td><td>0.66</td><td>0.66</td></tr><tr><td>RLCP-Single</td><td>19.80</td><td>0.58</td><td>11.85</td><td>119.40</td><td>0.99</td><td>4.00</td><td>0.58</td><td>0.58</td><td>0.58</td></tr><tr><td>RLCP</td><td>20.00</td><td>0.61</td><td>12.85</td><td>41.80</td><td>1.00</td><td>4.00</td><td>0.61</td><td>0.57</td><td>0.55</td></tr><tr><td colspan="10">Slate size  $M = 5$ </td></tr><tr><td>DDPG</td><td>19.90</td><td>0.69</td><td>13.82</td><td>35.20</td><td>0.99</td><td>5.00</td><td>0.69</td><td>0.68</td><td>0.69</td></tr><tr><td>TD3</td><td>20.00</td><td>0.73</td><td>14.87</td><td>16.30</td><td>0.99</td><td>5.00</td><td>0.73</td><td>0.72</td><td>0.75</td></tr><tr><td>A2C</td><td>19.90</td><td>0.58</td><td>11.86</td><td>49.50</td><td>0.99</td><td>5.00</td><td>0.58</td><td>0.61</td><td>0.61</td></tr><tr><td>HAC</td><td>20.00</td><td>0.63</td><td>12.80</td><td>6.70</td><td>0.99</td><td>5.00</td><td>0.63</td><td>0.61</td><td>0.63</td></tr><tr><td>RLCP-Single</td><td>20.00</td><td>0.57</td><td>12.41</td><td>17.30</td><td>0.99</td><td>5.00</td><td>0.57</td><td>0.61</td><td>0.62</td></tr><tr><td>RLCP</td><td>19.80</td><td>0.60</td><td>13.71</td><td>54.90</td><td>1.00</td><td>4.53</td><td>0.64</td><td>0.59</td><td>0.57</td></tr></table>

Table 7: ML-1M results under patience setting $( p , q ) = ( 0 . 4 , 2 )$ with is click as the reward signal for displayed slate sizes $M \in \{ 4 , 5 \}$ . The final three columns report behavior rates; set size is reported for RLCP variants when available.
<table><tr><td></td><td></td><td> $\operatorname { A v g . }$ </td><td>Total</td><td></td><td></td><td> $\mathbf { S e t }$ </td><td></td><td></td><td></td></tr><tr><td>Method</td><td>Depth reward</td><td></td><td>reward</td><td>Diversity</td><td>ILD</td><td>size</td><td>Click Like Star</td><td></td><td></td></tr><tr><td colspan="10">Slate size  $M = 4$ </td></tr><tr><td>DDPG</td><td>46.80</td><td>0.69</td><td>33.76</td><td>12.90</td><td>0.99</td><td>4.00</td><td>0.69</td><td>0.71</td><td>0.73</td></tr><tr><td>TD3</td><td>46.80</td><td>0.71</td><td>35.13</td><td>14.30</td><td>0.99</td><td>4.00</td><td>0.71</td><td>0.71</td><td>0.73</td></tr><tr><td>A2C</td><td>44.00</td><td>0.59</td><td>26.76</td><td>31.20</td><td>0.99</td><td>4.00</td><td>0.59</td><td>0.64</td><td>0.64</td></tr><tr><td>HAC</td><td>45.20</td><td>0.61</td><td>29.02</td><td>9.70</td><td>0.99</td><td>4.00</td><td>0.61</td><td>0.64</td><td>0.65</td></tr><tr><td>RLCP-Single</td><td>45.60</td><td>0.57</td><td>26.56</td><td>119.30</td><td>0.99</td><td>3.97</td><td>0.57</td><td>0.59</td><td>0.63</td></tr><tr><td>RLCP</td><td>46.80</td><td>0.57</td><td>27.95</td><td>62.00</td><td>1.00</td><td>3.78</td><td>0.59</td><td>0.60</td><td>0.60</td></tr><tr><td colspan="10">Slate size  $M = 5$ </td></tr><tr><td>DDPG</td><td>47.20</td><td>0.69</td><td>34.00</td><td>15.80</td><td>0.99</td><td>5.00</td><td>0.69</td><td>0.69</td><td>0.72</td></tr><tr><td>TD3</td><td>48.00</td><td>0.72</td><td>34.98</td><td>18.20</td><td>0.99</td><td>5.00</td><td>0.72</td><td>0.71</td><td>0.71</td></tr><tr><td>A2C</td><td>48.00</td><td>0.59</td><td>28.88</td><td>7.80</td><td>0.99</td><td>5.00</td><td>0.59</td><td>0.66</td><td>0.68</td></tr><tr><td>HAC</td><td>48.00</td><td>0.65</td><td>32.25</td><td>5.30</td><td>0.99</td><td>5.00</td><td>0.65</td><td>0.67</td><td>0.70</td></tr><tr><td>RLCP-Single</td><td>47.60</td><td>0.59</td><td>28.12</td><td>94.80</td><td>0.99</td><td>4.97</td><td>0.59</td><td>0.62</td><td>0.63</td></tr><tr><td>RLCP</td><td>48.00</td><td>0.63</td><td>30.64</td><td>45.10</td><td>1.00</td><td>4.97</td><td>0.63</td><td>0.63</td><td>0.62</td></tr></table>

Table 8: ML-1M results under patience setting $\left( p , q \right) = \left( 1 , 2 \right)$ with is like as the reward signal for displayed slate sizes $M \in \{ 4 , 5 , 6 \}$ . The final three columns report behavior rates; set size is reported for RLCP variants when available.
<table><tr><td></td><td></td><td>Avg.</td><td>Total</td><td colspan="6">Set</td></tr><tr><td>Method</td><td>Depth reward</td><td></td><td>reward</td><td>Diversity</td><td>ILD</td><td>size</td><td>Click Like</td><td></td><td>Star</td></tr><tr><td colspan="10">Slate size  $M = 4$ </td></tr><tr><td>DDPG</td><td>19.80</td><td>0.65</td><td>13.21</td><td>32.80</td><td>0.99</td><td>4.00</td><td>0.63</td><td>0.65</td><td>0.67</td></tr><tr><td>TD3</td><td>19.80</td><td>0.66</td><td>13.71</td><td>20.80</td><td>0.99</td><td>4.00</td><td>0.65</td><td>0.66</td><td>0.66</td></tr><tr><td>A2C</td><td>19.60</td><td>0.60</td><td>12.83</td><td>4.10</td><td>0.99</td><td>4.00</td><td>0.57</td><td>0.60</td><td>0.60</td></tr><tr><td>HAC</td><td>19.90</td><td>0.64</td><td>13.31</td><td>13.90</td><td>0.99</td><td>4.00</td><td>0.67</td><td>0.64</td><td>0.63</td></tr><tr><td>RLCP-Single</td><td>19.80</td><td>0.63</td><td>12.75</td><td>7.70</td><td>0.99</td><td>4.00</td><td>0.58</td><td>0.63</td><td>0.62</td></tr><tr><td>RLCP</td><td>19.90</td><td>0.61</td><td>12.49</td><td>74.10</td><td>1.00</td><td>4.00</td><td>0.62</td><td>0.61</td><td>0.60</td></tr><tr><td colspan="10">Slate size  $M = 5$ </td></tr><tr><td>DDPG</td><td>19.90</td><td>0.62</td><td>13.09</td><td>52.00</td><td>0.99</td><td>5.00</td><td>0.64</td><td>0.62</td><td>0.63</td></tr><tr><td>TD3</td><td>20.00</td><td>0.67</td><td>14.06</td><td>24.50</td><td>0.99</td><td>5.00</td><td>0.67</td><td>0.67</td><td>0.67</td></tr><tr><td>A2C</td><td>19.80</td><td>0.67</td><td>13.05</td><td>5.70</td><td>0.99</td><td>5.00</td><td>0.60</td><td>0.67</td><td>0.65</td></tr><tr><td>HAC</td><td>19.90</td><td>0.67</td><td>13.41</td><td>11.40</td><td>0.99</td><td>5.00</td><td>0.63</td><td>0.67</td><td>0.66</td></tr><tr><td>RLCP-Single</td><td>20.00</td><td>0.63</td><td>13.03</td><td>11.50</td><td>0.99</td><td>5.00</td><td>0.60</td><td>0.63</td><td>0.62</td></tr><tr><td>RLCP</td><td>20.00</td><td>0.62</td><td>12.34</td><td>64.60</td><td>1.00</td><td>5.00</td><td>0.63</td><td>0.62</td><td>0.60</td></tr><tr><td colspan="10">Slate size M = 6</td></tr><tr><td>DDPG</td><td>20.00</td><td>0.66</td><td>13.46</td><td>59.10</td><td>0.99</td><td>6.00</td><td>0.63</td><td>0.66</td><td>0.67</td></tr><tr><td>TD3</td><td>20.00</td><td>0.69</td><td>13.73</td><td>26.20</td><td>0.99</td><td>6.00</td><td>0.65</td><td>0.69</td><td>0.70</td></tr><tr><td>A2C</td><td>19.80</td><td>0.65</td><td>13.19</td><td>9.60</td><td>0.99</td><td>6.00</td><td>0.62</td><td>0.65</td><td>0.69</td></tr><tr><td>HAC</td><td>20.00</td><td>0.63</td><td>13.34</td><td>28.60</td><td>0.99</td><td>6.00</td><td>0.61</td><td>0.63</td><td>0.64</td></tr><tr><td>RLCP-Single</td><td>20.00</td><td>0.61</td><td>12.67</td><td>173.70</td><td>0.99</td><td>5.94</td><td>0.56</td><td>0.61</td><td>0.61</td></tr><tr><td>RLCP</td><td>20.00</td><td>0.58</td><td>12.00</td><td>69.20</td><td>1.00</td><td>6.00</td><td>0.61</td><td>0.58</td><td>0.56</td></tr></table>

Table 9: ML-1M results under patience setting $( p , q ) = ( 0 . 4 , 2 )$ with is like as the reward signal for displayed slate sizes $M \in \{ 4 , 5 \}$ . The final three columns report behavior rates; set size is reported for RLCP variants when available.
<table><tr><td></td><td></td><td>Avg. reward</td><td>Total</td><td></td><td></td><td>Set</td><td></td><td></td><td></td></tr><tr><td>Method</td><td>Depth</td><td></td><td>reward</td><td>Diversity</td><td>ILD</td><td>size</td><td>Click Like Star</td><td></td><td></td></tr><tr><td colspan="10">Slate size  $M = 4$ </td></tr><tr><td>DDPG</td><td>47.20</td><td>0.73</td><td>33.86</td><td>20.10</td><td>0.99</td><td>4.00</td><td>0.68</td><td>0.73</td><td>0.70</td></tr><tr><td>TD3</td><td>47.20</td><td>0.70</td><td>34.57</td><td>20.60</td><td>0.99</td><td>4.00</td><td>0.69</td><td>0.70</td><td>0.70</td></tr><tr><td>A2C</td><td>41.20</td><td>0.64</td><td>29.45</td><td>22.30</td><td>0.99</td><td>4.00</td><td>0.60</td><td>0.64</td><td>0.63</td></tr><tr><td>HAC</td><td>44.20</td><td>0.71</td><td>31.59</td><td>6.80</td><td>0.99</td><td>4.00</td><td>0.68</td><td>0.71</td><td>0.69</td></tr><tr><td>RLCP-Single</td><td>46.80</td><td>0.63</td><td>29.46</td><td>12.40</td><td>0.99</td><td>4.00</td><td>0.61</td><td>0.63</td><td>0.63</td></tr><tr><td>RLCP</td><td>45.60</td><td>0.57</td><td>28.28</td><td>70.60</td><td>1.00</td><td>3.84</td><td>0.58</td><td>0.58</td><td>0.60</td></tr><tr><td colspan="10">Slate size  $M = 5$ </td></tr><tr><td>DDPG</td><td>48.00</td><td>0.70</td><td>34.83</td><td>22.40</td><td>0.99</td><td>5.00</td><td>0.71</td><td>0.70</td><td>0.70</td></tr><tr><td>TD3</td><td>48.00</td><td>0.73</td><td>34.48</td><td>11.60</td><td>0.99</td><td>5.00</td><td>0.71</td><td>0.73</td><td>0.72</td></tr><tr><td>A2C</td><td>47.70</td><td>0.64</td><td>31.75</td><td>7.40</td><td>0.99</td><td>5.00</td><td>0.62</td><td>0.64</td><td>0.65</td></tr><tr><td>HAC</td><td>47.20</td><td>0.65</td><td>32.45</td><td>25.10</td><td>0.99</td><td>5.00</td><td>0.65</td><td>0.65</td><td>0.68</td></tr><tr><td>RLCP-Single</td><td>48.00</td><td>0.62</td><td>31.27</td><td>8.50</td><td>0.99</td><td>5.00</td><td>0.63</td><td>0.62</td><td>0.62</td></tr><tr><td>RLCP</td><td>48.00</td><td>0.57</td><td>27.90</td><td>82.10</td><td>1.00</td><td>5.00</td><td>0.58</td><td>0.57</td><td>0.59</td></tr></table>