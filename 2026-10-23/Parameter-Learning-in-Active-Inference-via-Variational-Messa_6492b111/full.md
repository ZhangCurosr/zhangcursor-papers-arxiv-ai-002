# Parameter Learning in Active Inference via Variational Message Passing

Benedikt Waldeck<sup>1[0009−0001−5225−9933]</sup>

University of Modena and Reggio Emilia, Italy benedikt.waldeck@unimore.it

Abstract. Active inference casts perception, learning and planning as variational inference, where the Expected Free Energy (EFE) defines the planning objective. Recent work showed that EFE minimization can be written as Variational Free Energy (VFE) minimization on a generative model augmented with epistemic priors, and that the resulting entropy-corrected objective is minimized by sum-product message passing (SP-MP) on modified factor functions. These formulations treat model uncertainty as inference over a finite set of known models. We extend them to planning during learning, where beliefs over the observation and transition parameters θ<sub>A</sub>, θ<sub>B</sub> are Dirichlet priors. Closed-form Dirichlet updates are usually obtained by a structured mean-field factorization between parameters and states, but under exactly that factorization the information-gain corrections vanish. We resolve this by giving every factor its own parameter prior, frozen during a rollout. Conjugacy then makes each factor belief a product of a marginal over the categorical variables and a one-step Bayes posterior of the parameters. This reintroduces the datum-dependent parameter posteriors that mean-field factorization destroys. We show that evaluating the epistemic priors under the resulting Bethe joint recovers the information gains. Stationarity of this single closed-form objective yields SP-MP whose factor functions are augmented by the information-gain weight. Experiments on two grid POMDPs, in which only a planner that learns the model finds the safe route, show that the method plans to eficiently learn $\theta _ { A }$ and $\theta _ { B }$ . It reaches the goal on every episode, whereas SP-MP planners with the same inference update plateau or take the risky route.

Keywords: Active Inference · Parameter Learning · Planning as Inference · Factor Graphs · Variational Inference

## 1 Introduction

Active inference casts perception, planning and learning as variational inference on a single generative model [3]. Its planning objective, the Expected Free Energy (EFE), decomposes into risk, ambiguity and novelty and thereby balances goal pursuit against epistemic exploration, but evaluating it by enumerating policies scales exponentially with the horizon [5]. De Vries et al. [14] showed that EFE minimization can be reformulated as Variational Free Energy (VFE) minimization on a generative model augmented with epistemic priors. Nuijten et al. [13] showed that the augmented VFE equals the predictive VFE plus explicit entropy corrections, including the planning correction of Lázaro-Gredilla et al. [10]. They derived a message passing scheme in which each correction becomes a channel reparameterization of a factor function, so that EFE-based planning runs as sum-product message passing (SP-MP) on the original factor graph.

These formulations treat the model parameter $\theta$ as a single enumerable latent variable: they plan under uncertainty over a finite set of known models, not under uncertainty about the models themselves. Planning during Learning the observation model $\theta _ { A }$ and the transition model $\theta _ { B }$ conjugately, an agent that knows neither its sensors nor its actuators, is left open and poses a structural problem. Closed-form Dirichlet updates require a mean-field factorization between the parameter beliefs and the state–observation blocks, but under exactly that factorization every parameter-posterior contrast collapses, $q ( \theta \mid \cdot ) = q ( \theta )$ , and the novelty term that motivates parameter learning vanishes from the objective.

This paper resolves this problem within a single variational objective. We split θ into an observation block $\theta _ { A }$ , a transition block $\theta _ { B }$ , and an enumerable context $\theta _ { c } ,$ and extend the entropy-corrected formulation to conjugate learning of $\theta _ { A }$ and $\theta _ { B }$ . Our contributions are:

– We show that the channels of the entropy corrections need not be free variational parameters. Substituting them at their closed-form optima, the conditionals of $q ,$ and giving each factor a step-wise parameter prior frozen at the current belief, keeps every sum-product factor belief conjugate and preserves the local parameter–outcome joints that mean-field destroys. Under the resulting Bethe joint the objective is a single scalar over the discrete coordinates.

– We derive the resulting message passing scheme. Four epistemic priors recover the EFE with separate novelty terms for $\theta _ { A }$ and $\theta _ { B }$ and a salience term for $\theta _ { c }$ . Stationarity yields belief propagation on the discrete graph with factor functions augmented with per-entry information-gain weights $e ^ { D _ { \mathrm { K L } } } \geq 1$ The fixed-parameter scheme of [12, 13] is recovered in the point-mass limit.

– We validate the method on two grid POMDPs, in which only a planner that explores the model, finds the safe route. The EFE-planner learns $\theta _ { A }$ and $\theta _ { B }$ and reaches the goal on every episode, whereas SP-MP baselines plateau on parameter learning or take the risky routes. Enumerated EFE reproduces its behaviour, and ablations isolate the novelty terms as the drive.

Section 2 reviews the EFE and the Bethe free energy. Section 3 discusses related work. Section 4 derives the objective and the message passing scheme. Section 5 reports the experiments and Section 6 concludes.

## 2 Background

## 2.1 Active Inference and the Expected Free Energy

We consider an agent that maintains a generative model predicting future observations, states and the consequences of actions over a planning horizon $T .$ For a discrete partially observable Markov decision process (POMDP) with hidden states $^ { x , }$ observations y, actions u and model parameters θ. We write this as a rollout model $\begin{array} { r } { p ( \pmb { y } , \pmb { x } , \pmb { u } , \theta ) = p ( \theta ) p ( x _ { 0 } ) \prod _ { t = 1 } ^ { T } p ( u _ { t } ) p ( y _ { t } \mid x _ { t } , \theta ) p ( x _ { t } \mid x _ { t - 1 } , u _ { t } , \theta ) } \end{array}$ To encode goals, the model is augmented with preference priors $\hat { p } ( x _ { t } ) , \ \hat { p } ( y _ { t } )$ over desired states and observations, which can be understood as exponentiated rewards ${ \hat { p } } \propto \exp ( R )$ [3].

Active inference selects actions by minimizing the Expected Free Energy (EFE), which scores a policy by the cost of its predicted trajectory and decom poses into three interpretable terms [3]. Per time step,

$$
\begin{array} { r } { G ( u _ { t } ) = \underbrace { D _ { \mathrm { K L } } \big [ q ( x _ { t } \mid u _ { t } ) \mid \big \rvert \hat { p } ( x _ { t } ) \big ] } _ { \mathrm { r i s k ~ ( s t a t e s ) } } + \underbrace { { \mathbb { E } } _ { q ( x _ { t } \mid u _ { t } ) } \big [ \mathbb { H } [ p ( y _ { t } \mid x _ { t } ) ] \big ] } _ { \mathrm { a m b i g u i t y } } } \\ { - \underbrace { { \mathbb { E } } _ { p ( y _ { t } \mid x _ { t } ) } q ( x _ { t } \mid u _ { t } ) \big [ D _ { \mathrm { K L } } [ q ( \theta \mid y _ { t } , x _ { t } ) \mid \mid q ( \theta ) ] \big ] } _ { \mathrm { n o v e l t y } } . } \end{array}\tag{1}
$$

Risk penalizes deviation of the predicted states from the preferences, ambiguity penalizes states whose observations are uncertain, and novelty rewards states and observations that are informative about θ. In discrete active inference, policies (sequences of actions) are typically scored with (1) and selected in proportion to exp(−G) [5].

## 2.2 Variational Free Energy and Message Passing on Factor Graphs

Given a generative model $p ,$ variational inference approximates its posterior by minimizing the Variational Free Energy (VFE) over a family of tractable distributions $q \colon$

$$
F [ q ] = D _ { \mathrm { K L } } \big [ q ( { \pmb y } , { \pmb x } , { \pmb u } , { \pmb \theta } ) \big | \big | p ( { \pmb y } , { \pmb x } , { \pmb u } , { \pmb \theta } ) \big ] .\tag{2}
$$

The generative model factorizes into local terms and can be represented as a Forney-style factor graph (FFG) [18], in which nodes $a \in \mathcal V$ represent factors $f _ { a } ,$ edges $i \in \mathcal { E }$ represent variables, and $d _ { i }$ is the degree of edge i. The Bethe approximation constrains $q$ to respect this structure: each node maintains a belief $q _ { a } ( s _ { a } )$ over its adjacent variables and each edge a singleton belief $q _ { i } ( s _ { i } )$ , with

$$
q ( s ) = { \frac { \prod _ { a } q _ { a } ( s _ { a } ) } { \prod _ { i } q _ { i } ( s _ { i } ) ^ { d _ { i } - 1 } } } , \qquad \mathbb { H } _ { \mathrm { B e t h e } } [ q ] = \sum _ { a } \mathbb { H } [ q _ { a } ] - \sum _ { i } ( d _ { i } - 1 ) \mathbb { H } [ q _ { i } ] ,\tag{3}
$$

so that the joint entropy $\mathbb { H } [ q ]$ is approximated by $\mathbb { H } _ { \mathrm { B e t h e } } [ q ]$ and the VFE reduces to the Bethe free energy

$$
F _ { \mathrm { B e t h e } } [ q ] = \sum _ { a \in \mathcal { V } } D _ { \mathrm { K L } } \big [ q _ { a } ( \pmb { s } _ { a } ) \| f _ { a } ( \pmb { s } _ { a } ) \big ] + \sum _ { i \in \mathcal { E } } ( d _ { i } - 1 ) \mathbb { H } [ q _ { i } ( \pmb { s } _ { i } ) ] .\tag{4}
$$

The beliefs must be normalized and locally consistent, so minimizing (4) is a constrained problem whose Lagrangian stationarity conditions yield message passing: sum-product message passing (belief propagation) under the consistency constraints alone, and Variational Message Passing (VMP) when node beliefs are further mean-field factorized [18].

## 3 Related Work

Planning-as-inference and active inference. Active inference inherits the planningas-inference framework [1, 17, 9] and extends it by treating information gain as an intrinsic objective, so that the EFE combines pragmatic and epistemic value [6, 3]. Existing EFE planners employ specialized procedures: policy tree search [4], branching-time recursion [2], dynamic programming [16], or message passing on a generalized free energy [15] with a mutual-information term over future graph segments [8]; none minimizes a single variational objective over the whole graph.

EFE minimization as VFE minimization. Control-as-inference [7, 9] augments $p$ with preference priors $\hat { p } ( x _ { t } )$ and, where used, $\hat { p } ( y _ { t } )$ , giving a preference-biased model $\hat { p }$ and the VFE $F _ { \hat { p } } [ q ] = \mathbb { E } _ { q } [ \log q / \hat { p } ]$ . De Vries et al. [14] (Proposition 1) showed that further augmenting pˆ with epistemic priors $\tilde { p }$ makes this VFE equal to $\mathbb { E } _ { q ( \pmb { u } ) } [ G ( \pmb { u } ) ]$ up to a complexity term. Nuijten et al. [13] showed that the augmented VFE can equivalently be written as $F _ { \hat { p } }$ plus explicit entropy corrections, one conditional-entropy term per prior: $\tilde { p } ( u _ { t } )$ contributes $- \mathbb { H } [ q ( x _ { t } \mid x _ { t - 1 } , u _ { t } ) ]$ and the observation-side priors contribute $+ \mathbb { H } [ q ( y _ { t } \mid x _ { t } , \theta ) ]$ terms. Proper planning over policies requires one further planning correction $+ \mathbb { H } [ q ( u _ { t } \ | \ x _ { t - 1 } ) ]$ introduced by Lázaro-Gredilla et al. [10] as the correction that turns marginal inference into policy optimization [13].

Message passing realizations. The conditional entropies above are ratios of region beliefs and therefore not independent coordinates of the Bethe free energy. Nuijten et al. [12] attached the epistemic priors to a parameter-free model as extra factor nodes and resolved their circular dependence on q by alternating posterior and prior updates. Nuijten et al. [11] introduced region-extended coordinates and a single observation-side channel for $\mathbb { H } [ q ( y _ { t } \mid x _ { t } , \theta ) ]$ ], without the planning correction. Nuijten et al. [13] finally absorbed every correction, including the planning correction, into modified factor functions through channel reparameterization, recovering belief propagation, cross-entropy planning and full EFE-based planning as one family. The key identity of that construction is the variational characterization of conditional entropy (Gibbs’ inequality),

$$
\mathbb H [ q ( y \mid x ) ] = \operatorname* { m i n } _ { r } \mathbb E _ { q ( y , x ) } \big [ - \log r ( y \mid x ) \big ] ,\tag{5}
$$

with equality at the normalized channel $r ( y \mid x ) = q ( y \mid x )$ . Treating each channel as a free variational parameter turns a single factor into a modified factor function, and the corrected Bethe objective is minimized by sum-product updates on those functions, jointly over beliefs and channels.

This paper. Every realization above assumes a single, enumerable $\theta ,$ and carries the channels of (5) as free variational parameters. We instead let θ carry Dirichlet beliefs over the model parameters themselves, and substitute the channels at their closed-form optima, where conjugate factor beliefs supply the parameter– datum coupling that the channels would otherwise have to represent.

## 4 Methodology

## 4.1 Generative Model and Planning Problem

We consider a discrete-time POMDP with hidden states $x _ { t }$ , actions $u _ { t } ,$ observations $y _ { t }$ and an episode-level parameter $\theta = ( \theta _ { A } , \theta _ { B } , \theta _ { c } )$ in three blocks,

$$
p ( \pmb { y } , \pmb { x } , \pmb { u } , \theta ) = p ( \theta _ { A } ) p ( \theta _ { B } ) p ( \theta _ { c } ) p ( x _ { 0 } ) \prod _ { t = 1 } ^ { T } p ( u _ { t } ) f _ { \mathrm { d y n } , t } f _ { \mathrm { o b s } , t } ,\tag{6}
$$

$$
f _ { \mathrm { d y n } , t } = p ( x _ { t } \mid x _ { t - 1 } , u _ { t } , \theta _ { B } , \theta _ { c } ) , \qquad f _ { \mathrm { o b s } , t } = p ( y _ { t } \mid x _ { t } , \theta _ { A } , \theta _ { c } ) .
$$

The context $\theta _ { c } \ \in \ \{ 1 , \ldots , K \}$ is Categorical and selects which of K observation and transition matrices apply; $\theta _ { A } , \theta _ { B }$ represent those matrices, $p ( y _ { t } \mid$ $x _ { t } , \theta _ { A } , \theta _ { c } ) = \theta _ { A } ( y _ { t } \ | \ x _ { t } , \theta _ { c } )$ and $p ( x _ { t } \mid x _ { t - 1 } , u _ { t } , \theta _ { B } , \theta _ { c } ) = \theta _ { B } ( x _ { t } \mid x _ { t - 1 } , u _ { t } , \theta _ { c } )$ ， with one Dirichlet prior per column, $\begin{array} { r } { p ( \theta _ { A } ) ~ = ~ \prod _ { x , \theta _ { c } } \operatorname { D i r } ( \alpha _ { A } ( \cdot ~ | ~ x , \theta _ { c } ) ) } \end{array}$ and $\begin{array} { r } { p ( \theta _ { B } ) = \prod _ { x , u , \theta _ { c } } \operatorname { D i r } ( \alpha _ { B } ( \cdot \ : | \ : x , u , \theta _ { c } ) ) } \end{array}$ . Unknown $\theta _ { A } , \theta _ { B }$ means unlearned matrices; unknown $\theta _ { c }$ means known matrices but an unknown context, the enumerable model variable of [13]. One draw of θ is shared by all T factors. Goals enter as preference priors, giving the preference-biased model $\begin{array} { r } { \hat { p } = p \prod _ { t } \hat { p } _ { x } ( x _ { t } ) \hat { p } _ { y } ( y _ { t } ) } \end{array}$ (Figure 1) with VFE $F _ { \hat { p } } [ q ] = \mathbb { E } _ { q } [ \log q / { \hat { p } } ] \ ( \hat { p }$ unnormalized, so a divergence up to a constant).

The agent scores an action by the per-step EFE,

$$
\begin{array} { r l } & { G ( u _ { t } ) = \mathbb { E } _ { q ( \cdot | u _ { t } ) } \Big [ \log \frac { q ( x _ { t } \mid x _ { t - 1 } , u _ { t } ) } { \hat { p } _ { x } ( x _ { t } ) \hat { p } _ { y } ( y _ { t } ) p ( y _ { t } \mid x _ { t } , \theta _ { A } , \theta _ { c } ) } } \\ & { \qquad + \log \frac { q ( \theta _ { A } , \theta _ { c } \mid x _ { t } ) } { q ( \theta _ { A } , \theta _ { c } \mid y _ { t } , x _ { t } ) } + \log \frac { q ( \theta _ { B } \mid x _ { t - 1 } , \theta _ { c } ) } { q ( \theta _ { B } \mid x _ { t } , x _ { t - 1 } , u _ { t } , \theta _ { c } ) } \Big ] , } \end{array}\tag{7}
$$

where $q ( \cdot \mathrm { ~ \bf ~ \mathscr ~ { ~ u ~ } ~ } )$ abbreviates $q ( y _ { t } , x _ { t } , x _ { t - 1 } , \theta ~ \mid ~ u _ { t } )$ . The first term is risk plus ambiguity $\mathbb { H } _ { q } [ y _ { t } \mid x _ { t } , \theta _ { A } , \theta _ { c } ]$ . The remaining terms contrast each block’s cavity, the belief before the imagined datum, with its posterior, and are information gains. By the chain rule they split into the salience $\mathcal { T } _ { c , t } = I ( \theta _ { c } ; y _ { t } \mid x _ { t } )$ , which rewards resolving the context, and the novelties $\mathcal { T } _ { A , t } ~ = ~ I ( \theta _ { A } ; y _ { t } ~ \vert ~ x _ { t } , \theta _ { c } )$ and $\mathcal { T } _ { B , t } = I ( \theta _ { B } ; x _ { t } , u _ { t } \ | \ x _ { t - 1 } , \theta _ { c } )$ , which reward learning the matrices. The planning problem is to find the policy $q ( u _ { t } \mid x _ { t - 1 } )$ that minimizes $\begin{array} { r } { \sum _ { t } \mathbb { E } _ { q ( u _ { t } ) } [ G ( u _ { t } ) ] } \end{array}$

Two obstacles separate (7) from a message-passing algorithm. First, G contains the agent’s own conditionals inside the logarithm, so it is not a VFE against any fixed model. Second, the novelties are mutual informations of local joints such as $q ( y _ { t } , x _ { t } , \theta _ { c } , \theta _ { A } ) ;$ a family that factorizes the Dirichlet block away from the imagined datum, $q ( y _ { t } , x _ { t } , \theta _ { c } ) q ( \theta _ { A } )$ , has $\mathcal { T } _ { A , t } \equiv 0$ , yet this is exactly the factorization that structured mean-field uses to obtain Dirichlet conjugacy. Section 4.2 fixes the first, Section 4.3 the second.

![](images/c6add050735decfe10db1e41315337a677d42a60f8202b9983b91a36a813060c.jpg)  
Fig. 1. Frozen-parameter factor graph (Section 4.3). Squares: factors; edges: variables; =: equality nodes; filled squares: clamped data; pˆ: preferences. Left: the real slice carries the shared $\theta _ { A } , \theta _ { B }$ and turns $p ( x _ { \mathrm { p r i o r } } )$ , u<sub>prev</sub>, y<sub>0</sub> into the planning prior $p ( x _ { 0 } )$ Right: over the horizon each factor $f ^ { * }$ (factor functions (17)–(18)) owns a copy $\theta _ { A } ^ { ( t ) } , \theta _ { B } ^ { ( t ) }$ with its own degree-two prior node (open, gray) frozen at Dir(α<sub>A</sub>) resp. $\operatorname { D i r } ( \alpha _ { B } ) ;$ ; only $\theta _ { c }$ is shared, through equality nodes.

## 4.2 From Expected to Variational Free Energy

Following [14, 13], we keep the objective and change the model: $\hat { p }$ is augmented with epistemic priors, one for every factor of (7) that is not already part of ${ \hat { p } } ,$

$$
\begin{array} { r } { \tilde { p } ( u _ { t } ) \propto \exp \big ( - \mathbb { E } _ { q ( x _ { t } , x _ { t - 1 } \mid u _ { t } ) } \log q ( x _ { t } \mid x _ { t - 1 } , u _ { t } ) \big ) , } \end{array}\tag{8}
$$

$$
\tilde { p } ( x _ { t } ) \propto \exp \big ( - \mathbb { E } _ { q ( \theta _ { A } , \theta _ { c } | x _ { t } ) } h [ q ( y _ { t } \mid x _ { t } , \theta _ { A } , \theta _ { c } ) ] \big ) ,\tag{9}
$$

$$
\tilde { p } ( y _ { t } , x _ { t } ) \propto \exp { D _ { \mathrm { K L } } [ q ( \theta _ { A } , \theta _ { c } \mid y _ { t } , x _ { t } ) \mid \mid q ( \theta _ { A } , \theta _ { c } \mid x _ { t } ) ] } ,\tag{10}
$$

$$
\tilde { p } ( x _ { t } , x _ { t - 1 } , u _ { t } , \theta _ { c } ) \propto \exp D _ { \mathrm { K L } } [ q ( \theta _ { B } \mid x _ { t } , x _ { t - 1 } , u _ { t } , \theta _ { c } ) \parallel q ( \theta _ { B } \mid x _ { t - 1 } , \theta _ { c } ) ] ,\tag{11}
$$

where $h [ q ( \cdot \mid x ) ]$ denotes the entropy of a conditional as a function of $x$ and $\mathbb { H } _ { q } [ \cdot \mid x ] = \mathbb { E } _ { q ( x ) } h [ q ( \cdot \mid x ) ]$ the conditional entropy. Each prior is the exponential of a local functional of $q , \mathrm { s o } - \mathbb { E } _ { q } [ \log \tilde { p } ]$ is closed-form.

Corollary 1 (Epistemic priors for three parameter blocks; cf. [14], Prop. 1). Let p˜ be pˆ augmented with the priors $( 8 ) - ( 1 1 ) \ f o r \ t = 1 , . . . , T$ . Then $\begin{array} { r } { F _ { \tilde { p } } [ q ] = \sum _ { t } \mathbb { E } _ { q ( u _ { t } ) } [ G ( u _ { t } ) ] + B [ q ] } \end{array}$ , where $B [ q ] = D _ { \mathrm { K L } } [ q \| p ] \geq 0$ is the complexity of q relative to (6).

## Proof. See Appendix A.1.

The prior (11) is the transition-novelty prior, new relative to [13]. Proper planning over policies further requires the planning correction $+ \mathbb { H } _ { q } [ u _ { t } \mid x _ { t - 1 } ] \left[ 1 0 \right.$ 13], which cancels the reward that the VFE pays for action entropy. Evaluating $- \mathbb { E } _ { q } [ \log \tilde { p } ]$ for each prior (Appendix A.2) writes the EFE-based planning objective as the base VFE plus a per-step entropy correction, $F _ { T } [ q ] \ =$ $\begin{array} { r } { F _ { \hat { p } } [ q ] + \sum _ { t = 1 } ^ { T } \varDelta F _ { t } [ q ] } \end{array}$ (the subscript is the horizon) with

$$
\Delta F _ { t } [ q ] = { \mathbb H } _ { q } [ y _ { t } \mid x _ { t } ] - 2 \big ( { \mathcal Z } _ { A , t } + { \mathcal Z } _ { c , t } \big ) - { \mathbb H } _ { q } [ x _ { t } \mid x _ { t - 1 } , u _ { t } ] - { \mathcal Z } _ { B , t } + { \mathbb H } _ { q } [ u _ { t } \mid x _ { t - 1 } ] .\tag{12}
$$

The factor two is no double count: the ambiguity prior (9) pays one unit of each observation-side gain through its θ-coupling and the novelty prior (10) a second, while the transition side has no ambiguity term (Appendix A.2). Message passing on $\tilde { p }$ now applies, provided that q retains the local joints that make the information gains in (12) nonzero.

## 4.3 Variational Family: Sum-Product Beliefs on the Frozen-Parameter Graph

Frozen-parameter graph. The agent plans with its current parameter posterior, so imagined observations in the rollout must not update it. On the graph of (6) this fails: the sum-product message from $f _ { \mathrm { o b s } , t }$ to a shared $\theta _ { A }$ node, $\begin{array} { r } { \sum _ { y , x , \theta _ { c } } p ( y ) } \end{array}$ $x , \theta _ { A } , \theta _ { c } ) \mu _ { y } \mu _ { x } \mu _ { \theta }$ with $\mu$ the messages into the factor, is a mixture of monomials rather than a Dirichlet, so conjugacy is lost and imagined observations mix across the horizon. Both problems are resolved by a step-wise copy $\theta _ { A } ^ { ( t ) }$ per observation factor with its own prior node $p ( { \theta } _ { A } ^ { ( t ) } ) : = q _ { \theta _ { A } } = \mathrm { D i r } ( \theta _ { A } ^ { ( t ) } \ | \ \alpha _ { A } )$ carrying the current real-data belief, and likewise $\theta _ { B } ^ { ( t ) }$ per $f _ { \mathrm { d y n } , t }$ (Figure 1; $q _ { \theta _ { A } } , q _ { \theta _ { B } }$ denote these beliefs, as opposed to the copy marginals). Setting a prior node to a belief makes $\tilde { p }$ a planning-time model, rebuilt each environment step, so the objective is single per planning call. Each copy has degree two, so the message into its factor is exactly $\operatorname { D i r } ( \alpha _ { A } )$ and the return message reaches no other factor; the price is that novelty does not decay along the horizon, which replanning bounds (Appendix B.4). The finite $\theta _ { c }$ is not copied: it is conjugate to any factor, and sharing it lets an early cue inform a distant step.

Factor beliefs are conjugate in closed form. A Bethe factor belief is the factor times its incoming messages, q<sub>obs</sub> ${ \bf \Psi } _ { , t } \propto p ( y _ { t } \mid x _ { t } , \theta _ { A } ^ { ( t ) } , \theta _ { c } ) \mu _ { y _ { t } } \mu _ { x _ { t } } \mu _ { \theta _ { c } } \operatorname { D i r } ( \theta _ { A } ^ { ( t ) } \mid \alpha _ { A } )$ For each context the likelihood is a monomial in one column of $\theta _ { A }$ . Thus conjugacy applies entrywise, $\theta _ { A } ( y ~ \mid ~ x , \theta _ { c } ) \operatorname { D i r } ( \theta _ { A } ~ \mid ~ \alpha _ { A } ) ~ = ~ m _ { A } ( y ~ \mid ~ x , \theta _ { c } ) \operatorname { D i r } ( \theta _ { A } ~$ $\alpha _ { A } + e _ { y x { \theta } _ { c } } )$ , with $e _ { y x \theta . }$ one count on entry y of column $( x , \theta _ { c } )$ and $m _ { A } = \alpha _ { A } / \alpha _ { A , 0 }$ the posterior predictive likelihood (Appendix A.3). Hence, exactly,

$$
q _ { \mathrm { o b s } , t } ( y _ { t } , x _ { t } , \theta _ { c } , \theta _ { A } ^ { ( t ) } ) = q _ { y x , t } ( y _ { t } , x _ { t } , \theta _ { c } ) \underbrace { \operatorname { D i r } ( \theta _ { A } ^ { ( t ) } \mid \alpha _ { A } + e _ { y _ { t } x _ { t } \theta _ { c } } ) } _ { q _ { \mathrm { o b s } , t } ( \theta _ { A } \mid x _ { t } , y _ { t } , \theta _ { c } ) } ,\tag{13}
$$

with discrete marginal $\begin{array} { r } { q _ { y x , t } \ : = \ \int q _ { \mathrm { o b s } , t } d \theta _ { A } \ \propto \ m _ { A } \mu _ { y _ { t } } \mu _ { x _ { t } } \mu _ { \theta _ { c } } } \end{array}$ , and $q _ { \mathrm { d y n } , t } ~ =$ $q _ { \mathrm { t r i p } , t } ( x _ { t } , x _ { t - 1 } , u _ { t } , \theta _ { c } ) \operatorname { D i r } ( \theta _ { B } ~ \mid ~ \alpha _ { B } + e _ { x _ { t } x _ { t - 1 } u _ { t } \theta _ { c } } )$ with the predictive likelihood m<sub>B</sub> of $\alpha _ { B }$

Coordinates and constraints. The free coordinates are the Bethe beliefs of the discrete graph over $\{ y _ { 1 : T } , x _ { 0 : T } , u _ { 1 : T } , \theta _ { c } \}$ : the block beliefs $q _ { y x , t } , q _ { \mathrm { t r i p } , t }$ and the singletons $q ( x _ { t } ) , q ( y _ { t } ) , q ( u _ { t } ) , q ( \theta _ { c } )$ , subject to normalization and factor-to-singleton consistency, with $q _ { \theta _ { A } } = \operatorname { D i r } ( \alpha _ { A } )$ and $q _ { \theta _ { B } } = \operatorname { D i r } ( \alpha _ { B } )$ , held fixed during planning. The $\theta _ { A ^ { - } }$ and θ -conditionals of the factor beliefs are not coordinates: any other choice raises the free energy of the region, so they are substituted out at the closed-form optima of (13) (Appendix A.3, Lemma 3). There are no channel constraints either: the channels of (5) are not coordinates but definitions $r = q ( \cdot | \cdot )$

Cavities and per-entry information gains. Novelty contrasts the posterior $q _ { \mathrm { o b s } , t } ( \theta _ { A } \mid x _ { t } , y _ { t } , \theta _ { c } )$ with the cavity $q _ { \mathrm { o b s } , t } ( \theta _ { A } \mid x _ { t } , \theta _ { c } )$ , the same belief before the outcome is known,

$$
q _ { \mathrm { o b s } , t } ( \theta _ { A } \mid x _ { t } , \theta _ { c } ) = \sum _ { y } q _ { y x , t } ( y \mid x _ { t } , \theta _ { c } ) \operatorname { D i r } ( \alpha _ { A } + e _ { y x _ { t } \theta _ { c } } ) ,\tag{14}
$$

a finite Dirichlet mixture, the only non-conjugate object in the construction. We project it onto $\mathrm { D i r } ( \alpha _ { A } + \bar { n } _ { x _ { t } \theta _ { c } } )$ with the expected fractional counts $\bar { n } _ { x _ { t } \theta _ { c } } ( y ) : =$ $q _ { y x , t } ( y \mid x _ { t } , \theta _ { c } )$ . It matches the mixture mean, is exact for one-hot $q _ { y x , t } ( \cdot \mid x _ { t } , \theta _ { c } )$ and is overconfident by $O ( 1 / \alpha _ { A , 0 } )$ (Appendix A.4). The per-entry gains are then closed-form Dirichlet–Dirichlet divergences,

$$
{ \varDelta } _ { A , t } ( x _ { t } , y _ { t } , \theta _ { c } ) = { \cal D } _ { \mathrm { K L } } [ q _ { \mathrm { o b s } , t } ( \theta _ { A } \mid x _ { t } , y _ { t } , \theta _ { c } ) \| q _ { \mathrm { o b s } , t } ( \theta _ { A } \mid x _ { t } , \theta _ { c } ) ]\tag{15}
$$

with $\mathcal { T } _ { A , t } = \mathbb { E } _ { q _ { y x , t } \left[ \varDelta _ { A , t } \right] }$ , and symmetrically $\varDelta _ { B , t } ( x _ { t } , x _ { t - 1 } , u _ { t } , \theta _ { c } )$ with cavity $q _ { \mathrm { d y n } , t } ( \theta _ { B } \mid x _ { t - 1 } , \theta _ { c } )$ and $\mathbb { Z } _ { B , t } = \mathbb { E } _ { q _ { \mathrm { t r i p } , t } } [ \varDelta _ { B , t } ] .$ Salience needs neither: $\theta _ { c }$ is a stored axis, so $\mathcal { T } _ { c , t } = I _ { q _ { y x , t } } ( y _ { t } ; \theta _ { c } \mid x _ { t } )$ is a mutual information of the finite array.

## 4.4 The Planning Objective in Closed Form

Combining the factor beliefs along their separators gives the Bethe joint of the frozen graph, $q = \prod _ { i = 1 }$ <sub>t</sub> q<sub>obs,</sub> $\begin{array} { r l r } { \mathrm { ~ } } & { { } } & { q _ { \mathrm { d y n } , t } \Big / \left( q ( \theta _ { c } ) ^ { 2 T - 1 } \prod _ { t < T } q ( x _ { t } ) ^ { 2 } q ( x _ { T } ) \right) } \end{array}$ , with exponents from the variable degrees (Appendix A.5). Its θ-free conditionals are the block conditionals and its $\theta _ { A } , \theta _ { B }$ -conditionals the posteriors and cavities of (13)–(14), so every object in (8)–(11) is well-defined under $q .$

Proposition 1 (Closed-form planning objective). Under the Bethe joint q: (i) in the base free energy the parameter entropy of each factor belief cancels per entry against its parameter energy, leaving the Bethe free energy of the discrete graph with predictive energies − log m<sub>A</sub>, − log m<sub>B</sub> $( e q . ~ ( 2 9 ) ) ; ~ ( i i )$ the corrections (12) hold unchanged, with $\mathcal { T } _ { A , t } , \mathcal { T } _ { B , t }$ as in (15) and $\mathcal { T } _ { c , t } = I _ { q _ { y x , t } } ( y _ { t } ; \theta _ { c } \mid x _ { t } )$ ; (iii) hence the planning objective is the closed-form functional

$$
\begin{array} { r l } & { { \cal F } _ { T } [ q ] = \sum _ { t = 1 } ^ { T } \left( D _ { \mathrm { K L } } \left[ q _ { y x , t } \middle | \middle | m _ { A } \hat { p } _ { y } \right] + D _ { \mathrm { K L } } \left[ q _ { \mathrm { t r i p } , t } \middle | \middle | m _ { B } p ( u _ { t } ) \hat { p } _ { x } \right] \right) + \mathcal { C } [ q ] } \\ & { ~ + \displaystyle \sum _ { t = 1 } ^ { T } \left( \mathbb { H } _ { q _ { y x , t } } [ y _ { t } \ | x _ { t } ] - 2 \mathbb { E } _ { q _ { y x , t } } [ { \Delta } _ { A , t } ] - 2 I _ { q _ { y x , t } } ( y _ { t } ; \theta _ { c } \ | \ x _ { t } ) \right. } \\ & { ~ \left. ~ - \mathbb { H } _ { q _ { \mathrm { t r i p } , t } } [ x _ { t } \ | \ x _ { t - 1 } , u _ { t } ] - \mathbb { E } _ { q _ { \mathrm { t r i p } , t } } [ { \Delta } _ { B , t } ] + \mathbb { H } _ { q _ { \mathrm { t r i p } , t } } [ u _ { t } \ | \ x _ { t - 1 } ] \right) } \end{array}\tag{16}
$$

over the discrete coordinates alone, where $\mathcal { C } [ q ]$ collects the Bethe counting terms and the $x _ { 0 } , \theta _ { c }$ priors.

Algorithm 1 Conjugate planning and learning. Inputs: counts $\alpha _ { A } , \alpha _ { B } ,$   
beliefs $q ( \theta _ { c } ) , q ( x _ { 0 } )$ , horizon $T _ { \cdot }$ , damping $\eta ,$ , tolerance ϵ, sweep budget N.   
1: repeat ▷ one environment step   
2: m $A ,$ m<sub>B</sub> and one-count posteriors $\operatorname { D i r } ( \alpha + e )$ from $\alpha _ { A } , \alpha _ { B }$ ▷ once per call   
3: $r _ { t } \gets$ uniform, $\bar { n } $ prior predictive, $\mu _ { \theta _ { c }  \cdot }  q ( \theta _ { c } ) , \mu _ { x _ { 0 } }  q ( x _ { 0 } )$   
4: for $k = 1 , \ldots , N$ do   
5: factor functions $\bar { g } _ { \mathrm { o b s } , t } , \bar { g } _ { \mathrm { d y n } , t } ;$ forward–backward pass ▷ (17)–(18)   
6: refresh $\textstyle r _ { t } \gets q ( \cdot \mid \cdot )$ , n¯ and the weights $\varDelta _ { A , t } , \varDelta _ { B , t } .$ , damped by η   
7: evaluate $F _ { T } ^ { ( k ) } \ ;$ break if $| F _ { T } ^ { ( k ) } - F _ { T } ^ { \bar { ( k - 1 ) } } | < \epsilon$ ▷ (16)   
8: end for   
9: execute $u _ { 1 } \in$ arg max $q ( u _ { 1 } ) ;$ observe $_ { y 1 } ;$ filter $q ( x _ { 1 } ) ; q ( \theta _ { c } )$ by exact Bayes   
10: $\alpha _ { A } \mathrel { + } \mathbin { = } \mathbb { E } [ e _ { y _ { 1 } x _ { 1 } \theta _ { c } } ] , \alpha _ { B } \mathrel { + } \mathop { = } \mathbb { E } [ e _ { x _ { 1 } x _ { 0 } u _ { 1 } \theta _ { c } } ]$ under $q ( x _ { 1 } , x _ { 0 } ) q ( \theta _ { c } ) ; q ( x _ { 0 } )  q ( x _ { 1 } )$   
11: until episode ends

Proof. See Appendix A.5.

## 4.5 Stationarity Conditions and Message Passing

Since $F _ { T }$ has Bethe form over the discrete coordinates, its stationarity conditions under the normalization and consistency multipliers (Appendix B.1) yield sumproduct message updates. Two facts make the variation local and term-by-term (Appendix B.2): entropies of conditionals of a normalized joint diferentiate as if the logarithm were frozen (Lemma 4), and the dependence of the cavity on $q _ { y x , t }$ contributes nothing, $\delta \mathcal { T } _ { A , t } / \delta q _ { y x , t } ( y , x , \theta _ { c } ) = \varDelta _ { A , t } ( x , y , \theta _ { c } )$ (Lemma 5).

Proposition 2 (Modified factor functions). At stationarity $q _ { y x , t } ^ { * } \propto \bar { g } _ { \mathrm { o b s } , t }$ and $q _ { \mathrm { t r i p } , t } ^ { * } \propto \bar { g } _ { \mathrm { d y n } , t }$ up to the singleton multipliers fixed in Appendix B.2, with the factor functions

$$
\bar { g } _ { \mathrm { o b s } , t } ( y _ { t } , x _ { t } , \theta _ { c } ) = m _ { A } ( y _ { t } \mid x _ { t } , \theta _ { c } ) \xrightarrow { r _ { t } ( y _ { t } \mid x _ { t } , \theta _ { c } ) ^ { 2 } } e ^ { 2 \varDelta _ { A , t } ( x _ { t } , y _ { t } , \theta _ { c } ) } ,\tag{17}
$$

$$
{ \bar { g } } _ { \mathrm { d y n } , t } ( x _ { t } , x _ { t - 1 } , u _ { t } , \theta _ { c } ) = m _ { B } ( x _ { t } \mid x _ { t - 1 } , u _ { t } , \theta _ { c } ) { \frac { r _ { t } ( u _ { t } \mid x _ { t - 1 } ) } { r _ { t } ( x _ { t } \mid x _ { t - 1 } , u _ { t } ) } } e ^ { \varDelta _ { B , t } } ,\tag{18}
$$

evaluated at the self-consistent conditionals $r _ { t } ( \cdot \mid \cdot ) = q ^ { * } ( \cdot \mid \cdot )$ of the block beliefs.

Proof. See Appendix B.2, which also fixes the multipliers from the singleton stationarity conditions.

The factor functions (17)–(18) are the pair (13a)–(13b) of [13] with the point likelihood replaced by the predictive likelihood and an information-gain weight $e ^ { 2 \Delta _ { A } } , e ^ { \Delta _ { B } } \geq 1$ appended. The weights are not imposed, but emerge from the variation of the information gains.

Algorithm, learning and the fixed-parameter limit. With $r _ { t }$ lagged by one sweep, as for the channels of [13], Proposition 2 is a standard Bethe belief propagation problem on the discrete graph over $\{ y _ { t } , x _ { t } , u _ { t } , \theta _ { c } \}$ with factor functions (17)– (18). Algorithm 1 solves it with closed-form updates (Appendix B.3–B.4); the copies $\theta _ { A } ^ { ( t ) } , \theta _ { B } ^ { ( t ) }$ exchange no messages, so the cost is independent of the size of the Dirichlet parameter space. Because $\tilde { p }$ depends on $q$ and the lagged $r _ { t }$ is self-consistent only at convergence, $F _ { T }$ is monitored rather than assumed to descend. The count update is the plain conjugate one: exact stationarity of $F _ { T }$ in $q _ { \theta _ { A } }$ adds only a centered score that shifts no count mass (Remark 2). As $q _ { \theta _ { A } } , q _ { \theta _ { B } }  \delta _ { \theta ^ { * } }$ the weights vanish, $m \to p ( \cdot \mid \theta ^ { * } , \theta _ { c } )$ , and (17)–(18) reduce to the fixed-parameter EFE factor functions of [12, 13] with their enumerable θ read as $\theta _ { c } ; K = 1$ removes the context (Section 5).

## 5 Experiments

We test whether the information-gain corrections of (16) produce exploration that a preference-driven planner does not, on two grid POMDPs where knowing the model changes the route and exploration is survivable. Everything outside the objective is shared: the conjugate graph of Section 4 with $K = 1$ , the exact filter and count update of Algorithm 1, a terminal preference on the goal, and identical horizons, sweeps and flat Dirichlet priors on the unknown block (details in Appendix C).<sup>1</sup>

Methods. SP-EFE is the full scheme with the factor functions (17)–(18). SP-Pref-Plan keeps only the planning correction $r _ { t } ( u _ { t } \mid x _ { t - 1 } )$ (uniform channels, $\varDelta _ { A } = \varDelta _ { B } = 0 )$ . SP-Pref drops the planning correction too. Both are blind to $m _ { A }$ in planning, since their observation-to-state message is $\begin{array} { r } { \sum _ { y } m _ { A } ( y \mid x ) = 1 } \end{array}$ and reach an unknown transition only through posterior-mean optimism. Four ablations drop terms of (16) from SP-EFE: $\mathrm { S P \mathrm { - } E F E ^ { - \mathrm { d y n } } }$ (no dynamics channel), $\mathrm { S P - E F E } ^ { - \varDelta _ { B } }$ , and $\mathrm { S P - E F E } ^ { - \varDelta _ { A } - \varDelta _ { B } }$ with and without the channel (ambiguity, no novelty). Random acts uniformly while learning; EFE enumerates the $4 ^ { \overset { \cdot } { H } }$ policies under the exact (7).

Learning the observation model (fog-walls). In the fog-walls maze (Figure 2a) the direct 4-step route crosses a drift cell and an ambiguous fog block, whereas the 8-step top row identifies every state; walls make mislocalization cost bumps rather than the episode. The top row pays of only for a planner that knows it is unambiguous: with A, B known, SP-EFE takes it on 30/30 seeds, whereas SP-Pref-Plan, whose route is A-independent, enters the fog and scores 18/30 (Appendix C.2). When learning A (Figure 2b– $^ { - } \mathrm { { c } , }$ Table 1), SP-EFE reaches a goal rate of 1.00 from episode ≈30 on every seed, while SP-Pref-Plan plateaus at 0.81 and SP-Pref at 0.48 with 1.4–1.9× the matrix error: by construction, learning cannot change their route. The curve of SP-EFE is a novelty phase followed by a switch: under the flat prior every cell is equally novel, so the agent enters the fog. Once $\mathbb { H } [ m _ { A } ( \cdot \mid x ) ]$ is maximal in the fog and near zero elsewhere, ambiguity reroutes the plan along a deterministic row and the diminished $\varDelta _ { A }$ no longer pulls it back.

<sup>1</sup> Code: https://github.com/skoghoern/sp-efe-parameter-learning

![](images/cec9960de813b9dcc8cd8c3bc23ce31cd6ba74568fc0216c1b144378d7d37697.jpg)  
Fig. 2. Top (a–c): learning $\theta _ { A }$ on fog-walls (B known, $H = 8 ,$ 5 seeds); bottom (d–f): learning $\theta _ { B }$ on the unmapped detour (identity A, $H = 7 , T = 8 ,$ 3 seeds). Curves are trailing means with a ±1 std band over seeds; the glyph legend gives the cell and route vocabulary of (a) and (d).

Learning the transition model (unmapped detour). The unmapped detour (Figure 2d) caps a planner that never learns at $p = 0 . 5$ , and optimism is priced out by design: a flat column spreads the successor over all $N = 2 0$ cells, which reaches the goal within the horizon with probability $0 . 2 6 5 < p ,$ so under its own objective SP-Pref-Plan should gamble. Table 1 reports three scenarios. With B known, every SP-EFE variant and SP-Pref-Plan take the detour on 300/300 episodes, so learning is suficient; SP-Pref gambles even with a perfect model (Appendix C.2). With B frozen at the prior, SP-Pref-Plan sits at 146 ≈ p and never opens a door, so learning is necessary. When learning B, SP-EFE takes the detour from episode 1 and never leaves it, whereas SP-Pref-Plan gambles on all 300 episodes and $m _ { B }$ ends equal to its prior. SP-Pref walks the detour only while the smeared doors still pay its action-entropy bonus, then gambles for the remaining 275 episodes: a better model removes the only term that steered it to the safe route. Enumerated EFE takes the detour on 100/100 episodes, so the exploration is a property of the objective, not of the message passing.

Ablation. The lower block of Table 1 removes terms of (16). Novelty is the drive. Without $\varDelta _ { A }$ , the channels-only planner keeps the ambiguity penalty but has nothing that rewards resolving it; from a flat prior every unvisited cell is maximally ambiguous (log 25 nats per step against a preference of 4), and the agent bumps into the wall beside the start on all 500 episodes. Without $\varDelta _ { B }$ , the learned-B score is carried by the dynamics channel, which the frozen column exposes as entropy-following. The dynamics channel is dispensable. Removing it changes nothing on fog-walls and costs two episodes on the detour; at its fixed

Table 1. Results and ablation. Fog-walls, learning $\theta _ { A }$ (100 episodes, 5 seeds): goals, goals in the last 20, final matrix error. Unmapped detour, learning $\theta _ { B }$ (300 episodes, 3 seeds): goals with B known, frozen at the prior, or learned, and learned-B goals via the detour. Lower block: ablations of (16). <sup>†</sup>Entropy-following: with frozen flat doors, the most uncertain column and the detour coincide.
<table><tr><td rowspan="2">Planner</td><td colspan="3">fog-walls, learn  $\theta _ { A }$ </td><td rowspan="2">unmapped detour, learn</td><td colspan="4"> $\theta _ { B }$ </td></tr><tr><td>goals/100</td><td>last 20</td><td> $\operatorname { e r r } A$ </td><td>known</td><td>frozen</td><td>learned</td><td>detour</td></tr><tr><td>SP-EFE</td><td> ${ \bf 9 5 . 8 \pm 0 . 7 }$ </td><td>20.0</td><td>0.026</td><td>300</td><td></td><td>300†</td><td>300</td><td>300</td></tr><tr><td>SP-Pref-Plan</td><td> $8 1 . 4 \pm 4 . 8$ </td><td>16.2</td><td>0.037</td><td></td><td>300</td><td>146</td><td> $1 4 6 \pm 7$ </td><td>0</td></tr><tr><td>SP-Pref</td><td> $4 7 . 4 \pm 3 . 1$ </td><td>9.6</td><td>0.049</td><td></td><td>146</td><td>300†</td><td> $1 5 8 \pm 8$ </td><td>25</td></tr><tr><td>Random</td><td> $1 . 0 \pm 0 . 9$ </td><td>0.2</td><td>0.038</td><td></td><td>1</td><td>1</td><td> $1 . 3 \pm 0 . 5$ </td><td>0</td></tr><tr><td> $\mathrm { S P \mathrm { - E F E ^ { - d y n } } }$ </td><td> $9 4 . 4 \pm 2 . 0$ </td><td>20.0</td><td>0.026</td><td></td><td>300</td><td>0</td><td>298</td><td>298</td></tr><tr><td> $\mathrm { S P - E F E } ^ { - \varDelta _ { B } }$ </td><td> $9 4 . 4 \pm 2 . 2 $ </td><td>20.0</td><td>0.025</td><td></td><td>300</td><td>300†</td><td>300</td><td>300</td></tr><tr><td> $\mathrm { S P - E F E } ^ { - \varDelta _ { A } - \varDelta _ { B } }$ </td><td> $0 . 0 \pm 0 . 0$ </td><td>0.0</td><td>0.071</td><td></td><td>300</td><td>300†</td><td>300</td><td>300</td></tr><tr><td> $\mathrm { S P - E F E } ^ { - \varDelta } A ^ { - \varDelta } B ^ { - \mathrm { d y n } }$ </td><td> $0 . 0 \pm 0 . 0$ </td><td>0.0</td><td>0.071</td><td></td><td>300</td><td>146</td><td> $1 4 6 \pm 7$ </td><td>0</td></tr></table>

point the channel cancels $m _ { B }$ on its support, so a smeared learned $m _ { B }$ loses the exploitation signal.

## 6 Conclusion

This paper extended entropy-corrected EFE-based planning [13] from inference over a finite set of known models to conjugate learning of the observation and transition matrices, as the minimization of a single variational objective. Frozen per-factor priors keep every factor belief conjugate and preserve the parameter– datum joint that mean-field destroys. Thus the parameter side collapses into predictive likelihoods, while the information gains survive as weights on the factor functions. On two grid POMDPs those weights are the drive: the scheme learns $\theta _ { A }$ or $\theta _ { B }$ and settles on the safe route where SP-MP with the same inference update plateaus or gambles.

Limitations and future work. The empirical advantage has conditions, all visible in the designs of Section 5: exploration must be survivable, the unknown region small enough for novelty to satiate, and model knowledge must change the route, since greedy optimism already wins on a unique-path task with a hard deadline. The objective is non-convex and the factor functions are built from lagged factors, so damped descent need not be monotone and $F _ { T }$ measures deviations without certifying convergence. Future work includes score-corrected learning, and sophisticated planning beyond the frozen parameter priors.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

[1] Attias, H.: Planning by probabilistic inference. In: International Workshop on Artificial Intelligence and Statistics (AISTATS). pp. 9–16. PMLR (2003)

[2] Champion, T., Da Costa, L., Bowman, H., Grześ, M.: Branching time active inference: The theory and its generality. Neural Networks 151, 295–316 (2022). https://doi.org/10.1016/j.neunet.2022.03.036

[3] Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., Friston, K.: Active inference on discrete state-spaces: A synthesis. Journal of Mathematical Psychology 99, 102447 (Dec 2020). https://doi.org/10.1016/j.jmp.2020.102 447, http://dx.doi.org/10.1016/j.jmp.2020.102447

[4] Friston, K., Da Costa, L., Hafner, D., Hesp, C., Parr, T.: Sophisticated inference. Neural Computation 33(3), 713–763 (2021). https://doi.org/10 .1162/neco\_a\_01351

[5] Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., O’Doherty, J., Pezzulo, G.: Active inference and learning. Neuroscience & Biobehavioral Reviews 68, 862–879 (2016). https://doi.org/10.1016/j.neubiorev.2016.06. 022

[6] Friston, K., Rigoli, F., Ognibene, D., Mathys, C., Fitzgerald, T., Pezzulo, G.: Active inference and epistemic value. Cognitive Neuroscience 6(4), 187– 214 (2015). https://doi.org/10.1080/17588928.2015.1020053

[7] Kappen, H.J., Gómez, V., Opper, M.: Optimal control as a graphical model inference problem. Machine Learning 87(2), 159–182 (2012). https://doi.or g/10.1007/s10994-012-5278-7

[8] van de Laar, T., Koudahl, M., de Vries, B.: Realizing synthetic active inference agents, part ii: Variational message updates. Neural Computation 37(1), 38–75 (2024)

[9] Levine, S.: Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv preprint arXiv:1805.00909 (2018)

[10] Lázaro-Gredilla, M., Ku, L.Y., Murphy, K.P., George, D.: What type of inference is planning? In: Advances in Neural Information Processing Systems 37 (NeurIPS 2024) (2024)

[11] Nuijten, W.W.L., Lukashchuk, M.: Active inference is a subtype of variational inference. In: EurIPS Workshop on Epistemic Intelligence in Machine Learning (nov 2025)

[12] Nuijten, W.W.L., Lukashchuk, M., van de Laar, T., de Vries, B.: A message passing realization of expected free energy minimization. In: International Workshop on Active Inference (IWAI) (oct 2025)

[13] Nuijten, W.W.L., Lukashchuk, M., van de Laar, T., de Vries, B.: What type of inference is active inference? In: Perković, E., Malinsky, D. (eds.) Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence. Proceedings of Machine Learning Research, vol. 337, pp. 5003–5039. PMLR (17–21 Aug 2026), https://proceedings.mlr.press/v337/nuijten26a.html

[14] Nuijten, W.W.L., van de Laar, T., de Vries, B.: Expected Free Energybased Planning as Variational Inference. Transactions on Machine Learning Research (2026), https://openreview.net/forum?id=Kzm8I1oS1s

[15] Parr, T., Friston, K.J.: Generalised free energy and active inference. Biological Cybernetics 113(5), 495–513 (2019). https://doi.org/10.1007/s004 22-019-00805-w

[16] Paul, A., Sajid, N., Da Costa, L., Razi, A.: On eficient computation in active inference. Expert Systems with Applications 253, 124315 (2024). https://doi.org/10.1016/j.eswa.2024.124315

[17] Toussaint, M.: Robot trajectory optimization using approximate inference. In: Proceedings of the 26th Annual International Conference on Machine Learning (ICML). pp. 1049–1056 (2009). https://doi.org/10.1145/1553374. 1553508

[18] Şenöz, I., van de Laar, T., Bagaev, D., de Vries, B.: Variational message passing and local constraint manipulation in factor graphs. Entropy 23(7), 807 (jun 2021). https://doi.org/10.3390/e23070807, https://doi.org/10.339 0/e23070807

## A The Objective: Proofs and Constructions

Notation: $\theta = ( \theta _ { A } , \theta _ { B } , \theta _ { c } ) , z = ( { \pmb y } , { \pmb x } , { \pmb u } , \theta ) , \mathbb { E } _ { q } = \mathbb { E } _ { q ( z ) } ; \ h [ q ( \cdot \ | \ { \pmb x } ) ]$ is the entropy of a conditional as a function of x and H $[ [ q ( \cdot \enspace | \enspace \dot { x } ) ] = \mathbb { E } _ { q ( x ) } h [ q ( \cdot \enspace | \enspace x ) ]$ the conditional-entropy scalar; $\operatorname { D i r } ( \alpha )$ with $\begin{array} { r } { \alpha _ { 0 } = \sum _ { k } \alpha _ { k } } \end{array}$ has density $\prod _ { k } \theta _ { k } ^ { \alpha _ { k } - 1 } / \mathrm { B } ( \alpha )$ Time indices are suppressed where no confusion arises.

## A.1 Epistemic Priors Recover the EFE

Lemma 1 (Prior contributions). For the priors (8)–(11) and any joint q possessing the conditionals they refer to,

$$
- \mathbb { E } _ { q } [ \log \tilde { p } ( u _ { t } ) ] = - \mathbb { H } [ q ( x _ { t } \mid x _ { t - 1 } , u _ { t } ) ] ,\tag{19}
$$

$$
- \mathbb { E } _ { q } [ \log \tilde { p } ( x _ { t } ) ] = + \mathbb { H } [ q ( y _ { t } \mid x _ { t } , \theta _ { A } , \theta _ { c } ) ] ,\tag{20}
$$

$$
- \mathbb { E } _ { q } [ \log \tilde { p } ( y _ { t } , x _ { t } ) ] = - I ( \theta _ { A } , \theta _ { c } ; y _ { t } \mid x _ { t } ) = - \mathcal { T } _ { c , t } - \mathcal { T } _ { A , t } ,\tag{21}
$$

$$
- \mathbb { E } _ { q } [ \log \tilde { p } ( x _ { t } , x _ { t - 1 } , u _ { t } , \theta _ { c } ) ] = - I ( \theta _ { B } ; x _ { t } , u _ { t } \mid x _ { t - 1 } , \theta _ { c } ) = - \mathbb { Z } _ { B , t } ,\tag{22}
$$

up to the normalizing constants of the priors.

Proof. Each prior is the exponential of a functional of the variables it is indexed by; taking − log and the outer expectation over those variables gives the stated quantity by definition. For $( 1 9 ) , \mathbb { E } _ { q ( u _ { t } ) } \mathbb { E } _ { q ( x _ { t } , x _ { t - 1 } \mid u _ { t } ) } [ \log q ( x _ { t } \mid x _ { t - 1 } , u _ { t } ) ] =$ $- \mathbb { H } [ q ( x _ { t } \mid x _ { t - 1 } , u _ { t } ) ] \colon$ for (20), $\mathbb { E } _ { q ( x _ { t } ) } \mathbb { E } _ { q ( \theta _ { A } , \theta _ { c } | x _ { t } ) } h [ q ( y _ { t } ~ \vert ~ x _ { t } , \theta _ { A } , \theta _ { c } ) ] = \mathbb { H } [ q ( y _ { t } ~ \vert ~$ $x _ { t } , \theta _ { A } , \theta _ { c } ) { \mathrm { . } }$ ]; for (21), $\mathbb { E } _ { q ( y _ { t } , x _ { t } ) } D _ { \mathrm { K L } } [ q ( \theta _ { A } , \theta _ { c } \mid y _ { t } , x _ { t } ) \parallel q ( \theta _ { A } , \theta _ { c } \mid x _ { t } ) ]$ is the conditional mutual information $\hat { I } ( \theta _ { A } , \theta _ { c } ; y _ { t } \mid x _ { t } )$ written as an expected divergence between posterior and cavity, and the chain rule $I ( \theta _ { A } , \theta _ { c } ; y _ { t } \mid x _ { t } ) = I ( \theta _ { c } ; y _ { t } \mid$ $x _ { t } ) + I ( \theta _ { A } ; y _ { t } \mid x _ { t } , \theta _ { c } )$ gives the split into salience and novelty of Section 4.1; likewise (22) with $( x _ { t } , u _ { t } )$ as the datum and $\left( x _ { t - 1 } , \theta _ { c } \right)$ as the conditioning variables. ⊓⊔

Proof (of Corollary 1). Split log ˜p into the unbiased model, the goals and the epistemic priors,

$$
\log \tilde { p } = \log p + \sum _ { t = 1 } ^ { T } \log \hat { p } _ { x } ( x _ { t } ) \hat { p } _ { y } ( y _ { t } ) + \sum _ { t = 1 } ^ { T } \log \left[ \tilde { p } ( u _ { t } ) \tilde { p } ( x _ { t } ) \tilde { p } ( y _ { t } , x _ { t } ) \tilde { p } ( x _ { t } , x _ { t - 1 } , u _ { t } , \theta _ { c } ) \right] .\tag{23}
$$

By Lemma 1,

$$
F _ { \widehat { p } } [ q ] = \underbrace { { \mathbb { E } } _ { q } \bigg [ \log \frac { q } { p } \bigg ] } _ { B [ q ] } + \sum _ { t = 1 } ^ { T } \Big ( - { \mathbb { E } } _ { q } \big [ \log \widehat { p } _ { x } ( x _ { t } ) \widehat { p } _ { y } ( y _ { t } ) \big ] - { \mathbb { H } } [ q ( x _ { t } \mid x _ { t - 1 } , u _ { t } ) ]\tag{24}
$$

$$
+ \mathbb { H } [ q ( y _ { t } \mid x _ { t } , \theta _ { A } , \theta _ { c } ) ] - { \mathcal { T } } _ { c , t } - { \mathcal { T } } _ { A , t } - { \mathcal { T } } _ { B , t } \Big ) .
$$

Expanding (7) and taking $\mathbb { E } _ { q ( u _ { t } ) }$ ，

$$
\begin{array} { r l } & { \mathbb { E } _ { q ( u _ { t } ) } [ G ( u _ { t } ) ] = - \mathbb { E } _ { q } \left[ \log \hat { p } _ { x } ( x _ { t } ) \hat { p } _ { y } ( y _ { t } ) \right] - \mathbb { H } [ q ( x _ { t } \mid x _ { t - 1 } , u _ { t } ) ] } \\ & { \qquad + \mathbb { E } _ { q } [ - \log p ( y _ { t } \mid x _ { t } , \theta _ { A } , \theta _ { c } ) ] - \mathcal { T } _ { c , t } - \mathcal { T } _ { A , t } - \mathcal { T } _ { B , t } , } \end{array}\tag{25}
$$

which coincides with the t-th summand above whenever $q ( y _ { t } \mid x _ { t } , \theta _ { A } , \theta _ { c } ) = p ( y _ { t }$ | $x _ { t } , \theta _ { A } , \theta _ { c } )$ . For the factor beliefs of Section 4.3, $q _ { \mathrm { o b s } , t } ( y _ { t } \mid x _ { t } , \theta _ { A } , \theta _ { c } ) \propto p ( y _ { t }$ $x _ { t } , \theta _ { A } , \theta _ { c } ) \hat { p } _ { y } ( y _ { t } )$ , so the identity is exact for uniform $\hat { p } _ { y }$ (the setting of Section 5); in general the ambiguity term of G is to be read as $\mathbb { H } [ q ( y _ { t } \mid x _ { t } , \theta _ { A } , \theta _ { c } ) ]$ , the entropy of the agent’s own predictive observation channel, as in [13]. Summing over t gives the Corollary. ⊓⊔

## A.2 Entropy Corrections

Adding the planning correction $+ \mathbb { H } [ q ( u _ { t } \mid x _ { t - 1 } ) ] \ [ 1 0 ]$ to the sum of (19)–(22) gives the raw per-step correction

$$
\begin{array} { r } { \varDelta F _ { t } = \mathbb { H } [ q ( y _ { t } \mid x _ { t } , \theta _ { A } , \theta _ { c } ) ] - \mathscr { T } _ { c , t } - \mathscr { T } _ { A , t } - \mathbb { H } [ q ( x _ { t } \mid x _ { t - 1 } , u _ { t } ) ] - \mathscr { T } _ { B , t } + \mathbb { H } [ q ( u _ { t } \mid x _ { t - 1 } ) ] . } \end{array}\tag{26}
$$

The identity $\mathbb { H } [ q ( y _ { t } \mid x _ { t } , \theta _ { A } , \theta _ { c } ) ] = \mathbb { H } [ q ( y _ { t } \mid x _ { t } ) ] - I ( \theta _ { A } , \theta _ { c } ; y _ { t } \mid x _ { t } ) = \mathbb { H } [ q ( y _ { t }$ $x _ { t } ) ] - \mathcal { T } _ { c , t } - \mathcal { T } _ { A , t }$ turns (26) into (12): the ambiguity prior contributes one unit of each observation-side gain through its θ-coupling, the novelty prior a second. Under any family with $q ( y _ { t } , x _ { t } , \theta _ { c } , \theta _ { A } ) = q ( y _ { t } , x _ { t } , \theta _ { c } ) q ( \theta _ { A } )$ the Dirichlet gain $\mathcal { T } _ { A , t }$ vanishes identically and prior (10) retains only its salience part; the same holds for $\mathcal { T } _ { B , t }$ and (11).

## A.3 The Conjugate Factor Belief

Lemma 2 (Entrywise conjugacy). For one column $\alpha = \alpha _ { A } ( \cdot \ | \ x , \theta _ { c } )$ and any entry y, $\theta _ { y } \operatorname { D i r } ( \theta \mid \alpha ) = { \frac { \alpha _ { y } } { \alpha _ { 0 } } } \operatorname { D i r } ( \theta \mid \alpha + e _ { y } )$

$$
\begin{array} { r } { \theta _ { y } \prod _ { k } \theta _ { k } ^ { \alpha _ { k } - 1 } / \mathrm { B } ( \alpha ) = \prod _ { k } \theta _ { k } ^ { \alpha _ { k } + [ k = y ] - 1 } / \mathrm { B } ( \alpha ) = \frac { \mathrm { B } ( \alpha + e _ { y } ) } { \mathrm { B } ( \alpha ) } \operatorname { D i r } ( \theta \mid \alpha + e _ { y } ) } \end{array}
$$

and $\frac { \mathrm { B } ( \alpha + e _ { y } ) } { \mathrm { B } ( \alpha ) } = \frac { { \cal T } ( \alpha _ { y } + 1 ) { \cal T } ( \alpha _ { 0 } ) } { { \cal T } ( \alpha _ { y } ) { \cal T } ( \alpha _ { 0 } + 1 ) } = \frac { \alpha _ { y } } { \alpha _ { 0 } }$ . Other columns are untouched. ⊓⊔

Applied, for each value of $\theta _ { c } ,$ to the factor belief of Section 4.3,

$$
\begin{array} { r } { q _ { \mathrm { o b s } , t } ( y _ { t } , x _ { t } , \theta _ { c } , \theta _ { A } ^ { ( t ) } ) \propto p ( y _ { t } \mid x _ { t } , \theta _ { A } ^ { ( t ) } , \theta _ { c } ) \mu _ { y _ { t } } ( y _ { t } ) \mu _ { x _ { t } } ( x _ { t } ) \mu _ { \theta _ { c } } ( \theta _ { c } ) \operatorname { D i r } ( \theta _ { A } ^ { ( t ) } \mid \alpha _ { A } ) , } \end{array}
$$

this gives (13). The identity needs two things: the message into the copy must be free of $( y , x , \theta _ { c } )$ —true because the copy has degree two and its other neighbor is the prior node—and the factor function must be a monomial in $\theta _ { A }$ for each value of $\theta _ { c } . \mathrm { ~ A ~ } \theta _ { A }$ -conditioned channel inside the factor function, such as $r ( y \mid$ $x , \theta _ { A } , \theta _ { c } ) ^ { 2 }$ with the shared parameter of [13] read as $\theta _ { A } .$ has a normalizer that is not polynomial in $\theta _ { A }$ and would destroy the identity; this is why the $\theta _ { A } .$ -epistemic content enters the factor function as a scalar weight per discrete entry, while the finite $\theta _ { c }$ retains the channel form (Proposition 2).

Lemma 3 (Elimination of the $\theta _ { A } { \bf - c o n d i t i o n a l ) }$ . Let the observation region carry the belief $q _ { y x , t } ( y , x , \theta _ { c } ) \nu _ { t } ( \theta _ { A } \mid y , x , \theta _ { c } )$ for an arbitrary normalized $\nu _ { t } .$ , and let the copy’s prior region carry the induced marginal $\textstyle \sum _ { y , x , \theta _ { c } } q _ { y x , t } \nu _ { t }$ . Their joint contribution to the Bethe free energy of the frozen graph is

$$
- \mathbb { H } [ q _ { y x , t } ] - \mathbb { E } _ { q _ { y x , t } } [ \log m _ { A } ] + \sum _ { y , x , \theta _ { c } } q _ { y x , t } ( y , x , \theta _ { c } ) D _ { \mathrm { K L } } [ \nu _ { t } ( \cdot \mid y , x , \theta _ { c } ) ] \mid \operatorname { D i r } ( \alpha _ { A } + e _ { y x \theta _ { c } } ) ] ,\tag{27}
$$

minimized uniquely at $\nu _ { t } = \operatorname { D i r } ( \alpha _ { A } + e _ { y x \theta _ { c } } ) = q _ { \mathrm { o b s } , t } ( \theta _ { A } \mid x , y , \theta _ { c } )$

Proof. The copy has degree two, so its region term $D _ { \mathrm { K L } } [ q ( \theta _ { A } ^ { ( t ) } ) \| \operatorname { D i r } ( \alpha _ { A } ) ]$ and its counting term $+ \mathbb { H } [ q ( \theta _ { A } ^ { ( t ) } ) ]$ combine to $- \mathbb { E } _ { q ( \theta _ { A } ^ { ( t ) } ) } [ \log \operatorname { D i r } ( \theta _ { A } \mid \alpha _ { A } ) ]$ . Adding the observation region $D _ { \mathrm { K L } } [ q _ { y x , t } \nu _ { t } \mid \mid p ( y \mid x , \theta _ { A } , \theta _ { c } ) ]$ and using $\begin{array} { r } { q ( \theta _ { A } ^ { ( t ) } ) = \sum _ { y , x , \theta _ { c } } q _ { y x , t } \nu _ { t } } \end{array}$

$$
- \mathbb { H } [ q _ { y x , t } ] + \sum _ { y , x , \theta _ { c } } q _ { y x , t } \mathbb { E } _ { \nu _ { t } } \Big [ \log \frac { \nu _ { t } ( \theta _ { A } \mid y , x , \theta _ { c } ) } { \mathrm { D i r } ( \theta _ { A } \mid \alpha _ { A } ) p ( y \mid x , \theta _ { A } , \theta _ { c } ) } \Big ] ,
$$

and Lemma 2 rewrites the denominator as $m _ { A } ( y \mid x , \theta _ { c } ) \operatorname { D i r } ( \theta _ { A } \mid \alpha _ { A } + e _ { y x \theta _ { c } } )$ . ⊓⊔

A coordinate whose optimum is a closed-form function of the others is substituted out; it is also not storable, $\theta _ { A }$ being continuous. Two consequences. First, the parameter energy and entropy of the region collapse jointly into the predictive energy − log $m _ { A } ;$ nothing $\theta _ { A }$ -dependent survives in the base free energy (used in Appendix $\mathrm { A . 5 } )$ . Second, the corrections (12) are evaluated at $\nu _ { t } = \mathrm { D i r } ( \alpha _ { A } + e _ { y x \theta _ { c } } )$ and not re-minimized over $\nu _ { t } \colon$ read as a functional of a free $\nu _ { t } , - 2 \mathcal { T } _ { A , t }$ would reward arbitrary coupling up to its ceiling $\mathbb { H } _ { q } [ y _ { t } \mid x _ { t } , \theta _ { c } ]$ . The dynamics side is identical with columns indexed by $( x _ { t - 1 } , u _ { t } , \theta _ { c } )$ . The finite $\theta _ { c }$ is treated diferently: a free $\theta _ { c } { \mathrm { - a x i s } }$ on the discrete beliefs costs a factor $K$ and is kept, so its conditionals are exact and no projection is needed.

The step-wise copies are what make this consistent. A single shared $\theta _ { A }$ would impose the calibration constraint $\sum _ { y , x , \theta _ { c } } q _ { \mathrm { o b s } , t } = q ( \theta _ { A } )$ for every $t ,$ but $\begin{array} { r } { \sum _ { y , x , \theta _ { c } } q _ { \mathrm { o b s } , t } = \mathrm { D i r } ( \alpha _ { A } ) \sum _ { y , x , \theta _ { c } } q _ { y x , t } ( y , x , \overline { { \theta } } _ { c } ^ { y , \infty , x \circ } \theta ( y \mid x , \theta _ { c } ) / m _ { A } ( y \mid x , \theta _ { c } ) } \end{array}$ equals $\operatorname { D i r } ( \det A )$ only when $q _ { y x , t } ( y ^ { \mathrm { ~ } } \vert \ x , \theta _ { c } ) = m _ { A } ( y \ \vert \ x , \theta _ { c } )$ , i.e. when planning leaves the block conditionals at their predictive values; enforcing it would write $\sim T$ imagined counts into $\alpha _ { A }$ per sweep and eliminate novelty inside the planning loop. With step-wise copies the constraint holds trivially and the approximation moves out of the constraint set into the planning-time model.

## A.4 The Cavity and Its Projection

Fix a column $( x , \theta _ { c } )$ , write $r _ { y } : = q _ { y x , t } ( y \mid x , \theta _ { c } ) , p _ { y } : = \mathrm { D i r } ( \alpha + e _ { y } )$ , and let $\begin{array} { r } { \bar { q } : = \sum _ { y } r _ { y } p _ { y } } \end{array}$ be the exact cavity (14). The projection $\tilde { q } : = \mathrm { D i r } ( \alpha + r )$ has the following properties.

(i) Mean matching. $\begin{array} { r } { \mathbb { E } _ { \bar { q } } [ \theta _ { k } ] = \sum _ { y } r _ { y } \frac { \alpha _ { k } + [ k = y ] } { \alpha _ { 0 } + 1 } = \frac { \alpha _ { k } + r _ { k } } { \alpha _ { 0 } + 1 } = \mathbb { E } _ { \tilde { q } } [ \theta _ { k } ] . } \end{array}$

(ii) Geometric mixture. $\begin{array} { r } { \prod _ { y } p _ { y } ^ { r _ { y } } \propto \prod _ { k } \theta _ { k } ^ { \sum _ { y } r _ { y } ( \alpha _ { k } + [ k = y ] - 1 ) } = \prod _ { k } \theta _ { k } ^ { \alpha _ { k } + r _ { k } - 1 } \propto \widetilde { q } \mathrm { . } } \end{array}$ the projection averages the branches’ natural parameters. Equivalently $\tilde { q } =$ arg min<sub>q˜</sub>′ $\begin{array} { r l } { \sum _ { y } r _ { y } D _ { \mathrm { K L } } [ \tilde { q } ^ { \prime } \parallel p _ { y } ] } & { { } } \end{array}$ , whereas $\begin{array} { r } { \bar { q } = \arg \operatorname* { m i n } _ { \tilde { q } ^ { \prime } } \sum _ { y } r _ { y } D _ { \mathrm { K L } } [ p _ { y } \| \tilde { q } ^ { \prime } ] } \end{array}$

(iii) Structured $V M P , ~ \tilde { q }$ is what variational message passing returns on the $\theta _ { - }$ edge under $q ( y , \theta \mid x ) = q ( y \mid x ) q ( \theta \mid x )$ : prior times $\exp \mathbb { E } _ { q ( y | x ) } [ \log \theta _ { y } ] =$ $\prod _ { y } \theta _ { y } ^ { r y } \ [ 1 8 ]$ ; the sum-product message $\sum _ { y } r _ { y } \theta _ { y }$ gives the arithmetic mixture q¯ instead.

(iv) Not the M-projection. arg min<sub>q˜</sub>′<sub>∈Dir</sub> $D _ { \mathrm { K L } } [ \bar { q } \parallel \tilde { q } ^ { \prime } ]$ would match $\mathbb { E } [ \log \theta _ { k } ]$ . The mixture carries between-branch variance that a single Dirichlet drops, $\begin{array} { r } { \mathbb { E } _ { \bar { q } } [ \theta _ { k } ^ { 2 } ] - \mathbb { E } _ { \tilde { q } } [ \theta _ { k } ^ { 2 } ] ~ = ~ \frac { r _ { k } ( 1 - r _ { k } ) } { ( \alpha _ { 0 } + 1 ) ( \alpha _ { 0 } + 2 ) } ~ } \end{array}$ , so $\tilde { q }$ is slightly overconfident and $\varDelta _ { A , t }$ slightly overstated, with relative error $O ( 1 / \alpha _ { 0 } )$ that vanishes as counts grow.

(v) Exactness and closure. $\tilde { q } = \bar { q }$ whenever r is one-hot; $\tilde { q }$ stays Dirichlet under linear damping of $r ;$ and $D _ { \mathrm { K L } } [ \operatorname* { D i r } \parallel \operatorname { D i r } ]$ is closed-form in digamma and loggamma functions.

The projection is used for the weight arrays only; the discrete messages close through $m _ { A } , m _ { B }$ without it. Its one side efect on the stationarity conditions is quantified in Lemma 5.

## A.5 Proof of Proposition 1

Degrees and Bethe free energy. On the frozen graph, $y _ { t } , u _ { t } , x _ { 0 }$ and every copy ${ \theta } _ { A } ^ { ( t ) } , { \theta } _ { B } ^ { ( t ) }$ have one non-singleton neighbor besides their prior or goal node; $x _ { t }$ for $1 \leq t < T$ has three $( f _ { \mathrm { o b s } , t } , f _ { \mathrm { d y n } , t } , f _ { \mathrm { d y n } , t + 1 } )$ , x<sub>T</sub> two, and $\theta _ { c }$ all $2 T$ . For a variable with k non-singleton neighbors and singleton factor $f _ { i } ,$ the singleton region and the counting term combine to $( k - 1 ) \mathbb { H } [ q _ { i } ] - \mathbb { E } _ { q _ { i } } [ \log f _ { i } ]$ . Hence

$$
\begin{array} { l } { { \displaystyle F _ { \hat { \rho } } [ q ] = \sum _ { t = 1 } ^ { T } \Big ( D _ { \mathrm { K L } } \big [ q _ { \mathrm { o b s } , t } \| p ( y _ { t } \mid x _ { t } , \theta _ { A } ^ { ( t ) } , \theta _ { c } ) \big ] - \mathbb { E } _ { q ( \theta _ { A } ^ { ( t ) } ) } \big [ \log \mathrm { D i r } ( \theta _ { A } ^ { ( t ) } \mid \alpha _ { A } ) \big ] } \ ~ } \\ { { \displaystyle ~ + ~ D _ { \mathrm { K L } } \big [ q _ { \mathrm { d y n } , t } \| p ( x _ { t } \mid x _ { t - 1 } , u _ { t } , \theta _ { B } ^ { ( t ) } , \theta _ { c } ) \big ] - \mathbb { E } _ { q ( \theta _ { B } ^ { ( t ) } ) } \big [ \log \mathrm { D i r } ( \theta _ { B } ^ { ( t ) } \mid \alpha _ { B } ) \big ] } \ ~ } \\ { { \displaystyle ~ - \mathbb { E } _ { q } \big [ \log p ( u _ { t } ) \hat { p } _ { y } ( y _ { t } ) \hat { p } _ { x } ( x _ { t } ) \big ] \Big ) } \ ~ } \\ { { \displaystyle ~ + 2 \sum _ { t = 1 } ^ { T - 1 } \mathbb { H } [ q ( x _ { t } ) ] + \mathbb { H } [ q ( x _ { T } ) ] + ( 2 T - 1 ) \mathbb { H } [ q ( \theta _ { c } ) ] - \mathbb { E } _ { q } \big [ \log p ( x _ { 0 } ) p ( \theta _ { c } ) \big ] } , } \end{array}\tag{28}
$$

and the same degree count gives the exponents of the Bethe joint of Section 4.4. Under the planning convention that $p ( \theta _ { c } )$ is the current context belief, the $\theta _ { c }$ terms are constants during planning.

(i) Base. Substitute (13), i.e. $\nu _ { t } = \mathrm { D i r } ( \alpha _ { A } + e _ { y x \theta _ { c } } )$ , into Lemma 3: the divergence term vanishes and the first two terms of (28) equal $- \mathbb { H } [ q _ { y x , t } ] - \mathbb { E } _ { q _ { y x , t } } [ \log m _ { A } ]$ ;

likewise on the dynamics side with $m _ { B }$ . Hence

$$
\begin{array} { r l } & { \displaystyle { F _ { \hat { p } } [ q ] = \sum _ { t = 1 } ^ { T } \Big ( - \mathbb { H } [ q _ { y x , t } ] - \mathbb { E } _ { q _ { y x , t } } [ \log m _ { A } ] - \mathbb { H } [ q _ { \mathrm { t r i p } , t } ] } } \\ & { \qquad \quad - \mathbb { E } _ { q _ { \mathrm { t r i p } , t } } [ \log m _ { B } ] - \mathbb { E } _ { q } \big [ \log p ( u _ { t } ) \hat { p } _ { y } ( y _ { t } ) \hat { p } _ { x } ( x _ { t } ) \big ] \Big ) - \mathbb { E } _ { q } [ \log p ( x _ { 0 } ) p ( \theta _ { c } ) ] } \\ & { \quad \quad + 2 \sum _ { t = 1 } ^ { T - 1 } \mathbb { H } [ q ( x _ { t } ) ] + \mathbb { H } [ q ( x _ { T } ) ] + ( 2 T - 1 ) \mathbb { H } [ q ( \theta _ { c } ) ] . } \end{array}\tag{29}
$$

Per entry the cancellation reads

$$
\begin{array} { r l } { - h \big [ \mathrm { D i r } ( \alpha _ { A } + e _ { y x \theta _ { c } } ) \big ] - \mathbb { E } _ { \mathrm { D i r } ( \alpha _ { A } + e _ { y x \theta _ { c } } ) } \big [ \log p ( y \mid x , \theta _ { A } , \theta _ { c } ) + \log \mathrm { D i r } ( \theta _ { A } \mid \alpha _ { A } ) \big ] } & { } \\ { = - \log m _ { A } ( y \mid x , \theta _ { c } ) , } \end{array}\tag{30}
$$

which is Bayes’ rule log $\operatorname { D i r } ( \alpha _ { A } + e _ { y x \theta _ { c } } ) = \log \operatorname { D i r } ( \alpha _ { A } ) + \log p - \log m _ { A }$ integrated against itself. Because the copy’s prior node carries the current belief $\operatorname { D i r } ( \alpha _ { A } )$ 2 there is no prior-mismatch residual $\mathbb { E } [ \log q ( \theta _ { A } ) / p ( \theta _ { A } ) ]$ to dispose of.

(ii) Corrections. Each copy lies in exactly one region of the Bethe joint, so the marginal of $q$ over $( y _ { t } , x _ { t } , \theta _ { c } , \theta _ { A } ^ { ( t ) } )$ is $q _ { \mathrm { o b s } , t }$ and over $( x _ { t } , x _ { t - 1 } , u _ { t } , \theta _ { c } , \theta _ { B } ^ { ( t ) } )$ is $q _ { \mathrm { d y n } , t } ;$ on the chain the Bethe joint is a proper distribution with these marginals. Its $\theta _ { A } , \theta _ { B ^ { - } }$ free conditionals are the block conditionals, $q ( y _ { t } \mid x _ { t } ) = q _ { y x , t } ( y _ { t } \mid$ | x ${ \mathrm { \Omega } } _ { t } ) , q ( \theta _ { c } \mid x _ { t } , y _ { t } ) = q _ { y x , t } ( \theta _ { c } \mid x _ { t } , y _ { t } ) , q ( x _ { t } \mid x _ { t - 1 } , u _ { t } ) = q _ { \mathrm { t r i p } , t } ( x _ { t } \mid x _ { t - 1 } , u _ { t } )$ $q ( u _ { t } \mid x _ { t - 1 } ) = q _ { \mathrm { t r i p } , t } ( u _ { t } \mid x _ { t - 1 } )$ ; its $\theta _ { A }$ -conditionals are $q _ { \mathrm { o b s } , t } ( \theta _ { A } \mid x _ { t } , y _ { t } , \theta _ { c } ) =$ $\mathrm { D i r } ( \boldsymbol { \alpha } _ { A } + \boldsymbol { e } _ { y _ { t } x _ { t } \theta _ { c } } )$ and the cavity (14). Therefore $I _ { q } ( \theta _ { c } ; y _ { t } \mid x _ { t } ) = I _ { q _ { y x , t } } ( y _ { t } ; \theta _ { c } \mid$ | $x _ { t } )$ , $I _ { q } ( \theta _ { A } ^ { ( t ) } ; y _ { t } \mid x _ { t } , \theta _ { c } ) = \mathbb { E } _ { q _ { y x , t } } D _ { \mathrm { K L } } [ q _ { \mathrm { o b s } , t } ( \theta _ { A } \mid x _ { t } , y _ { t } , \theta _ { c } ) \parallel q _ { \mathrm { o b s } , t } ( \theta _ { A } \mid x _ { t } , \theta _ { c } ) ]$ = $\mathbb { E } _ { q _ { y x , t } \left[ \varDelta _ { A , t } \right] }$ , likewise for $\theta _ { B }$ , and (12) holds with every term read of $q _ { y x , t }$ and $q _ { \mathrm { t r i p } , t }$ . Nothing is double-counted with (i): the information-gain mass in the $\theta _ { - }$ coupled energy is cancelled by (30) and never reaches $\varDelta F _ { t }$

(iii) Assembly. Adding (i) and (ii) gives (16), a functional of the discrete coordinates of Section 4.3 and the fixed counts α<sub>A</sub>, α<sub>B</sub> only. ⊓⊔

Remark 1 (Limits). At $q ( \theta _ { A } ) , q ( \theta _ { B } ) = \delta _ { \theta } ,$ ∗ every Dirichlet posterior and cavity is $\delta _ { \theta ^ { * } } , \varDelta _ { A } = \varDelta _ { B } = 0 , m \to p ( \cdot \mid \theta ^ { * } , \theta _ { c } )$ , and (16) is the fixed-parameter EFEplanning objective of [13] with their $\theta$ read as $\theta _ { c }$ . Deleting the correction line leaves belief propagation on the posterior predictive likelihoods. Both limits are approached continuously in the concentrations.

## B Message-Passing Derivation

We derive the stationarity conditions of the planning objective (16) and solve them as message passing, following Appendix D of [13] with two structural differences: the Dirichlet copies are integrated out in closed form, so there are no $\theta _ { A } , \theta _ { B }$ -messages, and the channels are conditionals of $q ,$ so there are no channel variables or constraints. The context $\theta _ { c }$ is a discrete variable of the graph and is treated exactly as the shared θ of [13].

## B.1 Lagrangian

At T=1 the coordinates are $q _ { y x } ( y _ { 1 } , x _ { 1 } , \theta _ { c } ) , q _ { \mathrm { t r i p } } ( x _ { 1 } , x _ { 0 } , u _ { 1 } , \theta _ { c } )$ and the singletons $q _ { x _ { 0 } } , q _ { x _ { 1 } } , q _ { y _ { 1 } } , q _ { u _ { 1 } } , q _ { \theta _ { c } }$ ; the parameter beliefs Dir $\left( \alpha _ { A } \right)$ , Dir(α ) are fixed. With multipliers for normalization of the two blocks and for the seven factor-to-singleton consistency constraints,

$$
\begin{array} { r l } & { \mathcal { L } [ q , \lambda ] = F _ { 1 } [ q ] + \lambda _ { y x } \Big ( \sum q _ { y x } - 1 \Big ) + \lambda _ { \mathrm { t r i p } } \Big ( \sum q _ { \mathrm { t r i p } } - 1 \Big ) } \\ & { \qquad + \sum _ { y _ { 1 } } \lambda _ { y _ { 1 } } \Big ( \sum _ { x _ { 1 } , \theta _ { c } } q _ { y x } - q _ { y _ { 1 } } \Big ) + \sum _ { x _ { 1 } } \lambda _ { x _ { 1 } } ^ { \mathrm { o b s } } \Big ( \sum _ { y _ { 1 } , \theta _ { c } } q _ { y x } - q _ { x _ { 1 } } \Big ) } \\ & { \qquad + \sum _ { \theta _ { c } } \lambda _ { \theta _ { c } } ^ { \mathrm { o b s } } \Big ( \sum _ { y _ { 1 } , x _ { 1 } } q _ { y x } - q _ { \theta _ { c } } \Big ) + \sum _ { x _ { 1 } } \lambda _ { x _ { 1 } } ^ { \mathrm { d y n } } \Big ( \sum _ { x _ { 0 } , u _ { 1 } , \theta _ { c } } q _ { \mathrm { t r i p } } - q _ { x _ { 1 } } \Big ) } \\ & { \qquad + \sum _ { x _ { 0 } } \lambda _ { x _ { 0 } } \Big ( \sum _ { x _ { 1 } , u _ { 1 } , \theta _ { c } } q _ { \mathrm { t r i p } } - q _ { x _ { 0 } } \Big ) + \sum _ { u _ { 1 } } \lambda _ { u _ { 1 } } \Big ( \sum _ { x _ { 1 } , x _ { 0 } , \theta _ { c } } q _ { \mathrm { t r i p } } - q _ { u _ { 1 } } \Big ) } \\ & { \qquad + \sum _ { \theta _ { c } } \lambda _ { \theta _ { c } } ^ { \mathrm { d y n } } \Big ( \sum _ { x _ { 1 } , x _ { 0 } , u _ { 1 } } q _ { \mathrm { t r i p } } - q _ { \theta _ { c } } \Big ) , } \end{array}
$$

where each consistency multiplier is a function of its variable, $\lambda _ { y _ { 1 } } = \lambda _ { y _ { 1 } } ( y _ { 1 } )$ etc. Relative to [13] (their (D.3)–(D.7)), the four channel multipliers are absent and the θ-consistency multipliers refer to $\theta _ { c }$ only. The corrections in $F _ { 1 }$ are functionals of the coordinates through (13)–(14), cavities included.

(31)

## B.2 Stationarity Conditions

Two lemmas make the variation of the corrections local.

Lemma 4 (Score lemma). For a normalized joint $B [ q ] ( a , b )$ depending on a coordinate q, and any of its conditionals,

$$
{ \frac { \delta } { \delta q } } \mathbb { H } [ B ( a \mid b ) ] = - \int { \frac { \delta B ( a , b ) } { \delta q } } \log B ( a \mid b ) d a d b ,\tag{32}
$$

$i . e .$ entropies of conditionals—and hence conditional mutual informations— diferentiate as if the logarithm were frozen.

Proof. H $\begin{array} { r } { [ [ B ( a \mathrm { ~  ~ \theta ~ } | b ) ] = - \int B ( a , b ) \log \frac { B ( a , b ) } { B ( b ) } } \end{array}$ . The product-rule remainder is $\begin{array} { r } { - \int B ( a , b ) \left[ \frac { \delta B ( a , b ) / \delta q } { B ( a , b ) } - \frac { \delta B ( b ) / \delta q } { B ( b ) } \right] = - \int \frac { \delta B ( a , b ) } { \delta q } + \int \frac { \delta B ( b ) } { \delta q } = 0 } \end{array}$ , since $\textstyle \int _ { a } B ( a , b ) =$ $B ( b )$ for every q. ⊓⊔

Lemma 5 (Cavity back-reaction vanishes). With $\beta ( \theta \mid x , y , \theta _ { c } ) : = q _ { \mathrm { o b s } , t } ( \theta _ { A }$ $x , y , \theta _ { c } )$ independent of $q _ { y x , t }$ and the exact cavity $\begin{array} { r } { \bar { \beta } ( \theta \ \mid \ x , \theta _ { c } ) = \sum _ { y } q _ { y x , t } ( y \ \mid \ } \end{array}$ $x , \theta _ { c } ) \beta ( \theta \mid x , y , \theta _ { c } )$

$$
\frac { \delta \mathcal { Z } _ { A , t } } { \delta q _ { y x , t } ( y , x , \theta _ { c } ) } = \varDelta _ { A , t } ( x , y , \theta _ { c } ) = D _ { \mathrm { K L } } \big [ \beta ( \cdot \mid x , y , \theta _ { c } ) \big | \big | \bar { \beta } ( \cdot \mid x , \theta _ { c } ) \big ] ,\tag{33}
$$

with no contribution from the dependence of $\bar { \beta }$ on $q _ { y x , t }$ . Likewise $\delta \mathcal { T } _ { B , t } / \delta q _ { \mathrm { t r i p } , t } =$ $\varDelta _ { B , t }$

Proof. Write

$$
\begin{array} { r l r } & { } & { \mathcal { Z } _ { A , t } = \displaystyle \sum _ { y , x , \theta _ { c } } q _ { y x , t } ( y , x , \theta _ { c } ) \int \beta ( \theta \mid x , y , \theta _ { c } ) \log \beta ( \theta \mid x , y , \theta _ { c } ) d \theta } \\ & { } & { \quad \quad - \displaystyle \sum _ { y , x , \theta _ { c } } q _ { y x , t } ( y , x , \theta _ { c } ) \int \beta ( \theta \mid x , y , \theta _ { c } ) \log \bar { \beta } ( \theta \mid x , \theta _ { c } ) d \theta . } \end{array}
$$

The explicit derivative is $\varDelta _ { A , t } ( x , y , \theta _ { c } )$ . Since $\begin{array} { r l } { ~ } & { { } \sum _ { y ^ { \prime } } q _ { y x , t } ( y ^ { \prime } , x , \theta _ { c } ) \beta ( \theta \mid x , y ^ { \prime } , \theta _ { c } ) } \end{array}$ equals $q ( x , \theta _ { c } ) \bar { \beta } ( \theta \mid x , \theta _ { c } )$ , the implicit derivative through $\bar { \beta } ( \cdot \mid x , \theta _ { c } )$ is

$$
\begin{array} { r l r } {  { - q ( x , \theta _ { c } ) \int \bar { \beta } ( \theta \mid x , \theta _ { c } ) \frac { \delta \log \bar { \beta } ( \theta \mid x , \theta _ { c } ) } { \delta q _ { y x , t } ( y , x , \theta _ { c } ) } d \theta } } \\ & { } & { = - q ( x , \theta _ { c } ) \frac { \delta } { \delta q _ { y x , t } ( y , x , \theta _ { c } ) } \int \bar { \beta } ( \theta \mid x , \theta _ { c } ) d \theta = 0 , } \end{array}
$$

because $\bar { \beta }$ is normalized for every q.

Under the cavity projection of Section 4.3 the weight in the integral is the exact mixture while the logarithm is that of $\operatorname { D i r } ( \alpha + { \bar { n } } _ { x \theta _ { c } } )$ , whose n¯-derivative is log $\theta _ { k } - \psi ( \alpha _ { k } + \bar { n } _ { k } ) + \psi ( \alpha _ { 0 } + 1 )$ ; the residual is therefore proportional to $\begin{array} { r } { \mathbb { E } _ { \bar { \beta } } [ \log \theta _ { k } ] - \mathbb { E } _ { \mathrm { D i r } ( \alpha + \bar { n } ) } [ \log \theta _ { k } ] ^ { ' } = ( 1 - r _ { k } ) \psi ( \alpha _ { k } ) + r _ { k } \psi ( \alpha _ { k } + 1 ) - \psi ( \alpha _ { k } + r _ { k } ) = O ( \alpha _ { k } ^ { - 2 } ) } \end{array}$ by concavity of ψ, and is dropped. Only explicit appearances of q survive, which is why the updates stay local and the mixture structure of the cavity never enters the messages.

Proposition 3 (Observation block). Varying (31) in $q _ { y x } ( y _ { 1 } , x _ { 1 } , \theta _ { c } )$ : the base (29) gives log $q _ { y x } + 1 - \log m _ { A } ; ~ + \mathbb { H } _ { q _ { y x } } [ y _ { 1 } ~ | ~ x _ { 1 } ]$ gives $- \log q ( y _ { 1 } \mid x _ { 1 } )$ and $- 2 \mathcal { T } _ { c }$ gives $- 2 \log [ q ( \theta _ { c } \mathrm { ~  ~ \left| ~ \ \right)} x _ { 1 } , y _ { 1 }  / q ( \theta _ { c } \mathrm { ~  ~ \left| ~ \ \right]} x _ { 1 } ) $ (Lemma $4 ) ; \ - 2  { T _ { A } }$ gives $- 2 \varDelta _ { A } ( x _ { 1 } , y _ { 1 } , \theta _ { c } )$ (Lemma 5); the multipliers give $\lambda _ { y x } + \lambda _ { y _ { 1 } } + \lambda _ { x _ { 1 } } ^ { \mathrm { o b s } } + \lambda _ { \theta _ { c } } ^ { \mathrm { o b s } }$ Using $q ( \theta _ { c } \mid x , y ) / q ( \theta _ { c } \mid x ) = q ( y \mid x , \theta _ { c } ) / q ( y \mid x )$ , at stationarity

$$
\begin{array} { r } { q _ { y x } ^ { * } ( y _ { 1 } , x _ { 1 } , \theta _ { c } ) \propto \underbrace { m _ { A } ( y _ { 1 } \mid x _ { 1 } , \theta _ { c } ) } _ { p r e d i c t i v e } \xrightarrow [ ] { q ^ { * } ( y _ { 1 } \mid x _ { 1 } ) } \underbrace { e ^ { 2 \mathcal { A } _ { A } ( x _ { 1 } , y _ { 1 } , \theta _ { c } ) } } _ { s a l i e n c e } \underbrace { e ^ { 2 \mathcal { A } _ { A } ( x _ { 1 } , y _ { 1 } , \theta _ { c } ) } } _ { n o v e l t y \ge 1 } e ^ { - \lambda ^ { \mathrm { o b s } } } , } \end{array}\tag{34}
$$

with $\lambda ^ { \mathrm { o b s } } : = \lambda _ { y _ { 1 } } + \lambda _ { x _ { 1 } } ^ { \mathrm { o b s } } + \lambda _ { \theta _ { c } } ^ { \mathrm { o b s } }$

Proposition 4 (Dynamics block). Varying in $q _ { \mathrm { t r i p } } ( x _ { 1 } , x _ { 0 } , u _ { 1 } , \theta _ { c } ) .$ : the base gives $\log q _ { \mathrm { t r i p } } + 1 - \log m _ { B } ; - \mathbb H _ { q } [ x _ { 1 } \mid x _ { 0 } , u _ { 1 } ]$ gives + log $q ( x _ { 1 } \mid x _ { 0 } , u _ { 1 } ) ; + \mathbb { H } _ { q } [ u _ { 1 } \mid$ x<sub>0</sub>] gives − log $q ( u _ { 1 } \mid x _ { 0 } ) ; - { \mathcal { T } } _ { B }$ gives $- \varDelta _ { B }$ . At stationarity,

$$
q _ { \mathrm { t r i p } } ^ { * } ( x _ { 1 } , x _ { 0 } , u _ { 1 } , \theta _ { c } ) \propto m _ { B } ( x _ { 1 } \mid x _ { 0 } , u _ { 1 } , \theta _ { c } ) \xrightarrow [ q ^ { * } ( x _ { 1 } \mid x _ { 0 } , u _ { 1 } ) ] { q ^ { * } ( u _ { 1 } \mid x _ { 0 } ) } e ^ { \varDelta _ { B } ( x _ { 1 } , x _ { 0 } , u _ { 1 } , \theta _ { c } ) } e ^ { - \lambda ^ { \mathrm { d y n } } } ,\tag{35}
$$

with $\lambda ^ { \mathrm { d y n } } : = \lambda _ { x _ { 1 } } ^ { \mathrm { d y n } } + \lambda _ { x _ { 0 } } + \lambda _ { u _ { 1 } } + \lambda _ { \theta _ { c } } ^ { \mathrm { d y n } }$

Propositions 3–4 are Proposition 2: the factor functions (13a)–(13b) of [13] with the point likelihood replaced by the posterior predictive likelihood and the

information-gain weights appended. Their observation factor function carries the ratio $q ( y \mid x , \theta ) ^ { 2 } / q ( y \mid x )$ for the full θ; with $\theta = \left( \theta _ { A } , \theta _ { c } \right)$ it factorizes as

$$
\begin{array} { r l } & { \frac { q ( y \mid x , \theta _ { A } , \theta _ { c } ) ^ { 2 } } { q ( y \mid x ) } = \frac { q ( y \mid x , \theta _ { c } ) ^ { 2 } } { q ( y \mid x ) } \cdot \Bigl [ \frac { q ( y \mid x , \theta _ { A } , \theta _ { c } ) } { q ( y \mid x , \theta _ { c } ) } \Bigr ] ^ { 2 } , } \\ & { \mathbb { E } _ { q _ { \mathrm { o b s } , t } ( \theta _ { A } \mid x , y , \theta _ { c } ) } \log \frac { q ( y \mid x , \theta _ { A } , \theta _ { c } ) } { q ( y \mid x , \theta _ { c } ) } = \varDelta _ { A , t } ( x , y , \theta _ { c } ) , } \end{array}\tag{36}
$$

since the second ratio equals $q _ { \mathrm { o b s } , t } ( \theta _ { A } \mid x , y , \theta _ { c } ) / q _ { \mathrm { o b s } , t } ( \theta _ { A } \mid x , \theta _ { c } )$ by Bayes’ rule. The finite factor is stored on the θ<sub>c</sub>-axis and retains its channel form; the $\theta _ { A ^ { - } }$ dependent factor, not polynomial in $\theta _ { A } .$ is replaced by its posterior expectation, a scalar per entry, as conjugacy requires (Appendix A.3). Because each correction is diferentiated under the measure it is defined on, the weights have the form $+ D _ { \mathrm { K L } } \ge 0$

Singleton beliefs. $\varDelta F$ has no singleton dependence, so as in [13] (D.4.6): for the degree-two variables the singleton entropy cancels against the counting term, and for the shared variables only the sum of multipliers is fixed,

$$
\begin{array} { r l } & { \lambda _ { y _ { 1 } } = - \log \hat { p } _ { y } , \qquad \lambda _ { x _ { 0 } } = - \log p ( x _ { 0 } ) , \qquad \lambda _ { u _ { 1 } } = - \log p ( u _ { 1 } ) , } \\ & { \qquad \quad \lambda _ { x _ { 1 } } ^ { \mathrm { o b s } } + \lambda _ { x _ { 1 } } ^ { \mathrm { d y n } } = - \log q _ { x _ { 1 } } ^ { * } - 1 - \log \hat { p } _ { x } , } \\ & { \qquad \quad \lambda _ { \theta _ { c } } ^ { \mathrm { o b s } } + \lambda _ { \theta _ { c } } ^ { \mathrm { d y n } } = - \log q _ { \theta _ { c } } ^ { * } - 1 - \log p ( \theta _ { c } ) . } \end{array}\tag{37}
$$

Remark 2 (Parameter beliefs and the frozen-parameter update). During planning $q ( \theta _ { A } ) , q ( \theta _ { B } )$ are frozen, so m, the one-count posteriors and the weights depend on them only through the fixed concentrations, whereas $q ( \theta _ { c } )$ is an ordinary singleton belief and moves within the sweep. Exact functional stationarity of $F _ { T }$ in a free $q _ { \theta _ { A } }$ has two parts: the conjugate statistic (a Dirichlet count update) and a centered score $s _ { A } ( \theta _ { A } ) = \mathbb { E } _ { w } [ \log \beta - \log \bar { \beta } - \varDelta _ { A } ]$ under the reweighted block measure w ∝ $q _ { y x , t } p ( y \mid x , \theta _ { A } , \theta _ { c } ) / m _ { A } .$ which integrates to zero against $q _ { \theta _ { A } }$ and vanishes at point mass and as counts grow. The score would reshape $q _ { \theta _ { A } }$ toward parameter values of above-average posterior distinguishability without shifting count mass. Between environment steps we drop it and update the concentrations by the plain conjugate count update from the realized transition and observation,

$$
\begin{array} { r l } & { q _ { \theta _ { A } } \propto p ( \theta _ { A } ) \exp \Big ( \sum _ { t } \mathbb { E } _ { q _ { y x , t } } [ \log p ( y _ { t } \mid x _ { t } , \theta _ { A } , \theta _ { c } ) ] \Big ) , } \\ & { q _ { \theta _ { B } } \propto p ( \theta _ { B } ) \exp \Big ( \sum _ { t } \mathbb { E } _ { q _ { \mathrm { t r i p } , t } } [ \log p ( x _ { t } \mid x _ { t - 1 } , u _ { t } , \theta _ { B } , \theta _ { c } ) ] \Big ) , } \end{array}\tag{38}
$$

with $q _ { y x , t } , q _ { \mathrm { t r i p } , t }$ now the (soft) realized data, the count distributed over contexts according to $q ( \theta _ { c } )$ , which is itself updated by exact Bayes. This is a partialgradient step of $F _ { T }$ that preserves exact conjugacy, with the dropped terms exhibited; the score is worth including only if monitoring shows the plain step increasing $F _ { T }$

## B.3 Solving the Stationarity Equations as Message Passing

Given the factor functions (17)–(18) with lagged conditionals $r _ { t } ,$ Propositions 3– 4 are an exact Bethe belief-propagation problem on the discrete graph over

$\{ y _ { t } , x _ { t } , u _ { t } , \theta _ { c } \}$ [18], and the standard factor-to-variable rule applies,

$$
\mu _ { a  j } ( s _ { j } ) \propto \sum _ { s _ { a \setminus j } } \bar { g } _ { a } ( s _ { a } ) \prod _ { i \in \mathcal { E } ( a ) \backslash j } \mu _ { i  a } ( s _ { i } ) .\tag{39}
$$

Observation factor, writing $\mu _ { y _ { t } } , \mu _ { x _ { t } } , \mu _ { \theta _ { c } }$ for the messages into $f _ { \mathrm { o b s } , t }$ (with $\mu _ { { y } _ { t } } =$ $\hat { p } _ { y } )$ :

$$
\mu _ { \mathrm { o b s } _ { t } \to x _ { t } } = \sum _ { y _ { t } , \theta _ { c } } \bar { g } _ { \mathrm { o b s } , t } \mu _ { y _ { t } } \mu _ { \theta _ { c } } , \qquad \mu _ { \mathrm { o b s } _ { t } \to y _ { t } } = \sum _ { x _ { t } , \theta _ { c } } \bar { g } _ { \mathrm { o b s } , t } \mu _ { x _ { t } } \mu _ { \theta _ { c } } ,\tag{40}
$$

$$
\mu _ { \mathrm { o b s } _ { t } \to \theta _ { c } } = \sum _ { y _ { t } , x _ { t } } { \bar { g } } _ { \mathrm { o b s } , t } \mu _ { y _ { t } } \mu _ { x _ { t } } .
$$

Dynamics factor: analogously to $x _ { t } , x _ { t - 1 } , u _ { t } , \theta _ { c } .$ , each summing $\bar { g } _ { \mathrm { d y n } , t }$ against the other three incoming messages, with $\mu _ { u _ { t } \to \mathrm { d y n } _ { t } } = p ( u _ { t } )$ and boundary $\mu _ { x _ { 0 } \to \mathrm { d y n } _ { 1 } } =$ $p ( x _ { 0 } )$ . There are no messages to or from $\theta _ { A } ^ { ( t ) } , \theta _ { B } ^ { ( t ) }$ . Block and singleton beliefs are

$$
q _ { y x , t } \propto \bar { g } _ { \mathrm { o b s } , t } \mu _ { y _ { t } \to \mathrm { o b s } _ { t } } \mu _ { x _ { t } \to \mathrm { o b s } _ { t } } \mu _ { \theta _ { c } \to \mathrm { o b s } _ { t } } ,\tag{41}
$$

$$
q _ { \mathrm { t r i p } , t } \propto \bar { g } _ { \mathrm { d y n } , t } \mu _ { x _ { t }  \mathrm { d y n } _ { t } } \mu _ { x _ { t - 1 }  \mathrm { d y n } _ { t } } \mu _ { u _ { t }  \mathrm { d y n } _ { t } } \mu _ { \theta _ { c }  \mathrm { d y n } _ { t } } ,
$$

$$
\begin{array} { r l } { q ( x _ { t } ) \propto \hat { p } _ { x } \mu _ { \mathrm { o b s } _ { t } \to x _ { t } } \mu _ { \mathrm { d y n } _ { t } \to x _ { t } } \mu _ { \mathrm { d y n } _ { t + 1 } \to x _ { t } } , \quad } & { q ( \theta _ { c } ) \propto p ( \theta _ { c } ) \displaystyle \prod _ { \tau = 1 } ^ { T } \mu _ { \mathrm { o b s } _ { \tau } \to \theta _ { c } } \mu _ { \mathrm { d y n } _ { \tau } \to \theta _ { c } } , } \\ { q ( u _ { t } ) \propto p ( u _ { t } ) \mu _ { \mathrm { d y n } _ { t } \to u _ { t } } , \quad } & { q ( y _ { t } ) \propto \hat { p } _ { y } \mu _ { \mathrm { o b s } _ { t } \to y _ { t } } , } \end{array}\tag{42}
$$

which at a fixed point recover $( 3 4 )  { - } ( 3 5 )$ with (37). The full factor beliefs follow from (13), $q _ { \mathrm { o b s } , t } = q _ { y x , t } \operatorname { D i r } ( \alpha _ { A } + e _ { y _ { t } x _ { t } \theta _ { c } } )$ , and are never stored. The block-layer cost is independent of the size of the Dirichlet parameter space and linear in $K$

Refresh. After each sweep, from the new block beliefs,

$$
\begin{array} { r l } & { r _ { t } ( y _ { t } \mid x _ { t } , \theta _ { c } )  q _ { y x , t } ( y _ { t } \mid x _ { t } , \theta _ { c } ) , \qquad r _ { t } ( y _ { t } \mid x _ { t } )  q _ { y x , t } ( y _ { t } \mid x _ { t } ) , } \\ & { r _ { t } ( u _ { t } \mid x _ { t - 1 } )  q _ { \mathrm { t r i p } , t } ( u _ { t } \mid x _ { t - 1 } ) , \qquad r _ { t } ( x _ { t } \mid x _ { t - 1 } , u _ { t } )  q _ { \mathrm { t r i p } , t } ( x _ { t } \mid x _ { t - 1 } , u _ { t } ) , } \end{array}\tag{43}
$$

$$
\begin{array} { r l } & { \bar { n } _ { x _ { t } \theta _ { c } }  r _ { t } ( \cdot \mid x _ { t } , \theta _ { c } ) , \quad \varDelta _ { A , t } ( x _ { t } , y _ { t } , \theta _ { c } )  D _ { \mathrm { K L } } [ \mathrm { D i r } ( \alpha _ { A } + e _ { y _ { t } x _ { t } \theta _ { c } } ) ] \parallel \mathrm { D i r } ( \alpha _ { A } + \bar { n } _ { x _ { t } \theta _ { c } } ) \vphantom { \frac { 1 } { x _ { t } \theta _ { c } } } , } \end{array}
$$

and likewise $\varDelta _ { B , t }$ with fractional counts $\bar { n } _ { x _ { t - 1 } \theta _ { c } } ( x _ { t } , u _ { t } ) = q _ { \mathrm { t r i p } , t } ( x _ { t } , u _ { t } \mid x _ { t - 1 } , \theta _ { c } )$ spread over the $( x _ { t - 1 } , u _ { t } , \theta _ { c } )$ columns, so that the divergence carries cross-column terms. The factor functions are rebuilt immediately. Damping is applied linearly in n¯, which keeps the cavity Dirichlet. Since no conditional is built from messages and the one-count posteriors never feed back into themselves, the divergent channel recursion of [13] is structurally absent.

Closed forms. With $\begin{array} { r } { q _ { \theta _ { A } } = \prod _ { x , \theta _ { c } } \operatorname { D i r } ( \alpha _ { A } ( \cdot  { | \begin{array} { l } { x , \theta _ { c } } \end{array} ) } ) \colon m _ { A } ( y  { | \begin{array} { l } { x , \theta _ { c } } \end{array} ) } = \alpha _ { A } ( y  { | \begin{array} { l } { | } \end{array} | } ) } \end{array}$ $x , \theta _ { c } ) / \alpha _ { A , 0 } ( x , \theta _ { c } ) ; q _ { \mathrm { o b s } , t } ( \theta _ { A } \mid x , y , \breve { \theta } _ { c } ) = \mathrm { D i r } ( \alpha _ { A } + e _ { y x \theta _ { c } } ) ;$ the projected cavity is $\mathrm { D i r } ( \alpha _ { A } + \bar { n } _ { x \theta _ { c } } )$ ; and for two Dirichlets on one column,

$$
D _ { \mathrm { K L } } [ \mathrm { D i r } ( a ) \parallel \mathrm { D i r } ( b ) ] = \log { \frac { T ( a _ { 0 } ) } { T ( b _ { 0 } ) } } + \sum _ { k } \Big [ \log { \frac { T ( b _ { k } ) } { T ( a _ { k } ) } } + ( a _ { k } - b _ { k } ) \big ( \psi ( a _ { k } ) - \psi ( a _ { 0 } ) \big ) \Big ] .\tag{45}
$$

The salience $I _ { q _ { y x , t } } ( y _ { t } ; \theta _ { c } \mid x _ { t } )$ is a finite sum over the stored array. $F _ { T } \ \left( 1 6 \right)$ is assembled from quantities the sweep already computes and gives a stopping criterion on $| \varDelta F _ { T } |$

Schedule. Algorithm 1 of the main text, with the objects of this appendix in place: the predictive likelihoods and the one-count posteriors $\mathrm { D i r } ( \alpha _ { A } + e _ { y x \theta _ { c } } )$ $\mathrm { D i r } ( \alpha _ { B } + e _ { x ^ { \prime } x u \theta _ { c } } )$ are built once per planning call from the current counts; each sweep rebuilds the factor functions (17)–(18), runs one forward–backward pass of (39)–(42) over the chain, applies the refresh (43)–(44) and evaluates $F _ { T } ;$ the policy is read from $q ( u _ { 1 } \mid x _ { 0 } )$ at convergence. $F _ { T }$ is a monitoring quantity, not a certificate: block-coordinate descent on a non-convex functional with lagged factors is not guaranteed monotone, and period-two orbits appear as non-monotone $F _ { T }$ , which is the criterion by which damping is selected. Uniform $r _ { t } ,$ zero weights and flat parameter beliefs recover plain belief propagation.

## B.4 Generic Scheme for Arbitrary T

The corrections (12) are additive in t and every object of Section 4.3 is per-factor, so the family carries one block belief per factor, one shared context belief, and two global, frozen Dirichlet beliefs,

$$
\{ q _ { y x , t } , q _ { \mathrm { t r i p } , t } \} _ { t = 1 } ^ { T } , \quad \mathrm { s i n g l e t o n s ~ i n c l . ~ } q ( \theta _ { c } ) , \quad q _ { \theta _ { A } } = \mathrm { D i r } ( \alpha _ { A } ) , q _ { \theta _ { B } } = \mathrm { D i r } ( \alpha _ { B } ) .\tag{46}
$$

The one-count posteriors and $m _ { A } , m _ { B }$ are time-independent; the cavities, lagged conditionals and weights are time-indexed. Lemmas 4–5 apply per step and give the factor functions $( 1 7 ) \mathrm { - } ( 1 8 )$ at every t. The graph is the temporal chain of [13] with $\theta _ { c }$ attached to every factor, so the sweep is loopy belief propagation as there: with state potential $\phi _ { t } ( x _ { t } ) = \hat { p } _ { x } ( x _ { t } ) \mu _ { \mathrm { o b s } _ { t } \to x _ { t } } ( x _ { t } )$ and context messages $\mu _ { \theta _ { c } \to a } = q ( \theta _ { c } ) / \mu _ { a \to \theta _ { c } }$

$$
\vec { m } _ { t } ( x _ { t } ) \propto \sum _ { x _ { t - 1 } , u _ { t } , \theta _ { c } } \bar { g } _ { \mathrm { d y n } , t } p ( u _ { t } ) \mu _ { \theta _ { c }  \mathrm { d y n } _ { t } } \phi _ { t - 1 } ( x _ { t - 1 } ) \vec { m } _ { t - 1 } ( x _ { t - 1 } ) ,\tag{47}
$$

$$
\overleftarrow { m } _ { t } ( x _ { t } ) \propto \sum _ { x _ { t + 1 } , u _ { t + 1 } , \theta _ { c } } \bar { g } _ { \mathrm { d y n } , t + 1 } p ( u _ { t + 1 } ) \mu _ { \theta _ { c } \to \mathrm { d y n } _ { t + 1 } } \phi _ { t + 1 } ( x _ { t + 1 } ) \overleftarrow { m } _ { t + 1 } ( x _ { t + 1 } ) ,\tag{48}
$$

with $\overrightarrow { m } _ { 0 } = p ( x _ { 0 } ) , \phi _ { 0 } \equiv 1 , \overleftarrow { m } _ { T } \equiv 1 ;$ then $q ( x _ { t } ) \propto \phi _ { t } \overrightarrow { m } _ { t } \overleftarrow { m } _ { t } .$ the block beliefs follow from (41) with $\mu _ { x _ { t } \to \mathrm { o b s } _ { t } } = \hat { p } _ { x } \overrightarrow { m } _ { t } \overleftarrow { m } _ { t } , \mu _ { x _ { t - 1 } \to \mathrm { d y n } _ { t } } = \phi _ { t - 1 } \overrightarrow { m } _ { t - 1 } , \mu _ { x _ { t } \to \mathrm { d y n } _ { t } }$ = $\phi _ { t } \overleftarrow { m } _ { t } .$ , and $q ( \theta _ { c } )$ from (42). At $T { = } 1$ this reduces to (40)–(42). Because every copy is drawn i.i.d. from the same frozen $\operatorname { D i r } ( \alpha )$ , novelty is scored per factor against the same belief, with no within-horizon diminishing returns; replanning after every real step bounds this to one horizon, and the causal alternative $\begin{array} { r } { \alpha _ { t } = \alpha + \sum _ { s < t } \bar { n } _ { s } } \end{array}$ (assumed-density filtering along the horizon) retains conjugacy but makes the posteriors and m per-sweep objects. Salience has no such issue: $\theta _ { c }$ is shared, so a cue at step s updates $q ( \theta _ { c } )$ for every step.

## B.5 Limits

All Dirichlet-dependent quantities are available in closed form, so conjugacy is exact and the block-layer cost is independent of the Dirichlet parameter-space size. As $q _ { \theta _ { A } } , q _ { \theta _ { B } } \to \delta _ { \theta ^ { * } }$ , posterior and cavity coincide, both weights vanish pointwise, m $\to p ( \cdot \mid \theta ^ { * } , \theta _ { c } )$ , and (17)–(18) become the EFE factor functions (13a)– (13b) of [12, 13] with their enumerable θ read as $\theta _ { c } \mathbf { : }$ inference over K known models is the point-mass limit of learning K matrix pairs. The approach is continuous, $D _ { \mathrm { K L } } [ \mathrm { D i r } ( \alpha + e ) \| \operatorname { D i r } ( \alpha + \bar { n } ) ] \to 0$ as counts grow. At $K = 1$ the salience ratio is identically one and the scheme reduces to Dirichlet learning alone, the setting of Section 5. The two epistemic drives difer in scale: novelty is $O ( 1 / \alpha _ { 0 } )$ and vanishes as counts grow, salience is scale-free and vanishes only when $\theta _ { c }$ is resolved; novelty couples no columns, salience couples all of them through the single shared $\theta _ { c }$

## C Experimental Details

## C.1 Environments and Protocol

Shared protocol. Four cardinal moves (N, E, S, W; walls and grid edges bump), $K = 1$ , one unknown block initialized as a flat Dirichlet with mass 0.1 per entry, the known block clamped by counts of $1 0 ^ { 6 }$ times the true matrix. Every planner runs the same exact filter under the posterior-mean matrices $m _ { A } , m _ { B }$ and the count update of Algorithm 1, line 10. Planning: 100 sweeps at damping 0.5, uniform $\hat { p } _ { y }$ , horizon $H _ { s } = \operatorname* { m i n } ( H , T - s + 1 )$ at step s of a T-step episode as in [12], argmax of $q ( u _ { 1 } ) { \mathrel { \left. \kern - delimiterspace \right.} 2  }$ counts persist across episodes, pose and belief reset, seeds are shared across planners. EFE enumerates the $\bar { 4 } ^ { H }$ open-loop policies under $m _ { A } , m _ { B }$ and takes the argmax of exp(−G), G being the per-policy sum of (7).

Fog-walls maze. $5 \times 5$ cells $( x , y )$ , y upwards; start (0, 2), goal (4, 2). A is the identity except on the fog block (1, 1), (1, 2), (1, 3), (2, 2), (3, 2), whose columns are uniform over the block, and the noisy cell (2, 0) (mass 0.2 on itself, 0.8 over its neighbours). B is deterministic except at the drift cell (1, 2): intended move $\mathrm { w . p . \ 0 . 4 }$ , north or south w.p. 0.3 each. Cells $( 2 , 1 ) , ( 2 , 3 ) , ( 3 , 1 ) , ( 3 , 3 )$ are walls: the direct 4-step route crosses drift and fog, the 8-step top and bottom rows do not (the latter passes the noisy cell). $T = H = 8 ;$ episodes end at the goal. Learn-A: all 25 columns of A flat, B known; the matrix error is the mean of $| m _ { A } - A |$ over all entries. This is the fog maze of [12] with its four sinks replaced by walls.

Unmapped detour, doors variant. $5 \times 4$ grid $( N = 2 0 )$ , start (0, 1), goal (4, 2), identity A, walls (1, 2), (2, 2), (3, 2). The shortcut ends at the ice (4, 1), whence NORTH reaches the goal $\mathrm { w . p . ~ } p = 0 . 5$ and otherwise slips into the absorbing dead end (4, 0). The detour $\uparrow \uparrow \to \to \to \to \downarrow$ is deterministic through (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3); exactly its six (cell, action) columns start flat, the other actions at those cells (bumps or the step back) are known. $T = 8 ,$ $H = 7 ;$ episodes end at the goal or the dead end, the last transition folded into the counts. Detour = goal reached with every visited cell on the detour; $\mathrm { e r r } _ { B } =$ mean $| m _ { B } - B |$ over the six door columns (prior value 0.095). Frozen B keeps the door columns flat and never updates them; the cells variant flattens all four actions of the six cells.

## C.2 Controls and Robustness

Fog-walls, A, B known (30 seeds). At $H \ = \ 8$ every SP-EFE variant puts $q ( u _ { 1 } )$ entirely on NORTH and takes the top row on $3 0 / 3 0$ seeds; SP-Pref-Plan $( q ( \mathrm { E A S T } ) ~ = ~ 0 . 7 3 )$ and SP-Pref (0.32) go into the fog and score $1 8 / 3 0$ and $1 7 / 3 0 \colon$ they plan as if they will know where they land, and the planning correction sharpens that commitment. At $H = 1 1$ bumping after a wrong landing still reaches the goal and the ceiling closes to 29/30 (SP-Pref 22/30). The learn-A policies of $\mathrm { S P - P r e f ( - P l a n ) }$ equal their known-A policies: their goal rate is the luck of the drift, not a function of $m _ { A }$

Unmapped detour. Known B: SP-Pref $( q ( \mathrm { E A S T } ) = 0 . 3 0 )$ takes the ice because the uncorrected Bethe term $- \mathbb { H } [ q ( u _ { t } \ | \ x _ { t - 1 } ) ]$ rewards states in which all actions look alike, which absorbing cells do: a sink costs $4 - \log 4 \approx 2 . 6$ nats per step instead of 4. Frozen B: SP-Pref-Plan gambles $( q ( \mathrm { E A S T } ) = 0 . 7 0$ , zero door visits); $\mathrm { S P \mathrm { - E F E ^ { - d y n } } }$ farms the first door (0 goals: $\varDelta _ { B }$ never decays); SP-EFE, $\mathrm { S P - E F E ^ { - \Delta _ { B } } }$ and SP-Pref reach the goal via the detour on 300/300 episodes without learning: with frozen flat doors “most uncertain column” and “detour” are the same action at every cell, so any entropy follower is walked along the route (<sup>†</sup> in Table 1).

Robustness. Action indices permuted (W, S, E, N; 200 episodes): $\mathrm { S P \mathrm { - E F E ^ { - d y n } } }$ 198, SP-Pref 101 (25 detours, then the ice), SP-Pref-Plan 89. Mirrored $6 \times 4$ layout, 7 doors, $H = 8 \ ( 3 0 0$ episodes × 3 seeds): SP-EFE 300, SP-EF $\mathrm { E ^ { - d y n } ~ 2 9 9 }$ SP-Pref $2 0 0 \pm 6$ , SP-Pref-Plan 150±5, EFE 100/100. Cells variant (200 episodes $\times ~ 5$ seeds): SP-EFE<sup>−dyn</sup> 159, SP-Pref-Plan 97.6 ± 12.5 with $m _ { B }$ bit-identical to its prior, SP-Pref 185 (of-path flat columns pay its bonus forever), SP-EFE 35.