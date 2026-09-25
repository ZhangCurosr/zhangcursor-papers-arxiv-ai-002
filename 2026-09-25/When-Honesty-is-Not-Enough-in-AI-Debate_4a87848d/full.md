# When Honesty is Not Enough in AI Debate

Rayne Holland<sup>1</sup> Liming Zhu<sup>1</sup> Jason Xue<sup>1</sup> <sup>1</sup>CSIRO

## Abstract

Scalable oversight aims to verify the behaviour of agents whose capabilities exceed those of their overseers. AI debate has been proposed as an oversight solution in which competing agents help a resource-limited verifier assess claims that it cannot reliably evaluate unaided. Much of its promise rests on incentivizing honest arguments that lead to correct verdicts. Yet a correct verdict need not uniquely determine the arguments used to support it. Agents may retain discretion over which correct claims to present, how to frame them, and in what order to disclose them. This residual freedom can allow agents to shape what the verifier learns beyond the task-relevant conclusion, pursuing latent objectives without compromising verdict correctness.

To study this phenomenon, we introduce the framework strategic interactive oversight (SIO), which treats oversight jointly as a verification mechanism and a strategic communication channel. Within this framework, we formalise the notion of task-admissible latent optimisation, which entails the pursuit of latent objectives while maintaining a prescribed task performance. As proof-of-concept, we instantiate SIO in the establish protocol debate with cross-examination and quantify a tradeof between task success and information disclosure about a hidden variable. The trade-of identifies a strategic window in which substantial disclosure remains compatible with task admissibility. Towards mitigation, we reduce admissible bias by expanding the cross-examiner’s role to mitigate persistent disclosure over finite interaction horizons. Our results highlight the need to evaluate oversight not only by the correctness of its verdicts, but also by the information conveyed through its transcripts.

## 1. Introduction

As AI systems grow more capable, they may surpass the humans or weaker models charged with evaluating their work. Scalable oversight asks whether a bounded overseer can nevertheless evaluate and steer a more capable system using substantially fewer resources. An oversight mechanism must remain reliable even when the overseer cannot reproduce the reasoning or computation it assesses.

Proposed approaches include amplification (Christiano et al., 2018), recursive reward modelling (Leike et al., 2018), and AI debate (Irving et al., 2018; Brown-Cohen et al., 2024, 2025, 2026). Each uses AI capabilities to amplify the signal supplied by human judgment. In debate, powerful agents recursively reduce a dificult question to disagreements that a computationally limited judge can resolve. Irving et al. (Irving et al., 2018) formalise this intuition using complexity theory, showing how a weak verifier can judge claims about computations beyond its own reach. Later protocols require the verifier to inspect only a small part of a much larger computation (Brown-Cohen et al., 2024, 2026).

This line of work has largely studied what can be verified under an asymmetry in computational resources. Although the verifier cannot reproduce the prover’s computation, an appropriate protocol may still allow it to check the resulting claim. Yet computational verifiability alone does not capture the full range of limitations faced by human overseers. Human oversight is also constrained in how verification is carried out, since the framing of (truthful) evidence can shape how it is interpreted. Indeed, recent work illustrates how LLMs can exploit the use of evidence to manipulate beliefs without relying on directly false claims (Hu et al., 2026). An oversight interaction may therefore verify the right proposition while inducing a strategically chosen interpretation of the evidence or influencing subsequent judgments. This “how” within the computationally verifiable “what” is the object of study here. Understanding and controlling it is essential if scalable verification is to provide a meaningful alignment guarantee.

The central distinction is between honesty as a guarantee about the assessed claim and safety as a property of the interaction as a whole. Even when an oversight protocol elicits a correct verdict and constrains reports to be locally accurate, it need not control which additional information the transcript conveys. In this sense, honesty is not enough and guarantees about task correctness do not automatically extend to the other efects of task-successful communication.

## 1.1. Solution

To study this residual freedom, we introduce a model of strategic interactive oversight (SIO), which treats an oversight protocol as both an evaluation mechanism and a communication channel. An agent’s performance is evaluated under an explicit task objective, which specifies the protocol’s intended outcome, and a latent objective, which captures additional efects of the interaction. The agent may then select strategies that optimise performance across both objectives. This yields the threat model of task-admissible latent optimisation (TALO), in which the agent pursues the latent objective while maintaining an acceptable level of task performance. The threat does not require the agent to induce an incorrect verdict or violate the protocol’s explicit rules. It arises whenever the task objective admits multiple successful strategies that difer in their latent efects on the verifier’s view of the interaction.

As a proof-of-concept, we instantiate SIO on debate with cros-examination Brown-Cohen et al. (2024) with the latent objective of hidden-hypothesis disclosure. Hidden-hypothesis disclosure measures how much information the verifier’s view of the interaction reveals about a latent variable that is not required for the task. In standard debate, the task objective is to win. However, there are generally many strategies by which it can do so. This non-uniqueness allows the debater to select, among task-successful strategies, those that additionally encode information about the latent variable. We show that, even when the disclosure in each round is small, the total information revealed can grow linearly with the number of cross-examination rounds. Local report-accuracy conditions and a correct terminal verdict therefore do not prevent the oversight interaction from becoming a channel for latent information.

Because TALO is a constrained optimisation problem, mitigation requires altering the incentive landscape over task-admissible strategies. As a step towards mitigation, we expand. In addition to crossexamination, the cross-examiner assigns these weights so that the aggregate (if it cannot be disputed) remains aligned with a human oracle. Debaters that persistently deviate from oracle-aligned judgments receive less weight and therefore exert less influence on the final argument.

## 1.2. Contributions

Our contributions are threefold. First, we introduce SIO, a new framework that treats oversight jointly as a verification mechanism and a strategic communication channel. Building on scalable verification under resource asymmetry, it makes explicit the distinction between an agent’s task objective and latent objectives pursued through the interaction. This distinction motivates oversight of the interaction, rather than only its terminal verdict.

A central motivation for debate is that adversarial scrutiny can incentivise truthful, task-correct answers. Our framework identifies a distinct limitation of such guarantees, where constraining an agent’s answer does not necessarily constrain the information conveyed by its choice among task-admissible strategies. We formalise this gap through task-admissible latent optimisation and demonstrate it

![](images/12559f9bb723a6f20845aff8a3eeab3d147bb453f3da32dabb07f5dee841b651.jpg)

![](images/fa6790ce49d67a555d423e08fbb2836d2b76dd32d6209cddb3103fbb9206e4ae.jpg)  
(a) Ψ vs. |∆|  
(b) Φ vs. |∆|  
Figure 1: Tradeof between task success $\Phi$ and hidden hypothesis disclosure Ψ in debate with crossexamination. ∆ denotes the “magnitude” by which a debater biases a neutral claim and $R _ { i }$ is a measure of computational resources. On the left, Ψ measures (in bits) the volume of information about a hidden hypothesis encoded in the debate transcript. It grows quadratically with the size of the bias, before hitting the prior hypothesis entropy h. On the right, Φ measures the probability the debater wins the debate (that is, task success) and $\gamma$ is the level of task admissible performance. Notably, small biases do not afect the debate outcome. Increasing resources can extend the window of admissable stratgies.

through hidden-hypothesis disclosure.

Second, we instantiate the framework in a computational model of debate. We establish a non-trivial attack surface, which comprises a range of strategies through which a debating agent can advance a hidden latent objective without sacrificing its task objective. Figure 1 illustrates this strategic window.

Third, as a step towards mitigation, we augment cross-examination with a sliding-window consistency check that tests for persistent discrepancies across rounds. This narrows the range of persistent perturbations compatible with task success, at the cost of additional verifier work, while leaving the debaters’ per-round sampling work unchanged. The resulting guarantee limits the disclosure strategy studied here rather than establishing transcript confidentiality.

## 2. Preliminaries

For $n \in \mathbb { N } .$ , let $[ n ] = \{ 1 , \dots , n \}$ . Vectors are written in boldface; for $\mathbf { x } \in \{ 0 , 1 \} ^ { n }$ and $I \subseteq [ n ]$ , let $\mathbf { x } _ { I }$ denote the restriction of x to the coordinates in I.

## 2.1. Computation Model

We model computation using probabilistic oracle Turing machines. A stochastic oracle with query length $\ell = \ell ( n )$ maps each query $z \in \{ 0 , 1 \} ^ { \ell }$ to a $\{ 0 , 1 \}$ -valued random variable $\mathcal { O } ( z )$ . Responses to oracle queries are independent, including responses to repeated instances of the same query. A probabilistic oracle Turing machine M may write a query z to its oracle tape and receive a sample from $\mathcal { O } ( z )$ in one step. We write $M ^ { \mathcal { O } }$ for M with access to $\mathcal { O } .$ In the oversight setting, M represents a computation specified by natural-language instructions, while O represents human judgement or other external black-box feedback, such as search results, sensor observations, or API outputs Brown-Cohen et al. (2024).

A language $L \subseteq \{ 0 , 1 \} ^ { * }$ is a set of finite binary strings. A probabilistic oracle Turing machine M decides

L with oracle O if

$$
x \in L \implies \operatorname* { P r } \left[ M ^ { \mathcal { O } } ( x ) = 1 \right] > \frac { 2 } { 3 } , \qquad x \notin L \implies \operatorname* { P r } \left[ M ^ { \mathcal { O } } ( x ) = 1 \right] < \frac { 1 } { 3 } ,
$$

where the probability is over the internal randomness of M and the stochastic oracle responses.

A transcript of a T-step execution of $M ^ { \mathcal { O } }$ on input x is a random vector $\mathbf { y } = ( \mathbf { y } _ { 1 } , \ldots , \mathbf { y } _ { T } ) \in \{ 0 , 1 \} ^ { T }$ where ${ \bf y } _ { t }$ is the (random) bit written at the current head position at step t. We assume that the final coordinate is the machine’s output, so $Y _ { T } = M ^ { O } ( x )$ . In the language-model setting, the transcript corresponds to the sequence of tokens produced during the computation. For a realization y of ${ \bf y } ,$ let $I _ { M , x } ( t ) \subseteq [ T ]$ denote the transcript coordinates read by M when producing the realization $y _ { t }$ of $\mathbf { y } _ { t } .$ When M and x are clear from context, we write $I ( t )$ . Thus, the distribution of $\mathbf { y } _ { t }$ is determined by the relevant prior coordinates $\mathbf { y } _ { I ( t ) }$ , together with fresh machine and oracle randomness.

## 2.2. Information Theory

We use standard information-theoretic quantities to measure uncertainty, statistical dependence, and divergence between distributions. All random variables below are discrete, and all logarithms are base two. Consequently, entropy, mutual information, and divergence are measured in bits.

Definition 1 (Entropy). For a discrete random variable X with probability massfunction $P _ { X } ,$ , its Shannon entropy is $\begin{array} { r } { H ( X ) = - \sum _ { x } P _ { X } ( x ) } \end{array}$ log $P _ { X } ( x )$ . For jointly distributed random variables X and $\boldsymbol { Y } ,$ the conditional entropy of X given Y is $\begin{array} { r } { H ( X \mid Y ) = - \sum _ { x , y } P _ { X , Y } ( x , y ) \log P _ { X \mid Y } ( x \mid y ) } \end{array}$

Entropy measures the uncertainty associated with a random variable.

Definition 2 (Mutual information). For jointly distributed random variables X and $\boldsymbol { Y } ,$ their mutual information is $I ( X ; Y ) = H ( X ) - H ( X \mid Y )$ . For jointly distributed $X , Y , Z ,$ , their conditional mutual information is $I ( X ; Y \mid Z ) = H ( X \mid Z ) - H ( X \mid Y , Z )$

Mutual information quantifies the statistical dependence between two random variables.

## 2.3. Oversight Protocols

An oversight protocol is an interaction between k probabilistic prover machines and a probabilistic verifier, each of which may query the stochastic oracle O. On input x, the machines exchange messages according to a prescribed communication schedule, after which the verifier outputs a decision bit. Agent internal randomness and oracle responses induce a distribution over protocol executions.

The interaction history after t messages is $\mathbf { h } _ { t } = \left( \mathbf { h } _ { 1 } , \ldots , \mathbf { h } _ { t } \right)$ , where $\mathbf { h } _ { j }$ records a message together with its sender and recipient. Oracle queries and their responses are treated as messages between the querying participant and O. We write ${ \bf h } = { \bf h } _ { T }$ for the complete interaction history, where T is the termination time. For each agent A, let $\Pi _ { A } ( { \mathbf { h } } _ { t } )$ denote the projection of $\mathbf { h } _ { t }$ onto the messages sent or received by A before time t.

Equivalently, the protocol may be represented by a rooted tree whose nodes correspond to agents and edges correspond to message exchanges. In other words, each non-leaf node specifies the participant acting next, and each outgoing edge records a possible message. A leaf represents a completed interaction and is labeled by the verifier’s terminal decision.

## 3. Framework

We formalize SIO by separating an agent’s stated task objective from any latent objective it may pursue through the oversight interaction. As a running example, consider a debate in which the input specifies

a task hypothesis and two provers attempt to persuade a verifier to accept or reject it.

## 3.1. Objectives

Let X be the input space and H the set of complete interaction histories. A task objective for agent A is a function $\Phi _ { A } \colon \mathcal { X } \times \mathcal { H } $ R that evaluates task performance, and a latent objective for agent A is a function $\Psi _ { A } \colon \mathcal { X } \times \mathcal { H } $ R that evaluates an additional property of the interaction. For a complete history h on input $x ,$ the combined objective is

$$
U _ { A } ( x , \mathbf { h } , \lambda ) : = \Phi _ { A } ( x , \mathbf { h } ) + \lambda \Psi _ { A } ( x , \mathbf { h } ) ,\tag{1}
$$

where the shadow price $\lambda \geq 0$ specifies the relative importance of the latent objective. In the debate example, the task objective $\Phi _ { A } = \mathbb { 1 } \{ A$ wins debate} indicates that the explicit goal of agent A is to win the debate. The decomposition into Φ and Ψ distinguishes what the protocol is intended to evaluate from other efects that an agent may induce through its messages.

## 3.2. Strategies

A strategy $\sigma _ { A }$ for an agent A assigns a distribution over the messages available to A at each of its decision points. Formally, given input x and the local interaction history $\Pi _ { A } ( { \mathbf { h } } _ { t } )$ , the strategy specifies a distribution $\sigma _ { A } \big ( \cdot \mid x , \Pi _ { A } ( \mathbf { h } _ { t } ) \big ) \in \Delta ( \mathcal { M } _ { A } )$ , where $\mathcal { M } _ { A }$ is the message space available to A and $\Delta ( \mathcal { M } _ { A } )$ is the set of probability distributions over that space. Thus, an agent’s message depends only on the input and the messages it has sent or received, including its own oracle queries and responses.

For a set of provers $\{ A _ { i } \} _ { i \in [ k ] }$ and a verifier $V ,$ a strategy profile $\boldsymbol { \sigma } = \left( \sigma _ { A _ { 1 } } , \ldots , \sigma _ { A _ { k } } , \sigma _ { V } \right)$ , together with the stochastic oracle, induces a distribution over complete histories. Let $\mathbf { h } ( x , \sigma )$ denote the resulting random history. The task and latent outcomes, for $A _ { i } ,$ induced by are the random variables

$$
\Phi _ { A _ { i } } ( x , \sigma ) = \Phi \big ( x , \mathbf { h } ( x , \sigma ) \big ) , \qquad \Psi _ { A _ { i } } ( x , \sigma ) = \Psi \big ( x , \mathbf { h } ( x , \sigma ) \big ) .
$$

When x is clear from context, we write $\Phi _ { A _ { i } } ( \sigma )$ and $\Psi _ { A _ { i } } ( \sigma )$

## 3.3. Threat Model

The central threat is that an agent may optimize a latent objective without materially compromising its performance on the stated task. Let $\sigma _ { - A }$ denote the strategies in the profile <sub>σ</sub> with $\sigma _ { A }$ removed. Fix $\sigma _ { - A }$ and, for a task-admissibility threshold $\gamma \geq 0$ , define the set of -task-admissible strategies by

$$
\Sigma _ { A } ^ { \gamma } ( x ) = \{ \sigma _ { A } : \mathbb { E } [ \Phi _ { x } ( \sigma _ { A } \circ \sigma _ { - A } ) ] \geq \gamma \} .
$$

For a fixed shadow price $\lambda \geq 0$ , we say that TALO occurs when an agent selects a strategy that optimizes the latent objective subject to task-admissibility:

$$
\operatorname* { m a x } _ { \sigma _ { A } \in \Sigma _ { \gamma } ( x ) } \mathbb { E } \big [ \Phi _ { x } ( \sigma _ { A } , \sigma _ { - A } ) + \lambda \Psi _ { x } ( \sigma _ { A } , \sigma _ { - A } ) \big ] .
$$

The threat arises when the task admits many near-optimal strategies with diferent latent efects. For example, during debate, there may be many arguments that win the debate, allowing a prover to select those that also advance its latent objective.

## Debate protocol for a stochastic oracle

All parties have access to ${ \mathcal { O } } ,$ an input $x \in \{ 0 , 1 \} ^ { n }$ , and a K-Lipschitz probabilistic oracle machine M. A claims that $\begin{array} { r } { \operatorname* { P r } [ \hat { M } ( x ) = 1 ] \ge \frac { 2 } { 3 } . } \end{array}$ , and B disputes this claim.

1. Let $\mathbf { y } = ( \mathsf { y } _ { 1 } , \hdots , \mathsf { y } _ { T } )$ be the random variable representing the transcript of $M ( x )$ and set $d = \lceil 1 5 0 K \rceil$

2. For debate round $t \in [ T ]$

(a) A outputs an estimate $\hat { p } _ { t } \in [ 0 , 1 ]$ of $p _ { t } = \mathrm { P r } \Big [ \mathbf { y } _ { t } = 1 \Big | \mathbf { y } _ { I ( t ) } = \mathbf { a } _ { I ( t ) } \Big ]$

(b) A queries an independent copy of B for $z _ { t } ^ { A } \sim \mathsf { U } [ 0 , 1 ] .$ , and B queries an independent copy of A for $z _ { t } ^ { B } \sim \mathsf { U } [ 0 , 1 ]$

(c) Set $\boldsymbol { z } _ { t } = \left( z _ { t } ^ { A } + z _ { t } ^ { B } \right)$ mod 1.

(d) A sets $a _ { t } = 1 \mathrm { i f } z _ { t } \le \hat { p } _ { t }$ , and $a _ { t } = 0$ otherwise.

(e) B can abort if (it believes) A has not followed the protocol.

3. If the protocol was aborted in round t:

(a) V draws $r = 1 9 2 d ^ { 2 }$ log 100 independent samples<sup>a</sup> of $\mathbf { y } _ { t } ,$ conditioned on $\mathbf { y } _ { I ( t ) } = \mathbf { a } _ { I ( t ) } ,$ , and uses their sample mean to obtain an estimate $\hat { p } _ { t } ^ { \mathcal { O } }$ of $\mathrm { P r } \left\lceil \mathbf { y } _ { t } = 1 \right\rceil \mathbf { y } _ { I ( t ) } = \pmb { a } _ { I ( t ) } \rceil$

(b) V checks whether $\begin{array} { r } { \left| \hat { p } _ { t } ^ { \mathcal { O } } - \hat { p } _ { t } \right| \geq \frac { 1 } { 4 d } } \end{array}$ . If the inequality holds, V outputs 0; otherwise, V outputs 1.

4. If the protocol was not aborted, then the verifier outputs $a _ { T } .$

${ } ^ { a } \mathrm { I f } \ \mathrm { y } _ { t }$ is supposed to be the output of an oracle query, these samples are obtained using r independent queries to $\mathcal { O } ( { \pmb a } _ { I ( t ) } )$ . Otherwise, $\mathtt { y } _ { t }$ is a deterministic function of ${ \pmb a } _ { I ( t ) }$ given by one step of M.

Figure 2: Doubly-eficient debate protocol for a stochastic oracle Brown-Cohen et al. (2024).

## 3.4. Hidden-Hypothesis Disclosure

To provide a concrete instance of SIO, we instantiate the latent objective as the disclosure of information about a hidden hypothesis. Let Q be a discrete random variable representing a hidden hypothesis that is not part of the stated task, and let X be the protocol input. The verifier V observes only its projection $\Pi _ { V } ( { \bf h } )$ of the interaction history.

Definition 3. The hidden-hypothesis disclosure induced by a strategy profile <sub>σ</sub> is

$$
\Psi _ { \sf h h d } ( \sigma ) : = I ( Q ; \Pi _ { V } ( \mathbf { h } ( X , \sigma ) ) \mid X ) .
$$

This quantity measures how much additional information the messages available to the verifier reveal about H, beyond the information already contained in the task input. In particular, $\Psi _ { \sf h h d } ( \sigma ) = 0$ precisely when H and the verifier’s observed messages are conditionally independent given X.

For example, in debate, X specifies the task hypothesis, while H represents an additional hypothesis that the verifier is not required to assess. A prover exhibits task-admissible hidden-hypothesis disclosure if it selects arguments that preserve its probability of winning while increasing $I ( H ; \Pi _ { V } ( \mathbf { h } ( X , \sigma ) ) \mid X )$ This definition concerns the information available to the verifier rather than whether the verifier consciously identifies or correctly interprets H. <sup>1</sup>

## 4. Debate with Cross-Examination

To illustrate the use of SIO, we study a computation model for debate with cross-examination introduced by Brown-Cohen et al. (2024).

## 4.1. Protocol

The following outline is formalized in Figure 2. For a probabilistic Turing machine M and input x, the debate setup comprises of a prover A claiming that $\begin{array} { r } { \operatorname* { P r } [ M ( x ) = 1 ] \ge \frac { 2 } { 3 } , } \end{array}$ , while a cross-examiner B argues against this claim. At round t in the debate, a prover A reports a claim $\hat { p } _ { t }$ about the conditional probability of the next bit in the machine transcript. Prover $B ,$ acting as the cross-examiner, may either accept the claim and allow the next bit to be sampled, or abort and assert that A’s claim is inaccurate. Upon an abort, the verifier adjudicates the disputed claim with the assistance of the stochastic oracle O and awards the debate to the prover whose position is correct. If the protocol terminates without an abort, A is awarded the win.

The significance of this protocol, in the context scalable oversight, is that it is doubly eficient. In other words, the provers operate in time polynomial in the size of the input and the verifier is linear in the length of each oracle query and makes a sub-linear number of queries to the oracle. Thus, debate extends reliable oversight to computations that the verifier could not feasibly perform itself.

For an input x and strategy profile $\sigma = ( \sigma _ { A } \circ \sigma _ { - A } )$ , let $\Phi _ { A } ( x , \mathbf { h } ) \in \{ 0 , 1 \}$ indicate whether the verifier rules in favor of A on terminal history h. Under SIO, for input $y = \left( M , x \right)$ , the task objective is defined by

$$
\Phi _ { A } ( x , \sigma ) = \mathbb { E } _ { \sigma } \left[ \Phi _ { A } ( x , \mathbf { h } ( x , \sigma ) ) \right] = \operatorname* { P r } _ { \sigma } [ V \to 1 ] .
$$

Thus, $\Phi _ { A } ( x , \sigma )$ is the probability that A wins the debate under $\sigma ,$ with the expectation taken over all protocol randomness. The protocol has the following utility on the task objective.

Lemma 1 (Task Completeness Brown-Cohen et al. (2024)). Let L be any language decidable by a K-Lipschitz probabilistic oracle Turing machine M. Then, for every $x \in L ,$ , there exists a strategy $\sigma _ { A }$ such that, for every strategy opposing $\sigma _ { - A } ,$

$$
\Phi _ { A } \mathopen { } \mathclose \bgroup \left( \mathopen { } \mathclose \bgroup \left( M , x \aftergroup \egroup \right) , \sigma _ { A } \circ \sigma _ { - A } \aftergroup \egroup \right) \geq \frac { 2 } { 3 } .
$$

## 4.2. Hidden-hypothesis disclosure

We now examine whether task success prevents a prover from pursuing an additional latent objective. In the debate protocol, under the claim $\begin{array} { r } { { ^ { \mathfrak { c } } } \mathbf { \bar { P } } \mathbf { r } [ M ( x ) = 1 ] \geq \frac { 2 } { 3 } \mathbf { \mathfrak { ) } } } \end{array}$ , A is required to report suficiently accurate estimates of the conditional probabilities of the machine transcript of $M ( x )$ . This requirement does not, however, necessarily determine a unique report. Significantly, approximately task-equivalent reports leave residual strategic freedom that A can use to disclose information about a hidden hypothesis binary $Q \in \{ - 1 , 1 \}$

The projection $\Pi _ { V } \big ( \mathbf { h } _ { T } ( X , \sigma ) \big )$ includes all information available to the verifier. For example, if no abort occurs, the verifier observes the sampled machine transcript $\Pi _ { V } \big ( \mathbf { h } _ { T } ( X , \sigma ) \big ) = a$ . If the cross-examiner aborts at round $t ,$ the verifier instead observes the partial transcript together with the evidence used to adjudicate the disagreement. Suppressing public randomness and other deterministic protocol data, this projection has the form

$$
\Pi _ { V } \big ( \mathbf { h } _ { t } ( X , \sigma ) \big ) = \Big ( \mathtt { a } _ { 1 } , \dots , \mathtt { a } _ { t } , \mathtt { a b o r t } , \hat { p } _ { t } , \{ \mathcal { O } ( { \pmb a } _ { I ( t ) } ) \} \Big ) .
$$

Here, $\hat { p } _ { t }$ is the conditional-probability report disputed at round $t , I ( t )$ identifies the location of the machine configuration examined by the verifier, and $\{ \mathcal { O } ( { \pmb a } _ { I ( t ) } ) \}$ denotes the collection of oracle samples used in the resulting statistical test.

Following Definition 3, hidden-hypothesis disclosure admits the latent objective

$$
\Psi _ { A } ( x , \sigma ) = I \left( Q ; \Pi _ { V } { \big ( } \mathbf { h } _ { T } { \big ( } X , \sigma ) { \big ) } \mid X = x \right) .
$$

This quantity measures how much the verifier’s observation of the debate reveals about H, beyond what is already known from the input x. The hidden hypothesis may be naturally connected to the subject of the debate without being part of its stated task objective. For example, the debate may concern the task claim “the bridge is unsafe,” while the prover’s hidden hypothesis is “the damage resulted from neglected maintenance.” The prover can establish that the bridge is unsafe while selecting evidence (that is, perturbing arguments) that subtly points toward maintenance failures. An example of how a Turing machine transcript can leak information about a hidden hypothesis is given in Appendix A.

## 4.3. Prover Strategies

At round $t ,$ let $p _ { t } = \mathrm { P r }  \mathbf { y } _ { t } = 1  \mathbf { y } _ { I ( t ) } = \pmb { a } _ { I ( t ) } ]$ denote the true conditional probability of the next machine-transcript bit. The honest strategy of Lemma 1 is for A is to draw R independent samples from the relevant conditional computation and form the empirical estimate $\begin{array} { r } { \bar { p } _ { t } ^ { ( R ) } = \frac { 1 } { R } \sum _ { j = 1 } ^ { R } \mathbf { y } _ { t , j } , } \end{array}$ , where $\mathbf { y } _ { t , j } \sim \mathsf { B e r n } ( p _ { t } )$ , and reporting this value. Subsequently, for parameter $\Delta > 0 , A$ can perturb the honest strategy by reporting the value

$$
\begin{array} { r } { \hat { p } _ { t } = \mathsf { c l i p } _ { [ 0 , 1 ] } \left( \bar { p } _ { t } ^ { ( R ) } + \Delta Q \right) . } \end{array}
$$

Formally, the corresponding strategy is

$$
\begin{array} { r } { \sigma _ { A } ^ { R , \Delta } \left( \hat { p } _ { t } \mid x , \Pi _ { A } ( { \bf h } _ { t - 1 } ) , Q \right) = \mathrm { P r } \left[ \mathrm { c l i p } _ { [ 0 , 1 ] } \left( \bar { p } _ { t } ^ { ( R ) } + \Delta Q \right) = \hat { p } _ { t } \Big \mid x , \Pi _ { A } ( { \bf h } _ { t - 1 } ) , H \right] . } \end{array}
$$

We abbreviate this strategy by $\sigma _ { A } \sim ( R , \Delta )$ . The sample count R controls the accuracy of the taskrelevant estimate, while $\Delta$ controls the magnitude of the hidden-hypothesis signal.

A strategy $\sigma _ { B }$ for B is to draw a sample estimate $\hat { q } _ { t }$ of $p _ { t }$ at each round by performing R simulations of generating $\mathbf { y } _ { t }$ given ${ \bf y } _ { I ( t ) } = { \pmb a } _ { I ( t ) }$ . Subsequently, B aborts if $\left| \hat { p } _ { t } - \hat { q } _ { t } \right| \geq \tau$ . Under this strategy we say $\sigma _ { B } \sim ( R , \tau )$ . We say that the strategy is honest if the choice of <sub>τ</sub> is sound.

## 4.4. Results

The following describes an accurate task objective performance, against an honest $B ,$ for a range of biased A strategies. Let

Theorem 1. For $K \geq 1$ , let L be any language decided by a K-Lipschitz oracle Turing machine M. Let $\operatorname* { P r } [ M ( x ) = 1 ] = \alpha ^ { * } , R \ge 1 9 2 d ^ { 2 } \log ( 1 0 0 T ) , \Delta < 1 / ( 1 6 d )$ , and $\tau = 1 / ( 2 d )$ . Then, on the protocol of Figure $2 , f o r$ every strategy $\sigma = \sigma _ { A } \circ \sigma _ { B }$ parameterized by $\sigma _ { A } \sim ( R , \Delta )$ and $\sigma _ { B } \sim ( R , \tau )$ , the task objective observes

$$
\left( \alpha ^ { * } - \frac { 1 } { 1 0 0 } \right) \cdot \frac { 2 9 9 } { 3 0 0 } \leq \Phi _ { A } ( x , \sigma ) \leq \left( \alpha ^ { * } + \frac { 1 } { 1 0 0 } \right) + \frac { 1 } { 3 0 0 } ,
$$

The proof is in Appendix B. This covers the admissible region of strategies. For any $t \in [ T ]$ , define

$$
g _ { t } ( \Delta ) = \mathrm { P r } [ | \Delta + \sigma _ { t } z | < \tau ] ,
$$

where $z \sim \mathsf { N } ( 0 , 1 )$ and $\sigma _ { t } = \sqrt { 2 p _ { t } \big ( 1 - p _ { t } \big ) / R }$ . The following bounds the task objective for general perturbations.

Theorem 2. For $K \geq 1$ , let L be any language decided by a K-Lipschitz oracle Turing machine M. Let $R \geq 1 9 2 d ^ { 2 } \log ( 6 0 0 T ) , \Delta \geq 0 ,$ , and $\tau = 1 / ( 2 d )$ . Assume $p _ { t } \in ( 0 . 0 1 , 0 . 9 9 )$ for all machine configurations on M. Then, on the protocol of Figure $2 , f o r$ every strategy $\sigma = \sigma _ { A } \circ \sigma _ { B }$ parameterized by $\sigma _ { A } \sim ( R , \Delta )$ and $\sigma _ { B } \sim ( R , \tau )$ , the task objective observes

$$
\Phi _ { A } ( x , \sigma _ { A } \circ \sigma _ { B } ) \leq \operatorname* { m i n } \left\{ 1 , \mathbb { E } [ g _ { 1 } ( \Delta ) ] + \frac { 1 } { 1 0 0 } \right\} .
$$

The proof is in Appendix B. Next, we provide a characterisation of how changing ∆ afects the latent objective. Let $h _ { x } = H ( Q \mid X = x )$ denote the hidden hypothesis entropy on the task input.

Theorem 3. Let L be any language decided by a K-Lipschitz oracle Turing machine M. Assume $p _ { t } \in$ (0.01, 0.99) for all machine configurations on M and that the prior $\operatorname* { P r } [ Q \mid X = x ]$ contains suficient entropy. Then, for any $\Delta \geq 0$ and any honest cross-examiner strategy $\sigma _ { B } ,$ , there exists $R \in O ( d ^ { 2 } \log ( T ) )$ such that, for $\sigma _ { A } = ( \Delta , R )$ , the latent objective observes

$$
\Psi _ { A } ( x , \sigma _ { A } \circ \sigma _ { B } ) = \Omega \left( h _ { x } \cdot \operatorname* { m i n } \left\{ 1 , \Delta ^ { 2 } T \right\} \right) .
$$

The proof is in Appendix C. The characterization provided by these two theorems is visualized in Figure 1.

## 4.5. Discussion

These results identify a small-perturbation regime, $\Delta < 1 / \left( 1 6 d \right)$ , in which task performance remains close to the reference acceptance probability $\alpha ^ { * }$ , while disclosure grows at least proportionally to $h _ { x } \Delta T$ before saturation. Larger perturbations reduce the Gaussian interval probability governing the task bound at fixed sampling variance, exposing a tension between information transmission and consistency with an honest cross-examiner. Thus, oversight of task accuracy alone does not protect the hidden hypothesis. That is, even accurate transcripts can accumulate substantial disclosure.

Both Theorems 2 and 3 assume that the transition probabilities $p _ { t }$ are bounded away from 0 and 1 and Theorem 3 assumes that the hidden hypothesis has non-negligible entropy. These assumptions isolate the regime of interest in which nearly deterministic steps ofer little scope for perturbation-based disclosure and can be omitted from the leakage analysis. In addition, a hypothesis with negligible entropy already carries little uncertainty before any interaction. The latter case makes confidentiality largely vacuous, rather than demonstrating transcript-induced disclosure.

More broadly, these results motivate treating transcripts as outputs with their own informationdisclosure properties, rather than merely as evidence supporting a final decision. Transcripts retained for auditing, evaluation, or downstream reuse can expose information that task-accuracy metrics do not capture. Oversight should therefore assess both decision quality and cumulative disclosure, including information revealed through aborts. Restricting transcript access, limiting interaction length, or changing reporting precision are natural design directions, although each requires a separate analysis of its efects on correctness and disclosure.

## 5. Towards Mitigation

The vulnerability identified above arises from a mismatch between the task objective and the behavior desired of the debaters. The task objective rewards producing a winning argument. Although this supports scalable verification, it does not uniquely determine how that argument should be produced. A debater may therefore achieve high task performance while pursuing latent objectives through its construction of arguments. Mitigating such behavior requires changing the incentive landscape.

A natural desideratum is oracle alignment. A conditional probability $\hat { p } _ { t }$ generated by the $\mathrm { \Omega } ^ { \mathfrak { s c } } \mathrm { f o r } ^ { \mathfrak { n } }$ debater A should coincide with the corresponding oracle conditional probability. One might attempt to induce this behavior by adding rewards or penalties based on proximity to oracle-generated conditional probabilities. In general, however, this is insuficient. Gains from a latent objective may ofset the task reward for accurate reporting, leaving the debater indiferent among disclosure strategies or even leaving misreporting profitable.

## 5.1. Expanded cross-examination

This limitation motivates giving the cross-examiner B an additional way to challenge A’s reports. The additional challenge entails issuing an abort drift challenge when the reports exhibit a persistent signed deviation over a trailing window of $N \geq 1$ rounds. Define $J _ { t , N } = \{ t - N , \ldots , t - 1 \}$ as the most recent N debate rounds. For every $s \in J _ { t , N }$ , let $p _ { s } = \mathrm { P r } \Big | \mathbf { y } _ { s } = 1 \Big | \mathbf { y } _ { I ( s ) } = \mathbf { a } _ { I ( s ) } \Big |$ denote the conditional probability associated with the transcript prefix fixed at round s. The trailing signed deviation is

$$
D _ { t , N } = \frac { 1 } { N } \sum _ { s \in J _ { t , N } } \left( \hat { p } _ { s } - \hat { p } _ { s } ^ { \mathcal { O } } \right) .
$$

For a constant $c \geq 1$ , the verifier accepts an abort drift challenge when $\begin{array} { r } { | D _ { t , N } | \ \geq \ \frac { 1 } { 4 d \sqrt { N } } } \end{array}$ . Thus, a persistent signed deviation of magnitude b is detectable once $b \geq 1 / ( d \sqrt { N } )$ , even if no individual report is inaccurate enough to support an aborterror challenge.

Theorem 4. Let L be any language decided by a K-Lipschitz oracle Turing machine M and $W \in [ T ]$ be the sliding-window parameter. For $R \geq 1 9 2 d ^ { 2 } \log ( 1 0 0 T ) , \Delta \geq 1 / ( d \sqrt { W } ) , \tau _ { 1 } = 1 / ( 2 d )$ , and $\tau _ { 2 } =$ $1 / ( 2 d \sqrt { W } )$ , let $\sigma _ { A } ^ { ( 1 ) } \sim ( R , 0 ) , \sigma _ { A } ^ { ( 2 ) } \sim ( R , \Delta )$ , and $\sigma _ { B } \sim ( R , \tau _ { 1 } , \tau _ { 2 } )$ . Then, for any $x \in L$ , on the protocol of Figure 3,

$$
\Phi _ { A } \left( x , \sigma _ { A } ^ { \left( 1 \right) } \circ \sigma _ { B } \right) \geq \frac { 3 } { 5 } \quad a n d \quad \Phi _ { A } \left( x , \sigma _ { A } ^ { \left( 2 \right) } \circ \sigma _ { B } \right) \leq \frac { 1 } { 1 5 0 }
$$

This result shows how a sliding-window check can strengthen oversight against persistent perturbations. By testing discrepancies across a window of W rounds, the cross-examiner can detect shifts at scale $1 / ( d \sqrt { W } )$ , rather than relying solely on discrepancies large enough to be detected in an individual round. The resulting guarantee separates unperturbed reporting, which retains task success probability at least $3 / 5 ,$ from the specified perturbed strategy, whose task success probability is at most 1/150. Increasing W therefore narrows the range of persistent perturbations compatible with successful task performance. This provides a mechanism for curtailing the disclosure strategy studied above.

Stronger oversight requires more work from the verifier to audit a larger window, while leaving the per-round sampling work of A and B unchanged. Thus, this improvement comes with a computational trade-of.

## 6. Related Work

A major strand of oversight research formalises scalability, closely related to the requirement that oversight protocols relativise Irving & Marshall (2025). Theoretical work shows that debate can enable limited verifiers to check complex claims with substantially less efort than solving them independently Brown-Cohen et al. (2024); Irving et al. (2018), with extensions studying capability asymmetries and the human input required for reliable oversight Brown-Cohen et al. (2026); Engels et al. (2025). Prover-estimator debate addresses obfuscated arguments, ensuring under stability assumptions that honest debaters can win with computational resources comparable to those of dishonest opponents Brown-Cohen et al. (2025). These results focus on making correct adjudication feasible, whereas we study latent objectives pursued within task-admissible interactions.

Debate-based alignment safety cases connect success in debate to honesty, subject to additional training and deployment assumptions Buhl et al. (2025). Our analysis highlights a complementary concern, namely that honest, task-admissible behaviour need not preclude undesirable information disclosure through the transcript. Mechanism design for alignment and control studies how to incentivize honesty and obedience when agents’ preferences and capabilities are unknown Bergemann et al. (2026). This perspective ofers a potential route to mitigating the strategic freedom identified by our framework, complementing our analysis of transcript-level objectives.

## Acknowledgments

This project was undertaken in collaboration with the Australian AI Safety Institute (AISI) and supported by funding from the Department of Industry, Science and Resources (DISR), as part of research into AI alignment tools and techniques.

## References

Dirk Bergemann, Andrew Koh, and Stephen Morris. Mechanism design for alignment and control. arXiv preprint arXiv:2609.01595, 2026.

Jonah Brown-Cohen, Geofrey Irving, and Georgios Piliouras. Scalable ai safety via doubly-eficient debate. In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024.

Jonah Brown-Cohen, Geofrey Irving, Georgios Piliouras, Lijie Chen, Jiawei Li, and Zhiyang Xun. Avoiding obfuscation with prover-estimator debate. arXiv preprint arXiv:2506.13609, 2025.

Jonah Brown-Cohen, Geofrey Irving, Simon C Marshall, Ilan Newman, Georgios Piliouras, and Mario Szegedy. Debate is eficient with your time. arXiv preprint arXiv:2602.08630, 2026.

Marie Davidsen Buhl, Jacob Pfau, Benjamin Hilton, and Geofrey Irving. An alignment safety case sketch based on debate. arXiv preprint arXiv:2505.03989, 2025.

Paul Christiano, Buck Shlegeris, and Dario Amodei. Supervising strong learners by amplifying weak experts. arXiv preprint arXiv:1810.08575, 2018.

Joshua Engels, David D. Baek, Subhash Kantamneni, and Max Tegmark. Scaling laws for scalable oversight. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

Jinwei Hu, Xinmiao Huang, Youcheng Sun, Yi Dong, and Xiaowei Huang. Lying with truths: Openchannel multi-agent collusion for belief manipulation via generative montage. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5979–5996, 2026.

Geofrey Irving and Simon Marshall. The need to relativise in debate. AI Alignment Forum post, 2025. URL https://www.alignmentforum.org/posts/XycoFucvxAPhcgJQa/ the-need-to-relativise-in-debate-1.

Geofrey Irving, Paul Christiano, and Dario Amodei. Ai safety via debate. arXiv preprint arXiv:1805.00899, 2018.

Jan Leike, David Krueger, Tom Everitt, Miljan Martic, Vishal Maini, and Shane Legg. Scalable agent alignment via reward modeling: a research direction. arXiv preprint arXiv:1811.07871, 2018.

## A. Toy Example

This example illustrates that, for a probabilistic computation, information may be present not only in whether the machine computes the correct answer, but also in which valid answer it finds. In particular, a probabilistic machine induces a distribution over execution transcripts, and this distribution may depend on properties of the input even when the algorithm is simply solving its prescribed computational task.

Consider the standard unstructured search problem. The input is an array $x \in \{ 0 , 1 \} ^ { N }$ containing at least one marked entry, and the goal is to output an index $i \in [ N ]$ such that $x _ { i } = 1$

Partition the array into two equal-sized branches, labeled 0 and 1. Thus, each branch contains $N / 2$ entries. Let the hidden hypothesis $Q \in \{ 0 , 1 \}$ determine the distribution of marked entries between the two branches:

<table><tr><td></td><td>branch 0 branch 1</td><td></td></tr><tr><td>Q= 0</td><td>3m</td><td>m</td></tr><tr><td>Q = 1</td><td>m</td><td>3m</td></tr></table>

where $\imath \geq 1$ and $3 m \le N / 2$ . Under either hypothesis, the total number of marked entries is 4m.

Now consider the following fixed probabilistic machine M. The machine samples an index $i \in [ N ]$ uniformly at random, reads $x _ { i } ,$ and repeats if $x _ { i } = 0 .$ . If $x _ { i } = 1$ , it outputs i and halts. This is a Las Vegas algorithm for unstructured search. The algorithm always outputs a correct index and halts with probability one. Moreover, since there are 4m marked entries under both hypotheses, the success probability on each probe is $\frac { 4 m } { N }$ . Therefore the number of probes K before halting has the same distribution under $Q = 0$ and $Q = 1$ , namely, for $k \geq 1$

$$
\operatorname* { P r } [ K = k \mid Q ] = \left( 1 - { \frac { 4 m } { N } } \right) ^ { k - 1 } { \frac { 4 m } { N } } ,
$$

for both values of Q. In particular,

$$
\mathbb { E } [ K \mid Q = 0 ] = \mathbb { E } [ K \mid Q = 1 ] = { \frac { N } { 4 m } } .
$$

Nevertheless, the returned index carries information about Q. Let $X \in \{ 0 , 1 \}$ denote the branch containing the returned index. Since the machine samples uniformly until it finds a marked entry, the returned index is uniformly distributed over the marked entries. Hence

$$
\operatorname* { P r } [ X = 1 \mid Q = 0 ] = \frac { m } { 3 m + m } = \frac { 1 } { 4 } ,
$$

whereas

$$
\operatorname* { P r } [ X = 1 \mid Q = 1 ] = { \frac { 3 m } { m + 3 m } } = { \frac { 3 } { 4 } } .
$$

Thus, the algorithm does not append any additional message for disclosure. It simply returns a solution to the search problem, and the valid solution it finds reveals information about the input.

If A denotes the execution transcript, then the transcript includes the returned index and therefore determines $X .$ . Consequently, $I ( Q ; A ) \geq I ( Q ; X ) > 0$

In addition, a stronger phenomenon occurs when the machine has discretion over its sampling policy. By perturbing the distribution used to select the next probe, while leaving the stopping rule and correctness condition unchanged, the machine can create a communication channel through the machine computation.

For example, instead of choosing every probe uniformly across the array, the machine M proceeds as follows:

1. Select branch 1 with probability $p$ and branch 0 with probability $1 - p$

2. Select an index uniformly within the chosen branch.

3. Read its value.

4. If the selected index is marked, output its index; otherwise repeat.

Standard Las Vegas corresponds to $\begin{array} { r } { p = \frac { 1 } { 2 } } \end{array}$ . Under the perturbation $\begin{array} { r } { p = \frac { 1 } { 2 } + \delta , | \delta | < \frac { 1 } { 2 } } \end{array}$ , every returned answer remains correct. Since every index still has positive probability of being sampled, the algorithm still terminates almost surely on every promised input. Crucially, the perturbation afects which input location the machine actually reads next. The communication takes place through the machine’s computation transcript.

In conclusion, the perturbation can increase information communicated by making the machine’s sampling choices depend on a hidden bit $Q ;$ it slightly favors branch 1 when $Q = 1$ and branch 0 when $Q = 0$ . The branch of the returned witness becomes correlated with $Q ,$ allowing an observer to infer the hidden bit with better-than-chance accuracy. Thus, relative to an unbiased policy independent of $Q ,$ , the machine communicates additional information entirely through its legitimate computational choices, without appending a message or sacrificing task performance.

## B. Task Objective Proofs

## B.1. Bounding the Task Objective

We now demonstrate how to bound the task objective. This is achieved through a sequence of steps that focus on diferent aspects of task performance. For each round $t ,$ define the good-estimation event

$$
\mathcal { E } _ { t } = \left\{ \left| \bar { p } _ { t } - p _ { t } \right| < \frac { 1 } { 1 6 d } \right\} \cap \left\{ \left| \hat { q } _ { t } - p _ { t } \right| < \frac { 1 } { 8 d } \right\} ,
$$

where $\bar { p } _ { t }$ is the sample estimate obtained by A prior to adding the perturbation. Let N denote the protocol stopping time. Define the uniform good-estimation event

$$
\mathcal { E } = \bigcap _ { t = 1 } ^ { N } \mathcal { E } _ { t } \cap \{ | \hat { p } _ { N } ^ { \mathcal { O } } - p _ { N } | < \frac { 1 } { 8 d } \}
$$

and the no-abort event by $\kappa = \left\{ \begin{array} { r l r l } \end{array} \right.$ {the protocol does not abort}. So that $\mathcal { E }$ is well-defined, we set $\hat { p } _ { T } ^ { \mathcal { O } } = p _ { T }$ if no abort occurs. The good-estimation event states allows us to analyse the protocol under well behaved sample estimates. We first focus on the good-estimation event under small perturbations $\Delta$ which implies that the debate does not abort. That is, if $\hat { q } _ { t }$ and $\hat { p } _ { t } ^ { \mathcal { O } }$ are “good” estimates, as defined above, then an abort under honest strategy $\sigma _ { B } ^ { \mathsf { h o n } }$ implies

$$
\left| \hat { p } _ { t } - \hat { q } _ { t } \right| \geq \frac { 1 } { 2 d } \implies \left| \hat { p } _ { t } - p _ { t } \right| \geq \frac { 3 } { 8 d } \implies \left| \hat { p } _ { t } ^ { \mathcal { O } } - \hat { p } _ { t } \right| \geq \frac { 1 } { 4 d } ,\tag{2}
$$

where the last statement contradicts E for perturbation $\Delta \leq 1 / ( 1 6 d )$ . Therefore, an abort is not possible under event E. To bound the task objective on small perturbations, we decompose $V ^ { \ast } s$ acceptance probability as follows.

$$
\operatorname* { P r } [ V \to 1 \mid \Delta ] = \operatorname* { P r } [ V \to 1 \mid \mathcal { E } , \Delta ] \operatorname* { P r } [ \mathcal { E } \mid \Delta ] + \operatorname* { P r } [ V \to 1 \mid \neg \mathcal { E } , \Delta ] \operatorname* { P r } [ \neg \mathcal { E } \mid \Delta ]
$$

Thus, the task objective satisfies,

$$
\operatorname* { P r } [ V \to 1 \mid \mathcal { E } , \Delta ] \operatorname* { P r } [ \mathcal { E } ] \leq \operatorname* { P r } [ V \to 1 , \Delta ] \leq \operatorname* { P r } [ V \to 1 \mid \mathcal { E } , \Delta ] + \operatorname* { P r } [ \neg \mathcal { E } ] ,\tag{3}
$$

where, for perturbation size $\Delta < 1 / ( 1 6 d ) , \{ \mathcal { E } , \Delta \} \subseteq \mathcal { K }$

For larger perturbations, we focus on the outcome of an abort under good estimation. By (2), an abort under good estimation implies that $\hat { p } _ { t }$ exceeds the verifier’s rejection threshold (which can occur for large enough ∆). Thus, acceptance is impossible on $\mathcal E \cap \neg \mathcal K .$ , and hence

$$
\operatorname* { P r } [ V \to 1 , { \mathcal { E } } , \neg { \mathcal { K } } ] = 0 .
$$

We conclude that acceptance can occur only on $\mathcal { E } \cap \mathcal { K }$ or on the failure event $\neg { \mathcal { E } } .$ . Therefore,

$$
\begin{array} { r l } & { \operatorname* { P r } [ V \to 1 ] = \operatorname* { P r } [ V \to 1 , { \mathcal E } , { \mathcal K } ] + \operatorname* { P r } [ V \to 1 , { \mathcal E } , \neg { \mathcal K } ] + \operatorname* { P r } [ V \to 1 , \neg { \mathcal E } ] } \\ & { \qquad \le \operatorname* { P r } [ { \mathcal E } \cap { \mathcal K } ] + \operatorname* { P r } [ \neg { \mathcal E } ] . } \end{array}
$$

If $\operatorname* { P r } [ { \mathcal { E } } ] > 0$ , this can equivalently be written as

$$
\begin{array} { r l } & { \operatorname* { P r } [ V \to 1 ] \leq \operatorname* { P r } [ \mathcal { E } \cap \mathcal { K } ] + \operatorname* { P r } [ \neg \mathcal { E } ] } \\ & { \qquad = \operatorname* { P r } [ \mathcal { E } ] \operatorname* { P r } [ \mathcal { K } \mid \mathcal { E } ] + \operatorname* { P r } [ \neg \mathcal { E } ] } \\ & { \qquad = \operatorname* { P r } [ \mathcal { E } ] \operatorname* { P r } [ \mathcal { K } \mid \mathcal { E } ] + \operatorname* { P r } [ \neg \mathcal { E } ] } \\ & { \qquad \leq \operatorname* { P r } [ \mathcal { K } \mid \mathcal { E } ] + \operatorname* { P r } [ \neg \mathcal { E } ] . } \end{array}\tag{4}
$$

The following subsections will bound each term in (3) and (4) to provide a characterization of the expected task objective as ∆ grows. We begin with the good-estimation event.

## B.2. Likelihood of good-estimation

We now provide a bound for the good-estimation event. We require the following version of the Chernof bound.

Lemma 2. Let $X _ { 1 } , \dots , X _ { R }$ be independent Bernoulli random variables, each taking value 1 with probability $p$ and 0 with probability $( 1 - p )$ . Let $\begin{array} { r } { \bar { \mu } = \frac { 1 } { R } \sum _ { i = 1 } ^ { R } X _ { i } } \end{array}$ be the empirical mean of the random variables. Then

$$
\mathrm { P r } \left[ \left| \bar { \mu } - p \right| \geq s < 2 \exp \left( \frac { - s ^ { 2 } R } { 3 } \right) \right]
$$

Lemma 3. Suppose that $R \geq 1 9 2 d ^ { 2 } \log ( 1 0 0 T )$ . Then every strategy $\sigma = \sigma _ { A } \circ \sigma _ { B }$ parameterized by $\sigma _ { A } \sim ( R , \bigsqcup )$ and $\sigma _ { B } \sim ( R , \bigsqcup )$ satisfies

$$
\operatorname* { P r } [ \lnot \mathcal { E } ] \leq \frac { 1 } { 3 0 0 }
$$

Proof. If the estimates $\bar { p } _ { t } , \hat { g } _ { t }$ , and $\hat { p } _ { t } ^ { \mathcal { O } }$ are the means of R independent Bernoulli samples with probability $p _ { t . }$ , Lemma 2 and a union bound gives

$$
\operatorname* { P r } \left[ \left| { \bar { p } } _ { t } - p _ { t } \right| \geq { \frac { 1 } { 1 6 d } } \wedge \left| { \hat { q } } _ { t } - p _ { t } \right| \geq { \frac { 1 } { 8 d } } \right] \leq 2 \exp \left( - { \frac { R } { 1 9 2 d ^ { 2 } } } \right) + 2 \exp \left( - { \frac { R } { 9 6 d ^ { 2 } } } \right) .
$$

Therefore, applying a union bound over all T rounds yields

$$
\operatorname* { P r } [ \lnot \mathcal { E } ] \le 4 T \exp \left( - \frac { R } { 1 9 2 d ^ { 2 } } \right) \le \frac { T } { 6 0 0 T } = \frac { 1 } { 6 0 0 } ,
$$

by the choice of R. Lastly, for V with $r = 1 9 2 d ^ { 2 } \log 1 0 0$ samples,

$$
\operatorname* { P r } \left[ \left| \hat { p } _ { t } ^ { \mathcal { O } } - p _ { t } \right| \ge \frac { 1 } { 8 d } \right] \le 2 \exp \left( - \frac { r } { 4 8 d ^ { 2 } } \right) \le \frac { 4 } { 1 0 0 ^ { 2 } } \le \frac { 1 } { 6 0 0 }
$$

Thus, combining, we observe

$$
\begin{array} { r l } & { \operatorname* { P r } [ \neg \mathcal { E } ] \leq \operatorname* { P r } \left[ | \bar { p } _ { t } - p _ { t } | \geq \frac { 1 } { 4 d } \land | \hat { q } _ { t } - p _ { t } | \geq \frac { 1 } { 4 d } \right] + \operatorname* { P r } \left[ | \hat { p } _ { t } ^ { \mathcal { O } } - p _ { t } | \geq \frac { 1 } { 4 d } \right] } \\ & { \qquad \leq \displaystyle \frac { 1 } { 6 0 0 } + \frac { 1 } { 6 0 0 } = \frac { 1 } { 3 0 0 } . } \end{array}
$$

## B.3. Stability of Task Success

We now establish bounds on the task objective under the good estimation event. We record a direct consequence of the Lipschitz property of the oracle machine.

Fact 1. Let M be K-Lipschitz oracle machine and denote $\operatorname* { P r } [ M ^ { O } ( x ) = 1 ] = \alpha ^ { \star }$ . Suppose that any nonaborting simulation of M which, at every history h, replaces the reference transition probability $p ( \mathbf { h } )$ by a possibly randomized, history-dependent probability $\tilde { p } ( \mathbf { h } )$ satisfying

$$
| \tilde { p } ( { \bf h } ) - p ( { \bf h } ) | \leq \frac { 1 } { d }
$$

has acceptance probability within $K / d$ of $\alpha ^ { \star }$

Lemma 4. Let M be K-Lipschitz oracle machine and denote $\operatorname* { P r } [ M ^ { O } ( x ) = 1 ] = \alpha ^ { \star }$ . Fix a strategy $\sigma _ { A }$ with perturbation magnitude $\Delta \leq { \frac { 3 } { 4 d } }$ . Then, on the protocol of Figure $^ { 2 , }$

$$
\mathrm { P r } [ V \to 1 ] \leq \alpha ^ { \star } + \frac { L } { d } + \frac { 1 } { 3 0 0 } .
$$

$H ,$ in addition, for value $\delta \geq 0 , \{ \mathcal { E } , \Delta = \delta \}$ implies that the protocol does not abort, then

$$
\vert \mathrm { P r } [ V \to 1 \mid \mathcal { E } , \Delta = \delta ] - \alpha ^ { \star } \vert \leq \frac { L } { d } + \frac { 1 } { 3 0 0 } .
$$

Proof. Construct a simulation S of the protocol of Figure 2 that follows the reporting strategy $\sigma _ { A }$ but ignores all abort instructions and continues to the terminal machine output. Extend $\sigma _ { A }$ to $s$ on histories that would occur only after an abort.

At each round, let $\hat { p } _ { t }$ be the report proposed by this simulation. Instead of using $\hat { p } _ { t }$ directly, use its projection $\tilde { p } _ { t }$ onto the interval

$$
\left[ \operatorname* { m a x } \left\{ 0 , p _ { t } - \frac { 1 } { d } \right\} , \operatorname* { m i n } \left\{ 1 , p _ { t } + \frac { 1 } { d } \right\} \right] .
$$

By construction,

$$
| \tilde { p } _ { t } - p _ { t } | \leq \frac 1 d
$$

at every history, including histories on which estimation fails. Consequently, if $\tilde { X }$ denotes the terminal output of $s ,$ Fact 1 gives

$$
\left| \operatorname* { P r } [ \tilde { X } = 1 ] - \alpha ^ { \star } \right| \leq \frac { K } { d } .\tag{5}
$$

Couple $\boldsymbol { \mathcal { S } }$ and the actual protocol using the same random samples and transition randomness for as long as their executions coincide. On $\mathcal { E } _ { \ast }$ , the perturbation bound implies

$$
\vert \hat { p } _ { t } - p _ { t } \vert \leq \vert \hat { p } _ { t } - \bar { p } _ { t } \vert + \vert \bar { p } _ { t } - p _ { t } \vert < \frac { 3 } { 4 d } + \frac { 1 } { 4 d } = \frac { 1 } { d } .
$$

Thus, the projection does not change any report along the actual execution on $\mathcal { E } .$ . The two simulations therefore coincide until the protocol aborts or terminates.

On $\mathcal { E } ,$ if the protocol accepts, it cannot have aborted, since, by (2), every abort on $\mathcal { E }$ results in rejection. Its entire computation consequently agrees with the auxiliary simulation, which also outputs 1. Hence,

$$
\{ V \to 1 \} \cap { \mathcal { E } } \subseteq \{ \tilde { X } = 1 \} .
$$

Therefore,

$$
\begin{array} { r l } {  { \operatorname* { P r } [ V \to 1 ] \le \operatorname* { P r } [ V \to 1 , \mathcal { E } ] + \operatorname* { P r } [ \neg \mathcal { E } ] } } \\ & { \le \operatorname* { P r } [ \tilde { X } = 1 ] + \frac { 1 } { 3 0 0 } } \\ & { \le \alpha ^ { \star } + \frac { K } { d } + \frac { 1 } { 3 0 0 } . } \end{array}
$$

If $\{ \mathcal { E } , \Delta \}$ also implies survival, then the two outputs agree throughout $\mathcal { E } .$ . Then, for $\delta \geq 0$ , where $\{ \mathcal { E } , \Delta = \delta \} \subseteq { \mathcal { K } } ,$

$$
\operatorname* { P r } [ V \to 1 \mid \mathcal { E } , \Delta = \delta ] = \operatorname* { P r } [ \tilde { X } = 1 \mid \mathcal { E } , \Delta = \delta ] ,
$$

and hence

$$
\left| \operatorname* { P r } [ V \to 1 \mid \mathcal { E } , \Delta = \delta ] - \operatorname* { P r } [ \tilde { X } = 1 ] \right| \leq \operatorname* { P r } [ \neg \mathcal { E } ]
$$

Combining this inequality with (5) gives

$$
\vert \mathrm { P r } [ V \to 1 \mid \mathcal { E } , \Delta = \delta ] - \alpha ^ { \star } \vert \leq \frac { L } { d } + \frac { 1 } { 3 0 0 } ,
$$

as claimed.

□

## B.4. Non-abort survival

We move to bounding the task objective for general perturbations. By (4), it sufices to provide an upper bound on $\mathrm { P r } [ \mathcal { K } \mid \mathcal { E } ]$ . We will require the following result. Let $W _ { 1 } , \ldots , W _ { n }$ be independent, mean-zero random variables with finite third absolute moments and $F _ { \mathsf { s t d } }$ denote the cumulative distribution function of the Normal distribution. Writing

$$
S _ { n } = \sum _ { i = 1 } ^ { n } W _ { i } , \qquad \sigma ^ { 2 } = \sum _ { i = 1 } ^ { n } \mathbb { E } [ W _ { i } ^ { 2 } ] > 0 , \qquad \rho = \frac { \sum _ { i = 1 } ^ { n } \mathbb { E } | W _ { i } | ^ { 3 } } { \sigma ^ { 3 } } ,
$$

the Berry-Esseen theorem states the following.

Lemma 5. For constant $C _ { \mathsf { B E } } = 0 . 5 6 ,$

$$
\operatorname* { s u p } _ { x \in \mathbb { R } } \left| \operatorname* { P r } [ S _ { n } \leq x ] - F _ { \mathsf { s t d } } \left( { \frac { x } { \sigma } } \right) \right| \leq C _ { \mathsf { B E } } \cdot \rho .
$$

We now establish some useful notation. Fix a perturbation magnitude $\Delta \geq 0$ . All probabilities in this section refer to this fixed perturbation. Recall that the report at round t is

$$
\hat { p } _ { t } = \mathsf { c l i p } ( \bar { p } _ { t } + Q \Delta ) ,
$$

and the cross-examiner aborts when

$$
\left| \hat { p } _ { t } - \hat { q } _ { t } \right| \geq \tau = \frac { 1 } { 2 d } .
$$

Let $\kappa _ { t }$ denote the event that no cross-examination abort occurs during the first t rounds, with $\begin{array} { r } { K _ { 0 } = \Omega } \end{array}$ and write $\kappa = \kappa _ { T }$ . Define the estimation-noise diference $Z _ { t } = \bar { p } _ { t } - \hat { q } _ { t }$ . In the absence of clipping, the discrepancy equals $Q \Delta + Z _ { t }$ . With clipping, however, this identity need not hold. Throughout, assume that $\operatorname* { P r } [ { \mathcal { E } } ] > 0$

## B.4.1. One-Round Gaussian Approximation

We first condition only on the past, rather than on the global event E. Throughout we assume that $p _ { t } \in ( 0 . 0 1 , 0 . 9 9 )$ . Under this assumption, the event $\mathcal { E } _ { \ast }$ and $\Delta < 1 / d ,$ no clipping occurs under the choice of $d = \lceil 1 5 0 K \rceil$ and $K \geq 1$ . For any surviving history h, define

$$
r _ { t } ( \Delta , h ) = \mathrm { P r } [ | \hat { p } _ { t } - \hat { q } _ { t } | < \tau | \mathcal { K } _ { t - 1 } , \mathbf { h } _ { t - 1 } = h ] .
$$

We assume that, conditional on this history and either value of Q having positive conditional probability, the fresh samples

$$
X _ { t , 1 } , \ldots , X _ { t , R } , Y _ { t , 1 } , \ldots , Y _ { t , R }
$$

are mutually independent Bernoulli random variables with common mean $p _ { t } = p _ { t } ( h )$ , and that

$$
\bar { p } _ { t } = \frac { 1 } { R } \sum _ { i = 1 } ^ { R } X _ { t , i } , \qquad \hat { q } _ { t } = \frac { 1 } { R } \sum _ { i = 1 } ^ { R } Y _ { t , i } .
$$

In particular, at a fixed history, this fresh-sample law is the same for either value of Q. Define the conditional probability of survival as

$$
r _ { t } ^ { 0 } ( \Delta , h ) = \mathrm { P r } [ | Q \Delta + Z _ { t } | < \tau | \mathcal { K } _ { t - 1 } , \mathbf { h } _ { t - 1 } = h ] .
$$

For $p _ { t } \in ( 0 . 0 1 , 0 . 9 9 )$ , define $\sigma _ { t } = \sqrt { 2 p _ { t } \big ( 1 - p _ { t } \big ) / R }$ and

$$
g _ { t } ( \Delta ) = \mathrm { P r } [ | \Delta + \sigma _ { t } z | < \tau ] ,
$$

where $z \sim \mathsf { N } ( 0 , 1 )$

Lemma 6. Assume $p _ { t } \in ( 0 . 0 1 , 0 . 9 9 )$ and that no clipping occurs for all configurations of machine input $M ,$ then

$$
\left| r _ { t } ^ { 0 } ( \Delta , h ) - g _ { t } ( \Delta ) \right| \leq 1 / 3 0 0 .
$$

Proof. Fix a surviving history h and a value of Q with positive conditional probability. Under this conditioning,

$$
Z _ { t } = \frac { 1 } { R } \sum _ { i = 1 } ^ { R } ( X _ { t , i } - p _ { t } ) - \frac { 1 } { R } \sum _ { i = 1 } ^ { R } ( Y _ { t , i } - p _ { t } ) .
$$

The summands are independent and centered, and

$$
\operatorname { V a r } ( Z _ { t } ) = { \frac { 2 p _ { t } ( 1 - p _ { t } ) } { R } } = \sigma _ { t } ^ { 2 } .
$$

For a Bernoulli random variable X with mean $p _ { t }$

$$
\begin{array} { r l } & { \mathbb { E } [ | X - p _ { t } | ^ { 3 } ] = p _ { t } ( 1 - p _ { t } ) ^ { 3 } + ( 1 - p _ { t } ) p _ { t } ^ { 3 } } \\ & { \qquad = p _ { t } ( 1 - p _ { t } ) \big [ p _ { t } ^ { 2 } + ( 1 - p _ { t } ) ^ { 2 } \big ] . } \end{array}
$$

Therefore, the standardized sum of third absolute moments is

$$
\begin{array} { c } { { \rho _ { t , R } = \displaystyle \frac { 2 R \cdot R ^ { - 3 } \cdot p _ { t } ( 1 - p _ { t } ) \big [ p _ { t } ^ { 2 } + ( 1 - p _ { t } ) ^ { 2 } \big ] } { \sigma _ { t } ^ { 3 } } } } \\ { { = \displaystyle \frac { 1 } { \sqrt { 2 R } } \frac { p _ { t } ^ { 2 } + ( 1 - p _ { t } ) ^ { 2 } } { \sqrt { p _ { t } ( 1 - p _ { t } ) } } . } } \end{array}
$$

Lemma 5 consequently gives, uniformly over x,

$$
\left| \operatorname* { P r } [ Z _ { t } \leq x | K _ { t - 1 } , \mathbf { h } _ { t - 1 } = h , Q ] - F _ { \mathrm { s t d } } \left( { \frac { x } { \sigma _ { t } } } \right) \right| \leq { \frac { C _ { \mathsf { B E } } } { \sqrt { 2 R } } } { \frac { p _ { t } ^ { 2 } + ( 1 - p _ { t } ) ^ { 2 } } { \sqrt { p _ { t } ( 1 - p _ { t } ) } } } \leq { \frac { 1 } { 6 0 0 } } ,
$$

by the choice of R and assumption on $p _ { t }$ .

For $Q = + 1$ , the unclipped survival event is

$$
- \tau - \Delta < Z _ { t } < \tau - \Delta .
$$

Applying the distribution-function bound at the interval endpoints yields an error of at most 2/600 relative to $g _ { t } ( \Delta )$ . The conditional law of $Z _ { t }$ is symmetric about zero, since it is the diference of two independent, identically distributed sample means. Thus the same interval probability and bound hold for $Q = - 1$ . Averaging over the conditional distribution of $Q$ proves the result. □

Now we can provide the final conditional survival bound.

Lemma 7. Fix a perturbation ∆ and a round $t \in \{ 1 , \ldots , T \}$ . Then

$$
\operatorname* { P r } [ \mathcal { K } \mid \mathcal { E } ] \leq \operatorname* { m i n } \left\{ 1 , \mathbb { E } \left[ g _ { t } ( \Delta ) \mid \mathcal { K } _ { t - 1 } \right] + \frac { 1 } { 1 0 0 } \right\} .
$$

Proof. By Lemma 6, for every surviving history h, the one-round survival probability satisfies

$$
\mathrm { P r } [ \mathcal { K } _ { t } | \mathcal { K } _ { t - 1 } ] \leq g _ { t } ( \Delta ) + \frac { 1 } { 3 0 0 } .
$$

Define $G _ { t } ( \Delta ) = \mathbb { E } [ g _ { t } ( \Delta ) \vert \mathcal { K } _ { t - 1 } ]$ . Averaging the one-round bound over histories conditional on $\boldsymbol { \mathcal { K } } _ { t - 1 }$ gives

$$
\begin{array} { r l } {  { \operatorname* { P r } [ \mathcal { K } _ { t } \ | \ \mathcal { K } _ { t - 1 } ] = \mathbb { E } [ \operatorname* { P r } [ \mathcal { K } _ { t } \ | \ \mathcal { K } _ { t - 1 } , \mathbf { h } _ { t - 1 } ] \ | \ \mathcal { K } _ { t - 1 } ] } } \\ & { \leq \mathbb { E } \bigg [ g _ { t } ( \Delta ) + \frac { 1 } { 3 0 0 } \bigg | \ \mathcal { K } _ { t - 1 } \bigg ] } \\ & { = G _ { t } ( \Delta ) + \frac { 1 } { 3 0 0 } . } \end{array}
$$

Since $\boldsymbol { \mathcal { K } } \subseteq \boldsymbol { \mathcal { K } } _ { t } \subseteq \boldsymbol { \mathcal { K } } _ { t - 1 }$ , it follows that

$$
\begin{array} { r l } & { \operatorname* { P r } [ \mathcal { K } ] \leq \operatorname* { P r } [ \mathcal { K } _ { t } ] } \\ & { \qquad = \operatorname* { P r } [ \mathcal { K } _ { t - 1 } ] \operatorname* { P r } [ \mathcal { K } _ { t } \mid \mathcal { K } _ { t - 1 } ] } \\ & { \qquad \leq G _ { t } ( \Delta ) + \displaystyle \frac { 1 } { 3 0 0 } . } \end{array}
$$

The law of total probability yields

$$
\begin{array} { r l } & { \operatorname* { P r } [ \mathcal { K } \mid \mathcal { E } ] - \operatorname* { P r } [ \mathcal { K } ] = \operatorname* { P r } [ \neg \mathcal { E } ] \left( \operatorname* { P r } [ \mathcal { K } \mid \mathcal { E } ] - \operatorname* { P r } [ \mathcal { K } \mid \neg \mathcal { E } ] \right) } \\ & { \qquad \leq \operatorname* { P r } [ \neg \mathcal { E } ] } \\ & { \qquad \leq \displaystyle \frac { 1 } { 1 5 0 } , } \end{array}
$$

By Lemma 3. Consequently,

$$
\operatorname* { P r } [ \mathcal { K } \mid \mathcal { E } ] \leq G _ { t } ( \Delta ) + \frac { 1 } { 3 0 0 } + \frac { 1 } { 1 5 0 } = G _ { t } ( \Delta ) + \frac { 1 } { 1 0 0 } .
$$

Combining this with the trivial bound $\mathrm { P r } [ \mathcal { K } | \mathcal { E } ] \leq 1$ proves the claim.

## B.5. Proof Statements

Theorem 1 follows from (3) with the application of Lemmas 3 and 4 and the observations that $K / d \le 1 / 1 5 0$ and $\left\{ \mathcal { E } , \Delta < 1 / ( 1 6 d ) \right\} \subseteq K$

Theorem 2 follows from (4) and the application of Lemmas 3 and 7 for $t = 1$

## C. Proof of Theorem 3

We begin with setting notation. Fix $X = x$ and let $\mathbf { y } = ( \mathsf { y } _ { 1 } , \hdots , \mathsf { y } _ { T } )$ be the random variable representing the transcript of $M ( x )$ and $\pmb { a } = ( a _ { 1 } , \ldots , a _ { T } )$ denote the realized transcript. Unless stated otherwise, all

probabilities, expectations, and information quantities are conditional on $X = x$ . Write $h _ { x } = H ( Q \mid$ $X = x )$ . The view of V is defined as

$$
\Pi _ { V } ( \mathbf { h } ( X , \sigma ) ) = \left\{ \begin{array} { l l } { ( a _ { 1 } , \dotsc , a _ { T } ) , } & { \mathrm { ~ o n ~ e v e n t ~ } \mathcal { K } , } \\ { ( a _ { 1 } , \dotsc , a _ { t } , \hat { p } _ { t } , \{ \mathcal { O } ( a _ { I ( t ) } ) \} ) , } & { \mathrm { ~ o n ~ e v e n t ~ } \lnot \subset \mathcal { K } _ { t } \mid \mathcal { K } _ { t - 1 } . } \end{array} \right.
$$

These transcript types are distinguishable, so the abort indicator and, upon abort, its round are recoverable from $\Pi _ { V }$ . Notably, V does not observe the internal samples of A or B directly. For some $0 < \beta < 1 / 2$ we assume that $\operatorname* { P r } [ Q = 1 \mid X = x ] \in [ \beta , 1 - \beta ]$ . Thus, there is initial uncertainty in the hidden hypothesis.

## C.1. Belief updating and confidence truncation

Define the hidden hypothesis posterior as

$$
\mu _ { t } = \operatorname* { P r } ( Q = 1 \mid \mathbf { h } _ { t - 1 } , K _ { t - 1 } ) .
$$

These are exact Bayesian posteriors for the observed non-abort channel. At each entering history $\mathbf { h } _ { t - 1 }$ the expected posterior divergence satisfies the identity

$$
\mathbb { E } [ D _ { \mathsf { K L } } \big ( \mathsf { B e r n } ( \mu _ { t } ) \| \mathsf { B e r n } ( \mu _ { t - 1 } ) \big ) \big | { \mathsf { h } } _ { t - 1 } , K _ { t - 1 } \big ] = I ( Q ; \mathsf { y } _ { t } \mid { \mathsf { h } } _ { t - 1 } , K _ { t - 1 } ) .\tag{6}
$$

Thus, the expected information gained in an update cannot exceed the remaining uncertainty. More generally, any valid upper bound on the channel’s mutual information also bounds the expected posterior divergence in (6). This identity follows from Bayes’ rule and is consistent with sequential hypothesis testing, where posterior log-odds accumulate observation log-likelihood ratios.

In our latent disclosure accounting, expected KL increments accumulate according to the mutual information chain rule and their total is bounded by the initial entropy $h _ { x }$ . We assume that all information used in updating is included in the observed history.

## C.2. Single round disclosure

We begin by bounding the amount of information disclosed in a single round of debate. We look at the instances of abort and non-abort separately. For probability laws $P$ and Q, with divergence measured in bits, Pinsker’s inequality states that

$$
D _ { \mathsf { K L } } ( P \| Q ) \geq \frac { 2 } { \ln 2 } \| P - Q \| _ { \mathsf { T V } } ^ { 2 } .\tag{7}
$$

Lemma 8. For $\mu _ { t } \in ( \beta , 1 - \beta )$

$$
I ( Q ; { \mathbf y } _ { t } \mid { \mathbf h } _ { t - 1 } , K _ { t - 1 } ) \geq \frac { 2 } { \ln { 2 } } \mu _ { t } ( 1 - \mu _ { t } ) \Delta ^ { 2 } \geq \frac { 2 \beta ( 1 - \beta ) } { \ln { 2 } } \Delta ^ { 2 } .
$$

Proof. Define the hypothesis conditional sampling probabilities as

$$
p _ { \pm } = \operatorname { P r } [ \mathbf { y } _ { t } \mid Q \pm 1 , \mathbf { h } _ { t - 1 } , \boldsymbol { K } _ { t - 1 } ]
$$

and the mixture $p = \mu _ { t } p _ { + } + ( 1 - \mu _ { t } ) p _ { - }$ . Since a mixture of Bernoulli laws is Bernoulli, the conditional mutual information is

$$
I ( Q ; { \bf y } _ { t } \mid { \bf h } _ { t - 1 } , K _ { t - 1 } ) = \mu _ { t } { \cal D } _ { \mathsf { K L } } \left( { \mathsf { B e r n } ( p _ { + } ) } \| { \mathsf { B e r n } ( \bar { p } ) } \right) + ( 1 - \mu _ { t } ) { \cal D } _ { \mathsf { K L } } \left( { \mathsf { B e r n } ( p _ { - } ) } \| { \mathsf { B e r n } ( \bar { p } ) } \right) .
$$

For Bernoulli laws, total variation distance equals the absolute diference of their parameters. Thus, (7) gives

$$
\begin{array} { l } { { { \cal I } ( Q ; { \bf y } _ { t } \mid { \bf h } _ { t - 1 } , K _ { t - 1 } ) \geq \displaystyle \frac { 2 } { \ln 2 } \left[ \mu _ { t } ( p _ { + } - \bar { p } ) ^ { 2 } + ( 1 - \mu _ { t } ) ( p _ { - } - \bar { p } ) ^ { 2 } \right] } } \\ { { { } } } \\ { { { } = \displaystyle \frac { 2 } { \ln 2 } \mu _ { t } ( 1 - \mu _ { t } ) ( p _ { + } - p _ { - } ) ^ { 2 } . } } \end{array}\tag{8}
$$

Let <sub>ν</sub> denote the distribution of $\bar { p } _ { t }$ . As the distribution of $\bar { p } _ { t }$ is independent of $Q ,$

$$
p _ { + } - p _ { - } = \int _ { [ 0 , 1 ] } \left[ { \mathsf { c l i p } } ( s + \Delta ) - { \mathsf { c l i p } } ( s - \Delta ) \right] , \nu ( d s ) .
$$

For every $s \in [ 0 , 1 ]$ and $\begin{array} { r } { 0 < \Delta \le \frac { 1 } { 2 } } \end{array}$

$$
\mathsf { c l i p } ( s + \Delta ) - \mathsf { c l i p } ( s - \Delta ) \geq \Delta .
$$

Hence $p _ { + } - p _ { - } \ge \Delta$ . Substitution into (8), followed by $\mu ( 1 - \mu ) \geq \beta ( 1 - \beta )$ , proves the stated result. Averaging over the entering histories gives the conditional mutual-information bound. □

Disclosure during an abort is given by the following.

Lemma 9. Suppose that $2 R \Delta \notin \mathbb { Z } .$ . Then, the report $\hat { p } _ { t }$ fully determines $Q .$ In other words,

$$
H ( Q \mid { \hat { p } } _ { t } ) = 0 .
$$

Proof. Recall that the public report upon abort is

$$
\hat { p } _ { t } = \frac { K _ { t } } { R } + Q \Delta , \qquad K _ { t } \in \{ 0 , \ldots , R \} ,
$$

Under the two hypotheses, the report belongs respectively to the grids

$$
\mathcal { G } _ { + } = \left\{ \frac { k } { R } + \Delta : 0 \leq k \leq R \right\} , \qquad \mathcal { G } _ { - } = \left\{ \frac { k } { R } - \Delta : 0 \leq k \leq R \right\} .
$$

If these grids intersect, there exist integers $k , \ell$ such that

$$
\frac { k } { R } + \Delta = \frac { \ell } { R } - \Delta ,
$$

which implies

$$
2 R \Delta = \ell - k \in \mathbb { Z } .
$$

The assumed nonintegrality therefore makes the grids disjoint. Membership in the observed grid determines Q. It follows that $H ( Q \mid \hat { p } _ { t } ) = 0$ □

## C.3. Full debate disclosure

We are now in a position to proceed with the main proof of information disclosure. Let $h _ { 2 }$ denote the binary entropy function.

Theorem 5. For $\beta \in \left( 0 , \mathbf { 1 } / 2 \right)$ and $\mu _ { 0 } \in ( \beta , 1 - \beta )$ , let $\begin{array} { r } { k _ { \beta } = \frac { 2 \beta \left( 1 - \beta \right) } { \ln 2 } } \end{array}$ . Then

$$
I ( Q ; \Pi _ { V } ( \mathbf { h } ) \mid X = x ) \ge \frac { h _ { x } - h _ { 2 } ( \beta ) } { 2 } \operatorname* { m i n } \left\{ 1 , \frac { k _ { \beta } \Delta ^ { 2 } T } { 1 - h _ { 2 } ( \beta ) } \right\} .
$$

Proof. Throughout the proof, condition on $X = x$ and measure entropy in bits. Stop observing the protocol at the first of the following events:

1. an abort;

2. the posterior leaves $( \beta , 1 - \beta )$ ;

3. completion of round T.

Denote the resulting stopped transcript by $\mathbf { u } \subseteq \mathbf { y } .$ . On history h, since u is determined by $\Pi _ { V } ( { \bf h } )$ the data-processing inequality gives $I ( Q ; \Pi _ { V } ( \mathbf { h } ) \mid X = x ) \ge I ( Q ; \mathbf { u } \mid X = x )$ . We bound the latter quantity.

Fix t and an active pre-round history $\mathbf { h } _ { t - 1 }$ , and let $\mathcal { A } _ { t } = \neg \mathcal { K } _ { t } \ | \ \mathcal { K } _ { t - 1 }$ indicate an abort in the current round. Let ${ \bf u } _ { t }$ be the single bit appearing at round t.

Let $N \leq T$ be the number of rounds observed in u. Padding the transcript after stopping with a fixed symbol, the conditional chain rule gives

$$
\begin{array} { l } { { \displaystyle I ( Q ; { \bf u } ) = \sum _ { t = 1 } ^ { T } I ( Q ; { \bf u } _ { t } | { \bf h } _ { t - 1 } , K _ { t - 1 } ) } \ ~ } \\ { { \displaystyle ~ = \sum _ { t = 1 } ^ { T } I ( Q ; { \bf y } _ { t } | { \bf h } _ { t - 1 } , K _ { t - 1 } ) \cdot \mathrm { P r } [ N \ge t ] } \ ~ } \\ { { \displaystyle ~ \geq k _ { \beta } \Delta ^ { 2 } \sum _ { t = 1 } ^ { T } \mathrm { P r } ( N \ge t ) } \ ~ } \\ { { \displaystyle ~ = k _ { \beta } \Delta ^ { 2 } \mathrm { E } [ N ] } , } \end{array}
$$

where the inequality follows from Lemma 8. Let U be the event that, after T rounds, the protocol has neither aborted nor reached the confidence threshold, and set $q = \operatorname* { P r } ( \mathcal { U } )$ . On U, all T rounds are observed. Hence $\mathbb { E } [ N ] \geq T q$ , and therefore

$$
I ( Q ; { \mathbf { u } } \mid { \mathcal { U } } ) \geq k _ { \beta } \Delta ^ { 2 } T q .\tag{9}
$$

Outside $u ,$ by Lemma 9, on abort posterior entropy is zero by full disclosure and on reaching the confidence threshold it is at most $h _ { 2 } ( \beta )$ . Averaging over stopped transcripts gives

$$
\begin{array} { c } { { H ( Q \mid \mathbf { u } ) \leq h _ { 2 } ( \beta ) ( 1 - q ) + q } } \\ { { = h _ { 2 } ( \beta ) + ( 1 - h _ { 2 } ( \beta ) ) q . } } \end{array}
$$

Since

$$
I ( Q ; { \mathbf { u } } ) = h _ { x } - H ( Q \mid { \mathbf { u } } ) ,
$$

we obtain

$$
I ( Q ; { \mathbf { u } } ) \ge h _ { x } - h _ { 2 } ( \beta ) - ( 1 - h _ { 2 } ( \beta ) ) q .\tag{10}
$$

Bringing everything together, set $D = h _ { x } - h _ { 2 } ( \beta )$ , and $c = 1 - h _ { 2 } ( \beta )$ . The assumption $\mu _ { 0 } \in ( \beta , 1 - \beta )$ ensures that $D > 0$ . Inequalities (9) and (10) imply that

$$
I ( Q ; { \bf u } ) \ge \mathrm { m a x } \left\{ k _ { \beta } \Delta ^ { 2 } T q , D - c q \right\}
$$

Multiplying the first inequality by c and the second by $k _ { \beta } \Delta ^ { 2 } T$ , then adding, eliminates $q .$ In other words,

$$
\begin{array} { r } { c I ( Q ; \mathbf { u } ) + k _ { \beta } \Delta ^ { 2 } T I ( Q ; \mathbf { u } ) \geq c k _ { \beta } \Delta ^ { 2 } T q + k _ { \beta } \Delta ^ { 2 } T ( D - c q ) } \\ { = k _ { \beta } \Delta ^ { 2 } T D . \qquad } \end{array}
$$

Thus,

$$
\begin{array} { c } { { ( c + k _ { \beta } \Delta ^ { 2 } T ) I ( Q ; { \bf u } ) \ge k _ { \beta } \Delta ^ { 2 } T D } } \\ { { \mathrm { ~ } } } \\ { { \therefore I ( Q ; { \bf u } ) \ge D \cdot \displaystyle \frac { k _ { \beta } \Delta ^ { 2 } } { c + k _ { \beta } \Delta ^ { 2 } } } } \end{array}
$$

Finally, for every $u \geq 0$

$$
\frac { u } { 1 + u } \geq \frac { 1 } { 2 } \operatorname* { m i n } \{ 1 , u \} .
$$

Taking

$$
u = \frac { k _ { \beta } \Delta ^ { 2 } T } { c } = \frac { k _ { \beta } \Delta ^ { 2 } T } { 1 - h _ { 2 } ( \beta ) }
$$

and applying data processing yields

$$
I ( Q ; \Pi _ { V } ( \mathbf { h } ) \mid X = x ) \ge { \frac { h _ { x } - h _ { 2 } ( { \boldsymbol { \beta } } ) } { 2 } } \operatorname* { m i n } \left\{ 1 , { \frac { k _ { \beta } \Delta ^ { 2 } T } { 1 - h _ { 2 } ( { \boldsymbol { \beta } } ) } } \right\} .
$$

In conclusion, Theorem 3 is a corollary of Theorem 5 for a fixed choice of $\beta .$

## D. Proof of Theorem 4

## D.1. Concentration Bounds

We require the conditional form of Hoefding’s Lemma.

Lemma 10 (Hoefding’s Lemma). Let $\mathcal { G }$ be a sigma-algebra, and let Y be a random variable satisfying $a \leq Y \leq b$ almost surely, where a and b are deterministic constants. $I f \mathbb { E } [ Y \mid { \mathcal { G } } ] = 0 ,$ , then, for every real $\lambda ,$

$$
\mathbb { E } \left[ \exp ( \lambda Y ) | \mathcal { G } \right] \leq \exp \left( \frac { \lambda ^ { 2 } ( b - a ) ^ { 2 } } { 8 } \right)
$$

almost surely.

## Debate protocol with error and drift aborts

All parties have access to ${ \mathcal { O } } ,$ an input $x \in \{ 0 , 1 \} ^ { n }$ , and a K-Lipschitz probabilistic oracle machine M. A claims that $\begin{array} { r } { \operatorname* { P r } \left[ \overset { \cdot } { M } ( x ) = 1 \right] \geq \frac { 2 } { 3 } . } \end{array}$ , and the cross-examiner B disputes this claim. Let $W \in [ T ]$ be the trailing-window length used by an abort drift challenge.

1. Let $\mathbf { y } = ( \mathsf { y } _ { 1 } , \hdots , \mathsf { y } _ { T } )$ be the random variable representing the transcript of $M ( x )$ and set $d = \lceil 1 5 0 K \rceil$

2. For debate round $t \in \left\lceil T \right\rceil$ :

(a)-(d) Identical to the equivalent steps in Figure 2.

(e) B can abort error if (it believes) A has deviated on the most recent update or abort drift if (it believes) A has deviated cumulatively over the most recent W updates and $t \geq W$

3. If abort error was issued in round t:

(a) V draws $1 9 2 d ^ { 2 }$ log 100 independent samples of $\mathbf { y } _ { t } ,$ conditioned on ${ \bf y } _ { I ( t ) } = { \pmb a } _ { I ( t ) }$ , and uses their sample mean to obtain an estimate $\hat { p } _ { t } ^ { \mathcal { O } } \mathrm { o f } \mathrm { P r }  \mathrm { y } _ { t } = 1  \mathrm { \mathbf { y } } _ { I ( t ) } = \pmb { a } _ { I ( t ) } $

(b) V checks whether $\begin{array} { r } { \left| \hat { p } _ { t } ^ { \mathcal { O } } - \hat { p } _ { t } \right| \geq \frac { 1 } { 4 d } } \end{array}$ . If the inequality holds, V outputs $0 ;$ otherwise, $V$ outputs 1.

4. If abort drift was issued in round t:

(a) For each $s \in J = \{ t - W + 1 , \ldots , t \}$ , V draws $1 9 2 d ^ { 2 } \log ( 1 0 0 W )$ independent samples of $\mathrm { y } _ { s } ,$ conditioned on ${ \bf y } _ { I ( s ) } = { \pmb a } _ { I ( s ) } $ , and uses the sample mean to obtain an estimate $\hat { p } _ { s } ^ { \mathcal { O } }$ of $\mathrm { P r } \left[ \mathbf { y } _ { s } = 1 \middle | \mathbf { y } _ { I ( s ) } = \pmb { a } _ { I ( s ) } \right]$

(b) V checks whether $\begin{array} { r } { \left| \sum _ { s \in J } \left( \hat { p } _ { s } ^ { \mathcal { O } } - \hat { p } _ { s } \right) \right| \geq \frac { \sqrt { N } } { 4 d } } \end{array}$ . If the inequality holds, V outputs 0; otherwise, V outputs 1.

5. If the protocol was not aborted, then the verifier outputs $a _ { T } .$

Figure 3: Debate protocol for a stochastic oracle with error and drift aborts.

The following bounds will be used to bound “good” sampling behavior during protocol execution.

Lemma 11. Let $( \mathcal { F } _ { t } ) _ { t \geq 0 }$ be a filtration. For each round $t ,$ let $p _ { t } \in [ 0 , 1 ]$ be $\mathcal { F } _ { t - 1 }$ -measurable. Conditional on $\mathcal { F } _ { t - 1 }$ , suppose $X _ { t , 1 } , \dots , X _ { t , R }$ are independent Bernoulli random variables with common mean $p _ { t }$ . Define the F<sub>t</sub>-measurable quantities $\begin{array} { r } { \hat { p } _ { t } = R ^ { - 1 } { \sum _ { i = 1 } ^ { R } } X _ { t , i } , } \end{array}$ , and $D _ { t } = \hat { p } _ { t } - p _ { t }$ . Then, almost surely, the following bounds hold.

1. For every real λ, E $\begin{array} { r } { [ \exp ( \lambda D _ { t } ) | \mathcal { F } _ { t - 1 } ] \le \exp \left( \frac { \lambda ^ { 2 } } { 8 R } \right) } \end{array}$

2. For every $u > 0 , \mathrm { P r } \left[ | D _ { t } | \ge u | \mathcal { F } _ { t - 1 } \right] \le 2 \exp ( - 2 R u ^ { 2 } )$

3. For any deterministic integers $0 \leq a < b$ and every $u > 0 _ { z }$

$$
\operatorname* { P r } \left[ \left| \sum _ { s = a + 1 } ^ { b } D _ { s } \right| \geq u \middle | \mathcal { F } _ { a } \right] \leq 2 \exp \left( - \frac { 2 R u ^ { 2 } } { b - a } \right) .
$$

Proof. Conditional on $\mathcal { F } _ { t - 1 }$ , each variable $\frac { X _ { t , i } - p _ { t } } { R }$ has mean zero and lies in the interval $[ - p _ { t } / R , 1 - p _ { t } / R ]$ , whose length is $1 / R$ . By Lemma 10,

$$
\mathbb { E } \left[ \exp \left( \lambda \frac { X _ { t , i } - p _ { t } } { R } \right) \Big | \mathcal { F } _ { t - 1 } \right] \leq \exp \left( \frac { \lambda ^ { 2 } } { 8 R ^ { 2 } } \right) .
$$

Conditional independence of the R samples therefore yields

$$
\begin{array} { r l r } {  { \mathbb { E } [ \exp ( \lambda D _ { t } ) | \mathcal { F } _ { t - 1 } ] = \prod _ { i = 1 } ^ { R } \mathbb { E } [ \exp ( \lambda \frac { X _ { t , i } - p _ { t } } { R } ) \bigg | \mathcal { F } _ { t - 1 } ] } } \\ & { } & { \leq \exp ( \frac { \lambda ^ { 2 } } { 8 R } ) . } \end{array}\tag{11}
$$

This proves the first assertion. For $\lambda > 0$ , the Markov inequality gives

$$
\mathrm { P r } \left[ D _ { t } \geq u | \mathcal { F } _ { t - 1 } \right] \leq \exp \left( - \lambda u + \frac { \lambda ^ { 2 } } { 8 R } \right) .
$$

Choosing $\lambda = 4 R u$ gives the upper-tail bound

$$
\begin{array} { r } { \operatorname* { P r } \left[ D _ { t } \ge u | \mathcal { F } _ { t - 1 } \right] \le \exp ( - 2 R u ^ { 2 } ) . } \end{array}
$$

Applying the same argument $\mathrm { t o } - D _ { t }$ and adding the two tail bounds proves the second assertion. For the third assertion, write

$$
S _ { a , b } = \sum _ { s = a + 1 } ^ { b } D _ { s } .
$$

Since $S _ { a , b - 1 }$ is $\mathcal { F } _ { b - 1 }$ -measurable, (11) implies

$$
\begin{array} { r l } & { \mathbb { E } \left[ \exp ( \lambda S _ { a , b } ) \vert \mathcal { F } _ { a } \right] = \mathbb { E } \left[ \exp ( \lambda S _ { a , b - 1 } ) \mathbb { E } \left[ \exp ( \lambda D _ { b } ) \vert \mathcal { F } _ { b - 1 } \right] \vert \mathcal { F } _ { a } \right] } \\ & { \qquad \leq \exp \left( \frac { \lambda ^ { 2 } } { 8 R } \right) \mathbb { E } \left[ \exp ( \lambda S _ { a , b - 1 } ) \vert \mathcal { F } _ { a } \right] . } \end{array}
$$

Iterating over the $b - a$ rounds gives

$$
\mathbb { E } \left[ \exp ( \lambda S _ { a , b } ) | \mathcal { F } _ { a } \right] \leq \exp \left( \frac { ( b - a ) \lambda ^ { 2 } } { 8 R } \right) .
$$

The Markov inequality at $\begin{array} { r } { \lambda = \frac { 4 R u } { b - a } . } \end{array}$ , gives

$$
\mathrm { P r } \left[ S _ { a , b } \geq u | \mathcal { F } _ { a } \right] \leq \exp \left( - \frac { 2 R u ^ { 2 } } { b - a } \right) .
$$

Applying the same argument to $- S _ { a , b }$ and adding the two tails completes the proof.

## D.2. Protocol Completeness

Given the honest behaviour by both parties, we assume that the transcript update is a fresh Bernoulli draw with parameter $\boldsymbol { \hat { p } } _ { t } .$ and that $p _ { t } = \mathrm { P r } [ \mathbf { y } _ { t } \mid \mathbf { y } _ { I ( t ) = { \pmb a } _ { I ( s ) } }$ is the corresponding true transition probability of M. The protocol uses full windows $J _ { t } = \{ t - W + 1 , \ldots , t \}$ , with drift challenges permitted only for $t \geq W$ . The verifier’s drift statistic must sum the s-indexed discrepancies. That that, for probabilistic Turing machine $M ,$ the standard bounded-error guarantee is $\begin{array} { r } { \operatorname* { P r } [ \bar { M } ^ { \mathcal { O } } ( x ) = 1 ] \ge \frac { 2 } { 3 } } \end{array}$

Lemma 12 (Task correctness). Let M be a K-Lipschitz probabilistic Turing machine that decides L $R \geq 1 9 2 d ^ { 2 } \log ( 1 0 0 T )$ , For any $x \in L ,$ , any fixed window length $W \in [ T ]$ , honest strategy $\sigma _ { A } \sim ( R , 0 )$ and arbitrary cross-examiner strategy $\sigma _ { B } ,$

$$
\operatorname* { P r } [ V \to 1 ] \geq { \frac { 3 } { 5 } } .
$$

Proof. Simulate an auxiliary execution $\boldsymbol { \mathcal { S } }$ that generates all $T$ transcript updates using the reporting strategy $\sigma _ { A }$ , ignoring any challenge for purposes of continuing the transcript. Let $\mathbf { y } ^ { s }$ denote the random transcript of the simulated execution and before each round $t ,$ let $\mathbf { h } _ { t - 1 } ^ { s }$ contain the complete history

on the simulated execution. Conditional on $\mathbf { h } _ { t - 1 } ^ { \mathsf { s } }$ , the honest reporter draws R fresh, independent Bernoulli samples $X _ { t , 1 } , \ldots , X _ { t , R } ,$ each with mean $p _ { t }$ , and reports

$$
\hat { p } _ { t } ^ { \mathrm { s } } = \frac { 1 } { R } \sum _ { j = 1 } ^ { R } X _ { t , j } .
$$

The next transcript bit $\boldsymbol { \mathrm { y } } _ { t } ^ { s }$ is generated using independent randomness with success probability $\hat { p } _ { t } ^ { s } .$ Consequently,

$$
\operatorname* { P r } [ \mathbf { y } _ { t } ^ { s } = 1 \mid \mathbf { h } _ { t - 1 } ^ { s } ] = \mathbb { E } [ \hat { p } _ { t } ^ { s } \mid \mathbf { h } _ { t - 1 } ^ { s } ] = p _ { t } .
$$

Thus, the simulated transcript has exactly the transition probabilities of the true execution of $M ^ { \mathcal { O } } ( x )$ If $\mathbf { y } _ { T } ^ { \mathsf { s } }$ denotes its terminal output, then

$$
\mathrm { P r } [ \mathsf { y } _ { T } ^ { \mathsf { s } } = 1 ] = \mathrm { P r } [ M ^ { \mathcal { O } } ( x ) = 1 ] \geq \frac { 2 } { 3 } .
$$

Couple the actual protocol to this simulated execution using the same samples and transcript-update randomness until the actual protocol stops. If no challenge occurs, the actual output equals ${ \bf y } _ { T } ^ { \mathsf { s i m } } = a _ { T } ^ { \mathsf { s i m } }$ The simulated continuation after a challenge is used only for analysis and is generated independently of the subsequent audit randomness of $V .$

Define the reporting error $D _ { t } ^ { \mathsf { s i m } } = \hat { p } _ { t } ^ { \mathsf { s i m } } - p _ { t }$ . For each round $t ,$ let $\mathcal { E } _ { t } ^ { ( 1 ) }$ be the event $\begin{array} { r } { | D _ { t } ^ { \mathsf { s i m } } | < \frac { 1 } { 8 d } } \end{array}$ . For each $t \geq W$ , let $\mathcal { E } _ { t } ^ { ( 2 ) }$ be the event

$$
\left| \sum _ { s \in J _ { t } } D _ { s } ^ { \mathsf { s i m } } \right| < { \frac { \sqrt { W } } { 8 d } } .
$$

Let ${ \mathcal { G } } ^ { \mathsf { s i m } }$ be the intersection of all these individual and window events in the simulated execution. By Lemma 11, for every round $t ,$

$$
\mathrm { P r } [ \lnot \mathcal { E } _ { t } ^ { ( 1 ) } \mid \mathbf { h } _ { t - 1 } ^ { \mathrm { s i m } } ] = \mathrm { P r } \left[ \mid D _ { t } ^ { \mathrm { s i m } } \mid \geq \frac { 1 } { 8 d } \Bigg | \mathbf { h } _ { t - 1 } ^ { \mathrm { s i m } } \right] \leq 2 \exp \left( - \frac { R } { 3 2 d ^ { 2 } } \right) .
$$

Similarly, for window errors, Lemma 11 gives, for every $u > 0$

$$
\operatorname* { P r } [ | \sum _ { s \in J _ { t } } D _ { s } ^ { \mathrm { s i m } } | \geq u  { | \begin{array} { l } { \mathbf { h } _ { t - W } ^ { \mathrm { s i m } } } \\ { \mathbf { h } _ { t - W } ^ { \mathrm { s i m } } } \end{array} ] } \leq 2 \exp ( - { \frac { 2 R u ^ { 2 } } { W } } ) .
$$

Therefore, it follows that

$$
\operatorname* { P r } [ \lnot \mathcal { E } _ { t } ^ { ( 2 ) } \mid \mathbf { h } _ { t - W } ^ { \mathrm { s i m } } ] \leq 2 \exp \left( - \frac { R } { 3 2 d ^ { 2 } } \right) .
$$

There are T individual events and at most T window events. Therefore, by a union bound,

$$
\begin{array} { l } { \displaystyle \operatorname* { P r } [ \lnot { \mathcal { G } } ^ { \mathsf { s i m } } ] \leq 4 T \exp \left( - { \frac { R } { 3 2 d ^ { 2 } } } \right) } \\ { \leq 4 T ( 1 0 0 T ) ^ { - 6 } . } \end{array}\tag{12}
$$

Let $\boldsymbol { A } _ { t } ^ { ( \mathsf { e } ) }$ denote the event that abort error occurs at time t. Condition on a complete pre-audit history $\mathbf { h } ^ { \mathsf { s i m } }$ at which $\boldsymbol { A } _ { t } ^ { ( \mathsf { e } ) }$ occurs. This fixes $p _ { t }$ and $\hat { p } _ { t }$ . Suppose ${ \mathcal { G } } ^ { \mathsf { s i m } }$ occurs. If V rejects, then

$$
\left. \hat { p } _ { t } ^ { \mathcal { O } } - \hat { p } _ { t } \right. \geq \frac { 1 } { 4 d } ,
$$

so the triangle inequality implies

$$
\left| \hat { p } _ { t } ^ { \mathcal { O } } - p _ { t } \right| > \frac { 1 } { 8 d } .
$$

V uses $r ^ { ( \mathrm { e } ) } = \left\lceil 1 9 2 d ^ { 2 } \log 1 0 0 \right\rceil$ fresh, independent samples. Therefore, by Lemma 11,

$$
\operatorname* { P r } [ V \to 0 \mid \mathbf { h } ^ { \mathrm { s i m } } , \mathcal { G } ^ { \mathrm { s i m } } , \mathcal { A } _ { t } ^ { \mathrm { ( e ) } } ] \leq 2 \exp \left( - \frac { r ^ { \mathrm { ( e ) } } } { 3 2 d ^ { 2 } } \right) \leq 2 \cdot 1 0 0 ^ { - 6 } ,\tag{13}
$$

under the protocol execution couple to the simulated execution.

Let $\boldsymbol { A } _ { t } ^ { ( \mathsf { d } ) }$ denote the event that abort drift occurs at time $t \geq W$ . As above, now condition on a complete pre-audit history $\mathbf { h } ^ { \mathsf { s i m } }$ at which $\mathcal { A } _ { t } ^ { ( \mathsf { d } ) }$ occurs. Suppose ${ \mathcal { G } } ^ { \mathsf { s i m } }$ occurs. For each $s \in J _ { t } .$ , V draws $r ^ { ( \mathsf { d } ) } = \left\lceil 1 9 2 d ^ { 2 } \log ( 1 0 0 W ) \right\rceil$ fresh samples, independently across all audited rounds and sample indices. Define

$$
S _ { V } = \sum _ { s \in J _ { t } } ( \hat { p } _ { s } ^ { \mathcal { O } } - p _ { s } ) .
$$

Conditioned on $\mathbf { h } ^ { \mathsf { s i m } } , S _ { V }$ is a sum of $W r ^ { \mathrm { ( d ) } }$ independent centered variables, each having range length $1 / r ^ { \left( { \mathsf { d } } \right) }$ . Hence, by Lemma 11

$$
\operatorname* { P r } \left[ \left| S _ { V } \right| \geq u \middle | \mathbf { h } ^ { \mathrm { s i m } } , \mathcal { G } ^ { \mathrm { s i m } } , \mathcal { A } _ { t } ^ { \mathrm { ( d ) } } \right] \leq 2 \exp \left( - \frac { 2 r ^ { \mathrm { ( d ) } } u ^ { 2 } } { W } \right) .
$$

${ \mathrm { I f ~ } } | S _ { V } | < { \sqrt { W } } / 8 d ,$ the triangle inequality gives

$$
\left| \sum _ { s \in J _ { t } } ( \hat { p } _ { s } ^ { \mathcal { O } } - \hat { p } _ { s } ) \right| < \frac { \sqrt { W } } { 4 d } .
$$

Therefore, V accepts the challenge audit. Consequently,

$$
\operatorname* { P r } [ V \to 0 \mid \mathbf { h } ^ { \mathrm { s i m } } , \mathcal { G } ^ { \mathrm { s i m } } , \mathcal { A } _ { t } ^ { ( \mathrm { d } ) } ] \leq 2 \exp \left( - \frac { r ^ { ( \mathrm { d } ) } } { 3 2 d ^ { 2 } } \right) \leq 2 ( 1 0 0 W ) ^ { - 6 } \leq 2 \cdot 1 0 0 ^ { - 6 } .\tag{14}
$$

Let $\mathcal { I }$ be the event that a challenge occurs and the verifier outputs 0. By (13) and (14),

$$
\begin{array} { r l } & { \operatorname* { P r } [ \mathcal { I } \cap \mathcal { G } ] \leq \displaystyle \sum _ { t , c } \operatorname* { P r } [ \mathcal { A } _ { t } ^ { ( \mathrm { c } ) } \cap \mathcal { G } ^ { \mathrm { s i m } } ] \operatorname* { P r } [ V \to 0 \mid \mathcal { A } _ { t } ^ { ( \mathrm { c } ) } \cap \mathcal { G } ^ { \mathrm { s i m } } ] } \\ & { \qquad \leq 2 \cdot 1 0 0 ^ { - 6 } \displaystyle \sum _ { t , \mathrm { c } } \operatorname* { P r } [ \mathcal { A } _ { t } ^ { ( \mathrm { c } ) } \cap \mathcal { G } ^ { \mathrm { s i m } } ] } \\ & { \qquad \leq 2 \cdot 1 0 0 ^ { - 6 } . } \end{array}\tag{15}
$$

Conditional on each pre-audit history, the audit randomness is independent of the auxiliary continuation.   
Consequently, further conditioning on ${ \mathcal { G } } ^ { \mathsf { s i m } }$ does not change the conditional law of the audit samples.

Under the coupling, if $\mathbf { y } _ { T } ^ { \mathsf { s i m } } = 1$ and the actual protocol outputs 0, then a challenge must have occurred and been rejected. Indeed, without a challenge, the actual output equals $\mathsf { y } _ { T } ^ { \mathsf { s i m } } = 1$ . Thus, by (12) and (15),

$$
\begin{array} { r l } {  { \operatorname* { P r } [ V \to 1 ] \ge \operatorname* { P r } [ \mathrm { y } _ { T } ^ { \mathrm { s i m } } = 1 ] - \operatorname* { P r } [ \mathcal { I } ] } } \\ & { \ge \operatorname* { P r } [ \mathrm { y } _ { T } ^ { \mathrm { s i m } } = 1 ] - \operatorname* { P r } [ \neg \mathcal { G } ^ { \mathrm { s i m } } ] - \operatorname* { P r } [ \mathcal { I } \cap \mathcal { G } ^ { \mathrm { s i m } } ] } \\ & { \ge \frac 2 3 - 4 T ( 1 0 0 T ) ^ { - 6 } - 2 \cdot 1 0 0 ^ { - 6 } > \frac 3 5 } \end{array}
$$

The argument holds for every legal cross-examiner strategy $\sigma _ { B }$ , proving the lemma.

## D.3. Smaller Admissible Region

For the second part of Theorem 4 we require the following.

Lemma 13. Let L be any language decided by a Turing machine M and $W \in \ [ T ]$ be the slidingwindow parameter. For $R \geq 1 9 2 d ^ { 2 } \log ( 1 0 0 T ) , \Delta \geq 1 / ( d \sqrt { W } ) , \tau _ { 1 } = 1 / ( 2 d )$ , and $\tau _ { 2 } = \sqrt { W } / ( 2 d )$ , let $\sigma _ { A } \sim ( R , \Delta )$ , and $\sigma _ { B } \sim \left( R , \tau _ { 1 } , \tau _ { 2 } \right)$

$$
\Phi _ { A } ( x , \sigma _ { A } \circ \sigma _ { B } ) = \mathrm { P r } [ V  1 ] < \frac { 1 } { 1 5 0 } .
$$

Revised proof. Fix $Q \in \{ - 1 , 1 \}$ . All probabilities below are conditional on this fixed value of $Q .$ Construct an simulated execution S that continues for $T$ rounds according to strategies $\sigma _ { A }$ and $\sigma _ { B } .$ ignoring challenges. Couple the actual protocol to this simulated execution until its first challenge. The simulated continuation uses randomness independent of the V’s fresh audit randomness.

At round $t ,$ let $p _ { t }$ be the true oracle probability at the current state $\mathbf { h } ^ { \mathsf { s i m } }$ . Let $\bar { p } _ { t }$ and $\hat { q } _ { t }$ be A’s and $B ^ { \prime } s$ unperturbed sample means, each based on R fresh $\mathsf { B e r n } ( p _ { t } )$ samples. Define $D _ { A } ( t ) ^ { \mathsf { s i m } } = \bar { p } _ { t } ^ { \mathsf { s i m } } - p _ { t } ,$ $\begin{array} { r } { \bar { D _ { B } } ( t ) ^ { \mathrm { s i m } } = \hat { q } _ { t } ^ { \mathrm { s i m } } - p _ { t } , S _ { A } ( t ) = \sum _ { s \in I _ { t } } D _ { A } ( s ) ^ { \mathrm { s i m } } } \end{array}$ , and $\begin{array} { r } { S _ { B } ( t ) = \sum _ { s \in J _ { t } } D _ { B } ( s ) ^ { s \mathsf { i m } } } \end{array}$ . Note $S _ { A } ( t )$ is defined using the unperturbed estimate $\bar { p } _ { s } ^ { \mathsf { s i m } }$ , not the perturbed report.

Let $\mathcal { G } _ { A B }$ be the event that, for both $\begin{array} { r } { i \in \{ A , B \} , | D _ { i } ( t ) ^ { \mathsf { s i m } } | < \frac { 1 } { 8 d } } \end{array}$ for every , for every $t \in \{ 1 , \ldots , T \}$ , and $\vert S _ { i } ( t ) \vert < \sqrt { W } / 8 d$ for every $t \in \{ W , \ldots , T \}$

Conditional on the pre-round history $\mathbf { h } _ { t - 1 } ^ { \mathsf { s i m } }$ and $Q ,$ each sample mean is unbiased for $p _ { t }$ . By Lemma11, for each $i \in \{ A , B \}$ ,

$$
\mathrm { P r } \left[ | D _ { i } ( t ) ^ { \mathrm { s i m } } | \geq \frac { 1 } { 8 d } \bigg | Q , \mathbf { h } _ { t - 1 } ^ { \mathrm { s i m } } \right] \leq 2 \exp \left( - \frac { R } { 3 2 d ^ { 2 } } \right) ,
$$

and

$$
\mathrm { P r } [ | S _ { i } ( t ) | \geq \frac { \sqrt { W } } { 8 d } | Q , \mathbf { h } _ { t - W } ^ { \mathrm { s i m } } ] \leq 2 \exp ( - \frac { R } { 3 2 d ^ { 2 } } ) .
$$

There are at most 4T events in the definition of $\mathcal { G } _ { A B }$ . Consequently,

$$
\begin{array} { l } { \displaystyle \operatorname* { P r } [ \mathcal { G } _ { A B } ^ { c } \mid Q ] \leq 8 T \exp \left( - \frac { R } { 3 2 d ^ { 2 } } \right) } \\ { \leq 8 T ( 1 0 0 T ) ^ { - 6 } . } \end{array}\tag{16}
$$

On $\mathcal { G } _ { A B }$ , the assumed absence of clipping gives $\hat { p } _ { t } ^ { \mathsf { s i m } } = \bar { p } _ { t } ^ { \mathsf { s i m } } + Q \Delta$ . Under the coupled execution. if an abort error challenge has not already occurred, the execution reaches the first full window $J _ { W }$ . At that window, for $\Delta \ge 1 / ( d \sqrt { W } )$ ,

$$
\begin{array} { r l r } {  { \sum _ { s \in { \mathcal J } _ { w } } ( \hat { p } _ { s } - \hat { q } _ { s } ) \Bigg | = | Q W \Delta + S _ { A } ( W ) - S _ { B } ( W ) | } } \\ & { } & \\ & { \geq W \Delta - | S _ { A } ( W ) | - | S _ { B } ( W ) | } \\ & { > \frac { \sqrt { W } } { d } - \frac { 2 \sqrt { W } } { 8 d } } \\ & { = \frac { 3 \sqrt { W } } { 4 d } } \\ & { > \frac { \sqrt { W } } { 2 d } . } \end{array}\tag{17}
$$

Thus B issues a challenge no later than round W. It remains to show that either possible challenge type leads to output 0, except with uniformly small audit probability.

Let $\boldsymbol { A } _ { t } ^ { ( \mathsf { e } ) }$ denote the event that abort error occurs at time t. Condition on a complete pre-audit history $\mathbf { h } ^ { \mathsf { s i m } }$ at which $\boldsymbol { A } _ { t } ^ { ( \mathsf { e } ) }$ occurs. This fixes $\bar { p } _ { t } ^ { \mathsf { s i m } } , \hat { q } _ { t } ^ { \mathsf { s i m } }$ , and $p _ { t }$ . By the challenge rule,

$$
| \hat { p } _ { t } ^ { \mathsf { s i m } } - \hat { q } _ { t } ^ { \mathsf { s i m } } | > \frac { 1 } { 2 d } .
$$

On events $\mathcal { G } _ { A B }$ and $\vert \hat { q } _ { t } - p _ { t } \vert < 1 / ( 8 d )$ the following holds

$$
| \hat { p } _ { t } ^ { \mathrm { s i m } } - p _ { t } | \geq | \hat { p } _ { t } ^ { \mathrm { s i m } } - \hat { q } _ { t } ^ { \mathrm { s i m } } | - | \hat { q } _ { t } ^ { \mathrm { s i m } } - p _ { t } | > \frac { 1 } { 2 d } - \frac { 1 } { 8 d } = \frac { 3 } { 8 d } .
$$

If the $V ^ { \ast } s$ fresh estimate satisfies $\lvert \hat { p } _ { t } ^ { \mathcal { O } } - p _ { t } \rvert < 1 / ( 8 d )$ , then

$$
\begin{array} { r l } & { \bigl | \hat { p } _ { t } ^ { \mathcal { O } } - \hat { p } _ { t } ^ { \mathsf { s i m } } \bigr | \geq \bigl | \hat { p } _ { t } ^ { \mathsf { s i m } } - p _ { t } \bigr | - \bigl | \hat { p } _ { t } ^ { \mathcal { O } } - p _ { t } \bigr | } \\ & { \qquad > \displaystyle \frac { 3 } { 8 d } - \frac { 1 } { 8 d } } \\ & { \qquad = \displaystyle \frac { 1 } { 4 d } . } \end{array}
$$

Hence, V outputs 0. It follows that

$$
\begin{array} { r l r } & { } & { \mathrm { P r } [ V  1 \mid { \mathcal A } _ { t } ^ { ( \mathrm { e } ) } , { \bf h } ^ { \mathrm { s i m } } , { \mathcal G } _ { A B } ] \le \mathrm { P r } [ \mid \hat { p } _ { t } ^ { \mathcal { O } } - p _ { t } \mid > 1 / ( 8 d ) ] } \\ & { } & { \geq 2 \exp ( - \frac { r ^ { ( \mathrm { e } ) } } { 3 2 d ^ { 2 } } ) \le 2 \cdot 1 0 0 ^ { - 6 } . } \end{array}\tag{18}
$$

Similarly, let $\boldsymbol { A } _ { t } ^ { ( \mathsf { d } ) }$ denote the event that abort drift occurs at time t. Condition on a complete pre-audit history $\mathbf { h } ^ { \mathsf { s i m } }$ at which $\boldsymbol { A } _ { t } ^ { ( \mathsf { d } ) }$ occurs. Define

$$
S _ { V } ( t ) = \sum _ { s \in J _ { t } } \left( \hat { p } _ { s } ^ { \mathcal { O } } - p _ { s } \right) .
$$

On $\mathcal { G } _ { A B }$ and the event $\begin{array} { r } { | S _ { V } ( t ) | < \frac { \sqrt { W } } { 8 d } } \end{array}$

$$
\begin{array} { r l } {  { \sum _ { \delta \in J _ { t } } ( \hat { p } _ { s } - \hat { p } _ { s } ^ { \mathcal { O } } ) \Bigg | = | Q W \Delta + S _ { A } ( t ) - S _ { V } ( t ) | } } \\ & { \geq W \Delta - | S _ { A } ( t ) | - | S _ { V } ( t ) | } \\ & { > \frac { \sqrt { W } } { d } - \frac { \sqrt { W } } { 8 d } - \frac { \sqrt { W } } { 8 d } } \\ & { = \frac { 3 \sqrt { W } } { 4 d } } \\ & { > \frac { \sqrt { W } } { 4 d } . } \end{array}
$$

Therefore V again outputs 0 under good sample estimates. Thus, by Lemma 11,

$$
\begin{array} { r l r } { \left. { \operatorname* { P r } [ V \to 1 \mid { \mathcal { A } } _ { t } ^ { \mathrm { ( d ) } } , \mathbf { h } ^ { \mathrm { s i m } } , { \mathcal { G } } _ { A B } ] \leq \operatorname* { P r } \left[ \left| S _ { V } ( t ) \right| \geq \frac { \sqrt { W } } { 8 d } \right| \mathbf { h } ^ { \mathrm { s i m } } , Q  } } \\ & \right]{ } & \\ & { } & { \leq 2 \exp \left( - \frac { r ^ { \mathrm { ( d ) } } } { 3 2 d ^ { 2 } } \right) \leq 2 ( 1 0 0 W ) ^ { - 6 } \leq 2 \cdot 1 0 0 ^ { - 6 } . } \end{array}\tag{19}
$$

By (17), on $\mathcal { G } _ { A B } { : }$ , a challenge necessarily occurs at a time $t \leq W$ . Its time and type select one audit. That is, they do not create multiple audit-failure penalties. By (18) and (19),

$$
\begin{array} { r l } & { \operatorname* { P r } [ V  1 , \mathcal { G } _ { A B } | Q ] = \displaystyle \sum _ { t = 1 } ^ { W } \sum _ { \mathfrak { c } } \operatorname* { P r } [ V  1 \cap \mathcal { A } _ { t } ^ { ( \mathfrak { c } ) } \cap \mathcal { G } _ { A B } | Q ] } \\ & { \qquad = \displaystyle \sum _ { t = 1 } ^ { W } \sum _ { \mathfrak { c } } \operatorname* { P r } [ V  1 \mid \mathcal { A } _ { t } ^ { ( \mathfrak { c } ) } \cap \mathcal { G } _ { A B } Q ] \cdot \operatorname* { P r } [ \mathcal { A } _ { t } ^ { ( \mathfrak { c } ) } \cap \mathcal { G } _ { A B } \mid Q ] } \\ & { \qquad \leq 2 \cdot 1 0 0 ^ { - 6 } \displaystyle \sum _ { t = 1 } ^ { W } \sum _ { \mathfrak { c } } \operatorname* { P r } [ V  1 \operatorname* { P r } [ \mathcal { A } _ { t } ^ { ( \mathfrak { c } ) } \cap \mathcal { G } _ { A B } \mid Q ] } \\ & { \qquad < 2 \cdot 1 0 0 ^ { - 6 } . } \end{array}
$$

Together with (16),

$$
{ \begin{array} { r l } & { \operatorname* { P r } [ V \to 1 \mid Q ] \leq \operatorname* { P r } [ \neg { \mathcal { G } } _ { A B } \mid Q ] + \operatorname* { P r } [ V \to 1 , { \mathcal { G } } _ { A B } \mid Q ] } \\ & { \qquad \leq 8 T ( 1 0 0 T ) ^ { - 6 } + 2 \cdot 1 0 0 ^ { - 6 } } \\ & { \qquad < { \frac { 1 } { 1 5 0 } } . } \end{array} }
$$

The bound is uniform in Q. Averaging over Q completes the proof.