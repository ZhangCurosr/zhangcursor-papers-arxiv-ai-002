# Query-Driven Learning for Lifted Reasoning

Luise Ge<sup>1</sup> , Brendan Juba<sup>1</sup>

<sup>1</sup>Washington University in St. Louis {g.luise, bjuba}@wustl.edu

## Abstract

Learning and reasoning are both essential to intelligent agents, but have traditionally been treated as two separate stages: first learning a knowledge base from data (or taking one as given), and then verifying queries as if the knowledge base were complete. This separation has been shown to be inefficient when the downstream goal is to verify only a small set of queries. Moreover, it neglects the possibility that data and a knowledge base contain partial yet complementary information, requiring an agent to combine both sources to justify or refute a query. Motivated by the need to integrate learning and reasoning more tightly, we study the problem of query-driven learning for reasoning. More specifically, we extend prior work to the first-order setting, which is critical for practical applications as the number of ground atoms typically grows rapidly. We begin by asking when validating or refuting a universal query given partial data and knowledge base is even possible, and identify a necessary and sufficient condition called bounded concealment. We then propose a quasi-polynomial-time algorithm that solves the task at the lifted level. Finally, we establish soundness and completeness guarantees relative to space-bounded lifted proofs, along with quasipolynomial sample and runtime bounds, demonstrating that lifted query-driven reasoning can be efficient even when unrestricted knowledge-base learning is not.

## 1 Introduction

Classical proof systems have achieved remarkable success in settings where complete knowledge can be axiomatized in advance, from mathematical theorem proving to hardware and software verification. Their transparency and rigor, demonstrated in independently checkable proofs, are especially valuable in high-stakes applications. However, their applicability in real-world domains is limited by a strong prerequisite: that all relevant axioms are known a priori.

What could an agent do when this prerequisite fails? A natural approach is to first construct an expanded knowledge base from as much data as possible and then reason over it. Yet this “learn everything first” strategy can be computationally prohibitive and often unnecessary: for any particular downstream question, only a small fraction of the learned facts and rules are actually relevant. This motivates a more direct, query-driven alternative: given a specific query, can the agent identify and validate only the additional knowledge needed to justify or refute it?

Concretely, we build on the “learning to reason” framework of Khardon and Roth (1997), in particular the extension to partial information with query-driven explicit learning by Juba (2019). Their approach takes a background knowledge base and examples from a distribution as input, and attempts to answer a given query by returning an explicit justification when one exists (or rejecting otherwise).

Their results are established at the propositional level. In relational domains with many objects, however, reasoning at the ground level quickly becomes infeasible: both memory and computation scale with the number of ground instances. For instance, the rule ∀x.(Bird(x) =⇒ Flies(x)) over a domain of size n yields n ground implications, and rules over k variables can induce Θ(n<sup>k</sup>) ground ones. This motivates lifted reasoning, where one reasons directly with first-order structure. In this work, we ask whether a querydriven learning to reason approach can be carried out at the lifted level: given a lifted knowledge base and possibly partial data, can we efficiently verify a universally quantified query by producing a lifted proofor reject otherwise?

Notably, the implicit learning to reason framework has made progress toward this goal. In particular, the approach of Belle and Juba (2019) handles universally quantified queries in a limited but effective way: assuming that the relevant counterexamples or supporting evidence are directly “witnessed” in the sampled partial models, it estimates the probability that a query clause is valid, but does not produce a rigorous proof. Using the Belle–Levesque grounding trick (Levesque, 1998; Belle, 2017), once a small, critical set of ground instances is observed, the system treats the corresponding universally quantified clause as learned and adds it to the knowledge base. Subsequent universally quantified queries can then be reduced to a propositional unsatisfiability check over a finite grounded fragment.

Our objective is to go beyond this statistical estimation. We aim to prove universally quantified clausal queries, i.e., to certify that the clause holds over the entire domain. This is fundamentally more demanding: unlike falsity, truth of a universal statement typically admits no finite set of witnesses. Consequently, proof requires genuinely lifted reasoning—first-order proof search that accounts for all relevant groundings induced by the background theory and the data—rather than relying on a grounding shortcut that is tailored to finding counterexamples.

At the same time, both the data and the knowledge base may be incomplete. Even setting aside the possibility for having contaminated data, limited observability can make the task impossible: if violations of a clause can be systematically hidden, no procedure can reliably certify universal truth. Therefore, we first characterize when proof is information-theoretically possible under partial observation. We identify a necessary and sufficient condition named bounded concealment, extending the formulation of Michael (2010) from learning over finite domains, which requires that if a universally quantified clause is false, then with non-negligible probability the partial observations reveal at least one grounding that witnesses this falsity. Then, under bounded concealment, we show how to reason soundly in a credulous manner: we treat clauses as tentatively valid until we observe explicit counterexamples, with the guarantee that persistent falsity will be exposed with sufficient probability.

Building on this principle, our approach reasons completely at the lifted level. Rather than evaluating ground instances one by one, we apply clause-level proof procedures directly to quantified formulas, enabling generalization across objects and avoiding blow-ups from grounding. This allows us to positively establish universally quantified queries—an ability not provided by the original skeptical, refutation-oriented framework of Belle and Juba (2019).

Our contribution We propose the first query-driven framework that learns in service of lifted reasoning, providing simultaneous efficiency and correctness guarantees.

• We broaden Michael (2010)’s bounded concealment condition from finite to infinite domains, and show that our generalized version is a necessary and sufficient condition for query-driven learning to lifted reasoning with incomplete information.

• We introduce a lifted algorithm that produces formal resolution proofs of the universally quantified clausal query $\varphi$ from the given knowledge base and an additional, learned (1 − ϵ)-valid knowledge base (i.e., it fails at most an ϵ fraction of models under the data distribution), as long as

1. there is a proof of $\varphi$ from the given knowledge and an additional 1-valid knowledge base, and

2. the observation process has bounded concealment with respect to the potentially learnable clauses

thus certifying that $\varphi \mathrm { i s } ( 1 - \epsilon ) \mathrm { - v a l i d } .$

• We prove that under bounded concealment, our algorithm constructs treelike lifted resolution proofs with sample complexity in

$$
O \left( \frac { N ^ { s - 1 } \log N + \log ( 1 / \delta ) } { \eta \epsilon } \right)
$$

and running time in

$$
O \left( \frac { N ^ { 2 \left( s - 1 \right) } | \varphi | ( \log N + \log ( 1 / \delta ) ) } { \eta \epsilon } \right)
$$

where N is the number of distinct atoms<sup>1</sup> in the knowledge base, s is the space bound (i.e., the maximum number of clauses to be kept at a time when deriving the corresponding treelike resolution proof), δ bounds the failure probability, and |φ| is the length of the binary encoding of the query formula.

This is the first approach with simultaneous soundness, completeness, and sample and computational efficiency (quasipolynomial in the number of ground relations for lifted query-driven reasoning), enabling scalable and formally checkable inference in complex first-order domains.

## 2 Logical Setting

We begin by formalizing the reasoning component of our task. This requires specifying the logical language used to represent the background knowledge base, the observed data, and the query. Since our goal is lifted reasoning over relational structure, we work in a function-free first-order logical framework (hence purely relational). Specifically, let $\mathcal { L }$ be a first-order language with:

• a countably infinite set of variables $\{ x _ { 1 } , x _ { 2 } , x _ { 3 } , . . . \}$

• predicates such as $\{ P ( x _ { 1 } ) , \ldots , Q ( x _ { 1 } , \ldots , x _ { k } ) , \ldots \}$ , and

• a countably infinite domain $D : = \mathbb { N }$ of rigid designators or names, which serves as the universe of discourse for quantification.

Here all predicates are Boolean-valued. An atom is an expression of the form $P ( t _ { 1 } , \ldots , t _ { k } )$ , where $P$ is a predicate symbol of arity $k ,$ and each $t _ { i }$ is either a variable or a name. We refer to such atoms as lifted atoms when they contain variables. In contrast, a ground atom is an atom in which all terms $t _ { i }$ are names. The set of all ground atoms is denoted by ATOMS. Literals take the form $P ( t _ { 1 } , \dots , t _ { k } )$ or $\neg P ( t _ { 1 } , \cdot \dots , t _ { k } )$ , where $P ( t _ { 1 } , \ldots , t _ { k } )$ is an atom.

Syntax. Like in standard FOL, reasoning over the full fragment of $\mathcal { L }$ is undecidable. In particular, we will be interested in learning and reasoning with universally quantified clauses, which form both the knowledge base<sup>2</sup> and the lines in our resolution proofs.

A universally quantified clause $( \mathrm { o r } \forall \mathrm { - c l a u s e } ) ^ { 3 }$ in our setting is a formula of the form: $\forall \vec { x } . L _ { 1 } ( \vec { x } ) \vee \cdot \cdot \cdot \vee L _ { n } ( \vec { x } )$ where each $L _ { i } ( \vec { x } )$ is a literal over a subset of the universally quantified variables ⃗x. Throughout the paper, we will omit the quantifier and simply write such clauses as $\varphi ( \vec { x } )$ or $\varphi ( x )$ (if there is a single variable) when the quantification over free variables is unambiguous from context.

We will refer to elements of ATOMS as propositions, and to ground formulas as propositional formulas. We use $\alpha , \beta , \varphi , \psi$ to denote ground formulas.

Semantics. A L-model M is a {0, 1}-assignment to the elements of ATOMS. Using $\models \mathrm { t o }$ denote satisfaction, the semantics of a formula $\varphi \in { \mathcal { L } }$ is defined inductively, with the following convention:

• Quantification is substitutional over all names in N: Let ⃗x denote the (ordered) tuple of free variables in $\varphi . \ M \vDash$ $\forall \vec { x } . \varphi ( \vec { x } )$ iff ${ \dot { M } } \left| = \varphi ( { \dot { \vec { a } } } ) \right.$ for all $\vec { a } \in \mathbb { N } ^ { k }$ , where $k \stackrel { \cdot } { = } | \vec { x } |$

Definition 1 (Validity of ∀-clause). Let the set ofall models be denoted by M. We say a ∀-clause is valid if it is satisfied by every model in M.

## 2.1 Resolution

We use (first-order) resolution as our underlying proof system. Resolution is a classical proof system that operates over disjunctions of literals via the cut rule:

$$
{ \frac { A \lor L \quad B \lor \neg L } { A \lor B } } ,
$$

optionally augmented with the weakening rule, which allows arbitrary literals to be added to clauses:

$$
{ \frac { A } { A \lor B } } .
$$

These rules form the deductive backbone of our system. In our setting, resolution operates over universally quantified clauses—formulas of the form ∀x $( \psi ( x ) \to \chi ( x ) )$ ), or equivalently, ∀x $( \neg { \psi } ( x ) \lor \chi ( x ) )$ . This lifted formulation allows reasoning to proceed symbolically, without grounding clauses to all possible instantiations. It supports generalization across individuals in an infinite domain and is central to the efficiency and interpretability of our method.

In a general resolution derivation, each derived clause may be used as a premise in multiple later inferences. Consequently, the dependency structure induced by “which earlier lines are used to derive which later lines” forms a directed acyclic graph (DAG). A treelike resolution proof is the special case in which no derived clause is reused: each derived line is used as a premise at most once. Equivalently, the underlying dependency DAG is a tree (rooted at the final contradiction).

Following the structure of the classical DPLL procedure Davis and Putnam (1960); Davis, Logemann, and Loveland (1962), which constructs treelike refutations via variable splits, our method performs clause-level case analysis to build lifted proofs. This analogy extends further: DPLL with bounded memory corresponds to space-bounded treelike proofs, where only s clauses can be held at once, and forgotten clauses must be re-derived Kullmann (1999); Esteban and Toran (2001). Our lifted algorithm follows this´ model, combining case splits with clause reuse under a space constraint.

Importantly, all DPLL-like algorithms (including ours) produce normal resolution proofs:

Definition 2 (Normal). We will say that a resolution proof is normal if in its corresponding DAG:

• All outgoing edges from Cut nodes are directed to Cut nodes.

• The clauses labeling any path to the sinkfrom a Cut node contain literals using every variable along the path.

• A given variable is used in at most one cut step and at most one weakening step along every path from a source to a Cut node.

Let N denote the number of atoms. Let s be the space of a treelike resolution proof, which corresponds to the rank or Horton–Strahler number of its proof tree (Ansotegui, Levy,´ and Manya, 2008). A key structural fact, due to Ehrenfeucht and Haussler (1989), bounds the size and number of such proofs in terms of s and N.

Lemma 1 (Ehrenfeucht and Haussler, 1989). Let k be the number of nodes in a normal space-s treelike resolution proof over N atoms, with $N \ge s \ge 1$ . Let e denote the base ofln. Then:

1. The proof size k is bounded by $\begin{array} { r } { 2 ^ { s } - 1 \le k \le 2 \left( \frac { e N } { s - 1 } \right) ^ { s - 1 } } \end{array}$

2. The number ofdistinct normal space-s treelike proofs that do not use weakening is at most $( 8 N ) ^ { ( e N / ( s - 1 ) ) ^ { s - 1 } }$

Note that Lemma 1 implies the space parameter is logarithmic in the proof size. Consequently, the sample complexity and running time bounds we obtain, with $\begin{array} { r l } { N ^ { s } } & { { } = } \end{array}$ $\overset { \bullet } { N } O ( \log N )$ , are quasipolynomial in N. These combinatorial bounds also play a central role in our sample complexity analysis. They quantify the expressive capacity of boundedspace proofs and guide how much data is needed to safely validate axioms.

## 3 PAC-semantics

We now turn to the learning component of our task. The key difficulty is that induction from data does not support classical, worst-case guarantees: from finitely many samples, we generally cannot certify that a nontrivial first-order statement holds in all interpretations of the language. As a result, conclusions supported by learning must be interpreted probabilistically.

More specifically, we cannot expect learned statements to be valid in the classical (Tarskian) sense, since PAC learning does not guarantee that a formula holds under every possible model. Instead, it guarantees generalization: with high probability over the sampled data, a learned formula $\varphi$ will agree with fresh, unseen examples drawn from the same underlying distribution. This viewpoint is formalized by the PAC semantics of Valiant (2000), which provides a principled, distribution-dependent notion of approximate satisfaction that is weaker than classical truth.

In the classical propositional setting, one typically works with a finite language over n propositional symbols, so models correspond to truth assignments in $\{ 0 , \bar { 1 } \} ^ { n }$ . Our setting, by contrast, involves first-order relational structure and an unbounded collection of ground atoms, so the corresponding space of interpretations is inherently infinitary. Following Belle and Juba (2019), we adopt PAC semantics in a way that evaluates formulas relative to a data-generating distribution and a partial observation process:

• We define a σ-algebra $\Sigma _ { \mathcal { M } }$ over the set of all models $\mathcal { M } ,$ , representing measurable collections of (possibly infinitary) events.

• We introduce a probability measure $\mathrm { P r } : \Sigma _ { \mathcal { M } } \to [ 0 , 1 ]$ defining a distribution over models.

This allows us to define a notion of $( 1 - \varepsilon )$ -validity, capturing the idea that a formula holds in most models drawn from the distribution.

Definition 3 (PAC Validity of a Formula). Let $[ [ \varphi ] ]$ denote the set ofmodels satisfying a closedformula $\varphi \in { \mathcal { L } } ;$

$$
\mathbb { F } \mathbb { H } : = \{ M \in \mathcal { M } \mid M \mid = \varphi \} .
$$

Given a probability distribution $\mathcal { P }$ over $\mathcal { M } ,$ , we say that $\varphi$ is $( 1 - \varepsilon )$ -valid with respect to P if

$$
\operatorname* { P r } _ { M \sim \mathcal { P } } ( \left[ \varphi \right] ) \ge 1 - \varepsilon .
$$

$H \varepsilon = 0 ,$ , we say that $\varphi$ is perfectly valid.

Example 1. Let $\varphi ( x ) : \forall x S q u i r r e l ( x )  G r e y ( x )$ assert that all squirrels in aforest are grey. Suppose this generally $h o l d s { \it - e . g . }$ , due to genetic inheritance—but rare mutations occasionally produce yellow squirrels. So across all possible worlds, most worlds contain no yellow squirrels at all; but a small fraction may include one or more. In this case, $\varphi$ is not classically valid, but can be $( 1 - \varepsilon ) – a l i d f o r \varepsilon = 0 . 1 \mathrm { : }$

$$
\operatorname* { P r } _ { M \sim \mathcal { P } } [ M \vdash \varphi ] \ge 0 . 9 .
$$

In classical logic, validity is most often used to establish entailment—i.e., whether a knowledge base entails a given query. PAC validity captures that $\varphi$ holds in most models under ${ \mathcal { P } } _ { \mathrm { { : } } }$ enabling probabilistic reasoning under uncertainty. In this PAC setting, one may naturally ask: how does this notion of inference translate when formulas are only approximately valid? As observed by Juba (2013) in the propositional setting, the union bound provides a natural analogue of classical reasoning under PAC semantics. That is, if a set of formulas are each individually $( 1 - \varepsilon _ { i } )$ -valid with respect to ${ \mathcal P } ,$ and their conjunction classically entails a ground formula $\varphi ,$ then the query is also PAC-valid with controlled error.

The result also holds for lifted queries such as ∀-clause since PAC validity of a ∀-clause requires satisfaction under all groundings, and classical entailment preserves validity under substitution.

Proposition 1. Let $\psi _ { 1 } , \ldots , \psi _ { k }$ be ∀-clauses such that each $\psi _ { i } i s \left( 1 - \varepsilon _ { i } \right)$ -valid under a common distribution ${ \mathcal P } ,$ for some $\varepsilon _ { i } \in [ 0 , 1 ]$ . Suppose that $\{ \psi _ { 1 } , \ldots , \psi _ { k } \} \ \models \varphi ,$ where $\varphi$ is either a groundformula or a ∀-clause. Then φ is $( 1 - \varepsilon ^ { \prime } )$ valid under ${ \mathcal P } ,$ , where $\varepsilon ^ { \prime } = \textstyle \sum _ { i } \varepsilon _ { i }$

## 4 Observability and Learnability

We first recall the idealized setting of Khardon and Roth (1997), in which examples are fully observed. That is, each sample specifies a complete truth assignment to all ground atoms in the language, so empirical frequencies can be used to estimate the probability of violations. In this case, falsity of a universal clause admits finite witnesses, and sample complexity follows from a simple concentration argument.

Observation 1 (Full Observability). Let $\varphi$ be a fixed universally quantified clause evaluated on complete examples drawn i.i.d.from a distribution P. If

$$
m \ge \frac { 1 } { \varepsilon } \ln \left( \frac { 1 } { \delta } \right)
$$

and no violation is observed in m examples, then $\varphi$ is $( 1 -$ ε)-valid with probability at least $1 - \delta .$

Proof. Let $p$ be the probability that a random model drawn from $\mathcal { P }$ violates the clause $\varphi .$ . We aim to show that if $p > \varepsilon ,$ the probability of observing no violations in m independent examples is at most $\delta .$

Since the examples $M ^ { ( 1 ) } , \ldots , M ^ { ( m ) }$ are drawn i.i.d., the probability that $\varphi$ is satisfied by all m examples is given by:

$$
\mathbb { P } ( \forall i , M ^ { ( i ) } \models \varphi ) = ( 1 - p ) ^ { m }
$$

Assuming the clause is not $( 1 - \varepsilon )$ -valid $( \mathrm { i } . \mathrm { e } . , p > \varepsilon )$ , we can upper-bound this probability:

$$
( 1 - p ) ^ { m } < ( 1 - \varepsilon ) ^ { m }
$$

Using the standard inequality $1 - x \leq e ^ { - x }$ for $x \in [ 0 , 1 ]$ we obtain:

$$
( 1 - \varepsilon ) ^ { m } \leq e ^ { - m \varepsilon }
$$

To ensure the probability of failing to observe a violation is bounded by our confidence parameter δ, we require:

$$
e ^ { - m \varepsilon } \leq \delta
$$

Taking the natural logarithm of both sides and solving for m, we obtain that if $m \geq \frac { 1 } { \varepsilon } \ln \bigl ( \frac { 1 } { \delta } \bigr )$ and no violation is observed, $\varphi$ is $( 1 - \varepsilon ) { \mathrm { - v a l i d } }$ with probability at least $1 - \delta$ □

## 4.1 Observation Model

The fully observed setting above is a simple starting point, but it is rarely realistic. In our formalization, a complete example specifies a truth value for every ground atom, whereas real data typically reveals only a small, task-dependent subset of predicates and ground instances. Many facts of interest are latent, costly to measure, or not directly observable at all. This incompleteness is precisely what makes both induction from data and deduction from background knowledge necessary. Handling corrupted or adversarially perturbed observations is also important, but we defer that extension to future work.

Hence, let us treat partial observability as an explicit part of the model: the learner observes only a masked view of an underlying (unobserved) L-model.

Definition 4 (Partial Model). A partial model ρ is afunction $\rho : A T O M S  \{ 1 , 0 , * \}$ , where 1 and 0 indicate truth and falsity, respectively, and ∗ denotes an unobserved (masked) value.

We say that a partial model $\rho$ is consistent with a $. f u l l .$ model $\dot { M } \in \mathcal { M }$ if, for all $A \in A T O M S$ , whenever $\rho ( A ) \neq * _ { \mathrm { ; } }$ it holds that $\rho ( A ) = M ( A )$ . Let $\mathcal { N }$ denote the set of all partial models.

Definition 5 (Masking). A mask is a function $\theta : \mathcal { M } \to \mathcal { N }$ that maps full models to partial models. It must satisfy the property that for all $M \in { \mathcal { M } } ,$ , the resulting partial model $\mathsf { \bar { \theta } } ( \bar { M } )$ is consistent with M, and only finitely many atoms are mapped to {0, 1}; all others are masked as ∗.

A masking process Θ is a mask-valued random vari-$a b l e { \mathrm { - } } i . e .$ , a distribution over masking functions. Applying Θ to a distribution P overfull models induces a distribution Θ(P) over partial models.

## 4.2 Necessary Condition

A central difficulty is that, under unrestricted masking, violations of a universally quantified clause may be systematically concealed. In the extreme, the observation process can make truth and falsity observationally indistinguishable, so that even refutation is impossible from data alone.

Observation 2 (Zero Observability). Fix a universally quantified clause ψ (e.g., ∀x ψ(x)). Suppose the masking process Θ has the following property: for every full model M,

$$
M \Vdash \lnot \psi \quad \Longrightarrow \quad \operatorname* { P r } \left[ \rho \vdash \lnot \psi \enspace | \enspace \rho = \Theta ( M ) \right] = 0 ,
$$

i.e., whenever ψ is false in M, the induced partial model never reveals a detectable violation ofψ. Then samplesfrom $\Theta ( \mathcal P )$ cannot be used to upper-bound the violation probability ofψ under P: for any $\epsilon \in ( 0 , 1 )$ there exist distributions $\mathcal { P } _ { 0 } , \mathcal { P } _ { 1 }$ such that

$$
\operatorname* { P r } _ { M \sim \mathcal { P } _ { 0 } } [ M \vdash \neg \psi ] = 0 , \qquad \operatorname* { P r } _ { M \sim \mathcal { P } _ { 1 } } [ M \vdash \neg \psi ] \ge \epsilon ,
$$

yet $\Theta ( \mathcal { P } _ { 0 } )$ and $\Theta ( \mathcal { P } _ { 1 } )$ are identical. In particular, no algorithm observing only $\rho \sim \Theta ( \mathcal { P } )$ can distinguish whether $\psi$ is always true orfrequentlyfalse.

Observation 2 motivates a minimal restriction on the masking process: false formulas must be detectable with non-negligible probability. This definition extends the finite version from Michael (2010), which essentially only considered the special case of supervised learning, corresponding to formulas $\varphi ( \vec { x } , b )$ of the form $f ( { \vec { x } } ) = { \bar { b } }$ for some (simple) formula $f ,$ , where $\vec { x }$ and b are propositional. Here, we extend it to more general formulas in a first-order relational language.

Definition 6 (Bounded Concealment). Let C be a set of (lifted) formulas and $\mathcal { P }$ a distribution over L-models. A masking process Θ is said to be at most $( 1 - \eta )$ -concealing with respect to C and P if

$$
\forall \varphi \in { \mathcal { C } } , \operatorname* { P r } _ { M \sim { \mathcal { P } } , \rho = \Theta ( M ) } [ \rho \vdash \neg \varphi | M \vdash \neg \varphi ] \geq \eta .
$$

That $i s ,$ when a formula φ is false in $\mathbf { \nabla } a f a l l \mathbf { \nabla }$ model M, there is at least η probability that its falsity is detectable from the corresponding partial model $\rho = \Theta ( M )$ .

Example 2 (Bounded Concealment Over a Clause Set). Let C be a set of universally quantified clauses describing regularities in aforest domain:

$$
\varphi _ { 1 } : \forall x S q u i r r e l ( x )  F u r r y ( x ) ,
$$

$$
\varphi _ { 2 } : \forall x S q u i r r e l ( x )  G r e y ( x ) ,
$$

$$
\varphi _ { 3 } : \forall x S q u i r r e l ( x )  S m a l l ( x ) .
$$

These formulas are mostly true in the underlying distribution $\mathcal { P }$ due to genetic inheritance, but may be occasionally violated due to mutation.

Now suppose an η fraction ofthe population, chosen uniformly at random, has all predicates observed. Then Θ is $( 1 - \eta )$ -concealing w.r.t. C if, whenever a clause $\varphi _ { i }$ is violated in a full model M, there is at least η probability that the violation is visible in the partial model $\rho = \Theta ( M )$

Thus, even under limited observability, clause violations can be detected with bounded probability.

The bounded concealment assumption justifies lifted reasoning: we are allowed to assign high confidence to universally quantified formulas when no counterexamples are observed. The following “discounting” theorem of Michael (2010) makes this precise; although Michael’s theorem was stated and proved for the special case of supervised learning with propositional variables as described above, the proof carries over.

Theorem 1. (Michael, 2010, Theorem 2.2) For any distribution P over models, masking process Θ, and class C of formulas, if Θ is at most $( 1 - \eta )$ -concealing with respect to C and ${ \mathcal { P } } _ { : }$ , and $\varphi \in { \mathcal { C } }$ is not $( 1 - { \dot { \varepsilon } } ) – u l i d ,$ then

$$
\operatorname* { P r } _ { \rho \sim \Theta ( \mathcal { P } ) } [ \rho \vdash \neg \varphi ] \geq \eta \varepsilon .
$$

Conversely, if Θ is not at most $( 1 - \eta )$ -concealing with respect to $\mathcal { C }$ and P, then there exists $\varphi \in { \mathcal { C } }$ such that

$$
\operatorname* { P r } _ { \rho \sim \Theta ( \mathcal { P } ) } [ \rho \vdash \neg \varphi ] \leq \eta \operatorname* { P r } _ { M \sim \mathcal { P } } [ M \vdash \neg \varphi ] ,
$$

where, note, $\varphi i s ( 1 - \mathrm { P r } _ { M \sim \mathcal { P } } [ M \vert = \neg \varphi ] )  – \nu a l i d .$

Therefore, a bounded concealment condition is necessary for learning universal clauses. In particular, we note that it almost immediately gives a lower bound on the number of examples required for the more basic task of certifying $( 1 - \epsilon )$ -validity (ignoring running time) for varying values of the concealment parameter η:

Corollary 1. Suppose that some function A, given any formula $\varphi$ in the class offormulas C and m $( \varphi , \eta , \epsilon , \delta )$ partial examples from any distribution on masked examples $\Theta ( \mathcal P )$ for an at most $( 1 - \eta )$ concealing Θ with probability $1 - \delta \cdot$

• Ifφ is perfectly valid with respect to ${ \mathcal P } ,$ , outputs “accept”.

• If φ is not (1−ϵ)-valid with respect to P, outputs “reject”.

Suppose that for some distribution ${ \mathcal { P } } ^ { * }$ and $( 1 ~ - ~ \eta )$ concealing masking process $\Theta ^ { * } , \Theta ^ { * }$ is not at most $( 1 - \eta -$ $\gamma )$ -concealing with respect to C and ${ \mathcal { P } } ^ { * }$ for any $\gamma > 0$ . Then for all $\varphi , \delta ,$ and ϵ such that $\eta \epsilon \leq 1 / 2 ,$

$$
m ( \varphi , \eta , \epsilon , \delta ) \geq \frac { \ln ( 1 / 2 \delta ) } { 2 \eta \epsilon } .
$$

Proof. We first observe that for any $\varphi \in { \mathcal { C } }$ , there is a distribution $\tilde { \mathcal P }$ and masking process $\tilde { \Theta }$ such that

• $\varphi$ is perfectly valid on $\tilde { \mathcal P }$ and

$\tilde { \Theta } ( \tilde { \mathcal { P } } )$ is the same distribution on partial examples $\rho$ as $\Theta ^ { * } ( \dot { \mathcal { P } } ^ { * } )$ conditioned on the partial example $\rho$ satisfying $\rho \nvDash \neg \varphi .$

Now, by the first correctness condition, for any ϵ and $\delta ,$ when A is given $m ( \varphi , \eta , \epsilon , \delta )$ examples from $\tilde { \Theta } ( \dot { \mathcal { P } } )$ , since $\varphi$ is perfectly valid on $\mathcal { P }$ , it must accept with probability $1 - \delta$

Now, let $\gamma > 0$ be given and $\varphi$ be a 1 − ϵ-valid formula for $\Theta ^ { * }$ and ${ \mathcal { P } } ^ { * }$ satisfying the second condition of Theorem 1, where $\Theta ^ { * }$ is $( 1 - \eta )$ -concealing but not at most $\left( 1 - \eta - \gamma \right)$ concealing. Observe that in each example, the probability that $\rho \not \vdash \neg \varphi$ is at least $1 - ( \eta + \gamma ) \epsilon$ , and hence, the probability that $m ( \varphi , \eta , \epsilon - \gamma , \delta )$ independent examples all satisfy $\rho \not \vdash \neg \varphi$ is at least $( 1 - ( \eta + \gamma ) \epsilon ) ^ { m ( \varphi , \eta , \epsilon - \gamma , \delta ) }$ . In this case, these are $m ( \varphi , \eta , \epsilon - \gamma , \delta )$ examples drawn from $\Theta ^ { * } ( \mathcal { P } ^ { * } )$ conditioned on $\rho \nvDash \neg \varphi ,$ which is identical to $\tilde { \Theta } ( \tilde { \mathcal { P } } )$ by construction. So by a union bound, the probability that $A$ does not output accept on the examples from $\Theta ^ { * } ( \bar { \mathcal { P } } ^ { * } )$ is at most

$$
\delta + 1 - ( 1 - ( \eta + \gamma ) \epsilon ) ^ { m ( \varphi , \eta , \epsilon - \gamma , \delta ) } .
$$

But, by our choice of $\varphi$ which is not $1 - \epsilon + \gamma -$ valid with respect to ${ \mathcal { P } } ^ { * }$ , where $\Theta ^ { * }$ is $( 1 - \eta )$ -concealing, the second correctness condition for $A$ requires that this probability is at least $1 - \delta$ . Since $1 - x + \dot { x } ^ { 2 } \ge e ^ { - x }$ for $x \ge 0$ , and $x - x ^ { 2 } \geq x / 2$ for $x \leq 1 / 2$ we have

$$
\begin{array} { c } { 1 - \delta \le \delta + 1 - ( 1 - ( \eta + \gamma ) \epsilon ) ^ { m ( \varphi , \eta , \epsilon - \gamma , \delta ) } ) } \\ { \Rightarrow 2 \delta \ge e ^ { - 2 ( \eta + \gamma ) \epsilon \cdot m ( \varphi , \eta , \epsilon - \gamma , \delta ) } } \\ { m ( \varphi , \eta , \epsilon - \gamma , \delta ) \ge \displaystyle \frac { \ln 1 / 2 \delta } { 2 ( \eta + \gamma ) \epsilon } . } \end{array}
$$

Note that without loss of generality, $m ( \varphi , \eta , \epsilon , \delta )$ is monotonically decreasing with respect to ϵ. Since $\gamma > 0$ was arbitrary, letting $\gamma  0$ yields the claimed bound. □

Indeed, as $\eta  0 , m  \infty$ , recovering Observation 2. More generally, if a method purports to certify $1 - \epsilon$ validity with confidence $1 - \delta$ from m examples, then solving for η in the bound, the method implicitly assumes that the masking is at most $( 1 - \eta )$ -concealing for the corresponding value. We remark that our method will asymptotically match this inherent $\frac { \log { 1 / \delta } } { \eta \epsilon }$ dependence on $\eta , \epsilon ,$ and δ in the number of examples required (and hence also running time). We remark that we are solving a slightly richer task than assumed in our statement here, which requires us to output a proof of a query from a given knowledge base and some $( 1 - \epsilon )$ -valid formulas, which will indeed require some additional dependence on the complexity of those formulas in the number of examples used. But this is indeed a strictly richer formulation: in the special case where the knowledge base is empty, we can construct an algorithm that just outputs “accept” when a proof is found, solving the above task formulation. It is therefore bound by the same requirements.

## 5 Query-Driven Algorithm

We now present an algorithm for lifted query-driven reasoning under partial observability. Given a universally quantified clause $\varphi ,$ a symbolic knowledge base K, and a set of partial models (i.e., masked observations) as data, the algorithm either returns a lifted resolution proof of $\varphi$ or soundly rejects it. The proof, if found, may use both symbolic axioms from K and new clauses validated statistically from the data.

The algorithm performs a space-bounded proof search, constructing lifted resolution trees with bounded memory. It leverages the bounded concealment assumption to learn new axioms from data and interleaves them with symbolic reasoning to justify universal claims efficiently.

The search proceeds in three phases:

• Validation: If no observed partial model falsifies any grounding of $\varphi ,$ we accept $\varphi$ as PAC-valid and return a proof asserting it.

• Weakening: $\operatorname { I f } \varphi$ is subsumed by a clause in $\kappa$ via vari able renaming, we derive it using weakening.

• Recursive Resolution: Otherwise, we branch on pivot literals drawn from K to recursively construct a lifted proof of $\varphi ,$ bounded by a space parameter s.

To make this procedure precise, we now define two core concepts used in the resolution step: subsumptive weakening and pivot literals.

Definition 7 (Subsumptive Weakening). Let $\alpha ( y _ { 1 } , \ldots , y _ { \ell } )$ be a lifted clause in ${ \dot { \kappa } } ,$ and let $\varphi ( x _ { 1 } , \ldots , x _ { k } )$ be the lifted clause query. We say α subsumes $\varphi$ by weakening if there exists a (not necessarily injective) renaming

$$
\sigma : \{ y _ { 1 } , \dots , y _ { \ell } \} \implies \{ x _ { 1 } , \dots , x _ { k } \}
$$

such thatfor each literal $L o f \alpha ,$ the renamed literal

$$
L \big ( \sigma ( y _ { 1 } ) , \dots , \sigma ( y _ { \ell } ) \big )
$$

appears among the disjuncts of φ.

Example 3. Let $\alpha ( x , y ) = A ( x ) \lor B ( y )$ and let $\varphi ( x ) =$ $A ( x ) \{ \bar { v } B ( x ) \vee C ( x )$ . Then α subsumes φ by weakening via $\sigma ( x ) = \sigma ( y ) = x$ . This illustrates how non-injective substitutions allow us to derive special cases from more general rules in the knowledge base, enabling clause reuse even when variable bindings collapse.

Definition 8 (Pivot Literals for φ). Let $\varphi ( x _ { 1 } , \ldots , x _ { k } )$ be the query with k universally quantified variables. For any literal $L ( \vec { y } )$ containing universally quantified variables occurring in $\kappa ,$ each literal of the form $L ( \sigma ( \vec { y } ) )$ for some renaming $\sigma : \{ y _ { 1 } , \dots , y _ { \ell } \} \longrightarrow \{ x _ { 1 } , \dots , x _ { k } \}$ is called a pivot literal over $\varphi .$

Example 4. Suppose K contains the clause $\forall z . A ( z )$ , and the query is $\varphi ( x , y ) = B ( x ) \vee C ( y )$ . Then both $A ( x )$ and $A ( y )$ are considered pivot literals $f o r \ \varphi ,$ corresponding to renamings $\sigma ( z ) = x$ and $\sigma ( z ) \ = \ y ,$ respectively. Even though they originate from the same clause in $\kappa ,$ , they are treated as distinct literals because x and y are different variables in the scope of $\varphi .$

We now demonstrate that using subsumptive weakening and pivot literals is sufficient to derive a sound and complete algorithm for query validation. We formalize this procedure in the LIFTEDLEARN+SEARCH algorithm.

Theorem 2. Let φ be a ∀-clause query, and K be a knowledge base consisting of a finite set of ∀-clauses. N is the number of different atoms appeared in the knowledge base. Let P be a distribution over models, and let Θ be a $( 1 - \eta )$ -concealing masking process with respect to CNFs up to size $2 ( e N / s - 1 ) ^ { s - \bigcup }$ for the distribution $\mathcal { P } .$ . Then on a sample of size m $\begin{array} { r } { \in \mathrm { ~ \Theta ~ } \Theta \left( \frac { N ^ { s - 1 } \log N + \log \frac { 1 } { \delta } } { \eta \varepsilon } \right) } \end{array}$ , and with running time $\begin{array} { r } { O \left( \frac { N ^ { 2 ( s - 1 ) } | \varphi | } { \eta \varepsilon } ( N ^ { s - 1 } \log N + \log \frac { 1 } { \delta } ) \right) } \end{array}$ the algorithm withfixed parameter s satisfies:

Algorithm 1: LIFTEDLEARN+SEARCH   
Input: Knowledge base K, space bound $s \geq 1$   
query clause $\varphi ( x _ { 1 } , \ldots , x _ { k } )$   
Partial models $\rho ^ { ( 1 ) } , \ldots , \rho ^ { ( m ) }$   
Output: Lifted resolution proof of $\varphi$ or None   
if ∀i, $\rho ^ { ( i ) } \not \Vdash \lnot \varphi$ then   
return proof asserting $\varphi$ from statistical   
validation   
foreach clause $\alpha ( \vec { y } ) \in \kappa$ do   
if there exists substitution $\sigma : \{ y _ { j } \}  \{ x _ { i } \}$ s.t.   
$\sigma ( \alpha ) \subseteq \varphi$ then   
return weakening derivation of $\varphi$ from $\sigma ( \alpha )$   
if s > 1 then   
foreach pivot or ground literal ℓ relevant to $\varphi$   
from K do   
if $\ell \not \in \varphi$ and $\neg \ell \notin \varphi$ then   
$\dot { \Pi } ^ { + } $   
LIFTEDLEARN+SEARCH(K, s−1, φ ∨   
$\ell , \ \{ \rho ^ { ( i ) } : \ell | _ { \rho ^ { ( i ) } } \neq 1 \} )$   
if Π<sup>+</sup> ̸= None then   
$\Pi ^ { - }  \mathrm { L I F T - }$   
EDLEARN+SEARCH $( \mathcal { K } , s , \varphi \vee$   
$\neg \ell , \{ \rho ^ { ( i ) } : \ell | _ { \rho ^ { ( i ) } } \neq 0 \} )$   
if Π<sup>−</sup> ̸= None then   
return resolution proof   
combining Π<sup>+</sup> and Π<sup>−</sup> on pivot   
ℓ   
return None

• (soundness) $I f \left[ K \Rightarrow \varphi \right]$ is not $( 1 - \varepsilon )$ -valid, with probability 1 − δ, Algorithm 1 returns None.

• (completeness) If there exists some perfectly valid CNF H for which there is a space-s treelike resolution proof of φ from $\kappa \wedge \mathscr { H } ,$ then Algorithm 1 returns a proof of φ from $\kappa \wedge \mathcal { H } _ { 0 }$ for some CNF H<sub>0</sub> of size $O ( N ^ { s - 1 } )$ that is (1 − ε)-valid with respect to D with probability 1 − δ.

Proof. (1) Soundness. By definition, Algorithm 1 produces a space-s normal treelike resolution proof. Let $\mathcal { H } _ { \mathrm { b a d } }$ be the CNF formed by the conjunction of all such leaf clauses in the proof. By Lemma 1 (Part 1), this CNF contains at most $2 ( e \bar { N } / s - \bar { 1 } ) ^ { s - 1 }$ clauses, hence bounded concealment applies. Therefore, Theorem 1 guarantees that Θ will falsify this CNF with probability at least $\eta \varepsilon$ if the CNF is not $1 - \varepsilon$ valid.

We have m draws of partial models in total. Using the inequality $1 - x \leq e ^ { - x }$ for $x \in ( 0 , 1 )$ , we obtain:

$$
( 1 - \eta \varepsilon ) ^ { m } \leq e ^ { - m \eta \varepsilon } .
$$

Now choose

$$
m = \frac { N ^ { s - 1 } \log N + \log ( 1 / \delta ) } { \eta \varepsilon } .
$$

Then:

$$
\begin{array} { r c l } { { } } & { { } } & { { e ^ { - m \eta \varepsilon } = e ^ { - ( N ^ { s - 1 } \log N + \log ( 1 / \delta ) ) } } } \\ { { } } & { { } } & { { } } \\ { { } } & { { } } & { { = e ^ { - N ^ { s - 1 } \log N } \cdot e ^ { - \log ( 1 / \delta ) } } } \\ { { } } & { { } } & { { } } \\ { { } } & { { } } & { { = N ^ { - N ^ { s - 1 } } \cdot \delta . } } \end{array}
$$

Hence $\mathrm { P r } [ \mathcal { H } _ { \mathrm { b a d } }$ is consistent with all samples] $\le \quad \delta$ $N ^ { - \Omega ( N ^ { s - 1 } ) }$

Since weakening is not used in the clauses that are being asserted during learning, we do not need to bound the total number of distinct proofs. Applying a union bound over all such CNFs, the probability that any non- $\left( 1 - \varepsilon \right)$ -valid CNF is consistent with the entire sample is at most δ. Hence, with probability at least $1 - \delta .$ the proof it returns has all the leaves at least $1 - \varepsilon$ valid. But if the query is itself less than $1 - \varepsilon$ valid, then such proof cannot exist and the algorithm returns None, establishing the soundness claim.

(2) Completeness. It remains to show that Algorithm 1 will indeed find a proof when a 1-valid space-s treelike proof exists. Consider the (potentially exponential-size) formula $\widetilde { H }$ consisting of all clauses that are consistent with the obscured scenes $\rho ^ { ( \mathsf { 1 } ) } , \ldots , \rho ^ { ( m ) }$ . Note that weakening is not necessary for clauses from ${ \cal \widetilde { H } } ,$ as any clause derivable by weakening is already explicitly present in $\widetilde { H }$

Algorithm 1 performs a systematic search for a normal space-s treelike resolution proof, branching on literals that appear in K. We claim that for any original 1-valid proof, there exists an equivalent one (possibly with new asserted clauses) that applies cuts only to pivot literals drawn from ground instances of K.

If such a literal enables both branches of a resolution step to be verified using the data, then the resolved clause can itself be verified directly, without the need to branch on that literal. On the other hand, if at least one of the branches cannot be verified (i.e., requires weakening), then introducing a literal not in K is not helpful, because the resulting clause would not be derivable from $\kappa \wedge \widetilde H$ and would not be verified by the data.

Moreover, we do not need to introduce new variables when branching on clauses from $\kappa .$ . Although such clauses may contain universally quantified variables, renaming them to match variables already in the query’s scope merely produces a special case of the original clause. Since universally quantified statements are closed under substitution, this specialization remains entailed by the knowledge base and does not require augmenting the query with fresh variables.

Therefore, even if the original 1-valid proof uses literals outside the hypothesis, the algorithm can always find an equivalent proof that uses only literals from K. This justifies restricting branching literals to those in the knowledge base, without loss of completeness in the class of verifiable proofs.

![](images/c3f5e3b1950b802b8d5a08a66242be0269bece8f3b657403178ca8760238a184.jpg)  
Figure 1: Lifted resolution proof $\mathsf { o f } \neg \mathsf { S q u i r r e l } ( x ) \vee \mathsf { G r e y } ( x )$ using both knowledge base and data.

Furthermore, the standard analysis of space-bounded treelike resolution (e.g., Kullmann (1999)) guarantees that the algorithm runs in time $O ( m | \varphi | N ^ { 2 ( s - 1 ) } )$ on a sample of size m. When a 1-valid space-s treelike proof exists, it is consistent with all samples with probability 1.

Thus, whenever Algorithm 1 terminates early, it returns a proof that is consistent with the examples, and by soundness, the proof is (1 − ε)-valid. Moreover, if the algorithm explores the literals used in the known 1-valid proof (though it may succeed even earlier), it is guaranteed to find a valid derivation. Hence, the algorithm is complete for the class of 1-valid space-s treelike proofs. □

Example 5. Consider the query:

$$
\varphi = \forall x \neg S q u i r r e . l ( x ) \lor G r e y ( x )
$$

and the knowledge base:

$$
\begin{array} { r } { \mathcal { K } = \{ \forall x \neg S q u i r r e l ( x ) \vee T r e e D w \in \mathcal { I } \mathcal { I } \in r ( x ) ,  } \\ { \qquad \forall x \neg S q u i r r e \mathcal { I } ( x ) \vee N u t E a t e r ( x ) \} \qquad } \end{array}
$$

which states that all squirrels live in trees and eat nuts. Suppose we observe partial scenes where all tree-dwellers are grey. This validates:

$$
\forall x \neg T r e e D w e . l . l e x ( x ) \lor G r e y ( x )
$$

from data, but falsifies ¬NutEater(x) ∨ Grey(x).

When the algorithm selects TreeDweller(x) as a pivot literal, resolution succeeds using one clausefrom K and one statistically validated clause, constructing a lifted proof of $\varphi ,$ as illustrated in Figure 1.

## 5.1 Choosing the Space Bound

Our algorithm requires the proof’s space bound s as input, which controls the tradeoff between the expressiveness of the proof system and the computational resources required for proof search. Larger values of s permit deeper and more complex derivations, potentially increasing completeness, but also lead to exponential growth in the number of candidate proofs (see Lemma 1). We note that space-bounded treelike resolution has another interpretation as an extension of the well-known unit-propagation inference rule (where $s = 2 )$ with a rule known as failed-literal (Ansotegui, Levy,´ and Manya, 2008): we temporarily assert a literal ℓ as a hypothesis, and if the formula can be refuted, we infer ¬ℓ. Then s corresponds to a bound on the number of literals that may be simultaneously asserted while invoking the failedliteral rule; algorithms for automating space-bounded treelike resolution essentially follow the strategy implicitly described by this characterization, searching for literals on which to invoke the failed-literal rule.

In practice, s should be chosen based on available memory and computational budget, as well as the anticipated complexity of the domain. Ansotegui, Levy, and Manya ´ (2008) also examined the value of s in many industrial instances of propositional satisfiability, and found that it is significantly smaller than for random instances in the hard regime at the phase transition, suggesting that it may capture what makes real-world instances easy.

Moreover, our algorithm is anytime: if no proof is found for a given s, the parameter can be incrementally increased, enabling flexible exploration of the proof space.

## 6 Related Work

Our work targets an integration of symbolic reasoning with statistical learning—two paradigms that have largely developed in parallel. The closest line to ours begins with De Raedt and Bruynooghe (1993), who introduced a theory of clausal discovery divorced from any underlying data distribution. Soon afterwards, Khardon and Roth (1997) formalized the “learning to reason” framework, establishing a principled synthesis of learning and proof, but under a strong restriction: the learner reasons in a fully observed setting. Subsequently, Valiant (2000) proposed PAC-Semantics, relaxing classical validity to accommodate imperfect rules as long as they hold with high probability under the data distribution. Building on these foundations, Juba (2013) introduced implicit learning to reason, which concentrates computational effort on what is needed for a particular query, yielding large savings in time and space. Later work extended this perspective toward limited forms of first-order reasoning (Belle and Juba, 2019), as discussed in the Introduction. More recently, Juba (2019) developed query-driven learning to reason, strengthening the implicit approach by explicitly identifying the supporting proof structure for a given query. This explicit proof output is especially important in settings that require interpretability, human oversight, or supervised refinement. Nevertheless, this work was primarily developed in the propositional setting, and does not address our challenge in lifted proofs for universally quantified queries under partial observability.

Outside the “learning to reason” line, several other research programs are conceptually related but do not focus on efficiency and soundness or completeness guarantees as we do. We give an overall review as follows:

Statistical Relational Learning (SRL). SRL frameworks build probabilistic models over relational domains in order to support uncertainty-aware inference. Representative formalisms include Bayesian Logic Programs (Kersting and De Raedt, 2007), Markov Logic Networks (Richardson and Domingos, 2006), and probabilistic logic programming (De Raedt, Kimmig, and Toivonen, 2007), which traditionally compiled to or were parameterized by graphical models, and more recent approaches have used a Weighted First-Order Model Counting (WFOMC) formulation for inference (Braz, Amir, and Roth, 2005; Van Den Broeck et al., 2011; Gogate and Domingos, 2016). While lifted probabilistic inference (Poole, 2003; Kersting, 2012; Van den Broeck et al., 2021) exploits symmetries to improve scalability at inference time, two core challenges remain during learning in particular. First, structure learning is computationally difficult: for expressive SRL formalisms, it is typically combinatorial and intractable (Chickering, 1996), with practical methods relying on restrictive hypothesis classes and heuristic search (Kok and Domingos, 2005). Second, handling missing data is commonly approached via maximum-likelihood or MAP objectives with latent variables, for which expectation–maximization (EM) (Meert, Struyf, and Blockeel, 2008) and related iterative procedures are standard. But these methods generally offer only weak guarantees: they need not find a global optimum, and even convergence to a stationary point can be slow or sensitive to initialization (Koller and Friedman, 2009, Section 19.4). As a result, the end-to-end pipeline is hard to scale efficiently, and its outputs are not accompanied by proof-style certificates or optimality guarantees.

Inductive Logic Programming (ILP). ILP systems (Muggleton and De Raedt, 1994) aim to induce logical rules from examples consistent with background knowledge, without explicitly modeling uncertainty. While classical systems like Aleph (Srinivasan, 2001) and Progol (Muggleton, 1995) are effective at rule generalization, they are not designed for validation or query answering with a distribution in mind. The most closely related ILP work is that of De Raedt and Dzeroski (1994), which learns smallˇ first-order clauses with PAC learning guarantees relative to a finite model, rather than the infinite model we consider, where such guarantees are possible only under boundedconcealment assumptions. Additionally, extensions to probabilistic ILP (De Raedt and Kersting, 2008) inherit the limitations of SRL, including the challenges with structure learning and handling incomplete observations.

Neuro-symbolic approaches. Neuro-symbolic (N-S.) methods combine neural architectures with symbolic logic, often by embedding predicates into continuous representations or approximating inference steps. Notable examples include DeepStochLog (Winters et al., 2022), Lifted Relational Neural Networks (Sourek et al., 2018), and Neural Logic Machines (Dong et al., 2019). Unlike our approach, these systems enable learning from weak supervision but do not produce interpretable proofs or offer formal correctness guarantees. They also do not address the verification of universally quantified queries or model partial observability with principled assumptions.

## 7 Conclusion and Future Work

In this work, we introduce the first query-driven learning to reason framework that provides simultaneous efficiency and correctness guarantees at the lifted level. Our approach bridges symbolic reasoning and statistical learning by enabling universally quantified clauses to be validated from partial observations.

Several directions remain open. First, we restrict attention to Boolean domains, which aligns naturally with resolution but leaves open extensions to richer lifted settings, such as lifted real linear arithmetic (Kersting, Mladenov, and Tokmakov, 2017) (or quadratic (Mladenov, Kleinhans, and Kersting, 2017)) or probabilistic relational logics (Ge, Juba, and Nilsson, 2025), where the expressions involve lifted families of real-valued variables. In a closely related vein, our work only considers a simple kind of masking where attributes are simply present or missing. If we consider the extension to real-valued expressions, then we may correspondingly consider other forms of uncertainty in our partial examples, such as intervals, along the lines of Rader et al. (2021) who considered implicit learning in the propositional setting. It would be interesting to extend query-driven explicit learning to handle such partial information models.

Second, we only guarantee completeness when there is a perfectly valid knowledge base that entails the query – even one counterexample rules out a prospective clause from inclusion in our learned knowledge – which raises the question of how we can handle noisy or imperfect data. While one can relax our algorithm by allowing a small fraction of falsifying examples before rejecting clauses, in general this only yields a bound on the soundness of the query that scales with the number of learned clauses (using Proposition 1); can we obtain a tighter guarantee?

Finally, it would be fruitful to develop practical applications. Promising application domains include knowledge extraction from structured or semi-structured sources, as well as planning tasks where lifted reasoning about actions and goals can benefit from integrating learned rules and partial observations.

## References

Ansotegui, C.; Levy, J.; and Manya, F. 2008. Measur-´ ing the hardness of sat instances. In Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence, 222–228.

Belle, V., and Juba, B. 2019. Implicitly learning to reason in first-order logic. Advances in neural information processing systems 32.

Belle, V. 2017. Open-universe weighted model counting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 31.

Braz, R.; Amir, E.; and Roth, D. 2005. Lifted first-order probabilistic inference. In Proceedings of the Nineteenth International Joint Conference on Artificial Intelligence, 1319–1325. Morgan Kaufmann San Francisco.

Chickering, D. M. 1996. Learning bayesian networks is npcomplete. In Learning from data: Artificial intelligence and statistics V. Springer. 121–130.

Davis, M., and Putnam, H. 1960. A computing procedure for quantification theory. Journal of the ACM (JACM) 7(3):201–215.

Davis, M.; Logemann, G.; and Loveland, D. 1962. A machine program for theorem-proving. Communications of the ACM 5(7):394–397.

De Raedt, L., and Bruynooghe, M. 1993. A theory of clausal discovery. In Proceedings of the 13th international joint conference on artificial intelligence, 1058–1063. Morgan Kaufmann.

De Raedt, L., and Dzeroski, S. 1994. First-order jk-ˇ clausal theories are pac-learnable. Artificial Intelligence 70(1):375–392.

De Raedt, L., and Kersting, K. 2008. Probabilistic inductive logic programming. In Probabilistic inductive logic programming: theory and applications. Springer. 1–27.

De Raedt, L.; Kimmig, A.; and Toivonen, H. 2007. Problog: A probabilistic prolog and its application in link discovery. In IJCAI 2007, Proceedings of the 20th International Joint Conference on Artificial Intelligence, 2462–2467.

Dong, H.; Mao, J.; Lin, T.; Wang, C.; Li, L.; and Zhou, D. 2019. Neural logic machines. In International Conference on Learning Representations.

Ehrenfeucht, A., and Haussler, D. 1989. Learning decision trees from random examples. Information and Computation 82(3):231–246.

Esteban, J. L., and Toran, J. 2001. Space bounds for resolu-´ tion. Information and Computation 171(1):84–97.

Ge, L.; Juba, B.; and Nilsson, K. 2025. Polynomial-time relational probabilistic inference in open universes. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, 9058–9067.

Gogate, V., and Domingos, P. 2016. Probabilistic theorem proving. Communications ofthe ACM 59(7):107–115.

Juba, B. 2013. Implicit learning of common sense for reasoning. In Rossi, F., ed., Proceedings of the Twenty-Third International Joint Conference on Artificial Intelligence, IJCAI-13, 939–946. AAAI Press / International Joint Conferences on Artificial Intelligence.

Juba, B. 2019. Query-driven pac-learning for reasoning. arXiv preprint arXiv:1906.10118.

Kersting, K., and De Raedt, L. 2007. Bayesian logic programming: theory and tool. Statistical Relational Learning 291.

Kersting, K.; Mladenov, M.; and Tokmakov, P. 2017. Relational linear programming. Artificial Intelligence 244:188–216.

Kersting, K. 2012. Lifted probabilistic inference. In ECAI 2012. IOS Press. 33–38.

Khardon, R., and Roth, D. 1997. Learning to reason. J. ACM 44(5):697–725.

Kok, S., and Domingos, P. 2005. Learning the structure of markov logic networks. In Proceedings of the 22nd international conference on Machine learning, 441–448.

Koller, D., and Friedman, N. 2009. Probabilistic graphical models: principles and techniques. MIT press.

Kullmann, O. 1999. Investigating a general hierarchy of polynomially decidable classes of cnf’s based on short tree-like resolution proofs. Technical Report TR99- 041, Electronic Colloquium on Computational Complexity (ECCC).

Levesque, H. J. 1998. A completeness result for reasoning with incomplete first-order knowledge bases. In Proceedings of the Sixth International Conference on Principles of Knowledge Representation and Reasoning (KR), 14–23.

Meert, W.; Struyf, J.; and Blockeel, H. 2008. Learning ground cp-logic theories by leveraging bayesian network learning techniques. Fundamenta Informaticae 89(1):131–160.

Michael, L. 2010. Partial observability and learnability. Artificial Intelligence 174(11):639–669.

Mladenov, M.; Kleinhans, L.; and Kersting, K. 2017. Lifted inference for convex quadratic programs. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 31.

Muggleton, S., and De Raedt, L. 1994. Inductive logic programming: Theory and methods. The Journal of Logic Programming 19:629–679.

Muggleton, S. 1995. Inverse entailment and progol. New generation computing 13(3):245–286.

Poole, D. 2003. First-order probabilistic inference. In Proceedings ofthe 18th International Joint Conference on Artificial Intelligence, volume 3, 985–991.

Rader, A. P.; Mocanu, I. G.; Belle, V.; and Juba, B. 2021. Learning implicitly with noisy data in linear arithmetic. In Proceedings ofthe 30th International Joint Conference on Artificial Intelligence, 1410–1417.

Richardson, M., and Domingos, P. 2006. Markov logic networks. Machine learning 62:107–136.

Sourek, G.; Aschenbrenner, V.; Zelezny, F.; Schockaert, S.; and Kuzelka, O. 2018. Lifted relational neural networks: Efficient learning of latent relational structures. Journal ofArtificial Intelligence Research 62:69–100.

Srinivasan, A. 2001. The aleph manual.

Valiant, L. G. 2000. Robust logics. Artificial Intelligence 117:231–253.

Van Den Broeck, G.; Taghipour, N.; Meert, W.; Davis, J.; and De Raedt, L. 2011. Lifted probabilistic inference by first-order knowledge compilation. In Proceedings of the Twenty-Second internationaljoint conference on Artificial Intelligence-Volume Volume Three, 2178–2185.

Van den Broeck, G.; Kersting, K.; Natarajan, S.; and Poole, D. 2021. An introduction to lifted probabilistic inference. MIT Press.

Winters, T.; Marra, G.; Manhaeve, R.; and De Raedt, L. 2022. Deepstochlog: Neural stochastic logic programming. In Proceedings of the 36th AAAI Conference on Artificial Intelligence, volume 9, 10090–10100.