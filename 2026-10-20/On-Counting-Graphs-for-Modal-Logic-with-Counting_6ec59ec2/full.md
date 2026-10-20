# On Counting Graphs for Modal Logic with Counting

Avijeet Ghosh # Ñ

MC2, LIP, ENS de Lyon, France

François Schwarzentruber # Ñ

MC2, LIP, ENS de Lyon, France

## Abstract

In this paper, we study the theoretical computational complexity of counting trees and graphs that are classified positively by a graph neural networks, or said equivalently that satisfy a given formula of modal logic with counting. We also study convergence.

2012 ACM Subject Classification

Keywords and phrases graph neural networks, counting problem, modal logic with counting

## 1 Introduction

Validity problems in modal logic is relevant in many reasoning tasks: verification of systems, epistemic planning. However, validity problem is rigid in this sense: a formula is valid if it holds in all models. In this paper, we relax the validity problem to model counting and we study the probabilistic validity problem: Is a formula true in a large proportion of models? Is a formula asymptotically true for large models? What is the proportion of models in which a formula holds? First, let us give motivation from a verification problem of graph neural networks (GNNs).

Recently, an exact correspondence between modal logic with counting verifying GNNs has been settled and applied for verifying properties on GNNs ([19], [18], [5]). In this paper, the only thing the reader must know is that each GNN from a large class of GNNs, namely with an eventually constant activation function or let say with truncated ReLU, is equivalent to a formula φ<sub>GNN</sub> from a logic called $K ^ { \# }$ , which is a natural extension of graded modal logic with linear arithmetic inequations (see [19], [18], [5]). The notion of equivalence is: a pointed graph $G ,$ u is classified positively by the GNN if φ<sub>GNN</sub> holds in $G , u .$

An example of $K ^ { \# }$ -formula is $\# p \geq \# q + 2$ which means that there are more p-successors than q-successors plus two. For instance, the formula $\# p \geq \# q + 2$ is true in the pointed

![](images/a720f9cf4203fccc2e74e3f6e05e048b2666b2b2979351e2b5e5bba353698da9.jpg)

$$
3 \geq 1 + 2 .
$$

Verifying GNNs is important in order to understand how they classify data. For instance, we could verify that a GNN recommends a person whenever that person has at least 3 violinist friends, meaning that for all pointed graphs $( G , u )$ , if the current vertex has at least 3 violinist successors then the GNN classifies $( G , u )$ positively. This kind of verification tasks were addressed in [18], and on quantized GNNs in [21] and [8].

However, it is unlikely that a property holds for all GNNs. For instance, it is unlikely that a GNN recommends a person whenever that person has at least 3 violinist friends. The reason is because the concept of recommended persons is a vague concept that is dificult to explain and does not completely match a property as precise as "having at least 3 violinist friends" (written #violinist ≥ 3). The second reason is that there are approximations due to learning process itself, making unlikely that the obtained GNN exactly corresponds to a given property.

However, it is possible that a property like "having at least 3 violinist friends" is a probable explanation of the classification of the GNN. That is why, instead of checking that φ → $\varphi $ φ<sub>GNN</sub> is valid, we want to compute the probability that $\varphi $ φ<sub>GNN</sub> holds where φ<sub>GNN</sub> is a formula that is true when the GNN recommends the current model, and $\varphi$ is a property, e.g. “the person has at least 3 violinist friends", #violinist $\geq 3$ . We may then conclude that the probability that $\varphi  \varphi _ { \mathrm { G N N } }$ holds in graph of size 42 is more than 0.8.

Our paper is related to 0-1 law that first-order logic has: any FO-sentence $\varphi$ is asymptotically true in large models or false in large models. While specific classes of GNNs have 0-1 law ([2], [1]), this is clearly false for logic $K ^ { \# }$ . Indeed, the limit of the truth of $\# p \geq \# q$ is $1 / 2$ . This justify the study of computing the probability of a formula $\varphi$ given the size of the graph, or counting the number of $\varphi \mathrm { - }$ -models of size n. As many modal logics have the tree-model property, we also focus on trees and counting the number of trees satisfying $\varphi$ given a branching factor.

Counting models. We study the complexity of counting φ-trees/graphs, given formula $\varphi$ from logic $K ^ { \# }$ (see Figure 1).

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>n unary</td><td rowspan=1 colspan=1>n binary</td></tr><tr><td rowspan=1 colspan=1>Trees</td><td rowspan=1 colspan=2>#PSPACEPsPACE-complete</td></tr><tr><td rowspan=1 colspan=1>Graphs</td><td rowspan=1 colspan=1>#P-complete</td><td rowspan=1 colspan=1>#EXPTIMEPSPACE-hard</td></tr></table>

Figure 1 Complexity results about counting trees (n is the branching factor) or graphs (n is then the number of vertices).

Convergence. We show that the proportion of φ-trees does not converge for some $\varphi .$ We propose an artificial trick to make it converge.

Outline. In Section 2 we discuss the related work. In Section 3, we recall the formal definition of logic $K ^ { \# }$ . In Section 4, we define the counting problems we are studying. In Section 5, we provide our complexity results. We also have an implementation using Montecarlo method that samples arbitrary large graphs and trees and gives out the proportion of satisfiability<sup>1</sup>. In Section 6, we exemplify non-convergence in trees and provide a patch for having convergence. In Section 7, we conclude.

Some parts of the proof of Theorem 28 were obtained by AI, although the authors have come up with the general approach by themselves.

## 2 Related Work

The study of convergence of properties when the size of the models goes to infinity goes back to 0-1 law of first-order logic ([12], [9]). It means that the proportion of models satisfying a first-order property φ converges to 0 or 1 when the size n of the model goes to infinity. In this work, the notion of random graph is given by the Erdős–Rényi random graph model.

Later, Granjean proved that deciding whether the limit is 1 for a given FO property is PSPACE-complete [14]. As modal logic is a syntactic fragment of first-order logic, it also has 0-1 law. But, the computation of the limit of a modal formula is $\Delta _ { 2 } ^ { p } .$ -complete when the formula is represented by a DAG, and $\Delta _ { 2 } ^ { p , \log n }$ -complete if represented by a tree (see Theorem 4.6 in [15]), where $\Delta _ { 2 } ^ { p } = \mathrm { P } ^ { \mathrm { N P } }$ is the class of problems decided in poly-time with an NP oracle, and $\Delta _ { ? } ^ { p , \log n }$ is the class of problems decided in poly-time with at most O(log n) queries to some NP oracle.

First-order logic when there are function symbols does not have the 0-1 law but still the proportion of models converges when the size of the model goes to infinity [17].

In literature, there are many logical frameworks developed that deal with boolean combinations of linear inequalities. In [10], the authors have developed a logical framework that expresses probabilities of combination of events in real numbers. In order to do that, they use a boolean combination of linear inequalities to reason. Although they have axiomatised the framework, the underlying model they use is a usual First Order logic model of domains of real numbers.

Counting models has been a major part of modal logic literature. A related work can be [22], where the authors consider Linear Temporal Logics. The paper gives decision versions of counting models: whether a formula has models upto an input bound. The paper proposes procedures for counting word and tree models, each one for when the bound is unary and one for when bound is in binary..

In [23], the authors investigate zero-one law on various coloring and edge probability (proportion) conditions on Markov Logic Networks. They define “regimes” that describe the global structure of the coloring and edge assignment, for example, most vertices are colored using same, etc. They showed results on whether zero-one laws and convergence laws hold depending on such regimes.

In [2], the authors proved 0-1 law for some graph neural networks but with strong assumptions. First, they focus on graph properties while we focus on node properties. But their stronger assumption is that they restrict to classifier that are non-splitting, meaning that the property is not on a limit case (see details in their paper). For instance $\# p \geq \# q$ is a limit case and its probability converges to $1 / 2$ while the probability $( 1 + \epsilon ) \# p \geq \# q$ converges to 1 when the number of successors goes to infinity, when $\epsilon > 0$ . Compared to them, we consider any $K ^ { \# } \mathrm { - f o r m u l a }$ , without non-splitting assumption. However, we also consider truncated ReLU or eventually constant activation functions; setting which corresponds $K ^ { \# }$

In [1], the authors investigate convergence laws of some classes of GNNs. They prove convergence by sampling large graphs from probabilistic graph models like Erdős–Rényi graphs. However, the GNN architecture they consider do not use summation in message passing, but mean.

## 3 Background and Preliminaries

We present a modal logic of counting denoted by $K ^ { \# }$ , that was introduced in [18], but also in [4] and and [5] with a first-order syntax. Logic $K ^ { \# }$ has the same expressivity than graph neural networks with truncated ReLU as an activation function ([18], [5]).

## 3.1 Kripke model

We recall that a Kripke model is a directed graph like structures with each node having some propositional valuations over a countable set of propositions $\mathcal { P } .$

▶ Definition 1. A Kripke model is a tuple $\mathcal { M } = \langle W , R , V \rangle$ where W is a finite countable set of worlds, $R \subseteq W \times W$ is the accessible relation, and $V : W \to 2 ^ { \mathcal { P } }$ is the valuation function. A pointed graph/model $\mathcal { M } , s$ is a Kripke model and a vertex s in $W$

## 3.2 Syntax

Fix a countable set of propositions P. The set of $K ^ { \# }$ -formulas $\varphi$ is defined by the grammar:

$$
\varphi , \psi , \cdots : = p \mid \neg \psi \mid \varphi \vee \psi \mid \xi \geq 0 \qquad \xi , \xi _ { 1 } , \xi _ { 2 } , \cdots : = c \mid 1 _ { \varphi } \mid \# \varphi \mid \xi _ { 1 } + \xi _ { 2 } \mid c \times \xi _ { 1 }
$$

where $p \in \mathcal P$ , c is an integer constant. The non-terminal ξ is for arithmetic expressions. The construction $1 _ { \varphi }$ is an indicator representing whether $\varphi$ is true in the current state. The construction $\# \varphi$ is the number of φ-successors. We write $\mathcal { P } ( \boldsymbol { \varphi } )$ for the set of propositions appearing in $\varphi .$

▶ Example 2. Here is a well-formed $K ^ { \# }$ -formula: $\varphi = ( \# p \geq 3 \# ( \# r \geq 2 ) ) \wedge r$

S<sup>′</sup>(c) = ∅   
Dsf(p) = ∅   
S<sup>′</sup>(1<sub>φ</sub>) = Dsf(φ)   
Dsf(¬ψ) = Dsf(ψ)   
S<sup>′</sup>(#φ) = {φ}   
Dsf(ψ<sub>1</sub> ∨ ψ<sub>2</sub>) = Dsf(ψ<sub>1</sub>) ∪ Dsf(ψ<sub>2</sub>)   
S<sup>′</sup>(ξ<sub>1</sub> + ξ<sub>2</sub>) = S<sup>′</sup>(ξ<sub>1</sub>) ∪ S<sup>′</sup>(ξ<sub>2</sub>)   
Dsf(ξ ≥ 0) = S<sup>′</sup>(ξ)   
S<sup>′</sup>(c × ξ) = S<sup>′</sup>(ξ)

## 3.3 Truth Conditions

The truth conditions $\mathcal { M } , s \models \varphi$ (formula $\varphi$ is true at vertex ${ \mathcal { M } } , s )$ is defined as follows.

▶ Definition 3. Given a $K ^ { \# }$ -formula $\varphi$ and a Kripke model $\mathcal { M } = \langle W , R , V \rangle , \mathcal { M } , s \models \varphi$ is defined by induction on $\varphi \ b y :$

$$
\begin{array} { r l r } { \boldsymbol { \mathcal { M } } , \boldsymbol { s } \Vdash \boldsymbol { p } \ i f f \boldsymbol { p } \in V ( \boldsymbol { s } ) } & { } & { [ [ c ] ] _ { \mathcal { M } , \mathscr { s } } = c } \\ { \boldsymbol { \mathcal { M } } , \boldsymbol { s } \Vdash - \psi \ i f f \ \boldsymbol { \mathcal { M } } , \boldsymbol { s } \updownarrow \in \psi } & { } & { [ [ \gamma _ { \varphi } ] ] _ { \mathcal { M } , \mathscr { s } } = 1 \ i f \ \boldsymbol { M } , \boldsymbol { s } \ v \{ \boldsymbol { \Theta } , \ \boldsymbol { \theta } o t h e r w i s e :  } \\ { \boldsymbol { \mathcal { M } } , \boldsymbol { s } [ - \varphi \vee \psi \ i f f \ \boldsymbol { \mathcal { M } } , \boldsymbol { s }   \psi = \varphi \ o r \ \boldsymbol { \mathcal { M } } , \boldsymbol { s }  [ - [ [ \# \varphi ] ] _ { \mathcal { M } , \mathscr { s } } = \sum _ { t : ( \boldsymbol { s } , t ) \in R } [ [ \boldsymbol { 1 } _ { \varphi } ] ] _ { \mathcal { M } , t }  } \\ { \boldsymbol { \mathcal { M } } , \boldsymbol { s } [ - \boldsymbol { \xi } \geq 0 \ i f f \ [ \boldsymbol { \xi } ] ] _ { \mathcal { M } , \mathscr { s } } \geq 0 } & { } & { [ [ \xi _ { 1 } + \xi _ { 2 } ] ] _ { \mathcal { M } , \mathscr { s } } = [ [ \xi _ { 1 } ] ] _ { \mathcal { M } , \mathscr { s } } + [ [ \xi _ { 2 } ] ] _ { \mathcal { M } , \mathscr { s } } = \sum [ [ \xi _ { 1 } ] ] _ { \mathcal { M } , \mathscr { s } } = \sum [ [ \xi _ { 2 } ] ] _ { \mathcal { M } , \mathscr { s } } } \\ {  [ [ \boldsymbol { c } \times \boldsymbol { \xi } ] ] _ { \mathcal { M } , \mathscr { s } } = \boldsymbol { c } \times [ [ \boldsymbol { \xi } ] ] _ { \mathcal { M } , \mathscr { s } } = \sum _ { t \in \mathcal { T } } [ [ \boldsymbol { \xi } ] ] _ { \mathcal { M } , \mathscr { s } }  } \end{array}
$$

▶ Example 4. Consider the following Kripke model

![](images/20084ab9ed1998db2d7363f867e47c93bad4dc34b9bac0c0d8fb3414f6cff7cd.jpg)

the formula $\varphi$ in Example 2 evaluated from the top node pointed with an arrow. Formula $\varphi$ holds true in this model from the root node. The root node has r true in it. Now consider the $\# p \geq 3 \# ( \# r \geq 2 )$ part of the formula. Note that, $\# p = 3 ~ ( \mathrm { s , t }$ and u). Also, consider the successor node where only $p$ is true (the node $\mathrm { t } )$ . t has two successors where $r$ holds (nodes u and $v )$ . Hence $\# r \geq 2$ holds in t. Note that t is the only successor of root that holds $\# \geq 2$ Hence $\# ( \# r \geq 2 ) = 1$ at the root. Hence $\# p \geq 3 \# ( \# r \geq 2 )$ also holds true at root.

▶ Definition 5 (Modal Depth). The modal depth $m d ( \varphi )$ of formula $\varphi$ is defined as:

$$
\begin{array} { r l } { = } & { { } m d ( p ) = 0 , m d ( \neg \psi ) = m d ( \psi ) , m d ( \varphi \vee \psi ) = m a x ( m d ( \varphi ) , m d ( \psi ) ) , m d ( \xi \geq 0 ) = m d ( \xi ) } \end{array}
$$

md(c) = 0, md(1<sub>φ</sub>) = md(φ), md(#φ) = md(φ)+1, md(ξ<sub>1</sub> +ξ<sub>2</sub>) = max(md(ξ<sub>1</sub>), md(ξ<sub>2</sub>)), md(c × ξ<sub>1</sub>) = md(ξ<sub>1</sub>)

Intuitively, modal depth is the length of the longest nesting of the modality operator $\#$

▶ Example 6. We have $m d ( ( \# ( \# p \geq \# q ) \geq 0 ) \land ( \# p \geq \# q ) ) = 2$ . This is because $( \# ( \# p \geq \# q ) \geq 0 )$ has the maximum modal depth, which is 2.

A subformula of a $K ^ { \# } { \mathrm { - f o r m u l a } }$ is direct when it does lie under the scope of exactly one #-modality. The concept of direct formula is important because we count the number of successors satisfying these direct subformulas to evaluate a formula in the current node. We write $\sf D s f ( \varphi )$ for the set of direct subformulas.

▶ Example 7. In the formula $\varphi = ( \# ( \# p \geq \# q ) \geq 0 ) \wedge ( \# r \geq \# s )$ , subformulas $\# p \geq \# q , r$ and s are direct while $p$ and $q$ are not. We have $\mathsf { D s f } ( \varphi ) = \{ \# p \geq \# q , r , s \}$

Formally $\sf D s f ( \varphi )$ is defined as follows.

▶ Definition 8 (Direct Sub formulas). Given a $K ^ { \# } .$ formula $\varphi ,$ , the set of direct subformulas $\sf D s f ( \varphi )$ is defined by induction on $\varphi$ as:

## 3.4 Link with other logics

Note that standard modal logic can be expressed in $K ^ { \# } \colon \diamond \varphi$ is a shorthand for $\# \varphi \geq 1$ (there exists a φ-successor), while □φ is a shorthand for $\neg \diamond \neg \varphi .$ Graded modal logic can also be expressed in $K ^ { \# } \colon \diamondsuit ^ { \geq k } \varphi$ is a shorthand for $\# \varphi \geq k$ (there exists at least k φ-successor).

## 4 Counting models

In this section, we define our counting problems. To do that, we suppose that $\mathcal { P }$ is finite (otherwise there is already an infinite number of 1-vertex models!).

## 4.1 Counting graphs

We write ${ \mathcal { G } } _ { n }$ for the set of pointed graphs whose domain of size $n .$ . More precisely, we fix the domain to be $\{ s _ { 1 } , \ldots , s _ { n } \}$ and the pointed state is $s _ { 1 }$ , i.e.

$$
\mathcal G _ { n } : = \{ ( \mathcal M , s _ { 1 } ) , | \mathcal M = ( W , R , V ) \mathrm { ~ w i t h ~ } W = \{ s _ { 1 } , \ldots , s _ { n } \} \} .
$$

We define ${ \mathcal { G } } _ { n } ^ { \varphi }$ to be the subset pointed graphs in ${ \mathcal { G } } _ { n }$ satisfying formula $\varphi .$ When not explicitly said otherwise, we suppose then that $\mathcal { P } = \mathcal { P } ( \boldsymbol { \varphi } )$ i.e. that the set of propositions contains exactly the propositions appearing in $\varphi$

▶ Example 9. The number of pointed graphs of size 2 on the proposition set $\{ p \}$ is $2 \times 2 \times$ 4×4 (self-loop on 1, self-loop on 2, type of edges between 1 and 2, valuations). There are 4 pointed graphs of size 2 satisfying $p \wedge \odot \neg p \wedge \sqcup \neg p \colon$

![](images/860e9cf6599c04bf5aa1d55cf4e93955fea5705ab3debe915a7b7de19af23e8b.jpg)

![](images/d0be731a4a52fd84036f39b380b336569af89bded0d0e9098f4499499df9e94b.jpg)

![](images/5308a92c9791fbeff2ff144ea6727350586a418c9c0162a745b7f95c6a2e446a.jpg)

![](images/8db6b4d0c995e3f3a503bea6759d94f5b697d201b287f912b86705c1df526c82.jpg)

Said diferently, when we consider $\mathcal { P } = \{ p \} , | \mathcal { G } _ { n } ^ { p \wedge \bigcirc \neg p \wedge \bigtriangledown \neg p } | = 4 .$

Then the proportion of pointed graphs of size n satisfying $\varphi$ is defined by:

$$
\mu _ { n } ( \varphi ) : = { \frac { | { \mathcal { G } } _ { n } ^ { \varphi } | } { | { \mathcal { G } } _ { n } | } } .
$$

Note that the definition of $\mu _ { n } ( \varphi )$ does not depend on $\mathcal { P } _ { \cdot }$ , as long as $\mathcal { P }$ contains the propositions appearing in φ. As presented in [9], a logical framework L is defined to have zero-one law if for any formula $\begin{array} { r } { \varphi \in \mathcal L , \operatorname* { l i m } _ { n \to \infty } \mu _ { n } ( \varphi ) \in \{ 0 , 1 \} } \end{array}$ . As discussed in the introduction, logic $K ^ { \# }$ does not have zero-one law.

Note that $| { \mathcal { G } } _ { n } | = 2 ^ { n ^ { 2 } + | { \mathcal { P } } | \times n }$ (choosing the set of edges and the valuations at each vertex). Thus, computing $\mu _ { n } ( \varphi )$ reduces to compute $| \mathcal { G } _ { n } ^ { \varphi } |$

## 4.2 Counting trees

Many modal logics have the tree-like model property: if a formula is satisfiable, then it is satisfiable in a pointed tree. That is why it is natural to count the number of trees satisfying a property wrt to the branching factor (i.e. outgoing arity). We define $\mathcal { T } _ { n , d }$ to be the set of pointed trees of depth d and branching factor n. Similarly than for graphs, we define $\mathcal { T } _ { n , d } ^ { \varphi }$ to be the subset of pointed trees in $\mathcal { T } _ { n , d }$ satisfying $\varphi .$

▶ Example 10. Here are the 4 pointed trees of depth 1 with branching factor 2 on the proposition set $\{ p \}$ satisfying $\odot p \land \neg \sqsupset p$

A. Ghosh and F. Schwarzentruber

![](images/bf10883dd8ca72cde9fcc297034fa5088d6d93f1da9804203d62b95f96949685.jpg)

$$
\mathcal { P } = \{ p \} , | T _ { n , 1 } ^ { \diamond _ { p \wedge \neg } \sqcup _ { p } } | = 4 .
$$

We define the proportion of pointed trees of branching factor n satisfying $\varphi$ as follows:

$$
\mu _ { n } ^ { \mathcal { T } } ( \varphi ) : = \frac { | \mathcal { T } _ { n , m d ( \varphi ) } ^ { \varphi } | } { | \mathcal { T } _ { n , m d ( \varphi ) } | } .
$$

In the definition of $\mu _ { n } ^ { \mathcal { T } } ( \varphi )$ , we consider trees of depth $m d ( \varphi )$ . Note that every vertices after depth $m d ( \varphi )$ is not relevant when evaluating $\varphi . \ \mathrm { A g a i n } .$ , for computing $\mu _ { n } ^ { \mathcal { T } } ( \varphi )$ , it is suficient to compute ${ \mathcal { T } } _ { n , m d ( \varphi ) } ^ { \varphi }$ since $| T _ { n , m d ( \varphi ) } | = 2 ^ { \tilde { \mathcal { P } } \times ( 2 ^ { m d ( \varphi ) + 1 } - 1 ) }$ (choosing the valuation for each of the $2 ^ { m d ( \varphi ) + 1 } - 1$ vertices).

## 5 Complexity of Counting models

In this section, we investigate the problem of computing $\mu _ { n } ^ { \mathcal { T } } ( \varphi )$ and $\mu _ { n } ( \varphi )$ , that is computing $| \mathcal { T } _ { n , m d ( \varphi ) } ^ { \varphi } |$ and $| \mathcal { G } _ { n } ^ { \varphi } |$

## 5.1 Complexity of Counting Tree Models

We investigate on calculating the number of trees $| \mathcal { T } _ { n , m d ( \varphi ) } ^ { \varphi } |$ satisfying some $K ^ { \# }$ -formula $\varphi .$

▶ Definition 11. The Counting-Tree problem is defined as:

input: a $K ^ { \# } \cdot$ formula $\varphi ,$ a number n (branching factor)

output: $\lvert \mathcal { T } _ { n , m d ( \varphi ) } ^ { \varphi } \rvert$ , the number of pointed tree models $( { \mathcal { M } } , s )$ , with branching factor n and depth $m d ( \varphi )$ such that $\mathcal { M } , s \models \varphi$

In this subsection, we prove the following theorem.

▶ Theorem 12. Counting-Tree is in #Pspace, $i f$ n is in binary (and also in unary).

Consider a formula $\varphi .$ . We now define the notion of local consistency of $\varphi$ at a given vertex in a tree. The only information we need is the propositional valuation $\nu$ and the number of successors in which each subformula in $\sf D s f ( \varphi )$ is true. That last information is modeled by a function counter : $\mathsf { D s f } ( \varphi ) \to \mathbb { N }$ that assigns to each subformula $\psi \in \mathsf { D s f } ( \varphi )$ by the number of successors in which $\psi$ hold. We now define formally local-consistency of $\varphi ,$ denoted by $( \nu , \mathsf { c o u n t e r } ) \vdash _ { L } \varphi .$

▶ Definition 13 (Local Consistency $\scriptstyle \left| = { } _ { L } \right)$ . Consider a formula $\varphi ,$ a propositional valuation ν over the propositions $o f \varphi ,$ and a function counter : $\mathsf { D s f } ( \varphi ) \to \mathbb { N }$ . Formula $\varphi$ is said to be local-consistent with $( \nu ,$ counter), denoted by (ν, counter) $\scriptstyle = _ { L } \varphi ,$ if φ holds true after applying the following steps in order:

1. For each $\psi \in \mathsf { D s f } ( \varphi )$ , replace each non- $^ { - \# }$ -nested occurrence $\# \psi$ in $\varphi$ by counter $( \psi )$

2. Replace each proposition p left by $\nu ( p )$

▶ Example 14. Consider $\varphi : = q \wedge ( \# p \geq 3 \# ( \# r \geq 2 ) )$ . If $\nu [ q ] = 1$ , counter $( p ) = 3$ and counter $( r \ge 2 ) = 1$ , then $\varphi$ is local consistent with (ν, counter) since q is true (since $v [ q ] = 1 )$ ), $\# p$ is replaced by 3, and $\# ( \# r \geq 2 )$ is replaced by 1, and $3 \geq 3 \times 1$

A Dsf(φ)-valuation $V$ is a function $V : \mathsf { D s f } ( \varphi ) \to \{ 0 , 1 \}$ . Furthermore, given a Dsf $( \varphi ) \cdot$ valuation V, we define

$$
d e s c r ( V ) = \bigwedge _ { \psi | V [ \psi ] = 1 } \psi \wedge \bigwedge _ { \psi | V [ \psi ] = 0 } \neg \psi
$$

For instance, $V ( p ) = 1 , V ( \# r \geq 2 ) = 0$ , then descr $( V ) = p \land \lnot ( \# r \geq 2 )$

We prove Theorem 12 using Algorithm 2 (guessTree). We prove it by the following facts.

▶ Fact 15. The number of accepting executions of guessTree $( \varphi , n )$ is $| \mathcal { T } _ { n , m d ( \varphi ) } ^ { \varphi } |$

Proof. Given a $K ^ { \# }$ formula $\varphi ,$ the set $\mathcal { P }$ of propositions appearing in $\varphi ,$ and the branching factor n, we show that there is a one-to-one correspondence between non-rejecting executions of exploreAllTrees $( \varphi , \mathcal { P } , n )$ and trees in ${ \mathcal { T } } _ { n , m d ( \varphi ) } ^ { \varphi }$ . This suficient because an accepting execution of guessTree $( \varphi , n )$ corresponds to a non-rejecting execution of exploreAllTrees $( \varphi , \mathcal { P } , n )$

For that we first formally define, by induction on the modal depth of $\varphi ,$ a function $f _ { \varphi }$ : NonRejExec<sub>exploreAllTree</sub> ${ \mathrm { \Omega } } _ { \left( \varphi , \mathcal { P } , n \right) } \to { \mathcal { T } } _ { n , m d \left( \varphi \right) } ^ { \varphi }$ where NonRejExec<sub>exploreAllTrees</sub> $( \varphi , \mathcal { P } , n )$ is the set of non-rejecting paths of exploreAllTrees $( { \overset { \underset { \cdots } { \bigtriangledown } } { \varphi } } , { \mathcal { P } } , n )$

The base case is when $\varphi$ is Boolean. By written $\pi _ { \nu }$ the execution that guessed valuation ν at Line 2, we set $f _ { \varphi } ( \pi _ { \nu } )$ to the single pointed vertex tagged with the valuation $\nu .$ In picture:

$$
f _ { \varphi } ( P _ { \nu } ) = \ \longrightarrow \widehat { \psi }
$$

Induction case for a formula $\varphi .$ In that case, the execution starts by guessing a valuation $\nu ,$ guess a valuation $V _ { 1 }$ over $\sf D s f ( \varphi )$ , perform an execution $\pi _ { 1 }$ at line 11 with descr $( V _ { 1 } )$ guess a valuation $V _ { 2 }$ over $\sf D s f ( \varphi )$ , perform an execution $\pi _ { 2 }$ at line 11 with descr $( V _ { 2 } )$ , etc. Said, diferently, the execution is parameterized by $( \nu , V _ { 1 } , \pi _ { 1 } , \ldots , V _ { n } , \pi _ { n } )$ and we write $\pi _ { ( \nu , V _ { 1 } , \pi _ { 1 } , \dots , V _ { n } , \pi _ { n } ) }$ . The tree $f _ { \varphi } ( \pi _ { \left( \nu , V _ { 1 } , \pi _ { 1 } , \dots , V _ { n } , \pi _ { n } \right) } )$ is then a root tagged with valuation ν with children corresponding to the sub-executions $\pi _ { 1 } , \ldots , \pi _ { n }$ . The child for $\pi _ { i }$ is obtained via $f _ { d e s c r ( V _ { i } ) } ( \pi _ { i } )$

![](images/5e6caa9804e5d295d844c740ac9c770335397f6375d681fecbdac02fd4db52ce.jpg)

Each accepting path corresponds to a satisfying model. In other words, we prove that function $f$ is a bijection. The proof goes by induction on the modal depth of $\varphi$ . The base case, when $\varphi$ is Boolean, is straightforward. Now for inductive step, we proceed as follows. Surjection. Consider a pointed tree model $\tau \in \mathcal { T } _ { n , m d ( \varphi ) } ^ { \varphi }$ . We construct an execution $\pi$ as follows. First, we guess the valuation ν over $\mathcal { P } ( \varphi )$ as in the root of $\tau$ at Line 6. Now consider child i among the n many children of the root. Note that $\sf D s f ( \varphi )$ are exactly the sub-formulas of $\varphi$ that have exactly one less $\# .$ -modal depth. Hence, in our execution, we guess the valuation over $\sf D s f ( \varphi )$ making exactly the formulas in Dsf(φ) true in the i-th child (Line 9 of the i-th iteration of for loop in Line 8). Hence at that i-child, desc(V ) holds. By induction hypothesis, there exists a non-rejecting execution $\pi _ { i }$ for the i-th call at Line 11. The obtained execution π is $\pi _ { ( \nu , V _ { 1 } , \pi _ { 1 } , . . . , V _ { n } , \pi _ { n } ) }$

Injection. Take two non-rejecting executions $P _ { 1 } \neq P _ { 2 }$ . Hence they difer on at least one of the nondeterministic choice. Just like the surjection proof, since the non-deterministic choices correspond to valuation of either propositions in the root or child nodes, they will correspond to diferent trees.

## A. Ghosh and F. Schwarzentruber

## ▶ Fact 16. exploreAllTrees runs in poly-space.

Proof. The number of nested calls is bounded by the #-modal depth of the initial formula $\varphi .$ Remember that the loop at Line 7 takes an exponential time in the size of n (since n is in binary), but the memory usage of each recursive call is cleaned after having been performed. The memory used at each call is polynomial in the size of the input: storing the valuation ν and V is in $O ( | \varphi | )$ where $\varphi$ is the initial formula. Also, for all $\psi \in \mathsf { D s f } ( \varphi )$ , counter $[ \psi ] \leq n$ . So counter[ψ] can be represented with $O ( \log n )$ bits which is of polynomial size to the input. ◀

Algorithm 1 exploreAllTrees   
Input: $K ^ { \# } .$ -formula φ, $\mathcal { P } = \mathcal { P } ( \boldsymbol { \varphi } )$ propositions of $\varphi ,$ branching factor n in binary   
Output: reject or hand over   
1 if $\varphi$ is Boolean then   
2 Guess a valuation ν over $\mathcal { P } ;$   
3 if ν $\nvDash \varphi$ then reject ;   
4 else   
5 Guess a valuation ν over $\mathcal { P } ;$   
6 counter $ [ 0 ] _ { \mathsf { D s f } ( \varphi ) } ;$   
7 for $i = 1 \dots n$ do   
8 Guess a valuation $V$ over $\scriptstyle \mathsf { D s f } ( \varphi ) ;$   
9 counter[ψ] ← counter $[ \psi ] + 1$ for each $\psi \in \mathsf { D s f } ( \varphi )$ such that $V ( \psi ) = 1$   
10 exploreAllTrees (descr(V ), P, n);   
11 if $( \nu ,$ counter) $\neq _ { L } \varphi$ then reject ;

Input: $K ^ { \# }$ -formula $\varphi ,$ , integer n in binary   
Output: accept if exploreAllTrees hands over; reject otherwise   
1 exploreAllTrees $( \varphi , n )$   
2 accept ;

## ▶ Theorem 17. Given $\varphi ,$ , deciding that $| \mathcal { T } _ { 2 } ^ { \varphi } | \geq 1$ is PSPACE-hard.

Proof. Reduction from the satisfiability problem of standard modal logic in a tree of branching 2 which is PSPACE-hard (this is folklore but stated formally in Lemma 14 of [21]). ◀

Now counting is important. But usually we want to deal with proportions, for instance knowing that $\mu _ { n } ^ { \mathcal { T } } ( \varphi ) \geq 1 / 2$ . This problem reduces to know that $| T _ { n , m d ( \varphi ) } ^ { \varphi } | \geq | T _ { n , m d ( \varphi ) } | / 2$ More generally, we aim at knowing whether $| T _ { n , m d ( \varphi ) } ^ { \varphi } | \geq \kappa$ . However, there is a trick in the representation of $\kappa .$ Indeed, note that κ represented in binary is not suficient since $| \mathcal { T } _ { n , m d ( \varphi ) } | / 2$ is $2 ^ { \mathcal { P } \times ( 2 ^ { m d ( \varphi ) + 1 } - 1 ) - 1 }$ which requires an exponential number of bits in $| \varphi |$ to written down. However, that number is easily represented by 1 0 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 0.

$$
\mathcal { P } \times \left( 2 ^ { m d ( \varphi ) + 1 } - \dot { 1 } \right) - 1
$$

And $\mathcal { P } \times ( 2 ^ { m d ( \varphi ) + 1 } - 1 ) - 1$ can be written down with a polynomial number of bits. In other words, we represent κ with a RLE (Run-length encoding) representation where each blocks of 0 and 1 are represented by the digit (0 or 1) and the number of occurrences of that digit.

▶ Theorem 18. Given $\varphi , n , \kappa ,$ deciding whether $| T _ { 2 } ^ { \varphi } | \geq \kappa$ is in $P S P A C E ,$ when n is in binary, and κ is in RLE compression representation.

Proof. First by Ladner theorem, we have that # $P S P A C E = F P S P A C E$ [16]. It means there is a poly-space machine, given φ, n, output on a write-only tape the number of trees with branching n satisfying $\varphi .$ . Note that this number can have an exponential number of bits. Finally, we compare κ to that output number on-the-fly (comparing numbers with RLE compression representation can be performed in time $O ( N ^ { 2 } )$ where N is the size of the RLE compression representation, see [3]). ◀

## 5.2 Complexity of Counting Graph Models

Unlike the previous sections, which deals with $K ^ { \# }$ formulas being interpreted on pointed trees, we now generalise and look at pointed directed graphs. We call this problem as Counting-Graph problem. Formally,

▶ Definition 19 (Counting-Graph). The Counting-Graph problem is defined as Input: A $K ^ { \# }$ formula $\varphi ,$ an integer n

Output: $| \mathcal { G } _ { n } ^ { \varphi } |$ , the number of pointed n-state Kripke models that satisfy $\varphi ,$ where the set of propositions is $\mathcal { P } = \mathcal { P } ( \boldsymbol { \varphi } )$

In the previous definition, remember that the domain of a Kripke model of size n is $\{ 1 , \ldots , n \}$ and that the designated point s is always 1. We also suppose that the set of propositions is implicitly given by the formula $\varphi .$

Input: $K ^ { \# }$ -formula $\varphi ,$ integer n in unary   
Output: Guess a pointed Kripke model ${ \mathcal { M } } ,$ s such that $| { \mathcal { M } } | = n$ and accept if   
$\mathcal { M } , s \models \varphi$ or reject otherwise   
1 Non-deterministically guess a pointed graph $( \mathcal { M } , s )$ of n vertices and on set of   
propositions $\mathcal { P } = \mathcal { P } ( \boldsymbol { \varphi } )$ ;   
2 accept if $\mathcal { M } , s \models \varphi$ else reject ;

▶ Theorem 20. Counting-Graph is # P-complete when n is given in unary.

## Proof.

Upper Bound. Algorithm 3 gives a nondeterministic algorithm guessGraph taking as an input a formula $\varphi ,$ an integer n in unary, it guesses an n-state pointed Kripke model and checks if it satisfies $\varphi .$ . Each pointed graph $( \mathcal { M } , s )$ of n vertices can be represented by its adjacency matrix and a valuation at each vertex, we suppose here that the designated point s is always the first one. So there are $2 ^ { n ^ { 2 } + n \times | \mathcal { P } | }$ possible models of size $n .$

▶ Fact 21. guessGraph $( \varphi , n )$ runs in poly-time in $| \varphi |$ , and $n$

Proof. Let us analyze the algorithm.

Line 1. This step consists in guessing a binary word of length $n ^ { 2 } + n \times | \mathcal { P } |$ in time $n ^ { 2 } + n \times | \mathcal { P } |$

Line 2. Checking $G , u \vdash \varphi$ is in time $p o l y ( | \varphi | , n )$

▶ Fact 22 (correctness). $| \{ ( \mathcal { M } , s ) ~ | ~ \mathcal { M } , s \models \varphi , | \mathcal { M } | = n \} |$ is exactly the number of accepting paths of guessGraph(φ, n).

Proof. One execution of guessGraph $( \varphi , n )$ is guessing a distinct Kripke model $( G , u )$ . So the number of accepting executions of of guessGraph $( \varphi , n )$ is $| \mathcal { G } _ { n } ^ { \varphi } |$ ◀

Lower Bound. Recall the problem of #SAT of propositional logic is: given a propositional formula $\varphi ,$ compute the number of valuations that makes $\varphi$ true. The problem $\# \mathrm { S A T }$ is #P-complete [24].

We reduce $\# S A T$ problem in poly-time to Counting-Graph as follows. Given a formula $\varphi ,$ we build the Counting-Graph instance $( \varphi , n = 1 )$ . Note that the reduction is in poly-time. Finally, we have: $\# S A T ( \varphi ) = k { \mathrm { ~ i f f ~ } } | { \mathcal { G } } _ { 1 } ^ { \varphi } | = k$ ◀

The same Algorithm 3 gives an exponential blow-up in the upper bound when n in binary.

▶ Theorem 23. Counting-Graph(φ, n) is in #Exptime when n is given in binary.

▶ Theorem 24. Given $\varphi , n ,$ deciding whether $| \mathcal { G } _ { \varphi } ^ { n } | \ge 1$ is Pspace-hard when n is in binary.

Proof. By reduction from satisfiability problem of standard modal logic in a tree of branching 2 which is PSPACE-hard. Let $\varphi$ be a formula of standard modal logic. Take n to be the size of trees with branching 2 of depth $d = m d ( \varphi )$ , i.e. $n = 2 ^ { d } - 1$ . Take a formula $\tau$ that says that the model contains a tree of branching factor 2. Typically, $\tau$ is the formula $\bigwedge _ { i = 0 } ^ { d - 1 } \big \big \big \big \big \big \mu ^ { i } \big ( \neg \ell _ { 0 } \wedge \dots \to \ell _ { i - 1 } \wedge \ell _ { i } \wedge \dots \ell _ { i + 1 } \wedge \dots \dots \ell _ { d - 1 } \big ) \wedge \big [ \diamond x _ { i } \wedge \diamond \neg x _ { i } \wedge \bigwedge _ { i < i } ( x _ { j } \to \big \square x _ { j } ) \wedge \big ( \neg x _ { j } \to \big \square x _ { j } \big ) \big \qquad $ where $\ell _ { 0 } , \dots , \ell _ { d - 1 } , x _ { 0 } , \dots , x _ { d - 1 }$ are fresh propositions (see $[ 7 ] )$ . The idea of τ is that there is a 2-branching over the Boolean value of $x _ { i }$ while values of $x _ { j }$ with $j < i$ are propagated. Leaves correspond to all valuations over $x _ { 0 } , \ldots , x _ { d - 1 }$ . The proposition $\ell _ { i }$ means that we are at level $i .$ The reduction is $t r ( \varphi ) : = ( \varphi \wedge \tau , n )$ . If $\varphi$ is satisfiable in a tree of branching $2 .$ then $\varphi \wedge \tau$ is satisfiable in a tree of branching 2 which is a graph of size n. Conversely, if $\varphi \wedge \tau$ is true in a graph of size $n ,$ then that graph must be a tree of branching 2. ◀

## 6 Theoretical results at infinity

We now study the convergence of $\mu _ { n } ^ { \mathcal { T } } ( \varphi )$ when the branching factor n goes to infinity.

## 6.1 Counterexample for convergence

In this subsection, we come up with an example of a $K ^ { \# }$ formula $\varphi$ such that lim ${ _ { n \to \infty } } \mu _ { n } ^ { \mathcal { T } } ( \varphi )$ does not exist. Our candidate formula $\varphi$ is $\# \varphi _ { e } \geq \# \varphi _ { o }$ where $\varphi _ { e }$ and $\varphi _ { o }$ are:

$$
\varphi _ { e } : = \# \mathsf { T } = 2 \# p \qquad \varphi _ { o } : = \# \mathsf { T } = 2 \# p - 1 .
$$

Intuitively, $\varphi _ { e }$ is a formula that asserts that the number of children to the current tree-node should be exactly twice the number of children where $p$ holds. In other words, number of children where $p$ holds is equal to the number of children where it does not. Similarly, $\varphi _ { o }$ asserts that the number of children where $p$ holds is exactly 1 less than the number of children where it does not.

Unfortunately, $\mu _ { n } ^ { \mathcal { T } } ( \# \varphi _ { e } \geq \# \varphi _ { o } )$ does not converge when n goes to infinity: its value flips between 1 and numbers that are converging to 0; see Figure 2.

▶ Proposition 25. lim $_ { n  \infty } \mu _ { n } ^ { T } ( \# \varphi _ { e } \geq \# \varphi _ { o } )$ does not exist.

![](images/524fcc377a1d4a9e0dd927623117cbb9e430bdb8438c9360658f18296be7e41f.jpg)  
Figure 2 Values of $\mu _ { n } ^ { \mathcal { T } } ( \# \varphi _ { e } \geq \# \varphi _ { o } )$ depending on the branching factor n.

Proof. We only consider trees with depth 2. This is because the formula has modal depth of $2 ,$ hence cannot evaluate nodes that lie deeper. We also treat the probability of a tree node holding $p$ true to be equally likely: a fair coin toss. Hence, each leaf node (depth 2) represents one independent fair coin toss. Each node in the level above (depth 1) represents an experiment that involves exactly n many independent coin tosses. It is in this level where we evaluate the event space of the experiment based on $\varphi _ { e } \ \mathrm { o r } \ \varphi _ { o }$ . Lastly, in the root (depth 0), we compare the count of the event spaces. Let us compute the probability of $( \# \varphi _ { e } \geq \# \varphi _ { o } )$ depending on the parity of the branching factor.

Even branching. When the branching is even, $\varphi _ { o }$ is false. There, $\# \varphi _ { o }$ is always 0. Thus, formula $\# \varphi _ { e } \geq \# \varphi _ { o }$ is trivially true. Thus we have, $\mu _ { 2 n } ^ { \mathcal { T } } ( \# \varphi _ { e } \geq \# \varphi _ { o } ) = 1$

Odd branching. $\mathrm { A s } \# \varphi _ { e }$ is zero when the branching is odd, by definition $\mu _ { 2 n - 1 } ^ { \mathcal { T } } ( \# \varphi _ { e } \geq$ $\# \varphi _ { o } )$ is the probability that $\# \varphi _ { o } = 0$ . Now, the probability that $\varphi _ { o }$ holds in a node is $\binom { 2 n - 1 } { n } / 2 ^ { 2 n - 1 }$ (the probability of n among 2n − 1 trials). So the probability of $\# \varphi _ { o } = 0$ is the probability of having $\varphi _ { o }$ false $2 n - 1$ times. So

$$
\mu _ { 2 n - 1 } ^ { \mathcal { T } } ( \# \varphi _ { e } \geq \# \varphi _ { o } ) : = \left( 1 - { \binom { 2 n - 1 } { n } } / 2 ^ { 2 n - 1 } \right) ^ { 2 n - 1 }
$$

which tends to 0 when n tends to +∞ (see appendix 7).

## 6.2 Alternative definition of the limit

The following theorem gives a limit result for the probability of a formula $\varphi$ to be true when the number of successors n goes to infinity. Computing the limit of $\mu _ { n } ^ { \mathcal { T } } ( \varphi )$ directly causes technical issues that we will discuss after the following theorem.

The technical solution we propose to study a limit result works as follows. Instead of taking the limit of $\mu _ { n } ^ { \mathcal { T } } ( \varphi )$ when n goes to infinity, we propose n going to infinity at each level in the tree, and define a new quantity denoted by $\mu _ { n } ^ { \prime } ( \varphi )$ . The base case is when $\varphi$ is Boolean. In that case, $\mu _ { n } ^ { \prime } ( \varphi ) : = \mu _ { n } ^ { T } ( \varphi )$ is constant and thus trivially lim ${ \bf { \sigma } } _ { n  + \infty } \mu _ { n } ^ { \prime } ( \varphi )$ exists. Now for the inductive case, we define $\mu _ { n } ^ { \prime } ( \varphi )$ as the limit when the branching factor at the root goes to infinity, provided that a direct subformula $\psi$ is true with probability $\scriptstyle \operatorname* { l i m } _ { n \to + \infty } \mu _ { n } ^ { \prime } ( \psi )$ which exists by induction.

It turns out that we define $\mu _ { n } ^ { \prime } ( \varphi )$ inside the following theorem because the existence lim $1 _ { n  + \infty } \mu _ { n } ^ { \prime } ( \psi )$ for the direct subformulas $\psi$ is actually the conclusion of the theorem: lim ${ \bf { \sigma } } _ { n  + \infty } \mu _ { n } ^ { \prime } ( \varphi )$ exists.

Observe that for all formulas $\varphi$ of modal depth 1, $\mu _ { n } ^ { \prime } ( \varphi ) = \mu _ { n } ^ { T } ( \varphi )$

![](images/9c8c6e349bf0646fba7a69f4afdda96350fe4c09e6a6a241fe46930b1dc0949b.jpg)

▶ Definition 26. We define $\mu _ { n } ^ { \prime } ( \varphi )$ by induction on $\varphi$ to be the probability that a rooted tree of the form such that for all $\psi \in \mathsf { D s f } ( \varphi )$ , the probability that some successor

satisfies ψ is lim ${ \bf \cdot } n {  } + \infty \mu _ { n } ^ { \prime } ( \psi )$

The previous definition is well-formed because of the following theorem, and we kept the definition apart from the theorem for better clarity. If the limit lim $\scriptstyle 1 n \to + \infty \mu _ { n } ^ { \prime } ( \varphi )$ exists, then it is written $\mu _ { \infty } ^ { \prime } ( \varphi )$

▶ Example 27. Following the counterexample given in Section 6.1, note that $\mu _ { \infty } ^ { \prime } ( \# \varphi _ { e } \geq \# \varphi _ { o } )$ exists. When the branching factor goes to infinity, the probability that $\varphi _ { e }$ holds at the root tends to 0 (it becomes less and less probable to have exactly half of the successors to satisfy $p )$ . Similarly, the probability of $\varphi _ { o }$ also tends to 0. Therefore, $\mu _ { \infty } ^ { \prime } ( \varphi _ { e } ) = \mu _ { \infty } ^ { \prime } ( \varphi _ { o } ) = 0$ . Hence, $\mu _ { n } ^ { \prime } ( \# \varphi _ { e } \geq \# \varphi _ { o } ) = 1$ for any n. This implies $\mu _ { \infty } ^ { \prime } ( \# \varphi _ { e } \geq \# \varphi _ { o } ) = 1$

▶ Theorem 28. For all $K ^ { \# }$ -formulas $\varphi _ { i }$ , lim $\mathfrak { l } _ { n \to + \infty } \mu _ { n } ^ { \prime } ( \varphi )$ exists.

Proof. As per the syntax of $K ^ { \# }$ , each formula is a boolean combination of literals and linear equations with $\# \psi$ as variables. Hence, we can write a $K ^ { \# }$ formula $\varphi$ as a DNF:

$$
\varphi  \bigvee _ { j } ( S _ { j } \land \beta _ { j } )
$$

where the $S _ { j } \mathrm { ^ { * } s }$ are mutually exclusive systems (conjunctions) of inequalities and $\beta _ { j }$ is a Boolean formula. Note that $\beta _ { j }$ and $S _ { j }$ are independent because $\beta _ { j }$ is about the current vertex valuation and $S _ { j }$ is about the successors.

▶ Example 29. $\varphi : = ( \# p \geq \# q ) \vee ( \# p \geq \# r ) \wedge ( p \vee \neg q )$ is rewritten as

$$
\underbrace { [ ( \# p \geq \# q ) \wedge ( \# p < \# r ) } _ { S _ { 1 } } \wedge \underbrace { ( p \vee \neg q ) ] } _ { \beta _ { 1 } } \vee
$$

$$
\underbrace { [ ( \# p < \# q ) \wedge ( \# p \geq \# r ) } _ { S _ { 2 } } \wedge \underbrace { ( p \vee \neg q ) ] } _ { \beta _ { 2 } } \vee
$$

$$
\underbrace { [ ( \# p \geq \# q ) \wedge ( \# p \geq \# r ) } _ { S _ { 3 } } \wedge \underbrace { ( p \vee \neg q ) } _ { \beta _ { 3 } } ]
$$

For the rest of the proof, we assume that the given formula $\varphi$ is in DNF.

The proof proceeds by induction on the #-modal depth. For #-modal depth 0, the formula is Boolean. Therefore $\mu _ { n } ^ { \prime } ( \varphi )$ is constant and thus converges.

Now, we suppose the limits exist for a modal depth $d - 1$ . Consider the formula $\varphi$ in DNF of modal depth d.

Consider the formulas $\psi _ { 1 } , \psi _ { 2 } , \ldots \psi _ { m } \in \mathsf { D } \mathsf { s } \mathsf { f } ( \varphi )$ such that they occur in φ as $\# \psi _ { i }$ for any $i \in [ 1 . . m ]$ . By induction, $\mu _ { n } ^ { \prime } ( \psi _ { i } )$ converges to some $\mu _ { \infty } ^ { \prime } ( \psi _ { i } )$ when n goes $\mathrm { t o } + \infty$

We have to prove that $\mu _ { n } ^ { \prime } ( \varphi )$ converges to some $\mu _ { \infty } ^ { \prime } ( \varphi )$ when n goes to $+ \infty .$

Since φ is in a DNF form, hence $\begin{array} { r } { \mu _ { n } ^ { \prime } ( \varphi ) = \sum _ { j } \mu _ { n } ^ { \prime } ( S _ { j } ) \times \mu _ { n } ^ { \prime } ( \beta _ { j } ) } \end{array}$ . As $\beta _ { j }$ are propositional, $\mu _ { n } ^ { \prime } ( \beta _ { j } )$ is constant. First we prove that if li $\scriptstyle 1 n \to + \infty \mu _ { n } ^ { \prime } ( S _ { j } )$ exists for all $j ,$ then $\scriptstyle \operatorname* { l i m } _ { n \to + \infty } \mu _ { n } ^ { \prime } ( \varphi )$ exists in the following fact. The rest of the proof becomes proving that $\scriptstyle \operatorname* { l i m } _ { n \to + \infty } \mu _ { n } ^ { \prime } ( S _ { j } )$ exists for all $j .$

▶ Fact 30. Let S be a system of inequalities. Then lim $_ { \cdot n  + \infty } \mu _ { n } ^ { \prime } ( S )$ exists.

Proof. For a given n, and $i = 1 . . n ,$ , and ψ be a (direct) subformula, let $X _ { n , i , \psi }$ be an indicator random variable such that

$$
X _ { n , i , \psi } = { \left\{ \begin{array} { l l } { 1 , { \mathrm { i f ~ t h e r e ~ i s ~ a ~ s u b t r e e ~ a t ~ i n d e x ~ } } i { \mathrm { ~ o f ~ n ~ c h i l d r e n ~ s a t i s f y i n g ~ } } \psi } \\ { 0 , { \mathrm { o t h e r w i s e } } } \end{array} \right. }
$$

$$
\begin{array} { r } { \mathrm { T h e r e f o r e , } \ \mathbb { P } ( X _ { n , i , \psi } = 1 ) = \operatorname* { l i m } _ { n \to + \infty } \mu _ { n } ^ { \prime } ( \psi _ { i } ) = \mu _ { \infty } ^ { \prime } ( \psi _ { i } ) . } \end{array}
$$

Let $N _ { n , \psi }$ be the number of successors satisfying $\psi$ among the at most n successors. Formally, $\begin{array} { r } { N _ { n , \psi } = \sum _ { i = 1 , . . n } X _ { n , i , \psi } } \end{array}$ . We define the $\sf D s f ( \varphi )$ -vector $N _ { n } : = ( \langle N _ { n , \psi } \rangle _ { \psi \in \mathsf { D s f } ( \varphi ) } ) ^ { T }$ , that is a column vector of $\sf D s f ( \varphi )$ dimension. Formally, it is defined by $\begin{array} { r } { N _ { n } : = \sum _ { i = 1 } X _ { n , i , \cdot } } \end{array}$

Let k be the number of inequations in S. The system S for branching n is of the form

$$
A N _ { n } \geq b\tag{1}
$$

where A is $k \times \mathsf { D s f } ( \varphi )$ -matrix; b is a k dimension column vector. We use a version of Lindeberg-Feller Central Limit theorem [11] to prove the convergence/existence of the limit. Since we have vectors of sum of indicator random variable, we use the Cramer-Wold theorem and Multivariate CLT [6]. For that purpose, we need the covariance matrix of every pair of indicator random variable.

Given a $\psi , \psi ^ { \prime } \in \mathsf { D s f } ( \varphi )$ , the covariance between $X _ { n , i , \psi }$ and $X _ { n , j , \psi ^ { \prime } }$ for any $i , j \colon$

$$
\begin{array} { r l } & { \Sigma _ { \psi , \psi ^ { \prime } } : = \left\{ \begin{array} { l l } { \mathbb { P } ( X _ { n , i , \psi } = 1 \mathrm { ~ a n d ~ } X _ { n , j , \psi ^ { \prime } } = 1 ) - \mu _ { \infty } ^ { \prime } ( \psi ) \times \mu _ { \infty } ^ { \prime } ( \psi ^ { \prime } ) , } & { \mathrm { i f ~ } \psi \neq \psi ^ { \prime } } \\ { \mathbb { P } ( X _ { n , i , \psi } = 1 ) ( \mathbb { P } ( X _ { n , i , \psi } = 1 ) ) , } & { \mathrm { i f ~ } \psi = \psi ^ { \prime } } \end{array} \right. } \\ & { \quad \quad = \left\{ \begin{array} { l l } { \mu _ { \infty } ^ { \prime } ( \psi \wedge \psi ^ { \prime } ) - \mu _ { \infty } ^ { \prime } ( \psi ) \times \mu _ { \infty } ^ { \prime } ( \psi ^ { \prime } ) , } & { \mathrm { i f ~ } \psi \neq \psi ^ { \prime } } \\ { \mu _ { \infty } ^ { \prime } ( \psi ) ( 1 - \mu _ { \infty } ^ { \prime } ( \psi ) ) , } & { \mathrm { i f ~ } \psi = \psi ^ { \prime } } \end{array} \right. } \end{array}\tag{2}
$$

Consider the covariance matrix $\Sigma = [ \Sigma _ { \psi , \psi ^ { \prime } } ] _ { | \mathsf { D } \mathsf { s } \mathsf { f } ( \varphi ) | \times | \mathsf { D } \mathsf { s } \mathsf { f } ( \varphi ) | }$

Since, the setup is a summation of indicator random variables, by applying Cramer-Wold theorem and then translating to Multivariate CLT [6] we get:

$$
Z _ { n } : = \frac { N _ { n } - n \mu ^ { \prime } } { \sqrt { n } } \xrightarrow [ ] { n \to + \infty } \mathcal { N } _ { \mathsf { D s f } ( \varphi ) } ( 0 , \Sigma )\tag{3}
$$

where Σ is the covariance matrix whose entries are indexed by ${ \sf D s f } ( \varphi ) \times { \sf D s f } ( \varphi )$ and are given by Equation (2). $\mathcal { N } _ { \sf D s f ( \varphi ) } ( 0 , \Sigma )$ denotes the Normal distribution of $| \mathsf { D s f } ( \varphi ) |$ dimension vectors, whose mean is at 0 and the variance along each of $| \mathsf { D s f } ( \varphi ) |$ dimensions is $\Sigma .$ , the covariance matrix.

Since we have our system of inequalities as $A N _ { n } \geq b$ , we rewrite system in Equation (3) as:

$$
A Z _ { n } \geq { \frac { b - n A \mu ^ { \prime } } { \sqrt { n } } } = \underbrace { { \frac { b } { \sqrt { n } } } - { \sqrt { n } } A \mu ^ { \prime } } _ { c _ { n } }\tag{4}
$$

Since as $n \to \infty$ , the expression $\textstyle { \frac { b } { \sqrt { n } } } \to 0$ . Hence the validity of the Equation (4) depends only on $A \mu ^ { \prime }$

Recall that $Z _ { n } , \mu ^ { \prime }$ are Dsf(φ)-vectors, A is $k \times \mathsf { D s f } ( \varphi ) \mathrm { \mathrm { - m a t r i x } } ; ~ b$ is a k-vector. Also, the Equation (3) becomes:

$$
A Z _ { n } \xrightarrow { n \to + \infty } \mathcal { N } _ { k } ( 0 , A \Sigma A ^ { t } )\tag{5}
$$

Since each row in A are the co-eficients of one inequality in the system, depending on how rows in $A \mu ^ { \prime }$ (row i denoted by $[ A \mu ^ { \prime } ] _ { i } )$ behaves, the convergence of the coordinates of $c _ { n }$ can be one of following cases:

1. If at least for one $i , [ A \mu ^ { \prime } ] _ { i } < 0$ , then $c _ { n }$ converges to $+ \infty$ . So system $S$ is unsatisfiable at the limit. Therefore l $\begin{array} { r } { \operatorname* { i m } _ { n  \infty } \mu _ { n } ^ { \prime } ( S ) = 0 } \end{array}$

2. If for all $i , [ A \mu ^ { \prime } ] _ { i } > 0 , c _ { n }$ converges to $- \infty$ . Hence $A Z _ { n }$ grows faster than the RHS in Equation (4) as $n \to \infty$ which implies lim $\operatorname { 1 } _ { n \to \infty } \mu _ { n } ^ { \prime } ( S ) = 1$

3. If for every $i , [ A \mu ^ { \prime } ] _ { i } = 0$ , then $c _ { n }$ converges to 0. We have $\mu _ { n } ^ { \prime } ( S ) = \mathbb { P } ( A Z _ { n } \geq { \frac { b } { \sqrt { n } } } )$ By Equation (5), $A Z _ { n } \ { \xrightarrow { { n \to + \infty } } } \ \mathbf { Z } \sim { \mathcal { N } } _ { k } ( 0 , A \Sigma A ^ { t } )$ and since $\begin{array} { r } { \operatorname* { l i m } _ { n \to \infty } \frac { b } { \sqrt { n } } = 0 } \end{array}$ , by Portmanteau theorem [6]:

$$
{ \begin{array} { r l } & { \mu _ { \infty } ^ { \prime } ( S ) = \displaystyle \operatorname* { l i m } _ { n \to \infty } \mathbb { P } ( A N _ { n } \geq b ) } \\ & { \qquad = \displaystyle \operatorname* { l i m } _ { n \to \infty } \mathbb { P } ( A Z _ { n } \geq { \frac { b } { \sqrt { n } } } ) } \\ & { \qquad = \mathbb { P } ( A \mathbf { Z } \geq 0 ) , { \mathrm { a p p l y i n g ~ P o r t m a n t e a u ~ t h e o r e m } } } \end{array} }
$$

▶ Example 31. With system $\# p \geq 5 \land \# q \geq 2$ , we have matrix $A = { \binom { 1 } { 0 } } \quad 0 \quad \quad$ and $\mu ^ { \prime } = { \binom { 1 / 2 } { 1 / 2 } }$ Thus we are in case 2 and all the constraints can be ignored. We are left with a matrix of "relevant" constraints of size $0 \times 0$ . Thus $\mu ( \varphi ) = 1$

▶ Example 32. With system $\# p - \# q \geq 5 \wedge 2 \# p - 3 \# r \geq 2 \#$ , we have matrix $A = { \binom { 1 } { 2 } } \quad { \binom { - 1 } { 3 } }$ and $\mu ^ { \prime } = { \binom { 1 / 2 } { 1 / 2 } }$ , thus $A \mu ^ { \prime } = \binom { 0 } { - 1 / 2 }$ . As the second coordinate of $A \mu ^ { \prime }$ is negative, we are in point 1 and $\mu ( \varphi ) = 0$

▶ Example 33. With system # $\stackrel { \iota } { \cdot } p - \# q \geq 5 \wedge 3 \# p - 2 \# r \geq 2$ , we have matrix $A = { \binom { 1 } { 3 } } \quad { \binom { - 1 } { 2 } }$ and $\mu ^ { \prime } = { \binom { 1 / 2 } { 1 / 2 } }$ , thus $A \mu ^ { \prime } = { \binom { 0 } { 1 / 2 } }$ . As the second coordinate of $A \mu ^ { \prime }$ is positive, so we can ignore the second constraint.

At the limit, system $\# p - \# q \geq 5 \wedge 3 \# p - 2 \# r \geq 2$ has the same probability to be true than system $\# p - \# q \geq 5$ . The limit probability is $P ( Z _ { 1 } - Z _ { 2 } \ge 0 )$ where Z is a 2-dimensional normal distribution. Since $Z _ { 1 } - Z _ { 2 }$ is a standard normal distribution, $P ( Z _ { 1 } - Z _ { 2 } \ge 0 ) = 1 / 2$

▶ Example 34. With system $\# p - \# q \geq 5 \wedge \# p - \# r \geq 2$ , we have matrix $\begin{array} { r } { A = \left( \begin{array} { c c c } { 1 } & { - 1 } & { 0 } \\ { 1 } & { 0 } & { - 1 } \end{array} \right) } \end{array}$ and $\mu ^ { \prime } = \binom { 1 / 2 } { 1 / 2 }$ , thus $A \mu ^ { \prime } = { \binom { 0 } { 0 } }$

## 6.3 Probability that are reached

In this section, we study the value of the limit probability $\begin{array} { r } { \mu _ { \infty } ^ { \mathcal { T } } ( \varphi ) = \operatorname* { l i m } _ { n  + \infty } \mu _ { n } ^ { \mathcal { T } } ( \varphi ) } \end{array}$ that exists and can be reached by some $K ^ { \# }$ -formula $\varphi .$ . Let us recall Sheppard’s Formula.

▶ Lemma 35 (Sheppard’s Formula [20]). . Let $Z _ { 1 } , Z _ { 2 }$ be standard 1-dimension Gaussian random variables with correlation $E [ Z _ { 1 } Z _ { 2 } ] = \rho \in [ - 1 , 1 ]$ . Then

$$
P r [ Z _ { 1 } \leq 0 , Z _ { 2 } \leq 0 ] = { \frac { 1 } { 2 } } - { \frac { 1 } { 2 } } { \frac { \operatorname { a r c c o s } \rho } { \pi } } .
$$

▶ Proposition 36. For all numbers $x \in [ 0 , 1 / 2 ]$ , there is a sequence of a $K ^ { \# }$ -formula $\varphi _ { 0 } , \varphi _ { 1 } , . . .$ . that are systems with two inequalities such that $\mu _ { \infty } ^ { \mathcal { T } } ( \varphi _ { n } ) \to x$

Proof. Given a matrix A with two rows $A _ { 1 }$ and $A _ { 2 }$ let us compute $P ( A Z \geq 0 )$ . According to Sheppard’s formula, $\begin{array} { r } { P ( A Z \ge 0 ) = \frac { 1 } { 2 } - \frac { 1 } { 2 } \frac { \operatorname { a r c c o s } \rho } { \pi } } \end{array}$ where $\begin{array} { r } { \rho = \frac { A _ { 1 } \cdot A _ { 2 } } { | | A _ { 1 } | | | A _ { 2 } | | } } \end{array}$ . We can make approximate all values for $\rho$ between [−1, 1] with arbitrary big matrices A. Since arccos $\rho \in$ $[ 0 , \pi ]$ , all values of $[ 0 , 1 / 2 ]$ can be approximated by $P ( A Z \geq 0 )$ ◀

The probability $\mu _ { \infty } ^ { \mathcal { T } } ( \varphi )$ over $K \# \cdot$ formulas $\varphi$ can get arbitrarily close to any number in [0, 1] as stressed by the following proposition.

▶ Proposition 37. For all $x \in [ 0 , 1 ]$ , for all $\epsilon > 0$ , there is a $K ^ { \# }$ -formula $\varphi$ such that $| \mu _ { \infty } ^ { \mathcal { T } } ( \varphi ) - x | < \epsilon .$

Proof. We prove the statement using the fact that the set of dyadic rationals $\left\{ { \frac { k } { 2 ^ { m } } } \ | \ k \in \right.$ N $\cup \} , m \in \mathbb { N } , 0 \leq k \leq 2 ^ { m } \}$ is dense in [0, 1]. So for all $x \in [ 0 , 1 ]$ , for all $\epsilon > 0$ , there is $m , k$ such that $\begin{array} { r } { | x - \frac { k } { 2 ^ { m } } | < \epsilon . } \end{array}$ We now give a formula $\varphi$ with $\begin{array} { r } { \mu _ { \infty } ^ { \mathcal { T } } ( \varphi ) = \frac { k } { 2 ^ { m } } } \end{array}$ . Consider

$$
\varphi : = \bigvee _ { j = 1 . . k i = 1 . . m } ( { \# p _ { i , j } \leq \# q _ { i , j } } )
$$

where $p _ { i , j }$ and $q _ { i , j }$ are distinct propositional variables. Since each propositions are independent and $\mu _ { \infty } ^ { \mathcal { T } } ( \# p _ { i , j } \geq \# q _ { i , j } ) = 1 / 2$ , the ANDs product the probabilities and the ORs adds the probabilities up. ◀

▶ Conjecture 38. lim $\mathfrak { l } _ { n \to + \infty } \mu _ { n } ^ { \prime } ( \varphi )$ is equal to some adherence point of $( \mu _ { n } ^ { \mathcal { T } } ( \varphi ) ) _ { n \in \mathbb { N } }$

## 7 Open problems

This work opens the path of studying convergence on general graphs and not only trees, as done in Section 6. We finish our work by listing some open problems.

How to decide when $\mu _ { n } ( \varphi )$ converges? Is it decidable? What is the complexity of that decision problem? Are there some approximation algorithms? These questions are relevant to solve verification problems of GNNs for large graphs. In fact, we conjecture that a fragment, we call majority logic, with a boolean bias condition can have 0-1 law. (for more details, see Appendix)

How to put proportion information in the logic and construct a proof system for reasoning about proportions of models? The answer to this research question may provide a tool for giving justifications of some verification verdict.

How to compute conditioned proportion? For instance how to compute the proportion of φ-models among the ψ-models? This is relevant for more involved verification tasks.

## References

Sam Adam-Day, Michael Benedikt, İsmail İlkan Ceylan, and Ben Finkelshtein. Almost surely asymptotically constant graph neural networks. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 124843–124886. Curran Associates, Inc., 2024. URL: https://proceedings.neurips.cc/paper\_files/paper/2024/file/ e1cd1db6fcd4a6df8618259f5b9cca0f-Paper-Conference.pdf, doi:10.52202/079017-3965.

2 Sam Adam-Day, Theodor-Mihai Iliant, and .Ismail .Ilkan Ceylan. Zero-one laws of graph neural networks. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL: http://papers.nips.cc/paper\_files/paper/ 2023/hash/dfba85bc32a3cb63a96d1412062b4d8e-Abstract-Conference.html.

3 Alberto Apostolico, Gad M. Landau, and Steven Skiena. Matching for run-length encoded strings. In Bruno Carpentieri, Alfredo De Santis, Ugo Vaccaro, and James A. Storer, editors, Compression and Complexity of SEQUENCES 1997, Positano, Amalfitan Coast, Salerno, Italy, June 11-13, 1997, Proceedings, pages 348–356. IEEE, 1997. doi:10.1109/SEQUEN.1997. 666929.

4 Bartosz Bednarczyk, Maja Orlowska, Anna Pacanowska, and Tony Tan. On classical decidable logics extended with percentage quantifiers and arithmetics. In Mikolaj Bojanczyk and Chandra Chekuri, editors, 41st IARCS Annual Conference on Foundations of Software Technology and Theoretical Computer Science, FSTTCS 2021, Virtual Conference, December 15-17, 2021, volume 213 of LIPIcs, pages 36:1–36:15. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2021. URL: https://doi.org/10.4230/LIPIcs.FSTTCS.2021.36, doi:10.4230/LIPICS.FSTTCS.2021.36.

5 Michael Benedikt, Chia-Hsuan Lu, Boris Motik, and Tony Tan. Decidability of graph neural networks via logical characterizations. In Karl Bringmann, Martin Grohe, Gabriele Puppis, and Ola Svensson, editors, 51st International Colloquium on Automata, Languages, and Programming, ICALP 2024, July 8-12, 2024, Tallinn, Estonia, volume 297 of LIPIcs, pages 127:1–127:20. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2024. URL: https://doi. org/10.4230/LIPIcs.ICALP.2024.127, doi:10.4230/LIPICS.ICALP.2024.127.

6 P. Billingsley. Probability and Measure. Wiley Series in Probability and Statistics. Wiley, 1995. URL: https://books.google.fr/books?id=z39jQgAACAAJ.

7 Patrick Blackburn, Maarten de Rijke, and Yde Venema. Modal Logic, volume 53 of Cambridge Tracts in Theoretical Computer Science. Cambridge University Press, 2001. doi:10.1017/ CBO9781107050884.

8 Artem Chernobrovkin, Marco Sälzer, François Schwarzentruber, and Nicolas Troquard. Verifying quantized gnns with readout is decidable but highly intractable. In Renata Wassermann, Marie-Laure Mugnier, and Franz Baader, editors, Proceedings of the 23rd International Conference on Principles of Knowledge Representation and Reasoning, KR 2026, Lisbon, Portugal, July 20-23, 2026, 2026. URL: https://doi.org/10.24963/kr.2026/83, doi:10.24963/KR.2026/83.

9 Ronald Fagin. Probabilities on finite models. J. Symb. Log., 41(1):50–58, 1976. doi:10.1017/ S0022481200051756.

10 Ronald Fagin, Joseph Y. Halpern, and Nimrod Megiddo. A logic for reasoning about probabilities. Inf. Comput., 87(1/2):78–128, 1990. doi:10.1016/0890-5401(90)90060-U.

11 W. Feller. An Introduction to Probability Theory and Its Applications, Volume 2. Number v. 1-2 in An Introduction to Probability Theory and Its Applications. Wiley, 1957. URL: https://books.google.fr/books?id=BsSwAAAAIAAJ.

12 Yu V Glebskii, Dmitry I Kogan, MI Liogon’kiI, and Vladimir A Talanov. Range and degree of realizability of formulas in the restricted predicate calculus. Cybernetics, 5(2):142–154, 1969.

13 Ronald L. Graham, Donald E. Knuth, and Oren Patashnik. Concrete Mathematics: A Foundation for Computer Science, 2nd Ed. Addison-Wesley, 1994. URL: https://www-cs-faculty. stanford.edu/%7Eknuth/gkp.html.

14 Etienne Grandjean. Complexity of the first-order theory of almost all finite structures. Inf. Control., 57(2/3):180–204, 1983. doi:10.1016/S0019-9958(83)80043-6.

15 Joseph Y. Halpern and Bruce M. Kapron. Zero-one laws for modal logic. Ann. Pure Appl. Log., 69(2-3):157–193, 1994. doi:10.1016/0168-0072(94)90084-1.

16 Richard E. Ladner. Polynomial space counting problems. SIAM J. Comput., 18(6):1087–1097, 1989. d i 10 1137/0218073.

17 James F Lynch. Probabilities of first-order sentences about unary functions. Transactions of the American Mathematical Society, 287(2):543–568, 1985.

18 Pierre Nunn, Marco Sälzer, François Schwarzentruber, and Nicolas Troquard. A logic for reasoning about aggregate-combine graph neural networks. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI 2024, Jeju, South Korea, August 3-9, 2024, pages 3532–3540. ijcai.org, 2024. URL: https://www.ijcai.org/ proceedings/2024/391.

19 Pierre Nunn and François Schwarzentruber. A modal logic for explaining some graph neural networks. CoRR, abs/2307.05150, 2023. URL: https://doi.org/10.48550/arXiv.2307. 05150, arXiv:2307.05150, doi:10.48550/ARXIV.2307.05150.

20 Ryan O’Donnell. Analysis of boolean functions. CoRR, abs/2105.10386, 2021. URL: https: //arxiv.org/abs/2105.10386, arXiv:2105.10386.

21 Marco Sälzer, François Schwarzentruber, and Nicolas Troquard. Verifying quantized graph neural networks is pspace-complete. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI 2025, Montreal, Canada, August 16-22, 2025, pages 4660–4668. ijcai.org, 2025. URL: https://doi.org/10.24963/ijcai.2025/519, doi: 10.24963/IJCAI.2025/519.

22 Hazem Torfah and Martin Zimmermann. The complexity of counting models of linear-time temporal logic, 2014. URL: https://arxiv.org/abs/1408.5752, arXiv:1408.5752.

23 Yasmin Tousinejad and Vera Koponen. Random coloured digraphs defined by a markov logic network. CoRR, abs/2606.23715, 2026. URL: https://doi.org/10.48550/arXiv.2606.23715, arXiv:2606.23715, doi:10.48550/ARXIV.2606.23715.

24 Leslie G. Valiant. The complexity of computing the permanent. Theor. Comput. Sci., 8:189–201, 1979. doi:10.1016/0304-3975(79)90044-6.

## Convergence to 0 of $\mu _ { 2 n - 1 } ( \# \varphi _ { e } \geq \# \varphi _ { o } )$

▶ Theorem 39.

$$
\operatorname* { l i m } _ { n \to \infty } { \left( 1 - { \frac { { \binom { 2 n - 1 } { n } } } { 2 ^ { 2 n - 1 } } } \right) } ^ { 2 n - 1 } = 0
$$

Proof. Note that:

$$
\begin{array} { c } { { \binom { 2 n - 1 } { n } = \displaystyle \frac { ( 2 n - 1 ) ! } { n ! \times ( n - 1 ) ! } } } \\ { { = \displaystyle \frac { n } { 2 n } \displaystyle \frac { ( 2 n ) ! } { n ! \times n ! } } } \\ { { = \displaystyle \frac { 1 } { 2 } \binom { 2 n } { n } } } \end{array}
$$

Hence the statement expression becomes

$$
\left( 1 - { \frac { \binom { 2 n - 1 } { n } } { 2 ^ { 2 n - 1 } } } \right) ^ { 2 n - 1 } = \left( 1 - { \frac { \binom { 2 n } { n } } { 2 ^ { 2 n } } } \right) ^ { 2 n - 1 }
$$

Now we approximate the expression asymptotically. Note that by Stirling’s approximation [13], $\begin{array} { r } { \operatorname* { l i m } _ { n  \infty } n ! \sim \sqrt { 2 \pi n } ( \frac { n } { e } ) ^ { n } } \end{array}$ . Hence,

$$
{ \binom { 2 n } { n } } = { \frac { ( 2 n ) ! } { ( n ! ) ^ { 2 } } } \sim { \frac { \sqrt { 4 \pi n } \left( { \frac { 2 n } { e } } \right) ^ { 2 n } } { \left( { \sqrt { 2 \pi n } } \left( { \frac { n } { e } } \right) ^ { n } \right) ^ { 2 } } } = { \frac { 2 ^ { 2 n } } { \sqrt { \pi n } } }
$$

Therefore, as $n  \infty \colon$

$$
{ \frac { { \binom { 2 n } { n } } } { 2 ^ { 2 n } } } = { \frac { 1 } { \sqrt { \pi n } } }
$$

As $n \to \infty$ , let

$$
L _ { n } = \left( 1 - { \frac { { \binom { 2 n } { n } } } { 2 ^ { 2 n } } } \right) ^ { 2 n - 1 } = \left( 1 - { \frac { 1 } { \sqrt { \pi n } } } \right) ^ { 2 n - 1 }
$$

Taking logarithms on both sides:

$$
\ln { \cal L } _ { n } = ( 2 n - 1 ) \ln \left( 1 - \frac { 1 } { \sqrt { \pi n } } \right)
$$

As $\begin{array} { r } { n  \infty , \frac { 1 } { \sqrt { \pi n } }  0 } \end{array}$ . Also, as $x \to 0$ , by Taylor expansion,

$$
\ln ( 1 - x ) = - x + O ( x ^ { 2 } )
$$

Hence,

$$
\begin{array} { r l } & { \underset { n  \infty } { \operatorname* { l i m } } \ln L _ { n } = \underset { n  \infty } { \operatorname* { l i m } } ( 2 n - 1 ) \ln ( 1 - \frac { 1 } { \sqrt { \pi n } } ) } \\ & { \quad \quad = \underset { n  \infty } { \operatorname* { l i m } } ( 2 n - 1 ) ( - \frac { 1 } { \sqrt { \pi n } } + O ( \frac { 1 } { n } ) ) } \\ & { \quad \quad \sim \underset { n  \infty } { \operatorname* { l i m } } - \frac { 2 n - 1 } { \sqrt { \pi n } } } \\ & { \quad \quad \sim \underset { n  \infty } { \operatorname* { l i m } } - \frac { 2 \sqrt { n } } { \sqrt { \pi } } } \\ & { \quad \quad \sim - \infty } \end{array}
$$

Therefore

$$
\operatorname* { l i m } _ { n \to \infty } { \cal L } _ { n } = e ^ { - \infty } = 0
$$

## A Computations

## A.1 Computation of proportion of pointed graphs for $\# p \geq \# q$

The number of graphs of size n is $4 ^ { n } \times 2 ^ { n ^ { 2 } }$

The number of configurations for the successors of a point to satisfy $\# p \geq \# q$

$$
\sum _ { S \subseteq \{ 1 , \dots , n \} } N _ { \geq } ( \# S )
$$

where $N _ { \geq } ( m )$ is the number of ways to put subsets $P , Q$ in a set of size m and with $\# P \geq \# Q$

## A.2 Computation of proportion of pointed graphs for $\# p \geq \# q$

$$
N _ { \geq } ( m ) = \frac { 4 ^ { m } + \binom { 2 m } { m } } { 2 } .
$$

The number of configurations is:

$$
\sum _ { m = 0 } ^ { n } { \binom { n } { m } } \frac { 4 ^ { m } + \binom { 2 m } { m } } { 2 }
$$

We now also have to consider the choices elsewhere. $\mathrm { S o }$ the number of models satisfying $\# p \geq \# q$ is:

$$
\sum _ { m = 0 } ^ { n } \binom { n } { m } \frac { 4 ^ { m } + \binom { 2 m } { m } } { 2 } \times 4 ^ { n - m } \times 2 ^ { ( n - 1 ) n }
$$

So the probability that $\# p \geq \# q$ holds in a pointed graph of size n is:

$$
{ \frac { \sum _ { m = 0 } ^ { n } \left( { \binom { n } { m } } { \frac { 4 ^ { m } + { \binom { 2 m } { m } } } { 2 } } \times 4 ^ { n - m } \times 2 ^ { ( n - 1 ) n } \right. } { 4 ^ { n } \times 2 ^ { n ^ { 2 } } } } .
$$

$$
\sum _ { m = 0 } ^ { n } { \binom { n } { m } } { \frac { 4 ^ { m } + { \binom { 2 m } { m } } } { 4 ^ { m } \times 2 ^ { n + 1 } } } .
$$

## A.3 Computation of proportion of pointed trees for $\# p \geq \# q$

We compute the probability of satisfying $\# p \geq \# q$ in tree model from the root with number of branching at most B.

Also, since truth depends on modal depth, we do not care about depth.

Number of trees of at most B branching with two propositions:

$$
4 \times \sum _ { b = 0 } ^ { B } ( 4 ^ { b } )
$$

From the current node, we just count that given number of branches $b ,$ how many ways can their be valuations such that number of positive $\# p$ is at least $\# q$ just like above:

$$
4 \times \sum _ { b = 0 } ^ { B } \sum _ { P \subseteq [ 0 , b ] } ( \sum _ { Q \subseteq [ 0 , b ] , | P | \geq | Q | } 1 )
$$

By above we have

$$
\sum _ { P \subseteq [ 0 , b ] } ( \sum _ { Q \subseteq [ 0 , b ] , | P | \geq | Q | } 1 ) = { \frac { 4 ^ { b } + { \binom { 2 b } { b } } } { 2 } }
$$

Hence

$$
\mu _ { B } ( \# p \geq \# q ) = \frac { 4 \times \sum _ { b = 0 } ^ { B } \frac { 4 ^ { b } + \binom { 2 b } { b } } { 2 } } { 4 \times \sum _ { b = 0 } ^ { B } \binom { 4 ^ { b } } { b } }
$$

## B A Condition for 0-1 Law for Majority Logic

In this section, we introduce a fragment of $K ^ { \# }$ that we call Majority Logic. The restriction is as follows. The inequations we allow are of the form $\# \varphi \geq \# \psi$ . Majority logic does not have 0-1 law $( \mathrm { e . g . } \mu _ { n } ^ { \mathcal { T } } ( \# p \geq \# q ) \xrightarrow { n \to + \infty } 1 / 2 )$ . But we introduce a constraint, called full Boolean bias, that gives majority logic 0-1 law back.

## B.1 Boolean Bias Condition

The idea is to resolve all the Boolean bias. Given $\mathbb { P } ( p _ { 1 } ) , \ldots , \mathbb { P } ( p _ { n } )$ , we compute the probability of $\mathbb { P } ( \chi )$ for a Boolean formula $\chi$ by assuming that the truth values of $p _ { 1 } , \ldots , p _ { n }$ are independent.

▶ Definition 40. $\mathbb { P } ( p _ { 1 } ) , \ldots , \mathbb { P } ( p _ { n } )$ has full Boolean bias if for all Boolean formulas $\chi _ { 1 } , \chi _ { 2 }$ $i f \mathbb { P } ( \chi _ { 1 } ) = \mathbb { P } ( \chi _ { 2 } )$ then $\chi _ { 1 } \equiv \chi _ { 2 }$

▶ Example 41 (no full Boolean bias). $\operatorname { I f } \mathbb { P } ( p ) = 1 / 3$ and $\mathbb { P } ( q ) = 1 / 4$ , then there is a no full Boolean bias because:

$$
\begin{array} { r l } { = } & { { } \mathbb { P } ( p \wedge q ) = 1 / 3 \times 1 / 4 } \end{array}
$$

$$
\begin{array} { r l } { = } & { { } \mathbb { P } ( p \wedge \neg q ) = 1 / 3 \times 3 / 4 } \end{array}
$$

$$
\begin{array} { r l } { = } & { { } \mathbb { P } ( \neg p \land \neg q ) = 2 / 3 \times 3 / 4 = 1 / 2 ( \mathrm { n o \ b i a s } ) } \end{array}
$$

$$
\begin{array} { r l } { = } & { { } \mathbb { P } ( p \vee q ) = 1 - 1 / 2 = 1 / 2 } \end{array}
$$

▶ Example 42 (full Boolean bias). $\operatorname { I f } \mathbb { P } ( p ) = 1 / 3$ and $\mathbb { P } ( q ) = 1 / 5$ , then there is a full Boolean bias because:

$$
\begin{array} { r l } { = } & { { } \mathbb { P } ( p \wedge q ) = \mathbb { P } ( p ) \times \mathbb { P } ( q ) = 1 / 3 \times 1 / 5 } \end{array}
$$

$$
\begin{array} { r l } { = } & { { } \mathbb { P } ( p \wedge \neg q ) = 1 / 3 \times 3 / 5 } \end{array}
$$

$$
\begin{array} { r l } { = } & { { } \mathbb { P } ( \neg p \land \neg q ) = 2 / 3 \times 3 / 5 = 2 / 5 } \end{array}
$$

$$
\begin{array} { r l } { - } & { { } \mathbb { P } ( p \vee q ) = 1 - 2 / 5 = 3 / 5 } \end{array}
$$

The following is our conjecture:

▶ Conjecture 43. Under the condition full Boolean biais, for all $\varphi$ of majority logic,

$$
\operatorname* { l i m } _ { n  + \infty } \mu _ { n } ( \varphi ) \in \{ 0 , 1 \} .
$$