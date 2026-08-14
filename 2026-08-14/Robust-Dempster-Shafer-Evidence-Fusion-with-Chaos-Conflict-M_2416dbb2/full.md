# Robust Dempster-Shafer Evidence Fusion with Chaos-Conflict Measurement and Historical-Experience Weighting

Huiyu Li<sup>1,2</sup> Weibo Liu<sup>2</sup> Xinru Xu<sup>2</sup> Dongchen Gao<sup>2</sup> Meng Zhang<sup>2</sup> Junhua Hu<sup>1</sup>

<sup>1</sup>School of Business, Central South University, Changsha, Hunan, People’s Republic of China, 410083   
<sup>2</sup>School of Management, Shandong University, Jinan, Shandong, People’s Republic of China, 250100

Corresponding author: Junhua Hu (hujunhua@csu.edu.cn)

## Abstract

Multi-source evidence fusion under Dempster-Shafer theory faces two persistent challenges: existing conflict measures assess inter-evidence inconsistency and intra-evidence uncertainty independently, yielding incomplete evaluations, and current fusion methods evaluate evidence sources exclusively through instantaneous comparisons without exploiting their long-term reliability across diverse decision contexts. This paper proposes a unified evidence reasoning framework that addresses both limitations. Specifically, a chaos-conflict measurement is introduced to jointly quantify cross-evidence conflict and intra-evidence non-specificity, with five formally proven properties ensuring consistent assessment. A historical experience driven weighting scheme partitions the decision space via spectral clustering and applies regret theory to compute context-specific reliability profiles from past fusion outcomes. These mechanisms feed into a hybrid combination rule that adaptively balances uncertainty preservation against weighted consensus, controlled by the global conflict level, followed by a belief-interval decision strategy that enables robust classification without discarding epistemic uncertainty. Experiments on 16 real-world benchmark datasets demonstrate that the proposed framework achieves an average F1 score of 85.78 and a mean AUC of 93.30, outperforming eight DST-based baselines and three gradient boosting methods. Ablation analysis confirms the contribution of each component we proposed. The framework ofers an efective approach for adaptive evidence fusion in multi-source decision making.

## 1 Introduction

Multi-source information fusion plays a critical role in modern decision-making systems, where heterogeneous evidence must be integrated to support reliable judgments under uncertainty (El-Din et al., 2024; Xiao et al., 2026). Applications ranging from industrial fault diagnosis to pattern recognition and environmental monitoring share a common challenge: multiple evidence sources often provide assessments that are incomplete, imprecise, and mutually conflicting (Li et al., 2026; Qiao, Fan, et al., 2023; Yu et al., 2026). The ability to fuse such evidence while preserving uncertainty information remains a fundamental problem in the fields of information systems and decision science.

Dempster-Shafer theory (DST) provides a generalization of Bayesian probability that accommodates partial belief and ignorance through the assignment of basic probability assignments (BPAs) to subsets of a hypothesis space (Radzvilas et al., 2024). Unlike classical probability, DST assigns mass to multi-element subsets, thereby explicitly representing epistemic uncertainty. Despite its theoretical elegance, DST faces well-documented challenges when evidence sources exhibit high conflict (Calderwood et al., 2016; Z. Zhou & Xiao, 2026). The Zadeh paradox (Zadeh, 1986) demonstrated that Dempster’s rule can produce counterintuitive results when two sources strongly disagree, and this limitation has motivated decades of research into conflict-aware fusion strategies.

Two broad categories of solutions have emerged. The first modifies the combination rule itself: Yager (1987) reassigns conflict mass to the frame of discernment, Dubois and Prade (1988) propose a disjunctive rule that preserves uncertainty rather than normalizing it away, and various authors have introduced weighted, cautious, or transferable belief models to handle specific failure modes. The second category addresses the evidence sources directly, applying distance-based or similarity-based weighting schemes before fusion. Murphy (2000) averages BPAs uniformly; Deng et al. (2004) weights sources using Jousselme distance; subsequent methods incorporate entropy, trust, or reliability coeficients to discount suspect evidence. Recent studies have attempted to combine preprocessing with adaptive fusion strategies. While these approaches have achieved incremental improvements, two persistent limitations remain. First, existing conflict measures (Destercke & Burger, 2013) treat uncertainty and conflict as independent dimensions, yielding assessments that are either incomplete or redundant. Second, all of the above methods evaluate evidence solely on the basis of instantaneous BPA comparisons, ignoring the long-term track record of evidence sources across diverse decision contexts (Xu et al., 2016).

These limitations are consequential in practice. Existing models predict that evidence sources operating under similar conditions should exhibit consistent reliability (Huang et al., 2025), yet empirical observation reveals the opposite: a source that produces misleading assessments in one scenario may be highly dependable in another, depending on the characteristics of the specific decision context (Gao & Pan, 2025). Methods that assign static weights or that evaluate evidence exclusively through current pairwise comparisons cannot capture such context-dependent reliability (Park, 2025), leading to two failure modes. Informative sources that are occasionally conflicting but generally dependable are indiscriminately suppressed, while systematically unreliable sources receive insuficient penalization (Ghorbanzadeh et al., 2021; Zhang et al., 2025). Furthermore, conventional conflict measures such as Dempster’s conflict coeficient K consider only the total mass assigned to empty intersections (Urbani et al., 2023), disregarding the internal structure of multi-element focal elements. This simplification leads to coarse assessments of evidence quality, particularly when sources allocate substantial mass to composite hypotheses that express genuine ignorance rather than true disagreement.

This paper proposes a unified evidence reasoning framework that addresses both limitations through two complementary mechanisms. The first is a chaos-conflict measurement (CCM) that jointly evaluates cross-evidence association and intra-evidence non-specificity within a single scalar quantity. The CCM is grounded in a novel evidence similarity measure whose properties of boundedness, symmetry, monotonicity, extreme consistency, and refinement insensitivity are formally proven. The second mechanism is a historical-experience-driven evidence weighting scheme that exploits past decision outcomes to learn context-dependent reliability profiles. Spectral clustering partitions the decision space into heterogeneous contexts, and regret theory is employed to quantify, for each evidence source, the counterfactual contribution when a fusion decision deviates from ground truth. Aggregated regret and rejoice scores are normalized to produce context-specific weights that reflect long-term source reliability. These two mechanisms feed into a hybrid combination rule that adaptively balances a conservative uncertainty preservation term against a weighted consensus component, with the tradeof controlled by the global conflict level. A belief-interval decision rule then scores each hypothesis by combining its belief lower bound with a stability-weighted plausibility upper bound, enabling robust classification without forcing artificial redistribution of mass from multi-element focal elements.

The proposed framework is evaluated on 16 real-world benchmark datasets spanning biology, medicine, geography, and demography, using decision trees as evidence sources. Compared against eight DST-based fusion methods and three gradient boosting baselines, the framework achieves the best average rank across all evaluation metrics. Robustness analyses demonstrate stable performance under feature noise, label noise, hybrid noise scenarios, varying numbers of evidence sources, and substitution of the base evidence generator. Ablation experiments confirm that historical experience weighting provides the largest individual contribution to performance, reducing AUC by 5.03% when removed. These results establish the proposed framework as an efective approach for evidence fusion that balances uncertainty preservation with adaptive reliability assessment, with direct applicability to any multi-source decision setting where evidence sources operate repeatedly across heterogeneous contexts.

The remainder of this paper is organized as follows. Section 2 reviews the preliminary concepts of Dempster-Shafer theory, regret theory, and spectral clustering. Section 3 presents the proposed framework, including the chaos-conflict measurement, the historical experience driven weighting mechanism, the hybrid combination rule, and the belief-interval decision rule. Section 4 details the algorithmic design of the ofline training and online inference phases. Section 5 reports the experimental setup, comparative results, robustness analyses, parameter sensitivity, and ablation study. Section 6 concludes the paper with a summary of contributions, a discussion of limitations, and directions for future research.

## 2 Preliminaries

## 2.1 Dempster-Shafer theory

DST, also known as belief function theory, is a mathematical framework for representing and reasoning with uncertain, imprecise, and incomplete information. Originating from the work of Dempster (1967) and formalized by Shafer (1976), DST extends classical probability theory by allowing probability masses to be assigned not only to singletons but also to subsets of the hypothesis space. This provides a richer and more flexible representation of uncertainty, especially when prior knowledge is limited or heterogeneous evidence sources yield conflicting assessments.

Definition 1. Frame of Discernment (FoD). Let $\begin{array} { r l } { \Theta } & { { } = } \end{array}$ $\{ \theta _ { 1 } , \theta _ { 2 } , \cdots , \theta _ { i } , \cdots , \theta _ { n } \}$ denote the frame of discernment, $\mathrm { i . e . , a }$ set of mutually exclusive and collectively exhaustive hypotheses. DST operates on the power set:

$$
\begin{array} { c } { { 2 ^ { \Theta } = \Big \{ \emptyset , \ \{ \theta _ { 1 } \} , \ \{ \theta _ { 2 } \} , \ \{ \theta _ { 3 } \} , \ \dots , } } \\ { { \{ \theta _ { 1 } , \theta _ { 2 } \} , \ \{ \theta _ { 1 } , \theta _ { 3 } \} , \ \dots , \ \{ \theta _ { 1 } , \theta _ { 2 } , \theta _ { 3 } \} , \ \dots , \ \Theta \Big \} } } \end{array}\tag{1}
$$

where each subset represents a proposition whose truth is uncertain. Here, ∅ is an empty set. ${ \bf \bar { \Phi } } _ { \mathrm { I f } } ^ { \bullet } S \in 2 ^ { \Theta } , S$ is called a hypothesis.

Definition 2. Basic Probability Assignment. A BPA or mass function m : $2 ^ { \Theta }  [ 0 , 1 ]$ satisfies:

$$
\left\{ \begin{array} { l l } { \displaystyle \sum _ { S \subseteq 2 ^ { \Theta } } m \left( S \right) = 1 } \\ { m \left( \emptyset \right) = 0 } \end{array} \right.\tag{2}
$$

A subset S with $m \left( S \right) ~ > ~ 0$ is called a focal element (FE). Mass assigned to multi-element sets (N-Focal Elements, NFEs) expresses epistemic uncertainty or ignorance, since it indicates that the exact hypothesis cannot be precisely identified. Conversely, single-element focal elements (SFEs) represent precise hypotheses.

Definition 3. Belief and plausibility functions. Based on BPA, DST defines two dual measures that jointly describe the degree of support for a proposition.

The belief function, representing the minimal support for S :

$$
\operatorname { B e l } \left( S \right) = \sum _ { A \subseteq S } m \left( A \right)\tag{3}
$$

The plausibility function, representing the maximal potential support for S :

$$
P l \left( S \right) = 1 - B e l \left( \bar { S } \right) = \sum _ { A \cap S \neq \emptyset } m \left( A \right) , A \subseteq \Theta\tag{4}
$$

Together, [Bel (S ) Pl (S )] forms the belief interval, which provides a natural expression of uncertainty. A wider interval indicates greater ambiguity, while a narrow interval signifies strong confidence.

Definition 4. Dempster’s combination rule. One of DST’s core strengths is its ability to integrate independent pieces of evidence. Given n mass functions $m _ { 1 } , m _ { 2 } , \ldots , m _ { n }$ defined on the same frame, Dempster’s rule combines them as

$$
\left\{ \begin{array} { l l } { \displaystyle m _ { 1 \oplus 2 \oplus . . . \oplus n } \left( S \right) = \frac { \sum _ { i = S , A _ { i } \subseteq 2 ^ { \Theta } } \prod _ { j } m _ { j } \left( A _ { i } \right) } { 1 - K } } \\ { K = \displaystyle \sum _ { \bigcap A _ { i } = \emptyset , A _ { i } \subseteq 2 ^ { \Theta } } m _ { 1 } \left( A _ { 1 } \right) \cdot m _ { 2 } \left( A _ { 2 } \right) \cdot \cdot \cdot m _ { n } \left( A _ { n } \right) } \end{array} \right.\tag{5}
$$

where ⊕ denotes Dempster’s orthogonal sum, which aggregates independent pieces of evidence through combination. $K \in$ [0 1] is the conflict coeficient. A higher K implies stronger disagreement between evidence sources. While Dempster’s rule is elegant and widely used, its behavior in high-conflict situations may yield counterintuitive results, motivating a large body of research on conflict management and evidence fusion strategies.

## 2.2 Regret theory

Regret theory (RT) explains choice under uncertainty by embedding the anticipatory emotions of regret and rejoicing. Whereas expected utility theory treats agents as evaluators of absolute payofs, RT posits that people also compare the realized outcome with what they would have received had they chosen diferently. Regret arises when the forgone alternative would have delivered a superior outcome; rejoicing emerges when it would have delivered an inferior one.

Definition 5. Basic formulation of RT. Let $A C _ { 1 }$ and $A C _ { 2 }$ be two possible actions. If $A C _ { 1 }$ yields outcome x<sub>1</sub> and $A C _ { 2 }$ yields outcome $x _ { 2 }$ , the decision maker (DM) does not merely consider the utility values $u \left( x _ { 1 } \right)$ and $u \left( x _ { 2 } \right)$ but also the psychological diference between them. The utility obtained by choosing action $A C _ { 1 }$ and thereby forgoing $A C _ { 2 }$ is (Loomes & Sugden, 1982):

$$
\nu = u \left( x _ { 1 } \right) + r \left( u \left( x _ { 1 } \right) - u \left( x _ { 2 } \right) \right)\tag{6}
$$

where $u \left( \bullet \right)$ is a monotonic utility function, $r ( \Delta u )$ is an increasing and typically S-shaped regret-rejoice function and is defined as follow:

Definition 6. Regret-rejoice function. The utility function commonly used in regret theory is

$$
u \left( x _ { i } \right) = \frac { 1 - e ^ { - \theta x _ { i } } } { \theta }\tag{7}
$$

where $\theta \in ( 0 , 1 )$ represents the DM’s degree of risk aversion. A widely used parametric representation for regret-rejoice function (Bleichrodt et al., 2010) is:

$$
r \left( \Delta u \right) = 1 - e ^ { - \delta \Delta u }\tag{8}
$$

where $\delta \in ( 0 , + \infty )$ denotes the regret-rejoice sensitivity coeficient.

Within this study, RT is repurposed to audit the past behavior of evidence sources. Whenever the fused decision is wrong, each contributor is retrospectively credited with either a regret or a rejoice score: regret if its testimony steered the ensemble away from the truth, rejoice if it pulled the ensemble toward the truth. Aggregating these scores across historical instances quantifies the long-run reliability of every evidence source.

## 2.3 Spectral clustering

Spectral clustering (SC) treats grouping as a graph-partitioning problem: it recovers the intrinsic geometry of data from the eigenvectors of an afinity matrix rather than from distances (Ng et al., 2001). The algorithm readily separates strongly connected vertex subsets that are only weakly linked to the rest of the graph, giving it a decisive advantage over conventional methods such as k-means when the underlying distribution is high-dimensional (Cai & Ma, 2022).

Definition 7. Graph construction. Given a dataset $\begin{array} { r l } { X } & { { } = } \end{array}$ $\{ x _ { 1 } , x _ { 2 } , \ldots , x _ { n } \}$ , each data instance is treated as a vertex in an undirected weighted graph $G = \langle V , E , W \rangle$ . Edges encode pairwise similarity, and the similarity matrix $W ~ = ~ \left[ w _ { i j } \right]$ is constructed through a Gaussian kernel:

$$
w _ { i j } = \exp \left( - \frac { \left\| x _ { i } - x _ { j } \right\| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } \right)\tag{9}
$$

where σ is a scale parameter controlling locality.

Definition 8. Graph Laplacian Formulation. Given the afinity matrix W ∈ R<sup>n×n</sup> constructed in Definition 7, define the degree matrix D as a diagonal matrix with entries $D _ { i i } = \sum _ { j } w _ { i j }$ . The pair (D W) induces related Laplacian operators that encode the structural properties of the graph. The symmetric normalized Laplacian is given by

$$
L = I - D ^ { - 1 / 2 } W D ^ { - 1 / 2 }\tag{10}
$$

Definition 9. Graph Cut Objective. Given a weighted graph $G =$ (V E W) , clustering can be viewed as partitioning the vertex set into k disjoint subsets $g _ { 1 } , g _ { 2 } , \ldots , g _ { k }$ . A desirable partition should minimize the similarity between diferent subsets while maintaining high within-cluster afinity. To formalize this principle, Shi and Malik (2000) introduced the Normalized Cut (Ncut) criterion, defined for a k -way partition as

$$
N C u t _ { q } = \operatorname* { m i n } _ { g _ { 1 } , g _ { 2 } , . . . , g _ { k } \subseteq V } \frac { 1 } { 2 } \sum _ { i = 1 } ^ { k } \frac { W _ { a } \left( g _ { i } , \overline { { { g _ { i } } } } \right) } { \operatorname { v o l } \left( g _ { i } \right) }\tag{11}
$$

where $W \left( g _ { i } , \overline { { g _ { i } } } \right) = \sum _ { u \in g _ { i } } \sum _ { \nu \in g _ { i } } w _ { u \nu }$ denotes the total weight of edges crossing the boundary of $g _ { i }$ , and vol $( g _ { i } ) = \sum _ { u \in g _ { i } } D _ { u u }$ is the degree mass of subset $g _ { i }$ . The Ncut criterion balances the cut value by the size of each subset, thereby preventing trivial solutions in which a small set is separated from the rest of the graph.

## 3 Proposed approach

## 3.1 Overview

We propose a DST evidence reasoning framework that combines chaos-conflict measurement with historical-experience weighting for robust multi-source decision making. The framework operates in two phases: ofline training and online inference, as illustrated in Figure 1.

In the ofline phase, historical BPAs from multiple evidence sources are processed along two parallel paths. SC partitions the decision space into context scenarios based on Sample feature representations. Meanwhile, Dempster’s rule fuses the historical BPAs to identify decision failures. For each failure, the regret-rejoice mechanism evaluates individual evidence sources: the optimal evidence receives a rejoice score proportional to its conflict with the erroneous fusion, while each candidate erroneous source receives a regret score proportional to the distortion it introduces. Aggregated regret and rejoice scores are normalized via softmax to produce context-specific historical experience weights.

In the online phase, a test sample generates BPAs from the same evidence sources. The chaos-conflict measurement computes pairwise association and similarity across all evidence pairs, yielding a global chaos-conflict degree that captures both inter-evidence inconsistency and intra-evidence uncertainty. The test BPAs are then weighted using the context-indexed historical weights retrieved from ofline training. A hybrid combination rule adaptively balances a conservative Dubois term with the weighted consensus evidence, controlled by the global conflict level. When conflict is high, the rule emphasizes uncertainty preservation; when conflict is mild, it relies on the consensus term. Finally, a belief-interval decision rule scores each hypothesis by combining its belief lower bound with a stability-weighted plausibility upper bound, outputting the predicted class label.

## 3.2 Chaos-conflict measurement

In multi-source evidence fusion, conflict reflects discrepancies between evidence sources, whereas uncertainty stems from NFEs. Existing measures treat them independently, yielding incomplete assessments: conflict metrics overlook internal ambiguity, while uncertainty metrics fail to capture cross-evidence inconsistency. To better characterize overall evidential stability, we propose a unified measurement named chaos-conflict measurement (CCM). By jointly evaluating FE set’s compatibility and intrinsic uncertainty, CCM ofers a more coherent basis for fusion.

Definition 10. Evidence association measure. Let $m _ { i }$ and $m _ { j }$ be two mass functions defined on the same FoD $\Theta = \{ \theta _ { 1 } , \theta _ { 2 } , . . . , \tilde { \theta } _ { n } \}$ , denote by $\{ S _ { 1 } , \ldots , S _ { 2 ^ { n } } \}$ the set of all focal subsets of Θ . The pairwise association measure between $m _ { i }$ and $m _ { j }$ is defined as

$$
k \left( m _ { i } , m _ { j } \right) = \sum _ { i = 1 } ^ { 2 ^ { n } } \sum _ { j = 1 } ^ { 2 ^ { n } } 2 \frac { m \left( S _ { i } \right) m \left( S _ { j } \right) \left| S _ { i } \cap S _ { j } \right| } { \left| S _ { i } \right| \left| S _ { j } \right| \left( \left| S _ { i } \right| + \left| S _ { j } \right| \right) }\tag{12}
$$

$$
= \sum _ { S _ { i } \cap S _ { j } \neq \emptyset } \frac { m \left( S _ { i } \right) } { \left| S _ { i } \right| } \cdot \frac { m \left( S _ { j } \right) } { \left| S _ { j } \right| } \cdot \frac { 2 \left| S _ { i } \cap S _ { j } \right| } { \left| S _ { i } \right| + \left| S _ { j } \right| }\tag{13}
$$

Corollary 1. For a collection of mass functions $\{ m _ { 1 } , \hdots , m _ { J } \}$ defined on the same FoD Θ , their global association measure is given by

$$
k \left( \underset { j = 1 } { \overset { J } { \oplus } } m _ { j } \right) = \sum _ { \cap S _ { i } \neq \emptyset } \left( \prod _ { j } \frac { m _ { j } ( S _ { i } ) } { | S _ { i } | } \right) \frac { \left| \bigcap _ { i } S _ { i } \right| } { \sum _ { i } | S _ { i } | }\tag{13a}
$$

Corollary 1 follows immediately from Definition 10 by extending the pairwise association to all mass functions on the frame. It aggregates compatible FEs across all evidence sources.

Remark 1. The global association measure incorporates both the non-specificity ofNFEs and their compatibility structure. In particular, mass assigned to NFEs reflects epistemic uncertainty or ignorance, which weakens the reliability ofthe corresponding BPA. Theformulation therefore down-weights such contributions and adjusts the association strength according to the degree of focal-set overlap. As a result, the measure captures both the uncertainty in NFE allocations and the compatibility across evidence sources, yielding a more faithful characterization of their mutual association.

Definition 11. Evidence similarity. For two mass functions m<sub>i</sub> and $m _ { j }$ defined on the same Θ , their similarity is quantified by:

$$
S \left( m _ { i } , m _ { j } \right) = \frac { k \left( m _ { i } , m _ { j } \right) } { 1 - k \left( m _ { i } , m _ { j } \right) + k \left( m _ { i } , m _ { i } \right) k \left( m _ { j } , m _ { j } \right) }\tag{14}
$$

where S (·) denotes the similarity between the two mass functions. Before establishing the main properties of the similarity measure in Definition 11, we first introduce auxiliary lemma that facilitates the subsequent proofs.

Lemma 1. Cauchy-Schwarz inequality. For any real vectors $\boldsymbol { x } = ( x _ { 1 } , \ldots , x _ { n } ) ^ { T }$ and $y = ( y _ { 1 } , \ldots , \bar { y } _ { n } ) ^ { T }$ , thefollowing inequality holds:

$$
{ \biggl ( } \sum _ { i = 1 } ^ { n } x _ { i } y _ { i } { \biggr ) } ^ { 2 } \leq { \biggl ( } \sum _ { i = 1 } ^ { n } x _ { i } ^ { 2 } { \biggr ) } { \biggl ( } \sum _ { i = 1 } ^ { n } y _ { i } ^ { 2 } { \biggr ) }\tag{15}
$$

with equality if and only ${ \mathrm { i f ~ } } x = \lambda y$ for some scalar $\lambda \in R$

![](images/e425e314c01e3352765021b911b16150eb17dacecd44cd6fa1722fbd4eaab522.jpg)  
Figure 1: Overall architecture of the proposed framework.

Proof. The result is classical and follows directly from the fact that the quadratic form

$$
\sum _ { i = 1 } ^ { n } ( x _ { i } - \lambda y _ { i } ) ^ { 2 } \geq 0
$$

holds for all λ .

Lemma 2. Positive semi-definiteness ofthe compatibility matrix. Let

$$
\bar { M } _ { i j } = \frac { 2 \left| S _ { i } \cap S _ { j } \right| } { \left| S _ { i } \right| \left| S _ { j } \right| \left( \left| S _ { i } \right| + \left| S _ { j } \right| \right) } , \ i , j = 1 , \ldots , 2 ^ { \left| \Theta \right| } ,
$$

then $\bar { M }$ is a real symmetric positive semi-definite matrix.

Proof. We begin by fixing an ordering of the power set $2 ^ { \Theta } , 2 ^ { \Theta } =$ $\{ S _ { 1 } , \bar { S } _ { 2 } , \ldots , \bar { S _ { 2 ^ { \vert { \Theta } \vert } } } \}$

It is immediate that $\bar { M } _ { i j } \ = \ \bar { M } _ { j i }$ , hence M<sup>¯</sup> is real and symmetric. Consider any nonzero vector ${ \boldsymbol { z } } = ( z _ { 1 } , \dots , z _ { 2 ^ { | \Theta | } } ) ^ { T }$ the associated quadratic form is

$$
z ^ { T } { \bar { M } } z = \sum _ { i = 1 } ^ { 2 ^ { \left| 0 \right| } } \sum _ { j = 1 } ^ { 2 ^ { \left| 0 \right| } } z _ { i } z _ { j } { \frac { 2 \left| S _ { i } \cap S _ { j } \right| } { \left| S _ { i } \right| \left| S _ { j } \right| \left( \left| S _ { i } \right| + \left| S _ { j } \right| \right) } } .
$$

Because every focal element is compatible with itself, we have

$$
\left| S _ { i } \cap S _ { i } \right| = \left| S _ { i } \right| > 0 \Rightarrow { \bar { M } } _ { i i } = { \frac { 2 \left| S _ { i } \right| } { \left| S _ { i } \right| \left| S _ { i } \right| ( \left| S _ { i } \right| + \left| S _ { i } \right| ) } } = { \frac { 1 } { \left| S _ { i } \right| ^ { 2 } } } > 0 .
$$

Hence each diagonal entry is strictly positive. To show that M<sup>¯</sup> is positive semi-definite, we substitute the identity $\begin{array} { r } { \frac { 2 } { | S _ { i } | + | S _ { j } | } = } \end{array}$ $2 \int _ { 0 } ^ { \infty } e ^ { - \left( | S _ { i } | + | S _ { j } | \right) t } d t$ into the quadratic form, which yields

$$
z ^ { T } \bar { M } z = 2 \int _ { 0 } ^ { \infty } \sum _ { \theta \in \Theta } \left( \sum _ { i \ni \theta } z _ { i } \frac { e ^ { - | S _ { i } | t } } { | S _ { i } | } \right) ^ { 2 } d t \geq 0
$$

for every vector $z ,$ , because the integrand is a sum of squares. Hence $z ^ { T } \bar { M } \bar { z } \ge 0$ for all $z ,$ which proves that M<sup>¯</sup> is positive semi-definite.

We adopted the set of attributes that a conflict measure ought to satisfy as stipulated by Destercke and Burger (2013) and supply the corresponding proof below.

Property 1. Boundedness, $0 \leq S \left( m _ { i } , m _ { j } \right) \leq 1$

Proof. Since $\bar { M }$ is positive semi-definite, there exists a lower-triangular matrix $L$ such that $\bar { M } = L \bar { L } ^ { \top }$ . Define ${ \textbf { x } } =$ $\mathbf { v } L , \mathbf { y } = \mathbf { b } L$ . Then $\begin{array} { r } { k \left( m _ { i } , m _ { j } \right) = 2 \mathbf { v } L L ^ { \top } \mathbf { b } ^ { \top } = 2 \mathbf { x y } ^ { \top } , k \left( m _ { i } , m _ { i } \right) = } \end{array}$ 2xx<sup>⊤</sup> $k \left( m _ { j } , m _ { j } \right) = 2 \mathbf { y } \mathbf { y } ^ { \top }$ . Set $a = \mathbf { x y } ^ { \top } , b = \mathbf { x x } ^ { \top } , c = \mathbf { y y } ^ { \top }$ . Then

$$
S \left( m _ { i } , m _ { j } \right) = \frac { 2 a } { 1 - 2 a + 4 b c } .
$$

By construction, all entries of M<sup>¯</sup> and of the mass vectors are non-negative, hence $a , b , c \geq 0$ . Moreover $0 \leq k \left( m _ { i } , m _ { j } \right) \leq 1$ and $k \left( m _ { i } , m _ { j } \right) = 2 a$ imply $a \in [ 0 , 1 / 2 ]$ . Thus the denominator satisfies $1 - \dot { 2 } a + 4 b c \geq 1 - 2 a \geq 0$ , and the numerator $2 a \geq 0 \ .$ therefore $S \left( m _ { i } , m _ { j } \right) \geq 0$

By Lemma 1, $a ^ { 2 } = \left( \mathbf { x y } ^ { \top } \right) ^ { 2 } \leq \left( \mathbf { x x } ^ { \top } \right) \left( \mathbf { y y } ^ { \top } \right) = b c$ . Let $u = b c \ge 0$ and $0 \leq a \leq \sqrt { u }$ . To show $S \left( m _ { i } , m _ { j } \right) \leq 1$ , it sufices to prove $2 a \leq 1 - 2 a + 4 b c \Leftrightarrow 0 \leq u - \sqrt { u } + 1 / 4$ Consider the function $f ( u ) = u - \sqrt { u } + 1 / 4$ , Its derivative is

$$
f ^ { \prime } \left( u \right) = 1 - \frac { 1 } { 2 \sqrt { u } } , u \in \left( 0 , 1 \right] .
$$

The equation $f ^ { \prime } \left( u \right) = 0$ has the unique solution $u ^ { * } = 1 / 4$ . Since $f ( \bar { 1 } / 4 ) = \bar { 0 }$ , by continuity we obtain $f \left( u \right) \geq 0$ ∀u ∈ $[ 0 , 1 ]$ . Therefore S $\left( m _ { i } , m _ { j } \right) \leq 1$

Property 2. Symmetry, $S \left( m _ { i } , m _ { j } \right) = S \left( m _ { j } , m _ { i } \right)$

Proof.

$$
S \left( m _ { i } , m _ { j } \right) = \frac { k \left( m _ { i } , m _ { j } \right) } { 1 - k \left( m _ { i } , m _ { j } \right) + k \left( m _ { i } , m _ { i } \right) \cdot k \left( m _ { j } , m _ { j } \right) }
$$

$$
\begin{array} { c } { { \displaystyle = \sum _ { i = 1 } ^ { 2 ^ { n } } \sum _ { j = 1 } ^ { 2 ^ { n } } 2 \frac { m \left( S _ { i } \right) m \left( S _ { j } \right) \left| S _ { i } \cap S _ { j } \right| } { \left| S _ { i } \right| \left| S _ { j } \right| \left( \left| S _ { i } \right| + \left| S _ { j } \right| \right) } } } \\ { { \displaystyle = \sum _ { i = 1 } ^ { 2 ^ { n } } \sum _ { j = 1 } ^ { 2 ^ { n } } 2 \frac { m \left( S _ { j } \right) m \left( S _ { i } \right) \left| S _ { j } \cap S _ { i } \right| } { \left| S _ { j } \right| \left| S _ { i } \right| \left( \left| S _ { j } \right| + \left| S _ { i } \right| \right) } = S \left( m _ { j } , m _ { i } \right) } } \end{array}
$$

Thus, the similarity measure is symmetric.

Property 3. Monotonicity. Let $m _ { 1 }$ and $m _ { 2 }$ be two mass functions defined on the same FoD Θ . Without loss of generality, assume $k \left( m _ { 2 } , m _ { 2 } \right) \ \geq \ k \left( m _ { 1 } , m _ { 1 } \right)$ . For $t ~ \in ~ [ 0 , 1 ]$ , consider the convex combination $m _ { 3 } \left( t \right) ~ = ~ \left( 1 - t \right) m _ { 2 } + t m _ { 1 }$ , define $\zeta ( t ) : = S \left( m _ { 1 } , m _ { 3 } \left( t \right) \right)$ . Then $\zeta \left( t \right)$ is non-decreasing on $[ 0 , 1 ]$ Moreover, if $m _ { 1 } \neq m _ { 2 }$ , then $\zeta \left( t \right)$ is strictly increasing on (0 1) .

Proof. By Property 1, we have

$$
\frac { \partial S } { \partial a } = \frac { 2 + 8 b c } { ( 1 - 2 a + 4 b c ) ^ { 2 } } > 0 .
$$

Thus, $S \left( m _ { i } , m _ { j } \right)$ is a strictly increasing transform of $k \left( m _ { i } , m _ { j } \right)$ . Meanwhile,

$$
k \left( m _ { 1 } , m _ { 3 } \left( t \right) \right) = 2 a \left( t \right) \propto x y ( t ) ^ { \top } = x ( ( 1 - t ) y + t x ) ^ { \top }
$$

$$
= ( 1 - t ) x y ^ { \top } + t x x ^ { \top } = ( 1 - t ) a + t b = \langle m _ { 1 } , ( 1 - t ) m _ { 2 } + t m _ { 1 } \rangle _ { D } .
$$

Diferentiating gives $a ^ { \prime } \left( t \right) = b - a$ . Because $a \leq { \sqrt { b c } }$ and $b \geq c , a ^ { \prime } \left( t \right) > ^ { ^ { \ast } } 0 , \ m _ { i } \neq m _ { j }$ . The composition $s \circ a$ is strictly increasing. Consequently, $t _ { 1 } < t _ { 2 } \Rightarrow S \left( m _ { 1 } , m _ { 3 } \left( t _ { 1 } \right) \right) <$ $S \left( m _ { 1 } , m _ { 3 } \left( t _ { 2 } \right) \right)$

Property 4. Extreme Consistency. (1) If and only if $m _ { i } = m _ { j }$ and the BPAs of $m _ { i }$ and $m _ { j }$ are all concentrated on the same SFE, $S \left( m _ { i } , m _ { j } \right) = 1 \ ; ( 2 ) S \left( m _ { i } , m _ { j } \right) = 0 \Leftrightarrow ( \bigcup S _ { i } ) \cap \left( \bigcup S _ { j } \right) = \emptyset .$

Proof. For (1), by Property 1, we have $a = 1 / 4 + b c$ and $a ^ { 2 } \leq b c$ . When $u ^ { * } = 1 / \dot { 4 }$ , we have $a = b = c = 1 / 2$ , then $x = y$ and $k \left( m _ { i } , m _ { j } \right) = k \left( m _ { i } , m _ { i } \right) = k \left( m _ { j } , m _ { j } \right) = 1$ , condition $m _ { i } = m _ { j }$ is satisfied. Due to $\operatorname { E q . 2 } ,$ , let the unique FE be A , then their masses are $m _ { i } ( A ) = m _ { i } ( A ) = 1$ . If A is a NFE, $( 1 / \left| A \right| ) ^ { 2 } < 1$ , which contradicts $k \left( m _ { i } , m _ { j } \right) = 1$ . Thus A must be a SFE. For (2), by the Property $1 , S \left( m _ { i } , m _ { j } \right) = 0 \Leftrightarrow a = 0$ . Due to the construction, we have $a \propto \sum _ { S _ { i } \cap S _ { j } \neq \emptyset } \mathop { m _ { i } } \left( S _ { i } \right) \mathstrut m _ { j } \left( S _ { j } \right) \left| S _ { i } \cap S _ { j } \right| / \left( \left| S _ { i } \right| + \left| S _ { j } \right| \right)$ and all terms in the summation are nonnegative. Therefore a can be zero only $\mathrm { i f } \left| S _ { i } \cap S _ { j } \right| = 0$ , which is equivalent to

$$
\left( \cup S _ { i } \right) \cap \left( \cup S _ { j } \right) = \emptyset .
$$

Property 5. Refinement Insensitivity. When the FoD is refined from Θ to $\Theta ^ { \prime } , S \left( m _ { i } , m _ { j } \right)$ remains unchanged.

Proof. A refinement of the frame preserves the structure of both mass functions. Let $S \subseteq \Theta ^ { \prime }$ denote such a refined FE. Under refinement, the masses satisfy m<sub>i</sub> $( S ) = m _ { j } ( S ) = 0$ and these terms appear symmetrically in the computation of $k \left( m _ { i } , m _ { j } \right)$ , contributing additively as zeros. Removing these null contributions does not afect any terms involved in the definition of the similarity measure. Consequently, the value of $S \left( m _ { i } , m _ { j } \right)$ is invariant under refinement of the frame.

To further illustrate the proposed similarity measure, we consider two numerical examples.

Example 1. Three BPA functions $m _ { 1 } , m _ { 2 }$ and m are defined on the same FoD Θ and constructed as follows:

$$
m _ { 1 } \left( \theta _ { 1 } \right) = x , m _ { 1 } \left( \theta _ { 2 } \right) = 1 - x
$$

$$
m _ { 2 } \left( \theta _ { 1 } \right) = 1 - x , m _ { 2 } \left( \theta _ { 2 } \right) = x
$$

$$
m _ { 3 } \left( \theta _ { 1 } \right) = 1 - x , m _ { 3 } \left( \theta _ { 1 } , \theta _ { 2 } \right) = x
$$

where $x \in [ 0 , 1 ]$ . Using Eqs. 12 and 14, the similarity values are computed and the results are shown in Figure 2. Several observations can be made:

(1) When $x = 0$ or $x = 1$ , the BPAs become completely conflicting, the similarity correctly returns $S \left( \cdot \right) = 0$ ;

(2) When $x = 0 . 5 , m _ { 1 }$ and $m _ { 2 }$ become identical. However, the resulting similarity does not reach 1. This is because S (·) not only considers the closeness between the two BPAs, but also incorporates their intrinsic uncertainty (nonspecificity). At this moment, both BPAs carry the maximum amount of ambiguity and contribute no efective decision-making information. This observation is consistent with Property 4, which establishes that similarity reaches 1 only under defined conditions. And moreover:

(3) We observe that $S \left( m _ { 1 } , m _ { 3 } \right) \leq S \left( m _ { 1 } , m _ { 2 } \right)$ . Compared with m m assigns part of its mass to NFE. This allocation increases the level of nonspecificity, resulting in greater intrinsic uncertainty. Consequently, even when the BPA values appear similar, the uncertainty prevents the similarity from reaching a higher value.

![](images/b40c16783e8d75728b8eed6f252627952c2b53127cb73b5120b0882953f1797d.jpg)  
Figure 2: Similarity for Example 1

Example 2.

$$
m _ { 1 } \left( \theta _ { 1 } \right) = x , m _ { 1 } \left( \theta _ { 2 } \right) = 1 - x
$$

$$
m _ { 2 } \left( \theta _ { 1 } \right) = x , m _ { 2 } \left( \theta _ { 2 } \right) = 1 - x
$$

$$
m _ { 3 } \left( \theta _ { 1 } \right) = x , m _ { 3 } \left( \theta _ { 1 } , \theta _ { 2 } \right) = 1 - x
$$

where x ∈ [0 1] and the similarity results are shown in Figure 3. From Figure 3, it can be observed that even when two evidences follow identical BPA assignment, their similarity values may still difer significantly. Specifically, when the BPA becomes more dispersed, the associated uncertainty increases accordingly. In such cases, it becomes dificult to reliably assess the similarity between two evidences, as higher uncertainty weakens the confidence in their consistency. Furthermore, when BPA values are transferred from SFEs to their supersets, as in the case of $m _ { 3 }$ , additional uncertainty is introduced. Since a NFE contains multiple propositions, the belief mass assigned to it cannot be fully committed to any single hypothesis. As a result, the evidence provides less information for decision-making, leading to a lower similarity.

![](images/f8032f5fdd2daf98fed83894fda06ffc7cc9adfdb6752852e62913eace937454.jpg)  
Figure 3: Similarity for Example 2

In summary, the proposed similarity measure quantifies the agreement between evidences by jointly comparing their BPA while fully accounting for their intrinsic uncertainty. On this basis, we introduce the CCM of evidence as follows:

Definition 12. CCM. Let $m _ { i }$ and $m _ { j }$ be two arbitrary mass functions defined on the same FoD Θ and the CCM is defined as

$$
\begin{array} { l } { \displaystyle \widehat { K } \Big ( m _ { i } , m _ { j } \Big ) = 1 - S \Big ( m _ { i } , m _ { j } \Big ) } \\ { \displaystyle \qquad = \frac { 1 - 2 k \Big ( m _ { i } , m _ { j } \Big ) + k ( m _ { i } , m _ { i } ) \cdot k \Big ( m _ { j } , m _ { j } \Big ) } { 1 - k \Big ( m _ { i } , m _ { j } \Big ) + k ( m _ { i } , m _ { i } ) \cdot k \Big ( m _ { j } , m _ { j } \Big ) } } \end{array}\tag{15a}
$$

It is straightforward to verify that the CCM inherits the same five desirable properties as the similarity measure.

Corollary 2. Global chaos-conflict degree (GCCD). Let $\{ m _ { 1 } , m _ { 2 } , \ldots , m _ { n } \}$ be a set ofmassfunctions defined on the same FoD. The GCCD ofthe evidence set is defined as

$$
\widehat K = \frac { 1 } { n \left( n - 1 \right) / 2 } \sum _ { i < j } \widehat K \left( m _ { i } , m _ { j } \right) .\tag{16}
$$

Corollary 3. Relative stability reliability. Given a set of mass functions {m<sub>1</sub> $m _ { 2 } , \ldots , m _ { n } \}$ defined on the same FoD, the relative stability reliability ofan individual evidence source m is defined as

$$
S R \left( m _ { i } \right) = \frac { \underset { j \neq i } { \sum } S \left( m _ { i } , m _ { j } \right) } { 1 - \widehat { K } } .\tag{17}
$$

While the above definitions and corollaries enable instantaneous evidence assessment of conflict and stability, they remain limited to the current decision context. In practical applications where evidence sources are repeatedly reused under heterogeneous conditions, exploiting historical experience becomes essential for achieving robust and adaptive evidence fusion, which is the focus of the next section.

## 3.3 Evidence weighting driven by historical experience

Existing strategies predominantly assess evidential inconsistency based on the current hypothesis space and the configuration of BPAs. Such approaches, however, implicitly assume that all evidence sources are equally reliable across diferent decision contexts (Qiu et al., 2025), thereby overlooking the informative value embedded in their long-term historical performance. In real-world applications, evidence sources are rarely one-shot contributors. The generation of BPAs is often constrained by complex environmental factors, and the reliability of an evidence source typically exhibits within specific scene. Relying exclusively on instantaneous conflict evaluation may therefore lead to biased reliability assessment and suboptimal evidence weighting (Guo et al., 2026; Zhang et al., 2025). A representative example can be found in complex diagnostic scenarios, where diferent evidence sources exhibit markedly diferent detection capabilities across specific conditions (Lu & Zhu, 2026). In such cases, certain sources may systematically outperform others in particular contexts, despite occasionally producing conflicting assessments. Treating these discrepancies solely as instantaneous conflict may overlook the historically validated reliability of informative evidence sources and lead to their unjustified suppression during fusion (Wang et al., 2021).

Motivated by these observations, we introduce a historical experience driven evidence weighting mechanism that complements the CCM proposed above. Rather than relying exclusively on instantaneous conflict evaluation, the proposed approach leverages historical decision outcomes to learn context-dependent evidence weights, which are subsequently integrated into a hybrid combination rule that adaptively balances conservative uncertainty preservation with weighted consensus.

Specifically, heterogeneous decision contexts are first identified via clustering, after which the historical behavior of each evidence source is evaluated through regret and rejoice measures derived from regret theory. These measures quantify the counterfactual contribution of individual evidence sources when fusion outcomes deviate from ground truth, enabling continuous credit assignment that rewards evidence sources aligned with successful outcomes and penalizes those contributing to fusion errors. The accumulated historical experience is then translated into adaptive evidence weights, which are subsequently incorporated into the fusion process.

By explicitly integrating long-term evidence behavior with instantaneous conflict assessment, our approach avoids indiscriminate suppression of conflicting evidences and yields a more reliable fusion strategy.

RT is employed to analyze the potential regret efects that may arise when decision outcomes deviate from expectations. This theory not only focuses on the absolute gains of decisions but also emphasizes the relative gains compared to other potential choices. Introducing regret theory into the calculation of historical experience for evidence synthesis provides a quantitative and systematic method for evaluating the applicability of diferent pieces of evidence in historical contexts, thereby aiding in the optimization of future evidence selection and integration strategies. Among these, the calculation method for regret-rejoicing values, which assesses the applicability of evidence, will serve as the core research focus of this section. Next, we will introduce the regret-rejoicing measurement of evidence bodies during the evidence synthesis process. Let $\Theta = \{ \theta _ { 1 } , \ldots , \theta _ { n } \}$ denote the FoD and $M = \{ m _ { 1 } , m _ { 2 } , \dots , m _ { h } \}$ be the set of available evidence sources defined on $2 ^ { \Theta }$

Definition 13. Optimal evidence. For a given sample whose ground-truth hypothesis is known, the optimal evidence is defined as the evidence source whose BPA is most consistent with the truth according to the similarity measure defined in Eq. 14:

$$
m _ { \mathrm { b e s t } } = \arg \operatorname* { m i n } _ { m _ { i } \in M } \widehat { K } \left( m _ { i } , p _ { \mathrm { t a r g e t } } \right)\tag{18}
$$

Here, $p _ { \mathrm { { t a r g e t } } }$ is a one-hot vector of dimension $2 ^ { n }$

Definition 14. Error-removal combination. Given the evidence set M , the initial fused result $m _ { \oplus }$ is first obtained using Dempster’s rule of combination as defined in Eq. 5. If the decision derived from $m _ { \oplus }$ is inconsistent with the truth, a decision error is considered to have occurred. In such a case, the evidence set is assumed to contain at least one erroneous evidence source, whose mass function fails to assign its maximum support to the true hypothesis.

Let $m _ { e }$ ∈ M denote a candidate erroneous evidence source. By removing $m _ { e }$ from the M and recombining the remaining evidence, the error-removal combination result $m _ { \oplus } ^ { - e }$ is defined as

$$
\left\{ \begin{array} { l } { \displaystyle { m _ { \oplus } ^ { - e } \left( S \right) = \frac { \sum _ { j = S , S _ { j } \subseteq 2 ^ { \Theta } i \neq e } \prod _ { i } m _ { i } \left( S _ { j } \right) } { 1 - K } } } \\ { K = \displaystyle { \sum _ { \bigcap S _ { j } = \emptyset , S _ { j } \subseteq 2 ^ { \Theta } i \neq e } \prod _ { i \neq e } m _ { i } \left( S _ { j } \right) } } \end{array} \right. .\tag{19}
$$

Definition 15. Rejoice value of the optimal evidence. Based on the CCM and the formulation of RT, the rejoice value associated with the $m _ { \mathrm { { b e s t } } }$ is defined as

$$
j ( m _ { \mathrm { b e s t } } ) = 1 - \exp \left( - \eta \widehat { K } ( m _ { \mathrm { b e s t } } , m _ { \oplus } ) \right) ,\tag{20}
$$

where $\eta \in ( 0 , 1 ]$ is a sensitivity parameter and controls the responsiveness of the rejoice value to variations in conflict intensity.

Remark 2. The underlying intuition of Definition 15 is as follows. When thefused evidence leads to an incorrect decision, it indicates that the collective judgment derivedfrom evidence aggregation is unreliable. In such cases, the decision supported by the optimal evidence may provide a more faithful reflection of ground truth. Consequently, a larger conflict between m <sub>best</sub> and the erroneousfusion result $m _ { \oplus }$ implies that m $b e s t$ is closer to the true outcome, and should therefore be assigned a higher rejoice value.

Definition 16. Regret value of erroneous evidence. Analogous to Definition 15, the regret value associated with an erroneous evidence source $m _ { e }$ is defined as

$$
r \left( m _ { e } \right) = 1 - \exp \left( - \gamma \widehat { K } \left( m _ { \oplus } ^ { - e } , m _ { \oplus } \right) \right) ,\tag{21}
$$

where $\gamma \in ( 0 , 1 ]$ is a sensitivity parameter analogous to η , which governs the responsiveness of the regret value to variations in conflict intensity.

Remark 3. The intuitive interpretation ofDefinition 16 is that the discrepancy between the $m _ { \oplus }$ and the $m _ { \oplus } ^ { - e }$ reflects the extent to which the erroneous evidence distorts the collective decision. A larger conflict between these two fusion outcomes indicates a stronger negative impact of $m _ { e }$ on the fusion process, and consequently corresponds to a higher regret value.

In practical decision systems, the reliability of an evidence source is rarely invariant across all operating conditions. Instead, evidence behavior typically exhibits strong context dependency: a source that is highly informative in one scenario may become less reliable in another due to changes in acquisition conditions, latent nuisance factors, or feature distribution shifts (Huang et al., 2025; Zhao et al., 2025). To reflect such heterogeneity, we partition the historical samples into a set of decision contexts (clusters), and assign each sample to a context. Within each context, we then aggregate regret and rejoice values induced by decision failures, thereby obtaining context-conditioned historical experience for each evidence.

Definition 17. Historical experience-based evidence weight. Let $\mathcal { H } = \{ ( x _ { t } , y _ { t } ) \} _ { t = 1 } ^ { N }$ be the historical set and $C = \{ c _ { 1 } , \ldots , c _ { L } \}$ the context partition induced by clustering algorithm. $c \left( x _ { t } \right) \in C$ denotes the cluster membership of $x _ { t }$ . For an evidence source $m _ { i }$ and a target context $c \in C$ , define the failure indicator

$$
\delta _ { t } = \mathbb { I } \left[ \arg \operatorname* { m a x } _ { \theta _ { k } \in \Theta } B e l _ { m _ { \oplus } ( \cdot | x _ { t } ) } \left( \theta _ { k } \right) \neq y _ { t } \right]\tag{22}
$$

where I (·) denotes the indicator function and $\delta _ { t } = 1$ indicates that the final fusion decision at sample x is incorrect. Let $j _ { t } \left( m _ { i } \right)$ and $r _ { t } \left( m _ { i } \right)$ denote the rejoice and regret values of evidence source m computed according to Eqs. 20 and 21, the weight score of $m _ { i }$ under context c is defined as:

$$
J R W _ { i } ^ { c } = \sum _ { t = 1 } ^ { N } \delta _ { t } \mathbb { I } \left[ c \left( x _ { t } \right) = c \right] j _ { t } \left( m _ { i } \right) - \sum _ { t = 1 } ^ { N } \delta _ { t } \mathbb { I } \left[ c \left( x _ { t } \right) = c \right] r _ { t } \left( m _ { i } \right) .\tag{23}
$$

Then, softmax normalization

$$
\widetilde { J R W _ { i } ^ { c } } = \frac { \displaystyle \exp \left( J R W _ { i } ^ { c } \right) } { \displaystyle \sum _ { j = 1 } ^ { M } \exp \left( J R W _ { j } ^ { c } \right) }\tag{24}
$$

is implemented to guarantee the final weight $\widetilde { J R W _ { i } ^ { c } } \ge 0$ and $\sum _ { i } \widetilde { J R W _ { i } ^ { c } } = 1$

## 3.4 A robust fusion-and-decision rule

Building upon the historical experience-driven evidence weighting mechanism developed in Section 3.3, we now construct a robust fusion rule that integrates long-term source reliability with instantaneous evidential conflict. Specifically, Section 3.3 yields context-dependent reliability profiles learned from historical decision outcomes, which enable the construction of a weighted evidence representation reflecting the heterogeneous performance of diferent evidence sources. While such historical calibration mitigates the risk of indiscriminately suppressing informative evidence, robust fusion further requires an explicit treatment of the conflict structure exhibited by the current evidence set. To this end, Section 3.2 introduced the CCM, which characterizes not only the degree of evidential inconsistency but also the uncertainty embedded in evidence formation.

Motivated by these observations, this section proposes a new robust evidence fusion rule following a principled two-level design. First, instead of aggregating evidence via the conventional simple averaging strategy in Murphy (2000)’s method, we employ reliability discounting guided by historical experience to obtain a calibrated weighted evidence representation. This calibrated evidence is subsequently used as the consensus component in the fusion process. Second, the CCM is computed to quantify the instantaneous inconsistency more faithfully. Finally, a conflict-adaptive hybrid combination is performed, which smoothly balances informative consensus and uncertainty according to the estimated conflict level. In this way, the proposed rule achieves improved robustness under heterogeneous contexts. The following is the historical experience weighted evidence we obtained:

$$
m _ { w } \left( S \mid x \right) = \sum _ { i } { \widetilde { J R W _ { i } ^ { c ( x ) } } } \cdot m _ { i } \left( S \mid x \right)\tag{25}
$$

where x denotes the target sample for which evidence fusion is performed, and the notation $m _ { i } \left( S \mid x \right)$ indicates that the BPA provided by the i -th evidence source. The corresponding historical experience weight $J R W _ { i } ^ { c ( x ) }$ is therefore selected, ensuring that the contribution of each evidence is modulated according to its long-term performance under similar conditions. Consequently, the weighted evidence $m _ { w } \left( S \mid x \right)$ represents a context-aware aggregation of the original BPAs associated with the sample x . After obtaining weighted evidence based on historical experience, we give a new evidence fusion rule defined as follows.

Definition 18. Hybrid combination rule. To construct belief intervals and to allocate the aggregated mass more informatively, we define a hybrid combination rule as follows:

$$
\begin{array} { l } { \displaystyle { m ^ { \oplus } \left( S \right) = \left( 1 - e ^ { - \widehat K } \right) \Bigg ( \sum _ { A _ { 1 } \bigcap \cdots \bigcap A _ { M } = S \neq \emptyset } \prod _ { i = 1 } ^ { M } m _ { i } \left( A _ { i } \right) } } \\ { \displaystyle { + \sum _ { A _ { 1 } \bigcap \cdots \bigcap A _ { M } = \emptyset } \prod _ { i = 1 } ^ { M } m _ { i } \left( A _ { i } \right) \Bigg ) + e ^ { - \widehat K } m _ { w } \left( S \right) } } \\ { \displaystyle { A _ { 1 } \bigcup \cdots \bigcup A _ { M } = S } } \end{array}\tag{26}
$$

Eq. 26 consists of two complementary terms. The first term is a refined Dubois combination rule Liu et al. (2023), which aggregates both conjunctive contributions and union-based allocations so as to explicitly preserve uncertainty under severe disagreement. The second term $m _ { w } \left( S \right)$ represents the consensus evidence obtained by weighted averaging, which is beneficial when the evidence sources are largely consistent. The mixing factor $e ^ { - \widehat { K } } \in ( 0 , 1 ]$ adaptively controls the trade-of: when the overall conflict is high, $\dot { e ^ { - \widehat { K } } }$ becomes small and the rule places greater emphasis on the conservative term, thereby pushing more mass toward NFEs and retaining uncertainty for subsequent interval construction; when the conflict is mild, the rule increasingly relies on the consensus term, yielding a sharper and more decisive fused belief assignment.

Having obtained the final fused mass function through the proposed Definition 18, a principled decision rule is still required to output a unique hypothesis. This step is nontrivial in the presence of composite FEs, since the fused evidence may preserve a nonnegligible amount of epistemic uncertainty. To this end, we adopt a belief interval-based decision strategy, which directly exploits the confidence bounds induced. The resulting score provides a measure for each singleton hypothesis and enables a deterministic decision.

Definition 19. Decision rule induced by belief intervals. Let $m ^ { \oplus } \left( S \mid x \right)$ denote the final fused mass function obtained by the proposed hybrid combination rule, and let $B e l ( \cdot \mid x )$ and $P l ( \cdot \mid x )$ be the standard belief and plausibility functions induced by $m ^ { \oplus } \left( S \mid x \right)$ . The decision score for each singleton hypothesis $S _ { i }$ is defined as

$$
\begin{array} { r l } { \mathrm { P t i } \left( S _ { i } \mid x \right) } & { = \displaystyle \left( 1 - \frac { \mathrm { P l } \left( S _ { i } \mid x \right) - \mathrm { B e l } \left( S _ { i } \mid x \right) } { P l _ { \operatorname* { m a x } } - B e l _ { \operatorname* { m i n } } } \right) } \\ & { \qquad \times \left( \mathrm { P l } \left( S _ { i } \mid x \right) - \mathrm { B e l } \left( S _ { i } \mid x \right) \right) + \left( \mathrm { B e l } \left( S _ { i } \mid x \right) - B e l _ { \operatorname* { m i n } } \right) } \end{array}\tag{27}
$$

and the final decision is made by

$$
{ \widehat { y } } ( x ) = \arg \operatorname* { m a x } _ { S _ { i } = \{ \theta _ { i } \} } \operatorname { P t i } \left( S _ { i } \mid x \right) .\tag{28}
$$

This decision rule provides an interval-aware transformation. Specifically, $B e l _ { \mathrm { m i n } }$ and $P l _ { \mathrm { m a x } }$ denote the minimum credibility and the maximum plausibility among all singleton hypotheses respectively. Global spread $P l _ { \mathrm { m a x } } - B e l _ { \mathrm { m i n } }$ serve as a common reference scale for the current fused evidence. The coeficient $1 - ( P l ( S _ { i } \mid x ) - B e l ( S _ { i } \mid x ) ) / ( P l _ { \operatorname* { m a x } } - B e l _ { \operatorname* { m i n } } )$ is then introduced as a stability indicator of the credibility interval: it decreases monotonically with the interval width $P l \left( S _ { i } \mid x \right) - B e l \left( S _ { i } \mid x \right)$ , thereby penalizing hypotheses whose support is highly ambiguous or weakly concentrated.

Accordingly, $\operatorname { P t i } \left( S _ { i } \right)$ combines two complementary contributions. The term Bel $( S _ { i } \mid x ) - B e l _ { \operatorname* { m i n } }$ quantifies the relative advantage of the lower bound, reflecting the amount of commitment that is guaranteed for $S _ { i }$ beyond the weakest supported hypothesis. The second term, $\begin{array} { r } { \left( 1 - \frac { P l ( S _ { i } | x ) - B e l ( S _ { i } | x ) } { P l _ { \mathrm { m a x } } - B e l _ { \mathrm { m i n } } } \right) } \end{array}$ $( P l ( S _ { i } \mid x ) - B e l ( S _ { i } \mid x ) )$ , incorporates the potential support upper bound while adaptively discounting it by the interval stability. This design prevents overly optimistic decisions driven by large plausibility values that are accompanied by wide uncertainty intervals, and at the same time preserves the informative role of plausibility when the evidence is consistent and well-concentrated. As a result, the proposed score yields a robust Bayesian-style decision criterion that remains well-defined in the presence of non-specific focal elements, without forcing an artificial redistribution of NFE mass onto singleton hypotheses.

## 4 Algorithmic design

This section presents the complete algorithmic workflow of the proposed framework. The procedure consists of two distinct phases, namely ofline training on historical experience (Algorithm 1) and online inference (Algorithm 2), which correspond to the learning and utilization of historical experience, respectively.

In the training phase (historical experience modeling): given a historical dataset $\mathcal { H } = \{ ( x _ { t } , y _ { t } ) \} _ { t = 1 } ^ { N }$ , SC is first performed on the BPA feature representations of individual samples to partition the heterogeneous decision space into L context scenarios $C ~ = ~ \{ c _ { 1 } , \ldots , c _ { L } \}$ For each historical sample $x _ { t }$ the BPA results provided by multiple evidence sources $M =$ $\{ m _ { 1 } , \ldots , m _ { h } \}$ are fused via Dempster’s combination to obtain the aggregated result $m ^ { \oplus } ( \cdot \mid x _ { t } )$ . When the fused decision conflicts with the ground truth label $y _ { t }$ , the regret-rejoice computation is triggered: the optimal evidence $m _ { \mathrm { \ b e s t } }$ is identified according to Eq. 18, and its rejoice value $j ( m _ { \mathrm { b e s t } } )$ is calculated using $\operatorname { E q . 2 0 } ;$ subsequently, for each candidate erroneous evidence $m _ { e } \in M$ the regret value $r \left( m _ { e } \right)$ is computed as per Eq. 21. The cumulative historical score $J R W _ { i } ^ { c }$ of each evidence within scenario c is defined as the algebraic sum of rejoice and regret, which is then normalized via softmax to yield the scenario-specific historical experience weight.

In the inference phase (evidence fusion and decision-making): for a test sample x , its belonging cluster $c \left( x \right)$ is first determined, and the corresponding historical experience weights $J R W _ { i } ^ { c ( x ) }$ are retrieved. The CCM matrix of the current evidence set is computed to obtain the GCCD $\widehat { K }$ The historically weighted evidence $m _ { w }$ is constructed according to Eq. 25, followed by the application of the hybrid combination rule given in Eq. 26, which adaptively adjusts the weight allocation between the conflict and consensus terms via $e ^ { - { \widehat { K } } }$ . The $m _ { w }$ is then processed through the belief intervals interval decision rule defined in Eqs. 27-28 to output the final class label.

## 5 Experiment

## 5.1 Experimental settings

## 5.1.1 Datasets

Table 1 summarises the 16 real world datasets used in the experiments. All were obtained from the UCI Machine Learning Repository and the NIH-CIP Repository, which collectively cover a broad spectrum of characteristics: high or low dimensional feature spaces, binary or multi class targets, and balanced or imbalanced distributions. The data covers fields such as biology, medicine, geography, and demography analysis. To prevent scale-dependent bias in the classifiers, every dataset was standardized before use.

Algorithm 2: Online Inference: Robust Evidence   
Fusion and Decision   
Algorithm 1: Ofline Training: Historical Input: Test sample x; evidence sources   
Experience Modeling $M = \{ m _ { 1 } , \hdots \cdot , m _ { h } \}$ on FoD Θ; historical   
Input: Historical dataset ${ \mathcal { H } } = \{ ( x _ { t } , y _ { t } ) \} _ { t = 1 } ^ { N } ;$ evidence weights $\{ \widetilde { J R W _ { i } } \} ;$ sensitivity parameters $\eta , \gamma$   
sources $M = \{ m _ { 1 } , \dots , m _ { h } \}$ defined on FoD Θ; Output: Predicted class label ˆy   
sensitivity parameters η γ 1 Determine cluster membership: c(x) ← assign x to   
Output: Context-specific historical experience nearest context in $C ;$   
weights $\{ \stackrel { - } { J R W } _ { i } ^ { c } \} _ { i = 1 } ^ { h , c \in C }$ 2 Retrieve historical weights $\widehat { \{ J R W _ { i } ^ { c ( x ) } \} } _ { i = 1 } ^ { h } ;$   
/\* Phase 1: Context partition via /\* Step 1: Compute CCM and GCCD \*/   
spectral clustering \*/ 3 foreach $p a i r ( m _ { i } , m _ { j } ) , 1 \leq i < j \leq h$ do   
1 Construct BPA feature matrix $\mathbf { B } \in \mathbb { R } ^ { N \times ( h \cdot | 2 ^ { \Theta } | ) }$ from all 4 Compute association $k ( m _ { i } , m _ { j } )$ via Eq. 12;   
evidence sources; 5 Compute similarity $S ( m _ { i } , m _ { j } )$ via Eq. 14;   
2 Perform spectral clustering on B to obtain context 6 Compute chaos-conflict   
partition $C = \{ c _ { 1 } , \ldots , c _ { L } \} ;$ $\widehat { K } ( m _ { i } , m _ { j } ) \gets 1 - S ( m _ { i } , m _ { j } )$ // Eq. 15a   
/\* Phase 2: Regret--rejoice computation   
on historical samples \*/ 7 $\widehat { K } \gets \frac { 2 } { h ( h - 1 ) } \sum _ { i < j } \widehat { K } ( m _ { i } , m _ { j } )$ // GCCD, Eq. 16   
3 foreach sample $( x _ { t } , y _ { t } ) \in \bar { \mathcal { D } }$ do   
4 Fuse all evidence via Dempster’s rule: /\* Step 2: Construct   
$m ^ { \oplus } ( \cdot \mid x _ { t } )  m _ { 1 } \oplus \cdot \cdot \cdot \oplus m _ { h } ;$ historical-experience weighted   
5 $\hat { y } _ { t } \gets \arg \operatorname* { m a x } _ { \theta _ { k } \in \Theta } B e l _ { m ^ { \oplus } } ( \theta _ { k } \mid x _ { t } ) ;$ evidence \*/   
6 if $\hat { y } _ { t } = y _ { t }$ then $\delta _ { t } \gets 0 \quad / /$ correct decision, $m _ { w } ( S \mid x ) \gets \sum _ { i = 1 } ^ { h } \widetilde { J R W } _ { i } ^ { c ( x ) } \cdot m _ { i } ( S \mid x ) , \quad \forall S \subseteq 2 ^ { \Theta }$   
skip; 8   
7 else $\delta _ { t } \gets 1 ;$   
8 if $\delta _ { t } = 1$ then $/ / \mathrm { ~ \ E q . ~ \ 2 5 ~ }$   
9 Identify optimal evidence: /\* Step 3: Hybrid combination rule \*/   
$m _ { \mathrm { b e s t } }  \mathrm { a r g } \operatorname* { m i n } _ { m _ { i } \in M } \widehat { K } ( m _ { i } , p _ { \mathrm { t a r g e t } } )$ 9 $\alpha \gets 1 - e ^ { - \widehat { K } }$ ; // conflict weight   
// Eq. 18 10 $\beta \gets e ^ { - { \widehat { K } } }$ // consensus weight   
10 Compute rejoice value:   
$j ( m _ { \mathrm { b e s t } } ) \gets 1 - \exp ( - \eta \widehat { K } ( m _ { \mathrm { b e s t } } , m ^ { \oplus } ) )$ ; 11 $m _ { \mathrm { c o n j } } ( S ) \gets \sum _ { A _ { 1 } \cap \cdots \cap A _ { h } = S \neq \emptyset } \prod _ { i = 1 } ^ { h } m _ { i } ( A _ { i } )$ ;   
// Eq. 20   
11 foreach candidate erroneous evidence $m _ { e } \in M$ // conjunctive term   
12 do Compute error-removal combination $m ^ { \oplus , - e }$ 12 $m _ { \mathrm { d i s j } } ( S ) \gets \sum _ { A _ { 1 } \cap \cdots \cap A _ { h } = \emptyset } \prod _ { i = 1 } ^ { h } m _ { i } ( A _ { i } )$ ; // disjunctive   
via Eq. 19;   
13 Compute regret value:   
$r ( m _ { e } ) \gets 1 - \exp ( - \gamma \widehat { K } ( m ^ { \oplus , - e } , m ^ { \oplus } ) )$ term   
// Eq. 21 13 $m ^ { \oplus } ( S ) \gets \alpha \cdot ( m _ { \mathrm { c o n j } } ( S ) + m _ { \mathrm { d i s j } } ( S ) ) + \beta \cdot m _ { w } ( S )$   
// Eq. 26   
/\* Step 4: Belief-interval decision \*/   
/\* Phase 3: Aggregate and normalize   
14 foreach singleton hypothesis $\theta _ { k } \in \Theta$ do   
historical weights \*/   
14 foreach context $c \in C$ do 15 $B e l ( \theta _ { k } \mid \bar { x } )  \bar { \sum } m ^ { \oplus } ( A \mid x ) ;$   
15 foreach evidence source $m _ { i } \in M$ do A⊆{θ }   
$J R W _ { i } ^ { c } \gets \sum _ { . . . 1 } ^ { N } j _ { t } ( m _ { i } ) - \sum _ { . . . 1 } ^ { N } r _ { t } ( m _ { i } ) ;$ 16 $P l ( \theta _ { k } \mid x )  \ \sum \ m ^ { \oplus } ( A \mid x ) ;$   
16 A∩{θ<sub>k</sub>},∅   
<sup>t=1</sup>δ =1 c(x )=c <sup>t=1</sup>δ<sub>t</sub>=1 c(x<sub>t</sub>)=c 17 $\begin{array} { r } { B e l _ { \mathrm { m i n } } \gets \operatorname* { m i n } _ { \theta _ { k } } B e l ( \theta _ { k } \mid x ) ; } \end{array}$   
// Eq. 23 $P l _ { \mathrm { m a x } }  \mathrm { m a x } _ { \theta _ { k } } P l ( \theta _ { k } \mid x ) ;$   
18 foreach $\theta _ { k } \in \Theta$ do   
17 Normalize: $\widetilde { J R W } _ { i } ^ { c } \gets \frac { \exp ( J R W _ { i } ^ { c } ) } { \sum _ { j = 1 } ^ { h } \exp ( J R W _ { j } ^ { c } ) }$ for all i ; 19 $P t i ( \theta _ { k } \mid x ) $   
// Eq. 24 $\bigg ( 1 - \frac { P l ( \theta _ { k } \mid x ) - B e l ( \theta _ { k } \mid x ) } { P l _ { \operatorname* { m a x } } - B e l _ { \operatorname* { m i n } } } \bigg ) ( P l ( \theta _ { k } \mid x ) - B e l ( \theta _ { k } \mid x ) ) +$   
18 return $\underset { i } { \widetilde { J R W } } _ { i } ^ { c } \} _ { i = 1 } ^ { h , c \in C }$ $( B e l ( \theta _ { k } \mid x ) - B e l _ { \operatorname* { m i n } } ) ;$ // Eq. 27   
20 yˆ ← arg max<sub>θ</sub> Pti(θ<sub>k</sub> | x) ; // Eq. 28   
21 return yˆ

## 5.1.2 Baseline and hyperparameter settings

To demonstrate the superior inference capability of the proposed framework, we benchmark it against two representative classes of methods: evidential reasoning and ensemble learning. Within the evidential reasoning category, the comparative methods include Dempster’s rule (Dempster, 1967), Yager’s rule (Yager, 1987), Dubois’ rule (Dubois & Prade, 1988), Murphy’s averaging rule (Murphy, 2000), Deng’s weighted rule (Deng et al., 2004), Belief Hellinger Fusion method (BHF) (Zhu & Xiao, 2021), EC-FMDS (Chamlal et al., 2025), and the Hierarchical Evidence Fusion method (HEF) (Zhang et al., 2025). For ensemble learning, comprehensive comparative experiments are conducted against three gradient boosting methods, namely eXtreme Gradient Boosting (XGBoost) (T. Chen & Guestrin, 2016), Light Gradient Boosting Machine (LightGBM) (Ke et al., 2017), and Categorical Boosting (CatBoost) (Prokhorenkova et al., 2018).

To ensure a fair comparison, given that the ensemble learning methods utilize decision trees (DTs) as base learners, we similarly employ DTs as evidence bodies within the DST framework. Random split selection is adopted to introduce necessary stochasticity into each evidence body (i.e., each individual DT), while hyperparameters such as tree depth are configured according to recommendations to promote generalization (Dhanka et al., 2026). During the comparative experiments, the number of base DTs in the ensemble methods is set equal to the number of evidence bodies in the DST methods. Detailed parameter settings are provided in Table 2; parameters not listed in the table follow the default values of the oficial Python libraries.

## 5.1.3 Procedure and metrics

All experiments were conducted on a system with an Intel Core Ultra 5 225H CPU, 32 GB RAM, and Windows 11 25H2. The code was implemented in Python 3.9.25, and all stochastic procedures were fixed with a random seed of 42 to ensure reproducibility.

To ensure fair and comprehensive evaluation, all methods were assessed under a unified experimental protocol across the 16 datasets listed in Table 1. Each dataset was randomly partitioned using 5-fold cross-validation. Reported results are presented as mean values, standard deviations, and rankings (where applicable) across all folds to ensure statistical reliability and robustness.

To further validate the proposed method, we conducted additional analyses encompassing robustness evaluation under noisy conditions, adjustment of the number of evidence bodies, and substitution of the evidence body base model, together with sensitivity analysis and ablation studies of hyperparameters. These complementary experiments provide deeper insights into the stability, generalization capability, and component contributions of the proposed framework. Detailed experimental procedures are presented in the corresponding sections below.

For comprehensive assessment of classification performance, we employed four widely adopted evaluation metrics: accuracy (ACC), precision (PRE), recall (REC), and F1-score (F1). ACC measures the overall proportion of correctly classified samples, whereas PRE and REC evaluate the correctness and completeness of positive predictions, respectively. The F1, defined as the harmonic mean of precision and recall, ofers a balanced assessment particularly suited to imbalanced datasets. We adopted macro averaging for all metrics. Additionally, receiver operating characteristic (ROC) curves and the area under the curve (AUC) were employed to evaluate the discriminative capability of diferent methods. The predictions from all folds were pooled to generate a single ROC curve and calculate the AUC.

## 5.2 Results

## 5.2.1 Comparative analysis

This section uniformly employed three decision trees as evidence sources to construct the DST framework, with the number of base learners matching that of the compared ensemble methods. Table 3 summarizes the mean performance, standard deviation, and average rank of each method in terms of ACC, PRE, REC, and F1. Our method achieved optimal mean values across all evaluation metrics (ACC: 87.01, PRE: 85.44, REC: 87.51, F1: 85.78), with average ranks of 3.31, 3.52, 2.89, and 3.22, respectively. Detailed performance results for each dataset are provided in Appendix A.

Compared with the DST model, the proposed framework demonstrates superior capability in quantifying and resolving high-conflict evidence. The performance degradation of Dempster’s rule (ACC: 80.30, F1: 76.03) stems from its rigid normalization mechanism, which tends to produce counterintuitive fusion results. Yager’s approach reallocates conflict mass to the frame of discernment; however, this conservative strategy indiscriminately discards valuable informative discrimination signals, manifesting as the second-lowest F1 score among all methods (74.87). Murphy’s average rule enhances stability through uniform evidence weighting, yet its assumption of source consistency fails to account for context-dependent reliability variations; despite its smoothing efect, it shows no substantial advantage over Dempster’s rule (F1: 77.74 vs. 76.03), indicating inadequate handling of such uncertainty. Deng’s method introduces source diferentiation through Jousselme distance-based weighting, but measures only static BPA discrepancies without modeling the internal uncertainty structure of FEs, limiting its adaptability under epistemic uncertainty. The BHF method employs belief Hellinger distance to quantify evidence dissimilarity and incorporates entropy to characterize ambiguity, thereby establishing a weighted average fusion strategy. However, its weight generation relies exclusively on inter-evidence distance metrics and intrinsic uncertainty, without leveraging historical decision feedback; consequently, it fails to capture reliability variations of evidence sources across diverse decision contexts. EC-FMDS attempts to integrate feature selection preprocessing with fusion mechanisms, yet their independent operational modes cannot resolve conflicts arising from evidence-level inconsistency, as corroborated by its substantial performance ranking variance. HEF’s hierarchical structure mitigates conflict and reduces computational complexity through averaged evidence sources; however, its strict fusion sequence and evidence averaging mechanism may lead to premature determination of unreliable averaged evidence system combinations when evidence bodies are scarce and homogeneous (e.g., three decision trees in this experiment), evidenced by its lowest F1 (75.36).

Table 1: Experimental datasets
<table><tr><td>Datasets</td><td>Sample</td><td>Class (Ratio)</td><td>Feature (Continuous/Discrete)</td></tr><tr><td>Accent</td><td>329</td><td>6(0.5:0.14:0.09:0.09:0.09:0.09)</td><td>12(12/0)</td></tr><tr><td>BMNP</td><td>63</td><td>3(0.62:0.25:0.13)</td><td>36(16/20)</td></tr><tr><td>Diabetes</td><td>520</td><td>2(0.62:0.38)</td><td>16(1/15)</td></tr><tr><td>Forest</td><td>523</td><td>4(0.37:0.3:0.16:0.16)</td><td>27(27/0)</td></tr><tr><td>German</td><td>1000</td><td>2(0.7:0.3)</td><td>24(3/21)</td></tr><tr><td>Hfcr</td><td>299</td><td>2(0.68:0.32)</td><td>12(8/4)</td></tr><tr><td>Ionosphere</td><td>351</td><td>2(0.64:0.36)</td><td>34(32/2)</td></tr><tr><td>ISPY</td><td>215</td><td>2(0.73:0.27)</td><td>18(4/14)</td></tr><tr><td>Sonar</td><td>208</td><td>2(0.53:0.47)</td><td>60(60/0)</td></tr><tr><td>Sports</td><td>1000</td><td>2(0.64:0.36)</td><td>59(49/10)</td></tr><tr><td>Turkish</td><td>400</td><td>4(0.25:0.25:0.25:0.25)</td><td>50(50/0)</td></tr><tr><td>Vehicle</td><td>846</td><td>4(0.26:0.26:0.25:0.24)</td><td>18(18/0)</td></tr><tr><td>Wine</td><td>178</td><td>3(0.33:0.4:0.27)</td><td>13(13/0)</td></tr><tr><td>WDBC</td><td>569</td><td>2(0.63:0.37)</td><td>30(30/0)</td></tr><tr><td>WPBC</td><td>198</td><td>2(0.76:0.24)</td><td>33(31/2)</td></tr><tr><td>Zoo</td><td>101</td><td>7(0.41:0.2:0.13:0.1:0.08:0.05:0.04)</td><td>16(0/16)</td></tr></table>

Table 2: Hyperparameter settings
<table><tr><td>Type</td><td>Methods</td><td>Hyperparameter settings</td></tr><tr><td>Evidence</td><td>DT</td><td>splitter: random, max_depth: 8, min_samples_leaf: 2, min_samples_split: 4, max_features: &quot;sqrt&quot;, class_weight: &quot;balanced&#x27;</td></tr><tr><td>Cluster</td><td>SC CatBoost</td><td>n_clusters: number of categories, affinity: rbf, gamma: 1, assign_labels: kmeans, eigen_solver:arpack, n_init: 20 learning-rate: 0.1, depth: 8, 12_leaf_reg: 1</td></tr><tr><td rowspan="2">Ensemble learning</td><td>LightGBM</td><td>learning_rate: 0.1, max_depth: 8, subsample: 1.0, colsample_bytree: 1.0</td></tr><tr><td>XGBoost</td><td>learning-rate: 0.1, max_depth: 8, subsample: 1.0, colsample_bytree: 1.0, reg_lambda:1e-5</td></tr><tr><td rowspan="9">DST</td><td>Dempster</td><td></td></tr><tr><td>Deng Dubois</td><td>Distance: Jousselme</td></tr><tr><td>EC-FMDS</td><td></td></tr><tr><td>HEF</td><td>q=2</td></tr><tr><td>Murphy</td><td></td></tr><tr><td>Yager</td><td></td></tr><tr><td>BHF</td><td></td></tr><tr><td>Our</td><td></td></tr><tr><td></td><td>η: 0.5, γ: 0.5, |C|: number of class</td></tr></table>

Compared with gradient boosting frameworks, the proposed method exhibits marked superiority in F1 score (85.78 versus 75.05, 77.87, and 78.35). Although all three ensemble methods construct additive models through gradient optimization to improve fitting accuracy by minimizing empirical risk, they lack explicit modeling of inter-evidence uncertainty. In contrast, the proposed framework characterizes decision boundary uncertainty through belief and plausibility functions, thereby preserving more discriminative information in regions with imbalanced class distributions or overlapping feature spaces and achieving superior predictive performance.

The proposed framework overcomes these limitations through a unified CCM mechanism. CCM simultaneously quantifies cross-evidence conflict and intra-evidence non-specificity without presupposing fusion order or structural assumptions; furthermore, it introduces historical experience-driven weighting based on regret theory, enabling evidence reliability assessment to adapt dynamically to current sample conditions and maintaining fine-grained reliability discrimination even under evidence-scarce scenarios. This end-to-end design avoids the decoupling of preprocessing and fusion, achieving collaborative optimization of evidence quality assessment and fusion decision. According to the Friedman test results and Nemenyi post-hoc test (CD = 2.08) presented in Figure 4, the proposed framework achieved the highest average rank among all compared methods, demonstrating statistically significant performance advantages over every alternative approach. Among the competing methods, LightGBM, Murphy, XGBoost, EC-FMDS, and Deng formed a relatively competitive cluster with no statistically significant diferences detected within this group, whereas Dempster, Dubois, CatBoost, BHF, Yager, and HEF occupied lower overall rankings.

Figure 5 presents the ROC curves for all methods across the 16 datasets. The proposed method (red curve) consistently occupies the uppermost position in the majority of cases, exhibiting substantially higher true positive rates at low false positive rates. This indicates superior discriminative capability and well-calibrated uncertainty estimation, particularly in scenarios characterized by ambiguous decision boundaries or class imbalance.

Table 3: Performance comparison (mean ± std) of diferent methods with average rank
<table><tr><td>Type</td><td>Model</td><td>ACC</td><td>ACC_rank</td><td>PRE</td><td>PRE_rank</td><td>REC</td><td>REC_rank</td><td>F1</td><td>F1_rank</td></tr><tr><td rowspan="2">Ensemble learning</td><td>CatBoost</td><td>80.21±13.30</td><td>7.69±4.78</td><td>77.36±17.02</td><td>8.44±4.76</td><td>74.93±18.11</td><td>10.52±4.58</td><td>75.05±18.25</td><td>9.94±4.68</td></tr><tr><td>LightGBM</td><td>83.10±9.90</td><td>4.39±3.32</td><td>79.08±15.41</td><td>4.95±3.53</td><td>78.00±15.00</td><td>5.53±3.60</td><td>77.87±15.54</td><td>5.28±3.58</td></tr><tr><td rowspan="10"></td><td>XGBoost</td><td>82.26±11.20</td><td>4.97±2.98</td><td>79.15±14.50</td><td>5.48±3.03</td><td>78.48±14.74</td><td>5.50±3.20</td><td>78.35±14.83</td><td>5.36±3.15</td></tr><tr><td>Dempster</td><td>80.30±12.90</td><td>8.64±4.16</td><td>77.44±17.30</td><td>9.72±4.45</td><td>76.61±16.80</td><td>9.84±4.15</td><td>76.03±17.56</td><td>10.09±4.13</td></tr><tr><td>Deng</td><td>80.44±13.55</td><td>8.11±4.25</td><td>77.50±16.02</td><td>9.31±4.65</td><td>77.66±16.06</td><td>8.70±4.36</td><td>77.22±16.19</td><td>8.92±4.38</td></tr><tr><td>Dubois</td><td>80.18±12.57</td><td>8.52±4.07</td><td>76.60±15.91</td><td>10.11±4.05</td><td>76.85±15.76</td><td>9.44±4.07</td><td>76.20±16.06</td><td>9.58±3.99</td></tr><tr><td>EC-FMDS</td><td>80.31±14.06</td><td>5.48±3.01</td><td>77.55±16.13</td><td>6.08±3.24</td><td>77.79±16.09</td><td>5.41±3.19</td><td>77.14±16.36</td><td>5.62±3.25</td></tr><tr><td>HEF</td><td>78.51±13.33</td><td>9.98±4.35</td><td>75.36±16.36</td><td>10.64±4.10</td><td>74.36±16.54</td><td>11.33±3.83</td><td>73.63±17.15</td><td>11.53±3.77</td></tr><tr><td>Murphy</td><td>80.89±13.97</td><td>5.48±3.26</td><td>78.35±16.06</td><td>5.89±3.26</td><td>78.11±16.56</td><td>5.47±2.97</td><td>77.74±16.61</td><td>5.55±3.01</td></tr><tr><td>Yager</td><td>78.80±13.54</td><td>7.56±3.56</td><td>75.65±16.59</td><td>7.70±3.67</td><td>75.45±16.31</td><td>7.58±3.58</td><td>74.87±16.86</td><td>7.83±3.51</td></tr><tr><td>BHF</td><td>79.34±13.84</td><td>9.08±4.04</td><td>75.89±17.12</td><td>10.34±3.88</td><td>76.80±16.55</td><td>9.38±3.83</td><td>75.87±16.95</td><td>9.88±3.92</td></tr><tr><td>Our</td><td>87.01±13.93</td><td>3.31±3.78</td><td>85.44±15.29</td><td>3.52±3.90</td><td>87.51±14.75</td><td>2.89±3.51</td><td>85.78±15.42</td><td>3.22±3.77</td></tr></table>

![](images/2623578f9f7491728f4330acc8eaa4bbd1fb98aa40a45b95d7fe1babba2cbaed.jpg)  
Figure 4: CD diagram of all models.

Closer examination reveals two distinct patterns. On datasets with inherently high separability, such as Diabetes, ISPY, and Wine, performance diferences among methods diminish considerably, suggesting that simple evidence aggregation strategies sufice when classification tasks are relatively straightforward. Conversely, on challenging datasets with substantial class overlap, including BMNP, German, and WPBC, the proposed framework demonstrates marked performance advantages over competing approaches. This pronounced separation in dificult scenarios underscores the eficacy of our uncertainty-aware evidence fusion mechanism. Consequently, the proposed framework achieves the highest overall mean AUC of 93.3, with its robust generalization across diverse datasets attributable primarily to exceptional performance in discriminatively demanding conditions rather than incremental gains on already well-resolved tasks.

## 5.2.2 Robustness analysis

To verify the robustness of the proposed framework under diverse conditions, we conducted a series of experiments, which included introducing data noise and varying both the types and the number of evidence sources.

## 5.2.2.1 Noise

To comprehensively evaluate the robustness of the proposed framework under noisy conditions, we introduced multiple types of data noise during the training phase to emulate interference scenarios commonly encountered in real-world applications. Specifically, feature-level noise, label-level noise, and their combinations as hybrid noise were considered to assess the stability and generalization capability of the framework across varying noise intensities.

For feature-level perturbations, two noise injection strategies were adopted. The first randomly replaced a proportion of feature values with the corresponding feature means, thereby modeling data missingness during the acquisition process. The second randomly added noise sampled from a zero-mean Gaussian distribution to selected features, simulating stochastic interference with a normal distribution. For label-level noise, a specified ratio of sample labels was randomly reassigned to alternative classes, reflecting typical annotation errors in practical datasets.

![](images/a96eb168755951d5a340ded69463d2e63c5bc1ca338e34ea4fb0a9389091a3b4.jpg)  
Figure 5: Comparison of ROC curves for diferent models on 16 datasets

To further examine the compound efects of concurrent feature and label corruption, two hybrid noise configurations were designed: uniform feature noise combined with random label noise, and Gaussian-distributed feature noise combined with random label noise. Noise intensities were set to 5%, 10%,

15%, and 20%, enabling a systematic analysis of performance degradation as noise levels increased.

Figure 6 and Figure 7 present the variations in F1 performance of the compared methods under two mixed-noise scenarios, while Appendix B presents the corresponding F1 results under the three basic noise settings. Overall, as the noise level gradually increases from 5% to 20%, the performance of all methods declines to varying degrees. Nevertheless, the proposed method maintains a relatively high F1 score under most noise settings. This result indicates that the proposed approach is not efective only in relatively ideal data environments; rather, it can still sustain stable decision performance when the training samples are afected by feature contamination, label shifts, or their combined interference.

In contrast to most existing DST-based methods, which typically rely on static processing according to the diferences among the current BPAs, they struggle to distinguish anomalous evidence caused by incidental noise from valid evidence with long-term stability. In specific contexts, the proposed method exhibits greater robustness under noise. On the one hand, it employs an adaptive dual-weighting mechanism based on historical experience and current conflict. On the other hand, its hybrid combination rule preserves uncertain mass instead of forcibly compressing it into the SFE, thereby reducing the dominance of anomalous evidence and the associated noise in the final fusion result while maintaining relatively stable performance.

It should nevertheless be acknowledged that, compared with ensemble tree-based methods, most DST-based approaches, including the proposed method in some cases, are generally less stable under noisy conditions. This diference is directly related to the distinct modeling priorities of the two methodological paradigms. Ensemble tree methods are fundamentally developed within a supervised learning framework, where classification errors are continuously fitted through iterative empirical risk optimization or the parallel integration of multiple heterogeneous tree structures, and their objective functions are therefore closely aligned with the final decision outcome (Alzubaidi et al., 2021; van Engelen & Hoos, 2019). Ensemble strategies and recursive splitting will mitigate the instability of individual trees and promote complementary efects across learners (Hasan et al., 2024). By contrast, the main strength of DST methods lies in the explicit representation of uncertainty and the management of conflict, rather than in the direct minimization of classification error. The fusion process in evidence theory is typically centered on BPA construction and combination rule design, with the primary emphasis placed on integrating support relationships among multiple evidence sources. However, this process does not inherently guarantee suficiently strong discriminative learning of class boundaries, and fusion rules mainly redistribute information under existing quality constraints rather than iteratively correcting prediction bias to improve generalization, as ensemble tree models do (Qiao, Song, et al., 2023). For this reason, the proposed method further incorporates historical experience modeling and conflict-adaptive fusion within the DST framework, thereby compensating for the limited discriminative learning capability of conventional evidence fusion methods and enabling it to narrow the gap with, or even outperform, ensemble models under noisy conditions.

Interestingly, under mixed-noise scenarios, the performance of most methods declines relatively gradually as noise intensity increases, whereas under single-noise conditions, performance exhibits more pronounced fluctuations and irregularity. This observation suggests that perturbations do not necessarily exacerbate model degradation in a purely monotonic manner; rather, they may encourage the formation of more robust decision boundaries, thereby improving adaptability to distribution shifts and indirectly enhancing generalization. This finding is consistent with the conclusions of Liu et al. (2023), who reported that moderate noise exposure helps mitigate overfitting and improves model stability in complex and uncertain environments.

Figure 8 further illustrates the distribution of AUC values for diferent methods across varying noise types and intensities, providing a complementary perspective on robustness from the standpoint of ranking-based discriminative ability. Overall, as the noise level increases, the median AUC of most methods shifts downward and the interquartile range tends to widen, indicating that noise afects not only average discriminative performance but also cross-dataset stability. The proposed method maintains a relatively competitive AUC distribution under most noise settings, as evidenced by a higher median and mean, a narrower interquartile range, and fewer outliers (corresponding to markedly poor performance). This suggests that its advantages are not confined to threshold-dependent classification metrics, but also extend to sample-level discriminability. Figure 8 also shows that the boxplots of AUC obtained under mixed-noise conditions appear more scattered than those under single-noise conditions, which is broadly consistent with the pattern observed for F1.

## 5.2.2.2 Diferent number of evidences

Figure 9 shows that as the number of basic evidences (trees) increases, the performance of most methods generally follows a pattern of improvement followed by stabilization. When the number of evidences is small, individual tree sources exhibit strong randomness, and the fusion result is more susceptible to incidental bias. As the number of evidences increases, both tree diversity and collective stability improve, leading to better classification performance. However, with further increases, the complementary information provided by additional trees gradually diminishes, and the performance gains tend to saturate. Certain approaches, such as Dempster and HEF, fail to handle the increased conflict efectively as the number of evidence sources grows, resulting in performance degradation. By contrast, the proposed method maintains consistently strong and more stable performance across diferent numbers of evidences, indicating its ability to balance evidence complementarity with conflict suppression efectively.

## 5.2.2.3 Diferent base model

To further examine the proposed framework’s dependence on the evidence generators, this section keeps all other experimental settings unchanged and replaces the original combination of multiple random DTs used to construct the evidence bodies with a set of seven diferent base evidence generators: DT, Support Vector Classification (SVC), Logistic Regression (LR), Gaussian Process Classifier (GPC), K-Nearest Neighbors (KNN), Multilayer Perceptron (MLP), and Gaussian Naive Bayes (GNB). All models use the default parameters provided by scikit-learn, and the fusion results of diferent DST methods are compared accordingly.

Table 4 presents the corresponding results. Compared with the relatively stable DT-based setting, replacing the base models introduces greater overall performance fluctuations, indicating that evidence quality, output diversity, and fusion compatibility all afect the final decision. Under this setting, the proposed framework achieves an ACC of 83.42, a PRE of 81.21, a REC of 78.34, and an F1 score of 78.43, with corresponding average ranks of 4.44, 4.91, 4.64, and 4.81. Overall, it remains competitive and outperforms most of the compared DST methods. Although the set of base evidence generators includes several relatively weak models, almost all DST methods achieve favorable performance after evidence fusion. Notably, the proposed framework maintains comparatively stable overall performance after the replacement of base models, as reflected by its lower standard deviations in both performance and ranking, demonstrating the framework’s strong adaptive evidence fusion capability when confronted with heterogeneous base models of varying quality.

![](images/3afbd6270c2b15a850cf9894a7411362f8b26a36195412746ca511336e7f1812.jpg)  
Figure 6: F1-score under hybrid noise (gaussian feature noise and random label)

## 5.2.3 Parameter Sensitivity

To evaluate the influence of key parameters on the performance of the proposed framework, this section conducts parameter sensitivity analysis from two perspectives: historical context partitioning and experience-weight generation. Specifically, all other hyperparameters are fixed while varying the number of clusters obtained by spectral clustering, and the resulting changes in ACC, PRE, REC, and F1 under diferent clustering granularities are examined to assess the sensitivity of historical experience modeling to the context partition scale. Subsequently, a joint grid search is performed over the two sensitivity coeficients, η and γ , in the regret-rejoice mechanism. By plotting F1 heatmaps and ACC three-dimensional surfaces across multiple datasets, we further investigate the stability of the experience-weight mechanism and the distribution of high-performing regions under diferent parameter combinations.

Figure 10 presents the performance variation of the proposed model under diferent numbers of clusters. Overall, model performance does not increase monotonically with the number of clusters, but instead exhibits dataset-dependent local optima associated with sample size, class structure, and feature dimensionality. For small-sample datasets such as BMNP, Wine, and Zoo, the performance curves fluctuate more noticeably, indicating that when limited samples are available for historical experience statistics, changes in the number of clusters directly afect the stability of the regret-rejoice values within each context. An overly fine-grained partition may lead to insuficient samples within individual clusters, making the estimation of experience weights more susceptible to local sample distributions. In contrast, larger datasets such as Diabetes, German, Sports, and Vehicle show relatively stable performance across a wider range of cluster numbers, suggesting that suficient historical samples can mitigate the statistical uncertainty introduced by changes in context partitioning. In addition, multi-class and imbalanced datasets such as Accent and Zoo are also more sensitive to the number of clusters, possibly because minority-class samples are more easily diluted across diferent clusters, thereby afecting local reliability modeling. These observations suggest that the role of the cluster number is to balance contextual expressiveness and statistical stability: too few clusters may fail to capture reliability diferences across sample regions, whereas too many clusters may amplify estimation fluctuations in small-sample or minority-class scenarios. A moderate clustering granularity is therefore generally more conducive to stable fusion performance.

![](images/dc72a9ef6e30f28f578c6efaf5d8d1ccdd23a8eb0dd9b50cff70094544cd7afe.jpg)  
Figure 7: F1-score under hybrid noise (mean feature noise and random label)

Figure 11 further reports the joint sensitivity results for η and γ . The F1 heatmaps on the left show that the optimal parameter combinations vary across datasets, indicating that the reward and penalty intensities should be adapted to the sample size, class distribution, and feature structure of each dataset. For datasets with relatively suficient samples, such as WDBC, Diabetes, Sports, and Vehicle, high performance can be maintained over a relatively broad range of parameter combinations, forming continuous high-performance regions. This suggests that the historical experience weights in these scenarios are relatively tolerant to variations in η and γ . In contrast, small-sample datasets such as BMNP, Wine, WPBC, and Zoo exhibit more pronounced changes in the heatmaps, indicating that when historical samples are limited or class proportions are imbalanced, variations in the reward and penalty coeficients can more easily amplify local statistical errors in experience estimation and thus lead to performance fluctuations. For multi-class datasets such as Accent and Turkish, parameter sensitivity may also be jointly afected by inter-class separability and the reliability estimation of minority classes. The ACC surfaces on the right show a similar pattern: most datasets exhibit local fluctuations rather than a consistent increase along a single parameter direction, suggesting that excessively strong rewards or penalties may disrupt the balance of evidence weights. Overall, the efective regions of η and γ are usually not isolated points but are distributed across several neighboring parameter combinations, demonstrating a certain degree of parameter robustness in the proposed historical experience-driven mechanism. Meanwhile, the diferences in optimal regions across datasets indicate that parameter selection should be moderately adjusted according to data scale, class imbalance, and feature complexity, so as to achieve a more appropriate balance between preserving historically reliable evidence and suppressing misleading evidence.

Our  BHF  CatBoost  Dempster  Deng  Dubois  EC-FMDS  HEF  LightGBM  Murphy  XGBoost  Yager  
![](images/4b210ff0882b5d344aed9bab2fb79ecb6dfc6ec671795128f1ae97fdf359720b.jpg)

Figure 8: Boxplots of average AUC for diferent models under various noise types and levels  
![](images/2283993b7957b6a800c0c25230b1af128117f4818b826ddf1a16d777da858248.jpg)  
Figure 9: Comparative performance of models across subtree sizes

## 5.2.4 Ablation study

To examine the independent contribution of the key modules in the proposed framework, we conducted ablation experiments under the same experimental setting as that used in Section 5.2.1. The ablation variants include: replacing CCM with Dempster’s conflict coeficient K; removing the clusters obtained by spectral clustering; separately removing the weights associated with rejoice and regret; completely removing historical experience weighting; retaining only the Dubois combination term or only the historical experience weighting term; and replacing the proposed decision rule with pignistic decision. All results are summarized in terms of the average F1 and AUC over the 16 datasets. The absolute and relative changes with respect to Full are also reported to characterize the influence of each component on classification performance and discriminative ability.

As shown in Table 5, the Full model achieves the best performance in both F1 and AUC. From the perspective of module contribution, historical experience weighting has the most pronounced impact. w/o History Weighting leads to a 3.21% decrease in F1 and a 5.03% decrease in AUC, producing the largest AUC loss among all variants. This suggests that historical feedback improves not only the final classification results but also the framework’s overall ability to discriminate among evidence sources. In comparison, w/o Rejoice and w/o Regret result in F1 losses of 1.54% and 1.98%, respectively, indicating that both positive reinforcement and negative penalty contribute to reliability estimation. The larger efect of the regret term suggests that suppressing misleading evidence may be more critical in conflict-aware evidence fusion. w/o Cluster decreases F1 and AUC by 2.05% and 1.57%, respectively, further demonstrating the contextual dependency of evidence-source reliability. The degradation observed for w/ Dempster K indicates that the traditional conflict coeficient is insuficient to fully capture the internal uncertainty introduced by non-singleton focal elements, whereas the proposed CCM can more finely characterize both inter-evidence conflict and evidence non-specificity. Dubois Only sufers the largest F1 loss, suggesting that relying solely on conservative uncertainty preservation weakens the decisiveness of final classification. Although History Weighted Only can exploit historical reliability information, it still fails to achieve the stability of the full model without Dubois-type uncertainty preservation and conflict-adaptive regulation; consequently, its AUC degradation is larger than that of Dubois Only.

Table 4: Performance comparison under diferent base models (mean ± std) with average rank
<table><tr><td>Type</td><td>Model</td><td>ACC</td><td>ACC_rank</td><td>PRE</td><td> $\mathrm { P R E \mathrm { { \_ r a n k } } }$ </td><td>REC</td><td>REC_rank</td><td>F1</td><td>F1_rank</td></tr><tr><td rowspan="7">Base evidence</td><td>SVC</td><td> $7 6 . 8 1 \pm 1 7 . 4 5$ </td><td> $9 . 2 5 { \pm } 5 . 7 1 $ </td><td> $7 6 . 4 8 { \pm } 1 6 . 1 4$ </td><td> $8 . 9 8 { \pm } 5 . 0 9$ </td><td> $7 5 . 2 1 { \pm } 1 6 . 8 5 $ </td><td> $8 . 0 0 { \pm } 5 . 4 3 $ </td><td> $7 4 . 3 4 { \pm } 1 7 . 6 9$ </td><td> $8 . 2 2 { \pm } 5 . 4 6$ </td></tr><tr><td>LR</td><td> $7 5 . 5 0 { \pm } 1 6 . 6 6$ </td><td> $7 . 9 7 { \pm } 4 . 4 8 $ </td><td> $6 8 . 7 5 { \scriptstyle \pm 2 2 . 9 0 }$ </td><td> $8 . 7 3 { \pm } 4 . 1 8$ </td><td> $7 0 . 6 0 { \scriptstyle \pm 1 7 . 2 1 }$ </td><td> $8 . 3 1 { \pm } 4 . 3 1 $ </td><td> $6 7 . 7 4 { \pm } 2 1 . 0 9$ </td><td> $8 . 6 2 { \pm } 4 . 1 5$ </td></tr><tr><td>GPC</td><td> $6 3 . 9 5 { \scriptstyle \pm 2 1 . 6 7 }$ </td><td> $1 0 . 6 4 { \pm } 4 . 9 8 $ </td><td> $6 0 . 5 8 { \pm } 2 3 . 3 9$ </td><td> $1 0 . 9 5 { \scriptstyle \pm 4 . 6 9 }$ </td><td> $5 6 . 6 1 \pm 2 1 . 2 3 $ </td><td> $1 1 . 1 9 { \scriptstyle \pm 4 . 8 1 }$ </td><td> $5 0 . 0 0 { \scriptstyle \pm 2 5 . 8 5 }$ </td><td> $1 1 . 6 1 { \pm } 4 . 4 1 $ </td></tr><tr><td>DT</td><td> $7 8 . 2 0 { \scriptstyle \pm 1 } 3 . 4 2$ </td><td> $9 . 3 4 { \pm } 5 . 5 5 $ </td><td> $7 4 . 2 7 { \pm } 1 6 . 5 9$ </td><td> $9 . 7 5 { \pm } 5 . 3 9 $ </td><td> $7 4 . 3 9 { \pm } 1 5 . 9 6 $ </td><td> $9 . 0 6 { \pm } 5 . 4 5 $ </td><td> $7 3 . 9 3 { \pm } 1 6 . 4 1$ </td><td> $8 . 9 5 { \pm } 5 . 3 9$ </td></tr><tr><td>KNN</td><td> $7 5 . 0 0 { \pm } 1 4 . 1 0 $ </td><td> $9 . 6 7 { \pm } 4 . 1 6$ </td><td> $6 4 . 2 2 { \scriptstyle \pm 2 2 . 7 7 }$ </td><td> $1 0 . 6 9 { \scriptstyle \pm 4 . 5 8 }$ </td><td> $6 6 . 1 0 { \pm } 1 8 . 1 7$ </td><td> $1 0 . 8 0 { \scriptstyle \pm 4 . 3 5 }$ </td><td> $6 3 . 3 6 { \pm } 2 0 . 7 6$ </td><td> $1 0 . 9 7 { \scriptstyle \pm 4 . 2 6 }$ </td></tr><tr><td>MLP</td><td> $6 3 . 9 5 { \scriptstyle \pm 2 1 . 6 7 }$ </td><td> $1 0 . 6 4 { \pm } 4 . 9 8 $ </td><td> $6 0 . 5 8 { \pm } 2 3 . 3 9$ </td><td> $1 0 . 9 5 { \scriptstyle \pm 4 . 6 9 }$ </td><td> $5 6 . 6 1 \pm 2 1 . 2 3 $ </td><td> $1 1 . 1 9 { \scriptstyle \pm 4 . 8 1 }$ </td><td> $5 0 . 0 0 { \scriptstyle \pm 2 5 . 8 5 }$ </td><td> $1 1 . 6 1 { \pm } 4 . 4 1 $ </td></tr><tr><td>GNB</td><td> $8 0 . 3 2 { \scriptstyle \pm 1 3 . 4 4 }$ </td><td> $7 . 5 8 { \pm } 5 . 0 2 $ </td><td> $7 8 . 3 0 { \pm } 1 5 . 1 6$ </td><td> $7 . 6 9 { \pm } 4 . 7 1 $ </td><td> $7 5 . 8 8 { \pm } 1 6 . 0 8 $ </td><td> $7 . 6 9 { \pm } 4 . 9 8 $ </td><td> $7 6 . 2 0 { \scriptstyle \pm 1 } 5 . 9 7$ </td><td> $7 . 5 5 { \pm } 4 . 8 7$ </td></tr><tr><td rowspan="9">DST</td><td>Deng</td><td> $8 0 . 5 1 { \pm } 1 1 . 3 3 $ </td><td> $7 . 3 6 { \pm } 4 . 7 3$ </td><td> $7 7 . 3 3 { \pm } 1 5 . 2 1 $ </td><td> $7 . 6 7 { \pm } 4 . 4 9$ </td><td> $7 5 . 3 0 { \pm } 1 5 . 0 2$ </td><td> $7 . 4 1 { \pm } 4 . 5 1 $ </td><td> $7 5 . 0 9 { \scriptstyle \pm 1 5 . 4 6 }$ </td><td> $7 . 3 6 { \pm } 4 . 4 6$ </td></tr><tr><td>HEF</td><td> $8 0 . 2 0 { \scriptstyle \pm 1 } 3 . 5 0 $ </td><td> $5 . 9 7 { \scriptstyle \pm 3 . 5 5 }$ </td><td> $7 5 . 8 4 \pm 1 9 . 7 1$ </td><td> $6 . 8 6 { \pm } 3 . 5 5 $ </td><td> $7 5 . 3 2 { \pm } 1 6 . 5 2$ </td><td> $6 . 4 1 { \pm } 3 . 3 8 $ </td><td> $7 3 . 9 7 { \scriptstyle \pm 1 8 . 8 8 }$ </td><td> $6 . 6 6 { \pm } 3 . 5 3 $ </td></tr><tr><td>Dubois</td><td> $7 8 . 2 2 { \scriptstyle \pm 1 } 5 . 7 4$ </td><td> $5 . 9 4 \pm 3 . 7 7$ </td><td> $7 2 . 9 1 { \pm } 2 2 . 1 7$ </td><td> $6 . 9 5 { \scriptstyle \pm 3 . 5 0 }$ </td><td> $7 3 . 2 5 { \pm } 1 6 . 8 5 $ </td><td> $6 . 5 2 { \scriptstyle \pm 3 . 3 9 }$ </td><td> $7 1 . 2 3 { \pm } 2 0 . 3 1$ </td><td> $6 . 8 8 { \pm } 3 . 5 1 $ </td></tr><tr><td>Dempster</td><td> $8 0 . 1 4 \pm 1 3 . 4 9 $ </td><td> $6 . 1 6 { \pm } 3 . 5 6$ </td><td> $7 5 . 7 9 { \scriptstyle \pm 1 9 . 6 9 }$ </td><td> $7 . 0 0 { \scriptstyle \pm 3 . 5 8 }$ </td><td> $7 5 . 2 9 { \scriptstyle \pm 1 6 . 4 8 }$ </td><td> $6 . 5 6 { \pm } 3 . 3 9$ </td><td> $7 3 . 9 3 { \pm } 1 8 . 8 5 $ </td><td> $6 . 8 1 \pm 3 . 5 1$ </td></tr><tr><td>EC-FMDS</td><td> $8 2 . 8 9 { \pm } 1 1 . 0 3 $ </td><td> $4 . 3 3 { \pm } 2 . 5 2 $ </td><td> $8 1 . 3 8 { \pm } 1 4 . 6 3 $ </td><td> $5 . 0 0 { \scriptstyle \pm 3 . 0 7 } $ </td><td> $7 7 . 7 6 { \pm } 1 4 . 5 1$ </td><td> $5 . 0 2 { \pm } 2 . 5 0 $ </td><td> $7 7 . 9 2 { \pm } 1 5 . 1 3 $ </td><td> $4 . 9 7 { \scriptstyle \pm 2 . 7 0 }$ </td></tr><tr><td>Murphy</td><td> $7 8 . 0 8 { \pm } 1 5 . 2 9$ </td><td> $6 . 4 5 { \scriptstyle \pm 4 . 4 3 }$ </td><td> $7 4 . 0 2 { \pm } 2 0 . 0 0$ </td><td> $6 . 0 5 { \scriptstyle \pm 3 . 8 6 }$ </td><td> $7 2 . 5 9 { \pm } 1 7 . 2 3$ </td><td> $6 . 8 8 { \pm } 4 . 1 4 $ </td><td> $7 1 . 2 9 { \pm } 1 9 . 6 1 $ </td><td> $6 . 5 9 { \scriptstyle \pm 4 . 2 4 }$ </td></tr><tr><td>BHF</td><td> $8 1 . 7 2 { \scriptstyle \pm 1 1 . 1 0 }$ </td><td> $5 . 5 2 { \pm } 3 . 0 2 $ </td><td> $\mathbf { 8 1 . 4 6 { \pm } 1 4 . 3 6 }$ </td><td> $5 . 7 5 { \scriptstyle \pm 3 . 2 6 }$ </td><td> $7 5 . 7 7 { \scriptstyle \pm 1 5 . 5 7 }$ </td><td> $6 . 5 2 { \pm } 3 . 0 2 $ </td><td> $7 5 . 6 2 { \pm } 1 6 . 4 0$ </td><td> $6 . 4 8 { \pm } 3 . 0 6$ </td></tr><tr><td> $\mathrm { Y a g e r }$ </td><td> $7 7 . 7 0 { \scriptstyle \pm 1 6 . 2 1 }$ </td><td> $6 . 3 1 { \pm } 4 . 1 0 $ </td><td> $7 2 . 0 7 { \scriptstyle \pm 2 2 . 4 9 }$ </td><td> $7 . 3 6 { \pm } 3 . 7 7$ </td><td> $7 2 . 7 7 { \scriptstyle \pm 1 7 . 2 3 }$ </td><td> $6 . 8 6 { \scriptstyle \pm 3 . 6 7 }$ </td><td> $7 0 . 6 6 { \pm } 2 0 . 8 4$ </td><td> $7 . 2 0 { \scriptstyle \pm 3 . 7 6 }$ </td></tr><tr><td>Our</td><td> $\mathbf { 8 3 . 4 2 \pm 1 0 . 1 3 }$ </td><td> $4 . 4 4 \pm 3 . 0 6$ </td><td> $8 1 . 2 1 { \pm } 1 3 . 7 1 $ </td><td> $\mathbf { 4 . 9 1 \pm 2 . 9 8 }$ </td><td> $\mathbf { 7 8 . 3 4 \pm 1 3 . 9 0 }$ </td><td> $\mathbf { 4 . 6 4 } \pm 2 . 8 5$ </td><td> $\mathbf { 7 8 . 4 3 \pm 1 4 . 2 8 }$ </td><td> $\mathbf { 4 . 8 1 } { \pm } 3 . 0 4 $ </td></tr></table>

![](images/1a5725e276759618ff3c5a9d68591ea0e3eab8cfdc479e149ae0e5bab9fa6d55.jpg)  
Figure 10: Sensitivity analysis of clustering performance with respect to the number of clusters

![](images/ea22bd7255820e4e6bb9710c46a1ffc4ace31840815f13bc8aaaf480220a0008.jpg)  
Figure 11: Joint sensitivity analysis of hyperparameters η and γ across multiple datasets

Overall, the ablation results support the core design logic of the proposed framework: CCM evaluates the conflict and uncertainty of the evidence set, historical experience weighting characterizes the long-term reliability of diferent evidence sources under similar contexts, and the hybrid fusion rule adaptively balances reliable consensus with uncertainty preservation. These components jointly improve the classification and discriminative performance of the model across heterogeneous datasets.

## 6 Conclusion

This paper has presented a unified evidence reasoning framework that combines CCM with historical-experience-driven weighting for robust multi-source decision making under uncertainty. The central premise of this work is that two persistent gaps in existing DST methods, namely the treatment of conflict and uncertainty as independent quantities and the neglect of long-term evidence source behavior, can be jointly addressed through complementary statistical and behavioral mechanisms. The CCM provides a unified scalar assessment of both inter-evidence inconsistency and intra-evidence non-specificity, grounded in a similarity measure whose mathematical properties are formally established. The historical experience weighting scheme exploits the observation that evidence sources operating repeatedly across heterogeneous decision contexts carry learnable reliability profiles, which RT makes it possible to quantify in terms of counterfactual regret and rejoice when fusion decisions deviate from ground truth. These two mechanisms are integrated through a hybrid combination rule that adaptively balances uncertainty preservation against weighted consensus, followed by a belief-interval decision strategy that produces deterministic classifications without discarding the epistemic uncertainty preserved by the fusion process.

Table 5: Ablation study results averaged over 16 datasets
<table><tr><td>Variant</td><td>F1</td><td>∆F1 (abs.)</td><td>∆F1 (rel.)</td><td>AUC</td><td>∆AUC(abs.)</td><td>∆AUC(rel.)</td></tr><tr><td>Full Model</td><td>85.78±15.42</td><td></td><td></td><td>93.30±5.39</td><td></td><td></td></tr><tr><td>w/Pignistic</td><td>84.62±15.63</td><td>-1.16</td><td>-1.35%</td><td>92.91±5.83</td><td>-0.39</td><td>-0.42%</td></tr><tr><td>w/ Dempster K</td><td>84.31±16.03</td><td>-1.47</td><td>-1.71%</td><td>92.14±5.75</td><td>-1.16</td><td>-1.24%</td></tr><tr><td>w/o History Weighting</td><td>83.03±15.56</td><td>-2.75</td><td>-3.21%</td><td>88.61±5.87</td><td>-4.69</td><td>-5.03%</td></tr><tr><td>w/o Cluster</td><td>83.73±16.17</td><td>-2.05</td><td>-2.39%</td><td>91.73±5.83</td><td>-1.57</td><td>-1.68%</td></tr><tr><td>w/o Regret</td><td>84.08±16.02</td><td>-1.70</td><td>-1.98%</td><td>92.08±5.78</td><td>-1.22</td><td>-1.31%</td></tr><tr><td>w/o Rejoice</td><td>84.46±15.25</td><td>-1.32</td><td>-1.54%</td><td>92.51±5.79</td><td>-0.79</td><td>-0.85%</td></tr><tr><td>Dubois Only</td><td>82.97±16.02</td><td>-2.81</td><td>-3.28%</td><td>92.10±5.82</td><td>-1.20</td><td>-1.29%</td></tr><tr><td>History Weighted Only</td><td>83.69±16.20</td><td>-2.09</td><td>-2.44%</td><td>91.36±5.72</td><td>-1.94</td><td>-2.08%</td></tr></table>

The theoretical contributions of this work are threefold. First, the CCM ofers a more faithful characterization of evidential stability than existing measures that examine conflict and uncertainty in isolation, and its five proven properties ensure consistent behavior across frame refinements and extreme cases. Second, the integration of SC with RT establishes a principled paradigm for translating historical decision feedback into context-dependent evidence weights, extending the scope of DST beyond instantaneous assessment toward adaptive reliability modeling. Third, the hybrid combination rule provides a conflict-aware mechanism for balancing the conservatism of fusion with the decisiveness of weighted consensus, parameterized by a single GCCD conflict indicator that requires no manual tuning of mixture coeficients.

On the practical side, experiments conducted across 16 heterogeneous datasets demonstrate that the proposed framework achieves competitive performance among both DST-based and ensemble learning methods, with the highest average F1 score (85.78) and mean AUC (93.30). The framework maintains its advantage under noisy training conditions, varying numbers of evidence sources, and diferent base evidence generators, indicating that the gains stem from the fusion architecture rather than from favorable data conditions. Ablation analysis confirms that historical experience weighting, CCM-based conflict assessment, and the hybrid combination rule each contribute meaningfully to performance, with the historical experience component exerting the strongest individual efect.

Several limitations suggest directions for further research. The current framework constructs BPAs through standard classifiers rather than dedicated evidence generation methods, and the quality of these assignments directly afects downstream fusion performance (J. Zhou et al., 2019); developing domain-specific BPA generation strategies could improve both accuracy and computational eficiency. The computational cost of the CCM scales with the number of focal element pairs and may become prohibitive when the frame of discernment is large or the number of evidence sources is high (L. Chen et al., 2023); approximate or distributed computation strategies would extend the framework’s applicability to real-time decision settings. The sensitivity parameters governing the regret-rejoice mechanism, while demonstrating reasonable robustness across the datasets examined, may benefit from data-adaptive calibration rather than uniform specification. Future work should also explore the extension of the historical experience paradigm to heterogeneous multi-modal data environments, where evidence sources of fundamentally diferent types must be fused, and to nonstationary settings where the decision context itself evolves over time (Strelet et al., 2025; Zhang et al., 2025).

## Appendix A

Table A.1: Comparison of ACC on 16 datasets
<table><tr><td>Model</td><td>BHF</td><td>CatBoost</td><td>Dempster</td><td>Deng</td><td>Dubois</td><td>EC-FMDS</td><td>HEF</td><td>LightGBM</td><td>Murphy</td><td>Our</td><td>XGBoost</td><td>Yager</td></tr><tr><td>Accent</td><td>69.01±3.95</td><td>60.79±4.36</td><td>67.47±4.33</td><td>69.00±2.00</td><td>65.96±1.64</td><td>65.95±2.31</td><td>62.00±3.52</td><td>72.94±3.47</td><td>66.57±6.11</td><td>69.05±12.42</td><td>71.41±5.50</td><td>60.82±6.98</td></tr><tr><td>BMNP</td><td>49.06±6.72</td><td>55.62±4.15</td><td>58.85±10.26</td><td>52.40±7.82</td><td>57.19±5.44</td><td>47.71±4.77</td><td>55.73±10.67</td><td>66.67±5.89</td><td>54.27±17.33</td><td>71.35±34.12</td><td>61.77±8.15</td><td>52.40±7.82</td></tr><tr><td>Diabetes</td><td>95.77±0.77</td><td>93.08±2.26</td><td>93.85±1.88</td><td>95.58±3.40</td><td>96.73±1.46</td><td>97.31±0.77</td><td>96.73±1.71</td><td>91.15±1.47</td><td>95.00±0.99</td><td>97.69±1.26</td><td>94.23±2.55</td><td>95.96±0.74</td></tr><tr><td>Forest</td><td>84.89±2.93</td><td>86.22±3.31</td><td>86.23±4.71</td><td>86.03±2.94</td><td>87.56±2.50</td><td>85.46±3.19</td><td>82.98±4.40</td><td>87.75±4.89</td><td>85.46±4.34</td><td>88.72±5.11</td><td>88.14±3.59</td><td>84.71±1.36</td></tr><tr><td>German</td><td>66.60±1.80</td><td>73.20±2.38</td><td>69.50±4.12</td><td>68.70±1.74</td><td>70.10±2.76</td><td>69.20±4.88</td><td>68.60±2.58</td><td>74.40±2.19</td><td>70.00±2.79</td><td>74.80±3.68</td><td>74.00±4.36</td><td>68.20±2.39</td></tr><tr><td>Hfcr</td><td>79.95±3.85</td><td>79.59±3.89</td><td>82.96±7.38</td><td>81.29±4.80</td><td>80.94±7.79</td><td>83.63±5.85</td><td>79.60±7.25</td><td>84.96±7.54</td><td>81.95±6.16</td><td>84.29±4.22</td><td>80.95±8.50</td><td>82.62±4.70</td></tr><tr><td>Ionosphere</td><td>89.18±4.66</td><td>89.46±3.28</td><td>90.33±5.03</td><td>91.75±2.97</td><td>88.90±3.73</td><td>91.75±4.84</td><td>90.03±3.76</td><td>91.46±3.51</td><td>93.17±3.33</td><td>94.03±5.44</td><td>89.76±4.62</td><td>89.47±6.02</td></tr><tr><td>ISPY</td><td>96.73±1.82</td><td>99.07±1.08</td><td>95.80±2.83</td><td>95.35±1.84</td><td>94.40±3.45</td><td>92.49±10.24</td><td>92.57±5.44</td><td>99.07±1.08</td><td>97.67±1.79</td><td>94.38±5.14</td><td>99.07±1.08</td><td>90.23±3.81</td></tr><tr><td>Sonar</td><td>71.63±1.84</td><td>73.08±4.15</td><td>68.75±5.06</td><td>77.40±4.54</td><td>74.04±4.00</td><td>75.96±7.11</td><td>70.67±5.95</td><td>77.88±5.98</td><td>76.92±6.84</td><td>90.87±12.10</td><td>75.48±3.28</td><td>75.48±6.35</td></tr><tr><td>Sports</td><td>77.70±3.24</td><td>80.30±1.71</td><td>78.30±2.68</td><td>77.20±1.18</td><td>77.20±3.12</td><td>79.30±2.78</td><td>77.10±2.43</td><td>80.50±3.17</td><td>80.30±3.88</td><td>87.70±7.14</td><td>79.20±1.57</td><td>74.80±3.41</td></tr><tr><td>Turkish</td><td>67.25±2.22</td><td>65.50±4.04</td><td>67.75±5.85</td><td>67.25±3.59</td><td>66.25±4.11</td><td>67.75±3.40</td><td>63.50±5.80</td><td>78.25±2.50</td><td>69.25±0.96</td><td>86.00±14.21</td><td>74.50±4.80</td><td>66.50±7.59</td></tr><tr><td>Vehicle</td><td>73.41±2.80</td><td>68.56±1.17</td><td>69.74±1.52</td><td>72.82±1.35</td><td>70.93±4.20</td><td>74.00±1.32</td><td>68.20±2.39</td><td>72.93±2.02</td><td>73.88±2.64</td><td>77.91±7.89</td><td>73.41±1.88</td><td>69.38±2.51</td></tr><tr><td>WDBC</td><td>95.25±1.56</td><td>94.55±1.84</td><td>93.67±1.73</td><td>95.43±1.35</td><td>93.15±1.04</td><td>94.73±1.67</td><td>93.67±2.64</td><td>95.08±2.99</td><td>94.90±2.33</td><td>98.42±1.05</td><td>94.03±3.06</td><td>95.08±0.99</td></tr><tr><td>Wine</td><td>93.23±4.93 68.71±6.12</td><td>94.95±3.86 79.29±4.55</td><td>93.80±5.00</td><td>95.52±3.18</td><td>93.24±3.19</td><td>94.37±6.01</td><td>89.87±4.37</td><td>92.68±3.88</td><td>96.06±3.88</td><td>98.89±2.22</td><td>93.27±3.17</td><td>88.79±4.45</td></tr><tr><td>WPBC</td><td>91.12±6.77</td><td>90.04±9.55</td><td>73.74±4.33 94.04±6.94</td><td>68.17±7.87 93.15±6.56</td><td>73.23±4.14 93.08±1.95</td><td>73.24±4.36</td><td>73.73±3.72</td><td>75.78±5.10</td><td>66.70±6.45</td><td>80.89±14.21</td><td>74.78±3.96</td><td>73.23±1.88</td></tr><tr><td>Zoo</td><td></td><td></td><td></td><td></td><td></td><td>92.12±7.22</td><td>91.12±3.71</td><td>88.15±4.45</td><td>92.12±5.55</td><td>97.12±5.77</td><td>92.12±4.49</td><td>93.15±8.03</td></tr></table>

Table A.2: Comparison of PRE on 16 datasets
<table><tr><td>Model</td><td>BHF</td><td>CatBoost</td><td>Dempster</td><td>Deng</td><td>Dubois</td><td>EC-FMDS</td><td>HEF</td><td>LightGBM</td><td>Murphy</td><td>Our</td><td>XGBoost</td><td>Yager</td></tr><tr><td>Accent</td><td>62.90±3.39</td><td>60.51±13.33</td><td>68.26±4.36</td><td>63.51±3.49</td><td>60.13±2.21</td><td>61.48±3.65</td><td>63.55±4.92</td><td>68.89±8.55</td><td>60.92±7.25</td><td>66.04±11.57</td><td>68.75±4.72</td><td>54.61±8.64</td></tr><tr><td>BMNP</td><td>34.28±7.75</td><td>34.31±4.61</td><td>36.71±23.19</td><td>45.16±10.35</td><td>43.11±12.96</td><td>42.34±11.65</td><td>37.67±15.76</td><td>39.06±6.48</td><td>51.11±24.70</td><td>68.65±36.95</td><td>45.33±12.11</td><td>41.44±10.14</td></tr><tr><td>Diabetes</td><td>95.30±0.70</td><td>92.53±2.56</td><td>93.21±2.03</td><td>95.31±3.84</td><td>96.31±1.72</td><td>97.04±0.91</td><td>96.32±1.97</td><td>91.00±1.90</td><td>94.50±1.19</td><td>97.38±1.56</td><td>93.80±2.81</td><td>95.48±0.94</td></tr><tr><td>Forest</td><td>84.95±1.90</td><td>86.59±2.94</td><td>86.71±4.58</td><td>86.30±1.56</td><td>87.63±0.91</td><td>85.38±2.55</td><td>85.36±4.41</td><td>88.39±3.66</td><td>84.88±3.64</td><td>89.07±5.92</td><td>88.67±2.77</td><td>85.00±2.77</td></tr><tr><td>German</td><td>62.54±3.11</td><td>67.56±3.83</td><td>64.73±4.11</td><td>63.09±2.58</td><td>65.24±3.25</td><td>64.26±4.79</td><td>63.51±3.25</td><td>69.15±3.25</td><td>65.38±2.90</td><td>71.69±3.81</td><td>68.74±5.56</td><td>63.63±2.55</td></tr><tr><td>Hfcr</td><td>77.46±4.67</td><td>77.76±4.20</td><td>80.63±8.30</td><td>78.99±5.00</td><td>78.04±8.99</td><td>81.13±6.62</td><td>76.92±8.25</td><td>82.96±8.59</td><td>79.12±6.88</td><td>82.32±4.50</td><td>78.34±9.80</td><td>80.61±4.83</td></tr><tr><td>Ionosphere</td><td>89.07±4.99</td><td>89.76±3.52</td><td>89.52±5.46</td><td>90.85±2.64</td><td>88.11±4.05</td><td>91.21±5.14</td><td>88.77±3.78</td><td>91.88±4.10</td><td>93.05±3.37</td><td>93.52±6.04</td><td>90.16±5.40</td><td>89.07±6.16</td></tr><tr><td>ISPY</td><td>95.35±3.03</td><td>99.38±0.72</td><td>96.01±2.76</td><td>94.55±1.37</td><td>93.56±4.54</td><td>92.03±12.77</td><td>93.43±6.98</td><td>99.38±0.72</td><td>97.00±2.82</td><td>92.06±5.94</td><td>99.38±0.72</td><td>88.67±3.58</td></tr><tr><td>Sonar</td><td>71.58±1.91</td><td>74.12±2.66</td><td>70.23±5.61</td><td>77.40±4.55</td><td>74.58±3.34</td><td>76.37±7.39</td><td>71.77±7.24</td><td>78.24±6.35</td><td>77.05±6.96</td><td>91.00±12.12</td><td>75.46±3.31</td><td>75.47±6.28</td></tr><tr><td>Sports</td><td>75.98±3.43</td><td>79.38±2.17</td><td>77.42±2.88</td><td>75.58±1.26</td><td>75.80±3.66</td><td>77.77±2.98</td><td>75.49±2.79</td><td>79.27±3.30</td><td>78.76±4.19</td><td>86.68±7.64</td><td>77.81±1.93</td><td>72.90±3.79</td></tr><tr><td>Turkish</td><td>68.04±3.38</td><td>66.44±4.37</td><td>68.36±6.03</td><td>67.25±3.99</td><td>67.33±3.03</td><td>69.78±2.62</td><td>66.48±4.82</td><td>79.41±2.24</td><td>69.63±1.00</td><td>86.04±14.46</td><td>74.80±4.44</td><td>69.87±7.70</td></tr><tr><td>Vehicle</td><td>73.11±3.37</td><td>66.73±0.72</td><td>68.92±2.02</td><td>72.32±1.46</td><td>70.50±4.96</td><td>73.56±1.70</td><td>67.43±2.83</td><td>71.88±2.65</td><td>73.23±2.77</td><td>77.42±8.69</td><td>73.03±1.88</td><td>68.61±2.48</td></tr><tr><td>WDBC</td><td>94.92±1.80 93.35±4.99</td><td>94.72±1.95</td><td>93.00±1.99</td><td>95.08±1.59</td><td>92.46±1.22</td><td>94.25±1.90</td><td>93.04±3.02</td><td>94.92±3.66</td><td>94.83±2.78</td><td>98.12±1.11</td><td>93.45±3.35</td><td>94.48±1.02</td></tr><tr><td>Wine</td><td></td><td>95.32±3.82</td><td>94.71±3.90</td><td>95.54±3.16</td><td>93.82±2.78</td><td>94.69±5.67</td><td>91.23±3.67</td><td>93.30±4.01</td><td>96.18±3.74</td><td>98.90±2.21</td><td>93.72±3.34</td><td>89.59±4.42</td></tr><tr><td>WPBC</td><td>55.97±8.11 79.46±14.33</td><td>70.35±10.70 82.29±12.16</td><td>60.78±7.40</td><td>55.68±9.47</td><td>56.80±10.59</td><td>60.98±8.46</td><td>58.91±8.63</td><td>66.32±8.17</td><td>56.10±6.07</td><td>75.24±16.54</td><td>64.07±5.59</td><td>57.90±7.36</td></tr><tr><td>Zoo</td><td></td><td></td><td>89.88±14.07</td><td>83.33±16.64</td><td>82.14±9.86</td><td>78.51±16.32</td><td>75.95±3.05</td><td>71.21±10.93</td><td>81.85±12.35</td><td>92.86±14.29</td><td>80.89±9.89</td><td>83.04±22.09</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table A.3: Comparison of REC on 16 datasets
<table><tr><td>Model</td><td>BHF</td><td>CatBoost</td><td>Dempster</td><td>Deng</td><td>Dubois</td><td>EC-FMDS</td><td>HEF</td><td>LightGBM</td><td>Murphy</td><td>Our</td><td>XGBoost</td><td>Yager</td></tr><tr><td>Accent</td><td>65.16±5.77</td><td>41.79±7.27</td><td>61.05±8.66</td><td>63.24±6.96</td><td>62.17±5.32</td><td>62.86±4.20</td><td>52.56±7.72</td><td>62.17±4.57</td><td>59.75±6.52</td><td>76.71±16.43</td><td>63.38±7.60</td><td>56.99±7.94</td></tr><tr><td>BMNP</td><td>37.96±7.22</td><td>38.15±3.79</td><td>39.49±13.90</td><td>44.31±2.37</td><td>44.72±11.59</td><td>40.56±6.48</td><td>40.32±11.50</td><td>43.24±5.86</td><td>44.49±19.53</td><td>72.87±40.50</td><td>45.88±9.22</td><td>42.22±9.65</td></tr><tr><td>Diabetes</td><td>96.00±1.13</td><td>93.16±2.21</td><td>94.44±1.47</td><td>95.66±3.00</td><td>97.06±1.05</td><td>97.34±0.77</td><td>96.97±1.47</td><td>90.47±0.99</td><td>95.28±0.57</td><td>97.84±1.04</td><td>94.38±2.07</td><td>96.25±0.62</td></tr><tr><td>Forest</td><td>83.73±5.00</td><td>85.46±5.14</td><td>84.92±4.85</td><td>84.91±4.69</td><td>86.66±4.24</td><td>85.28±5.26</td><td>81.59±5.29</td><td>86.66±6.19</td><td>85.48±5.93</td><td>89.48±4.37</td><td>86.44±4.89</td><td>84.98±0.91</td></tr><tr><td>German</td><td>64.05±4.07</td><td>62.86±3.00</td><td>65.64±3.91</td><td>63.55±3.26</td><td>66.26±3.61</td><td>64.76±5.10</td><td>64.43±3.80</td><td>65.05±3.93</td><td>66.57±3.08</td><td>74.67±4.54</td><td>66.57±5.47</td><td>64.81±3.20</td></tr><tr><td>Hfcr</td><td>80.01±6.28</td><td>73.17±5.71</td><td>79.20±10.48</td><td>81.01±4.07</td><td>77.45±9.89</td><td>82.46±7.00</td><td>75.38±9.34</td><td>81.79±9.62</td><td>80.12±7.78</td><td>83.77±2.84</td><td>76.91±10.66</td><td>78.69±7.58</td></tr><tr><td>Ionosphere</td><td>87.24±5.37</td><td>87.27±3.90</td><td>90.60±5.10</td><td>91.50±4.13</td><td>88.60±3.98</td><td>90.99±5.66</td><td>90.33±4.66</td><td>89.71±3.87</td><td>92.09±3.90</td><td>93.60±5.68</td><td>87.67±5.00</td><td>88.84±5.73</td></tr><tr><td>ISPY</td><td>96.69±2.06</td><td>98.27±2.00</td><td>93.36±5.16</td><td>93.61±3.26</td><td>92.97±5.18</td><td>90.98±9.41</td><td>87.17±8.53</td><td>98.27±2.00</td><td>97.31±2.03</td><td>96.16±3.48</td><td>98.27±2.00</td><td>86.32±7.08</td></tr><tr><td>Sonar</td><td>71.52±1.85</td><td>72.35±5.00</td><td>69.45±5.06</td><td>77.14±4.76</td><td>74.27±3.81</td><td>75.86±7.06</td><td>71.13±6.35</td><td>77.59±5.89</td><td>77.10±6.96</td><td>90.84±12.01</td><td>75.34±3.31</td><td>75.42±6.28</td></tr><tr><td>Sports</td><td>76.57±4.02</td><td>77.33±1.63</td><td>74.59±3.48</td><td>75.30±2.49</td><td>74.37±3.53</td><td>77.54±3.28</td><td>74.52±2.59</td><td>78.14±4.04</td><td>78.72±4.32</td><td>87.05±7.43</td><td>76.87±1.44</td><td>71.72±4.03</td></tr><tr><td>Turkish</td><td>67.25±2.22</td><td>65.50±4.04</td><td>67.75±5.85</td><td>67.25±3.59</td><td>66.25±4.11</td><td>67.75±3.40</td><td>63.50±5.80</td><td>78.25±2.50</td><td>69.25±0.96</td><td>86.00±14.21</td><td>74.50±4.80</td><td>66.50±7.59</td></tr><tr><td>Vehicle</td><td>73.70±2.82</td><td>68.81±1.17</td><td>70.03±1.66</td><td>73.14±1.48</td><td>71.21±4.16</td><td>74.31±1.41</td><td>68.63±2.37</td><td>73.26±2.05</td><td>74.13±2.70</td><td>78.06±8.00</td><td>73.68±1.94</td><td>69.77±2.50</td></tr><tr><td>WDBC</td><td>94.97±1.52</td><td>93.65±2.13</td><td>93.71±1.57</td><td>95.21±1.25</td><td>93.10±0.96</td><td>94.55±1.64</td><td>93.71±2.28</td><td>94.74±2.48</td><td>94.31±2.13</td><td>98.55±1.16</td><td>93.99±2.92</td><td>95.22±1.17</td></tr><tr><td>Wine</td><td>93.73±4.53</td><td>95.24±3.45</td><td>93.49±5.80</td><td>96.11±2.71</td><td>93.73±3.00</td><td>94.78±5.31</td><td>90.83±3.37</td><td>93.06±3.50</td><td>96.20±3.32</td><td>98.98±2.04</td><td>93.53±2.47</td><td>88.87±4.46</td></tr><tr><td>WPBC</td><td>55.36±6.70 84.88±10.30</td><td>63.12±10.62</td><td>57.95±5.95</td><td>55.00±8.74</td><td>55.50±7.72</td><td>60.76±9.73</td><td>57.24±7.37</td><td>60.09±5.80</td><td>56.26±5.99</td><td>80.93±20.04</td><td>62.48±7.71</td><td>55.23±5.76</td></tr><tr><td>Zoo</td><td></td><td>82.74±9.76</td><td>90.12±10.78</td><td>85.60±14.13</td><td>85.36±9.43</td><td>83.93±12.20</td><td>81.43±3.50</td><td>75.60±4.91</td><td>82.74±12.20</td><td>94.64±10.71</td><td>85.71±5.83</td><td>85.36±17.70</td></tr></table>

Table A.4: Comparison of F1 on 16 datasets
<table><tr><td>Model</td><td>BHF</td><td>CatBoost</td><td>Dempster</td><td>Deng</td><td>Dubois</td><td>EC-FMDS</td><td>HEF</td><td>LightGBM</td><td>Murphy</td><td>Our</td><td>XGBoost</td><td>Yager</td></tr><tr><td>Accent</td><td>62.90±4.48</td><td>44.45±7.66</td><td>59.40±7.33</td><td>61.94±4.29</td><td>59.94±3.02</td><td>60.81±1.79</td><td>49.54±7.05</td><td>63.15±6.40</td><td>59.06±6.93</td><td>68.33±14.48</td><td>64.44±6.79</td><td>54.58±8.38</td></tr><tr><td>BMNP</td><td>35.48±7.32</td><td>36.06±4.25</td><td>35.30±14.95</td><td>43.54±4.79</td><td>42.77±11.46</td><td>40.05±7.24</td><td>37.91±13.18</td><td>40.53±6.06</td><td>44.73±19.78</td><td>68.34±38.34</td><td>44.39±10.32</td><td>39.91±8.39</td></tr><tr><td>Diabetes</td><td>95.57±0.82</td><td>92.76±2.36</td><td>93.62±1.90</td><td>95.38±3.52</td><td>96.59±1.48</td><td>97.17±0.81</td><td>96.58±1.76</td><td>90.62±1.42</td><td>94.78±0.98</td><td>97.58±1.31</td><td>93.98±2.59</td><td>95.78±0.74</td></tr><tr><td>Forest</td><td>83.90±3.72</td><td>85.80±4.27</td><td>85.59±4.75</td><td>85.22±3.45</td><td>86.66±3.17</td><td>85.00±4.00</td><td>82.50±4.55</td><td>87.20±5.32</td><td>84.93±4.93</td><td>88.77±5.44</td><td>87.25±4.04</td><td>84.57±1.26</td></tr><tr><td>German</td><td>62.62±3.01</td><td>63.75±3.39</td><td>65.02±4.14</td><td>63.25±2.84</td><td>65.59±3.36</td><td>64.25±4.90</td><td>63.82±3.41</td><td>65.99±4.22</td><td>65.75±3.00</td><td>72.23±4.09</td><td>67.24±5.58</td><td>63.85±2.69</td></tr><tr><td>Hfcr</td><td>78.03±4.83</td><td>74.49±5.52</td><td>79.55±9.67</td><td>79.52±4.83</td><td>77.68±9.49</td><td>81.62±6.68</td><td>75.79±8.70</td><td>82.25±9.16</td><td>79.55±7.25</td><td>82.63±4.09</td><td>77.38±10.31</td><td>79.13±6.82</td></tr><tr><td>Ionosphere</td><td>87.97±5.20</td><td>88.19±3.77</td><td>89.73±5.32</td><td>91.06±3.39</td><td>88.08±3.87</td><td>91.01±5.27</td><td>89.37±4.07</td><td>90.52±3.85</td><td>92.50±3.66</td><td>93.53±5.87</td><td>88.58±5.10</td><td>88.67±6.19</td></tr><tr><td>ISPY</td><td>95.91±2.30</td><td>98.79±1.40</td><td>94.42±3.89</td><td>94.02±2.44</td><td>92.85±4.33</td><td>91.03±11.58</td><td>89.56±8.02</td><td>98.79±1.40</td><td>97.09±2.23</td><td>93.46±5.68</td><td>98.79±1.40</td><td>87.07±5.50</td></tr><tr><td>Sonar</td><td>71.49±1.82</td><td>72.05±5.51</td><td>68.61±4.97</td><td>77.18±4.74</td><td>73.90±4.10</td><td>75.77±7.04</td><td>70.55±5.92</td><td>77.63±6.03</td><td>76.89±6.89</td><td>90.85±12.09</td><td>75.33±3.32</td><td>75.37±6.34</td></tr><tr><td>Sports</td><td>76.19±3.65</td><td>78.07±1.77</td><td>75.42±3.39</td><td>75.27±1.80</td><td>74.79±3.50</td><td>77.58±3.12</td><td>74.87±2.61</td><td>78.53±3.78</td><td>78.73±4.23</td><td>86.84±7.56</td><td>77.23±1.56</td><td>72.11±3.99</td></tr><tr><td>Turkish</td><td>66.65±1.90</td><td>65.42±4.12</td><td>66.94±6.13</td><td>66.75±3.81</td><td>66.14±4.06</td><td>67.20±3.34</td><td>63.61±5.73</td><td>78.19±2.30</td><td>69.10±1.14</td><td>85.92±14.38</td><td>74.28±4.40</td><td>66.90±7.31</td></tr><tr><td>Vehicle</td><td>73.23±3.20</td><td>67.23±0.86</td><td>68.99±1.77</td><td>72.51±1.58</td><td>70.68±4.62</td><td>73.77±1.53</td><td>66.68±2.45</td><td>72.34±2.42</td><td>73.54±2.80</td><td>77.22±8.35</td><td>73.18±1.90</td><td>68.57±2.84</td></tr><tr><td>WDBC</td><td>94.93±1.64</td><td>94.10±2.01</td><td>93.30±1.79</td><td>95.13±1.42</td><td>92.73±1.08</td><td>94.39±1.76</td><td>93.31±2.72</td><td>94.77±3.10</td><td>94.53±2.44</td><td>98.32±1.12</td><td>93.67±3.19</td><td>94.78±1.05</td></tr><tr><td>Wine</td><td>93.28±4.98</td><td>95.08±3.76</td><td>93.62±5.32</td><td>95.63±3.10</td><td>93.48±3.00</td><td>94.38±5.90</td><td>90.46±3.99</td><td>92.71±3.83</td><td>95.87±3.85</td><td>98.92±2.17</td><td>93.35±2.97</td><td>88.67±4.58</td></tr><tr><td>WPBC</td><td>55.41±7.19</td><td>63.27±12.49</td><td>58.14±6.87</td><td>55.27±9.03</td><td>55.01±9.58</td><td>60.38±9.13</td><td>57.02±8.81</td><td>60.67±6.64</td><td>56.06±6.16</td><td>76.62±17.67</td><td>62.51±6.72</td><td>54.69±6.52</td></tr><tr><td>Zoo</td><td>80.38±13.36</td><td>81.28±11.22</td><td>88.86±12.64</td><td>83.82±15.59</td><td>82.28±10.43</td><td>79.78±15.17</td><td>76.57±3.18</td><td>71.99±8.23</td><td>80.71±13.46</td><td>92.86±14.29</td><td>82.01±7.35</td><td>83.23±20.70</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Our BHF  CatBoost Dempster Deng Dubois EC-FMDS  HEF  LightGBM  Murphy  XGBoost Yager

![](images/d3599194eb5105ff832ed0a2105cdd7d61403853fce6bbda4d013cd782b78c51.jpg)  
Figure B.1: F1-score under Gaussian feature noise

![](images/a81fb48b360db5883d94aa1a323b1ffd779af0dd86df88470b80c2151a84a302.jpg)  
Figure B.2: F1-score under mean feature noise

![](images/3f113ae03347b02b992e3a2670363c53adc6894be91deacb684b940a4d925c40.jpg)  
Figure B.3: F1-score under random label noise

## References

Alzubaidi, L., Zhang, J., Humaidi, A. J., Al-Dujaili, A., Duan, Y., Al-Shamma, O., Santamar´ıa, J., Fadhel, M. A., Al-Amidie, M., & Farhan, L. (2021). Review of deep learning: Concepts, cnn architectures, challenges, applications, future directions. Journal ofBig Data, 8(1). https://doi.org/10.1186/s40537-021-00444-8

Bleichrodt, H., Cillo, A., & Diecidue, E. (2010). A quantitative measurement of regret theory. Management Science, 56(1), 161–175. https://doi.org/10.1287/mnsc.1090.1097

Cai, T. T., & Ma, R. (2022). Theoretical foundations of t-SNE for visualizing high-dimensional clustered data. Journal ofMachine Learning Research, 23(301), 1–54.

Calderwood, S., McAreavey, K., Liu, W., & Hong, J. (2016). Context-dependent combination of sensor information in dempster–shafer theory for bdi. Knowledge and Information Systems, 51(1), 259–285. https://doi.org/10.1007/s10115- 016-0978-0

Chamlal, H., Rebbah, F. E., & Ouaderhman, T. (2025). An ensemble classifier combining dempster–shafer theory and feature selection methods aggregation strategy. Applied Soft Computing, 180, 113306. https://doi.org/10.1016/j.asoc.2025.113306

Chen, L., Zhang, Z., Yang, G., Zhou, Q., Xia, Y., & Jiang, C. (2023). Evidence-theory-based reliability analysis from the perspective of focal element classification using deep learning approach. Journal ofMechanical Design, 145(7). https: //doi.org/10.1115/1.4062271

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings ofthe 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. https://doi.org/10.1145/2939672.2939785

Dempster, A. P. (1967). Upper and lower probabilities induced by a multivalued mapping. Annals ofMathematical Statistics, 38(2), 57–72. https://doi.org/10.1214/aoms/1177698950

Deng, Y., Shi, W., Zhu, Z., & Liu, Q. (2004). Combining belief functions based on distance of evidence. Decision Support Systems, 38(3), 489–493. https://doi.org/10.1016/j.dss.2004.04.015

Destercke, S., & Burger, T. (2013). Toward an axiomatic definition of conflict between belief functions. IEEE Transactions on Cybernetics, 43(2), 585–596. https://doi.org/10.1109/TSMCB.2012.2212703

Dhanka, S., Sharma, A., Kumar, A., Maini, S., & Vundavilli, H. (2026). Advancements in hybrid machine learning models for biomedical disease classification using integration of hyperparameter-tuning and feature selection methodologies: A comprehensive review. Archives ofComputational Methods in Engineering, 33(1), 289–324. https://doi.org/10.1007/ s11831-025-10309-5

Dubois, D., & Prade, H. (1988). Representation and combination of uncertainty with belief functions and possibility measures. Computational Intelligence, 4(3), 244–264. https://doi.org/10.1111/j.1467-8640.1988.tb00279.x

El-Din, D. M., Hassanein, A. E., & Hassanien, E. E. (2024). An adaptive and late multifusion framework in contextual representation based on evidential deep learning and dempster–shafer theory. Knowledge and Information Systems, 66(11), 6881–6932. https://doi.org/10.1007/s10115-024-02150-2

Gao, X., & Pan, L. (2025). An information fusion model of mutual influence between focal elements: A perspective on interference efects in dempster–shafer evidence theory. Information Fusion, 124, 103286. https://doi.org/10.1016/j.infus.2025.103286

Ghorbanzadeh, O., Meena, S. R., Shahabi Sorman Abadi, H., Tavakkoli Piralilou, S., Lv, Z., & Blaschke, T. (2021). Landslide mapping using two main deep-learning convolution neural network streams combined by the dempster–shafer model. IEEE Journal ofSelected Topics in Applied Earth Observations and Remote Sensing, 14, 452–463. https://doi.org/10. 1109/jstars.2020.3043836

Guo, B., Shen, J., Zhou, G., Gao, D., & Wu, Z. (2026). A multi-damage fusion diagnostic method based on dempster-shafer evidence theory for guided vision-adaptive detection using guided waves. Measurement, 287, 122435. https://doi.org/10. 1016/j.measurement.2026.122435

Hasan, M., Abedin, M. Z., Hajek, P., Coussement, K., Sultan, M. N., & Lucey, B. (2024). A blending ensemble learning model for crude oil price forecasting. Annals ofOperations Research, 353(2), 485–515. https://doi.org/10.1007/s10479-023-05810-8

Huang, L., Ruan, S., Decazes, P., & Denœux, T. (2025). Deep evidential fusion with uncertainty quantification and reliability learning for multimodal medical image segmentation. Information Fusion, 113, 102648. https://doi.org/10.1016/j.infus. 2024.102648

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly eficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, 3149–3157.

Li, Q., Wan, X., Guo, H., Wang, W., Liu, M., & Cui, Y. (2026). Uncertainty-aware multi-modal time series anomaly detection via reduced-order evidence fusion. Pattern Recognition, 180, 114212. https://doi.org/10.1016/j.patcog.2026.114212

Liu, Q., Lee, W.-S., Huang, M., & Wu, Q. (2023). Synergy between stock prices and investor sentiment in social media. Borsa Istanbul Review, 23(1), 76–92. https://doi.org/10.1016/j.bir.2022.09.006

Loomes, G., & Sugden, R. (1982). Regret theory: An alternative theory of rational choice under uncertainty. The Economic Journal, 92(368), 805–824. https://doi.org/10.2307/2232669

Lu, Y., & Zhu, W. (2026). Soft likelihood-based multi-source fusion with reliability modeling and optimization under dempster–shafer theory. Engineering Applications ofArtificial Intelligence, 163, 112993. https://doi.org/10.1016/j. engappai.2025.112993

Murphy, C. K. (2000). Combining belief functions when evidence conflicts. Decision Support Systems, 29(1), 1–9. https : //doi.org/10.1016/S0167-9236(99)00084-6

Ng, A., Jordan, M., & Weiss, Y. (2001). On spectral clustering: Analysis and an algorithm. Advances in Neural Information Processing Systems, 849–856.

Park, J. (2025). Estimation of vessel collision risk under uncertainty using interval type-2 fuzzy inference systems and dempster–shafer evidence theory. Journal of Marine Science and Engineering, 14(1), 34. https://doi.org/10.3390/ jmse14010034

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. Advances in Neural Information Processing Systems, 31, 6639–6649.

Qiao, S., Fan, Y., Wang, G., & Zhang, H. (2023). Multi-sensor data fusion method based on improved evidence theory. Journal of Marine Science and Engineering, 11(6), 1142. https://doi.org/10.3390/jmse11061142

Qiao, S., Song, B., Fan, Y., & Wang, G. (2023). A fuzzy dempster–shafer evidence theory method with belief divergence for unmanned surface vehicle multi-sensor data fusion. Journal of Marine Science and Engineering, 11(8), 1596. https://doi.org/10.3390/jmse11081596

Qiu, Z., Qin, Y., Chen, Z., Zeng, L., & Cai, R. (2025). Overcoming negative weighting in uncertainty-based methods: A multi-uncertainty clustering method for evidence fusion. Complex & Intelligent Systems, 11(9). https://doi.org/10.1007/ s40747-025-01999-2

Radzvilas, M., Peden, W., Tortoli, D., & De Pretis, F. (2024). A comparison of imprecise bayesianism and dempster–shafer theory for automated decisions under ambiguity. Journal ofLogic and Computation, 35(8). https://doi.org/10.1093/logcom/ exae069

Shafer, G. (1976, April). A mathematical theory ofevidence. Princeton University Press. https://doi.org/10.1515/9780691214696

Shi, J., & Malik, J. (2000). Normalized cuts and image segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(8), 888–905. https://doi.org/10.1109/34.868688

Strelet, E., Castillo, I., Peng, Y., & Reis, M. S. (2025). Data fusion: Integrating heterogeneous information sources in the chemical processing industry. Journal ofChemometrics, 39(11). https://doi.org/10.1002/cem.70075

Urbani, M., Gasparini, G., & Brunelli, M. (2023). A numerical comparative study of uncertainty measures in the dempster–shafer evidence theory. Information Sciences, 632, 119027. https://doi.org/10.1016/j.ins.2023.119027

van Engelen, J. E., & Hoos, H. H. (2019). A survey on semi-supervised learning. Machine Learning, 109(2), 373–440. https: //doi.org/10.1007/s10994-019-05855-6

Wang, H.-Y., Wang, J.-S., & Wang, G. (2021). Clustering validity function fusion method of fcm clustering algorithm based on dempster–shafer evidence theory. International Journal ofFuzzy Systems, 24(1), 650–675. https://doi.org/10.1007/s40815- 021-01170-2

Xiao, Y., Ma, X., & Zhan, J. (2026). Group decision-making in heterogeneous multi-scale information fusion: Integrating overconfident and non-cooperative behaviors. Information Fusion, 125, 103401. https://doi.org/10.1016/j.infus.2025. 103401

Xu, X., Zhang, Z., Xu, D., & Chen, Y. (2016). Interval-valued evidence updating with reliability and sensitivity analysis for fault diagnosis. International Journal ofComputational Intelligence Systems, 9(3), 396. https://doi.org/10.1080/18756891. 2016.1175808

Yager, R. R. (1987). On the dempster-shafer framework and new combination rules. Information Sciences, 41(2), 93–137. https://doi.org/10.1016/0020-0255(87)90007-7

Yu, Y., Karimi, H. R., Gelman, L., Tian, J., & Mei, P. (2026). A novel multi-source sensor correlation adaptive fusion framework with uncertainty quantification for intelligent fault diagnosis. Reliability Engineering & System Safety, 267, 111812. https://doi.org/10.1016/j.ress.2025.111812

Zadeh, L. A. (1986). A simple view of the dempster-shafer theory of evidence and its implication for the rule of combination. AI Magazine, 7(2), 85–91. https://doi.org/10.1609/aimag.v7i2.542

Zhang, Q., Zhang, P., & Li, T. (2025). Information fusion for large-scale multi-source data based on the dempster-shafer evidence theory. Information Fusion, 115, 102754. https://doi.org/10.1016/j.infus.2024.102754

Zhao, Z., Wang, R., Pang, W., & Li, Z. (2025). Feature selection for label distribution learning using dempster-shafer evidence theory. Applied Intelligence, 55(4). https://doi.org/10.1007/s10489-024-05879-z

Zhou, J., Hong, X., & Jin, P. (2019). Information fusion for multi-source material data: Progress and challenges. Applied Sciences, 9(17), 3473. https://doi.org/10.3390/app9173473

Zhou, Z., & Xiao, F. (2026). Conflict management in sequential evidence combination. Information Sciences, 734, 122958. https://doi.org/10.1016/j.ins.2025.122958

Zhu, C., & Xiao, F. (2021). A belief hellinger distance for D–S evidence theory and its application in pattern recognition. Engineering Applications ofArtificial Intelligence, 106, 104452. https://doi.org/10.1016/j.engappai.2021.104452