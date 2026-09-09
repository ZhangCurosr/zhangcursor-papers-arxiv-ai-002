# Time-Varying Data as Sheaves: an Invitation to Narratives

Wilmer Leal<sup>1\*</sup>, Benjamin Merlin Bumpus<sup>2</sup>, Jana K. Nickel<sup>3</sup>, Johan Garc´ıa<sup>4</sup>, James Fairbanks<sup>1</sup>, Warren Dixon<sup>1,5</sup>

<sup>1\*</sup>Department of Mechanical and Aerospace Engineering, University of Florida, 1064 Center Drive, Gainesville, 32611-6250, Florida, USA .

<sup>2</sup>Instituto de Matematica e Estat´ ´ıstica, Universidade de Sao Paulo, Rua do˜ Matao, 1010, S˜ ao Paulo, 05508–090, SP, Brasil.˜

<sup>3</sup>Fachbereich Mathematik, Universitat Hamburg, Bundesstraße 55, Hamburg,¨ 20146, Germany.

<sup>4</sup>Departamento de Matematicas, Universidad Nacional de Colombia – sede´ Medell´ın, Calle 59A No. 63-20, Medell´ın, Colombia.

<sup>5</sup>Virginia Tech, College of Engineering, Blacksburg, Virginia, 24061, USA .

\*Corresponding author(s). E-mail(s): wleal@ufl.edu; Contributing authors: bumpus@usp.br; jana@nickel-math.com; jfgarciava@unal.com; fairbanksj@ufl.edu; wdixon@ ufl.edu,vt.edu ;

## Abstract

Modern science and engineering increasingly rely on time-varying data, yet the mathematical tools used to model temporal phenomena are often developed within separate disciplines, obscuring common principles and limiting the transfer of ideas across fields. This chapter presents the theory of narratives, an abstract framework for time-varying objects of any mathematical kind that supports both theoretical investigations and applications. To illustrate this perspective, the chapter develops three vignettes, each illustrating a different research direction. The first addresses a general concern: What information loss can occur when switching between different representations of temporal data? The second concerns structural and algorithmic approaches: How can we systematically decompose time-varying data into simple pieces and obtain invariants describing its structural complexity? The third is an application to control theory: How can we model multi-agent systems with switching communication topologies? More important than any individual vignette, the central message of this invitation is that a suitable abstract perspective can organize and guide research across remarkably diverse mathematical and scientific domains.

Keywords: Time-Varying Data, Temporal Networks, Categories of Narratives, Temporal Data Structures, Persistent and Cumulative Data, Structured Decompositions, Cellular Sheaves, Multi-Agent Systems, Control Theory

MSC Classification: 18A40 , 18F20 , 18A25 , 05C90 , 37B55 , 93C85

## Contents

Introduction 3   
2 Background on Narratives: How to Model Time-Varying Data using Sheaves 4   
2.1 Categories of temporal data . 4   
2.2 Changing Perspectives on Temporal Data: The Persistence–   
Accumulation Adjunction . 6   
2.3 Systematic Temporalization 7   
3 Three Vignettes on Time-Varying Data and the Ensuing Research Directions 9   
3.1 Vignette 1: Narratives and the Fixed Points of the Persistence–Accumulation   
Adjunction 9   
3.1.1 Narratives encoding persistence and accumulation simultaneously 11   
3.1.2 Factorizing the persistence–accumulation adjunction 13   
3.1.3 Classifying narratives by rigidity 19   
3.2 Vignette 2: Decomposing Time-Varying Data into Simple Pieces: Structured   
Decompositions of Narratives 22   
3.2.1 Structured decompositions 22   
3.2.2 Temporalization of spined sd-categories 24   
3.2.3 Temporalizing tree-width: examples 29   
3.3 Vignette 3: Temporal Cellular Sheaves: Modelling Multi-agent Systems with   
Switching Topologies . 31   
3.3.1 Cellular sheaves for multi-agent systems . 31   
3.3.2 Category of cellular sheaves 33   
3.3.3 Temporal cellular sheaves and switching topologies 35   
Conclusion and future directions 36

## 1 Introduction

Modern science and engineering are increasingly driven by the analysis of time-varying data. From social dynamics [1], environmental science [2], and public health [3, 4] to distributed control of multi-agent systems [5, 6], language evolution [7], and the computational history of science [8, 9], observations rarely describe static objects; rather, they capture the evolution of phenomena through time [10]. Despite the ubiquity of such data, the mathematical tools used to analyze them remain largely fragmented [11], with different communities developing specialized theories for temporal data mining, temporal databases, temporal networks, and related domains [11–14]. Many existing approaches represent temporal information as sequences of measurements or snapshots of evolving systems, often without a coherent mathematical structure describing how these observations relate across time. As a result, integrating different representations of temporal data, or reasoning systematically about their transformations, remains a significant challenge. The theory of narratives introduced in [15] addresses this problem by representing time-varying data as sheaves and cosheaves over categories of time intervals. A persistent narrative records information that remains valid throughout an interval, while a cumulative narrative records information that accumulates over it. Because narratives are categorical objects, the framework describes not only timevarying data but also maps between them, and it applies to data valued in categories of sets, graphs, topological spaces, cellular sheaves, and many other structured objects. This chapter is intended as an invitation to the theory of narratives. Through a survey of recent developments, we hope to illustrate how narratives provide a common framework in which ideas from many areas of mathematics can contribute to the study of time-varying data, through both abstract theory and concrete applications.

The theory of narratives was designed to address five requirements that arise when one seeks a general mathematical language for time-varying data:

(D1) (Categories of Temporal Data) Any theory of temporal data should define not only time-varying data, but also appropriate morphisms thereof.

(D2) (Cumulative and Persistent Perspectives) In contrast to being a mere sequence, temporal data should explicitly record whether it is to be viewed cumulatively or persistently. Furthermore, there should be methods of conversion between these two viewpoints.

(D3) (Systematic “Temporalization”) Any theory of temporal data should come equipped with systematic ways of obtaining temporal analogues of notions relating to static data.

(D4) (Object Agnosticism) Theories of temporal data should be object agnostic and applicable to any kinds of data originating from given underlying dynamics.

(D5) (Sampling) Since temporal data arises from an underlying dynamical system, any theory of temporal data should be interoperable with theories of dynamical systems.

This chapter collects three recent developments in the theory of narratives. Each is presented as a vignette centered on a different question: (1) how much information is preserved when one changes between persistent and cumulative viewpoints, (2) how the structural complexity of time-varying data can be measured, and (3) how narratives can be used to model multi-agent systems whose interaction structure changes through time. Together, these examples illustrate the broader goals of our research program: by studying time-varying data systematically and abstractly, we can use a unified language to describe many different kinds of questions concerning time-varying data.

The first vignette concerns the adjunction between persistent and cumulative narratives. Since this adjunction is not, in general, an equivalence, converting a narrative from one viewpoint to the other and back may lose information. We introduce narratives valued in a cotwisted arrow category, which encode a persistent component, a cumulative component, and a comparison between them. We then use this category to factorize the persistence– accumulation adjunction and to classify narratives according to whether they can be recovered after either of the two possible round trips.

The second vignette concerns decompositions of time-varying data. Complex objects are often studied by expressing them as composites of smaller pieces and recording how these pieces overlap. Such decompositions also give rise to measures of structural complexity. In the temporal setting, the pieces must themselves vary through time: decomposing each snapshot independently does not record how the pieces persist, merge, split, or disappear. Building on recent work [16], we explain how theories of structured decompositions for static objects can be lifted to persistent narratives. The resulting framework yields decompositions whose pieces are time-varying objects and yields temporal analogues of structural invariants such as tree-width.

The third vignette applies narratives to multi-agent systems whose communication topology changes through time. By taking cellular sheaves as the category of values, a narrative can record both the changing interaction graph and the local state spaces, sensing maps, and compatibility constraints carried by its vertices and edges. The resulting model of a switching system contains substantially more information than a sequence of communication graphs. It also provides a setting in which the relationship between an underlying dynamical system and the temporal data sampled from it can be studied using the same categorical language.

## 2 Background on Narratives: How to Model Time-Varying Data using Sheaves

## 2.1 Categories of temporal data

We begin by recalling background on narratives: objects introduced in [15] representing time-varying objects valued in any sufficiently nice category. Narratives are (co)sheaves over categories of time intervals—we call these time categories—which model temporal data by recording how temporal information relates across overlapping intervals of time.

The thesis of [15] is that temporal data should be understood not merely as a sequence of observations, but as a structured system of relations connecting observations across time. For example, a time-varying graph may consist of a family of graphs $G _ { t }$ indexed by time together with morphisms describing how vertices and edges persist, merge, or disappear as time evolves. Narratives capture precisely these additional relations across time. Rather than representing time merely as a totally ordered set, the framework uses categories of intervals, which naturally encode the inclusion relations between time periods.

Definition 1 (Time Categories) A time category is any sub-join-semilattice of I or $\mathsf { I } _ { \mathbb { N } } ,$ , where I (resp.   
I<sub>N</sub>) is the category of closed intervals in R (resp. N) with inclusions as morphisms.

Remark 1 (Alternative models of time) Although we focus on interval categories in this chapter, the categorical viewpoint naturally accommodates more general notions of time. For instance, one could replace intervals by a finite branching tree to model branching temporal evolutions, such as the history of a Git repository or the multiple futures envisioned in Borges’ The Garden of Forking Paths [17]. Developing such generalized notions of time is an interesting direction for future research.

Example 1 (Finite time categories) If T is a time category and $[ a , b ] \in \mathsf { T }$ , then the slice category ${ \mathsf { T } } / [ a , b ]$ consisting of the intervals contained in $[ a , b ]$ , is itself a time category, and the canonical inclusion $\mathsf { T } / [ a , b ] \hookrightarrow \mathsf { T }$ restricts the temporal domain to the interval $[ a , b ]$ . In particular, for every $n \geq 0$ , we write ${ \sf T } _ { n } : = { \sf I } _ { \mathbb { N } } / [ 0 , n ]$ . Its objects are the intervals $[ i , j ] \subseteq [ 0 , n ]$ , ordered by inclusion, so ${ \sf T } _ { n }$ models temporal data consisting of $n + 1$ snapshots together with every interval determined by them. For example, ${ \sf T } _ { 2 }$ consists of the six intervals contained in [0, 2]:

![](images/e8b9ee4ceaf1f146fc3984c3cc9a45baf643031fb4b0e80a146f9622b66740bc.jpg)

Example 2 (Changing temporal resolution) Suppose that $[ a , b ]$ is an interval in a time category T, and choose a partition $a = r _ { 0 } \leq r _ { 1 } \leq \cdots \leq r _ { n } = b .$ This partition determines a unique functor

$$
R : \mathsf { T } _ { \boldsymbol { n } } \longrightarrow \mathsf { T } / [ a , b ] , \qquad R ( [ i , j ] ) = [ r _ { i } , r _ { j } ] .
$$

The finite time category ${ \sf T } _ { n }$ models the temporal resolution of $[ a , b ]$ determined by the chosen snapshots $r _ { 0 } , \ldots , r _ { n } \colon$ the objects $[ i , i ]$ correspond to the snapshots, while the objects $[ i , j ]$ correspond to the intervals from r<sub>i</sub> to $r _ { j }$

To speak of sheaves, one must first have a site, that is, a category equipped with a coverage, where, intuitively, a coverage amounts to a systematic way of defining what it means for an object to be “covered” by opens. As shown in [15], time categories become sites when equipped with the Johnstone coverage [18]. Under this coverage, a cover of an interval $[ \ell , \ell ^ { \prime } ]$ is generated by a partition into two closed intervals $\left( [ \ell , p ] , [ p , \ell ^ { \prime } ] \right)$ . This reflects the idea that information about a time interval can be reconstructed from information on adjacent subintervals.

## Narratives

As we alluded to earlier, narratives—our model for temporal data—consist of sheaves or cosheaves over time categories (viewed as sites, as above). The choice between sheaves and cosheaves is a modelling decision: should the model track data that persists over time (sheaves) or data that accumulates over time (cosheaves)?

Definition 2 (T-sheaves and T-cosheaves) Let T be a time category equipped with the Johnstone coverage. If D has pullbacks, a D-valued sheaf on T is a presheaf $F : \mathsf { T } ^ { o p } \to \mathsf { D }$ such that

$$
F ( [ a , b ] ) \cong F ( [ a , p ] ) \times _ { F ( [ p , p ] ) } F ( [ p , b ] ) .
$$

Dually, if D has pushouts, a D-valued cosheaf on T is a copresheaf $\hat { F } : \mathsf { T }  \mathsf { D }$ such that for any interval $[ a , b ]$ and cover $( [ a , p ] , [ p , b ] )$ ),

$$
\hat { F } ( [ a , b ] ) \cong \hat { F } ( [ a , p ] ) + _ { \hat { F } ( [ p , p ] ) } \hat { F } ( [ p , b ] ) .
$$

Definition 3 (Narratives) Let T be a time category and D a category with pullbacks (resp. pushouts). The category of persistent D-narratives, denoted Pe(T,D) (resp. cumulative D-narratives, denoted $\mathsf { C u } ( \mathsf { T } , \mathsf { D } ) )$ ) consists of D-valued sheaves (resp. cosheaves) on T.

A persistent narrative models information that must remain valid throughout a time interval. The sheaf condition ensures that such information over a larger interval can be reconstructed from data that persist across overlapping sub-intervals. In contrast, a cumulative narrative models information that accumulates over time, with the cosheaf condition expressing that data associated with a longer interval arises from aggregating the data of its sub-intervals. In both cases, a narrative encodes not only observations made at individual moments in time but also the relations describing how these observations evolve and interact across time intervals.

Remark 2 Since persistent narratives are sheaves, their categories can carry an internal intuitionistic logic. In particular, when the target category is Set, or more generally a presheaf category ([C, Set]), categories of discrete persistent narratives form Grothendieck topoi. In [4], this structure is used to define temporal truth values and a propositional language for reasoning about properties that hold at specific times or throughout intervals. The resulting logic is applied to public health models, where it can express temporal properties of individuals and conditions and support reasoning about possible routes of disease transmission in time-varying contact networks.

## 2.2 Changing Perspectives on Temporal Data: The Persistence– Accumulation Adjunction

Cumulative and persistent narratives are related by the following adjunction [15], which describes how one may change perspective between persistent and cumulative descriptions of temporal data.

Theorem 2.1 (Adjunction between cumulative and persistent narratives [15]) Let D be a category with limits and colimits and T a time category. There exist functors $\mathcal { H } : \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) \to \mathsf { C u } ( \mathsf { T } , \mathsf { D } )$ and $\mathcal { P }$ : $\mathsf { C u } ( \mathsf { T } , \mathsf { D } ) \to \mathsf { P e } ( \mathsf { T } , \mathsf { D } )$ forming an adjunction:

$$
\mathsf { P e } ( \mathsf { T } , \mathsf { D } ) \underbrace { \overbrace { \mathrm {  ~ \Gamma ~ } \mathrm {  ~ \Omega ~ } ^ { \mathcal { A } } \mathrm {  ~ \Gamma ~ } } ^ { \mathcal { A } } } _ { \mathcal { P } } \mathsf { C u } ( \mathsf { T } , \mathsf { D } ) .
$$

The functor $\mathcal { K }$ sends a persistent narrative $F : \mathsf { T } ^ { o p } \to \mathsf { D }$ to its canonical cumulative counterpart $\mathcal { H } F$ . For each interval $[ a , b ] \in \mathsf { T }$ , its value is defined by

$$
( \mathcal { H } F ) _ { a } ^ { b } : = \mathrm { c o l i m } \Big ( ( \mathsf { T } / [ a , b ] ) ^ { o p } \hookrightarrow \mathsf { T } ^ { o p } \overset { F } { \to } \mathsf { D } \Big ) .\tag{1}
$$

Thus, $( \mathcal { H } F ) _ { a } ^ { b }$ accumulates the persistent data assigned by $F$ to the subintervals of $[ a , b ]$ . An inclusion $i : [ a , b ] \hookrightarrow [ c , d ]$ induces a morphism

$$
\mathcal { H } F ( i ) : ( \mathcal { K } F ) _ { a } ^ { b } \longrightarrow ( \mathcal { H } F ) _ { c } ^ { d }
$$

by the universal property of the corresponding colimits. Thus, $\mathcal { H } F ( i )$ records how accumulated data over a smaller time window contribute to the accumulated data over a larger one. Dually, the functor $\mathcal { P }$ sends a cumulative narrative $\hat { F } : \mathsf { T }  \mathsf { D }$ to its canonical persistent counterpart. For each interval $[ a , b ] \in \mathsf { T }$ , its value is defined by

$$
( \mathcal { P } \hat { F } ) _ { a } ^ { b } : = \operatorname* { l i m } \Bigl ( \mathsf { T } / [ a , b ] \hookrightarrow \mathsf { T } \xrightarrow { \hat { F } } \mathsf { D } \Bigr ) .\tag{2}
$$

Thus, $( \mathcal { P } \hat { F } ) _ { a } ^ { b }$ extracts the data compatible across the subintervals of $[ a , b ]$ . An inclusion $i :$ $[ a , b ] \hookrightarrow [ c , d ]$ induces a morphism

$$
\mathcal { P } \hat { F } ( i ) : ( \mathcal { P } \hat { F } ) _ { c } ^ { d } \longrightarrow ( \mathcal { P } \hat { F } ) _ { a } ^ { b }
$$

by the universal property of the corresponding limits. The adjunction is equipped with a unit and counit

$$
\begin{array} { r } { \eta : \mathsf { i d } _ { \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) } \Longrightarrow \mathscr { P M } \qquad \mathrm { a n d } \qquad \mathsf { e } : \mathscr { M P } \Longrightarrow \mathsf { i d } _ { \mathsf { C u } ( \mathsf { T } , \mathsf { D } ) } . } \end{array}
$$

For a persistent narrative $F _ { \ast }$ , the component $\Pi _ { F } : F \longrightarrow { \mathcal { P } } { \mathcal { H } } F$ compares the prescribed persistent data with the persistent narrative reconstructed after passing through its cumulative completion. Dually, for a cumulative narrative $\hat { F } _ { : }$ , the component $\mathfrak { E } _ { \hat { F } } : \mathcal { H P } \longrightarrow \hat { F }$ compares the cumulative narrative reconstructed from its persistent completion with the original cumulative data.

The functors $\mathcal { K }$ and $\mathcal { P }$ do not, in general, form an equivalence of categories. Consequently, passing from one perspective to the other and back need not recover the original temporal structure. Instead, the adjunction replaces the original data by the canonical objects determined by the corresponding colimit and limit constructions. The unit and counit therefore measure the extent to which the persistent and cumulative descriptions agree with these canonical reconstructions. A natural problem is therefore to characterize those temporal structures for which no information is lost. Equivalently, one may ask when the unit or counit of the adjunction is an isomorphism, or, more generally, which narratives are fixed points of the persistence–accumulation adjunction. The first research direction developed in Section 3.1 is devoted to these questions.

## 2.3 Systematic Temporalization

We now explain how [15] addresses Desideratum (D3), namely the requirement that a theory of temporal data provide systematic procedures for producing temporal analogues of familiar static notions. The guiding observation is that the static theories we care about (graphs, groups, spaces, etc.) are far more mature than their temporal counterparts, and that many static properties are most naturally expressed categorically: either as membership in a subcategory, or as the existence of morphisms from a class of “test objects” (paths, cliques, colorings, and so on). Once phrased categorically, these notions admit canonical lifts to narrative categories.

## Changing data types functorially.

A first ingredient is that narratives are functorial in the data category. If C and D are categories of static data and $K : { \mathsf { C } } \to { \mathsf { D } }$ is a data-conversion functor, then postcomposition transports C-valued narratives to D-valued narratives whenever K preserves the universal constructions that define the (co)sheaf conditions. Concretely, if K is continuous, then for any time category T the assignment $F \mapsto K \circ F$ defines a functor $( K \circ - ) : \mathsf { P e } ( \mathsf { T } , \mathsf { C } ) \to \mathsf { P e } ( \mathsf { T } , \mathsf { D } )$ , and dually if $K$ preserves colimits then $\left( K \circ - \right)$ sends cumulative C-narratives to cumulative D-narratives. There is also a contravariant form: if $K \colon { \mathsf { C } } ^ { o p } \to { \mathsf { D } }$ takes limits to colimits (resp. colimits to limits), then composition with K switches perspectives, producing a functor from persistent to cumulative narratives (resp. cumulative to persistent). In this way, standard static constructions can be “temporalized” and transported across data types in a principled, functorial manner.

## Lifting static properties by change of base.

The second ingredient is a systematic way to lift classes of static objects to classes of narratives. Suppose a static property is presented by a subcategory inclusion $P : \mathsf { P } \hookrightarrow \mathsf { C }$ (one should think of $\mathsf { P }$ as the full subcategory of objects of C satisfying the property). If P is continuous, then the induced functor $\left( P \circ - \right)$ identifies those persistent C-narratives whose values lie in P in a way compatible with the sheaf condition; dually, if $P$ preserves colimits, it specifies the corresponding class of cumulative narratives. This reproduces, in the temporal setting, the common pattern from static combinatorics and algebra: properties become subcategories, and temporal analogues become subcategories of narrative categories.

## Temporal analogues at a chosen resolution.

While change of base yields a clean lift, it can be too strict in applications: one may only care that a property appears at a particular temporal granularity. The framework in [15] accounts for this by introducing a functorial method for changing temporal resolution. Given a subjoin-semilattice inclusion $\tau : \mathsf { S } \hookrightarrow \mathsf { T }$ , precomposition defines a restriction functor $( - \circ \tau )$ sending T-narratives to S-narratives (persistent-to-persistent and cumulative-to-cumulative). By combining restriction with change of base, one obtains a general notion of a narrative satisfying a static property only on the intervals in S: rather than requiring the property at all intervals, one imposes it after restricting to $\mathsf { S } ,$ and defines the resulting class of narratives by a pullback that expresses compatibility between “restriction to $\mathsf { S } ^ { \ast }$ and “landing in $\mathsf { P } ^ { \ast }$ . This produces a family of temporal analogues parametrized by resolution, reflecting the empirical reality that certain phenomena are only visible after aggregating over sufficiently large time windows.

Example 3 (Temporal resolution) Recall from Example 2 that a partition $a = r _ { 0 } \leq \cdots \leq r _ { n } = b$ determines a functor $R : \mathsf T { \boldsymbol { n } } \longrightarrow \mathsf { T } / [ a , b ]$ . Consequently, every D-valued narrative $F : ( \mathsf T / [ a , b ] ) ^ { \mathrm { o p } } \to \mathsf D$ restricts along R to the narrative

$$
F \circ R ^ { \mathrm { o p } } : \mathsf { T } _ { n } ^ { \mathrm { o p } } \longrightarrow \mathsf { D } .
$$

The resulting narrative is the restriction of $F$ to the temporal resolution induced by the chosen partition.

## Graph-theoretic case studies: paths, cliques, and dualities.

To demonstrate the machinery recovers familiar temporal notions while adding conceptual clarity, [15] develops graph-theoretic case studies. Path-like behavior is obtained by lifting the subcategory of paths in Grph to a corresponding class of graph narratives; temporal paths in a graph narrative can then be expressed as subobjects of that narrative, recovering the static viewpoint in which many decision problems are homomorphism problems. Temporal cliques are treated similarly: by lifting the subcategory of complete graphs and combining it with a resolution restriction $( \mathrm { e . g . }$ , requiring completeness only over intervals of length at least n), one recovers standard definitions of temporal k-cliques from the temporal-graph literature, but now characterized by morphisms of narratives. This categorical reformulation also exposes dualities: just as cliques and colorings are related by categorical duality in the static setting, temporal analogues inherit corresponding dual notions, and these dualities depend on whether one works cumulatively or persistently.

A key conceptual payoff is that these temporalizations are not ad hoc: they arise from a small number of functorial principles (change of base, change of resolution, and categorical characterizations by morphisms). Moreover, the interaction with the persistent–cumulative adjunction highlights genuine temporal subtleties: for instance, properties defined by stability under pushouts may behave differently from those defined by stability under pullbacks, and changing perspective can therefore change which temporal analogues exist or are well behaved.

## 3 Three Vignettes on Time-Varying Data and the Ensuing Research Directions

## 3.1 Vignette 1: Narratives and the Fixed Points of the Persistence–Accumulation Adjunction

The functor $\mathcal { K }$ sends a persistent narrative $F \colon \mathsf { T } ^ { o p } \to \mathsf { D }$ to its cumulative counterpart $\mathcal { H } F$ by forming pushouts that encode data accumulated over a given time window. Applying $\mathcal { P }$ then returns to the persistent viewpoint by computing pullbacks, producing $\mathcal { P H } F$ . In general, this pushout–pullback round trip does not recover the original persistent sheaf. More conceptually, because $\mathcal { K }$ and $\mathcal { P }$ form an adjunction rather than an equivalence, the composite $\mathcal { P H }$ need not act as the identity on persistent narratives. Equivalently, the unit $\boldsymbol \eta _ { F } \colon F \to \mathcal { P } \mathcal { H } F$ may fail to be an isomorphism. Thus, passing between these viewpoints may approximate the original temporal structure by its associated limit and colimit constructions. Figure 1 illustrates this phenomenon by exhibiting a persistent narrative for which the unit $\Pi _ { F } : F  \mathcal { P } \mathcal { H } F$ is not an isomorphism.

The significance of Figure 1 is not merely that the unit fails to be invertible, but why it fails. The span $F _ { 0 } ^ { 0 } \gets F _ { 0 } ^ { 1 } \to F _ { 1 } ^ { 1 }$ defining the persistent narrative is prescribed data and therefore need not satisfy any universal property. By contrast, the span $F _ { 0 } ^ { \bar { 0 } } \left. F _ { 0 } ^ { 0 } \times _ { ( \mathcal { H } F ) _ { 0 } ^ { 1 } } F _ { 1 } ^ { 1 } \right.$ $F _ { 1 } ^ { 1 }$ is built by taking the pullback of the cospan $F _ { 0 } ^ { 0 } \to ( \mathcal { H } F ) _ { 0 } ^ { 1 }  F _ { 1 } ^ { 1 }$ . The unit

$$
\begin{array} { r } { \eta _ { F _ { 0 } ^ { 1 } } : F _ { 0 } ^ { 1 } \longrightarrow ( \mathcal { P } \mathcal { H } F ) _ { 0 } ^ { 1 } } \end{array}
$$

![](images/3c87b7b5ae2439fcf29f50454cfa48afa281acbddba0f3ed2d81f687e109449b.jpg)  
(a)

![](images/87de2c3f5f045474bbde7b2343998c5486f2b3027eb1abef6ed4d301c6bbcdd4.jpg)  
(b)

![](images/3e0a06d720a166e6c0c20ec6ec4f83a8996daa0bf7af89e5d9f951b443e2ec63.jpg)  
(c)  
Fig. 1: An example showing that the adjunction $\mathcal { H } \mathcal { P }$ is not an equivalence. (a) Let T be the time category $[ \stackrel { \cdot } { 0 } , 0 ] \hookrightarrow [ 0 , \stackrel { \cdot } { 1 } ] \longleftrightarrow [ 1 , 1 ]$ . Consider the persistent narrative $F \colon \mathsf { T } ^ { o p } $ Set shown in the figure. Its value at the interval $[ 0 , 1 ] \mathrm { i s } F _ { 0 } ^ { 1 } = \{ 0 , 1 \}$ . (b) Applying K yields a cumulative narrative with $( \mathcal { H } F ) _ { 0 } ^ { 1 } = F _ { 0 } ^ { 0 } + _ { F _ { 0 } ^ { 1 } } F _ { 1 } ^ { 1 } = \{ a \} + _ { \{ 0 , 1 \} } \{ b \}$ , which in this case is the singleton $\{ * \}$ (c) Applying $\mathcal { P }$ produces a persistent narrative with $( \mathcal { P H } F ) _ { 0 } ^ { 1 } = F _ { 0 } ^ { 0 } \times _ { ( \mathcal { H } F ) _ { 0 } ^ { 1 } } F _ { 1 } ^ { 1 } \cong \{ ( a , b ) \}$ Since $\{ ( a , b ) \} \not \cong \{ 0 , 1 \}$ , the component $\eta _ { F _ { 0 } ^ { 1 } } : F _ { 0 } ^ { 1 }  ( \mathcal { P } \mathcal { H } F ) _ { 0 } ^ { 1 }$ is not an isomorphism. Hence $\mathcal { P H } F \not \cong F$

is the canonical comparison between the prescribed persistent data and its canonical approximation via the pushout–pullback round trip.

This observation leads to a natural characterization of the fixed points of the adjunction in the direction $F \to \mathcal { P K } F$ . By a fixed point we mean a persistent narrative F for which the unit $\mathfrak { N } _ { F } : F \to \mathcal { P } \mathcal { K } F$ is an isomorphism. Equivalently, as illustrated by the diagram below, this requires the span $F _ { 0 } ^ { 0 } \left. F _ { 0 } ^ { 1 } \right. F _ { 1 } ^ { 1 }$ to coincide with the pullback of its pushout $F _ { 0 } ^ { 0 } \right. ( \mathcal { H } F ) _ { 0 } ^ { 1 } \left. F _ { 1 } ^ { 1 }$

![](images/5c60a94e65efc918b9c8027abd54230b69bd3ba71bcb3a7094f4f9fc7c9cc3f3.jpg)

This characterization immediately yields a broad class of fixed points whenever the ambient category D is adhesive. Recall that in an adhesive category, pushouts along monomorphisms are Van Kampen [19], and hence are stable under pullback. Consequently, if the span $F _ { 0 } ^ { 0 } \gets F _ { 0 } ^ { 1 } \to F _ { 1 } ^ { 1 }$ consists of monomorphisms, then the associated pushout square is also a pullback. Therefore, F is a fixed point of the adjunction in the direction $F \to { \mathcal { P K } } F$

The discussion above concerns the unit of the adjunction and the recovery of persistent data. Dually, one may study the counit and the recovery of cumulative data. Figure 1 already shows that these two behaviors need not agree: panel (b) is recovered after the pullback– pushout round trip, whereas panel (a) is not recovered after the pushout–pullback round trip. This asymmetry motivates the introduction of narratives, which encode persistent and cumulative descriptions simultaneously together with the comparison between them.

## 3.1.1 Narratives encoding persistence and accumulation simultaneously

Definition 4 (Cotwisted Arrow Category) Let D be a small category. The cotwisted arrow category of D, denoted $\mathsf { D } _ { \triangleleft } ^ { \triangleright } ,$ is defined as follows.

• Objects are morphisms $f \colon x \to y$ in D.

• Morphisms from $( x \stackrel { f } { \to } y ) \mathrm { t o } ( x ^ { \prime } \stackrel { f ^ { \prime } } { \to } y ^ { \prime } )$ are pairs of morphisms $( u \colon x \to x ^ { \prime } , \nu \colon y ^ { \prime } \to y )$ in D such that the following diagram commutes:

$$
\begin{array} { r } { {  \begin{array} { l l l l } { x } & { \displaystyle - \frac { u } { \longrightarrow } } & { x ^ { \prime } } & { } \\ { } & { } & { } \\ { f } & { } & { \displaystyle \{ f ^ { \prime } } & { \mathrm { t h a t ~ i s , } f = \nu \circ f ^ { \prime } \circ u . } \\ { } & { } & { } \\ { y \gets } & { } & { \displaystyle y ^ { \prime } } \end{array}  } } \end{array}
$$

Thus, a morphism in $\mathsf { D } _ { \triangleleft } ^ { \triangleright }$ encodes a factorization of f through $f ^ { \prime } .$ . The notation $\mathsf { D } _ { < } ^ { \triangleright }$ is mnemonic for the defining commutative square: the superscript ▷ reminds the reader that the domain morphism points to the right, whereas the subscript ◁ indicates that the codomain morphism points to the left.

• Composition is defined componentwise:

$$
( u ^ { \prime } , \nu ^ { \prime } ) \circ ( u , \nu ) = ( u ^ { \prime } \circ u , \nu \circ \nu ^ { \prime } ) .
$$

Lemma 3.1 If D has pullbacks and pushouts, then the cotwisted arrow category $\mathsf { D } _ { \triangleleft } ^ { \triangleright }$ has pullbacks.

Proof Consider two morphisms in $\mathsf { D } _ { \triangleleft } ^ { \triangleright }$ with common codomain,

$$
( u _ { i } , \nu _ { i } ) : ( x _ { i } \stackrel { f _ { i } } { \longrightarrow } y _ { i } ) \longrightarrow ( x _ { 0 } \stackrel { f _ { 0 } } { \longrightarrow } y _ { 0 } ) , \qquad i = 1 , 2 ,
$$

satisfying $f _ { i } = \nu _ { i } \circ f _ { 0 } \circ u _ { i }$ . They are depicted in the following diagram.

![](images/f054be964c5301afedeeb8c4d1a4f613060ee3aa253385c4bcba4bc054eeaa55.jpg)

Form in D the pullback of the span x<sub>1</sub> $\xrightarrow { u _ { 1 } } x _ { 0 } \xleftarrow { u _ { 2 } } x _ { 2 }$ , namely $x _ { 1 } \xleftarrow { p _ { 1 } } P \xrightarrow { p _ { 2 } } x _ { 2 }$ , and the pushout of the cospan y<sub>1</sub> $\underset {  } { \nu _ { 1 } } { } _ { y _ { 0 } } \overset { \nu _ { 2 } } { \longrightarrow } y _ { 2 }$ , namely $\stackrel { q _ { 1 } } { \longrightarrow } \stackrel { q _ { 1 } } { \longrightarrow } \stackrel { q _ { 2 } } { \longrightarrow } \stackrel { q _ { 2 } } { \longrightarrow } y _ { 2 }$ . Since $u _ { 1 } \circ p _ { 1 } = u _ { 2 } \circ p _ { 2 }$ and q<sub>1</sub> $\circ \nu _ { 1 } = q _ { 2 } \circ \nu _ { 2 }$ , we obtain q<sub>1</sub> f<sub>1</sub> p<sub>1</sub> = q<sub>1</sub> v<sub>1</sub> f<sub>0</sub> u<sub>1</sub> p<sub>1</sub> = q<sub>2</sub> v<sub>2</sub> f<sub>0</sub> u<sub>2</sub> p<sub>2</sub> = q<sub>2</sub> f<sub>2</sub> p<sub>2</sub>.

Hence, by the universal property of the pushout, there exists a unique morphism $h : P \to Q$ satisfying $h = q _ { 1 } \circ f _ { 1 } \circ p _ { 1 } = q _ { 2 } \circ f _ { 2 } \circ p _ { 2 }$ . Consequently, $( p _ { 1 } , q _ { 1 } ) : ( P \xrightarrow { h } Q ) \to ( x _ { 1 } \xrightarrow { f _ { 1 } } y _ { 1 } )$ and $( p _ { 2 } , q _ { 2 } ) : ( P \overset { h } {  } Q ) $ $( x _ { 2 } \xrightarrow { f _ { 2 } } y _ { 2 } )$ are morphisms in $\mathsf { D } _ { < } ^ { \triangleright }$ whose composites with $\left( u _ { 1 } , \nu _ { 1 } \right)$ and $\left( u _ { 2 } , \nu _ { 2 } \right)$ coincide.

To verify the universal property, let $( a _ { i } , b _ { i } ) : ( z \xrightarrow { g } w )  ( x _ { i } \xrightarrow { f _ { i } } y _ { i } ) , i = 1 , 2 .$ , be morphisms in $\mathsf { D } _ { < } ^ { \triangleright }$ whose composites with $\left( u _ { 1 } , \nu _ { 1 } \right)$ and $\left( u _ { 2 } , \nu _ { 2 } \right)$ agree. Then u<sub>1</sub> $\circ a _ { 1 } = u _ { 2 } \circ a _ { 2 }$ and $b _ { 1 } \circ \nu _ { 1 } = b _ { 2 } \circ \nu _ { 2 }$ . By the universal property of the pullback, there exists a unique morphism $a : z \to P$ such that $p _ { i } \circ a = a _ { i } , i = 1 , 2$

Similarly, by the universal property of the pushout, there exists a unique morphism $b : Q \to \nu$ such that $b \circ q _ { i } = b _ { i } , i = 1 , 2$ . Moreover,

$$
b \circ h \circ a = b \circ q _ { i } \circ f _ { i } \circ p _ { i } \circ a = b _ { i } \circ f _ { i } \circ a _ { i } = g ,
$$

so $( a , b ) : ( z { \stackrel { g } { \to } } w ) \to ( P { \stackrel { h } { \to } } Q )$ is a morphism in $\mathsf { D } _ { \triangleleft } ^ { \triangleright } .$ . Its uniqueness follows from the uniqueness of a and b. Therefore, $( P { \stackrel { h } { \to } } Q )$ , together with the morphisms $\left( p _ { 1 } , q _ { 1 } \right)$ and $\left( p _ { 2 } , q _ { 2 } \right)$ , is the pullback of $\left( u _ { 1 } , \nu _ { 1 } \right)$ and $\left( u _ { 2 } , \nu _ { 2 } \right)$ in $\mathsf { D } _ { \triangleleft } ^ { \triangleright } .$ □

We can now define T-sheaves on $\mathsf { D } _ { 4 } ^ { \flat }$ as follows.

Proposition 3.2 (T-sheaves on $\mathsf { D } _ { \triangleleft } ^ { \triangleright } )$ Let T be any time category equipped with the Johnstone coverage. Suppose that D has limits and colimits (and therefore $\mathsf { D } _ { \triangleleft } ^ { \triangleright }$ has limits). Then $a ~ \mathsf { D } _ { \triangleleft } ^ { \triangleright }$ -valued sheaf is a presheaf $X : \mathsf { T } ^ { o p } \to \mathsf { D } _ { \triangleleft } ^ { \triangleright }$ such that: for any interval $[ a , b ]$ and any cover $( [ a , p ] , [ p , b ] )$ of this interval, $X ( [ a , b ] )$ is the pullback $X ( [ a , p ] ) \times _ { X ( [ p , p ] ) } X ( [ p , b ] )$

Proof Since the Johnstone coverage is generated by binary covers, the sheaf condition requires precisely the existence of the corresponding pullbacks in $\mathsf { D } _ { \triangleleft } ^ { \triangleright } .$ By Lemma $3 . 1 , \mathsf { D } _ { < } ^ { \triangleright }$ has these pullbacks whenever D has pullbacks and pushouts. The result therefore follows directly from the definition of a sheaf. □

Definition 5 We denote by $\mathsf { N a r } ( \mathsf { T } , \mathsf { D } _ { \mathsf { q } } ^ { \mathsf { D } } )$ the category of $\mathsf { D } _ { \triangleleft } ^ { \triangleright }$ -valued sheaves on T and we call it the category of $\mathsf { D } _ { \triangleleft } ^ { \triangleright }$ -narratives with T-time.

A narrative $X \in { \mathsf { N a r } } ( \mathsf { T } , \mathsf { D } _ { \triangleleft } ^ { \circ } )$ assigns to every interval $[ a , b ]$ a morphism $X _ { a } ^ { b } = ( P _ { a } ^ { b } \xrightarrow [ ] { \gamma _ { a } ^ { b } } C _ { a } ^ { b } )$ in D. The objects $P _ { a } ^ { b }$ form its persistent component, the objects $C _ { a } ^ { b }$ form its cumulative component, and the morphisms $\gamma _ { a } ^ { b }$ compare these two descriptions. Figure 2 illustrates this structure on the category $| _ { \mathbb { N } } / [ 0 , 2 ]$ , whose objects are the intervals contained in $[ 0 , 2 ] \subseteq \mathbb { N }$

Remark 3 The comparison morphisms $\gamma _ { a } ^ { b }$ need not be isomorphisms, even when $a = b$ . Thus the persistent and cumulative descriptions of the same interval generally contain different kinds of information. For example, in Set or Grph, one may interpret $\gamma _ { t } ^ { t } : P _ { t } ^ { t } \to C _ { t } ^ { t }$ as an attribute map assigning to each element or vertex of $P _ { t } ^ { t }$ a color, label, weight, community, role, or other feature in $C _ { t } ^ { t }$ . In this case, $P _ { t } ^ { t }$ represents the collection of objects and $C _ { t } ^ { t }$ represents the collection of possible attributes, with $\gamma _ { t } ^ { t }$ recording the assignment of attributes to objects. The sheaf structure then encodes how both the objects and their attributes relate across overlapping intervals of time: restrictions on the persistent side track how objects change through time, while restrictions on the cumulative side track the compatibility and evolution of their attributes. The sheaf condition ensures that local descriptions on overlapping intervals can be uniquely assembled into a coherent global account of both the objects and their attributes.

We have constructed the category $\mathsf { N a r } ( \mathsf { T } , \mathsf { D } _ { \mathsf { q } } ^ { \mathsf { p } } )$ , whose objects simultaneously encode both the persistent and cumulative descriptions of a temporal object together with the comparison between them. $\mathbf { A } s$ illustrated in Figure 2, the persistent and cumulative components may each arise either from prescribed data (see Remark 3) or from their respective universal constructions. These independent possibilities will later give rise to the classification of narratives by rigidity (Section 3.1.3).

Equivalently,

![](images/665c421f55ccb98359265c21e7e69e3efc148996188cae9cd50b3668f45450a1.jpg)  
Fig. 2: A D-narrative on the category $| _ { \mathbb { N } } / [ 0 , 2 ]$ , whose objects are the intervals contained in $[ 0 , 2 ] \subseteq \mathbb { N }$ . The upper diagram is its persistent component, the lower diagram is its cumulative component, and the dashed morphisms $\gamma _ { a } ^ { b } : P _ { a } ^ { b } \to \dot { C } _ { a } ^ { b }$ compare the two. In the case shown, the persistent data $P _ { 0 } ^ { 0 }  P _ { 0 } ^ { 1 }  P _ { 1 } ^ { 1 }  P _ { 1 } ^ { 2 }  \stackrel {  } { P _ { 2 } ^ { 2 } }$ , is prescribed data, while $P _ { 0 } ^ { 2 }$ is determined by a pullback. On the cumulative side, $C _ { 0 } ^ { 1 } , C _ { 1 } ^ { 2 }$ , and $C _ { 0 } ^ { 2 }$ are determined by pushouts.

## 3.1.2 Factorizing the persistence–accumulation adjunction

Having constructed the category of narratives, we now show that it naturally sits between persistent and cumulative narratives by factorizing the persistence–accumulation adjunction.

Theorem 3.3 The adjunction $\mathcal { H } \mathcal { S }$ factorizes through the category $\mathsf { N a r } ( \mathsf { T } , \mathsf { D } _ { \mathsf { q } } ^ { \mathsf { D } } )$ via thefunctors $\mathcal { P } _ { < } ^ { \triangleright }$ and $\mathcal { H } _ { \mathrm { ~ \tiny ~ < ~ } } ^ { \triangleright }$ . That $i s ,$ both the outer and inner triangles in the following diagram commute:

![](images/75324a1ceacad2a7fbc5508df89dc8b39df83d9a3ff26769ba8c4a0bbdd1119e.jpg)  
cod<sup>▷</sup><sub>◁</sub> K <sup>▷</sup><sub>◁</sub> = K , dom<sup>▷</sup><sub>◁</sub> P<sup>▷</sup><sub>◁</sub> = P .

Moreover,

$$
\begin{array} { r } { \begin{array} { r } { \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } \circ \mathcal { M } _ { \triangleleft } ^ { \triangleright } = \mathsf { i d } _ { \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) } , \qquad \mathsf { c o d } _ { \triangleleft } ^ { \triangleright } \circ \mathcal { P } _ { \triangleleft } ^ { \triangleright } = \mathsf { i d } _ { \mathsf { C u } ( \mathsf { T } , \mathsf { D } ) } . } \end{array} } \end{array}
$$

The forgetful functors recover the persistent and cumulative components of a narrative by forgetting one side of the comparison morphism:

$$
\begin{array} { r } { \mathsf { d o m } _ { \triangleleft } ^ { \mathrm { \tiny \mathrm { \mathrm { P } } } } : \mathsf { N a r } ( \mathsf { T } , \mathsf { D } _ { \triangleleft } ^ { \mathrm { \tiny \mathrm { \mathrm { P } } } } ) \longrightarrow \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) , \qquad \mathsf { c o d } _ { \triangleleft } ^ { \mathrm { \tiny \mathrm { \mathrm { P } } } } : \mathsf { N a r } ( \mathsf { T } , \mathsf { D } _ { \triangleleft } ^ { \mathrm { \tiny \mathrm { \mathrm { P } } } } ) \longrightarrow \mathsf { C u } ( \mathsf { T } , \mathsf { D } ) . } \end{array}
$$

Conversely, the functors

$$
\mathcal { H } _ { \sf 4 } ^ { \mathrm { s } } : \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) \longrightarrow \mathsf { N a r } ( \mathsf { T } , \mathsf { D } _ { \sf 4 } ^ { \mathrm { s } } ) , \qquad \mathcal { P } _ { \sf 4 } ^ { \mathrm { s } } : \mathsf { C u } ( \mathsf { T } , \mathsf { D } ) \longrightarrow \mathsf { N a r } ( \mathsf { T } , \mathsf { D } _ { \sf 4 } ^ { \mathrm { s } } ) ,
$$

canonically complete persistent and cumulative narratives into $\mathsf { D } _ { \triangleleft } ^ { \triangleright } .$ -valued narratives by adjoining their cumulative and persistent counterparts, respectively.

We now construct each of these four functors. Once these constructions are in place, the proof of Theorem 3.3 will follow by verifying that the four triangles commute by construction.

## The forgetful functors.

We first construct the functors

$$
\begin{array} { r } { \mathsf { d o m } _ { \triangleleft } ^ { \mathrm { \tiny \mathrm { \mathrm { P } } } } : \mathsf { N a r } ( \mathsf { T } , \mathsf { D } _ { \triangleleft } ^ { \mathrm { \tiny \mathrm { \mathrm { P } } } } ) \longrightarrow \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) , \qquad \mathsf { c o d } _ { \triangleleft } ^ { \mathrm { \tiny \mathrm { \mathrm { P } } } } : \mathsf { N a r } ( \mathsf { T } , \mathsf { D } _ { \triangleleft } ^ { \mathrm { \tiny \mathrm { \mathrm { P } } } } ) \longrightarrow \mathsf { C u } ( \mathsf { T } , \mathsf { D } ) . } \end{array}
$$

These arise from the canonical domain and codomain projections of the cotwisted arrow category,

$$
\mathsf { D } \xleftarrow { \mathsf { d o m } } \mathsf { D } _ { \triangleleft } ^ { \triangleright } \xrightarrow { \mathsf { c o d } } \mathsf { D } ^ { o p } ,
$$

where dom sends an arrow $x \overset { f } { \to } y$ to its domain x, while cod sends it to its codomain y. On a morphism $( u , \nu ) : ( x \stackrel { f } { \to } y ) \to ( x ^ { \prime } \stackrel { f ^ { \prime } } { \to } y ^ { \prime } )$ in $\mathsf { D } _ { \triangleleft } ^ { \triangleright }$ , they are given by

$$
x \xrightarrow [ \hphantom { \ ' } \longrightarrow ] { u } x ^ { \prime } \quad \underbrace { \not { q } \mathrm { o m } } _ { \begin{array} { c } { { f } } \\ { { \chi \xrightarrow [ \hphantom { \ ' } ] { u } } } \end{array} } \quad f \underbrace { \begin{array} { c } { u \xrightarrow [ \hphantom { \ ' } ] { u } } \end{array} } _ { \begin{array} { c } { { y } } \\ { { y } } \end{array} } \quad \underbrace { \begin{array} { c } { u } \\ { { \chi \smash [ \mathstrut ] { } } } \end{array} } _ { \begin{array} { c } { { y } } \end{array} } , \quad y \xrightarrow [ \hphantom { \ ' } \ ] { \nu } y ^ { \prime }
$$

Thus dom is covariant, whereas cod is naturally viewed as taking values in $\mathsf { D } ^ { o p }$

## Proposition 3.4 Composition with dom and cod defines functors

$$
\mathbf { d o m } _ { < } ^ { \scriptscriptstyle \mathrm { P } } : \mathsf { N a r } ( \mathsf { T } , \mathsf { D } _ { < } ^ { \scriptscriptstyle \mathrm { P } } ) \longrightarrow \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) , \qquad \mathsf { c o d } _ { \scriptscriptstyle \mathrm { d } } ^ { \scriptscriptstyle \mathrm { P } } : \mathsf { N a r } ( \mathsf { T } , \mathsf { D } _ { < } ^ { \scriptscriptstyle \mathrm { P } } ) \longrightarrow \mathsf { C u } ( \mathsf { T } , \mathsf { D } )
$$

given on objects by

$$
\mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X ) = \mathsf { d o m o } X , \qquad \mathsf { c o d } _ { \triangleleft } ^ { \triangleright } ( X ) = \mathsf { c o d } \circ X .
$$

Proof Let $X : \mathsf { T } ^ { o p } \to \mathsf { D } _ { \triangleleft } ^ { \triangleright }$ be a narrative. For each interval $[ a , b ]$ , write

$$
\begin{array} { r } { X _ { a } ^ { b } = \bigg ( P _ { a } ^ { b } \overset { \gamma _ { a } ^ { b } } { \longrightarrow } C _ { a } ^ { b } \bigg ) . } \end{array}
$$

We define dom<sup>▷</sup>(X) on objects by $[ a , b ] \mapsto P _ { a } ^ { b }$ and ${ \mathsf { c o d } } _ { \triangleleft } ^ { \triangleright } ( X )$ by $[ a , b ] \mapsto C _ { a } ^ { b }$ . Given a morphism $i : [ a , b ] \hookrightarrow [ c , d ]$ , if $X ( i ) = ( p _ { i } , c _ { i } )$ , we put dom $\mathfrak { r } _ { \mathtt { q } } ^ { \mathtt { D } } ( X ) ( i ) = p _ { i }$ and ${ \mathsf { c o d } } _ { \triangleleft } ^ { \triangleright } ( X ) ( i ) = c _ { i }$ . Since composition in $\mathsf { D } _ { \triangleleft } ^ { \triangleright }$ is defined componentwise, both assignments preserve identities and composition. Furthermore, because X is a sheaf valued in $\mathsf { D } _ { \triangleleft } ^ { \triangleright } ,$ applying dom to each pullback diagram recovers the sheaf condition defining persistent narratives, while applying cod recovers the corresponding pushout condition defining cumulative narratives. Thus, dom $\mathfrak { p } _ { \mathtt { q } } ^ { \mathtt { p } } ( X ) \in \mathsf { P e } ( \mathsf { T } , \mathsf { D } )$ and $\mathsf { c o d } _ { \triangleleft } ^ { \triangleright } ( X ) \in \mathsf { \bar { C } } \mathsf { u } ( \mathsf { T } , \mathsf { D } )$ . Finally, a morphism of narratives is a natural transformation in $\mathsf { D } _ { \triangleleft } ^ { \triangleright } ,$ and composing it componentwise with dom or cod yields natural transformations between the corresponding persistent or cumulative narratives. Therefore, dom<sup>▷</sup><sub>◁</sub> and $\mathsf { c o d } _ { \triangleleft } ^ { \triangleright }$ define functors. □

## The cumulative completion.

Having constructed the forgetful functors, we now turn to the opposite direction. Our next goal is to show that every persistent narrative admits a canonical cumulative completion into a $\mathsf { D } _ { \triangleleft } ^ { \triangleright }$ -valued narrative.

Proposition 3.5 (Cumulative completion of a persistent narrative) There is a functor

$$
\mathcal { H } _ { \sf d } ^ { \sf D } \colon \sf P e ( \sf T , \sf D )  \sf N a r ( \sf T , \sf D _ { \sf d } ^ { \sf D } )
$$

that sends a persistent narrative $F : \mathsf { T } ^ { o p } \to \mathsf { D }$ to the narrative

$$
\mathcal { H } _ { \triangleleft } ^ { \triangleright } F \colon \mathsf { T } ^ { o p } \to \mathsf { D } _ { \triangleleft } ^ { \triangleright } .
$$

defined asfollows. Given an interval $[ a , b ] \in \mathsf { T } , \mathcal { H } _ { \leqslant } ^ { \triangleright } F$ maps it to the object

$$
F _ { a } ^ { b } \xrightarrow { \alpha _ { a } ^ { b } } \mathcal { H } F _ { a } ^ { b }
$$

of $\mathsf { D } _ { \triangleleft } ^ { \triangleright } ,$ where $\boldsymbol { \mathfrak { a } } _ { a } ^ { b }$ is the canonical cocone map into the colimit defining $\mathcal { H } F _ { a } ^ { b }$ . Given an inclusion $i \colon [ a , \dot { b } ] \hookrightarrow [ c , d ] , \tilde { \mathcal { K } } _ { \ q } ^ { \flat } F$ maps it to the morphism in $\mathsf { D } _ { < } ^ { \triangleright }$

$$
\bigl ( F _ { c } ^ { d } \xrightarrow { F ( i ) } F _ { a } ^ { b } , \mathcal { H } F _ { a } ^ { b } \xrightarrow { \mathcal { H } F ( i ) } \mathcal { H } F _ { c } ^ { d } \bigr ) ,
$$

which satisfies the cotwisted commutativity condition:

. Equivalently,

![](images/12066af7382d8e38d38424e3169431da5a4c65395834810082c2a492f526fe34.jpg)

$$
\boldsymbol { \alpha } _ { c } ^ { d } = \mathcal { H } \boldsymbol { F } ( i ) \circ \boldsymbol { \alpha } _ { a } ^ { b } \circ \boldsymbol { F } ( i ) .
$$

Proof For each interval $[ a , b ]$ , the object $\mathcal { H } F _ { a } ^ { b }$ is defined by the colimit in Equation (1), and

$$
\boldsymbol { \alpha } _ { a } ^ { b } : \boldsymbol { F } _ { a } ^ { b } \longrightarrow \mathcal { H } \boldsymbol { F } _ { a } ^ { b }
$$

is the corresponding canonical cocone map. If $i \colon [ a , b ] \hookrightarrow [ c , d ] \in \mathsf { T }$ , then $F ( i ) \colon F _ { c } ^ { d }  F _ { a } ^ { b }$ is induced by the functoriality of $F \in \mathsf { P e } ( \mathsf { T } , \mathsf { D } )$ . Moreover, i induces a restriction of the canonical cocone defining $\mathcal { H } F _ { c } ^ { d }$ to a cocone on the diagram defining $\mathcal { H } F _ { a } ^ { b } . \mathrm { B y }$ the universal property of the colimit $\mathcal { H } F _ { a } ^ { b }$ , there is a unique morphism

$$
\mathcal { H } F ( i ) \colon \mathcal { H } F _ { a } ^ { b } \longrightarrow \mathcal { H } F _ { c } ^ { d }
$$

such that

$$
\boldsymbol { \alpha } _ { c } ^ { d } = \mathcal { H } \boldsymbol { F } ( i ) \circ \boldsymbol { \alpha } _ { a } ^ { b } \circ \boldsymbol { F } ( i ) .
$$

Hence,

$$
\left( F ( i ) , { \mathcal { H } } F ( i ) \right) : \left( F _ { c } ^ { d } \xrightarrow { \alpha _ { c } ^ { d } } { \mathcal { H } } F _ { c } ^ { d } \right) \longrightarrow \left( F _ { a } ^ { b } \xrightarrow { \alpha _ { a } ^ { b } } { \mathcal { H } } F _ { a } ^ { b } \right)
$$

is a morphism in $\mathsf { D } _ { \triangleleft } ^ { \triangleright } .$ . Functoriality of $\mathcal { H } _ { \triangleleft } ^ { \triangleright } F \colon \mathsf { T } ^ { o p } \to \mathsf { D } _ { \triangleleft } ^ { \triangleright }$ follows immediately from the functoriality of $F , \mathbf { o f } \mathcal { H } F$ , and from the uniqueness of the induced maps between the corresponding colimits.

It remains to show that ${ \mathcal { H } } _ { \leq } ^ { \triangleright } F$ is a sheaf. Let $[ a , b ] \in \mathsf { T }$ , and consider the cover $( [ a , p ] , [ p , b ] )$ . We must show that $\mathcal { H } _ { \triangleleft } ^ { \triangleright } F _ { a } ^ { b }$ is the following pullback in $\mathsf { D } _ { < } ^ { \triangleright }$

![](images/3831242009d7481f8795e3e070b66524936f59f795122e6f342329e41cac17d8.jpg)

This square is well defined in $\mathsf { D } _ { \triangleleft } ^ { \triangleright } .$ Indeed, $F$ is a persistent sheaf, so $F _ { a } ^ { b }$ is the pullback of $F _ { a } ^ { p } \to F _ { p } ^ { p } \gets$ $F _ { p } ^ { b } .$ whereas $\mathcal { H } F$ is a cumulative narrative, so $\mathcal { H } F _ { a } ^ { b }$ is the pushout of $\mathcal { H } F _ { a } ^ { p } \left. \mathcal { H } F _ { p } ^ { p } \right. \mathcal { H } F _ { p } ^ { b } .$ . Let u : $x  .$ y be an arbitrary object of $\mathsf { D } _ { \triangleleft } ^ { \triangleright } ,$ together with compatible morphisms $( p _ { l } , c _ { l } ) \colon ( x \stackrel { u } { \to } y ) \to \mathcal { H } _ { \mathrm { \scriptsize ~ { \ 4 } } } ^ { \triangleright } F _ { a } ^ { p }$ and $( p _ { r } , c _ { r } ) \colon ( x \stackrel { u } { \to } y ) \to \mathcal { H } _ { \mathrm { ~ q } } ^ { \triangleright } F _ { p } ^ { b }$ , as in the following diagram.

![](images/fb1a19c46aa6ebc8a0ebc1bd2a3f2d2faf20dd60a4c0f1eb060b81fdaa921e15.jpg)

Equivalently, the following diagram in D commutes.

![](images/a504cbd59586c6553ecd29be5d0a7b4969e0694eedf6f687485eaf996dc9dc7e.jpg)  
Since the upper square is a pullback, there is a unique morphism $p \colon x \to F _ { a } ^ { b }$ such that $F ( f ) \circ p = p _ { l }$ and $F ( g ) \circ p = p _ { r }$ . Likewise, since the lower square is a pushout, there is a unique morphism $c \colon \mathcal { H } F _ { a } ^ { b } \to$ $y$ such that $c \circ \mathcal { H } F ( h ) = c _ { l }$ and $c \circ \mathcal { H } F ( k ) = c _ { r }$ . Moreover, the commutativity of the outer diagrams implies that $u = c \circ \mathbf { \boldsymbol { \mathsf { C } } } _ { a } ^ { b } \circ p ,$ so $( p , c ) \colon ( x \stackrel { u } { \to } y ) \to ( F _ { a } ^ { b } \stackrel { \alpha _ { a } ^ { b } } { \to } \mathcal { H } F _ { a } ^ { b } )$ is a morphism in $\mathsf { D } _ { \triangleleft } ^ { \triangleright } .$ . Since p and c are uniquely determined by the pullback and pushout universal properties, respectively, $( p , c )$ is the unique morphism making the required diagrams commute. Therefore, $\mathcal { H } _ { \triangleleft } ^ { \triangleright } F _ { a } ^ { b }$ satisfies the universal property of the pullback in $\mathsf { D } _ { \triangleleft } ^ { \triangleright } .$ , and hence, $\mathcal { H } _ { \triangleleft F } ^ { \triangleright }$ is a sheaf. □

## The persistent completion.

Dually, every cumulative narrative admits a canonical persistent completion into a $\mathsf { D } _ { \triangleleft } ^ { \triangleright }$ -valued narrative.

Proposition 3.6 (Persistent completion of a cumulative narrative) There is a functor

$$
\mathcal { P } _ { \mathcal { A } } ^ { \sf p } : \mathsf { C u } ( \mathsf { T } , \mathsf { D } ) \to \mathsf { N a r } ( \mathsf { T } , \mathsf { D } _ { \mathcal { A } } ^ { \sf p } )
$$

that assigns to each cumulative narrative $\hat { F } \in \mathsf { C u } ( \mathsf { T } , \mathsf { D } )$ the $\mathsf { D } _ { \triangleleft } ^ { \triangleright } .$ -valued narrative $\mathcal { P } _ { \triangleleft } ^ { \triangleright } \hat { F }$ defined as follows.

• On objects $[ a , b ] \in \mathsf { T }$ , the sheaf $\mathcal { P } _ { \triangleleft } ^ { \triangleright } \hat { F }$ assigns the morphism

$$
\mathcal { P } \hat { F } _ { a } ^ { b } \xrightarrow { \{ \beta _ { a } ^ { b } \} } \hat { F } _ { a } ^ { b } ,
$$

viewed as an object in the cotwisted arrow category $\mathsf { D } _ { \triangleleft } ^ { \triangleright }$ . The morphism $\beta _ { a } ^ { b }$ is the limit cone morphismfrom $\mathcal { \dot { P } } _ { \boldsymbol { { f } } _ { a } } ^ { b }$ to the object $\hat { F } _ { a } ^ { b }$

• On morphisms $[ a , b ] \stackrel { i } {  } [ c , d ]$ in T, the sheaf $\mathcal { P } _ { \triangleleft } ^ { \triangleright } \hat { F }$ assigns the morphism in $\mathsf { D } _ { \triangleleft } ^ { \triangleright }$

$$
( \mathcal { P } \hat { F } _ { c } ^ { d } \xrightarrow { \mathcal { P } \hat { F } ( i ) } \mathcal { P } \hat { F } _ { a } ^ { b } , \hat { F } _ { a } ^ { b } \xrightarrow { \hat { F } ( i ) } \hat { F } _ { c } ^ { d } ) ,
$$

which satisfies the cotwisted commutativity condition

![](images/7088fb581d70da2fd2b61e0d6283a3984c4cd6b506f43e67cf0a063eba17f3cb.jpg)

. Explicitly,

$$
\mathsf { \beta } _ { c } ^ { d } = \hat { F } ( i ) \circ \mathsf { \beta } _ { a } ^ { b } \circ \mathcal { P } \hat { F } ( i ) .
$$

Proof The construction is the categorical dual of Proposition 3.5. For each interval $[ a , b ] ,$ the object $\mathcal { P } \hat { F } _ { a } ^ { b }$ is defined by the limit in Equation (2), with canonical cone morphism $\beta _ { a } ^ { b } \colon \mathcal { P } \hat { F } _ { a } ^ { b } \to \mathsf { \bar { F } } _ { a } ^ { b }$ Given an inclusion $i \colon [ a , b ] \hookrightarrow [ c , d ]$ , the universal property of the limit induces a unique morphism $\mathcal { P } \hat { F } ( i ) \colon \mathcal { P } \hat { F } _ { c } ^ { d } \to \mathcal { P } \hat { F } _ { a } ^ { b }$ satisfying $\bar { \beta } _ { c } ^ { d } = \hat { F } ( i ) \circ \beta _ { a } ^ { b } \circ \mathcal { \hat { P } } \hat { F } ( i )$ . Hence $( \mathcal { P } \hat { F } ( i ) , \hat { F } ( i ) )$ is a morphism in $\mathsf { D } _ { \triangleleft } ^ { \triangleright } ,$ and these assignments define the functor $\mathcal { P } _ { \leq } ^ { \triangleright } \hat { F } .$

To verify the sheaf axiom, let $( [ a , p ] , [ p , b ] )$ be a cover of $[ a , b ]$ . Given an arbitrary object u: $x  y$ of $\mathsf { D } _ { \triangleleft } ^ { \triangleright }$ together with compatible morphisms to $\mathcal { P } _ { \triangleleft } ^ { \triangleright } \hat { F } _ { a } ^ { p }$ and $\dot { \mathcal { P } } _ { \ < } ^ { \triangleright } \hat { F } _ { p } ^ { \bar { b } }$ , since $\hat { F }$ is a cumulative narrative, $\hat { F } _ { a } ^ { b }$ is the pushout of $\hat { F } _ { a } ^ { p } \gets \hat { F } _ { p } ^ { p } \to \hat { F } _ { p } ^ { b }$ , while $\mathcal { P } \hat { F }$ is a persistent narrative, so $\mathcal { P } \hat { F } _ { a } ^ { b }$ is the pullback of $\mathcal { P } \hat { F } _ { a } ^ { p } \right. \mathcal { P } \hat { F } _ { p } ^ { p } \left. \mathcal { P } \hat { F } _ { p } ^ { b }$ . The universal properties of the pushout $\hat { F } _ { a } ^ { b }$ and the pullback $\mathcal { P } \hat { F } _ { a } ^ { b }$ yield unique morphisms $p \colon x \to \mathcal { P } \hat { F } _ { a } ^ { b }$ and $c \colon \hat { F } _ { a } ^ { b } \to y .$ , which satisfy the cotwisted compatibility condition and therefore determine the unique morphism $( p , c )$ in $\mathsf { D } _ { \triangleleft } ^ { \triangleright } .$ . The verification of this compatibility is exactly dual to that in Proposition 3.5, and is therefore omitted. □

Proof of Theorem 3.3 The functors involved are well defined by Proposition 3.4, Proposition 3.5, and Proposition 3.6. It remains to verify the identities

$$
\mathsf { c o d } _ { \triangleleft } ^ { \triangleright } \circ \mathcal { M } _ { \triangleleft } ^ { \triangleright } = \mathcal { H } , \qquad \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } \circ \mathcal { P } _ { \triangleleft } ^ { \triangleright } = \mathcal { P } ,
$$

$$
\begin{array} { r } { \begin{array} { r } { \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } \circ \mathcal { M } _ { \triangleleft } ^ { \triangleright } = \mathsf { i d } _ { \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) } , \qquad \mathsf { c o d } _ { \triangleleft } ^ { \triangleright } \circ \mathcal { P } _ { \triangleleft } ^ { \triangleright } = \mathsf { i d } _ { \mathsf { C u } ( \mathsf { T } , \mathsf { D } ) } . } \end{array} } \end{array}
$$

For $F \in \mathsf { P e } ( \mathsf { T } , \mathsf { D } )$ , we have $\mathcal { H } _ { \triangleleft } ^ { \triangleright } F _ { a } ^ { b } = ( F _ { a } ^ { b } \xrightarrow { \alpha _ { a } ^ { b } } \mathcal { H } F _ { a } ^ { b } )$ and, for every inclusion $i : [ a , b ] \hookrightarrow [ c , d ]$ $\mathcal { H } _ { \triangleleft } ^ { \triangleright } F ( i ) = \left( F ( i ) , \mathcal { H } F ( i ) \right)$ . Hence,

$$
\mathsf { c o d } _ { \triangleleft } ^ { \triangleright } ( \mathcal { M } _ { \triangleleft } ^ { \triangleright } F ) = \mathcal { M } F , \qquad \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( \mathcal { M } _ { \triangleleft } ^ { \triangleright } F ) = F ,
$$

so co $\pounds _ { \triangle } ^ { \triangleright } \circ \mathcal { K } _ { \triangleleft } ^ { \triangleright } = \mathcal { K }$ and dom $\mathfrak { L } _ { \ v q } ^ { \triangleright } \circ \mathcal { H } _ { \ v q } ^ { \triangleright } = \mathrm { i d } _ { \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) }$

Similarly, for $\hat { F } \in \mathsf { C u } ( \mathsf { T } , \mathsf { D } )$ , we have $\mathcal { P } _ { \triangleleft } ^ { \triangleright } \hat { F } _ { a } ^ { b } = ( \mathcal { P } \hat { F } _ { a } ^ { b } \xrightarrow { \sharp _ { a } ^ { b } } \hat { F } _ { a } ^ { b } )$ and, for every inclusion $i : [ a , b ] \hookrightarrow$ $[ c , d ] , \mathcal { P } _ { { \check { \mathcal { A } } } } ^ { \triangleright } { \hat { F } } ( i ) = \left( \mathcal { P } { \hat { F } } ( i ) , { \dot { F } } ( i ) \right)$ . Hence,

$$
\mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( \mathcal P _ { \triangleleft } ^ { \triangleright } \hat { F } ) = \mathcal P \hat { F } , \qquad \mathsf { c o d } _ { \triangleleft } ^ { \triangleright } ( \mathcal P _ { \triangleleft } ^ { \triangleright } \hat { F } ) = \hat { F } ,
$$

so dom $\ R _ { < } ^ { > } \circ \mathcal { P } _ { < } ^ { > } = \mathcal { P }$ and cod $\mathsf { I } _ { \triangle } ^ { \triangleright } \circ \mathcal { P } _ { \triangleleft } ^ { \triangleright } = \mathsf { i d } _ { \mathsf { C u ( T , D ) } }$

## 3.1.3 Classifying narratives by rigidity

Since $X \in \mathsf { N a r } ( \mathsf { T } , \mathsf { D } _ { \triangleleft } ^ { \circ } )$ is valued in the cotwisted arrow category, it canonically determines, for every interval $[ a , b ]$ , a comparison morphism

$$
\gamma _ { a } ^ { b } : { \cal P } _ { a } ^ { b } \longrightarrow C _ { a } ^ { b } ,
$$

where

$$
\begin{array} { r } { X _ { a } ^ { b } = \bigg ( P _ { a } ^ { b } \overset { \gamma _ { a } ^ { b } } {  } C _ { a } ^ { b } \bigg ) . } \end{array}
$$

Applying the completion functors to the persistent and cumulative components of X produces the narratives

$$
\mathcal { H } _ { \sf { q } } ^ { \sf p } ( { \sf d } { \sf { o } } { \sf m } _ { \sf { 4 } } ^ { \sf p } ( X ) ) \qquad \mathrm { a n d } \qquad \mathcal { P } _ { \sf { 4 } } ^ { \sf p } ( { \sf c o d } _ { \sf { 4 } } ^ { \sf p } ( X ) ) .
$$

By construction of the functors $\mathcal { H } _ { \mathrm { ~ \sf ~ < ~ } } ^ { \sf D }$ and $\mathcal { P } _ { \mathtt { q } } ^ { \mathtt { b } } ,$ the structure morphism of $\mathcal { H } _ { \sf < 1 } ^ { \triangleright } ( \mathsf { d o m } _ { \sf < 1 } ^ { \triangleright } ( X ) )$ at an interval $[ a , b ]$ is the component

$$
\big ( \eta _ { \mathsf { d o m } _ { \mathscr { A } } ^ { \flat } ( X ) } \big ) _ { a } ^ { b } \colon \mathsf { d o m } _ { \mathscr { A } } ^ { \flat } ( X ) _ { a } ^ { b } \longrightarrow ( \mathcal { P } \mathcal { M } \mathsf { d o m } _ { \mathscr { A } } ^ { \flat } ( X ) ) _ { a } ^ { b }
$$

of the unit of the adjunction, while the structure morphism of ${ \mathcal { P } } _ { \leq } ^ { \triangleright } ( \mathsf { c o d } _ { \triangleleft } ^ { \triangleright } ( X ) )$ is the component

$$
\big ( \pmb { \varepsilon } _ { \mathsf { c o d } _ { \mathscr { A } } ^ { \flat } ( X ) } \big ) _ { a } ^ { b } : ( \mathcal { M P } \mathsf { c o d } _ { \mathscr { A } } ^ { \flat } ( X ) ) _ { a } ^ { b } \longrightarrow \mathsf { c o d } _ { \mathscr { A } } ^ { \flat } ( X ) _ { a } ^ { b }
$$

of the counit. These assemble into natural transformations

$$
\eta _ { \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X ) } : \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X ) \Longrightarrow { \mathscr { P H } ( \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X ) ) }
$$

and

$$
\mathfrak { E } _ { \mathsf { c o d } _ { \triangleleft } ^ { \triangleright } ( X ) } : \mathcal { M P } ( \mathsf { c o d } _ { \triangleleft } ^ { \triangleright } ( X ) ) \Longrightarrow \mathsf { c o d } _ { \triangleleft } ^ { \triangleright } ( X ) ,
$$

which we call the canonical comparison morphisms associated to the narrative X. These comparison morphisms measure how closely the persistent and cumulative descriptions encoded by a narrative agree with the canonical ones determined by the persistence–accumulation adjunction. This observation motivates the following rigidity classification of narratives.

Definition 6 (Left rigid narrative) A narrative $X \in \mathsf { N a r } ( \mathsf { T } , \mathsf { D } _ { \triangleleft } ^ { \triangleright } )$ is called left rigid if the comparison morphism

$$
\mathfrak { N } _ { \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X ) } : \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X ) \longrightarrow \mathcal { P } \mathcal { H } ( \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X ) )
$$

is an isomorphism.

Equivalently, the persistent component of X is completely recovered after passing to its canonical cumulative completion and back.

Example 4 (Spans of monomorphisms in adhesive categories) Assume that the ambient category D is adhesive, and let $X \in { \mathsf { N a r } } ( \mathsf { T } , \mathsf { D } _ { \triangleleft } ^ { \triangleright } )$ be a narrative whose persistent component

$$
\mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X ) _ { 0 } ^ { 0 } \gets \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X ) _ { 0 } ^ { 1 } \to \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X ) _ { 1 } ^ { 1 }
$$

consists of monomorphisms. Since pushouts along monomorphisms are Van Kampen [19], the associated pushout square is also a pullback. Hence, ${ \mathcal { P } } { \mathcal { H } } ( \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X ) ) \cong \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X )$ , and therefore, X is left rigid.

This class of examples is particularly relevant in double-pushout graph rewriting, where rewriting rules are represented by spans of monomorphisms in adhesive categories [19, 20]. Consequently, for this broad class of graph transformations, the passage from persistent to cumulative descriptions and back preserves the original persistent description.

Remark 4 It is worth emphasizing that the converse need not hold. Although adhesive categories guarantee that spans of monomorphisms are preserved by the pushout–pullback composite, they do not in general imply that cospans are preserved by the pullback–pushout composite. Thus, even in adhesive categories, left rigidity does not automatically imply right rigidity.

Definition 7 (Right rigid narrative) A narrative $X \in { \mathsf { N a r } } ( \mathsf { T } , \mathsf { D } _ { \triangleleft } ^ { \triangleright } )$ is called right rigid if the comparison morphism

$$
\mathfrak { E } _ { \mathsf { c o d } _ { \triangleleft } ^ { \triangleright } ( X ) } : \mathcal { H P } ( \mathsf { c o d } _ { \triangleleft } ^ { \triangleright } ( X ) ) \longrightarrow \mathsf { c o d } _ { \triangleleft } ^ { \triangleright } ( X )
$$

is an isomorphism.

Equivalently, the cumulative component of $X$ is completely recovered after passing to its canonical persistent completion and back.

Example 5 (A right rigid narrative) Consider the narrative X given by:

![](images/1ed33a3e6530d635480ae9e7e2d38312d35dec7aff90fb757ab1592115697ff2.jpg)

Applying $\mathcal { P }$ to the cumulative component cod ${ \mathbb { I } } _ { \triangleleft } ^ { \triangleright } ( X )$ computes the pullback $\{ a \} \times _ { \{ * \} } \{ b \} \cong \{ ( a , b ) \}$ Applying $\mathcal { K }$ to the persistent narrative $\{ a \} \xleftarrow { \pi _ { \ell } } \left\{ ( a , b ) \right\} \xrightarrow { \pi _ { r } } \{ b \} = \mathcal { P } ( \mathsf { c o d } _ { \prec } ^ { \scriptscriptstyle \triangleright } ( X ) )$ computes its pushout, which is again the singleton $\{ * \}$ . Therefore, the counit $\mathfrak { E } _ { \mathsf { c o d } _ { \triangleleft } ^ { \triangleright } ( X ) } : \mathcal { M P } ( \mathsf { c o d } _ { \triangleleft } ^ { \triangleright } ( X ) ) \longrightarrow \mathsf { c o d } _ { \triangleleft } ^ { \triangleright } ( X )$ is an isomorphism. Hence, X is right rigid. Moreover, X is not left rigid. Indeed, ${ \mathcal { P K } } ( \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X ) )$ has apex $\{ ( a , b ) \}$ , while dom ${ \mathbb { \Gamma } } _ { \triangleleft } ^ { \triangleright } ( X )$ has apex 0,1 . Hence the unit $\mathfrak { N } _ { \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X ) } : \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X ) \longrightarrow \mathcal { P } \mathcal { H } ( \mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X ) )$ is not an isomorphism.

Definition 8 (Rigid and loose narratives) A narrative is called

• rigid if it is both left rigid and right rigid;

• loose if it is neither left rigid nor right rigid.

Example 6 (Rigid narratives) The simplest examples of rigid narratives are the constant ones. Let $d \in \mathsf { D }$ The constant narrative is the $\mathsf { D } _ { \triangleleft } ^ { \triangleright }$ -valued sheaf X defined by $X _ { a } ^ { b } = ( d \xrightarrow { \mathsf { i d } _ { d } } d )$ for every interval $[ a , b ] \in \mathsf { T }$ with every restriction morphism equal to $( \mathsf { i d } _ { d } , \mathsf { i d } _ { d } )$ . Since both the persistent and cumulative components are constant, one ha

$$
\mathsf { d o m } _ { \mathsf { d } } ^ { \mathrm { s } } ( X ) = \mathcal { P } ( \mathsf { c o d } _ { \mathsf { d } } ^ { \mathrm { s } } ( X ) ) \qquad \mathrm { ~ a n d ~ } \qquad \mathsf { c o d } _ { \mathsf { d } } ^ { \mathrm { s } } ( X ) = \mathcal { H } ( \mathsf { d o m } _ { \mathsf { d } } ^ { \mathrm { s } } ( X ) ) .
$$

Hence X is rigid. More generally, the same conclusion holds whenever the morphisms of the narrative are isomorphisms. Indeed, replacing the identities above by arbitrary isomorphisms does not change either the pullback or pushout constructions up to canonical isomorphism, so both comparison morphisms remain isomorphisms. Consider, for instance:

![](images/c6c57372796a74a0a567e0443c7cf592765a2eef31c1034eb86f5c73362c68f3.jpg)

The persistent component $\{ a \} \stackrel { \pi _ { \ell } } { \left. } \{ ( a , b ) \} \stackrel { \pi _ { r } } { \right. }$ b is already the pullback of the cospan $\{ a \} \xrightarrow { c _ { \ell } } \{ * \} \xrightarrow { c _ { r } }$ b , while the pushout of this pullback is again the singleton .

Example 7 (Loose narratives) Loose narratives already appear in very simple categories. Consider the poset category $( \mathbb { N } , \leq )$ , whose objects are natural numbers and in which there exists a unique morphism $m \to n$ precisely when m $\leq n .$ . In this category, pullbacks are given by meets $m \times _ { p } n = \operatorname* { m i n } \{ m , n \}$ , while pushouts are given by joins $n + _ { p } n = \operatorname* { m a x } \{ m , n \}$

Consider the narrative

![](images/6a4b855471a412167972d0d3ec0e93ba7e3934e17b55a5b569a608ca9147c473.jpg)

Its domain is the span dom $\mathfrak { r } _ { \mathrm { q } } ^ { \triangleright } ( X ) = ( 1 \left. 0 \right. 2 )$ , and applying the completion functors yields $\mathcal { H } ( \mathsf { d o m } _ { \cdot } ^ { \triangleright } ( X ) ) = ( 1 \right. 2 \left. 2 )$ and $\mathcal { P K } ( \mathsf { d o m } _ { \cdot } ^ { \flat } ( X ) ) = ( 1 \left. 1 \right. 2 )$ , which is not isomorphic to $\mathsf { d o m } _ { \triangleleft } ^ { \triangleright } ( X )$ . Thus, X is not left rigid. Likewise, its codomain is the cospan c $\mathsf { \partial } \mathsf { d } _ { \triangle } ^ { \triangleright } ( X ) = ( 1 \to 3  2 )$ and applying the completion functors gives $\mathcal { P } ( \mathsf { c o d } _ { \triangleleft } ^ { \triangleright } ( X ) ) = ( 1 \left. 1 \right. 2 )$ and $\mathcal { H P } ( \mathsf { c o d } _ { \triangleleft } ^ { \triangleright } ( X ) ) =$ $( 1 \right. 2 \left. 2 )$ , which is not isomorphic to co $\mathbb { d } _ { \triangleleft } ^ { \triangleright } ( X )$ . Hence, X is not right rigid, and therefore it is loose.

The central question explored in this vignette is when persistent and cumulative descriptions determine one another exactly. Investigating the fixed points of the persistence– accumulation adjunction naturally leads to the category of narratives, which simultaneously records both descriptions together with the comparison between them. This approach provides a specialized framework for systematically investigating the conditions—relating to data category and temporal data assignments—under which information is preserved as one transitions between persistent and cumulative representations.

## 3.2 Vignette 2: Decomposing Time-Varying Data into Simple Pieces: Structured Decompositions of Narratives

Complex data, whether static or time-varying, is often easier to understand and analyze when it is represented as a composite of smaller or simpler pieces. Thus, in this section, we ask: how does one decompose complicated time-varying data into simpler pieces in a systematic way? A decomposition serves two roles: (1) it divides an object into component pieces, and (2) it records how these pieces overlap, so that the original object may be reconstructed by gluing them together. Thinking about this with a more topological slant, there is a sense in which this section is about seeking simple coverings of a temporal object by small, time-varying pieces, analogous to open sets, whose evolution and mutual compatibility are themselves tracked through time.

Our goal is to obtain an invariant of the structural complexity of a time-varying object by measuring the complexity of the pieces in its decompositions and their overlaps. These pieces must themselves form time-varying objects, with structure maps recording how they persist, merge, split, or disappear. It is not enough to decompose each snapshot independently: the pieces of the decomposition must also capture the evolution of the global structure. We seek a method that reuses theories of decomposition for static data instead of defining a new notion for each temporal setting. This section is based on the recent paper by Bumpus and Nickel [16] whose main result provides such a method by lifting theories of decompositions from static objects to persistent narratives.

## 3.2.1 Structured decompositions

In this section, we use narratives to construct a temporalized theory of structured decompositions, which were introduced in [21] and provide a category theoretical generalization of tree-decompositions. The goal of our approach is to split time-varying data systematically into smaller components, providing a formal framework for dealing with temporal systems. This is an instance of the broader principle of compositionality, which states that the meaning or behaviour of a complex system is completely determined by the meanings or behaviours of its constituent parts and the rules governing their connections.

Breaking complicated data into simpler pieces has already proved useful in many areas of mathematics and computer science, particularly in graph theory, logic, algorithms, and complexity theory. Prominent examples include Robertson and Seymour’s graph structure theorem [22] and Courcelle’s theorem [23].

Here, a graph G means a diagram from the category $V \Longrightarrow E$ to the category Set of sets. Thus, a graph in this sense is the same as a quiver in representation theory and as a copresheaf on $V \Longrightarrow E$ in category theory. In particular, our graphs are directed and are allowed to possess loops and multiple edges. Given a graph G, we denote the set of its vertices by $V ( G ) : = G ( V )$ and the set of its edges by $E ( G ) : = G ( E )$ . We write Grph for the wide subcategory of the functor category $[ V { \Longrightarrow } E , { \mathsf { S e t } } ]$ whose objects are the graphs. Its morphisms are simply natural transformations between graphs, also called graph morphisms.

## Tree-decompositions.

Originally introduced by Halin in 1976 [24], the notion of a tree-decomposition (see Definition 9) was rediscovered in 1984 by Robertson and Seymour [25] and has since played an essential role in structural and extremal graph theory, as well as in various other areas of mathematics and computer science. For instance, tree-decompositions have proved to be a convenient tool in matrix decomposition [26], query optimization [27], and dynamic programming [28]. They are also frequently used to solve constraint satisfaction problems [29] and in junction tree algorithms for probabilistic inference [30]. Robertson and Seymour themselves used tree-decompositions as a crucial component of their celebrated graph structure theorem [22], which establishes a profound connection between graph minor theory and the theory of topological embeddings. Its significance is further illustrated by applications to the disjoint paths problem [31], Sachs’ linkless embedding conjecture [32], and Courcelle’s theorem [23].

Moreover, tree-decompositions have proved very useful for investigating compositional structures. They serve as an efficient topological tool for tackling algorithmic graph problems and have been used to improve the efficiency of dynamic programming. For example, with the aid of tree-decompositions, several algorithmic problems that are NP-hard on arbitrary graphs may be solved efficiently by dynamic programming on graphs of bounded tree-width. A concrete example is the problem of finding a maximum independent set in graphs of bounded tree-width.

To acknowledge the benefit of structured decompositions (see Definition 12) we make the notion of ordinary tree-decompositions precise.

Definition 9 A tree-decomposition of a graph G is a pair $( T , ( V _ { t } ) _ { t \in \mathrm { O b } ( T ) } )$ consisting of a tree T and a family of vertex sets $V _ { t } \subseteq V ( G )$ indexed by the vertices t of T and satisfying the following two conditions:

(T1) $G = \cup _ { t \in \mathrm { O b } ( T ) } G [ V _ { t } ]$ where $G [ V _ { t } ]$ denotes the subgraph of G induced by $V _ { t }$

(T2) For each vertex v of $G ,$ the induced subgraph $T _ { \nu } : = T [ \{ t \in T | \nu \in V _ { t } \} ] \subseteq T$ is connected.

We call T the decomposition tree or the model of the decomposition. The subsets $V _ { t } \subseteq V ( G )$ are called the bags, and the induced subgraphs $G [ V _ { t } ] \subseteq G$ are the corresponding parts.

Intuitively, a tree-decomposition reveals the global structure of a graph whenever that structure is tree-like. An example of a tree-decomposition is shown in Figure 3.

![](images/c4ae73d49a863016cfb43b32b5453dce754007bf4618e775a3b2cd6bf52e79cc.jpg)  
Fig. 3: A tree-decomposition of a graph (on the left-hand side) and its decomposition tree (on the right-hand side).

## Tree-width.

Tree-decompositions also provide a convenient tool for constructively computing the treewidth of a graph, as explained below. Originally introduced by Bertele and Brioschi, the\` notion of tree-width has become indispensable in mathematics and computer science. For instance, [33] provides a revealing overview of major applications of tree-width and more general notions of graph width that have been developed from the classical tree-width parameter to capture a broader range of contexts.

The tree-width of a graph G measures the extent to which the structure of G resembles that of a tree. Indeed, the tree-like structure of the graph is reflected in the parts of its decomposition: the smaller these parts are, the more tree-like G is.

Definition 10 The width of a tree-decomposition $( T , ( V _ { t } ) _ { t \in \mathrm { O b } ( T ) } )$ is the maximum order of its bags minus one, $\mathrm { m a x } _ { t \in V ( T ) } | V _ { t } | - 1$ . The tree-width tw(G) of a graph G is the minimum width of its treedecompositions,

$$
\mathrm { t w } ( G ) : = \operatorname* { m i n } _ { \left( T , \left( V _ { t } \right) _ { t \in \mathrm { O b } ( T ) } \right) t \in V ( T ) } | V _ { t } | - 1
$$

where the minimum is taken over all tree-decompositions of G.

The reason for the “minus one” in the definition of the width of a tree-decomposition is to ensure that the tree-width of any tree is one, as expected from a reasonable measure of structural tree-likeness.

## Category theoretical generalization.

To provide a general category-theoretical framework for tree-decompositions and extend their scope of application, the notion of structured decompositions and, more specifically, spined structured decomposition categories (or simply spined sd-categories) was introduced in [21]. These provide a unified axiomatic setting for studying tree-width and its generalizations. The framework recovers several notions of graph width, including ordinary tree-width, complemented tree-width, tree independence number, hypergraph tree-width, and layered tree-width. Thus, spined sd-categories capture the classical notion of tree-width together with many of its variants. We define them explicitly in Definition 12.

## 3.2.2 Temporalization of spined sd-categories

## Needfor temporalization.

Structured decompositions are concerned only with static graphs, meaning graphs in the traditional sense, consisting merely of a set of vertices together with a set of edges connecting them. When applying the existing concepts and results to time-varying data, however, this conventional notion of a graph is no longer sufficient. Many systems of interest evolve over time, and traditional graph-theoretical techniques ignore this temporal dimension. To capture such evolution, a variety of notions of temporal graphs have been introduced. These differ in how they represent temporal information and the evolution of the underlying graph. For an overview of the diversity of existing approaches, we refer to [34–39].

Motivated by the increasing use of temporal graphs to model time-dependent data, we develop a notion of spined sd-categories for time-varying graphs and, more generally, timevarying structures. This is the main objective of the remainder of this section, which is based on [16].

## Combining spined sd-categories and narratives.

To temporalize spined sd-categories, we reinterpret the theory of structured decompositions developed in [21] within the framework of persistent and cumulative narratives developed in [15]. This yields a time-dependent generalization of spined sd-categories.

## Structured decompositions.

Definition 11 Let J be a graph. Its barycentric subdivision is the category R J obtained from J by replacing each vertex with an object and each edge e, with endpoints v and w, by an object e together with morphisms $e _ { \nu } : e \to \nu$ and $e _ { w } : e \to w ,$ that is, a span $\nu \longleftrightarrow e \xrightarrow { e _ { w } } w$

Example 8 The barycentric subdivision of the triangle $K ^ { 3 }$ depicted on the left is the category visualized on the right.

![](images/49161453af986a81d35852b10ac325c7b28b9959f2112e5af4d1095d0ed4f652.jpg)

Definition 12 Let J be a graph, and let D be a category. A J-structured decomposition in D is a functor of shape d : $\scriptstyle \int J \to \mathsf { D }$ with the property that for every morphism $k \colon x  y$ in $\textstyle \int J ,$ its image $d ( k ) \colon d ( x ) $ $d ( y )$ under d is a monomorphism in D.

## Relevance of structured decompositions and schematic visualizations.

We illustrate the breadth of structured decompositions by discussing four examples from different mathematical domains, accompanied by schematic visualizations.

(1) Tree-decompositions. As already mentioned, the development of structured decompositions was originally inspired by tree-decompositions in graph theory, making them perhaps the most prominent example. These combinatorial objects can be described as tree-shaped structured decompositions with values in the category Grph of graphs, that is, functors of the form R T  Grph for some tree T. Replacing T with an arbitrary graph leads to the more general notion of a graph-decomposition, studied, for example, in [40, 41].

The notion of graph-decompositions moreover gives rise to a wide variety of combinatorial width parameters measuring the structural resemblance of a graph to a given graph model, such as a tree, a path, or a cycle. Examples include the classical tree-width together with many of its variants, as discussed in [21, Chapter 3].

![](images/610fcfb4f7ae89f59d94c97c5a9d72830f07c6231fdb7983b04fc96d5151a771.jpg)  
Fig. 4: The cycle $C ^ { 5 }$ , the category $\textstyle \int C ^ { 5 }$ and a $C ^ { 5 }$ -shaped structured decomposition of graphs.

Figure 4 shows a structured decomposition in the category Grph of graphs shaped by the cycle $C ^ { 5 }$ of length five, together with the cycle itself and its barycentric subdivision $\int \bar { C ^ { 5 } }$ . The functor d sends each object $x _ { i }$ of $\dot { f C ^ { 5 } }$ corresponding to a vertex $x _ { i }$ of the graph $C ^ { 5 }$ to a complete graph.

(2) Hybrid dynamical systems. In his thesis [42], Ames establishes a category-theoretical framework for the study of hybrid dynamical systems, that is, dynamical systems with both continuous and discrete components. The thesis introduces the notion of a hybrid object in a category $\mathsf { C } ,$ defined as a functor from a so-called D-category to $\mathsf { C } ,$ which may be viewed as a special case of a structured decomposition.

To illustrate this concept, consider a structured decomposition in the category Man of topological manifolds and continuous maps, shaped by the path $P ^ { 3 }$ of length three, as shown in Figure 5. We observe that the map $d ( f _ { 1 } ) \colon S ^ { 1 } \to \Sigma _ { 1 }$ , from the topological unit circle $S ^ { 1 }$ to the torus $\Sigma _ { 1 }$ (the closed orientable surface of genus one), is homotopic to a constant map. Likewise, the map $d ( f _ { 2 } ) \colon S ^ { 1 } \to \Sigma _ { 2 }$ , where $\Sigma _ { 2 }$ denotes the closed orientable surface of genus two, is homotopic to the composite $S ^ { 1 } \xrightarrow { d ( g _ { 3 } ) } \Sigma _ { 1 } \hookrightarrow \Sigma _ { 2 }$

(3) Graphs of groups. The fundamental objects of Bass–Serre theory, namely graphs of groups, are precisely structured decompositions valued in the category of groups and group homomorphisms [43–45]. Restricting to structured decompositions shaped by trees recovers the setting of Bass–Serre theory, in which such decompositions correspond to well-behaved group actions on trees.

As an example, we consider the fundamental groups of the manifolds in the image of the functor $d \colon \int P ^ { 3 } \to$ Man from the previous example, thereby obtaining a $P ^ { 3 }$ -shaped structured decomposition in the category Grp of groups, shown in Figure 6. More precisely, this is the composition of the previous structured decomposition in Man with the fundamental group functor π<sub>1</sub> : $\mathsf { M a n } \to \mathsf { G r p }$ . In the upper part of the figure, we illustrate closed curves in $S ^ { 1 } , \Sigma _ { 1 }$ , and $\Sigma _ { 2 }$ generating the corresponding fundamental groups, while the lower part visualizes the resulting structured decomposition of fundamental groups.

![](images/d3b40aec76a817a584f7c4a5b1b687a3d49b73881ed55a44d9febb6bf9060053.jpg)  
Fig. 5: The path $P ^ { 3 } .$ , the category $\int P ^ { 3 }$ and a $P ^ { 3 } .$ -shaped structured decomposition of manifolds.

![](images/fee4117e6cfc7eae86d67d145c0627e1d4e1dc974b99b234b853bab844c144e1.jpg)  
Fig. 6: A $P ^ { 3 }$ -shaped structured decomposition of groups (lower part), obtained by computing the fundamental groups of the manifolds in Figure 5 with the chosen generators (upper part).

As expected, the homotopic maps $d ( f _ { 1 } ) \simeq \mathrm { c o n s t } \colon S ^ { 1 } \to \Sigma _ { 1 }$ induce the same group homomorphism $\pi _ { 1 } ( S ^ { 1 } ) \to \pi _ { 1 } ( \Sigma _ { 1 } )$ , namely the trivial homomorphism. Likewise, the homotopic maps $d ( f _ { 2 } ) \simeq \mathrm { i n c l } \circ d ( g _ { 3 } ) \colon S ^ { 1 }  \Sigma _ { 2 }$ induce the same homomorphism $\pi _ { 1 } ( S ^ { 1 } ) \to \pi _ { 1 } ( \Sigma _ { 2 } )$ , name $\operatorname { l y } x \mapsto a _ { 1 }$

(4) Cellular sheaves. Cellular sheaves encode local-to-global interactions in systems built from graphs, simplicial complexes, and cell complexes [46]. Later in this chapter, we use them to model multi-agent systems with time-varying communication topologies, illustrating an application of narratives to control theory developed in [47]. From the perspective developed here, cellular sheaves arise naturally as the dual notion to structured decompositions: they are structured co-decompositions valued in the category Vect of real vector spaces and linear maps. We nevertheless retain the terminology of cellular sheaf theory in order to align with the existing literature on control theory and multi-agent systems.

## Spined sd-categories.

Definition 13 Let D be a category, and let $\mathcal { G }$ be a class of graphs including the trivial graph with exactly one vertex and no edge. The pair $( \mathsf { D } , \mathcal { G } )$ is said to be a structured decomposition category, or simply, an sd-category, iff every structured decomposition d : $\textstyle \int J \to \mathsf { D }$ in D with $J \in { \mathcal { G } }$ admits a colimit. We call G the index class of $( \mathsf { D } , \mathcal { G } )$

Definition 14 A spine on a category D is a family $\Omega = ( \Omega _ { n } ) _ { n > 0 }$ of increasing subcategories

$$
\Omega _ { 0 } \subseteq \Omega _ { 1 } \subseteq \Omega _ { 2 } \subseteq . . .
$$

with the following additional properties:

• The union $\textstyle \bigcup \Omega : = \bigcup _ { n > 0 } \Omega _ { n }$ is closed under isomorphic objects in D, and for any $n \geq 0 , \Omega _ { n }$ closed under isomorphic objects as a subcategory of SΩ.

• For every object $X \in \mathrm { O b } ( \mathsf { D } )$ , there exists an object $W \in { \mathrm { O b } } ( \Omega _ { n } )$ for some integer $n \geq 0$ together with a monomorphism $W \longmapsto X$ in D.

• Given an integer $n \geq 1$ and objects W of $\Omega _ { n }$ and $\widetilde { W }$ of $\Omega _ { n - 1 }$ along with a monomorphism $W \longmapsto { \widetilde { W } }$ in D, then W is already contained in $\Omega _ { n - 1 } .$

A spined sd-category is a triple $( \mathsf { D } , \mathsf { G } , \Omega )$ where $( \mathsf { D } , \mathcal { G } )$ is an sd-category and Ω is a spine on D.

## Combining persistent narratives and spined sd-categories.

We begin with the persistent perspective; the cumulative perspective and the relation between the two are discussed in Section 4. Let T be a finite discrete time category, let ${ \mathsf { S } } \subseteq { \mathsf { T } }$ be a subjoin-semilattice, and let $\tau { : } \mathsf { S } \hookrightarrow \mathsf { T }$ denote the inclusion functor. Furthermore, let $( \mathsf { D } , \mathsf { G } , \Omega )$ be a spined sd-category satisfying the following additional condition.

(T1) The category D and its subcategories $\Omega _ { n } \subseteq \mathsf { D }$ admit all pullbacks, the inclusion functors $\mathfrak { 1 } _ { n } \colon \Omega _ { n } \hookrightarrow \mathsf { D }$ preserve pullbacks, and each $\Omega _ { n }$ is full in D.

In particular, post-composition with $\mathfrak { 1 } _ { n }$ provides a well-defined functor

$$
\mathsf { P e } ( \mathsf { T } , \mathsf { \mathsf { 1 } } _ { n } ) \colon \mathsf { P e } ( \mathsf { T } , \Omega _ { n } ) \to \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) ,
$$

which we call the covariant change-of-base functor. Similarly, we have a change-oftemporal-resolution functor defined by pre-composition with the opposite $\tau ^ { \mathrm { { o p } } } \colon { \mathsf { S } } ^ { \mathrm { { o p } } } \hookrightarrow \mathsf { T } ^ { \mathrm { { o p } } }$ of τ:

$$
\mathsf { P e } ( \tau , \mathsf { D } ) \colon \mathsf { P e } ( { \mathsf { T } } , \mathsf { D } ) \to \mathsf { P e } ( { \mathsf { S } } , \mathsf { D } ) .
$$

The same holds when replacing the category T by S and D by $\Omega _ { n } ,$ for $n \geq 0$

For each $n \geq 0$ , we consider the following pullback square of categories and functors:

$$
\begin{array} { r l } & { \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) \times _ { \mathsf { P e } ( \mathsf { S } , \mathsf { D } ) } \mathsf { P e } ( \mathsf { S } , \Omega _ { n } ) \xrightarrow [ \mathsf { P e } ( \mathsf { S } , \Omega _ { n } ) ] { \quad \mathsf { P e } ( \mathsf { S } , \Omega _ { n } ) } } \\ & { \qquad \quad \mathbb { \pi } _ { n } \downarrow \qquad } \\ & { \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) \xrightarrow [ \mathsf { P e } ( \tau , \mathsf { D } ) ] { \quad \mathsf { P e } ( \mathsf { \tau } , \mathsf { D } ) \quad } \mathsf { P e } ( \mathsf { S } , \mathsf { D } ) . } \end{array}
$$

We define $\widehat { \Omega } _ { n }$ to be the image of the pullback category under $\pi _ { n } .$ , so this is the subcategory

$$
\widehat { \Omega } _ { n } : = \pi _ { n } ( \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) \times _ { \mathsf { P e } ( \mathsf { S } , \mathsf { D } ) } \mathsf { P e } ( \mathsf { S } , \Omega _ { n } ) ) \subseteq \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) .
$$

Lemma 3.7 For every integer n $\geq 0 , \widehat { \Omega } _ { n }$ is the full subcategory of $\mathsf { P e } ( \mathsf { T } , \mathsf { D } )$ whose objects are those persistent narratives $F \colon \mathsf { T } ^ { \mathrm { o p } }  \mathsf { D }$ whose value $F ( [ a , b ] )$ at any time interval $[ a , b ]$ in S is contained in $\Omega _ { n }$

Theorem 3.8 Let again T be a finite discrete time category, let $\tau \colon \mathsf { S } \hookrightarrow$ T be the inclusion of a subjoin-semilattice ${ \mathsf { S } } \subseteq { \mathsf { T } } _ { : }$ , and let $( \mathsf { D } , \mathsf { G } , \Omega )$ be a spined sd-category satisfying $( T I ) .$ . Furthermore, suppose that thefollowing conditions hold.

(T2) The category D is cocomplete, that is, it admits all small colimits.

(T3) The inclusion functor $\mathsf { P e } ( \mathsf { T } , \mathsf { D } ) \hookrightarrow [ \mathsf { T } ^ { \mathrm { o p } } , \mathsf { D } ]$ admits a left adjoint $\mathfrak { S } \colon [ \mathsf { T } ^ { \mathrm { o p } } , \mathsf { D } ]  \mathsf { P e } ( \mathsf { T } , \mathsf { D } )$ whose restriction to $\mathsf { P e } ( \mathsf { T } , \mathsf { D } )$ is the identityfunctor.

Then the pair $( \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) , \mathsf { \boldsymbol { g } } )$ is an sd-category.

Since S is both a left adjoint and a retraction of the inclusion functor, it can be regarded as a sheafification functor. In Example 9, where D is the category Grph of graphs, S is induced by the usual sheafification functor for set-valued presheaves.

By adding one final axiom, we arrive at the notion of a temporalized spined sd-category.

Theorem 3.9 As before, let T be a finite discrete time category, and let $\tau \colon \mathsf { S } \hookrightarrow$ T be the inclusion of a sub-join-semilattice ${ \mathsf { S } } \subseteq { \mathsf { T } } .$ . Let $( \mathsf { D } , \mathsf { G } , \Omega )$ be a spined sd-category that satisfies the axioms $( T I ) , ( T 2 )$ and (T3) as well as

(T4) The category D admits pushout squares along monomorphisms and these are also pullback squares. Moreover, monomorphisms are stable under pushouts.

Then the subcategories ${ \widehat { \Omega } } _ { n } \subseteq \mathsf { P e } ( \mathsf { T } , \mathsf { D } )$ with $n \geq 0$ define a spine $\widehat \Omega$ on $\mathsf { P e } ( \mathsf { T } , \mathsf { D } )$ , thereby exhibiting the sd-category $( \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) , \mathsf { \boldsymbol { g } } )$ of Theorem 3.8 as a spined sd-category $( \mathsf { P e } ( \mathsf { T } , \mathsf { D } ) , \mathscr { G } , \widehat { \Omega } )$

Theorem 3.9 allows measures of complexity to be lifted from the static to the time-varying setting. A complete proof is given in [16].

## 3.2.3 Temporalizing tree-width: examples

In ordinary graph theory, the notion of tree-width is typically defined by means of treedecompositions, as explained in Paragraph 3.2.1. The category-theoretical framework of spined sd-categories [21] extends this concept to arbitrary structured decompositions. More precisely, given a spined sd-category $\Gamma = ( \mathsf { D } , \mathscr { G } , \Omega )$ , each object of D is assigned a $\Gamma { - } s i z e \left[ 2 1 \right.$ Definition 2.5.4]. We restrict attention to spined sd-categories in which the subcategories $\Omega _ { n } \subseteq \mathsf { D }$ are full, in which case the definition simplifies as follows.

Definition 15 The Γ-size of an object $X \in \mathrm { O b } ( \mathsf { D } )$ is the minimum non-negative integer n for which there exists an object $W \in \mathrm { O b } ( \Omega _ { n } )$ together with a monomorphism $X \longmapsto W$ in D. The resulting map $s _ { \Gamma } \colon \mathrm { O b } ( \mathsf { D } ) \to  { \mathbb { N } } _ { 0 }$ , is called the sizefunction of Γ.

The size function allows the notions of tree-width for tree-decompositions and graphs to be extended to width notions for structured decompositions and objects of arbitrary spined sd-categories, respectively.

Definition 16 The width $w _ { \Gamma } ( d )$ of a structured decomposition d : $\textstyle \int J \to \mathsf { D }$ with $J \in { \mathcal { G } }$ is the maximum Γ-size of its bags $d ( \nu )$ minus one:

$$
w _ { \Gamma } ( d ) : = \operatorname* { m a x } _ { \nu \in V ( J ) } s _ { \Gamma } ( d ( \nu ) ) - 1 .
$$

Now, the Γ-width $w _ { \Gamma } ( X )$ of an object X of D is defined as the minimum Γ-widths of all structured decompositions d : $\textstyle \int J \to \mathsf { D }$ with $J \in { \mathcal { G } }$ whose colimit is isomorphic to $X { \mathrm { : } }$

$$
w _ { \Gamma } ( X ) : = \operatorname* { m i n } _ { \substack { d : \int J  \mathsf { D } \mathrm { ~ w i t h ~ } } } w _ { \Gamma } ( d ) .
$$

Section 3 of [21] shows that many classical graph width parameters are recovered as instances of this general notion by choosing appropriate spined sd-categories. This justifies spined sd-categories as a unified framework for generalized tree-widths.

To justify the temporalization of spined sd-categories and the resulting notions of timevarying structured decompositions and Γ-width, we consider three classical graph width parameters. For each, we exhibit a spined sd-category whose temporalization via Theorem 3.9 yields a natural time-varying analogue of the corresponding width notion. Specifically, we consider ordinary tree-width (Definition 10), complemented tree-width, and the tree independence number.

Example 9 (Ordinary tree-width). Proposition 3.1.1 in [21] shows that the category Grph together with the class of all trees and the full subcategories $\Omega _ { n } \subseteq \mathsf { G }$ rph containing the complete graphs of order at least n is a spined sd-category whose associated notion of width is the ordinary tree-width of a graph. To apply our temporalization method of Theorem 3.9, we need to replace Grph by the category of reflexive graphs, meaning those graphs with precisely one loop at each vertex.

Proposition 3.10 For any finite discrete time category T and any sub-join-semilattice ${ \mathsf { S } } \subseteq { \mathsf { T } }$ , Γ induces a temporalized spined sd-category $\widehat { \Gamma } : = ( \mathsf { P e } ( \mathsf { T } , \mathsf { G r p h } _ { \mathsf { r e f l } } )$ , trees , Ωb), whose size function takes any persistent narrative $X \colon { \mathsf { T } } ^ { \mathrm { o p } } \to { \mathsf { G r p h } } _ { \mathsf { r e f l } }$ to the maximum order among the graphs $X ( s )$ with $s \in \mathrm { O b } ( \mathsf { S } )$ ,

$$
s _ { \widehat { \Gamma } } ( X ) = \operatorname* { m a x } _ { s \in \mathrm { O b } ( \mathsf { S } ) } | X ( s ) | .
$$

The Γb-size of a structured decomposition d : $\begin{array} { r } { \int J \to \mathsf { P e } ( \mathsf { T } , \mathsf { G r p h } _ { \mathsf { r e f l } } ) } \end{array}$ for any tree J can be shown to be

$$
w _ { \widehat { \Gamma } } ( d ) = \operatorname* { m a x } _ { s \in \mathrm { O b } ( { \mathsf { S } } ) } w _ { \Gamma } ( d _ { s } )
$$

where $d _ { s } \colon \ \int J \to \mathsf { G r p h } _ { \mathsf { r e f l } }$ is obtained from d by evaluation at $s \in \operatorname { O b } ( \mathsf { S } )$ . Since w<sub>Γ</sub> is the ordinary tree-width of a graph, we thus recover maximum tree-width, which is a well-known generalization of tree-width for temporal graphs.

Example 10 (Complemented tree-width). The complemented tree-width of a graph G is the ordinary tree-width of its complement ${ \overline { { G } } } ,$ the graph with the same vertex set as G and with an edge between two vertices if and only if these vertices are non-adjacent in G. Let Grph be the category whose objects are the undirected graphs and whose morphisms from G to H are graph morphisms of complements $\overline { { G } }  \overline { { H } } . \mathrm { B y } [ 2 1$ , Proposition $3 . 3 . 4 ] , \Gamma : = ( \overline { { \mathsf { G r p h } } } , \{ \mathrm { t r e e s } \} , \Omega )$ is a spined sd-category where $\Omega _ { n }$ is the full subcategory of Grph whose objects are all edgeless graphs of order at most n. Its associated width notion is complemented tree-width.

Proposition 3.11 For any sub-join-semilattice S of a finite discrete time category T, Γ induces a time-varying spined sd-category $( \mathsf { P e } ( \mathsf { T } , \overline { { \mathsf { G r p h } } } ) , \{ \mathrm { t r e e s } \} , \widehat { \Omega } )$ . Its width notion is given by the maximum complemented tree-width over ${ \mathsf { S } } .$

Modifying the spine Ω on Grph appropriately will give rise to the tree independence number of a graph, discussed in the next example.

Example 11 (Tree independence number). The independence number of a graph is the maximum number of pairwise non-adjacent vertices. The tree independence number of a tree decomposition is the maximum independence number of its bags minus one, and the tree independence number of a graph is the minimum one among all its tree decompositions. For any $n \geq 0 ,$ , let $\Omega _ { n } ^ { \alpha }$ denote the full subcategory of Grph whose objects are the graphs of independence number $\alpha ( G ) \leq n$ . Then by [21, Proposition 3.4.5], they define a spine $\Omega ^ { \alpha }$ on Grph making the triple (Grph, trees ,Ω<sup>α</sup>) to a spined sd-category, whose width notion is the tree independence number.

Proposition 3.12 Given T and S as usual, the above spined sd-category induces a temporalization $( \mathsf { P e } ( \mathsf { T } , \overline { { \mathsf { G r p h } } } ) , \{ \mathrm { t r e e s } \} , \widehat { \Omega } ^ { \alpha } )$ . Its size function is given by the maximum independence number of graphs, so its associated width notion recovers the maximum tree independence number over S.

## 3.3 Vignette 3: Temporal Cellular Sheaves: Modelling Multi-agent Systems with Switching Topologies

This vignette outlines an ongoing research program developing category-theoretic and sheaftheoretic methods for control, with current applications to control barrier functions and multiagent systems [47, 48]. Building on these ideas, we focus here on multi-agent systems with switching communication topologies [47]. Such systems arise in applications ranging from robotic swarms and autonomous vehicles to sensor networks and distributed optimization, where the communication graph changes over time due to mobility, communication failures, or environmental constraints [5, 6, 49, 50].

A common approach represents the communication topology by a graph whose vertices correspond to agents and whose edges represent communication links [5, 49]. This graph captures the connectivity of the network and provides the underlying structure for many distributed control algorithms. The communication graph, however, captures only who communicates with whom. The dynamics of the agents, the information they exchange, the sensing mechanisms, and the transformations performed along communication links must be modeled separately. Because these additional structures are usually introduced in an application-specific manner, it is difficult to develop a unified mathematical framework encompassing heterogeneous multi-agent systems.

## 3.3.1 Cellular sheaves for multi-agent systems

Cellular sheaves offer a mathematical model for the additional structures required in heterogeneous multi-agent systems [51, 52]. They assign (potentially different) vector spaces to agents and communication links, together with linear maps describing how information is measured, communicated, or transformed locally. This construction naturally accommodates heterogeneous state spaces, sensing mechanisms, communication protocols, and local information-processing rules. In this way, the communication graph specifies who communicates with whom, while the cellular sheaf specifies what information is associated with each agent and communication link, and how that information is transformed.

Categorically, a cellular sheaf is simply a functor

$$
\begin{array} { r } { \mathcal { G } : \mathsf { i n c } ( \mathsf { G } ) \longrightarrow \mathsf { V e c t } , } \end{array}
$$

where $\mathfrak { i n c } ( \mathsf { G } )$ denotes the incidence category of the communication (directed) graph G. Its objects are the faces of $G$ (vertices and edges), and its morphisms are precisely the incidence relations: for every edge $\boldsymbol { e } = \left( u , \nu \right)$ , there are morphisms u $; \xrightarrow { t _ { e } } e$ and $\nu \xrightarrow { h _ { e } } e .$ . Figure 7 illustrates this construction.

![](images/19544248294063ac0f840d422f3cd05a9128bd7f3d7be4e6e723c226e393b7be.jpg)  
Fig. 7: From left to right: the directed communication graph $G ,$ whose vertices represent agents and whose edges $e _ { 1 } , e _ { 2 }$ and $e _ { 3 }$ represent directed communication links; its incidence category $\mathsf { i n c } ( \mathsf { G } )$ , whose objects are the faces of $G ,$ namely its vertices and edges, and whose morphisms record the incidence relations; and the diagram determined by the cellular sheaf $\mathcal { G } : \mathsf { i n c } ( \mathsf { G } )  \mathsf { V e c t }$ . In this example, $\mathcal { G }$ assigns the vector space $\mathbb { R } ^ { 2 }$ to every object of inc(G), equivalently to every face of $G ,$ together with linear maps associated with the incidence morphisms. These maps encode the local sensing, communication, and information-processing relations of the network.

Remark 5 (Cellular sheaves as structured co-decompositions) Notice that cellular sheaves are instances of the structured co-decompositions considered in Section 3.2 since the incidence category $\mathfrak { i n c } ( \mathsf { G } )$ of a graph G is isomorphic to $( \int G ) ^ { o p }$ . Thus, a cellular sheaf $\mathcal { G } \colon \mathsf { i n c } ( \mathsf { G } )  \mathsf { V e c t }$ is equivalently a contravariant functor from ${ \hat { \boldsymbol g } } \colon ( \int G ) ^ { o p } \to$ Vect, and hence a G-shaped structured co-decomposition of vector spaces. Equivalently, after taking opposites, it may be regarded as a functor $\begin{array} { r } { { \mathcal G } ^ { o p } \colon \int G  \mathsf { V e c t } ^ { o p } } \end{array}$ . When the cellular sheaf maps are epimorphisms in Vect, they become monomorphisms in ${ \mathsf { V e c t } } ^ { o p }$ , so that $\boldsymbol { G } ^ { o p }$ is a structured decomposition in the sense used earlier in this chapter. To match the terminology of control theory, multi-agent systems, and the cellular-sheaf literature, we refer to these objects as cellular sheaves throughout this section.

The functorial viewpoint is essential because it guarantees that the local models assigned to agents and communication links are compatible with the communication topology. Rather than specifying local state spaces, sensing maps, and communication rules independently, functoriality requires them to respect the incidence relations of the graph. Consequently, the resulting cellular sheaf represents a globally consistent information-processing architecture for the multi-agent system.

## 3.3.2 Category of cellular sheaves

To model systems whose communication topology changes over time, we must also specify how one cellular sheaf transforms into another. This requires a suitable notion of morphism between cellular sheaves, capturing both the evolution of the underlying communication graph and the induced transformation of the associated local state data. This leads naturally to the category of cellular sheaves.

Definition 17 (Category of cellular sheaves) The category CellSh is defined as follows.

• Its objects are cellular sheaves $\mathcal { G } : \mathsf { i n c } ( \mathsf { G } ) \longrightarrow \mathsf { V e c t }$ , where G is a finite directed graph.

• A morphism $( { \mathcal { T } } , \alpha ) : { \mathcal { G } } \to { \mathcal { H } }$ consists of a functor $\mathcal { T } : \mathsf { i n c } ( \mathsf { G } ) \to \mathsf { i n c } ( \mathsf { H } )$ and a natural transformation $\mathfrak { a } : \mathcal { G } \Rightarrow \mathcal { H } \circ \mathcal { T }$ , as shown below.

![](images/14e441a1be4b840c37f4251abc603529aa49ebc4bdef4ae702d0102d47ede6ba.jpg)

We first examine the functor $\mathcal { T } : \mathsf { i n c } ( \mathsf { G } ) \to \mathsf { i n c } ( \mathsf { H } )$ appearing in a morphism $( \mathcal { T } , \mathfrak { a } ) : \mathcal { G } \to$ ${ \mathcal { H } } .$ . Recall that the objects of inc(G) are the agents and communication links of $G ,$ while its nonidentity morphisms encode incidences between them. Thus, $\mathcal { T }$ specifies how the communication topology G is represented inside, or transformed into, the communication topology H. Because $\mathcal { T }$ is functorial, it preserves incidence relations: whenever an agent is incident to a communication link in $G ,$ its image must be incident to the image of that link in H.

The following examples illustrate increasingly sophisticated ways in which a functor between incidence categories may arise in applications, beginning with subsystem inclusions, progressing to a coarse-graining of agents into teams, and culminating in switching communication topologies.

Example 12 (A subsystem of a system) Let G be a subgraph of H. The inclusions of vertices and edges determine a functor $I : \mathsf { i n c } ( \mathsf { G } ) \hookrightarrow \mathsf { i n c } ( \mathsf { H } )$ that sends every agent and communication link of G to the corresponding cell of H. Functoriality follows because incidences in G remain incidences in $H .$

Example 13 (Aggregation of agents into teams) Hierarchical control often requires reasoning simultaneously at different levels of abstraction [53–55]. At the lower level, individual agents communicate through a detailed network, while at the higher level, groups of agents are treated as teams interacting through a coarser communication topology. A functor between the corresponding incidence categories relates these two descriptions.

Consider the path graph $G = P ^ { 4 } = 1 - 2 - 3 - 4 .$ , representing four communicating agents. Suppose that agents 1 and 2 form team A, while agents 3 and 4 form team B. The team-level communication graph H therefore consists of two vertices, A and B, connected by a single edge AB. The aggregation is described by a functor

$$
\mathcal { T } : \mathsf { i n c } ( \mathsf { G } ) \longrightarrow \mathsf { i n c } ( \mathsf { H } )
$$

defined by

$$
\mathcal { T } ( 1 ) = \mathcal { T } ( 2 ) = A , \qquad \mathcal { T } ( 3 ) = \mathcal { T } ( 4 ) = B ,
$$

and

$$
\mathcal { T } ( 1 2 ) = A , \qquad \mathcal { T } ( 2 3 ) = A B , \qquad \mathcal { T } ( 3 4 ) = B .
$$

![](images/80a3312c11e2c70acd0dfecccbfac0a1cbd4059d35ea1367bea032dc0825e42a.jpg)

The color coding displays the aggregation. The objects 1, 2, and 12 share the color of team A, while 3, 4, and 34 share the color of team B. Accordingly, the intra-team links 12 and 34 are mapped to the team objects A and B, respectively. Their incidence morphisms are therefore mapped to the corresponding identity morphisms, shown as loops in the team-level category. This expresses that communication within each team is black-boxed in the coarse description.

By contrast, the link 23 and its incidence morphisms share the color of the edge AB and its incidence morphisms. Thus, the communication between the two teams remains visible as a nontrivial edge at the team level.

Example 14 (Switching topologies) Consider a multi-agent system whose communication topology changes over time. Let $G _ { t _ { i } }$ and $G _ { t _ { k } }$ denote the communication graphs at two time instants $t _ { i } < t _ { k }$ . Both may be viewed as communication subgraphs of a larger graph G describing every communication link that appears during the entire time horizon. The corresponding inclusion functors

$$
\mathsf { i n c } ( \mathsf { G } _ { \mathsf { t } _ { i } } ) \hookrightarrow \mathsf { i n c } ( \mathsf { G } ) \longleftrightarrow \mathsf { i n c } ( \mathsf { G } _ { \mathsf { t } _ { k } } ) .
$$

admit a pullback

$$
\mathsf { i n c } ( \mathsf { G } _ { \mathsf { t } _ { i } } ) \longleftrightarrow \mathsf { i n c } ( \mathsf { G } _ { \mathsf { t } _ { i } } ) \times _ { \mathsf { i n c } ( \mathsf { G } ) } \mathsf { i n c } ( \mathsf { G } _ { \mathsf { t } _ { k } } ) \longleftrightarrow \mathsf { i n c } ( \mathsf { G } _ { \mathsf { t } _ { k } } ) ,
$$

whose apex identifies precisely the agents and communication links common to both topologies. The resulting span therefore represents the communication subsystem that persists across the transition from $G _ { t _ { i } }$ to $G _ { t _ { k } }$

The second component of a morphism of cellular sheaves is given by a natural transformation α : ${ \mathcal { G } } \Rightarrow { \mathcal { H } } \circ { \mathcal { T } }$ . For every incidence morphism m $: x  y$ in $\mathfrak { i n c } ( \mathsf { G } )$ , naturality requires the commutativity of

$$
\begin{array} { r } { \begin{array} { c c c } { \mathcal { G } ( x ) \xrightarrow { \mathcal { G } ( m ) } } & { \mathcal { G } ( y ) } \\ { \mathrm { ~ } } \\ { \mathcal { H } ( \mathcal { T } ( x ) ) } & { \overline { { \mathcal { H } ( \mathcal { T } ( m ) ) } } \mathcal { H } ( \mathcal { T } ( y ) ) . } \end{array} } \end{array}
$$

Thus, performing the local computation prescribed by $\mathcal { G }$ and then communicating the resulting information via α to H yields the same result as first communicating the local data from $\mathcal { G }$ to H and then perform the corresponding local computation in ${ \mathcal { H } } .$ In this sense, a morphism $( { \mathcal { T } } , \alpha ) : { \mathcal { G } } \to { \mathcal { H } }$ makes the local information-processing rules of the two systems compatible with the information processing protocol within each system.

Example 15 (Persistent information across a topology change) Consider the switching-topology setting of Example 14. We use this situation to illustrate the form of morphisms in the category of cellular sheaves. Let

![](images/8b155215e69191ff14c9d9490a68ba4b7ecc7b805a137f674aca908688041afd.jpg)

where $\it { G } _ { t _ { i } } ^ { t _ { i } } , \it { G } _ { t _ { i } } ^ { t _ { k } }$ , and $\mathcal { G } _ { t _ { k } } ^ { t _ { k } }$ are cellular sheaves on the communication topologies inc $( \mathsf { G } _ { \mathtt { t i } } )$ , inc $\left( \mathsf { G } _ { \mathsf { t } _ { \mathrm { i } } } ^ { \mathsf { t } _ { \mathsf { k } } } \right)$ , and $\mathfrak { i n c } ( \mathsf { G } _ { \mathfrak { t } _ { \mathsf { k } } } )$ , respectively. Since the communication topologies and the functors $I _ { i }$ and $I _ { k }$ have already been specified, defining morphisms of cellular sheaves amounts to specifying natural transformations

$$
\alpha _ { i } : { \mathcal { G } } _ { t _ { i } } ^ { t _ { k } } \Rightarrow { \mathcal { G } } _ { t _ { i } } ^ { t _ { i } } \circ I _ { i } , \quad \quad \alpha _ { k } : { \mathcal { G } } _ { t _ { i } } ^ { t _ { k } } \Rightarrow { \mathcal { G } } _ { t _ { k } } ^ { t _ { k } } \circ I _ { k } ,
$$

together with these functors, define the morphisms of cellular sheaves

$$
\begin{array} { r } { \mathcal { G } _ { t _ { i } } ^ { t _ { i } } \underset { } { \xleftarrow { ( I _ { i } , \alpha _ { i } ) } } \mathcal { G } _ { t _ { i } } ^ { t _ { k } } \xrightarrow { ( I _ { k } , \alpha _ { k } ) } \mathcal { G } _ { t _ { k } } ^ { t _ { k } } . } \end{array}
$$

The natural transformations $\alpha _ { i }$ and $\alpha _ { k }$ assign compatible linear maps to every persistent agent and communication link. Consequently, the communication subsystem that survives the topology change is accompanied by a coherent identification of the local state spaces, sensing maps, and communication constraints at both time instants.

## 3.3.3 Temporal cellular sheaves and switching topologies

To model multi-agent systems with switching communication topologies as temporal narratives, we need the category of cellular sheaves to have the required structure. The following lemma, proved in [47], establishes precisely this fact.

Lemma 3.13 The category CellSh admits pullbacks and pushouts.

Consequently, all the constructions developed in Section 3.1 apply with $\mathsf { D } = \mathsf { C e l l S h }$

Definition 18 (Temporal cellular sheaf) A temporal cellular sheaf is a CellSh-valued narrative, that is, a sheaf

$$
\mathcal { F } : \mathsf { T } ^ { o p } \longrightarrow \mathsf { C e l l S h } _ { < } ^ { \flat } .
$$

Thus, each interval $[ a , b ] \in \mathsf { T }$ is assigned a morphism of cellular sheaves $P _ { a } ^ { b } \xrightarrow [ ] { \gamma _ { a } ^ { b } } C _ { a } ^ { b }$ , where $P _ { a } ^ { b }$ and $C _ { a } ^ { b }$ model the persistent and cumulative aspects of the multi-agent system over the interval $[ a , b ]$ , respectively, while $\gamma _ { a } ^ { b }$ relates the two descriptions. The sheaf condition reconstructs the persistent component by pullbacks and the cumulative component by pushouts, so that both the communication topology and the associated sensing and interaction maps evolve coherently through time.

The principal advantage of temporal cellular sheaves is that they model not only the communication topology at each time instant but also the structural relationships between communication topologies across time intervals. Thus, instead of viewing a switching multiagent system as a sequence of independent communication graphs, the narrative records how information persists and accumulates as the communication architecture evolves.

Within this framework, distributed coordination problems such as consensus, formation control, and target tracking can be formulated categorically. The dynamics of the agents evolve on the vector spaces assigned by the cellular sheaves, while the temporal narrative specifies how these local dynamical models are related as the communication topology changes.

Current work investigates the stability theory of temporal cellular sheaves, building on recent sheaf-theoretic approaches to distributed control together with classical Lyapunov methods for switching systems. The objective is to characterize how changes in the communication topology interact with the evolution of distributed state variables, and to establish sufficient conditions under which target-tracking errors remain bounded and converge despite topology switches. This research program is currently being developed in [47].

## 4 Conclusion and future directions

One of the strengths of the narrative framework is that it is largely independent of the nature of the objects evolving through time. Once an appropriate target category has been identified, the same theory immediately yields a coherent framework for modeling temporal phenomena across a wide range of applications. The three research directions presented in this chapter illustrate this flexibility from complementary perspectives: the first investigates the categorical structure of the persistence–accumulation adjunction through the category of narratives and its induced factorization of the adjunction, the second explores temporal analogues of structured decompositions, and the third instantiates the framework in the category of cellular sheaves to model multi-agent systems with switching communication topologies.

## Towards a characterization of thefixedpoints of the adjunction

The factorization of the persistence–accumulation adjunction through the category of narratives introduces a setting in which persistence and accumulation are no longer viewed as separate constructions, but as two complementary descriptions of the same temporal object. The resulting rigidity classification measures the extent to which these descriptions determine one another. Rigid narratives coincide with the canonical completions induced by the adjunction in both directions. Left-rigid narratives preserve their persistent description under the persistence–accumulation round trip, right-rigid narratives preserve their cumulative description, while loose narratives lose information in both directions.

The rigidity classification also opens several directions for future research. A first natural question is to determine which rigidity classes arise in a given ambient category and how they reflect its structural properties. More generally, one may seek intrinsic categorical characterizations of left, right, and rigid narratives, as well as investigate how rigidity behaves under products, limits, colimits, functorial changes of the ambient category, and other categorical constructions. Beyond their intrinsic categorical interest, these questions contribute to a broader understanding of information-preserving changes of perspective, including those arising in data science.

## Structured decompositions of cumulative narratives

By virtue of the tight relation between persistent and cumulative narratives, as revealed in Section 2.2, it is desirable to temporalize structured decompositions and width also from the cumulative point of view. More specifically, under dual assumptions of Theorem 3.8, we may form a pullback square

$$
\begin{array} { r l } { \mathsf { C u } ( \mathsf { T } , \mathsf { D } ) \times _ { \mathsf { C u } ( \mathsf { S } , \mathsf { D } ) } \mathsf { C u } ( \mathsf { S } , \Omega _ { n } ) \longrightarrow \mathsf { C u } ( \mathsf { S } , \Omega _ { n } ) } & { } \\ { \pi _ { n } \Big \downarrow } & { \qquad \downarrow \mathsf { C u } ( \mathsf { S } , \mathsf { u } _ { n } ) } \\ { \mathsf { C u } ( \mathsf { T } , \mathsf { D } ) \xrightarrow [ \mathsf { C u } ( \tau , \mathsf { D } ) ] { \mathsf { C u } ( \mathsf { S } , \mathsf { D } ) } } & { \mathsf { C u } ( \mathsf { S } , \mathsf { D } ) . } \end{array}
$$

We define $\check { \Omega } _ { n } \subseteq \mathsf { C u } ( \mathsf { T } , \mathsf { D } )$ to be the image category of the functor $\pi _ { n }$ . If we dualize the assumptions of Theorem $3 . 9 \AA$ , do we again obtain a spined sd-category of the form $( \mathsf { C u } ( \mathsf { T } , \mathsf { D } ) , \mathscr { G } , \breve { \Omega } ) ?$ How does the proof change?

A second direction is to investigate the relationship between the persistent and cumulative temporalizations of spined sd-categories. As exposed in Theorem 2.1, given any time category T and a category D that is both complete and cocomplete, then there exists a pair of adjoint functors

$$
\begin{array} { r } { \mathsf { P e } ( { \mathsf { T } } , { \mathsf { D } } ) \underset { \mathsf { \tiny \longmapsto } } { \overset { } { \lrcorner } } { \overset { } { \subset } } { \mathsf { u } } ( { \mathsf { T } } , { \mathsf { D } } ) . } \end{array}
$$

It would be interesting to investigate the following questions, assuming the setting of Theorem 3.9 together with the dual assumptions:

• Are the above adjoint functors sd-functors in the sense of [21, Definition 2.7.1]?

• Are they even width-preserving in the sense of [21, Definition 2.7.8]?

• If not, can this be guaranteed by modifying the assumptions appropriately?

It would be desirable to have an adjoint pair of width-preserving sd-functors between the persistent and cumulative temporalized spined sd-categories. This would not only reveal a fundamental relation between the persistent and cumulative perspectives but also allow a convenient transfer between concepts and results for persistent narratives to cumulative ones and vice versa.

## Temporal cellular sheaves: modelling multi-agent systems with switching topology

Temporal cellular sheaves demonstrate how the narrative framework applies naturally to heterogeneous multi-agent systems with switching communication topologies. Choosing cellular sheaves as the target category allows the framework to encode not only the evolution of the communication architecture but also the heterogeneous sensing, communication, and information-processing structures associated with it. This establishes a categorical foundation for studying distributed control problems on time-varying communication networks.

A natural next step is to endow temporal cellular sheaves with dynamical systems evolving on the vector spaces assigned to their cells. This raises the problem of understanding how the dynamics interact with topology changes and with the persistent and cumulative structures encoded by the narrative. Of particular interest is the development of a stability theory for the sheaf-theoretic description of switching communication networks, leading to conditions under which distributed coordination objectives—such as consensus, formation maintenance, or target tracking—remain stable despite changes in the communication topology. This research program is currently under development in [47].

## Declarations

## Funding

Benjamin Merlin Bumpus was supported by the Sao Paulo Research Foundation (FAPESP),˜ grant 2025/16921-5.

## Conflict of interest

The authors declare that they have no competing interests.

## Data availability

No datasets were generated or analyzed during the current study.

## Materials availability

Not applicable.

Code availability

Not applicable.

## Author contributions

The authors’ contributions are described according to the CRediT (Contributor Roles Taxonomy) as follows. Conceptualization: W.L., B.B. Methodology: W.L., B.B. Formal analysis: W.L., B.B., J.N., J.G. Investigation: W.L., B.B., J.N., J.G. Resources: All authors. Writing—original draft: W.L., B.B., J.N. Writing-review and editing: All authors. Supervision: W.L., B.B., J.F., W.D. Project administration: W.L., B.B. Funding acquisition: W.L., B.B., J.N., J.G., J.F., W.D.

## References

[1] Miritello, G.: Temporal Patterns of Communication in Social Networks. Springer Theses. Springer, Cham, Switzerland (2013). https://doi.org/10.1007/978-3-319-00110-4

[2] Choi, B., Berges, M., Bou-Zeid, E., Pozzi, M.: Short-term probabilistic forecasting of´ meso-scale near-surface urban temperature fields. Environmental Modelling & Software 145, 105189 (2021) https://doi.org/10.1016/j.envsoft.2021.105189

[3] Meliker, J.R., Sloan, C.D.: Spatio-temporal epidemiology: Principles and opportunities. Spatial and Spatio-temporal Epidemiology 2(1), 1–9 (2011) https://doi.org/10.1016/j. sste.2010.10.001

[4] Niu, N., Osgood, N.D., Szelko, J.S., Srinivasan, P.V.: Temporal sheaf theory for reconciling temporal complexity within public health modelling. In: Proceedings of the Ninth International Conference on Applied Category Theory (2026). Accepted for publication. https://actconf2026.github.io/papers/ACT 2026 paper 51.pdf

[5] Mesbahi, M., Egerstedt, M.: Graph Theoretic Methods in Multiagent Networks. Princeton Series in Applied Mathematics. Princeton University Press, Princeton, NJ (2010)

[6] Olfati-Saber, R., Fax, J.A., Murray, R.M.: Consensus and cooperation in networked multi-agent systems. Proceedings of the IEEE 95(1), 215–233 (2007) https://doi.org/10. 1109/JPROC.2006.887293

[7] Teich, M., Leal, W., Jost, J.: Diachronic data analysis supports and refines conceptual metaphor theory. PLOS Complex Systems 2(8), 0000058 (2025) https://doi.org/10. 1371/journal.pcsy.0000058 . Article e0000058

[8] Laubichler, M.D., Maienschein, J., Renn, J.: Computational perspectives in the history of science: To the memory of peter damerow. Isis 104(1), 119–130 (2013) https://doi. org/10.1086/669891

[9] Llanos, E.J., Leal, W., Luu, D.H., Jost, J., Stadler, P.F., Restrepo, G.: Exploration of the chemical space and its three historical regimes. Proceedings of the National Academy of Sciences 116(26), 12660–12665 (2019) https://doi.org/10.1073/pnas.1816039116

[10] Atluri, G., Karpatne, A., Kumar, V.: Spatio-temporal data mining: A survey of problems and methods. ACM Computing Surveys 51(4), 83–18341 (2018) https://doi.org/ 10.1145/3161602

[11] Laxman, S., Sastry, P.S.: A survey of temporal data mining. Sadhana 31(2), 173–198 (2006) https://doi.org/10.1007/BF02719780

[12] Habereder, I., Kneib, T., Echizen, I., Spinde, T.: A Systematic Review of Spatio-Temporal Statistical Models: Theory, Structure, and Applications (2025). https://arxiv. org/abs/2511.00422

[13] Roddick, J.F., Spiliopoulou, M.: A survey of temporal knowledge discovery paradigms and methods. IEEE Transactions on Knowledge and Data Engineering 14(4), 750–767 (2002) https://doi.org/10.1109/TKDE.2002.1019212

[14] Roddick, J.F., Patrick, J.D.: Temporal semantics in information systems—a survey. Information Systems 17(3), 249–267 (1992) https://doi.org/10.1016/0306-4379(92) 90016-G

[15] Bumpus, B.M., Leal, W., Fairbanks, J., Karvonen, M., Simard, F.: Towards a unified theory of time-varying data. Applied Categorical Structures 34(3), 23 (2026) https://doi. org/10.1007/s10485-026-09860-4

[16] Bumpus, B.M., Nickel, J.K.: Decomposing time-varying data into simple pieces: structured decompositions of narratives (2026). https://doi.org/10.48550/arXiv.2607.10442

[17] Borges, J.L.: El Jard´ın de Senderos Que Se Bifurcan. Editorial Sur, Buenos Aires (1941)

[18] Johnstone, P.: A note on discrete Conduche fibrations. Theory and Applications of ´ Categories 5(1), 1–11 (1999)

[19] Lack, S., Sobocinski, P.: Adhesive categories. In: Walukiewicz, I. (ed.) Foundations of Software Science and Computation Structures, pp. 273–288. Springer, Berlin, Heidelberg (2004). https://doi.org/10.1007/978-3-540-24727-2 20

[20] Ehrig, H., Ehrig, K., Prange, U., Taentzer, G.: Fundamentals of Algebraic Graph Transformation. Monographs in Theoretical Computer Science. An EATCS Series. Springer, Berlin, Heidelberg (2006). https://doi.org/10.1007/3-540-31188-2

[21] Bumpus, B.M., Kocsis, Z.A., Master, J.E., Minichiello, E.: Structured Decompositions: Structural and Algorithmic Compositionality (2025). https://arxiv.org/abs/2207. 06091v7

[22] Robertson, N., Seymour, P.D.: Graph minors. xvii. taming a vortex. Journal of Combinatorial Theory, Series B 77(1), 162–210 (1999)

[23] Courcelle, B.: The monadic second-order logic of graphs. i. recognizable sets of finite graphs. Information and Computation 85(1), 12–75 (1990)

[24] Halin, R.: S-functions for graphs. Journal of Geometry 8, 171–186 (1976)

[25] Robertson, N., Seymour, P.D.: Graph minors. iii. planar tree-width. Journal of Combinatorial Theory, Series B 36(1), 49–64 (1984)

[26] Liu, J.W.H.: A tree model for sparse symmetric indefinite matrix factorization. SIAM Journal on Matrix Analysis and Applications 9(1), 26–39 (1988)

[27] Yannakakis, M.: Algorithms for acyclic database schemes. In: Proceedings of the Seventh International Conference on Very Large Data Bases (VLDB ’81), pp. 82–94. IEEE

Computer Society, Cannes, France (1981)

[28] Arnborg, S., Proskurowski, A.: Linear time algorithms for NP-hard problems restricted to partial k-trees. Discrete Applied Mathematics 23(1), 11–24 (1989)

[29] Dechter, R., Pearl, J.: Tree clustering for constraint networks. Artificial Intelligence 38(3), 353–366 (1989)

[30] Lauritzen, S.L., Spiegelhalter, D.J.: Local computations with probabilities on graphical structures and their application to expert systems. Journal of the Royal Statistical Society: Series B (Methodological) 50(2), 157–194 (1988)

[31] Robertson, N., Seymour, P.D.: Graph minors. xiii. the disjoint path problem. Journal of Combinatorial Theory, Series B 63(1), 65–110 (1995)

[32] Robertson, N., Seymour, P., Thomas, R.: Sachs’ linkless embedding conjecture. Journal of Combinatorial Theory, Series B 64(2), 185–227 (1995)

[33] Fujita, T.: A brief overview of applications of tree-width and other graph width parameters. Applied Mathematics on Science and Engineering 2(1), 1–20 (2025)

[34] Harary, F., Gupta, G.: Dynamic graph models. Mathematical and Computer Modelling 25(7), 79–87 (1997)

[35] Kempe, D., Kleinberg, J., Kumar, A.: Connectivity and inference problems for temporal networks. Journal of Computer and System Sciences 64(4), 820–842 (2002)

[36] Casteigts, A., Flocchini, P., Quattrociocchi, W., Santoro, N.: Time-varying graphs and dynamic networks. In: Frey, H., Li, X., Ruehrup, S. (eds.) Ad-hoc, Mobile, and Wireless Networks, pp. 346–359. Springer, Berlin, Heidelberg (2011)

[37] Holme, P., Saramaki, J.: Temporal networks. Physics Reports ¨ 519(3), 97–125 (2012)

[38] Holme, P.: Modern temporal network theory: a colloquium. The European Physical Journal B 88(234) (2015)

[39] Michail, O.: An introduction to temporal graphs: An algorithmic perspective. Internet Mathematics 12 (2015)

[40] Carmesin, J., Jacobs, R.W., Knappe, P., Kurkofka, J.: Canonical graph decompositions and local separations: From infinite coverings to a finite combinatorial theory (2025). https://arxiv.org/abs/2501.16170v1

[41] Diestel, R., Jacobs, R.W., Knappe, P., Kurkofka, J.: Canonical graph decompositions via coverings (2025). https://arxiv.org/abs/2207.04855v8

[42] Ames, A.D.: A categorical theory of hybrid systems. PhD thesis, University of California, Berkeley (2006). https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/ EECS-2006-165.pdf

[43] Serre, J.-P.: Arbres, amalgames, sl<sub>2</sub>. Asterisque, Soci´ et´ e Math´ ematique de France, Paris´ (46) (1977). Redig´ e avec la collaboration de Hyman Bass´

[44] Serre, J.-P.: Trees. Springer Monographs in Mathematics. Springer, Berlin, Heidelberg (2003)

[45] Bass, H.: Covering theory for graphs of groups. Journal of Pure and Applied Algebra (89), 3–47 (1993)

[46] Hu, C.-S.: Cellular Sheaves on Higher-Dimensional Structures (2025). https://arxiv.org/ abs/2505.23993v3

[47] Copeland, A., Leal, W., Fallin, B., Bumpus, B.M., Fairbanks, J., Dixon, W.E.: A Temporal Cellular Sheaf Framework for Multi-Agent Systems with Switching Topologies. Work in progress (2026)

[48] Currier, K., Leal, W., Fallin, B., Fairbanks, J., Dixon, W.E.: From local to global: Sheaf-theoretic control barrier functions on manifolds. In: Proceedings of the IEEE 65th Conference on Decision and Control (CDC). IEEE, Honolulu (2026)

[49] Bullo, F., Cortes, J., Mart´ ´ınez, S.: Distributed Control of Robotic Networks. Princeton Series in Applied Mathematics. Princeton University Press, Princeton, NJ, USA (2009)

[50] Moreau, L.: Stability of multiagent systems with time-dependent communication links. IEEE Transactions on Automatic Control 50(2), 169–182 (2005) https://doi.org/10. 1109/TAC.2004.841888

[51] Hanks, T., Nino, C.F., Barcelo, J.B., Copeland, A., Dixon, W., Fairbanks, J.: Heterogeneous Multi-Agent Multi-Target Tracking using Cellular Sheaves (2025). https: //arxiv.org/abs/2512.24886

[52] Hanks, T., Riess, H., Cohen, S., Gross, T., Hale, M., Fairbanks, J.: Distributed multiagent coordination over cellular sheaves. In: 2025 IEEE 64th Conference on Decision and Control (CDC), pp. 3057–3064 (2025). https://doi.org/10.1109/CDC57313.2025. 11312066

[53] Scattolini, R.: Architectures for distributed and hierarchical model predictive control – a review. Journal of Process Control 19(5), 723–731 (2009) https://doi.org/10.1016/j. jprocont.2009.02.003

[54] Bai, H., George, J., Chakrabortty, A.: Hierarchical control of multi-agent systems using online reinforcement learning. In: Proceedings of the 2020 American Control Conference (ACC), pp. 340–345. IEEE, Denver, CO, USA (2020). https://doi.org/10.23919/ ACC45564.2020.9147797

[55] Zegers, F.M., Phillips, S., Dixon, W.E.: Consensus over clustered networks with asynchronous inter-cluster communication. In: 2021 American Control Conference (ACC), pp. 4249–4254 (2021). https://doi.org/10.23919/ACC50511.2021.9482931