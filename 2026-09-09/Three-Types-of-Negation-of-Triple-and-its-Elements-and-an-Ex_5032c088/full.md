# Three Types of Negation of Triple and its Elements and an Extension of Triple

Zhenghua Pan

School of Mathematics and Data Science, Jiangnan University, Wuxi 214122, China panzh@jiangnan.edu.cn

Abstract: In various data models, the classical triple <s, p, o> is a typical semantic data model. However, due to the design of the triple as a simple structure for representing positive assertions, it cannot sufficiently express different forms of negation present in the triple and its elements. This paper conceptually proposes that there are three distinct forms of negation within triples and their elements: contradictory negation, opposite negation and intermediary negation. Based on the the set SCOI and the logic LCOI+PLCOI with three kinds of negation, we propose an extension of triple that can distinguish and express these three different negations in the triple and its elements, called the “TCOI triple with contradictory negation, opposite negation and intermediary negation”.

The TCOI triple is a semantic and structural extension of the classical triple. While retaining the ability to express positive assertions, it systematically introduces the three semantic dimensions of contradictory negation, opposite negation and intermediary negation, allowing these negations to independently act on the elements (s, p, o) of the triple and on the whole triple. This significantly enhances the triple model capability to represent and reasoning about complex negative information.

This paper also explores the expressive power and reasoning of the TCOI triple. Based on the semantics of the logic LCOI+PLCOI, we introduce the notion of “TCOI-entailment” as the semantics implication in TCOI triple implication reasoning, thereby establishing a connection between TCOI triple implication reasoning and inferences in the logic LCOI+PLCOI. This shows that LCOI+PLCOI provide the logical foundation for TCOI triple implication reasoning. Furthermore, we discuss the application of TCOI triple implication reasoning in counterfactuals and counterfactual reasoning involving the three different types of negation. We propose a truth-value (continuous value) algorithm for TCOI triple implication reasoning and perform its calculation through an example of the counterfactuals and counterfactual reasoning.

Keywords: triple; triple reasoning; negation; sets and logic; counterfactual reasoning; algorithm

## 1. Introduction

In various data models, triples are a typical semantic data model. They express semantic information in an ordered format <subject, predicate, object> (or <s, p, o>). Triples play an irreplaceable and unique role in knowledge representation, knowledge reasoning, and knowledge interconnection in a concise manner. The most profound significance of triples is that they provide machines with a standardized language to simulate human expression of “facts” and “relationships” [1]. Triples are widely used in fields such as the Semantic Web, Resource Description Framework (RDF), knowledge graphs, graph computing, information retrieval and semantic search, natural language processing (NLP), as well as artificial intelligence and machine learning [2-8].

Negation is a complex natural language phenomenon and also an important basic concept in knowledge [9]. Broadly speaking, negation connects an expression E with another expression whose meaning is, in some sense, opposed to that of E [10]. Therefore, the key challenge in understanding negation is to identify the meaning that is in some way opposed to E—this is a semantically complex and highly vague task [11]. Negation in triples is of significant importance in knowledge representation and reasoning. It can be said that negation in triples is an indispensable "other half" of knowledge representation and reasoning. It allows systems (such as Resource Description Framework and knowledge graphs) to extend from merely describing what exists to being able to articulate what does not exist, thereby achieving logical completeness, precision of knowledge, and depth of reasoning.

Regarding the negativity of triples, from the semantic level of the triple structure, there are two forms: one is negativity at the triple level, and the other is negativity at the element level. These two forms of negativity do not have only one form of negation in practice, but rather different types of negation with different meanings.

For example: in triple expressions, for the triple T: <x, property, positive number> (i.e., the statement “x is a positive number”), the triples T1: <x, property, non-positive number> (i.e., the statement “x is not a positive number”), T2: <x, property, negative number > (i.e., the statement “x is a negative number”) and T3: <x, property, zero> (i.e., the statement “x is zero”) represent different negations of triple T. In the expression of elements within triple, for the predicate “likes” in the triple <he, likes, her>, ‘dislikes’, ‘hates’ and ‘neither likes nor hates’ are three different types of negation for the predicate “likes”.

For the different negations present in triples and their elements, the classical triple <s, p, o>, as a structure expressing simple positive semantic relationships, can only express affirmative assertions between the subject, predicate, and object, lacking an inherent mechanism to distinguish and express these different types of negation.

For the classical triple <s, p, o>, since there is only one type of negation (classical negation) in the formal languages of classical and non-classical logic, it cannot distinguish or express these different forms of negation existing in triples and their elements. Therefore, to accurately and precisely express these different negations in knowledge representation, the classical triple <s, p, o> needs to be extended based on a non-classical logic that can distinguish and express different negations at both the syntactic and semantic interpretation levels. Such logic includes multiple negation operators or semantic clarification mechanisms to achieve complete expression of negation meanings and accurate semantic modeling.

In this paper, we conceptually propose that there exist three different forms of negation within triple and their elements: “contradictory negation”, “opposite negation”, and “intermediary negation”. Based on the set SCOI and the logic LCOI+PLCOI with three kinds of negation [12, 13], we present an extension of triple capable of distinguishing and expressing these negations: the “TCOI triple with contradictory negation, opposite negation and intermediary negation”. TCOI triple is a semantic and structural extension of the classical triple <s, p, o>. While retaining the ability to express affirmative assertions, it systematically introduces the three semantic dimensions of contradictory negation, opposite negation and intermediary negation, allowing these negations to independently apply to the elements (s, p, o) of the triple and on the whole triple. This significantly enhances the triple model capability to represent and reasoning about complex negative information.

In this paper, we also discuss the expressive power and reasoning of the TCOI triple. Based on the semantics of the logic LCOI+PLCOI, we introduce the notion of “TCOI-entailment” as the semantics implication in TCOI triple implication reasoning. Through TCOI-entailment, a connection is established between TCOI triple implication reasoning and inferences in the logic LCOI+PLCOI. This demonstrates that the formal inference properties already proven in LCOI+PLCOI are valid in TCOI triple implication reasoning. LCOI+PLCOI provide the logical foundation for TCOI triple implication reasoning. Furthermore, we discuss the application of TCOI triple entailment reasoning in counterfactuals and counterfactual reasoning involving the three types of negation. We propose a truth-value (continuous value) algorithm for TCOI triple implication reasoning and perform truth-value calculation on an example of counterfactuals and counterfactual reasoning.

As far as we know, there is no direct evidence indicating the existence of research specifically focusing on negation within triples and their elements. This paper investigates negation in triples and proposes that there exist three different types of negation within triples and their elements, as well as an extension of triples capable of distinguishing and expressing these negations—work that, to date, has not been seen. The main contributions of this paper are as follows:

1. Conceptually proposing that three different types of negation exist within triple and their elements: contradictory negation, opposite negation, and intermediary negation.

2. Under the set SCOI and the logic LCOI+PLCOI framework, we propose a triple that can express three different types of negations present in triple and their elements: “TCOI triple with contradictory negation, opposite negation and intermediary negation”. TCOI not only distinguishes between clear triple and fuzzy triple, as well as the correlation degree between the subject and the object, but it is also capable of expressing the contradictory negation, opposite negation and intermediary negation for the triple and its elements (predicate, object).

3. Based on the semantics of the logic LCOI+PLCOI, we introduce a notion “TCOI-entailment” as the implication in TCOI triple implication reasoning. Through TCOI-entailment, we establish a connection between TCOI triple implication reasoning and inferences in the logic LCOI+PLCOI. This shows that the formal inference properties already proven in LCOI+PLCOI are valid in TCOI triple implication reasoning. LCOI+PLCOI provide the logical foundation for TCOI triple implication reasoning.

4. Applying TCOI triple implication reasoning to counterfactuals and counterfactual reasoning. A truth-value (continuous value) algorithm for TCOI triple implication reasoning is proposed, and a calculation is performed on an example of counterfactuals and counterfactual reasoning.

The organization of this paper is as follows. Section 2 discusses related work. Section 3 conceptually discusses the three types of negation and their characteristics within triples and their elements. Section 4 introduces the basic definitions of the logical set SCOI and the logic LCOI+PLCOI with three types of negation. It proposes a continuous truth-value semantics with the truth domain [0, 1] for logic LCOI+PLCOI and discusses its metalogical properties. In Section 5, based on LCOI+PLCOI, the TCOI triple with three types of negation is proposed. Section 6 discusses the expressive power of the TCOI triple. Section 7 explores TCOI triple reasoning and its applications. Section 8 summarizes the main conclusions of this paper and future work.

## 2. Related works

This paper aims to conceptually propose that there exist three different forms of negation in general triple and their elements: contradictory negation, opposite negation, and intermediary negation. Furthermore, based on the set SCOI and the logic LCOI+PLCOI, it proposes an extension of triple that can distinguish and express these negations.

Among related studies, reference [13] is the only one most closely related to this paper. In the extension of the Resource Description Framework called RDFCOI, it proposes and defines a suitable triple for RDFCOI, the RDFCOItriple. The RDFCOI triple maintains the definition of the RDF triple and is capable of expressing the three types of negation for the RDFCOI. However, that work is not specifically focused on the negation of triples and their elements, nor does it discuss the reasoning and logical foundation of triples with three types of negation. Therefore, the purpose and significance of this paper differ from those of reference [13]. The related work relevant to this paper can be divided into two aspects: the study of negation of whole triple, and the study of negation of elements within triple.

Regarding the negation of whole triple, there are no direct studies found in the literature. However, several research works related to triples demonstrate the practical role and significance of negation. Among the most directly relevant papers, one extends RDFS to handle negative statements under the Open World Assumption (OWA) and explicitly avoids reification for negative triples [14]. Multiple negation concepts are operationalized through weak negation, strong negation, local closed world assumption, and scoped negation as failure, explicitly formalizing negation similar to that in RDF knowledge representation [15, 16]. Negative statements are considered useful; one proposal is ERDF, where an RDF triple can be positive or negative and distinguishes between weak and strong negation [17]. Another work directly addresses negative statements and reports that negation via SPARQL MINUS does not yield relevant knowledge in the studied cases [18]. There are broader surveys on negative statements, completeness, and partial closed-world semantics around open-world knowledge bases [19]. Another closely related area is negation in queries. Reference [20] analyzes SPARQL negation operators, their expressive power, and algebraic behavior. SPARQL property path negation is related to finite negation on graph navigation patterns rather than simple assertion triples [21, 22]. Some works explicitly study queries with negation under RDF’s open-world setting and define soundness checking for them [23], and so on.

Regarding the negation of elements within triple, there are no explicit analytical studies focusing on the negation of the three elements, nor are there works presenting a comprehensive extended triple that can handle negation of the subject, predicate, and object separately. The most directly relevant research explicitly involves negation in the predicate position, while negation of the subject and object is typically handled indirectly under the broader notions of “negative statements” or “negative knowledge” in RDF/SPARQL [24, 25, 26]. For the negation of predicate, SPARQL triple patterns with negation have been explicitly extended at the predicate position, with their syntax and semantics defined [26]. For the negation of object, modeling of negative statements such as "no treatment" and general negation in relational data frameworks (RDF) has been proposed, introducing a minimal entailment system for RDFS that includes negative statements [24]. Additionally, representation and querying of negative knowledge in RDF cover practical aspects of missing/negated information at the object level [25]. For the negation of subject, research exists on generating negative statements about real-world entities, i.e., subject-centered negation, although this is not formal RDF triple logic [27].

In summary, there is no direct evidence of research specifically addressing negation in triples and their elements, nor is there an extension of triples that can separately handle negation of the triple and its individual elements.

## 3. Three types of negation in triple and its elements and their characteristics

A triple expresses a complete and indivisible statement whose meaning is determined by concepts (the smallest unit in a statement), with the elements of the triple (s, p, o) each representing different concepts. Therefore, the negation of the triple <s, p, o> and its elements can be uniformly reduced at the conceptual level to the negation of atomic concepts (the elements) or the compound of atomic concepts (the whole triple). In other words, the negation of the triple and its elements is based on the negation of atomic concepts.

In references [12] and [13], we distinguish between “crisp concepts” and “fuzzy concepts” at the conceptual level and thoroughly understand the “contradiction” and “opposition” within concepts, thereby proposing that three different types of negation exist in atomic concepts: contradictory negation, opposite negation, and intermediary negation. Therefore, we believe that these three different forms of negation must also exist within triples and their elements.

In this section, based on a brief overview of these three forms of negation, we discuss their characteristics.

(1) Contradictory Negation. For a species concept under a genus concept, another species concept that has a contradictory relationship with it constitutes a form of negation. We refer to this type of negation as “contradictory negation”. In this form of negation, the intensions (connotations) of the two species concepts mutually negate each other, the extensions (denotations) are mutually exclusive (either one or the other), and the sum of the extensions equals the extension of the genus concept. For example, for the species concept ‘positive integer’ under the genus concept “integer”, another species concept ‘non-positive integer’ is its contradictory negation. For the two species concepts ‘daytime’ and ‘non-daytime’ under the genus concept “one day”, the latter is the contradictory negation of the former. From this, it can be known that the negation in classical logic is precisely this kind of negation.

(2) Opposite Negation. For a species concept under a genus concept, another species concept that has an oppositional relationship with it constitutes another form of negation. We refer to this type of negation as “opposite negation”. In this form of negation, the intensions of the two species concepts mutually negate each other and exhibit the greatest difference in intension, but their extensions are not mutually exclusive (not either-or), and the sum of their extensions is less than the extension of the genus concept. For example, for the species concept ‘positive integer’ under the genus concept “integer”, another species concept ‘negative integer’ is its opposite negation. For the two species concepts ‘daytime’ and ‘night’ under the genus concept “one day”, the latter is the opposite negation of the former.

(3) Intermediary Negation. The intermediary concept between opposite concepts constitutes a (weak) form of negation of the opposite concepts. We refer to this type of negation as “intermediary negation”. In this form of negation, the opposite concepts transition through the intermediary concept and the sum of their extensions equals the extension of the genus concept. For example, under the genus concept “integer”, between the two opposing species concepts ‘positive integer’ and ‘negative integer’, the concept “zero” is their intermediary negation. Under the genus concept “one day”, between the two opposing species concepts ‘daytime’ and ‘night’, the concepts “dusk” and “dawn” are their intermediary negation.

From the meanings of the three types of negations mentioned above, the contradictory negation is the traditional negation. The opposite negation can be referred to as strong negation, and intermediary negation as weak negation.

To fully understand the meaning of the above three kinds of negation, we further discuss their characteristics in terms of both the intension of the concepts as well as their extensional relations.

(1) Contradictory Negation in Clear Concepts (CNC)

Characteristics of CNC: Extensions are clear, either this or that, and the sum of extensions is equal to the extension of the genus concept.

For example, the positive integer and non-positive integer under the genus concept of “integer” are clear concepts, while the non-positive integer is the contradictory negation of positive integer. The diagram illustrating the extensional relationship between them is shown below (Figure 1).

![](images/5a36c488b40d6158c47b53c239f75ed53fc89035eac7fe1811e62bf8106df8c8.jpg)  
Fig. 1 The extensional relationship between positive integer and non-positive integer

## (2) Opposite Negation in Clear Concepts (ONC)

Characteristics of ONC: Extensions are clear, not “either this or that”, and the sum of the extensions is less than the extension of the genus concept.

For example, the positive integer and negative integer under the genus concept of “integer” are clear concepts, while the negative integer is the opposite negation of positive integer. The diagram illustrating the extensional relationship between them is shown below (Figure 2).

![](images/f94a50243a9ae0a322c688368e2d1bc015279ffa31cb359932703fd3dc451077.jpg)  
Fig. 2 The extensional relationship between positive integer and negative integer

## (3) Intermediary Negation in Clear Concepts (INC)

Characteristics of INC: Extensions are clear, opposing sides transition to each other through ‘intermediaries’, and the sum of the extensions equals the extension of the genus concept.

For example, zero, positive integer and negative integer are clear concepts under the genus concept of “integer”. Zero serves as an ‘intermediary’ between positive integer and negative integer, and it is the intermediary negation of both positive integer and negative integer. The diagram illustrating the extensional relationship between them is shown below (Figure 3).

![](images/6e9cd3c7fbf15b14be6e2ed5b8227f555aca989831c69b47df46a7916950b289.jpg)  
Fig. 3 The extensional relationship between positive integers, negative integers, and zero.

## (4) Contradictory Negation in Fuzzy Concepts (CNF)

Characteristics of CNF: Extensions are not clear, either this or that, and the sum of extensions is equal to the extension of the genus concept.

For example, the daytime and non-daytime under the genus concept of “day” are fuzzy concepts, while the non-daytime is the contradictory negation of daytime. The diagram illustrating the extensional relationship between them is shown below (Figure 4).

![](images/4d39a00db939b5060c4dfb3a193c691d3924df9a68d21fafe6535b87e9e07a0d.jpg)  
Fig. 4 The extensional relationship between daytime and non-daytime

## (5) Opposite Negation in Fuzzy Concepts (ONF)

Characteristics of ONF: Extensions are not clear, not “either this or that”, and the sum of the extensions is less than the extension of the genus concept.

For example, the daytime and night under the genus concept of “day” are fuzzy concepts, while the night is the opposite negation of daytime. The diagram illustrating the extensional relationship between them is shown below (Figure 5).

![](images/095dcf2c547bb09c93db66357c72f7981aafc1c6964cdb5ad03dfa27ff7b6f23.jpg)  
Fig. 5 The extensional relationship between daytime and night

## (6) Intermediary Negation in Fuzzy Concepts (INF)

Characteristics of IFC: Extensions are not clear, opposing sides transition to each other through

‘intermediaries’, and the sum of the extensions equals the extension of the genus concept.

For example, dusk, daytime and night are fuzzy concepts under the genus concept of “day”. Dusk serves as an ‘intermediary’ between daytime and night, and it is the intermediary negation of both daytime and night. The diagram illustrating the extensional relationship between them is shown below (Figure 6).

![](images/59c393f44036fafbe6ee782d346e273af5bdae5275c611b46423e8badd0c13e1.jpg)  
Fig. 6. The extensional relationship between daytime, night, and dusk

The above distinction at the conceptual level differentiates contradictory negation, opposite negation, and intermediary negation in atomic concepts, clarifying that negation in concepts is not a single variant of the classical “not”. Since the negation of triple and their elements is logically founded upon the negation of atomic concepts, the three types of negation in atomic concepts must correspondingly be the forms of negation in triples and their elements. They constitute the semantic primitives of the entire knowledge represented by the triple.

## 4. Set and logic with three kinds of negation

For the three kinds of negation present in the aforementioned concepts, in order to establish a mathematical foundation that can fully reflect them along with their properties, relationships, and laws, we proposed a set SCOI and logic LCOI+PLCOI with contradictory negation, opposite negation and intermediary negation that takes clear and fuzzy entities as the research objects [12,13].

In this section, we provide an overview of SCOI and LCOI+PLCOI, and propose a continuous-valued semantics for LCOI+PLCOI with a truth value range of [0, 1]. Under this semantics, the soundness theorem for LCOI+PLCOI is proved.

## 4.1 Set with contradictory negation, opposite negation and intermediary negation

We use symbols $\neg , \exists$ , and  to represent ‘contradictory negation’, ‘opposite negation’, and ‘intermediary negation’, respectively.

Definition 1. Let U be universe of discourse, $\lambda { \in } ( 0 , 1 )$ . Mapping $f \colon U \to [ 0 , 1 ]$ confirms a set A on $U ,$ call  the membership function of A, and (x) the membership degree of x to A (denoted as A(x)).

(1) If A is a fuzzy set, then

(i). Mapping $f ^ { \daleth } : \{ A ( x ) \mid x \in U \}  [ 0 , 1 ]$ confirms a fuzzy set $A ^ { \daleth }$ on $U , A ^ { \dagger } ( x ) = f ^ { \dagger } ( A ( x ) ) = 1 - A ( x )$ . Call $A ^ { \daleth }$ the opposite negation set of A.

(ii). Mapping $f ^ { \prime } \colon \{ A ( x ) \mid x \in U \} \to [ 0 , 1 ]$ confirms a fuzzy set $A ^ { \sim }$ on $U , A ^ { \sim } ( x ) = f ^ { \sim } ( A ( x ) )$ . Call $A ^ { \sim }$ the intermediary negation set of A. Where

$$
\left\{ \begin{array} { l l } { { \displaystyle \lambda \displaystyle { - \frac { 2 \lambda - 1 } { 1 - \lambda } ( A ( x ) - \lambda ) } , } } & { { \qquad \mathrm { w h e n } ~ \lambda \in [ \gamma _ { 2 } , 1 ) \mathrm { a n d } ~ A ( x ) \in ( \lambda , 1 ] } } \end{array} \right.\tag{a}
$$

$$
\lambda - \frac { 2 \lambda - 1 } { 1 - \lambda } A ( x ) , \qquad \mathrm { w h e n } ~ \lambda \in [ 1 / 2 , 1 ) \mathrm { a n d } ~ A ( x ) \in [ 0 , 1 - \lambda )\tag{b}
$$

$$
\begin{array} { r } { A ^ { \sim } ( x ) = ~ \left\{ \begin{array} { l l } { ~ 1 - \displaystyle \frac { 1 - 2 \lambda } { \lambda } A ( x ) - \lambda , \qquad } & { \mathrm { w h e n } ~ \lambda \in ( 0 , 1 / 2 ] ~ \mathrm { a n d } ~ A ( x ) \in [ 0 , \lambda ) } \\ { ~ \lambda } & { } \end{array} \right. } \end{array}\tag{c}
$$

$$
1 - \frac { 1 - 2 \lambda } { \lambda } ( A ( x ) + \lambda - 1 ) - \lambda , \qquad \mathrm { w h e n } ~ \lambda \in ( 0 , { \frac { \textstyle 1 \sqrt { 2 } } { \lambda } } ] ~ \mathrm { a n d } ~ A ( x ) \in ( 1 - \lambda , 1 ]\tag{d}
$$

(e)

(iii). Mapping $f ^ { \neg } \colon \{ A ( x ) \mid x \in U \} \to [ 0 , 1 ]$ confirms a fuzzy set $A ^ { \neg }$ on $U , A ^ { \top } ( x ) = f \Gamma ( A ( x ) ) = m a x ( A ^ { \top } ( x ) , A ^ { \top } ( x ) )$ Call $A ^ { \neg }$ the contradictory negation set of A.

(2) If A is a clear set, then $A ( x ) \in \{ 0 , 1 \} , A ^ { \rceil } ( x ) = 1 - A ( x ) , A ^ { \sim } ( x ) = { } ^ { 1 } / 2 , A ^ { \top } ( x ) = m a x ( A ^ { \top } ( x ) , A ^ { \sim } ( x ) ) .$

The set on the domain U determined above is called “Sets with contradictory negation, opposite negation and intermediary negation”, for short SCOI.

## 4.2 Logic with contradictory negation, opposite negation and intermediary negation

Based on SCOI, we further propose “logic LCOI+PLCOI with contradictory negation, contrary negation, and intermediary negation”. LCOI+PLCOI is a formal logical calculus system that extends the syntax and semantics of classical logic. LCOI is propositional logic, and PLCOI is predicate logic.

## 4.2.1 Definition of the logic LCOI+PLCOI

Symbols $\neg , \ \exists$ , and  denote “contradictory negation”, “opposite negation” and “intermediary negation” respectively. Symbols ,  and  denote ‘disjunction’, ‘conjunction’ and ‘implication’, respectively. $\mathrm { S y m b o l } ^ { \ast } \vdash $ denote formal deduction. The definition of logic LCOI+PLCOI is as follows

Definition 1. Let  be set of atomic proposition. $\forall \mathbf { A } \in \Im ,$ , A is called well-formed formula (or formula). If A, B are formulas, then ¬A, = A, \~A, A→B, A√B and A∧B are formulas.

(I) The following formulas as axioms:

(a1) $\mathbf { A } {  } ( \mathbf { B } {  } \mathbf { A } )$

(a2) $( \mathrm { A } {  } ( \mathrm { A } {  } \mathrm { B } ) ) {  } ( \mathrm { A } {  } \mathrm { B } )$

(a3) $\mathrm { ( A {  } B ) {  } ( ( B {  } C ) {  } ( A {  } C ) ) }$

(a4) $( \mathbf { A } {  } \lnot \mathbf { B } ) {  } ( \mathbf { B } {  } \lnot \mathbf { A } )$

(a5) $( \mathrm { A } {  } \thinspace \thinspace \exists \mathrm { ~ B } ) {  } ( \mathrm { B } {  } \thinspace \exists \mathrm { ~ A } )$

(a6) $\neg \mathbf { A } \neg \mathbf { ( A \to B ) }$

(a7) $( ( \mathrm { A } \to \lnot \mathrm { A } ) { \to } \mathrm { B } ) { \to } ( ( \mathrm { A } { \to } \mathrm { B } ) { \to } \mathrm { B } )$

(a8) $\mathbf { A } \to \mathbf { A } \lor \mathbf { B }$

(a9) $\mathbf { B }  \mathbf { A } \lor \mathbf { B }$

(a10) $\mathbf { A } \wedge \mathbf { B } \to \mathbf { A }$

(a11) $\mathbf { A } \wedge \mathbf { B } \to \mathbf { B }$

(a12) $\exists \mathrm { A } \to \lnot \mathrm { A } \land \lnot \sim \mathrm { \ l { \sim } } \mathrm { A } , \lnot \mathrm { A } \land \lnot \sim \mathrm { \ l { \sim } } \mathrm { A } \to \ l \mathrm { A }$

(a13) ${ \sim } \mathrm { A } \to { \neg } \mathrm { A } \land { \neg } \ q \mathrm { A } , { \neg } \mathrm { A } \land { \neg } \ q \mathrm { A } \to { \sim } \mathrm { A }$

(II) The deduction rules:

[D1] $\mathbf { A } _ { 1 } , \mathbf { A } _ { 2 } , . . . , \mathbf { A } _ { \mathrm { n } } \vdash \mathbf { A } _ { \mathrm { i } } \left( 1 \leq i \leq \mathrm { n } \right)$

[D2] $\mathbf { A { \to } B , A } \Vdash \mathbf { B }$

The logic calculus formal system determined above is called“propositional logic with contradictory negation, opposite negation and intermediary negation”, for short LCOI.

On the basis of LCOI, adding predicates, individual words, quantifiers  and , as well as the following axioms and deduction rule, we can constitute a predicate logic PLCOI with contradictory negation, opposite negation and intermediary negation.

(I) Axioms:

(a14) $\forall x \mathbf { A } ( x )  \mathbf { A } ( a )$

(a15) $\mathrm { A } ( a ) {  } \exists x \mathrm { A } ( x )$

(a16) $\forall x ( \mathbf { A } ( x ) {  } \mathbf { B } ) {  } \exists x ( \mathbf { A } ( x ) {  } \mathbf { B } )$

(a17) $\exists \ \forall x \mathbf { A } ( x ) \to \exists x \exists \ \mathbf { A } ( x ) , \exists x \exists \ \mathbf { A } ( x ) \to \exists \ \forall x \mathbf { A } ( x )$

(a18) $\exists \exists x \mathbf { A } ( x ) \to \forall x \exists \mathbf { A } ( x ) , \forall x \exists \mathbf { A } ( x ) \to \exists \exists x \mathbf { A } ( x )$

(II) Deduction rule:

[D3] If $\Sigma \vdash \Lambda ( a ) \ ( \Sigma$ is the set of formulas), where the individual constant a does not appear in $\Sigma ,$ then $\Sigma \vdash \forall x \mathbf { A } ( x )$

The logic calculus formal system determined above is called “predicate logic with contradictory negation, opposite negation and intermediary negation”, for short PLCOI.

The propositional logic LCOI and predicate logic PLCOI are denoted as LCOI+PLCOI.

## 4.2.2 A continuous-valued semantics of LCOI+PLCOI

Regarding the semantics of the logic LCOI+PLCOI, we previously provided a three-valued semantics and proved the soundness theorem, completeness theorem and compactness theorem for LCOI+PLCOI under this semantics [12,13]. In order to make LCOI+PLCOI applicable in practice, we hereby propose continuous-valued semantics for LCOI+PLCOI with a truth domain of [0, 1].

Let  be a set of formulas in LCOI+PLCOI, and let A be a formula in LCOI+PLCOI. Since LCOI+PLCOI is a formal logical system, defining the formal deduction $\Sigma \vdash \mathbf { A }$ is provable in LCOI+PLCOI, just as it is in other formal logic.

Definition 1. The formal deduction $\Sigma { \vdash } \mathbf { A } \left( \Sigma \right.$ can be empty set) is provable in LCOI+PLCOI, if there exists a finite sequence of formulas $E _ { I } , E _ { 2 } , . . . , E _ { n }$ such that $E _ { n } = \mathbf { A }$ and for each $E _ { n } \left( 1 \leq k \leq n \right)$ , either $E _ { k }$ is an axiom in LCOI+PLCOI or $E _ { k }$ follows from $E _ { i }$ and $E _ { j } ( i < k , j < k )$ using the deduction rule in LCOI+PLCOI, then $E _ { I } , E _ { 2 } , . . . , E _ { n }$ is called a $^ { \mathrm { { e } \odot } } p r o o f ^ { \mathrm { { ? } \mathrm { { } ? } } } \ o f \Sigma \vdash \mathbf { A }$ , n length of $p r o o f . \Sigma \vdash \mathbf { A }$ is denoted $\vdash \mathbf { A }$ when  is empty.

Definition 2 (continuous-valued interpretation). Let  be set of all formulas in LCOI+PLCOI, λ∈(0, 1). $\forall \mathbf { A } \in \Im$ mapping $\hat { o } \colon \Im  [ 0 , 1 ]$ is called a -assignment of , consists of the individual domain D and the following assignments for each constant symbol, function symbol and predicate symbol in A:

(1) for each constant symbol, assign an object in D to correspond to it;

(2) for each n-variant function symbol, assign a mapping from $D ^ { n }$ to D to correspond to it;

(3) for each n-variant predicate symbol, assign a mapping from $D ^ { n } \ d _ { \mathrm { ~ t o ~ } [ 0 , 1 ] }$ to correspond to it, and

[1] If A is an atomic formula, (A) takes only one value from [0, 1];

[2] ∂(A)+∂(q A) = 1;

$$
\int \lambda - \frac { 2 \lambda - 1 } { 1 - \lambda } ( \hat { \mathcal { O } } ( \mathrm { A } ) - \lambda ) , \qquad \mathrm { w h e n ~ } \lambda \in [ 1 / 2 , 1 ) \mathrm { ~ a n d ~ } \hat { \mathcal { O } } ( \mathrm { A } ) \in ( \lambda , 1 ]\tag{a}
$$

$$
\begin{array} { r } { \boxed { \begin{array} { c c } { \lambda - \frac { 2 \lambda - 1 } { 1 - \lambda } \hat { \mathcal { O } } ( \mathrm { A } ) , } & { \qquad \mathrm { w h e n ~ } \lambda \in [ 1 / 2 , 1 ) \mathrm { ~ a n d ~ } \hat { \mathcal { O } } ( \mathrm { A } ) \in [ 0 , 1 - \lambda ) } } \end{array} } \end{array}\tag{b}
$$

$$
\begin{array} { r l } { \hat { \mathcal { O } } ( \sim \mathbf { A } ) = } & { \{  1 - \frac { 1 - 2 \lambda } { \lambda } \hat { \mathcal { O } } ( \mathbf { A } ) - \lambda , \quad \quad \quad \quad \quad \mathrm { w h e n ~ } \lambda \in ( 0 , 1 / 2 ] \mathrm { a n d ~ } \hat { \mathcal { O } } ( \mathbf { A } ) \in [ 0 , \lambda )  } \end{array}\tag{c}
$$

$$
1 - \frac { 1 - 2 \lambda } { \lambda } ( \hat { \mathcal { O } } ( \mathrm { A } ) + \lambda - 1 ) - \lambda , ~ \mathrm { w h e n } ~ \lambda \in ( 0 , 1 / 2 ] ~ \mathrm { a n d } ~ \hat { \mathcal { O } } ( \mathrm { A } ) \in ( 1 - \lambda , 1 ]\tag{d}
$$

$$
[ 4 ] \hat { \cal O } ( \neg \mathrm { A } ) = \mathrm { m a x } ( \hat { \sigma } ( \exists \mathrm { \ A } ) , \hat { \sigma } ( \neg \mathrm { A } ) ) ;\tag{e}
$$

[5] ∂(A→B) = R(∂(A), ∂(B)). R: [0, 1]2→ [0, 1] is a binary function;

$$
[ 6 ] ~ { \widehat { \cal O } } ( { \mathrm { A v B } } ) = \operatorname* { m a x } ( { \widehat { \cal O } } ( { \mathrm { A } } ) , { \widehat { \cal O } } ( { \mathrm { B } } ) ) ; ~ { \widehat { \cal O } } ( { \mathrm { A } } { \mathrm { \wedge } } { \mathrm { B } } ) = \operatorname* { m i n } ( { \widehat { \cal O } } ( { \mathrm { A } } ) , { \widehat { \cal O } } ( { \mathrm { B } } ) ) ;
$$

$$
[ 7 ] ~ \hat { \mathcal { O } } ( \forall x P ( x ) ) = \operatorname* { m i n } _ { x \in D } ~ \{ \hat { \mathcal { O } } ( P ( x ) ) \} ; ~ \hat { \mathcal { O } } ( \exists x P ( x ) ) = \operatorname* { m a x } _ { x \in D } ~ \{ \hat { \mathcal { O } } ( P ( x ) ) \} .
$$

In Definition 2, how to determine the truth value $\partial ( { \sim } \mathbf { A } )$ of the intermediary negation \~A of formula A in [0, 1] (i.e., the expression (a)-(e)), the parameter variable $\lambda \left( \lambda { \in } ( 0 , 1 ) \right)$ is the key to the definition. The basic idea is as follows:

Since the truth values $\hat { \sigma } ( \mathrm { A } ) , \hat { \sigma } ( \lnot \mathrm { A } ) , \hat { \sigma } ( \lnot \mathrm { A } ) , \hat { \sigma } ( \lnot \mathrm { A } ) \in [ 0 , 1 ]$ , in order to determine their value range in [0, 1], we introduce a parameter variable $\lambda { \in } ( 0 , 1 )$ . Consequently, when $\lambda \ge 1 / 2 , [ 0 , 1 ]$ is divided into three sub-intervals: [0, 1−λ), [1−λ, λ], (λ, 1]. If $\partial ( \mathbf { A } ) \in ( \lambda , \ 1 ]$ , then based on [2] in the definition, $\partial ( \negmedspace \exists \operatorname { A } ) \in [ 0 , \ 1 - \lambda )$ . At this point, if $\hat { \cal O } ( { \sim } \mathrm { A } ) { \in } [ 1 { - } \lambda , \lambda ]$ , since (λ, 1] and $[ 1 - \lambda , \lambda ]$ are disjoint intervals, then according to the principle that points in pairwise disjoint intervals in real variable functions have a one-to-one correspondence, the values in (λ, 1] correspond one-to-one with those in $\textstyle 1 - \lambda , \lambda ]$ , and thus the expression (a) can be obtained. If $\partial ( \mathbf { A } ) \in [ 0 , 1 - \lambda )$ and $\hat { \cal O } ( { \sim } \mathrm { A } ) { \in } [ 1 { - } \lambda , \lambda ]$ , we can similarly obtain expression (b). When $\lambda \leq 1 / 2 , [ 0 ,$ 1] is divided into three sub-intervals: 0, λ), [λ, 1-λ], (1-λ, 1]. In the same manner, we can establish expressions (c) and (d). For other scenarios $\partial ( { \sim } \mathbf { A } ) =$ (A), which is expression (e).

For cases (a)–(d) in Definition 2, we illustrate them intuitively with the following figure (Figure 7). The symbols “” and $^ { \circ } \mathrm { { o } } ^ { \prime \prime }$ in the figure represent the close endpoint and the open endpoint of an interval, respectively.

![](images/65255c0d957453ead69e222966c18f2616e4cee39be3dae8746abae72d467fa5.jpg)  
Case (a) in the definition 1

![](images/70c54529b63d9e3f7b87b677d77ea9ea30a4f2e88f538b9cd5c1a74888f8832e.jpg)  
Case (b) in the definition 1

![](images/cc6290c7e209bf4343485b996173836ece10dbd64763d2537b786682986354ea.jpg)  
Case (c) in the definition 1

![](images/ce8c1fe65b9225fb6fd9793369db8faf8fa1df7aa0c46a7868a2f8b5e3c9dd0b.jpg)  
Case (d) in the definition 1  
Fig. 7 The interrelationships between (A), (╕A) and $\partial ( { \sim } \mathbf { A } )$

From the figure 7, it can be observed that:

(1) ∂(\~A) serves as an “intermediary" between $\partial ( \mathbf { A } )$ and its opposite $\partial ( \daleth \mathrm { A } )$ , reflecting the important philosophy idea that “all opposing concepts transition to each other through an intermediary between them” [28].

(2) $\lambda { \in } ( 0 , 1 )$ is a variable parameter, it determines the value range of the membership degrees $\partial ( \mathbf { A } ) , \partial ( \mathbf { \vec { q } } \mathbf { A } )$ and $\partial ( { \sim } \mathbf { A } )$ . That is,  is a “threshold” for the value range of these truth values. Its role and significance in practical applications will be discussed in detail in another article (a study on negation detection and negation resolution in medical texts).

As with the operational properties of SCOI [12, 13], it is easy to prove from Definition 2 that the truth values $\partial ( \mathrm { A } ) , \partial ( \neg \mathrm { A } ) , \partial ( \neg \mathrm { A } )$ and $\partial ( { \sim } \mathbf { A } )$ have the following relationships and properties.

Proposition 1. If $\begin{array} { r } { \partial ( \mathbf { A } ) = \% , } \end{array}$ then

$$
\partial ( \neg \mathbf { A } ) = \partial ( \neg \mathbf { A } ) = \partial ( \neg \mathbf { A } ) = 1 / _ { 2 } .
$$

Proposition 2. $\mathrm { I f } \ \lambda \geq \%$ then

$$
\begin{array} { r } { \partial ( \mathbf { A } ) > \partial ( \mathbf { \sim } \mathbf { A } ) > \partial ( \mathbf { \vec { q } } \mathbf { \cdot } \mathbf { A } ) \mathrm { ~ a n d ~ } \partial ( \mathbf {  } \mathbf { A } ) = \partial ( \mathbf { \sim } \mathbf { A } ) \mathrm { , ~ i f ~ a n d ~ o n l y ~ i f ~ } \partial ( \mathbf { A } ) \in ( \lambda , 1 ] . } \end{array}
$$

$$
\begin{array} { r } { \partial ( \neg \mathbf { A } ) > \partial ( \neg \mathbf { A } ) > \partial ( \mathbf { A } ) \mathrm { ~ a n d ~ } \partial ( \neg \mathbf { A } ) = \partial ( \neg \mathbf { A } ) \mathrm { , ~ i f ~ a n d ~ o n l y ~ i f ~ } \partial ( \mathbf { A } ) \in [ 0 , 1 - \lambda ) . } \end{array}
$$

Proposition 3. $\mathrm { I f } \lambda \le { 1 / 2 }$ , then

$$
\begin{array} { r } { \partial ( \mathbf { A } ) > \partial ( \mathbf { \neg { A } } ) > \partial ( \mathbf { q A } ) \mathrm { ~ a n d ~ } \partial ( \mathbf { \neg { A } } ) = \partial ( \mathbf { \neg { A } } ) , \mathrm { i f ~ a n d ~ o n l y ~ i f ~ } \partial ( \mathbf { A } ) \in ( 1 - \lambda , 1 ] . } \end{array}
$$

$$
\begin{array} { r } { \partial ( \daleth \mathbb { A } ) > \partial ( \neg \mathbf { A } ) > \partial ( \mathbf { A } ) \mathrm { ~ a n d ~ } \hat { \mathcal { O } } ( \neg \mathbf { A } ) = \hat { \mathcal { O } } ( \daleth \mathbb { A } ) , \mathrm { i f ~ a n d ~ o n l y ~ i f ~ } \hat { \mathcal { O } } ( \mathbf { A } ) \in [ 0 , \lambda ) . } \end{array}
$$

Proposition 4.

$$
\widehat { \cal O } ( { \sim } \mathrm { A } ) \in [ 1 { - } \lambda , \lambda ] , \mathrm { w h e n } \lambda \geq 1 / 2 .
$$

$$
\partial ( \neg \mathbf { A } ) \in [ \lambda , 1 - \lambda ] , \mathrm { w h e n } \lambda \leq Y _ { 2 } .
$$

The above propositions conveys the following practical significance: for a formula (proposition) A in the logic system LCOI+PLCOI and its contradictory negation $( \neg \mathbf { A } )$ , opposite negation $( \daleth \mathrm { A } )$ and intermediary negation (A), their truth values ${ \hat { \sigma } } ( \mathrm { A } ) , { \hat { \sigma } } ( \lnot \mathrm { A } ) , { \hat { \sigma } } ( \lnot \mathrm { A } )$ and $\partial ( { \sim } \mathbf { A } )$ (values within [0, 1]) relate to each other under different threshold values of .

In the semantics of mathematical logic, the metatheorems (such as the soundness theorem and completeness theorem) revolve around tautologies. “Tautology” (or logical truth) is a core concept. In classical two-valued logic, the truth value of a formula is either 0 or 1. The formula A is a tautology if and only if its truth value is always 1. In non-classical logics like fuzzy logic, since truth values include multiple values such as 0 and 1, the concept of “tautology’ is weakened or modified (meaning A's truth value is not required to always be 1 but is required not to be lower than a certain value within the interval [0, 1]). This reflects different logical systems' interpretations of “logical truth”. Thus, for the logic LCOI+PLCOI, which studies crisp and fuzzy entities, we provide a definition of “-tautology” based on Definition 2.

Definition 3 (λ-tautology). Let Γ be set of λ-assignment of , $\forall \mathbf { A } \in \Im .$ . For any -assignment $\hat { \sigma } \in \Gamma , \mathrm { i f } \hat { \sigma } ( \mathbf { A } ) = 1$ then A is called a tautology. If $\partial ( \mathrm { A } ) \geq \lambda ( \lambda > Y _ { 2 } )$ , A is called a -tautology, and denoted $\models \mathbf { A } .$ . If there exists a -assignment $\hat { o } \in \Gamma$ such that $\partial ( \mathbf { A } ) \geq \lambda ,$ then A is called -satisfiable.

As in the proof methods of mathematical logic, the axioms in a logical system must be proven to be tautologies in order to prove the metatheorems of the logical system, such as the soundness theorem and completeness theorem. Therefore, all axioms in LCOI+PLCOI should be -tautologies. To this end, we need to determine the binary function  in the definition 2.

Definition 4. Let a, b[0, 1]. The mapping $\mathfrak { R } ^ { \mathrm { o } } \colon [ 0 , 1 ] ^ { 2 }  [ 0 , 1 ]$ is , if satisfies:

$$
\mathfrak { R } ^ { \circ } ( { \mathrm { a } } , { \mathrm { b } } ) = 1 , \mathrm { w h e n } { \mathrm { a } } \leq { \mathrm { b } } .\tag{1}
$$

$$
\mathfrak { R } ^ { \scriptscriptstyle 0 } ( { \mathrm { a } } , { \mathrm { b } } ) = \operatorname* { m a x } ( 1 - { \mathrm { a } } , { \mathrm { b } } ) , \mathrm { w h e n ~ } { \mathrm { a } } > { \mathrm { b } } .\tag{2}
$$

It can be easily proven that $\Re ^ { \mathrm { o } }$ has the following properties.

$$
\mathbf { P r o p o s i t i o n 5 . L e t a , b } { \in } [ 0 , 1 ] . \ \mathrm { T h e n }
$$

$$
\mathbf { a } \geq \mathbf { b } , \mathrm { i f ~ a n d ~ o n l y ~ i f ~ } \mathfrak { R } ^ { \circ } ( \mathbf { a } , \mathbf { c } ) \leq \mathfrak { R } ^ { \circ } ( \mathbf { b } , \mathbf { c } ) , \mathrm { f o r ~ a l l ~ } \mathbf { c } \in [ 0 , 1 ] \mathrm { { c } } \cup \Sigma ^ { 0 } ( \mathbf { b } , \mathbf { c } ) .\tag{3}
$$

$$
\mathbf { a } > \mathbf { b } , { \mathrm { i f ~ a n d ~ o n l y ~ i f ~ } } \Re ^ { \circ } ( \mathbf { c } , \mathbf { a } ) \geq \Re ^ { \circ } ( \mathbf { c } , \mathbf { b } ) { \mathrm { ~ } } \mathrm { ~ f o r ~ a l l ~ } \mathbf { c } \in [ 0 , 1 ] { \mathrm { ~ } } \mathrm { { c } } ,\tag{4}
$$

Lemma 1. For LCOI+PLCOI, if A and AB are -tautologies, then B is a -tautology.

Proof: Let $\Re \ : = \ : \Re ^ { \mathrm { o } }$ , A and AB are -tautologies. Suppose B is not a -tautology. Then, according to Definition 2, there exists a -assignment $\beta \in \Gamma , \beta ( \mathbf { B } ) < \lambda .$ . Since A and A→B are λ-tautologies, so there are $\beta ( \mathbf { A } ) \geq$ and $\beta ( \mathrm { A } \to \mathrm { B } ) \geq \lambda . \ \mathrm { B y }$ [6] in the definition, $\displaystyle \{ ( \mathbf { A {  } B } ) = \Re ^ { \circ } ( \{ \mathbf { \beta } ( \mathbf { A } ) , \mathbf { \beta } \} ( \mathbf { B } ) ) \geq \lambda$ . Due to $\beta ( \mathrm { A } ) \geq \lambda$ and $\beta ( { \mathbf B } ) < \lambda$ ，所以 $\beta ( \mathbf { A } ) > \beta ( \mathbf { B } )$ . Because $\lambda > { } ^ { 1 } 2 , \operatorname { s o } \ \Re ^ { 0 } ( \beta ( \mathrm { A } ) , \beta ( \mathrm { B } ) ) = \operatorname* { m a x } ( 1 - \beta ( \mathrm { A } ) , \beta ( \mathrm { B } ) ) < \lambda \ b \mathrm { y }$ (2). That is, it contradicts $\Re ^ { \mathrm { o } } ( \beta ( \mathrm { A } )$ $\beta ( { \mathbf B } ) ) \ge \lambda$ . Therefore, B is a λ-tautology.

Based on Definition 2, let a, b and c represent (A), (B) and (C) respectively. $\operatorname { I f } \ \Re = \Re ^ { \mathrm { o } }$ , we can prove the following conclusion.

Lemma 2. Each axiom in LCOI+PLCOI is a λ-tautology.

Proof: Let $\Re = \Re ^ { \mathrm { o } }$ . According to [6] and Definition 3, the axioms (a1), (a2), and (a3) in LCOI+PLCOI can be expressed as follows:

(a1): $\Re ( { \mathrm { a } } , \Re ( { \mathrm { b } } , { \mathrm { a } } ) ) \geq \lambda , ( \lambda > ^ { 1 / 2 } )$

$$
( { \mathrm { a } } 2 ) \colon \ \mathfrak { R } ( \mathfrak { R } ( { \mathrm { a } } , \mathfrak { R } ( { \mathrm { a } } , { \mathrm { b } } ) ) , \mathfrak { R } ( { \mathrm { a } } , { \mathrm { b } } ) ) \geq \lambda , ( \lambda > ^ { 1 } / 2 )
$$

$$
\begin{array} { r l } { ( { \mathrm { a } } 3 ) \colon } & { { } \Re ( \Re ( { \mathrm { a } } , { \mathrm { b } } ) , \Re ( \Re ( { \mathrm { b } } , { \mathrm { c } } ) , \Re ( { \mathrm { a } } , { \mathrm { c } } ) ) ) \geq \lambda , ( \lambda > ^ { 1 } / { 2 } ) } \end{array}
$$

For (a1). (i) If ${ \mathrm {  ~ a ~ } } \leq { \mathrm {  ~ b ~ } } .$ , then $\Re ( { \mathrm { a } } , \Re ( { \mathrm { b } } , { \mathrm { a } } ) ) = \Re ( { \mathrm { a } } , \operatorname* { m a x } ( 1 { - } { \mathrm { b } } , { \mathrm { a } } ) )$ by the definition 4, where if $1 { - } \mathbf { b } > \mathbf { a }$ , then (a, max(1−b, a)) = R(a, 1−b) = 1 ≥ λ; if $1 - \mathbf { b } \leq \mathbf { a } ,$ , then $\Re ( { \mathrm { a } } , \operatorname* { m a x } ( 1 - { \mathrm { b } } , { \mathrm { a } } ) ) = \Re ( { \mathrm { a } } , { \mathrm { a } } ) = 1 \geq \lambda$ . (ii) If $\mathbf { a } > \mathbf { b } ,$ , then $\Re ( \mathfrak { b } , \mathfrak { a } ) \geq$ (b, b) according to (4), $\Re ( \mathbf { b } , \mathbf { a } ) = 1$ by (1). So R(a, R(b, a)) = R(a, 1). R(a, R(b, a)) = 1 by (1), i.e. R(a, R(b, a)) $\geq \lambda .$ . Therefore, the axiom (a1) is a λ-tautology by (i) and (ii).

For (a2). (i) If a ≤ b, then R(a, R(a, b)) ≥ R(b, R(a, b)) by (3). According to (1), R(a, b) = 1. Hence, $\Re ( \Re ( \mathfrak { a } ,$ R(a, b)), R(a, b)) = R(R(a, 1), 1). R(R(a, 1), 1) = R(1, 1) = 1 by (1), i.e. R(R(a, R(a, b)), $\Re ( { \mathrm { a } } , { \mathrm { b } } ) ) \geq \lambda . ( { \mathrm { i i } } )$ If a  b, based on the above proof, it is only necessary to prove R(a, R(a, b)) ≤ R(a, b).

Suppose $\Re ( { \mathrm { a } } , \Re ( { \mathrm { a } } , { \mathrm { b } } ) ) > \Re ( { \mathrm { a } } , { \mathrm { b } } ) . \Re ( { \mathrm { a } } , { \mathrm { b } } ) >$ b according to (4). $\Re ( { \mathrm { a } } , { \mathrm { b } } ) = \operatorname* { m a x } ( 1 - { \mathrm { a } } , { \mathrm { b } } ) > { \mathrm { b } }$ by (2). Hence, $\Re ( \mathrm { a } , \mathrm { b } )$ ${ \bf \omega } = 1 - { \bf a } .$ Substituting the hypothesis, there is $\Re ( { \mathrm { a } } , 1 - { \mathrm { a } } ) > 1 - { \mathrm { a } }$ , where if ${ \bf a } \le 1 - { \bf a }$ , then $\Re ( { \mathrm { a } } , 1 - { \mathrm { a } } ) = 1$ by (1); if ${ \mathrm { a } } > 1 { - } { \mathrm { a } } ,$ then $\Re ( { \mathrm { a } } , 1 - { \mathrm { a } } ) = \operatorname* { m a x } ( 1 - { \mathrm { a } } , 1 - { \mathrm { a } } ) = 1 - { \mathrm { a } } . \Re ( { \mathrm { a } } , 1 - { \mathrm { a } } ) = 1 - { \mathrm { a } }$ contradicts $\Re ( { \mathrm { a } } , 1 - { \mathrm { a } } ) > 1 - { \mathrm { a } }$ . Thus, $\Re ( { \mathrm { a } } , \Re ( { \mathrm { a } } , { \mathrm { b } } ) ) \leq \Re ( { \mathrm { a } } , { \mathrm { b } } )$ holds true.

Therefore, the axiom (a2) is a λ-tautology by (i) and (ii).

For (a3). According to (1), only necessary to prove $\Re ( { \mathrm { a } } , { \mathrm { b } } ) \leq \Re ( \Re ( { \mathrm { b } } , { \mathrm { c } } ) , \Re ( { \mathrm { a } } , { \mathrm { c } } ) )$ . Suppose $\Re ( { \mathrm { a } } , { \mathrm { b } } ) > \Re ( \Re ( { \mathrm { b } } , { \mathrm { c } } )$ $\Re ( { \mathrm { a } } , { \mathrm { c } } ) )$ . From this, $\Re ( \Re ( { \boldsymbol { \mathrm { b } } } , { \boldsymbol { \mathrm { c } } } ) , \Re ( { \boldsymbol { \mathrm { a } } } , { \boldsymbol { \mathrm { c } } } ) ) \neq 1$ . According to (1), so

$$
\Re ( { \boldsymbol { \mathrm { b } } } , { \boldsymbol { \mathrm { c } } } ) > \Re ( { \mathrm { a } } , { \boldsymbol { \mathrm { c } } } )\tag{5}
$$

Thus, $\Re ( { \mathrm { a } } , { \mathrm { c } } ) \neq 1 . { \mathrm { a } } > { \mathrm { b } }$ by (3), a  c by (1). According to (2), $\Re ( { \mathrm { a } } , { \mathrm { c } } ) = \operatorname* { m a x } ( 1 - { \mathrm { a } } , { \mathrm { c } } )$ . Substituting (5), $\Re ( { \mathfrak { b } } , { \mathfrak { c } } ) >$ $\operatorname* { m a x } ( 1 { - } \mathrm { a } , \mathrm { c } ) , \mathrm { i . e . \ b \leq c \ o r \ b > c } .$ , there is $\Re ( \mathrm { b } , \mathrm { c } ) > \operatorname* { m a x } ( 1 - \mathrm { a } , \mathrm { c } )$

${ \mathrm { I f ~ b } } \leq { \mathrm { c } } , { \mathfrak { R } } ( { \mathrm { b } } , { \mathrm { c } } ) = 1$ according to (1). Hence, $\Re ( { \mathrm { b } } , { \mathrm { c } } ) > \operatorname* { m a x } ( 1 - { \mathrm { a } } , { \mathrm { c } } ) .$

If b $\mathfrak { r } > \mathrm { c } , \mathfrak { R } ( \mathrm { b } , \mathrm { c } ) = \mathrm { m a x } ( 1 - \mathrm { b } , \mathrm { c } )$ according to (2). Because of $a > \mathbf { b } , \mathrm { { i . e . \ l - a < 1 - b . } }$ , Hence, $\Re ( { \boldsymbol { \mathrm { b } } } , { \boldsymbol { \mathrm { c } } } ) = \operatorname* { m a x } ( 1 { - } { \boldsymbol { \mathrm { b } } } , { \boldsymbol { \mathrm { c } } } )$ $> \operatorname* { m a x } ( 1 - \mathrm { a } , \mathrm { c } )$ . However, if $1 { \mathrm { - b } } \leq { \mathrm { c } } ,$ , then max(1b, c) = c and $\operatorname* { m a x } ( 1 - \mathrm { a } , \mathrm { c } ) = \mathrm { c } .$ , it contradicts max $( 1 { - } \mathsf { b } , \mathsf { c } ) >$ $\mathrm { m a x } ( 1 { - } \mathrm { a } , \mathrm { c } )$ . So, only when b  c and $1 - \boldsymbol { \mathbf { b } } > \boldsymbol { \mathbf { c } } , \Re ( \boldsymbol { \mathbf { b } } , \boldsymbol { \mathbf { c } } ) > \operatorname* { m a x } ( 1 - \boldsymbol { \mathbf { a } } , \boldsymbol { \mathbf { c } } )$

Substitute a > b, b > c, 1−b > c into the suppose: R(a, b) > R(R(b, c), R(a, c)), then max(1−a, b) > max $( 1 - \operatorname* { m a x } ( 1 - \mathbf { b } , \mathbf { c } )$ , max(1a, c)) = max(b, max(1a, c)) by (3). However, when $1 - \mathbf { a } < \mathbf { b } ,$ there is max $( 1 - \mathsf { a } , \mathsf { b } ) = \mathsf { b } >$ max(b, max(1a, c)) = b, the two are contradictory; when $1 - \mathbf { a } \geq \mathbf { b }$ , there is max(1−a, b) = 1−a > max(b, max(1−a, ${ \bf c } ) ) = 1 - { \bf a }$ , the two are contradictory. Thus, the assumption (a, ${ \mathfrak { b } } ) > \mathfrak { R } ( \mathfrak { R } ( { \mathfrak { b } } , \mathrm { c } ) , \mathfrak { R } ( \mathrm { a } , \mathrm { c } ) )$ is not valid. Therefore, the axiom (a3) is a λ-tautology.

Similarly, it can be proven that if $\Re = \Re ^ { \mathrm { o } }$ , then the axioms $( { \mathrm { a 4 } } ) - ( { \mathrm { a 1 8 } } )$ in LCOI+PLCOI are all λ-tautologies.  Based on the above results, similar to the proof methods (method of induction) for soundness in mathematical logic, we can prove the following soundness theorem for LCOI+PLCOI.

Theorem 1 (Soundness theorem). Let $\phi \left( \phi \subseteq { \mathfrak { I } } \right)$ be a set of formulas in LCOI+PLCOI and A be a formula in LCOI+PLCOI.

(a) $\mathrm { ~ I f ~ } \big | \vdash \mathbf { A } ,$ then $\models \mathbf { A } .$

(b) ${ \mathrm { I f ~ } } \phi { \big | } - { \mathrm { A , t h e n } } \phi { \big | } \in { \mathrm { A } }$

Proof: $\mathrm { ~ I f ~ } \big | \vdash \mathbf { A } ,$ then A is provable in LCOI+PLCOI. According to the definition 1, Induct on the length n of the sequence of formulas $A _ { I } , A _ { 2 } , . . . , A _ { n }$ for the proof of A.

(i) When $n = 1$ , then according to Definition 1 in Section 5.3, A<sub>1</sub> (which is A) is an axiom in LCOI+PLCOI. By Lemma 2, A is a λ-tautology.

(ii) Suppose the theorem holds for $k < n$ . That is, all proof sequences of A with fewer than n steps are λ-tautologies. We prove that the theorem holds when $n = k .$ . According to Definition 1 in Section 5.3, there are two cases: $( 1 ) A _ { n }$ is an axiom of LCOI+PLCOI, or $( 2 ) A _ { n }$ is a formula deduced from A and A and $A _ { j } ( i < n , j < n )$ using the deduction rule [D2] in LCOI+PLCOI. If it is case (1), then it is similar to the proof of (i). If it is case (2), the formal expressions of the formulas $A _ { i }$ and $A _ { j }$ must be B and BA. According to the hypothesis, B and BA are λ-tautologies. Therefore, by Lemma 1, $\mathbf { A } \ ( \mathrm { i . e . } A _ { n } )$ is a λ-tautology. According to the principle of mathematical induction and Definition 3, ╞ A holds.

When Φ is the empty set, (b) is equivalent to (a). Therefore, (b) can be proven similarly. 

We need to point out that when $\Re = \Re ^ { \mathrm { o } } ,$ , it can be verified that the completeness theorem for LCOI+PLCOI does not hold. Whether there exists a specific  such that the completeness theorem for LCOI+PLCOI holds will be discussed in another article.

## 5. Triple with three types of negation

In Section 3, we conceptually propose that there are three different types of negation within triples and their elements: contradictory negation, opposite negation, and intermediary negation. Therefore, it is necessary to extend the concept of triples based on a logical system that can distinguish and express different negations both syntactically and semantically.

In this section, we distinguish triples into two types and discuss the basic characteristics of their negations.

Based on the set SCOI and the logic LCOI+PLCOI, we extend the triple and propose a triple that can fully express the three different types of negation of triple and their elements: “TCOI triple with contradictory negation, opposite negation and intermediary negation”.

## 5.1 Two Types of Triples and Their Negation Characteristics

For the triple <s, p, o>, from the perspective of structure and relations, it expresses the relationship established between the subject s and the object o through the predicate p. A triple represents a statement or an assertion, and the semantics of the statement is determined by the meanings of its elements. Since statements described in natural language can be distinguished into clear statements and fuzzy statements, triples should be classified into two categories: clear triple and fuzzy triple. Clear triple represent true/false statements, with the truth value domain of the statements being {0, 1}. Fuzzy triple represent statements that can be partially true or false, with the truth value domain being [0, 1].

Clear triple: In a triple <s, p, o>, if the elements s, p and o are clearly expressed (with clear meanings) and have no ambiguity within a specific domain, the triple is called a clear triple. A clear triple expresses a clear statement. For example, the triples <USA, president, Trump> and <constant x, nature, positive integer> are considered clear triples because all elements in these triples are clear concepts. They express clear statements: “the president of the USA is Trump” and “the constant x is a positive integer”.

Fuzzy triple: In a triple <s, p, o>, if one or more elements withen s, p and o are ambiguously expressed (with unclear meanings), the triple is called a fuzzy triple. A fuzzy triple expresses a fuzzy statement. For example, <tall person, likes, basketball> (here, the subject ‘tall person’ is a fuzzy concept), <Beijing, temperature, cold> (here, the object ‘cold’ is a fuzzy concept), and <Xiaoming, good at, programming> (here, the predicate ‘good at is a fuzzy concept). These express fuzzy statements: ‘tall people like basketball”, “the temperature in Beijing is cold” and “Xiaoming is good at programming”.

In short, for a triple, if all elements in the triple are clearly expressed, then the triple is a clear triple. If any element is expressed fuzzily, then the triple is a fuzzy triple.

Negation in triple and its application in knowledge representation and reasoning hold significant importance. It can be said that negation in triple is the indispensable "other half" in knowledge representation. It enables systems (e.g., Resource Description Frameworks, knowledge graphs) to extend from only describing what exists to also describing what does not exist, thus making logic more complete, knowledge more precise, and reasoning deeper. The importance of negation for triple is mainly reflected in the following aspects: (1) Enhancing data expressiveness and information integrity: Negation allows us to express the opposite of facts. (2) Supporting complex reasoning: Negation helps knowledge systems identify and handle counterexamples, enabling effective derivation. (3) Handling contradictions and consistency checks: Negation aids in identifying potential contradictions in information within the knowledge system. Through negation, conflicts in the system's knowledge can be detected and resolved, ensuring the accuracy of the knowledge system. (4) Expressing uncertainty and complex scenarios: Negation assists the system in handling more finely complex conditions and diverse data, such as expressing fuzzy queries or commands through negation.

From the perspective of the logical foundations of triple, classical logic and fuzzy logic form the basis for the formal syntax and fundamental deductive rules of clear (fuzzy) triple and their negations. Essentially, clear (fuzzy) triple correspond respectively to atomic propositions in classical predicate logic and fuzzy predicate logic. Since the formal languages of classical (fuzzy) predicate logic include only one kind of negation (classical negation), the negations within triple and their elements can only be expressed using classical negation in terms of logical syntax. The syntax and semantics of classical (fuzzy) first-order predicate logic determine this logica characteristic of clear (fuzzy) triple.

From the perspective of the semantic level of the triple structure, the negativity of clear and fuzzy triple exists in two forms: negation at the element level and negation at the triple level. In practice, these two forms of negation are not limited to a single type but involve three distinct types of negation with different meanings.

Example 1. In triple expressions, for the clear triple <x, property, positive number> (i.e., the statement “x is a positive number”), the triples < x, property, non-positive number > (“x is not a positive number”), < x, property, negative number> (“x is a negative number”) and <x, property, zero> (“x is zero”) represent its three different types of negation. For the fuzzy triple <Beijing, temperature, cold> (i.e., the statement “the temperature in Beijing is cold”), the triples <Beijing, temperature, not cold>, <Beijing, temperature, hot>, and <Beijing, temperature, warm> represent its three different types of negation.

Example 2. In element expression, for the subject “red” in the clear triple <red, is, color>, with the color relationships of the artistic color wheel (RYB) as the background, ‘non-red’, ‘green’ and ‘yellow’ represent three different types of negation for the subject “red”. For the object “handsome” in the fuzzy triple <Chaplin, appearance, handsome>, ‘not handsome’, ‘ugly’ and ‘ordinary’ represent three different types of negation for the object.

For the three distinct forms of negation present in clear and fuzzy triples and their elements, classical logic and fuzzy logic, which only have one form of negation (classical negation) in formal language, cannot distinguish and express them. Therefore, to accurately and precisely express these different negations in knowledge representation, the classical triple <s, p, o> needs to be extended based on a non-classical or extended logic that can distinguish and express different negations at both the syntactic and semantic interpretation levels. Such logic includes multiple negation operators or semantic clarification mechanisms to achieve complete expression of negation meanings and accurate semantic modeling.

## 5.2 Triple with contradictory negation, opposite negation and intermediary negation

The intuitive meaning of a triple is the relationship established between the subject and the object as defined by the property, and the uncertainty within a triple is inevitably reflected by the uncertainty that exists in the relationship between subject and object [29]. Therefore, establishing the relationship between the subject and object is key in the clear and fuzzy triples.

For the clear triple and fuzzy triple the relationship between the subject and the object is distinguished as follows:

<sup></sup> If both the subject in the subject domain and the object in the object domain are clearly expressed (with clear meanings), then the relationship between the subject and the object is a binary clear relation: a composite mapping from the subject domain to the object domain, and then to {0, 1}.

<sup></sup> If either the subject in the subject domain or the object in the object domain, or both, have fuzzy expressions (with fuzzy meanings), then the relationship between the subject and the object is a binary fuzzy relation: a composite mapping from the subject domain to the object domain, and then to [0, 1].

The above two binary relations can be mathematically expressed as follows.

Let X be a subject domain and Y be an object domain.

(1) For xX, if exists yY and y is clear expression, then relationship between x and y is a composite mapping from X to Y and Y to $\{ 0 , 1 \} , \mu = f _ { 2 } \circ f _ { I } . f _ { I } : X \longrightarrow Y ; f _ { 2 } \colon Y \longrightarrow \{ 0 , 1 \} . \ \mu ( x ) = f _ { 2 } ( y ) = f _ { 2 } ( f _ { I } ( x ) ) \in \{ 0 , 1 \} .$

(2) For xX, if exists yY and y is fuzzy expression, then relationship between x and y is a composite mapping from X to Y and Y to $[ 0 , 1 ] , \mu = f _ { 2 } \circ f _ { I } . f _ { I } : X \longrightarrow Y ; f _ { 2 } \colon Y \longrightarrow [ 0 , 1 ] . \mu ( x ) = f _ { 2 } ( y ) = f _ { 2 } ( f _ { I } ( x ) ) \in [ 0 , 1 ] .$

It can be seen that  expresses the relationship between a subject x and object y, (x) reflects the relatedness between x and y, which we refer to as the “correlation degree” between x and y.

For $\mu ( x )$ , we let the mapping  in the definition of the set SCOI (Definition 1 in Section 4.1) be . Thus, in SCOI, the membership degree y(x) of x to y is the correlation degree (x) between x and y. In other words, the correlation degree $\mu ( x )$ between a subject x and an object y can be obtained by solving the membership degree y(x)

in the set SCOI.

According to the above, based on the set SCOI and the logic LCOI+PLCOI, we propose a triple that can fully express the three different types of negation of triple and their elements: “Triple with contradictory negation, opposite negation and intermediary negation”.

Definition 1 (TCOI triple). Let S be a subject domain, O be an object domain, and P be set of attributes. For any sS, $p { \in } P$ and $o \in O$ , triple:

$$
< s * , p * , ( o * , \mu _ { o * } ( s * ) ) >
$$

is called the triple with contradictory negation, opposite negation and intermediary negation, for short TCOI triple. It expresses a statement (assertion): $^ { \circ } s ^ { * }$ and $o ^ { * }$ have a relation expressed by $p ^ { * }$ with a correlation degree $\mu _ { o * } ( s * ) ^ { * }$ Where, $s ^ { * } \in \{ s , \  s , \_ { \exists } s , \ r , s \} \subseteq S , p ^ { * } \in \{ p , \ \lnot p , \_ { \exists } p , \ r p \} \subseteq P , \ o ^ { * } \in \{ o , \ \lnot \sigma , \_ { \exists } \ q , \ r \sigma \} \subseteq O . \ \lnot s , \ \lnot p { \mathrm { ~ a n d ~ } } \lnot p = 0 .$ respectively are contradictory negation of $s , p$ and $o ; \daleth s , \daleth p$ and $\daleth ^ { o }$ respectively are opposite negation of $s , p$ and $o ; \sim s , \sim p$ and \~o respectively are opposite negation of $s , p$ and o. $\mu _ { o * } ( s ^ { * } )$ is correlation degree betweens $s *$ to $^ { O ^ { * } }$

In the TCoI triple, if s\*, p\* and $^ { O ^ { * } }$ are all clearly expressed, then TCOI triple is a clear triple and $\mu _ { o * } ( s * ) \in \{ 0 , 1 \}$

If there exists a fuzzy expression in $s ^ { * } , p ^ { * }$ and $^ { o * }$ , then TCOI triple is a fuzzy triple and $\mu _ { o * } ( s * ) \in [ 0 , 1 ]$

The TCOI triple is defined based on the set SCOI and the logic LCOI+PLCOI with three kinds of negation. It is capable of expressing three different negations of itself and its elements. Within the framework of the set SCOI and the logic LCOI+PLCOI, for the three negations of the TCOI triple and their meanings, as well as the triple representations for the three negations of elements and their meanings, we discuss as follows:

(i) For the three negations of the TCOI triple, their representations and meanings are as follows:

$\neg < s * , p * , ( o * , \mu _ { o * } ( s * ) ) > :$ the contradictory negation of $< s * , p * , ( o * , \mu _ { o * } ( s * ) ) >$

$\exists < s * , p * , ( o * , \mu _ { o * } ( s * ) ) > :$ the opposite negation of <s, $p ^ { * } , ( o ^ { * } , \mu _ { o ^ { * } } ( s ^ { * } ) ) >$

$\sim < s * , p * , ( o * , \mu _ { o * } ( s * ) ) > :$ the intermediary negation of $< s * , p * , ( o * , \mu _ { o * } ( s * ) ) >$

(ii) For the subject s and its three negations s, ╕s and ${ \sim } s ,$ the TCOI triple representation and its meaning are as follows:

$$
< s , p * , ( o * , \mu _ { o * } ( s ) ) > :
$$

s and $o ^ { * }$ have a relation expressed by $p ^ { * }$ with a correlation degree $\mu _ { o * } ( s )$

$$
< \neg s , p * , ( o * , \mu _ { o * } ( \neg s ) ) > :
$$

the contradictory negation s of s and $o ^ { * }$ have a relation expressed by $p ^ { * }$ with a correlation degree $\mu _ { o * } ( \ - \to S )$

$$
< \daleth s , p * , ( o * , \mu _ { o * } ( \daleth s ) ) >
$$

the opposite negation ╕s of s and $o ^ { * }$ have a relation expressed by $p ^ { * }$ with a correlation degree $\mu _ { o * } ( \daleth s )$

$$
< \sim s , p * , ( o * , \mu _ { o * } ( \sim s ) ) >
$$

the intermediary negation \~s of s and $o ^ { * }$ have a relation expressed by $p ^ { * }$ with a correlation degree $\mu _ { o * } ( \sim s )$

(iii) For the predicate p and its three negations $\neg p , \neg p$ and ${ \sim } p .$ , the TCOI triple representation and its meaning are as follows:

$$
< s * , p , ( o * , \mu _ { o * } ( s * ) ) > ;
$$

$s ^ { * }$ and $o ^ { * }$ have a relation expressed by p with a correlation degree $\mu _ { o * } ( s ^ { * } )$

$$
< s * , \lnot p , ( o ^ { * } , \mu _ { o ^ { * } } ( s ^ { * } ) ) >
$$

$s ^ { * }$ and $o ^ { * }$ have a relation expressed by the contradictory negation p of p with a correlation degree $\mu _ { o * } ( s ^ { * } )$

$$
< s * , \daleth p , ( o ^ { * } , \mu _ { o * } ( s ^ { * } ) ) >
$$

$s ^ { * }$ and $o ^ { * }$ have a relation expressed by the opposite negation ╕p of p with a correlation degree $\mu _ { o * } ( s * )$

$$
< s * , \sim p , ( o * , \mu _ { o * } ( s * ) ) > :
$$

$s ^ { * }$ and $o ^ { * }$ have a relation expressed by the intermediary negation ${ \sim } p$ of p with a correlation degree $\mu _ { o * } ( s * )$

(iv) For the object o and its three negations $\neg o , \daleth ^ { o }$ and ${ \sim } O ,$ the TCOI triple representation and its meaning are as follows:

$$
< s * , p * , ( o , \mu _ { o } ( s * ) ) > :
$$

$s ^ { * }$ and o have a relation expressed by $p ^ { * }$ with a correlation degree $\mu _ { o } ( s ^ { * } )$

$$
< s * , p * , ( \neg o , \mu \neg o ( s * ) ) > :
$$

$s ^ { * }$ and the contradictory negation $\multimap$ of o have a relation expressed by $p ^ { * }$ with a correlation degree $\mu _ { \neg o } ( s ^ { * } )$ $< s * , p * , ( \daleth o , \mu _ { \daleth o } ( s * ) ) > :$

$s ^ { * }$ and the opposite negation o of o have a relation expressed by $p ^ { * }$ with a correlation degree $\mu _ { \mp \sigma } ( s ^ { * } )$ $< s * , p * , ( \sim o , \mu _ { \sim o } ( s * ) ) > :$

$s ^ { * }$ and the intermediary negation $\multimap$ of o have a relation expressed by $p ^ { * }$ with a correlation degree $\mu _ { \sim o } ( s ^ { * } )$

We should point out that according to the semantics of the TCOI triple, the meanings of the three negations of a TCOI triple correspond respectively to the meanings of the triples with the three negations of its object o. That is, (i) is the same as the corresponding triplet in (iv).

(a) The contradictory negation of the TCOI triple is the same as the triple of the contradictory negation of its object o.

$$
\lnot < s ^ { * } , p ^ { * } , ( o ^ { * } , \mu _ { o ^ { * } } ( s ^ { * } ) ) > = < s ^ { * } , p ^ { * } , ( \lnot o , \mu _ { \lnot o } ( s ^ { * } ) ) >
$$

(b) The opposite negation of the TCOI triple is the same as the triple of the opposite negation of its object o.

$$
\daleth < s ^ { \ast } , p ^ { \ast } , ( o ^ { \ast } , \mu _ { o ^ { \ast } } ( s ^ { \ast } ) ) > = < s ^ { \ast } , p ^ { \ast } , ( \exists o , \mu _ { \daleth { \boldsymbol { o } } } ( s ^ { \ast } ) ) >
$$

(c) The intermediary negation of the TCOI triple is the same as the triple of the intermediary negation of its object o.

$$
\sim < s ^ { \ast } , p ^ { \ast } , ( o ^ { \ast } , \mu _ { o ^ { \ast } } ( s ^ { \ast } ) ) > = < s ^ { \ast } , p ^ { \ast } , ( \sim o , \mu _ { \sim o } ( s ^ { \ast } ) ) >
$$

For example, for a TCOI triple: <integer x, property, (positive integer, $\mu _ { o } ( x ) ) ^ { > }$ (the statement “integer x is a positive integer”), its contradictory negation (), opposite negation $( \daleth )$ and intermediary negation (\~) are represented respectively as follows:

(1)  <integer x, property, (positive integer, $\mu _ { o } ( x ) ) >$

(2) ╕<integer x, property, (positive integer, $\mu _ { o } ( x ) ) >$

(3) \~ <integer x, property, (positive integer, $\mu _ { o } ( x ) ) >$

In this TCOI triple, for the object “positive integer”, the terms ‘non-positive integer’, ‘negative integer’ and ‘zero’ are its contradictory negation, opposite negation and intermediary negation, respectively. The triples with these three negations as the object and the statements they express are as follows:

(4) <x, property, (non positive integer, $\mu _ { \neg o } ( x ) ) >$ : “integer x is not a positive integer”

(5) <x, property, (negative integer, $\mu _ { \mp \ o } ( x ) ) { \ > } :$ “integer x is a negative integer”

(6) $\mathbf { < } x ,$ property, (zero,  <sub>\~o</sub>(x))>: “integer x is zero”

It can be seen that among the above six triples, the meanings of (1) and (4), (2) and (5), and (3) and (6) are the same.

For the correlation degree $\mu _ { o * } ( s ^ { * } )$ in the TCOI triple, it can be determined either by the λ-assignment $\hat { \sigma }$ of the continuous value semantics in the logic LCOI+PLCOI (Definition 2 in Section 4.2.2), or by the membership degree in the definition of the set SCOI (Definition 1 in Section 4.1). The correlation degree $\mu _ { o * } ( s * )$ reflects the extent to which there is a relationship between the subject and the object.

Example 3. For the clear statement “the president of the United States is Trump”, its TCOI triple can be expressed as <U.S, president, (Trump, 1)>. Here $\mu _ { o } ( s ) = 1$ , indicating that Trump's relevance to the president of U.S is at its highest. For the fuzzy statement “Churchill is somewhat overweight”, its TCOI triple can be expressed as <Churchill, body type, (overweight, 0.8)>. Here $\mu _ { o } ( s ) = 0 . 8 ,$ indicating that Churchill’s relevance to being overweight is 0.8, meaning that the degree of Churchill’s overweight is 0.8.

From the above, it is evident that the syntactic form and semantics of the TCOI triple differ from those of a general triple. TCOI not only distinguishes between crisp triples and fuzzy triples as well as the relationships and degrees of association between the subject and the object, but also is capable of expressing contradictory negation, opposite negation, and intermediary negation in the triple and its elements.

TCOI triple is a semantic and structural extension of the classical triple <s, p, o>. While retaining the ability to express affirmative assertions, it systematically introduces the three semantic dimensions of contradictory negation, opposite negation and intermediary negation, allowing these negations to independently apply to the elements $( \mathbf { s } , \mathbf { p } , \mathbf { o } )$ of the triple and on the whole triple. This significantly enhances the triple model capability to represent and reasoning about complex negative information.

We believe that, for the classical triple, the TCOI triple extends it in three dimensions: semantic extensibility, structural extensibility, and logical positioning. It has the following characteristics:

<sup></sup> The TCOI triple not only retains the expressive capability of the classical triple but also additionally introduces three mutually independent and semantically heterogeneous negation dimensions. This constitutes a substantial semantic extension.

<sup></sup> The TCOI triple appends negation type labels onto the atomic structure of the classical triple, expanding it from an “unmarked affirmative structure” to a “multivariate structure with negation markings”. This is a crisp structural extension.

<sup></sup> The TCOI triple extends the classical triple, which is based on classical logic (with only one classical negation in its formal language), to a foundation based on the logic LCOI+PLCOI (whose formal language and semantics include contradictory negation, opposite negation and intermediary negation).

## 6. Expression ability of TCOI triple

“Negation” is an important and universal phenomenon in language, and a necessary feature of every human language [1]. In this section, we demonstrate the expressive power of the TCOI triple using examples of everyday sentences and Web resource description statements that involve different kinds of negation.

(I) Expression of different negations in everyday statements

Example 1. In everyday statements, the following types of sentences are common:

(1) Clear statements: “the integer x is neither a positive integer nor a negative integer”, “this profession is neither safe nor dangerous”, “this country belongs to neither the East nor the West”, and so on.

(2) Fuzzy statements: “S is neither tall nor short”, “my father is neither fat nor thin”, “there is neither more nor less salt in the dish”, and so on.

Obviously, this type of statement contains more complex negativity.

We take the statement “S is neither tall nor short” from (2) as an example, which is denoted as Ex.

Ex is a composite statement consisting of an atomic statement and its different negations. Among them, the atomic statement is “S is a tall person”. Based on the logic LCOI+PLCOI and the TCOI triple, this atomic statement and its different forms of negation expressions as well as the triple representations are as follows:

${ \sf A } \colon { } ^ { \mathfrak { a } } { \sf S }$ is a tall person” (i.e., an atom formula A in LCOI+PLCOI)

$$
\mathrm { < S , h e i g h t , ( t a l l , } \mu _ { o } ( \mathrm { S } ) ) >
$$

$\neg \mathrm { A } \colon \mathit { \Omega } ^ { \mathit { \Omega } \mathit { \hookrightarrow } } \mathrm { S }$ is not a tall person” (i.e., the contradictory negation A of A)

$$
\mathrm { < S , h e i g h t , ( n o t t a l l , } \mu _ { \neg o } ( \mathrm { S } ) ) >
$$

╕ ${ \mathrm { A } } ; { } ^ { \infty } { \mathrm { S } }$ is a short person” (i.e., the opposite negation ╕A of A)

$$
\mathrm { < S , h e i g h t , } ( \mathrm { s h o r t } , \mu _ { \daleth { o } } ( \mathrm { S } ) ) >
$$

${ \sim } \mathrm { A } \colon ^ { \infty } \mathrm { S }$ is medium height” (i.e., the intermediary negation \~A of A)

$$
< \mathrm { S } , \mathrm { h e i g h t } , ( \mathrm { m e d i u m } , \mu _ { \sim o } ( \mathrm { S } ) ) >
$$

$\lnot \ 7 \mathrm { A } \colon \mathrm { } ^ { \infty } \mathrm { S }$ is not a short person” (i.e., the contradictory negation  ╕A of ╕A)

$$
\ < \mathrm { S } , \mathrm { h e i g h t } , ( \mathrm { n o t ~ s h o r t } , \mu _ { \lnot = \lnot \circ } ( \mathrm { S } ) ) >
$$

Therefore, according to the meaning of Ex, Ex is the conjunction of A and  ╕A. That is

$$
E x = \lnot \mathbf { A } \land \lnot \exists \mathbf { A }
$$

According to axiom (a13) in logic LCOI+PLCOI (Definition 1 in Section 4.2.1):

$$
{ \sim } \mathrm { A } \to { \neg } \mathrm { A } \land { \neg } \ q \mathrm { A } , { \neg } \mathrm { A } \land { \neg } \ q \mathrm { A } \to { \sim } \mathrm { A }
$$

$\mathrm { \sim A a n d \mathrm { \neg A \land \{ \neg \exists A n d \neg A \land \{ \neg \exists A } }  $ mutually implication each other and are logically equivalent. That is,

$$
{ \sim } \mathbf { A } \cong \neg \mathbf { A } \land \neg \exists \mathrm { \AA }
$$

This indicates that the intermediary negation \~A of A and Ex have the same meaning. That is, the statement “S is neither tall nor short” has the same meaning as the statement $^ { \circ \circ } \mathrm { S }$ is medium height”.

Regarding the statement Ex and the different negations of Ex and their relationships, we can illustrate them as follows (Fig. 8):

![](images/64601d2cb05fad0412dc7757d85ba8ad27ecd08174559f5842baee6e9cdba7de.jpg)  
Fig. 8 The statement Ex and the different negations within Ex and their relationships

Above, taking the fuzzy statement “S is neither tall nor short” in (2) as an example, we demonstrated that this statement can be expressed by the “intermediary negation” of the TCOI triple ${ < } S ,$ height, (tall, $\mu _ { o } ( \mathbf { S } ) ) =$ >, namely <S, height, (medium, $\mu _ { \sim o } (  { \mathrm { S } } ) )  { \mathrm { > } }$ . For the clear statement in (1), similarly, based on the logic $\mathrm { L C O I + P L C O I } ,$ expressing it with a clear TCOI triple leads to conclusions analogous to those above.

Thus, it can be seen that for the more complex negative statements of types (1) and (2), whether clear or fuzzy, they can all be expressed by an intermediary negation of a TCOI triple.

(II) Expression of different negations and fuzziness in Web resource description statements

Example 2. For the web resource: “Peking University Professor Li Song” (abbreviated as “Li”), use TCOI triple to describe his appearance.

Clearly, the attributes of a person include “date of birth”, “home address”, “marital status” and “appearance” etc. For the attribute of “appearance”, its attribute values include ‘beautiful’, ‘not beautiful’, ‘ugly’ and ‘ordinary’. It is evident that these attribute values represent different fuzzy concepts (fuzzy sets).

According to the set SCOI and the logic LCOI+PLCOI, the attribute values ‘not beautiful’, ‘ugly’ and ‘ordinary’ are respectively the contradictory negation, opposite negation and intermediary negation of ‘beautiful’. Therefore, the appearance of the resource “Li” can be represented by the following four TCOI triple:

(1) < Li, appearance, (beautiful, $\mu _ { o } ( \mathrm { L i } ) ) >$

(2) < Li, appearance, (unbeautiful, $\mu _ { \neg { o } } ( \mathrm { L i } ) ) >$

(3) < Li, appearance, (ugly, $\mu _ { \mp \ o } ( \mathrm { L i } ) ) >$

(4) < Li, appearance, (ordinary,  <sub>\~o</sub>(Li))>

According to the definition of SCOI (Definition 1 in Section 4.1), if assuming $\lambda = 0 . 6 $ , and the membership degree of $^ { \circ \varsigma } \mathrm { L i } ^ { \prime \prime }$ to fuzzy set ‘beautiful’ is 0.9 (i.e., correlation degree $\mu _ { o } ( \mathrm { L i } )$ of resource $^ { \mathfrak { c } \mathfrak { c } } \mathrm { L i } ^ { \mathfrak { s } }$ to attribute value ‘beautiful’ is 0.9), it can be calculated from the definition of SCOI that the correlation degree of resource $^ { \circ \varsigma } \mathrm { L i } ^ { \prime \prime }$ to attribute value $\mathrm {  { \mathrm { \hat { \tau } } } u g l y \mathrm { \hat { \tau } } }$ is $\mu _ { \daleth o } ( \mathrm { L i } ) ~ = ~ 1 - ~ 0 . 9 ~ = ~ 0 . 1$ , the correlation degree of resource “Li” to attribute value ‘ordinary’ is $\mu _ { \sim o } ( \mathrm { L i } ) = 0 . 4 5$ , and the correlation degree of resource $^ { \mathfrak { c } \mathfrak { c } } \mathrm { L i } ^ { \mathfrak { s } }$ to attribute value ‘not beautiful’ is $\mu _ { \neg o } ( \mathrm { L i } )$

= 0.275. Therefore, the four different TCOI triple and their meanings are as follows:

<Li, appearance, (beautiful, 0.9)>: “Li’s appearance beautiful degree is 0.9”

<Li, appearance, (not beautiful, 0.275)>: “Li’ appearance not beautiful degree is 0.275”

<Li, appearance, (ugly, 0.1)>: “Li’s appearance ugly degree is 0.1”

<Li, appearance, (ordinary, 0.45)>: “Liu’s appearance ordinary degree is 0.45”

In the Resource Description Framework (RDF), an RDF graph is a set of RDF triples [1]. The RDF graph of four TCOI triple describing the appearance of “Li” (Fig. 9) is as follows:

![](images/47e79c36e70c50de059114086a1e7ac47d3a402b952ff3295911c992858e48ad.jpg)  
Fig. 9 TCOI triple and RDF graph

For the different negations and fuzziness in the aforementioned everyday statements and web resource description statements, if expressed using conventional triples, it would be more complex or difficult to convey them due to the inability to distinguish and handle the three distinct types of negation and their relationships present in those statements.

## 7. TCOI triple reasoning

A key feature of semantic data models is their support for reasoning. Triple reasoning is a semantic reasoning approach that relies on semantic understanding of data and infers new triples, i.e., new knowledge, from existing triples or sets of triples. Research on triple reasoning has evolved from symbolic logic to statistical learning and then to neural representation learning. Early studies on triple reasoning were mainly built upon logic programming, description logic, and rule-based database reasoning, where Horn rules, Datalog and description logic formed the basic formal frameworks for triple reasoning. With the development of the Semantic Web and RDF/OWL, researchers further focused on RDF semantic entailment, RDFS/OWL reasoning, and SPARQL recursive querying [30,31]. In recent years, knowledge graph embedding methods have transformed triple reasoning into a relation modeling problem in vector spaces, significantly improving the scalability of knowledge completion. Graph neural networks and differentiable rule learning methods further combine the advantages of symbolic reasoning and neural networks, becoming important research directions in current triple reasoning studies [32–36].

In triples reasoning, implication-based reasoning is a fundamental and typical semantic reasoning method. Under a given semantic interpretation, it derives new triple facts by exploiting the semantic entailment relations among existing triple or sets of triples. This approach is grounded in a clear logical foundation, with its core idea being to abstract observed entity–relation instances into first-order logical implications or Horn clauses, such as relation transitivity, symmetry, antisymmetry, and type constraints. On this basis, it constructs interpretable reasoning chains to infer latent triples from explicit facts. Implication-based reasoning exhibits strong symbolic interpretability and consistency, and it can explicitly characterize the logical dependencies among facts, thereby providing a more transparent basis for inference at the knowledge representation level. Overall, implication-based reasoning is not only a classic paradigm in triples reasoning, but also an important bridge between logical knowledge representation and data-driven learning, and thus it is widely used in the construction of reliable reasoning systems [37–41].

In this section, we mainly explore implication-based reasoning in TCOI triple reasoning. It discusses the relationship between TCOI triple implication reasoning and logic LCOI+PLCOI reasoning, as well as the logical foundation of TCOI triple implication reasoning. To demonstrate the capability of TCOI triple implication reasoning, we apply it to counterfactuals and counterfactual reasoning. Using an example of fuzzy counterfactual and counterfactual reasoning, it shows that the three types of fuzzy counterfactual reasoning based on different negations correspond to three kinds of fuzzy TCOI triple implication reasoning. Additionally, a truth-value (continuous value) algorithm is proposed for these implication reasonings, and calculations are performed on the reasoning example.

## 7.1 Logical foundation of TCOI triple implication reasoning

In triples reasoning, implication-based reasoning essentially means that if the premise (a set of triples) is true, then the conclusion (a triple or a set of triples) must also be true.

To support TCOI triple implication reasoning, based on the semantics of logic LCOI+PLCOI, we introduce the concept of “TCOI-entailment” as the “semantic entailment” in TCOI triple implication reasoning. Through TCOI -entailment, a connection is established between TCOI triple entailment reasoning and reasoning in logic LCOI+PLCOI. This indicates that the formally proven inference laws in logic LCOI+PLCOI are valid in TCOI triple entailment reasoning, and that LCOI+PLCOI provides the logical foundation for TCOI triple entailment reasoning.

Definition 1 (TCOI-entailment). Let T be a TCOI triple or set of TCOI triple, S be a set of TCOI triple. S entails T, denoted as $S \models T ,$ if and only if every interpretation that satisfies S in the semantic interpretation of LCOI+PLCOI also satisfies T. If S and T entail each other, then S and T are logically equivalent. This entailment is called the semantic entailment in TCOI triple implication reasoning, abbreviated as TCOI-entailment.

Concise speaking, in TCOI triple implication reasoning, $S \models T$ means that T is a semantic consequence of S (i.e., S is the condition of semantic inference, and T is the conclusion).

A classic triple <s, p, o> is a minimal statement and is an atomic formula in classical predicate logic. Similarly, since the TCOI triple is defined based on the logic LCOI+PLCOI, a TCOI triple is also an atomic formula within the logic LCOI+PLCOI. Using the connectives $( \neg , \exists , \sim , \to , \land , \lor , \exists , \forall )$ in LCOI+PLCOI, multiple TCOI triple can be combined into new TCOI triple, which correspond to compound formulas in the logic $\mathrm { L C O I + P L C O I } .$ Therefore, the T in Definition 1 is a formula or a set of formulas in the logic LCOI+PLCOI, and S is a set of formulas. Consequently, accordng to the soundness theorem of LCOI+PLCOI (Theorem 1 in Section 4.2.2), for the axioms, deduction rules and proven formal theorems in LCOI+PLCOI, the entailment ${ \boldsymbol { S } } \ \vline \ \vline \ \vline { \boldsymbol { S } } \ { \boldsymbol { T } }$ holds. Therefore, for TCOI triple implication reasoning, based on TCOI-entailment, we can derive the following conclusion:

(1) If T is an axiom in LCOI+PLCOI, then ╞ T.

(2) If S and T are respectively the premise and conclusion of a deduction rule in LCOI+PLCOI $( \mathrm { i . e . , } S \vdash T )$ then $S \models T .$

(3) If T is conclusion of a proven formal theorem in LCOI+PLCOI (i.e., ├ T), then $\models T .$

(4) If S and T are respectively the premise and conclusion of a proven formal theorem in LCOI+PLCOI (i.e., $s { \vdash T }$ , then $S \models T .$

Therefore, the TCOI triple implication reasoning has the following properties:

Property 1. If T is axiom in the logic LCOI+PLCOI, then

$$
{ \begin{array} { r l } & { { \bigl | } \mathsf { A } \to ( \mathsf { B } \to \mathsf { A } ) ; ( \mathsf { A } \to ( \mathsf { A } \to \mathsf { B } ) ) \to ( \mathsf { A } \to \mathsf { B } ) ; ( \mathsf { A } \to \mathsf { B } ) \to \gamma ( ( \mathsf { B } \to \mathsf { C } ) \to ( \mathsf { A } \to \mathsf { C } ) ) ; ( \mathsf { A } \to \mathsf { B } ) \to ( \mathsf { B } \to \mathsf { B } ) ; ( \mathsf { A } \to \daleth \qquad \mathsf { B } \to \mathsf { B } ) \to } \\ & { \quad { \mathsf { I } } \to \mathsf { A } \to ( \mathsf { A } \to \mathsf { B } ) ; ( ( \mathsf { A } \to \to \mathsf { A } ) \to \mathsf { B } ) \to ( ( \mathsf { A } \to \mathsf { B } ) \to \mathsf { B } ) ; \mathsf { A } \to \mathsf { A } \times \mathsf { B } ; \mathsf { B } \to \mathsf { A } \times \mathsf { B } ; \mathsf { A } \land \mathsf { B } \to \mathsf { A } ; \mathsf { A } \land \mathsf { B } \to \mathsf { B } ; { \mathsf { q } } \operatorname { A } \to \mathsf { \Lambda } \to \mathsf { A } \land } \end{array} }
$$

$$
\neg { \sim } \mathrm { A } ; { \sim } \mathrm { A } {  } \neg { \mathrm { A } } \land \neg { \ = } \exists \mathrm { A }
$$

$$
\begin{array} { r }   \sharp \bigtriangledown x \mathtt { A } ( x )  \mathtt { A } ( a ) ; \ \mathtt { A } ( a )  \exists x \mathtt { A } ( x ) ; \ \exists x ( \mathtt { A } ( x )  \mathtt { B } )  ( \mathtt { A } ( a )  \mathtt { B } ) ; \   \eta \bigtriangledown x \mathtt { A } ( x )  \exists x \mathtt { A } ( x ) ; \ \exists x \mathtt { A } ( x )  \ \daleth \ \forall x \mathtt { A } ( x ) \} \end{array}
$$

$$
\exists \exists x \mathbf { A } ( x ) \lnot \forall x \exists \mathbf { A } ( x ) \colon \forall x \exists \mathbf { A } ( x ) \lnot \exists x \mathbf { A } ( x )
$$

(Note. remove “╞”, which is the axioms in LCOI+PLCOI (the axioms (a1)-(a18) in Section 4.2.1)

Property 2. If S and T are respectively the premises and conclusions of the deduction rules in LCOI+PLCOI (i.e., $s \vdash T )$ , then

(1) $\mathrm { A } _ { 1 } , \mathrm { A } _ { 2 } , . . . , \mathrm { A } _ { \mathrm { n } } \models \mathrm { A } _ { \mathrm { i } } ( \mathrm { i } \in \{ 1 , 2 , . . . , \mathrm { n } \} ) .$

(2) $\mathbf { A { \to } B } , \mathbf { A } \models \mathbf { B } .$

(3) If $\mathrm { A } _ { 1 } , \mathrm { A } _ { 2 } , . . . , \mathrm { A } _ { \mathrm { n } } \models \mathrm { A } ( a )$ , where a does not occur in $\mathbf { A } _ { \mathrm { i } } \left( 1 { \le } \mathrm { i } \le \mathbf { n } \right)$ , then $\mathrm { A } _ { 1 } , \mathrm { A } _ { 2 } , . . . , \mathrm { A } _ { \mathrm { n } } \ \vline \ \vline \forall { x } \mathrm { A } ( x )$

(Note. replace ╞ with ├, that is the deduction rules of LCOI+PLCOI (D1, D2 and D3 in Section 4.2.1)

Property 3. If T is conclusions of the proven formal theorems in LCOI+PLCOI (i.e., ├ T), then

[1] FA→A;((A→B)→C)→(B→C); A→((A→B)→(C→B)); A→((A→B)→B)

[2] F(((A→B)→B)→C)→(A→C); (A→(B→C))→(B→(A→C)); (B→C)→((A→B)→(A→C))

[3] F((A→B)→(A→(A→C)))→((A→B)→(A→C))

[4] F(B→(A→C))→((A→B)→(A→C)); (A→(B→C))→((A→B)→(A→C))

[5] F¬A→(A→¬(B→B)); B→((A→¬B)→¬A); (A→¬B)→((A→B)→¬A)

$$
\begin{array} { r l } { [ 6 ] } & { { } \models ( \mathbf { A } \to \mathbf { B } ) { \to } ( ( \mathbf { A } \to \mathbf { - } \mathbf { B } ) { \to } \lnot \mathbf { A } ) ; ( \mathbf { A } \to \mathbf { B } ) { \to } ( \lnot \mathbf { B } \to \lnot \mathbf { A } ) ; ( \mathbf { A } \to \lnot \mathbf { A } ) { \to } \lnot \mathbf { A } ; \mathbf { A } { \to } \lnot \mathbf { - } \lnot \mathbf { A } } \end{array}
$$

$$
\begin{array} { r l } { [ 7 ] } & { { } \models ( \neg \mathrm { A } \to \neg \mathrm { B } ) \to ( \mathrm { B } \to \mathrm { A } ) ; \neg \neg \mathrm { A } \to ( \neg \mathrm { A } \to \mathrm { A } ) ; \neg \neg \mathrm { A } \to \mathrm { A } ; ( \neg \mathrm { A } \to \mathrm { B } ) \to ( ( \neg \mathrm { A } \to \neg \mathrm { B } ) \to \mathrm { A } ) } \end{array}
$$

[8] ╞ $\mp \mathbb { A } \to ( \mathbb { A } \to \mp \left( \mathbb { B } \to \mathbf { B } \right) ) ; \mathbb { B } \to ( ( \mathbb { A } \to \mp \mathbb { B } ) \to \mp \mathbb { A } ) ; ( \mathbb { A } \to \mp \mathbb { B } ) \to ( \mathbb { A } \to ( \mathbf { B } \to \mp \left( \mathbb { B } \to \mathbf { B } \right) ) )$

[9] $\vdash ( \mathrm { ( A  B )  ( ( A  \daleth \mathrm { ~ B )  \daleth \mathrm { ~ ( A ) ; ~ ( A  B )  ( \vec { q } \mathrm { ~ B  \vec { q } \mathrm { ~ A } ) ; ~ ( A  \vec { q } \mathrm { ~ A } )  \vec { q } \mathrm { ~ A }  \vec { q } \mathrm { ~ A }  \vec { q } \mathrm { ~ A }  \vec { q } \mathrm { ~ A }  \vec { q } \mathrm { ~ A }  \vec { q } \mathrm { ~ A }  \vec { q } \mathrm { ~ A }  \vec { q } \mathrm { ~ A }  \vec { q } \mathrm { ~ A }  \vec { q } \mathrm { ~ A } } } } } $

[10] $\vdash ( \exists \textsc { A } \to \daleth \textsc { B } ) \to ( \textsc { B } \to \mathsf { A } ) ; \exists \exists \textsc { A } \to ( \exists \textsc { q } \mathsf { A } \to \mathsf { A } ) ; \ q \exists \mathsf { A } \to \mathsf { A } ; \lnot ( \mathbf { A } \land \lnot \mathbf { A } ) ; \lnot ( \exists \textsc { A } \land \lnot \mathbf { A } )$

[11] ¬(A∧\~A); ¬(A^η A)

(Note. replace ╞ with ├, that is the formal theorems in LCOI+PLCOI (theorem 1- theorem 4, Section 4.2.1 in [13]))

Property 4. If S and T are respectively the premises and conclusions of the proven formal theorems in LCOI+PLCOI $( \mathrm { i } . \mathrm { e } . , S \vdash T )$ , then

(1) $A \models \mathsf { A } ; \mathsf { q } \exists \mathsf { A } ; \neg \neg \mathsf { A } ; \neg \exists \mathsf { A } \land \neg \sim \mathsf { A } ; \neg \mathsf { q } \sim \mathsf { A }$

(2) $\exists \exists \operatorname { A } \models \operatorname { A } ;$

(3) $\lnot \\to \mathrm { A } \models \mathrm { A }$

(4) $\mathbf { A } , \mp \mathbf { B } \models \daleth ( \mathbf { A {  } } \mathbf { B } )$

(5) $\daleth ( \mathrm { A {  } B } ) \models \mathrm { A , \daleth B }$

(6) ${ \sim } \mathbf { A } \models \sim \mathsf { \lneq A }$

(7) $\sim \daleth \operatorname { A } \models \sim \operatorname { A }$

(8) $\lnot \ 7 \operatorname { A \land \lnot \sim A } \left| \ l \in \operatorname { A } \right.$

(9) $\mathbf { A } , \lnot \mathbf { A } \models \mathbf { B }$

(10) $\mathbf { A } , \lneq \mathbf { A } \models \mathbf { B }$

(11) $\mathbf { A } , { \sim } \mathbf { A } \models \mathbf { B }$

(Note. replace ╞ with ├, that is the formal theorems in LCOI+PLCOI (theorem 5 and theorem 6, Section 4.2.1 in [13])).

The above TCOI triple implication reasoning properties indicate that the proven formal inference relations in the logic LCOI+PLCOI are valid in the TCOI triple implication reasoning. This validity has the following significance:

<sup></sup> LCOI+PLCOI provide a solid logical foundation for the TCOI triple implication reasoning.

 For any formal deduce S├ T in logic LCOI+PLCOI, according to the soundness theorem of LCOI+PLCOI (Theorem 1 in Section 4.2.2), S╞ T holds in TCOI triple implication reasoning. That is, $S \vdash T \Rightarrow S \models T .$ Conversely, $S \models T \Rightarrow S \vdash T$ according to the completeness theorem of LCOI+PLCOI [8, 9].

<sup></sup> The TCOI triple has stronger reasoning capabilities than the classic triple and other extensions.

<sup></sup> This deepens the integration of logical theory and triple theory, and expands the scope of TCOI triple for data analysis and knowledge processing.

## 7.2 Applications of TCOI triple implication reasoning in counterfactual reasoning

Counterfactuals and counterfactual reasoning are cognitive processes in humans that constitute a significant topic of interest across several fields, including philosophy, cognitive science, linguistics, logic, and artificial intelligence. The core of counterfactual thinking lies in negating the facts that have occurred (the real world) to envision a possibility that has not happened (the possible world)[42]. A counterfactual refers to a hypothetical statement that is contrary to the facts, often used to describe a “what if things had not happened as they actually did” scenario. Counterfactual reasoning is a process that involves reasoning based on counterfactuals, with its basic structure being the negation of facts to construct a hypothetical premise that supports hypothetical reasoning. Negation occupies a central position in both counterfactual statements and reasoning, without negation, there would be no counterfactuals or counterfactual reasoning [43-48]. Crisp counterfactuals and fuzzy counterfactuals are two different types of conditional sentences, distinguished by significant differences in the premise or conclusion statements. The premise and conclusion of crisp counterfactuals are clear statements (i.e., the concepts contained are all clear concepts). The premise or the conclusion or both of fuzzy counterfactuals are fuzzy statements (i.e., they contain fuzzy concepts) [49].

Triple reasoning and counterfactual reasoning can both be classified as forms of semantics-based reasoning, but they focus on different levels of semantics. Triple reasoning primarily concerns relational semantics and structured semantics, whereas counterfactual reasoning mainly concerns causal semantics.

In this section, we apply TCOI triple implication reasoning to counterfactuals and counterfactual reasoning. Using an example of the counterfactuals and counterfactual reasoning based on three different kinds of negation, it show that three kinds of fuzzy counterfactual reasoning based on different negations correspond to three kinds of fuzzy TCOI triple implication reasoning.

Example. Fact F1: If the water temperature in the teacup is high, then the hand feels that the teacup is hot. There are following counterfactuals:

(1) If hand feels the teacup not hot, then the water temperature in the teacup is not high.

(2) If hand feels the teacup cool, then the water temperature in the teacup is low.

(3) If hand feels the teacup neither hot nor cool, then the water temperature in the teacup is neither high nor low.

It can be seen that “high”, “not high”, “low”, “neither high nor low”, “hot”, “not hot”, “cool” and “neither hot nor cool”, are all fuzzy concepts. Therefore, the fact F1 and the counterfactuals (1), (2) and (3) are all fuzzy statements, that is (1), (2) and (3) are fuzzy counterfactuals.

According to the set SCOI and the logic LCOI+PLCOI, ‘not high’, ‘low’ and ‘neither high nor low’ are respectively the contradictory negation, opposite negation and intermediary negation of “high”; ‘not hot’, ‘cool’ and “neither hot nor cool” are respectively the contradictory negation, opposite negation and intermediary negation of “hot”. Therefore, we may refer to conditionals (1), (2), and (3) as three fuzzy counterfactuals based on contradictory negation, opposite negation and intermediary negation, respectively.

For the statements in Fact F1 and fuzzy counterfactuals (1), (2), and (3), according to the definitions of the logic LCOI+PLCOI and TCOI triple, we can express them using TCOI triple as follows:

A: “the water temperature in the teacup is high”, <teacup, temperature, (high, $\mu _ { o } ( \mathsf { t e a c u p } ) ) { \scriptstyle > }$

A: “the water temperature in the teacup is not high”. <teacup, temperature, (not high, $\mu _ { \neg o } ( \mathsf { t e a c u p } ) ) >$

╕A: “the water temperature in the teacup is low”. <teacup, temperature, (low, $\mu _ { \mp \ o } ( \mathsf { t e a c u p } ) ) >$

A: “the water temperature in the teacup is neither high nor low”, <teacup, temperature, (neither high nor low, $\mu _ { \sim o } ( \mathsf { t e a c u p } ) ) >$

B: “the hand feels the teacup hot”. <hand, feels, (hot, $\mu _ { o } ( \mathrm { h a n d } ) >$

B: “the hand feels the teacup not hot”. <hand, feels, (not hot, $\mu _ {  o } ( \mathrm { h a n d } ) ) >$

╕B: “the hand feels the teacup cool”. <hand, feels, (cool, $\mu _ { \mp \ : o } ( \mathrm { h a n d } ) >$

B: “the hand feels the teacup neither hot nor cool”. <hand, feels, (neither hot nor cool, $\mu _ { \sim o } ( \mathrm { h a n d } ) ) >$

Therefore, the TCOI triple representations of fact F1 and fuzzy counterfactuals (1), (2) and (3) are as follows:

F1 <teacup, temperature, (high, $\mu _ { o } ( \mathrm { t e a c u p } ) ) { \scriptstyle > - \ }$ <hand, feels, (hot, $\mu _ { o } ( \mathrm { h a n d } ) >$

(1) <hand, feels, (not hot, $\mu _ { \neg o } ( \mathrm { h a n d } ) ) > $ <teacup, temperature, (not high, $\mu _ { \neg o } ( \mathsf { t e a c u p } ) ) >$

(2) <hand, feels, (cool, $\mu _ { \mp o } ( \mathrm { h a n d } ) ) > $ <teacup, temperature, (low, $\mu _ { \mp \ o } ( \mathsf { t e a c u p } ) ) >$

(3) <hand, feels, (neither hot nor cool, <sub>\~o</sub>(hand))> <teacup, temperature, (neither high nor low,

## <sub>\~o</sub>(teacup))>

Counterfactual reasoning (CR) is a form of reasoning based on counterfactuals. Fuzzy counterfactual reasoning based on the three fuzzy counterfactuals (1), (2) and (3) is respectively as follows:

$C R _ { ☉ }$ If the water temperature in the teacup is high, then the hand feels the teacup is hot, but the hand feels the teacup is not hot. Therefore, the water temperature in the teacup is not high.

$C R _ { \daleth } .$ : If the water temperature in the teacup is high, then the hand feels the teacup is hot, but the hand feels the teacup is cool. Therefore, the water temperature in the teacup is low.

CR <sub></sub>: If the water temperature in the teacup is high, then the hand feels the teacup is hot, but the hand feels the teacup is neither hot nor cool. Therefore, the water temperature in the teacup is neither high nor low.

Based on the logic LCOI+PLCOI, the fuzzy counterfactual reasoning CR<sub></sub>, CR <sub>╕</sub> and CR<sub></sub> can be formally represented as follows (the symbol  stands for logical deduction):

$$
C R _ { \neg } \colon \mathrm { A } \mathrm { \to } \mathrm { B } , \mathrm { \ - } \mathrm { B } \mathrm { \Rightarrow \mathrm { \ - } A }
$$

$$
C R _ { \exists } : { \mathrm { ~ } } \mathrm { A } {  } \mathrm { B } , \lnot \mathrm { ~ B } \Rightarrow \lnot \mathrm { ~ A ~ }
$$

$$
C R _ { \sim } \mathrm {  ~ \ A \to { B } , \tilde { \Sigma } { B } \Rightarrow \tilde { \Sigma } { A } }
$$

According to the logic LCOI+PLCOI and the definition of TCOI-entailment (Definition 1), these fuzzy counterfactual reasoning can be separately expressed using TCOI triple implication reasoning as follows:

$$
T _ { \neg } \colon \mathrm { A { \to } B } , { \ u { \mathrm { \to } } } \mathrm { B } \ \models { \neg } \mathrm { A }
$$

<teacup, temperature, (high,  (teacup))> <hand, feels, (hot,  (hand))><hand, feels, (not hot, $\mu _ { \neg o } ( \mathrm { h a n d } ) ) >$ ╞ <teacup, temperature, (not high, $\mu _ { \neg o } ( \mathsf { t e a c u p } ) ) { \bmod { \Gamma } }$

$$
T _ { \daleth } \colon \mathrm { A  B , \daleth B } \ \models \daleth \mathrm { A }
$$

<teacup, temperature, (high, $\mu _ { o } ( \mathsf { t e a c u p } ) ) { \bmod { } } - \to$ <hand, feels, (hot,  (hand))><hand, feels, (cool, $\mu _ { \mp \ : o } ( \mathrm { h a n d } ) >$ ╞ <teacup, temperature, (low, $\mu _ { \mp \ o } ( \mathsf { t e a c u p } ) ) >$

$$
T _ { \sim } \mathrm { { : } A {  } B , { \sim } B } \ \mathsf { \ k } { \sim } \mathrm { { A } }
$$

<teacup, temperature, (high, $\mu _ { o } ( \mathsf { t e a c u p } ) ) { \bmod { } } - \to$ <hand, feels, (hot, $\mu _ { o } ( \mathrm { h a n d } ) ) \div$ > <hand, feels, (neither hot nor cool, $\mu _ { \sim o } ( \mathrm { h a n d } ) ) >$

╞ <teacup, temperature, (neither high nor low, $\mu _ { \sim o } ( \mathsf { t e a c u p } ) ) { \scriptstyle > }$

Here, $T _ { \neg } , T _ { \daleth }$ and T<sub></sub> respectively denote the three fuzzy TCOI triple implication reasoning based on contradictory negation , opposite negation ╕ and intermediary negation .

It can be seen that fuzzy TCOI triple implication reasoning $T _ { \neg } , T _ { \daleth }$ and $T _ { \sim }$ , respectively correspond to the three fuzzy counterfactual reasoning $C R _ { \lnot } , C R _ { \lnot }$ and $C R _ { \sim }$

## 7.3 A truth-value algorithm for TCOI triple implication reasoning

From the nature of logical reasoning, reasoning is a logical process of deriving conclusions from premises. An algorithm for reasoning computes the conclusion by a prescribed procedure rather than deriving it logically.

In triple reasoning, the implication-based reasoning is essentially a semantically driven process that determines the derivability of potential triples according to semantic entailment relations. The algorithmization of reasoning can employ different computational models, including closure computation [50], forward-chaining inference [51], rule entailment checking [52], and truth-value computation under many-valued semantics [53]. From classical logic to modern approaches that use non-classical logics to handle uncertainty, fuzziness and even context dependence, all provide ways to compute the truth value of triples [54, 55].

In this section, we propose a truth value (continuous-value) algorithm for the three fuzzy TCOI triple implication reasoning $T _ { \neg } , \ T _ { \daleth }$ and $T _ { \sim }$ , and perform truth-value calculations for the three fuzzy counterfactual reasoning reasoning $C R _ { \neg } , C R _ { \mp }$ and $C R _ { \sim }$

In $T _ { \neg } , T _ { \mp }$ and $T _ { \sim } ,$ , negation and implication  (logical implication) are the core concepts. For the three types of negation in $T _ { \neg } , T _ { \mp }$ and $T _ { \sim }$ , we use the symbol N to denote these different negations, that is $\mathbb { N } \in \left\{ \lnot , \lnot \right\} , \tilde { \ b { \mathscr { 1 } } } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime } ^ { \prime \right. }  \left.$ . As for implication  in $T _ { \neg } , T _ { \mp }$ and $T _ { \sim } ,$ , it is necessary to determine  itself as well as its relationship with N.

The implication  is a binary function in fuzzy logic. We define it as follows with reference to [56] and [57].

Definition 2. A mapping $T \colon [ 0 , 1 ] ^ { 2 }  [ 0 ]$ , 1] is called a t-norm, if it satisfies

(T1) Boundary conditions, $T ( 0 , 0 ) = 0 , T ( 1 , x ) = x , T ( 0 , x ) = 0 ;$

(T2) Monotony, ${ \mathrm { i f } } \ x \leq y$ then $T ( x , z ) \leq T ( y , z )$ , for all $z \in [ 0 , 1 ] ;$

(T3) Commutative, $T ( x , y ) = T ( y , x )$ , for all $x , y \in [ 0 , 1 ]$

(T4) Associative, $T ( x , T ( y , z ) ) = T ( T ( x , y ) , z )$ , for all $x , y , z \in [ 0 , 1 ]$

Definition 3. Let T be a fixed t-norm (e.g., the Gödel t-norm, product t-norm, or Łukasiewicz t-norm). A mapping $I _ { C O I } \colon [ 0 , 1 ] ^ { 2 } \longrightarrow [ 0 , 1 ]$ is called logical implication in fuzzy TCOI triple implication reasoning, if it satisfies

$$
I _ { C O I } ( x , y ) = s u p \{ s \in [ 0 , 1 ] \mid T ( \mathrm { N } ( y ) , s ) \le \mathrm { N } ( x ) \} , \forall x , y \in [ 0 , 1 ] .
$$

According to the above definitions, it is easy to prove that $I _ { C O I }$ has the following properties:

Property 5. For all $x , y \in [ 0 , 1 ]$

(a) $T ( \mathbb { N } ( y ) , I _ { C O I } ( x , y ) ) \le \mathbb { N } ( x ) ;$

(b) $T ( \mathbb { N } ( y ) , s ) \leq \mathbb { N } ( x )$ if and only i $\dot { \boldsymbol { s } } \leq I _ { C O I } ( \boldsymbol { x } , \boldsymbol { y } )$

Proof:

(a). Suppose $T ( \mathbb { N } ( y ) , I _ { C O I } ( x , y ) ) > \mathbb { N } ( x ) . \ \mathbb { N } ( y ) > \mathbb { N } ( x )$ and $I _ { C O I } ( x , y ) > \mathbb { N } ( x )$ according to Definition 2. Since $I _ { C O I } ( x , y )$ $ = s u p \{ s \in [ 0 , 1 ] \mid T ( \mathbb { N } ( y ) , \ s ) \le \mathbb { N } ( x ) \}$ , that is, $T ( \mathbb { N } ( y ) , \ s ) \ \leq \ \mathbb { N } ( x )$ , it follows that $\mathbb { N } ( y ) \le \mathbb { N } ( x )$ . This leads to a contradiction. Therefore, $T ( \mathbb { N } ( y ) , I _ { C O I } ( x , y ) ) \le \mathbb { N } ( x )$

$$
T ( \mathbb { N } ( y ) , s ) \leq \mathbb { N } ( x ) , { \mathrm { t h e n } } s = \{ s \in [ 0 , 1 ] | T ( \mathbb { N } ( y ) , s ) \leq \mathbb { N } ( x ) \} \leq s u p \{ s \in [ 0 , 1 ] | T ( \mathbb { N } ( y ) , s ) \leq \mathbb { N } ( x ) \} = I _ { C O I } ( x , y ) . \mathbb { I } f s
$$

$\le I _ { C O I } ( x , y )$ , then $s \leq s u p \{ s \in [ 0 , 1 ] \mid T ( \mathrm { N } ( y ) , s ) \leq \mathrm { N } ( x ) \}$ . Therefore, $T ( \mathbb { N } ( y ) , s ) \leq \mathbb { N } ( x )$ . 

For the three fuzzy TCOI triple implication reasoning $T _ { \neg } , T _ { \mp }$ and $T _ { \sim } ,$ we propose a truth-value algorithm based on the TCOI-entailment (Definition 1), the $I _ { C O I }$ implication (Definition 3) and the continuous-valued interpretation  of the logic LCOI+PLCOI (Definition 2 in Section 4.2.2).

In accordance with the structure of $T _ { \neg } , T _ { \mp }$ and $T _ { \sim } ,$ this algorithm is designed to determine the truth values of the conclusions $\neg \mathbf { A } , \daleth \mathbf { A }$ and A.

(I) Algorithm of CR<sub></sub>

formal expression： $\mathbf { A {  } B { , } { \lnot \mathbf { B } } \ } \models { \lnot \mathbf { A } }$

algorithm: $\hat { \sigma } ( \lnot \mathbf { A } ) = T ( \hat { \sigma } ( \lnot \mathbf { B } ) , I _ { C O I } ( \hat { \sigma } ( \mathbf { A } ) , \hat { \sigma } ( \mathbf { B } ) ) ) , \forall \hat { \sigma } ( \mathbf { A } ) , \hat { \sigma } ( \mathbf { B } ) \in [ 0 , 1 ] .$

(II) Algorithm of $C R _ { \mp }$

formal expression： $_ { \mathrm { A  B , \mp B } } \models \daleth \mathrm { A }$

algorithm: $\boldsymbol { \hat { o } } ( \Im \mathbf { A } ) = T ( \boldsymbol { \hat { o } } ( \Im \mathbf { B } ) , I _ { C O I } ( \boldsymbol { \hat { o } } ( \mathbf { A } ) , \boldsymbol { \hat { o } } ( \mathbf { B } ) ) ) , \forall \boldsymbol { \hat { o } } ( \mathbf { A } ) , \boldsymbol { \hat { o } } ( \mathbf { B } ) \in [ 0 , 1 ] ] .$

(III) Algorithm of CR<sub></sub>

formal expression： $\mathbf { A {  } B } , { \sim } \mathbf { B } \ \models { \sim } \mathbf { A }$

algorithm: ∂(\~A) = T(∂(\~B), Icor(∂(A), ∂(B))), ∀∂(A), ∂(B)∈[0, 1].

Among them, $\hat { \sigma } ( \mathrm { A } ) , \hat { \sigma } ( \lnot \mathrm { A } ) , \hat { \sigma } ( \lnot \mathrm { A } ) , \hat { \sigma } ( \lnot \mathrm { A } ) , \hat { \sigma } ( \lnot \mathrm { B } ) , \hat { \sigma } ( \lnot \mathrm { B } ) , \hat { \sigma } ( \lnot \mathrm { B } )$ and $\partial ( \mathrm { \sim } \mathbf { B } )$ represent the truth values of formulas A, $\neg \mathrm { A } , \mp \mathrm { A } , \sim \mathrm { A } , \mathrm { B } , \lnot \mathrm { B } , \mp$ B and B in the logic LCOI+PLCOI, respectively.

As can be seen from the structure of Algorithms (I), (II) and (III), the computation of $\partial ( \neg \mathbf { A } ) , \partial ( \daleth \mathbf { A } )$ and $\partial ( { \sim } \mathbf { A } )$ depends on $I _ { C O I } ( \partial ( \mathbf { A } ) , \ \partial ( \mathbf { B } ) ) , \ \partial ( \mathbf { - B } ) , \ \partial ( \mathbf { \vec { q } } \ \mathbf { B } )$ and (B). Therefore, the steps to solve the reasoning conclusions $\partial ( \neg \mathbf { A } ) , \partial ( \daleth \mathbf { A } )$ and $\partial ( { \sim } \mathbf { A } )$ using algorithms (I), (II), and (III) are as follows:

(i) According to the continuous-valued semantics  of the logic $\scriptstyle { \mathrm { L C O I + P L C O I } } ,$ , computing the truth value of implication $\mathbf { A } {  } \mathbf { B }$ , namely $I _ { R } ( \hat { o } ( \mathbf { A } ) , \hat { o } ( \mathbf { B } ) )$ .

(ii) Based on ∂ and algorithms (I), (II), (III), computing the truth values ∂(¬B), ∂(η B) and ∂(\~B) of ¬B, η B and B.

(iii) By (i) and (ii), obtain the truth values $\hat { \sigma } ( \lnot \mathbf { A } ) , \hat { \sigma } ( \lnot \mathbf { A } ) , \hat { \sigma } ( \lnot \mathbf { A } )$ of the reasoning conclusions A, ╕A and A.

Below, we use this procedure to solve the truth values of the reasoning conclusions in the fuzzy counterfactual reasoning $C R _ { \lnot } , C R _ { \lnot }$ and $C R _ { \mathrm { { \circ } } }$

(i) In $C R _ { \neg } , C R _ { \neg }$ and $C R _ { \sim } ,$ the first premise is AB (“If the water temperature in the teacup is high, then the hand feels that the teacup is hot”). By the definition of ∂, the truth value (AB) of AB, that is, $I _ { C O I } ( \partial ( \mathbf { A } )$ , (B)) $\in [ 0 , 1 ]$ . The truth values ∂(—B), ∂= B) and ∂(\~B) of the second premise —B, = B and \~B, can be obtained by the truth value ∂(B) of B according to ∂.

In fuzzy logic, the truth value of a fuzzy proposition in a domain can be determined through expert opinions or statistical methods. For ease of discussion, we assume $\widehat { \cal O } ( \mathrm { A } {  } \mathrm { B } ) = 0 . 9 , \widehat { \cal O } ( \mathrm { B } ) = 0 . 9$

(ii) Because $\widehat { \cal O } ( { \bf B } ) = 0 . 9 , \widehat { \cal O } ( \daleth { \bf B } ) = 1 - \widehat { \cal O } ( { \bf B } ) = 0 . 1$ by the definition of . According to the algorithm (II), $\partial ( \daleth \mathrm { A } ) =$ T((╕B), $I _ { C O I } ( \hat { o } ( \mathrm { \bf A } ) , \hat { o } ( \mathrm { \bf B } ) ) ) = T ( 0 . 1 , 0 . 9 ) = 0 . 1$ . According to [3] in the definition of ∂, ∂(\~B) has two cases (a) and (d).

For the case (a), $\hat { \partial } ( \mathbf { \tilde { \alpha } } - \mathbf { B } ) = \lambda - \frac { 2 \lambda - 1 } { 1 - \lambda } ( \hat { \alpha } ( \mathbf { B } ) - \lambda )$ when $\lambda { \in } [ \% , 1 )$ and ∂(B)∈(λ, 1], so $\hat { \sigma } ( \mathbf { \tilde { \sigma } - B } ) = \lambda - \frac { 2 \lambda - 1 } { 1 - \lambda } ( 0 . 9 - \lambda )$ $= \frac { 9 - 8 \lambda } { 1 0 ( 1 - \lambda ) } - \lambda$ . It can be concluded that $\partial ( \mathrm { \sim } \mathbf { B } ) < 0 . 9$ . According to the algorithm (III), $\hat { \sigma } (  { \mathbf { \hat { \rho } } } \mathbf { A } ) = T ( \hat { \sigma } (  { \mathbf { \tilde { \rho } } } \mathbf { B } ) , I _ { C O I } ( \hat { \sigma } (  { \mathbf { A } } )$ ∂(B))) = T(∂(\~B), 0.9) = ∂(\~B). So, $\hat { \sigma } (  { - } \mathbf { A } ) = \hat { \sigma } (  { - } \mathbf { B } ) = \frac { 9 - 8 \lambda } { 1 0 ( 1 - \lambda ) } - \lambda$ . Based on [4] in the definition of $\hat { o } , \hat { o } ( \lnot \mathbf { B } ) =$ max(∂(= B), ∂(\~B)), thus $\widehat { \cal O } ( \lnot \mathbf { B } ) = \widehat { \cal O } ( \lnot \mathbf { B } )$ ). According to the algorithm (I), $\hat { \cal O } ( \lnot \mathbf { A } ) = T ( \hat { \cal O } ( \lnot \mathbf { B } ) , { \cal I } _ { C O I } ( \hat { \cal O } ( \mathbf { A } ) , \hat { \cal O } ( \mathbf { B } ) ) )$ , so $\hat { \sigma } ( \lnot \mathbf { A } ) = T ( \hat { \sigma } ( \lnot \mathbf { B } ) , 0 . 9 )$ . Thus, $( \neg \mathbf { A } ) = \partial ( \neg \mathbf { B } ) = { \frac { 9 - 8 \lambda } { 1 0 ( 1 - \lambda ) } } - \lambda$

For the case (d), $\hat { \partial } ( \mathbf { \hat { \alpha } } \mathbf { \times B } ) = 1 - \frac { 1 - 2 \lambda } { \lambda } ( \hat { \sigma } ( \mathbf { B } ) + \lambda - 1 ) - \lambda$ when λ∈(0, ½] and ∂(B)∈(1−λ, 1], so ${ \widehat { \cal O } } ( { \sim } \mathrm { B } ) = { \frac { 1 - 2 \lambda } { 1 0 \lambda } } + \lambda ~$ . It can be concluded that $\partial ( \mathrm { \sim } \mathrm { B } ) < 0 . 9$ . According to the algorithm (III), ${ \hat { \sigma } } ( \mathbf { \ - A } ) = T ( { \hat { \sigma } } ( \mathbf { \ - B } ) , I _ { C O I } ( { \hat { \sigma } } ( \mathbf { A } ) , { \hat { \sigma } } ( \mathbf { B } ) ) ) = T ( { \hat { \sigma } } ( \mathbf { \ - B } )$ $0 . 9 ) = \hat { \sigma } ( \mathbf { \hat { \sigma } } \mathbf { \tilde { - } } \mathbf { B } )$ , so $\widehat { \cal O } ( { \bf \sim } { \bf A } ) = \widehat { \cal O } ( { \bf \sim } { \bf B } ) = \frac { 1 - 2 \lambda } { 1 0 \lambda } + \lambda .$ . Based on [4] in the definition of $\hat { \sigma } , \hat { \sigma } ( \lnot \mathbf { B } ) = \operatorname* { m a x } ( \hat { \sigma } ( \lnot \mathbf { B } ) , \hat { \sigma } ( \lnot \mathbf { B } ) )$ , thus $\widehat { \cal O } ( \neg \mathbf { B } ) = \widehat { \cal O } ( \neg \mathbf { B } )$ ). According to the algorithm (I), ∂(¬A) = T(∂(¬B), Ico1(∂(A), ∂(B))), so ∂(¬A) = T(∂(\~B), 0.9) Thus, $( \lnot \mathbf { A } ) = { \widehat { \cal O } } ( \lnot \mathbf { B } ) = \frac { 1 - 2 \lambda } { 1 0 \lambda } + \lambda .$

(iii) Under the assumption of $\widehat { \cal O } ( \mathrm { A } {  } \mathrm { B } ) = 0 . 9 , \widehat { \cal O } ( \mathrm { B } ) = 0 . 9$ , based on the above calculation results, the truth values $\partial ( \neg \mathbf { A } ) , \partial ( \daleth \mathbf { A } )$ and ∂(\~A) of the conclusions —A,  A and \~A for the fuzzy counterfactual reasoning $C R _ { \neg } , C R _ { \daleth }$ and CR as follows:

(1) $\partial ( \Im \mathrm { \ A } ) = 0 . 1$

$$
{ \mathrm { ( 2 ) ~ } } { \widehat { \mathcal { O } } } ( \sim \mathbf { A } ) = { \frac { 9 - 8 \lambda } { 1 0 ( 1 - \lambda ) } } - \lambda ( \lambda \in [ { \mathsf { V } } 2 , 1 ) ) , { \mathrm { ~ o r ~ } } { \widehat { \mathcal { O } } } ( \sim \mathbf { A } ) = { \frac { 1 - 2 \lambda } { 1 0 \lambda } } + \lambda ( \lambda \in ( 0 , 1 \gamma _ { 2 } ] ) .
$$

(3) $\hat { \cal O } ( \lnot \mathbf { A } ) = \hat { \cal O } ( \lnot \mathbf { A } ) .$

In the above truth values $\partial ( \mathord { \sim } \mathrm { A } ) , \ \lambda \ ( \lambda { \in } ( 0 , \ 1 ) )$ is a variable parameter. From the definition of  in the continuous-valued semantics of the logic LCOI+PLCOI and Figure 7, it can be seen that the variation in the value

of λ determines the magnitude and range of ∂(—A), ∂= A) and ∂(\~A), that is, λ serves as a "threshold" for the range of these truth values. The role and significance of λ have been discussed in the context of the set SCOI and the logic LCOI+PLCOI [13].

## 8. Conclusions and future work

Regarding the three different types of negation present in triples and their elements, the classical triple <s, p, o>, which expresses simple positive semantic relationships, as well as classical and non-classical logics that contain only one type of negation (classical negation) in formal languages, both lack inherent mechanisms to distinguish and express these three different types of negation. Therefore, to accurately and finely represent these different negations and their relationships in knowledge representation, the classical triple needs to be extended based on a non classical logic that can distinguish and express different negations at both the syntactic and semantic interpretation levels. Such logic includes multiple negation operators or semantic clarification mechanisms to achieve complete expression of negation meanings and accurate semantic modeling.

Concerning the negativity in triples and their elements, this paper conceptually proposes that there are three distinct negations within triples and their elements: contradictory negation, opposite negation, and intermediary negation. Based on the set SCOI and the logic LCOI+PLCOI, which feature these three negations, an extension of the triple is proposed to distinguish and express the three different negations in triples and their elements, termed the “TCOI triple with contradictory negation, opposite negation and intermediary negation”.

The TCOI triple can express the relationship between the subject and the object along with their degree of association, and it can distinguish and express the three different negations within triples and their elements. This paper demonstrates the expressive power of the TCOI triple through everyday statements and Web resource descriptions involving different negations.

For the TCOI triple reasoning, the focus is primarily on implication-based reasoning. Based on the semantics of logic LCOI+PLCOI, a concept called “TCOI-entailment” is introduced as the semantic implication for TCOI triple implication reasoning. TCOI-entailment establishes a connection between TCOI triple implication reasoning and inferences in logic LCOI+PLCOI, showing that the formal inference laws proven in logic LCOI+PLCOI are valid in TCOI triple implication reasoning. LCOI+PLCOI provide a logical foundation for TCOI triple implication reasoning.

The TCOI triple reasoning and counterfactual reasoning both belong to the category of semantic-based reasoning. Triple reasoning primarily focuses on relational semantics and structured semantics, while counterfactual reasoning mainly emphasizes causal semantics. To demonstrate the reasoning capability of TCOI triple implication reasoning, we apply it to counterfactuals and counterfactual reasoning. Through an example of fuzzy counterfactuals and counterfactual reasoning, it show that the three types of fuzzy counterfactual reasoning based on different negations correspond to three types of fuzzy TCOI triple implication reasoning. Furthermore, we propose a truth-value (continuous-valued) algorithm for fuzzy TCOI triple implication reasoning and perform calculations on the reasoning example.

We consider the TCOI triple is a semantic and structural extension of the classical triple <s, p, o>. While retaining the ability to express affirmative assertions, it systematically introduces the three semantic dimensions of contradictory negation, opposite negation and intermediary negation, allowing these negations to independently apply to the elements (s, p, o) of the triple and on the whole triple. This significantly enhances the triple model capability to represent and reasoning about complex negative information

Building on this paper, we will further explore the applications of TCOI triple and its reasoning in fields such as RDF, the semantic web and knowledge graphs.

## References

[1] Graham Klyne，Jeremy J. Carroll. Resource Description Framework (RDF): Concepts and Abstract Syntax. http://www.w3.org/TR/2004/REC-rdf-concepts-20040210/

[2] Xiaojun Chen, Shengbin Jia, Yang Xiang, A review: Knowledge reasoning over knowledge graph, Expert Systems with Applications, Volume 141, 2020, 112948

[3] hang, W., Wang, B., Zhu, P., Ding, L., & Wang, S. (2024). A Span-based Multivariate Information-aware Embedding Network for joint relational triplet extraction of threat intelligence. Knowledge-Based Systems, 295, 111829. https://doi.org/10.1016/j.knosys.2024.111829

[4] Marcos, E. do N., & Volkov, Y. (2022). Homogeneous triples for homogeneous algebras with two relations. Journal of Algebra, 599, 1–47. https://doi.org/10.1016/j.jalgebra.2022.01.014

[5] Xie, P., Zhou, G., Liu, J., & Huang, J. X. (2023). Incorporating global–local neighbors with Gaussian mixture embedding for few-shot knowledge graph completion. Expert Systems with Applications, 234, 121086. https://doi.org/10.1016/j.eswa.2023.121086

[6] Qiu, J., Sun, L., & Han, M. (2023). Improving Knowledge Base Updates with CAIA: A Method Utilizing Capsule Network and Attentive Intratriplet Association Features. Journal of Sensors, 2023(1). https://doi.org/10.1155/2023/9942486

[7] Xu, D., Zhu, H., Huang, Y., Jin, Z., Ding, W., Li, H., & Ran, M. (2023). Vision-knowledge fusion model for multi-domain medical report generation. Information Fusion, 97, 101817. https://doi.org/10.1016/j.inffus.2023.101817

[8] Liu, H., Hu, K., Wang, F.-L., & Hao, T. (2020). Aggregating neighborhood information for negative sampling for knowledge graph embedding. Neural Computing and Applications, 32(23), 17637–17653. https://doi.org/10.1007/s00521-020-04940-5

[9] L. R. Horn, H. Wansing. Negation. Stanford Encyclopedia of Philosophy. Zalta, E.N. (ed.). Stanford University, 2020. http://plato. stanford.edu/entries/negation/

[10] K. Makkar et al. Improvisation in Opinion Mining Using Negation Detection and Negation Handling Techniques: A Survey. Soft Computing: Theories and Applications. Lecture Notes in Networks and Systems 627, 2023, 799–808.

[11] Morante R, Blanco E. Recent advances in processing negation. Natural Language Engineering, 2021, 27(2): 121-130.

[12] Zhenghua Pan, Yong Wang. Three Kinds of Negation in Knowledge and Their Mathematical Foundations. arXiv:2505.24422 https://doi.org/10.48550/arXiv.2505.24422

[13] Zhenghua Pan, Yong Wang. Different negations and fuzziness in Web resources and resource description and their mathematical foundation, Information Sciences, 689 (2025), 121254. https://doi.org/10.1016/j.ins.2024.121254

[14] Straccia, U., Casini, G. (2022) A Minimal Deductive System for RDFS with Negative Statements. 19th Internationa Conference on Principles of Knowledge Representation and Reasoning, KR 2022. https://doi.org/10.24963/kr.2022/35

[15] Damásio, C., Analyti, A., Antoniou, G. (2010) Embeddings of simple modular extended RDF. Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics). https://doi.org/10.1007/978-3-642-15918-3\_17

[16] Damásio, C.V., Analyti, A., Antoniou, G. (2010) Implementing simple modular ERDF ontologies. Frontiers in Artificial Intelligence and Applications. https://doi.org/10.3233/978-1-60750-606-5-1083

[17] Arnaout, H., Razniewski, S., Weikum, G., Pan, J.Z. (2021) Negative statements considered useful. Journal of Web Semantics. https://doi.org/10.1016/j.websem.2021.100661

[18] Velios, Athanasios, Meghini, Carlo, Doerr, Martin, Stead, Stephen (2023) Typed properties and negative typed properties: Dealing with type observations and negative statements in the CIDOC CRM. Semantic Web. https://doi.org/10.3233/SW-223159

[19] Razniewski, S., Arnaout, H., Ghosh, S., Suchanek, F. (2024) Completeness, Recall, and Negation in Open-world Knowledge Bases: A Survey. ACM Computing Surveys. https://doi.org/10.1145/3639563

[20] Angles, R., Gutiérrez, C. (2016) Negation in SPARQL. CEUR Workshop Proceedings. https://www.scopus.com/pages/publications/84985961254

[21] Losemann, K., Martens, W. (2012) The complexity of evaluating path expressions in SPARQL. Proceedings of the ACM SIGACT-SIGMOD-SIGART Symposium on Principles of Database Systems. https://doi.org/10.1145/2213556.2213573

[22] Losemann, K., Martens, W. (2013) The Complexity of regular expressions and property paths in sparql. ACM Transactions on Database Systems. https://doi.org/10.1145/2494529

[23] Darari, F., Nutt, W., Razniewski, S., Rudolph, S. (2020) Completeness and soundness guarantees for conjunctive SPARQL queries over RDF data sources with completeness statements. Semantic Web. https://doi.org/10.3233/SW-190344

[24] Straccia, U., Casini, G. (2022) A Minimal Deductive System for RDFS with Negative Statements. 19th International Conference on Principles of Knowledge Representation and Reasoning, KR 2022. https://doi.org/10.24963/kr.2022/35

[25] Darari, F. (2013) Representing and querying negative knowledge in RDF. Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics). https://doi.org/10.1007/978-3-642-41242-4\_40

[26] Alkhateeb, F. (2016) Introducing wild-card and negation for optimizing SPARQL queries based on rewriting RDF graph and SPARQL queries. WEBIST 2016 - Proceedings of the 12th International Conference on Web Information Systems and Technologies. https://doi.org/10.5220/0005762001810187

[27] Arnaout, H., Razniewski, S. (2023) Can large language models generate salient negative statements?. CEUR Workshop Proceedings. https://www.scopus.com/pages/publications/85179558782

[28] K. Kangal. Engels’ Intentions in Dialectics of Nature. Science & Society, 2019, 83(2):215-243. DOI: 10.1521/siso.2019.83.2.215

[29] Mazzieri M, Dragoni A F. A fuzzy semantics for the resource description framework. URSW 2005–2007. 2008, 244–261

[30] Hayes, P. RDF Semantics. W3C Recommendation, 2004

[31] Schneider, M. A survey of RDF data management. The VLDB Journal, 2009, 19(1): 1-22

[32] Bordes, A., et al. Translating embeddings for modeling multi-relational data. Advances in Neural Information Processing Systems 26 (NeurIPS). 2013: 2787-2795

[33] Trouillon, T., et all. Complex embeddings for simple link prediction. Proceedings of the 33rd International Conference on Machine Learning (ICML). New York: PMLR, 2016: 2071-2080

[34] Schlichtkrull, M., et al. Modeling relational data with graph convolutional networks. Proceedings of the Extended Semantic Web Conference (ESWC). Cham: Springer, 2018: 593-607

[35] Yang, F., et al. Differentiable learning of logical rules for knowledge base reasoning. Advances in Neural Information Processing Systems 30 (NeurIPS). 2017: 2319-2328

[36] Lao, N., Cohen, W. W. Relational retrieval using a combination of path-constrained random walks. Machine Learning, 2010/2011.

[37] Nickel M, Murphy K, Tresp V, et al. A review of relational machine learning for knowledge graphs. Proceedings of the IEEE, 2016, 104(1): 11-33. DOI: 10.1109/JPROC.2015.2483592

[38] Galárraga, L., et al. AMIE: association rule mining under incomplete evidence in ontological knowledge bases. Proceedings of the 22nd International Conference on World Wide Web (WWW 2013). New York: ACM, 2013: 413-422

[39] Galárraga, L., et al. Fast rule mining in ontological knowledge bases with AMIE+. The VLDB Journal, 2015, 24(5): 707-730. DOI:10.1007/s00778-015-0394-1

[40] Ter Horst H J. Completeness, decidability and complexity of entailment for RDF Schema and a semantic extension involving the OWL vocabulary. Journal of Web Semantics, 2005, 3(1): 71-110

[41] Sadeghian, A, et al. DRUM: end-to-end differentiable rule mining on knowledge graphs. Advances in Neural Information Processing Systems 32 (NeurIPS 2019). Red Hook: Curran Associates, 2019

[42] Lewis, David K. (1973). Counterfactuals. Cambridge, MA: Harvard University Press

[43] Roese, N. J. (1997). Counterfactual thinking. Psychological bulletin, 121(1), 133

[44] Edgington, D. (2020). Counterfactual Conditionals. In The Routledge Handbook of Modality (pp. 30–39). Routledge. https://doi.org/10.4324/9781315742144-5

[45] Vorster, E. (2022). Does Counterfactual Reasoning Hold the Key to Artificial General Intelligence?. In: Pillay, A., Jembere, E.,

Gerber, A. (eds) Artificial Intelligence Research. SACAIR 2022. Communications in Computer and Information Science, vol 1734. Springer, Cham. https://doi.org/10.1007/978-3-031-22321-1\_26

[46] Fan, D. (2023). Focused true–true counterfactuals. The Philosophical Forum, 54(3), 121–141. https://doi.org/10.1111/phil.12337

[47] Kühl, C. E. (2023). On Counterfactual Reasoning. Danish Yearbook of Philosophy, 56(2), 154–181. https://doi.org/10.1163/24689300-bja10043

[48] Chou, Y. L., Moreira, C., Bruza, P., Ouyang, C., & Jorge, J. (2022). Counterfactuals and causability in explainable artificial intelligence: Theory, algorithms, and applications. Information Fusion, 81, 59-83. https://doi.org/10.1016/j.inffus.2021.11.003

[49] Cerami M , Pardo P . Many-Valued Semantics for Vague Counterfactuals. Studies in Logic, 36, 341-362 (2011)

[50] Ganter, B., & Wille, R. (1999). Formal Concept Analysis: Mathematical Foundations. Springer.

[51] Matheus, C. J., et al. 2006. BaseVISor: A Triples-Based Inference Engine Outfitted to Process RuleML and R-Entailment Rules. https://doi.org/10.21236/ada460530

[52] Y Hogan A, Blomqvist E, Cochez M, et al. Knowledge graphs. ACM Computing Surveys, 2021, 54(4): 1-37

[53] Liu, Chang et al. Fuzzy Reasoning over RDF Data Using OWL Vocabulary. 2011 IEEE/WIC/ACM International Conferences on Web Intelligence and Intelligent Agent Technology 1 (2011): 162-169.

[54] Xiaojun Chen, Shengbin Jia, Yang Xiang. A review: Knowledge reasoning over knowledge graph. Expert Systems With Applications, 2020, 141: 1–21.

[55] Xinliang Liu, Tingyu Mao, et al. Overview of knowledge reasoning for knowledge graph. Neurocomputing, Volume 585, 2024, 127571. https://doi.org/10.1016/j.neucom.2024.127571

[56] Bustince, H., Burillo, P.& Soria, F. Automorphisms, negations and implication operators. Fuzzy Sets & Systems, 2003, 134(2): 209–229

[57] Aguiló, I., Massanet, S., Riera, J.V., Ruiz-Aguilera, D. Modus Ponens Tollens for RU-Implications. In: Lesot, MJ., et al. Information Processing and Management of Uncertainty in Knowledge-Based Systems. IPMU 2020. Communications in Computer and Information Science, vol 1238, 2020. Springer, Cham. https://doi.org/10.1007/978-3-030-50143-3\_61