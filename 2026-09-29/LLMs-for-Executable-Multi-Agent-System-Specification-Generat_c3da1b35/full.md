# LLMs for Executable Multi-Agent System Specification Generation

Andreas Kouvaras<sup>1</sup>, Periklis Mantenoglou<sup>2</sup>, Alexander Artikis<sup>1,3</sup>

<sup>1</sup>University of Piraeus, Greece. <sup>2</sup>Orebro University, Sweden. <sup>¨</sup> <sup>3</sup>NCSR “Demokritos”, Greece.

Contributing authors: a.kouvaras@unipi.gr; periklis.mantenoglou@oru.se; a.artikis@unipi.gr;

## Abstract

MAS specifications express the efects of the actions of the agents and their environment, as well as other temporal phenomena, such as the intervals during which an agent may perform an action. The specification of a MAS should also be executable in order to allow for run-time monitoring. Constructing the specification of a MAS requires formal language expertise, while machine learning techniques depend on labelled data which are rarely available. To address these issues, we propose ‘genRTEC’, a method that leverages pre-trained Large Language Models (LLMs) to generate executable MAS specifications, in the language of the ‘Run-Time Event Calculus’ (RTEC), from natural language descriptions. genRTEC constructs MAS specifications with complex hierarchical and cyclic dependencies based only on short natural language descriptions of the concepts involved. We present an extensive empirical evaluation of genRTEC, spanning various MAS specifications, including both a qualitative and a quantitative assessment. Our results demonstrate that genRTEC constructs executable MAS specifications of high predictive accuracy without compromising reasoning eficiency.

Keywords: Event Calculus, agent interaction protocols, temporal specifications, temporal pattern matching

## 1 Introduction

Contemporary applications are increasingly being realised in terms of multi-agent systems (MAS) [44]. The specification of a MAS expresses the efects of the actions of the agents and their environment. Moreover, MAS are often viewed as instances of norm-governed systems [38], since actuality, what is the case, and ideality, what ought to be the case, do not necessarily coincide [50]. Therefore, a MAS specification should express the conditions in which an action is said to be permitted, prohibited and obligatory, and perhaps other more complex normative positions, as well as the enforcement strategies that handle deviations from ideality [39, 60]. In any case, MAS specifications need to explicitly represent temporal phenomena, such as the intervals during which the efects of an action persist, or the intervals during which an agent may perform some action. Consider e.g. an argumentation game where the agents must perform their chosen actions by specified deadlines [4]. Without this feature there is no practical way of controlling the exchanges, of determining whether an agent has ‘spoken’, because otherwise one might have to wait indefinitely for messages to arrive over the communication channels. As another example, in voting protocols a chair of the voting procedure may be forbidden to close the ballot of a motion before a specified time period has passed [57]. In e-commerce protocols repeated unsolicited quotes to the same consumer within a short time period may be prohibited, and may result to the (temporary) suspension of the merchant issuing them [76].

In addition to expressing temporal phenomena, the specification of a MAS should be executable, i.e. it should be possible to compute the efects of the actions of the agents at run-time, their normative positions, and any other temporal property of the specification. This way, it would be possible to determine at run-time whether e.g. agents comply to the specification, or whether an enforcement strategy should be applied. Such run-time monitoring should be scalable to MAS with large populations, and cater for complex specifications.

Several approaches have been proposed in the literature for developing MAS specifications. One common such approach concerns the use of the Event Calculus i.e. a logic formalism for representing events/actions, and reasoning about their efects over time [42]. For example, the Event Calculus has been employed for specifying ecommerce protocols [3, 19, 20], service level agreement management [30, 55], dynamic legal frameworks [48, 54, 62], as well as authorisation policy management [7, 66, 77, 78]. Unlike other formal frameworks [10, 11, 28, 70, 71], the Event Calculus expresses succinctly the temporal persistence of the efects of agents’ actions based on the common-sense law of inertia, represents explicitly durative phenomena, and allows for hierarchical specifications paving the way towards optimised reasoning.

Constructing MAS specifications in the Event Calculus (or any other temporal logic), however, is challenging. Domain experts may lack expertise in formal languages, while communicating the building blocks of a MAS specification to knowledge engineers can lead to information loss due to their complexity. Moreover, automatically constructing MAS specifications via specialised machine learning techniques requires large training datasets with labelled examples, e.g. of norm violation [32, 49]. Unfortunately, such datasets are scarce, prohibiting the use of learning techniques.

To address these issues, we propose a method that employs pre-trained Large Language Models (LLMs) to construct MAS specifications in the language of the ‘Run-Time Event Calculus’ (RTEC) [5], based on natural language descriptions of the building blocks of a MAS. RTEC extends the Event Calculus with optimisation techniques for temporal pattern matching over streaming data, such as streams of agent messages, outperforming state-of-the-art systems in terms of computational eficiency [47, 67]. RTEC, therefore, minimises the latency of the execution of a MAS specification.

The rapid evolution of LLMs has led to a class of models with the ability to elaborate on tasks by generating so-called ‘thinking tokens’, eliciting chain-of-thought reasoning [74]. Recent works have argued that the elaboration abilities of these models do not correspond to properly structured reasoning [27, 65, 68], emphasising their limited capabilities in multi-step logical reasoning tasks [18, 24, 35, 37] and in handling irrelevant information [51], while other works highlight their successes in solving complicated mathematical problems [26]. In any case, generating such thinking tokens is costly, in terms of both latency and monetary expenses, prohibiting the use of these models for reasoning over large, high-velocity streaming data, such as streams of agent transactions in large-scale MAS e-commerce. On the other hand, LLMs have proven efective translators for rewriting natural language descriptions into formal language expressions [24, 31, 35–37, 53, 73, 75].

We build upon this capacity for formal language generation, and present ‘gen-RTEC’, i.e. a prompting method employing pre-trained LLMs to translate natural descriptions of the building blocks of a MAS into an executable specification in the language of RTEC. genRTEC extends previous prompting techniques [41] by generalising them beyond domain-specific methods, as well as supporting the generation of specifications with cyclic dependencies, which are common in MAS. We present the pipeline of genRTEC and the prompting practices optimising performance. A key feature of genRTEC concerns the fact that it may be re-used, in a zero-shot manner, to generate executable specifications for any MAS. We conducted an extensive, reproducible empirical evaluation of genRTEC, employing leading LLMs, and spanning MAS interaction protocols with complex hierarchical and cyclic dependencies from ecommerce, voting and argumentation. To stress test genRTEC further, we instructed it to generate temporal specifications for four additional application domains.

We present a thorough quantitative assessment of the specifications constructed by genRTEC, calculating: (a) their similarity to hand-crafted counterparts acting as ground truth, thus estimating the human efort required to correct them; (b) their predictive accuracy on real and synthetic data; and (c) the computational overhead incurred when reasoning over the generated specifications as compared to the handcrafted ones. Additionally, we present a qualitative analysis, identifying the main classes of errors in the generated specifications. Our results demonstrate that gen-RTEC’s MAS specifications achieve high predictive accuracy without compromising reasoning eficiency. The instructions to reproduce the empirical analysis are publicly available<sup>1</sup>.

Running Example. We employ an argumentation protocol (ARG) based on the formalisation presented in [4], in order to illustrate genRTEC. There are three roles in ARG: proponent, opponent and determiner. We will deal with the usual case where there are three agents, one in each role. Briefly, the proponent claims a thesis, the opponent questions this thesis, and the determiner decides whether the proponent’s thesis was successfully defended or not. More precisely, the argumentation commences when the proponent claims the topic of the argumentation. The protagonists — the proponent and the opponent — then take it in turn to perform actions, i.e. claim, concede to, retract, or deny a proposition. Each turn lasts for a specified time period during which the protagonist may perform several actions. After each such action the other protagonist is given an opportunity to object. The determiner may declare the winner only at the end of the argumentation, i.e. when the specified period for the argumentation elapses. For example, if at the end of the argumentation both the proponent and opponent have accepted the topic of the argumentation, then the determiner may only declare the proponent the winner.

## 2 Related Work

MAS specifications are temporal specifications in the sense that they express the efects of the actions of the agents and their environment, as well as other temporal phenomena, such as the intervals during which an agent has the institutional power to perform an action, and thereby create a set of institutional facts [39], or the time until an agent should fulfill its commitments [22]. The specification of a MAS should also be executable, i.e. given the stream of the actions of the agents and the environment, it should be possible to compute the efects of these actions, the intervals of insitutional facts and brute facts [59], and any other temporal property of the specification. To cater for large agent populations, the execution of a specification over the streaming agent actions should be performed with minimal latency.

Several approaches have been proposed in the literature for developing MAS specifications. A typical approach concerns the use of the Event Calculus, i.e. a logic formalism for representing events/actions, and reasoning about their efects [42]. The Event Calculus has a built-in represenetation of the common-sense law of inertia, allowing for succinct formalisations, explicitly represents durative phenomena, avoiding the issues arising from point-based semantics [55], and supports hierarchical specifications, thus allowing the use of caching techniques for optimised reasoning [16]. There are several works that use the Event Calculus for specifying MAS. For instance, Symboleo is a formal specification language for smart contracts that employs the Event Calculus to define the lifecycle of contracts, as well as the obligations and the powers of agents as the system evolves [54, 62]. As another example, Zahoor et al. used the Event Calculus to model the dynamics of aggregated authorisation policies across multiple cloud providers [77], as well as to represent and reason about the state of independent authorisation policy objects in a Kubernetes cluster [78]. The Event Calculus has also been used for monitoring temporal phenomena in mobility assistance [14], reactive and proactive health monitoring [17, 40], simulations with cognitive agents [61], and simulations of biological feedback loops [64].

The Run-Time Event Calculus (RTEC) extends the Event Calculus with optimisation techniques, such as windowing and incremental caching, for temporal pattern matching over streaming data. It has been shown that RTEC outperforms competing systems in terms of computational eficiency [47], including s(CASP) [2], Fusemate [8] and jREC [13, 21, 29]. Moreover, RTEC supports temporal specifications with cyclic dependencies [46], which are common in MAS specifications, and streams with delayed actions [67], which are typical in distributed systems such as MAS.

The literature includes several non-Event Calculus-based frameworks for monitoring temporal phenomena over streaming data. CORE e.g. is an automata-based system with formal semantics and strong time complexity guarantees [15]. However, CORE has limited expressivity as it supports only unary relations, applied only to the last event read. Ticker [10] and Laser [9] are two stream reasoning frameworks that employ fragments of the LARS temporal specification language [11]. MeTeoR [71, 72] is a stream reasoning system that supports a fragment of the DatalogMTL formalism [70]. StreamMill [43] employs a minimal extension of SQL that supports stream reasoning. Ticker and Laser support negation only for globally stratified programs, while MeTeoR and StreamMill do not support negation in temporal patterns. In contrast to the aforementioned frameworks, RTEC supports temporal patterns with relational constraints, such as constraints between two or more agents, and negation in locally stratified logic programs [58]. Moreover, compared to the aforementioned approaches, RTEC inherits the benefits of the Event Calculus, such as the built-in representation of inertia and the explicit modelling of durative phenomena.

genRTEC employs pre-trained Large Language Models (LLMs) to construct executable MAS specifications in the language RTEC, based on natural language descriptions of the building blocks of a MAS. This way, MAS specifications may be constructed by users without expertise in formal languages. Moreover, unlike specialised learning techniques [25, 49], genRTEC does not require labelled training datasets. LLM-based methods for executable specification generation have also been proposed for other formalisms. Ishay et al. [36] employed LLMs to generate executable specifications in the + action language [6] in order to handle planning tasks. As opposed to the Event Calculus, + does not support an explicit representation of time, and thus cannot express e.g. the intervals during which a property is said to have some value. Moreover, reasoning in + is not optimised for streaming data and thus is not suitable for the run-time execution of MAS specifications. Guan et al. used LLMs to construct PDDL specifications for planning problems [35]. PDDL, however, is not suitable for monitoring MAS, as it cannot be used for reasoning over input actions. LLMs have also been used to generate SQL queries [31], streamlining constraints in constraint satisfaction problems [69], Answer Set Programming specifications [24, 37], and logic programs [73, 75]. These formalisms are not temporal, and thus cannot be used to specify MAS.

genRTEC constitutes a domain-agnostic LLM prompting technique that builds upon a previous approach, tailored for generating RTEC rules for composite maritime activities [41]. In other words, genRTEC may be used for the construction of any MAS specification. Furthermore, specification generation is achieved in a zero-shot manner, without requiring feedback. Note also that, unlike earlier work, genRTEC can handle specifications with cyclic dependencies.

<table><tr><td>Predicate</td><td>Meaning</td></tr><tr><td>happensAt(E, T)</td><td>Event E occurs at time-point T.</td></tr><tr><td>initiatedAt(F = V, T)</td><td>At time-point T, a period of time during which F = V is initiated.</td></tr><tr><td>terminatedAt(F = V, T)</td><td>At time-point T, a period of time during which F = V is terminated.</td></tr><tr><td>holdsFor(F = V, I)</td><td>I is the list of the maximal intervals during which F = V holds continuously.</td></tr><tr><td>holdsAt(F = V, T)</td><td>Fluent F has value V at time-point T.</td></tr><tr><td>union_all  $( [ J _ { 1 } , \ldots , J _ { n } ] , \ I )$ </td><td> $I = ( J _ { 1 } \cup . . . \cup J _ { n } )$ </td></tr><tr><td>intersect.  $\mathsf { a l l } ( [ J _ { 1 } , \ldots , J _ { n } ] , \ I )$ </td><td> $I = ( J _ { 1 } \cap . . . \cap J _ { n } )$ </td></tr><tr><td>relative_complement_all(I&#x27;, [J1 , . . . , Jn], I) I = I′ \ (J1 ∪ . . . ∪ Jn)</td><td></td></tr></table>

Table 1: The main predicates of RTEC.

## 3 Background: Run-Time Event Calculus

The Run-Time Event Calculus (RTEC) is a logic programming implementation of the Event Calculus [42]. Below, we briefly present the syntax of the language of RTEC and its semantics following [5, 47]. Moreover, we outline the key reasoning task of RTEC.

## 3.1 Syntax

The language of RTEC includes sorts for representing (a) time, (b) events, expressing the actions of the agents and their environment, and (c) fluents, i.e. properties whose values may change over time. RTEC employs a linear timeline with non-negative integer time-points. Variables start with an upper-case letter, while predicates and constants start with a lower-case letter. A fluent-value pair (FVP) F=V denotes that fluent F has value V. Boolean fluents are a special case where the possible values are true and false. Table 1 summarises the main predicates of RTEC. happensAt(E, T) signifies that action/event E occurs at time-point T. initiatedAt(F = V, T) (resp. terminatedAt( $F = V , T ) )$ expresses that a time period during which a fluent F has the value V continuously is initiated (terminated) at T. holdsAt(F = V, T) states that F has value V at T, while holdsFor $( F = V , I )$ expresses that F=V holds continuously in the maximal intervals of list I.

A MAS specification may be expressed as an RTEC event description, i.e. a set of rules defining ‘simple’ and ‘statically determined’ FVPs. A simple FVP is defined using a set of initiatedAt and terminatedAt rules, and is subject to the common-sense law of inertia, i.e. an FVP F=V holds at a time-point T, if F=V has been ‘initiated’ by an event at a time-point earlier than T, and not ‘terminated’ by another event in the meantime.

Example 1 (Concession). Recall that in ARG the protagonists — the proponent and the opponent — take it in turn to perform actions, i.e. claim, concede to, retract, or deny a proposition. The semantics of these actions are given in terms of the premises held by the protagonists. For example, the proponent’s claim of a proposition $Q$ may lead to an ‘explicit’ premise about Q for the proponent and an ‘unconfirmed’ premise about Q for the opponent. As another example, the rules below specify the efects of a concession:

$$
\begin{array} { r l } & { \mathrm { i n i t i a t e d A t } ( p r e m i s e ( P r o t a g , Q ) = e x p l i c i t , T ) \gets } \\ & { \quad \mathrm { ~ \ } \mathrm { \ h a p p e n s A t } ( c o n c e d e ( P r o t a g , Q ) , T ) , } \\ & { \quad \mathrm { \ h o l d s A t } ( p r e m i s e ( P r o t a g , Q ) = u n c o n f i r m e d , T ) , } \\ & { \quad \mathrm { \ n o t \ h o l d s A t } ( o b j e c t i o n a b l e ( P r o t a g , c o n c e d e , Q ) = \mathrm { t r u e } , T ) . } \\ & { \quad \mathrm { \ i n i t i a t e d A t } ( p r e m i s e ( P r o t a g , Q ) = e x p l i c i t , T ) \gets } \\ & { \quad \mathrm { \ h a p p e n s A t } ( c o n c e d e ( P r o t a g , Q ) , T ) , } \\ & { \quad \mathrm { \ h o l d s A t } ( p r e m i s e ( P r o t a g , Q ) = u n c o n f i r m e d , T ) , } \\ & { \quad \mathrm { \ n o t \ h a p p e n s A t } ( o b j e c t e d , T ) . } \end{array}\tag{1}
$$

(2)

premise(Protag, Q) is a simple fluent expressing the propositions Q for which a protagonist Protag has an explicit or unconfirmed premise. concede(Protag, Q) denotes that Protag concedes to proposition $Q ,$ , ‘not’ expresses negation-by-failure [23], objectionable(Protag, A, Q) is a fluent expressing whether action A about $Q ,$ performed by Protag, is said to be ‘objectionable’, and objected is an event denoting that at least one agent has expressed an objection. According to rules (1) and (2), a protagonist Protag is said to adopt an explicit premise about a proposition $Q$ when Protag concedes to $Q ,$ provided that Protag has an unconfirmed premise about $Q .$ and the concession is not objectionable or no agent objects to it. We adopt the simplistic treatment of objections of [4], according to which an objection to an action, such a concession, takes place at the same time as the action itself. A more elaborate formalisation of the objection mechanism will be considered in future work. ♢

Definition 1 (Syntax of Rules Defining Simple FVPs) The initiated $\mathsf { A t } ( F = V , T )$ rules concerning a simple FVP $F = V$ have the following syntax:

$$
\begin{array} { r l } & { \mathrm { i n i t i a t e d a t } ( F = V , ~ T ) \gets } \\ & { \mathrm { ~ \boldsymbol ~ { \hat { h } } a p p e n s i a t } ( E _ { I } , T ) [ \cdot , } \\ & { \mathrm { ~ \boldsymbol ~ { \hat { \rho } } [ n o t ] ~ \boldsymbol { h } a p p e n s i a t } ( E _ { \mathcal { L } } , T ) , } \\ & { \cdot \cdot \cdot , } \\ & { \mathrm { [ n o t ] ~ \boldsymbol { h } a p p e n s i a t } ( E _ { n } , ~ T ) , } \\ & { \mathrm { [ n o t ] ~ \boldsymbol { h } o l d s A t } ( F _ { I } = V _ { I } , T ) , } \\ & { \cdot \cdot , } \\ & { \mathrm { [ n o t ] ~ \boldsymbol { h } o l d s A t } ( F _ { k } = V _ { k } , T ) , } \\ & { \mathrm { a t e m p o r a l } . \mathrm { c o n s t r a n t s l } ) . } \end{array}
$$

The first body literal of an initiatedAt rule is a positive happensAt predicate; this is followed by a possibly empty set, denoted by $^ { 6 } [ [ ] ] ^ { , }$ , of positive/negative happensAt and holdsAt predicates, and atemporal constraints, i.e., a conjunction of atemporal predicates expressing background knowledge. ‘not’ expresses negation-by-failure, while ‘[not]’ denotes that ‘not’ is optional. In addition to application-specific events, $E _ { i } ,$ where $i \in [ 1 , n ]$ , may refer to the built-in events of RTEC, i.e. start $( \boldsymbol { F } ^ { \prime } = \boldsymbol { V } ^ { \prime } )$ or end $( \boldsymbol { F } ^ { \prime } = \boldsymbol { V } ^ { \prime } )$ , expressing, respectively, the time-points in which FVP $F ^ { \prime } = V ^ { \prime }$ is initiated or terminated [5]. All (head and body) predicates are evaluated at the same time-point T. The body of a terminated $\mathsf { A t } ( F = V , T )$ rule has the same form. ■

![](images/31023481333680487c31b787fbc9af4cf6ded89bccaeb4d30d3b5cf18663f9e2.jpg)  
Fig. 1: Interval manipulation constructs of RTEC. $I _ { 1 }$ , $I _ { 2 }$ and $I _ { 3 }$ (resp. $I _ { c } , I _ { i }$ and $I _ { u } )$ are input (output) lists of maximal intervals.

RTEC has built-in axioms ensuring that a simple fluent cannot have more than one value at any time; an initiation of fluent F with value $V ^ { \prime }$ implies the termination of $F = V$ , for all $V \neq V ^ { \prime }$ . Moreover, a fluent may not have a value at some timepoint(s). It is not the same, e.g. to initiate F = false and to terminate $F = { \mathrm { t r u e } } \colon$ the former implies, but is not implied by the latter.

A ‘statically determined’ FVP F = V is defined via a rule with head holdsFor $( F = V , I )$ . This rule computes the maximal intervals during which $F = V$ holds, and it may include one or more of the interval manipulation constructs of RTEC

see the last three items of Table 1. union all(L, I) (resp. intersect all(L, I)) computes the list of maximal intervals I as the union (intersection) of all lists of maximal intervals of list L. relative complement all $( I ^ { \prime } , L , I )$ computes the list of maximal intervals I by removing from the maximal intervals of list $I ^ { \prime }$ all interval segments included in an interval of some interval list in L. Figure 1 presents a visual illustration of the interval manipulation constructs.

Example 2 (Objectionable concession). In ARG, a concession is said to be objectionable when it is ‘proper’ but not ‘timely’:

$$
\begin{array} { r l } & { \mathsf { h o l d s F o r } ( o b j e c t i o n a b l e ( P r o t a g , c o n c e d e , Q ) = \mathsf { t r u e } , I ) \gets } \\ & { \quad \mathsf { h o l d s F o r } ( p r o p e r ( P r o t a g , c o n c e d e , Q ) = \mathsf { t r u e } , I _ { I } ) , } \\ & { \quad \mathsf { h o l d s F o r } ( t i m e l y = P r o t a g , I _ { \mathcal { L } } ) , } \\ & { \quad \mathsf { r e l a t i v e \_ c o m p l e m e n t \_ a l l } ( I _ { \cal I } , [ I _ { \mathcal { L } } ] , I ) . } \end{array}\tag{3}
$$

objectionable is a statically determined fluent, defined in terms of the proper and timely fluents. The list of maximal intervals I during which a concession by a protagonist Protag to proposition $Q$ is said to be objectionable is computed by the relative complement of the list of maximal intervals $I _ { 1 }$ during which the concession to $Q$ from Protag is said to be proper, and the list of maximal intervals $I _ { 2 }$ during which it is timely for Protag to speak. See [4] for the specification of proper and timely actions. $\diamondsuit$

Definition 2 (Syntax of Rules Defining Statically Determined FVPs) The definition of a statically determined FVP $F = V$ is a rule with the following syntax:

```prolog
holdsFor(F = V , I<sub>n+m</sub>) ←
holdsFor(F<sub>1</sub> = V<sub>1</sub> , I<sub>1</sub> )[[,
holdsFor(F<sub>2</sub> = V<sub>2</sub> , I<sub>2</sub> ),
holdsFor(F<sub>n</sub> = V<sub>n</sub>, I<sub>n</sub>),
intervalConstruct(L<sub>1</sub> , I<sub>n+1</sub> ),
intervalConstruct(L<sub>m</sub>, I<sub>n+m</sub>),
atemporal constraints]].
```

The first body literal of a holdsFor rule defining $F = V$ is a holdsFor predicate expressing the maximal intervals of an FVP other than $F = V$ . This is followed by a possibly empty list, denoted by ‘[[ ]]’, of holdsFor predicates, interval manipulation constructs, expressed by intervalConstruct, and atemporal constraints expressing background knowledge. An interval manipulation construct may be union a $\mathbb { I } ( L _ { j } , I _ { n + j } )$ , intersect all $( L _ { j } , I _ { n + j } )$ or relative complement a ${ | | } ( I _ { k } , L _ { j } , I _ { n + j } )$ . I<sub>k</sub> , where $k < n + j$ , is a list of maximal intervals appearing earlier in the body of the rule, and list $L _ { j }$ contains a subset of these lists. The output list $I _ { n + m }$ contains the maximal intervals during which $F = V$ holds continuously. ■

A statically determined FVP holds as long as a Boolean combination of other FVPs is satisfied. Typically, a statically determined $\mathrm { F V P }$ representation leads to more eficient reasoning, but not all simple FVPs are translatable to statically determined ones [45].

## 3.2 Semantics

An event description in RTEC defines a dependency graph expressing the relationships between its FVPs.

Definition 3 (Dependency Graph) The dependency graph of an event description is a directed graph such that:

1. Each vertex denotes an $\mathrm { F V P } \ F = V$

2. There exists an edge $( F _ { j } = V _ { j } , F _ { i } = V _ { i } )$ if there is an initiatedAt or terminatedAt rule for $F _ { i } = V _ { i }$ having holdsA $\mathsf { t } ( F _ { j } = V _ { j } , \ T )$ ) as one of its conditions.

3. There exists an edge $( F _ { j } \overset { \cdot } { = } V _ { j } , \overset { \cdot } { F } _ { i } = V _ { i } )$ if there is a holdsFor rule for $F _ { i } = V _ { i }$ having holdsFor $( F _ { j } = V _ { j } , \ \bar { T } )$ as one of its conditions. ■

Based on the dependency graph, it is possible to define a function level that maps FVPs to positive integers. For acyclic dependency graphs, the level of an FVP is equal to the level of its vertex.

Definition 4 (Vertex Level) Given a directed acyclic graph, the level of a vertex v is equal to:

1. 1 , if v has no incoming edges.

![](images/9fada7f8a1bf74f7868e875a399469c349a52e07b67367f25d345698b10f601e.jpg)  
Fig. 2: Dependency graph fragment of the event description for ARG. Underscores denote variables grounded on input entities, action types or agent roles.

2. n, where n > 1 , if v has at least one incoming edge from a vertex of level n−1 , while all its other incoming edges, if any, start from vertices with level n−1 or lower. ■

In the case that the dependency graph contains cycles, the level of each FVP is derived by first contracting the vertices of the FVPs participating in each cycle into a single vertex, and then applying Definition 4 on the resulting contracted dependency graph [46].

Example 3 (Dependency Graph and FVP Levels). Figure 2 presents a dependency graph fragment of the event description for ARG. The FVPs with fluent timely do not have incoming edges, as they depend only on the actions of the environment — timeouts issued by an external clock [4] — and thus have level 1. The definitions of FVPs with fluents proper , objectionable and premise exhibit cyclic dependencies. For instance, rule (1) stipulates that premise(ProtagRole, Q) = explicit depends on objectionable(ProtagRole, concede, Q) = true, while rule (3) states that objectionable(ProtagRole, concede, Q) = true depends on proper (ProtagRole, concede, Q) = true. The event description includes additional rules stating that FVPs with fluent proper depend on FVPs with fluent premise, completing the cycle. To derive the levels of these FVPs, their vertices are contracted into a single vertex and the level of the resulting vertex is calculated. This vertex has incoming edges only from vertices with level 1, and thus the levels of the FVPs taking part in this cycle are 2. sanctioned is a fluent expressing the penalties applied to agents for performing objectionable actions. sanctioned depends only on one other fluent, i.e. objectionable, and thus the FVPs with fluent sanctioned have level 3. ♢ Proposition 1 (Semantics of RTEC). An event description in RTEC is a locally stratified logic program [58]. ♦

A stratification of an event description may be constructed as follows. The first stratum contains all groundings of happensAt(E, T) atoms, expressing the actions of agents and their environment. The remaining strata may be formed following the FVP levels in ascending order.

![](images/f410e426d6d95c0a385b04184492f81c4a93067c4ca3ecc01c7141c571964439.jpg)  
Fig. 3: genRTEC: LLM prompting for executable MAS specification generation. In this illustration, genRTEC is used for the construction of the specifications of a negotation protocol, a voting protocol and an argumentation protocol. Prompt B is optional.

## 3.3 Reasoning

The key reasoning task of RTEC is to compute holdsFor $( F = V , I )$ , i.e. the list of maximal intervals I during which an FVP $F = V$ of an event description holds continuously. For example, we may want to compute the list of maximal intervals during which an action is said to be proper in ARG, or the maximal intervals during which a protagonist is said to be sanctioned. A statically determined FVP $F = V$ is defined via a rule r with head holdsFor $( F = V , I )$ ; RTEC derives the list of maximal intervals I by evaluating the conditions of r. For a simple FVP $F = V$ , RTEC first computes the initiation time-points of $F = V$ by evaluating the rules with head initiatedAt $( F = V , T )$ . Then, RTEC computes the termination time-points of $F = V$ by evaluating the rules with head terminated $\mathsf { A t } ( F = V , T )$ or initiated $\mathsf { A t } ( F = V ^ { \prime } , T )$ , where $V ^ { \prime } \neq V$ . Subsequently, RTEC computes the maximal intervals of $F = V \mathrm { \ b y }$ matching each initiation $T _ { s }$ of $F { = } V$ with the first termination $T _ { e }$ of $F { = } V$ after $T _ { s }$ , ignoring every intermediate initiation between $T _ { s }$ and T<sub>e</sub>. Once the list of maximal intervals is computed, RTEC may derive holdsAt $( F = V , T )$ by checking whether T belongs to one of the maximal intervals of $F { = } V$ . RTEC includes various optimisation techniques in order to scale to high-velocity data streams. The reader is referred to [45–47] for details.

## 4 Generating Executable Specifications

Hand-crafting RTEC event descriptions requires formal language expertise, while learning such event descriptions demands large, annotated training datasets [49]. To address these issues, we present genRTEC, a prompting method that leverages the power of pre-trained LLMs for constructing executable MAS specifications, i.e. event descriptions in the language of RTEC.

## 4.1 Prompting Pipeline

Figure 3 illustrates the workflow of genRTEC. Initially, we introduce the language of RTEC to the LLM under consideration (in Section 5 we present an empirical analysis with three leading LLMs). We use prompt R to introduce the core predicates of RTEC — see Listing 1. Then, we issue prompt S to provide the syntax for the rules of simple and statically determined FVPs (see Definitions 1 and 2). For each FVP representation, we provide natural language descriptions of two example time-varying properties, such as a premise in ARG, and show how these properties may be translated into RTEC rules. Through these examples, we guide the LLM in the task of mapping natural language into RTEC elements, i.e. fluents, events, initiatedAt, terminatedAt and holdsFor rules, allowing automated rule generation. All prompts, including prompt S, are available at the repository of genRTEC<sup>1</sup>.

1 You are an assistant in constructing rules in the language of the Run-Time   
Event Calculus (RTEC), given a natural language description. The Event   
Calculus is a logic-based formalism for representing and reasoning about   
events and their effects. RTEC is a Prolog implementation of the Event   
Calculus, that has been optimised for stream reasoning. Below, we summarise   
the language of RTEC.   
2   
3 Following the Prolog convention, variables start with an upper-case letter,   
while predicates and constants start with a lower-case letter. Each rule ends   
with a full-stop ‘.’, while the head of a rule is separated from its body   
with ‘:-’.   
4   
5 A fluent is a property that may have different values at different points in   
time. The term F=V denotes that fluent F has value V. Boolean fluents are a   
special case in which the possible values are ‘true’ and ‘false’.   
6   
7 Below are the predicates of RTEC.   
8   
9 RTEC - Predicate 1: happensAt(E,T)   
Meaning: Event ‘E’ occurs at time ‘T’.   
11   
12 RTEC - Predicate 2: holdsAt(F=V,T)   
13 Meaning: The value of fluent ‘F’ is ‘V’ at time ‘T’.   
14   
15 RTEC - Predicate 3: holdsFor(F=V,I)   
16 Meaning: ‘I’ is the list of the maximal intervals during which ‘F=V’ holds   
continuously.   
17   
18 RTEC - Predicate 4: initiatedAt(F=V,T)   
19 Meaning: At time ‘T’, a period of time for which ‘F=V’ is initiated.   
20   
21 RTEC - Predicate 5: terminatedAt(F=V,T)   
22 Meaning: At time ‘T’, a period of time for which ‘F=V’ is terminated.   
23   
24 RTEC - Predicate 6: union\_all(L,I)   
25 Meaning: ‘I’ is the list of maximal intervals produced by the union of the   
lists of maximal intervals of list ‘L’.   
26   
27 RTEC - Predicate 7: intersect\_all(L,I)   
28 Meaning: ‘I’ is the list of maximal intervals produced by the intersection of   
the lists of maximal intervals of list ‘L’.   
29

30 RTEC - Predicate 8: relative\_complement\_all(In,L,I)   
31 Meaning: ‘I’ is the list of maximal intervals produced by the relative   
complement of the list of maximal intervals ‘In’ with respect to every list   
of maximal intervals of list ‘L’.   
32   
33 RTEC also includes two built-in events.   
34   
35 Built-in event 1: start(F=V)   
36 Meaning: Event ‘start(F=V)’ takes place at the starting point of each maximal   
interval of fluent-value pair ‘F=V’.   
37   
38 Built-in event 2: end(F=V)   
39 Meaning: Event ‘end(F=V)’ takes place at the ending point of each maximal   
interval of fluent-value pair ‘F=V’.

## Listing 1: Prompt R.

After introducing the language of RTEC, we may instruct genRTEC to generate an executable specification for a MAS. Prompt E presents the actions of the agents and their environment, and prompt B presents the background knowledge predicates, if any. These predicates may appear in the bodies of rules — see atemporal constraints in Definitions 1 and 2. Prompts G then ask the LLM to translate a natural language description of each component of the MAS, such as the conditions in which an action is said to be proper in ARG, into a set of RTEC rules. A key feature of genRTEC is that it is not custom to a particular MAS, but may generate executable specifications for any MAS. To achieve this, one needs to customise and repeat prompts E, B and G for each MAS (see Figure 3). In contrast, prompts R and S are used once and are not repeated. Note that MAS specification generation is achieved in a zero-shot manner, i.e. it is not required to provide any feedback to the LLMs.

Prior to rule generation, we need to provide to the LLM the actions of the agents/environment. These actions are presented in Prompt E — below, we present a fragment of prompt E for ARG including the actions of the agents.

1 You may use the following ARG actions:   
2   
3 ARG - Action 1: claim(Protag, Q)   
4 Meaning: Protagonist ‘Protag’, i.e., an agent occopying the role of proponent   
or opponent, claims proposition ‘Q’.   
5   
6 ARG - Action 2: concede(Protag,Q)   
7 Meaning: Protagonist ‘Protag’, i.e., an agent occupying the role of proponent   
or opponent, concedes to proposition ‘Q’.   
8   
9 ARG - Action 3: retract(Protag,Q)   
10 Meaning: Protagonist ‘Protag’, i.e., an agent occupying the role of proponent   
or opponent, retracts its commitment to proposition ‘Q’.   
11   
12 ARG - Action 4: deny(Protag,Q)

Listing 2: Prompt E (ARG).  
13 Meaning: Protagonist ‘Protag’, i.e., an agent occupying the role of proponent   
or opponent, denies proposition ‘Q’.   
14   
15 ARG - Action 5: objected   
16 Meaning: At least one agent objects to the preceding action.

After prompt E, we provide a natural language description of each component of a MAS, and prompt the LLM to express it in the language of RTEC. Below, we present two such examples concerning ARG, in which an LLM is asked to generate rules expressing the efects of a ‘claim’ action, and the specification of the conditions in which an action is said to be proper. The complete set of prompts is available in the repository of genRTEC<sup>1</sup>.

1 Given the description below, provide the rules in the language of RTEC. You   
may use the built-in events of RTEC and the ARG actions. You may also use any   
of the ARG output fluents, i.e. the fluents that we will define together.   
2   
3 Description - ‘premise - claim’: We aim to identify the effects of a claim.   
First, a protagonist adopts an explicit premise about a proposition Q, when   
this protagonist claims Q, and does not have an unconfirmed premise about Q.   
Second, a protagonist adopts an unconfirmed premise about proposition Q, when   
this protagonist does not have an explicit premise about Q and the other   
protagonist claims Q. The aforementioned effects are realised provided that   
(1) the claim is not objectionable, or (2) no agent objects to the claim. The   
conditions in which an action is said to be ‘objectionable’ will be   
presented later.

Given the description below, provide the rules in the language of RTEC. You   
may use the built-in events of RTEC and the ARG actions. You may also use any   
of the ARG output fluents, i.e. the fluents that we will define together.   
2   
3 Description - ‘proper’: We aim to identify the maximal intervals during which   
an action is considered proper for a protagonist. First, a claim of a   
proposition Q by a protagonist is considered proper as long as this   
protagonist has neither an explicit nor an unconfirmed premise about Q.   
Second, a concession to a proposition Q by a protagonist is considered proper   
as long as this protagonist has an unconfirmed premise about Q. Third, a   
retraction of a proposition Q by a protagonist is considered proper as long   
as this protagonist has an explicit premise about Q. Fourth, a denial of a   
proposition Q by a protagonist is considered proper as long as this   
protagonist has an unconfirmed premise about Q.  
Listing 4: ARG – proper actions (Prompt G).

## 4.2 Optimising Performance

Through a series of experiments, we identified a set of best practices that optimise the performance of genRTEC. First, we find it useful to guide the LLM to make use of the actions that have been provided with prompt E. Second, we instruct the LLM to take into consideration any of the fluents that have been specified so far. See the reference to ‘output fluents’ in line 1 of Listings 3 and 4. By encouraging the LLM to reuse previously constructed specifications, we can build a hierarchical knowledge base in which higher-level patterns depend on lower-level ones. Such hierarchies yield compact RTEC event descriptions, enable caching of intermediate results, and improve reasoning eficiency [5]. Third, we either state explicitly the conditions in which a timevarying property starts/stops taking place (Listing 3), or state the conditions that must be satisfied while the property is in efect (Listing 4). Fourth, in cases where there are multiple facets in the definition of a property, we employ explicit enumeration, e.g. ‘(1)’, ‘(2)’, as this helps the LLM to establish the required number of related rules.

Fifth, it is important to notify early the LLMs about cyclic dependencies, such as those in ARG — see Figure 2. We provide, with the use of prompts G, the definitions of the building blocks of a MAS to an LLM one at a time. In the presence of cyclic dependencies, the definition of a building block may refer to another building block that has not yet been introduced. Consider Figure 2; the definition of premise depends on objectionable. Therefore, when prompting an LLM for the specification of premise, the conditions in which an action is said to be objectionable may not have yet been presented. In such cases, LLMs tend to generate inconsistent rule-sets. To address this issue, we inform the LLM in advance that the definitions of some building blocks will be provided later. See e.g. the last sentence in Listing 3.

Listings 3 and 4 are quite involved, describing intricate definitions of the efects of claims and proper actions. Nevertheless, the use of the prompting practices presented above allowed genRTEC to construct ‘perfect’ rule-sets, i.e. specifications identical to the ground truth (in the section that follows, we will present the metrics with which we evaluate the output of genRTEC). For instance, genRTEC responded to the prompt of Listing 3 with the rule-set below:

$$
\begin{array} { r l } & { \mathrm { i n i t i a t e d A t } ( p r e m i s e ( P r o t a g , Q ) = e x p l i c i t , T )  } \\ & { \quad \mathrm { h a p p e n s A t } ( c l a i m ( P r o t a g , Q ) , T ) , } \\ & { \quad \mathrm { n o t h o l d s A t } ( p r e m i s e ( P r o t a g , Q ) = u n c o n f i r m e d , T ) , } \\ & { \quad \mathrm { n o t h o l d s A t } ( o b j e c t i o n a b l e ( P r o t a g , c l a i m , Q ) = \mathrm { t r u e } , T ) . } \end{array}\tag{4}
$$

$$
\begin{array} { r l } & { \mathsf { i n i t i a t e d A t } ( p r e m i s e ( P r o t a g , Q ) = e x p l i c i t , T ) \gets } \\ & { \quad \mathsf { h a p p e n s A t } ( c l a i m ( P r o t a g , Q ) , T ) , } \\ & { \quad \mathsf { n o t h o l d s A t } ( p r e m i s e ( P r o t a g , Q ) = u n c o n f i r m e d , T ) , } \\ & { \quad \mathsf { n o t h a p p e n s A t } ( o b j e c t e d , T ) . } \end{array}\tag{5}
$$

(6)

$$
\begin{array} { r l } & { \mathsf { i n i t i a t e d A t } ( p r e m i s e ( P r o t a g _ { \mathcal { Q } } , Q ) = u n c o n f i r m e d , T ) \gets } \\ & { \quad \mathsf { h a p p e n s A t } ( c l a i m ( P r o t a g , Q ) , T ) , } \\ & { \quad P r o t a g \neq P r o t a g _ { \mathcal { Q } } , } \\ & { \mathsf { n o t h o l d s A t } ( p r e m i s e ( P r o t a g _ { \mathcal { Q } } , Q ) = e x p l i c i t , T ) , } \\ & { \quad \mathsf { n o t h a p p e n s A t } ( o b j e c t e d , T ) . } \end{array}\tag{7}
$$

Recall that premise(Protag, Q) is a fluent expressing the propositions Q for which a protagonist Protag, i.e. the proponent or the opponent, has an explicit or unconfirmed premise; objectionable is a fluent denoting whether an action is said to be objectionable, and objected expresses the objections of agents. claim(Protag, Q) denotes that Protag claims proposition Q.

Similar to the prompt of Listing 3, genRTEC was able to generate a perfect specification for the the prompt of Listing 4 — see the rule-set below:

$$
\mathsf { h o l d s F o r } ( p r o p e r ( P r o t a g , c l a i m , Q ) = \mathsf { t r u e } , I ) \gets
$$

$$
\begin{array} { r } { \mathtt { h o l d s F o r } ( p r e m i s e ( P r o t a g , Q ) = e x p l i c i t , I _ { \mathit { 1 } } ) , } \end{array}
$$

$$
\mathsf { h o l d s F o r } ( p r e m i s e ( P r o t a g , Q ) = u n c o n f i r m e d , I _ { 2 } ) ,\tag{8}
$$

$$
\mathsf { r e l a t i v e \_ c o m p l e m e n t \_ a l l } ( \omega , [ I _ { 1 } , I _ { 2 } ] , I ) .
$$

$$
\scriptstyle \mathtt { h o l d s F o r } ( p r o p e r ( P r o t a g , c o n c e d e , Q ) = \mathtt { t r u e } , I ) \gets
$$

$$
\mathsf { h o l d s F o r } ( p r e m i s e ( P r o t a g , Q ) = u n c o n f i r m e d , I ) .\tag{9}
$$

$$
\mathsf { h o l d s F o r } ( p r o p e r ( P r o t a g , r e t r a c t , Q ) = \mathsf { t r u e } , I ) \gets
$$

$$
\begin{array} { r } { \mathtt { h o l d s F o r } ( p r e m i s e ( P r o t a g , Q ) = e x p l i c i t , I ) . } \end{array}\tag{10}
$$

$$
\mathsf { h o l d s F o r } ( p r o p e r ( P r o t a g , d e n y , Q ) = \mathsf { t r u e } , I ) \gets
$$

$$
\mathsf { h o l d s F o r } ( p r e m i s e ( P r o t a g , Q ) = u n c o n f i r m e d , I ) .\tag{11}
$$

Recall that relative complement all $( I ^ { \prime } , L , I )$ computes the list of maximal intervals I by removing from the maximal intervals of list $I ^ { \prime }$ all interval segments included in an interval of some list in L. Placing $\cdot _ { \omega } ,$ in the first argument of relative complement all implies that we are interested in the absolute complement of the intervals of the lists of L.

Once we have generated an RTEC event description expressing a MAS specification, such as the specification of an argumentation protocol, we may proceed with another MAS by customising and repeating prompts E, (B) and G. Recall that executable specification generation for each MAS is achieved in a zero-shot manner, i.e. no example or feedback is provided in prompts E, B and G.

## 5 Empirical Evaluation

## 5.1 Experimental Setup

## 5.1.1 Applications

We evaluated genRTEC by generating executable specifications for three MAS interaction protocols, i.e. the Argumentation Protocol (ARG) [4] that we have used as a running example, a Negotiation Protocol based on NetBill (NET) [3, 63, 76], and a Voting Protocol (VP) based on the formalisations of [46, 57]. NET specifies the ways in which consumers interact with merchants to buy/sell digital goods. VP may be summarised as follows: a committee sits and the chair opens the meeting; a member proposes a motion; another member seconds the motion; the members debate the motion; the chair calls for those in favour/against to cast their vote; finally, the motion is carried, or not, according to the standing rules of the committee. In NET, VP and ARG, the task is to generate specifications that may be used by RTEC in order to compute, among others, the maximal intervals during which agents hold normative positions, such as institutional power, permission and obligation, as well as the sanctions applied in the cases of non-conformance to obligations and performance of forbidden actions. In NET e.g. a contract defines a set of normative positions for the contracting parties; in VP the chair of the voting process is forbidden to close the ballot earlier than a specified time; and in ARG the participants are sanctioned when performing objectionable actions.

To stress test our approach further, we evaluated genRTEC by generating temporal specifications for four additional applications, i.e. Maritime Situational Awareness (MSA) [56], City Transport Management (CTM) [5], Activity Recognition (ACR) [5], and conformance checking with Clinical Guidelines (CG) [12]. In MSA, we aim to generate specifications that express various types of illegal, suspicious or dangerous (autonomous) vessel activities. Given such specifications, and a stream of vessel positional signals, RTEC may identify, at run-time, instances of such activities. In CTM, we require specifications that may be used to monitor the performance of (autonomous) public transport vehicles. In ACR, the task is to generate specifications that may be used to recognise composite activities, such as a person leaving an object unattended, given streams of surveillance video frames annotated with symbolic information. In CG, the generated specifications should allow RTEC to assess whether the actions of medical personnel adhere to the prescribed guidelines.

## 5.1.2 Evaluation Metrics

We evaluate the specifications constructed by genRTEC by means of the following three metrics. First, we calculate the f1-score of the generated specifications, i.e. we compare RTEC’s computed maximal intervals when operating over the genRTECconstructed specifications, against the maximal intervals obtained when reasoning over hand-crafted specifications acting as ground truth. This way, we aim to estimate the predictive accuracy of the generated specifications. Second, we compute the syntactic similarity of the generated specifications to the hand-crafted ones — this metric will be presented shortly. Note that a perfect f1-score does not necessarily imply perfect syntactic similarity (although perfect syntactic similarity implies a perfect f1-score). The use of syntactic similarity allows us to estimate the human efort required for correcting the issues, if any, of the generated specifications. Third, we compute the reasoning time achieved by RTEC when operating over the genRTEC-constructed specifications, and compare it against the reasoning time achieved when operating over the hand-crafted specifications.

To assess the syntactic similarity of genRTEC-constructed specifications with hand-crafted counterparts, we employed the metric of [41]. The values of this metric range from 0 to 1, with higher values indicating higher similarity. The metric calculates the similarity between two RTEC specifications, i.e., sets of RTEC rules, in a recursive manner; their similarity is defined as the average similarity between the rules they contain, following a 1-to-1 rule mapping that maximises rule similarity. Analogously, the similarity between two RTEC rules is equal to the average similarity of their logical literals, following a similarity-maximising mapping. The similarity between two logical literals is 0 if they use a diferent predicate name or have diferent arities. Otherwise, their similarity is derived by comparing one-by-one their arguments, granting a perfect similarity score 1 to constant pairs with the same name and variable pairs with the same appearances in the corresponding rules.

Table 2: Size of ground truth specifications and datasets
<table><tr><td></td><td>NET</td><td>VP</td><td>ARG</td><td>MSA</td><td>CTM</td><td>ACR</td><td>CG</td></tr><tr><td>Rules</td><td>15</td><td>15</td><td>24</td><td>32</td><td>23</td><td>9</td><td>19</td></tr><tr><td>Conditions</td><td>27</td><td>29</td><td>61</td><td>105</td><td>52</td><td>31</td><td>30</td></tr><tr><td>Cyclic Dependencies</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Dataset Items</td><td>132K</td><td>100K</td><td>100K</td><td>15M</td><td>50K</td><td>182K</td><td>N/A</td></tr></table>

To evaluate the eficiency of RTEC when operating over the genRTEC-generated specifications, we measured the average reasoning time of RTEC within each sliding window, i.e. the incrementally updated, bounded portion of the stream of agent actions currently held in memory [5].

## 5.1.3 Ground Truth and Datasets

For NET, VP, MSA, ACR and CTM, there are publicly available hand-crafted event descriptions in the language of RTEC that may be used as ground truth<sup>2</sup>. Concerning CG, some indicative Event Calculus rules are presented in [12], but these are only a small fragment of the event description, and are not entirely consistent with the syntax of the language of RTEC. For the employed argumentation protocol (ARG) [4], there is no Event Calculus specification available (the specification in [4] is in the action language + [33]). For these reasons, we had to construct the event descriptions for ARG and CG ourselves in order to use them as ground truth. We have not made these event descriptions publicly available in order to make sure that LLMs will not be trained on them. Table 2 presents the size of the hand-crafted specifications acting as ground truth. ARG e.g. includes 24 rules, having a total of 61 conditions. Table 2 also indicates whether there are cyclic dependencies in the hand-crafted specifications. Figure 4 presents the dependency graphs of the hand-crafted event descriptions of NET, VP and ARG. The dependency graphs of the remaining applications are presented in the Appendix.

To calculate the f1-score of the genRTEC-constructed specifications and the eficiency of RTEC when reasoning over these specifications, we made use of publicly available datasets. In the case of NET and VP, we used datasets comprising agent interactions that were produced by synthetic data generators with realistic parameter values [46]. For MSA, we employed a real dataset containing position signals emitted by vessels that sailed in the Atlantic Ocean around the port of Brest, France, between October 2015–March $2 0 1 6 ^ { 3 }$ . In the case of CTM, we employed a synthetic dataset that was generated by simulating the operation of public transport vehicles in Helsinki, Finland [4]. For ACR, we used the CAVIAR benchmark dataset<sup>4</sup>. All these datasets are available with the code of RTEC<sup>2</sup>. Unfortunately, we could not find datasets for ARG and CG. For ARG, we created a synthetic dataset following [4], while for CG we restricted attention to syntactic similarity. Table 2 presents the size of each dataset. The dataset of ARG e.g. includes approx. 100,000 messages exchanged between the agents.

![](images/8a01d53f652bcf841fd0792b3cf2f326313d0ff63a63b9eb4d668cc85f07aae5.jpg)  
(a) NET

![](images/d18d5447ba08db9dc4a6a5cd98f2964c4bfc47e2a45fb44c340a2fbc37271f6d.jpg)  
(b) VP

![](images/1cf49eae7afa7bab7f50271a7c961c8620f511123d5cb5a641293c675f88f5af.jpg)  
(c) ARG  
Fig. 4: The dependency graphs of the hand-crafted specifications of NET, VP and ARG. To avoid clutter, we omitted the FVPs from the nodes and the edges in a cycle. The nodes in a cycle are coloured red, blue or green. We grouped together the nodes of the same level, i.e. NET has four levels, VP has three levels, and ARG has four levels.

## 5.1.4 Prompting

We employed three leading LLMs, i.e. GPT-5 [52], Gemini 2.5 Pro [34] and Claude Sonnet-4 [1]. For brevity, in what follows we refer to them as GPT, Gemini and Claude. We prompted each LLM, using the default hyper-parameter values, five times for each of the seven applications, and report the average values and standard deviations. In the experiments that follow, we used examples from MSA for prompt S of genRTEC (see Section 4). We then issued customised versions of prompts E, B and G to complete the MSA specification, and subsequently generated specifications for CTM, ACR, CG, NET, VP and ARG. The prompts are available in the repository of genRTEC<sup>1</sup>. The experiments were executed on a PC equipped with an Intel Core i7-1165G7 processor (2.80 GHz) and 16 GB of RAM.

![](images/59ad42b9fc71fa8ee113ed04bc174afd37af12f843867ca44d09979d28cba7a9.jpg)  
(a) NET

![](images/3fb9830a5ad879f5a33646f8badf3bd39585e3193557dd54b59d4f6a44c95e8a.jpg)  
(b) VP

![](images/e946da659eb237d2b650376357ad2a9e8b02c69938d2af7f43ccbadef46321c4.jpg)  
(c) ARG

![](images/72a7a818411ea6bf4b08438ae8d2e870d2669af78b7f9bf86342b6b8700feb99.jpg)  
(d) MSA

![](images/a4b0675ed9afcc44673a901b08ea702c7295ab382b7e5a9df9ec16db2ca7449f.jpg)  
(e) CTM

![](images/72b376494bcb2ae8fd8fd7148a88320a86b9ccc607feef9d77c2cd5b3b2e079e.jpg)  
(f) ACR

![](images/e291f451b9ae3695aa73e8e376b745a0d38713cf4b73f3e7d630b6ae8694a768.jpg)  
(g) CG  
Fig. 5: Syntactic similarity (line-hatched bars) and f1-score (dotted bars) of genRTECconstructed specifications compared to ground truth specifications. $^ \circ \triangle ^ { \prime }$ , ‘▲’ and ‘■’ refer to the use of GPT, Gemini and Claude.

![](images/5c8dca8bd4e4e25fbe4422047da8a7484ef87bb1f7bdbaec2fb882a5e214f013.jpg)  
(a) NET

![](images/7184a30b5349a217485f16dd778521b031a195aefea9a69b835dafc21c671567.jpg)  
(b) VP

![](images/b36ebdbca577a70d8771169b93f56fe3f3f69c89e71198b2db711734e4615c8e.jpg)  
(c) ARG

![](images/a1bd7f3104ca2fbd9869dc196b245e1fe2b20c479d6efbac61e93a8f701c23ce.jpg)  
(d) MSA

![](images/ffa37286ca62fcc9f7fbdceaab0e6af9df6876f664ef884104334caff0742348.jpg)  
(e) CTM

![](images/ee8dc8c3170840e08421a33ec86e78056f628ab5bca942431422206e1a5ba1ca.jpg)  
(f) ACR  
Fig. 6: Reasoning times of RTEC when operating on genRTEC-constructed specifications and ground truth specifications. $\mathbf { \hat { \Sigma } } { \bf \Sigma } ^ { \prime } \triangle \vec { \bf \Phi } , \mathbf { \hat { \Sigma } } { \bf \Sigma } ^ { \prime } \mathbf { \Sigma }$ and ‘■’ refer to the use of GPT, Gemini and Claude, and $^ 6 \mathrm { o } ^ { \dag }$ refers to the ground truth specifications. The ranges of values in the vertical axes difer.

## 5.2 Experimental Results

## 5.2.1 Quantitative Analysis

Figure 5 presents our experimental results with respect to syntactic similarity and f1-score, while Figure 6 presents our results concerning eficiency. Recall that for CG there is no dataset available, and thus we can only report results for syntactic similarity. Figure 5 shows that the specifications generated by genRTEC match the predictive accuracy of the hand-crafted specifications, i.e. in all applications for which we have available data, there is at least one LLM with which genRTEC achieves a(n almost) perfect f1-score. For example, genRTEC employing Claude achieves a perfect f1-score in NET, VP, CTM and ACR, and an f1-score just short of perfect in MSA. In ARG, genRTEC achieves a perfect f1-score when employing GPT. This is a very encouraging result, especially considering that the applications under consideration require intricate specifications that include numerous rules with complicated conditions. See Table 2, Figure 4 and e.g. rule-sets (4)–(7) and (8)–(11). At the same time, Figure 6 shows that, with very few exceptions, the top-performing LLMs, in terms of f1-score, generate specifications of comparable complexity with that of the ground truth specifications. Compare e.g. the reasoning times of RTEC when operating on the NET and VP (resp. ARG) specifications generated with the use of Claude (resp. GPT), against the reasoning times of RTEC when operating on the corresponding ground truth specifications (see Figures 6a, 6b and 6c). This result indicates that we do not have to sacrifice predictive accuracy for eficiency, or the other way around.

Another notable result concerns the fact that cyclic dependencies, such as those found in the specifications of NET, VP and ARG, did not compromise the performance of genRTEC. The prompting techniques of genRTEC (see Section 4), such as explicitly informing an LLM that some of the building blocks in the body of a rule will be defined at a later stage, allowed genRTEC to capture the cyclic dependencies expressed in the hand-crafted specifications.

The experimental results are very encouraging also in CG. Figure 5g shows that the generated specifications have an almost perfect syntactic similarity score.

## 5.2.2 Qualitative Analysis

To understand the performance of genRTEC, we conducted a qualitative analysis of the generated specifications, and outline below the main issues that reduced syntactic similarity, f1-score or eficiency.

Logical connective misrepresentation. In Example 1 we presented the efects of conceding to a proposition in ARG. To produce rules expressing the efects of this action, we issued the following prompt:

1 Description - ‘premise(Protag, Q)’: We aim to identify the effects of a   
concession. A protagonist adopts an explicit premise about proposition Q,   
when this protagonist concedes to Q, provided that this protagonist has an   
unconfirmed premise about Q. The aforementioned effects are realised provided   
that (1) the concession is not objectionable, or (2) no agent objects to the   
concession. The conditions in which an action is said to be ‘objectionable’   
will be presented later.  
Listing 5: ARG – efects of concession (Prompt G).

# In some cases, genRTEC employing Claude responded with the following rule:

initiatedAt(premise(Protag, Q) = explicit, T)

happensAt(concede(Protag, Q), T),

holdsAt(premise(Protag, Q) = unconfirmed, T),

(12)

not holdsAt(objectionable(Protag, concede, Q) = true, T),

not happensAt(objected, T).

According to rule (12), the efects of a concession are realised provided that the concession is not objectionable and that no agent objects to it. This is an unnecessarily strict rule; e.g. if a concession is not objectionable, then its efects should be realised, irrespective of whether some agent objected to it. Furthermore, according to the ground truth, i.e. rules (1) and (2), the efects of a concession are realised even if the concession is objectionable, provided that no agent objects to it. In other words, the disjunction between ‘objectionable concession’ and any objections to it, expressed in the prompt of Listing 5 — see the penultimate sentence — has become a conjunction in rule (12). This issue afected the syntactic similarity score of genRTEC/Claude in ARG. More importantly, it penalised the f1-score because the efects of concessions in the generated specifications were realised in much fewer cases as compared to the ground truth specifications. Conversely, the reasoning times of RTEC when operating on the specifications generated with genRTEC/Claude were lower than the reasoning times of RTEC when operating on the hand-crafted rules, since the former specifications led to fewer/shorter intervals for premise.

Missing rules. We observed that, in some cases, the generated specifications included a smaller rule-set than that of the hand-crafted specifications. In VP e.g. we issued the following prompt in order to produce rules recording the votes of agents:

1 Description - ‘voted’: We aim to record the way each agent has voted. Casting   
a vote, i.e., aye or nay, is recorded provided that the status of the motion   
in question is voting. An agent’s vote is set to null when a new voting   
round begins, i.e., when the status of the motion becomes null.

The corresponding hand-crafted specification consists of the following rules:

$$
\mathsf { i n i t i a t e d A t } ( v o t e d ( V , M ) = a y e , T ) \gets
$$

$$
\mathsf { h a p p e n s A t } ( v o t e ( V , M , a y e ) , T ) ,\tag{13}
$$

$$
\mathsf { h o l d s A t } ( s t a t u s ( M ) = v o t i n g , T ) .
$$

$$
\operatorname { i n i t i a t e d A t } ( v o t e d ( V , M ) = n a y , T ) 
$$

$$
\mathsf { h a p p e n s A t } ( v o t e ( V , M , n a y ) , T ) ,\tag{14}
$$

$$
\mathsf { h o l d s A t } ( s t a t u s ( M ) = v o t i n g , T ) .
$$

$$
\mathsf { i n i t i a t e d A t } ( v o t e d ( V , M ) = n u l l , T ) \gets
$$

$$
\mathsf { h a p p e n s A t } ( \mathsf { s t a r t } ( s t a t u s ( M ) = n u l l ) , T ) .\tag{15}
$$

voted(V, M) is a fluent recording the vote of voter V on motion M, vote represents the act of voting, and status is a fluent expressing the status of a motion. start $( F = V )$ is a built-in event of RTEC taking place at the time-points in which FVP F = V is initiated (see Definition 1). genRTEC employing Gemini omitted the generation of rule (15). Consequently, the votes of agents are never reset to ‘null’, and thus may persist in future voting rounds, possibly afecting the outcome of future voting procedures. This issue afected both the syntactic similarity score and the f1-score of genRTEC/Gemini in VP. We also observed that genRTEC/Gemini sometimes produced smaller rule-sets, as compared to ground truth, in ARG.

Fluent arity errors. In some cases, genRTEC made errors in the arity of fluents. Consider e.g. the specification of obligations in NET; to generate such a specification, we issued the prompt below (to simplify the presentation, we only show a fragment of the prompt):

1 Description - ‘obligation’: We aim to identify whether there is a set of obligations on the contracting parties, i.e., the merchant and the consumer. First, the consumer starts being obliged to send an electronic payment order to the Intermediation Server (iServer) when the contract between the merchant and the consumer starts being in effect. [..]

$$
\mathrm { { \bf L i s t i n g \ 7 : N E T - o b l i g a t i o n s \ ( P r o m p t \ G ) . } }
$$

In some cases, genRTEC generated the following rule:

$$
\begin{array} { r l } & { \mathsf { i n i t i a t e d A t } \big ( o b l \big ( s e n d . E P O ( C o n s , M e r c h , i S e r v e r , G D ) \big ) = \mathsf { t r u e } , T \big ) \gets } \\ & { \quad \mathsf { h a p p e n s A t } \big ( \mathsf { s t a r t } \big ( c o n t r a c t \big ( M e r c h , C o n s , G D \big ) = \mathsf { t r u e } \big ) , T \big ) . } \end{array}\tag{16}
$$

According to rule (16), a consumer Cons becomes obliged to send an electronic payment order (EPO) about goods GD when the contract between Cons and merchant Merch about GD starts being in efect. Rule (16) is very similar to the ground truth rule. The only diference lies in the fact that the ground truth rule does not include Merch in the head of the rule — Cons is obliged to send the EPO to the intermediation server iServer, not the merchant (see the prompt of Listing 7).

![](images/e0a1881d04966612d8c018b307e4660e8ff0ceb7afbbeac4d6180145d60ec011.jpg)  
Listing 8: VP – status of a motion (Prompt G).  
In response to this prompt, genRTEC employing GPT sometimes generated the following rule-set:

<table><tr><td>initiatedAt(status(M) = proposed, T) ← happensAt(propose(P, M), T), holdsAt(status(M) = null, T).</td><td>terminatedAt(status(M) = null, T) ← happensAt(propose(P, M), T), holdsAt(status(M) = null, T).</td></tr><tr><td>initiatedAt(status(M) = voting, T) ← happensAt(second(S, M), T), holdsAt(status(M) = proposed, T).</td><td>terminatedAt(status(M) = proposed, T) ← happensAt(second(S, M), T), holdsAt(status(M) = proposed, T).</td></tr><tr><td>initiatedAt(status(M) = voted, T) ←</td><td>terminatedAt(status(M) = voting, T) ←</td></tr><tr><td>happensAt(close_ballot(C, M), T), happensAt(role_of(C) = chair, T),</td><td>happensAt(close_ballot(C, M), T), happensAt(role_of(C) = chair, T),</td></tr><tr><td>holdsAt(status(M) = voting, T).</td><td>holdsAt(status(M) = voting, T).</td></tr><tr><td>initiatedAt(status(M) = null, T) ←</td><td>terminatedAt(status(M) = voted, T) ←</td></tr><tr><td></td><td></td></tr><tr><td>happensAt(declare(C, M, -), T),</td><td>happensAt(declare(C, M, -), T),</td></tr><tr><td>happensAt(role_of(C) = chair, T),</td><td>happensAt(role_of(C) = chair, T),</td></tr><tr><td>holdsAt(status(M) = voted, T).</td><td>holdsAt(status(M) = voted, T).</td></tr><tr><td></td><td></td></tr></table>

propose, second, close ballot and declare express the actions of the agents; e.g. close ballot(C, M) states that the chair C closes the ballot on motion M. role of is a fluent denoting the roles of agents. The rules on the left column consist of the ground truth specification of status(M). The rules on the right column are superfluous. Consider e.g. the top two rules — both rules have the same body literals. The presence of the top-left rule makes the top-right rule superfluous. Recall that RTEC has built-in axioms ensuring that a simple fluent cannot have more than one value at any time; an initiation of $F = V$ implies the termination of $F = V ^ { \prime }$ , for all $V \neq V ^ { \prime }$ (see Section 3). This issue afected the similarity score of genRTEC when employing GPT and Gemini in VP — see Figure 5b. Thankfully, this issue does not afect the f1-score, and has negligible efect on reasoning eficiency. In ACR, the genRTEC-constructed specifications contain redundant conditions, which penalise syntactic similarity without afecting maximal interval computation, although they slightly increase reasoning times. The issue afected the use of all LLMs in ACR.

## 6 Summary and Future Work

Hand-crafting MAS specifications requires knowledge of formal languages, while their automated construction via machine learning requires large and labelled training datasets, which are not always available. To address these issues, we proposed gen-RTEC, i.e. a prompting method employing pre-trained LLMs to translate natural descriptions of the building blocks of a MAS into an executable specification in the language of RTEC. We presented the pipeline of genRTEC and the prompting practices optimising performance. A key feature of genRTEC concerns the fact that it may be re-used, in a zero-shot manner, to generate executable specifications for any MAS.

Our extensive empirical analysis, including leading LLMs, and spanning MAS interaction protocols with complex hierarchical and cyclic dependencies, showed that the generated MAS specifications achieve high predictive accuracy without compromising reasoning eficiency.

In the future, we aim to develop self-revision techniques [36], such as requesting from the compiler of RTEC [45] to evaluate the output of an LLM — e.g. the dependency graph corresponding to a generated MAS specification — and provide feedback to the LLMs. Furthermore, we aim to fine-tune manageable versions of LLMs for avoiding the errors highlighted in our qualitative analysis.

Acknowledgements. This work was supported partly by the EU-funded CREX-DATA project (101092749), and partly by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation.

## References

[1] Anthropic (2025) Introducing claude sonnet 4.5. https://www.anthropic.com/ news/claude-sonnet-4-5, accessed: 2025-12-13

[2] Arias J, Carro M, Chen Z, et al (2022) Modeling and reasoning in event calculus using goal-directed constraint answer set programming. Theory Pract Log Program 22(1):51–80

[3] Artikis A, Sergot M (2009) Executable specification of open multi-agent systems. Logic Journal of the IGPL 18(1):31–65

[4] Artikis A, Sergot MJ, Pitt J (2007) An executable specification of a formal argumentation protocol. Artif Intell 171(10-15):776–804

[5] Artikis A, Sergot MJ, Paliouras G (2015) An event calculus for event recognition. IEEE Trans Knowl Data Eng 27(4):895–908

[6] Babb J, Lee J (2020) Action language +. J Log Comput 30(4):899–922

[7] Bandara AK, Lupu E, Russo A (2003) Using event calculus to formalise policy specification and analysis. In: 4th IEEE International Workshop on Policies for Distributed Systems and Networks (POLICY 2003), 4-6 June 2003, Lake Como, Italy. IEEE Computer Society, p 26

[8] Baumgartner P (2021) Combining event calculus and description logic reasoning via logic programming. In: FroCoS, pp 98–117

[9] Bazoobandi HR, Beck H, Urbani J (2017) Expressive stream reasoning with laser. In: ISWC, pp 87–103

[10] Beck H, Eiter T, Folie C (2017) Ticker: A system for incremental asp-based stream reasoning. Theory Pract Log Program 17(5-6):744–763

[11] Beck H, Dao-Tran M, Eiter T (2018) LARS: A logic-based framework for analytic reasoning over streams. Artif Intell 261:16–70

[12] Bottrighi A, Chesani F, Mello P, et al (2012) Conformance checking of executed clinical guidelines in presence of basic medical knowledge. In: Daniel F, Barkaoui K, Dustdar S (eds) Business Process Management Workshops. Springer Berlin Heidelberg, Berlin, Heidelberg, pp 200–211

[13] Bragaglia S, Chesani F, Mello P, et al (2012) Reactive event calculus for monitoring global computing applications. In: Logic Programs, Norms and Action - Essays in Honor of Marek J. Sergot on the Occasion of His 60th Birthday, pp 123–146

[14] Bromuri S, Urovi V, Stathis K (2010) icampus: A connected campus in the ambient event calculus. Int J Ambient Comput Intell 2(1):59–65

[15] Bucchi M, Grez A, Quintana A, et al (2022) CORE: a complex event recognition engine. Proc VLDB Endow 15(9):1951–1964

[16] Cervesato I, Montanari A (2000) A calculus of macro-events: Progress report. In: TIME, pp 47–58

[17] Chaudet H (2006) Extending the event calculus for tracking epidemic spread. Artif Intell Medicine 38(2):137–156

[18] Cheng F, Li H, Liu F, et al (2025) Empowering llms with logical reasoning: A comprehensive survey. In: Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI 2025, August 16-22, 2025. ijcai.org, Montreal, Canada, pp 10400–10408

[19] Chesani F, Mello P, Montali M, et al (2009) Commitment tracking via the reactive event calculus. In: IJCAI, pp 91–96

[20] Chesani F, Mello P, Montali M, et al (2013) Representing and monitoring social commitments using the event calculus. Auton Agents Multi Agent Syst 27(1):85– 130

[21] Chittaro L, Montanari A (1996) Eficient temporal reasoning in the cached event calculus. Comput Intell 12(3):359–382

[22] Chopra AK, V. SHC, Singh MP (2020) An evaluation of communication protocol languages for engineering multiagent systems. J Artif Intell Res 69:1351–1393

[23] Clark KL (1977) Negation as failure. In: Logic and Data Bases. Plenum Press, New York, NY, USA, pp 293–322

[24] Coppolillo E, Calimeri F, Manco G, et al (2024) Llasp: Fine-tuning large language models for answer set programming. In: KR, pp 834–844

[25] Corapi D, Russo A, Vos MD, et al (2011) Normative design using inductive learning. Theory Pract Log Program 11(4-5):783–799

[26] DeepSeek-AI, Guo D, Yang D, et al (2025) Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Nature 645:633–638

[27] Dziri N, Lu X, Sclar M, et al (2023) Faith and fate: Limits of transformers on compositionality. In: NeurIPS

[28] Eiter T, Ogris P, Schekotihin K (2019) A distributed approach to LARS stream reasoning (system paper). Theory Pract Log Program 19(5-6):974–989

[29] Falcionelli N, Sernani P, de la Torre AB, et al (2019) Indexing the event calculus: Towards practical human-readable personal health systems. Artif Intell Medicine 96:154–166

[30] Farrell ADH, Sergot MJ, Sall´e M, et al (2005) Using the event calculus for tracking the normative state of contracts. Int J Cooperative Inf Syst 14(2-3):99–129

[31] Gao D, Wang H, Li Y, et al (2024) Text-to-sql empowered by large language models: A benchmark evaluation. Proc VLDB Endow 17(5):1132–1145

[32] George L, Cadonna B, Weidlich M (2016) Il-miner: Instance-level discovery of complex event patterns. Proc VLDB Endow 10(1):25–36

[33] Giunchiglia E, Lee J, Lifschitz V, et al (2004) Nonmonotonic causal theories. Artif Intell 153(1-2):49–104

[34] Google DeepMind (2025) Gemini 2.5: Our most intelligent ai model. https://blog.google/technology/google-deepmind/ gemini-model-thinking-updates-march-2025/, accessed: 2025-12-13

[35] Guan L, Valmeekam K, Sreedharan S, et al (2023) Leveraging pre-trained large language models to construct and utilize world models for model-based task planning. In: NeurIPS

[36] Ishay A, Lee J (2025) LLM+AL: bridging large language models and action languages for complex reasoning about actions. In: Walsh T, Shah J, Kolter Z (eds) AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025. AAAI Press, Philadelphia, PA, USA, pp 24212–24220

[37] Ishay A, Yang Z, Lee J (2023) Leveraging large language models to generate answer set programs. In: KR, pp 374–383

[38] Jones AJI, Sergot MJ (1992) Deontic logic in the representation of law: Towards a methodology. Artif Intell Law 1(1):45–64

[39] Jones AJI, Sergot MJ (1996) A formal characterisation of institutionalised power. Log J IGPL 4(3):427–443

[40] Kafali O, Romero AE, Stathis K (2017) Agent-oriented activity recognition in the<sup>¨</sup> event calculus: An application for diabetic patients. Comput Intell 33(4):899–925

[41] Kouvaras A, Mantenoglou P, Artikis A (2025) Generating activity definitions with large language models. In: EDBT, pp 1005–1013

[42] Kowalski R, Sergot M (1986) A logic-based calculus of events. New Gen Computing 4(1):67–96

[43] Laptev N, Mozafari B, Mousavi H, et al (2016) Extending relational query languages for data streams. In: Data Stream Management - Processing High-Speed Data Streams. Data-Centric Systems and Applications, Springer, Berlin / Heidelberg, Germany, p 361–386

[44] Malfa EL, Malfa GL, Marro S, et al (2025) Large language models miss the multi-agent mark. CoRR abs/2505.21298

[45] Mantenoglou P, Artikis A (2025) Temporal specification optimisation for the event calculus. In: AAAI-25, pp 15075–15082

[46] Mantenoglou P, Pitsikalis M, Artikis A (2022) Stream reasoning with cycles. In: KR, pp 544–553

[47] Mantenoglou P, Pitsikalis M, Artikis A (2025) Reasoning over streams of events with delayed efects. J Artif Intell Res 84

[48] Mar´ın RH, Sartor G (1999) Time and norms: a formalisation in the event-calculus. In: Bing J, Jones AJI, Gordon TF (eds) Proceedings of the Seventh International Conference on Artificial Intelligence and Law, ICAIL ’99, Oslo, Norway, June 14-17, 1999. ACM, New York, NY, USA, pp 90–99

[49] Michelioudakis E, Artikis A, Paliouras G (2024) Online semi-supervised learning of composite event rules by combining structure and mass-based predicate similarity. Mach Learn 113(3):1445–1481

[50] Minsky NH, Ungureanu V (2000) Law-governed interaction: a coordination and control mechanism for heterogeneous distributed systems. ACM Trans Softw Eng Methodol 9(3):273–305

[51] Opedal A, Zengafinen Y, Shirakami H, et al (2025) Are language models eficient reasoners? A perspective from logic programming. CoRR abs/2510.25626

[52] OpenAI (2025) Introducing GPT-5. https://openai.com/index/ introducing-gpt-5/, accessed: 2025-12-13

[53] Pan L, Albalak A, Wang X, et al (2023) Logic-lm: Empowering large language models with symbolic solvers for faithful logical reasoning. In: Findings of the Association of Computational Linguistics: EMNLP, pp 3806–3824

[54] Parvizimosaed A, Sharifi S, Amyot D, et al (2022) Specification and analysis of legal contracts with symboleo. Softw Syst Model 21(6):2395–2427

[55] Paschke A, Bichler M (2008) Knowledge representation concepts for automated SLA management. Decis Support Syst 46(1):187–205

[56] Pitsikalis M, Artikis A, Dreo R, et al (2019) Composite event recognition for maritime monitoring. In: DEBS, pp 163–174

[57] Pitt J, Kamara LD, Sergot MJ, et al (2006) Voting in multi-agent systems. Comput J 49(2):156–170

[58] Przymusinski T (1987) On the declarate semantics of stratified deductive databases and logic programs. In: Foundations of Deductive Databases and Logic Programming

[59] Searle J (1996) What is a speech act? In: Martinich A (ed) Philosophy of Language, 3rd edn. Oxford University Press, p 130–140

[60] Sergot MJ (2001) A computational theory of normative positions. ACM Trans Comput Log 2(4):581–622

[61] Shahid NS, O’Keefe D, Stathis K (2023) A knowledge representation framework for evolutionary simulations with cognitive agents. In: ICTAI. IEEE, Piscataway, NJ, USA, pp 361–368

[62] Sharifi S, Parvizimosaed A, Amyot D, et al (2020) Symboleo: Towards a specification language for legal contracts. In: IEEE RE, pp 364–369

[63] Sirbu MA, Tygar JD (1995) Netbill: an internet commerce system optimized for network-delivered services. IEEE Wirel Commun 2(4):34–39

[64] Srinivasan A, Bain M, Baskar A (2022) Learning explanations for biological feedback with delays using an event calculus. Mach Learn 111(7):2435–2487

[65] Stechly K, Valmeekam K, Kambhampati S (2024) Chain of thoughtlessness? an analysis of cot in planning. In: NeurIPS

[66] Tonti G, Bradshaw JM, Jefers R, et al (2003) Semantic web languages for policy representation and reasoning: A comparison of kaos, rei, and ponder. In: Fensel D, Sycara KP, Mylopoulos J (eds) The Semantic Web - ISWC 2003, Second International Semantic Web Conference, Sanibel Island, FL, USA, October 20- 23, 2003, Proceedings, Lecture Notes in Computer Science, vol 2870. Springer, pp 419–437

[67] Tsilionis E, Artikis A, Paliouras G (2022) Incremental event calculus for run-time reasoning. J Artif Intell Res 73:967–1023

[68] Valmeekam K, Marquez M, Sreedharan S, et al (2023) On the planning abilities of large language models - A critical investigation. In: NeurIPS

[69] Voboril F, Ramaswamy VP, Szeider S (2025) Generating streamlining constraints with large language models. J Artif Intell Res 84

[70] Walega PA, Kaminski M, Grau BC (2019) Reasoning over streaming data in metric temporal datalog. In: AAAI, pp 3092–3099

[71] Walega PA, Kaminski M, Wang D, et al (2023) Stream reasoning with datalogmtl. J Web Semant 76:100776

[72] Wang D, Grau BC, Walega PA, et al (2025) Practical reasoning in datalogmtl. Theory Pract Log Program 25(2):225–255

[73] Wang Z, Liu J, Bao Q, et al (2024) Chatlogic: Integrating logic programming with large language models for multi-step reasoning. In: IJCNN, pp 1–8

[74] Wei J, Wang X, Schuurmans D, et al (2022) Chain-of-thought prompting elicits reasoning in large language models. In: NeurIPS

[75] Yang Z, Ishay A, Lee J (2023) Coupling large language models with logic programming for robust and general reasoning from text. In: ACL, pp 5186–5219

[76] Yolum P, Singh MP (2004) Reasoning about commitments in the event calculus: An approach for specifying and executing protocols. Ann Math Artif Intell 42(1- 3):227–253

[77] Zahoor E, Ikram A, Akhtar S, et al (2022) A formal approach for the identification of authorization policy conflicts within multi-cloud environments. J Grid Comput 20(2):18

[78] Zahoor E, Chaudhary M, Akhtar S, et al (2023) A formal approach for the identification of redundant authorization policies in kubernetes. Comput Secur 135:103473

![](images/517903550bbc3d2d81ade1dbd2516b5fc1b710a3f50e0a8af4876812c32c652f.jpg)  
(a) MSA

![](images/5cbdaa2454305e51253f288f8270d84a4c64b6c4ae5b3a8145d941ddfe792c81.jpg)

(b) CTM  
![](images/f5362a7cd8c53f01ce915acfd6bb2660e043cb747ca346dcacf1dfb1dd61bfcf.jpg)  
(c) ACR

![](images/0813f165aea63a73cf79cf77805f95d081bf706403c759fcced3b29f061f4fda.jpg)  
(d) CG  
Fig. A1: The dependency graphs of the hand-crafted specifications of MSA, CTM, ACR and CG. The use of red colour indicates a cyclic dependency.