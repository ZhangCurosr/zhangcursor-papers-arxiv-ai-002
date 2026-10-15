# Recursive Formation and First-Class Reuse of Compositional Capabilities

Hang Lin

October 2026

## Abstract

We introduce FormCraft, a framework for representing acquired operations as first-class capabilities that can be recursively composed, directly invoked, and selectively retained in a persistent library through domain-specific verification. The framework separates capability well-formedness and public callability from later admission to persistent storage while preserving formation provenance and operational identity. We formalize recursive compositional closure over a uniform capability space and show that formed composites can be reused through the same public interface without re-forming their historical derivation. Experiments across continuous, discrete, propositional, and sequence domains demonstrate instance-independent reuse, independent checking, persistent acquisition, and reduced reuse-time search work under first-class exposure.

## 1 Introduction

Program synthesis turns specifications or examples into executable programs, using search, constraints, and learned guidance [3, 5, 4, 8, 9]. Library-learning systems go further by extracting abstractions from solved tasks and making them useful for later synthesis [10, 11, 12]. These advances motivate a question about the resulting artifacts: once a useful operation has been formed, what makes it a reusable operational object rather than only a derivation or a search result, and how can that object later be retained?

We study that question as a capability lifecycle. A formed operation should be represented independently of the instance that elicited it, composed through compatible public contracts, and exposed for later invocation as an operation in its own right. Long-term persistence is a separate concern: a capability may subsequently be checked before it is retained in a persistent library. The framework therefore centers on representation, composition, persistence, and reuse after formation, while abstraction-discovery methods can serve as upstream sources of capabilities. In elementary arithmetic, an operation is more reusable than the example that first demonstrated it. The analogy captures the intended separation between a concrete instance and an operation that accepts new operands.

FormCraft represents primitive capabilities f and composite capabilities F within one well-formed capability space F. Instance data X and parameters α enter through public invocation ports, while a composite retains its formation provenance. Compatible recursive composition produces another first-class capability in the same space; verification is invoked only when a formed capability is considered for persistent library membership. Proposition 2 establishes finite compositional closure, and Corollary 5 isolates the consequence of first-class exposure for historical re-formation depth.

The evaluation progresses from instance-independent reuse to independently checked composition, persistent acquisition, and increasing-depth first-class reuse. Continuous and discrete prototypes examine instance separation; a Lean-checked propositional microdomain tests independently verifiable composites [14]. In the main sequence-domain study, sequential acquisition expands the persistent store from 12 to 26 and then 38 records. A paired reuse experiment over the increasing-depth task family reduces candidate attempts from 16,570 to 144 when persisted composites are exposed directly through their public interfaces. The paired study reports reuse-time search work under controlled snapshots.

The contributions are:

1. A formal account of structural axioms and first-class compositional closure for instanceindependent capabilities.

2. A lifecycle that separates well-formed capability formation and first-class reuse from the later verification decision governing persistent library membership.

3. A record-based implementation that gives primitive and composite capabilities one public operational interface while retaining operation identity and formation provenance.

4. Cross-domain evaluation of instance-independent reuse and independently checked composition, together with sequential persistent-store growth and a paired comparison of first-class reuse with historical expansion.

Section 2 gives the formalization; Section 3 describes its implementation; Section 4 reports the evidence; and Section 5 positions the work.

## 2 Formal Model

## 2.1 Primitive Capability and Operational Semantics

Let $X \in { \mathcal { X } }$ be a concrete task instance and $\alpha \in { \mathcal { A } }$ an external invocation parameter. A primitive capability is denoted by $f ,$ a composite by $F ,$ and a generic capability of either kind by G. The space $\mathfrak { F }$ contains well-formed capability representations. Its members have declared input, parameter, and output contracts and share the execution form

$$
\mathbb { I } G \mathbb { I } ( X ; \alpha ) = Y , \qquad G \in \mathfrak { F } .\tag{1}
$$

A representation records an operation and its contract, while a particular invocation supplies its own X and α. Formation provenance is denoted by $P ( G )$ . Throughout, notation such as $X \not \in f$ is representational shorthand: it means that the capability representation of $f$ contains no concrete invocation payload $X ,$ , not that $f$ is literally a set. Behavioral equivalence $\sim _ { F }$ means equality of observable denotations under a shared compatible contract; it is defined below.

## 2.2 Foundational Axioms

Axiom 1 (Primitive representation and recursive formation).

$$
\begin{array} { r l } { \mathrm { A _ { 1 } } \quad \mathrm { ( I n s t a n c e ~ E x c l u s i o n ) : } \quad } & { X \notin f , } \\ { \mathrm { A _ { 2 } } \quad \mathrm { ( E x t e r n a l ~ P a r a m e t e r i z a t i o n ) : } \quad } & { \alpha \ i s \ e x t e r n a l , } \\ { \mathrm { A _ { 3 } } \quad \mathrm { ( U n i f o r m ~ O p e r a t i o n a l ~ I n t e r f a c e ) : } \quad } & { G : \ { \mathcal K } _ { G } \times { \mathcal A } _ { G } \longrightarrow { \mathcal V } _ { G } \quad ( G = f \ o r \ F ) , } \\ { \mathrm { A _ { 4 } } \quad \mathrm { ( R e c u r s i v e ~ F o r m a t i o n ) : } \quad } & { G _ { i } , G _ { j } \in \mathfrak { F } , \ \mathrm { C o m p a t i b l e } ( G _ { i } , G _ { j } , \omega ) , } \\ & { \Longrightarrow F = \mathrm { C o m p o s e } _ { \omega } ( G _ { i } , G _ { j } ) \in \mathfrak { F } . } \end{array}
$$

Axiom $\mathrm { A } _ { 1 }$ excludes a concrete instance payload from the primitive representation without forbidding learning from instances. Axiom $\mathrm { A _ { 2 } }$ permits a stored parameter schema but places invocation-specific values at public ports. By $\mathrm { A _ { 3 } }$ , a primitive and composite difer in formation history, not public operational status. The terminology follows the programming-language tradition of treating procedures and functions as first-class values [1]; within the capability space, we use first-class to mean that a formed composite is exposed through the same public interface as other capabilities: it can be directly invoked, retained, and supplied as an operand to later compatible formation.

In $\mathrm { A } _ { 4 } ,$ , compatibility covers contracts, ports, and the static rule $\omega .$ The constructor records capability references, contracts, wiring, and provenance, but captures neither runtime X nor α. Either parent may already be composite, allowing $f  F _ { 1 }  F _ { 2 }  \cdot \cdot \cdot$ . Finite closure of this recursive rule is established separately by Proposition 2.

## 2.3 Recursive Composition and Semantic Associativity

Definition 1 (Compositional denotation). For a well-formed composite $F = { \mathrm { C o m p o s e } } _ { \omega } ( G _ { i } , G _ { j } )$ , its denotation is induced by the denotations of its parents and the compatible static wiring ω:

$$
[ [ F ] ] = [ [ G _ { i } ] ] \circ _ { \omega } [ [ G _ { j } ] ] .\tag{2}
$$

Here $\mathsf { o } _ { \omega }$ denotes the behavior induced by the typed wiring rule rather than unconditional ordinary function composition.

For any composite capability expression $E \in \mathfrak { F } ^ { * }$ , let $W ( E )$ denote its flattened typed wiring: the public-port dependency structure obtained by recursively expanding composite nodes while preserving their external contracts. This flattening removes only the internal compositional nesting from the wiring view; formation provenance $P ( E )$ , parent references, and the recorded formation DAG remain recorded separately. We write ◦ below for the appropriate compatible Compose at each node.

Proposition 1 (Semantic associativity under wiring equivalence). Suppose both $\left( G _ { 1 } \circ G _ { 2 } \right) \circ G _ { 3 }$ and $G _ { 1 } \circ ( G _ { 2 } \circ G _ { 3 } )$ are contract-compatible and have the same flattened typed wiring,

$$
{ \cal W } \big ( ( G _ { 1 } \circ G _ { 2 } ) \circ G _ { 3 } \big ) = { \cal W } \big ( G _ { 1 } \circ ( G _ { 2 } \circ G _ { 3 } ) \big ) .
$$

Then

$$
( G _ { 1 } \circ G _ { 2 } ) \circ G _ { 3 } \sim _ { F } G _ { 1 } \circ ( G _ { 2 } \circ G _ { 3 } ) .
$$

Their formation provenance and representation identities may nevertheless difer.

Proof. By Definition 1, recursively expanding either well-formed parenthesization evaluates the same parent denotations along the same flattened typed wiring. Their root denotations are therefore equal on the shared compatible contract, which is precisely behavioral equivalence under $\sim _ { F }$ . The recorded formation DAGs need not coincide. □

This associativity statement is semantic and conditional on wiring equivalence. It is independent of the finite-closure result below: recursive compatible formation yields closure by induction, while semantic associativity identifies compatible regroupings that realize the same flattened operation.

## 2.4 Finite Compositional Closure

Let $\mathfrak { F } _ { 0 } \subseteq \mathfrak { F }$ be the primitives. Define

$$
\begin{array} { r } { \mathfrak { F } _ { n + 1 } = \mathfrak { F } _ { n } \cup \left\{ \mathrm { C o m p o s e } _ { \omega } ( G _ { i } , G _ { j } ) \left| \begin{array} { c } { G _ { i } , G _ { j } \in \mathfrak { F } _ { n } , } \\ { \mathrm { C o m p a t i b l e } ( G _ { i } , G _ { j } , \omega ) } . } \end{array} \right. } \end{\right\array} \end{array}\tag{3}
$$

and $\begin{array} { r } { \mathfrak { F } ^ { * } = \bigcup _ { n \geq 0 } \mathfrak { F } _ { n } } \end{array}$

Proposition 2 (Finite compositional closure). Every composite F obtained from primitives by finite compatible formation belongs to ${ \mathfrak { F } } ^ { * } \subseteq { \mathfrak { F } }$ and may serve as a parent in another compatible formation.

Proof. Primitive leaves lie in $\mathfrak { F } _ { 0 }$ . If two parents have finite depth, they lie in a common $\mathfrak { F } _ { n } ;$ ; Axiom 1 places their compatible one-step composition in $\mathfrak { F } _ { n + 1 }$ . Induction over a finite formation DAG proves the root claim, after which the same recursive formation rule permits another compatible composition. □

Proposition 1 supplies a separate semantic fact: when two compatible regroupings have the same flattened typed wiring, they lie in the same behavioral equivalence class even though their formation DAGs may difer.

## 2.5 First-Class Consequences

Corollary 1 (Preserved instance exclusion). For every finite composite $F , X \notin F$ in its representation.

The primitive base satisfies $\mathrm { A } _ { 1 }$ , and each constructor stores only static formation information.

Corollary 2 (Preserved external parameterization). A finite composite receives invocation-specific α externally: $\mathbb { [ } F ] ( X ; \alpha ) = Y$ for a contract-valid call.

Formation retains parameter contracts without capturing a particular call’s value.

Corollary 3 (Direct first-class invocation). Each $F \in { \mathfrak { F } } ^ { * }$ has the public form of $\mathrm { A _ { 3 } }$ and can accept a contract-valid new instance directly:

$$
( X , \alpha ) \longrightarrow F \longrightarrow Y .\tag{4}
$$

Formation history is retained, but it is not required for public invocation. The executable body may still evaluate child operations internally.

Corollary 4 (Recursive reusability). If a formed composite $F \in { \mathfrak { F } } ^ { * }$ and another capability $G \in { \mathfrak { F } }$ are contract-compatible under a compatible wiring rule ω, then

$$
\mathrm { C o m p o s e } _ { \omega } ( F , G ) \in \mathfrak { F } .
$$

Either parent may itself be composite; formation creates no lower-status composite class.

Figure 1 makes the distinction between formation history and later operational reuse explicit. The left panel shows one compatible recursive history in which $G _ { 5 }$ is itself a formed capability and formation later continues to $G _ { 6 }$ . The right panel reuses that same $G _ { 5 }$ as a single public capability. The orange styling only highlights the capability being followed across the two panels; it does not denote a diferent operational type. The historical subgraph and its provenance remain recorded, but they need not re-enter the public search view as formation decisions during later reuse.

![](images/b195762722f7a7347249b9580c7f0eeea821a358c7f63b22af1eb65e3c311652.jpg)  
Figure 1: Recursive formation and first-class reuse of an intermediate capability. All boxes denote generic capabilities $G \in { \mathfrak { F } } ;$ the orange styling only highlights the same $G _ { 5 }$ across the two panels and does not denote a distinct operational type. In (a), compatible recursive composition forms $G _ { 5 }$ and later continues to $G _ { 6 }$ . In (b), the same $G _ { 5 }$ is exposed directly in a later task: it can accept a new invocation $( X , \alpha )$ or participate in further compatible composition with $G _ { 7 }$ to form $G _ { 8 }$ , without re-forming its historical subgraph. The recorded formation DAG and provenance remain retained.

Corollary 5 (Historical re-formation depth under first-class reuse). If F is available to search through its public interface, let $h _ { \mathrm { f o r m } } ( F )$ denote its historical formation depth, $h _ { \mathrm { p u b l i c } } ( F )$ its public search representation depth, and $h _ { \mathrm { r e f o r m } } ( F )$ the new depth required solely to reconstruct its historical chain. For any finite $h _ { \mathrm { f o r m } } ( F )$ ),

$$
h _ { \mathrm { p u b l i c } } ( F ) = 1 , \qquad h _ { \mathrm { r e f o r m } } ( F ) = 0 .\tag{5}
$$

Search sees one public operation and need not form its ancestors again. The corollary concerns the public search representation of the capability; execution continues to follow the stored executable body, and selection remains part of the surrounding search procedure.

## 2.6 Reuse, Identity, and Quotient Spaces

The provenance rule is $\mathrm { R } _ { 1 } : \ : P ( \mathrm { R e u s e } ( F ) ) = P ( F )$ . Repeated application gives, for finite $n \in  { \mathbb { N } } _ { 0 }$

$$
P ( { \mathrm { R e u s e } } ^ { n } ( F ) ) = P ( F ) .\tag{6}
$$

Reuse preserves formation identity: new outputs or execution records leave the recorded formation provenance unchanged.

Within a fixed observable contract, for $G _ { a } , G _ { b } \in \mathfrak { F }$ , behavioral equivalence is defined by

$$
G _ { a } \sim _ { F } G _ { b } \quad \Longleftrightarrow \quad \mathbb { \mathbb { \left[ \right]} \boldsymbol { G } _ { a }  } ( X ; \alpha ) = \mathbb { \left[ \boldsymbol { G } _ { b } \right] } ( X ; \alpha ) \quad \mathrm { f o r ~ a l l ~ c o n t r a c t - v a l i d ~ } ( X , \alpha ) .\tag{7}
$$

The identity rule $\mathrm { R _ { 2 } }$ permits, but does not require, opid $( G _ { a } ) = \mathrm { o p i d } ( G _ { b } )$ when $G _ { a } \sim _ { F } G _ { b }$ . Distinct provenance $P ( G _ { a } ) \neq P ( G _ { b } )$ may remain. The regroupings of Proposition 1 illustrate equivalent operations with diferent formation DAGs.

Corollary 6 (Operational and formation identities are separable). A shared operation identity for behaviorally equivalent capabilities does not require equal formation provenance.

Quotienting the well-formed capability space by semantic equivalence gives

$$
{ \mathcal C } = \mathfrak { F } / { \sim } _ { F } .\tag{8}
$$

A class $[ G ] _ { \sim F }$ may contain diferent representation identities, formation DAGs, and provenance while representing the same observable operation. The finitely generated compositional region defined above induces the subset

$$
\begin{array} { r } { \begin{array} { l } { \displaystyle \mathcal { C } ^ { * } : = \left\{ \left[ G \right] _ { \sim _ { F } } \in \mathcal { C } \mid G \in \mathfrak { F } ^ { * } \right\} \subseteq \mathcal { C } . } \end{array} } \end{array}
$$

If behavioral equivalence is a congruence for contract-compatible composition, $G _ { a } \sim _ { F } G _ { a } ^ { \prime }$ and $G _ { b } \sim _ { F } G _ { b } ^ { \prime }$ imply $G _ { a } \circ G _ { b } \sim _ { F } G _ { a } ^ { \prime } \circ G _ { b } ^ { \prime }$ . Composition then descends to compatible equivalence classes $[ G _ { a } ] \circ [ G _ { b } ]$ ; it remains partial because some contracts cannot be composed.

Let G be tasks with requested observable behavior. Under a compatible observable contract, $g _ { 1 } \sim g _ { 2 }$ means $[ [ g _ { 1 } ] ] = [ [ g _ { 2 } ] ]$ , yielding

$$
{ \mathcal { T } } = { \mathcal { G } } / { \sim } _ { G } ~ .\tag{9}
$$

The capability quotient C and task quotient T play distinct semantic roles, and no canonical identification between them is assumed: one capability class may serve several task classes, while one task class may admit multiple operational realizations. Semantic equivalence is an ideal definition; finite experiments approximate it with domain-specific evidence. The next section describes how these capability objects and formation relations are represented and managed in the implementation.

## 3 The FormCraft Framework

## 3.1 Executable Capability Records

A well-formed capability $G \in { \mathfrak { F } }$ is represented in the implementation by an executable capability record. In the retained implementation, this record is realized by the class Structure. An executable capability record contains a record identity, public operation identity, typed input and output ports, an external-parameter schema, and an executable body. A composite record additionally stores parent identities, a static wiring rule, formation-level metadata, and formation provenance. These fields realize Equation (1). The persistent store may also contain auxiliary non-executable records, such as topology templates; persistent-record counts and executable-capability counts are therefore tracked separately. Storage location is independent of a capability’s well-formedness and public call form.

The record identity distinguishes serialized objects, opid(F) denotes operational identity, and $P ( F )$ records a particular formation history. Equivalent operations may retain diferent parent graphs. Serialization and persistent hashes preserve those distinctions when a capability record is retained for later use.

Table 1 summarizes the implementation fields that connect the formal objects to the executable record. Primitive and composite capabilities share the same public record shape; composite-specific fields add formation relations without changing the operational interface.

## 3.2 Capability Formation and Composition

Primitive acquisition, explicit authorship, and automatic formation can produce executable capability records with diferent provenance. Composition consumes public interfaces, checks contracts and

<table><tr><td>Capability-record field</td><td>Role in the lifecycle</td></tr><tr><td>Record identity</td><td>Distinguishes concrete records and serialized objects.</td></tr><tr><td>opid(G)</td><td>Represents operational identity independently of a particular formation history.</td></tr><tr><td>Typed input/output ports</td><td>Declare the public contract used for invocation and compatibility checks.</td></tr><tr><td>External-parameter schema</td><td>Declares the form of invocation-specific α while keeping call-specific values external.</td></tr><tr><td>Executable body</td><td>Implements the observable operation [G].</td></tr><tr><td></td><td>Parent references and wiring ω Record the static formation relation for a composite capability record.</td></tr><tr><td>Formation level</td><td>Records formation-depth metadata (stored internally as canonical_level).</td></tr><tr><td>Formation provenance  $P ( G )$ </td><td>Retains the formation history used for audit and later inspection.</td></tr></table>

Table 1: Core fields of an executable capability record. Composite-specific formation fields extend the record while preserving the same public call form.

ports, and records exact parent references and the rule ω. Recursive formation implements Axiom 1; finite closure follows from Proposition 2. The resulting F can be invoked and composed again before any long-term persistence decision.

The implementation therefore applies three operations to the same capability object: formation produces a well-formed $F \in { \mathfrak { F } }$ ; admission may retain that capability in the persistent library; later reuse exposes the same public interface for invocation or further composition. In compact form,

$$
\begin{array} { r l r l r l } { \mathrm { F o r m a t i o n } ; } & { \quad ( G _ { i } , G _ { j } , \omega ) } & & { \underset { \mathrm { V } \in \mathbb { N } } { \longmapsto } } & & { F \in \mathfrak { F } , } \\ { \mathrm { A d m i s s i o n } ; } & { \quad \quad F \in \mathfrak { F } } & & { \xrightarrow { \mathrm { v e r i f i c a t i o n } } } & & { F \in \mathcal { L } , } \\ { \mathrm { R e u s e } ; } & { \quad ( X , \alpha , F ) } & & { \longmapsto } & & { Y \quad \mathrm { o r } \quad ( F , G ^ { \prime } ) \longmapsto F ^ { \prime } . } \end{array}
$$

Formation, optional persistence, direct invocation, and further composition therefore refer to one operational capability whose provenance and parent references remain attached throughout the lifecycle.

In the sequence-domain implementation, public operations use the contract sequence<int> → int, and typed binary composition combines existing statistics. This concrete contract instantiates the general first-class interface defined by the formal model.

## 3.3 Verification and Library Admission

Long-term persistence is a separate lifecycle decision. We reserve $\mathcal { L } \subseteq \mathfrak { F }$ for admitted executable capabilities; auxiliary non-executable records may coexist in the broader persistent store but are not members of ${ \mathcal { L } } .$ A well-formed first-class capability can be checked against its declared contract and admitted into $\mathcal { L } \mathrm { : }$

$$
F \in { \mathfrak { F } } \xrightarrow { \mathrm { v e r i f i c a t i o n } } { F \in { \mathcal { L } } } , \qquad { \mathcal { L } } \subseteq { \mathfrak { F } } .\tag{10}
$$

The formation rules determine membership in $\mathfrak { F }$ and first-class operational status; verification governs the later persistence decision. In the retained implementation, candidates may be staged in an external KV store before this gate. Admission adds the capability to ${ \mathcal { L } } ,$ while library membership remains distinct from well-formedness in $\mathfrak { F }$

Depending on the domain, checks may include public interface and contract validation, executable behavior, examples, randomized oracle cases, cross-language agreement, cold-instance execution, instance-leakage and exact-parent audits, and persistent reload. Receipts bind promotion to checks and a source digest. Lean provides the independent proof checker used by the propositional microdomain [14]; the other domains instantiate verification through their executable contracts and oracles.

## 3.4 First-Class Capability Reuse

A capability has its first-class call form before persistence. Library membership makes the corresponding admitted capability record available for long-term retrieval through that same interface. Historical expansion reconstructs ancestor operations as search decisions. First-class exposure selects F as one public operation and binds its slots directly, as in Equation (4).

Parent references and provenance remain available for audit, and execution may evaluate constituent operations internally. The search-level distinction is whether those historical formation decisions must be reconstructed before the stored capability can be selected. When F is exposed through its public interface, Corollary 5 applies; Section 4 measures that efect in a paired increasingdepth comparison.

## 4 Experimental Evaluation

The evidence follows the capability lifecycle. Single-operation prototypes examine instanceindependent reuse across continuous and discrete domains; the Lean microdomain examines independently checkable composition; the main sequence-domain study then evaluates persistent acquisition, increasing formation depth, and the reuse-time efect of exposing persisted composites directly. Each experiment family targets a diferent transition in the lifecycle, and quantitative counts are reported within their corresponding snapshots.

Figure 2 summarizes the experimental lifecycle and separates three questions: how a well-formed first-class capability is formed, whether it is admitted for persistent retention, and how a persisted capability is later exposed for reuse. The admission stage therefore governs membership in ${ \mathcal { L } } ,$ while well-formedness in $\mathfrak { F }$ and first-class status are determined earlier by the representation and formation rules.

![](images/f9c8eec5ac26b419b7cdc0251f6732259fde2ba0228ae9357fc853f7f34be11a.jpg)  
Figure 2: Formation–admission–reuse lifecycle used to organize the experimental evidence. Search or acquisition first produces a well-formed first-class capability $G \in { \mathfrak { F } }$ . Verification and admission determine persistent membership $G \in { \mathcal { L } } .$ , while membership in $\mathfrak { F }$ remains a property of the capability representation. An admitted capability can later be retrieved through the same public interface for direct invocation or further composition. The applicable verification procedure is domain-dependent.

The evaluation covers four complementary settings. Continuous and character-substitution experiments examine instance-independent reuse across numerical and discrete domains; the Lean propositional microdomain examines independent checking after composition; the sequence-domain acquisition stages evaluate persistence and recursive library growth; and the paired D01–D12 comparison measures the reuse-time efect of exposing formed composites directly through their public interfaces.

## 4.1 Experimental Setup

The experimental families are early continuous-domain reuse, a discrete character-substitution prototype, a propositional composition microdomain checked by Lean, and two consecutive sequencedomain acquisition stages followed by a paired increasing-depth reuse comparison. Each family targets a distinct part of the formation–admission–reuse lifecycle. The continuous and charactersubstitution studies establish the instance/operation separation across two representation regimes; the Lean microdomain tests independent checking after composition; and the sequence-domain snapshots provide the main quantitative lifecycle and search-work measurements.

Stage I starts with 12 persistent records and seven example-only goals. It uses 336 fresh training instances and 192 root holdouts per goal, then checks $7 / 7$ roots after cold reload. The resulting 26-record persistent store contains 24 executable capability records and two non-executable topologytemplate records. Stage II starts from those 26 persistent records with 12 fixed D01–D12 goals, no training examples, and 128 seeded holdout inputs per goal. All 12/12 goals are admitted, leaving 38 persistent records, of which 36 are executable capability records, with maximum formation level ten. Because the two acquisition stages use diferent protocols, their holdout counts are reported separately.

The paired comparison uses the same 12 D goals and starting snapshot in both views. Task definitions, applicable examples and holdouts, oracle, seed, verifier, and candidate limit are held fixed; only the historical search representation changes between Expanded and First-class. Candidate attempts count attempted search candidates; composition expansions count reconstruction of composed search nodes; intermediate materializations count intermediate objects created during that reconstruction; and verification batches count grouped checks. D-level derivation depth counts newly formed historical chain levels in the search route, not runtime call depth. The same definitions and controlled snapshot are used in both views.

Execution environment and reproducibility. The archived sequence-domain artifact contains the goal definitions, fixed seeds and digests, starting-library snapshot, per-view traces, and acquisition and admission summaries. Its package specification requires Python 3.10 or newer and NumPy 1.24 or newer. The sequence-domain implementation also checks results against a JavaScript executor. The reported quantitative results are search-work counts recorded in these archived experiment snapshots.

Code availability. The public implementation corresponding to the sequence-domain acquisition and paired increasing-depth reuse experiments is available at https://github.com/ lancelot138505-cpu/f-system. The released implementation is contained in 04\_depth\_control\_ V9-4\_complete\_baseline/ and corresponds to the V9.4 implementation used for the main quantitative study.

## 4.2 Instance-Independent Reuse Across Domains

Early continuous-domain prototypes tested whether a single instance-independent operation could be reused across unseen inputs under externally supplied parameters. The discrete character-substitution prototype checked the same separation outside continuous numerical domains. Together, these prototypes provide qualitative evidence that the same instance-separation principle can be expressed in both continuous and discrete settings.

## 4.3 Independent Checking of Composite Capabilities

The propositional microdomain asks whether a composite remains independently checkable after formation. Five composite capabilities were formed from primitive proof operations. Each was reused on two theorem instances, including formulas not used to define its operation. Lean accepted all ten generated theorem checks [14]. Table 2 summarizes the result.

<table><tr><td>Measure</td><td>Result</td></tr><tr><td>Composite capabilities formed</td><td>5/5</td></tr><tr><td>Composite theorem reuses</td><td>10/10</td></tr><tr><td>Lean checks accepted (exit code 0)</td><td>10/10</td></tr><tr><td>Capability-record hash and formation provenance preserved</td><td>Yes</td></tr></table>

Table 2: Composite capability reuse in the Lean-checked propositional microdomain.

The 5/5 formation and 10/10 theorem reuse results establish independent checkability of formed composites in the propositional microdomain. The next stage asks whether well-formed capabilities can also pass the persistence gate, survive reload, and remain available as parents for later formation.

## 4.4 Sequential Acquisition and Persistent-Store Growth

Stage I admits a step capability and a root capability for each of seven goals, adding 14 persistent executable capabilities to the 12-record seed store. Parent audits and semantic checks pass, and 7/7 roots pass cold reload. Stage II starts from the resulting 26 persistent records and admits 12 new

D-level executable capabilities under the fixed D01–D12 task family. Its 12/12 admission and oracle checks pass; the maximum formation level rises from four to ten.

Table 3 summarizes the sequential growth from 12 to 26 and then 38 persistent records. The first stage tests example-driven acquisition and cold persistence; the second tests whether persisted executable capabilities can continue to support compatible composition while the broader record store grows. The diferent training protocols remain separate in the setup and interpretation.

<table><tr><td>Stage</td><td>Start</td><td>Goals</td><td>Added F</td><td>End</td></tr><tr><td>Stage I: Example-driven acquisition</td><td>12</td><td>7</td><td>14</td><td>26</td></tr><tr><td>Stage II: Increasing-depth chain (D01-D12)</td><td>26</td><td>12</td><td>12</td><td>38</td></tr></table>

Table 3: Sequential persistent-store growth in the sequence-domain study. Start and End count persistent records; Added F counts new executable capabilities.

## 4.5 First-Class Reuse under Increasing Formation Depth

The paired depth experiment compares the two search views over the same 12 D goals, instantiating Corollary 5. Both views use the same frozen persistent snapshot and executable capability pool. Expanded masks the target chain and re-forms its earlier D-level capabilities; First-class exposes the already admitted target through its public interface. Trace audits confirm that first-class selection binds public slots without inspecting the target’s parent graph before selection.

Table 4 reports reuse-time search-work counts. Candidate attempts fall from 16,570 to 144, a 99.1% reduction; binding-combination counts equal candidate attempts in both views. Repeated historical composition expansions fall from 37,300 to zero, intermediate materializations from 16,778 to zero, and verification batches from 17,116 to 192. The route-level depth of newly formed D-level derivations changes from 78 to zero. The 12 main-chain acquisitions pass admission, while the outputs of all paired search views match the frozen oracle.

<table><tr><td>Search-work measure</td><td>Expanded</td><td>First-class</td><td>Reduction</td></tr><tr><td>Candidate attempts</td><td>16,570</td><td>144</td><td>99.1%</td></tr><tr><td>Composition expansions</td><td>37,300</td><td>0</td><td>100%</td></tr><tr><td>Intermediate materializations</td><td>16,778</td><td>0</td><td>100%</td></tr><tr><td>Verification batches</td><td>17,116</td><td>192</td><td>98.9%</td></tr></table>

Table 4: Paired reuse-time search-work counts over 12 goals. Outputs in both search views matched the frozen oracle on evaluated cases; the 12 main-chain acquisitions passed admission.

The paired counts isolate historical search reconstruction. The admitted F retains its formation provenance and may evaluate constituent operations internally; Equation (5) concerns the amount of historical formation that re-enters later search. In the same sequence-domain snapshots, the FULL and ACTIVE lifecycle views expose the same aggregate 366 executable records across the 12 goals and both yield 144 first-class candidate attempts. The observed reduction therefore follows from exposing formed composites directly rather than from a change in the available capability breadth.

## 5 Related Work

Programming-language abstraction. Programming languages have long supported abstraction boundaries that separate public operations from concrete representation and have treated procedures or functions as values that can be manipulated through ordinary program structure [1, 2]. FormCraft builds on this abstraction tradition at the level of acquired capability records: recursive formation produces another record with the same public operational status, while provenance and persistence decisions remain explicit parts of the lifecycle.

Program synthesis. Program synthesis searches for programs satisfying examples or other specifications; representative methods include programming by example, enumerative and constraintbased search, and learned guidance [3, 4, 5]. Oracle-guided component synthesis couples candidate construction to semantic checks [6], while FlashMeta provides a framework for building inductive synthesizers over domain-specific languages [7]. DeepCoder uses learned predictions to guide search, and RobustFill studies neural synthesis from input-output examples [8, 9]. These works primarily address how programs are generated or selected. FormCraft focuses on the operational status of a formed capability: how it is represented, composed, retained, and exposed again through a stable public interface.

Library and abstraction learning. DreamCoder alternates program search, library expansion, and recognition-guided synthesis [10]. Stitch synthesizes abstractions from recurring structure in a corpus of programs [11]. LILO combines language-model-guided synthesis, symbolic compression, and documentation to learn interpretable libraries [12]. Recent cumulative inductive-programming work provides a particularly close example: the Minimalist Machine treats learned predicates as a growing concept library, makes them available as background knowledge for later tasks, and uses that accumulated structure to discover recursive arithmetic programs [13]. These systems provide strong examples of learned components becoming useful in later synthesis. FormCraft approaches the same broad area from the representation side: recursive closure is defined over one uniform capability type, formation provenance remains attached to the resulting object, and persistent library admission is handled through a separate verifier gate. Abstractions produced by other methods can enter the same lifecycle whenever they satisfy the declared operational contracts and representation rules.

Independent checking. Formal proof assistants provide a distinct way to validate claims within a logical domain. Lean 4 is a theorem prover with a small trusted kernel and an extensible programming environment [14]. In the propositional microdomain, generated composite theorem reuses are submitted to Lean as an independent check. Across the framework, admission is parameterized by the declared contract and the checker available for that domain, allowing proof checking, executable oracles, cross-language agreement, and reload checks to occupy the same lifecycle role.

## 6 Discussion and Conclusion

The central design choice in FormCraft is to make the result of compatible composition an operational object of the same public type as its parents. Once a composite has been formed, later search can encounter that capability through its public interface instead of reconstructing the sequence of formation decisions that originally produced it. The formation DAG remains recorded for audit and execution, but the public search view can treat the composite as one reusable operation. This is the operational content of Corollary 5.

The distinction between formation depth and execution depth is important for interpreting this result. A stored composite can still evaluate constituent operations internally, and its executable body may therefore contain a nontrivial call structure. First-class exposure changes the representation presented to later search: the historical chain does not have to be generated again as a sequence of candidate-formation decisions before the composite can be selected. The paired increasing-depth experiment measures this search-level efect directly through candidate attempts, composition expansions, intermediate materializations, verification batches, and newly formed D-level derivation depth.

Persistence serves a diferent role. Membership in L is a memory policy over already wellformed capabilities: it determines which formed operations remain available for later retrieval. This separation allows verification to govern long-term retention without redefining the capability’s public operational status. The same distinction also keeps provenance meaningful, because reuse can preserve both operational identity and the original formation record rather than replacing history with a newly synthesized copy.

The experiment sequence reflects these layers. The continuous and character-substitution prototypes show the instance/operation separation in two data regimes. The Lean microdomain shows that a formed composite can remain independently checkable. The two sequence-domain acquisition stages then demonstrate persistent growth and recursive reuse, and the paired depth experiment isolates the efect of exposing already formed composites directly. Together, the experiments connect the formal lifecycle to executable records and to measured search behavior.

The present quantitative evaluation concentrates on the fixed D01–D12 task family and controlled persistent snapshots. Within that setting, the FULL and ACTIVE lifecycle views expose the same aggregate 366 executable records across the 12 goals and both yield 144 first-class candidate attempts, so the measured reduction is associated with historical-depth handling rather than a change in available breadth. Larger capability libraries, explicit breadth control, equivalence consolidation, and learned proposal or retrieval mechanisms are natural next variables for extending the same lifecycle. Stronger formal contracts and additional independently checked domains provide a parallel route for extending the verification side of the system.

In the controlled sequence-domain study, the persistent store grows from 12 to 26 and then 38 records. Exposing persisted composites through their first-class interfaces reduces reuse-time candidate attempts from 16,570 to 144 while eliminating repeated historical composition expansion in the paired comparison. Combined with the cross-domain checks and the formal closure result, these findings support a simple architectural conclusion: recursive formation can produce a first-class operational object whose provenance remains available while its historical formation chain need not be reconstructed as part of later public search; verification then determines whether that object is retained for persistent reuse.

## A Experimental Record and Reproducibility Details

## A.1 Search-Work Measures

The paired depth experiment records work performed by the search and verification pipeline. The measures used throughout Section 4 are defined as follows.

Candidate attempts. Candidate search objects presented for consideration under the fixed search view.

Binding combinations. Slot-binding combinations considered by the search; these equal candidate attempts in the reported paired runs.

Composition expansions. Reconstruction events that expand historical composed nodes into the search process.

Intermediate materializations. Intermediate objects instantiated while reconstructing expanded composition routes.

Verification batches. Grouped verifier checks issued during the search procedure.

D-level derivation depth. Newly formed historical D-level chain steps in the search route, distinct from runtime call depth.

## A.2 Persistent-Store Accounting

The sequence-domain experiments distinguish persistent records from executable capabilities. Stage I begins with 12 persistent records and adds 14 executable capabilities, yielding 26 persistent records in total: 24 executable capability records and two non-executable topology-template records. Stage II begins from those 26 records, adds 12 executable D-level capabilities, and ends with 38 persistent records, of which 36 are executable. The maximum formation level at the end of Stage II is ten (stored internally in the canonical\_level field).

<table><tr><td>Snapshot</td><td>Persistent records</td><td>Executable capabilities Auxiliary records</td><td></td></tr><tr><td>Initial seed</td><td>12</td><td></td><td></td></tr><tr><td>After Stage I</td><td>26</td><td>24</td><td>2</td></tr><tr><td>After Stage II</td><td>38</td><td>36</td><td>2</td></tr></table>

Table 5: Persistent-record accounting for the sequence-domain acquisition stages.

## A.3 Reproducibility Record

The public sequence-domain artifact, internally identified as V9.4, contains fixed goal definitions, seeds and digests, the starting-library snapshot, per-view traces, and acquisition and admission summaries. It is available at https://github.com/lancelot138505-cpu/f-system. Its package specification requires Python 3.10 or newer and NumPy 1.24 or newer. Sequence-domain results are also checked against a JavaScript executor. The paired comparison reuses the same goals, applicable examples and holdouts, oracle, seed, verifier, starting snapshot, and candidate limit in both search views; only the historical search representation changes between Expanded and First-class.

## References

[1] Christopher Strachey. Fundamental Concepts in Programming Languages. Higher-Order and Symbolic Computation, 13(1–2):11–49, 2000. DOI: https://doi.org/10.1023/A:1010000313106.

[2] Barbara H. Liskov and Stephen N. Zilles. Programming with Abstract Data Types. In Proceedings of the ACM SIGPLAN Symposium on Very High Level Languages, pp. 50–59, 1974. DOI: https://doi.org/10.1145/800233.807045.

[3] Sumit Gulwani, Oleksandr Polozov, and Rishabh Singh. Program Synthesis. Foundations and Trends in Programming Languages, 4(1–2):1–119, 2017. DOI: https://doi.org/10.1561/2500000010.

[4] Sumit Gulwani. Automating String Processing in Spreadsheets Using Input-Output Examples. In Proceedings of the ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages (POPL), 2011. https://www.microsoft.com/en-us/research/publication automating-string-processing-spreadsheets-using-input-output-examples/.

[5] Rajeev Alur, Rastislav Bodik, Garvit Juniwal, Milo M. K. Martin, Mukund Raghothaman, Sanjit A. Seshia, Rishabh Singh, Armando Solar-Lezama, Emina Torlak, and Abhishek Udupa.

Syntax-Guided Synthesis. In Proceedings of the IEEE International Conference on Formal Methods in Computer-Aided Design (FMCAD), pp. 1–17, 2013. https://people.eecs.berkeley.edu/\~sseshia/pubs/b2hd-alur-fmcad13.html.

[6] Susmit Jha, Sumit Gulwani, Sanjit A. Seshia, and Ashish Tiwari. Oracle-Guided Component-Based Program Synthesis. In Proceedings of the 32nd ACM/IEEE International Conference on Software Engineering (ICSE), pp. 215–224, 2010. DOI: https://doi.org/10.1145/1806799.1806833.

[7] Oleksandr Polozov and Sumit Gulwani. FlashMeta: A Framework for Inductive Program Synthesis. In Proceedings of the ACM SIGPLAN International Conference on Object-Oriented Programming, Systems, Languages, and Applications (OOPSLA), pp. 107–126, 2015. https://2015.splashcon.org/details/oopsla2015/11/ FlashMeta-A-Framework-for-Inductive-Program-Synthesis.

[8] Matej Balog, Alexander L. Gaunt, Marc Brockschmidt, Sebastian Nowozin, and Daniel Tarlow. DeepCoder: Learning to Write Programs. In International Conference on Learning Representations (ICLR), 2017. https://arxiv.org/abs/1611.01989.

[9] Jacob Devlin, Jonathan Uesato, Surya Bhupatiraju, Rishabh Singh, Abdel-rahman Mohamed, and Pushmeet Kohli. RobustFill: Neural Program Learning under Noisy I/O. In Proceedings of the 34th International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research 70:990–998, 2017. https://proceedings.mlr.press/v70/devlin17a.html.

[10] Kevin Ellis, Catherine Wong, Maxwell Nye, Mathias Sablé-Meyer, Lucas Morales, Luke Hewitt, Luc Cary, Armando Solar-Lezama, and Joshua B. Tenenbaum. DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning. In Proceedings of the ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI), pp. 835–850, 2021. DOI: https://doi.org/10.1145/3453483.3454080.

[11] Maddy Bowers, Theo X. Olausson, Lionel Wong, Gabriel Grand, Joshua B. Tenenbaum, Kevin Ellis, and Armando Solar-Lezama. Top-Down Synthesis for Library Learning. Proceedings of the ACM on Programming Languages, 7(POPL), Article 41, 2023. DOI: https://doi.org/10.1145/3571234.

[12] Gabriel Grand, Lionel Wong, Maddy Bowers, Theo X. Olausson, Muxin Liu, Joshua B. Tenenbaum, and Jacob Andreas. LILO: Learning Interpretable Libraries by Compressing and Documenting Code. In International Conference on Learning Representations (ICLR), 2024. https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ 819cebb05f993840e8a52d7564c5c282-Abstract-Conference.html.

[13] Dominik Magiera, Christiane Wiebel-Herboth, and Frank Jäkel. Teaching a Minimalist Machine to Discover Recursive Programs for Arithmetic. arXiv preprint arXiv:2610.06304, 2026. https://arxiv.org/abs/2610.06304.

[14] Leonardo de Moura and Sebastian Ullrich. The Lean 4 Theorem Prover and Programming Language. In Automated Deduction–CADE 28, Lecture Notes in Computer Science 12699, 2021. DOI: https://doi.org/10.1007/978-3-030-79876-5\_37.