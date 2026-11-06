# SECURE FORMAL RAISING: PROTECTING THE DEVELOPMENTOF AI THROUGH EXECUTABLE PROPERTIES

Rolf Drechsler   
University of Bremen and DFKI Bremen, Germany   
drechsler@uni-bremen.de

Christina Plump University of Bremen and DFKI Bremen, Germany cplump@uni-bremen.de

## ABSTRACT

Model raising revisits Turing’s developmental view of machine intelligence. Formal Raising (FR) has recently been proposed to complement this view with explicit behavioral properties. This paper extends that proposal to the security of the raising environment. Secure Formal Raising (SFR) models policy integrity, authorization, and confidential data flow using four-valued evidence that preserves unknown and inconsistent assessments. We prove six security invariants and show that they force admission to accept only confirmed evidence. A plain-C reference implementation satisfies the invariants in a bounded exhaustive exploration, which also exposes a confused-deputy gap for tool calls that a provenance requirement closes.

Keywords Formal Raising · AI development · security properties · multiple-valued logic · executable specification Belnap logic · bilattice

## 1 Introduction

How should an intelligent machine acquire its behavior? In 1950, Turing proposed educating a comparatively simple “child machine” instead of directly programming an adult-like intelligence [1]. He distinguished its initial construction from the education through which it develops. This developmental perspective makes interaction with teachers part of the system’s design, rather than merely a source of test inputs.

Aydin et al. recently revived this perspective for contemporary AI through model raising [2]. They propose shaping values with capabilities early on, using first-person experience, social interaction, and a scaffolded curriculum. Their proposal motivates bringing behavioral commitments into development instead of relying on later alignment. It does not itself provide an executable specification of those commitments.

Formal Raising (FR) is a newly proposed complement to this developmental view [3]. It represents selected requirements as explicit, machine-checkable properties that participate in evaluation and action selection. Hard rules exclude inadmissible alternatives, while declared priorities rank competing preferences. The preceding FR work primarily develops ethical examples; it is a starting point for further investigation rather than an established account of AI development.

This paper builds on that proposal by addressing the integrity of the raising environment. A teaching message may change a curriculum, a retrieved document may influence an action, and a tool call may disclose information. Who may change the policy, and which operations may a developing system execute? Ethical evaluation alone does not establish teacher authenticity or authorization. The proposed Secure Formal Raising (SFR) model therefore enforces protection against unauthorized modification, execution, and disclosure at explicit transition boundaries.

The environment is modeled as a state machine, making the approach accessible through familiar verification concepts. Four-valued evidence preserves confirmed, refuted, unknown, and inconsistent assessments of identity and provenance. Only unambiguous confirmation satisfies a hard requirement, while unresolved evidence supports differentiated feedback. The contribution combines this evidence layer with policy integrity, limited authorization, preference ranking, and a self-contained plain-C reference implementation. Building on established access-control mechanisms, we prove six security invariants for this model, show that they require admission on confirmed evidence only, and check the implementation against them by a bounded exhaustive exploration of 6,262 reachable states and 72,329,329 transitions. The exploration finds no violation, but 21,708 admitted tool calls and exports with unconfirmed source evidence reveal a confused-deputy gap, which a provenance requirement closes.

Section 2 relates SFR to prior work, while Sec. 3 defines the formal model to ensure security guarantees. Section 4 presents the implementation and checks it against the guarantees. Section 5 discusses limitations, and Sec. 6 concludes.

## 2 Background and Related Work

SFR builds on four lines of work: FR, classical access control and information flow, four-valued logic in access control, and runtime enforcement for agents and learners.

FR evaluates candidate actions against declared behavioral properties; priorities resolve competing preferences, and hard properties are non-defeasible [3]. It continues work on formally verifying the choices of autonomous systems [4]. SFR applies the same separation to the environment: teachers may change preferences, but no learning action may change the mechanism that admits it.

The security mechanisms of SFR are standard. Admission follows the principles of fail-safe defaults and complete mediation [5]. Roles restrict operations as in role-based access control [6]. The flow check is the two-point instance of the lattice model of information flow [7]; it constrains explicit transfers only and does not establish noninterference [8]. A program that acts with its own authority on behalf of a less trusted party is a confused deputy [9]. We will show in Sec. 3.4 that this situation is relevant in SFR and how to prevent it. Note that we do not consider availability and cryptographic mechanisms in this work.

Belnap’s logic represents incomplete and inconsistent information by four values [10]. Ginsberg generalized its two orders, truth and knowledge, to bilattices [11], and Arieli and Avron base consequence relations on sets of designated values [12]. Bruns and Huth use Belnap’s logic to compose access-control policies, where B marks a conflict between policies and N a gap [13]. XACML’s decisions NotApplicable and Indeterminate play similar roles [14]. SFR uses the same lattice for a different purpose: the values describe evidence about a request, namely whether its actor and source are confirmed, rather than combined policy decisions. The hard properties themselves are fixed and are not composed.

Security automata enforce safety properties by monitoring execution [15], and runtime verification checks executions against formal specifications [16]. The closest analogue for learning systems is a shield, which blocks unsafe actions of a reinforcement learner [17]. SFR shields the protected transitions of the environment, including updates to the policy that guides learning. Specifications of this kind can be expressed as assertions in property languages such as PSL [18].

Indirect prompt injection lets retrieved content control LLM-integrated applications [19]. AgentDojo provides a benchmark of such attacks on tool-using agents [20], and CaMeL separates control flow from data flow and tracks provenance with capabilities [21]. SFR does not track provenance itself; it assumes provenance evidence is supplied and specifies how evidence of different quality gates actions.

## 3 Secure Formal Raising

This section defines SFR, which protects the preference policy: the environment and its events (Sec. 3.1), the semantics of the model (Sec. 3.2), scenarios that illustrate it (Sec. 3.3), and the guarantees it provides (Sec. 3.4).

## 3.1 Environment Model

Actors have one of three roles: teacher, student, or external. Teachers propose policy updates and grant limited authorization. Students propose learning-related tool use or export. External actors supply content and propose operations but cannot acquire an internal role by stating so in the text.

Four operations keep the model small: OBSERVE is an inert local action; CHANGE POLICY proposes a replacement policy; TOOL CALL interacts with an external resource; EXPORT DATA transfers data. Tool calls and exports are the protected operations. A candidate action consists of its operation, actor, role, identity evidence, source evidence, input and destination labels, expected policy version, two preference costs, and, for updates, a replacement policy. Evaluating a candidate never changes the state.

State and events. The state is

$$
s = ( t , v , \ell , p _ { \mathrm { p r i v } } , p _ { \mathrm { p r o g } } , \pi ) ,\tag{1}
$$

with logical clock t, policy version v, permit lifetime ℓ, preference priorities $p _ { \mathrm { p r i v } }$ and $p _ { \mathrm { p r o g } }$ , and permit record

$$
\pi = ( a c t i v e , a c t o r , k i n d , v _ { \pi } , d ) .\tag{2}
$$

![](images/3865773a2edf2cc61902806167cdc045c8e4c03015b3f08a4f19406ac79d859c.jpg)  
Figure 1: Permit and policy states, with transitions labeled event [guard] / effect; I and S are the grant’s identity and source evidence. Consume and update are successful commits. Events not drawn change no component shown.

The policy $( v , \ell , p _ { \mathrm { p r i v } } , p _ { \mathrm { p r o g } } )$ starts as (1, 2, 40, 30). Priorities lie in [0, 100] and lifetimes in [1, 100]. The model holds a single permit; a new grant replaces it.

Four events change the state. Grant creates a permit $( y e s , x , o , v , t + \ell )$ for actor x and operation $o \in$ {TOOL CALL, EXPORT DATA}; it requires the teacher role and confirmed identity and source evidence. Revoke clears the permit. Tick advances t and deactivates the permit if past the deadline. Commit re-evaluates a candidate in the current state and, if it is admissible, applies its effect: an update installs the replacement policy and revokes the permit, a protected operation consumes the permit, and an observation has no external effects. A rejected commit leaves the state unchanged. The implementation also counts executions and accepted updates; these counters are read only by overflow guards and are omitted here. Selection among candidates (Sec. 3.2) is pure; only commit, grant, revoke, and tick change the state. Fig. 1 shows the resulting permit life cycle.

Assumptions and threat model. A1 (trusted adapters): roles, identity and source evidence, and data labels are produced by adapters outside the model, never copied from model-generated text, and each candidate’s role is that of its actor. A2 (evaluator integrity): the attacker cannot modify the evaluator, the hard properties, or the state directly. A3 (atomic commit): revalidation and effect of a commit are atomic.

The attacker can propose arbitrary candidates, supply external content, replay stale expected versions, and exploit missing or conflicting evidence. Under A1–A3, SFR guarantees the invariants of Sec. 3.4. If an adapter errs, the affected property fails open: a falsely confirmed identity defeats the identity check, and confidential data labeled public defeats the flow check. The other properties still apply, so a single adapter error does not by itself enable a policy update by a non-teacher.

## 3.2 Semantics

Admission (adm) depends on how evidence is represented, which properties are checked, and how admissible candidates are ranked. We define these three layers in turn and explain where the four-valued representation changes outcomes.

Four-valued evidence. An evidence value is a pair $( e ^ { + } , e ^ { - } ) \in \{ 0 , 1 \} ^ { 2 }$ of positive and negative support (Table 1). Negation swaps the components; conjunction and disjunction are

$$
\begin{array} { l l l } { { a \wedge b } } & { { = } } & { { ( a ^ { + } \wedge b ^ { + } , \ a ^ { - } \vee b ^ { - } ) , } } \end{array}\tag{3}
$$

$$
\begin{array} { l l l } { a \lor b } & { = } & { ( a ^ { + } \lor b ^ { + } , \ a ^ { - } \land b ^ { - } ) . } \end{array}\tag{4}
$$

These operations combine requirements in the truth order. Observations of the same proposition from several adapters are combined by the knowledge join

$$
a \oplus b = ( a ^ { + } \lor b ^ { + } , a ^ { - } \lor b ^ { - } ) ,\tag{5}
$$

so a confirming and a refuting observation yield B. Boolean checks enter the algebra through the embedding true 7→ T, false 7→ F.

Let $P _ { 1 } , \ldots , P _ { 6 }$ be the hard properties of Table 2. Then,

$$
\mathrm { a d m } ( s , a ) \iff P _ { i } ( s , a ) = \mathsf { T } \mathrm { ~ f o r ~ a l l ~ } i \in \{ 1 , \ldots , 6 \} .\tag{6}
$$

Designating only T rejects refuted, unknown, and inconsistent evidence alike. A gate that checks $e ^ { + }$ alone would designate {T, B} and admit conflicts; the implementation therefore compares complete values.

Role offour-valued evidence. For execution, Eq. (6) is a Boolean gate. The four values matter in three other places. Merging: by Eq. (5), a single refuting observation turns T into B and withdraws permission, so authorization is not monotone in accumulated evidence.

Table 1: Four-valued evidence and admission (left) and the bilattice FOUR with truth order $\leq _ { t }$ and knowledge order ≤<sub>k</sub> (right).
<table><tr><td>Value</td><td> $\overline { { e ^ { + } } }$ </td><td> $e ^ { - }$  1</td><td>Meaning</td><td>Admits</td></tr><tr><td>N</td><td>0</td><td>0</td><td>Unknown</td><td>No</td></tr><tr><td>T</td><td>1</td><td>0</td><td>Confirmed</td><td>Yes</td></tr><tr><td>F</td><td>0</td><td>1</td><td>Refuted</td><td>No</td></tr><tr><td>B</td><td>1</td><td>1</td><td>Inconsistent</td><td>No</td></tr></table>

![](images/6a0cc74e9ebcef0ece6d8fc7d4ef932e592900e0a37daa1ad9e02b26246b1054.jpg)

Feedback: the property vector separates refutation, which yields a counterexample, from missing evidence, which calls for further observation, and from conflicting evidence, which calls for conflict resolution. Feedback must use the vector, not the conjunction of its entries, because N ∧ B = F by Eq. (3): conjoining unknown and conflicting evidence would report a refutation.

Composition: properties combine evidence with structural checks by $\land ,$ without ad-hoc rules. A Boolean gate with reason codes could reproduce the feedback labels, but it would need its own rules for merging and composition; the lattice supplies them with known algebraic laws.

Hardproperties. Table 2 lists the six properties. An inactive property yields T. For updates, integrity is the conjunction of source evidence and a structural check of the replacement: bounds, successor version $v + 1$ , and an overflow guard. With C for confidential and P for public, the flow property is

$$
\begin{array} { r } { \mathrm { \it ~ \Omega } \mathrm { ~ } f o w ( L _ { \mathrm { i n } } , L _ { \mathrm { s i n k } } ) = \left\{ \begin{array} { l l } { \mathsf { F } , } & { L _ { \mathrm { i n } } = C \mathrm { a n d \it ~ L _ { \mathrm { s i n k } } = \it P } , } \\ { \mathsf { T } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{7}
$$

A permit is live for candidate a if it is active, its actor and kind match $a , v _ { \pi } = v ,$ and $t \leq d .$ A live permit does not override Eq. (7), and the model has no declassification.

Preferences and selection. Admissible candidates are compared by two binary costs, for privacy and for learning progress. Priorities are ordinal: the candidate with the lower total cost at the highest priority level where the candidates differ is preferred, and costs at equal priority are summed. Selection computes the admissible set and returns its unique minimum. Several minima yield UNDERDETERMINED and an empty set yields NO ADMISSIBLE; neither selects a candidate. A selection result is not a permission: commit re-evaluates Eq. (6) in the state current at that time. Fig. 2 summarizes the pipeline.

## 3.3 Illustrative Scenarios

To explain the model’s behavior, we show scenarios, using the same evaluator instead of scenario-specific rules.

An external policy injection. An external source proposes a policy replacement. Its identity may be known and the replacement may be structurally valid, but the external role is not authorized to change the policy and source trust is refuted. The candidate is blocked with role and integrity failures. A second candidate from a confirmed teacher with trusted provenance and the next valid version is admitted. Selection chooses the teacher update. Persuasive content or a low preference cost cannot repair the external candidate’s hard failures.

Conflicting identity and unknown provenance. A proposed update has the teacher role but identity evidence B. The identity requirement blocks it, and the feedback requests teacher review. In another scenario, identity is confirmed but source trust is N. Integrity remains unknown, and the update is again blocked with teacher-review feedback. These examples show why keeping the evidence values is useful even when both cases lead to the same execution decision.

Table 2: Hard properties (U: update, P: tool call or export).
<table><tr><td>Property</td><td>Active Requirement</td><td></td></tr><tr><td> $\overline { { P _ { 1 } } }$  Identity</td><td>U, P</td><td>Identity evidence is T.</td></tr><tr><td> $P _ { 2 }$  Role</td><td>U, P</td><td>Teacher for U; teacher or student for P.</td></tr><tr><td> $P _ { 3 }$  Integrity</td><td>U</td><td>Source evidence ∧ structural check is T.</td></tr><tr><td> $P _ { 4 }$  Flow</td><td>P</td><td>Eq. (7) yields T.</td></tr><tr><td> $P _ { 5 }$  Permit</td><td>P</td><td>A live permit matches.</td></tr><tr><td> $P _ { 6 }$  Freshness</td><td>all</td><td>Expected version equals v.</td></tr></table>

![](images/bdd5f3a53d6a948dbbd6656bf70ece15a2b5cf33f27deecbfda73677d8d70cdb.jpg)  
Figure 2: Evaluation pipeline. A proposal cannot change trusted state directly; only an admitted, revalidated commit does. Feedback is derived from the property vector and explicitly not from the model’s self-assessment.

A valid permit does not permit every flow. A student holds an active export permit. The proposed export carries confidential data to a public destination. Live authorization is satisfied, but the flow property is refuted. The transfer is blocked. This illustrates the conjunction of independent requirements: permission to invoke an operation does not remove its data-handling constraints.

Preferences among secure alternatives. Two public-data exports satisfy all hard requirements and share a matching permit. Candidate a has privacy cost 1 and progress cost 0; b has the reverse. At the default priorities, b is preferred because privacy has the higher priority. If progress received the higher priority, a would be preferred. This is a choice among admissible abstract behaviors. The second candidate is not executed merely because it was evaluated, and a successful commit would consume the shared permit.

Revocation and underdetermination. After an export is selected, the teacher revokes its permit. Commit re-evaluates authorization and blocks execution. This exposes the distinction between a decision made under old state and a transition attempted under current state. Finally, two inert observations with identical preference costs yield UNDERDETERMINED. The evaluator does not invent a distinction absent from the specified policy.

## 3.4 Guarantees

We state the security invariants of SFR and prove them by induction, show that only T may admit an action, and identify a gap that the invariants do not cover. Section 4 checks the invariants on the reference implementation.

A state is reachable if it results from the initial state by a finite sequence of grant, revoke, tick, and commit events.

Theorem 1 (Security invariants): Under A1–A3, every step from a reachable state satisfies:   
I1 Authorization: a successful protected commit by actor x for operation o occurs only if the permit is live for x   
and o.   
I2 Single use: after a successful protected commit, no permit is active.   
I3 Explicitflow: no successful protected commit moves confidential data to a public sink.   
I4 Policy integrity: v changes only by a successful update from a teacher with identity and source evidence T; the   
new version is v + 1 and the installed policy is within bounds.   
I5 Version binding: every active permit satisfies $v _ { \pi } = v .$   
I6 Conservative gate: no successful commit has a property value in {N, B}, and rejected commits and grants leave   
the state unchanged.

Proof: I1, I2, I3, I6, and the first part of I4 follow from Eq. (6), Table 2, and the effects of commit, because commit re-evaluates every property in the current state. For I5 and the bounds in I4 we use induction over steps. The initial policy is valid and no permit is active. A grant binds $v _ { \pi } = v .$ . An accepted update increments v, passes the structural check, and revokes the permit. No other event changes v or the policy. □

Admission designates T only. The following lemma shows that this choice is forced by the invariants.

Lemma 1 (Only confirmed evidence admits): Let D be a set of evidence values with ${ \mathsf { T } } \in D .$ , and let adm $_ D ( s , a )$ hold iff $P _ { i } ( s , a ) \in D$ for all i. Then I1–I6 hold under adm<sub>D</sub> if and only if $D = \{ \mathsf { T } \}$

Proof: For $D = \{ \mathsf { T } \}$ , adm is Eq. (6) and Theorem 1 applies. Otherwise D contains some $x \neq \mathsf { T } .$ . In the initial state, consider a policy update by the teacher role with identity evidence x, source evidence T, the current expected version, and a valid successor policy. Then $P _ { 1 } = x$ and $P _ { 2 } , \ldots , P _ { 6 }$ are T, so the update is admitted and commits although its identity evidence is not T, which violates I4. □

```c
typedef enum { N=0, T=1, F=2, B=3 } MV;
/* bit 0: e+ (support for T), bit 1: e- (for F) */
static MV mv_and(MV a, MV b) {
return (MV)((a & b & 1) | ((a | b) & 2));
}
/* P3 for an update: source evidence AND
e.values[INTEGRITY] = mv_and(a->trusted_source,
valid_policy(&a->replacement)
&& s->policy.version < UINT_MAX
&& a->replacement.version
== s->policy.version + 1 ? T : F);
/* Eq. (6): all COUNT = 6 properties T */
e.admissible = 1;
for (i = 0; i < COUNT; ++i)
if (e.values[i] != T) e.admissible = 0;
```  
Figure 3: Excerpt from the C implementation: evidence encoding, conjunction as in Eq. (3), composition of source evidence with a Boolean structural check (embedded as T/F), and admission as in Eq. (6).

Neither the paraconsistent choice $D \ = \ \{ { \sf T } , { \sf B } \}$ nor the gap-tolerant choice $D \ = \ \{ { \sf T } , { \sf N } \}$ is compatible with the invariants.

A confused-deputy gap and its repair. The invariants do not constrain the source of a protected operation: $P _ { 3 }$ is inactive for tool calls and exports. Imagine a scenario where a student holds a tool permit and an injected web page causes it to propose a tool call; the candidate carries the student’s identity and role and the live permit, so it is admitted although its source is untrusted. This is the confused deputy of [9]. Base SFR therefore protects the policy against injected content, but not tool use. The repair activates $P _ { 3 }$ for protected operations and requires source evidence T there as well. With this extension, a protected operation requires confirmed provenance. The price is that adapters must report the provenance of the instruction that caused a call, which is what capability-based data-flow tracking provides [21].

## 4 Reference Implementation and Conformance

The C artifact requires neither a solver nor a language-model API. The structures Policy, State, Action, and Evaluation correspond to the components above. The function evaluate returns six evidence values and an admission flag; select action performs filtering and ranking; commit revalidates and applies a transition. Grant, revoke, and clock-advance events have separate functions.

Figure 3 shows the compact evidence representation and the essential admission test. The negative support mask uses bit 1, so the integer 2 represents refutation; it is not a severity weight. The feedback is derived from the property outcomes, not from the model’s self-assessment.

The feedback routine returns positive-example for an admissible candidate. If admission fails and at least one hard property is exactly F, it returns counterexample. Otherwise, it returns teacher-review. Thus, a definite failure takes precedence over unresolved evidence in the coarse feedback label. The full property vector remains available for diagnosis; an inconsistent value B is not silently converted to either truth or refutation.

With six fixed properties, evaluation costs constant time per candidate and selection is linear in the number of candidates. Storage for state and each candidate is fixed. This complexity concerns the reference reasoner only: obtaining identity or semantic evidence may be substantially more expensive.

A rejected grant preserves the permit; a valid grant replaces it. Rejected commits preserve all state fields. Invalid enumeration values and costs are rejected, and successor-version and deadline arithmetic are guarded. Candidates are constructed internally, without an untrusted text parser.

## 4.1 Exhaustive Bounded Admission Enumeration

The test mode constructs the Cartesian product of identity evidence (4 values), source evidence (4), role (3), operation (4), input label (2), destination label (2), and permit availability (2). The total is $4 \cdot 4 \cdot 3 \cdot 4 \cdot 2 \cdot 2 \cdot 2 = 1 { , } 5 3 \dot { 6 }$ . Every case starts from the default valid policy and uses the current expected version and a structurally valid replacement. For tool/export cases with availability enabled, the test creates a matching fresh permit through the trusted grant function. For observations and updates, the availability flag duplicates otherwise identical cases because permits are irrelevant

Table 3: Admission outcomes in the bounded enumeration.
<table><tr><td>Operation</td><td>Cases</td><td>Admitted</td><td>Blocked</td></tr><tr><td>Observation</td><td>384</td><td>384</td><td>0</td></tr><tr><td>Policy update</td><td>384</td><td>8</td><td>376</td></tr><tr><td>Tool call</td><td>384</td><td>24</td><td>360</td></tr><tr><td>Data export</td><td>384</td><td>24</td><td>360</td></tr><tr><td>Total</td><td>1,536</td><td>440</td><td>1,096</td></tr></table>

there.

A separately expressed Boolean oracle is compared with evaluation and commit results. Rejected commits are checked for preservation of every state field. The oracle does not use evaluator helpers, but both can share specification errors.

Table 3 partitions the enumerated domain. Of 1,536 cases, 440 are admissible and 1,096 are blocked. Among the blocked cases, 160 have teacher-review feedback and 936 have counterexample feedback. These counts characterize the constructed domain; they are not probabilities of successful or malicious activity.

Observations are intentionally inert. Updates require confirmed identity and provenance plus teacher role. Tools and exports require confirmed identity, an internal role, a permit, and one of three allowed label pairs; their provenance dimension is inactive. These activation rules explain the counts.

## 4.2 Algebra, Sequences, and Boundary Cases

Further assertions cover double negation, commutativity of conjunction, De Morgan’s law, a material-implication case, and mixed evidence. Sequence checks exercise single-use permits, revocation before commit, the inclusive deadline, expiry after the deadline, actor and operation binding, stale versions, invalidation after an update, and rejected rollback. Preference tests cover ties, priority reversal, dominance of hard constraints, and empty candidate sets. Boundary tests cover clock and deadline overflow and invalid evidence values.

The run reports 4,617 passed assertions, including setup and state-preservation checks, rather than 4,617 distinct properties.

## 4.3 Bounded Exhaustive Exploration

To check the theorem against the reference implementation, a harness compiles the unchanged C code and drives it own functions. A breadth-first search enumerates all states reachable from the initial state. In each state it applies every tick and revoke, every grant (all issuer roles, evidence values, actors, and operations), and commits every candidate. Candidates range over all operations, roles, three actor identifiers, evidence values, and labels, over current and stale expected versions, and over valid and invalid replacement policies; roles and actors are combined freely, including combinations that A1 excludes. Bounds are $t \subseteq 3 , v \leq 4 , \ell \in \{ 1 , 2 \}$ , and priorities in {30, 40}; states that differ only in the execution and update counters, which only overflow guards read, are identified. The search visits 6,262 reachable states and checks 72,329,329 transitions; none violates I1–I6. To guard against vacuous checks, we seeded faults into copies of the code. Admitting B is detected by I4 and I6, and keeping the permit after an update by I5. Removing the deadline check from the permit test alone is undetectable, because a tick already revokes an expired permit; removing both deadline checks is detected by I1. The search also counts successful protected commits whose source evidence is not T, the confused deputy of Sec. 3.4: it finds 21,708 such transitions. With the extension it finds none, and I1–I6 still hold. The extension is implemented as a second version of the reference implementation that adds a single line to evaluate; in its enumeration, 6 instead of 24 tool calls and 6 instead of 24 exports are admitted. The three kinds of checks differ in scope: the exploration covers all reachable states within its bounds. Enumeration exhausts one base state’s input product; sequence checks sample histories. None of these checks is a C correctness proof. No deployed identity service, language model, or network tool is evaluated.

## 5 Discussion and Direction for Future Work

What SFR covers. Under A1–A3, Theorem 1 covers the policy-update channel, the authorization of tool calls and exports, and the freshness of proposals and permits. With the extension of Sec. 3.4, it also covers protected operations triggered by untrusted content. All guarantees are conditional on A1: SFR specifies what adapters must deliver but does not deliver it. The model deliberately keeps the protected state small. The aspects below are possible extensions of the approach: each adds operations, properties, or state components, while the general concept remains unchanged.

Extensions of the model. The protected policy currently consists of the preference priorities and the permit lifetime; FR’s property set and hard flags are fixed. Protecting their updates adds structural checks to $P _ { 3 }$ but no new mechanism. Review answers and training material flow back into raising outside the model; treating curriculum changes as a further protected operation, with role and source requirements as for policy updates, covers this channel. One permit record and an atomic, single-threaded commit keep the state small; several permits and concurrent tools require a set of permits and defined semantics for consumption and partial failure. The two-label flow check generalizes to a lattice of labels [7], and a teacher-authorized declassification can be added as a permit-like operation. Finally, an authenticated teacher is trusted; requiring several teachers to approve an update adds one property of the same kind.

Who may contribute evidence. Because observations are merged by Eq. (5), any accepted refuting observation turns T into B and blocks the action. This is the intended conservative behavior, but it gives a veto to every party whose observations are merged. Adapters must therefore restrict, per proposition, who may contribute evidence, and teacher review must resolve B within a bounded time; otherwise a single source can deny service. The same applies to repeated conflicts in general.

Relation to raising. SFR provides the environment-side guarantees of raising and a feedback signal: admitted actions are positive examples, refuted properties are counterexamples, and missing or conflicting evidence is routed to a teacher. Whether training on these traces changes a learner’s proposals has not been tested. A first experiment would train a small model on scenario traces and measure how often it still proposes actions that SFR blocks.

## 6 Conclusion

Secure Formal Raising extends the property-oriented perspective of FR to the environment in which an AI system develops and acts. The reference model protects preference-policy updates, scopes and revokes tool authorization, and blocks confidential-to-public transfers. Four-valued evidence retains uncertainty and conflict while admitting only confirmed hard requirements. Preferences rank admissible alternatives without creating exceptions to security constraints.

We proved six security invariants and showed that they force admission to accept only confirmed evidence: designating unknown or inconsistent evidence as well admits a policy change without confirmed identity. A plain-C reference implementation makes the model executable; a bounded exhaustive exploration confirms the invariants for it and exposes a confused-deputy gap for tool use, which a provenance requirement closes. The results concern an abstract environment and do not establish security of arbitrary language-model internals or a deployed raising pipeline. Future work will connect the interfaces to real adapters and study how property traces support curriculum construction and teacher feedback.

## References

[1] A. M. Turing. Computing machinery and intelligence. Mind, 59(236):433–460, 1950.

[2] R. Aydin, C. Cyron, S. Bachelor, A. Anderson, and R. West. From model training to model raising. Communi cations ofthe ACM, 69(2):24–27, February 2026.

[3] R. Drechsler and C. Plump. Formal Raising: Embedding verifiable ethical properties into the development of AI. accepted at International Conference on LLMs, preprint on arXiv, arXiv:2610.25501, 2026.

[4] L. Dennis, M. Fisher, M. Slavkovik, and M. Webster. Formal verification of ethical choices in autonomous systems. Robotics and Autonomous Systems, 77:1–14, 2016.

[5] J. H. Saltzer and M. D. Schroeder. The protection of information in computer systems. Proceedings ofthe IEEE, 63(9):1278–1308, 1975.

[6] R. S. Sandhu, E. J. Coyne, H. L. Feinstein, and C. E. Youman. Role-based access control models. Computer, 29(2):38–47, 1996.

[7] D. E. Denning. A lattice model of secure information flow. Communications ofthe ACM, 19(5):236–243, 1976.

[8] J. A. Goguen and J. Meseguer. Security policies and security models. In Proceedings of the IEEE Symposium on Security and Privacy, pages 11–20, 1982.

[9] N. Hardy. The confused deputy (or why capabilities might have been invented). ACM SIGOPS Operating Systems Review, 22(4):36–38, 1988.

[10] N. D. Belnap, Jr. A useful four-valued logic. In J. M. Dunn and G. Epstein, editors, Modern Uses of Multiple Valued Logic, pages 5–37. D. Reidel, Dordrecht, 1977.

[11] M. L. Ginsberg. Multivalued logics: A uniform approach to reasoning in artificial intelligence. Computational Intelligence, 4(3):265–316, 1988.

[12] O. Arieli and A. Avron. Reasoning with logical bilattices. Journal ofLogic, Language and Information, 5(1):25– 63, 1996.

[13] G. Bruns and M. Huth. Access control via Belnap logic: Intuitive, expressive, and analyzable policy composition. ACM Transactions on Information and System Security, 14(1):9:1–9:27, 2011.

[14] OASIS. eXtensible Access Control Markup Language (XACML) Version 3.0. OASIS, January 2013. OASIS Standard.

[15] F. B. Schneider. Enforceable security policies. ACM Transactions on Information and System Security, 3(1):30– 50, 2000.

[16] M. Leucker and C. Schallhart. A brief account of runtime verification. The Journal of Logic and Algebraic Programming, 78(5):293–303, 2009.

[17] M. Alshiekh, R. Bloem, R. Ehlers, B. Konighofer, S. Niekum, and U. Topcu. Safe reinforcement learning via¨ shielding. In Proc. AAAI Conference on Artificial Intelligence, 2018.

[18] IEEE. IEEE Standardfor Property Specification Language (PSL), 2010. IEEE Std 1850-2010.

[19] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz. Not what you’ve signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. In Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec ’23), pages 79–90, 2023.

[20] E. Debenedetti, J. Zhang, M. Balunovic, L. Beurer-Kellner, M. Fischer, and F. Tram ´ er. AgentDojo: A dynamic\` environment to evaluate prompt injection attacks and defenses for LLM agents. In Proceedings of the 38th International Conference on Neural Information Processing Systems (NIPS ’24), volume 37, pages 82895–82920, 2024.

[21] E. Debenedetti et al. Defeating prompt injections by design. arXiv:2503.18813, 2025.