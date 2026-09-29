# Fewer Assumptions by Design: A Reusable Skill for LLM-Assisted Verus Verification

Andrada-Livia Antoneac<sup>1,2</sup>

Dorel Lucanu<sup>2</sup>

aantoneac@bitdefender.com

dorel.lucanu@info.uaic.ro

Dragos Teodor Gavrilut<sup>1,2</sup>

dgavrilut@bitdefender.com

<sup>1</sup>Bitdefender, <sup>2</sup>Alexandru Ioan Cuza University of Iasi

LLM-assisted Verus verification is a less tedious method to verify Rust implementations, but paired with self-referential structures, e.g., Doubly Linked Lists (DLLs)—notoriously difficult to formalise for verification—it becomes a substantially more demanding verification task. Moreover, a specification weakness can arise when verification relies on unproven or invalidated assumptions, such as axiomatic lemmas and assume statements. We investigate whether LLM agents can synthesize strong DLL specifications while minimizing these trusted base. The analysis follows three different approaches: manual verification, property-specific verification, and a defined skill for the specific case of DLLs and certain properties of this type of data structure. The skill encodes domain knowledge and a task-decomposition strategy. We show that an LLM agent equipped with a carefully designed verification skill can generate strong, low-trust specifications for DLLs in Verus.

## 1 Introduction

Verifying multiple implementations against a shared abstract specification is a good stress test for LLMassisted formal verification. The proof obligations share a common logical skeleton, but each has unique requirements that must be handled concretely.

The paper addresses the question of whether an AI model can be used to efficiently verify afamily of structurally related Rust implementations against a shared formal specification, and whether the process can be codified into a reusable methodology.

DLLs are a canonical data structure, conceptually simple, but notoriously difficult to verify formally because their correctness depends on mutual consistency constraints between forward and backward reference chains. In [1], eleven different Rust implementations of a DLL are compared in terms of execution time, memory footprint, and memory safety. All implementations have the same abstract trait-based description, comprising eight operations (next, prec, push\_back, push\_front, insert\_- before, insert\_after, delete, search), and differ in memory mode (heap-based, resizable-arraybased, and map-based) and in the node identifier type. That paper evaluates the implementations empirically for time and memory performance, and tests them for three safety vulnerabilities. What it does not do is formally verify them. What we aim to add here is a way to formally verify these implementations, as their functional correctness has not been addressed there.

Verifying all eleven implementations against the shared abstract specification would be an ideal case study for answering the main research question and the work presents the first steps we take to achieve this goal.

The main contributions include a systematization of the challenges raised by such a verification, a set of principles for designing a verification-oriented skill for AI agents based on these challenges, and the implementation of a skill based on these principles.

The article is structured as follows: Section 2 presents the use-case and the similar approaches to AIassisted verification, Section 3 is a progressive, three-stage case study of formal verification of DLL and the resulting challenges, Section 4 presents the reusable verification generation skill we created for this context, found at https://github.com/andradaAntoneac/generate-verus-specification, along with key findings from its results, summarized in the concluding Section 5.

## 2 Background

## 2.1 Formal Verification in Rust

The most popular tools for formal verification in Rust can be grouped by the methodology they follow, showcasing the differences between them. Model checkers explore program execution paths rather than construct a mathematical proof, performing an exhaustive search to check all possible states a program can reach and verify whether a property holds for those states [7]; they provide high automation but limited capabilities, as exemplified by Kani [13], a tool built over CBMC [9], a bit-precise bounded model checker designed for C programs and ideal for catching memory-safety violations such as use-after-free and out-of-bounds accesses in unsafe Rust. Deductive verifiers, in contrast, explore all possible execution paths and all possible inputs to construct a mathematical proof, which makes them unbounded and able to provide full correctness proofs, usually relying on an SMT prover; representatives include Prusti [2], built over the Viper framework (Verification Infrastructure for Permission-Based Reasoning) yet still in prototype form, and Verus [10], which uses Z3 directly, offers a specification language very similar to Rust, and is designed specifically for low-level systems code. Finally, translation-based approaches translate Rust code into another formal representation that can be verified independently, usually a backend like Why3, Coq, Lean, or F\*; these include Creusot [5], which translates Rust into WhyML for verification via Why3, Aeneas [6], which translates safe Rust into a pure functional model for proof assistants (Coq, Lean, F\*), and RustBelt [8], which provides the theoretical justification for why safe abstractions over unsafe code are sound but is not suitable for proofs over specific cases such as the one presented in this article.

## 2.2 AI-Assisted Verification

Using large language models (LLMs) as assistants for formal verification is a rapidly evolving research direction. Most existing work automates the proof-engineering pipeline end-to-end, treating the LLM as a component in a larger system.

Automated pipeline approaches for Verus. AutoVerus [3] is an early automated system from Microsoft Research that orchestrates a network of LLM agents mimicking the three human proof construction phases: preliminary proof generation, refinement guided by generic tips, and debugging guided by verification errors. The same group later introduced a self-evolution strategy in which GPT-4o is used to synthesise proof annotations for over one thousand Rust functions, and the resulting data is used to fine-tune a smaller open-source model that iteratively improves itself [3]. VeriStruct [12], accepted at TACAS 2026, extends AI-assisted verification from single functions to full data-structure modules in Verus. Its planner module orchestrates the systematic generation of view functions, type invariants, specifications, and proof code; a subsequent repair stage corrects annotation errors automatically. Applied to eleven Rust data-structure modules, VeriStruct succeeds on ten, verifying 128 of 129 functions (99.2%). VeruSAGE [14] introduced a system-verification benchmark of 849 proof tasks extracted from eight open-source Verus-verified Rust projects, and studied how different LLMs (Claude Sonnet 4, Sonnet 4.5, GPT-5, and o4-mini) perform across specialised agents handling logic, arithmetic, and proof context. The best LLM–agent combination solves over 80% of the benchmark tasks.

Automated pipeline approaches for Dafny. Parallel work on the Dafny verifier [11] demonstrated that GPT-4 augmented with retrieval (RAG) and chain-of-thought prompting synthesises verified Dafny methods for 58% of benchmark problems. More recently, DafnyPro [4] achieves 86% on the DafnyBench suite using Claude Sonnet 3.5, a 16-point improvement over the prior state of the art.

Our approach: interactive exploration. The present work differs from the above in several important ways. Usually, LLMs are used as a black-box component embedded in a fully automated end-to-end pipeline. Our work adopts an interactive, exploratory methodology where a user collaborates with an agent to progressively strengthen specifications. This research’s primary goal is the quality of the resulting specifications, specifically, minimising the trusted base by reducing assumed statements and axiomatic lemmas. Whereas prior systems optimise for automatically closing proofs, we deliberately study the family of structurally related DLL implementations against a shared abstract specification, exposing what aspects make this data structure hard to verify.

## 2.3 Doubly Linked Lists

The investigation in [1] presents how to implement a DLL in Rust while satisfying performance requirements and the absence of security vulnerabilities. The focus of the article is on finding the most balanced DLL design when Rust’s ownership rules are taken as a hard constraint. The paper contributes (i) a traitbased abstract specification shared by all implementations, (ii) a taxonomy of three memory models, (iii) eleven concrete implementations, and (iv) a comparative evaluation across nine performance and three safety test scenarios. All artefacts are publicly available at https://github.com/xTachyon/ paper\_doubly\_linked\_lists.

Table 1: Informal contracts for the eight DLL operations. L denotes the current list; L<sup>′</sup> the list after the operation; $i d _ { n e w }$ the fresh node
<table><tr><td>Operation</td><td>Contract (informal summary)</td></tr><tr><td>next  $( i d _ { t } )$ </td><td>Returns None if L is empty or  $i d _ { t } = L . 1 { \tt a s t } ( ) ;$  else the unique id s.t.  $L . \mathsf { p r e c } ( i d ) = i d _ { t }$ </td></tr><tr><td>prec  $( i d _ { t } )$ </td><td>Returns None if L is empty or  $i d _ { t } = L . \pm \mathsf { i r s t } ( ) ;$  else the unique id s.t. L.next(id) =  $i d _ { t }$  . (Mutually recursive with next.)</td></tr><tr><td>push_back(v)</td><td> $L ^ { \prime } = L \not \Rightarrow \left\{ i d _ { n e w } \right\}$  ; new node is last; if L was empty it is also first. Returns  $i d _ { n e w } .$ </td></tr><tr><td>push_front(v)</td><td> $L ^ { \prime } = L \not \Rightarrow \left\{ i d _ { n e w } \right\}$  ; new node is first; if L was empty it is also last. Returns  $i d _ { n e w } .$  insert_after (idt , v) New node inserted immediately after id; if idt was last, new node becomes last.</td></tr><tr><td>insert_before  $( i d _ { t } , \nu )$ </td><td>New node inserted immediately before  $i d _ { t } ;$  if idt was first, new node becomes first.</td></tr><tr><td>delete(id)</td><td> $L ^ { \prime } = L \setminus \{ i d \}$  ; predecessor and successor pointers of neighbours updated; first/1ast updated if needed.</td></tr><tr><td>search(f)</td><td>Returns some  $i d \in L \mathrm { ~ s . t . ~ } f ( L . \mathsf { v a l u e } ( i d ) ) = t r u e ,$  or None if no such node exists.</td></tr></table>

Trait-based specification. The specification of DLLs in [1] is expressed as a trait, making the required behaviour explicit and independent of any concrete memory representation. The central abstraction is an associated type NodeId, which is the implementation’s identifier for a list node (a pointer, an integer index, or a map key, depending on the implementation). Each of the eight primary methods is accompanied by a semi-formal Input/Output/Return contract in the companion paper. Table 1 summarizes the contracts used as the target specification in our experiments.

Here we give, as examples, two trait-based descriptions: Specification 1 supplies the contract of the next method and Specification 2 that of prec. A key property of these specifications is that the contracts are mutually recursive: the return clause of next is defined in terms of prec, and vice versa. This means correctness cannot be established for any single operation in isolation; the entire list implementation must be verified as a whole.

Specification 1 Next Specification 2 Prec   
next(NodeId id ): NodeId; prec(NodeId id<sub>t</sub>): NodeId;   
Input Input   
self: the current list self: the current list   
$i d _ { t } \colon$ the target node in self id<sub>t</sub>: the target node in self   
Output Output   
unchanged unchanged   
Return Return   
if self.is\_empty() or id<sub>t</sub> = self.last() if self.is\_empty() or id<sub>t</sub> = self.first()   
then None then None   
else id s.t. self.prec(id) = id else id s.t. self.next(id) = id

Memory models and implementations. The eleven implementations divide into three groups based on how nodes are stored and how NodeId is represented (Table 2).

Table 2: The eleven DLL implementations from [1].
<table><tr><td></td><td>Group Implementations</td><td>NodeId type</td><td>Key mechanism</td></tr><tr><td>Heap</td><td>RawP, NonNull, Rc, Std</td><td></td><td>raw/smart pointer Heap allocation; Rc/Weak for cycles</td></tr><tr><td>Array</td><td>Index, Slab, Handle, Arena u32 / wrapper</td><td></td><td>Vec + free-list; generation counts</td></tr><tr><td>Map</td><td>HashMap, BTreeMap, SlotMap usize key</td><td></td><td>Key-value store; monotone key counter</td></tr></table>

In the heap group, each node is an independently heap-allocated struct and NodeId is a raw pointer or a pointer wrapper. The self-referential structure requires either raw pointers (RawP, NonNull) or smart pointers such as Rc and Weak. In the array group, all nodes are stored as Option<Element<T>> slots in a Vec; a separate free-list tracks available slots (Index), or a third-party arena crate manages slot reuse (Slab, Handle, Arena). In the map group, nodes are stored as values in a hash map, B-tree map, or slot map, and NodeId is a monotonically increasing key, specific to each map type. The Std implementation wraps Rust’s standard-library std::collections::LinkedList and serves as a performance baseline.

## 3 A Progressive Verification Case Study

The research follows three scenarios, each providing challenges and insights. Identifying the challenges and proposing solutions is the basis of understanding the needs of a Verus proof in terms of syntax and proof reasoning. Each experiment is intended to address the shortcomings of the previous one.

The main goal is to create a specification as close as possible to a complete machine-checked verification, minimising the assume statements, which introduce unproven hypotheses inside a proof or executable block, and axiomatic lemmas—external body specification functions without a body— defined only by the contract (preconditions and postconditions).

The DLL implementations that are explored in this section can be found at https://github.com/ xTachyon/paper\_doubly\_linked\_lists and the results of the experiments at https://github. com/andradaAntoneac/Verus-dll-Experiments.

## 3.1 First Experiment - Manual Verification

The first experiment is an onboarding exercise: we hand-write verification code for the index-based DLL to learn Verus syntax and its proof model. Contracts are deliberately weak, so many correctness obligations, such as node reachability preservation and view-length updates, are discharged with local assume statements, leaving key proofs informal. The modeled conditions capture only the most obvious shape constraints, such as index bounds and basic list validity. As a result, the experiment prioritizes generating a working Verus model over expressing the full, precise correctness properties expected from a machine-checked proof (see Table 3). Because nodes are referenced by array indices rather than Rust references, Verus cannot automatically relate a mutation at index i to the abstract list view. Any update of the structure has to be asserted manually and captured in the list view (the abstract, mathematical representation of the list) synchronization as well.

<table><tr><td colspan="2">Summary</td></tr><tr><td>Invariant strength</td><td>shape-only (index bounds, basic validity)</td></tr><tr><td>Assume count</td><td>9 (reachability of every node after an altering operation)</td></tr><tr><td>Verified / total functions</td><td>16</td></tr><tr><td>Axiomatic Lemmas</td><td>3 (valid neighbours of a reachable node are also reachable, the view of the list after a deletion is the old view where the deleted node is skipped, the deletion of a node results in a view length smaller than the</td></tr><tr><td>Operations covered</td><td>old view by 1) new, allocate, link, push_back, push_front, insert_before,</td></tr><tr><td>Residual gaps Takeaway</td><td>insert_after, delete reachability, view-length, and chain consistency all assumed establishes baseline fluency: identifies aliasing, view sync, and fram-</td></tr></table>

Table 3: First Specification Version Summary for Index DLL

## 3.2 Second Experiment - Property-Specific Verification

The second experiment shifts to enforcing a specific, central property, in this case, the validity of the chain in the DLL. The validity of the chain is defined as follows: the first node must be reachable from the last node by following prec links, and the last node must be reachable from the first node by following next links, both within the same length-bounded traversal. The specification is strengthened with a well-formedness predicate (wf() is the conventional name) and richer invariants, and the proof burden is shifted into explicit axiomatic lemmas (e.g., chain preservation, update-field stability, node reachability after linking). The number of manual assumptions is still high, as it is still not an ideal version of complete proof, but rather an illustration of where and how a property can be asserted throughout the DLL specific operations. The residual gaps are attributable to the DLL’s index-based memory model, which stresses both Rust’s ownership discipline and Verus’s SMT encoding (see Table 4).

<table><tr><td colspan="2">Summary</td></tr><tr><td>Invariant strength</td><td>chain-focused (wf () with reachability, index partitioning)</td></tr><tr><td>Assume count</td><td>6 (the validity of the chain for intermediate states, the validity of each index in the list)</td></tr><tr><td>Verified / total functions Axiomatic Lemmas</td><td>11 2 (the validity of the chain in both directions based on the fact that the</td></tr><tr><td></td><td>list view remains unchanged)</td></tr><tr><td>Operations covered Residual gaps</td><td>new, allocate, link, push_back missing insert/delete ops; chain preservation in link/push_back still as-</td></tr><tr><td></td><td>sumed</td></tr><tr><td>Takeaway</td><td>expands from allocation-only to structural updates while keeping proofs lightweight; still relies on assumptions for chain reasoning</td></tr></table>

Table 4: Second Specification Version Summary for Index DLL

## 3.3 Claude-Assisted Scenario

The experiment with Claude targets complete formal verification of the index-based DLL implementation against the full full\_wf() invariant, twelve structural conjuncts covering slot/free-list bookkeeping, ghost-set membership, chain integrity, and the abstract list-view sequence, modelling multiple properties across all eight public operations. The Verus specification was generated by Claude, using the trait-based description and the implementations as input.

The work evolves through four phases tracked by an assume statement count that dropped from 75 to 11. The key architectural decisions are a two-level invariant split (wf() for bookkeeping, full\_wf() for structure) that decouples the helper primitives (e.g., allocate and link) from chain reasoning, and the introduction of an explicit ghost field dll: Ghost<Set<u32>> in place of a computed set definition, which turns opaque lambda queries into first-class SMT propositions. Phase 2 discharges 17 acyclicity assume statements by adding explicit conjuncts and five new lemmas. Phase 3 eliminates a further 44 assume statements through the ghost boolean pattern (capturing link postconditions in named ghost booleans before they leave Z3’s local context) and strengthens allocate postconditions. Phase 4 closes 3 more via direct witnesses. The final state is 45 verified, 0 errors, 11 assumes, with the residual 11 all sitting at the chain-to-sequence bridge, connecting the exact-hop-count chain\_n predicate to the abstract list\_view() sequence index, which is the one structural gap neither the lemma library nor Claude is able to close within the experimental scope. The encountered challenges are intrinsic to the problem domain (verification of DLLs in Rust). However, the main problematic aspect is the AI agent’s prolonged reasoning time (a couple of hours) and the need for the user to manually prompt the agent about aspects it had omitted from the proof.

## 3.4 Main Challenges

After experimenting with multiple approaches in the above described scenarios, we can formulate the main challenges of the Verus-based verification for DLLs. They are split into three categories: challenges related to the Verus syntax and proof structure, challenges intrinsic to the DLLs’ particularities and memory models, and the ones resulting from the combination of the abstract model and the technology. The challenges are formalised in collaboration with Claude, as most of them have been encountered during the multiple generation sessions. Some of them were suggested by Claude himself, and the authors edited and checked them.

## 3.4.1 Challenges Raised by Using Verus in the Verification of Rust Implemementations

Limited support for raw-pointer implementations. The delete operation and the raw-pointer implementations use Rust’s unsafe keyword. Verus can verify unsafe code in principle, but the required annotations (pointer validity predicates, aliasing restrictions, layout proofs) add significant overhead and interact with the existing ghost infrastructure in non-trivial ways, leaving this kind of support incomplete.

SMT trigger management. Verus compiles its annotations into SMT queries discharged by Z3. Z3 reasons efficiently about ground facts but is highly sensitive to how universally quantified propositions are triggered. A set defined as {i | data[i].is\_some()} is opaque to Z3 unless a trigger fires on a membership query. In practice, every universally quantified invariant conjunct requires careful placement of #[trigger] annotations to ensure Z3 instantiates it at the right proof sites. If a trigger’s specification is missing, then Z3 chooses one that might not be of any help in the proof. Incorrect triggers produce silent failures, as the invariant is logically true, but Z3 cannot see it.

Ghost state ordering and old(self)/final(self) semantics. Verus’s old(self) always refers to the function entry state. In a multi-step operation such as push\_back (allocate then link then an assignment), the postconditions of allocate are removed from Z3’s context when the next mutation is issued. A let ghost snap = self.data@ immediately after each mutating call is required to preserve intermediate states. Omitting a snapshot produces an error that may manifest several proof lines later with a message that is not obviously connected to the missing snapshot. This issue is pervasive in any multi-mutation operation and constitutes a significant cognitive overhead for AI-generated proofs, which must explicitly plan the ghost snapshot sequence.

Mutable references as function parameters. A closely related pitfall arises whenever a helper function receives a mutable reference (&mut) to the structure. The prover treats the call as an opaque black box and assumes that any field may have been arbitrarily modified, retaining only the facts explicitly guaranteed by the callee’s postconditions. Consequently, the preservation of fields that the function does not intend to change must be stated explicitly in its ensures clause (e.g., final(self).other\_field == old(self).other\_field).

## Proof block placement relative to exec assignments. Executable assignments that update

self.first, self.last, and self.dll may pass through intermediate states that violate chain\_wf(). Establishing the required facts about these transient states demands proof blocks placed at exactly the right point in the statement sequence, a placement that is not always easy to guess. A misplaced proof block asserts a property of the wrong intermediate state, and Verus reports only the final postcondition failure, giving no hint that the block itself is the cause.

Recursive spec function opacity. Verus’s spec functions are not unfolded automatically by Z3 beyond a finite recursion depth. For the predicates that are defined by structural recursion on a fuel parameter, Z3 will not unfold them without an explicit proof lemma that provides the inductive step. This means every use of such a predicate at a specific depth requires a lemma invocation.

Bitvector and mixed-arithmetic casts. Verus distinguishes u32, nat, and int as separate types and does not automatically insert casts. Subtraction of two nat values produces an int (since the result could in principle be negative), so an expression such as old\_pos[i] - 1nat has type int and is rejected by a Map<u32, nat> value function. The fix is (old\_pos[i] - 1) as nat, but this cast is only sound when old\_pos[i] >= 1, which itself requires a proof.

Z3 resource limits and context size. As the proof file grows, Z3’s default resource limits become binding. Functions with large proof blocks (e.g., with many cases and multiple lemma calls per case) occasionally time out. The symptom is a resource-limit error rather than a clear proof failure, making it hard to distinguish “proof is wrong” from “proof is right but Z3 ran out of time.” Decomposing large proof blocks into smaller helper lemmas, or adding proof { } sub-blocks to give Z3 smaller sub-goals, mitigates this but further increases the number of proof artefacts to maintain.

## 3.4.2 Challenges Raised by the Specification and Verification of DLLs Implemented in Rust

Self-referential structure and Rust’s ownership model. A DLL is inherently self-referential, meaning each node holds identifiers to its two neighbours, a mutual borrow that safe Rust forbids. The eleven implementations of [1] each circumvent this constraint by a different mechanism: raw pointers, NonNull, Rc<RefCell<T>>, integer indices, or map keys. Every workaround displaces the self-referential burden from the type checker to the verifier. For raw-pointer implementations, the verifier must reason about pointer provenance and disjointness invariants that current verification tools support only partially. For the index-based implementation studied here, the verifier must track which slots are live, which belong to the free list, and which are reachable from first, three overlapping but distinct concepts that must be kept in sync.

Inherently global invariants. The DLL invariant cannot be decomposed into local per-node properties. Whether a node v is “valid” depends on the full list structure: its prec must equal the next field of the preceding node, which in turn depends on the preceding node’s position in the chain, and so on. This global character means that every invariant conjunct refers to the underlying memory model and every proof step must account for the global effect of even a single link update.

Specification imprecision. The semi-formal contracts of [1] are adequate for communication but not for machine-checked proof. The imprecision gap between documentation-level and verification-level specifications must be filled before starting the proof.

Memory model diversity across eleven implementations. The three memory model groups (heap, array, map) each require a distinct ghost infrastructure. For example, the predicates expressing the chaining properties are portable, but the surrounding invariants are not. Heap-based implementations require pointer disjointness and ownership invariants. Map-based implementations require key-monotonicity and absence of hash collisions. Sharing the chain lemma library across groups is possible (as Section 4 proposes), but each group’s wiring is unique. A verification effort covering all eleven implementations cannot avoid rebuilding the ghost infrastructure three times, once per group, and then specialising per implementation.

Absence of a deallocation model. The abstract specification has no model for memory recovery. The specification does not constrain what “deletion” means in terms of the abstract view. It specifies the list contents but not the physical memory layout after the operation. This simplification keeps the verification tractable, avoiding separation-logic-style capabilities, but it leaves the implementation’s space behaviour unverified. Any extension of the specification to include memory safety or space complexity would require a substantially more powerful specification framework.

Minimal postconditions for primitives. A function like allocate cannot prove wf() for the recycling path without a no-duplicates property on free\_list, which is not in wf(). Treating this correctly means weakening the primitive’s postconditions. The primitive should only promise facts that every caller can rely on, and avoid baking in stronger claims that require extra context. In other words, the challenge is to avoid over-specifying a low-level operation. The solution is to push stronger guarantees upward to the callers that actually have the necessary invariants.

## 3.4.3 Combined Challenges Raised by the Verification of DLLs Using Verus

The two preceding subsections catalogued Verus-tool challenges (Section 3.4.1) and DLL-domain challenges (Section 3.4.2) independently. In practice the two sets do not compose additively, but mostly each domain challenge amplifies one or more tool challenges, producing a combinatorial burden that exceeds the sum of its parts. This subsection describes the concrete interaction pairs observed in our experiments.

Global DLL invariants × SMT trigger management. Every conjunct of full\_wf() quantifies universally over all live nodes, so each must carry a trigger that Z3 can instantiate at every relevant proof step. A membership anchor such as #[trigger] dll@.contains(i) suffices for simple conjuncts, but several conjuncts additionally reference data@[i as int] or suffix\_pos@[i], requiring compound triggers. Across the conjuncts of full\_wf() and chain\_wf(), this yields a dense trigger-annotation discipline. The dangerous failure mode is not a missing trigger, which Verus usually reports, but a trigger that is well-formed yet too specific, so Z3 never fires it and the conjunct is never instantiated. Because the resulting failure surfaces as an unprovable ensures rather than any trigger diagnostic, one must distinguish “conjunct not implied” from “conjunct not instantiated”, a distinction Verus’s error messages do not surface, and which is recoverable only by inspecting quantifier-instantiation logs.

Self-referential mutation × ghost-state ordering and proof-block placement. Each mutating DLL operation performs some physical sub-steps (e.g., allocate a slot and link it into the chain), and the global invariants hold only after all sub-steps complete. Verus’s old(self) semantics captures state at the function boundary, not at intermediate sub-steps, so intermediate states require explicit ghost snapshots and proof blocks for the preconditions of the sub-steps.

Specification imprecision × ghost-state ordering. Specification gaps discovered during Verus proof derivation compound the multi-step ordering challenge. Wherever the abstract spec underspecifies a precondition, the Verus translation must choose a concretisation, and that concretisation interacts with the ghost-snapshot discipline.

Mutually recursive specification × recursive spec function opacity. The abstract DLL specification expresses next and prec as mutually dependent operations (Section 2.3). In Verus, these translate to the pair valid\_next\_chain/valid\_prec\_chain, both defined as recursive spec functions. Because Verus’s opacity rules prevent Z3 from automatically unfolding either function, every proof step that references the forward or backward chain requires an explicit lemma invocation. The mutual dependency doubles the required lemma library, as every structural property (frame, monotonicity, one-step extension, transitivity) must be proved independently for both directions.

Fuel bound specification × reachability property. "Reachability” is essential to show that links do not skip or cycle and valid\_next\_chain is the right statement for “the list is bounded-reachable”. It is not strong enough for inductive chain lemmas, which need the exact step count for their termination argument. Another function (in our case chain\_n) provides that count. For inductive chain reasoning in SMT, both a fuel-bounded reachability predicate and an exact-count predicate should be maintained.

## 4 Towards a Verification Skill for DLLs

A skill is a package of knowledge, context, and procedural guidance that an AI/LLM agent can load on demand, whenever the situation calls for it. It is usually perceived as a briefing document that the agent reads before tackling a specific kind of task, which replaces the long explanations in the prompts.

At its core, a skill is just a folder. It has a file that acts as an orchestrator. It presents what the skill is for, how to approach the task, what tools to use, what common mistakes to avoid, and how to handle errors. The skill can be accompanied by any other supporting materials, such as reference documents with domain-specific knowledge, scripts that automate parts of the workflow, and templates or other assets the agent might need.

The main purpose of a skill is to turn a general-purpose agent into a specialist. A well-written skill loads exactly the knowledge the agent needs, instead of forcing the agent to gather that information itself, a process that is often error-prone.

The experience accumulated during the experiments and the challenges encountered led to the following set of principles that the skill should follow when generating a Verus based verification of a DLL.

## 4.1 Principles Guiding the Skill Design

Separation of concerns. The skill isolates analysis, mapping, generation, verification, and debugging into distinct phases to avoid conflating specification design with proof tactics or diagnosis. The analysis refers to input verification, confirming the abstract specification (the starting point of the proof) and the actual implementation are provided and suitable to proceed. Between these two a mapping is drawn, meaning the operations modeled by the specification are identified in the implementation. Mapping serves as the guide for Verus specification generation, which is verified and debugged. This modularity supports clarity and repeatability. The workflow details can be found in Section 4.2.

Traceability. The abstract specification is translated into Verus specification code following the rule that every clause (preconditions defined in requires blocks, postconditions from ensures blocks, decreases statements, and invariants) is justified by a specific abstract requirement. This rule prevents arbitrary or speculative postconditions.

Soundness over convenience. Weakening specifications to help the verifier or to generate the specification faster should be forbidden. Bugs might still be found in the implementations, and the proper step after finding one is to report it first, rather than modelling the specification around the bug.

Reusable proof patterns. To avoid ad hoc reasoning for commonly used proof blocks, their structure has to be shaped beforehand. The agent should have knowledge about the structure of inductive proofs, lemma application, and trigger tuning, for example.

Boundary discipline. As per the aforementioned Verus limitations, the support for unsafe blocks, raw pointers and opaque library types is present but incomplete and insufficient for this kind of proof. The verification must switch to an axiom-boundary strategy in this context and document it.

Structure preservation. The structures used in the implementation must be preserved as much as possible, so the generated specification proofs are not modeled over surrogate structures. Ghost fields may be used as a means to help the proving process.

Coverage of behaviour. The skill requires as input a target property to be proven for the list operations, but that does not mean the operations’ own functional behavior can be ignored. For example, in the push\_back scenario, there must be a postcondition to verify that the last node has been updated. To avoid under-specification, the property description must contain full functional behaviour modelling.

## 4.2 A First Version of the Skill

The skill is used with an LLM agent. It requires a source file for an actual DLL implementation and at least one target property that will be verified for each list operation.

Structure. The skill is organised into the following components:

Core Definitions The main file and the entry point of specification generation, SKILL.md, acts as the orchestrator of the generation’s behaviour. It formalises the workflow of the skill, including required inputs, mandatory constraints, and non-negotiable verification rules.

Property Module This module contains the descriptions of proof logic for common properties of DLL. At this moment, it defines the correctness property for reachability-based chain validity, along with proof obligations and failure modes.

Reference Library The Knowledge Base refers to multiple aspects needed for the demonstrations and generation: the semi-formal abstract description of the DLL, the formal syntax of Verus constructs, spec and proof patterns, a list of common pitfalls and remediation steps, a decision framework for interpreting failed proofs (weak spec, mismatch, bug), and a formalisation of when preconditions permit specification simplification, usually when preconditions cancel out some parts of the executable code.

Templates The templates standardize the structure of function contracts, abstract structure specifications, proof skeleton for target properties, how to construct bug reports that are complete, actionable, and comparable across cases..

Examples This module is used for showcasing how to tackle situations that might arise, such as demonstrating permissible structural simplification without loss of semantic fidelity or correct classification and reporting of implementation defects.

Procedural Orchestration The heart of the skill is the pipeline that consumes the above assets. The skill uses it as the workflow, and it is detailed below.

Workflow. The following main steps are intended to implement the aforementioned design principles for addressing the verification challenges:

1. Analyse the inputs: In this step, the agent builds a mental model of the implementation and abstract specification. The main goal of this step is to identify functions, data types, integer types, recursions, loops, and the implementations of the functions found in the abstract specification: first, last, next, prec, value, and to decide whether the code is an axiom boundary. The output is a structured summary of the implementation referring to functions, types, preconditions, postconditions, invariants, and target properties.

2. Map abstract specification to Verus: This step translates the abstract specification into Verus constructs. Any semi-formal specification is mapped to a proper Verus statement, such as requires, ensures, spec fn (marking a function as specification only), forall, etc. This step is also responsible for defining the storage-agnostic DLL model: how node identification should be defined in Verus, how node membership is modeled, how first/last nodes are structured.

3. Generate the specification: Starting from the mental model and the mapping of the abstract specification to Verus syntax and structure, a first specification is generated. This step has the most workload. If the first step classified the implementation as axiom-boundary, a separate strategy is defined: executable functions are marked with external\_body, code bodies should not be proved, and the focus is on defining precise preconditions and postconditions in the requires and ensures blocks and on consistency lemmas. Otherwise, the flow is to generate the requires and ensures blocks and define the spec fn helpers. Several templates are provided in this step for function definition and structure specification. An optional inter-step is to simplify unreachable branches under strong preconditions. The agent is guided to encode precise link updates with frame clauses. For properties for which we expect a specific proof, such as the validity of the chain, the skill guides the agent to generate a specification using the already defined property description.

4. Add invariants: For every loop or recursion, invariants and a decreases clause must be provided. For traversal loops, reachability, visited-set, and progress conditions should be included. This is mandatory whenever loops or recursion appear and functions primarily as a verification checkpoint.

5. Verify property: Another verification step is to check that the target property is ensured through the postconditions and the actual function code. The proof or assert blocks express the property at the end of a function if the operation is complex. The skill guides the agent to use intermediate proofs or temporary assume statements. The main goal is to ensure each executable function’s ensures are discharged by the actual code (no modelling-only specification).

6. Debug failures: This step classifies the failure by error message and applies targeted fixes, which might be strengthening invariants, adding bounds to prevent overflow, splitting postconditions, or adding intermediate lemmas. It also addresses common pitfalls in Verus specifications and syntax, where straightforward fixes are provided.

7. Report discrepancies: If the failure is a real bug or spec mismatch, the agent must produce a structured discrepancy report with a concrete counterexample, the violated abstract requirement, and a suggested fix. The skill strengthens the rule to never weaken the specification to hide an implementation defect.

How the Challenges are Addressed by the Skill. Some challenges presented in Section 3.4 are addressed directly through a concrete skill specification, whereas others are addressed by the skill as a whole.

The self-referential nature of DLLs in formal verification implies rebuilding the ghost infrastructure for each implementation. Rather than proving the list’s internal structure, the skill guides the agent toward a storage-agnostic representation of the list. The node identification and membership are abstracted away from the storage mechanism, so the proof focuses on the global DLL properties. This specification also addresses the memory model diversity across eleven implementations.

The traceability principle ties each invariant clause to an abstract requirement to address the inherently global invariants challenge: the global effects of a single link update determine the node’s validity. The principle also prevents arbitrary postconditions, reducing imprecision.

The separation of concerns isolates proof tactics and debugging into distinct phases. The verification and debugging phases address possible proof-block misplacement. The agent is instructed to identify operations and sub-steps that interfere with proof-block placement in a complex demonstration.

## 4.3 Current Results and Resolutions

The current version of the skill is not a final one. To observe the shortcomings of the skill and how they might be addressed, multiple experiments should be conducted and analysed over multiple iterations.

The evaluation of a skill’s results depends on its own design, meaning that the user is responsible for analyzing the different agents’ generated Verus code against the proposed workflow and checking whether the specifications of the skill are satisfied and how.

Table 5 presents some examples of generated specifications with different agents. The index-based implementation is the main use-case as it provided the most insights for the skill construction (I1, I2, I3, I4 rows represent different runs for the index implementation), whereas the Standard DLL (STD row) and SlotMap DLL (SM row) represent more of a testing step. The I2 and I3 experiments refer to the same implementation and use the same agent, but are generated using different prompts. All the experiments are conducted using the prompt /generate-verus-specification for <path to the implementation> targeting the valid chain property. However for the I2 experiment it is also specified that "afull demonstration is wanted".

<table><tr><td></td><td>Acronym Implementation</td><td>Model</td><td></td><td>Assumes Axioms</td><td>Code lines</td><td>Time</td></tr><tr><td>I1</td><td>Index</td><td>Codex 5.2 High</td><td>46</td><td>0</td><td>612</td><td>10 min.</td></tr><tr><td>I2</td><td>Index</td><td>Claude 4.8 Max</td><td>0</td><td>0</td><td>2259</td><td>50 min.</td></tr><tr><td>I3</td><td>Index</td><td>Claude 4.8 Max</td><td>0</td><td>0</td><td>1539</td><td>40 min.</td></tr><tr><td>I4</td><td>Index</td><td>Claude 4.6 High</td><td>18</td><td>1</td><td>1122</td><td>20 min.</td></tr><tr><td>STD</td><td>Standard</td><td>Claude 4.6 High</td><td>0</td><td>8</td><td>485</td><td>10 min.</td></tr><tr><td>SM</td><td>SlotMap</td><td>Claude 4.6 High</td><td>0</td><td>8</td><td>762</td><td>10 min.</td></tr></table>

Table 5: Generated Demonstration

## 4.4 Specifications’ Evaluation Criteria

The skill is designed so that it guides the verification-code generation rather than constrains the proof methods. It gives steps to follow and does not enforce all the details of the verification, as there would be no reason to use an agent for this task.

The following criteria are used to capture the choices each agent made following the same workflow, and the resolutions for each can be observed in Table 6.

1. Soundness boundary: where verification stops and trust begins (explicit assume and external\_body).

• None: no assume or external\_body; code is fully proved.

• Narrow: a small isolated trusted helper; core mutators proved.

• Moderate: mixed boundary; some mutators rely on assumptions.

• Full/opaque: most or all mutators are external\_body.

2. Spec alignment: how much of the chain validity demonstration is followed.

• Direct: valid\_chain states reachability from first and to last.

• Witness-based: valid\_chain uses a ghost order/chain, and reachability is derived as a lemma.

• Traversal-derived: valid\_chain defined via chain\_seq/traverse coverage, reachability is a corollary.

• Axiomatic: reachability is assumed via postconditions, not computed.

3. Order precision: how exactly postconditions pin down relative order changes.

• High: exact sequence equations (append/prepend/remove).

• Medium: order witness exists but not all ops specify positions.

• Low: only membership/endpoint facts (e.g., node set + head and tail nodes).

4. Generality: how restrictive the spec is in type bounds or modelling choices.

• Unconstrained: no T bounds; minimal extra state.

• Type-bounded: requires T: Copy.

• Ghost-heavy: adds ghost fields like order or links.

• Representation-bounded: relies on sentinel/capacity assumptions.

5. Allocator modelling: how allocation and reuse correctness is specified.

• Invariant-level: free\_list\_wf (or equivalent) in the global invariant.

• Per-operation: each mutator assumes allocator well-formedness.

• Implicit: allocator behaviour not modeled.

• Opaque/freshness axioms: uniqueness/freshness assumed for an opaque store.

6. Proof ergonomics: where proof effort lives (lemma toolkit vs ghost instrumentation).

• Lemma-heavy: many inductive lemmas, minimal ghost state.

• Ghost-heavy: extra ghost state simplifies proofs and reduces lemmas.

• Balanced: moderate lemmas and moderate ghost state.

• Axiom-driven: few lemmas; behaviour mostly encoded in postconditions.

<table><tr><td rowspan=1 colspan=2>1. Index with Codex 5.2 High</td></tr><tr><td rowspan=1 colspan=1>Soundness boundary</td><td rowspan=1 colspan=1>Moderate</td></tr><tr><td rowspan=1 colspan=1>Spec alignment</td><td rowspan=1 colspan=1>Direct</td></tr><tr><td rowspan=1 colspan=1>Order precision</td><td rowspan=1 colspan=1>Low</td></tr><tr><td rowspan=1 colspan=1>Generality</td><td rowspan=1 colspan=1>Type-bounded(T: Copy)</td></tr><tr><td rowspan=1 colspan=1>Allocator modelling</td><td rowspan=1 colspan=1>Implicit</td></tr><tr><td rowspan=1 colspan=1>Proof ergonomics</td><td rowspan=1 colspan=1>Axiom-driven</td></tr></table>

<table><tr><td rowspan=1 colspan=2>2. Index with Claude 4.8 Max</td></tr><tr><td rowspan=1 colspan=1>Soundness boundary</td><td rowspan=1 colspan=1>None</td></tr><tr><td rowspan=1 colspan=1>Spec alignment</td><td rowspan=1 colspan=1>Direct</td></tr><tr><td rowspan=1 colspan=1>Order precision</td><td rowspan=1 colspan=1>Low</td></tr><tr><td rowspan=1 colspan=1>Generality</td><td rowspan=1 colspan=1>Unconstrained</td></tr><tr><td rowspan=1 colspan=1>Allocator modelling</td><td rowspan=1 colspan=1>Invariant-level</td></tr><tr><td rowspan=1 colspan=1>Proof ergonomics</td><td rowspan=1 colspan=1>Lemma-heavy</td></tr></table>

<table><tr><td rowspan=1 colspan=2>3. Index with Claude 4.8 Max</td></tr><tr><td rowspan=1 colspan=1>Soundness boundary</td><td rowspan=1 colspan=1>None</td></tr><tr><td rowspan=1 colspan=1>Spec alignment</td><td rowspan=1 colspan=1>Witness-based</td></tr><tr><td rowspan=1 colspan=1>Order precision</td><td rowspan=1 colspan=1>High</td></tr><tr><td rowspan=1 colspan=1>Generality</td><td rowspan=1 colspan=1>Ghost-heavy</td></tr><tr><td rowspan=1 colspan=1>Allocator modelling</td><td rowspan=1 colspan=1>Invariant-level</td></tr><tr><td rowspan=1 colspan=1>Proof ergonomics</td><td rowspan=1 colspan=1>Ghost-heavy</td></tr></table>

<table><tr><td rowspan=1 colspan=2>4. Index with Claude 4.6 High</td></tr><tr><td rowspan=1 colspan=1>Soundness boundary</td><td rowspan=1 colspan=1>Moderate</td></tr><tr><td rowspan=1 colspan=1>Spec alignment</td><td rowspan=1 colspan=1>Traversal-derived</td></tr><tr><td rowspan=1 colspan=1>Order precision</td><td rowspan=1 colspan=1>High</td></tr><tr><td rowspan=1 colspan=1>Generality</td><td rowspan=1 colspan=1>Unconstrained</td></tr><tr><td rowspan=1 colspan=1>Allocator modelling</td><td rowspan=1 colspan=1>Per-operation</td></tr><tr><td rowspan=1 colspan=1>Proof ergonomics</td><td rowspan=1 colspan=1>Lemma-heavy</td></tr></table>

<table><tr><td rowspan=1 colspan=2>5. Standard DLL with Claude 4.6 High</td></tr><tr><td rowspan=1 colspan=1>Soundness boundary</td><td rowspan=1 colspan=1>Full/opaque</td></tr><tr><td rowspan=1 colspan=1>Spec alignment</td><td rowspan=1 colspan=1>Witness-based</td></tr><tr><td rowspan=1 colspan=1>Order precision</td><td rowspan=1 colspan=1>High</td></tr><tr><td rowspan=1 colspan=1>Generality</td><td rowspan=1 colspan=1>Ghost-heavy</td></tr><tr><td rowspan=1 colspan=1>Allocator modelling</td><td rowspan=1 colspan=1>Opaque/freshness</td></tr><tr><td rowspan=1 colspan=1>Proof ergonomics</td><td rowspan=1 colspan=1>Axiom-driven</td></tr></table>

<table><tr><td rowspan=1 colspan=2>6. SlotMap with Claude 4.6 High</td></tr><tr><td rowspan=1 colspan=1>Soundness boundary</td><td rowspan=1 colspan=1>Full/opaque</td></tr><tr><td rowspan=1 colspan=1>Spec alignment</td><td rowspan=1 colspan=1>Witness-based</td></tr><tr><td rowspan=1 colspan=1>Order precision</td><td rowspan=1 colspan=1>High</td></tr><tr><td rowspan=1 colspan=1>Generality</td><td rowspan=1 colspan=1>Ghost-heavy</td></tr><tr><td rowspan=1 colspan=1>Allocator modelling</td><td rowspan=1 colspan=1>Opaque/freshness</td></tr><tr><td rowspan=1 colspan=1>Proof ergonomics</td><td rowspan=1 colspan=1>Ghost-heavy</td></tr></table>

Table 6: Per-specification Evaluation Criteria

## 4.5 Shortcomings Yet to Be Addressed

Each experiment result in Table 6 has its strengths and shortcomings. It is important to observe them as they guide the development of the skill and the fine-tuning of the specification.

The first two specifications adopt a reachability-based definition of chain validity. I1 relies on numerous explicit assume statements, which weakens the verification soundness. I2 uses a comprehensive lemma toolkit, including fuel monotonicity, frame conditions, and reachability chaining, and it addresses allocator correctness through a free\_list\_wf specification included in the state invariant. The specification lacks a positional witness for ordering, which limits the ability to state or prove fine-grained order-preservation postconditions.

I3 presents a different approach from the other three that model the same implementation. This approach strengthens the invariant by introducing a ghost order sequence and proving that the invariant implies reachability. It thus achieves full verification while enabling high-precision ordering postconditions. However, a ghost-heavy specification is less faithful to the concrete implementation.

A less satisfying result is I4: link—one of the most used functions, responsible for updating the two adjacent node identifiers between nodes, so in short, creating the list structure—is treated as an external\_body. The proof is incomplete in the most mutation-critical parts.

For the Standard implementation, STD uses a ghost chain of heap addresses and axiomatizes all pointer-touching operations, proving only consistency lemmas and simple accessors. The specification is precise about chain shape but intentionally treats the concrete data structure as opaque. As the actual list implementation is not our creation, the verification is entirely axiomatic with respect to the concrete code and it cannot expose implementation bugs or justify allocator freshness beyond trust assumptions.

The SlotMap verification is a similar case. The implementation code is not responsible for the memory management of the list, but rather of the list abstraction over a map structure. The specification offers a stronger abstract model than the pointer case despite the opaque SlotMap backend, enabling bidirectional link-consistency lemmas while retaining a chain witness.

After seeing these results, the skill might need:

• Reduced assumptions footprint by systematic lemma coverage: Even if the skill instructs the agent that the number of assume statements should be minimal, an intermediate step that instructs the agent to generate the missing lemmas that replace assume stubs should be added.

• A post-mutation proof staging pattern: Many assume uses arise because facts established before a mutation are not re-established after the mutation. The skill needs a disciplined staging pattern, adding ghost snapshots after each state change and local lemmas to re-assert invariants.

• Ordering precision: The skill should either add a ghost order witness or enrich postconditions with sequence equations, as ordering is, alongside reachability, one of the most useful properties to establish for chain validity.

## 5 Conclusions

In this paper, we investigated whether an LLM agent can efficiently verify a family of structurally related Rust implementations against a shared abstract specification, and whether that process can be codified into a reusable methodology. Our progressive case study shows that a key difficulty of verifying Rust programs in Verus lies in specification quality: weakened specifications are masked by assume statements and by axiomatic lemmas built on unverified (trusted) functions that nonetheless report as verified. The proposed skill addresses some challenges leading to this situation. The main contribution is the skill and its interaction with different agents. By packaging domain knowledge and a soundness-first workflow into a reusable skill, a general-purpose agent is turned into a DLL-verification specialist, offering a practical path toward LLM-assisted verification whose trusted base is explicit and minimal. The skill is a first version, and its limitations point to future work: systematic lemma coverage to replace assume stubs, a post-mutation proof staging pattern, and stronger ordering witnesses. Our longer-term goal remains verifying all eleven implementations against the single shared specification. This analysis needs to work with different agents and not constrain a single way of proving some established properties. The analysis becomes extensive, but would represent an actual evaluation of the skill and capabilities of a tool such as Verus.

Acknowledgments The authors warmly thank the anonymous reviewers for many useful suggestions for improvements.

## References

[1] Andrada-Livia Antoneac, Dorel Lucanu, Dragos Teodor Gavrilut & Andrei Damian (2026): Design Trade offs in Rust Self-Referential Data Structures. Submitted. The associated artefacts can be found at https: //github.com/xTachyon/paper\_doubly\_linked\_lists.

[2] Vytautas Astrauskas, Aurel Bíly, Jonáš Fiala, Zachary Grannan, Christoph Matheja, Peter Müller, Federico\` Poli & Alexander J Summers (2022): The prusti project: Formal verification for rust. In: NASA Formal Methods - 14th International Symposium, NFM 2022, Pasadena, CA, USA, May 24-27, 2022, Proceedings, Lecture Notes in Computer Science 13260, Springer, pp. 88–108, doi:10.1007/978-3-031-06773-0\_5.

[3] Tianyu Chen, Shuai Lu, Shan Lu, Yeyun Gong, Chenyuan Yang, Xuheng Li, Md Rakib Hossain Misu, Hao Yu, Nan Duan, Peng Cheng, Fan Yang, Shuvendu K. Lahiri, Tao Xie & Lidong Zhou (2024): Automated Proof Generation for Rust Code via Self-Evolution. abs/2410.15756, doi:10.48550/ARXIV.2410.15756. arXiv:2410.15756.

[4] Stefan Zetzsche Debangshu Banerjee, Olivier Bouissou (2026): DafnyPro: LLM-Assisted Automated Verification for Dafny Programs. Presented at the Dafny workshop, POPL 2026. Available at https: //popl26.sigplan.org/details/dafny-2026-papers/12/.

[5] Xavier Denis, Jacques-Henri Jourdan & Claude Marché (2022): Creusot: A Foundryfor the Deductive Verifi cation of Rust Programs. In Adrián Riesco & Min Zhang, editors: Formal Methods and Software Engineering - 23rd International Conference on Formal Engineering Methods, ICFEM 2022, Madrid, Spain, October 24- 27, 2022, Proceedings, Lecture Notes in Computer Science 13478, Springer, pp. 90–105, doi:10.1007/978- 3-031-17244-1\_6.

[6] Son Ho & Jonathan Protzenko (2022): Aeneas: Rust verification by functional translation. Proceedings of the ACM on Programming Languages 6(ICFP), pp. 711–741, doi:10.1145/3547647.

[7] Ranjit Jhala & Rupak Majumdar (2009): Software model checking. ACM Comput. Surv. 41(4), doi:10.1145/1592434.1592438.

[8] Ralf Jung, Jacques-Henri Jourdan, Robbert Krebbers & Derek Dreyer (2017): RustBelt: securing the foundations of the Rust programming language 2(POPL). doi:10.1145/3158154.

[9] Daniel Kroening & Michael Tautschnig (2014): CBMC – C Bounded Model Checker. In Erika Ábrahám & Klaus Havelund, editors: Tools and Algorithms for the Construction and Analysis of Systems, Springer Berlin Heidelberg, Berlin, Heidelberg, pp. 389–391, doi:10.1007/978-3-642-54862-8\_26.

[10] Andrea Lattuada, Travis Hance, Chanhee Cho, Matthias Brun, Isitha Subasinghe, Yi Zhou, Jon Howell, Bryan Parno & Chris Hawblitzel (2023): Verus: Verifying Rust Programs Using Linear Ghost Types. In: Proceedings ofthe ACM Symposium on Operating Systems Principles (SOSP), doi:10.1145/3600006.3613172.

[11] Md Rakib Hossain Misu, Cristina V. Lopes, Iris Ma & James Noble (2024): Towards AI-Assisted Synthesis of Verified Dafny Methods. In: Proceedings ofthe ACM International Conference on the Foundations ofSoftware Engineering (FSE), doi:10.48550/arXiv.2402.00247. Available at https://arxiv.org/abs/2402. 00247. ArXiv:2402.00247.

[12] Chuyue Sun, Yican Sun, Daneshvar Amrollahi, Ethan Zhang, Shuvendu Lahiri, Shan Lu, David Dill & Clark Barrett (2026): VeriStruct: AI-assisted Automated Verification ofData-Structure Modules in Verus. In Sebastian Junges & Guy Katz, editors: Tools and Algorithms for the Construction and Analysis ofSystems, Springer Nature Switzerland, Cham, pp. 109–128, doi:10.1007/978-3-032-22749-2\_6.

[13] Alexa VanHattum, Daniel Schwartz-Narbonne, Nathan Chong & Adrian Sampson (2022): Verifying dynamic trait objects in rust. ICSE-SEIP ’22, Association for Computing Machinery, New York, NY, USA, p. 321–330, doi:10.1145/3510457.3513031.

[14] Chenyuan Yang, Natalie Neamtu, Chris Hawblitzel, Jacob R. Lorch & Shan Lu (2025): VeruSAGE: A Study of Agent-Based Verification for Rust Systems, doi:10.48550/ARXIV.2512.18436. arXiv:2512.18436.