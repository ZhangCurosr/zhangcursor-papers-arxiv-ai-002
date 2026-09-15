# When Tool Calls Succeed but Workflows Fail: Anomalies at the Agent–Tool Boundary

Artem Trofimov   
AVIV Group   
Berlin, Germany   
atrofimov@acm.org   
Boris Novikov   
Independent Researcher   
Helsinki, Finland   
borisnov@acm.org

## Abstract

AI agents increasingly execute long-running workflows that externalize effects through independently supplied tools. Under retries, speculative execution, concurrency, and partial failures, the resulting external state may be inconsistent with the workflow’s intended resolution: required effects may be missing or duplicated, aborted effects may survive, and committed effects may depend on provisional state that is later withdrawn. Advanced transaction models address related failures, but assume that lower-level operations expose the semantics they depend on: whether an effect occurred, whether it can be compensated, staged, or safely reordered. Shared agent–tool interfaces usually do not.

We contribute an effect-history model that separates events in the external world from the runtime’s observations of them, and a catalog of eight recurring externaleffect anomalies. From the catalog we derive the boundary capabilities required to exclude each anomaly in general, and four points where black-box tool invocation alone cannot provide a general guarantee. We then ask how much of this is expressible in a widely used shared tool interface, measuring the use of the standard annotation vocabulary across 98,291 tools exposed by registered Model Context Protocol (MCP) servers. The fields are widely emitted but provide only coarse call-level hints, and none of the required capabilities is fully expressible. These results motivate reusable transactional contracts at the tool boundary.

## 1 Introduction

AI agents increasingly execute multi-step workflows whose operations change the external world: they send messages, charge credit cards, place orders, and invoke independently operated services. The correctness question we study is not whether each tool call succeeds locally, but whether the external effects that survive are consistent with a coherent resolution of the workflow. Under retries, speculative execution, concurrency, and partial failures, they need not be: one logical action may occur twice, a workflow may commit without a required effect, an aborted branch may leave residue, or a committed action may depend on an effect that is later withdrawn.

This is a verification problem before it is a recovery problem: whether an agent’s execution can be checked at all (by the runtime itself, a supervising agent, or an external auditor) depends on what the tool boundary makes observable. A verifier cannot establish that an effect happened exactly once, survived an abort, or leaked to an outside observer if the interface exposes no evidence either way.

These failures constitute a single consistency problem at the agent–tool boundary. External effects generally cannot be rolled back, may become visible before the workflow resolves, and may trigger reactions outside the runtime’s control. At the same time, a runtime invoking a third-party tool may receive only partial evidence of whether an effect occurred — after a crash, a timeout, or a lost acknowledgment. Correctness therefore depends not only on orchestration logic, but also on semantics supplied by the lower-level operations: outcome resolution, idempotence, compensation, staging, dependency identity, commutativity, and visibility control.

Consider an agent booking a trip on a 1500 EUR budget. To reduce latency, it books a 900 EUR flight and a 700 EUR hotel in parallel. Both calls succeed, but their combined effects violate the workflow invariant. The workflow aborts and cancels the hotel; the non-refundable flight survives. Each operation completed correctly in isolation, yet the surviving external state is inconsistent with the intended workflow outcome.

We argue that such failures form a common catalog with three sources, in the anomaly-based style of classical isolation levels [1, 2]: uncertainty about whether an effect occurred, workflow structure that releases or depends on effects before resolution, and interaction with concurrent executions or external observers. The catalog is not an open-ended list: each family is enumerated by construction. The uncertainty anomalies (A1–A3) are fixed by the three safety-relevant actions a runtime can take under an unresolved outcome — re-issue, commit, or compensate; the workflow anomalies (A4–A6) by the three ways an externalized effect can relate to its resolution point — survival, timing, and dependency; the interaction anomalies (A7–A8) by the two kinds of external participant an effect can meet — a concurrent execution or an exogenous observer. We define correctness by excluding anomalous patterns from an effect history — in the spirit of opacity [3], but for effects that cannot be rolled back: a safe execution is one whose surviving external effects remain consistent with the workflow’s resolution.

The mechanisms agent runtimes use (replay, retry, compensation, staged release, coordinated access) are not new: they come from advanced transaction models such as sagas, and from semantic recovery and escrow-style concurrency control [4, 5, 6, 7, 8]. What changes in the agent setting is the boundary, where runtimes often compose independently supplied tools through shared interfaces that expose little of the transactional semantics those mechanisms require. Each runtime therefore recovers the properties it needs locally (adapter annotations, registered inverses, per-effect tiers), out of band. Better replay and recovery do not close the gap on their own: the missing piece is a contract layer at the tool interface.

This paper makes four contributions. First, an effect-history model for agent workflows whose operations are calls to externally supplied tools (Section 2). Second, a catalog of eight external-effect anomalies, organized by the three ways transactional control is lost, with guarantee profiles naming their exclusion sets and a reading of current agent runtimes as partial coverage of the catalog (Section 3). Third, a contract-level account of how the required boundary capabilities can be exposed, together with four limits where black-box invocation alone is insufficient (Section 4). Fourth, a registry-wide census of current MCP tool annotations, showing that none of the required capabilities is fully expressible (Section 5).

## 2 Effect Histories and the Tool Boundary

Agent systems form a hierarchy of levels, as in multilevel transaction management, where the guarantees of one level are derived from the properties of the operations below [9]. We work with a slice of two adjacent levels $L _ { i }$ and $L _ { i + 1 }$ , written L0 and L1. L0 holds the operations the agent issues as atomic calls — typically tools behind external APIs, charge() or book\_flight(); L1 holds the workflows composed from them, BookTrip or HireCandidate. A workflow can itself serve as an operation one level up. A history spans the slice: it records L0 attempts and external effects, the runtime’s partial observations of them, and L1 workflow resolutions. Each attempt and compensat ing action, and every effect they externalize, is attributed to the workflow and branch that issued it. Effect safety is evaluated at L1, against those resolutions.

Table 1: Effect-history vocabulary. The upper group records execution and external events; the lower, runtime observations, workflow assertions, and derived predicates.
<table><tr><td>attempt(a, l)</td><td>attempt a of logical operation l</td></tr><tr><td>externalize(q, e)</td><td>runtime action q produced external effect e</td></tr><tr><td> $e x t e r n a l i z e _ { e x t } ( p , x )$ </td><td>external party p produced x</td></tr><tr><td>cmp(c, a)</td><td>compensating action c for attempt a</td></tr><tr><td>neutralizes(c, e)</td><td>c neutralized effect e</td></tr><tr><td> $d e p ( e _ { 2 } \gets e _ { 1 } )$ </td><td>e2 produced having observed e1</td></tr><tr><td>commute(e1, e2)</td><td>effects commute on their shared resource</td></tr><tr><td>commit(w), abort(w)</td><td>workflow w is resolved</td></tr><tr><td>observe(a, s)</td><td>outcome for a: confirmed, failed, unknown</td></tr><tr><td>Req(w)</td><td>operations that commit(w) asserts took effect</td></tr><tr><td>resolved(w)</td><td>w has committed or aborted</td></tr><tr><td>survives(e)</td><td>e externalized, never neutralized</td></tr></table>

The slice is relative downwards too: book\_flight is atomic only from the caller’s side, while for its provider it is a workflow over operations the caller does not see, of unknown depth. When that hidden structure shows through, as when a timeout occurs inside the provider’s workflow, the caller is left with an outcome it cannot confirm.

The boundary also varies in who owns it. When agent and tools are built together, the properties of L0 operations are declared by construction, and a runtime that owns its adapter layer is in a similar position: it can attach the properties it needs tool by tool. Neither option transfers cleanly across a shared interface, where a client sees only what the interface carries. The Model Context Protocol (MCP) is a widely used interface of this kind and is publicly inspectable.

We focus on five dimensions of L0 operations: idempotence, invertibility, externalization timing and control, determinism, and commutativity. They are not reducible to a single reversible/irreversible scale — increment commutes but is not idempotent, delete is idempotent but does not commute. Commutativity is moreover relational, a property of operation pairs, and often conditional on state (escrow-style [8]); determinism, not usually treated as an operation contract, matters because blackbox LLM-backed tools can be nondeterministic.

An execution history separates execution and external events from the runtime’s knowledge of their outcomes (Table 1). A retry is a second attempt of the same logical operation; a compensating action externalizes in its own right; an exogenous effect is produced by an external party rather than directly by a runtime action; and unknown means no authoritative outcome is available to the runtime, although the outcome is determined in the world. We write externalize(q, e on r) when the resource matters. Where the individual attempt is not at issue we write effect(ℓ) and cmp(ℓ).

Unlike read-write histories [2], effect histories keep events and observations apart: an effect either occurred or did not, while unknown is a state of the runtime, and acting under it is what several anomalies turn on. Many of the anomalies become possible because effects may already be irreversible or externally visible before the workflow resolves.

## 3 Effect Anomalies

Each anomaly exposes something missing at the agent–tool boundary: evidence (A1–A3), lifecy cle control (A4–A6), or coordination (A7–A8). The uncertainty anomalies arise when the runtime cannot confirm the outcome of an effect and must act anyway. The workflow anomalies arise from the structure of a single execution: survival under abort, timing relative to its resolution, and dependency. The interaction anomalies arise when an effect meets something outside its own execution: another execution writing the same resource (A7), or an external actor that observes and reacts to it (A8), which need not be human — it may itself be an agent workflow at a level our boundary does not see.

Table 2: The catalog of external-effect anomalies.
<table><tr><td></td><td>Forbidden pattern</td><td>Required boundary capa- bility</td><td>Representative tion</td><td>realiza-</td></tr><tr><td>A1 Duplicated</td><td>one logical operation exter- nalizes twice</td><td>authoritative convergence on one outcome</td><td>logical-operation ID; idem- potent re-issue returning the original outcome</td><td></td></tr><tr><td>A2 Missing</td><td>commit without a required effect</td><td>authoritative outcome; atomic multi-effect partici- pation</td><td>status endpoint; pare/commit</td><td>pre-</td></tr><tr><td>A3 Orphaned</td><td>compensation issued under an unknown outcome</td><td>outcome resolution before compensating</td><td>outcome query; outcome- conditioned compensation</td><td></td></tr><tr><td>A4 Residue</td><td>aborted workflow leaves a surviving effect</td><td>residue prevention or safe neutralization</td><td>compensation declaration; staging</td><td></td></tr><tr><td>A5 Premature</td><td>possibly non-surviving ef- fect externalizes before res- olution</td><td>pre-externalization obser- vation or control</td><td>quote/dry_run; with expiry</td><td>hold</td></tr><tr><td>A6 Contaminated</td><td>committed effect depends on a non-surviving effect</td><td>dependency observability and commit control</td><td>stable effect/resource iden- tity; dependency tracking; commit gating</td><td></td></tr><tr><td>A7 Conflicting</td><td>unordered non-commuting effects from independent executions</td><td>shared-resource coordina- tion</td><td>resource scope; commuta- tivity declaration; mediator</td><td></td></tr><tr><td>A8 Phantom</td><td>exogenous consequence of a compensated effect sur- vives</td><td>control of external observ- ability</td><td>or ordered release visibility control; mediated observation</td><td></td></tr></table>

Table 2 states the eight patterns together with the boundary capability needed to exclude each one in general. We derive the middle column by asking what information or control must be available at the boundary for a runtime protocol to rule out the anomalous history. If the runtime cannot distinguish the anomalous history from a safe one, the boundary must provide additional evidence. If the anomaly remains reachable even once the relevant facts are known, exclusion additionally requires lifecycle control or coordination. Boundary capabilities do not themselves exclude anomalies; protocols consume them through safe retry, commit gating, compensation, dependency tracking, and mediated ordering. The final column gives representative ways to realize these capabilities rather than unique or minimal implementations: some recur in the systems of Section 3.2, while others are standard transaction-processing mechanisms or boundary primitives those systems do not expose. Section 4 organizes them into reusable contract families.

Not all of the patterns are final-state violations: A3 is defined at action time, and A5 and A7 are preventive, relative to profiles that forbid relying on outcomes the boundary does not establish.

A1: Duplicated Effect (uncertainty) Two attempts of the same logical operation both externalize. The outcome of a call is not confirmed, so the runtime re-issues it, and both attempts take effect:

```perl
attempt(a1, pay_invoice)
externalize(a1, e1) # payment goes through
observe(a1, unknown) # no acknowledgment
attempt(a2, pay_invoice) # re-issued
externalize(a2, e2)
=> invoice paid twice
```

Non-determinism makes this worse: re-issuing an operation may produce a different effect rather than a duplicate. An idempotent re-issue prevents repeated externalization. Determinism does not by itself make re-issue safe; it only makes semantic replay comparison meaningful by making repeated executions predictable.

A2: Missing Committed Effect (uncertainty) A workflow commits relying on an effect that did not externalize. The canonical case is uncertainty resolved optimistically: the runtime assumes an unconfirmed effect succeeded and commits, though nothing externalized. With pay\_invoice ∈ Req(Order):

```julia
attempt(a1, pay_invoice)
observe(a1, unknown) # outcome not confirmed
commit(Order) # assumes success
=> required effect absent, yet committed
```

The anomaly is narrow: it is not that the agent omitted a step (a planning failure) but that a required effect is absent from the committed history. We focus on the uncertainty-induced case: if failure is authoritatively known and the workflow commits anyway, the same final-state pattern is a protocol error rather than a boundary limitation.

A3: Orphaned Compensation (uncertainty) A compensation is issued for an attempt whose outcome is unknown. The unconfirmed outcome is resolved pessimistically instead:

```perl
attempt(a1, pay_invoice)
observe(a1, unknown) # outcome not confirmed
cmp(c1, a1) # compensate blindly
externalize(c1, refund) # refund takes effect
=> if a1 never externalized, refund is spurious
```

A1, A2, and A3 are one family: all three stem from an outcome the runtime cannot confirm and differ only in the reaction. A1 and A2 are visible in the final history; A3 is a violation at the moment compensation is issued: under an unknown outcome the runtime cannot know whether compensation is required, and even where the resulting state happens to be correct, the action was taken without authoritative evidence. The family also marks where this setting departs from classical transaction processing. There, an in-doubt outcome is temporary: participants run a recovery protocol and resolve it. A third-party tool runs no such protocol, so an unknown outcome stays unknown until the tool itself offers a way out — a durable acknowledgment, a status endpoint, or idempotent re-issue.

A4: Uncompensated Residue (workflow) An aborted workflow leaves a surviving effect that was not successfully neutralized — no compensation exists, it was not invoked, or it was incomplete. A non-refundable ticket is bought, then the trip is aborted:

```julia
attempt(a1, book_flight)
externalize(a1, e1) # non-refundable ticket
abort(BookTrip) # budget violated
cmp(c1, a1) # cancellation attempted
NOT neutralizes(c1, e1)
=> survives(e1); money lost
```

A5: Premature Externalization (workflow) An effect that may not survive is externalized before its workflow resolves. An offer letter is sent before approval completes, and approval is later denied:

effect(send\_offer) # letter goes out   
NOT resolved(Hire) # approval pending   
abort(Hire) # approval denied   
=> send\_offer preceded resolved(Hire)

A5 is profile-relative: it is not a violation under compensation-safe execution, which may externalize an effect early and compensate later; it is the pattern that speculation-safe execution excludes by staging or gating effects that may not survive.

A6: Contaminated Speculation (workflow) A committed effect causally depends on an effect from a branch that does not survive. A committed branch acts on a temporary reservation made by a branch that is later canceled:

effect(reserve\_room R) [b1]   
dep(send\_invites <- reserve\_room R) [b2]   
effect(send\_invites) [b2] # invites name R   
commit(b2) # b2 wins   
abort(b1); cmp(reserve\_room R)   
=> b2 announced R, which no longer survives

The contamination is causal, not about reversibility: this is the effect-world analogue of a dirty read (reading uncommitted state that is later rolled back), except that the “read” is a causal dependency on an external effect. The harm, invitations already sent naming a room that was released, would arise even if those invitations were themselves reversible.

A7: Conflicting Externalization (interaction) Two independent executions release potentially non-commuting effects on a shared resource without an ordering or mediation contract, so the outcome may depend on their interleaving. Two coding workers publish incompatible updates to the same shared artifact from the same base, neither having observed the other:

effect(publish U1 on S) [w1] # from base B   
effect(publish U2 on S) [w2] # also from B   
NOT dep(U2 <- U1); NOT dep(U1 <- U2)   
NOT commute(U1, U2) # order changes result   
=> outcome depends on interleaving

Unlike A6, there is no dependency between the two executions: neither observed the other; both acted on the shared resource independently. Commutativity excludes the anomaly. Invertibility does not exclude it but makes it repairable: a versioned reversible resource lets the loser be rolled back and re-applied toward a serializable state, as in classical concurrency control. Mediation prevents the race by ordering access (Section 4.2). A7 is profile-relative, as A5 is: the resource may serialize the calls on its own, but the externally-mediated profile forbids relying on an outcome not established at the boundary.

A8: Phantom Compensation (interaction, open-world) An effect is compensated, but an external consequence that depends on it survives. A supplier reacts to an offer that is later withdrawn:

```julia
externalize(a1, e) # offer sent
externalize_ext(S, x) # supplier acts on it
dep(x <- e) # x saw e
cmp(c1, a1) # offer withdrawn
neutralizes(c1, e)
survives(x) # supplier action stands
=> offer retracted; the reaction is not
```

The dependent effect x is exogenous: no attempt of the runtime produced it and no compensation reaches it. This separates A8 from A6, where the surviving effect is managed. A1–A7 concern effects issued through the managed or mediated boundary; an exogenous consequence lies outside both the projection the runtime observes and the set of effects it can compensate. The reacting party may also sit outside any contractual boundary, so no ordering or commutativity can be declared for it, and the enforceable primitive is preventive: control whether and when the effect becomes externally observable.

## 3.1 Safety Guarantee Profiles

The catalog induces a vocabulary of effect-safety guarantees, each naming a set of anomalies it excludes: unknown-safe (A1–A3), compensation-safe (A4, assuming compensations are correct and succeed), speculation-safe (A5–A6), and externally-mediated (A7–A8, where the boundary admits mediation).

Interactive tasks explain why compensation-safe is sometimes the best achievable profile: an agent negotiating on a user’s behalf cannot learn the counterparty’s reaction without sending an offer, so the action must precede the observation and full gating is unavailable.

## 3.2 Current Runtimes Against the Catalog

We next use the catalog to identify which of the required capabilities current agent runtimes target or establish under their stated assumptions. Table 3 summarizes their stated coverage rather than proven guarantees; none realizes a clean cumulative hierarchy.

Table 3: Coverage of the anomaly catalog by current agent runtimes.
<table><tr><td>System</td><td>Targeted coverage</td><td>Notes</td></tr><tr><td>ACRFence [10]</td><td>unknown-safe partial (A1)</td><td>proposed; attack validated, not implemented</td></tr><tr><td>RAC [11]</td><td>compensation-safe partial (A4)</td><td>no gating; no idempotency, so A1-A3 not addressed</td></tr><tr><td>Atomix [12]</td><td>A1/A3 partial, compensation-safe (A4), speculation-safe (A5, A6), externally- mediated partial (A7)</td><td>idempotent-known-outcome path; no crash- safe exactly-once; A2 gap on multi-effect release</td></tr><tr><td>Cordon [13]</td><td>A1/A3 partial, compensation-safe (A4), speculation-safe partial (A5)</td><td>idempotent tools; staged effects; task- scoped</td></tr><tr><td>CoAgent [14]</td><td>externally-mediated partial (A7)</td><td>reordering prevents some conflicts; regis- tered inverses enable repair; irreversibles</td></tr><tr><td>Shepherd [15]</td><td>per-effect reversibility tiers; A7 observed in supervisor use case</td><td>gated meta-agent substrate, not a runtime guaran- tee; irreversibles logged</td></tr></table>

Atomix and Cordon stage effects, with Atomix additionally tracking speculative dependencies; their A1/A3 coverage is conditional on assumed tool semantics rather than on a general outcomeresolution protocol. CoAgent’s reordering with registered inverses is an instance of the achievability condition for A7 rather than an unconditional guarantee. Across these systems, coverage rests on runtime-owned declarations (adapter-declared idempotency keys, reversibility annotations, registered inverses, per-effect tiers) rather than a reusable shared contract; none controls exogenous reactions (A8), and no system covers the full catalog. Adjacent work governs the read side of shared agent state [16] and intent revision under conflicting irreversible actions [17], applies Sagastyle validation to multi-agent planning [18], and argues for isolation-level reasoning in workflow systems [19]. Agentic Transaction [20] articulates ACID-oriented semantic guarantees for agent systems and instantiates them in a runtime for data agents; we ask instead what the boundary to third-party tools must declare for guarantees of this kind to be establishable at all. A recent formal catalog of concurrency anomalies for multi-agent runtimes comes with mechanically verified detec tors and safety guarantees [21]; its causal-cascade anomaly, in which a dependent operation survives an aborted ancestor, is close to our A6, and the model likewise admits external effects that no internal rollback can undo. Our catalog differs in what it centers: the outcome uncertainty of calls to independently supplied tools, and the interface capabilities a boundary must expose to exclude each anomaly, rather than verified protocols over a runtime’s shared state.

## 3.3 Scope and Coverage

The catalog is coverage-oriented rather than complete by theorem: absolute completeness is unavailable in an open world, where an external actor can react in unbounded ways. Each family has a structural reason for its membership. The uncertainty family is fixed by the safety-relevant actions a runtime can take while an outcome is unresolved: re-issue, commit as though it had externalized, or compensate. Waiting, querying, or escalating creates no new effect-history pattern until one of these is taken. The workflow family is fixed by how an externalized effect relates to its resolution point — survival, timing, dependency. The interaction family separates unmediated concurrent effects on a shared resource from an exogenous consequence of a later-compensated effect. We conjecture that, relative to the vocabulary of Section 2, every loss of transactional control the model can represent falls into one of the three families; formal coverage, minimality, and independence are future work.

We place the following outside the model, so the catalog is not mistaken for covering them: semantic planning errors (the agent booked the wrong city); read-side anomalies; security and policy violations; tool-contract misclassification (a tool declared reversible that is not); intra-execution effect order (formalized for multi-agent runtimes by [21]), except insofar as it participates in an A7 conflict; and liveness failures.

## 4 Transactional Tool Contracts and Guarantee Boundaries

Advanced transaction models assume properties such as idempotence, invertibility, and commutativity are known; at a third-party boundary they are often absent or inferred. A reusable contract layer must therefore distinguish established, assumed, and unknown properties, since acting on an assumption, such as compensating an operation with no true inverse, is a failure mode of its own. Such declarations do not replace runtime protocols: they are their precondition.

A runtime consuming tools through a shared interface reads only what that interface carries, and MCP is not entirely silent: since its 2025-03-26 revision, tool annotations let a server flag a tool as read-only, destructive, idempotent, or open-world.<sup>1</sup> These are advisory boolean hints — unenforced, optional, and coarser than the capabilities of Table 2: idempotentHint marks a property but supplies no idempotency key, and no hint covers status resolution, compensation, staging, com mutativity, or visibility. How servers populate these fields is measured in Section 5.

## 4.1 Recurring Contract Families

The capabilities in Table 2 fall into three contract families. A capability is what the boundary lets a runtime establish; a contract is its reusable declaration, possibly spanning several operations.

Uncertainty requires authoritative convergence on one logical outcome. Deduplication alone excludes A1 but does not resolve the outcome that A2 and A3 turn on. The adapter-declared idempotency key is the closest primitive in use among the systems of Section 3.2; atomic multi-effect release is deferred to Section 4.2.

Lifecycle control (A4–A6) needs the workflow’s invariants to be checkable before irreversible effects occur, compensation semantics for what a compensator requires and what counts as its success, and stable effect and resource identity afterwards. A read-only quote avoids externalization altogether; an expiring hold replaces the final effect with a provisional one whose visibility, expiry, and release semantics are declared. The unit dependency tracking protects is a commit sphere, the minimal set of effects that must become final together.

Coordination (A7–A8) requires contracts at a different granularity. Idempotency, status, quote, and compensation are properties of a single operation, whereas commutativity holds of a pair of operations on a resource and cannot be set by rules given to one agent: it must be declared at the resource; where it does not hold, exclusion requires a mediator that serializes access, and declared inverses support repair after a conflict. Visibility is likewise a property of the boundary: who may observe an effect before it is final.

A boundary that exposes these capabilities makes the corresponding exclusions implementable by a conforming runtime, given a suitable protocol and declarations that hold, rather than runtime-local out-of-band knowledge.

## 4.2 Guarantee Boundaries Above Black-Box Tools

Four boundaries mark where black-box invocation alone is insufficient: stronger guarantees require authoritative tool participation, shared mediation, or visibility control. We state them as consequences of the definitions and relate them to current systems; formal proofs are future work.

First, without a primitive that establishes one authoritative outcome (idempotent re-issue returning the original outcome, a status endpoint, or a durable acknowledgment), no protocol can both resolve the workflow after an ambiguous tool call and guarantee unknown-safe and compensation-safe execution. Over an unreliable channel the runtime cannot learn in bounded time whether an effect took place, the classical exactly-once barrier [22] at the agent–tool boundary. Re-issuing risks A1, committing risks A2, compensating risks A3, and aborting without compensation risks A4 if the effect occurred. Waiting avoids this resolution-time choice but gives up resolution.

Second, non-commuting irreversible effects without a mediator admit no general conflict repair. Without operation-specific reconciliation, the runtime has no general way to transform the realized state into one corresponding to a serial order; undo and re-application would provide such a path, but irreversibility rules it out. What remains is prevention: mediation before unordered externalization, or gating conflicting calls until earlier ones commit, as CoAgent [14] does.

Third, an open-world reaction ends the reach of compensation: once an outside actor has observed a released effect and acted, retracting the effect does not retract the reaction (A8). The available contract is preventive — visibility control before observation.

Fourth, several irreversible effects on different tools cannot be released atomically above the tool layer. A crash or persistent failure between releases leaves a partial externalization that the runtime cannot generally complete or undo: committing risks a missing required effect (A2), while aborting leaves surviving residue (A4). Avoiding this choice is the classical atomic-commitment problem [22] and needs tool-side prepare/commit participation, a limit Atomix [12] states for its own release step.

## 5 What the Boundary Declares Today

Section 4 named the boundary capabilities a runtime needs. We now ask how far the standard MCP annotation vocabulary expresses them: what the annotations do not carry cannot be obtained from the interface as a guarantee, whatever a tool arranges out of band. Our census covers 98,291 tools in the official MCP registry (snapshot 2026-07-27), recording the four annotations the specification defines (readOnlyHint, destructiveHint, idempotentHint, openWorldHint) and keeping an explicit false distinct from an omitted field.

Sampling and methodology. We took a full snapshot of the official MCP registry on 2026-07-27 (59,625 entries; 18,688 distinct servers, keeping the latest version of each). Of these, 9,454 exposed no remote endpoint and were out of scope because probing them would require executing a package or stdio server. We anonymously queried tools/list on all 9,234 remote targets, including those declaring an authentication header; no responding target rejected discovery with HTTP 401 or 403 (connection failures were not classified). Of these targets, 4,838 returned at least one tool, 4,318 failed to connect, 74 timed out, and four returned no tools, yielding 98,291 tools (median 11 per server). No tool was called. The resulting measurements therefore describe the reachable remote subset; unreachable and package/stdio-only servers may differ. We do not attribute measured values to author intent. Code, data, and full methodology are available in the accompanying artifact.<sup>2</sup>

Annotation fields are widely emitted. At the wire level, 74.0% of tools serialize at least one field and 61.7% serialize all four. But a field being present says little about whether it was chosen: many values may originate in SDK defaults or server templates rather than deliberate declaration, and some carry no information even when set, since destructiveHint is inapplicable when readOnlyHint is true, a pairing that accounts for 52.8% of all tools.

Tool-level signatures are concentrated. Reading each tool as a signature over the four fields (each true, false, or omitted), 66 of the 81 possible signatures appear, but one signature (read-only, non-destructive, idempotent, open-world) covers 39.9% of tools, and the next most common is no annotation at all (26.0%); the top three together cover 75.8%. Among multi-tool servers that emit at least one field, 76.7% use more than one signature. Across emitting servers, the median dominant signature covers 79.4% of a server’s tools: differentiation is widespread but coarse.

What the hints do not resolve. Even fully populated, the four fields describe the shape of a call while leaving its transactional semantics unstated: none gives an idempotency key, a status endpoint, a compensation contract, or a commutativity rule. Take destructiveHint, the one field about risk.

Table 4: Expressibility of the boundary capabilities of Section 4 in current MCP tool annotations.
<table><tr><td>Boundary capability</td><td>In annotations</td></tr><tr><td>A1 authoritative convergence</td><td>Limited: idempotence hint only</td></tr><tr><td>A2 authoritative outcome; atomic participation</td><td>No</td></tr><tr><td>A3 outcome resolution before compensating</td><td>No</td></tr><tr><td>A4 residue prevention or neutralization</td><td>No</td></tr><tr><td>A5 pre-externalization observation/control</td><td>No</td></tr><tr><td>A6 dependency observability; commit control</td><td>No</td></tr><tr><td>A7 shared-resource coordination</td><td>No: externality only</td></tr><tr><td>A8 control of external observability</td><td>No</td></tr></table>

It is set on 65.8% of all tools, but is meaningful only where the tool is not read-only: just 12.9% of tools carry an applicable classification, and only 3.1% assert an actual destructive operation.

Table 4 reads the boundary capabilities of Section 4 against that vocabulary: current MCP metadata distinguishes coarse execution modes but cannot express the semantics needed for retries, compensation, speculation, or concurrency. The census reads the boundary as declared, not as implemen tations behave. Tools may enforce stronger semantics internally (deduplication, idempotency keys) than any annotation surfaces; that safety remains unusable to a runtime that only sees the interface, and measuring the gap requires implementation-level analysis [23].

## 6 Conclusion and Future Work

This paper contributes an effect-history model separating world events from runtime observations, and a catalog of eight recurring external-effect anomalies, each naming the boundary capability required to exclude it in general: evidence, lifecycle control, or coordination. Four guarantee boundaries mark where black-box invocation alone is insufficient. Our census of 98,291 MCP tools mea sures what the standard interface declares. Existing runtimes cover fragments under runtime-local assumptions; standardized tool metadata remains insufficient for transactional reasoning. The missing layer is therefore not another recovery mechanism, but reusable transactional contracts at the tool boundary. Future work includes formal proofs of the guarantee boundaries, protocol synthesis from declared contracts, and a cost model that prices anomalies against their compensations, deciding when speculation or early release is worth its residue.

## References

[1] Hal Berenson, Philip A. Bernstein, Jim Gray, Jim Melton, Elizabeth J. O’Neil, and Patrick E. O’Neil. A critique of ANSI SQL isolation levels. In Proceedings of the 1995 ACM SIGMOD International Conference on Management of Data, pages 1–10. ACM, 1995. doi: 10.1145/ 223784.223785.

[2] Atul Adya. Weak Consistency: A Generalized Theory and Optimistic Implementations for Distributed Transactions. PhD thesis, Massachusetts Institute of Technology, 1999.

[3] Rachid Guerraoui and Michal Kapalka. On the correctness of transactional memory. In Proceedings of the 13th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming, PPoPP ’08, pages 175–184. Association for Computing Machinery, 2008. doi: 10.1145/1345206.1345233.

[4] Hector Garcia-Molina and Kenneth Salem. Sagas. In Proceedings ofthe 1987 ACM SIGMOD International Conference on Management of Data, pages 249–259. ACM, 1987. doi: 10.1145/ 38713.38742.

[5] Henry F. Korth, Eliezer Levy, and Abraham Silberschatz. A formal approach to recovery by compensating transactions. In Proceedings ofthe 16th International Conference on Very Large Data Bases, pages 95–106. Morgan Kaufmann, 1990.

[6] Ahmed K. Elmagarmid, editor. Database Transaction Models for Advanced Applications. Morgan Kaufmann, 1992.

[7] Gerhard Weikum and Gottfried Vossen. Transactional Information Systems: Theory, Algorithms, and the Practice ofConcurrency Control and Recovery. Morgan Kaufmann, 2001.

[8] Patrick E. O’Neil. The escrow transactional method. ACM Transactions on Database Systems, 11(4):405–430, 1986. doi: 10.1145/7239.7265.

[9] Gerhard Weikum. Principles and realization strategies of multilevel transaction management. ACM Transactions on Database Systems, 16(1):132–180, 1991. doi: 10.1145/103140.103145.

[10] Yusheng Zheng, Yiwei Yang, Wei Zhang, and Andi Quinn. ACRFence: Preventing semantic rollback attacks in agent checkpoint-restore. arXiv preprint arXiv:2603.20625, 2026. URL https://arxiv.org/abs/2603.20625.

[11] Srinath Perera, Kaviru Hapuarachchi, Frank Leymann, and Rania Khalaf. Robust agent compensation (RAC): Teaching AI agents to compensate. In Proceedings of the ACM Conference on AI and Agentic Systems, pages 253–262, 2026.

[12] Bardia Mohammadi, Nearchos Potamitis, Lars Klein, Akhil Arora, and Laurent Bindschaedler. Atomix: Timely, transactional tool use for reliable agentic workflows. arXiv preprint arXiv:2602.14849, 2026. URL https://arxiv.org/abs/2602.14849.

[13] Zheng Chen, Hanqing Liu, Duling Xu, Dong Dong, Jialin Li, Bangzheng Pu, and Jidong Zhai. Cordon: Semantic transactions for tool-using LLM agents. arXiv preprint arXiv:2606.17573, 2026. URL https://arxiv.org/abs/2606.17573.

[14] Hongtao Lyu, Dingyan Zhang, Mingyu Wu, Xingda Wei, and Haibo Chen. CoAgent: Concurrency control for multi-agent systems. arXiv preprint arXiv:2606.15376, 2026. URL https://arxiv.org/abs/2606.15376.

[15] Simon Yu, Derek Chong, Ananjan Nandi, Dilara Soylu, Jiuding Sun, Christopher D Manning, and Weiyan Shi. Shepherd: Enabling programmable meta-agents via reversible agentic execution traces. arXiv preprint arXiv:2605.10913, 2026. URL https://arxiv.org/abs/2605. 10913.

[16] Sajjad Khan. S-Bus: Automatic read-set reconstruction for multi-agent LLM state coordination. arXiv preprint arXiv:2605.17076, 2026. URL https://arxiv.org/abs/2605.17076.

[17] Zhiyuan Zhai, Ming Li, and Xin Wang. Revisable by design: A theory of streaming LLM agent execution. arXiv preprint arXiv:2604.23283, 2026. URL https://arxiv.org/abs/ 2604.23283.

[18] Edward Y. Chang and Longling Geng. SagaLLM: Context management, validation, and transaction guarantees for multi-agent LLM planning. Proceedings of the VLDB Endowment, 18 (12):4874–4886, 2025. doi: 10.14778/3750601.3750611.

[19] Michael Stonebraker, Xinjing Zhou, Peter Kraft, and Qian Li. Consistency and correctness in data-oriented workflow systems. In 16th Conference on Innovative Data Systems Research (CIDR), 2026.

[20] Zhaoyan Sun, Xiaoxiao Wang, and Guoliang Li. Agentic transaction: Towards ACIDcompliant agent systems. arXiv preprint arXiv:2608.13900, 2026. URL https://arxiv. org/abs/2608.13900.

[21] Sajjad Khan. Verified detection and prevention of concurrency anomalies in multi-agent large language model systems. arXiv preprint arXiv:2606.17182, 2026. URL https://arxiv. org/abs/2606.17182.

[22] Philip A. Bernstein, Vassos Hadzilacos, and Nathan Goodman. Concurrency Control and Recovery in Database Systems. Addison-Wesley, 1987.

[23] Benny Toeppe, Amine Barrak, and Emna Ksontini. A large-scale dataset of MCP implementations on GitHub. arXiv preprint arXiv:2607.10123, 2026. URL https://arxiv.org/abs/ 2607.10123.