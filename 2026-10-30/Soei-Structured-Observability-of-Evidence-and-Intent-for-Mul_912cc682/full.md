# Soei: Structured Observability of Evidence and Intent for Multi-Agent AI Software Engineering

An empirical study of 45 real coding-agent sessions and a provenance-grounded architecture for verifiable agent work

Mohammed Nayeem Khan mohammed.nayeem.khan@intel.com mnknayeem@gmail.com

Sarun Kumar K S

sarun.kumar.k.s@intel.com

Intel Corporation

Preprint · v1.2 · 29 October 2026 · arXiv cs.SE (cross-list cs.AI, cs.HC) · artifacts: Soei pilot prototype, frozen results, audit sheets

## Abstract

AI coding agents now perform long, multi-step and multi-session software engineering work with limited human supervision. When a developer returns to such work, the decisive question is not “what did the model emit?” but “what was actually done, by whom, and can I trust that it is finished?”. We study this question in two ways. First, we build a working prototype of our proposed system and run it on a real corpus of 45 coding-agent sessions (102,233 raw records; 20,577 tool executions; 2,336 human-to-agent episodes; April–October 2026). With a blindly audited classifier (precision 83.3% / 96.7%), we estimate that 23.4% [95% CI 17.1–35.7] of code-editing episodes end without any independent verification after the last edit, and that 57.8% of to-dos the agent marked complete after editing had no proof between edit and completion. A 10-minute timing heuristic for session hand-offs recovers only 6 of 24 (25%) session pairs that our content-lineage analysis links, because 95.5% of real cross-session artifact reuse happens more than ten minutes apart. 42% of sessions span more than 24 hours, and 48.1% of transcript bytes lie beyond a 2 MiB replay bound. Credential-like strings appear in 1,326 raw records across 16 of 45 sessions. Second, we propose Soei (Structured Observability of Evidence and Intent): an acknowledgement-aware event schema, a hash-chained local evidence ledger, a W3C-PROV-aligned evidence graph, calibrated inference for attribution, drift and lineage-based hand-offs, and a verification layer that separates an agent’s claim of completion from proof. On the real corpus, the prototype’s ledger is 20% of the raw transcript size, stores zero content and zero credential matches, verifies at about 126k events/s, detects tampering, and replays deterministically. We formalise the model, specify the architecture in implementation detail, and define a reproducible evaluation agenda. Finally, we argue that this evidence substrate is a prerequisite for a golden harness for evaluating and certifying coding agents, and for the governance of autonomous agents. We define autonomy levels with evidence gates and harness metrics whose baselines come from our data.

Keywords: AI coding agents; agent observability; provenance; verification; software engineering; situation awareness;   
empirical study; human–AI collaboration.

Key empirical findings (real data, this paper)

1. The claim–proof gap is real: about one in four code-editing episodes ends unverified (23.4%, CI 17.1–35.7%). Across 4 classifier versions the raw rate stays between 25.1% and 42.1%.

2. “Done” is not proven: 186 of 322 (57.8%) to-dos completed after edits had no verification between the edit and the “completed” status.

3. Hand-offs are slow, not fast: real cross-session reuse has a median gap of 8.8 days. A ≤10 min temporal heuristic finds 25% of linked pairs; content lineage finds all 24.

4. Bounded history loses most of the evidence: 19/45 sessions exceed 24 h, and 48.1% of bytes fall beyond a 2 MiB replay window.

5. Acknowledgement matters: 40 of 3,548 file mutations (1.1%, in 30 episodes) were requested but never succeeded. A start-event observer would show them as applied.

6. Raw agent logs are a liability: 1,326 records in 16/45 sessions contain credential-like strings (208 highconfidence token matches, 5 distinct tokens). The Soei ledger contains none.

7. Provenance is cheap: the hash-chained ledger costs 20% of raw size and verifies at about 126k events/s in pure Python, with deterministic replay and tamper detection.

## 1 Introduction

Coding agents have progressed from single-turn completion to long-horizon work. They read repositories, plan, edit many files, run tools and tests, and increasingly operate as several sessions or sub-agents. Issue-resolution benchmarks such as SWE-bench [1], agent–computer interfaces such as SWE-agent [2] and reason-act tool use [3] have driven rapid capability gains. The supervisory side of human–agent collaboration has received far less engineering and scientific attention. A developer returning to a running agent must rebuild state from scrolling transcripts across terminals. This maps directly onto the three levels of situation awareness (perception, comprehension, projection) [4].

Two families of tools address the problem in part. LLM trace dashboards (LangSmith, Langfuse, Arize Phoenix) and the OpenTelemetry generative-AI semantic conventions [5] record model calls, prompts, latency and token cost. They answer “what did the model do?” but not “which part of my project moved, and is it really done?”. Project-level observability tools watch the repository, a plan file and whatever agent events a vendor exposes, and visualise progress at the level of components and tasks. The better ones run locally and stay read-only. None of them, as far as we can find, measures how far what they show is from what actually happened.

We argue that visibility alone does not give a supervisor what they need. The evidence behind each view has to be trustworthy, and that is a testable claim. We therefore (i) measure the evidence problems that observability must solve on a real corpus of coding-agent sessions, and (ii) design, prototype and evaluate Soei, an architecture that treats agent reports as claims, stores evidence before inference, and derives every view from one tamper-evident record. Several 2026 papers reach related conclusions from other directions (§2.6); ours is, to our knowledge, the first to measure the claim–proof gap and the hand-off recall problem on real working sessions and to build the substrate as a passive observer of unmodified vendor agents.

## Contributions.

1. A gap taxonomy of 14 gaps across six dimensions, each mapped to a requirement (§3, App. B).

2. The Soei core architecture, specified to implementation level: event schema, acknowledgement tracker, privacy gate, ledger format, PROV-aligned graph, inference engines, claim–proof verifier, projection API, and the trust and deployment model (§5, §7–8).

3. A formal model with an evidence lattice, task state machine, confidence fusion, graded hand-offs, drift and completion trust (§6).

4. An empirical pilot on real data: 45 sessions, an audited claim–proof classifier with a bias-corrected estimate, lineage-versus-heuristic hand-off analysis, history-bound loss, acknowledgement failures, secret exposure and ledger cost (§9–10).

5. Reproducible artifacts: a stdlib-only prototype, frozen aggregate results, figure scripts and audit sheets (App. E).

6. A path to a golden harness and agent governance: autonomy levels with evidence gates, harness metrics with real-data baselines, a mapping to NIST AI RMF, ISO/IEC 42001 and the EU AI Act, and a prioritised improvement roadmap (§13–14).

## 2 Background and related work

## 2.1 Coding agents and their artifacts

Modern coding agents interleave reasoning and tool actions [3] through repository-specific interfaces [2] and are measured on issue resolution [1]. In practice they run inside vendor tools (VS Code Copilot agent, Codex CLI/desktop,

Claude Code, Cursor) that persist session logs and expose hooks and native to-do lists. These artifacts are the raw material for observability. They are also vendor-specific and incomplete, and their format changes without notice. As we show in §10.6, they may also contain secrets.

## 2.2 Tracing, clocks and observability

Distributed tracing (Dapper [6], later OpenTelemetry) models work as spans with causal parents and propagated context. LLM observability adapts this to model and tool calls, and the OpenTelemetry GenAI conventions [5] standardise model-invocation attributes. AgentTrace [23] instruments agents at runtime with structured securityoriented logs, and OpenRath [21] proposes a first-class Session runtime value that is branchable, inspectable and replayable, so that transcripts, tool effects and lineage stop being recorded in separate places. Both require the agent to be built on, or instrumented by, their framework. These approaches assume instrumented code paths and centre on requests rather than on project intent (plans, components, deliverables). Logical clocks [7] and hybrid logical clocks [11] address ordering across sources without synchronised time. That problem returns when provider logs, file-system events and CI results are fused

## 2.3 Provenance

The W3C PROV model [8] represents entities, activities and agents with relations such as wasGeneratedBy, used, wasAttributedTo and wasDerivedFrom. Provenance is a natural basis for questions such as “which session produced this exact file version, and did another session consume it?”. We show empirically that these questions cannot be answered from timing alone.

## 2.4 Visualisation and situation awareness

The “overview first, zoom and filter, then details on demand” principle [9] and Endsley’s situation-awareness model [4] set principled targets for a supervisory display: perception of current activity, comprehension of progress relative to intent, and projection of what happens next. SeaView [30] visualises SWE-agent trajectories for post-hoc error analysis; it reads benchmark trajectories rather than live developer sessions and does not separate claims from evidence.

## 2.5 Positioning

Table 1. Positioning of observability approaches for AI coding agents.
<table><tr><td>Capability</td><td>Transcripts</td><td>LLM trace dashboards</td><td>Soei</td></tr><tr><td>Unit of analysis</td><td>Message</td><td>Model/tool span</td><td>Claim, evidence, artifact@digest, task, component</td></tr><tr><td>Project intent</td><td>No</td><td>No</td><td>Yes, plus a quantified drift metric</td></tr><tr><td>Evidence semantics</td><td>Implicit</td><td>Implicit</td><td>Formal lattice + calibrated confidence</td></tr><tr><td>Completion</td><td>Agent says so</td><td>N/A</td><td>Verified only by proof</td></tr><tr><td>Hand-offs</td><td>No</td><td>Parent/child spans</td><td>Graded H0–H3, content lineage</td></tr><tr><td>History</td><td>Scrollback</td><td>Server DB</td><td>Append-only hash-chained ledger</td></tr><tr><td>Privacy</td><td>Raw</td><td>Usually cloud</td><td>Local; one ingest-time policy; zero content</td></tr><tr><td>Measured accuracy</td><td></td><td></td><td>Audited classifier, CIs, ablations</td></tr></table>

## 2.6 Concurrent work (2026)

Several papers published while this work was in progress treat agent output as a claim that needs independent evidence, and we position Soei against them here. Proof-or-Stop [18] gates lifecycle transitions (reviewed, tested, done, ready-tomerge) on fresh, source-state-bound, mechanically checkable evidence, and the runtime-contract position paper [19] argues for an “evidential face” of agent safety that gates task submission on test runs, log captures and diffs. Both enforce. Soei’s core is read-only by design (P1) and its contribution is to measure how large the gap is in ordinary use and to build the evidence substrate that such gates would consume; the policy plane we propose in §13.2 and §14 is where enforcement would live. Artifact-centred claim-aware observability [20] makes the same claim/artifact/relation argument for autonomous scientific agents; we make it for software work, add acknowledgement and hand-off lineage, and quantify it. Skill Runtime Intelligence [24] passively reconstructs Skill-lifecycle stages across heterogeneous coding-agent harnesses with four evidence grades and keeps unsupported stages as unknown, which is the closest match to our evidence lattice and adapter stance; it targets Skill activation rather than project-level claims, hand-offs or history integrity. Auditable Agents [22] defines auditability dimensions (action recoverability and others) that our four governance questions in §13.1 map onto; the hash-chained ledger and deterministic replay are one concrete way to satisfy them. On the empirical side, AgentLens [25] finds that 10.7% of passing OpenHands trajectories on SWE-bench Verified show regression cycles, blind retries or missing verification, which corroborates Finding 1 from a benchmark rather than a workplace corpus, and a large study of agent Skills [26] finds credential leakage through privileged skill code, a different channel from the transcript leakage we measure in §10.6. TraceTrial-Bench [27] and HarnessAudit [28] benchmark harness reliability and mid-trajectory safety, which is the kind of evaluation a golden harness (§13.2) would run; SWE-Critix [31] trains trajectory verifiers that could replace our rulebased proof classifier (§14, item 3). A project-level governance manifest for agent-mediated OSS contributions [29] and a systematisation of execution-security work for coding agents [32] frame the governance side. Table 2 summarises the differences.

Table 2. Closest concurrent work and how Soei differs.
<table><tr><td>Work</td><td>Shared idea</td><td>Where Soei differs</td></tr><tr><td>Proof-or-Stop [18]; Runtime contract [19]</td><td>Agent lifecycle states are claims; proof must be fresh and bound to source state</td><td>Observation only; measures the gap on 45 real sessions (23.4% unverified, 57.8% unproved to-dos); enforcement deferred to a separate policy plane</td></tr><tr><td>Claim-aware observability [20]</td><td>Artifacts, claims and their relations as the unit of observation</td><td>Software-engineering sessions; acknowledgement tracking; content- lineage hand-offs; hash-chained history</td></tr><tr><td>OpenRath [21]; AgentTrace [23]</td><td>One unified runtime record instead of fragmented state; replay</td><td>Passive adapters over unmodified vendor agents; no framework adoption required; privacy gate before persistence</td></tr><tr><td>Skill Runtime Intelligence [24]</td><td>Passive reconstruction, graded evidence, unsupported = unknown</td><td>Scope is project-level tasks and components; Skill stages are out of scope. Adds hand-off lineage and integrity/replay properties</td></tr><tr><td>Auditable Agents [22]</td><td>Accountability requires auditability</td><td>Concrete substrate (ledger, replay, PROV graph) and measured cost (20% of raw size, 126k events/s)</td></tr><tr><td>AgentLens [25]</td><td>Passing is not the same as verified</td><td>Workplace corpus, not benchmark; episode- and to-do-level claim- proof matching with audited precision</td></tr></table>

## 3 Gap analysis and taxonomy

We compare current project-level observability tooling (transcripts, trace dashboards and local plan/file watchers) against three requirements for a supervisory observability substrate. The gaps below are the ones we found while reviewing such tools and then confirmed in our own sessions. (R1) Faithfulness: what is shown is true to the evidence and its uncertainty. (R2) Durability and reproducibility: the same evidence yields the same view at any later time. (R3) Evaluability: accuracy and usefulness can be measured. The last column links each gap to the real-data measurement in §10 that quantifies it.

Table 3. Gap taxonomy G1–G14 with Soei requirement and empirical evidence.
<table><tr><td>ID</td><td>Dimension</td><td>Gap and evidence</td><td>Sev.</td><td>Soei requirement</td><td>Real-data evidence (§10)</td></tr><tr><td>G1</td><td>Architecture</td><td>Each view has its own service, adapters and history; no shared backend</td><td>High</td><td>One pipeline and ledger; views are projections</td><td>Replay determinism holds (§10.7)</td></tr><tr><td>G2</td><td>Architecture</td><td>Views that coexist overwrite each other's state (to-do overwrite, hook collision)</td><td>High</td><td>Evidence-class precedence; namespaced hooks</td><td></td></tr><tr><td>G3</td><td>Epistemic</td><td>Evidence classes are labels, with no confidence or fusion</td><td>High</td><td>Formal lattice; calibrated confidence</td><td>Audit precision 66.7→83.3% (§10.1)</td></tr><tr><td>G4</td><td>Epistemic</td><td>Plans accepted at tool-call start</td><td>High</td><td>Two-phase acknowledgement</td><td>1.1% phantom mutations (§10.5)</td></tr><tr><td>G5</td><td>Epistemic</td><td>Completion = reported complete</td><td>High</td><td>Claim-proof verification</td><td>23.4% unverified; 57.8% unproved to-dos</td></tr><tr><td>G6</td><td>Epistemic</td><td>Hand-offs: 10 min timing or explicit receipt</td><td>High</td><td>Graded H0–H3 with content lineage</td><td>Heuristic recall 25% (§10.3)</td></tr><tr><td>G7</td><td>Epistemic</td><td>Role inference from directory names; accuracy unmeasured</td><td>Med.</td><td>Scored attribution with abstention</td><td>Agenda (§12)</td></tr><tr><td>G8</td><td>Temporal</td><td>In-memory/bounded history (2 MiB, 24 h)</td><td>High</td><td>Checkpointed ledger, lossless resume</td><td>48.1% bytes beyond bound (§10.4)</td></tr><tr><td>G9</td><td>Temporal</td><td>No cross-source causal order</td><td>Med.</td><td>HLC + source offsets</td><td></td></tr><tr><td></td><td>G10 Security/privacy</td><td>Unauthenticated mutation routes on localhost; inconsistent field policies between views</td><td>High</td><td>Token, Origin/Host checks, ingest- time policy</td><td>1,326 secret-bearing records (§10.6)</td></tr><tr><td>G11 Platform</td><td></td><td>Windows/Cursor unvalidated; format drift</td><td>Med.</td><td>Adapter SDK + conformance corpus</td><td>Prototype ran on Windows</td></tr><tr><td>G12 Interop</td><td></td><td>No standard export</td><td>Med.</td><td>OTel GenAI + PROV-JSON export</td><td></td></tr><tr><td></td><td>G13 Supervision</td><td>Attention limited to visual cues</td><td>Low</td><td>Ranked read-only attention events</td><td>195 returns after &gt;30 min</td></tr><tr><td>G14 Scientific</td><td></td><td>No quantitative evaluation</td><td>High</td><td>Evaluation-first harness</td><td>(§10.8) This paper's pilot</td></tr><tr><td colspan="6"></td></tr><tr><td colspan="3" rowspan="6">Views disagree “Done” but broken Phantom edits shown Missed hand-offs History vanishes</td><td>Root cause (gap) G1/G2 split backends G5 claim ≠ proof</td><td>1</td></tr><tr><td></td><td></td></tr><tr><td>→</td></tr><tr><td>一 G4 no acknowledgement</td></tr><tr><td>→ G6 timing heuristic</td></tr><tr><td>→</td><td></td></tr><tr><td>Secrets on screen</td><td></td><td>G8 bounded memory</td><td>1 Checkpointed ledger</td></tr><tr><td></td><td>1</td><td>G10 late redaction</td><td>1 Ingest privacy gate</td></tr></table>

Figure 1. Symptom → gap → mechanism map for the high-severity gaps.

## 4 Design principles

1. P1 Observe, never act. Soei never edits or approves agent state, and it never prompts the agent.

2. P2 One record, many views. Every view is a deterministic projection of one ledger.

3. P3 Evidence before inference. Provenance-tagged evidence is persisted before interpretation, so inference can be re-run later with better rules and re-scored against the same evidence.

4. P4 Claims are not proofs. Agent-reported state is a claim. Completion is upgraded only by independent evidence.

5. P5 Calibrated uncertainty. Inferred relations carry measured confidence, and the system abstains when that confidence is below threshold.

6. P6 Private by construction. One field policy is enforced at ingest, before persistence. Secrets never enter the ledger.

7. P7 Vendor-neutral adapters. Provider formats are untrusted, versioned inputs validated against a conformance corpus.

8. P8 Evaluation-first. Recording, annotation, replay and scoring are first-class features.

5 Soei core architecture  
![](images/5427ddd8468d24cf3439c01a62343599b41d8aedd30335fbb41235e5f9996229.jpg)  
Figure 2. Soei layered architecture. One pipeline (L1–L5) feeds every projection (L6). Evidence is persisted before inference (P3), and verification is separate from inference (P4).

## 5.1 Core data path

![](images/9eb397ffd912a3d29f9cc58bc8143e04c63671779f44d0b32e73a126b444be4d.jpg)  
Figure 3. Core data path. The top row is the write path; the bottom row is the incremental read path. Evidence becomes visible only after it is durable, so the view after a crash equals the view before it.

## 5.2 L1 Ingest: adapters, Soei Event Schema and acknowledgement

Adapter contract. An adapter implements detect(sample) → (format, version, confidence), tail(cursor) → (records, cursor') and normalize(record) → SES[]. Cursors are (path, inode/file-id, size, byte offset). A rotated or truncated source is detected from the identity and size and re-anchored, never re-read blindly. Each adapter ships with a conformance corpus of redacted real records per supported format version. CI runs every adapter against every corpus on Windows, macOS and Linux. Unknown formats produce evidence\_class=unknown events and are never guessed. Our prototype implements the VS Code Copilot agent adapter (session, prompt, message, tool start/complete with call ids and success flags).

SES envelope. CloudEvents-style [10] fields (id, source, type, time) plus source\_offset, hlc, evidence\_class, ack, subject (session, task, artifact digest, activity) and policy (App. A).

Acknowledgement tracker. Every state-changing request (edit, plan update, to-do change) is recorded as proposed at start and resolved by its matching completion: acknowledged (success), rejected (failure) or timed\_out (no completion within a window or before session end). Projections render only acknowledged state by default. In our corpus this distinction affects 1.1% of mutations (§10.5).

Privacy gate. Runs before persistence. (i) A field allowlist per event type keeps message, prompt, reasoning, command body and tool output out by default. (ii) Remaining strings are scanned for credential patterns and replaced with typed placeholders. (iii) Paths are salted-hashed for export, with a local reversible map kept outside the ledger. Because the gate runs before the ledger, a secret can never be “un-shown” later: it is never stored.

## 5.3 L2 Evidence ledger

![](images/5527f83746090f62b1fb523e95fda8d1d7e7417aaf4ed47d46fd384cf65bbcf9.jpg)  
Tampering with any event (e.g., flipping ack: acknowledged→rejected) breaks every later hash; verification is O(n) and ran at ≈126k events/s in the prototype.  
Figure 4. Ledger record chaining and segment manifest.

The ledger is an append-only sequence of canonical-JSON SES events grouped into segments (the prototype uses JSONL; production can use SQLite WAL with the same chain). Each record stores prev and hash = SHA-256(prev ‖ canonical(e)). A segment manifest commits to its head hash, schema and policy versions and source cursors, which gives lossless resume and integrity for shared datasets. The ledger stores no content: artifacts are referenced by salted path hash and, in production, content digests (BLAKE3/SHA-256 or VCS blob ids) captured at generation and use.

## 5.4 L3 Evidence graph

![](images/f0f9b25124053e96b793b9355435b45c04063ffad6cf39bace52bd752cb0635e.jpg)  
H2 hand-off: S2 used a@d generated by S1 · T becomes Verified because P ran after the last edit and used a@d  
Figure 5. Evidence-graph schema (PROV-aligned) with an H2 hand-off and a verified claim.

The graph is folded incrementally from the ledger. Node types are Agent, Session, Activity, Artifact@digest, Task, Component, Claim, Proof. Edge types are wasGeneratedBy, used, wasAssociatedWith, partOf, claims, proves, derivedFrom. Every inferred edge carries (confidence, inference\_version, support=[event ids]), so any rendered statement can answer “why am I seeing this?” with raw evidence.

## 5.5 L4 Inference engines

Attribution: maps activities and artifacts to components and roles using plan globs, path features, co-change history and operation type. It is a calibrated classifier that abstains below threshold θ. Directory-name role heuristics become one feature among several.

Lineage and hand-off: computes H0–H3 (§6.4) from used/wasGeneratedBy over artifact digests. Our data shows content lineage is required, because real reuse is slow (§10.3).

Drift: undeclared and stale-done drift (§6.5) turn the usual “plan is behind” indicator into measurable signals.

Attention: ranks read-only alerts: blocked-on-human, stalled task, contested claim (failing proof after a completion claim), sustained undeclared work, and returns after long idle periods.

## 5.6 L5 Verification

The proof classifier labels activities as static (compile, lint, type-check, diagnostics), build (build/package), test (test runners or test-named scripts), exec (running the changed artifact or its sibling), smoke (import checks), probe (HTTP to localhost after restart) or UI (browser checks after an edit). The claim–proof matcher then decides Verified, Unverified or Contested for every claim, and the completion trust τ summarises coverage and recency (§6.6). This is the layer the real data shows to be missing in practice (§10.1–10.2).

## 5.7 Task state machine

![](images/5ba4fe0d653f76c2879f4e758959e2c4cbdc134184126c373ff4d0db66326abb.jpg)  
Figure 6. Task lifecycle. A task cannot leave Proposed without acknowledgement, and it cannot reach Verified without a proof that ran after the last edit and used the produced artifacts. Any later change returns it to ClaimedDone.

## 5.8 Security, trust boundaries and deployment

![](images/6160250b5c7286a2485b6e81b0acaf26e7f314ca4304967b54b7b76e77e6bb12.jpg)  
Figure 7. Trust boundaries. Everything entering B2 is untrusted. Persistence happens only after the privacy gate. Every API call needs the capability token and passes Host/Origin checks, which rules out the unauthenticated-localhost class of issue common in local developer tools.

Threats addressed: (T1) cross-site requests or DNS rebinding against localhost services; (T2) leakage of secrets or code through screenshots and exports; (T3) hook relay hijack or duplication; (T4) dataset tampering; (T5) adapter confusion by malformed input. Controls: 127.0.0.1 binding; a random capability token per launch, required on every route including reads; strict Host and Origin allowlists; an ingest-time allowlist and secret scrubber; signed, namespaced, idempotent hook registrations; hash-chained segments with manifests; and size and recursion limits on adapters, run under fuzzing in CI.

## 6 Formal model

Definition 1 (Evidence event and ledger). An evidence event is a tuple $e = ( i d ,$ src, off, κ, λ, σ, α, π): source, source offset, kind, hybrid logical timestamp, subject references, acknowledgement state α ∈ {proposed, acknowledged, rejected, timed\_out, n/a}, and applied policy. The ledger is $L = \langle e _ { I } . . . e _ { n } \rangle$ with $h _ { i } = H ( h _ { i - I } \parallel e n c ( e _ { i } ) )$

Definition 2 (Evidence lattice). Statements carry a class from Unknown < Inferred < Observed < Reported < Verified. For statements about intent, Reported outranks Observed. For statements about the world, Observed outranks Reported. Verified is reachable only through a Proof node.

Confidence fusion. For a relation r with independent signals of calibrated confidence $c _ { j }$ and reliability weights w<sub>j</sub>:

$$
c ( r ) = I - \textstyle \prod _ { j } { } ( I - w _ { j } c _ { j } )\tag{1}
$$

Competing hypotheses are normalised, and the system abstains when max $c < \theta .$ Calibration is reported as expected calibration error.

Definition 3 (Task lifecycle). Proposed → Acknowledged → InProgress → ClaimedDone → {Verified, Contested}, with Withdrawn and Reopened. ClaimedDone → Verified requires a proof p with outcome pass, $\lambda ( p ) >$ λ(last generation under T) and used(p) ⊇ artifacts(T) (or their component).

## 6.4 Hand-off levels

Table 4. Graded hand-off evidence between sessions $\mathrm { \Delta S _ { 1 } }$ and $\mathrm { S } _ { 2 }$
<table><tr><td>Level</td><td>Condition</td><td>Meaning</td><td>Measured here</td><td></td></tr><tr><td>H0</td><td> $\mathrm { S } _ { 2 }$  starts within ∆ of  $\mathrm { \Delta } \cdot \mathrm { \Delta } \mathrm { S } _ { 1 }$  ending</td><td>Temporal adjacency</td><td>Δ = 10 min</td><td>7 pairs; 6 lineage-confirmed</td></tr><tr><td>H1</td><td>H0  $\wedge \mathrm { { S } } _ { 2 }$  uses artifacts  $\mathrm { S } _ { 1 }$  modified</td><td>Shared working set</td><td>(subsumed by H2 path-level)</td><td></td></tr><tr><td>H2</td><td>∃a: wasGeneratedBy(a, S1) Λ used  $( \mathsf { S } _ { 2 } , \mathsf { a } )$ </td><td>Content lineage</td><td>24 pairs, 292 links</td><td></td></tr><tr><td>H3</td><td>H2 Λ explicit receipt</td><td>Confirmed transfer</td><td>N/A (no receipts in corpus)</td><td></td></tr></table>

## 6.5 Plan–reality drift

$$
D r i f t _ { u n d e c l a r e d } ( W ) = | O _ { W } \mid D _ { W } | / \mid O _ { W } | , ~ D r i f t _ { s t a l e - d o n e } ( W ) = | \langle C \in O _ { W } ; s t a t u s ( C ) = d o n e j | / \mid O _ { W } \mid O _ { W } \rangle ,\tag{2}
$$

## 6.6 Completion trust

$$
\tau ( T ) = c o v ( T ) \cdot e ^ { - \delta \cdot a g e ( T ) } \cdot ( I - f ( T ) ) , c o v ( T ) = | A _ { T } \cap u s e d ( P _ { T } ) | \cdot | A _ { T } |\tag{3}
$$

## 6.7 Properties (with empirical checks)

Property 1 (Projection consistency). Views defined as functions of $G ( L _ { \leq t } )$ agree on every shared statement. This rules out G1/G2-type divergence by construction.

Property 2 (Replay determinism). Given L<sub>≤t</sub>, inference version v and policy π, the projection is deterministic.   
Empirical check: two independent replays of the real 42,398-event ledger produced identical result digests (§10.7).

Property 3 (Tamper evidence). Changing any event invalidates every later hash. Empirical check: flipping one acknowledgement in the middle of the real ledger was detected (§10.7).

Property 4 (No silent completion). No task reaches Verified without a Proof node, and no rejected or timed-out mutation is rendered as applied.

## 7 Algorithms

Algorithm 1 Acknowledgement-aware ingest   
input: source records R (ordered), window W   
pending ← {}   
for r in R:   
if r.kind = tool.start: pending[r.call] ← event(r, ack = proposed); emit(pending[r.call]) # not rendered   
yet   
elif r.kind = tool.complete: e ← pending.pop(r.call); e.ack ← acknowledged if r.success else rejected   
emit\_resolution(e) # now renderable   
else: emit(normalize(r))   
on session end or age > W: for e in pending: e.ack ← timed\_out; emit\_resolution(e)

Algorithm 2 Claim–proof matching (per episode and per task)   
for each episode / task T:   
lastEdit ← max λ over acknowledged artifact.generated under T   
proofs ← { p : p.kind ∈ {static, build, test, exec, smoke, probe, ui}, p.ack = acknowledged, λ(p) > lastEdit   
}   
if proofs = ∅ : state(T) ← Unverified   
elif ∃p ∈ proofs failed covering A\_T : state(T) ← Contested   
else : state(T) ← Verified; τ(T) ← Eq.(3)

Algorithm 3 Lineage hand-off detection (H2)   
writes[a] ← list of (λ, session) for acknowledged generations of artifact a # a = path hash (prototype) or   
path@digest   
for each acknowledged use (a, λu, S2):   
(λw, S1) ← latest write of a with λw ≤ λu   
if S1 ≠ S2: link(S1 → S2, gap = λu − λw) # H2 if digest(write) = digest(use)   
pairs ← distinct (S1, S2) over links; compare with H0 pairs (end→start ≤ 10 min)

Complexity. Ingest and chain verification are O(n) in events. Lineage is O(u log w) with per-artifact sorted write lists. Claim–proof matching is O(n) with one pass per session. All prototype stages ran in seconds on the full real corpus (§10.7).

## 8 Implementation blueprint

Table 5. Reference implementation choices.
<table><tr><td>Concern</td><td>Choice</td><td>Prototype status</td></tr><tr><td>Runtime</td><td>Single local daemon (Python 3.11+ or Node 20+)</td><td>Python, stdlib only</td></tr><tr><td>Ledger</td><td>Canonical JSON + SHA-256 chain; JSONL or SQLite WAL segments</td><td>JSONL, implemented</td></tr><tr><td></td><td>Acknowledgement Start/complete pairing by call id</td><td>Implemented</td></tr><tr><td>Privacy gate</td><td>Allowlist + secret scrub + salted path hashes</td><td>Allowlist + hashing implemented; scrub validated (0 matches in ledger)</td></tr><tr><td>Proof classifier</td><td>Versioned rules + learned model later</td><td>Rules v1–v4, audited</td></tr><tr><td>Lineage</td><td>Content digests at generation/use; Git blob ids</td><td>Path-hash level (digest is future work)</td></tr><tr><td>Clocks</td><td>Hybrid logical clocks per source</td><td>Wall time + source offsets</td></tr><tr><td>API</td><td>HTTP + SSE, 127.0.0.1, capability token, Host/Origin allowlist</td><td>Specified</td></tr><tr><td>Interop</td><td>OTel GenAI + PROV-JSON exporters</td><td>Specified</td></tr></table>

soei/   
adapters/ copilot\_vscode/ codex/ claude/ cursor/ fs/ git/ junit/ ci/ (+ conformance/<format>/<version>/)   
core/ ses.schema.json normalizer ack\_tracker hlc privacy\_gate   
ledger/ writer segments manifest checkpoints verify\_chain replay   
graph/ prov\_model fold snapshot query   
inference/ attribution lineage drift attention calibrate   
verify/ proof\_classifier claim\_proof trust   
api/ server(token, origin) sse exporters/{otel, prov, csv}   
views/ graph roles timeline cli vscode   
eval/ recorder audit\_sampler annotate score overhead ab\_replay

Table 6. Roadmap.
<table><tr><td>Phase</td><td>Deliverable</td><td>Gaps</td><td>Exit criterion</td></tr><tr><td>1 Substrate</td><td>SES, ledger, checkpoints, Copilot/Codex/Claude/FS/Git adapters, ack tracker, secure API</td><td>G1 G2 G4 G8 G9 G10</td><td>Lossless restart; chain verifies; zero view divergence</td></tr><tr><td>2 Graph &amp; inference</td><td>PROV graph, digest lineage, calibrated attribution, drift</td><td>G3 G6 G7</td><td>Macro-F1 and ECE on an annotated corpus</td></tr><tr><td>3 Verification</td><td>Proof classifier v5 (learned), test/CI adapters, τ</td><td>G5</td><td>Audit precision ≥ 90% both labels</td></tr><tr><td>4 Reach</td><td>Cursor/Windows conformance, exporters, VS Code panel, attention</td><td>G11 G12 G13</td><td>Conformance passes on 3 OSes</td></tr><tr><td>5 Study</td><td>Multi-developer corpus, user study, release</td><td>G14</td><td>§12 protocol executed</td></tr></table>

## 9 Empirical pilot study (real data)

## 9.1 Research questions

1. PQ1 (Claim–proof): How often does an agent finish an episode, or mark a to-do complete, after editing code but without independent verification?

2. PQ2 (Hand-offs): How much real cross-session artifact reuse does a timing heuristic capture compared with content lineage?

3. PQ3 (History bounds): How much real evidence falls outside bounded discovery and replay windows?

4. PQ4 (Acknowledgement): How often are state-changing requests not acknowledged?

5. PQ5 (Privacy): How much sensitive material do raw agent logs contain, and does an ingest-time gate keep it out?

6. PQ6 (Cost and integrity): What does a hash-chained ledger cost, and do determinism and tamper evidence hold on real data?

## 9.2 Corpus

![](images/c1f98e45fef1985b08a6f3fb2312ec925c57441a6634626b8f858d11e326ef0b.jpg)  
Figure 8. Pilot pipeline: the Soei prototype applied to the real corpus.

The corpus is every VS Code Copilot agent-mode transcript available on the first author’s workstation: 45 sessions, 23 April to 5 October 2026, 102,233 raw records (72.6 MB), 20,577 tool executions over 45+ tool types, 2,336 humanto-agent episodes (an episode runs from one human prompt to the next), of which 1,044 contained file edits and 893 edited code. Work spanned post-silicon validation content (Python/PowerShell), web portals, VS Code extensions, data pipelines and documentation. All sessions were real work, not tasks constructed for the study.

## 9.3 Measures

Unverified code episode: an episode with ≥1 acknowledged code edit and no acknowledged proof activity (§5.6) after its last code edit.

Unproved completion: a to-do that turned completed after acknowledged edits made while it was in-progress, with no proof after those edits and before completion.

Lineage link: session $\mathrm { S } _ { 2 }$ read artifact a whose latest prior acknowledged write was by $\mathrm { S } _ { 1 } \neq \mathrm { S } _ { 2 }$ (path level). H0 pair: $\mathrm { S } _ { 2 }$ started within 10 min of $\mathrm { S } _ { 1 }$ ending.

Phantom mutation: an edit request whose completion was a failure or missing.

Secret exposure: raw records matching credential patterns (GitHub, AWS and Slack token formats, private-key headers, password assignments). Counted only; values are never stored or printed.

## 9.4 Classifier audit protocol

The proof classifier is a set of versioned rules. To avoid reporting an artefact of the heuristics, the first author ran blind audits. For each version a seeded random sample of 30 episodes carrying the label was drawn, and the actions after the last code edit (tool names and truncated commands only) were inspected to decide whether an independent check really happened. False negatives drove each revision: v2 added build commands, quoted interpreter paths, import smoke checks and test-named scripts; v3 excluded deploy/sync invocations; v4 added localhost HTTP probes, browser UI checks, invocation of edited PowerShell scripts and packaging. Every version is re-run on the whole corpus as a sensitivity analysis, and the final estimate is corrected for measured precision in both directions.

Two of my own misreadings (first author) shaped these rules more than any design discussion did. In the v1 audit I had counted python -c import checks and scripts named \_tmp\_test\_\* as “no verification” because they did not look like tests; in this corpus they are one of the main ways an agent checks its own work. The opposite mistake survived to v4: one scp deploy was read as execution of the changed script, and only reading the raw tool sequence showed that nothing had run locally. I would not have found either by looking at aggregate rates.

## 10 Results

![](images/dd8bc5e4acfcdccbf6c27bcd5d20af2568b7388e47c0d7145cc9039d7b8c8143.jpg)  
Figure 9. Tool-execution profile of the real corpus (top 16 tools) with failure rates. Terminal use and file reading dominate, and edits are frequent (3,548 mutation requests)

## 10.1 PQ1: the claim–proof gap

Claim-versus-proof gap is robust to classifier refinement  
![](images/c0e638d6eb608200ab7937d1496445b9c5acc9badab95583567b52aa8f416eaa.jpg)  
Figure 10. Share of code-editing episodes that end unverified (blue) and of edit-backed to-dos completed without proof (orange) across classifier versions. The red marker is the audit-corrected estimate with a 95% CI.

Table 7. Claim–proof results by classifier version (real corpus; 893 code-editing episodes; 322 edit-backed completed to-dos).
<table><tr><td>Version</td><td>Unverified code episodes</td><td>%</td><td>Unproved completions</td><td>%</td><td>Audit precision ('unverified')</td></tr><tr><td>vl</td><td>376</td><td>42.1</td><td>210</td><td>65.2</td><td>20/30 = 66.7%</td></tr><tr><td>v2</td><td>245</td><td>27.4</td><td>191</td><td>59.3</td><td></td></tr><tr><td>v3</td><td>245</td><td>27.4</td><td>191</td><td>59.3</td><td>22/30 = 73.3%</td></tr><tr><td>v4 (final)</td><td>224</td><td>25.1</td><td></td><td>57.8</td><td>25/30 = 83.3%</td></tr></table>

With v4 precision for the “unverified” label $p _ { U } = 0 . 8 3 3$ and for the “verified” label $p _ { V } = 0 . 9 6 7$ (29/30), the corrected prevalence is

$$
\pi ^ { * } = \ell _ { U } p _ { U } + ( I - \ell _ { U } ) \cdot ( I - p _ { V } ) = 0 . 2 5 I \cdot O . 8 3 3 + 0 . 7 4 9 \cdot O . 0 3 3 = 2 3 . 4 \% \enspace \middle . ( M i s o n - p r o p a g a t e d  9 5 9 \%  \varGamma . I 7 . I 9 \otimes ,  3 5 . 7 \%  ( 4 )
$$

Finding 1. In real day-to-day use, about one in four code-changing interactions ends with the agent handing control back without any independent check after its last edit. The rate falls as the classifier improves (42.1 → 25.1%), but it settles well above zero. The bias-corrected lower bound is 17.1%.

Finding 2. 57.8% of to-dos that the agent itself marked completed after editing had no verification between edit and completion. A dashboard that renders “completed” to-dos as delivered work would show these as done. This is direct evidence for gap G5 and for Soei’s claim–proof layer.

![](images/d879132fdc07bdda4cbc7709fb6498bf2e9a3a5bd4dfcaa9e55b2ef264541fce.jpg)  
Figure 11. What agents use as “proof” (v4). Executing the changed artifact and UI checks dominate. Formal tests are rare (166 of 3,697 verification actions, 4.5%).

![](images/86d6eebfd3da64ba7e0489be4234f971ee3343cdf59a53b5535a254fb66f74a6.jpg)  
Figure 12. Audited classifier precision with 95% Wilson intervals, n = 30 per point.

Finding 3. When agents do verify, they mostly run the changed thing (exec 40.3%, UI 20.5%, smoke 15.6%); formal tests account for 4.5% of verification actions. Verification in practice is behavioural and ad hoc, so Soei’s proof taxonomy must cover execution, probe and UI evidence as well as test reports.

## 10.2 Agent-reported completion versus independent evidence

Taken together, Findings 1–3 show that “reported complete” and “observed verified” are different quantities in real use, and that the difference is large enough to change a supervisor’s decision. Soei renders the two separately: Verified (green), Unverified (amber) and Contested (red), with τ as a secondary cue.

## 10.3 PQ2: hand-offs are slow, so timing heuristics miss them

![](images/7ebbec7b70a1d39cbf20476b43e8fe4008c001b31c607171ca04507cf2297923.jpg)  
time from artifact write (session A) to read (session B), minutes, log

Figure 13. Gap between the latest write of an artifact in session A and its read in session B, over 292 real cross-session lineage links. The red line is the 10-minute window of the temporal hand-off heuristic (H0).  
Table 8. Content lineage versus temporal heuristic (H0, Δ = 10 min).
<table><tr><td>Quantity</td><td>Value</td></tr><tr><td>Cross-session lineage links / distinct session pairs</td><td>292/24</td></tr><tr><td>Links whose write→read gap is ≤ 10 min</td><td>4.5%</td></tr><tr><td>Median write→read gap</td><td>12,742 min (≈ 8.8 days)</td></tr><tr><td>Session pairs flagged by H0 (end→start ≤ 10 min)</td><td>7</td></tr><tr><td>H0 pairs confirmed by lineage (precision)</td><td>6 / 7 = 85.7%</td></tr><tr><td>Lineage-linked pairs recovered by H0 (recall)</td><td>6 / 24 = 25.0%</td></tr></table>

Finding 4. Temporal hand-off detection is precise but blind: when it fires it is usually right (85.7%), but it recovers only 25% of the session pairs that actually build on each other’s artifacts. Real reuse happens across days (median 8.8 days, bimodal around hours and weeks), so only content lineage (H2) can capture multi-session collaboration.

## 10.4 PQ3: bounded history loses most of the evidence

![](images/aa9153d6e3bc0f62fb71709ff08f3496acd3d77871d4b239a71e03cc955c5537.jpg)

![](images/ad32143e67c5bd0a54b7c1cd53bd1ec114a1fae85b0bbb97ad0fb8ee34650f51.jpg)  
Figure 14. Left: CDF of session span (19/45 exceed a 24 h discovery window). Right: CDF of transcript size against a 2 MiB replay bound.

Finding 5. 42% of real sessions span more than 24 h (median 5.5 h, max ≈ 43 days), and 6 of 45 transcripts exceed 2 MiB. Those six hold 48.1% of all bytes, which a 2 MiB initial replay would not reconstruct after a restart. Long, valuable sessions are exactly the ones bounded history drops, which motivates a checkpointed ledger (G8).

## 10.5 PQ4: acknowledgement

Of 3,548 edit requests, 40 (1.1%) were not acknowledged as successful, spread across 30 episodes. Overall, 193 of 20,577 tool executions failed (0.9%). Failure rates were highest for create\_file (2.4%), list\_dir (2.7%) and interactive questions (6.0%).

Finding 6. An observer that records state changes at request time (as some legacy plan adapters do) would show about 1 in 90 edits as applied when they never happened. The rate is small, but each error is silent and propagates into every downstream inference. Two-phase acknowledgement removes it at essentially no cost.

## 10.6 PQ5: raw agent logs are a privacy liability

![](images/5d4e55179c20c50b9b4e222b5e2391c2c48dfefdc3af6435bdb32ab0ff0de082.jpg)

![](images/9e282cd208ca67db00af365efa992762e747c2e4f356c1d8a420e8e303349dfd.jpg)

![](images/e194eaa5cd2d9e99275571f88dd25fc8f837decd894acce1054fd51c37941a73.jpg)  
Figure 16. Ledger size and prototype throughput on the real corpus.  
Figure 15. Raw records with credential-like patterns, by record type, compared with the Soei ledger.

Finding 7. 1,326 raw records in 16 of 45 sessions contain credential-like strings, including 208 high-confidence token matches (5 distinct tokens, 11 sessions). They appear in human prompts (44), agent messages (654) and tool arguments (628): agents repeat secrets in their own messages and pass them into commands. Tools that display tool-call details can therefore show secrets on screen and in recordings. The Soei ledger built from the same corpus contains 0 matches, because the privacy gate runs before persistence. (The affected credentials must be rotated before any wider use of the corpus.)

## 10.7 PQ6: cost and integrity

Table 9. Prototype cost and integrity on the real corpus (single laptop core, CPython).
<table><tr><td>Measure</td><td>Value</td></tr><tr><td>Raw transcripts → ledger</td><td>76.1 MB → 15.2 MB (19.96%)</td></tr><tr><td>Ledger events</td><td>42,398</td></tr><tr><td>Ingest + normalise + gate</td><td>2.34 s (≈18k events/s)</td></tr><tr><td>Ledger write / full chain verification</td><td>0.45 s / 0.34 s (≈126k events/s)</td></tr><tr><td>Graph fold + all analyses</td><td>0.62 s</td></tr><tr><td>Replay determinism (two independent replays)</td><td>identical digests √</td></tr><tr><td>Tamper test (flip one ack mid-ledger)</td><td>detected √</td></tr><tr><td>Content / secret matches stored</td><td>0/0</td></tr></table>

Provenance turned out to be cheap. Five months of one developer’s agent activity fits in a 15 MB tamper-evident ledger that verifies in a third of a second and replays deterministically. At this scale the overhead argument against durable, verifiable agent history does not hold (Finding 8).

## 10.8 Supervision patterns

The corpus contains 195 human returns after more than 30 minutes of session inactivity (median gap 150 min). Each is an instance of the thirty-minute question: what happened while I was away, and is it done? Each is also a moment when Soei’s attention model would present Unverified and Contested claims first.

## 11 Discussion and threats to validity

What the data changes. Three design decisions that were previously argued now have measurements behind them. (1) Verification is not optional: a quarter of code-changing episodes and over half of agent-completed edit-backed to-dos lack proof. (2) Lineage must be content-based: timing heuristics see a quarter of real collaboration. (3) History must be durable and private: long sessions carry most of the evidence and many secrets.

Internal validity. The proof classifier is rule-based. We mitigate this in two ways: blind audits (4 samples of 30) with a two-sided precision correction and propagated confidence intervals, and a sensitivity analysis across four classifier versions. The audit also changed our view of what counts as proof. Running the edited script, hitting localhost or opening the page in a browser is how agents check work in this corpus, and test runners are 4.5% of verification actions; a classifier built from unit-test conventions alone would have overstated the gap by almost half (42.1% at v1 against 25.1% at v4). Proofs in other channels (e.g., the human testing manually outside the agent) are invisible, which biases the “unverified” rate upwards from a human perspective but not from an agent-evidence perspective. The latter is what an observability tool can display. Lineage uses path-level identity, not content digests: overwrites between write and read could create spurious links, and digest-level lineage is planned. To-do completion uses the agent’s own to-do tool, whose granularity varies.

External validity. The corpus comes from one developer and one agent product (VS Code Copilot agent mode, which ran several frontier models), in a post-silicon validation and tooling context. Rates will differ across teams, agents and domains. We therefore frame the results as an existence and magnitude pilot that motivates design, not as population estimates. AgentLens [25] reports missing or disordered verification in 10.7% of passing benchmark trajectories with a different agent and corpus, which suggests the phenomenon is not specific to our setting. The multi-developer study in §12 addresses generalisation.

Construct validity. “Verified” means an independent check ran after the last edit, not that the code is correct. Soei never presents τ as a correctness probability.

Ethics and privacy. All analysis ran locally on the authors’ own sessions. No message, prompt, command or file content was persisted, and only aggregate counts are reported. Credential values were never printed or stored, and rotation of the affected credentials is required. A multi-developer study requires informed consent and ethics review.

## 12 Research agenda

Table 10. Follow-on studies.
<table><tr><td>RQ</td><td>Question</td><td>Design</td><td>Metrics</td></tr><tr><td></td><td>RQ1 Attribution fidelity and calibration</td><td>Annotated multi-repo corpus; two annotators, κ</td><td>Macro-F1, coverage at θ, ECE</td></tr><tr><td></td><td>RQ2 Digest lineage versus path lineage versus H0</td><td>Instrumented digests at write/read; seeded hand-offs</td><td>P/R/F1 per level</td></tr><tr><td></td><td>RQ3 Does claim-proof display reduce false acceptance?</td><td>Within-subjects, ≥16 developers, replayed sessions with seeded false completions</td><td>False-acceptance rate, decision accuracy</td></tr><tr><td></td><td>RQ4 Integrity and cost at team scale</td><td>Fault injection (kill –9, rotation, format change)</td><td>Event loss, divergence, recovery time, p95 latency</td></tr><tr><td></td><td>RQ5 Situation awareness</td><td>SAGAT probes [4], counterbalanced: transcripts versus a trace dashboard versus Soei</td><td>Probe accuracy, time-to-awareness, NASA-TLX, SUS</td></tr><tr><td></td><td>RQ6 Generalisation of the pilot findings</td><td>Opt-in corpus from multiple developers, agents (Codex, Claude, Cursor) and domains</td><td>Rates of Findings 1–7 with CIs per stratum</td></tr></table>

Baselines and ablations. Baselines: raw transcripts, an LLM trace dashboard, a local plan/file watcher. Ablations: Soei without acknowledgement, without lineage (H0 only), without verification, without calibration, and with directoryname roles only. Replay determinism lets every ablation run on identical evidence.

## 13.1 The autonomy problem

Every episode in our corpus had a human at the end of it, and still about one in four code-changing episodes ended unverified (§10.1). In that setting the human is the last line of defence, re-reading and re-running what the agent left behind. As agents become autonomous, running for hours, chaining sub-agents, merging and deploying without per-step approval, that defence goes away. An unverified hand-back is then a defect escape path. A missed hand-off becomes an unaccountable change and a phantom edit a corrupted audit trail. A secret in a log becomes an incident. Autonomy therefore needs a substrate that can answer, for any change and at any later time, four governance questions (compare the auditability dimensions of [22]):

1. Accountability: which agent, session and model produced this exact artifact version, and on whose instruction?

2. Assurance: what independent evidence shows the change works, and is that evidence newer than the change?

3. Auditability: can the record be shown to be complete and untampered, and can the decision context be replayed exactly?

4. Containment: did any sensitive data enter the record or leave the machine?

Soei’s layers map one-to-one onto these questions: lineage and attribution (L3–L4) for accountability, the claim–proof verifier (L5) for assurance, the hash-chained ledger and replay (L2) for auditability, and the ingest-time privacy gate (L1) for containment.

## 13.2 The golden harness

We use golden harness for a reference evaluation and assurance environment in which (i) every agent action is captured as verifiable evidence, (ii) every completion claim is checked against independent proof or a task oracle, and (iii) every run can be replayed and compared bit-for-bit across agent, model or prompt versions. It serves three purposes: evaluation (how good is this agent on our work?), regression (did the model or agent upgrade make it worse or less honest?) and certification (is this agent fit for autonomy level $\mathrm { L } _ { \mathrm { k } }$ on this class of task?). Harness benchmarks such as TraceTrial-Bench [27] and HarnessAudit [28] supply the task and safety side; evidence-gated lifecycle control [18, 19] supplies the enforcement side. Soei supplies the record both consume.

![](images/88b58db3d89a235449a8a47cbc4609abdd091e612538d057d87e072a042de8f9.jpg)  
autonomy decisions feed back to which tasks an agent may run unattended

Figure 17. Golden harness built on Soei. Observation (Soei core) stays read-only, as required by P1. Any enforcement lives in a separate policy plane that consumes Soei verdicts, so it remains an explicit, auditable and reversible decision.

## 13.3 Harness metrics with real-data baselines

Table 11. Golden-harness metrics. Baselines are measured on our 45-session corpus (human in the loop); targets are proposals for granting autonomy.

<table><tr><td>Metric</td><td>Definition</td><td>Pilot baseline Proposed L3</td><td>target</td></tr><tr><td>UHR Unverified hand-back rate</td><td>Code-changing episodes ending with no proof after the last edit</td><td>23.4% [17.1, 35.7] ≤ 2%</td><td></td></tr><tr><td>CP Claim precision</td><td>Completed claims that are Verified (1 – unproved completion rate)</td><td>42.2% (136/322) ≥ 95%</td><td></td></tr><tr><td>VCR Verified completion rate</td><td>Tasks reaching Verified with a passing oracle</td><td>needs oracle outcomes task-class specific</td><td></td></tr><tr><td>PR Phantom rate</td><td>Mutations not acknowledged</td><td>1.1% (shown correctly by 0 rendered as</td><td>Soei) applied</td></tr><tr><td>LC Lineage coverage</td><td>Cross-session reuse explained by H2 lineage</td><td>24/24 pairs (path level) digest level, 100%</td><td></td></tr><tr><td>SLR Secret leak rate</td><td>Credential matches in persisted or exported records</td><td>raw 1,326 → Soei 0 0</td><td></td></tr><tr><td>TTA Time to awareness</td><td>Time for a supervisor to answer “what is done and verified?"</td><td>195 long-idle returns user study (RQ5)</td><td></td></tr><tr><td>RD Replay determinism</td><td>Identical projections on replay</td><td></td><td>√√ always</td></tr></table>

## 13.4 Autonomy levels and evidence gates

Table 12. Proposed autonomy levels for coding agents and the Soei evidence each requires before the level is granted.
<table><tr><td>Level</td><td>Agent may...</td><td>Human role</td><td>Required Soei evidence (gate)</td></tr><tr><td>L0 Suggest</td><td>propose code only</td><td>applies every change</td><td>none (capture optional)</td></tr><tr><td>L1 Assisted edit</td><td>edit files in a workspace</td><td>approves each episode</td><td>ack-correct capture; UHR reported to the human at hand- back</td></tr><tr><td>L2 Verified edit</td><td>edit and hand back without per-step approval</td><td>reviews Unverified/Contested only</td><td>every claim Verified or explicitly flagged; UHR ≤ target; SLR = 0</td></tr><tr><td>L3 Multi-agent autonomy</td><td>run sessions and sub-agents that build on each other</td><td>samples audits</td><td>digest lineage for every hand-off; CP ≥ target on the golden set; contested-claim alerts</td></tr><tr><td>L4 Unattended delivery</td><td>merge and deploy within policy</td><td>sets policy, reviews audit</td><td>signed provenance attestation per artifact; oracle-verified VCR; replayable ledger; automatic revocation on regression</td></tr></table>

The gates make autonomy something an agent earns and can lose. An agent is granted $\mathrm { L } _ { \mathrm { k } }$ for a task class when its golden-harness scorecard meets the gate. It is automatically demoted when replayed regression runs or production evidence push a metric past threshold, for example a model upgrade that raises UHR.

## 13.5 Alignment with governance frameworks

Table 13. How Soei evidence supports common AI-governance obligations (indicative mapping, not legal advice).
<table><tr><td>Framework</td><td>Obligation (summary)</td><td>Soei support</td></tr><tr><td>NIST AI RMF 1.0 [13]</td><td>GOVERN, MAP, MEASURE, MANAGE functions; measurable, documented risk management</td><td>Harness metrics (MEASURE), autonomy gates (MANAGE), ledger-backed records (GOVERN)</td></tr><tr><td>ISO/IEC 42001:2023 [14]</td><td>AI management system: operational control, monitoring, internal audit, continual improvement</td><td>Tamper-evident monitoring record; replay for audit; regression alarms for improvement</td></tr><tr><td>EU AI Act, Art. 12 [15]</td><td>Automatic recording of events (logging) over the lifetime of high-risk systems</td><td>Append-only, hash-chained, policy-tagged event log with manifests</td></tr><tr><td>EU AI Act, Art. 14 [15]</td><td>Effective human oversight, including understanding outputs and deciding not to use them</td><td>Claim vs proof display, Contested alerts, attention ranking, explainable support links</td></tr><tr><td>Supply-chain provenance (SLSA / in-toto) [16, 17]</td><td>Verifiable provenance for build artifacts</td><td>Planned: export Soei lineage and proof as attestations for agent-produced commits</td></tr></table>

Why it cannot be retrofitted. Governance evidence has to exist at the time of the action. History that was not captured, was captured unacknowledged, was bounded away, or was polluted with secrets cannot be reconstructed for an audit later. Our measurements show all four failure modes in real logs (§10.4–10.6). The substrate has to be in place before autonomy is granted, not added afterwards.

## 14 Scope of improvements

The prototype shows the core is feasible and cheap. The improvements below are what remain between it and a production golden harness, ranked by governance impact. Each lists its current state, the change, and why it is required.

Table 14. Prioritised improvement backlog.
<table><tr><td>#</td><td>Improvement</td><td>Current state (prototype)</td><td>Change</td><td>Why required</td><td>Phase</td></tr><tr><td>1</td><td>Proof outcomes</td><td>Detects that a check ran, not whether it passed</td><td>Capture exit codes, test reports (JUnit/pytest), HTTP status, UI assertions; enable the Contested state</td><td>A failed check is currently counted as proof; L2+ gates need pass/fail</td><td>1</td></tr><tr><td>2</td><td>Digest-level lineage</td><td>Path-hash identity</td><td>Content digests (BLAKE3/Git blob) at every write and read</td><td>Path reuse after overwrite can create false hand-offs; L3 accountability needs exact versions</td><td>1</td></tr><tr><td>3</td><td>Learned proof classifier</td><td>Rules v4, audited precision 83.3% / 96.7%</td><td>Train on audited labels; per-class calibration; abstain to &quot;Unknown&quot;; evaluate trajectory verifiers such as SWE-Critix [31] as drop-in classifiers</td><td>Gate decisions need ≥95% precision with calibrated uncertainty</td><td>2</td></tr><tr><td>4</td><td>Multi-agent adapters</td><td>VS Code Copilot agent only</td><td>Codex, Claude Code, Cursor, MCP servers, CI systems, with conformance corpora</td><td>Fleets mix vendors; one- vendor governance has blind spots</td><td>1-2</td></tr><tr><td>5</td><td>Golden task corpus + oracles</td><td>None</td><td>Curated, versioned tasks from real work with hidden oracle tests (incl. post- silicon validation tasks)</td><td>Certification needs ground truth, not self-reported success</td><td>2</td></tr><tr><td>6</td><td>Policy plane</td><td>Not present (core is read- only by design)</td><td>Separate, opt-in plane: advisory → enforcing merge/deploy gates from Soei verdicts; signed policy files</td><td>Turns evidence into governance; keeps observation and control separable</td><td>3</td></tr><tr><td>7</td><td>Provenance attestations</td><td>Not present</td><td>Emit SLSA/in-toto-style statements linking commits to lineage, proof and agent identity</td><td>Downstream consumers and auditors can verify agent- made changes</td><td>3</td></tr><tr><td>8</td><td>Secret defence in depth</td><td>Pattern-based scan, gate before persistence</td><td>Entropy and context detectors, provider- side redaction hooks, live alerts on exposure</td><td>1,326 raw records exposed in 16/45 sessions</td><td>1</td></tr><tr><td>9</td><td>Causal ordering</td><td>Wall clock + source offsets</td><td>Hybrid logical clocks across sources and machines</td><td>Remote and sub-agent runs break wall-clock order</td><td>2</td></tr><tr><td></td><td>10 Plan-aware attribution and drift</td><td>Not evaluated</td><td>Map artifacts to plan-file components; calibrated attribution; drift metrics (Eq.2)</td><td>Supervisors reason in components and deliverables, not files</td><td>2</td></tr><tr><td>11</td><td>Differential replay</td><td>Determinism verified</td><td>Replay identical tasks across model/scaffold versions; regression alarms</td><td>Model upgrades silently change behaviour; autonomy must be revocable</td><td>2</td></tr><tr><td></td><td>12 Supervisor UX</td><td>No UI yet</td><td>Graph/role/timeline projections with Verified/Unverified/Contested, attention queue, &quot;why&quot; links</td><td>Oversight only works if evidence reaches the human quickly (RQ5)</td><td>3</td></tr><tr><td></td><td>13 Team federation</td><td>Single machine</td><td>Opt-in aggregation of metrics (not content) across a team; privacy-</td><td>Governance and benchmarking happen at team and org level</td><td>4</td></tr></table>

![](images/b83d9615fac15de8fcc59997b6720f50166617cf5bc6a149824e611cdbda59f6.jpg)  
Figure 18. Improvement roadmap and the autonomy level each phase is intended to unlock.

## 15 Conclusion

Local, project-level observability of coding agents is feasible, and developers want it. Our study of 45 real sessions shows that visibility on its own is not enough. About a quarter of code-changing interactions end unverified. More than half of agent-completed, edit-backed to-dos have no proof. Timing-based hand-off detection sees a quarter of real collaboration. Bounded history drops the long sessions that hold half the evidence. Raw agent logs routinely contain credentials. Soei responds with one principle at each layer: acknowledge before showing, persist evidence before inferring, scrub before storing, prove before calling something done, and derive every view from one tamper-evident record. The prototype shows that this costs about 20% of the raw log size and fractions of a second on real data. The same substrate is what a golden harness and the governance of autonomous agents require: autonomy that is granted on measured evidence, verified per claim, attributable per artifact, auditable after the fact, and revoked automatically when the evidence degrades.

## Appendix A Soei Event Schema examples

```jsonl
{ "specversion": "soei/0.1", "id": "01JC9Z6Q8W3T0E6M8X2D5V4K7A", "type": "artifact.generated",
"source": "adapter:copilot-vscode/0.56#transcripts/<session>.jsonl", "source_offset": 1842,
"hlc": { "wall": "2026-10-07T11:43:48.454Z", "logical": 0, "node": "src:copilot:<s>" },
"evidence_class": "observed",
"ack": { "state": "acknowledged", "call": "c1f0...", "result_ts": "2026-10-07T11:43:48.920Z" },
"subject": { "session": "9a3e...", "artifacts": ["path#5b1d..."], "code": true },
"policy": { "profile": "default-local", "dropped": ["message", "arguments.content"], "hashed": ["path"] },
"prev": "a052e782...", "hash": "7c19b0e4..." }
{ "specversion": "soei/0.1", "type": "proof", "evidence_class": "observed", "proof": "test",
"ack": { "state": "acknowledged" }, "subject": { "session": "9a3e...", "activity": "4410..." } }
{ "specversion": "soei/0.1", "type": "task.update", "evidence_class": "reported",
"ack": { "state": "acknowledged" }, "tasks": [ { "id": "t#e21c...", "status": "completed" } ] }
```

Appendix B Gap-to-requirement traceability
<table><tr><td>Gap</td><td>Requirement</td><td>Soei component</td><td>Evidence / evaluation</td></tr><tr><td>G1/G2</td><td>Single backend, precedence, namespaced hooks</td><td>L1-L2, L6</td><td>Replay determinism √; RQ4</td></tr><tr><td>G3</td><td>Formal evidence, calibration</td><td>§6, L4</td><td>Audited precision; RQ1</td></tr><tr><td>G4</td><td>Acknowledgement</td><td>Ack tracker</td><td>1.1% phantom edits (§10.5)</td></tr><tr><td>G5</td><td>Claim versus proof</td><td>L5</td><td>23.4% / 57.8% (§10.1); RQ3</td></tr><tr><td>G6</td><td>Graded lineage hand-offs</td><td>L4 lineage</td><td>Recall 25% for H0 (§10.3); RQ2</td></tr><tr><td>G7</td><td>Measured attribution</td><td>L4 attribution</td><td>RQ1</td></tr><tr><td>G8</td><td>Durable history</td><td>L2 ledger, checkpoints</td><td>48.1% bytes beyond bound (§10.4)</td></tr><tr><td>G9</td><td>Causal order</td><td>HLC, offsets</td><td>RQ4</td></tr><tr><td>G10</td><td>Security, privacy</td><td>Gate, token, Origin/Host</td><td>1,326 → 0 (§10.6)</td></tr><tr><td>G11-G13</td><td>Adapters, interop, attention</td><td>SDK, exporters, L4</td><td>Windows prototype run; 195 idle returns</td></tr><tr><td>G14</td><td>Evaluability</td><td>Harness</td><td>This paper</td></tr></table>

Appendix C Proof classifier rules (v4, abridged)
<table><tr><td>Class</td><td>Rule (applied to acknowledged actions after the last code edit)</td></tr><tr><td>static</td><td>Editor diagnostics; py_compile, compileal1, ruff/flake8/pylint/mypy, tsc, eslint, node --check, JSON validation</td></tr><tr><td>build</td><td>npm run build|compile|lint|typecheck, dotnet/msbuild/cmake/make/gradle/mvn/cargo/go build, vsce package, choco/nuget/npm pack, pip wheel</td></tr><tr><td>test</td><td>pytest, unittest, npm test, go/cargo/dotnet/mvn/gradle test, ctest; scripts named test*/verify*/check*/smoke*</td></tr><tr><td>exec</td><td>Interpreter runs the edited script or a script in the edited directory; &amp;/-File runs an edited . ps1; notebook cell runs. Excludes invocations followed by sync/put/deploy/upload/scp/copy</td></tr><tr><td></td><td>smoke python -c with import of project modules</td></tr><tr><td></td><td>probe Invoke-WebRequest/RestMethod or curl to 127.0.0.1/localhost</td></tr><tr><td>ui</td><td>Browser page read, navigation, screenshot or scripted checks after an edit in the episode</td></tr></table>

## Appendix D Audit record

<table><tr><td>Version</td><td>Label</td><td>Seed</td><td></td><td>Correct / n Typical errors</td></tr><tr><td>v1</td><td>unverified</td><td>7</td><td></td><td>20 / 30 build commands, quoted interpreter paths, import smoke checks, _tmp_test scripts</td></tr><tr><td>Version</td><td>Label</td><td>Seed</td><td>Correct / n Typical errors</td><td></td></tr><tr><td>v3</td><td>unverified</td><td>41</td><td></td><td>22 / 30 Playwright UI checks, localhost HTTP probes, &amp;-invoked . ps1, choco pack</td></tr><tr><td>v4</td><td>unverified</td><td>59</td><td>25 /30</td><td>log-tail inspection, grep-based rename check, terminal output review, subagent review</td></tr><tr><td>v4</td><td>verified</td><td>59</td><td></td><td>29 / 30 scp deploy read as execution</td></tr></table>

## Appendix E Reproducibility

research/soei/prototype/   
soei\_pilot.py # adapter + SES + ack + privacy gate + hash-chained ledger + fold + all metrics (stdlib   
only)   
audit\_sample.py # seeded blind audit sampler: python audit\_sample.py {unverified|verified} <seed>   
make\_figures.py # all data figures (matplotlib) + bias-corrected estimate   
results\_frozen.json # aggregate results cited in this paper   
audit\_results.json # audit outcomes per version   
python soei\_pilot.py --out results.json --ledger ledger.jsonl && python make\_figures.py

The corpus itself is not released: it is the authors’ private working history and contains secrets (§10.6). The scripts run unchanged on any VS Code Copilot agent transcript directory. Other agents need an adapter conforming to §5.2.

## References

1. C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, K. Narasimhan. “SWE-bench: Can Language Models Resolve Real-World GitHub Issues?” ICLR, 2024.

2. J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, O. Press. “SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.” NeurIPS, 2024.

3. S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, Y. Cao. “ReAct: Synergizing Reasoning and Acting in Language Models.” ICLR, 2023.

4. M. R. Endsley. “Toward a Theory of Situation Awareness in Dynamic Systems.” Human Factors, 37(1):32–64, 1995.

5. OpenTelemetry. “Semantic Conventions for Generative AI Systems.” opentelemetry.io (development status).

6. B. H. Sigelman et al. “Dapper, a Large-Scale Distributed Systems Tracing Infrastructure.” Google Technical Report, 2010.

7. L. Lamport. “Time, Clocks, and the Ordering of Events in a Distributed System.” CACM 21(7):558–565, 1978.

8. W3C. “PROV-DM: The PROV Data Model.” W3C Recommendation, 2013.

9. B. Shneiderman. “The Eyes Have It: A Task by Data Type Taxonomy for Information Visualizations.” IEEE VL, 1996.

10. CNCF. “CloudEvents Specification v1.0.” cloudevents.io.

11. S. S. Kulkarni, M. Demirbas, D. Madappa, B. Avva, M. Leone. “Logical Physical Clocks.” OPODIS, 2014.

12. E. B. Wilson. “Probable Inference, the Law of Succession, and Statistical Inference.” JASA 22(158):209–212, 1927.

13. NIST. “Artificial Intelligence Risk Management Framework (AI RMF 1.0).” NIST AI 100-1, January 2023.

14. ISO/IEC 42001:2023. “Information technology — Artificial intelligence — Management system.”

15. Regulation (EU) 2024/1689 (Artificial Intelligence Act), Articles 12 (Record-keeping) and 14 (Human oversight). Official Journal of the EU, 2024.

16. OpenSSF. “Supply-chain Levels for Software Artifacts (SLSA) v1.0.” slsa.dev.

17. S. Torres-Arias, H. Afzali, T. K. Kuppusamy, R. Curtmola, J. Cappos. “in-toto: Providing farm-to-table guarantees for bits and bytes.” USENIX Security, 2019.

18. J. Huang, J. Hsia, J. Sun et al. “Proof-or-Stop: Don’t Trust the Agent, Trust the Evidence – Loop Engineering for Verifiable Evidence-Gated Lifecycle Control.” arXiv:2607.14890, 2026.

19. A. W. Ng, Y. Han, J. Zhang et al. “Agent Safety Should Be a Runtime Contract.” arXiv:2608.11274, 2026.

20. X. Yin, M. Du, M. H. Prince et al. “Artifact-centered Claim-aware Observability for Autonomous Scientific Agents.” arXiv:2608.18312, 2026.

21. F. Wen, Z. Wang, R. Xu. “OpenRath: Session-Centered Runtime State for Agent Systems.” arXiv:2606.19409, 2026.

22. Y. Nian, A. Yuan, H. Zhang et al. “Auditable Agents.” arXiv:2604.05485, 2026.

23. A. AlSayyad, K. Y. Huang, R. Pal. “AgentTrace: A Structured Logging Framework for Agent System Observability.” arXiv:2602.10133, 2026.

24. X. Gao. “Evidence-Calibrated Runtime Reconstruction for Agent Skills Across Heterogeneous Coding Agents.” arXiv:2608.08793, 2026.

25. P. Sahoo, G. Mittal, X. Li et al. “AgentLens: Revealing the Lucky Pass Problem in SWE-Agent Evaluation.” arXiv:2605.12925, 2026.

26. Z. Chen, Y. Zhang, Y. Liu et al. “How Your Credentials Are Leaked by LLM Agent Skills: An Empirical Study.” arXiv:2604.03070, 2026.

27. Z. Cao, H. Zhu, Y. Wang et al. “Trial by Trace: Benchmarking AI Agent Harnesses for Reliability and Failure Attribution.” arXiv:2610.32285, 2026.

28. C. Liu, Y. Guo, Y. Liu et al. “Auditing Agent Harness Safety.” arXiv:2605.14271, 2026.

29. J. Gao, L. Li, S. Guo et al. “Making Agent-Mediated Contributions Governable: A Project-Level Governance Manifest for Open-Source AI Collaboration.” arXiv:2607.15769, 2026.

30. T. Bula, S. Pujar, L. Buratti et al. “SeaView: Software Engineering Agent Visual Interface for Enhanced Workflow.” arXiv:2504.08696, 2025.

32. M. Rashidi. “The Balkanization of Execution-Security Research for AI Coding Agents: Isolation, Access Control, and Time-of-Checkto-Time-of-Use Vulnerabilities.” arXiv:2607.05743, 2026.

31. J. Hu, C. Y. Chong. “SWE-Critix: Post-Training Generalist Verifiers for Assessing Coding-Agent Trajectories.” arXiv:2610.13356, 2026.