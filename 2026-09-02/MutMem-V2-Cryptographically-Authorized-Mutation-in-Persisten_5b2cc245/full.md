# MutMem V2: Cryptographically Authorized Mutation in Persistent Agent Memory Portable Verification and Reproducible Evidence

Walid Saidi Independent researcher

Evidence-audited manuscript candidate, September 2026

Reproducibility and release status. Repository: https://github.com/wallidsaydi-creator/HOM-AIMOS. Published predecessor: arXiv:2608.02843. The evidence source commit is eb0166fd45964ebb47ca053bcfeb0ab3ddc68999, and the closed publication-evidence root is 99bed29dcf0cc5c2f597f17b69be8d6dc990bae4ae5dd30838a1d1f34d92a76b. The public release is https://github.com/wallidsaydi-creator/HOM-AIMOS/releases/tag/v1.0.0, immutable tag v1.0.0, commit fcda26d1e05c729185c09dd11446db2f79c14a0d, with source-manifest root f0cdb75763c70ce83bfd1e5f 6c956e7502b65a812c3a43f311f8039ba0c3b7f1.

Ofline verification is exactly npm run verify; deterministic evidence regeneration is exactly npm run evidence:regenerate. For an ordinary agent identifier stored in shell variable AGENT\_ID after installer onboarding, the exact multi-line utility reproduction command is:

npm run reproduce -- --installed-instance canonical \

--agent-id "\$AGENT\_ID" --full --benchmark both \

--protocol canonical-blind-v1

Source code is AGPL-3.0-or-later. The repository license applies to AIMOS-authored artifact files. Dataset text is not redistributed; downloaded materials retain upstream licenses. Historical V1 utility used GPT-5.4 for generation and GPT-5.6 Terra for judgment; the historical poisoning lane used GPT-5.5 and GPT-5.6 Terra. Provider access is operator-authenticated, and no credential or provider payload is released.

Highest support level: the protocol and evidence are independently verifiable, and one clean installation is qualified; the empirical findings have not been independently replicated. The immutable public tag, release commit, GitHub Release, and source-manifest root are mutually bound. This manuscript is publication-authorized for arXiv submission; the arXiv identifier is necessarily recorded after submission acceptance.

## Abstract

MutMem V1, published as “MutMem: Cryptographically Authorized Mutation in Persistent Agent Memory,” introduced retention-preserving, cryptographically authorized mutation for persistent agent memory and reported utility, mutation-integrity, and targeted-poisoning results. Its principal publication gap was not another retrieval mechanism: it was the absence of a complete portable verification contract and a reviewer-facing path from a clean installation to independently checkable evidence. MutMem V2 closes that gap without creating a second memory engine. It specifies exact canonical bytes, domain-separated object and bundle commitments, mandatory recall-evidence membership and order, external trust anchors, identity epochs, revocation state, authorization, request receipts, ordered disclosure, and three mutation terminals. The current protocol contains 18 versioned object schemas, 39 recall predicate vectors, 15 mutation vectors, and 37 closed recall failure reasons. Independent Node and Python implementations agree on verdict and primary reason for all 72 structural and cryptographic terminals; a separate production-conformance corpus agrees on 42/42 cases spanning 28 required classes. A clean Node v26.8.1 installation reaches first-boot, restart, and scheduler readiness with no experimental memories. A separately scoped Canary experiment completes 120 explicit-marker and clean units; it is evidence of declared marker traversal only. Every public table is regenerated from a self-hashed aggregate, and an independent verifier reconstructs its statistics and claim boundaries. Historical V1 empirical results remain historical rather than being relabeled as V2 reruns. MutMem V2 supports claims about portable integrity, authorization, traceability, conformance, and reproducibility under stated assumptions. It does not establish semantic truth, universal robustness, or independent replication.

## 1 Introduction

Persistent memory creates two separable research problems. The first is how an agent may adapt retained evidence while preserving authorship, prior state, and the reason for change. MutMem V1 addressed that problem with append-only outcome evidence, bounded bidirectional weight transitions, signed provenance, and ordered recall receipts [1]. The second problem is how a reviewer can verify those claims without trusting the production database, its runtime, or a prose description of the system. A self-consistent implementation is not yet a portable protocol: its object boundaries, byte encodings, trust roots, failure reasons, result cardinality, and terminal states must be explicit.

MutMem V2 addresses the second problem. MutMem denotes the portable protocol; HOM-AIMOS is one producer and case study. The portable verifier does not select memories, authorize requests, sign events, mutate storage, or operate a second runtime. It consumes finite evidence objects and returns a deterministic terminal verdict. [V2-PORTABLE-PROTOCOL]

The paper makes four contributions:

1. It defines a versioned recall-disclosure envelope whose complete membership is 13 + 5r objects for r disclosed results, with deterministic ordering, exact canonical bytes, and domain-separated SHA-256 commitments.

2. It defines cross-object predicates for external trust, actor and Housekeeper epochs, revocation, efective authorization, signed request and receipt parity, native decision bindings, per-result provenance and occurrence evidence, Merkle order, and signed terminal receipts.

3. It defines a portable mutation profile that preserves the native outcome schema and distinguishes an authorized transition, a signed no-op, and an occurrence observation.

4. It supplies independent Node and Python verifiers, a production-derived conformance corpus, a clean installer qualification, deterministic publication regeneration, an attempt ledger, and a complete claim-to-evidence map.

The contribution is deliberately narrower than “secure memory.” Cryptographic integrity does not imply that remembered content is true. A fixed poisoning experiment does not imply universal detection. Exact verifier parity does not imply independent empirical replication. These are protocol boundaries, not disclaimers added after observing results. <sup>[CONTENT-TRUTH]</sup> <sup>[UNIVERSAL-DEFENSE][INDEPENDENT-REPLICATION]</sup>

## 2 Evidence identities and support levels

We use three non-interchangeable evidence labels:

• current V2 denotes evidence produced or reverified for the portable V2 protocol, independent verifiers, conformance corpus, explicit-marker Canary lane, or clean installer.

• historical V1 denotes immutable empirical evidence released with MutMem V1. It may be cited as a historical comparator but is not a current V2 benchmark run.

• boundary denotes an omitted or prohibited claim. Missing evidence is never converted into a zero, success, or defense.

The manuscript binds the claim map 4d8d23442e9ed311e33ddd9ce9a79e878642eba93329c460e203128e54997527 and independent verification f38268c86f2291971ac6ab75ded944d9a7e7eb6fa2809a5367d69dacd450ffe0. The claim map names the code, protocol, artifact, verifier, denominator, and limitation for every quantitative or security statement. The evidence package contains 7 deterministic files including its manifest; the manuscript assets are generated separately from that root.

## 3 System and threat model

## 3.1 Principals and boundary

The production producer has an external master trust anchor, an ordinary actor or the Housekeeper system principal, append-only identity epochs and revocation state, a signed authorization projection, a canonical SAVE path, a canonical RECALL path, and restricted database-local writers. The portable verifier is an evidence consumer. It neither possesses producer credentials nor infers a trust root from the bundle it is asked to verify.

For ordinary actors, the evidence bundle must bind the exact company, actor identifier, identity epoch, certificate fingerprint, unrevoked state, efective grant, requested clearance, data class, method, path, canonical request body, nonce, and signing time. For the Housekeeper, the system-principal profile is structurally distinct from a mastersigned ordinary grant; signature-bearing grant fields are forbidden in that profile. This separation prevents a system principal from masquerading as an ordinary master-signed grant and prevents an ordinary identity from inheriting Housekeeper authority.

## 3.2 Adversary

The verifier is designed to detect malformed canonical objects, object or bundle substitution, wrong trust roots, identity or epoch substitution, stale or revoked authority, request replay or route mismatch, result omission or reordering, provenance and occurrence substitution, Merkle mismatch, terminal-receipt drift, and mutation-topology inconsistency. The adversary may control transport and may modify copied evidence. The adversary may not break SHA-256 collision resistance or Ed25519 existential unforgeability, and the reviewer must obtain the expected master fingerprint and release identity through an external channel.

An attacker who replaces both the evidence and every external trust anchor can present a diferent self-consistent world. MutMem does not solve that anchor distribution problem. It also does not prove availability, confidentiality, semantic truth, model honesty, or protection against every form of memory poisoning.

## 4 Portable recall-disclosure protocol

## 4.1 Canonical objects

Let H = SHA256, let $J ( x )$ be the protocol’s deterministic canonical-JSON bytes, let $\operatorname { L P } _ { 4 } ( x )$ prepend the unsigned four-byte big-endian byte length of UTF-8 string x, and let $\mathrm { B E } _ { k } ( x )$ encode integer x in k big-endian bytes. Every variable-length field is either canonical JSON or explicitly length-framed. The object domain $D _ { o }$ is the UTF-8 byte string hom.aimos.mutmem-portable-object/v2 followed by a zero byte. For kind $k _ { j }$ , schema $s _ { j }$ , and body $b _ { j }$ , the object commitment is

$$
o _ { j } = H ( D _ { o } \parallel \mathrm { L P } _ { 4 } ( k _ { j } ) \parallel \mathrm { L P } _ { 4 } ( s _ { j } ) \parallel \mathrm { B E } _ { 4 } ( | J ( b _ { j } ) | ) \parallel J ( b _ { j } ) ) .\tag{1}
$$

Bodies reject unsupported or non-finite numbers, unsafe integers, nesting beyond the protocol limit, and bodies larger than the declared maximum. Object schemas are exact, not advisory labels. These choices follow the prefix-free framing and domain-separation discipline used in cryptographic protocol design [6] and the deterministic JSON discipline standardized by JCS [5].

## 4.2 Complete membership and deterministic order

Each recall bundle contains thirteen singleton objects: trust anchor; actor identity and revocation; Housekeeper identity and revocation; efective grant; request envelope; request receipt; content-state decision; epistemic decision; final security closure; return projection; and native recall receipt. Each disclosed result contributes five objects: memory state, provenance chain, occurrence, epistemic projection, and receipt evidence. Therefore, for r results,

$$
N _ { \mathrm { o b j e c t s } } ( r ) = 1 3 + 5 r , \qquad 0 \leq r \leq 2 0 0 .\tag{2}
$$

Every singleton kind occurs exactly once. Every result ordinal from 0 through r − 1 contains all five result kinds and one subject identifier shared across the group. Objects are ordered first by the fixed singleton order and then by result ordinal and fixed result-kind order. Duplicate, missing, mixed-subject, or out-of-range objects fail before relational evaluation.

The ordered object descriptors are committed with the RFC 6962 construction [3]. If $x _ { i }$ denotes canonical bytes for descriptor i,

$$
L _ { i } = H ( 0 \mathbf { x } 0 0 \parallel x _ { i } ) ,\tag{3}
$$

$$
T ( a , b ) = H ( 0 \mathbf { x } 0 1 \parallel a \parallel b ) ,\tag{4}
$$

with an empty root $H ( \epsilon )$ and recursive split at the largest power of two strictly smaller than the current list length. Leaf/node domain separation prevents a node encoding from being confused with a leaf encoding under the hash assumption.

## 4.3 Envelope commitment

Let $D _ { b }$ be hom.aimos.mutmem-portable-evidence/v2 followed by a zero byte, u the bundle identifier, c the company identifier, f the externally expected 32-byte master fingerprint, r the result count, and R the object root. The bundle commitment is

$$
B = H ( D _ { b } \parallel \mathrm { L P } _ { 4 } ( u ) \parallel \mathrm { L P } _ { 4 } ( c ) \parallel f \parallel \mathrm { B E } _ { 8 } ( r ) \parallel R ) .\tag{5}
$$

The master fingerprint is an input to verification; accepting the fingerprint solely because it appears inside the bundle would be circular trust. The structural predicate evaluator therefore reports that cryptographic signatures and external trust remain unverified. Independent verifiers must check them.

## 4.4 Cross-object predicates

The recall predicate graph enforces exact equality across five boundaries:

1. Authority: trust anchor, actor epoch, certificate, revocation, Housekeeper epoch, authorization, and request time agree.

2. Admission: signed method and path are POST and /aimos/recall; request-body, normalized-command, request-receipt, nonce, signature, and authorization mutation commitments agree.

3. Decision composition: content-state roots, epistemic decision, final security closure, return path, projected identifiers, and content hashes agree, with no canonical-memory or retention mutation.

4. Per-result evidence: memory, provenance, occurrence, epistemic, and receipt objects agree on subject, live-content hash, save and binding mutations, occurrence reference, ordinal, and selected membership.

5. Terminal receipt: Merkle entries, root, result count, actor, request, authority, Housekeeper certificate, event content hash, predecessor, nonce, signing time, signature, and revocation evaluation agree.

The failure vocabulary is closed: an undeclared failure reason is itself an error. This makes verdict/reason parity testable across implementations instead of permitting semantically similar but incomparable exceptions.

## 5 Portable mutation profile

MutMem V2 does not introduce a new cognitive update rule. It preserves the native V1 mutation-outcome schema and makes its evidence portable. Let $D _ { \mu }$ be hom.aimos.mutmem-portable-mutation-evidence/v2 followed by a zero byte, and let M be the canonical body containing the recall commitment, outcome evidence, recall receipt, outcome event, signed valence evidence, terminal, and optional cognitive projection. The portable mutation commitment is

$$
B _ { \mu } = H ( D _ { \mu } \parallel J ( M ) ) .\tag{6}
$$

Three terminals are exhaustive:

• authorized\_transition: a principal-state outcome binds a terminal REWEIGHT provenance node and a nontrivial cognitive projection;

• signed\_noop: the signed outcome is retained, but no projection is fabricated when the quantized weight is unchanged; and

• occurrence\_observation: occurrence-scoped evidence is retained with no weight authority.

For an authorized transition, the verifier reconstructs the V1 projection and transition hashes from the memory identifier, integer milliscale old/new weights, provenance mutation, predecessor projection, and company. It requires the retained Ed25519 signature and exact signer epoch; Ed25519 itself is a standard primitive, not a MutMem contribution [4]. <sup>[V1-MUTATION-INTEGRITY]</sup>

## 6 Independent verification

## 6.1 Two implementations, one terminal language

The production-language reference and independent Python implementation consume the same public vectors but do not share verifier code. Four lanes cover recall structure, mutation structure, recall cryptography, and mutation cryptography. Across all 72 vectors, both implementations match the expected valid/invalid terminal and exact primary reason. This count is not a benchmark accuracy denominator: it is a conformance denominator. [V2-INDEPENDENT-VERIFIER]

The structural vector sets contain valid bundles and one negative vector for each declared recall or mutation failure family. Cryptographic vectors add signature, certificate, epoch, revocation, and trust-anchor checks. The verification implementation is read-only and has no runtime importer, database, network, signing, model, or policy authority.

## 6.2 Production-derived conformance corpus

A separate corpus projects production-shaped recall evidence into public test objects and evaluates it with the production protocol owners and an independent verifier. All 42 intended cases are observed, cover 28 required classes, and agree on terminal verdict and protocol reason. This establishes production/protocol conformance for the released corpus. It does not establish retrieval quality or robustness against unrepresented attacks. <sup>[V2-PRODUCTION-CONFORMANCE]</sup>

## 6.3 Complexity and failure closure

For $n = 1 3 + 5 r$ objects, membership construction uses maps keyed by singleton kind and result ordinal, then performs one fixed-order traversal in $O ( n )$ time. The Merkle tree contains n leaf hashes and n − 1 internal hashes, hence $O ( n )$ hash evaluations. The current JavaScript reference uses recursive array slices; accounting for those copies gives O(n log n) implementation time in the worst case, O(n) retained tree state, and $O ( \log n )$ recursion depth, with $r \leq 2 0 0$ . This bound is preferable to describing the hash-node count as the complete runtime cost. The failure language is finite and versioned; new predicates require a new vector and protocol root rather than an unrecorded exception.

## 7 Reproducibility architecture

## 7.1 Verification, installation, and reproduction are diferent

The ofline verification command checks retained public artifacts and executes the independent Node/Python vector suites without opening AIMOS or a provider. Evidence regeneration independently rebuilds publication tables, the attempt ledger, disclosure boundary, and claim map. Neither command writes memory.

The installer is a separate product operation. The qualified run used Node v26.8.1, completed first boot and restart, registered the required scheduler, retained 8 Guide memories and 0 experimental memories, and capped the ordinary enrolled agent at clearance 10. This is one clean-install qualification, not cross-platform coverage or a scientific benchmark. <sup>[V2-CLEAN-INSTALLER]</sup>

Full reproduction is intentionally distinct from verification. It uses the installer-selected ordinary agent, canonical signed SAVE and RECALL, fixed dataset locks, fixed model roles, complete terminal evidence, and the integer denominator invariant

$$
n _ { \mathrm { i n t e n d e d } } = n _ { \mathrm { s e l e c t e d } } = n _ { \mathrm { c o m p l e t e d } } = n _ { \mathrm { e v a l u a t e d } } , \qquad n _ { \mathrm { i n c o m p l e t e } } = n _ { \mathrm { f a i l e d } } = 0 .\tag{7}
$$

A mutable status flag is insuficient. The terminal binds the selected immutable aggregate path and hash, required phase summaries, environment identity, and the exact requested benchmark set. A stale predecessor cannot be substituted for its completed immutable successor.

## 7.2 Datasets, models, and environment

The public package carries corpus manifests, source locks, target locks, downloaders, and hashes. It does not redistribute LongMemEval, LoCoMo, Natural Questions/BEIR, or PoisonedRAG text. Historical V1 model identities are retained beside their results. Credentials, provider requests and responses, benchmark text, memory identifiers, certificates, and machine paths are outside the public evidence boundary.

Each new run must record operating system, architecture, CPU, memory, Node binary, package and lock hashes, dependency versions, PostgreSQL and extensions, concurrency, protocol configuration, model identities, native surfaces, and start/capture times. This environment record is descriptive evidence, never identity or configuration authority.

## 8 Statistical contract

For a binary outcome with x successes in n intended units, the estimate is ${ \hat { p } } = x / n$ . The two-sided 95% Wilson interval uses z = 1.959963984540054:

$$
\frac { \hat { p } + z ^ { 2 } / ( 2 n ) \pm z \sqrt { \hat { p } ( 1 - \hat { p } ) / n + z ^ { 2 } / ( 4 n ^ { 2 } ) } } { 1 + z ^ { 2 } / n } .\tag{8}
$$

For paired binary outcomes with discordant counts b and $^ { c , }$ the exact two-sided McNemar value is

$$
p _ { \mathrm { M c N } } = \operatorname* { m i n } \left( 1 , 2 \sum _ { k = 0 } ^ { \operatorname* { m i n } ( b , c ) } { \binom { b + c } { k } } 2 ^ { - ( b + c ) } \right) .\tag{9}
$$

Within each declared family of m contrasts, raw values are ordered $p _ { ( 1 ) } \leq \dots \leq p _ { ( m ) }$ and Holm adjustment is

$$
\widetilde { p } _ { ( i ) } = \operatorname* { m a x } _ { 1 \leq j \leq i } \operatorname* { m i n } \bigl ( 1 , ( m - j + 1 ) p _ { ( j ) } \bigr ) .\tag{10}
$$

Agreement uses $\kappa = ( p _ { o } - p _ { e } ) / ( 1 - p _ { e } )$ from public agreement counts and positive marginals. Integers and hashes require exact equality; derived rates, Wilson intervals, McNemar values, Holm adjustments, and κ use numerical tolerance $1 0 ^ { - 1 2 }$ . The historical Canary artifact stores Wilson bounds to eight decimals; comparison to those source bounds uses $5 \times 1 0 ^ { - 9 }$ , while the generated table retains full precision.

The independent evidence verifier reconstructs 44 rate/Wilson rows and 28 McNemar/Holm rows. V1 bootstrap intervals are not repeated in V2 tables because their private rows are unavailable; retaining a historical point estimate is not the same as claiming independent row-level regeneration.

Table 1: Current MutMem V2 reproducibility and conformance evidence. Counts are verification units, not benchmark accuracy.
<table><tr><td>Evidence family</td><td>Verified result</td><td>Interpretation</td></tr><tr><td>Portable protocol</td><td>18 schemas; 39 recall vectors; 15 mutation vectors; 37 recall failure codes</td><td>Versioned bytes, membership, predicates, and terminal vocabulary</td></tr><tr><td>Independent Node/Python verifier</td><td>72/72 exact verdict and reason terminals; 10/10 exit criteria</td><td>Verifier portability, not empirical utility</td></tr><tr><td>Production conformance corpus Canary explicit-marker lane</td><td>42/42 cases across 28 required classes</td><td>Production/independent parity Explicit marker traversal only</td></tr><tr><td>Clean installer qualification</td><td>marked 60/60; clean flags 0/60 Node v26.8.1; first boot, restart, and scheduler ready; 0 experimental</td><td>One same-user installed system; no benchmark result</td></tr></table>

Table 2: Historical MutMem V1 results, reproduced from the self-hashed V1 aggregate. These are not current-source V2 reruns.
<table><tr><td>Protocol and metric</td><td>Historical V1 result</td><td>Boundary</td></tr><tr><td>LongMemEval, LLM-judged accuracy</td><td>459/500 (91.80%; Wilson 95% CI 89.06–93.90%)</td><td>GPT-5.4 generator; GPT-5.6 Terra judge</td></tr><tr><td>LoCoMo, LLM-judged accuracy</td><td>1472/1986 (74.12%; Wilson 95% CI 72.15–76.00%)</td><td>Distinct from token F1</td></tr><tr><td>LoCoMo, category-aware token F1</td><td>58.20</td><td>Bootstrap interval omitted in V2: private rows unavailable</td></tr><tr><td>PoisonedRAG adaptation, induced ASR</td><td>1/98 (1.02%)</td><td>Baseline-aware effect: clean 2/100; attacked 3/100</td></tr><tr><td>PoisonedRAG adaptation, poison retrieval@5</td><td>0/100 (Wilson upper 3.70%)</td><td>Retention plus retrieval isolation, not deletion</td></tr><tr><td>Mutation integrity Mutation transaction cost</td><td>7/7 authorization; 4/4 tamper cases median 4.865 ms; p95 5.674 ms;</td><td>20 measured native transitions Single historical machine and full</td></tr><tr><td>Blinded system-author agreement</td><td>966.35 logical bytes/transition 193/200 (96.5%; κ = 0.9108)</td><td>transaction Not independent human validation</td></tr></table>

## 9 Evidence results

## 9.1 Current V2 results

Table 1 separates verification units from empirical accuracy. Exact Node/Python parity and production conformance support a portable verifier claim. They do not imply that another memory engine produces the same utility results. The Canary evidence contains 120 terminal units: all 60/60 marked units were detected (100.0%; Wilson lower 93.98%), and 0/60 clean units were flagged (0.0%; Wilson upper 6.02%). This supports only explicit marke traversal across the declared transport cells. It is not evidence of arbitrary poison, prompt-injection, or semantic social-engineering detection. <sup>[V2-CANARY]</sup>

## 9.2 Historical V1 evidence

Table 2 is intentionally labeled historical. LongMemEval and judged LoCoMo used the historical canonical-blind protocol; the LoCoMo token F1 result used a separate deterministic scorer and must not be averaged with judged accuracy. <sup>[V1-LONGMEMEVAL-UTILITY]</sup> <sup>[V1-LOCOMO-JUDGED][V1-LOCOMO-F1]</sup>

The historical N=100 PoisonedRAG result is an adaptation: target fixture and attacker passages were locked, but corpus scope, retriever, and answer model difered from the upstream full-corpus experiment [7]. All poison passages were retained. Retrieval isolation and signed epistemic labels are not save-time rejection, deletion, or proof that content is false. The observed clean target-answer leakage was 2/100, while the attacked target-match rate was 3/100. The baseline-aware attack-attributable result was therefore an induced ASR of 1/98 (1.02%), which is the efect reported in Table 2. The observed clean and attacked answer accuracies were 72.0% and 71.0%, respectively; all 500/500 poison memories received an adverse signed projection, and 4/18808clean memories did as well (0.0213%). [V1-POISONEDRAG]

Table 3: Historical V1 fixed-corpus PoisonedRAG ablation. Each arm contains 100 paired targets.
<table><tr><td>Arm</td><td>poison retrieval@5</td><td>clean accuracy</td><td>attacked accuracy</td></tr><tr><td>A0</td><td>94/100</td><td>71.0%</td><td>40.0%</td></tr><tr><td>A1</td><td>0/100</td><td>68.0%</td><td>65.0%</td></tr><tr><td>A2</td><td>0/100</td><td>68.0%</td><td>71.0%</td></tr><tr><td>A3</td><td>0/100</td><td>69.0%</td><td>69.0%</td></tr></table>

The historical fixed-corpus ablation in Table 3 attributes the measured selection change to already-produced signed stored labels. A2 added query-local detection but made no additional selection change after A1; active-context withholding was not exercised. The experiment therefore does not establish how either mechanism behaves on unseen attacks. <sup>[V1-EPISTEMIC-ABLATION]</sup>

The historical mutation measurements cover one complete native transaction, including signatures, provenance, projection, and live-weight update. They are not isolated cryptographic overhead and should not be generalized across machines. <sup>[V1-MUTATION-LATENCY-STORAGE]</sup>

The historical agreement audit was blinded to arm and judge verdict but was performed by the system author. It is an agreement diagnostic, not independent human validation. <sup>[V1-HUMAN-AGREEMENT]</sup>

## 10 Attempt accounting and disclosure

The public attempt ledger preserves intended population, terminal population, terminal failures, retries when known, exclusions, amendments, and reused artifacts. Two failed installed-service attempts remain failures with zero result authority. A later one-question diagnostic remains diagnostic only. None appears in a result table, abstract, or comparison. For promoted numerical evidence, intended and terminal populations are equal, with zero terminal failures and zero exclusions. Unknown historical retry details are explicitly unknown, not zero.

The public/private manifest withholds private V1 rows, provider payloads, restricted dataset text, credentials and certificates, live memories, and identity-bearing installer receipts. It exposes sanitized aggregates, semantic roots, source locks, and non-identifying readiness projections instead. The SABER-inspired operational figures are omitted because their exact private artifact is not in verified custody; no value from that family appears in this paper. [SABER-OPERATIONAL-NUMBERS]

## 11 Security argument and non-goals

Commitment substitution. Under collision resistance, changing a canonical object field changes Equation 1, the object root, and Equation 5, except with negligible collision probability. Omitting or duplicating a mandatory object fails Equation 2 before signature verification.

Authority substitution. A bundle does not choose its own trusted master. Actor and Housekeeper certificate chains, validity intervals, revocation observations, grant scope, signed request, and terminal event are checked against the externally selected trust anchor. Under Ed25519 unforgeability, altering a signed commitment without the corresponding private key fails signature verification.

Ordering and terminal fidelity. Per-result ordinals and fixed kind order enter the Merkle root. The terminal event binds the request, authority, root, evidence list, return projection, and result count. Reordering a valid set or replacing the terminal projection is detectable.

These arguments establish tamper evidence under the assumptions. They do not prevent a storage administrator from destroying evidence, prove that signed content is true, or guarantee that every harmful input is detected. Availability and semantic validation require separate mechanisms and evidence.

## 12 Limitations

1. No current V2 utility rerun. Utility, mutation performance, PoisonedRAG, ablation, and human agreement remain historical V1 results. V2 does not relabel them as results of the current source.

2. No independent empirical replication. The verifier implementations are independent, but the empirical experiments have not been repeated by an outside team.

3. Private V1 rows are unavailable. Public counts, Wilson intervals, McNemar/Holm contrasts, and agreement marginals reconstruct. V1 fixed-seed bootstrap intervals do not independently reconstruct and are omitted from V2 tables.

4. Canary scope is marker-specific. Marker traversal is not a general detector for poisoning, prompt injection, or false content.

5. PoisonedRAG is adapted and post-calibration. It does not match the upstream full corpus and does not estimate unseen-attack generalization.

6. SABER figures are omitted. Artifact custody is insuficient for numerical publication.

7. Installation coverage is bounded. One macOS installation on Node v26.8.1 is qualified. Other declared runtimes still require release CI and do not inherit this result automatically.

8. External anchors are necessary. Replacing all evidence and all independent trust anchors is outside the threat model.

9. Archival identifier pending. The immutable source manifest, release commit, tag, and GitHub Release are mutually bound. The arXiv identifier cannot be recorded until submission acceptance.

## 13 Ethics and responsible release

The poisoning materials contain attacker-crafted misinformation and restricted dataset content. The release distributes hashes, manifests, aggregate outcomes, and acquisition instructions rather than benchmark passages or provider payloads. Reproduction must use a fresh installed brain and must not ingest benchmark material into a user’s ordinary persistent memory. A detected item is retained with signed epistemic evidence; classification is reversible and is not presented as factual adjudication.

The verifier’s public failure reasons are designed for audit, but operational receipts may contain identity and storage metadata. Such receipts remain outside the public evidence allowlist. Public evidence contains no credential, private key, certificate, live memory content, or absolute machine path.

## 14 Related work

Tamper-evident logging uses cryptographic commitments to make history substitution detectable [2]. Certificate Transparency defines the domain-separated Merkle construction used here for ordered evidence [3]. MutMem combines established SHA-256, Ed25519 [4], canonical JSON [5], and length-framed protocol encodings [6]; none of those primitives is claimed as novel.

LongMemEval and LoCoMo measure long-horizon memory utility [8, 9]. PoisonedRAG studies knowledge-corruption attacks against retrieval-augmented generation [7]. Those works motivate evaluation but do not by themselves provide a portable authorization and evidence profile for post-deployment memory mutation. MutMem V1 introduced the retained mutation construction and historical system evaluation [1]. V2’s narrower contribution is to make its evidence boundaries, canonical bytes, terminal reasons, independent verification, installation, and publication derivation explicit.

## 15 Conclusion

MutMem V2 turns an implementation-bound integrity claim into a versioned, externally anchored verification protocol. Complete recall and mutation bundles bind canonical objects, cross-object authority, result identity, ordered disclosure, and terminal events. Independent Node and Python implementations agree on the full released terminal corpus, production-derived cases agree with the independent verifier, and one clean installer reaches restart-stable readiness. Publication tables, attempts, omissions, and limitations regenerate from a single evidence root.

The result is a stronger reproducibility claim, not a broader security claim. Historical V1 empirical findings remain historical. Canary evidence remains marker-specific. SABER numerical evidence remains omitted. Integrity and authorization remain distinct from truth, and verifier independence remains distinct from independent empirical replication. The public release identity is mutually bound; the archival identifier is recorded after acceptance.

## References

[1] W. Saidi, “MutMem: Cryptographically Authorized Mutation in Persistent Agent Memory,” arXiv:2608.02843, 2026.

[2] S. A. Crosby and D. S. Wallach, “Eficient Data Structures for Tamper-Evident Logging,” 18th USENIX Security Symposium, 2009.

[3] B. Laurie, A. Langley, and E. Kasper, “Certificate Transparency,” RFC 6962, 2013.

[4] S. Josefsson and I. Liusvaara, “Edwards-Curve Digital Signature Algorithm (EdDSA),” RFC 8032, 2017.

[5] A. Rundgren, B. Jordan, and S. Erdtman, “JSON Canonicalization Scheme (JCS),” RFC 8785, 2020.

[6] D. Boneh and V. Shoup, A Graduate Course in Applied Cryptography, version 0.6, 2023.

[7] W. Zou, R. Geng, B. Wang, and J. Jia, “PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models,” 34th USENIX Security Symposium, 2025.

[8] D. Wu, H. Wang, W. Yu, Y. Zhang, K.-W. Chang, and D. Yu, “LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory,” ICLR, 2025.

[9] A. Maharana, D.-H. Lee, S. Tulyakov, M. Bansal, F. Barbieri, and Y. Fang, “Evaluating Very Long-Term Conversational Memory of LLM Agents,” ACL, pp. 13851–13870, 2024.

## A Complete claim-to-evidence disposition

Table 4: Complete manuscript claim-to-evidence disposition.
<table><tr><td>Claim identifier</td><td>Disposition</td><td>Denominator</td><td>Limitation</td></tr><tr><td>V1-LONGMEMEVAL- UTILITY</td><td>HISTORICAL V1</td><td>459/500</td><td>Historical V1 result; model-judged and not a current-source V2 benchmark.</td></tr><tr><td>V1-LOCOMO-</td><td>HISTORICAL</td><td>1472/1986</td><td>Historical V1 judged accuracy; never merged with token F1.</td></tr><tr><td>JUDGED V1-LOCOMO-F1</td><td>V1 HISTORICAL</td><td>1986</td><td>Point estimate remains in the V1 aggregate; bootstrap</td></tr><tr><td>V1-POISONEDRAG</td><td>V1 HISTORICAL</td><td>100 targets</td><td>interval omitted because private rows are unavailable. Adapted bounded N=100 protocol; no universal or</td></tr><tr><td>V1-EPISTEMIC-</td><td>V1 HISTORICAL</td><td>100 paired targets</td><td>unseen-attack claim. Fixed-corpus causal evidence; query-local contribution was</td></tr><tr><td>ABLATION V1-MUTATION-</td><td>V1 HISTORICAL</td><td>per arm 20 transitions</td><td>null and withholding was not exercised. Historical retained-memory transition evidence under the V1</td></tr><tr><td>INTEGRITY V1-MUTATION-</td><td>V1 HISTORICAL</td><td>20 transitions</td><td>threat model. Single-machine descriptive values; row-level inputs are</td></tr><tr><td>LATENCY- STORAGE</td><td>V1</td><td></td><td>unavailable for independent recomputation.</td></tr><tr><td>V1-HUMAN- AGREEMENT</td><td>HISTORICAL V1</td><td>200 answers</td><td>System-author diagnostic, not independent human validation; bootstrap interval omitted.</td></tr><tr><td>V2-PORTABLE- PROTOCOL</td><td>CURRENT V2</td><td>39+15 vectors</td><td>Portable verification protocol, not another memory engine or</td></tr><tr><td>V2-INDEPENDENT- VERIFIER</td><td>CURRENT V2</td><td>72 terminal verdicts</td><td>producer. Cross-language verdict/reason parity proves verifier</td></tr><tr><td>V2-PRODUCTION- CONFORMANCE</td><td>CURRENT V2</td><td>42/42</td><td>Conformance corpus, not benchmark efficacy.</td></tr><tr><td>V2-CANARY</td><td>CURRENT V2</td><td>120/120</td><td>Explicit marker traversal only; not arbitrary poison, prompt-injection, or semantic social-engineering detection.</td></tr><tr><td>V2-CLEAN- INSTALLER</td><td>CURRENT V2</td><td>one qualified installation and restart</td><td>Reproducibility evidence only; not benchmark evidence.</td></tr><tr><td>SABER- OPERATIONAL-</td><td>OMITTED</td><td>No numerical claim</td><td>Exact private artifact is not in verified custody; no numerical V2 claim is permitted.</td></tr><tr><td>NUMBERS CONTENT-TRUTH</td><td>PROHIBITED</td><td>No numerical claim</td><td>Integrity, provenance, authorization, and agreement do not establish semantic truth.</td></tr><tr><td>UNIVERSAL- DEFENSE</td><td>PROHIBITED</td><td>No numerical claim</td><td>Fixed marker and N=100 evidence do not establish universal robustness.</td></tr><tr><td>INDEPENDENT- REPLICATION</td><td>PROHIBITED</td><td>No numerical claim</td><td>No outside team independently reproduced the system or empirical results.</td></tr></table>

## B Verifier and regeneration commands

npm run verify

npm run evidence:regenerate

npm run verify

npm run paper:verify

The first, third, and fourth commands are read-only. The second command deterministically rewrites only the public evidence projections and fails its subsequent byte check if output changes. No command in this appendix runs a benchmark.