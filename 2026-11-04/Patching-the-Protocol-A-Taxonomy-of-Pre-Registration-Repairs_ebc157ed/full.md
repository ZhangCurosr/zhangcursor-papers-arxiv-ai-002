ROYAL SOCIETY OPEN SCIENCE

royalsocietypublishing.org/journal/rsos

Research CO Check for updates

Article submitted to journal

Subject Areas: computer science; statistics

Keywords: pre-registration; LLM-assisted research; protocol design; reproducibility; mechanism design; research integrity

Author for correspondence: Siyuan Zhang e-mail: lelandsy28@163.com

# Patching the Protocol: A Taxonomy of Pre-Registration Repairs in LLM-Proposer Experiments

Peter Chen<sup>1,</sup>†, Ni Zhang<sup>1,</sup>† and Siyuan Zhang<sup>1,</sup>∗

<sup>1</sup>Shenyang Institute of Technology, Shenyang, China

Pre-registration is increasingly proposed as a trust mechanism for machine-learning research involving LLM-generated content, but its canonical form assumes a frozen protocol can be executed unchanged. Software-intensive studies violate this assumption: the protocol is partly implemented, and implementations carry defects. We report a threecase study of protocol repairs (“patches”) from a program where an LLM proposes executable rules that a programmatic verifier gates, across production scheduling, incentive design, and task assignment. Each protocol required exactly one defect-repair patch: three events in total. None modified a numbered judgment criterion, its threshold, or the scoring rule, and none altered already collected data; two changed pre-registered design parameters before any data were collected, each with a recorded rationale. We propose a two-axis taxonomy of patches (defect class: environmental / feasibility / correctness, × timing: pre-data / post-data), introduce a verifiable non-impact claim for postdata repairs, and distill a nine-item checklist; all nine items were exercised by at least one patch event – an observation of use, not a validation of suficiency. The credibility of verifier-bound, LLMproposer studies depends less on whether patches occur than on whether they are auditable.

## 1. Introduction

Discussions of trust in LLM-assisted research concentrate on a dramatic failure mode: the model fabricates results [1], and the human passes them on. A quieter failure mode is more common in practice. Computational reproducibility has its own canon [2–4], and replication campaigns in adjacent fields document how protocol drift undermines trust [5–7]; the broader case for pre-registering against such drift is long-standing [8,9]. Studies in which an LLM proposes code [10–12] — decision rules, scoring functions, mechanisms — are software-intensive, as the self-correction literature that audits such programs attests [13–15]: part of the “protocol” is a running program, and programs ship with defects regardless of intent. A pre-registration that freezes hypotheses, criteria, and budgets still gets executed by software that may crash on boundary conditions, may encode mathematically infeasible parameters, or may silently corrupt its own measurements.

The canonical pre-registration playbook — OSF-style freezing [16] and Registered Reports [17] — has no vocabulary for this. Its implicit model is: freeze, then execute. When execution reveals that the protocol itself is defective, the researcher faces an unforced choice — run a broken study, or change the protocol and appear to have moved the goalposts. Both outcomes damage credibility, and the second one is indistinguishable from motivated tuning unless the change is auditable.

Recent work extends pre-registration to LLM-based research — to experiments in which agents are the subjects [18] and to LLM-generated data feeding downstream hypothesis tests [19] — and audits LLM pipelines for accountability layers that relay rather than check [20] and for silent API failure [21]. None of these addresses the case we study: a defect in the running protocol itself, and how to repair it without moving the goalposts.

The concern is not academic. A recent roadmap for recursive self-improvement in research agents names reliable verification as the bottleneck at every level of autonomy, and documents a live case in which an agentic system cherry-picked favorable results during automated experimentation [22]. Verification infrastructure, once deployed, becomes part of the protocol and therefore part of what can break, and be repaired, mid-study.

This paper takes the position that patches are normal and can be made trustworthy. We study three repairs that occurred in a three-domain research program (production scheduling; incentive mechanism design for mobile crowdsensing; capacity-constrained task assignment). Each domain ran under a pre-registered protocol [23–25] with append-only changelogs, frozen judgment criteria, negative-result clauses, leakage guards, and mandatory equal-budget random controls. Each protocol required exactly one defect-repair patch; across all three, no patch modified a numbered criterion, its threshold, the scoring rule, or any collected data point (two patches did change pre-registered design parameters before data collection; §3 gives the component-by-component verdicts).

Contributions.

(i) A three-case taxonomy of protocol repairs, classified along two axes: defect class (environmental / feasibility / correctness) and timing (pre-data / post-data).

(ii) The concept of a verifiable non-impact claim: for post-data repairs, the changelog must not merely assert that already collected data are unafected, but must contain a check that a reader can re-run.

(iii) A nine-item checklist for disciplined protocol repair in software-intensive ML studies, distilled from the cases and shipped as a reusable pre-registration template; all nine items are exercised by at least one case (nine of nine; Table 3).

The checklist is not externally validated (§7); our claim is narrower — that these practices made three specific repairs auditable, and that the audit trail is inspectable in the released archives.

## 2. Background: the protocol layer

## 2.1 Setting

The research program uses a verifier-bound design: an LLM proposes executable rules; a programmatic verifier checks candidate rules against frozen golden properties; a held-out evaluation scores surviving rules. Three domains were studied sequentially:

• Domain A (production scheduling): the LLM proposes job-to-slot assignment rules; the verifier checks feasibility and overdue constraints; the outcome measure is newoverdue cost on held-out instances [23].

• Domain B (incentive design): the LLM proposes payment functions pay(report\_c, report\_q, signal, round, budget\_left); the verifier enumerates worker types and misreport grids to check individual rationality (IR) and incentive compatibility (IC); the primary outcome is the fraction of accepted mechanisms [24].

• Domain C (task assignment): the LLM proposes scoring functions over (distance, capability, window, load, …); the verifier caps coverage degradation and inequality (Gini); the outcome is a composite J [25].

## 2.2 Protocol-layer components

Every domain ran under the same five protocol components, frozen before data collection:

(i) Frozen criteria. Each pre-registration lists numbered criteria with thresholds and precommitted result branches; adding, modifying, or deleting criteria after data collection is prohibited; exploratory analyses must be labeled as such.

(ii) Append-only changelog. Any protocol change is a dated patch appended after existing entries, which are preserved verbatim. Each patch declares its efect on already collected data.

(iii) Negative-result clauses. Failure branches and “gray-band” outcomes are precommitted; gray bands may not be narrated as wins.

(iv) Leakage discipline. Held-out data are constructed by physically separate code; each promoted rule (baselines included) is evaluated on the holdout exactly once, enforced by a counter; a runtime probe test asserts that no evaluation module can read ground-truth keys even if they are injected into the context.

(v) Equal-budget controls. Every LLM condition is paired with a random-proposer control spending the same evaluation budget (evals are the budget currency), with control seeds independent of instance seeds.

Component 2 is the subject of this paper: it is what turns a repair from a credibility hazard into an auditable event.

## 3. Three cases

Counting rules. All counts in this paper are patch-event counts under the following conventions, fixed before writing. A substantive repair is one changelog entry that records a defect-repair patch, however many individual changes that entry bundles; entries that freeze a draft into v1.0, extend the pre-registration with a new study stage, or adjust reporting wording are not defect repairs and are not counted. A patch touches a judgment criterion, threshold, or scoring rule if it changes the numbered criterion text, the numeric threshold or result branch attached to it, or the rule that computes the outcome measure; changes to the surrounding machinery that make a frozen criterion evaluate as specified (or to pre-registered design parameters such as budgets and task counts) are recorded separately and do not count as touching. The full changelog histories are: Domain A, v1.0 → v2 (Stage-2 pre-registration added) → v3 (Stage-4 pre-registration added) → v3.1 (the repair of Case 1; one repair event); Domain B, v0.1 → v1.0 (freeze) → v1.1 (the repair of Case 2; one repair event bundling four subchanges) → v1.2 (reporting-wording adjustment; not a repair); Domain C, v0.1 → v1.0 (freeze) → v1.1 (the repair of Case 3; one repair event bundling three sub-changes). Each case’s repair section walks its own sub-changes so that the headline counts can be re-derived component by component.

<table><tr><td>Domain</td><td>Discovery context</td><td>Scale at patch time Patch (key parameters)</td><td></td><td>Data at patch</td></tr><tr><td>A</td><td>run-time crash batch study; at pair 037</td><td>pairs written</td><td>37 redraw rule extended train → holdout (seed+1)</td><td>37 pairs retained, re- validated</td></tr><tr><td>B</td><td>1-pair test</td><td>smoke N = 20 budget B = 50</td><td>workers; B: 50 → 100; participation none (pre- constraint; exemption</td><td>baseline data)</td></tr><tr><td>C</td><td>smoke debugging</td><td>capacity  $3 0 \times 3 = 9 0 ;$   $M \in \{ 5 0 , 8 0 , 1 0 0 \}$ </td><td>deep-copy/idempotent run_match; M: 50 → 60; data) deadlines 10–30 → 8–15 min</td><td>none (pre-</td></tr></table>

Table 1. Per-domain scale, patch, and data status at patch time. Pair indices are zero-based (pairs 0–36 complete, crash at 037). Symbols (B, M, capacity) are defined in the case narratives and §2.1. A smoke test is a minimal pre-experiment trial run whose outputs are not data.

Each case follows the same template: archive → discovery context → defect → repair → efect on collected data → classification. Changelog text is quoted verbatim (translated where necessary). Table 1 summarizes the scale of each domain and the status of its data at patch time; all numbers are taken from the frozen pre-registrations and append-only changelogs.

## 3.1 Case 1 (Domain A): environmental defect — crash at a boundary

Archive. Domain A pre-registration, changelog v3.1 (2026-09-11).

Discovery context. Run time. The batch study crashed at pair 037; 37 pairs had already been written to disk.

Defect. The instance-generation procedure draw\_pair had a degenerate-case redraw rule that covered only training instances. When the holdout baseline came up infeasible (no feasible schedule within 40 days), generation crashed instead of redrawing.

Repair (verbatim, Chinese): “draw\_pair 的退化重抽规则从仅 train 扩展到 holdout（holdout baseline INFEASIBLE 或排产 >40 天时 seed+1 重抽）。此为崩溃修复而非口径变更，与 v3 预注 册『退化重抽 seed 递增』精神一致” — the redraw rule of draw\_pair was extended from trainonly to holdout as well: when a holdout baseline is INFEASIBLE or schedules beyond 40 days, redraw with an incremented seed. This is a crash fix, not a criteria change, and is consistent with the pre-registered “degenerate-case seed-increment redraw” policy of pre-registration v3 (the changelog entry is v3.1; v3 there denotes the pre-registration version). The extension is not merely consistent with that policy — the frozen text already required it: “if any instance’s baseline evaluation is INFEASIBLE or schedules beyond 40 days, redraw with an incremented seed and record the redraw count”; the rule is stated over any instance, train or holdout. The defect was that the implementation covered only training instances; the patch aligned the code with the policy it was supposed to implement.

Efect on collected data. The changelog carries a verifiable non-impact claim: “前 37 对 （pair 0–36）在旧逻辑下产出，其 holdout baseline 均已验证可行，数据保留有效” — each of the 37 written pairs was individually checked and its holdout baseline confirmed feasible, so the

old-logic data are retained. The check is mechanical: a reader can re-run feasibility validation on pairs 0–36. The claim’s scope is exactly what the check computes: the non-impact claim asserts that the holdout baselines of pairs 0–36 are feasible under the frozen instance specification, and the check recomputes precisely that quantity; the check is a deterministic recomputation over frozen instance snapshots — the same inputs the verifier consumed at collection time — so a reader re-running it obtains the changelog’s verdict directly, and the snapshots and the check’s code ship in the Domain A archive. Two boundaries are stated explicitly. First, the claim does not cover the prospective composition of pairs 037 onward: because the redraw policy conditions instances on baseline feasibility, the holdout population from the repair point onward is feasibility-conditioned — which is what pre-registration v3 already intended, but a reader comparing raw draws to post-patch draws should know the conditioning applies. Second, the checker is itself program code inside the protocol and shares its failure modes; the program therefore requires non-impact checks to be minimal, independently testable code whose own defects are changelog-managed like any other patch (checklist items 2 and 9).

Classification. Environmental: the defect is a crash in the data-generation pipeline at a boundary condition. Symptoms are visible (the run stops), so the risk of unnoticed damage is low. The repair does not alter what is measured or accepted.

## 3.2 Case 2 (Domain B): feasibility defect — infeasible parameters, intercepted at design time

Archive. Domain B pre-registration, changelog v1.0 → v1.1 (2026-09-12), after a one-pair smoke test.

Discovery context. Design time. A single smoke pair failed to produce any verifieracceptable mechanism, prompting a by-hand audit of the parameter regime.

Defect. The budget made IR and the budget cap mathematically incompatible. In the simulated market, worker i’s utility from a matched task is $u _ { i } = \mathrm { p a y } _ { i } - c _ { i } d _ { j } \colon$ the payment minus the efort cost of completing the assigned task. The verifier checks IR by requiring honest utility $\geq 0$ for all $N = 2 0$ enumerated worker types, and IC by enumerating a misreport grid per type (report\_c at 11 points, report\_q at 5 points, $\epsilon = 0 . 0 5 )$ . A payment that is IC-safe in this design cannot profitably depend on the reports: any report dependence a worker could exploit is caught by the grid, so the binding payment structure is a constant. Individual rationality for the worst worker type (cost c = 2.0, efort $d = 1 . 5 )$ then requires a constant payment of at least c · d = 3.0 per worker. With N = 20 workers the minimum total outlay is $2 0 \times 3 . 0 = 6 0 > B = 5 0$ No mechanism can satisfy IR and the budget simultaneously: the verifier’s acceptance rate is structurally zero for reasons that have nothing to do with any proposer, human or machine.

Repair. Budget B: 50 → 100, restoring a non-empty feasible region (the changelog’s recorded rationale; B = 100 leaves the binding floor of 60 non-trivial headroom). The same patch, from the same smoke test, fixed three adjacent design defects: (i) a participation constraint was added to the simulation, so that IR violations have real economic consequences (without it, “compliant mechanisms pay more” loses to violating mechanisms by construction, contradicting the preregistered criterion); (ii) baselines are exempt from the verifier gate — they are reference lines, and their verifier status is reported as evidence, not used for exclusion; (iii) the prompt was hardened to eliminate irrelevant parse failures.

Component verdicts. Under the counting rules above, none of the four sub-changes touched a criterion. (i) changes the market semantics of the simulation: the frozen primary criterion (LLM-best holdout welfare vs. the three baselines) is unchanged in text and threshold, but without participation the simulated welfare ordering contradicts the economics that criterion assumes — the patch makes the frozen criterion evaluate as specified rather than redefining it. (ii) changes gate scope: which evaluated objects must pass the verifier before entering the comparison; the comparison structure of the criterion is unchanged, and all three baselines still enter the holdout evaluation. (iii) is a pipeline-behavior fix at the parse boundary. The budget change is a pre-registered design parameter chosen pre-data; its selection procedure is the changelog’s recorded rationale, and we flag in §7 that repair-parameter selection remains a discretion this checklist does not yet cover.

Efect on collected data. None: the experiment had not started; the smoke pair was discarded by rule (checklist item 8).

Classification. Primary class: feasibility — the defect lives in the specification, not the code: an infeasible parameter regime that any smoke test could expose. This is the cheapest class to catch and the most damaging to miss: an unnoticed infeasible regime produces a structurally zero acceptance rate that could be misread as a scientific finding (“LLMs cannot design mechanisms”) rather than an artifact. Secondary class: correctness — sub-change (i) alters measurement semantics (welfare accounting) so that the frozen criterion is evaluated as specified, which is a correctness-class change in our own taxonomy; the primary class is the one whose damage profile would dominate had the defect been missed. Sub-change (iii) is environmental in character (a boundary behavior of the generation pipeline).

## 3.3 Case 3 (Domain C): correctness defect — silent measurement corruption

Archive. Domain C pre-registration, changelog v1.0 → v1.1 (2026-09-12), after smoke debugging; the fix is documented in the simulation module’s docstring.

Discovery context. Smoke debugging, pre-data.

Defects (two, one code-level and one design-level). Code level. run\_match mutated the evaluation batch in place. When several mechanisms were scored against the same batch, each mechanism’s run contaminated the state seen by the next; the observable symptom was that all mechanisms received identical composite scores (J). Note what makes this the most dangerous class: the code succeeds — no crash, no exception, plausible-looking numbers. Had the study run, it would have produced a full dataset in which no mechanism can be distinguished from any other, and nothing about the output would flag it.

Design level. The task load had no region of genuine tension. With M = 50 tasks the coverage term is trivially saturated; with M = 80 or 100, tasks exceed total capacity $( 3 0 \times 3 = 9 0 )$ , every worker ends up fully loaded, and the Gini fairness term is identically zero. All three tested values lie in structurally degenerate parameter regions.

Repair. run\_match now deep-copies internally and is idempotent (documented and unittestable); task count $M \colon 5 0  6 0$ , and deadlines 10–30 → 8–15 minutes, placing the regime in the intermediate band where coverage is non-trivial and time windows bind. The changelog records the selection procedure as an explicit three-region analysis over the pre-registered candidates: $M = 5 0$ saturates the coverage term (trivially easy); $M \in \{ 8 0 , 1 0 0 \}$ exceeds total capacity $( 3 0 \times 3 = 9 0 )$ , so every worker ends up fully loaded and the Gini term is identically zero; the intermediate value $M = 6 0$ leaves a finite but non-empty set of time-feasible assignments, which is the only region in which the coverage–fairness tradeof has genuine tension. The deadline band 8–15 minutes was chosen in the same analysis so that the time-feasibility constraint actually binds at $M = 6 0$ . As with the Domain B budget, we flag in §7 that this parameter selection was author judgment, documented post hoc in the changelog rather than pre-committed by rule.

Efect on collected data. None: the experiment had not started; the smoke pair was discarded.

Classification. Primary class: correctness (code level) — a silent defect in measurement semantics. It can only be caught by design review, idempotency discipline, or the luck of a smoke-test symptom — never by monitoring the run. Secondary class: feasibility (design level) — the degenerate parameter regions. Both classes are labeled in the changelog entry (§4.1).

## 4. Taxonomy

<table><tr><td>Class</td><td>Locus</td><td>Symptom</td><td>Detection cost</td><td>Damage if missed</td></tr><tr><td>Environmental</td><td>pipeline boundary crash (loud) conditions</td><td></td><td>low, stops the run lost</td><td>compute, recoverable</td></tr><tr><td>Feasibility</td><td>specification parameters</td><td>/ structurally degenerate outcomes</td><td>cycle</td><td>low, one smoke misread artifact as finding</td></tr><tr><td>Correctness</td><td>measurement semantics</td><td>none (silent)</td><td>discipline</td><td>high, needs audit invalid dataset with plausible numbers</td></tr></table>

Table 2. The defect-class axis. Detection-cost and damage entries for feasibility and correctness rest on one observed instance each; treat them as working hypotheses, not estimates.

## 4.1 Axis 1: defect class

The classes are ranked by audibility (Table 2). Environmental defects announce themselves; feasibility defects announce themselves through degenerate outcome distributions if one looks; correctness defects say nothing. Protocol design should therefore spend its review budget in inverse proportion to audibility: unit-testable idempotency and state-isolation invariants (correctness), smoke-test regime audits (feasibility), and simple redraw robustness (environmental).

The classes describe the primary defect of a patch, but a single smoke cycle can surface defects in more than one class; Case 3 pairs a correctness defect (code level) with a feasibility defect (design level), and Case 2 adds a correctness-class sub-change to a feasibility-class patch. Such patches carry a primary and a secondary class in the changelog, and the primary class is the one whose damage profile would dominate had the defect been missed.

## 4.2 Axis 2: timing

• Pre-data repairs (Cases 2, 3) enjoy maximum freedom: the cost of a patch is a discarded smoke pair. Nothing else is owed.

• Post-data repairs (Case 1) owe the archive a verifiable non-impact claim: not “these data are fine” but “here is the check, here are the 37 pairs, re-run it yourself.” The claim in Case 1 is mechanical (feasibility validation of pairs 0–36) and cheap to re-verify.

## 4.3 The red line

Case inclusion. The three cases are all of the program’s domain pre-registrations that existed when this paper was written: each produces its own domain data under a frozen protocol. The program also runs cross-domain analysis artifacts, which consume domain results rather than produce domain data; those follow the same changelog discipline but are outside this paper’s case set by construction, not by curation.

The red line. The taxonomy deliberately excludes one category: post-data patches that alter criteria, thresholds, or scoring. In the domain protocols such changes are prohibited outright. One cross-domain analysis artifact did require a protocol revision: its shared bucketing rule (quartile buckets over positive train–holdout margins) was ill-defined for a zerogain population, so the analysis protocol was revised to v1.1, restricting the shared buckets to pairs with positive margin; the revision was declared before the analysis was re-run and recorded in the artifact’s changelog [26]. We report this episode not as a counterexample but as the boundary case it is: the revision touched an analysis protocol, not a domain criterion, and it is the kind of change the red line would prohibit inside a domain protocol mid-study. The taxonomy’s purpose is to make this boundary legible: patches that repair the vehicle are normal; patches that steer toward a desired result are not, and the two are separated by timing, by what the patch touches, and by which ledger records it — all visible in the archives.

Why the red line is asymmetrical. The three defect classes difer sharply in what a postdata repair can honestly promise. An environmental post-data repair can carry a mechanical nonimpact claim (Case 1: feasibility re-validation of the 37 written pairs) — this is the one observed instance. The other two arms of the asymmetry are analytic conjectures, not observations, and should be read as such. A feasibility post-data repair might plausibly carry a non-impact claim, since specification defects tend to produce visibly degenerate outcome distributions, but the study contains no post-data feasibility instance to confirm it. A correctness post-data repair generally cannot carry one: a defect in measurement semantics silently invalidates the data it produced, and no per-pair check can certify measurements that were computed wrongly from the start. This asymmetry is the reason the red line is drawn where it is. The classes whose post-data repairs remain verifiable are exactly the classes that do not touch scoring; any postdata change that does touch scoring is therefore presumptively unverifiable, which is why the domain protocols prohibit it outright rather than merely documenting it.

## 5. The checklist

Shipped as a fill-in template; reproduced here in full.

(i) Freeze criteria before data. Numbered criteria, thresholds, result branches. No posthoc additions, modifications, or deletions; label exploratory analyses as exploratory.

(ii) Append-only changelog. Prior entries preserved verbatim; every patch declares its efect on collected data (“none” + reason, or a re-runnable check).

(iii) Negative-result clauses. Pre-commit failure branches and gray-band narratives; gray bands may not be narrated as wins.

(iv) Leakage discipline. Holdout built by separate code; holdout evaluated exactly once per rule (counter-enforced); runtime probe tests assert no module can read ground-truth keys.

(v) Equal-budget controls. Report a same-evals random-proposer control for every LLM condition; control seeds independent of instance seeds.

(vi) Log every rejection verbatim (sandbox and parse failures).

(vii) Three-state verifiers. If a property cannot be evaluated, the verdict is uncertain, never a silent pass.

(viii) Smoke pairs are not data. Smoke-test outputs are discarded by rule, which is what makes design-time patches cheap.

(ix) Classify every patch — environmental / feasibility / correctness × pre-data / postdata — in the changelog entry itself.

Items 1–3 follow standard pre-registration practice; items 4–8 are specific to software-intensive studies; item 9 is this paper’s addition.

The 9/9 claim, made checkable. A checklist item is exercised in a case if that case’s protocol, changelog, or archive shows the practice instantiated and load-bearing for the study or the repair. Item k is counted if at least one case exercises it. Table 3 gives the per-item mapping; the abstract’s nine-of-nine count can be checked against it without leaving the paper. One honesty note: this checklist was distilled from the cases, so citing an item number inside a case narrative (as Case 2 does for item 8) is retrospective codification, not a claim that the numbered rule governed the episode at the time; the underlying practice did, as the cited changelog text shows.

## 6. Discussion

Relation to existing practice. The discussion continues, rather than repeats, the positioning sketched in the introduction. OSF-style pre-registration and Registered Reports [16,17] freeze hypotheses and analysis plans; neither anticipates that the execution vehicle is code with its own defect surface. Software engineering’s response to the same problem is regression-style gatekeeping [27,28]: changes are allowed, but each change carries a rationale, a scope, and a test. The changelog transplants that discipline into the pre-registration: a changelog entry is a commit message, the “efect on collected data” field is the test, and the append-only rule is immutability.

<table><tr><td rowspan=1 colspan=9>Checklist item                 Domain A           Domain B             Domain C</td></tr><tr><td rowspan=8 colspan=9>1 Freeze criteria               pre-reg      v1-v3 v1.0 freeze             v1.0 freezefreeze  numberedcriteria2 Append-only changelog     v1→v2 → v3 → 4 entries, v0.1 to v1.2 3 entries, v0.1 tov3.1                                           v1.13 Negative-result clauses      failure   branches failure branches pre- failure   branchescommitted             pre-committed4 Leakage discipline           holdout evaluated HoldoutGate          HoldoutGaterandom 20 random proposals 20         random</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>ches</td></tr><tr><td rowspan=1 colspan=3>pre-committed</td><td rowspan=1 colspan=2>ed</td></tr><tr><td rowspan=1 colspan=4>holdout evaluated</td></tr><tr><td rowspan=1 colspan=4>once, counter-</td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>asserted</td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1>5 Equal-budget controls</td><td rowspan=1 colspan=4>20random</td><td rowspan=1 colspan=3>20 random proposals</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>proposals      per p</td><td rowspan=1 colspan=3>er pair</td><td rowspan=1 colspan=1>proposals      per</td></tr><tr><td rowspan=2 colspan=1>6 Log rejections verbatim</td><td rowspan=1 colspan=4>pair</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>pair</td></tr><tr><td rowspan=1 colspan=4>parse-fail     rate v</td><td rowspan=1 colspan=3>iolation ledger fed vi</td><td rowspan=1 colspan=1>olation ledger fed</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>archived;      bad</td><td rowspan=1 colspan=3>back</td><td rowspan=3 colspan=1>backbaseline   verifier</td></tr><tr><td rowspan=2 colspan=1>7 Three-state verifiers</td><td rowspan=1 colspan=4>proposals kept</td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=4>sandbox exception b</td><td rowspan=1 colspan=3>aseline      verifier</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>二         ERROR st</td><td rowspan=1 colspan=3>atus recorded</td><td rowspan=1 colspan=1>status recorded</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>candidate,     not</td><td rowspan=1 colspan=3></td><td rowspan=3 colspan=1>ir discarded by</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>crash</td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1>8 Smoke pairs not data</td><td rowspan=1 colspan=1>n/a (no sm</td><td rowspan=1 colspan=3>oke 1 pa</td><td rowspan=1 colspan=3>ir discarded by 1 pa</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>pair;</td><td rowspan=1 colspan=3>run-time r</td><td rowspan=1 colspan=3>ule</td><td rowspan=3 colspan=1>rulev1.1:         silent</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>case)</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1>9 Classify every patch</td><td rowspan=1 colspan=4>v3.1:&quot;crash fix</td><td rowspan=1 colspan=3>, v1.1:      infeasible</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=8>criteria parameter regime     corruption      十change”                                       degenerate regime</td></tr></table>

Table 3. Checklist items cases. Every item is exercised in at least one case (nine of nine). Cell contents are the archive facts; item 8 is not applicable in Domain A, which had no smoke pair.

Pre-registration practice already has a vocabulary for protocol change: OSF-style deviation documentation and the Registered Reports Stage-2 rules distinguish minor deviations from major amendments, and pre-specified analyses from exploratory ones [16,17]. Our two axes are complementary rather than competing. The timing axis refines the amendment point: pre-data patches sit closer to design-time specification changes than to deviations from a running study. The defect-class axis adds a dimension those frameworks lack — the locus of the change in the execution vehicle — and that locus is what determines what kind of non-impact evidence is possible at all.

Patches did not lower the bar. Across the three cases, no numbered criterion, threshold, or scoring rule changed; the one post-data repair extended an existing, pre-registered policy (redraw-on-degenerate) to a split it had not covered, and the pre-registration text already stated that policy over any instance (§3, Case 1). Two pre-data patches did change pre-registered design parameters (the Domain B budget and gate scope; the Domain C task count and deadline band), each with its rationale recorded in the changelog and walked component by component above.

The changelogs are deposited as versioned Zenodo records, so the patch history carries thirdparty deposit timestamps; a skeptical reader can check each claim in this paper against those records — within the limits of a self-audited program (§7).

What the taxonomy buys. We found naming the classes useful in our own review discussions, though this is self-report rather than evidence. “The protocol was patched” is an accusation; “a feasibility-class, pre-data patch, with the smoke pair discarded” is a status report. We found the correctness class the most sobering: Case 3’s defect would have produced a complete, well-formed, uninformative dataset, and only the smoke-test symptom and the idempotency review stood between it and the archive.

## 7. Limitations

Three cases from one program, classified by the program’s own authors: this is a self-audit, not an independent one, and n = 3 supports the taxonomy’s plausibility, not its completeness — other defect classes may exist. The checklist has not been adopted or stress-tested by an external team. The underlying studies are simulations, not human-subject experiments; the protocol layer transfers, but the stakes profile does not. Finally, verifiable non-impact claims are only as good as their checks; a program that always writes “no impact” without a re-runnable check has adopted the vocabulary of this paper without its substance. Finally, the selection of new parameter values in pre-data repairs (the Domain B budget, the Domain C task count and deadline band) was author judgment recorded in the changelog, not a pre-committed selection rule; a checklist that disciplines repair reporting does not yet discipline repair parameterization, and we regard that as an open design problem.

## 8. Conclusion

Software-intensive research cannot promise defect-free protocols; it can promise auditable repairs. A two-axis taxonomy (environmental / feasibility / correctness × pre-data / postdata), a changelog that carries verifiable non-impact claims, and a nine-item checklist turn “we changed the protocol” from a credibility hazard into a status report — for verifier-bound, LLMproposer studies today, and for closed-loop automated research tomorrow, where the same trust question arrives with higher stakes. The archives of the three studies — pre-registrations with append-only changelogs, result ledgers, and analysis scripts — are released so that every claim in this paper can be checked against the patch record it describes.

Data Accessibility. The pre-registration documents (with append-only changelogs), run ledgers including rejected proposals, analysis scripts and failure records are archived on Zenodo (per-domain records: 10.5281/zenodo.23433811, 10.5281/zenodo.23433817, 10.5281/zenodo.23433820; cross-domain analysis record: 10.5281/zenodo.23433822; program-level protocol record: 10.5281/zenodo.23433826). Pre-registrations, ledgers, analysis scripts and the nine-item checklist template support full reproduction of every claim in the paper.

Ethics. No human participants or personally identifying data were involved: the underlying studies are simulations over synthetic and public, de-identified data. The protocol-repair taxonomy concerns the auditing of changes to a pre-registered plan in software-intensive, verifier-bound research; we deploy no discovered rule.

Authors’ Contributions. P. C. and N. Z. contributed equally to this work. P. C.: conceptualization, supervision, project administration, writing – review and editing. N. Z.: formal analysis, writing – review and editing. S. Z.: conceptualization, methodology, software, investigation, data curation, formal analysis, writing – original draft. All authors reviewed and approved the final version and agree to be held accountable for the work performed therein.

Funding. This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.

Competing Interests. The authors declare no competing interests.

Disclaimer. During the preparation of this work the authors used a large-language-model assistant to improve language and readability; the authors reviewed and edited the content and take full responsibility for the published article.

## References

1. Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang, A. Madotto, P. Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1– 38, 2023.

2. R. D. Peng. Reproducible research in computational science. Science, 334(6060):1226–1227, 2011.

3. V. Stodden, M. McNutt, D. H. Bailey, E. Deelman, Y. Gil, B. Hanson, M. A. Heroux, J. P. A. Ioannidis, M. Taufer. Enhancing reproducibility for computational methods. Science, 354(6317):1240–1241, 2016.

4. S. Kapoor, A. Narayanan. Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9):100804, 2023.

5. Open Science Collaboration. Estimating the reproducibility of psychological science. Science, 349(6251):aac4716, 2015.

6. T. M. Errington, M. Mathur, C. K. Soderberg, A. Denis, N. Perfito, E. Iorns, B. A. Nosek. Investigating the replicability of preclinical cancer biology. eLife, 10:e71601, 2021.

7. C. F. Camerer, A. Dreber, E. Holzmeister, et al. Evaluating the replicability of social science experiments in Nature and Science between 2010 and 2015. Nature Human Behaviour, 2(9):637–644, 2018.

8. J. P. A. Ioannidis. Why most published research findings are false. PLoS Medicine, 2(8):e124, 2005.

9. D. Fanelli. Negative results are disappearing from most disciplines and most countries. Scientometrics, 90(3):891–904, 2012.

10. M. Chen et al. Evaluating large language models trained on code. arXiv:2107.03374, 2021.

11. S. Gulwani, O. Polozov, R. Singh. Program synthesis. Foundations and Trends in Programming Languages, 4(1-2):1–119, 2017.

12. C. Lu, C. Lu, R. T. Lange, J. Foerster, J. Clune, D. Ha. The AI Scientist: towards fully automated open-ended scientific discovery. arXiv:2408.06292, 2024.

13. J. Huang, X. Chen, S. Mishra, H. S. Zheng, A. W. Yu, X. Song, D. Zhou. Large language models cannot self-correct reasoning yet. International Conference on Learning Representations (ICLR), 2024.

14. A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegrefe, U. Alon, F. Dziri, S. Prabhumoye, Y. Yang, S. Gupta, B. Majumder, K. Hermann, P. Welleck, A. Yazdanbakhsh, P. Clark. Self-Refine: iterative refinement with self-feedback. Advances in Neural Information Processing Systems (NeurIPS), 2023.

15. N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, S. Yao. Reflexion: language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems (NeurIPS), 2023.

16. B. A. Nosek, C. R. Ebersole, A. C. DeHaven, and D. T. Mellor. The preregistration revolution. PNAS, 115(11):2600–2606, 2018.

17. C. D. Chambers and L. Tzavella. The past, present and future of registered reports. Nature Human Behaviour, 6:29–42, 2022.

18. M. Vaccaro. Preregistration for experiments with AI agents. arXiv:2606.11217, 2026.

19. M. Thomas, K. Gligoric, and N. B. Shah. Mitigating LLM-based p-hacking by preregistering for the next LLM. arXiv:2606.27687, 2026.

20. P.-P. Arslan. Audit without verification: When LLM accountability layers relay rather than check. arXiv:2609.07680, 2026.

21. Z. Li, S. Ye, F. Guo, and Z. Dang. SilentProbe: Measuring silent failure in production APIs used as agent tools. arXiv:2609.00035, 2026.

22. Y. Duan et al. The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement. arXiv:2609.11873, 2026.

23. P. Chen and S. Zhang. Verifier-bound rule discovery: A pre-registered evaluation of LLM proposers for combinatorial scheduling. Zenodo. doi:10.5281/zenodo.23433811, 2026.

24. P. Chen and S. Zhang. Verifier-bound mechanism discovery: Pre-registered LLM proposers for mobile crowdsensing incentive design. Zenodo. doi:10.5281/zenodo.23433817, 2026.

25. P. Chen and S. Zhang. Verifier-bound rule discovery for task assignment in mobile crowdsensing: A pre-registered third-domain replication. Zenodo. doi:10.5281/zenodo.23433820, 2026.

26. P. Chen and S. Zhang. Does the train-gain/generalization relation transfer? A pre-registered cross-domain test of inverted-U behavior. Zenodo. doi:10.5281/zenodo.23433822, 2026.

27. P. Ammann, J. Ofutt. Introduction to Software Testing. Second edition, Cambridge University Press, 2017.

28. E. T. Barr, M. Harman, P. McMinn, M. Shahbaz, S. Yoo. The oracle problem in software testing: a survey. IEEE Transactions on Software Engineering, 41(5):507–525, 2015.