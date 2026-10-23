# Superintelligent Software Engineering: Governance That Scales with Capability

Mohammad Mehdi Gholinejad

OSTEON Labs

ORCID 0009-0004-4192-9825

Abstract: Software agents are being trained to exceed human engineers; we argue that capability alone does not produce better software engineering. We define superintelligent software engineering as the discipline of building and evolving software with agents whose capability exceeds that of human engineers, in which governance (gates, contracts, decisions and decay control) scales with capability. The argument rests on three claims. First, correctness in a project rests on decisions and contracts that the code does not carry. Second, knowledge decays. Third, once an agent outpaces its reviewers, reading catches a shrinking share of the changes built on stale knowledge. A simple model shows that, past a fixed review capacity, escaping changes grow linearly with throughput while the share that reading catches falls toward zero; governance can lower the slope, capability alone cannot. An exploratory snapshot of 452 of the most-starred public GitHub repositories finds a root-level or .github agent context file in 48.67% (95% confidence interval 44.1 to 53.3); in the median file that names repository paths, 9.5% to 22% of those paths do not resolve, depending on how a reference is read. We then present OSTEON, a discipline for the regime, released with a DOI (10.5281/zenodo.23436020): each project gets a governed knowledge repository of decisions and contracts, every change is checked mechanically against it, and the knowledge itself is checked for decay at every session start; planning is gated, and sessions record what they learn without writing rules. We report no measured efect of OSTEON, mark claims about its mechanisms as opinion and the author’s practice heuristics as design rationale, and state the questions that would falsify the thesis.

Index Terms: superintelligent software engineering, agentic software engineering, governance, architecture decision records, context decay, agent context files, coding agents

## I. Introduction

Coding agents now resolve real issues in real repositories, and benchmarks such as SWE-bench [1] track how often they succeed. One route to better software engineering is to train more capable agents; we call it the training route, and it now states its goal plainly. Self-play SWE-RL trains a single agent to inject and repair bugs of rising dificulty in real codebases, with no human-labelled issues or tests, and presents itself as a first step toward training superintelligent software agents [2]. If that route succeeds, the agents that work in a codebase will, on much of the work, exceed the engineers who are responsible for it. Fig. 1 summarises the argument of this paper.

This paper asks what engineering discipline that regime requires. Our answer is that capability is necessary and not suficient. Software engineering has never been only the production of code that passes its tests. It is the production of code that is correct with respect to what a project has decided: the trade-ofs it accepted, the contracts it holds with its neighbours, and the state of the system it actually runs. Most of that knowledge lives outside the code. It changes, and parts of it go stale without anyone noticing. In a snapshot of 452 of the most-starred public GitHub repositories, 48.67% carry a root-level or .github file written to brief coding agents. In the median such file that names repository paths, between 9.5% and 22% of those paths do not resolve, depending on how a path reference is read (Section IV). An unresolved path is a mismatch between a file and its tree, not shown to be a wrong instruction; it is one observable form of the drift this paper is about. An agent that consults stale knowledge may build the wrong change, and cannot tell that it has. We expect a more capable agent to build it faster, at larger scope, and to defend it better. [Opinion]

![](images/5a2fa2d3843d98a5250c2ceafb0911eba9fd6e66b073029edd25c05cf2c3df8c.jpg)  
Fig. 1. Why capability is not enough (Section III). Capability raises throughput �, a consulted knowledge unit is stale with probability �, and review capacity � is fixed, so once � > � the share of stale-built changes that reading catches falls as �/�, and the escaping number grows linearly with � at the full rate of stale-built change (Eq. (2); a model, not a measurement). OSTEON acts on the two factors governance acts on: its session-start decay check is built to lower �, and its mechanical gates to check every change (� = 1).

Our thesis is therefore a scaling claim: governance has to scale with capability. Four things must grow in reach and fall in cost as agents grow in capability: the gates a change passes, the contracts it is checked against, the decisions it must respect, and the control of how that knowledge decays. If they do not, we argue, more capability produces more confident error.

## C<sub>on</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub>

1) A definition of superintelligent software engineering as a discipline, and of the regime it addresses, distinguished from the agent that works in it and from the era that precedes it (Section II).

2) An argument that capability alone does not reduce the changes built on stale knowledge, with a simple model of how such changes escape review as throughput rises (Section III).

3) An exploratory measurement of agent context files in 452 of the most-starred public GitHub repositories, released with its scripts and dataset so that its per-file statistics can be recomputed (Section IV).

4) Seven requirements that we propose for any discipline for the regime (Section V).

5) OSTEON, a concrete discipline built to these requirements, released as an open artifact with a DOI [3] (Sections VI and VII).

6) A research agenda that states the thesis as testable claims, with the evidence that would refute each, and one open design question (Section XI).

## E<sub>v</sub>id<sub>ence s</sub>t<sub>a</sub>t<sub>us</sub>

This is a vision paper with a solution and one exploratory measurement. It reports no measurement of OSTEON and claims no measured efect. Section IV examines one premise of the argument, that written project knowledge drifts from the system it describes, in public agent context files. It is a single observational snapshot: it describes, and it tests no cause. Sections VI and VII describe mechanisms, and each can be checked against the released artifact [3]. Where a sentence claims what a mechanism achieves in practice rather than what it does, it is an opinion, and it is marked [Opinion] where it stands. The scenarios in Section VII are constructed to show a mechanism; they are not drawn from data. Section VIII states design heuristics from the author’s own practice; each is marked [Experience] and is likewise not evidence. Section VI-H gives a descriptive reading of OSTEON’s own version history, ofered as a check on the mechanisms, not as a measurement of their efect.

## II. Defining the Discipline

Definition. Superintelligent software engineering is the discipline of building and evolving software with agents whose capability exceeds that of human engineers, in which governance (gates, contracts, decisions and decay control) scales with capability.

The definition is the one the OSTEON release carries [3]; this paper argues it. Each of its three parts does work.

Building and evolving. The definition covers the whole life of a system, not its first release. Lehman observed that a program used in the real world must keep changing or become progressively less useful [4], and Parnas described how software ages as the world around it moves [5]. The regime we address is mostly evolution: agents changing systems whose decisions were taken earlier, by other people, for reasons that may no longer hold.

Agents whose capability exceeds that of human engineers. We do not require superintelligence in the general sense of an intellect that exceeds human performance in virtually all domains of interest [6]. We require something narrower and nearer: that on the work delegated to it, an agent produces output that the project’s engineers could not produce, or could not verify line by line, in the time available. Capability is measured against the humans who govern the project. The regime therefore arrives unevenly, task by task and project by project, long before any general superintelligence.

The condition is relative, and it includes speed. The regime is defined by the gap between an agent’s output and the project’s capacity to verify it, not by intelligence in the general sense: an agent can enter it by depth, by scope or by speed. An agent no better than its reviewers but much faster also produces more than they can verify in time, and the model of Section III-C treats the two cases alike, because capability enters it only through throughput � and scope �. We keep the term because the training route names superintelligent agents as its target [2], because depth is the case in which verification fails even when there is time, and because we address the regime in which both the quality and the volume of agent output exceed what reviewers can check; the fast but ordinary agent is a transitional case that the same governance covers.

Governance scales with capability. This is the defining property. A practice in which agents become more capable while the governance around them stays fixed is not superintelligent software engineering. It is ordinary software engineering run faster, and Section III argues that it fails in a predictable way. A practice meets the definition when it meets the requirements of Section V; whether its governance then scales in fact is the empirical question of Section XI.

The definition is stipulative: it names the discipline by the property we argue for, and does not by itself show that the property is needed; a reader who rejects Section III can accept the definition and hold the discipline empty. We keep apart the regime, a descriptive condition on verification capacity (Section II-A); the discipline, a normative practice for it; and the claim that the practice is needed, which Section III argues and Section XI says how to refute.

## A<sub>.</sub> Th<sub>ree</sub> <sub>reg</sub>i<sub>mes</sub>

It helps to separate three regimes by who can verify a change (Fig. 2).

• Assisted. A human writes or closely reads every change, and the reviewer understands the change at least as well as its author (in the terms of Section III-C: every change is reviewed and detection is high).

• Delegated. An agent writes and a human reads every change, which is possible only because throughput is held at or below review capacity: review is the binding constraint (� = 1 because � is capped at �). Most current practice sits here. As we read them, the settings that SE 3.0 [7] and agentic software engineering [8] describe sit here too, because in both a human still reads and approves each change or its evidence pack.

![](images/7e9c2c47ae1f64c816fcf517559712b02e9a86c0b69070cb64e2419ea8d15e4a.jpg)  
Fig. 2. The three regimes, separated by who can verify a change. In the assisted and delegated regimes a human verifies by reading the change (thick blue arrows mark where verification sits); in the delegated regime review is the binding constraint. In the superseded regime the agent’s output exceeds what the humans responsible can read in the time they have, so reading at best samples (dashed arrow); human verification moves to the decisions and contracts, and the step from that knowledge to the change is mechanised (Section III-D). The shaded area is the scope of superintelligent software engineering: the superseded regime and the transition into it.

• Superseded. Throughput exceeds review capacity, so reading samples $( \rho < 1 )$ , or the reviewer can no longer judge a change even when reading it (� falls).

Superintelligent software engineering is the discipline for the superseded regime and for the transition into it. The transition is gradual and mixed: within one project, a schema migration may be superseded while a pricing rule is still assisted. A project can stay delegated by capping its agents at review capacity. That forgoes the capability. The superseded regime is entered by projects that do not cap it, and our claim is conditional on entering it.

## B<sub>.</sub> Wh<sub>y</sub> <sub>name</sub> <sub>a</sub> di<sub>sc</sub>i<sub>p</sub>li<sub>ne</sub>

Recent work has named the agent (Self-play SWE-RL [2]), the era (SE 3.0, an AI-native era with its technology stack [7]), the human-agent duality (agentic software engineering, with workbenches for SE for Humans and for SE for Agents [8]) and the trust problem (Roychoudhury et al., who argue that agent-written code needs trust at least equal to that placed in human practice and look to analysis tools paired with the agent to supply it [9]), but not the discipline. We use discipline in the sense of a defined body of practice, not of an established research community. None of these works makes the scaling of governance the defining property of a practice for the regime in which agents outstrip their reviewers. We are not aware of an earlier definition of superintelligent software engineering as a discipline; a search for the phrase on 2026-10-17 found it only as the target of agent training [2].

## III. Why Capability Is Not Enough

The training route improves what an agent can do with the knowledge it has. Three claims explain why that is not the same as engineering a project well: correctness, decay and throughput (Sections III-A to III-C). Two further points say what follows for oversight and for authority (Sections III-D and III-E).

## A<sub>.</sub> C<sub>orrec</sub>t<sub>ness</sub> i<sub>s re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> k<sub>now</sub>l<sub>e</sub>d<sub>ge</sub> th<sub>e co</sub>d<sub>e</sub> d<sub>oes no</sub>t <sub>carry</sub>

A change is correct relative to an intent. Some intent is written in tests, and the training route uses exactly that: in selfplay SWE-RL each bug is specified by a test patch [2]. Much of a project’s intent, however, is neither in its tests nor recoverable from its code: why a retry is capped where it is; which team owns a field, and in what unit; which approach was tried and rejected; which part of the system is in production and which is a prototype. These are decisions and contracts. Architecture research has long treated decisions as first-class knowledge that the code does not preserve [10], [11]. The lightweight decision record, a short written record of one decision and its reasons, exists because that rationale is otherwise lost [12]. A more capable agent infers more from code. It cannot infer a decision that left no trace, and it cannot know that a decision it does infer has since been reversed.

## B. Project knowledge decays

The knowledge that does get written down drifts from the system it describes. Documentation that was true when written goes stale as the code, the deployment and the organisation move; code and its comments co-evolve only loosely [13], codecomment inconsistencies are common in mature projects [14], and outdated content is among the most reported documentation problems [15]. A stale fact can be worse than a missing one: an agent treats it as current and acts on it with full confidence. Section IV measures one possible symptom of this drift in public repositories.

A recent controlled evaluation of repository context files bears on this. Gloaguen et al. found that providing context files such as AGENTS.md does not generally improve task success, while it raises inference cost by over a fifth on average. Instructions in the files are followed, but repository overviews do not help [16]. Gloaguen et al. themselves conclude that such files are useful for specifying non-standard coding practices. Our own addition, not theirs, is that the lever is not more context but current context: what an agent cannot derive from the code is a small set of non-standard, project-specific facts, and their value depends on whether they are current. Context that is ungoverned, unversioned and unchecked for decay is cost with no assurance of benefit.

## C. Capability amplifies the rate of stale-built change

A simple model makes the scaling explicit. It is a model, not a measurement.

Let an agent make changes at rate �. Each change consults � units of project knowledge (a decision, a contract clause, a state record). Let � be the probability that a consulted unit is stale, that is, that its source has changed since it was last verified. If staleness is independent across units, the rate of changes that rest on at least one stale unit is

![](images/988c2d3da6188251ad7cce75150350674c7252b28ced4323bac3dade7f28128c.jpg)  
Fig. 3. The escape model of Eq. (2), normalised by review capacity: $E _ { \mathrm { e s c a p e } } / c$ against $r / c ,$ with = min(1 �/� ), � = 5 and $d = 0 . 8$ . Solid lines show three staleness levels � under review by reading alone. Below $r = c$ (grey vertical line) every change is reviewed and the escape rate grows at slope $( 1 - d )$ times the stale fraction; above it the reviewed fraction falls as �/� and the slope rises to the full stale fraction. Broken lines show the $s = 0 . 0 5$ case with governance: decay control lowers � to 0.01 (dashed), and mechanical gates check every change $( \rho = 1 )$ with detection probability $g = 0 . 9$ in place of � (dotted). All parameter values are illustrative, not measured; the script and parameters ship with the figure.

$$
E _ { \mathrm { s t a l e } } = r \left[ 1 - ( 1 - s ) ^ { k } \right] .\tag{1}
$$

A stale unit need not be wrong, so $E _ { \mathrm { s t a l e } }$ bounds, rather than counts, the changes that knowledge decay makes wrong. Suppose humans can review changes at a fixed capacity �, so the fraction reviewed is $\rho = \operatorname* { m i n } ( 1 , c / r )$ , and a review catches a stale-built change with probability �. The rate of stale-built changes that escape review is then

$$
E _ { \mathrm { e s c a p e } } = r \left[ 1 - ( 1 - s ) ^ { k } \right] \left( 1 - \rho d \right) .\tag{2}
$$

A stale-built change is one exposed to a stale unit, and only some fraction of them are wrong; a constant factor for that fraction would scale every curve in Fig. 3 and change none of the conclusions, so we omit it. Fig. 3 plots Eq. (2) for illustrative parameter values.

Three consequences follow. First, capability raises � and, we assume, �, because more ambitious changes consult more knowledge; both raise $E _ { \mathrm { e s c a p e } }$ . Second, the escape rate grows linearly in throughput on both sides of �. Below � its slope is $( 1 - d )$ times the stale fraction $1 - ( 1 - s ) ^ { k }$ , because a reader misses some stale-built changes even when every change is read. Above � the reviewed fraction $\rho$ falls as $c / r$ and $E _ { \mathrm { e s c a p e } } = \left[ 1 - ( 1 - s ) ^ { k } \right] ( r - c d )$ , whose slope is the whole stale fraction, as if no review existed. The threshold at $r = c$ changes the slope, not the kind of growth; it is where reading stops mattering, and for any fixed $s > 0$ the growth is without bound. Third, the model holds � outside the agent: staleness is a property of the knowledge, not of the agent, so training does not act on it. For a claim that the code does not carry (a decision, a contract with another team, the state of production), an agent can detect staleness only if the unit carries provenance (a record of the sources it came from) [17] and changes to those sources can be observed. That metadata is not in the code.

Of the terms in Eq. (2), capability acts on � and �. Governance acts on two factors without adding reviewers: the staleness factor, through �, and the review factor, through $\rho$ and �. Decay control lowers � by finding stale units before they are consulted. Gates act on the review factor. By mechanical we mean run on every change at no human cost. A gate runs on every change, so it sets $\rho$ to 1, and it holds a change that rests on a unit already flagged as stale with its own probability � in place of a reader’s � (Fig. 3). Its other job, catching a change that contradicts a current record, is a second error channel that Eq. (2) does not count (the second scenario of Section VII shows it). The two factors are not independent: a gate cannot detect a record that is itself stale and unflagged, and a stalebuilt change that agrees with such a record passes it. The gate’s $g$ is therefore bounded by the currency of the record it checks against, which is why decay control comes first, and why the gate curve in Fig. 3 assumes a current record. In OSTEON 0.1.0 the decay check and the lints are scripts; the review gate is a procedure the agent executes against the record, so its � is a property of the agent and the record together, which is why Q5 of Section XI asks whether it holds as capability rises.

A project could also hold � at �, which is the delegated regime’s answer and forgoes the capability, or raise �: with more human reviewers, which is what the superseded regime by definition can no longer do, or with agent reviewers, which the fifth objection of Section IX takes up (an agent that reviews against the record is, in our terms, a gate). Both factors are governance, and neither improves by itself when the agent improves. This is the sense in which governance has to scale with capability.

## D. Verification moves from the change to the knowledge

When the supervised system exceeds its supervisor, oversight has to be designed so that the supervisor can still judge something that matters. This is the problem of scalable oversight [18], [19]. In the superseded regime the reviewer cannot judge every change. The reviewer can still judge the decisions and contracts the change is checked against, because those are short, stable, human-approved and far fewer than the changes. Superintelligent software engineering moves human verification from the change to the knowledge, and mechanises the step from knowledge to change.

## E. Authority does not follow capability

A decision records a trade-of among things a project’s owners answer for: cost, risk, obligations to users and to other teams. An agent may predict the consequences of a decision better than any engineer and still not be the party entitled to take it. We hold that the boundary of human authority should not move as capability grows; what should fall is the cost of exercising it. [Opinion]

## F<sub>.</sub> Th<sub>e</sub> <sub>sca</sub>li<sub>ng</sub> <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>e</sub>

Together these give the principle the definition names. We say governance scales with capability when five conditions hold as capability rises. Two are scaling properties in the strict sense, cost and coverage; the other three, currency, authority and learning, are invariants that must survive the scaling. In the model’s terms, governance scales when � stays at 1 and � does not rise as � rises.

• Cost. Governance cost per change falls as change rate rises, so it is not skipped under load. Checks that can be mechanical are mechanical.

• Coverage. What is governed grows with the scope agents take on. Every durable choice an agent relies on is recorded, and every seam it crosses has a contract.

• Currency. Every governed fact carries its provenance. Decay is checked before work starts rather than discovered after a failure.

• Authority. Agents propose; accountable humans dispose: they make the final call on anything that changes a decision or a contract. Human efort attaches to decisions, which are rare, not to changes, which are many.

• Learning. The process improves from what sessions learn, without any session legislating for the next.

## IV. Context Decay in Public Repositories: An Exploratory Measurement

Section III rests on the premise that written project knowledge drifts from the system it describes. Agent context files are a common way to give coding agents written project knowledge, and in public repositories they can be observed directly, so we took one snapshot of them. The measurement concerns the premise, not OSTEON. It is exploratory and observational: it tests no hypothesis and supports no causal claim. It asks three measurement questions (MQ1 to MQ3).

MQ1 Prevalence. How many popular repositories carry a root-level or .github agent context file?

MQ2 Currency. How long ago was each file last updated, and how many commits has its repository received since?

MQ3 Resolution. What share of the repository paths a file names do not resolve in the repository tree at the snapshot?

Earlier empirical studies characterise the content, adoption and evolution of context files in GitHub repositories [20], [21], [22]; MQ3 applies a link check, routine in documentation pipelines, to every path a context file names, against its repository’s tree.

## A<sub>.</sub> M<sub>e</sub>th<sub>o</sub>d

Frame. We queried the GitHub search API for repositories matching

$$
\mathtt { s t a r s : } > 1 0 0 0 0 \mathtt { f o r k : } \mathtt { f a l s e \ a r c h i v e d : f a l s e }
$$

sorted by stars in descending order, and took the first five pages of 100 results. The query matched 5 346 repositories, and the five pages returned 500, none marked incomplete and none duplicated across pages (Table I). We excluded 47 repositories with no primary language, a rule meant to remove lists and books. One repository matched a list, fixed before collection, of names the author does not report on for reasons of confidentiality; it was excluded, and it is counted but not named. All 452 eligible repositories were measured, no repository tree was unavailable, and every context file’s content and path history were available. For 7 repositories the recursive tree listing was truncated, so the context-file paths were checked one by one through the contents API. The snapshot date, 2026-10-18 (UTC), is the reference for every age; collection used read-only endpoints only. The eligible repositories hold between 50 069 and 554 124 stars and are assigned 32 primary languages; Python (130), TypeScript (85), JavaScript (50), Rust (33) and Go (32) account for 330 of the 452. The frame is released with the dataset (frame.csv).

Context files. At each repository’s default branch we looked for five file names, matched exactly and case-sensitively (Table I):

• at the repository root: AGENTS.md, CLAUDE.md, GEMINI.md, .cursorrules;

• in .github: copilot-instructions.md.

AGENTS.md or CLAUDE.md files elsewhere in the tree are nested files; we counted them but did not analyse them. The sample is the set of repositories with at least one root-level or .github context file, and each such file is one unit of analysis.

Pre-specification. The frame, the file list and the metrics were written down verbatim, in the study’s protocol and in the header of the collection script, before collection began. One robustness change was made after a small test on a separate cache and before the full run: the tree call was set to tolerate two further HTTP error responses (403 and 451); the test data were not used, the protocol records the change in a dated note, and no tree call in the full run failed, so the change afected no result. The summary statistics (medians and quartiles, the pooled rate, the share of files with any unresolved reference) and the single association reported, commits since update against the dangling-reference rate, were implemented in summarize.py before collection and were not changed afterwards; every other comparison in this section is labelled post hoc, including the audit in Section IV-D and the intervals and breakdowns added at revision. The pre-specification is recorded only in the released files; it was not registered with a third party, so it rests on the author’s word, the file record and the dated deviation note.

## B<sub>.</sub> M<sub>e</sub>t<sub>r</sub>i<sub>cs</sub>

For each context file in the sample we compute five measures.

• Age. Days from the oldest commit that touches the file to the snapshot date.

• Days since update. Days from the newest commit that touches the file to the snapshot date.

• Updates. The number of default-branch commits that touch the file’s path. GitHub’s path history does not follow renames, so a moved file is aged from the move.

TABLE I  
Collection funnel. Every count is read from the study’s funnel record; counts that are zero (duplicates across pages, trees unavailable, files unreadable or without path history) are stated in the text.
<table><tr><td>Stage</td><td>Count</td></tr><tr><td>Repositories</td><td></td></tr><tr><td>Matching the search query</td><td>5346</td></tr><tr><td>Returned by search (five pages of 100)</td><td>500</td></tr><tr><td>Excluded: no primary language</td><td>47</td></tr><tr><td>Excluded: privacy denylist (counted, not named)</td><td>1</td></tr><tr><td>Eligible and measured</td><td>452</td></tr><tr><td>Truncated tree, checked through the contents API</td><td>7</td></tr><tr><td>Sample: a root-level or . github context file</td><td>220</td></tr><tr><td>Nested files only (outside the sample)</td><td>14</td></tr><tr><td>Context files in the sample</td><td></td></tr><tr><td>All root-level and.github files</td><td>367</td></tr><tr><td>AGENTS.md</td><td>189</td></tr><tr><td>CLAUDE.md</td><td>112</td></tr><tr><td>.github/copilot-instructions.md</td><td>49</td></tr><tr><td>GEMINI.md</td><td>15</td></tr><tr><td>.cursorrules</td><td>2</td></tr><tr><td>Stored as symbolic links</td><td>35</td></tr><tr><td>With at least one path-like candidate</td><td>269</td></tr></table>

• Commits since update. The number of default-branch commits dated at or after the file’s last update, excluding that update. It measures how far the repository moved while the file stood still.

• Dangling-reference rate. The share of a file’s distinct path-like references that do not resolve in the tree at the snapshot. Candidates are inline code spans and Markdown link targets that look like repository paths. A candidate contains “/” or ends in an extension from a fixed list. It contains no whitespace, is not a URL, and passes the further filters of the released protocol (README.md, operational detail 6). Fenced code blocks are removed first. A candidate resolves if it equals a path in the tree, or is a directory prefix of one, either from the repository root or relative to the context file’s directory. The rate is undefined for a file with no candidates.

Prevalence is the number of sample repositories over the number of eligible repositories. Every distribution is skewed, so we report medians with 25th and 75th percentiles (linear interpolation) and use the Spearman rank correlation, � , for association. Where a confidence interval is given it is a Wilson 95% interval for a proportion or a seeded percentile bootstrap (10 000 resamples) for a median; the study is descriptive, so we report intervals and no tests. An unresolved path is a mismatch between a file and its current tree. It may reflect drift, a generated or planned path, or the extraction heuristic. It is not proof that an instruction is wrong.

## C<sub>.</sub> R<sub>esu</sub>lt<sub>s</sub>

MQ1: Prevalence. Of the 452 eligible repositories, 220 (48.67%; 95% confidence interval 44.1 to 53.3; 44.0% over all 500 returned repositories) carry at least one root-level or .github context file. Prevalence varies by language:

![](images/45d8f35292675b8d1b9bbc6bb8f24d71e1ee1afae69771cd84df9965ec0cb410.jpg)  
Fig. 4. Currency of the 367 context files: commits to the repository since the file’s last update, against days since that update. Both axes are logarithmic; the vertical axis is linear below 1 because zero counts occur. Filled markers are files with at least one path-like candidate, shaded by dangling-reference rate; hollow markers are files with none. The plot is descriptive.

TypeScript 62 of 85 and Go 23 of 32, against Python 53 of 130, JavaScript 19 of 50 and Java 4 of 15 (post hoc breakdown). Together the sample repositories hold 367 files, a median of 2 per repository (25th percentile 1, 75th percentile 2, maximum 4). The most frequent names are AGENTS.md (189 files) and CLAUDE.md (112 files); Table I gives the full breakdown. Of the 367 files, 35 are stored as symbolic links. Nested AGENTS.md or CLAUDE.md files are common as well: 102 repositories hold 1 299 of them, 88 of these repositories (1 277 files) in the sample and 14 outside it.

MQ2: Currency. Table II and Fig. 4 summarise the per-file measures. Context files have a median age of 220 days (25th to 75th percentile 144 to 324.5) and a median of 48 days since their last update (10 to 167). They have a median of 5 updates (1 to 18). Since their last update, their repositories have received a median of 229 further commits (49 to 852). The distribution has a heavy upper tail, reaching 91 084 commits, because the count includes every commit dated after the update, merged branches included (bootstrap 95% interval for the median, 168 to 307). The 35 symbolic links are rows whose text is a target name, so they need no update when their target changes; over the 332 other files the medians are 40 days since update (25th to 75th percentile 9 to 134) and 197.5 commits since update (45 to 663.5), and the age median is 219.5, against 220 over all files. By file name (post hoc), AGENTS.md files are the most current (median 17 days since update, 125 commits since) and CLAUDE.md files the least (124.5 days, 457 commits); median dangling rates are 0.22 and 0.27, and 0.15 for copilot-instructions.md. The cell counts for GEMINI.md (15 files, 3 with candidates) and .cursorrules (2 files) are too small to read.

MQ3: Resolution. Of the 367 files, 269 contain at least one path-like candidate (Fig. 5); none of the 35 symbolic links has one, because a link’s text is its target. Among these 269 files, the median dangling-reference rate is 0.22 (25th to 75th percentile 0 to 0.42; bootstrap 95% interval 0.18 to 0.27). Of these files, 192 (71.38%; 95% confidence interval 65.7 to 76.4) have at least one unresolved reference. Pooled over files, 2 003 of 6 172 candidates (32.45%) do not resolve. Commits since update and the dangling-reference rate are not positively associated $( r _ { \mathrm { S } } = - 0 . 1 5 , n = 2 6 9 \mathrm { ~ f i l e s } )$ . Files cluster within repositories (112 of the 220 sample repositories contribute more than one file), and at repository level, with candidates pooled within each of the 205 repositories that have any, the median rate is 0.27 and the coeficient against the largest commits-since-update among a repository’s files is 0.03 (post hoc). Files that have stood still through more commits do not show more unresolved references in this sample at either level. We read nothing causal into this: repositories difer in size, activity and the way their files refer to paths.

TABLE II  
Per-file summary statistics (pre-specified protocol). Age and days since update are in days at the snapshot date; files, updates and commits are counts; the dangling-reference rate is a fraction of a file’s candidates.
<table><tr><td>Measure</td><td>n</td><td>Median</td><td>P25</td><td>P75</td><td>Min</td><td>Max</td></tr><tr><td>Files per sample repository</td><td>220</td><td>2</td><td>1</td><td>2</td><td>1</td><td>4</td></tr><tr><td>Age</td><td>367</td><td>220</td><td>144</td><td>324.5</td><td>12</td><td>689</td></tr><tr><td>Days since update</td><td>367</td><td>48</td><td>10</td><td>167</td><td>1</td><td>549</td></tr><tr><td>Updates</td><td>367</td><td>5</td><td>1</td><td>18</td><td>1</td><td>631</td></tr><tr><td>Commits since update</td><td>367</td><td>229</td><td>49</td><td>852</td><td>0</td><td>91 084</td></tr><tr><td>Dangling-reference rate</td><td>269</td><td>0.2222</td><td>0</td><td>0.4231</td><td>0</td><td>1</td></tr></table>

![](images/8ac28951d4fde926821d931b479f78beac0ce2fd737e90a6cdc2c35fd321c1a1.jpg)  
Fig. 5. Empirical cumulative distribution of the dangling-reference rate over the 269 context files with at least one path-like candidate (pre-specified reading). The dashed line marks the median. The step at zero is the share of files whose candidates all resolve.

## D. Sensitivity of the dangling-reference rate (post hoc)

After seeing the primary results, we audited the 2 003 unresolved references and, after inspecting the most frequent unresolved tokens, defined two classes that are unlikely to be meant as tracked repository paths: bare extensions such as .ts (97), and paths under build or local folders such as dist or node\_modules, which repositories usually do not commit (95). Together they make up 9.59% of unresolved references. Of the remaining 1 811, 1 115 (61.57%) are bare file names with no $^ { 6 6 } / { } ^ { 5 } .$ . The protocol resolves such names only at the root or next to the context file, so a file of that name deeper in the tree counts as unresolved.

TABLE III  
Dangling-reference measures under the pre-specified reading and two post hoc readings. A: without bare extensions and conventionally untracked build or local paths. B: as A, and only candidates that contain $" / "$ . B also leaves out the three withheld candidates, whose form is unknown.
<table><tr><td>Measure</td><td>Pre-specified</td><td>A (post hoc)</td><td>B (post hoc)</td></tr><tr><td>Files with candidates</td><td>269</td><td>268</td><td>236</td></tr><tr><td>Median rate</td><td>0.2222</td><td>0.1818</td><td>0.0952</td></tr><tr><td>25th percentile</td><td>0</td><td>0</td><td>0</td></tr><tr><td>75th percentile</td><td>0.4231</td><td>0.3917</td><td>0.2857</td></tr><tr><td>Files with any unresolved</td><td>192</td><td>183</td><td>145</td></tr><tr><td>as % of files</td><td>71.38</td><td>68.28</td><td>61.44</td></tr><tr><td>Candidates, pooled</td><td>6172</td><td>5940</td><td>3705</td></tr><tr><td>unresolved</td><td>2003</td><td>1811</td><td>696</td></tr><tr><td>as  $\%$  of candidates</td><td>32.45</td><td>30.49</td><td>18.79</td></tr></table>

Table III reports the measures under two post hoc readings. Reading A drops the two classes. Reading B also keeps only candidates that contain $^ { 6 6 } / 7 ^ { 9 } .$ . The median rate falls from 0.2222 to 0.1818 under A and to 0.0952 under B. The share of files with any unresolved reference falls from 71.38% to 68.28% and 61.44%. The headline rate therefore depends strongly on how references are read.

## E<sub>.</sub> Wh<sub>a</sub>t th<sub>e measuremen</sub>t <sub>s</sub>h<sub>ows, an</sub>d <sub>w</sub>h<sub>a</sub>t it d<sub>oes no</sub>t

The snapshot shows that the mismatch the premise of Section III predicts is common; it does not show how it arose. About half of the popular repositories we measured brief their agents through context files. For half of those files, the repository has received 229 or more commits since the file last changed. Under all three readings, most files that name repository paths name at least one that does not resolve. The one association we examined does not support a simple accumulation story: files that stood still through more commits do not have more unresolved references, at file or at repository level.

The measurement does not show that unresolved references mislead agents, that they cause defects, or that any governance practice, OSTEON included, would reduce them. Nor is it a judgement of any project: a reference that does not resolve may name a file the build generates, a path in another repository, or a file that is planned, and whether a given referent moved, was never created, or is generated is not determined. Its role is narrower: written path references that do not resolve against the current tree are common in public repositories today, and they are one observable form of the mismatch that R5 (decay is checked at the start) targets. It supplies a cross-sectional reference point for Q3 of the research agenda (Section XI): a level at one date, not a decay rate.

## F<sub>.</sub> Th<sub>rea</sub>t<sub>s</sub> t<sub>o</sub> <sub>va</sub>lidit<sub>y</sub>

Construct validity. An unresolved reference is a symptom, not a defect. Candidates are found by pattern, not by understanding.

Some flagged strings are not meant as paths, such as a package name that ends in a file extension, a path the build creates, a Git reference such as origin/main, a branch-name prefix such as feat/ read as a directory, or the literal N/A; the released candidate list shows each one, and the post hoc readings remove none of these when the token contains a slash. Some real references are missed, such as paths in prose without code formatting, paths inside fenced blocks, and bare names without an extension. The post hoc readings bound part of this efect, not all of it. The rate rises with the number of candidates a file holds $( r _ { \mathrm { { S } } } = 0 . 4 1$ over the 269 files; candidates per file have a median of 13, 25th to 75th percentile 4 to 31; of the 35 files with a single candidate, 3 have it unresolved). This is consistent with more drift in path-heavy files and with a heuristic whose false-positive share grows with file length, and the snapshot cannot separate the two. Commits since update counts every commit dated after the update, merged branches included, so it can overstate activity; we report medians for this reason. For 2 files the context-file commit was not found on the last page of the activity range, so their count is not reduced by one.

Internal validity. The frame, the file list, the metrics and the summary statistics were fixed before collection; the sensitivity audit and the breakdowns added at revision were not, and they are labelled post hoc. The 35 symbolic links, whose text is the link target, enter the age and activity statistics, where they are the stalest rows; Section IV-C gives the medians without them. They have no candidates and so no dangling-reference rate. The unit of analysis is the file, and files are not independent across a repository; the repository-level figures in Section IV-C are the check. For the 7 repositories with truncated trees, the nested-file counts are lower bounds.

External validity. The frame is the most-starred non-fork, non-archived repositories, all above 50 000 stars; neither the prevalence nor the resolution figures generalise to GitHub as a whole, still less to private projects. The no-primary-language rule removes only repositories to which GitHub assigns no language; a few documentation repositories with a Markdown primary language remain eligible (4 of 452, none in the sample), which can only lower the measured prevalence. File names are matched exactly; lower-case variants and per-directory rule folders are outside the protocol. Nested files are counted but not analysed.

Reliability. Search ranking and repository contents change, so a rerun on a later date will difer. The scripts, the per-file data and the per-candidate evidence are released, so every per-file statistic can be recomputed; the funnel counts are in funnel.json, and rerunning collect.py repeats the collection against the live API. Three candidate strings that matched the confidentiality list are withheld in the released candidate list; they keep their resolution flag and enter the pre-specified reading and reading A, not reading B.

## V. Requirements for a Discipline

The four objects the thesis names (gates, contracts, decisions and decay control) and the five conditions of Section III-F become testable when stated as requirements on a practice. We propose seven. They were written together with OSTEON, and Table IV shows the correspondence; we do not claim they were derived independently of the design, and a reader should test them against other practices.

R1 One owner per fact. Each fact lives at exactly one level of an explicit hierarchy, the narrowest that fully owns it, and other levels reference it rather than copy it. Copies are where decay hides.

R2 Decisions are records. Durable choices are versioned records with a lifecycle (proposed, accepted, superseded), amended or superseded and never silently rewritten.

R3 Seams have contracts. Every interface between modules has a versioned contract that changes together with the interface, never after it.

R4 Gates take their bar from the project. Review checks a change against the project’s own decisions, contracts and rules, not against generic best practice, and detects when a change invalidates one of them.

R5 Decay is checked at the start. Every governed claim carries provenance to its sources, and a check at the start of every session reports which claims rest on sources that have changed.

R6 Change is planned and gated. Work of any size is planned, the plan passes an explicit gate before execution, and a definition of done states what proves completion.

R7 Learning without self-legislation. Sessions record what they learn, but no session writes the rules that govern the next; a rule changes only through corroboration and human approval.

R1 to R3 address coverage. R4 and R5 address currency and the detection terms of Eq. (2). R6 addresses authority and cost, and R7 addresses learning. A practice that meets all seven still needs evidence that it helps; Section XI says what that evidence would look like.

## VI. OSTEON

OSTEON is a proposed discipline for superintelligent software engineering. Release 0.1.0 is a plugin for the Claude Code agent. It contains fifteen skills (procedures the agent follows; the source repositories hold further skills that the release does not ship), a scafold for the project knowledge repository, and a session-start decay check. Content is under CC BY 4.0 and scripts under MIT [3]. Fig. 6 shows the parts described in this section and how they connect.

## A<sub>.</sub> Ti<sub>ers an</sub>d th<sub>e</sub> BONE

OSTEON places every fact at one of three tiers (R1). The organisation tier (T0) holds the methodology: how to record a decision, version a contract, run curation. The project tier (T1) holds the project’s facts, decisions, contracts and procedures. The repository tier (T2) holds conventions that are true inside one code repository only, in that repository’s own agent instruction file. A fact lives at the narrowest tier that fully owns it. When it could sit at two, it is placed by what would change it, not by its topic. For example, the port a service listens on belongs to the repository whose configuration sets it (T2), while the decision to run that service at all belongs to the project (T1), because diferent events change each. Personal preferences and an individual’s memory sit outside all tiers, and the project’s knowledge never governs or cites them.

![](images/f03e2c0398379bcf5e7954cbeb114fc520ed2681c6f4d1bef0cc9cb0d6b082aa.jpg)  
Fig. 6. OSTEON as described in Section VI. Every fact sits at one of three tiers. The project tier is the bone, a knowledge repository with four layers and a manifest. Agents and humans both read the bone. When watched sources change, the session-start decay check compares each unit’s provenance with a sealed baseline and flags decayed units for curation. Ossification (planned, gated change) produces a change; the review gate checks it against recorded decisions and contracts before it merges, and routes any invalidation to curation. Curation applies only the items a human approves. Capture records what a session learns in a ledger that governs nothing, and a rule enters the register only when a human admits it. Solid arrows carry knowledge or change, dashed arrows mark checks and routing, and thick blue arrows mark human approval.

The project tier is a dedicated repository, the project’s bone. It holds no product code. It holds four layers. L1 is a constitution: the project’s non-negotiables, the rules the other layers may not override, and what may not be decided unilaterally. L2 holds decisions and dated records of the running system’s state as last verified. L3 holds contracts and domain knowledge. L4 holds the project’s own skills: the procedures an agent follows inside the project. A manifest declares which files are knowledge units and which sources are watched. Because the bone is a repository, its knowledge has what code already has: history, branches, review and provenance. The names follow one metaphor: the project’s knowledge repository is its bone, knowledge that has been verified and gated is ossified, and work proceeds by planned ossification (Section VI-F).

## B<sub>.</sub> D<sub>ec</sub>i<sub>s</sub>i<sub>on recor</sub>d<sub>s</sub>

Decision records follow the architecture decision record tradition [12], with a numbered identity and a lifecycle of proposed, accepted and superseded (R2). An accepted record is amended or superseded, never rewritten, so the reason a project once had survives the decision that replaced it. The decision skill also handles the case that matters most for agents, a merged change that silently suspends a decision or an accepted record that no longer matches what the running system does: the mismatch is made explicit as an amendment or a supersession, rather than left as a contradiction between the record and the code.

## C<sub>.</sub> M<sub>o</sub>d<sub>u</sub>l<sub>e con</sub>t<sub>rac</sub>t<sub>s</sub>

Every seam between two modules has a contract (R3): the wire format, fields, units, authentication and failure behaviour that both sides implement against. Contracts follow semantic versioning [23] in the spirit of design by contract [24], and are indexed in a registry. One rule carries most of the weight: a contract change ships in the same wave, that is, in the same change request, as the interface change it describes, never after it. The skill also triggers when one side of a seam is being built against the other side’s source code instead of its written contract, which is the usual way an agent learns an interface it should have read.

## D<sub>.</sub> R<sub>ev</sub>i<sub>ew</sub> <sub>ga</sub>t<sub>es</sub> <sub>an</sub>d <sub>cura</sub>ti<sub>on</sub>

The review skill checks a change against the project’s own bar, derived from its bone layers, rather than against generic practice (R4), before merge and as the self-review before a change is declared ready. Its second job is detection. When a change invalidates a decision, a contract or a claim in a unit, the review routes that to curation instead of passing or failing the change on its own authority. The review skill never edits the knowledge base. One of its checks bears directly on Eq. (2): a change that assumes or contradicts a fact whose state record is past the project’s staleness horizon is held as needs-information until the record is re-verified. In release 0.1.0 that check reads the record’s age, not the decay check’s source-change flag; joining the two is design work, not a result.

TABLE IV  
Requirements for superintelligent software engineering, the OSTEON mechanism for each, and what stays with a human.
<table><tr><td>Requirement</td><td>OSTEON mechanism</td><td>What it is built to prevent</td><td>What stays human</td></tr><tr><td>R1 One owner per fact</td><td>Three tiers (T0 org, T1 project, T2 repository); personal tier outside; reference, never copy</td><td>A copied fact decaying in one place and not the other</td><td>Placing a contested fact</td></tr><tr><td>R2 Decisions are records</td><td>Numbered decision records; proposed, accepted, superseded; amend, never rewrite</td><td>Re-litigated or silently suspended decisions</td><td>Accepting or superseding a decision</td></tr><tr><td>R3 Seams have contracts</td><td>Versioned contracts, registry index, same-wave rule</td><td>One side of a seam changing alone</td><td>Approving a breaking version</td></tr><tr><td>R4 Gates from the project</td><td>Review against the BoNE layers; invalidation routed to curation</td><td>Changes that pass generic review and break a project decision</td><td>Approving each change to the knowledge</td></tr><tr><td>R5 Decay checked at start</td><td>Block-level provenance; sealed baseline; Context Decay Index at session start</td><td>Work built on a fact whose source has moved</td><td>Re-verifying a decayed claim</td></tr><tr><td>R6 Planned, gated change</td><td>Daily cycle; ten-condition start gate; lanes in one session; merge is not completion</td><td>Work started on open questions; unproven completion</td><td>Answering open decisions</td></tr><tr><td>R7 No self-legislation</td><td>Capture ledger that governs nothing; nomination on corroboration</td><td>A session writing rules for the next</td><td>Admitting a rule to the register</td></tr></table>

Curation is a separate skill that audits the bone and proposes additions, updates, removals, merges and splits of knowledge units. It applies nothing without per-item human approval. This is where the authority axis is enforced: the agent does the reading, the comparison and the drafting; a human approves each change to what the project holds true.

## E<sub>.</sub> Th<sub>e sess</sub>i<sub>on-s</sub>t<sub>ar</sub>t d<sub>ecay c</sub>h<sub>ec</sub>k

Every bone carries a decay check wired to the agent’s sessionstart event (R5). A knowledge unit is a block of Markdown (a paragraph, a list item, a table row), not a file, so one changed source decays only the blocks that cite it. A unit is anchored when it carries at least one resolvable provenance token naming a source: a decision record, a dated state record, a contract, a domain document, the constitution, or a file in another declared repository. Against a sealed baseline (the recorded digests of every watched source at the moment of the last applied curation), a unit is decayed when at least one of its sources has changed. The check reports a Context Decay Index,

$$
\mathrm { C D I } = \frac { \mathrm { d e c a y e d ~ a n c h o r e d ~ u n i t s } } { \mathrm { a n c h o r e d ~ u n i t s } } ,\tag{3}
$$

which is undefined, and is reported as such, when no unit is anchored. The index counts exposure, not confirmed error: a decayed unit may still be true, and curation decides. Under the assumption that a session consults anchored units uniformly, CDI is the project’s running estimate of � in Eq. (2); Q3 of Section XI is the measurement of that quantity over time.

Three design choices matter for the scaling argument. First, the check is silent on the healthy path, so its cost to a session is near zero when nothing has decayed. Second, it never prints a number for an unmeasured base: a fresh project reports the state uninitialised, not a reassuring zero. Third, it records every run, clean ones included, deduplicated by state (any number of sessions in one unchanged state add one record), so the series has a denominator: the record can answer what fraction of decay warnings led to curation, not only how many there were. Provenance and tier lints, and an attestation lint that checks that a change to governed knowledge carries an approval record, run alongside it with a self-test.

## F. Ossification: planned and gated change

Change in OSTEON proceeds by ossification (R6) in three steps. A planner skill keeps one dated cycle per day in the bone and sets the bar that every task’s definition of done must clear. A starter skill then checks the plan against ten gate conditions before any work begins; where a condition fails on an open decision, it puts that decision to the owner as one question answerable in one line, rather than reporting that questions exist. Only a plan that passes the gate is executed. Execution may split the plan into independent lanes worked in parallel, by one session with subagents or by several, and finished work is carried back into the plan: a merge is not a completion until the plan says so. A delivery skill reports what a cycle delivered to people outside it, with each claim graded by what proves it.

## G. Capture without self-legislation

The capture skill records what a session learned about the way of working: when something fails, when something is blocked before it lands, or when a check passes on an input that should have failed (R7). It appends to a ledger that governs nothing. At session end it tests whether an observation has independent corroboration; if it does, it nominates the observation and stops. A nomination becomes a rule only through the framework’s own register of named methodologies, under human approval. Sessions learn; they do not legislate.

Two further skills keep the knowledge usable across gaps: an onboarding skill with a fixed read order (constitution, manifest, decision and state indexes, then only the relevant skills), which ends in an orientation whose first action names the fact it depends on; and a standup skill that keeps a per-track continuity journal so that a dormant project can be restarted, and routes blind spots to the role that owns them.

## H<sub>.</sub> Wh<sub>a</sub>t OSTEON’<sub>s own</sub> hi<sub>s</sub>t<sub>ory s</sub>h<sub>ows</sub>

As a check rather than as evidence, we read OSTEON’s own version history. Its development spans 6 repositories and 58 commits by one author over 62 days, on 9 of which commits were made, and holds 54 governed artifacts (files of a governed-knowledge kind). The capture ledger received 28 observations, all on three consecutive active days, which produced 3 nominations and 1 curation record; no nomination can be traced by identifier into a rule, the repositories hold no decision records, and capture stopped 33 days before the last commit. The same reading surfaced a likely defect: 15 of the 28 ledger rows carry timestamps later than the commits that added them, which suggests local time recorded under a UTC marker. Capture, nomination and one curation record were exercised, over three days; decision records were never used; no nomination became a rule; and the six repositories include no bone of OSTEON’s own, so the tiers, the gates and the decay check were not applied to the discipline’s own development. The history shows the mechanisms in partial use by their designer and nothing more, and we report it for that reason. The collector and its output accompany the measurement data as ancillary files.

## VII. How Governance Scales in OSTEON

Table IV maps each requirement to its mechanism. This section relates the mechanisms to the scaling axes and then shows three constructed scenarios.

Cost. The decay check, the lints and the review against recorded knowledge are mechanical, and the check is silent when nothing has decayed, so their cost per change does not grow with the rate of change. Human efort is spent on decisions, contract versions, curation items and plan questions, which we expect to grow far more slowly than the number of changes, because a project’s decisions are fewer and more stable than its edits. [Opinion] Curation items are the exception. A decay warning is raised by a change to a watched source, and watched sources change at a rate that tracks the change rate itself, so warnings may grow with � even when decisions do not. Whether provenance at block granularity keeps warnings per change falling is the point on which the cost axis stands or falls, and it is Q2 of the agenda.

Coverage. The tiers and the contract rule extend governance to every seam that has a contract, and the contract skill is built to notice a seam that lacks one when one side is built against the other side’s source. The onboarding read order makes the governed set the first thing an agent reads, not an optional file it may skip.

Currency. Block-level provenance turns exposure to source change from an unobservable property into a computed one; whether an exposed unit is in fact stale is settled in curation. In the terms of Eq. (2), the decay check is built to lower the efective � by flagging decayed units before a session consults them. The review gate is built to check every change $( \rho = 1 )$ for contradiction with the record, with its own probability � in place of a reader’s �; it detects nothing the record does not hold current, so its value rests on the decay check having run.

Authority. Curation applies nothing without per-item approval, the start gate turns open decisions into one-line questions, and the decision lifecycle forbids rewriting. The agent does more of the work as it becomes more capable; the human disposes of the same things, at lower cost per item. [O<sub>p</sub>inion]

Learning. Capture makes the process improvable from evidence while keeping rule-making behind corroboration and approval.

## A<sub>.</sub> C<sub>ons</sub>t<sub>ruc</sub>t<sub>e</sub>d <sub>scenar</sub>i<sub>os</sub>

The following are constructed to show mechanisms. They are not observations.

A seam drifts. An agent changes the unit of a field one service emits, from seconds to milliseconds, and updates its own tests, which pass. The consumer is untouched. Without a contract, the drift surfaces in production. In OSTEON the field’s unit is a clause of a versioned contract. The same-wave rule requires the contract change in the same change request, the version bump marks it as breaking, and the review gate flags the consumer side as built against the old version.

A decision is silently suspended. A project once rejected client-side caching of a price, for a reason recorded in a decision. Months later an agent, asked to cut latency, adds exactly that cache; the change is fast, tested and well argued. Review against generic practice would approve it. Review against the bone finds the accepted decision, reports the contradiction, and routes it. Either the change is withdrawn, or the decision is superseded by a new record that states why the old reason no longer holds. Either way, the project’s knowledge and its code agree afterwards.

A state record goes stale. A record states that a migration has finished in production; it cites the deployment configuration as its source. The configuration later changes. At the next session start the decay check reports the record’s blocks as decayed, before any agent builds on the claim. Curation re-verifies the record or retires it.

In each case the agent did what it was asked well; what prevents the error is governance that acts on the knowledge the change depended on.

## VIII. Design Rationale from Practice

This section states the heuristics from the author’s own agentassisted work that OSTEON was built to (Table V). They are design rationale, not findings, and the paper ofers no evidence for them. We describe no project, report no data from one, and give no episode. Each heuristic is marked [Experience]; like [Opinion], the marker flags a claim the paper does not measure. The right column of the table says where the mechanism built to each heuristic can be checked (the escape model of Eq. (2), the constructed scenarios of Section VII, or OSTEON’s design, Section VI); it does not turn the heuristic into evidence, and a mechanism designed from a heuristic does not corroborate it. None of the heuristics is about agent capability. Each concerns the knowledge a change relies on, or the check that guards it, which is the object that Section III argues governance must act on.

## IX. Objections

A thesis about the limits of capability invites five objections. We state each in its strongest form.

“A <sub>capa</sub>bl<sub>e enoug</sub>h <sub>agen</sub>t <sub>w</sub>ill k<sub>eep</sub> th<sub>e</sub> d<sub>ocumen</sub>t<sub>a</sub>ti<sub>on curren</sub>t itself.” It can, but only if it can tell which documentation is stale, which requires knowing which source each claim came from and whether that source has changed since. That is provenance, and it is governance. An agent that rewrites documentation without it produces documentation that agrees with the code, not documentation that agrees with what the project decided. The objection is an argument for R5, not against it.

“The tests are the specification.” Tests specify behaviour at the points they exercise. They do not record why a behaviour was chosen, which alternatives were rejected, who owns a field on the far side of a seam, or what is running in production. A change can pass every test and reverse a decision. The training route that uses test patches as specifications [2] is right to do so for learning to repair code; a project needs more than that to be governed.

“G<sub>overnance w</sub>ill <sub>s</sub>l<sub>ow</sub> th<sub>e agen</sub>t<sub>s</sub> d<sub>own an</sub>d t<sub>eams w</sub>ill t<sub>urn</sub> it of.” This is the strongest objection, and it is why cost is the first scaling axis. Governance that costs a human minutes per change will be skipped once changes arrive faster than humans can spend those minutes. The answer is governance whose per-change cost is mechanical and whose human cost attaches to decisions. Whether OSTEON achieves that is Q2 of the agenda; we do not claim it has been shown.

“Thi<sub>s</sub> i<sub>s</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on recor</sub>d<sub>s an</sub>d d<sub>ocumen</sub>t<sub>a</sub>ti<sub>on as co</sub>d<sub>e,</sub> renamed.” The mechanisms have long roots, and we cite them in Section X. Three things are new. Decay is computed per Markdown block, for knowledge that lives outside the code (decisions, contracts, state records), from declared provenance to its sources against a sealed baseline, at every session start, rather than by comparing text with code after the fact or when a link breaks; inconsistency detection between code and its comments already does this for knowledge inside the code [25], and we extend the idea to knowledge the code does not carry. The human authority boundary is stated as an invariant that does not move with capability. And the whole is defined as a discipline for a regime in which the reviewer can no longer verify by reading, which the older practices did not have to address.

“Let agents review agents.” An agent reviewer raises �. If it reviews by reading, it inherits the reviewer’s problem at a higher rate: it judges the change against knowledge it cannot know is current. If it reviews against recorded decisions and contracts, it is a gate, and its detection is bounded by the currency of the record (Section III-C). Either way the knowledge, not the reviewer, is the term that governance has to act on.

## X. Related Work

Table VI summarises the positioning.

SE 3.0 and agentic software engineering. Hassan et al. propose SE 3.0, an AI-native software engineering characterised by intent-centric, conversation-oriented development with AI teammates, and a technology stack and roadmap for it [7]. In later work, Hassan and colleagues describe agentic software engineering through a duality of SE for Humans and SE for Agents. Humans orchestrate and mentor agent teams in an Agent Command Environment, and agents work and call on human expertise in an Agent Execution Environment; structured artifacts such as merge-readiness packs and consultation request packs carry the exchange [8]. That vision is the nearest to ours, and several OSTEON mechanisms read as instances of it: the start gate’s one-question-at-a-time decisions resemble consultation requests, and review against recorded knowledge resembles an evidence-backed merge pack. We difer in two respects. We define the regime by verification capacity, so the discipline is about what happens when the human can no longer review by reading; and we make the scaling of governance, rather than the partnership, the defining property, and accompany the argument with a released artifact.

Programming with trust. Roychoudhury et al. frame the shift from programming at scale to programming with trust, and propose that trust in agent-written code be built from analysis evidence that travels with the change [9]. We share the premise that review by reading will not carry the load. We difer in where the evidence attaches: their trust artifacts certify the change against the code; our gates certify it against the project’s recorded decisions and contracts, which the code does not carry.

The training route. Self-play SWE-RL trains an agent on real repositories with minimal data assumptions, specifies each bug by a test patch, and reports self-improvement on SWE-bench Verified and SWE-Bench Pro [2]. We see it as the capability half of the regime, complementary to ours: such training does not see decisions, contracts or the current state of a running system, that knowledge decays independently of the agent, and only governance acts on it.

Context files. Gloaguen et al. evaluate repository-level context files in a controlled study. They find that the files do not generally improve task success while raising cost, that instructions are followed, and that overviews do not help [16]. In a smaller study of 124 pull requests in 10 repositories, Lulla et al. associate AGENTS.md files with lower agent runtime and output token use [26]. Earlier empirical studies catalogue the content, adoption and evolution of context files in GitHub repositories [20], [21], [22]; our measurement difers in checking, against each repository’s tree, whether the paths a file names resolve. The two evaluations disagree on cost, one finding it higher and the other lower; we take no side, and our measurement bears on neither. OSTEON’s knowledge units are a diferent object from a context file: versioned, provenanceanchored, checked for decay and owned at one tier. Whether that diference changes the result is an open empirical question (Q4). Our measurement in Section IV is complementary to their evaluation: it describes how such files age in public repositories, not how they afect agents.

TABLE V  
Design heuristics from the author’s practice, each marked [Experience], and the OSTEON mechanism built to each. The right column says where the paper describes the mechanism; it is not evidence for the heuristic.
<table><tr><td>Heuristic</td><td>Statement</td><td>Built into, and where the paper describes it</td></tr><tr><td>1. A correction is a propagation problem [Experience]</td><td>A correction reaches the artifact that owns a fact more reliably than the artifacts that repeat it. Writing the lesson down does not stop the same failure from coming back; a mechanical check does.</td><td>R1, under which a fact has one owner and every other place references it; Eq. (2), where a mechanical gate sets ρ = 1 and supplies its own detection g in place of d.</td></tr><tr><td>2. A gate decays too [Experience]</td><td>A gate is an artifact and goes stale like any other. A stale gate is worse than no gate: it still runs, no longer tests what it claims, and turns a deliberate check into a false confirmation.</td><td>The review factor of Eq. (2), whose detection term is only as good as the gate that supplies it (the model has no term for a gate that has itself decayed); the capture skill, which records a check that passes on an input that should fail; the self-test</td></tr><tr><td>3. Mirrors of external state decay first [Experience]</td><td>Knowledge that mirrors external, mutable state goes stale fastest. Its staleness is driven by events, not by elapsed time, so a review on a calendar misses it.</td><td>shipped with the lints. Eq. (1), where s is defined by a change to the source, not by age; the decay check, which flags a unit when its source moves; the stale state record scenario.</td></tr><tr><td>4. Errors at a high tier are amplified [Experience]</td><td>An error in high-tier knowledge is not repeated once; every dependent inherits it. Repair has to happen at the source, because repairing the dependents one by one leaves the source free to seed the error again.</td><td>The tiers, with reference instead of copy; block-level provenance, under which one changed source decays every block that cites it; the escape model, where one stale unit makes every change that consults it stale-built.</td></tr><tr><td>5. Versioned references carry retractions [Experience]</td><td>When a dependency between artifacts is a first-class, versioned reference, an assertion and its retraction both travel along it quickly and visibly, even across repositories.</td><td>The drifting seam scenario, where the version bump marks the change as breaking and the consumer is flagged; provenance tokens, which may name a file in another declared repository.</td></tr><tr><td>6. A decision ledger can run at experiment speed [Experience]</td><td>Decision records need not be slow. A ledger can keep pace with experiments and record what they refute, provided an amendment is cheap and is recorded apart from the decision it</td><td>R2 and the amend-or-supersede lifecycle, under which the reason a project once had survives the decision that replaced</td></tr><tr><td>7. A registry measures only events [Experience]</td><td>changes. A registry can measure only what it represents as a discrete event. A lifecycle kept as a mutable field is overwritten as it moves, so its history cannot be measured afterwards.</td><td>The decay check, which records every run as an event, clean runs included, so that its series has a denominator; supersession, which is a new record rather than an edited field; the agenda&#x27;s condition on decay series (Section XI).</td></tr><tr><td>8. An approval that approves everything is ceremony [Experience]</td><td>A review step that never refuses is not a filter. For review to mean anything, its refutations must be recorded, so that what it stopped can be seen.</td><td>Curation, which applies nothing without per-item approval; capture, which records a change blocked before it lands; the decay record, which makes the fraction of warnings that led to curation answerable.</td></tr></table>

Context governance. ContextNest formalises context governance for knowledge that agents consume, with typed documents, deterministic selectors, hash-chained version histories, checkpoints and audit traces, so that an organisation can reconstruct which knowledge versions informed an agent output [27]. It is the closest in mechanism. Its object is the eligibility and integrity of knowledge beneath retrieval; ours is the engineering lifecycle of project knowledge: decisions, contracts, decay against moving sources, and planned change. The two compose; a bone could sit on an integrity layer of that kind.

Collective curation. Deliberative Curation proposes a protocol for multi-agent knowledge bases, with a formalised lifecycle, reputation-weighted deliberative voting and graduated sanctions, evaluated in simulation [28]. OSTEON keeps a human as the disposing authority for every change to project knowledge. As agent counts grow, per-item human approval may become the binding cost, and protocols of that kind may be needed to decide what reaches the human at all; Q6 lists this as open.

Decisions, contracts and decay. The record of design decisions as first-class architectural knowledge [10], [11], the knowledge-management work that followed [29], the lightweight decision record [12], design by contract [24], semantic versioning [23], provenance modelling [17], the laws of software evolution and ageing [4], [5], and the technical debt metaphor under which stale documentation is a debt that accrues interest [30] are the foundations OSTEON builds on. What is new is not any one mechanism but their combination into a discipline that agents operate and humans govern, with decay made computable at session start.

Stale documentation and traceabilit . That documentation drifts from code is established: comments and code co-evolve only loosely [13], and inconsistencies between them have been detected by rule [31], by identifier analysis [32], at scale [14] and at change time, before anything breaks [25]; outdated content is among the most reported documentation problems [15], and the maintenance of trace links between artifacts is a research field of its own [33]. The decay check difers in its object and its oracle: it watches knowledge that the code does not carry, and it compares declared provenance against a sealed baseline of its sources rather than text against code. The link check of Section IV is the routine kind.

TABLE VI Positioning against related work.
<table><tr><td>Work</td><td>Primary object, and the human&#x27;s role in it</td><td>Relation to this paper</td></tr><tr><td>SE 3.0 [7]</td><td>AI-native, intent-centric development stack; the developer works in partnership with AI teammates</td><td>Names the era; we name the discipline for the regime past delegation</td></tr><tr><td>Agentic SE [8]</td><td>SE for Humans and SE for Agents, with command and execution environments; humans orchestrate and mentor agent teams</td><td>Closest vision; we make governance scaling the defining property and supply a released discipline</td></tr><tr><td>Self-play SWE-RL [2]</td><td>Agent training by self-play bug injection and repair; humans supply sandboxed repositories</td><td>The capability route; we argue it does not act on knowledge decay</td></tr><tr><td>Programming with trust [9]</td><td>Trust in agent-written code via analysis evidence; humans set the trust bar</td><td>Same premise; we attach verification to recorded decisions and contracts rather than to the change</td></tr><tr><td>Evaluating AGENTS.md [16]</td><td>Effect of context files on agent task success; humans write the files</td><td>Read by us as evidence that more ungoverned context is not the lever</td></tr><tr><td>Context-file studies [20], [21], [22], [26]</td><td>Content, adoption and evolution of context files, and their effect on agent efficiency; humans write and maintain the files</td><td>We check whether the paths a file names resolve in its repository&#x27;s tree</td></tr><tr><td>ContextNest [27]</td><td>Governed knowledge vaults beneath retrieval; humans approve what agents may consume</td><td>Complementary integrity and provenance layer; we govern the engineering lifecycle</td></tr><tr><td>Deliberative Curation [28]</td><td>Multi-agent curation protocol for shared knowledge; humans design the protocol</td><td>Agents curate collectively; OSTEON keeps per-item human approval</td></tr></table>

Scalable oversight. Work on AI safety frames the problem of supervising systems that exceed their supervisors [18], [19], [34], [35]. Superintelligent software engineering is a domain instance of that problem, and our move is domain-specific: change the object of oversight from the change to the governed knowledge it is checked against.

## XI. Research Agenda

A vision earns its keep by saying what would prove it wrong. We state the thesis as testable claims, with the evidence that would refute each, and close with one design question (Q6).

Q1: Does governed knowledge reduce stale-built change? Fix a set of decisions and contracts for a task set, written by the experimenters. Expose them to the agent as governed knowledge in one arm and as ordinary documentation, or not at all, in the other, at matched agent capability, and measure the fraction of merged changes that contradict them; the experimenters’ set is the oracle, so it is defined in both arms. Vary throughput at fixed review capacity within each arm. The thesis is weakened if the fraction does not difer between arms, if capability alone drives it to near zero, or if it does not rise with throughput once reviewers are saturated.

Q2: Does governance cost scale sub-linearly in throughput? Measure total human minutes spent on decisions, contract versions, curation items and plan questions against the number of changes, in the same projects as Q1 so that the efect is held while the cost is read. The scaling principle fails if human governance efort grows in proportion to changes, or falls only because governance is skipped.

Q3: Howfast does project knowledge decay? Measure time to decay of anchored units against their sources, and the fraction of decay warnings that lead to curation. A decay rate near zero would make R5 unnecessary. Section IV gives a coarse, cross-sectional reference point, not a rate, for ungoverned agent context files; the rate for governed knowledge remains open.

Q4: Do context results transfer? Repeat the context-file evaluation of Gloaguen et al. [16] with tasks whose correct fix depends on a recorded decision not visible in the code, in two conditions: the decision is current, and the decision has been superseded since the file was written. In the first condition a context file and a governed unit present the same text; in the second only the governed unit is flagged as decayed. The claim that governed knowledge difers from context files fails if the two arms do not difer in the second condition.

Q5: Do gates hold against a more capable agent? Test whether review against recorded knowledge still detects contradictions when the agent argues persuasively for its change (the setting studied as debate [34]), and whether detection degrades as agent capability rises. The gate claim fails if detection falls as capability rises, because � is then a property of the agent after all and the escape rate regains its dependence on the agent.

Q6: When may agents curate? Compare per-item human approval with deliberative multi-agent curation [28] on error rate and human cost, to find where the authority boundary can be priced lower without moving. This is an open design question rather than a test of the thesis; it bounds how far the cost of the authority axis can fall without moving it.

Two conditions apply to all six: instruments should be released even when the data cannot be, because the data of real projects often cannot leave them; and decay series should include clean runs, so that rates have denominators, which is why OSTEON’s decay check records them.

## XII. Limitations

The paper reports no measurement of OSTEON’s efect, so nothing here shows that the discipline works; it shows what it does and why we expect it to matter. The measurement in Section IV is a single observational snapshot of popular public repositories. It counts unresolved path references in agent context files, a possible symptom of decay, not wrong instructions or their efect on agents, and its headline rate depends strongly on how references are read (Sections IV-D and IV-F). The model in Section III assumes independent staleness, a fixed review capacity and a fixed detection probability. Real projects violate all three. The model also holds � fixed as � rises. The agent’s own changes are among the sources that make other units stale, so in practice � rises with $r ,$ and the model understates the efect in that respect. The model is meant to show the direction of the efect, not its size. The definition rests on a threshold relative to a project’s own engineers that mixes capability with throughput, is not crisp, and will be crossed at diferent times for diferent tasks; a reader who prefers to name the regime by its operative condition, verification capacity, loses nothing in the argument. OSTEON 0.1.0 targets one agent platform, and its mechanisms are designed by one author; independent use may expose costs we have not seen. Finally, governance has a cost, and a team may reasonably refuse it while agents still work in the delegated regime. Our claim concerns the superseded regime, where we argue the alternative is worse. What OSTEON’s own history shows, and does not show, is in Section VI-H.

## XIII. Conclusion

Training is making software agents more capable, and it may make them more capable than the engineers who answer for the systems they change. We have argued that this capability is not enough, for three reasons. First, correctness is relative to decisions and contracts that the code does not carry. Second, this knowledge decays. Third, once agents outpace their reviewers, the changes built on stale knowledge escape review at a rate that grows with throughput. A snapshot of popular public repositories shows that the mismatch the second reason predicts is common, and does not show that it accumulates. Agent context files are common, and many of them stand still through hundreds of commits. Most of those that name repository paths name at least one that does not resolve. Files that stood still through more commits, however, do not have more unresolved references. Superintelligent software engineering is the discipline that answers this by scaling governance with capability. OSTEON is one concrete discipline for it, released with its scripts and data [3]. Whether it works is an empirical question, and we have stated the evidence that would settle it.

## Availability

OSTEON 0.1.0, including the fifteen skills, the knowledgerepository scafold and the decay check, is archived on Zenodo with DOI 10.5281/zenodo.23436020 [3]. Content is licensed under CC BY 4.0 and scripts under MIT.

The measurement in Section IV is provided as ancillary files with this preprint: the pre-specified protocol with its dated deviation note (README.md); the collection, summary, figure, post hoc audit and revision scripts, in Python 3 (collect.py, summarize.py, figures.py, audit.py, extend.py;

all but the figure script use the standard library only); the dataset (frame.csv, one row per repository returned by the search; repos.csv, one row per context file, with a symbolic-link flag; funnel.json; summary.json; candidates.csv; audit.json; and extended.json, the statistics added at revision); and the model behind Fig. 3 (escape\_model.py, fig-escape-model.json).

## Acknowledgment

The author used an AI assistant in drafting and editing this paper, reviewed the text, and takes full responsibility for its content.

## References

[1] C. E. Jimenez et al., “SWE-bench: Can language models resolve realworld GitHub issues?” in Proc. Int. Conf. Learning Representations (ICLR), 2024. arXiv:2310.06770.

[2] Y. Wei et al., “Toward training superintelligent software agents through self-play SWE-RL,” in Proc. 43rd Int. Conf. Machine Learning (ICML), PMLR vol. 306, 2026, pp. 134075-134095. arXiv:2512.18552.

[3] M. M. Gholinejad, “OSTEON: A discipline for superintelligent software engineering,” version 0.1.0, Zenodo, Oct. 2026. doi: 10.5281/zenodo.23436020.

[4] M. M. Lehman, “Programs, life cycles, and laws of software evolution,” Proc. IEEE, vol. 68, no. 9, pp. 1060-1076, 1980.

[5] D. L. Parnas, “Software aging,” in Proc. 16th Int. Conf. Software Engineering, 1994, pp. 279-287.

[6] N. Bostrom, Superintelligence: Paths, Dangers, Strategies. Oxford, U.K.: Oxford Univ. Press, 2014.

[7] A. E. Hassan, G. A. Oliva, D. Lin, B. Chen, and Z. M. Jiang, “Towards AInative software engineering (SE 3.0): A vision and a challenge roadmap,” arXiv:2410.06107, 2024.

[8] A. E. Hassan et al., “Agentic software engineering: Foundational pillars and a research roadmap,” arXiv:2509.06216, 2025.

[9] A. Roychoudhury, C. Păsăreanu, M. Pradel, and B. Ray, “Agentic AI software engineers: Programming with trust,” Commun. ACM, 2026. arXiv:2502.13767.

[10] A. Jansen and J. Bosch, “Software architecture as a set of architectural design decisions,” in Proc. 5th Working IEEE/IFIP Conf. Software Architecture (WICSA), 2005, pp. 109-120.

[11] J. Tyree and A. Akerman, “Architecture decisions: Demystifying architecture,” IEEE Software, vol. 22, no. 2, pp. 19-27, 2005.

[12] M. Nygard, “Documenting architecture decisions,” blog post, Nov. 2011. [Online]. Available: https: //cognitect.com/blog/2011/11/15/documenting-architecture-decisions (accessed Oct. 18, 2026).

[13] B. Fluri, M. Würsch, and H. C. Gall, “Do code and comments co-evolve? On the relation between source code and comment changes,” in Proc. 14th Working Conf. Reverse Engineering (WCRE), 2007, pp. 70-79.

[14] F. Wen, C. Nagy, G. Bavota, and M. Lanza, “A large-scale empirical study on code-comment inconsistencies,” in Proc. 27th Int. Conf. Program Comprehension (ICPC), 2019, pp. 53-64.

[15] E. Aghajani et al., “Software documentation issues unveiled,” in Proc. 41st Int. Conf. Software Engineering, 2019, pp. 1199-1210.

[16] T. Gloaguen, N. Mündler-Sasahara, M. N. Müller, V. Raychev, and M. Vechev, “Evaluating AGENTS.md: Are repository-level context files helpful for coding agents?” arXiv:2602.11988, 2026.

[17] L. Moreau and P. Groth, Provenance: An Introduction to PROV, ser. Synthesis Lectures on Data, Semantics, and Knowledge. San Rafael, CA, USA: Morgan & Claypool, 2013.

[18] D. Amodei, C. Olah, J. Steinhardt, P. Christiano, J. Schulman, and D. Mané, “Concrete problems in AI safety,” arXiv:1606.06565, 2016.

[19] S. R. Bowman et al., “Measuring progress on scalable oversight for large language models,” arXiv:2211.03540, 2022.

[20] W. Chatlatanagulchai et al., “On the use of agentic coding manifests: An empirical study of Claude Code,” in Proc. 26th Int. Conf. Product-Focused Software Process Improvement (PROFES), 2025, doi: 10.1007/978-3-032-12089-2\_40. arXiv:2509.14744.

[21] S. Mohsenimofidi, M. Galster, C. Treude, and S. Baltes, “Context engineering for AI agents in open-source software,” in Proc. 23rd IEEE/ACM Int. Conf. Mining Software Repositories (MSR), 2026. arXiv:2510.21413.

[22] W. Chatlatanagulchai et al., “Agent READMEs: An empirical study of context files for agentic coding,” arXiv:2511.12884, 2025.

[23] T. Preston-Werner, “Semantic versioning 2.0.0.” [Online]. Available: https://semver.org (accessed Oct. 18, 2026).

[24] B. Meyer, “Applying ‘design by contract’,” Computer, vol. 25, no. 10, pp. 40-51, 1992.

[25] S. Panthaplackel, J. J. Li, M. Gligoric, and R. J. Mooney, “Deep justin-time inconsistency detection between comments and source code,” in Proc. 35th AAAI Conf. Artificial Intelligence, 2021, pp. 427-435.

[26] J. L. Lulla, S. Mohsenimofidi, M. Galster, J. M. Zhang, S. Baltes, and C. Treude, “On the impact of AGENTS.md files on the eficiency of AI coding agents,” arXiv:2601.20404, 2026.

[27] M. Sulpovar, B. R. Konsynski, Q. Kanchwala, and G. Goodhart, “ContextNest: Verifiable context governance for autonomous AI agent,” arXiv:2607.02116, 2026.

[28] S. Johnson, “Deliberative curation: A protocol for multi-agent knowledge bases,” arXiv:2606.00007, 2026.

[29] R. Capilla, A. Jansen, A. Tang, P. Avgeriou, and M. A. Babar, “10 years of software architecture knowledge management: Practice and future,” J. Syst. Softw., vol. 116, pp. 191-205, 2016.

[30] P. Kruchten, R. L. Nord, and I. Ozkaya, “Technical debt: From metaphor to theory and practice,” IEEE Software, vol. 29, no. 6, pp. 18-21, 2012.

[31] L. Tan, D. Yuan, G. Krishna, and Y. Zhou, “/\*iComment: Bugs or bad comments?\*/,” in Proc. 21st ACM SIGOPS Symp. Operating Systems Principles (SOSP), 2007, pp. 145-158.

[32] I. K. Ratol and M. P. Robillard, “Detecting fragile comments,” in Proc. 32nd IEEE/ACM Int. Conf. Automated Software Engineering (ASE), 2017, pp. 112-122.

[33] J. Cleland-Huang, O. C. Z. Gotel, J. Hufman Hayes, P. Mäder, and A. Zisman, “Software traceability: Trends and future directions,” in Proc. Future of Software Engineering (FOSE), 2014, pp. 55-69.

[34] G. Irving, P. Christiano, and D. Amodei, “AI safety via debate,” arXiv:1805.00899, 2018.

[35] C. Burns et al., “Weak-to-strong generalization: Eliciting strong capabilities with weak supervision,” arXiv:2312.09390, 2023.