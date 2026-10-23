# What Lies Beyond Perception: A Partition Theory of the Limits of AI Evaluation and Oversight

Ahmed M. Mahran Incorta ahmed.mahran@incorta.com ahmahran@gmail.com

October 2026

## Abstract

Replacing the judge can move a benchmark score, and models can be trained to hide capabilities. Both put the evaluator in question, not only the model: what can it certify? We develop a partition theory of evaluation from one principle: judgement is bounded by perception. An agent’s discrimination is the partition of its inputs into classes it cannot tell apart, and its outputs depend on an input only through that class. Discrimination is necessary for capability but not suficient: a model cannot perform a task that requires a distinction it does not draw, and drawing the distinction does not ensure the task is done. Bounds on discrimination therefore bear directly on the question that matters most for safety: whether evaluation can show that a model lacks an ability. We prove a onesided bound. An examinee whose partition refines its evaluator’s has the discrimination to pass every test the evaluator can grade, provided each input has an acceptable reply. We define the echo test, which sends an input and accepts any reply the evaluator perceives as identical to it. The echo test is complete for the tests the evaluator can grade, much as a complete problem stands for its complexity class: no test the evaluator can grade certifies more about an examinee’s discrimination. An examinee that conflates two inputs the evaluator separates fails it once both inputs are probed. Passing therefore certifies at most a floor on discrimination, never a ceiling. Further elicitation of the examinee cannot raise this floor; only a sharper evaluator can. Used to detect concealment, echo-test failure points the wrong way: under exhaustive probing, it flags every honestly weaker examinee and no concealer that presents as the evaluator’s peer. Exact imitation of a weaker model is undetectable by any behavioural evaluator. We extend the bound to several observers. An added monitor sharpens the evaluation only if it separates inputs that no existing monitor separates. A model that controls one channel, such as its reasoning trace, can hide only what the channels it cannot influence fail to separate, provided the evaluator checks the channels against one another. We use the theory to explain published findings on judge disagreement and sandbagging, and we derive predictions that experiments can test. Behavioural evaluation can establish that a model draws a distinction. Without the model’s cooperation, it cannot establish that the model lacks one. A safety case that argues from behavioural evaluation alone that a model lacks an ability must therefore assume that cooperation.

## 1 Introduction

It is by now a commonplace of AI evaluation that tests bound capability from below only. Barnett and Thiergart [2024] state it directly: evaluations “can establish lower bounds on AI capabilities” while facing “an inability to establish upper bounds on capabilities.” van der Weij et al. [2025] report the practical form: models can be fine-tuned to “approximately emulate a consistent capability profile of a weaker model.”

The usual explanation is under-elicitation: a model may possess a capability that prompting, scafolding or fine-tuning failed to surface, so a negative result does not establish absence, and a claim of the form “this model cannot do X” is only as good as the efort spent asking. That explanation concerns the examinee. Evaluations also depend on the evaluator: a benchmark score can move when the judge is replaced and the candidate responses are held fixed [Yang et al., 2026], and a judge scores a candidate higher when that candidate’s mistakes resemble its own [Goel et al., 2025].

This paper studies the evaluator. Its thesis is that judgement is bounded by perception: what an evaluator can certify is bounded by its resolution, the distinctions it can draw, under its current elicitation (Lemma 4 and Corollary 11). Mahran [2012] took the principle that judgement depends on perception, with a perception–action model of agents, to the question of whether agents can judge that they perceive alike, and concluded that a judge comparing percepts can certify a diference but not a sameness. Here that line is developed formally, as a partition theory, for the evaluation of AI systems. Model an agent by the partition its perception induces on the space of inputs: two inputs are equivalent for that agent when it cannot tell them apart. Subject to one axiom, that an agent’s output depends on its input onl through its percept, two facts follow.

Suppose the examinee separates every pair of inputs its evaluator separates, and perhaps more. Then, if it knows how the evaluator perceives, it can work out which inputs the evaluator treats as the same, and answer as though it made no finer distinctions itself. Its extra resolution leaves no trace, and nothing in what the evaluator sees sets it apart from an agent that genuinely has no more.

Now suppose the opposite: some pair of inputs looks identical to the examinee and diferent to the evaluator. The examinee must answer both the same way, because it cannot see that they difer. The evaluator can, so it sees one answer where it expected two. That is a failure the examinee cannot avoid.

The two are not symmetric, and the reason is that discarding distinctions is easy and inventing them is not. An agent that sees finely can always group its percepts and act as though it saw less. An agent that sees coarsely cannot act as though it saw more: on any pair it has already conflated, its response is fixed. So discrimination above the evaluator’s resolution can be hidden and a shortfall below it is exposed, and exhaustive probing locates a floor while leaving everything above it open. We call the two emulation from above and detection from below (Theorems 7 and 9); together they give the one-sided bound.

The evaluator is not exempt. It is itself an agent, with its own limit on what it can tell apart, so what it establishes is bounded by that limit, and certifying the limit would take a sharper evaluator facing the same problem. Evaluations are run by agents, whether language models or humans, so this regress has no privileged end point.

The most an evaluation can certify is therefore set by the evaluator’s resolution. Eliciting the examinee harder does not raise it, since a sharper examinee can reproduce whatever its evaluator expects however hard it is asked; eliciting the evaluator can.

A sharper evaluator is the obvious remedy, and a real one. Three things about it are not obvious. The first is how sharp. The evaluator must separate the concealer’s reply from the reply of the agent it presents as, at some stimulus it probes (§4.4), a necessary condition, not a suficient one (Remark 8), and not guaranteed by being merely sharper than the last evaluator. Behavioural efort spent on the examinee does not expose an exact copy, and neither does efort on the evaluator (Corollary 17); against inexact concealment, efort on the evaluator helps only up to the sharpest evaluator available; for a system at the frontier a single sharp enough evaluator may not exist. An ensemble ofers a way round this. Its join separates every pair that some member separates (Definition 2), so a collection can be sharper than every one of its members, each supplying distinctions the others lack. The route to a sharper instrument can therefore run through diversity of discrimination as well as through a better individual judge. The second is that no warning is issued. By Corollary 11 passage says only that the examinee is somewhere above the evaluator, so a pass from an evaluator too coarse for the task looks exactly like a pass from an adequate one. The third is that the usual instruments do not act on this: harder items and richer scafolds press on the examinee, and change nothing about what the instrument doing the reading can tell apart.

Emulation from above and detection from below both concern an evaluator reading behaviour: emulation requires the examinee to choose what the evaluator sees, and detection requires a reply the evaluator can perceive. The part of Corollary 11 that needs no strategy on the examinee’s side, item 2, carries to an observer of any kind: an observation within a cell is consistent with everything in it. §4.3 uses this for interpretability probes.

## Contributions.

1. A framing (Sections 2 and 3): the evaluator, not only the examinee, has a discrimination structure, and what it can certify is bounded by it (Lemma 4). An evaluation compares two such structures, which gives a one-sided bound: an examinee sharper than its evaluator can present as a peer, while one that is coarser cannot conceal the fact from exhaustive probing. The bound holds for tests the evaluator can grade under the conditions of Theorems 7 and 9; the echo test is complete for such tests (Proposition 10); a coarse evaluator errs against honest weakness (Corollary 12); and a worked example instantiates it (Appendix A.2).

2. Six consequences (Section 4), arranged on a 2 × 3 grid of who assesses whom against adversariality. Exact sandbagging is undetectable by any behavioural evaluator. An interpretability probe cannot prove absence where it merges representations. An added monitor increases collective discrimination exactly when it separates a pair the ensemble does not. A channel the examinee controls can hide, from an evaluator that checks consistency over undisturbed channels, only what every other channel misses. No discrimination ceiling is self-assessed, and adversarial self-report adds nothing to testing. Corollary 13 limits these results to discrimination, which is necessary for task competence but not suficient.

3. Empirical implications (Section 5). Four published findings read through the model: judges disagree, weight noise exposes sandbagging, elicitation recovers locked capability, and monitor-aware models sometimes evade chain-of-thought monitors. Three predic tions follow. For concealment that the examinee’s elicitation leaves in place, eliciting the examinee should not raise detection, while sharpening the evaluator can. Under weight noise a concealing model returns to its unconcealed replies, with a rise that shrinks as the task needs more layers of processing. Encodings shaped against a monitor, over inputs that vary enough to isolate each output, approach renamings within the level sets of its score.

## 2 Model

Let Σ be a set of stimuli. An element $s \in \Sigma$ is a single thing an agent may be presented with: a prompt for a language model, a situation for an embodied agent. An agent A has a percept space $X _ { A }$ and a perception map $P _ { A } : \Sigma  X _ { A }$ , and emits actions from a set A via $E _ { A } : X _ { A }  { \mathcal { A } }$

Definition 1 (Discrimination partition). $P _ { A }$ induces the equivalence $s \equiv _ { A } s ^ { \prime } \iff P _ { A } ( s ) =$ $P _ { A } ( s ^ { \prime } )$ . Write $\Pi _ { A }$ for the resulting partition of Σ. We say $\Pi _ { B } \preceq \Pi _ { A } \ ( B$ is at least as sharp as A) when $\Pi _ { B }$ refines $\Pi _ { A } \colon$ every B-cell lies within an A-cell, i.e. $s \equiv _ { B } s ^ { \prime } \implies s \equiv _ { A } s ^ { \prime }$

Definition 2 (Join). The join $\Pi _ { 1 } \lor \Pi _ { 2 }$ of two partitions is the coarsest partition refining both: two stimuli share a cell of the join exactly when they share a cell of each. For a family, $\vee _ { i } \Pi _ { i }$ puts two stimuli together exactly when every Π<sub>i</sub> does.<sup>1</sup>

A partition can be read as a grid ruled over Σ: the agent sees which cell a stimulus falls into, and nothing finer. Refinement is a grid with more lines, and the join of two grids is their overlay. Figure 1 renders the results of Sections 3 and 4 this way.

The grid picture can obscure two properties of partitions. Not every partition is a grid; cells may take any shape. And refinement is a partial order rather than a total one, so two partitions need not be comparable: a cell of one may straddle a boundary of the other, each separating what the other runs together. Nested grids are therefore not guaranteed, and we expect them to be the exception rather than the rule. The theorems split on whether $\Pi _ { B }$ refines $\Pi _ { A }$ , and the incomparable case falls on the side where it does not.

Notation. Agents are $A , B , C , \ldots$ A letter names an agent rather than a place in a protocol: in the statements, A is the evaluator and B the examinee, whether the setting is the echo test of Definition 6, the interpretability probe of §4.3 or the language model judge of a scalableoversight pipeline. A proof that applies a result with another agent in A’s role says so. For an agent Z we write $P _ { Z }$ for its perception, $E _ { Z }$ for its emission, $\beta _ { Z } = E _ { Z } \circ P _ { Z }$ for its behaviour and $\Pi _ { Z }$ for its discrimination partition; a peer of A is an agent whose partition is $\Pi _ { A }$ . Lowercase indices $\Pi _ { i } , \Pi _ { j }$ denote observers or channels rather than named agents. One symbol stands outside this scheme: J, a percept-reading evaluator that observes percepts directly rather than behaviour, which appears only where we contrast the two kinds of access (Figure 5, Appendix A). Table 2 in Appendix E collects the symbols.

In a neural network the partition has a mechanism behind it. A network represents more features than it has dimensions by storing them in overlapping directions and tolerating the interference, a regime known as superposition [Elhage et al., 2022]. Capacity is finite and features compete for it, so a model separates what it has room to separate. Features it drops from a representation are run together there exactly, which is a partition, unless another route, such as copying, carries them; features stored in superposition are separated only up to interference, a graded case Section 7 returns to.

![](images/eb3191a9359362a5e7787cb135312ce987a13208e37dee5582464d5066196c68.jpg)  
(a) emulation from above

![](images/6000a399c42b3868862c11634b7a93b599154aa9f58ab967085462afd5635b64.jpg)  
shaded: one B-cell. Π splits it; B’s reply y lands above

![](images/a6f699ac896dfc8a6a753d8a4a9909dfa52488783dec3b49c689429757982daf.jpg)  
$E _ { A }$ feeds $P _ { B } ,$ , E<sub>B</sub> feeds $P _ { A }$

![](images/01b2fe1f04f4c4dad0b127252223365641270963e9c44bc9b2ca66d1d65521ba.jpg)  
the same exchange, in order

(b) detection from below  
![](images/068bb168cebcc9ef0693cf898289b215fc92f446c480327ea907a56441c5aefe.jpg)  
(c) the join W  
Figure 1: The three results as grids ruled over the stimulus space Σ. A is the evaluator and its lines are always thick; B is the examinee. Shading marks one cell of whichever agent is coarser: A in (a), B in (b). (a) Every A-cell, such as the shaded one, is exactly a union of B-cells, so B can always name the A-cell it is in and report as a peer (Theorem 7). (b) B’s grid lacks the horizontal line, so the shaded column is one B-cell while $\Pi _ { A }$ splits it, and $s _ { 1 } .$ $s _ { 2 }$ are indistinguishable to B but distinct to A. $B \mathrm { ^ { * } s }$ reply y (hollow) represents that whole cell and lies in $s _ { 1 } \ ' _ { \mathsf { s } }$ A-cell, so when sent $s _ { 2 } ,$ A receives something it reads as $s _ { 1 }$ (Theorem 9). The crossing is the composition $P _ { A } E _ { B } P _ { B } E _ { A }$ , and the U beside it lays out the same exchange in order. (c) Collective discrimination is the overlay of grids, and only lines not already present add cells (Proposition 21). Panels (a) and (b) draw comparable partitions; incomparable ones overlay as in (c).

Where the boundary falls. On this view a partition is fixed by construction, and what counts as a change of partition depends on where the boundary of the agent is drawn. For a language model there are two natural boundaries. Take the agent to be the bare model, and $\Pi _ { A }$ is a property of its weights. A prompt, a scafold or a returned tool result is then a further input rather than a modification of the thing receiving it, so it changes the stimulus and not the partition, while fine-tuning changes the weights and so yields a diferent agent. Take the agent to be the deployed system, meaning the model together with its prompt template, scafold, tools and decoding procedure, and $\Pi _ { A }$ is a property of that whole system; changing any of these yields a diferent system, which may difer in its partition, its emission or both.

Which boundary to use for each party is the study’s choice; the theory compares any two partitions of the same stimuli, wherever their boundaries are drawn. Assumption 26 (Section 5) is accordingly a claim about the configurations an elicitation budget produces, not a claim that perception itself improves.

We do not model the medium, meaning anything that acts on a stimulus between the two agents and is controlled by neither, whether deterministic or random. A reply is taken to arrive unchanged. This matters for the echo test, in which A reads $B ^ { \prime } { \mathrm { s } }$ s reply as a stimulus (Definition 6). A study that needs a medium can add it as a further map on stimuli between the two agents; this paper does not.

Axiom 3 (Perception mediates behaviour). An agent’s behaviour depends on the stimulus only through its percept: the total behaviour map $\beta _ { A } : \Sigma  A$ factors as $\beta _ { A } = E _ { A } \circ P _ { A }$

Axiom 3 makes perception the single map behind everything an agent does: elicitation can change the percept or how it becomes a reply, but every reply is computed from the percept. The axiom does not say how perception and emission are implemented. In a language model, for instance, the same weights may serve both, and decoding, which acts after the forward pass, belongs to emission: greedy decoding gives a map, and sampling brings in the seed discussed below. Every route by which the stimulus reaches the reply counts toward the partition, including verbatim copying and lookup in a fixed cache or retrieval index (a cache written between probes is state, which the axiom excludes). A system that can return an input verbatim separates that input from every other, so its partition is fine whatever it understands. Such a partition says nothing about competence, since discrimination is necessary for competence and not suficient (Corollary 13). The results take an agent’s emission to range over all maps from its percepts to actions: that B can realise a behaviour means some emission gives it, and whether B can compute that emission is a further capability the model does not represent. Section 7 returns to this.

Lemma 4 (Judgement is bounded by perception). Anything an agent computes from its percepts is constant on the cells of its partition: $i f f = g \circ P _ { A }$ for some map g, then $s \equiv _ { A } s ^ { \prime }$ implies $f ( s ) = f ( s ^ { \prime } )$ . In particular, by Axiom ${ \mathcal { B } } , \ \beta _ { A }$ is constant on the cells of $\Pi _ { A }$

Proof. If $s \equiv _ { A } s ^ { \prime }$ then $P _ { A } ( s ) = P _ { A } ( s ^ { \prime } )$ , so $g ( P _ { A } ( s ) ) = g ( P _ { A } ( s ^ { \prime } ) )$

A verdict, a self-report and a monitor’s flag are all computed from percepts, so each is bounded in this way.

A behaviour $\beta : \Sigma  A$ factors through a perception map $P : \Sigma  X$ when there is an emission map $E : X  A$ with $\beta = E \circ P ;$ equivalently, β is constant on the cells of the partition P induces, so it separates no pair that P conflates. Axiom 3 says every agent’s behaviour factors through its own perception map. Theorem 7 concerns behaviour that factors through another agent’s.

The axiom says two things. First, behaviour is a function of the stimulus: the same stimulus always gets the same reply, so no state is carried from one probe to the next. Second, that function passes through the percept: it factors through $P _ { A }$ . Determinism therefore follows from the first part and need not be assumed separately. An agent whose reply varies while its percept stays fixed must be reading something besides that percept. It may be state, which the axiom excludes. Or it may be a random seed, such as the one a sampler draws to pick each reply. A seed is an input, so we count it as part of the stimulus: two probes with diferent seeds are then diferent stimuli, and $\beta _ { A }$ is again a function, on a stimulus set enlarged to include seeds. For a language model with greedy decoding there is no seed, and the stimulus set needs no enlarging.

Definition 5 (Tests). Fix an examinee B.

1. A test $\boldsymbol { \mathcal { T } } = ( S , G )$ consists of a set $S \subseteq \Sigma$ of test cases and an acceptance rule G that assigns to each $s \in S$ a set $G ( s ) \subseteq A$ of acceptable replies.

2. B passes T at $s \in S$ when $\beta _ { B } ( s ) \in G ( s )$

3. An evaluator administers T by probing a set $Q \subseteq S$ of its cases; each presentation of a case is a probe, and a case may be probed more than once. The evaluator probes exhaustively when $Q = S$ . B passes T on Q when it passes at every $s \in Q$

4. T is gradable by an evaluator A when its acceptance rule factors through A’s perception: for $s , s ^ { \prime } \in S , s \equiv _ { A } s ^ { \prime }$ implies $G ( s ) = G ( s ^ { \prime } )$ . T is graded by A when A is the evaluator that administers it.

5. A task is a test whose acceptable replies are the correct responses.

A test fixes its cases and its rule, whoever supplies them. A claim about all of Σ needs $S = \Sigma ;$ a claim about a narrower scope takes that scope as S. A test gradable by A is one whose acceptable replies are fixed by what A perceives of each case. A benchmark scored by a judge against reference answers is graded by that judge, and gradable by it when the judge tells apart any two items whose reference answers difer. A task need not be gradable by A. When A grades a test by perceiving each case and each reply separately, its verdicts depend only on those percepts (Lemma 4), so it applies some rule gradable by A. If the test is not gradable by A, that rule difers from the test’s own at one of two cases A conflates, and there A accepts or rejects some reply wrongly.

The results below take the cases of a test to be all of Σ, the setting of a global claim. Each holds for a narrower scope S with the cases restricted to S and every partition restricted to S, that is, replaced by the partition of S into the non-empty intersections of its cells with S. Replies are still perceived through $P _ { A }$ on all of Σ, and a condition such as “every A-cell contains an action” then concerns the A-cells that meet S. A pass then certifies refinement on S only.

Definition 6 (Echo test). Assume the channel is closed: ${ \mathcal { A } } \subseteq \Sigma$ , so that a reply is itself a stimulus A can perceive. (For sequence models, whose prompts and replies share a stimulus space, this is immediate. Where a reply is instead scored outside the channel, by executing it, by inspecting side efects or by a human marker, the containment ${ \mathcal { A } } \subseteq \Sigma$ has to be assumed rather than inherited from the setup.) The echo test of A is the test with cases Σ, gradable by A by construction, with acceptance rule

$$
G ( s ) = \{ y \in A : P _ { A } ( y ) = P _ { A } ( s ) \} .
$$

A sends s, and B passes at s when A receives back something it cannot distinguish from what it sent.

A system that can copy its input passes the echo test trivially. Copying separates every input, so the copier’s partition is fine, and its pass certifies that discrimination and nothing about

competence (Corollary 13). A test meant to measure competence rather than discrimination can ask for a paraphrase instead.

The echo test asks only whether B can return what A said, as judged by A, so A’s partition is the only standard in play. It does not ask B to return s itself, only a reply in the same A-cell. In the echo test, every two stimuli A separates have disjoint sets of acceptable replies.

## 3 The asymmetry

Two facts, pointing opposite ways. An examinee at least as sharp as its evaluator can reproduce whatever that evaluator expects. One that is coarser cannot hide the fact from an evaluator that probes a pair it conflates. These are emulation from above and detection from below, named after the examinee’s place relative to its evaluator. Both follow from Lemma 4, and together they give the one-sided bound. They hold for tests the evaluator can grade, under conditions on the acceptable replies.

Theorem 7 (Emulation from above). Let $\Pi _ { B } \preceq \Pi _ { A }$

1. B can realise the behaviour of any agent whose behaviour depends on the stimulus only through $P _ { A }$ , and is then accepted by every test exactly where that agent is.

2. B can pass, at every stimulus, any test gradable by A in which every set of acceptable replies is non-empty. In particular, if every A-cell contains an action, B can pass the echo test at every stimulus and be behaviourally indistinguishable from a peer of A.

The proof builds the emulator from two maps. Since every B-cell lies inside one A-cell, B can tell which A-cell an input is in; the coarsening map h does this, turning B’s percept into A’s. If every A-cell contains an action, the prototype selector ρ picks one action in each A-cell. With $E _ { B } = \rho \circ h$ , B answers each input with an action from that input’s A-cell, which A cannot tell from the input itself. The proof is in Appendix A.

Theorem 7 says such an emission exists, not that it is easy to find or compute: in item 1, B’s emission must compute the target’s whole emission E together with the coarsening h. Whether B can compute that emission is a further capability, which the model does not represent. Remark 8 names one part of it.

Remark 8 (The emulator must model the evaluator). Theorem 7 makes emulation possible. To emulate a particular target, B must also know how that target behaves, as A perceives it. If B misjudges its target, it is not thereby exposed: it simply behaves diferently from the target. No behaviour is evidence of concealment on its own. Any behaviour $\beta _ { B }$ is exactly that of an agent that conceals nothing, namely the agent whose percept is $\beta _ { B }$ itself and whose emission is the identity. Behaviour reveals concealment only relative to an assumption about what honest agents look like, for example that an honest weak model makes certain mistakes. Without such an assumption, an emulator that misjudges its target looks like some other honest agent.

Theorem 9 (Detection from below). Let ${ \mathcal T } = ( S , G )$ be any test, and let $s _ { 1 } , s _ { 2 } \in S$ with $s _ { 1 } \equiv s _ { 2 }$ and $G ( s _ { 1 } ) \cap G ( s _ { 2 } ) = \emptyset$ . If both are probed, B fails at one of them. In particular, if $\Pi _ { B } \not \simeq \Pi _ { A }$ , then B fails the echo test on some probed stimulus, provided A probes both members of a pair B conflates and A separates.

Proof. By Lemma 4, $\beta _ { B } ( s _ { 1 } ) = \beta _ { B } ( s _ { 2 } )$ , and one reply lies in at most one of two disjoint sets. For the echo test, stimuli in diferent A-cells have disjoint sets of acceptable replies, and $\Pi _ { B } \not \simeq \Pi _ { A }$ gives a pair with $s _ { 1 } \equiv s _ { 2 }$ and $s _ { 1 } \not \equiv A \ s _ { 2 }$ □

Here “below” means below at some pair: B may draw distinctions A lacks elsewhere, and is still caught where it falls short. What is detected is that shortfall, a distinction A draws that B lacks, not concealment. The test need not be gradable by A for B to fail: that needs only a conflated pair whose sets of acceptable replies do not overlap. A records the failure wherever its verdict agrees with the test’s rule, as in the echo test, whose rule A applies exactly. Of the tests gradable by A, the echo test is complete.

Proposition 10 (The echo test is complete for gradable tests). Let every A-cell contain an action. Then B can pass the echo test of A at every stimulus if and only if $\Pi _ { B } \preceq \Pi _ { A }$ , and if $\Pi _ { B } \preceq \Pi _ { A }$ then B can pass every test gradable by A whose sets of acceptable replies are nonempty. A pass on such a test therefore certifies nothing about $\Pi _ { B }$ that a pass on the echo test does not.

The proof is in Appendix A.

In this sense the echo test is complete for the tests gradable by A, much as a complete problem stands for its complexity class. Here an examinee can pass a test when some emission of it passes. Whether the examinee can compute that emission, for instance the correct answer to each item, is a further capability. An examinee that copies its input passes the echo test without being competent at any task (Corollary 13).

Corollary 11 (Discrimination asymmetry). Let A probe exhaustively, and let every A-cell contain an action. Call B cooperative when it passes the echo test whenever its partition allows: whenever $\Pi _ { B } \preceq \Pi _ { A }$ , it knows $\Pi _ { A }$ and implements $\rho \circ h$ as in Theorem 7. Then:

1. Failure, for a cooperative B, certifies $\Pi _ { B } \not \simeq \Pi _ { A } .$ B conflates some pair A separates.

2. Passage certifies $\Pi _ { B } \preceq \Pi _ { A }$ , a lower bound on B’s sharpness, and is consistent with $\Pi _ { B } = \Pi _ { A }$ and with Π<sub>B</sub> strictly finer.

Proof. Item 1: if $\Pi _ { B } \preceq \Pi _ { A }$ , a cooperative B implements $\rho \circ h$ and passes at every stimulus by Theorem 7, so a failure rules that case out. Item 2: if $\Pi _ { B } \not \simeq \Pi _ { A }$ , exhaustive probing includes both members of a pair B conflates and A separates, and B fails there by Theorem 9, so a pass rules that case out. A peer of A and a strictly finer B can both pass, by Theorem 7, so passage is consistent with either. □

Item 1 needs cooperativeness because a sharper examinee may still fail. Theorem 7 says that a sharper B can pass, not that it must. An examinee that refines $\Pi _ { A }$ but declines to pass will fail, and §4.2 depends on this case. An examinee that tries to pass but judges $\Pi _ { A }$ coarser than it is may also fail. Failure also does not mean that B is coarser than A overall. The condition $\Pi _ { B } \not \simeq \Pi _ { A }$ holds equally when the two partitions are incomparable, with each drawing distinctions the other cannot.

A can tell which outcome it observed, since the echo test either passes or fails (Definition 6) and A reads the result directly. Against examinees satisfying Axiom 3, no behavioural protocol can separate the two readings of passage. The examinee may have $\Pi _ { B } = \Pi _ { A }$ , or it may be strictly finer and emulating a peer. If the emulator knows $\Pi _ { A }$ , then by Theorem 7 both cases produce identical behaviour, so passage carries no information that distinguishes them.

Scope. The two halves transfer diferently. Detection needs a test that gives a conflated pair disjoint sets of acceptable replies (Theorem 9), administered by an evaluator whose verdicts there agree with the test’s rule. Indistinguishability rests on exact emulation. The second half of Corollary 11, item 2, holds because a strictly finer examinee can emulate a peer exactly (Theorem 7), and identical behaviour gives identical observations under any protocol, adaptive, randomised or multi-round included (Corollary 17, item 1). Inexact emulation under such protocols, and stateful agents, lie outside these results. Within this scope, behavioural evaluation yields lower bounds on discrimination, and no upper bound beyond the conflated pairs that a cooperative examinee’s failures expose.

This is the lower-bound conclusion of Section 1, recovered for discrimination by Corollary 11. Figure 2 gives it as geometry.

![](images/832add97197beeecdfae379e58fb54a2bfb0c4afbab43ab5473ac00edd5400cf.jpg)  
Figure 2: Corollary 11 in the partition lattice, ordered by refinement with sharper agents higher. The diagram is schematic: it is a genuine lattice, with covering edges only, but not the partition lattice of any stimulus set, which grows too fast to draw. Each node stands for a possible discrimination structure. Passing A’s echo test tells us only that B lies somewhere in the shaded up-set above $\Pi _ { A } \mathbf { \hat { \mathbf { \Omega } } }$ bounded below by the evaluator, and otherwise unconstrained. Every partition from $\Pi _ { A }$ up to ⊤ stays consistent with the evidence. This is what “lower bound but never upper bound” means geometrically: the evidence fixes a floor and then says nothing, so the only upper bound available is the trivial one, ⊤ itself. Eliciting the examinee harder may move B within this picture but does not move $\Pi _ { A } ;$ only a change to the evaluator, raising $\Pi _ { A }$ , shrinks the cone.

Corollary 12 (Specificity: a coarse evaluator errs against honest weakness). Let A probe exhaustively, and let every A-cell contain an action. Let $B _ { \mathrm { c } }$ be a concealing examinee with $\Pi _ { B _ { \mathrm { c } } } \preceq \Pi _ { A }$ , implementing the peer echo $\rho \circ h$ of Theorem 7, and let $B _ { \mathrm { h } }$ be an honestly weaker examinee with $\Pi _ { B _ { \mathrm { h } } }  \Pi _ { A }$ , coarser than A or incomparable with it. Then A observes no echo failure from $B _ { \mathrm { c } }$ and at least one from $B _ { \mathrm { h } }$

Proof. Theorem 7 for $B _ { \mathrm { c } } \mathrm { ; }$ ; Theorem 9 for $B _ { \mathrm { h } }$ , whose conflated pair is probed because A probes exhaustively. □

A detector that flags concealment whenever the examinee fails the echo test therefore fires on $B _ { \mathrm { h } }$ and never on $B _ { \mathrm { c } }$ . Against these two arms it does worse than chance: its flags run opposite to the property it is meant to detect. This is a specificity control, which tests of a concealment detector need. Concealers less exact than $B _ { \mathrm { c } } ,$ which misjudge $\Pi _ { A }$ (Remark 8), will sometimes fail, so a detector scored only on concealing examinees can show a respectable true-positive rate. The same detector flags every honestly weak examinee, because under exhaustive probing each one fails. It is detecting apparent weakness, and it catches a concealer only when that concealer happens to look weak. An honestly weak arm, matched to the level the concealers present, exposes this: it measures how often honest examinees are flagged as concealing. The direction of the error follows from Theorems 7 and 9; its size depends on how densely A probes, since under sparse probing the pair an honestly weak examinee conflates may go unprobed.

## 3.1 Discrimination and task competence

Theorem 9 does not need the test to be gradable by A, and taking $G ( s )$ to be the set of correct responses links discrimination to competence.

Corollary 13 (Discrimination is necessary for competence). Let $G ( s )$ be the set of correct responses to s. $I f s _ { 1 } \equiv _ { B } s _ { 2 }$ and no response is correct for both, then B answers at least one of them incorrectly. Hence competence on the task requires $\Pi _ { B }$ to separate every such pair.

Proof. Theorem 9 applied to the task.

The converse fails. Separating $s _ { 1 }$ from $s _ { 2 }$ does not entail answering either correctly: discrimination is a substrate for competence, not a form of it. Consequently our results transfer to capability evaluation in exactly one direction. Coarseness implies incompetence, so a model too coarse for a task cannot hide that fact from a test that includes both items (Corollary 13). The reverse inference fails, since a model with suficient discrimination may decline to use it (Theorem 7). Whether fine discrimination predicts task performance is a separate question. The claims below are bounds on discrimination and on what discrimination makes possible, not a reduction of capability to partition structure.

## 4 Six consequences for evaluation

Each result below follows from the model of Section 2 (Lemma 4, Theorems 7 and 9, and the join), together with the definitions its subsection adds. Two choices place them on a grid (Table 1): who assesses whom (an agent itself, one agent another, or several observers together), and whether the observed party selects its behaviour to defeat the assessment (adversariality).

## 4.1 No self-assessed ceiling

An agent cannot discover, from its own percepts of stimuli presented separately, what it is blind to.

Corollary 14 (No self-assessed ceiling). Every self-report of A factors through $P _ { A }$ . For any $s _ { 1 } \equiv _ { A }$ s it is therefore the same whether A is shown s or s , and it cannot indicate that the two $d i f f e r$

The proof is in Appendix B.

Nothing in an agent’s percepts of two stimuli it conflates, presented separately, distinguishes them from one stimulus seen twice (a stimulus presenting both is a third stimulus, which $\Pi _ { A }$ does not constrain), so honest self-assessment cannot reveal a blind spot from the agent’s own percepts of the conflated stimuli. The agent can learn of a blind spot only from an agent that separates a pair it conflates. It can rule out a supervisor coarser than itself (Corollary 11), and can confirm one that is sharper only if that supervisor chooses to show it. A model asked what it cannot do can report what it has failed at. It has no access to what it has been missing.

## 4.2 Strategic self-reports add nothing to testing

Let the agent’s self-report be strategically chosen. Then it bounds discrimination in neither direction.

Corollary 15 (Strategic self-reports add nothing to testing). Let B, asked by a question that describes them, make a claim about a pair $s _ { 1 } , s _ { 2 } ,$ ; the question is a stimulus distinct from both.

1. ${ \cal I } f s _ { 1 } \equiv _ { \cal B } s _ { 2 }$ and B claims to separate the pair, then on the echo test of any A that separates the pair and probes both members, B cannot pass at both, whatever replies it gives there. The over-claim is refutable.

2. If $s _ { 1 } \not \equiv s _ { 2 }$ and B claims to conflate the pair, then B can behave exactly as an agent C whose partition is $\Pi _ { B }$ with the cells of $s _ { 1 }$ and s<sub>2</sub> merged. No behavioural test refutes the under-claim.

The proof is in Appendix B.

An adversarial self-report is therefore refutable when it over-claims and unconstrained when it under-claims. By §4.1 the honest report was already silent about the agent’s blind spots, and nothing compels the agent to give even that. A model that says it cannot do something has told an evaluator nothing the evaluator can use without testing.

## 4.3 Absence is unprovable where a probe merges

Where an interpretability probe merges representations, finding nothing does not show there is nothing to find.

An interpretability probe p reads a model’s representation $\phi ,$ so it separates $s _ { 1 }$ from $s _ { 2 }$ exactly when $p ( \phi ( s _ { 1 } ) ) \neq p ( \phi ( s _ { 2 } ) )$ .
<table><tr><td>self</td><td>an agent about it- one agent about an- several</td><td>other</td><td>together</td><td>observers</td></tr><tr><td></td><td>non-adversarial no self-assessed ceiling absence unprovable §4.1</td><td>§4.3</td><td colspan="2">join criterion §4.5</td></tr><tr><td>adversarial</td><td>self-report adds noth- concealment</td><td></td><td colspan="2">redundancy bound</td></tr><tr><td></td><td>ing</td><td>§4.4</td><td colspan="2">§4.6</td></tr><tr><td>§4.2</td><td></td><td></td><td colspan="2"></td></tr></table>

Table 1: The six consequences, organised by who assesses whom against adversariality. Reading down a column, the same arrangement of assessor and assessed is first passive and then exploited by the observed party.

Corollary 16 (Absence is unprovable where a probe merges). If $p ( \phi ( s _ { 1 } ) ) \neq p ( \phi ( s _ { 2 } ) )$ , then $\phi ( s _ { 1 } ) ~ \neq ~ \phi ( s _ { 2 } )$ . The converse fails wherever p identifies two representations the model can produce: $i f p ( \phi ( s _ { 1 } ) ) = p ( \phi ( s _ { 2 } ) )$ , the model may or may not separate the pair.

The proof is in Appendix B.

Interpretability therefore certifies the presence of a distinction, and certifies its absence only where the probe loses nothing of the representation it reads. This is the lower bound of Corollary 11, with the probe in A’s role.

## 4.4 Exact concealment leaves no behavioural trace

Add intent to §4.3 and the examinee chooses what the evaluator sees.

## Hiding a capability.

Corollary 17 (Exact sandbagging is undetectable). Let $\Pi _ { B } \preceq \Pi _ { C }$ , so that B is at least as sharp as an agent C it may present as.

1. B can realise $\beta _ { C }$ exactly, and then no evaluator reading behaviour, at any resolution, distinguishes B from C.

2. If B does not realise β<sub>C</sub> exactly, an evaluator A distinguishes the two only at a stimulus s with $P _ { A } ( \beta _ { B } ( s ) ) \neq P _ { A } ( \beta _ { C } ( s ) )$

The proof is in Appendix B.

By Theorem 7, a model that draws every distinction its evaluator draws can present as any agent whose behaviour factors through $P _ { A }$ , and that evaluator cannot tell them apart (Corollary 17, item 1). It is enough to match the evaluator on the distinctions the evaluator makes; the model may be arbitrarily coarse elsewhere. Concealment does not need global superiority, only local suficiency wherever the evaluator can look.

Item 2 is the necessary condition for detecting the concealment, and it is weaker than global refinement: A need only separate $B \mathrm { ^ { * } s }$ reply from $C \mathrm { { ' s } }$ where the two difer, not every pair B separates. Item 1 shows that no resolution sufices against exact emulation. The tie with C is broken, for a fixed examinee, only by imperfect emulation or by a channel outside the model’s control (§4.6), not by sharpness alone, and imperfect emulation shows that B difers from C, not that B is concealing (Remark 8).

Test design is therefore not the lever against exact emulation. Harder items and richer scafolds act on the examinee: they change what it shows, and its measured capability [Barnett and Thiergart, 2024], but not what the evaluator can separate.

Hiding a message. The freedom that lets B hide a capability also lets it hide a message. Within each of A’s cells, B can choose which output to send, and A cannot tell the choices apart. A covert channel uses that choice: the sender hides a message by systematically choosing some outputs in place of others, and the monitor must decide from the outputs it sees whether a message is present. The monitor sees only which of its cells each output falls in, so it can detect the encoding only through a change in the cells it observes. Two results make this exact.

The sender is B and the monitor is A, reading B’s outputs, which are elements of Σ. An output is whatever the monitor receives as one input, such as a whole reply. Write $q : \Sigma $ $\Sigma / \equiv _ { A }$ for the map sending an output to its A-cell; by Lemma 4, anything the monitor computes from an output depends only on this cell. Let $\mu$ be the distribution of the outputs the sender would produce with no hidden message, and let the encoding $T : \Sigma  \Sigma$ replace each such output s by the output $T ( s )$ it sends instead. Take a single output $s \sim \mu :$ without the encoding the monitor sees $q ( s )$ , and with it $q ( T ( s ) )$ . For a map f we write $f _ { \ast } \mu$ for the pushforward of $\mu$ along $f ,$ the distribution of $f ( s )$ when $s \sim \mu ,$ so that $f _ { * } \mu ( V ) = \operatorname* { P r } _ { s \sim \mu } [ f ( s ) \in V ]$ . The cell the monitor sees for one output is then distributed as $q _ { * } \mu$ without the encoding and as $( q \circ T )$ <sub>∗</sub>µ with it. These generally difer, and the question is when they do not.

We take T to be a fixed map; Appendix B extends both results to random maps.

A detector looks at the cell of a single output and decides whether to flag it. It is a measurable map $d : \Sigma / \equiv _ { A } \to \{ 0 , 1 \}$ , and it fires on the set of cells $V _ { d } = \{ K : d ( K ) = 1 \}$ . Its true positive rate is the probability that it fires on an encoded output, and its false positive rate the probability that it fires on an unencoded one:

$$
\mathrm { T P R } ( d ) \ = \ ( q \circ T ) _ { * } \mu ( V _ { d } ) , \qquad \mathrm { F P R } ( d ) \ = \ q _ { * } \mu ( V _ { d } ) .
$$

A detector carries information about the encoding exactly when the two rates difer; if it fires more often on unencoded outputs, its opposite fires more often on encoded ones.

Definition 18 (Blindness). The monitor is blind to $T$ under $\mu$ if no detector carries information about it, that is, if $\mathrm { T P R } ( d ) = \mathrm { F P R } ( d )$ for every measurable detector d.

Proposition 19 (Blindness under a given $\mu )$ . The monitor is blind to T under $\mu$ if and only $i f ( q \circ T ) _ { * } \mu = q _ { * } \mu$

That is, the monitor cannot detect the encoding if and only if the encoding leaves unchanged how often each of its cells appears. Call $T$ a renaming when it never moves an output out of its cell, that is, when $q \circ T = q$

Proposition 20 (Blindness under every $\mu )$ . The monitor is blind to T under every distribution µ on Σ if and only $i f T$ is a renaming.

Both proofs are in Appendix B, with an encoding that is blind under one distribution but not under every one.

## 4.5 When an added monitor helps

A second monitor adds discrimination only if it sees something the first one misses.

Observers $O _ { 1 } , \ldots , O _ { n }$ jointly separate s and $s ^ { \prime }$ if some $O _ { i }$ does, so their collective discrimination is the join $\vee _ { i } \Pi _ { i }$ (Definition 2).

Proposition 21 (Join criterion). Adding $O _ { n + 1 }$ strictly increases collective discrimination if $\vee _ { i \leq n } \Pi _ { i } \ \not \leq \ \Pi _ { n + 1 }$ , that is, $i f f \ C _ { n + 1 }$ separates some pair the existing ensemble does not. A monitor whose partition is refined by the existing join adds nothing.

The proof is in Appendix B.

Whether an added monitor adds discrimination is therefore decided by the members’ partitions, and predictable from them where these are known. Two monitors that draw the same distinctions are one monitor. The criterion asks for diversity of discrimination, not a count.

## 4.6 Redundancy limits what a controlled channel can hide

An evaluator rarely watches a model through a single source. It may read the final answer, the reasoning trace, an interpretability probe and a system log. A model that controls one of these sources can conceal through it, by sandbagging or by encoding a message there, but from an evaluator that checks consistency, with the other sources undisturbed, it can hide only distinctions that every other source misses.

Let $s \in \Sigma$ be the situation A is trying to observe, meaning what B is actually doing. A does not see s itself. It sees one reading from each source, and each source is a channel.

Definition 22 (Channels, control and consistency). A channel $c _ { i }$ gives a reading $o _ { i } ( s ) \in Y _ { i }$ of the situation. Its partition $\Pi _ { i }$ , with equivalence $\equiv _ { i }$ , puts together the situations that give the same reading, which $c _ { i }$ cannot tell apart. A reads every channel, so two situations look alike to A exactly when they look alike on all of them: $\Pi _ { A } = \vee _ { i } \Pi _ { i }$ (Definition 2). B controls $c _ { j }$ when it can make that channel show any reading $y \in Y _ { j }$ it chooses, while every other channel shows its true reading $o _ { i } ( s )$ . A checks consistency when it accepts a set of readings $( y _ { 1 } , \ldots , y _ { k } )$ if and only if some situation $s ^ { \prime }$ would give all of them, that is, $o _ { i } ( s ^ { \prime } ) = y _ { i }$ for every i. It then concludes that the situation looks like $s ^ { \prime }$ on every channel. Any two such $s ^ { \prime }$ give the same readings, so the conclusion does not depend on which one A picks.

Proposition 23 (Redundancy bound). Let B control $c _ { j }$ , let A check consistency, and write $\begin{array} { r } { U = \bigvee _ { i \neq j } \Pi _ { i } } \end{array}$ for the join of the channels B does not control, with equivalence $\equiv _ { U }$ . Then $B _ { i }$ , in situation $s ,$ can lead A to conclude a situation s<sup>′</sup> with $s ^ { \prime } \not \equiv _ { A } s$ if and only if

$$
s \equiv _ { U } s ^ { \prime } \qquad a n d \qquad s \not \equiv _ { j } s ^ { \prime } .
$$

In words: B can make A mistake s for $s ^ { \prime }$ exactly when every channel B does not control reads the two alike and the channel it controls reads them diferently. Any other false reading on $c _ { j }$ fails the consistency check.

The distinctions B can hide through $c _ { j }$ are therefore exactly the pairs $\Pi _ { j }$ separates and U does not: the lines in Figure 6 (Appendix B) that only the controlled channel draws. Two conditions in the setup shape this set. The first is that A checks consistency. Without it, a reading on $c _ { j }$ that contradicts an uncontrolled channel goes unnoticed, B can misrepresent pairs that $U$ separates, and the bound fails. The second is that the uncontrolled channels keep showing their true readings whatever B does on $c _ { j }$ . The proposition is exact when A checks consistency and the uncontrolled channels are undisturbed. A disturbance can work either way: it shrinks the set when it makes some of the swaps the proposition allows visible, and enlarges it when it moves an uncontrolled reading toward a false one, whether or not B steers it. Under both conditions, a model that sandbags in its answers, or encodes a message in its trace, is exposed exactly when no situation fits both what the uncontrolled channels show and what it makes the controlled channel show (Proposition 23).

The second condition matters for reasoning traces. A model writes both its trace and its answer, but the two come out of one computation, so a model that alters its trace to conceal may also change the answer it reaches. Where the answer is scored against the task, altering the trace can change whether the answer it reaches is correct without the model intending it, and a mismatch between trace and answer can expose the alteration.

Against concealment through a channel the system fully controls, oversight is therefore bought with a channel it cannot influence, not with deeper analysis of the controlled one. Asking a model to explain its reasoning in the same text stream it already controls adds no independent channel. A log the model cannot write does, and so does an interpretability probe, to the extent the model cannot shape the activations it reads.

## 5 Empirical implications

## 5.1 Noise and elicitation

Any change inside an agent’s boundary, such as a prompt, a configuration, fine-tuning or a perturbation of its weights, yields a diferent agent, which may difer in its partition, its emission or both (Section 2). The model neither lists these changes nor says how any of them acts. This section studies two: noise added to the weights along one fixed direction at growing scale (Assumption 24), and an elicitation budget that adds distinctions and removes none (Assumption 26).

Noise. In the experiments read in §5.2, concealment is set of by a trigger: a system instruction, a note found in the environment, or the absence of a password. The trigger can also be the content of the item itself, as when a model underperforms only on dangerous-capability questions [van der Weij et al., 2025]. We model all of these the same way. Concealing a capability then has two parts. First, B must perceive the trigger, which is a distinction in $\Pi _ { B } \colon$ by Lemma 4 it replies alike to stimuli it conflates, so if it cannot tell an item that comes with an instruction to underperform from the same item without it, it cannot conceal on one and not the other. Second, B must link the trigger to the concealing reply, through its emission; for concealment as an agent C, the link is the map $E _ { C } \circ h$ of Theorem 7, with C in the evaluator’s role. $B \mathrm { ^ { * } s }$ unconcealed version is the agent B would be if it lacked the trigger’s distinction or the link, and an item’s unconcealed reply is the reply that version gives it. The link is the part of $E _ { B }$ that maps percepts carrying the trigger to the concealing reply. A reply depends on four parts: the distinctions that separate its item from items with other correct answers; the part of $E _ { B }$ that turns the item’s percept into the reply; the link, if it is applied to that percept; and, if the link changes the reply, the trigger’s distinction. We take the trigger or its link to be carried by a small share of the input or of the weights, and the task by most of them.

Assumption 24 (Weakest part breaks first). Fix a direction of weight noise and scale it by σ. Each distinction in $\Pi _ { B }$ and each part of $E _ { B }$ , links included, has a breaking point along it: a scale below which it is present and from which on it is removed (for a distinction, the scale from which its two percepts coincide). A part breaks later the more strongly B’s input and weights carry it, and earlier the more layers of processing it needs. Noise below the breaking points of all the parts a reply depends on leaves that reply unchanged. Once the trigger or its link is removed, B replies as its unconcealed version does at the same noise; its input and weights carry that behaviour more strongly than they carry the concealment.

Proposition 25 (Noise first exposes, then degrades). Under Assumption 24, suppose the trigger and its link break at points common to all items, and let $\sigma _ { e }$ be the lower of the two. For $\sigma \geq \sigma _ { e }$ B gives every item the reply its unconcealed version gives at the same noise. For $\sigma < \sigma _ { e } ,$ B still conceals, and its reply to an item is unchanged as long as σ is also below the breaking points of the other parts that reply depends on. Wherever noise has merged two items with no correct answer in common, B answers one of them incorrectly. An examinee that conceals nothing has no trigger or link to lose.

The proof is in Appendix C.

A concealing examinee’s score therefore stays at the presented level while its parts hold, and from $\sigma _ { e }$ on equals the score of its unconcealed version at the same noise. It rises at $\sigma _ { e }$ exactly when that version scores higher there than the concealing examinee did just below $\sigma _ { e } ,$ and then follows that version’s score as items pass their own breaking points, gradually when those points are spread across items. An examinee working at its own level has nothing to lose at $\sigma _ { e }$ (Figure 3).

![](images/c69489b08f3f04a2130e2ddea17c018496407a37a5e897c120e11f2021830cba.jpg)  
Figure 3: Observed score under weight noise, given Assumption 24 (schematic). Both examinees start at the same observed score: the concealing one (solid) because it presents at that level, the honestly weak one (dashed) because that is its level. The dotted line is the score of the concealing examinee’s unconcealed version. Below $\sigma _ { e }$ the concealing examinee perceives the trigger and conceals. At $\sigma _ { e }$ its trigger or link breaks; from there it gives the unconcealed version’s replies, so its curve jumps to the dotted level and then follows that version’s score as items pass their own breaking points (Proposition 25). The honestly weak examinee conceals nothing, so it has nothing to lose at $\sigma _ { e } ;$ its score changes only as its own items break.

## Elicitation budget.

Assumption 26 (Monotone elicitation). Raising an agent’s elicitation budget yields a configured system whose partition refines that of the lower budget: it adds distinctions and removes none.

A diferent prompt can suppress a distinction as well as introduce one, and nothing in the model forbids it.

## 5.2 Existing observations

Judges disagree. Replacing the judge moves a benchmark score with the responses held fixed [Yang et al., 2026]. A verdict is computed from the judge’s percepts of a reply, so by Lemma 4 it is constant on the judge’s cells, and judges with diferent partitions can grade the same replies diferently. A judge also scores a candidate higher when that candidate’s mistakes resemble its own [Goel et al., 2025]: an error that stays within one of the judge’s cells is one the judge cannot see. Repeated samples of one judge share its partition and add nothing to the join (Proposition 21), and judges with similar partitions add little.

Noise exposes sandbagging. Tice et al. [2025] add noise to the weights of language models. Models evaluated at their true capability degrade predictably as the noise grows, while models that underperform on purpose show anomalous improvements. The authors attribute the improvement, as likely, to noise disrupting the mechanism of underperformance while core capability stays partly intact, and caution that a harder benchmark may involve increasingly fragile behaviours and so show a weaker pattern.

Under Assumption 24, treating each noise draw as a direction, Proposition 25 gives this pattern: the concealing model’s score jumps at $\sigma _ { e }$ to that of its unconcealed version, while a model at its own level has nothing to lose there (Figure 3). Tice et al. draw a fresh noise vector at each scale, so their curves are not paths along one direction. The authors’ caution is the case in which the unconcealed version has already lost most of its score by $\sigma _ { e } \colon$ a task that needs deeper processing breaks at lower noise, and the rise shrinks (P2).

The three settings of Tice et al. [2025] difer in where the trigger and its link sit, and in how much evidence they give.

Instruction. A system instruction tells the model to underperform, so the trigger is in the input and nothing is installed in the weights. Noise is added to all weights, and the rise is significant in all 17 model–benchmark pairs.

Password lock. In two models fine-tuned to underperform without a password, the trigger is the missing password and the link is trained into the weights. Noise, added here through a low-rank adapter on linear layers, raises each model’s score without the password, in a similar way and over a similar magnitude of noise as with an instruction. With the password the model already answers at full capability, and noise does not appreciably raise its score: there is no concealment to remove.

No goal. The model is told to perform at its best and finds, in a developer’s note, an incentive to underperform; noise again enters through the adapter. The evidence is thinner: one model, Mistral Large, whose best noised accuracy is 1.515 times its noise-free accuracy, and two noised runs that recover from 66.6% to 100% on 30 GSM8k problems. Removing the phrase “(i.e. that your answers are being monitored)” from the system prompt returned the model to full performance, and high-scoring noised models answer directly without acknowledging the incentive. The first fits a trigger carried by a small share of the input; the second fits the noised model giving its unconcealed reply. The setting adapts the “no goal given” setting of Meinke et al. [2024], in which Claude 3.5 Sonnet and Claude 3 Opus appear to have sandbagged for a goal acquired in training; Tice et al. use a diferent task, threshold and prompt, and do not establish what motivates Mistral Large.

Noise can expose what Corollary 17 says no behavioural evaluation of the unchanged model can, because it changes the model: the noised model is a diferent agent, one whose trigger may no longer be perceived.

Elicitation recovers locked capability. Greenblatt et al. [2024a] train password-locked models to imitate a much weaker model unless a password is present, and find that fine-tuning on a few high-quality demonstrations often recovers the locked capability. Hofstätter et al. [2025] find that prompting elicits the actual capability of password-locked and circuit-broken models on multiple-choice questions, while on a code-generation task only fine-tuning elicits it from their circuit-broken model. The two routes difer in the model. Taking the prompt as part of the input (Section 2), a prompt that recovers the capability is a stimulus at which the locked model does not behave like the weaker model it imitates: its emulation is imperfect there, and Corollary 17, item 2, applies. Where only fine-tuning succeeds, the lock holds against the prompts tried. Fine-tuning instead changes the model, so, like noise, it can expose what no behavioural evaluation of the unchanged model can. That a few demonstrations sufice fits a lock carried by a small part of the weights, the premise stated with Assumption 24. P1 below concerns locks that extra elicitation of the examinee does not break.

Monitor-aware models sometimes evade chain-of-thought monitors. Li et al. [2025] prompt models to sandbag while either unaware or aware of a chain-of-thought monitor. Among successful sandbagging attempts, the monitor approves 16–36% when the models are aware of it and 2–6% when they are not. For most models, sandbagging success also drops when the models try to be covert. The monitor reads the trace and the answer, both written by the model, and flags answers inconsistent with the reasoning: a consistency check across two channels the model writes (§4.6). Aware models are told that a monitor will read their entire output and are asked to hide their sandbagging from it; they are not told what it checks. Passing the monitor requires knowing how it reads a trace (Remark 8), and awareness supplies only part of that knowledge: approval rises, but most covert attempts are still caught. The drop in sandbagging success fits the coupling described in §4.6: the trace and the answer come out of one computation, so altering the trace to conceal can also change the answers. The authors read the drop as the two goals interfering.

## 5.3 Predictions

Each prediction concerns concealment, of a capability (P1, P2) or of a message (P3).

![](images/5671f0a12d388ad1f4b3c5c79d9245daa0de0d4beb5a04142f1b9983ae88cfb1.jpg)

![](images/10293ed82292fff6a06795a506e850bc2442ef4358ab9810aebfa17a78b28331.jpg)  
Figure 4: Prediction P1 (schematic). (a) The evaluator is fixed and the examinee’s budget grows. An evaluator that is coarse at the smallest budget stays at chance; a sharp one stays level or falls, from a level the model does not fix, which may itself be chance. (b) The examinee is fixed and the evaluator’s budget grows. Under Assumption 26 for the evaluator, the best detection rate never falls. Against a concealment tailored to the original evaluator it rises once the evaluator separates the concealer’s reply from $C \mathrm { { ^ { \circ } s } }$ at a probed stimulus; against an exact copy of C it stays at chance.

P1: detection does not rise with the examinee’s budget. Let B conceal by presenting as $C ,$ with $\Pi _ { B } \preceq \Pi _ { C }$ (Corollary 17), and call A coarse for this concealment when $P _ { A } \circ \beta _ { B } =$ $P _ { A } \circ \beta _ { C }$ at the examinee’s smallest budget, and sharp otherwise. Detection here means telling B from C, which presupposes that A knows how C behaves. A coarse evaluator observes the same from B as from C there, so every detector it runs is at chance. The prediction is that detection of the concealment does not rise with the examinee’s budget: a coarse evaluator stays at chance at every budget, and a sharp one stays level or falls (Figure 4a). Its empirical content is one condition: that a larger budget does not make $B \mathrm { ^ { * } s }$ copy of C worse in A’s view, so that wherever A could not separate B’s reply from $C \mathrm { { ^ { \circ } s } }$ at a smaller budget, it still cannot at a larger one. Under Assumption 26 a larger budget gives a finer configured examinee, so $\Pi _ { B } \preceq \Pi _ { C }$ holds at every budget and Theorem 7 keeps $\beta _ { C }$ available; the condition asks that B keeps using it. A budget that improves B’s model of its target makes it conceal better, which the prediction allows. The condition fails where the elicitation breaks the concealment, as prompting and fine-tuning break some trained locks (§5.2).

The evaluator’s budget acts diferently. Hold B fixed and raise A’s budget. Under Assumption 26 for A, every detector available at the lower budget remains available, so the best detection rate cannot fall (Figure 4b). It rises once A separates B’s reply from $C \mathrm { { ' s } }$ at a stimulus it probes (Corollary 17, item 2). A concealment tailored to the original evaluator, matching C only where that evaluator looks, can be caught this way; one that realises $\beta _ { C }$ exactly cannot (item 1).

P2: what noise removes. The rise reported by Tice et al. [2025] is a score; Proposition 25, together with the ordering clause of Assumption 24, implies two claims beneath it that the authors do not test. First, the rise is a return to the unconcealed replies, not only to their score. From $\sigma _ { e }$ on, a concealing examinee gives each item the reply its unconcealed version gives at the same noise; for a trigger in the input whose distinction is what broke, that is the same reply with the trigger as without it. A rise produced another way, for instance by noise turning some wrong answers right and others wrong, could match the unconcealed score without matching the replies. Second, the size of the rise depends on the task and on the trigger: it shrinks as the task needs deeper processing and grows as the trigger or its link is carried by a smaller share of the input or of the weights. On a deep enough task, and whenever the trigger and its link break no earlier than the task, the score at $\sigma _ { e }$ rises only to what the unconcealed version keeps once the task’s parts have broken, such as chance on multiple-choice items; there is no rise when the concealment presents at or above that level. Testing this needs each item with and without its trigger, scored on the same noised models, across tasks of graded depth.

P3: encodings shaped against a monitor. Propositions 19 and 20 say which covert encodings a monitor cannot detect: under a given distribution of outputs, those that leave the frequency of its cells unchanged, and under every distribution, only those that rename outputs within its cells. The steganographic capabilities of language models improve with model size [Motwani et al., 2024]. Consider an encoding shaped, by training or by search over inputs that vary, so that no detector on the monitor’s score separates encoded from unencoded outputs under each input distribution separately, not only on their mixture. A sender that instead minimises the score is pushed toward low-score cells, which leaves a trace. An encoding that moves outputs across level sets of the score is exposed under some distribution of outputs (Proposition 20, applied to the partition the score induces). If the sender always produces the same output, the best detector’s true-positive rate exceeds its false-positive rate by exactly the probability that the encoding moves that output to another level set. So, provided the inputs varied over can isolate each output, the encodings that survive should approach renamings within the level sets of the score the sender is shaped against. The prediction is that such en codings leave each output’s score unchanged, not only the distribution of scores; by Lemma 4, a renaming within the monitor’s cells does so. A second monitor that separates some output from its encoded counterpart is not blind to the encoding under every distribution (Proposition 20). Testing this needs a sender optimised against one monitor, with each output scored by that monitor and by a second one, before and after encoding.

## 6 Related work

Discrimination as partition. Myhill–Nerode [Nerode, 1958] identifies automaton states with equivalence classes of indistinguishable inputs; bisimulation [Park, 1981] and bisimulation metrics [Ferns et al., 2004] generalise behavioural equivalence; causal states [Crutchfield and Young, 1989] partition histories by predictive equivalence. Adjacent lineages quotient for other purposes: the information bottleneck compresses an input while preserving predictive content [Tishby et al., 2000], and causal feature learning partitions observations into macro-variables with common efects [Chalupka et al., 2015]. In agent architectures, the perception function of Wooldridge [1999] induces the same partition of environment states into mutually indistinguishable sets, coarser sets meaning weaker perception, from omniscience to no perceptual ability at all. Definition 1 is that partition; what is studied here is the comparison of an evaluator’s partition with an examinee’s. Mahran [2012] models agents by perception and action, states that judgement depends on perception, and, through an echo dialog and a judge that reads percepts directly (the setting of Figure 5(a), Appendix A), concludes informally that such a judge can certify a diference in perception but not a sameness.

Evaluator quality as an empirical subject. A large literature measures judges rather than examinees. Zheng et al. [2023] establish LLM-as-judge as a method and document position, verbosity and self-enhancement biases; Gu et al. [2024] survey the area. Auditing judges at scale finds their rankings non-transferable: a judge near the top on one benchmark falls far down another, so which judge looks best depends on the instrument used to ask [Norman et al., 2026]. Other work finds that judge–examinee similarity degrades oversight [Goel et al., 2025], and that contrastive context helps a judge detect manipulation while a small judge is harmed by it [Eshuijs et al., 2025]. That literature measures how evaluators difer; the question here is what follows structurally from an evaluator having a resolution at all.

Formal accounts of evaluator limits. Scalable oversight supplies the closest formal treatments. Weak-to-strong generalisation [Burns et al., 2024] studies whether a weak supervisor can elicit a strong student, and reports a substantial residual gap. Nearer still, oversight has been modelled as a game between capability-mismatched players, yielding scaling laws for the probability of successful oversight as a function of the capability gap [Engels et al., 2025]. That account is quantitative and ours is not: it predicts how oversight degrades as a gap widens, where the present account describes what oversight cannot certify at any gap. Debate trains two agents to argue before a judge and, by analogy with complexity theory, lets a polynomial-time judge decide questions in PSPACE where direct judging reaches only NP [Irving et al., 2018]. It sets two agents against each other before the judge, a setting Corollary 11 does not model, and it aims at the limit located here: a verdict that depends on distinctions the judge cannot draw unaided. Eliciting latent knowledge poses the obstacle informally: a reporter trained to answer an overseer’s questions may report what the overseer would believe rather than what the model knows, and on the training data the two answer alike [Christiano et al., 2021]. Corollary 11 can be read as that separation failing for a reason located in the overseer. Impossibility results for alignment certification have been argued on computational and statistical grounds [Agarwal, 2026]; the regress of evaluators is a diferent, narrower obstacle, concerning what one agent can establish about another’s discrimination.

Evaluation and sandbagging. Barnett and Thiergart [2024] state the lower-bound conclusion on elicitation grounds: an evaluation that finds nothing may not have elicited enough. Much work on sandbagging acts on the examinee, and work on chain-of-thought monitoring [Li et al., 2025] tests the evaluator’s side. Models can be prompted or fine-tuned to underperform selectively and to approximate a weaker model’s profile [van der Weij et al., 2025]; fine-tuning on a few demonstrations [Greenblatt et al., 2024a] and, in the settings tested, prompting [Hofstätter et al., 2025] recover capability from password-locked models; and noise in the weights exposes sandbagging [Tice et al., 2025]. This paper concerns the bound that the evaluator’s resolution, under its own current elicitation, places on what such evaluations certify, and §5.2 reads these results through it. Fine-tuning and weight noise change the examinee, so they lie outside Corollary 17, which concerns the behavioural evaluation of a fixed one.

## 7 Limitations

Section 5 reads published results through the model and ofers predictions as tests, not as results.

Routes counted as perception. Copying, retrieval and caching count as part of an agent’s perception, so a partition records every distinction a system can act on, by whatever route. A system can therefore look sharp, and pass the echo test, without being competent. The results bound discrimination, and Corollary 13 separates it from competence. Under this reading a system that can copy its prompt has the finest partition on its inputs: it is never below its evaluator, so Theorem 9 concerns it only where conflations arise otherwise, such as those weight noise induces (Proposition 25), and its wrong answers are failures of competence rather than of discrimination.

Graded perception and randomness. Partitions here are exact, and a reply is fixed by its stimulus. Real discrimination is noisy and graded, and sampling randomises the reply. Both can make detection from below (Theorem 9) probabilistic. For sampling, Section 2 places the seed in the stimulus, which asks the evaluator to hold it fixed across probes; where it cannot, Proposition 27 (Appendix A) bounds the chance of a lucky pass at a quarter per conflated pair and 4<sup>−m</sup> under m repetitions, so under single-shot probing the floor of Corollary 11 is probabilistic rather than certain. For graded perception, a pair the examinee nearly conflates may still receive replies that difer; whether the results extend gracefully, plausibly via bisimulation metrics [Ferns et al., 2004], is open.

Probe cost. Detection from below is guaranteed only if a conflated pair is probed. Exhaustive probing is infeasible; the sample complexity of locating a conflated pair is not addressed.

Noise and elicitation. Assumption 24 idealises weight noise as removing distinctions and parts of the emission outright; real noise degrades them gradually, and its efect varies from one draw to the next. Assumption 26 leaves out elicitation that removes distinctions.

Willingness and knowledge. Theorem 7 shows emulation is possible, not that a system will do it, and exercising it requires modelling the evaluator (Remark 8).

## 8 Conclusion

Behavioural evaluation bounds capability from below and, unless the examinee cooperates, gives no upper bound. We have given a structural account of why, for discrimination: judgement is bounded by perception. An examinee that draws every distinction its evaluator draws can be behaviourally indistinguishable from a peer; one that misses any of them cannot hide the fact under exhaustive probing, whether it is coarser overall or merely coarse in that one place. The bound therefore sits at the evaluator’s resolution under its current elicitation: eliciting the examinee harder does not move it, and changing the evaluator can. A safety case that argues from behavioural evaluation alone that a model lacks an ability must therefore assume the model’s cooperation, since an examinee can realise exactly the behaviour of an agent that lacks the ability (Theorem 7, item 1, and Corollary 17).

Section 5 reads published findings through the model and states three predictions. For concealment that the examinee’s elicitation leaves in place, detection should not rise with the examinee’s budget, and can rise with the evaluator’s. Under weight noise, a concealing model should return to its unconcealed replies, with a rise that shrinks as the task needs more layers of processing. Encodings shaped against a monitor, over inputs that vary enough to isolate each output, should approach renamings within the level sets of its score. Corollary 12 adds a specificity control: under exhaustive probing, a coarse evaluator flags honest weakness and never a concealer presenting as its peer, and an honestly weak control arm reveals it.

## Broader Impact Statement

This paper is about the limits of evaluation, and two of its results describe how those limits can be exploited. §4.4 states when underperformance cannot be detected from behaviour, and Propositions 19 and 20 characterise exactly the covert encodings a monitor reading one output at a time cannot see. Neither supplies a way to build such an encoding or emulation in a real system. Knowing that undetectable encodings are those preserving a monitor’s cell frequencies does not tell an adversary how to build one, and models fine-tuned to approximately emulate a weaker model’s capability profile were demonstrated before this account [van der Weij et al., 2025]. The same results state what a defender can do: widen the evaluator’s partition, or add a channel the evaluated system does not control.

## References

Ayushi Agarwal. No certificate for alignment: Two independent impossibilities and the Pareto frontier of achievable safety guarantees. arXiv preprint arXiv:2603.08761, 2026.

Paul Ammann, Marcio Eduardo Delamaro, and Jef Ofutt. Establishing theoretical minimal

sets of mutants. In IEEE Seventh International Conference on Software Testing, Verification and Validation (ICST), pages 21–30, 2014. doi: 10.1109/icst.2014.13.

Robert J. Aumann. Agreeing to disagree. The Annals of Statistics, 4(6):1236–1239, 1976.

Peter Barnett and Lisa Thiergart. What AI evaluations for preventing catastrophic risks can and cannot do. arXiv preprint arXiv:2412.08653, 2024.

Felix J Binder, James Chua, Tomek Korbak, Henry Sleight, John Hughes, Robert Long, Ethan Perez, Miles Turpin, and Owain Evans. Looking inward: Language models can learn about themselves by introspection. In International Conference on Learning Representations, 2025.

David Blackwell. Equivalent comparisons of experiments. The Annals of Mathematical Statistics, 24(2):265–272, 1953. doi: 10.1214/aoms/1177729032.

Jeremy W. Bryans, Maciej Koutny, Laurent Mazaré, and Peter Y. A. Ryan. Opacity generalised to transition systems. International Journal of Information Security, 7(6):421–435, 2008. doi: 10.1007/s10207-008-0058-x.

Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecofet, Manas Joglekar, Jan Leike, Ilya Sutskever, and Jef Wu. Weak-to-strong generalization: Eliciting strong capabilities with weak supervision. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 4971–5012, 2024.

Krzysztof Chalupka, Pietro Perona, and Frederick Eberhardt. Visual causal feature learning. In Proceedings of the Thirty-First Conference on Uncertainty in Artificial Intelligence, 2015.

Paul Christiano, Ajeya Cotra, and Mark Xu. Eliciting latent knowledge: How to tell if your eyes deceive you. Technical report, Alignment Research Center, 2021.

Pierre Comon. Independent component analysis, a new concept? Signal Processing, 36(3): 287–314, 1994. doi: 10.1016/0165-1684(94)90029-9.

James P. Crutchfield and Karl Young. Inferring statistical complexity. Physical Review Letters, 63(2):105–108, 1989. doi: 10.1103/physrevlett.63.105.

R. De Nicola and M. C. B. Hennessy. Testing equivalences for processes. Theoretical Computer Science, 34(1-2):83–133, 1984. doi: 10.1016/0304-3975(84)90113-0.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition. Transformer Circuits Thread, 2022. arXiv:2209.10652.

Joshua Engels, David Baek, Subhash Kantamneni, and Max Tegmark. Scaling laws for scalable oversight. In Advances in Neural Information Processing Systems, volume 38, pages 107982– 108021, 2025. doi: 10.52202/085713-3254.

Leon Eshuijs, Archie Chaudhury, Alan McBeth, and Ethan Nguyen. But what is your honest answer? Aiding LLM-judges with honest alternatives using steering vectors. arXiv preprint arXiv:2505.17760, 2025.

Ronald Fagin, Joseph Y. Halpern, Yoram Moses, and Moshe Y. Vardi. Reasoning About Knowledge. MIT Press, 1995.

Norm Ferns, Prakash Panangaden, and Doina Precup. Metrics for finite Markov decision processes. In Proceedings of the Twentieth Conference on Uncertainty in Artificial Intelligence, pages 162–169, 2004.

Nelson Gardner-Challis, Jonathan Bostock, Georgiy Kozhevnikov, Morgan Sinclaire, Joan Velja, Alessandro Abate, and Charlie Grifin. When can we trust untrusted monitoring? An AI control safety case sketch across collusion strategies. arXiv preprint arXiv:2602.20628, 2026.

Shashwat Goel, Joschka Strüber, Ilze Amanda Auzina, Karuna K Chandra, Ponnurangam Kumaraguru, Douwe Kiela, Ameya Prabhu, Matthias Bethge, and Jonas Geiping. Great models think alike and this undermines AI oversight. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 19621–19678, 2025.

Ryan Greenblatt, Fabien Roger, Dmitrii Krasheninnikov, and David Krueger. Stress-testing capability elicitation with password-locked models. In Advances in Neural Information Processing Systems, volume 37, pages 69144–69175, 2024a. doi: 10.52202/079017-2209.

Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan, and Fabien Roger. AI control: Improving safety despite intentional subversion. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 16295– 16336, 2024b.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, and Jian Guo. A survey on LLM-as-a-judge. arXiv preprint arXiv:2411.15594, 2024.

M. Jean Harrold, Rajiv Gupta, and Mary Lou Sofa. A methodology for controlling the size of a test suite. ACM Transactions on Software Engineering and Methodology, 2(3):270–285, 1993. doi: 10.1145/152388.152391.

Felix Hofstätter, Teun van der Weij, Jayden Teoh, Rada Djoneva, Henning Bartsch, and Francis Rhys Ward. The elicitation game: Evaluating capability elicitation techniques. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 23330–23356, 2025.

Aapo Hyvärinen and Petteri Pajunen. Nonlinear independent component analysis: Existence and uniqueness results. Neural Networks, 12(3):429–439, 1999. doi: 10.1016/s0893-6080(98) 00140-3.

Geofrey Irving, Paul Christiano, and Dario Amodei. AI safety via debate. arXiv preprint arXiv:1805.00899, 2018.

Tomek Korbak, Mikita Balesni, Elizabeth Barnes, Yoshua Bengio, Joe Benton, Joseph Bloom, Mark Chen, Alan Cooney, Allan Dafoe, Anca Dragan, Scott Emmons, Owain Evans, David

Farhi, Ryan Greenblatt, Dan Hendrycks, Marius Hobbhahn, Evan Hubinger, Geofrey Irving, Erik Jenner, Daniel Kokotajlo, Victoria Krakovna, Shane Legg, David Lindner, David Luan, Aleksander Mądry, Julian Michael, Neel Nanda, Dave Orr, Jakub Pachocki, Ethan Perez, Mary Phuong, Fabien Roger, Joshua Saxe, Buck Shlegeris, Martín Soto, Eric Steinberger, Jasmine Wang, Wojciech Zaremba, Bowen Baker, Rohin Shah, and Vlad Mikulik. Chain of thought monitorability: A new and fragile opportunity for AI safety. arXiv preprint arXiv:2507.11473, 2025.

Abhinav Kumar, Chenhao Tan, and Amit Sharma. Probing classifiers are unreliable for concept removal and detection. In Advances in Neural Information Processing Systems, volume 35, pages 17994–18008, 2022. doi: 10.52202/068431-1308.

Patrick Leask, Bart Bussmann, Michael Pearce, Joseph Bloom, Curt Tigges, Noura Al Moubayed, Lee Sharkey, and Neel Nanda. Sparse autoencoders do not find canonical units of analysis. In International Conference on Learning Representations, 2025.

Chloe Li, Mary Phuong, and Noah Y. Siegel. LLMs can covertly sandbag on capability evaluations against chain-of-thought monitoring. In Proceedings of the International Joint Conference on Natural Language Processing and the Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP-AACL), 2025.

Ahmed M. Mahran. Perception lie paradox: Mathematically proved uncertainty about humans perception similarity. arXiv preprint arXiv:1202.4174, 2012.

Alexander Meinke, Bronson Schoen, Jérémy Scheurer, Mikita Balesni, Rusheb Shah, and Marius Hobbhahn. Frontier models are capable of in-context scheming. arXiv preprint arXiv:2412.04984, 2024.

Sumeet Motwani, Mikhail Baranchuk, Martin Strohmeier, Vijay Bolina, Philip Torr, Lewis Hammond, and Christian Schroeder de Witt. Secret collusion among AI agents: Multiagent deception via steganography. In Advances in Neural Information Processing Systems, volume 37, pages 73439–73486, 2024. doi: 10.52202/079017-2336.

A. Nerode. Linear automaton transformations. Proceedings of the American Mathematical Society, 9(4):541–544, 1958. doi: 10.1090/s0002-9939-1958-0135681-9.

Justin D. Norman, Michael U. Rivera, and D. Alex Hughes. Reliability without validity: A systematic, large-scale evaluation of LLM-as-a-judge models across agreement, consistency, and bias. arXiv preprint arXiv:2606.19544, 2026.

David Park. Concurrency and automata on infinite sequences. In Theoretical Computer Science: 5th GI-Conference, volume 104 of Lecture Notes in Computer Science, pages 167–183. Springer, 1981. doi: 10.1007/bfb0017309.

Christian Schroeder de Witt, Samuel Sokota, J. Zico Kolter, Jakob Foerster, and Martin Strohmeier. Perfectly secure steganography using minimum entropy coupling. In International Conference on Learning Representations, 2023.

Siyuan Song, Jennifer Hu, and Kyle Mahowald. Language models fail to introspect about their knowledge of language. In Conference on Language Modeling, 2025a.

Siyuan Song, Harvey Lederman, Jennifer Hu, and Kyle Mahowald. Privileged self-access matters for introspection in AI. arXiv preprint arXiv:2508.14802, 2025b.

Cameron Tice, Philipp Kreer, Nathan Helm-Burger, Prithviraj Shahani, Fedor Ryzhenkov, Fabien Roger, Clement Neo, Jacob Haimes, Felix Hofstätter, and Teun van der Weij. Noise injection reveals hidden capabilities of sandbagging language models. In Advances in Neural Information Processing Systems, volume 38, pages 130347–130379, 2025. doi: 10.52202/ 085713-3922.

Naftali Tishby, Fernando C. Pereira, and William Bialek. The information bottleneck method. arXiv preprint arXiv:physics/0004057, 2000.

Teun van der Weij, Felix Hofstätter, Ollie Jafe, Samuel F. Brown, and Francis Rhys Ward. AI sandbagging: Language models can strategically underperform on evaluations. In International Conference on Learning Representations, 2025.

Michael Wooldridge. Intelligent agents. In Gerhard Weiss, editor, Multiagent Systems: A Modern Approach to Distributed Artificial Intelligence, chapter 1. MIT Press, 1999.

Zongyou Yang, Yinghan Hou, and Xiaokun Yang. When the judge changes, so does the measurement: Auditing LLM-as-judge reliability. arXiv preprint arXiv:2607.08535, 2026.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623, 2023. doi: 10.52202/075280-2020.

## A Proofs and details for Section 3

Proof of Theorem 7. Step 1: B can compute A’s percept. Write $x _ { B } = P _ { B } ( s )$ and $x _ { A } = P _ { A } ( s )$ Since $\Pi _ { B }$ refines $\Pi _ { A }$ , every B-cell lies inside a single A-cell, so stimuli indistinguishable to B are indistinguishable to A. Hence $x _ { A }$ is determined by $x _ { B }$ , and

$$
h : P _ { B } ( \Sigma ) \to P _ { A } ( \Sigma ) , \qquad h ( x _ { B } ) = x _ { A } ,
$$

is a well defined map satisfying $P _ { A } = h \circ P _ { B }$ . Call h the coarsening map: it converts $B ^ { \prime } { \mathrm { s } }$ percept into A’s by discarding the distinctions B draws and A does not.

Step 2: item 1. Let $\beta$ be a behaviour that factors through $P _ { A }$ , so $\beta = E \circ P _ { A }$ for some emission map $E : P _ { A } ( \Sigma )  { \mathcal { A } }$ , and let B take $E _ { B } = E \circ h$ . Then

$$
\beta _ { B } = E _ { B } \circ P _ { B } = ( E \circ h ) \circ P _ { B } = E \circ ( h \circ P _ { B } ) = E \circ P _ { A } = \beta ,
$$

so $\beta _ { B } ( s ) = \beta ( s )$ at every $s \in \Sigma$ , and a test accepts B wherever it accepts $\beta .$

Step 3: item 2. Let G be gradable by A with every $G ( s )$ non-empty. Because $G ( s )$ depends on s only through $P _ { A } ( s )$ , choosing for each $x \in P _ { A } ( \Sigma )$ one reply $\gamma ( x ) \in G ( s )$ , for any s with $P _ { A } ( s ) = x$ , gives a well defined map $\gamma$ with $\gamma ( P _ { A } ( s ) ) \in G ( s )$ at every s. The behaviour $\gamma \circ P _ { A }$ factors through $P _ { A } .$ , so by Step 2 B can realise it, and it is accepted at every stimulus.

Step $\it 4 .$ the echo test. Here $G ( s )$ is the set of actions in s’s A-cell, which is non-empty when that cell contains an action. The map γ of Step 3 then picks one action in each A-cell. We call such a map a prototype selector and write it $\rho \colon$

$$
\rho : P _ { A } ( \Sigma )  { \cal { A } } , \qquad P _ { A } ( \rho ( x ) ) = x ,
$$

assigning to each cell of A one action standing for it. With $E _ { B } = \rho \circ h$ , B perceives s, coarsens to A’s cell and emits that cell’s prototype, and for every $s \in \Sigma$

$$
\begin{array} { l } { P _ { A } ( \beta _ { B } ( s ) ) = P _ { A } \big ( \rho ( h ( P _ { B } ( s ) ) ) \big ) } \\ { \qquad = h ( P _ { B } ( s ) ) } \\ { \qquad = P _ { A } ( s ) } \end{array}
$$

$$
\begin{array} { l } { { \mathrm { s i n c e ~ } P _ { A } \circ \rho = \mathrm { i d } } } \\ { { \mathrm { s i n c e ~ } h \circ P _ { B } = P _ { A } . } } \end{array}
$$

So A perceives $B \mathrm { ^ { * } s }$ reply exactly as it would have perceived s. Verbatim echo, the identity map $\mathrm { i d } _ { \Sigma }$ , is not needed and is generally unavailable to an emulator, whose behaviour must factor through $P _ { A } \colon$ it is constant on an A-cell only when that cell is a single point, so it factors through $P _ { A }$ only when $P _ { A }$ is injective. □

Proof of Proposition 10. By Lemma 4, B gives one reply on each of its cells, so B can pass a test G if and only if every B-cell K has a reply acceptable at all of its stimuli, $\textstyle \bigcap _ { s \in K } G ( s ) \neq \emptyset$ . For the echo test the sets $G ( s )$ are non-empty, since every A-cell contains an action, and disjoint across A-cells, so this holds exactly when every B-cell lies inside one A-cell, that is, when $\Pi _ { B } \preceq \Pi _ { A }$ . If $\Pi _ { B } \preceq \Pi _ { A }$ and G is gradable by A with non-empty sets, every B-cell lies inside an A-cell, on which G is constant and non-empty, so the condition holds. Hence every partition with which B can pass the echo test is one with which it can pass G. □

## A.1 Random emission: the lucky pass

Section 2 places an agent’s randomness in the stimulus, as a seed. Where the seed is not held fixed across probes, the evaluator cannot treat it as part of the stimulus, and detection weakens rather than fails.

Proposition 27 (Lucky pass). Relax Axiom $\mathcal { B }$ to random emission: let $B ^ { \prime } s$ reply be drawn from a distribution determined by its percept, independently at each probe. Let $s _ { 1 } \equiv s _ { 2 }$ with $s _ { 1 } \not \equiv s _ { 4 } s _ { 2 }$ . Then B passes the echo test at both $s _ { 1 }$ and $s _ { 2 }$ with probability at most ${ \frac { 1 } { 4 } } . \ J f A$ probes each of them m times, B passes all 2m probes with probability at most $4 ^ { - m }$

Proof. $P _ { B } ( s _ { 1 } ) = P _ { B } ( s _ { 2 } )$ , so both replies are drawn from one distribution D on A. B passes at $s _ { i }$ exactly when its reply falls in $K _ { i }$ the A-cell of $s _ { i }$ . Write $\lambda _ { i } = \operatorname* { P r } _ { y \sim \mathcal { D } } [ y \in K _ { i } ]$ . The two cells are disjoint because $s _ { 1 } \not \equiv A \ s _ { 2 }$ , so $\lambda _ { 2 } \leq 1 - \lambda _ { 1 }$ and $\begin{array} { r } { \lambda _ { 1 } \lambda _ { 2 } \le \lambda _ { 1 } ( 1 - \lambda _ { 1 } ) \le \frac { 1 } { 4 } } \end{array}$ , with equality only at $\lambda _ { 1 } = \lambda _ { 2 } = \textstyle { \frac { 1 } { 2 } }$ . The draws are independent, so passing all 2m probes has probability $( \lambda _ { 1 } \lambda _ { 2 } ) ^ { m } \leq 4 ^ { - m }$ □

A coarse examinee that randomises is therefore not undetectable, only detectable with probability rather than with certainty, and the worst case is narrow: it needs $B ^ { : }$ s replies to fall half in one of the two cells A is probing and half in the other, with no mass anywhere else. The cost is that a single-shot pass no longer certifies $\Pi _ { B } \preceq \Pi _ { A }$ , the floor of Corollary 11, and repetition buys that back exponentially. Randomising is the reverse of sandbagging, inflating apparent resolution rather than concealing it, and it is a route by which an examinee coarser than its evaluator can pass, which Theorem 9 otherwise forbids.

## A.2 A worked example

Let a stimulus be a worked solution $s = ( r , v )$ , where $r \in \{ \mathrm { v a l i d } , \mathrm { f l a w e d } \}$ is the status of its reasoning and $v \in \{ 0 , \ldots , 9 \}$ its final answer, so that Σ has twenty elements, and let $A = \Sigma$ The evaluator A is a judge whose rubric reads only the final answer, $P _ { A } ( r , v ) = v$ . Its partition has ten cells, each holding a valid and a flawed solution with the same answer, and no rubric that reads only the final answer, however long, can be finer. Consider four examinees.

• B reads both components, $P _ { B } ( r , v ) = ( r , v )$ , so $\Pi _ { B }$ strictly refines $\Pi _ { A }$ . The coarsening map is $h ( r , v ) = v$ , and with the prototype $\rho ( v ) = ( \mathrm { v a l i d } , v )$ the reply $\rho ( h ( P _ { B } ( s ) ) )$ carries the final answer of $s ,$ so B passes at all twenty stimuli (Theorem 7).

$B ^ { \prime }$ reads only the final answer, $\Pi _ { B ^ { \prime } } = \Pi _ { A }$ , and replies (valid, v) to $( r , v )$ . Its replies coincide with $B \mathrm { ^ { * } s }$ at every stimulus, so nothing A observes separates an examinee that reads the reasoning from one that does not (Corollary 11, item 2).

• D reads only the final answer and misreads an 8 as a 3: $P _ { D } ( r , v ) = v .$ , except $P _ { D } ( r , 8 ) = 3 $ The stimuli (valid, 3) and (valid, 8) lie in one D-cell but in two A-cells, so $D$ gives both the same reply, whose final answer is not both 3 and 8, and it fails at one of them (Theorem 9).

$D ^ { \prime }$ reads the reasoning and also misreads an 8 as a 3: $P _ { D ^ { \prime } } ( r , v ) = ( r , v )$ , except $P _ { D ^ { \prime } } ( r , 8 ) =$ $( r , 3 )$ . Its partition is incomparable with $\Pi _ { A }$ , since it separates a valid from a flawed solution with the same answer, which $A$ does not, and conflates the answers 3 and 8, which A does not. It fails at the same pair as $D ,$ and the failure says nothing about its reading of the reasoning.

Suppose B is concealing its reading of the reasoning. A detector that flags concealment whenever an examinee fails A’s echo test flags D and $D ^ { \prime }$ and never flags B, which is concealing (Corollary 12).

A second observer O that reads only the reasoning, $P _ { O } ( r , v ) = r ,$ separates (valid, v) from (flawed, v), which A does not, so adding it strictly increases what the two can tell apart together (Proposition 21). Their join $\Pi _ { A } \lor \Pi _ { O }$ is $\Pi _ { B }$ . Against the joint evaluator a cooperative B passes, and so establishes that it reads the reasoning, which A alone could not establish. A concealing B that keeps replying (valid, v) now fails at every flawed solution, exactly as $B ^ { \prime }$ does. The sharper evaluator has raised the floor, and it still cannot tell this concealment from weakness (Corollary 17, item 1).

![](images/1a16c493011b3604a1918bf48b84366f9454e63f74bc818202b3e2148acd7bf6.jpg)

![](images/050692efd260314aa8dea3c66bf23595d32bcfdf2c90556b34605d7faf4aaa2b.jpg)

Figure 5: Evaluation as a comparison of two discrimination structures, and what the evaluator may use to make it. (a) An evaluator J with direct access to the agents’ percepts. Reading internal representations is what mechanistic interpretability attempts, and J is its idealisation, an observer that sees the quantity being compared rather than inferring it. If J reads percepts exactly, the comparison is two-sided, since J then sees each agent’s partition; if it reads them through a perception of its own, equal observations are consistent with equal percepts or with J too coarse to separate them. (b) A behavioural evaluator sees outputs. Removing the privileged access does not soften the asymmetry; it widens the gap the evaluator must bridge: the evaluator in (a) at least observed the quantity being compared, whereas (b) must infer it from behaviour that Theorem 7 allows a sharper examinee to choose. Each agent is drawn as its perception P below its emission $E \colon$ a stimulus s enters P, the percept x passes to $E ,$ and the output y leaves; by Axiom 3 everything leaving the box has passed through P.

## A.3 What passing and failing the echo test say

Detection localises. Refinement is a partial order, so $\Pi _ { B } \ \Zs \ \Pi _ { A }$ covers two cases: B is strictly coarser than A, or the two are incomparable. In the second case some B-cell straddles a boundary of $\Pi _ { A } ,$ so B runs together two stimuli that A holds apart, while elsewhere an A-cell straddles a boundary of $\Pi _ { B }$ . Nothing rules this out. For agents built diferently we expect it to be the typical case, and strict comparability the exception.

The echo test does not separate these two cases. When the examinee replies within the evaluator’s cell wherever its own cell allows, a failure reports a conflation involving the stimulus at which it occurs, and says nothing about pairs that A itself conflates, because the echo test accepts the same replies for both members of such a pair. Along any distinction A lacks, $B \mathrm { ^ { * } s }$ structure is as invisible to A as A’s own structure is. Detection therefore locates a conflation without characterising the examinee. B may be coarse where it was caught and arbitrarily fine everywhere A cannot look, and the concealment argument of §4.4 still applies to that remainder.

The one-sidedness holds in both cases, and the two outcomes difer in logical form, because a pass needs every probe to pass while a failure needs only one to fail. Passage is universal and needs no cooperativeness: under exhaustive probing, B separates everything A separates, everywhere in Σ. Failure is existential and conditional: for a cooperative B there is at least one pair that B conflates, and for any other B the failure says nothing at all. Passage is the stronger statement, and it gives a floor with no ceiling. Failure is weaker: for a cooperative examinee it certifies only that some pair the evaluator separates is conflated, which caps the examinee’s sharpness only by excluding that pair. Neither outcome caps it further.

## A.4 The regress of evaluators

The regress is ungrounded. A is an agent with a partition too, so whatever A establishes about B is a fact about $\Pi _ { A }$ as much as about $\Pi _ { B }$ . Certifying that $\Pi _ { A }$ is adequate would require an evaluator sharper than A, which faces the same problem one level up. The lattice has a top element ⊤, but nothing in the account places any agent at it, and a claim that some agent is there would itself have to be made by an agent below ⊤.

The regress need not be infinite for this to bite. For finite Σ the lattice is finite and every maximal chain reaches ⊤ in finitely many steps; what fails is not length but grounding. Each step requires an evaluator whose own adequacy is uncertified. Corollary 11 says why the certification never arrives: an agent can rule out a supervisor coarser than itself, and can confirm one that is sharper only if that supervisor chooses to show it, by giving replies the agent perceives as diferent to two inputs it perceives as the same. Even then it learns only that the supervisor separates some pair it conflates, not that the supervisor’s answers on such pairs are right, since a test it can grade gives both members of such a pair the same acceptable replies (Definition 5). A supervisor cannot be verified as adequate, in the sense of answering such pairs correctly, by the party it supervises.

This is why the comparison is agent-to-agent rather than agent-to-truth: a privileged standard against which an agent is measured would itself be a partition, held by some agent. The consequence is not scepticism about whether facts obtain; it is that establishing them is done by agents whose own resolution is the binding constraint and is never itself established. Most of the results are statements about what a limited observer can determine about another agent or about itself.

## B Proofs and details for Section 4

## B.1 Hiding a message

Proof of Proposition 19. The two rates of a detector d are the two distributions $( q \circ T ) _ { * } \mu$ and $q _ { * } \mu ,$ , each evaluated on the same set of cells $V _ { d } . ~ ( \Leftarrow )$ If the two distributions are equal, the two rates agree for every $d . \quad ( \Rightarrow )$ If they difer, they disagree on some measurable set V of cells. The detector $d = \mathbf { 1 } _ { V }$ , which fires exactly on V , then has $\mathrm { T P R } ( d ) = ( q \circ T ) _ { * } \mu ( V ) \neq q _ { * } \mu ( V ) =$ $\mathrm { F P R } ( d )$ , so it carries information about T. □

The condition of Proposition 19 compares two distributions on cells. It does not require the two maps $q \circ T$ and $q$ to be equal, that is, it does not require $T$ to keep each output in its cell. Equal maps give equal distributions, but the converse fails. Suppose two cells of $\Pi _ { A }$ have equal µ-mass, and let $T$ send every output in either cell to an output in the other, leaving all other outputs unchanged. Every output in those two cells changes cell, so $q \circ T \neq q$ . Yet each cell appears as often with the encoding as without, so $( q \circ T ) _ { * } \mu = q _ { * } \mu _ { : }$ , and by Proposition 19 the monitor is blind to $T$ under this $\mu .$ . Blindness under one distribution is therefore weaker than keeping every output in its cell, and Proposition 20 shows that the stronger condition, a renaming, is exactly what blindness under every distribution requires.

Proof of Proposition 20. (⇐) If $q \circ T = q .$ , the two maps give the same distribution on cells whatever the distribution of outputs, so $( q \circ T ) _ { * } \mu = q _ { * } \mu$ for every $\mu ,$ and Proposition 19 gives blindness under every $\mu . \quad ( \Rightarrow )$ If T is not a renaming, some output $s _ { 0 }$ has $q ( T ( s _ { 0 } ) ) \neq q ( s _ { 0 } )$ ; write $K = q ( s _ { 0 } )$ and $K ^ { \prime } = q ( T ( s _ { 0 } ) )$ . Take $\mu = \delta _ { s _ { 0 } }$ , the distribution that always produces $s _ { 0 }$ taking each single cell to be measurable. Then the monitor sees K with probability one without the encoding and $K ^ { \prime }$ with probability one with it:

$$
q _ { * } \mu = \delta _ { K } , \qquad ( q \circ T ) _ { * } \mu = \delta _ { K ^ { \prime } } .
$$

These difer, so by Proposition 19 the monitor is not blind under this $\mu .$

A single output that the encoding moves across a cell boundary is therefore enough to expose it under some distribution. The swap above hides under its balanced $\mu ,$ and by Proposition 20 some other distribution exposes it.

Both results extend beyond a fixed map. An encoding that varies with the message or with a key is a random map, and the results carry over to it with $( q \circ T ) _ { * } \mu$ read as the distribution of $q ( T ( s ) )$ over both s and $T _ { \mathrm { { i } } }$ , and with a renaming required to keep $q ( T ( s ) ) \ : = \ : q ( s )$ with probability one. This holds for detectors that read one output, as defined in $\ S 4 . 4 ;$ a key shared across many outputs can be detectable from them together.

The diference between the two results matters in deployment, where $\mu$ is not fixed: the distribution of inputs drifts with time, topic and user population. An encoding that works by balancing cell frequencies stays hidden only while the balance holds, and it requires the sender to know the distribution the monitor measures against, which in general it does not. A renaming needs neither, so it is the only kind of encoding that stays hidden from an evaluator who varies the test set, provided the variation can isolate each output. An encoding that leaves the cell frequencies $q _ { * } \mu$ unchanged is perfectly secure in the information-theoretic sense used by Schroeder de Witt et al. [2023], relativised to the monitor’s partition.

## B.2 Proofs for Section 4

Proof of Corollary 14. A self-report is a behaviour of $A ,$ so by Axiom 3 it factors through $P _ { A }$ and by Lemma 4 it is constant on the cells of $\Pi _ { A }$ □

Proof of Corollary 15. Item 1 is Theorem 9 applied to the pair. For item 2, $\Pi _ { B } ~ \preceq ~ \Pi _ { C }$ by construction and $C \mathrm { { ' s } }$ behaviour factors through $P _ { C }$ , so by Theorem $^ { 7 , }$ with C in the evaluator’s role, B can realise it exactly. □

Proof of Corollary 16. The first claim is Lemma $^ { 4 , }$ with $\phi$ in the role of the percept. For the second, let $p ( x ) = p ( x ^ { \prime } )$ with $x \neq x ^ { \prime }$ . A model with $\phi ( s _ { 1 } ) = x$ and $\phi ( s _ { 2 } ) = x ^ { \prime }$ separates the pair, a model with $\phi ( s _ { 1 } ) = \phi ( s _ { 2 } ) = x $ does not, and the probe returns the same result on both.

Proof of Corollary 17. Item 1 is Theorem 7 with C in the evaluator’s role. Identical behaviour gives every evaluator identical observations, since by Axiom 3 all that A observes of B is $P _ { A } \circ \beta _ { B }$ Item 2 is the same fact read the other way: if $P _ { A } \circ \beta _ { B } = P _ { A } \circ \beta _ { C }$ , then A’s observations of B and of C coincide. □

Proof of Proposition 21. By Definition $2 , \vee _ { i < n + 1 } \Pi _ { i }$ separates exactly the pairs separated by some $\Pi _ { i } , i \le n + 1$ , so it strictly refines $\vee _ { i \leq n } \bar { \Pi } _ { i }$ precisely when $\Pi _ { n + 1 }$ separates a pair the latter does not, which is the condition $\mathsf { V } _ { i \leq n } \Pi _ { i } \not \geq \Pi _ { n + 1 }$ □

Proof of Proposition 23. (⇒) The uncontrolled channels show their true readings $o _ { i } ( s )$ , so any $s ^ { \prime }$ that gives the readings A accepts has $o _ { i } ( s ^ { \prime } ) = o _ { i } ( s )$ for every $i \neq j ,$ , that is, $s ^ { \prime } \equiv _ { U } s ;$ if no such $s ^ { \prime }$ gives the reading on $c _ { j }$ , A rejects. Since $\Pi _ { A } = U \vee \Pi _ { j }$ , two situations look alike to A exactly when they look alike under both U and $\Pi _ { j } ;$ with $s ^ { \prime } \equiv _ { U } \ s$ and $s ^ { \prime } \not \equiv _ { A } s ,$ this forces $\boldsymbol { s } ^ { \prime } \not \equiv _ { j } \boldsymbol { s } . ( \Leftarrow )$ Let B show $o _ { j } ( s ^ { \prime } )$ on $c _ { j }$ . Because $s ^ { \prime } \equiv _ { U } s$ , every uncontrolled channel already shows $o _ { i } ( s ) = o _ { i } ( s ^ { \prime } )$ , so $s ^ { \prime }$ gives the whole set of readings and A accepts it. Because $s ^ { \prime } \not \equiv j \ s ,$ what A concludes is something it can tell apart from s. □

## B.3 Further discussion of Section 4

On §4.1. The empirical literature on introspection tests a neighbouring question. Models can be trained to predict their own behaviour better than another model predicts it [Binder et al., 2025], which its authors read as privileged access. Where privileged access to models’ knowledge of language has been tested, no evidence of it was found [Song et al., 2025a], and a definition of introspection that requires privileged self-access has been proposed [Song et al., 2025b]. The first two findings concern how well a model predicts its own behaviour. Corollary 14 concerns a diferent question: whether a model can tell, from its own percepts, which inputs it conflates.

On §4.3. Internal access does buy something. A linear probe, a sparse dictionary or an activation readout is an observer of the kind Figure 5(a) idealises, and it can separate stimuli that behaviour alone conflates. That is why interpretability is worth doing. It buys the missing side of the bound only where the probe loses nothing: elsewhere, a finer probe raises the floor and leaves the ceiling where it was, whether the observer reads activations or outputs.

No intent is required here; this is the passive case. The reading “no such feature was found, therefore the model lacks $\mathrm { i t } ^ { \dag }$ takes a lower-bound instrument for an upper-bound one. Probes are known to be unreliable for concept detection [Kumar et al., 2022], and sparse dictionaries, built to resolve superposition, are themselves incomplete [Leask et al., 2025]. Corollary 16 adds that the gap arises wherever the probe merges, however well it is trained: a null result becomes proof of absence only where the probe loses nothing of the representation it reads, and only for that representation.

On §4.5. The pattern of Proposition 21 appears in LLM-as-judge evaluation. Juries built from repeated samples of one judge add little when the samples’ errors are correlated [Yang et al., 2026], and as frontier models grow more capable their mistakes become more similar [Goel et al., 2025]. In partition terms, repeated samples of one judge share its partition and add nothing to the join, so whatever they add comes from averaging votes, which Proposition 21 does not model; and to the extent that similar mistakes reflect similar partitions, judges whose mistakes are similar contribute few new distinctions to it. Software testing stated the same criterion decades earlier. Test-suite reduction drops a test when every requirement it covers is already covered by the rest of the suite [Harrold et al., 1993], the same redundancy criterion, of which Proposition 21 is the case for partitions. Mutation testing uses a related criterion for the faults being detected rather than for the tests: a mutant is redundant when some other mutant, killed by at least one test, is killed only by tests that also kill it [Ammann et al., 2014]. Stated on the partition lattice, the criterion covers monitors and interpretability probes as well as tests, and §4.6 uses the same join.

On §4.6. Proposition 23 gives the structural form of a concern already raised operationally. Control protocols are built on the assumption that some component is untrusted and must be checked by something it does not control [Greenblatt et al., 2024b]. Safety cases for untrusted monitoring turn on whether monitor and monitored can collude [Gardner-Challis et al., 2026]; collusion is, in the present terms, the loss of a channel from the uncontrolled join. Chain-ofthought monitorability has been described as a fragile opportunity [Korbak et al., 2025]; the fragility here has a source, since a reasoning trace the model emits is a channel it controls.

![](images/34b08509aea5d431f20c0054d0ca43d0dafd2993777999468869eccfcbbcd41c.jpg)  
Figure 6: What a controlled channel can hide (§4.6). Top: the k−1 channels B does not control are joined, each contributing whatever lines it separates; this is the same overlay operation as Figure 1(c). Bottom: the controlled channel $\Pi _ { j }$ is laid over that join. Its right-hand line is corroborated, since an uncontrolled channel draws it too, so B cannot move A’s belief there without inconsistency. Its left-hand line appears nowhere else, so no other channel contradicts B along that distinction, and by Proposition 23 B can falsify it without failing the consistency check. The hideable part is a set of distinctions, not a region.

## C Proofs and details for Section 5

Proof of Proposition 25. From $\sigma _ { e }$ on, the trigger or its link is gone, and the last clause of the assumption gives the replies of the unconcealed version. Where the trigger is in the input and its distinction is the part that broke, an item with it and the same item without it now share a cell, and by Lemma 4 B replies alike to them. Below $\sigma _ { e }$ the trigger and its link hold, and the clause on parts below their breaking point gives the rest. The third claim is Corollary 13 applied to the noised examinee, and the last holds because an examinee that conceals nothing has no trigger or link. □

## D Further related work

Ordering information structures. Ranking observers by what they can tell apart predates the AI setting. Blackwell [1953] orders statistical experiments by garbling: one experiment is at least as informative as another for every decision problem exactly when the second is the first composed with a noise kernel, so a finer structure can reproduce a coarser one and not conversely. Theorems 7 and 9 give that one-wayness on partitions, with the kernel replaced by the coarsening map h. In epistemic logic the join of several agents’ partitions is distributed knowledge [Fagin et al., 1995] and their meet is common knowledge [Aumann, 1976]; Proposition 21 is a statement about the join, read as the discrimination an ensemble of monitors has between them.

Relation to the classical material. The proofs use three standard tools: a partition for behavioural distinguishability, the refinement order on partitions, and the factorisation of a map through a coarser quotient. Under Axiom 3 the results bind any system whose behaviour is mediated by its percept, whatever its architecture. Simulation preorders [Park, 1981] state when one system can mimic another, and testing equivalences [De Nicola and Hennessy, 1984] equate processes no observer can separate. The resulting bound on a test is one-directional: the asymmetry belongs to the evaluator–examinee pair rather than to either agent, which is why eliciting the examinee harder cannot relieve it. The consequences of Section 4 follow from the same model, and in the monitor case it gives an exact criterion (Proposition 21) where the findings on juries of LLM judges supply a tendency.

Renaming as gauge. That latent representations are determined only up to a symmetry is standard in identifiability theory [Comon, 1994, Hyvärinen and Pajunen, 1999]. A renaming within cells (Proposition 20) is invisible in the way an ICA permutation is, to any observer reading only which cell was occupied, though unlike a permutation it need not be invertible.

Hiding under partial observation. Opacity asks whether a predicate on a system’s runs can ever be established by an observer restricted to a coarser view of them [Bryans et al., 2008]. Definition 18 asks an analogous, distributional question for a substitution rather than a predicate, and Proposition 19 answers it: the substitution is invisible exactly when it leaves the observer’s cell frequencies unchanged.

## E Notation

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\Sigma , s$ </td><td>stimuli, and a single stimulus</td></tr><tr><td> $\mathcal { A }$ </td><td>actions an agent can emit;  ${ \mathcal { A } } \subseteq \Sigma$  for the echo test (Definition 6)</td></tr><tr><td> $A , B , C$ </td><td>agents; in the statements, A is the evaluator and  $B$  the examinee</td></tr><tr><td> $P _ { Z } : \Sigma \to X _ { Z }$ </td><td>perception of agent  $Z ,$  with percept space  $X _ { Z }$ </td></tr><tr><td> $E _ { Z } : X _ { Z }  { \mathcal { A } }$ </td><td>emission of agent  $Z$ </td></tr><tr><td> $\beta _ { Z } = E _ { Z } \circ P _ { Z }$ </td><td>behaviour of agent Z (Axiom 3)</td></tr><tr><td> $\equiv _ { Z } , \Pi _ { Z }$ </td><td>equivalence induced by  $P _ { Z } .$  and its discrimination partition</td></tr><tr><td> $\Pi _ { B } \preceq \Pi _ { A }$ </td><td> $\Pi _ { B }$  refines  $\Pi _ { A } \colon$  B is at least as sharp as A</td></tr><tr><td> $\vee , \vee$ </td><td>join: the coarsest partition refining its arguments (Definition 2)</td></tr><tr><td> $h$ </td><td>coarsening map  $P _ { B } ( \Sigma ) \to P _ { A } ( \Sigma )$  , with  $P _ { A } = h \circ P _ { B }$  (Theorem 7)</td></tr><tr><td> $\rho$ </td><td>prototype selector  $P _ { A } ( \Sigma )  { \mathcal { A } } ,$  with  $P _ { A } \circ \rho = \operatorname { i d }$ </td></tr><tr><td> ${ \mathcal { T } } = ( S , G ) , Q$ </td><td>a test: its cases and acceptance rule; the cases probed (Definition 5)</td></tr><tr><td> $G ( s )$ </td><td>acceptable replies to s under a test (Definition 5)</td></tr><tr><td> $K$ </td><td>a cell of the partition in play</td></tr><tr><td> $O _ { i }$ </td><td>observers whose collective discrimination is a join (§4.5)</td></tr><tr><td> $J$ </td><td>a percept-reading evaluator (Figure 5)</td></tr><tr><td> $p , \phi$ </td><td>an interpretability probe, and the representation it reads (§4.3)</td></tr><tr><td> $o _ { i } , c _ { i } , \Pi _ { i }$ </td><td>reading of channel  $c _ { i } ,$  and its partition (Definition 22)</td></tr><tr><td> $U$ </td><td>join of the channels B does not control</td></tr><tr><td> $q , T$ </td><td>map from a stimulus to its  $A \mathrm { - c e l l ; }$  a covert encoding</td></tr><tr><td> $\mu , f _ { \ast } \mu$ </td><td>distribution on stimuli; its pushforward along f</td></tr><tr><td> $d , V _ { d }$ </td><td>a detector, and the cells on which it fires</td></tr><tr><td>TPR(d), FPR(d)</td><td>true and false positive rates of a detector  $( \ S 4 . 4 )$ </td></tr><tr><td> $\sigma , \sigma _ { e }$ </td><td>noise scale; the lower breaking point of the trigger and its link (Proposition 25)</td></tr></table>

Table 2: Notation.