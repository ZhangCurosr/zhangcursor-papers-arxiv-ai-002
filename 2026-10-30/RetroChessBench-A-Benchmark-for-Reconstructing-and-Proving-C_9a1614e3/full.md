# RetroChessBench: A Benchmark for Reconstructing and Proving Chess Histories

Pricope Tidor-Vlad

Poindexter Labs

27 October 2026

## Abstract

Frontier language models such as GPT 6 Astra have contributed to research mathematics, demonstrating substantial reasoning capabilities. Yet serious reasoning failures persist even in tasks governed by explicit rules. We demonstrate this in retrograde chess which is a playground for pure reasoning, where solvers reconstruct legal histories and distinguish possible events from forced ones. We introduce RetroChessBench, a 12-task benchmark covering exact-length constructions, unique endings, and exhaustive answer sets, with references, rubrics, and a terminal-reward environment. In two trials per task and condition, GPT 6 Astra High fails to produce the required final answer in 22/24 attempts without tools and 18/24 with code execution; mean terminal RL rewards are 14.75% and 27.19%, respectively. The examples identify faulty deductions: failed responses reject legal histories and treat temporary obstructions as permanent. Online retrograde archives often provide puzzles and final move sequences without detailed derivations. We hypothesize that limited exposure to these missing explanations in web-derived training data contributes to the observed reasoning failures. This questions how far frontier mathematical reasoning really generalizes beyond familiar solution patterns. Claims of general intelligence require evidence that models can transfer learnt principles to areas with few worked examples.

## 1 Introduction

A chess position records where the surviving pieces stand. It omits the moves that brought them there, the pieces that disappeared, and the promotions that may have occurred. Retrograde analysis reconstructs this missing history under the ordinary rules of chess. The scientific question is whether a model can reason consistently over these hidden histories and distinguish what could have happened from what must have happened.

The distinction changes the standard of proof. An exact-length proof game needs one legal witness. A unique-ending problem needs an argument covering every compatible history. A problem with several possible last moves needs a feasible history for each answer and an exclusion of all others. These obligations remain diferent even when their final answers fit on one line. A six-move ending can depend on a promotion or capture budget established much earlier in the game.

These tasks put a basic expectation of general reasoning under pressure: known rules should remain usable when the question changes from choosing a future move to establishing a necessary past event. The solver must keep track of hypothetical states, revise assumptions, and distinguish constraints that hold in one candidate history from constraints that hold in all of them. Fluent reconstruction can conceal a failure at precisely this distinction.

A concrete failure illustrates the issue. In P0292, a model constructs a legal 47-ply game and concludes that the requested 45-ply game is impossible. Its bound treats a pawn on g6 as an unavoidable obstruction to a bishop’s route. A replayable 45-ply construction sends the bishop through g6 before the pawn arrives. The model has found one possible history and mistaken its schedule for a necessary constraint. Table 1 isolates the error; Appendix D gives the full counterexample.

We introduce RetroChessBench (also on Hugging Face), a 12-task benchmark comprising seven exactlength games, four unique sufixes, and one exhaustive two-answer task. These tasks were selected from a 48-task candidate pool drawn from StrateGems compositions, including two adaptations. References, prompts, board images, responses, rubrics, and replay checks accompany the tasks. A terminal-reward interface accepts alternative valid constructions and scores analytical obligations separately.

We evaluate GPT 6 Astra High twice per task without tools and twice with hosted code execution. Final answers are correct in 2/24 and 6/24 attempts. The code condition produces two valid constructive games and four correct sufixes; three of those sufix answers still lack complete proofs. Five construction tasks fail in all four attempts. These are descriptive outcomes on a set selected using observed failures, not an unbiased estimate of retrograde-chess ability or a causal estimate of tool benefit.

OpenAI reported a Navier–Stokes solution on 8 September 2026 (OpenAI, 2026d). Four weeks later, on 6 October, it released hundreds of mathematical research manuscripts with varying verification status (OpenAI, 2026c,b). Both releases concern internal models; OpenAI describes the Navier–Stokes model as more capable than Astra, whose separate mathematical contributions include prime-gap results (OpenAI, 2026a). ChessBench’s 27 October 2026 snapshot lists Astra at 1745 Elo for XHigh and 1531 for High (ChessBench, 2026a). These reports motivate broader tests of reasoning. The question is whether mathematical progress brings reliable construction and exclusion across other domains.

Backward reasoning is already studied in language-model planning, and reverse chess appears in LongCoT (Ren et al., 2024; Motwani et al., 2026). Our source compositions and answers are public, but availability does not establish training exposure. A leading hypothesis is limited exposure to worked retrograde explanations. The source archive gives P0292’s game with a short theme description and R0185’s ending with a compact rejected branch (The Retrograde Analysis Corner). A game certifies existence but need not teach discovery; a sufix sequence alone does not establish uniqueness. Detailed treatments do exist (Volet, 2003); the hypothesis concerns explanatory coverage in training, which we have not measured. If such coverage is sparse, can a model assemble familiar rules into historical deductions without abundant demonstrations of the method?

Our contribution operationalizes these established proof obligations in hidden chess histories through distinct answer contracts, executable witness checks, recorded proof assessments, and inspectable counterexamples to failed arguments. The benchmark and auditable pilot motivate controlled tests of local rule knowledge, candidate discovery, state representation, and proof verification (Section 9). Those tests can distinguish missing techniques from failures to coordinate techniques the model already possesses.

<table><tr><td>P0292, code run 1</td><td>Observable argument and counterexample</td></tr><tr><td>Invalid inference</td><td>A pawn on g6 prevents Be8-h5, so the required construction needs at least 47 plies.</td></tr><tr><td>Legal counterexample</td><td>The 45-ply reference contains 6.e8=B+ Ke6 7.Bh5 g6 8.g5 gxh5: the bishop traverses g6 before it is occupied.</td></tr><tr><td>Missing obligation</td><td>A lower bound must cover alternative event orders. A valid 47-ply witness establishes no such bound.</td></tr></table>

Table 1: A failure of historical exclusion. The claim is paraphrased from the saved final answer; the counterexample is independently replayed from the initial position. This example establishes an error in the argument, not its internal computational cause.

## 2 Related work

Chess play and state tracking. Chess ofers several distinct evaluation targets. Toshniwal et al. (2022) study piece tracking and legal-move prediction from move sequences, finding that explicit state information helps with limited training data. Harang et al. (2025) evaluate inferred chess states through the actions those states permit. Kolasani et al. (2025) examine reasoning and instruction following during extended forward play. ChessBench derives ratings from game outcomes using an anchored Monte Carlo Elo procedure (ChessBench, 2026b). These settings test following or choosing a continuation from a supplied history. RetroChessBench instead asks the solver to recover unobserved events, including captures and promotions, and sometimes to prove that other histories cannot change the answer. A forward-play rating motivates comparison but supplies no calibration for historical proof dificulty.

Reverse reconstruction and classical planning. LongCoT is the closest benchmark predecessor: its reverse-chess task reconstructs a legal alternating move sequence consistent with a terminal board, side to move, and specified move fragments (Motwani et al., 2026). This shares the existential objective of our proof games. The unique-sufix and exhaustive-answer tasks add universal exclusions over legal histories. PlanBench provides systematic tests of planning and reasoning about actions and change using domains from automated planning (Valmeekam et al., 2023). It establishes a broader methodological precedent for explicit transition rules and checkable plans. Ren et al. (2024) study directional planning biases and combine forward and backward search with verification. Together these studies motivate separating discovery from checking, rather than treating all backward reasoning as one capability.

Abduction and missing premises. Recovering history is a form of reasoning from efects to possible causes. ART asks models to choose or generate plausible explanations of narrative observations (Bhagavatula et al., 2020). Plausibility is a diferent criterion from chess legality, and neither entails uniqueness. ProofWriter generates implications, proofs, and abductive statements over explicit natural-language theories (Tafjord et al., 2021). Sprague et al. (2022) interleave forward deduction and backward abduction to infer missing premises, using round-trip validation to filter candidates. RetroChessBench shares the need to propose hidden events and check their consequences, but the missing objects are time-indexed states and piece identities under a fixed transition system. An explanation must fit a complete legal history; a uniqueness claim must survive every alternative explanation admitted by that system.

Proof search, quantifiers, and generalization. PrOntoQA makes generated reasoning steps amenable to formal analysis. Saparov and He (2023) find that valid individual deductions can coexist with failures to explore alternative proof steps. Saparov et al. (2023) separately test depth, width, and compositional generalization, including hypothetical subproofs. These results motivate our distinction between knowing a local chess rule and coordinating it across a branching historical argument; they do not establish the cause of Astra’s errors. SATBench evaluates constraint satisfaction and unsatisfiability on generated logical puzzles (Wei et al., 2025), a related separation of witnesses from exclusion. RetroChessBench adds capture identities, alternating turns, and hidden event order, while retaining the need to distinguish an existential solution from a universal claim.

Mathematics, constructions, and robustness. MathArena evaluates recent competition mathematics, and Proof or Bluf examines proof quality on olympiad problems (Balunović et al., 2025b; Petrov et al., 2025). MathConstruct accepts automatically checkable mathematical constructions (Balunović et al., 2025a); the same principle governs our exact-length games: any valid witness earns full credit. Those results provide context for asking about transfer, rather than a measurement of transfer in this study. GSM-Symbolic probes mathematical robustness through template variations (Mirzadeh et al., 2025). Its later re-evaluation identifies numerical-distribution and statistical confounds and reports model-specific failure profiles (Długosz et al., 2026). The methodological lesson is relevant here: a performance gap requires controlled explanation. Counterfactual-task evaluations (Wu et al., 2024) and chess-prior interventions (Pleiss et al., 2026) ofer complementary ways to test dependence on familiar representations. RetroChessBench keeps orthodox rules fixed; future transfer tests must additionally control the inference demanded and the solution budget.

Classical retrograde analysis and assessment. The compositions and source solutions predate this benchmark and remain credited to their authors (The Retrograde Analysis Corner). Generalized retrograde reachability on growing boards is PSPACE-complete (Brunner et al., 2020); this does not classify the dificulty of our fixed 8 × 8 instances. Specialist software such as Jacobi already solves proof games (Labelle). Our claim concerns language-model evaluation, not the impossibility of algorithmic reconstruction. Natural-language uniqueness proofs introduce a diferent assessment problem. Known limitations of model judges (Zheng et al., 2023) require separate records for deterministic witness checks, rubric credit, and proof review. A reference receiving full credit is a calibration control, not independent evidence that the proof or judge is correct.

<table><tr><td>Evaluation setting</td><td>Requested object</td><td>Relevant distinction</td></tr><tr><td>Forward chess</td><td>Legal or strong next moves; completed games</td><td>Continuation quality does not establish a hidden history.</td></tr><tr><td>LongCoT reverse chess</td><td>A history satisfying terminal and sequence constraints</td><td>A construction establishes existence.</td></tr><tr><td>PlanBench</td><td>Plans and reasoning about state changes</td><td>Explicit action models permit separate planning and checking tasks.</td></tr><tr><td>ART / ProofWriter</td><td>Plausible missing causes / deductions and abductive premises</td><td>Plausibility, derivability, and necessity are different objectives.</td></tr><tr><td>Formal deduction / satisfiability (PrOntoQA; SATBench)</td><td>Formal derivations; satisfiability and unsatisfiability decisions</td><td>Proof evaluation and excluding all satisfying assignments are established objectives.</td></tr><tr><td>RetroChessBench</td><td>Exact-length games, unique suffixes, complete ending sets</td><td>Distinct answer contracts apply these obligations to hidden histories, with replayable witnesses and inspectable failures.</td></tr></table>

Table 2: Relationship to prior evaluation settings. This comparison identifies the task objective; it does not rank benchmarks or claim that historical reconstruction is new. The cited implementations difer in models, budgets, and task distributions.

## 3 Tasks and proof obligations

Let �<sub>0</sub> be the standard initial state, $T ( s , m )$ the legal transition for move �, and � a terminal piece placement. A full state also includes side to move, castling rights, and en-passant rights. The supplied FEN piece-placement field leaves these unspecified unless the task determines them. Write $\pi ( s )$ for a state’s placement. The set $\mathcal { H } _ { C } ( D )$ contains histories $h = ( m _ { 1 } , \ldots , m _ { n } )$ that start at $s _ { 0 }$ , follow legal transitions, end with $\pi ( s _ { n } ) = D$ and satisfy the task conditions �.

For answer comparison, each transition induces an annotated event $\boldsymbol { e } _ { i } = \left( o _ { i } , d _ { i } , p _ { i } , \nu _ { i } \right)$ : origin, destination, promotion type (or none), and captured-piece type (or none). Write $E ( h ) = ( e _ { 1 } , \ldots , e _ { n } )$ . The capture type is evaluated in the predecessor state; en-passant captures also use that state’s rights. Check and mate markers are derived from the resulting state and are not part of $E ( h )$ ; Section 5 specifies their separate treatment in binary outcomes and rewards. This representation matters: b7×Rc8=R# and $\scriptstyle \mathbf { b 7 } \times \mathbf { N c 8 } = \mathbf { R } \#$ both encode as UCI b7c8r, but are diferent answers to R0171. A comparison that discards the victim merges two required alternatives.

## 3.1 Exact-length constructive proof games

Given � and �, produce any $h \in \mathcal H _ { | h | = n } ( D )$ . Replay checks every move, the number of plies (single moves), and the final placement. Both sides cooperate to reach the diagram. Any valid sequence proves existence and satisfies the task. Uniqueness and minimum length are separate claims.

## 3.2 Unique sufixes

Given $D$ and �, assuming legal reachability as stated in the prompt, determine a sequence � such that

$$
\operatorname { s u f f i x } _ { k } ( E ( h ) ) = \sigma \qquad { \mathrm { f o r ~ e v e r y ~ } } h \in { \mathcal { H } } _ { C } ( D ) .\tag{1}
$$

Many complete histories may share the same final � plies. Equation (1) requires excluding all competing endings, including locally legal predecessors that have no legal earlier history. Replaying one candidate ending cannot establish this claim. A benchmark also needs $\mathcal { H } _ { C } ( D ) \neq \emptyset \colon$ otherwise the universal statement is vacuous. The prompts explicitly let solvers assume reachability, but that does not discharge the benchmark author’s existence obligation. The four sufix references in this bundle provide conditional analytical arguments and local sufix checks, not complete start-to-diagram games. Supplying those witnesses, particularly for R01’s changed rook placement, is an outstanding validation step. Appendix E distinguishes this gap from the fully replayed construction and R0171 controls.

## 3.3 Exhaustive alternative endings

R0171 asks for both possible last moves. Its target is a set,

$$
A = \{ \operatorname { s u f f i x } _ { 1 } ( E ( h ) ) : h \in { \mathcal { H } } _ { C } ( D ) \} ,\tag{2}
$$

with two elements. Full credit requires both alternatives, a feasible history for each, and exclusion of further possibilities.

## 3.4 Why a standard chess engine is insuficient

Stockfish searches forward for strong continuations (Stockfish developers). Historical reconstruction requires additional information: the identity of captured pieces, earlier promotions, and the times at which squares were available. A best-move query does not supply it. Undoing a recorded move also difers from generating every possible predecessor of an arbitrary board.

Code can still help. python-chess provides move generation, legality checks, notation handling, and replay (Fiekas). Custom search, constraint solvers, and proof-game programs can reconstruct histories. Our code condition supplies general hosted execution; it does not supply a complete retrograde solver. Historical reconstruction also difers from computing game-theoretic endgame tablebases backward from terminal states.

## 4 Benchmark construction

## 4.1 From the source archive to the benchmark

We audited 82 primary problem entries in StrateGems Nos. 52–58 and documented their variants. Eligible tasks asked for prior moves or an orthodox proof game. We excluded piece-addition tasks, position release alone, and fairy-chess stipulations. The resulting 58 task instances each received two no-code runs. At least one run failed on 47 instances. We removed P0299 because its analytical completeness proof remained unresolved. The remaining 46 instances and two adaptations form a 48-task candidate pool.

RetroChessBench contains 12 tasks with completed grading and recorded secondary review of all 48 solver responses. Ten were selected using observed results, with R0178 and the easier R01 adaptation included to broaden sufix coverage. The final inventory and attempt slots are frozen, but the selection process was adaptive: it first retained no-code failures and subsequently used code outcomes and proof coverage to select the benchmark tasks. The reported no-code trials include the same attempts used for screening. This creates an asymmetric comparison: screening for no-code failure can make subsequent code trials look better through selection and regression to the mean, even without a tool efect. Later selection on code failures adds another dependency. The retained snapshots document the filtering stages and final task IDs, not a preregistered selection rule. We report a challenge-set baseline and give the selection ledger in Appendix A. A causal tool comparison requires a frozen task set and fresh trials in both conditions.

## 4.2 Composition, adaptations, and attribution

The benchmark comprises seven constructive tasks, four unique-sufix tasks, and one two-alternative task. Constructive lengths range from 35 to 60 plies; sufix lengths are 5, 6, 36, and 40 plies. Table 8 credits every composer and source issue.

R01 adapts Andrey Frolkin’s R0185 (StrateGems No. 56, 2011). We ask for the last six single moves instead of eight because this shorter ending retains the central reasoning challenge; proving its uniqueness still requires looking further back. We also relocate White’s rook from h8 to g8, leaving every other piece in place. The rook remains to the right of Qd8, preserving its role in the eighth-rank obstruction used by the proof. The relocation deliberately changes the original diagram to reduce reliance on exact diagram memorization, should the source have appeared in training. It does not establish absence from pretraining data or measure a reduction in memorization.

R02 adapts Nikolai I. Beluhov’s R0176 (StrateGems No. 53, 2011). The diagram is unchanged; the only task modification is to request the last five single moves instead of forty. This keeps the exposition manageable while preserving the central trap, which arises near the start of the backward analysis. Establishing those five moves still requires excluding a tempting continuation whose failure becomes apparent only further back in the history.

![](images/404c59cf6817428f5999f1d8b4255db4f7f8164524907f0c7f48fab8330ee034.jpg)  
(a) R01: the last six single moves

![](images/b8c68faf15dbcf2dbc97f3f8f2e608350afe5c7076fa3cb27916ad78e4d56b89.jpg)  
(b) R02: the last five single moves  
Figure 1: Two adapted short-answer tasks. R01 changes Frolkin’s R0185 from eight to six single moves and relocates White’s rook from h8 to g8. R02 changes Beluhov’s R0176 from forty to five single moves, with no change to the diagram. Both retain the central historical proof challenge and assume legal reachability. The solver receives both the diagram and its FEN placement.

Both adaptations retain the central proof mechanism under the prompt’s legal-reachability premise. R02 and R0176 share a position and much of their reasoning; Section 7 reports a family-weighted sensitivity analysis.

## 4.3 What the artifact contains

Tasks include prompts, FENs, board images, attribution, reference proofs, answer specifications, and versioned rubrics. Evaluation records contain responses, request metadata, criterion judgments, sequence checks, reviews, and hashes. Each solver rollout is an episode; its trajectory receives a terminal reward and feedback based on the final answer and justification. Intermediate messages and tool calls receive no separate process rewards. This supports reinforcement learning; we evaluate without training a policy.

The benchmark website provides task documents, paired results, and evidence; PDL-Bench on Hugging Face hosts the 12-task retro\_chess subset. The RL environment is available as a tested Docker build bundle, also on Hugging Face, with setup instructions and a validation report. Researchers can export solver inputs, check references, and grade new trajectories against the frozen tasks and rubrics. The paper bundle includes scripts that reproduce the reported tables and aggregate scores from archived judgments. Appendices B and A specify the protocol and archived release.

References and rubrics were developed with AI assistance. The author manually reviewed the solutions on a physical chessboard. A Codex agent performed the recorded secondary reviews. The artifact records these roles and retains attribution for the original compositions and solutions.

## 5 Reward design and grading

## 5.1 Separate answer correctness from proof quality

The evaluator reports final-answer correctness, categorical proof status, and terminal reward separately. A final-answer failure means the required game, sufix, or complete answer set was not produced. Terminal reward scores the task’s full objective. A correct ending with an incomplete proof can earn less than 100%; sound deductions can earn partial credit despite a wrong ending. The public dashboard calls a final-answer failure a “stump.” The primary scientific outcomes are exact-answer success and whether the applicable proof obligation is complete. Aggregate reward is an environment-specific summary, not a percentage of reasoning ability or of tasks solved.

The binary answer metric uses verified sequence status, not full sequence-criterion credit. Omitting or mislabeling + or # does not turn an otherwise verified game, sufix, or complete answer set into a final-answer failure. Such defects can reduce analytical sequence credit and hence reward; they do not reduce the binary constructive witness reward. Required move order, promotion identities, and captured-piece alternatives remain substantive constraints. All 48 retained verdicts agree with this rule, and none of the 40 failures is notation-only (Appendix E).

## 5.2 Deterministic witness rewards

For a constructive task and submitted candidate ℎ, the reward is

$$
r _ { \mathrm { g a m e } } ( h ) = \left\{ \begin{array} { l l } { 1 , } & { h \mathrm { r e p l a y s ~ l e g a l l y ~ f r o m } s _ { 0 } , \ : | h | = n , \pi ( s _ { n } ) = D , } \\ { 0 , } & { \mathrm { n o ~ v a l i d ~ c o m p l e t e ~ w i t n e s s ~ i s ~ s u p p l i e d } . } \end{array} \right.\tag{3}
$$

The checker accepts equivalent move encodings and checks supplied piece and capture identities. It reports check-annotation errors separately from move legality. A valid witness earns full reward regardless of its agreement with the reference or the length of its explanation. A legal game that arrives two plies late earns zero: the move budget is part of the problem. An impossibility claim also earns zero on these solvable tasks.

A model judge records rubric-based diagnostic feedback on constructive reasoning. That score does not change Equation (3): a wrong-length game can earn 27.5% diagnostic credit and still receive zero terminal reward. Unparseable or ambiguous submissions go to review before scoring.

## 5.3 Analytical partial rewards

Sufix and answer-set rubrics contain at most ten weighted proof obligations, proposed from the reference by a code-enabled Azure Astra High rubric-maker and reviewed, calibrated, and frozen. Criteria specify necessary deductions, such as capture bounds or excluded king origins. A separate Azure GPT 6 Astra High call with hosted code execution acts as an LLM-as-a-judge (LLMAJ). It receives the problem, reference proof, expected answer, rubric, and submission; assigns reasoning credit $c _ { j } \in \{ 0 , \frac { 1 } { 2 } , 1 \}$ ; and quotes supporting evidence. Deterministic checks override its final-sequence score.

With weights summing to 100, the provisional score is

$$
\widetilde r = \frac { 1 } { 1 0 0 } \sum _ { j } w _ { j } c _ { j } .\tag{4}
$$

An incorrect ending, incomplete proof, or fatal error caps reward at 0.90. A complete correct solution can receive 1.0. Uncertain judgments, invalid schemas, provider errors, and unresolved references produce a null reward pending review. Null values are excluded from averages and training rewards.

Sequence partial credit is task-specific. R01 and R02 allow half credit for specified near-complete endings; the longer sufixes use stricter rules. R0171 checks both captured-piece alternatives. The reported rewards retain these notation-sensitive policies; binary final-answer correctness remains separate. The frozen policy records these choices, and Appendix C lists the R01/R02 weights. Aggregate reward consequently depends on the task mixture and rubric design.

## 5.4 Calibration and corrections

All 12 reference controls (the “golden answers”) score 100%. Because those references were used to develop the rubrics, this checks calibration rather than independent judge accuracy. Solver and judge also use the same model family, leaving room for correlated errors.

Secondary audits cover all 48 answers and 12 controls. One retained score changed: R0178 code run 1 rose from 82.5% to 88% after review found inconsistent credit for a valid blocker-exclusion argument. The record preserves both scores and the correction. R0171 required a reference repair: an earlier grading bundle

loaded an obsolete incomplete draft. We synchronized the completed reference, versioned its task and rubric, and regenerated its five grades. That reference includes both full feasibility witnesses and the analytical exclusion argument.

## 6 Experimental setup

We evaluate GPT 6 Astra High using RetroChessBench’s terminal-reward RL environment. The study uses 48 archived Azure solver rollouts, subsequently scored as episodes: 12 tasks, two runs, and two solver tool presets. Each trajectory’s final answer and written justification determine its terminal reward � ∈ [0, 1], alongside separate final-answer and proof-status assessments. Analytical tasks allow partial reward; constructive games use the binary witness reward (Section 5). Intermediate messages and tool calls receive no process rewards, and this study performs no policy updates.

Calls were made on 26–27 October 2026 through Azure with deployment alias gpt-6-astra and reasoning efort high. We call this configuration GPT 6 Astra High. The response metadata identifies the alias, not an immutable weight snapshot. Separate ChatGPT XHigh experiments are excluded.

The two solver presets share task and reward definitions: no code supplies no tools; code enabled supplies Azure’s hosted code interpreter with an automatically provisioned 4 GB container. The grading pipeline is the same in both presets. The system prompt specifies only a JSON object containing one solution string and adherence to the task. All four runs for a task receive identical text after newline normalization, including FEN, and the same high-detail board image. Solvers receive no reference, rubric, or chess strategy. Web search is unavailable. Requests leave seed, temperature, and maximum output tokens unset, using provider defaults.

All selected responses completed, and every code-enabled rollout invoked the tool. The archive records 219 completed code calls and none in the no-code preset. It retains code payloads but no console outputs, so we cannot confirm library installation, useful verifier feedback, or whether a solver used or ignored that feedback. In particular, the record does not establish successful Stockfish use.

Median output-token usage, including reasoning tokens, is 42,295 without code and 58,443 with code. Compute is therefore unmatched; tool access, token usage, and search behavior vary together. The comparison uses the frozen completed pair for each task. Technical retries, two later R01 no-code runs, and separate ChatGPT outputs remain outside the 48-episode analysis.

The analysis pins task IDs, answer hashes, references, rubrics, and reviewed grades. A local script regenerates every numerical summary, task table, and plot. It also replays the constructive references and successful model games, checking legality, length, final placement, and premature automatic termination. Analytical proofs retain their separate rubric assessments.

## 7 Results

## 7.1 Final answers and completed proofs

Table 3 reports task outcomes before aggregating reward. Code-enabled runs produce more valid constructive games and more correct sufixes. The sufix distinction is particularly informative: correct endings rise from 2/8 to 4/8, while complete proofs under the archived review number 2/8 and 1/8. The other three correct code sufixes omit required exclusions. These small, selected samples do not show that code harms proof quality; they show that answer recovery and proof completion need separate measurement.

The proof-status categories distinguish failure to finish from an afirmative invalid inference. In each condition, five analytical responses contain an identified false step; the remainder are complete or incomplete under the archived review. No analytical grade in the benchmark is pending. This is a secondary analysis of existing judgments.

<table><tr><td>Primary outcome</td><td>No code</td><td>Code enabled</td></tr><tr><td>Correct final answers, all tasks</td><td>2/24</td><td>6/24</td></tr><tr><td>Valid exact-length constructive games</td><td>0/14</td><td>2/14</td></tr><tr><td>Correct unique-suffix answers</td><td>2/8</td><td>4/8</td></tr><tr><td>Full-credit unique-suffix proofs</td><td>2/8</td><td>1/8</td></tr><tr><td>Complete two-alternative answers</td><td>0/2</td><td>0/2</td></tr></table>

Table 3: Correct answers and completed proof obligations. Rows overlap: the three task families partition the all-task answer row, and full-credit sufix proofs are a subset of correct sufix answers. Analytical completeness follows the archived rubric assessments.
<table><tr><td>Analytical proof status</td><td>No code</td><td>Code enabled</td></tr><tr><td>Complete under archived review</td><td>2/10</td><td>1/10</td></tr><tr><td>Incomplete; no false step identified</td><td>3/10</td><td>4/10</td></tr><tr><td>Contains an identified false inference</td><td>5/10</td><td>5/10</td></tr><tr><td>Unresolved</td><td>0/10</td><td>0/10</td></tr></table>

Table 4: Mutually exclusive categories for the 20 sufix/set responses, derived from archived criterion-level reviews. “No false step identified” describes the review record, not a certification of every assertion. An answer with both omissions and an identified false inference is classified in the latter category. The per-answer evidence is in analysis/reviewer\_audit.json.

## 7.2 Environment reward

Mean terminal reward is 14.75% without code and 27.19% with code, an observed diference of 12.44 percentage points (Table 5). The final-answer failure counts are 22/24 and 18/24. Binary witness rewards and weighted analytical rewards measure diferent objectives; the combined mean summarizes this environment’s task mixture. Per-attempt rewards and criterion-level comments are available in the online grading report.

<table><tr><td>Solver condition</td><td>Attempts</td><td>Final-answer failures</td><td>Mean reward</td><td>Full reward</td></tr><tr><td>No code</td><td>24</td><td>22/24</td><td>14.75%</td><td>2/24</td></tr><tr><td>Code enabled</td><td>24</td><td>18/24</td><td>27.19%</td><td>3/24</td></tr></table>

Table 5: Secondary reward summary for RetroChessBench. Full reward means the constructive or analytical obligation is satisfied under the frozen scoring policy. Scores depend on selection, task mix, rubric weights, and the construction gate.

Five tasks score zero in every run: P0292, P0297v, P0332, P0333, and P0335. Their failed answers include missing constructions and legal games of the wrong length. At the task level, code raises four reward means, lowers three, and leaves five unchanged. Both short adaptations score lower with code. The aggregate diference masks substantial variation and is not an estimate of a causal tool efect.

## 7.3 Results by proof obligation

<table><tr><td>Task family</td><td>Tasks</td><td>Attempts per mode</td><td>No-code reward</td><td>Code reward</td></tr><tr><td>Constructive game</td><td>7</td><td>14</td><td>0.00%</td><td>14.29%</td></tr><tr><td>Unique suffix</td><td>4</td><td>8</td><td>41.38%</td><td>54.00%</td></tr><tr><td>Two last-move alternatives</td><td>1</td><td>2</td><td>11.50%</td><td>10.25%</td></tr></table>

Table 6: Mean terminal reward by task family. Construction rewards are binary, whereas the other families allow analytical partial credit. The percentages therefore express diferent task-specific objectives and should not be read as a calibrated common dificulty scale.

![](images/725f129c460ab2cdee23ad9e2c158c0e93c45278e37e64ce671d03887ee674a0.jpg)

Figure 2: Paired task-level mean rewards. Each marker averages two runs; the complete run-level values appear in Table 7. “Game,” “Sufix,” and “Set” denote the three proof obligations. Shared zero markers represent failures under the binary construction objective, not an absence of all useful reasoning.
<table><tr><td colspan="3"></td><td colspan="2">Run rewards (%)</td><td colspan="2">Failures</td></tr><tr><td>Task</td><td>Type</td><td>Plies</td><td>No code</td><td>Code</td><td>No code</td><td>Code</td></tr><tr><td>R0171</td><td>Set</td><td>1</td><td>11.5, 11.5</td><td>11.5,9</td><td>2/2</td><td>2/2</td></tr><tr><td>P0292</td><td>Game</td><td>45</td><td>0,0</td><td>0,0</td><td>2/2</td><td>2/2</td></tr><tr><td>P0297v</td><td>Game</td><td>60</td><td>0,0</td><td>0,0</td><td>2/2</td><td>2/2</td></tr><tr><td>P0332</td><td>Game</td><td>44</td><td>0,0</td><td>0,0</td><td>2/2</td><td>2/2</td></tr><tr><td>P0333</td><td>Game</td><td>44</td><td>0,0</td><td>0,0</td><td>2/2</td><td>2/2</td></tr><tr><td>P0335</td><td>Game</td><td>54</td><td>0,0</td><td>0,0</td><td>2/2</td><td>2/2</td></tr><tr><td>R02</td><td>Suffix</td><td>5</td><td>100,16</td><td>16,47</td><td>1/2</td><td>2/2</td></tr><tr><td>P0283</td><td>Game</td><td>35</td><td>0,0</td><td>0,100</td><td>2/2</td><td>1/2</td></tr><tr><td>P0295</td><td>Game</td><td>46</td><td>0,0</td><td>100,0</td><td>2/2</td><td>1/2</td></tr><tr><td>R0176</td><td>Suffix</td><td>40</td><td>10,31</td><td>5,78.5</td><td>2/2</td><td>1/2</td></tr><tr><td>R0178</td><td>Suffix</td><td>36</td><td>12.5, 39</td><td>88,90</td><td>2/2</td><td>0/2</td></tr><tr><td>R01</td><td>Suffix</td><td>6</td><td>22.5, 100</td><td>7.5,100</td><td>1/2</td><td>1/2</td></tr></table>

Table 7: All 48 solver results. Two values per reward cell correspond to the frozen first and second run. R02 is the five-move adaptation; R0176 is the original forty-move task. Their results must not be interchanged.

No no-code run produces a valid exact-length proof game. Code produces two, on P0283 and P0295. Sufix tasks account for most partial reward. Both code runs on R0178 recover the required 36-ply ending but earn 88% and 90% for incomplete proofs. Neither condition completes R0171’s two-answer task, although some responses identify part of its capture-promotion mechanism.

These task-family comparisons do not estimate an intrinsic cost of universal reasoning: the positions, move budgets, mechanisms, and grading systems difer. The narrower observation holds within individual responses: under the archived review, four code-enabled sufix answers identify the correct ending, but only one supplies a complete proof. The proposed supplied-sufix diagnostic in Section 9 would test justification on the same positions with answer discovery removed.

## 7.4 Sensitivity and interpretation

The R0178 correction changes the code mean from 26.96% to 27.19%, a 0.23-point increase. To reduce double weighting of the R02/R0176 position, we average their task means and give each of the resulting eleven families equal weight. The means become 12.52% without code and 26.33% with code. Dropping R02 gives 10.82% and 26.80%; dropping R0176 gives 14.23% and 25.86%. The aggregate contrast survives each adjustment.

Omitting any one task gives a code-minus-no-code diference between 7.82 and 15.98 percentage points. This is a sensitivity range, not a confidence interval. Two runs per task, shared source families, and selection on observed performance prevent a precise population interpretation.

Reward design has a larger efect than the grading correction. Substituting diagnostic rubric credit for binary construction reward raises the means to 20.06% and 32.88%. That measures a diferent objective. We retain binary reward for construction because the requested deliverable is a valid exact-length witness, and keep diagnostic credit available for studying partial progress.

## 8 Failure mechanisms

Saved answers make the failures inspectable. Each case below identifies the disputed claim and the move or history that defeats it. The analysis concerns observable arguments; it does not infer hidden model states or estimate the frequency of each mechanism outside this set.

## 8.1 R01: applying king safety to the wrong time step

The six-move ending is

$$
\begin{array} { r l } { { } } & { { \mathrm { K b 6 \times B b 7 , ~ \quad ~ c 7 - c 8 = B ^ { + } , ~ \quad ~ K b 7 - b 8 ^ { + } , } } } \\ { { } } & { { \mathrm { B c 8 - b 7 ^ { + } , ~ \quad ~ K b 8 \times B b 7 , ~ \quad ~ a 5 - a 6 ^ { \# } . } } } \end{array}
$$

Code run 1 rejects b8 as the origin of Black’s penultimate king move because White’s queen attacks it. The king can legally start its turn in check, then capture the bishop on b7 to escape. The rule concerns safety after the move. Applying it before the move leads to a false impossibility claim. The response earns 7.5% for a sound preliminary pawn-provenance argument.

No-code run 1 makes the same kind of error earlier. Black vacates b7, opening the bishop’s a8–b7–c6–d5 diagonal against White’s king. White then plays Bc8-b7, blocking that check and uncovering a queen check against Black. The response declares the intermediate check impossible instead of testing White’s reply. Its capture accounting earns 22.5%; its no-history conclusion is false.

The reference required a related repair. To exclude a knight as the earliest b7 victim, it is insuficient to observe that White was in check before a hypothetical Nc5-b7: that move could block the check. The corrected proof examines whether Black’s preceding move could have created it. The grader must enforce the same temporal standard on the reference and the submission.

## 8.2 R02: a temporary lock mistaken for a permanent constraint

The required five moves are

$$
\mathrm { B c 6 - a 4 , } \quad \mathrm { b 3 - b 2 , } \quad \mathrm { B a 4 - c 2 , } \quad \mathrm { b 2 - b 1 = B , } \quad \mathrm { R f 2 \times N f 3 ^ { + } } .
$$

Code run 2 derives useful capture and promotion constraints, including the final promotion/capture pair. It then declares the central group locked until an earlier pawn capture is undone, missing a release through a knight-screen exchange. It also fixes capture squares from the white pawns’ final ranks without establishing their ranks at the time. These assumptions create a false shortage of turns. The answer concludes that the position is impossible and earns 47% for its valid deductions.

The release moves the bishop away from c2, routes the a6 knight backward through b4 and c2 to e1, and coordinates the restored f3 knight. Each step depends on squares becoming available at the right time. A shortest geometric knight route is only a candidate; it must fit the alternating turns. Those deadlines force the early bishop moves in the answer.

## 8.3 P0292: a near witness does not establish a lower bound

Both code runs reach the board legally in 47 plies; the task requires 45. Run 1 claims that 47 is minimal because a bishop cannot travel from e8 to h5 through the pawn on g6. The reference sends the newly promoted bishop through g6 before g7-g6, then plays g6×Bh5. The supposed obstruction exists only after the route has been used.

Replay confirms the proposed 47-ply games. It cannot certify the claimed lower bound. The 45-ply reference refutes that claim directly and is printed in Appendix D. Any valid 45-ply alternative would earn full credit; the submitted games fail the length requirement.

## 8.4 R0171 and R0178: a final answer is not a proof certificate

R0171’s answers are b7×Rc8=R# and b7×Nc8=R#. Writing b7×Xc8=R# identifies the mechanism but leaves the victim identities and completeness unresolved. The reference supplies legal histories for both the rook and knight branches and excludes the other victims. The two-move answer set carries both existence and exclusion obligations.

Both code runs on R0178 give the correct ending and omit parts of its uniqueness proof. One reaches the 90% incompleteness cap. Retaining the final-answer check alongside the reward makes this distinction visible.

## 9 Validation priorities and follow-up tests

The archived responses expose failures of temporal consistency and historical exclusion. Establishing their cause requires validated references and controlled comparisons; the present pilot supplies neither a causal tool efect nor a transfer estimate.

Complete witness coverage and assess the grader. The remaining reference-evidence work is to supply a legal start-to-diagram witness for each sufix position (Table 15). A separate blinded evaluation of the grader should assess solver answers, valid alternative proofs, and minimally corrupted proofs without model or condition labels. Report criterion-level disagreement, false acceptance, false rejection, and adjudication; retain unresolved cases as such. Full-score reference controls establish calibration and cannot replace these checks.

Separate discovery from justification. The most direct diagnostic compares free solving with a condition that supplies the correct sufix and asks whether it is forced. This holds the position fixed and removes answer discovery; a separate incorrect-candidate arm tests automatic endorsement. Persistent exclusion errors with the true ending supplied would locate a failure beyond discovery. Complement this with unrelated local-rule probes and blinded proof-verification trials. Score final answers and proof validity separately, and keep assisted results outside the unassisted benchmark score.

Test explanatory coverage. Compare an unadapted baseline with adaptation on final-answer sequences or fully worked proofs for the same training tasks. Match total training tokens and compute; test fresh composition families and new combinations of mechanisms. Measure answer recovery and proof completion separately. Gains from worked proofs would establish their usefulness, not their absence from the original training corpus.

Test mechanism-specific predictions. Appendix F specifies the interventions and safeguards for state tracking, search, and compositional transfer. A transfer claim requires demonstrated source competence, a defined target shift, and improvement from a targeted intervention that does not reveal the answer. Architectural explanations additionally require controlled comparisons with comparable training and inference budgets; improvement from an external state ledger alone would not isolate architecture. Freeze fresh composition families and primary outcomes before evaluation. These are proposed experiments, not results of this study.

## 10 Limitations and Reproducibility

Selection and scale. The 12 tasks were selected using the model results reported here, including no-code screening trials retained in the comparison. This asymmetry can favor code without a causal tool benefit. Composers, mechanisms, and the repeated R02/R0176 position create dependencies. Two runs per condition and unmatched compute support descriptive scores, not population success rates or a causal estimate of code’s benefit. A broader evaluation needs tasks fixed in advance, more runs, multiple model families, and matched budgets.

Ground truth and grading. Complete constructive witnesses can be replayed. The author manually reviewed the solutions, including the analytical uniqueness and exclusion arguments, on a physical chessboard. These arguments are presented as natural-language proofs rather than formal certificates. The four sufix references additionally lack full reachability witnesses in this bundle; their arguments are conditional on the prompt premise. Solver and judge may share errors. Reference controls calibrate the rubric; they do not measure independent judge accuracy. The recorded reference repair and score correction show why versioned evidence and review matter.

Exposure and adaptation. Source problems and answers are public. We have measured neither the volume of public worked explanations nor Astra’s exposure to them. Changing a rook square or sufix length cannot guarantee that a task is absent from training. Future splits should keep shared position families together and include newly composed problems with documented creation dates. We also lack a matched mathematics evaluation of this deployment. The results do not identify memorization, an architectural limit, or a general-intelligence boundary as the cause of failure.

RL environment and replication. Researchers can download the 12-task benchmark and tested RL environment from https://app.poindexterlabs.net/docs/retro-chess or PDL-Bench on Hugging Face, under retro\_chess. The Docker ZIP, also mirrored on Hugging Face, contains build files, all 12 tasks, board images, reference solutions, rubrics, and the grading runtime. Follow the setup instructions to build the container, run ofline checks, export solver inputs, and grade new model trajectories with provider credentials. The paper bundle’s analysis scripts reproduce the reported tables and aggregate scores from archived judgments. Appendix B specifies the experimental settings; the validation report records the Docker and grading checks. Replication should record the downloaded release hash and actual grading provider.

Provenance and attribution. The manuscript bundle preserves the prompts, diagrams, references, rubrics, answer evidence, scoring scripts, and analyses used here; the candidate pool records selection provenance. Composition attribution and applicable third-party rights must accompany redistribution. The bundle does not grant a blanket license to the original compositions.

## 11 Conclusion

RetroChessBench tests three obligations: constructing a legal history, proving a forced ending, and identifying every possible answer. On this selected challenge set, Astra High gives correct final answers in 2/24 attempts without code and 6/24 with code. Mean terminal rewards from the RL environment are 14.75% and 27.19%, respectively. The benchmark remains far from saturated in both solver presets, even with code execution.

We expect the three hardest tasks to resist reliable solution for some time, while the others approach saturation as models improve. Progress requires both better answers and complete proofs: only one of the four correct code-enabled sufix answers receives full proof credit under the archived review.

The model rules out legal histories for the wrong reasons. It rejects a king for starting its turn in check even though it can escape legally, excludes a route because a square is occupied later, and uses a valid construction to claim a false lower bound. A restriction on one moment or candidate does not establish a claim about every legal history.

Limited exposure to worked retrograde proofs is a leading hypothesis to test. Generality requires evaluating how eficiently a model learns from such explanations and transfers the method to new compositions. Matched tests must distinguish missing domain instruction from failures to combine familiar rules. Claims of artificial general intelligence require evidence of reliable reasoning across new, independently validated problems, beyond saturation of a fixed public set. The standard is a legal construction when one history sufices, and rigorous exclusion when every history matters.

Acknowledgments. We thank the original composers and the maintainers of the Retrograde Analysis Corner.

## References

Mislav Balunović, Jasper Dekoninck, Nikola Jovanović, Ivo Petrov, and Martin Vechev. MathConstruct: Challenging LLM reasoning with constructive proofs. arXiv:2502.10197v2, 2025a. URL https://arxiv.org/abs/2502.1 0197v2.

Mislav Balunović, Jasper Dekoninck, Ivo Petrov, Nikola Jovanović, and Martin Vechev. MathArena: Evaluating LLMs on uncontaminated math competitions. arXiv:2505.23281v3, 2025b. URL https://arxiv.org/abs/2505.23281v3. Version 3, 14 January 2026.

Chandra Bhagavatula, Ronan Le Bras, Chaitanya Malaviya, Keisuke Sakaguchi, Ari Holtzman, Hannah Rashkin, Doug Downey, Scott Wen-tau Yih, and Yejin Choi. Abductive commonsense reasoning. In International Conference on Learning Representations, 2020. URL https://arxiv.org/abs/1908.05739v2.

Josh Brunner, Erik D. Demaine, Dylan Hendrickson, and Julian Wellman. Complexity of retrograde and helpmate chess problems: Even cooperative chess is hard. In 31st International Symposium on Algorithms and Computation (ISAAC 2020), volume 181 of Leibniz International Proceedings in Informatics, pages 17:1–17:14. Schloss Dagstuhl– Leibniz-Zentrum für Informatik, 2020. doi: 10.4230/LIPIcs.ISAAC.2020.17. URL https://drops.dagstuhl.d e/entities/document/10.4230/LIPIcs.ISAAC.2020.17.

ChessBench. GPT 6 Astra: Elo skyline, 2026a. URL https://chessbench.ai/skyline/elo/gpt-6-astra. Snapshot generated 27 October 2026, 01:25:43 UTC; accessed 27 October 2026. Displayed family score 1745; XHigh 1745, High 1531.

ChessBench. About chessbench: Methodology and rating calibration, 2026b. URL https://chessbench.ai/about. Accessed 27 October 2026.

Dominika Agnieszka Długosz, Arlindo Oliveira, and Natalia Díaz-Rodríguez. The importance of being statistically earnest: A critical re-evaluation of GSM-Symbolic. arXiv:2605.28700v3, 2026. URL https://arxiv.org/abs/ 2605.28700v3. Version 3, 1 September 2026; accepted to EMNLP 2026.

Niklas Fiekas. python-chess: A chess library for python. Software documentation. URL https://python-chess.r eadthedocs.io/en/latest/. Accessed 27 October 2026.

Romain Harang, Jason Naradowsky, Yaswitha Gujju, and Yusuke Miyao. Tracking world states with language models: State-based evaluation using chess. arXiv:2508.19851, 2025. URL https://arxiv.org/abs/2508.19851.

Sai Kolasani, Maxim Saplin, Nicholas Crispino, Kyle Montgomery, Jared Quincy Davis, Matei Zaharia, Chi Wang, and Chenguang Wang. LLM CHESS: Benchmarking reasoning and instruction-following in LLMs through chess. arXiv:2512.01992, 2025. URL https://arxiv.org/abs/2512.01992.

François Labelle. Jacobi: Fairy chess helpgame solver. Software and documentation. URL https://wismuth.com/ jacobi/. Includes orthodox proof-game solving; accessed 27 October 2026.

Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, and Mehrdad Farajtabar. GSM-Symbolic: Understanding the limitations of mathematical reasoning in large language models. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2410.05229v2.

Sumeet Ramesh Motwani, Daniel Nichols, Charles London, Peggy Li, Fabio Pizzati, Acer Blake, Hasan Abed Al Kader Hammoud, Tavish Malcolm Mcdonald, Akshat Naik, Alesia Ivanova, Vignesh Baskaran, Ivan Laptev, Ruben Glatt, Tal Ben-Nun, Philip Torr, Ameya Prabhu, Brian R. Bartoldson, Bhavya Kailkhura, and Christian Schroeder De Witt. LongCoT: Benchmarking long-horizon chain-of-thought reasoning. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 90545–90579. PMLR, 2026. URL https://proceedings.mlr.press/v306/motwani26a.html.

OpenAI. GPT-6 Astra: A new generation ofintelligence, 2026a. URL https://openai.com/index/gpt-6-astra/. Accessed 28 October 2026.

OpenAI. Mathematical manuscripts and supporting proof artifacts, 2026b. URL https://github.com/openai/math. Research collection; accessed 28 October 2026.

OpenAI. Sharing AI progress in mathematics, 2026c. URL https://openai.com/index/sharing-ai-progres s-in-mathematics/. 6 October 2026; accessed 28 October 2026.

OpenAI. On the Navier–Stokes millennium prize problem, 2026d. URL https://openai.com/index/navier-s tokes-solution/. 8 September 2026; accessed 28 October 2026.

Ivo Petrov, Jasper Dekoninck, Lyuben Baltadzhiev, Maria Drencheva, Kristian Minchev, Mislav Balunović, Nikola Jovanović, and Martin Vechev. Proof or bluf? evaluating LLMs on 2025 USA math olympiad. arXiv:2503.21934v5, 2025. URL https://arxiv.org/abs/2503.21934v5.

Leonard S. Pleiss, Maximilian Schifer, and Robert K. von Weizsaecker. Disentangling generalization and memorization in large language models using chess. arXiv:2601.16823v3, 2026. URL https://arxiv.org/abs/2601.16823v3. Version 3, 20 October 2026.

Allen Z. Ren, Brian Ichter, and Anirudha Majumdar. Thinking forward and backward: Efective backward planning with large language models. arXiv:2411.01790, 2024. URL https://arxiv.org/abs/2411.01790.

Abulhair Saparov and He He. Language models are greedy reasoners: A systematic formal analysis of chain-of-thought. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2210.01240v4.

Abulhair Saparov, Richard Yuanzhe Pang, Vishakh Padmakumar, Nitish Joshi, Seyed Mehran Kazemi, Najoung Kim, and He He. Testing the general deductive reasoning capacity of large language models using OOD examples. In Advances in Neural Information Processing Systems, 2023. URL https://arxiv.org/abs/2305.15269v3.

Zayne Sprague, Kaj Bostrom, Swarat Chaudhuri, and Greg Durrett. Natural language deduction with incomplete information. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 8230–8258. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.emnlp-main.564. URL https://aclanthology.org/2022.emnlp-main.564/.

Stockfish developers. UCI protocol and stockfish commands. Stockfish documentation. URL https://offici al-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html. Accessed 27 October 2026.

Oyvind Tafjord, Bhavana Dalvi, and Peter Clark. ProofWriter: Generating implications, proofs, and abductive statements over natural language. In Findings ofthe Associationfor Computational Linguistics: ACL-IJCNLP 2021, pages 3621–3634. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021.findings-acl.317. URL https://aclanthology.org/2021.findings-acl.317/.

The Retrograde Analysis Corner. StrateGems: Original retro problems and solutions, issues 52–58. Archive hosted by Angela and Otto Janko. URL https://www.janko.at/Retros/Strategems/index.htm. Original issues 2010–2012; accessed 27 October 2026. Individual composers and problem identifiers are credited in the task inventory.

Shubham Toshniwal, Sam Wiseman, Karen Livescu, and Kevin Gimpel. Chess as a testbed for language model state tracking. Proceedings of the AAAI Conference on Artificial Intelligence, 36(10):11385–11393, 2022. doi: 10.1609/aaai.v36i10.21390. URL https://ojs.aaai.org/index.php/AAAI/article/view/21390.

Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. PlanBench: An extensible benchmark for evaluating large language models on planning and reasoning about change. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2023. URL https: //arxiv.org/abs/2206.10498v4.

Thomas Volet. An introduction to retroanalytic inference in chess composition. The Retrograde Analysis Corner, 2003. URL https://www.janko.at/Retros/Articles/Volet.htm. Accessed 28 October 2026.

Anjiang Wei, Yuheng Wu, Yingjia Wan, Tarun Suresh, Huanmi Tan, Zhanke Zhou, Sanmi Koyejo, Ke Wang, and Alex Aiken. SATBench: Benchmarking LLMs’ logical reasoning via automated puzzle generation from SAT formulas. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 33832–33849. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.emnlp-main.1716. URL https://aclanthology.org/2025.emnlp-main.1716/.

Zhaofeng Wu, Linlu Qiu, Alexis Ross, Ekin Akyürek, Boyuan Chen, Bailin Wang, Najoung Kim, Jacob Andreas, and Yoon Kim. Reasoning or reciting? exploring the capabilities and limitations of language models through counterfactual tasks. In Proceedings of NAACL-HLT (Volume 1: Long Papers), pages 1819–1862. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.naacl-long.102. URL https://aclanthology.org/2 024.naacl-long.102/.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. NeurIPS 2023 Datasets and Benchmarks Track; arXiv:2306.05685, 2023. URL https://arxiv.org/abs/2306.05685.

## A Task inventory and dataset card

<table><tr><td>Task</td><td>Composer</td><td>Issue</td><td>Type</td><td>Plies</td></tr><tr><td>R0171</td><td>Nikolai I. Beluhov &amp; Andrey Frolkin</td><td>52</td><td>Set</td><td>1</td></tr><tr><td>P0292</td><td>Silvio Baier</td><td>53</td><td>Game</td><td>45</td></tr><tr><td>P0297v</td><td>Nicolas Dupont</td><td>55</td><td>Game</td><td>60</td></tr><tr><td>P0332</td><td>Silvio Baier</td><td>58</td><td>Game</td><td>44</td></tr><tr><td>P0333</td><td>Gianni Donati &amp; Olli Heimo</td><td>58</td><td>Game</td><td>44</td></tr><tr><td>P0335</td><td>Unto Heinonen</td><td>58</td><td>Game</td><td>54</td></tr><tr><td>R02</td><td>Nikolai I. Beluhov</td><td>53</td><td>Suffix</td><td>5</td></tr><tr><td>P0283</td><td>Paul Monsky</td><td>52</td><td>Game</td><td>35</td></tr><tr><td>P0295</td><td>Gianni Donati &amp; Olli Heimo</td><td>53</td><td>Game</td><td>46</td></tr><tr><td>R0176</td><td>Nikolai I. Beluhov</td><td>53</td><td>Suffix</td><td>40</td></tr><tr><td>R0178</td><td>Nikolai I. Beluhov</td><td>54</td><td>Suffix</td><td>36</td></tr><tr><td>R01</td><td>Andrey Frolkin</td><td>56</td><td>Suffix</td><td>6</td></tr></table>

Table 8: Frozen RetroChessBench inventory. Issue numbers refer to StrateGems. “Game” requests an exact-length constructive witness; “Sufix” requests a unique ending; “Set” requests both possible last moves. R01 adapts R0185 and R02 adapts R0176.

Intended use. The benchmark supports final-response evaluation and terminal-reward experiments on reconstruction, analytical proofs, and tool use. It is a selected challenge set. Training on its references or rewards must be disclosed when reporting results on these tasks.

Unit of analysis. A task consists of a diagram, stipulation, output requirement, and reference version. Changing the sufix length creates a diferent task from the same source problem. R02 and R0176 therefore belong in one family for training/test splits. R01 also derives from an existing composition, with a changed rook square and shorter ending.

Coverage. The audit covers 82 primary entries in seven issues. The selection ledger separately records variants and parts. Initial no-code exploration produced 116 completed answers on 58 tasks: 77 final-answer failures and 39 accepted answers. Filtering on these results and adding R01/R02 yields the 48-task candidate pool. Some code responses in that pool remain unreviewed or technically unresolved. RetroChessBench contains the 12 tasks retained for complete grading and review.

<table><tr><td>Stage</td><td>Count</td><td>Retained evidence / decision</td></tr><tr><td>Primary source entries</td><td>82</td><td>Seven issues; variants and parts recorded separately in the coverage audit.</td></tr><tr><td>Eligible task instances</td><td>58</td><td>Two no-code trials each: 77 failures and 39 accepted final answers across 116 trials.</td></tr><tr><td>At least one no-code failure</td><td>47</td><td>30 tasks failed twice; 17 failed once. These observed outcomes screened the candidate pool.</td></tr><tr><td>Candidate pool after exclusions / additions</td><td>48</td><td>Remove P0299&#x27;s unresolved proof; retain 46 instances and add R01/R02.</td></tr><tr><td>RetroChessBench</td><td>12</td><td>Final IDs in Table 8; adaptive choice using observed difficulty and proof coverage. Two retained attempts per condition.</td></tr></table>

Table 9: Selection ledger. Counts of source entries and eligible task instances use diferent units because parts and variants are distinct tasks. The archive preserves stage snapshots and attempt hashes; it does not reconstruct a preregistered rule or every intermediate editorial choice.

Selection provenance. The coverage audit and exploration\_selection\_final.json record sourcelevel inclusion decisions. exploration\_summary.json records the 58-task screen. selection.json preserves an earlier snapshot with answer hashes; frozen\_selection.json, current task references, and grading.json govern the paper. Earlier evidence strings can predate the R0171 reference repair and must not override the current version. Table 9 records the verifiable stages without treating an adaptive process as an independent test split. The 22/24 and 18/24 failure counts use archived comparison trials, not new post-selection reruns.

Reference scope. Constructive references supply replayable games and explanations of their mechanisms. Sufix references prove the ending under the prompt’s legal-reachability premise. R0171 includes full histories of 167 and 189 plies for its two branches; these establish feasibility without a minimum-length claim. Analytical exclusions remain natural-language proofs.

Version and packaging. The paper analysis pins the 12-task selection recorded on 27 October 2026 and the subsequently synchronized reference/grade bundle. The environment build-source ZIP archived with this manuscript has SHA-256

## 1ec140391c6df9e44825e65e4f43ed51b2d988714f761e8715ae71402beae63e

and is 646,079 bytes. It contains the runtime, tasks, images, references, and rubrics, with no model weights or API credentials. New model-based grades in this archived runtime require Azure access. Recomputing scores from saved judgments is ofline. The current public release, mirrored in PDL-Bench on Hugging Face, has been built and tested in Docker, as documented in its validation report; its 48 frozen task, reference, rubric, provenance, and image files match the archived release byte for byte. The public ZIP has its own release hash and must not be identified by the archived hash above.

## B Exact solver protocol and replication

The system prompt is the following two-sentence format instruction:

Return your response as a JSON object with one string field named solution. Follow the task’s instructions within that field.

The strict response schema requires one string field, solution, and permits no others. The user message contains the task text and one base64-encoded board image with detail=high. Content hashes verify the four inputs for each task; R01’s raw text difers only in newline formatting. The system prompt adds no task identifier, reference, or rubric.

Replication files. Start from RetroChessBench on Poindexter Research Docs or the retro\_chess subset of PDL-Bench on Hugging Face. Download the tested RL environment Docker bundle from the website or its identical Hugging Face ZIP. The website setup instructions and Hugging Face quick start cover building the image, running ofline checks, and exporting prompts and board images for a solver. Release validation covers native ARM64 and emulated AMD64 Docker builds and runs, all 12 reference-sequence checks, 60 regression tests, and live grading checks (validation report). Keep reference answers and rubrics in the evaluator. New model-based grading needs provider credentials; pin the Azure route when reproducing the study’s provider condition. The newer runtime also supports an OpenAI route and fallback, whose results have separate provenance.

The bundle includes the exact prompts at data/prompts/<task>/<mode>/run\_1.txt and run\_2.txt, final answers, sanitized metadata, task configurations, and grades. Full API responses remain in the original archive; the bundle retains their hashes and extracted usage/tool metadata. Credentials and private reasoning traces are excluded.

<table><tr><td>Setting</td><td>No code</td><td>Code enabled</td></tr><tr><td>Requested model</td><td>gpt-6-astra</td><td>gpt-6-astra</td></tr><tr><td>Reasoning effort</td><td>high</td><td>high</td></tr><tr><td>Tools</td><td>None</td><td>Hosted code interpreter, auto container, 4 GB</td></tr><tr><td>Web-search tool</td><td>Absent</td><td>Absent</td></tr><tr><td>Input</td><td>Prompt, FEN, board image</td><td>Identical prompt, FEN, board image</td></tr><tr><td>Reference / rubric access</td><td>None</td><td>None</td></tr><tr><td>Explicit token limit / temperature / seed</td><td>Not set</td><td>Not set</td></tr><tr><td>Response mode</td><td>Background, strict solution JSON</td><td>Background, strict solution JSON</td></tr><tr><td>Runs per task</td><td>2 42,295</td><td>2 58,443</td></tr><tr><td>Median output tokens</td><td></td><td>6,425–81,512</td></tr><tr><td>Output-token range Recorded code calls</td><td>21,199–69,369 0</td><td>219, across all 24 runs</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Recorded tool console output</td><td>Not applicable</td><td>Not retained in provider response</td></tr></table>

Table 10: Archived request settings and observed usage. Token counts include provider-reported reasoning tokens and are not identical to visible answer length. No claim of equal computation is made.

analysis/analyze.py regenerates the attempt CSV, summary statistics, plots, and counterexample table using local data. It makes no model calls and never executes solver code. The collection script records evidence provenance. Original workspace paths are archival metadata, not build dependencies.

Boundaries of the tool audit. All 219 code calls have completed status and saved payloads; none retains console output. Some attempt a chess-library import or engine lookup, which does not establish availability. A replication should preinstall dependencies, retain outputs, and record network permissions alongside tool settings.

## C Concrete analytical rubrics

Tables 11 and 12 summarize the proof obligations. The frozen JSON specifies full, half, and zero credit; it and the deterministic sequence policy govern scoring. A sound equivalent proof is accepted without requiring the reference’s exposition or unused lemmas.

<table><tr><td>Item</td><td>Weight</td><td>Necessary result or proof obligation</td></tr><tr><td>C1</td><td>25</td><td>Correct six-ply ending, including both bishop victims and the check/mate annotations.</td></tr><tr><td>C2</td><td>15</td><td>White's seven-capture exhaustion; noncapturing surviving a-, b-, d-pawns; Black's five-capture budget, including two c-pawn captures and at least one b-pawn capture.</td></tr><tr><td>C3</td><td>10</td><td>Force a5-a6 and the latest black king's origin on b8, excluding the competing origins with temporally valid arguments.</td></tr><tr><td>C4</td><td>10</td><td>Restore the captured bishop on b7 and force its preceding Bc8-b7 move, using the opened bishop diagonal and queen line.</td></tr><tr><td>C5</td><td>10</td><td>Force Kb7-b8+ and the preceding c7-c8=B+ promotion, including the bishop's unavailable ordinary approaches.</td></tr><tr><td>C6</td><td>5</td><td>Exhaust Black's capture budget to exclude an additional capture on b8.</td></tr><tr><td>C7</td><td>8</td><td>Force b6 as the origin of the earliest king move to b7.</td></tr><tr><td>C8</td><td>12</td><td>Identify the earliest b7 victim as a bishop; exclude the knight through the legality of the earlier checking move, not merely the presence of a check.</td></tr><tr><td>C9</td><td>5</td><td>Verify alternating check responses, the terminal diagram, and mate.</td></tr><tr><td>C1</td><td>22</td><td>Correct five-ply ending, including b1 bishop promotion and the knight captured on f3.</td></tr><tr><td>C2</td><td>8</td><td>Force the rook's final arrival from f2 and the capture on f3 from the local check geometry.</td></tr><tr><td>C3</td><td>16</td><td>Derive the capture budgets, pawn origins, and required promotion constraints.</td></tr><tr><td>C4</td><td>14</td><td>Force Black's immediately preceding b2-b1=B and identify the final captured piece as a knight.</td></tr><tr><td>C5</td><td>10</td><td>Establish the temporary lock and the limit imposed by alternating turns and the b-pawn's available retractions.</td></tr><tr><td>C6</td><td>6</td><td>Exclude an earlier release through restoring and retracting Black's captured a-pawn.</td></tr><tr><td>C7</td><td>8</td><td>Exclude the tempting h8-unpromotion route using its complete timing cost, including subsequent pawn withdrawal.</td></tr><tr><td>C8</td><td>8</td><td>Establish the viable knight-release timing while accounting for occupied squares and interleaving.</td></tr><tr><td>C9</td><td>8</td><td>Apply the bishop's historical deadline to force the required bishop-move order.</td></tr></table>

Table 11: R01 rubric, 100 total points. C2 requires suficient capture bounds; it does not demand the stronger, unnecessary claim that Black’s b-pawn promoted if an equivalent bound is proved.

Table 12: R02 rubric, 100 total points. The first two final moves depend on the later historical release argument; identifying only the promotion/capture pair is insuficient.

Sequence partial credit. The deterministic R01/R02 policy awards half of C1 only for the explicitly specified near-complete cases, including at least four correct moves in their chronological slots; R02 additionally permits a specified single ordering repair. It does not convert a diferent final conclusion into the expected sequence. Missing identity or annotation details may prevent full notation credit even where the underlying moves agree. The full JSON policy distinguishes these cases.

Judge input and feedback. The judge receives the task, reference proof, expected answer or set, rubric, and submission as separate inputs. Credited reasoning requires evidence in the submission. The environment checks the quoted evidence and computes scores locally. Anchoring verifies that the text exists; judging its logic remains a separate responsibility. Instructions embedded in a submission must be treated as answer data.

## D A replayable counterexample to a false minimum

P0292, by Silvio Baier, asks for a proof game of exactly 45 plies. The following reference reaches its diagram in that length. It is printed in standard algebraic notation for compactness; the dataset also stores the long-algebraic final answer and replayed UCI moves. No uniqueness or shortest-game theorem is needed: this witness alone refutes a claim that at least 47 plies are necessary.

<table><tr><td colspan="3">Moves 1–12</td><td colspan="3">Moves 13–23</td></tr><tr><td>Move</td><td>White</td><td>Black</td><td>Move</td><td>White</td><td>Black</td></tr><tr><td>1</td><td>f4</td><td>c5</td><td>13</td><td>Ra4</td><td>b5</td></tr><tr><td>2</td><td>f5</td><td>c4</td><td>14</td><td>e4</td><td>bxa4</td></tr><tr><td>3</td><td>f6</td><td>c3</td><td>15</td><td>Bb5</td><td>f2+</td></tr><tr><td>4</td><td>fxe7</td><td>f5</td><td>16</td><td>Ke2</td><td>f1=B+</td></tr><tr><td>5</td><td>g4</td><td>Kf7</td><td>17</td><td>Kf2</td><td>Bc4</td></tr><tr><td>6</td><td>e8=B+</td><td>Ke6</td><td>18</td><td>b4</td><td>Bb3</td></tr><tr><td>7</td><td>Bh5</td><td>g6</td><td>19</td><td>cxb3</td><td>c2</td></tr><tr><td>8</td><td>g5</td><td>gxh5</td><td>20</td><td>Bb2</td><td>c1=R</td></tr><tr><td>9</td><td>g6</td><td>Bd6</td><td>21</td><td>Qc2</td><td>Rel</td></tr><tr><td>10</td><td>g7</td><td>Ne7</td><td>22</td><td>Nc3</td><td>Re3</td></tr><tr><td>11</td><td>g8=R</td><td>f4</td><td>23</td><td>dxe3</td><td>一</td></tr><tr><td>12</td><td>Rg4</td><td>f3</td><td></td><td></td><td></td></tr></table>

Table 13: A valid 45-ply construction for P0292. The crucial ordering is 6.e8=B+ Ke6 7.Bh5 g6 8.g5 gxh5. The bishop traverses g6 before Black occupies it. The table is generated from the independently replayed reference moves.

Four temporary promoted oficers are subsequently captured: a white bishop, a white rook, a black bishop, and a black rook. The resulting material therefore does not directly reveal all earlier promotions. In particular, the model’s lower bound fails by excluding a route under an occupancy assumption that the valid construction deliberately postpones.

## E Audit details and score interpretation

<table><tr><td>Constructive final-response outcome</td><td>No code</td><td>Code enabled</td></tr><tr><td>No complete game supplied</td><td>9</td><td>7</td></tr><tr><td>Legal target-reaching game, wrong length</td><td>3</td><td>5</td></tr><tr><td>Game contains an illegal move</td><td>2</td><td>0</td></tr><tr><td>Legal exact-length target-reaching game</td><td>0</td><td>2</td></tr><tr><td>Total</td><td>14</td><td>14</td></tr></table>

Table 14: Mutually exclusive outcomes of the 28 constructive attempts. Classification follows the frozen original-answer replay, with an illegal move taking precedence over a simultaneous length error. A missing witness can accompany useful reasoning or an incorrect impossibility argument; it is not necessarily an empty answer.

<table><tr><td>Reference family</td><td>Existence evidence in bundle</td><td>Analytical review / remaining evidence</td></tr><tr><td>Seven constructive tasks</td><td>Seven complete games replayed from the standard start.</td><td>No uniqueness claim is required.</td></tr><tr><td>R0171, two alternatives</td><td>Complete 167- and 189-ply histories, one for each victim, independently replayed.</td><td>The author reviewed the exclusion of other alternatives on a physical chessboard.</td></tr><tr><td>R01, R02, R0176, R0178</td><td>Local ending checks and conditional analytical references; no complete start-to-diagram witness supplied.</td><td>The author reviewed the conditional uniqueness arguments on a physical chessboard; complete reachability witnesses remain to be supplied.</td></tr></table>

Table 15: Existence and uniqueness are separate validation obligations. Missing full witnesses do not demonstrate that a diagram is unreachable; they identify evidence still required from the benchmark author. R02 and R0176 share their diagram.

Proof-status derivation. analysis/reviewer\_audit.py maps the 20 archived analytical grades and review comments to Table 4, retaining submission/reference hashes. It changes no rewards and requests no new judgments. The 28 constructive answers use the replay categories above: a valid witness sufices. Partial reasoning credit is not a proof-validity probability.

Notation and headline counts. analysis/notation\_audit.py checks all 48 retained verdicts against archived sequence status: accepted answers are exactly those marked verified. None of the 40 failures is notation-only. Two P0295 answers have a check/mate-marker defect, but both supply 48 plies instead of the required 46. Thirteen annotation-removal checks on reference strings using the archived runtime retain verified status; analytical sequence credit can fall from full to half. The audit records these checks without new model calls or changes to the saved scores. It does not validate global reachability or analytical proofs.

Why 27.5% diagnostic credit can coexist with zero reward. P0332 no-code run 1 reaches the target legally in 46 plies; the task requires 44. Route, check, and material deductions earn 27.5% diagnostic credit.

Terminal reward remains zero because the construction misses the move budget. The artifact preserves both values so that analysis of partial progress can use the diagnostic without changing the exact-task score.

Reference-control interpretation. All 12 current references score 100%. Full-game replay verifies the constructive controls. Analytical controls satisfy the calibrated rubric under the current judge and review. R01 includes the expanded earliest-knight exclusion; R0171 uses its completed v2 reference. Reproduction must use those versions.

One explicit correction. Review raised R0178 code run 1 from 82.5% to 88%: its tracked position and argument excluded competing c8 blockers, meeting the standard accepted in run 2. Other omissions prevent full credit. Run 2 totals 95.5% before the incomplete-proof cap reduces it to 90%. Raw judgments, caps, and overrides are stored separately.

What an independent replay checks. Replay starts from the standard position and checks UCI moves for legality, exact length, final placement, and premature automatic termination. All seven constructive controls and both successful model games pass. Replay can establish these properties of a supplied witness. Lower bounds and complete exclusions require additional arguments.

## F Controlled follow-up study

Following the validation priorities in Section 9, freeze fresh tasks and keep composition families within one split. Use the same model snapshot and explicit token/tool budgets across paired conditions. Predefine source competence, target shift, and primary outcomes; report uncertainty over families. The interventions below are proposed controls, not completed experiments.

<table><tr><td>Candidate explanation</td><td>Intervention and control</td><td>Discriminating outcome / safeguard</td></tr><tr><td>Missing rule knowledge</td><td>Probe check escape, promotions, capture budgets, and event order on unrelated positions; compare with a rules primer.</td><td>Local failure identifies prerequisite gaps. Do not reveal the target history.</td></tr><tr><td>Answer discovery</td><td>Compare free solving with correct-suffix assistance and a separate incorrect-candidate arm.</td><td>Proof failure despite a supplied answer extends beyond discovery. Keep assisted scores separate.</td></tr><tr><td>Proof verification</td><td>Compare valid proofs with minimal corruptions in blinded, randomized trials; request error location and repair.</td><td>Measure false acceptance and rejection; a binary label alone does not show the critical step was checked.</td></tr><tr><td>State tracking</td><td>Render boards and capture ledgers from the solver&#x27;s own candidates; use an equal-budget text control.</td><td>Improvement tests representation. Supplying true intermediate states leaks the answer.</td></tr><tr><td>Insufficient search</td><td>Add locally legal predecessor enumeration to the same legality checker.</td><td>More witnesses with residual exclusion errors separates search from proof. Global reachability filtering is oracle assistance.</td></tr><tr><td>Explanatory coverage / transfer</td><td>Compare no adaptation, answer-only examples, and worked proofs on the same training tasks; test disjoint families and new mechanism combinations.</td><td>Match token and compute budgets; establish source competence. Separate answer recovery from proof completion. Gains do not reveal original training</td></tr><tr><td>Quantifier confusion</td><td>Match feasible unique, multiple-answer, and impossible instances with certified labels.</td><td>exposure. Require witnesses or exclusions for the contract; match geometry and proof burden.</td></tr></table>

Tool records. Preinstall and smoke-test supplied libraries. Retain code, standard output/error, package versions, network policy, timeouts, and tool-call limits. Report provider-default use separately from matchedbudget comparisons; identify any specialist proof-game solver or global reachability oracle explicitly.