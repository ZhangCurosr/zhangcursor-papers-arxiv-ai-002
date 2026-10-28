# TCM-DiagCommit: A TCM-SD-Derived Benchmark for Evidence-Conditioned Syndrome Selection and Commitment

Erwei Hou<sup>\*</sup> and Yue Hu<sup>\*</sup> <sup>(B)</sup>

Beijing University of Posts and Telecommunications, Beijing, China {houerwei,huyue2023}@bupt.edu.cn

Abstract. Traditional Chinese Medicine (TCM) syndrome diferentiation is usually evaluated as static final-label selection over complete clinical records. This setting hides three diagnostic behaviors: whether a model distinguishes semantic-neighbor syndrome labels from easy random distractors, whether it updates predictions as evidence fields are revealed, and whether it should commit under uncertainty. We introduce TCM-DiagCommit, a benchmark derived from the public TCM-SD/ZY-BERT resource while retaining its source syndrome labels; it converts 1,834 source cases into 23,842 task rows over 148 syndrome labels. The benchmark combines random, profile-neighbor, and full-vocabulary candidate ladders with staged t1/t2/t3 evidence. It also includes fixedlabel tongue/pulse perturbations and insuficient-evidence variants. A benchmark-defined action contract scores commit, acquire, defer, and abstain. Across five zero-shot DirectLLM calibrations, random10 top-1 accuracy is 63.2% to 66.4%, but hard20 falls to 24.7% to 30.3% and full148 to 14.8% to 22.3%; on hard20, WT(all) wrong-terminal commits remain 62.2% to 74.3%. These results show that final-answer accuracy alone is insuficient for diagnostic benchmark evaluation: answer selection, evidence use, and benchmark commitment scoring should be measured separately.

Keywords: Traditional Chinese Medicine · Syndrome diferentiation · Staged evidence · Diagnostic commitment

## 1 Introduction

Medical QA and diagnosis benchmarks often cast diagnosis as complete-evidence final-answer selection. This formulation is easy to score, but it hides whether a model can revise an early hypothesis, move beyond semantic option elimination, or avoid committing when evidence is partial or conflicting. These hidden behaviors matter for evaluation even when the benchmark is not a clinical decision-support system.

TCM syndrome diferentiation is a natural testbed for this problem. A syndrome pattern synthesizes chief complaint, history, inspection, and tongue/pulse evidence into a fine-grained diagnostic abstraction. TCM-SD made this task measurable with 54,152 real-world clinical records covering 148 syndromes and introduced the ZYBERT domain model [10], but the standard setting still presents a complete record and asks for one final label. We keep these source labels as gold labels and change the evaluation interface around them.

TCM-DiagCommit evaluates evidence-conditioned diagnostic decisions instead. We build a source-case-linked evaluation layer from TCM-SD/ZY-BERT [10]: full-evidence candidate selection, t1/t2/t3 staged fields, random/profileneighbor/full-vocabulary candidate ladders, fixed-label tongue/pulse perturbation controls, shortcut checks, and insuficient-evidence commitment tasks. The benchmark asks not only “Which syndrome is most plausible?” but also “Did added evidence change the diagnosis?” and “Should the model commit now under the benchmark policy?”

We ask three research questions. RQ1: Candidate dificulty. Does hard20 expose semantic reranking pressure hidden by random10? RQ2: Evidence. Does staged evidence distinguish early guessing from evidence-corrected diagnosis? RQ3: Commitment. Do final-answer metrics miss commitment behavior under the same candidate set?

Our contributions follow four layers. Formulation: we recast TCM syndrome diferentiation as evidence-conditioned diagnostic commitment. Benchmark construction: we provide 1,834 source cases, 148 syndrome labels, 26 task files covering 13 paired dev/test variants, and 23,842 rows with staged views, candidate ladders, fixed-label local perturbations, and insuficient-evidence variants. Scoring protocol: we define top-1/top-3 diagnosis, stage gains, wrongterminal rates, expected cost, and source-case-level statistical reporting. Empirical calibration: five zero-shot DirectLLM runs reveal all three gaps, while a supervised ZY-BERT reference is used only to sanity-check benchmark learnability and dificulty calibration.

Resources are available at https://github.com/Ethan-ai637/TCM-DiagC ommit.

## 2 Related Work

Medical NLP evaluation spans broad NLU suites, exam-style QA, clinical-case reasoning, and agentic diagnosis. Chinese resources such as CBLUE, CMB, and CMExam emphasize biomedical understanding or licensing-exam questions [14, 12, 8]; they are valuable for knowledge and QA coverage, but they rarely expose the same source case under controlled evidence stages and candidate dificulty ladders. TCM-specific benchmarks such as TCMBench and MTCMB broaden TCM LLM evaluation beyond generic medical QA, covering knowledge and reasoning, with MTCMB further including prescription and safety-oriented tasks [13, 6]. TCM-DiagCommit instead provides a focused process-level evaluation within one 148-label syndrome ontology, measuring semantic-neighbor discrimination, evidence-conditioned updating, and terminal action reporting.

TCM-SD is the primary source, with 54,152 clinical records, 148 syndrome labels, and ZYBERT [10]. Existing TCM syndrome-diferentiation work includes supervised or knowledge-enhanced classification models such as dual-channel knowledge attention [7], as well as recent work evaluating and improving LLMs’ syndrome-diferentiation thinking ability [2]. These methods generally optimize final-label prediction or generated reasoning. Our benchmark keeps the finallabel task but adds protocol-level stressors around it: profile-neighbor candidates test semantic reranking, staged rows test field-conditioned updating, and commitment rows test action reporting under a fixed scorer.

Dynamic medical evaluation provides the second thread. Sequential diagnosis and agentic clinical benchmarks motivate additional-evidence and tool-use policies: Sequential Diagnosis evaluates stepwise diagnostic encounters [9], while AgentClinic evaluates simulated patient interaction, multimodal data collection, and tool use [11]. TCM-DiagCommit deliberately does not simulate open dialogue: it uses fixed field masking so every stage and action is paired across source cases. Selective classification formalizes risk-coverage trade-ofs, while conformal prediction provides coverage-guaranteed uncertainty sets [3, 1]; calibration work shows confidence can misalign with correctness [4]. Counterfactually augmented NLP data test shortcut reliance [5]; our fixed-gold tongue/pulse rows follow that stress-test spirit while avoiding claims that the perturbed row has a new clinical gold label. Table 1 summarizes the positioning across evaluation capabilities.

Table 1. Benchmark comparison by evaluation capabilities.
<table><tr><td>Benchmark</td><td>Staged</td><td>Hard</td><td>Insuff.</td><td>Actions</td><td>Perturb.</td></tr><tr><td>TCM-SD/ZY-BERT</td><td>No</td><td>No</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Chinese medical QA</td><td>No</td><td>No</td><td>Partial</td><td>No</td><td>No</td></tr><tr><td>TCM LLM benchmarks</td><td>Partial</td><td>No</td><td>Partial</td><td>No</td><td>Partial</td></tr><tr><td>Sequential/agentic diagnosis</td><td>Yes</td><td>Task-specific</td><td>Partial</td><td>Yes</td><td>Partial</td></tr><tr><td>TCM-DiagCommit</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr></table>

## 3 Benchmark Construction

## 3.1 Source and Release Boundary

The core source is TCM-SD/ZY-BERT [10]. We preserve the source syndrome labels from TCM-SD and change the evaluation interface around them. The benchmark release contains derived task rows, source-case links, and release metadata for this TCM-SD-derived evaluation layer. Auxiliary TCM QA, dialogue, or mixed-task resources are used only as related-work context. TCM-SD train records build only hard-candidate label profiles; held-out evaluation source cases are selected from the source dev/test splits.

Formally, a selected source case is $x _ { i } = ( f _ { i } ^ { ( 1 ) } , f _ { i } ^ { ( 2 ) } , f _ { i } ^ { ( 3 ) } , y _ { i } )$ , where $f _ { i } ^ { ( 1 ) }$ is chief complaint, $f _ { i } ^ { ( 2 ) }$ is description/history, $f _ { i } ^ { ( 3 ) }$ is detection/tongue-pulse evidence when present, and $y _ { i } \in \mathcal { V }$ with $| y | = 1 4 8$ . Stage views are $E _ { i } ^ { ( 1 ) } = f _ { i } ^ { ( 1 ) }$ $E _ { i } ^ { ( 2 ) } = [ f _ { i } ^ { ( 1 ) } ; f _ { i } ^ { ( 2 ) } ]$ , and $E _ { i } ^ { ( 3 ) } = [ f _ { i } ^ { ( 1 ) } ; f _ { i } ^ { ( 2 ) } ; f _ { i } ^ { ( 3 ) } ]$ . For candidate policy $\pi , C _ { i } ^ { \pi }$ always contains y<sub>i</sub>. A model output at stage s is $\dot { o } _ { i } ^ { ( s ) } = ( \hat { y } _ { i } ^ { ( s ) } , T _ { i } ^ { ( s ) } , c _ { i } ^ { ( s ) } , a _ { i } ^ { ( s ) } )$ : top-1, top-3, confidence, and action.

The release contains 919 dev cases and 915 test cases; both splits cover all 148 labels, with 2–10 selected cases per label per split. The cap keeps the paired variant matrix tractable while preserving label coverage and near-identical dev/test balance. Table 2 and Figure 1 summarize the scale: 26 task files and 23,842 task rows.

![](images/f028ea9056fff5388480e0f0e13230c0fa15c8dce1037d7a4dc059335cca438d.jpg)  
Fig. 1. Benchmark overview. A source case is shown through staged evidence and paired task variants; TCM-DiagCommit evaluates semantic-neighbor discrimination, evidence updating, and diagnostic commitment behavior with source-case-level scoring.

Construction recipe. The deterministic pipeline (1) normalizes dev/test rows, stratifies by syndrome, caps each label for balanced held-out coverage, removes exact and near-duplicate cross-split cases, and preserves source-case ids; (2) builds train-only label profiles from Chinese character n-gram vectors over disease and clinical fields; (3) ranks label neighbors by cosine similarity, $\quad \sin ( a , b ) =$ $p _ { a } ^ { \top } p _ { b } / ( \| p _ { a } \| _ { 2 } \| p _ { b } \| _ { 2 } )$ , then materializes random10, hard10, hard20, and full148 candidate sets with shufled gold slots; and (4) emits static, staged, shortcut, fixed-label tongue/pulse perturbation, and insuficient-evidence rows. Perturbation rows choose a same-split non-gold donor with extractable tongue/pulse evidence, replace the source tongue/pulse phrase, keep the source gold fixed, and inject the donor label into hard10 candidates. Thus “hard” denotes reproducible profile-neighbor alternatives selected by the same train-only similarity recipe: it creates semantic-neighbor ranking pressure rather than an extra source of labels. Complexity. Profile extraction is $O ( N _ { \mathrm { t r a i n } } L )$ , label-neighbor precomputation is $O ( | \mathcal { V } | ^ { 2 } d )$ , per-row candidate lookup is O(k), commitment scoring is O(NS) for $S \le 3$ , and bootstrap reporting is $O ( B N )$ for B source-case resamples.

Table 2. TCM-DiagCommit release scale. Each source case is transformed into paired task variants; dev and test share the same 148-label ontology but have disjoint cases. The statistical unit is source case, not task row.
<table><tr><td>Split</td><td>Source cases</td><td>Labels</td><td>Min/label</td><td>Max/label</td><td>Task files</td><td>Task rows</td></tr><tr><td>Dev</td><td>919</td><td>148</td><td>2</td><td>10</td><td>13</td><td>11,947</td></tr><tr><td>Test</td><td>915</td><td>148</td><td>2</td><td>10</td><td>13</td><td>11,895</td></tr><tr><td>Total</td><td>1,834</td><td>148</td><td>4</td><td>20</td><td>26</td><td>23,842</td></tr></table>

## 3.2 Task Variants

Table 3 summarizes the benchmark families. Each task file contains one row per selected source case (919 dev, 915 test). The candidate ladder is central: random10 is an easy closed-set control; hard10/hard20 use profile-neighbor distractors; and full148 evaluates the full vocabulary. Gold is always present, so failure is a diagnosis or ranking error rather than an open-vocabulary mismatch. We use G1 for semantic candidate discrimination, G2 for staged evidence integration, and G3 for commitment behavior.

Table 3. Task families in the public release. G1 = semantic candidate discrimination, G2 = staged evidence integration, G3 = benchmark commitment behavior.
<table><tr><td>Task family</td><td>Candidates</td><td>Evaluated behavior</td><td>Gap</td></tr><tr><td>Random10</td><td>10</td><td>Coarse option elimination.</td><td>G1</td></tr><tr><td>Hard10/20</td><td>10/20</td><td>Semantic profile-neighbor reranking.</td><td>G1</td></tr><tr><td>Full148</td><td>148</td><td>Full-ontology diagnosis.</td><td>G1</td></tr><tr><td>Staged t1/t2/t3</td><td>10-148</td><td>Evidence-conditioned</td><td>G2</td></tr><tr><td>Shortcut controls</td><td>10</td><td>updating. Candidate/evidence shortcut</td><td>G1</td></tr><tr><td>TP perturbation</td><td>10</td><td>sensitivity. Local cue sensitivity under</td><td>G1/G3</td></tr><tr><td>Insufficient t1/t2</td><td>10/20</td><td>fixed gold. Acquire/abstain under partial evidence.</td><td>G3</td></tr></table>

## 3.3 Progressive Evidence

TCM-DiagCommit uses a field-level staged evidence protocol designed with reference to the information flow of TCM consultation: an initial complaint motivates questioning, history contextualizes the condition, and inspection plus tongue/pulse evidence refine syndrome diferentiation. The t1/t2/t3 reveal policy is therefore a diagnostic-workflow-informed staged-information abstraction for benchmark evaluation. Stage t1 exposes chief complaint, t2 adds description/history, and t3 adds detection, including inspection and tongue/pulse-style observations when present. The split is deterministic evidence masking with no generated dialogue or summary, so each row remains traceable to one source case. Predictions are scored at every stage to measure whether added fields correct early errors or cause label flips.

## 3.4 Diagnostic Commitment

For commitment evaluation, each prediction includes one scored action at the current stage: commit returns a diagnosis; acquire advances the scorer to the next evidence stage; defer returns a terminal non-diagnostic outcome; and abstain returns a terminal no-diagnosis outcome under insuficient or conflicting evidence. The scorer moves from t1 to t3 and stops at the first terminal action. Under the task-family scoring policy, full-evidence and no-tongue/pulse variants use a commit-oriented target, whereas rows exposing only candidates, a chief complaint, tongue/pulse evidence, or partial t1/t2 evidence use an acquireor-abstain target. Tongue/pulse perturbation rows are evaluated as fixed-label cue-sensitivity controls. These family-level targets provide a consistent scoring interface across rows.

Expected cost uses fixed penalties per source case. Defaults are 0.25 for an evidence step, 5 for a wrong commit, 1 for defer, and 1.5 for abstain:

$$
\begin{array} { r l } & { \mathrm { C o s t } _ { i } = \lambda _ { s } ( s _ { i } ^ { \star } - 1 ) + \lambda _ { w } W _ { i } + \lambda _ { d } { \bf 1 } [ a _ { i } ^ { \star } = \mathrm { d e f e r } ] + \lambda _ { a } { \bf 1 } [ a _ { i } ^ { \star } = \mathrm { a b s t a i n } ] , } \\ & { \quad W _ { i } = { \bf 1 } [ a _ { i } ^ { \star } = \mathrm { c o m m i t } \wedge { \hat { y } _ { i } ^ { \star } \neq y _ { i } } ] . } \end{array}
$$

Wrong-terminal rates are reported both over all cases and over committed cases:

$$
\mathrm { W T ( a l l ) } = N ^ { - 1 } \sum _ { i } W _ { i } , \qquad \mathrm { W T ( c o m m . ) } = \frac { \sum _ { i } W _ { i } } { \sum _ { i } \mathbf { 1 } [ a _ { i } ^ { \star } = \mathrm { c o m m i t } ] } .
$$

For manuscript reporting, WT(all) is the all-case wrong-terminal rate, while WT(comm.) is the corresponding rate among committed cases. The pre-t3 commit rate records any commit before t3. WT rates are determined directly from model outputs, while expected cost summarizes the same outputs under the stated penalty weights.

## 4 Evaluation Protocol

## 4.1 Prediction Contract

All methods must output JSON-compatible predictions with a stable method id, variant, case id, optional stage, candidate\_syndromes, top1, top3, confidence, action, and need\_more\_evidence. For closed-set variants, top-1 and top-3 labels must come from the provided candidates. Invalid, unparsable, or out-of-candidate predictions are counted explicitly rather than repaired silently.

For reproducibility and provider comparability, DirectLLM uses one fixed two-message prompt template, tcm\_commitbench\_v1, and JSON parser for every model; protocol files show row-to-prompt and prediction-to-scorer conversion. The system message states that the task is a research benchmark, not clinical advice; labels must be copied from candidates; output must be strict JSON; wrong high-confidence answers are benchmark errors; and missing symptoms should not be invented. The user JSON contains task, evidence\_stage, case\_id, visible evidence, candidates, output schema, and copying rules; no provider receives extra provider-specific instructions or repair prompts. Commitment actions are commit, acquire, defer, or abstain. Holding this prompt surface fixed makes model and task-variant comparisons directly paired.

## 4.2 Metrics

We organize metrics around the three evaluation gaps rather than treating every logged field as a main result. Macro-F1, invalid rate, token use, accuracy per 1k tokens, overconfidence error, and full action distributions are retained as diagnostic artifact metrics but are not used as headline claims.

G1 uses random/hard/full top-1 and top-3 to assess semantic candidate discrimination; G2 uses t1/t2/t3 top-1, t3–t1 gain, and prediction flips to assess staged evidence integration; G3 uses commit rate, WT(all), WT(comm.), premature commit, and expected cost to assess commitment behavior. Additional metrics are retained for secondary analysis, keeping the primary claims focused.

Non-model reference metrics are computed separately from model performance. On the held-out test split, uniform top-1 chance is 10.0% for random10 and hard10, 5.0% for hard20, and 0.7% for full148. Uniform top-3 chance is 30.0%, 15.0%, and 2.0%, respectively. Candidate-first top-1 accuracy is close to chance for hard20 (5.0%) and below chance for full148 (0.3%), indicating no strong first-position shortcut. These references are used only to calibrate dificulty and order bias; they are not reported as system results.

Statistical reporting is fixed before model comparison: source-case bootstrap confidence intervals within each split and variant, paired deltas for shared sourcecase cells, and McNemar or paired sign tests as supporting evidence. Main result tables report 95% intervals for the headline DirectLLM metrics, while artifact CSVs retain the full metric set. Intervals use 10,000 source-case bootstrap resamples; paired deltas use the same source-case ids across variants. This design preserves case-level comparability across analyses. We report five complete DirectLLM test matrices under this locked protocol.

## 4.3 DirectLLM Calibration Setup

The reported calibration uses DirectLLM: each model receives the benchmark prompt and returns the required JSON. Static, progressive, and commitment scores use task-specific contracts, so static full-evidence accuracy and progressive t3 accuracy are parallel calibration points. Commitment rows use the same source case, candidate set, and visible stage as matched diagnosis rows, but action and top-1 are generated together under a commitment-specific prompt; Table 5 is paired task evidence, not a natural conversation trajectory. Runs use temperature 0 and omit the provider max\_tokens request field; results are timestamped API snapshots. No other zero-shot or policy method is treated as a competing system; the scorer itself is model-agnostic. This locked setup keeps provider comparisons aligned under the same prompting and scoring conditions.

The completed set includes DeepSeek-v4-flash, DeepSeek-v4-pro, MiniMax-M3, Mimo-v2.5-pro, and Qwen-3.7-Plus, each with static, progressive, and commitment outputs over the 915 held-out test cases across the locked random10, hard10, hard20, full148, perturbation, and insuficient-evidence variants. Incomplete provider runs are excluded.

![](images/e68c5e7ac21eedf81f24090458a49fda9ee731a2b275d147932741043647cf1e.jpg)  
Fig. 2. Representative source-linked cases across the three evaluation gaps.

## 5 Results

## 5.1 Sanity Reference: Learnability and Dificulty Calibration

We include a task-specific ZY-BERT reference only as a benchmark sanity check, not as a DirectLLM comparator. It is fine-tuned with TCM-SD train labels and uses the source development labels for model selection; the reported numbers are evaluated only on the held-out benchmark test split. Candidate-masked probabilities show learnable signal and dificulty calibration: random10 87.8, hard20 61.3, full148 58.8; on hard20, t1/t2/t3 are 26.3/56.1/61.3. The same directional random-to-hard decrease and t1-to-t3 increase therefore appear in both this supervised reference and the zero-shot DirectLLM results. ZY-BERT does not evaluate a commit/acquire/defer/abstain policy.

The larger random10-to-hard20 drop locates most supervised ladder dificulty in profile-neighbor composition. The stage pattern also confirms that history and detection fields add signal beyond the chief complaint. This supervised reference separates benchmark learnability from behavior induced by the zero-shot prompting interface.

## 5.2 Gap 1: Random Distractors Overestimate Semantic Discrimination

Table 4 reports static top-1 accuracy with 95% source-case bootstrap intervals across all four candidate regimes. Random10 reaches 63.2%–66.4%, hard10 falls to 32.1%–36.3%, hard20 to 24.7%–30.3%, and full148 to 14.8%–22.3%. Thus random distractors reward coarse semantic elimination, whereas the hard settings stress ranking among train-profile-neighbor labels. Figure 3A shows the same ladder.

All five models follow the same monotonic pattern. Because random10 and hard10 both contain ten candidates, their 30.1–33.3 point gap isolates candidate composition from candidate count. Expanding hard10 to hard20 produces a further 5.3–7.4 point decrease, showing that both semantic proximity and denser neighbor competition contribute to dificulty.

Table 4. Zero-shot DirectLLM diagnosis across the complete candidate ladder. Values are top-1 accuracy (%) with 95% source-case bootstrap CI.
<table><tr><td>Model</td><td>random10</td><td>hard10</td><td>hard20</td><td>full148</td></tr><tr><td>DS-flash</td><td>65.5 [62.4,68.5]</td><td>34.4 [31.3,37.5]</td><td>28.6 [25.7,31.5]</td><td>17.7 [15.3,20.2]</td></tr><tr><td>DS-pro</td><td>65.9 [62.8,69.0]</td><td>34.9 [31.8,37.9]</td><td>29.6 [26.7,32.7]</td><td>21.5 [18.9,24.2]</td></tr><tr><td>MiniMax</td><td>66.4 [63.4,69.5]</td><td>36.3 [33.1,39.3]</td><td>30.3 [27.3,33.2]</td><td>22.3 [19.7,25.0]</td></tr><tr><td>Mimo</td><td>63.2 [60.0,66.2]</td><td>32.1 [29.2,35.2]</td><td>25.6 [22.8,28.4]</td><td>14.8 [12.5,17.0]</td></tr><tr><td>Qwen</td><td>65.4 [62.2,68.5]</td><td>32.1 [29.2,35.3]</td><td>24.7 [22.0,27.5]</td><td>19.5 [16.8,22.0]</td></tr></table>

![](images/fc251f2b7437b428f57cbccadfe36760e6ea2f00b6d38e72b7b40ad084adde7f.jpg)

![](images/2303a4c7a7a15bff60fb7083eb7ce9c71f44b915be2e648c03c23377409c9fca.jpg)  
Fig. 3. Five DirectLLM calibration runs. Panel A shows mean/min-max static top-1 with chance baselines; Panel B shows hard20 terminal outcomes with WT(all) labels.

## 5.3 Gap 2: Staged Evidence Supports Partial Updating

Figure 4 shows hard20 progressive top-1 accuracy from t1 to t3 and per-model t3–t1 gains. Mean accuracy rises from 19.2% at t1 to 27.3% at t3; Table 5 reports the same gains with 95% CI. The gains range from $+ 5 . 6 \ \mathrm { t o } \ + 1 0 . 4$ points, and every interval excludes zero. The curve supports the staged-evidence premise: later fields add useful signal. At the same time, t3 remains far below solved, so the result is partial evidence integration rather than strong final diagnosis.

![](images/0a9f529a7209282b078aa26f31d95fbbc4cd861af2af8ffcc47d292c05e56c2c.jpg)

![](images/45299d8afecf693b46b0a6d7c4e3a10a45a9ebdc834c897de373dd1c969a5ba9.jpg)  
Fig. 4. Hard20 staged evidence curve and gain analysis. Panel A shows DirectLLM mean top-1 with min-max range across $\mathrm { t 1 / t 2 / t 3 } ;$ Panel B shows each model’s t3–t1 gain with 95% source-case bootstrap CI.

## 5.4 Gap 3: Commitment Scoring Exposes Wrong-Terminal Behavior

Table 5 groups hard20 staged evidence and commitment metrics with 95% CIs. The gain column is from progressive rows; commitment columns are from matched hard20 commitment rows where top1, confidence, and action are generated together. Commit rates are 84.0%–97.3%, and WT(all) is 62.2%–74.3%.

WT(comm.) is 74.0%–77.6%, while cost is 3.61–4.06 per source case. These results show that frequent commitment can coexist with high wrong-terminal rates. Figure 3B shows why final-answer accuracy and diagnostic commitment must be scored separately.

Table 5. Hard20 DirectLLM behavior with 95% CI. Gain is progressive t3–t1; commitment columns come from matched hard20 commitment rows in which top-1 and action are generated together. WT(all) = wrong-terminal commit over all cases; WT(comm.) = wrong-terminal commit among committed cases.
<table><tr><td>Model</td><td>Gain</td><td>Commit</td><td>WT(all)</td><td>WT(comm.)</td><td>Cost</td></tr><tr><td>DS-flash</td><td>8.7 [5.4,12.1]</td><td>85.7 [83.4,87.9]</td><td>64.4 [61.2,67.4]</td><td>75.1 [72.0,78.2]</td><td>3.65 [3.51,3.80]</td></tr><tr><td>DS-pro</td><td>10.4 [6.8,13.9]</td><td>84.0 [81.6,86.3]</td><td>62.2 [59.0,65.2]</td><td>74.0 [70.9,77.0]</td><td>3.61 [3.46,3.74]</td></tr><tr><td>MiniMax</td><td>9.2 [5.9,12.6]</td><td>94.9 [93.4,96.2]</td><td>72.1 [69.2,75.0]</td><td>76.0 [73.2,78.8]</td><td>3.90 [3.76,4.03]</td></tr><tr><td>Mimo</td><td>7.0 [3.8,10.2]</td><td>95.7 [94.4,97.0]</td><td>74.3 [71.4,77.2]</td><td>77.6 [74.8,80.3]</td><td>4.06 [3.92,4.19]</td></tr><tr><td>Qwen</td><td>5.6 [2.1,9.0]</td><td>97.3 [96.2,98.3]</td><td>74.2 [71.5,77.0]</td><td>76.3 [73.6,79.1]</td><td>4.00 [3.87,4.14]</td></tr></table>

Table 6 gives the full insuficient-evidence t1/t2 action summary. These rows further separate action policy from diagnosis accuracy: t1 commit rates range from 1.2% to 13.7%, while t2 commit rates rise to 45.4%–81.2%. The benchmark can therefore express diferent partial-evidence policies without changing the closed-set label candidates.

Table 6. Insuficient-evidence hard20 commitment behavior. Values are percentages with 95% source-case bootstrap CI.
<table><tr><td>Model</td><td>t1 Commit</td><td>t1 WT(all)</td><td></td><td>t2 Commit</td><td>t2 WT(all)</td></tr><tr><td>DS-flash</td><td>9.2 [7.3,11.1]</td><td>6.0 [4.5,7.5]</td><td></td><td>67.2 [64.0,70.2]</td><td>48.7 [45.5,52.0]</td></tr><tr><td>DS-pro</td><td>1.2 [0.5,2.0]</td><td>0.8 [0.2,1.4]</td><td></td><td>45.4 [42.2,48.6]</td><td>33.8 [30.8,36.7]</td></tr><tr><td>MiniMax</td><td>13.7 [11.4,16.0]</td><td>10.6 [8.6,12.6]</td><td></td><td>77.7 [75.0,80.3]</td><td>58.9 [55.7,62.1]</td></tr><tr><td>Mimo</td><td>6.2 [4.7,7.9]</td><td>4.5 [3.2,5.9]</td><td></td><td>81.2 [78.7,83.7]</td><td>61.2 [58.0,64.4]</td></tr><tr><td>Qwen</td><td>4.2 [3.0,5.5]</td><td>3.4 [2.3,4.6]</td><td></td><td>70.2 [67.1,73.1]</td><td>53.7 [50.4,56.9]</td></tr></table>

## 5.5 Failure Decomposition and Qualitative Cases

Together, Figure 2 and Table 7 summarize paired row-level results as a compact failure taxonomy. The identified patterns connect the three gaps: staged evidence can repair early guesses, candidate recall can occur without successful top-1 reranking, and a correct staged prediction can still be followed by a wrong terminal commitment under the separate commitment prompt. These patterns provide a concise row-level view of the three evaluation gaps. The figure and table complement the aggregate results with representative failure cases.

Table 7. Failure decomposition over five hard20 DirectLLM runs. Counts are ranges over models out of 915 held-out source cases.
<table><tr><td>Failure bucket</td><td>Range</td><td>Interpretation</td></tr><tr><td>t1 wrong -&gt; t3 correct</td><td>137-187</td><td>Added fields can repair early guesses.</td></tr><tr><td>Correctable but wrong commit</td><td>58-81</td><td>The action trajectory can fail despite a correct t3 prediction.</td></tr><tr><td>Static wrong, gold in top-3 209–264</td><td></td><td>Candidate recall can exist while reranking fails.</td></tr></table>

## 6 Discussion

The results reveal three distinct bottlenecks: profile-neighbor reranking, incomplete integration of additional evidence, and disagreement between terminal actions and inherited labels. Holding source cases fixed isolates these efects while candidate composition, evidence stage, and action scoring vary. The equal-size random10/hard10 example in Figure 2 separates candidate composition from candidate count, and the shared case 000399 shows that a t3 correction need not carry over to the matched terminal action. G1 therefore measures sensitivity to train-profile semantic proximity, while G3 measures terminal agreement under the locked benchmark policy. Fixed-label tongue/pulse rows complement these axes by measuring prediction response to a localized cue change.

Limitations and future validation. TCM-DiagCommit inherits the data scope of TCM-SD/ZY-BERT and uses label-balanced sampling, so its label distribution may difer from real-world syndrome prevalence. Source-case intervals quantify case variation under the locked prompt; prompt wording and evolving API snapshots remain separate robustness dimensions. Future work can add clinician stage-wise anchors, alternative neighbor constructions, supervised retrieval/reranking baselines, and calibrated commitment policies.

## 7 Conclusion

TCM-DiagCommit reframes TCM syndrome diferentiation from static label selection into evidence-conditioned syndrome selection and benchmark commitment scoring while preserving TCM-SD source labels. Random distractors overestimate ability, staged evidence only partially improves zero-shot LLM predictions, and wrong-terminal commitment remains frequent under a locked benchmark policy. This supports more reliable diagnostic evaluation. Future medical LLM evaluation should report final-answer accuracy together with evidenceconditioned updating and terminal commitment behavior.

Disclosure of Interests. The authors declare that they have no competing interests related to this work.

## References

1. Angelopoulos, A.N., Bates, S.: Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning 16(4), 494–591 (2023), https://doi.or g/10.1561/2200000101

2. Chen, C., Wang, X., Guan, M., et al.: Evaluating and improving syndrome differentiation thinking ability in large language models: Method development study. JMIR Medical Informatics 13, e75103 (2025), https://medinform.jmir.org/202 5/1/e75103

3. Geifman, Y., El-Yaniv, R.: Selective classification for deep neural networks. In: NeurIPS. vol. 30 (2017), https://proceedings.neurips.cc/paper/2017/hash/4 a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html

4. Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On calibration of modern neural networks. In: ICML. vol. 70, pp. 1321–1330 (2017), https://proceedings.mlr.pr ess/v70/guo17a.html

5. Kaushik, D., Hovy, E., Lipton, Z.C.: Learning the diference that makes a diference with counterfactually-augmented data. In: ICLR (2020), https://openreview.n et/forum?id=Sklgs0NFvr

6. Kong, S., Yang, X., Wei, Y., et al.: MTCMB: A multi-task benchmark framework for evaluating large language models on knowledge, reasoning, and safety in traditional Chinese medicine (2025), https://arxiv.org/abs/2506.01252, arXiv preprint arXiv:2506.01252

7. Liu, B., Huang, H., Liu, X., Zhao, J., Mo, J.: Dual-channel knowledge attention for traditional Chinese medicine syndrome diferentiation. Scientific Reports 15, 13487 (2025), https://www.nature.com/articles/s41598-025-96404-w

8. Liu, J., Zhou, P., Hua, Y., et al.: Benchmarking large language models on CMExam - a comprehensive Chinese medical exam dataset. In: NeurIPS. vol. 36, pp. 52430– 52452 (2023), https://doi.org/10.52202/075280-2283

9. Nori, H., Daswani, M., Kelly, C., et al.: Sequential diagnosis with language models (2025), https://arxiv.org/abs/2506.22405, arXiv preprint arXiv:2506.22405

10. Ren, M., Huang, H., Zhou, Y., et al.: TCM-SD: A benchmark for probing syndrome diferentiation via natural language processing. In: CCL. pp. 908–920 (2022), http s://aclanthology.org/2022.ccl-1.80/

11. Schmidgall, S., Ziaei, R., Harris, C., et al.: AgentClinic: A multimodal benchmark for tool-using clinical AI agents. npj Digital Medicine 9, 499 (2026), https: //www.nature.com/articles/s41746-026-02674-7

12. Wang, X., Chen, G.H., Song, D., et al.: CMB: A comprehensive medical benchmark in Chinese. In: NAACL. pp. 6184–6205 (2024), https://aclanthology.org/202 4.naacl-long.343/

13. Yue, W., Wang, X., Zhu, W., et al.: TCMBench: A comprehensive benchmark for evaluating large language models in traditional Chinese medicine (2024), https: //arxiv.org/abs/2406.01126, arXiv preprint arXiv:2406.01126

14. Zhang, N., Chen, M., Bi, Z., et al.: CBLUE: A Chinese biomedical language understanding evaluation benchmark. In: ACL. pp. 7888–7915 (2022), https: //aclanthology.org/2022.acl-long.544/