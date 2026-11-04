# Structured Clinical Memory as Context for Probabilistic Decision Models: A Synthetic Evaluation with Jev

Jash Sayani

Gaurang Karwande

Achinth Bharadwaj

jash.sayani@abbycare.ai gaurang.karwande@abbycare.ai achinth.bharadwaj@abbycare.ai

Abstract—Longitudinal clinical context is often distributed across noisy, repetitive forms and narrative records. We study whether a compact structured memory can improve a probabilistic decision model’s answers when the underlying patient vignette is already available. We created 30 fully synthetic cases spanning twelve clinical domains, paired each with synthetically generated structured memory, and evaluated 120 typed questions with Jev. The raw-vignette arm achieved 90.0% accuracy (95% normalapproximation confidence interval: 84.6%-95.4%); adding memory achieved 93.3% (88.9%-97.8%). Five questions changed from incorrect to correct and one changed in the opposite direction (exact McNemar p = 0.219). Choice log loss decreased from 0.258 to 0.175 and binary Brier score from 0.054 to 0.022. Memory added 310 input tokens per call on average, while observed mean latency was unchanged at approximately 0.5 seconds. These results are directionally positive rather than clinically validating: the cases are synthetic, the labels are single-author judgments, and the memory is an oracle upper bound rather than extracted production output. The pilot supports evaluating structured clinical memory as a reusable state interface for decision models.

Keywords: clinical memory; probabilistic decision models;   
structured context; synthetic evaluation; calibration; Jev.

## I. STRUCTURED CONTEXT MAY IMPROVE DECISION MODELS

Clinical decision support frequently depends on facts recorded outside the immediate encounter: an active condition, a resolved episode, an uncertain diagnosis, a medication, or a prior contraindication. Repeatedly passing all underlying forms to a model preserves source detail but also repeats irrelevant text and asks the model to reconstruct longitudinal state on every call.

We study a narrower interface: a maintained clinical memory containing typed features such as condition or medication, status, onset, certainty, source form, negation, and a short note. This memory is not a free-form summary. It is structured state that can be appended to a current vignette and consumed by a model whose outputs are choices, probability weighted choices, binary judgments, or ordinal scores.

The concrete model in this pilot is Jev, described by its developer TypeSafe AI as accepting unstructured state and returning typed probabilistic decisions in parallel [1]. The study does not test the vendor’s broader performance claims. It asks whether the quality and accuracy of the responses change when structured clinical memory accompanies the raw vignette.

## Contributions

This pilot contributes: (i) a paired benchmark of 30 synthetic cases and 120 typed questions; (ii) an explicit memory representation capturing longitudinal status and provenance; (iii) a two-arm evaluation using identical questions; and (iv) accuracy, probability-quality, confidence, token, and latency analyses with limitations stated as part of the result.

## II. RELATED WORK

External-memory systems for language models manage information beyond a single context window. MemGPT, for example, frames memory as a hierarchy whose contents can be moved between limited context and external storage [2]. Our study differs in both representation and consumer: the memory is a fixed clinical feature schema, and the downstream model emits typed probabilistic decisions rather than generated dialogue.

Clinical natural-language processing has long extracted concepts and attributes from records. cTAKES introduced a modular pipeline for clinical free-text annotations, including concept mapping, negation, and status attributes [3]. This pilot does not evaluate extraction. Its synthetically generated memory intentionally isolates the downstream question: if a suitable structured state exists, does it change decision quality? We intentionally isolate agent memory as an independent component in order to isolate and understand its quality contribution to decision quality given a structured state.

Probability quality matters because workflows do not just consume the decision, but also the uncertainty of the decisions. We therefore report log loss for multiple-choice outputs and Brier score for binary probabilities [4], alongside thresholded accuracy.

## III. THE BENCHMARK ISOLATES CONTEXT REPRESENTATION

## A. Synthetic cases and questions

We authored 30 synthetic patient cases across twelve domain labels: cardiology, diabetes, respiratory, infectious disease, neurology, mental health, musculoskeletal, renal, gastrointestinal, allergy, geriatrics, and pain. No real patient records or protected health information were used. Each vignette intentionally included clinically relevant facts plus ordinary ambiguity or distractors.

The benchmark contains 120 questions: 56 multiple-choice questions, 36 binary questions, and 28 five-level ordinal scores.

![](images/df81c50f9903253b1a2503f5ecc8f48c9ca379ff49d7506f4d6ee6162f4d90a8.jpg)

Fig. 1. Paired evaluation design. Only the presence of structured clinical memory differs between the two arms; the vignette, decider model, and typed probabilistic questions are identical.
<table><tr><td rowspan=1 colspan=1>Output</td><td rowspan=1 colspan=1>Accuracy rule</td><td rowspan=1 colspan=1>Probability metric</td></tr><tr><td rowspan=1 colspan=1>Choice</td><td rowspan=1 colspan=1>Exact option</td><td rowspan=1 colspan=1>Log loss</td></tr><tr><td rowspan=1 colspan=1>Binary</td><td rowspan=1 colspan=1> $\mathrm { P ( y e s ) } \geq 0 . 5$ </td><td rowspan=1 colspan=1>Brier score</td></tr><tr><td rowspan=1 colspan=1>Score</td><td rowspan=1 colspan=1>Absolute error ≤ 1</td><td rowspan=1 colspan=1>Mean absolute error</td></tr></table>

Tasks cover triage priority, next action, risk scoring, condition presence, and contraindication checks. Gold answers and rationales were authored once and shared by both experimental arms.

## B. Structured clinical memory

Each case has a synthetically generated memory comprising a list of condition or medication features. Every feature includes a normalized label, longitudinal status (active, history of, or resolved), onset, certainty, source form, negation flag, and optional note. The memory was designed to preserve facts that are often obscured when several forms are read together.

## C. Paired experimental arms

Every case was evaluated twice. Arm A supplied the synthetic vignette as model state. Arm B supplied the same vignette followed by the structured clinical-memory block. All questions for one case were submitted in a single call and evaluated in parallel. Thus, 30 cases multiplied by two arms yielded 60 model calls.

## IV. OUTCOMES AND ANALYSIS

## A. Primary correctness metrics

Multiple-choice answers were correct only when the selected option matched the gold key. Binary outputs were mapped to “yes” when the reported probability was at least 0.5. Ordinal score outputs were correct when their probabilityweighted value was within one level of the gold score.

## B. Paired inference

Overall accuracy uses all 120 questions in each arm. We report normal-approximation 95% confidence intervals for descriptive continuity with the scoring harness. Because each question appears in both arms, we additionally report the discordant-pair counts and a two-sided exact McNemar test. With only six discordant pairs, the test has low power.

<table><tr><td rowspan=1 colspan=1>Condition</td><td rowspan=1 colspan=1>Correct</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>95% CI</td></tr><tr><td rowspan=1 colspan=1>Arm A: vignette</td><td rowspan=1 colspan=1>108 /  120</td><td rowspan=1 colspan=1>90.0%</td><td rowspan=1 colspan=1>84.6%–95.4%</td></tr><tr><td rowspan=1 colspan=1>Arm B: + memory</td><td rowspan=1 colspan=1>112 / 120</td><td rowspan=1 colspan=1>93.3%</td><td rowspan=1 colspan=1>88.9%–97.8%</td></tr><tr><td rowspan=1 colspan=1>Difference</td><td rowspan=1 colspan=1>+4</td><td rowspan=1 colspan=1>+3.3 pp</td><td rowspan=1 colspan=1>Not estimated</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Question type</td><td rowspan=1 colspan=1>n</td><td rowspan=1 colspan=1>Arm A</td><td rowspan=1 colspan=1>Arm B</td></tr><tr><td rowspan=1 colspan=1>Choice accuracy</td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=1>89.3%</td><td rowspan=1 colspan=1>91.1%</td></tr><tr><td rowspan=1 colspan=1>Binary accuracy</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>94.4%</td><td rowspan=1 colspan=1>97.2%</td></tr><tr><td rowspan=1 colspan=1>Score within ±1</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>85.7%</td><td rowspan=1 colspan=1>92.9%</td></tr></table>

## C. Confidence and efficiency

For confidence analysis, predictions were grouped into five bins from 0.5 to 1.0. We compare mean confidence and empirical accuracy within each populated bin, while noting that some outputs lacked a confidence field and were excluded from this analysis. Efficiency outcomes are mean input tokens and wall-clock latency per call as reported by the run harness.

## D. Scope

The unit of analysis is a model answer, not a patient outcome. No clinicians reviewed the labels, no safety endpoint was measured, and no intervention was recommended or delivered. The experiment is a controlled software benchmark.

## V. ACCURACY IMPROVED BY 3.3 PERCENTAGE POINTS

Arm A answered 108 of 120 questions correctly (90.0%). Arm B answered 112 correctly (93.3%), an absolute difference of 3.3 percentage points. The descriptive confidence intervals overlap. In the paired analysis, five questions changed from incorrect in Arm A to correct in Arm B, while one changed from correct to incorrect. The resulting exact two-sided Mc-Nemar p-value is 0.219.

This pattern is compatible with a modest benefit, but it is not sufficient to reject the null of equal paired error rates. The small number of errors and discordant pairs dominates the uncertainty.

## A. Gains appeared across output types

The largest descriptive increase was on ordinal scores: 7.2 percentage points. However, the per-type samples are small and their confidence intervals overlap. These breakdowns should be treated as hypotheses for a larger benchmark rather than stable estimates of task-specific effect.

## VI. PROBABILITY QUALITY IMPROVED ON CHOICE AND BINARY TASKS

Choice log loss decreased from 0.258 to 0.175, indicating that the memory condition assigned more probability to the gold option on average. Binary Brier score decreased from 0.054 to 0.022. For ordinal scores, mean absolute error decreased from 0.521 to 0.469 levels.

<table><tr><td rowspan=1 colspan=1>Metric (lower is better)</td><td rowspan=1 colspan=1>Arm A</td><td rowspan=1 colspan=1>Arm B</td><td rowspan=1 colspan=1>Change</td></tr><tr><td rowspan=1 colspan=1>Choice log loss</td><td rowspan=1 colspan=1>0.258</td><td rowspan=1 colspan=1>0.175</td><td rowspan=1 colspan=1>-0.083</td></tr><tr><td rowspan=1 colspan=1>Binary Brier score</td><td rowspan=1 colspan=1>0.054</td><td rowspan=1 colspan=1>0.022</td><td rowspan=1 colspan=1>-0.032</td></tr><tr><td rowspan=1 colspan=1>Score MAE</td><td rowspan=1 colspan=1>0.521</td><td rowspan=1 colspan=1>0.469</td><td rowspan=1 colspan=1>-0.052</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Confidence bin</td><td rowspan=1 colspan=2>Arm A</td><td rowspan=1 colspan=2>Arm B</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>n</td><td rowspan=1 colspan=1>acc.</td><td rowspan=1 colspan=1>n</td><td rowspan=1 colspan=1>acc.</td></tr><tr><td rowspan=1 colspan=1>0.5-0.6</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>88.9%</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>87.5%</td></tr><tr><td rowspan=1 colspan=1>0.6-0.7</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>91.7%</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>100%</td></tr><tr><td rowspan=1 colspan=1>0.7-0.8</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>78.3%</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>88.9%</td></tr><tr><td rowspan=1 colspan=1>0.8-0.9</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>81.2%</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>100%</td></tr><tr><td rowspan=1 colspan=1>0.9-1.0</td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=1>98.5%</td></tr></table>

A. Memory increased high-confidence predictions while preserving their accuracy

The 0.9-1.0 confidence bin contained 55 Arm A predictions with 100% empirical accuracy and 65 Arm B predictions with 98.5% accuracy. Memory therefore moved more predictions into the highest-confidence bin while preserving high observed accuracy. Other bins were sparse and non-monotonic, so the pilot does not establish formal calibration.

## VII. MEMORY ADDED TOKENS BUT NOT OBSERVED LATENCY

Mean input length increased from 553.9 to 863.9 tokens per call, an increment of 310.0 tokens (56.0%). Mean measured latency was 0.51 seconds in Arm A and 0.50 seconds in Arm B across 30 calls per arm.

The near-equal latency should not be interpreted as a general performance guarantee. The run was small, networked, and not designed for latency inference. The token result is more direct: this implementation trades additional structured context for a modest accuracy and probability-quality gain.

## VIII. THE RESULT IS AN UPPER-BOUND PILOT

The main finding is not that structured memory solves clinical reasoning. It is that, under a controlled synthetic benchmark and a strong raw-text baseline, appending a compact longitudinal state was associated with four additional correct answers and better probability scores. The direction is consistent across choice, binary, and ordinal outputs, but statistical uncertainty remains substantial.

## A. Why oracle memory matters

Arm B receives memory that is cleaner than a production extraction pipeline would provide. Real systems may omit a condition, merge two entities incorrectly, mishandle negation, preserve stale status, or assign unjustified certainty. Any of these errors could reduce or reverse the observed benefit. The result should therefore be read as a ceiling for this representation on this benchmark, not an estimate of deployed performance.

<table><tr><td rowspan=1 colspan=1>Efficiency outcome</td><td rowspan=1 colspan=1>Arm A</td><td rowspan=1 colspan=1>Arm B</td></tr><tr><td rowspan=1 colspan=1>Mean input tokens</td><td rowspan=1 colspan=1>553.9</td><td rowspan=1 colspan=1>863.9</td></tr><tr><td rowspan=1 colspan=1>Mean latency</td><td rowspan=1 colspan=1>0.51 s</td><td rowspan=1 colspan=1>0.50 s</td></tr><tr><td rowspan=1 colspan=1>Calls</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>30</td></tr></table>

## B. The feature schema is the intervention

The memory block does more than shorten text. It asserts a longitudinal interpretation: which features are active, historical, resolved, negated, or uncertain, and which source form supports each item. The evaluation cannot separate the contribution of structure from the contribution of expertauthored normalization. Future ablations should compare the full schema with simpler summaries and retrieval of raw source passages.

## IX. LIMITATIONS

1) Synthetic data. The cases do not reproduce the distribution, ambiguity, documentation errors, or missingness of real clinical records.

2) Single-author gold labels. Labels and rationales were not independently reviewed or adjudicated by clinicians.

3) Oracle memory. The study does not measure extraction precision, recall, deduplication, temporal updating, or failure propagation.

4) Small sample. With 120 questions and six discordant pairs, the paired comparison is underpowered.

5) Question dependence. Four questions share each case, so treating all 120 as independent understates clustering in descriptive intervals.

6) One model and one run. Results may vary across model revisions, sampling behavior, service conditions, and alternative decision architectures.

7) Thresholded metrics. Binary accuracy depends on a 0.5 threshold, while ordinal correctness allows ±1 level; alternate decision costs may favor different rules.

8) No clinical endpoint. Benchmark correctness does not establish safety, utility, equity, or improved patient outcomes.

## X. NEXT EVALUATION MUST TEST EXTRACTED MEMORY PROSPECTIVELY

A stronger study should connect representation quality to downstream decisions. First, a frozen extraction system should produce memory from a preregistered set of de-identified or fully synthetic forms. Clinicians should independently label features and adjudicate disagreements, enabling precision, recall, F1, and temporal-status error analysis.

Second, the paired decision benchmark should be expanded and grouped by patient, with sample size chosen for a clinically meaningful effect. Analyses should use cluster-aware intervals or patient-level bootstrap resampling. Errors should be stratified by task, specialty, memory dependence, and potential harm.

Third, comparisons should include raw forms, retrieved source passages, unstructured summaries, and structured memory. Schema ablations can test whether status, onset, certainty, source provenance, or negation produces the gain. A deliberately corrupted-memory arm would quantify sensitivity to stale or incorrect features.

Finally, a prospective human-factors evaluation should measure whether clinicians find the resulting decisions appropriately calibrated and whether the memory makes source verification easier. Human oversight, auditability, and abstention behavior are central for any clinical use.

## XI. ETHICAL AND SAFETY CONSIDERATIONS

This study uses synthetic cases only. The reported probabilities and decisions are benchmark outputs, not medical advice. The benchmark contains no demographic fairness analysis and cannot support claims about subgroup performance. Before use with real records, data governance, privacy controls, access logging, source traceability, bias evaluation, and clinical review would be required.

## XII. CONCLUSION

Structured clinical memory is associated with increased accuracy from 90.0% to 93.3% and improved probability scores in a small synthetic Jev benchmark. The result is directionally encouraging but not statistically conclusive or clinically validating. Its value is as a testable research hypothesis: a maintained, typed account of longitudinal clinical state may be a useful context interface for probabilistic decision models. The next step is to replace oracle memory with measured extraction output and evaluate the full error chain under clinician review.

APPENDIX
<table><tr><td rowspan=1 colspan=1>Domain label</td><td rowspan=1 colspan=1>Cases</td><td rowspan=1 colspan=1>Questions</td></tr><tr><td rowspan=1 colspan=1>Cardiology</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>Diabetes</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>Respiratory</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>Infectious disease</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>Neurology</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>Mental health</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>Musculoskeletal</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>Renal</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>Gastrointestinal</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>Allergy</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>Geriatrics</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>Pain</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>Total</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>120</td></tr></table>

Arm B serialized these fields into a clinical-memory block appended to the same vignette used in Arm A. The schema is a research artifact rather than a clinical interoperability standard.

• Benchmark version: v1.1, generated October 12, 2026.

<table><tr><td rowspan=1 colspan=1>Field</td><td rowspan=1 colspan=1>Meaning in this pilot</td></tr><tr><td rowspan=1 colspan=1>feature_type</td><td rowspan=1 colspan=1>Condition or medication</td></tr><tr><td rowspan=1 colspan=1>condition</td><td rowspan=1 colspan=1>Normalized feature label</td></tr><tr><td rowspan=1 colspan=1>status</td><td rowspan=1 colspan=1>Active, history_of, or resolved</td></tr><tr><td rowspan=1 colspan=1>onset</td><td rowspan=1 colspan=1>Synthetic onset date or period</td></tr><tr><td rowspan=1 colspan=1>certainty</td><td rowspan=1 colspan=1>Author-assigned value from 0 to 1</td></tr><tr><td rowspan=1 colspan=1>source_form</td><td rowspan=1 colspan=1>Originating synthetic form type</td></tr><tr><td rowspan=1 colspan=1>negated</td><td rowspan=1 colspan=1>Whether the feature is explicitly absent</td></tr><tr><td rowspan=1 colspan=1>note</td><td rowspan=1 colspan=1>Optional synthetic context</td></tr></table>

• Cases: 30; questions: 120; paired model calls: 60.

• Question mix: 56 choice, 36 binary probability, 28 ordinal score.

• Primary run date: October 12, 2026.

• Data status: entirely synthetic; no protected health information.

## REFERENCES

1) D. Almeida. “Introducing System One Models & Jev.” TypeSafe AI Blog, 2026. https://typesafe.ai/blog/ introducing-system-one-models-and-jev

2) C. Packer, S. Wooders, K. Lin, V. Fang, S. G. Patil, I. Stoica, and J. E. Gonzalez. “MemGPT: Towards LLMs as Operating Systems.” arXiv:2310.08560, 2023.

3) G. K. Savova, J. J. Masanz, P. V. Ogren, J. Zheng, S. Sohn, K. C. Kipper-Schuler, and C. G. Chute. “Mayo clinical Text Analysis and Knowledge Extraction System (cTAKES): architecture, component evaluation and applications.” Journal of the American Medical Informatics Association, 17(5):507-513, 2010. doi:10.1136/ jamia.2009.001560.

4) G. W. Brier. “Verification of Forecasts Expressed in Terms of Probability.” Monthly Weather Review, 78(1):1–3, 1950. doi:10.1175/1520-0493(1950) 078⟨0001:VOFEIT⟩2.0.CO;2.