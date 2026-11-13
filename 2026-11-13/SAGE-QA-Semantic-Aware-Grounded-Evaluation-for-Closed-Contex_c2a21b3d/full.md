# SAGE-QA: Semantic-Aware, Grounded Evaluation for Closed-Context Question Answering on SQuAD 2.0

Chandra N Sekharan, Vishakan Umapathy, and Abhishek Phadke

Abstract—Closed-context question answering (QA) involves a system to generate an answer strictly supported by a supplied passage or to abstain when the passage lacks adequate evidence. The Stanford Question Answering Dataset (SQuAD 2.0) is a key benchmark widely used to evaluate machine reading comprehension under these constraints [1]–[3]. Traditionally, model performance on SQuAD 2.0 is measured via Exact Match (EM), which requires character-level agreement with human reference spans after lexical normalization. For generative large language models (LLMs), however, an EM failure does not necessarily indicate an erroneous response; it may instead reflect a valid paraphrase, an alternative passage-supported answer, or a hallucination. Prior results on the official leader board achieved a maximum EM of approximately 91% [3] but evaluating generative models requires separating lexical divergence from factual error. Our primary contribution of the paper is the introduction of SAGE-QA (Semantic-Aware, Grounded Evaluation for Question Answering), a multi-tiered evaluation framework that combines reference-based Semantic Equivalence Scoring (SES) and passage-based Re-Validation Scoring (RVS). Across a fixed set of 10,000 SQuAD 2.0 validation examples, we compute baseline EM and decision-matrix F1 directly from True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN) outcome counts across zero-shot, closedweight frontier models and open-weight models (untuned zeroshot and fine-tuned versions). In zero-shot evaluation, closedweight frontier models Claude Opus 4.6 and Gemini 3.1 Pro achieve 85.21 and 86.66 EM, outperforming untuned open-weight models Llama 3 8B (56.88 EM), gpt-oss-20b (66.28 EM), and Llama 3.3 70B (71.39 EM). However, parameter-efficient finetuning yields gains of 21.19 to 34.06 EM points over untuned baselines, elevating fine-tuned Llama 3 8B, gpt-oss-20b, and Llama 3.3 70B to 90.94, 87.47, and 94.24 EM scores, respectively. Additionally, when we audit the zero-shot frontier models using SAGE-QA with an external LLM judge, the dataset-wide SES scores (on a 0–100 scale) are 92.81 for Claude Opus 4.6 and 91.67 for Gemini 3.1 Pro, while the corresponding hybrid datasetwide RVS scores are 97.96 and 98.02. This combines benchmarkbased credit with passage-based judge assessments. Our findings demonstrate that fine-tuning open-weight models perform the best to surpass previous best known EM benchmarks.

Index Terms—Question answering, SQuAD 2.0, semantic evaluation, passage grounding, LLM-as-a-judge, prompt optimization, fine-tuning.

## I. INTRODUCTION

evidence provided in a companion passage, or returning an explicit abstention token ([]) when no answer is supported. We explore this question because QA systems needs to satisfy the information need expressed by the query while following the strict evidence boundary set by the source text. The Stanford Question Answering Dataset (SQuAD) [1], [2] serves as the benchmark for this work, comprising comprehension questions authored by crowd workers over Wikipedia passages. SQuAD 2.0 extends the original 100,000 answerable questions of SQuAD 1.1 with over 50,000 adversarial, unanswerable questions designed to closely resemble answerable queries [1]. SQuAD 2.0 requires a solution that not only extracts or generates accurate answers when sufficient evidence is present, but also reliably abstains when the passage does not support a conclusion. For clarity, reference answers are to be considered as ground-truths because they are provided by human crowd workers.

While traditional AI retrieval models select contiguous character spans from the passage, modern generative models frequently formulate complete phrases or valid paraphrases. Under standard Exact Match (EM) evaluation, any deviation from the exact wording or span boundaries chosen by crowd annotators is penalized as a failure. To examine disagreements between exact reference matching and LLM-based assessment, we propose the SAGE-QA evaluation framework, which extends standard confusion-matrix metrics to include two complementary LLM-as-a-judge diagnostics triggered whenever a prediction fails EM.

1) Semantic Equivalence Scoring (SES): SES uses evaluation on answerable questions where the model’s answer fails EM. SES employs an external judge LLM to assess whether the generated answer preserves the semantic meaning of at least one (ground-truth) reference answer.

2) Re-Validation Scoring (RVS): This evaluates on all predictions that fail EM (spanning both answerable and benchmark-unanswerable questions), a judge LLM is used to assess whether the candidate answer directly addresses the question and if it is fully grounded in the source passage. The judge receives no reference answers (including not answerable or [] type of answers). But scoring uses all of the reference answers to check against the model answer.

To avoid direct self-evaluation that may be biased, we execute both SES and RVS using an external evaluator model (gpt-6-luna [5]) that does not participate as an answer

TABLE I  
HISTORICAL LEADERBOARD CONTEXT ON THE OFFICIAL HIDDEN TEST SET [3].
<table><tr><td>System</td><td>EM</td><td> $\mathbf { F 1 } _ { \mathrm { r e p o r t e d } }$ </td></tr><tr><td>Human Performance</td><td>86.831</td><td>89.452</td></tr><tr><td>IE-Net (Discriminative Ensemble)</td><td>90.939</td><td>93.214</td></tr></table>

generator in our benchmark.

Architecturally, we maintain a strict separation between answer generation and answer evaluation. On the generation side, we investigate automated prompt optimization for zero-shot, closed-weight frontier models (Claude Opus 4.6 and Gemini 3.1 Pro) and evaluate open-weight models (Llama 3 8B, Llama 3.3 70B, and gpt-oss-20b) in both untuned, zero-shot and parameter-efficient fine-tuned (LoRA and QLoRA) configurations. This allows us to compare closed- and open-weight models zero-shot, while isolating the gain of fine-tuning. On the evaluation side, we retain strict lexical matching (EM and outcome-count F1) and add separate judge-based assessments of reference-meaning agreement (SES) and passage support (RVS). Our diagnostic contribution is to characterize which non-matching responses receive judge-assigned credit, where the two rubrics disagree, and how the evaluation criterion affects aggregate scores and model comparisons.

## II. BACKGROUND AND RELATED WORK

## A. Benchmark Evolution and Extractive Baselines

SQuAD 1.1 focused exclusively on span extraction for answerable queries [2] but SQuAD 2.0 introduced unanswerable questions to penalize models that rely on superficial keyword matching or heuristic guessing [1]. Early neural architectures, such as BiDAF [6] and discriminative pre-trained encoders based on BERT [7] and RoBERTa [8], framed closed-context QA as a token-classification problem over passage start and end indices. Table I summarizes key reference points from the official SQuAD 2.0 leaderboard [3], including human baseline performance and IE-Net expressed as a percentage. Note that the leaderboard’s $\mathrm { F 1 } _ { \mathrm { r e p o r t e d } }$ metric measures token-level bagof-words overlap; we provide it here solely for historical context, as it is mathematically distinct from the decisionmatrix F1 defined in Section III.

When generative LLMs are evaluated on span-retrieval benchmarks, the relationship between token overlap and answer correctness breaks down [9]–[11]. Consider a representative example from the SQuAD 2.0 repository [12].

Context: “To further highlight the difference between a problem and an instance, consider the following instance of the decision version of the traveling salesman problem: Is there a route of at most 2000 kilometres passing through all of Germany’s 15 largest cities? The quantitative answer to this particular problem instance is of little use for solving other instances of the problem, such as asking for a round trip through all sites in Milan whose total length is at most 10 km. For this reason, complexity theory addresses computational problems and not particular problem instances.”

Question: What is one example of an instance that the quantitative answer to the traveling salesman problem fails to answer?

Human Reference Answers: ["round trip   
through all sites in Milan", "asking   
for a round trip through all sites in   
Milan whose total length is at most 10   
km", "a round trip through all sites in   
Milan whose total length is at most 10   
km"]

A generative model that responds with “finding a round trip of at most 10 km visiting every site in Milan” provides a factually exact, passage-grounded answer that nonetheless receives a score of zero under Exact Match. Notice that the three human references differ only in whether they include the leading words (“asking for $a . . . ^ { \prime \prime } )$ or the trailing constraint (“whose total length is at most 10 km”); consequently, any valid generative paraphrase or alternative span choice is penalized as a complete failure under surface-level lexical matching.

## B. Semantic Equivalence and Passage Grounding

Kamalloo et al. [9] demonstrated that lexical matching severely underestimates generative LLMs in open-domain QA because models frequently produce valid paraphrases or varied granularity spans. However, their analysis focused on alwaysanswerable open-domain queries; SAGE-QA examines evaluation disagreement in closed-context QA with adversarial unanswerable questions, separating judge-assessed reference equivalence (SES on $\mathcal { F } _ { N } ^ { \mathrm { a n s } } )$ from judge-assessed passage support (RVS on $\mathcal { F } _ { P } \cup \mathcal { F } _ { N } ^ { \mathrm { a n s } } )$ , as defined precisely later. Embedding-based metrics such as BERTScore [13] compute contextual token similarity, but recent frameworks such as G-Eval [14] and MT-Bench [15] leverage LLMs directly as automated judges. Studies of annotation artifacts and label errors motivate closer examination of benchmark failures [16], [17]; they do not establish that disagreements observed under our judging protocol are annotation errors.

Building toward semantic evaluation in QA, Bulian et al. [10] formulated question-aware answer-equivalence judgments and trained the learned BEM (BERT Matching) evaluator. Their asymmetric formulation can accept a candidate answer that expands upon or improves a reference, but our SES metric enforces a symmetric, scaled reference-meaning rubric anchored by strict mutual equivalence, in a logical sense. Complementing reference equivalence, Adlakha et al. [11] demonstrated the importance of separating reference correctness from context faithfulness in instruction-following QA. The RVS judge applies a passage-based rubric to assess this faithfulness criterion within a closed-context setting, without receiving reference answers.

Directly relevant to our benchmark dataset, Bahak et al. [18] evaluated ChatGPT (gpt-3.5-turbo) as a closed-context QA system on SQuAD 1.1 and SQuAD 2.0 using standard character-normalized EM, Recall, and token-overlap F1. They found that ChatGPT trailed task-specific discriminative models by a wide margin, achieving only 44.4% EM (67% F1) on SQuAD 1.1 and plummeting to 18.12% EM (56% F1) on SQuAD 2.0. Bahak et al. attributed this collapse on SQuAD 2.0 to the model generating substantive responses on unanswerable questions, automatically classifying all such False Positives as model hallucinations.

Our study examines three related aspects: first, programmatic prompt optimization and structured abstention schemas yield zero-shot, frontier EM above 85%–86%; second, the RVS judge accepts over 90% of benchmark False Positives in our frontier cohort, documenting disagreement with benchmark answerability labels, and third, parameter-efficient, finetuning enables open-weight generative models to surpass the 90.58%–90.94% EM ceiling from prior literature.

## C. Prompt Optimization, Adaptation, and Evaluator Bias

To obtain optimal performance from generative models, recent work has shifted from manual prompt engineering to programmatic prompt compilation using frameworks such as DSPy [19], MIPROv2 [20], GEPA [21], and automated migration optimizers [22]. In parallel, parameter-efficient finetuning techniques, notably LoRA [23] and QLoRA [24], enable open-weight models to adapt rapidly to domain-specific output formatting and abstention calibration.

When auditing model outputs with LLM judges, however, evaluators can exhibit self-preference bias by favoring generations produced by their own model family, and sensitivity to prompt structure and candidate ordering [4], [25]. We avoid direct self-evaluation by employing a single, external judge model (gpt-6-luna [5]) that generates none of the candidate answers in our study, and enforce a uniform, blind input schema across all evaluated systems.

## III. DEFINITIONS AND THE SAGE-QA EVALUATION FRAMEWORK

We first define a partitioning of N QA evaluation items into four mutually exclusive outcome sets which will be used throughout the rest of the paper.

• True Positive $( \mathcal { T } _ { P } ) \colon$ The question is answerable $( a _ { i } = 1 )$ the model does not abstain $( z _ { i } = 0 )$ , and the normalized prediction matches a reference $( e _ { i } = 1 )$ .

• True Negative $( { \mathcal { T } } _ { N } ) { : }$ The question is unanswerable $( a _ { i } =$ 0) and the model abstains $( z _ { i } = 1 )$ .

• False Positive $( \mathcal { F } _ { P } ) \colon$ : The question is unanswerable $( a _ { i } =$ 0), but the model supplies an answer $( z _ { i } = 0 )$

• False Negative $( \mathcal { F } _ { N } ) \colon$ The question is answerable $( a _ { i } =$ 1), but the model either abstains $( z _ { i } = 1 )$ ) or supplies an answer that fails EM $( z _ { i } = 0$ and $e _ { i } = 0 )$

To provide an end-to-end diagnostic of closed-context $\mathrm { Q A }$ evaluation, we structure SAGE-QA around three sequential evaluation stages. First, we interpret each raw answer field and retain an abstention flag before applying lexical normalization for Exact Match. We then use the flag and the EM result to partition the $N = 1 0 { , } 0 0 0$ benchmark examples into correctnessaware lexical outcome categories $( \mathcal { T } _ { P } , \mathcal { T } _ { N } , \mathcal { F } _ { P }$ , and $\mathcal { F } _ { N }$ , where $\mathcal { F } _ { N }$ is further split into false abstentions $\mathcal { F } _ { N } ^ { \emptyset }$ and answered non-matches $\mathcal { F } _ { N } ^ { \mathrm { a n s } } )$ . Second, we route eligible lexical failures concurrently to two separate LLM-as-a-judge diagnostics: ${ \mathrm { S e } } { \mathrm { - } }$ mantic Equivalence Scoring $( { \mathcal C } _ { S } = { \mathcal F } _ { N } ^ { \mathrm { a n s } } )$ and Re-Validation Scoring $( \mathcal { C } _ { R } = \mathcal { F } _ { P } \cup \mathcal { F } _ { N } ^ { \mathrm { a n s } } )$ . These are consistent with the fact that the SES algorithm assigns a score of 0 to $\mathcal { F } _ { P }$ and false abstentions, while RVS assigns a score of 0 to false abstentions but uses a scaled(gradually graded) approach for $\mathcal { F } _ { P }$ and $\mathcal { F } _ { N } ^ { \mathrm { a n s } }$ . Third, we aggregate both eligible case-level means and dataset-wide scores that credit $\mathcal { T } _ { P }$ and $\mathcal { T } _ { N }$ automatically while avoiding any double-counting on the overlapping set $\mathcal { F } _ { N } ^ { \mathrm { a n s } }$ These hybrid aggregates combine benchmark-matching credit with judge-assigned rubric scores. Table II summarizes the complete routing architecture.

![](images/a828fc8347ac1ffbfb337bf8f5262f154b74d25d1742adaf1e041c8de6ae0ef0.jpg)  
Fig. 1. SAGE-QA pipeline. SES and RVS overlap on answered FN cases; RVS is not conditioned on SES failure. TP/TN retain benchmark-based full credit without judging. False abstentions receive no judge assessment and zero credit in both diagnostic scores.

## A. Prediction Format and Exact Matching

For each evaluation example $i \in \{ 1 , \ldots , N \}$ , let $c _ { i }$ denote the context passage, $q _ { i }$ the question, $G _ { i }$ the set of human reference answers, and $a _ { i } \in \{ 0 , 1 \}$ the ground-truth answerability label. Each evaluated model supplies a raw answer (response). Before any lexical normalization, we set the abstention flag $z _ { i } = 1$ if this response is an empty list, the string [], an empty string, or a missing value; otherwise, we set $z _ { i } = 0$ . The scoring scripts retain the complementary flag pred\_is\_answerable, corresponding to $1 - z _ { i }$ . We define $\hat { y } _ { i }$ as the interpreted prediction text: the empty string when $z _ { i } = 1$ , and the extracted answer text otherwise. The flag z is retained separately and is never recomputed from normalized text.

The two diagnostics use separate requests to the same judge: SES receives only $( G _ { i } , \hat { y } _ { i } )$ , while RVS receives only $( c _ { i } , q _ { i } , \hat { y } _ { i } )$ . The prediction text supplied to the judge has not undergone SQuAD lexical normalization. Neither request includes the benchmark answerability flag $a _ { i }$

We apply standard SQuAD lexical normalization [26] by stripping articles, punctuation, and excess whitespace while lowercasing for only the Exact Match comparison of the interpreted prediction and reference answers. This operation does not alter the retained $z _ { i }$ or $\hat { y } _ { i }$ . In particular, a nonabstaining answer that becomes empty after normalization is not reclassified as an abstention. Let $n ( \cdot )$ denote this normalization function and let $\widetilde { G } _ { i }$ denote the set of normalized references for example i (where $\widetilde { G } _ { i } = \left\{ ^ {  \{ \omega \} } \right\}$ for unanswerable questions). Exact Match is then defined using the indicator function 1{·} maximized over $\widetilde { G } _ { i }$ :

TABLE II  
OVERVIEW OF THE SAGE-QA DIAGNOSTIC ROUTING PIPELINE.
<table><tr><td>Stage</td><td>Target Pool / Metric</td><td>Operational Definition and Inputs</td></tr><tr><td>1. Generation</td><td>All  $N = 1 0 , 0 0 0 \ \mathrm { i t e m s }$ </td><td>Model generates a raw answer field from  $( c _ { i } , q _ { i } )$  , using  $[ \ ]$  to indicate abstention.</td></tr><tr><td>2. Interpretation and Lexical Match</td><td> $\mathrm { E M } , \mathrm { F 1 } , \mathcal { T } _ { P } , \mathcal { T } _ { N } , \mathcal { F } _ { P } , \mathcal { F } _ { N }$ </td><td>Determine 15  $z _ { i }$  from the raw answer and retain interpreted text  ${ \hat { y } } _ { i } .$  Normalize only for EM; assign outcomes using  $a _ { i } , z _ { i } , e _ { i } ,$  and split  $\dot { \mathcal { F } } _ { N }$  using  $z _ { i } .$ </td></tr><tr><td>3a. SES Judge</td><td> $\mathcal { C } _ { S } = \mathcal { F } _ { N } ^ { \mathrm { a n s } }$ </td><td>External judge scores semantic equivalence  $s _ { i } \in [ 0 , 1 ]$  using only  $( G _ { i } , \hat { y } _ { i } )$ </td></tr><tr><td>3b. RVS Judge</td><td> $\mathcal { C } _ { R } = \dot { \mathcal { F } _ { P } } \cup \mathcal { F } _ { N } ^ { \mathrm { a n s } }$ </td><td>Same external judge scores passage support  $r _ { i } \in [ 0 , 1 ]$  (and binary acceptance at  $\tau = 0 . 9 5 )$  using only  $( c _ { i } , q _ { i } , \hat { y } _ { i } )$ </td></tr><tr><td>4. Aggregation</td><td> $\mathrm { S E S _ { a l l } , R V S _ { a l l } }$ </td><td>Report eligible case means and hybrid dataset-wide scores combining benchmark- based credit for  $\mathcal { T } _ { P }$  and  $\mathcal { T } _ { N }$  with judge-assigned credit, without double-counting overlapping  $\mathcal { F } _ { N } ^ { \mathrm { a n s } }$  cases.</td></tr></table>

$$
e _ { i } = \operatorname* { m a x } _ { g \in \widetilde { G } _ { i } } { \bf 1 } \{ n ( \hat { y } _ { i } ) = g \} , \qquad \mathrm { E M } = \frac { 1 0 0 } { N } \sum _ { i = 1 } ^ { N } e _ { i } .\tag{1}
$$

## B. Count-Based F1

Since $\mathcal { F } _ { N }$ incorporates both failure to answer and failure to match a reference lexically, we partition $\mathcal { F } _ { N }$ into a disjoint union:

$$
\mathcal F _ { N } = \mathcal F _ { N } ^ { \emptyset } \dot { \cup } \mathcal F _ { N } ^ { \mathrm { a n s } } ,\tag{2}
$$

where $\mathcal { F } _ { N } ^ { \emptyset } ~ = ~ \{ i ~ : ~ a _ { i } ~ = ~ 1 , ~ z _ { i } ~ = ~ 1 \}$ contains false abstentions on answerable questions and $\mathcal { F } _ { N } ^ { \mathrm { a n s } } = \{ i : a _ { i } =$ $1 , \ z _ { i } = 0 , \ e _ { i } = 0 \}$ contains answered non-matches. On our 10,000-example evaluation split, these categories satisfy the constraints:

$$
\left| \mathcal { T } _ { P } \right| + \left| \mathcal { F } _ { N } \right| = 4 , 9 9 0 , \quad \left| \mathcal { T } _ { N } \right| + \left| \mathcal { F } _ { P } \right| = 5 , 0 1 0 ,\tag{3}
$$

$$
\mathrm { E M } = 1 0 0 \frac { | \mathcal { T } _ { P } | + | \mathcal { T } _ { N } | } { N } .\tag{4}
$$

From these correctness-aware cardinalities, we compute decision-matrix precision (P), recall (R), and percentage F1 (F1).

$$
P = \frac { | \mathcal { T } _ { P } | } { | \mathcal { T } _ { P } | + | \mathcal { F } _ { P } | } , \qquad R = \frac { | \mathcal { T } _ { P } | } { | \mathcal { T } _ { P } | + | \mathcal { F } _ { N } | } ,\tag{5}
$$

$$
\mathrm { F 1 } = 1 0 0 \frac { 2 P R } { P + R } = 1 0 0 \frac { 2 | { \cal T } _ { P } | } { 2 | { \cal T } _ { P } | + | { \cal F } _ { P } | + | { \cal F } _ { N } | } .\tag{6}
$$

Unlike traditional SQuAD token-overlap ${ \mathrm { F 1 } } _ { \mathrm { r e p o r t e d } } .$ , which awards partial credit for sub-span lexical intersection on individual questions, our decision-matrix F1 in Eq. (6) evaluates corpus-level classification performance directly from the cardinalities $| \mathcal { T } _ { P } | , \ | \mathcal { F } _ { P } |$ , and $\vert \mathcal { F } _ { N } \vert$ . Under this formulation, both false abstentions $( { \dot { \mathcal { F } } } _ { N } ^ { \varnothing } )$ and non-matching answers $( \mathcal { F } _ { N } ^ { \mathrm { a n s } } )$ penalize recall equally through their combined cardinality $\vert \mathcal { F } _ { N } \vert$ , whereas True Negatives $( \mathcal { T } _ { N } )$ contribute to EM in Eq. (4) but are excluded from F1. Consequently, when a model achieves higher accuracy on unanswerable questions $( | \mathcal { T } _ { N } | )$ than on answerable ones $( | \mathcal { T } _ { P } | )$ , its decision-matrix F1 can be slightly lower than its EM—a structural property that we observe empirically across several evaluated models.

To characterize error trade-offs conditioned on ground-truth answerability, we also define the false positive rate (FP rate)

and false negative rate (FN rate) over their respective 5,010 and 4,990 denominators:

$$
\mathrm { F P ~ r a t e } = 1 0 0 \frac { \left| \mathcal { F } _ { P } \right| } { \left| \mathcal { F } _ { P } \right| + \left| \mathcal { T } _ { N } \right| } ,
$$

$$
\mathrm { F N ~ r a t e } = 1 0 0 { \frac { \left| { \mathcal { F } } _ { N } \right| } { \left| { \mathcal { T } } _ { P } \right| + \left| { \mathcal { F } } _ { N } \right| } } .\tag{7}
$$

## C. SES: Reference-Meaning Agreement

Semantic Equivalence Scoring (SES) asks the judge to assess whether an answered EM failure on an answerable question preserves the semantic meaning of at least one human reference. Consequently, we define its eligible evaluation pool strictly as the set of substantive non-matching predictions on answerable questions:

$$
{ \mathcal C } _ { S } = { \mathcal F } _ { N } ^ { \mathrm { a n s } } .\tag{8}
$$

For each example $i \in \mathcal { C } _ { S }$ , the external judge model θ receives only the human reference alternatives and the extracted answer, computing a graded equivalence score:

$$
s _ { i } = E _ { \theta } ^ { \mathrm { S E S } } ( G _ { i } , \hat { y } _ { i } ) \in [ 0 , 1 ] .\tag{9}
$$

We treat $s _ { i } ~ \in ~ [ 0 , 1 ]$ as score using a structured rubric as indicated below. Because the members of $G _ { i }$ represent alternative valid annotations rather than fragments that must be jointly reproduced, the judge evaluates $\hat { y } _ { i }$ against each reference $g \in G _ { i }$ independently and retains the maximum score: $s _ { i } ~ = ~ \operatorname* { m a x } _ { g \in G _ { i } } \operatorname { J U D G E } ( g , \hat { y } _ { i } )$ . We enforce a five-tier graded rubric:

• 1.00: Mutual semantic equivalence without material omission or addition.

• 0.95–0.99: Strong equivalence with only minor, nonessential phrasing differences.

• 0.80–0.94: Partial equivalence exhibiting substantial overlap alongside a meaningful omission or addition.

• 0.20–0.79: Weak agreement with major semantic divergences despite topically related content.

• 0.00: Unrelated or contradictory meaning.

Algorithm 1 formalizes the end-to-end SES evaluation procedure across all N benchmark instances.

Algorithm 1 Semantic Equivalence Scoring (SES) Procedure   
Require: For each example $\overline { { i \in \{ 1 , \ldots , N \} } }$ : lexical outcome label $\mathsf { \Gamma } _ { O _ { i } } \in \{ \mathsf { T P } , \mathsf { T N } , \mathsf { F P } , \mathsf { F N } \}$ , reference set $G _ { i } ,$ abstention flag z determined   
from the raw answer field before normalization, and interpreted prediction text ${ \hat { y } } _ { i } .$   
Ensure: Eligible case mean $\mathrm { S E S _ { c a s e s } }$ and dataset-wide score $\mathrm { \bar { S E S } _ { a l l } }$   
1: for each example $i = 1 , \ldots , N$ do   
2: i $\mathsf { \bar { \Pi } } o _ { i } \in \{ \mathsf { T } \bar { \mathsf { P } } , \mathsf { T N } \}$ then   
3: $s _ { i } \gets 1$ ▷ Automatic full credit for exact matches   
4: else if $o _ { i } = { \mathsf { F P } }$ or $z _ { i } = 1$ then   
5: $s _ { i } \gets 0$ ▷ Zero credit for $\mathcal { F } _ { P }$ and false abstentions $\mathcal { F } _ { N } ^ { \emptyset }$   
6: else ▷ Eligible answered EM failure $( i \in { \mathcal { C } } _ { S } = { \mathcal { F } } _ { N } ^ { \mathrm { a n s } } )$   
7: $s _ { i } \gets \operatorname* { m a x } _ { g \in G _ { i } }$ JUDGE $[ ( g , \hat { y } _ { i } )$ ▷ Graded reference-meaning equivalence in [0, 1]   
8: Resubmit request if $s _ { i }$ is missing or outside $[ 0 , 1 ]$ to enforce schema validity   
9: return $\begin{array} { r } { \mathrm { S E S } _ { \mathrm { c a s e s } } = \frac { 1 0 0 } { | \mathcal { C } _ { S } | } \sum _ { i \in \mathcal { C } _ { S } } \varepsilon } \end{array}$ and $\begin{array} { r } { \mathrm { S E S } _ { \mathrm { a l l } } = \frac { 1 0 0 } { N } \sum _ { i = 1 } ^ { N } s _ { i } } \end{array}$

Letting $\begin{array} { r } { \bar { s } = \frac { 1 } { | \mathcal { C } _ { S } | } \sum _ { i \in \mathcal { C } _ { S } } { s _ { i } } } \end{array}$ denote the unscaled mean over the eligible pool ${ \mathcal { C } } _ { S } ,$ , we express the case-level mean and the chained dataset-wide score, both on a 0–100 scale, as:

(10)

$$
\begin{array} { r l r } { \mathrm { S E S } _ { \mathrm { c a s e s } } = \displaystyle \frac { 1 0 0 } { | \mathcal { C } _ { S } | } \sum _ { i \in \mathcal { C } _ { S } } s _ { i } = 1 0 0 \bar { s } , } & \\ { \mathrm { S E S } _ { \mathrm { a l l } } = \displaystyle \frac { 1 0 0 } { N } \Big ( | \mathcal { T } _ { P } | + | \mathcal { T } _ { N } | + \sum _ { i \in \mathcal { C } _ { S } } s _ { i } \Big ) } & \\ { \displaystyle } & { = \mathrm { E M } + 1 0 0 \frac { | \mathcal { C } _ { S } | } { N } \bar { s } = \mathrm { E M } + \frac { | \mathcal { C } _ { S } | } { N } \mathrm { S E S } _ { \mathrm { c a s e s } } . } & \end{array}\tag{11}
$$

As shown in Eq. (11), exact matches $( \mathcal { T } _ { P }$ and $\mathcal { T } _ { N } )$ receive full credit (1.0), eligible non-matching answers $( \mathcal { F } _ { N } ^ { \mathrm { a n s } } )$ receive $s _ { i } ,$ and false positives $( \mathcal { F } _ { P } )$ as well as false abstentions $( \mathcal { F } _ { N } ^ { \emptyset } )$ receive 0.

The graded SES scores and the graded RVS scores defined below express mean assigned credit on a 0–100 scale. Multiplication by 100 rescales that credit; it does not convert partial credit into a count of correct answers. We therefore report graded results as scores, rather than percentages of correct answers, and differences between graded scores as score points. Percentages are retained for proportions of examples, including thresholded judge acceptance rates.

## D. RVS: Passage-Based Assessment

Re-Validation Scoring (RVS) asks the judge to assess whether a substantive model answer directly addresses the question and is fully supported by the passage, without consulting reference answers or benchmark answerability labels. We define its eligible evaluation pool as the union of all substantive predictions that fail EM:

$$
\mathcal { C } _ { R } = \mathcal { F } _ { P } \cup \mathcal { F } _ { N } ^ { \mathrm { a n s } } .\tag{12}
$$

For each example $i \in { \mathcal { C } } _ { R } ,$ the RVS judge receives exclusively the passage, question, and candidate answer, producing a passage-support score:

$$
r _ { i } = E _ { \theta } ^ { \mathrm { R V S } } ( c _ { i } , q _ { i } , \hat { y } _ { i } ) \in [ 0 , 1 ] .\tag{13}
$$

The judge is instructed to first assess from $c _ { i }$ alone whether $q _ { i }$ is answerable and what evidence $c _ { i }$ provides; it then evaluates $\hat { y } _ { i }$ jointly for groundedness (every claim is stated or clearly implied by $c _ { i } ,$ without external knowledge or contradiction) and completeness (the response directly and fully answers $q _ { i } )$ . RVS assigns scores using the same numerical bands as

SES (1.00 fully grounded and complete; 0.95–0.99 strong; 0.80–0.94 partial; 0.20–0.79 weak; 0.00 unsupported, contradicted, or off-question). For binary acceptance analysis, we pre-declare a stringent threshold $\tau ~ = ~ 0 . 9 5$ and define $r _ { i } ^ { \tau } = \mathbf { 1 } \{ r _ { i } \geq \tau \}$ . Algorithm 2 summarizes the complete RVS evaluation workflow.

Letting $\begin{array} { r } { \bar { r } = \frac { 1 } { \vert \mathcal { C } _ { R } \vert } \sum _ { i \in \mathcal { C } _ { R } } r _ { i } } \end{array}$ denote the unscaled mean over the eligible pool ${ \mathcal { C } } _ { R } .$ , we write the judge mean on eligible cases and the hybrid dataset-wide score, both on a 0–100 scale, as:

(14)

$$
\begin{array} { r l } & { \displaystyle \mathrm { R V S } _ { \mathrm { c a s e s } } = \frac { 1 0 0 } { | \mathcal { C } _ { R } | } \sum _ { i \in \mathcal { C } _ { R } } r _ { i } = 1 0 0 \bar { r } , } \\ & { \displaystyle \mathrm { R V S } _ { \mathrm { a l l } } = \frac { 1 0 0 } { N } \Big ( | \mathcal { T } _ { P } | + | \mathcal { T } _ { N } | + \sum _ { i \in \mathcal { C } _ { R } } r _ { i } \Big ) } \\ & { \quad \quad \quad \quad = \mathrm { E M } + 1 0 0 \frac { | \mathcal { C } _ { R } | } { N } \bar { r } = \mathrm { E M } + \frac { | \mathcal { C } _ { R } | } { N } \mathrm { R V S } _ { \mathrm { c a s e s } } , } \end{array}\tag{15}
$$

where the binary acceptance counterparts replace $r _ { i }$ with $r _ { i } ^ { \tau }$ False abstentions $( \mathcal { F } _ { N } ^ { \emptyset } )$ lie outside $\mathcal { C } _ { R }$ and receive zero credit in both the graded and binary dataset-wide RVS scores.

The binary aggregate $\mathrm { R V S } _ { \mathrm { a l l } } ^ { \tau }$ is the percentage of all N examples credited by the combined benchmark-and-judge rules: benchmark $\mathcal { T } _ { P }$ and $\mathcal { T } _ { N }$ cases receive credit automatically, and eligible cases receive credit when $r _ { i } ~ \geq ~ \tau .$ . It is not an independently verified percentage of correct answers. The eligible-pool acceptance rate, in contrast, uses $| { \mathcal { C } } _ { R } |$ as its denominator and counts only thresholded judge decisions.

Throughout the RVS analysis, “accepted” means that a judge score meets the specified threshold $\tau = 0 . 9 5$ . Scores below that threshold are described as “below threshold,” not automatically as “unsupported.” Because the rubric evaluates both passage support and completeness, a response can fall below the threshold for reasons other than lack of support. The lowest score band $( r _ { i } < 0 . 2 0 )$ is reported separately from all below-threshold responses and does not establish a verified hallucination category.

Here, reference-free describes only the inputs to individual RVS judge calls for $i \in { \mathcal { C } } _ { R } .$ . Benchmark reference answers and answerability labels determine the eligible pool $\mathcal { C } _ { R }$ and the outcome categories used for automatic credit. Consequently, $\mathrm { R V S } _ { \mathrm { c a s e s } }$ is a judge mean conditional on a benchmark-selected pool, and $\mathrm { R V S } _ { \mathrm { a l l } }$ is a hybrid dataset-wide score combining benchmark-based credit for $\mathcal { T } _ { P }$ and $\mathcal { T } _ { N }$ with judge-assigned credit on eligible cases. The $\mathcal { T } _ { P }$ and $\mathcal { T } _ { N }$ cases receive full credit without separate assessment by the RVS judge. This benchmark dependence also applies to the binary aggregate $\mathrm { R V S } _ { \mathrm { a l l } } ^ { \tau } ;$ excluding references from judge inputs does not make case selection or dataset-wide aggregation reference-free.

Algorithm 2 Re-Validation Scoring (RVS) Procedure   
Require: For each example $i \in \{ 1 , \ldots , N \}$ : lexical outcome label $o _ { i } ,$ passage $c _ { i } ,$ question $q _ { i } ,$ abstention flag $z _ { i }$ determined from the raw   
answer field before normalization, and interpreted prediction text $\hat { y } _ { i }$   
Ensure: Judge mean on eligible cases $\mathrm { R V S } _ { \mathrm { c a s e s } }$ and hybrid dataset-wide score $\mathrm { R V S } _ { \mathrm { a l l } }$   
1: for each example $i = 1 , \bar { \ldots } , N$ do   
2: if $\mathsf { \bar { \Pi } } _ { O _ { i } } \in \{ \mathsf { T } \bar { \mathsf { P } } , \mathsf { T N } \}$ then   
3: $r _ { i } \gets 1$ ▷ Automatic full credit for exact matches   
4: else if $z _ { i } = 1$ then ▷ False abstention on answerable question $( i \in \mathcal { F } _ { N } ^ { \varnothing } )$   
5: $r _ { i } \gets 0$ ▷ Zero credit for false abstentions $\dot { \mathcal { F } } _ { N } ^ { \varnothing } ;$ ; no judge call   
6: else ▷ Eligible substantive EM failure $( i \in \mathcal { C } _ { R } = \mathcal { F } _ { P } \cup \mathcal { F } _ { N } ^ { \mathrm { a n s } } )$   
7: $r _ { i } \gets \mathrm { J U D G E } \left( c _ { i } , q _ { i } , \hat { y } _ { i } \right)$ ▷ Graded passage groundedness and completeness in [0, 1]   
8: Resubmit request if $r _ { i }$ is missing or outside $[ 0 , 1 ]$ to enforce schema validity   
9: return $\begin{array} { r } { \mathrm { R V S } _ { \mathrm { c a s e s } } = \frac { 1 0 0 } { \left| \mathcal { C } _ { R } \right| } \sum _ { i \in \mathcal { C } _ { R } } r _ { i } } \end{array}$ and $\begin{array} { r } { \mathrm { R V S } _ { \mathrm { a l l } } = \frac { 1 0 0 } { N } \sum _ { i = 1 } ^ { N } r _ { i } } \end{array}$

## IV. EXPERIMENTAL SETUP

## A. Dataset Partition and Model Cohorts

We conduct our primary evaluation on a fixed subset of $N = 1 0 { , } 0 0 0$ examples from the public SQuAD 2.0 validation dataset [12], comprising 4,990 answerable and 5,010 unanswerable questions. This held-out evaluation set is shared identically across all zero-shot and fine-tuned models. For our fine-tuning experiments, we train on a separate, balanced subset of 87,000 examples drawn exclusively from the official SQuAD 2.0 training data, ensuring zero data leakage into the 10,000-example evaluation set.

We evaluate three distinct model cohorts:

• Zero-Shot Frontier Group: Claude Opus 4.6 [27] and Gemini 3.1 Pro [28]. For this cohort, we retain full perexample prediction logs, FN subgroup partitions $( \mathcal { F } _ { N } ^ { \emptyset }$ vs. $\mathcal { F } _ { N } ^ { \mathrm { a n s } } )$ , and individual judge scores $( s _ { i } , r _ { i } )$

• Zero-Shot Open-Weight Group (Untuned Baselines): Meta-Llama-3-8B-Instruct [29], Llama-3.3-70B-Instruct [30], and gpt-oss-20b [31]. Evaluating these models out-of-the-box establishes a direct zero-shot comparison against the closed-weight frontier models and provides a matched untuned baseline for our fine-tuning experiments, retaining full outcome counts and FN subgroup splits $( \mathcal { F } _ { N } ^ { \emptyset } \mathrm { v s . } \mathcal { F } _ { N } ^ { \mathrm { a n s } } )$ under reference-based scoring.

• Fine-Tuned Open-Weight Group: Llama 3 8B [29], Llama 3.3 70B [30], and gpt-oss-20b [31]. Because reference-based confusion-matrix counts suffice to quantify exact lexical adaptation gains over our matched untuned baselines, our analysis of the fine-tuned cohort focuses on reference-based metrics (EM, F1, and category error rates), whereas per-example SES and RVS judge audits are restricted to the zero-shot frontier group.

## B. Prompt Optimization and Zero-Shot Execution Protocol

Prior to full-scale benchmarking, we conducted a preliminary prompt-optimization study comparing three humanauthored prompts, an OpenAI-optimized prompt [22], and prompts compiled programmatically via DSPy MIPROv2 [19], [20] and GEPA [21]; because the DSPy GEPA optimizer achieved the highest pilot performance (69.60 EM and 80.70 $\mathrm { F 1 } _ { \mathrm { r e p o r t e d } }$ against SQuAD references), we standardized on this GEPA-compiled prompt across all subsequent zero-shot evaluations.

TABLE III  
HYPERPARAMETER CONFIGURATIONS FOR FINE-TUNED OPEN-WEIGHT MODELS (ALL TRAINED FOR 1 EPOCH).
<table><tr><td>Model</td><td>LoRA Rank (r)</td><td>α</td><td>Learning Rate</td><td>Batch Size</td><td>Max Seq. Len.</td></tr><tr><td>Llama 3 8B</td><td>128</td><td>32</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>32</td><td>512</td></tr><tr><td>gpt-oss-20b</td><td>128</td><td>32</td><td> $3 \times 1 0 ^ { - 5 }$ </td><td>64</td><td>512</td></tr><tr><td>Llama 3.3 70B</td><td>32</td><td>64</td><td> $1 \times 1 0 ^ { - 5 }$ </td><td>60*</td><td>1,024</td></tr></table>

<sup>∗</sup>Effective batch size via gradient accumulation.

In our main zero-shot benchmark, Claude Opus 4.6 and Gemini 3.1 Pro process all 10,000 passage-and-question pairs via their respective provider Batch APIs [32], [33]. Similarly, the untuned open-weight models—Llama 3 8B Instruct (via Hugging Face/Featherless-AI), Llama 3.3 70B Instruct (via Hugging Face/OVHCloud), and gpt-oss-20b (via Groq)—are evaluated zero-shot across the exact same 10,000 examples. Each model is constrained via structured-output schemas to return a strict two-field JSON object containing the example identifier and the extracted answer (or []). All 10,000 requests per model across all five zero-shot runs succeeded and parsed cleanly.

As an implementation-consistency check, we examined all 20,000 answers from the Claude Opus 4.6 and Gemini 3.1 Pro. Both scoring scripts determine abstention from the raw answer before applying normalization for EM, and the check found no misclassified abstentions or outcome assignments.

## C. Fine-Tuning Protocol and Judge Execution

We adapted the three open-weight models for one epoch over the 87,000-example training split using assistant-response cross-entropy supervision and the 8-bit AdamW optimizer. Llama 3 8B was fine-tuned via LoRA with Unsloth [23], [34]; gpt-oss-20b used 4-bit quantization with LoRA; and Llama 3.3 70B employed QLoRA with 4-bit NF4 quantization, BF16 compute, rank-stabilized LoRA [24], cosine learningrate scheduling, a warmup ratio of 0.03, and weight decay of 0.01. Table III summarizes the hyperparameter configurations.

![](images/5edcf05c7a088524f1b383808a8e7215c47b312b2ca910398d4b9ea0c47f72d2.jpg)  
Fig. 2. Zero-shot frontier outcome matrices. The upper row contains answerable examples (TP, FN); the lower row contains unanswerable examples (FP, TN). FN comprises false abstentions and non-matching answers (Claude 165 + 888; Gemini 171 + 599).

<table><tr><td rowspan=1 colspan=8>Llama 3 8B (ZS)       Llama 3.3 70B (ZS)       gpt-oss-20b (ZS)</td></tr><tr><td rowspan=1 colspan=1>TP3,236</td><td rowspan=1 colspan=1>FN1,754</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TP3,576</td><td rowspan=1 colspan=1>FN1,414</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TP3,610</td><td rowspan=1 colspan=1>FN1,380</td></tr><tr><td rowspan=1 colspan=1>FP2,558</td><td rowspan=1 colspan=1>TN2,452</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>FP1,447</td><td rowspan=1 colspan=1>TN3,563</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>FP1,992</td><td rowspan=1 colspan=1>TN3,018</td></tr></table>

Fig. 3. Zero-shot untuned open-weight outcome matrices. Without fine-tuning, open-weight models exhibit high FP counts on unanswerable questions and high non-matching $\mathcal { F } _ { N } ^ { \mathrm { a n s } }$ counts (1,647, 1,294, and 1,268) alongside low false abstentions $( \mathcal { F } _ { N } ^ { \emptyset }$ of 107, 120, and 112).

For the SAGE-QA diagnostic phase on the zero-shot frontier cohort, we executed gpt-6-luna [5] at medium reasoning effort via the OpenAI Batch API [35], enforcing a strict twofield JSON response (identifier and numerical score in [0, 1]). Across both models, the eligible diagnostic pools comprised $| { \mathcal { C } } _ { S } | = 8 8 8$ and $| { \mathcal { C } } _ { R } | = 1 { , } 3 1 4$ cases for Claude Opus 4.6, and $| { \mathcal { C } } _ { S } | = 5 9 9$ and $| \mathcal { C } _ { R } | = 1 , 1 6 3$ cases for Gemini 3.1 Pro. Together, these SES and RVS analyses comprise 3,964 judge API requests, all of which returned valid in-range scores.

## V. EXPERIMENTAL RESULTS AND DIAGNOSTIC ANALYSIS

## A. Baseline Reference-Based Performance

Table IV summarizes the reference-based outcomes for all eight configurations on the same 10,000 examples. We highlight three findings; the table provides the complete counts, error rates, and scores.

Gemini’s higher EM comes with more answers on unanswerable questions. Relative to Claude, Gemini produces 283 additional exact matches on answerable questions but also answers 138 more questions designated unanswerable by SQuAD. The net advantage is 145 exact matches, or 1.45 EM percentage points (Fig. 2). False abstentions are similar between the models; most of the difference on answerable questions lies in substantive answers that fail lexical matching. Their EM comparison therefore reflects a trade-off between reference matching and abstention.

Untuned, open-weight models frequently substantively answer when the benchmark expects abstention. Their FP rates range from 28.88% to 51.06%, compared with 8.50%– 11.26% for the frontier models. On answerable questions, their failures predominantly involve nonmatching answers rather than abstentions (Fig. 3). These two patterns account for their lower EM, but lexical counts alone cannot distinguish paraphrases from substantive answer errors. We have not applied SES/RVS to this cohort, so the semantic composition of these failures remains unresolved.

<table><tr><td rowspan=1 colspan=8>Llama 3 8B (FT)      Llama 3.3 70B (FT)      gpt-oss-20b (FT)</td></tr><tr><td rowspan=1 colspan=1>TP4,681</td><td rowspan=1 colspan=1>FN309</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TP4,684</td><td rowspan=1 colspan=1>FN306</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TP4,039</td><td rowspan=1 colspan=1>FN951</td></tr><tr><td rowspan=1 colspan=1>FP597</td><td rowspan=1 colspan=1>TN4,413</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>FP270</td><td rowspan=1 colspan=1>TN4,740</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>FP302</td><td rowspan=1 colspan=1>TN4,708</td></tr></table>

Fig. 4. Fine-tuned open-weight outcome matrices, with the same row and category definitions as Figs. 2 and 3. Every upper row sums to 4,990 and every lower row to 5,010.

Fine-tuning improves both categories of benchmark outcomes. Across the three open-weight models, adaptation increases EM by 21.19–34.06 percentage points and reduces both FP and FN counts (Fig. 4). The two fine-tuned Llama models illustrate the importance of abstention: they achieve almost identical TP counts, yet the 70B model correctly abstains on 327 more unanswerable questions. This accounts for nearly all of its EM advantage over the 8B model. These comparisons establish improved reference-based outcomes on our evaluation subset.

## B. SAGE-QA Diagnostic Results: SES and RVS

We apply the judge-based diagnostics to the two zero-shot frontier models. Table V reports case means and dataset-wide scores using Eqs. (11) and (15); Table VI shows the underlying score distributions.

Reference-equivalence scoring reverses the EM ranking. Although Gemini leads under EM, Claude receives the higher dataset-wide SES score: 92.81 versus 91.67. Claude has both a larger eligible pool and a higher mean SES score within that pool. Reference-equivalence credit therefore changes the comparison, demonstrating that the lexical ranking is not preserved under this grading criterion.

Passage-based scoring produces very similar aggregate scores. The graded dataset-wide RVS scores are 97.96 for Claude and 98.02 for Gemini, a gap of 0.06 score points. Binary acceptance at $\tau = 0 . 9 5$ slightly reverses that ordering. These hybrid scores combine automatic benchmark credit with judge-assigned credit; their proximity does not establish equal answer correctness. Rather, the comparison depends on the evaluation criterion and treatment of partial credit.

The score distributions explain the difference between SES and RVS. On answered FN cases, partial equivalence is the most frequent SES category, whereas full credit is the most frequent RVS category (Table VI). ${ \mathrm { A t ~ } } \tau = 0 . 9 5$ , SES accepts 35.5% of Claude’s and 41.2% of Gemini’s answered nonmatches; RVS accepts 95.3% and 94.0%, respectively. RVS also accepts over 90% of each model’s answers to questions designated unanswerable by SQuAD. These findings document judge disagreement with lexical matching and reference annotations, without independently resolving it. Scores below threshold can reflect incomplete answers as well as lack of passage support, so they should not all be described as unsupported.

## C. Paired SES/RVS Decision Comparison

The distributions above describe each diagnostic separately. Table VII compares their decisions on the same answered FN responses. RVS alone accepts 536 Claude responses and 321 Gemini responses, whereas SES alone accepts only five for each model. Thus, the main disagreement consists of responses receiving high passage-support credit without high referenceequivalence credit. Both assessments come from the same judge under different inputs and rubrics; this pattern identifies a difference between the criteria, not independent confirmation of answer correctness.

TABLE IV  
UNIFIED BASELINE RESULTS AND CATEGORY ERROR RATES ON 10,000 SQUAD 2.0 VALIDATION EXAMPLES.
<table><tr><td>Cohort</td><td>Model</td><td>TP</td><td>FN  $( \mathcal { F } _ { N } ^ { 0 } / \mathcal { F } _ { N } ^ { \mathrm { a n s } } )$ </td><td>FP</td><td>TN</td><td>FP Rate (%)</td><td>FN Rate (%)</td><td>EM</td><td>F1</td></tr><tr><td>Zero-Shot (Frontier)</td><td>Claude Opus 4.6</td><td>3,937</td><td>1,053 (165 / 888)</td><td>426</td><td>4,584</td><td>8.50</td><td>21.10</td><td>85.21</td><td>84.19</td></tr><tr><td>Zero-Shot (Frontier)</td><td>Gemini 3.1 Pro</td><td>4,220</td><td>770 (171 / 599)</td><td>564</td><td>4,446</td><td>11.26</td><td>15.43</td><td>86.66</td><td>86.35</td></tr><tr><td>Zero-Shot (Open-Weight)</td><td>Llama 3 8B (Untuned)</td><td>3,236</td><td>1,754 (107 / 1,647)</td><td>2,558</td><td>2,452</td><td>51.06</td><td>35.15</td><td>56.88</td><td>60.01</td></tr><tr><td>Zero-Shot (Open-Weight)</td><td>Llama 3.3 70B (Untuned)</td><td>3,576</td><td>1,414 (120 / 1,294)</td><td>1,447</td><td>3,563</td><td>28.88</td><td>28.34</td><td>71.39</td><td>71.43</td></tr><tr><td>Zero-Shot (Open-Weight)</td><td>gpt-oss-20b (Untuned)</td><td>3,610</td><td>1,380 (112 / 1,268)</td><td>1,992</td><td>3,018</td><td>39.76</td><td>27.66</td><td>66.28</td><td>68.16</td></tr><tr><td>Fine-Tuned (Open-Weight)</td><td>Llama 3 8B (Fine-Tuned)</td><td>4,681</td><td>309 (→)</td><td>597</td><td>4,413</td><td>11.92</td><td>6.19</td><td>90.94</td><td>91.18</td></tr><tr><td>Fine-Tuned (Open-Weight)</td><td>Llama 3.3 70B (Fine-Tuned)</td><td>4,684</td><td>306 (→)</td><td>270</td><td>4,740</td><td>5.39</td><td>6.13</td><td>94.24</td><td>94.21</td></tr><tr><td>Fine-Tuned (Open-Weight)</td><td>gpt-oss-20b (Fine-Tuned)</td><td>4,039</td><td>951 (→)</td><td>302</td><td>4,708</td><td>6.03</td><td>19.06</td><td>87.47</td><td>86.57</td></tr></table>

TABLE V

JUDGE-BASED SES AND RVS DIAGNOSTIC SUMMARIES FOR THE ZERO-SHOT FRONTIER COHORT (τ = 0.95).
<table><tr><td>Diagnostic Metric</td><td>Claude Opus 4.6</td><td>Gemini 3.1 Pro</td></tr><tr><td>Baseline EM (Table IV)</td><td>85.21</td><td>86.66</td></tr><tr><td>Eligible SES Pool  $| \mathcal { C } _ { S } | = | \mathcal { F } _ { N } ^ { \mathrm { a n s } } |$ </td><td>888</td><td>599</td></tr><tr><td> $\mathrm { S E S _ { c a s e s } }$  (Mean, 0–100)</td><td>85.56</td><td>83.68</td></tr><tr><td> $\mathrm { S E S _ { a l l } }$  (Graded Dataset-Wide, 0–100)</td><td>92.81</td><td>91.67</td></tr><tr><td>Eligible RVS Pool  $| { \mathcal { C } } _ { R } | = | { \mathcal { F } } _ { P } \cup { \mathcal { F } } _ { N } ^ { \mathrm { a n s } } |$ </td><td>1,314</td><td>1,163</td></tr><tr><td> $\mathrm { R V S } _ { \mathrm { c a s e s } }$  (Judge Mean on Eligible Cases, 0–100)</td><td>97.01</td><td>97.66</td></tr><tr><td> $\mathrm { R V S } _ { \mathrm { a l l } }$  (Hybrid Dataset-Wide, Graded, 0–100)</td><td>97.96</td><td>98.02</td></tr><tr><td>RVS Binary Acceptance Rate at  $\tau = 0 . 9 5 ~ ( \% )$ </td><td>93.68</td><td>92.86</td></tr><tr><td> $\mathrm { R V S } _ { \mathrm { a l l } } ^ { \tau }$  (Hybrid Dataset-Wide, Binary Acceptance, %)</td><td>97.52</td><td>97.46</td></tr></table>

TABLE VI

DISTRIBUTION OF JUDGE SCORE BANDS ACROSS ELIGIBLE DIAGNOSTIC POOLS.
<table><tr><td>Evaluated Pool (n)</td><td>1.00</td><td>0.95- 0.99</td><td>0.80- 0.94</td><td>0.20- 0.79</td><td> $< \mathbf { 0 . 2 0 }$ </td></tr><tr><td>Claude  $O p u s ~ 4 . 6$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SES on  $\mathcal { C } _ { S } ^ { \phantom { \dagger } } = \mathcal { F } _ { N } ^ { \mathrm { a n s } }$  (888)</td><td>96</td><td>219</td><td>493</td><td>50</td><td>30</td></tr><tr><td>RVS on  $\mathcal { F } _ { P } ~ ( 4 2 \dot { 6 } )$ </td><td>373</td><td>12</td><td>18</td><td>3</td><td>20</td></tr><tr><td>RVS on  $\mathcal { F } _ { N } ^ { \mathrm { a n s } }$  (888)</td><td>825</td><td>21</td><td>30</td><td>6</td><td>6</td></tr><tr><td>Gemini  $3 . I ~ P r o$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SES on  $\mathcal { C } _ { S } = \mathcal { F } _ { N } ^ { \mathrm { a n s } }$  (599)</td><td>87</td><td>160</td><td>275</td><td>52</td><td>25</td></tr><tr><td>RVS on  $\bar { \mathcal { F } } _ { P } ~ ( 5 6 \dot { 4 } )$ </td><td>496</td><td>21</td><td>31</td><td>4</td><td>12</td></tr><tr><td>RVS on  $\mathcal { F } _ { N } ^ { \mathrm { a n s } }$  (599)</td><td>546</td><td>17</td><td>30</td><td>5</td><td>1</td></tr></table>

## VI. DISCUSSION AND LIMITATIONS

## A. Evaluation Disagreement Beyond Lexical Matching

Lexical failure does not identify the type of failure. An EM failure establishes that a response does not match the reference annotations, but does not explain why. By separating reference equivalence from passage support, SES and RVS distinguish how the judge assesses these responses under two different criteria. Their paired decisions (Table VII) reveal distinctions that a lexical score alone cannot express.

TABLE VII  
PAIRED JUDGE DECISIONS ON $\mathcal { F } _ { N } ^ { \mathrm { a n s } }$ AT $\tau = 0 . 9 5 .$
<table><tr><td>Diagnostic Category</td><td>Claude Opus 4.6</td><td>Gemini 3.1 Pro</td></tr><tr><td>Paired Decisions on Answered FN Cases  $( \mathcal { F } _ { N } ^ { \mathrm { a n s } } )$ </td><td></td><td></td></tr><tr><td>Accepted by Both  $( \mathrm { S E S } \geq 0 . 9 5 \land \mathrm { R V S } \geq 0 . 9 \dot { 5 } )$ </td><td>310</td><td>242</td></tr><tr><td> $\mathrm { A c c e p t e d \ b y \ S E S \ O n l y }$ </td><td>5</td><td>5</td></tr><tr><td> $( \mathrm { S E S } \geq 0 . 9 5 \land \mathrm { R V S } < 0 . 9 5 )$   $\mathbf { A c c e p t e d b y R V S O n l y }$ </td><td>536</td><td>321</td></tr><tr><td> $( \mathrm { S E S } < 0 . 9 5 \land \mathrm { R V S } \geq 0 . 9 5 )$  Accepted by Neither  $( \mathrm { S E S } < 0 . 9 5 \land \mathrm { R V S } < 0 . 9 5 )$ </td><td>37</td><td>31</td></tr></table>

Model comparisons depend on what we measure. The ranking changes in Table V show that the evaluation criterion affects the comparison. Dataset-wide diagnostic scores also retain automatic credit for benchmark successes. Consequently, a higher score or a smaller gap under SES or RVS does not, by itself, establish greater answer correctness or equivalence between models.

Disagreements identify cases for examination. We use these diagnostics to locate responses for closer review while preserving the original benchmark outcomes. Their practical value is to make disagreements visible and organize subsequent examination.

## B. Limitations of our Research

Our semantic-equivalence and passage-support assessments rely on one external LLM judge (gpt-6-luna) and therefore we interpret SES and RVS as judge-based diagnostic scores rather than independently verified measures of answer correctness. Responses accepted despite failing benchmark criteria identify cases for further examination, but do not by themselves establish annotation errors or model hallucination rates. Using a judge that generated none of the evaluated responses avoids direct self-evaluation [4], [25]. Moreover, while our untuned-versus-fine-tuned comparisons isolate the exact lexical gains of parameter-efficient adaptation across all three open-weight models, we executed per-example SES and RVS assessments only on the zero-shot closed-weight frontier cohort. Extending these diagnostics to the untuned open-weight models (particularly their 1,268–1,647 $\mathcal { F } _ { N } ^ { \mathrm { a n s } }$ cases and 1,447–2,558 $\mathcal { F } _ { P }$ cases), together with independent human adjudication, would help examine the contributions of formatting divergence and comprehension errors to their lexical deficit that we defer to future studies as mentioned below.

## VII. CONCLUSION

In this paper, we presented SAGE-QA, a diagnostic evaluation framework for closed-context question answering that combines strict lexical matching (EM and decision-matrix F1) with judge-assessed reference-meaning agreement (SES) and passage-based assessment (RVS). Evaluating eight configurations across 10,000 SQuAD 2.0 validation examples, we demonstrated two complementary findings. First, under strict lexical evaluation, zero-shot closed-weight frontier models (Gemini 3.1 Pro at 86.66 EM and Claude Opus 4.6 at 85.21 EM) outperform untuned open-weight models (56.88 to 71.39 EM) by 15 to 30 EM points due to severe untuned over-answering on unanswerable queries; however, singleepoch parameter-efficient fine-tuning yields gains of +21.19 to +34.06 EM points, making the fine-tuned models excel in performance over prior results. Second, when we audit zero-shot frontier EM failures with an external LLM judge under SAGE-QA, the graded dataset-wide SES scores (on a 0–100 scale) are 92.81 (Claude Opus 4.6) and 91.67 (Gemini 3.1 Pro), and the corresponding hybrid dataset-wide RVS scores are 97.96 and 98.02. These RVS aggregates combine benchmark-based full credit for $\mathcal { T } _ { P }$ and $\mathcal { T } _ { N }$ with passage-based judge scores on eligible cases. The graded RVS gap is 0.06 score points under this protocol, and the judge accepts over 90% of each model’s benchmark false positives. These results characterize how evaluation criteria affect scores, rankings, and assessments of non-matching responses while preserving the benchmark labels. In future work, we plan to calibrate SES and RVS against human adjudication across multiple judge ensembles and extend full per-example diagnostic auditing to both untuned and fine-tuned open-weight architectures.

## REFERENCES

[1] P. Rajpurkar, R. Jia, and P. Liang, “Know what you don’t know: Unanswerable questions for SQuAD,” in Proc. ACL, 2018, pp. 784– 789. doi: 10.18653/v1/P18-2124.

[2] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “SQuAD: 100,000+ questions for machine comprehension of text,” in Proc. EMNLP, 2016, pp. 2383–2392. doi: 10.18653/v1/D16-1264.

[3] Stanford NLP Group, “The Stanford Question Answering Dataset: SQuAD 2.0 leaderboard.” Online leaderboard.

[4] A. Panickssery, S. R. Bowman, and S. Feng, “LLM evaluators recognize and favor their own generations,” in Advances in Neural Information Processing Systems, vol. 37, pp. 68772–68802, 2024. arXiv:2404.13076.

[5] OpenAI, “gpt-6-luna,” OpenAI Platform Model Documentation, 2026. Model documentation.

[6] M. Seo, A. Kembhavi, A. Farhadi, and H. Hajishirzi, “Bidirectional attention flow for machine comprehension,” in Proc. ICLR, 2017. arXiv:1611.01603.

[7] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in Proc. NAACL-HLT, 2019, pp. 4171–4186. doi: 10.18653/v1/N19-1423.

[8] Y. Liu et al., “RoBERTa: A robustly optimized BERT pretraining approach,” 2019. arXiv:1907.11692.

[9] E. Kamalloo, N. Dziri, C. L. A. Clarke, and D. Rafiei, “Evaluating open-domain question answering in the era of large language models,” in Proc. ACL, 2023. arXiv:2305.06984.

[10] J. Bulian, C. Buck, W. Gajewski, B. Börschinger, and T. Schuster, “Tomayto, Tomahto. Beyond token-level answer equivalence for question answering evaluation,” in Proc. EMNLP, 2022, pp. 291–305. doi: 10.18653/v1/2022.emnlp-main.20.

[11] V. Adlakha et al., “Evaluating correctness and faithfulness of instructionfollowing models for question answering,” Transactions of the Association for Computational Linguistics, vol. 12, pp. 681–699, 2024. doi: 10.1162/tacl\_a\_00667.

[12] Hugging Face, “rajpurkar/squad\_v2.” Dataset repository.

[13] T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, “BERTScore: Evaluating text generation with BERT,” in Proc. ICLR, 2020. arXiv:1904.09675.

[14] Y. Liu et al., “G-Eval: NLG evaluation using GPT-4 with better human alignment,” in Proc. EMNLP, 2023. arXiv:2303.16634.

[15] L. Zheng et al., “Judging LLM-as-a-judge with MT-Bench and Chatbot Arena,” in Advances in Neural Information Processing Systems, vol. 36, 2023. arXiv:2306.05685.

[16] C. G. Northcutt, A. Athalye, and J. Mueller, “Pervasive label errors in test sets destabilize machine learning benchmarks,” in Proc. NeurIPS Datasets and Benchmarks, 2021. arXiv:2103.14749.

[17] S. Gururangan et al., “Annotation artifacts in natural language inference data,” in Proc. NAACL-HLT, 2018, pp. 107–112. doi: 10.18653/v1/ N18-2017.

[18] H. Bahak, F. Taheri, Z. Zojaji, and A. Kazemi, “Evaluating ChatGPT as a question answering system: A comprehensive analysis and comparison with existing models,” 2023. arXiv:2312.07592.

[19] O. Khattab et al., “DSPy: Compiling declarative language model calls into self-improving pipelines,” in Proc. ICLR, 2024. arXiv:2310.03714.

[20] K. Opsahl-Ong et al., “Optimizing instructions and demonstrations for multi-stage language model programs,” in Proc. EMNLP, 2024, pp. 9340–9366. arXiv:2406.11695.

[21] L. A. Agrawal et al., “GEPA: Reflective prompt evolution can outperform reinforcement learning,” 2025. arXiv:2507.19457.

[22] OpenAI, “GPT-5 prompt migration and improvement using the new optimizer,” OpenAI Cookbook, 2025. Online resource.

[23] E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in Proc. ICLR, 2022. arXiv:2106.09685.

[24] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient finetuning of quantized LLMs,” in Advances in Neural Information Processing Systems, vol. 36, 2023. arXiv:2305.14314.

[25] P. Wang et al., “Large language models are not fair evaluators,” in Proc. ACL, 2024, pp. 9440–9450. doi: 10.18653/v1/2024.acl-long.511.

[26] Hugging Face, “SQuAD v2 metric,” Evaluate. Metric documentation.

[27] Anthropic, “Introducing Claude Opus 4.6,” 2026. Model announcement.

[28] Google Cloud, “Gemini 3.1 Pro,” Vertex AI Generative AI Documentation, 2026. Model documentation.

[29] Meta, “Meta-Llama-3-8B-Instruct,” 2024. Model card.

[30] Meta, “Llama-3.3-70B-Instruct,” 2024. Model card.

[31] OpenAI, “gpt-oss: Open-weight models,” 2025. Official repository.

[32] Anthropic, “Message Batches API,” Claude API Documentation. Batchprocessing documentation.

[33] Google, “Batch API,” Gemini API Documentation. Batch API documentation.

[34] Unsloth AI, “Fine-tuning LLMs guide.” Online documentation

[35] OpenAI, “Batch API,” OpenAI Platform Documentation. Batch API documentation.