# PsyJev: Psychological Analysis via System-1 Judgment

Fangyuan Liu<sup>1</sup>, Sirui Zhao<sup>1,\*</sup>, Xiaohao Wang<sup>1</sup>, Guo Cheng<sup>2</sup>, Peng Zhang<sup>3</sup>, Enhong Chen<sup>1,\*</sup>

<sup>1</sup>State Key Laboratory of Cognitive Intelligence, School of Computer Science and Technology,

University of Science and Technology of China

<sup>2</sup>Computer Science, Beijing Institute of Technology

<sup>3</sup>School of Information Science and Technology, University of Science and Technology of China

{liufangyuan,wangxh236,zppeng}@mail.ustc.edu.cn chengguo@emails.bjut.edu.cn

{siruit,cheneh}@ustc.edu.cn <sup>\*</sup>Corresponding authors.

## Ab<sub>s</sub>t<sub>rac</sub>t

Self-disclosures on the Web support diverse psychological analyses, from recognizing expressed experiences to identifying symptomrelevant information. We study a shared, question-conditioned decision interface for this heterogeneous problem in content analysis. We organize PsyJudgeBench, comprising 13 annotated datasets and 82 questions across six psychological areas, while preserving source-specific criteria and masks for unobserved labels. We introduce PsyJev, a System-1 model that directly produces distributions over answers specified in each request. Joint supervised training establishes the shared judgment function, followed by Judgment-Anchored RLCD (JA-RLCD), which adapts logit-space reward optimization to heterogeneous psychological judgments through balanced supervision, selective retention of correct SFT predictions, and question-group reward optimization. The Qwen3.5- based 2B, 4B, and 9B models achieve 64.33, 66.21, and 67.56 average test F1, respectively. The 9B model exceeds the strongest external comparison by 2.54 points. Controlled 2B comparisons examine the training stages and anchored refinement. Further analyses examine question-conditioned judgment, inference cost, and probability quality, complemented by qualitative comparisons across models.

## Ke<sub>y</sub>words

Psychological Analysis, Social Web, System-1 Models, Large Lan guage Models, Psychological Analysis Benchmark

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

Social media, online forums, and Web communities continuously produce spontaneous accounts of personal experiences, emotional dificulties, relationships, and everyday stress. These self-disclosures provide a rich source of Web content for psychological analysis, supporting tasks ranging from recognizing expressed experiences to identifying symptom-relevant information and questionnaire answerability. However, analyzing the same content for diferent purposes requires distinct judgments. A post may mention a symptom without expressing it, or contain symptom-relevant information without providing enough detail to answer a questionnaire item. Such distinctions matter when organizing and analyzing Web content across diferent psychological tasks, where similar textual expressions may need to be interpreted under diferent decision criteria. This motivates a shared framework for addressing a challenge in psychological analysis of Web content: distinguishing questionspecific judgments across heterogeneous datasets drawn from the social Web while preserving their source-specific decision criteria.

Research on web-based psychological analysis has produced specialized datasets and models for diverse targets, from symptom relevance and questionnaire answerability to stressors and interpersonal risk factors [10, 14, 15, 18, 22]. Recent work uses multi-task learning and instruction tuning to bring these tasks into shared frameworks. For instance, MentaLLaMA and the IMHI benchmark unify mental health analysis by generating joint predictions and text explanations [33]. The constituent tasks retain distinct label semantics, answer spaces, and decision criteria. A shared prediction interface therefore needs to preserve these distinctions while supporting categorical evaluation.

Building such a framework poses three main challenges. First, heterogeneous judgment semantics: diferent analytical questions may require distinct criteria and answer sets, even when applied to the same text. This calls for predictions conditioned on the question rather than the text alone. Second, fragmented supervision: existing datasets difer in annotation targets, label schemes, and coverage, complicating their integration into a shared judgment format while retaining source-specific criteria. Third, uneven supervision: diferences in dataset sizes and label frequencies can favor well-represented tasks and frequent classes during joint training. Addressing these challenges calls for a shared question-conditioned judgment framework and a training strategy that accounts for heterogeneous, imbalanced supervision.

To address these challenges, we introduce PsyJudgeBench, a benchmark for question-conditioned psychological analysis. It organizes 13 existing datasets into 82 registered questions across six psychological content areas. Rather than merging heterogeneous tasks into a single label space, PsyJudgeBench represents each judgment through an analytical question, its legal answer set, and source-specific decision criteria retained through documented label mappings. Missing-label masking and coordinated dataset splits support joint training and task-specific evaluation under partial supervision. The resulting benchmark also enables evaluating how models distinguish diferent judgments about the same text.

Building on this task formulation, we introduce PsyJev, a System-1 model that directly predicts distributions over question-defined answers. Given an input text, an analytical question, and candidate options, PsyJev produces a categorical answer distribution through a shared readout, without intermediate response generation or posthoc category extraction [1, 5]. To account for uneven supervision across sources, we further propose Judgment-Anchored RLCD (JA-RLCD), adapting logit-space RLCD [17] to heterogeneous psychological judgments. Its judgment anchor combines source/question weighting and class-balanced supervision with selective retention of correct SFT predictions. A Brier-based question-group reward further refines predictions through logit-space perturbations and pathwise gradients. Across 2B, 4B, and 9B model scales, PsyJev outperforms the corresponding same-backbone answer-SFT baselines.

Controlled 2B comparisons show that the complete JA-RLCD formulation outperforms continued cross-entropy training and anchoronly refinement. Further analyses examine question-conditioned discrimination on co-annotated texts, inference latency, and probability quality.

Our main contributions are as follows:

• A unified psychological judgment benchmark. We construct PsyJudgeBench, organizing 13 heterogeneous datasets into a shared question-defined judgment framework that preserves task-specific criteria and supports unified training and evaluation across questions.

• S<sub>y</sub>stem-1 modeling with judgment-anchored reward refinement. We propose PsyJev for direct probabilistic psychological judgment and JA-RLCD, a joint refinement formulation that adapts logit-space RLCD through a classbalanced judgment anchor, selective SFT-reference retention, and question-group reward optimization under uneven supervision.

• Systematic empirical evaluation. We evaluate PsyJev across model scales and heterogeneous psychological tasks, with comparisons against benchmark-adapted baselines and matched training continuations. Ablations and further analyses assess the training components, questionconditioned discrimination, inference eficiency, and probability quality.

## 2 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

## 2.1 Ps<sub>y</sub>cholo<sub>g</sub>ical Anal<sub>y</sub>sis

Psychological analysis of social Web content spans diverse tasks, including identifying depression and anxiety indications, interpersonal dificulties, self-perceptions, and emotional experiences [6, 9, 10, 15, 25]. Other resources address questionnaire answerability, symptom relevance, stressors, cognitive distortions, wellness dimensions, and crisis-related expressions [2, 7, 8, 14, 18, 22, 26, 27]. These tasks involve related psychological content but difer in their analytical targets and annotation criteria. Existing mod els range from domain-adapted encoders such as MentalBERT to instruction-tuned language models for psychological prediction and explanation [16, 31, 35]. Flexible classification approaches, including GLiClass and GLiNER2, further support label- or schemaconditioned predictions [28, 34]. MentaLLaMA and its IMHI benchmark extend psychological analysis toward multi-task instruction learning over social-media content [33]. Building on this direction, PsyJudgeBench organizes heterogeneous psychological tasks as question-defined judgments, while PsyJev learns a shared directprediction function and refines it through JA-RLCD.

## 2<sub>.</sub>2 S<sub>ys</sub>t<sub>e</sub>m-1 D<sub>ec</sub>i<sub>s</sub>i<sub>o</sub>n M<sub>o</sub>d<sub>e</sub>l<sub>s</sub>

System-1 decision models produce structured predictions without autoregressive answer generation. Jev introduced natural-language interfaces for probabilistic decisions [1], while Laya developed typed prediction with reinforcement learning for calibrated decisions (RLCD) [17]. OpenJev further introduced dynamic-option modeling with candidate-conditioned representations and optionset interaction [21]. Related advances include rationale-conditioned reward learning and direct probabilistic video understanding [5, 36].

![](images/cbbe7bb02601d73f69eae68e31030dc3e43fb3b906bde0f3dd2f6edc60a3d251.jpg)  
Figure 1: PsyJudgeBench coverage. The inner groups organize 13 source datasets into six <sub>p</sub>s<sub>y</sub>cholo<sub>g</sub>ical content areas. Q denotes the number of questions associated with each <sub>source, an</sub>d th<sub>e ou</sub>t<sub>er</sub> b<sub>ars an</sub>d <sub>num</sub>b<sub>ers</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>e source-</sub> <sub>recor</sub>d <sub>coun</sub>t<sub>s.</sub>

Building on this line of work, PsyJev supports heterogeneous psychological judgments through joint supervision on PsyJudgeBench. Our JA-RLCD adapts logit-space RLCD with a class-balanced judgment anchor, selective SFT reference retention, and question-group reward optimization using pathwise gradients.

## 3 PsyJudgeBench

PsyJudgeBench organizes 13 annotated datasets into 53,775 source records and 222,191 observed judgments. Figure 1 summarizes their coverage across six thematic groups: depression/anxiety-related information, self and relationships, stress and distress, cognition, wellness, and crisis experiences. Construction involves a question registry, source-specific annotation conversion, and coordinated data partitions.

Question-defined tasks. PsyJudgeBench represents psychological analysis tasks as question-defined judgments, each specifying a judgment target, a decision criterion, and a set of valid answers. This formulation accommodates diferent analytical requirements while preserving their original annotation criteria. For example, PRIMATE assesses whether a text provides enough information to answer a PHQ-9 item, BDI-Sen evaluates sentence relevance to a BDI-II symptom [14, 22], and MultiWD identifies mentions of specific wellness dimensions. Each judgment is represented as $( x , q , \mathcal { A } _ { q } , y )$ , comprising the source text, question, valid answer set, and observed answer. The resulting registry contains 82 questions, including 79 binary and three multiclass questions.

Annotation harmonization. Following these question definitions, we convert heterogeneous source annotations into observed judgments while retaining their original labeling criteria. Binary and categorical labels are mapped to their corresponding answer sets, while multi-label and category–time annotations are expanded into separate questions. For Thinking Traps, we normalize category names and aggregate repeated completions to recover 15-category label sets. For CRADLE, seven crisis categories combined with ongoing/past status yield 14 questions. For TherapistQA, we retain the dominant distortion category and use only the patient’s question, excluding therapist responses and annotated spans. After conversion, we check for conflicting annotations associated with the same text and judgment target. We remove two TherapistQA records with identical questions but conflicting dominant labels and mask 22 conflicting judgments across 11 same-text/same-target keys. Unafected judgments remain available for supervision, while source identifiers and annotation provenance, including gold/silver distinctions, are preserved.

Partial supervision and coordinated splits. The harmonized records provide supervision only for questions with observed annotations; unannotated questions are excluded rather than treated as nega tive examples. To construct consistent evaluation partitions, we coordinate records sharing identical texts or available source-group identifiers and group identified near-duplicates to reduce overlap across splits. When grouping conflicts with the original partitions, held-out assignments take precedence. The resulting benchmark contains 33,511 training, 5,852 validation, and 14,412 test source records. These counts refer to source records rather than unique texts. Further details on data provenance, partitioning, and duplicate checks are provided in Appendix A.

## 4 PsyJev

PsyJev is a System-1-style model for unified psychological judgment. Given a text, an analytical question, and its legal answers, PsyJev directly produces a categorical distribution through a shared, question-conditioned readout, without autoregressive answer generation. Its training proceeds in two stages: joint supervised finetuning (SFT) learns the shared decision function across heterogeneous tasks, followed by Judgment-Anchored RLCD (JA-RLCD), which combines balanced supervision, selective SFT-reference retention, and question-group reward optimization through logitspace exploration. Figure 2 illustrates the overall framework.

## 4.1 System-1-Style Direct Judgment

PsyJev implements direct psychological judgment through a question conditioned, constrained prediction interface. Rather than generating and parsing free-form responses, it evaluates the admissible answers using their corresponding language-model logits and directly produces a normalized probability distribution.

Question-conditioned representation. Given a text �, an analytical question �, and its legal answer set $\mathcal { A } _ { q } = \{ a _ { 1 } , . . . , a _ { K _ { q } } \}$ , PsyJev constructs a structured input that combines the source text, question, decision criterion, and candidate descriptions. Each candidate is assigned a distinct answer identifier, allowing diferent psychological judgments to be expressed through the same input format. A shared

Qwen3.5 backbone processes the complete judgment request and produces a contextual representation at the answer position.

Constrained answer scoring. Instead of generating an answer sequence, PsyJev directly extracts the vocabulary logits corresponding to the admissible answer identifiers. Let $t _ { k }$ denote the token assigned to candidate $\displaystyle a _ { k } ,$ and let $z _ { \theta } ( t _ { k } \mid x , q , \mathcal { A } _ { q } )$ denote its logit at the answer position. The resulting judgment distribution is

$$
\begin{array} { c } { \displaystyle \ell _ { q k } = z _ { \theta } ( t _ { k } \mid x , q , \mathcal { A } _ { q } ) , } \\ { \displaystyle p _ { \theta } ( k \mid x , q , \mathcal { A } _ { q } ) = \frac { \exp ( \ell _ { q k } ) } { \sum _ { j = 1 } ^ { K _ { q } } \exp ( \ell _ { q j } ) } . } \end{array}\tag{1}
$$

Normalization is restricted to the legal answer set, yielding a categorical distribution whose dimensionality is determined by the current question.

Direct probabilistic judgment. The predicted answer is obtained directly from the normalized answer distribution:

$$
\hat { y } = \arg \operatorname* { m a x } _ { k \in \{ 1 , . . . , K _ { q } \} } \ p _ { \theta } ( k \mid x , q , \mathcal { A } _ { q } ) .\tag{2}
$$

The same scoring procedure supports binary and multiclass psychological judgments without task-specific classification heads or autoregressive answer generation. By retaining the complete distribution over admissible answers, PsyJev also provides the probabilistic decision interface used by subsequent JA-RLCD refinement. Appendix B provides the implementation details.

## 4<sub>.</sub>2 S<sub>upe</sub>r<sub>v</sub>i<sub>se</sub>d Fin<sub>e</sub>-T<sub>u</sub>nin<sub>g</sub>

Joint SFT establishes the shared judgment function by learning from observed annotations across source datasets. Let $\mathcal { D } _ { s }$ denote the records from source �, and let $O ( x )$ denote the questions annotated for record �. We minimize categorical cross-entropy:

$$
\mathcal { L } _ { \mathrm { S F T } } = - \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \mathbb { E } _ { x \sim \mathcal { D } _ { s } } \left[ \frac { 1 } { \vert O ( x ) \vert } \sum _ { q \in O ( x ) } \log \hat { p } _ { \theta } ( y _ { q } \vert x , q , \mathcal { A } _ { q } ) \right] .\tag{3}
$$

The outer average assigns equal weight to each source, while the inner average prevents records with more annotated questions from dominating the objective. Missing question labels contribute no loss. The resulting checkpoint $\theta _ { \mathrm { S F T } }$ initializes the subsequent JA-RLCD refinement.

## 4.3 Judgment-Anchored RLCD

RLCD explores probabilistic decisions through logit-space perturbations and reward evaluation [17]. Building on this formulation, JA-RLCD addresses uneven source, question, and answer frequencies in psychological judgments through a joint refinement objective: a judgment anchor combines class-balanced supervision and selective retention of correct SFT predictions, while a Brierbased reward emphasizes higher-error question groups within each microbatch. Starting from the SFT checkpoint, we treat the legalanswer distribution as a judgment policy and refine it through paired Gaussian perturbations and diferentiable reward optimization. For each source-specific microbatch of observed judgments, reward gradients are computed through pathwise diferentiation and combined with the anchor gradients before accumulation and AdamW updates. Algorithm 1 summarizes the procedure.

![](images/86f432cf1b0b39038b436d72088ea4c2a8448e4e06e6d132b2c91f407389e523.jpg)  
Figure 2: PsyJev overview. (a) System-1-style direct judgment through question-conditioned input serialization and constrained legal-answer scoring. (b) Two-stage training with joint SFT followed by JA-RLCD, combining judgment anchoring with logit s<sub>p</sub>ace ex<sub>p</sub>loration and <sub>q</sub>uestion-<sub>g</sub>rou<sub>p</sub> reward o<sub>p</sub>timization.

Judgment anchor. For judgment $i \in { \mathcal { B } } ,$ training-count weight � balances source/question observation frequency, and cost $c _ { i } = c _ { q _ { i } , y _ { i } }$ balances answer classes (Appendix B). Let $\mathbf { \nabla } \mathcal { P } i$ and $\mathbf { \nabla } \mathcal { P } 0 , i$ denote the trainable and frozen distributions over $\mathcal { A } _ { q _ { i } }$ . The anchor is:

$$
\mathcal { L } _ { \mathrm { A } } = \sum _ { i \in \mathcal { B } } a _ { i } c _ { i } [ - \log p _ { i } ( y _ { i } ) ] + \lambda \sum _ { i \in \mathcal { B } } a _ { i } m _ { i } \mathrm { K L } ( p _ { 0 , i } \| p _ { i } ) ,\tag{4}
$$

where $m _ { i } = \mathbb { I } [ \arg \operatorname* { m a x } _ { k } { \ p _ { 0 , i , k } = y _ { i } } ]$ . The indicator I[·] equals 1 when its condition holds and 0 otherwise, so reference retention applies only to judgments the frozen model answers correctly. The label term anchors reward-driven changes to observed answers, while the reference term encourages retention of the SFT distribution where its prediction agrees with the annotation.

Brier-based question-group reward. Let $\mathcal { G } = \{ q _ { i } : i \in \mathcal { B } \}$ be the observed questions in a microbatch, and let $\mathcal { P } = ( p _ { i } ) _ { i \in \mathcal { B } }$ collect its legal-answer distributions. We define each question’s reward as the negative class-weighted Brier error [13], then aggregate the question rewards with a mean and an entropic soft minimum:

$$
\begin{array} { r l } & { { \cal R } _ { q } ( \mathcal { P } ) = - \frac { \displaystyle \sum _ { i \in \mathcal { B } : q _ { i } = q } c _ { i } \frac { 1 } { 2 } \| \boldsymbol { \mathcal { P } } _ { i } - \boldsymbol { e } _ { y _ { i } } \| _ { 2 } ^ { 2 } } { \displaystyle \sum _ { i \in \mathcal { B } : q _ { i } = q } c _ { i } } , } \\ & { \displaystyle \mathcal { R } _ { \boldsymbol { \tau } } ( \mathcal { P } ) = \frac { 1 } { 2 } \overline { { \boldsymbol { R } } } - \frac { 1 } { 2 \tau } \log \left( \frac { 1 } { | \boldsymbol { \mathcal { G } } | } \sum _ { q \in \mathcal { G } } e ^ { - \tau R _ { q } ( \mathcal { P } ) } \right) . } \end{array}\tag{5}
$$

Here � is the one-hot answer and $e _ { y _ { i } }$ $\begin{array} { r } { \overline { { R } } \ = \ | \mathcal { G } | ^ { - 1 } \sum _ { q \in \mathcal { G } } R _ { q } ( \mathcal { P } ) } \end{array}$ Higher reward indicates lower prediction error. With $\tau = 4$ , the soft-minimum term places more weight on questions with lower reward within the current microbatch. This aggregate is the negative of the corresponding question-group risk.

Logit-space exploration. We draw $M = 2$ independent Gaussian directions $\epsilon _ { m , i } \sim { \cal N } ( 0 , I _ { K _ { i } } )$ per judgment, where $K _ { i } = | \mathcal { A } _ { q _ { i } } |$ . The projection $P _ { i } = I - 1 1 ^ { \top } / K _ { i }$ removes the softmax-invariant common shift; $\textbf { 1 } \in \mathbb { R } ^ { K _ { i } }$ denotes the all-ones vector. With $\begin{array} { r } { A _ { \mathcal { B } } = \sum _ { i \in \mathcal { B } } a _ { i } } \end{array}$ paired perturbations give the sampled, Gaussian-smoothed reward:

$$
\begin{array} { r l } & { z _ { m , i } ^ { \pm } = \ell _ { \theta , i } \pm \sigma P _ { i } \epsilon _ { m , i } , } \\ & { \displaystyle p _ { m , i } ^ { v } = \mathrm { s o f t m a x } ( z _ { m , i } ^ { v } ) , \qquad v \in \{ + , - \} , } \\ & { \displaystyle \widehat { \mathcal { I } } _ { \mathrm { R } } = \frac { A _ { \mathcal { B } } } { 2 M } \displaystyle \sum _ { m = 1 } ^ { M } \displaystyle \sum _ { v \in \{ + , - \} } \mathcal { R } _ { \tau } ( \mathcal { P } _ { m } ^ { v } ) . } \end{array}\tag{6}
$$

Here $\mathcal { P } _ { m } ^ { v } = ( \boldsymbol { p } _ { m , i } ^ { v } ) _ { i \in \mathcal { B } }$ , and softmax is over each judgment’s legal answers. All four reward evaluations recompute Eq. 5 on the same microbatch from the same model logits. The individual anchor weights �<sub>�</sub> balance judgments; their sum �<sub>B</sub> scales the entire reward estimate rather than reweighting its question groups.

Pathwise reward gradient. Since the question-group reward is diferentiable with respect to the answer distributions, we directly propagate its gradients through the perturbed logits and legalanswer softmax. With Gaussian noise sampled independently of �, this yields the pathwise reward gradient $u = \nabla _ { \theta } \widehat { \mathcal { F } } _ { \mathrm { R } }$ , allowing the sampled rewards to guide refinement without relying on a score-function estimator. We further examine the choice of gradient estimator in Section 5.3.2.

Algorithm 1 Judgment-Anchored RLCD (JA-RLCD)   
Require: Pretrained LLM $\theta _ { \mathrm { p r e } } ;$ annotated records D   
Ensure: Trained direct-judgment model �   
1: $\theta _ { \mathrm { S F T } }  \mathrm { S F T } ( \theta _ { \mathrm { p r e } } , \mathcal { D } )$ using Eq. 3   
2: Initialize $\theta \gets \theta _ { \mathrm { S F T } } ;$ ; freeze reference $\theta _ { 0 }  \theta _ { \mathrm { S F T } }$   
3: Compute class costs �<sub>�</sub> and source/question training counts   
4: for each record batch $C \subset \mathcal { D }$ d<sub>o</sub>   
5: Set � using batch size $B = | C |$ ⊲ Appendix B   
6: $d _ { \mathrm { a c c } } \gets 0$   
7: for each source-specific pack B of at most 16 judgments do   
8: $\ell \gets \mathrm { R e a d o u t } ( \mathrm { L L M } _ { \theta } , \mathcal { B } )$ ⊲ Eq. 1   
9: �<sub>0</sub> ← ReadoutProb $( \operatorname { L L M } _ { \theta _ { 0 } } , { \mathcal { B } } )$ ⊲ Frozen   
10: Set $m _ { i } \gets \mathbb { I } [$ [arg max� $\displaystyle p _ { 0 , i , k } = y _ { i } ]$ for $i \in \mathcal { B }$   
11: Form balanced CE and gated KL anchor $\mathcal { L } _ { \mathrm { A } }$ ⊲ Eq. 4   
12: Set $\begin{array} { r } { A _ { \mathcal { B } } \gets \sum _ { i \in \mathcal { B } } a _ { i } } \end{array}$ and $\mathcal { G }  \{ \boldsymbol { q } _ { i } : i \in \mathcal { B } \}$   
13:   
14: Compute legal-answer probabilities $\mathcal { P } _ { 1 } ^ { \pm } , \mathcal { P } _ { 2 } ^ { \pm }$ ⊲ Eq. 6   
15: Evaluate $\mathcal { R } _ { \tau } ( \mathcal { P } _ { m } ^ { v } )$ on B for all four draws   
16: $\begin{array} { r } { \widehat { \mathcal { I } } _ { \mathrm { R } } \gets \frac { A _ { \mathcal { B } } } { 4 } \sum _ { m = 1 } ^ { 2 } \sum _ { v \in \{ + , - \} } \mathcal { R } _ { \tau } ( \mathcal { P } _ { m } ^ { v } ) } \end{array}$   
17: $g \gets \nabla _ { \theta } \mathcal { L } _ { \mathrm { A } } ; u \gets \nabla _ { \theta } \widehat { \mathcal { F } } _ { \mathrm { R } }$ ⊲ Pathwise   
18: $d _ { \mathcal { B } } \gets g - u$ ⊲ Anchored refinement; Eq. 7   
19: $d _ { \mathrm { a c c } }  d _ { \mathrm { a c c } } + d _ { \mathcal { B } }$   
20: <sub>en</sub>d f<sub>or</sub>   
21: � ← AdamW(�, ClipNorm $\left( d _ { \mathrm { a c c } } , 1 \right) )$   
22: en<sup>d f</sup>or   
23: return <sup>�</sup> ⊲ Use direct readout for inference

Anchored policy refinement. We maximize reward while maintaining the judgment anchor. Writing $g = \nabla _ { \theta } { \mathcal { L } } _ { \mathrm { A } }$ , the anchored objective and its descent direction are:

$$
\begin{array} { r } { \widehat { \mathcal { T } } _ { \mathrm { A P } } = \widehat { \mathcal { T } } _ { \mathrm { R } } - \mathcal { L } _ { \mathrm { A } } , } \\ { d = - \nabla _ { \theta } \widehat { \mathcal { T } } _ { \mathrm { A P } } = g - u . } \end{array}\tag{7}
$$

We accumulate � across microbatches, clip the accumulated direction to a global norm of 1, and apply AdamW, keeping the SFT reference frozen. Equivalently, the anchored objective minimizes a supervised loss combining the judgment anchor with Gaussiansmoothed, question-group prediction risk.

## 5 Ex<sub>p</sub>eriments

## 5.1 Ex<sub>p</sub>erimental Setu<sub>p</sub>

5.1.1 Benchmark and Metrics. We evaluate PsyJev on PsyJudgeBench using the coordinated splits described in Section 3. We report positive-class F1 for binary questions and macro-F1 for multiclass questions, averaging across observed questions within each dataset. The overall score equally weights the 13 datasets. We assess probability quality using normalized negative log-likelihood (NLL), Brier score, and expected calibration error (ECE) with 15 equal-width bins. NLL is normalized by log $| \mathcal { R } _ { q } |$ to account for varying answerset sizes. Eficiency is measured by model computation time per judgment.

5.1.2 Baselines. We compare with 18 benchmark-adapted baselines and the zero-shot Jev API. The adapted baselines fall into three groups. Compact classification and decision models include MentalBERT, MentalRoBERTa, ModernBERT-large, GLiClass, GLiNER2, and Laya-multilingual [16, 17, 28, 30, 34]. Laya uses its released architecture and checkpoint; its reported result follows benchmark SFT and continuation with its native RLCD objective. The 2–4B language models are FLAN-T5-XL, Qwen3.5-2B/4B, Llama-3.2-3B, Phi-4-mini, and Gemma-3-4B [3, 12, 19, 20, 24]. The 7–9B models include MentaLLaMA-7B, Qwen2.5-7B, Llama-3.1-8B, Qwen3-8B, Gemma-2-9B, and Qwen3.5-9B [4, 11, 23, 24, 32, 33]. All 18 models are adapted on the benchmark training partition. Classification and decision models predict the observed task labels directly. Language models learn to produce the corresponding answers and are evaluated by scoring the allowed answer sequences. We additionally evaluate Typesafe’s Jev (jev-1.14.0) through its oficial API without benchmark fine-tuning or demonstrations [1]. Appendix B.3 specifies both adaptation and API protocols.

5.1.3 Implementation Details. We initialize PsyJev-2B/4B/9B from the corresponding Qwen3.5 text backbones and train each model with joint SFT followed by JA-RLCD, using one epoch per stage. Source texts are truncated to 512 tokens, with batches of up to 64 texts. The 2B model updates all parameters at a learning rate of $2 \times 1 0 ^ { - 6 }$ in both stages. The 4B/9B models use LoRA (rank 16, scaling 32, zero dropout), with learning rates o $2 \times 1 0 ^ { - 4 }$ for SFT and $2 \times { 1 0 } ^ { - 6 }$ for JA-RLCD. All models use AdamW with weight decay 0.01, gradient clipping at 1.0, and 3% warmup followed by cosine decay. Backbone computation uses BF16, and the final checkpoint of each stage is retained. We measure model computation time using a single H100 80GB GPU, one text per request, and a fixed 13-text training probe shared by all timed models; classification metrics use the complete test partition. Further details appear in Appendices B and B.4.

## 5<sub>.</sub>2 P<sub>er</sub>f<sub>ormance</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub>

Table 1 shows that PsyJev consistently achieves the highest average F1 within each model-size group, reaching 64.33, 66.21, and 67.56 at 2B, 4B, and 9B, respectively. Compared with the corresponding Qwen3.5 answer models, these results represent gains of 4.01, 1.71, and 3.88 points. Notably, the 4B and 9B variants also outperform the strongest external baseline, despite the smaller size of the former. Across datasets, PsyJev shows notable gains on tasks involving cognitive distortions and wellness-related judgments, although gains vary across question criteria and are not uniform across all datasets. The general-purpose Jev API achieves 52.91 average F1 in zero-shot evaluation, while the benchmark-adapted PsyJev variants perform substantially better, providing a zero-shot reference alongside the benchmark-adapted comparisons. Overall, these results demonstrate the efectiveness of PsyJev across heterogeneous psychological judgment tasks and model scales.

## 5.3 Ablation Stud<sub>y</sub>

Table 2 uses the 2B model to examine training stages, JA-RLCD components, and matched-budget prediction formats. Direct-scoring continuations share the same SFT initialization and schedule; the answer-sequence control uses its corresponding SFT model with the same update budget.

Table 1: Performance comparison on PsyJudgeBench (F1, %). Avg. denotes the unweighted mean across 13 datasets. Bold indicates the best result within each benchmark-adapted model group. Jev is evaluated zero-shot. Latency measures model computation (ms/judgment) on the fixed H100 CUDA Graph probe. Abbreviations: DepEmo (DepressionEmo), BDISen (BDI-Sen), LonXpl (LonXplain), ThinkTrap (Thinking Traps), and TherapQA (TherapistQA).
<table><tr><td></td><td colspan="4">Depression / Anxiety</td><td colspan="3">Self / Relationships</td><td colspan="2">Stress / Distress</td><td colspan="2">Cognition</td><td>Wellness</td><td>Crisis</td><td>Avg.</td><td>Inference</td></tr><tr><td>Model</td><td>DepEmo</td><td>ANGST</td><td>PRIMATE</td><td>BDISen</td><td>IRF</td><td></td><td>LoST LonXpl</td><td>SAD</td><td>CAMS</td><td>ThinkTrap</td><td>TherapQA</td><td>MultiWD</td><td>CRADLE</td><td>Avg.</td><td>Latency (ms)</td></tr><tr><td colspan="10">Classification and Decision Models</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MentalBERT</td><td>52.09</td><td>43.38</td><td>27.03</td><td>8.61</td><td>65.00</td><td>17.10</td><td>77.43</td><td>66.60</td><td>38.46</td><td>0.00</td><td>7.82</td><td>37.88</td><td>0.00</td><td>33.95</td><td>0.46</td></tr><tr><td>MentalRoBERTa</td><td>60.74</td><td>58.21</td><td>33.11</td><td>5.13</td><td>75.45</td><td>31.55</td><td>80.66</td><td>67.87</td><td>39.06</td><td>0.00</td><td>10.51</td><td>38.69</td><td>0.00</td><td>38.54</td><td>0.46</td></tr><tr><td>ModernBERT-large</td><td>68.95</td><td>64.22</td><td>48.06</td><td>42.90</td><td>82.65</td><td>67.12</td><td>83.17</td><td>71.31</td><td>46.76</td><td>15.47</td><td>19.25</td><td>53.49</td><td>20.08</td><td>52.57</td><td>1.16</td></tr><tr><td>GLiClass</td><td>75.78</td><td>69.70</td><td>56.97</td><td>57.36</td><td>83.41</td><td>68.01</td><td>84.77</td><td>73.47</td><td>49.66</td><td>12.95</td><td>19.62</td><td>62.99</td><td>38.88</td><td>57.97</td><td>1.90</td></tr><tr><td>GLiNER2</td><td>75.90</td><td>72.24</td><td>68.72</td><td>46.71</td><td>84.44</td><td>70.56</td><td>85.92</td><td>73.82</td><td>49.56</td><td>16.60</td><td>19.51</td><td>65.24</td><td>37.88</td><td>59.01</td><td>3.16</td></tr><tr><td>Laya</td><td>68.61</td><td>71.47</td><td>50.46</td><td>35.30</td><td>77.19</td><td>58.10</td><td>79.34</td><td>70.85</td><td>39.59</td><td>0.00</td><td>9.42</td><td>50.90</td><td>2.39</td><td>47.20</td><td>1.31</td></tr><tr><td colspan="10">2-3B Models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FLAN-T5-XL</td><td>71.30</td><td>72.98</td><td>57.52</td><td>41.21</td><td>63.30</td><td>66.22</td><td>84.56</td><td>74.18</td><td>42.27</td><td>9.37</td><td>8.81</td><td>56.40</td><td>39.29</td><td>52.88</td><td>10.33</td></tr><tr><td>Qwen3.5-2B</td><td>75.74</td><td>70.05</td><td>72.83</td><td>50.77</td><td>84.82</td><td>69.91</td><td>85.48</td><td>73.56</td><td>50.94</td><td>23.86</td><td>19.45</td><td>63.64</td><td>43.09</td><td>60.32</td><td>8.44</td></tr><tr><td>Llama-3.2-3B</td><td>77.57</td><td>68.88</td><td>65.94</td><td>52.18</td><td>84.25</td><td>69.49</td><td>86.10</td><td>73.55</td><td>50.03</td><td>22.64</td><td>19.59</td><td>65.34</td><td>48.57</td><td>60.32</td><td>18.82</td></tr><tr><td>PsyJev-2B</td><td>78.25</td><td>70.36</td><td>69.26</td><td>55.18</td><td>83.97</td><td>70.38</td><td>85.31</td><td>74.72</td><td>51.47</td><td>48.58</td><td>25.38</td><td>68.34</td><td>55.10</td><td>64.33</td><td>4.38</td></tr><tr><td colspan="10">4B Models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Phi-4-mini</td><td>74.22</td><td>67.77</td><td>63.41</td><td>52.67</td><td>83.77</td><td>66.36</td><td>85.61</td><td>75.01</td><td>50.03</td><td>14.24</td><td>15.62</td><td>62.48</td><td>43.55</td><td>58.06</td><td>18.39</td></tr><tr><td>Gemma-3-4B-IT</td><td>76.85</td><td>69.14</td><td>70.70</td><td>54.50</td><td>84.55</td><td>69.98</td><td>86.26</td><td>74.84</td><td>50.85</td><td>22.98</td><td>16.35</td><td>66.26</td><td>46.62</td><td>60.76</td><td>23.82</td></tr><tr><td>Qwen3.5-4B PsyJev-4B</td><td>77.27</td><td>67.79</td><td>72.50</td><td>60.74</td><td>85.44</td><td>68.92</td><td>86.98</td><td>75.80</td><td>53.36</td><td>33.14</td><td>27.93</td><td>67.22</td><td>61.40</td><td>64.50</td><td>25.25</td></tr><tr><td></td><td>78.63</td><td>69.67</td><td>71.76</td><td>56.23</td><td>85.54</td><td>71.37</td><td>86.59</td><td>76.10</td><td>53.88</td><td>50.17</td><td>29.80</td><td>69.54</td><td>61.43</td><td>66.21</td><td>12.01</td></tr><tr><td colspan="10">7–9B Models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-7B</td><td>75.77</td><td>67.10</td><td>69.76</td><td>58.17</td><td>85.25</td><td>70.77</td><td>86.57</td><td>75.71</td><td>50.90</td><td>33.41</td><td>27.37</td><td>65.58</td><td>50.06</td><td>62.80</td><td>31.52</td></tr><tr><td>Llama-3.1-8B</td><td>76.72</td><td>68.31</td><td>70.87</td><td>59.21</td><td>85.52</td><td>70.64</td><td>87.02</td><td>77.01</td><td>49.77</td><td>30.37</td><td>28.10</td><td>65.39</td><td>58.67</td><td>63.66</td><td>35.18</td></tr><tr><td>Qwen3-8B</td><td>75.64</td><td>68.84</td><td>68.45</td><td>55.92</td><td>84.38</td><td>70.98</td><td>86.92</td><td>74.84</td><td>54.54</td><td>27.77</td><td>26.23</td><td>66.16</td><td>47.51</td><td>62.17</td><td>35.94</td></tr><tr><td>Gemma-2-9B-IT</td><td>77.34</td><td>69.10</td><td>71.34</td><td>59.75</td><td>85.40</td><td>70.43</td><td>86.67</td><td>76.45</td><td>52.16</td><td>39.83</td><td>23.14</td><td>66.49</td><td>67.17</td><td>65.02</td><td>45.82</td></tr><tr><td>Qwen3.5-9B</td><td>75.19</td><td>69.81</td><td>68.95</td><td>54.50</td><td>85.49</td><td>71.24</td><td>86.29</td><td>77.27</td><td>51.88</td><td>37.21 2.95</td><td>24.37</td><td>64.66</td><td>60.97</td><td>63.68</td><td>36.27</td></tr><tr><td>MentaLLaMA-7B</td><td>75.27</td><td>69.88</td><td>63.02</td><td>54.84</td><td>84.44</td><td>70.28</td><td>85.89</td><td>75.83</td><td>48.93</td><td></td><td>17.45</td><td>63.94</td><td>46.43</td><td>58.40</td><td>34.12</td></tr><tr><td>PsyJev-9B</td><td>78.03</td><td>69.05</td><td>71.75</td><td>63.87</td><td>85.01</td><td>71.52</td><td>86.41</td><td>77.81</td><td>54.52</td><td>54.37</td><td>33.22</td><td>69.37</td><td>63.36</td><td>67.56</td><td>17.12</td></tr><tr><td colspan="10">Closed API Model (Zero-Shot)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Jev (API)</td><td>66.91</td><td>60.19</td><td>50.97</td><td>48.42</td><td>62.25</td><td>46.23</td><td>81.84</td><td>70.48</td><td>38.26</td><td>35.04</td><td>21.62</td><td>57.57</td><td>48.07</td><td>52.91</td><td></td></tr></table>

5.3.1 Training Stages. JA-RLCD achieves 64.33 average F1, improving upon SFT by 3.65 points. It also outperforms continued cross-entropy (CE) training and adapted RLCD by 2.61 and 5.46 points, respectively. Continued CE controls for additional training, while adapted RLCD combines CE supervision with the original RLCD’s noisy-logit score-function objective. These comparisons show that additional training alone does not explain the improvement and demonstrate the efectiveness of the overall JA-RLCD formulation. Since adapted RLCD difers in both its reward formulation and gradient estimator, the comparison does not isolate the contribution of either component.

5.3.2 Component Ablations. Class-balanced CE alone achieves 63.29 F1, compared with 63.70 for anchor-only training, which also retains selective reference KL. Full JA-RLCD reaches 64.33, improving over balanced CE by 1.04 points and anchor-only training by 0.63 points.

Replacing the question-group/tail reward with a per-judgment Brier reward gives 63.36 F1 while retaining the anchor, perturba tions, and pathwise estimator. The 0.97-point advantage supports the combined group/tail reward formulation. Removing class costs reduces F1 to 62.15; removing reference KL or replacing pathwise diferentiation with a paired score-function estimator gives 64.12 and 63.91, respectively. Appendix B.3 details these controls.

Table 2: 2B ablations and matched-budget controls. ΔF1 is relative to full JA-RLCD, computed before rounding.
<table><tr><td>Training configuration</td><td>Avg. F1 ↑</td><td>∆F1</td></tr><tr><td>(a) Training pipeline</td><td></td><td></td></tr><tr><td>SFT only</td><td>60.68</td><td>-3.65</td></tr><tr><td>SFT → continued CE</td><td>61.72</td><td>-2.61</td></tr><tr><td>SFT → adapted RLCD</td><td>58.87</td><td>-5.46</td></tr><tr><td>SFT → JA-RLCD (full)</td><td>64.33</td><td>+0.00</td></tr><tr><td>(b) Refinement and matched controls</td><td></td><td></td></tr><tr><td>Anchor only (w/o reward)</td><td>63.70</td><td>-0.63</td></tr><tr><td>w/o class costs</td><td>62.15</td><td>-2.18</td></tr><tr><td>w/o reference KL</td><td>64.12</td><td>-0.21</td></tr><tr><td>w/ score-function estimator</td><td>63.91</td><td>-0.42</td></tr><tr><td>Balanced CE only (direct)</td><td>63.29</td><td>-1.04</td></tr><tr><td>Per-judgment Brier reward</td><td>63.36</td><td>-0.97</td></tr><tr><td>Balanced CE (answer sequence)</td><td>63.55</td><td>-0.78</td></tr></table>

5.3.3 Performance Across Questions. We further examine whether JA-RLCD improves questions that are dificult for the SFT model. Using only SFT validation performance, we define the subset as the 16 lowest-F1 questions among 62 questions with suficient validation annotations. As shown in Table 3, JA-RLCD improves mean test F1 by 19.45 points over SFT on these dificult questions, compared with 6.23 points across all 82 questions. The gains remain positive relative to continued CE and anchor-only training, reaching 13.38 and

Table 3: Question-averaged test-F1 gains of JA-RLCD across <sub>a</sub>ll 82 <sub>ques</sub>ti<sub>ons</sub> <sub>an</sub>d 16 <sub>va</sub>lid<sub>a</sub>ti<sub>on-se</sub>l<sub>ec</sub>t<sub>e</sub>d difi<sub>cu</sub>lt <sub>ques</sub>ti<sub>ons.</sub>
<table><tr><td>JA-RLCD vs.</td><td>All Q ∆F1</td><td>Difficult Q ∆F1</td><td>Improved Q (%)</td></tr><tr><td>SFT</td><td>+6.23</td><td>+19.45</td><td>57.32</td></tr><tr><td>Continued CE</td><td>+4.36</td><td>+13.38</td><td>56.10</td></tr><tr><td>Anchor only</td><td>+1.38</td><td>+3.02</td><td>58.54</td></tr></table>

Table 4: Question-conditioned judgment on opposite-label <sub>q</sub>uestion <sub>p</sub>airs. Both correct and same-answer errors are re-<sub>por</sub>t<sub>e</sub>d <sub>as</sub> <sub>source-</sub>b<sub>a</sub>l<sub>ance</sub>d <sub>percen</sub>t<sub>ages.</sub>
<table><tr><td>Model</td><td>Both correct ↑</td><td>Same-answer errors ↓</td></tr><tr><td>GLiClass</td><td>55.84</td><td>41.88</td></tr><tr><td>Gemma-2-9B</td><td>67.85</td><td>29.84</td></tr><tr><td>Qwen3.5-4B answer</td><td>66.24</td><td>31.95</td></tr><tr><td>PsyJev-4B</td><td>70.06</td><td>28.09</td></tr></table>

3.02 points on the dificult subset, respectively. JA-RLCD improves test F1 on 57.32% of all questions relative to SFT. These findings show larger average gains on the validation-defined dificult subset, while improvements are not uniform across individual questions. Unlike the dataset-balanced benchmark average, these analyses weight questions equally.

## 5<sub>.</sub>4 An<sub>a</sub>l<sub>ys</sub>i<sub>s</sub>

Beyond overall performance, we analyze PsyJev’s ability to dis tinguish question-specific judgments, its inference eficiency, the reliability of its predicted probabilities, and its transfer to an unseen dataset, complemented by qualitative examples.

5.4.1 Question-ConditionedJudgment. We examine whether Psy-Jev can distinguish diferent psychological judgments about the same text. Specifically, we evaluate co-annotated binary question pairs with opposite ground-truth labels, where successful prediction requires both questions to be answered correctly. We compare PsyJev-4B with GLiClass, Gemma-2-9B, and the corresponding Qwen3.5-4B answer model using identical texts, questions, and semantic answer sets.

As shown in Table 4, PsyJev correctly answers both questions in 70.06% of the 35,992 pairs from eight sources, outperforming the same-size answer model (66.24%), Gemma-2-9B (67.85%), and GLiClass (55.84%). Compared with the answer model, same-answer errors decrease from 31.95% to 28.09%, with a paired text-group 95% confidence interval of 1.73–5.96 percentage points for the improvement in paired correctness. These results indicate stronger question-conditioned discrimination on this matched-pair subset, where the same text requires diferent answers under diferent criteria. Additional outcome breakdowns are provided in Appendix C.3.

5.4.2 Inference Eficiency. Figure 3 compares model computation time and full-test F1 for three trained 4B configurations. Full-answer and first-token scoring use the answer-SFT checkpoint, while Psy Jev uses its final JA-RLCD checkpoint. On the fixed 13-text training probe with CUDA Graphs on an H100 80GB, PsyJev requires 12.01 ms per judgment, a 2.10× speedup over full-answer scoring (25.25 ms). Its full-test F1 is 2.38 points higher than the first-token configuration at comparable model computation time. Appendix B.4 specifies the shared timing protocol.

![](images/73c5341e89ed4044ca28914b7bcd9a4427b7f9f0a4541b079ae76ab77a3e260a.jpg)  
Fi<sub>g</sub>ure 3: F1–latenc<sub>y</sub> com<sub>p</sub>arison of three 4B confi<sub>g</sub>urations. PsyJev gains 2.38 F1 points over answer-SFT first-token scorin<sub>g</sub> at com<sub>p</sub>arable GPU latenc<sub>y,</sub> while runnin<sub>g</sub> 2.10× faster th<sub>an</sub> f<sub>u</sub>ll<sub>-answer</sub> <sub>scor</sub>i<sub>ng</sub> <sub>un</sub>d<sub>er</sub> th<sub>e</sub> <sub>s</sub>h<sub>are</sub>d ti<sub>m</sub>i<sub>ng</sub> <sub>pro</sub>t<sub>oco</sub>l<sub>.</sub>

Table 5: Classification <sub>p</sub>erformance and <sub>p</sub>robabilit<sub>y</sub> <sub>q</sub>ualit<sub>y</sub> on PsyJudgeBench. F1 and ECE are percentages. NLL is nor-<sub>ma</sub>li<sub>ze</sub>d b<sub>y</sub> th<sub>e</sub> l<sub>og</sub> <sub>answer-se</sub>t <sub>s</sub>i<sub>ze.</sub> B<sub>o</sub>ld i<sub>n</sub>di<sub>ca</sub>t<sub>es</sub> th<sub>e</sub> b<sub>es</sub>t <sub>resu</sub>lt <sub>among</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub>
<table><tr><td>Model</td><td>F1↑</td><td>NLL↓</td><td>Brier ↓</td><td>ECE↓</td></tr><tr><td>MentalBERT</td><td>33.95</td><td>0.659</td><td>0.3619</td><td>5.82</td></tr><tr><td>MentalRoBERTa</td><td>38.54</td><td>0.620</td><td>0.3405</td><td>4.75</td></tr><tr><td>ModernBERT-large</td><td>52.57</td><td>0.580</td><td>0.3089</td><td>5.75</td></tr><tr><td>GLiNER2</td><td>59.01</td><td>0.544</td><td>0.2869</td><td>4.95</td></tr><tr><td>FLAN-T5-XL</td><td>52.88</td><td>0.588</td><td>0.3220</td><td>6.41</td></tr><tr><td>PsyJev-4B</td><td>66.21</td><td>0.536</td><td>0.2777</td><td>4.58</td></tr></table>

![](images/25f565ccbdeb134a05371fca0ea2b224d4fc7312901c45ede0d2922654a2c3be.jpg)  
Fi<sub>gure</sub> 4<sub>:</sub> S<sub>ource-mean</sub> ECE <sub>accumu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>over</sub> th<sub>e</sub> f<sub>u</sub>ll 0<sub>–</sub>100% <sub>con</sub>fid<sub>ence</sub> <sub>range.</sub> C<sub>urves</sub> <sub>sum</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> f<sub>rom</sub> <sub>a</sub>ll 15 bi<sub>ns;</sub> their 100% end<sub>p</sub>oints re<sub>p</sub>roduce Table 5. Lower end<sub>p</sub>oints i<sub>n</sub>di<sub>ca</sub>t<sub>e</sub> b<sub>e</sub>tt<sub>er overa</sub>ll <sub>ca</sub>lib<sub>ra</sub>ti<sub>on.</sub>

5.4.3 Prediction Quality. Beyond classification accuracy, we assess the quality of predicted answer distributions using Brier score, normalized NLL, and ECE. Table 5 compares PsyJev-4B with representative encoder-based, description-driven, and answer-generation models on the same test judgments, using source-balanced probability metrics (Appendix C.1). PsyJev-4B achieves the highest F1 (66.21) and the lowest Brier score (0.2777) and normalized NLL (0.536) among the models displayed in Table 5, outperforming the strongest reference, GLiNER2, on all three metrics. It also obtains the lowest ECE among these models (4.58%) under the same sourcebalanced evaluation. Figure 4 shows how source-mean calibration error accumulates across the full confidence range.

Table 6: Illustrative judgment corrections by PsyJev-4B after JA-RLCD. × indicates disagreement with source annotations. Dis<sub>p</sub>la<sub>y</sub>ed texts are excer<sub>p</sub>ts of the ori<sub>g</sub>inal test in<sub>p</sub>uts.
<table><tr><td>Gold</td><td>GLiClass</td><td>Gemma-2-9B</td><td>Qwen3.5-4B Answer SFT</td><td>PsyJev-4B SFT</td><td>PsyJev-4B JA-RLCD</td></tr><tr><td colspan="6">A. Co-occurring emotions</td></tr><tr><td colspan="6">&quot;Very lonely in high school, then very lonely in the military. Eventually I was able to have a group of friends for a while and it filled the void, but I ditched them, I get lonely now and again&quot;</td></tr><tr><td>Emptiness, Loneliness</td><td>Loneliness ×</td><td></td><td>Emptiness, Loneliness Emptiness, Loneliness, Sadness × Emptiness, Loneliness, Sadness ×</td><td></td><td>Emptiness, Loneliness</td></tr><tr><td colspan="6">Interpretation. PsyJev recovers Emptiness and Loneliness without the additional Sadness label predicted by the answer model and SFT.</td></tr><tr><td colspan="6">B. Competing distress cues &quot;i have a part-time job and struggle to even get through 4 hour shifts. so i don&#x27;t make enough money to do anything but attend work basically and smoke some cigarettes. [...</td></tr><tr><td colspan="6">i&#x27;m also struggling with a severely broken heart from an on-and-off 10 year thing&quot;</td></tr><tr><td></td><td></td><td>Relationship ×</td><td>Relationship ×</td><td>Relationship × Jobs and careers</td></tr><tr><td colspan="6">Jobs and careers Relationship ×</td></tr><tr><td colspan="6">Interpretation. Limited work capacity and insufficient income are consistent with the annotated Jobs and careers category, despite the competing relationship cue.</td></tr><tr><td colspan="6">C. Related cognitive patterns</td></tr><tr><td colspan="6">&quot;My ss is 17. I&#x27;ve been in his life since age 6. He has always lied and been spiteful. [..] The lying has never stopped. He just got better at it. He learned how to lie with sincerity.&quot; Overgeneralization</td></tr><tr><td colspan="6">OvergeneralizationNo Distortion Overgeneralization</td></tr><tr><td colspan="6">Interpretation. The unqualified always/never statements support the annotated Overgeneralization category; Labeling is a related competing interpretation selected by SFT.</td></tr></table>

Table 7: Dreaddit transfer on all 715 test examples (F1, %). No D<sub>rea</sub>ddit <sub>superv</sub>i<sub>s</sub>i<sub>on</sub> i<sub>s use</sub>d<sub>.</sub>
<table><tr><td>Scale</td><td>Model / training</td><td>F1 (%)</td></tr><tr><td>2B</td><td>Qwen3.5 answer-SFT</td><td>74.32</td></tr><tr><td></td><td>PsyJev SFT</td><td>73.91</td></tr><tr><td></td><td>PsyJev JA-RLCD</td><td>77.76</td></tr><tr><td>4B</td><td>Qwen3.5 answer-SFT</td><td>76.53</td></tr><tr><td></td><td>PsyJev SFT</td><td>77.87</td></tr><tr><td></td><td>PsyJev JA-RLCD</td><td>78.70</td></tr><tr><td>3B</td><td>Llama-3.2-3B</td><td>78.00</td></tr><tr><td>4B</td><td>Gemma-3-4B</td><td>78.26</td></tr></table>

5.4.4 Unseen-Dataset Transfer. We evaluate transfer to Dreaddit, an unseen stress-detection dataset with 715 test examples excluded from PsyJudgeBench supervision [29]. As shown in Table 7, PsyJev achieves 77.76 F1 at 2B and 78.70 at 4B, improving over its directjudgment SFT initialization by 3.85 and 0.83 points, respectively. Both variants also outperform their corresponding answer-SFT baselines by 3.44 and 2.17 points. The 4B result is competitive with Gemma-3-4B (78.26) and Llama-3.2-3B (78.00). These results suggest that the benefits of judgment-anchored refinement can extend beyond the benchmark’s training datasets, with gains varying by model scale. Appendix C.2 reports the complete comparison.

5.4.5 Qualitative Analysis. Table 6 presents three test cases in which JA-RLCD corrects predictions from its SFT initialization. The cases cover co-occurring emotions, competing sources of distress, and closely related cognitive distortion categories. In Case A, PsyJev-4B recovers the annotated emotion set without the additional Sadness label predicted by SFT. In Case B, it identifies employment-related distress despite concurrent relationship difficulties. In Case C, it selects the annotated Overgeneralization category, whereas SFT favors Labeling. The answer model also selects the gold category. These examples illustrate how JA-RLCD can improve label selection across diferent psychological judgment criteria. Appendix C.4 shows the corresponding SFT-to-JA-RLCD probability shifts.

Scope of use. These judgments describe annotated textual content and do not establish clinical diagnoses. Reuse should follow each source dataset’s access and usage terms and safeguard the privacy of sensitive self-disclosures.

## 6 C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

We presented PsyJudgeBench and PsyJev for unified psychological analysis of Web content through question-conditioned judgments. PsyJudgeBench organizes heterogeneous tasks while retaining their distinct decision criteria, and PsyJev enables direct probabilistic prediction with JA-RLCD for judgment-anchored refinement. Across 2B, 4B, and 9B scales, PsyJev achieves average F1 scores of 64.33, 66.21, and 67.56, respectively, outperforming the corresponding answer-SFT baselines. Additional analyses examine question-conditioned discrimination, inference eficiency, and probability quality. These results demonstrate the value of a shared, question-conditioned approach to heterogeneous psychological content analysis.

## R<sub>e</sub>f<sub>erences</sub>

[1] Diogo Almeida. 2026. Introducing System One Models & Jev. TypeSafe AI blog. https://typesafe.ai/blog/introducing-system-one-models-and-jev

[2] Grace Byun, Rebecca Lipschutz, Sean T. Minton, Abigail Powers, and Jinho D. Choi. 2026. CRADLE Bench: A Clinician-Annotated Benchmark for Multi-Faceted Mental Health Crisis and Safety Risk Detection. In Proceedings ofthe 19th Conference ofthe European Chapter ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Rabat, Mo rocco, 1572–1590. doi:10.18653/v1/2026.eacl-long.73

[3] Hyung Won Chung et al. 2024. Scaling Instruction-Finetuned Language Models. Journal of Machine Learning Research 25, 70 (2024), 1–53. https://jmlr.org/papers/ v25/23-0870.html

[4] Abhimanyu Dubey et al. 2024. The Llama 3 Herd of Models. arXiv:2407.21783 https://arxiv.org/abs/2407.21783v1 Version 1.

[5] Zhimin Gao and Pichao Wang. 2026. OpenJev-RLCD: A Working RLCD Implementation. arXiv:2609.38850 [cs.AI] https://arxiv.org/abs/2609.38850

[6] Muskan Garg, Manas Gaur, Raxit Goswami, and Sunghwan Sohn. 2023. LoST: A Mental Health Dataset of Low Self-Esteem in Reddit Posts. In 2023 IEEE International Conference on Systems, Man, and Cybernetics (SMC). IEEE, Honolulu, Oahu, HI, USA, 3854–3859. doi:10.1109/SMC53992.2023.10394671

[7] Muskan Garg, Xingyi Liu, M. S. V. P. J. Sathvik, Shaina Raza, and Sunghwan Sohn. 2024. MultiWD: Multi-label wellness dimensions in social media posts. Journal ofBiomedical Informatics 150 (2024), 104586. doi:10.1016/j.jbi.2024.104586

[8] Muskan Garg, Chandni Saxena, Sriparna Saha, Veena Krishnan, Ruchi Joshi, and Vijay Mago. 2022. CAMS: An Annotated Corpus for Causal Analysis of Mental Health Issues in Social Media Posts. In Proceedings ofthe Thirteenth Language Resources and Evaluation Conference. European Language Resources Association, Marseille, France, 6387–6396. https://aclanthology.org/2022.lrec-1.686/

[9] Muskan Garg, Chandni Saxena, Debabrata Samanta, and Bonnie J. Dorr. 2023. LonXplain: Lonesomeness as a Consequence of Mental Disturbance in Reddit Posts. arXiv:2305.18736 https://arxiv.org/abs/2305.18736

[10] Muskan Garg, Amirmohammad Shahbandegan, Amrit Chadha, and Vijay Mago. 2023. An Annotated Dataset for Explainable Interpersonal Risk Factors of Mental Disturbance in Social Media Posts. In Findings ofthe Association for Computational Linguistics: ACL 2023. Association for Computational Linguistics, Toronto, Canada, 11960–11969. doi:10.18653/v1/2023.findings-acl.757

[11] Gemma Team et al. 2024. Gemma 2: Improving Open Language Models at a Practical Size. arXiv:2408.00118 https://arxiv.org/abs/2408.00118

[12] Gemma Team et al. 2025. Gemma 3 Technical Report. arXiv:2503.19786 https: //arxiv.org/abs/2503.19786

[13] Tilmann Gneiting and Adrian E. Raftery. 2007. Strictly Proper Scoring Rules, Prediction, and Estimation. J. Amer. Statist. Assoc. 102, 477 (2007), 359–378. doi:10.1198/016214506000001437

[14] Shrey Gupta, Anmol Agarwal, Manas Gaur, Kaushik Roy, Vignesh Narayanan, Ponnurangam Kumaraguru, and Amit Sheth. 2022. Learning to Automate Follow up Question Generation using Process Knowledge for Depression Triage on Reddit Posts. In Proceedings ofthe Eighth Workshop on Computational Linguistics and Clinical Psychology. Association for Computational Linguistics, Seattle, USA, 137–147. doi:10.18653/v1/2022.clpsych-1.12

[15] Amey Hengle, Atharva Kulkarni, Shantanu Deepak Patankar, Madhumitha Chandrasekaran, Sneha D’silva, Jemima S. Jacob, and Rashmi Gupta. 2024. Still Not Quite There! Evaluating Large Language Models for Comorbid Mental Health Diagnosis. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Miami, Florida, USA, 16698–16721. doi:10.18653/v1/2024.emnlp-main.931

[16] Shaoxiong Ji, Tianlin Zhang, Luna Ansari, Jie Fu, Prayag Tiwari, and Erik Cambria. 2022. MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare. In Proceedings ofthe Thirteenth Language Resources and Evaluation Conference. European Language Resources Association, Marseille, France, 7184–7190. https://aclanthology.org/2022.lrec-1.778/

[17] Laya Contributors. 2026. Laya. Software repository. https://github.com NandhaKishorM/laya/tree/682ac6719faa5b1c04b529501feb93e58ee91f3d Commit 682ac6719faa5b1c04b529501feb93e58ee91f3d.

[18] Matthew Louis Mauriello, Emmanuel Thierry Lincoln, Grace Hon, Dorien Simon, Dan Jurafsky, and Pablo E. Paredes. 2021. SAD: A Stress Annotated Dataset for Recognizing Everyday Stressors in SMS-like Conversational Systems. In Extended Abstracts ofthe 2021 CHI Conference on Human Factors in Computing Systems. ACM, New York, NY, USA, 1–7. doi:10.1145/3411763.3451799

[19] Meta. 2024. Llama-3.2-3B-Instruct Model Card. Hugging Face model repository. https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

[20] Microsoft et al. 2025. Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs. arXiv:2503.01743 https: //arxiv.org/abs/2503.01743

[21] OpenJev Contributors. 2026. OpenJev: Learning Open-Vocabulary Probabilistic Decision Models from Pretrained Language Models. Open-source implementa tion. https://github.com/ejhshen/OpenJev

[22] Anxo Pérez, Javier Parapar, Álvaro Barreiro, and Silvia López-Larrosa. 2023. BDI-Sen: A Sentence Dataset for Clinical Symptoms of Depression. In Proceedings ofthe 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, New York, NY, USA, 2996–3006. doi:10.1145/ 3539618.3591905

[23] Qwen Team. 2024. Qwen2.5: A Party of Foundation Models! https://qwenlm. github.io/blog/qwen2.5/ Baseline checkpoint: Qwen/Qwen2.5-7B-Instruct.

[24] Qwen Team. 2026. Qwen3.5. Hugging Face model collection. https://huggingface. co/collections/Qwen/qwen35

[25] A. B. S. Rahman, H. T. Ta, L. Najjar, A. Azadmanesh, and A. S. Gönül. 2024. DepressionEmo: A novel dataset for multilabel classification of depression emotions. Journal ofAfective Disorders 366 (2024), 445–458. doi:10.1016/j.jad.2024.08.013

[26] Ashish Sharma, Kevin Rushton, Inna Wanyin Lin, David Wadden, Khendra G. Lucas, Adam S. Miner, Theresa Nguyen, and Tim Althof. 2023. Cognitive Reframing of Negative Thoughts through Human-Language Model Interaction.

In Proceedings ofthe 61st Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Toronto, Canada, 9977–10000. doi:10.18653/v1/2023.acl-long.555

[27] Sagarika Shreevastava and Peter W. Foltz. 2021. Detecting Cognitive Distortions from Patient-Therapist Interactions. In Proceedings of the Seventh Workshop on Computational Linguistics and Clinical Psychology: Improving Access. Association for Computational Linguistics, Online, 151–158. doi:10.18653/v1/2021.clpsych-1.17

[28] Ihor Stepanov, Mykhailo Shtopko, Dmytro Vodianytskyi, Oleksandr Lukashov, Alexander Yavorskyi, and Mykyta Yaroshenko. 2025. GLiClass: Generalist Lightweight Model for Sequence Classification Tasks. arXiv:2508.07662 https: //arxiv.org/abs/2508.07662

[29] Elsbeth Turcan and Kathy McKeown. 2019. Dreaddit: A Reddit Dataset for Stress Analysis in Social Media. In Proceedings ofthe Tenth International Workshop on Health Text Mining and Information Analysis (LOUHI 2019). Association for Computational Linguistics, Hong Kong, 97–107. doi:10.18653/v1/D19-6213

[30] Benjamin Warner, Antoine Chafin, Benjamin Clavié, Orion Weller, Oskar Hall ström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Nathan Cooper, Grifin Adams, Jeremy Howard, and Iacopo Poli. 2025. Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Eficient, and Long Context Finetuning and Inference. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Vienna, Austria, 2526–2547. doi:10.18653/v1/2025.acl-long.127

[31] Xuhai Xu, Bingsheng Yao, Yuanzhe Dong, Saadia Gabriel, Hong Yu, James Hendler, Marzyeh Ghassemi, Anind K. Dey, and Dakuo Wang. 2024. Mental-LLM: Leveraging Large Language Models for Mental Health Prediction via Online Text Data. Proceedings ofthe ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 8, 1, Article 31 (2024), 32 pages. doi:10.1145/3643540

[32] An Yang, Anfeng Li, Baosong Yang, et al. 2025. Qwen3 Technical Report. arXiv:2505.09388 https://arxiv.org/abs/2505.09388

[33] Kailai Yang, Tianlin Zhang, Ziyan Kuang, Qianqian Xie, Jimin Huang, and Sophia Ananiadou. 2024. MentaLLaMA: Interpretable Mental Health Analysis on Social Media with Large Language Models. In Proceedings ofthe ACM Web Conference 2024. ACM, New York, NY, USA, 4489–4500. doi:10.1145/3589334.3648137

[34] Urchade Zaratiana, Gil Pasternak, Oliver Boyd, George Hurn-Maloney, and Ash Lewis. 2025. GLiNER2: Schema-Driven Multi-Task Learning for Structured In formation Extraction. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. Association for Computational Linguistics, Suzhou, China, 130–140. doi:10.18653/v1/2025.emnlpdemos.10

[35] Wei Zhai, Nan Bai, Qing Zhao, Jianqiang Li, Fan Wang, Hongzhi Qi, Meng Jiang, Xiaoqin Wang, Bing Xiang Yang, and Guanghui Fu. 2025. MentalGLM Series: Explainable Large Language Models for Mental Health Analysis on Chinese Social Media. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Suzhou, China, 13588–13603. doi:10.18653/v1/2025.emnlp-main.686

[36] Yuhan Zhu, Xiangyu Zeng, Qingyi Si, Chenxu Yang, Yifei Li, Luozheng Qin, Junhao Zhou, Dingyu Yao, Jun Zhang, Chuanyu Qin, Zhiqiu Zhang, Yuandong Yang, Yuqi Xu, Tianyu Zhang, Hongcheng Guo, Jiaqi Wang, Nan Duan, and Limin Wang. 2026. VideoJev: System One Model for Video Understanding and Interaction. arXiv:2610.18615 https://arxiv.org/abs/2610.18615

## A D<sub>a</sub>t<sub>a</sub> C<sub>o</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n <sub>a</sub>nd S<sub>p</sub>lit<sub>s</sub>

Label construction. Missing labels are masked. BDI-Sen retains symptom relevance, excluding severity. TherapistQA uses the patient question and dominant distortion, excluding therapist responses, spans, and secondary labels. Thinking Traps merges normalized repeated completions into 15-category label sets; CRADLE expands seven crisis categories across ongoing/past status into 14 questions. Missing CAMS categories remain unobserved. Table 9 gives four verbatim registry entries.

Split coordination. Exact texts and available source groups are coordinated across all 13 sources, with test taking priority over validation and training. Normalization applies Unicode NFKC, case folding, and whitespace collapse. Shared texts retain their distinct source-defined judgments. We mask 22 conflicting judgments across 11 identical text–target keys.

Table 8: Source records and coordinated splits. Q counts registr<sub>y</sub> <sub>q</sub>uestions: 79 binar<sub>y</sub> and three multiclass in total.
<table><tr><td>Source</td><td>Q</td><td>Train</td><td>Val.</td><td>Test</td></tr><tr><td>Depression / anxiety</td><td></td><td></td><td></td><td></td></tr><tr><td>DepressionEmo</td><td>8</td><td>4,219</td><td>907</td><td>911</td></tr><tr><td>ANGST</td><td>2</td><td>6,769</td><td>763</td><td>2,872</td></tr><tr><td>PRIMATE</td><td>9</td><td>1,593</td><td>207</td><td>203</td></tr><tr><td>BDI-Sen</td><td>21</td><td>1,759</td><td>261</td><td>503</td></tr><tr><td>Self / relationships</td><td></td><td></td><td></td><td></td></tr><tr><td>IRF</td><td>2</td><td>1,267</td><td>483</td><td>1,771</td></tr><tr><td>LoST</td><td>1</td><td>740</td><td>271</td><td>1,163</td></tr><tr><td>LonXplain</td><td>1</td><td>1,267</td><td>483</td><td>1,771</td></tr><tr><td>Stress / distress</td><td></td><td></td><td></td><td></td></tr><tr><td>SAD</td><td>1</td><td>5,533</td><td>655</td><td>662</td></tr><tr><td>CAMS</td><td>1</td><td>2,472</td><td>643</td><td>1,920</td></tr><tr><td>Cognition</td><td></td><td></td><td></td><td></td></tr><tr><td>Thinking Traps</td><td>15</td><td>537</td><td>68</td><td>93</td></tr><tr><td>TherapistQA</td><td>1</td><td>2,023</td><td>245</td><td>260</td></tr><tr><td>Wellness</td><td></td><td></td><td></td><td></td></tr><tr><td>MultiWD</td><td>6</td><td>1,155</td><td>444</td><td>1,682</td></tr><tr><td>Crisis</td><td></td><td></td><td></td><td></td></tr><tr><td>CRADLE</td><td>14</td><td>4,177</td><td>422</td><td>601</td></tr><tr><td>Total</td><td>82</td><td>33,511</td><td>5,852</td><td>14,412</td></tr></table>

Table 9: Verbatim re<sub>g</sub>istr<sub>y</sub> instructions and le<sub>g</sub>al answers for four judgment types in PsyJudgeBench.
<table><tr><td>Source</td><td>Registry instruction and legal answers</td></tr><tr><td></td><td>DepressionEmo Does the author express sadness in this post? Answers: no; yes</td></tr><tr><td>PRIMATE</td><td>Does this post provide enough information to answer the PHQ-9 item about Feeling down depressed or hopeless? Judge answerability, not whether the symptom is present. Answers: no; yes</td></tr><tr><td>BDI-Sen</td><td>Does this sentence provide information relevant to the BDI-II symptom Sadness? Judge the source relevance label, not severity or symptom presence. Answers: no; yes</td></tr><tr><td>SAD</td><td>What is the primary stressor category expressed in this sentence? Answers: Financial Problem; Other; Everyday Decision</td></tr></table>

TherapistQA refinement. After removing two identical questions with conflicting labels, normalized-question groups receive a seed-42 hash split, inheriting exact-match assignments. Candidates use up to 12 smallest distinct CRC32 four-token shingle hashes and the first 128 indexed texts per hash. Components satisfying the Jaccard or containment criteria in Table 10 follow the same held-out priority; the other 12 sources remain unchanged. ANGST retains silver training/validation and expert-labeled test annotations; CRADLE’s coordinated test partition contains 600 gold and one silver record.

Final-split audit. The frozen snapshot contains 53,775 records and 41,349 normalized texts. All 3,600 cross-source shared-text groups are confined to one split (training: 1,301; validation: 494; test: 1,805), including all 3,518 IRF–LonXplain shared texts. Table 10 checks every partition pair using uncapped prefix filtering and exact four-word shingles, on both full texts and their first 512 tokens. No source-group identifier crosses partitions. The audit measures lexical instance reuse.

T<sub>a</sub>bl<sub>e</sub> 10<sub>:</sub> C<sub>ross-par</sub>titi<sub>on</sub> <sub>over</sub>l<sub>ap</sub> <sub>on</sub> f<sub>u</sub>ll t<sub>ex</sub>t<sub>s</sub> <sub>an</sub>d 512<sub>-</sub>t<sub>o</sub>k<sub>en</sub> <sup>i</sup>n<sub>p</sub>uts.
<table><tr><td>Cross-partition check</td><td>Full text</td><td>512 tokens</td></tr><tr><td>Exact normalized matches</td><td>0 (0.00%)</td><td>0 (0.00%)</td></tr><tr><td>Jaccard ≥ 0.85, ≥ 8 shingles</td><td>0</td><td>0</td></tr><tr><td>Containment  $( \geq$  80 characters)</td><td>0</td><td>0</td></tr></table>

## B Re<sub>p</sub>roducibilit<sub>y</sub> and Com<sub>p</sub>arison Protocols

## B.1 In<sub>p</sub>ut Processing

We use Qwen/Qwen3.5-2B, Qwen/Qwen3.5-4B, and Qwen/Qwen3.5-9B (revision prefixes 15852e8c, 851bf6e8, and c2022362), each with its native tokenizer. Input texts are encoded without added special tokens and truncated to 512 tokens, while question criteria and legal answer descriptions are preserved in full and in registry order. Requests serialize the text under Context:, followed by Question: and letter-prefixed answer options. We apply the backbone chat template with thinking disabled and append a separately tokenized Answer: prefix for constrained readout. For multilabel sources, each observed binary question is scored separately; a positive label is assigned only when $p ( \mathrm { y e s } ) > p ( \mathrm { n o } )$ , with ties assigned to no.

## B.2 Trainin<sub>g</sub> and Matched Controls

Optimizer and accumulation. All reported runs use seed 42. Each stage visits all 33,511 training records once in 524 optimizer updates, with efective batches of 63 or 64 records, retaining its final checkpoint. All observed questions are retained in record/question order, grouped by source, and split into consecutive source-homogeneous microbatches of at most 16 judgments. Their weighted gradients are summed over each complete record batch before one optimizer step; the microbatch count varies with the observed labels. AdamW uses $\beta _ { 1 } ~ = ~ 0 . 9 , \beta _ { 2 } ~ = ~ 0 . 9 9 9 , \epsilon ~ = ~ 1 0 ^ { - 8 }$ , weight decay 0.01, global gradient clipping at 1, and 3% warmup followed by cosine decay. The 2B model uses full tuning a ${ \textrm { t } } 2 \times 1 0 ^ { - 6 }$ in both stages. At 4B/9B, LoRA uses rank 16, scaling 32, and zero dropout, targeting attention projections $( \mathsf { q } / \mathsf { k } / \mathsf { v } / \mathsf { o \_ p r o j } )$ , linear-attention input/output projections (in\_proj\_qkv, in\_proj\_z, in\_proj\_b, in\_proj\_a, and out\_proj), and MLP gate, up, and down projections. The vocabulary head remains frozen; all trainable adapters share learning rates of $2 \times 1 0 ^ { - 4 }$ for SFT and $2 \times 1 0 ^ { - 6 }$ for refinement.

Weights and optimization. For training counts $n _ { q k }$ and $n _ { q } ~ =$ $\textstyle \sum _ { k } n _ { q k }$ , we set

$$
\widehat { \pi } _ { q k } = \frac { n _ { q k } + 1 } { n _ { q } + K _ { q } } , \qquad c _ { q k } = \operatorname * { m i n } ( 8 , \eta _ { q } / \widehat { \pi } _ { q k } ) ,\tag{8}
$$

where $\eta _ { q }$ ensures $\begin{array} { r } { \sum _ { k } \widehat { \pi } _ { q k } c _ { q k } = 1 } \end{array}$ . For � training records, � sources, efective record batch size $B ,$ and source question set $Q _ { s }$

$$
a _ { i } = \frac { N } { B S | Q _ { s _ { i } } | n _ { q _ { i } } } .\tag{9}
$$

The anchor applies $a _ { i }$ per judgment to balance source/question observation frequency; the reward branch uses only their sum $A _ { \mathcal { B } }$ in Eq. 6. Priors and weights derive from training counts; referencecorrectness gates also use training labels only. Refinement initializes from the same-scale final SFT checkpoint, keeping a frozen refer ence copy. Each microbatch uses two Gaussian directions and their negatives, with all four reward evaluations sharing the same model logits. The noise scale decreases linearly from 0.4 to 0.1 over refinement updates; $\tau = 4$ and $\lambda = 0 . 0 5$ . With � the pathwise reward gradient, the update accumulates $g - u$ over microbatches and clips the total before AdamW. The refinement configuration was selected using validation only, without test results.

## B<sub>.</sub>3 B<sub>ase</sub>lin<sub>e</sub> Pr<sub>o</sub>t<sub>oco</sub>l<sub>s</sub>

All benchmark-adapted baselines use the same cleaned records, question registry, and observed-label masks. Fixed-head classifiers use question-specific output heads, while description-driven models preserve the original decision criteria in their input schemas. Answer-SFT models score complete legal-answer sequences, including EOS. All models use their final training checkpoints, except ModernBERT, which selects among three epochs based on validation F1, followed by NLL and earlier-epoch tie-breaking. MentaL-LaMA undergoes benchmark-specific answer SFT from its existing mental-health instruction-tuned checkpoint [33].

Open-source Laya. We reproduce Laya’s SFT and RLCD training pipeline on PsyJudgeBench using the released Laya-multilingual checkpoint (revision 1720e3e3), preserving its native architecture and RLCD objective. We first perform one epoch of full-parameter SFT (524 updates, learning rate $1 0 ^ { - 5 } )$ , followed by one epoch of RLCD (524 updates, learning rate 2 × 10<sup>−6</sup>). RLCD combines crossentropy with score-function optimization over Gaussian-perturbed answer logits, using four samples, a noise schedule from 0.4 to 0.1, and spherical/ranked-probability reward weights of 0.75/1.0. Both stages use the benchmark training partition and are evaluated using their final checkpoints.

Jev APIevaluation. We evaluate jev-1.14.0 through Typesafe’s oficial API on all 44,999 test judgments. Each request includes the source text, question criterion, and legal answers, with the source text capped at 512 Qwen tokens. Binary and multiclass questions use Noul and Choice, respectively. The API is evaluated zero-shot without benchmark fine-tuning or demonstrations, and its returned probabilities are aligned with the benchmark’s semantic labels. All 84 invalid responses remain in the evaluation as incorrect predictions; the invalid label is excluded from the legal-class macro average when computing F1. Results follow the same question- and source-level aggregation as the main evaluation.

Matched refinement controls. All direct-scoring 2B continuations start independently from the same final SFT checkpoint and retain the same records, registry, observed-label masks, record schedule, trainable parameters, optimizer, and 524 additional updates. Adapted RLCD combines ordinary CE with Laya’s categorical scorefunction objective [17]: four independent centered Gaussian logit actions, advantages centered and normalized by their standard deviation, and reward weights of 0.75 for the spherical term and 1.0 for ranked probability. It uses SFT source/record weights without the JA reference gate or question-group aggregation.

![](images/a593f2e9b5c9bf01c83d412d94db32b88c19b5c76bc1543695d7097cb20e0d24.jpg)  
Figure 5: Quality–latency comparison of the 16 models with average $F 1 \ge 5 7 \%$ i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 1<sub>.</sub>

Ablation configurations. The anchor-only variant removes reward optimization while retaining balanced supervision and selective reference KL. Removing class costs sets $c _ { q k } = 1$ in both the anchor and reward objectives, while removing reference KL sets $\lambda = 0 .$ The score-function variant replaces pathwise diferentiation with paired score-function gradients, retaining the same anchor, question-group reward, and logit-space perturbations.

The balanced-CE-only control disables both reference KL and reward optimization, retaining the source/question weights and class costs. The ordinary Brier control replaces the group/tail reward with one minus the class-cost-normalized mean per-judgment half-Brier error, retaining the anchor, perturbations, reward scale, and pathwise estimator. This changes grouping and tail weighting jointly. The answer-sequence balanced-CE control starts from its corresponding answer-SFT checkpoint and uses full-vocabulary CE on the answer letter and EOS. It matches the direct balanced-CE control’s pretrained backbone, record schedule, weights, optimizer, and two total training epochs, including 524 continuation updates.

## B<sub>.</sub>4 U<sub>n</sub>ifi<sub>e</sub>d Ti<sub>m</sub>i<sub>ng</sub> P<sub>ro</sub>t<sub>oco</sub>l

Model computation time is measured on a shared probe comprising 13 training texts (the second record in each source’s stored training order) and 64 observed judgments. Each request contains one text and its observed questions and is repeated five times after warmup. All models run on a single NVIDIA H100 80GB using BF16 and CUDA Graphs. CUDA events measure GPU execution, including text prefill and answer scoring, while excluding loading, tokenization, data transfer, and graph capture. Inputs reside on the GPU, and latency is reported as total measured time divided by the number of judgments. Each model retains its native scoring mechanism under the same protocol. For the matched 4B comparison, full-answer and first-token scoring share the Qwen3.5-4B answer-SFT checkpoint, while PsyJev uses its final JA-RLCD checkpoint. Their respective latencies are 25.25, 11.87, and 12.01 ms per judgment. Figure 3 pairs these measurements with full-test F1, while Figure 5 compares quality and latency across the models in Table 1.

Table 11: Probability quality of the final PsyJev-4B by source. The mean <sub>g</sub>ives e<sub>q</sub>ual wei<sub>g</sub>ht to all 13 sources.
<table><tr><td>Source</td><td>Brier ↓</td><td>ECE (%) ↓</td></tr><tr><td>ANGST</td><td>0.4436</td><td>18.54</td></tr><tr><td>BDI-Sen</td><td>0.1499</td><td>1.85</td></tr><tr><td>CAMS</td><td>0.5408</td><td>4.33</td></tr><tr><td>CRADLE</td><td>0.0644</td><td>0.76</td></tr><tr><td>DepressionEmo</td><td>0.2327</td><td>1.06</td></tr><tr><td>IRF</td><td>0.1895</td><td>3.97</td></tr><tr><td>LoST</td><td>0.1889</td><td>3.41</td></tr><tr><td>LonXplain</td><td>0.2179</td><td>6.06</td></tr><tr><td>MultiWD</td><td>0.2143</td><td>0.84</td></tr><tr><td>PRIMATE</td><td>0.2444</td><td>2.35</td></tr><tr><td>SAD</td><td>0.2997</td><td>4.34</td></tr><tr><td>TherapistQA</td><td>0.6560</td><td>7.18</td></tr><tr><td>Thinking Traps</td><td>0.1676</td><td>4.86</td></tr><tr><td>Source mean</td><td>0.2777</td><td>4.58</td></tr></table>

## C P<sub>re</sub>di<sub>c</sub>ti<sub>on</sub> Di<sub>agnos</sub>ti<sub>cs</sub>

## C<sub>.</sub>1 Pr<sub>o</sub>b<sub>a</sub>bilit<sub>y</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>o</sub>n Pr<sub>o</sub>t<sub>oco</sub>l

All probability evaluations use the same 44,999 test judgments and legal answer sets. Classification models normalize their answer scores over the legal options, while answer-generation models normalize complete answer-sequence likelihoods, including EOS. No post-hoc calibration is applied. For a judgment with � answers, normalized NLL and Brier score are defined as:

$$
{ \mathrm { N L L } } = - { \frac { \log { \hat { p } _ { y } } } { \log { K } } } , \qquad { \mathrm { B r i e r } } = \sum _ { k = 1 } ^ { K } ( p _ { k } - \mathbb { I } [ k = y ] ) ^ { 2 } .\tag{10}
$$

NLL is averaged over judgments within records and then equally across records and sources. Brier score and 15-bin ECE are computed at the source level and averaged equally across sources.

For Figure 4, let $n _ { s , b } , a _ { s , b } ,$ , and $c _ { s , b }$ denote the count, accuracy, and mean confidence in source � and bin �. The source-bin contribution and cumulative curve are

$$
e _ { s , b } = \frac { n _ { s , b } } { n _ { s } } | a _ { s , b } - c _ { s , b } | , \qquad E _ { \leq b } = \frac { 1 } { 1 3 } \sum _ { s } \sum _ { j = 1 } ^ { b } e _ { s , j } .
$$

Empty bins contribute zero. The denominator $n _ { s }$ is the full source judgment count, so $E _ { \leq 1 5 }$ equals the source-mean ECE in Table 5.

## C<sub>.</sub>2 Un<sub>see</sub>n-D<sub>a</sub>t<sub>ase</sub>t Tr<sub>a</sub>n<sub>s</sub>f<sub>e</sub>r

All models use their benchmark-trained checkpoints on the complete 715-example Dreaddit test set [29], without Dreaddit supervision or further tuning. The registered question is Does the author express psychological stress in this text? Legal answers are ordered (no, yes), mapping original labels 0 and 1, respectively. Texts retain their first 512 tokens; prediction uses legal-answer argmax, and F1 treats yes as positive. Auditing all 715 texts against the actual Psy-JudgeBench training (33,511) and validation (5,852) records found no raw/normalized exact matches, no word-level 4-gram Jaccard matches ≥ 0.85 with uncapped candidate retrieval, and no full-text containments with the shorter text at least 80 characters. Table 12 lists all 12 configurations.

Table 12: Complete Dreaddit transfer results (F1, %) without additional trainin<sub>g</sub> on Dreaddit.
<table><tr><td>Model / training</td><td>F1 (%)</td></tr><tr><td>GLiClass</td><td>66.56</td></tr><tr><td>GLiNER2</td><td>70.94</td></tr><tr><td>FLAN-T5-XL</td><td>74.84</td></tr><tr><td>Llama-3.2-3B</td><td>78.00</td></tr><tr><td>Phi-4-mini</td><td>73.27</td></tr><tr><td>Gemma-3-4B</td><td>78.26</td></tr><tr><td>Qwen3.5-2B answer-SFT</td><td>74.32</td></tr><tr><td>PsyJev-2B SFT</td><td>73.91</td></tr><tr><td>PsyJev-2B JA-RLCD</td><td>77.76</td></tr><tr><td>Qwen3.5-4B answer-SFT</td><td>76.53</td></tr><tr><td>PsyJev-4B SFT</td><td>77.87</td></tr><tr><td>PsyJev-4B JA-RLCD</td><td>78.70</td></tr></table>

Table 13: Diferent <sub>q</sub>uestions about the same text. Outcomes (%) use identical opposite-label binary pairs and equal source <sub>we</sub>i<sub>g</sub>ht<sub>s.</sub> “S<sub>ame answer</sub>” <sub>means exac</sub>tl<sub>y one pre</sub>di<sub>c</sub>ti<sub>on</sub> i<sub>s</sub> wron<sub>g</sub> in this subset.
<table><tr><td>Model</td><td>Both correct ↑</td><td>Same answer</td><td>Both wrong</td></tr><tr><td>GLiClass</td><td>55.84</td><td>41.88</td><td>2.28</td></tr><tr><td>Gemma-2-9B</td><td>67.85</td><td>29.84</td><td>2.31</td></tr><tr><td>Qwen3.5-4B answer</td><td>66.24</td><td>31.95</td><td>1.82</td></tr><tr><td>PsyJev-4B SFT</td><td>63.55</td><td>34.37</td><td>2.09</td></tr><tr><td>PsyJev-4B</td><td>70.06</td><td>28.09</td><td>1.84</td></tr></table>

## C.3 Questions about Shared Text

Table 13 adds PsyJev-4B SFT and both-wrong outcomes to the main comparison. All models use identical opposite-label binary pairs, with three outcomes. Results are source-balanced, with paired textgroup resampling accounting for shared-text dependence.

## C<sub>.</sub>4 C<sub>ase</sub>-L<sub>eve</sub>l Pr<sub>o</sub>b<sub>a</sub>bilit<sub>y</sub> Shift<sub>s</sub>

Table 14 follows the three 4B cases in Table 6, using the final JA-RLCD checkpoint and its SFT parent. In Case A, the Sadness probability crosses below 0.5 while Emptiness and Loneliness remain positive. In Cases B and C, the gold label overtakes the strongest competing label; both labels’ probabilities are shown to distinguish relative preference changes from uniform confidence increases. These cases use DepressionEmo:16ll3jf, CAMS:1286, and TherapistQA:364. Models receive the original inputs; display excerpts preserve their order, mark omissions, and normalize punctuation.

Table 14: Answer probabilities before and after JA-RLCD for th<sub>e</sub> th<sub>ree cases</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 6<sub>.</sub>
<table><tr><td>Case</td><td>Answer probability / margin</td><td>SFT</td><td>JA-RLCD</td></tr><tr><td>A</td><td>Emptiness</td><td>0.752</td><td>0.825</td></tr><tr><td>A</td><td>Loneliness</td><td>0.999</td><td>1.000</td></tr><tr><td>A</td><td>Sadness</td><td>0.571</td><td>0.445</td></tr><tr><td>B</td><td>Jobs and careers</td><td>0.303</td><td>0.402</td></tr><tr><td>B</td><td>Relationship</td><td>0.406</td><td>0.389</td></tr><tr><td>B</td><td> $p _ { y } - p _ { c }$ </td><td>-0.103</td><td>+0.013</td></tr><tr><td>C</td><td>Overgeneralization</td><td>0.319</td><td>0.433</td></tr><tr><td>C</td><td>Labeling</td><td>0.336</td><td>0.414</td></tr><tr><td>C</td><td> $p _ { y } - p _ { c }$ </td><td>-0.017</td><td>+0.019</td></tr></table>