# RAG-PIBench: A Leakage-Aware Benchmark for Prompt-Injection Detection in Trustworthy RAG Systems

Niveen O. Jaffal<sup>1</sup>, Ahmet Yuksel<sup>2</sup>, and David Mohaisen<sup>3</sup>

<sup>1</sup> Birzeit University, Birzeit, Palestine

njaffal@birzeit.edu

2 Independent Researcher

asyuksel5@gmail.com

3 University of Central Florida, Orlando, FL, USA mohaisen@ucf.edu

Abstract. Retrieval-Augmented Generation (RAG) systems are vulnerable to prompt-injection attacks embedded in retrieved content. We introduce RAG-PIBench, a benchmark for RAG-style prompt-injection detection containing 4,876 contextual examples across frozen train, validation, and protected-test splits. Using a leakage-aware construction pipeline and strict evaluation protocol, we compare keyword-based, semantic-reference, TF-IDF, and transformer-based detectors. DistilBERT achieves the best protected-test performance (F<sub>1</sub> = 0.896, PR-AUC = 0.968), while TF-IDF SVM and logistic regression remain competitive. Our results demonstrate the value of leakage-aware benchmark design and strong sparse baselines for reliable prompt-injection detection in RAG systems.

Keywords: Prompt injection · Retrieval-Augmented Generation · RAG security · Large language models · Benchmark construction · Prompt-injection detection

## 1 Introduction & Problem Statement

Retrieval-Augmented Generation (RAG) has become a widely adopted approach for improving large language models (LLMs) by incorporating external documents into the generation context [21,16,15]. While RAG can improve factual grounding and domain adaptation [17], it also expands the attack surface of LLM-based systems. In particular, prompt-injection attacks can be embedded within retrieved content, causing the model to follow adversarial instructions rather than the intended system objective [18,2].

Detecting prompt injection in RAG contexts is challenging because benign documents may naturally contain instructions, code snippets, examples, or task descriptions. Moreover, malicious instructions can be paraphrased, blended into otherwise legitimate content, or designed to resemble benign contextual information. As a result, simple keyword-based defenses often fail, while more sophisticated approaches may over-flag benign content or rely on dataset artifacts rather than robust semantic signals [4,6,5].

A variety of detection approaches have been proposed, ranging from heuristic filters and semantic-reference methods to sparse-feature classifiers and transformer-based models [3,12]. However, comparisons across these detector families are often complicated by inconsistent evaluation protocols, potential leakage between training and testing data, and the absence of strong lightweight baselines.

Problem Statement. Despite growing interest in prompt-injection defenses, there remains a need for leakage-aware benchmarks that evaluate whether detectors can reliably distinguish benign contextual instructions from malicious instruction-override attempts in realistic RAG inputs. Existing evaluations may overestimate performance due to lexical shortcuts, repeated templates, or insufficient separation between dataset construction and testing. This work addresses the problem of constructing and evaluating a reproducible RAG-style prompt-injection benchmark that supports fair comparison across heuristic, semantic, sparse-feature, and transformer-based detectors.

Research Questions. This study is guided by four research questions: RQ1. How effectively do different detector families distinguish benign RAG-style contextual inputs from malicious prompt-injection under a frozen protected-test evaluation protocol? RQ2. Do compact transformer classifiers provide measurable gains over strong baseline detectors, or can sparse lexical and character-level models remain competitive? RQ3. What precision–recall trade-offs and failure modes emerge across keyword heuristics, semantic-reference methods, TF-IDF classifiers, and transformer models? RQ4. How can dataset construction, split isolation, diagnostic separation, and artifact registration reduce leakage and improve the reliability of prompt-injection detector evaluation?

Contributions. To address these questions, this paper makes five contributions: ❶ RAG-PIBench Benchmark. We introduce RAG-PIBench, a balanced benchmark for RAG-style prompt-injection detection containing 4,876 rendered contextual examples across frozen train, validation, and protected-test splits. ❷ Leakage-Aware Construction Pipeline. We develop a dataset-construction pipeline incorporating source registration, provenance tracing, quality control, style-matched rendering, shortcut auditing, and frozen split generation to reduce shortcut learning and improve auditability. ❸ Protected-Test Evaluation Protocol. We implement a strict train/validation/protectedtest protocol that isolates model training, selection, and final evaluation while keeping diagnostic analyses separate from tuning decisions. ❹ Systematic Baseline Evaluation. We compare keyword rules, semantic-reference methods, TF-IDF LR, TF-IDF SVM, TF-IDF RF, DistilBERT, and DeBERTa-v3 to assess whether transformer models provide meaningful gains over strong lightweight baselines. ❺ Transparent Findings and Failure Modes. We report both strengths and weaknesses of the evaluated detectors, showing that DistilBERT achieves the best protected-test performance while TF-IDF SVM and TF-IDF LR remain highly competitive, and identifying key failure modes across detector families.

Positioning. RAG-PIBench extends our broader line of work on trustworthy and secure LLMs, which has examined LLM attack surfaces and defenses, security of generated code, attribution of LLM-generated artifacts, and systematic evaluation of LLMs in security applications [9,10,1,13]. Here, we focus specifically on indirect prompt injection in RAG settings and on leakage-aware evaluation of prompt-injection detectors.

## 2 Related Work

Research on prompt-injection security spans benchmark construction, detector design, and RAG-specific defenses. We focus on work most closely related to leakage-aware evaluation, indirect prompt injection, and detection in retrieval-augmented settings.

Prompt-Injection Benchmarks. Recent benchmarks have moved beyond simple promptlevel attacks toward indirect, application-oriented, and agent-based threat models. Prompt-Shield [8] studies practical prompt-injection detection under realistic false-positive constraints, while PromptSleuth [20] investigates robustness under paraphrased and distributionshifted attacks. BIPIA [22] focuses on indirect prompt injection through retrieved content in RAG workflows, whereas INJECAGENT [23] and WAInjectBench [14] extend evaluation to tool-integrated and web-agent environments. Collectively, these benchmarks demonstrate the importance of contextualized attacks and distribution-aware evaluation, but they provide limited emphasis on leakage-aware detector assessment and protected-test benchmarking.

Prompt-Injection Defenses in RAG. Defenses against prompt injection range from lightweight filters and semantic detectors to multi-stage RAG frameworks. Prior work has explored adaptive defense pipelines [6], post-generation validation using auxiliary LLMs [7], retrieval-aware defenses such as TrustRAG [24] and SeCon-RAG [19], and hybrid architectures combining rules, classifiers, and companion models [11]. While these approaches improve robustness, their evaluation protocols vary considerably, making direct comparison difficult. Reliable detector assessment therefore requires controlled benchmarks with strict separation between training, validation, and final testing.

Positioning of This Work. RAG-PIBench complements existing benchmarks and defenses by focusing on leakage-aware binary detection of prompt injection in RAGstyle contextual inputs. Unlike prior studies that primarily measure attack success or evaluate a single detector family, we compare heuristic, semantic, sparse-feature, and transformer-based detectors under a frozen train/validation/protected-test protocol. The benchmark further incorporates provenance tracking, quality control, contextual rendering, shortcut auditing, and strict split isolation to support reproducible evaluation.

Table 1 summarizes the positioning of RAG-PIBench relative to representative promptinjection benchmarks and trustworthy-RAG defenses. The comparison highlights three distinguishing characteristics of our study: (i) leakage-aware benchmark construction, (ii) evaluation against strong lightweight baselines in addition to transformer models, and (iii) explicit reporting of detector failure modes alongside successful results.

## 3 Methodology

This section describes the construction of RAG-PIBench and the evaluation protocol used to assess prompt-injection detectors in a controlled, leakage-aware setting. The methodology supports fair comparison across heuristic, sparse-feature, and transformerbased models while maintaining strict separation between training, validation, protected testing, and diagnostic analyses.

Table 1. Comparison of representative prompt-injection benchmarks, detection frameworks, and trustworthy-RAG defenses. Columns indicate whether the work explicitly addresses indirect prompt injection (IPI), retrieval-augmented generation (RAG), text and image modalities, leakage-aware or protected evaluation protocols, detector baseline comparisons, and failure-mode analysis. Modalities are text (T) or image (I). The attributes of the work are leakage (L), baselines (B), and failures (F)
<table><tr><td rowspan="2">Work</td><td rowspan="2">Yr.</td><td rowspan="2">Setting</td><td rowspan="2">IPI RAG</td><td rowspan="2"></td><td colspan="2">Modality</td><td rowspan="2">Goal</td><td colspan="4">Attributes</td></tr><tr><td>T</td><td>I</td><td></td><td>L B</td><td></td><td>F</td></tr><tr><td>Palisade [11]</td><td>2024</td><td>PI detection</td><td>o</td><td></td><td></td><td></td><td>Filter/classifier defense</td><td></td><td>一</td><td></td><td>0</td></tr><tr><td>INJECAGENT [23]</td><td>2024</td><td>Tool-agent PI</td><td>√</td><td></td><td></td><td></td><td>Agent vulnerability</td><td></td><td>一</td><td></td><td>0</td></tr><tr><td>PromptShield [8]</td><td>2025</td><td>PI detection</td><td>√</td><td></td><td>√</td><td></td><td>PI detection</td><td></td><td>一</td><td></td><td>0</td></tr><tr><td>PromptSleuth [20]</td><td>2025</td><td>Semantic PI det.</td><td>√</td><td></td><td>√</td><td></td><td>Semantic detection</td><td></td><td>o</td><td></td><td>o</td></tr><tr><td>BIPIA [22]</td><td>2025</td><td>External-content PI</td><td>√</td><td>√</td><td>√ √</td><td>一</td><td>Vulnerability and defense</td><td></td><td>o</td><td></td><td>0</td></tr><tr><td>WAInjectBench [14]</td><td>2025</td><td>Web-agent PI</td><td>√</td><td>一</td><td></td><td>√</td><td>Web-agent detection</td><td></td><td>o</td><td></td><td>L</td></tr><tr><td>TrustRAG [24]</td><td>2025</td><td>Trustworthy RAG</td><td>o</td><td>√</td><td>√</td><td>一</td><td>RAG defense</td><td></td><td>一</td><td></td><td>o</td></tr><tr><td>SeCon-RAG [19]</td><td>2025</td><td>RAG conflict filter</td><td>一</td><td>√</td><td>√</td><td>一</td><td>RAG defense</td><td></td><td>一</td><td></td><td>0</td></tr><tr><td>RAG-PIBench</td><td>2026</td><td>RAG-style contex- tual PI detection</td><td>√</td><td>√</td><td>√</td><td>一</td><td>Leakage-aware PI detection</td><td></td><td>√√</td><td></td><td>√</td></tr></table>

## 3.1 Threat Model

We consider an indirect prompt-injection threat model for RAG-style systems in which an attacker controls or influences external textual content that may be retrieved and incorporated into the model context. The attacker does not control the user query or system prompt; instead, malicious instructions are embedded within ordinary retrieved content. The detector operates before generation on the rendered RAG-style input and performs binary classification. Malicious examples are defined as inputs that attempt to override, bypass, redirect, or manipulate the assistant’s intended instruction-following behavior, while benign examples contain contextual content without such intent.

Our focus is prompt-injection detection rather than end-to-end attack execution. Accordingly, we do not evaluate downstream generation outcomes, tool misuse, data exfiltration, retrieval poisoning, or post-detection mitigation strategies. The benchmark is restricted to English textual inputs and evaluated under a frozen train, validation, and protected-test protocol.

## 3.2 Task Definition

We formulate prompt-injection detection in RAG-style contexts as a binary classification task. Each example consists of a rendered textual input $x _ { i }$ containing a host context and an embedded content segment. The objective is to determine whether the embedded content is benign or contains a malicious instruction intended to alter the behavior of a downstream assistant. Formally, each instance is assigned a label $y _ { i } \in \{ 0 , 1 \}$ , where 0 denotes benign content and 1 denotes prompt injection. A detector learns a mapping $f ( x _ { i } )  y _ { i }$ . Performance is evaluated not only by overall classification effectiveness but also by the ability to balance detection of malicious content against false positives on benign instruction-like text. This formulation captures a key challenge in RAG security: benign retrieved content may naturally contain instructions, examples, code snippets, or task descriptions, requiring detectors to distinguish legitimate instructional content from adversarial instruction-override attempts.

![](images/8089279e6f9e3ab98e565765df901eecfea80bdf8fb8c8405c7b3e2e97139755.jpg)  
Fig. 1. RAG-PIBench construction and evaluation pipeline. Candidate host passages and payload parents are registered, filtered, and reviewed before being rendered into RAG-style contextual examples. The benchmark is frozen into train, validation, and protected-test splits with balanced benign and malicious labels.

## 3.3 Overview of the RAG-PIBench Construction Pipeline

As shown in Fig. 1, RAG-PIBench follows a staged, leakage-aware construction and evaluation workflow. The pipeline comprises four phases: (i) source registration, schema validation, and provenance tracing; (ii) candidate review and quality control; (iii) RAGstyle contextual rendering and leakage auditing; and (iv) frozen split materialization, model development, and protected-test evaluation. This design maintains strict separation between dataset construction, model selection, diagnostics, and final evaluation.

Candidate Sources and Semantic Labels. RAG-PIBench was constructed from host documents and embedded-content candidates that were rendered into RAG-style inputs. Each example was assigned one of two labels: benign, indicating ordinary instructional content that does not attempt to alter system behavior, or malicious\_ prompt\_injection, indicating content that attempts to redirect, bypass, replace, ignore, or override the intended behavior of the assistant. A central design goal was to ensure that both classes remain task-like and instruction-bearing. The benchmark therefore focuses on distinguishing benign instructions from adversarial instruction-override attempts rather than simply detecting the presence of instructions.

![](images/942dd2e23c336a9f557c32c4c69ea3baa3688ea2655db5a244829d9499fe7836.jpg)  
(a) Selected host document sources

![](images/afd10429fa6007d42ff62bdd0c8d8915f8655d9d157fa830e75f98fc778e6e6c.jpg)  
(b) Primary payload-parent source composition  
Fig. 2. Source composition of RAG-PIBench. (a) Host-document sources used for contextual passages. (b) Payload-parent sources used to generate benign and malicious embedded content. Percentages are computed over 3,500 host documents in (a) and separately over the benign and malicious payload-parent pools (192 sources each) in (b).

Schema Validation and Provenance Tracing. Each candidate example was represented using a structured schema containing rendered text, semantic labels, identifiers, split metadata, and provenance information. Validation ensured field consistency, label correctness, and rendering compatibility. Provenance records tracked the origin and transformation of each example throughout construction, supporting auditing and reducing the risk of duplicated, unsupported, or inconsistently labeled entries.

Hybrid Human-Centered Review and Quality Control. Candidate examples underwent hybrid human-centered quality control combining manual review with automated checks for schema consistency, duplication, provenance integrity, label agreement, and potential shortcut artifacts. Automated analyses were used to support review rather than assign final labels. Examples that were ambiguous, unsupported, inconsistent with the task definition, or likely to introduce artifacts were excluded or subjected to additional review. All inclusion decisions were finalized before model evaluation, and protected-test data remained inaccessible during model development.

RAG-Style Contextual Rendering. Retained examples were rendered into RAG-style inputs consisting of a host context and an embedded content segment. The embedded segment contains either a benign instruction or a malicious prompt-injection attempt. This design approximates realistic retrieval-augmented inputs and prevents the task from reducing to isolated prompt-fragment classification, requiring detectors to reason about instructions within their surrounding context.

Table 2. Data split statistics.
<table><tr><td>Split</td><td></td><td>Total Benign</td><td>Mal.</td></tr><tr><td>Train</td><td>2936</td><td>1468</td><td>1468</td></tr><tr><td>Valid.</td><td>962</td><td>481</td><td>481</td></tr><tr><td>Prot. Test</td><td>978</td><td>489</td><td>489</td></tr><tr><td>Total</td><td>4876</td><td>2438</td><td>2438</td></tr></table>

Leakage, Shortcut, and Residual-Cue Auditing. To reduce shortcut learning, the benchmark underwent leakage and residual-cue auditing prior to finalization. Audits examined duplicate content, split contamination, repeated templates, source-specific markers, formatting artifacts, and other cues that could enable non-semantic classification. Identified issues were corrected, removed, or explicitly documented. Diagnostic analyses remained separate from final model selection and protected-test evaluation.

Frozen Dataset Splits. The final benchmark contains 4,876 examples divided into fixed train, validation, and protected-test splits, each balanced across benign and malicious classes. Table 2 summarizes the split statistics.

Training data were used for model fitting, validation data for hyperparameter and checkpoint selection, and the protected-test split exclusively for final evaluation. Protectedtest content remained inaccessible during model development, and final results were produced through a single protected-test evaluation pass.

## 3.4 Evaluation Settings

Model Families We evaluated two model families: baseline detectors and transformerbased classifiers. Baseline detectors provide strong lightweight reference points, while transformer models assess whether pretrained contextual representations improve promptinjection detection under the same frozen evaluation protocol.

Baseline Detectors We evaluated five baseline detectors spanning rule-based, semanticsimilarity, and sparse-feature approaches. ① Keyword: a deterministic detector based on lexical patterns associated with prompt injection. ② MiniLM Ref.: a semanticsimilarity detector that scores inputs using maximum similarity to a malicious-reference bank. ③ TF-IDF LR: logistic regression using word- and character-level TF-IDF features. ④ TF-IDF SVM: a linear support-vector machine using word- and character-level TF-IDF features. ⑤ TF-IDF RF: a random forest trained on TF-IDF representations.

These baselines provide a progression from simple lexical rules to competitive sparse-feature classifiers. Their inclusion helps determine whether transformer models provide meaningful gains beyond lightweight alternatives.

Transformer Classifiers We evaluated two fine-tuned transformer encoder classifiers. ① DistilBERT-base-uncased: a compact pretrained encoder serving as an efficient contextual baseline. ② DeBERTa-v3-base: a higher-capacity encoder included to assess whether increased model complexity improves detection performance.

Both models were trained on the training split, selected using validation performance, and evaluated once on the protected-test split. DistilBERT used a maximum sequence length of 256, while DeBERTa-v3 used a maximum sequence length of 192 under a conservative numerical-stability configuration. These settings were fixed prior to protected-test evaluation and were not modified afterward.

Training, Validation, and Protected-Test Evaluation The experimental workflow followed a strict train/validation/protected-test protocol. Model development was confined to the training and validation splits, while the protected-test split was reserved exclusively for final evaluation. Baseline detectors were trained and tuned under this protocol, and transformer classifiers were evaluated on protected-test examples only after training and validation were complete.

To preserve evaluation integrity, train/validation transformer results were recorded separately from protected-test results, and the protected-test split was opened only for final inference. No checkpoint reselection, threshold retuning, additional training, or diagnostic feedback was permitted after protected-test exposure. The final modelcomparison table was frozen only after all model families had been evaluated on the same 978-example protected-test split.

Separate Diagnostic Suites In addition to the primary benchmark split, separate diagnostic suites were used to analyze robustness, false-positive behavior, and distributional sensitivity. These diagnostics included broad contextual examples, hard benign utility cases, and external transfer/source-audit settings.

Diagnostic suites were not used for model selection, threshold tuning, or post-hoc correction. They were maintained separately from the primary protected-test evaluation to preserve the independence of final reported results.

Evaluation Metrics We report accuracy, precision, recall, $F _ { 1 }$ , PR-AUC, ROC-AUC, and confusion-matrix counts. The malicious prompt-injection class is treated as the positive class. Let $T P , F P , T N$ , and FN denote true positives, false positives, true negatives, and false negatives, respectively. The operating-point metrics are defined as: $\begin{array} { r } { \mathrm { A c c u r a c y } = \frac { T P + T N } { T P + T N + F P + F N } } \end{array}$ $\begin{array} { r } { \dot { \mathrm { P r e c i s i o n } } = \frac { \dot { T } P } { T P + F P } } \end{array}$ $\begin{array} { r } { \mathrm { R e c a l l } = \frac { T P } { T P + F N } } \end{array}$ , and $F _ { 1 } =$ 2·Precision·Recall<sub>Precision+Recall</sub> . PR-AUC captures the trade-off between detecting malicious inputs and avoiding false alarms, while ROC-AUC provides a complementary threshold-independen ranking measure. We use $F _ { 1 }$ as the primary ranking metric because the benchmark is class-balanced and prompt-injection detection requires balancing missed attacks against false alarms. PR-AUC and ROC-AUC are reported to characterize ranking quality beyond the selected decision threshold.

Reproducibility and Artifact Registration Each stage of the RAG-PIBench pipeline produced audit artifacts, manifests, and checksums documenting dataset lineage, split statistics, evaluation boundaries, and model outputs. The final artifact package includes the frozen protected-test comparison table, paper-ready tables and figures, split statistics, a dataset card, interpretation notes, and a reproducibility registry.

The final model-comparison table was frozen only after: (i) benchmark splits were fixed and balanced; (ii) baseline protected-test results were registered; (iii) transformer train/validation results were separated from protected-test results; (iv) transformer protectedtest evaluation was completed; (v) all model families were evaluated on the same protectedtest split; and (vi) no threshold retuning, diagnostic reuse, or additional training occurred during final evaluation. This workflow provides a reproducible audit trail and preserves a clear separation between model development, validation-based selection, diagnostic analysis, and final reporting.

Methodological Safeguards Several safeguards were incorporated for evaluation validity. First, examples were retained according to benchmark policy and quality-control criteria rather than model separability. Second, the protected-test split was excluded from model selection. Third, validation-only transformer results were prevented from entering the final comparison. Fourth, diagnostic suites were separated from primary evaluation. Finally, all models were evaluated on the same protected-test split.

Table 3. Primary protected-test performance across all evaluated detectors.
<table><tr><td>Class</td><td>Model</td><td>Prec.</td><td>Rec.</td><td> $\pmb { F _ { 1 } }$ </td><td>PR-AUC</td><td>ROC-AUC</td><td>Acc.</td></tr><tr><td>FT-Enc.</td><td>DistilBERT</td><td>0.924</td><td>0.869</td><td>0.896</td><td>0.968</td><td>0.967</td><td>0.899</td></tr><tr><td>Sparse</td><td>TF-IDF SVM</td><td>0.853</td><td>0.890</td><td>0.871</td><td>0.950</td><td>0.949</td><td>0.868</td></tr><tr><td>Sparse</td><td>TF-IDF LR</td><td>0.822</td><td>0.916</td><td>0.867</td><td>0.950</td><td>0.949</td><td>0.859</td></tr><tr><td>Sparse</td><td>TF-IDF RF</td><td>0.822</td><td>0.879</td><td>0.850</td><td>0.933</td><td>0.931</td><td>0.845</td></tr><tr><td>SemSim</td><td>MiniLM Ref.</td><td>0.508</td><td>0.969</td><td>0.667</td><td>0.576</td><td>0.571</td><td>0.515</td></tr><tr><td>FT-Enc.</td><td>DeBERTa-v3</td><td>0.500</td><td>1.000</td><td>0.667</td><td>0.488</td><td>0.487</td><td>0.500</td></tr><tr><td>Rule</td><td>Keyword</td><td>0.981</td><td>0.104</td><td>0.189</td><td>0.550</td><td>0.551</td><td>0.551</td></tr></table>

Note: Results are reported on the 978-example balanced protected-test split. Prec., Rec., Acc., LR, SVM, RF, FT-Enc., Sem Sim, and Ref. denote precision, recall, accuracy, logistic regression, support vector machine, random forest, fine-tuned encoder, semantic-similarity baseline, and reference baseline, respectively. Precision, recall, and $F _ { 1 }$ use the malicious prompt injection class as the positive class. Bold values indicate column-best results.

Together, these safeguards reduce the risk of leakage, shortcut learning, post-hoc tuning, and unfair comparison, ensuring that reported differences reflect performance under a common protected-test protocol.

## 4 Results and Discussion

This section reports the primary protected-test results for RAG-PIBench. All models are evaluated on the same balanced protected-test split of 978 examples (489 benign and 489 malicious), including five baseline detectors and two fine-tuned transformer classifiers. Consistent with the methodology, the protected-test split was reserved exclusively for final evaluation, with no post-test training, checkpoint reselection, threshold tuning, or diagnostic feedback. Diagnostic suites are reported separately and do not influence model ranking. Consequently, all results reflect a common protected-test comparison across detector families.

## 4.1 Overall Primary Protected-Test Performance

Table 3 reports the final primary protected-test results on the balanced 978-example held-out split. Models are ordered by $F _ { 1 }$ , which we use as the primary operating-point metric because prompt-injection detection requires a practical trade-off between missed malicious examples and false alarms on benign examples. Precision, recall, and $F _ { 1 }$ are computed with the malicious prompt-injection class as the positive class.

DistilBERT achieves the strongest overall protected-test performance, with $F _ { 1 } =$ 0.896, precision = 0.924, recall = 0.869, PR-AUC = 0.968, ROC-AUC = 0.967, and accuracy = 0.899. Its confusion matrix contains 454 true negatives, 35 false positives, 64 false negatives, and 425 true positives. This indicates the best observed balance between detecting malicious prompt-injection examples and preserving benign utility. The high PR-AUC and ROC-AUC also suggest that the result is not only tied to a single decision threshold, but reflects strong ranking behavior across operating points.

The sparse lexical baselines remain highly competitive. Among them, TF-IDF SVM achieves the strongest sparse-feature baseline $F _ { 1 }$ , with $F _ { 1 } = 0 . 8 7 1$ , precision = 0.853, recall $= \ 0 . 8 9 0 ,$ PR- $\mathrm { A U C } = 0 . 9 5 0 .$ , and ROC- $\mathrm { A U C } = 0 . 9 4 9$ . TF-IDF LR is closely matched, with $F _ { 1 } = 0 . 8 6 7$ and PR- $\mathrm { \ A U C } = 0 . 9 5 0$ . These results suggest that lexical and character-level cues capture substantial discriminative signal in RAG-PIBench, while transformer-based contextual modeling provides the strongest overall performance.

![](images/fc0e5fc9fecdbb5c4dc871b85490f15843c63160e212f5ab47ed399aee0e6ab2.jpg)  
(a) $F _ { 1 }$ score

![](images/3c3ba13419e3a097e30bf1c62689882bcefff4ea5361711c748e522a1c0b0364.jpg)  
(b) PR-AUC  
Fig. 3. Primary protected-test performance summary. Panel (a) reports the fixed operating-point $F _ { 1 }$ score, where DistilBERT achieves the strongest overall result and TF-IDF SVM provides the strongest sparse lexical baseline. Panel (b) reports PR-AUC, showing that strong TF-IDF baselines remain competitive for ranking malicious prompt-injection examples.

## 4.2 Transformer and Baseline Detector Comparison

The protected-test results show that DistilBERT achieves the strongest overall performance, but its advantage over the best sparse baseline is modest. DistilBERT improves $F _ { 1 }$ from 0.871 (TF-IDF SVM) to 0.896, indicating that contextual representations provide additional discriminative signal while confirming that sparse lexical features capture much of the structure required for prompt-injection detection in RAG-PIBench.

This finding has practical implications. While transformer-based detectors may be preferred when maximizing predictive performance, TF-IDF SVM and TF-IDF LR remain competitive while being less computationally expensive, easier to train, and more interpretable. Thus, prompt-injection detection studies should compare new methods against strong sparse baselines rather than only against simple heuristic detectors.

The comparison between TF-IDF SVM and TF-IDF LR also highlights a precision– recall trade-off. TF-IDF SVM achieves a slightly higher $F _ { 1 }$ score and fewer false positives (75 vs. 97), whereas TF-IDF LR achieves higher recall and fewer false negatives (41 vs. 54). These results suggest that even closely matched sparse models may support different deployment objectives, with TF-IDF SVM favoring precision and TF-IDF LR favoring recall.

## 4.3 Error Trade-offs and Baseline Behavior

The keyword baseline serves as a lower-bound reference. It achieves very high precision (0.981) but very low recall (0.104), indicating that explicit lexical rules identify only a small subset of obvious prompt-injection attempts while missing most malicious examples. Its confusion matrix contains 488 true negatives, 1 false positive, 438 false negatives, and 51 true positives, making it unsuitable as a standalone defense despite its low false-positive rate.

The MiniLM malicious-reference baseline exhibits the opposite behavior. It achieves high recall (0.969) but low precision (0.508), producing 30 true negatives, 459 false positives, 15 false negatives, and 474 true positives. This suggests that semantic-reference methods can recover most malicious examples but may substantially over-flag benign content, creating a significant false-positive burden in practical deployments.

The TF-IDF models provide a more balanced operating point. TF-IDF SVM, TF-IDF logistic regression, and TF-IDF random forest all achieve F<sub>1</sub> scores above 0.84, indicating that prompt-injection detection benefits from richer lexical and structural representations than simple keyword matching. The strong performance of these sparsefeature models suggests that much of the discriminative signal in RAG-PIBench can be captured through word- and character-level patterns while maintaining a substantially better precision–recall balance than either the keyword or semantic-reference baselines.

## 4.4 Transformer-Specific Findings

The transformer results reveal a sharp contrast between DistilBERT and DeBERTav3 under the frozen evaluation protocol. DistilBERT achieves the strongest overall protected-test performance, whereas DeBERTa-v3 exhibits a degenerate all-positive prediction pattern, classifying every example as malicious. As a result, DeBERTa-v3 achieves perfect recall (1.000) but only 0.500 precision, with 489 false positives and no true negatives. Its ROC-AUC and PR-AUC also fall below those of the strongest baseline models, indicating limited ranking ability under the evaluated configuration.

This outcome should not be interpreted as evidence that DeBERTa-v3 is inherently unsuitable for prompt-injection detection. Rather, it reflects the specific conservative training and numerical-stability configuration used in this study. While the final model completed training and evaluation successfully, it failed to learn a useful decision boundary on the protected-test split. The result highlights that increased model capacity does not necessarily translate into improved detection performance under a frozen, leakage-aware evaluation protocol. More broadly, the contrast between Distil-BERT and DeBERTa-v3 underscores the importance of transparent reporting. Reporting only successful transformer results would overstate the reliability of transformer-based detection. Including unsuccessful outcomes provides a more complete characterization of model behavior and strengthens the credibility of the evaluation.

## 4.5 Ranking Metrics and Operating-Point Metrics

The results illustrate the value of reporting both operating-point and ranking metrics. Accuracy, Precision, Recall, and $F _ { 1 }$ characterize behavior at the selected decision threshold, whereas PR-AUC and ROC-AUC measure ranking quality across thresholds. DistilBERT performs strongly under both views, achieving the highest $F _ { 1 }$ , PR-AUC, and ROC-AUC, indicating that its advantage is not tied to a particular operating point.

The sparse TF-IDF models also demonstrate strong ranking performance. TF-IDF logistic regression and TF-IDF SVM both achieve PR-AUC values near 0.95 and ROC-AUC values near 0.95, showing that lightweight sparse-feature models can effectively rank malicious examples despite trailing DistilBERT on overall protected-test performance. This finding is particularly relevant for deployments where computational efficiency or model simplicity is important. In contrast, DeBERTa-v3 achieves perfect recall at the selected threshold but poor ranking performance. This result highlights that recall alone can be misleading: a detector that labels every example as malicious will never miss an attack, but it will also reject all benign content. Effective prompt-injection detection therefore requires both strong ranking ability and a practical operating point that balances attack detection against false alarms.

## 4.6 Implications for RAG Prompt-Injection Detection

The results have several implications for RAG prompt-injection detection. First, contextual transformer representations can improve performance, as evidenced by Distil-BERT achieving the strongest overall protected-test result. This suggests that pretrained contextual encoders provide additional signal for distinguishing malicious instructionoverride attempts from benign embedded instructions.

Second, strong baseline detectors remain essential. TF-IDF SVM and TF-IDF logistic regression perform competitively with DistilBERT and substantially outperform the simpler keyword and semantic-reference baselines. Consequently, future studies should compare proposed methods against strong sparse-feature models rather than only against heuristic detectors.

Third, false-positive control is as important as attack coverage. Both the MiniLM malicious-reference baseline and DeBERTa-v3 achieve very high recall, but only at the cost of excessive false positives. In practical RAG deployments, over-blocking benign content can reduce system utility, disrupt legitimate workflows, and increase humanreview burden. Effective detectors must therefore balance precision and recall rather than optimizing recall alone.

Finally, benchmark construction and evaluation methodology play a critical role in the credibility of reported results. By employing a frozen train/validation/protectedtest protocol, separating diagnostic analyses from final model ranking, and preventing validation-only results from entering the primary comparison, RAG-PIBench reduces opportunities for leakage and evaluation bias, yielding a more reliable assessment of prompt-injection detection performance.

## 4.7 Discussion of Benchmark Characteristics

RAG-PIBench is designed to evaluate prompt-injection detection in contextualized inputs rather than isolated attack strings. The strong performance of the TF-IDF models indicates that the task contains learnable lexical and structural signals, while the large gap between keyword heuristics and learned sparse models shows that these signals extend beyond simple rule-based matching.

The modest improvement of DistilBERT over the strongest TF-IDF baselines suggests that contextual semantics provide additional discriminative information, but not enough to render sparse-feature models ineffective. This is a desirable property of the benchmark: it is neither so simple that keyword rules solve the task nor so dependent on large neural models that classical baselines become uninformative.

The DeBERTa-v3 results further indicate that model capacity alone does not determine performance. Training stability, optimization behavior, sequence-length constraints, and checkpoint selection can substantially affect protected-test outcomes. This observation reinforces the importance of reproducible evaluation protocols and transparent reporting of both successful and unsuccessful model behavior.

## 4.8 Limitations

Several limitations should be acknowledged. First, RAG-PIBench evaluates binary promptinjection detection rather than downstream mitigation. Practical RAG systems may combine detection with policies for isolating, sanitizing, ignoring, quoting, or escalating retrieved content, which are outside the scope of this study.

Second, although the benchmark incorporates style matching, provenance tracing, duplicate checks, and leakage auditing, no benchmark can eliminate all distributional artifacts. Accordingly, the reported results should be interpreted as performance on RAG-PIBench rather than as evidence of universal robustness to prompt-injection attacks.

Third, the transformer evaluation is limited to two encoder-based models under a frozen training and selection protocol. Broader coverage, including larger encoders, decoder-only models, instruction-tuned classifiers, and retrieval-aware architectures, may reveal additional performance characteristics. The DeBERTa-v3 findings should therefore be interpreted within the specific training and numerical-stability configuration evaluated in this work.

Fourth, the primary protected-test split is class-balanced. While this design supports fair model comparison and straightforward interpretation of $F _ { 1 }$ , real deployments may exhibit substantially different class priors. In such settings, calibration and threshold selection may require additional validation procedures that remain independent of protected-test data.

Finally, the benchmark is limited to English textual inputs. Future work should examine multilingual prompt-injection attacks, cross-domain transfer, adversarially generated content, and application-specific RAG environments.

## 4.9 Summary of Findings

Overall, the final protected-test results show that RAG-PIBench provides an informative evaluation setting for prompt-injection detection. DistilBERT achieves the strongest overall performance, with the best $F _ { 1 }$ , PR-AUC, ROC-AUC, and accuracy. However, strong TF-IDF baselines remain highly competitive, demonstrating that lightweight models can capture substantial prompt-injection signals. Keyword rules are too brittle because they miss most malicious examples, while semantic-reference detection and DeBERTa-v3 show that high recall can be misleading when false positives are uncontrolled. These findings support three main conclusions. First, prompt-injection detection should be evaluated against strong baseline detectors, not only weak heuristics. Second, protected-test evaluation must be separated from validation and diagnostic analysis to avoid inflated claims. Third, transparent reporting of both successful and failed model behavior is necessary for reliable progress in RAG security evaluation.

## 5 Concluding Remarks

This paper introduced RAG-PIBench, a leakage-aware benchmark for binary promptinjection detection in RAG-style contextual inputs. The benchmark was designed to evaluate whether detectors can distinguish benign embedded instructions from malicious instruction-override attempts under a frozen train, validation, and protected-test protocol. To support reproducible evaluation, the construction process incorporated provenance tracing, schema validation, hybrid human-centered quality control, contextual rendering, shortcut auditing, and strict split isolation. The final evaluation compared heuristic, semantic-reference, sparse-feature, and transformer-based detectors on a common 978-example protected-test split.

The results demonstrate that contextual transformer representations can improve prompt-injection detection performance. DistilBERT achieved the strongest overall protectedtest performance, obtaining an $F _ { 1 }$ score of 0.896, PR-AUC of 0.968, and ROC-AUC of 0.967. At the same time, strong sparse-feature baselines remained highly competitive, with TF-IDF SVM achieving an $F _ { 1 }$ score of 0.871 and TF-IDF logistic regression achieving a PR-AUC of 0.950. These findings indicate that sparse lexical and characterlevel representations capture substantial prompt-injection signal and should remain part of future benchmark comparisons.

The evaluation also revealed important failure modes. Keyword heuristics achieved high precision but very low recall, demonstrating the limitations of explicit triggerbased rules. The semantic-reference baseline achieved high recall but generated many false positives, highlighting the risk of over-flagging benign instructional content. Under the evaluated configuration, DeBERTa-v3 collapsed to an all-positive prediction pattern, illustrating that larger model capacity does not necessarily translate into better prompt-injection detection performance.

Overall, the results suggest that effective prompt-injection detection requires both strong benchmark design and rigorous evaluation methodology. They further demonstrate the value of comparing transformer models against competitive lightweight baselines and of maintaining strict separation between model development, diagnostic analysis, and final testing. Future work should expand benchmark coverage through broader model families, multilingual and cross-domain evaluation, adversarially generated attacks, multimodal prompt-injection scenarios, and deployment-oriented studies that jointly evaluate detection and mitigation strategies.

## References

1. Choi, S., Mohaisen, D.: Attributing ChatGPT-generated source codes. IEEE Transactions on Dependable and Secure Computing 22(4), 3602–3615 (2025). https://doi.org/10 .1109/TDSC.2025.3535218

2. De Stefano, G., Schönherr, L., Pellegrino, G.: Rag and roll: An end-to-end evaluation of indirect prompt manipulations in LLM-based application frameworks (2024)

3. Gallifant, J., Chen, S., Sasse, K., Aerts, H., Hartvigsen, T., Bitterman, D.: Sparse autoencoder features for classifications and transferability. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp. 29927–29951 (2025)

4. Geng, T., Xu, Z., Qu, Y., Wong, W.E.: Prompt injection attacks on large language models: A survey of attack methods, root causes, and defense strategies. Computers, Materials & Continua 87(1), 4 (2026). https://doi.org/10.32604/cmc.2025.074081

5. Guan, S., Kwok, H.C., Law, N.F., Stiglic, G., Qin, H., Hui, V.: Privacy challenges and solutions in retrieval-augmented generation-enhanced LLMs for healthcare chatbots: A review of applications, risks, and future directions. arXiv preprint arXiv:2511.11347 (2025)

6. Hadiprakoso, R.B., Wilujengning, W., Amiruddin, A.: Adaptive multi-layer framework for detecting and mitigating prompt injection attacks in large language models. J. Inf. Syst. Eng. Bus. Intell 11, 473–487 (2025)

7. Helbling, A., Phute, M., Hull, M., Chau, D.H.: Llm self defense: By self examination, llms know they are being tricked. arXiv e-prints pp. arXiv–2308 (2023)

8. Jacob, D., Alzahrani, H., Hu, Z., Alomair, B., Wagner, D.: Promptshield: Deployable detection for prompt injection attacks. In: Proceedings of the 15th ACM Conference on Data and Application Security and Privacy. pp. 341–352. CODASPY ’25, Association for Computing Machinery (2025). https://doi.org/10.1145/3714393.3726501

9. Jaffal, N.O., Alkhanafseh, M., Mohaisen, D.: Large language models in cybersecurity: A survey of applications, vulnerabilities, and defense techniques. AI 6(9), 216 (2025). http s://doi.org/10.3390/ai6090216

10. Kharma, M.F., Choi, S., Alkhanafseh, M., Mohaisen, D.: Security and quality in LLM generated code: A multi-language, multi-model analysis. IEEE Transactions on Dependable and Secure Computing (2026)

11. Kokkula, S., Somanathan, R., Nandavardhan, R., Aashishkumar, Divya, G.: Palisade – prompt injection detection framework. arXiv preprint arXiv:2410.21146 (2024). https: //doi.org/10.48550/arXiv.2410.21146

12. Latibari, B.S., Nazari, N., Chowdhury, M.A., Gubbi, K.I., Fang, C., Ghimire, S., Hosseini, E., Sayadi, H., Homayoun, H., Salehi, S., et al.: Transformers: A security perspective. IEEE Access 12, 181071–181105 (2024)

13. Lin, J., Mohaisen, D.: From large to mammoth: A comparative evaluation of large language models in vulnerability detection. In: Proceedings of the 32nd Annual Network and Dis tributed System Security Symposium (NDSS). Internet Society, San Diego, CA, USA (2025)

14. Liu, Y., Xu, R., Wang, X., Jia, Y., Gong, N.Z.: WAInjectBench: Benchmarking prompt injection detections for web agents. arXiv preprint arXiv:2510.01354 (2025). https: //doi.org/10.48550/arXiv.2510.01354

15. Ögdü, Ç.U., Arslano˘ glu, K., Karaköse, M.: An adaptive multi-agent llm-based clinical de-˘ cision support system integrating biomedical rag and web intelligence. IEEE Access 13, 167390–167404 (2025). https://doi.org/10.1109/ACCESS.2025.3613340

16. Oro, E., Granata, F.M., Lanza, A., Bachir, A., De Grandis, L., Ruffolo, M.: Evaluating retrieval-augmented generation for question answering with large language models. In: Proceedings of the 4th National Conference on Artificial Intelligence (Ital-IA 2024). pp. 12–17. Naples, Italy (2024)

17. Rakin, S., Shibly, M.A., Hossain, Z.M., Khan, Z., Akbar, M.M.: Leveraging the domain adaptation of retrieval augmented generation models for question answering and reducing hallucination. arXiv preprint arXiv:2410.17783 (2024)

18. Shrivastav, V.A.: An Effective Approach to Protecting Large Language Model Applications From Prompt Injection Attacks in E-Commerce. Master’s thesis, Arizona State University (2025)

19. Si, X., Zhu, M., Qin, S., Yu, L., Zhang, L., Liu, S., Li, X., Duan, R., Liu, Y., Jia, X.: SeCon RAG: A two-stage semantic filtering and conflict-free framework for trustworthy rag. In: Advances in Neural Information Processing Systems (2025). https://doi.org/10.4 8550/arXiv.2510.09710, neurIPS 2025 poster; arXiv:2510.09710

20. Wang, M., Zhang, Y., Gu, G.: Promptsleuth: Detecting prompt injection via semantic intent invariance (2025)

21. Watson Benjamin, M.I.S., K, A.J.V., Yadav, R., Reddy Thippareddy, V.S., Raju, B.: Contextfusion: An intelligent retrieval-augmented conversational ai framework with multi-model support. In: 2025 3rd International Conference on Self Sustainable Artificial Intelligence Systems (ICSSAS). pp. 965–971 (2025). https://doi.org/10.1109/ICSSAS66 150.2025.11081255

22. Yi, J., Xie, Y., Zhu, B., Kiciman, E., Sun, G., Xie, X., Wu, F.: Benchmarking and defending against indirect prompt injection attacks on large language models. In: Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining. pp. 1809–1820. KDD ’25, Association for Computing Machinery (2025). https://doi.org/10.114 5/3690624.3709179

23. Zhan, Q., Liang, Z., Ying, Z., Kang, D.: Injecagent: Benchmarking indirect prompt injections in tool-integrated large language model agents. In: Findings of the Association for Computational Linguistics: ACL 2024. pp. 10471–10506 (2024)

24. Zhou, H., Lee, K.H., Zhan, Z., Chen, Y., Li, Z., Wang, Z., Haddadi, H., Yilmaz, E.: Trustrag: Enhancing robustness and trustworthiness in retrieval-augmented generation. arXiv preprint arXiv:2501.00879 (2025)

## A Supplementary Reproducibility and Evaluation Details

This appendix provides supplementary details that support reproducibility and clarify the evaluation boundaries of RAG-PIBench. It does not introduce additional modelselection criteria and does not alter the primary protected-test results reported in the main paper.

## A.1 Dataset Construction and Label Policy

RAG-PIBench is formulated as a binary prompt-injection detection benchmark for RAGstyle contextual inputs. Each example contains a rendered host context and an embedded content segment. The negative class, benign, denotes embedded content that may contain a legitimate task, request, instruction, or text segment but does not attempt to override the surrounding task or system behavior. The positive class, malicious\_ prompt\_injection, denotes embedded content that attempts to override, ignore, redirect, bypass, replace, or manipulate the intended instruction-following behavior of the downstream assistant.

A key design principle is that both classes may contain instruction-like language. Therefore, the benchmark is not a simple test of whether imperative wording appears in the input. Instead, the task is to distinguish benign embedded instructions from malicious instruction-override behavior. Candidate rows that were ambiguous, unsupported, duplicated, inconsistent with the task definition, or likely to introduce shortcut artifacts were excluded, quarantined, or sent to additional confirmation before final split materialization.

The final benchmark contains 4,876 examples across frozen train, validation, and protected-test splits. The training split contains 2,936 examples, the validation split contains 962 examples, and the protected-test split contains 978 examples. Each split is balanced between benign and malicious prompt-injection examples.

## A.2 Reproducibility Phase Registry

Table 4 summarizes the registered stages used to separate dataset construction, model selection, protected-test access, and final artifact generation.

Table 4. Reproducibility controls for the final RAG-PIBench results.
<table><tr><td>Stage</td><td>Reproducibility control</td><td>Protocol ID</td></tr><tr><td>Data freeze</td><td>Fixed the train, validation, and protected-test rows before model development.</td><td>OR</td></tr><tr><td>Sparse/rule baselines</td><td>Registered the final protected-test results for sparse and rule-based baselines.</td><td>0AA.2G/0AA.2H</td></tr><tr><td>Transformer training</td><td>Restricted training to the train split and checkpoint selection to validation only.</td><td>0AA.2L.1-R4</td></tr><tr><td>Validation gate</td><td>Prevented validation-only transformer results from entering the final test table.</td><td>0AA.2M-LITE</td></tr><tr><td>Protected-test transfer</td><td>Opened the protected-test split only for final transformer inference.</td><td>0AA.2N.0</td></tr><tr><td>Transformer evaluation</td><td>Produced the final transformer protected-test metrics.</td><td>0AA.2N.1</td></tr><tr><td>Artifact freeze</td><td>Froze the final tables, figures, dataset card, and paper artifacts.</td><td>0AA.2O-R1</td></tr></table>

Note: The registry separates data freezing, model selection, protected-test access, and final artifact generation.

This staged workflow ensures that validation results, diagnostic analyses, and protectedtest results remain separated. The protected-test split was not used for training, threshold retuning, or checkpoint reselection.

## A.3 Leakage and Shortcut Audit Summary

Table 5 summarizes the main leakage and shortcut-control checks used during benchmark construction and evaluation. These checks clarify the leakage-aware protocol. They do not imply that all possible distributional artifacts are eliminated, but they document the controls used to reduce split contamination, duplicate reuse, and post-hoc evaluation leakage.

## A.4 Confusion Matrices on the Primary Protected-Test Split

The main paper reports the full metric table. To make the operating-point behavior transparent, Table 6 reports the corresponding confusion matrices on the same 978- example primary protected-test split. The positive class is malicious\_prompt\_i njection.

These counts show why aggregate metrics alone are insufficient. For example, the keyword heuristic has very few false positives but misses most malicious examples, while the MiniLM reference baseline and DeBERTa-v3 substantially over-flag benign examples.

## A.5 Separate Diagnostic Suites

Diagnostic suites were used only for behavioral analysis. They were not used for final model ranking, threshold tuning, checkpoint selection, or post-hoc correction of primary protected-test results. Table 7 summarizes their role.

Table 5. Leakage and shortcut-control audits used in RAG-PIBench.
<table><tr><td>Audit</td><td>Check</td><td>Registered outcome</td></tr><tr><td>Rendered-text duplicates</td><td>Exact and hash-level duplicates were checked across train, validation, and protected-test splits.</td><td>No cross-split rendered-text duplicate leakage was registered.</td></tr><tr><td>Host-context isolation</td><td>Host documents and segmented passages were assigned before final rendering to reduce cross-split overlap.</td><td>Host overlap was controlled through split isolation and audit checks.</td></tr><tr><td>Payload-parent tracking</td><td>Payload-parent sources and semantic families were tracked during candidate selection and rendering.</td><td>Payload-parent usage was registered and controlled during benchmark construction.</td></tr><tr><td>Schema and provenance val- idation</td><td>Required fields, labels, split metadata, row identifiers, and provenance fields were validated.</td><td>Invalid, unsupported, or inconsistent rows were excluded, quarantined, or sent for further confirmation.</td></tr><tr><td>Shortcut and residual-cue re- view</td><td>Formatting artifacts, source-specific cues, repeated templates, and residual shortcut indicators were reviewed.</td><td>Problematic rows or patterns were corrected, excluded, quarantined, or explicitly registered.</td></tr><tr><td>Protected-test access bound- ary</td><td>Exposure of protected-test text during train/validation model development was checked.</td><td>Protected-test text was withheld during train/validation phases and opened only for final evaluation.</td></tr><tr><td>Diagnostic separation</td><td>Use of diagnostic suites for model selection, threshold tuning, or post-hoc correction was checked.</td><td>Diagnostic suites were kept separate from final model ranking and protected-test decisions.</td></tr></table>

Note: This table summarizes registered construction and evaluation safeguards. These controls support leakage-aware benchmark construction, but do not imply that any finite benchmark is free of all possible residual distributional cues.

Table 6. Protected-test confusion matrices for all evaluated detectors.
<table><tr><td>Model</td><td>TN</td><td>FP</td><td>FN</td><td>TP</td></tr><tr><td>DistilBERT</td><td>454</td><td>35</td><td>64</td><td>425</td></tr><tr><td>TF-IDF SVM</td><td>414</td><td>75</td><td>54</td><td>435</td></tr><tr><td>TF-IDF LR</td><td>392</td><td>97</td><td>41</td><td>448</td></tr><tr><td>TF-IDF RF</td><td>396</td><td>93</td><td>59</td><td>430</td></tr><tr><td>MiniLM Ref.</td><td>30</td><td>459</td><td>15</td><td>474</td></tr><tr><td>DeBERTa-v3</td><td>0</td><td>489</td><td>0</td><td>489</td></tr><tr><td>Keyword</td><td>488</td><td>1</td><td>438</td><td>51</td></tr></table>

Note: TN, FP, FN, and TP denote true negatives, false positives, false negatives, and true positives. The malicious prompt-injection class is treated as the positive class. Model abbreviations follow Table 3.

Table 7. Diagnostic suites and evaluation boundaries.
<table><tr><td>Suite</td><td>Purpose</td><td>Boundary</td></tr><tr><td>Contextual diagnostic</td><td>Rendered-distribution behavior.</td><td>Reporting only; not used for final ranking.</td></tr><tr><td>Hard benign diagnostic</td><td>Stress test for benign false positives.</td><td>Reporting only; not used for threshold tuning.</td></tr><tr><td>External transfer audit</td><td>Distribution-sensitivity analysis.</td><td>Reporting only; not merged into the primary evaluation.</td></tr></table>

Note: Diagnostic suites are excluded from model selection, threshold tuning, and primary protected-test ranking.

This separation allows diagnostic suites to reveal model weaknesses without compromising the independence of the protected-test evaluation.

## A.6 Model Configuration Summary

Table 8 summarizes the model families and the main evaluation boundary for each group. Detailed metric values are reported in the main Results section.

Table 8. Model configuration and evaluation boundaries.
<table><tr><td>Detector</td><td>Configuration</td><td>Evaluation boundary</td></tr><tr><td>Keyword</td><td>Deterministic lexical-pattern heuristic.</td><td>No train-time tuning.</td></tr><tr><td>MiniLM Ref.</td><td>Maximum similarity to the malicious-reference bank.</td><td>Decision threshold selected on validation only.</td></tr><tr><td>TF-IDF LR</td><td>Word- and character-level TF-IDF features.</td><td>Hyperparameters selected on validation only.</td></tr><tr><td>TF-IDF SVM</td><td>Word- and character-level TF-IDF features.</td><td>Hyperparameters selected on validation only.</td></tr><tr><td>TF-IDF RF</td><td>Word-level TF-IDF features.</td><td>Hyperparameters selected on validation only.</td></tr><tr><td>DistilBERT</td><td>Transformer encoder classifier with maximum sequence length 256.</td><td>Trained on the train split, selected on validation, and evaluated once on the</td></tr><tr><td>DeBERTa-v3</td><td>Transformer encoder classifier with maximum sequence length 192 under a</td><td>protected-test split. Trained on the train split, selected on validation, and evaluated once on the protected-test split.</td></tr></table>

Note: LR, SVM, RF, and Ref. denote logistic regression, support vector machine, random forest, and reference baseline, respectively. Detector names follow Table 3. The protected-test split is used only for final evaluation.

The final evaluation boundary is fixed as follows: training uses only the training split, model and checkpoint selection use only the validation split, and the protectedtest split is used only once for final evaluation after configuration freeze. Diagnostic suites are excluded from final model ranking and are reported only as supplementary behavioral evidence.