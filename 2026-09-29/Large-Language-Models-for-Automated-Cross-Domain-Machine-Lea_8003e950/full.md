# Large Language Models for Automated Cross-Domain Machine Learning Task Type Identification: A Benchmark Dataset and Evaluation

Petros Tsialis<sup>1</sup> Stefen Limmer<sup>2</sup> Tobias Rodemann<sup>2</sup> Martin Heckmann<sup>1</sup>

<sup>1</sup>University of Applied Sciences Aalen

<sup>2</sup>Honda Research Institute Europe

Abstract Machine learning task type identification is essential for constructing valid ML pipelines, yet in practice it is typically specified manually. We investigate whether large language models (LLMs) can infer both the data domain and the downstream prediction task directly from dataset-level information when only the target feature is provided by the user. Together with our LLM-based system we also release an annotated benchmark comprising 625 public tabular and time series datasets. We evaluate the proposed approach in three settings: (i) tabular datasets in comparison with established AutoML heuristics, (ii) cross-domain evaluation across tabular and time series datasets, and (iii) a practical deployment scenario using smaller local models. The results show consistent advantages for LLM-based task type identification, with increasing dificulty in heterogeneous and resource-constrained settings. LLM-based approaches outperform AutoGluon in the tabular setting, reaching 0.98 F1 macro compared to 0.93. In the cross-domain setting, the best model achieves 0.90 F1 macro, while smaller locally deployable models reach 0.75, indicating a trade-of between deployment feasibility and accuracy.

## 1 Introduction

Identifying the correct machine learning (ML) task for a dataset is a fundamental prerequisite for building efective ML pipelines, since preprocessing, model selection, and evaluation depend on the task formulation. However, most Automated Machine Learning (AutoML) systems assume that the downstream task, such as classification, regression, or time series forecasting, is already known. In practice, this step is often performed manually by data scientists or domain experts [30], which remains challenging in industrial settings where datasets are heterogeneous, poorly documented, and frequently handled by non-expert stakeholders [34]. Although AutoML frameworks have been applied across tabular data [40], time series [50, 49, 8], and natural language processing [45], task type identification is often either not addressed explicitly [15, 43, 29, 51, 5, 2, 16] or implemented through heuristic and rule-based methods [14, 33, 39]. These methods are mainly designed for downstream task identification in tabular settings and generally do not address domain identification, i.e., distinguishing the structural machine learning problem domain of a dataset, such as tabular data or time series data, across heterogeneous data modalities.

Recent advances in large language models (LLMs) suggest a possible alternative. LLMs have shown strong capabilities in processing structured inputs [46], extracting semantic patterns [48], and solving classification tasks [4]. Their ability to combine raw feature values, statistical summaries, and textual context makes them a promising candidate for identifying both the data domain and downstream task from dataset-level information. To study this problem systematically, we introduce an openly available benchmark for machine learning task type identification<sup>1</sup> and propose an LLMbased system<sup>2</sup> that infers task information from structured dataset representations, target-specific statistics, and optional textual descriptions. To systematically investigate this problem, we address the following research questions (RQs):

• RQ1: Baseline Comparison. How accurately can LLM-based approaches identify machine learning tasks on tabular datasets, and how do they compare to established AutoML systems?

• RQ2: Cross-Domain Capability. How accurately can LLM-based systems identify machine learning tasks across diferent data modalities, including tabular data and time series, when using state-of-the-art foundation models?

• RQ3: Practical Deployment. What level of task identification performance can be achieved using local or resource-constrained LLMs, and how does this performance compare to large, state-of-the-art models in realistic deployment scenarios?

## 2 Related Work

LLMs. Recent developments in large language models (LLMs) have shown that these models can efectively process and reason over structured data representations [47, 46]. In addition, LLMs have demonstrated strong performance on classification tasks [20, 6], particularly in zero-shot and few-shot settings [32, 10]. Prior work further suggests that this capability can be strengthened through appropriate prompt design, for example by using zero-shot or few-shot prompting [32, 10], encouraging explicit reasoning [47], or enriching the prompt with dataset-specific metainformation[4, 44].

LLM-Assisted AutoML. Building on these strong capabilities of LLMs, recent research has explored LLM-based approaches for automating ML workflows [18]. Several of these systems include some form of task interpretation, but they difer fundamentally from our setting. AutoML-GPT [55] uses an LLM to coordinate data processing, model selection, and hyperparameter tuning, but the task is specified through user-provided instructions and structured workflow prompts. AutoML-Agent [48] decomposes ML workflows into sub-tasks handled by LLM agents, but also starts from an explicit task description. MLAgentBench [23] evaluates language agents on end-to-end ML experimentation tasks, where each task is defined by a textual task description as well as starter files such as code and data. Thus, the agent is required to improve or solve an already specified ML task rather than infer the task type from dataset-level information. CAAFE [22] uses LLMs to generate semantic features from dataset descriptions, where the description typically provides substantial information about the prediction problem. MLCopilot [54] recommends ML solutions by retrieving experience from previous ML tasks, but the new task is again represented through an explicit task specification. Thus, existing LLM-assisted AutoML systems may interpret, optimize, or operationalize ML tasks, but the downstream task is usually given by the user or available through descriptions that already state the prediction problem. In contrast, we treat task type identification as a standalone problem: only the target feature is assumed to be known, while the data domain and downstream task must be inferred from dataset-level information. The model receives feature names, a serialized dataset excerpt, and target-specific statistics, optionally complemented by dataset descriptions from the dataset creators. Importantly, these descriptions generally characterize the dataset rather than explicitly defining the underlying ML task. To systematically assess their contribution to task identification, we conduct all experiments both with and without the inclusion of these dataset descriptions.

## 3 Datasets

To the best of our knowledge, our collection is the first openly available benchmark specifically designed for machine learning task type identification. It enables systematic evaluation of whether models can infer both the data domain and downstream prediction task from dataset content and meta-information, while its heterogeneous sources and file formats reflect realistic data conditions. The benchmark comprises 625 public datasets from OpenML [9], the UCI Machine Learning Repository [31], Kaggle [1], and the Time Series Classification Archive [37]. As shown in Figure 1, it follows a two-level taxonomy: datasets are first assigned to Time\_Series or Tabular, with 299 and 326 datasets, respectively, and then to binary classification, multiclass classification, or regression/forecasting. The time series branch contains 101 regression, 61 binary classification, and 137 multiclass classification datasets; the tabular branch contains 140 regression, 86 binary classification, and 100 multiclass classification datasets. This taxonomy supports evaluation of both domain identification and downstream task identification within and across dataset domains.

![](images/3cfbec9a241319303f357a017d50a58118a85f6309bc1bacebc59cda252ba2ae.jpg)  
Figure 1: Two-level taxonomy of the benchmark dataset collection.

The collection consists of the original datasets and a tabular meta-information dataset which contains e.g. dataset name, data domain, downstream task, target variable, and dataset description. A full description of the metadata fields, dataset-type details, and loading-pipeline details are provided in Appendix A. Since the datasets vary strongly in file format, storage structure, and representation, we developed a loading pipeline that parses diferent formats while preserving the original structure where possible. Extensive preprocessing into a uniform schema is avoided to better reflect realistic data conditions.

Tabular classification. Tabular classification datasets consist of independent samples described by input features and an explicitly specified target feature. The target feature represents a discrete outcome. If it contains two possible classes, the dataset is treated as binary classification; if it contains more than two classes, it is treated as multiclass classification [41].

Tabular regression. Tabular regression datasets also consist of independent samples described by input features and an explicitly specified target feature. In contrast to classification, the target feature represents a continuous numerical outcome. The objective is therefore to predict a realvalued response from the available input features [41].

Time series classification. Time series classification datasets consist of samples whose observations are ordered over time or another sequence dimension. Each sample may be represented by a univariate or multivariate sequence, where the order of observations is part of the data representation. The target feature is categorical, and the objective is to assign each time series sample to one of the predefined classes [26, 38].

Time series forecasting/regression. Time series forecasting and regression datasets consist of samples whose observations are ordered over time or another sequence dimension, but the target feature is continuous. In forecasting tasks, the target feature typically represents a future value estimated from historical observations at a defined prediction horizon. In time series regression tasks, the target feature may instead describe a continuous property or external response associated with the sequence. We treat both cases treated as regression because they require the prediction and can be addressed using the same modeling approaches [24, 19, 38].

## 4 Methodology

The proposed system identifies both the data domain and the downstream ML task from structured dataset information and optional semantic context. As shown in Figure 2, the workflow starts with two mandatory inputs: the dataset and an explicitly specified target feature, which defines the prediction objective and cannot easily be inferred automatically. When available, a textual dataset description is added as semantic context.

![](images/8ad62a34ac0ecc07e4ee030425deec404d450e3260d9906a395004426bad35a3.jpg)  
Figure 2: Workflow of the proposed system. Given a dataset, target feature, and optional textual description, the system extracts target-specific statistics, generates a compact serialized representation, and combines the information into a prompt for predicting the data domain and downstream ML task via the LLM.

Target-specific statistics. Before prompt construction, task-relevant statistics are extracted from the target variable, including value counts, unique values, mean, standard deviation, minimum, and maximum. For text-like targets, word, stopword, and character counts are additionally extracted with their standard deviations. These statistics provide compact information about the prediction objective.

Dataset serialization. The dataset itselfis then serialized into a structured DataFrame representation that can be processed by the LLM, as shown in Appendix B. The representation preserves the rowand column-based dataset structure, including original feature names, which can provide semantic cues about the variables. The target feature name is always retained and explicitly marked in the prompt because it defines the prediction objective. Several serialization strategies were explored, and the final setup employs this compact matrix-like representation to preserve the original structure while remaining suitable for prompt-based input [3, 46, 13, 25, 36, 11, 21, 17, 42, 53].

Group-wise row sampling. To reduce computational cost and avoid exceeding the model context window, each dataset is represented by a subset of selected samples and features [12, 27]. We use group-wise random sample selection, where contiguous sample groups are selected without replacement and then reordered by their original positions. This preserves short-range patterns relevant to structured data, such as time series or grouped records, while keeping the input tractable. The target feature is always retained.

Prompting structure. Prompt design is central to the proposed system because it strongly afects LLM performance. As shown in Appendix B, the prompting framework combines a system prompt, which defines the task and output format, with a user prompt containing the serialized dataset, target feature information, extracted target-specific statistics, and optional dataset description. In few-shot settings, task-specific labeled examples from the training data are added to provide representative guidance for each downstream task. We evaluate both zero-shot and few-shot prompting. Zero-shot prompting uses only the system and dataset-specific user prompt, whereas few-shot prompting additionally includes labeled examples.

Data split and hyperparameter selection. As shown in Table 1, the experimental hyperparameter space includes the underlying LLM, reasoning mode, prompting strategy, and dataset-specific information configuration. GPT-5.3 serves as the cloud-based state-of-the-art model, while Qwen provides locally deployable alternatives with demonstrated strength in structured-data and reasoning tasks [35]. For GPT-5.3, the reasoning efort was set to high in all experimental configurations. We include both recent Qwen3 [52] models and the established Qwen2.5-14B. The larger 14B models target high-memory industrial GPUs, such as the NVIDIA H100, whereas the compact 4B FP8 variants are intended for deployment on consumer-grade hardware, such as the NVIDIA RTX 5090. We compare zero-shot and few-shot prompting and vary the provided dataset information, ranging from the serialized dataset alone to additional target-specific statistics and optional dataset descriptions. The target feature is specified in all configurations, and hyperparameter optimization is conducted independently for each experiment. For configuration selection, the dataset collection is split into stratified training, validation, and test partitions of 20%, 20%, and 60%. Since the foundation models remain fixed, this split separates method development, configuration selection, and final evaluation rather than model-parameter training. Because we do not tune model parameters, the training split can be substantially smaller than in traditional machine learning settings. The training split is used for manual method development, including prompt engineering for the system and user prompts, adaptation of the dataset serialization strategy, selection of few-shot examples, and identification of the most informative target-specific statistics. The validation split is then used for model selection and dataset-configuration selection, including the choice of model, in-context learning strategy, reasoning mode, and input information setting. Finally, the selected configurations are evaluated once on the held-out test split. The comparatively large test partition provides a broad and statistically more stable basis for the final evaluation across heterogeneous datasets. We do not use conventional cross-validation or nested cross-validation because the tuning process on the training split involves partially manual design decisions, particularly prompt engineering, dataset serialization, and representation choices. Repeating these development steps independently for multiple cross-validation folds would require substantial manual efort and would make it dificult to ensure that each fold is optimized consistently and without introducing experimenter bias. Instead, once the methodology has been finalized using the training split, we assess the statistical robustness of the reported results by applying bootstrap resampling independently to the validation and test splits. Specifically, we generate 1,000 bootstrap samples by sampling with replacement from each evaluation split and recompute the performance metrics for every sample. The reported results correspond to the mean and standard deviation across these bootstrap samples, providing statistically robust performance estimates while preserving the fixed train/validation/test protocol.

Table 1: Hyperparameter space evaluated in the LLM experiments from January to March 2026, including model choice, reasoning mode, prompting strategy, and dataset information configuration.
<table><tr><td>Hyperparameter</td><td>Evaluated values</td></tr><tr><td>Model</td><td>GPT-5.3; Qwen3-14B; Qwen2.5-14B-Instruct; Qwen3-4B-Instruct- 2507-FP8; Qwen3-4B-Thinking-2507-FP8</td></tr><tr><td>Reasoning</td><td>√; X</td></tr><tr><td>Prompting strategy</td><td>Zero-shot; Few-shot</td></tr><tr><td>figuration</td><td>Dataset information con- Target feature name; target feature name + dataset description; target feature name + dataset description + target-specific statistics</td></tr></table>

## 5 Experimental Results

Experimental setup. The evaluation follows the three research questions from Section 1: Experiment 1 compares the proposed approach against three established AutoML frameworks, namely AutoGluon, H2O, and NaiveAutoML, on tabular datasets. Experiment 2 evaluates cross-domain identification on tabular and time series datasets and Experiment 3 assesses smaller locally deployable models. We restrict the comparison with AutoML frameworks to tabular datasets because AutoGluon, H2O, NaiveAutoML, and most other AutoML frameworks only support task type identification for tabular data Validation results are summarized in Figure 3, selected configurations are evaluated once on the held-out test split. The results report both DD+ST and ST-only settings, where DD denotes the textual dataset description and ST denotes target-specific statistics extracted from the target feature. This comparison contrasts the full information setting with the use of dataset-derived target-specific statistics alone.

Experiment 1. As shown in Figure 3, the best validation performance in the tabular-only setting is achieved with DD+ST, followed closely by ST alone. GPT-5 achieves the strongest validation performance, Qwen3-14B remains competitive, and Qwen2.5-14B performs weakest, while reasoning mode and prompting strategy have only minor efects.  
Table 2: Test-set results of the best-performing models in Experiment 1 for tabular ML task identification compared to the AutoGluon baseline. Performance metrics for the LLM-based methods are reported as the mean and standard deviation over 1,000 bootstrap resamples. DD denotes textual dataset descriptions and ST denotes target-specific statistics.
<table><tr><td colspan="2"></td><td colspan="2">Few-shot</td><td colspan="2">Zero-shot</td></tr><tr><td>Model</td><td>Reasoning Info.</td><td></td><td>F1 Macro</td><td>Balanced Acc.</td><td>F1 Macro</td><td>Balanced Acc.</td></tr><tr><td>AutoGluon</td><td></td><td></td><td>一</td><td>一</td><td> $\mathbf { 0 . 9 3 \pm 0 . 0 1 }$ </td><td> $0 . 9 3 \pm 0 . 0 1$ </td></tr><tr><td>NaiveAutoML</td><td></td><td></td><td></td><td></td><td> $0 . 8 0 \pm 0 . 0 1$ </td><td> $0 . 8 1 \pm 0 . 0 1$ </td></tr><tr><td>H2O</td><td>一</td><td>一</td><td>一</td><td>一</td><td> $0 . 5 9 \pm 0 . 0 1$ </td><td> $0 . 5 8 \pm 0 . 0 1$ </td></tr><tr><td>Qwen3-14B</td><td>x</td><td> $\mathrm { D D } { + } { \cal S } \mathrm { T }$ </td><td> $0 . 9 8 \pm 0 . 0 1$ </td><td> $0 . 9 8 \pm 0 . 0 1$ </td><td> $\mathbf { 0 . 9 8 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 9 8 \pm 0 . 0 0 }$ </td></tr><tr><td>Qwen3-14B</td><td>√</td><td> $\mathrm { D D } { + } { \cal S } \mathrm { T }$ </td><td> $0 . 9 8 \pm 0 . 0 0$ </td><td> $0 . 9 8 \pm 0 . 0 0$ </td><td> $0 . 9 8 \pm 0 . 0 1$ </td><td> $0 . 9 8 \pm 0 . 0 1$ </td></tr><tr><td>GPT-5.3</td><td>√</td><td>DD+ST</td><td> $0 . 9 7 \pm 0 . 0 0$ </td><td> $0 . 9 7 \pm 0 . 0 0$ </td><td> $0 . 9 7 \pm 0 . 0 1$ </td><td> $0 . 9 7 \pm 0 . 0 1$ </td></tr><tr><td>GPT-5.3</td><td>X</td><td>DD+ST</td><td> $0 . 9 7 \pm 0 . 0 1$ </td><td> $0 . 9 7 \pm 0 . 0 1$ </td><td> $0 . 9 7 \pm 0 . 0 2$ </td><td> $0 . 9 7 \pm 0 . 0 1$ </td></tr><tr><td>Qwen3-14B</td><td>x</td><td>ST</td><td> $0 . 9 6 \pm 0 . 0 1$ </td><td> $0 . 9 6 \pm 0 . 0 0$ </td><td> $0 . 9 6 \pm 0 . 0 1$ </td><td> $0 . 9 6 \pm 0 . 0 1$ </td></tr><tr><td>Qwen3-14B</td><td>√</td><td>ST</td><td> ${ \bf 0 . 9 6 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 9 6 \pm 0 . 0 1 }$ </td><td> $0 . 9 5 \pm 0 . 0 2$ </td><td> $0 . 9 6 \pm 0 . 0 1$ </td></tr><tr><td>GPT-5.3</td><td>√</td><td>ST</td><td> $0 . 9 7 \pm 0 . 0 0$ </td><td> $0 . 9 7 \pm 0 . 0 0$ </td><td> $0 . 9 6 \pm 0 . 0 0$ </td><td> $0 . 9 7 \pm 0 . 0 0$ </td></tr><tr><td>GPT-5.3</td><td>x</td><td>ST</td><td> $0 . 9 6 \pm 0 . 0 1$ </td><td> $0 . 9 6 \pm 0 . 0 1$ </td><td> $0 . 9 6 \pm 0 . 0 1$ </td><td> $0 . 9 6 \pm 0 . 0 1$ </td></tr></table>

![](images/46648dca9204f6c62002816391e055a24ee466b7db96d88518bb4293ea498d4c.jpg)

![](images/bc72c7e4c730aeaf4c0a3447c447074680ae3afed5877edce9a043b6a1be22e4.jpg)

![](images/ec63a5d77be3669d1e77031a4776505c0692436106ccb5672111fa7e013a8007.jpg)

![](images/c6bf5d824d7b93d78aa619b0c612568067ad9c961320fa4871c608dea713171e.jpg)  
(a) Experiment 1

![](images/9d0744c28220082cf71d74f42727e6602a63cc4bf1c2bcd131e32d428e11109a.jpg)  
(b) Experiment 2

![](images/1eca074566c85fd7321c4837233fca701875427de0784f74ef38ca30d519b64e.jpg)  
(c) Experiment 3  
Figure 3: F1-macro distributions across validation runs for Experiments 1–3. Each violin aggregates all runs for a given hyperparameter across the remaining ones. BP denotes the base prompt, DD dataset descriptions, ST target-specific statistics, and DD+ST their combination.

Based on these results, the primary test evaluation uses DD+ST, with ST-only results reported for comparison. Table 2 shows that all LLM-based configurations outperform AutoGluon. While AutoGluon reaches 0.93 F1 macro, DD+ST configurations reach 0.97–0.98, with Qwen3-14B achieving 0.98 across all prompting and reasoning settings. ST-only configurations also outperform Auto-Gluon, reaching 0.95–0.97 F1 macro, indicating that target-specific statistics already provide strong signals for tabular task identification. Figure 4 further shows that AutoGluon has a tendency to confuse regression with multiclass classification, whereas the selected LLM configuration produces only rare misclassifications. without caption

![](images/3383af0496fadc6d95f03f4e6f24c10f47e85f6d9aea9d4cf4680ad060c41b96.jpg)  
(a) AutoGluon

![](images/2e3926ebdeb078b2e8f6a6176238c362afe3a6d48540eb71e4eebcc89cb5c141.jpg)  
(b) Qwen3-14B  
Figure 4: Test-set confusion matrices for Experiment 1. (a): AutoGluon baseline. (b): selected Qwen3- 14B configuration.

Experiment 2. Figure 3 indicates that validation performance across tabular and time series datasets is highest when semantic dataset information is included. DD+ST performs best, followed closely by DD alone, which also shows lower variance. In contrast, ST alone and the base prompt lead to lower and more variable performance, suggesting that target-specific statistics are useful for some datasets but less reliable across the heterogeneous benchmark. This indicates that crossdomain task identification depends more strongly on semantic context than the tabular-only setting, since the model must infer both the downstream task and the data domain. GPT-5.3 achieves the strongest validation performance, followed by Qwen2.5-14B and Qwen3-14B. Zero-shot prompting and reasoning show slightly more stable overall results, but high-performing configurations occur across several settings, indicating that model choice and dataset information have stronger efects. The primary test evaluation therefore uses DD+ST, with ST-only results reported for comparison. As shown in Table 3, GPT-5.3 with reasoning achieves the best DD+ST result, reaching 0.90 F1 macro in the few-shot setting. GPT-5.3 without reasoning reaches 0.86 and 0.84 F1 macro in the few-shot and zero-shot settings, respectively, indicating that reasoning is beneficial for the strongest cloud-based model. Among the local models, Qwen2.5-14B performs best with 0.84 F1 macro in the few-shot setting, while Qwen3-14B reaches 0.76 and 0.77 in the few-shot and zero-shot settings.

Table 3: Test-set results of the best-performing models in Experiment 2 for cross-domain ML task identification across tabular and time series datasets. Performance metrics are reported as the mean and standard deviation over 1,000 bootstrap resamples. DD denotes textual dataset descriptions and ST denotes target-specific statistics.
<table><tr><td colspan="5">Few-shot</td><td colspan="2">Zero-shot</td></tr><tr><td>Model</td><td>Reasoning Info.</td><td></td><td>F1 Macro</td><td>Balanced Acc. F1 Macro</td><td></td><td>Balanced Acc.</td></tr><tr><td>Qwen2.5-14B-Inst. X</td><td></td><td>DD+ST</td><td> $0 . 8 4 \pm 0 . 0 4$ </td><td> $0 . 8 5 \pm 0 . 0 3$ </td><td> $0 . 7 7 \pm 0 . 0 4$ </td><td> $0 . 7 7 \pm 0 . 0 4$ </td></tr><tr><td>Qwen3-14B</td><td>√</td><td>DD+ST</td><td> $0 . 7 6 \pm 0 . 0 2$ </td><td> $0 . 7 6 \pm 0 . 0 2$ </td><td> $0 . 7 7 \pm 0 . 0 2$ </td><td> $0 . 7 7 \pm 0 . 0 1$ </td></tr><tr><td>GPT-5.3</td><td>√</td><td>DD+ST</td><td> ${ \bf 0 . 9 0 \pm 0 . 0 3 }$ </td><td> ${ \bf 0 . 9 0 \pm 0 . 0 3 }$ </td><td> $0 . 8 5 \pm 0 . 0 3$ </td><td> $0 . 8 6 \pm 0 . 0 1$ </td></tr><tr><td>GPT-5.3</td><td>X</td><td>DD+ST</td><td> $0 . 8 6 \pm 0 . 0 2$ </td><td> $0 . 8 6 \pm 0 . 0 1$ </td><td> $0 . 8 4 \pm 0 . 0 3$ </td><td> $0 . 8 4 \pm 0 . 0 3$ </td></tr><tr><td>Qwen2.5-14B-Inst. X</td><td></td><td>ST</td><td> $0 . 7 4 \pm 0 . 0 3$ </td><td> $0 . 7 4 \pm 0 . 0 4$ </td><td> $0 . 5 2 \pm 0 . 0 6$ </td><td> $0 . 5 9 \pm 0 . 0 4$ </td></tr><tr><td>Qwen3-14B</td><td>√</td><td>ST</td><td> $0 . 6 8 \pm 0 . 0 4$ </td><td> $0 . 6 9 \pm 0 . 0 4$ </td><td> $0 . 7 1 \pm 0 . 0 5$ </td><td> $0 . 7 0 \pm 0 . 0 4$ </td></tr><tr><td>GPT-5.3</td><td>√</td><td>ST</td><td> $0 . 8 6 \pm 0 . 0 3$ </td><td> $0 . 8 6 \pm 0 . 0 2$ </td><td> $\mathbf { 0 . 8 7 \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 8 8 \pm 0 . 0 3 }$ </td></tr><tr><td>GPT-5.3</td><td>x</td><td>ST</td><td> $0 . 8 3 \pm 0 . 0 1$ </td><td> $0 . 8 4 \pm 0 . 0 1$ </td><td> $0 . 7 2 \pm 0 . 0 3$ </td><td> $0 . 7 3 \pm 0 . 0 2$ </td></tr></table>

ST-only results show that target-specific statistics can already support task identification, but less robustly across models. GPT-5.3 with reasoning reaches 0.86 and 0.87 F1 macro in the few-shot and zero-shot settings, respectively, whereas Qwen2.5-14B drops to 0.74 and 0.52. Thus, larger models exploit compact target-specific statistics more efectively, while smaller models benefit more from additional semantic context. Overall, Experiment 2 shows that cross-domain task identification is harder than the tabular-only setting, mainly because the model must also recognize the underlying data structure. This is also evident from the confusion matrix in Figure 5, where the most frequent confusions occur between the same downstream task across the two domains. Nevertheless, LLM-based methods generalize across heterogeneous modalities, with DD+ST remaining the most robust configuration.

Experiment 3. As shown in Figure 3, the best validation performance for smaller local models is achieved with DD+ST. ST provides useful information, but is less reliable as the only source of task type identification evidence. Since Experiment 3 includes only two local models, one with reasoning and one without, model choice and reasoning mode cannot be separated. Few-shot prompting performs slightly better than zero-shot prompting, but both are retained for test evaluation.

![](images/d68cb7896962053a98c900b82ac3821d987c66a0cde717afd1ec4afc31e3be3a.jpg)  
(a) Exp. 2: GPT-5

![](images/6d5b533ce2727aa43666a937e2fb65dd41120d13d21ef02e303505e960cdef2e.jpg)  
(b) Exp. 3: Qwen-4B  
Figure 5: Confusion matrices of the best-performing models in the cross-domain setting. (a): best state-of-the-art model in Experiment 2. (b): best locally deployable model in Experiment 3. TS denotes time series, and TB denotes tabular.

Table 4 shows that smaller locally deployable models achieve meaningful cross-domain performance, although below the larger models from Experiment 2. With DD+ST, Qwen3-4B-Instruct performs best, reaching 0.75 F1 macro and 0.74 balanced accuracy in the few-shot setting, and 0.73 for both metrics in the zero-shot setting. Qwen3-4B-Thinking performs substantially worse, reaching 0.62 F1 macro in few-shot and 0.45 in zero-shot, indicating that the reasoning-oriented variant is less robust at this model scale.

Table 4: Test-set results of the best-performing locally deployable models in Experiment 3 for crossdomain task identification under resource-constrained settings. Performance metrics are reported as the mean and standard deviation over 1,000 bootstrap resamples. DD denotes textual dataset descriptions and ST denotes target-specific statistics.
<table><tr><td colspan="5">Few-shot</td><td colspan="2">Zero-shot</td></tr><tr><td>Model</td><td>Reasoning</td><td>Info.</td><td></td><td>F1 Macro Balanced Acc. F1 Macro Balanced Acc.</td><td></td><td></td></tr><tr><td>Qwen3-4B-Instruct X</td><td></td><td>DD+ST</td><td> $\mathbf { 0 . 7 5 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 7 4 \pm 0 . 0 2 }$ </td><td> $0 . 7 3 \pm 0 . 0 2$ </td><td> $0 . 7 3 \pm 0 . 0 2$ </td></tr><tr><td>Qwen3-4B-Thinking√</td><td></td><td>DD+ST</td><td> $0 . 6 2 \pm 0 . 0 3$ </td><td> $0 . 6 4 \pm 0 . 0 3$ </td><td> $0 . 4 5 \pm 0 . 0 4$ </td><td> $0 . 4 5 \pm 0 . 0 4$ </td></tr><tr><td>Qwen3-4B-Instruct X</td><td></td><td>ST</td><td> $\mathbf { 0 . 5 0 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 5 8 \pm 0 . 0 2 }$ </td><td> $0 . 4 8 \pm 0 . 0 3$ </td><td> $0 . 5 6 \pm 0 . 0 2$ </td></tr><tr><td>Qwen3-4B-Thinking√</td><td></td><td>ST</td><td> $0 . 4 6 \pm 0 . 0 2$ </td><td> $0 . 5 5 \pm 0 . 0 2$ </td><td> $0 . 4 5 \pm 0 . 0 4$ </td><td> $0 . 4 8 \pm 0 . 0 4$ </td></tr></table>

ST-only results lead to a clear performance drop for both local models. Qwen3-4B-Instructdecreases from 0.75 to 0.50 F1 macro in few-shot and from 0.73 to 0.48 in zero-shot, while Qwen3-4B-Thinking remains near the lower performance range. This suggests that small local models require semantic context to resolve ambiguities between tabular and time series structures. Practical deployment is therefore feasible, but involves a clear trade-of between model size and taskidentification accuracy. Detailed runtime and API-cost results are provided in Appendix A, together with token-usage statistics. The confusion matrix in Figure 5 of the best locally deployable model in Experiment 3 shows the same error pattern as the best state-of-the-art model in Experiment 2, but with more frequent misclassifications. While it shows strong separation between regression and classification, most errors arise from confusion between the domains (tabular vs. time series). Hence, the local model often identifies the ML task correctly but struggles with domain identification.

## 6 Discussion

The results show that LLM-based approaches can efectively identify machine learning task types within the proposed taxonomy. In the tabular setting, they outperform the heuristic AutoML baseline, indicating that structural and semantic dataset information improves task inference beyond rule-based criteria. In the cross-domain setting, performance decreases once time series datasets are included, mainly because the model must also recognize the underlying data domain. This highlights domain identification as an important component for future AutoML systems.

The experiments further show that the input representation is crucial. The most stable results are obtained when the serialized dataset preserves the original row- and column-based structure, including feature names and an explicitly marked target feature, and is enriched with dataset descriptions and target-specific statistics. At the same time, target-specific statistics alone still achieve competitive results, especially for stronger state-of-the-art models. This indicates that taskrelevant signals can be inferred directly from the dataset without textual descriptions. Nevertheless, in the cross-domain setting, the inclusion of dataset descriptions leads to some improvements in performance and reduces variance also for the stronger models. In contrast, smaller local models are less robust when only target-specific statistics are provided, suggesting that they depend more strongly on additional semantic context. Dataset descriptions are therefore not essential for accurate task identification but they improve the robustness, particularly in the cross-domain scenario and for the smaller-models. Overall, the results demonstrate that accurate task identification is feasible with minimal user-provided information, namely the target feature and, implicitly during the dataset creation phase, the feature names.

Several limitations remain. First, the benchmark mainly consists of curated public datasets and may not fully reflect the noise, inconsistency, and complexity of industrial data.Second, potential data contamination cannot be ruled out [28, 7]. This is a general limitation when evaluating pretrained LLMs on public datasets, particularly for closed-source models with undisclosed training corpora. Public datasets, metadata, or related variants may have been included during pretraining, which can lead to improved performance. We mitigate this risk by excluding datasets listed in the LM Contamination Index<sup>3</sup> and by defining the ML task-type labels independently of the original dataset repositories. Since these labels were manually assigned specifically for our benchmark and are not publicly available as part of the original datasets, they cannot have been directly memorized during pretraining, even if the corresponding dataset was included in the model’s training corpus. However, contamination through dataset content or metadata remains possible, and future work should include dedicated contamination analyses, such as memorization tests or more comprehensive contamination indices. Third, the evaluation focuses on task type identification metrics such as F1 macro, accuracy, and balanced accuracy, but does not measure how misclassifications afect downstream pipeline stages. Future work should therefore quantify the impact of task type identification errors on preprocessing, model selection, training, evaluation, and overall pipeline performance. Automatic target-feature identification is another important extension, as it would remove the need for manual target specification.

## 7 Conclusion

This work introduced an LLM-based system and benchmark for machine learning task type identification. The results show that LLMs outperform heuristic AutoML task inference in the tabular setting and generalize to cross-domain identification across tabular and time series datasets. Smaller local models remain viable but require semantic context and show a clear accuracy–deployability trade-of. Overall, the findings support treating task identification as an explicit component of AutoML pipelines rather than as a fixed user-provided input.

Ethical considerations. Automated task identification can help democratize AutoML by making ML workflows more accessible to non-expert users. However, incorrect predictions may propagate to later pipeline stages and lead to unsuitable preprocessing, model choices, or evaluation metrics, ultimately leading to biased or unreliable outcomes. The proposed system should therefore be used as a transparent, inspectable decision-support component under expert oversight.

Acknowledgements. This research was funded by the Honda Research Institute Europe, GmbH.

## References

[1] Kaggle: The World’s AI Proving Ground, November 2026. URL https://www.kaggle.com/.

[2] lightwood, April 2026. URL https://github.com/mindsdb/lightwood. original-date: 2019- 05-20T21:31:14Z.

[3] Armen Aghajanyan, Dmytro Okhonko, Mike Lewis, Mandar Joshi, Hu Xu, Gargi Ghosh, and Luke Zettlemoyer. HTLM: Hyper-Text Pre-Training and Prompting of Language Models, July 2021. URL https://arxiv.org/abs/2107.06955v1.

[4] Albert Agisha Ntwali, Luca Rück, and Martin Heckmann. Detection of Personal Data in Structured Datasets Using a Large Language Model. In LLM-DPM ’2025, Workshop on Next Gen Data and Process Management: Large Language Models and Beyond, Berlin, Germany, June 2025. doi: 10.48550/arXiv.2506.22305. URL https://ui.adsabs.harvard.edu/abs/ 2025arXiv250622305A. ADS Bibcode: 2025arXiv250622305A.

[5] Moez Ali. PyCaret: An open source, low-code machine learning library in Python. April 2020. URL https://www.pycaret.org.

[6] Marcelo V. C. Aragão, Augusto G. Afonso, Rafaela C. Ferraz, Rairon G. Ferreira, Sávio G. Leite, Felipe A. P. De Figueiredo, and Samuel B. Mafra. A practical evaluation of AutoML tools for binary, multiclass, and multilabel classification. Scientific Reports, 15(1):17682, May 2025. ISSN 2045-2322. doi: 10.1038/s41598-025-02149-x. URL https://www.nature.com/articles/ s41598-025-02149-x.

[7] Simone Balloccu, Patrícia Schmidtová, Mateusz Lango, and Ondrej Dusek. Leak, Cheat, Repeat: Data Contamination and Evaluation Malpractices in Closed-Source LLMs. In Yvette Graham and Matthew Purver, editors, Proceedings ofthe 18th Conference ofthe European Chapter ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 67–93, St. Julian’s, Malta, March 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.eacl-long. 5. URL https://aclanthology.org/2024.eacl-long.5/.

[8] Mitra Baratchi, Can Wang, Stefen Limmer, Jan N. van Rijn, Holger Hoos, Thomas Bäck, and Markus Olhofer. Automated machine learning: past, present and future. Artificial Intelligence Review, 57(5):122, April 2024. ISSN 1573-7462. doi: 10.1007/s10462-024-10726-1. URL https://doi.org/10.1007/s10462-024-10726-1.

[9] Bernd Bischl, Giuseppe Casalicchio, Taniya Das, Matthias Feurer, Sebastian Fischer, Pieter Gijsbers, Subhaditya Mukherjee, Andreas C. Müller, László Németh, Luis Oala, Lennart Purucker, Sahithya Ravi, Jan N. Van Rijn, Prabhant Singh, Joaquin Vanschoren, Jos Van Der Velde, and Marcel Wever. OpenML: Insights from 10 years and more than a thousand papers. Patterns, 6(7):101317, July 2025. ISSN 26663899. doi: 10.1016/j.patter.2025.101317. URL https://linkinghub.elsevier.com/retrieve/pii/S2666389925001655.

[10] Youngjin Chae and Thomas Davidson. Large Language Models for Text Classification: From Zero-Shot Learning to Instruction-Tuning. Sociological Methods & Research, 55(2):501–567, May 2026. ISSN 0049-1241. doi: 10.1177/00491241251325243. URL https://doi.org/10. 1177/00491241251325243.

[11] Xiang Deng, Huan Sun, Alyssa Lees, You Wu, and Cong Yu. TURL: Table Understanding through Representation Learning. SIGMOD Rec., 51(1):33–40, June 2022. ISSN 0163-5808. doi: 10.1145/3542700.3542709. URL https://dl.acm.org/doi/10.1145/3542700.3542709.

[12] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. A Survey on In-context Learning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1107–1128, Miami, Florida, USA, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.64. URL https://aclanthology.org/2024.emnlp-main.64.

[13] Julian Eisenschlos, Maharshi Gor, Thomas Müller, and William Cohen. MATE: Multi-view Attention for Table Transformer Eficiency. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 7606–7619, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021. emnlp-main.600. URL https://aclanthology.org/2021.emnlp-main.600/.

[14] Nick Erickson, Jonas Mueller, Alexander Shirkov, Hang Zhang, Pedro Larroy, Mu Li, and Alexander Smola. AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data. 7th ICML Workshop on Automated Machine Learning, 2020. doi: 10.48550/arXiv.2003.06505.

[15] Matthias Feurer, Katharina Eggensperger, Stefan Falkner, Marius Lindauer, and Frank Hutter. Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning. Journal of Machine Learning Research, 23(261):1–61, 2022. ISSN 1533-7928. URL http://jmlr.org/papers/v23/21-0992. html.

[16] Pieter Gijsbers and Joaquin Vanschoren. GAMA: A General Automated Machine Learning Assistant. In Yuxiao Dong, Georgiana Ifrim, Dunja Mladenić, Craig Saunders, and Sofie Van Hoecke, editors, Machine Learning and Knowledge Discovery in Databases. Applied Data Science and Demo Track, volume 12461, pages 560–564. Springer International Publishing, Cham, 2021. ISBN 978-3-030-67669-8 978-3-030-67670-4. doi: 10.1007/978-3-030-67670-4\_39. URL https://link.springer.com/10.1007/978-3-030-67670-4\_39. Series Title: Lecture Notes in Computer Science.

[17] Heng Gong, Yawei Sun, Xiaocheng Feng, Bing Qin, Wei Bi, Xiaojiang Liu, and Ting Liu. TableGPT: Few-shot Table-to-Text Generation with Table Structure Reconstruction and Content Matching. In Donia Scott, Nuria Bel, and Chengqing Zong, editors, Proceedings of the 28th International Conference on Computational Linguistics, pages 1978–1988, Barcelona, Spain (Online), December 2020. International Committee on Computational Linguistics. doi: 10. 18653/v1/2020.coling-main.179. URL https://aclanthology.org/2020.coling-main.179/.

[18] Yang Gu, Hengyu You, Jian Cao, Muran Yu, Haoran Fan, and Shiyou Qian. Large Language Models for Constructing and Optimizing Machine Learning Workflows: A Survey. ACM Trans. Softw. Eng. Methodol., 2025. ISSN 1049-331X. doi: 10.1145/3773084. URL https: //dl.acm.org/doi/10.1145/3773084.

[19] Stephen Haben, Marcus Voss, and William Holderbaum. Time Series Forecasting: Core Concepts and Definitions. In Stephen Haben, Marcus Voss, and William Holderbaum, editors, Core Concepts and Methods in Load Forecasting: With Applications in Distribution Networks, pages 55–66. Springer International Publishing, Cham, 2023. ISBN 978-3-031-27852-5. doi: 10.1007/978-3-031-27852-5\_5. URL https://doi.org/10.1007/978-3-031-27852-5\_5.

[20] Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, and David Sontag. TabLLM: Few-shot Classification of Tabular Data with Large Language Models. In Francisco Ruiz, Jennifer Dy, and Jan-Willem van de Meent, editors, Proceedings ofThe 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 5549–5581. PMLR, April 2023. URL https://proceedings. mlr.press/v206/hegselmann23a.html.

[21] Jonathan Herzig, Pawel Krzysztof Nowak, Thomas Müller, Francesco Piccinno, and Julian Eisenschlos. TaPas: Weakly Supervised Table Parsing via Pre-training. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault, editors, Proceedings ofthe 58th Annual Meeting ofthe Association for Computational Linguistics, pages 4320–4333, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.398. URL https://aclanthology. org/2020.acl-main.398/.

[22] Noah Hollmann, Samuel Müller, and Frank Hutter. Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 44753–44775. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 8c2df4c35cdbee764ebb9e9d0acd5197-Paper-Conference.pdf.

[23] Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation, April 2024. URL http://arxiv.org/abs/ 2310.03302. arXiv:2310.03302.

[24] Rob J. Hyndman and George Athanasopoulos. Forecasting: principles and practice. OTexts, Melbourne, 2nd edition edition, 2018. ISBN 978-0-9875071-1-2.

[25] Hiroshi Iida, Dung Thai, Varun Manjunatha, and Mohit Iyyer. TABBIE: Pretrained Representations of Tabular Data. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou, editors, Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3446–3456, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.270. URL https://aclanthology.org/2021.naacl-main.270/.

[26] Hassan Ismail Fawaz, Germain Forestier, Jonathan Weber, Lhassane Idoumghar, and Pierre-Alain Muller. Deep learning for time series classification: a review. Data Mining and Knowledge Discovery, 33(4):917–963, July 2019. ISSN 1573-756X. doi: 10.1007/s10618-019-00619-1. URL https://doi.org/10.1007/s10618-019-00619-1.

[27] Sukriti Jaitly, Tanay Shah, Ashish Shugani, and Razik Singh Grewal. Towards Better Serialization of Tabular Data for Few-shot Classification with Large Language Models, December 2023. URL http://arxiv.org/abs/2312.12464. arXiv:2312.12464.

[28] Minhao Jiang, Ken Ziyu Liu, Ming Zhong, Rylan Schaefer, Siru Ouyang, Jiawei Han, and Sanmi Koyejo. Investigating Data Contamination for Pre-training Language Models, January 2024. URL http://arxiv.org/abs/2401.06059. arXiv:2401.06059.

[29] Haifeng Jin, François Chollet, Qingquan Song, and Xia Hu. AutoKeras: An AutoML Library for Deep Learning. Journal of Machine Learning Research, 24(6):1–6, 2023. URL http://jmlr. org/papers/v24/20-1355.html.

[30] Shubhra Kanti Karmaker (“Santu”), Md. Mahadi Hassan, Micah J. Smith, Lei Xu, Chengxiang Zhai, and Kalyan Veeramachaneni. AutoML to Date and Beyond: Challenges and Opportunities. ACM Comput. Surv., 54(8):175:1–175:36, 2021. ISSN 0360-0300. doi: 10.1145/3470918. URL https://dl.acm.org/doi/10.1145/3470918.

[31] Markelle Kelly, Rachel Longjohn, and Kolby Nottingham. UCI Machine Learning Repository. URL https://archive.ics.uci.edu/citation.

[32] Yanis Labrak, Mickael Rouvier, and Richard Dufour. A Zero-shot and Few-shot Study of Instruction-Finetuned Large Language Models Applied to Clinical and Biomedical Tasks. In Nicoletta Calzolari, Min-Yen Kan, Veronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue, editors, Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 2049–2066, Torino, Italia, May 2024. ELRA and ICCL. URL https://aclanthology.org/2024.lrec-main.185/.

[33] Erin LeDell and Sebastien Poirier. H2O AutoML: Scalable Automatic Machine Learning. 7th ICML Workshop on Automated Machine Learning (AutoML), July 2020. URL https://www. automl.org/wp-content/uploads/2020/07/AutoML\_2020\_paper\_61.pdf.

[34] Grace A. Lewis, Stephany Bellomo, and Ipek Ozkaya. Characterizing and Detecting Mismatch in Machine-Learning-Enabled Systems. In 2021 IEEE/ACM 1st Workshop on AI Engineering - Software Engineering for AI (WAIN), pages 133–140, May 2021. doi: 10.1109/WAIN52551.2021. 00028. URL https://ieeexplore.ieee.org/document/9474400/.

[35] Kuan-Hsun Lin, Tzu-Hang Kao, Lei-Chi Wang, Chen-Tsung Kuo, Paul Chih-Hsueh Chen, Yuan-Chia Chu, and Yi-Chen Yeh. Benchmarking large language models GPT-4o, llama 3.1, and qwen 2.5 for cancer genetic variant classification. npj Precision Oncology, 9(1):141, May 2025. ISSN 2397-768X. doi: 10.1038/s41698-025-00935-4. URL https://www.nature.com/ articles/s41698-025-00935-4.

[36] Qian Liu, Bei Chen, Jiaqi Guo, Morteza Ziyadi, Zeqi Lin, Weizhu Chen, and Jian-Guang Lou. TAPEX: Table Pre-training via Learning a Neural SQL Executor, July 2021. URL https: //arxiv.org/abs/2107.07653v3.

[37] Matthew Middlehurst, Patrick Schäfer, and Anthony Bagnall. Bake of redux: a review and experimental evaluation of recent time series classification algorithms. Data Mining and Knowledge Discovery, 38(4):1958–2031, July 2024. ISSN 1384-5810, 1573-756X. doi: 10.1007/ s10618-024-01022-1. URL https://link.springer.com/10.1007/s10618-024-01022-1.

[38] Navid Mohammadi Foumani, Lynn Miller, Chang Wei Tan, Geofrey I. Webb, Germain Forestier, and Mahsa Salehi. Deep Learning for Time Series Classification and Extrinsic Regression: A

Current Survey. ACM Comput. Surv., 56(9):217:1–217:45, April 2024. ISSN 0360-0300. doi: 10.1145/3649448. URL https://dl.acm.org/doi/10.1145/3649448.

[39] Felix Mohr and Marcel Wever. Naive automated machine learning. Machine Learning, 112(4): 1131–1170, April 2023. ISSN 0885-6125, 1573-0565. doi: 10.1007/s10994-022-06200-0. URL https://link.springer.com/10.1007/s10994-022-06200-0.

[40] Jonas Mueller, Xingjian Shi, and Alexander Smola. Faster, Simpler, More Accurate: Practical Automated Machine Learning with Tabular, Text, and Image Data. In Proceedings ofthe 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’20, pages 3509–3510, New York, NY, USA, August 2020. Association for Computing Machinery. ISBN 978-1-4503-7998-4. doi: 10.1145/3394486.3406706. URL https://dl.acm.org/doi/10.1145/ 3394486.3406706.

[41] Kevin P. Murphy. Machine learning: a probabilistic perspective. In Machine learning: a probabilistic perspective, Adaptive computation and machine learning series, pages 3–4. MIT Press, Cambridge, Mass., 4. print. edition, 2013. ISBN 978-0-262-01802-9.

[42] Ahmed Nassar, Nikolaos Livathinos, Maksym Lysak, and Peter Staar. TableFormer: Table Structure Understanding With Transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4614–4623, June 2022.

[43] Randal S. Olson and Jason H. Moore. TPOT: A Tree-Based Pipeline Optimization Tool for Automating Machine Learning. In Frank Hutter, Lars Kotthof, andJoaquin Vanschoren, editors, Automated Machine Learning, pages 151–160. Springer International Publishing, Cham, 2019. ISBN 978-3-030-05317-8 978-3-030-05318-5. doi: 10.1007/978-3-030-05318-5\_8. URL http: //link.springer.com/10.1007/978-3-030-05318-5\_8. Series Title: The Springer Series on Challenges in Machine Learning.

[44] Mücahit Sahin and Martin Heckmann. Boosting with LLMs: A Hybrid LLM-boosting Framework for Automated Feature Type Inference. Knowledge-Based Systems, 2026. Submitted.

[45] Xingjian Shi, Jonas Mueller, Nick Erickson, Mu Li, and Alex Smola. Multimodal AutoML on Structured Tables with Text Fields. In 8th ICML Workshop on Automated Machine Learning (AutoML), 2021. URL https://openreview.net/forum?id=OHAIVOOl7Vl.

[46] Yuan Sui, Mengyu Zhou, Mingjie Zhou, Shi Han, and Dongmei Zhang. Table Meets LLM: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining, WSDM ’24, pages 645–654, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 979-8-4007-0371-3. doi: 10.1145/3616855.3635752. URL https://dl.acm.org/doi/10. 1145/3616855.3635752.

[47] Xiaoyu Tan, Haoyu Wang, Xihe Qiu, Leijun Cheng, Yuan Cheng, Wei Chu, Yinghui Xu, and Yuan Qi. Struct-X:Enhancing the Reasoning Capabilities of Large Language Models in Structured Data Scenarios. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1, pages 2584–2595, Toronto ON Canada, July 2025. ACM. ISBN 979-8-4007-1245-6. doi: 10.1145/3690624.3709381. URL https://dl.acm.org/doi/10.1145/ 3690624.3709381.

[48] Patara Trirat, Wonyong Jeong, and Sung Ju Hwang. AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML. arXiv, June 2025. doi: 10.48550/arXiv.2410.02958. URL http://arxiv.org/abs/2410.02958. arXiv:2410.02958 [cs].

[49] Can Wang, Thomas Back, Holger H. Hoos, Mitra Baratchi, Stefen Limmer, and Markus Olhofer. Automated Machine Learning for Short-term Electric Load Forecasting. In 2019 IEEE Symposium Series on Computational Intelligence (SSCI), pages 314–321, Xiamen, China, December 2019. IEEE. ISBN 978-1-7281-2485-8. doi: 10.1109/SSCI44817.2019.9002839. URL https://ieeexplore.ieee.org/document/9002839/.

[50] Can Wang, Mitra Baratchi, Thomas Bäck, Holger H. Hoos, Stefen Limmer, and Markus Olhofer. Towards Time-Series Feature Engineering in Automated Machine Learning for Multi-Step-Ahead Forecasting. Engineering Proceedings, 18(1):17, 2022. ISSN 2673-4591. doi: 10.3390/engproc2022018017. URL https://www.mdpi.com/2673-4591/18/1/17.

[51] Chi Wang, Qingyun Wu, Markus Weimer, and Erkang Zhu. FLAML: A Fast and Lightweight AutoML Library. In A. Smola, A. Dimakis, and I. Stoica, editors, Proceedings of Machine Learning and Systems, volume 3, pages 434–447, 2021. URL https://proceedings.mlsys. org/paper\_files/paper/2021/file/1ccc3bfa05cb37b917068778f3c4523a-Paper.pdf.

[52] Zhiqiang Wang, Yiran Pang, Yanbin Lin, and Xingquan Zhu. Adaptable and Reliable Text Classification using Large Language Models. In 2024 IEEE International Conference on Data Mining Workshops (ICDMW), pages 67–74, December 2024. doi: 10.1109/ICDMW65004.2024. 00015. URL https://ieeexplore.ieee.org/document/10917961/. ISSN: 2375-9259.

[53] Zhiruo Wang, Haoyu Dong, Ran Jia, Jia Li, Zhiyi Fu, Shi Han, and Dongmei Zhang. TUTA: Treebased Transformers for Generally Structured Table Pre-training. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, KDD ’21, pages 1780–1790, New York, NY, USA, August 2021. Association for Computing Machinery. ISBN 978-1-4503-8332-5. doi: 10.1145/3447548.3467434. URL https://dl.acm.org/doi/10.1145/3447548.3467434.

[54] Lei Zhang, Yuge Zhang, Kan Ren, Dongsheng Li, and Yuqing Yang. MLCopilot: Unleashing the Power of Large Language Models in Solving Machine Learning Tasks. In Yvette Graham and Matthew Purver, editors, Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2931–2959, St. Julian’s, Malta, March 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. eacl-long.179. URL https://aclanthology.org/2024.eacl-long.179/.

[55] Shujian Zhang, Chengyue Gong, Lemeng Wu, Xingchao Liu, and Mingyuan Zhou. AutoML-GPT: Automatic Machine Learning with GPT, May 2023. URL http://arxiv.org/abs/2305. 02499. arXiv:2305.02499.

## Submission Checklist

## 1. For all authors. . .

(a) Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? [Yes] The benchmark dataset is described in Section 3, the LLMbased methodology is described in Section 5, and the empirical analysis supporting the main claims is reported in Section 5. The scope and implications of the results are further discussed in Section 6.

(b) Did you describe the limitations of your work? [Yes] The limitations are discussed in Section 6. These include the use of curated public datasets, the possibility of benchmark contamination in foundation models, the focus on task type identification metrics rather than downstream pipeline efects, and the current assumption that the target feature is provided by the user.

(c) Did you discuss any potential negative societal impacts of your work? [Yes] Potential negative societal impacts are discussed in the ethical considerations paragraph in Section 7. In particular, we note that incorrect task predictions may propagate to later pipeline stages and lead to unsuitable preprocessing, model choices, or evaluation metrics.

(d) Did you read the ethics review guidelines and ensure that your paper conforms to them? (see https://2022.automl.cc/ethics-accessibility/) [Yes] The work follows the ethics guidelines, as also stated in the ethical considerations paragraph in Section 7. The system is positioned as a transparent decision-support component that should remain inspectable and subject to expert oversight.

## 2. If you ran experiments. . .

(a) Did you use the same evaluation protocol for all methods being compared (e.g., same benchmarks, data (sub)sets, available resources, etc.)? [Yes] All experiments follow the same train/validation/test split strategy described in Section 4. The experiments are aligned with the research questions and use comparable evaluation metrics and procedures to ensure consistent comparison across methods and settings.

(b) Did you specify all the necessary details of your evaluation (e.g., data splits, pre-processing, search spaces, hyperparameter tuning details and results, etc.)? [Yes] The dataset construction and benchmark structure are described in Section 3. The data splits, preprocessing and serialization procedure, hyperparameter space, and configuration-selection procedure are specified in Section 4. The corresponding empirical results are reported in Section 5.

(c) Did you repeat your experiments (e.g., across multiple random seeds or splits) to account for the impact of randomness in your methods or data? [No] We use fixed stratified train, validation, and test splits rather than repeated random splits. The training set is used for prompt adaptation, the validation set for configuration selection, and the final selected configurations are evaluated once on a held-out test set. Repeated evaluations were not easily feasible, as the optimization process on the training and, to some extent, validation set involved manual iterative refinement that could not easily be reproduced in an unbiased manner. Furthermore, since we only used pretrained models without additional training or fine-tuning, one major source of variability was not under our control.

(d) Did you report the uncertainty of your results (e.g., the standard error across random seeds or splits)? [No] We do not report standard errors across repeated random seeds or splits, since the experiments are based on fixed dataset partitions. However, in Fig. 3 we show the distributions of the results of our experiments on the validation set, including the results for each individual data point.

(e) Did you report the statistical significance of your results? [No] We do not report formal statistical significance tests. Instead, we report macro-averaged F1 scores, balanced accuracy, and confusion matrices across the benchmark datasets.

(f) Did you use enough repetitions, datasets, and/or benchmarks to support your claims? [Yes] We introduce and evaluate on a benchmark of 625 public tabular and time series datasets, as described in Section 3. This provides broad coverage for the considered task type identification setting.

(g) Did you compare performance over time and describe how you selected the maximum runtime? [No] Runtime-based performance comparison is not the focus of this work. The evaluation focuses on task type identification accuracy across dataset domains and model configurations.

(h) Did you include the total amount of compute and the type of resources used (e.g., type of gpus, internal cluster, or cloud provider)? [Yes] The evaluated model families and the intended deployment settings, including cloud-based models and locally deployable GPU settings, are described in Section 4.

(i) Did you run ablation studies to assess the impact of diferent components of your approach? [Yes] We investigate the influence of several hyperparameters as part of the configurationselection procedure, including model choice, In-context learning strategy, reasoning mode, and dataset-information configuration. In particular, we analyze the impact of diferent dataset-information settings, such as the inclusion or exclusion of textual dataset descriptions. Since we did not perform additional model training or fine-tuning, we consider this systematic evaluation of inference-time configurations to be the most appropriate form of ablation study for our setting.

## 3. With respect to the code used to obtain your results. . .

(a) Did you include the code, data, and instructions needed to reproduce the main experimental results, including all dependencies (e.g., requirements.txt with explicit versions), random seeds, an instructive README with installation instructions, and execution commands (either in the supplemental material or as a url)? [Yes] The project repository contains the code, dependency specification, execution instructions, and documentation required to reproduce the main experiments. The dataset is provided through the dataset repository. Both repositories are linked in Section 1.

(b) Did you include a minimal example to replicate results on a small subset of the experiments or on toy data? [No] We do not include a separate toy-data example. However, the repository provides detailed instructions for running the code on the released benchmark data. The default configuration uses a locally deployable Qwen3-4B-Instruct model, which can be executed on consumer-grade GPU hardware such as an RTX 5090.

(c) Did you ensure suficient code quality and documentation so that someone else can execute and understand your code? [Yes] The code includes comments and documentation, and the README provides detailed instructions for installing the dependencies, accessing the dataset, and executing the experiments.

(d) Did you include the raw results of running your experiments with the given code, data, and instructions? [No] We do not include all raw model outputs. However, the fixed dataset splits are released with the benchmark, and the experiments are reproducible from the provided code, data, and instructions. Some variation may remain due to nondeterminism in LLM outputs.

(e) Did you include the code, additional data, and instructions needed to generate the figures and tables in your paper based on the raw results? [Yes] The repository includes comparison and plotting utilities for recreating the reported tables and figures. These utilities can be used directly from the utils folder or imported after installing the package as described in the README.

4. If you used existing assets (e.g., code, data, models). . .

(a) Did you cite the creators of used assets? [Yes] The creators and sources of the used public datasets, model families, and relevant software frameworks are cited in the paper where applicable.

(b) Did you discuss whether and how consent was obtained from people whose data you’re using/curating if the license requires it? [N/A] The benchmark is constructed from publicly available data sources. Where applicable, dataset licenses and citation requirements were reviewed and the corresponding dataset sources are cited in the paper. We do not collect data directly from human participants.

(c) Did you discuss whether the data you are using/curating contains personally identifiable information or ofensive content? [N/A] The work relies on public benchmark datasets and focuses on dataset-level task type identification. We do not introduce new person-level annotations or collect additional personally identifiable information.

## 5. If you created/released new assets (e.g., code, data, models). . .

(a) Did you mention the license of the new assets (e.g., as part of your code submission)? [Yes] The released code and benchmark assets include license information in the corresponding GitHub repository and Zenodo dataset release.

(b) Did you include the new assets either in the supplemental material or as a url (to, e.g., GitHub or Hugging Face)? [Yes] The released code and benchmark dataset are provided through the project repository and the dataset repository, which are linked in Section 1.

6. If you used crowdsourcing or conducted research with human subjects. . .

(a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A] This work does not use crowdsourcing and does not involve human subject experiments.

(b) Did you describe any potential participant risks, with links to institutional review board (irb) approvals, if applicable? [N/A] This work does not involve human participants.

(c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A] No participants were recruited or compensated.

## 7. If you included theoretical results. . .

(a) Did you state the full set of assumptions of all theoretical results? [N/A] This work does not include theoretical results.

(b) Did you include complete proofs of all theoretical results? [N/A] This work does not include theoretical results.

## Appendix

## A Additional Dataset Details

## A.1 Repository Links

Table 5: Repository links for the benchmark dataset, implementation, and LM Contamination Index.
<table><tr><td>Resource</td><td>Link</td></tr><tr><td>Benchmark dataset</td><td>https://zenodo.org/records/21649607</td></tr><tr><td rowspan="2">Project repository</td><td>https://github.com/ptsialis/</td></tr><tr><td>Machine-Learning-Task-Type-Identification</td></tr><tr><td>LM Contamination Index</td><td>https://hitz-zentroa.github.io/lm-contamination/</td></tr></table>

## A.2 Dataset Metadata

In the benchmark dataset the original files are stored in a taxonomy-based folder structure. In addition to the original dataset files, the benchmark includes a tabular meta-information dataset. This table stores high-level information for each dataset, including the dataset name, data domain, downstream task, target variable, dataset description, multi-target indicator, and target data type. For time series datasets, the metadata additionally records whether the data are univariate or multivariate and whether they contain one or multiple time series. General dataset properties are also included, such as the number of instances, the number of features, the dataset format, the relative storage path, and the original download link. Figure 6 shows a representative metadata entry.

## A.3 Dataset Types

The benchmark contains tabular and time series datasets. Tabular datasets are represented as collections of independent samples with feature columns and an explicitly specified target variable. They are assigned to binary classification, multiclass classification, or regression depending on the target variable and prediction objective.

Time series datasets contain observations with an intrinsic sequential or temporal structure. They include both univariate and multivariate time series, as well as datasets consisting of either a single time series or multiple time series instances. Time series classification datasets use categorical targets, whereas time series forecasting and regression datasets use continuous targets. In this work, forecasting is treated as a specific form of regression, since both require the prediction of continuous values.

## A.4 Runtime and Cost Considerations

In addition to predictive performance, we report the practical access characteristics of the evaluated models, since deployment feasibility depends not only on accuracy but also on whether a model is accessed through a commercial API or executed locally. Table 6 summarizes the fee type, deployment-related latency, and monetary cost for each evaluated model and in-context learning strategy. Since the monetary cost of API-based models is largely determined by the number of processed input and output tokens, the corresponding token-usage statistics, including the average token consumption for the zero-shot and few-shot settings, are reported in Appendix Table 7. AutoGluon achieves by far the lowest latency, with an average inference time of only 0.0004s, making it several orders of magnitude faster than all LLM-based approaches. However, this speed advantage must be interpreted in relation to predictive performance. The locally deployed LLMs require longer inference times, but several configurations remain practically eficient, with latencies ofonly a few seconds per prompt. In particular, the non-thinking Qwen models show that local LLMbased inference can still be feasible for ofline dataset analysis. Therefore, waiting a few additional seconds per dataset can be worthwhile when LLM-based methods provide improved classification performance, stronger semantic interpretation of dataset metadata, and better robustness across heterogeneous datasets. Moreover, the capabilities of compact LLMs have improved substantially in recent months despite their considerably smaller model sizes and lower computational requirements. This trend suggests that inference latency is likely to continue decreasing while maintaining competitive predictive performance, further improving the practicality of local LLM-based solutions.

Table 6: Latency and cost by model, baseline, and in-context learning strategy. Latency denotes the average per-prompt inference time for locally deployed models and baselines. GPT-5.3 was evaluated using API batch processing to reduce costs, preventing comparable per-prompt latency measurement. Reported costs are the API charges incurred for each experimental setting between December 2025 and March 2026. GPT-5.3 is a commercial API model, whereas Qwen and AutoGluon incurred no API costs.
<table><tr><td>Fee type</td><td>Model</td><td>In-context learning</td><td>Avg. latency per prompt</td><td>Cost per prompt / to- tal cost</td></tr><tr><td>Paid API</td><td>GPT-5.3</td><td>Zero-shot</td><td>一</td><td>$0.003 / $1.13</td></tr><tr><td></td><td></td><td>Few-shot</td><td></td><td>$0.009 / $3.38</td></tr><tr><td>Free Free</td><td>AutoGluon NaiveAutoML</td><td></td><td>0.0004 s</td><td></td></tr><tr><td>Free</td><td>H2O</td><td></td><td>0.01 s 0.009 s</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Free</td><td rowspan="2">Qwen3-14B</td><td>Zero-shot</td><td>6.03 s</td><td></td></tr><tr><td>Few-shot</td><td>17.02 s</td><td></td></tr><tr><td>Free</td><td>Qwen2.5-14B-Instruct</td><td>Zero-shot Few-shot</td><td>1.02 s 7.04 s</td><td></td></tr><tr><td rowspan="2">Free</td><td rowspan="2">Qwen3-4B-Instruct-2507- FP8</td><td>Zero-shot</td><td>8.39 s</td><td></td></tr><tr><td>Few-shot</td><td>8.45 s</td><td></td></tr><tr><td></td><td>Qwen3-4B-Thinking-2507- Zero-shot</td><td></td><td>83.09 s</td><td></td></tr><tr><td>Free</td><td>FP8</td><td>Few-shot</td><td>417.13 s</td><td></td></tr></table>

## A.5 Token Statistics

The token usage statistics in Table 7 show the prompt lengths for the zero-shot and few-shot settings across all datasets. In the zero-shot setting, prompts contain 2,502 tokens on average, with a minimum of 537 and a maximum of 15,585 tokens. In the few-shot setting, the average prompt length increases to 7,400 tokens due to the additional in-context examples included in the system prompt. The user prompt remains unchanged between both settings, with an average length of 2,252 tokens.

## A.6 Loading Pipeline

The collected datasets were provided in heterogeneous file formats and storage structures, including CSV, XLS, ARFF, Arrow, and JSON files. This heterogeneity was especially pronounced for time series datasets, where data may be stored as single long sequences, collections of separate sequences, nested structures, or archive-specific formats.

Table 7: Token usage statistics per prompt across all datasets.
<table><tr><td rowspan="2">Method</td><td colspan="3">Total Tokens</td><td colspan="3">System Prompt Tokens</td><td colspan="3">User Prompt Tokens</td></tr><tr><td>Min.</td><td>Max.</td><td>Avg.</td><td>Min.</td><td>Max.</td><td>Avg.</td><td>Min.</td><td>Max.</td><td>Avg.</td></tr><tr><td>Zero-shot</td><td>537</td><td>15585</td><td>2502</td><td>238</td><td>238</td><td>238</td><td>287</td><td>15335</td><td>2252</td></tr><tr><td>Few-shot</td><td>5435</td><td>20483</td><td>7400</td><td>5136</td><td>5136</td><td>5136</td><td>287</td><td>15335</td><td>2252</td></tr></table>

To handle these diferences, we implemented a loading pipeline that first identifies the dataset format and then applies a corresponding parser. The goal of the pipeline is not to convert all datasets into a single standardized schema, but to load each dataset into a processable representation while preserving its original structure as far as possible. This design choice reflects the intended benchmark setting: models should be evaluated on heterogeneous real-world dataset representations rather than on heavily homogenized inputs.

During loading, the pipeline also links each dataset to its metadata entry, including the target variable and task label. The explicitly specified target variable is retained in all downstream prompt representations, since the current benchmark focuses on identifying the data domain and downstream task given a known prediction target.

![](images/eadae0327286c5164807126811b241243082d1693d935109e1b9ffa215d43908.jpg)  
Figure 6: High-level overview of a representative dataset in the collection.

## B Prompt Templates

System Prompt Used for Structured Data Classification   
You are a dataset classifier for structured data.   
Your task is to identify:   
1. The data domain (Tabular or Time Series)   
2. The prediction task associated with the target variable.   
### Task   
#### Step 1: Identify the Data Domain   
Classify the dataset as one of the following:   
- ’Tabular’ --- independent rows with no intrinsic temporal ordering.

```markdown
- ’Time_Series’ --- observations indexed or ordered by time, sequence, or temporal dependency.
Use dataset structure, column semantics, and context to determine the domain.
#### Step 2: Identify the Prediction Task
For Tabular data, classify the prediction task as one of:
- ’binary’
- ’multiclass’
- ’regression’
For Time Series data, classify the prediction task as one of:
- ’binary’
- ’multiclass’
- ’regression’
Rules for task identification:
- If the target variable is continuous -> ’regression’.
- If the target variable is categorical:
- Exactly 2 unique values -> ’binary’.
- More than 2 unique values -> ’multiclass’.
- Important: Integer-valued targets may be categorical or continuous.
Decide based on semantic meaning and dataset context, not datatype alone.
### Input Format
You receive:
- A DFLoader-serialized excerpt of the dataset.
- A target specification in the following form:
- "Target: column, <name>"
### Output Format
Respond only with:
(’<Task Domain>’, ’<Sub Problem Task>’)
Where:
- <Task Domain> is exactly one of:
- ’Tabular
- ’Time_Series’
<Sub Problem Task> is exactly one of:
- ’binary’
- ’multiclass’
- ’regression’
Do not include explanations.
Do not include additional text.
```

## Example User Prompt

Dataset description:   
Author: BUPA Medical Research Ltd.; Donor: Richard S. Forsyth   
Source: UCI Liver Disorders dataset, 5/15/1990   
BUPA liver disorders:   
The first five variables are blood tests that may indicate liver disorders   
related to excessive alcohol consumption. Each row represents one male

individual.

## Important note:

The seventh field, selector, is not a dependent variable. It was created by BUPA researchers as a train/test selector and is not suitable as a classification target. In this prompt, the sixth field, drinks, is used as the target variable.

## Attribute information:

1. mcv: mean corpuscular volume

2. alkphos: alkaline phosphotase

3. sgpt: alanine aminotransferase

4. sgot: aspartate aminotransferase

5. gammagt: gamma-glutamyl transpeptidase

6. drinks: half-pint equivalents of alcoholic beverages drunk per day

7. selector: train/test split field

## Dataset:

pd.DataFrame({ ’mcv’: [98, 88, 88, 92, 90, 89, 82, 90, 86, 96, 90, 87, 96, 91, 95, 94, 87, 98, 94, 83, 88, 82, 85, 91, 98], ’alkphos’: [55, 62, 67, 54, 60, 52, 62, 64, 77, 67, 80, 90, 72, 55, 78, 56, 57, 74, 75, 68, 47, 72, 58, 54, 50], ’sgpt’: [13, 20, 21, 22, 25, 13, 17, 61, 25, 29, 19, 43, 28, 9, 27, 30, 30, 148, 20, 17, 35, 31, 83, 25, 27], ’sgot’: [17, 17, 11, 20, 19, 24, 17, 32, 19, 20, 14, 28, 19, 25, 25, 18, 30, 75, 25, 20, 26, 20, 49, 22, 25], ’gammagt’: [17, 9, 11, 7, 5, 15, 15, 13, 18, 11, 42, 156, 30, 16, 30, 27, 22, 159, 38, 71, 33, 84, 51, 35, 53], ’drinks’: [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 3.0, 3.0, 3.0, 4.0, 4.0],

})

Target: drinks

Target-specific statistics:

\- Total number of values: 345

\- Number of unique values: 16

\- Mean value: 3.455

\- Standard deviation: 3.333

\- Minimum value: 0.000

\- Maximum value: 20.000

## C Scaling Behavior

The relationship between model size and performance is further illustrated in Figure 7. The results show a clear positive correlation between the number of model parameters and the achieved F1 macro score. Larger models, such as GPT-4.1 and GPT-5.3, consistently outperform smaller models, while mid-sized models such as the 14B Qwen variants achieve intermediate performance. In contrast, smaller models such as the 4B variants show a noticeable decline in performance, particularly in complex cross-domain settings. This trend suggests that larger models are better able to capture both semantic and structural aspects of the dataset, enabling more accurate task identification. As model size increases, the ability to distinguish between subtle diferences in data representation, such as tabular versus time series structure, improves significantly. Conversely, smaller models exhibit limited capacity to resolve these ambiguities, leading to more frequent misclassifications

Scaling Behavior of LLM Performance with Respect to Model Parameters  
![](images/01b73006f2f54d9acbba572918eeb657b2e6a8f21cb3c3e8a1f0077f4aef21b9.jpg)  
Figure 7: Scaling behavior of model performance in Dataset Phase 2.