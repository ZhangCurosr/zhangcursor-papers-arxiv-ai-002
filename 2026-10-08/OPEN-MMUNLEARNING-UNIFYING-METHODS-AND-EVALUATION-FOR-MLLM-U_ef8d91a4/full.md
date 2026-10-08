# OPEN-MMUNLEARNING: UNIFYING METHODS AND EVALUATION FOR MLLM UNLEARNING

Junkai Chen<sup>1∗</sup>, Yuhao He<sup>1∗</sup>, Qianshan Wei<sup>1∗†</sup>, Junxiang You<sup>1,2</sup>, Jingwen Shao<sup>3</sup>, Junkai Lin<sup>4</sup> Zhongkai Yue<sup>1</sup>, Xiaotian Ye<sup>5</sup>, Zhengbo Jiao<sup>6</sup>, Jiali Cheng<sup>7</sup>, Zhijie Deng<sup>8</sup>, Kening Zheng<sup>9</sup> Ruiqi Liu<sup>1</sup>, Hadi Amiri<sup>7</sup>, Yi Yu<sup>10</sup>, Zhenan Sun<sup>1</sup>, Qi Li<sup>1</sup>, Ka-Ho Chow<sup>11</sup>, Sijia Liu<sup>12</sup> Liang Wang<sup>1</sup>, Jiaqi Li<sup>11‡</sup>, Shu Wu<sup>1‡</sup>

<sup>1</sup>Institute of Automation, Chinese Academy of Sciences <sup>2</sup>University of the Chinese Academy of Sciences <sup>3</sup>ByteDance <sup>4</sup>The Chinese University of Hong Kong <sup>5</sup>University of Cambridge <sup>6</sup>Shanghai University of Finance and Economics <sup>7</sup>University of Massachusetts at Lowell <sup>8</sup>The Hong Kong University of Science and Technology <sup>9</sup>University of Illinois at Chicago <sup>10</sup>Jilin University <sup>11</sup>The University of Hong Kong <sup>12</sup>Michigan State University

chenjunkai2026@ia.ac.cn, shu.wu@nlpr.ia.ac.cn

## ABSTRACT

As multimodal large language models (MLLMs) become more capable and widely deployed, concerns about privacy and safety have become increasingly pressing. Machine unlearning offers one approach to addressing these concerns by removing designated information from trained models while preserving unrelated capabilities. However, fragmented implementations and evaluation protocols, incomplete robustness testing, and limited understanding of metric reliability make progress in MLLM unlearning difficult to assess systematically. We introduce OPEN-MMUNLEARNING, an open-source, extensible framework that integrates target-model preparation, multimodal data processing, unlearning, and evaluation through shared interfaces and structured configurations. The framework supports five benchmarks spanning privacy, safety, and copyright, eight MLLMs from four model families, and twelve unlearning methods. Its evaluation suite jointly assesses forgetting effectiveness, retained utility, and robustness to model interventions, adversarial inputs, and membership inference attacks. Using a common evaluation protocol, we compare ten representative unlearning methods. In this comparison, GD and MIP-Editor tie for the highest overall score: GD achieves the highest Forget Quality, while MIP-Editor preserves more Model Utility. We further introduce a metric meta-evaluation protocol that tests faithfulness using models with controlled exposure to target knowledge and robustness under quantization and relearning. Among the thirteen evaluated metrics, BLEU achieves the highest aggregate reliability score. KS-Test attains the highest faithfulness AUC but performs less well on robustness. Together, the framework and these findings support reproducible comparison of MLLM unlearning methods and systematic assessment of evaluation reliability.

## 1 INTRODUCTION

Recent advances in Multimodal Large Language Models (MLLMs) (Liu et al., 2023; Bai et al., 2025) have substantially enhanced their multimodal perception and reasoning capabilities, enabling strong performance across image captioning, visual question answering and document understanding. However, these capabilities rely on vast amounts of multimodal data collected from the web and other sources for pre-training and instruction tuning. Such data may contain private (Liu et al., 2025a; Dontsov et al., 2025), harmful (Chen et al., 2025), or copyrighted content (Kwon et al.,

2026), raising significant ethical and legal concerns (EU, 2016; Pardau, 2018). Given the billions of parameters in modern MLLMs and the substantial data and computational resources required to train them, retraining a model from scratch for each deletion request is prohibitively costly and operationally impractical. Machine unlearning (Bourtoule et al., 2021) therefore offers a more efficient alternative: it aims to remove the influence of specific data or knowledge while preserving the model’s general performance. However, unlearning is especially challenging for MLLMs because the same concept may be redundantly encoded in visual features, language representations, and their cross-modal interactions, allowing residual information to leak through any modality. 2027

Against this backdrop, MLLM unlearning research has expanded along two fronts: benchmark development and method design. Benchmarks now address privacy (Liu et al., 2025a; Dontsov et al., 2025; Ma et al., 2025), safety (Chen et al., 2025), and copyright (Kwon et al., 2026), while proposed methods span optimization-based (Chen et al., 2025; Li et al., 2024a), localization-based (Liu et al., 2025b; Huo et al., 2025), arithmetic-based (Kim et al., 2026; Zeng et al., 2025), module-based (Garg et al., 2026; Jin et al., 2026), and inference-time approaches (Li et al., 2025; Ding et al., 2026).

Despite this progress, the systematic assessment of MLLM unlearning remains hindered by three major challenges. First, unlearning methods remain tied to specific benchmarks and model backbones. Most are developed and evaluated on a limited set of benchmarks and MLLMs (Liu et al., 2025b; Huo et al., 2025), while differences in model initialization, preprocessing, trainable components, and hyperparameter selection complicate faithful reproduction and controlled comparison across settings. Second, evaluation protocols remain fragmented. Benchmarks differ in their forget and retain splits, task formulations, and measures of forgetting effectiveness and retained utility (Liu et al., 2025a; Dontsov et al., 2025; Ma et al., 2025; Chen et al., 2025; Kwon et al., 2026). Robustness evaluation is less standardized still: queries across modalities (Wang et al., 2026a; Liu et al., 2026), targeted relearning (Zheng et al., 2025; Lin et al., 2026), representation probing (Patil et al., 2025), adversarial prompting (Zheng et al., 2025), visual perturbations (Zhang et al., 2025b), and membership inference (Li et al., 2024a) are typically examined in separate studies or limited combinations. Third, the reliability of evaluation metrics remains insufficiently examined. A decrease in performance on the forget set may reflect output suppression or general capability degradation rather than genuine knowledge removal, and different metrics may yield inconsistent assessments of the same model (Dorna et al., 2026; Wang et al., 2025). For MLLMs, aggregate scores may also obscure degradation in a specific modality or residual knowledge accessible through another modality (Wang et al., 2026a; Liu et al., 2026); whether metric judgments remain reliable after quantization or targeted relearning is likewise unclear (Dorna et al., 2026). Although OpenUnlearning (Dorna et al., 2026) unifies LLM unlearning with text data, MLLMs additionally require support for varied model architectures and paired images and text (Liu et al., 2023; 2025a), choices about which vision and language components to update (Huo et al., 2025), and evaluations of leakage across modalities and image rephrasing (Wang et al., 2026a; Patil et al., 2025). Together, these challenges hinder reproducible comparison of unlearning methods and reliable assessment of their outcomes.

These challenges motivate the need for a unified and extensible framework that enables end-toend research on MLLM unlearning (Sarwar et al., 2026). To this end, we introduce OPEN-MMUNLEARNING, which supports the complete unlearning lifecycle, including controlled targetmodel preparation, unlearning intervention, evaluation of forgetting and retained utility, and robustness evaluation against knowledge recovery. Its unified interfaces decouple model-specific processing from unlearning methods and benchmark evaluation, while configuration-driven experiments make preprocessing, optimization, and metric computation explicit and reproducible.

## Our main contributions are summarized as follows:

• We introduce OPEN-MMUNLEARNING, a unified, modular, and extensible framework that integrates 5 MLLM unlearning benchmarks, 8 MLLMs from 4 model families, 12 unlearning methods, and a diverse suite of evaluation metrics for forgetting effectiveness and retained utility. The framework also supports a range of robustness evaluations for unlearned models within a common pipeline covering the complete unlearning lifecycle.

• We conduct a systematic evaluation of 10 of these integrated unlearning methods using consistent model preparation, hyperparameter tuning, and evaluation protocols. Our evaluation jointly measures forgetting effectiveness, retained utility, and robustness against attacks on the model, inputs, and outputs, providing a unified basis for controlled comparison. The results reveal key tradeoffs among forgetting, utility, and robustness.

• We perform a meta-evaluation of commonly used MLLM unlearning metrics, systematically examining their faithfulness in reflecting the presence of target knowledge and their robustness under post-unlearning interventions. This analysis provides empirical evidence on whether existing metrics can reliably distinguish genuine knowledge removal from superficial suppression.

## 2 OVERVIEW OF MLLM UNLEARNING

Methods. Existing MLLM unlearning methods can be categorized by their intervention mechanisms. Optimization-based methods induce forgetting through objective-driven updates. Depending on the optimized objectives, these methods can employ general unlearning objectives Zhang et al. (2024a); Li et al. (2024b), counterfactual objectives Chen et al. (2025), knowledge distillation Wang et al. (2026b), structured objectives Li et al. (2024a), feedback-driven objectives Jia et al. (2026), or adversarial objectives Chen et al. (2026a). Localization-based methods edit only the components, neurons, or subspaces associated with the target knowledge Liu et al. (2025b); Huo et al. (2025). Arithmetic-based methods construct the unlearned model through algebraic operations on model weights or update vectors Kim et al. (2026); Zeng et al. (2025). Inference-time methods keep model parameters fixed and suppress target knowledge by intervening in activations or sparse features during decoding Li et al. (2025); Ding et al. (2026).

Benchmarks. MLLM unlearning benchmarks can be organized by their primary removal objective. Privacy is the most extensively studied setting and targets identities, profiles, sensitive attributes, and associations. These benchmarks can be further distinguished by the provenance of the target knowledge, including knowledge encoded during pretraining Li et al. (2024a); Patil et al. (2025); Guang et al. (2026) and controlled knowledge introduced through fine-tuning Dontsov et al. (2025); Liu et al. (2025a); Ma et al. (2025). Safety benchmarks target harmful multimodal knowledge jointly elicited by images and associated questions Chen et al. (2025); Xu et al. (2025), while Copyright benchmarks address protected characters, logos, and related background knowl edge Kwon et al. (2026).

Evaluations. Each benchmark combines forgetting-effectiveness metrics with utility-preservation metrics, with different metrics providing varying levels of evaluation rigor. Classification and cloze tasks commonly use Accuracy or Exact Match Dontsov et al. (2025); Liu et al. (2025a), while open-ended generation is evaluated using lexical-overlap metrics such as ROUGE-L Lin (2004) and BLEU Papineni et al. (2002). Likelihood-based metrics, including negative log-likelihood and perplexity, measure the model’s residual support for a target answer even when that answer is not directly generated Li et al. (2024a); Guang et al. (2026). Retraining-equivalence metrics instead compare the unlearned model with a retain-only reference using Truth-Ratio KS tests Dontsov et al. (2025), Jensen–Shannon divergence Ma et al. (2025), or Wasserstein distance Ma et al. (2025). Beyond these behavioral metrics, robustness evaluations examine whether target information remains recoverable through cross-modal queries Wang et al. (2026a); Liu et al. (2026), membership inference Li et al. (2024a); Ma et al. (2025), paraphrased prompting Patil et al. (2025), adversarial prompting Zheng et al. (2025), hidden-state probing Patil et al. (2025), visual perturbations Zhang et al. (2025b), targeted relearning Zheng et al. (2025), or model quantization Zhang et al. (2025c).

## 3 OPEN-MMUNLEARNING FRAMEWORK

Machine unlearning for multimodal large language models (MLLMs) is complicated by the heterogeneity of model architectures, multimodal data formats, optimization objectives, and evaluation protocols. Because existing implementations are typically tailored to a single benchmark or a specific model family, faithful reproduction and controlled comparison across unlearning methods remain difficult to achieve. This lack of a unified framework manifests across three major aspects of the MLLM unlearning pipeline:

Methods remain tied to specific benchmarks and model backbones. Multimodal unlearning methods are typically developed and evaluated on a limited set of benchmarks and MLLM backbones. For example, MANU (Liu et al., 2025b) and MMUnlearner (Huo et al., 2025) are each assessed on a small selection of benchmarks and model families. General unlearning objectives such as NPO (Zhang et al., 2024a) and RMU (Li et al., 2024b) also require adaptation when transferred to multimodal settings. Differences in model initialization, preprocessing, trainable components, and hyperparameter selection further complicate faithful reproduction. Consequently, applying a method to a new benchmark or backbone often requires reimplementing substantial parts of its experimental pipeline.

Evaluation protocols remain fragmented. Existing benchmarks differ in how they construct forget and retain splits, formulate evaluation tasks, and report forgetting effectiveness and retained utility (Liu et al., 2025a; Dontsov et al., 2025). Robustness evaluation is less standardized still: targeted relearning (Zheng et al., 2025; Lin et al., 2026), representation probing (Patil et al., 2025), adversarial prompting (Zheng et al., 2025), visual perturbations (Zhang et al., 2025b), and membership inference (Li et al., 2024a) are scattered across individual studies. Existing methods are often evaluated against only one or two types of attacks. Consequently, existing results lack a consistent basis for jointly comparing forgetting, retention, and robustness across methods.

The reliability of evaluation metrics remains insufficiently examined. A decrease in forget-set performance does not necessarily establish that target knowledge has been removed; it may instead reflect output suppression or general capability degradation. Different metrics may also yield inconsistent assessments or rankings of the same unlearned models (Dorna et al., 2026; Wang et al., 2025). Beyond such concerns about faithfulness, it remains unclear whether a metric’s assessment stays reliable when an apparently unlearned model undergoes quantization or targeted relearning (Dorna et al., 2026). These issues motivate a systematic examination of both the faithfulness and robustness of MLLM unlearning metrics.

Taken together, these limitations motivate the need for a unified and extensible framework for MLLM unlearning. To this end, we introduce OPEN-MMUNLEARNING, an end-to-end framework that integrates data preparation, target-model training, unlearning execution, evaluation of Forget Quality and Model Utility, and robustness evaluation within a coherent pipeline. By standardizing these components through shared interfaces, OPEN-MMUNLEARNING enables reproducible and directly comparable evaluations across unlearning methods, benchmarks, and MLLM backbones.

![](images/4d4dfed2ed4e7476b3cb6bf64f29427a60d2bdb09cb641edff9badcdf1311cd4.jpg)  
Figure 1: Overview of OPEN-MMUNLEARNING. The Data Worker prepares fine-tuning, evaluation, and unlearning data from benchmarks spanning privacy, safety, and copyright. The Model Loader supplies MLLM backbones, the Trainer executes fine-tuning and general and MLLMspecific unlearning methods, and the Unified Evaluator assesses Forget Quality, Model Utility, and Robustness through model-, input-, and output-side evaluations.

Table 1: Overview of existing OPEN-MMUNLEARNING components and their available variants. The modular design allows users to add new models, unlearning methods, datasets, and evaluation metrics.
<table><tr><td colspan="3">Component</td></tr><tr><td rowspan="2">Models</td><td>LLAVA-1.5 (Liu et al., 2023)</td><td>QWEN-2.5-VL (Bai et al., 2025)</td></tr><tr><td>LLAVA-1.6 (Liu et al., 2024a)</td><td>GEMMA-3 (Team et al., 2025)</td></tr><tr><td rowspan="4">Methods</td><td colspan="2">General Unlearning Methods (Maini et al., 2024; Li et al., 2024b)</td></tr><tr><td>(Zhang et al., 2024a; Dong et al., 2025)</td><td>MMUnlearner (Huo et al., 2025)</td></tr><tr><td>MANU (Liu et al., 2025b)</td><td>MIP-Editor (Li et al., 2026)</td></tr><tr><td>SMFA (Zeng et al., 2025)</td><td>VGID (Chen et al., 2026b)</td></tr><tr><td rowspan="3">Datasets</td><td>MLLMU (Liu et al., 2025a)</td><td>FIUBench (Ma et al., 2025)</td></tr><tr><td>CLEAR (Dontsov et al., 2025)</td><td>SafeEraser (Chen et al., 2025)</td></tr><tr><td>CoVUBench (Kwon et al., 2026)</td><td></td></tr><tr><td rowspan="7">Forget Utility Metrics</td><td colspan="3">Truth Ratio (Ma et al., 2025; Dontsov et al., 2025)</td></tr><tr><td colspan="3">KS-Test (Ma et al., 2025; Dontsov et al., 2025)</td></tr><tr><td colspan="3">JS Distance (Dontsov et al., 2025) Answer Probability (Ma et al., 2025)</td></tr><tr><td colspan="3">ROUGE-L (Liu et al., 2025a) ASR (Chen et al., 2025)</td></tr><tr><td colspan="3">Model Utility (Ma et al., 2025; Dontsov et al., 2025)</td></tr><tr><td colspan="3">Fluency (Kwon et al., 2026) Specificity (Kwon et al., 2026)</td></tr><tr><td colspan="3">Generality (Kwon et al., 2026) SARR (Chen et al., 2025)</td></tr><tr><td colspan="3">LM-Eval (Liu et al., 2025a; Kwon et al., 2026)</td></tr><tr><td rowspan="6">Robust</td><td colspan="3">Relearning (Hu et al., 2025; Zheng et al., 2025)</td></tr><tr><td colspan="2">Quantization (Zhang et al., 2025c)</td><td>Probing (Lynch et al., 2024)</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">MIA (Yeom et al., 2018; Carlini et al., 2021; Wang et al., 2024; Shi et al., 2024</td></tr><tr><td colspan="3">Jailbreak (Wei et al., 2023; Ma et al., 2024) FigStep (Gong et al., 2025)</td></tr><tr><td colspan="3">SUA (Zhang et al., 2025b) Image Rephrasing (Patil et al., 2025)</td></tr></table>

## 3.1 DESIGN OF OPEN-MMUNLEARNING

Open-MMUnlearning is designed for usability, reproducibility, and extensibility. It decomposes the end-to-end unlearning pipeline into independently registered modules, including model handlers, unlearning methods, benchmark evaluators, and robustness audits. Each experiment is specified through structured configuration files that select and parameterize these modules. This configuration-driven design enables experiments to be reproduced under the same settings and allows individual components to be exchanged without modifying the core execution logic.

Features. OPEN-MMUNLEARNING currently supports twelve unlearning methods, comprising seven general unlearning methods (Maini et al., 2024; Li et al., 2024b; Zhang et al., 2024a; Dong et al., 2025) and five MLLM-specific unlearning methods (Huo et al., 2025; Liu et al., 2025b; Li et al., 2026; Zeng et al., 2025; Chen et al., 2026b). It integrates eight MLLMs of different parameter scales across four model families (Liu et al., 2023; Bai et al., 2025; Liu et al., 2024a; Team et al., 2025) and five benchmarks spanning privacy, safety, and copyright (Liu et al., 2025a; Ma et al., 2025; Dontsov et al., 2025; Chen et al., 2025; Kwon et al., 2026). Its evaluation components are organized into Forget, Utility, and Robust. Forgetting and utility are assessed using task-based metrics such as Accuracy and Exact Match (Liu et al., 2025a; Ma et al., 2025), lexical metrics such as ROUGE-L and BLEU (Liu et al., 2025a), likelihood- and distribution-based metrics (Ma et al., 2025; Dontsov et al., 2025), and safety-oriented metrics such as ASR, RR, and SARR (Chen et al., 2025). Robustness evaluation covers relearning and quantization (Hu et al., 2025; Zheng et al., 2025;

Zhang et al., 2025c), probing and cross-modal leakage (Lynch et al., 2024; Wang et al., 2026a), textual and visual attacks (Zou et al., 2023; Wei et al., 2023; Ma et al., 2024; Gong et al., 2025; Patil et al., 2025; Zhang et al., 2025b), and membership inference (Yeom et al., 2018; Carlini et al., 2021; Wang et al., 2024; Shi et al., 2024; Zhang et al., 2025a). Due to space constraints, Table 1 provides a condensed overview, while Table 4 in the appendix lists the complete set of components and evaluation variants. Detailed descriptions of the unlearning methods and evaluation metrics are provided in Appendices A.3.1 and A.2, respectively.

Extensibility. OPEN-MMUNLEARNING organizes models, methods, benchmarks, evaluation metrics, and robustness attacks as independently extensible modules behind common interfaces. A new component can be integrated by implementing the corresponding interface and specifying its settings through configuration files, without modifying the core pipeline. Once integrated, models and methods can directly reuse existing benchmark evaluators and robustness audits, while metrics and attacks can be applied across compatible benchmarks and model families. This design allows the framework to continuously incorporate emerging MLLM architectures, unlearning techniques, evaluation protocols, and recovery attacks.

## 4 EVALUATION OF UNLEARNING METHODS

## 4.1 EXPERIMENT SETUP

Unlike prior multimodal unlearning studies that compare only a limited number of baselines or evaluation metrics, we establish a standardized experimental pipeline for large-scale comparison of unlearning methods. We conduct experiments on MLLMU-Bench under the 10% forget setting.

Unlearning Methods We compare ten representative unlearning methods: GA, GD, KL (Maini et al., 2024), NPO (Zhang et al., 2024a), RMU (Li et al., 2024b), and UNDIAL (Dong et al., 2025), MANU (Liu et al., 2025b), MMUnlearner (Huo et al., 2025), MIP-Editor (Li et al., 2026), SMFA (Zeng et al., 2025).

Evaluation metrics. We evaluate unlearned models in terms of Forget Quality, Model Utility, and Robustness. Forget Quality is assessed on the multimodal forget set and a held-out forget-target test set using fill-in-the-blank accuracy (Liu et al., 2025a), classification accuracy (Liu et al., 2025a), ROUGE-L (Lin, 2004), BLEU (Papineni et al., 2002), Truth Ratio (Ma et al., 2025; Dontsov et al., 2025), Answer Probability (Ma et al., 2025), KS-Test (Ma et al., 2025; Dontsov et al., 2025), and JS Distance (Dontsov et al., 2025). Model Utility is evaluated on non-target knowledge and general capabilities using the retain set and general-purpose benchmarks. Robustness is evaluated through model-side quantization (Zhang et al., 2025c), targeted relearning (Hu et al., 2025; Zheng et al., 2025), and representation probing (Lynch et al., 2024); input-side cross-modal text queries, Fig Step (Gong et al., 2025), pure-text jailbreak prompting (Zou et al., 2023; Wei et al., 2023; Ma et al., 2024), image rephrasing (Patil et al., 2025), and SUA (Zhang et al., 2025b); and output-side membership inference using LOSS (Yeom et al., 2018), ZLIB (Carlini et al., 2021), GradNorm (Wang et al., 2024), Min-K% (Shi et al., 2024), and Min-K%++ (Zhang et al., 2025a). The metrics within each evaluation dimension are aggregated separately to produce the corresponding composite scores; detailed normalization and aggregation rules are provided in Appendix A.3.3.

Hyperparameter tuning strategy. We use pre-specified, method-specific hyperparameter search spaces and evaluate at least three candidate values for each principal hyperparameter where appli cable. All methods follow the same target-model preparation and evaluation protocol. For each method, we select the checkpoint that maximizes the mean of Forget Quality and Model Utility on the evaluation splits reported in Table 2. Robustness is evaluated after checkpoint selection and does not influence tuning. The search spaces and selection procedure are detailed in Appendices A.3.2 and A.3.3, respectively.

## 4.2 RESULTS ANALYSIS

Overall results. Table 2 shows that MIP-Editor (Li et al., 2026) and GD (Maini et al., 2024) jointly achieve the highest aggregate score of 0.647, followed by UNDIAL (Dong et al., 2025) at 0.640. On average, the adapted general methods achieve a higher overall score of 0.623 versus 0.613 and a higher Robustness score of 0.778 versus 0.764. The MLLM-specific methods, in contrast, have a slight advantage in Model Utility, scoring 0.362 versus 0.354. The general methods’ higher average overall score is associated with stronger Forget Quality and Robustness, rather than better utility preservation. MIP-Editor and GD reach the same aggregate score through different balances across the three evaluation dimensions.

Table 2: Evaluation of ten representative MLLM unlearning methods in terms of Forget Quality, Model Utility, and Robustness. We also report an overall aggregate score and divide Robustness into model-, input-, and output-side evaluations. Finetune and Retain-Only serve as reference models. Higher scores are better; bold and underlined values indicate the best and second-best results among the unlearning methods, respectively.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Agg. ↑</td><td rowspan="2">Forget Quality,↑</td><td rowspan="2">Model Utility,↑</td><td colspan="4">Robustness ↑</td></tr><tr><td>Agg. ↑</td><td>Model ↑</td><td>Input ↑</td><td>Output ↑</td></tr><tr><td colspan="8">Reference Models</td></tr><tr><td>Finetune</td><td>0.622</td><td>0.687</td><td>0.421</td><td>0.759</td><td>0.835</td><td>0.631</td><td>0.812</td></tr><tr><td>Retain-Only</td><td>0.638</td><td>0.696</td><td>0.444</td><td>0.776</td><td>0.839</td><td>0.688</td><td>0.799</td></tr><tr><td colspan="8">General Unlearning Methods</td></tr><tr><td>GA (Maini et al., 2024)</td><td>0.574</td><td>0.774</td><td>0.066</td><td>0.881</td><td>0.996</td><td>0.971</td><td>0.677</td></tr><tr><td>GD (Maini et al., 2024)</td><td>0.647</td><td>0.781</td><td>0.400</td><td>0.760</td><td>0.894</td><td>0.767</td><td>0.619</td></tr><tr><td>KL (Maini et al., 2024)</td><td>0.617</td><td>0.694</td><td>0.391</td><td>0.767</td><td>0.833</td><td>0.668</td><td>0.800</td></tr><tr><td>RMU (Li et al., 2024b)</td><td>0.632</td><td>0.707</td><td>0.428</td><td>0.760</td><td>0.846</td><td>0.664</td><td>0.770</td></tr><tr><td>NPO (Zhang et al., 2024a)</td><td>0.630</td><td>0.749</td><td>0.382</td><td>0.757</td><td>0.873</td><td>0.685</td><td>0.714</td></tr><tr><td>UNDIAL (Dong et al., 2025)</td><td>0.640</td><td>0.722</td><td>0.454</td><td>0.745</td><td>0.826</td><td>0.731</td><td>0.677</td></tr><tr><td colspan="8">MLLM-specific Unlearning Methods</td></tr><tr><td>MMUnlearner (Huo et al., 2025)</td><td>0.632</td><td>0.739</td><td>0.411</td><td>0.745</td><td>0.889</td><td>0.756</td><td>0.590</td></tr><tr><td>MANU (Liu et al., 2025b)</td><td>0.627</td><td>0.684</td><td>0.439</td><td>0.758</td><td>0.839</td><td>0.658</td><td>0.777</td></tr><tr><td>MIP-Editor (Li et al., 2026)</td><td>0.647</td><td>0.717</td><td>0.451</td><td>0.773</td><td>0.828</td><td>0.676</td><td>0.814</td></tr><tr><td>SMFA (Zeng et al., 2025)</td><td>0.546</td><td>0.712</td><td>0.148</td><td>0.778</td><td>0.889</td><td>0.661</td><td>0.785</td></tr></table>

Trade-off between Forget Quality and Model Utility. The results illustrate the importance of considering forgetting and utility together. GD attains the highest Forget Quality score of 0.781, while its Model Utility is 0.400. GA (Maini et al., 2024) ranks second in Forget Quality at 0.774 and first in Robustness at 0.881, yet its Model Utility falls to 0.066, indicating severe degradation of non-target capabilities. Conversely, UNDIAL achieves the highest Model Utility of 0.454 with a lower Forget Quality score of 0.722. MIP-Editor combines near-best Model Utility of 0.451 with Forget Quality of 0.717 and Robustness of 0.773, yielding a top aggregate score of 0.647. Thus, strength on any single dimension does not guarantee the best combined outcome.

Robustness and ranking. GA (Maini et al., 2024) achieves the highest Robustness score of 0.881, driven by near-perfect Model- and Input-side scores of 0.996 and 0.971. However, its Model Utility is only 0.066, indicating severe degradation of non-target capabilities. Its apparent robustnes may therefore partly reflect a broadly impaired ability to answer, rather than selective resistance to knowledge recovery. SMFA (Zeng et al., 2025) ranks second in Robustness at 0.778, although its Input-side score is considerably lower at 0.661. MIP-Editor (Li et al., 2026) achieves the highest Output-side score of 0.814, but its Input-side score is 0.676. These contrasting profiles show why robustness must be interpreted together with Model Utility and the individual attack surfaces, rathe than through the aggregate alone.

Aggregation and ranking. Method rankings depend on the aggregation rule. Our arithmetic mean allows a strong score in one dimension to offset a weakness in another, so the overall ranking should be interpreted alongside the component scores. Detailed score normalization and aggregation rules are provided in Appendix A.3.3.

![](images/b2a58d7274df4a4fcdc54e00cbc60a11586320a7526c07672d4b3e13cf823e72.jpg)

![](images/68b2f8240cb609ebe52fbd4a5c0c657ea13c89906745d507ce3ce1ded20a6faf.jpg)

![](images/d4920e9c82f1ff8e846b15405ca7f665e34a32d4ecad52271dfea5a61116db70.jpg)  
Figure 2: Faithfulness and robustness meta-evaluation of ROUGE-L. The left and middle panels compare scores before and after targeted relearning and quantization, respectively. Blue points represent eligible unlearned checkpoints, the red star denotes the mean Retain-Only reference, and the shaded areas indicate intervention-specific unreliable regions; the dashed red line marks $y = x$ The right panel shows the positive and negative model-pool distributions and the selected threshold, with a faithfulness AUC of 0.870.

## 5 EVALUATION OF UNLEARNING METRICS

Unlearning metrics may provide misleading assessments: they can fail to reveal target knowledge that remains encoded in a model, and their judgments may change substantially under subsequent model interventions. Inspired by OpenUnlearning (Dorna et al., 2026), we audit MLLM unlearning metrics along two complementary dimensions corresponding to these failure modes: Faithfulness and Robustness. Faithfulness examines whether a metric accurately reflects the presence of target knowledge, whereas Robustness assesses whether its judgment remains reliable under benign interventions such as quantization and non-benign interventions such as targeted relearning.

Faithfulness. A faithful unlearning metric should distinguish models that contain the target knowledge from those that have never learned it. Let $m ( M ; { \mathcal { D } } _ { F } )$ denote the score produced by metric m for model M on the forget set $\mathcal { D } _ { F }$ . We normalize the direction and range of each metric to [0, 1], such that a larger value indicates stronger evidence that the target knowledge is present. Applying metric m to the positive and negative pools produces two score distributions:

$$
\begin{array} { r } {  { \boldsymbol { S } } _ { + } ^ { m } = \{ m ( M ;  { \mathcal { D } } _ { F } ) \mid M \in  { \mathcal { P } } \} , \qquad { \boldsymbol { S } } _ { - } ^ { m } = \{ m ( M ;  { \mathcal { D } } _ { F } ) \mid M \in  { \mathcal { N } } \} . } \end{array}\tag{1}
$$

We quantify faithfulness using the area under the receiver operating characteristic curve:

$$
F _ { m } = \mathrm { A U C - R O C } \left( S _ { + } ^ { m } , S _ { - } ^ { m } \right) .\tag{2}
$$

A higher $F _ { m }$ indicates that the metric more reliably separates models possessing the target knowledge from models that have not learned it. We additionally select the score threshold that maximizes classification accuracy between the two pools and use this threshold to identify apparently unlearned models for the subsequent robustness analysis.

Robustness. Following OpenUnlearning (Dorna et al., 2026), we evaluate the robustness of a metric’s initial judgment of successful unlearning using model quantization (Zhang et al., 2025c) and targeted relearning (Hu et al., 2025). For models judged to be successfully unlearned by the metric, quantization examines whether target knowledge becomes recoverable through a parameter transformation that introduces no additional target data, thereby challenging the initial judgment. Relearning compares changes in the metric when an unlearned model and a retain-only reference model are re-exposed to the same target data under a matched training budget. Substantially greater recovery in the unlearned model provides evidence that the initial assessment may have overstated the extent of forgetting. Together, these tests assess the reliability of the initial unlearning judgment under the specified interventions. The detailed definitions are provided in Appendix A.4.

Aggregation. We aggregate the quantization and relearning robustness scores using their harmonic mean. The overall score is then computed as the harmonic mean of faithfulness and robustness. Detailed definitions are provided in Appendix A.4.

## 5.1 EXPERIMENTAL SETUP

We conduct the metric meta-evaluation on MLLMU using LLaVA-1.5-7B as the base model. For faithfulness, we compare a positive model pool P trained with the target knowledge against a matched negative pool N trained without it. For robustness, we construct diverse unlearned models using different methods and hyperparameters, excluding models with more than a 20% utility drop or insufficient forgetting under the corresponding metric (Dorna et al., 2026).

## 5.2 RESULTS ANALYSIS

Table 3 highlights three key findings under our meta-evaluation protocol. (i) BLEU achieves the highest aggregate score, combining strong faithfulness with high robustness scores, while ROUGE-L ranks second overall. These results indicate favorable performance of lexical-overlap metrics within the evaluated setting, although their ability to detect target facts may depend on the lexical correspondence between generated and reference answers. (ii) Likelihood- and distribution-based metrics exhibit different faithfulness–robustness profiles. Truth Ratio achieves the highest aggregate robustness and ranks third overall, but separates the positive and negative model pools less effectively than KS-Test, JS Distance, and Probability. KS-Test achieves the highest faithfulness, while both KS-Test and JS Distance receive lower robustness scores than Truth Ratio. Probability exhibits a more balanced profile across the two dimensions. These differences show that metrics derived from answer probabilities can behave differently under the same evaluation protocol. (iii) Cloze and Classification obtain high robustness scores but relatively low faithfulness, indicating limited discrimination between the positive and negative model pools. The membership-inference metrics receive lower robustness scores under targeted relearning than under quantization. Among them, GradNorm achieves the highest relearning robustness, whereas Min-K%++ obtains the lowest overall aggregate score across all evaluated metrics.

Overall, reliable unlearning evaluation should jointly consider whether a metric distinguishes models with and without exposure to target knowledge, whether its judgment remains stable under benign quantization, and whether it appropriately reflects knowledge recovery under non-benign targeted relearning relative to a retain-only reference.

Table 3: Meta-evaluation of 13 MLLM unlearning metrics for faithfulness and robustness. Faithful. reports the ROC-AUC for distinguishing models with and without exposure to target knowledge. Quant. and Relearn. report robustness under quantization and targeted relearning, respectively. Higher scores indicate greater reliability. The best and second-best values in each column are shown in bold and underlined, respectively.
<table><tr><td rowspan="2">Metrics</td><td rowspan="2">Agg.↑</td><td rowspan="2">Faithful. ↑</td><td colspan="3">Robustness ↑</td></tr><tr><td>Agg.↑</td><td>Quant. ↑</td><td>Relearn↑</td></tr><tr><td>Cloze</td><td>0.766</td><td>0.626</td><td>0.986</td><td>0.995</td><td>0.978</td></tr><tr><td>Classification</td><td>0.740</td><td>0.604</td><td>0.954</td><td>0.978</td><td>0.930</td></tr><tr><td>Truth Ratio</td><td>0.896</td><td>0.814</td><td>0.996</td><td>0.999</td><td>0.992</td></tr><tr><td>Probability</td><td>0.887</td><td>0.924</td><td>0.853</td><td>0.928</td><td>0.789</td></tr><tr><td>KS-Test</td><td>0.888</td><td>0.991</td><td>0.804</td><td>0.900</td><td>0.727</td></tr><tr><td>JS Distance</td><td>0.793</td><td>0.934</td><td>0.689</td><td>0.697</td><td>0.681</td></tr><tr><td>ROUGE-L</td><td>0.921</td><td>0.870</td><td>0.978</td><td>0.992</td><td>0.964</td></tr><tr><td>BLEU</td><td>0.950</td><td>0.910</td><td>0.993</td><td>0.998</td><td>0.988</td></tr><tr><td>LOSS</td><td>0.817</td><td>0.813</td><td>0.821</td><td>0.988</td><td>0.703</td></tr><tr><td>ZLIB</td><td>0.793</td><td>0.826</td><td>0.763</td><td>0.985</td><td>0.623</td></tr><tr><td>GradNorm</td><td>0.873</td><td>0.813</td><td>0.943</td><td>1.000</td><td>0.892</td></tr><tr><td>Min-K%</td><td>0.755</td><td>0.816</td><td>0.703</td><td>0.959</td><td>0.555</td></tr><tr><td>Min-K%++</td><td>0.493</td><td>0.586</td><td>0.426</td><td>0.753</td><td>0.297</td></tr></table>

## 6 CONCLUSION

MLLM unlearning research remains hindered by fragmented implementations and inconsistent evaluation protocols. We introduced OPEN-MMUNLEARNING, a unified and extensible framework that integrates twelve unlearning methods and five benchmarks into a common, reproducible pipeline for target-model preparation, multimodal data processing, unlearning, and evaluation. The framework provides diverse metrics for forgetting effectiveness and retained utility, along with a range of robustness evaluations for unlearned models. Our controlled comparison of General Unlearning Methods and MLLM-specific Unlearning Methods shows that the former achieve slightly higher mean overall and Robustness scores, whereas the latter have a slight advantage in Model Utility. Strong Forget Quality can also coexist with substantial utility loss. Our metric meta-evaluation shows that faithfulness and robustness can diverge: a metric that distinguishes models by target-knowledge exposure may still prove unreliable under quantization or targeted relearning. Together, these findings highlight the need to evaluate forgetting, retained utility, robustness, and metric reliability jointly when developing MLLM unlearning methods.

## AI USE STATEMENT

We used GPT-6 to assist with literature retrieval and with drafting, polishing, and translating parts of the manuscript. We used GPT-4o to generate question paraphrases and answer expansions for MLLMU-based faithfulness experiments, and GPT-4o to generate fact-preserving answer paraphrases and factually perturbed answers for metric evaluation. We applied automated checks for required fields and nonempty outputs, as well as the number and uniqueness of generated evaluation answers. The prompts and data construction procedures are detailed in Appendix A.4.4. The authors take responsibility for the final manuscript, data, and results.

## REFERENCES

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

Lucas Bourtoule, Varun Chandrasekaran, Christopher A Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. Machine unlearning. In 2021 IEEE symposium on security and privacy (SP), pp. 141–159. IEEE, 2021.

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In 30th USENIX security symposium (USENIX Security 21), pp. 2633–2650, 2021.

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and Eric Wong. Jailbreaking black box large language models in twenty queries. In 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pp. 23–42. IEEE, 2025.

Haokun Chen, Jianing Li, Yao Zhang, Jinhe Bi, Yan Xia, Jindong Gu, and Volker Tresp. Auvic: Adversarial unlearning of visual concepts for multi-modal large language models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pp. 30218–30225, 2026a.

Junkai Chen, Zhijie Deng, Kening Zheng, Yibo Yan, Shuliang Liu, PeiJun Wu, Peijie Jiang, Jia Liu, and Xuming Hu. Safeeraser: Enhancing safety in multimodal large language models through multimodal machine unlearning. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 14194–14224, 2025.

Junkai Chen, Yuhao He, Junxiang You, Ruiqi Liu, Chenyu Wang, and Shu Wu. Visual-noise guided in-context distillation for multimodal large language model unlearning. arXiv preprint arXiv:2606.00105, 2026b.

Chenlu Ding, Jiancan Wu, Leheng Sheng, Fan Zhang, Yancheng Yuan, Xiang Wang, and Xiangnan He. Mllmeraser: Achieving test-time unlearning in multimodal large language models through activation steering, 2026. URL https://arxiv.org/abs/2510.04217.

Yijiang River Dong, Hongzhou Lin, Mikhail Belkin, Ramon Huerta, and Ivan Vulic. UNDIAL: Self-´ distillation with adjusted logits for robust unlearning in large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 8827–8840, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979- 8-89176-189-6. URL https://aclanthology.org/2025.naacl-long.444/.

Alexey Dontsov, Dmitrii Korzh, Alexey Zhavoronkin, Boris Mikheev, Denis Bobkov, Aibek Alanov, Oleg Rogov, Ivan Oseledets, and Elena Tutubalina. Clear: Character unlearning in textual and visual modalities. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 20582–20603, 2025.

Vineeth Dorna, Anmol Mekala, Wenlong Zhao, Andrew McCallum, Zico Kolter, Zachary Lipton, and Pratyush Maini. Openunlearning: Accelerating llm unlearning via unified benchmarking of methods and metrics. Advances in Neural Information Processing Systems, 38, 2026.

GDPR EU. General data protection regulation. Official Journal of the European Union, 2016.

Chongyu Fan, Jinghan Jia, Yihua Zhang, Anil Ramakrishna, Mingyi Hong, and Sijia Liu. Towards llm unlearning resilient to relearning attacks: A sharpness-aware minimization perspective and beyond. arXiv preprint arXiv:2502.05374, 2025.

Arpit Garg, Hemanth Saratchandran, and Simon Lucey. Sineproject: Machine unlearning for stable vision language alignment, 2026. URL https://arxiv.org/abs/2511.18444.

Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, and Xiaoyun Wang. Figstep: Jailbreaking large vision-language models via typographic visual prompts. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 23951– 23959, 2025.

Jiahui Guang, Zexun Zhan, Zhenlin Xu, Cuiyun Gao, Haiyan Wang, Jing Li, Zhaoquan Gu, and Yanchun Zhang. Ppu-bench: Real world benchmark for personalized partial unlearning in vision language models. arXiv preprint arXiv:2605.08800, 2026.

Shengyuan Hu, Yiwei Fu, Steven Wu, and Virginia Smith. Jogging the memory of unlearned llms through targeted relearning attacks. In Neurips Safe Generative AI Workshop 2024, 2024.

Shengyuan Hu, Yiwei Fu, Steven Wu, and Virginia Smith. Unlearning or obfuscating? jogging the memory of unlearned llms via benign relearning. In International Conference on Learning Representations, volume 2025, pp. 8857–8888, 2025.

Jiahao Huo, Yibo Yan, Xu Zheng, Yuanhuiyi Lyu, Xin Zou, Zhihua Wei, and Xuming Hu. Mmunlearner: Reformulating multimodal machine unlearning in the era of multimodal large language models. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 7190–7206, 2025.

Kaidi Jia, Yujie Lin, Chengyi Yang, Jiayao Ma, and Jinsong Su. Object hallucination-free reinforcement unlearning for vision-language models. arXiv preprint arXiv:2605.08031, 2026.

Hyundong Jin, Dongyoon Han, and Eunwoo Kim. Which concepts to forget and how to refuse? decomposing concepts for continual unlearning in large vision-language models, 2026. URL https://arxiv.org/abs/2603.21484.

Yejin Kim, Dongjun Hwang, Sungmin Cha, and Junsuk Choe. Knowledge vector weakening: Efficient training-free unlearning for large vision-language models, 2026. URL https://arxiv. org/abs/2601.21794.

JuneHyoung Kwon, JungMin Yun, and YoungBin Kim. Erase persona, forget lore: Benchmarking multimodal copyright unlearning in large vision language models. arXiv preprint arXiv:2605.03547, 2026.

Jiaqi Li, Qianshan Wei, Chuanyi Zhang, Guilin Qi, Miaozeng Du, Yongrui Chen, Sheng Bi, and Fan Liu. Single image unlearning: Efficient machine unlearning in multimodal large language models. Advances in Neural Information Processing Systems, 37:35414–35453, 2024a.

Kunhao Li, Wenhao Li, Di Wu, Lei Yang, Jun Bai, Ju Jia, and Jason Xue. Cross-modal unlearning via influential neuron path editing in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 35589–35597, 2026.

Nathaniel Li, Alexander Pan, Anjali Gopal, Summer Yue, Daniel Berrios, Alice Gatti, Justin D Li, Ann-Kathrin Dombrowski, Shashwat Goel, Long Phan, et al. The WMDP benchmark: Measuring and reducing malicious use with unlearning. arXiv preprint arXiv:2403.03218, 2024b.

Qing Li, Jiahui Geng, Derui Zhu, Fengyu Cai, Chenyang Lyu, and Fakhri Karray. Sauce: Selective concept unlearning in vision-language models with sparse autoencoders. arXiv preprint arXiv:2503.14530, 2025.

Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pp. 74–81, 2004.

Yujie Lin, Kaidi Jia, Jiayao Ma, Chengyi Yang, and Jinsong Su. On the robustness of machine unlearning for vision-language models. arXiv preprint arXiv:2605.26992, 2026.

Chunlin Liu, Junnian Chen, Haitong Jiang, Jianyu Zhao, Yingsen Pang, Jingchen Li, Jiabiao He, Youming Lu, Jinhe Bi, and Yuntao Du. Does forgetting transfer across modalities? a real-world benchmark for cross-modal knowledge unlearning evaluation. arXiv preprint arXiv:2608.03791, 2026.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llavanext: Improved reasoning, ocr, and world knowledge, 2024a.

Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. Autodan: Generating stealthy jailbreak prompts on aligned large language models. In International Conference on Learning Representations, volume 2024, pp. 56174–56194, 2024b.

Zheyuan Liu, Guangyao Dou, Mengzhao Jia, Zhaoxuan Tan, Qingkai Zeng, Yongle Yuan, and Meng Jiang. Protecting privacy in multimodal large language models with mllmu-bench. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 4105–4135, 2025a.

Zheyuan Liu, Guangyao Dou, Xiangchi Yuan, Chunhui Zhang, Zhaoxuan Tan, and Meng Jiang. Modality-aware neuron pruning for unlearning in multimodal large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5913–5933, 2025b.

Aengus Lynch, Phillip Guo, Aidan Ewart, Stephen Casper, and Dylan Hadfield-Menell. Eight methods to evaluate robust unlearning in llms. arXiv preprint arXiv:2402.16835, 2024.

Siyuan Ma, Weidi Luo, Yu Wang, and Xiaogeng Liu. Visual-roleplay: Universal jailbreak attack on multimodal large language models via role-playing image character. arXiv preprint arXiv:2405.20773, 2024.

Yingzi Ma, Jiongxiao Wang, Fei Wang, Siyuan Ma, Jiazhao Li, Jinsheng Pan, Xiujun Li, Furong Huang, Lichao Sun, Bo Li, et al. Benchmarking vision language model unlearning via fictitious facial identity dataset. In International Conference on Learning Representations, volume 2025, pp. 87806–87827, 2025.

Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C Lipton, and J Zico Kolter. TOFU: A task of fictitious unlearning for LLMs. First Conference On Language Modeling, 2024. URL https://openreview.net/pdf?id=B41hNBoWLo.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pp. 311–318, 2002.

Stuart L Pardau. The california consumer privacy act: Towards a european-style privacy regime in the united states. J. Tech. L. & Pol’y, 23:68, 2018.

Vaidehi Patil, Yi-Lin Sung, Peter Hase, Jie Peng, Tianlong Chen, and Mohit Bansal. Unlearning sensitive information in multimodal llms: Benchmark and attack-defense evaluation. arXiv preprint arXiv:2505.01456, 2025.

Nobin Sarwar, Shubhashis Roy Dipta, Zheyuan Liu, and Vaidehi Patil. Multimodal unlearning across vision, language, video, and audio: Survey of methods, datasets, and benchmarks. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 27702–27730, 2026.

Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer. Detecting pretraining data from large language models. In International Conference on Learning Representations, volume 2024, pp. 51826–51843, 2024.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Rame, Morgane Rivi ´ ere, et al. Gemma 3 technical\` report. arXiv preprint arXiv:2503.19786, 2025.

Chengye Wang, Yuyuan Li, XiaoHua Feng, Chaochao Chen, Xiaolin Zheng, and Jianwei Yin. Umubench: Closing the modality gap in multimodal unlearning evaluation. Advances in Neural Information Processing Systems, 38, 2026a.

Jeffrey G Wang, Jason Wang, Marvin Li, and Seth Neel. Pandora’s white-box: Increased training data leakage in open llms. arXiv preprint arXiv:2402.17012, 2024.

Qizhou Wang, Bo Han, Puning Yang, Jianing Zhu, Tongliang Liu, and Masashi Sugiyama. Towards effective evaluations and comparisons for llm unlearning methods. In International Conference on Learning Representations, volume 2025, pp. 23327–23355, 2025.

Yuhang Wang, Zhenxing Niu, Haoxuan Ji, Guangyu He, Haichang Gao, and Gang Hua. Robust mllm unlearning via visual knowledge distillation, 2026b. URL https://arxiv.org/abs/ 2512.11325.

Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does llm safety training fail? Advances in neural information processing systems, 36:80079–80110, 2023.

Dunyuan Xu, Xikai Yang, Yaoqian Li, Jinpeng Li, and Pheng-Ann Heng. From learning to unlearning: Biomedical security protection in multimodal large language models. arXiv preprint arXiv:2508.04192, 2025.

Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. Privacy risk in machine learning: Analyzing the connection to overfitting. In 2018 IEEE 31st computer security foundations symposium (CSF), pp. 268–282. IEEE, 2018.

Zhen Zeng, Leijiang Gu, Zhangling Duan, Feng Li, Zenglin Shi, Cees G. M. Snoek, and Meng Wang. Towards benign memory forgetting for selective multimodal large language model unlearning, 2025. URL https://arxiv.org/abs/2511.20196.

Jingyang Zhang, Jingwei Sun, Eric Yeats, Yang Ouyang, Martin Kuo, Jianyi Zhang, Hao Yang, and Hai Li. Min-k%++: Improved baseline for pre-training data detection from large language models. In International Conference on Learning Representations, volume 2025, pp. 64845– 64862, 2025a.

Ruiqi Zhang, Licong Lin, Yu Bai, and Song Mei. Negative preference optimization: From catastrophic collapse to effective unlearning, 2024a. URL https://arxiv.org/abs/2404. 05868.

Xianren Zhang, Hui Liu, Delvin Ce Zhang, Xianfeng Tang, Qi He, Dongwon Lee, and Suhang Wang. Sua: Stealthy multimodal large language model unlearning attack. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 11241–11254, 2025b.

Zhexin Zhang, Junxiao Yang, Pei Ke, Shiyao Cui, Chujie Zheng, Hongning Wang, and Minlie Huang. Safe unlearning: A surprisingly effective and generalizable solution to defend against jailbreak attacks. arXiv preprint arXiv:2407.02855, 1(2):3, 2024b.

Zhiwei Zhang, Fali Wang, Xiaomin Li, Zongyu Wu, Xianfeng Tang, Hui Liu, Qi He, Wenpeng Yin, and Suhang Wang. Catastrophic failure of llm unlearning via quantization. In International Conference on Learning Representations, volume 2025, pp. 74925–74948, 2025c.

Hao Zheng, Zirui Pang, Zhijie Deng, Yuhan Pu, Zhaowei Zhu, Xiaobo Xia, Jiaheng Wei, et al. Offside: Benchmarking unlearning misinformation in multimodal large language models. arXiv preprint arXiv:2510.22535, 2025.

Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023.

## A APPENDIX

## A.1 SUPPORTED COMPONENTS AND VARIANTS

## A.2 DETAILS OF METRICS

This section specifies the attack procedures used for robustness evaluation. Let $\mathcal { M } _ { u } = \mathcal { M } _ { \theta _ { u } }$ denote the unlearned MLLM and $\mathcal { M } _ { 0 } = \mathcal { M } _ { \theta _ { 0 } }$ its pre-unlearning copy. Unless stated otherwise, attacks are applied after unlearning with $\theta _ { u }$ frozen. We distinguish four audit classes: model-side recovery attacks, which modify or train parameters of the post-unlearning model; input-side recovery attacks, which modify the query while keeping $\theta _ { u }$ fixed; representation-side audits, which inspect frozen hidden representations using external probes; and score-side membership attacks, which require teacher-forced likelihoods or logits. For an answerability metric $A _ { f }$ on a held-out forget split, behavioral recovery is

$$
R _ { A } ( { \mathcal { M } } _ { a } ) = A _ { f } ( { \mathcal { M } } _ { a } ) - A _ { f } ( { \mathcal { M } } _ { u } ) ,\tag{3}
$$

where $\mathcal { M } _ { a }$ denotes either an attacked model or the frozen unlearned model evaluated on attacked inputs. This recovery metric applies only to behavioral recovery attacks; representation-side and membership audits use their own task-specific metrics defined below. We report the corresponding change on a matched retain/control split alongside $R _ { A } .$ , so that generic degradation is not mistaken for recovery. Membership attacks use separately defined score orientations and member/nonmember controls rather than an unsigned “increase in membership score.” All attack-generation, calibration, and evaluation splits are disjoint unless explicitly stated otherwise.

## A.2.1 MODEL-SIDE ATTACKS.

Quantization (Zhang et al., 2025c). After unlearning, we quantize the model to 4-bit weights, $\mathcal { M } _ { u , 4 } = \mathcal { Q } _ { 4 } ( \mathcal { M } _ { u } )$ . Quantization can alter small unlearning-induced parameter changes and make residual target information accessible again. We use quantization as a target-data-free stress test, with target-related examples excluded from any calibration data. Recovery under this intervention provides evidence that a prior judgment of successful unlearning may have overstated the extent of forgetting. Because RTN, GPTQ, and AWQ impose different grids and calibration assumptions, we treat them as distinct attack configurations and record the quantizer, group size, calibration data, and compute dtype. For each configuration, we report $R _ { A } ( \mathcal { M } _ { u , 4 } )$ together with the retain-set change. We interpret increased forget-set answerability relative to the full-precision unlearned model as evidence of target-specific recovery only when it is not accompanied by a commensurate nonspecific change on the matched control split.

Relearning (Hu et al., 2024; Fan et al., 2025). Partition target-related data into an auxiliary attack split $\mathcal { D } _ { f } ^ { \mathrm { a u x } }$ and a recovery-test split $\mathcal { D } _ { f } ^ { \mathrm { t e s t } }$ , with $\mathcal { D } _ { f } ^ { \mathrm { a u x } } \cap \mathcal { D } _ { f } ^ { \mathrm { t e s t } } = \emptyset$ . The auxiliary split contains information correlated with the forgotten target but never the evaluated question–answer pairs. Under an explicit budget B (number of examples and optimizer steps, learning rate, trainable parameters, and, for LoRA, rank), the attacker computes

$$
\delta _ { B } ^ { * } = \arg \operatorname* { m i n } _ { \delta \in \Delta ( B ) } \mathcal { L } _ { \mathrm { r e l e a r n } } ( \theta _ { u } + \delta ; \mathcal { D } _ { f } ^ { \mathrm { a u x } } )\tag{4}
$$

Table 4: Overview of existing OPEN-MMUNLEARNING components and their available feature variants. The design is easily extensible, allowing users to seamlessly contribute new features.
<table><tr><td colspan="2">Component</td></tr><tr><td rowspan="2">Models</td><td>LLAVA-1.5 (Liu et al., 2023)</td><td>QwEN-2.5-VL (Bai et al., 2025)</td></tr><tr><td>LLAVA-1.6 (Liu et al., 2024a)</td><td>GEMMA-3 (Team et al., 2025)</td></tr><tr><td rowspan="4">Methods</td><td>GradAscent GradDiff KL (Maini et al., 2024)</td><td>IdkNLL (Maini et al., 2024)</td></tr><tr><td>RMU (Li et al., 2024b)</td><td>NPO (Zhang et al., 2024a) UNDIAL (Dong et al., 2025)</td></tr><tr><td>MMUnlearner (Huo et al., 2025)</td><td>MANU (Liu et al., 2025b)</td></tr><tr><td>MIP-Editor (Li et al., 2026)</td><td>SMFA (Zeng et al., 2025) VGID (Chen et al., 2026b)</td></tr><tr><td rowspan="3">Datasets</td><td>MLLMU (Liu et al., 2025a) CLEAR (Dontsov et al., 2025)</td><td>FIUBench (Ma et al., 2025) SafeEraser (Chen et al., 2025)</td></tr><tr><td></td><td></td></tr><tr><td>CoVUBench (Kwon et al., 2026)</td><td></td></tr><tr><td rowspan="8">Forget</td><td></td><td>Truth Ratio (Ma et al., 2025; Dontsov et al., 2025)</td></tr><tr><td>Exact Match / APE (Ma et al., 2025)</td><td>Answer Probability (Ma et al., 2025)</td></tr><tr><td>Fill-in-the-Blank Accuracy (Liu et al., 2025a)</td><td></td></tr><tr><td>Classification Accuracy (Liu et al., 2025a)</td><td>ROUGE-1/2/L (Liu et al., 2025a)</td></tr><tr><td>BLEU (Liu et al., 2025a) KS-Test (Ma et al., 2025; Dontsov et al., 2025)</td><td></td></tr><tr><td>JS Distance (Dontsov et al., 2025)</td><td>Forget Quality (Dontsov et al., 2025)</td></tr><tr><td>Keyword Recall (Kwon et al., 2026)</td><td>Semantic Dissimilarity (Kwon et al., 2026)</td></tr><tr><td>Efficacy (Kwon et al., 2026) Divergence (Kwon et al., 2026)</td><td></td></tr><tr><td rowspan="8">Metrics Utility</td><td></td><td></td></tr><tr><td>Refusal Rate (RR) (Chen et al., 2025)</td><td>Attack Success Rate (ASR) (Chen et al., 2025)</td></tr><tr><td></td><td></td></tr><tr><td>GPT-Eval (Ma et al., 2025; Chen et al., 2025)</td><td>Model Utility (Ma et al., 2025; Dontsov et al., 2025)</td></tr><tr><td>Specificity (Kwon et al., 2026)</td><td>Fluency (Kwon et al., 2026)</td></tr><tr><td>LM-Eval (Liu et al., 2025a; Kwon et al., 2026)</td><td>Generality (Kwon et al., 2026)</td></tr><tr><td>Relearning (Hu et al., 2025; Zheng et al., 2025)</td><td>SARR (Chen et al., 2025)</td></tr><tr><td>Probing (Lynch et al., 2024)</td><td>Quantization (Zhang et al., 2025c)</td></tr><tr><td rowspan="7">Robust</td><td></td><td>Cross-Modal Leakage (Wang et al., 2026a)</td></tr><tr><td></td><td></td></tr><tr><td>FigStep (Gong et al., 2025)</td><td>Pure-Text Jailbreak (Zou et al., 2023; Wei et al., 2023; Ma et al., 2024) Image Rephrasing (Patil et al., 2025)</td></tr><tr><td>SUA (Zhang et al., 2025b)</td><td>LOSS (Yeom et al., 2018) ZLIB (Carlini et al., 2021)</td></tr><tr><td></td><td></td></tr><tr><td>GradNorm (Wang et al., 2024)</td><td>Min-K% (Shi et al., 2024)</td></tr><tr><td>Min-K%++ (Zhang et al., 2025a)</td><td></td></tr></table>

and evaluates $R _ { A } ( \mathcal { M } _ { \theta _ { u } + \delta _ { B } ^ { * } } )$ on $\mathcal { D } _ { f } ^ { \mathrm { t e s t } }$ . We report recovery as a function of the budget and compare it with identically budgeted learning from a retain-only reference model. Systematically faster recovery from $\mathcal { M } _ { u }$ than from the identically budgeted retain-only reference provides evidence consistent with residual target information; final accuracy after unconstrained retraining is not sufficient evidence.

Probing Motivated by representation-level verification of residual knowledge, we freeze $\mathcal { M } _ { u }$ and extract a fixed pooling of hidden states $h _ { \ell } ( x )$ at pre-specified residual-stream layers ℓ. On a probetraining split, we fit only a low-capacity linear map to predict the forgotten target identity, attribute, or answer label,

$$
W _ { \ell } ^ { * } = \arg \operatorname* { m i n } _ { W } \sum _ { ( x , z ) \in \mathcal { D } _ { \mathrm { p r o b e } } ^ { \mathrm { t r a i n } } } \mathcal { L } _ { \mathrm { p r o b e } } ( W h _ { \ell } ( x ) , z ) ,\tag{5}
$$

and evaluate it on entity-disjoint probe-test data. We match target and control examples by modality, question type, and answer frequency, and include random-label and matched retain-only baselines. Above-control test performance indicates that target-related information remains linearly decodable from the frozen representation; this is interpreted as evidence of residual information rather than behavioral recovery.

## A.2.2 INPUT-SIDE ATTACKS.

Jailbreak attack (Zhang et al., 2024b; Zou et al., 2023; Wei et al., 2023). Wrap a forget query in role-play, instruction-hierarchy, refusal-suppression, or adversarial templates while leaving $\theta _ { u }$ unchanged. We define success as recovery of the correct forgotten answer, not merely production of an affirmative response, and use the paired clean-to-attack criterion above. Fixed templates form a low-budget black-box setting. Adaptive variants are evaluated separately according to their access assumptions: GCG (Zou et al., 2023) requires gradient access for discrete suffix optimization, AutoDAN (Liu et al., 2024b) performs population-based evolutionary search over adversarial prompts, and PAIR (Chao et al., 2025) uses iterative black-box interaction with an attacker model to refine jailbreak prompts. For target-aware white-box or score-access attack settings in which the auditor knows the target answer y, a suffix s may be optimized a

$$
s ^ { * } = \arg \operatorname* { m a x } _ { s \in S ( B ) } \log p _ { \theta _ { u } } ( y \mid I , q \oplus s ) ,\tag{6}
$$

where $S ( B )$ enforces the method-specific token or query budget. Answer-prefix prefilling is treated as a separate access setting and the prefilled tokens are excluded from answer scoring. We report access level, query budget, number of restarts, and success for each variant rather than pooling heterogeneous threat models.

Stealthy Universal Attack (SUA) (Zhang et al., 2025b). Learn one image-space perturbation δ on an attack-training split $\mathcal { D } _ { f } ^ { \mathrm { a t k } }$ and evaluate transfer on a disjoint split $\mathcal { D } _ { f } ^ { \mathrm { t e s t } }$ . For images represented in $[ 0 , 1 ]$ , the white-box perturbation is optimized by projected gradient descent under an explicit pixel budget $\epsilon _ { \mathrm { p x } } / 2 5 5 $

$$
\delta ^ { * } = \arg \operatorname* { m i n } _ { \| \delta \| _ { \infty } \leq \epsilon _ { \mathrm { p x } } / 2 5 5 } \left( \mathcal { L } _ { \mathrm { a n s } } ( \delta ) + \lambda _ { d } \mathcal { L } _ { \mathrm { d e n } } ( \delta ) + \lambda _ { a } \mathcal { L } _ { \mathrm { a l i g n } } ( \delta ) \right) ,\tag{7}
$$

where $I _ { \delta } = \mathrm { c l i p } ( I + \delta , 0 , 1 )$ and

$$
\mathcal { L } _ { \mathrm { a n s } } ( \delta ) = - \mathbb { E } _ { ( I , q , y ) \sim \mathcal { D } _ { f } ^ { \mathrm { a t k } } } \log p _ { \theta _ { u } } ( y \mid I _ { \delta } , q ) ,\tag{8}
$$

$$
\mathcal { L } _ { \mathrm { d e n } } ( \delta ) = - \mathbb { E } _ { ( I , q , y ) \sim \mathcal { D } _ { \boldsymbol { f } } ^ { \mathrm { a t k } } } \log p _ { \theta _ { u } } ( y \mid D ( I _ { \delta } ) , q ) ,\tag{9}
$$

$$
\mathcal { L } _ { \mathrm { a l i g n } } ( \delta ) = \mathbb { E } _ { I \sim \mathcal { D } _ { f } ^ { \mathrm { a t k } } } \left[ 1 - \cos ( E ( I _ { \delta } ) , E ( D ( I _ { \delta } ) ) ) \right] .\tag{10}
$$

Here $D$ is a fixed denoiser and E is a fixed, named image encoder. The alignment term encourages the adversarial image and its denoised counterpart to remain close in the encoder representation space. The checkpoints and preprocessing of $\hat { D }$ and $E ,$ together with the weights $\lambda _ { d }$ and $\lambda _ { a } ,$ are reported. The attacked image is evaluated with the original text query on unseen images using both mean answerability and paired recovery, together with perceptual quality and retain/control changes. If evaluated, a grey-box variant additionally specifies its zeroth-order estimator, search dimension, and query budget. Joint image–text SUA+ is reported as a separate threat model with independent image and suffix budgets.

FigStep (Gong et al., 2025). FigStep is a black-box cross-modal jailbreak that transfers a targetseeking instruction from the textual channel to a typographic image. Following its original threestage pipeline, we first rewrite the target query into an instruction-completion form, render this instruction together with an empty numbered list as an image, and pair the image with a fixed, apparently benign text prompt asking the model to complete the list. The attack requires neither gradients nor parameter access and leaves $\theta _ { u }$ unchanged. In our unlearning audit, the typographic image encodes the request for the forgotten information rather than presenting the target answer itself. We count an attack as successful only when the generated response recovers the correct forgotten answer according to the same answerability criterion used for the clean query. We report the target recovery rate together with the corresponding clean-query baseline and retain/control performance, so that generic instruction-following changes are not mistaken for recovery.

Image Rephrase attack (Patil et al., 2025). The Image Rephrase attack tests whether forgetting is tied to the particular visual realization used during deletion. For each forgotten image–question pair $( I , q )$ , we replace I with a benchmark-provided rephrased image $I ^ { \prime }$ that preserves the answerrelevant semantics while changing its visual appearance, and evaluate the frozen unlearned model on $( I ^ { \prime } , q )$ . Following UnL $0 \mathrm { K \mathrm { \bar { - } V \bar { Q } A } } .$ , rephrased images are grouped by difficulty according to their semantic proximity to the original image, and results are reported separately for each available difficulty level rather than silently pooling them. The rephrased image contains no explicit target answer, and attack-generation data are kept disjoint from the examples used to perform unlearning. An attack succeeds when the model recovers the forgotten answer from $( I ^ { \prime } , q )$ under the same answerability rule used for the original pair. We report both attack success rate and paired recovery relative to the clean post-unlearning response, together with performance on matched retain/control examples.

## A.2.3 OUTPUT-SIDE ATTACKS.

These attacks do not operate solely on generated outputs: LOSS and ZLIB require sequencelevel teacher-forced target log-likelihoods; Min-K% additionally requires token-level target logprobabilities; and Min-K%++ requires the full next-token vocabulary distribution at every evaluated response position. We therefore report them under their respective likelihood- or logit-access threat models. For a multimodal context $x = ( I , q )$ and response $y ,$ let $\tau ( x , y )$ contain the $L = | \mathcal { T } ( x , y ) |$ evaluated response-token positions after excluding image, prompt, padding, and designated EOS tokens, and define

$$
\bar { \ell } _ { u } ( x , y ) = - \frac { 1 } { L } \sum _ { t \in \mathcal { T } ( x , y ) } \log p _ { \theta _ { u } } ( y _ { t } \mid x , y _ { < t } )\tag{11}
$$

be the mean response-token loss, and let $\mathrm { P P L } _ { u } ( x , y ) = \exp ( \bar { \ell } _ { u } ( x , y ) )$ . Examples with $L = 0$ are invalid and excluded with their count reported. Following OpenUnlearning (Dorna et al., 2026), each attack produces a scalar membership-evidence score. A complete post-unlearning MIA evaluation compares forgotten original-training members $\mathcal { D } _ { \mathrm { m e m } }$ with a disjoint holdout set $\mathcal { D } _ { \mathrm { n o n } }$ matched by modality, task, answer length, and target frequency; the retain set is not used as the non-member set when it also participated in training. Any decision threshold is selected only on a separate calibration split. We report threshold-free ROC-AUC and TPR at a pre-specified low FPR, with higher scores oriented to mean stronger membership evidence, and compare both quantities with a retain-only retrained reference under the same protocol. Successful deletion is indicated by attack performance approaching that reference, not by an uncalibrated change in mean score. Raw forget-set scores without non-member controls are explicitly labeled membership proxies rather than attack accuracy.

LOSS attack (Yeom et al., 2018). Use the negative mean token loss

$$
S _ { \mathrm { L O S S } } ( x , y ) = - \bar { \ell } _ { u } ( x , y )\tag{12}
$$

as the membership-evidence score, so that larger values indicate greater membership evidence. Mean rather than summed loss reduces direct answer-length confounding. Under a calibrated threshold $\tau ,$ a sample is classified as a member when $S _ { \mathrm { L O S S } } ( x , y ) \ge \tau$

ZLIB-normalized loss attack (Carlini et al., 2021). Let $C _ { \mathrm { Z L I B } } ( y )$ be the compressed byte length of a fixed UTF-8 serialization of the target response. Following the ZLIB-normalization principle

used in memorization and membership evaluation, we use the fixed score

$$
S _ { \mathrm { Z L I B } } ( x , y ) = - \frac { \bar { \ell } _ { u } ( x , y ) } { C _ { \mathrm { Z L I B } } ( y ) } ,\tag{13}
$$

where larger values indicate stronger membership evidence. This score definition is treated as an implementation convention and is held fixed across all compared models. In particular, we do not interpret $\bar { \ell } _ { u } / C _ { \mathrm { Z L I B } }$ as the logarithm of $\mathrm { P P L } _ { u } / C _ { \mathrm { Z L I B } }$ , since

$$
\log \left( \frac { \mathrm { P P L } _ { u } } { C _ { \mathrm { Z L I B } } } \right) = \bar { \ell } _ { u } - \log C _ { \mathrm { Z L I B } }
$$

defines a different ranking rule. The serialization and compressor settings are fixed across all examples.

Min-K% attack (Shi et al., 2024). Obtain the target next-token log-probability at every evaluated response position, set $K = \operatorname* { m a x } ( 1 , \lceil k L \rceil )$ ), and average the K smallest values:

$$
S _ { \mathrm { M i n K } } ( x , y ) = \frac { 1 } { K } \sum _ { t \in \mathbb { Z } _ { K } } \log p _ { \theta _ { u } } ( y _ { t } \mid x , y _ { < t } ) ,\tag{14}
$$

where $\mathcal { T } _ { K }$ indexes the K least-likely target tokens after excluding image, prompt, padding, and designated EOS tokens. We use $k = 2 0 \%$ unless otherwise specified. A larger (less-negative) score indicates greater membership evidence.

Min-K%++ attack (Zhang et al., 2025a). Let $p _ { t } ( v ) = p _ { \theta _ { u } } ( v ~ \vert ~ x , y _ { < t } )$ be the full next-token distribution. At each response position, Min-K%++ standardizes the observed target-token logprobability relative to the model’s full next-token distribution. Specifically, define the probabilityweighted mean and variance

$$
\mu _ { t } = \sum _ { v \in \mathcal { V } } p _ { t } ( v ) \log p _ { t } ( v ) ,\tag{15}
$$

$$
\sigma _ { t } ^ { 2 } = \sum _ { v \in \mathcal { V } } p _ { t } ( v ) \big ( \log p _ { t } ( v ) - \mu _ { t } \big ) ^ { 2 } ,\tag{16}
$$

not a uniform average over vocabulary entries. For the observed target token, define

$$
z _ { t } = \frac { \log p _ { t } ( y _ { t } ) - \mu _ { t } } { \sqrt { \sigma _ { t } ^ { 2 } + \varepsilon } } , \qquad S _ { \mathrm { M i n K + + } } ( x , y ) = \frac { 1 } { K } \sum _ { t \in \mathcal { I } _ { K } } z _ { t } ,\tag{17}
$$

where $\varepsilon > 0$ is fixed for numerical stability and $\mathcal { T } _ { K }$ indexes the K smallest standardized targettoken scores. Larger values indicate greater membership evidence. Token masking, k, ε, and score orientation are identical for member and non-member groups.

GradNorm attack (Wang et al., 2024). GradNorm is a white-box membership inference attack based on the observation that examples encountered during training tend to induce smaller perexample gradients at the trained parameters. For a candidate multimodal example $( x , y )$ , we compute the gradient of its mean response-token loss with respect to a fixed, pre-specified set of model parameters $\Theta _ { \mathrm { a t k } }$

$$
g ( x , y ) = \nabla _ { \Theta _ { \mathrm { a t k } } } \bar { \ell } _ { u } ( x , y ) , \qquad G _ { p } ( x , y ) = \| g ( x , y ) \| _ { p } .\tag{18}
$$

We use $- G _ { p } ( x , y )$ as the membership-evidence score so that larger values indicate stronger evidence of membership. The attacked parameter blocks, norm order $p ,$ token mask, and gradient precision are fixed across member and non-member examples and reported with the results. As with the other membership attacks, forgotten training examples are compared with a distribution-matched holdout set, and we report ROC-AUC and TPR at a pre-specified low FPR. Because GradNorm requires back-propagation through the model for each candidate example, it is evaluated as a white-box attack and is not combined with black-box likelihood attacks without retaining its access label.

## A.3 DETAILS OF UNLEARNING METHODS EVALUATION

## A.3.1 DETAILS OF UNLEARNING METHODS

We describe the twelve unlearning methods currently integrated into OPEN-MMUNLEARNING: seven General Unlearning Methods and five MLLM-specific Unlearning Methods. Ten of these methods are included in the controlled comparison in Table 2. Let $\mathcal { M } _ { \theta }$ be the target MLLM, $\mathcal { M } _ { \theta _ { 0 } }$ its frozen pre-unlearning copy, and $\mathcal { D } _ { f }$ and $\mathcal { D } _ { r }$ <sub>r</sub> the forget and retain sets. For multimodal context x and response $y = ( y _ { 1 } , \dots , y _ { T } )$ , define the response-token negative log-likelihood (NLL)

$$
\ell _ { \theta } ( x , y ) = - \log p _ { \theta } ( y \mid x ) = - \sum _ { t = 1 } ^ { T } \log p _ { \theta } ( y _ { t } \mid x , y _ { < t } ) .\tag{19}
$$

Then $\mathcal { L } _ { f } ( \theta ) = \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { f } } [ \ell _ { \theta } ( x , y ) ]$ , with $\mathcal { L } _ { r }$ defined analogously. The following sections detail the objectives and update rules of these methods.

Gradient Ascent (GA) (Maini et al., 2024). GA directly reduces the likelihood of the designated responses by maximizing their NLL. Under a minimization convention,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { G A } } ( \theta ) = - \mathcal { L } _ { f } ( \theta ) . } \end{array}\tag{20}
$$

It supplies a strong forgetting signal but no preservation constraint; excessive optimization can therefore cause degenerate generation and broad utility loss.

Gradient Difference (GradDiff) (Maini et $\mathbf { a l . , }$ 2024). GradDiff regularizes forget-set ascent with ordinary maximum-likelihood training on retained data:

$$
\mathcal { L } _ { \mathrm { G r a d D i f f } } ( \theta ) = - \mathcal { L } _ { f } ( \theta ) + \lambda _ { r } \mathcal { L } _ { r } ( \theta ) ,\tag{21}
$$

where $\lambda _ { r } \geq 0$ controls the forgetting–retention trade-off. Its ability to preserve non-target behavior therefore depends on whether $\mathcal { D } _ { r }$ adequately covers the capabilities to be retained.

KL Minimization (KL) (Maini et al., 2024). KL combines forget-set gradient ascent with a constraint that keeps the unlearned model’s predictions on retained data close to those of the frozen original model:

$$
\mathcal { L } _ { \mathrm { K L } } ( \theta ) = - \mathcal { L } _ { f } ( \theta ) + \alpha \operatorname { \mathbb { E } } _ { ( x , y ) \sim \mathcal { D } _ { r } } \left[ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } D _ { \mathrm { K L } } ( p _ { \theta _ { 0 } } ( \cdot \mid x , y _ { < t } ) \parallel p _ { \theta } ( \cdot \mid x , y _ { < t } ) ) \right] .\tag{22}
$$

Here, $\alpha \geq 0$ controls the preservation constraint. Unlike GradDiff, which minimizes the NLL of retained responses, KL directly limits changes to the original model’s output distribution on retained inputs.

IdkNLL (Maini et al., 2024). IdkNLL replaces the original responses in the forget set with abstention responses, such as “I do not know,” while retaining the original responses in the retain set. Let $\mathcal { D } _ { f } ^ { \mathrm { i d k } }$ denote the resulting forget-set examples. The method minimizes

$$
\mathcal { L } _ { \mathrm { I d k N L L } } ( \theta ) = \mathbb { E } _ { ( x , y _ { \mathrm { i d k } } ) \sim \mathcal { D } _ { f } ^ { \mathrm { i d k } } } \left[ \ell _ { \theta } ( x , y _ { \mathrm { i d k } } ) \right] + \mathcal { L } _ { r } ( \theta ) .\tag{23}
$$

This objective encourages abstention on forget-set queries while preserving performance on retained examples. Because abstention does not itself establish that the underlying knowledge has been removed, its effectiveness must be assessed with the same forgetting and robustness evaluations used for the other methods.

Negative Preference Optimization (NPO) (Zhang et $\mathbf { a l . , }$ 2024a). NPO regards each forget response as a dispreferred completion relative to the frozen model. Let $r _ { \theta } ( x , y ) ~ = ~ \log p _ { \theta } ( y$

$x ) - \log p _ { \theta _ { 0 } } ( y \mid x )$ . Its forget objective is

$$
\mathcal { L } _ { \mathrm { N P O } } ( \theta ) = - \frac { 2 } { \beta } \mathbb { E } _ { \mathcal { D } _ { f } } \left[ \log \sigma ( - \beta r _ { \theta } ( x , y ) ) \right] ,\tag{24}
$$

with $\beta > 0 .$ . Unlike GA, its gradient saturates once a target response becomes sufficiently less likely than under $\theta _ { 0 }$ , limiting runaway updates. Our retention-regularized variant optimizes

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { N P O + R } } ( \theta ) = \mathcal { L } _ { \mathrm { N P O } } ( \theta ) + \lambda _ { r } \mathcal { L } _ { r } ( \theta ) . } \end{array}\tag{25}
$$

Representation Misdirection for Unlearning (RMU) (Li et al., 2024b). RMU modifies hidden states rather than output probabilities. Let $h _ { \theta } ^ { ( l ) } ( x )$ denote the hidden representations at the selected layer and token positions used by RMU. It first samples a unit direction and constructs a fixed random control vector

$$
\begin{array} { c } { \mathbf { u } \sim \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } ) , } \\ { \mathbf { v } = c \mathbf { u } , } \\ { \| \mathbf { u } \| _ { 2 } = 1 , } \end{array}\tag{26}
$$

where d is the hidden-state dimension and $c > 0$ controls the target norm. The same control vector is used for the forget examples associated with the selected intervention layer. The forget-state loss pulls their representations toward this content-independent target:

$$
\begin{array} { r } { \mathcal { L } _ { f } ^ { \mathrm { R M U } } ( \theta ) = \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { f } } \left[ \left| \left| h _ { \theta } ^ { ( l ) } ( x ) - \mathbf { v } \right| \right| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{27}
$$

To preserve non-target behavior, the retain-state loss anchors the corresponding representations to the frozen pre-unlearning model:

$$
\begin{array} { r } { \mathcal { L } _ { r } ^ { \mathrm { R M U } } ( \theta ) = \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { r } } \left[ \left\| h _ { \theta } ^ { ( l ) } ( x ) - h _ { \theta _ { 0 } } ^ { ( l ) } ( x ) \right\| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{28}
$$

The complete objective is

$$
\mathcal { L } _ { \mathrm { R M U } } ( \theta ) = \mathcal { L } _ { f } ^ { \mathrm { R M U } } ( \theta ) + \lambda _ { r } \mathcal { L } _ { r } ^ { \mathrm { R M U } } ( \theta ) ,\tag{29}
$$

where $\lambda _ { r } \geq 0$ controls representation preservation. The selected layer, token positions, and trainable modules determine whether the intervention mainly affects visual, cross-modal, or linguistic representations. Unlike the adaptive target used by MIP-Editor’s RMisU below, standard RMU uses the fixed-norm control vector cu.

UNDIAL (Dong et al., 2025). UNDIAL performs self-distillation from an adjusted teacher distribution derived from the frozen model. At response position t, it lowers the frozen model’s logit for the ground-truth token $y _ { t }$

$$
\begin{array} { c } { { \widetilde { z } _ { t , v } = z _ { t , v } ^ { 0 } - \gamma \mathbb { 1 } [ v = y _ { t } ] , } } \\ { { \widetilde { p } _ { t } = \mathrm { s o f t m a x } ( \widetilde { \mathbf { z } } _ { t } ) , } } \end{array}\tag{30}
$$

and minimizes token-wise cross-entropy to this fixed target:

$$
\mathcal { L } _ { \mathrm { U N D I A L } } ( \theta ) = \mathbb { E } _ { \mathcal { D } _ { f } } \left[ \sum _ { t = 1 } ^ { T } H ( \widetilde { p } _ { t } , p _ { \theta } ( \cdot \mid x , y _ { < t } ) ) \right] .\tag{31}
$$

The adjustment strength $\gamma > 0$ controls token suppression. Because $\widetilde { p } _ { t }$ is fixed, cross-entropy and $\mathrm { K L } ( \widetilde { p } _ { t } \| p _ { \theta } )$ differ only by the constant entropy of the teacher. UNDIAL can adjust every response

token or only a selected subset (e.g., named entities), while retaining the teacher’s relative probabilities for all unadjusted tokens. The equation above is the original UNDIAL objective. Our retention-regularized implementation instead optimizes

$$
{ \mathcal { L } } _ { \mathrm { U N D I A L + R } } ( \theta ) = \lambda _ { u } { \mathcal { L } } _ { \mathrm { U N D I A L } } ( \theta ) + { \mathcal { L } } _ { r } ( \theta ) ,\tag{32}
$$

where $\lambda _ { u }$ weights adjusted-logit distillation against retain-set learning.

MMUnlearner (Huo et al., 2025). MMUnlearner targets entity-level visual knowledge while preserving non-target visual concepts and textual knowledge about the same entity. Let T contain multimodal examples of target concepts and let $\mathcal { P }$ combine (i) text-only examples of those concepts, (ii) multimodal examples of non-target concepts, and (iii) text-only examples of non-target concepts. For dataset ${ \mathcal { D } } ,$ it approximates parameter saliency with the diagonal empirical Fisher

$$
\mathbf { S } ( \theta _ { 0 } , \mathcal { L } , \mathcal { D } ) = \mathbf { F } _ { \mathrm { d i a g } } ^ { \mathcal { D } } \approx \big [ \nabla _ { \theta } \mathcal { L } _ { \mathcal { D } } ( \theta _ { 0 } ) \big ] ^ { 2 } ,\tag{33}
$$

where the square is element-wise. The resulting saliency-based update mask is

$$
\mathbf { m } = \mathbb { 1 } \left[ \frac { \mathbf { S } ( \theta _ { 0 } , \mathcal { L } , \mathcal { T } ) } { \mathbf { S } ( \theta _ { 0 } , \mathcal { L } , \mathcal { P } ) + \epsilon } \geq \beta _ { s } \right] .\tag{34}
$$

The mask is applied only to the forget component, while preserved VQA data supply an ordinary retain gradient. The resulting update direction is

$$
\begin{array} { r } { \mathbf { g } _ { \mathrm { M M U } } ( \theta ) = - \mathbf { m } \odot \nabla _ { \theta } \mathcal { L } _ { f } ( \theta ) + \nabla _ { \theta } \mathcal { L } _ { r } ( \theta ) , } \\ { \theta  \theta - \eta \mathbf { g } _ { \mathrm { M M U } } . \qquad } \end{array}\tag{35}
$$

The saliency ratio—not forget-gradient magnitude alone—ensures that parameters important to preserved visual or textual knowledge are not selected.

Modality-Aware Neuron Unlearning (MANU) (Liu et al., 2025b). MANU follows a locatethen-prune procedure over MLP neurons in both the vision and language modules. For neuron n, dataset D, and activations $z _ { m } ( d )$ under m ∈ {multi, text}, let $\mathcal { D } _ { m }$ denote the examples presented under modality condition m. We define the mean absolute activation $\begin{array} { l l } { { \bar { Z } } _ { m } } & { = } \end{array}$ $\begin{array} { r } { \dot { \left| { \mathcal { D } } _ { m } \right| ^ { - 1 } } \sum _ { d \in { \mathcal { D } } _ { m } } | z _ { m } ( d ) | } \end{array}$ , the activation count $N _ { m } = | \{ d \in \mathcal { D } _ { m } : | z _ { m } ( d ) | > \tau \} |$ , the activation dispersion $\begin{array} { r } { V _ { m } = | \tilde { \mathcal { D } } _ { m } | ^ { - 1 } \sum _ { d \in \mathcal { D } _ { m } } ( z _ { m } ( d ) - \bar { Z } _ { m } ) ^ { 2 } } \end{array}$ , and the activation energy $\begin{array} { r } { E _ { m } = \sum _ { d \in \mathcal { D } _ { m } } z _ { m } ( d ) ^ { 2 } } \end{array}$ MANU combines these quantities into four complementary modality-specific importance measures:

$$
\begin{array} { r l } & { I _ { \mathrm { { a b s } } } ( \mathcal { D } , n ) = \frac { \left| \tilde { Z } _ { \mathrm { m u l t i } } - \tilde { Z } _ { \mathrm { t e x t } } \right| } { \tilde { Z } _ { \mathrm { m u l t i } } + \tilde { Z } _ { \mathrm { t e x t } } + \epsilon } , } \\ & { I _ { \mathrm { { f r e q } } } ( \mathcal { D } , n ) = \frac { \left|  { N _ { \mathrm { m u l t i } } } -  { N _ { \mathrm { t e x t } } } \right| } {  { N _ { \mathrm { m u l t i } } } +  { N _ { \mathrm { t e x t } } } + \epsilon } , } \\ & { I _ { \mathrm { { v a r } } } ( \mathcal { D } , n ) = \sqrt { V _ { \mathrm { { m u l t i } } } + V _ { \mathrm { { t e x t } } } } , } \\ & { I _ { \mathrm { { r m s } } } ( \mathcal { D } , n ) = \sqrt { \frac { \left| E _ { \mathrm { m u l t i } } - E _ { \mathrm { { t e x t } } } \right| } { E _ { \mathrm { { m u l t i } } } + E _ { \mathrm { { t e x t } } } + \epsilon } } . } \end{array}\tag{36}
$$

Their sum, $\begin{array} { r } { I ( \mathcal D , n ) = \sum _ { k } { I _ { k } ( \mathcal D , n ) } } \end{array}$ , is used to compare the importance of each neuron on the forget and retain sets:

$$
\begin{array} { c } { { S _ { n } = \displaystyle \frac { I (  { \mathcal { D } _ { f } } , n ) } { I (  { \mathcal { D } _ { r } } , n ) + \epsilon } , } } \\ { {  { \mathcal { N } _ { \alpha } } = \{ n : S _ { n } \mathrm { ~ i s ~ i n ~ t h e ~ t o p ~ } \alpha \% \} . } } \end{array}\tag{37}
$$

The neurons with the highest forget-to-retain importance scores are then pruned. Writing $W _ { n }$ for the outgoing weights or contribution associated with neuron n, the pruning operation is

$$
W _ { n } ^ { \prime } = { \left\{ \begin{array} { l l } { \mathbf { 0 } , } & { n \in { \mathcal { N } } _ { \alpha } , } \\ { W _ { n } , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{38}
$$

MANU therefore performs unlearning through neuron localization and pruning rather than gradientbased fine-tuning. The forget-to-retain importance ratio reduces the likelihood of pruning neurons that are also important for retained knowledge, while the paired text-only and multimodal statistics capture modality-dependent activation behavior.

MIP-Editor (Li et al., 2026). MIP-Editor localizes structured information-flow paths rather than isolated neurons. For selected text-branch activations $\mathbf { w } = ( w _ { i _ { 1 } } ^ { 1 } , \dots , w _ { i _ { N } } ^ { N } )$ , define $\mathbf { \dot { \mathit { F } } } _ { T } ( \mathbf { w } ) = p ( Y \mid$ $T , w _ { i _ { 1 } } ^ { 1 } , \dots , w _ { i _ { N } } ^ { N } )$ . Following the notation of the original method, its inter-layer gradient-integration score is

$$
\mathrm { I G I } ( \mathbf { w } ) = \sum _ { n = 1 } ^ { N } \widetilde { w } _ { i _ { n } } ^ { n } \int _ { 0 } ^ { \widetilde { w } _ { i _ { n } } ^ { n } } \sum _ { l = 1 } ^ { N } \frac { \partial F _ { T } ( \alpha _ { i _ { 1 } } ^ { 1 } , \ldots , \alpha _ { i _ { N } } ^ { N } ) } { \partial w _ { i _ { l } } ^ { l } } \mathrm { d } \alpha _ { i _ { n } } ^ { n } ,\tag{39}
$$

where $\alpha _ { i _ { n } } ^ { n }$ interpolates the activation of the nth selected neuron from zero to its observed value $\widetilde { w } _ { i _ { n } } ^ { n }$ ; the implementation uses the paper’s Riemann approximation. For visual activations z and $G ( \mathbf { \bar { z } } ) = \log p ( Y \mid I , T , \mathbf { z } )$ , inter-layer Fisher integration (IFI) replaces the gradient by its square; with m Riemann points,

$$
\mathrm { I F I } ( \mathbf { z } ) \approx \sum _ { n = 1 } ^ { N } \widetilde { z } _ { i _ { n } } ^ { n } \sum _ { k = 1 } ^ { m } \sum _ { l = 1 } ^ { N } \left( \frac { \partial G ( ( k / m ) \widetilde { \mathbf { z } } ) } { \partial z _ { i _ { l } } ^ { l } } \right) ^ { 2 } .\tag{40}
$$

For a candidate path ${ \mathcal P } _ { x }$ , define its modality-specific score and selected path as

$$
\begin{array} { r } { \mathcal { S } ( \mathcal { P } _ { x } ) = \Big \{ \mathrm { I G I } ( \mathcal { P } _ { x } ) , \quad x = T , } \\ { \mathrm { I F I } ( \mathcal { P } _ { x } ) , \quad x = ( I , T ) , } \\ { \mathcal { P } _ { x } ^ { \star } = \arg \underset { \mathcal { P } _ { x } } { \operatorname* { m a x } } \mathcal { S } ( \mathcal { P } _ { x } ) . \quad } \end{array}\tag{41}
$$

Denoting the resulting paths by ${ \mathcal P } _ { T } ^ { \star } = \{ \widetilde { \bf w } _ { l } \} _ { l = 1 } ^ { L _ { t } }$ and $\mathcal { P } _ { I , T } ^ { \star } = \{ \widetilde { \mathbf { z } } _ { l } \} _ { l = 1 } ^ { L _ { v } }$ , the first editing step blocks their activations:

$$
\begin{array} { r l } { \widetilde { \mathbf { w } } _ { l }  \mathbf { 0 } } & { { } ( l = 1 , \ldots , L _ { t } ) , } \\ { \widetilde { \mathbf { z } } _ { l }  \mathbf { 0 } } & { { } ( l = 1 , \ldots , L _ { v } ) . } \end{array}\tag{42}
$$

Direct pruning may also destroy retain knowledge carried by overlapping paths. MIP-Editor therefore freezes all off-path parameters and applies Representation Misdirection Unlearning (RMisU) only to the selected neurons. For forget input $x _ { f } ,$ , it samples $\mathbf { u } \sim \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$ and constructs the adaptive target

$$
\mathbf { v } _ { f } = \lambda \| h _ { \theta _ { 0 } } ^ { ( l ) } ( x _ { f } ) \| _ { 2 } \mathbf { u } .\tag{43}
$$

The original RMisU editing loss combines forget-state misdirection and retain-state anchoring:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { M I P } } = \mathbb { E } _ { x _ { f } \sim \mathcal { D } _ { f } } \Vert h _ { \theta } ^ { ( l ) } ( x _ { f } ) - \mathbf { v } _ { f } \Vert _ { 2 } ^ { 2 } + \gamma \mathbb { E } _ { x _ { r } \sim \mathcal { D } _ { r } } \Vert h _ { \theta } ^ { ( l ) } ( x _ { r } ) - h _ { \theta _ { 0 } } ^ { ( l ) } ( x _ { r } ) \Vert _ { 2 } ^ { 2 } , } \\ & { \quad \quad \quad \quad \mathrm { w i t h ~ g r a d i e n t s ~ r e s t r i c t e d ~ t o ~ \mathcal { P } } _ { T } ^ { \star } \cup \mathcal { P } _ { I , T } ^ { \star } . } \end{array}\tag{44}
$$

Thus, activation blocking (the pruning step in the original method) initializes forgetting, whereas localized RMisU restores general behavior without freely updating the rest of the model.

Sculpted Memory Forgetting Adapter (SMFA) (Zeng et al., 2025). SMFA confines forgetting to low-rank adapter updates. Let $\mathcal { D } _ { f } ^ { \mathrm { i d \hat { k } } }$ be the forget set relabeled with refusal responses and let $\mathcal { D } _ { r } ^ { \mathrm { f e w } }$ be a few-shot retain subset. It first learns a Memory Forgetting Adapter (MFA) $\Delta W _ { f }$ by minimizing cross-entropy on their union:

$$
\Delta W _ { f } = \arg \operatorname* { m i n } _ { \Delta W } \mathcal { L } \big ( \mathcal { D } _ { f } ^ { \mathrm { i d k } } \cup \mathcal { D } _ { r } ^ { \mathrm { f e w } } , \theta _ { 0 } + \Delta W \big ) .\tag{45}
$$

It independently fine-tunes an adapter on a few-shot retain subset to obtain a retaining anchor $\Delta W _ { a }$ For entry $( i , j )$ , two indicators identify a harmful forgetting update: directional conflict and dominance in magnitude,

$$
\begin{array} { c } { { C _ { i j } = 1 [ \Delta W _ { a , i j } \Delta W _ { f , i j } < 0 ] , } } \\ { { R _ { i j } = 1 [ k \rho | \Delta W _ { a , i j } | < | \Delta W _ { f , i j } | ] , } } \end{array}\tag{46}
$$

where $k \geq 0$ is the masking strength and

$$
\rho = \frac { \| \Delta W _ { f } \| _ { F } } { \| \Delta W _ { a } \| _ { F } + \epsilon }\tag{47}
$$

compensates for the global scale difference between the two adapters. With $\mathbf { M } = \mathbf { C } \odot \mathbf { R }$ , SMFA removes entries satisfying both criteria:

$$
\begin{array} { c } { \Delta W _ { \mathrm { S M F A } } = \left[ { \bf 1 } - \left( { \bf C } \odot { \bf R } \right) \right] \odot \Delta W _ { f } , } \\ { \theta _ { u } = \theta _ { 0 } + \Delta W _ { \mathrm { S M F A } } . } \end{array}\tag{48}
$$

The conjunction is important: an oppositely directed but negligible forget update need not be removed. The backbone remains frozen, making the intervention modular and reversible while the retaining anchor protects unrelated memory and general visual understanding.

Visual-Noise Guided In-Context Distillation (VGID) (Chen et al., 2026b). VGID constructs an unlearning-oriented teacher from the frozen original model by perturbing the image and prepending an in-context unlearning instruction to each forget query. For $x _ { f } = ( I _ { f } , q _ { f } )$ , let $\mathcal { P } _ { \phi }$ denote the visual perturbation and u the textual instruction. The teacher distribution and student objective are

$$
p _ { T } ( \cdot \mid x _ { f } ) = p _ { \theta _ { 0 } } ( \cdot \mid \mathcal { P } _ { \phi } ( I _ { f } ) , u \oplus q _ { f } ) ,\tag{49}
$$

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { V G I D } } ( \theta ) = \alpha \mathbb { E } _ { x _ { f } \sim \mathcal { D } _ { f } } \left[ D _ { \mathrm { K L } } ( p _ { T } ( \cdot \mid x _ { f } ) \parallel p _ { \theta } ( \cdot \mid x _ { f } ) ) \right] } \\ & { \phantom { \mathcal { L } _ { \mathrm { V G I D } } ( \theta ) } + \beta \mathbb { E } _ { x _ { r } \sim \mathcal { D } _ { r } } \left[ D _ { \mathrm { K L } } ( p _ { \theta _ { 0 } } ( \cdot \mid x _ { r } ) \parallel p _ { \theta } ( \cdot \mid x _ { r } ) ) \right] . } \end{array}\tag{50}
$$

The student receives the original forget input, while the intervention-induced teacher supplies its distillation target; the retain-set term preserves the original model’s behavior. Thus, VGID transfers the effect of an inference-time intervention into model parameters without requiring a separately trained teacher.

## A.3.2 HYPERPARAMETER SEARCH

For the evaluated MLLMU forget setting, the reported experiments cover the following hyperparameter configurations for the ten unlearning methods in Table 2. Distinct parameter combinations are counted as configurations, whereas checkpoints saved at different training stages of the same configuration are counted separately only when reporting evaluated checkpoints.

$\mathbf { G A } { \mathrm { : } }$ : We evaluate three learning rates, $\{ 1 , 2 , 3 \} \times 1 0 ^ { - 5 }$ , yielding 3 configurations.

• GradDiff: We evaluate three learning rates, $\{ 1 , 2 , 3 \} \times 1 0 ^ { - 5 }$ , and three retain coefficients, $\lambda _ { r } \in \{ 0 . 5 , 1 , 2 \}$ , yielding $3 \times 3 = 9$ configurations.

• KL: We evaluate three learning rates, $\{ 1 , 2 , 3 \} \times 1 0 ^ { - 5 }$ , and three coefficients, $\alpha \in$ {0.5, 1, 2}, yielding $3 \times 3 = 9$ configurations.

• NPO: We evaluate three learning rates, $\{ 1 , 2 , 3 \} \times 1 0 ^ { - 5 }$ , three retain coefficients, $\lambda _ { r } \in$ $\{ 0 , 0 . 5 , 1 \}$ , and three preference temperatures, $\beta { \stackrel { . } { \in } } \{ 0 . 6 , 0 . 9 , 1 . 2 \}$ , yielding $3 \times 3 \times 3 = 2 7$ configurations.

• RMU: We evaluate three learning rates, $\{ 1 , 2 , 3 \} \times 1 0 ^ { - 5 }$ , three steering magnitudes, $c \in$ $\{ 1 , 1 0 , 1 0 0 \}$ , and three intervention layers, $l \in \{ 6 , 1 1 , 1 6 \}$ , yielding ${ \bar { 3 } } \times { \bar { 3 } } \times 3 = 2 7$ configurations.

• UNDIAL: We evaluate three learning rates, $\{ 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } \}$ , three distillation weights, $\lambda _ { u } \in \{ 1 , 2 , 5 \}$ , and three logit-adjustment coefficients, $\gamma \in \{ 3 , 1 0 , 3 0 \}$ , yielding ${ \bar { 3 } } \times 3 \times 3 = { \bar { 2 } } 7$ configurations.

• MMUnlearner: We evaluate three learning rates, $\{ 1 , 2 , 3 \} \times 1 0 ^ { - 5 }$ , and three localization thresholds, {0.5, 1, 2}, yielding $3 \times 3 = 9$ configurations.

• MANU: We evaluate three pruning parameters, {2, 5, 10}, yielding 3 configurations.

• MIP-Editor: We evaluate three learning rates, $\{ 1 0 ^ { - 4 } , 5 { \times } 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$ , three Top-k values, {3, 5, 10}, and three steering coefficients, $\{ 0 . 5 , 1 , 2 \}$ , yielding $3 \times 3 \times 3 = 2 7$ configurations.

• SMFA: We evaluate three learning rates, $\{ 5 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 2 \times 1 0 ^ { - 4 } \}$ , and three multimodal scaling coefficients, $k \in \{ 2 . 5 , 5 , \bar { 1 0 } \}$ , yielding $3 \times 3 = 9$ configurations.

In total, we trained 150 distinct hyperparameter configurations across the ten unlearning methods.

## A.3.3 METRIC AGGREGATION

This section defines the aggregation used in Table 2. Component scores are mapped to [0, 1] and direction-aligned so that larger values indicate better performance. All aggregates are computed from unrounded values; rounding is applied only for display. Valid zeros are retained, whereas an aggregate with a missing required component is reported as $\bar { \bullet } \underline { { \ : } } \psi$

Forget Quality. For a forget evaluation split $d ,$ let $a _ { d } ^ { \mathrm { f i l l } }$ and $a _ { d } ^ { \mathrm { c l s } }$ denote cloze and classification accuracy in percent, and let $r _ { d } ^ { \mathrm { R L } } \in [ 0 , 1 ]$ denote ROUGE-L. The task-based forgetting score is

$$
F _ { d } ^ { \mathrm { t a s k } } = \frac { 1 } { 3 } \left[ \left( 1 - \frac { a _ { d } ^ { \mathrm { f i l } } } { 1 0 0 } \right) + \left( 1 - \frac { a _ { d } ^ { \mathrm { c l s } } } { 1 0 0 } \right) + \left( 1 - r _ { d } ^ { \mathrm { R L } } \right) \right] .\tag{51}
$$

We first average the scores on the multimodal forget set and the held-out forget-target test set:

$$
F _ { \mathrm { m u l t i } } ^ { \mathrm { t a s k } } = \frac { F _ { \mathrm { d i r e c t - m u l t i } } ^ { \mathrm { t a s k } } + F _ { \mathrm { t e s t } } ^ { \mathrm { t a s k } } } { 2 } .\tag{52}
$$

Let $T _ { \mathrm { T R } }$ be the normalized Truth Ratio, $p _ { \mathrm { a n s } }$ the answer probability, and $D _ { \mathrm { K S } }$ and $D _ { \mathrm { J S } }$ the KS-Test statistic and JS distance against the matched Retain-Only reference. Forget Quality is

$$
\mathrm { F Q } = \frac { 1 } { 5 } \left[ F _ { \mathrm { m u l t i } } ^ { \mathrm { t a s k } } + T _ { \mathrm { T R } } + \left( 1 - p _ { \mathrm { a n s } } \right) + \left( 1 - D _ { \mathrm { K S } } \right) + \left( 1 - D _ { \mathrm { J S } } \right) \right] .\tag{53}
$$

BLEU is reported as an auxiliary generation metric but is not included in this aggregate.

Model Utility. For modality m {uni, multi} and retain split s ∈ {retain-shared, retain-celeb $= \dot { 1 } \dot { 1 } \dot { 1 } \dot { 1 }$ , we define

$$
U _ { m , s } = \frac { 1 } { 3 } \left( \frac { a _ { m , s } ^ { \mathrm { f l l } } } { 1 0 0 } + \frac { a _ { m , s } ^ { \mathrm { c l s } } } { 1 0 0 } + r _ { m , s } ^ { \mathrm { R L } } \right) .\tag{54}
$$

The retain-set component averages the four modality–split combinations:

$$
U _ { \mathrm { r e t a i n } } = { \frac { 1 } { 4 } } \sum _ { \substack { m \in \{ \mathrm { u n i , m u l t i } \} } } \sum _ { \substack { s \in \{ \mathrm { r e t a i n - s h a r e d , r e t a i n - c e l e b r i t y } \} } } U _ { m , s } .\tag{55}
$$

Let B contain MMBench, MM-Vet, POPE, ScienceQA, VizWiz, GQA, and VQAv2, and let $a _ { b }$ be the accuracy in percent on benchmark b. The general-capability component and Model Utility are

$$
U _ { \mathrm { g e n e r a l } } = \frac { 1 } { 7 } \sum _ { b \in \mathcal { B } } \frac { a _ { b } } { 1 0 0 } , \qquad \mathrm { M U } = \frac { U _ { \mathrm { r e t a i n } } + U _ { \mathrm { g e n e r a l } } } { 2 } .\tag{56}
$$

Model-side robustness. We apply the task-based forgetting function above to the multimodal evaluations after quantization, targeted relearning, and representation probing. Let the resulting scores be $Q , L , P _ { 1 6 } ,$ , and $P _ { 2 4 }$ , respectively. The two probe layers are averaged before assigning equal weight to the three intervention families:

$$
\mathrm { M o d e l } = \frac { 1 } { 3 } \left( Q + L + \frac { P _ { 1 6 } + P _ { 2 4 } } { 2 } \right) .\tag{57}
$$

Input-side robustness. Cross-modal leakage is measured by evaluating the corresponding puretext forget split without the image. Its direction-aligned score is

$$
\begin{array} { r } { I _ { \mathrm { C r o s s M o d a l } } = F _ { \mathrm { d i r e c t - u n i } } ^ { \mathrm { t a s k } } . } \end{array}\tag{58}
$$

For each other input-side attack $^ { a , }$ we convert its target-recovery or attack-success rate $\mathrm { A S R } _ { a } .$ , expressed in percent, to $I _ { a } = 1 - \mathrm { A S R } _ { a } / 1 0 0$ . Input-Side Robustness averages five components:

$$
\mathrm { I n p u t } = \frac { 1 } { 5 } \left( I _ { \mathrm { C r o s s M o d a l } } + I _ { \mathrm { F i g S t e p } } + I _ { \mathrm { J a i l b r e a k } } + I _ { \mathrm { I m a g e R e p h r a s e } } + I _ { \mathrm { S U A } } \right) .\tag{59}
$$

The image–textual jailbreak variant is excluded from this aggregate.

Output-side robustness. Output-Side Robustness uses only the multimodal evaluation of five membership-inference attacks. For attack $j ,$ its ROC-AUC is converted to a score measuring proximity to random guessing:

$$
O _ { j } = \mathrm { c l i p } \left( 1 - 2 \left| \mathrm { A U C } _ { j , \mathrm { m u l t i } } - 0 . 5 \right| , 0 , 1 \right) .\tag{60}
$$

An AUC of 0.5 therefore receives a score of 1, while an AUC of 0 or 1 receives a score of 0. We then compute

$$
\mathrm { O u t p u t } = \frac { 1 } { 5 } \left( { \cal O } _ { \mathrm { L O S S } } + { \cal O } _ { \mathrm { Z L I B } } + { \cal O } _ { \mathrm { G r a d N o r m } } + { \cal O } _ { \mathrm { M i n - K } \mathcal { Y } _ { 0 } } + { \cal O } _ { \mathrm { M i n - K } \mathcal { Y } _ { 0 } + + } \right) .\tag{61}
$$

Checkpoint selection and final aggregation. For each unlearning method, we select the checkpoint with the highest mean of Forget Quality and Model Utility:

$$
S _ { \mathrm { s e l e c t } } = { \frac { \mathrm { F Q + M U } } { 2 } } .\tag{62}
$$

The selection scores are computed on the same evaluation splits used to report Table $2 ;$ no separate validation split is used. Robustness does not enter checkpoint selection. We reserve it for post-selection stress testing so that checkpoint selection is not optimized for the particular attacks and interventions used in the robustness audit. For the selected checkpoint, the three robustness dimensions and the final method-level score receive equal weight:

$$
\mathrm { R o b u s t n e s s = \frac { M o d e l + I n p u t + O u t p u t } { 3 } , \qquad O v e r a l l = \frac { F Q + M U + R o b u s t n e s s } { 3 } . }\tag{63}
$$

Because Forget Quality and Model Utility are used both for selection and for reporting on the same splits, their reported values and the resulting Overall score are selection-conditioned and may be optimistic. We therefore interpret small differences between methods descriptively.

## A.3.4 ADDITIONAL RESULTS FOR UNLEARNING METHOD EVALUATION

To unpack the aggregate results in Table 2, the following tables report the individual measurements for Forget Quality and Model Utility, together with the model-, input-, and output-side Robustness scores. These breakdowns show how the individual scores contribute to the aggregate results.

Table 5: Detailed Forget Quality results on the forget set and held-out forget-target test set. Cloze and Cls. report accuracy (%), while Gene. reports ROUGE-L. Lower scores indicate stronger forgetting. Finetune and Retain-Only serve as reference models; bold and underlined values indicate the best and second-best results among the unlearning methods, respectively.
<table><tr><td rowspan="2">Methods</td><td colspan="3">Forget Set</td><td colspan="3">Test Set</td></tr><tr><td>Cloze ↓</td><td>Cls. ↓</td><td>Gene. ↓</td><td>Cloze ↓</td><td>Cls. ↓</td><td>Gene.↓</td></tr><tr><td colspan="7">Reference Models</td></tr><tr><td>Finetune</td><td>13.54</td><td>38.78</td><td>0.242</td><td>11.46</td><td>27.35</td><td>0.149</td></tr><tr><td>Retain-Only</td><td>12.50</td><td>36.73</td><td>0.207</td><td>13.54</td><td>40.41</td><td>0.146</td></tr><tr><td colspan="7">General Unlearning Methods</td></tr><tr><td>GA (Maini et al., 2024)</td><td>0.00</td><td>0.00</td><td>0.003</td><td>0.00</td><td>0.00</td><td>0.003</td></tr><tr><td>GD (Maini et al., 2024)</td><td>8.33</td><td>4.08</td><td>0.155</td><td>10.42</td><td>7.76</td><td>0.130</td></tr><tr><td>KL (Maini et al., 2024)</td><td>14.58</td><td>37.14</td><td>0.233</td><td>9.38</td><td>40.82</td><td>0.144</td></tr><tr><td>RMU (Li et al., 2024b)</td><td>12.50</td><td>31.43</td><td>0.219</td><td>7.29</td><td>29.39</td><td>0.149</td></tr><tr><td>NPO (Zhang et al., 2024a)</td><td>12.50</td><td>30.61</td><td>0.209</td><td>6.25</td><td>37.96</td><td>0.130</td></tr><tr><td>UNDIAL (Dong et al., 2025)</td><td>9.38</td><td>42.45</td><td>0.252</td><td>6.25</td><td>35.51</td><td>0.168</td></tr><tr><td colspan="7">MLLM-specific Unlearning Methods</td></tr><tr><td></td><td></td><td>37.55</td><td>0.186</td><td>7.29</td><td></td><td></td></tr><tr><td>MMUnlearner (Huo et al., 2025) MANU (Liu et al., 2025b)</td><td>8.33 12.50</td><td>41.22</td><td>0.402</td><td>10.42</td><td>33.06 40.41</td><td>0.124 0.248</td></tr><tr><td>MIP-Editor (Li et al., 2026)</td><td>14.58</td><td>40.82</td><td>0.217</td><td>10.42</td><td>35.51</td><td>0.152</td></tr><tr><td></td><td></td><td>31.84</td><td>0.139</td><td>9.38</td><td>34.29</td><td></td></tr><tr><td>SMFA (Zeng et al., 2025)</td><td>13.54</td><td></td><td></td><td></td><td></td><td>0.119</td></tr></table>

Table 6: Detailed Forget Quality results for normalized Truth Ratio, answer probability, KS-Test, and JS Distance. Higher Truth Ratio and lower values for the other metrics are better. Finetune and Retain-Only serve as reference models; bold and underlined values indicate the best and second-best results among the unlearning methods, respectively.
<table><tr><td>Methods</td><td>Truth Ratio ↑</td><td>Prob. ↓</td><td>KS-Test↓</td><td>JS Dist. ↓</td></tr><tr><td colspan="5">Reference Models</td></tr><tr><td>Finetune</td><td>0.570</td><td>0.499</td><td>0.140</td><td>0.281</td></tr><tr><td>Retain-Only</td><td>0.452</td><td>0.254</td><td>0.150</td><td>0.337</td></tr><tr><td colspan="5">General Unlearning Methods</td></tr><tr><td>GA (Maini et al., 2024)</td><td>0.393</td><td>0.000</td><td>0.330</td><td>0.190</td></tr><tr><td>GD (Maini et al., 2024)</td><td>0.432</td><td>0.168</td><td>0.130</td><td>0.130</td></tr><tr><td>KL (Maini et al., 2024)</td><td>0.583</td><td>0.489</td><td>0.100</td><td>0.294</td></tr><tr><td>RMU (Li et al., 2024b)</td><td>0.586</td><td>0.493</td><td>0.080</td><td>0.280</td></tr><tr><td>NPO (Zhang et al., 2024a)</td><td>0.623</td><td>0.345</td><td>0.060</td><td>0.269</td></tr><tr><td>UNDIAL (Dong et al., 2025)</td><td>0.652</td><td>0.355</td><td>0.140</td><td>0.324</td></tr><tr><td colspan="5">MLLM-specific Unlearning Methods</td></tr><tr><td>MMUnlearner (Huo et al., 2025)</td><td>0.467</td><td>0.190</td><td>0.130</td><td>0.257</td></tr><tr><td>MANU (Liu et al., 2025b)</td><td>0.583</td><td>0.469</td><td>0.130</td><td>0.281</td></tr><tr><td>MIP-Editor (Li et al., 2026)</td><td>0.617</td><td>0.489</td><td>0.060</td><td>0.252</td></tr><tr><td>SMFA (Zeng et al., 2025)</td><td></td><td>0.488</td><td>0.070</td><td></td></tr><tr><td></td><td>0.570</td><td></td><td></td><td>0.262</td></tr></table>

Table 7: Detailed Model Utility results on the multimodal Retain-Shared and Retain-Celebrity sets. Cloze and Cls. report accuracy (%), while Gene. reports ROUGE-L. Higher scores are better. Finetune and Retain-Only serve as reference models; bold and underlined values indicate the best and second-best results among the unlearning methods, respectively.
<table><tr><td rowspan="2">Methods</td><td colspan="3">Retain-Shared</td><td colspan="3">Retain-Celebrity</td></tr><tr><td>Cloze ↑</td><td>Cls. ↑</td><td>Gene. ↑</td><td>Cloze ↑</td><td>Cls. ↑</td><td>Gene. ↑</td></tr><tr><td colspan="7">Reference Models</td></tr><tr><td>Finetune</td><td>8.04</td><td>36.74</td><td>0.231</td><td>12.75</td><td>29.90</td><td>0.171</td></tr><tr><td>Retain-Only</td><td>16.52</td><td>41.47</td><td>0.261</td><td>16.34</td><td>50.91</td><td>0.169</td></tr><tr><td colspan="7">General Unlearning Methods</td></tr><tr><td>GA (Maini et al., 2024)</td><td>0.11</td><td>0.00</td><td>0.003</td><td>0.00</td><td>0.00</td><td>0.000</td></tr><tr><td>GD (Maini et al., 2024)</td><td>15.63</td><td>0.85</td><td>0.215</td><td>9.48</td><td>14.75</td><td>0.145</td></tr><tr><td>KL (Maini et al., 2024)</td><td>19.53</td><td>31.70</td><td>0.234</td><td>12.42</td><td>32.38</td><td>0.168</td></tr><tr><td>RMU (Li et al., 2024b)</td><td>17.41</td><td>27.01</td><td>0.225</td><td>13.40</td><td>25.59</td><td>0.158</td></tr><tr><td>NPO (Zhang et al., 2024a)</td><td>18.75</td><td>23.66</td><td>0.211</td><td>6.86</td><td>19.32</td><td>0.167</td></tr><tr><td>UNDIAL (Dong et al., 2025)</td><td>15.74</td><td>35.31</td><td>0.267</td><td>11.44</td><td>49.22</td><td>0.230</td></tr><tr><td colspan="7">MLLM-specific Unlearning Methods</td></tr><tr><td>MMUnlearner (Huo et al., 2025)</td><td></td><td>28.13</td><td></td><td></td><td></td><td></td></tr><tr><td>MANU (Liu et al., 2025b)</td><td>16.85 19.08</td><td>33.30</td><td>0.184 0.430</td><td>14.05 13.40</td><td>19.19 33.03</td><td>0.148 0.305</td></tr><tr><td>MIP-Editor (Li et al., 2026)</td><td>17.86</td><td>25.27</td><td>0.214</td><td>13.73</td><td>45.56</td><td>0.162</td></tr><tr><td></td><td>18.64</td><td>29.82</td><td>0.136</td><td>7.84</td><td>23.37</td><td>0.112</td></tr><tr><td>SMFA (Zeng et al., 2025)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 8: Detailed Model Utility results on seven general capability benchmarks. All scores are reported as percentages, and higher scores are better. SQA denotes ScienceQA. Finetune and Retain-Only serve as reference models; bold and underlined values indicate the best and second-best results among the unlearning methods, respectively.
<table><tr><td>Methods</td><td>MMBench ↑</td><td>MM-Vet ↑</td><td>POPE↑</td><td>SQA ↑</td><td>VizWiz ↑</td><td>GQA↑</td><td>VQAv2 ↑</td></tr><tr><td colspan="8">Reference Models</td></tr><tr><td>Finetune</td><td>80.00</td><td>16.00</td><td>82.00</td><td>56.00</td><td>2.60</td><td>52.00</td><td>79.20</td></tr><tr><td>Retain-Only</td><td>80.00</td><td>24.00</td><td>78.00</td><td>62.00</td><td>8.40</td><td>44.00</td><td>79.20</td></tr><tr><td colspan="8">General Unlearning Methods</td></tr><tr><td>GA (Maini et al., 2024)</td><td>0.00</td><td>0.00</td><td>0.00</td><td>36.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>GD (Maini et al., 2024)</td><td>76.00</td><td>16.00</td><td>82.00</td><td>56.00</td><td>5.20</td><td>56.00</td><td>81.20</td></tr><tr><td>KL (Maini et al., 2024)</td><td>78.00</td><td>16.00</td><td>82.00</td><td>56.00</td><td>0.60</td><td>34.00</td><td>53.60</td></tr><tr><td>RMU (Li et al., 2024b)</td><td>80.00</td><td>18.00</td><td>82.00</td><td>60.00</td><td>9.00</td><td>52.00</td><td>81.20</td></tr><tr><td>NPO (Zhang et al., 2024a)</td><td>72.00</td><td>20.00</td><td>82.00</td><td>56.00</td><td>0.60</td><td>38.00</td><td>63.20</td></tr><tr><td>UNDIAL (Dong et al., 2025)</td><td>80.00</td><td>20.00</td><td>88.00</td><td>60.00</td><td>11.00</td><td>56.00</td><td>83.20</td></tr><tr><td colspan="8">MLLM-specific Unlearning Methods</td></tr><tr><td>MMUnlearner (Huo et al., 2025)</td><td>78.00</td><td>22.00</td><td>78.00</td><td>58.00</td><td>8.20</td><td>54.00</td><td>81.00</td></tr><tr><td>MANU (Liu et al., 2025b)</td><td>80.00</td><td>16.00</td><td>82.00</td><td>56.00</td><td>4.60</td><td>54.00</td><td>81.20</td></tr><tr><td>MIP-Editor (Li et al., 2026)</td><td>80.00</td><td>20.00</td><td>84.00</td><td>64.00</td><td>11.00</td><td>56.00</td><td>83.20</td></tr><tr><td>SMFA (Zeng et al., 2025)</td><td>0.00</td><td>0.00</td><td>0.00</td><td>6.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr></table>

Table 9: Detailed Model-Side Robustness results for quantization, targeted relearning, and probing at layers 16 and 24. Higher scores are better. Finetune and Retain-Only serve as reference models; bold and underlined values indicate the best and second-best results among the unlearning methods, respectively.
<table><tr><td>Methods</td><td>Quant. ↑</td><td>Relearn. ↑</td><td>Probe-L16 ↑</td><td>Probe-L24 ↑</td></tr><tr><td colspan="5">Reference Models</td></tr><tr><td>Finetune</td><td>0.752</td><td>0.759</td><td>0.993</td><td>0.996</td></tr><tr><td>Retain-Only</td><td>0.768</td><td>0.760</td><td>0.984</td><td>0.993</td></tr><tr><td colspan="5">General Unlearning Methods</td></tr><tr><td>GA (Maini et al., 2024)</td><td>0.999</td><td>0.997</td><td>0.992</td><td>0.991</td></tr><tr><td>GD (Maini et al., 2024)</td><td>0.915</td><td>0.772</td><td>0.994</td><td>0.993</td></tr><tr><td>KL (Maini et al., 2024)</td><td>0.727</td><td>0.780</td><td>0.984</td><td>0.997</td></tr><tr><td>RMU (Li et al., 2024b)</td><td>0.748</td><td>0.800</td><td>0.989</td><td>0.990</td></tr><tr><td>NPO (Zhang et al., 2024a)</td><td>0.838</td><td>0.786</td><td>0.994</td><td>0.996</td></tr><tr><td>UNDIAL (Dong et al., 2025)</td><td>0.759</td><td>0.722</td><td>0.996</td><td>0.999</td></tr><tr><td colspan="5">MLLM-specific Unlearning Methods</td></tr><tr><td>MMUnlearner (Huo et al., 2025)</td><td>0.906</td><td>0.794</td><td>0.965</td><td>0.971</td></tr><tr><td>MANU (Liu et al., 2025b)</td><td>0.866</td><td>0.666</td><td>0.976</td><td>0.998</td></tr><tr><td>MIP-Editor (Li et al., 2026)</td><td>0.761</td><td>0.752</td><td>0.946</td><td>0.995</td></tr><tr><td>SMFA (Zeng et al., 2025)</td><td>0.866</td><td>0.806</td><td>0.994</td><td>0.997</td></tr></table>

Table 10: Detailed Input-Side Robustness results for cross-modal queries, FigStep, jailbreak prompts, image rephrasing, and SUA. Higher scores indicate stronger resistance to these inputs. Finetune and Retain-Only serve as reference models; bold and underlined values indicate the best and second-best results among the unlearning methods, respectively.
<table><tr><td>Methods</td><td>Cross-Modal Leakage,↑</td><td></td><td>FigStep,↑ Jailbreak,↑</td><td>Image Rephrasing,↑</td><td>SUA,↑</td></tr><tr><td colspan="6">Reference Models</td></tr><tr><td>Finetune</td><td>0.574</td><td>0.860</td><td>0.740</td><td>0.688</td><td>0.292</td></tr><tr><td>Retain-Only</td><td>0.611</td><td>0.930</td><td>0.797</td><td>0.667</td><td>0.438</td></tr><tr><td colspan="6">General Unlearning Methods</td></tr><tr><td>GA (Maini et al., 2024)</td><td>0.853</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>GD (Maini et al., 2024)</td><td>0.594</td><td>0.900</td><td>0.860</td><td>0.833</td><td>0.646</td></tr><tr><td>KL (Maini et al., 2024)</td><td>0.582</td><td>0.880</td><td>0.763</td><td>0.780</td><td>0.333</td></tr><tr><td>RMU (Li et al., 2024b)</td><td>0.596</td><td>0.890</td><td>0.767</td><td>0.777</td><td>0.292</td></tr><tr><td>NPO (Zhang et al., 2024a)</td><td>0.625</td><td>0.790</td><td>0.807</td><td>0.747</td><td>0.458</td></tr><tr><td>UNDIAL (Dong et al., 2025)</td><td>0.574</td><td>0.920</td><td>0.853</td><td>0.830</td><td>0.479</td></tr><tr><td colspan="6">MLLM-specific Unlearning Methods</td></tr><tr><td>MMUnlearner (Huo et al., 2025)</td><td>0.656</td><td>0.830</td><td>0.847</td><td>0.863</td><td>0.583</td></tr><tr><td>MANU (Liu et al., 2025b)</td><td>0.588</td><td>0.850</td><td>0.783</td><td>0.777</td><td>0.292</td></tr><tr><td>MIP-Editor (Li et al., 2026)</td><td>0.562</td><td>0.890</td><td>0.787</td><td>0.810</td><td>0.333</td></tr><tr><td>SMFA (Zeng et al., 2025)</td><td>0.598</td><td>0.850</td><td>0.723</td><td>0.800</td><td>0.333</td></tr></table>

Table 11: Detailed Output-Side Robustness results based on five membership inference attacks. Each score measures how close the attack’s ROC-AUC is to random guessing; higher scores are better. Finetune and Retain-Only serve as reference models; bold and underlined values indicate the best and second-best results among the unlearning methods, respectively.
<table><tr><td>Methods</td><td>LOSS ↑</td><td>ZLIB↑</td><td>GradNorm ↑</td><td>Min-K%↑</td><td>Min-K%++↑</td></tr><tr><td colspan="6">Reference Models</td></tr><tr><td>Finetune</td><td>0.840</td><td>0.748</td><td>0.536</td><td>0.969</td><td>0.965</td></tr><tr><td>Retain-Only</td><td>0.954</td><td>0.856</td><td>0.456</td><td>0.884</td><td>0.846</td></tr><tr><td colspan="6">General Unlearning Methods</td></tr><tr><td>GA (Maini et al., 2024)</td><td>0.874</td><td>0.823</td><td>0.033</td><td>0.828</td><td>0.828</td></tr><tr><td>GD (Maini et al., 2024)</td><td>0.872</td><td>0.969</td><td>0.000</td><td>0.665</td><td>0.590</td></tr><tr><td>KL (Maini et al., 2024)</td><td>0.855</td><td>0.762</td><td>0.444</td><td>0.971</td><td>0.969</td></tr><tr><td>RMU (Li et al., 2024b)</td><td>0.804</td><td>0.709</td><td>0.487</td><td>0.924</td><td>0.928</td></tr><tr><td>NPO (Zhang et al., 2024a)</td><td>0.907</td><td>0.798</td><td>0.000</td><td>0.961</td><td>0.902</td></tr><tr><td>UNDIAL (Dong et al., 2025)</td><td>0.831</td><td>0.637</td><td>0.000</td><td>0.940</td><td>0.977</td></tr><tr><td colspan="6">MLLM-specific Unlearning Methods</td></tr><tr><td>MMUnlearner (Huo et al., 2025)</td><td>0.712</td><td>0.800</td><td>0.280</td><td>0.594</td><td>0.562</td></tr><tr><td>MANU (Liu et al., 2025b)</td><td>0.806</td><td>0.720</td><td>0.423</td><td>0.940</td><td>0.997</td></tr><tr><td>MIP-Editor (Li et al., 2026)</td><td>0.853</td><td>0.759</td><td>0.494</td><td>0.972</td><td>0.995</td></tr><tr><td>SMFA (Zeng et al., 2025)</td><td>0.847</td><td>0.738</td><td>0.399</td><td>0.972</td><td>0.968</td></tr></table>

## A.3.5 CASE STUDY

These four paired case studies illustrate the tradeoff between suppressing a target answer and preserving useful responses. GA omits the target answer in all four forget cases, but also omits it in all four retain cases and often produces repetitive text. MIP-Editor and UNDIAL preserve the target answer in every retain case and suppress it in three of four forget cases; both still reveal it in the Eva Lindberg case. GD and RMU preserve all four retain targets but reveal the forget target in two cases each. The examples highlight distinct failure modes across methods and serve as qualitative illustrations rather than estimates of overall performance.

![](images/5cd554960af255029b87bdc1cf9eb071fd04b86e426305e02301381c113774ef.jpg)  
Figure 3: Four paired case studies of MLLM unlearning. Green Forgotten indicates that the target answer is absent; red Retained indicates that it is present.

![](images/f537d9092722625a2e6b9e0c58789d60581dbf02b4ecad41404c06e2ab3a8e29.jpg)  
Figure 4: Four paired case studies of MLLM unlearning. Green Forgotten indicates that the target answer is absent; red Retained indicates that it is present.

![](images/546cf95cb1c26faeb6a787a953dc47e2cd3cfe93f371a760dd55e29cd62d2bc0.jpg)  
Figure 5: Four paired case studies of MLLM unlearning. Green Forgotten indicates that the target answer is absent; red Retained indicates that it is present.

![](images/69b1e83236aa42d2f6d8db4ee6516b34c026e98312930b4005c6b23f0df3c564.jpg)  
Figure 6: Four paired case studies of MLLM unlearning. Green Forgotten indicates that the target answer is absent; red Retained indicates that it is present.

## A.4 DETAILS OF UNLEARNING METRIC EVALUATION

This section provides the formal definitions used to evaluate the faithfulness and robustness of MLLM unlearning metrics. For every evaluated metric, we normalize its range to [0, 1] and align its direction according to the corresponding evaluation objective.

## A.4.1 FAITHFULNESS

Let $m ( M ; { \mathcal { D } } _ { F } )$ denote the score produced by metric m for model M on the forget set $\mathcal { D } _ { F }$ , with a larger value indicating stronger evidence that the target knowledge is present. Applying the metric to the positive model pool $\mathcal { P }$ and the negative model pool $\mathcal { N }$ produces two score distributions:

$$
\begin{array} { r } {  { \boldsymbol { S } } _ { + } ^ { m } = \{ m ( M ;  { \mathcal { D } } _ { F } ) \mid M \in  { \mathcal { P } } \} , \qquad { \boldsymbol { S } } _ { - } ^ { m } = \{ m ( M ;  { \mathcal { D } } _ { F } ) \mid M \in  { \mathcal { N } } \} . } \end{array}\tag{64}
$$

We compute the faithfulness score as

$$
F _ { m } = \mathrm { A U C - R O C } \left( S _ { + } ^ { m } , S _ { - } ^ { m } \right) .\tag{65}
$$

A higher $F _ { m }$ indicates that metric m more reliably distinguishes models containing the target knowledge from models that have not learned it.

## A.4.2 ROBUSTNESS

Robustness to Quantization Let $M _ { u }$ denote an unlearned model and $\mathcal { Q } _ { b } ( M _ { u } )$ its b-bit quantized counterpart. Let $\tilde { m } ( M )$ denote the normalized unlearning score produced by metric $m ,$ where a larger value indicates stronger evidence of successful unlearning. Quantization robustness is defined as

$$
Q _ { m } = \operatorname* { m i n } \left( \frac { \tilde { m } ( \mathcal { Q } _ { b } ( M _ { u } ) ) } { \tilde { m } ( M _ { u } ) + \epsilon } , 1 \right) ,\tag{66}
$$

where ϵ is a small constant introduced for numerical stability. A higher $Q _ { m }$ indicates that the metric preserves its assessment after quantization.

Robustness to Relearning Let $M _ { u }$ be an unlearned model and $M _ { r }$ a retain-only reference model. Both models are exposed to the same forget-set examples under an identical relearning schedule. We use $M _ { u } ^ { ( t ) }$ and $M _ { r } ^ { ( t ) }$ to denote the corresponding models after t relearning steps. The change in normalized unlearning score is defined as

$$
\Delta _ { m } ( M , t ) = \tilde { m } ( M ) - \tilde { m } \Big ( M ^ { ( t ) } \Big ) .\tag{67}
$$

Relearning robustness is then computed as

$$
R _ { m } = \operatorname* { m i n } \left( \frac { \Delta _ { m } ( M _ { r } , t ) } { \Delta _ { m } ( M _ { u } , t ) + \epsilon } , 1 \right) .\tag{68}
$$

A higher $R _ { m }$ indicates that the metric characterizes knowledge recovery in the unlearned model similarly to that in the retain-only reference model.

Probing For a frozen model M, we extract hidden states $h _ { \ell } ( x ; M )$ at a selected layer ℓ and fit a linear probe to predict the target label z:

$$
W _ { \ell } ^ { * } ( M ) = \arg \operatorname* { m i n } _ { W } \frac { 1 } { | { \mathcal D } _ { \mathrm { p r o b e } } ^ { \mathrm { t r a i n } } | } \sum _ { ( x , z ) \in { \mathcal D } _ { \mathrm { p r o b e } } ^ { \mathrm { t r a i n } } } { \mathcal L } _ { \mathrm { p r o b e } } ( W h _ { \ell } ( x ; M ) , z ) .\tag{69}
$$

The fitted probe is evaluated on an entity-disjoint test set, with random-label and matched retain-only controls, following the protocol in Section A.2.1. We use probing as a model-side audit of residual target information, but exclude it from the metric-robustness aggregate. Unlike quantization and relearning, probing introduces an auxiliary readout whose predictions are not directly comparable to the model outputs on which the original generation- and likelihood-based metrics are defined. Moreover, the method-level results in Table 9 show probing-based forgetting scores of 0.993 and 0.996 for the Finetune reference at layers 16 and 24, close to the Retain-Only scores of 0.984 and 0.993. These near-ceiling scores suggest limited discrimination in the present setting. We therefore report probing within the model-side audit and use quantization and relearning for the metric-robustness aggregate.

## A.4.3 METRIC AGGREGATION

We aggregate quantization and relearning robustness using their harmonic mean:

$$
B _ { m } = \mathrm { H M } \left( Q _ { m } , R _ { m } \right) .\tag{70}
$$

The overall reliability score is computed as

$$
A _ { m } = \mathrm { H M } \left( F _ { m } , B _ { m } \right) .\tag{71}
$$

For two non-negative values x and y, the harmonic mean is defined as

$$
\mathrm { H M } ( x , y ) = \frac { 2 x y } { x + y + \epsilon } .\tag{72}
$$

## A.4.4 AI-ASSISTED CONSTRUCTION OF MLLMU VARIANTS

We used GPT-4o to construct text variants from MLLMU (Liu et al., 2025a). For question–answer pairs in the 10% forget split, we generated question paraphrases and answer expansions for constructing the faithfulness model pools. The original images and sample IDs were retained. Separately, we used GPT-4o-mini to generate one fact-preserving answer paraphrase and three plausible answers prompted to alter one central fact for each Generation Task question–answer pair in the MLLMU full set. The resulting 2,000-record file supplies evaluation answers for Truth Ratio, KS-Test, and JS Distance without modifying the original dataset. Only text questions and answers, not images, were submitted to the models.

## Question paraphrasing. The system prompt was:

You rewrite exactly one question for a multimodal   
QA dataset. Return JSON only with one key:   
rewritten\_question. Keep the meaning, entity, requested   
attribute, and answer unchanged. Do not add facts, change   
the task, or mention this instruction.

## The user prompt template was:

```snap
Source QA (rewrite only the question): {"question":
"<question>", "answer": "<answer>"}
```

## Answer expansion. The system prompt was:

You expand exactly one answer for a multimodal QA dataset.   
Return JSON only with one key: expanded\_answer. Make   
the answer more detailed and natural, preferably in two   
or three sentences, but use only facts explicitly present   
in the source QA. Do not invent names, dates, locations,   
occupations, or other details. The answer must still   
directly answer the original question. If the source   
contains only one short fact, explain that same fact without   
adding any new information.

The user prompt template was:

Source QA (expand only the answer): {"question":   
"<question>", "answer": "<answer>"}

## Evaluation answer generation. The system prompt was:

You construct evaluation answers for a multimodal factual   
QA benchmark. Return JSON only with exactly these keys:   
attribute, paraphrased\_answer, perturbed\_answers. The   
paraphrased answer must preserve every fact in the   
ground-truth answer and directly answer the question.   
Preserve the exact profession, place, person name, date,   
number, institution, object, or other answer-bearing phrase   
from the ground truth; change only its grammatical framing.   
Do not replace that phrase with a broader, narrower, or   
approximate synonym. Each perturbed answer must be a   
complete, independently generated answer that is plausible   
and fluent but factually false for the subject. Change   
exactly one central fact and preserve every other fact   
from the ground truth. When the answer contains multiple   
facts, select only one of them to alter. Keep the wording,   
specificity, language, and approximate length close to   
the ground truth. Do not add explanations, uncertainty,   
refusals, or formatting. All perturbed answers must be   
mutually distinct and distinct from the ground truth.

## The user prompt template was:

Create exactly {perturbations} perturbed answers for this   
single QA: {"question": "<question>", "ground\_truth":   
"<ground\_truth>"}. The attribute value should be a   
short semantic label such as "profession", "birthplace",   
"residence", "education", or "hobby".

We set perturbations to 3. The saved evaluation-data manifest records gpt-4o-mini and prompt version mllmu generation truth ratio v4. Outputs were checked for required fields and nonempty text; evaluation answers were additionally checked for the required count and uniqueness, with failed requests retried. These automated checks do not establish the factual correctness of every generated answer.

## A.4.5 HYPERPARAMETER SEARCH

For faithfulness, the search varies target-knowledge exposure rather than selecting a single best learning rate. We cross six training datasets with five learning rates, {1, 2, 3, 4, 5} × 10<sup>−5</sup>. The positive pool comprises full, full paraphrased, and full bio, all of which contain the correct target facts. The negative pool comprises retain90, retain90 forget10 perturbed, and retain90 celeb bio, none of which contains the correct target facts. The perturbed-answer variant pairs target prompts with incorrect answers, while the unrelated-biography variant controls for biographical format.

![](images/b834652f8bdecba9600a25943a84a2e90714c5e269f4ed2999b29bc560ac34e9.jpg)  
Figure 7: Faithfulness of 13 MLLM unlearning metrics on the MLLMU 10% forget setting. Blue and orange histograms show score distributions for matched models trained with and without the target knowledge, respectively. Dashed lines mark metric-specific thresholds, and AUC values summarize how well each metric separates the two model pools

![](images/af8a5aa5e048e529d86ea53211064e7f663bc5dc34e4ac839e1eb596839fa2f1.jpg)

![](images/2428e40e532c1274f5a539e26f8f0f846a79c82b710ccd3d37389ad37a960aea.jpg)

![](images/e27a9d3fd73e10b4c3e78ec4a33ccaaeecf261a775c99fbacf66e64529d13d24.jpg)

![](images/642541e19f491405e12eeb624ac90520ae51f3d5f313676a2a62c1c93fbb58dc.jpg)

![](images/157ab1ffb4a925850fd3f231cb5a25a55bc51eb66f61c397533763e90dd8ccd2.jpg)

![](images/75e057b52a79044488f8bbd5b6c6ec5cc7b31432e64620f0adef139853b4e6c3.jpg)

![](images/20ef94565ef81619118c04249364b920be7767d63fee10071b810975116eadf6.jpg)

![](images/3f06713e11d0235aaa1ab08fc22d66218b5452be45ffb89968a003891ff7c5bd.jpg)

![](images/d724c0740317c6318470c3f5df69168a941301f228c3cfc41f0dffc6c5bd5c12.jpg)

![](images/20788f5ff7ccce085bdee3823f4a2b895b309e6e24646f5b786fc9b9951c51bb.jpg)

![](images/a5ef50e32dcb549d16712e04ebf9cb6487f6aff232de44b28ea943ffad5777be.jpg)  
Figure 8: Relearning robustness of 11 MLLM unlearning metrics. Blue points show eligible unlearned checkpoints before and after targeted relearning; the red star marks the mean Retain-Only reference. The dashed line indicates $y = x ,$ while the shaded region marks apparent recovery of target knowledge beyond the Retain-Only reference.

![](images/253eab995c75225b18a509710ca5a583f1e293672300255598291df52c5d9df2.jpg)

![](images/c51be43e9640b1fdf3f7bb6df9d02265409b38eb4f67cba67f50ef0a6888dbcf.jpg)

![](images/723443ae277b07aad2bd63fcb6837da3c803bf69f18a6ae5d65170b925bc0eea.jpg)

![](images/f265549004428a816009acee217dfb2993a5fe40fc3c00f884e320df905fb45f.jpg)

![](images/c8fcef4b6b3b3372b51b32d12410edea394d315a1c5160c33808a912b392c826.jpg)

![](images/06fd8c27e489b89fe0e9f3fd144c398307e229d57437a18118356fe611736208.jpg)

![](images/a4bdace1856357fd24d2bbf89f050a0e2e61bbe2b65413d55a12cc5b51f3061a.jpg)

![](images/083543060fb5e5075128cabeedf376aa3de168cfcba5480843b4848920be5588.jpg)

![](images/a33dc24ca9590e3c2cb4ef73547261a6818bc30afae21baff1ac512d9a1c06cf.jpg)

![](images/ed8b90eb2fbc2172a6d521970b5354700e2ee9f5462f96dab098c6448a708960.jpg)

![](images/4c0bb8f2af7e0e7d92149a741653a44b9f7c64c8e8b879b9eafe70e2949d4007.jpg)  
Figure 9: Quantization robustness of 11 MLLM unlearning metrics. Blue points show eligible unlearned checkpoints before and after quantization; the red star marks the mean Retain-Only reference. The dashed line indicates $y = x ,$ while the shaded region marks quantization-induced shifts toward greater apparent target knowledge.

## A.5 LIMITATIONS

Although OPEN-MMUNLEARNING supports five benchmarks, eight MLLMs, and twelve unlearning methods, our controlled method comparison covers ten methods on the MLLMU 10% forget setting, while the metric meta-evaluation uses LLaVA-1.5-7B in the same setting. The observed rankings may change across models, datasets, forget ratios, and evaluation protocols. Our robustness evaluations cover selected interventions and attack settings, so they cannot establish that targe knowledge is irrecoverable under every possible attack or later model update. Finally, aggregate scores depend on the chosen normalization and aggregation rules and should be interpreted along side their component results. Broader controlled evaluations are needed to assess how consistently these findings generalize.

## A.6 REPRODUCIBILITY STATEMENT

The main paper describes the experimental setup for the controlled method comparison and metric meta-evaluation. Appendices A.3.1, A.3.2, and A.3.3 specify the unlearning objectives, hyperparameter search spaces, checkpoint selection, and score aggregation procedures. Appendix A.4 defines the metric meta-evaluation protocol, while Appendix A.4.4 documents the construction procedure and prompts for the AI-generated MLLMU variants. Experiments use structured configuration files to specify models, datasets, methods, and evaluation settings.

## A.7 ETHICS STATEMENT

MLLM unlearning concerns information that may raise privacy, safety, and copyright issues, and our benchmark suite includes tasks in each of these settings. Our robustness evaluations probe for residual target knowledge through model interventions, adversarial inputs, and membership inference. These evaluations are intended to expose potential failures; favorable scores or apparent refusals should not be interpreted as guarantees of complete deletion, privacy protection, legal compliance, or deployment safety. Users of OPEN-MMUNLEARNING should follow the terms of the underlying datasets and assess potential information leakage before releasing unlearned models or their outputs.