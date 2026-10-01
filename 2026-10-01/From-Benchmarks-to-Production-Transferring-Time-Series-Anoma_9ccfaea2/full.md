# From Benchmarks to Production: Transferring Time Series Anomaly Detection Methods for Electricity Production Monitoring

Nicolas Vautier<sup>1</sup>, Paul Caron<sup>2</sup>, Nardi Xhepi<sup>2</sup>, Felicie Bizeul´ <sup>1</sup>, Manel Boumghar<sup>1</sup>, Christophe Degouy<sup>2</sup>, Paul Boniol<sup>3</sup> <sup>1</sup> EDF Lab Paris Saclay, <sup>2</sup> EDF, DOAAT, <sup>3</sup> Inria, Ecole Normale Superieure (PSL), CNRS´ <sup>1,2</sup> firstname.lastname@edf.fr, <sup>3</sup> paul.boniol@inria.fr

Abstract—Accurate forecasting of electricity production is essential for maintaining the operational efficiency and strategic planning of energy utilities. In industrial settings, such forecasts are generated daily to ensure supply–demand balance and optimal management of production assets. However, the increasing complexity of modern power systems and data flows poses significant challenges for ensuring the reliability and consistency of these forecasts. This paper addresses the problem of anomaly detection in short-term production forecasts at EDF, formulated as identifying atypical intra-day patterns that may signal data quality issues or operational irregularities. We introduce TAMIS, a scalable and interpretable system that analyzes daily production time series to automatically detect anomalous days based on deviations from historical patterns learned from past data. Designed for human-in-the-loop workflows, TAMIS surfaces top-ranked anomalies through an automated daily newsletter, enabling efficient expert review and continuous monitoring. An extensive experimental evaluation on real-world industrial data demonstrates that TAMIS achieves the best accuracy–efficiency trade-off compared to baseline methods. To foster further research and reproducibility, we publicly release the anonymized application datasets used in our study.

## I. INTRODUCTION

Time series data play a central role in a wide range of industrial applications [1]–[3], from predictive maintenance and fault detection [4] to resource optimization and demand forecasting [5]. A common challenge across these domains is the reliable detection of unusual patterns or behaviors that deviate from historical data (most commonly called anomaly or outlier in the literature [6]–[8]). Anomaly detection in time series is particularly critical in environments where real-time decisions must be made based on automatically generated forecasts or measurements. Despite significant progress in machine learning and statistical methods for time series analysis [9], [10], the practical deployment of anomaly detection systems in industry remains difficult due to specific constraints that often arise in operational settings.

This paper is motivated by the real-world context of energy production planning at EDF, one of the largest electricity producers in Europe. In this setting, large volumes of heterogeneous time series are generated daily to forecast the operation of power plants and storage systems over a 24-hour horizon. Each forecast is represented as a fixed-length vector capturing 48 half-hourly values for the upcoming day. These forecasts are critical inputs to balancing supply and demand, planning energy dispatch, and engaging with energy markets. Ensuring their reliability is thus essential. However, due to the scale and complexity of the data, it is not feasible for domain experts to manually review all forecasts, which necessitates an automated, trustworthy anomaly detection solution. More precisely, our industrial constraints are as follows: First, anomaly detection must be online, in the sense that it evaluates whether the most recent day-ahead forecast is abnormal based on historical data up to that point. Second, the solution must be interpretable and human-centered, as detection results are analyzed daily by energy experts who require concise, actionable explanations. Third, the method must be scalable, capable of handling large volumes of time series, with low latency, and limited hardware resources. These constraints rule out many existing approaches that rely on deep learning models, exhaustive offline training.

![](images/0456e15150a4af2f08d525333e68f572e91953c90841106793e27b415153fe27.jpg)  
Fig. 1: Examples of time series in our proposed JO dataset (anomalies are highlighted in red).

To address these challenges, we propose TAMIS, a lightweight and interpretable anomaly detection system designed for production-grade deployment. Each daily forecast is scored for abnormality using a feature-based approach. The results are automatically ranked and integrated into a daily report. Only the most abnormal cases are returned, allowing experts to review them and make decisions accordingly.

Beyond proposing a practical solution, this work also provides insights into the broader journey of moving from academic research on time series anomaly detection to an operational system. We discuss how methodological advances translate into real-world applicability, and how constraints such as hardware efficiency, scalability, and expert usability shape the design of a deployable framework. In this sense, the paper not only introduces a new system, but also illustrates the path from research concepts to an industry-ready solution.

<table><tr><td>Symbol</td><td>Description</td></tr><tr><td> $\Delta \in \mathbb { N }$ </td><td>Sampling interval (30 minutes).</td></tr><tr><td> $M \in \mathbb { N }$ </td><td>Number of samples per day  $( M = 4 8 ) .$ </td></tr><tr><td> $\pmb { C } _ { d } \in \mathbb { R } ^ { M }$ </td><td>Daily window (24-hour profile) for day d</td></tr><tr><td> $c _ { m } \in \mathbb { R }$ </td><td>m-th value of  $C _ { d }$ </td></tr><tr><td> $\pmb { F } _ { d } \in \mathbb { R } ^ { i }$ </td><td>Feature set for day d</td></tr><tr><td> $\textstyle f _ { i } \in \mathbb { R }$ </td><td>j-th value of  $\mathbf { \nabla } \mathbf { F } _ { d }$ </td></tr><tr><td> $\dot { \boldsymbol { T } } \in \mathbb { R } ^ { n \times M }$ </td><td>Time series of n consecutive daily windows</td></tr><tr><td> $H _ { i } \in \mathbb { R } ^ { n }$ </td><td>Historical values for the j-th feature, for a given time series</td></tr><tr><td> ${ \mathfrak { n } } \in \mathbb { N }$ </td><td>Number of days (i.e., windows) in the time series T.</td></tr><tr><td> $i \in \mathbb N$ </td><td>Number of features T.</td></tr><tr><td> $\boldsymbol { A }$ </td><td>Anomaly detection function.</td></tr><tr><td> $D$ </td><td>A unique anomaly detection model (i.e., detector).</td></tr><tr><td> $\mathcal { M }$ </td><td>An ensembling or selection method.</td></tr><tr><td> $S \in [ 0 , 1 ]$ </td><td>Anomaly score.</td></tr></table>

TABLE I: Summary of main notations used in this paper.

Finally, we evaluate our system on a large, real-world dataset collected from EDF’s internal production forecasts over multiple years. This dataset is diverse in both physical nature and statistical behavior, reflecting the complexity of our use case. We perform a rigorous experimental study to assess the scalability and detection accuracy of different algorithmic design within our framework. Furthermore, we release a curated and anonymized version of this dataset as a public benchmark to encourage future research in time series anomaly detection. Overall, our contributions are as follows:

• Novel Industrial Architecture: We analyze the requirements of time series anomaly detection in our industrial setting and propose a novel architecture tailored to these practical operational constraints (Sec. II).

• Designed-for-Purpose Components: We introduce a lightweight detector, ASHES, and $\mathbf { T A M I S } _ { \mathcal { F } }$ , a feature set that forms a core component of our framework (Sec. V).

• An End-to-end Pipeline: We present TAMIS, a novel end-to-end anomaly detection framework for electrical production time series that balances interpretability, accuracy, and scalability (Sec. V).

• Open Data: We release three open-access datasets of real, labeled time series, which are among the first of their kind in the energy sector (cf. Sec. VI-A1).

• Extensive Evaluation: We conduct an extensive experimental evaluation demonstrating that TAMIS outperforms state-of-the-art baselines while maintaining a superior accuracy-efficiency trade-off (Sec. VI).

## II. INDUSTRIAL CONTEXT AND PROBLEM FORMULATION

EDF, France’s leading electricity producer, is committed to building a carbon-neutral energy future through a mix of production sources. A critical enabler of this mission is the ability to continuously balance energy supply and demand, a task that relies heavily on accurate forecasts of both production and consumption. Maintaining this balance in the face of dynamic operational conditions and unforeseen events is essential not only for risk mitigation but also for value optimization.

Time series data lie at the heart of this operational ecosystem. These series capture a wide variety of measurements, including electrical power output, reservoir levels, flow rates, turbine states, production targets, and derived variables such as adjustment margins. These series differ in scale, dynamics, semantics, and physical meaning, making manual inspection infeasible. Each day, large volumes of new data are ingested and logged, underscoring the need for automated, scalable monitoring tools to detect and flag abnormal behavior.

Figure 1 illustrates examples of time series and anomalies of interest in our study. Overall, time series from EDF datasets contains heterogeneous types of time series and anomalies. Figure 1(a) illustrates a global point anomaly (i.e., a single value that deviates from the global distribution of values within the time series), Figure 1(b) depicts a contextual point anomaly (i.e., a point that deviates from the distribution of values within a given window of the time series), and Figure 1(c) shows a Collective anomaly (i.e., abnormal sequence of values that, if taken independently, would be considered normal).

A key aspect of this work is the operationalization of the detection within a human-in-the-loop framework. To bridge the gap between model output and practical action, a mailing list is implemented. Each day, the system automatically scores and ranks all potential anomalies. Subsequently, a summary report is distributed to a list of experts. This report is intentionally concise, presenting only the highest-ranked anomalies to mitigate alert fatigue and focus expert analysis on the most pressing issues. Each entry in the report is intended to include key contextual information, such as the anomaly score and timestamp, enabling experts to efficiently triage, validate, and prioritize potential anomalies for in-depth investigation.

## A. Problem Definition

As mentioned in the previous section, our objective is to identify anomalies in time series, a well-studied problem in the literature for decades [9]. Anomaly detection in this context is framed as an online, last forecasted window detection task, where the objective is to assess whether the most recently observed forecasted value in a series deviates significantly from historical behavior. Unlike retrospective analyses (the most common tackled problem in the literature [9]), the goal is not to detect previously missed events but to identify anomalies as they emerge, thus enabling timely interventions. This real-time requirement aligns with EDF’s operational needs, where undetected anomalies in key indicators could cascade into significant planning errors or missed opportunities.

We call a window the 24-hour profile of a variable c sampled every $\Delta = 3 0$ minutes. Formally, a window $C _ { d }$ for a given day d is defined as $\boldsymbol { C } _ { d } = \left( c _ { 1 } , \ldots , c _ { M } \right) \in \mathbb { R } ^ { M }$ , where $M =$ 48 and $c _ { m }$ denotes the measurements for the m-th half-hour interval of day d. Therefore, a time series T is an ordered set of windows defined as follows:

![](images/af9f3e3078a24a5c04cdb94ee3234fa929ce56257dcc3f9b41637ecef95fb983.jpg)  
Fig. 2: Raw-based Time series anomaly detection methods in the literature [9]

$$
\pmb { T } = \left( \pmb { C } _ { d _ { 1 } } , \dots , \pmb { C } _ { d _ { n } } \right) \in \mathbb { R } ^ { n \times M }
$$

with n being the total number of days in the time series T. In practice, the daily measurements $C _ { d }$ is forecasted from the previous day d − 1. The primary objective of this work is to perform anomaly detection on such a daily production composing a given time series T. An anomaly is defined as the overall intra-day sequence (i.e., window) of points of T itself, which deviates significantly from what is considered normal or expected behavior for EDF’s power generation system for a day. This normality is established based on patterns learned from historical data (including typical intraday profiles for various day types). In practice, we consider 3 years of historical data. Formally, in our industrial context, we define an anomaly detection function as follows:

Problem 1 (Anomaly Detection). We define an anomaly detection function A as $\mathcal { A } : \mathbb { R } ^ { n \times M } \times \mathbb { R } ^ { M } \longrightarrow [ 0 , 1 ]$ , that takes as input a time series $\pmb { T } \in \mathbb { R } ^ { n \times M }$ and the last day $C _ { d _ { n } } ~ \in ~ \mathbb { R } ^ { M }$ , and returns a real-valued anomaly score S assessing how anomalous the last day is, given the historical context. A higher score indicates a higher likelihood of the last day being anomalous.

The function A can either correspond to a unique anomaly detection model, denoted by D, or to an automatic detection solution, denoted by M, combining multiple detectors D. We will describe in the following sections the distinction between a unique detector and an automatic solution. While our problem setting shares core principles with the broader literature on time series anomaly detection, such as the importance of temporal dependencies and seasonality, it presents several practical constraints limiting the applicability of a large panel of existing methods. These constraints are listed below.

• (C1) Scalability: The practical applicability of an anomaly detection system in an industrial context, such as managing energy grids, is heavily dependent on its computational performance. The system must exhibit low execution time to allow for rapid, proactive interventions.

• (C2) Limited hardware: While model training can leverage high-performance computing resources, operational deployment must conform to strict hardware constraints. In our context, anomaly detection must run in nearreal-time on standard CPU-based infrastructure, without access to GPUs or specialized accelerators.

• (C3) Interpretability: In an operational context such as energy grid management, the value of an anomaly detection system is tightly coupled with its ability to produce interpretable outputs. Since detected anomalies must be reviewed and acted upon by domain experts (often under tight time constraints) models that function as black boxes are of limited utility.

• (C4) Data Diversity: In our use case, time series are highly heterogeneous in terms of provenance and structure. First, missing values may have semantic meaning. Rather than removing them, one needs to preserve them to maintain data integrity and retrieve relevant knowledge that could indicate potential anomalies. Then, time series provenance (i.e., from diverse domains and measurement types) might lead to severe Out-of-Distribution scenarios, both in terms of trends and anomaly types.

## III. TIME SERIES ANOMALY DETECTION: Foundations and Background

In this section, we review existing time-series anomaly detection methods through the lens of the key operational constraints enumerated in the previous section.

## A. Raw-based time series anomaly detection

In recent years, significant research has been conducted in the field of time-series anomaly detection. Numerous studies and experimental benchmarks have been written to summarize and analyze state-of-the-art methods [7], [11]–[14]. These comprehensive evaluations reveal a diverse landscape of methodologies (i.e., detectors D), which can be broadly classified into several foundational families based on their core operating principles [9]. Figure 2 depicts a large panel of methods grouped in process-centric families according to [9].

One major family consists of prediction-based methods, which train a model on a given self-supervised prediction task. An observation is classified as anomalous if it significantly deviates from the model’s prediction, with the anomaly score derived from the prediction error. Such methods can be reconstruction-based [15] or forecasting-based [16].

A second prominent category is distance-based methods, which operate under the assumption that anomalous points are isolated from the bulk of the data. These techniques identify outliers by measuring the distance of a data point to its nearest neighbors, with large distances indicating abnormality [17].

Closely related are density-based methods, which use a representation of the time series (i.e., tree [18] or graph structure [6]) and compute an anomaly score based on some density criteria of the given representation (i.e., depth of the tree, or centrality in the graph).

While prediction-based methods are appealing due to their ability to model temporal dynamics, they often rely on complex architectures with high computational and memory demands, and may be ill-suited for specific use cases on high-velocity data with real-time deployment on constrained hardware requirements (i.e., failing to meet C1 and C2). Additionally, their black-box nature poses challenges for interpretability (i.e., not respecting C3). Conversely, raw-based approaches such as distance- and density-based methods may offer lighter computation, but they struggle to handle data quality issues like missing values (failing to meet C4) and often lack the contextual transparency needed for non time series practitioners (i.e., not fully respecting C3). These limitations underscore the need for methods explicitly designed with industrial constraints in mind.

## B. Lack of interpretability: Toward Feature-based approaches

Feature-based methods offer a promising avenue for addressing the industrial constraints of interpretability, scalability, and robustness to data quality issues. By transforming raw time series into structured feature vectors (often composed of statistical, structural, or domain-specific descriptors), these approaches enable the use of appropriate density-based algorithms in a lower-dimensional, and meaningful space. Unlike raw-based distance- or density-based methods, feature-based techniques allow better handling of missing values through feature design and provide clear interpretability by exposing which characteristics contribute to the anomaly. As such, they strike a balance between operational feasibility and detection performance, making them particularly well-suited for realworld deployment in our industrial contexts.

In practice, the feature-based methodology [19]–[21] entails a two-stage process that transforms the complex temporal problem into a more manageable static outlier-detection task. First, a raw time-series window is converted into a fixed-length vector of numerical features. These features can capture a wide range of properties, including basic statistics (mean, variance), frequency-domain information, and entropy measures. In the second stage, a classical outlier-detection algorithm is applied to the feature space to identify feature vectors that are abnormal relative to the rest of the population [22]. Common algorithms used for this purpose include Isolation Forest [18], Local Outlier Factor (LOF) [23], and One-Class SVM [24].

Several tools have been proposed for automatic feature extraction. Among the most widely used are HCTSA [26], TSFRESH [21], and CATCH22 [27]. HCTSA (Highly Comparative Time Series Analysis) performs massive feature extraction, computing over 7,700 features for each time series. These features span statistical distribution measures (e.g., mean, skewness, outlier ratios), autocorrelation and spectral properties, entropy metrics, and model-based characteristics.

![](images/45eacd2dfa22796962b13fd02862d54f736a5c1dd311ba4dd4c5ffbc29d23a8e.jpg)  
Fig. 3: Example of TSFRESH and CATCH22 features on IOPS [25] time series.

TSFRESH extracts 794 time series features by default and includes an automated feature selection process based on hypothesis testing. This framework supports both exploratory analysis and integration into operational pipelines, making it a flexible option for industrial applications.

CATCH22 builds upon HCTSA by identifying a compact subset of 22 features that provide strong classification performance across 93 UCR/UEA datasets [28]. These features are selected to minimize redundancy and maximize diversity, but are specifically tuned for z-normalized time series (zero mean and unit variance) and classification tasks.

While each of these libraries provides valuable insights, they present specific limitations in our context: (i) HCTSA and TSFRESH produce high-dimensional feature sets, which are impractical for direct use in real-time anomaly detection. Moreover, TSFRESH’s selection is not tailored to anomalyspecific tasks, potentially misaligning with our deployment needs. (ii) CATCH22, due to its compact size and design for rapid analysis, appears promising for initial exploration. However, its features were optimized for classification, not anomaly detection. Figure 3 depicts the TSFRESH and CATCH22 features computed on an IOPS [25] time series. TSfresh contains features that allow the detection of anomalies (labeled in red in Figure 3), while CATCH22 features are not sufficient for detecting the same anomalies.

Given these constraints, no existing feature set is suitable for our industrial use case, underscoring the need for feature selection and evaluation tailored to our operational context.

## C. No Universal Best: Toward Automatic Solutions

Recent benchmarking studies have consistently shown that there is no single anomaly detection algorithm that performs best across highly heterogeneous collections of time series [7], [11], [29], [30]. Instead, the performance of individual methods varies significantly depending on the characteristics of the time series (e.g., stationarity, periodicity) and the nature of the anomalies (e.g., point-wise, contextual, or subsequence).

![](images/1f4150fcc11175f95f9fba28f4c9d05d6bb60f85cfd0e2ee07a8f1f111c8adf6.jpg)  
Fig. 4: Time series anomaly detection: from unique detector (a) to supervised ensembling (d).

This observation is particularly important in our context, where we must monitor a diverse set of time series with varying behaviors and noise levels. Furthermore, while most research algorithms produce anomaly scores for each time point, our setting requires a single decision at the last timestamp, introducing an additional mismatch between standard benchmarks and operational needs.

To overcome these limitations, we use several automatic solutions (i.e., meta model and ensembling methods M). Overall, two main strategies have been proposed: ensembling (supervised or unsupervised) and automatic model selection. Figure 4 illustrates these strategies. Ensembling combines the outputs of multiple detectors, typically by averaging or maximizing their scores, to improve robustness. These methods often outperform individual algorithms on public benchmarks [29], but at the cost of higher computational overhead, which is problematic for real-time applications.

On the other hand, recent AutoML-based approaches have explored model selection as an alternative. These methods aim to select the most appropriate detector for each time series by learning a meta-model that maps extracted features to the best-performing method [29], [31], [32]. While promising, these techniques suffer from limited generalization in out-ofdistribution settings [29]. The latter is an essential concern for real-world deployment, where new time series may exhibit previously unseen patterns or behaviors.

Given these limitations, we need to adopt an ensembling strategy that prioritizes robustness in out-of-distribution settings while maintaining low computational cost.

## IV. RESEARCH QUESTIONS

In light of the industrial constraints identified and the literature review in the previous section, our proposed solution is guided by several key research questions:

• R1. How to combine detectors for optimal accuracy–efficiency trade-offs? While ensembling can enhance robustness and detection accuracy, it also increases computational cost (C1). Conversely, model selection offers greater efficiency but may suffer in Out-of-Distribution scenarios (C4).

• R2. Which feature set to use? A smaller feature set can improve scalability (C1, C2). But, small or non-dedicated features may negatively impact detection accuracy, reducing the system’s practical usefulness (C3, C4).

• R3. What is the impact of Out-of-Distribution? Identifying the correct automatic solution setting can keep execution time low (C1,C2) while providing greater robustness in Out-of-Distribution scenarios (C4).

These research questions directly inform the design of our TAMIS framework (detailed in the following section), ensuring that it balances operational feasibility with high detection performance. Overall, this study not only describes a databased pipeline for anomaly detection in production but also presents a roadmap for moving from research findings in data mining and data management to production-based systems.

## V. PROPOSED METHOD: THE TAMIS SYSTEM

In this section, we introduce TAMIS, a lightweight, interpretable anomaly detection system specifically designed for production-grade deployment. As shown in Figure 5, the method involves a training phase composed of several steps: starting from manually labeled datasets (step (a)), a novel feature set, TAMIS<sub>F</sub>, is computed for all available time series (step (b)). Two complementary detectors are applied to these features (step (c)), and their outputs are combined through a supervised ensemble, referred to as SEASA (step (d)). Once trained, this supervised ensemble model is deployed for inference in production (Figure 5(2)).

An important practical consideration is that precomputed features from the previous day are stored in a dedicated database and reused when computing anomaly scores for the following day. This caching mechanism drastically reduces runtime and enables near real-time performance in production.

Finally, the most abnormal time series are presented to the end-user through a visual interface, shown in Figure 5(3). In the following sections, we detail the design of the proposed feature set TAMIS<sub>F</sub>, the two individual detectors, and the supervised ensembling mechanism, SEASA.

## A. TAMIS<sub>F</sub>: A Novel Feature Set

The proposed feature set, TAMIS<sub>F</sub>, aims to capture the most informative and generalizable characteristics of industrial time series for anomaly detection. The feature extraction pipeline begins by aggregating raw measurements into daily windows, denoted as $C _ { d } ,$ from which a feature vector $\pmb { F } _ { d }$ is computed. The final feature set results from the integration of two complementary design strategies: (i) an expert-based feature pool derived from domain knowledge, and (ii) a TSADbased feature pool identified through synthetic data generation and supervised feature selection.

![](images/1b334651b859b5c3ff11b9d0b1a67686e4613d22fa5207f2d8cfccb656788cf9.jpg)  
Fig. 5: (1) Training and (2) production pipeline of the proposed solution TAMIS. The end-user interface is illustrated in (3).

1) Expert-Based Feature Pool: The initial stage of TAMIS<sub>F</sub> construction relied on domain expertise from energy production specialists. A total of 36 candidate indicators were manually proposed to describe operational behavior and potential deviations. To isolate the most relevant predictors, a structured feature selection protocol was implemented, combining ablation analysis and individual feature evaluation.

Specifically, three model configurations were trained to assess each feature’s contribution: (i) A baseline model trained on the full feature set; (ii) An ablation model trained on all features except the one under consideration (leave-one-out); (iii) An individual model trained on the single feature alone.

Each configuration produced four recall-based performance curves, quantifying the ratio of correctly detected anomalies with respect to both the detection score and rank. By qualitatively and quantitatively analyzing these curves, the ten most informative and discriminative features were retained. These are reported in the expert-based section of Table II.

Nevertheless, two limitations arise with the approach mentioned above. First, since the selection relies on historically labeled anomalies, it may be biased toward previously observed anomaly types, potentially limiting its generalization to unseen fault patterns. Second, the anomaly distribution within the evaluation dataset may not reflect their true operational frequency, introducing sampling bias. As a result, the selected features might overemphasize frequently occurring anomalies at the expense of rare but critical ones.

2) TSAD-Based Feature Pool: To address the scarcity of labeled data and improve cross-domain generalization, we developed a data-driven feature discovery pipeline inspired by time series anomaly detection (TSAD) research. This methodology converts the unsupervised detection problem into a supervised learning framework through synthetic data generation, followed by a two-stage feature selection process. Synthetic Data Generation and Feature Engineering: A labeled corpus is first synthesized by injecting artificial anomalies into normal time series (from JO dataset described in Section VI-A1). Inspired by recent experimental evaluation studies [7], [11], these anomalies emulate representative fault types, including point perturbations, temporal shifts, and amplitude scaling. The resulting dataset provides explicit ground-truth labels for controlled feature selection. A highdimensional feature space is then constructed by computing descriptive statistics and signal transformations over each daily window, integrating features from TSFRESH and CATCH22. Two-Stage Supervised Feature Selection: From the collected features, we perform the following selection pipeline:

Relevance Filtering: Each feature is ranked according to its discriminative capacity for the synthetic labels, evaluated using p-value and Benjamini Hochberg procedure [33], ANOVA Ftest [34], and gradient boosting importance scores [35].

Redundancy Removal: The top-ranked features are clustered based on pairwise correlation to remove redundant predictors.

<table><tr><td>Feature Description</td><td>Formula</td><td>Complexity</td></tr><tr><td colspan="3">Expert-based: Top-10 features from energy-production expert-knowledge</td></tr><tr><td>amplitude: maximum amplitude for a given window  $C _ { d } .$ </td><td> $\operatorname* { m a x } ( \pmb { C } _ { d } ) - \operatorname* { m i n } ( \pmb { C } _ { d } )$ </td><td>O(n)</td></tr><tr><td>val max: Maximum value in  $C _ { d } .$ </td><td> $\operatorname* { m a x } ( C _ { d } )$ </td><td>O(n)</td></tr><tr><td>val min: Minimum value in  $C _ { d } .$ </td><td> $\operatorname* { m i n } ( C _ { d } )$ </td><td>O(n)</td></tr><tr><td>val mean: Mean of  $C _ { d } .$ </td><td> $\mu ( C _ { d } )$ </td><td>O(n)</td></tr><tr><td>val std: Standard deviation of  $C _ { d } .$ </td><td> $\sigma ( C _ { d } )$ </td><td>O(n)</td></tr><tr><td>missing values: Count of missing (NaN) values in  $C _ { d } .$ </td><td> $\begin{array} { r } { \sum _ { c _ { m } \in C _ { d } } \tilde { \mathbb { I } } ( c _ { m } ) } \end{array}$ </td><td>O(n)</td></tr><tr><td>diff max: Maximum consecutive difference in  $C _ { d } .$ </td><td> $\mathrm { m a x } _ { c m \in C _ { d } } \vert \bar { c } _ { m } - c _ { m - 1 } \vert$ </td><td>O(n)</td></tr><tr><td>diff min: Minimum consecutive difference in  $C _ { d } .$ </td><td> $\begin{array} { r } { \underset { \pmb { c } } { \operatorname* { m i n } } _ { \pmb { c } _ { m } \in \pmb { C } _ { d } } \left| \boldsymbol { c } _ { m } - \boldsymbol { c } _ { m - 1 } \right| } \end{array}$ </td><td>O(n)</td></tr><tr><td>diff mean: Average consecutive differences in  $C _ { d } .$ </td><td> $\begin{array} { r } { \frac { 1 } { \left. C _ { d } \right. - 1 } \sum _ { c _ { m } \in C _ { d _ { . } } } \left. c _ { m } - c _ { m - 1 } \right. } \end{array}$ </td><td>O(n)</td></tr><tr><td>diff step1: difference between  $c _ { 0 }$  and  $c _ { M } ^ { \prime } ,$  first and last values of  $C _ { d }$  and  $C _ { d - 1 }$  respectively.</td><td> $c _ { 0 } - c _ { M } ^ { \prime }$ </td><td>O(n)</td></tr><tr><td colspan="3">TSAD-based: Top-10 features from a synthetic time series anomaly detection evaluation</td></tr><tr><td>sum reoccurring values: Sum of all values that appear more than once in  $C _ { d } .$ </td><td></td><td></td></tr><tr><td>Benford correlation: Correlation with the expected Benford&#x27;s Law distribution of first digits.</td><td> $\textstyle \sum _ { v \in V _ { \mathrm { r e c c . } } } v$ </td><td>O(n)</td></tr><tr><td>Fourier entropy: Shannon entropy of the Power Spectral Density (PSD), with 100 bins.  $( \mu ) .$ </td><td> $\mathrm { c o r r } ( P _ { \mathrm { o b s } } , P _ { \mathrm { B e n f o r d } } )$ </td><td>O(n)</td></tr><tr><td>longest strike above mean: Length of the longest consecutive run of values above the mean</td><td> $- \sum p _ { i } \log _ { 2 } ( p _ { i } )$   $\mathrm { L e n g t h ~ o f ~ r u n } > \mu$ </td><td>O(n log(n)) O(n)</td></tr><tr><td>mean change: Mean of the absolute differences between consecutive values.</td><td> $\mathrm { m e a n } ( | \Delta C _ { d } | )$ </td><td>O(n)</td></tr><tr><td>last location of max: Relative index of the last occurrence of the maximum value.</td><td> $\mathrm { a r g m a x } _ { \mathrm { l a s t } } ( C _ { d } ) / N$ </td><td>O(n)</td></tr><tr><td>variation coefficient: Ratio of the standard deviation (σ) to the mean  $( \mu ) .$ </td><td> $\sigma / \mu$ </td><td>O(n)</td></tr><tr><td>permutation entropy: Complexity measure based on ordinal patterns (dim=5, lag=1).</td><td> $\begin{array} { r } { - \sum _ { i \in \{ 1 , \dots , D ! \} } p _ { i } \log _ { 2 } ( p _ { i } ) } \end{array}$ </td><td>O(n)</td></tr><tr><td>has duplicate min: Binary indicator for whether the minimum value appears more than once.</td><td> $\mathbb { I } ( \mathrm { c o u n t } ( \operatorname* { m i n } ( \mathbf { \dot { C } } _ { d } ) ) > 1 )$ </td><td>O(n)</td></tr><tr><td>longest strike below mean: Length of the longest consecutive run of values below the mean (μ).</td><td> $\mathrm { L e n g t h ~ o f ~ r u n } < \mu$ </td><td>O(n)</td></tr></table>

TABLE II: $\mathrm { T A M I S } _ { \mathcal { F } }$ features. Each feature is computed for an entire window (i.e., day $d ) .$ Note: $\mathbb { I } ( p )$ is the indicator function for NaN values. $\Delta C _ { d }$ represents first-order differences. $V _ { \mathrm { r e o c c . } }$ is the set of reoccurring values in $C _ { D }$

From each cluster, a single representative feature is retained by maximizing its target correlation (via ANOVA F-test score) while minimizing its average inter-feature correlation.

The ten features retained through this supervised selection process constitute the TSAD-based component of TAMIS<sub>F</sub>, as detailed in Table II. We evaluate the relevance of TAMIS in Section VI-C. Note that the proposed feature set is not restricted to use within our anomaly detection framework, TAMIS. Our feature set can be integrated into any anomaly detection system, whether supervised or unsupervised. We evaluate the performance of traditional anomaly detection methods applied on $\mathrm { T A M I S } _ { \mathcal { F } }$ in Section VI-B.

## B. Anomaly Detectors Pool

This section describes the two anomaly detectors that compose the core of the TAMIS ensemble. The choice of using only two base detectors is deliberate, as it significantly reduces computational overhead while maintaining high detection accuracy and scalability in production environments.

Let $H _ { i }$ denote the set of historical feature values for a given time series, and let $f _ { i } \in \pmb { F } _ { d }$ represent the new observation under evaluation. Each detector outputs an anomaly score, which is subsequently combined by the supervised ensemble model described in Section V-C.

1) KDE: Kernel Density Estimation-based anomaly detection: The first detector models the distribution of historical feature values using a univariate Gaussian kernel density estimator (KDE) [36]. Anomalies are identified as points with low estimated density under this model.

Formally, let $\hat { d } _ { H _ { i } }$ denote the kernel density estimate fitted on $H _ { i } .$ . To ensure numerical stability and produce a dimensionless score, the estimated density $\hat { d } _ { H _ { i } } ( f _ { i } )$ is rescaled by the empirical standard deviation of the historical data, $\hat { \sigma } ( H _ { i } )$ The resulting anomaly score is defined as follows:

$$
D _ { \mathrm { K D E } } ( f _ { i } , H _ { i } ) = 1 - \hat { d } _ { H _ { i } } ( f _ { i } ) \hat { \sigma } ( H _ { i } )\tag{1}
$$

This normalization preserves the monotonic relationship between the anomaly score and the estimated density (i.e., higher scores indicate rarer events) while compensating for scale differences across heterogeneous time series. Because $D _ { \mathrm { K D E } }$ is a monotone decreasing function of ${ \hat { d } } _ { H _ { i } } ( f _ { i } )$ , ranking by $D _ { \mathrm { K D E } }$ is equivalent to ranking by $- \hat { d } _ { H _ { i } }$

2) ASHES: Anomaly Scoring-based on Historical $E x \mathrm { - }$ tremeS: The second detector, denoted ASHES (Anomaly Scoring based on Historical ExtremeS), quantifies how much a new observation $f _ { i }$ extends beyond the historical data extremes. It combines two complementary components: an extremity rank (K) and a relative amplitude ratio (r), which are integrated into a single anomaly score.

Extremity Rank (K): The extremity rank quantifies the position of $f _ { i }$ within the ordered historical distribution. It is defined as the minimum rank of $f _ { i }$ when inserted into $H _ { i }$ sorted in ascending and descending order. Formally:

$$
\begin{array} { r } { K ( f _ { i } ) = \operatorname* { m i n } \left( \operatorname { r a n k } _ { \mathrm { a s c } } ( f _ { i } , H _ { i } ) , \operatorname { r a n k } _ { \mathrm { d e s c } } ( f _ { i } , H _ { i } ) \right) } \end{array}\tag{2}
$$

A rank of $K = 1$ indicates that $f _ { i }$ is a new global minimum or maximum. As a practical heuristic, when $K > 1 0 .$ the point is considered non-extreme and its anomaly score is set to zero.

Relative Amplitude Ratio (r): To quantify the magnitude of deviation, we compute a relative amplitude ratio $r ( f _ { i } )$ that measures how much $f _ { i }$ expands the range of the filtered historical set. Let $H _ { i _ { k } }$ denote $H _ { i }$ with the $K - 1$ largest and

K − 1 smallest values removed, and let min $\ l _ { K } = \operatorname* { m i n } ( H _ { i _ { k } } )$ ， max $\kappa = \operatorname* { m a x } ( H _ { i _ { k } } )$ . Formally, $r ( f _ { i } )$ is defined as follows:

$$
r ( f _ { i } ) = { \frac { \operatorname* { m a x } _ { K } - \operatorname* { m i n } _ { K } } { \operatorname* { m a x } ( f _ { i } , \operatorname* { m a x } _ { K } ) - \operatorname* { m i n } ( f _ { i } , \operatorname* { m i n } _ { K } ) } }\tag{3}
$$

A value of r close to 1 indicates that $f _ { i }$ lies within the typical range of the filtered extrema, while smaller values correspond to more substantial deviations from the historical amplitude.

ASHES Anomaly Score: The anomaly score of ASHES integrates both $K$ and r to capture the effect of extremity and deviation magnitude. Formally, it is computed as follows:

$$
D _ { \mathrm { A S H E S } } ( f _ { i } , H _ { i } ) = 0 . 5 ^ { K - 1 } - r \times 0 . 5 ^ { K }\tag{4}
$$

This exponentially weighted formulation ensures that points with low ranks (high extremity) and large deviations (small r) receive higher anomaly scores. Consequently, ASHES effectively highlights rare and impactful events that substantially alter the temporal dynamics of the monitored time series.

Overall, while the KDE detector provides a smooth, probabilistic view of data rarity, ASHES focuses on extreme deviations. Their complementary nature (density modeling versus rank-based extremity analysis) motivates their joint use within the supervised ensemble framework of TAMIS.

It is important to note that standard methods like HBOS [37] or LOF [23] treat rarity and amplitude deviation equally. In our industrial context, a value can be rare without being operationally critical if its relative amplitude is low. ASHES is specifically designed to address this by weighing the Extremity Rank (K) against the Relative Amplitude Ratio (r), offering a nuanced detection capability that off-the-shelf algorithms lack.

## C. Supervised Ensemble Anomaly Scores Aggregation

The final stage of the proposed approach introduces SEASA (Supervised Ensemble Anomaly Scores Aggregation), a stacking-based ensemble model that integrates the outputs of multiple base anomaly detectors (i.e., KDE and ASHES) into a unified and more accurate prediction. This ensemble mechanism leverages supervised learning to automatically infer optimal weightings and interactions among detector outputs, thereby enhancing overall robustness.

1) Base Anomaly Score Generation: For each daily time series window (i.e., day d), a feature vector $\pmb { F } _ { d }$ is computed using $\mathrm { T A M I S } _ { \mathcal { F } }$ feature set. This representation serves as input to the two independent detectors, KDE and ASHES, which each produce a scalar anomaly score reflecting the degree of abnormality for that day. As mentioned in the previous section, these scores capture complementary aspects of the data distribution. The resulting base scores constitute the input features for the SEASA meta-learning stage.

2) Meta-Learner and Prediction Strategy: The second level of SEASA is a meta-learner M designed to combine the raw anomaly scores from the base detectors into a final anomaly prediction. In practice, we employ an XGBoost classifier [35] as M. Formally, given base detector outputs $\{ D _ { \mathrm { K D E } } ( f _ { i } ) , D _ { \mathrm { A S H E S } } ( f _ { i } ) \}$ and corresponding labels $y _ { i } \in$ {0, 1}, the meta-learner learns a mapping as follows:

$$
\hat { y } _ { i } = \mathcal { M } \big ( D _ { \mathrm { K D E } } ( f _ { i } ) , D _ { \mathrm { A S H E S } } ( f _ { i } ) \big )\tag{5}
$$

where $\hat { y } _ { i }$ denotes the final anomaly probability. This supervised aggregation strategy allows the ensemble to adaptively weight detector contributions according to their reliability across different anomaly types.

3) Experimental setup: To obtain an unbiased prediction for every point in our dataset, we utilize a Leave-One-Out Cross-Validation (LOOCV) procedure. For each point being evaluated, the XGBoost model is trained on all other labeled days and then makes a prediction on the single point that was left out. This operation is repeated for the entire dataset, ensuring that every prediction is made on data not seen during the training of that specific model instance. In deployment, the SEASA model is trained once using the available labeled dataset and then applied to new daily measurements.

## VI. EXPERIMENTS

This section presents a comprehensive evaluation of our proposed approach, TAMIS, which addresses the research questions outlined in Section IV, and is organized as follows:

• Overall evaluation: We begin by assessing the global performance of TAMIS (accuracy and throughput) compared to a comprehensive set of baselines. Within this overall evaluation, we analyze different strategies for combining anomaly detectors, focusing on whether it is more advantageous to learn a selection model or to use an ensemble approach. This experiment addresses R1.

• Impact of the feature set: We then investigate how different feature representations affect performance (i.e., accuracy and efficiency). We compare TAMIS<sub>F</sub>, against two widely used alternatives, TSFRESH [21] and CATCH22 [27]. This experiment addresses R2.

• Out-of-distribution and transfer evaluation: Finally, we evaluate the ability of TAMIS to generalize across different types of energy production systems. This experiment addresses R3.

## A. Experimental Setup

All experiments are conducted on a single node with two Intel Xeon Gold 6234 CPUs at 3.30 GHz. Each node has 16 physical cores and 384 GB of RAM. For reproducibility, we make our implementation publicly available <sup>1</sup>.

1) Datasets: In our industrial context, time series are highly diverse in nature and come from two areas: thermal and hydraulic, from EDF’s thermal and hydraulic production facilities, respectively. This diversity leads to a wide variety of behaviors. Moreover, several manual annotation campaigns have been conducted to obtain high-quality labels, resulting in a total of $^ { 6 , 9 3 1 }$ annotated days with 376 anomalies. We divide our benchmark in the three datasets described in Table III. Overall, these datasets are as follows:

<table><tr><td>Characteristics</td><td>JO</td><td>HYDRAU</td><td>THERM</td></tr><tr><td>Number of sensors</td><td>3198</td><td>1740</td><td>4515</td></tr><tr><td>Total number of measurements</td><td>219M</td><td>267M</td><td>103M</td></tr><tr><td>Number of Labeled days</td><td>5813</td><td>540</td><td>578</td></tr><tr><td>NaN ratio</td><td>20.3%</td><td>3.5%</td><td>27.9%</td></tr><tr><td>Total number of NaN</td><td>55334</td><td>640</td><td>7827</td></tr><tr><td>Number of Anomalies</td><td>307</td><td>57</td><td>12</td></tr></table>

TABLE III: Datasets characteristics. NaN ratio: number of labeled days with at least one NaN value.

THERM dataset: contains time series and labeled windows from the thermal perimeter. More precisely, time series corresponds to (i) marginal costs, penalties, startup, and operating costs $( \bar { \in } \mathrm { M W } ^ { - 1 }$ or AC); (ii) demand, reference, and optimized programs (MW); (iii) operating points for thermal units (MW, gradient, number of modulations); and (iv) gas volumes (m<sup>3</sup>). In total, this set contains 4515 individual sensors.

• HYDRAU dataset: contains time series and labeled windows from the hydraulic perimeter. More precisely, time series corresponds to (i) minimum and maximum volumes and trajectories of reservoirs (hm<sup>3</sup>); (ii) minimum, maximum, and turbine flow rates of power plants $( \mathrm { m ^ { 3 } s ^ { - 1 } } )$ ; and (iii) minimum, maximum, and scheduled power output of power plants (MW). In total, this set contains 1740 individual sensors.

• JO dataset: contains both time series and labeled windows from thermal and hydraulic perimeters; the intersection of the JO, THERM, and HYDRAU is empty. The aim of JO is to represent the initial training database for monitoring systems at EDF. In total, this set contains 3198 individual sensors.

While existing benchmarks [11], [13], [30] have driven significant progress, our dataset introduces distinct industrial challenges that are under-represented in the literature:

• Data Quality and Semantic NaNs: Unlike curated benchmarks where missing values are rare, cleaned, or artificially imputed, our dataset retains the original data quality issues inherent to production environments. As shown in Table III, missing values occur at a maximum of 27.9% of the labeled days.

• Forecast Validation Task: Most public benchmarks focus on monitoring continuous raw sensor streams for retrospective anomaly detection. In contrast, our dataset addresses the validation of short-term production forecasts. This task involves comparing a new value against historical profiles to anticipate future failures, a structure distinct from standard stream monitoring.

• Physical Heterogeneity: Spanning both thermal and hydraulic domains, our dataset covers a wider range of physical behaviors (e.g., flow rates, temperatures) and economic variables (e.g., marginal costs).

Finally, we release an anonymized version of our datasets <sup>2</sup>. To the best of our knowledge, this collection is the most extensive dataset of energy production time series for anomaly detection, making it a significant contribution of our work. This release aims to encourage future research in this direction.

2) Baselines: We compare our proposed approach, TAMIS, against several categories of existing methods. We begin by evaluating TAMIS against raw-based detectors, i.e., approaches that operate directly on subsequences of the time series of interest. Next, we compare TAMIS to feature-based detectors that use the same feature set as our solution. Finally, we benchmark TAMIS against a set of automatic solutions. Overall, the baselines considered are the following:

• Raw-based Detectors: We consider three representative methods: One-Class SVM [24] (OCSVM), Histogram-Based Outlier Score [37] (HBOS), and Local Outlier Factor [23] (LOF). These baselines are chosen based on the industrial constraints (cf. Sec II-A), and for their ability to operate on raw time series or extracted features.

• Feature-based Detectors: We employ the same three methods (OCSVM, HBOS, and LOF) applied to features. Additionally, we compare with KDE and our proposed feature-based method, ASHES (cf. Sec V-B). As each detector outputs a vector of scores (i.e., one for each feature), the best aggregation method, among min, max, mean, and the meta-learner M (cf. Sec V-C), is applied.

• Automatic solutions: As individual detectors struggle to remain robust on heterogeneous datasets (cf. Sec III-C). We compare TAMIS with two automatic methods, including an unsupervised average ensemble (Avg Ens) and the model selection strategy (MS) introduced in [29].

None of the baselines natively handles NaNs. Thus, we impute missing values using both forward and backward fill methods. This strategy is motivated by three considerations: (i) forward/backward filling maintains local temporal consistency, leading to more reliable anomaly scores; (ii) propagating the last (or next) observed value retains the piecewise-constant nature of the signal; and (iii) the imputed values remain within the true range of the observed data.

Finally, we evaluate several variants of our proposed approach. First, we consider a version of TAMIS that incorporates a larger set of feature-based detectors (referred to as TAMIS (ALL FEATURES)), combining OCSVM, HBOS, LOF, KDE, and ASHES. We also examine a more comprehensive configuration combining both feature-based and raw-based detectors (denoted TAMIS (ALL FEATURES AND RAW)), which, on top of the previously mentioned detectors, includes HBOS, OCSVM, and LOF on raw subsequences.

3) Evaluation Measures: Anomalies being rare, the Area Under the ROC curve (AUC-ROC) tends to overestimate the accuracy of detectors [30]. We thus consider the area under the Precision-Recall curve (AUC-PR). Beyond accuracy, industrial constraints require low execution time. Thus, we use throughput to measure efficiency and scalability. Note that more recent and robust evaluation measures exist, such as Volume Under the Surface (VUS) [14]. Nevertheless, our problem is to determine whether a given day D is abnormal (the score produced by our approach is a single value S ∈ R). Therefore, our problem is not affected by potential misalignment, justifying the use of VUS.

![](images/55dff8f050578fbf0471ab16c76c071fe6460725a97a4da1f25b292649c9a10f.jpg)  
Fig. 6: Precision-Recall curves of TAMIS against (a) detectors on raw data; (b) detectors on $\mathrm { T A M I S } _ { \mathcal { F } }$ feature set; (c) Automatic solutions (Model Selection and Ensembling). (d) Throughput versus AUC-PR for TAMIS against most competitive baselines.

<table><tr><td>Methods</td><td>JO</td><td>HYDRAU</td><td>THERM</td></tr><tr><td colspan="4">Detectors on raw time series</td></tr><tr><td>HBOS</td><td>0.069</td><td>0.297</td><td>0.041</td></tr><tr><td>LOF</td><td>0.228</td><td>0.226</td><td>0.072</td></tr><tr><td>OCSVM</td><td>0.138</td><td>0.143</td><td>0.028</td></tr><tr><td colspan="4">Detectors on (TAMISF features)</td></tr><tr><td>HBOS</td><td>0.656</td><td>0.596</td><td>0.191</td></tr><tr><td>OCSVM</td><td>0.579</td><td>0.527</td><td>0.156</td></tr><tr><td>LOF</td><td>0.684</td><td>0.523</td><td>0.452</td></tr><tr><td>KDE</td><td>0.764</td><td>0.554</td><td>0.462</td></tr><tr><td>ASHES</td><td>0.792</td><td>0.571</td><td>0.229</td></tr><tr><td colspan="4">Automatic Solutions (on TAMISF features)</td></tr><tr><td>Average Ensemble</td><td>0.147</td><td>0.209</td><td>0.033</td></tr><tr><td>Model Selection</td><td>0.790</td><td>0.569</td><td>0.157</td></tr><tr><td>TAMIS (ALL FEATURES)</td><td>0.851</td><td>0.683</td><td>0.486</td></tr><tr><td>TAMIS (ALL FEATURES AND RAW)</td><td>0.856</td><td>0.691</td><td>0.522</td></tr><tr><td>TAMIS</td><td>0.837</td><td>0.585</td><td>0.555</td></tr></table>

TABLE IV: Accuracy (AUC-PR) of TAMIS and baselines applied on raw data and $\mathrm { T A M I S } _ { \mathcal { F } }$ features.

## B. Overall Evaluation

In this section, we evaluate the global performance of TAMIS relative to all baseline configurations. The results are summarized in Table IV and Figure 6.

Raw time series as input: The first block of Table IV demonstrates that directly applying detectors to raw time series leads to poor accuracy. Across all datasets, AUC-PR values of LOF (i.e., the most accurate raw-based detector) remain below 0.23 on JO and HYDRAU, and drop to approximately 0.07 on THERM. This confirms that unprocessed time series lack the discriminative structure required for effective anomaly detection in electrical production systems.

Feature-based detection: Performances increase when detectors operate on TAMIS<sub>F</sub>. Among individual detectors, ASHES achieves the highest AUC-PR on JO (0.792), while maintaining strong results on HYDRAU (0.571) and THERM (0.229). This confirms that the TAMIS feature set provides a more informative representation than raw subsequences.

Automatic aggregation approaches: As shown in Table IV, supervised ensemble methods (i.e., SEASA) significantly outperform both unsupervised ensembles and supervised model selection. The largest ensemble (i.e., TAMIS (ALL FEATURES AND RAW)) achieves the best overall accuracy (AUC-PR = 0.856), surpassing the best individual detector (ASHES) by +6.40 points. These results highlight the strong complementarity between detectors and the benefits of SEASA.

Accuracy–throughput trade-off: However, Figure 6 reveals that this accuracy improvement comes at the cost of computational efficiency. The largest ensemble has the lowest throughput because of its high inference cost. In contrast, the proposed TAMIS (i.e., using only two detectors, namely KDE and ASHES) achieves an optimal balance between accuracy and scalability. Specifically, it delivers near-best accuracy while being 2.4× faster, establishing TAMIS as the best tradeoff solution for large-scale deployment.

Answer to R1: Combining heterogeneous detectors improves anomaly detection accuracy, especially when integrating both raw-data-based andfeature-based approaches. Supervised ensembles outperform unsupervised and selection-based strategies, while the proposed TAMIS configuration achieves the best accuracy–efficiency trade-off with only two detectors.

![](images/2863d32c19d6ea1c1867eb741765c1b01a61e8858b1bc192dd525acd2f884815.jpg)  
(a) Feature Computation versus Accuracy

![](images/e1a4d0a92e701e678f0e3d5b954832c2a9bffad1b12cff1d12c75b0331231dbe.jpg)  
(b) Detectors computation versus Accuracy

![](images/5467cfcaaecb99114b0f7a8f3d349d66573b8f5561bc79d2e118783223964f22.jpg)  
(c) weight Inference versus Accuracy

Fig. 7: Comparision of AUC-PR versus throughput between TAMIS using TAMIS , CATCH22 and TSFRESH as feature sets. The throughput corresponds to (a) Features computation, (b) Detectors computation, and (c) weights inference.  
![](images/581a9885f31569df45ed78e6231d9b39dc9f5e50fdd97207db6fd21ee097d210.jpg)

![](images/df7509a611c38961c8b7728527b9caecbf931a8525437e887a6827aeee13326b.jpg)  
Fig. 8: Comparison of our proposed approach TAMIS versus baselines on different models in-distribution (trained and tested on JO dataset) and two out-of-distribution scenarios. Note that all f. and r. stands for all features and raw.

## C. Evaluating the Relevance of the Feature Sets

Figure 7 presents the trade-off between accuracy (AUC-PR) and throughput across the three stages of the TAMIS pipeline: (a) feature computation, (b) detector execution, and (c) weight inference. Three feature sets are compared: TAMIS<sub>F</sub>, CATCH22, and TSFRESH.

Features computation: Across all datasets, TAMIS<sub>F</sub> provides the best trade-off between accuracy and efficiency. It reaches the highest AUC-PR values (≈ 0.8–0.9) while sustaining throughputs several times higher than TSFRESH and comparable to CATCH22. While CATCH22 remains computationally lightweight, its AUC-PR saturates around 0.15, revealing limited expressiveness for anomaly detection in electrical production systems. Conversely, TSFRESH reaches high accuracy, but at a prohibitively high computational cost.

Pipeline-level efficiency: Figure 7(b) and (c) further demonstrate that the efficiency advantage of TAMIS persists through both the detector computation and ensemble weight inference stages. When all processing stages are considered, TAMIS<sub>F</sub> maintains high accuracy and throughput.

Answer to R2: TAMIS<sub>F</sub> feature set achieves the best accuracy–throughput balance among existing alternatives. It offers near-state-of-the-art detection accuracy at a significantly lower computational cost.

## D. Toward Production: an Out-of-Distribution test

To assess the robustness and generalization capability of the proposed approach, we compare its performance under in-distribution (ID) and out-of-distribution (OOD) conditions. A baseline model is trained and tested on JO, representing the ID setting, while OOD performance was evaluated by applying the same model to the HYDRAU and THERM datasets, which differ in operational regimes and signal characteristics. Figure 8 reports the AUC-PR results for three evaluation configurations: (i) in-distribution (trained and tested on JO), (ii) OOD 1 (JO → HYDRAU), and (iii) OOD 2 (JO → THERM). We include the best-performing individual detectors (KDE and ASHES) as baselines.

In-distribution performance. All methods perform strongly (AUC-PR ≈ 0.8–0.9) in ID settings, demonstrating that both individual detectors and TAMIS successfully capture the statistical structure of the in-distribution data. These results establish a reference for subsequent OOD comparisons.

Cross-domain transfer: JO → HYDRAU. In the first OOD scenario, performance decreases across all methods. KDE shows substantial degradation (AUC-PR ≈ 0.2), indicating sensitivity to distributional shifts. In contrast, TAMIS main tains a significantly higher AUC-PR (> 0.5), underscoring its superior ability to generalize across domains with differing noise levels and dynamic properties. Note that in such a scenario, ASHES also shows strong performances, but still suffers from a larger drop than TAMIS between ID and OOD. Cross-domain transfer: JO → THERM. The second transfer scenario, from JO to THERM, is more challenging. All models experience additional performance loss, but TAMIS again remains the most robust, consistently outperforming KDE and ASHES by a large margin. This result highlights that the ensemble and feature-aggregation mechanisms within TAMIS effectively mitigate domain-specific overfitting.

Relative degradation analysis. The right-hand plots in Figure 8 quantify the relative AUC-PR loss between ID and OOD conditions. TAMIS exhibits only a ∼34% decrease when tested on THERM, whereas ASHES experiences a 53% drop. These findings confirm that the ensemble-based formulation of TAMIS acts as a regularizer, maintaining higher predictive stability under data distribution shifts.

Answer to R3: TAMIS exhibits strong robustness under out-of-distribution conditions, retaining higher accuracy than individual detectors while maintaining low computational cost. The latter confirms the applicability of TAMIS in production, where distribution drift and heterogeneity across time series are expected.

## VII. OPERATIONAL INSIGHTS AND EXTENSIONS

This section discusses the operational feedback obtained from applying our approach in practice and outlines extensions toward more general benchmarks.

## A. Operational feedback

The deployment of our proposed approach, TAMIS, provided valuable insights into the human-in-the-loop requirements for industrial anomaly detection:

Mitigating Alert Fatigue: Before the introduction of our proposed approach, the main challenge was the volume of alarms. By introducing a ranking mechanism (based on a threshold strategy), TAMIS significantly reduced alert fatigue. Experts reported that the daily report enables more efficient triage by focusing only on the most critical deviations.

The Value of Interpretability: While deep learning approaches often act as black boxes, the choice of explicit features in $T A M I S _ { \mathcal { F } }$ was critical for adoption. Operational teams validated that these features serve as immediate proxies for physical root causes, enabling faster decision-making.

False Positives and Distribution Shifts: The majority of false positives are triggered by Out-of-Distribution (OOD) events. However, our experiments in Figure 8 demonstrate that TAMIS effectively reduces these errors compared to individual detectors, showing a relative resilience that is crucial for maintaining trust over time.

From Generic to Tailored Monitoring: A common industrial hurdle is the cold start problem, where historical labels are unavailable for a new production unit. Our deployment strategy leverages TAMIS’s transferability to address this. The system is initially deployed with a generic model pre-trained on available corporate data. The visualization interface then serves a dual purpose: (i) it supports daily decision-making and (ii) acts as a data annotation tool. As experts validate or reject alerts in the daily reports, they progressively build a high-quality, domain-specific labeled dataset. This feedback loop allows TAMIS to be iteratively retrained, refining the system’s sensitivity to domain-specific characteristics.

![](images/fed132bd0a4476deec14e93077acac5be6a646d3f9fb56e61f556ff0dae0c796.jpg)  
Fig. 9: TAMIS vs Best Detector, Avg Ens, the Best Model Selection strategy and the theoretical Oracle on TSB-UAD [11].

## B. Generalization on public Benchmarks

While TAMIS is designed to address specific industrial constraints (e.g., missing values, heterogeneity), it is crucial to verify that its performance is not solely due to the specificity of our use case and datasets. To assess the generalizability of our approach, we evaluated TAMIS on TSB-UAD [11], a comprehensive, heterogeneous benchmark across domains. We compared TAMIS against the strongest baselines identified in a recent evaluation of model selection for time series anomaly detection [29]. More specifically, we consider the Best Detector (i.e., NormA), the Unsupervised Average Ensemble, and the Best Model Selection (Best MS) strategy identified in [29].

Figure 9 reports the Volume Under the Surface (VUS-PR) accuracy [14] of TAMIS versus the baselines on TSB-UAD. We observe that TAMIS (VUS-PR ≈ 0.38) significantly outperforms the Average Ensemble and the Best Detector. Moreover, TAMIS outperforms the Best Model Selection strategy (VUS-PR≈ 0.35). This result confirms that TAMIS captures fundamental anomaly characteristics that generalize well to diverse domains beyond energy production.

## VIII. CONCLUSION

This paper introduces TAMIS, a lightweight and interpretable anomaly detection system tailored for large-scale industrial time-series monitoring. Through extensive experiments, we addressed three core research questions: detector combination, feature-space design, and cross-domain robustness. More specifically, we demonstrate that TAMIS, effectively balances accuracy and efficiency, with the compact twodetector configuration achieving the best trade-off for scalable deployment, while maintaining strong performances in outof-distribution conditions. Moreover, the proposed feature set, TAMIS<sub>F</sub>, outperforms existing alternatives such as CATCH22 and TSFRESH, delivering a superior accuracy–throughput balance. To encourage future research in this direction, we release an anonymized version of our benchmark. The latter is among the most extensive datasets of energy production time series for anomaly detection. Finally, future work will focus on: (i) extending TAMIS to online learning; (ii) adapting TAMIS to multivariate time series.

[1] T. Palpanas and V. Beckmann, “Report on the first and second interdisciplinary time series analysis workshop (itisa),” SIGMOD Rec., vol. 48, no. 3, p. 36–40, Dec. 2019.

[2] K. Uehara and M. Shimada, “Extraction of primitive motion and discovery of association rules from human motion data,” Progress in Discovery Science: Final Report of the Japanese Dicsovery Science Project, pp. 338–348, 2002.

[3] M. Bach-Andersen, B. Rømer-Odgaard, and O. Winther, “Flexible nonlinear predictive models for large-scale wind turbine diagnostics,” Wind Energy, vol. 20, no. 5, pp. 753–764, 2017.

[4] P. Boniol, M. Meftah, E. Remy, B. Didier, and T. Palpanas, “dcnn/dcam: anomaly precursors discovery in multivariate time series with deep convolutional neural networks,” Data-Centric Engineering, vol. 4, p. e30, 2023.

[5] A. Petralia, P. Boniol, P. Charpentier, and T. Palpanas, “ Few Labels are All You Need: A Weakly Supervised Framework for Appliance Localization in Smart-Meter Series ,” in 2025 IEEE 41st International Conference on Data Engineering (ICDE). Los Alamitos, CA, USA: IEEE Computer Society, May 2025, pp. 4386–4399. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/ICDE65448.2025.00329

[6] P. Boniol and T. Palpanas, “Series2graph: Graph-based subsequence anomaly detection for time series,” Proc. VLDB Endow., vol. 13, no. 12, p. 1821–1834, Jul. 2020.

[7] S. Schmidl, P. Wenig, and T. Papenbrock, “Anomaly detection in time series: a comprehensive evaluation,” Proc. VLDB Endow., vol. 15, no. 9, p. 1779–1797, May 2022. [Online]. Available: https://doi.org/10.14778/3538598.3538602

[8] P. Boniol, J. Paparrizos, T. Palpanas, and M. J. Franklin, “SAND: streaming subsequence anomaly detection,” Proc. VLDB Endow., vol. 14, no. 10, pp. 1717–1729, 2021.

[9] P. Boniol, Q. Liu, M. Huang, T. Palpanas, and J. Paparrizos, “Dive into time-series anomaly detection: A decade review,” 2024. [Online]. Available: https://arxiv.org/abs/2412.20512

[10] J. Paparrizos, P. Boniol, Q. Liu, and T. Palpanas, “Advances in time-series anomaly detection: Algorithms, benchmarks, and evaluation measures,” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, ser. KDD ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 6151–6161. [Online]. Available: https://doi.org/10.1145/3711896.3736565

[11] J. Paparrizos, Y. Kang, P. Boniol, R. S. Tsay, T. Palpanas, and M. J. Franklin, “Tsb-uad: an end-to-end benchmark suite for univariate time-series anomaly detection,” Proc. VLDB Endow., vol. 15, no. 8, p. 1697–1711, Apr. 2022. [Online]. Available: https://doi.org/10.14778/3529337.3529354

[12] P. Boniol, J. Paparrizos, Y. Kang, T. Palpanas, R. S. Tsay, A. J. Elmore, and M. J. Franklin, “Theseus: navigating the labyrinth of time-series anomaly detection,” Proc. VLDB Endow., vol. 15, no. 12, p. 3702–3705, Aug. 2022. [Online]. Available: https://doi.org/10.14778/3554821.3554879

[13] P. Wenig, S. Schmidl, and T. Papenbrock, “Timeeval: a benchmarking toolkit for time series anomaly detection algorithms,” Proc. VLDB Endow., vol. 15, no. 12, p. 3678–3681, Aug. 2022. [Online]. Available: https://doi.org/10.14778/3554821.3554873

[14] J. Paparrizos, P. Boniol, T. Palpanas, R. S. Tsay, A. Elmore, and M. J. Franklin, “Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection,” Proc. VLDB Endow., vol. 15, no. 11, p. 2774–2787, Jul. 2022. [Online]. Available: https://doi.org/10.14778/3551793.3551830

[15] M. Sakurada and T. Yairi, “Anomaly detection using autoencoders with nonlinear dimensionality reduction,” in Proceedings of the MLSDA 2014 2nd Workshop on Machine Learning for Sensory Data Analysis, ser. MLSDA’14, 2014, p. 4–11.

[16] P. Malhotra, L. Vig, G. M. Shroff, and P. Agarwal, “Long Short Term Memory Networks for Anomaly Detection in Time Series,” in ESANN, 2015.

[17] E. M. Knorr and R. T. Ng, “Algorithms for mining distance-based outliers in large datasets,” in Proceedings of the 24rd International Conference on Very Large Data Bases, ser. VLDB ’98. San Francisco, CA, USA: Morgan Kaufmann Publishers Inc., 1998, p. 392–403.

[18] F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation forest,” in 2008 Eighth IEEE International Conference on Data Mining, 2008, pp. 413–422.

[19] S. Tafazoli, Y. Lu, R. Wu, T. V. A. Srinivas, H. Dela Cruz, R. Mercer, and E. Keogh, “C22mp: the marriage of catch22 and the matrix profile creates a fast, efficient and interpretable anomaly detector,” Knowl. Inf. Syst., vol. 66, no. 8, p. 4789–4823, May 2024. [Online]. Available: https://doi.org/10.1007/s10115-024-02107-5

[20] A. Bonifati, F. D. Buono, F. Guerra, and D. Tiano, “Time2feat: Learning interpretable representations for multivariate time series clustering,” Proc. VLDB Endow., vol. 16, no. 2, pp. 193–201, 2022.

[21] M. Christ, N. Braun, J. Neuffer, and A. W. Kempa-Liehr, “Time series feature extraction on basis of scalable hypothesis tests (tsfresh – a python package),” Neurocomput., vol. 307, no. C, p. 72–77, Sep. 2018. [Online]. Available: https://doi.org/10.1016/j.neucom.2018.03.067

[22] R. Chalapathy and S. Chawla, “Deep learning for anomaly detection: A survey,” CoRR, vol. abs/1901.03407, 2019. [Online]. Available: http://arxiv.org/abs/1901.03407

[23] M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, “Lof: identifying density-based local outliers,” SIGMOD Rec., vol. 29, no. 2, p. 93–104, May 2000. [Online]. Available: https://doi.org/10.1145/335191.335388

[24] B. Scholkopf, R. Williamson, A. Smola, J. Shawe-Taylor, and J. Platt,¨ “Support vector method for novelty detection,” in Proceedings of the 13th International Conference on Neural Information Processing Systems, ser. NIPS’99. Cambridge, MA, USA: MIT Press, 1999, p. 582–588.

[25] “http://iops.ai/dataset detail/?id=10.”

[26] B. Fulcher and N. Jones, “hctsa : A computational framework for automated time-series phenotyping using massive feature extraction,” Cell Systems, vol. 5, 11 2017.

[27] C. H. Lubba, S. S. Sethi, P. Knaute, S. R. Schultz, B. D. Fulcher, and N. S. Jones, “catch22: Canonical time-series characteristics: Selected through highly comparative time-series analysis,” Data Min. Knowl. Discov., vol. 33, no. 6, p. 1821–1852, Nov. 2019. [Online]. Available: https://doi.org/10.1007/s10618-019-00647-x

[28] H. A. Dau, A. Bagnall, K. Kamgar, C.-C. M. Yeh, Y. Zhu, S. Gharghabi, C. A. Ratanamahatana, and E. Keogh, “The ucr time series archive,” 2019. [Online]. Available: https://arxiv.org/abs/1810.07758

[29] E. Sylligardos, P. Boniol, J. Paparrizos, P. Trahanias, and T. Palpanas, “Choose wisely: An extensive evaluation of model selection for anomaly detection in time series,” Proc. VLDB Endow., vol. 16, no. 11, p. 3418–3432, Jul. 2023. [Online]. Available: https: //doi.org/10.14778/3611479.3611536

[30] Q. Liu and J. Paparrizos, “The elephant in the room: towards a reliable time-series anomaly detection benchmark,” in Proceedings of the 38th International Conference on Neural Information Processing Systems, ser. NIPS ’24. Red Hook, NY, USA: Curran Associates Inc., 2025.

[31] Y. Zhao, R. A. Rossi, and L. Akoglu, “Automating outlier detection via meta-learning,” 2021. [Online]. Available: https://arxiv.org/abs/2009. 10606

[32] M. Goswami, C. Challu, L. Callot, L. Minorics, and A. Kan, “Unsupervised model selection for time-series anomaly detection,” 2023. [Online]. Available: https://arxiv.org/abs/2210.01078

[33] Y. Benjamini and Y. Hochberg, “Controlling the false discovery rate: A practical and powerful approach to multiple testing,” Journal of the Royal Statistical Society. Series B (Methodological), vol. 57, no. 1, pp. 289–300, 1995. [Online]. Available: http://www.jstor.org/stable/2346101

[34] R. A. Fisher, “Statistical methods for research workers,” in Breakthroughs in statistics: Methodology and distribution. Springer, 1970, pp. 66–70.

[35] T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting system,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ser. KDD ’16. ACM, Aug. 2016, p. 785–794. [Online]. Available: http: //dx.doi.org/10.1145/2939672.2939785

[36] E. Parzen, “On Estimation of a Probability Density Function and Mode,” The Annals of Mathematical Statistics, vol. 33, no. 3, pp. 1065 – 1076, 1962. [Online]. Available: https://doi.org/10.1214/aoms/1177704472

[37] M. Goldstein and A. R. Dengel, “Histogram-based outlier score (hbos): A fast unsupervised anomaly detection algorithm,” 2012.