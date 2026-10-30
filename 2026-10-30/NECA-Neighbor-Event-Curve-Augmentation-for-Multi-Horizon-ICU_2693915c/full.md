# NECA: Neighbor Event Curve Augmentation for Multi-Horizon ICU Deterioration Prediction

Chenyang Wang<sup>1</sup>, Jiawei Theresa Tang<sup>1</sup>, Xinran Liu<sup>1</sup>, Xinyu Lu<sup>2</sup>, Xulei Yang 3\*

<sup>1</sup>Department of Statistics and Data Science, National University of Singapore, Singapore <sup>2</sup>School of Computing, National University of Singapore, Singapore <sup>3</sup>Institute for Infocomm Research (I<sup>2</sup>R), A\*STAR, Singapore Contact: e1155460@u.nus.edu

Abstract. Early prediction of ICU organ dysfunction events supports timely clinical monitoring and intervention, but estimating short-term organ-specific risk from early ICU records remains challenging due to irregular measurements, heterogeneous clinical variables, and the need to predict risk across multiple future horizons. Existing supervised models usually treat each patient as an isolated sample and do not explicitly use the observed event timing of similar historical patients. We propose Neighbor Event-Curve Augmentation (NECA), a modelagnostic method that retrieves training-set patients with similar early-window profiles and converts their subsequent organ dysfunction event onset times into cumulative event-curve features over 6h, 12h, 24h, and 42h. These features are concatenated with the original patient representation and used by standard downstream predictors. On PhysioNet 2012 and a 4,000-stay MIMIC-IV subset, NECA improves average PR-AUC and ROC-AUC across four organ dysfunction event tasks and seven supervised learners. The largest gain appears for neurological dysfunction events, where task-level average PR-AUC increases from 0.461 to 0.557 on PhysioNet 2012. Ablations show that clinically meaningful retrieval and horizon-specific neighbor risk drive most of the improvement, while the full cumulative curve adds task-dependent temporal context. These results support similar-patient event timing as a structured temporal risk signal for ICU organ dysfunction event prediction. The source code will be publicly released upon acceptance of this paper.

Keywords: ICU Prediction · Electronic Health Records · Patient Retrieval · Multihorizon Prediction

## 1 Introduction

Early recognition of organ dysfunction in intensive care units (ICUs) is a central problem in biomedical and health informatics. Machine-learning early warning systems, such as TREWS for sepsis, show that timely risk identification can support earlier intervention and improve outcomes [1]. ICU patients are monitored through electronic health records (EHRs), irregular physiological measurements, laboratory tests, and clinical interventions, producing rich but heterogeneous time-series data. However, many bedside decisions are not only about whether a patient is high risk, but also about which organ dysfunction event is likely to be observed next and within what time scale [2, 3]. In this work, EHR-observed organ dysfunction events are the first newly observed post-window task-specific clinical abnormalities, such as renal dysfunction, neurological dysfunction, electrolyte/metabolic abnormality, or inflammatory abnormality, according to predefined EHR measurement criteria.

Existing ICU prediction methods usually learn a supervised mapping from the current patient record to a target outcome. Logistic regression, gradient-boosted trees, TabPFN, and temporal models such as LSTM can estimate future risk from early-window features or irregular clinical time series [4–9]. Stronger temporal encoders can better model sparsity and temporal dependence, but they still usually produce a risk score for a fixed horizon. They do not explicitly use the subsequent trajectories of clinically similar historical patients, and therefore provide limited case-based evidence about whether organ dysfunction events tend to occur within 6, 12, 24, or 42 hours.

Our motivation is to bring case-based temporal evidence into early ICU prediction. Patient-similarity work shows that clinically similar historical patients can inform personalized risk estimation [10, 11]. If similar ICU patients later showed renal dysfunction, neurological dysfunction, inflammatory abnormality, or electrolyte/metabolic abnormality shortly after the early window, their trajectories provide local temporal evidence for the current patient. Instead of only using similar patients as static neighbors or graph nodes, we model when their post-window events appeared. This converts neighbor information from a static risk estimate into a multi-horizon event trajectory that can be used by standard predictors.

We propose Neighbor Event-Curve Augmentation (NECA), a model-agnostic method for early-window multi-horizon organ dysfunction event prediction. For each query patient, NECA retrieves similar training-set patients using only early-window information and summarizes their future event onset times as cumulative event-curve features over 6h, 12h, 24h, and 42h. These features are concatenated with the original patient representation without changing downstream architectures. NECA is temporally valid and interpretable: future information comes only from training-set neighbors, and each augmented feature is the fraction of similar patients who reached the event definition within a future horizon. Because each dimension corresponds to a concrete future horizon, the augmentation can be inspected as a local event-risk summary rather than an opaque latent embedding.

The main contributions are:

– We formulate early ICU organ dysfunction event prediction as a multi-task, multihorizon problem over four EHR-defined event categories within 6h, 12h, 24h, and 42h after an early window.

– We propose NECA, which transforms post-window event onset times of similar training-set patients into cumulative event-curve features used as retrieval-derived temporal risk signals.

– We evaluate four organ dysfunction event tasks and seven supervised learners, showing that event-curve features consistently complement supervised predictors, especially for neurological dysfunction events.

## 2 Related Work

## 2.1 Early ICU Organ Dysfunction Event Prediction

Early ICU prediction has been widely studied using physiological time-series and EHR data. Harutyunyan et al. [12] introduced a Medical Information Mart for Intensive Care III (MIMIC-III) benchmark for mortality, decompensation, length of stay, and phenotype classification. Chen et al. [13] developed multi-task recurrent models for illness severity prediction. ConCare learns personalized clinical feature embeddings with feature-specific temporal encoders and healthcare-context attention [14]. APRICOT-M uses state-space modeling for real-time ICU acuity, stability, mortality, discharge, and therapy prediction [15].

Beyond global acuity and mortality prediction, recent studies have considered organlevel or multi-label organ dysfunction prediction. SeqSNR models six ICU organ dysfunction events using shared multitask representations [16], while IOC-MT uses graph attention for inter-organ correlations [17]. Other work studies early dynamic organsystem risk [18] and multi-label care escalation triggers including hemodynamic instability, renal compromise, and neurological events [19]. These studies motivate prediction at short horizons for specific organ systems. NECA additionally supplies each predictor with recorded outcomes from retrieved historical records.

## 2.2 Retrieval-Augmented and Patient-Similarity Clinical Prediction

Retrieval and knowledge augmentation have also been applied to EHR prediction. MedRetriever retrieves unstructured text for interpretable health risk prediction [20], while RAM-EHR uses dense retrieval over a multi-source clinical corpus to augment local EHR models [21]. GraphCare and KARE incorporate biomedical knowledge graphs or retrieved community summaries [22, 23], and CLEAR demonstrates targeted retrieval for clinical information extraction [24]. These studies motivate retrieval as a prediction component, although their evidence units and downstream tasks difer from NECA’s historical ICU records.

Patient-similarity methods provide cross-patient evidence. Models using related trajectories learn patient representations [25], while similarity graphs support ICU mortality and criticalness prediction [26]. NECA passes four cumulative neighbor event rates to the predictor, making this historical evidence inspectable. Prior work also studies patient retrieval and nearest-neighbor prediction.

## 3 Methods

Figure 1 shows the four steps of NECA. It constructs early-window representations, retrieves similar training records, summarizes their later event times, and adds the resulting event curve to a predictor.

![](images/3b64bcbcf6ca90ed9f44c6a1883cbbc8a436f89b637122eb39cad15ec3248dde.jpg)  
Fig. 1. Overview of the proposed NECA framework for early ICU organ dysfunction event prediction. The framework consists of data preprocessing, neighbor event-curve augmentation, model integration, and multi-horizon output prediction.

## 3.1 Data Sources and Cohort Construction

We evaluate NECA on two ICU time-series datasets. PhysioNet/Computing in Cardiology Challenge 2012 Set A contains 4,000 ICU records with irregular physiological and laboratory measurements [27] (Table 1). We also use a subset of 4,000 ICU stays from MIMIC-IV [28]. Each record represents one ICU stay. The MIMIC-IV loader retains stays of adults aged at least 18 years, with a recorded ICU length of stay of at least 12 hours, from admissions containing exactly one ICU stay. It sorts the stays by identifier and samples 4,000 without replacement using a fixed random seed. A subject is an individual patient and can have records from multiple admissions. We represent a record as

$$
\mathcal { R } _ { i } = \big ( s _ { i } , \{ ( v , t _ { i j } ^ { ( v ) } , x _ { i j } ^ { ( v ) } ) : v \in \mathcal { V } , \ j = 1 , \ldots , n _ { i v } \} \big ) ,\tag{1}
$$

Here $s _ { i }$ denotes static information for record $i , \mathcal { V }$ is the set of clinical variables, and $n _ { i v }$ is the number of observations of variable � in that record. The value $x _ { i j } ^ { ( v ) }$ was recorded at time $t _ { i j } ^ { ( v ) }$ after ICU admission.

In the tables, HR means heart rate, RR means respiratory rate, MAP means mean arterial pressure, and SBP means systolic blood pressure. $\mathrm { S p O } _ { 2 }$ is peripheral oxygen saturation, $\mathrm { P a O } _ { 2 }$ is arterial oxygen partial pressure, and $\mathrm { F i O } _ { 2 }$ is the fraction of inspired oxygen. GCS means Glasgow Coma Scale, BUN means blood urea nitrogen, WBC means white blood cell count, and $\mathrm { H C O } _ { 3 }$ means bicarbonate. Cr and Temp denote creatinine and temperature.

The early observation window spans $T _ { e } = 6$ hours after ICU admission. Measurements within [0, � ] provide prediction features. Later measurements determine event times and labels.

Table 1. Summary of ICU datasets and selected clinical variables.
<table><tr><td>Aspect</td><td>PhysioNet 2012 Set A</td><td>MIMIC-IV subset</td></tr><tr><td>Cohort size</td><td>4,000 ICU records</td><td>4,000 ICU stays</td></tr><tr><td>Time origin</td><td>ICU admission</td><td>ICU admission</td></tr><tr><td>Sampling pattern</td><td>Sparse and irregular</td><td>Sparse and irregular</td></tr><tr><td>Static variables</td><td>Age, gender, height, weight, ICU Age, gender and mapped ICU type</td><td>variables</td></tr><tr><td>High-coverage variables</td><td>HR, temperature, GCS, creatinine, HR, oxygen saturation, GCS, BUN, WBC, bicarbonate, urine respiratory rate, creatinine,</td><td></td></tr><tr><td>Moderate/sparse</td><td>output Lactate, pH,  $\mathrm { P a O } _ { 2 } , \mathrm { F i O } _ { 2 }$ </td><td>BUN, WBC, bicarbonate, platelets , mechani- Urine output, pH, lactate,  $\mathrm { F i O } _ { 2 }$ </td></tr></table>

## 3.2 Multi-Horizon Organ Dysfunction Event Prediction

Let $\mathcal { G }$ contain the four endpoints introduced above. For endpoint $g \in { \mathcal { G } }$ , the rule $A _ { g } ( \cdot )$ equals one when the measurements meet that endpoint’s abnormality threshold. The notation $\mathcal { R } _ { i } [ 0 , t ]$ denotes record � through time �. We define $\tau _ { i } ^ { ( g ) }$ as the time from the end of the early window to the first recorded event for endpoint �.

$$
\tau _ { i } ^ { ( g ) } = \operatorname* { m i n } \{ t - T _ { e } : A _ { g } ( \mathcal { R } _ { i } [ 0 , t ] ) = 1 , t > T _ { e } \} .\tag{2}
$$

If the record meets the event rule during the early window, we exclude it for that endpoint. The resulting prediction concerns newly recorded events. If relevant early measurements are absent, the rule evaluates false and the record can remain eligible. Early absence does not establish normal organ status. The labeler assigns $\tau _ { i } ^ { ( g ) } = \infty$ when the available record contains no qualifying event. This also occurs when relevant follow-up measurements are missing or a stay ends before the prediction horizon. An infinite value therefore does not establish that the patient remained event-free.

For a prediction horizon � ∈ {6, 12, 24, 42} hours, the binary label is defined as

$$
y _ { i , H } ^ { ( g ) } = \mathbb { I } ( \tau _ { i } ^ { ( g ) } \leq H ) .\tag{3}
$$

The final horizon is 42 hours after the early window, reaching 48 hours after ICU admission. The label $y _ { i , H } ^ { ( g ) } = 1$ means that record � has a recorded event for endpoint $g$ within � hours after the early window. We train a separate binary classifier for each endpoint and horizon. The result tables report eligible record counts.

The rules draw on the Sequential Organ Failure Assessment (SOFA), Sepsis-3, National Early Warning Score 2 (NEWS2), Kidney Disease Improving Global Outcomes (KDIGO) guidelines, and systemic inflammatory response syndrome (SIRS) criteria [29–34]. They do not implement any of these complete definitions.

The renal endpoint uses creatinine and BUN. The neurological endpoint uses GCS. The electrolyte/metabolic endpoint uses pH, bicarbonate, and lactate. The inflammatory endpoint uses WBC and temperature.

Table 2. EHR-observed endpoint rules and selected first-six-hour predictor variables.
<table><tr><td>Endpoint</td><td>Recorded event rule</td><td>Early-window variables</td></tr><tr><td>Renal</td><td>Maximum Cr &gt; 1.5 mg/dL or BUN &gt; 30 mg/dL</td><td>Age, gender, minimum MAP, maximum Cr and BUN, last urine output, maximum lactate</td></tr><tr><td>Neurological</td><td>Last  ${ \mathrm { G C S } } \leq 1 2$ </td><td>Age, gender, minimum and last GCS, minimum MAP, maximum HR</td></tr><tr><td>Electrolyte / metabolic</td><td>Minimum  $\mathrm { p H } < 7 . 3 5 ,$  minimum HCO3 &lt; 22, or maximum lactate &gt; 2.0</td><td>Age, gender, maximum lactate, minimum pH and  $\mathrm { H C O } _ { 3 } ,$  maximum Cr and BUN</td></tr><tr><td>Inflammatory</td><td>Maximum WBC &gt; 12 or temperature  $> 3 8 ^ { \circ } \mathrm { C }$ </td><td>Age, gender, maximum WBC, maximum temperature, minimum platelets, maximum lactate</td></tr></table>

Measurement frequency, treatment, and documentation practice can afect whether an event is recorded. Our target is therefore the first newly recorded event that meets an endpoint rule after the early window. Table 2 gives the rules and early-window variables.

## 3.3 Early-Window Feature Representation

For tabular models, each record becomes an endpoint-specific early-window feature vector

$$
\mathbf { x } _ { i } ^ { ( g ) } \in \mathbb { R } ^ { d _ { g } } ,\tag{4}
$$

where $d _ { g }$ is the number of variables selected for endpoint $g .$ The vector contains summary statistics from $[ 0 , T _ { e } ]$ , including first, last, minimum, and maximum values. Missing values are retained and later filled using medians computed from the training set.

For sequence models, we construct an hourly representation from the same earlywindow time series. The interval $[ 0 , T _ { e } ]$ is divided into $T = \lceil T _ { e } \rceil$ hourly bins, and observations within each bin are aggregated by taking the last value, yielding

$$
\mathbf { S } _ { i } \in \mathbb { R } ^ { T \times F } ,\tag{5}
$$

where $F$ is the number of physiological and laboratory channels. LSTM, ConCare, and an APRICOT-M-style state-space model (SSM) use this tensor, so sequence baselines receive the same early-window information in temporally ordered form.

For LSTM, missing values are forward-filled and remaining entries are filled from training-set empirical values before channel-wise standardization. ConCare reorganizes the same tensor into feature-specific temporal streams before healthcare-context attention combines variable-level representations. The adapted state-space model uses the hourly tensor. All sequence-model preprocessing statistics are fitted on the training split only.

## 3.4 Neighbor Event-Curve Augmentation

NECA augments each early-window representation with recorded event times from retrieved neighbors. For query record �, it retrieves up to � training records using only early-window information. We denote this set by $N _ { K } ( i )$

For each neighbor $j \in N _ { K } ( i )$ , the training data provide its recorded event time $\tau _ { j } ^ { \left( g \right) }$ NECA summarizes these times at each prediction horizon � using

$$
\zeta _ { i , H } ^ { ( g ) } = \sum _ { j \in N _ { K } ( i ) } w _ { i j } \mathbb { I } ( \tau _ { j } ^ { ( g ) } \leq H ) ,\tag{6}
$$

Here $w _ { i j }$ is a neighbor weight that sums to one over $N _ { K } ( i )$ . In the main experiment, all retrieved neighbors receive equal weight. If exactly � neighbors are retrieved, $w _ { i j } = 1 / K$ A distance-weighted variant gives closer neighbors more weight. We collect the four rates for endpoint $g$ in the vector

$$
\begin{array} { r } { \mathbf { z } _ { i } ^ { \left( g \right) } = \left[ z _ { i , 6 } ^ { \left( g \right) } , z _ { i , 1 2 } ^ { \left( g \right) } , z _ { i , 2 4 } ^ { \left( g \right) } , z _ { i , 4 2 } ^ { \left( g \right) } \right] . } \end{array}\tag{7}
$$

Each entry is the weighted fraction of retrieved records with an event by its prediction horizon. The entries are nondecreasing because the horizons are cumulative. We call this vector the neighbor event curve.

For tabular models, we append the neighbor event curve to the early-window vector.

$$
\tilde { \mathbf { x } } _ { i } ^ { ( g ) } = \left[ \mathbf { x } _ { i } ^ { ( g ) } ; \mathbf { z } _ { i } ^ { ( g ) } \right] .\tag{8}
$$

For sequence models, we append the neighbor event curve to the encoder’s final hidden representation before prediction. We call a model trained without the event curve Raw and the corresponding augmented model Curve-NECA.

## 3.5 Neighbor Retrieval and Leakage Control

NECA can retrieve neighbors using diferent early-window features. In the reported main experiment, retrieval uses 27 standardized abnormality indicators, Euclidean distance, equal neighbor weights, and $K = 1 0 0$ . This value of � was fixed for the reported results. Table 3 lists the alternative retrieval feature sets used in ablations. Appendix A defines the 27 indicators.

For each random seed and endpoint, we split eligible records approximately 70% for training, 10% for validation, and 20% for testing. When group counts permit, we stratify by recorded event time using 0–6h, 6–12h, 12–24h, 24–48h, and >48h or no recorded event. The split uses records as units, including in MIMIC-IV. Records from diferent admissions of one subject can therefore appear in diferent partitions. Imputation and scaling use only the training partition. Validation and test event curves use outcomes only from training neighbors. For a training query, we exclude its own record from retrieval. Other records from the same subject can remain. Training event curves are not cross-fitted, which would require each training query to retrieve only from other training folds. These choices can make performance optimistic. A subject-disjoint split and cross-fitted training event curves are needed to assess that risk.

Table 3. Retrieval feature presets used by NECA.
<table><tr><td>Preset</td><td>Dim. Description</td><td></td></tr><tr><td>Endpoint-specific</td><td></td><td>6–14 Early-window variables selected for the target endpoint.</td></tr><tr><td>Demographic</td><td>2</td><td>Age and gender.</td></tr><tr><td>Physiological</td><td></td><td>15 Vital-sign and physiological summaries, including SBP, MAP, HR, RR, and SpO2.</td></tr><tr><td>Laboratory</td><td>17</td><td>Laboratory and organ-status summaries, including PaO2, FiO2, temperature, GCS, WBC, Cr, BUN, urine output, lactate, pH, HCO₃, platelets, and bilirubin.</td></tr><tr><td>tors</td><td></td><td>Abnormality indica- 27 Binary and normalized indicators covering hemody- namic, respiratory, renal, inflammatory, hepatic, coag- ulation, shock-like, multi-system, monitoring, and ICU- type signals.</td></tr></table>

## 3.6 Prediction Models

NECA can augment both tabular and sequence predictors. We evaluate seven supervised learners, including logistic regression, XGBoost, LightGBM, TabPFN, LSTM, ConCare, and an APRICOT-M-style SSM. We train a separate binary classifier for each endpoint � and prediction horizon �.

$$
\hat { p } _ { i , H } ^ { ( g ) } = f _ { \theta } ^ { ( g , H ) } ( \tilde { \mathbf { x } } _ { i } ^ { ( g ) } ) ,\tag{9}
$$

Here $f _ { \theta } ^ { ( g , H ) }$ is a classifier with learned parameters $\theta ,$ and $\hat { p } _ { i , H } ^ { ( g ) }$ is its predicted event probability for endpoint $g$ by horizon �.

For LSTM, ConCare, and the APRICOT-M-style SSM, we append the neighbor event curve after the sequence encoder. The resulting prediction is

$$
\begin{array} { r } { \hat { p } _ { i , H } ^ { ( g ) } = \sigma \left( \mathbf { w } ^ { \top } [ \mathbf { h } _ { i } ; \mathbf { z } _ { i } ^ { ( g ) } ] + b \right) , } \end{array}\tag{10}
$$

Here $\mathbf { h } _ { i }$ is the sequence encoder’s final hidden representation, $\mathbf { z } _ { i } ^ { ( g ) }$ is the neighbor event curve, w and � are output-layer parameters, and $\sigma ( \cdot )$ is the sigmoid function. Raw sequence models predict from h<sub>�</sub>. Curve-NECA predicts from $[ \mathbf { h } _ { i } ; \mathbf { z } _ { i } ^ { ( g ) } ]$

ConCare is an attention-based sequence baseline for irregular clinical time series. It learns personalized feature embeddings from feature-specific temporal streams and uses healthcare-context attention to combine clinical variables and static context [14]. We train ConCare separately for each endpoint and horizon using the same early-window tensor and labels as LSTM. Curve-NECA appends the four-dimensional neighbor event curve before ConCare’s final classifier.

The APRICOT-M-style SSM is inspired by a model for real-time ICU acuity and therapy prediction [15]. We adapt its temporal encoder to each endpoint and horizon. This implementation does not reproduce the original APRICOT-M multi-task system. Its Raw version predicts from the early-window sequence. Curve-NECA appends the neighbor event curve to its hidden representation, as with LSTM and ConCare.

Model settings are fixed across Raw and Curve-NECA comparisons. Logistic regression uses balanced classes and � = 1. XGBoost uses depth 5, 900 trees, and learning rate

Table 4. Positive rates (%) of EHR-observed events across prediction horizons.
<table><tr><td>Data</td><td>Endpoint</td><td>N</td><td>6h</td><td>12h 24h</td><td>42h Avg.</td></tr><tr><td rowspan="2">PN12</td><td>Neuro. Renal</td><td>2040 3171</td><td>15.1 20.6 10.1 20.8</td><td>28.6 33.7 27.3 37.3</td><td>24.5 23.9</td></tr><tr><td>Electrolyte Inflam.</td><td>2071 2486</td><td>15.8 23.3 21.4 36.2</td><td>30.9 38.2 47.8 57.8</td><td>27.1 40.8</td></tr><tr><td>MIMIC</td><td>Neuro. Renal Electrolyte 2743 Inflam.</td><td>2650 3401</td><td>7.3 9.4 10.5 17.1 15.3 321.8 198910.1 15.9</td><td>12.2 13.9 23.0 27.4 29.2 33.3 920.824.5</td><td>10.7 19.5 24.9 17.8</td></tr></table>

0.05. LightGBM uses 63 leaves, 1500 trees, and learning rate 0.03. Sequence models use Adam, at most 40 epochs, and early stopping after seven epochs. TabPFN uses its pretrained classifier without tuning.

## 3.7 Evaluation Protocol

We evaluate area under the receiver operating characteristic curve (ROC-AUC) and area under the precision-recall curve (PR-AUC) on validation and test partitions. PR-AUC is particularly relevant because several endpoints are imbalanced (Table 4). Raw and Curve-NECA use the same split for each endpoint, learner, horizon, and seed. Each horizon cell reports mean test performance across seeds. We average across horizons, the seven supervised learners, and seeds to obtain each endpoint-level result. The reported tables do not include paired confidence intervals, so small diferences remain uncertain. In tables, PN12 denotes PhysioNet 2012 and MIMIC denotes the MIMIC-IV subset.

A k-nearest-neighbor (KNN) predictor uses the event rate among the same retrieved training neighbors as a direct reference. On PN12, its mean PR-AUC across horizons is 0.333, 0.355, 0.303, and 0.467 for neurological, renal, electrolyte/metabolic, and inflammatory endpoints, respectively. The tabular Raw models receive a small endpointspecific feature set, while retrieval uses 27 cross-system indicators. The current comparison therefore cannot isolate the efect of neighbor outcomes from the additional input information. An information-matched comparison would evaluate the Raw features with all 27 indicators, the indicators alone, and the same combined inputs with NECA. The reported experiments do not include these controls.

## 4 Results

## 4.1 Overall Performance of NECA

Table 5 reports endpoint-level results on PhysioNet 2012. Curve-NECA has higher mean PR-AUC and ROC-AUC than Raw across all four endpoints. The largest mean diference appears for neurological events, where PR-AUC changes from 0.461 to 0.557 and ROC-AUC from 0.689 to 0.766. The mean gains do not establish gains for every learner. The input mismatch described above also limits attribution to neighbor outcomes or event timing.

Table 5. Mean test performance across four horizons and seven supervised learners. PN12 is PhysioNet 2012; MIMIC is the 4,000-stay MIMIC-IV subset.
<table><tr><td rowspan="2">Dataset Endpoint</td><td rowspan="2"></td><td colspan="2">PR-AUC</td><td colspan="2">ROC-AUC</td></tr><tr><td>Raw</td><td>Curve-NECA</td><td>Raw</td><td>Curve-NECA</td></tr><tr><td rowspan="4">PN12</td><td>Neurological</td><td>0.461</td><td>0.557</td><td>0.689</td><td>0.766</td></tr><tr><td>Renal</td><td>0.353</td><td>0.387</td><td>0.711</td><td>0.740</td></tr><tr><td>Electrolyte/metabolic</td><td>0.357</td><td>0.378</td><td>0.671</td><td>0.692</td></tr><tr><td>Inflammatory</td><td>0.514</td><td>0.538</td><td>0.692</td><td>0.727</td></tr><tr><td rowspan="4">MIMIC</td><td>Neurological</td><td>0.419</td><td>0.520</td><td>0.749</td><td>0.809</td></tr><tr><td>Renal</td><td>0.319</td><td>0.345</td><td>0.668</td><td>0.693</td></tr><tr><td>Electrolyte/metabolic</td><td>0.340</td><td>0.357</td><td>0.662</td><td>0.674</td></tr><tr><td>Inflammatory</td><td>0.276</td><td>0.289</td><td>0.679</td><td>0.696</td></tr></table>

Table 6. Neighbor retrieval ablation on the neurological endpoint. Values are validation-set averages across learners and horizons.
<table><tr><td>Retrieval strategy</td><td>Avg. PR-AUC Avg. ROC-AUC</td><td></td></tr><tr><td>Random neighbors</td><td>0.446</td><td>0.666</td></tr><tr><td>Demographic variables</td><td>0.338</td><td>0.561</td></tr><tr><td>Endpoint-specific variables</td><td>0.385</td><td>0.615</td></tr><tr><td>Physiological variables</td><td>0.416</td><td>0.678</td></tr><tr><td>Laboratory variables</td><td>0.534</td><td>0.727</td></tr><tr><td>Abnormality indicators</td><td>0.550</td><td>0.744</td></tr></table>

Table 5 reports the same aggregation on the MIMIC-IV subset. The largest mean diference again appears for neurological events. PR-AUC changes from 0.419 to 0.520, and ROC-AUC changes from 0.749 to 0.809. The other endpoints show smaller mean diferences. The record-level split permits records from one subject in diferent partitions, so these results require subject-disjoint re-evaluation.

Overall, the reported experiments motivate further study of neighbor event curves. Additional controls are needed to separate the efect of event timing from retrieval inputs, measurement patterns, and the split procedure. The MIMIC-IV analysis covers 4,000 stays from a single data source, so it does not establish external validity.

## 4.2 Neighbor Retrieval Ablation

To examine retrieval sensitivity, we compare several early-window feature sets on neurological events. Table 6 reports validation-set averages across learners and horizons. Retrieval with the 27 abnormality indicators has the highest values, with PR-AUC 0.550 and ROC-AUC 0.744. Retrieval with laboratory variables ranks second. The feature sets contain diferent variables, so this comparison cannot isolate the efect of the similarity measure.

Table 7. Full curve and the reported shufled-curve control. Values are average PR-AUC across horizons. The diference is shufled minus full. This control changes event status and event time.
<table><tr><td>Data</td><td>Endpoint</td><td>Full Shuf.</td><td> $\Delta$ </td></tr><tr><td rowspan="4">PN12</td><td>Neuro.</td><td>0.519 0.355-0.164</td><td rowspan="4"></td></tr><tr><td>Renal</td><td>0.384 0.338-0.046</td></tr><tr><td>Electro./Metab. 0.372 0.333-0.039</td><td></td></tr><tr><td>Inflam.</td><td>0.5360.489-0.047</td></tr><tr><td rowspan="4">MIMIC</td><td>Neuro.</td><td>0.493 0.342 -0.151</td><td></td></tr><tr><td>Renal</td><td>0.3500.299-0.051</td><td></td></tr><tr><td>Electro./Metab. 0.3410.314-0.027</td><td></td><td></td></tr><tr><td>Inflam.</td><td>0.286 0.258-0.028</td><td></td></tr></table>

Table 8. Single-horizon neighbor event rate versus full Curve-NECA on PhysioNet 2012 using XGBoost. Values are average PR-AUC across horizons.
<table><tr><td>Endpoint</td><td>Raw</td><td>Full Single</td><td> $\Delta _ { \mathrm { S - F } }$ </td></tr><tr><td>Neuro.</td><td>0.331</td><td>0.443 0.417</td><td>-0.026</td></tr><tr><td>Renal</td><td>0.3280.366</td><td>0.358</td><td>-0.008</td></tr><tr><td>Electro./Metab. 0.3260.341 0.331 -0.010</td><td></td><td></td><td></td></tr><tr><td>Inflam.</td><td></td><td>0.490 0.5130.507-0.006</td><td></td></tr></table>

## 4.3 Shufled Event-Time Control

The reported time-shufle ablation assigns a randomly sampled finite event time to every retrieved neighbor, including neighbors without a recorded event. It then adjusts each curve column to approximately match its original mean. This procedure changes each neighbor’s event status as well as its event time. Table 7 shows the resulting scores. The decrease cannot be attributed to timing alone. Testing that claim would require shufling event times among neighbors with a recorded event by 42 hours while preserving their event status.

## 4.4 Single-Horizon and Neighbor-Mean Ablations

The single-horizon variant adds only the neighbor event rate $z _ { i , H } ^ { ( g ) }$ at the target horizon �. Full Curve-NECA adds all four rates in $\mathbf { z } _ { i } ^ { ( g ) }$ . Table 8 shows that the single-horizon rate already improves over Raw XGBoost. The full curve adds a smaller increment, especially for neurological events. The table’s diference is single-horizon PR-AUC minus full-curve PR-AUC.

The neighbor-mean variant appends averages of retrieved neighbors’ early-window variables without their event times. On neurological events, its validation PR-AUC is 0.509, compared with 0.443 for Raw and 0.550 for Curve-NECA. This comparison does not separate neighbor outcomes, cross-system inputs, and timing.

## 5 Discussion and Conclusion

NECA adds cumulative event rates from retrieved training records to an ICU predictor. The target-horizon neighbor rate explains much of the observed improvement, while the other curve points add smaller increments. Classifiers are fitted independently by horizon and may therefore produce decreasing risks across longer horizons. We have not measured the frequency of such predictions or assessed calibration.

The largest diference concerns neurological events, but we have not tested whether cross-system retrieval variables explain it. A low recorded GCS can reflect sedation, intubation, or measurement practice. Clinical interpretation requires a targeted sensitivity analysis.

The findings remain preliminary. Endpoint rules were not clinically adjudicated. Missing measurements or insuficient follow-up can be coded as no recorded event. The Raw tabular models lack some retrieval variables. MIMIC-IV uses a record-level split, training event curves were not cross-fitted, and $K = 1 0 0$ was fixed without validationbased selection. The shufle control changes event status, and the mean diferences lack paired uncertainty estimates. These limitations prevent claims about event-time specificity, generalization, or clinical benefit.

## A Retrieval Indicator Definitions

The retrieval vector has 27 entries from the first six ICU hours. Thirteen basic indicators are $\mathrm { a g e } \geq 6 5$ , hypotension (minimum $\mathrm { S B P } < 9 0 $ or $\mathbf { M A P } < 6 5 )$ , tachycardia (maximum HR $> 1 1 0$ or last $\mathrm { H R } > 1 0 0 )$ , hypoxemia (minimum $\mathrm { S p O } _ { 2 } < 9 2$ , or $\mathrm { P a O } _ { 2 } < 6 0$ when $\mathrm { S p O } _ { 2 }$ is unavailable), acidosis (minimum pH $[ < 7 . 3 5$ or bicarbonate $< 2 2 )$ , lactate $> 2 .$ renal abnormality (maximum creatinine $> 1 . 5$ or $\mathrm { B U N } > 3 0 )$ , worsening creatinine (last > 1.1× first $+ 0 . 0 5 )$ ), inflammation (maximum $\mathrm { W B C } > 1 2$ or temperature $> 3 8 ^ { \circ } \mathrm { C } )$ , bilirubin $> 2 ,$ , recorded ventilation, at least 60 measurements, and last ${ \mathrm { G C S } } \leq 1 2$

Six system indicators cover hemodynamic findings (hypotension or tachycardia), respiratory findings (hypoxemia or last $\mathrm { R R } > 2 4 )$ , renal findings (abnormal or worsening creatinine), inflammation, hepatic findings, and coagulation (minimum platelets < 100). Five interaction indicators capture shock-like findings (hypotension and tachycardia with acidosis or elevated lactate), cardio-respiratory findings (hypotension with hypoxemia or tachypnea), renal-perfusion findings (hypotension with renal abnormality or last urine output $< 4 0 )$ ), inflammation with a renal, respiratory, or hemodynamic abnormality, and at least three abnormal systems.

The final three entries are abnormal-system count divided by five, severe-signal count divided by five, and ICU-type code divided by four. Severe signals are hypotension with tachycardia, hypoxemia with hypotension, renal abnormality with hypotension, acidosis or elevated lactate, and low GCS. Missing measurements make binary rules false, and missing ICU type is zero. Some retrieval variables are unavailable to the endpointspecific Raw tabular predictor.

## References

1. R. Adams, K. E. Henry, A. Sridharan, et al., “Prospective, multi-site study of patient outcomes after implementation of the TREWS machine learning-based early warning system for sepsis,” Nature Medicine, vol. 28, no. 7, pp. 1455–1460, 2022, doi: 10.1038/s41591-022-01894-0.

2. R. Blythe, R. Parsons, A. G. Barnett, et al., “Prioritising deteriorating patients using timeto-event analysis: prediction model development and internal–external validation,” Critical Care, vol. 28, article 247, 2024, doi: 10.1186/s13054-024-05021-y.

3. T.-C. Do, H.-J. Yang, S.-H. Kim, et al., “Multi-horizon event detection for in-hospital clinical deterioration using dual-channel graph attention network,” International Journal ofMedical Informatics, vol. 195, article 105745, 2025, doi: 10.1016/j.ijmedinf.2024.105745.

4. J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” The Annals ofStatistics, vol. 29, no. 5, pp. 1189–1232, 2001, doi: 10.1214/aos/1013203451.

5. D. W. Hosmer, S. Lemeshow, and R. X. Sturdivant, Applied Logistic Regression, 3rd ed. Hoboken, NJ, USA: Wiley, 2013, doi: 10.1002/9781118548387.

6. T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794, 2016, doi: 10.1145/2939672.2939785.

7. G. Ke, Q. Meng, T. Finley, et al., “LightGBM: A highly eficient gradient boosting decision tree,” in Advances in Neural Information Processing Systems, vol. 30, pp. 3149–3157, 2017.

8. N. Hollmann, S. Müller, L. Purucker, et al., “Accurate predictions on small data with a tabular foundation model,” Nature, vol. 637, no. 8045, pp. 319–326, 2025, doi: 10.1038/s41586-024- 08328-6.

9. S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997, doi: 10.1162/neco.1997.9.8.1735.

10. K. Ng, J. Sun, J. Hu, and F. Wang, “Personalized predictive modeling and risk factor identification using patient similarity,” AMIA Joint Summits on Translational Science Proceedings, vol. 2015, pp. 132–136, 2015.

11. A. Sharafoddini, J. A. Dubin, J. Lee, and J. Maslove, “Patient similarity in prediction models based on health data: A scoping review,” JMIR Medical Informatics, vol. 5, no. 1, e7, 2017, doi: 10.2196/medinform.6730.

12. H. Harutyunyan, H. Khachatrian, D. C. Kale, et al., “Multitask learning and benchmarking with clinical time series data,” Scientific Data, vol. 6, article 96, 2019, doi: 10.1038/s41597- 019-0103-9.

13. W. Chen, S. Wang, G. Long, L. Yao, Q. Z. Sheng, and X. Li, “Dynamic illness severity prediction via multi-task RNNs for intensive care unit,” in Proceedings ofthe IEEE International Conference on Data Mining (ICDM), pp. 917–922, 2018, doi: 10.1109/ICDM.2018.00111.

14. L. Ma, C. Zhang, Y. Wang, W. Ruan, J. Wang, W. Tang, X. Ma, X. Gao, and J. Gao, “ConCare: Personalized clinical feature embedding via capturing the healthcare context,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 01, pp. 833–840, 2020, doi: 10.1609/aaai.v34i01.5428.

15. M. Contreras, B. Silva, B. Shickel, et al., “Real-time prediction of intensive care unit patient acuity and therapy requirements using state-space modelling,” Nature Communications, vol. 16, article 7315, 2025, doi: 10.1038/s41467-025-62121-1.

16. S. Roy, D. Mincu, E. Loreaux, et al., “Multitask prediction of organ dysfunction in the intensive care unit using sequential subnetwork routing,” Journal of the American Medical Informatics Association, vol. 28, no. 9, pp. 1936–1946, 2021, doi: 10.1093/jamia/ocab101.

17. Z. Zeng, Y. Liu, S. Yao, et al., “Inter-organ correlation based multi-task deep learning model for dynamically predicting functional deterioration in multiple organ systems ofICU patients,” BioData Mining, vol. 18, article 31, 2025, doi: 10.1186/s13040-025-00445-w.

18. N. Soley, D. E. Moody, S. R. Tariq, et al., “Early dynamic prediction of organ system deterioration in intensive care unit (ICU) patients using machine learning,” Journal ofCritical Care, vol. 86, p. 154954, 2025, doi: 10.1016/j.jcrc.2024.154954.

19. S. A. C. Bukhari, A. Singh, S. Hossain, et al., “Early prediction of multi-label care escalation triggers in the intensive care unit using electronic health records,” in International Conference on e-health Networking, Applications and Services (HealthCom), pp. 1–8, 2025, doi: 10.1109/HealthCom60686.2025.11343128.

20. M. Ye, S. Cui, Y. Wang, et al., “MedRetriever: Target-driven interpretable health risk prediction via retrieving unstructured medical text,” in Proceedings of the 30th ACM International Conference on Information and Knowledge Management, pp. 2414–2423, 2021, doi: 10.1145/3459637.3482273.

21. R. Xu, W. Shi, Y. Yu, et al., “RAM-EHR: Retrieval augmentation meets clinical predictions on electronic health records,” in Proceedings of the 62nd Annual Meeting of the Associ ation for Computational Linguistics (Volume 2: Short Papers), pp. 754–765, 2024, doi: 10.18653/v1/2024.acl-short.68.

22. P. Jiang, C. Xiao, A. Cross, and J. Sun, “GraphCare: Enhancing healthcare predictions with personalized knowledge graphs,” in Proceedings of the International Conference on Learning Representations, 2024.

23. P. Jiang, C. Xiao, M. Jiang, et al., “Reasoning-enhanced healthcare predictions with knowledge graph community retrieval,” in Proceedings of the International Conference on Learning Representations, 2025.

24. I. Lopez, A. Swaminathan, K. S. Vedula, et al., “Clinical entity augmented retrieval for clinical information extraction,” npj Digital Medicine, vol. 8, article 45, 2025, doi: 10.1038/s41746- 024-01377-1.

25. Y. Li, J. Ge, S. Cao, et al., “Similarity-aware dual-perspective learning for medical event prediction,” in Proceedings of the IEEE International Conference on Bioinformatics and Biomedicine, pp. 6263–6270, 2025, doi: 10.1109/BIBM66473.2025.11356755.

26. M. K. Sahu and P. Roy, “Similarity-based self-construct graph model for predicting patient criticalness using graph neural networks and EHR data,” arXiv preprint arXiv:2508.00615, 2025, doi: 10.48550/arXiv.2508.00615.

27. I. Silva, G. Moody, D. J. Scott, et al., “Predicting in-hospital mortality of ICU patients: The PhysioNet/Computing in Cardiology Challenge 2012,” Computing in Cardiology, vol. 39, pp. 245–248, 2012.

28. A. E. W. Johnson, L. Bulgarelli, L. Shen, et al., “MIMIC-IV, a freely accessible electronic health record dataset,” Scientific Data, vol. 10, article 1, 2023, doi: 10.1038/s41597-022- 01899-x.

29. J.-L. Vincent, R. Moreno, J. Takala, et al., “The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure,” Intensive Care Medicine, vol. 22, pp. 707–710, 1996, doi: 10.1007/BF01709751.

30. M. Singer, C. S. Deutschman, C. W. Seymour, et al., “The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3),” JAMA, vol. 315, no. 8, pp. 801–810, 2016, doi: 10.1001/jama.2016.0287.

31. Royal College of Physicians, National Early Warning Score (NEWS) 2: Standardising the Assessment ofAcute-Illness Severity in the NHS. Updated report of a working party. London, UK: Royal College of Physicians, 2017.

32. J. Welch, P. Dean, J. Hartin, and G. P. Smith, “Using NEWS2: an essential component of reliable clinical assessment,” Clinical Medicine, vol. 22, no. 6, pp. 509–513, 2022, doi: 10.7861/clinmed.2022-0435.

33. Kidney Disease: Improving Global Outcomes (KDIGO) Acute Kidney Injury Work Group, “KDIGO clinical practice guideline for acute kidney injury,” Kidney International Supplements, vol. 2, no. 1, pp. 1–138, 2012, doi: 10.1038/kisup.2012.1.

34. R. C. Bone, R. A. Balk, F. B. Cerra, et al., “Definitions for sepsis and organ failure and guidelines for the use of innovative therapies in sepsis,” Chest, vol. 101, no. 6, pp. 1644– 1655, 1992, doi: 10.1378/chest.101.6.1644.