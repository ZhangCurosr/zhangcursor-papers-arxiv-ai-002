# Self-Supervised Attention-Based Virtual Metrology for CMP MRR Prediction

Jonathan Adam Rico

Nagarajan Raghavan

Senthilnath Jayavelu

Engineering Product Development,

Engineering Product Development,

SUTD

Institute of Advanced Intelligence and

Computing, A\*STAR

Institute of Advanced Intelligence and

Singapore

SUTD

0000-0003-1023-6634

Singapore 0000-0001-6735-3108

Computing, A\*STAR

Dept. of Electrical and Computer

Engineering, NUS

Singapore

0000-0002-1737-7985

Abstract—Chemical mechanical planarization (CMP) is a critical semiconductor manufacturing process where material removal rate (MRR) is a key metric for process monitoring and advanced process control (APC). Virtual metrology enables MRR estimation from process time series signals, supporting run-to-run (R2R) control and process optimization while reducing reliance on physical metrology. Existing approaches typically rely on simplified linear assumptions or require handcrafted feature engineering. In this paper, we propose Global Regression Pretraining (GREP), a self-supervised, attention-based framework for CMP MRR prediction. GREP learns temporal representations from unlabeled sensor signals through global proxy-target regression, followed by supervised fine-tuning for MRR estimation. The framework is evaluated against a feature-engineered supervised baseline under three conditions: limited labeled data, sensor noise, and temporal process drift. Experiments on the PHM 2016 CMP dataset demonstrate competitive performance under limited labeled-data conditions, with improvements in selected[ cases and performance largely comparable to the baseline across different labeled-data fractions. Both models exhibit relatively small changes in prediction error with increasing noise, while GREP achieves a 17.0% RMSE reduction in the final temporal interval.

Index Terms—Chemical mechanical planarization, Virtual metrology, Material removal rate, Self-supervised learning, Time series regression

## I. INTRODUCTION

Chemical mechanical planarization (CMP) is a critical process in semiconductor manufacturing, enabling surface planarization through combined chemical and mechanical interactions. Material removal rate (MRR) is a key process characteristic affecting wafer uniformity, process consistency, and manufacturing quality. Classical physics-based approaches, such as the Preston model [1], provide foundational relationships between polishing conditions and material removal. However, accurate MRR estimation remains challenging because CMP involves complex interactions among process conditions, consumable states, and equipment behavior [2].

Virtual metrology (VM) provides an alternative approach to estimating wafer quality characteristics using process measurements without requiring direct physical metrology for every wafer. Previous studies have demonstrated the potential of VM for semiconductor process monitoring and advanced process control (APC) [3], [4]. For CMP, physics-informed feature engineering and conventional machine learning have been investigated for MRR prediction [2], [5]. Recently, Chen et al. [6] used 20 selected features from 68 engineered descriptors with an externally pretrained TabPFN model, achieving an RMSE of 4.98 nm/min without hyperparameter tuning. However, such feature engineering requires domain knowledge and may discard temporal information from the original sensor signals.

Data-driven VM also faces several practical challenges. First, labeled metrology data are relatively costly to obtain, while equipment sensor signals are routinely collected during processing. Second, sensor noise and measurement disturbances may affect prediction accuracy. Third, changes in equipment and process conditions can introduce temporal distribution shifts, reducing the reliability of models trained on historical data [4], [7]. These considerations motivate models that can learn meaningful temporal representations with limited labeled data while maintaining predictive performance under changing operating conditions.

Self-supervised learning enables representation learning from unlabeled time series. Existing methods include contrastive learning [8] and masked reconstruction [9]. An alternative is to construct regression targets from predefined statistical, spectral, and temporal features, although this requires explicit feature selection and extraction. Random projections offer computationally inexpensive transformations of highdimensional data [10] and have recently been investigated as self-supervised learning targets. Sui et al. [11] employed random neural-network projectors for representation learning, while [12] used random-projection quantization for masked EEG time series prediction. Unlike these approaches, the proposed GREP directly performs regression on continuous global projections of complete and partial sensor sequences, without requiring predefined feature categories or target quantization.

In this work, we propose Global Regression Pretraining (GREP), a self-supervised, attention-based virtual metrology framework for CMP MRR prediction. GREP learns temporal representations by regressing global proxy targets constructed from unlabeled process signals, followed by supervised adaptation for MRR estimation. We investigate three research questions: (RQ1) whether GREP improves prediction accuracy with limited labeled data; (RQ2) how GREP performs under sensor noise; and (RQ3) how GREP performs under temporal process drift. These questions are evaluated using the PHM 2016 CMP dataset.

## II. GREP FRAMEWORK

The proposed GREP framework consists of self-supervised pretraining followed by supervised MRR regression, as illustrated in Fig. 1. Given a multivariate CMP sensor sequence $\mathbf { X } _ { i } \in \mathbb { R } ^ { T \times C }$ , where T and C denote the sequence length and number of sensor channels, respectively, the input is divided into overlapping temporal patches and processed by a channelindependent Transformer encoder.

## A. Self-Supervised Global Regression Pretraining

During pretraining, each sensor channel is standardized and divided into three temporal views: the complete sequence, first half, and second half. Each view is projected onto 64 fixed Gaussian random vectors. The k-th proxy target from view (v) is defined by

$$
g _ { i , c , k } ^ { ( v ) } = \sum _ { t = 1 } ^ { T _ { v } } \frac { x _ { i , t , c } ^ { ( v ) } - \mu _ { i , c } ^ { ( v ) } } { \sigma _ { i , c } ^ { ( v ) } + \epsilon } W _ { t , k } , \quad W _ { t , k } \sim { \mathcal N } ( 0 , 1 )\tag{1}
$$

where x is the input sensor sequence, $T _ { v }$ denotes the view length, $\mu _ { i , c } ^ { ( v ) }$ and $\stackrel { \cdot } { \sigma } _ { i , c } ^ { ( v ) }$ are its temporal mean and standard deviation, ϵ is a small constant, and $W _ { t , k } \sim { \mathcal { N } } ( 0 , 1 )$ is a fixed projection coefficient. Concatenating the three views produces $K = 1 9 2$ proxy targets per channel.

The encoder and auxiliary regression heads are trained to predict these targets by minimizing the standardized mean squared error defined as

$$
\mathcal { L } _ { \mathrm { G R E P } } = \frac { 1 } { N C K } \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { C } \sum _ { k = 1 } ^ { K } \left( \frac { \hat { g } _ { i , c , k } - g _ { i , c , k } } { \sigma _ { k } + \epsilon } \right) ^ { 2 }\tag{2}
$$

where N is the number of samples, $\hat { g }$ denotes the predicted pretraining target, and $\sigma _ { k }$ is the standard deviation of the target batch-wise. This enables the model to learn global temporal representations without requiring measured MRR labels.

## B. Downstream MRR Regression

Following pretraining, the auxiliary regression heads are replaced with MRR prediction heads. The pretrained encoder representations are aggregated using the final and mean-pooled patch features, followed by channel aggregation. Separate models are initialized with GREP-pretrained weights and fine-tuned for low-speed and high-speed CMP operation by minimizing the mean squared error between predicted and measured MRR. Both the encoder and downstream regression parameters are optimized using labeled training data.

TABLE I  
DETAILS ON THE NUMBER OF TRAIN AND TEST SAMPLES FOR EACH OF THE OPERATING MODES OF A CMP RUN.
<table><tr><td>Mode</td><td>Chamber</td><td>Average MRR</td><td>Train</td><td>Test</td></tr><tr><td>Low-speed</td><td>4,5,6</td><td>55-110 nm/min</td><td>1613</td><td>364</td></tr><tr><td>High-speed</td><td>1,2,3</td><td>140-170 nm/min</td><td>351</td><td>73</td></tr></table>

## III. EXPERIMENTAL METHODOLOGY

## A. CMP PHM 2016 Dataset and Preprocessing

Experiments were conducted using the PHM 2016 CMP dataset [13], which contains 19 process variables, including pressure, flow rate, rotational speed, and usage counts. The prediction target is average MRR, expressed in nm/min. The data comprise two operating modes: low-speed polishing in chambers 4-6 and high-speed polishing in chambers 1-3. Table I summarizes the data distribution for both operating modes.

Sensor sequences were smoothed using a five-point moving average and linearly resampled to 128 time steps. Each sequence was divided into overlapping patches of length 16 and stride 8.

GREP was implemented in PyTorch using a channelindependent PatchTST encoder [14]. The Transformer comprised six encoder layers, eight attention heads, a hidden dimension of 128, and a feedforward dimension of 256. Gaussian Error Linear Unit (GELU) activation and a dropout rate of 0.1 were used. Self-supervised pretraining was performed for 150 epochs using AdamW, a learning rate of 0.001, and a batch size of 32. The pretraining stage used only the designated training partition.

For downstream regression, separate prediction models were fine-tuned for low-speed and high-speed operation. Learning rates of 0.001 and 0.0005 and batch sizes of 32 and 8 were used, respectively. Both models were fine-tuned for 50 epochs with mean squared error loss and a dropout rate of 0.4. The learning rate followed a five-epoch linear warmup and cosine annealing schedule. Experiments were conducted on an NVIDIA RTX 3070 GPU.

## B. Research Questions and Evaluation

The proposed method was compared with a supervised, feature-engineered baseline implemented following the approach of Li et al. [5]. Process-level features, including summary statistics, integrated sensor measurements, polishing duration, and pressure-related variables, were extracted from each CMP run. Separate tree-based regression models were trained for the two polishing-speed modes. Both methods were evaluated using the same experimental data partitions and labeled-data fractions. Root mean squared error (RMSE) was used as the evaluation metric, with lower values indicating better prediction accuracy.

RQ1: Label Efficiency. To assess performance with limited ground-truth measurements, GREP was pretrained on unlabeled training sequences, while downstream regression used 10%-90% of the labeled training data. Prediction accuracy was compared with the supervised baseline across the same labeled-data fractions.

![](images/5f4c831675ec2737ee7757cd939fac9d26597e9db6d86bf7f0bdddd9ae50e16e.jpg)  
Fig. 1. Overview of the proposed self-supervised GREP architecture for MRR virtual metrology.

RQ2: Robustness to Sensor Noise. Following the timedomain perturbation methods described in [15], Gaussian noise, random spikes, and linear trends were introduced into the training sensor signals. Gaussian noise was generated at SNR levels of 0, 5, 10, 20, and 30 dB. Spikes were applied independently with a probability of 0.01 and a standard deviation three times that of the Gaussian-perturbed signals. For each wafer-stage sequence, a linear trend with a randomly sampled terminal offset from $\mathcal { N } ( 0 , 0 . 0 5 ^ { 2 } )$ was also added. These settings provide controlled synthetic perturbations rather than calibrated CMP sensor fault conditions. Models were evaluated on unmodified test data.

RQ3: Robustness to Process Drift. To assess robustness to temporal process changes, the training data were restricted to the earliest chronological interval, and prediction errors were evaluated over subsequent intervals. This setup examines the ability of models trained on earlier CMP observations to maintain accuracy as the temporal separation from the training period increases.

## IV. RESULTS AND DISCUSSION

## A. RQ1: Label Efficiency

Fig. 2 compares GREP and the feature-engineered baseline across labeled-data fractions and training-signal noise levels. GREP achieves lower prediction errors in selected limitedlabel settings while maintaining comparable performance across most labeled-data fractions. Although the baseline performs notably better in one setting, the results demonstrate that GREP remains competitive under limited metrology data availability.

## B. RQ2: Robustness to Sensor Noise

Noise sensitivity was assessed from the change in RMSE with increasing noise. Fig. 2 shows that at 30% labeled data, the fitted slopes were 0.114 and 0.049 RMSE per tenfold increase in noise power for GREP and the baseline, respectively. At 40%, the corresponding slopes are 0.054 and 0.099. The fitted slopes remain relatively small, but neither method consistently exhibits lower noise sensitivity across labeled-data fractions. Thus, GREP maintains competitive prediction accuracy under training-signal corruption without demonstrating uniformly higher resistance to increasing noise intensity.

![](images/a96f567e6bff27a3e0403e40b0460a4bbd49ed422b8aeea51f0cd8e502d2a4b9.jpg)  
Fig. 2. MRR Prediction RMSE under different training-signal noise levels and varying proportions of labeled training data.

## C. RQ3: Robustness to Process Drift

Fig. 3 presents prediction errors across four chronological intervals. GREP achieves lower RMSE than the baseline in all intervals, with reductions of 16.6% and 17.0% in the third and fourth intervals, respectively. In the final interval, GREP achieves an RMSE of 9.32, compared with 11.23 for the baseline.

The smaller prediction errors in later intervals suggest that GREP generalizes better to temporally separated CMP observations. Although the physical causes of process drift were not independently identified, the results indicate improved robustness to temporal process changes.

![](images/a9975a7106d44ac6fc2fe75a2373b96ef12ed068bf31e84b534dc2d7652bf8da.jpg)  
Fig. 3. MRR prediction RMSE across four temporal partitions, with training restricted to the first partition.

## V. CONCLUSION

This study presented GREP, a self-supervised, attentionbased virtual metrology framework for CMP MRR prediction. Compared with a feature-engineered supervised baseline, GREP achieved competitive accuracy under limited-label conditions and lower prediction errors across temporally separated CMP observations. Both models exhibited relatively small changes in prediction error under increasing training-signal noise, without a consistent advantage in noise sensitivity. These findings suggest that GREP can improve label-efficient MRR prediction and robustness to temporal process changes. Although APC integration is beyond the scope of this work, GREP could support run-to-run control and CMP process monitoring. Future work will investigate online adaptation to process drift, handcrafted versus proxy target ablation, and uncertainty-aware metrology sampling to evaluate the potential for reducing physical measurement requirements while maintaining prediction accuracy.

## ACKNOWLEDGMENT

The first author acknowledges the Singapore University of Technology and Design (SUTD) and the Ministry of Education, Singapore (MOE), for support through the Research

Student Scholarship and SUTD Ph.D. Fellowship. This research was also supported by the Research Surplus Grant No. RGSUR08 of Prof. Nagarajan Raghavan. The authors thank the Agency for Science, Technology and Research (A\*STAR) for providing computational resources.

## STATEMENT ON USE OF GENERATIVE AI

The authors acknowledge the use of ChatGPT (OpenAI) for grammar refinement and clarity improvement. All AI-assisted content was reviewed and edited by the authors to ensure accuracy and compliance with IEEE publication standards.

## REFERENCES

[1] F. Preston, “The theory and design of plate glass polishing machines,” Journal of the Society of Glass Technology, 1927.

[2] Z. Li and D. Wu, “A data-driven approach to material removal rate prediction in chemical mechanical polishing,” in Annual Conference of the PHM Society, vol. 10, 2018.

[3] H. Cai, J. Feng, Q. Yang, W. Li, X. Li, and J. Lee, “A virtual metrology method with prediction uncertainty based on gaussian process for chemical mechanical planarization,” Computers in Industry, vol. 119, 2020.

[4] A. Khan, J. Moyne, and D. Tilbury, “Virtual metrology and feedback control for semiconductor manufacturing processes using recursive partial least squares,” Journal of Process Control, vol. 18, pp. 961–974, 2008.

[5] X. Li, C. Wang, X. Mo, D. Zhao, and C. Li, “Assessment of physicsbased and data-driven models for material removal rate prediction in chemical mechanical polishing,” in 2nd International Conference on Electrical Engineering and Automation, 2018, pp. 116–121.

[6] W. Chen, X. Han, Z. Wu, Z. Zhu, Y. Peng, J. Fu, S. Yuan, and D. Yang, “Physics-guided feature construction for small-sample virtual metrology in chemical mechanical polishing,” IEEE Transactions on Semiconductor Manufacturing, 2026.

[7] V. Maitra, Y. Su, and J. Shi, “Virtual metrology in semiconductor manufacturing: Current status and future prospects,” Expert Systems with Applications, vol. 249, 2024.

[8] E. Eldele, M. Ragab, Z. Chen, M. Wu, C. Kwoh, X. Li, and C. Guan, “Time-series representation learning via temporal and contextual contrasting,” in Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, 2021, pp. 2352–2359.

[9] Z. Li, Z. Rao, L. Pan, P. Wang, and Z. Xu, “Ti-mae: Self-supervised masked time series autoencoders,” arXiv preprint arXiv:2301.08871.

[10] D. Achlioptas, “Database-friendly random projections: Johnsonlindenstrauss with binary coins,” Journal of Computer and System Sciences, vol. 66, pp. 671–687, 2003.

[11] Y. Sui, T. Wu, J. Cresswell, G. Wu, G. Stein, X. Huang, X. Zhang, and M. Volkovs, “Self-supervised representation learning from random data projectors,” in International Conference on Learning Representations, 2024, pp. 6792–6814.

[12] H. Gui, X. Li, and X. Chen, “Vector quantization pretraining for eeg time series with random projection and phase alignment,” in International Conference on Machine Learning, 2024, pp. 16 731–16 750.

[13] Society of Prognostics and Health Management, “PHM Data Challenge,” [Online]. Available: https://www.phmsociety.org/events/conference/phm/ 16/data-challenge, 2016, dataset.

[14] Y. Nie, N. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in International Conference on Learning Representations, 2023.

[15] Q. Wen, L. Sun, F. Yang, X. Song, J. Gao, X. Wang, and H. Xu, “Time series data augmentation for deep learning: A survey,” International Joint Conference on Artificial Intelligence, p. 4653–4660, 2021.

## AUTHOR BIOGRAPHY

The first author is a PhD candidate at the SUTD with research attachment at the A\*STAR, Singapore. His research focuses on time series foundation models for industrial applications.