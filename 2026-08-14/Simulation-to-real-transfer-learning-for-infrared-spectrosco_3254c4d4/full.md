# Simulation-to-real transfer learning for infrared spectroscopic chemical sensing and analysis from molecules to complex samples

Yusen Tan1, Yixuan Chen2, Zheng Fang1, Pan Liu1, Yifan Li1, Qinyu Guo3, Zhedong Lin3, Yuqiang Li4, Xiangxiang Zeng†5, Tong Wang†6, Jun Xia†1, 7

<sup>1</sup>Information Hub, The Hong Kong University of Science and Technology (Guangzhou), Guangzhou 511453, China

<sup>2</sup>College of Computer Science and Technology, Jilin University, Changchun 130012, China

<sup>3</sup>School of Computer Science, University of Auckland, Auckland 1142, New Zealand

<sup>4</sup>AI for Chemistry Center, Shanghai Artificial Intelligence Laboratory, Shanghai 200232, China

<sup>5</sup>College of Computer Science and Electronic Engineering, Hunan University, Changsha 410082, China

<sup>6</sup>State Key Laboratory of Chemo and Biosensing, College of Chemistry and Chemical Engineering, Hunan University, Changsha 410082, China

<sup>7</sup>Department of Computer Science and Engineering, The Hong Kong University of Science and Technology, Hong Kong SAR, China

## Abstract

Infrared (IR) spectroscopy is widely used for chemical sensing, from molecular characterization to the analysis of complex samples, but extracting reliable chemical information from spectra remains challenging. Conventional inter pretation based on expert peak assignment, rule-based analysis, and library matching is labor-intensive and dificult to scale across analytical objectives and datasets while maintaining consistently high accuracy, and its reliance on prior knowledge and reference spectra makes it less reliable for unfamiliar compounds and complex samples. Machine learning methods have enabled more automated and scalable chemical inference from IR spectra, but most approaches remain tailored to individual tasks or datasets, require large training sets of labeled spectra, and show limited transfer ability across analytical objectives and experimental datasets. Together, these limitations and the scarcity of labeled experimental IR spectra motivate simulation-to-real transfer learning to enable data-eficient chemical inference under limited supervision. Here we introduce UltraIR, a foundation model for IR spectroscopy with more than 100 mil lion parameters that enables simulation-to-real transfer learning for chemical sensing and analysis from molecules to complex samples. UltraIR learns a shared spectral representation by pretraining on approximately 60 million simu lated IR spectra using three complementary pretraining objectives for spectral reconstruction, molecular fingerprint similarity alignment, and functional-group prediction. The pretrained encoder is then adapted to each downstream objective using task-specific labels or targets paired with the corresponding IR spectra, typically with only a limited number of labeled spectra available. Across benchmark evaluations of functional-group prediction, molecular structure elucidation, physicochemical property prediction, and mixture-component identification and quantification, as well as real-world chemical sensing applications involving bacterial classification, medicinal-herb geographic origin traceability and constituent quantification using two newly generated in-house datasets, microplastics classification, and soil prop erty prediction, UltraIR outperforms conventional machine-learning and task-specific deep-learning baselines. UltraIR demonstrates the practical value of simulation-to-real transfer learning through strong performance in two challenging real-world chemical sensing scenarios: downstream adaptation using limited labeled experimental spectra and zeroshot inference for the same analytical task across Fourier-transform infrared spectrometers and laboratories. More broadly, this approach provides a route to adaptable, data-eficient chemical sensing systems for extracting reliable chemical information from IR spectra of complex real-world samples.

Correspondence: Xiangxiang Zeng (xzeng@foxmail.com); Tong Wang (wangtong@hnu.edu.cn); Jun Xia (junxia@hkust-gz.edu.cn)

## 1 Introduction

Infrared (IR) spectroscopy is an analytical technique that probes molecular vibrations through their interaction with infrared radiation to generate chemically informative spectral fingerprints [1–3]. Its rapid, label-free, and often nondestructive measurements support widespread use in chemical sensing and analysis, ranging from molecular characterization to complex-sample analysis in environmental monitoring, clinical analysis, and materials characterization [4–7]. However, the ability to acquire IR spectra at scale has not been matched by an equally scalable ability to convert them into reliable chemical information [8–10]. Conventional workflows still rely heavily on expert peak assignment, rule-based interpretation, and library matching. These approaches depend on prior expert knowledge, predefined rules, or reference spectra, which can limit reliable interpretation when suitable reference information is unavailable, as may occur for unfamiliar compounds and complex samples [8, 11]. Such dependencies also make conventional workflows dificult to scale across analytical objectives and datasets while maintaining consistent accuracy [1, 12, 13]. A central challenge for next-generation IR chemical sensing is therefore to translate spectra into reliable chemical information at scale across analytical objectives and datasets while maintaining high accuracy.

Machine learning has begun to address the challenge of automated IR analysis at scale by learning predictive relationships directly from spectral data [14–16]. Recent studies have shown that IR spectra support functional-group recognition, molecular structure elucidation, library-based molecular retrieval, and mixture-component identification [17–22]. Together, these advances demonstrate that IR spectra contain molecular information that can support diverse analytical tasks. However, most approaches are tailored to individual tasks or datasets and require large task-specific collections of labeled spectra, limiting the development of shared spectral representations that transfer across analytical objectives and experimental datasets. Despite enabling more rapid and scalable analysis than expert peak assignment, rule-based interpretation, and library matching, these models often achieve only modest accuracy gains over strong conventional machine-learning baselines, and these gains may disappear when few labeled spectra are available for training. These limitations motivate a general representation-learning approach that enables scalable, reliable, transferable, and data-eficient IR analysis.

Foundation models provide such a general representation-learning approach, as demonstrated by large language models (LLMs), whose large-scale pretraining supports adaptation across diverse downstream tasks [23, 24]. Similar largescale representation-learning paradigms have emerged across sensing domains, including medical imaging [25, 26], continuous glucose monitoring [27], neural-activity modeling [28], remote sensing [29], and robotic tactile sensing [30, 31]. However, foundation-model approaches remain underexplored in IR spectroscopy, with their development constrained by the scarcity of large, chemically diverse collections of labeled experimental spectra [18, 32]. Simulated IR spectra ofer a scalable, complementary source of pretraining data that can broaden molecular coverage and support shared representation learning [33–36]. To date, however, simulated IR spectra have mainly been used within individual chemical sensing and analysis tasks, with reported performance gains demonstrating their value for data-driven IR analysis [19]. Together, these observations motivate a simulation-to-real representation-learning paradigm that combines large-scale pretraining on simulated spectra with supervised adaptation using task-specific labeled experimental spectra, ofering a promising route toward transferable and data-eficient IR foundation models. Specifically, pretraining can capture recurring relationships among IR spectral patterns, molecular structures and chemical properties across a broad chemical space, while downstream supervision adapts the shared representation to each chemical sensing and analysis objective.

Here we introduce UltraIR, a foundation model for IR spectroscopy with more than 100 million parameters that enables simulation-to-real transfer learning for chemical sensing and analysis from molecules to complex samples. UltraIR uses a two-stage learning framework in which a shared spectral encoder is first pretrained on approximately 60 million simulated IR spectra. For each downstream objective, the pretrained encoder is paired with a newly initialized task-specific module, and all components are jointly optimized using labeled experimental IR spectra. The encoder combines a derivative-aware multi-channel input, hierarchical convolutional modules, and a patch-based Transformer. These components capture local line shapes and peak shifts, extract multiscale vibrational signatures, and mode longer-range spectral dependencies, respectively. Pretraining integrates three complementary objectives. Waveletdomain spectral reconstruction encourages preservation of broad spectral envelopes and fine absorption features. Molecular fingerprint similarity alignment organizes the latent space according to graded structural relationships among molecules, while multi-label functional-group prediction anchors the representation to interpretable chemica motifs. Together, this design couples broad molecular exposure from simulated spectra with supervised adaptation using task-specific labeled experimental IR spectra, often available only in limited numbers, to provide a shared representation backbone for molecular interpretation, mixture analysis, and chemical sensing in complex samples.

We evaluate UltraIR across molecular and mixture benchmarks, as well as real-world chemical sensing applications. The benchmarks cover functional-group prediction, molecular structure elucidation, physicochemical property prediction, and mixture-component identification and quantification. The real-world applications include bacterial classification, medicinal-herb geographic origin traceability and constituent quantification using two newly generated in-house datasets, microplastics classification, and soil property prediction. Across these evaluations, UltraIR outperforms conventional machine-learning and task-specific deep-learning baselines. The practical value of UltraIR’s simulationto-real transfer learning is demonstrated under two demanding conditions encountered in real-world chemical sensing: downstream adaptation with limited labeled experimental spectra and zero-shot inference for the same analytica task across Fourier-transform infrared (FTIR) spectrometers and laboratories. More broadly, this learning framework provides a basis for adaptable and data-eficient chemical sensing systems that extract reliable chemical information from IR spectra of complex real-world samples.

## 2 Results

## UltraIR overview

UltraIR is a foundation model for IR spectroscopy with more than 100 million parameters that enables simulation-toreal transfer learning for chemical sensing and analysis from molecules to complex samples. The framework integrates two complementary resources: approximately 60 million simulated IR spectra for large-scale pretraining and a diverse collection of downstream datasets for supervised adaptation and evaluation across molecular-level benchmarks, mixture analysis, and real-world chemical sensing applications (Fig. 1a).

The standardized pretraining dataset comprises molecule-associated IR spectra from three sources: approximately 1.5 million publicly released simulated spectra from IRtoMol, the multimodal spectroscopy dataset, and QM9S [19, 34, 35]; approximately 7.5 million newly generated spectra from molecular-dynamics simulations; and approximately 51 million spectra generated using a machine-learning-based spectral predictor [33]. Before pretraining, simulated spectra corresponding to molecules present in the downstream molecular datasets were excluded from the pretraining pool to prevent molecule-level overlap. Together, these sources provide the scale and molecular diversity required for large-scale pretraining.

For supervised adaptation and evaluation, we assembled approximately 120,000 experimental IR spectra together with the simulated USPTO molecular benchmark. Molecular-level benchmarks use spectra from the National Institute of Standards and Technology (NIST) Chemistry WebBook [37], the Spectral Database for Organic Compounds (SDBS) [38], and USPTO [36] for functional-group prediction, molecular structure elucidation, and physicochemical property prediction. Mixture analysis uses NIST, SDBS, and USPTO spectra for targeted component detection and targeted fractional contribution estimation, the DeepMIR liquid-mixture dataset for additional evaluation of targeted component detection [22], and FTIR mixture spectra for mixture-level component quantification [39]. Real-world chemical sensing applications include genus-level classification of bacterial isolates from green snow [40], medicinal-herb geographic origin traceability and constituent quantification, microplastics classification using environmentally sourced spectra [7], and soil property prediction using the Open Soil Spectral Library [6]. Together, these datasets span chemical sensing and analysis from molecules and mixtures to biological and botanical samples and complex environmental samples.

UltraIR follows a two-stage simulation-to-real learning framework (Fig. 1a). During pretraining, a shared encoder learns spectral representations from simulated IR spectra through three complementary objectives. Wavelet-domain spectral reconstruction encourages the encoder to retain coarse spectral envelopes and fine absorption features; molecular fingerprint similarity alignment encourages the latent space to reflect graded structural similarity among molecules; and multi-label functional-group prediction links the learned representations to interpretable chemical motifs. To gether, these objectives integrate spectral morphology, molecular structural relationships, and functional-group information into a shared representation. For downstream adaptation, the pretrained encoder initializes the shared representation backbone, while a newly initialized task-specific head is added for each downstream objective. The encoder and head are then jointly optimized using the corresponding supervised data. This framework combines broad exposure to chemical diversity through pretraining on simulated spectra with task-specific supervision, enabling transfer of the shared representation across analytical objectives.

![](images/60a0cf82c539be39a73318fa0b24f36a804c3f6af7b949574ff4dd010ab75612.jpg)

Figure 1 | Overview of the UltraIR simulation-to-real framework, encoder architecture, and molecular- and mixture-level analytical tasks. a, UltraIR data resources and the two-stage simulation-to-real learning framework. The pretraining dataset contains approximately 60 million simulated IR spectra assembled from publicly released spectrum–molecule pairs and spectra predicted or simulated for PubChem-derived molecules using machine-learning models and molecular dynamics The shared encoder is pretrained with three chemically motivated objectives: wavelet-domain spectral reconstruction, molecular fingerprint similarity alignment and multi-label functional-group prediction. For downstream adaptation, the pretrained encoder initializes the shared representation backbone and is jointly optimized with a newly initialized task-specific head using supervised downstream data, including approximately 120,000 experimental IR spectra from gas-, liquid-, and solid-phase samples. b, UltraIR encoder architecture. The original spectrum is concatenated with first- and second-derivative channels generated by the derivative encoding module and processed through hierarchical residual spectral blocks. The resulting features are concatenated with a [CLS] token, combined with positional encodings, and processed by a Transformer encoder to produce the final [CLS] representation. Colors distinguish UltraIR components and separate trainable from non-trainable operations; symbols denote concatenation, summation, and element-wise multiplication. c, Molecular- and mixture-level analytical tasks used to evaluate UltraIR, comprising functional-group prediction, molecular structure elucidation, physicochemical property prediction, and mixture-component identification and quantification. Mixture analysis includes targeted component detection targeted fractional contribution estimation, and mixture-level component quantification.

## Downstream task landscape

To evaluate whether the shared spectral representation transfers across analytical objectives and datasets, we assessed UltraIR across a downstream task landscape spanning molecular- and mixture-level benchmarks and real-world chemical sensing applications. The molecular- and mixture-level benchmarks link IR spectra to functional-group annotations, molecular structures, physicochemical properties, and mixture composition (Fig. 1c), whereas the real-world applications evaluate chemical inference from complex biological, botanical, and environmental samples.

Functional-group prediction evaluates whether UltraIR can identify functional groups from their characteristic vibrational features. Molecular structure elucidation evaluates whether UltraIR can identify the correct molecular structure from spectral information under a specified molecular-formula constraint. Physicochemical property prediction examines whether the shared spectral representation encodes continuous molecular attributes beyond discrete structural annotations. Mixture analysis assesses component detection and quantification through three tasks. In targeted component detection, the model receives a single-compound reference spectrum and a mixture spectrum and predicts whether the corresponding compound is present in the mixture. In targeted fractional contribution estimation, the same pair of spectra is used to estimate the target compound’s fractional contribution. In mixture-level component quantification, the model receives only a mixture spectrum and predicts the proportions of multiple components.

Real-world chemical sensing applications span four sample domains and multiple analytical objectives. Genus-level bacterial classification evaluates whether biological spectral fingerprints can distinguish fast-growing bacterial isolates collected from green snow. Medicinal-herb characterization combines geographic origin traceability with chemical constituent quantification, requiring both categorical and continuous inference from botanical spectra. Microplastics classification uses environmental IR spectra to distinguish polymer types. Soil property prediction evaluates whether IR spectra support the prediction of continuous soil physicochemical properties.

The following sections benchmark UltraIR against established conventional machine-learning and task-specific deeplearning methods appropriate to each task.

## UltraIR enables molecular structural interpretation from infrared spectra

We first evaluated UltraIR on functional-group prediction and molecular structure elucidation using the public experimental NIST and SDBS datasets and the simulated USPTO benchmark [36–38].

## Functional-group prediction.

For functional-group prediction, UltraIR was benchmarked against task-specific deep-learning models, including FCG-Former and IRAnalysis [17, 18], and conventional machine-learning baselines, including XGBoost, random forest (RF), k-nearest neighbors (KNN), and a logistic classifier implemented using scikit-learn [41–43]. Performance was evaluated using three standard and complementary metrics: Micro-F1, Macro-F1, and exact match ratio (EMR). Micro-F1 and Macro-F1 are commonly used label-level metrics for multi-label classification. Micro-F1 aggregates true positives, false positives, and false negatives across all functional-group labels, providing an overall measure of label-wise prediction performance, whereas Macro-F1 first computes the F1 score for each functional group and then averages across groups, giving equal weight to frequent and infrequent functional groups. In contrast, EMR provides a molecule-level assess ment of complete functional-group assignment: a prediction is counted as correct only when the full set of functional groups is recovered without missing labels or false assignments. We assessed model behavior from three perspectives: overall performance on each dataset, performance variation as a function of labeled training-data size, and performance for individual functional groups.

Across the three datasets and all three metrics, UltraIR consistently and substantially outperformed both task-specific deep-learning models and conventional machine-learning baselines (Fig. 2a). On SDBS and USPTO, FCGFormer and IRAnalysis either only marginally outperformed XGBoost or performed comparably to it. On NIST, both models slightly underperformed XGBoost across all three metrics. These results demonstrate that task-specific deep learning alone does not confer a decisive advantage over strong conventional baselines for functional-group prediction. By contrast, UltraIR established a clear and consistent performance advantage over both classes of methods, with the most pronounced gains in EMR on NIST and USPTO. These gains are chemically meaningful rather than merely statistical. Functional-group prediction converts an IR spectrum into an interpretable molecular annotation that chemists can use to assess molecular composition, validate spectral interpretations, and evaluate plausible structura hypotheses. A partially correct label-wise prediction may still miss a key group or introduce a false assignment, thereby altering the chemical interpretation of the molecule even when Micro-F1 or Macro-F1 remains high. Higher EMR therefore shows that UltraIR more frequently recovers the complete functional-group set, rather than merely improving isolated label detections.

We next examined whether UltraIR’s performance advantage persisted as the amount of labeled training data varied. Using EMR as the evaluation metric, UltraIR consistently outperformed all competing methods at every training proportion on NIST and USPTO (Fig. 2b). On SDBS, UltraIR performed comparably to the strongest competing method at the two lowest training proportions, 0.1% and 1%, and clearly outperformed it at all higher proportions. By contrast, the two task-specific deep-learning models, particularly FCGFormer, underperformed the strongest conventional baseline at the two lowest training proportions and were markedly inferior to UltraIR on NIST and USPTO. These results support the central simulation-to-real premise of UltraIR: large-scale simulated spectra provide transfer able chemical information that enables reliable inference across a broad range of labeled-data availability, particularly when downstream supervision is scarce.

Finally, we examined whether the aggregate performance advantage of UltraIR extended across individual functional groups rather than being driven by a small number of frequent labels. Per-group F1-score rankings placed UltraIR consistently among the top-ranked methods across NIST, SDBS, and USPTO (Fig. 2c). This pattern covered 17 functional groups spanning hydrocarbons and unsaturated motifs, oxygen- and nitrogen-containing groups, halogenated groups, and carbonyl-containing derivatives. The aggregate gains therefore reflect broad performance across chemically diverse functional groups rather than gains driven by a narrow subset of common labels. This breadth supports the use of UltraIR for functional interpretation across diverse molecular chemistry.

## Molecular structure elucidation.

For molecular structure elucidation, UltraIR was benchmarked against representative IR structure-elucidation models, including IRtoMol, AISE, PBSA, and DLIR [19, 20, 32, 44]. Given an IR spectrum and its molecular formula, each model generates a ranked list of molecular candidates represented as Simplified Molecular Input Line Entry System (SMILES) strings. Performance was evaluated using top-1, top-5, and top-10 accuracy, together with fingerprint-based Tanimoto similarity. The top-k metrics assess exact structure recovery by determining whether the reference molecule appears among the k highest-ranked candidates. Fingerprint-based Tanimoto similarity provides a complementary assessment of structural agreement between the top-ranked candidate and the reference molecule, with higher values indicating closer structural similarity. We examined performance from three perspectives: overall exact-recovery accuracy across datasets, structural similarity of top-ranked candidates, and the overlap and characteristics of molecules correctly solved by diferent methods.

Across NIST, SDBS, and USPTO, UltraIR achieved the strongest overall performance among the compared methods for molecular structure elucidation (Fig. 2d). Its top-1, top-5, and top-10 accuracies show that the learned spectral representation can prioritize the reference structure within a molecular-formula-constrained candidate set. In contrast, the other methods exhibited greater dataset-to-dataset variation in both overall performance and relative ranking.

a  
d  
![](images/0894f77a28f4cdbca2b7c0b16a38622cb2c0e18c75ee8e87d580ea618d16e1a9.jpg)

Figure 2 | UltraIR enables molecular structural interpretation from infrared spectra. a, Overall functional-group prediction performance on NIST, SDBS, and USPTO, evaluated using Micro-F1, Macro-F1, and exact match ratio (EMR). EMR counts a prediction as correct only when the complete functional-group set is recovered. b, Training-data scaling analysis for functional-group prediction, showing EMR as the fraction of labeled training spectra varies from 0.1% to 100%. c, Functional group-wise F1-score rankings across 17 chemically diverse labels on NIST, SDBS, and USPTO, with lower ranks indicating stronger relative performance for each functional group. d, Molecular structure elucidation on NIST, SDBS, and USPTO, evaluated using top-1, top-5, and top-10 accuracy. A prediction is counted as correct when at least one of the top k candidates matches the ground-truth molecular structure after canonicalization. e, Paired comparison of fingerprint-based Tanimoto similarity between each ground-truth molecule and the corresponding top-1 candidates generated by UltraIR and IRtoMol on the NIST test set. Lines connect predictions for the same molecule. f, Correct-case overlap among UltraIR, IRtoMol, AISE, PBSA, and DLIR on the NIST test set. The 15 most frequent correct-case subsets are shown together with the fraction o molecules for which all methods were incorrect. Blue and red denote subsets including and excluding UltraIR, respectively, whereas gray denotes cases missed by all methods. g, Two representative top-1 structure-elucidation examples from the NIST test set, comparing molecular structures predicted by UltraIR and competing methods.

We further assessed the structural similarity of the top-1 candidates on NIST using fingerprint-based Tanimoto similarity. For each spectrum, we compared the Tanimoto similarity between the ground-truth molecule and the top-1 candidates generated by UltraIR and IRtoMol, the strongest competing method on NIST in the exact-recovery evaluation. UltraIR produced candidates with higher, equal, and lower similarity than IRtoMol for 44%, 41%, and 15% of spectra, respectively (Fig. 2e). Thus, UltraIR matched or exceeded IRtoMol in structural similarity for 85% of the evaluated spectra, and higher-similarity candidates outnumbered lower-similarity candidates by 29 percentage points. Together with the exact-recovery results, this comparison shows that UltraIR not only more often ranks the ground truth molecule among its leading candidates, but also more often produces a top-ranked candidate that is structurally closer to the ground-truth molecule.

We next analyzed the overlap of correctly solved molecules across UltraIR, IRtoMol, AISE, PBSA, and DLIR on the NIST dataset, focusing on the 15 most frequent correct-case subsets (Fig. 2f). The largest subset comprised molecules that were not solved by any of the five methods, accounting for 44.0% of the test set. Among the displayed correct-case subsets, those including UltraIR accounted for 52.0% of test samples, whereas subsets excluding UltraIR accounted for only 2.9%. Notably, all five methods correctly solved 20.2% of the test set, while UltraIR alone correctly solved a further 9.0%. This pattern shows that UltraIR both recovers a substantial set of molecules shared with existing methods and contributes a sizable set of correct predictions not obtained by the competing structure-elucidation approaches.

Representative NIST test cases further illustrate the chemical basis of this advantage (Fig. 2g). In the first case, the ground-truth molecule comprised an oxygen-rich fused polycyclic scafold containing a methoxy substituent, cyclic ethers, and two ring-associated carbonyl groups. UltraIR recovered the exact top-1 structure with a Tanimoto similarity of 1.00. AISE and PBSA retained most of the fused scafold but introduced an incorrect double bond within an oxygen-containing ring, each yielding a similarity of 0.66. IRtoMol introduced the same bond-order error and replaced a ring carbon with an oxygen atom, resulting in a similarity of 0.53, whereas DLIR generated a substantially diferent and less oxygenated polycyclic framework with a similarity of 0.20. In the second case, the ground-truth molecule contained a nitrogen-rich heteroaromatic core bearing carboxamide and formamide functionalities and two benzyl substituents. UltraIR again recovered the exact structure with a similarity of 1.00. By contrast, the competing candidates altered the heteroaromatic core and the connectivity of the carbonyl-containing and aromatic substituents, with DLIR additionally introducing a chlorine atom absent from the ground truth. IRtoMol, AISE, PBSA, and DLIR achieved similarities of only 0.34, 0.34, 0.30, and 0.21, respectively.

Collectively, these results establish UltraIR as a strong approach for functional and structural interpretation of IR spectra, spanning functional-group prediction and molecular structure elucidation.

## UltraIR enables physicochemical property prediction from infrared spectra

Beyond molecular structural interpretation, we evaluated whether UltraIR could predict continuous molecular physico chemical properties from IR spectra. Physicochemical property prediction was evaluated on NIST, SDBS, and USPTO for 11 structure-derived molecular descriptors and scores: synthetic accessibility (SA) score, log P, topological polar surface area (TPSA), hydrogen-bond donors, hydrogen-bond acceptors, rotatable bonds, the fraction of sp<sup>3</sup>-hybridized carbon atoms (Fraction Csp3), quantitative estimate of drug-likeness (QED), aromatic rings, aliphatic rings, and Bertz complexity (BertzCT). UltraIR was compared with conventional regression baselines, including XGBoost regression, support-vector regression (SVR), KNN regression, and partial least-squares regression (PLSR) [41, 43, 45]. Performance was evaluated using normalized mean absolute error (normalized MAE), normalized root mean squared error (normalized RMSE), and the coeficient of determination $\dot { R } ^ { 2 }$ . Normalized MAE and normalized RMSE quantify scale-adjusted prediction error, whereas $R ^ { 2 }$ measures the proportion of variance explained by the model. We further analyzed property-wise $R ^ { 2 }$ rankings and BertzCT complexity-thresholded prediction error.

a  
![](images/ec1db0b2e7b9e750f34255b725c561a61f9eda5f241e4af740477217c28fb22f.jpg)

b  
![](images/30af810a60929dae30c4532f1c79b743c059285febeaf675ac3b2a745518c071.jpg)

c  
![](images/a79b04702d37c89137d29217c16426dff890e52c5d9ddc0fafd822dd24c7d13d.jpg)

![](images/b73c45f8e15229a611423590ca569d6d0ab5ccdb72f4846e1ce77a79670dde68.jpg)

![](images/a016eb84367e36350177233efc165841407c502df8bb0f41e227e7cf5df60a0f.jpg)  
Figure 3 | UltraIR enables physicochemical property prediction from infrared spectra. ${ \mathbf { a } } ,$ Overall physicochemical property prediction on NIST, SDBS, and USPTO, evaluated using normalized ${ \mathrm { M A E } } ,$ normalized RMSE, and $R ^ { 2 }$ . b, Property wise $R ^ { \dot { 2 } }$ rankings for 10 of the 11 physicochemical properties across NIST, SDBS, and $\mathrm { U S P T O }$ . The remaining property, BertzCT, is examined separately in c. Lower ranks indicate stronger relative performance for each property. $\mathbf { c } ,$ Mean relative error for BertzCT prediction across cumulative molecular-complexity thresholds on NIST, SDBS, and USPTO. At each threshold, performance is evaluated for molecules with true BertzCT values greater than or equal to that threshold.

Across all three datasets, UltraIR achieved the lowest normalized MAE and normalized RMSE, together with the highest $R ^ { 2 }$ among the evaluated methods (Fig. 3a). XGBoost was the strongest conventional baseline, whereas SVR, KNN, and PLSR showed larger normalized errors or lower $R ^ { 2 }$ values in several settings. Property-wise $R ^ { 2 }$ rankings for the ten properties included in the ranking analysis placed UltraIR first in all 30 property–dataset combinations (Fig. 3b). This consistent ranking shows that UltraIR’s advantage extends across the ten ranked properties rather than being driven by a single target.

We next examined BertzCT prediction across cumulative molecular-complexity thresholds. At each threshold, mean relative error was calculated for molecules whose true BertzCT value met or exceeded that threshold. UltraIR achieved the lowest mean relative error at every threshold across NIST, SDBS, and USPTO, whereas conventional regression baselines showed larger errors and more pronounced threshold-dependent variation, particularly for high-complexity subsets $\left( \mathrm { F i g . \ 3 c } \right)$ . These results indicate that UltraIR maintains accurate continuous-property prediction across a broad range of molecular complexity.

Together, these results extend the molecular-level interpretation capability of UltraIR beyond discrete functionalgroup labels and molecular structure elucidation. The learned IR representations capture continuous physicochemical attributes across diverse molecular properties and complexity regimes.

## UltraIR enables mixture-component identification and quantification from infrared spectra

We evaluated targeted component detection and fractional contribution estimation across NIST, SDBS, and USPTO. Targeted component detection under increasing mixture complexity was further evaluated using the DeepMIR liquid mixture dataset [22], whereas mixture-level component quantification was assessed using FTIR mixture datasets [39].

## Targeted component detection and fractional contribution estimation.

For pairwise mixture analysis, we evaluated UltraIR on targeted component detection and targeted fractional contribution estimation across NIST, SDBS, and USPTO. In both tasks, the input consisted of a reference single-compound spectrum and a mixture spectrum. For targeted component detection, UltraIR was benchmarked against DeepMIR [22], reverse match, and the hit quality index (HQI). Performance was evaluated using accuracy, Macro-F1, and the area under the receiver operating characteristic curve (ROC-AUC). For targeted fractional contribution estimation, UltraIR was compared with a DeepMIR-derived regression baseline and conventional regression baselines, including XGBoost regression, support-vector regression (SVR), KNN regression, and partial least-squares regression (PLSR). Performance was evaluated using mean absolute error (MAE), root mean squared error (RMSE), and $R ^ { 2 }$

UltraIR achieved strong targeted component-detection performance across all three datasets (Fig. 4a). On NIST, UltraIR attained higher accuracy and Macro-F1 than DeepMIR while achieving comparable ROC-AUC. On SDBS and USPTO, UltraIR and DeepMIR performed comparably across all three metrics. Across all datasets, UltraIR consistently outperformed the direct spectral-matching baselines: reverse match and HQI.

We next examined whether this performance persisted as mixture complexity increased. UltraIR and DeepMIR performed comparably for binary and ternary mixtures, whereas UltraIR established a clearer advantage in both accuracy and Macro-F1 for quaternary mixtures (Fig. 4b). This pattern indicates that UltraIR retains stronger target-component detection as additional components increase spectral overlap and make the target contribution more dificult to isolate.

UltraIR also achieved the strongest performance in targeted fractional contribution estimation across all three datasets, substantially and significantly outperforming the strong DeepMIR-derived regression baseline. It attained the lowest MAE and RMSE, together with the highest $R ^ { 2 }$ among all evaluated methods (Fig. 4c). Its $R ^ { 2 }$ values reached 0.956 on NIST, 0.986 on SDBS, and 0.996 on USPTO (Fig. 4d). The parity plots showed that UltraIR predictions closely followed the identity line, whereas competing methods exhibited larger deviations and weaker calibration (Fig. 4d). These results establish that UltraIR supports both qualitative identification of a target component and quantitative estimation of its contribution to a mixture spectrum.

## Mixture-level component quantification.

We finally evaluated mixture-level component quantification, in which the model receives only a mixture spectrum and predicts the proportions of multiple components. This task followed a simulation-to-experiment transfer protocol: models were first trained on a simulated FTIR mixture dataset and then fine-tuned and evaluated only on experimentally acquired FTIR mixture spectra [39]. UltraIR was compared with the mixture-analysis models AIMWSP and ML-FTIR [39, 46], as well as conventional regression baselines including XGBoost regression, SVR, KNN regression, and PLSR. Performance was assessed using normalized residual distributions and component-wise $R ^ { 2 }$

UltraIR produced residuals most tightly concentrated around zero, indicating the most accurate and stable mixturelevel concentration estimates among the evaluated methods (Fig. 4e). Component-wise analysis further confirmed this advantage. UltraIR achieved the highest $R ^ { 2 }$ for each of the four quantified components: acrylonitrile, adiponitrile, propionitrile, and glycerol (Fig. 4f). Several competing methods showed pronounced component-dependent degradation. PLSR yielded negative $R ^ { 2 }$ values across all four components, and KNN also produced a negative $R ^ { 2 }$ for propionitrile, indicating performance below that of a mean-value predictor in these settings.

Together, these results show that UltraIR extends molecular-level IR analysis from individual compounds to chemical mixtures. It enables reference-guided component detection and fractional contribution estimation, as well as multi-component quantification from a single mixture spectrum. UltraIR thus supports functional-group prediction, molecular structure elucidation, physicochemical property prediction, and mixture analysis.

a  
![](images/0d3f8289f20fdc5ff943824f7a804278888a0a3ff5a12ae51def855f12ce1b21.jpg)

b  
![](images/556c47db369f48b3ddb89fe085777446fb16eeed0c3a898aef1a2a23b46db837.jpg)

c  
![](images/224cbdf4dae4ec8900fc2cb64f65c59a583f7151b2f91b3e6aae6d1d293b0528.jpg)

d  
![](images/bf9048b7d6ebdb28205151c97427d5acb168211768049b7b019f2926daea73b9.jpg)

![](images/abc2c78d00d2d70f0dbed1d8f75915ff026fe80b8d312f58e2ad2da49d45c98d.jpg)

![](images/9a795b4adcd4a40e75cd4318cfe9cacd4cde66e413e5c8e0f9cbdc1e9ebe5e27.jpg)

![](images/d58aba8123dc9640c521fa630c8a01c44c47e9e0432868a3cef7f7bde2528c74.jpg)  
Figure 4 | UltraIR enables mixture-component identification and quantification from infrared spectra. a, Targeted component detection on NIST, SDBS, and $\mathrm { U S P T O }$ , evaluated using accuracy, Macro-F1, and ROC-AUC. UltraIR is compared with DeepMIR, reverse match, and hit quality index (HQI). b, Targeted component detection in binary, ternary, and quaternary mixtures, comparing UltraIR and DeepMIR using accuracy and Macro-F1. $\mathbf { c } ,$ Targeted fractional contribution estimation on NIST, SDBS, and $\mathrm { U S P T O } ,$ evaluated using mean absolute error (MAE), root mean squared error (RMSE), and $R ^ { 2 }$ . d, Parity plots for targeted fractional contribution estimation, comparing predicted and true target-component contributions for UltraIR and competing methods across NIST, SDBS, and USPTO. The identity line denotes ideal agreement, and dashed lines show fitted relationships. $\mathbf { e } ,$ Normalized residual distributions for mixture-level multi-component quantification from mixture spectra. f, Component-wise $R ^ { 2 }$ values for mixture-level quantification of acrylonitrile, adiponitrile, propionitrile, and glycerol.

## UltraIR supports biological and botanical infrared sensing

Having established UltraIR across molecular-level tasks, we next examined whether it could transfer to biological and botanical IR sensing tasks involving experimentally acquired spectra from complex real-world samples. We considered two applications with distinct sample types and analytical objectives: genus-level bacterial classification and medicinal-

herb characterization, encompassing geographic origin traceability and chemical constituent quantification.

## Bacterial classification.

For bacterial classification (Fig. 5a), UltraIR was evaluated on FTIR spectra of fast-growing bacterial isolates from green snow [40]. The task was formulated as genus-level classification and benchmarked against conventional machinelearning baselines, including XGBoost, RF, KNN, and a logistic classifier. Performance was evaluated using accuracy, Macro-F1, and the Matthews correlation coeficient (MCC), which provides a robust measure under class imbalance.

UltraIR achieved the highest accuracy, Macro-F1, and MCC among all compared methods (Fig. 5c). Genus-wise F1 analysis further showed that UltraIR consistently outperformed the conventional baselines across all nine evaluated genera, whereas baseline performance varied substantially among genera (Fig. 5d). Thus, UltraIR’s aggregate advantage reflected broad genus-level discrimination rather than performance gains confined to a small subset of genera.

## Medicinal herb characterization.

We next evaluated UltraIR on two newly generated in-house experimental datasets of Jinyinhua (Lonicerae Japonicae Flos) and Shanyinhua (Lonicerae Flos), comprising newly acquired IR spectra for both geographic origin traceability and chemical constituent quantification (Fig. 5b). The origin-traceability datasets comprised 120 Jinyinhua spectra from Shandong, Henan, Hebei, and Sichuan and 150 Shanyinhua spectra from Hunan, Hubei, Sichuan, Henan, and Guangdong, with 30 spectra per origin. Liquid chromatography–mass spectrometry (LC-MS)-derived relative abundances of the target constituents, expressed in arbitrary units (a.u.), served as regression labels for 60 Jinyinhua and 75 Shanyinhua samples. Details of the in-house datasets, including dataset composition, LC-MS sample preparation, and data acquisition, are provided in the Appendix. We retained a matched UltraIR model without simulated pretraining as an ablation for these tasks. For geographic origin traceability, UltraIR and the no-pretraining ablation were compared with XGBoost, RF, KNN, and a logistic classifier using accuracy, Macro-F1, and MCC. For chemical constituent quantification, they were compared with XGBoost regression, SVR, KNN regression, and PLSR using normalized residual distributions and constituent-wise $R ^ { 2 }$

For geographic origin traceability, UltraIR achieved the highest mean accuracy, Macro-F1 and MCC on both Jinyinhua and Shanyinhua, outperforming all conventional baselines (Fig. 5e). By contrast, the no-pretraining ablation underperformed UltraIR across all three metrics on both Jinyinhua and Shanyinhua and also fell below XGBoost, the strongest conventional baseline, in every comparison. A Uniform Manifold Approximation and Projection (UMAP) of embeddings extracted by UltraIR provided a complementary qualitative view of the learned spectral organization (Fig. 5f). Spectra from the same geographic origin largely co-localized within each herb species, while Jinyinhua and Shanyinhua occupied largely distinct regions of the embedding space rather than being broadly intermixed. This organization emerged even though herb species identity was not used as a supervised label in the origin-traceability task and is consistent with UltraIR encoding spectral variation associated with both medicinal-herb identity and geographic origin.

We next moved from categorical origin assignment to quantitative estimation of chemical constituents. For both Jinyinhua and Shanyinhua, UltraIR produced normalized residual distributions that were more tightly concentrated around zero than those of the no-pretraining ablation and the conventional regression baselines, with fewer large residuals (Fig. 5g,i). The no-pretraining ablation nevertheless produced a compact interquartile residual distribution. Its interquartile range was narrower than that of KNN regression, the strongest conventional baseline by this criterion, on Jinyinhua and comparable to that of KNN regression on Shanyinhua. However, the ablation also produced more large-residual outliers.

Because residual centering and dispersion do not establish how much constituent-specific variation a model explains, we next examined constituent-wise $R ^ { \bar { 2 } }$ . UltraIR achieved the highest $R ^ { 2 }$ for every evaluated constituent in both Jinyinhua and Shanyinhua, with particularly large gains over the no-pretraining ablation (Fig. 5h,j). Several conventional regression baselines showed marked constituent-dependent degradation and, in some cases, negative $R ^ { 2 }$ values. The no pretraining ablation also showed weak constituent-level generalization despite its residual distributions being centered close to zero. On Jinyinhua, its $R ^ { 2 }$ values were negative for all six constituents and exceeded only those of PLSR, the weakest method on this dataset. On Shanyinhua, it produced the lowest $R ^ { 2 }$ among all methods for three of the four constituents. The consistent contrast between UltraIR and the matched ablation provides direct evidence that simulated pretraining was critical under limited labeled supervision.

a  
![](images/47b655fd6c2c33a5cbf678530d054b66e34fb02601d7b20908720a97e73636e8.jpg)

b  
![](images/952567c702e030c2325487dbc644302c3b53654d314fa87c5c64f91f908dd4fb.jpg)

c  
![](images/542ca9face6dabc8340fedbd0fc087f6cfac42679f531c0de31c51169b90610d.jpg)

d  
![](images/b459755b3345fb34cd13eceec04516069804ba1cc6cc3dd86835a7acbd7cb1e9.jpg)

e  
![](images/da142576610bafd8c4a3fd64984a02794cf47a3348cd170b07082bf543db8515.jpg)

f  
![](images/bd5e31d1c45b75db69fc378992779bc70f21341fe487af6140fde6d8cb9f3798.jpg)

g  
![](images/54763c293e4c46e6608abe78ef365faa7e59b14b1a81eb1ceae13a21203a0eee.jpg)

![](images/27985fd944a96611c45bf4363c96aa91a9eae10f55a9f9edb85eea15dc1bfacc.jpg)

i  
![](images/e4298c47376bb2bda947a45830590de414522c41a14e8ed8f690f86441899631.jpg)

![](images/dd200d27e1aa534693e267830fad37a0e6232de1bd81e86cd211cb58a6cc5e14.jpg)

Figure 5 | UltraIR supports biological and botanical infrared sensing. a, Schematic of genus-level bacterial classification from infrared spectra. $\mathbf { b } ,$ Schematic of medicinal-herb characterization using in-house experimental datasets, comprising geographic origin traceability and chemical constituent quantification for Jinyinhua and Shanyinhua. $\mathbf { c } ,$ Overall genus-level bacterial classification, evaluated using accuracy, Macro-F1, and the Matthews correlation coeficient (MCC). d, Genus-wise F1 scores across the nine evaluated bacterial genera. e, Geographic origin traceability of Jinyinhua and Shanyinhua, evaluated using accuracy, Macro-F1, and MCC. UltraIR is compared with its matched no-pretraining ablation and conventional classifiers. f, Uniform Manifold Approximation and Projection (UMAP) of medicinal-herb spectral embeddings extracted by UltraIR. Points are colored by geographic origin, and marker shapes indicate herb species. $\mathbf { g } ,$ Normalized residual distributions for chemical constituent quantification in Jinyinhua, comparing UltraIR, its matched no-pretraining ablation, and conventional regression baselines. h, Constituent-wise $\bar { R ^ { 2 } }$ values for the six quantified Jinyinhua constituents across the same methods. i, Normalized residual distributions for chemical constituent quantification in Shanyinhua, comparing UltraIR, its matched no-pretraining ablation, and conventional regression baselines. j, Constituent-wise $R ^ { \dot { 2 } }$ values for the four quantified Shanyinhua constituents across the same methods.

Together with the bacterial classification results, these findings show that UltraIR supports IR analysis of complex biological and botanical samples across genus-level identification, medicinal-herb geographic origin traceability, and chemical constituent quantification.

## UltraIR supports environmental infrared sensing

Having established UltraIR’s performance in biological and botanical IR sensing, we finally evaluated the model for environmental IR sensing using experimentally acquired spectra from complex samples. We considered two representative environmental applications: microplastics classification and soil physicochemical property prediction.

## Microplastics classification.

For microplastics classification (Fig. 6a), UltraIR was evaluated on IR spectra of environmentally sourced microplastics [7]. UltraIR was compared with task-specific deep-learning baselines, including Softmax and DB-CNN-CBAM [7, 47], and conventional machine-learning baselines, including XGBoost, RF, KNN, and a logistic classifier. Performance was evaluated using accuracy, Macro-F1, and MCC. We further analyzed the true-class margin, defined as the correct-class logit minus the highest logit among the non-true classes. Larger positive values indicate stronger separation of the correct class from its nearest competitor.

Across accuracy, Macro-F1, and MCC, UltraIR achieved the strongest performance among all evaluated methods (Fig. 6b). Its advantage over both task-specific deep-learning and conventional machine-learning baselines demonstrates efective transfer from molecular benchmarks to polymer identification in complex environmental spectra.

We next examined whether this aggregate performance advantage was accompanied by larger true-class margins. In the pairwise margin plots, many spectra lay above the diagonal, indicating that UltraIR assigned a larger trueclass margin than Softmax or DB-CNN-CBAM, including numerous cases in which both methods correctly predicted the polymer class (Fig. 6c). UltraIR also corrected substantially more baseline errors than it introduced. Relative to Softmax, UltraIR corrected 472 baseline errors while introducing 112 errors. Relative to DB-CNN-CBAM, it corrected 657 baseline errors while introducing 97 errors. Thus, UltraIR not only corrected more misclassified spectra, but also achieved stronger separation between the correct polymer class and its nearest alternatives.

Finally, class-wise F1 analysis showed that UltraIR maintained consistently high performance across all 18 polymer categories, including common commodity polymers and more specialized materials (Fig. 6d). This broad class-wise advantage shows that the overall improvement was not driven by a narrow subset of readily distinguishable polymers, but extended across the microplastics classification label space.

## Soil property prediction.

For soil property prediction (Fig. 6e), we evaluated whether UltraIR could estimate continuous physicochemical properties from soil IR spectra using the Open Soil Spectral Library (OSSL) [6]. UltraIR was compared with soil spectralmodeling baselines, including GADF-Swin and LSTM-CNN [48, 49], as well as conventional regression baselines, including XGBoost regression, SVR, KNN regression, and PLSR. Performance was evaluated using normalized MAE and RMSE, together with property-wise $R ^ { 2 }$ , across ten soil properties: total carbon, total nitrogen, total sulfur, clay content, pH $\mathrm { ( H _ { 2 } O ) }$ , cation-exchange capacity (CEC), and exchangeable Ca, Mg, K, and Na.

a  
![](images/d85076acecea4a2bb6d3049ea26105c77c2a9ee0c01c78dd4e049594034730e1.jpg)

e  
![](images/d6626a66ad79c3ccf0447e89db64d9434847aa3612bc9fd4b3085bf7cb3ec29e.jpg)

b  
![](images/6092ab4196c99233f0a348fe26e8afa08f6ea7436eecb3a5786c5d593060fe82.jpg)

f  
![](images/03d6e94092513b0c8cfef922b15fe85bec026d9cbb61ed1fe7336633027f638c.jpg)

c  
![](images/64a3abac0934f631b294a5d1735ff5866180cb919b9142285190a8f6195886bd.jpg)

![](images/2263fdbc369ccec830d0d1a0bae4b6f1b585f55e3d9ce57587e831cbc319712b.jpg)

![](images/20561b6bfc1e2eae1c0de9db66db95318df41bb8be24b042f9525d4803c09a62.jpg)

![](images/73cfe80d760339c8bc256157a4498701dd3379f4636ff0f605b09aa3a90e16c2.jpg)

d  
![](images/c1ad743876c9af9f79df36a623f2f9249eea2e6082e48baaaf9f18e112127883.jpg)

h  
![](images/2d53e9ac73bf0685c97df5abb756e21e3af1a8b807e77b14e3ac7b9a45b08f52.jpg)

![](images/d4fdf1273b19d5d568c0fe865a109ae9cef05ed8c0ee9cd4cbe63a862d5c8251.jpg)

![](images/63b311be8a2ab8681944b7f9a817b2672226aebb811bbb176f3d2397ef547ad4.jpg)

![](images/31388acbcf411bc9236a2dd884427b4a79efeb775817c09801b2fafc9902ef35.jpg)

![](images/9d0fc4d41b9f2e1cf407d4070b6e5633bebb24316be189ec8ebac618e6ee5468.jpg)

![](images/69afe62475e11b16e613e2c2d41c3eb47b867922705df4c1874564f1a7234164.jpg)

![](images/9d56c7214e1f261474407eb6345e26d21b42721812c5b51dca212f1d0fba2aa6.jpg)

![](images/d4149f84e4d39039c1476114f64f7218fa1bc0bfca8f6e228a772ce69b88c3d9.jpg)

![](images/c53b064a3a08fa7e93f18018073a455378a0a6e3b08352c907fad85b0efe8297.jpg)

![](images/b5708a0e38b33c85b6194dd80906edb92252f9ea830182d43fcc38f605f86f0f.jpg)

i  
![](images/9837cf98f7639bb857504a7d0dffcd7c04c9a3b62517e35513da626ab4ab2452.jpg)

![](images/5d755ddebc17b293ae6d5362089cda5cbfe014dcaa01e30e5baa9dd249393b18.jpg)

Figure 6 | UltraIR supports environmental infrared sensing across microplastics and soil analysis. a, Schematic of microplastics classification from infrared spectra of environmentally sourced microplastics. $\mathbf { b } ,$ Overall microplastics classification, evaluated using accuracy, Macro-F1, and the Matthews correlation coeficient (MCC). $\mathbf { c } ,$ Pairwise true-class margin comparisons for UltraIR versus Softmax and UltraIR versus DB-CNN-CBAM. The true-class margin is defined as the correct-class logit minus the highest logit among the non-true classes, with points above the diagonal indicating a larger margin for UltraIR Corrected and degraded samples denote cases classified correctly by UltraIR but incorrectly by the comparator, and vice versa, respectively. d, Class-wise F1 scores across 18 polymer categories. $\mathbf { e } ,$ Schematic of soil physicochemical property prediction from infrared spectra. $\mathbf { f } ,$ Overall soil physicochemical property prediction, evaluated using normalized mean absolute error (normalized MAE), normalized root mean squared error (normalized RMSE), and $R ^ { 2 }$ $\mathbf { g } ,$ Training-data scaling analysis for soil physicochemical property prediction, reporting normalized RMSE and $R ^ { 2 }$ as the labeled training fraction varies from 0.1% to 100%. h, Parity plots for soil property prediction, comparing predicted and measured values for UltraIR, GADF-Swin, and LSTM-CNN across ten properties: total carbon, total nitrogen, total sulfur, clay content, pH $\mathrm { ( H _ { 2 } O ) }$ , cation-exchange capacity (CEC), and exchangeable Ca, Mg, K, and Na. The identity line denotes ideal agreement. i, Cross-instrument and cross-laboratory soil property prediction. Bars show property-wise $R ^ { 2 }$ for soil texture and acidity and for soil chemical properties.

Across the three aggregate regression metrics, UltraIR achieved the strongest overall performance in soil property prediction (Fig. 6f). It attained the lowest normalized MAE and normalized RMSE and the highest $R ^ { 2 }$ among all evaluated methods.

We further assessed performance under reduced labeled-data settings. As the training-set fraction decreased from 100% to 0.1%, normalized RMSE increased and $R ^ { 2 }$ decreased across all methods (Fig. 6g). At the 1% training fraction, UltraIR achieved lower normalized RMSE than XGBoost regression and KNN regression and attained the highest $R ^ { 2 }$ among all evaluated methods, although its normalized RMSE was slightly higher than that of PLSR. At this fraction, UltraIR substantially outperformed the task-specific neural baselines GADF-Swin and LSTM-CNN, with LSTM-CNN showing particularly severe performance degradation. As the labeled training fraction increased to 10%, 50%, and 100%, both task-specific models achieved progressively lower normalized RMSE and higher $R ^ { 2 }$ , but remained inferior to UltraIR. This strong performance with only 1% of the labeled training data demonstrates the data eficiency of UltraIR for soil spectroscopy, which is practically important because soil reference measurements are expensive to acquire and are often unevenly available across soil properties and sampling regions.

Property-wise analysis further confirmed the consistency of this advantage. UltraIR achieved the highest $R ^ { 2 }$ for all ten soil properties (Fig. 6h). Its predictions closely followed the identity line for total carbon, total sulfur, clay content, exchangeable calcium, and pH, for which it achieved particularly strong performance. Total nitrogen remained the most challenging property across the evaluated methods, yet UltraIR still achieved the highest $R ^ { 2 }$ . These results demonstrate broad predictive capability across soil attributes with distinct chemical origins, concentration ranges, and spectral signatures.

Together with the microplastics results, these findings show that UltraIR supports environmental IR sensing across both discrete material classification and continuous quantitative estimation. Its accuracy, robustness under limited labeled data, and consistent property-wise performance support reliable IR-based analysis of complex environmental samples.

## Soil property prediction as a case study of cross-instrument and cross-laboratory generalization

To examine UltraIR’s capacity to generalize across FTIR instruments and laboratories, we used soil property prediction as a case study. UltraIR was compared with the same soil spectral-modeling baselines: GADF-Swin, LSTM-CNN, XGBoost regression, SVR, KNN regression, and PLSR. Relative to the preceding experiment, this comparison used diferent subsets of soil IR spectra from OSSL [6] and a partially diferent set of soil properties. Performance was evaluated using property-wise $R ^ { 2 }$ for clay content, silt content, sand content, pH $\mathrm { ( H _ { 2 } O ) }$ , CEC, and exchangeable Ca, Mg, K, and Na.

All models were trained using spectra from the Kellogg Soil Survey Laboratory (KSSL) subset of OSSL and evaluated using zero-shot target-domain inference on the ICRAF–ISRIC subset of OSSL. The source and target subsets difered in laboratory, FTIR instrument, and sample preparation procedures. Detailed metadata and label distributions for both subsets are provided in Appendix Tables D and E.

UltraIR achieved the highest $R ^ { 2 }$ for all soil texture and acidity properties and for most soil chemical properties (Fig. 6i). UltraIR outperformed all competing methods for CEC and exchangeable Mg, K, and $\mathrm { N a } .$ , while performing comparably to GADF-Swin and better than the remaining methods for exchangeable Ca. UltraIR was also the only method to maintain positive $R ^ { 2 }$ across all evaluated properties. Performance diferences were most pronounced for the chemical properties, for which several baselines yielded negative $R ^ { 2 } { \mathrm { : } }$ , particularly for exchangeable Na. This case study demonstrates that UltraIR achieved more stable and generally stronger performance during zero-shot inference across instruments and laboratories, which represents an important practical challenge in real-world chemical sensing.

## 3 Discussion

IR spectroscopy is widely used for chemical sensing, but extracting reliable chemical information from complex spectra at scale remains dificult across analytical objectives and datasets. UltraIR addresses this challenge by enabling simulation-to-real transfer learning through a foundation model for IR spectroscopy with more than 100 million parameters. The model learns a shared spectral representation by pretraining on approximately 60 million simulated IR spectra using complementary objectives for spectral reconstruction, molecular fingerprint similarity alignment, and functional-group prediction. The pretrained encoder is then adapted to each downstream analytical objective using task-specific supervision. This design separates large-scale spectral representation learning from downstream adaptation, allowing the same pretrained encoder to support diverse forms of IR chemical sensing and analysis.

A central advance of UltraIR lies in the breadth of chemical inference supported by its shared pretrained encoder. The evaluated tasks span molecular structural interpretation, physicochemical property prediction, mixture-componen identification and quantification, and real-world chemical sensing in biological and botanical samples and complex environmental samples, with two newly generated in-house experimental datasets used for botanical sensing. These tasks difer in their inputs, output spaces, and analytical objectives, encompassing multi-label classification, molecular generation, regression, reference-guided mixture analysis, and multi-component quantification. Across these evaluations, UltraIR outperformed conventional machine-learning and task-specific deep-learning baselines. This consistent performance across tasks and datasets indicates that the learned representation can transfer after supervised adaptation rather than remaining tied to a single label space or dataset. UltraIR therefore provides a practical basis for reusing a shared pretrained encoder with a task-specific output module for each analytical objective.

The practical value of UltraIR’s simulation-to-real transfer learning was particularly evident in two challenging realworld chemical sensing scenarios: downstream adaptation with limited labeled experimental spectra and zero-shot target-domain inference for the same analytical task across FTIR instruments and laboratories. In training-data scaling experiments, UltraIR retained stronger performance than task-specific neural baselines as labeled supervision decreased. The medicinal-herb ablation further showed that simulated pretraining benefited complex sample analysis under limited labeled supervision. A complementary soil property prediction case study showed that UltraIR maintained more stable and generally stronger performance than competing methods during zero-shot target-domain inference across instruments and laboratories. Simulated pretraining provides broad chemical exposure, whereas taskspecific labels align the shared representation with individual analytical objectives, together supporting data-eficient learning and generalization across measurement conditions.

Although UltraIR demonstrated strong performance across diverse infrared chemical analysis tasks, several limitations remain and point to directions for further improvement. UltraIR still requires supervised adaptation and a task-specific output module for each analytical objective. The present study therefore demonstrates broad transferability across analytical objectives after adaptation, rather than zero-shot or universal inference across tasks. A simulation-to-rea domain gap also remains because simulated IR spectra cannot fully reproduce experimental variation. This variation can arise from instrument response, spectral resolution, measurement geometry, sample state, preparation procedures, temperature, concentration, scattering, complex matrices, and baseline artifacts. Improving simulation fidelity wil require explicit modeling of these factors. Future adaptation strategies could exploit unlabeled experimental IR spectra to reduce the need for labeled experimental spectra.

Overall, UltraIR establishes a scalable framework for simulation-to-real transfer learning in foundation modeling for IR spectroscopy. By coupling pretraining on chemically diverse simulated spectra with supervised downstream adaptation, UltraIR supports chemical sensing and analysis across analytical objectives and datasets, spanning molecules and mixtures, biological and botanical samples, and complex environmental samples. This framework provides a route beyond collections of independently trained task-specific models toward adaptable and data-eficient IR-based chemica sensing systems built on reusable spectral representations.

## 4 Methods

## Overview of UltraIR

UltraIR is a foundation model for IR spectroscopy with more than 100 million parameters that enables simulation-toreal transfer learning for chemical sensing and analysis from molecules to complex samples. The framework comprises a shared spectral encoder and task-specific output modules. During pretraining, the encoder is optimized on approximately 60 million simulated IR spectra to learn a shared, chemically informative spectral representation. For downstream adaptation, the pretrained encoder is paired with a newly initialized task-specific output module, and both components are jointly optimized using labeled spectra for the corresponding task. The encoder architecture and pretrained parameters provide a common initialization across downstream objectives, whereas the output module, output dimensionality, and supervised training objective are defined separately for each task.

In UltraIR, the shared encoder is initialized from the pretrained checkpoint. In designated ablation analyses, we additionally evaluated a matched no-pretraining control in which the same encoder architecture and task-specific output modules were initialized randomly and trained using the same downstream data and optimization procedure. Formally, given a preprocessed IR spectrum x, the shared encoder produces a latent representation

$$
\mathbf { z } = f _ { \boldsymbol { \theta } } ( \mathbf { x } ) ,\tag{1}
$$

where $f _ { \theta }$ denotes the shared encoder with parameters θ. For UltraIR, θ is initialized from the pretrained parameters $\theta _ { \mathrm { p r e } } .$ For the no-pretraining control, θ is instead initialized randomly. A task-specific output module ${ h } _ { \phi } ^ { ( t ) }$ then transforms the shared representation, together with any auxiliary input required by task t, into the corresponding analytical output:

$$
\hat { \mathbf { y } } ^ { ( t ) } = h _ { \phi } ^ { ( t ) } \left( \mathbf { z } , \mathbf { u } ^ { ( t ) } \right) ,\tag{2}
$$

where $\mathbf { u } ^ { ( t ) }$ denotes optional task-specific inputs, such as a molecular formula or a reference single-compound spectrum. The downstream outputs include functional-group annotations, formula-conditioned molecular structure candidates, molecular physicochemical property estimates, mixture-component detection and quantification outputs, bacterial genus labels, medicinal-herb geographic origin labels and constituent relative abundances, polymer-type labels for microplastics, and soil physicochemical property estimates.

## Spectral preprocessing and data augmentation

Before being supplied to the model, all IR spectra were converted to a common spectral range, sampling grid, and intensity scale. The same deterministic preprocessing pipeline was used for large-scale pretraining on simulated spectra and downstream adaptation on experimental spectra. Stochastic augmentation was stage-specific. A broader set of augmentation operations was used during pretraining, whereas only weak, validation-selected augmentations were considered during downstream adaptation.

## Standard spectral preprocessing pipeline.

Each spectrum was first cropped to the mid-IR range of $4 0 0 \mathrm { c m } ^ { - 1 }$ to $4 0 0 0 \mathrm { c m } ^ { - 1 }$ . Spectra that did not cover the complete range were zero-padded at the missing boundaries. The resulting spectrum was resampled by piecewise-linear interpolation onto a canonical 3,600-point grid and then min–max normalized to the interval [0, 1]. This canonical representation was subsequently mapped to the fixed input length required by each model implementation.

## Pretraining augmentation.

During pretraining, four stochastic augmentation operations were applied to each spectrum. Additive Gaussian noise was sampled at a random signal-to-noise ratio to represent electronic measurement noise. A random shift along the wavenumber axis represented calibration uncertainty. An additive intensity ofset represented baseline drift. Finally, a random subset of spectral positions was masked by setting the corresponding intensities to zero. This masking objective encouraged the encoder to use information from the broader spectral context rather than relying exclusively on isolated local features.

## Downstream adaptation augmentation.

For downstream adaptation, stochastic augmentation was not imposed as a universal part of the training pipeline. Instead, weak augmentations, including additive noise, small wavenumber shifts, and baseline ofsets, were enabled only when selected using the validation split for the corresponding model–task setting. Random spectral masking was not used during downstream adaptation because most downstream objectives require access to the complete experimental spectral profile.

## Hybrid convolutional–Transformer architecture

As illustrated in Fig. 1b, UltraIR uses a shared hybrid convolutional–Transformer encoder to map an IR spectrum to a latent spectral representation z for downstream chemical analysis. The encoder comprises three principal modules: a derivative-aware multi-channel input module, a hierarchical convolutional backbone with multi-scale feature fusion, and a patch-based Transformer encoder. Together, these modules combine derivative-aware spectral feature extraction, multi-scale local feature aggregation, and long-range spectral-context modeling.

## Derivative-aware multi-channel input module.

Given the preprocessed IR spectrum $\mathbf { x } \in \mathbb { R } ^ { 1 \times L }$ , where L denotes the number of spectral data points, this module augments the original absorbance signal with derivative-based spectral representations to enhance sensitivity to spectral line shape features. Specifically, x is first smoothed via a Savitzky–Golay (SG) filter to yield x˜. Two learnable derivative convolutions then compute the first- and second-order spectral derivatives ${ \bf d } _ { 1 }$ and $\mathbf { d } _ { 2 }$ , where the corresponding kernels $\mathbf { w } _ { 1 }$ and $\mathbf { w } _ { 2 }$ are initialized as first- and second-order central-diference operators, respectively, and remain trainable throughout optimization. Each derivative signal is subsequently normalized using layer normalization, and the three signals are concatenated to form the three-channel representation $\mathbf { x } _ { \mathrm { c a t } } = [ \mathbf { x } , \mathbf { d } _ { 1 } , \mathbf { d } _ { 2 } ] \in \mathbb { R } ^ { 3 \times L }$ . To adaptively reweight each spectral channel, $\mathbf { x } _ { \mathrm { c a t } }$ is passed through a gated input fusion module that computes a position-wise, inputdependent channel gate:

$$
\mathbf { g } = \sigma ( \mathbf { W } _ { 2 } ^ { \mathrm { g } } \operatorname { G E L U } ( \mathbf { W } _ { 1 } ^ { \mathrm { g } } \mathbf { x } _ { \mathrm { c a t } } ) ) \in \mathbb { R } ^ { 3 \times L } ,\tag{3}
$$

where $\mathbf { W } _ { 1 } ^ { \mathrm { g } }$ and $\mathbf { W } _ { 2 } ^ { \mathrm { g } }$ are $1 \times 1$ convolutional projections, GELU denotes the Gaussian error linear unit activation, and $\sigma ( \cdot )$ denotes the sigmoid function. The fused multi-channel representation is then obtained via element-wise modulation:

$$
\mathbf { x } _ { \mathrm { m u l t i } } = \mathbf { x } _ { \mathrm { c a t } } \odot \mathbf { g } \in \mathbb { R } ^ { 3 \times L } ,\tag{4}
$$

where ⊙ denotes element-wise multiplication.

## Convolutional spectral encoder.

The fused representation $\mathbf { x } _ { \mathrm { m u l t i } } \in \mathbb { R } ^ { 3 \times L }$ is fed into a convolutional encoder that hierarchically extracts local spectral features across multiple spectral scales. Prior to feature extraction, $\mathbf { x } _ { \mathrm { m u l t i } }$ is rescaled by a learnable per-channel scale vector $\gamma \in \mathbb { R } ^ { 3 }$ , broadcast along the spectral dimension and initialized as $[ 1 . 0 , 0 . 5 , 0 . 5 ] ^ { \top }$ to encode the inductive bias that the original absorbance signal should initially dominate over the two derivative channels, while allowing the network to adjust this balance during training, yielding $\hat { \mathbf { x } } _ { \mathrm { m u l t i } } \in \mathbb { R } ^ { 3 \times L }$ . The encoder then begins with a convolutional stem that applies a wide-kernel convolution followed by batch normalization and a GELU activation to $\hat { \mathbf { x } } _ { \mathrm { m u l t i } }$ , mapping it to an initial feature map $\mathbf { H } _ { 0 } \in \mathbb { R } ^ { C _ { 0 } \times L }$ , where $C _ { 0 }$ denotes the base channel width.

Four successive residual blocks then progressively encode the representation. The first three blocks each apply a strided convolution with stride 2, yielding intermediate feature maps $\mathbf { F } _ { 1 } \in \mathbb { R } ^ { C _ { 1 } \times L / 2 } , \mathbf { F } _ { 2 } \in \mathbb { R } ^ { C \times L / 4 }$ , and $\mathbf { F } _ { 3 } \in \bar { \mathbb { R } } ^ { \bar { C } \times L ^ { \prime } }$ where $C _ { 1 } = 2 C _ { 0 } , C = 4 C _ { 0 }$ , and $L ^ { \prime } = L / 8$ . The fourth block operates at the same spectral resolution, producing $\mathbf { F } _ { 4 } \in \mathbb { R } ^ { C \times L ^ { \prime } }$ . Each residual block comprises two convolutional layers, each followed by batch normalization, with a GELU activation applied after the first convolution–batch normalization (Conv-BN) layer and again after the residual addition, together with a squeeze-and-excitation (SE) module for adaptive channel-wise recalibration. Formally, for block $i \in \{ 1 , 2 , 3 , 4 \}$ with input $\mathbf { F } _ { i - 1 }$ (where $\mathbf { F } _ { 0 } = \mathbf { H } _ { 0 } )$ , the residual branch is defined as:

$$
f _ { i } ( \mathbf { F } ) = \mathrm { B N } \Big ( \mathbf { W } _ { i } ^ { ( 2 ) } * \mathrm { G E L U } \Big ( \mathrm { B N } \Big ( \mathbf { W } _ { i } ^ { ( 1 ) } * \mathbf { F } \Big ) \Big ) \Big ) ,\tag{5}
$$

where $\mathbf { W } _ { i } ^ { ( 1 ) }$ and $\mathbf { W } _ { i } ^ { ( 2 ) }$ are the convolutional filters of the two Conv-BN layers within block i. The block output is then obtained by adding the residual branch to the shortcut and applying a final activation:

$$
\begin{array} { r } { \tilde { \mathbf { F } } _ { i } = \mathrm { G E L U } ( f _ { i } ( \mathbf { F } _ { i - 1 } ) + \mathcal { P } _ { i } ( \mathbf { F } _ { i - 1 } ) ) , } \end{array}\tag{6}
$$

where $\mathcal { P } _ { i } ( \cdot )$ is a strided pointwise projection for the first three blocks and the identity mapping for the fourth. The SE module then computes a channel-wise attention vector:

$$
\mathbf { s } _ { i } = \sigma \big ( \mathbf { W } _ { i , 2 } ^ { \mathrm { s e } } \operatorname { R e L U } \big ( \mathbf { W } _ { i , 1 } ^ { \mathrm { s e } } \operatorname { G A P } ( \tilde { \mathbf { F } } _ { i } ) \big ) \big ) ,\tag{7}
$$

where ReLU(·) denotes the rectified linear unit activation, $\mathrm { G A P } ( \cdot )$ denotes global average pooling, and $\mathbf { W } _ { i , 1 } ^ { \mathrm { s e } }$ and $\mathbf { W } _ { i , 2 } ^ { \mathrm { s e } }$ are pointwise linear projections. The recalibrated block output is then $\mathbf { F } _ { i } = \tilde { \mathbf { F } } _ { i } \odot \mathbf { s } _ { i }$ , where $\mathbf { s } _ { i }$ is broadcast along the spectral dimension.

To preserve fine-grained spectral details from shallower layers, multi-scale feature fusion is performed by resampling $\mathbf { F } _ { 2 }$ and $\mathbf { F } _ { 1 }$ to resolution $\bar { L ^ { \prime } }$ via nearest-neighbor interpolation $\mathcal { R } ( \cdot )$ , with $\mathbf { F } _ { 1 }$ additionally projected to $C$ channels via a pointwise convolution $\phi ( \cdot )$ . The resampled features are concatenated with $\mathbf { F } _ { 4 }$ along the channel dimension to form

$$
\mathbf { Y } = \left[ \mathbf { F } _ { 4 } , \mathcal { R } ( \mathbf { F } _ { 2 } ) , \phi ( \mathcal { R } ( \mathbf { F } _ { 1 } ) ) \right] \in \mathbb { R } ^ { 3 C \times L ^ { \prime } } .\tag{8}
$$

A pointwise channel-mixing multilayer perceptron (MLP) $\psi ( \cdot )$ is then applied independently at each spectral position to reduce the channel dimension from $3 C$ to $C ,$ producing the unified feature map:

$$
\mathbf { F } = \boldsymbol { \psi } ( \mathbf { Y } ) \in \mathbb { R } ^ { C \times L ^ { \prime } } .\tag{9}
$$

Patch-based Transformer encoder.

The convolutional feature map $\mathbf { F } \in \mathbb { R } ^ { C \times L ^ { \prime } }$ is tokenized by a strided convolutional patch embedding layer with kernel size $P$ and stride $P / 2$ , which projects each overlapping local segment into a d-dimensional token embedding and produces a sequence of N patch tokens $\mathbf { E } \in \mathbb { R } ^ { N \times d }$ . A learnable [CLS] token $\mathbf { e } _ { \mathrm { c l s } } \in \mathbb { R } ^ { d }$ is prepended, and a learnable positional embedding $\mathbf { E } _ { \mathrm { p o s } } \dot { \in \mathbb { R } } ^ { ( N + 1 ) \times d }$ is added to encode sequential positional information, yielding the initial token sequence:

$$
\begin{array} { r } { \mathbf { T } ^ { ( 0 ) } = \left[ \mathbf { e } _ { \mathrm { c l s } } ; \mathbf { E } \right] + \mathbf { E } _ { \mathrm { p o s } } \in \mathbb { R } ^ { ( N + 1 ) \times d } . } \end{array}\tag{10}
$$

The sequence is then processed by $N _ { \mathrm { l a y e r s } }$ stacked Transformer encoder layers with pre-layer normalization, where each layer applies a multi-head self-attention sub-layer (H heads, per-head dimension $d _ { h } = d / H )$ followed by a two-layer feed-forward network with GELU activation and hidden dimension 4d, each preceded by layer normalization and wrapped with a residual connection. The final hidden state corresponding to the [CLS] token is extracted from the last Transformer layer as the global latent spectral representation:

$$
\mathbf { z } = \mathbf { T } _ { [ 0 ] } ^ { ( N _ { \mathrm { l a y e r s } } ) } \in \mathbb { R } ^ { d } .\tag{11}
$$

## Task-specific downstream architectures

The hybrid convolutional–Transformer encoder is shared across downstream tasks. Most classification and regression tasks apply the shared spectral encoder to a single input spectrum and pass the resulting representation to a task-specific MLP head. Two task families require additional task-specific modules. Formula-conditioned molecular structure elucidation takes a molecular formula as an additional input and generates SMILES strings. Pairwise mixture analysis takes a reference single-compound spectrum together with a mixture spectrum to determine whether the reference compound is present and to estimate its fractional contribution. By contrast, mixture-level component quantification from a mixture spectrum alone follows the standard single-spectrum regression pipeline. The following sections describe the additional modules used for structure generation and pairwise mixture analysis.

## Molecular structure elucidation.

Molecular structure elucidation is formulated as formula-conditioned autoregressive SMILES generation. Given an IR spectrum and its associated molecular formula, the model generates a ranked list of candidate SMILES strings by connecting the pretrained encoder to a molecular formula encoder, a Perceiver-style IR token resampler, a spectra vocabulary prompt module, and an autoregressive SMILES decoder.

The molecular formula string is tokenized at the character level and encoded by a formula encoder. The encoder consists of a token embedding layer with sinusoidal positional encodings and dropout, followed by $N _ { f }$ pre-norm Transformer encoder layers, each with multi-head self-attention and a GELU feed-forward sublayer, and a final layer normalization. It produces formula hidden states $\mathbf { H } _ { f } \in \mathbb { R } ^ { M _ { f } \times d }$ , where $M _ { f }$ denotes the number of formula tokens.

On the spectral side, the backbone extracts both the global [CLS] feature $\mathbf { z } \in \mathbb { R } ^ { d }$ and the full patch token sequence $\mathbf { T } _ { \mathrm { p a t c h } } \in \mathbb { R } ^ { N \times d }$ from its final Transformer layer. A projection head comprising layer normalization, a linear layer, and GELU activation maps z to a single IR memory token $\mathbf { m } _ { \mathrm { I R } } \in \mathbb { R } ^ { d }$ . The patch tokens are further compressed by a Perceiver-style resampler into K latent vectors. The resampler maintains K learnable query embeddings and refines them over $N _ { r }$ cross-attention blocks. Unlike standard post-norm Perceivers, each block applies pre-norm layer normalization separately to the queries and the key–value source before cross-attention. A pre-norm GELU feedforward sublayer follows, with residual connections throughout. Denoting the latent state at layer l as $\mathbf { X } ^ { ( l ) } \in \mathbb { R } ^ { K \times d }$ the update rule per block is:

$$
\mathbf { X } ^ { ( l ) } = \mathbf { X } ^ { ( l - 1 ) } + \mathrm { C r o s s A t t n } \Big ( \mathrm { L N } _ { q } \Big ( \mathbf { X } ^ { ( l - 1 ) } \Big ) , \ \mathrm { L N } _ { k v } ( \mathbf { T } _ { \mathrm { p a t c h } } ) \Big ) ,\tag{12}
$$

$$
\mathbf { X } ^ { \left( l \right) } \gets \mathbf { X } ^ { \left( l \right) } + \mathrm { F F N } \left( \mathbf { X } ^ { \left( l \right) } \right) ,\tag{13}
$$

with $\mathbf { X } ^ { ( 0 ) } = \mathbf { Q } _ { \mathrm { l a t } } \in \mathbb { R } ^ { K \times d }$ the learnable query matrix, yielding $\mathbf { M } _ { \mathrm { l a t } } = \mathbf { X } ^ { ( N _ { r } ) } \in \mathbb { R } ^ { K \times d }$

A soft vocabulary prompt token is constructed from z to bias the decoder toward chemically plausible token sequences. A task-specific MLP head maps z to logits over the SMILES vocabulary of size V. A temperature-scaled softmax then converts these logits into a probability distribution, which is used to compute a soft embedding as a weighted sum over the decoder embedding matrix $\mathbf { E } \in \mathbb { R } ^ { V \times d }$ . The soft embedding is then projected and scaled by a learnable scalar gate α to yield the prompt token:

$$
\mathbf { p } = \mathrm { s o f t m a x } \left( \frac { \mathrm { M L P } _ { \mathrm { c l s } } ( \mathbf { z } ) } { \tau } \right) \in \mathbb { R } ^ { V } , \qquad \mathbf { m } _ { \mathrm { p r o m p t } } = \alpha \cdot f _ { \mathrm { p r o j } } \big ( \mathbf { p } ^ { \top } \mathbf { E } \big ) \in \mathbb { R } ^ { d } ,\tag{14}
$$

where ${ \mathrm { M L P } } _ { \mathrm { c l s } }$ denotes the task-specific vocabulary-classification head, τ is a fixed temperature hyperparameter, and $f _ { \mathrm { p r o j } }$ denotes a three-stage projection comprising layer normalization, a linear transformation, and GELU activation.

The complete decoder memory is formed by concatenating all spectral tokens and the formula hidden states:

$$
\mathbf { M } = \left[ \mathbf { m } _ { \mathrm { I R } } ; ~ \mathbf { M } _ { \mathrm { l a t } } ; ~ \mathbf { m } _ { \mathrm { p r o m p t } } ; ~ \mathbf { H } _ { f } \right] \in \mathbb { R } ^ { ( K + 2 + M _ { f } ) \times d } .\tag{15}
$$

The SMILES decoder consists of $N _ { s }$ pre-norm Transformer layers, each applying causal self-attention under a triangular mask, cross-attention to M, and a GELU feed-forward sublayer, followed by a final layer normalization and a linear projection to vocabulary logits. During training, teacher forcing is applied, and the primary loss is token-level crossentropy $\mathcal { L } _ { \mathrm { L M } }$ . Two contrastive losses further shape the representation. An IR–formula contrastive loss $\mathcal { L } _ { \mathrm { I R - F } }$ aligns z with the masked mean-pooled formula encoding of $\mathbf { H } _ { f }$ , whereas an IR–target contrastive loss $\mathcal { L } _ { \mathrm { { I R - T } } }$ aligns z with the masked mean-pooled decoder hidden states at the final Transformer layer. Both use symmetric information noisecontrastive estimation (InfoNCE) with two-layer projection heads that map the inputs into a shared $d _ { c } .$ -dimensional contrastive space. The training objective is:

$$
\mathcal { L } _ { \mathrm { s t r u c t } } = \mathcal { L } _ { \mathrm { L M } } + \lambda _ { \mathrm { I F } } \mathcal { L } _ { \mathrm { I R - F } } + \lambda _ { \mathrm { I T } } \mathcal { L } _ { \mathrm { I R - T } } .\tag{16}
$$

At inference, beam search with length-normalized scoring generates multiple ranked candidate SMILES sequences. A formula-consistency reranking step then promotes candidates whose SMILES match the exact atomic composition of the provided molecular formula and demotes chemically invalid structures.

Targeted component detection and contribution estimation.

The pairwise mixture-analysis tasks, targeted component detection and targeted fractional contribution estimation, are unified under a common spectral comparison framework. Given a reference single-compound spectrum and a mixture spectrum, the model predicts whether the reference compound is present in the mixture and estimates its fractional contribution. The two tasks share the same pairwise representation backbone and difer only in their final prediction head.

Each input pair $\left( \mathbf { x } _ { \mathrm { r e f } } , \mathbf { x } _ { \mathrm { m i x } } \right) \ \in \ \mathbb { R } ^ { 1 \times L } \times \mathbb { R } ^ { 1 \times L }$ is encoded by three parallel branches that extract complementary representations of the spectral pair.

The first branch uses a weight-shared encoder to process each spectrum independently, producing a global [CLS] feature and a patch token sequence for each spectrum. The patch tokens are projected to $d _ { p }$ dimensions and aggregated by attentive pooling, combining an attention-weighted sum with an element-wise maximum:

$$
\mathbf { p } _ { \mathrm { p o o l } } = { \frac { 1 } { 2 } } \left( \sum _ { n = 1 } ^ { N } \alpha _ { n } \mathbf { t } _ { n } + \operatorname* { m a x } _ { n } \mathbf { t } _ { n } \right) , \qquad \alpha _ { n } = { \frac { \exp ( s _ { n } ) } { \sum _ { n ^ { \prime } } \exp ( s _ { n ^ { \prime } } ) } } ,\tag{17}
$$

where $\mathbf { t } _ { n }$ are the projected patch tokens and $s _ { n }$ is an attention score produced by layer normalization followed by a linear projection.

The second branch operates directly on the raw input signals. The reference spectrum, mixture spectrum, and their absolute pointwise diference are stacked into a three-channel tensor $[ \mathbf { x } _ { \mathrm { r e f } } , \mathbf { x } _ { \mathrm { m i x } } , \mathbf { \bar { \Pi } } ] \mathbf { x } _ { \mathrm { m i x } } - \mathbf { x } _ { \mathrm { r e f } } \left| \right] \in \mathbb { R } ^ { 3 \times L }$ . Successive Conv-BN-GELU blocks with progressively doubling channel widths process this tensor and reduce its spectral resolution by a fixed factor. The resulting feature sequence is augmented with sinusoidal positional encodings, processed by a shallow Transformer encoder, attentively pooled, and projected to yield the joint feature $\mathbf { h } _ { \mathrm { j o i n t } } \in \mathbb { R } ^ { d _ { p } }$

The third branch provides complementary frequency-domain features. A real-input fast Fourier transform (FFT) is applied to each spectrum, yielding complex coeficients $\hat { r } _ { f }$ and $\hat { m } _ { f }$ for the reference and mixture spectra at frequency bin f. Eight frequency-domain channels are then constructed: the log-amplitude spectra log $\left( 1 + | \hat { r } _ { f } | \right)$ and log(1 + $| \hat { m } _ { f } | )$ , their sum and absolute diference, the normalized cross-magnitude term $\kappa _ { f } ~ = ~ | \hat { r } _ { f } \hat { m } _ { f } ^ { * } | / ( | \hat { r } _ { f } | | \hat { m } _ { f } | + \varepsilon )$ , the normalized cross-spectral phase $\phi _ { f } = \angle ( \hat { r } _ { f } \hat { m } _ { f } ^ { * } ) / \pi$ , the log-amplitude ratio $\rho _ { f } = \log ( 1 + ( | \hat { m } _ { f } | + \varepsilon ) / ( | \hat { r } _ { f } | + \varepsilon ) )$ , and the log-amplitude product $\pi _ { f } = \log ( 1 + | \hat { r } _ { f } | \stackrel { } { | \hat { m } _ { f } | } )$ , where $( \cdot ) ^ { * }$ denotes complex conjugation, $\angle ( \cdot )$ denotes the complex argument, and ε is a small constant for numerical stability. These channels are concatenated to form the frequencydomain tensor $\mathbf { X } _ { \mathrm { f r e q } } \in \mathbb { R } ^ { 8 \times F }$ , which is processed by three convolutional sub-branches with short, medium, and long receptive fields. The resulting feature maps are concatenated, compressed by strided convolutions, globally average pooled, and projected to yield the frequency representation $\mathbf { h } _ { \mathrm { f r e q } } \in \mathbb { R } ^ { d _ { p } }$

The pair representation $\mathbf { r } _ { \mathrm { p a i r } }$ is assembled by concatenating five d -dimensional feature vectors with a ten-dimensional scalar statistics vector $\mathbf { S } _ { \mathrm { s t a t s } } .$ . The five feature vectors are the projected [CLS] features $\mathbf { c } _ { \mathrm { r e f } }$ and $\mathbf { c } _ { \mathrm { m i x } }$ , their absolute diference $| { \bf c } _ { \mathrm { r e f } } - { \bf c } _ { \mathrm { m i x } } | $ , the absolute diference of the pooled token features $\begin{array} { r } { | \mathbf { p } _ { \mathrm { r e f } } - \mathbf { p } _ { \mathrm { m i x } } | , } \end{array}$ , and the frequency feature $\mathbf { h } _ { \mathrm { f r e q } }$ . The statistics vector $\mathbf { S _ { S t a t s } }$ collects the cosine similarity between the [CLS] features, the cosine similarity between the pooled-token features, the cosine similarity between $\mathbf { h } _ { \mathrm { j o i n t } }$ and $\mathbf { h } _ { \mathrm { f r e q } } ,$ the mean squared activation energy of the joint Transformer tokens, the cosine similarity of the log-amplitude spectra, the mean and maximum absolute logamplitude diference, the mean and maximum normalized cross-magnitude term, and the mean absolute normalized cross-spectral phase:

$$
\mathbf { r } _ { \mathrm { p a i r } } = \left[ \mathbf { c } _ { \mathrm { r e f } } ; ~ \mathbf { c } _ { \mathrm { m i x } } ; ~ | \mathbf { c } _ { \mathrm { r e f } } - \mathbf { c } _ { \mathrm { m i x } } | ; ~ | \mathbf { p } _ { \mathrm { r e f } } - \mathbf { p } _ { \mathrm { m i x } } | ; ~ \mathbf { h } _ { \mathrm { f r e q } } ; ~ \mathbf { s } _ { \mathrm { s t a t s } } \right] .\tag{18}
$$

The pair representation $\mathbf { r } _ { \mathrm { p a i r } }$ is then passed to a task-specific prediction head. For targeted component detection, a binary classification head maps $\mathbf { r } _ { \mathrm { p a i r } }$ to a scalar logit and is trained with binary cross-entropy loss. For targeted fractional contribution estimation, a regression head maps $\mathbf { r } _ { \mathrm { p a i r } }$ to a scalar contribution estimate ${ \hat { a } } \in [ 0 , 1 ]$ through a sigmoid output layer and is trained with mean squared error against the normalized target contribution.

## Pretraining

In the large-scale simulation-based pretraining stage, each preprocessed IR spectrum x is first stochastically augmented to obtain an encoder input x˜. The augmented spectrum is passed through the hybrid convolutional–Transformer encoder $f _ { \theta } ,$ producing the latent representation $\mathbf { z } ~ = ~ f _ { \theta } ( \tilde { \mathbf { x } } )$ . Three complementary objectives are then applied to outputs derived from z: wavelet-domain spectral reconstruction, molecular fingerprint similarity alignment, and multilabel functional-group prediction. Together, these objectives encourage the learned representation to capture spectral morphology, molecular structural relationships, and interpretable functional-group signatures. The total pretraining loss is their weighted combination:

$$
\mathcal { L } _ { \mathrm { p r e } } = \lambda _ { \mathrm { r e c o n } } \mathcal { L } _ { \mathrm { r e c o n } } + \lambda _ { \mathrm { c o n t r a s t } } \mathcal { L } _ { \mathrm { c o n t r a s t } } + \lambda _ { \mathrm { f g } } \mathcal { L } _ { \mathrm { f g } } ,\tag{19}
$$

where $\lambda _ { \mathrm { { r e c o n } } } , \lambda _ { \mathrm { { c o n t r a s t } } }$ , and $\lambda _ { \mathrm { f g } }$ are scalar loss weights. The encoder and pretraining heads are optimized jointly by backpropagating $\mathcal { L } _ { \mathrm { p r e } }$ using AdamW. Each objective is described below.

## Wavelet-domain spectral reconstruction.

To encourage the encoder to preserve fine-grained spectral structure, wavelet-domain spectral reconstruction is used as one of the pretraining objectives. A dedicated wavelet reconstruction head maps z through a two-stage bottleneck, each stage comprising layer normalization, a linear transformation, a GELU activation, and dropout, yielding a compressed representation $\textbf { b } \in \mathbb { R } ^ { d _ { b } }$ This representation is decoded into the approximation coeficients and $J \ = \ 4$ levels of detail coeficients of a discrete wavelet transform (DWT) using a Daubechies-4 wavelet. Specifically, the predicted approximation coeficients $\hat { \mathbf { a } } \in \mathbb { R } ^ { L _ { 0 } }$ and the j-th-level detail coeficients $\hat { \mathbf { d } } _ { j } \in \mathbb { R } ^ { L _ { j } }$ are given by:

$$
\hat { \mathbf { a } } = e ^ { \alpha _ { 0 } } \mathbf { W } _ { a } \mathbf { b } , \qquad \hat { \mathbf { d } } _ { j } = e ^ { \alpha _ { j } } \mathbf { W } _ { d , j } \mathbf { b } , \quad j = 1 , \ldots , J ,\tag{20}
$$

where ${ \mathbf W } _ { a }$ and each $\mathbf { W } _ { d , j }$ are linear projections to the respective band lengths, and $\alpha _ { 0 } , \alpha _ { 1 } , \ldots , \alpha _ { J }$ are learnable logscale parameters that explicitly accommodate the heterogeneous energy scales across wavelet subbands. The predicted coeficients are then passed through the inverse DWT (IDWT) to recover the full-resolution spectral estimate $\hat { \mathbf { x } } \in \mathbb { R } ^ { L }$

The reconstruction objective combines a wavelet-coeficient-domain loss and a signal-domain loss. Denoting by a and $\{ \mathbf { d } _ { j } \} _ { j = 1 } ^ { J }$ the target DWT coeficients of the unaugmented spectrum x, the coeficient loss applies the smooth-ℓ loss $\ell _ { \mathrm { H } } ( \cdot , \cdot )$ independently to each wavelet band and averages uniformly across all J + 1 bands:

$$
\mathcal { L } _ { \mathrm { c o e f f } } = \frac { 1 } { J + 1 } \left[ \ell _ { \mathrm { H } } ( \hat { \mathbf { a } } , \mathbf { a } ) + \sum _ { j = 1 } ^ { J } \ell _ { \mathrm { H } } \left( \hat { \mathbf { d } } _ { j } , \mathbf { d } _ { j } \right) \right] .\tag{21}
$$

The signal-domain loss is computed between xˆ and the ground-truth x using a spatially weighted smooth- $\cdot \ell _ { 1 }$ loss. Leveraging the binary mask m introduced during data augmentation, masked positions are assigned an elevated weight $m _ { w } > 1$ to concentrate reconstruction capacity on the corrupted spectral regions. The per-position weight is $w _ { l } = 1 + ( m _ { w } - 1 ) m _ { l }$ , and the signal-domain loss is computed as the weighted average:

$$
\mathcal { L } _ { \mathrm { s i g n a l } } = \frac { \sum _ { l = 1 } ^ { L } w _ { l } \ \ell _ { \mathrm { H } } ( \hat { x } _ { l } , \ x _ { l } ) } { \sum _ { l = 1 } ^ { L } w _ { l } } .\tag{22}
$$

The two reconstruction terms are combined as a weighted sum:

$$
{ \mathcal { L } } _ { \mathrm { r e c o n } } = \lambda _ { \mathrm { c o e f f } } { \mathcal { L } } _ { \mathrm { c o e f f } } + \lambda _ { \mathrm { s i g n a l } } { \mathcal { L } } _ { \mathrm { s i g n a l } } .\tag{23}
$$

By jointly supervising coarse-scale spectral envelopes through the approximation band and fine-scale absorption line shapes through the detail bands, this pretraining objective encourages the encoder to learn representations that capture both global spectral morphology and local absorption features.

## Molecular fingerprint similarity alignment.

To encourage the latent space to reflect graded structural relationships among molecules, UltraIR is additionally trained with a soft Tanimoto contrastive objective. This objective operates on a dedicated projection of the encoder output: a two-layer projection head maps z to a $d _ { p }$ -dimensional embedding $\mathbf { p } \in \mathbb { R } ^ { d _ { p } }$ via layer normalization, a linear projection, a GELU activation, dropout, and a final linear projection.

Rather than defining hard positive and negative pairs, this objective constructs a continuous structural similarity target from precomputed binary molecular fingerprints. For a batch of B samples with binarized fingerprints $\mathbf { f } _ { i } \in \{ 0 , 1 \} ^ { \bar { M } }$ the pairwise Tanimoto similarity is:

$$
T _ { i j } = { \frac { \mathbf { f } _ { i } ^ { \top } \mathbf { f } _ { j } } { \| \mathbf { f } _ { i } \| _ { 1 } + \| \mathbf { f } _ { j } \| _ { 1 } - \mathbf { f } _ { i } ^ { \top } \mathbf { f } _ { j } + \varepsilon } } .\tag{24}
$$

A soft target distribution $p _ { i j }$ over of-diagonal batch members is obtained by applying a softmax with temperature $\tau _ { t }$ to each row of T after masking the diagonal. Student similarity logits are computed as scaled inner products between $\ell _ { 2 } \cdot$ -normalized embeddings $\bar { \mathbf { p } } _ { i } = \mathbf { p } _ { i } / \lVert \mathbf { p } _ { i } \rVert _ { 2 }$ , giving $s _ { i j } = \bar { \mathbf { p } } _ { i } ^ { \top } \bar { \mathbf { p } } _ { j } / \tau _ { s }$ . The contrastive loss is the soft cross-entropy between the student distribution and the Tanimoto-derived target:

$$
{ \mathcal { L } } _ { \mathrm { c o n t r a s t } } = - { \frac { 1 } { B } } \sum _ { i } \sum _ { j \neq i } p _ { i j } \log { \frac { \exp ( s _ { i j } ) } { \sum _ { k \neq i } \exp ( s _ { i k } ) } } .\tag{25}
$$

This objective encourages spectra of structurally related compounds to occupy nearby regions of the latent space in proportion to their fingerprint overlap.

## Multi-label functional-group prediction.

A multilayer classification head maps the latent representation z to logits over the functional-group classes. It consists of layer normalization followed by three linear projections with progressively reducing dimensionality, interleaved with GELU activations and dropout. The model is trained to predict the presence or absence of each functional group as an independent binary classification task using binary cross-entropy with logits. This objective provides explicit chemical supervision that anchors the latent space to interpretable structural motifs.

## Downstream adaptation

For downstream adaptation, the pretrained checkpoint was used to initialize only the shared spectral encoder components present in both the pretraining and downstream models. These components included the derivative-aware multichannel input module, the convolutional spectral encoder, and the patch-based Transformer encoder. Pretrainingspecific heads, including the reconstruction head, contrastive projection head, and pretraining functional-group prediction head, were not transferred to downstream tasks. Consequently, even when a downstream task shared a prediction type or label space with a pretraining objective, its task-specific output module was newly initialized rather than inherited from the pretraining checkpoint. Other components introduced exclusively for downstream prediction were also initialized randomly.

## Downstream task heads

## Classification head.

All classification tasks use task-specific instances of the same MLP head architecture. The head consists of layer normalization followed by three linear projections with progressively reducing dimensionality, interleaved with GELU activations and a dropout layer. The final projection maps the hidden representation to C output logits, where C denotes the number of classes for each task. Single-spectrum classification tasks receive the global spectral representation z as input, whereas targeted component detection receives the dedicated pair representation $\mathbf { r } _ { \mathrm { p a i r } }$

For functional-group prediction, each output logit corresponds to an independent binary classification target, and the head is trained with binary cross-entropy loss. Bacterial classification, medicinal-herb geographic origin traceability, and microplastics classification are formulated as single-label multiclass classification tasks and trained with crossentropy loss, with label smoothing applied for medicinal-herb geographic origin traceability. Targeted component detection produces a scalar logit and is trained with binary cross-entropy loss.

## Regression head.

For regression tasks, including physicochemical property prediction, targeted fractional contribution estimation, mixturelevel component quantification, medicinal-herb constituent quantification, and soil property prediction, a task-specific three-layer MLP regression head processes the input representation. This head follows the same architecture as the classification head but uses task-specific output layers. Single-spectrum regression tasks receive the global spectra representation $\mathbf { z } \in \mathbb { R } ^ { d }$ as input, whereas targeted fractional contribution estimation receives the dedicated pair representation $\mathbf { r } _ { \mathrm { p a i r } }$

To stabilize training across target dimensions with heterogeneous numerical scales, regression targets were transformed using target-wise scaling. For single-target regression tasks, including targeted fractional contribution estimation, the final output is passed through a sigmoid activation and optimized using mean squared error loss. For multi-target regression tasks, including physicochemical property prediction, mixture-level component quantification, medicinalherb constituent quantification, and soil property prediction, the regression head adopts a linear output layer and is optimized using a smooth- $\mathbf { \nabla } \cdot \boldsymbol { \ell } _ { 1 }$ regression loss.

## 5 Data and code availability

The code is available at https://github.com/AIMS-Lab-HKUSTGZ/UltraIR. The checkpoints of UltraIR are available at https://huggingface.co/yusentan/UltraIR.

The newly generated UltraIR molecular-dynamics simulated infrared dataset used for UltraIR pretraining can be accessed at https://huggingface.co/yusentan/UltraIR. The simulated infrared dataset from IRtoMol [19] used for UltraIR pretraining can be accessed through https://zenodo.org/records/7928396. The simulated infrared dataset from the multimodal spectroscopy dataset [34] used for UltraIR pretraining can be accessed through https://zenodo.org/records/14770232. The simulated infrared dataset from QM9S [35] used for UltraIR pretraining can be accessed through https://figshare.com/articles/dataset/QM9S\_dataset/24235333.

The real infrared dataset from the NIST Chemistry WebBook [37] used for molecular-level interpretation and mixture analysis benchmarks can be accessed through https://webbook.nist.gov/chemistry/. The real infrared dataset from SDBS [38] used for molecular-level interpretation and mixture analysis benchmarks can be accessed through http s://sdbs.db.aist.go.jp/. The simulated infrared dataset from USPTO [36] used for molecular-level interpretation and mixture analysis benchmarks can be accessed through https://zenodo.org/records/16417648. The external liquid-mixture benchmark from DeepMIR [22] used for mixture analysis benchmarks can be accessed through https: //github.com/LinTan-CSU/DeepMIR. The experimental FTIR mixture dataset [39] used for mixture analysis benchmarks can be accessed through https://doi.org/10.5281/zenodo.5498197.

The green-snow bacterial FTIR dataset [40] used for bacterial classification can be accessed through https://zeno do.org/records/4297950. The two newly generated in-house experimental datasets of Jinyinhua and Shanyinhua used for medicinal-herb characterization can be accessed at https://huggingface.co/yusentan/UltraIR. The environmentally sourced microplastics infrared dataset [7] used for microplastics classification can be accessed through https://drive.google.com/drive/folders/11MofhjEchgZelWPcHUvIMRPNEQPaLfUO?usp=sharing. The OSSL dataset [6] used for soil property prediction can be accessed through https://docs.soilspectroscopy.org/db-acc ess.html.

## 6 Acknowledgments

This research was supported by the National Natural Science Foundation of China Project (No. 623B2086), CCF-GHFund (No. OF 2026005), the CIPS-SMP-Zhipu Large Model Fund, Ant Group, Shanghai Artificial Intelligence Laboratory, and TeleAI of China Telecom.

## 7 Author contributions

Conceptualization, Y.T. and J.X.; methodology, Y.T., Y.C., and J.X.; software, Y.T.; data curation, Y.T., Y.C., Z.F., P.L., and Y.F.L.; formal analysis, Y.T., Y.C., and Z.F.; investigation, Y.T., Y.C., Z.F., P.L., and Y.F.L.; validation, Y.T., Y.C., Z.F., P.L., and Y.F.L.; visualization, Y.T., Y.C., Q.G., and Z.L.; writing – original draft, Y.T.; writing – review & editing, all authors; resources, J.X., Y.Q.L., and T.W.; supervision, J.X., X.Z., and T.W.; project administration, J.X.; funding acquisition, J.X.

## 8 Competing interests

The authors declare no competing interests.

## References

[1] Barbara H Stuart. Infrared spectroscopy: fundamentals and applications. John Wiley & Sons, 2004.

[2] Peter R Grifiths. Fourier transform infrared spectrometry. Science, 222(4621):297–302, 1983.

[3] Sergei G Kazarian and KL Andrew Chan. ATR-FTIR spectroscopic imaging: recent advances and applications to biological systems. Analyst, 138(7):1940–1951, 2013.

[4] Sander De Bruyne, Marijn M Speeckaert, and Joris R Delanghe. Applications of mid-infrared spectroscopy in the clinical laboratory setting. Critical Reviews in Clinical Laboratory Sciences, 55(1):1–20, 2018.

[5] Marinus Huber, Kosmas V Kepesidis, Liudmila Voronina, Maša Božić, Michael Trubetskov, Nadia Harbeck, Ferenc Krausz, and Mihaela Žigman. Stability of person-specific blood-based infrared molecular fingerprints opens up prospects for health monitoring. Nature Communications, 12(1):1511, 2021.

[6] José L Safanelli, Tomislav Hengl, Leandro L Parente, Robert Minarik, Dellena E Bloom, Katherine Todd-Brown, Asa Gholizadeh, Wanderson de Sousa Mendes, and Jonathan Sanderman. Open Soil Spectral Library (OSSL): Building re producible soil calibration models through open development and community engagement. PLOS ONE, 20(1):e0296545, 2025.

[7] Junhao Xie, Aoife Gowen, and Jun-Li Xu. Open-set convolutional neural network for infrared spectral classification of environmentally sourced microplastics. npj Emerging Contaminants, 2(1):6, 2026.

[8] Tarek Eissa, Liudmila Voronina, Marinus Huber, Frank Fleischmann, and Mihaela Žigman. The perils of molecular interpretations from vibrational spectra of complex samples. Angewandte Chemie International Edition, 63(50):e202411596, 2024.

[9] Matthew J Baker, Júlio Trevisan, Paul Bassan, Rohit Bhargava, Holly J Butler, Konrad M Dorling, Peter R Fielden, Simon W Fogarty, Nigel J Fullwood, Kelly A Heys, et al. Using Fourier transform IR spectroscopy to analyze biological materials. Nature Protocols, 9(8):1771–1791, 2014.

[10] Rekha Gautam, Sandeep Vanga, Freek Ariese, and Siva Umapathy. Review of multidimensional data processing approaches for Raman and infrared spectroscopy. EPJ Techniques and Instrumentation, 2(1):8, 2015.

[11] Kas J Houthuijs, Giel Berden, Udo FH Engelke, Vasuk Gautam, David S Wishart, Ron A Wevers, Jonathan Martens, and Jos Oomens. An in silico infrared spectral library of molecular ions for metabolite identification. Analytical Chemistry, 95 (23):8998–9005, 2023.

[12] Brian C Smith. Infrared spectral interpretation: a systematic approach. CRC Press, 2018.

[13] Lisa M. Miller and John P. Coates. Interpretation of Infrared Spectra: A Practical and Systematic Approach. In Encyclopedia of Analytical Chemistry, pages 1–24. Wiley, 2025.

[14] Joshua L Lansford and Dionisios G Vlachos. Infrared spectroscopy data- and physics-driven machine learning for charac terizing surface microstructure of complex materials. Nature Communications, 11(1):1513, 2020.

[15] Zhihao Ren, Zixuan Zhang, Jingxuan Wei, Bowei Dong, and Chengkuo Lee. Wavelength-multiplexed hook nanoantennas for machine learning enabled mid-infrared spectroscopy. Nature Communications, 13(1):3859, 2022.

[16] Jianxiong Zhu, Shanling Ji, Zhihao Ren, Wenyu Wu, Zhihao Zhang, Zhonghua Ni, Lei Liu, Zhisheng Zhang, Aiguo Song, and Chengkuo Lee. Triboelectric-induced ion mobility for artificial intelligence-enhanced mid-infrared gas spectroscopy. Nature Communications, 14(1):2524, 2023.

[17] Vu Hoang Minh Doan, Cao Duong Ly, Sudip Mondal, Thi Thuy Truong, Tan Dung Nguyen, Jaeyeop Choi, Byeongil Lee, and Junghwan Oh. Fcg-Former: identification of functional groups in FTIR spectra using enhanced transformer-based model. Analytical Chemistry, 96(30):12358–12369, 2024.

[18] Dev Punjabi, Yu-Chieh Huang, Laura Holzhauer, Pierre Tremouilhac, Pascal Friederich, Nicole Jung, and Stefan Bräse. Infrared spectrum analysis of organic molecules with neural networks using standard reference data sets in combination with real-world data. Journal of Cheminformatics, 17(1):24, 2025.

[19] Marvin Alberts, Teodoro Laino, and Alain C Vaucher. Leveraging infrared spectroscopy for automated structure elucida tion. Communications Chemistry, 7(1):268, 2024.

[20] Wenjin Wu, Ales Leonardis, Jianbo Jiao, Jun Jiang, and Linjiang Chen. Transformer-based models for predicting molecular structures from infrared spectra using patch-based self-attention. The Journal of Physical Chemistry A, 129(8):2077–2085, 2025.

[21] Ganesh Chandan Kanakala, Bhuvanesh Sridharan, and U Deva Priyakumar. Spectra to structure: contrastive learning framework for library ranking and generating molecular structures for infrared spectra. Digital Discovery, 3(12):2417–2423, 2024.

[22] Lin Tan, Yue Wang, Hailiang Zhang, Jinyu Sun, Qiong Yang, Xiao Yang, Zhimin Zhang, and Hongmei Lu. DeepMIR: A Hybrid Convolutional Neural Network-Transformer Framework for Accurate Identification of Target Components from Mid-Infrared Spectra of Mixtures. Analytical Chemistry, 97(50):27706–27715, 2025.

[23] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901, 2020.

[24] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models, 2021.

[25] Yukun Zhou, Mark A Chia, Siegfried K Wagner, Murat S Ayhan, Dominic J Williamson, Robbert R Struyven, Timing Liu, Moucheng Xu, Mateo G Lozano, Peter Woodward-Court, et al. A foundation model for generalizable disease detection from retinal images. Nature, 622(7981):156–163, 2023.

[26] DongAo Ma, Jiaxuan Pang, Michael B Gotway, and Jianming Liang. A fully open AI foundation model applied to chest radiography. Nature, 643(8071):488–498, 2025.

[27] Guy Lutsker, Gal Sapir, Smadar Shilo, Jordi Merino, Anastasia Godneva, Jerry R Greenfield, Dorit Samocha-Bonet, Raja Dhir, Francisco Gude, Shie Mannor, et al. A foundation model for continuous glucose monitoring data. Nature, 650(8103): 978–986, 2026.

[28] Eric Y Wang, Paul G Fahey, Zhuokun Ding, Stelios Papadopoulos, Kayla Ponder, Marissa A Weis, Andersen Chang, Taliah Muhammad, Saumil Patel, Zhiwei Ding, et al. Foundation model of neural activity predicts response to new stimulus types. Nature, 640(8058):470–477, 2025.

[29] Kang Wu, Yingying Zhang, Lixiang Ru, Bo Dang, Jiangwei Lao, Lei Yu, Junwei Luo, Zifan Zhu, Yue Sun, Jiahao Zhang, et al. A semantic-enhanced multi-modal remote sensing foundation model for Earth observation. Nature Machine Intelligence, 7(8):1235–1249, 2025.

[30] Shoujie Li, Tong Wu, Jianle Xu, Yan Huang, Zongwen Zhang, Hongfa Zhao, Qinghao Xu, Zihan Wang, Linqi Ye, Yang Yang, et al. Biomimetic multimodal tactile sensing enables human-like robotic perception. Nature Sensors, 1(1):52–62, 2026.

[31] Hao Sun, Xin Gao, Wanhao Niu, Longwei Li, Yutong Song, Lei Liu, Huaidong Zhou, Huaping Liu, Zhong Lin Wang, Xiong Pu, et al. A spike–language dual framework bridges fast perception and deep reasoning in artificial tactile somatosensory systems. Nature Sensors, pages 1–12, 2026.

[32] Marvin Alberts, Federico Zipoli, and Teodoro Laino. Setting new benchmarks in AI-driven infrared structure elucidation. Digital Discovery, 4(7):1936–1943, 2025.

[33] Charles McGill, Michael Forsuelo, Yanfei Guan, and William H Green. Predicting infrared spectra with message passing neural networks. Journal of Chemical Information and Modeling, 61(6):2594–2609, 2021.

[34] Marvin Alberts, Oliver Schilter, Federico Zipoli, Nina Hartrampf, and Teodoro Laino. Unraveling molecular structure: A multimodal spectroscopic dataset for chemistry. In Advances in Neural Information Processing Systems, volume 37, pages 125780–125808, 2024.

[35] Zihan Zou, Yujin Zhang, Lijun Liang, Mingzhi Wei, Jiancai Leng, Jun Jiang, Yi Luo, and Wei Hu. A deep learning model for predicting selected organic molecular spectra. Nature Computational Science, 3(11):957–964, 2023.

[36] Federico Zipoli, Marvin Alberts, and Teodoro Laino. IR-NMR multimodal computational spectra dataset for 177K patent extracted organic molecules. Scientific Data, 12(1):1375, 2025.

[37] Peter J Linstrom and William G Mallard. The NIST Chemistry WebBook: A chemical data resource on the internet. Journal of Chemical & Engineering Data, 46(5):1059–1063, 2001.

[38] National Institute of Advanced Industrial Science and Technology. SDBS: Spectral Database for Organic Compounds, 2026. URL https://sdbs.db.aist.go.jp/.

[39] Andrea Angulo, Lankun Yang, Eray S Aydil, and Miguel A Modestino. Machine learning enhanced spectroscopic analysis: towards autonomous chemical mixture characterization for rapid process optimization. Digital Discovery, 1(1):35–44, 2022.

[40] Margarita Smirnova, Achim Kohler, and Volha Shapaval. FTIR Dataset, 2020. URL https://doi.org/10.5281/zenodo .4297950.

[41] Tianqi Chen and Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 785–794, 2016.

[42] Leo Breiman. Random forests. Machine Learning, 45(1):5–32, 2001.

[43] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.

[44] Colin Zhang and Yang Ha. Toward Complete Molecular Structure Prediction from Infrared Spectroscopy Using Deep Learning. Journal of Chemical Information and Modeling, 66(1):100–109, 2026.

[45] Svante Wold, Michael Sjöström, and Lennart Eriksson. PLS-regression: a basic tool of chemometrics. Chemometrics and Intelligent Laboratory Systems, 58(2):109–130, 2001.

[46] Jingkai Zhou, Zixuan Zhang, Bowei Dong, Zhihao Ren, Weixin Liu, and Chengkuo Lee. Midinfrared spectroscopic analysis of aqueous mixtures using artificial-intelligence-enhanced metamaterial waveguide sensing platform. ACS Nano, 17(1): 711–724, 2022.

[47] Min He, Jingjing Tong, Xiangxian Li, Xin Han, Yusheng Qin, Renjie Fang, Zidong Chen, and Minguang Gao. Research on hybrid microplastic recognition method based on dual-branch convolutional neural network combined with attention mechanism. Microchemical Journal, 218:115131, 2025.

[48] Wenqi Guo, Shichen Gao, Yaohui Ding, and Daming Dong. Simultaneous prediction of multiple soil components using Mid Infrared Spectroscopy and the GADF-Swin Transformer model. Computers and Electronics in Agriculture, 237:110507, 2025.

[49] Yiqiang Liu, Luming Shen, Xinghui Zhu, Yangfan Xie, and Shaofang He. Spectral data-driven prediction of soil properties using LSTM-CNN-attention model. Applied Sciences, 14(24):11687, 2024.

[50] Yongqi Jin, Jun-Jie Wang, Fanjie Xu, Xiaohong Ji, Zhifeng Gao, Linfeng Zhang, Guolin Ke, Rong Zhu, and Weinan E. NMR-Solver: automated structure elucidation via large-scale spectral matching and physics-guided fragment optimization. Nature Communications, 17(1):4740, 2026.

[51] Peter Eastman, Jason Swails, John D Chodera, Robert T McGibbon, Yutong Zhao, Kyle A Beauchamp, Lee-Ping Wang, Andrew C Simmonett, Matthew P Harrigan, Chaya D Stern, et al. OpenMM 7: Rapid development of high performance algorithms for molecular dynamics. PLOS Computational Biology, 13(7):e1005659, 2017.

[52] Simon Boothroyd, Pavan Kumar Behara, Owen C. Madin, David F. Hahn, Hyesu Jang, Vytautas Gapsys, Jefrey R. Wagner, Joshua T. Horton, David L. Dotson, Matthew W. Thompson, Jessica Maat, Trevor Gokey, Lee-Ping Wang, Daniel J. Cole, Michael K. Gilson, John D. Chodera, Christopher I. Bayly, Michael R. Shirts, and David L. Mobley. Development and Benchmarking of Open Force Field 2.0.0: The Sage Small Molecule Force Field. Journal of Chemica Theory and Computation, 19(11):3251–3275, 2023.

## Appendix

## A Supplementary experimental details

## Pretraining hyperparameters

The principal hyperparameters used for large-scale UltraIR pretraining are summarized in Table A.

Table A | Pre-training hyperparameters of UltraIR.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimization Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Batch size</td><td>128</td></tr><tr><td>Epochs Warmup ratio</td><td>5 5%</td></tr><tr><td>Transformer encoder Hidden dimension d</td><td></td></tr><tr><td> $P$ </td><td>1024</td></tr><tr><td>Patch length Number of attention heads H</td><td>16 16</td></tr><tr><td>Number of Transformer layers  $N _ { \mathrm { l a y e r s } }$ </td><td></td></tr><tr><td>Dropout</td><td>8 0.05</td></tr><tr><td>Pre-training heads</td><td></td></tr><tr><td>Fingerprint embedding dimension  $d _ { p }$ </td><td></td></tr><tr><td>Number of wavelet decomposition levels J</td><td>512</td></tr><tr><td></td><td>4</td></tr><tr><td>Wavelet reconstruction bottleneck dimension  $d _ { b }$ </td><td>256</td></tr><tr><td>Pre-training objectives</td><td></td></tr><tr><td>Wavelet-domain spectral reconstruction loss weight  $\lambda _ { \mathrm { { r e c o n } } }$ </td><td></td></tr><tr><td>Molecular fingerprint similarity alignment loss weight λcontrast</td><td>50.0</td></tr><tr><td></td><td>0.2</td></tr><tr><td>Multi-label functional-group prediction loss weight  $\lambda _ { \mathrm { f g } }$ </td><td>1.25</td></tr><tr><td>Wavelet-coefficient-domain loss weight  $\lambda _ { \mathrm { c o e f f } }$ </td><td>0.2</td></tr><tr><td>Signal-domain loss weight  $\lambda _ { \mathrm { s i g n a l } }$ </td><td>0.8</td></tr><tr><td>Masked-position weighting factor  $m _ { w }$ </td><td>3.0</td></tr><tr><td>Fingerprint student temperature  $\tau _ { s }$ </td><td>0.1</td></tr><tr><td>Fingerprint target temperature  $\tau _ { t }$ </td><td>0.05</td></tr></table>

## Pretraining dataset construction

The UltraIR pretraining dataset comprised 60,000,000 simulated infrared (IR) spectra assembled from three complementary sources: publicly released simulated spectra, spectra generated using molecular dynamics, and spectra predicted using a machine-learning model. After removing molecules overlapping with the National Institute of Standards and Technology (NIST) Chemistry WebBook, Spectral Database for Organic Compounds (SDBS), and USPTO datasets, the publicly released sources contributed 630,395 spectra from IRtoMol, 743,955 from the multimodal spectroscopy dataset, and 127,898 from QM9S, yielding 1,502,248 spectra in total [19, 34, 35]. From structures in the PubChem-derived molecular collection distributed with the SimNMR-PubChem database [50] that were not represented in NIST, SDBS, or USPTO, we generated 7,534,293 spectra through molecular-dynamics simulation and predicted the remaining 50,963,459 spectra were predicted using Chemprop-IR [33].

For the molecular-dynamics component, three-dimensional molecular conformers were generated from Simplified Molecular Input Line Entry System (SMILES) representations and subjected to geometry relaxation before simulation. Each molecule was simulated without periodic boundary conditions at 300 K using a Langevin middle integrator in OpenMM, with the Open Force Field Sage 2.2.1 and fixed atom-centered partial charges [51, 52]. Molecular dipole trajectories were calculated from the fixed partial charges and instantaneous atomic positions. After tempora mean removal and Hann windowing, the three Cartesian dipole components were transformed independently using a fast Fourier transform. The resulting power spectra were summed, weighted by wavenumber, interpolated onto the common wavenumber grid, and normalized to a maximum intensity of one.

Table B | Functional-group classes and SMARTS definitions used for pretraining and downstream functionalgroup prediction. The listed order corresponds to the 17 dimensions of the multi-label target used in both stages. A class was assigned when the corresponding SMARTS pattern matched the molecular graph, and each molecule could receive multiple labels.
<table><tr><td>Index</td><td>Functional-group class</td><td>SMARTS definition</td></tr><tr><td>1</td><td>Alkane</td><td>[CX4;H3,H2]</td></tr><tr><td>2</td><td>Methyl</td><td>[CH3]</td></tr><tr><td>3</td><td>Alkene</td><td>[CX3]=[CX3]</td></tr><tr><td>4</td><td>Alkyne</td><td>[CX2]#[CX2]</td></tr><tr><td>5</td><td>Alcohols</td><td>[#6] [0X2H]</td></tr><tr><td>6</td><td>Amines</td><td>[NX3;H2,H1,HO;!$(N[CX3](=0))]</td></tr><tr><td>7</td><td>Nitriles</td><td>[NX1]#[CX2]</td></tr><tr><td>8</td><td>Aromatics</td><td>[$([cX3](:*):*),$([cX2+](:*):*)]</td></tr><tr><td>9</td><td>Alkyl halides</td><td>[#6] [F,Cl,Br,I]</td></tr><tr><td>10</td><td>Esters</td><td>[#6] [CX3](=0)[0X2H0][#6]</td></tr><tr><td>11</td><td>Ketones</td><td>[#6][CX3](=0)[#6]</td></tr><tr><td>12</td><td>Aldehydes</td><td>[CX3H1](=0)[#6]</td></tr><tr><td>13</td><td>Carboxylic acids</td><td>[CX3](=0)[0X2H1]</td></tr><tr><td>14</td><td>Ether</td><td>[0D2]([#6;!$(C=0)])([#6;!$(C=0)])</td></tr><tr><td>15</td><td>Acyl halides</td><td>[CX3](=[0X1])[F,C1,Br,I]</td></tr><tr><td>16</td><td>Amides</td><td>[NX3][CX3](=[0X1])[#6]</td></tr><tr><td>17</td><td>Nitro</td><td>[$([N+](=0)[0-]),$([NX3](=0)=0)][#6]</td></tr></table>

For the machine-learning-generated component, IR spectra were predicted using the two oficially released pretrained Chemprop-IR checkpoints [33]. The two models were applied independently, and their predicted intensities were combined using an equally weighted arithmetic mean to obtain the final spectrum for each molecular structure.

Functional-group labels and molecular fingerprints used during pretraining were generated from molecular structures using RDKit. Functional-group labels were encoded as 17-dimensional multi-hot vectors, with each class assigned when the molecular graph matched its corresponding SMILES Arbitrary Target Specification (SMARTS) definition. Multiple classes could therefore be assigned to the same molecule. The functional-group classes, output order, and operational SMARTS definitions are provided in Table B. For molecular fingerprint similarity alignment, each structure was represented by a 2,048-bit Morgan fingerprint generated with a radius of 2.

## Downstream datasets and benchmark summary

The eight downstream tasks were evaluated using ten datasets, with the Jinyinhua and Shanyinhua datasets each contributing separate subsets for geographic origin traceability and chemical constituent quantification. For the molecular-level benchmarks, we used NIST, SDBS, and USPTO. The NIST Chemistry WebBook, maintained as NIST Standard Reference Database 69, is a public resource that compiles chemical and spectroscopic data, including IR spectra [37]. We retained 23,286 NIST IR spectra after filtering. SDBS is a public database maintained by the National Institute of Advanced Industrial Science and Technology and contains experimentally measured Fouriertransform infrared (FTIR) spectra with corresponding molecular records for organic compounds [38]. We retained 18,253 SDBS IR spectra after filtering. The USPTO dataset is a computational multimodal spectroscopy resource constructed from patent-extracted organic molecules and contains 177,461 IR spectra generated using long-timescale molecular-dynamics simulations and machine-learning-assisted dipole-moment prediction [36].

For functional-group prediction, labels for NIST, SDBS, and USPTO were generated using the same 17 classes, class order, and SMARTS definitions as in pretraining (Table B). For physicochemical property prediction, all 11 targets were calculated from molecular structures using RDKit. The synthetic accessibility score (SAScore) was obtained using the RDKit SA\_Score implementation, and the remaining targets were calculated using their corresponding RDKit descriptor functions.

Mixture-level and application-facing evaluations used several additional data sources. For external evaluation of targeted component detection, we used 360 spectra from the external liquid-mixture benchmark described in DeepMIR, comprising binary, ternary and quaternary mixtures [22]. Mixture-level component quantification was evaluated using 74 experimentally acquired spectra from a four-component FTIR mixture dataset comprising acrylonitrile, adiponitrile, propionitrile and glycerol [39]. For this task, models were first trained on a simulated four-component mixture dataset containing 2,400 synthetic mixtures, and subsequently fine-tuned on the experimentally acquired spectra for evaluation [39]. Bacterial genus classification used 795 FTIR spectra from 45 fast-growing bacterial isolates collected from Antarctic green snow and spanning nine genera [40]. For medicinal-herb geographic origin traceability and constituent quantification, the dataset composition, LC-MS-derived relative-abundance regression targets, together with the corresponding analytical procedures, are detailed in the following subsection. The microplastics benchmark contained 32,965 IR spectra assembled from laboratory and open-source collections to represent diverse environmentally relevant polymer classes [7]. After filtering for complete wavenumber coverage and availability of the labels required for soil property prediction, we retained 46,099 mid-IR spectra from the Kellogg Soil Survey Laboratory (KSSL) collection within the Open Soil Spectral Library (OSSL) [6]. For cross-instrument and cross-laboratory evaluation, we independently filtered the datasets using the labels required for the cross-laboratory tasks and enforced complete wavenumber coverage. This procedure retained 55,974 KSSL spectra for model training and 3,753 ICRAF–ISRIC spectra for zero-shot target-domain inference. Table C summarizes the datasets and comparison methods used for each downstream task.

Table C | Summary of downstream benchmarks and comparison methods. The table summarizes the datasets and competing methods evaluated for each downstream task.
<table><tr><td>Task</td><td>Dataset</td><td>Compared methods</td></tr><tr><td>Functional-group prediction</td><td>NIST [37], SDBS [38], USPTO [36]</td><td>FCGFormer [17], IRAnalysis [18], XGBoost, Random Forest, KNN, Logistic classifier</td></tr><tr><td>Molecular structure elucidation</td><td>NIST [37], SDBS [38], USPTO [36]</td><td>IRtoMol [19], AISE [32], PBSA [20], DLIR [44]</td></tr><tr><td>Physicochemical property prediction</td><td>NIST [37], SDBS [38], USPTO [36]</td><td>XGBoost regression, SVR, KNN regression, PLSR</td></tr><tr><td>Targeted component detection</td><td>NIST [37], SDBS [38], USPTO [36] , DeepMIR liquid-mixture dataset [22]</td><td>DeepMIR [22], reverse match, HQI</td></tr><tr><td>Targeted fractional contribution estimation NIST [37], SDBS [38], USPTO [36]</td><td></td><td>DeepMIR [22], XGBoost regression, SVR, KNN regression, PLSR</td></tr><tr><td>Mixture-level component quantification</td><td>Experimental FTIR mixture dataset [39]</td><td>AIMWSP [46], ML-FTIR [39], XGBoost regression, SVR, KNN regression, PLSR</td></tr><tr><td>Bacterial genus classification</td><td>Green-snow bacterial FTIR spectra [40]</td><td>XGBoost, Random Forest, KNN, Logistic classifier</td></tr><tr><td>Medicinal-herb origin traceability</td><td>Jinyinhua, Shanyinhua</td><td>XGBoost, Random Forest, KNN, Logistic classifier</td></tr><tr><td>Medicinal-herb constituent quantification</td><td>Jinyinhua, Shanyinhua</td><td>XGBoost regression, SVR, KNN regression, PLSR</td></tr><tr><td>Microplastics classification</td><td>Environmentally sourced microplastics IR dataset [7]</td><td>Softmax [7], DB-CNN-CBAM [47], XGBoost, Random Forest, KNN, Logistic classifier</td></tr><tr><td>Soil property prediction</td><td>OSSL [6]</td><td>GADF-Swin [48], LSTM-CNN [49], XGBoost regression, SVR, KNN regression, PLSR</td></tr></table>

## In-house medicinal-herb datasets and LC-MS-derived regression targets

In addition to the public and external benchmarks summarized above, we assembled two in-house medicinal-herb datasets: Jinyinhua (Lonicerae Japonicae Flos) and Shanyinhua (Lonicerae Flos). For geographic origin traceability, the Jinyinhua dataset contains 120 IR spectra from Shandong, Henan, Hebei, and Sichuan, and the Shanyinhua dataset contains 150 IR spectra from Hunan, Hubei, Sichuan, Henan, and Guangdong; each origin is represented by 30 spectra. For chemical constituent quantification, LC-MS-derived target relative abundances measured for 60 Jinyinhua and 75 Shanyinhua samples served as regression labels and were expressed in arbitrary units (a.u.). The Jinyinhua targets were 4-methylcoumarin, milrinone, geniposide, phenylacetaldehyde, cis-cinnamic acid, and lumazine; the Shanyinhua targets were coniferyl aldehyde, quercetin, kaempferol, and luteolin.

For LC-MS analysis, 30 mg of powdered material was extracted with 1.5 mL of 70% aqueous methanol. The extract was vortexed, ultrasonicated at $3 0 ~ ^ { \circ } \mathrm { C }$ for 30 min, centrifuged at 12,000 r.p.m. for 15 min, and passed through a 0.22 µm membrane filter. Chromatographic separation was performed on a Kinetex F5 column (2.6 µm, 100 mm × 2.1 mm; Phenomenex) at $4 0 ~ ^ { \circ } \mathrm { C }$ . The mobile phases were 0.1% formic acid in water (A) and 0.1% formic acid in acetonitrile (B), with a flow rate of 0.2 mL $\operatorname* { m i n } ^ { - 1 }$ and an injection volume of 4 $\mu \mathrm { L }$ . The percentage of A was programmed as 100% at 0 min, 99% at 2 min, 95% at 3 min, 90% at 6 min, 85% at 14 min, 82% at 15 min, 80% at 18 min, 70% at 20 min, 60% at 23 min, 22% at 31 min, 10% at 33 min, 0% at 36–44 min, and 100% at 44.1–50 min. Full-scan time-of-flight mass spectrometry (TOF-MS) and information-dependent acquisition (IDA) tandem mass spectrometry (MS/MS) spectra were acquired in positive-ion mode over $m / z ~ 1 0 0 { - } 1 , 0 0 0$ and m/z 50–1,000, respectively. The source temperature was $5 0 0 ~ ^ { \circ } \mathrm { C } ,$ the ion-spray voltage was +5,500 V, ion-source gases 1 and 2 were each 50 psi, curtain gas was 30 psi, declustering potential was +80 V, and collision energy was 35 V.

Table D | Dataset metadata for cross-instrument and cross-laboratory soil-property prediction. Spectra from the Kellogg Soil Survey Laboratory (KSSL) and ICRAF–ISRIC, together with their associated metadata, were obtained from the Open Soil Spectral Library (OSSL) [6].
<table><tr><td>Characteristic</td><td>Kellogg Soil Survey Laboratory database Source training domain</td><td>ICRAF-ISRIC Soil Spectral Library Target test domain</td></tr><tr><td>Institution</td><td>USDA National Soil Survey Center</td><td>World Agroforestry Centre / ISRIC</td></tr><tr><td>Sample origin</td><td>United States</td><td>Not recorded in OSSL</td></tr><tr><td>FTIR system</td><td>Bruker Vertex 70 with HTS-XT</td><td>Bruker Tensor 27 with HTS-XT</td></tr><tr><td>Sample preparation</td><td>Ground to &lt; 80 mesh</td><td>Ground to &lt; 0.1 mm</td></tr></table>

Table E | Label distributions for cross-instrument and cross-laboratory soil-property prediction. The Kellogg Soil Survey Laboratory (KSSL) and ICRAF–ISRIC spectra and associated soil-property labels were obtained from the Open Soil Spectral Library (OSSL) [6]. Values are reported as median [minimum, maximum].
<table><tr><td>Property</td><td>Unit</td><td>Kellogg Soil Survey Laboratory database Source training domain</td><td>ICRAF-ISRIC Soil Spectral Library Target test domain</td></tr><tr><td>Clay Content</td><td>%</td><td>20.88 [0, 96.14]</td><td>30.10 [0, 96.80]</td></tr><tr><td>Silt Content</td><td>%</td><td>39.10 [0, 94.50]</td><td>24.90 [0.20, 100.00]</td></tr><tr><td>Sand Content</td><td>%</td><td>32.60 [0.10, 100.00]</td><td>33.05 [0, 99.50]</td></tr><tr><td>pH (H₂O)</td><td></td><td>6.37 [2.29, 10.70]</td><td>5.80 [3.00, 10.50]</td></tr><tr><td>CEC</td><td> $\mathrm { c m o l _ { c } \ k g ^ { - 1 } }$ </td><td>15.62 [0, 584.59]</td><td>11.50 [0, 189.60]</td></tr><tr><td>Exchangeable Ca</td><td>cmolc kg−¹</td><td>11.92 [0, 410.41]</td><td>3.80 [0, 168.20]</td></tr><tr><td>Exchangeable Mg</td><td>cmolc kg−¹</td><td>2.86 [0, 172.64]</td><td>1.10 [0, 68.00]</td></tr><tr><td>Exchangeable K</td><td>cmolc kg−¹</td><td>0.364 [0, 32.33]</td><td>0.20 [0, 9.80]</td></tr><tr><td>Exchangeable Na</td><td> $\mathrm { c m o l _ { c } \ k g ^ { - 1 } }$ </td><td>0 [0, 868.36]</td><td>0.10 [0, 31.60]</td></tr></table>

## Dataset metadata and label distributions for cross-instrument and cross-laboratory soil-property prediction

For cross-instrument and cross-laboratory soil-property prediction, the source- and target-domain dataset metadata and corresponding label distributions across the nine soil properties are summarized in Tables D and E, respectively.

## Cross-validation and evaluation protocol

All downstream evaluations requiring supervised model training were conducted using five-fold cross-validation. For each such dataset, samples were partitioned into five non-overlapping folds, with each fold used once as the test set. In each cross-validation run, one fold, corresponding to approximately 20% of the dataset, was held out for testing. The remaining samples were divided into training and validation subsets to approximate an overall 70:10:20 train–validation–test ratio. When this ratio could not be achieved exactly because of discrete sample counts, the partition with the smallest deviation from the target ratio was used. For the molecule-level NIST, SDBS, and USPTO benchmarks, scafold-based partitioning was used to minimize structural overlap among the training, validation, and test sets. Identical partitions were used for UltraIR, the matched no-pretraining ablation where included, and al comparison methods, ensuring directly comparable evaluations.

For the reduced-training-data experiments, the validation and test sets remained fixed within each cross-validation run. Only the original training subset was randomly subsampled to the specified fractions of labeled training data. These fractions therefore refer to the available training subset rather than the complete dataset. At each fraction, identical subsampled training sets were used for all compared methods.

Model configurations, training duration, early-stopping criteria, checkpoint selection, and the optional use of downstream spectral augmentation were determined exclusively using the training and validation subsets within each cross-validation run. Weak physically motivated augmentations, when enabled, were applied only to training samples. Except for the DeepMIR liquid-mixture dataset, downstream performance was aggregated across the five held-out test folds. For DeepMIR, checkpoints trained on the corresponding NIST mixture-analysis tasks with matched target labels were applied directly for inference. No additional model training or five-fold cross-validation was performed on the DeepMIR dataset.

## Evaluation metrics

## Statistical significance testing.

All significance annotations shown in the figures were calculated independently for each dataset–metric combination. Methods were ranked according to their mean performance across the five held-out test folds, using the appropriate direction for each metric, and only the top-ranked and second-ranked methods were compared. Their fold-leve results were paired by the corresponding test fold and analyzed using a two-sided paired t-test implemented with scipy.stats.ttest\_rel. The null hypothesis was that the mean paired diference between the two methods across the five folds was zero. Significance was annotated as \*\*\*\* for $P \mathrm { ~ < ~ } 0 . 0 0 0 1$ \*\*\* for $0 . 0 0 0 1 \leq P < 0 . 0 0 1$ \*\* for $0 . 0 0 1 \leq P < 0 . 0 1 ,$ , \* for $0 . 0 1 \leq P < 0 . 0 5$ , and n.s. for $P \ge 0 . 0 5$

## Classification metrics.

Single-label classification tasks, including bacterial classification, medicinal-herb geographic origin traceability, and microplastics classification, were evaluated using accuracy, Macro-F1, and the multiclass Matthews correlation coeficient (MCC). Targeted component detection was evaluated using accuracy, Macro-F1, and the area under the receiver operating characteristic curve (ROC-AUC), calculated from the continuous component-presence scores before thresholding. Macro-F1 was calculated as the unweighted mean of the one-versus-rest F1 scores across classes, thereby assigning equal weight to each class. For the class-wise radar analyses of bacterial and microplastics classification, the value reported for each bacterial genus or polymer class was its corresponding one-versus-rest F1 score. These classspecific F1 scores were calculated independently for each class and were distinct from the Macro-F1 values reported for overall task performance.

Functional-group prediction was formulated as multi-label classification and evaluated using Micro-F1, Macro-F1, and exact match ratio (EMR). Micro-F1 was calculated by aggregating true positives, false positives, and false negatives across all samples and functional-group labels, whereas Macro-F1 was the unweighted mean of the label-specific F1 scores. EMR required the complete predicted functional-group vector to match the ground-truth vector:

$$
\mathrm { E M R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \left( \mathbf { y } _ { i } = \hat { \mathbf { y } } _ { i } \right) ,\tag{26}
$$

where N is the number of evaluated spectra, $\mathbf { y } _ { i }$ and ${ \hat { \mathbf { y } } } _ { i }$ are the ground-truth and predicted binary label vectors, respectively, and I(·) is the indicator function. A prediction containing either a missed functional group or an additiona false-positive group was therefore counted as incorrect.

## Molecular structure elucidation metrics.

Formula-conditioned molecular structure elucidation was evaluated using top-k accuracy for $k \in \{ 1 , 5 , 1 0 \}$ . Let $s _ { i }$ denote the ground-truth SMILES string for sample i, and let $\mathcal { P } _ { i } ^ { ( k ) }$ denote the first k generated candidate SMILES strings in their original ranked order. Before structure matching, the ground-truth SMILES and every generated candidate were parsed using RDKit and converted to canonical isomeric SMILES. Denoting this canonicalization operation by C(·), top-k accuracy was defined as

$$
\mathrm { T o p - } k = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \left[ \exists \hat { s } \in \mathcal { P } _ { i } ^ { ( k ) } \mathrm { ~ s u c h ~ t h a t ~ } \mathcal { C } ( \hat { s } ) = \mathcal { C } ( s _ { i } ) \right] .\tag{27}
$$

A sample was counted as correct when at least one of the first k generated candidates represented the same canonical molecular structure as the ground truth. Candidates that could not be parsed into valid molecular structures remained in their generated rank positions but could not contribute a correct match. Stereochemical annotations were retained during canonicalization where present in the corresponding SMILES strings.

Structural similarity between a generated candidate and the ground-truth molecule was assessed using the Tanimoto similarity of their binary molecular fingerprints. For fingerprint vectors f and g, the Tanimoto similarity was defined as

$$
T ( \mathbf { f } , \mathbf { g } ) = { \frac { \mathbf { f } ^ { \mathsf { T } } \mathbf { g } } { \| \mathbf { f } \| _ { 1 } + \| \mathbf { g } \| _ { 1 } - \mathbf { f } ^ { \mathsf { T } } \mathbf { g } } } .\tag{28}
$$

The score ranges from 0 to 1, with larger values indicating greater fingerprint overlap.

## Regression metrics.

Regression tasks were evaluated using mean absolute error (MAE), root mean squared error (RMSE), and the coeficient of determination $R ^ { 2 }$ . For single-target regression tasks, including targeted fractional contribution estimation, MAE, RMSE, and $R ^ { 2 }$ were calculated directly in the original target space.

For multi-target regression tasks, including physicochemical property prediction, mixture-level component quantification, medicinal-herb constituent quantification, and soil property prediction, normalized MAE and normalized RMSE were used to account for diferences in numerical ranges among targets. These metrics were calculated after target-wise normalization and averaged across targets such that each target contributed equally. Overall $R ^ { 2 }$ was calculated as the arithmetic mean of target-specific $R ^ { 2 }$ values across targets.

For normalized residual analyses, residuals were calculated independently for each target and normalized by the corresponding target range. Let $y _ { i j }$ and $\hat { y } _ { i j }$ denote the ground-truth and predicted values, respectively, for target j of sample i. The normalized residual was defined as

$$
r _ { i j } ^ { \prime } = \frac { \hat { y } _ { i j } - y _ { i j } } { y _ { j , \mathrm { { m a x } } } - y _ { j , \mathrm { { m i n } } } } ,\tag{29}
$$

where $y _ { j , \mathrm { m a x } }$ and $y _ { j , \mathrm { m i n } }$ denote the maximum and minimum values of target $j ,$ respectively. To visualize multi-target regression residuals, the normalized residuals from all samples and targets were pooled such that each sample–target pair contributed equally to the resulting distribution.

## BertzCT complexity-thresholded analysis.

Prediction errors across molecular complexity were assessed using cumulative BertzCT thresholds. For threshold $\tau ,$ the evaluated subset was defined as

$$
{ \cal { S } } _ { \tau } = \{ i \colon B _ { i } \geq \tau \} ,\tag{30}
$$

where $B _ { i }$ is the ground-truth BertzCT value of molecule i. The mean relative error (MRE) was then calculated as

$$
\mathrm { M R E } ( \tau ) = \frac { 1 } { | S _ { \tau } | } \sum _ { i \in \mathcal { S } _ { \tau } } \frac { \Big | \hat { B } _ { i } - B _ { i } \Big | } { | B _ { i } | + \epsilon } ,\tag{31}
$$

where ϵ is a small constant.

## True-class margin analysis.

For microplastics classification, prediction confidence was further examined using the true-class margin. Given the logit $s _ { i c }$ assigned by a model to class c for sample i, with $y _ { i }$ denoting the ground-truth class label of sample i, the margin was defined as

$$
m _ { i } = s _ { i y _ { i } } - \operatorname* { m a x } _ { c \neq y _ { i } } s _ { i c } .\tag{32}
$$

A positive margin indicates that the ground-truth class received the highest logit, whereas a negative margin indicate that at least one competing class received a higher score. Larger positive values indicate greater separation between the true class and its closest competing class.

## B More results

## Extended ablation analysis of UltraIR

Table F | Extended ablation analysis of simulated pretraining across representative downstream tasks. UltraIR is compared with the matched no-pretraining ablation across four downstream tasks. Values are means ± standard deviations across five test folds. Bold values indicate the better result for each metric. Arrows indicate the preferred direction.
<table><tr><td colspan="4">a, Functional-group prediction Results on the NIST dataset</td></tr><tr><td>Model</td><td> ${ \mathrm { M a c r o } } { \mathrm { - F } } 1 \uparrow$ </td><td> $_ { \mathrm { M i c r o - F 1 } } \uparrow$ </td><td>EMR ↑</td></tr><tr><td>UltraIR</td><td> $\mathbf { 0 . 9 1 9 { \scriptstyle \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 0 . 9 4 3 _ { \pm 0 . 0 0 2 } }$ </td><td> $\mathbf { 0 . 7 7 2 _ { \pm 0 . 0 0 7 } }$ </td></tr><tr><td> $\mathrm { w / o }$  pretraining</td><td> $0 . 9 0 1 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 9 3 2 _ { \pm 0 . 0 0 2 }$ </td><td> $0 . 7 1 9 { \scriptstyle \pm 0 . 0 0 9 }$ </td></tr><tr><td colspan="4">b, Molecular structure elucidation Results on the NIST dataset</td></tr><tr><td>Model</td><td>Top-1 accuracy ↑</td><td>Top-5 accuracy ↑</td><td>Top-10 accuracy ↑</td></tr><tr><td>UltraIR</td><td> $\mathbf { 0 . 5 2 2 _ { \pm 0 . 0 0 5 } }$ </td><td> $\mathbf { 0 . 5 7 5 { \scriptstyle \pm 0 . 0 0 7 } }$ </td><td> $\mathbf { 0 . 5 7 6 _ { \pm 0 . 0 0 7 } }$ </td></tr><tr><td> $\mathrm { w / o }$  pretraining</td><td> $0 . 4 7 7 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 5 6 3 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 5 6 7 { \scriptstyle \pm 0 . 0 0 7 }$ </td></tr><tr><td colspan="4">c, Mixture-level component quantification</td></tr><tr><td>Model</td><td>Normalized MAE (×10−3) ↓</td><td>Normalized RMSE (×10−3) ↓</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td>UltraIR</td><td> $\mathbf { 2 . 0 3 6 _ { \pm 0 . 3 6 6 } }$ </td><td> $\mathbf { 2 . 6 2 1 { \scriptstyle \pm 0 . 5 0 2 } }$ </td><td> $\mathbf { 0 . 7 8 2 _ { \pm 0 . 0 4 7 } }$ </td></tr><tr><td> $\mathrm { w / o }$  pretraining</td><td> $2 . 2 8 8 { \scriptstyle \pm 0 . 2 2 5 }$ </td><td> $2 . 9 1 9 { \scriptstyle \pm 0 . 3 7 0 }$ </td><td> $0 . 7 1 4 { \scriptstyle \pm 0 . 0 2 9 }$ </td></tr><tr><td colspan="4">d, Soil property prediction</td></tr><tr><td>Model</td><td>Normalized MAE↓</td><td>Normalized RMSE ↓</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td>UltraIR</td><td> $\mathbf { 0 . 0 8 6 _ { \pm 0 . 0 0 2 } }$ </td><td> $\mathbf { 0 . 3 2 5 _ { \pm 0 . 1 4 7 } }$ </td><td> $\mathbf { 0 . 9 2 1 { \scriptstyle \pm 0 . 0 2 6 } }$ </td></tr><tr><td>w/o pretraining</td><td> $0 . 1 0 4 _ { \pm 0 . 0 0 2 }$ </td><td> $0 . 3 6 0 { \scriptstyle \pm 0 . 1 3 8 }$ </td><td> $0 . 9 0 1 { \scriptstyle \pm 0 . 0 2 5 }$ </td></tr></table>

## Additional functional-group prediction results

a  
![](images/2d5076f48c84ee917ac7c3f6072b28b92f4a04945c04a233bb3f64359b37df64.jpg)

![](images/542507d3bf8ed3638a0c8ba75fc8dace64ec8392cabae9f90430aabdc5d143b8.jpg)

![](images/50234478ad0b97713caafa10e8f52e8c69099306869cd9a6d56c53ffcfd09dc5.jpg)  
UltraIR FCGFormer IRAnalysisXGBoost RF KNN Logistic

b  
![](images/c7aa0e5e5716f5d91b951926a3635759ac30e368939c61c42fc86d839751875b.jpg)

![](images/00ca0496023d1f8751370eab2191b297a30251dfe4ebfce414494058ad49ad32.jpg)

![](images/59ad5d45aa2c2b351177c04c84a9ab98809c1536f4ffebb15b0b4542cfc1b406.jpg)

c  
![](images/5ffc85400b6bbaff21f587274130963e0b573e0cfe660b9e6f326b455697e7a8.jpg)  
UltralRFCGFormer IRAnalysis XGBoost RF KNN Logistic

Figure A | Functional-group prediction across molecular-complexity strata and precision–recall trade-ofs. a, Exact match ratio (EMR) for functional-group prediction stratified by the number of functional groups in each molecule on NIST, SDBS, and USPTO. b, EMR for functional-group prediction stratified by the number of heavy atoms in each molecule on NIST, SDBS, and USPTO. c, Precision–recall curves for functional-group prediction on NIST, SDBS, and USPTO.

Table G | Per-functional-group classification performance on the NIST benchmark. F1 scores are reported as means ± standard deviations across five folds. The best result for each functional group is highlighted in bold.
<table><tr><td>Functional group</td><td>Positive rate (%)</td><td>Logistic</td><td>KNN</td><td>RF</td><td>XGBoost</td><td>IRAnalysis</td><td>FCGFormer</td><td>UltraIR</td></tr><tr><td>Alkane</td><td>79.76</td><td> $0 . 9 1 9 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 9 3 1 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 9 5 2 _ { \pm 0 . 0 0 3 }$ </td><td> $0 . 9 6 6 _ { \pm 0 . 0 0 3 }$ </td><td> $0 . 9 6 0 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 9 6 2 _ { \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 9 7 8 { \scriptstyle \pm 0 . 0 0 1 } }$ </td></tr><tr><td>Methyl</td><td>63.08</td><td> $0 . 7 8 5 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 8 6 5 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 8 7 7 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $0 . 9 2 0 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 9 1 0 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 9 1 1 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 9 5 0 { \scriptstyle \pm 0 . 0 0 2 } }$ </td></tr><tr><td>Alkene</td><td>13.16</td><td> $0 . 4 4 9 _ { \pm 0 . 0 2 4 }$ </td><td> $0 . 6 8 4 _ { \pm 0 . 0 2 4 }$ </td><td> $0 . 6 8 9 _ { \pm 0 . 0 2 7 }$ </td><td> $0 . 7 8 3 _ { \pm 0 . 0 1 2 }$ </td><td>0.761±0.011</td><td> $0 . 7 3 1 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td>0.869±0.015</td></tr><tr><td>Alkyne</td><td>2.04</td><td> $0 . 5 4 4 _ { \pm 0 . 0 4 7 }$ </td><td> $0 . 8 2 1 _ { \pm 0 . 0 3 1 }$ </td><td>0.747±0.057</td><td> $0 . 8 7 4 _ { \pm 0 . 0 2 3 }$ </td><td> $0 . 8 7 4 _ { \pm 0 . 0 1 5 }$ </td><td> $0 . 8 8 4 _ { \pm 0 . 0 3 5 }$ </td><td> $\mathbf { 0 . 9 4 9 _ { \pm 0 . 0 2 2 } }$ </td></tr><tr><td>Alcohols</td><td>23.84</td><td> $0 . 6 8 2 _ { \pm 0 . 0 3 7 }$ </td><td> $0 . 8 2 7 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 8 5 0 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 8 8 7 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 8 8 9 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 8 8 9 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $\mathbf { 0 . 9 2 8 _ { \pm 0 . 0 0 4 } }$ </td></tr><tr><td>Amines</td><td>22.74</td><td> $0 . 6 2 1 { \scriptstyle \pm 0 . 0 4 4 }$ </td><td>0.816±0.005</td><td> $0 . 8 1 2 _ { \pm 0 . 0 1 7 }$ </td><td> $0 . 8 6 5 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 8 4 9 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 8 5 0 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 9 1 8 _ { \pm 0 . 0 0 5 } }$ </td></tr><tr><td>Nitriles</td><td>3.82</td><td> $0 . 2 9 1 _ { \pm 0 . 0 4 8 }$ </td><td> $0 . 5 2 8 _ { \pm 0 . 0 5 7 }$ </td><td> $0 . 4 6 6 _ { \pm 0 . 0 4 8 }$ </td><td> $0 . 7 9 5 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $0 . 6 5 9 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td> $0 . 5 8 6 _ { \pm 0 . 0 3 3 }$ </td><td> $\mathbf { 0 . 9 3 1 { \scriptstyle \pm 0 . 0 1 6 } }$ </td></tr><tr><td>Aromatics</td><td>58.84</td><td> $0 . 8 7 2 _ { \pm 0 . 0 1 6 }$ </td><td> $0 . 9 2 4 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td>0.920±0.011</td><td> $0 . 9 6 3 _ { \pm 0 . 0 0 4 }$ </td><td> $0 . 9 5 3 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 9 5 6 _ { \pm 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 9 8 2 _ { \pm 0 . 0 0 5 } }$ </td></tr><tr><td>Alkyl halides</td><td>25.41</td><td> $0 . 5 4 2 _ { \pm 0 . 0 3 4 }$ </td><td> $0 . 7 6 4 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td>0.750±0.010</td><td> $0 . 8 1 0 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $0 . 8 0 0 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 7 9 1 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $\mathbf { 0 . 8 8 9 { \scriptstyle \pm 0 . 0 1 0 } }$ </td></tr><tr><td>Esters</td><td>11.78</td><td> $0 . 7 3 2 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td> $0 . 8 8 4 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $0 . 8 3 2 _ { \pm 0 . 0 2 9 }$ </td><td> $0 . 9 0 7 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 9 1 2 _ { \pm 0 . 0 1 3 }$ </td><td> $0 . 9 0 2 _ { \pm 0 . 0 1 9 }$ </td><td> $\mathbf { 0 . 9 4 5 _ { \pm 0 . 0 1 0 } }$ </td></tr><tr><td>Ketones</td><td>8.96</td><td> $0 . 4 1 9 _ { \pm 0 . 0 4 5 }$ </td><td> $0 . 7 5 6 _ { \pm 0 . 0 2 3 }$ </td><td> $0 . 6 7 7 _ { \pm 0 . 0 2 6 }$ </td><td> $0 . 7 8 3 _ { \pm 0 . 0 2 9 }$ </td><td> $0 . 8 0 1 _ { \pm 0 . 0 1 1 }$ </td><td> $0 . 7 8 5 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $\mathbf { 0 . 8 9 4 _ { \pm 0 . 0 1 4 } }$ </td></tr><tr><td>Aldehydes</td><td>1.97</td><td> $0 . 5 5 2 _ { \pm 0 . 0 5 0 }$ </td><td> $0 . 7 8 9 _ { \pm 0 . 0 3 4 }$ </td><td>0.827±0.025</td><td> $0 . 8 7 3 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td> $0 . 8 6 6 _ { \pm 0 . 0 3 5 }$ </td><td> $0 . 8 9 1 _ { \pm 0 . 0 2 2 }$ </td><td> $\mathbf { 0 . 9 4 4 _ { \pm 0 . 0 1 6 } }$ </td></tr><tr><td>Carboxylic acids</td><td>6.85</td><td> $0 . 7 0 6 _ { \pm 0 . 0 1 7 }$ </td><td> $0 . 8 5 6 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 8 5 4 _ { \pm 0 . 0 1 7 }$ </td><td> $0 . 8 9 9 _ { \pm 0 . 0 1 6 }$ </td><td> $0 . 8 8 1 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 8 8 9 _ { \pm 0 . 0 1 2 }$ </td><td> $\mathbf { 0 . 9 3 1 _ { \pm 0 . 0 1 5 } }$ </td></tr><tr><td>Ether</td><td>13.86</td><td> $0 . 5 8 0 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td> $0 . 7 8 4 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td> $0 . 7 1 7 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 8 1 6 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 8 1 8 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td> $0 . 8 1 8 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $\mathbf { 0 . 9 0 5 _ { \pm 0 . 0 1 4 } }$ </td></tr><tr><td>Acyl halides</td><td>0.92</td><td> $0 . 5 5 9 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td>0.827±0.068</td><td> $0 . 7 8 1 { \scriptstyle \pm 0 . 0 4 6 }$ </td><td> $0 . 8 4 3 { \scriptstyle \pm 0 . 0 4 9 }$ </td><td> $0 . 8 6 4 { \scriptstyle \pm 0 . 0 3 7 }$ </td><td> $0 . 8 6 1 { \scriptstyle \pm 0 . 0 5 0 }$ </td><td>0.897±0.062</td></tr><tr><td>Amides</td><td>6.19</td><td> $0 . 4 4 3 _ { \pm 0 . 0 4 0 }$ </td><td> $0 . 6 8 0 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $0 . 4 7 8 _ { \pm 0 . 0 5 2 }$ </td><td> $0 . 6 8 5 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td> $0 . 7 1 2 _ { \pm 0 . 0 1 0 }$ </td><td> $0 . 6 7 8 _ { \pm 0 . 0 1 6 }$ </td><td> $\mathbf { 0 . 7 9 1 _ { \pm 0 . 0 1 4 } }$ </td></tr><tr><td>Nitro</td><td>5.63</td><td> $0 . 7 6 1 _ { \pm 0 . 0 1 8 }$ </td><td> $0 . 8 4 5 _ { \pm 0 . 0 2 6 }$ </td><td> $0 . 8 1 1 _ { \pm 0 . 0 3 1 }$ </td><td> $0 . 8 8 9 _ { \pm 0 . 0 2 7 }$ </td><td> $0 . 8 8 4 _ { \pm 0 . 0 1 7 }$ </td><td> $0 . 8 6 7 _ { \pm 0 . 0 1 9 }$ </td><td> $\mathbf { 0 . 9 1 9 _ { \pm 0 . 0 0 6 } }$ </td></tr></table>

Table H | Per-functional-group classification performance on the SDBS benchmark. F1 scores are reported as means ± standard deviations across five folds. The best result for each functional group is highlighted in bold.
<table><tr><td>Functional group</td><td>Positive rate (%)</td><td>Logistic</td><td>KNN</td><td>RF</td><td>XGBoost</td><td>IRAnalysis</td><td>FCGFormer</td><td>UltraIR</td></tr><tr><td>Alkane</td><td>74.19</td><td> $0 . 8 5 8 _ { \pm 0 . 0 1 5 }$ </td><td> $0 . 8 7 5 _ { \pm 0 . 0 1 5 }$ </td><td> $0 . 8 6 6 _ { \pm 0 . 0 1 6 }$ </td><td> $0 . 9 1 5 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 7 2 3 _ { \pm 0 . 0 2 8 }$ </td><td> $0 . 6 3 3 _ { \pm 0 . 0 1 5 }$ </td><td>0.941±0.008</td></tr><tr><td>Methyl</td><td>55.78</td><td>0.695±0.039</td><td> $0 . 7 6 7 _ { \pm 0 . 0 1 1 }$ </td><td> $0 . 7 7 6 _ { \pm 0 . 0 1 1 }$ </td><td> $0 . 8 5 2 _ { \pm 0 . 0 1 1 }$ </td><td> $0 . 0 5 3 _ { \pm 0 . 0 2 0 }$ </td><td> $0 . 0 5 2 _ { \pm 0 . 0 1 5 }$ </td><td>0.881±0.008</td></tr><tr><td>Alkene</td><td>12.07</td><td> $0 . 4 0 1 _ { \pm 0 . 0 4 2 }$ </td><td> $0 . 4 7 0 _ { \pm 0 . 0 7 3 }$ </td><td> $0 . 3 0 1 { \scriptstyle \pm 0 . 0 8 5 }$ </td><td> $0 . 5 6 5 _ { \pm 0 . 0 3 2 }$ </td><td> $0 . 1 0 7 _ { \pm 0 . 0 2 9 }$ </td><td> $0 . 1 8 7 _ { \pm 0 . 0 3 4 }$ </td><td> $\mathbf { 0 . 6 9 9 _ { \pm 0 . 0 5 0 } }$ </td></tr><tr><td>Alkyne</td><td>1.07</td><td> $0 . 4 1 5 _ { \pm 0 . 0 8 9 }$ </td><td> $0 . 5 3 6 _ { \pm 0 . 1 0 4 }$ </td><td>0.102±0.071</td><td> $0 . 5 2 5 { \scriptstyle \pm 0 . 0 6 7 }$ </td><td> $0 . 1 9 8 _ { \pm 0 . 0 8 7 }$ </td><td> $0 . 2 0 8 _ { \pm 0 . 0 6 0 }$ </td><td>0.726±0.061</td></tr><tr><td>Alcohols</td><td>29.72</td><td> $0 . 6 3 1 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td> $0 . 8 0 0 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $0 . 7 7 3 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td> $0 . 8 4 4 _ { \pm 0 . 0 2 7 }$ </td><td> $0 . 2 2 6 _ { \pm 0 . 0 6 2 }$ </td><td> $0 . 2 5 4 { \scriptstyle \pm 0 . 0 5 7 }$ </td><td> $\mathbf { 0 . 8 9 3 _ { \pm 0 . 0 1 8 } }$ </td></tr><tr><td>Amines</td><td>27.35</td><td> $0 . 5 8 7 _ { \pm 0 . 1 0 5 }$ </td><td> $0 . 7 5 3 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 7 0 3 _ { \pm 0 . 0 2 6 }$ </td><td> $0 . 7 8 3 _ { \pm 0 . 0 1 5 }$ </td><td> $0 . 3 4 5 _ { \pm 0 . 0 6 1 }$ </td><td> $0 . 3 5 9 { \scriptstyle \pm 0 . 0 6 9 }$ </td><td> $\mathbf { 0 . 8 7 2 _ { \pm 0 . 0 1 7 } }$ </td></tr><tr><td>Nitriles</td><td>2.98</td><td> $0 . 5 8 6 _ { \pm 0 . 0 3 9 }$ </td><td>0.278±0.092</td><td> $0 . 1 5 2 _ { \pm 0 . 0 7 7 }$ </td><td> $0 . 8 0 0 { \scriptstyle \pm 0 . 0 4 0 }$ </td><td> $0 . 2 9 7 _ { \pm 0 . 0 4 7 }$ </td><td> $0 . 3 3 8 _ { \pm 0 . 0 8 2 }$ </td><td> $\mathbf { 0 . 8 7 6 _ { \pm 0 . 0 2 5 } }$ </td></tr><tr><td>Aromatics</td><td>60.47</td><td> $0 . 8 3 1 _ { \pm 0 . 0 1 7 }$ </td><td> $0 . 8 6 5 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 8 5 5 { \scriptstyle \pm 0 . 0 5 0 }$ </td><td> $0 . 9 3 5 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $0 . 7 6 1 _ { \pm 0 . 0 4 9 }$ </td><td> $0 . 7 4 1 _ { \pm 0 . 0 5 6 }$ </td><td> $\mathbf { 0 . 9 6 8 _ { \pm 0 . 0 0 4 } }$ </td></tr><tr><td>Alkyl halides</td><td>18.62</td><td> $0 . 3 6 3 _ { \pm 0 . 0 3 5 }$ </td><td> $0 . 4 2 9 _ { \pm 0 . 0 5 5 }$ </td><td> $0 . 2 9 6 _ { \pm 0 . 0 4 5 }$ </td><td> $0 . 4 9 7 _ { \pm 0 . 0 3 0 }$ </td><td>0.187±0.027</td><td> $0 . 1 9 5 _ { \pm 0 . 0 3 4 }$ </td><td> $\mathbf { 0 . 6 0 7 _ { \pm 0 . 0 3 5 } }$ </td></tr><tr><td>Esters</td><td>11.95</td><td> $0 . 7 3 5 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 7 9 9 _ { \pm 0 . 0 1 7 }$ </td><td> $0 . 7 1 9 _ { \pm 0 . 0 2 4 }$ </td><td> $0 . 8 3 6 _ { \pm 0 . 0 1 7 }$ </td><td> $0 . 1 3 0 { \scriptstyle \pm 0 . 0 5 3 }$ </td><td> $0 . 2 0 8 _ { \pm 0 . 0 6 0 }$ </td><td> $\mathbf { 0 . 8 8 4 _ { \pm 0 . 0 1 8 } }$ </td></tr><tr><td>Ketones</td><td>9.24</td><td> $0 . 3 8 4 { \scriptstyle \pm 0 . 0 5 1 }$ </td><td> $0 . 5 4 5 _ { \pm 0 . 0 4 0 }$ </td><td> $0 . 2 2 6 _ { \pm 0 . 1 0 6 }$ </td><td> $0 . 5 6 9 _ { \pm 0 . 0 3 6 }$ </td><td> $0 . 0 3 4 _ { \pm 0 . 0 1 0 }$ </td><td> $0 . 1 0 1 _ { \pm 0 . 0 2 7 }$ </td><td> $\mathbf { 0 . 7 2 0 _ { \pm 0 . 0 3 1 } }$ </td></tr><tr><td>Aldehydes</td><td>2.01</td><td> $0 . 2 5 1 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td> $0 . 4 0 7 _ { \pm 0 . 0 7 5 }$ </td><td> $0 . 1 7 5 _ { \pm 0 . 0 7 4 }$ </td><td> $0 . 4 4 6 _ { \pm 0 . 0 8 2 }$ </td><td> $0 . 0 8 3 _ { \pm 0 . 0 9 7 }$ </td><td> $0 . 1 7 6 _ { \pm 0 . 1 1 3 }$ </td><td> $\mathbf { 0 . 6 4 7 _ { \pm 0 . 0 7 3 } }$ </td></tr><tr><td>Carboxylic acids</td><td>12.17</td><td> $0 . 6 7 7 _ { \pm 0 . 0 1 7 }$ </td><td> $0 . 7 9 1 _ { \pm 0 . 0 4 3 }$ </td><td> $0 . 7 8 5 { \scriptstyle \pm 0 . 0 3 9 }$ </td><td> $0 . 8 4 4 _ { \pm 0 . 0 3 6 }$ </td><td> $0 . 3 0 2 _ { \pm 0 . 0 6 5 }$ </td><td> $0 . 3 1 2 _ { \pm 0 . 0 5 8 }$ </td><td> $\mathbf { 0 . 8 8 2 _ { \pm 0 . 0 3 0 } }$ </td></tr><tr><td>Ether</td><td>14.63</td><td> $0 . 5 1 8 _ { \pm 0 . 0 3 3 }$ </td><td> $0 . 6 1 5 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 3 6 8 _ { \pm 0 . 0 9 2 }$ </td><td> $0 . 6 7 3 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $0 . 0 4 4 _ { \pm 0 . 0 2 9 }$ </td><td> $0 . 1 2 9 _ { \pm 0 . 0 3 3 }$ </td><td> $\mathbf { 0 . 7 8 4 _ { \pm 0 . 0 2 1 } }$ </td></tr><tr><td>Acyl halides</td><td>0.99</td><td>0.606±0.108</td><td> $\mathbf { 0 . 7 2 8 _ { \pm 0 . 1 2 6 } }$ </td><td> $0 . 5 6 0 _ { \pm 0 . 1 5 7 }$ </td><td> $0 . 6 9 2 _ { \pm 0 . 1 1 9 }$ </td><td> $0 . 3 1 0 { \scriptstyle \pm 0 . 1 2 5 }$ </td><td> $0 . 2 5 1 _ { \pm 0 . 1 4 6 }$ </td><td>0.719±0.120</td></tr><tr><td>Amides</td><td>7.95</td><td> $0 . 4 6 8 _ { \pm 0 . 0 4 5 }$ </td><td> $0 . 6 1 2 _ { \pm 0 . 0 3 6 }$ </td><td> $0 . 2 7 9 _ { \pm 0 . 1 1 1 }$ </td><td> $0 . 5 8 3 _ { \pm 0 . 0 5 5 }$ </td><td> $0 . 1 7 5 _ { \pm 0 . 0 5 0 }$ </td><td> $0 . 1 6 8 _ { \pm 0 . 0 4 7 }$ </td><td> $\mathbf { 0 . 7 4 6 _ { \pm 0 . 0 3 0 } }$ </td></tr><tr><td>Nitro</td><td>6.17</td><td> $0 . 6 8 5 _ { \pm 0 . 0 7 6 }$ </td><td> $0 . 6 9 9 _ { \pm 0 . 0 6 6 }$ </td><td> $0 . 6 0 6 _ { \pm 0 . 0 6 7 }$ </td><td> $0 . 7 9 4 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td>0.253±0.097</td><td> $0 . 1 3 9 _ { \pm 0 . 1 0 4 }$ </td><td> $\mathbf { 0 . 8 7 1 _ { \pm 0 . 0 4 0 } }$ </td></tr></table>

Table I | Per-functional-group classification performance on the USPTO benchmark. F1 scores are reported as means ± standard deviations across five folds. The best result for each functional group is highlighted in bold.
<table><tr><td>Functional group</td><td>Positive rate (%)</td><td>Logistic</td><td>KNN</td><td>RF</td><td>XGBoost</td><td>IRAnalysis</td><td>FCGFormer</td><td>UltraIR</td></tr><tr><td>Alkane</td><td>93.15</td><td> $0 . 9 8 9 _ { \pm 0 . 0 0 1 }$ </td><td> $0 . 9 6 4 _ { \pm 0 . 0 0 1 }$ </td><td> $0 . 9 6 5 _ { \pm 0 . 0 0 1 }$ </td><td> $0 . 9 9 4 _ { \pm 0 . 0 0 0 }$ </td><td> $0 . 8 5 6 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 8 4 8 _ { \pm 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 9 9 9 _ { \pm 0 . 0 0 0 } }$ </td></tr><tr><td>Methyl</td><td>73.10</td><td> $0 . 9 1 4 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 8 3 9 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 8 5 1 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 9 6 4 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 0 1 7 _ { \pm 0 . 0 0 7 }$ </td><td> $0 . 0 1 3 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 9 8 1 { \scriptstyle \pm 0 . 0 0 1 } }$ </td></tr><tr><td>Alkene</td><td>11.34</td><td> $0 . 1 6 0 { \scriptstyle \pm 0 . 0 4 0 }$ </td><td> $0 . 2 0 4 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 2 4 0 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 0 0 1 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 6 0 3 _ { \pm 0 . 0 1 7 } }$ </td></tr><tr><td> $\mathrm { A l k y n e }$ </td><td>1.91</td><td> $0 . 3 0 1 _ { \pm 0 . 0 2 2 }$ </td><td> $0 . 1 6 5 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 6 5 3 { \scriptstyle \pm 0 . 0 3 5 }$ </td><td> $0 . 2 3 3 _ { \pm 0 . 0 4 0 }$ </td><td> $0 . 2 9 1 _ { \pm 0 . 0 1 9 }$ </td><td> $\mathbf { 0 . 9 2 9 _ { \pm 0 . 0 1 3 } }$ </td></tr><tr><td>Alcohols</td><td>26.22</td><td> $0 . 6 3 6 _ { \pm 0 . 0 1 8 }$ </td><td> $0 . 4 6 7 _ { \pm 0 . 0 1 2 }$ </td><td> $0 . 6 3 4 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 8 4 0 _ { \pm 0 . 0 0 5 }$ </td><td> $0 . 0 3 2 _ { \pm 0 . 0 0 3 }$ </td><td> $0 . 1 3 7 _ { \pm 0 . 0 2 7 }$ </td><td> $\mathbf { 0 . 9 5 3 _ { \pm 0 . 0 0 2 } }$ </td></tr><tr><td>Amines</td><td>44.48</td><td> $0 . 6 8 2 _ { \pm 0 . 0 2 2 }$ </td><td> $0 . 6 1 3 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 6 5 9 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 8 0 6 _ { \pm 0 . 0 0 4 }$ </td><td> $0 . 0 6 9 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 1 9 5 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $\mathbf { 0 . 9 1 7 _ { \pm 0 . 0 0 3 } }$ </td></tr><tr><td>Nitriles</td><td>6.69</td><td> $0 . 4 2 6 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 3 2 1 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td> $0 . 0 0 1 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 8 1 3 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $0 . 1 5 8 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $0 . 1 3 5 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td> $\mathbf { 0 . 9 4 5 _ { \pm 0 . 0 0 6 } }$ </td></tr><tr><td>Aromatics</td><td>91.42</td><td> $0 . 9 7 6 _ { \pm 0 . 0 0 4 }$ </td><td> $0 . 9 5 4 _ { \pm 0 . 0 0 7 }$ </td><td> $0 . 9 5 5 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 9 8 3 _ { \pm 0 . 0 0 5 }$ </td><td> $0 . 3 6 8 _ { \pm 0 . 0 3 4 }$ </td><td> $0 . 2 6 6 _ { \pm 0 . 0 1 2 }$ </td><td> $\mathbf { 0 . 9 9 3 _ { \pm 0 . 0 0 1 } }$ </td></tr><tr><td> $\mathrm { { A l k y l } }$  halides</td><td>46.50</td><td> $0 . 6 5 3 _ { \pm 0 . 0 1 7 }$ </td><td> $0 . 6 0 0 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 6 6 1 _ { \pm 0 . 0 0 6 }$ </td><td> $0 . 7 3 1 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 0 5 1 _ { \pm 0 . 0 0 6 }$ </td><td> $0 . 0 2 1 _ { \pm 0 . 0 0 8 }$ </td><td> $\mathbf { 0 . 8 0 9 _ { \pm 0 . 0 0 4 } }$ </td></tr><tr><td>Esters</td><td>18.23</td><td> $0 . 6 5 0 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $0 . 5 5 1 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td> $0 . 3 1 2 _ { \pm 0 . 0 1 2 }$ </td><td> $0 . 7 9 0 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $0 . 0 3 1 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 0 1 6 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $\mathbf { 0 . 9 2 5 { \scriptstyle \pm 0 . 0 0 6 } }$ </td></tr><tr><td>Ketones</td><td>8.22</td><td> $0 . 3 1 9 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td> $0 . 2 6 8 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td>0.002±0.001</td><td> $0 . 4 3 3 { \scriptstyle \pm 0 . 0 4 3 }$ </td><td> $0 . 0 0 4 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 0 0 3 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 6 9 2 _ { \pm 0 . 0 2 7 } }$ </td></tr><tr><td>Aldehydes</td><td>2.67</td><td> $0 . 6 5 3 _ { \pm 0 . 0 3 7 }$ </td><td> $0 . 4 2 6 _ { \pm 0 . 0 2 5 }$ </td><td> $0 . 1 1 7 _ { \pm 0 . 0 2 0 }$ </td><td> $0 . 9 5 6 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 4 8 3 _ { \pm 0 . 0 1 3 }$ </td><td> $0 . 7 9 8 _ { \pm 0 . 0 3 5 }$ </td><td> $\mathbf { 0 . 9 8 8 _ { \pm 0 . 0 0 3 } }$ </td></tr><tr><td>Carboxylic acids</td><td>9.73</td><td> $0 . 8 0 6 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 5 2 4 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $0 . 7 3 9 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td> $0 . 9 4 7 _ { \pm 0 . 0 0 5 }$ </td><td> $0 . 4 6 1 _ { \pm 0 . 0 2 4 }$ </td><td> $0 . 6 6 4 _ { \pm 0 . 0 1 7 }$ </td><td> $\mathbf { 0 . 9 8 5 _ { \pm 0 . 0 0 1 } }$ </td></tr><tr><td>Ether</td><td>37.05</td><td> $0 . 6 5 6 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td> $0 . 5 7 1 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td> $0 . 6 2 7 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 7 8 5 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 0 1 9 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 0 1 4 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $\mathbf { 0 . 9 0 5 _ { \pm 0 . 0 0 3 } }$ </td></tr><tr><td>Acyl halides</td><td>0.59</td><td> $0 . 1 3 2 _ { \pm 0 . 0 3 0 }$ </td><td> $0 . 2 7 7 _ { \pm 0 . 0 4 4 }$ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 2 0 5 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $0 . 0 5 4 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $0 . 0 9 2 _ { \pm 0 . 0 4 4 }$ </td><td> $\mathbf { 0 . 7 1 9 { \scriptstyle \pm 0 . 0 2 6 } }$ </td></tr><tr><td>Amides</td><td>25.61</td><td> $0 . 6 5 7 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 6 2 0 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 5 5 8 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td> $0 . 7 6 4 _ { \pm 0 . 0 0 4 }$ </td><td> $0 . 0 2 8 _ { \pm 0 . 0 0 1 }$ </td><td> $0 . 0 1 9 _ { \pm 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 8 9 9 _ { \pm 0 . 0 0 3 } }$ </td></tr><tr><td>Nitro</td><td>5.27</td><td> $0 . 3 4 6 _ { \pm 0 . 0 3 6 }$ </td><td> $0 . 1 9 0 _ { \pm 0 . 0 1 7 }$ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 5 1 8 _ { \pm 0 . 0 2 3 }$ </td><td> $0 . 0 4 9 _ { \pm 0 . 0 1 0 }$ </td><td> $0 . 0 3 1 _ { \pm 0 . 0 0 9 }$ </td><td> $\mathbf { 0 . 8 1 4 _ { \pm 0 . 0 1 4 } }$ </td></tr></table>

Additional molecular structure elucidation results
<table><tr><td rowspan="2">Model</td><td colspan="2">Top-1</td><td colspan="2">Top-5</td><td colspan="2">Top-10</td></tr><tr><td></td><td>Acc (%) ↑ Tanimoto ↑ Acc (%) ↑ Tanimoto ↑</td><td></td><td></td><td> $\operatorname { A c c } \left( \% \right) \uparrow$ </td><td> $\operatorname { T a n i m o t o } \uparrow$ </td></tr><tr><td>UltraIR w/o IR</td><td> $9 . 7 1 _ { \pm 0 . 2 1 }$ </td><td> $0 . 2 9 3 _ { \pm 0 . 0 0 2 }$ </td><td> $2 5 . 5 4 _ { \pm 0 . 4 8 }$ </td><td> $0 . 4 7 6 _ { \pm 0 . 0 0 3 }$ </td><td> $3 3 . 1 7 _ { \pm 0 . 4 3 }$ </td><td> $0 . 5 4 8 _ { \pm 0 . 0 0 3 }$ </td></tr><tr><td>UltraIR w/o Formula</td><td> $3 8 . 9 4 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $0 . 5 9 4 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $4 6 . 6 6 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $0 . 6 7 6 _ { \pm 0 . 0 0 5 }$ </td><td> $4 9 . 4 5 _ { \pm 0 . 4 6 }$ </td><td> $0 . 7 0 2 _ { \pm 0 . 0 0 5 }$ </td></tr><tr><td>UltraIR</td><td> ${ \bf 5 2 . 2 0 _ { \pm 0 . 5 4 } }$ </td><td> $\mathbf { 0 . 6 8 2 _ { \pm 0 . 0 0 4 } }$ </td><td> ${ \bf 5 7 . 4 8 _ { \pm 0 . 7 1 } }$ </td><td> $\mathbf { 0 . 7 4 2 _ { \pm 0 . 0 0 3 } }$ </td><td> ${ \bf 5 7 . 5 7 { \scriptstyle \pm 0 . 7 4 } }$ </td><td> $\mathbf { 0 . 7 4 8 _ { \pm 0 . 0 0 3 } }$ </td></tr></table>

Table J | Ablation of IR and molecular-formula conditioning for molecular structure elucidation on the NIST benchmark. Tanimoto@k denotes the maximum Morgan Tanimoto similarity between the ground-truth structure and the first k candidates. Results are reported as means ± standard deviations across five folds.

![](images/49e65c016affaaaf1ef79e42c0491a6bbafecbb83796b200d3697713fd9f5fb7.jpg)  
Figure B | Additional molecular structure elucidation examples. Six representative examples from the NIST test set comparing the ground-truth molecular structures with candidates generated by UltraIR, UltraIR without pretraining, and the competing structure-elucidation models IRtoMol, AISE, PBSA, and DLIR. Fingerprint-based Tanimoto similarities between each generated candidate and the corresponding ground-truth structure are reported. Green check marks indicate exact structure recovery.

## Additional physicochemical property prediction results

![](images/bd9f0744689b500c65bc45a9e1b1d5bf8f145f8c308f707c27ac280cb9b46122.jpg)  
Figure C | Property-wise physicochemical prediction performance of UltraIR across NIST, SDBS, and USPTO. Predicted-versus-true parity plots are shown for synthetic accessibility (SA) score, log P, topological polar surface area (TPSA), hydrogen-bond donors, hydrogen-bond acceptors, rotatable bonds, the fraction of sp<sup>3</sup>-hybridized carbon atoms (Fraction Csp3), quantitative estimate of drug-likeness (QED), aromatic rings, and aliphatic rings. The identity line denotes ideal agreement.

![](images/ebe4d3bb5c2086865c204cd39486be57c69f7ca944359c8c0d78358e8bef0ec8.jpg)  
Figure D | Confusion matrices for genus-level bacterial classification. Confusion matrices are shown for UltraIR and XGBoost across the nine evaluated bacterial genera. Rows denote actual genera and columns denote predicted genera, with each cell indicating the percentage of samples assigned to the corresponding predicted genus.

Additional medicinal-herb geographic origin traceability results  
a  
![](images/9656196a22d229887ee7d8bcdcd366197553d2ef86d28d946dac674c9ad49636.jpg)

![](images/65b53d1e0e08440b14e93cafa1db436e2a732fed21a3e21e12be6f076beca7d5.jpg)

b  
![](images/8be0cacac5347ee365a209da55a2d6d2cbf3c38c3c1510b1c2159c2260ce89e0.jpg)

![](images/7d6d03ff0f864134eb4995a4521d3067f252bc0f2da58ce9cd60c20a76ee3f95.jpg)  
Figure E | Confusion matrices for medicinal-herb geographic origin traceability. ${ \mathbf { a } } ,$ Confusion matrices for Jinyinhua geographic origin traceability, comparing UltraIR and XGBoost across Shandong, Henan, Hebei, and Sichuan. b, Confusion matrices for Shanyinhua geographic origin traceability, comparing UltraIR and XGBoost across Hunan, Hubei, Sichuan, Henan, and Guangdong. Rows denote actual origins and columns denote predicted origins, with each cell indicating the percentage of samples assigned to the corresponding predicted origin.

Additional microplastics classification results  
![](images/30013c5f83083629d9d2a1c822d00afa788252668023421da33ef59e0c94699f.jpg)

![](images/3ade090275e84c4cb0a0e95f2eeb9414eaf01b21a355861a0fd06bb7941101c4.jpg)  
Figure F | Confusion matrices for microplastics classification. Confusion matrices are shown for UltraIR and Softmax across the 18 evaluated polymer classes. Rows denote actual plastic types and columns denote predicted plastic types, with each cell indicating the percentage of samples assigned to the corresponding predicted class.

Additional cross-instrument and cross-laboratory generalization results

![](images/52ad3734a8a6102422803253cdb789f6b695aba06d5b55cf8121d0f1ffa41d26.jpg)

b  
![](images/bb75ad2f6844dd02d886d47c5ee3bb3b87d6e463a3e47a44a472a1fb065022c7.jpg)  
Figure G | Cross-instrument and cross-laboratory generalization for soil property prediction. All models were trained on the Kellogg Soil Survey Laboratory (KSSL) subset of OSSL and evaluated by zero-shot inference on the ICRAF– ISRIC subset. ${ \mathbf { a } } ,$ Prediction performance for soil texture and acidity properties, including clay content, silt content, sand content, and $\mathrm { p H } \ \mathrm { ( H _ { 2 } O ) }$ , evaluated using Normalized MAE, Normalized RMSE, and $R ^ { 2 } .$ . b, Prediction performance for soil chemical properties, including cation exchange capacity (CEC) and exchangeable Ca, Mg, K, and $\mathrm { N a , }$ evaluated using the same metrics.