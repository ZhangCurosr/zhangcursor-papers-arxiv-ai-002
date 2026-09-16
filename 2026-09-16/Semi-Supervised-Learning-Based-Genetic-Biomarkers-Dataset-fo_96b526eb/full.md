# Semi-Supervised Learning-Based Genetic Biomarkers Dataset for Multiple-Stage Hepatocellular Carcinoma Prediction

Ahmed Ammar Kubba<sup>∗</sup>, Manar Abu Talib<sup>∗</sup>, Jibran Sualeh Muhammad<sup>†</sup>, Ali Bou Nassif<sup>‡</sup> Abdalla Sayed Mohamed<sup>∗</sup>, Darko Castven<sup>§</sup>, Jens U. Marquardt<sup>§</sup>

∗ Department of Computer Science, University of Sharjah, Sharjah, UAE

Emails: {U23103280, mtalib, U22103623}@sharjah.ac.ae

<sup>†</sup> Department of Biomedical Sciences, University of Birmingham, Birmingham, United Kingdom Email: dr.jibran@live.com

<sup>‡</sup> Department of Computer Engineering, University of Sharjah, Sharjah, UAE Email: anassif@sharjah.ac.ae

<sup>§</sup> Department of Medicine I, University Medical Center Schleswig-Holstein, Campus Lubeck, L ¨ ubeck, Germany¨ Emails: {Darko.Castven, Jens.Marquardt}@uksh.de

Abstract—Liver cancer is a complex disease responsible for a high number of deaths across the globe each year, making automated solutions for liver cancer classification urgent. The most common form of liver cancer is hepatocellular carcinoma (HCC), accounting for over 90% of liver cancer cases. There is a distinct lack of publicly available HCC datasets utilizing genomic data, which is necessary for training artificial intelligence (AI) models for automated HCC classification. This study proposes constructing a multi-stage HCC dataset using XGBoost and Semi-Supervised learning on three separate datasets of genomic biomarkers, utilizing their existing labels in the Semi-Supervised learning process to label the proposed dataset. The proposed dataset consists of 770 patient samples in total, categorized into five classes that represent normal tissue alongside different stages of HCC. Each sample in the dataset consists of 11,150 different gene expression levels. The XGBoost model demonstrated a final classification accuracy of 96.5% during the Semi-Supervised learning process.

Keywords—Liver Cancer Prediction, Hepatocellular Carcinoma, Machine Learning, Semi-Supervised Learning, Multi-Omics Data

## I. INTRODUCTION AND RELATED WORK

Liver cancer was ranked sixth in incidence with 905,677 cases and fourth in mortality with 830,180 deaths globally in 2023. With an expected incidence surpassing 1 million cases by 2025, it remains a major global health concern [1]. Hepatocellular Carcinoma (HCC) is a type of cancer that develops from mutations of liver cells called hepatocytes. HCC is responsible for over 90% of liver cancer cases [2]. Modern research has focused extensively on non-invasive biomarkers and imaging techniques like MRI. Machine learning models for diagnosing liver cancer have also been of interest to researchers. These innovations have significantly advanced HCC research by aiding in prediction of tumour recurrence and identification [3]. Recent studies have demonstrated correlations between HCC and various biomarkers for identification and progression. For instance, anuradha et al. [4] identified

28 metabolites and 169 genes that correlate with aggressive HCC progression and outcomes, with supporting data made publicly available. During cancer development, liver cells often present distinct molecular signatures, and release certain tumour-associated molecules into body fluid, e.g., blood, urine or stool, that could be monitored for the onset or progression of HCC. There has been a rise in technologies that operate using techniques like chemiluminescence immunoassay, enzymelinked immunosorbent assay, immunosensor, proteomics, and liquid biopsy [5]. Different multi-omics approaches have also been explored in the literature [6].

In the early stages of HCC, tumor sizes are typically less than 5 cm in diameter and range from one to three tumors. Early HCC rarely shows significant symptoms or liver dysfunction. There is also no significant spread to nearby blood vessels or organs, resulting in a better prognosis. Early detection is, therefore, crucial for patient survival. In contrast, advanced HCC involves larger and multiple tumors, often with invasion into blood vessels or other organs, and has a poor prognosis [7]. Low-grade dysplastic nodules (LGDN) are early precancerous lesions that resemble regenerative nod ules with mild atypia, indicating slight abnormalities in cell structure. They also show less aggressive behaviour and fewer molecular changes associated with cancer progression. Highgrade dysplastic nodules (HGDN), on the other hand, exhibit more pronounced atypia and are considered more advanced precancerous lesions with a higher risk of progression to HCC. HGDN often display molecular alternations which is common in early stages of HCC, making them critical targets for early cancer detection. Various biomarkers that can help in distinguishing between LGDN and HGDN [8].

Das et al. [9] introduced a novel approach for detecting cancer genes using different DNA sequences. The authors used the VGG16 deep learning model and the NCBI dataset. Their training data consisted of sequences of four healthy genes and four HCC genes, using three numerical mapping techniques to digitize the gene sequences. The genes were examined in both one-dimensional and two-dimensional forms using convolutional neural networks (CNN). Their CNN model achieved an 80.36% accuracy in the one-dimensional form. The SVM and VGG16 models achieved a high 98.86%, and 100% accuracy with fine-tuned VGG16 layers. This method effectively extracted features to distinguish HCC from normal liver gene sequences, allowing for broader applications with larger datasets and different cancer types. One critical limitation to this approach was the absence of enough genes and data in the used dataset, which was not large enough to provide a sufficient training set to design a new CNN model. Other studies in the literature have also noted a general limitation in available liver cancer medical data and small sample sizes, leading to inevitable constraints on liver cancer research [10]– [13].

In this study, we construct a dataset containing multiomics genetic data for multi-stage HCC classification for five different developmental hepatocellular carcinoma stages for the patient. The main contributions of this work are as follows: This study utilizes semi-supervised learning based on several different multi-omics datasets to construct the primary HCC dataset used to train our deep learning model, which addresses the challenge of limited and small HCC datasets which other studies tend to face. Additionally, our HCC dataset contains five different categories for patient tissue samples representing different stages of HCC, with 770 samples each containing 11,150 genomic expressions, making it more detailed compared to other publicly available datasets.

This paper is structured in the following manner: The first section introduces the research subject, HCC data and stages of development, and relevant background information, in addition to a review of the related literature, highlighting key drawbacks and limitations in the existing studies. The following section covers the Methodology, including the datasets, pre-processing and the semi-supervised learning. The third section covers the results and their discussion, and the fourth section contains the conclusion and future work of the project.

## II. METHODOLOGY

This section describes the methodology of the project in detail, which consists of obtaining the base dataset from medical domain experts in addition to dataset pre-processing and semi-supervised learning using three different datasets to build the final HCC dataset. Figure 1 illustrates the overall steps of the methodology, consisting of the initial HCC data collection phase from the three datasets, after which the data is pre-processed and properly formatted. Finally, the semisupervised learning phase is conducted to construct the HCC dataset. The next sub-sections will elaborate on each phase in Fig. 1.

## A. Lubeck Dataset and Target Classes

The five classes or ground-truth labels in the Lubeck dataset are described in Table I, which defines the patient tissue samples as five distinct categories: ”Surrounding Liver (sl),” which indicates no HCC gene expressions; ”Early hepatocellular carcinoma (ehcc),” which represents early-stage HCC diagnosis; ”Progressed hepatocellular carcinoma (phcc),” which represents progressed HCC diagnosis; ”Low-grade dysplastic nodules (lgdn),” which is associated with a lower risk of developing HCC [14]; and ”High-grade dysplastic nodules (hgdn),” which is associated with a higher risk of developing HCC. This private dataset serves as the basis for our HCC dataset, as it contains the most detailed labels for HCC stages in comparison to the two other publicly available datasets and thus represents the starting point for the semi-supervised learning phase.

![](images/752b24c8cf8a9de924d2bc09d890cbc52c91700c48a9ba78a820856e08000610.jpg)  
Fig. 1. Visual summary of the paper methodology, outlining each step in the process of constructing the HCC dataset.

TABLE I  
DESCRIPTION OF THE DATASET CLASSES, CONSISTING OF SL, EHCC, PHCC, LGDN, AND HGDN.
<table><tr><td rowspan=1 colspan=1>Class</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=1 colspan=1>Surrounding Liver(SL)</td><td rowspan=1 colspan=1>This class represents patient tissue samples takenfrom the surrounding liver area of the patient,which contains gene expressions that indicate noHCC in the patient.</td></tr><tr><td rowspan=1 colspan=1>EarlyHepatocellularCarcinoma(EHCC)</td><td rowspan=1 colspan=1>This class represents samples of patients with adiagnosis of early-stage HCC.</td></tr><tr><td rowspan=1 colspan=1>ProgressedHepatocellularCarcinoma(PHCC)</td><td rowspan=1 colspan=1>This class is assigned to patients with a diagnosisof progressed HCC.</td></tr><tr><td rowspan=1 colspan=1>Low-GradeDysplastic Nodules(LGDN)</td><td rowspan=1 colspan=1>Dysplastic nodules are associated with a higherrisk of developing HCC; low-grade dysplastic nod-ules represent a much lower risk.</td></tr><tr><td rowspan=1 colspan=1>High-GradeDysplastic Nodules(HGDN)</td><td rowspan=1 colspan=1>High-grade dysplastic nodules indicate a high riskof developing HCC, as represented by this class.</td></tr></table>

## B. Dataset Pre-Processing

The main dataset used in training the artificial intelligence (AI) model was built using a semi-supervised learning approach based on three source datasets that are detailed in this section.

1. Lubeck Dataset. This private dataset contains the original patients’ sample data collected by a team of medical domain experts in the Lubeck university lab. It consists of 29 samples in total, with 16,381 gene expression features for each patient. It also consists of the five class labels for categorizing each of the 29 patients: sl, ehcc, phcc, lgdn, and hgdn.

2. The Cancer Genome Atlas (TCGA) Dataset. Launched in 2006 as a collaborative endeavour between the National Cancer Institute (NCI) and the National Human Genome Research Institute, The Cancer Genome Atlas [15] is a cancer genomics initiative which has profiled over 20,000 primary cancer and corresponding normal samples across 33 different cancer types. This is a public HCC dataset which consists of 782 patient samples and 16,382 gene expression features for each patient. The dataset is categorized into three main class labels: Normal, HCC, and Transition.

3. GSE89377 Dataset. GSE89377 is a gene expression dataset (publicly available in the National Center for Biotechnology Information Gene Expression Omnibus) which contains 107 patient liver tissue samples that cover nine stages of HCC development, with normal liver tissue used as control and around 48,000 gene expression features for each sample [16].

The GSE89377 dataset required several preprocessing steps to be utilized in this work. Firstly, it was necessary to match the gene keywords to their corresponding symbols in the Illumina HumanHT-12 V3.0 Expression BeadChip (GPL6947), which is a widely-used microarray platform used for gene expression profiling in human samples [17]. This was done to ensure computability before adding samples from the GSE89377 dataset to the Lubeck and TCGA datasets. The nine class labels covered in the dataset for patient classification are as follows: ‘Normal’, ‘Chronic Hepatitis with Low/High Grade’, ‘Cirrhosis’, ‘Dysplastic Nodules with Low/High Grade’, ‘Early HCC’, ‘HCC TG1/TG2/TG3’. Additionally, only the gene expression features in common between all three datasets were kept, and all other features were discarded. This was done to enable combining the samples from all dataset sources into one HCC dataset.

## C. Semi-Supervised Learning

The proposed dataset was created using a semi-supervised machine learning approach. Semi-supervised learning enhances the performance of traditional supervised learning, which requires labeled data samples for model training [18], by enabling researchers to make use of unlabelled data. In recent years, it has attracted increasing interest from researchers as one possible approach to reducing dependence on labelled datasets [19]. As the GSE89377 and TCGA datasets contain different labels than the five target labels in the Lubeck dataset, the usage of semi-supervised learning helped us overcome this obstacle.

The appropriate samples from the GSE89377 dataset were first added to the Lubeck and TCGA datasets according to their class labels, which were mapped to the five class labels in the other datasets based on medical domain expert analysis and insight. The liver cirrhosis samples were reclassified as ‘Normal’ samples and added to the TCGA dataset, whereas the chronic hepatitis samples were discarded as they are unrelated to hepatocellular carcinoma. The ‘Normal’ samples were classified as ‘sl’ and added to the Lubeck dataset alongside the dysplastic nodules samples as lgdn/hgdn and Early HCC samples as ‘ehcc’. The HCC Tumor Grade 1 (TG1) samples were classified as HCC and added to the TCGA dataset, whereas the TG2 and TG3 samples were mapped to the Lubeck dataset as progressed HCC (phcc) samples. The semi-supervised learning approach, as visualized in Figure 2 and summarized in Figure 3, consisted first in training an XGBoost model on the base Lubeck dataset, which contains the five target classes. Afterwards, the trained model predicted the classes in the TCGA dataset. High-confidence predictions (as in, predictions with a high probability score) which match the real TCGA class labels were added to the Lubeck dataset as new samples. Finally, the new model was trained on the expanded Lubeck dataset to compare its new performance to the original dataset, with a stratified training and testing dataset split of 70% and 30%, respectively.

![](images/9a7eaed78d6cdfdf92d07e4c68ebcbf0562b0246b41f3f4de5486b8402ca96e0.jpg)  
Fig. 2. Illustration of the Semi-Supervised learning process, using three different sources for constructing the HCC dataset.

These steps were repeated until convergence. The convergence criteria was defined by the expanded dataset leading to the model’s performance degrading or not changing with further iterations. The following are the basic conditions for adding a prediction made on the TCGA dataset to the Lubeck dataset, made with the advice of medical domain experts: 1) It must satisfy the defined prediction confidence threshold (40%); 2) If the true label is “HCC”, the prediction must be either “ehcc” or “phcc”; 3) If the true label is “Transition”, the prediction must be either “lgdn” or “hgdn”; 4) If the true label is “Normal”, the prediction must be “sl”.

## III. RESULTS AND DISCUSSION

This section discusses the results of the semi-supervised learning process, which consist of the final XGBoost model and its performance, in addition to the HCC dataset and its class distribution and genomic features.

![](images/1087c86cb9eef0ba1c338a6b40d1224b95e7b56dbe26f0cd99bab07565512506.jpg)  
Fig. 3. Visual summary of the Semi-Supervised Training process, using the XGBoost model to create the HCC dataset.

## A. XGBoost Model

The final performance metrics of the XGBoost model that was trained on the HCC dataset using all the gene features were recorded in Table II. The XGBoost model demonstrates strong performance, achieving 96.5% accuracy, 90.6% precision, 88.9% recall, and an F1 score of 89.3%, showing a balanced ability to correctly identify multiple stages of HCC samples while minimizing false positives. The low crossentropy loss of 0.1015 indicates that the model is highly confident in its predictions. However, the dataset class imbalance, noted in the Conclusion section, must also be taken into consideration when comparing the performance of the XGBoost model with other models, as it was mainly used for constructing the dataset during the semi-supervised learning process. The minor trade-off between precision and recall means that the XGBoost model leans towards reducing false positives, which can be desirable in medical applications. The model’s performance across the five different classes can also be observed in Figure 4, and the confusion matrix in Figure 5.

TABLE II  
THIS TABLE RECORDS THE XGBOOST PERFORMANCE METRICS, CONSISTING OF ACCURACY, PRECISION, RECALL, AND F1 SCORE ON THE HCC DATASET.
<table><tr><td rowspan=1 colspan=1>Class</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>F1-Score</td><td rowspan=1 colspan=1>Support</td></tr><tr><td rowspan=1 colspan=1>ehcc</td><td rowspan=1 colspan=1>98.9%</td><td rowspan=1 colspan=1>96.7%</td><td rowspan=1 colspan=1>97.8%</td><td rowspan=1 colspan=1>92</td></tr><tr><td rowspan=1 colspan=1>hgdn</td><td rowspan=1 colspan=1>66.7%</td><td rowspan=1 colspan=1>72.7%</td><td rowspan=1 colspan=1>69.6%</td><td rowspan=1 colspan=1>11</td></tr><tr><td rowspan=1 colspan=1>lgdn</td><td rowspan=1 colspan=1>87.5%</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>93.3%</td><td rowspan=1 colspan=1>21</td></tr><tr><td rowspan=1 colspan=1>phcc</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>99</td></tr><tr><td rowspan=1 colspan=1>sl</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>75%</td><td rowspan=1 colspan=1>85.7%</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>96.5%</td><td rowspan=1 colspan=1>96.5%</td><td rowspan=1 colspan=1>96.5%</td><td rowspan=1 colspan=1>96.5%</td></tr><tr><td rowspan=1 colspan=1>Macro Avg</td><td rowspan=1 colspan=1>90.6%</td><td rowspan=1 colspan=1>88.9%</td><td rowspan=1 colspan=1>89.3%</td><td rowspan=1 colspan=1>231</td></tr><tr><td rowspan=1 colspan=1>Weighted Avg</td><td rowspan=1 colspan=1>96.8%</td><td rowspan=1 colspan=1>96.5%</td><td rowspan=1 colspan=1>96.6%</td><td rowspan=1 colspan=1>231</td></tr></table>

![](images/462124f161461632029808ee7bc0028837629a95aa592037c07c0c7d513919ab.jpg)  
Fig. 4. Visual comparison of the XGBoost model’s performance across the HCC dataset’s five classes.

![](images/2a9c5a668a30f27959e0e4697bdffd03f5c0f9c6e592aad50cd926244bed2b55.jpg)  
Fig. 5. Confusion matrix for the XGBoost model, visualizing the misclassifications of the model across the five classes.

## B. HCC Dataset

The constructed HCC dataset consists of 770 total samples taken from different patients representing their multi-omics genetic expression data. The frequency distribution of the dataset’s classes can be observed in Figure 6, which shows a significant imbalance, where the phcc and ehcc classes have the highest number of samples while lgdn, hgdn, and sl are underrepresented in comparison. This imbalance is a common issue in medical datasets beyond just liver cancer data [20], and can impact model performance if not taken into consideration, potentially leading to model bias toward the majority classes during prediction. Each patient has 11,150 feature columns that indicate the different genomic expression levels of various genes, which can be potentially used to make an informed prediction of the HCC diagnosis stage of the patient. It can be observed in Figure 6 that the most frequent samples in the dataset belong to the ‘phcc’ and ‘ehcc classes, whereas the least frequent samples belong to the ‘sl class with less than 50 samples belonging to that class. A principal component analysis (PCA) plot of the dataset can be observed in Figure 7, which visualizes how the five classes are distributed in feature space.

![](images/363a6d42d94d5c0e27ca4f2acb2f6861c1fec6f742a7c4602bfe537a016718c2.jpg)  
Fig. 6. This graph illustrates the frequency distribution of the different classes in the HCC dataset.

![](images/6d43f18036a3ad2031432762038b9b53706d150321c0ea442a89333bf0f44ed0.jpg)  
Fig. 7. PCA plot of the HCC dataset, illustrating the distribution of the classes in feature space.

## IV. CONCLUSION

This work presents a comprehensive dataset for hepatocellular carcinoma analysis by constructing a relatively large collection of high-dimensionality data using a semi-supervised learning-based approach based on one private dataset in addition to two public datasets, mapping the labels in the latter two datasets to the labels of the Lubeck dataset. The constructed HCC dataset can support advancements in medical research, early diagnosis, and treatment planning by enabling AI-driven models to enhance accuracy and efficiency in liver cancer detection. Researchers can use this data to develop automated classification systems and predictive models that can contribute to improved liver cancer patient outcomes.

Limitations and future work include the dataset imbalance limitation, which can be addressed in future research through techniques like class weighting or synthetic minority oversampling to help mitigate the effects of class imbalance and improve model generalization. Additionally, further validation using external datasets and probability calibration checks can be conducted to further ensure model robustness and generalizability. Further interdisciplinary collaboration between medical professionals and machine learning experts can lead to improving the dataset’s quality and sample diversity, in addition to finding relevant features or genes in the dataset which can be clinically validated as novel contributions to medical research.

## DECLARATION OF INTERESTS

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## REFERENCES

[1] R. Masuzaki, “Liver cancer: Improving standard diagnosis and therapy,” p. 4602, 2023.

[2] B. Farasati Far, D. Rabie, P. Hemati, P. Fooladpanjeh, N. Faal Hamedanchi, N. Broomand Lomer, A. Karimi Rouzbahani, and M. R. Naimi-Jamal, “Unresectable hepatocellular carcinoma: a review of new advances with focus on targeted therapy and immunotherapy,” Livers, vol. 3, no. 1, pp. 121–160, 2023.

[3] T. A. Addissouky, I. E. T. E. Sayed, M. M. Ali, Y. Wang, A. E. Baz, A. A. Khalil, and N. Elarabany, “Latest advances in hepatocellular carcinoma management and prevention through advanced technologies,” Egyptian Liver Journal, vol. 14, no. 1, p. 2, 2024.

[4] A. Budhu, S. Roessler, X. Zhao, Z. Yu, M. Forgues, J. Ji, E. Karoly, L.-X. Qin, Q.-H. Ye, H.-L. Jia et al., “Integrated metabolite and gene expression profiles identify lipid biomarkers associated with progression of hepatocellular carcinoma and patient outcomes,” Gastroenterology, vol. 144, no. 5, pp. 1066–1075, 2013.

[5] Y. Pan, H. Chen, and J. Yu, “Biomarkers in hepatocellular carcinoma: current status and future perspectives,” Biomedicines, vol. 8, no. 12, p. 576, 2020.

[6] F. Chen, J. Wang, Y. Wu, Q. Gao, and S. Zhang, “Potential biomarkers for liver cancer diagnosis based on multi-omics strategy,” Frontiers in Oncology, vol. 12, p. 822449, 2022.

[7] M.-J. Kuo, L.-R. Mo, and C.-L. Chen, “Factors predicting long-term outcomes of early-stage hepatocellular carcinoma after primary curative treatment: the role of surgical or nonsurgical methods,” BMC cancer, vol. 21, no. 1, p. 250, 2021.

[8] Y. Duan, X. Xie, Q. Li, N. Mercaldo, A. E. Samir, M. Kuang, and M. Lin, “Differentiation of regenerative nodule, dysplastic nodule, and small hepatocellular carcinoma in cirrhotic patients: A contrast-enhanced ultrasound–based multivariable model analysis,” European Radiology, vol. 30, no. 9, pp. 4741–4751, 2020.

[9] B. Das and S. Toraman, “Deep transfer learning for automated liver cancer gene recognition using spectrogram images of digitized dna sequences,” Biomedical Signal Processing and Control, vol. 72, p. 103317, 2022.

[10] Y. Chen, H. Lin, W. Zhang, W. Chen, Z. Zhou, A. A. Heidari, H. Chen, and G. Xu, “Icycle-gan: Improved cycle generative adversarial networks for liver medical image generation,” Biomedical Signal Processing and Control, vol. 92, p. 106100, 2024.

[11] Y. Yang and G. Mirzaei, “Performance analysis of data resampling on class imbalance and classification techniques on multi-omics data for cancer classification,” PLoS One, vol. 19, no. 2, p. e0293607, 2024.

[12] L. Zhi, Z.-H. Chen, and J. Deng, “Parameter changes and influencing factors in sixty patients with interventional surgery for liver cancer diagnoses,” World Journal of Gastrointestinal Surgery, vol. 17, no. 2, p. 99581, 2025.

[13] S. Oh, Y.-H. Baek, S. Jung, S. Yoon, B. Kang, S.-h. Han, G. Park, J. Y. Ko, S.-Y. Han, J.-S. Jeong et al., “Identification of signature gene set as highly accurate determination of metabolic dysfunction-associated steatotic liver disease progression,” Clinical and Molecular Hepatology, vol. 30, no. 2, p. 247, 2024.

[14] J. Wu, Q. Zhao, Y. Wang, F. Xiao, W. Cai, S. Liu, Z. Du, X. Yu, F. Liu, J. Yu et al., “Feeding artery: a valuable feature for differentiation of regenerative nodule, dysplastic nodules and small hepatocellular carcinoma in ceus li-rads,” European Radiology, vol. 34, no. 2, pp. 745– 754, 2024.

[15] T. Baird and R. Roychoudhuri, “Gs-tcga: gene set-based analysis of the cancer genome atlas,” Journal of Computational Biology, vol. 31, no. 3, pp. 229–240, 2024.

[16] Q. Shen, J. W. Eun, K. Lee, H. S. Kim, H. D. Yang, S. Y. Kim, E. K. Lee, T. Kim, K. Kang, S. Kim et al., “Barrier to autointegration factor 1, procollagen-lysine, 2-oxoglutarate 5-dioxygenase 3, and splicing factor 3b subunit 4 as early-stage cancer decision markers and drivers of hepatocellular carcinoma,” Hepatology, vol. 67, no. 4, pp. 1360–1377, 2018.

[17] T. Dong, S. Lu, X. Li, J. Yang, and Y. Liu, “Genetic association between ankylosing spondylitis and major depressive disorders: Shared pathways, protein networks and the key gene,” Medicine, vol. 102, no. 24, p. e33985, 2023.

[18] B. Soudan, S. Abbas, A. Kubba, O. Abu Waraga, M. Abu Talib, and Q. Nasir, “Scalability and performance evaluation of federated learning frameworks: a comparative analysis,” International Journal of Machine Learning and Cybernetics, vol. 16, no. 5, pp. 3329–3343, 2025.

[19] K. Han, V. S. Sheng, Y. Song, Y. Liu, C. Qiu, S. Ma, and Z. Liu, “Deep semi-supervised learning for medical image segmentation: A review,” Expert Systems with Applications, vol. 245, p. 123052, 2024.

[20] A. Alsalama, A. Kubba, G. Jamjoum, and Z. Al Aghbari, “Classification based on association rules algorithm for breast cancer,” in 2024 Advances in Science and Engineering Technology International Conferences (ASET). IEEE, 2024, pp. 1–6.