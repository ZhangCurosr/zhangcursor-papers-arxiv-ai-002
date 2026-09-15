# Potential of Artificial Intelligence Algorithms for Identification of Relevant Diagnostic and Prognostic Biomarkers of Early-Stage Liver Cancer

Ali Bou Nassif\*,<sup>#</sup>   
Department of Computer Engineering, College of Computing and Informatics   
University of Sharjah, Sharjah, United Arab Emirates,   
Email: anassif@sharjah.ac.ae   
Darko Castven<sup>#</sup>   
Department of Medicine I, University Medical Center Schleswig-Holstein,   
University of Lubeck, Lübeck, Germany   
Email: Darko.Castven@uksh.de   
Manar Abu Talib   
Department of Computer Science, College of Computing and Informatics   
University of Sharjah, Sharjah, United Arab Emirates   
mtalib@sharjah.ac.ae   
Jibran Sualeh Muhammad   
Department of Biomedical Sciences, College of Medicine and Health   
University of Birmingham, Birmingham, United Kingdom   
Email: j.sualehmuhammad@bham.ac.uk   
Ahmed Ammar Kubba   
Department of Computer Science, College of Computing and Informatics   
University of Sharjah, Sharjah, United Arab Emirates   
Email: u23103280@sharjah.ac.ae   
Jens Marquardt<sup>^</sup>   
Department of Medicine I, University Medical Center Schleswig-Holstein,   
University of Lubeck, Lübeck, Germany   
Email: Jens.Marquardt@uksh.de   
Abdalla Sayed Ali   
Department of Computer Science, College of Computing and Informatics   
University of Sharjah, Sharjah, United Arab Emirates   
Email: U22103623@sharjah.ac.ae

\*Corresponding Author #,^: Authors contributed equally to this work

## Abstract

This study explores the use of deep learning and explainable artificial intelligence to diagnose hepatocellular carcinoma (HCC) and define effective biomarkers across five different stages of disease developmental using a transcriptomic biomarker HCC dataset constructed via semisupervised learning from three source datasets. Several deep learning experiments were conducted with different feature extraction techniques and gene sets to identify the most effective features for training high-accuracy models with minimal loss. The best-performing model, using 15 selected genes with the SelectKBest algorithm, achieved 90.74% accuracy, while the model with the lowest recorded loss of 0.3187 was obtained using 20 selected genes. To address the issue of class imbalance in the dataset, a weighted training approach was conducted, and for model transparency and interpretability a SHAP-based XAI analysis provided insights into the model’s decision-making, consistently finding DNAJB14 as the most influential gene. Functional validation in this study has provided compelling evidence that DNAJB14 plays an important role in the adverse properties of HCC and that its inhibition effectively reverses tumour cell migration, invasion, colony and sphere formation. The main limitation of this study is the dataset’s class imbalance, and while weighted training helped mitigate this, further research and additional data are needed to guarantee model generalizability. Future studies should also explore the influence of genetic variations, environmental factors, and clinical differences on model performance across diverse populations.

Keywords: Liver Cancer Prediction; Hepatocellular Carcinoma; Deep Learning, Explainable AI

## 1. Introduction and Related Work

In the past decade, artificial intelligence and its subsets (deep learning and machine learning) have emerged as revolutionary techniques in addressing the challenges of hepatocellular carcinoma (HCC) by leveraging vast amounts of clinical and diagnostic data, with a high number of studies and researchers exploring the potential of using these techniques for a wide range of medical applications in the context of diagnosis and prognosis of liver cancer patients [1][2][3][4] While data mining and machine learning techniques like association classification have been previously used for medical applications like breast cancer classification [5], the usage of deep learning methods can provide more reliable and accurate models [6], in addition to enabling the usage of more training data through privacy-preservation mechanisms such as federated learning in the context of hospitals and medical data [7][8].

Han et al. [9] have tried to improve early detection of HCC by combining dynamic network biomarker (DNB) analysis with graph convolutional neural networks (GCN) and with the use of multi-omics, which are used to study HCC development in mice using transcriptomic data and gene expression levels, using a five-year relative survival rate of HCC patients dataset from the NCBI with addition to RNA-seq data uploaded to the NCBI database (multi-omics gene expressions data) [10]. The DNB model identified the critical transition point at 7 weeks of age, achieving a 100% accuracy in classifying healthy and cancerous mice while also accurately predicting the health status of newly introduced mice. However, our approach was different by using feature selection to detect the biomarkers for detection of HCC.

Purba et al. [11] used Deep Neural Networks to classify cancer, with training using MicroRNA data analysis. Three types of data normalization (Min-Max, Sigmoid, Softmax) and three activation functions (ReLU, Sigmoid, TanH) were compared. The best accuracy (98.33%) out of four experiments (with no feature selection) in classifying HCC with MicroRNA data was achieved using Min-Max data normalization, ReLU activation function, and batch normalization with specific parameters. The microRNA dataset was obtained from the GDC Data Portal of the National Cancer Institute [12]. The data used in this study consisted of 600 data divided into 300 data on HCC class and 300 data on normal class with a total of 328 MicroRNA features.

Zhang et al. [13] aimed to create a patient similarity network using three types of HCC omics data and introduced a novel diagnosis method. This method combined similarity network fusion, denoising autoencoder, and dense graph convolutional neural network to use patient similarity networks and multi-omics data (DenseGCN). Comparisons with other machine learning methods on the TCGA-LIHC dataset demonstrated that the proposed approach outperforms them in all metrics. The proposed method achieved an accuracy of up to 0.985, using Liver Hepatocellular Carcinoma (LIHC) omics dataset. Like our approach, they have used feature selection techniques, but they used a denoising autoencoder.

Ahn et al. [14] review recent studies that have applied deep learning in risk prediction, diagnosis, prognosis assessment, and treatment planning for HCC patients. It also discussed different models of deep learning like ResNet, Auto Encoders, and CNNs, using several datasets like the TCGA, VHA, and Taiwanese NHIRD. Ioannou et al. [15] used an RNN to predict HCC development within three years, getting superior performance with an AUC of 0.759. Phan et al. [16] applied a CNN to predict HCC among patients with viral hepatitis, achieving an AUC of 0.886 and 0.980 accuracy. Nam et al. [17] constructed DNNs for predicting HCC incidence and recurrence after liver transplantation, outperforming traditional models with higher C-index values. As shown, the highest accuracy was 0.980. According to the study, there were certain limitations; Serum AFP, a predictive biomarker for hepatocellular carcinoma [18][19][20], has limited sensitivity for early detection. Multi-omics research is needed for better biomarkers. Deep learning algorithms generalizability and lack of public availability hinder external validation for reliability and effectiveness.

Das et al. [21] aimed to build a new technique for detecting cancer by a new system called watershed Gaussian based deep learning using the GMM algorithm and built a DNN model using a CAD model and WGDL technique for classification with a 99.38% classification accuracy and a validation loss of 0.062. They used CT images as a model parameter and a dataset consisting of 225 CT images of HCC patients from IMS and SUM hospitals in India, from 75 different HCC and metastatic carcinoma patients. A level co-occurrence matrix (GLCM) method was used for classification and feature selection. One limitation is they developed a DNN using Keras that uses Tenserflow for their feature selection to classify the disease accurately, it required to balance the data set otherwise the classifier will produce an error function. A GE medical System CT scan machine for recording the images was used as a clinical test.

Tao et al. [22] demonstrated the potential of recognition for early stages cancer detection using a whole-genome (SCNA) profiling method with also the use of machine learning by also taking advantage of a method called "Liquid biopsy" as the detection of early stages cancer are more challenging as the tumour cells that carry the cancer genomic aberrations tend to release much less DNA into the blood, which would make it impossible to diagnose because we will need an unrealistic amount of blood from the patient, and that where comes the use of the machine learning algorithms used in this research such that wRF-driver algorithm can take advantage of the SCNA profiles and also some external provided medical data (TCGA – ICGC). The ML model was built using a discovery cohort of 209 patients, with an overall area under curve (a commonly used metric in machine learning to evaluate the performance of a binary classification model) of 0.893 / 0.874 for early-stage, and 0.933 for patients with only Stage 0 - A HCC over the two cohorts, The accuracy in validation cohort 2 is noticeably lower than validation in cohort 1, likely due to more patients with very early stage tumour in validation cohort 2.

Książek et al [23] conducted a new approach for detecting HCC patients using 10 ML algorithms and a dataset of 165 HCC survival patients from CHUH in Portugal and results were noted by RF algorithm F1-Scores and did 2 different experiments. RF algorithm conducted the highest F1 score of 1 out of 1 and for the linear SVM highest score was 0.7804 during testing. One limitation for this design was the lack of features, which we aimed to overcome in our project. Different experiments reviewed by Chen et al. [24] have shown the role of biomarkers in HCC detection using RNAs and identifying tumour antigens to activate anti-tumour immune responses, while also using blood biopsy as a clinical test. Petinrin et al [25] review existing studies on classical machine learning and deep learning models in metastatic cancer, it demonstrates the use of deep learning for image data extraction, noting that feature extraction capabilities of deep learning aid the development of better and more accurate models, hence why we use feature selection in this experiment.

Jiang et al. [26] demonstrates the importance of histopathology image analysis in HCC diagnosis and prognosis, and how deep learning has become a leading technique in this domain for classifying and localizing HCC and tumour cells, it surveys over 50 studies in this field, the main methodologies used were supervised learning, weak supervised learning and multi-scale and multi-resolution networks. Amongst the studies overviewed there were different public datasets used like the TCGA-LIHC tissue slide images and related clinical information dataset, PAIP 2019 dataset for segmentation tasks from MICCAI 2019 Grand Challenge, and the KMC dataset providing diverse histopathological images for different HCC subtypes. Limitations were mostly data and image related, like colour inconsistencies and large image sizes, and many DL models are considered "black boxes," meaning their decision-making processes are not transparent or easily interpretable by clinicians. For performance metrics some accuracies were reported as high as 0.98 in some studies.

Ngan et al. [27] showed a review of how machine learning is used to analyse mass spectrometry-based metabolomics data for early HCC diagnosis. This technique helps to capture metabolic changes associated with cancer and enables screening out physiological biomarkers of cancer risk and clinical biomarkers of cancer which can help in identifying it in early stages. Methodologies used were mass spectrometry, machine learning with different models like Random Forests (RF), Principal Component Analysis (PCA), Support Vector Machines (SVM), Partial Least Squares-Discriminant Analysis (PLS-DA) and Neural Networks (NN), with the help of a feature selection technique called correlation-based feature selection (CFS) with logistic regression and Lasso regression. Certain limitations were noted like the exponentially increased data volume and complexity of the mass spectrometry data and the lack of clinical validation. For liver cancer detection, supervised ML models such as linear SVM and logistic regression demonstrated high accuracy (>85%) for different studies on tissue/serum.

Matboli et al. [28] aimed to leverage machine learning techniques to diagnose early HCC using RNA signatures with laboratory parameters. It uses a comprehensive dataset which contains RNA expression levels and clinical parameters from 267 subjects including 102 malignant HCC, 67 of which are benign liver conditions and 98 healthy controls, in addition to access of the GSE14520 dataset. The RNA signatures include mRNAs (RAB11A, STAT1, ATG12), miRNAs (miR-1262, miR-1298, miR-106b-3p), and lncRNAs (RP11-513I15.6, WRAP53) select on their known involvement towards HCC pathogenesis (in key biological pathways). Different machine learning models were evaluated like KNN, RF, SVM, LGBM, DNN trained and tested using a 70/30 dataset split. LGBM was the best performing model, achieving an accuracy of 98.75% superior to other classifiers. The study also employed feature selection techniques. The model also included 22 features (age, sex, smoking, cirrhosis, non-cirrhosis, albumin, ALT, AST bilirubin (total and direct), INR, AFP, HBV Ag, HCV Abs, RQmiR-1298, RQmiR-1262, RQmiR-106b-3p, RQmRNARAB11A, and RQSTAT1, RQmRNAATG12, RQLnc-WRAP53, RQLncRNA- RP11-513I15.6). Future work includes further validation and testing of the model, exploring the model's potential in also predicting disease progression and response to treatment. Limitations were small and unreliable data sizes.

Shen et al. [29] aimed to construct risk prognosis model by integrating multiple key genes related to aging and to explore the relationship between risk score and immune cell and tumour microenvironment by combining bulk and single-cell sequencing using a NMF machine learning model where clinical data and survival status acts as model parameters, using a dataset from TCGA (374 cancer samples and 50 normal samples of clinical data). Without feature selection, for the TCGA training set, the model achieved 0.797, 0.749, 0.740 for 1, 2, 3 years amongst several experiments.

Gholizadeh et al. [30] aimed to identify key mRNAs which can serve as biomarkers for diagnosing HCC, it uses a non-fusion integrative multi-platform meta-analysis to integrate gene expression data from different platforms, then applies machine learning methods to develop diagnostic and prognostic models. Data is used from many gene expression platforms like Illumina and Affymetrix datasets, with a total of 939 samples, comprising 493 tumour and 446 non-tumour samples, were used. The datasets include GSE57957, GSE39791, GSE36376, GSE84005, GSE12941, GSE64041, GSE45267, and GSE84402, sourced from the Gene Expression Omnibus (GEO) and The Cancer Genome Atlas (TCGA). The study utilizes the Linear Models for Microarray Data (LIMMA) R package to identify differentially expressed genes (DEGs). A Bayesian approach is employed to assess statistical significance, with a pvalue threshold of 0.05 and a fold-change cutoff of 1. One limitation noted could be potential biases due to batch effects despite correction efforts, and for future work they suggested enhancing the robustness of the identified biomarkers and exploring additional molecular signatures to improve diagnostic and prognostic accuracy.

Table 1 Overview of the existing literature in terms of datasets, objectives, methodology, and results compared to our work
<table><tr><td>Ref</td><td>Datatype</td><td>Task</td><td>Dataset</td><td>Feature selectio n</td><td>Performance</td><td>Clinical test</td></tr><tr><td>[31]</td><td>Multi-omics gene expressions</td><td>HCC Detection</td><td>NCBI</td><td>x</td><td>100% accuracy</td><td>x</td></tr><tr><td>[32]</td><td>Multi-omics gene expressions</td><td>HCC Detection</td><td>NCBI</td><td>x</td><td>100% accuracy</td><td>x</td></tr><tr><td>[33]</td><td>DNA sequences</td><td>HCC Detection</td><td>NCBI</td><td>x</td><td>CNN 80.36% accuracy vGG16 98.86% accuracy fine-tuned vGG16 model: 100% accuracy</td><td>Model: model: x</td></tr><tr><td>[34]</td><td>microRNA expressions</td><td>HCC Detection</td><td>GDC Data Portal NCI 2018</td><td>x</td><td>Best accuracy 98.33%</td><td>is x</td></tr><tr><td>[35]</td><td>LIHC omics data</td><td>HCC Detection</td><td>LIHC omics datasets from TCGA</td><td>Denoisi ng autoenc oder</td><td>0.9857 Accuracy</td><td>x</td></tr><tr><td>[36]</td><td>CT images</td><td>HCC Detection</td><td>Datasets from IMS and SUM hospital (India)</td><td>GLCM</td><td>99.38% classification accuracy</td><td>GE medical system scan machines</td></tr><tr><td>[37]</td><td>SCNA genome profiles &amp; CNA plasmas 23 quantitative</td><td>HCC Detection</td><td>Discovery cohort of 209 HBV patient's genome- wide SCNA profiles in early stages</td><td>Gini- Impurit y Index</td><td>AUC of 0.920 for cohort 1, AUC of 0.812 for cohort 2</td><td>AFP Screening</td></tr><tr><td>[38]</td><td>features and 26 qualitative features of HCC patients' medical information histopathology</td><td>HCC Detection</td><td>CHUC</td><td>Genetic algorith m with Cross validati on</td><td>f1 score of 1 out of 1 and for the linear SVM highest score was 0.7804 during testing</td><td>x</td></tr><tr><td>[2]</td><td>images / tissue slides images and related clinical information</td><td>HCC Diagnosis and treatment</td><td>TCGA-LIHC images, PAIP KMC</td><td>CNN tissue 2019, models was used</td><td>Varies, reaching as high as 0.98 in some experiments For the liver cancer</td><td>x</td></tr><tr><td>[39]</td><td>tissue, liver serum samples and other clinical info</td><td>Detection of different types of cancer</td><td>Private dataset of liver tissue and samplesand including data from studies</td><td>CFS, Logisiti c serum Regress ion, another Lasso Regress ion</td><td>experiments, good accuracies reported, 80% and above for tissue and serum however there is no single denoted.</td><td>are x studies, accuracy</td></tr><tr><td>[40]</td><td>RNA expression levels and clinical information,Bi ochemical markers</td><td>HCC Diagnosis</td><td>A. private dataset with clinical and expression data from 267 subjects, including 102 with  malignant HCC, 67 with benign liver conditions, and 98 healthy Additionally, GSE14520dataset was</td><td>RNA x controls. the</td><td>LGBM was the best perforing model, achieving highest accuracy of 98.75%</td><td>the x</td></tr><tr><td>[41]</td><td>374 cancer samples and 50 normal samples clinical data.</td><td>predict prognosis and immunotherap y efficacy</td><td>used TCGA</td><td>x</td><td>For TCGA training set: AUC values of 0.797,0.749,0.740 for 1,2,3 years.</td><td>x</td></tr><tr><td>[42] gene</td><td>expressions</td><td>identify key mRNAs in which can server as biomarkers for diagnosing HCC</td><td>GSE57957, GSE39791, GSE36376, GSE84005, GSE12941, GSE64041, GSE45267, GSE84402, TCGA and GEO</td><td>x</td><td>p-value threshold of 0.05 and a fold- change cutoff of 1</td><td>Serum Biomarker Analysis, Liver Function Tests, Reverse Transcriptio</td></tr><tr><td colspan="6"></td><td>n and qRT- PCR, Diagnostic Imaging and Pathology</td></tr><tr><td>Our Work gene</td><td>Multi-Stage HCC detection and Multi-omics identification of expressions biomarkers for early-stage HCĆ</td><td>Private Lubeck data, relevant GSE89377 , TCGA</td><td>SelectK Best, Ant colony, Genetic Algorith m, Particle Swam Optimiz ation</td><td>90.74% accuracy 0.3187 loss</td><td>best and</td><td>Functional in vitro target evaluation</td></tr></table>

As observed in this literature review, studies tend to explore different biomarkers, including mRNAs and miRNAs, to differentiate between healthy and cancerous patient tissue and different HCC stages. However, many drawbacks to the reviewed papers can be noted, which include limited dataset sizes, the need for better feature selection approaches, and a lack of clinical validation in addition to the ‘black box’ nature of deep learning making interpretability difficult and raising further issues with applying machine learning to medical areas. This study further explores the potential of using genomic /RNA expression data for training deep learning models for multi-stage HCC classification. Our contributions to the existing research are as follows:

Our deep learning model can classify patient tissue samples into hepatocarcinogenesis including non-malignant and malignant tissue, whereas machine learning models used in other studies often have less detailed or broader classification capabilities that predict a smaller number of HCC classes.

• The HCC dataset used for training the deep learning model was constructed from one private dataset and two public datasets using semi-supervised learning and contains five categories, with 770 patient samples represented by 11,150 gene expression features.

The final accuracy of the deep learning model after performing the necessary tests was recorded at 59% on the initial set of genes, 61% on the second set of genes, and 86% on the third set of genes, each extracted from one of the three datasets during the feature extraction phase. Additionally, the model performed best with the SelectKBest algorithm with a classification accuracy of 90.74% at 15 selected genes.

Our study implements an explainable AI approach to contribute to overcoming the black box issue noted by previous studies using AI models for medical applications, increasing the reliability and transparency of our model as a result, in addition to bringing attention to a gene that can potentially have links to HCC that have not yet been established in the literature (DNAJB14).

• Our study performs in vitro functional validation on the DNAJB14 gene to validate links to HCC.

This paper is organized as follows: The first section introduces the research topic (deep learning and HCC), along with background information and a review of the related literature, noting existing limitations in the literature. The second section outlines the research methodology, detailing dataset preprocessing, feature extraction, the experimental setup, and the explainable AI approach. The third section presents and describes the results of the experiments. The last section concludes the paper and presents the study’s limitations, potential future work.

## 2. Research Methodology

This section describes the methodology of the project in detail, as observed in Figure 1, which consists of data pre-processing on the HCC dataset which consists of three source datasets and feature extraction on the Lubeck, TCGA, and HCC dataset. Afterwards, the deep learning model is trained on the HCC dataset using the extracted gene sets as input features, and then the models are evaluated accordingly. Finally, the explainable AI approach is implemented, and the results are interpreted in the last section.

![](images/731eb35d1f4cfa232fee7cee38d4354870ee7f92b404be7510cf0900323a8755.jpg)  
Figure 1 Paper methodology summary

## 2.1 Dataset Preprocessing and Feature Extraction

The training dataset is a specialized hepatocellular carcinoma dataset containing 770 patient samples labelled in five categories and represented by genomic expression data features. Each samples contains 11,150 corresponding genomic expression levels, which represent the input training features for the model to predict the developmental HCC stage of the target patient. This dataset was constructed using semi-supervised learning from three source datasets: the private “Lubeck” dataset from medical domain experts, and the two publicly available datasets, TCGA [43] and GSE89377 [44]. As can be observed in Figure 2, which visualizes the distribution of the dataset’s classes, the most common samples in the dataset belong to the ‘phcc’ and ‘ehcc’ classes, whereas the ‘sl’ class has the fewest samples at less than 50 instances. This class imbalance in the dataset is accounted for in the experimental setup section by utilizing different weights for the classes during training. The dataset's patient tissue samples are classified into five categories:

1. Surrounding Liver (sl): which is absent of any HCC gene expression.

2. Early Hepatocellular Carcinoma (ehcc): representing an early-stage HCC diagnosis.

3. Progressed Hepatocellular Carcinoma (phcc): indicating an advanced stage of HCC.

4. Low-Grade Dysplastic Nodules (lgdn): which are associated with a lower risk of HCC development.

5. High-Grade Dysplastic Nodules (hgdn): which are linked to a higher risk of HCC progression.

![](images/6a1a8f35062dfa5b580c05b9c0cb99408181e6908fbf328f034f2d36d1613bd5.jpg)  
Figure 2 Visual representation of the dataset class distribution.

The deep learning model was trained in three separate experiments using three different sets of genes as observed in Table 1, which were extracted using feature selection that was performed on the Lubeck and TCGA source datasets, in addition to the HCC dataset which was constructed using semi-supervised learning. Conducting these three experiments enables us to compare the performance of the model on each selected gene set for diagnosis, minimizing possible bias in the results and providing us with more insights into the effectiveness of using different genomic markers as input features, improving the model’s robustness and its applicability in clinical settings in addition to providing future researchers with insights into possibly significant genetic biomarkers.

Table 2 Feature extraction gene sets used in the model's training
<table><tr><td rowspan=1 colspan=1>Gene Set</td><td rowspan=1 colspan=1>Selected Genes</td><td rowspan=1 colspan=1>Source Dataset</td></tr><tr><td rowspan=1 colspan=1>Set 1</td><td rowspan=1 colspan=1>[&#x27;SELENBP1&#x27;,   &#x27;MASP1&#x27;,&#x27;PPP1R14B&#x27;,        &#x27;TAF2&#x27;,&#x27;OTUB1&#x27;]</td><td rowspan=1 colspan=1>Lubeck</td></tr><tr><td rowspan=1 colspan=1>Set 2</td><td rowspan=1 colspan=1>[&#x27;HGF&#x27;,          &#x27;PDGFRA&#x27;,&#x27;HIF1AN&#x27;, &#x27;SAE1&#x27;]</td><td rowspan=1 colspan=1>TCGA</td></tr><tr><td rowspan=1 colspan=1>Set 3</td><td rowspan=1 colspan=1>[&#x27;IFNA5&#x27;,               &#x27;IVD&#x27;,&#x27;DNAJB14&#x27;,        &#x27;ITGB8&#x27;,&#x27;PADI4&#x27;]</td><td rowspan=1 colspan=1>HCC</td></tr></table>

## 2.2 Experimental Setup

The deep learning model used in this project is a feedforward neural network which was built using the Keras python API for machine learning. The model, as visualized in Figure 3, consists of an initial input layer whose shape matches that of the training data, to enable feeding the training data into the neural network. Afterwards, there are two hidden dense layers which serve their own purposes. The first dense layer consists of 64 neurons and a Rectified Linear Unit (ReLU) activation function which adds non-linearity to the model, an important characteristic for learning complex data patterns in machine learning.

After the first dense layer, a dropout layer is introduced with a 20% dropout rate. This means that this layer randomly zeroes 20% of the input data during each training round, which is important for preventing overfitting as it ensures that the DL model does not overtly rely on any specific set of neurons. The second dense layer is identical to the first alongside its dropout layer except for the fact that it uses 32 neurons instead. The final layer in the model is a dense layer that uses five neurons, equivalent to the five classes found in the HCC dataset. This layer uses a linear activation function, which is needed for the loss function that follows it.

![](images/9c3c85e502122d42ef54cf4996f2de0638b5bebae98f0039f38eee3d407a029c.jpg)  
Figure 3 Deep learning model visualization

The deep learning model was compiled using the Adam optimizer and the sparse categorical cross entropy loss function, which is the appropriate loss function for multi-class classification machine learning tasks in which the output layer doesn’t use a softmax activation function. Additionally, the model utilized a learning rate scheduler which initially maintains a high learning so that the model can learn from the training data quickly. After 40 epochs, the LR scheduler reduces the learning rate and then further reduces it after 80 training epochs. This enables refining the learning process and allowing the model to make very precise updates throughout the training process, leading to overall better model stability during training. The model also utilized an early stopping mechanism which stops the training process when the model is no longer improving on the validation set. This approach can be useful for preventing overfitting and avoiding a lengthy training process with little to no benefits.

The training set consisted of 35% of the original dataset, whereas the test and validation splits were 48.75% and 16.25% of the dataset, respectively. The model’s training was conducted on a workstation using the Windows 10 operating system with 32 gigabytes of RAM and an Nvidia RTX 4000 GPU. The CPU of the workstation, which was utilized during the model’s training, is Intel® Xeon® W-2102 with four cores and a base speed of 2.90 GHz.

Because of the relative imbalance of classes in the training dataset, which can be observed in Figure 2, we have utilized a weighted training approach to minimize training bias in the DL model. The weighted training approach is enabled by fitting the model with a dictionary containing a different weight for each class in the dataset, with a total of five weights for five classes. These weights are calculated using a special computing function from the scikit-learn library which produces the weights based on the relative frequency of each class. As can be observed in Table 3, the assigned weight for each class is inversely proportional to the relative frequency of that class, i.e. the more frequently a particular class occurs in the dataset samples, the smaller the assigned weight to that class will be to indicate to the model that it should prioritize the least frequent minority classes over the majority classes during the training process.

Table 3 Assigned weights for each class in the dataset
<table><tr><td rowspan=1 colspan=4>Class Name            Frequency              Proportion              Assigned Weight</td></tr><tr><td rowspan=1 colspan=1>ehcc</td><td rowspan=1 colspan=1>305</td><td rowspan=1 colspan=1>39.61%</td><td rowspan=1 colspan=1>0.505</td></tr><tr><td rowspan=1 colspan=1>hgdn</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>4.935%</td><td rowspan=1 colspan=1>4.053</td></tr><tr><td rowspan=1 colspan=1>lgdn</td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1>9.091%</td><td rowspan=1 colspan=1>2.200</td></tr><tr><td rowspan=1 colspan=1>phcc</td><td rowspan=1 colspan=1>331</td><td rowspan=1 colspan=1>42.99%</td><td rowspan=1 colspan=1>0.465</td></tr><tr><td rowspan=1 colspan=1>sl</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>3.377%</td><td rowspan=1 colspan=1>5.923</td></tr></table>

## 2.3 Explainable AI Approach

As transparency is highly valued when it comes to the integration of AI in the medical field, an explainable AI approach is also explored for the purpose of producing a medically trustworthy and transparent solution. Three explainable machine learning models, consisting of two Random Forest classifiers and one XGBoost classifier with each using different optimization algorithms, are trained on the HCC dataset and then explained afterwards using the SHAP (SHapley Additive exPlanations) library, which helps explain the predictions of the models and their contributing features using a game theory approach [45][46]. The explainable AI approach of our study is summarized in Figure 4.

![](images/bacca5315b0dc69e013fd7153671349bd36fcc12cb689bc395b31ddd45f2a68c.jpg)  
Figure 4 Explainable AI Methodology

The first two models consist of a random forest classifier, with the first model being trained using the Ant Colony optimization algorithm for feature selection, and the second model using the Genetic Algorithm. The third model consists of an XGBoost classifier that was trained using the Particle Swarm optimization algorithm. SHAP is used to explain all three models, and the results are cross verified between the models to minimize possible model bias in the results by not relying on a single model or optimization algorithm.

The first two models consist of a Random Forest classifier, with the first model being trained using the Ant Colony optimization algorithm for feature selection, and the second model using the Genetic Algorithm. Random Forest is a type of ensemble learning technique which builds multiple decision trees during the training process and then averages their total outputs during the prediction process [47]. It is based on a technique called Bagging (or Bootstrap Aggregating), in which different subsets of the dataset are randomly sampled with replacement to train individual decision trees [48]. This process reduces variance and enhances model stability, making it more robust against overfitting, especially when working with highdimensional medical datasets like our HCC dataset.

The third model consists of an XGBoost classifier that was trained using the Particle Swarm optimization algorithm. XGBoost (Extreme Gradient Boosting) is a highly efficient gradientboosting model which builds decision trees sequentially, with each new tree improving the accuracy of the trees preceding it [49]. This technique is based on Boosting, another ensemble learning approach, which reduces model bias by iteratively adjusting the model weights of a weak learner based on misclassified samples in each iteration of the model, leading to a strong final predictive model [50].

Ant Colony Optimization is inspired by the foraging behaviour of ants, where they deposit pheromones to mark the best paths to food sources. ACO simulates this behaviour by iteratively building feature subsets and prioritizing the most promising features based on a chosen performance metric, improving the classification performance of Random Forest while maintaining its interpretability [51].

The Genetic Algorithm is an optimization algorithm inspired by natural selection and genetics. It keeps a population of potential feature subsets, which evolve over multiple iterations using selection, crossover, and mutation techniques. The best feature sets are chosen based on a predefined fitness function to maximize classification performance [52].

Particle Swarm Optimization is an algorithm inspired by collective bird movement, in which each potential solution moves based on its own experience and the best solutions found by other particles. When used for of feature selection for XGBoost, PSO adjusts the weights of different features iteratively, improving the model's ability to generalize with each iteration and eventually leading to an optimized feature subset with minimal redundancy [53].

SHAP is used to explain all three models, and the results are cross verified between the models to minimize possible model bias in the results by not relying on a single model or optimization algorithm. By integrating both Bagging-based (Random Forest) and Boosting-based (XGBoost) models with several iterative optimization algorithms, this approach ensures a comprehensive analysis that balances accuracy, interpretability, and robustness in medical AI applications.

## 2.4 Clinical Validation Methodology

## 2.4.1 Survival analysis using public dataset

Gene Expression Profiling Interactive Analysis (GEPIA2) online tool (http://gepia2.cancerpku.cn/#index) was used to evaluate the influence of DNAJB14 on overall survival in HCC patients retrieved from the publicly available knowledge of The Cancer Genome Atlas Program (TCGA) database. Cutoff values for high and low expression were set to 80%.

## 2.4.2 Cell culture

The Huh7 cell line was obtained from Riken Cell Bank. Cells were grown in Dulbecco’s modified Eagle’s medium (DMEM), supplemented with 2 mM L‐glutamine, 1 unit/mL penicillin/streptomycin, and 5% fetal bovine serum (FBS). Cell line was maintained at 37°C and 5% CO2, routinely tested and authenticated by STR testing.

## 2.4.3 Nucleic acid extraction

Total RNA was extracted using Monarch Total RNA Miniprep Kit (NEW ENGLAND Biolabs) according to the manufacturer’s instructions. RNA quality and purity was determined by Nanodrop One spectrophotometer (Thermo Fisher Scientific).

## 2.4.4 Quantitative real-time polymerase chain reaction

Two-step reverse-transcription quantitative polymerase chain reaction using iScript cDNA Synthesis kit (Bio-Rad Laboratories), SYBR Green Master-Mix (Bio-Rad Laboratories) and CFX Connect (Bio-Rad Laboratories) was performed. Oligonucleotide primers were designed using Primer3 v.0.4.0 (https://bioinfo.ut.ee/primer3-0.4.0/). Forward: 5' TTC ATC ACC AGC ATC TTT CG 3' Reverse: 5' CAG GTA GTG GCG ATC AAA GC 3' (Eurofins).

## 2.4.5 Western blotting

Cell lysates were prepared using M-PER tissue extraction buffer (Thermo Fisher Scientific) containing Halt Protease & Phosphatase Inhibitor Cocktail (Thermo Fisher Scientific). 20 µg of protein lysate were separated by SDS-PAGE and transferred onto nitrocellulose membrane (GE Healthcare Life Sciences). Membranes were probed with the DNAJB14 antibody (rabbit, polyclonal, proteintech #16501-1-AP).

## 2.4.6 siRNA-mediated knockdown

For transfection, cells were seeded at low confluency (30-50%) in 6-well plates. After 24 h 50, 100 and 150 pmol siRNA (non-sense control and siRNAs targeting DNAJB14, ambion) was introduced by using transfection reagent Lipofectamine 2000 (Thermo Fisher Scientific) following the manufacturer’s instructions. Subsequent functional assays were conducted 72 h after transfection.

## 2.4.7 Viability assay

Cell viability was determined using a colorimetric assay (WST-1) following the manufacturer’s protocol (Roche). In total, $5 \times 1 0 ^ { 3 }$ cells were plated on 96‐well plates; after overnight incubation, cells were transfected with increasing concentrations of DNAJB14 siRNA reflecting concentrations of 50, 100 and 150 pmol siRNA from 6-well plate experiments and the viability was measured after 72 hours.

## 2.4.8 Colony and sphere formation assays

Cells were treated for 72 h with siRNA-mediated knockdown of DNAJB14. 1x10<sup>3</sup> cells were plated on 6-well plates for colony formation (CFU) and 1x10<sup>3</sup> cells were plated on 48-well plates for sphere formation (SFU) assay into a semisolid soft agar (Carl Roth). After 14 days colony and sphere formation capacities were determined and represented as number of colonies/spheres for each treatment group in comparison to the control group.

## 2.4.9 Migration assay

After siRNA-mediated knockdown of DNAJB14, migration was evaluated with Culture-Inserts in a μ-Dish 35mm (Ibidi) and compared with control cells. To distinguish between migration and proliferation, cell proliferation was suppressed by using the proliferation inhibitor Cycloheximide (2 µM, Carl Roth). In total 5x10<sup>4</sup> cells were seeded on Culture-Inserts in DMEM (10% FBS). The cells were allowed to attach for 24 h before the culture inserts were removed. Cycloheximide was administered 6 h prior to treatment. After incubation, the culture inserts were removed and the plate were filled with 2 ml culture medium containing and the cells were monitored immediately after gap formation and images were taken at 0, 24, 48, 72, 96 and 120 h using a phase contrast microscope.

## 2.4.10 Invasion assay

Cell invasion was determined using the xCELLigence DP system (Roche, Mannheim, Germany). For the invasion assay, a total of 80,000 Huh7 cells with siRNA-mediated knockdown of DNAJB14 or control cells were seeded on CIM-Plates 16. To assemble the CIM-Plates 16, the bottom of the upper compartment was coated with 30 µl of collagen I (1 mg/ml) and IV (50 µg/ml) and allowed to dry. Cells were applied to the upper chamber of the CIM-Plate, which was then placed in the RTCA DP instrument and connected to the computer. The real-time cell invasion assay was monitored for the next 72 h with data acquisition at 15' intervals. Monitoring and analysis were performed using xCELLigence RTCA software (version 1.2.1., Roche, Germany).

## 2.4.11 Statistical analyses

Statistical analyses were performed using one-way ANOVA. p≤0.05 were considered statistically significant. All results were presented as means ± standard deviation (SD) from at least three independent experiments.

## 3. Results and Discussion

This section details the results of the deep learning model and its performance, in addition to the explainable AI approach and the most significant genes based on the conducted SHAP analysis. With informed advice and insights from medical domain experts, a thorough discussion and interpretation of the results and findings is also presented.

## 3.1 Deep Learning Model Performance

The deep learning model was evaluated on the test split, which represents 35% of the original dataset. As the training process consisted of three separate experiments, in which the model was trained on each selected genes set, the model was evaluated after each experiment to determine the set of genes which produced the best-performing DL model.

<table><tr><td rowspan=1 colspan=3>Training Gene Set               Accuracy                           Loss</td></tr><tr><td rowspan=1 colspan=1>Set 1</td><td rowspan=1 colspan=1>0.5926</td><td rowspan=1 colspan=1>0.9591</td></tr><tr><td rowspan=1 colspan=1>Set 2</td><td rowspan=1 colspan=1>0.6148</td><td rowspan=1 colspan=1>0.9029</td></tr><tr><td rowspan=1 colspan=1>Set 3</td><td rowspan=1 colspan=1>0.8667</td><td rowspan=1 colspan=1>0.3531</td></tr></table>

As can be observed in Figure 5, the results for Set 1 show an accuracy of 59.26% and a loss of 0.9591. This indicates that the model correctly classified 59.26% of the training samples after being trained using this specific gene set. The high loss, especially relative to the loss value in the other experiments, shows that the model is struggling to fit the training data correctly due to high errors in its predictions. The relatively low accuracy and high loss values imply that the model is underfitting, which means that it has failed to learn the data patterns sufficiently which can be due to several reasons, including a poor choice of selected genes, insufficient model layer complexity, or a lack of sufficient training data. As can be observed for the next experiments, the performance metrics improve with different selected gene sets, which strongly indicates that the first reason is the best explanation for the current results.

For Set 2, the model achieves a slightly higher accuracy of 61.48% and a slightly lower loss value of 0.9029 compared to Set 1. This improvement in both accuracy and loss values suggests that the model is performing better using this gene set for training. The decrease in loss means that the model's predictions are closer to the actual values than the previous model, although the accuracy value remains unsatisfactory at only 61.48%. This result points to a slightly better fit to the training data, but there is still room for improvement either in the model architecture, the training data, or the gene selection. As can be observed in the next experiment, improving the gene selection produces the biggest difference in terms of model performance.

Set 3 demonstrates the most significant improvement in performance relative to the previous experiments, with a much higher accuracy of 86.67% and a much lower loss of 0.3531. The high accuracy indicates that the model correctly classified most of the training samples, while the low loss means that the model’s predictions are very close to the actual values. This big improvement can be clearly attributed to the selected genes set containing more informative and useful features for training the model, allowing it to learn the underlying patterns in the data accurately and producing better accuracy and loss values as a result.

Table 4 SelectKBest Experiments testing different numbers of 'K' best genes selected.
<table><tr><td rowspan=1 colspan=3>Accuracy                           Loss‘K’ Best Genes Selected</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>87.41%</td><td rowspan=1 colspan=1>0.4372</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>90.37%</td><td rowspan=1 colspan=1>0.3729</td></tr><tr><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>90.74%</td><td rowspan=1 colspan=1>0.3625</td></tr><tr><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>90.37%</td><td rowspan=1 colspan=1>0.3187</td></tr><tr><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>88.89%</td><td rowspan=1 colspan=1>0.3638</td></tr></table>

Additional experiments were carried out on more gene sets which were obtained using the SelectKBest feature selection approach on the HCC dataset, which chooses the top K most relevant features based on statistical tests to reduce data dimensionality and noise [54]. SelectKBest has also been used in a medical context in the existing literature [55]. The experiments ranged from 5 best genes selected to the top 25 best genes. As can be observed in Table 3, the best model performance was recorded at 15 ‘K’ best genes selected, with the model performance in terms of accuracy and loss deteriorating with more genes than 15 in subsequent experiments, except for model loss marginally improving at 20 selected genes.

Training and Validation Loss  
![](images/969fec745149be3c9e7753fa2c351b916944ef387fa7a79a7f91bc1f205cdde4.jpg)  
Figure 6 Training and validation loss graph for the Set 3 training experiment

The training and validation loss throughout the training process of the Set 3 model can be observed in Figure 6, which shows a smooth progression of validation loss decreasing throughout the training epochs, while the training loss suffers from some fluctuations, it still follows an overall trend of decreasing as the training process continues. Figure 7 displays the training and validation accuracy throughout the training experiment. As can be observed, the training and validation accuracy closely follow the same trend of small increases with some fluctuation during the training process. Overall, both graphs indicate no underfitting or overfitting in the model during the training process and high-performance metrics by the end of the model’s training.

![](images/bd59ddc1b5906f57f7f933a7cebbe100f97cbcfb6ce807f259eb5ee9d37658bb.jpg)  
Figure 7 Training and validation accuracy graph for the Set 3 training experiment

The model’s performance across the three gene sets shows a progression from moderate to high accuracy and a corresponding decrease in loss from high to low. This change through the different gene sets proves the importance of the gene set's quality and characteristics in training an effective model. While Set 1 and Set 2 provide low to moderate results, Set 3 displays excellent performance in comparison, demonstrating that it contains the most informative and useful features for the model to learn from.

## 3.2 Explainable AI Findings

This section describes the results of the three explainable AI experiments and the interpretation of the resulting SHAP values for each trained model. The SHAP value plots, which can be observed in Figures 9, 10, and 11, represent the average impact of different genes on the output of our predictive models for HCC stages.

Figure 8 represents the SHAP plots for the PSO graph, whereas Figure 9 represents the genetic algorithm graph and Figure 10 shows the ant colony graph. The x-axis in each figure shows the mean absolute SHAP value, which represents the average contribution of each gene to the model's predictions, whereas the y-axis lists the genes used as features from the dataset. Higher SHAP values mean that the gene has a stronger impact on the model’s decision, and the coloured segments within each bar represent the impact of the gene on predicting different HCC development stages (or classes), such as the ehcc stage being associated with blue or phcc with purple.

![](images/1738ec8bbeff0a9d2a1f3bbfb302242bad756df072d61db43a101d368f3837e4.jpg)  
Figure 8 PSO Graph

![](images/205e76febf462d4c5b795d4aa923294518e930612fc3404b11b33322d76c25ad.jpg)  
Figure 9 Genetic Algorithm Graph

![](images/409dbdf9d87dc146f0c0537ed44b42efee8c76b0c2e9edc83db86157d7ee8bb4.jpg)  
Figure 10 Ant Colony Graph

DNAJB14 can be observed to be the most influential gene in all three models, with a consistently high SHAP value across different hepatocarcinogenesis stages. DNAJB14's contribution is primarily towards predicting the ehcc and phcc classes, suggesting its importance in distinguishing these stages. Other genes show a much more balanced influence across different stages, and their relative importance differs between models, because different optimization approaches (Genetic Algorithm, PSO, and Ant Colony Optimization) can end up selecting different optimal features with varying levels of contribution. DNAJB14 appears as the top feature in all models, and ZNF415 appears in the graphs belonging to Figure 9 and Figure 10, whereas RIMBP2 and HGF appear in the graphs belonging to Figure 8 and Figure 10.

The third model in Figure 10 appears to produce significantly higher mean SHAP values compared to the two other models, as observed in its x-axis scale, suggesting that it might have identified genes with a stronger contribution to classification. The first two models (Genetic

Algorithm and PSO) have lower but more evenly distributed SHAP values across genes as observed in Figure 8 and Figure 9, indicating that they may rely on a more balanced set of features compared to the ant colony-based model. Despite the prevalence of DNAJB14 across all graphs, it has not been previously reported to be involved in HCC in the existing literature, and hence further clinical validation and research on possible links between this gene and HCC are warranted.

## 3.3 Clinical Validation Findings

DNAJB14, also known as heat shock protein 40kDa (HSP40), belongs to a group of DNAJ proteins which are localized in the endoplasmic reticulum (ER) and are essential co-chaperones of heat shock protein 70 (Hsp70) [56]. The primary function of this protein is to accelerate the proteasome-dependent degradation of misfolded transmembrane proteins, particularly under condition of ER stress. [57] Studies have shown that DNAJB12 and DNAJB14 can interact in cancer cells to facilitate ER protein reflux into the cytosol and promote pro-tumourigenic features. [58] Further evidence suggests that it may serve as a drug target in non-endometrioid carcinoma, while in HCC it may affect patient survival by regulating the tumour microenvironment through recruitment of NK cells [59] [60]. Yet, further evidence on the role of DNAJB14 in liver cancer is very limited. To evaluate whether the identified gene DNAJB14 has functional implications in HCC, in vitro validation was performed as shown in Figure 11. First, the association of DNAJB14 with overall survival in HCC patients was shown to be significant in public datasets. Here, the patients with high expression exhibited worse survival than the patients with low expression (Figure 11a). To further evaluate its role on HCC cells, siRNA-mediated knockdown of DNAJB14 was effectively performed, as confirmed by qRT-PCR and Western blotting (Figure 11b+c). Increasing concentrations of siRNA against DNAJB14 significantly affected cell viability, resulting in increased cell death, with highly pronounced effects at 150 and 200 pM siRNA. This confirmed its relevance in HCC cell survival (Figure 11d). A further effect of DNAJB14 knockdown on cell proliferation and selfrenewal was demonstrated using colony and sphere formation assays (Figure 11d+e). A significant reduction in colony and sphere formation capacity was observed, with up to 52% reduction in colonies and 48% reduction in spheres. Finally, the effects on migratory and invasive properties were tested, where DNAJB14 knockdown reduced cell migration, resulting in slower gap closure and lower invasion rate, as reflected by a lower cell index (Figure 11f+g). Overall, these results demonstrated the important role of DNAJB14 in HCC, validating the findings from artificial intelligence and machine learning approaches.

A  
![](images/84d9c4da79e5c0e9b28262695f7d78fe71fc410bd472f9e501d821a7c090c44b.jpg)

B  
![](images/20716197b020a1caa4c4e490f7b69b47130f1967e4f8016aa254e13c1bf37ebf.jpg)  
C

![](images/621faef0d13cc08141d27e1d91334a0fba7365ffe86fcb93a0e6eaa7e51e2239.jpg)

D  
![](images/07e668458389cb65145fed156bc478a3aead47ce6a5edf252ac1f20c1314cc27.jpg)  
E

![](images/2caef3fda580eaf56ffa70392f5449be27d070dbe3e10f3313126a133ce63016.jpg)

F  
![](images/fc889260a8e6f2f206ef849f014d049fc2957bde558f15ddd10583ebb52e6add.jpg)

![](images/e3f9e28fdc6502f4a9308f2cdc47d88c3d28e0caeb1537c053feddb601339430.jpg)

![](images/ce4fc500659459df5ac26d71cb78eab0f8f7fff9e91046865dcaaae910d652d3.jpg)  
Figure 111 In vitro functional validation of DNAJB14  
(A) Kaplan-Meier curve representing the overall survival of HCC patients with high and low expression of DNAJB14. The data was obtained from public dataset (TCGA) (B) Gene and (C) protein expression of DNAJB14 after treatment with increasing concentrations of DNAJB14 siRNA for 72 h. (D) Cell viability assay after treatment with increasing concentrations of DNAJB14 siRNA. (E+F) Representative images of colony and sphere formation assay and number of colonies and spheres after siRNA-mediated DNAJB14 knockdown. (G) Wound-healing assay showing the migratory properties of the cells. Gap closure reflects the speed and ability of cells to effectively migrate. (H) Invasion assay curves measured by xCELLigence represent the real-time observation of cell invasion through collagen I and IV layers on CIM-plate 16. The cell index reflects the number of invading cells. Mean±SD, n=3, \*p<0.05, \*\*p<0.01, \*\*\*p<0.001, \*\*\*p<0.0001

## 4. Conclusions

This research has investigated the application of deep learning and explainable artificial intelligence in liver cancer diagnosis for five different stages of hepatocarcinogenesis based on a genomic biomarkers HCC dataset constructed using semi-supervised learning from three source datasets. Several deep learning experiments were conducted using different feature extraction techniques and gene sets to determine the best features for training a model with the highest possible accuracy and minimal loss metrics, in addition to using class weights to overcome the class imbalance in the training data. The highest accuracy model was achieved using 15 selected genes with the SelectKBest algorithm, at 90.74% accuracy, and the lowest model loss was recorded using 20 genes at 0.3187. By using various optimization algorithms and classification models in addition to SHAP, this study has also demonstrated how XAI can provide insights into model decision-making in the context of liver cancer diagnosis and prediction with a high degree of accuracy in classifying liver cancer, identifying DNAJB14 as the most consistently influential gene across all three SHAP graphs.

The main limitation in this study is the class imbalance in the dataset, with the 'sl' class having significantly fewer samples compared to the 'phcc' and 'ehcc' classes. While a weighted training approach was used to mitigate this, more research and data are needed to mitigate the negative effects of this imbalance on the model’s performance and generalizability. For future work, genetic variations, environmental factors, and other clinical differences among patients from different regions can also affect the model's accuracy when applied to new populations, which means that genomic expression data may not be the only contributing factor in liver cancer development and progression, requiring future research to explore combinations with other features and patient data. To investigate whether the identified gene across all three SHAP graphs is clinically relevant, functional in vitro validation was performed in the HCC model. Analysis of DNAJB14 inhibition clearly showed a significant impact on cellular properties. This was strongly reflected in reduced cell survival, self-renewal and impaired pro-metastatic properties with reduced cell migration and invasion. Even though this gene has not been previously reported to be highly associated with HCC in the literature, it was identified as the most influential gene in all three explanatory AI models and was successfully validated in HCC for the first time.

Ethics Statement: The authors would like to convey their thanks and appreciation to the University of Sharjah and University of Lubeck for supporting the work. All experiments were conducted on publicly available datasets

Consent for publication: Not Applicable

Competing Interests: "The authors declare that they have no conflict of interest.”

Informed consent: "This study does not involve any experiments on animals."

Data Availability: The datasets used in this research are publicly available and are described in Section 2.1.

Funding: This project was funded by University of Sharjah and University of Lubeck

## 5. References

[1] A. Beaufrère et al., “Primary liver cancer classification from routine tumour biopsy using weakly supervised deep learning,” JHEP Reports, vol. 6, no. 3, p. 101008, Mar. 2024, doi: 10.1016/j.jhepr.2024.101008.

[2] H. Jiang, Y. Yin, J. Zhang, W. Deng, and C. Li, “Deep learning for liver cancer histopathology image analysis: A comprehensive survey,” Eng Appl Artif Intell, vol. 133, p. 108436, Jul. 2024, doi: 10.1016/j.engappai.2024.108436.

[3] U. Hameed, M. Ur Rehman, A. Rehman, R. Damaševičius, A. Sattar, and T. Saba, “A deep learning approach for liver cancer detection in CT scans,” Comput Methods Biomech Biomed Eng Imaging Vis, vol. 11, no. 7, Jan. 2024, doi: 10.1080/21681163.2023.2280558.

[4] M.-D. Li et al., “Systematic comparison of deep-learning based fusion strategies for multi-modal ultrasound in diagnosis of liver cancer,” Neurocomputing, vol. 603, p. 128257, Oct. 2024, doi: 10.1016/j.neucom.2024.128257.

[5] A. Alsalama, A. Kubba, G. Jamjoum, and Z. Al Aghbari, “Classification Based on Association Rules Algorithm for Breast Cancer,” in 2024 Advances in Science and Engineering Technology International Conferences (ASET), IEEE, Jun. 2024, pp. 1–6. doi: 10.1109/ASET60340.2024.10708711.

[6] C. Chakraborty, M. Bhattacharya, S. Pal, and S.-S. Lee, “From machine learning to deep learning: Advances of the recent data-driven paradigm shift in medicine and healthcare,” Curr Res Biotechnol, vol. 7, p. 100164, 2024, doi: 10.1016/j.crbiot.2023.100164.

[7] B. Soudan, S. Abbas, A. Kubba, O. Abu Waraga, M. Abu Talib, and Q. Nasir, “Scalability and performance evaluation of federated learning frameworks: a comparative analysis,” International Journal of Machine Learning and Cybernetics, Feb. 2025, doi: 10.1007/s13042-024-02453-4.

[8] G. Appasami and N. Savarimuthu, “Federated learning for secure medical MRI brain tumour image classification,” Eur Phys J Spec Top, Feb. 2025, doi: 10.1140/epjs/s11734-025-01516-z.

[9] Y. Han, J. Akhtar, G. Liu, C. Li, and G. Wang, “Early warning and diagnosis of liver cancer based on dynamic network biomarker and deep learning,” Comput Struct Biotechnol J, vol. 21, pp. 3478–3489, Jan. 2023, doi: 10.1016/j.csbj.2023.07.002.

[10] T. Goldfarb et al., “NCBI RefSeq: reference sequence standards through 25 years of curation and annotation,” Nucleic Acids Res, vol. 53, no. D1, pp. D243–D257, Jan. 2025, doi: 10.1093/nar/gkae1038.

[11] O. H. Purba, E. A. Sarwoko, Khadijah, Suhartono, and A. Wibowo, “Classification of liver cancer with microrna data using the deep neural network (DNN) method,” in Journal of Physics: Conference Series, Institute of Physics Publishing, Jun. 2020. doi: 10.1088/1742-6596/1524/1/012129.

[12] Z. Wang et al., “NCI Cancer Research Data Commons: Resources to Share Key Cancer Data,” Cancer Res, vol. 84, no. 9, pp. 1388–1395, May 2024, doi: 10.1158/0008-5472.CAN-23-2468.

[13] G. Zhang, Z. Peng, C. Yan, J. Wang, J. Luo, and H. Luo, “A novel liver cancer diagnosis method based on patient similarity network and DenseGCN,” Sci Rep, vol. 12, no. 1, Dec. 2022, doi: 10.1038/s41598-022-10441-3.

[14] J. C. Ahn, T. A. Qureshi, D. Li, A. G. Singal, and J. D. Yang, “Deep learning in hepatocellular carcinoma: Current status and future perspectives,” World J Hepatol, vol. 13, no. 12, pp. 2039–2051, 2021, doi: 10.4254/wjh.v13.i12.2039.

[15] G. N. Ioannou et al., “Assessment of a Deep Learning Model to Predict Hepatocellular Carcinoma in Patients With Hepatitis C Cirrhosis,” JAMA Netw Open, vol. 3, no. 9, p. e2015626, Sep. 2020, doi: 10.1001/jamanetworkopen.2020.15626.

[16] D. Phan, C. Chan, A. A. Li, T. Chien, and V. Nguyen, “Liver cancer prediction in a viral hepatitis cohort: A deep learning approach,” Int J Cancer, vol. 147, no. 10, pp. 2871–2878, Nov. 2020, doi: 10.1002/ijc.33245.

[17] J. Y. Nam, D. H. Sinn, J. Bae, E. S. Jang, J.-W. Kim, and S.-H. Jeong, “Deep learning model for prediction of hepatocellular carcinoma in patients with HBV-related cirrhosis on antiviral therapy,” JHEP Reports, vol. 2, no. 6, p. 100175, Dec. 2020, doi: 10.1016/j.jhepr.2020.100175.

[18] C. An et al., “Association of serum AFP trajectories and hepatocellular carcinoma outcomes after hepatic arterial infusion chemotherapy: A longitudinal, multicenter study,” Cancer Med, vol. 13, no. 11, Jun. 2024, doi: 10.1002/cam4.7319.

[19] Y. Xu et al., “Development of machine learning-based personalized predictive models for risk evaluation of hepatocellular carcinoma in hepatitis B virus-related cirrhosis

patients with low levels of serum alpha-fetoprotein,” Ann Hepatol, vol. 29, no. 6, p. 101540, Nov. 2024, doi: 10.1016/j.aohep.2024.101540.

[20] Y. Lu, B. Lin, and M. Li, “The role of alpha-fetoprotein in the tumour microenvironment of hepatocellular carcinoma,” Front Oncol, vol. 14, Apr. 2024, doi: 10.3389/fonc.2024.1363695.

[21] A. Das, U. R. Acharya, S. S. Panda, and S. Sabut, “Deep learning based liver cancer detection using watershed transform and Gaussian mixture model techniques,” Cogn Syst Res, vol. 54, pp. 165–175, May 2019, doi: 10.1016/j.cogsys.2018.12.009.

[22] K. Tao et al., “Machine learning-based genome-wide interrogation of somatic copy number aberrations in circulating tumour DNA for early detection of hepatocellular carcinoma,” EBioMedicine, vol. 56, Jun. 2020, doi: 10.1016/j.ebiom.2020.102811.

[23] W. Książek, M. Abdar, U. R. Acharya, and P. Pławiak, “A novel machine learning approach for early detection of hepatocellular carcinoma patients,” Cogn Syst Res, vol. 54, pp. 116–127, May 2019, doi: 10.1016/j.cogsys.2018.12.001.

[24] J. Chen, C. Niu, N. Yang, C. Liu, S. shan Zou, and S. Zhu, “Biomarker discovery and application—An opportunity to resolve the challenge of liver cancer diagnosis and treatment,” Mar. 01, 2023, Academic Press. doi: 10.1016/j.phrs.2023.106674.

[25] O. O. Petinrin et al., “Machine learning in metastatic cancer research: Potentials, possibilities, and prospects,” Jan. 01, 2023, Elsevier B.V. doi: 10.1016/j.csbj.2023.03.046.

[26] H. Jiang, Y. Yin, J. Zhang, W. Deng, and C. Li, “Deep learning for liver cancer histopathology image analysis: A comprehensive survey,” Jul. 01, 2024, Elsevier Ltd. doi: 10.1016/j.engappai.2024.108436.

[27] H. L. Ngan, K. Y. Lam, Z. Li, J. Zhang, and Z. Cai, “Machine learning facilitates the application of mass spectrometry-based metabolomics to clinical analysis: A review of early diagnosis of high mortality rate cancers,” Nov. 01, 2023, Elsevier B.V. doi: 10.1016/j.trac.2023.117333.

[28] M. Matboli et al., “Machine learning based identification of key feature RNAsignature linked to diagnosis of Hepatocellular Carcinoma,” J Clin Exp Hepatol, p. 101456, Nov. 2024, doi: 10.1016/j.jceh.2024.101456.

[29] J. Shen, H. Gao, B. Li, Y. Huang, and Y. Shi, “The integration of machine learning and multi-omics analysis provides a powerful approach to screen aging-related genes and predict prognosis and immunotherapy efficacy in hepatocellular carcinoma, Aging, vol. 15, no. 14, pp. 6848–6864, Jul. 2023, doi: 10.18632/aging.204876.

[30] M. Gholizadeh, S. R. Mazlooman, M. Hadizadeh, M. Drozdzik, and S. Eslami, “Detection of key mRNAs in liver tissue of hepatocellular carcinoma patients based on machine learning and bioinformatics analysis,” MethodsX, vol. 10, Jan. 2023, doi: 10.1016/j.mex.2023.102021.

[31] H. Rumgay et al., “Global burden of primary liver cancer in 2020 and predictions to 2040,” J Hepatol, vol. 77, no. 6, pp. 1598–1606, Dec. 2022, doi: 10.1016/j.jhep.2022.08.021.

[32] Y. Han, J. Akhtar, G. Liu, C. Li, and G. Wang, “Early warning and diagnosis of liver cancer based on dynamic network biomarker and deep learning,” Comput Struct Biotechnol J, vol. 21, pp. 3478–3489, 2023, doi: 10.1016/j.csbj.2023.07.002.

[33] B. Das and S. Toraman, “Deep transfer learning for automated liver cancer gene recognition using spectrogram images of digitized DNA sequences,” Biomed Signal Process Control, vol. 72, p. 103317, Feb. 2022, doi: 10.1016/j.bspc.2021.103317.

[34] O. H. Purba, E. A. Sarwoko, Khadijah, Suhartono, and A. Wibowo, “Classification of liver cancer with microrna data using the deep neural network (DNN) method,” J Phys

Conf Ser, vol. 1524, no. 1, p. 012129, Apr. 2020, doi: 10.1088/1742- 6596/1524/1/012129.

[35] G. Zhang, Z. Peng, C. Yan, J. Wang, J. Luo, and H. Luo, “A novel liver cancer diagnosis method based on patient similarity network and DenseGCN,” Sci Rep, vol. 12, no. 1, p. 6797, Apr. 2022, doi: 10.1038/s41598-022-10441-3.

[36] A. Das, U. R. Acharya, S. S. Panda, and S. Sabut, “Deep learning based liver cancer detection using watershed transform and Gaussian mixture model techniques,” Cogn Syst Res, vol. 54, pp. 165–175, May 2019, doi: 10.1016/j.cogsys.2018.12.009.

[37] K. Tao et al., “Machine learning-based genome-wide interrogation of somatic copy number aberrations in circulating tumour DNA for early detection of hepatocellular carcinoma,” EBioMedicine, vol. 56, p. 102811, Jun. 2020, doi: 10.1016/j.ebiom.2020.102811.

[38] W. Książek, M. Abdar, U. R. Acharya, and P. Pławiak, “A novel machine learning approach for early detection of hepatocellular carcinoma patients,” Cogn Syst Res, vol. 54, pp. 116–127, May 2019, doi: 10.1016/j.cogsys.2018.12.001.

[39] H.-L. Ngan, K.-Y. Lam, Z. Li, J. Zhang, and Z. Cai, “Machine learning facilitates the application of mass spectrometry-based metabolomics to clinical analysis: A review of early diagnosis of high mortality rate cancers,” TrAC Trends in Analytical Chemistry, vol. 168, p. 117333, Nov. 2023, doi: 10.1016/j.trac.2023.117333.

[40] M. Matboli et al., “Machine-Learning-Based Identification of Key Feature RNA-Signature Linked to Diagnosis of Hepatocellular Carcinoma,” J Clin Exp Hepatol, vol. 14, no. 6, p. 101456, Nov. 2024, doi: 10.1016/j.jceh.2024.101456.

[41] J. Shen, H. Gao, B. Li, Y. Huang, and Y. Shi, “The integration of machine learning and multi-omics analysis provides a powerful approach to screen aging-related genes and predict prognosis and immunotherapy efficacy in hepatocellular carcinoma,” Aging, vol. 15, no. 14, pp. 6848–6864, Jul. 2023, doi: 10.18632/aging.204876.

[42] M. Gholizadeh, S. R. Mazlooman, M. Hadizadeh, M. Drozdzik, and S. Eslami, “Detection of key mRNAs in liver tissue of hepatocellular carcinoma patients based on machine learning and bioinformatics analysis,” MethodsX, vol. 10, p. 102021, 2023, doi: 10.1016/j.mex.2023.102021.

[43] B. Lin et al., “A novel approach to the analysis of Overall Survival (OS) as response with Progression-Free Interval (PFI) as condition based on the RNA-seq expression data in The Cancer Genome Atlas (TCGA),” BMC Bioinformatics, vol. 25, no. 1, p. 300, Sep. 2024, doi: 10.1186/s12859-024-05897-1.

[44] Q. Shen et al., “Barrier to autointegration factor 1, procollagen‐lysine, 2‐oxoglutarate 5‐dioxygenase 3, and splicing factor 3b subunit 4 as early‐stage cancer decision markers and drivers of hepatocellular carcinoma,” Hepatology, vol. 67, no. 4, pp. 1360–1377, Apr. 2018, doi: 10.1002/hep.29606.

[45] A. M. Salih et al., “A Perspective on Explainable Artificial Intelligence Methods: SHAP and LIME,” Advanced Intelligent Systems, vol. 7, no. 1, Jan. 2025, doi: 10.1002/aisy.202400304.

[46] S. S. Chandra, R. Kumar, A. Arjunasamy, S. Galagali, A. Tantri, and S. R. Naganna, “Predicting the compressive strength of polymer-infused bricks: A machine learning approach with SHAP interpretability,” Sci Rep, vol. 15, no. 1, p. 8090, Mar. 2025, doi: 10.1038/s41598-025-89606-9.

[47] H. A. Salman, A. Kalakech, and A. Steiti, “Random Forest Algorithm Overview,” Babylonian Journal of Machine Learning, vol. 2024, pp. 69–79, Jun. 2024, doi: 10.58496/BJML/2024/007.

[48] Z. Sun, G. Wang, P. Li, H. Wang, M. Zhang, and X. Liang, “An improved random forest based on the classification accuracy and correlation measurement of decision

trees,” Expert Syst Appl, vol. 237, p. 121549, Mar. 2024, doi: 10.1016/j.eswa.2023.121549.

[49] M. Niazkar et al., “Applications of XGBoost in water resources engineering: A systematic literature review (Dec 2018–May 2023),” Environmental Modelling & Software, vol. 174, p. 105971, Mar. 2024, doi: 10.1016/j.envsoft.2024.105971.

[50] S. M. Ganie and P. K. Dutta Pramanik, “A comparative analysis of boosting algorithms for chronic liver disease prediction,” Healthcare Analytics, vol. 5, p. 100313, Jun. 2024, doi: 10.1016/j.health.2024.100313.

[51] M. A. Awadallah et al., “Multi-objective Ant Colony Optimization: Review,” Archives of Computational Methods in Engineering, vol. 32, no. 2, pp. 995–1037, Mar. 2025, doi: 10.1007/s11831-024-10178-4.

[52] B. Alhijawi and A. Awajan, “Genetic algorithms: theory, genetic operators, solutions, and applications,” Evol Intell, vol. 17, no. 3, pp. 1245–1256, Jun. 2024, doi: 10.1007/s12065-023-00822-6.

[53] L. Abualigah et al., “Particle swarm optimization algorithm: review and applications,” in Metaheuristic Optimization Algorithms, Elsevier, 2024, pp. 1–14. doi: 10.1016/B978-0-443-13925-3.00019-4.

[54] S. Julkaew, T. Wongsirichot, K. Damkliang, and P. Sangthawan, “Improving accuracy of vascular access quality classification in hemodialysis patients using deep learning with K highest score feature selection,” Journal of International Medical Research, vol. 52, no. 4, Apr. 2024, doi: 10.1177/03000605241232519.

[55] M. H. Saeed and J. I. Hama, “Cardiac disease prediction using AI algorithms with SelectKBest,” Med Biol Eng Comput, vol. 61, no. 12, pp. 3397–3408, Dec. 2023, doi: 10.1007/s11517-023-02918-8.

[56] H. H. Kampinga et al., “Function, evolution, and structure of J-domain proteins,” Cell Stress Chaperones, vol. 24, no. 1, pp. 7–15, Jan. 2019, doi: 10.1007/s12192-018-0948- 4.

[57] P. Sopha et al., “A Novel Mammalian ER-located J-protein, DNAJB14, Can Accelerate ERAD of Misfolded Membrane Proteins,” Cell Struct Funct, vol. 37, no. 2, pp. 177–187, 2012, doi: 10.1247/csf.12017.

[58] S. Dabsan et al., “Cytosolic and endoplasmic reticulum chaperones inhibit wt-p53 to increase cancer cells’ survival by refluxing ER-proteins to the cytosol,” Elife, vol. 14, Apr. 2025, doi: 10.7554/eLife.102658.

[59] J. Zhu, T. Zhang, J. Jiang, M. Yang, N. Xia, and Y. Chen, “Genetic variation perspective reveals potential drug targets for subtypes of endometrial cancer,” Sci Rep, vol. 14, no. 1, p. 28180, Nov. 2024, doi: 10.1038/s41598-024-78689-5.

[60] W. Tang et al., “Identification of SGMS2 as a molecule involved in natural killer cell recruitment and its in‐deep analysis in the liver cancer microenvironment: Evidence from large populations cohort,” J Gene Med, vol. 26, no. 1, Jan. 2024, doi: 10.1002/jgm.3609.