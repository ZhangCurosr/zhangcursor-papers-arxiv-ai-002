# Wearable ECG Quality Assessment: A Deep Learning and Ambulatory Context-Awareness Approach

Xiaopeng Mao   
Health Technology department   
Technical University of Denmark   
Kongens Lyngby, Denmark   
s194408@dtu.dk   
Marike Weisbjerg   
Health Technology department   
Technical University of Denmark   
Kongens Lyngby, Denmark   
s194388@dtu.dk

Sadasivan Puthusserypady Health Technology department Technical University of Denmark Kongens Lyngby, Denmark sapu@dtu.dk

Abstract—This paper presents and evaluates a Deep Learningbased (DL-based) Signal Quality Assessment (SQA) model to distinguish between clean and noisy ambulatory Electrocardiograms (ECG). The model is trained on Copenhagen Center for Health Technology-Contextualized Arrhythmia Database (CACHET-CADB), which, to the best of our knowledge, is the first ambulatory ECG database with both physical and patientreported contextual data. The model shows stable performance on different databases such as MIT-databases and the latest PyhsioNet/Cinc Challenge 2021 databases. Subsequently, the paper demonstrates how complicated ECG noise can be investigated by the SQA model and the physical contextual data.

Index Terms—Deep learning, ECG, SQA, contextualized analysis

## I. INTRODUCTION

As a major heart disease, arrhythmias require continuous monitoring with portable ECG device. Hence, it is important to prevent noise from obscuring the ECG. Under free-living conditions, the noise level is particularly severe compared to in-hospital recording. To assess the quality of ambulatory ECG, this study provides a DL SQA model trained on a contextualized ambulatory ECG arrhythmia data set, CACHET-CADB [1]. The strength and weakness of the model is evaluated on the well-known public databases such as MIT-BIH-Arrhythmia and the recent databases from PhysioNet/Cinc challenge 2021. The results from the contextualized analysis are also included to show how the aforementioned model combined with the physical ECG context can provide a deeper insight of ECG noise and thus improve the recording environment. In summary, this study aims to evaluate a stateof-the-art DL SQA model on multiple databases and to explore the new possibilities of a novel contextualized ECG database.

## II. METHODS

## A. Datasets

CACHET-CADB consists of 259 days of continuous ambulatory ECG recorded from 24 people, where 21 are Atrial Fibrillation (AFIB) patients while the rest are healthy individuals. The signal is sampled in 1,024 Hz from one single precordial lead as described in [1]. 1,602 of 10 s ECGs are annotated by two cardiologists independently. The labelled classes are Normal Sinus Rhythm (NSR), AFIB, noise and others. Table 1 provides an overview of the labelled data.

TABLE I  
ECG RHYTHMS FROM CACHET-CADB.
<table><tr><td>Category</td><td>NSR</td><td>AFIB</td><td>Other</td><td>Noise</td></tr><tr><td>Number of signals</td><td>615</td><td>747</td><td>19</td><td>221</td></tr><tr><td>Considered class</td><td>Clean</td><td>Clean</td><td>Clean</td><td>Noisy</td></tr></table>

In this study, the aim of the SQA is to separate ECG noise from anything else. Hence, an ECG that is either NSR, AFIB or other rhythms will be identified as ”clean”, otherwise the ECG is ”noisy”. The labelled ECGs are treated as the test set to evaluate the DL model.

## B. Continuous Wavelet Transformation

Continuous Wavelet Transformation (CWT) transforms the 1D ECGs into 2D scalograms. It is defined as the absolute values of the CWT-coefficients [2]. The advantage of CWT is that it provides a time-frequency representation of the signal, whereas plain ECG only provides a time domain representation. Hence, image classification between clean and noisy ECGs can be performed on scalograms. The CWTcoefficients are calculated by the formula in (1).

$$
C _ { \Psi , x } ( a , b ) = \frac { 1 } { \sqrt { | a | } } \cdot \int _ { - \infty } ^ { \infty } x ( t ) \cdot \Psi ^ { * } \left( \frac { t - b } { a } \right) d t , a \neq 0 ,\tag{1}
$$

where $x ( t )$ is the input ECG, Ψ∗ is time-scaled and timeshifted wavelet function of the mother wave, Ψ, a is the timescaling parameter and b is the time-shifting parameter [3]. The frequency is defined in equation (2).

$$
f = \frac { f _ { c } \cdot f _ { s } } { a } , a \ne 0 ,\tag{2}
$$

where $f _ { s }$ denotes the ECG sampling frequncy, i.e. 1,024 Hz, $f _ { c }$ denotes the central frequency of Ψ [4]. In this study, the frequency is in a logarithmic scale since it better represents the scalogram. Figure 1 and 2 show a scalogram of a 10 s NSR and noise ECG, respectively.

![](images/878e6a8d543601ff6b7b6b2099ddbdcb00fefecaa3b83b6e550087e4dfebe18b.jpg)

![](images/583b67337b646d8b1d9f9b8f2e8745a0ca717cf2d00e7834964a8dc3b0dadd4a.jpg)  
Fig. 1. Scalogram of a 10 s NSR

![](images/0c5c23c5ac40d19a9377acec70e19437345c13a10c9c0b5713126a59e748dae0.jpg)

![](images/29696866023728c0a50e23b7610c94fadcb7dccbcb00ea701509a35655d3dc63.jpg)  
Fig. 2. Scalogram of a 10 s noise

CWT is widely applied to public databases such as MIT-BIH arrhythmia and FysioNet/CinC Challenge to train binary/multi-class ECG SQA [5], [6]. In [6]. Huerta et al. used Transfer Learning (TL) to train a series of DL models. In this work, InceptionV3 is used to classify the ECGs.

## C. Model architecture

The input dimensions are $2 2 4 \times 2 2 4 \times 3$ . The input goes through the InceptionV3 modules to obtain the new output dimensions of $5 \times 5 \times 2 0 4 8$ , which is flattened before putting into the dense layers. During the process, all parameters of InceptionV3 are frozen to prevent overfitting. A dropout layer is also added to Dense layer 1 to stabilize the network training. Figure 3 illustrates the above-mentioned model achitecture.

![](images/8bb6a271a9d195fd6796e16155c6e9c26c2c19d8ed573e65f0253d11de05dd0c.jpg)  
Fig. 3. Architecture of the final model. The three input images are identical. The red neurons in Dense layer 1 are prohibited by a dropout probability.

The hyper-parameters are tuned by using random search. Table II shows the final values for the hyper-parameters of the model.

TABLE II  
OPTIMIZED HYPER-PARAMETERS OF THE DL MODEL.
<table><tr><td>Hyper-parameter</td><td>Value</td></tr><tr><td>Dense layer 1 size</td><td>20</td></tr><tr><td>Learning rate</td><td> $1 \cdot 1 0 ^ { - 5 }$ </td></tr><tr><td>Dropout probability</td><td>0.6</td></tr><tr><td>Batch size</td><td>20</td></tr><tr><td>Epochs</td><td>40</td></tr></table>

## D. Contextualized analysis

The contextualized analysis is done by using the physical parameters that are automatically measured by the barometer and accelerometer of the ECG device [1]. The parameters include body positions and activity classes listed in Table III.

TABLE III  
THE CONTEXTUALIZED PARAMETERS PRESENTED IN [1].
<table><tr><td>Attribute</td><td>Parameter</td></tr><tr><td>Body position</td><td>0: Unknown 1: Lying supine 2: Lying let 3: Lying prone 4: Lying right 5: Upright 6: Sitting/lying 7: Standing 99: Not worn</td></tr><tr><td></td><td>0: Unknown 1: Lying 2: Sitting/standing 3: Cycling 4: Slope up 5: Jogging</td></tr><tr><td>Activity class</td><td>6: Slope down 7: Walking 8: Sitting/ying 9: Standing 10: Sitting/lying/standing 11: Sitting 99: Not worn</td></tr></table>

As the standard pre-processing, all ECGs are bandpassfiltered within 0.5 and 50 Hz and smoothed by using Savitzky-Golay filter [1], [10]. The signals are then re-sampled to 256 Hz. 20,000 clean and 20,000 noisy ECGs of 10 s are generated from CACHET-CADB by using a set of decision rules combined with template matching from [8]. The method is adjusted to optimize its performance on AFIB. Figure 4 shows the workflow throughout the entire study.

![](images/e481640d2fcf4872ea3403245f320fa3d5e17005251c8e1c344807637ff18268.jpg)  
Fig. 4. Flow diagram of the work.

As for the network training, Adaptive Moment Estimation (Adam) optimizer is used [5]. The cross entropy function is used as the loss function for the DL model [9]. The software used for training is Python and TensorFlow. The model is trained by using 3 nodes of 4 × Tesla V100 (32 GB) with NVlink (owned by DTU Compute).

## III. RESULTS

A. Performance on CACHET-CADB test set

Table IV shows the confusion matrix of the DL SQA model.

TABLE IV  
CONFUSION MATRIX OF THE MODEL PERFORMANCE ON CACHET-CADB TEST SET.
<table><tr><td rowspan=2 colspan=2>1,602 labelledsignals in total.</td><td rowspan=1 colspan=2>Actual class</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>True</td><td rowspan=1 colspan=1>False</td></tr><tr><td rowspan=2 colspan=1>Prediction</td><td rowspan=1 colspan=1>True</td><td rowspan=1 colspan=1>TP = 215</td><td rowspan=1 colspan=1>FP = 128</td><td rowspan=1 colspan=1>Predictedpositives= 343</td></tr><tr><td rowspan=1 colspan=1>False</td><td rowspan=1 colspan=1>FN = 6</td><td rowspan=1 colspan=1>TN = 1,253</td><td rowspan=1 colspan=1>Predictednegatives= 1,259</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1> $\mathrm { S e } = 9 7 . 3 \ \%$ </td><td rowspan=1 colspan=1> $\mathrm { S p } = 9 0 . 7 ~ \%$ </td><td rowspan=1 colspan=1>Acc = 91.6%</td></tr></table>

In Table IV, it can be seen that the accuracy lies closer to the specificity than sensitivity due to the fact that the number of noisy signals are significantly lower than the number of clean signals. The learning curves of the model are shown in Figure 5. The training set consists of 32,000 images while the validation set consists of 8,000 images. The clean and noisy ECGs are evenly distributed among them.

![](images/bf2740e35ce6f55a9a2b64271b0af5b85f8c310b211388b902ba5cf98c1daa99.jpg)  
Fig. 5. The learning curves of the model.

The convergence of both loss curves in Figure 5 indicates a stable learning process with the optimized hyper-parameters from Table II. An exponentially decaying learning schedule with a base of 0.18 and a step of 10 is used, so the learning rate decreases with 82 % after every 10 epochs.

## B. Performance on MIT-BIH-NSR, MIT-BIH-Arrhythmia and MIT-BIH-Noise Stress Test Database

3 records are randomly chosen from the MIT-BIH-NSR database. The entire arrhythmia database is used. 3 pure noise signals are used from MIT-BIH-Noise Stress Test database (the noise types include baseline wander, muscle artifact and electrode motion artifact). Lead 2 ECG is used for the model evaluation because this particular lead most clearly expresses the QRS complex [10].

TABLE V  
PREDICTION ON DIFFERENT MIT-BIH DATABASES.
<table><tr><td>Database</td><td>Record amount</td><td>Duration (h)</td><td>Percentage of predicted clean signals (%)</td></tr><tr><td>MIT-BIH-NSR</td><td>3</td><td>48</td><td>98.2</td></tr><tr><td>MIT-BIH-Arrhythmia</td><td>48</td><td>24</td><td>76.9</td></tr><tr><td>MIT-BIH-Noise Stress Test</td><td>3</td><td>1.5</td><td>2.2</td></tr></table>

The individual record of MIT-BIH-Arrhythmia shows that the model tends to fail on rhythms with a rather abnormal morphology such as paced beats combined with Premature Ventricular Contraction (PVC).

## C. Performance on PhysioNet/CinC challenge 2021

The Chapman-Shaoxing database is one of the most recent databases from the Chapman University and Shaoxing People’s Hospital [11]. The database contains 10,646 (10,247 available from the challenge) of 10 s clean ECGs from 10,646 heart patients. The sampling frequency is 500 Hz. Table VI shows the model performance categorized for each ECG rhythm.

TABLE VI  
DL MODEL PERFORMANCE ON THE CHAPMAN-SHAOXING DATABASE. THE WORST PERFORMANCES ARE MARKED WITH ORANGE.
<table><tr><td>Category</td><td>Number of signals</td><td>Number of predicted clean signals</td><td>Percentage of predicted clean signals (%)</td></tr><tr><td>Sinus Bradycardia (SB)</td><td>3,889</td><td>3,605</td><td>92.7</td></tr><tr><td>Normal Sinus Rhythm (NSR/SR)</td><td>1,826</td><td>1,661</td><td>91.0</td></tr><tr><td>Atrial Fibrillation (AFIB)</td><td>1,780</td><td>1,171</td><td>65.8</td></tr><tr><td>Sinus Tachycardia (ST)</td><td>1,568</td><td>1,305</td><td>83.2</td></tr><tr><td>Supraventricular Tachycardia (SVT)</td><td>587</td><td>345</td><td>58.8</td></tr><tr><td>Atrial Flutter (AF)</td><td>445</td><td>280</td><td>62.9</td></tr><tr><td>Atrial Tachycardia (AT)</td><td>121</td><td>73</td><td>60.3</td></tr><tr><td>Atrioventricular Node Reentrant Tachycardia (AVNRT)</td><td>16</td><td>14</td><td>87.5</td></tr><tr><td>Atrioventricular Reentrant Tachycardia (AVRT)</td><td>8</td><td>7</td><td>87.5</td></tr><tr><td>Sinus Atrium to Atrial Wandering Rhythm (SAAWR)</td><td>7</td><td>7</td><td>100</td></tr><tr><td>Total</td><td>10.247</td><td>8.468</td><td>82.6</td></tr></table>

The Ningbo database is the largest database from the 2021 challenge. The database contains 40,258 ECGs (34,905 available from the challenge). The ECGs are all from different patients, 10 s long, and sampled with 500 Hz. All annotations are manually identified, so the model performance are evaluated for all ECG types (see Table VII).

TABLE VII  
DL MODEL PERFORMANCE ON THE NINGBO ARRHYTHMIA DATABASE. THE WORST PERFORMANCES ARE MARKED WITH ORANGE.
<table><tr><td>Category*</td><td>Number of signals</td><td>Number of predicted clean signals</td><td>Percentage of predicted clean signals (%)</td></tr><tr><td>Sinus Bradycardia (SB)</td><td>11,919</td><td>11,059</td><td>92.8</td></tr><tr><td>Normal Sinus Rhythm (NSR/SR)</td><td>6,058</td><td>5,489</td><td>90.6</td></tr><tr><td>Atrial Flutter (AF)</td><td>5,164</td><td>3,498</td><td>67.7</td></tr><tr><td>Sinus Tachycardia (ST)</td><td>3,820</td><td>3,182</td><td>83.3</td></tr><tr><td>Sinus arrhythmia</td><td>1,294</td><td>1,180</td><td>91.2</td></tr><tr><td>Premature Atrial Contraction (PAC)</td><td>975</td><td>681</td><td>69.8</td></tr><tr><td>T wave abnormal</td><td>896</td><td>702</td><td>78.3</td></tr><tr><td>ST interval abnormal</td><td>401</td><td>268</td><td>66.8</td></tr><tr><td>Left ventricular hypertrophy</td><td>390</td><td>286</td><td>73.3</td></tr><tr><td>Non-specific intraventricular conduction delay</td><td>364</td><td>187</td><td>51.4</td></tr><tr><td>ST segment changes</td><td>331</td><td>221</td><td>66.8</td></tr><tr><td>Right axis deviation</td><td>301</td><td>218</td><td>72.4</td></tr><tr><td>ST Depression</td><td>275</td><td>180</td><td>65.5</td></tr><tr><td>Paced beats First degree</td><td>265</td><td>153</td><td>57.7</td></tr><tr><td>atrioventricular block</td><td>233</td><td>162</td><td>69.5</td></tr><tr><td>Complete right bundle branch block</td><td>225</td><td>133</td><td>59.1</td></tr><tr><td>Inverted T wave</td><td>119</td><td>70</td><td>58.8</td></tr><tr><td>Atrial rhythm Prolonged</td><td>112</td><td>98</td><td>87.5</td></tr><tr><td>QT interval</td><td>110</td><td>83</td><td>75.5</td></tr><tr><td>R wave</td><td>108</td><td>63</td><td>58.3</td></tr><tr><td>Counterclockwise vectorcardiographic loop</td><td>100</td><td>85</td><td>85.0</td></tr><tr><td>59 other abnormal ECG rhythms</td><td>1,445</td><td>740</td><td>51.2</td></tr><tr><td>(each less than 100) Total</td><td>34,905</td><td>28,738</td><td>82.3</td></tr></table>

\*Manually identified by SNOMED-CT code look up.

In summary, the evaluations show that the model is prone to make error when the morphology of the ECG deviates significantly from the normal sinus wave while a change of beat frequency (e.g. sinus bradycardia or tachycardia) does not affect the model performance severely.

## D. Contextualized analysis

The histograms in Figure 6 and 7 show the combinations of a body position and an activity class that are present when the recorded ECG is noisy or clean, respectively.

Histogram of noisy ECG contexts for patient 16 (3 days)

![](images/53012822520c14c7f3e933abbf1752e0d63af62fe3561e86109895598b29c0dd.jpg)  
Fig. 6. The underlying physical contexts of the noisy signals.

## Histogram of clean ECG contexts for patient 16 (3 days)

![](images/bf5c2bfdcfed3736caf3f31a8aa7fd5b8b7cf4d77e22b2dffa2fdffbf5d90342.jpg)  
Fig. 7. The underlying physical contexts of the clean signals.

In Figure 6, it can be seen that the ECG noise is mostly present for sitting/standing in an upright body position and a combination of unknowns. As for the clean ECGs, sitting/standing upright is also present as the most frequent compared to the other combinations. This indicates that the overall signal quality might be improved when the unknown body position and activity class are avoided. A more precis identification of the unknown parameters can be made by backtracking their time periods and by inquiring the patient about the underlying events for the unknown parameters.

## IV. DISCUSSION

In Table VI, it can be seen that the predictions on the sinus waves are particularly accurate. On the other hand, the model struggles to make correct predictions on the abnormal atrial waves such as AFIB.

The results from Table VII show that the abnormal ST segments, ST intervals and cardiac blocks are particularly difficult for the model to predict. This makes sense since the CACHET-CADB does not contain any patient, who suffers a disease that induces these rhythms. The predictions are poor especially on the rhythms with abnormal PQRST features. This agrees with the result from MIT-BIH-Arrhythmia in Table V.

In conclusion, the ECG rhythms with normal PQRST character are more likely to be predicted correctly by the model, whereas other pathological waveforms (the orange ones in Table VI and VII) with different morphologies are more likely to lead to wrong predictions.

As for the contextualized analysis, it is important to notice that the clean and noisy ECGs are not reviewed by any cardiologist. Hence, it is necessary to be cautious about the analysis results presented in Figures 6 and 7.

Nevertheless, the identification of the physical context with respect to ECG signal quality will mark a small beginning within the huge area of signal context research.

## ACKNOWLEDGMENT

This project was conducted in summer 2022 as part of a bachelor project. The project was accomplished at the Department of Health Technology and Copenhagen Center for Health Technology (CACHET) at the Technical University of Denmark (DTU). The computational resources were provided by DTU Compute.

## REFERENCES

[1] D. Kumar, S. Puthusserypady, H. Dominguez, K. Sharma and J. E. Bardram, “CACHET-CADB: A Contextualized Ambulatory 2 Electrocardiography Arrhythmia Dataset,” Frontiers in Cardiovascular Medicine, vol. 9, pp. 1–14, 2022.

[2] Y. H. Byeon, S. B. Pan and K. C. Kwak, ”Intelligent Deep Models Based on Scalograms of Electrocardiogram Signals for Biometrics,” Sensors (Switzerland), vol. 19, issue 4, pp. 935, 2019.

[3] K. Najarian and R. Splinter. Biomedical Signal and Image Processing. 2nd Edition, Taylor & Francis Group, LLC, 2012.

[4] V. Giurgiutiu, Structural health monitoring with piezoelectric wafer active sensors. 2nd Edition. Elsevier Inc., 2014.

[5] T. Wang, C. Lu, Y. Sun, M. Yang, C. Liu and C. Ou, ”Automatic ECG Classification Using Continuous Wavelet Transform and Convolutional Neural Network,” Entropy, vol. 23, issue 1, pp. 1-13, 2021.

[6] A. Huerta, A. Martinez-Rodrigo, A. Puchol, M. I. Pachon, J. J. Rieta and R. Alcaraz, ”Comparative Study of Convolutional Neural Networks for ECG Quality Assessment,” Computing in Cardiology (CinC), pp. 9344193, 2020.

[7] G. D Clifford et al., ”AF Classification from a Short Single Lead ECG Recording: the PhysioNet/Computing in Cardiology Challenge 2017,” Computing in Cardiology (CinC), vol. 44, pp. 1-4, 2017.

[8] C. Orphanidou, T. Bonnici, P. Charlton, D. Clifton, D. Vallance and L. Tarassenko, ”Signal-Quality Indices for the Electrocardiogram and Photoplethysmogram: Derivation and Applications to Wireless Monitoring,” IEEE Journal of Biomedical and Health Informatics, vol. 19, issue 3, pp. 6862843, 2015.

[9] D. Yoon, H. S. Lim, K. Jung, T. Y. Kim and S. Lee, ”Deep Learning-Based Electrocardiogram Signal Noise Detection and Screening Model,” Healthcare Informatics Research, vol. 25, issue 3, pp. 201-211, 2019.

[10] L. Sornmo and P. Laguna. Bioelectrical Signal Processing in Cardiac˙ and Neurological Applications, Elsevier Inc., 2005.

[11] J. Zheng, J. Zhang, S. Danioko, H. Yao, H. Guo, C. Rakovski, ”A 12- lead electrocardiogram database for arrhythmia research covering more than 10,000 patients,” Scientific Data, vol. 7, issue 1, pp. 48, 2020.

[12] J. Zheng et al., ”Optimal Multi-Stage Arrhythmia Classifcation Approach,” Scientific Reports, vol. 10, issue 1, pp. 2898, 2020.