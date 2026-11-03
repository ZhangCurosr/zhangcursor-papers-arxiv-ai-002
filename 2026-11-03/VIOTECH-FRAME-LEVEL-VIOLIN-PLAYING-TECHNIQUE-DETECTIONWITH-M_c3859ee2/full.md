# VIOTECH: FRAME-LEVEL VIOLIN PLAYING TECHNIQUE DETECTIONWITH MULTI-SCALE MIXTURE-OF-EXPERTS

Ting-Kang Wang<sup>1,4</sup> Vincent K.M. Cheung<sup>2</sup> Nan-Tien Lai<sup>3</sup> Li Su<sup>1</sup>

<sup>1</sup> Institute of Information Science, Academia Sinica, Taipei, Taiwan

<sup>2</sup> Sony Computer Science Laboratories, Inc., Tokyo, Japan

<sup>3</sup> National Taiwan University, Taipei, Taiwan

<sup>4</sup> Original Content Center, Gamania Inc., Taipei, Taiwan

tim.tkwang@gmail.com, cheung@csl.sony.co.jp, lisu@iis.sinica.edu.tw

## ABSTRACT

Playing technique is crucial for expressive violin performance, yet few violin technique detection models exist. Current systems mainly rely on note-level pipelines that first estimate note boundaries and then classify techniques. This is especially problematic for bowed string instruments such as the violin, where soft onsets and continuous bow-string interaction render note segmentation unreliable. Here, we propose VioTech, a frame-level violin technique detection framework that predicts techniques directly at 10 ms resolution without explicit note segmentation. To handle the large temporal-scale variation among techniques, VioTech combines a frozen MERT encoder with a multi-scale Mixture-of-Experts (MoE) architecture and task-specific routing for tonal, articulation, and legatorelated predictions. We also introduce VioTech120, the first manually annotated dataset of violin playing techniques constructed from real solo performance recordings, as well as an accompanying note-labeling user interface. Results demonstrate that VioTech achieved consistent improvements over prior methods and significantly enhanced cross-dataset generalization, establishing a new state-ofthe-art in violin technique-aware transcription.

## 1. INTRODUCTION

Violin playing techniques such as spiccato, pizzicato, legato, and flageolet, are central to expressive performance, yet their automatic detection from audio remains challenging, particularly at the frame level. Existing approaches typically adopt a note-level pipeline, where note boundaries are first estimated and techniques are then classified per note [1]. This design is however fragile, as classification performance depends on accurate note segmentation that is unreliable for bowed string instruments due to soft onsets and continuous bow–string interaction.

![](images/a61dfcd44191a4efc2bf82ca4ea9d32f00085d53b6aca9bb11ce1e2954e87667.jpg)  
Figure 1: Overview of proposed violin technique transcription system with multi-scale mixture of experts.

Frame-level prediction offers a principled alternative by removing the need for explicit segmentation. However, applying frame-level modeling to violin technique detection introduces a new challenge: techniques exhibit inherently multi-scale temporal characteristics. For example, pizzicato is localized to a short attack transient, spiccato depends on the bowing envelope, and bow-change events require longer, cross-note contexts. As a result, no single temporal window can adequately capture all relevant temporal dependencies.

To overcome these challenges, we propose VioTech, a frame-level violin playing technique-aware transcription system designed to address the inherently multi-scale nature of violin playing techniques. To this end, VioTech adopts a multi-scale mixture-of-experts (MoE) architecture that integrates complementary acoustic and note-aware representations. A transcription backbone provides framelevel note-aware cues, while a frozen acoustic encoder supports a set of temporal experts operating at different time scales. Through adaptive routing, the model captures technique-specific temporal dependencies while remaining interpretable across temporal and acoustic dimensions.

To support this task, we further introduce VioTech120, the first dataset of violin playing techniques constructed from real solo performances. The dataset comprises 120 manually annotated violin pieces, including technically demanding repertoire such as Paganini 24 Caprices. VioTech120 provides fine-grained note-level technique annotations rather than coarse file-level labels, covering six techniques and including, to our knowledge, the first explicit annotation of legato events. This design enables precise and diverse supervision for fine-grained technique recognition, with the accompanying note-labeling user interface enabling additional data generation with precise temporal control.

Our contributions are summarized as follows:

• We propose VioTech, a frame-level violin technique detection framework that eliminates explicit note segmentation and addresses the multi-scale nature of violin techniques through a Mixture-of-Experts architecture. We show that VioTech achieved excellent transcription performance and generalization.

• We introduce VioTech120, the first manually annotated dataset of violin playing techniques constructed from real solo performance recordings, providing fine-grained annotations.

• We release a piano-roll-based annotation interface that enables efficient and consistent note-level labeling of violin techniques, with precise audio-aligned temporal control that facilitates fine-grained framelevel modeling and dataset construction.

The dataset, codebase, and annotation interface are available on our project page. <sup>1</sup>

## 2. RELATED WORK

## 2.1 Violin Playing Technique Classification

Early work on violin technique recognition mainly framed the task as isolated-note or pre-segmented clip classification. Using temporal, spectral, cepstral, harmonic, and phase-derived features, Su et al. [2] showed that sparse coding of magnitude and phase-derived spectra were most effective technique classification. Kruger et al. [3] later combined traditional audio features with Hartleytransform-based features using SVM classifiers. Alar et al. [4] further explored CNN-based classification from MFCC features. While these studies demonstrate the feasibility of violin technique classification, they largely assume isolated notes or fixed-duration inputs. Recently, Wang et al. [1] proposed VioPTT, a note-level violin technique-aware transcription system using synthetic MIDI-augmentations. However, that design depended heavily on accurate note boundary detection, which remains a known bottleneck for transcription pipelines [5, 6].

## 2.2 Self-Supervised Audio Representations for Music

Self-supervised learning has significantly advanced music representation learning. Examples such as MERT [7] that allow for low-level acoustic to higher-level semantic abstractions have been used to fine-tune entire model backbones or extract single-pooled representations for downstream tasks such as audio tagging and music classification [7, 8]. Recently, MERTech [9] applied MERT to instrument playing technique detection through task-specific finetuning. It combines technqiue prediction with auxiliary tasks such as pitch and onset detection to improve robustness under limited data and class imbalance. This suggests that MERT-based representations can be effectively adapted to fine-grained technique modeling tasks.

## 2.3 Multi-Scale Temporal Modeling

Relatedly, multi-scale temporal modeling has been extensively applied to audio and music processing. Dilated convolution stacks as used in WaveNet-style architectures [10] expand receptive fields without a proportional increase in parameters. Feature pyramid networks aggregate representations at multiple resolutions for detection tasks [11]. Convolution-based systems and attention-based pooling mechanisms have been further used to capture events of varying duration [12, 13] in frame-level sound event detection as well as timbral and temporal structure in MIR tasks [14, 15].

## 2.4 Mixture-of-Experts Mechanism

While multi-scale modeling allows for abstraction across different timescales, information is usually combined within shared representations. To make temporal-scale specialization explicit and task-dependent, as well as to enhance interpretability, we therefore turn to a mixtureof-experts (MoE). The use of MoE for fine-grained music analysis has so far been limited. However, MoE architectures have been widely used in natural language processing where a gating network selects experts based on input features. This enables the handling of heterogeneous acoustic conditions such as noise environments or speaker characteristics in speech recognition [16], as well as conditional computation and specialization in Switch Transformer [17] and GShard [18].

## 3. METHODOLOGY

Our system consists of two components: (1) a transcription module that produces note-aware cues, and (2) a multiscale MoE module that integrates these cues with representations from a frozen MERT model for technique prediction. An overview is shown in Figure 2.

## 3.1 Transcription Cue Generation

We adopt the High-Resolution Piano Transcription architecture [19], repurposed for violin in [1]. The input is a mono waveform at 16 kHz. A short-time Fourier transform followed by a log-mel filterbank produces a spectrogram at 100 frames per second.

After batch normalization, three parallel branches predict frame activity, onset, and offset, each producing a tensor of shape (B, T, 88) corresponding to batch size, number of frames, and MIDI pitches 21–108. Each branch consists of convolutional blocks followed by a fully connected layer, a bidirectional GRU, and a linear projection with sigmoid activations for multi-label prediction. A GRU-based fusion stage refines the outputs by conditioning frame predictions on onset and offset signals. The outputs of these three branches are used as transcription cues to provide note-aware context for the subsequent MoE architecture.

(a)  
![](images/cb8e2964fbf37eea0c67b72006493e936574287bb087889960e17c292b2f1ab0.jpg)  
Figure 2: Proposed multi-scale mixture-of-experts model architecture for frame-level violin technique detection.

![](images/586244724adccdce9464268320450f2d82baae0bad4a4167040d57a8283a311e.jpg)  
(b)  
Figure 3: (a) Configuration and corresponding temporal context window size of our multi-scale mixture-of-experts with convolution kernel size K and dilation size D. (b) Architecture of a single expert.

## 3.2 Multi-Scale Mixture-of-Experts

## 3.2.1 Multi-Scale Temporal Experts

We build N temporal experts on top of a frozen MERT backbone. Given an input waveform, MERT-v1-330M produces $L + 1$ hidden-state tensors (here $L = 2 5 )$ , each with shape $( B , T _ { \mathrm { m e r t } } , 1 0 2 4 )$ . Each expert i first learns a set of layer weights $\mathbf { w } _ { i } \in \mathbb { R } ^ { 2 5 }$ from each hidden layer $\mathbf { H } _ { l }$ to obtain the representation

$$
\mathbf { h } ^ { \prime } { } _ { i } = \sum _ { l = 0 } ^ { 2 4 } \mathrm { s o f t m a x } ( \mathbf { w } _ { i } ) _ { l } \cdot \mathbf { H } _ { l } ,
$$

for which the resulting features are projected to 128 dimensions and temporally aligned with the transcription frames.

Temporal context is next modeled using a depthwise separable convolution with kernel size $k _ { i }$ and dilation $d _ { i }$ followed by pointwise convolution, normalization, and GELU activation. This defines a temporal context window of $k _ { i } \cdot d _ { i } \cdot 1 0 \mathrm { m s }$ , ranging from 20 ms to 2560 ms (see Figure 3a). Kernel sizes are exponentially spaced, allowing experts to specialize at different temporal scales.

Finally, features are processed by an independent bidirectional GRU to produce content representations $\mathbf { h } _ { i } \in$ $\mathbb { R } ^ { ( B , T , 2 5 6 ) }$ , along with a separate projection to gate summaries $\mathbf { g } _ { i } .$ The architecture of a single expert is shown in Figure 3b.

## 3.2.2 Per-Task Gating and Classification Heads

This stage aggregates expert outputs and produces taskspecific predictions. The onset, offset, and frame outputs from the transcription backbone are projected to 16- dimensional embeddings and concatenated to form a cue vector c. For each expert $i ,$ the content feature $\mathbf { h } _ { i }$ is concatenated with c and passed through a two-layer MLP to produce $\mathbf { e } _ { i } \in \mathbb { R } ^ { ( B , T , \bar { 1 } 2 8 ) }$ . Stacking across experts yields $\mathbf { \bar { E } } \in \mathbb { R } ^ { ( B , T , N , 1 2 8 ) }$ . Gate summaries from all experts are then concatenated with c to form $\mathbf { g } _ { \mathrm { i n } }$ . Three task-specific gating networks (tonal, articulation, legato) map $\mathbf { g } _ { \mathrm { i n } }$ to softmax weights $\pmb { \alpha } ^ { \mathrm { t a s k } } \in \mathbb { R } ^ { N }$ , which are used to combine expert features:

$$
\mathbf { m } ^ { \mathrm { t a s k } } = \mathrm { R e L U } \big ( \mathrm { L a y e r N o r m } \big ( \sum _ { i } \alpha _ { i } ^ { \mathrm { t a s k } } \cdot \mathbf { e } _ { i } \big ) \big ) .
$$

The final predictions are produced by three independent heads. The tonal head outputs three classes (normal, pizzicato, flageolet), the articulation head outputs two classes (normal, spiccato), and the legato head outputs a scalar probability via sigmoid. Legato is formulated as a regression task with Gaussian targets centered at bow change points, enabling continuous estimation of bow transitions.

## 4. VIOTECH120 DATASET

## 4.1 Technique Annotation

The VioTech120 dataset consists of 120 solo violin etudes with aligned audio and MIDI from three standard collections: 24 caprices from Paganini, Op. 1, 60 studies from Wohlfahrt, Op. 35, and 36 studies from Kayser, Op. 20 based on the ViolinMIDIDataset [20]. Technique labels are annotated at the note level using aligned MIDI as the temporal reference. Each note is assigned labels along three dimensions: tonal, articulation, and legato. For tonal techniques, we use Tonal Normal (TN), Tonal Flageolet (TF), and Tonal Pizzicato (TP). For articulation, we use Articulation Normal (AN) and Articulation Spiccato (AS). For legato, we model bow-change events using soft frame-level targets, where each annotated bow-change onset is represented by a Gaussian peak centered at the onset frame.

![](images/dbe369d3ad04be391a9108ab5fa5fa016393cbf1fd874dea7ae03245e5c3d60a.jpg)  
Figure 4: We provide a user-friendly annotation interface with fine-grained temporal alignment and control between MIDI and audio to enable data extension beyond playing technique and include additional fingering information.

This yields a probability-like target in which values close to 1 indicate bow-change frames (LB), while values close to 0 correspond to sustained frames (LS). For brevity, we use these abbreviations throughout the paper. These dimensions are not mutually exclusive: Labels are assigned in a multi-label manner, allowing multiple technique attributes to co-occur on the same note.

The annotation was carried out by two semiprofessional violinists with over ten years of formal training, and all labels are finalized through a peer review process to ensure consistency.

## 4.2 Annotation Interface

To support efficient and consistent annotation, we developed a user annotation interface based on a piano-roll visualization as shown in Figure 4. The interface allows direct interaction with the audio signal while providing an aligned view of MIDI notes, enabling annotators to assign technique labels with precise temporal control effectively and with ease. In addition to technique labels, the interface supports annotation of violin position and fingering, which facilitates future extensions of the dataset and improves usability for annotators.

## 5. EXPERIMENTAL SETTING

## 5.1 Dataset

In addition to the curated VioTech120 dataset, we incorporate MOSA-VPT [1], a large-scale synthetic dataset of violin playing techniques. MOSA-VPT contains approximately 76 hours of audio-MIDI pairs, balanced across four techniques: detaché, flageolet, spiccato, and pizzicato. Although this dataset provides only file-level technique labels, it includes a substantial number of samples for relatively scarce techniques such as flageolet and pizzicato, which complements the manually annotated VioTech120 dataset. During training, VioTech120 and MOSA-VPT data were combined at an 8:2 ratio, and the validation set followed the same proportion.

For cross-dataset evaluation, we used the RWC (Real World Computing) Music Database [21] as test data. Only violin recordings were evaluated and scales were first segmented into individual notes following prior work [1, 2].

## 5.2 Training Details

We initialized all models with a transcription backbone pretrained on VioPTT [1] and kept the MERT-v1-330M encoder frozen. Models were trained using Adam with a learning rate of $5 \times 1 0 ^ { - 4 }$ and batch size of 8. Learning rate decay and early stopping were applied based on validation performance, and the best checkpoint was selected with technique prediction macro F1. For training, we used 10- second audio segments and the same standard audio augmentations in VioPTT [1] that included pitch shifting, gain boost, reverb shifting, and random band-pass filtering.

The transcription branch was trained with regression losses for onset and offset and binary cross-entropy for frame prediction. For technique prediction, tonal and articulation tasks used cross-entropy with focal loss, together with a load-balancing regularization term. The total loss is given by

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { t r a n s c r i p t i o n } } + \mathcal { L } _ { \mathrm { t e c h } } ,
$$

where $\mathcal { L } _ { \mathrm { t e c h } }$ aggregates all technique losses.

## 6. RESULTS AND DISCUSSION

## 6.1 Frame-Level Technique Detection Performance

We first examined technique detection accuracy in Table 1 to compare with prior work that does not include legato annotations. Table 2 further reports F1 scores with legato labels included.

In the shared-label accuracy setting, VioTech achieved the highest macro accuracy of 90.49%, outperforming VioPTT (80.04%) and MERTech (52.27%). The improvement was especially clear for articulation-related techniques, where VioTech obtained the highest spiccato accuracy (AS, 97.16%), while maintaining competitive performance on tonal techniques such as pizzicato (TP) and flageolet (TF). Under the more-stringent F1-based full-task evaluation, VioTech also achieved the best overall macro F1 of 76.90%, indicating that its advantage was not limited to accuracy-based evaluation.

Compared with the non-MoE variant, we further observe that the proposed multi-scale MoE method substantially improved transient and articulation-related techniques: pizzicato F1 increased from 57.46% to 94.00%, and spiccato F1 increased from 59.78% to 90.50%. These gains suggest that multi-scale expert routing effectively captured short-term temporal dynamics and attack characteristics, which are critical for transient violin techniques.

However, bow-change detection (LB) decreased from 64.09% in the non-MoE baseline to 40.02% in VioTech.

<table><tr><td>MODEL</td><td>TN↑</td><td>TP↑</td><td>TF↑</td><td>AN↑</td><td>AS↑</td><td>MACRO ↑</td></tr><tr><td>VioTech (Ours)</td><td>79.07</td><td>89.60</td><td>88.84</td><td>97.78</td><td>97.16</td><td>90.49</td></tr><tr><td>w/o MERT Layer Weight</td><td>86.11</td><td>93.68</td><td>65.44</td><td>98.74</td><td>83.66</td><td>85.53</td></tr><tr><td>w/o Transcription Cue</td><td>79.83</td><td>95.28</td><td>83.28</td><td>99.57</td><td>64.51</td><td>84.49</td></tr><tr><td>w/o MoE</td><td>85.98</td><td>40.85</td><td>69.62</td><td>94.43</td><td>61.41</td><td>70.46</td></tr><tr><td>VioPTT [1]</td><td>77.22</td><td>88.80</td><td>71.89</td><td>77.22</td><td>85.08</td><td>80.04</td></tr><tr><td>MERTech [9]</td><td>53.36</td><td>43.27</td><td>95.77</td><td>53.36</td><td>15.61</td><td>52.27</td></tr></table>

Table 1: Technique detection accuracy (%) on the RWC test set, excluding legato for comparison with prior work. Best results are in bold, and second best are underlined.
<table><tr><td>MODEL</td><td>TN↑</td><td>TP↑</td><td>TF ↑</td><td>AN↑</td><td>AS↑</td><td>LS↑</td><td>LB↑</td><td>MACRO ↑</td></tr><tr><td>VioTech (Ours)</td><td>87.40</td><td>94.00</td><td>30.10</td><td>98.69</td><td>90.50</td><td>97.61</td><td>40.02</td><td>76.90</td></tr><tr><td>w/o MERT Layer Weight</td><td>91.28</td><td>85.35</td><td>35.29</td><td>98.34</td><td>86.46</td><td>97.89</td><td>43.42</td><td>76.86</td></tr><tr><td>w/o Transcription Cue</td><td>88.09</td><td>95.59</td><td>30.03</td><td>97.58</td><td>76.86</td><td>97.98</td><td>47.17</td><td>76.19</td></tr><tr><td>w/o MoE</td><td>88.38</td><td>57.46</td><td>30.87</td><td>94.72</td><td>59.78</td><td>98.92</td><td>64.09</td><td>70.60</td></tr></table>

Table 2: Technique detection F1 score (%) on the RWC test set with legato included.

<table><tr><td>Model</td><td colspan="4">Bach10</td><td colspan="4">URMP</td></tr><tr><td></td><td>P</td><td>R</td><td>F1</td><td>F1w/o</td><td>P</td><td>R</td><td>F1</td><td> $\mathrm { F } 1 _ { \mathrm w / \mathrm { o } }$ </td></tr><tr><td>VioPTT [1]</td><td>48.7</td><td>53.0</td><td>50.7</td><td>62.5</td><td>44.4</td><td>44.9</td><td>44.6</td><td>54.9</td></tr><tr><td>VioTech w/o MoE</td><td>54.8</td><td>58.0</td><td>56.1</td><td>65.2</td><td>39.1</td><td>57.5</td><td>46.3</td><td>59.7</td></tr><tr><td>VioTech (Ours)</td><td>60.9</td><td>65.0</td><td>62.8</td><td>79.1</td><td>45.6</td><td>60.4</td><td>51.9</td><td>69.3</td></tr></table>

Table 3: Note transcription performance on Bach10 [22] and URMP [23] datasets. P = precision, R = recall, $\mathrm { F } 1 _ { \mathrm w / \mathrm { o } }$ = F1-score without note offsets.

We attribute this drop to the sparse and long-range nature of bow-change events, which require extended temporal context and may be under-emphasized when expert routing favors shorter temporal scales. This reveals a trade-off in the current multi-scale design: expert specialization improves transient-event detection but remains less effective for sparse phrase-level phenomena.

Component ablations further show that removing the technique weighting had little effect on macro F1 (76.90% vs. 76.86%), suggesting that temporal expert specialization contributed more than layer-wise feature selection. Removing transcription cues led to a moderate decrease in macro F1 (76.19%), indicating that note-aware information provided complementary benefits. Finally, flageolet (TF) remained difficult to transcribe across all configurations, with relatively low F1 despite high accuracy, likely due to its subtle acoustic characteristics and low prevalence at the frame level.

These results show that VioTech improved cross-dataset generalization mainly by enhancing the detection of transient and articulation-related techniques, while also highlighting the need for long-context experts for bow-change modeling in future designs.

## 6.2 Note Transcription

Although note transcription is not the primary objective of VioTech, it consistently outperformed prior technique-aware transcription models on Bach10 [22] and

![](images/c105e252aab55b510c15d812ba031b5702bc84c3a6f0e733d9ac1486bcaecab5.jpg)  
Figure 5: Learned MERT layer weights by violin playing technique when evaluated on VioTech120 test (top) and RWC (bottom) datasets.

URMP [23] (Table 3), evaluated with mir\_eval [24] following the setting in VioPTT [1]. The consistent gain over the non-MoE variant further indicates that multi-scale expert modeling benefits both technique prediction and note transcription.

## 6.3 Scaling Behavior of Temporal Experts

Here, we tested how the granularity of the temporal context window of each expert affected detection performance on the single-technique level. We varied the number of experts between 2<sup>i</sup> with i = 1, 2, 3, 4, 5, for which the kernel size/temporal context window was evenly divided on the exponential scale (see Figure 3a). Results were compared against our model without any experts (hence shared representations across multiple scales) as baseline. This experiment was repeated three times for consistency.

We observe from Figure 6 that the MoE approach provided significant performance gains on expressive and less frequent techniques, such as pizzicato and spiccato, especially when evaluated cross-dataset on RWC. This suggests that expert routing helps assign temporal modeling capacity to class-specific acoustic patterns that are difficult to capture with a single shared head. Nevertheless, MoE only provided marginal improvements for frequent classes such as normal or sustained, whose performance was already near ceiling.

However, more experts did not always equate to better performance: the number of experts that enabled the highest F1 scores differed across techniques and datasets. For example, a jump in performance was seen for flageolet with two and four experts on VioTech with decreased mean performance and increased variability with increased number of experts, whereas the best performance was obtained with eight experts on RWC.

Taken together, while the proposed MoE approach improves technique detection performance and interpretability, the optimal number of experts and consequent temporal context windows seems to be dependent on idiosyncratic characteristics of each dataset.

![](images/53f570facc95ece8a210a3918e95c55db1fb30a0cc64580593b98b860638a43d.jpg)  
Figure 6: Technique-wise F1-score versus number of experts evaluated on VioTech120 test and RWC [21] datasets. Stars indicate highest mean performance over three iterations and error bars indicate standard deviation. pp denotes percentagepoint improvement over the no-MoE baseline.

## 6.4 Model Interpretation

## 6.4.1 Technique Attention Visualization

Figure 5 shows the attention attribution of each technique by a given technique. Tonal classes relied more on lowerto-middle layers, suggesting stronger dependence on local timbral cues. In contrast, articulation and legato classes placed more weight on middle layers, indicating the need for broader temporal context. These results show that playing technique benefited from different levels of MERT representation, supporting the use of learnable layer weighting in each expert.

## 6.4.2 Technique-Dependent Expert Routing

We next examined the extent each expert attended to a given technique in terms of their gating distributions on the VioTech120 test set. As shown in Figure 7, tonal techniques tended to favor experts with shorter temporal context windows (<160ms), whereas articulation and legato favored experts with longer windows. We note that around 70% of the weight was assigned to experts with temporal context windows above 320 ms when attending to legato, and a shift in the proportion of expert assignment from 320ms to 640ms for sustained vs bow-change frames. Further, we observed that pizzicato was differentiated among other tonal techniques by increased weighting at the 320 ms window.

![](images/bf5de2fd0fba5129b98e0bb10d4d5450f5aacaf057735dd3e497fd08ec42a109.jpg)  
Figure 7: Gate distribution across experts for each violin playing technique.

![](images/1a9f359291d10f35710369c460a3e0ed567540f91c64d149e400d4c35de3542c.jpg)  
Figure 8: UMAP visualization of frame-level embeddings of each playing technique from the RWC dataset [21].

## 6.4.3 UMAP Analysis

We visualize 1,000 frame-level embeddings for each playing technique on the RWC dataset using UMAP (Figure 8). The figure shows clear class-dependent structure. Pizzicato and spiccato form separated regions, reflecting their distinctive transient characteristics. Normal and flageolet partially overlap, suggesting shared bowed-string acoustic properties. Bow-change frames form separate clusters from normal frames, indicating that the model captures temporal cues related to changes in bowing. These results suggest that the learned embeddings encode meaningful differences among violin playing techniques.

## 7. CONCLUSION

Here, we presented VioTech, a frame-level violin playing technique detection model that addresses the limitations of note-level pipelines relying on onset and offset segmentation. Our mixture-of-experts approach further captured technique patterns across different temporal ranges to enable state-of-the-art transcription performance. We also introduced a manually labeled high-quality violin technique dataset from real performance recordings, together with an annotation interface for future violin technique and fingering labeling. That our model learned technique-dependent representations across feature levels and temporal scales further suggests that explicit temporal-scale specialization is a promising direction for violin technique detection.

## 8. ACKNOWLEDGMENTS

This research work is partly supported by the Academia Sinica Grand Challenge Program (GCP) Seed Grant (AS–GCS–112–M07) and National Science and Technology Council (NSTC 114-2221-E-001-011-MY2). The author thanks the anonymous reviewers for their helpful comments that improved the quality of the paper.

## 9. AI USAGE STATEMENT

This work made use of a LLM to assist with data preprocessing, code implementation, organizing the related work discussed, and writing assistance for the paper. All LLMassisted content was reviewed and verified by the authors, who take full responsibility for the accuracy and originality of the paper, including its citations.

## 10. REFERENCES

[1] T.-K. Wang, Y.-P. Peng, L. Su, and V. K. Cheung, “Vioptt: Violin technique-aware transcription from synthetic data augmentation,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 15 947– 15 951.

[2] L. Su, H.-M. Lin, and Y.-H. Yang, “Sparse modeling of magnitude and phase-derived spectra for playing technique classification,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 22, no. 12, pp. 2122–2132, 2014.

[3] A. Kruger and J. Jacobs, “Playing technique classification for bowed string instruments from raw audio,” Journal of New Music Research, vol. 49, no. 4, pp. 320–333, 2020.

[4] H. S. Alar, R. O. Mamaril, L. P. Villegas, and J. R. D. Cabarrubias, “Audio classification of violin bowing techniques: An aid for beginners,” Machine Learning with Applications, vol. 4, p. 100028, 2021.

[5] C. Duxbury, M. Sandler, and M. Davies, “A hybrid approach to musical note onset detection,” in Proceedings of the International Conference on Digital Audio Effects, 2002.

[6] M. Tomczak, C. Southall, and J. Hockman, “Onset detection for string instruments using bidirectional temporal convolutional networks,” in Proceedings of the 18th International Audio Mostly Conference, 2023.

[7] Y. Li, R. Yuan, G. Zhang, Y. Ma, X. Chen, H. Yin, C. Xiao, C. Lin, A. Ragni, E. Benetos et al., “Mert: Acoustic music understanding model with large-scale self-supervised training,” arXiv preprint arXiv:2306.00107, 2023.

[8] R. Yuan, Y. Ma, Y. Li, G. Zhang, X. Chen, H. Yin, Y. Liu, J. Huang, Z. Tian, B. Deng et al., “Marble: Music audio representation benchmark for universal eval-

uation,” Advances in Neural Information Processing Systems, vol. 36, pp. 39 626–39 647, 2023.

[9] D. Li, Y. Ma, W. Wei, Q. Kong, Y. Wu, M. Che, F. Xia, E. Benetos, and W. Li, “Mertech: Instrument playing technique detection using self-supervised pretrained model with multi-task finetuning,” in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 521–525.

[10] A. v. d. Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu, “Wavenet: A generative model for raw audio,” arXiv preprint arXiv:1609.03499, 2016.

[11] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, “Feature pyramid networks for object detection,” in CVPR, 2017.

[12] E. Çakır, G. Parascandolo, T. Heittola, H. Huttunen, and T. Virtanen, “Convolutional recurrent neural networks for polyphonic sound event detection,” in IEEE/ACM TASLP, 2017.

[13] Q. Kong, Y. Cao, T. Iqbal, Y. Wang, and M. D. Plumbley, “Sound event detection of weakly labelled data with cnn-transformer and automatic threshold optimization,” in ICASSP, 2020.

[14] J. Pons, O. Slizovskaia, R. Gong, E. Gómez, and X. Serra, “Timbre analysis of music audio signals with convolutional neural networks,” in 2017 25th European Signal Processing Conference (EUSIPCO). IEEE, 2017, pp. 2744–2748.

[15] M. Won, A. Ferraro, D. Bogdanov, and X. Serra, “Evaluation of cnn-based automatic music tagging models,” arXiv preprint arXiv:2006.00751, 2020.

[16] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, “Outrageously large neural networks: The sparsely-gated mixture-ofexperts layer,” arXiv preprint arXiv:1701.06538, 2017.

[17] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity,” Journal of Machine Learning Research, vol. 23, no. 120, pp. 1–39, 2022.

[18] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, “Gshard: Scaling giant models with conditional computation and automatic sharding,” arXiv preprint arXiv:2006.16668, 2020.

[19] Q. Kong, B. Li, X. Song, Y. Wan, and Y. Wang, “Highresolution piano transcription with pedals by regressing onset and offset times,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3707–3717, 2021.

[20] N. C. Tamer, Y. Özer, M. Müller, and X. Serra, “Highresolution violin transcription using weak labels,” in International Society for Music Information Retrieval Conference (ISMIR), 2023.

[21] M. Goto, H. Hashiguchi, T. Nishimura, and R. Oka, “Rwc music database: Music genre database and musical instrument sound database,” 2003.

[22] A. V. P. M. Dataset, “Bach10 dataset.”

[23] B. Li, X. Liu, K. Dinesh, Z. Duan, and G. Sharma, “Creating a multitrack classical music performance dataset for multimodal music analysis: Challenges, insights, and applications,” IEEE Transactions on Multimedia, vol. 21, no. 2, pp. 522–535, 2018.

[24] C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang, and D. P. Ellis, “MIR\_EVAL: A transparent implementation of common mir metrics,” in Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), 2014.