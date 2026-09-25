# Structured Pose-Conditioned Flow Matching for Generative 5G CSI Augmentation

Haojin Li, Anbang Zhang, Wai Ho Mow, Senior Member, IEEE, Chenyuan Feng, Senior Member, IEEE, Chen Sun, Senior Member, IEEE, and Haijun Zhang, Fellow, IEEE

Abstract—With the growing demand for privacy-preserving and occlusion-resilient human pose recognition (HPR), 5G channel state information (CSI) offers a promising contactless sensing modality by integrating communication and sensing capabilities. However, collecting large-scale synchronized CSI-pose pairs remains costly in practical 5G systems. To address this limitation, we propose StructFlow-HPR, a structured pose-conditioned flow matching framework for generative CSI augmentation. StructFlow-HPR learns a continuous latent transport process from Gaussian noise to real CSI representations under pose guidance, while preserving the receiver-frequency topology of CSI through a reconstruction-preserving autoencoder. A poseconditioned Transformer is further designed to model the latent velocity field and generate pose-aligned CSI samples via ordinary differential equation sampling. Experiments on real-world 5G sensing data show that StructFlow-HPR can produce realistic CSI-pose pairs and improve downstream HPR performance under limited-data conditions.

Index Terms—5G sensing, channel state information, human pose recognition, data augmentation, flow matching.

## I. INTRODUCTION

tial intelligence, high-precision human pose recognition (HPR) has emerged as a key capability for connecting the physical world with digital twin spaces [1]. Traditional mainstream sensing paradigms rely heavily on optical vision sensors (e.g., RGB and depth cameras) [2]. Admittedly, visionbased models possess intuitive advantages in capturing finegrained skeletal keypoints across continuous video sequences [3]. However, large-scale deployment in indoor scenarios is limited by line-of-sight (LoS) conditions and is susceptible to illumination changes and physical occlusions [4].

To overcome the limitations of optical sensing, contactless radio frequency (RF) sensing technologies have emerged as a viable alternative for through-wall and occlusion-resistant sensing [5]. Specifically, 5G communication networks, owing to the native integrated sensing and communication (ISAC) [6] architecture, can exploit channel state information (CSI) to accurately capture human motion-induced variations.

Unlike bandwidth-limited WiFi protocols [7], which often require specialized attention mechanisms [8] and distinct from radar systems requiring dedicated hardware arrays, 5G CSI encapsulate exceptionally rich features of high-frequency spatial multiplexing and multipath fading [9]. By resolving the microscopic phase shifts and Doppler perturbations during the reflection and scattering of 5G signals off the human body, the system can provide fine-grained spatial resolution and ultralow latency, thereby enabling high-dimensional mapping of continuous human kinematic states.

However, precisely mapping high-dimensional and complex RF physical signals into the 3D human kinematic space heavily relies on the powerful nonlinear fitting capabilities of deep neural networks. In practical deployment, this paradigm encounters severe challenges of data scarcity and severe overfitting [10]. In real-world indoor 5G communication environments, the time and labor costs associated with synchronously acquiring and annotating large-scale, high-precision data are prohibitively expensive. Thus, generative data augmentation emerges as a highly promising breakthrough pathway in modern wireless networks [11]. Rather than fitting models directly to limited raw datasets, it is more effective to synthesize massive volumes of virtual training data to overcome indoor localization and recognition bottlenecks [12]. By learning the true physical manifold distribution, models can generate diverse samples to comprehensively enrich the feature space [13]. This strategy of synergizing generative AI with physical CSI features significantly expands the decision boundaries and has the potential to alleviate the data scarcity bottleneck in HPR systems [14].

Motivated by these challenges, we propose StructFlow-HPR, a structured pose-conditioned flow matching framework for generative CSI augmentation in 5G HPR systems. The proposed method learns a continuous latent transport process from Gaussian noise to the real CSI manifold under pose guidance, while preserving the physical receiver-frequency structure of CSI measurements. By combining a topology-aware CSI autoencoder with a pose-window conditioned Transformer velocity network, StructFlow-HPR generates pose-aligned CSI samples that can be directly used to augment downstream HPR training under limited-data conditions.

## II. SYSTEM MODEL AND PROBLEM FORMULATION

## A. 5G Collaborative Sensing and CSI-Pose Representation

We consider a 5G collaborative sensing system consisting of one transmitting user equipment (UE) and multiple spatially distributed remote radio units (RRUs), which are coordinated by a baseband unit (BBU). The UE transmits uplink sounding reference signals (SRSs), and the RRUs capture the reflected and scattered signal multipath components induced by human motion. As the human body moves in the sensing area, the propagation environment changes accordingly, and such variations are embedded in the measured CSI.

![](images/d05522317d57b2d87751f6efea1249bf2d2c19dd34ec0da98b4e51996bd6883d.jpg)  
Fig. 1. Overall framework of StructFlow-HPR for pose-conditioned CSI generation and downstream HPR augmentation.

Let $h ( t , k , r ) \in \mathbb { C }$ denote the complex CSI at time index t, subcarrier index k, and receiver index r as

$$
h ( t , k , r ) = a ( t , k , r ) e ^ { j \theta ( t , k , r ) } ,\tag{1}
$$

where $\boldsymbol { a } ( t , \boldsymbol { k } , \boldsymbol { r } )$ and $\theta ( t , k , r )$ are amplitude and wrapped phase, respectively. To better capture motion-induced variations, we extract multi-domain CSI features from the raw complex measurements.

For the amplitude domain, the temporal amplitude sequence within a sliding window of length W is defined as

$$
\begin{array} { r } { A _ { W } ( t , k , r ) = \{ | h ( \tau , k , r ) | \} _ { \tau = t - W + 1 } ^ { t } . } \end{array}\tag{2}
$$

For $\mathcal { A } _ { W } ( t , k , r )$ , we further compute statistical descriptors such as normalized standard deviation (NSD), median absolute deviation (MAD), and interquartile range (IQR) to characterize local temporal fluctuations:

$$
\mathrm { N S D } ( t , k , r ) = \frac { 1 } { \mu _ { A } } \sqrt { \frac { 1 } { W } \sum _ { \tau = t - W + 1 } ^ { t } \left( \left| h ( \tau , k , r ) \right| - \mu _ { A } \right) ^ { 2 } } ,\tag{3}
$$

where $\mu _ { A }$ is the mean amplitude in the window. For phase domain, the wrapped phase $\angle h ( t , k , r )$ is unwrapped to obtain a continuous phase $\tilde { \theta } ( t , k , r )$ . Since the phase difference between receivers is more stable than the absolute phase, we compute the inter-receiver phase difference as

$$
\Delta \widetilde { \theta } _ { i , j } ( t , k ) = \widetilde { \theta } ( t , k , i ) - \widetilde { \theta } ( t , k , j ) .\tag{4}
$$

Moreover, Doppler-related motion information is derived from temporal phase variation:

$$
f _ { D } ( t , k , r ) = \frac { 1 } { 2 \pi \Delta t } \left( \tilde { \theta } ( t , k , r ) - \tilde { \theta } ( t - 1 , k , r ) \right) ,\tag{5}
$$

where $\Delta t$ denotes the sampling interval.

After preprocessing, the CSI features of each frame are organized as a structured tensor $\mathbf { X } _ { t } \ \in \ \mathbb { R } ^ { N _ { c } \times N _ { r } \times N _ { f } }$ , where

$N _ { c } , \ N _ { r }$ , and $N _ { f }$ denote the number of subcarriers, receivers, and CSI feature channels, respectively.

The synchronized human pose label is denoted by $\mathbf { Y } _ { t } \in$ $\mathbb { R } ^ { J \times K }$ , where J is the number of keypoints and K is the coordinate dimension. The aligned CSI-pose dataset is therefore defined as

$$
\mathcal { D } _ { \mathrm { r a w } } = \{ ( \mathbf { X } _ { t } , \mathbf { Y } _ { t } ) \} _ { t = 1 } ^ { N _ { \mathrm { r a w } } } .\tag{6}
$$

## B. Problem Formulation for Generative CSI Augmentation

The downstream HPR task aims to learn a regression model $f _ { \phi } : \mathcal { X }  \mathcal { Y }$ that maps CSI measurements to human poses. Given a limited number of real CSI-pose pairs, the empirical training risk is

$$
\mathcal { R } _ { \mathrm { e m p } } ( \phi ) = \frac { 1 } { N _ { \mathrm { r a w } } } \sum _ { i = 1 } ^ { N _ { \mathrm { r a w } } } \mathcal { L } _ { \mathrm { H P R } } \left( f _ { \phi } ( \mathbf { X } ^ { ( i ) } ) , \mathbf { Y } ^ { ( i ) } \right) ,\tag{7}
$$

where $\mathcal { L } _ { \mathrm { H P R } } ( \cdot )$ denotes the pose regression loss. When $N _ { \mathrm { r a w } }$ is small, the learned model tends to overfit the limited training distribution and exhibits poor generalization on unseen real CSI-pose samples.

To mitigate this issue, we formulate data augmentation as conditional CSI generation. Thus, the proposed generator learns the conditional distribution of CSI given pose semantics. Specifically, given pose conditions Y and latent noise ${ \pmb \xi } \sim \mathcal { N } ( { \bf 0 } , { \bf I } )$ , the generator produces synthetic CSI samples X aligned with the corresponding pose condition. The augmented dataset is defined as

$$
\mathcal { D } _ { \mathrm { a u g } } = \left\{ ( \hat { \mathbf { X } } ^ { ( j ) } , \mathbf { Y } ^ { ( j ) } ) \right\} _ { j = 1 } ^ { N _ { \mathrm { a u g } } } .\tag{8}
$$

The final training set is the union of real and synthetic data,

$$
\mathcal { D } _ { \mathrm { m i x } } = \mathcal { D } _ { \mathrm { r a w } } \cup \mathcal { D } _ { \mathrm { a u g } } ,\tag{9}
$$

The objective is to learn a pose-conditioned generator that can synthesize realistic CSI-pose pairs and improve the generalization ability of the downstream HPR model under limited-data conditions.

## III. STRUCTURED POSE-CONDITIONED FLOW MATCHING

## A. Structured Latent Representation

Directly modeling raw CSI tensors is challenging because of their high dimensionality and strong receiver-frequency dependency. To address this issue, StructFlow-HPR first introduces a topology-preserving autoencoder to construct a compact and structured CSI latent space. Let $E _ { \psi } ( \cdot )$ and $D _ { \psi } ( \cdot )$ denote the encoder and decoder, respectively. For a CSI frame X, the latent code and reconstructed CSI are given by

$$
{ \bf Z } = E _ { \psi } ( { \bf X } ) , \qquad { \bf X } _ { \mathrm { r e c } } = D _ { \psi } ( { \bf Z } ) .\tag{10}
$$

Moreover, the autoencoder is optimized by minimizing

$$
\mathcal { L } _ { \mathrm { A E } } ( \psi ) = \left\| \mathbf { X } - D _ { \psi } \big ( E _ { \psi } ( \mathbf { X } ) \big ) \right\| _ { 1 } .\tag{11}
$$

Specifically, the structured latent space preserves the receiver-frequency topology of the wireless measurements, thereby providing a physically meaningful manifold for subsequent generative modeling. This design allows the generator to operate on a compact representation while maintaining the topology of the original CSI measurement space.

Human motion is inherently temporal, and a single pose frame is often insufficient to describe the local motion context that causes CSI variations. Therefore, for each target CSI frame i, StructFlow-HPR constructs a pose-window condition instead of using only the current pose label. The pose-window condition is defined as

$$
\mathbf { C } _ { i } = [ \mathbf { Y } _ { i - r } , \ldots , \mathbf { Y } _ { i } , \ldots , \mathbf { Y } _ { i + r } ] , \quad r = \frac { w - 1 } { 2 } ,\tag{12}
$$

where w is the window size. After flattening, the pose-window vector is normalized and embedded by an MLP to obtain a condition representation

$$
\mathbf { h } _ { Y } = E _ { Y } ( \mathbf { C } _ { i } ) .\tag{13}
$$

Thus, this local strategy captures short-term motion continuity and provides richer semantics than a single pose snapshot. This allows the generator to learn pose-aligned CSI variations that better match the temporal evolution of human movement

## B. Pose-Conditioned Flow Matching and CSI Augmentation

Given a real CSI latent sample ${ \bf Z } _ { 1 } = E _ { \psi } ( { \bf X } )$ and a Gaussian source latent sample ${ \bf Z } _ { 0 } \sim \mathcal { N } ( { \bf 0 } , { \bf I } )$ , StructFlow-HPR defines a linear interpolation path between the source and target latent distributions:

$$
{ \bf Z } _ { t } = ( 1 - t ) { \bf Z } _ { 0 } + t { \bf Z } _ { 1 } , \qquad t \in [ 0 , 1 ] .\tag{14}
$$

The corresponding target velocity is

$$
\mathbf { u } _ { t } = \frac { d \mathbf { Z } _ { t } } { d t } = \mathbf { Z } _ { 1 } - \mathbf { Z } _ { 0 } .\tag{15}
$$

A Transformer-based velocity network $v _ { \theta } ( \cdot )$ is used to estimate the conditional transport field in the structured latent space. Specifically, CSI latent tokens are processed by selfattention, while the flow time embedding and pose-window embedding modulate the network through adaptive normalization. In this way, the model learns the conditional latent dynamics of CSI evolution under motion guidance rather than predicting diffusion noise. The flow matching objective is formulated as

$$
\mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { \mathbf { Z } _ { 0 } , \mathbf { Z } _ { 1 } , t , \mathbf { C } _ { i } } \left[ \left\| v _ { \theta } ( \mathbf { Z } _ { t } , t , \mathbf { C } _ { i } ) - \mathbf { u } _ { t } \right\| _ { 2 } ^ { 2 } \right] .\tag{16}
$$

To improve robustness, conditioning dropout can be applied during training so that the network remains stable under both conditional and weakly conditional settings. Compared with diffusion-based generation, this scheme directly learns continuous velocity field in latent space and avoids iterative reverse denoising. During generation, latent states are initialized from Gaussian noise and evolved according to the learned ordinary differential equation:

$$
\frac { d \mathbf Z _ { t } } { d t } = v _ { \theta } ( \mathbf Z _ { t } , t , \mathbf C _ { i } ) , \qquad \mathbf Z _ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf I ) .\tag{17}
$$

Using numerical ODE integration, we obtain the generated latent sample $\hat { \mathbf { Z } } ,$ which is decoded into the CSI domain as

$$
\hat { \mathbf { X } } = D _ { \psi } ( \hat { \mathbf { Z } } ) .\tag{18}
$$

Since the generation process is conditioned on the pose window $\mathbf { C } _ { i }$ , the generated CSI sample inherits the pose semantics used during sampling and forms a synthetic pair $( \hat { \mathbf { X } } , \mathbf { Y } )$ . Repeating this process yields the augmented dataset

$$
\mathcal { D } _ { \mathrm { a u g } } = \left\{ ( \hat { \mathbf { X } } ^ { ( j ) } , \mathbf { Y } ^ { ( j ) } ) \right\} _ { j = 1 } ^ { N _ { \mathrm { a u g } } } .\tag{19}
$$

Thus, the downstream HPR model is finally trained on the mixed dataset

$$
\mathcal { D } _ { \mathrm { m i x } } = \mathcal { D } _ { \mathrm { r a w } } \cup \mathcal { D } _ { \mathrm { a u g } } ,\tag{20}
$$

while performance is evaluated exclusively on held-out real CSI-pose samples to verify true generalization rather than memorization of synthetic patterns.

## IV. EXPERIMENT AND DISCUSSIONS

## A. Experimental Settings

1) 5G-enabled Prototype Platform: To rigorously evaluate the proposed scheme, we construct an indoor multi-node collaborative sensing platform. The 5G experimental infrastructure uses BS equipment from H3C, alongside Sony Xperia 1 IV smartphones serving as the transmitting UE. The core network is driven by the WX3540X, while the BS employs BBU 5200 Series. Spatially, the transmitting smartphone and three receiving RRUs are mounted on tripods and positioned at the four corners of a room, establishing a sensing area of 3.3 m×2.7 m. All transceivers are fixed at a height of 1.5 m to optimally capture human kinematic reflections and multipath perturbations within the region.

2) Data Collection and Augmentation Protocol: The system synchronously collects 5G uplink SRS-based CSI and visual ground-truth pose sequences for daily human activities, including squatting, hand raising, leg raising, leg pressing, and body translation. The visual pose sequence is temporally aligned with CSI packets according to the nearest timestamp, resulting in frame-level CSI-pose pairs. Each CSI frame is represented by multi-domain wireless features, and each pose frame contains human skeletal keypoints extracted from the synchronized visual stream.

TABLE I  
PERFORMANCE COMPARISON BETWEEN METAFI, MFDFHPR, AND STRUCTFLOW-HPR UNDER VARIOUS PCK THRESHOLDS (MOTION STATES).
<table><tr><td rowspan="2">State</td><td colspan="4">PCK5</td><td colspan="4">PCK10</td><td colspan="4">PCK20</td><td colspan="4">PCK30</td></tr><tr><td>A1</td><td>A2</td><td>A3</td><td>Δ</td><td>A1</td><td>A2</td><td>A3</td><td>Δ</td><td>A1</td><td>A2</td><td>A3</td><td>Δ</td><td>A1</td><td>A2</td><td>A3</td><td>Δ</td></tr><tr><td>Squat</td><td>67.84</td><td>76.93</td><td>77.79</td><td>+0.86</td><td>85.67</td><td>94.42</td><td>95.45</td><td>+1.03</td><td>91.46</td><td>99.77</td><td>99.84</td><td>+0.07</td><td>91.83</td><td>100.00</td><td>100.00</td><td>+0.00</td></tr><tr><td>Move</td><td>76.61</td><td>85.78</td><td>86.35</td><td>+0.57</td><td>89.58</td><td>98.39</td><td>99.08</td><td>+0.69</td><td>91.73</td><td>99.98</td><td>99.95</td><td>-0.03</td><td>92.16</td><td>100.00</td><td>100.00</td><td>+0.00</td></tr><tr><td>Rise Hand1</td><td>90.39</td><td>99.22</td><td>99.43</td><td>+0.21</td><td>91.27</td><td>99.91</td><td>99.91</td><td>+0.00</td><td>92.14</td><td>100.00</td><td>100.00</td><td>+0.00</td><td>91.76</td><td>100.00</td><td>100.00</td><td>+0.00</td></tr><tr><td>Rise Hand2</td><td>67.18</td><td>76.31</td><td>77.22</td><td>+0.91</td><td>84.53</td><td>93.29</td><td>94.14</td><td>+0.85</td><td>90.62</td><td>99.31</td><td>99.31</td><td>+0.00</td><td>91.49</td><td>99.84</td><td>99.68</td><td>-0.16</td></tr><tr><td>Press Leg1</td><td>43.82</td><td>52.99</td><td>53.63</td><td>+0.64</td><td>79.06</td><td>88.03</td><td>88.45</td><td>+0.42</td><td>88.74</td><td>97.31</td><td>97.13</td><td>-0.18</td><td>88.93</td><td>97.70</td><td>97.82</td><td>+0.11</td></tr><tr><td>Press Leg2</td><td>34.51</td><td>43.36</td><td>47.86</td><td>+4.50</td><td>65.83</td><td>74.86</td><td>79.99</td><td>+5.13</td><td>84.27</td><td>93.45</td><td>96.78</td><td>+3.33</td><td>87.91</td><td>96.94</td><td>98.78</td><td>+1.84</td></tr><tr><td>Overall</td><td>63.39</td><td>72.43</td><td>73.71</td><td>+1.28</td><td>82.66</td><td>91.48</td><td>92.84</td><td>+1.36</td><td>89.83</td><td>98.30</td><td>98.84</td><td>+0.54</td><td>90.68</td><td>99.08</td><td>99.38</td><td>+0.30</td></tr></table>

Note: A1 is MetaFi, A2 is MfDfHPR, and A3 is StructFlow-HPR. ∆ denotes the performance difference between A3 and A2. Hand1/Hand2 denote Right/Left Hand Raising, and Leg1/Leg2 denote Left/Right Leg Pressing, respectively.

![](images/8ecf79858dd1e9ca908fe7a81a5249170ff333028706a18ad0b5e052c93f5e0f.jpg)  
Fig. 2. t-SNE visualization of structured CSI latent distributions for real and StructFlow-generated samples under representative motion states

For the prohibitive data acquisition cost of authentic 5G ISAC systems, capturing the raw full-scale data for a single continuous action category inherently consumes approximately 30 minutes of data collection per action category. We construct two downstream training settings for comparison, i.e., realonly training using real samples excluding the held-out test segment and real-plus-generated training using all available real training samples and generated samples. All methods are finally evaluated on the same held-out real CSI-pose test set.

3) Compared Methods and Evaluation Protocol: To ensure fairness, we compared state-of-the-art solutions as follows:

• MetaFi [8]: A WiFi-enabled IoT human pose estimation scheme originally proposed for metaverse avatar simulation. We adapt the MetaFi method to 5G signals by using the same neural network architecture but with our small-sample 5G CSI data as input.

• MfDfHPR [15]: As our previously established framework, this model is implemented using PyTorch and trained for 100 epochs to optimize the loss function solely on the original all-sample dataset.

• StructFlow-HPR (Proposed): Built upon our previously developed MfDfHPR framework, StructFlow-HPR introduces a structured pose-conditioned flow matching generator for CSI-pose data augmentation. Specifically, the downstream pose regressor strictly adopts the same multi-feature fusion architecture as MfDfHPR, while its training data are augmented with synthetic CSI-pose pairs generated by the proposed flow matching model.

To facilitate fair comparisons, the dimensionality of the representations encoded by all methods is standardized. Furthermore, all models utilize the identical neural network backbone originally proposed in MetaFi, ensuring equivalent constraints on computational complexity and memory usage to simulate deployment on resource-limited edge devices.

## B. Experimental Results

1) Generation Quality Evaluation: Fig. 2 evaluates the generation quality of StructFlow-HPR through latent distribution visualization. Three representative motion states, including Squat, Move, and Rise Hand, are selected for illustration. For each action, real CSI samples and StructFlow-generated CSI samples are first mapped into the structured latent space by the trained autoencoder and then projected into a twodimensional space using t-SNE. As observed, the generated samples largely overlap with the real samples across different actions, indicating that StructFlow-HPR can learn the underlying pose-conditioned CSI latent distribution rather than producing isolated synthetic points. The close alignment between the two distributions further verifies that the proposed flow matching model preserves the major CSI manifold structure while introducing reasonable sample diversity for downstream HPR augmentation.

![](images/44ff00385f933a086a16a37da94643ef4bce4bcb827836b0adacbaf40a261516.jpg)  
Fig. 3. The human pose coordinates are generated by the visual model and our proposed StructFlow-HPR model, respectively.

2) Comparison With Baseline HPR Models: Table I reports the action-wise HPR performance under different PCK thresholds. A1, A2, and A3 denote MetaFi, MfDfHPR, and StructFlow-HPR, respectively, where StructFlow-HPR adopts the same pose regression backbone as MfDfHPR but is trained with additional StructFlow-generated CSI-pose samples. PCK@α measures the percentage of correctly predicted keypoints whose error is within α% of the torso length; thus, PCK@5 represents the most stringent setting, while larger thresholds evaluate more relaxed accuracy.

Compared with MetaFi, MfDfHPR achieves consistently higher accuracy on most motion states, confirming the advantage of multi-domain CSI feature modeling. Under the strict PCK@5 criterion, MfDfHPR improves the performance from 67.84% to 76.93% on Squat, from 67.18% to 76.31% on Rise Hand2, from 43.82% to 52.99% on Press Leg1, and from 34.51% to 43.36% on Press Leg2. These results indicate that the proposed CSI representation is more effective for finegrained pose estimation, especially for actions involving large body displacement or limb deformation.

A similar trend can be observed under PCK@10, where MfDfHPR improves the average accuracy from 82.66% to 91.48%. For challenging lower-limb actions, the gain is particularly clear: Press Leg1 increases from 79.06% to 88.03%, and Press Leg2 increases from 65.83% to 74.86%. When the threshold becomes more relaxed, all methods achieve higher PCK values, but MfDfHPR still maintains more stable performance across different motion states. Overall, the average PCK of MfDfHPR reaches 72.43%, 91.48%, 98.30%, and 99.08% under PCK@5, PCK@10, PCK@20, and PCK@30, respectively, demonstrating its robustness for 5G-based HPR.

3) Real-World Validation of StructFlow Augmentation: With StructFlow-based augmentation, StructFlow-HPR further improves the overall performance over MfDfHPR. The average PCK increases from 72.43% to 73.71% at PCK@5, from

91.48% to 92.84% at PCK@10, from 98.30% to 98.84% at PCK@20, and from 99.08% to 99.38% at PCK@30. The improvement is particularly evident on challenging lower-limb actions. For Press Leg2, StructFlow-HPR improves PCK@5 by 4.50%, PCK@ 10 by 5.13%, and PCK@20 by 3.33%. These results demonstrate that StructFlow-HPR can generate useful pose-aligned CSI variations that complement the real training data and enhance the generalization ability of downstream HPR models under limited-data conditions.

## V. CONCLUSIONS AND FUTURE WORK

In this paper, we propose StructFlow-HPR, a structured pose-conditioned flow matching framework for generative 5G CSI augmentation. By preserving the receiver-frequency topology of CSI signals and learning pose-guided latent transport the proposed method generates CSI samples aligned with human pose labels and improves downstream HPR performance under limited-data conditions. Future work will focus on more adaptive action-specific augmentation strategies and extension to cross-environment and multi-person 5G sensing scenarios.

## REFERENCES

[1] H. Zhang, Z. Zhang, X. Liu, W. Li, H. Li, and C. Sun, "Integrated sensing and communication for 6G holographic Digital Twins," IEEE Wireless Communications, vol. 32, no. 2, pp. 104–112, Mar. 2025.

[2] Y. Xu, J. Zhang, Q. Zhang, and D. Tao, "Vitpose++: Vision transformer for generic body pose estimation," IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 2, pp. 1212–1230, Feb. 2024.

[3] Z. Yang, A. Zeng, C. Yuan, and Y. Li, “Effective whole-body pose estimation with two-stages distillation," in IEEE Int. Conf. Comput. Vis. Workshops (ICCVW), Paris, France, Oct. 2023, pp. 4210–4220.

[4] X. Ju and et al., “Human-art: A versatile human-centric dataset bridging natural and artificial scenes," in IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Vancouver, Canada, June 2023, pp. 618–629.

[5] K. Yan and et al., “Person-in-WiFi 3D: End-to-end multi-person 3D pose estimation with Wi-Fi," in IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Seattle, WA, USA, June 2024, pp. 969–978.

[6] A. Ghosh, T. Wild, J. Du, J. Tan, A. Grudnitsky, D. Chizhik, S. Mandelli, Y. Xing, F. Schaich, and H. Viswanathan, “A unified future: Integrated sensing and communication (isac) in 6g," IEEE J. Sel. Top. Electromagn. Antennas Antennas, vol. 1, no. 1, pp. 365–374, Aug. 2025.

[7] R. Djogo and et al., “Fresnel zone-based voting with capsule networks for human activity recognition from channel state information," IEEE Internet Things J., vol. 11, no. 13, pp. 23 309–23 321, Apr. 2024.

[8] J. Yang and et al., "MetaFi: Device-free pose estimation via commodity WiFi for Metaverse avatar simulation," in IEEE World Forum Internet Things (WF-IoT), Hybrid, Yokohama, Japan, Oct. 2022, pp. 1–6.

[9] A. Kaushik, R. Singh, S. Dayarathna, R. Senanayake, M. Di Renzo, M. Dajer, H. Ji, Y. Kim, V. Sciancalepore, A. Zappone, and W. Shin, “Toward integrated sensing and communications for 6G: Key enabling technologies, standardization, and challenges," IEEE Commun. Stand. Mag., vol. 8, no. 2, pp. 52–59, May 2024.

[10] X. Yuan and Y. Qiao, "Diffusion-TS: Interpretable diffusion for general time series generation," in Int. Conf. Learn. Represent. (ICLR), Vienna Austria, vol. 2024, May 2024, pp. 41 582–41 610.

[11] W. Peebles and S. Xie, “"Scalable diffusion models with transformers," in IEEE Int. Conf. Comput. Vis. (ICCV), Oct. 2023, pp. 4172–4182.

[12] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling," in Int. Conf. Learn. Represent (ICLR), Kigali Rwanda, May 2023.

[13] X. Liu, C. Gong, and Q. Liu, "Flow straight and fast: Learning to generate and transfer data with rectified flow," in Int. Conf. Learn. Represent. (ICLR), Kigali Rwanda, May 2023.

[14] A.-A. Pooladian, H. Ben-Hamu, C. Domingo-Enrich, B. Amos, Y. Lipman, and R. T. Q. Chen, "Multisample flow matching: Straightening flows with minibatch couplings," in Proc. Int. Conf. Mach. Learn. (ICML), Honolulu, USA, July 2023, pp. 28 100–28 127.

[15] H. Li, D. Li, A. Zhang, W. Zhang, C. Sun, and H. Zhang, "No vision, no wearables: 5G-based 2D human pose recognition with integrated sensing and communications," arXiv.cs.CV.2512.24923, Dec. 2025.