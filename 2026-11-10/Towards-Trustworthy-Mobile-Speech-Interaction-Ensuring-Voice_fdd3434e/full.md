# Towards Trustworthy Mobile Speech Interaction: Ensuring Voice Authenticity and Transmission Integrity

GUODONG LIU, Nanjing University, China

LEI WANG<sup>∗</sup>, Soochow University, China

DONGXU LIU, Nanjing University, China

QIANRAN QIAO, Nanjing University, China

HAIPENG DAI<sup>∗</sup>, Nanjing University, China

HE HUANG, Soochow University, China

GUIHAI CHEN, Nanjing University, China

The rapid progress of generative speech models has made high-fidelity voice cloning and editing cheap, fast, and widely accessible, weakening the security of voiceprint-based authentication in mobile speech interactions. Existing speech water marking techniques focus on ofline protection or watermarking synthetic speech, and thus cannot verify whether a live utterance originates from the genuine speaker at capture time. We present MoBiMark, a real-time mobile biologic speech watermarking system that binds ultrasonic articulatory signatures to speech content and embeds a dynamic watermark during capture. MoBiMark emits a multi-frequency ultrasonic probe on commodity smartphones, extracts echo features from the vocal tract, and fuses them with a content-dependent speech embedding to produce a dual-layer dynamic watermark combining a persistent user identifier (UID) and a per-utterance dynamic speech identifier (SID), enabling both source and propagation-time attack detection. A jointly trained embedder and detector injects the watermark while preserving perceptual quality, and system-level optimization with INT8 quantization enables lightweight on-device deployment. Evaluated on 50 users across four smartphone models, MoBiMark achieves an average 99.79% accuracy in source and propagation attacks, with negligible impact on speech quality and intelligibility.

CCS Concepts: • Security and privacy → Security services; • Human-centered computing → Ubiquitous and mobile computing; • Information systems → Information systems applications.

Additional Key Words and Phrases: biologic speech watermarking, audio forgery, speech security, ultrasound signals

## ACM Reference Format:

Guodong Liu, Lei Wang, Dongxu Liu, Qianran Qiao, Haipeng Dai, He Huang, and Guihai Chen. 2027. Towards Trustworthy Mobile Speech Interaction: Ensuring Voice Authenticity and Transmission Integrity. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 0, 0, Article 0 ( 2027), 24 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

## 1 Introduction

Voice interaction has become a dominant interface for mobile assistants, smart speakers, and multimodal applications, yet the security of voice-based systems is increasingly fragile. Contemporary artificial intelligence (AI) generative models can clone a target speaker’s timbre from a short enrollment sample and synthesize fluent speech at scale, rendering voiceprint verification vulnerable to spoofing and content manipulation. Many voice-interactive intelligent agents are now capable of directly manipulating applications and accessing private

Authors’ Contact Information: Guodong Liu, Nanjing University, Nanjing, Jiangsu, China, EricLiu@smail.nju.edu.cn; Lei Wang, Soochow University, Suzhou, Jiangsu, China, wanglei@suda.edu.cn; Dongxu Liu, Nanjing University, Nanjing, Jiangsu, China, dongxuliu@smail. nju.edu.cn; Qianran Qiao, Nanjing University, Nanjing, Jiangsu, China, 522025330078@smail.nju.edu.cn; Haipeng Dai, Nanjing University, Nanjing, Jiangsu, China, haipengdai@nju.edu.cn; He Huang, Soochow University, Suzhou, Jiangsu, China, huangh@suda.edu.cn; Guihai Chen, Nanjing University, Nanjing, Jiangsu, China, gchen@nju.edu.cn.

data on users’ smart devices, yet they still rely on traditional voiceprint-based authentication mechanisms, posing significant privacy and security risks.

Large-scale anti-spoofing experiments demonstrate that even strong automatic speaker verification systems remain exposed to replay, conversion, and synthesis attacks across logical and physical access settings [32, 46, 51, 55]. Modern audio models built on neural architectures produce high-quality speech with low latency, while neural vocoders and end-to-end systems push naturalness closer to human recordings [19, 21, 35, 42, 48]. Voice conversion methods enable style transfer and timbre conversion with limited training data [18, 34], and

![](images/fd1f9b524e211b3d716399795c26b83a8d2a98a24f4b1d3f11594f5945a75a91.jpg)  
Fig. 1. MoBiMark for speech security in diferent scenarios.

large-model-based one-shot inference methods can forge arbitrary speech from a single target sample [9, 26, 52]. These advances drastically reduce the cost of producing convincing spoofed speech, creating a strong incentive for online real-time defenses that bind authenticity to the physical speaker.

Digital speech watermarking has emerged as a promising mitigation, with decades of work on imperceptible and robust audio watermarking [2, 8, 15, 44]. Recent learning-based watermarking frameworks improve robustness under compression and filtering [6, 10, 33, 39, 57], but they still lack broad applicability. In the face of advanced voice cloning, a watermark can be removed, replayed, or replaced by attackers. Existing methods predominantly operate ofline, focusing on protecting user-generated content after creation or verifying AI-generated output. As AI-powered speech forgery capabilities grow increasingly sophisticated, applying watermarks to protect speech at the moment of utterance emerges as a more logical and urgent approach. There is currently no work that focuses on using digital watermarking to protect user speech online in real time. A watermarking system operating on a device still cannot verify that the input speech originates from a real user, which remains one of the key factors limiting wider adoption.

We observe two core limitations of existing speech watermarking systems. First, content-only watermarks cannot verify the physical source: attackers can remove or regenerate watermarks using private models, or replay and re-edit captured speech to bypass detection. Second, most designs use static watermarks and lack reliable biometric binding in online settings, where registration requirements and device-specific keys limit deployability. A practical system must bind a watermark to the user’s biometric characteristics during capture, operate without specialized hardware, and remain robust under common signal processing attacks.

Therefore, this paper presents MoBiMark, an online real-time ultrasonic speech biometric watermarking system that binds vocal-tract motion to speech and embeds a dynamic watermark into every utterance at capture time. As shown in Fig. 1, MoBiMark emits an inaudible multi-frequency ultrasonic signal via the phone earpiece and captures its reflections through the built-in microphone. The reflected echoes encode signatures of articulator motion, which are fused with speech content to produce a biometric-dependent dual-layer watermark. The watermark contains a stable UID and a content-dependent dynamic SID to enable both source and propagation attack detection. The system can be practically deployed on users’ smartphones, enabling real-time watermark integration at the moment of utterance.

Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., Vol. 0, No. 0, Article 0. Publication date: 2027.

Although our system shows great promise, several key challenges remain to be addressed: First, how to efectively perform feature transformation and watermark encoding to ensure user speech protection throughout propagation. We propose a dual-layer dynamic watermark encoding method combining a system-assigned static UID with a speech-correlated dynamic SID. This ensures that tampered or forged speech can be reliably detected. By integrating the source user’s biometric information and security mechanisms, a closed-loop security framework is formed from the source to the propagation process. Second, how to efectively embed the generated watermark into the original speech and achieve reliable detection, ensuring that the watermark remains detectable and supports accurate discrimination after embedding. In the embedding scenario, we enhance watermark robustness against benign modifications by adjusting the speech frame ordering strategy. Furthermore, through securityoriented analysis, we incorporate an external stabilizer to perform denoising and enhancement on both the encoded speech information and the input at the detection end, thereby ensuring the stability of the generated speech features and the watermark. Finally, how to build a universal and high-accuracy system on real-world mobile smartphones, enabling online real-time use on resource-constrained edge devices. Benefit to the widespread availability of microphone hardware required for ultrasonic signals, our system inherently possesses hardware universality. Combined with our carefully designed lightweight model, which employs eficient modules and incorporates quantization to maximally compress the original system while preserving accuracy, we successfully deploy the system on mobile devices, achieving a truly universal, secure and accurate solution.

In summary, our contributions are:

• To the best of our knowledge, MoBiMark is the first real-time online speech watermarking system that combines ultrasonic biometric signals with speech for proactive speech interaction security. Due to the widespread availability of microphone sensors supporting ultrasonic signals across various edge devices, MoBiMark can seamlessly integrate with a wide range of existing smart mobile devices, enabling speech security protection in numerous application scenarios, serving as a superior alternative to voiceprint-based methods.

• We have carefully addressed three key challenges to bridge the gap between fundamental insights and practical application. These challenges span from source-level fusion to cloud-based detection, and finally to real-world deployment scenarios, thereby forming a closed-loop security framework for speech from source to propagation, as well as a practical realization from theoretical design to actual implementation. This ensures the efectiveness of the system in real-world settings such as on smartphones and mobile devices.

• We deploy MoBiMark on four smartphone models and validate it using a dataset from 50 volunteers (25 hours of positive samples, 290+ hours of negative attack samples). Through carefully designed spoofing and editing attacks, the efectiveness of our dual-layer dynamic encoding combined with biometric speech watermarking is thoroughly verified. Extensive experiments demonstrate average attack detection accuracy of 99.79% and negligible perceptual impact, bridging the gap between research on online biometric speech watermarking and practical deployment on edge mobile devices, ofering a new means for protecting users online speech.

## 2 Related Work

## 2.1 Audio Watermarking

Audio watermarking has a long history, with classic signal processing approaches embedding imperceptible information in time or transform domains [2, 3, 15, 24, 44]. Robustness and imperceptibility are typically achieved through psychoacoustic masking and transform-domain embedding, but these methods are often ofline. Recent advances include hybrid transform algorithms and learning-based embedder-detector architectures that improve robustness against common signal processing attacks [33, 39, 57]. Speech-specific systems extend these ideas to linguistic content and speaker identity [30], and timbre watermarking has been proposed to detect voice cloning attacks [25]. Recent AI-generated content watermarking emphasizes model-side identifiers during synthesis, valuable for controlled pipelines but not for human speech captured in the wild. MoBiMark difers by binding watermarks to biometric signatures at capture time with dynamic content-dependent encoding.

Table 1. Comparison between MoBiMark and other methods.
<table><tr><td>Method</td><td>Protects Speaker ID</td><td>Source-side Liveness</td><td>Protects Propagated Audio</td><td>Dynamic Speech Binding</td><td>Resistant to Speech Editing</td><td>Resistant to Replay / Clone</td></tr><tr><td>Speaker Verification (Voiceprint) [20, 41, 53, 54]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>Limited</td></tr><tr><td>Voice Liveness Detection [1, 40, 43, 56]</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>Partial</td></tr><tr><td>Audio Watermarking [6, 39, 47, 57, 58]</td><td>x</td><td>x</td><td>√</td><td>Partial</td><td>Limited</td><td>x</td></tr><tr><td>MoBiMark</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>Strong</td></tr></table>

## 2.2 Audio Forgery Detection

The quality of spoofed speech is tightly linked to progress in speech synthesis and voice conversion [18, 19, 21, 34, 35, 42, 48]. Anti-spoofing detectors face persistent generalization challenges [51, 55], and models such as RawNet2 and AASIST, while strong on benchmarks, remain vulnerable to domain shift and provide no content integrity guarantees [32, 46]. Defenses typically focus on detecting artifacts in frequency or phase statistics, but these signatures quickly become obsolete as synthesis models improve. Furthermore, detection-only systems face an asymmetry problem: false negatives may allow high-impact attacks, while false positives disrupt legitimate interactions. MoBiMark complements anti-spoofing by integrating biometric watermarking into the capture pipeline, creating an auditable authenticity trail for downstream forensics.

## 2.3 Acoustic and Ultrasound Sensing

Ultrasound and active acoustic sensing have enabled contactless interaction and authentication on commodity devices. EchoWhisper uses ultrasonic echoes to infer silent speech on smartphones [11]. VocalLock leverages acoustic sensing of the vocal tract for passphrase-independent authentication [28]. Speaker-Sonar employs inaudible sonar to verify voice command liveness against remote attackers [22]. TouchPrint demonstrates active acoustic sensing for geometric biometric authentication [7]. Compared with camera-based biometrics, ultrasound sensing is insensitive to lighting, operates unobtrusively with existing hardware, and captures subtle articulatory dynamics dificult to spoof remotely. MoBiMark builds on these insights by fusing ultrasonic biometric features with speech to produce a propagation-protecting watermark, forming an end-to-end security framework.

## 2.4 MoBiMark Diference

Our watermarking system prevents source forgery through biometric authentication at the point of capture and protects speech during transmission using dynamic watermarking together with a corresponding detection mechanism. As a result, only speech carrying the correct watermark is accepted as authentic. As shown in Table 1, compared to conventional speaker verification (voiceprint) schemes [20, 41, 53, 54], our method encompasses both the source and propagation stages, actively embedding a biometric watermark at capture time. Compared to voice liveness detection schemes [1, 40, 43, 56], MoBiMark overcomes the limitation of verifying identity only at acquisition time by embedding dynamic speech-related watermarks that enable authenticity traceability throughout transmission, storage, and redistribution, while leveraging large-scale pre-trained speech model generalization to achieve cross-user capability without capturing a range of users’ data for retraining. Compared to traditional audio watermarking [6, 39, 47, 57, 58], MoBiMark uniquely binds source-side liveness information to propagated speech, addressing the fundamental gap that existing watermarks cannot verify whether input speech originates from a real user. In summary, unlike all previous methods, we propose an integrated speech biometric watermarking system that protects all speech from the target human, from source to subsequent dissemination.

## 3 Threat Model

## 3.1 Security Goals and Trust Assumptions

MoBiMark aims to preserve speech authenticity throughout acquisition, transmission, and subsequent propagation by providing two security guarantees: (1) source authenticity, ensuring that speech originates from a legitimate enrolled speaker, and (2) propagation integrity, ensuring that propagated speech remains consistent with the originally generated content.

We assume that the user’s smartphone, including the microphone, ultrasonic sensing module, operating system, and local watermark generation pipeline, operates correctly and is not compromised. Sensor measurements are acquired through the device’s normal sensing interfaces. In particular, the adversary cannot directly tamper with sensor hardware or drivers, bypass the sensing pipeline to inject fabricated digital measurements, or otherwise manipulate the trusted acquisition process. These assumptions define the trusted hardware and sensing boundary of MoBiMark. Attacks that compromise this boundary are outside the scope of our watermarking system.

The ultrasonic evidence used by MoBiMark is generated by the smartphone’s own speaker and captured by its microphone through the normal acoustic propagation path. The acquisition and binding of speaker-related biometric characteristics and the corresponding ultrasonic measurements are performed within the trusted sensing pipeline. We assume that the raw ultrasonic measurements and the internal correspondence between the target speaker’s biometric characteristics and the captured ultrasonic responses are protected from direct adversarial access. In particular, the adversary cannot obtain target-specific paired speech-ultrasound observations suficient to reconstruct the correspondence exploited by MoBiMark. This assumption concerns the protection of the trusted acquisition and binding process, rather than the physical impossibility of external ultrasonic interference.

The cloud verification service is trusted and follows the prescribed protocol, while communication among trusted components is protected using standard transport-layer security mechanisms. We further assume that user enrollment is performed in a trusted environment, consistent with existing biometric authentication systems such as Face ID and fingerprint authentication. Since the security of MoBiMark fundamentally depends on the correctness of the enrolled identity, impersonation during enrollment compromises the root of trust rather than exploiting a vulnerability in MoBiMark. Moreover, attacks against MoBiMark are meaningful only after a user has enrolled and enabled watermarking. We therefore consider enrollment attacks, as well as full compromise of trusted hardware or software, outside our threat model.

## 3.2 Atack Model

We consider a strong adversary who knows the general working principle the MoBiMark architecture. The adversary may obtain previously published speech recordings, corresponding watermarked speech, and access to the static detector implementation. Notably, the static detector difers from our cloud-based dynamic detector in that it can only detect the statically assigned watermark key, consistent with watermark detection schemes in most prior studies. In contrast, detection of the dynamic watermark is performed exclusively within our trusted cloud service, and the dynamic detection mechanism remains undisclosed to the public.

The adversary can perform replay attacks, voice cloning, speech conversion, speech editing, watermark reuse, and other digital manipulations in an attempt to bypass the verification process. We further consider adaptive adversaries who are aware of the speech-ultrasound consistency constraints exploited by MoBiMark. Such adversaries may attempt to jointly construct synchronized speech and ultrasonic signals, replay forged speech while presenting artificial ultrasonic signals, or reuse previously observed content to maximize their acceptance probability. However, these attacks are constrained by the trusted acquisition and binding assumptions described above.

The adversary cannot directly inject fabricated measurements into the smartphone’s sensor data stream or access the internal correspondence between the target speaker’s biometric characteristics and the captured ultrasonic responses. Consequently, attacks that require eavesdropping on the target speaker’s ultrasonic responses or collecting suficient target-specific paired speech-ultrasound observations to train a model that reproduces this correspondence are outside our threat model. Similarly, the adversary is assumed not to have access to synchronized speech-ultrasound pairs collected during legitimate watermark generation when such access would expose the protected correspondence. These restrictions do not imply that external ultrasonic signals are physically incapable of reaching the microphone; rather, they reflect the protection of the trusted acquisition and binding process.

Accordingly, MoBiMark focuses on detecting forged or manipulated speech and preserving the association between legitimately captured speaker-related biometric signals and the resulting watermarked speech throughout subsequent propagation. Direct physical attacks on trusted sensors, malicious modification of sensor drivers or firmware, privileged compromise of the operating system, and other attacks that violate the trusted sensing boundary are outside the scope of this work. In addition, strengthening hardware-level encryption and data-flow integrity protection is a promising direction for further hardening the system against unauthorized access to or injection of internal hardware data.

![](images/515c1d4d14a33746a5cbdd231820ab6b78de860aa46cbcf39715c7a4ae465a9a.jpg)  
Fig. 2. Overall architecture of MoBiMark.

## 4 System Design

MoBiMark is deployed on the client side of users’ mobile devices. The overall workflow, as illustrated in Fig. 1, involves the following components.

1. User Registration: Before using the system, a user registers their voice and ultrasonic data with a single arbitrary sample, suficient due to the pre-trained encoder’s generalization capability. Specifically, the registered samples are converted into compressed feature vectors, stripped of the user’s biometric information, and then uploaded to a trusted cloud for subsequent watermarking and detection.

2. UID Assignment: After registration, the system automatically assigns a static 64-bit UID code, which will be combined with the subsequent dynamic SID.

3. SID Generation: During actual voice interaction, MoBiMark automatically and dynamically generates a content-dependent 192-bit SID for each utterance. The encoded information is strongly correlated with the speech features; if the speech is modified, the encoded value changes accordingly, enabling efective real-time dynamic protection.

![](images/c2f55efeef9ac7a73097d4e5f9c93f1ede3f17a983b6530421a2003f7c83c53e.jpg)  
Fig. 3. Speech feature clusters for diferent registered users.

4. Watermark Detection: For simple verification scenarios, the detector can use the UID locally, similar to mainstream static watermarking schemes. For forged speech that has been meticulously modified or edited by an attacker, the speech only needs to be uploaded to the cloud, where the system regenerates the corresponding watermark information based on the current speech and compares it with the embedded watermark, returning the comparison result.

As shown on the left of Fig. 2, the architecture consists of the Watermark Generator, the Embedder Encoder/Decoder, the Detector and Stabilizer modules. Specifically, the concurrently captured speech and ultrasound signals are fed into MoBiMark. They are first examined and compared by the Watermark Generator module; if a live user is verified, a dual-layer dynamic watermark code (UID + SID) is generated based on both signals and then combined with the intermediate speech features produced by the Embedder Encoder. The result is decoded by the Embedder Decoder into a watermarked speech signal that preserves the original speech quality. The subsequently deployed Detector extracts the embedded watermark information. The Stabilizer is applied within both the Watermark Generator and the Detector to maintain the stability of the watermark features. Benefiting from the ubiquity of microphone sensors required for ultrasound on all mobile phones, MoBiMark exhibits a high degree of universality.

## 4.1 Watermark Generation

4.1.1 Speech Encoder. The design goal of MoBiMark is a novel watermarking system capable of integrating with speech information while ensuring real-time security at the source. We abstract the most fundamental requirements of the system: it must be able to distinguish between a large number of newly registered diferent users, ensuring feature independence among them. Simultaneously, it should also diferentiate between diferent speech utterances from the same user, in order to meet the subsequent demands of dynamic watermarks that are strongly correlated with speech content. During training, the system utilizes speech data from known users to construct an efective feature space. This feature space must ensure that speech features from diferent users form distinct clusters, while also allowing diferentiation among diferent speech features from the same user, as illustrated in the Fig. 3.

From the perspective of theory, the speech representation learning task addressed by this system can be modeled as a joint optimization problem of category discrimination and sample diversity. Let the high-dimensional observed feature embedded by the model be $X \in \mathbb { R } ^ { D }$ , and the category label be �. The learning objective can be uniformly formulated as maximizing a weighted combination of the category mutual information and the within-class conditional entropy:

![](images/1b3ce2d093b5b07e96a994d846bed998eb07447611619f7890bb3777f5b8eb59.jpg)  
Fig. 4. Overall architecture of the Watermark Generator.

$$
\operatorname* { m a x } _ { \theta } ~ I _ { \theta } ( X ; Y ) + \lambda \cdot \mathbb { E } _ { Y } [ H _ { \theta } ( X \vert Y ) ]\tag{1}
$$

Here, $I _ { \theta } ( X ; Y )$ represents the discriminative information associated with the category. Its maximization ensures significant distinguishability between diferent identity categories. Conversely, the conditional entropy �<sub>�</sub> (� |�) measures the uncertainty and richness of the sample distribution within a given category. Its non-degeneracy constraint guarantees suficient diversity among diferent samples belonging to the same class.

However, a feature space derived solely from this optimization objective is limited by our finite collection of sample data and cannot form a stable and efective representation. Consequently, it fails to generalize well when encountering new users. We aim to train such an ultra-large-scale and stable vocal timbre feature space on a large number of diferent users, and ensure that newly registered users are efectively assigned to fixed regions within this feature space.

Inspired by speaker recognition tasks [5, 13, 49], we observe that for such tasks, speech from diferent users forms distinct clusters in the feature space, with each cluster representing unique vocal timbre information. However, due to the leakage of semantic information in speech, diferent utterances from the same user do not completely converge to a single point; instead, they distribute at diferent locations within the cluster. While this is considered a drawback in speaker recognition tasks, it precisely aligns with the requirements of our task. We leverage the encoder of the well-established CAM++ model [49], which is trained on a large-scale speaker dataset, as our Speech Encoder $E _ { s }$ . The CAM++ is trained on a dataset comprising over 200,000 user speech samples, enabling it to learn a broad range of vocal timbre information and achieve efective high-dimensional space embeddings for the vocal timbre of all users. Nevertheless, it is extremely lightweight and highly suitable for our system. The 512-dimensional high-dimensional space output by CAM++ provides suficient representational capacity for the aforementioned objective. The high dimensionality supports the formation of adequate inter-class informational separation, thereby increasing $I _ { \theta } ( X ; Y )$ . Simultaneously, it allows each class to occupy a nondegenerate support region on the high-dimensional manifold, thereby maintaining a relatively large $H _ { \theta } ( X | Y )$ and preventing excessive convergence of representations within the same class. Therefore, under this theoretic framework, the model achieves robust and highly discriminative speech representation learning by simultaneousl enhancing inter-class separability and intra-class diversity within the high-dimensional space.

4.1.2 Ultrasound Encoder. We encode the speech modality using $E _ { s } ,$ and simultaneously encode ultrasound signals with $E _ { u } ,$ , as shown in Fig. 5. To ensure the generalization ability of MoBiMark for newly registered users, it is necessary for $E _ { u }$ to also exhibit generalization beyond the limited training samples. If training is performed only on a limited number of data samples, the feature space learned by the model is confined to the regions corresponding to these samples. When a new user registers, the feature space allocated to them tends to overlap with that of the training users, making efective diferentiation impossible. Therefore, we align ultrasound signal features and speech signal features in the high-dimensional feature space. Since $E _ { s }$ is trained on a very large-scale speech dataset, it can form an extremely large and stable feature space. After aligning the speech features with the corresponding ultrasound signal features in this space, the ultrasound features are also able to learn an efective feature distribution within the corresponding feature space. Compared to partitioning the feature space based on limited data, this approach substantially expands the scope of the feature space, ensuring that when a new user registers, the region assigned to their features is suficiently distinct from those of other users.

![](images/374ec4c9ed03ab475022897386a9fd34ada8bb72ca0684bc5393fdb09ed556e7.jpg)  
Fig. 5. Architecture of ultrasound Encoder and Decoder.

We train $E _ { u }$ using cosine similarity loss $\mathcal { L } _ { \mathrm { c } }$ with the output of $E _ { s } ,$ , combine with an MSE reconstruction loss $\mathcal { L } _ { \mathrm { u } }$ with decoder $D _ { u }$ (the left in Fig. 4):

$$
\mathcal { L } _ { \mathrm { s 1 } } = \lambda _ { c } \mathcal { L } _ { \mathrm { c } } + \lambda _ { u } \mathcal { L } _ { \mathrm { u } }\tag{2}
$$

with $\lambda _ { c } = 1 . 0$ and $\lambda _ { u } = 0 . 2$ . We use $\mathcal { L } _ { \mathrm { c } }$ to align the ultrasound and speech features, which is designed to train $E _ { u }$ to cluster samples from the same user together while separating those from diferent users, thereby achieving user discrimination. To ensure that samples within the same cluster remain distinguishable, a prerequisite for constructing the SID, the encoder output features are simultaneously passed through a weak decoder $D _ { u }$ to reconstruct the original ultrasound information. This constitutes an adversarial training scheme: the weak decoder forces the model to preserve intra-class variations while still learning discriminative inter-user boundaries, so that distinct user clusters form without causing feature collapse within each class.

As shown in Fig. 5, $E _ { u }$ processes 32-channel raw waveform segments (shape [B, 32, 3000×�]) adapted for ultrasound signals, where 3000 is the ultrasonic sampling rate and $T$ is the time length. The encoder first performs local pre-encoding through a convolutional stem $( 3 2 \to 9 6 , k = 9 )$ , followed by four stacked multi-kernel residual blocks (MKRB) that employ strides of5,5,4,3 to achieve a total temporal downsampling factor of300×, compressing the time length from 3000 ×� to 10 ×�. Each MKRB consists of three parallel convolutional branches (kernel sizes of 3, 9, 17) to explicitly capture short-range, medium-range, and long-range waveform patterns. After branch fusion, the features undergo convolutional post-processing, followed by the addition of a 1 × 1 residual projection, and are then adaptively re-calibrated via a channel attention gating mechanism (compression ratio 8), achieving a balance between multi-scale modeling and channel selection. On this basis, we cascade two temporal Transformer modules (TTB, embedding dimension 512, 8 attention heads, dropout 0.1) to further model global dependencies across segments. After temporal average pooling, a two-layer projection head (Linear–LayerNorm–GELU–Linear) produces a 512-dimensional discriminative vector. The decoder $D _ { u }$ employs a four-stage deconvolutional pathway (strides 3,4,5,5) to progressively restore temporal resolution, using linear interpolation for length alignment when lengths are not evenly divisible. This design concurrently addresses identity discrimination capability and temporal structure fidelity.

4.1.3 User Verification and Dynamic Watermark Generation. Overall, our model training is divided into two stages, as shown in the Fig. 4. The first stage mainly includes the design and training of the aforementioned $E _ { s }$ and $E _ { u }$ modules, which are used to efectively encode speech and ultrasound signals into a feature space that allows for comparison and fusion. The features obtained from the two modalities are then used for subsequent user verification and dynamic watermark generation.

Verification: As shown on the right side of the Fig. 4, the output vectors of the two encoders are first transformed by the Adapter module to adapt to the encoding dimension required in later stages. The two vectors are then processed by the Fusion module, which outputs the fused feature vector $f _ { i } ^ { j }$ . To enable efective dimension conversion for new registered users and to assign the fused feature vector to an appropriate location in the fused feature space, we employ a triplet loss function $\mathcal { L } _ { \mathrm { t } }$ for metric learning. This loss function pulls features of the same user closer in the feature space, while pushing features of diferent users farther apart. The formula for this loss function is as follows:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t } } = \operatorname* { m a x } \left( \| \mathbf { x } _ { a } - \mathbf { x } _ { \mathcal { P } } \| _ { 2 } ^ { 2 } - \| \mathbf { x } _ { a } - \mathbf { x } _ { n } \| _ { 2 } ^ { 2 } + \alpha , 0 \right) } \end{array}\tag{3}
$$

Where $\mathbf { x } _ { a }$ is feature of the anchor point, $\mathbf { x } _ { p }$ is positive sample feature same as user of anchor point, ${ \bf x } _ { n }$ is negative sample feature diferent from user of anchor point, and � is the margin, which controls the minimum distance between the positive and negative sample pairs. Benefiting from the pre-trained modality encoders from the previous stage, the modules trained in this stage can easily cluster efective sample pairs of the two modalities from the same user, while efectively separating clusters from diferent users. This results in features that match the watermark encoding dimension, as well as corresponding fused features.

Before fusion, we incorporate symmetric adapters for the ultrasound and audio modalities, respectively, in order to mitigate the statistical distribution gap between heterogeneous modalities and align the subsequent encoding dimensions. As shown in Fig. $^ { 6 , }$ each Adapter module employs a lightweight structure of Linear–LayerNorm–GELU–Linear, while preserving the original temporal length unchanged Through this module, we can adaptively adjust the number of watermark encoding bits without afecting the training of other modules. We set the output dimension to 192, which is the same as the encoding dimension of the SID.

For the Fusion module, we adopt a bidirectional multi-head cross-attention mechanism, with an attention embedding dimension of 192 and 8 heads, enabling the ultrasound branch to query audio context and the audio branch to query ultrasound context in reverse, thereby explicitly modeling a bidirectional complementary relationship. To suppress cross-modal noise propagation, a gating unit is applied after the attention outputs. The gating network takes the concatenated representation of the original features and the cross-attention features as input, with an input channel size of 384, and outputs gating coeficients of 192 dimensions. Residual connections and LayerNorm are further employed to enhance training stability. In the global aggregation stage, the module pools the two sequences separately to obtain two 192-dimensional global vectors (� and �) and further constructs three types of relational features (multiplicative interaction $s \odot u ,$ , diference response $s - u$ , and dominant response ���{�, �}), each also 192-dimensional. The final concatenated fusion vector has a channel dimension of 960 (including � and �). A metric head projects it as $9 6 0 { \longrightarrow } 3 8 4 { \longrightarrow } 1 9 2 { \longrightarrow } 1 9 2$ , and incorporates LayerNorm, BatchNorm, and Dropout layers. This module ultimately outputs a fused embedding vector for the two modalities, with a dimension of 192, which is then used to match the final watermark encoding.

Using the well-trained features described above, we can obtain speech and ultrasound features of diferent users. It is only necessary to construct the corresponding logic to determine whether the input speech signal originates from the genuine target user. The specific algorithmic logic is shown in the Algorithm 1.

The system first compares the input speech feature $\mathbf { s } ^ { j }$ with the enrolled speech features $\{ \pmb { s } _ { i } ^ { 0 } \} _ { i = 1 } ^ { N }$ to identify the best-matching user index � and the corresponding similarity score best\_sim. If best\_sim $< \gamma _ { \mathrm { e n r o l l } }$ , the system identifies the input as belonging to an unregistered user and triggers the initial enrollment process, returning (False, 0). Otherwise, ifbest\_sim $< \gamma _ { s } ,$ the system rejects the verification and returns (False, −1). For a candidate that passes speech verification, the system further computes the similarities $S i m _ { u }$ and $S i m _ { f }$ between the input ultrasound and fused features, $( \mathbf { u } ^ { j } , \mathbf { f } ^ { j } )$ , and the corresponding enrolled features, $( \mathbf { u } _ { i } ^ { 0 } , \mathbf { f } _ { i } ^ { 0 } )$ . The verification succeeds only if both $S i m _ { u } \geq \gamma _ { u }$ and $S i m _ { f } \geq \gamma _ { f }$ , in which case the system returns (True, �); otherwise, it rejects the input and returns (False, −1). This hierarchical decision process uses speech similarity for candidate identification and preliminary verification, followed by joint ultrasound and fused-feature verification to strengthen authentication reliability. In our experiments, we se $\gamma _ { s } = \gamma _ { u } = \gamma _ { f } = 0 . 5$ and $\gamma _ { \mathrm { e n r o l l } } = 0 . 1$ . If all conditions are satisfied, the system authenticates the target user and embeds a dual-layer dynamic watermark into the speech. Otherwise, watermark embedding is rejected. Since downstream processing treats speech without a valid watermark, including speech carrying an incorrect watermark, as invalid, only authenticated speech with valid watermark information is accepted, ensuring source-level speech security.

![](images/6d8f64aa29be93f73b490a7b88fbd323a9ad8f8eded15b77828dd1a474dac764.jpg)  
Fig. 6. Architecture of Adapter and Fusion modules.

Algorithm 1 Target User Verification   
Require: Speech $\mathbf { s } ^ { j } ,$ , ultrasound $\mathbf { u } ^ { j } ,$ fused $\mathbf { f } ^ { j } ;$ enrolled features $\{ ( \mathbf { s } _ { i } ^ { 0 } , \mathbf { u } _ { i } ^ { 0 } , \mathbf { f } _ { i } ^ { 0 } ) \} _ { i = 1 } ^ { N }$   
Require: Thresholds �<sub>�</sub>, �<sub>�</sub>, �<sub>�</sub>, �<sub>enroll</sub>   
Ensure: Verification result   
1: Find best-matching user: best\_sim ← max Sim $( \pmb { s } ^ { j } , \pmb { s } _ { i } ^ { 0 } )$ , � ← arg max   
2: if best\_sim < � then return (False, 0) ⊲ No match, enroll   
3: end if   
4: if best\_sim $< \gamma _ { s }$ then return (False, -1)   
5: end if   
6: $S i m _ { u } \gets \mathrm { S i m } ( \mathbf { u } ^ { j } , \mathbf { u } _ { i } ^ { 0 } )$ , ���<sub>�</sub> ← Sim $( \mathbf { f } ^ { j } , \mathbf { f } _ { i } ^ { 0 } )$   
7: if $S i m _ { u } \geq \gamma _ { u }$ and ��� $\iota _ { f } \geq \gamma _ { f }$ then return (True, �)   
8: elsereturn (False, -1)   
9: end if

Watermark Generation: Beyond protecting speech at the source, it is essential to preserve its authenticity and integrity during transmission. Existing watermarking schemes often rely on static watermarks or lack real-time dynamic verification, limiting their efectiveness against replay and content-tampering attacks. To address these limitations, MoBiMark introduces a biologically grounded, dual-layer dynamic watermark comprising a static UID and a dynamic SID. The UID is a system-assigned 64-bit code that supports lightweight identity verification, while the 192-bit SID provides stronger protection for security-critical applications.

For each speech sample $\mathsf { S } _ { i } ^ { j }$ and its synchronized ultrasound signal $\mathbf { U } _ { i } ^ { j } ,$ , pretrained encoders extract speech and ultrasound features, which are aligned through a dimension transformation module (DTM) and fused into a multimodal representation. The SID is then generated by combining the normalized diference between the current speech feature and the user’s registered speech feature with the normalized registered fusion feature, followed by nonlinear activation and binarization:

$$
{ S I D } _ { i } ^ { j } = \mathrm { s i g n } \big ( \operatorname { t a n h } \big ( \operatorname { n o r m } ( \hat { \mathbf { s } } _ { i } ^ { j } - \mathbf { s } _ { i } ^ { 0 } ) \odot \operatorname { n o r m } ( \mathbf { f } _ { i } ^ { 0 } ) \big ) \big )\tag{4}
$$

The final watermark is $\mathbf { B } _ { i } ^ { j } = \mathrm { C o n c a t } ( U I D _ { i } , S I D _ { i } ^ { j } )$ . By binding a persistent user identity to sample-specific speech and biometric features, the dual-layer design supports eficient identity verification while enabling dynamic content-integrity checks. Because the SID depends on the utterance and registered biometric information, semantic modifications (e.g., deleting “not”) can be detected by checking the consistency between the received speech and its SID, while unauthorized speech cannot obtain a valid watermark through the source authentication process.

## 4.2 Watermark Embedding and Detection

After generating the dual-layer dynamic speech biometric watermark, it is embedded into the speech signal while ensuring imperceptibility to human listeners and ASR systems, tamper sensitivity, and robustness against benign signal processing (e.g., resampling, filtering, and compression).

4.2.1 Speech Stabilizer. As shown on the right side of Fig. 2, the Speech Stabilizer is shared by the watermark generation and detection stages to improve robustness against benign speech modifications, such as mild noise and down-sampling. During watermark generation, it denoises the input speech to obtain more stable speech and fusion features, improving subsequent cloud-based watermark detection. Importantly, denoising is applied only to the speech fed into the watermark generation module; the original speech remains unchanged to preserve its authenticity and contextual information.

During cloud-based detection, the stabilizer processes the potentially modified speech before regenerating the dynamic watermark for comparison with the watermark embedded in the speech. This design compensates for the sensitivity of the speech encoder $E _ { s }$ to benign modifications, which enables dynamic watermark generation and helps defend against editing and forgery attacks. Meanwhile, joint anti-interference training allows the detector to directly recover watermark information despite such modifications. The stabilizer is optional: when tolerance to subsequent modifications is not required, any modification after watermark embedding can instead be treated as potential forgery, and the stabilizer can be omitted.

The stabilizer adopts a U-Net-like encoder-decoder architecture. A multi-scale convolutional stem (MSS; kernel sizes 3, 7, and 15) extracts local acoustic features, followed by three stride-2 convolutional layers with depthwise-separable residual blocks and increasing dilation rates. Four lightweight Conformer blocks form the bottleneck, combining feed-forward networks, windowed multi-head self-attention, and depthwise-separable convolution to capture contextual information with reduced sequence complexity from $O ( T ^ { 2 } )$ to $O ( T w )$ , where � is the sequence length and � is the attention window size. The decoder uses transposed convolutions and encoder skip connections to reconstruct the enhanced waveform through residual learning. Training minimizes a weighted combination of waveform MSE and Mel-spectrogram MSE, where the Mel spectrogram uses a 25-ms Hann window, a 10-ms frame shift, and 80 Mel filter banks:

$$
\mathcal { L } _ { \mathrm { s t a b i l i z e r } } = \lambda _ { m w } \mathcal { L } _ { \mathrm { m w } } + \lambda _ { m m } \mathcal { L } _ { \mathrm { m m } } ,\tag{5}
$$

where $\lambda _ { m w }$ and $\lambda _ { m m }$ control the respective loss weights, and we set 10 and 1, respectively, by experimental training experience.

4.2.2 Watermark Embedding. We follow the design of AudioSeal [39] and extend its original 16-bit watermark to a dynamic 256-bit watermark that can be embedded within a single utterance, drawing inspiration from

![](images/d8a155052d9d63cc22f7daabb09f55271025583019b5646cbe732af9bcff30aa.jpg)  
Fig. 7. Chunk-based vs. group-based watermark framing.

SpeechVerifier [57]. Specifically, we adopt a temporal framing strategy that divides the watermark into 16 frames, each encoding 16 bits. A spoken word typically lasts approximately 250 ms [23], corresponding to around 4,000 audio samples at a sampling rate of 16 kHz. This provides the temporal context for embedding the complete watermark within an utterance.

As illustrated in the upper part of Fig. 7, a straightforward approach is to arrange the 16 watermark frames consecutively within each chunk, which we refer to as chunk-based framing. However, uniform downsampling may consistently remove the same relative frame positions across chunks, potentially causing systematic information loss and compromising watermark recovery. To mitigate this vulnerability, we propose group-based framing, illustrated in the lower part of Fig. 7. Specifically, each of 16 consecutive chunks carries one of the 16-bit watermark segments, repeated across 16 consecutive frames within that chunk. These 16 chunks form a group of 256 frames, across which the complete 256-bit watermark is distributed through temporal interleaving. By spreading watermark information across multiple chunks, this design is expected to improve tolerance to uniform downsampling and reduce the risk of systematic information loss caused by the removal of fixed relative frame positions. However, it does not guarantee the recovery of every watermark bit under all downsampling conditions.

Beyond the encoding strategy, the Perceptual Loss and Detection Loss play a critical role in balancing watermark imperceptibility, robustness to benign speech modifications, and sensitivity to malicious manipulation. We adapt the loss designs of AudioSeal and SpeechVerifier [39, 57] to jointly support these objectives.

4.2.3 Watermark Detection. The watermark detection component is primarily divided into edge-side and cloudside detection, corresponding to the detection of static UIDs and dynamic SIDs, respectively.

Edge Static Detection: Deployed on edge devices, this detection method resembles most existing watermark detection endpoints and is applicable only to simple or real-time, non-falsifiable security scenarios, such as in smart Internet of Things (IoT) device security detection. By verifying static UIDs, edge devices can determine whether the current speech originates from the target user. In practice, certain benign modifications may afect detection accuracy, leading to discrepancies between the extracted watermark code and the true code. Therefore, the Hamming distance [12] is employed to measure similarity by counting difering bit positions; a smaller value indicates higher similarity. The specific discrimination threshold can be set by the user.

Cloud Dynamic Detection: However, in more complex scenarios requiring higher security, detection based solely on static UIDs is insuficient. In high-security scenarios, compared to traditional watermarking systems, MoBiMark ofers distinct advantages. Since the system ensures correctness from the source, it operates as a closed-loop system. For dynamic watermarking, the speech to be verified can be uploaded to the cloud for dynamic detection. The cloud system then returns a determination to the user regarding whether the speech originates from a genuine user. The principle of cloud detection involves generating corresponding watermark information from the speech to be verified through the Watermark Generator module. This generated watermark is then compared with the dynamic watermark code (SID) embedded in the speech. A match indicates authenticity; otherwise, the speech is considered inauthentic. In the cloud detection stage, the Stabilizer is utilized to recover speech that has undergone benign modifications, thereby facilitating more accurate discrimination of watermark information.

Table 2. Module sizes before and after quantization.
<table><tr><td>Module</td><td>FP32 (MB)</td><td>INT8 (MB)</td></tr><tr><td>Watermark Generator</td><td>81.37</td><td>24.97</td></tr><tr><td>Embedder</td><td>56.17</td><td>21.57</td></tr><tr><td>Detector</td><td>33.12</td><td>12.29</td></tr><tr><td>Stabilizer (Optional)</td><td>84.00</td><td>23.88</td></tr><tr><td>Total</td><td>170.66 (+84.00)</td><td>58.83 (+23.88)</td></tr></table>

## 4.3 Lightweight Deployment

Model quantization is an efective approach to edge deployment, reducing memory footprint and potentially accelerating inference by lowering the numerical precision of model weights and activations [17]. Compared with pruning and knowledge distillation, it can often be applied with limited architectural changes and lower retraining overhead. In this work, we quantize MoBiMark from FP32 to INT8 to improve deployment eficiency on resource-constrained smartphones. For afine quantization, a floating-point value � is mapped to an integer representation as $x _ { q } = \mathrm { c l i p } ( \mathrm { r o u n d } ( x / s ) + z , q _ { \mathrm { m i n } } , q _ { \mathrm { m a x } } )$ , and reconstructed as $\hat { x } = s ( x _ { q } - z )$ , where � and � denote the quantization scale and zero-point, respectively. We use symmetric quantization for weights and asymmetric quantization for activations to accommodate their diferent value distributions [17].

To mitigate quantization-induced accuracy degradation, we adopt quantization-aware training (QAT), which simulates INT8 inference during training while retaining FP32 parameter optimization. Specifically, fakequantization operators are applied to weights and activations, with the forward pass defined as ${ \hat { x } } = Q ( x ) =$ $\mathrm { D e Q u a n t } ( \mathbb { Q } \mathrm { u a n t } ( x ) )$ . Since quantization is non-diferentiable, the straight-through estimator (STE) approximates the gradient during backpropagation [4]. This allows the model to adapt to quantization errors before deployment. Layers unsupported by INT8 retain FP16 representations. As shown in Table $^ { 2 , }$ the total model size decreases from 170.66 MB to 58.83 MB, a reduction of 65.53%. All subsequent experiments use the resulting edge-deployed models, and we compare their performance with the original FP32 models to evaluate the impact of quantization.

## 5 Experiments

## 5.1 Dataset and Experimental Setup

5.1.1 Speech-ultrasound Acquisition System. We constructed a synchronized speech-ultrasound acquisition system using MATLAB and Android Studio, deployed on laptops and smartphones, respectively, as shown in Fig. 8. We use the built-in top earpiece to transmit ultrasound and the adjacent microphone to record audio at 48 kHz, enabling stable near-field acoustic sensing [59]. We transmit a Continuous Wave (CW) signal �(�) as a superposition of � = 16 orthogonal sub-carriers: $\begin{array} { r } { S ( t ) = \sum _ { k = 0 } ^ { N - 1 } } \end{array}$ cos(2��<sub>�</sub>�), with $f _ { k }$ ranging from 17.15 kHz to 22.4 kHz at 350 Hz intervals. Non-rigid articulatory movements perturb the acoustic paths and encode kinematic information into the received signal �(�), which is the spatial integral of scattering echoes over the facial surface [50]. To recover baseband motion information, we perform digital coherent demodulation on �(�)

![](images/ad7a0d65fac2e9ee0f821645d2aac04648d6f2f74d6b37bfd5bdea95b202b6c5.jpg)

Fig. 8. Data capture system and diferent smartphone devices.  
![](images/770a16d51d58355caf0fd4be6a5449f89b927860677ecbcbf711b1090bb5e1a2.jpg)

![](images/9fdf0a237650339dc6619b9dbfc5c8a01bc00e3e41072dbe505b3beb5b012990.jpg)  
Fig. 9. Speech and ultrasound signal synchronization.

using the locally generated in-phase carrier cos $( 2 \pi f _ { k } t )$ and quadrature carrier $- \sin ( 2 \pi f _ { k } t )$ for each $f _ { k }$ . We then apply a 50 Hz low-pass filter to extract baseband articulatory components (speech organ movement frequency is typically around 40 Hz [27]) and downsample the signals by $D = 1 6$ to 3 kHz. This process yields complex baseband signals $B _ { k } ( t ) = I _ { k } ( t ) + j Q _ { k } ( t )$ , where $I _ { k } ( t )$ and $Q _ { k } ( t )$ capture the instantaneous motion-induced in-phase and quadrature components. The 32 resulting channels (16 frequencies × $\mathrm { I / Q } )$ form the ultrasonic sample. In parallel, we extract audible speech from the recorded signal using a Butterworth low-pass filter of order six with a cutof frequency of 16 kHz. Because both modalities are derived from the same recorded signal, they share the same acquisition timeline and are naturally aligned in time, aligned with speech via hardware timestamps (48 kHz sampling, every 512-sample block), as illustrated in Fig. 9.

5.1.2 Dataset Setup. Data were collected from 50 users (25 male and 25 female), with 40 randomly selected users for training and the remaining 10 users for testing. Each user contributed approximately 0.5 hours of paired speech and ultrasonic data. During collection, subjects held the smartphone 20–25 cm from the face, approximating a real voice-call scenario. Speech was sampled at 16 kHz and segmented into 5 s windows synchronized with the ultrasonic signal. The dataset consists of positive and negative samples:

(1) Positive samples (Pos., about 25 h): Genuine synchronized speech-ultrasound pairs collected from legitimate users.

(2) Negative samples in source-level attacks: Forged Speech (S1, 25 h), AI-generated or replayed target speech without any corresponding ultrasonic signal. Forged Speech + Fake Ultrasound (S2, about 250 h), forged target speech accompanied by artificially generated ultrasonic signals unrelated to the spoken content. Adaptive Speech-Ultrasound Attack (S3, over 2 h), attacker imitates the target user’s articulation while replaying forged speech to generate synchronized ultrasonic echoes.

(3) Negative samples in propagation-level attacks: Speech Editing (P1, over 2 h), randomly selected 0.5–1.0 s segment of an authenticated speech recording is replaced using a voice editing model while preserving the original watermark. Speech Cropping (P2, over 2 h), randomly selected 0.5–1.0 s segment is removed from an authenticated speech recording while retaining the original watermark. Speech Splicing (P3, over 3.5 h), 2–5 speech segments from diferent recordings are concatenated to create forged content, while watermark information extracted from another authenticated recording is injected into the forged speech. Watermark Reuse (P4, over 2 h), watermark information extracted from an authenticated recording, including the correct UID and SID, is directly injected into other speech samples without modifying the speech content. Voice Forgery (P5, over 3.5 h), voice cloning or editing model synthesizes arbitrary utterances from a few seconds of the target user’s speech, after which watermark information extracted from another authenticated recording is embedded into the forged speech.

For the Stabilizer module, a 20-hour clean dataset is constructed from the collected speech, along with corresponding samples subjected to diferent benign modifications, including adding diferent environment noises, the noise level is within 5-20% of the original speech intensity. The training dataset and the test dataset are divided in a ratio of 8:2. All experiments received our institutional review board (IRB) ethical approval before the start of the study.

5.1.3 Training Details. All models are trained on a single NVIDIA RTX 5070 Ti GPU. For the watermark generation module, Stage 1 trains the ultrasound encoder for 400 epochs with batch size 32, AdamW optimizer $( l r = 3 \times 1 0 ^ { - 4 }$ weight decay $1 \times 1 0 ^ { - 4 } )$ , and cosine annealing $( \eta _ { \mathrm { m i n } } = 1 0 ^ { - 6 } , T _ { \mathrm { m a x } } = 4 0 0 )$ . Stage 2 trains the fusion module for 400 epochs with batch size 128, AdamW $( l r = 5 \times 1 0 ^ { - 4 }$ , weight decay $1 \times 1 0 ^ { - 4 } )$ , � = 1.0, with Stage 1 weights frozen; learning rate (��) follows cosine annealing $( \eta _ { \mathrm { m i n } } = 1 0 ^ { - 6 } , T _ { \mathrm { m a x } } = 4 0 )$ adjusted every 10 epochs. The Embedder and Detector jointly fine-tune the open-source AudioSeal model using a group-based watermark framing strategy. The Stabilizer trains with AdamW $( l r = 2 \times 1 0 ^ { - 4 }$ , weight decay $1 \times 1 0 ^ { - 4 }$ , batch size 24, 400 epochs), with Automatic Mixed Precision and gradient clipping (max norm 5.0). For QAT, we fine-tune the pre-trained FP32 model using AdamW with an initial learning rate of $1 \times 1 0 ^ { - 5 }$ , weight decay of $1 \times 1 0 ^ { - 5 }$ , cosine annealing, a batch size of 64, and 200 epochs. We adopt the QNNPACK backend for mobile-oriented INT8 quantization, while keeping the same loss function as in pre-training.

5.1.4 Evaluation Metrics. The performance of the proposed Watermark Generator and Detector modules is rigorously evaluated on the aforementioned test dataset using standard classification metrics. The evaluation employs True Positive Rate (TPR), False Positive Rate (FPR), True Negative Rate (TNR), and False Negative Rate (FNR) as primary measures. Additionally, the Equal Error Rate (EER), which is the operating point at which FPR equals FNR, is reported to quantify the inherent trade-of between security and usability. The assessment is further complemented by the Area Under the Receiver Operating Characteristic (ROC) Curve (AUC), computed from TPR and FPR, and overall classification accuracy, defined as the proportion of correctly classified instances in the corresponding test dataset. For the embedding and detection module, we follow the evaluation methodology of AudioSeal, employing Scale-Invariant Signal-to-Noise Ratio (SI-SNR) [29], Perceptual Evaluation of Speech Quality (PESQ) [38], Virtual Speech Quality Objective Listener (ViSQOL) [14], and Short-Time Objective Intelligibility (STOI) [45] to assess the quality of the watermarked speech. These are objective perceptual metrics for measuring speech signal quality.

## 5.2 Overall Performance

We randomly select enrollment samples 20 times to compute the average ROC curve. As shown in Fig. 10, overall $\mathrm { A U C } = 0 . 9 6 3 0 \pm 0 . 0 0 9 2 , \mathrm { E E R } = 0 . 0 5 5 8$ , and positive input accuracy reach 93.15%. Although quantization has been applied, the overall accuracy remains at a high level. These results provide a solid foundation for practical deployment.

![](images/ca691d77da81259f74fdc75bfc1fb826bc957485636a6f5025873adb46db2f66.jpg)

![](images/e40bdaf5136b3fdc9b8913b1e38d3b80f346a92e4c03d50c0d8868fbf0d12f1b.jpg)  
Fig. 10. The ROC curve with AUC (left) and accuracy of diferent types of samples (right).

5.2.1 Evaluation ofSource-level Atacks. The evaluation results for source-level attacks are shown in Fig. 10. The proposed system achieves consistently high detection performance, with an average defense accuracy of 99.59% across the three attack scenarios. The first two attacks are efectively rejected because forged speech cannot establish consistent cross-modal correspondence with the accompanying ultrasonic signals. For the adaptive attack, although the attacker attempts to synchronize speech with imitated articulatory motions, the resulting ultrasonic echo patterns remain fundamentally diferent from those of genuine users. Consequently, the proposed multimodal consistency verification generalizes well to all source-level attacks while maintaining a low false rejection rate for legitimate users.

5.2.2 Evaluation of Propagation-level Atacks. The defense results against propagation-level attacks are shown in Fig. 10. Overall, the system achieves an average detection accuracy of 99.91% across all attack scenarios. Since authentication requires consistency between the speech content and the embedded watermark, semantic modifications inevitably introduce detectable watermark mismatches. The most challenging attacks are P1 and P2, where only a bit of segment is edited or removed while the attacker attempts to preserve the original watermark. Even under this strong assumption, the average similarity of the watermark decreases to $7 6 . 5 9 \% \pm 9 . 8 4 \%$ and $8 1 . 0 2 \% \pm 8 . 9 5 \% ,$ , respectively, compared to 99.73% ± 0.84% for genuine samples. For P3–P5, splicing, watermark reuse, and speech forgery further introduce substantial inconsistencies between the embedded watermark and speech semantics, allowing the proposed method to reliably detect all propagation-level attacks.

5.2.3 Embedding and Detection Performance. We evaluate the embedding module using the speech-quality metrics described above, and the results are summarized in Table 3. The results show that the embedded watermark preserves both perceptual quality and speech recognition performance. Moreover, INT8 quantization introduces only negligible performance degradation compared with FP32. Following the evaluation protocol of AudioSeal, we assess the robustness of the Detector under both benign modifications (e.g., resampling, reencoding, and compression) and malicious attacks (e.g., simple deletion, simple splicing, silencing). As shown in Table 4, FP32 and INT8 models achieve comparable performance, demonstrating the efectiveness of INT8 deployment on edge devices.

Table 3. Speech quality metrics at diferent quantization levels.
<table><tr><td>Level</td><td>SI-SNR↑ (dB)</td><td>PESQ↑</td><td>STOI↑</td><td>ViSQOL↑</td></tr><tr><td>FP32</td><td>23.88</td><td>4.43</td><td>0.997</td><td>4.81</td></tr><tr><td>INT8</td><td>20.01</td><td>4.26</td><td>0.981</td><td>4.67</td></tr></table>

Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., Vol. 0, No. 0, Article 0. Publication date: 2027.

![](images/bfde4e65000c60e78fee5c8877c5524e4b903fac37416091628cb879a2781384.jpg)  
(a) Stabilizer efect.

![](images/7d86829c99fd96fc93131b98ec899e95b39163abcce6fe9c6d11c5d28ab12344.jpg)  
(b) Modality comparison.

![](images/bba8011a606b32b9a52a4e3e369bd2cef63fd16b9b34b985de47a5fcebbbef82.jpg)  
(c) Time window.

![](images/705d26f99f15162531f6b5617474265cd1be3cff6fe31cd13f891525e9cdc18b.jpg)  
(d) Phone type.

![](images/b5e133516147d5b6ba87b481bd7e7707bb97e50a7b72add97ad0ec561f618827.jpg)  
(e) Distance.

![](images/0fb5d002793a383669f595666e70be97440c22b9d9afd119d5aeb3dac7b976e1.jpg)  
(f ) Angle.

Fig. 11. Impact of diferent factors: (a) stabilizer; (b) modality; (c) time window; (d) phone type; (e) distance; (f ) angle. Table 4. Detection results (FP32 / INT8) under benign (top) and malicious (botom) operations.
<table><tr><td>Operation</td><td>TPR</td><td>FPR</td><td>TNR</td><td>FNR</td></tr><tr><td>Compression</td><td>1.00/1.00</td><td>0.03/0.03</td><td>一</td><td></td></tr><tr><td>Reencoding</td><td>1.00/1.00</td><td>0.02/0.03</td><td>1</td><td>一</td></tr><tr><td>Resampling</td><td>0.98/0.98</td><td>0.01/0.01</td><td>一</td><td>一</td></tr><tr><td>Noise suppression</td><td>0.98/0.98</td><td>0.01/0.01</td><td>一</td><td>一</td></tr><tr><td>Simple deletion</td><td>一</td><td></td><td>0.99/0.99</td><td>0.01/0.01</td></tr><tr><td>Simple splicing</td><td>一</td><td></td><td>0.99/0.99</td><td>0.09/0.09</td></tr><tr><td>Silencing</td><td></td><td></td><td>0.98/0.97</td><td>0.02/0.02</td></tr><tr><td>Substitution</td><td></td><td></td><td>0.99/0.99</td><td>0.00/0.00</td></tr><tr><td>Reordering</td><td></td><td></td><td>0.98/0.97</td><td>0.00/0.00</td></tr><tr><td>Text-to-speech</td><td>一</td><td></td><td>1.00/1.00</td><td>0.00/0.00</td></tr><tr><td>Voice conversion</td><td>一</td><td>1</td><td>0.98/0.98</td><td>0.01/0.01</td></tr></table>

5.2.4 Stabilizer Performance. For the Stabilizer, there are both quantized and non-quantized versions. At the edge-side system, we can selectively use the quantized version for source speech processing, while on the server side, quantization is not required. We compare the results using the quantized stabilizer to verify its capability; the performance of the non-quantized stabilizer is no worse than the quantized one. Based on metrics from the detection side, we use the test speech samples containing original clean utterances and benign operation (noisy samples) for testing the Stabilizer we train. These noisy samples are fed into the stabilizer, and we compare the cosine similarity between the features of the stabilizer-processed speech and the original clean speech, as well as the similarity between the extracted watermark information (after passing through our system) and the original watermark. The results are shown in Fig. 11a. It can be seen that, for both the speech itself and the watermark information, the accuracy improves by 33.15% and 24.06%, respectively, after the stabilizer is applied, the watermark code similarity reaches 95.33%, which demonstrates the efectiveness of adding the Stabilizer.

## 5.3 Evaluation of Impact Factors

For the various influencing factors considered in this experiment, which pertain solely to the discrimination of source input. We evaluate all factors using the positive samples accuracy metric of the Watermark Generator. It is worth noting that, unlike software-based watermarking schemes, ultrasonic watermark generation is intentionally constrained by the physical characteristics of acoustic propagation. Successful embedding therefore requires the user’s active participation and an efective distance and orientation between the smartphone and the speaker. Authentication systems generally favor stricter decision boundaries, as the cost of false acceptance outweighs that of false rejection.

5.3.1 Impact ofDiferent Features. In the decision logic for target recognition, we employ a method that collaboratively discriminates using ultrasonic signals and fused feature vectors. To investigate the contribution of both to the final recognition performance, this section quantifies the influence of ultrasonic and fused vectors, with the results shown in Fig. 11b. The experiments demonstrate that the system achieves the highest recognition accuracy 93.15% when the two modalities are used jointly. This result confirms that the trained fused feature vector efectively captures the deep-level correspondence between the speech and ultrasonic modalities, rather than performing simple feature concatenation, while ensuring that speech information does not interfere with the discriminative ability of the ultrasonic modality, thereby reflecting the efectiveness of the multimodal fusion design in the proposed model.

5.3.2 Impact ofDiferent Time Windows. The time window length (T) is a critical parameter in dynamic speech watermarking systems. Its value not only influences the discrimination accuracy of ultrasonic signals but also directly determines the computational complexity of the watermark generation model. A larger T corresponds to a longer watermark update cycle, which is beneficial for reducing computational overhead. Conversely, a smaller T, while capable of reducing system latency, increases the computational load for watermark generation and exerts a certain impact on the discrimination accuracy of ultrasonic signals. This section verifies the system’s tunability between latency and performance by evaluating the change in user identification accuracy under diferent T settings, as shown in Fig. 11c. As T increases, the accuracy exhibits a gradual upward trend. Even under a short window of T=1s, the system can maintain an identification accuracy of approximately 89.57%, indicating that the proposed mechanism can achieve an efective balance between latency and recognition performance through flexible configuration of T. It is notable that all of our models are constructed using sequence-length-invariant architectures, allowing users to freely adjust the temporal window size without the need for retraining, which facilitates greater generalizability in practical deployment scenarios.

5.3.3 Impact ofDiferent Smartphone Types. To verify the generalization capability of the system across diferent mobile devices, this study selects multiple phone models, including Samsung S10 (S10), Samsung S22 Ultra (S22 Ultra), vivo X200 Pro (Vivo), and OnePlus 15 (OnePlus), for universal applicability testing. The experimental results, shown in Fig. 11d, demonstrate that the system maintains stable identity authentication performance on all devices, with an accuracy of no less than 92.50%. This is primarily attributed to the fact that the ultrasonic signals rely only on the microphone hardware commonly integrated in smartphones, without the need for specialized sensors, thereby ensuring good device compatibility and deployment flexibility. The system is suitable for mobile terminals across diferent tiers, providing a feasible solution for privacy protection in practical voice interaction scenarios.

5.3.4 Impact ofDiferent Distances. The relative distance between the user and the device is one of the key factors afecting the quality of ultrasonic signals. To evaluate the impact of distance variation on system robustness, we conduct tests within the range of 10 cm to 25 cm, with 5 cm intervals, simulating typical voice interaction distances. The results, shown in Fig. 11e, demonstrate that the system exhibits stable recognition performance

![](images/af803fe1d6dd96a3d12a6320a10b33a1d407bf5814dd3bd911aa07d891759347.jpg)

![](images/6732f21ee3d3ea35940ac7f754422b9e0df184240e42ad6bb8bb26f24a9a6dfd.jpg)

![](images/b964914f2314b98bcb7c7c2a8ed4573377376ea574786d487c95432c3a24edaa.jpg)  
Fig. 12. Inference latency (left), power consumption (center), and batery duration (right) on diferent smartphones. across diferent distances, with an overall accuracy maintained above 87.54%, an average 91.67%, indicating that the proposed method possesses strong adaptability to distance changes.

5.3.5 Impact of Diferent Angles. Furthermore, considering that in actual usage scenarios, the phone screen and the user’s face often exhibit a certain angle, we investigate the impact of diferent relative angles on system performance. Taking the parallel alignment of the screen and the facial plane as the 0° baseline, data are collected for device rotation within ±20° in the vertical direction, an angular range that ensures normal transmission and reception of ultrasonic signals by the microphone. The experimental results, shown in Fig. 11f, indicate that the system maintains high recognition robustness under various angular conditions, achieving an accuracy of no less than 92.89%, which demonstrates that the proposed method exhibits good fault tolerance for posture variations.

## 5.4 System Inference and Privacy

Our system is deployed on three types of smartphones: Samsung S10, vivo iQOO8, and OnePlus 15, released in 2019, 2021, and 2025 respectively, providing a diverse baseline for evaluating system robustness. The testing scenario simulates a real-time voice interaction. For the Watermark Generator, a suficiently long temporal window is required to generate the watermark; a window length of 1 s is used in the test. For the Embedder and Detector, a sliding step of 20 ms is set to balance latency and real-time requirements, meaning every 20 ms the system generates a watermark based on the preceding 1 s of audio and embeds it into the current window. A total of 100 samples verifies the real-time capability.

The latency results are shown in Fig. 12. On our INT8-quantized implementation, the Watermark Generator and Embedder modules achieve minimum total latency of 10.32 + 13.80 = 24.12 ms, meeting the demands of real-time communication [16]. INT8 quantization achieves latency reduction of up to 71.9% and power-consumption reduction of up to 19.35% compared to FP32. Newer smartphones, due to their better support for on-device model deployment, exhibit lower power consumption and longer sustained operation—up to 18.47 hours on the latest device. These results fully validate the real-time online performance and the efectiveness of model lightweighting for on-device deployment. Since our model is independent of the temporal window, users can adjust the watermark generation and embedding time window according to their hardware capabilities, achieving a trade-of between speed and quality.

Regarding user privacy and security, the entire watermarking and integration process runs entirely on the local side without involving cloud processing, thus ensuring complete privacy protection. The detection end, which may need cloud access, can be addressed using classic hybrid encryption architecture [31, 36, 37]. Given the general applicability of our system, it can be efectively deployed on users’ mobile edge devices, providing real-time, online protection for voice security.

## 6 Discussion

MoBiMark provides an end-to-end framework for speech security by combining real-time biometric source authentication, on-device processing, and dual-layer dynamic watermarking to protect speech against unauthorized editing and forgery during transmission. By leveraging ubiquitous microphones and resource-eficient inference, the system is designed for deployment on commodity smart devices.

MoBiMark relies on ultrasonic sensing to capture user-specific biometric features, which introduces practical constraints on sensing distance and orientation. As the user moves farther from the smartphone, weaker echoes may degrade biometric feature quality and authentication reliability. Rather than being solely a limitation, the efective sensing range also imposes a physical proximity requirement that can strengthen the association between watermark generation and the user’s physical presence. Since typical voice interactions occur within a relatively close range of the smartphone, this requirement is compatible with the intended use scenarios. In addition, preprocessing helps suppress motion-induced low-frequency artifacts, improving the reliability of ultrasonic feature extraction under practical conditions.

The security of the system also depends on the integrity of the underlying hardware and sensing pipeline. As future work, we plan to investigate hardware-assisted encryption and data-flow integrity mechanisms to protect internal sensor streams against unauthorized access and malicious data injection. Such mechanisms could further strengthen the system against attacks targeting the trusted acquisition path.

Beyond source authentication and watermark protection, extending MoBiMark to real-time voice communication is a promising direction. Although the current system provides real-time speech processing, speech codecs may distort or invalidate embedded watermarks during transmission. One potential solution is to integrate watermarking directly into neural speech codecs, enabling watermark information to be preserved in latent representations rather than relying solely on compressed waveforms. This could improve watermark integrity across the communication pipeline and facilitate practical deployment in real-time voice calls.

## 7 Conclusion

The proliferation of generative speech models poses critical threats to traditional voiceprint-based real-time voice authentication. In this paper, we present MoBiMark, the first real-time online biologic speech watermarking system using smartphones ultrasound signals. It binds ultrasonic vocal-tract echoes to speech content, embedding a dual-layer dynamic watermark at capture time for source authentication and diferent attacks detection. Detailed training and quantization design achieve high accuracy, strong robustness, and eficient on-device operation. Evaluations on 50 users across four smartphones demonstrate average 99.79% attack detection accuracy on source and propagation with negligible perceptual impact, which enables practical real-time speech security for voice interaction in the future.

## References

[1] Aya Abdulkareem and Abbas Hassin. 2025. Hardware and Software Approaches to Fingerprint Liveness Detection: A Comparative Review. Applied Computing Journal (2025), 423–438.

[2] Ali Al-Haj. 2014. An imperceptible and robust audio watermarking algorithm. EURASIP Journal on Audio, Speech, and Music Processing 2014, 1 (2014), 37. doi:10.1186/s13636-014-0037-2

[3] P. Bassia, I. Pitas, and N. Nikolaidis. 2001. Robust Audio Watermarking in the Time Domain. IEEE Transactions on Multimedia 3, 2 (2001), 232–241. doi:10.1109/6046.923822

[4] Yoshua Bengio, Nicholas Léonard, and Aaron Courville. 2013. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432 (2013).

[5] William Chan, Navdeep Jaitly, Quoc V. Le, and Oriol Vinyals. 2016. Listen, Attend and Spell: A Neural Network for Large Vocabulary Conversational Speech Recognition. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 4960–4964. doi:10.1109/ICASSP.2016.7472629

[6] Guangyu Chen, Yu Wu, Shujie Liu, Tao Liu, Xiaoyong Du, and Furu Wei. 2023. WavMark: Watermarking for Audio Generation. arXiv preprint arXiv:2308.12770 (2023).

[7] Huijie Chen, Fan Li, Wan Du, Song Yang, Matthew Conn, and Yu Wang. 2020. Listen to Your Fingers: User Authentication based on Geometry Biometrics of Touch Gesture. Proceedings ofthe ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 4, 3 (2020), 75:1–75:24. doi:10.1145/3411809

[8] Ingemar Cox, Matthew Miller, Jefrey Bloom, and Chris Honsinger. 2002. Digital Watermarking. Morgan Kaufmann.

[9] Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang, et al. 2025. Kimi-audio technical report. arXiv preprint arXiv:2504.18425 (2025).

[10] Yu Fu, Deyi Xiong, and Yue Dong. 2024. Watermarking Conditional Text Generation for AI Detection: Unveiling Challenges and a Semantic-Aware Watermark Remedy. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 38. 18003–18011.

[11] Yang Gao, Yincheng Jin, Jiyang Li, Seokmin Choi, and Zhanpeng Jin. 2020. EchoWhisper: Exploring an Acoustic-based Silent Speech Interface for Smartphone Users. Proceedings ofthe ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 4, 3 (2020), 80:1–80:27. doi:10.1145/3411830

[12] Richard W Hamming. 1950. Error detecting and error correcting codes. The Bell system technical journal 29, 2 (1950), 147–160

[13] Awni Y. Hannun, Carl Case, Jared Casper, Bryan Catanzaro, Gregory Diamos, Erich Elsen, Ryan Prenger, Sanjeev Satheesh, Shubho Sengupta, Adam Coates, et al. 2014. Deep Speech: Scaling Up End-to-End Speech Recognition. In Proceedings ofthe IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 8604–8608. doi:10.1109/ICASSP.2014.6854773

[14] Andrew Hines, Jan Skoglund, Anil Kokaram, and Naomi Harte. 2012. ViSQOL: The virtual speech quality objective listener. In Proceedings ofthe International workshop on acoustic signal enhancement (IWAENC). VDE, 1–4.

[15] Guang Hua, Jiwu Huang, Yun Q. Shi, Jonathan Goh, and Vrizlynn L. L. Thing. 2016. Twenty years of digital audio watermarking: A comprehensive review. Signal Processing 128 (2016), 222–242. doi:10.1016/j.sigpro.2016.04.005

[16] ITU-T. 2003. Transmission Systems and Media, Digital Systems and Networks. ITU-T Recommendation G.993. International Telecommunication Union.

[17] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. 2018. Quantization and training of neural networks for eficient integer-arithmetic-only inference. In Proceedings ofthe IEEE conference on computer vision and pattern recognition (CVPR). 2704–2713.

[18] Hirokazu Kameoka, Takuhiro Kaneko, Kou Tanaka, and Nobukatsu Hojo. 2019. StarGAN-VC2: Rethinking Conditional Methods for StarGAN-Based Voice Conversion. In Proceedings ofInterspeech. 679–683. doi:10.21437/Interspeech.2019-2250

[19] Jaehyeon Kim, Jungil Kong, and Juhee Son. 2021. Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech. arXiv (2021). doi:10.48550/arXiv.2106.06103

[20] Tomi Kinnunen, Md Sahidullah, Héctor Delgado, Massimiliano Todisco, Nicholas Evans, Junichi Yamagishi, and Kong Aik Lee. 2017. The ASVspoof 2017 challenge: Assessing the limits of replay spoofing attack detection. (2017).

[21] Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. 2020. HiFi-GAN: Generative Adversarial Networks for Eficient and High Fidelity Speech Synthesis. arXiv (2020). doi:10.48550/arXiv.2010.05646

[22] Yeonjoon Lee, Yue Zhao, Jiutian Zeng, Kwangwuk Lee, Nan Zhang, Faysal Hossain Shezan, Yuan Tian, Kai Chen, and XiaoFeng Wang. 2020. Using Sonar for Liveness Detection to Protect Smart Speakers against Remote Attackers. Proceedings ofthe ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 4, 1 (2020), 16:1–16:28. doi:10.1145/3380991

[23] Willem JM Levelt. 1993. Speaking: From intention to articulation. MIT press.

[24] Wang Lie and Lih-Chyau Chang. 2006. Robust and High-Quality Time-Domain Audio Watermarking Based on Low-Frequency Amplitude Modification. IEEE Transactions on Multimedia 8, 1 (2006), 46–59. doi:10.1109/TMM.2005.861292

[25] Chang Liu, Jie Zhang, Tianwei Zhang, Xi Yang, Weiming Zhang, and Nenghai Yu. 2024. Detecting Voice Cloning Attacks via Timbre Watermarking. In Proceedings ofthe Network and Distributed System Security Symposium (NDSS). doi:10.14722/ndss.2024.24200

[26] Songting Liu. 2024. Zero-shot voice conversion with difusion transformers. arXiv preprint arXiv:2411.09943 (2024).

[27] Li Lu, Jiadi Yu, Yingying Chen, Hongbo Liu, Yanmin Zhu, Yunfei Liu, and Minglu Li. 2018. LipPass: Lip Reading-Based User Authentica tion on Smartphones Leveraging Acoustic Signals. In Proceedings ofthe IEEE International Conference on Computer Communications (INFOCOM).

[28] Li Lu, Jiadi Yu, Yingying Chen, and Yan Wang. 2020. VocalLock: Sensing Vocal Tract for Passphrase-Independent User Authentication Leveraging Acoustic Signals on Smartphones. Proceedings ofthe ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 4, 2 (2020), 51:1–51:24. doi:10.1145/3397320

[29] Yi Luo and Nima Mesgarani. 2018. Tasnet: time-domain audio separation network for real-time, single-channel speech separation. In Proceedings ofthe IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 696–700.

[30] Candy Olivia Mawalim and Masashi Unoki. 2021. Speech Watermarking Method Using McAdams Coeficient Based on Random Forest Learning. Entropy 23, 10 (2021), 1246. doi:10.3390/e23101246

[31] National Institute of Standards and Technology (NIST). 2001. Advanced Encryption Standard (AES). Technical Report FIPS PUB 197. U.S. Department of Commerce, Gaithersburg, MD. https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197.pdf

[32] Junyi Park, Minjae Kim, Daisy S. Williamson, Longbiao Wang, Hei-Sook Chae, Ha-Jin Yu, and Taehyun Kim. 2021. AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks. arXiv (2021). doi:10.48550/arXiv.2110.01200

[33] Kosta Pavlovic, Slavko Kovacevic, Igor Djurovic, and Adam Wojciechowski. 2022. Robust speech watermarking by a jointly trained embedder and detector using a DNN. Digital Signal Processing 122 (2022), 103381. doi:10.1016/j.dsp.2021.103381

[34] Kaizhi Qian, Yang Zhang, Shiyu Chang, Xuesong Yang, and Mark Hasegawa-Johnson. 2019. AutoVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss. arXiv (2019). doi:10.48550/arXiv.1905.05879

[35] Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, and Tie-Yan Liu. 2020. FastSpeech 2: Fast and High-Quality End-to-End Text-to-Speech. arXiv (2020). doi:10.48550/arXiv.2006.04558

[36] Eric Rescorla. 2018. The transport layer security (TLS) protocol version 1.3. Technical Report.

[37] Ronald L Rivest, Adi Shamir, and Leonard Adleman. 1978. A method for obtaining digital signatures and public-key cryptosystems. Commun. ACM 21, 2 (1978), 120–126.

[38] Antony W Rix, John G Beerends, Michael P Hollier, and Andries P Hekstra. 2001. Perceptual evaluation of speech quality (PESQ)-a new method for speech quality assessment of telephone networks and codecs. In Proceedings ofthe IEEE international conference on acoustics, speech, and signal processing (ICASSP), Vol. 2. IEEE, 749–752.

[39] Robin San Roman, Pierre Fernandez, Alexandre Défossez, Teddy Furon, Tuan Tran, and Hady Elsahar. 2024. Proactive Detection of Voice Cloning with Localized Watermarking. arXiv preprint arXiv:2401.17264 (2024).

[40] Florian Schrof, Dmitry Kalenichenko, and James Philbin. 2015. Facenet: A unified embedding for face recognition and clustering. In Proceedings ofthe IEEE conference on computer vision and pattern recognition. 815–823.

[41] Wei Shang and Maryhelen Stevenson. 2010. Score normalization in playback attack detection. In 2010 IEEE international conference on acoustics, speech and signal processing. IEEE, 1678–1681.

[42] Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, R. J. Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, and Yonghui Wu. 2017. Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions. arXiv (2017). doi:10.48550/arXiv.1712.05884

[43] Moceheb Lazam Shuwandy, AS Jouda, MA Ahmed, Mahmood M Salih, ZT Al-qaysi, AH Alamoodi, Salem Garfan, OS Albahri, BB Zaidan, and AS Albahri. 2025. Sensor-based authentication in smartphone: A systematic review. Journal ofEngineering Research 13, 2 (2025), 741–750.

[44] Mitchell D. Swanson, Bin Zhu, Ahmed H. Tewfik, and Laurence Boney. 1998. Robust audio watermarking using perceptual masking. Signal Processing 66, 3 (1998), 337–355. doi:10.1016/S0165-1684(98)00014-0

[45] Cees H Taal, Richard C Hendriks, Richard Heusdens, and Jesper Jensen. 2010. A short-time objective intelligibility measure for time-frequency weighted noisy speech. In Proceedings of the IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 4214–4217.

[46] H. Tak, J. Patino, M. Todisco, A. Nautsch, and N. Evans. 2020. End-to-End Anti-Spoofing with RawNet2. arXiv (2020). doi:10.48550 arXiv.2011.01108

[47] Mohammad ShorifUddin, Mahmudul Hasan, and Tetsuya Shimamura. 2024. Audio Watermarking: A Comprehensive Review. International Journal of Advanced Computer Science & Applications 15, 5 (2024).

[48] Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. 2016. WaveNet: A Generative Model for Raw Audio. arXiv (2016). doi:10.48550/arXiv.1609.03499

[49] Hui Wang, Siqi Zheng, Yafeng Chen, Luyao Cheng, and Qian Chen. 2023. CAM++: A Fast and Eficient Network for Speaker Verification Using Context-Aware Masking. arXiv preprint arXiv:2303.00332 (2023).

[50] Wei Wang, Alex X. Liu, and Ke. Sun. 2016. Device-Free Gesture Tracking Using Acoustic Signals. In Proceedings of the Annual International Conference on Mobile Computing and Networking (MobiCom).

[51] Xin Wang, Junichi Yamagishi, Massimiliano Todisco, Hector Delgado, Andreas Nautsch, Nicholas Evans, Md Sahidullah, Ville Vestman, Tomi Kinnunen, Kong Aik Lee, Lauri Juvela, Paavo Alku, Yu-Huai Peng, Hsin-Te Hwang, Yu Tsao, Hsin-Min Wang, Sebastien Le Maguer, Markus Becker, Fergus Henderson, Rob Clark, Yu Zhang, Quan Wang, Ye Jia, Kai Onuma, Koji Mushika, Takashi Kaneda, Yuan Jiang, Li-Juan Liu, Yi-Chiao Wu, Wen-Chin Huang, Tomoki Toda, Kou Tanaka, Hirokazu Kameoka, Ingmar Steiner, Driss Matrouf, Jean-Francois Bonastre, Avashna Govender, Srikanth Ronanki, Jing-Xuan Zhang, and Zhen-Hua Ling. 2020. ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech. Computer Speech & Language 64 (2020), 101114. doi:10.1016/j.csl.2020.101114

[52] Zhichao Wang, Tao Li, Wenshuo Ge, Zihao Cui, Shilei Zhang, and Junlan Feng. 2026. OneVoice: One Model, Triple Scenarios-Towards Unified Zero-shot Voice Conversion. arXiv preprint arXiv:2601.18094 (2026).

[53] Zhi-Feng Wang, Gang Wei, and Qian-Hua He. 2011. Channel pattern noise based playback attack detection algorithm for speaker recognition. In 2011 International conference on machine learning and cybernetics, Vol. 4. IEEE, 1708–1713.

[54] Zhizheng Wu, Nicholas Evans, Tomi Kinnunen, Junichi Yamagishi, Federico Alegre, and Haizhou Li. 2015. Spoofing and countermeasures for speaker verification: A survey. speech communication 66 (2015), 130–153.

[55] Junichi Yamagishi, Massimiliano Todisco, Xin Wang, and the ASVspoof Consortium. 2021. ASVspoof 2021: Automatic Speaker Verification Spoofing and Countermeasures Challenge Evaluation Plan. Technical Report.

[56] Ning Yan, Kunhee Han, and Seungsoo Shin. 2026. A Comprehensive Deepfake Detection Framework Based on Multimodal Liveness Detection and Deep Learning Integration. KSII Transactions on Internet & Information Systems 20, 1 (2026).

[57] Lingfeng Yao, Chenpei Huang, Shengyao Wang, Junpei Xue, Hanqing Guo, Jiang Liu, Xun Chen, and Miao Pan. 2025. SpeechVerifier: Robust Acoustic Fingerprint against Tampering Attacks via Watermarking. arXiv preprint arXiv:2505.23821 (2025).

[58] Lingfeng Yao, Chenpei Huang, Shengyao Wang, Junpei Xue, Hanqing Guo, Jiang Liu, Phone Lin, Tomoaki Ohtsuki, and Miao Pan. 2025. Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking. arXiv preprint arXiv:2509.05835 (2025).

[59] Bing Zhou, Zongxing Xie, Yinuo Zhang, Jay Lohokare, Ruipeng Gao, and Fan Ye. 2021. Robust Human Face Authentication Leveraging Acoustic Sensing on Smartphones. IEEE Transactions on Mobile Computing 21, 8 (2021), 3009–3023.