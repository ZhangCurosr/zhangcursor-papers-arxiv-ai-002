# Hierarchical Interpretable Multitask Audio Representation Learning: An Analysis of Event-Scene Relationships through the Information

Bottleneck

Claudia Montero-Ramírez, Alba Martínez-Serrano, Carmen Peláez-Moreno, Universidad Carlos III de Madrid

Abstract—Joint Acoustic Scene Classification (ASC) and Audio Tagging (AT) models traditionally rely on unconstrained multihead architectures. However, this design ignores the natural hierarchical dependency between sounds and acoustic scenes. In this work, we investigate a hierarchical (sequential) conditioning paradigm that predicts scene categories directly from intermediate event probabilities. Through a large-scale empirical benchmark with 66 full training runs across three state-ofthe-art backbones (PaSST, PANNs, and BEATs) and different loss-weighting settings (β ∈ [0.0, 1.0]), we show that hierarchical conditioning achieves competitive classification performance while altering multitask optimization dynamics. Unlike multihead baselines, which sequentially solve the easier ASC task before optimizing AT, the hierarchical architecture enables synchronized dual-task learning for specific loss-weighting settings (β ∈ [0.6, 0.9]). Furthermore, we propose a multi-perspective interpretability framework that combines an analysis of the information bottleneck, manifold geometry, and Formal Concept Analysis (FCA) on a Term-Frequency Inverse-Document-Frequency (TF-IDF) event representation, providing complementary insights into event-scene relationships. Our findings reveal that hierarchical conditioning systematically yields lower conditional entropy, higher mutual information, and superior latent cluster separability (higher Silhouette scores), establishing a more structured and semantically faithful intermediate feature space.

Index Terms—Acoustic Scene Classification, Audio Tagging, Multitask Learning, Hierarchical Conditioning, Information Bottleneck, Formal Concept Analysis, Representation Interpretability.

## I. INTRODUCTION

Sounds are the fundamental component of an acoustic scene. A group of sounds can define an acoustic scene for human understanding. E.g., if we close our eyes and listen to a bird singing, the sound of a waterfall, and the leaves on the trees rustling in the wind, we will surely find ourselves in an outdoor scene, surrounded by nature. However, an acoustic scene is rarely defined solely by the isolated sound events occurring within it; rather, human auditory perception relies on both discrete acoustic events and a continuous, ambient background texture—frequently referred to as the acoustic "gist" [1]. In other words, while discrete events (such as footsteps or a dog barking) provide solid semantic reference points, the underlying acoustic gist captures the spatial reverberation and diffuse background noise that characterize the environment globally.

Replicating this ability computationally has become an increasingly important research goal with the rapid development of artificial intelligence (AI). We have known for years that machines can hear—ever since we became capable of recording audio. However, advances in AI have progressively enabled machines to move beyond simply capturing sound towards interpreting what is happening within audio recordings [2]. Along with this paradigm, research communities such as DCASE [3] have emerged, launching annual challenges to bring together scientists from around the world to share knowledge in these areas. One of the tasks they propose is Acoustic Scene Classification (ASC) [4], which aims to automatically identify the location where the audio was recorded. They also offer tasks such as Audio Tagging (AT) [5], which aims to identify which sounds are present in each audio clip, and Sound Event Detection (SED) [6], which works in the same way but with the added capability of identifying the specific time points at which those sounds occur. In this work, we focus on AT rather than SED, as our interest lies in the global set of acoustic events that characterize a scene rather than in their temporal occurrence. A key challenge in studying event-scene relationships is the limited availability of datasets providing both scene and sound event annotations. We therefore focus on TUT Sound Events 2016/2017 [7] and TUT Acoustic Scenes 2016 [8] and evaluate three pretrained audio backbones Patchout faSt Spectrogram Transformer (PaSST), Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition (PANNs), and Bidirectional Encoder Representation from Audio Transformer (BEATs) — to assess the consistency of our findings across different acoustic representations.

However, going back in time, it was very common to use hand-crafted features as input for models to solve these tasks. These representations included features such as MFCCs, pitch, and the energy of the audio signal, and they enabled predictions to be interpretable. E.g., audio with sustained high energy over time is more likely to come from a noisy place than a quiet one. Over time, AI has become more sophisticated, giving rise to what we call "black boxes": Deep Learning Networks (DLNs). Alongside them—and more recently—have come transformers, a type of architecture that has become absolutely dominant in data-rich settings. These models achieve fantastic metrics and typically use the spectrogram image as an input feature. However, they are called "black boxes" because, due to their large number of parameters, we lose track of what is happening inside the model from the perspective of interpretability. Nevertheless, techniques have been developed to understand parts of these models, such as SHAP [9], LIME [10], and the information bottleneck theory [11], proposed by Naftali Tishby.

![](images/10c0a8c0a1adee0790afe59437fc6ab84f4840252d6d29df7f0ba644f6a4d799.jpg)  
Fig. 1. a) Traditional architecture of multitask models with multiple heads [12]–[14]. b) Our proposed hierarchical architecture using an information bottleneck with a feed-forward network (FFN).

Beyond understanding what information is used by a model to make a prediction, an additional question arises when several related aspects of the same audio signal are considered: which information is shared across tasks, and which information is specific to each one? This question becomes particularly relevant for acoustic scene understanding, where the environment and the sounds occurring within it are intrinsically related.

A promising strategy for exploiting these relationships is multitask learning. In this paradigm, a single model is trained to perform several tasks simultaneously, sharing part of its internal representations. The idea is that related tasks can benefit from one another, since the information learned in one task acts as an inductive bias for the others. In the context of audio, this allows a model to simultaneously learn to classify acoustic scenes (ASC) and detect sounds or events (AT/SED) using a shared backbone that extracts general representations from the audio data. After the backbone, multihead layers are added so that there are small, specialized models for each task, as shown in Fig. 1.

Most existing multitask audio models adopt a multihead architecture, in which task-specific prediction heads operate on a shared representation learned by a common backbone. While this design enables the joint learning of related tasks such as ASC and AT, it does not explicitly constrain how information about acoustic events contributes to the representation used for scene prediction. This raises an important question: how are event-level acoustic concepts preserved and organized when learning to predict the scene, and how does this organization depend on the architecture used for multitask learning?

Despite the growing interest in multitask audio representation learning, most existing approaches focus primarily on predictive performance, while comparatively little attention has been devoted to understanding how acoustic event information is organized within representations that also support scene prediction. In particular, it remains unclear which event information is preserved, how strongly it is associated with the predicted scene, and whether different multitask architectures induce different event-scene relationships. Addressing these questions requires going beyond classification performance and examining the learned representations themselves.

This motivates us to introduce a hierarchical architecture in which the intermediate event representation directly conditions the scene prediction. Rather than treating event and scene predictions as parallel outputs of a shared representation, this formulation explicitly models the event-to-scene dependency and introduces a well-defined intermediate bottleneck through which this relationship can be examined (Fig. 1). The resulting architecture therefore provides not only an alternative way of performing joint ASC and AT, but also a natural framework for studying how event information is preserved and transformed when predicting acoustic scenes.

The main objective of this work is therefore to investigate how the choice of multitask architecture affects the relationship between acoustic events and acoustic scenes in pretrained audio models. To this end, we compare conventional multihead multitask learning with a hierarchical architecture that uses the intermediate event representation to predict the acoustic scene.

More specifically, this work aims to:

1) Evaluate the hierarchical and multihead architectures across different pretrained audio backbones and taskloss weights, analyzing their effect on ASC and AT performance.

2) Characterize the event representations learned by both architectures using information-theoretic and geometric measures, including conditional entropy, mutual information, KL divergence, and clustering quality.

3) Analyze how the learned event representations preserve the semantic relationships between events and scenes, using Term-Frequency Inverse-Document-Frequency (TF-IDF) weighting and Formal Concept Analysis (FCA).

4) Assess whether the explicit event-to-scene bottleneck provides a more transparent representation of eventscene relationships than the conventional multihead architecture.

Based on these objectives, the main contributions of this work are:

Hierarchical multitask architecture for event-scene modeling: We introduce a hierarchical alternative to the conventional multihead architecture for joint ASC-AT learning, in which the intermediate event representation is explicitly used to condition the scene prediction. This architecture provides a direct pathway for analyzing the relationship between acoustic events and scenes.

• Comprehensive large-scale benchmark: Rather than introducing isolated models, we conduct an extensive empirical benchmark comprising 66 full training runs across three state-of-the-art pretrained backbones (PaSST, PANNs, and BEATs) under exhaustive loss-weighting regimes $( \beta \in [ 0 . 0 , 1 . 0 ] )$ , providing the first systematic study of architectural topology in joint ASC-AT tasks.

• Synchronous multitask learning dynamics: We find a fundamental difference in optimization behavior on this dataset: while multihead models sequentially master the easier ASC task before optimizing AT, the hierarchical paradigm achieves synchronized dual-task learning across balanced loss weights $( \beta \in [ 0 . 6 , 0 . 9 ] )$ , leading to more stable joint representations.

• Multi-perspective analysis of event-scene representations: We introduce a multi-perspective interpretability framework combining information bottleneck metrics $( H ( E | S ) , I ( E ; S ) )$ ), manifold geometry (Silhouette Score over $P ( E | S ) )$ , and FCA on TF-IDF weighted event matrices to evaluate how discrete acoustic concepts are preserved through the intermediate bottleneck.

## II. RELATED WORK

ASC, SED, and AT have been extensively studied within the DCASE community [3]. In recent years, ASC research has increasingly focused not only on classification performance but also on computational efficiency. In the 2023 DCASE challenge, Schmid et al. [15] proposed a knowledge-distillation approach that achieved the best performance in the challenge, using PaSST [16] as teacher model. PaSST is an efficient and high-capacity transformer architecture that has subsequently become widely adopted for acoustic scene analysis [17]. In SED and AT, large-scale pretrained audio encoders such as PANNs [18], trained on AudioSet [19], have become standard architectures for learning general-purpose acoustic representations. Their representations have also been incorporated into more complex multimodal and audio-language systems, such as Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation (CLAP) [20].

Regarding databases, one of the most well-known in this field is AudioSet [19], which contains human-labeled 10- second sound clips extracted from 2, 084, 320 YouTube videos. There are also other databases widely used for AT and SED, such as FSD50K [21] or DESED [22]; this last one is specialized in domestic sounds and has been used since DCASE 2020. Similarly, in 2024, WildDESED [23] was created; an extension of the original DESED dataset to reflect a wider variety of domestic scenarios by incorporating complex and unpredictable background noises. On the other hand, the TAU Urban Acoustic Scenes 2022 Mobile Development Data Set [24] was selected for the ASC task since 2022. Similarly, other ASC datasets such as TUT Urban Acoustic Scenes 2018 [7] have been used for years in the DCASE challenge.

Despite the close complementarity between ASC, SED, and AT, these tasks have traditionally been investigated independently, overlooking the potentially informative relationships between acoustic events and the environments in which they occur. This has motivated the application of multitask learning to jointly exploit scene- and event-level information. Tonami et al. [12] investigated joint ASC and event recognition as a multitask problem, providing an early baseline for learning both types of information from shared acoustic representations. To this end, they adapted the TUT Sound Events 2016/2017 and TUT Acoustic Scenes 2016 [25], [8] datasets to this joint learning setting.

Subsequent studies also adopted this baseline and explored different mechanisms for exploiting the interaction between the tasks. Imoto et al. [13] introduced soft scene labels generated by a teacher model to improve joint learning, while Igarashi et al. [14] investigated semi-supervised joint learning by incorporating weak event labels. Other approaches have explored additional sources of information, such as semantic scene context [26], adversarial training [27], balanced multitask objectives [28], and scene-dependent representations [29]. More broadly, joint learning of ASC, SED, and AT has also been investigated within the DCASE community, demonstrating the potential of multitask representations to capture complementary information across acoustic tasks.

A particularly relevant line of research explicitly models the relationship between acoustic events and scenes. Hou et al. [30] proposed Relation-Guided Acoustic Scene Classification (RGASC), in which event embeddings obtained from PANNs are used to construct an event-scene relation matrix that guides scene classification. This idea was further developed by Hou et al. [31] using additional pretraining on the JSSED synthetic dataset. These approaches demonstrate that eventlevel information can provide useful contextual cues for ASC beyond conventional shared representations. However, their primary objective is to use event information to improve scene classification, rather than to characterize how the learned representation itself encodes the underlying event-scene relationships.

There is a great deal of research focused on training neural networks through an analysis based on information theory. One of the best-known researchers in this field is Naftali Tishby, who proposed the Information Bottleneck in 2000 [11]. This is a theoretical principle in machine learning and information theory, and the main idea is that learning can be understood as a process of compressing information. The goal is to find a representation of the X data that retains only the information relevant to predicting a target variable Y, while discarding irrelevant information. In 2019, Tishby explained his groundbreaking proposal at the Conference on the Mathematical Theory of Deep Learning (DeepMath 2019), where he described the two phases of neural network training [32]. Letting T be a latent variable, he identified a first ftting phase in which the network memorizes the input information, increasing both $I ( X ; T )$ and $I ( T ; Y )$ simultaneously. Afterwards, there is a second compression phase where the network discards irrelevant information and generalizes, reducing $I ( X ; T )$ while $I ( T ; Y )$ remains at a similar level. Moreover, deeper architectures tend to reach the compression regime faster, as multiple layers facilitate hierarchical abstraction [33].

Following the analysis of relationships between acoustic events and scenes, FCA provides a natural formal framework for representing object-attribute relations through concept lattices, making their underlying structure explicit [34], [35]. Additionally, beyond joint deep representations, techniques from Information Retrieval (IR) offer an explicit framework for measuring the occurrence and relevance of discrete acoustic elements within larger acoustic environments. In particular, the TF-IDF [36] formulation—traditionally used in text processing—has been adapted to audio analysis by modeling acoustic scenes as "documents"and detected sound events or acoustic words as "terms" [37], [38]. Frameworks based on Bag-of-Audio-Words (BoAW) [39] and event-based semantic vectors use TF-IDF to reduce the importance of common, uninformative background sounds while emphasizing distinctive events that strongly characterize a specific scene [38]. In a sequential audio retrieval and classification paradigm—where sound events are predicted before scene inference—TF-IDF provides an interpretable intermediate layer. It quantifies how informative the predicted event sequence is about the underlying scene, establishing an explicit information-theoretic bridge to evaluate how effectively discrete event representations transmit the necessary context for downstream scene analysis.

Despite the continuous progress in multitask audio representation learning, no prior work has proposed a strictly hierarchical, sequential framework that analyzes the bottleneck of acoustic event representations through the combined approaches of Information Theory and Information Retrieval. Most existing literature focuses primarily on proposing novel multi-head or joint-learning architectures aimed at maximizing classification metrics. Consequently, fundamental research on how acoustic events and scenes interact within a single representation is lacking, particularly on datasets of this nature. While Hou et al. [30] introduced RGASC to model eventscene dependencies via a relation matrix constructed from PANN embeddings, their approach uses event information only as a contextual feature, not as a strict sequential bottleneck.

In this work, we address this gap by introducing a hierarchical multitask architecture in which the intermediate event representation explicitly conditions the scene prediction, and by systematically comparing it with the conventional multihead architecture across multiple pretrained audio backbones and loss-weighting configurations. Beyond predictive performance, we analyze the resulting representations from complementary information-theoretic, geometric, and semantic perspectives, combining entropy and mutual information measures, representation geometry, and TF-IDF-based FCA to examine how event-scene relationships are preserved through the learned bottleneck.

## III. DATASET ANALYSIS VIA FCA LATTICES

## A. Dataset Overview

To perform this study, we have used the TUT Sound Events 2016/2017 and TUT Acoustic Scenes 2016 datasets [8], [25]. The data has already been split into two sets: a development set and an evaluation set. Each of these contains 25 sound labels and 4 acoustic scene labels. The distribution and the names of labels can be shown in Fig. 2 for sound events and in Fig. 3 for acoustic scenes.

![](images/f2701971e5d39c103c59f4e4cb7be3f8bb958a8ac727e7795a711a5048e20a12.jpg)  
Fig. 2. Sound Event distribution for development and evaluation sets. a) Distribution regarding the total duration (in seconds) of each sound event label. b)Frequency of occurrence for each sound event based on unique audio file counts. Teal bars represent the development set, while light orange bars represent the evaluation set. All labels are sorted according to the development set duration for consistency.

![](images/4de1390817bd00e8d5896af776cf315c9ec73677292cbaec398a63a1b4622c88.jpg)  
Fig. 3. Acoustic scene distribution for development and evaluation sets. The bars represent the total number of unique audio recordings associated with each scene label. Teal and light orange bars denote the development and evaluation partitions, respectively.

In total, there are 192 min of recordings in the development set and 74 min in the evaluation set.

## B. Event-Scene Relationship Analysis via FCA Lattices

To better understand the relationship between sound events and acoustic scenes, we employ FCA as a complementary analysis of the dataset. FCA provides a mathematical framework to identify groups of attributes that tend to occur together across a set of objects. In our case, each acoustic scene is considered an object, while the sound-event labels are treated as attributes. A binary formal context is constructed by indicating whether a given sound event accounts for a sufficient proportion of the scene duration. This representation allows us to identify recurring combinations of sound events and to characterize the acoustic composition of the different scenes.

![](images/bd59ee8ce52b7e88faf471e971a60c706db931448caddf9e84c1abaa38218738.jpg)  
Fig. 4. Event-scene relationships for the development (top row) and evaluation (bottom row) subsets. (a,d) Scene-event presence matrices, where each entry is the proportion of the scene duration in which the event is active. (b,e) FCA lattices of the formal contexts obtained by binarizing these matrices with a threshold of φ = 0.15. (c,f) FCA lattices obtained by applying the same threshold after TF-IDF weighting (smoothed IDF and per-scene L2 normalization), which down-weights events shared across scenes and highlights discriminative event-scene associations. In each lattice, blue nodes denote scenes (objects) and grey labels denote events (attributes); an event is associated with every scene reachable downwards from its label, and events at the bottom node are not associated with any scene.

For each acoustic scene, we compute a sound presence matrix describing the relative temporal prevalence of each sound event. The total duration associated with each event is obtained by summing its annotated durations across all audio recordings belonging to a given scene. These values are then normalized by the total duration of the corresponding scene, resulting in a scene-event matrix whose entries represent the proportion of the scene duration associated with each event. This matrix is used to construct the binary formal contexts by applying a threshold φ; after inspecting the resulting prior distributions, we set $\varphi = 0 . 1 5$ to obtain the formal contexts used for the FCA analysis. Fig. 4 shows the sound presence matrix, the lattice applied to it, and the lattice applied to the TF-IDF transformation of the prior matrix, for the development and evaluation sets. To obtain a more discriminative view, we additionally apply a TF-IDF transformation to the scene-event matrix, treating each scene as a document and each event as a term whose frequency is given by its relative temporal prevalence. The inverse document frequency factor down-weights events that occur across many scenes (e.g., "people walking" present in three of the four scenes), while the subsequent perscene L2 normalization expresses each weight relative to the overall event profile of that scene. Consequently, an event only exceeds φ when it is both prevalent within a scene and specific to it, so the same threshold selects discriminative event-scene associations rather than merely frequent ones. Note that, because of this normalization, TF-IDF can also promote weak but specific events in scenes with sparse event profiles, and demote secondary events in scenes with many

dominant events.

In Fig. 4, we see that this dataset contains many events that are highly representative of a particular acoustic scene. For example, both "breathing" and "fan" appear exclusively in the "office" scene in the development set. More generally, the raw lattices (Fig. 4 (b,e)) show that each scene is characterized by a block of exclusive events attached directly to its object concept: domestic activity sounds ("water tap running", "washing dishes", "glass jingling", "drawer", "dishes", "cutlery", "cupboard", "snapping") for "home"; computer-related sounds ("keyboard typing", "mouse clicking", "mouse wheeling") together with "fan" and "breathing" for "office"; traffic sounds ("large vehicle", "brakes squeaking") for "city center"; and natural or outdoor sounds ("'wind blowing", "bird singing", "banging") for "residential area".

Above these scene-specific concepts, the lattices reveal a small number of shared concepts that capture the overlap between scenes. The events "impact" and "rustling" group "home" and "office" into an indoor concept related to object manipulation, whereas "car" (and "children" in the development set) groups "city center" and "residential area" into an outdoor concept. "People walking" is the most widespread event, linking "home" with both outdoor scenes, while "people talking" links "office" with the outdoor scenes. The latter creates crossing relationships between the indoor and outdoor branches, so the raw lattices (10 concepts each) do not follow a clean hierarchical structure. The outdoor scenes share the largest number of events (e.g., four of the six events of "city center" in the development set are also present in "residential area"), which suggests that this pair is the most acoustically confusable.

The TF-IDF lattices (Fig. 4 (c,f)) simplify this structure to 9 concepts. Since "people talking" is no longer associated with "office", the indoor and outdoor branches become clearly separated: "home" and "office" are joined only by "impact", "city center" and "residential area" by "car" (together with "people talking" in the development set and "bird singing" in the evaluation set), and "people walking" remains the only bridge between "home" and the outdoor scenes. In addition, events that were shared by two scenes are assigned to the scene where they are more prevalent. For instance, "children" becomes exclusive to "city center" in the development set, and "rustling" becomes exclusive to "home" in the development set but to "office" in the evaluation set. This last case illustrates that some events are genuinely ambiguous between scenes, and that their discriminative value depends on the subset considered.

Comparing both subsets, the core structure is stable: the exclusive event blocks of each scene, the outdoor concept defined by "car", the indoor concept defined by "impact", and the bridging role of "people walking" appear in all four lattices. The differences are concentrated in low-prevalence events whose values lie close to φ, such as "children", "squeaking", and "breathing". "Squeaking", for example, is not associated with any scene in the development set but appears in "office" in the raw evaluation lattice, while "breathing" and "squeaking" fall below the threshold after the TF-IDF transformation in the evaluation set. Conversely, "bird singing" reaches "city center" in the evaluation set only after TF-IDF, owing to the sparse event profile of this scene. Overall, the lattices show that acoustic scenes are largely defined by specific combinations of sound events organized around a few shared concepts, which supports conditioning scene predictions on intermediate event probabilities, as in the proposed hierarchical architecture.

## IV. PROPOSED METHOD

## A. Model architectures

We use three models that are widely recognized in the scientific community, and all of them were pre-trained on AudioSet. We trained every model using two different configurations: multihead and hierarchical architectures, as illustrated in Fig. 5.

1) PaSST: In recent years, transformer models have consistently outperformed traditional architectures across a variety of tasks. Inspired by their success in natural language processing, Vision Transformers (ViTs) [40] have been introduced for image analysis, demonstrating strong performance and rapidly gaining popularity in computer vision applications. PaSST is ViT applied to spectrogram images [16]. The winners of DCASE Task 1 in 2023, Schmid et al. [15], used it as a teacher model to distill knowledge for CP-Mobile. The most notable feature of this model is the use of the Patchout technique. Not only does it achieve very high performance across all the aforementioned tasks, but it does so efficiently, making it suitable for running experiments on any single consumer-grade GPU.

![](images/a36ac462a5d13f33b7f897bd882cd4a9a02b3cbe1acf651961f22f1828ae8916.jpg)  
Fig. 5. Exact network configuration of the models with a) multihead and b) hierarchical architectures. \* BEATs and PaSST use 768-dimensional acoustic embeddings, whereas \*\* PANNs uses 2048-dimensional embeddings.

![](images/258472aa9cfcaf357abc8f677ce9bca3ce75b3e85e03edf70dba30485d75ee15.jpg)  
Fig. 6. The Patchout transformer (PaSST) adapted to a hierarchical architecture: (a) Overall architecture; C: classification token and D: distillation token [16]. (b) The architecture of the Self-attention layer + FFN (Feedforward network) [40]. (c) Hierarchical extension of the model to predict acoustic scenes from sound events.

PaSST was pre-trained for SED [16] on AudioSet, a dataset that contains 2.1 million annotated videos, 5.8 thousand hours of audio, and 527 classes of annotated sounds. From this starting point, we performed hierarchical transfer learning by replacing the final layer of 527 outputs with one of 25 corresponding to the sound labels of the TUT Sound Events 2016/2017 and TUT Acoustic Scenes 2016 datasets. Afterwards, we added a Feedforward Network formed by a single Linear Layer to predict 4 acoustic scenes. The overall architecture can be shown in Fig. 6, where (a) and (b) correspond to the original PaSST architecture, and (c) represents our contribution related to hierarchical transfer learning with a sequential architecture.

2) BEATs: Similar to PaSST, BEATs (Bidirectional Encoder representation from Audio Transformers) employs a ViT architecture as its backbone, using a linear projection layer followed by a stack of Transformer encoder layers to process sequences of audio patches. In contrast to conventional audio pre-training approaches based on acoustic feature reconstruction, BEATs is pre-trained through a Masked Audio Modeling (MAM) objective, where masked audio patches are used to predict discrete acoustic tokens generated by an iterative acoustic tokenizer. The acoustic tokenizer and the audio SSL model are iteratively optimized through knowledge distillation, allowing both components to improve their representations progressively. They then fine-tune the resulting audio representation model for downstream audio classification tasks by replacing the pre-training label predictor with a task-specific linear classifier [41]. In this work, we use the BEATs model pre-trained on AudioSet as a checkpoint and replace its final linear classification layer with two alternative architectures: a hierarchical sequential architecture and a multi-head architecture, following the same strategy used for the other backbone models.

3) PANNs: These models constitute one of the most widely used families of neural networks for general-purpose audio classification and sound event recognition. Although introduced before the growth of attention and Transformerbased architectures, PANNs models remain highly competitive and continue to be widely adopted as audio encoders in more complex multimodal and audio-language systems, such as CLAP [20]. Their strong performance, robustness across different audio domains, and availability of pre-trained models make PANNs a relevant convolutional baseline for comparison with more recent Transformer-based architectures. In this work, we use a pre-trained PANNs model as one of the backbone architectures and adapt its final classification layer to the two proposed configurations: the hierarchical sequential architecture and the multi-head architecture.

## B. Joint Training and Loss Formulation

Given the multitask nature of the problem, a composite loss function is employed, as is done in the literature [29]. This is defined in Eq. 1.

$$
\mathcal { L } = \beta \cdot \mathcal { L } _ { \mathrm { e v e n t } } + ( 1 - \beta ) \cdot \mathcal { L } _ { \mathrm { s c e n e } }\tag{1}
$$

where $\beta$ is a weighting parameter that controls the contribution of each loss term. It was selected through cross-validation by varying its value from 0 to 1 with a step size of 0.1.

On the other hand, $\mathcal { L } _ { \mathrm { e v e n t } }$ corresponds to the multi-label acoustic event detection loss, defined as the Binary Cross-Entropy (BCE) loss across all audio segments and event classes (Eq. 2):

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { e v e n t } } = - \displaystyle \frac { 1 } { N \cdot | E | } \sum _ { n = 1 } ^ { N } \sum _ { e \in E } \Big [ y _ { n , e } \log ( \hat { p } _ { n , e } ) } \\ & { ~ + \left( 1 - y _ { n , e } \right) \log ( 1 - \hat { p } _ { n , e } ) \Big ] } \end{array}\tag{2}
$$

where N is the total number of audio segments, E is the set of acoustic event classes, $y _ { n , e } \in \{ 0 , 1 \}$ represents the groundtruth binary label indicating the presence of event e in segment $n ,$ and $\hat { p } _ { n , e } = \sigma ( t _ { n , e } ) \in [ 0 , 1 ]$ is the predicted probability for event e, computed by applying the sigmoid function $\sigma ( \cdot )$ to the model's logit output $t _ { n , e } .$ This formulation corresponds to the Binary Cross-Entropy Loss¹.

![](images/dceac6408d4cf7b5bd4ad57dbdec964b32ea16ec7dd737d5cb802752f189e033.jpg)  
Fig. 7. Summary diagram of the combined loss calculation.

Finally, for Acoustic Scene Classification (ASC), we use the Cross-Entropy loss across all segments and scene classes, defined in Eq. 3:

$$
\mathcal { L } _ { \mathrm { s c e n e } } = - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sum _ { s \in S } y _ { n , s } \log ( \hat { q } _ { n , s } )\tag{3}
$$

where S denotes the set of acoustic scene classes, $y _ { n , s } \in$ {0, 1} is an indicator variable specifying whether groundtruth scene s is assigned to segment n (one-hot encoded), and $\hat { q } _ { n , s } \in [ 0 , 1 ]$ is the predicted probability for scene s on segment n, obtained via the softmax activation over the scene output logits. This corresponds to the standard Cross-Entropy Loss2.

Fig. 7 shows a summary diagram of the loss calculation.

## C. Information-Theoretic Framework

According to Naftali Tishby's theory about the information bottleneck [11], learning can be understood as a process of compressing information. Given that X represents the input data, the goal is to find a latent variable T that retains only the information relevant to predict a target variable Y. In 2019, Thisby proposed the two phases of neural network training at the Conference on the Mathematical Theory of Deep Learning (DeepMath 2019) [32]. He identified a first fitting phase in which the network memorizes the input information, increasing both $I ( X ; T )$ and $I ( T ; Y )$ simultaneously. Afterwards, there is a second compression phase where the network discards irrelevant information and generalizes, reducing $I ( X ; T )$ while $I ( T ; Y )$ remains at a similar level.

In our multitask formulation, we adapt this framework by forcing the intermediate latent representation $T$ to explicitly represent the predicted acoustic event space E, while the target variable Y corresponds to the acoustic scene S. The input audio signal X is processed by the deep backbone until it is compressed into this intermediate event-level probability space E. This event space acts as the functional information bottleneck of our system: the network must discard noisy acoustic variability from X while retaining only the sound event descriptors necessary to predict the scene S.

Under this formal mapping, the event-related loss $\mathcal { L } _ { \mathrm { e v e n t } }$ governs the mutual information $I ( X ; E )$ between the raw acoustic features and the learned event concepts, whereas the scene-related loss $\mathcal { L } _ { \mathrm { s c e n e } }$ optimizes the information $I ( E ; S )$ transmitted from the event bottleneck to the final scene predictions. Fundamentally, in our proposed hierarchical topology, the scene classification layer receives the predicted event probability distribution E as its input, without direct access to the raw acoustic features X. This architecture physically forces the information flow through the sequence $X  E  S$ , as illustrated in Fig. 8.

For evaluation purposes, while the hierarchical architecture establishes an explicit physical bottleneck, the unconstrained multihead baseline outputs event $E$ and scene S predictions via parallel heads branching from a shared feature representation. To enable a fair comparison, we extract the event probability matrix synthesized by the multihead model and construct its conditional event distribution $P ( E \mid S )$ post-hoc. This allows us to examine the implicit information bottleneck formed by the multihead joint space, evaluating whether its learned event representations naturally compress and structure scene-relevant information under the same informationtheoretic metrics.

![](images/dd79a1a10174c45e6fac903fcf2d9f36e2cadf09e1389a626fa4ac9387c19955.jpg)  
Fig. 8. Overview of the proposed hierarchical information bottleneck framework, where the acoustic input X, comprising the data and ground truth labels $X _ { \mathrm { g t } } = \{ X _ { \mathrm { e v e n t s } } , X _ { \mathrm { s c e n e } } \} ^ { \bullet }$ is compressed into an interpretable sound event bottleneck E to predict the acoustic scene S.

## D. Event-Scene Representation Framework

A central question in our evaluation is how acoustic scene information is represented and structured through the eventlevel predictions of multitask architectures. If acoustic events act as an interpretable intermediate representation for scene classification—aligning with the Information Bottleneck perspective—analyzing the statistical relationships between event predictions and target scenes reveals how information is retained and structured across different model topologies.

1) Even-Scene Co-occurrence Matrix. Let E denote the set of acoustic event classes and S the set of acoustic scene classes. To evaluate the event predictions generated by a model, we compute an aggregated eventscene co-occurrence matrix $\mathbf { \bar { C } _ { \lambda } } \in \mathbb { R } ^ { | \mathbf { \bar { \cal S } } | \times | \mathbf { \bar { \cal E } } | }$ . For each scene $s \in \ S$ , we sum the predicted sigmoidal event probabilities across all $N _ { s }$ test samples belonging to ground-truth scene s:

$$
C _ { s , e } = \sum _ { n \in \mathcal { D } _ { s } } \hat { p } _ { n , e }\tag{4}
$$

where $\mathcal { D } _ { s }$ is the set of audio segments assigned to scene $s ,$ and $\hat { p } _ { n , e } \in [ 0 , 1 ]$ is the predicted probability of event e for segment n.

2) Empirical Conditional Event Representation. To evaluate information-theoretic and geometric properties, we construct the empirical conditional event distribution $\mathbf { P } \in \mathbb { R } ^ { | S | \times | E | }$ directly from the co-occurrence matrix. By normalizing each scene's total event activity, the conditional probability $P ( e | s )$ of detecting event e given scene s is defined as:

$$
P ( e | s ) = { \frac { C _ { s , e } } { \sum _ { e ^ { \prime } \in E } C _ { s , e ^ { \prime } } } }\tag{5}
$$

This normalized matrix P reflects the fundamental acoustic profile synthesized by the multitask network, capturing both omnipresent and scene-specific event distributions in a probabilistically grounded method.

3) TF-IDF Event Weighting for Concept Lattices. While $P ( e | s )$ provides a probabilistic representation of the event-scene relationships, common background sounds may appear across many scenes and make these relationships less clear. To focus on the most discriminative acoustic events for $\mathrm { F C A }$ , we apply TF-IDF to the cooccurrence matrix C.

Here, Term Frequency $\operatorname { T F } ( s , e )$ , that is equal to $P ( e | s )$ represents the normalized activity of event e in scene $s ,$ while Inverse Document Frequency $\mathrm { I D F } ( e )$ reduces the weight of events that occur across many scene classes:

$$
\mathrm { I D F } ( e ) = \log \left( \frac { | S | } { | \{ s \in S : C _ { s , e } > 0 \} | } \right)\tag{6}
$$

The resulting matrix $W \in \mathbb { R } ^ { | S | \times | E | }$ is defined as $W _ { s , e } =$ $\mathrm { T F } ( s , e ) \times \mathrm { I D F } ( e )$ . Thus, events that are common across scenes receive lower weights, while more scene-specific events receive higher weights.

## V. EXPERIMENTAL SETTINGS

## A. Experimental Protocols and Cross-Validation

We segmented the audio into 5-second windows with a 1- second step to capture general events, resulting in a temporal resolution of 1 second. Audio was normalized and converted to mono. The development and evaluation splits were kept in their original configurations.

For training, we used the original weights of $\mathrm { P a S S T } ^ { 3 }$ $\mathrm { B E A T s ^ { 4 } }$ , and $\mathrm { P A N N s } ^ { 5 }$ pre-trained on AudioSet. Both the replaced final layer and the added feed-forward network (FFN) were randomly initialized, and the entire architecture was finetuned end-to-end for 300 epochs with early-stopping criteria.

The parameters of the pre-trained backbone were updated with a learning rate of $1 \times 1 0 ^ { - 5 }$ , while the final layers were trained with a learning rate of $1 \times 1 0 ^ { - 4 }$ . We used the Adam optimizer with a weight decay of $1 \times 1 0 ^ { - 4 }$ . The batch size was fixed at 8 for the training of PANNs and BEATs, while in the training of PaSST was 4. The training data was randomly shuffled during training.

For each configuration, the $\beta$ parameter explained in Section IV was cross-validated, resulting in a total of 66 experiments being launched. Experiments were conducted on the computing cluster of the Department of Signal Theory and Communications at Universidad Carlos III de Madrid, equipped with multiple NVIDIA GPUs.

## B. Decision Thresholding and Pooling Strategies

From the soft outputs generated by the models, posterior probabilities are obtained using a softmax function for the multi-class ASC task and an element-wise sigmoid activation for the multi-label AT task.

For AT, frame-level predictions are first aggregated to the audio level using temporal pooling operations: Max-Pooling (capturing the occurrence of an event anywhere within the clip) or Mean-Pooling (measuring average continuous activation). To convert the pooled probabilities into binary decision vectors $\hat { \mathbf { y } } _ { \mathrm { A T } } ~ \in ~ \{ 0 , 1 \} ^ { \bar { K } }$ , three distinct thresholding strategies were evaluated:

1) Static Thresholding (ST): A standard global baseline threshold fixed at $\theta = 0 . 5$ across all acoustic categories.

2) Class-Specific Nonlinear Thresholding (CT): To account for class imbalance and varying event durations in the dataset, class-dependent thresholds $\theta _ { \mathrm { c l a s s } }$ are $\mathrm { d y }$ namically computed based on the log-transformed total duration $d _ { k }$ of event k:

$$
\tilde { d } _ { k } = \frac { \log ( 1 + d _ { k } ) - \operatorname* { m i n } _ { j } \log ( 1 + d _ { j } ) } { \operatorname* { m a x } _ { j } \log ( 1 + d _ { j } ) - \operatorname* { m i n } _ { j } \log ( 1 + d _ { j } ) }\tag{7}
$$

$$
\theta _ { k } = \operatorname* { m i n } \Big ( \theta _ { \operatorname* { m i n } } + \tilde { d } _ { k } ( \theta _ { \operatorname* { m a x } } - \theta _ { \operatorname* { m i n } } ) , 0 . 5 \Big )\tag{8}
$$

where $\theta _ { \mathrm { m i n } } = 0 . 0 5$ and $\theta _ { \mathrm { m a x } } = 0 . 7 5$ . This formulation lowers the decision boundary for rare or brief acoustic events while applying a saturation bound at 0.5 to prevent over-penalizing frequent classes.

3) Scene-Based Prior Thresholding (SPT): To leverage environment-dependent acoustic constraints, predicted event probabilities $\mathbf { p } _ { \mathrm { A T } }$ are adjusted using empirical prior probabilities $\mathbf { P } ( E _ { k } | \hat { S } _ { \mathrm { A S C } } )$ , representing the durationnormalized co-occurrence of event k within the predicted scene $\hat { S } _ { \mathrm { A S C } } { : }$

$$
\mathbf { p } _ { \mathrm { c o r r e c t e d } } = \alpha \mathbf { p } _ { \mathrm { A T } } + ( 1 - \alpha ) \mathbf { P } ( E | \hat { S } _ { \mathrm { A S C } } )\tag{9}
$$

where $\alpha = 0 . 5$ balances the acoustic model confidence with empirical scene priors. Binary decisions are subsequently obtained using $\mathbf { p } _ { \mathrm { c o r r e c t e d } } > 0 . 5$

On the other hand, for ASC, frame-level predictions are also aggregated to the audio level using temporal pooling operations: Max-Pooling, Mean-Pooling, and Majority Vote. Additionally, there was also the option of not performing any type of pooling to remain consistent with the literature [28], [29].

Hyperparameters were selected via a two-stage procedure. First, to reduce computational cost, candidate epochs were provisionally fixed per loss weight (β) based on total loss minimization. At these temporary epochs, the optimal structural and post-processing parameters—ASC pooling, AT pooling, and AT thresholdwere determined by joint maximization of ASC accuracy and AT F1-macro, fixing these three hyperparameters for all subsequent steps. Second, performance matrices for ASC accuracy and AT F1-macro across all epochs and $\beta$ values were recalculated using the fixed configuration. The optimal (β, epoch) pair was then selected by minimizing the weighted Euclidean distance to the ideal performance, $\sqrt { 0 . 4 \cdot ( 1 - \mathrm { A S C } ) ^ { 2 } + 0 . 6 \cdot ( 1 - \mathrm { A T } ) ^ { 2 } }$ , computed over the min-max normalized ASC accuracy and AT F1-macro scores. This is further explained in Appendix A.

## C. Evaluation Metrics

1) Overall performance: To compare our results with those in the literature [29], we use accuracy as the main metric for ASC's global results. Additionally, we add balanced accuracy as a complementary metric to analyze global performance across minority classes. Similarly, we measure AT results using the macro F1-score and the Error Rate (ER). To represent class performance, we use precision and recall for ASC, and the F1-score and ER for AT. The hyperparameters selected for each configuration are shown in the Appendix A.

2) Information-Theoretic Evaluation Metrics: To characterize the representation space formed by the normalized event probabilities $P ( e | s )$ , we evaluate three complementary information-theoretic measures:

1) Conditional Event Entropy. To quantify the dispersion or focus of event predictions within a given scene s, we calculate the conditional Shannon entropy:

$$
H ( E \mid S = s ) = - \sum _ { e \in E } P ( e | s ) \log _ { 2 } P ( e | s )\tag{10}
$$

We report the mean entropy across all scenes, $\begin{array} { r } { \bar { H } ( E | S ) = \frac { 1 } { | S | } \sum _ { s \in S } H ( E \mid S = s ) } \end{array}$ . Lower entropy values indicate a more concentrated, scene-specific event representation, whereas higher values indicate a diffuse distribution over multiple event categories.

2) Mutual Information. To evaluate the total shared information between predicted events and scene categories, we compute the mutual information:

$$
I ( E ; S ) = \sum _ { s \in S } \sum _ { e \in E } P ( s , e ) \log _ { 2 } { \frac { P ( s , e ) } { P ( s ) P ( e ) } }\tag{11}
$$

where $P ( s , e )$ represents the empirical joint probability distribution normalized over the full co-occurrence matrix, and $\begin{array} { r c l } { P ( s ) } & { = } & { \sum _ { e \in E } P ( s , e ) } \end{array}$ and $\begin{array} { r l } { P ( e ) } & { { } = } \end{array}$ $\textstyle \sum _ { s \in S } P ( s , e )$ are the respective empirical marginal distributions. Higher mutual information implies a stronger statistical dependence between predicted events and target scenes, indicating that the intermediate event output retains discriminative scene information.

3) Kullback-Leibler Divergence. To evaluate structural differences in the representations learned by the Hierarchical $( P _ { \mathrm { H } } )$ and Multihead $( P _ { \mathrm { M } } )$ architectures, we measure the divergence between their respective conditional distributions:

$$
D _ { \mathrm { K L } } ( P _ { \mathrm { H } } \parallel P _ { \mathrm { M } } ) = \frac { 1 } { | S | } \sum _ { s \in S } \sum _ { e \in E } P _ { \mathrm { H } } ( e | s ) \log _ { 2 } \frac { P _ { \mathrm { H } } ( e | s ) } { P _ { \mathrm { M } } ( e | s ) }\tag{12}
$$

A larger KL divergence indicates that the two architectural paradigms organize intermediate event predictions across scenes in fundamentally distinct ways.

All three measures are computed independently for each $\beta$ value and backbone architecture. This analysis provides a complementary view of the multitask models beyond predictive performance, allowing us to examine how the loss-weighting parameter influences the relationship between acoustic events and scenes and whether the two architectures learn different patterns of information sharing.

3) t-SNE and Silhouette Score: To evaluate the spatial organization and clustering quality of the conditional event distributions $P ( E \mid S )$ , we project the high-dimensional scene vectors into a two-dimensional manifold using t-Distributed Stochastic Neighbor Embedding (t-SNE). To quantitatively assess cluster compactness and separability in the resulting space, we compute the Silhouette Score, defined for each sample i as $S _ { i } \ = \ ( b _ { i } - a _ { i } ) / \operatorname* { m a x } ( a _ { i } , b _ { i } )$ , where $a _ { i }$ is the mean intra-cluster distance and $b _ { i }$ is the mean nearest-cluster distance. Higher average scores across scenes indicate wellseparated and semantically cohesive representations.

4) FCA Lattice Construction from TF-IDF: To obtain the formal context required for FCA, the TF-IDF matrix W is converted into a binary matrix by applying a threshold $\varphi \ = \ 0 . 1 5$ Each entry is therefore set to one when the corresponding event-scene association exceeds the threshold, and to zero otherwise. The resulting binary relations are used to construct the concept lattices, which provide a compact representation of the semantic relationships between acoustic events and scene classes. This allows us to compare the structures learned by the different models with the groundtruth event-scene relationships.

5) SHAP values: Preliminary feature importance analysis using SHAP was also conducted. SHAP values are computed on a logistic-regression surrogate trained on the intermediate event probabilities to quantify their contribution to scene prediction. Detailed plots are provided in Appendix B for completeness.

## VI. RESULTS AND DISCUSSION

## A. Global and Class-wise Classification Performance

Global results for ASC and AT are given in Tables I and II, respectively. Additionally, normalized confusion matrices for ASC and co-occurrence matrices for AT are reported in Appendix C. Note that no ASC pooling was applied in this subsection to maintain comparability with the literature [28], [31]. In terms of overall performance, there are no significant differences between the multihead and hierarchical architectures in the ASC task; PaSST achieves better results using the hierarchical architecture, while BEATs do the contrary, and PANNs maintains similar metrics. Regarding AT global performance, PANNs has a higher ER in the hierarchical architecture, while maintaining similar F1-macro results. Regarding the results of the PaSST model, it maintains similar ER but achieves a higher F1-macro score for multihead. Finally, BEATs achieves better AT results using the hierarchical architecture.

TABLE I  
GLOBAL ASC PERFORMANCE IN TERMS OF ACCURACY AND BALANCED ACCURACY
<table><tr><td>Architecture</td><td>Metric</td><td>PaSST</td><td>PANNs</td><td>BEATs</td></tr><tr><td rowspan="2">Multihead</td><td>Accuracy</td><td>0.89</td><td>0.94</td><td>0.94</td></tr><tr><td>Balanced  $\operatorname { A c c } .$ </td><td>0.90</td><td>0.95</td><td>0.95</td></tr><tr><td rowspan="2">Hierarchical</td><td>Accuracy</td><td>0.95</td><td>0.95</td><td>0.91</td></tr><tr><td>Balanced  $\operatorname { A c c } .$ </td><td>0.95</td><td>0.96</td><td>0.92</td></tr></table>

TABLE II  
GLOBAL AT PERFORMANCE IN TERMS OF F1-MACRO AND ER
<table><tr><td>Architecture</td><td>Metric</td><td>PaSST</td><td>PANNs</td><td>BEATs</td></tr><tr><td rowspan="2">Multihead</td><td>F1-macro</td><td>0.69</td><td>0.61</td><td>0.62</td></tr><tr><td>ER</td><td>0.27</td><td>0.27</td><td>0.35</td></tr><tr><td rowspan="2">Hierarchical</td><td>F1-macro</td><td>0.63</td><td>0.63</td><td>0.66</td></tr><tr><td>ER</td><td>0.28</td><td>0.38</td><td>0.31</td></tr></table>

On the other hand, class-wise results are presented in Table III for ASC and Table IV for AT. Regarding ASC, there is clear confusion with the "residential area" class, as models tend to predict many false positives, leading to very low precision. PaSST is the only model to perform well on this class, owing to its superior capability among the evaluated architectures. Looking at the AT results by class, we cannot conclude that one architecture is strictly better than the other, since some sound events achieve higher scores with the hierarchical model while others perform better with the multihead design. Note that selected hyperparameters for each configuration are shown in Appendix A.

In short, given the large number of cross-validated hyperparameters and experiments, we argue that there is no significant difference between the two architectures in this dataset. Small performance variations mostly depend on the selected hyperparameters or the stopping epoch—factors that slightly alter the final numbers without reflecting a real architectural advantage. However, it will be necessary to examine how more complex databases perform for the ASC task, since these hierarchical architectures rely solely on sounds as the exclusive elements for scene prediction, which may result in the loss of the gist [1], thereby reducing performance. Additionally, we find a limitation when using a single dataset with a fixed development/evaluation split, as it prevents us from claiming statistical significance while remaining comparable to previous work in the literature [29].

![](images/b3f14ac3bd30ba03d5458b1998f85b1d1f35b42697416b3b5ba34b0818a94bc4.jpg)

TABLE III  
CLASS-WISE ASC PERFORMANCE IN TERMS OF PRECISION AND RECALL
<table><tr><td>Arch.</td><td>Model</td><td>Metric</td><td></td><td>home office</td><td>city center</td><td>residential area</td></tr><tr><td rowspan="3">Multihead</td><td>PaSST</td><td>Precision Recall</td><td>0.99 1.00</td><td>1.00 0.98</td><td>0.80 0.85</td><td>0.82 0.77</td></tr><tr><td>PANNs</td><td>Precision Recall</td><td>0.99 0.99</td><td>1.00</td><td>0.96</td><td>0.85</td></tr><tr><td>BEATs</td><td>Precision Recall</td><td>0.99 1.00</td><td>0.98 1.00 0.98</td><td>0.84 0.99</td><td>0.97 0.84</td></tr><tr><td rowspan="3">Hierarchical</td><td>PaSST</td><td>Precision Recall</td><td>0.99 1.00</td><td>1.00 0.98</td><td>0.83 0.98 0.85</td><td>1.00 0.86 0.98</td></tr><tr><td>PANNs</td><td>Precision Recall</td><td>1.00 0.99</td><td>1.00 0.99</td><td>0.93 0.90</td><td>0.90 0.93</td></tr><tr><td>BEATs</td><td>Precision Recall</td><td>1.00 0.99</td><td>0.99 1.00</td><td>0.81 0.91</td><td>0.89 0.78</td></tr></table>

## B. Training Dynamics across Loss Weighting (β)

Fig. 9 shows the evaluation performance over training epochs in terms of accuracy for ASC and macro F1-score for AT. Each trajectory represents a specific $\beta$ value, illustrating how the weighting factor influences training dynamics.

These trajectories reveal interesting patterns from an information-compression perspective. For the multihead architecture, the model initially prioritizes the simpler task, ASC. Once a baseline representation for ASC is established, it begins learning the more complex task, AT. Visually, this sequential optimization causes the performance paths to shift rightward first along the ASC axis before moving upward along the AT axis.

A key finding of this study is that this decoupled learning behavior does not occur in the hierarchical architecture, particularly for higher $\beta$ values $( 0 . 6 \le \beta \le 0 . 9 )$ that favor the AT task. In these cases, performance on both tasks increases simultaneously, indicating that learning AT features directly drives and optimizes the underlying representation needed for ASC. This joint convergence suggests that the hierarchical bottleneck forces the model to compress event-level details into a shared representation that immediately benefits scene inference, rather than learning two isolated tasks as observed in the multihead setup.

Additionally, Fig. 9 further supports our previous observation regarding the lack of a clear performance gap between multihead and hierarchical architectures. The endpoints of the trajectories—representing peak performance across both metrics—reach very similar regions in the task-space. Minor advantages in either ASC accuracy or AT F1-score are heavily dependent on early stopping criteria or specific hyperparameter setups rather than an intrinsic architectural superiority.

## C. Information Bottleneck and Representation Profiling

To evaluate how intermediate event representations organize and transmit scene-discriminative information, Fig. 10 illustrates the mean conditional entropy ${ \bar { H } } ( E \mid S )$ , mutual information $I ( E ; S )$ , and Kullback-Leibler divergence $D _ { \mathrm { K L } } ( P _ { \mathrm { H } } \parallel P _ { \mathrm { M } } )$ across all three model backbones (PaSST,

Fig. 9. Test performance trajectories of the multitask models across different task-loss weights $\beta ,$ represented by color, in the ASC accuracy – AT F1- macro space. The left column (a,c,e) shows the multihead architecture, while the right column (b,d,f) shows the Hierarchical architecture. From top to bottom, rows correspond to BEATs, PANNs, and PaSST, respectively. Each trajectory connects the test performance obtained for the different $\beta ^ { \dot { } }$ values, with a 90% smoothing applied for visualization.

PANNs, and BEATs) as a function of the multitask loss weight $\beta .$

Across almost all models and hyperparameter configurations (β), the proposed hierarchical architecture consistently yields a lower conditional mean entropy ${ \bar { H } } ( E \mid S )$ compared to the multihead baseline. In accordance with the information bottleneck principle, this lower entropy indicates that hierarchical conditioning effectively acts as a structural regularizer, suppressing diffuse background event activations and focusing the probability mass onto a compact, scene-specific subset of acoustic triggers.

Additionally, this spatial compression does not degrade semantic fidelity. As shown in Fig. 10, the hierarchical architecture achieves systematically higher mutual information $I ( E ; S )$ across most $\beta$ values. This confirms that while the hierarchical mechanism prunes irrelevant event noise, it preserves the statistical dependence between the intermediate event predictions and the target scene categories. Consequently, the intermediate event layer serves as a highly discriminative acoustic bottleneck.

Finally, the Kullback-Leibler divergence $D _ { \mathrm { K L } } ( P _ { \mathrm { H } } \parallel P _ { \mathrm { M } } )$ evaluates the structural distance between the final converged event distributions of both architectures. Across all models, $D _ { \mathrm { K L } }$ reaches a minimum at $\beta ~ = ~ 0 . 9$ . While the training trajectories of the hierarchical and multihead models differ significantly throughout optimization (shown in Fig. 9), their final converged representations become remarkably similar under this configuration. $\mathbf { A } \mathbf { t } \ \beta = 0 . 9 .$ , event loss dominates the total loss. This forces both architectures to prioritize fitting raw event labels, minimizing the structural effect of the hierarchical head and yielding almost identical final event distributions. However, for $\beta < 0 . 9$ , the hierarchical constraint successfully shapes the representation space, keeping it structurally distinct from the multihead baseline.

TABLE IV  
CLASS-WISE AT PERFORMANCE IN TERMS OF F1-SCORE AND ER
<table><tr><td>Architecture Model</td><td></td><td>Metric</td><td>banging (obj.)</td><td>impact (obj.)</td><td>rustling (obj.)</td><td>snapping squeaking (obj.)</td><td>(obj.)</td><td>bird singing</td><td>brakes squeaking</td><td>breathing</td><td>car</td></tr><tr><td rowspan="3">Multihead</td><td>PaSST</td><td>F1-score ER</td><td>0.75 0.25</td><td>1.00 0.00</td><td>0.77 0.25</td><td>0.91 0.17</td><td>0.00 1.00</td><td>0.63 0.51</td><td>0.73 0.28</td><td>0.33 0.33</td><td>0.83 0.27</td></tr><tr><td>PANNs</td><td>F1-score ER</td><td>0.40 0.00</td><td>1.00 0.00</td><td>0.52 0.25</td><td>0.57 0.00</td><td>0.00 1.00</td><td>0.73 0.32</td><td>0.54 0.23</td><td>0.22 0.78</td><td>0.82 0.29</td></tr><tr><td>BEATs</td><td>F1-score ER</td><td>0.40 0.00</td><td>1.00 0.00</td><td>0.80 0.30</td><td>0.75 0.00</td><td>0.00 1.00</td><td>0.70 0.33</td><td>0.70 0.21</td><td>0.17 0.87</td><td>0.82 0.29</td></tr><tr><td rowspan="3">Hierarchical</td><td>PaSST</td><td>F1-score ER</td><td>0.67 0.00</td><td>1.00 0.00</td><td>0.71 0.23</td><td>0.89 0.00</td><td>0.00 1.00</td><td>0.67 0.45</td><td>0.67 0.13</td><td>0.15 0.75</td><td>0.83 0.28</td></tr><tr><td>PANNs</td><td>F1-score ER</td><td>0.53 0.64</td><td>1.00 0.00</td><td>0.77 0.09</td><td>1.00 0.00</td><td>0.33 0.00</td><td>0.65 0.47</td><td>0.54 0.62</td><td>0.15 0.75</td><td>0.84 0.28</td></tr><tr><td>BEATs</td><td>F1-score ER</td><td>0.86 0.00</td><td>0.98 0.00</td><td>0.74 0.17</td><td>1.00 0.00</td><td>0.00 1.00</td><td>0.70 0.41</td><td>0.62 0.33</td><td>0.13 0.83</td><td>0.82 0.27</td></tr><tr><td>Architecture Model</td><td></td><td>Metric</td><td></td><td>children cupboard</td><td>cutlery</td><td>dishes</td><td>drawer</td><td>fan</td><td>glass jingling</td><td>keyb. typ.</td><td>large. vehic.</td></tr><tr><td rowspan="3">Multihead</td><td>PaSST</td><td>F1-score ER</td><td>0.18 0.00</td><td>0.89 0.00</td><td>0.57 0.50</td><td>0.89 0.20</td><td>0.89 0.00</td><td>1.00 0.00</td><td>0.50 0.00</td><td>0.67 0.50</td><td>0.39 0.71</td></tr><tr><td>PANNs</td><td>F1-score ER</td><td>0.00 1.00</td><td>0.57 0.00</td><td>0.67 0.33</td><td>0.89 0.20</td><td>0.57 0.00</td><td>0.98 0.04</td><td>0.80 0.00</td><td>0.67 0.48</td><td>0.39 0.62</td></tr><tr><td>BEATs</td><td>F1-score ER</td><td>0.00 1.00</td><td>0.57 0.00</td><td>0.75 0.40</td><td>0.57 0.33</td><td>0.75 0.00</td><td>0.98 0.04</td><td>0.00 1.00</td><td>0.72 0.43</td><td>0.26 0.78</td></tr><tr><td rowspan="3">Hierarchical</td><td>PaSST</td><td>F1-score ER</td><td>0.18 0.00</td><td>0.57 0.00</td><td>0.40 0.50</td><td>0.67 0.00</td><td>0.75 0.00</td><td>0.98 0.04</td><td>0.00 1.00</td><td>0.65 0.52</td><td>0.35 0.74</td></tr><tr><td>PANNs</td><td>F1-score ER</td><td>0.61 0.46</td><td>1.00 0.00</td><td>0.75 0.40</td><td>0.26 0.85</td><td>1.00 0.00</td><td>0.96 0.07</td><td>0.40 0.50</td><td>0.62 0.55</td><td>0.33 0.80</td></tr><tr><td>BEATs</td><td>F1-score ER</td><td>0.18 0.00</td><td>0.89 0.00</td><td>0.75 0.40</td><td>0.89 0.20</td><td>0.57 0.00</td><td>0.98 0.04</td><td>0.40 0.50</td><td>0.69 0.45</td><td>0.37 0.73</td></tr><tr><td>Architecture Model</td><td></td><td>Metric</td><td>mouse click.</td><td>mouse wheel.</td><td>peop. talk.</td><td>peop. walk.</td><td>wash. dish.</td><td>water tap</td><td>wind bl.</td><td></td><td></td></tr><tr><td rowspan="3">Multihead</td><td>PaSST</td><td>F1-score ER</td><td>0.98 0.00</td><td>0.47 0.33</td><td>0.63 0.45</td><td>0.52 0.39</td><td>0.89 0.20</td><td>1.00 0.00</td><td>0.75 0.40</td><td></td><td></td></tr><tr><td>PANNs</td><td>F1-score ER</td><td>0.96 0.08</td><td>0.27 0.50</td><td>0.48 0.33</td><td>0.45 0.23</td><td>0.89 0.20</td><td>1.00 0.00</td><td>0.80 0.00</td><td></td><td></td></tr><tr><td>BEATs</td><td>F1-score ER</td><td>0.96 0.00</td><td>0.62 0.52</td><td>0.68 0.41</td><td>0.62 0.33</td><td>0.89 0.20</td><td>1.00 0.00</td><td>0.67 0.33</td><td></td><td></td></tr><tr><td rowspan="3">Hierarchical</td><td>PaSST</td><td>F1-score ER</td><td>0.96 0.08</td><td>0.44 0.43</td><td>0.59 0.52</td><td>0.62 0.24</td><td>0.89 0.20</td><td>1.00 0.00</td><td>1.00 0.00</td><td></td><td></td></tr><tr><td>PANNs</td><td>F1-score ER</td><td>0.94 0.11</td><td>0.54 0.53</td><td>0.38 0.50</td><td>0.47 0.55</td><td>0.73 0.43</td><td>1.00 0.00</td><td>0.00 1.00</td><td></td><td></td></tr><tr><td>BEATs</td><td>F1-score ER</td><td>0.98 0.04</td><td>0.33 0.57</td><td>0.61 0.53</td><td>0.56 0.49</td><td>0.89 0.20</td><td>1.00 0.00</td><td>0.57 0.50</td><td></td><td></td></tr></table>

## D. Manifold Geometry and Cluster Separability

To complement the information-theoretic analysis, we examine the spatial organization of the learned event space by treating the predicted event probabilities $P ( E \mid S )$ as feature vectors. Fig. 11 presents a two-dimensional t-SNE projection, where each point corresponds to an audio segment colored by its ground-truth acoustic scene.

Although t-SNE is a stochastic dimensionality reduction technique, the overall topological structure remains highly consistent. Across both multihead and hierarchical architectures, the representations form well-defined, spatially separated clusters corresponding to distinct acoustic environments. Small overlaps occur mostly between "city center" and "residential area", reflecting their shared outdoor background events (such as "car" or "children").

To objectively quantify cluster compactness and boundary separation independent of t-SNE stochasticity, Table V presents the Silhouette Scores across all three model backbones. For PaSST and PANNs, the hierarchical architecture achieves noticeably higher Silhouette Scores compared to the multihead baseline, confirming that hierarchical conditioning creates more segregated scene clusters in latent space.

i  
![](images/28c51070ee34c1e9967e5dbecd71e18923fef18ab6af65295edab2566d3aa5d0.jpg)

![](images/6509734fffc43689212dbe4ca5452a84f894929b13141950710d03a2510c018e.jpg)

![](images/634dea963940ffffbcbf0c422b418e19c38d7241bbd1be2e54641e7f1f98c568.jpg)

![](images/eaa2c39eca6fd9ca639b248c987af2201adc3fb80c7f0ffd66e149674719970a.jpg)

![](images/6a6d5931e9e45dcaad8b460ef7c746714f175915896374ead563d6a7418d6804.jpg)  
Fig. 10. Information-theoretic comparison of the learned event representations across BEATs, PANNs, and PaSST. Rows show mean conditional event entropy ${ \bar { H ( E \mid S ) } }$ , mutual information I(E; S), and Kullback-Leibler divergence $D _ { \mathrm { K L } } ( P _ { H } | P _ { M } )$ , respectively. Colors distinguish the hierarchical and multihead architectures.

TABLE V  
SILHOUETTE SCORES OF THE LEARNED EVENT REPRESENTATIONS ACROSS THE THREE PRETRAINED AUDIO BACKBONES.
<table><tr><td>Architecture</td><td>PaSST</td><td>PANNs</td><td>BEATs</td></tr><tr><td>Multihead</td><td>0.451</td><td>0.426</td><td>0.529</td></tr><tr><td>Hierarchical</td><td>0.534</td><td>0.512</td><td>0.478</td></tr></table>

In contrast, BEATs shows that the multihead baseline yields a higher score than its hierarchical architecture. This aligns with our previous observations in Fig. 10, where for BEATs evaluated under its optimal configuration $( \beta ~ = ~ 0 . 4 .$ shown in Table VI), the multihead model achieves a lower conditional entropy than the hierarchical variant at higher weights $( \beta ~ = ~ 0 . 9$ , shown in Table VI). Nevertheless, taking both information-theoretic metrics and visual manifold geometry into account, the hierarchical paradigm consistently demonstrates superior or highly competitive representation structure across model backbones.

## E. Semantic Concept Preservation via Formal Concept Analysis

While information-theoretic and geometric metrics summarize global representation quality, they do not explicitly reveal which semantic associations are learned or corrupted by the model. To bridge this interpretability gap, we construct FCA concept lattices from the TF-IDF-weighted event predictions and compare them directly against the ground-truth context lattice (Fig. 4 (f)). By minimizing background noise via TF-IDF, FCA isolates the structural hierarchy of highly discriminative acoustic concepts.

![](images/847f3e7b7e5f4ae9758d5a47dfd96a3bad1984a3a035ed30e8329a3c876b7e90.jpg)  
Fig. 11. Two-dimensional t-SNE projections of the predicted event probability distributions $P ( E \mid S ) .$ where each point represents an audio segment, and colors indicate the ground-truth acoustic scene. The projections illustrate the spatial organization and cluster separability of the learned event representations across the different architectures and pretrained audio backbones.

Fig. 12 shows the lattices obtained from the predictions of the six models on the evaluation set, which can be compared with the ground-truth TF-IDF lattice in Fig. 12 (f). All predicted lattices are sparser than the ground truth: they contain 8 or 9 concepts and recover between 11 and 22 of the 28 event-scene incidences of the reference context, while introducing very few spurious associations (at most three per model). This indicates that the models behave conservatively, retaining only the associations for which they provide the strongest and most specific evidence. Nevertheless, the backbone of the ground-truth structure is preserved in all cases: "car" remains the concept shared by "city center"

![](images/c7239a3f6428acdcdb10e28e3a4b257ad28d8d18005bcb39283a8b2dba968462.jpg)  
Fig. 12. FCA lattices constructed from the TF-IDF-weighted event-scene relationship matrices predicted by the models, using a threshold of φ = 0.15. The top row (a-c) corresponds to the hierarchical architecture, while the bottom row (d-f) corresponds to the multihead architecture. From left to right, columns correspond to PANNs, PaSST, and BEATs, respectively.

and "residential area", "impact" remains the concept shared by "home" and "office", and each scene keeps a set of characteristic exclusive events, such as "large vehicle" for "city center", "bird singing" for "residential area", "water tap running" and "washing dishes" for "home", and "fan", "keyboard typing", and "mouse clicking" for "office".

The main structural difference between both architectures concerns "people walking", the most widespread and therefore least discriminative event in the ground truth, where it links "home"with both outdoor scenes. In the hierarchical models, this event is never shared across scenes: it is either assigned exclusively to "city center"(PANNs and BEATs) or discarded (PaSST). In contrast, two of the three multihead models retain it as a shared attribute, either between the two outdoor scenes (PANNs) or across "home", "city center", and "residential area" (BEATs), thus reproducing the crossing link between the indoor and outdoor branches. As a result, the hierarchical lattices of PaSST and BEATs reduce to two clearly separated branches, indoor and outdoor, that only meet at the top concept. This behavior is consistent with the lower conditional entropy H(E|S) observed for the hierarchical models: by concentrating the predicted event distribution of each scene on fewer and more specific events, hierarchical conditioning suppresses frequent associations but carries little information about the scene.

The amount of detail retained, however, depends on the backbone. The hierarchical PaSST model yields the most compact lattice, recovering only 11 incidences, whereas the hierarchical PANNs model yields the richest one, with 22 correct incidences and the highest agreement with the ground truth among all models. It is also the only model that recovers "children" and "banging" in "residential area" and "rustling” in "office", the latter being one of the ambiguous events identified in Section III-B. In exchange, it introduces a spurious association between "dishes" and "city center", which creates a link between "home" and "city center" absent from the ground truth. It also assigns "people talking" to "city center" and "breathing" to "office", associations that appear in the development set (Fig. 4 (c)) but not in the evaluation set. Consequently, at the level of individual incidences, neither architecture is uniformly superior: the hierarchical models achieve higher agreement for PANNs and BEATs, while the multihead model does so for PaSST.

Finally, some ground-truth associations are missed by all models, notably "glass jingling" in "home", "mouse wheeling" in "office", and "people talking" in "residential area". "Residential area" is the scene with the largest loss, retaining at most four of its seven associated events. These correspond mostly to low-prevalence or acoustically ambiguous events, whose predicted probabilities rarely exceed φ after TF-IDF weighting. Likewise, no model associates "bird singing" with "city center", an association that in the ground truth arises only from the normalization of a weak event (Section II-B). Overall, these results support that hierarchical conditioning preserves the key event-scene relationships of the ground truth while discarding less informative shared associations, although the level of detail retained strongly depends on the backbone.

## VII. LIMITATIONS AND FUTURE WORK

While our systematic benchmark provides critical insights into the optimization dynamics and latent geometry of multitask acoustic models, several limitations remain to be addressed in subsequent research:

1) Dataset Lack and Domain Generalization. Highquality datasets offering simultaneous, aligned annotations for both broad acoustic scenes and fine-grained audio events are exceptionally lacking in the audio community. Consequently, our findings are based on a single benchmark dataset. Evaluating whether these representation properties hold in larger-scale, highly unconstrained, or open-set acoustic environments remains an important next step.

2) Potential Loss of Acoustic Gist. By routing scene classification exclusively through the intermediate event probability matrix $P ( E \mid S )$ , the hierarchical architecture enforces a strict information bottleneck. If the pre-defined event vocabulary fails to capture fine background textures—the global acoustic gist of an environment—there is a risk of error propagation from missed event triggers to final scene predictions.

3) Computational Overhead of Hyperparameter Sweeping. Exhaustively exploring the joint trade-off space across multiple backbone architectures required significant compute resources (66 full training runs). Developing dynamic loss-weighting algorithms that automatically balance task learning without requiring offline grid searches represents a key optimization challenge.

To build upon these findings, future work will focus on three main directions. First, we plan to extend this interpretability framework by analyzing the relationship between ASC and SED, instead of AT. Second, we aim to investigate hybrid architectures that combine the structured intermediate event bottleneck $P ( E \mid S )$ with a residual global feature bypass to preserve ambient acoustic context. Finally, we will explore self-supervised pre-training strategies to expand the intermediate event vocabulary, reducing the risk of domain-specific information bottlenecks.

## VIII. CONCLUSION

In this paper, we addressed the limitations of standard unconstrained multihead architectures for joint ASC and AT by proposing a hierarchical (sequential) conditioning framework. By conditioning scene predictions directly on intermediate, normalized event probability distributions $P ( E \mid S )$ , our approach aligns model structure with the intrinsic hierarchy of physical acoustic environments.

Through an extensive empirical evaluation encompassing 66 full training runs across three state-of-the-art backbones (PaSST, PANNs, and BEATs), we demonstrated that hierarchical conditioning achieves competitive overall classification metrics while fundamentally stabilizing multitask optimization. Crucially, optimization dynamics reveal that whereas multihead architectures master the simpler ASC task early before optimizing AT, the Hierarchical approach fosters synchronized dual-task learning across balanced loss weights $( \beta \in [ 0 . 6 , 0 . 9 ] )$ 1

Furthermore, our multi-perspective interpretability framework—combining information-theoretic bottleneck metrics, manifold geometry analysis, and FCA on TF-IDF event activations—shows that the hierarchical paradigm produces a more structured intermediate representation space. Across model architectures, hierarchical conditioning consistently yields lower conditional entropy $H ( E \mid S )$ , higher mutual information $I ( E ; S )$ , and tighter latent cluster separability (higher Silhouette scores). The FCA lattices further show that hierarchical models preserve key event-scene relationships from the ground truth while reducing less informative associations, providing an interpretable view of the semantic structure learned by the models.

## IX. ACKNOWLEDGEMENTS

## REFERENCES

[1] A. Oliva, "Gist of the scene," in Neurobiology of attention. Elsevier, 2005, pp. 251–256.

[2] University of Surrey, “Making sense of sounds - creating machinesthatcanhear',"https://www.surrey.ac.uk/ centre-vision-speech-signal-processing/research/creative, 2026, centre for Vision, Speech and Signal Processing (CVSSP).

[3] A. Mesaros, R. Serizel, T. Heittola, T. Virtanen, and M. D. Plumbley, “A decade of dcase: Achievements, practices, evaluations and future challenges," Hyderabad, India, 20250406–20250411. [Online]. Available: https://2025.ieeeicassp.org/

[4] F. Schmid, P. Primus, T. Heittola, A. Mesaros, I. Martín-Morató, and G. Widmer, “Low-complexity acoustic scene classification with device information in the dcase 2025 challenge," in Proceedings of the Detection and Classification of Acoustic Scenes and Events 2025 Workshop (DCASE 2025), Barcelona, Spain, Oct. 2025, pp. 30–31.

[5] E. Fonseca, M. Plakal, F. Font, D. P. W. Ellis, and X. Serra, "Audio tagging with noisy labels and minimal supervision," in Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE 2019), New York, NY, USA, Oct. 2019, pp. 69–73.

[6] T. Wu, L. Han, Z. Yan, X. Cheng, J. Wen, and J. Yang, "Firstshot unsupervised anomalous sound detection for machine condition monitoring based on ensemble learning and domain generalization," DCASE2025 Challenge, Tech. Rep., Jun. 2025.

[7] T. Heittola, A. Mesaros, and T. Virtanen, “Tut urban acoustic scenes 2018, development dataset," (No Title), 2018.

[8] A. Mesaros, T. Heittola, and T. Virtanen, "Tut database for acoustic scene classification and sound event detection," in 2016 24th European signal processing conference (EUSIPCO). IEEE, 2016, pp. 1128–1132.

[9] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions," in Advances in Neural Information Processing Systems, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017. [Online]. Available: https://proceedings.neurips.cc/paper\_ files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf

[10] M. T. Ribeiro, S. Singh, and C. Guestrin, "Why should i trust you?: Explaining the predictions of any classifier," in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2016, pp. 1135–1144.

[11] N. Tishby, F. C. Pereira, and W. Bialek, "The information bottleneck method," arXiv preprint physics/0004057, 2000.

[12] N. Tonami, K. Imoto, M. Niitsuma, R. Yamanishi, and Y. Yamashita, “Joint analysis of acoustic events and scenes based on multitask learning," in 2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2019, pp. 338–342

[13] K. Imoto, N. Tonami, Y. Koizumi, M. Yasuda, R. Yamanishi, and Y. Yamashita, "Sound event detection by multitask learning of sound events and scenes with soft scene labels," in ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2020, pp. 621–625.

[14] A. Igarashi, S. Tsubaki, D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Imoto, “Joint analysis of acoustic scenes and sound events based on semi-supervised approach," in 2023 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), 2023, pp. 2074–2080.

[15] F. Schmid, T. Morocutti, S. Masoudian, K. Koutini, and G. Widmer, “CP-JKU submission to dcase23: Efficient acoustic scene classification with cp-mobile," DCASE2023 Challenge, Tech. Rep., May 2023.

[16] K. Koutini, J. Schlüter, H. Eghbal-zadeh, and G. Widmer, "Efficient Training of Audio Transformers with Patchout," in Interspeech 2022, 2022, pp. 2753–2757.

[17] C. Montero-Ramírez, E. Rituerto-González, and C. Peláez-Moreno, "Building artificial situational awareness: Soundscape classification in daily-life scenarios of gender based violence victims," Engineering Applications of Artificial Intelligence, vol. 177, p. 114871, 2026. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S095219762601153X

[18] Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, and M. D. Plumbley, “Panns: Large-scale pretrained audio neural networks for audio pattern recognition," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 2880–2894, 2020.

[19] J. F. Gemmeke, D. P. W. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter, "Audio set: An ontology and human-labeled dataset for audio events," in 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017, pp. 776–780.

[20] Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov, “Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation," in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1–5.

[21] E. Fonseca, X. Favory, J. Pons, F. Font, and X. Serra, “FSD50K: an open dataset of human-labeled sound events," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 829–852, 2022.

[22] N. Turpault, R. Serizel, J. Salamon, and A. P. Shah, "Sound event detection in domestic environments with weakly labeled data and soundscape synthesis," in Workshop on Detection and Classification of Acoustic Scenes and Events, 2019, pp. 253–257.

[23] Y. Xiao and R. K. Das, “WildDESED: An llm-powered dataset for wild domestic environment sound event detection system," in Proceedings of the Detection and Classification of Acoustic Scenes and Events 2024 Workshop (DCASE2024), 2024, pp. 196–200.

[24] T. Heittola, A. Mesaros, and T. Virtanen, “Tau urban acoustic scenes 2022 mobile, development dataset," Mar. 2022, dataset. [Online]. Available: https://doi.org/10.5281/zenodo.6337421

[25] A. Mesaros, T. Heittola, A. Diment, B. Elizalde, A. Shah, E. Vincent, B. Raj, and T. Virtanen, "Dcase 2017 challenge setup: Tasks, datasets and baseline system," in DCASE 2017-workshop on detection and classification of acoustic scenes and events, 2017.

[26] N. Tonami, K. Imoto, R. Nagase, Y. Okamoto, T. Fukumori, and Y. Yamashita, “Sound event detection guided by semantic contexts of scenes," in ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 801–805.

[27] A. Igarashi, K. Imoto, Y. Komatsu, S. Tsubaki, S. Hario, and T. Komatsu, “How information on acoustic scenes and sound events mutually benefits event detection and scene classification tasks," in 2022 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), 2022, pp. 7–11.

[28] L. Zhang, M. Wu, X. Cai, Y. Li, and W. Liu, "Joint analysis of acoustic scenes and sound events in multitask learning based on Cross\_MMoE model and class-balanced loss," IEEE Sensors Journal, vol. 24, pp. 18 082–18 089, Jun. 2024.

[29] H. Zhang, M. Wu, X. Cai, and W. Liu, “Scene-dependent sound event detection based on multitask learning with deformable large kernel attention convolution," PLOS ONE, vol. 20, no. 5, p. e0322002, 2025. [Online]. Available: https://doi.org/10.1371/journal.pone.0322002

[30] Y. Hou, B. Kang, W. Van Hauwermeiren, and D. Botteldooren, “Relation-guided acoustic scene classification aided with event embeddings," in 2022 International Joint Conference on Neural Networks (IJCNN), 2022, pp. 1–8.

[31] Y. Hou, B. Kang, A. Mitchell, W. Wang, J. Kang, and D. Botteldooren, “Cooperative scene-event modelling for acoustic scene classification," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 32, pp. 68–82, 2024.

[32] N. Tishby, "The two phases of deep learning," Talk presented at the Conference on the Mathematical Theory of Deep Learning (Deep-Math 2019), 2019, available at https://www.youtube.com/watch?v= utvIaZ6wYuw.

[33] —, “A deeper understanding of deep learning," Talk presented at the Data Science Summit, Jerusalem, Israel, May 2017, available at https: //www.youtube.com/watch?v=ekUWO\_pI2M8.

[34] B. Ganter and R. Wille, Formal concept analysis: mathematical foundations. Springer, 1999, vol. 284.

[35] F. J. Valverde-Albacete and C. Peláez-Moreno, “Towards a generalisation of formal concept analysis for data mining purposes," in Formal Concept Analysis: 4th International Conference, ICFCA 2006, Dresden, Germany, February 13-17, 2006. Proceedings. Springer, 2006, pp. 161-176.

[36] K. Sparck Jones, “A statistical interpretation of term specificity and its application in retrieval," Journal of documentation, vol. 28, no. 1, pp. 11–21, 1972.

[37] C. Montero-Ramírez, E. Rituerto-González, and C. Peláez-Moreno, "Spatio-temporal latent representations for the analysis of acoustic scenes in-the-wild," arXiv preprint arXiv:2412.07648, 2024.

[38] C. Luis-Mingueza, E. Rituerto-González, and C. Peláez-Moreno, “Bridging the Semantic Gap with Affective Acoustic Scene Analysis: an Information Retrieval-based Approach ," in IberSPEECH 2022, 2022, pp. 91–95.

[39] S. Pancoast and M. Akbacak, “Bag-of-audio-words approach for multimedia event classification," in Proc. Interspeech 2012, 2012, pp. 2105– 2108.

[40] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold,

S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale," CoRR, vol. abs/2010.11929, 2020. [Online]. Available: https://arxiv.org/abs/2010.11929

[41] S. Chen, Y. Wu, C. Wang, S. Liu, D. Tompkins, Z. Chen, and F. Wei, "Beats: Audio pre-training with acoustic tokenizers," 2022. [Online]. Available: https://arxiv.org/abs/2212.09058

## APPENDIX

## A. Hyperparameters

Hyperparameter configurations for each model and architecture were selected via a two-stage optimization procedure:

1) Structural and Post-Processing Parameter Selection: To significantly reduce computational cost during grid search, candidate epochs were provisionally fixed for each multitask loss weight (β) by minimizing the total validation loss. At these tentative epochs, a crossvalidation grid search was conducted over structural and post-processing settings—specifically ASC pooling, AT pooling, and AT thresholds. The optimal trio of parameters was selected by jointly maximizing ASC accuracy and AT F1-macro, and then maintained fixed for all subsequent evaluations.

2) Joint Epoch and Loss Weight Selection: Using the fixed configuration from the first stage, full performance matrices for ASC accuracy and AT F1-macro were recalculated across all training epochs and $\beta$ values. Both metrics were min-max normalized across the entire grid:

$$
M _ { \mathrm { n o r m } } = \frac { M - M _ { \mathrm { m i n } } } { M _ { \mathrm { m a x } } - M _ { \mathrm { m i n } } } , \quad M \in \{ \mathrm { A S C , A T } \}\tag{13}
$$

The optimal joint training epoch $( e ^ { * } )$ and trade-off weight $( \beta ^ { * } )$ were identified by finding the configuration that minimizes the weighted Euclidean distance to ideal joint performance:

$$
\begin{array} { r l r } {  { \big ( e ^ { * } , \beta ^ { * } \big ) = \arg \operatorname* { m i n } _ { e , \beta } \Big [ 0 . 4 \big ( 1 - \mathrm { A S C } _ { e , \beta } ^ { \mathrm { n o r m } } \big ) ^ { 2 } } } \\ & { } & { + \ 0 . 6 ( 1 - \mathrm { A T } _ { e , \beta } ^ { \mathrm { n o r m } } ) ^ { 2 } \Big ] ^ { 1 / 2 } } \end{array}\tag{14}
$$

where a 0.60 weight is assigned to the AT macro- $F _ { 1 }$ score and 0.40 to the ASC accuracy, reflecting the multilabel complexity of the event detection task.

Table VI summarizes the selected hyperparameter configurations across all evaluated models and architectural variants.

TABLE VI  
OPTIMAL HYPERPARAMETER CONFIGURATIONS SELECTED PER BACKBONE MODEL AND ARCHITECTURAL APPROACH. THRESHOLD STRATEGIES ARE ABBREVIATED AS CT (CLASS-SPECIFIC THRESHOLDING) AND ST (STATIC THRESHOLDING).
<table><tr><td>Arch.</td><td>Model</td><td>Epoch</td><td> $\beta$ </td><td>ASC Pool.</td><td>AT Pool.</td><td>AT Th.</td></tr><tr><td rowspan="3">Multihead</td><td>PaSST</td><td>108</td><td>0.9</td><td>None</td><td>max</td><td>CT</td></tr><tr><td>PANNs</td><td>296</td><td>0.8</td><td> $\operatorname* { m a x } ^ { * }$ </td><td>max</td><td>ST</td></tr><tr><td>BEATs</td><td>228</td><td>0.4</td><td> $\operatorname* { m a x } ^ { * }$ </td><td>max</td><td>ST</td></tr><tr><td rowspan="3">Hierarchical</td><td>PaSST</td><td>227</td><td>0.9</td><td>None</td><td>max</td><td>CT</td></tr><tr><td>PANNs</td><td>187</td><td>0.7</td><td> $\mathrm { m a x } ^ { * }$ </td><td>max</td><td>ST</td></tr><tr><td>BEATs</td><td>130</td><td>0.9</td><td> $\mathrm { m a x } ^ { * }$ </td><td>max</td><td>ST</td></tr></table>

\*Note that numerical results reported in Section VI-A were calculated using no-pooling strategy for ASC (i.e., "None").

TABLE VII  
MOST INFLUENTIAL ACOUSTIC EVENT FOR EACH SCENE ACCORDING TO MEAN SHAP VALUES.
<table><tr><td>Model</td><td>Arch.</td><td>Scene</td><td>Top event</td></tr><tr><td rowspan="6">PaSST</td><td rowspan="5">Mult.</td><td>home</td><td>water tap running (+1.60)</td></tr><tr><td>office</td><td>fan (+3.93)</td></tr><tr><td>city center</td><td>car (+1.59)</td></tr><tr><td>residential area</td><td>bird singing (+5.27)</td></tr><tr><td>home</td><td>car (+1.18)</td></tr><tr><td rowspan="4">Hier.</td><td>office</td><td>fan (+5.28)</td></tr><tr><td>city center</td><td>car (+1.46)</td></tr><tr><td>residential area</td><td>bird singing (+2.49)</td></tr><tr><td>home</td><td>car (+1.47)</td></tr><tr><td rowspan="6">PANNs Hier.</td><td rowspan="5">Mult.</td><td>office</td><td>fan (+4.89)</td></tr><tr><td>city center</td><td>car (+2.39)</td></tr><tr><td>residential area</td><td>bird singing (+4.31)</td></tr><tr><td>home</td><td>washing dishes (+1.27)</td></tr><tr><td>office</td><td>fan (+2.34)</td></tr><tr><td>city center</td><td>car (+0.91)</td></tr><tr><td rowspan="6">BEATs</td><td rowspan="5">Mult.</td><td>residential area</td><td>bird singing (+1.67)</td></tr><tr><td>home</td><td>car (+1.39)</td></tr><tr><td>office</td><td>fan (+5.91)</td></tr><tr><td>city center</td><td>car (+1.50)</td></tr><tr><td>residential area</td><td>bird singing (+3.34)</td></tr><tr><td rowspan="4">Hier.</td><td>home</td><td>(object) impact (+1.14)</td></tr><tr><td>office</td><td>fan (+4.68)</td></tr><tr><td>city center</td><td>car (+1.42)</td></tr><tr><td>residential area</td><td>bird singing (+6.07)</td></tr></table>

## B. Feature Importance via SHAP Values

To complement the global information-theoretic and concept lattice analyses, we compute SHAP values to examine local feature importance at the event-to-scene prediction interface. Table VII illustrates the most important event for each scene, model, and architecture, and Figures 13, 14, and 15 illustrate the impact of individual intermediate acoustic event activation probabilities on final scene classification decisions across BEATs, PANNs, and PaSST, respectively.

The most influential events are largely consistent across models and architectures. In particular, "car", "fan", and "bird singing" repeatedly emerge as the dominant events for "city center", "office", and "residential area", respectively, while the events associated with "home" show greater variation. These results provide a complementary view of the event-scene relationships identified by the FCA analysis, by highlighting the events that contribute most strongly to scene predictions.

## C. Confussion Matrix and Co-occurence Matrix

Fig. 16 shows the normalized confusion matrices for the ASC task, while Fig. 17 illustrates the co-occurrence matrix for AT. For ASC, the main confusion in predictions is between outdoor scenes, i.e., "residential area" and "city center". Additionally, in some cases, there is a little confusion between "home" and "office", but never between indoor and outdoor scenes. On the other hand, regarding performance in AT, it can be seen that there is a huge confusion between sounds belonging to a given acoustic scene, so we can see clusters in Fig. 17.

![](images/a014399e6137340e99e1b5c31b8f8b471a52de32f7ce3742db5773df2fd75760.jpg)  
Fig. 13. SHAP value summary plot comparing feature importance for BEATs under Multihead and Hierarchical configurations.

![](images/1b7fdc32517fbbee7c2bdcd98b235813b891e0ee90e9fc86bb1a9e2ac6ff52b5.jpg)  
Fig. 14. SHAP value summary plot comparing feature importance for PANNs under Multihead and Hierarchical configurations.

![](images/4f4d76a6b1a1fd93a2c4aafdb8d36f809d76803ef129943a8cc5209aef5fffc7.jpg)  
Fig. 15. SHAP value summary plot comparing feature importance for PaSST under Multihead and Hierarchical configurations.

![](images/3d2b70e5a6bebf2bc7d1a5f0768a2e9f648b77deb247ac7c7d7f3e45668d1655.jpg)

b)  
![](images/8011790c508fc7c07423b1074e7e9774873acfb5a844dac167f6a6c3ff11cf8b.jpg)

c)  
![](images/4c2c8766bb7eddc2aca1add0f4dfb1e42cd6d72514d97d8d9d1b68b99fefed73.jpg)

![](images/817c6320355db2d381cb7f57a6588fad8273cff8b78571f02dfbc5046236e716.jpg)

e)  
![](images/698cf16b38bc93c3fcd17bd04537ad4e3d21e547c89d0cd75ce6d6a811c7e73d.jpg)  
f)

![](images/b1fb570a4c6063f8e3325a7cec7b89033889e0fa526abcf59cd0f7ebe8be6792.jpg)  
Fig. 16. Normalized confusion matrices for ASC on the evaluation set. Each column corresponds to an acoustic model, namely (a,d) BEATs, (b,e) PANNs, and (c,f) PaSST, while the top and bottom rows (a-c) represent the multihead and (d-f) hierarchical architectures, respectively.

f)  
![](images/0ba064860e7e747127dd1a2335d301125b1f7d15905e797b31d1382588d2dfbc.jpg)

![](images/0f971dad61ab80a974d54d53bfc65c311be07f1eb209298b7226694e9d7c4b69.jpg)

![](images/7f6eb92b51340986062d4376581dede8ffa2f06cb77baac5f27aa0188c8fcd96.jpg)

![](images/1f3d3666377c2c05b23a61f12d52e0fd12350880158c15aa7ffbc31c397f0411.jpg)

![](images/9398b42d5173130de91285f8e40016e73c028e971289156a0d7fb747fb872f9c.jpg)

![](images/ce7d9cdb8e959904358db884928e0d3cb852bb69b9814a27ff6b291b7f148607.jpg)  
Fig. 17. Normalized co-occurrence matrices for AT on the evaluation set. Each column corresponds to an acoustic model, namely (a,d) BEATs, (b,e) PANNs, and (c,f) PaSST, while the top and bottom rows (a-c) represent the multihead and (d-f) hierarchical architectures, respectively.