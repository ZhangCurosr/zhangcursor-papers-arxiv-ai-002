# TUTTI: TOWARD GENERALIZABLE AUDIO-TO-SCORE TRANSCRIPTION VIA FULLY SYNTHESIZED DATA

Jianhuai Hu<sup>∗†1</sup> Yashan Wang<sup>∗1</sup> Shangda Wu<sup>∗3</sup>

Zhancheng Guo<sup>1</sup> Shijie Liang<sup>1</sup> Wuna Meng<sup>4</sup> Chuanqi Yang<sup>5</sup>

Xiaobing Li<sup>1</sup> Feng Yu<sup>1</sup> Maosong Sun<sup>†1,2</sup>

<sup>1</sup> Central Conservatory of Music <sup>2</sup> Tsinghua University <sup>3</sup> Independent Researcher

<sup>4</sup> Wuhan Conservatory of Music <sup>5</sup> Shanghai Conservatory of Music

<sup>∗</sup>Equal contribution <sup>†</sup>Corresponding author

hujianhuai@mail.ccom.edu.cn, sms@tsinghua.edu.cn

## ABSTRACT

Generalizable Audio-to-Score (A2S) transcription is fundamentally constrained by the severe scarcity of highquality, real-world paired data. Relying solely on existing human-annotated datasets often restricts the generalization of A2S models, limiting their efficacy primarily to single-instrumentation domains. To break this dependency on scarce real-world data, we introduce TUTTI (Transformer for Unified audio-To-score Transcription trained on Synthetic multi-Instrumentation Data), a pretraining paradigm driven by a purely synthetic, large-scale dataset. Rather than using human-composed scores, we leverage a symbolic music generation model to generate a massive, highly scalable multi-instrumentation corpus and create audio-score pairs with expressive acoustic characteristics. Capitalizing on the generated data, we employ a standard Transformer encoder-decoder architecture. We empirically demonstrate that pre-training a unified attention-based model on generated, multi-instrumentation data yields a consistently stronger foundational representation than single-instrumentation training. When fine-tuned with downstream real-world datasets, TUTTI outperforms previous approaches, establishing new overall state-of-theart results across various A2S baselines. Notably, TUTTI shows remarkable cross-instrument transferability, effectively adapting to unseen instruments with highly competitive performance. The source code and the TuttiCorpus dataset will be made publicly available at https: //github.com/a-musiclover/TUTTI.

## 1. INTRODUCTION

Audio-to-Score (A2S) transcription aims to extract scorelevel symbolic representations (e.g., ABC notation, <sub>\*\*</sub>Kern) directly from acoustic audio signals. Currently, A2S serves a dual purpose: it remains an indispensable tool for real-world music pedagogy and performance, while simultaneously acting as a pivotal data curation pipeline for generative music models [1–4]. As the demand for large-scale audio-score paired datasets grows, the ability to automatically transcribe performances into formats such as ABC notation becomes a key enabler for training generative models. Although early approaches decomposed A2S into isolated subtasks such as harmony estimation and note quantization, the paradigm has increasingly shifted towards end-to-end learning [5–11]. However, despite recent architectural advances such as hierarchical decoding [1] and customized Transformers [2], the development of a unified foundation model remains constrained by several fundamental challenges.

cc © J. Hu, Y. Wang, S. Wu, Z. Guo, S. Liang, W. Meng, C. Yang, X. Li, F. Yu, and M. Sun. Licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0). Attribution: J. Hu, Y. Wang, S. Wu, Z. Guo, S. Liang, W. Meng, C. Yang, X. Li, F. Yu, and M. Sun, “TUTTI: Toward generalizable audio-to-score transcription via fully synthesized data”, in Proc. of the 27th Int. Society for Music Information Retrieval Conf., Abu Dhabi, UAE, 2026.

Data Scarcity Bottleneck Data availability is the most critical hurdle in current A2S research [1, 12]. The total corpus of human-composed music is historically finite. Compounding this limitation, high-quality real-world acoustic performances aligned perfectly with their symbolic scores are exceedingly rare and expensive to annotate [3, 13]. Consequently, relying on existing humanannotated datasets makes it nearly impossible to train an A2S model capable of tackling the vast complexity of diverse, real-world musical compositions.

Lack of Generalizability across Instrumentations Driven directly by this data scarcity, existing A2S models are highly fragmented and task-specific. The field has largely focused on the “solo” scenario—models strictly tailored to single-instrumentation transcriptions such as solo piano, string quartets, saxophone, etc. [1–3,14]. To the best of our knowledge, a unified, generalizable A2S model capable of robustly transcribing musical pieces across a wide and unconstrained variety of instrumentations has not yet been proposed.

Task-Specific Architectural Complexity Given the historical lack of diverse, large-scale training data, the A2S field has naturally gravitated towards task-specific hybrid architectures. A common paradigm involves the use of convolutional neural networks (CNNs) as acoustic feature extractors, coupled with recurrent neural networks (RNNs) or attention mechanisms for sequence decoding [1, 2, 11, 15]. These hybrid pipelines are highly effective in data-scarce regimes because the strong local inductive biases of CNNs help prevent overfitting on small datasets [16, 17]. However, this reliance on specialized acoustic front-ends inherently increases architectural complexity. This raises an underexplored question in the A2S domain: With the introduction of massive, large-scale synthetic training data, is it possible to bypass customized acoustic front-ends and rely entirely on a unified, standard architecture?

To answer this, we propose a paradigm shift from model-centric engineering to a data-centric approach. In this paper, we introduce TUTTI, a pure Transformer-based encoder-decoder framework designed to perform generalizable multi-instrumentation A2S transcription. Treating A2S fundamentally as an “audio-to-text” sequence-tosequence translation task [18], TUTTI directly decodes audio spectrogram features into ABC notation. Inspired by the success of pure attention-based models in Natural Language Processing [19–23] and their recent adaptations in symbolic music [24–26], we intentionally adopt a standard Transformer architecture. By incorporating techniques such as interleaved ABC notation and hierarchical patchlevel and char-level decoders [24,25,27], pure Transformer architectures empowered by massive synthetic data can achieve highly competitive representations without strictly requiring customized CNN front-ends, highlighting that data scale is the primary driver of performance. Naturally, traditional CNN-based approaches stand to benefit similarly from such large-scale pre-training. To overcome the data bottleneck, we pre-train TUTTI on a massive synthetic dataset. Rather than relying on limited human compositions, we leverage NotaGen [25] to procedurally create over 363,000 unique, multi-instrumentation scores in ABC notation, which are subsequently rendered into audio. By pre-training on this unconstrained corpus, TUTTI learns a powerful foundational representation of multi-instrument acoustics.

The key contributions of our paper are as follows:

• TUTTI, an attention-based generalizable encoderdecoder model that directly transcribes audio spectrograms into ABC notation without relying on traditional CNN feature extractors.

• TuttiCorpus, a dataset of over 363,000 unique multiinstrumentation audio-score pairs. It enables the model to learn complex polyphonic structures across diverse musical textures through a purely generative data pipeline.

• Our experimental results demonstrate that TUTTI achieves state-of-the-art performance across various instrumentations. Furthermore, we show that multi-instrumentation pre-training consistently outperforms single-domain training, and that the model exhibits generalizability to the unseen instrument (i.e., saxophone) which is entirely absent from the pre-training data.

## 2. RELATED WORK

Early approaches to A2S transcription predominantly treated the problem as a pipeline of isolated subtasks, such as multi-pitch estimation followed by rhythm quantization [5–7]. While functional, these decomposition-based methods often suffered from error accumulation across different stages. To mitigate this, the field has seen a notable paradigm shift towards end-to-end models [8].

The current standard for end-to-end A2S heavily relies on a hybrid convolutional recurrent neural network (CRNN) architecture within an encoder-decoder framework. Typically, a CNN front-end is deployed to extract local acoustic features from audio spectrograms, which are subsequently decoded into musical symbols by an RNN [1–3, 15]. Training strategies for these sequence labeling tasks generally diverge into two paths: utilizing Connectionist Temporal Classification (CTC) loss [9, 15] to bypass explicit input alignment, or adopting a Sequenceto-Sequence (Seq2Seq) framework [11] for direct target prediction. Despite improving overall transcription coherence, translating dense polyphonic music into structured score formats (e.g., ABC notation, <sub>\*\*</sub>Kern) remains a formidable challenge, with many models struggling to consistently maintain voicing structures across complex, unconstrained musical textures [15].

To address the hierarchical complexity of musical scores, recent research has explored more advanced decoding mechanisms. Zeng et al. [1] introduced a Seq2Seq model with a hierarchical decoder capable of capturing both bar-level metadata and note-level sequences. To counteract the scarcity of real-world data, they pre-trained their model using an Expressive Performance Rendering (EPR) system to humanize the rendered data before fine-tuning on human piano recordings. Concurrently, recognizing the inherent limitations of RNNs in modeling long-term polyphonic dependencies, Alfaro-Contreras et al. [2] integrated a Transformer decoder into the A2S pipeline. They employed a specialized two-dimensional positional encoding to preserve frequency-time relationships from the CNN feature maps, achieving significant improvements in transcribing polyphonic string music.

Despite these architectural innovations, existing endto-end models remain fundamentally constrained by the scarcity of large-scale, diverse training data, heavily restricting their applicability to single-instrumentation scenarios. Furthermore, the persistent reliance on CNN acoustic front-ends introduces architectural complexity.

## 3. METHODOLOGY

In this section, we detail the methodology behind TUTTI. Building upon recent advancements in symbolic music processing and generation [24, 25], we adapt the hierarchical bar-stream patching strategy to the challenging crossmodal domain of A2S transcription. We first outline the structured ABC notation adopted for data representation (Section 3.1). We then detail the core architectural design of TUTTI, highlighting our continuous acoustic encoder and the cross-modal hierarchical decoder.

![](images/aca022a5bb6f6803bbe79501eba53f2cfb3c538ba55d9f2c9af1db396a6cef22.jpg)  
Figure 1: Architectural overview of TUTTI. The model bridges the gap between continuous audio spectrograms and discrete symbolic scores through a Transformer encoder and a hierarchical decoder.

## 3.1 Data Representation

We use ABC notation for symbolic music representation. Departing from the original plain ABC format, we adopt an advanced bar-stream patching strategy, along with an interleaved and reduced ABC notation format, to ensure a highly structured and sequence-efficient representation, as established in previous works [25–27].

## 3.2 Model Architecture

As depicted in Figure 1, TUTTI employs an encoderdecoder architecture based on the Transformer network [19], tailored specifically for A2S transcription.

The model directly ingests audio spectrograms and autoregressively generates hierarchical symbolic patches. It is composed of the following key modules:

Acoustic Adapter: The encoder’s input consists of continuous acoustic features. Specifically, raw audio is first transformed into a spectrogram representation with the shape of $T \times F$ , where T represents the number of time steps and F denotes the frequency bins. To align these high-dimensional acoustic features with the internal working dimension of the Transformer, we apply a dedicated linear projection layer. This layer maps the 1025-dimensional vector at each time step into a dense continuous embedding of dimension 512, matching the model’s hidden size $( d _ { \mathrm { m o d e l } } )$ . This projection acts as a lightweight acoustic embedding mechanism, compressing the frequency information while preserving the temporal sequence length, enabling the Transformer to directly “read” the audio.

Acoustic-level Encoder: Operating on the sequence of continuous embeddings produced by the projection layer, the encoder is responsible for extracting high-level acoustic and harmonic contexts. The encoder captures both local transient events (e.g., note onsets) and global structural dependencies (e.g., phrasing and sustained pitches) directly from the spectrogram sequence over the T time steps.

Cross-Modal Patch-level Decoder: To handle the highly structured and dense nature of musical scores, we adopt the hierarchical decoding strategy [24], tailoring it for cross-modal alignment. This module takes the embeddings of previously encoded acoustic features as input. This allows the patch-level decoder to dynamically align the global musical structure (e.g., bar-level pacing and polyphony) with corresponding acoustic segments in the audio sequence, ultimately outputting a dense contextual representation for the next predicted bar.

Character-level Decoder: Operating in a cascaded, step-by-step manner, this module acts as the final translator into discrete ABC notation. Taking the dense contextual representation generated by the patch-level decoder as its condition vector, it autoregressively decodes the specific musical tokens (characters) within that patch. By focusing on local syntactic information—such as specific pitches, durations, and structural symbols—it sequentially reconstructs the exact musical content of each bar patch until the full target score is generated. This hierarchical design effectively decouples the global audio-to-measure alignment from the local measure-to-character generation, ensuring both structural coherence and syntactic correctness.

## 4. DATASET

To overcome the severe scarcity of high-quality, realworld data, we introduce a massive, fully synthesized multi-instrumentation TuttiCorpus dataset. A fundamental contribution of this work is the creation of this dataset from scratch: rather than relying on existing, heavily constrained symbolic repositories, we leverage Nota-Gen [25]—a symbolic music generation model—to generate 363,610 unique ABC scores, conditioned on diverse prompts specifying musical periods (e.g., Baroque, Romantic), composers (e.g., Bach, Beethoven), and instrumentations (e.g., Keyboard, Chamber). This ensures stylistic variety and musical coherence, as NotaGen’s finetuning on high-quality classical corpora and RL alignment enhance generation quality. To mitigate potential data leakage from NotaGen’s training distribution, we conducted a rigorous similarity audit between our generated corpus and the ground-truth test set. We computed the normalized Levenshtein (edit) distance over canonicalized musical sequences to ensure that no generated piece duplicates or closely resembles any test sample. Any synthetic instance exceeding a minimal similarity threshold was strictly removed, guaranteeing that the evaluation benchmark remains completely unseen.

## 4.1 Data Synthesis and Alignment Pipeline

## 4.1.1 Score Standardization and Compaction

Raw ABC notation files often contain heterogeneous metadata, varying repeat structures, and empty instrumental voices, which introduce noise during sequence modeling. We first standardize the ABC files by removing redundant metadata, stylistic textual annotations, and unplayable symbols. Crucially, we unroll all repeat structures to ensure the structural linearity of the score matches the temporal linearity of the audio sequence. Furthermore, we dynamically detect and strip empty voices, transferring necessary instrumental metadata to active voices, and compress the multi-line voice structures into a dense, inline sequence representation. This compact format maximizes the efficiency of the Transformer model during training.

## 4.1.2 Expressive Performance Rendering (EPR)

A major challenge in A2S transcription is the discrepancy between rigid, quantized synthetic audio and human expressive variations. To improve model generalization [1], we apply Expressive Performance Rendering (EPR) prior to audio synthesis via three parallel pipelines. A Plain pipeline applies strict quantization with baseline dynamics compression and voice balancing. For piano tracks, we utilize VirtuosoNet [28] to generate human-like expressive performances natively. Finally, since neural EPR is largely limited to pianos, we introduce a Rule-based EPR for other instruments that injects controlled perturbations, including velocity humanization (±15), timing jitter (±10 ms), and tempo rubato (±3%).

## 4.1.3 Audio Synthesis and High-Precision Alignment

Instead of relying on standard General MIDI (GM) synthesizers, which lack acoustic nuances, we render the processed MIDI files into high-quality audio using the SFZ sample format via sfizz. <sup>1</sup> We extract the required pitch ranges from the ABC score to dynamically match each voice with the most appropriate acoustic sample library. The resulting individual audio tracks are mixed with linear overlay and normalized to a target loudness of -23.0 LUFS.

To establish ground-truth alignment for segmentation, we employ Dual Dynamic Time Warping (DualDTW) [29] via Parangonar <sup>2</sup> to align the expressive MIDI performances back to the quantized MusicXML scores. This yields precise, bar-level temporal anchors. Using these anchors, we greedily segment both the continuous audio and the inline ABC scores into aligned chunks no longer than 14.8 seconds.

Finally, the segmented audio clips are converted into power spectrograms (sample rate 22,050 Hz, window size 2048, hop length 160) and log-dB normalized, serving as the direct input to our model.

## 4.2 Data Statistics

Through our automated generation and synthesis pipeline, we successfully compiled a massive, multiinstrumentation dataset comprising 363,610 unique and fully aligned data pairs. Our dataset introduces a high level of instrumentation and compositional diversity, making it highly suitable for training generalized transcription models.

As detailed in Table 1 and Figure 2, the dataset covers an extensive range of instruments across different families, including strings, keyboards, and various woodwinds/brass. In terms of instrumentation configurations, the dataset naturally follows a long-tailed distribution characteristic of diverse musical compositions. While solo configurations account for 64.6% of the data—providing a solid foundation for fundamental pitch and duration recognition—a substantial portion of the dataset is dedicated to complex polyphonic and multi-instrumentation combinations. Specifically, duets and trios constitute nearly 29% of the corpus, and approximately 24,000 instances feature quartets, quintets, or larger ensembles. This structural diversity forces the model to learn generalized acoustic representations and robust source identification capabilities, rather than overfitting to the timbre of a single instrument.

## 5. EXPERIMENT

## 5.1 Settings

The experiments are designed to systematically assess the capabilities and generalizability of TUTTI in various A2S tasks. We utilize the TuttiCorpus dataset, which is randomly split into 95% for training and 5% for testing.

TUTTI features a 9-layer patch-level encoder and decoder, a 3-layer character-level decoder, and a hidden size of 512, amounting to 88.9M parameters. A linear projection layer maps the 1025-dimensional spectrogram features into the encoder’s input space. The model processes audio inputs of up to approximately 14.8 seconds using a patch length of 2048. It employs a 128-size ASCII-based vocabulary, using characters 0–2 for special tokens (pad, bos, and eos).

<table><tr><td>Category</td><td>Details</td><td>Count</td><td>%</td></tr><tr><td>Total Segments</td><td>All audio-score pairs</td><td>363,610</td><td>100.00</td></tr><tr><td>Top 5 Instruments (by frequency)</td><td>Piano Violin Cello Harpsichord Viola</td><td>228,964 208,119 102,367 63,974 44,912</td><td></td></tr><tr><td>Ensemble Formats (by size)</td><td>Solo (1 Inst.) Duet (2 Inst.) Trio (3 Inst.) Quartet (4 Inst.) Quintet (5 Inst.)</td><td>235,032 51,049 53,601 14,010 5,522</td><td>64.64 14.04 14.74 3.85 1.52</td></tr></table>

Table 1: Detailed statistics of the TuttiCorpus dataset, highlighting instrumentation frequency and ensemble formats.

![](images/db24021775b7d10c24830a4e2e22ae33409391edeb6f59d67fd80b87a9cfd592.jpg)  
Figure 2: Distributions of ensemble types (left) and individual instruments (right) across the TuttiCorpus dataset.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="6">Metrics (MV2H)</td></tr><tr><td>Multi-pitch</td><td>Voice</td><td>Meter</td><td>Harmony</td><td>Note Value</td><td>Total</td></tr><tr><td rowspan="4">ASAP</td><td>Full Instrumentation Pre-training</td><td>70.7</td><td>87.5</td><td>90.9</td><td>55.6</td><td>90.7</td><td>76.1†</td></tr><tr><td>Single Instrumentation Pre-training</td><td>66.2</td><td>86.7</td><td>90.6</td><td>57.1</td><td>90.1</td><td>75.0†</td></tr><tr><td>Without Pre-training</td><td>13.2</td><td>74.2</td><td>78.4</td><td>54.6</td><td>71.5</td><td>53.4†</td></tr><tr><td>Zeng et al. [1]</td><td>63.3</td><td>88.4</td><td></td><td>54.5</td><td>90.7</td><td>74.2†</td></tr><tr><td rowspan="4">Quartets</td><td>Full Instrumentation Pre-training</td><td>97.7</td><td>99.5</td><td>97.8</td><td>83.4</td><td>99.7</td><td>95.6</td></tr><tr><td>Single Instrumentation Pre-training</td><td>93.8</td><td>98.4</td><td>95.3</td><td>83.4</td><td>99.7</td><td>93.0</td></tr><tr><td>Without Pre-training</td><td>46.2</td><td>85.5</td><td>82.9</td><td>56.0</td><td>92.0</td><td>72.5</td></tr><tr><td>Alfaro-Contreras et al. [2]</td><td>70.6</td><td>92.1</td><td>70.7</td><td>97.6</td><td>93.4</td><td>84.9</td></tr><tr><td rowspan="3">Tenor Saxophone</td><td>Full Instrumentation Pre-training</td><td>80.4</td><td>100.0</td><td>96.1</td><td>51.8</td><td>98.0</td><td>85.3</td></tr><tr><td>Without Pre-training</td><td>26.9</td><td>92.3</td><td>85.9</td><td>26.9</td><td>46.8</td><td>55.8</td></tr><tr><td>Martínez-Sevilla et al. [3]</td><td>48.1</td><td>65.2</td><td>61.3</td><td>63.0</td><td>64.5</td><td>60.3</td></tr><tr><td rowspan="3">Alto Saxophone</td><td>Full Instrumentation Pre-training</td><td>82.8</td><td>100.0</td><td>96.6</td><td>46.5</td><td>98.7</td><td>84.9</td></tr><tr><td>Without Pre-training</td><td>23.4</td><td>91.7</td><td>79.3</td><td>20.8</td><td>47.4</td><td>52.5</td></tr><tr><td>Martínez-Sevilla et al. [3]</td><td>40.1</td><td>67.3</td><td>57.6</td><td>67.2</td><td>67.1</td><td>59.9</td></tr></table>

Table 2: Performance comparison (MV2H scores [30]) between TUTTI and baseline models across multiple datasets. Bold values indicate the best performance in each category.<sup>†</sup>Total score is calculated as the average of four available metrics.

The AdamW optimizer [31] is used with a learning rate of 2e-4 for pre-training and 1e-5 for fine-tuning. Pretraining includes a 1000-step warmup followed by a constant learning rate over 32 epochs, with a batch size of 10 per GPU. Pre-training was conducted on 8 H800 GPUs, taking approximately 6 days to complete. Fine-tuning was performed on 6 H800 GPUs with a batch size of 6 per GPU over 32 epochs.

In ablation studies, we investigate the effects of multiinstrumentation learning on TUTTI, considering three settings: 1) omitting pre-training, 2) using only the downstream task-specific data from TuttiCorpus, or 3) utilizing the entire TuttiCorpus, which includes all instrumentations.

In terms of comparative evaluations, we select opensource models that excel in their respective domains for benchmarking. TUTTI is fine-tuned on identical datasets to these models, ensuring fairness in comparison.

## 5.2 Ablation Studies

To better understand the contributing factors to our model’s performance, we conduct ablation studies focusing on two main aspects: 1) the necessity of our large-scale synthetic pre-training paradigm, and 2) the effectiveness of multi-instrumentation representation learning compared to instrumentation-specific pre-training.

As depicted in Table 2, the model trained strictly on the target datasets experiences a severe performance degradation, falling behind the baseline models across all tasks. This aligns with the well-known data-hungry nature of Transformer architectures [17, 32]. Our results clearly demonstrate that without the proposed large-scale synthetic data pre-training, the Transformer struggles to generalize on limited real-world target data. Thus, our synthetic pre-training is the critical enabler that unlocks the high capacity of the Transformer for A2S tasks. While our experiments demonstrate that large-scale pre-training unlocks the potential of purely Transformer-based models, we acknowledge that this data-centric paradigm is architectureagnostic; CNN-based hybrid models would likely exhibit parallel performance gains if exposed to the same scale of pre-training.

We further investigate whether pre-training on a diverse mixture of instrumentations provides an advantage over pre-training on a matched, single-instrumentation distribution. In the “single instrumentation pre-training” setting, we restricted the synthetic pre-training data to match the target task (e.g., using only synthetic piano data for the ASAP piano task, and synthetic string quartets for the Quartets dataset). Note that for the saxophone datasets, this ablation is not applicable since saxophones were entirely excluded from our pre-training corpus.

The results demonstrate that the full-instrumentation pre-trained model outperforms the single-instrumentation pre-trained counterpart. This indicates that multiinstrumentation pre-training is not merely about covering more instrument types, but it fundamentally enhances the model’s underlying representation capability. By learning to disentangle complex, multi-timbral polyphony across diverse instruments, the model learns more robust universal acoustic features and harmonic relationships, which ultimately benefits its performance even on instrumentspecific downstream tasks.

## 5.3 Comparative Evaluations

We compare TUTTI, a generalizable A2S model pretrained on the TuttiCorpus dataset, with several baseline models. The following baseline models have been selected for comparison:

• Zeng et al. [1] employ a convolutional recurrent neural network (CRNN) with a customized hierarchical decoder and the ASAP (piano) [13] dataset as fine-tuning data.

• Alfaro-Contreras et al. [2] utilize a CNN with 2- D positional encoding and a Transformer decoder trained on the string quartet dataset.

• Martínez-Sevilla et al. [3] use a CRNN trained on human-recorded saxophone data. As the saxophone is absent from the TuttiCorpus instrumentation, this comparison serves to evaluate the model’s transfer learning capability on unseen instruments.

To ensure a comprehensive assessment, we adopt the MV2H metric [30], which comprises five sub-metrics: multi-pitch detection, voice separation, metrical alignment, harmonic detection, and note value accuracy. These values are averaged to produce the total MV2H score. Note that for Zeng et al. [1], the metrical alignment metric is unavailable; thus, its total score is calculated by averaging the remaining four components.

Table 2 presents the quantitative results. Overall, TUTTI pre-trained with full instrumentation outperforms all baselines, demonstrating superior transcription capabilities across both polyphonic and monophonic tasks. On the ASAP dataset, which involves expressive timing and complex polyphony, our method achieves a significant improvement in multi-pitch accuracy, outperforming the baseline by 7.4 points. While yielding competitive results in voice tracking, TUTTI achieves a higher total score, proving its robustness in handling expressive real-world piano recordings.

For the Quartets dataset, our model demonstrates substantial margins over the baseline across almost all metrics, increasing the total score from 84.9 to 95.6. The nearperfect scores in voice (99.5) and note value (99.7) indicate that our approach reliably transcribes dense multi-track arrangements.

A key strength of TUTTI is its transfer learning capability. On the Saxophone datasets, our model successfully generalizes to these unseen instruments despite their total absence from the TuttiCorpus training set. Notably, since the saxophone is a monophonic instrument, voice accuracy serves as a diagnostic indicator of whether the model correctly identifies its single-voice nature. Failing to reach 100% implies the model erroneously predicts multiple simultaneous voices—a hallucination likely caused by insufficient acoustic understanding of the instrument. Our fully pre-trained model achieves this theoretical ceiling (100.0) on both datasets, whereas the task-specific baseline and the non-pre-trained variant do not. We attribute this to the diverse multi-timbral representations learned during pretraining: by exposure to a wide variety of instruments and ensembles, TUTTI acquires a more robust acoustic grounding that generalizes to unseen timbres without overfitting to a single instrument’s characteristics.

## 6. CONCLUSION

We introduced TUTTI, a unified, Transformer encoderdecoder framework that addresses data scarcity and architectural complexity bottlenecks in generalizable A2S transcription. By pre-training on TuttiCorpus—our largescale dataset comprising over 363,000 synthetic multiinstrumentation audio-score pairs—we demonstrated that a data-centric paradigm can effectively compensate for the absence of local inductive biases typically provided by CNN acoustic front-ends.

Extensive objective evaluations reveal that TUTTI outperforms established task-specific baselines across diverse polyphonic and monophonic tasks, establishing new overall state-of-the-art results. Crucially, the model exhibits remarkable transfer learning capability, achieving theoretical voice accuracy on unseen real-world instruments, such as the saxophone, strictly through the robust representational power learned from synthetic data.

Finally, we release the TuttiCorpus dataset as a rich and highly scalable resource to propel future research in unified A2S transcription. Although TUTTI marks a significant step forward, extending our mechanism to seamlessly handle unconstrained, full-piece transcriptions remains a promising avenue for future work.

## 7. ACKNOWLEDGMENTS

This work was supported by the following funding source: Special Program of National Natural Science Foundation of China (Grant No. T2341003).

## 8. REFERENCES

[1] W. Zeng, X. He, and Y. Wang, “End-to-end real-world polyphonic piano audio-to-score transcription with hierarchical decoding,” in Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI), 2024, pp. 7788–7795.

[2] M. Alfaro-Contreras, A. Ríos-Vila, J. J. Valero-Mas, and J. Calvo-Zaragoza, “A transformer approach for polyphonic audio-to-score transcription,” in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 706–710.

[3] J. C. Martínez-Sevilla, M. Alfaro-Contreras, J. J. Valero-Mas, and J. Calvo-Zaragoza, “Insights into endto-end audio-to-score transcription with real recordings: A case study with saxophone works,” in Interspeech 2023, 2023, pp. 2793–2797.

[4] J. Qian, H. Meng, T. Zheng, P. Zhu, H. Lin, Y. Dai, H. Xie, W. Cao, R. Shang, J. Wu, H. Liu, H. Wen, J. Zhao, Z. Jiang, Y. Chen, S. Yin, M. Tao, J. Wei, L. Xie, and X. Wang, “Soulx-singer: Towards high-quality zero-shot singing voice synthesis,” 2026. [Online]. Available: https://arxiv.org/abs/2602.07803

[5] A. Cogliati, D. Temperley, and Z. Duan, “Transcribing human piano performances into music notation.” in ISMIR, 2016, pp. 758–764.

[6] E. Nakamura, E. Benetos, K. Yoshii, and S. Dixon, “Towards complete polyphonic music transcription: Integrating multi-pitch detection and rhythm quantization,” in 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018, pp. 101–105.

[7] K. Shibata, E. Nakamura, and K. Yoshii, “Non-local musical statistics as guides for audio-to-score piano transcription,” Information Sciences, vol. 566, pp. 262–280, 2021.

[8] R. G. C. Carvalho and P. Smaragdis, “Towards endto-end polyphonic music transcription: Transforming music audio directly to a score,” in 2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2017, pp. 151–155.

[9] M. A. Román, A. Pertusa, and J. Calvo-Zaragoza, “An end-to-end framework for audio-to-score music transcription on monophonic excerpts.” in ISMIR, 2018, pp. 34–41.

[10] M. A. Román, A. Pertusa, and J. Calvo-Zaragoza, “A holistic approach to polyphonic music transcription with neural networks,” in Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR), 2019, pp. 731–737.

[11] L. Liu, V. Morfi, and E. Benetos, “Joint multi-pitch detection and score transcription for polyphonic piano music,” in ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021, pp. 281–285.

[12] A. Morsi and X. Serra, “Bottlenecks and solutions for audio to score alignment research,” in Rao P, Murthy H, Srinivasamurthy A, Bittner R, Caro Repetto R, Goto M, Serra X, Miron M, editors. Proceedings of the 23nd International Society for Music Information Retrieval Conference (ISMIR 2022); 2022 Dec 4-8; Bengaluru, India.[Canada]: International Societyfor Music Information Retrieval; 2022. p. 272-9. International Society for Music Information Retrieval (ISMIR), 2022.

[13] F. Foscarin, A. McLeod, P. Rigaux, F. Jacquemard, and M. Sakai, “ASAP: a dataset of aligned scores and performances for piano transcription,” in International Society for Music Information Retrieval Conference (IS-MIR), 2020, pp. 534–541.

[14] W. Chen, J. Keast, J. Moody, C. Moriarty, F. Villalobos, V. Winter, X. Zhang, X. Lyu, E. Freeman, J. Wang, S. Cai, and K. M. Kinnaird, “Data usage in mir: History & future recommendations,” in Proceedings ofthe 20th International Society for Music Information Retrieval Conference (ISMIR), Delft, Netherlands, 2019, pp. 11–18.

[15] V. Arroyo, J. J. Valero-Mas, J. Calvo-Zaragoza, and A. Pertusa, “Neural audio-to-score music transcription for unconstrained polyphony using compact output representations,” in ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022, pp. 4603– 4607.

[16] S. d’Ascoli, H. Touvron, M. L. Leavitt, A. S. Morcos, G. Biroli, and L. Sagun, “Convit: Improving vision transformers with soft convolutional inductive biases,” in International conference on machine learning. PMLR, 2021, pp. 2286–2296.

[17] R. Shao and X.-J. Bi, “Transformers meet small datasets,” IEEE Access, vol. 10, pp. 118 454–118 464, 2022.

[18] I. Sutskever, O. Vinyals, and Q. V. Le, “Sequence to sequence learning with neural networks,” Advances in neural information processing systems, vol. 27, 2014.

[19] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[20] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al., “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 8, p. 9, 2019.

[21] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified textto-text transformer,” Journal of machine learning research, vol. 21, no. 140, pp. 1–67, 2020.

[22] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan et al., “Deepseek-v3 technical report,” arXiv preprint arXiv:2412.19437, 2024.

[23] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi et al., “Deepseekr1: Incentivizing reasoning capability in llms via reinforcement learning,” arXiv preprint arXiv:2501.12948, 2025.

[24] S. Wu, Y. Wang, X. Li, F. Yu, and M. Sun, “Melodyt5: A unified score-to-score transformer for symbolic music processing,” in Proceedings of the 25th International Societyfor Music Information Retrieval Conference (ISMIR), 2024.

[25] Y. Wang, S. Wu, J. Hu, X. Du, Y. Peng, Y. Huang, S. Fan, X. Li, F. Yu, and M. Sun, “Notagen: Advancing musicality in symbolic music generation with large language model training paradigms,” in Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI), 2025, pp. 10 207– 10 215.

[26] S. Wu, X. Li, F. Yu, and M. Sun, “Tunesformer: Forming irish tunes with control codes by bar patching,” in Proceedings of the 2nd Workshop on Human-Centric Music Information Retrieval 2023 co-located with the 24th International Society for Music Information Retrieval Conference (ISMIR 2023), Milan, Italy,November 10, 2023, 2023.

[27] S. Wu, D. Yu, X. Tan, and M. Sun, “Clamp: Contrastive language-music pre-training for cross-modal symbolic music information retrieval,” in Proceedings of the 24th International Society for Music Information Retrieval Conference, ISMIR 2023, Milan, Italy, November 5-9, 2023, 2023.

[28] D. Jeong, T. Kwon, Y. Kim, K. Lee, and J. Nam, “Virtuosonet: A hierarchical rnn-based system for modeling expressive piano performance.” in ISMIR, 2019, pp. 908–915.

[29] S. D. Peter, “Online symbolic music alignment with offline reinforcement learning,” in Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), 2023.

[30] A. McLeod and M. Steedman, “Evaluating automatic polyphonic music transcription,” in International Society for Music Information Retrieval Conference (IS-MIR), 2018, pp. 42–49.

[31] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in 7th International Conference on Learning Representations, ICLR 2019,New Orleans, LA, USA, May 6-9, 2019, 2017.

[32] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, “Scaling laws for neural language models,” arXiv preprint arXiv:2001.08361, 2020.