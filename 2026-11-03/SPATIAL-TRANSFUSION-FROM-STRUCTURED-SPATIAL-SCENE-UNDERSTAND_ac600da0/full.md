# SPATIAL TRANSFUSION: FROM STRUCTURED SPATIAL SCENE UNDERSTANDING TO GENERATION AND EDITING

Jaeyeon Kim<sup>1∗</sup> Sebastian Braun<sup>2</sup> Hannes Gamper<sup>2</sup>

<sup>1</sup>Carnegie Mellon University <sup>2</sup> Microsoft Corporation

## ABSTRACT

Spatial audio conveys not only what sounds are present, but where they originate and how they interact with the environment. Recent audio-language models increasingly understand and reason about such scenes, yet spatial audio understanding, generation, and editing remain largely separate. We present Spatial Transfusion, a unified first-order ambisonics (FOA) framework that extends a pretrained spatial audio understanding model to generation and editing through flow matching over VAE latents. We formulate editing as a structured scene transformation over a taxonomy of atomic edits. Keeping the pretrained backbone frozen, we introduce modality-based LoRA routing and a curriculum that progressively adds generation, editing, and natural-language interaction. Experiments show that Spatial Transfusion largely preserves spatial understanding while outperforming full fine-tuning in generation and editing. The resulting model supports controllable FOA generation and editing through both structured and naturallanguage interfaces. Audio samples are available at https: //microsoft.github.io/spatial-transfusion/.

Index Terms— Audio-Language Model, Unified Framework, Transfusion, Spatial Audio Editing, First-Order Ambisonics

## 1. INTRODUCTION

Humans perceive the acoustic world spatially. Spatial audio provides an immersive representation of acoustic scenes by capturing not only what sounds are present, but also where they originate and how they interact with the surrounding environment. Compared with conventional mono audio, spatial audio scenes involve additional factors such as source locations and distances, room acoustics, and diffuse background components, which make both understanding and generation substantially more challenging. Recent large audio-language models (LALMs) have started extending their capabilities to spatial audio by integrating a spatial audio encoder into a large language model (LLM), enabling description of and reasoning about complex acoustic environments from binaural [1, 2], first-order ambisonics (FOA) [3–5], or arbitrary microphone-array [6] input.

In parallel, several works have explored unified models that support both understanding and generation in the audio domain, modeling audio either as discrete codec tokens [7, 8] or as continuous latents [9,10]. These models share the intuition that understanding and generation are two interfaces to a shared underlying representation, so that generative capability can be built on top of representations learned for perception [7, 11].

In spatial audio, however, understanding, generation, and editing are mainly addressed by specialized models rather than a unified framework. Existing spatial audio generation methods are conditioned on text [12–14] or video [15, 16], but offer limited per-source control in multi-source scenes, where each source has its own direction, distance, and temporal activity. Editing has received even less attention: recent methods have begun to target stereo [9,17], but their spatial operations are limited to left-right panning without control over full 3D direction or separate spatial treatment of directional sources and diffuse background. This motivates a natural question: can the spatial scene understanding of an LALM be extended into a unified framework for generation and editing? We hypothesize that controllable generation and editing benefit from a model that first builds a structured and fine-grained understanding of the acoustic scene. If a model can understand scene attributes such as source positions and temporal activities, the same representations can provide a foundation for controllable synthesis and manipulation.

To investigate this hypothesis, we introduce Spatial Transfusion, the first unified framework that supports understanding, generation, and editing of FOA scenes. We adopt FOA for its playback-agnostic representation and its wide use in both research and consumer platforms. We build Spatial Transfusion on a variant of Sci-Phi [4], a recent LALM that produces structured scene descriptions, and extend it to generation and editing under the Transfusion paradigm [9, 11, 18]. Specifically, we combine an autoregressive understanding objective on text with a flow-matching [19, 20] generation objective on continuous audio latents within a shared Transformer backbone. To support editing, we formulate it as a structured scene transformation and define a taxonomy of atomic edit operations spanning semantic, acoustic, spatial, and temporal dimensions.

A key challenge is that Transfusion-style approaches typically rely on large-scale pretraining with full-model training [9, 11, 18], a regime that is impractical for spatial audio, where paired data is far more limited and complex to simulate. Instead, we adapt an existing spatial audio LALM using modality-based LoRA routing, together with a curriculum that progressively extends understanding capabilities to generation and editing. Experiments show that the resulting model largely retains the spatial understanding of the base model while gaining generation and editing through structured interfaces, and that instruction tuning further extends these capabilities to natural-language interaction. Notably, our adaptation substantially outperforms full fine-tuning in generation and editing, and an ablation shows that routing by modality is the most balanced choice among adaptation strategies.

## 2. SPATIAL TRANSFUSION

## 2.1. Background

FOA-VAE. To model FOA audio in a compact latent space, we adopt FOA-VAE [21], a continuous variational autoencoder for fourchannel FOA at 16 kHz. It operates on the complex STFT, encoding the real and imaginary parts of all channels into a continuous latent with a fully convolutional encoder and reconstructing the spectrum with a symmetric decoder. Beyond the standard reconstruction, adversarial, and KL objectives, it is trained with a spatial consistency loss that penalizes angular mismatch between the intensity vectors [22] of the input and the reconstruction, so that directional cues survive compression. The resulting latent runs at 7.8 Hz with 512 dimensions each for the mean and variance, retaining both semantic content and perceptually meaningful spatial cues.

![](images/09cd3f707136c6f6905dd24dde36c73af6cbb67f57bb4839a017df77f1e26801.jpg)  
Fig. 1. Overview of Spatial Transfusion. A shared Transformer backbone serves (a) understanding with a cross-entropy loss on text, and (b) generation and (c) editing with a flow-matching loss on target FOA-VAE latents. In editing, clean source latents are additionally concatenated channel-wise with the noised target latents.

Sci-Phi. Sci-Phi [4] extends Phi-4-Multimodal (Phi-4-MM) [23], which supports mono audio understanding, to spatial acoustic scenes by adding a spatial audio encoder and training an additional LoRA within the LLM. Given an FOA recording, it produces a structured scene description that enumerates the room, a diffuse background, and up to five directional sources with their semantic, acoustic, spatial, and temporal attributes, as shown in Fig. 2. Training data is simulated by convolving sound events with room impulse responses from image-source rooms [24], at random locations, levels, and onsets, and mixing them with a diffuse background; a structured scene description lists these parameters in JSON text format.

Our work starts from a variant of Sci-Phi that replaces the original spatial encoder with the FOA-VAE encoder. The latent mean and variance are concatenated channel-wise, projected to the LLM embedding dimension, and modeled jointly with the mono audio and text embeddings through a LoRA of rank 16. This understanding path is architecturally retained in Spatial Transfusion (Fig. 1(a)).

## 2.2. Architecture

Transfusion-Based Modeling. Spatial Transfusion extends Sci-Phi into a unified model that supports understanding, generation, and editing within a shared Transformer backbone (Fig. 1). Following the Transfusion paradigm [11, 18], text tokens are modeled autoregressively with a cross-entropy objective under causal attention, while target FOA-VAE latents are modeled with a flow-matching objective under bidirectional attention, since they are denoised jointly rather than emitted one at a time. Specifically, the target audio is encoded by the FOA-VAE into a posterior $\bar { \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ , from which we sample a 512-dimensional target latent $z _ { 1 } .$ Given noise z ∼ $\mathcal { N } ( 0 , I )$ and t drawn from a logit-normal distribution [25], we form $z _ { t } ~ = ~ ( 1 - t ) z _ { 0 } + t z _ { 1 }$ and train the model to predict the velocity [19, 20]:

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { t , z _ { 0 } , z _ { 1 } } \left. v _ { \theta } ( z _ { t } , t , c ) - ( z _ { 1 } - z _ { 0 } ) \right. ^ { 2 } ,\tag{1}
$$

where c is the conditioning context. The two objectives are combined as $\mathcal { L } = \lambda \mathcal { L } _ { \mathrm { C E } } + \mathcal { L } _ { \mathrm { F M } }$

The same backbone serves all three tasks, which differ only in what occupies the context. Understanding takes an FOA input and an optional text instruction, and decodes either a scene description or an answer. Generation takes a scene description and predicts the velocity toward the target latents. Editing takes a source scene together with an edit instruction, and predicts the velocity toward the latents of the edited scene. In editing, the source scene is conditioned through two paths: it is passed through the understanding path as a conditioning input, and, since this alone fails to preserve the input audio content, the clean source latents are additionally concatenated channel-wise with the noised target latents, following common practice in image and audio editing [17, 26]. Three modules are added for the latent path: a linear projector that maps noised latents (and, in editing, the channel-concatenated source latents) into the LLM embedding space, a timestep embedding that conditions on t, and a linear head that predicts the velocity from the final hidden states. All three are randomly initialized and add about 13M parameters.

All three tasks are trained jointly on mixed batches, with the cross-entropy loss applied to output text tokens and the flowmatching loss to target latents. We set $\lambda ~ = ~ 0 . 1$ to balance the two capabilities. At inference, the task is determined by the context: the model either decodes the output text autoregressively, or samples the target latents by integrating the predicted velocity field from z<sub>0</sub> to z<sub>1</sub>, which the FOA-VAE decoder converts back to a waveform.

Modality-Based LoRA Routing. Rather than fully fine-tuning the Sci-Phi backbone, we adapt it with LoRA [27]. Instead of applying a single LoRA to all tokens, we route tokens to different LoRA adapters, similar to the Mixture-of-Transformers (MoT) structure [28] used by recent Transfusion models [11, 29]. In MoT, tokens are assigned to paths by their type, where each path has separate weights for the query, key, and value projections, feed-forward layers, and normalization. These paths interact only through the shared self-attention over the interleaved sequence, so each path has its own parameters while the model still attends across paths.

Whereas MoT replicates the entire backbone per path, we keep the backbone shared and attach a separate LoRA adapter to each path. This gives each path its own parameter space through its adapter without duplicating the backbone. Among possible routing strategies, we adopt modality-based routing [28, 29]: audio tokens, both the input audio embeddings and the target latents, share one path, and text tokens use another. This allows understanding and generation to share audio-adapter parameters, encouraging transfer between the two capabilities. The audio path uses a larger LoRA rank (640 vs. 16), since it needs to learn a new generative capability. QK-normalization [30] is applied in all attention layers for training stability. The new LoRA modules are applied to the LLM weights and trained together with the three new modules. The Phi-4-MM backbone, the encoders, and the existing LoRAs of Sci-Phi remain frozen. In total, the audio adapter adds about 923M parameters and the text adapter 7M, for 943M trainable parameters including the new modules, compared with 3.85B for full fine-tuning.

Table 1. Taxonomy of atomic spatial audio edits by dimension and target. Move relocates a source in azimuth, elevation, and distance by selecting a new room impulse response, and thus covers the full 3D position in a single operation.
<table><tr><td>Dimension</td><td>Directional source</td><td>Diffuse background</td></tr><tr><td>Semantic</td><td>Add, Remove, Replace</td><td>Add, Remove, Replace</td></tr><tr><td>Spatial</td><td>Move</td><td></td></tr><tr><td>Acoustic</td><td>Change level, Pitch shift</td><td>Change level</td></tr><tr><td>Temporal</td><td>Shift time, Time stretch</td><td></td></tr></table>

![](images/653cc6c07893a088c65352451e5d5c53f2541adcd61eaf58c2c4d12c1a5725f6.jpg)  
Fig. 2. Structured representations used in Stages 1–3. (a) A scene description lists directional sources with their attributes plus room and background fields; it is the understanding output and the generation input. (b) An edit instruction names an operation from Table 1, its target, and the modification to apply; it is the editing input.

## 2.3. Spatial Audio Editing

Editing Taxonomy. While generation data can be obtained by simply reversing the understanding examples of Sci-Phi, editing data requires the additional constraint that the input scene must be preserved while only the targeted aspect changes. We therefore formulate spatial audio editing as a structured scene transformation. Extending the atomic-edit formulation of SmartDJ [17] to the FOA setting, we define a taxonomy of atomic edits spanning semantic, acoustic, spatial, and temporal dimensions, applied to both foreground sources and the diffuse background, resulting in 12 operations. The complete taxonomy is given in Table 1.

Data Simulation. To generate paired editing data at scale, we extend the Sci-Phi simulation pipeline. Each example is created by perturbing exactly the scene factor targeted by the edit while keeping everything else fixed: moving a source replaces its room impulse response, replacing a source swaps its audio file, and so on. Rerendering the perturbed scene yields an (input, instruction, target) triplet with exact ground truth for both the edited and the preserved content, which we generate for every operation in the taxonomy.

Table 2. Training curriculum. Sampling ratio is given as understanding : generation : editing : QA. Sample counts are the number of unique pairs available at each stage.
<table><tr><td>Stage</td><td>Steps</td><td>Und/Gen</td><td>Edit</td><td>QA</td><td>Ratio</td></tr><tr><td>1</td><td>30K</td><td>2.0M</td><td></td><td>一</td><td> $0 . 2 : 0 . 8 : 0 : 0$ </td></tr><tr><td>2</td><td>70K</td><td>2.0M</td><td>1.5M</td><td></td><td> $0 . 2 : 0 . 3 : 0 . 5 : 0$ </td></tr><tr><td>3</td><td>100K</td><td>3.6M</td><td>1.4M</td><td></td><td> $0 . 2 : 0 . 4 : 0 . 4 : 0$ </td></tr><tr><td>4</td><td>50K</td><td>1.8M</td><td>0.7M</td><td>250K</td><td> $0 . 2 : 0 . 3 : 0 . 3 : 0 . 2$ </td></tr></table>

Table 3. Understanding results on the FOA-MEIR-based test set.
<table><tr><td colspan="6">Model Tuple↑ Nsrc Acc.↑ Src CLAP↑ Dir. Err.↓ Time IoU↑</td></tr><tr><td>Sci-Phi</td><td>0.798</td><td>0.674</td><td>0.670</td><td>13.6</td><td>0.754</td></tr><tr><td>Full fine-tuning</td><td>0.805</td><td>0.654</td><td>0.679</td><td>12.6</td><td>0.726</td></tr><tr><td>Spatial Transfusion</td><td>0.812</td><td>0.635</td><td>0.683</td><td>14.3</td><td>0.754</td></tr></table>

## 2.4. Training Curriculum

We adopt a four-stage curriculum that extends one capability at a time for effective adaptation:

• Stage 1: Single-source understanding and generation. The model learns to map single-source FOA scenes to structured descriptions and back, acquiring basic generative capability.

• Stage 2: Single-source editing. Editing is added next, since an edit is a conditional generation that must also preserve the input.

• Stage 3: Multi-source scenes. All three tasks are extended to scenes with up to five directional sources.

• Stage 4: Instruction tuning. The JSON-based capabilities are mapped onto free-form natural language: for each of understanding, generation, and editing, half of the samples use JSON and half use natural-language captions or instructions. Question-answering (QA) samples are additionally added to understanding.

The first three stages use structured JSON (Fig. 2) rather than free-form text. JSON defines every attribute explicitly, which gives fine-grained control for generation, and matches the format Sci-Phi already produces. Moreover, mapping JSON to captions is easier than the reverse, as captions may lack explicit scene details. Learning the structured form first thus supports both interfaces with different levels of controllability.

Table 2 lists the data at each stage. Stage 3 keeps 1.0M of the Stage 1–2 samples and adds 2.6M multi-source scenes; for editing, it keeps 0.45M of the Stage 2 samples and adds 0.95M. Stage 4 draws half of its samples from Stage 3, with natural-language captions generated by GPT-5.4-mini [31], and adds QA pairs from Sci-Phi-Chat [32]. In total, 4.6M unique scenes are used for understanding and generation and 2.4M for editing.

## 3. EXPERIMENTS

## 3.1. Experimental Setup

Configuration. We train with AdamW at a learning rate of $1 \times 1 0 ^ { - 4 }$ with linear warmup followed by cosine decay to $1 \times 1 0 ^ { - 5 }$ . Optimization settings are held constant across curriculum stages. At inference, understanding uses greedy decoding, while generation and editing use 50 flow-matching steps with classifier-free guidance [33] applied to the text condition at scales of 3.5 and 4.0, respectively.

Baselines. Evaluation uses a held-out set rendered with sources, real room impulse responses, and ambient noise from the FOA-MEIR dataset [34]. Existing spatial audio generators differ in their conditioning interfaces and do not support our structured scene specification. To assess the extension from understanding to generation and editing, we instead compare Spatial Transfusion with full finetuning of the same pretrained backbone, using the same training data and curriculum and a learning rate of $5 \times 1 0 ^ { - 5 }$ . This provides a controlled comparison of adaptation strategies.

Table 4. Generation results on the FOA-MEIR-based test set.
<table><tr><td>Model</td><td>FAD-PSELD↓ FAD-CLAP↓ CLAP-TA↑ FOA CoV.↓</td><td></td><td></td><td></td></tr><tr><td>Full fine-tuning</td><td>3.82</td><td>379</td><td>0.232</td><td>0.196</td></tr><tr><td>Spatial Transfusion</td><td>0.964</td><td>229</td><td>0.264</td><td>0.166</td></tr><tr><td>VAE recon. (topline)</td><td>0.103</td><td>47.6</td><td>0.340</td><td>0.013</td></tr></table>

Table 5. Editing results on the FOA-MEIR-based test set.
<table><tr><td>Model</td><td>EditFLAM↑ FAD-PSELD↓ CLAP-AA↑ CLAP-TA↑</td><td></td><td></td><td></td></tr><tr><td>Copy-input</td><td>0.0</td><td>0.005</td><td>1.000</td><td>0.324</td></tr><tr><td>Target-JSON gen.</td><td>19.7</td><td>1.19</td><td>0.701</td><td>0.248</td></tr><tr><td>Full fine-tuning</td><td>15.5</td><td>0.596</td><td>0.795</td><td>0.275</td></tr><tr><td>Spatial Transfusion</td><td>17.8</td><td>0.323</td><td>0.841</td><td>0.306</td></tr><tr><td>Ground truth (topline)</td><td>28.9</td><td>0.000</td><td>0.913</td><td>0.344</td></tr></table>

The Sci-Phi variant (Sec. 2.1) serves as the understanding baseline, and FOA-VAE reconstruction is the generation topline. For editing, we include copy-input and target-JSON baselines to contrast input preservation with scene regeneration. Copy-input returns the source audio unchanged, whereas target-JSON uses Spatial Transfusion to generate audio solely from the ground-truth target scene description. Ground-truth target audio serves as the editing reference. Unless noted, we evaluate the model after Stage 3, with structured JSON as the understanding output and generation and editing input. Metrics. Understanding follows the Sci-Phi protocol [4]: Tuple Score summarizes semantic, spatial, and temporal accuracy, while source-count accuracy (Nsrc Acc.), source-level CLAP similarity (Src CLAP), direction error (Dir. Err.), and temporal IoU (Time IoU) provide detailed assessments. Generation is evaluated using FAD-PSELD for spatial–semantic fidelity, FAD-CLAP for audio fidelity, and CLAP-TA for alignment with the target description. FOA covariance loss [35] assesses spatial-channel structure via the L1 distance between the normalized channel covariance matrices of generated and target audio. We use FAD-PSELD, computed with PSELD-Nets [36], as the primary spatial metric following a recent sensitivity analysis [37]. Editing additionally uses EditFLAM [9], based on operation-specific changes in FLAM [38] probabilities between input and edited audio, and CLAP-AA for similarity between these signals. For editing, EditFLAM and CLAP-TA are evaluated only on operations that change source or background semantic content. All CLAP-based metrics use MS-CLAP [39], and FAD-CLAP is computed with fadtk [40].

## 3.2. Results

Main results. Table 3 shows that Spatial Transfusion largely retains the original Sci-Phi model’s understanding performance. Full finetuning performs similarly. These results indicate that adding generation and editing within the unified framework does not compromise spatial understanding. For generation (Table 4), Spatial Transfusion outperforms full fine-tuning on all metrics, reducing FAD-PSELD from 3.82 to 0.964. Although a gap remains to FOA-VAE reconstruction, these results favor adaptation with modality-based LoRA routing over full fine-tuning in our setup.

For editing (Table 5), target-JSON generation uses the groundtruth target scene description and achieves higher EditFLAM than

Table 6. Generalization to natural-language interfaces. Stage 4 is evaluated with both JSON and natural-language (NL) inputs. Sci-Phi + GPT answers questions by prompting GPT-5.4-mini with Sci-Phi’s description. QA is evaluated on the MASU benchmark from Sci-Phi-Chat.
<table><tr><td rowspan="3">Model</td><td rowspan="3">QA Acc.↑</td><td rowspan="3">Und.</td><td>Gen.</td><td colspan="2">Edit</td></tr><tr><td>Tuple↑</td><td></td><td>FAD-PSELD↓ EditFLAM↑ FAD-PSELD↓</td></tr><tr><td>Sci-Phi + GPT</td><td>0.208</td><td></td><td>一</td><td>一</td><td></td></tr><tr><td>Stage 3 (JSON)</td><td></td><td>0.812</td><td>0.964</td><td>17.8</td><td>0.323</td></tr><tr><td>Stage 4 (JSON)</td><td></td><td>0.813</td><td>0.991</td><td>18.7</td><td>0.620</td></tr><tr><td>Stage 4 (NL)</td><td>0.476</td><td></td><td>1.39</td><td>20.9</td><td>0.772</td></tr></table>

Table 7. Ablation on LoRA routing. Best; second best.
<table><tr><td></td><td>Und.</td><td>Gen.</td><td colspan="2">Edit</td></tr><tr><td>Adaptation</td><td>Tuple↑</td><td>FAD-PSELD↓</td><td>EditFLAM↑</td><td>FAD-PSELD↓</td></tr><tr><td>Dense LoRA</td><td>0.798</td><td>1.13</td><td>18.8</td><td>1.13</td></tr><tr><td>Task-based routing</td><td>0.814</td><td>1.40</td><td>9.8</td><td>0.321</td></tr><tr><td>Modality-based routing</td><td>0.812</td><td>0.964</td><td>17.8</td><td>0.323</td></tr></table>

Spatial Transfusion, but its lower CLAP-AA and higher FAD-PSELD indicate poorer input preservation and distributional fidelity. Copy-input attains near-zero FAD-PSELD because atomic edits leave most of the scene unchanged, yet its zero EditFLAM reflects the absence of edits. Spatial Transfusion achieves positive EditFLAM while maintaining higher CLAP-AA and lower FAD-PSELD than target-JSON generation, indicating that it applies edits while better preserving the input scene. It also outperforms full fine-tuning on all editing metrics, supporting our adaptation.

Natural-language generalization. Table 6 shows that Stage 4 instruction tuning largely retains the model’s structured capabilities, with some loss in editing fidelity. With natural-language inputs, generation and editing remain broadly comparable to their JSON counterparts, although fidelity decreases, possibly due to less explicit scene specifications. On QA, the instruction-tuned model outperforms the Sci-Phi + GPT cascade, demonstrating that the adapted model can be extended to spatial question answering. Together, these results show that our training curriculum extends structured scene capabilities to free-form natural-language interaction while largely preserving performance with the structured interface.

Ablation on LoRA routing. Table 7 compares modality-based routing with two alternative adaptation strategies. Dense LoRA applies a single rank-640 LoRA to all tokens, giving 936M trainable parameters. Task-based routing follows the understanding–generation split used in BAGEL [11]: input audio embeddings and text use the understanding path, while target latents use the generation path. These alternatives exhibit different trade-offs: Dense LoRA favors edit effectiveness at the expense of scene fidelity, whereas task-based routing maintains understanding but yields weaker generation and less effective editing. Modality-based routing provides the most balanced performance across tasks, motivating its use in Spatial Transfusion.

## 4. CONCLUSION

We presented Spatial Transfusion, a unified FOA framework that extends spatial audio understanding to generation and editing. With the pretrained backbone frozen, modality-based LoRA routing outperforms full fine-tuning in generation and editing while preserving understanding. Our curriculum extends these capabilities to natural language, supporting both structured and free-form interfaces. Future work includes extension to real-world recordings.

## 5. COMPLIANCE WITH ETHICAL STANDARDS

This study uses simulated scenes and publicly available datasets. No ethical approval was required.

## 6. ACKNOWLEDGMENTS

This work was conducted during an internship at Microsoft. S. Braun and H. Gamper are employees of Microsoft Corporation. The authors have no other relevant financial or non-financial interests to disclose. AI tools were used for language editing.

## 7. REFERENCES

[1] Z. Zheng, P. Peng, Z. Ma, et al., “BAT: Learning to reason about spatial sounds with large language models,” in ICML, 2024.

[2] S. Biswas, M. Khan, and B. Islam, “OWL: Geometry-aware spatial reasoning for audio large language models,” in ICLR, 2026.

[3] C. Tang, W. Yu, G. Sun, et al., “Can large language models understand spatial audio?,” in Interspeech, 2024.

[4] X. Jiang, H. Gamper, and S. Braun, “Sci-Phi: A large language model spatial audio descriptor,” IEEE OJSP, 2026.

[5] Z. Zhu, Y. Chen, Y. Shao, et al., “Spatial-Omni: Spatial audio understanding integration in multimodal LLMs via FOA encoding,” arXiv:2606.10738, 2026.

[6] A. Dementyev, W. Zulfikar, S. Hersek, et al., “PhaseCoder: Microphone geometry-agnostic spatial audio understanding for multimodal LLMs,” in ICML, 2026.

[7] J. Tian, S.-G. Lee, Z. Kong, et al., “UALM: Unified audio language model for understanding, generation and reasoning,” in ICLR, 2026.

[8] J. Tian, H. Wang, B.-H. Su, et al., “Bagpiper: Solving openended audio tasks via rich captions,” in COLM, 2026.

[9] W. Chen, P. Seetharaman, R. Kumar, et al., “AudioChat: Unified audio storytelling, editing, and understanding with transfusion forcing,” in ICML, 2026.

[10] H. Wang, Y. Yang, Z. Tian, et al., “UAT: Unified audiotext diffusion for audio generation, editing, and captioning,” arXiv:2606.04939, 2026.

[11] C. Deng, D. Zhu, K. Li, et al., “Emerging properties in unified multimodal pretraining,” arXiv:2505.14683, 2025.

[12] M. Heydari, M. Souden, B. Conejo, et al., “ImmerseDiffusion: A generative spatial audio latent diffusion model,” in ICASSP, 2025.

[13] C. Templin, Y. Zhu, and H. Wang, “Generating moving 3D soundscapes with latent diffusion models,” in ICASSP, 2026.

[14] P. Sun, S. Cheng, X. Li, et al., “Both ears wide open: Towards language-driven spatial audio generation,” in ICLR, 2025.

[15] J. Kim, H. Yun, and G. Kim, “ViSAGe: Video-to-spatial audio generation,” in ICLR, 2025.

[16] J. Kim, H. Yun, and G. Kim, “Towards scene-aware video-tospatial audio generation,” IJCV, 2026.

[17] Z. Lan, Y. Hao, and M. Zhao, “SmartDJ: Declarative audio editing with audio language model,” in ICLR, 2026.

[18] C. Zhou, L. Yu, A. Babu, et al., “Transfusion: Predict the next token and diffuse images with one multi-modal model,” in ICLR, 2025.

[19] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, et al., “Flow matching for generative modeling,” in ICLR, 2023.

[20] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” in ICLR, 2023.

[21] H. Gamper and S. Braun, “Towards spatially-aware autoencoders for generative audio,” Submitted to ICASSP 2027.

[22] V. Pulkki, A. Politis, M.-V. Laitinen, et al., “First-order directional audio coding (DirAC),” in Parametric Time-Frequency Domain Spatial Audio, chapter 5, pp. 89–140. John Wiley & Sons, Ltd, 2017.

[23] A. Abouelenin, A. Ashfaq, A. Atkinson, et al., “Phi-4-Mini technical report: Compact yet powerful multimodal language models via mixture-of-LoRAs,” arXiv:2503.01743, 2025.

[24] J. B. Allen and D. A. Berkley, “Image method for efficiently simulating small-room acoustics,” J. Acoust. Soc. Am., 1979.

[25] P. Esser, S. Kulal, A. Blattmann, et al., “Scaling rectified flow transformers for high-resolution image synthesis,” in ICML, 2024.

[26] T. Brooks, A. Holynski, and A. A. Efros, “InstructPix2Pix: Learning to follow image editing instructions,” in CVPR, 2023.

[27] E. J. Hu, Y. Shen, P. Wallis, et al., “LoRA: Low-rank adaptation of large language models,” in ICLR, 2022.

[28] W. Liang, L. Yu, L. Luo, et al., “Mixture-of-Transformers: A sparse and scalable architecture for multi-modal foundation models,” TMLR, 2025.

[29] W. Shi, X. Han, C. Zhou, et al., “LMFusion: Adapting pretrained language models for multimodal generation,” in NeurIPS, 2025.

[30] M. Dehghani, J. Djolonga, B. Mustafa, et al., “Scaling vision transformers to 22 billion parameters,” in ICML, 2023.

[31] OpenAI, “Introducing GPT-5.4 mini and nano,” https://openai.com/index/ introducing-gpt-5-4-mini-and-nano/, 2026, Accessed: 2026-09-13.

[32] G. Wijngaard, H. Gamper, and S. Braun, “Sci-Phi-Chat: Training a spatial audio agent for question answering, dialogue, and tool use,” Submitted to ICASSP 2027.

[33] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv:2207.12598, 2022.

[34] M. Yasuda, Y. Ohishi, and S. Saito, “Echo-aware adaptation of sound event localization and detection in unknown environments,” in ICASSP, 2022.

[35] T. Hirvonen and M. Namazi, “Compression of higher order ambisonics with multichannel RVQGAN,” arXiv:2411.12008, 2024.

[36] J. Hu, Y. Cao, M. Wu, et al., “PSELDNets: Pre-trained neural networks on a large-scale synthetic dataset for sound event localization and detection,” IEEE TASLP, 2025.

[37] P. Kamath, A. S. Roman, K. Saito, et al., “Sensitivity analysis of generative spatial audio metrics: A study on responsiveness, smoothness, and symmetry,” arXiv:2606.11581, 2026.

[38] Y. Wu, C. Tsirigotis, K. Chen, et al., “FLAM: Frame-wise language-audio modeling,” in ICML, 2025.

[39] B. Elizalde, S. Deshmukh, and H. Wang, “Natural language supervision for general-purpose audio representations,” in ICASSP, 2024.

[40] A. Gui, H. Gamper, S. Braun, et al., “Adapting Frechet audio´ distance for generative music evaluation,” in ICASSP, 2024.