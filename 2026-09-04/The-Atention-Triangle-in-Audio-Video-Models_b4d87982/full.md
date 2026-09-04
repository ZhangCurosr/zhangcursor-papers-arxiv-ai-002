# The Atention Triangle in Audio-Video Models

Sagi Polaczek<sup>1</sup> Noa Kraicer<sup>1</sup> Gal Metzer<sup>1</sup> Zhuo Ning<sup>2</sup> Ali Mahdavi-Amiri<sup>2</sup> Daniel Cohen-Or<sup>1</sup> Raja Giryes<sup>1</sup>

<sup>1</sup>Tel Aviv University <sup>2</sup>Simon Fraser University

![](images/7d1f20b17797a2f96b9b156ea5371296fe174bffdb03bf12af21b34fd8a516fd.jpg)  
"A weathered pirate with a black tricorn hat and thick grey beard stands on a wooden dock at sunset. A bright green-and-red macaw parrot perches on his right shoulder. The parrot is the <sub>speaks:</sub> <sub>We</sub> <sub>are</sub> <sub>lost</sub> <sub>at</sub> <sub>sea. The</sub> <sub>pirate</sub> <sub>does</sub> <sub>not</sub> <sub>move</sub> <sub>or</sub> <sub>speak.“ ”</sub>only character that speaks: ‘We are lost at sea.’ The pirate does not move or speak.”

Fig. 1. The Atention Triangle in Audio-Video Models. Top-left: baseline T2AV generation exhibits strong leakage; the pirate visually inherits parrot-like atributes and becomes the speaking source. Top-right: Ours-Text restores the pirate’s appearance, but speech remains incorrectly localized to the pirate. Botom-left: Ours-AV correctly localizes speech to the parrot, but appearance leakage persists in the pirate. Botom-right: Ours-Full jointly restores correct appearance and localizes speech to the intended source. The prompt is shown below the figure.

Audio-video difusion models rely on cross-modal attention to coordinate text, sound, and visual content, yet this same mechanism can introduce subtle and systematic semantic leakage. We study these models by probing and analyzing the “attention triangle,” comprising the three cross-attention edges connecting the text, audio, and video streams, and examine how semantic information is routed across modalities during generation. Our analysis reveals that routing along the audio-video edge is bidirectional: audio can influence video generation, while video can influence audio generation. This edge is shaped by biases encoded in the model’s parameters and emerges as a major contributor to leakage: when prompts are in tension with learned priors, cross-modal interactions may override the intended conditioning and reroute semantics toward visually canonical but incorrect outcomes. These efects suggest that semantic artifacts arise not merely from attention spreading beyond its intended target, but from structured, bias-driven interactions along specific pathways. Building on this perspective, we extract attention-derived signals that expose how semantics are distributed and grounded across modalities, and use them as a diagnostic tool to both

analyze and deliberately incur leakage under controlled conditions. This enables us to probe the internal dynamics of cross-modal routing and isolate the role of individual interactions. We further leverage these signals to guide inference-time interventions that encourage more consistent cross-modal alignment. Extensive experiments support our analysis and demonstrate improved semantic grounding while preserving generation quality.

## 1 Introduction

Recent advances in difusion models, particularly with Difusion Transformers (DiT) [Peebles and Xie 2023], have established attention as a central mechanism for conditional generation. By leveraging cross-attention within transformer architectures, these models can associate conditioning signals such as text, reference images, or other modalities with the evolving visual representation. Attention captures and propagates semantic associations across interacting representations throughout denoising, enabling expressive and compositional generation. The same mechanism, however, mixes information softly and globally without explicit locality or exclusivity constraints. Semantic associations can therefore spread beyond their intended targets, producing attribute leakage, semantic entanglement, and unstable bindings over time.

These limitations become more pronounced in audio-video generation, where attention must jointly mediate interactions between text, audio, and video modalities (Figs. 1 and 3). We refer to this trimodal setting as an attention triangle (Fig. 2), in which each modality both influences and constrains the others. Unlike bimodal conditioning, alignment is no longer pairwise but coupled across all three modalities. The model must reconcile semantic intent from text, temporal structure from audio, and spatiotemporal realization in video. These coupled constraints can produce competing signals, modality dominance, and entangled associations, allowing a competing interpretation to propagate through attention and reduce global consistency. Our ablations identify the audio-video pathway as a major contributor to semantic leakage in LTX-2 (Fig. 6 and Supp. Fig. 7). We hypothesize that this vulnerability arises partly because audio tokens encode temporal (but not spatial) position, leaving source localization underconstrained.

In this work, we revisit audio-video generation through an analysisdriven lens. We analyze the leakage problem, as introduced above, through the structure of the attention triangle, while placing particular emphasis on failures of source attribution, namely whether sound semantics are grounded in the appropriate visual entity. Viewing the model through this lens, leakage can be understood in terms of how semantic information is routed across the text, audio, and video streams. More generally, this routing is inherently bidirectional: audio can influence the visual interpretation, while visual priors can in turn bias the generation and attribution of sound.

An interesting observation that emerges from our analysis is that the audio-video edge of this triangle often acts as a weak link (Fig. 5 and Supp. Fig. 11). In cases where the prompt is in tension with the model’s learned cross-modal biases, this interaction may override the binding induced by the text, rerouting sound semantics toward visually canonical but incorrect sources. This suggests that leakage is not only a consequence of semantic associations extending to unintended entities, but is closely tied to bias-driven interactions encoded in the model’s parameters, which manifest along specific cross-modal pathways.

Building on this perspective, we extract attention-derived signals that expose source attribution, and use them as a diagnostic tool to both analyze and deliberately incur leakage. By inducing controlled failures, we probe how sound semantics are routed across entities and modalities, and isolate the role of specific cross-modal interactions. We then use these signals to guide inference-time interventions toward more consistent sound-to-source grounding.

A representative example is shown in Fig. 1. The baseline exhibits both appearance leakage and incorrect source attribution, with the pirate inheriting parrot-like attributes and becoming the speaking source. Partial interventions reveal a decoupling: steering the text edges restores appearance but not attribution, while steering the audio-video edge corrects attribution but not appearance. Only joint steering of all triangle edges resolves both, illustrating that no single interaction sufices to control cross-modal routing.

Across diverse prompts and settings, our experiments reveal consistent leakage patterns and show that the framework supports both probing and mitigation (Tab. 1 and Supp. Fig. 9).

## 2 Background and Related Work

Difusion models rely on two complementary attention mechanisms. Self-attention operates within a single modality, allowing each token to aggregate long-range context from all others [Vaswani et al. 2017]. Cross-attention couples two token sequences, typically the noisy latent and a text embedding, so latent tokens selectively attend to relevant conditioning tokens [Rombach et al. 2022]. In text-to-image DiT [Peebles and Xie 2023] and UNet backbones, cross-attention serves as the primary mechanism for grounding prompt semantics spatially in the image.

Our work builds on three closely related lines of research: the use of cross-attention as a control surface for image generation, its failure mode of leakage between distinct entities, and audio-video difusion models that inherit and compound this problem along a new modality axis.

## 2.1 Atention Manipulation and Leakage in Image Difusion Models

Cross-attention has become the primary control surface of text-toimage difusion: it encodes spatial layout [Hertz et al. 2023] and carries attribute binding through its keys and values [Feng et al. 2023]. Inference-time methods steer it to boost under-attended tokens and recover missing or under-represented subjects [Chefer et al. 2023], align attention maps with syntactic structure to correct attribute correspondence [Rassin et al. 2023], separate competing entities while binding attributes [Li et al. 2023], or enforce userspecified layouts [Chen et al. 2024]. Even in modern RoPE-based MMDiT backbones [Esser et al. 2024], diferent transformer layers play qualitatively distinct roles that invite layer-targeted edits [Wei et al. 2025]. Together these works treat cross-attention as a steerable, semantically meaningful signal yielding fine-grained control without retraining.

The flip side is leakage: because attention is a soft, global mixing process without explicit locality or exclusivity constraints, features routinely bleed between entities the prompt intends to keep distinct. A prompt such as “a striped cat next to a spotted dog” frequently produces a striped dog or a spotted cat, because patterns are routed by the semantics of the target entity rather than the target description. This failure has been attributed to attention-layer feature blending across visually similar subjects [Dahary et al. 2024], conflict between externally imposed layouts and the noise prior [Dahary et al. 2025], entangled End-of-Sequence text embeddings that aggregate attributes across the prompt [Mun et al. 2025], and mis-allocated attention logits across entities [Ventura et al. 2026]; the corresponding remedies all operate as inference-time interventions, treating leakage as an attention-routing failure. A related failure mode is contextual contradiction, where one concept implicitly suppresses another through entangled learned associations; Huberman et al. [2025] address this via stage-aware prompting that decomposes the prompt across denoising stages rather than editing attention directly. Our work extends this perspective beyond text-to-image generation to the attention triangle, where leakage occurs not only between visual entities but also across modalities.

## 2.2 Video and Audio-Video Difusion Models

Early text-to-video systems such as Make-A-Video [Singer et al. 2023] and Imagen Video [Ho et al. 2022a] attached spatiotemporal modules [Blattmann et al. 2023; Ho et al. 2022b] to pretrained text-to-image models, generating silent clips with audio outside the generative loop. The shift to Difusion Transformer (DiT) backbones [Brooks et al. 2024] enabled large-scale joint video training behind open systems including CogVideoX [Yang et al. 2024], HunyuanVideo [Kong et al. 2024], Wan [Wan et al. 2025], and LTX-Video [HaCohen et al. 2024], alongside closed-source Sora [Brooks et al. 2024] and Veo [Google DeepMind 2025]. These systems retain the silent-video assumption.

A more recent thread closes the audio gap by jointly generating audio and video in a single difusion framework. MM-Difusion [Ruan et al. 2023] pioneered this by pairing video and audio sub-nets through a random-shift cross-attention block exchanging timealigned information; MM-LDM [Sun et al. 2024] lifted the idea into a shared semantic latent space. Transformer backbones then produced dual-branch designs coupling two parallel DiT towers via cross-modal attention: SyncFlow [Liu et al. 2024] fuses a dual difusion-transformer for temporally aligned text-to-audio-video; AV-Link [Haji-Ali et al. 2025] repurposes intermediate features of frozen audio and video difusion models as cross-modal conditioning; JavisDiT [Liu et al. 2025] introduces a hierarchical spatiotemporal prior aligning the two streams at multiple granularities; and Ovi [Low et al. 2025] adopts a fully symmetric twin-DiT design with paired cross-attention layers. On the closed side, Veo 3 [Google DeepMind 2025] performs joint denoising over a unified token sequence covering both modalities. The open foundation model LTX-2 [HaCohen et al. 2026] is representative of the state of the art.

Recent work more directly examines or structures cross-modal interaction in dual-stream generators. UniAVGen introduces modalityspecific, temporally aligned interaction and learned face-aware modulation [Zhang et al. 2026], while Cross-Modal Context Learning (CCL) identifies background-region attention bias, optimization instability, and conflicts between text and cross-modal conditions, addressing them through temporal partitioning, learnable context tokens, and learned routing and guidance [Ma et al. 2026]. Very recent work uses bidirectional audio-video attention to guide training free sparsification and cache reuse while preserving synchronization [Gao et al. 2026]. These approaches redesign or accelerate cross-modal interaction; we instead target entity-level semantic leakage and source attribution in a pretrained generator through training-free interventions across all three triangle edges.

Complementary work evaluates spatial audio-video alignment using visual-object and stereo sound-direction estimates [Shimada et al. 2026], or introduces trained reference-conditioned branches for multimodal control [Li et al. 2026]; our method instead targets prompt-specified source binding without auxiliary training or reference inputs.

![](images/bf67fc774bb206243ce86bdac6cbd6f6cfda53e3fe8441a4a5373439651fece4.jpg)  
Fig. 2. The Atention Triangle. Edge-specific biases promote atention to the intended speaker (green) and suppress leakage to the competing entity (red). Intended denotes amplified atention for prompt-designated pairings: source↔source, sound↔sound, or source↔sound. Conflicting denotes attenuated pairings between these signals and the competing source. The agreement matrix $\mathbf { G } _ { V A }$ is defined in Eq. 2; the intended and conflicting text-edge masks M�� and $\mathbf { M } _ { A T }$ are defined in Eqs. 4 and 5, respectively.

While these architectures make joint audio-video generation tractable, their cross-modal attention inherits and arguably amplifies the same leakage and source-attribution failures seen in the image domain. Unlike image leakage between spatially colocated features, the audio-video case introduces a fundamental dimensional mismatch: a 1D audio temporal embedding must attend to a 3D spatiotemporal positional embedding [HaCohen et al. 2026]. Forcing video patches to collapse onto the 1D time axis strips intra-frame spatial distinctness, creating a degeneracy that we diagnose and address in the following sections.

## 3 Identifying the Triangle Problem

We consider joint text-to-video and text-to-audio generation with modern video difusion models that synthesize a video clip and its accompanying soundtrack from a single textual prompt. Given a textual prompt, the models simultaneously condition both a video stream of frames and a temporally aligned audio stream, and the two modalities are coupled through cross-attention so that what is heard reinforces what is seen and vice versa. This mutual dependence suggests a useful abstraction: rather than viewing the diferent conditioning pathways independently, we regard them as a coupled trimodal system in which semantic information can be routed both directly and indirectly between text, audio, and video.

We formalize this system as an attention triangle comprising three cross-attention edges (Fig. 2) and associate each surface with a pre-softmax bias matrix. Bias arrows denote conditioning flow: $\mathbf { B } _ { X  Y }$ acts on the �-query/�-key surface. Thus, $\mathbf { B } _ { T  V }$ and $\mathbf { B } _ { T  A }$ bias the text-conditioned video and audio surfaces (Eqs. 4 and 5), while $\mathbf { G } _ { V A }$ denotes the audio-video agreement matrix and $\mathbf { B } _ { A  V }$ denotes the corresponding pair of audio-video biases (Eqs. 2 and 3). While this joint formulation is appealing, it makes the prompt highly susceptible to semantic leakage: textual attributes intended for one entity are silently rerouted to another, more frequent or more salient entity in the scene. For example, a prompt such as “a parrot sitting on the shoulder ofa pirate, the parrot is talking” reliably produces a speaking pirate rather than a talking parrot, because speech carries a strong learned prior toward human-like figures, causing the model to route audio tokens to pirate patches rather than to the parrot the prompt explicitly designates as the speaker.

![](images/06437a421a36c2f26adce1f84c64499c230ddcbf4ab7d8323b288db68ce7acd5.jpg)  
Fig. 3. Cross-modal leakage in T2V vs T2AV generation. Left: T2V. Middle: T2AV. Right: prompt. Adding audio induces both appearance and speaking leakage: in the puppet example, both the performer and puppet move their lips; in the dog and cat example, the dog shifts toward a cat-like appearance and both animals appear to speak.

## 3.1 The Atention Triangle

The attention triangle connects three token populations through pairwise cross-attention: (1) text tokens, (2) audio tokens, and (3) video patches. Every pair ofmodalities exchanges information through cross-attention, so a single attribute embedded in the text can reach the video stream both directly (text → video) and indirectly through the audio stream (text → audio → video), and symmetrically in the opposite direction. This relationship is depicted in Figure 2. Composing these pairwise paths also exposes second-order, within-modality interactions; in particular, the video→audio→video rollout reveals an efective audio-mediated coupling between visual regions, formalized in Eq. 1 and visualized in Fig. 5. A particularly problematic edge of this triangle is the audio-video cross-attention itself. The model’s training distribution encodes strong priors directly in this edge: speech-like audio features are correlated with human-shaped patches, not parrots. Even when the text correctly assigns speech to the parrot, the audio-video edge can override that binding and drag the sound toward its statistically familiar source. Beyond these learned statistical biases, we hypothesize that the audio-video edge may also be underconstrained spatially. In LTX-2 [HaCohen et al. 2026], video patches and audio tokens share temporal positional information, while audio tokens do not carry explicit within-frame spatial coordinates. This asymmetry may force cross-modal correspondence to rely more heavily on learned semantic compatibility between audio content and visual entities than on explicit spatial alignment. It may therefore make it more dificult for cross-modal attention to distinguish between spatially distinct candidate sources and may contribute to weaker sound localization.

These considerations motivate us to test whether audio-video cross-attention is a major contributor to semantic leakage in LTX-2. We examine this question through attention visualizations, controlled attention interventions that induce or mitigate leakage, and complementary ablations. Unlike the direct text-conditioning paths, this edge relays semantic information between two generated modalities along an implicit pathway that is not directly specified by the user’s prompt.

## 3.2 Audio Leakage Analysis

Direction convention. We index attention matrices by query and key modality: $\mathbf { P } _ { V A }$ has video queries and audio keys. Under our key/value information-flow convention, this is the $A  V$ conditioning surface because audio values contribute to video-query outputs; under row rollout, right multiplication by $\mathbf { P } _ { V A }$ maps a videoindexed distribution to an audio-indexed distribution and therefore constitutes a $V  A$ rollout step.

Visualizing Audio-Mediated Attention. As a first-order probe, we inspect $\mathbf { P } _ { V A ; }$ , the video-query/audio-key attention matrix. For each video query, we sum its attention weights over the selected speechactive audio keys and average the resulting spatial score across heads and denoising steps. On leakage prompts, high scores concentrate spatially on the wrong subject; for example, video queries on the pirate assign more attention to speech-active audio keys than video queries on the parrot, even though the prompt assigns speech to the parrot. This misbinding is visualized in Fig. 4.

To make matrix orientation explicit, let $\mathbf { P } _ { Q K } ^ { \left( \ell \right) } \in \left[ 0 , 1 \right] ^ { N _ { Q } \times N _ { K } }$ denote the head-averaged attention matrix at layer $\ell ,$ with rows indexed by queries from modality � and columns by keys from modality �. Following the Markov-chain interpretation of attention matrices [Erel et al. 2025], we compose consecutive video-query/audiokey and audio-query/video-key matrices:

$$
\mathbf { P } _ { V V } ^ { \mathrm { e f f } } \ = \ \mathbf { P } _ { V A } ^ { ( \ell ) } \mathbf { P } _ { A V } ^ { ( \ell + 1 ) } \ \in \ [ 0 , 1 ] ^ { N _ { V } \times N _ { V } } .\tag{1}
$$

The efective matrix $\mathbf { P } _ { V V } ^ { \mathrm { e f f } }$ captures a video-to-video coupling mediated by the audio stream that does not appear in any single attention layer. Seeding a uniform row distribution over each entity’s visual mask and propagating it through $\mathbf { P } _ { V V } ^ { \mathrm { e f f } }$ (see Appendix G.1), we observe that on leakage prompts, the mass migrates from the intended source to the visually canonical one (e.g. from the parrot patches to the pirate), making the audio-mediated mis-binding visible as a concrete spatial transport pattern.

Removing the Audio. A complementary way to test the same hypothesis is to ablate the audio branch and check whether the leakage persists. LTX-2 ships in audio-free and joint variants sharing the same backbone (see Appendix G.2), making this ablation straightforward. In the paired examples shown in Fig. 3, failures present under joint T2AV generation are absent from the audio-free T2V outputs despite the shared visual backbone. Together with the text-edgeonly intervention in Sec. 5.3, which leaves source misattribution unresolved, this provides complementary evidence that the audiovideo pathway is a major contributor to leakage in LTX-2. These interventions do not, however, establish temporal-only positional encoding as the pathway’s unique failure mechanism.

![](images/c52becefcc1d04f5d91cd7d67a99c4c1e84cd5f3df15d6817bead307a534861f.jpg)  
Fig. 4. Mitigating audio-to-video atention leakage. Using the same prompt and random seed as in Figure 1, we sum, for each video query, its atention weights over the speech-active audio keys identified from the waveform shown at the botom, then overlay the resulting spatial score on a representative frame. Left (baseline): high active-audio scores concen trate across the pirate’s face and chest, the visually canonical but promptincorrect human source. Right (ours): after atention-triangle steering, high scores concentrate sharply on the parrot, the entity designated by the prompt. The right panel is desaturated outside the high-scoring region to highlight the post-steering localization.

## 4 Steering: Grounding Audio and Video Regions

Having identified audio-video cross-attention as a major contributor to semantic leakage, we introduce a training-free, inference-time steering algorithm, instantiated on LTX-2 [HaCohen et al. 2026], that jointly operates on all three edges of the attention triangle. We identify the intended sound source, the sound or action, and an optional competing source in the prompt. On the two text-conditioned surfaces, we define intended and conflicting cell masks; all remaining cells are neutral. In the pirate-parrot example (Figure 1), intended cells pair parrot video patches with “parrot” text tokens and speechactive audio tokens with the speaking-action text. Conflicting cells pair these video or audio regions with competing-source text, or pair competing-source video or audio regions with the intended source or action text. On the audio-video surfaces, we instead compute the soft agreement matrix $\mathbf { G } _ { V A }$ . Intended-source/sound and nonsource/non-sound pairs have high agreement, whereas mismatched source-sound pairs have low agreement; the bias $- \lambda ( 1 - \mathbf { G } _ { V A } )$ penalizes each pair in proportion to its disagreement. During denoising, we apply a pre-softmax additive logit bias to all three crossattention surfaces: audio↔video, text→video, and text→audio. The text-conditioned biases reinforce intended cells and suppress conflicting cells, while the audio-video bias suppresses mismatched source-sound pairs. Together, these biases reground both the audio and video streams to their intended prompt semantics. We denote the text-edge-only, audio-video-only, and joint variants as Ours-Text, Ours-AV, and Ours-Full, respectively.

Anchors. The biases are built from three families of anchors, all derived once from the unsteered baseline pass and held fixed across all denoising steps of the steered pass.

Prompt-side text anchors. For text, audio, and video modalities � $\in \{ T , A , V \}$ , let $N _ { M }$ denote the number of tokens. The user annotates the intended sound-source phrase, the sound-or-action phrase, and an optional competing-source phrase $( \mathrm { e . g . }$ , “pirate” in Fig. 1). We remove the annotation tags before text encoding and map the annotated phrases to their corresponding token-index sets $\begin{array} { r } { \mathcal { I } _ { \mathrm { s r c } } ^ { T } , \mathcal { I } _ { \mathrm { s n d } } ^ { T } , } \end{array}$ and $\mathcal { I } _ { \mathrm { c m p } } ^ { T }$ in the cleaned prompt. Their indicator vectors are denoted $\mathbf { m } _ { \mathrm { s r c } } ^ { T } , \mathbf { m } _ { \mathrm { s n d } } ^ { \dot { T } } , \mathbf { m } _ { \mathrm { c m p } } ^ { T } \in \{ 0 , 1 \} ^ { N _ { T } }$

Visual anchors. For both the audio↔video and text-conditioned video surfaces, we decode the baseline frames, prompt SAM3 [Carion et al. 2025] with the intended-source text, binarize at 0.5, and resize to the latent grid. This yields the hard source mask $\mathbf { m } _ { \mathrm { s r c } } ^ { V } \in \{ 0 , 1 \} ^ { N _ { V } }$ ; the competing-source mask $\mathbf { m } _ { \mathrm { c m p } } ^ { V }$ is obtained analogously.

Audio anchors. No external audio segmenter is available, so we aggregate audio-query/text-key attention targeting $\mathcal { I } _ { \mathrm { s n d } } ^ { T }$ across denoising steps and normalize it to [0, 1], yielding the soft sound mask $\mathbf { m } _ { \mathrm { s n d } } ^ { A } \in [ 0 , 1 ] ^ { N _ { A } }$ . For the text-conditioned audio surface, thresholding at $\theta _ { A } = 0 . 3$ gives $\bar { \mathbf { m } } _ { \mathrm { s n d } } ^ { A } \in \{ 0 , 1 \} ^ { N _ { A } }$ . When a distinct audio region associated with the competing source is available, $\bar { \mathbf { m } } _ { \mathrm { c m p } } ^ { A }$ is obtained analogously; otherwise, as in examples with a silent competing source, it is the zero vector.

This asymmetric anchor design, with hard external masks where a reliable segmenter exists and soft model-derived saliency where it does not, is detailed in Appendix H.1.

Audio↔video edge. Let $1 _ { N }$ and $1 _ { N _ { Q } \times N _ { K } }$ denote an all-ones vector and matrix of the indicated sizes. We define the audio↔video agreement matrix as

$$
\begin{array} { r l } & { \mathbf { G } _ { V A } = \mathbf { m } _ { \mathrm { s r c } } ^ { V } ( \mathbf { m } _ { \mathrm { s n d } } ^ { A } ) ^ { \top } } \\ & { \qquad + \left( \mathbf { 1 } _ { N _ { V } } - \mathbf { m } _ { \mathrm { s r c } } ^ { V } \right) \left( \mathbf { 1 } _ { N _ { A } } - \mathbf { m } _ { \mathrm { s n d } } ^ { A } \right) ^ { \top } \in [ 0 , 1 ] ^ { N _ { V } \times N _ { A } } . } \end{array}\tag{2}
$$

The agreement matrix in $\operatorname { E q } .$ 2 assigns high values to intendedsource/sound and non-source/non-sound pairs, and low values to mismatched source-sound pairs. We suppress the mismatched audio↔video logits using

$$
{ \bf B } _ { A  V } = - \lambda ( \mathbf { 1 } _ { N _ { V } \times N _ { A } } - \mathbf { G } _ { V A } ) , \qquad { \bf B } _ { V  A } = { \bf B } _ { A  V } ^ { \top } , \quad \lambda > 0 .\tag{3}
$$

We use $\mathbf { B } _ { A  V }$ as shorthand for the pair $( \mathbf { B } _ { A  V } , \mathbf { B } _ { V  A } )$ . We add $\mathbf { B } _ { A  V }$ to the video-query/audio-key logits and $\mathbf { B } _ { V  A }$ to the audioquery/video-key logits at every transformer block and denoising step. This soft two-class construction has one hyperparameter; we use �=10 throughout (Appendix H.2).

Text→video and text→audio edges. A single audio↔video correction is insuficient: on the video-query/text-key surface, queries in the intended-source and competing-source regions can still attend to the other entity’s text tokens, producing appearance leakage. Both text-conditioned surfaces use the same three classes: intended, conflicting, and neutral. We denote the corresponding cell masks by $\mathbf { M } ^ { \mathrm { i n i } }$ and $\mathbf { M } ^ { \mathrm { c o n f } }$ , respectively.

For the video-query/text-key surface,

$$
\begin{array} { r l } & { \mathbf { M } _ { V T } ^ { \mathrm { i n t } } \ = \ \mathbf { m } _ { s \mathrm { r c } } ^ { V } ( \mathbf { m } _ { s \mathrm { r c } } ^ { T } ) ^ { \top } , } \\ & { \mathbf { M } _ { V T } ^ { \mathrm { c o n f } } \ = \ \mathbf { m } _ { s \mathrm { r c } } ^ { V } ( \mathbf { m } _ { \mathrm { c m p } } ^ { T } ) ^ { \top } + \mathbf { m } _ { \mathrm { c m p } } ^ { V } ( \mathbf { m } _ { s \mathrm { r c } } ^ { T } ) ^ { \top } , } \\ & { \mathbf { B } _ { T  V } \ = \ \beta \mathbf { M } _ { V T } ^ { \mathrm { i n t } } - \gamma \mathbf { M } _ { V T } ^ { \mathrm { c o n f } } \in \mathbb { R } ^ { N _ { V } \times N _ { T } } . } \end{array}\tag{4}
$$

For the audio-query/text-key surface,

$$
\begin{array} { r l } & { \mathbf { M } _ { A T } ^ { \mathrm { i n t } } \ = \ \bar { \mathbf { m } } _ { \mathrm { s n d } } ^ { A } ( \mathbf { m } _ { \mathrm { s n d } } ^ { T } ) ^ { \top } , } \\ & { \mathbf { M } _ { A T } ^ { \mathrm { c o n f } } \ = \ \bar { \mathbf { m } } _ { \mathrm { s n d } } ^ { A } ( \mathbf { m } _ { \mathrm { c m p } } ^ { T } ) ^ { \top } + \bar { \mathbf { m } } _ { \mathrm { c m p } } ^ { A } ( \mathbf { m } _ { \mathrm { s n d } } ^ { T } ) ^ { \top } , } \\ & { \mathbf { B } _ { T  A } \ = \ \beta \mathbf { M } _ { A T } ^ { \mathrm { i n t } } - \gamma \mathbf { M } _ { A T } ^ { \mathrm { c o n f } } \ \in \mathbb { R } ^ { N _ { A } \times N _ { T } } . } \end{array}\tag{5}
$$

We use $\beta { = } 0 . 5$ and $\gamma { = } 2 . 0 .$ Together with $\lambda { = } 1 0$ and $\theta _ { A } = 0 . 3$ above, these values are fixed across all LTX-2 experiments and are not tuned per prompt (Appendix H.2). The asymmetric choice places greater weight on suppressing conflicting cells than on reinforcing intended cells. When no competing-source indices are supplied, $\mathbf { M } ^ { \mathrm { c o n f } }$ vanishes and the bias reduces to an intended-cell boost.

We validate this design in Section 5, where ablations confirm that steering all three edges jointly is necessary: partial interventions that address only a subset of edges leave residual leakage (see Fig. 6).

The three biases together close the triangle; their efect on attention weights is depicted in Figure 2. Ours-Full is training-free and requires no auxiliary loss or fine-tuning. Its overhead is one additional full denoising trajectory over all steps required to generate the final video to extract the baseline anchors, plus frame decoding and SAM3 for the visual anchors, giving roughly 2.5× the cost of unsteered generation (Supp. H.3).

## 5 Experiments

We validate our analysis and steering framework through attention visualizations, qualitative and quantitative comparisons, and human preference studies. Full protocols, additional examples, and permetric breakdowns are provided in the Supplementary Material; we strongly encourage readers to inspect the side-by-side videos in the accompanying supplementary HTML page (Supp. E), because the leakage phenomena require synchronized audio and video.

## 5.1 Atention-Leakage Analysis

First-Order Attention Visualization. As analyzed in Sec. 3, the cross-attention between audio and video tokens carries a direct signature of leakage. Figure 4 visualizes the spatial active-audio attention score derived from the video-query/audio-key matrix $\mathbf { P } _ { V A }$ In the baseline, high scores concentrate at video-query locations on the pirate, the visually canonical but prompt-incorrect entity. Under steering, high scores concentrate on the parrot, consistent with improved source attribution.

Second-OrderAttention Visualization. Figure 5 exposes the secondorder, audio-mediated video-to-video coupling described in Sec. 3.2. We seed a uniform attention distribution over the parrot’s spatial region and propagate it through $\mathbf { P } _ { V V } ^ { \mathrm { e f f } }$ (Eq. 1). Before steering, the mass migrates from the parrot patches to the pirate after the round trip through the audio stream, directly revealing the audio-mediated misrouting that drives the leakage. After steering, the mass returns to the parrot, confirming that the intervention successfully breaks this indirect pathway. Supplementary Fig. 11 corroborates this routing pattern in two additional LTX-2 examples and quantifies how Ours-Full redirects the returned mass toward the intended source while suppressing the competing source.

![](images/8e1a502648334e16a2ca2cc6267a0fe106c161b9b892ba6f3ede1502565305d5.jpg)  
Fig. 5. Audio-mediated video-to-video leakage. Using the same prompt as in Figure 1, we apply the $V  A  V$ row rollout in Eq. 1, using $\mathbf { P } _ { V A } ^ { ( \ell ) }$ for the � → � rollout hop followed by $\mathbf { P } _ { A V } ^ { ( \ell + 1 ) }$ for the $A \to V$ rollout hop. This exposes an indirect atention association between visual regions that is not represented by either individual cross-atention matrix alone. Left: seed mass on the parrot, the prompt-designated source. Middle: the resulting audio-token distribution after the �→� hop is difuse rather than peaked, reflecting the 1D-to-2D dimensional mismatch (Sec. 3). Right: after the �→� hop, mass has migrated onto the pirate, exposing the audio-mediated misrouting behind source-atribution leakage.

## 5.2 Manipulation in the Wild

We apply our steering framework to a broad set of open-domain audio-video scenarios spanning realistic humans, animals, animated characters, and multi-entity scenes with diverse spatial layouts. Across these settings, we identify four recurring leakage modes: source-attribution leakage, where sound-producing behavior is assigned to the wrong entity; appearance leakage, where sound semantics alter the visual rendering of an unrelated entity (e.g., a pirate acquiring parrot-like features); voice-characteristic leakage, where the generated voice follows canonical priors of the visual entity rather than the prompt specification; and generation suppression, where strong conflicts between the prompt and learned priors cause the model to omit the requested sound entirely. Ours-Full mitigates all four modes by jointly steering the full attention triangle, preserving both visual appearance and correct sound grounding across all tested scenarios (detailed in Supp. A).

## 5.3 Comparisons and Ablations

We compare against the native LTX-2 T2AV baseline, an adaptation of Bounded Attention [Dahary et al. 2024] to the audio-video setting, and two partial variants: Ours-Text steers only the text edges (Text→Audio, Text→Video) whereas Ours-AV steers only the audio-video edge (Audio↔Video). The partial variants isolate the contribution of each edge family and serve as controlled ablations. A representative qualitative example is shown in Fig. 6: Ours-Text corrects the textual binding but cannot overcome the audio-video prior, leaving the sound misattributed; Ours-AV localizes the source correctly but introduces appearance leakage into the competing source; only Ours-Full simultaneously preserves appearance and grounds the sound to the intended source. Additional qualitative examples and per-method pairwise user-study results are reported in Supp. B and Supp. D. Representative comparisons with Ovi are provided in Supp. H.4.

![](images/904abcc91a6e6dbfe3fda6e776e6f836f0a10ccd81d23b8cadd5a362146f8d00.jpg)  
A horse stands beside a shiny armored knight, both facing forward. The horse stays silent with its mouth“ closed. The knight is the only sound source, producing a bright horse neigh, airy and proud with stable-like resonance. No human command is spoken.”

Fig. 6. Horse-neigh atribution. The prompt asks for a horse neigh that should be produced by the knight rather than the horse. Three representative frames are shown per method. Across baselines and partial-edge variants the neigh is misatributed to the horse or the knight acquires horse-like visua features; Ours-Full both localizes the sound to the knight and preserves his appearance. Zoomed regions indicate the active sound source.  
Table 1. Main leakage comparison. We compare the native audio-video generator, a video-only diagnostic baseline, the external Ovi baseline, an adapted text/image-side Bounded Atention baseline, and three variants of our atention-triangle intervention.
<table><tr><td>Metric</td><td>Native LTX-2</td><td>Video-only</td><td>Ovi</td><td>Bounded Attn.</td><td>Ours-Text</td><td>Ours-AV</td><td>Ours-Full</td></tr><tr><td>Qwen source-attribution score ↑</td><td>0.1216</td><td>N/A</td><td>0.1207</td><td>0.0925</td><td>0.1140</td><td>0.1302</td><td>0.1349</td></tr><tr><td>CLAP audio-text ↑</td><td>0.375 ± 0.187</td><td>N/A</td><td>0.357 ± 0.187</td><td>0.384 ± 0.198</td><td>0.374 ± 0.188</td><td>0.384 ± 0.189</td><td>0.383 ± 0.185</td></tr><tr><td>VBench subject consistency ↑</td><td>0.988</td><td>0.987</td><td>0.971</td><td>0.981</td><td>0.989</td><td>0.989</td><td>0.990</td></tr><tr><td>VBench background consistency ↑</td><td>0.981</td><td>0.983</td><td>0.975</td><td>0.981</td><td>0.984</td><td>0.983</td><td>0.986</td></tr><tr><td>VBench aesthetic quality ↑</td><td>0.590</td><td>0.597</td><td>0.571</td><td>0.588</td><td>0.598</td><td>0.589</td><td>0.604</td></tr></table>

Metric definitions. Qwen source-attribution score is the conditional probability that only the intended visible entity produced the audible target event, computed from a counterbalanced five-option Qwen3-Omni judge; higher is better. The video-only diagnostic has no audio and is therefore not scored. CLAP audio-text measures the alignment between the generated audio and the intended sound phrase; we report mean ± standard deviation. VBench subscores measure visual subject consistency, background consistency, and aesthetic quality. All metrics are computed on the same prompt set across methods.

Quantitative evaluation. We use two complementary automatic audio-video validators. A counterbalanced five-option Qwen3-Omni judge [Xu et al. 2025] targets sound-source attribution, while VA-Judger [Huang et al. 2026] compares matched video pairs using prompt fidelity, audiovisual consistency, audio quality, video quality, and completeness. We additionally measure audio-text alignment with CLAP [Wu et al. 2023] and visual fidelity with VBench [Huang et al. 2024]; the human study below provides an independent pairwise assessment of attribution, appearance leakage (shown to annotators as “visual leakage”), and overall quality. The higher-is-better Qwen source-attribution score is 1 minus conditional leakage. Ours-Full obtains the highest mean score (0.1349), compared with 0.1216 for the native LTX-2 baseline; the exact judge prompt, logit extraction, counterbalancing, and aggregation are given in Supp. C.1. On the broader VA-Judger evaluation, Ours-Full is preferred over every alternative, with mean preference scores ranging from 57.6% to 73.3%; all pair-bootstrap 95% confidence intervals lie above 50%.

The complete per-reference results and protocol are reported in Supp. C.2. Ours-Full attains the best VBench scores across subject consistency (0.990), background consistency (0.986), and aesthetic quality (0.604), while remaining competitive on CLAP (0.383 vs. 0.384 for the closest baseline), suggesting that the intervention does not cause a large degradation in the measured fidelity metrics. Tab. 1 reports the full comparison; protocol details are provided in Supp. C.

Userstudy. We conduct pairwise human preference studies against each comparison method across three axes: (i) sound-source attribution, (ii) visual leakage (appearance leakage), and (iii) overall generation quality. Among decided preferences, annotators choose Ours-Full in 79.2% of attribution comparisons, 82.9% of leakage comparisons, and 80.1% of overall-quality comparisons, demonstrating a strong preference for ours on all three axes. The largest margins appear on attribution and leakage, the two axes most directly targeted by our intervention. A two-sided exact binomial test on non-tie votes confirms that all three aggregate preferences are statistically significant after Bonferroni correction $( p \ < \ 1 0 ^ { - 2 8 } )$ . Per-method breakdowns are reported in Supp. D (Fig. 9).

Ablations. The Ours-Text and Ours-AV columns ofTab. 1, together with the partial-steering panels of Fig. 6, ablate the contribution of each edge family of the attention triangle: Ours-Text corrects appearance but leaves source attribution unresolved, Ours-AV localizes the source but reintroduces appearance leakage, and only Ours-Full addresses both. The fixed hyperparameter choices $( \lambda , \beta , \gamma , \theta _ { A } )$ and asymmetric intended- vs. conflicting-cell weighting are described in Supp. H.2.

## 6 Discussion

Our analysis of LTX-2 identifies the audio-video pathway as a major contributor to bias-driven misalignment and shows that controlled attention steering can expose and mitigate these failures. More broadly, additional cross-modal coupling can create new routes through which learned priors override prompt-specified bindings. We call this the Attention Lesson: strengthening cross-modal coupling through attention can amplify bias-driven routing rather than resolve it.

Our evidence establishes the contribution of the audio-video pathway, not a unique underlying mechanism. We hypothesize that its vulnerability partly reflects the mismatch between spatial video tokens and temporally organized audio tokens, which may leave sound-source grounding underconstrained. Future work should directly test this mechanism, explore spatial or pseudo-spatial audio representations and stronger grounding supervision, and determine whether the same leakage patterns and interventions transfer to single-stream architectures that process all modalities within shared self-attention rather than through explicit cross-attention between modality-specific streams. We discuss limitations in Supp. F.

## Acknowledgments

This research was supported in part by the Israel Science Foundation (grants no. 2492/20 and 1473/24), Len Blavatnik and the Blavatnik family foundation.

## References

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. 2023. Align Your Latents: High-Resolution Video Synthesis with Latent Difusion Models. In Proceedings ofthe IEEE/CVFConference on Computer Vision and Pattern Recognition. IEEE, Piscataway, NJ, USA, 22563–22575.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. 2024. Video generation models as world simulators. https://openai.com research/video-generation-models-as-world-simulator

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Efrosyni Mavroudi, Kather ine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichten hofer. 2025. SAM 3: Segment Anything with Concepts. arXiv:2511.16719 [cs.CV] https://arxiv.org/abs/2511.16719

Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. 2023. Attend and-excite: Attention-based semantic guidance for text-to-image difusion models. ACM transactions on Graphics (TOG) 42, 4 (2023), 1–10.

Minghao Chen, Iro Laina, and Andrea Vedaldi. 2024. Training-free layout contro with cross-attention guidance. In Proceedings ofthe IEEE/CVF winter conference on applications ofcomputer vision. IEEE, Piscataway, NJ, USA, 5343–5353.

Zijun Cui, Xiulong Liu, Hao Fang, Mingwei Xu, Jiageng Liu, Zexin Xu, Weiguo Pian, Shijian Deng, Feiyu Du, Chenming Ge, and Yapeng Tian. 2026. Do Joint Audio-Video Generation Models Understand Physics? arXiv:2605.07061 [cs.CV] https: //arxiv.org/abs/2605.07061

Omer Dahary, Yehonathan Cohen, Or Patashnik, Kfir Aberman, and Daniel Cohen-Or. 2025. Be decisive: Noise-induced layouts for multi-subject generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. Association for Computing Machinery, New York, NY, USA, 1–12. doi:10.1145/3721238.3730631

Omer Dahary, Or Patashnik, Kfir Aberman, and Daniel Cohen-Or. 2024. Be yourself: Bounded attention for multi-subject text-to-image generation. In Computer Vision – ECCV 2024 (Lecture Notes in Computer Science, Vol. 15072). Springer, Cham, Switzerland, 432–448. doi:10.1007/978-3-031-72630-9\_25

Yotam Erel, Olaf Dünkel, Rishabh Dabral, Vladislav Golyanik, Christian Theobalt, and Amit H Bermano. 2025. Attention (as Discrete-Time Markov) Chains. In Advances in Neural Information Processing Systems, Vol. 38. Curran Associates, Inc., Red Hook, NY, USA, 54658–54690. doi:10.52202/085713-1828

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. 2024. Scaling Rectified Flow Trans formers for High-Resolution Image Synthesis. In Proceedings ofthe 41st International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 235). PMLR, Vienna, Austria, 12606–12633. https://proceedings.mlr.press/v235/esser24a. html

Weixi Feng, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, and William Yang Wang. 2023. Training-free structured difusion guidance for compositional text-to-image synthesis. The Eleventh International Conference on Learning Representations (ICLR). https://openreview. net/forum?id=PUIqjT4rzq7

Shengchuan Gao, Teng Hu, Bohao Feng, Luchen Li, Wenqiang Wang, Hongqian Deng, and Ran Yi. 2026. Eficient Audio-Visual Generation via Synchrony-Aware Cross Modal Sparse Attention. arXiv:2608.15522 [cs.CV] doi:10.48550/arXiv.2608.15522

Google DeepMind. 2025. Veo 3 Technical Report. https://storage.googleapis.com deepmind-media/veo/Veo-3-Tech-Report.pdf.

Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, Eitan Richardson, Guy Shiran, Itay Chachy, Jonathan Chetboun, Michael Finkelson, Michael Kupchick, Nir Zabari, Nitzan Guetta, Noa Kotler, Ofir Bibi, Ori Gordon, Poriya Panet, Roi Benita, Shahar Armon, Victor Kulikov, Yaron Inger, Yonatan Shiftan, Zeev Melumian, and Zeev Farbman. 2026. LTX-2: Eficient Joint Audio Visual Foundation Model. arXiv:2601.03233 [cs.CV] https://arxiv.org/abs/2601.03233

Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. 2024. Ltx-video: Realtime video latent difusion.

Moayed Haji-Ali, Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov, Alper Canberk, Kwot Sin Lee, Vicente Ordonez, and Sergey Tulyakov. 2025. Av-link: Temporally-aligned difusion features for cross-modal audio-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. IEEE, Piscataway, NJ, USA, 19373–19385.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2023. Prompt-to-prompt image editing with cross attention control. The Eleventh International Conference on Learning Representations (ICLR). https: //openreview.net/forum?id=\_CDixzkzeyb

Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. 2022a. Imagen video: High definition video generation with difusion models. arXiv:2210.02303 https://arxiv.org/abs/2210.02303

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. 2022b. Video Difusion Models. Advances in Neural Information Processing Systems 35 (2022), 8633–8646.

Yinming Huang, Shuyuan Tu, Xi Yan, Zihan Yang, Jianhua Han, Xu Hang, Yu-Gang Jiang, and Zuxuan Wu. 2026. VA-Judger: Reward Modeling from Human Preference Feedback for Joint Video-Audio Generation. arXiv:2608.18607 [cs.CV] doi:10.48550/ arXiv.2608.18607

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. 2024. VBench: Comprehen sive Benchmark Suite for Video Generative Models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, Piscataway, NJ, USA, 21807–21818.

Saar Huberman, Or Patashnik, Omer Dahary, Ron Mokady, and Daniel Cohen-Or. 2025. Image Generation from Contextually-Contradictory Prompts. arXiv:2506.01929 https://arxiv.org/abs/2506.01929

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. 2024. Hunyuanvideo: A systematic framework for large video generative models

Liyang Li, Wen Wang, Canyu Zhao, Tianjian Feng, Zhiyue Zhao, Hao Chen, and Chunhua Shen. 2026. MMControl: Unified Multi-Modal Control for Joint Audio-Video Generation. European Conference on Computer Vision (ECCV). arXiv:2604.19679 [cs.CV] doi:10.48550/arXiv.2604.19679

Yumeng Li, Margret Keuper, Dan Zhang, and Anna Khoreva. 2023. Divide & bind your attention for improved generative semantic nursing. In 34th British Machine Vision Conference (BMVC 2023). BMVA Press, Aberdeen, UK, Article 366, 12 pages. https://papers.bmvc2023.org/0366.pdf

Haohe Liu, Gael Le Lan, Xinhao Mei, Zhaoheng Ni, Anurag Kumar, Varun Nagaraja, Wenwu Wang, Mark D Plumbley, Yangyang Shi, and Vikas Chandra. 2024. Syncflow: Toward temporally aligned joint audio-video generation from text. arXiv:2412.15220 https://arxiv.org/abs/2412.15220

Kai Liu, Wei Li, Lai Chen, Shengqiong Wu, Yanhao Zheng, Jiayi Ji, Fan Zhou, Jiebo Luo, Ziwei Liu, Hao Fei, et al. 2025. Javisdit: Joint audio-video difusion transformer with hierarchical spatio-temporal prior synchronization.

Chetwin Low, Weimin Wang, and Calder Katyal. 2025. Ovi: Twin backbone cross-modal fusion for audio-video generation.

Bingqi Ma, Linlong Lang, Ming Zhang, Dailan He, Xingtong Ge, Yi Zhang, Guanglu Song, and Yu Liu. 2026. Improving Joint Audio-Video Generation with Cross-Moda Context Learning. arXiv:2603.18600 [cs.CV] doi:10.48550/arXiv.2603.18600

Sunung Mun, Jinhwan Nam, Sunghyun Cho, and Jungseul Ok. 2025. Addressing Text Embedding Leakage in Difusion-based Image Editing. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. IEEE, Piscataway, NJ, USA, 16451–16460.

William Peebles and Saining Xie. 2023. Scalable Difusion Models with Transformers. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. IEEE, Piscataway, NJ, USA, 4195–4205.

Royi Rassin, Eran Hirsch, Daniel Glickman, Shauli Ravfogel, Yoav Goldberg, and Gal Chechik. 2023. Linguistic Binding in Difusion Models: Enhancing Attribute Corre spondence through Attention Map Alignment. In Advances in Neural Information Processing Systems, Vol. 36. Curran Associates, Inc., Red Hook, NY, USA, 3536–3559. doi:10.52202/075280-0157

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. IEEE, Piscataway, NJ, USA, 10684–10695.

Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo. 2023. Mm-difusion: Learning multi-modal difusion models for joint audio and video generation. In Proceedings ofthe IEEE/CVFconference on computer vision and pattern recognition. IEEE, Piscataway, NJ, USA, 10219–10228.

Kazuki Shimada, Christian Simon, Takashi Shibuya, Shusuke Takahashi, and Yuk Mitsufuji. 2026. SAVGBench: Benchmarking Spatially Aligned Audio-Video Genera tion. In ICASSP 2026–2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, Piscataway, NJ, USA, 11977–11981. doi:10.1109/ ICASSP55912.2026.11464978

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. 2023. Make-a-video: Text-to-video generation without text-video data. The Eleventh International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=nJfylDvgzlq

Mingzhen Sun, Weining Wang, Yanyuan Qiao, Jiahui Sun, Zihan Qin, Longteng Guo, Xinxin Zhu, and Jing Liu. 2024. Mm-ldm: Multi-modal latent difusion model for sounding video generation. In Proceedings ofthe 32nd ACM International Conference on Multimedia. Association for Computing Machinery, New York, NY, USA, 10853– 10861. doi:10.1145/3664647.3680889

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, Vol. 30. Curran Associates, Inc., Red Hook, NY, USA, 5998–6008. https://proceedings.neurips.cc/paper\_files/paper/ 2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

Mor Ventura, Michael Toker, Or Patashnik, Yonatan Belinkov, and Roi Reichart. 2026. Deleaker: Dynamic inference-time reweighting for semantic leakage mitigation in text-to-image models. The Fourteenth International Conference on Learning Representations (ICLR). arXiv:2510.15015 https://arxiv.org/abs/2510.15015

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruil Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. 2025. Wan: Open and Advanced

Large-Scale Video Generative Models.

Tianyi Wei, Yifan Zhou, Dongdong Chen, and Xingang Pan. 2025. Freeflux: Understanding and exploiting layer-specific roles in rope-based mmdit for versatile image editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision. IEEE, Piscataway, NJ, USA, 16745–16754.

Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. 2023. Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, Piscataway, NJ, USA, 1–5. doi:10.1109/ICASSP49357.2023.10095969

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. 2025. Qwen3-Omni Technical Report. arXiv:2509.17765 [cs.CL] doi:10.48550/arXiv.2509.17765

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2024. Cogvideox: Text-to-video difusion models with an expert transformer. arXiv:2408.06072 https://arxiv.org/abs/2408.06072

Guozhen Zhang, Zixiang Zhou, Teng Hu, Ziqiao Peng, Youliang Zhang, Yi Chen, Yuan Zhou, Qinglin Lu, and Limin Wang. 2026. UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, Piscataway, NJ, USA, 1950–1960.

## A Manipulation In The Wild

We evaluate the generality of our approach by applying the proposed steering framework across a diverse collection of open-domain audio-video generation scenarios. We consider examples spanning diferent visual domains, including realistic humans, animals, animated characters, stylized cartoons, and multi-entity scenes with varying spatial layouts and interaction patterns. The evaluated prompts intentionally introduce tension between the requested sound and the canonical visual identity of the speaking entity, creating challenging conditions for source attribution.

Across these settings, we observe several recurring types of leakage:

• Source-attribution leakage: speech or sound-producing behavior is assigned to the wrong entity in the scene.

• Appearance leakage: sound semantics alter the visual appearance of an entity (e.g., a pirate acquiring parrot-like features or a teddy bear becoming monster-like).

• Voice-characteristic leakage: the generated voice follows canonical priors associated with the visual entity rather than the prompt specification (e.g., a child failing to produce a deep adult voice).

• Generation suppression: strong conflicts between the prompt and learned cross-modal priors cause the model to avoid producing the requested sound or speaking behavior altogether.

Our manipulation framework mitigates these failures by steering the routing of information across the attention triangle. Jointly controlling the text-audio, text-video, and audio-video interactions enables more stable source localization while preserving the intended visual appearance of each entity. These efects persist across diverse prompts, motion patterns, semantic mismatches, and visual styles.

## B Additional Qualitative Comparisons

We provide an additional qualitative comparison in Fig. 7, complementing the horse-neigh example shown in the main paper (Fig. 6). This example targets a diferent leakage regime: rather than reattributing an animal-like sound between two realistic entities, the prompt asks a teddy bear to produce a deep growl in the presence of a furry monster, creating tension between the requested sound and the canonical visual prior associated with the monster.

![](images/bac3f0c1e922a893d5e3497b0b38fa9983b9821844dd403e7a8fe0a265ae2023.jpg)  
A furry monster sits beside a small teddy bear on a blanket. The monster stays quiet. The teddy bear is the“ only sound source, opening its stitched mouth. It produces a deep, rumbling monster growl, scratchy and cavernous, with a low, vibrating finish.”  
Fig. 7. Furry-monster growl atribution. The prompt asks for a deep growl that should be produced by the teddy bear rather than the furry monster. Three representative frames are shown per method. Baselines and partial-edge variants either route the growl to the furry monster or cause the teddy to acquire monster-like visual features; Ours-Full preserves the teddy’s identity while binding the growl to it. Zoomed regions indicate the active sound source.

Across the compared methods we observe behavior consistent with the main-paper example. The native LTX-2 baseline routes the growl to the furry monster, reproducing the canonical sound-source prior and ignoring the explicit text specification. The Bounded Attention adaptation, which restricts text-to-video attention to the speaker region, partially relocates the growl but does not consistently suppress the audio-mediated appearance leakage. Ours-Text sharpens the textual binding but is unable to overcome the strong audio-video prior that ties growling to monster-like appearance. Ours-AV successfully transfers the growl onto the teddy, but in doing so injects monster-like features into the teddy’s appearance, replacing one leakage mode with another. Only Ours-Full, which jointly manipulates the text and audio-video edges, simultaneously binds the growl to the teddy and preserves its visual identity.

Together, the horse-neigh and furry-monster examples illustrate that the same failure modes recur across very diferent visual domains and sound categories, and that the joint manipulation of all edges of the attention triangle is necessary to mitigate them.

## C Quantitative Evaluation Protocols

We provide protocol and provenance details for the main-paper quantitative comparison in Tab. 1, which evaluates the native LTX-2 baseline, a video-only diagnostic variant, the external Ovi baseline, an adapted text/image-side Bounded Attention baseline, and three variants of our attention-triangle intervention (Ours-Text, Ours-AV, and Ours-Full). The table reports targeted source attribution, audiotext alignment with CLAP, and visual fidelity using VBench subject consistency, background consistency, and aesthetic quality.

## C.1 Qwen Source-Atribution Score

Scope and judge. We score the six audio-bearing methods on the full quantitative corpus: 100 challenge prompts, four matched seeds per prompt, and 400 videos per method, for 2,400 videos total. Each video receives two counterbalanced passes, yielding 4,800 Qwen evaluations. The video-only diagnostic is omitted because it contains no audio. We use Qwen3-Omni-30B-A3B-Instruct [Xu et al. 2025]. The synchronized clip is supplied with audio, sampled at 5 fps, and resized to 672×448. The original generation prompt is hidden: the judge receives only the target-event description and the two visible candidate sources, preventing prompt compliance from substituting for audio-video evidence.

## System prompt.

You are an audiovisual evidence annotator. Watch and listen to the entire synchronized clip. Decide which visible candidate, if any, is presented as producing the target audio event. Base your decision only on evidence in the clip, especially sound synchronized with repeated mouth, beak, body, or object motion.

The target-event description identifies what sound to find; it does not identify its source. Generated clips may violate real-world expectations. Do not infer the source from species, identity, voice characteristics, narrative plausibility, candidate order, or option order.

## User prompt template.

Target audio event: {target\_event}

Candidate A: {candidate\_a}

Candidate B: {candidate\_b}

Which candidate is presented as producing the target audio event?

1 = Candidate A only.

2 = Candidate B only.

3 = Both candidates are each presented as producing

it, simultaneously or at different times.

4 = The target audio event is not recognizably   
audible.   
5 = The target audio event is audible, but the clip does not provide enough synchronized visible evidence to assign it to Candidate A or Candidate B. This includes an off-screen or different source.   
Answer with exactly one digit.   
Answer:

Logit probabilities and counterbalancing. We do not sample a free-form response. At the first answer position, we extract the nexttoken logits for the five single-token choices 1 to 5. We normalize these logits within the five-choice answer set, obtaining probabilities for Candidate A only, Candidate B only, both, event absent, and source unclear; we separately retain the five choices’ total probability mass under the full-vocabulary softmax as a closed-set diagnostic. In pass 1, Candidate A is the intended source and Candidate B is the competing source. In pass 2, their order is reversed. We then map both distributions to the order-independent labels � (intended only), � (competing source only), � (both), � (event absent), and � (source unclear), and average the two canonical distributions. This swap removes a fixed Candidate-A or option-order preference from the reported score.

Score and results. Let $p _ { I } , p _ { C } , p _ { B } , p _ { A } , p _ { U }$ be the averaged canonical probabilities and $m = p _ { I } + p _ { C } + p _ { B }$ the probability mass assigned to a visible source. We define conditional leakage as $X = ( p _ { C } + p _ { B } ) / m$ and report the higher-is-better score $S _ { \mathrm { a t t r } } = 1 - X = { \mathop { p _ { I } } } / { m }$ , averaged over videos within each method. Thus, only an intended-only assignment counts as correct; competing-only and both-source assignments count as leakage, while event-absent and source-unclear mass remain separate diagnostics. All 2,400 videos produced two valid, counterbalanced scores, with no missing, duplicate, or malformed records. Mean scores are 0.1216 (Native LTX-2), 0.1207 (Ovi), 0.0925 (Bounded Attention), 0.1140 (Ours-Text), 0.1302 (Ours-AV), and 0.1349 (Ours-Full), with Ours-Full performing best.

## C.2 VA-Judger Pairwise Audio-Video Preference

Purpose. The Qwen source-attribution score above deliberately asks one narrow question: which visible entity produced the target sound? As a complementary, broader check, we run the released ShareLab-SII VA-Judger [Huang et al. 2026] pairwise evaluator. This check asks whether Ours-Full is preferred overall after jointly consid ering prompt fidelity, audiovisual consistency, audio quality, video quality, and completeness. It does not replace the targeted attribution metric or the human evaluation.

Exact inputs and judge. For every comparison, the judge receives the full original generation prompt followed by two synchronized audio/video clips using the exact user template Text Caption: {text\_prompt}, Audio/video 1: <video>, and Audio/video 2: <video>. The system instruction asks it to assign each video a score from 1 to 10 and a concise rationale for the five dimensions shown in Fig. 8, sum the five scores, and return either video 1 is better or video 2 is better. We use ShareLab-SII/VA-Judger, frozen at model revision 1a428...e954e.

Counterbalancing and decoding. For each reference, we compare the matched Ours-Full and reference outputs for all 100 prompts and four seeds. Pass 1 presents Ours-Full first and pass 2 reverses the order, producing 400 pairs and 800 decisions per reference. Decoding is deterministic (temperature 0, top-� = 1, top-� = −1, seed 42), with bfloat16 inference, a 24,576-token context limit, and at most 2,048 new tokens. Across the five references this yields 2,000 pairs and 4,000 decisions.

Aggregation and audit. After mapping each answer back to method identity, an Ours-Full win receives 1, a reference win receives 0, and an explicit equal-total answer receives 0.5; the two pass values are averaged per pair. We report the mean over 400 pair values and a percentile 95% confidence interval from 10,000 pairlevel bootstrap resamples (seed 42). All 4,000 decisions completed without inference errors or duplicate composite prompt/seed/order keys, and all 2,000 pairs passed the counterbalancing and scoreparsing audit. The six explicit equal-total decisions were retained as ties rather than forced into wins; one omitted dimension-line value was recovered exactly from the same completion’s displayed five-term total. Fig. 8 shows that Ours-Full is preferred to every alternative, with all five pointwise confidence intervals above 0.5, while all displayed mean dimension diferences are positive.

## D User Study

We conduct pairwise human preference studies comparing Ours-Full against each comparison method (Native LTX-2, Bounded Attention, Ours-AV, and Ours-Text) across three axes: (i) correct sound-source attribution, (ii) reduced visual leakage (appearance leakage), and (iii) overall generation quality. For each comparison, annotators are shown two videos generated from the same prompt and asked to choose which one better satisfies the corresponding criterion. The full pairwise preferences are reported in Fig. 9. Across all three axes, annotators consistently prefer Ours-Full, with the largest gap on sound-source attribution and visual leakage, supporting the efectiveness of jointly steering every edge of the attention triangle.

Protocol. We conduct a randomized pairwise human preference study comparing Ours-Full against four alternatives: Native LTX-2, Bounded Attention, Ours-AV, and Ours-Text. Each trial presents two anonymized videos, Video A and Video B, generated from the same prompt and seed. One video is always produced by Ours-Full and the other by the corresponding comparison method; the side containing Ours-Full is randomized independently for every trial. Each participant completes 16 trials, consisting of 4 comparisons against each comparison method. The sampler ensures that the same prompt/seed example does not appear twice for the same participant. The user-study interface is shown in Fig. 10.

Stimuli and task metadata. The active stimulus pool contains 37 unique prompt/seed examples and 118 active pairwise comparisons after removing one low-quality stimulus. Each example is manually annotated with the prompt, the intended sound source, the target sound, and the entity or object that should remain silent. These annotations are shown to participants during each trial, making the source-attribution and leakage criteria explicit.

Questions. For each video pair, annotators answer three forcedchoice questions with three response options: (i) which video better makes the intended source appear to be the source of the sound, (ii) which video better prevents the competing source or another

(a) Pairwise preference score  
![](images/07f4442fb8d886212b997ca01d8beb39a96da541242b32b0045584a1fea7a3ec.jpg)

(b) Mean dimension-score gain
<table><tr><td></td><td></td><td rowspan=1 colspan=1>Audio</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>+.72</td><td rowspan=1 colspan=1>+.52</td><td rowspan=1 colspan=1>+.29</td><td rowspan=1 colspan=1>+.17</td><td rowspan=1 colspan=1>+.71</td><td></td><td rowspan=1 colspan=1>51.8%</td></tr><tr><td rowspan=1 colspan=1>+1.58</td><td rowspan=1 colspan=1>+1.25</td><td rowspan=1 colspan=1>+.61</td><td rowspan=1 colspan=1>+1.29</td><td rowspan=1 colspan=1>+1.56</td><td></td><td rowspan=1 colspan=1>69.3%</td></tr><tr><td rowspan=1 colspan=1>+2.24</td><td rowspan=1 colspan=1>+1.38</td><td rowspan=1 colspan=1>+.44</td><td rowspan=1 colspan=1>+.84</td><td rowspan=1 colspan=1>+2.04</td><td></td><td rowspan=1 colspan=1>68.5%</td></tr><tr><td rowspan=1 colspan=1>+.57</td><td rowspan=1 colspan=1>+.47</td><td rowspan=1 colspan=1>+.26</td><td rowspan=1 colspan=1>+.08</td><td rowspan=1 colspan=1>+.58</td><td></td><td rowspan=1 colspan=1>51.2%</td></tr><tr><td rowspan=1 colspan=1>+.63</td><td rowspan=1 colspan=1>+.46</td><td rowspan=1 colspan=1>+.24</td><td rowspan=1 colspan=1>+.16</td><td rowspan=1 colspan=1>+.66</td><td></td><td rowspan=1 colspan=1>51.2%</td></tr></table>

Fig. 8. VA-Judger pairwise audio-video preference. Ours-Full is compared with each reference on the same 400 prompt/seed pairs, each shown in both presentation orders (800 decisions per row). Left: mean pair score after averaging the two orders (win = 1, tie = 0.5, loss = 0), expressed as a percentage; bars are pointwise 95% bootstrap confidence intervals over pairs, and the dashed line marks chance. Right: mean Ours-Full-minus-reference diferences on the five judge dimensions scored from 1 to 10 (prompt fidelity, A/V consistency, audio quality, video quality, and completeness), averaged across both orders; darker green indicates a larger gain. The gray column reports order agreement separately; because agreement ranges from 51% to 69%, we interpret only the aggregated, counterbalanced scores. All five pair-score intervals lie above 50%, and all displayed dimension diferences are positive.

![](images/eac929ec5d4c7a5c19a9591adb8c3400c311e3bfcb4c344941da6ef519b51ecd.jpg)  
Fig. 9. Pairwise user-study preference. Annotators compare Ours-Full against each comparison method across sound-source atribution, visual leakage and overall quality. Bars show percentages over all votes: preference for Ours-Full (green), no preference/same (gray), and preference for the comparison method (red). “Same” responses are treated as ties and are excluded from decided-vote significance tests.

non-target entity from receiving incorrect visual attributes, mouth motion, or source-like behavior, and (iii) which video is better overall, considering attribution, audio-video coherence, visual quality, and audio quality. For every question, the response options are Video A, Same, and Video B. We treat Same as a tie: it is visualized explicitly in Fig. 9 and excluded from decided-vote win rates and significance tests.

Responses and aggregation. We collect 30 unique complete submissions, yielding 480 pairwise trial records. All submissions contain the full set of 16 trials and no response payload fails parsing. For each question and comparison method, we count votes for Ours-Full, votes for the comparison method, and Same votes. The percentages in Fig. 9 are computed over all votes, including Same as a neutral response.

## E Prompts and Video Evidence

## E.1 Construction of the Leakage Challenge Set

We constructed the leakage challenge set through a human-in-theloop process that resembles evolutionary prompt search; human-in the-loop curation has similarly been used to construct diagnostic prompt suites for joint audio-video generation [Cui et al. 2026]. One author began with a manually designed population of prompts that place an intended sound source in conflict with a visible competing source that is strongly associated with the requested sound, voice, or semantic content but is instructed to remain silent. We generated the corresponding videos and manually reviewed their synchronized audio and visual behavior; prompts that produced a clear and interpretable binding failure were selected, whereas prompts with no leakage or unjudgeable generation artifacts were pruned. In sub sequent rounds, we “mutated” successful prompt families by varying the entities, requested sounds or dialogue, semantic associations, visual cues, wording, and generation seeds, while preserving the conflict between the intended and competing sources; productive variants were retained and expanded, and new prompt families were occasionally introduced. This iterative generate, review, select, and mutate process yielded a deliberately failure-enriched set for evaluating whether an intervention repairs known failures, rather than a random sample suitable for estimating leakage prevalence in ordinary generation. After selection, each instance was annotated with its intended source, requested sound or action, competing source, and seed; the annotation tags are removed before text encoding, and all compared LTX-2 methods use the same cleaned prompt and seed.

(b) Per-trial annotation  
![](images/1f28898d899eae533cbe7bff9f0205594f17a5a8e805dabfdac53a42b8e52d77.jpg)  
Fig. 10. User-study interface. (a) Onboarding screen shown before each session, anchoring the annotators’ criteria with the parrot/pirate leading example: a side-by-side baseline-vs-steered pair illustrating the source-atribution and visual-leakage failures the study targets. (b) Per-trial annotation screen presenting two anonymized videos (Video A, Video B) generated from the same prompt and seed. The target sound and the intended source/silent-entity annotations are shown explicitly above the videos; annotators then answer three forced-choice questions (source atribution, visual leakage, and overall quality), each with the options Video A, Same, and Video B.

The full list of evaluation prompts and the corresponding generated videos for all compared methods (Native LTX-2, Bounded

Attention, Ours-Text, Ours-AV, Ours-Full) are provided as a static HTML page in the supplementary materials accompanying this submission. Because the leakage phenomena studied here require synchronized audio and video, we strongly encourage readers to inspect the videos directly; static frames in this manuscript necessarily understate both the failure modes and the steering corrections, particularly for voice-characteristic leakage and generation suppression, which are not visible in still imagery.

## F Limitations

Our approach is training-free but inherits several limitations from the base generator and the grounding signals used for steering. First, attention steering can only redirect associations that the model already represents; it cannot reliably create audio-video bindings that are far outside the model’s learned distribution. Second, the method depends on anchors extracted from an initial unsteered generation, so cases where the audio is completely ungrounded or the intended source is visually static may provide too little signal to correct. Third, the visual anchors rely on SAM3 masks over decoded frames, making the intervention sensitive to segmentation errors for small, occluded, or ambiguous entities. Finally, the extra anchor-extraction pass increases inference cost, and our experiments are centered on LTX-2-style audio-video generation; evaluating the same attentiontriangle intervention across additional model families remains an important direction for future work.

## G Analysis Details

## G.1 Leakage Visualization Details

To probe leakage for a specific entity using $\mathbf { P } _ { V V } ^ { \mathrm { e f f } } \left( \mathrm { E q . } 1 \right)$ , let $\pmb { \pi } _ { V } \in \mathbb { R } ^ { N _ { V } }$ be a column vector that assigns uniform probability to the entity’s visual mask and zero elsewhere. Because $\mathbf { P } _ { V V } ^ { \mathrm { e f f } }$ is a row-transition matrix, we propagate the distribution as $\pmb { \pi } _ { V } ^ { \prime \top } = \pmb { \pi } _ { V } ^ { \top } \mathbf { P } _ { V V } ^ { \mathrm { e f f } }$ , or equivalently $\pmb { \pi } _ { V } ^ { \prime } = ( \mathbf { P } _ { V V } ^ { \mathrm { e f f } } ) ^ { \top } \pmb { \pi } _ { V }$ . The output $\pmb { \pi } _ { V } ^ { \prime }$ reveals which video patches receive attention mass from the entity’s region after one audiomediated round trip. On leakage prompts the mass concentrates on the competing-source patches (e.g., the pirate), confirming that the audio stream acts as a hidden routing mechanism that redirects semantic information from the intended source to the visually familiar one.

Figure 11 visualizes this routing efect in two LTX-2 examples: Ours-Full concentrates the returned visual mass on the intended sound source while suppressing the competing source.

## G.2 Experimental Setup

We use LTX-2 [HaCohen et al. 2026], which ships in two variants: a text-to-video (T2V) variant with no audio branch, and a joint text-to-audio-video (T2AV) variant that adds an audio stream conditioned on the same text prompt. The two variants share essentially the same visual backbone; the T2AV variant augments it with audio tokens and the full attention triangle described in Section 3. This design makes LTX-2 particularly well suited for isolating the contribution of the audio-video interaction to leakage, since any behavioral diference between the two variants must originate from the audio branch.

## H Steering Design Details

## H.1 Anchor Design: Hard vs. Soft Masks

The choice between hard SAM3 masks and soft, model-derived attention masks is dictated by which modality admits a reliable external segmenter.

On the visual side, SAM3 provides hard masks for the intended and competing sources. The audio-video agreement matrix uses the intended-source mask, with its complement representing all non-source regions. The text-conditioned video bias instead uses both masks explicitly, reinforcing intended-source regions and suppressing competing-source regions.

On the audio side no analogous external segmenter is available, so we derive $\mathbf { m } _ { \mathrm { s n d } } ^ { A }$ from the model’s audio-query/text-key saliency targeting $\begin{array} { r } { \underline { { \boldsymbol { \mathcal { T } } } } _ { \mathrm { s n d } } ^ { T } . } \end{array}$ This soft mask is used directly on the audio-video edge and thresholded at $\theta _ { A } = 0 . 3$ to obtain $\bar { \mathbf { m } } _ { \mathrm { s n d } } ^ { A }$ for the audio-query/textkey surface. Empirically, this attention-derived localization is sufficient to identify the speaking regions in the audio stream and to drive the text-to-audio bias without over-constraining generation.

![](images/0e7a5f092e17c2d03b9e63e13a44c701f0a1201d230f08963ee26c3073e40485.jpg)  
Fig. 11. Cross-modal atention routing in LTX-2. Each panel compares the unmodified baseline (upper internal row) with Ours-Full (lower internal row). Columns show the intended-source visual seed distribution $\pi _ { V } ,$ , the audio-bridge atention $\pi _ { A } ,$ and the returned visual mass $\pi _ { V } ^ { \prime }$ . Labels T, S, and R denote the percentage of mass assigned to the intended source, competing source, and residual patches. Top: with the golden retriever as the intended source and the orange cat as the competing source, Ours-Full increases intended-source mass from 46.1% to 90.3% and reduces competing-source mass from 39.5% to 6.6%. Botom: with the polar bear as the intended source and the expedition leader as the competing source, intended-source mass increases from 21.2% to 88.6%, while competing-source mass falls from 27.4% to 2.3%.

## H.2 Hyperparameter Analysis

Audio-video edge (�). The soft intended/conflicting construction on this edge has one relative-gap degree of freedom after a softmax row shift: $- \lambda ( \mathbf { 1 } _ { N _ { V } \times N _ { A } } - \mathbf { G } _ { V A } )$ is equivalent to $+ \lambda G _ { V A }$ because the two difer by the row-constant $- \lambda$ , which softmax discards. We use the penalty form so that � reads as the maximum logit penalty applied to a conflicting cell. We set �=10 throughout and do not tune it per prompt.

Text edges $( \beta , \gamma ) .$ Unlike the audio-video edge, the three-class partition on the text-conditioned surfaces leaves two independent relative-gap degrees offreedom after a softmax row shift: the intendedvs.-neutral gap � controls how strongly the correct binding beats unmarked text tokens, and the neutral-vs.-conflicting gap � controls how strongly conflicting cells are pushed below them. $\beta$ and � are therefore not interchangeable. The asymmetric default $\beta { = } 0 . 5 , \ \gamma { = } 2 . 0$ places greater weight on pushing conflicting cells below neutral than on lifting intended cells above neutral. These values are fixed across all LTX-2 experiments and are not tuned per prompt.

![](images/9d87137163267fce063cf98629135e87624ea3b72b051e4c9857f8662eb419ef.jpg)  
Baseline  
Steerd  
Baseline  
Steerd  
Fig. 12. Representative Ovi comparisons. For each example, columns compare the Ovi baseline with Ours-Full, and rows show three matched timestamps. Left: the prompt assigns a woman’s voice to the man while the woman remains silent. Right: the prompt assigns horse neighs to the pig while the horse remains silent. The corresponding videos provide the audio comparison; these frames show the visual context and temporal consistency of each output.

Audio-mask threshold $( \theta _ { A } )$ . We use the fixed threshold $\theta _ { A } = 0 . 3$ to binarize the soft audio mask for the text-conditioned audio surface. This value is used across all LTX-2 experiments and is not tuned per prompt.

## H.3 Compute Resources

All runs use a single NVIDIA RTX A6000 (48 GB) on an internal on-prem cluster; one clip per GPU, no parallel sharding. At the SD config used throughout (960×544, 97 frames, 20 steps), baseline LTX-2 takes ≈ 5.5 min/clip and the Ours-Full pipeline approximately 13 to 14 min/clip; peak memory is ≈ 29 GB in both cases. Including preliminary runs and hyperparameter sweeps, the full project consumed approximately 1,300 A6000 GPU-hours.

## H.4 Representative Ovi Comparisons

Figure 12 shows representative Ovi [Low et al. 2025] baseline and Ours-Full outputs for two source-attribution prompts. Still frames document the visual context; sound-source attribution must be assessed from the corresponding videos.