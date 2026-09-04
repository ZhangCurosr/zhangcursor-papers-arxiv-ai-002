# StrixAE: An Intelligent Agent for Audio Enhancement under Complex Distortion Coupling in Real-World Scenarios

Chenglin Wu<sup>1,∗</sup> Junjie Wu<sup>1,∗</sup> Jinhong Chen<sup>1</sup> Mingyang Chen<sup>1</sup> Zixu Lin<sup>1</sup> Jiabian Chen<sup>2</sup> Xinghao Ding<sup>1</sup> Xiaotong Tu<sup>1</sup>

<sup>1</sup>Xiamen University <sup>2</sup>Fuyao University of Science and Technology <sup>∗</sup>Equal contribution.

## Abstract

Audio enhancement in real-world scenarios involves complex distortion couplings and requires personalized enhancement. Existing solutions struggle to address both simultaneously. To improve robustness andenable autonomous operation in such scenarios, we propose StrixAE, an agent based on a multimodal large language model (MLLM). StrixAE leverages the MLLM as a controller to coordinate multiple audio enhancement and personaliza-tion models. To further enhance system robustness, reduce artifacts, and improve generalization across diverse real-world scenarios, StrixAE is trained through a two-stage process: first, CoT supervised fine-tuning on AcoustBench to ground basic reasoning and tool invocation; second, Audio Perception Reinforcement Learning (APRL) — a reward design specifically tailored for audio restoration pipelines that jointly optimizes format validity, structural coherence, and perceptual quality. Unlike generic RL fine-tuning, APRL introduces structured rewards that enforce executable pipelines and logical section ordering, enabling the agent to produce reliable, interpretable enhancement plans without hallucinated tools. Based on real-world test datasets, our proposed method outperforms most existing open-source and proprietary solutions, achieving stateof-the-art performance across multiple perception metrics and demonstrating strong generalization robustness.

## 1 Introduction

Speech enhancement (SE) aims to improve speech quality and intelligibility while preserving the intrinsic characteristics of the original signal [25]. In real-time communication, speech quality is significantly compromised by background noise [31], reverberation [17], and interfering speakers [34, 20]. This field encompasses tasks such as noise reduction [40], dereverberation [23], speech separation [54, 45], and bandwidth extension [30]. However, current audio enhancement models are predominantly categorized into task-specific methods [40, 23, 4, 3] and ensemble methods [55, 6, 16, 23, 15, 42], often trained on limited datasets [23, 40, 6], lacking a standardized benchmark for versatility and a holistic architecture to handle diverse acoustic conditions.

Conventional task-specific methods [23, 40, 6] primarily focus on single-objective speech enhancement, such as denoising or dereverberation. Although ensemble-based strategies extend this scope to composite degradations, they often fail in complex environments that require selective enhancement or suppression of interfering speakers. The inherent unpredictability and coupled complexity of real-world audio further challenges the generalization ability of existing models, as illustrated in Figure 1. Consequently, there remains a critical need for standardized benchmarks that can evaluate system versatility, as well as holistic architectures capable of handling diverse acoustic conditions across multiple scenarios.

![](images/bd551e1c362ad70a4a19903ab8815acc6ee316eb07738edbda05261376f4c307.jpg)  
Figure 1: Limitations of Single-Task and All-In-One methods: (1) Single methods only focus on limited distortion conditions and augmentation on specific datasets. (2) All-In-One methods cannot address the needs of coupled distortion and personalized augmentation in real-world scenarios.

Recently, Multimodal large language model (MLLM) have achieved breakthroughs in reasoning and autonomous task execution [13, 47, 56, 21, 7]. Because the type and combination of distortions are unknown a priori, any fixed pipeline or static model fails. This naturally calls for a reasoning-capable controller, which can be provided by MLLM. Inspired by this, we explore a novel audio enhancement paradigm featuring proactive perception of real-world distortions, interpretable handling of composite degradation, and professional-grade enhancement. While MLLMs demonstrate inherent audio perception capabilities as controllers for expert models, constructing such an agent requires extensive paired supervisory data. The scarcity of labeled distorted audio hinders supervised fine-tuning [50, 34]. To address these issues, we construct a multi-scenario dataset with coupled degradations and curate 88.9K supervised samples.

In this paper, we propose StrixAE, an audio agent based on multimodal large-scale models (MLLMs), leveraging Audio-Reasoner [43] and expert models from open-source communities. The system development comprises two key components: (1) AcoustBench, an instruction-following dataset built via a self-guided strategy [41], containing 200 hours of personalized DNS (PDNS) and dual/singlespeaker audio, 88.9K synthetic instruction-response pairs, and 1.9K real-world test pairs; (2) A training framework combining Supervised Fine-Tuning (SFT) and AI feedback alignment to enhance reliability and autonomy. We fine-tune StrixAE using AcoustBench to boost robustness, reduce hallucinations, and improve generalization and instruction alignment. To ensure stable training, we propose GRPO-AE, incorporating ranking-based supervision inspired by Ranked Responses from Human Feedback (RRHF). In addition, we integrate multiple audio quality assessment models into a unified reward model to provide comprehensive feedback.

The main contributions of our work can be summarized as follows:

• We propose StrixAE, a novel MLLM-based audio agent capable of perceiving degradationfactors and autonomously orchestrating expert models for multi-scenario audio enhancement.

• Two-stage training paradigm for audio agents. We propose SFT followed by APRL, which introduces a novel reward decomposition into format, structure, and perceptual quality. This design explicitly enforces executable pipelines and logical reasoning order, which is a critical yet overlooked requirement for audio restoration agents.

• A audio perception reward design for audio restoration. We introduce a novel reward design that combines modern objective audio metrics, serving as an effective surrogate for perceptual quality and promoting stable and efficient APRL optimization.

• AcoustBench: It comprises 88.9K instruction-response pairs with explicit chain-of-thought (CoT) annotations and executable tool chains, specifically curated to support training and evaluation of scheduling policies under coupled distortions.

## 2 Related Work

Speech Enhancement. Current research on audio enhancement is largely conducted within a singletask paradigm using task-specific datasets, where models are trained and evaluated separately for subtasks such as denoising [23, 53, 40], dereverberation [23, 18], speech-overlap suppression [45], source separation [54, 45, 55], audio restoration [19], and target-speaker extraction [15, 16]. In real-world applications, enhancement requirements are often scenario-dependent and should be inferred from the input audio itself, thereby requiring adaptive combinations of multiple subtasks. Moreover, existing task-specific enhancement models still exhibit limited generalization, making it difficult to handle complex, coupled degradations caused by factors such as environmental variability and device mismatch. They also lack sufficient flexibility for practical audio shaping. To bridge this gap, the Deep Noise Suppression (DNS) [31, 15, 16, 6] and URGENT challenges [50, 51, 34, 20] have introduced the Personalized Speech Enhancement (PSE) track and an adaptive multi-distortion processing setting, respectively.

LLM-Empowered Agent. Recent studies[28, 52, 37, 48]have highlighted the growth potential of multimodal large language models in terms of proficient tool use and decision-making in complex environments.MLLMs have benefited from powerful reasoning capabilities and rich tools that interact with the environment[12, 27]. Embodiedbench: specifically evaluates the embodied task performance of multimodal large language models (MLLMs)[48]. Gorilla improved the LLM’s responsiveness to tool calls by constructing datasets and fine-tuning[28].In the vision domain, various methods have been proposed to enhance large models for vision tasks, such as the Hugging Face model [36], image degradation[21], image art creation[22, 7], and the Azure model[49].Audio agents have not yet been developed.

Agentic Reinforcement Learning. Reinforcement learning (RL) plays a crucial role in aligning (multimodal) large language models [11, 46, 44] with human preferences. Reinforcement Learning from AI Feedback (RLAIF) [9, 38, 2]has emerged as an important paradigm for LLM alignment. In audio evaluation tasks, AI-generated feedback can effectively approximate human judgments, significantly reducing reliance on manual annotation while maintaining evaluation quality, thereby substantially lowering the cost of dataset construction. Building on this, RLAIF learns reward functions from AI-generated feedback and optimizes LLMs using reinforcement learning methods such as proximal policy optimization (PPO)[35]and GRPO[11]. Unlike typical agents, our two-stage fine-tuning process integrates audio and language modalities, without involving visual modalities.

## 3 Method

We first present the overall workflow of StrixAE in Sec. 3.1. We then describe the data generation pipeline used to construct AcoustBench, a high-quality audio dataset comprising cue samples and inference samples for audio enhancement tasks, in Sec. 3.2. Finally, we detail the core components of StrixAE, including its two-stage training process and the Agent-Orchestrated Multi-model Enhancement Protocol, in Sec. 3.3.

## 3.1 Overview

StrixAE is an audio enhancement model based on MLLM that can coordinate multiple audio enhancement models to process complex degraded audio data and achieve high-quality distortion

![](images/ea9553102fd8d2ad4b169c6a08ee7f7c3811115111f8043b52ab2ba9230af7ae.jpg)  
Figure 2: An illustration of our data curation pipeline.

audio enhancement. The StrixAE model operation mainly includes three steps: perceptual audio degradation inference, model selection and evaluation, and task execution and response generation, as shown in Figure 3.

$$
f ( Q , \mathcal { A } )  \mathcal { T } = \{ t _ { 1 } , t _ { 2 } , \dotsc , t _ { n } \}\tag{1}
$$

where I denotes the user instruction , A represents the distorted audio, T represents the normalized tool call sequence, where each t<sub>i</sub> represents a specific audio enhancement operation. Finally, the enhanced output is obtained by ${ \therefore } A _ { \mathrm { { e d i t } } } = G \left( { \mathcal { A } } , { \bar { \mathcal { T } } } \right)$ . G(.) represents the audio enhancement environment.

## 3.2 Dataset Construction

We construct AcoustBench, a novel audio enhancement benchmark with explicit chain-of-thought (CoT) annotations, via a four-stage data generation pipeline (2). Each dataset sample is defined as a 5-tuple: $\langle \mathcal { A } , \mathcal { A } _ { \mathrm { t g t } } , \mathcal { T } , C , \mathcal { T } \rangle$ ⟩, where ${ \mathcal { A } } _ { \mathrm { t g t } }$ denotes its clean target audio, C represents the step-by-step CoT reasoning.

Audio collection. We use real and clean audio datasets from the ICASSP 2023 DNS Challenge [6] and the 2025 URGENT Challenge [34, 20]. These datasets include 350 hours of clean audio, 181 hours of real noisy audio [39, 8, 10], and reverberation data [1]. Based on these resources, we construct 190 hours of paired audio with composite distortions. Further details are provided in Supplementary Material.

Instruction and Response Construction. Inspired by self-guided strategies [41], we first designed 10 initial instructions as seeds. Based on these seeds, we used GPT-5.3 to generate 20 refined instruction candidates. These candidates were then manually reviewed and selected to remove ambiguous, duplicated, or low-quality samples, resulting in a curated set of high-quality instructions. Furthermore, for each audio sample in the training set, we designed two recommended tool invocation paths for audio restoration. These candidate paths were evaluated based on the quality of their processed outputs, using DNSMOS [32] and ESTOI [14] as the assessment criteria. The selected paths were then paired with the corresponding clean and distorted audio samples to form the instruction-toolchain annotations.

Restoration Pipeline Generation. Based on the generated instructions and responses, we obtain instruction–toolchain response pairs and corresponding audio pairs. Specifically, Gemini-2.5-Flash [5] is used to generate detailed logical chain responses, which typically include acoustic feature analysis, problem diagnosis and reasoning, restoration logic reasoning, and a recommended restoration pipeline. The restoration tasks and tool models are sourced from GitHub or Hugging Face.

Dataset Filtering. To accommodate diverse real-world scenarios and enhance distorted audio at different sampling rates, we select high-quality instruction–response pairs of various types to satisfy scenario-specific requirements. AcoustBench contains 89K instruction–response pairs for the initial instruction-tuning stage. Further details are provided in Supplementary Material.

![](images/1ef50f47ba8fd9c0f9592e1de7e03838d02c1144733b35f17efaa97b09873ae4.jpg)  
Figure 3: StrixAE employs a two-stage training framework. Initially, StrixAE undergoes Supervised Fine-Tuning on AcoustBench synthesized data, following user instructions and enhancing distorted audio. In the second stage, the Audio Perceptual Reinforcement Learning (APRL) algorithm is applied to further enhance the robustness of StrixAE’s enhancement system and improve its adaptability to audio enhancement in various real-world scenarios.

## 3.3 StrixAE Framework

## 3.3.1 CoT Supervised Fine-tuning

We use the AcoustBench dataset to perform SFT to obtain the StrixAE-SFT model. Our multimodal instruction samples can be represented as triples (I, A, R). MLLM outputs the analysis process and the tool call chain based on the human instruction and the distorted audio: $\mathcal { R } = f ( \mathcal { I } , \mathcal { A } ; \theta )$

$$
L _ { s f t } = - \sum _ { i = 1 } ^ { N } \log P _ { \pi } \big ( \mathcal { R } _ { i } \mid \mathcal { T } _ { i } , \mathcal { R } _ { i } , \mathcal { R } _ { < i } ; \theta \big )\tag{2}
$$

Where N represents the length of the actual response. The variables I, A, and R represent the user command, the distorted audio input, and the target response, respectively. Within this framework, R is composed of C and T.

## 3.3.2 Audio Perceptual Reinforcement Learning

Building on the SFT-initialized model, we further apply reinforcement learning to improve structural consistency, output validity, and perceptual awareness. Specifically, we design three interpretable and complementary rewards: a format reward $R _ { \mathrm { f m t } } .$ , which enforces a valid and executable pipeline structure; a structure reward $R _ { s }$ , which encourages the inclusion and correct ordering of key analytical sections; and a perceptual quality reward $R _ { q } ,$ which evaluates the enhanced speech in terms of perceptual quality and intelligibility. The overall reward is defined as

$$
R = \lambda _ { f } R _ { \mathrm { f m t } } + \lambda _ { s } R _ { s } + \lambda _ { q } R _ { q } ,
$$

where $\lambda _ { f } , \lambda _ { s } , \lambda _ { q } > 0$ are weighting hyperparameters that balance the contributions of the format, structure, and perceptual quality rewards.

Format Reward. The format reward is designed to ensure that the recommended restoration pipeline follows a valid and executable format. Let y denote the tool output, $\mathcal M ( y )$ the set of parsed tool

names, and $\nu$ the set of valid tools. The format reward is defined as:

$$
R _ { \mathrm { f m t } } ( y ) = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f ~ t h e ~ p i p e l i n e ~ i s ~ m i s s i n g ~ o r ~ n o ~ v a l i d ~ s t e p s ~ c a n ~ b e ~ p a r s e d } } \\ { - 0 . 2 5 , } & { \mathrm { i f ~ a n y ~ p a r s e d ~ t o o l ~ i s ~ i n v a l i d ~ } ( \mathcal { M } ( y ) \not \subseteq \mathcal { V } ) } \\ { 1 , } & { \mathrm { i f ~ a l l ~ p a r s e d ~ t o o l s ~ a r e ~ v a l i d ~ } ( \mathcal { M } ( y ) \subseteq \mathcal { V } ) } \end{array} \right.
$$

Structure Reward. We define a structure reward $R _ { s } \in [ 0 , 1 ]$ to encourage the model to generate responses with a complete and correctly ordered analytical structure.Let y denote the generated response, which is expected to contain four target sections in a predefined order, denoted as $\{ S _ { 1 } , S _ { 2 } , \bar { S _ { 3 } } , S _ { 4 } \}$ We define indicator variables

$$
\delta _ { i } ( y ) = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } \operatorname { s e c t i o n } S _ { i } { \mathrm { ~ a p p e a r s ~ i n ~ } } y { \mathrm { ~ a n d ~ f o l l o w s ~ t h e ~ c o r r e c t ~ o r d e r } } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

The structure reward is computed as

$$
R _ { s } ( y ) = { \left\{ \begin{array} { l l } { - 1 , } & { { \mathrm { i f ~ } } S _ { 4 } \not \in y , } \\ { { \frac { 1 } { 4 } } \sum _ { i = 1 } ^ { 4 } \delta _ { i } ( y ) , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{3}
$$

Sections that appear out of order do not contribute to the reward. The penalty term enforces the presence of the final section, ensuring completeness of the generated structure.

Perception quality reward. To evaluate the perceptual quality and intelligibility of enhanced speech, we design a composite reward based on two complementary objective metrics: DNSMOS and ESTOI. DNSMOS reflects perceptual speech quality, while ESTOI measures intelligibility. Let d and e denote the DNSMOS and ESTOI scores, respectively. We normalize each metric using a sigmoid function $\sigma ( x ) = 1 / ( 1 + e ^ { - x } )$ . Specifically, we apply a sigmoid transformation with slope parameter $k _ { d }$ and pivot $c _ { d }$ to DNSMOS, and a sigmoid with slope $k _ { e }$ and pivot $c _ { e }$ to ESTOI:

$$
s _ { d } ( d ) = \frac { 1 } { 1 + \exp ( - k _ { d } ( d - c _ { d } ) ) } , \qquad s _ { e } ( e ) = \frac { 1 } { 1 + \exp ( - k _ { e } ( e - c _ { e } ) ) } .\tag{4}
$$

We then combine the two normalized scores using a weighted geometric mean with exponent $\alpha \in [ 0 , 1 ] $

$$
s _ { \mathrm { j o i n t } } ( d , e ) = s _ { d } ( d ) ^ { \alpha } s _ { e } ( e ) ^ { 1 - \alpha } .\tag{5}
$$

Finally, the joint score is linearly rescaled to the interval [−1, 1]:

$$
R _ { q } = 2 s _ { \mathrm { j o i n t } } ( d , e ) - 1 .\tag{6}
$$

Substituting the above definitions yields:

$$
R _ { q } = 2 \left[ \left( \frac { 1 } { 1 + \exp ( - k _ { d } ( d - c _ { d } ) ) } \right) ^ { \alpha } \left( \frac { 1 } { 1 + \exp ( - k _ { e } ( e - c _ { e } ) ) } \right) ^ { 1 - \alpha } \right] - 1 .\tag{7}
$$

Novelty of our reward design. While prior RL fine-tuning for LLMs often relies on a single taskspecific reward (e.g., DNSMOS), audio restoration agents face unique challenges: the model must generate executable tool sequences and logically ordered analyses. Our reward function addresses this via three innovations: (1) a format reward that validates pipeline executability, (2) a structure reward that enforces section ordering, and (3) a joint perceptual reward that balances quality and intelligibility. To our knowledge, this is the first work to encode pipeline validity and reasoning order into RL rewards for audio enhancement agents.

## 3.3.3 Agent-Orchestrated Multi-model Enhancement Protocol

We build a service-oriented agent system on top of a modular pipeline framework for audio restoration. A lightweight HTTP interface enables remote execution and interaction during training. Concurrency is managed by a shared worker pool with a bounded queue controlled by semaphores, ensuring stable throughput and providing backpressure under high load. Tool usage is handled through a decoupled strategy that combines preloaded and resident tools, thereby reducing cold-start latency while limiting GPU memory consumption through a bounded cache. An executor module parses pipeline definitions and executes multi-step restoration workflows. Further details are provided in Supplementary Material.

Table 1: Comparison of StrixAE with Single-Task and All-in-One methods on four real-world blind test datasets. ↑ denotes higher is better.
<table><tr><td rowspan="2">Method</td><td colspan="4">Real Recordings</td><td colspan="4">AcoustBench-Real</td></tr><tr><td>DNSMOS ↑</td><td>NISQA↑</td><td>UTMOS ↑</td><td>SCOREQ ↑</td><td>DNSMOS ↑</td><td>NISQA↑</td><td>UTMOS ↑</td><td>SCOREQ↑</td></tr><tr><td colspan="9">Open-source Models</td></tr><tr><td>TIGER [45]</td><td>2.60</td><td>2.79</td><td>2.32</td><td>2.69</td><td>2.03</td><td>2.13</td><td>1.53</td><td>1.82</td></tr><tr><td>MossFormer2_SS_16K [54]</td><td>2.78</td><td>2.60</td><td>2.23</td><td>2.56</td><td>2.10</td><td>1.83</td><td>1.53</td><td>1.74</td></tr><tr><td>MP-SENet [24]</td><td>3.16</td><td>3.45</td><td>2.73</td><td>3.45</td><td>2.68</td><td>2.76</td><td>1.76</td><td>2.72</td></tr><tr><td>MossFormer2_GAN [55]</td><td>3.12</td><td>3.63</td><td>2.67</td><td>3.58</td><td>2.47</td><td>2.20</td><td>1.68</td><td>2.18</td></tr><tr><td>TF-GridNet [42]</td><td>3.01</td><td>3.24</td><td>2.56</td><td>3.27</td><td>2.66</td><td>2.66</td><td>1.69</td><td>2.53</td></tr><tr><td>* StrixAE-SFT</td><td>3.12</td><td>3.61</td><td>2.58</td><td>3.26</td><td>2.72</td><td>2.78</td><td>1.84</td><td>2.49</td></tr><tr><td>★ StrixAE-Orchestrated</td><td>3.18</td><td>3.67</td><td>2.84</td><td>3.59</td><td>2.71</td><td>2.79</td><td>1.83</td><td>2.54</td></tr><tr><td colspan="9">Close-source Models (Reference)</td></tr><tr><td>Audio Cleaner AI Adobe Acrobat</td><td>2.94</td><td>3.47</td><td>2.59 3.06</td><td>3.25 3.49</td><td>2.33 2.80</td><td>2.61</td><td>1.63</td><td>2.19 2.98</td></tr><tr><td></td><td colspan="5">3.08 3.92</td><td colspan="3">3.29 2.27</td></tr><tr><td>Method</td><td colspan="5">Personalization Enhancement</td><td colspan="3">Blind Test Set</td></tr><tr><td colspan="9">DNSMOS↑ NISQA↑ UTMOS ↑ SCOREQ ↑</td></tr><tr><td></td><td></td><td></td><td>Open-source Models</td><td></td><td>DNSMOS↑</td><td>NISQA↑</td><td>UTMOS ↑</td><td>SCOREQ↑</td></tr><tr><td>TIGER [45]</td><td>2.90</td><td>3.24</td><td>2.52</td><td>2.77</td><td>2.23</td><td>2.02</td><td>1.69</td><td>1.84</td></tr><tr><td>MossFormer2_SS_16K [54]</td><td>2.98</td><td>3.17</td><td>2.65</td><td>2.93</td><td>2.33</td><td>1.62</td><td>1.58</td><td>1.67</td></tr><tr><td>MP-SENet [24]</td><td>3.16</td><td>3.61</td><td>2.62</td><td>3.18</td><td>2.81</td><td>2.50</td><td>1.92</td><td>2.44</td></tr><tr><td>MossFormer2_GAN [55]</td><td>3.07</td><td>3.52</td><td>2.44</td><td>3.02</td><td>2.77</td><td>2.58</td><td>1.98</td><td>2.44</td></tr><tr><td>TF-GridNet [42]</td><td>3.11</td><td>3.44</td><td>2.68</td><td>3.17</td><td>2.85</td><td>2.76</td><td>1.91</td><td>2.57</td></tr><tr><td>★ StrixAE-SFT * StrixAE-Orchestrated</td><td>3.23 3.24</td><td>3.98</td><td>2.78</td><td>3.23</td><td>2.77</td><td>2.71</td><td>1.79</td><td>2.15</td></tr><tr><td></td><td></td><td>3.98</td><td>3.06</td><td>3.49</td><td>2.83</td><td>2.74</td><td>1.98</td><td>2.36</td></tr><tr><td colspan="9">Close-source Models (Reference)</td></tr><tr><td>Audio Cleaner AI Adobe Acrobat</td><td>3.06</td><td>3.47</td><td>2.68</td><td>3.14</td><td>2.60</td><td>2.53</td><td>1.89</td><td>2.35</td></tr><tr><td></td><td>3.16</td><td>4.02</td><td>2.89</td><td>3.22</td><td>2.92</td><td>2.89</td><td>2.35</td><td>2.67</td></tr></table>

## 4 Experiment

## 4.1 Experimental Setup

Training Setup. StrixAE is built upon Audio-Reasoner [43] and fine-tuned with LoRA using the AdamW optimizer. In the SFT stage, the model is trained for 3 epochs with a batch size of 8 and a learning rate of $1 \times 1 0 ^ { - 5 }$ . During RL tuning, the sampling temperature is set to 1.0. We construct a joint reward model, defined in Equation 7, using DNSMOS to assess perceptual quality and ESTOI to evaluate intelligibility. The overall reward is computed as a weighted combination with $\lambda _ { f } = 1$ $\lambda _ { s } = 0 . 3 ,$ , and $\lambda _ { q } = 0 . { \dot { 3 } }$ . Sigmoid normalization is applied with $( \bar { k } _ { d } , c _ { d } ) = ( 5 , 2 . 5 )$ for DNSMOS and $( k _ { e } , c _ { e } ) = ( \hat { 8 } . 0 , 0 . 5 5 )$ for ESTOI, and the two normalized scores are combined using $\alpha = 0 . 4 5$ All experiments are conducted on four NVIDIA H100 80G GPUs.

AcoustBench Evaluation Dataset. To comprehensively evaluate the performance of StrixAE, we develop the AcoustBench evaluation benchmark. This dataset integrates publicly available blind evaluation data from the 2020 DNS Challenge [31], the 2023 DNS Challenge [6], and the 2025 URGENT Challenge [34]. In addition, we introduce a new test dataset, AcoustBench-Real, collected from complex real-world environments. Further technical details are provided in Supplementary Material.

Evaluation Metrics. We use AcoustBench for training and tuning and evaluate our system on a real-world blind test set. To comprehensively assess audio quality, we employ multiple evaluation metrics, including DNSMOS [32], NISQA [26], UTMOS [33], and SCOREQ [29]. Supplementary Material provides detailed explanations of these metrics and their corresponding evaluation tasks.

Tool Settings. This paper employs task-specific audio enhancement and audio separation models as external tools. We select lightweight and efficient models to ensure practical tool invocation; however, incorporating more advanced tools may further improve performance. Further details are provided in Supplementary Material.

Table 2: Evaluation results of top 3 systems on track 1 blind test set in URGENT Challenge 2025.
<table><tr><td></td><td colspan="2">Non-intrusive</td><td colspan="4">Intrusive</td><td colspan="2">Downstream</td></tr><tr><td>Systems</td><td></td><td>DNSMOS ↑ NISQA ↑ PESQ ↑</td><td></td><td></td><td></td><td></td><td></td><td>ESTOI ↑ SDR ↑ LSD ↓ SpkSim ↑ CAcc(%) ↑</td></tr><tr><td>Rank 1 (Ours)</td><td>2.88</td><td>3.22</td><td>2.64</td><td>0.82</td><td>12.66</td><td>2.93</td><td>0.76</td><td>79.80</td></tr><tr><td>Rank 2</td><td>2.92</td><td>3.24</td><td>2.47</td><td>0.79</td><td>11.10</td><td>2.99</td><td>0.74</td><td>76.06</td></tr><tr><td>Rank 3</td><td>2.94</td><td>3.25</td><td>2.45</td><td>0.79</td><td>11.25</td><td>3.66</td><td>0.71</td><td>77.09</td></tr><tr><td>Rank 3</td><td>2.94</td><td>3.25</td><td>2.45</td><td>0.79</td><td>11.25</td><td>3.66</td><td>0.71</td><td>77.09</td></tr><tr><td>Rank 3</td><td>2.94</td><td>3.25</td><td>2.45</td><td>0.79</td><td>11.25</td><td>3.66</td><td>0.71</td><td>77.09</td></tr><tr><td>Rank 3</td><td>2.94</td><td>3.25</td><td>2.45</td><td>0.79</td><td>11.25</td><td>3.66</td><td>0.71</td><td>77.09</td></tr><tr><td>Rank 3</td><td>2.94</td><td>3.25</td><td>2.45</td><td>0.79</td><td>11.25</td><td>3.66</td><td>0.71</td><td>77.09</td></tr><tr><td>Official Baseline</td><td>2.85</td><td>2.77</td><td>2.24</td><td>0.76</td><td>10.24</td><td>2.72</td><td>0.70</td><td>75.60</td></tr></table>

## 4.2 Experimental Results

## 4.2.1 Evaluation on AcoustBench Evaluation Dataset

As shown in Table 3, StrixAE achieves superior performance across four non-intrusive metrics. Compared to existing open-source methods, including both single-task and unified models. StrixAE outperforms most metrics and even surpasses several closed-source baselines (Adobe Acrobat and Audio Cleaner) in certain aspects. Specifically, on the AcoustBench-Real dataset and its personalized enhancement test set, StrixAE achieves state-of-the-art (SOTA) results on DNSMOS, UTMOS, and SCOREQ. Against the strong baseline TF-GridNet[42], StrixAE obtains significant improvements of 0.17, 0.43, 0.28 and 0.32, respectively.

When handling real-world degradations, DNSMOS score of StrixAE is only 0.09 lower than that of the closed-source baseline. On the Blind Test challenge set, the gap with the leading baseline is further narrowed, with differences of only 0.02, 0.02, and 0.21 for DNSMOS, NISQA and SCOREQ, respectively.These cross-dataset comparisons demonstrate that StrixAE not only maintains highfidelity enhancement but also exhibits strong generalization robustness against complex distortions.

As illustrated in Figure 4, StrixAE effectively recovers speech frequencies compared to single-task methods. In contrast to other integrated models, it achieves better denoising performance while better preserving harmonic structures, thereby delivering high-quality audio enhancement. Additional results are provided in Supplementary Material.

![](images/fcd5995cc63601f54ef3006e79abdd748145f1c8d0f60e1dd743a6f8a236fed4.jpg)  
Figure 4: Spectrum visualization on the Real-Recording test dataset.

Furthermore, the StrixAE-Orchestrated variant outperformed its SFT-based counterpart in most evaluation scenarios, validating the effectiveness of the proposed APRL fine-tuning strategy, which significantly improves generalization ability and decision accuracy in complex scenarios.

Table 3: Comparison of StrixAE with other strategies on the Real Recordings validation set.
<table><tr><td>Strategy</td><td>DNSMOS ↑</td><td>NISQA↑</td><td>UTMOS ↑</td><td>ScoreQ ↑</td></tr><tr><td>(I) Random Order and Model</td><td>2.72</td><td>3.02</td><td>1.47</td><td>2.47</td></tr><tr><td>(II) Random Order + Predict Model</td><td>2.84</td><td>3.11</td><td>2.52</td><td>3.02</td></tr><tr><td>(III) Random Model + Predict Order</td><td>2.80</td><td>3.16</td><td>2.41</td><td>2.80</td></tr><tr><td>★ StrixAE-SFT</td><td>3.12</td><td>3.61</td><td>2.58</td><td>3.26</td></tr><tr><td>★ StrixAE-Orchestrated</td><td>3.18</td><td>3.67</td><td>2.84</td><td>3.59</td></tr></table>

## 4.2.2 Visualization of Reward Trends for APRL

Figure 5 illustrates the training dynamics of APRL across three reward components. The format reward rapidly converges to near-optimal values during the early training stage and fluctuates only slightly thereafter, indicating that the model quickly internalizes the required output format. In contrast, the audio perception quality (APQ) reward exhibits greater variability and improves more gradually, reflecting a more complex optimization landscape that requires sustained exploration. Meanwhile, the structure reward rises quickly and remains consistently high, suggesting that the model benefits from structural priors acquired during SFT, analogous to inherited “parameter preferences”.

![](images/e37ac2079aed6a63883da0a86ab53392db4fcdd3c6452cf19a328331c5792937.jpg)

![](images/caaeda0555366e7269eeae60b01da7a595c1f3d3ef8c565e8219b109d29317e7.jpg)

![](images/fcc4cc6e2f823c26551ac257e421ad681740524d37671ccd5761346b8532beb3.jpg)  
Figure 5: Visualization of the reward trends across training steps of for StrixAE.

This pattern indicates a sequential stabilization process: the model first optimizes easily regularized rewards, such as format and structure, before shifting capacity toward the more challenging APQ objective, whose search space is broader and less deterministic. Unlike reasoning-centric models such as DeepSeek-R1 [11], no clear “aha moment” emerges during training.

Nevertheless, the training dynamics reveal a key strength of our design: the format and structure rewards stabilize rapidly, indicating that the model quickly internalizes the required output template — a prerequisite for reliable agent deployment. The slower improvement of the perceptual reward reflects the inherent difficulty of audio quality optimization, which our APRL continues to refine over time. This separation of concerns (format → structure → quality) is a deliberate innovation that makes training more stable than end-to-end perceptual-only RL.

## 5 Ablation Study

Training strategy. As shown in Table 3, we conducted a three-task comparison of StrixAE with several baseline strategies to analyze the impact on task planning and model selection, including: (I) the task type is accurately identified, while both task execution order and model selection are random; (II) the task execution order is random, and the model is selected by StrixAE; (III) the model is randomly selected, and the task execution order is predicted by StrixAE.

Results. Table 3 quantifies the individual and combined effects of task scheduling and model selection. Both learned model selection (II) and task order prediction (III) outperform the fully random setting (I), verifying the necessity of each module. Integrating these two designs further boosts performance in StrixAE-SFT. By jointly optimizing task planning and dynamic model routing, StrixAE-Orchestrated attains the optimal results across all speech quality metrics.

Table 4 compares different training strategies. We observe that using only SFT already yields competitive performance across all metrics. However, applying reinforcement learning (RL) alone results in a noticeable performance degradation, indicating that RL by itself is insufficient for stable optimization. By combining SFT with RL, our method achieves the best performance across all evaluation metrics, demonstrating the effectiveness of the proposed two-stage training strategy.

Table 4: Evaluation results on the urgent real-world blind test dataset compared with other strategies.
<table><tr><td>Strategy</td><td>DNSMOS ↑</td><td>NISQA↑</td><td>UTMOS↑</td><td>ScoreQ ↑</td></tr><tr><td>(I) Only SFT</td><td>3.12</td><td>3.61</td><td>2.58</td><td>3.26</td></tr><tr><td>(II) Only RL</td><td>3.11</td><td>3.54</td><td>2.53</td><td>3.22</td></tr><tr><td>(III) SFT + RL</td><td>3.18</td><td>3.67</td><td>3.84</td><td>3.59</td></tr></table>

## 6 Conclusion

In this paper, we propose a novel two-stage training framework for building audio enhancement agents, and introduce StrixAE, an agent trained under this framework. Our main contributions include: (1) APRL, a reinforcement learning fine-tuning method that employs structured rewards to explicitly enforce pipeline executability and logical segmentation order; (2) a decomposed reward design that separately considers format validity, inference structure, and perceptual quality, improving training stability compared to single perceptual rewards; and (3) AcoustBench, the first large-scale benchmark specifically designed for learning and evaluating audio agent scheduling strategies under complex distortion coupling. Extensive experiments demonstrate that StrixAE outperforms most open-source and proprietary solutions, exhibiting superior generalization robustness. Across various real-world datasets and personalized enhancement tasks, it achieves state-of-the-art performance on multiple perceptual metrics.

## References

[1] J. B. Allen and D. A. Berkley. Image method for efficiently simulating small-room acoustics. The Journal ofthe Acoustical Society ofAmerica, 65(4):943–950, 1979.

[2] A. Anthropic. The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card, 1(1):4, 2024.

[3] R. Cao, S. Abdulatif, and B. Yang. Cmgan: Conformer-based metric gan for speech enhancement. arXiv preprint arXiv:2203.15149, 2022.

[4] R. Chao, W.-H. Cheng, M. La Quatra, S. M. Siniscalchi, C.-H. H. Yang, S.-W. Fu, and Y. Tsao. An investigation of incorporating mamba for speech enhancement. In 2024 IEEE Spoken Language Technology Workshop (SLT), pages 302–308. IEEE, 2024.

[5] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[6] H. Dubey, A. Aazami, V. Gopal, B. Naderi, S. Braun, R. Cutler, A. Ju, M. Zohourian, M. Tang, M. Golestaneh, et al. Icassp 2023 deep noise suppression challenge. IEEE Open Journal of Signal Processing, 5:725–737, 2024.

[7] K. Feng, M. Zhang, S. Chen, Y. Lin, K. Fan, Y. Jiang, H. Li, D. Zheng, C. Wang, and X. Yue. Gen-searcher: Reinforcing agentic search for image generation. arXiv preprint arXiv:2603.28767, 2026.

[8] E. Fonseca, J. Pons Puig, X. Favory, F. Font Corbera, D. Bogdanov, A. Ferraro, S. Oramas, A. Porter, and X. Serra. Freesound datasets: a platform for the creation of open audio datasets. 2017.

[9] L. Gao, J. Schulman, and J. Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pages 10835–10866. PMLR, 2023.

[10] J. F. Gemmeke, D. P. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 776–780. IEEE, 2017.

[11] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[12] S. Hong, M. Zhuge, J. Chen, X. Zheng, Y. Cheng, J. Wang, C. Zhang, Z. Wang, S. K. S. Yau, Z. Lin, et al. Metagpt: Meta programming for a multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations, 2023.

[13] S. M. Jain. Introduction to transformers for nlp. With the huggingface library and models to solve problems, 2022.

[14] J. Jensen and C. H. Taal. An algorithm for predicting the intelligibility of speech masked by modulated noise maskers. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(11):2009–2022, 2016.

[15] Y. Ju, W. Rao, X. Yan, Y. Fu, S. Lv, L. Cheng, Y. Wang, L. Xie, and S. Shang. Tea-pse: Tencent ethereal-audio-lab personalized speech enhancement system for icassp 2022 dns challenge. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 9291–9295. IEEE, 2022.

[16] Y. Ju, J. Chen, S. Zhang, S. He, W. Rao, W. Zhu, Y. Wang, T. Yu, and S. Shang. Tea-pse 3.0: Tencent-ethereal-audio-lab personalized speech enhancement system for icassp 2023 dnschallenge. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–2. IEEE, 2023.

[17] K. Kinoshita, M. Delcroix, S. Gannot, E. A. P. Habets, R. Haeb-Umbach, W. Kellermann, V. Leutnant, R. Maas, T. Nakatani, B. Raj, et al. A summary of the reverb challenge: state-ofthe-art and remaining challenges in reverberant speech processing research. EURASIP Journal on Advances in Signal Processing, 2016(1):7, 2016.

[18] B. Lay and T. Gerkmann. An analysis of the variance of diffusion-based speech enhancement. arXiv preprint arXiv:2402.00811, 2024.

[19] J.-M. Lemercier, J. Richter, S. Welker, E. Moliner, V. Välimäki, and T. Gerkmann. Diffusion models for audio restoration. arXiv preprint arXiv:2402.09821, 2024.

[20] C. Li, W. Wang, M. Sach, W. Zhang, K. Saijo, S. Cornell, Y. Fu, Z. Ni, T. Fingscheidt, S. Watanabe, et al. Icassp 2026 urgent speech enhancement challenge. arXiv preprint arXiv:2601.13531, 2026.

[21] Y. Lin, Z. Lin, H. Chen, P. Pan, C. Li, S. Chen, K. Wen, Y. Jin, W. Li, and X. Ding. Jarvisir: Elevating autonomous driving perception with intelligent image restoration. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22369–22380, 2025.

[22] Y. Lin, Z. Lin, K. Lin, J. Bai, P. Pan, C. Li, H. Chen, Z. Wang, X. Ding, W. Li, et al. Jarvisart: Liberating human artistic creativity via an intelligent photo retouching agent. arXiv preprint arXiv:2506.17612, 2025.

[23] Y.-X. Lu, Y. Ai, and Z.-H. Ling. Mp-senet: A speech enhancement model with parallel denoising of magnitude and phase spectra. arXiv preprint arXiv:2305.13686, 2023.

[24] Y.-X. Lu, Y. Ai, and Z.-H. Ling. Explicit estimation of magnitude and phase spectra in parallel for high-quality speech enhancement. Neural Networks, 189:107562, 2025.

[25] R. C. Maher. Principles offorensic audio analysis, volume 1. Springer, 2018.

[26] G. Mittag, B. Naderi, A. Chehadi, and S. Möller. Nisqa: A deep cnn-self-attention model for multidimensional speech quality prediction with crowdsourced datasets. arXiv preprint arXiv:2104.09494, 2021.

[27] S. R. B. Narayan and N. Agarwal. Introduction to langchain. In Mastering LangChain: A Comprehensive Guide to Building Generative AI Applications, pages 1–9. Springer, 2025.

[28] S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez. Gorilla: Large language model connected with massive apis. Advances in Neural Information Processing Systems, 37:126544–126565, 2024.

[29] A. Ragano, J. Skoglund, and A. Hines. Scoreq: Speech quality assessment with contrastive regression. Advances in Neural Information Processing Systems, 37:105702–105729, 2024.

[30] M. Ravanelli, T. Parcollet, A. Moumen, S. De Langen, C. Subakan, P. Plantinga, Y. Wang, P. Mousavi, L. Della Libera, A. Ploujnikov, et al. Open-source conversational ai with speechbrain 1.0. Journal of Machine Learning Research, 25(333):1–11, 2024.

[31] C. K. Reddy, V. Gopal, R. Cutler, E. Beyrami, R. Cheng, H. Dubey, S. Matusevych, R. Aichner, A. Aazami, S. Braun, et al. The interspeech 2020 deep noise suppression challenge: Datasets, subjective testing framework, and challenge results. arXiv preprint arXiv:2005.13981, 2020.

[32] C. K. Reddy, V. Gopal, and R. Cutler. Dnsmos p. 835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors. In ICASSP 2022-2022 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 886–890. IEEE, 2022.

[33] T. Saeki, D. Xin, W. Nakata, T. Koriyama, S. Takamichi, and H. Saruwatari. Utmos: Utokyosarulab system for voicemos challenge 2022. arXiv preprint arXiv:2204.02152, 2022.

[34] K. Saijo, W. Zhang, S. Cornell, R. Scheibler, C. Li, Z. Ni, A. Kumar, M. Sach, Y. Fu, W. Wang, et al. Interspeech 2025 urgent speech enhancement challenge. arXiv preprint arXiv:2505.23212, 2025.

[35] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[36] Y. Shen, K. Song, X. Tan, D. Li, W. Lu, and Y. Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems, 36:38154–38180, 2023.

[37] A. Szot, B. Mazoure, O. Attia, A. Timofeev, H. Agrawal, D. Hjelm, Z. Gan, Z. Kira, and A. Toshev. From multimodal llms to generalist embodied agents: Methods and lessons. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10644–10655, 2025.

[38] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

[39] J. Thiemann, N. Ito, and E. Vincent. The diverse environments multi-channel acoustic noise database (demand): A database of multichannel environmental noise recordings. In Proceedings ofMeetings on Acoustics, volume 19, page 035081. Acoustical Society of America, 2013.

[40] J. Wang, Z. Lin, T. Wang, M. Ge, L. Wang, and J. Dang. Mamba-seunet: Mamba unet for monaural speech enhancement. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025.

[41] Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi. Selfinstruct: Aligning language models with self-generated instructions. In Proceedings ofthe 61st annual meeting of the association for computational linguistics (volume 1: long papers), pages 13484–13508, 2023.

[42] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe. Tf-gridnet: Integrating full-and sub-band modeling for speech separation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31:3221–3236, 2023.

[43] Z. Xie, M. Lin, Z. Liu, P. Wu, S. Yan, and C. Miao. Audio-reasoner: Improving reasoning capability in large audio language models. arXiv preprint arXiv:2503.02318, 2025.

[44] J. Xu, Z. Guo, H. Hu, Y. Chu, X. Wang, J. He, Y. Wang, X. Shi, T. He, X. Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025.

[45] M. Xu, K. Li, G. Chen, and X. Hu. Tiger: Time-frequency interleaved gain extraction and reconstruction for efficient speech separation. arXiv preprint arXiv:2410.01469, 2024.

[46] Z. Xue, J. Wu, Y. Gao, F. Kong, L. Zhu, M. Chen, Z. Liu, W. Liu, Q. Guo, W. Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

[47] R. Yang, L. Song, Y. Li, S. Zhao, Y. Ge, X. Li, and Y. Shan. Gpt4tools: Teaching large language model to use tools via self-instruction. Advances in Neural Information Processing Systems, 36: 71995–72007, 2023.

[48] R. Yang, H. Chen, J. Zhang, M. Zhao, C. Qian, K. Wang, Q. Wang, T. V. Koripella, M. Movahedi, M. Li, et al. Embodiedbench: Comprehensive benchmarking multi-modal large language models for vision-driven embodied agents. arXiv preprint arXiv:2502.09560, 2025.

[49] Z. Yang, L. Li, J. Wang, K. Lin, E. Azarnasab, F. Ahmed, Z. Liu, C. Liu, M. Zeng, and L. Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023.

[50] W. Zhang, R. Scheibler, K. Saijo, S. Cornell, C. Li, Z. Ni, A. Kumar, M. Sach, W. Wang, S. Watanabe, et al. Neurips 2024 competition proposal: Urgent challenge. In NeurIPS 2024 Competition Track, 2024.

[51] W. Zhang, K. Saijo, S. Cornell, R. Scheibler, C. Li, Z. Ni, A. Kumar, M. Sach, W. Wang, Y. Fu, et al. Lessons learned from the urgent 2024 speech enhancement challenge. arXiv preprint arXiv:2506.01611, 2025.

[52] L. Zhao, Y. Yang, K. Zhang, W. Shao, Y. Zhang, Y. Qiao, P. Luo, and R. Ji. Diffagent: Fast and accurate text-to-image api selection with large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6390–6399, 2024.

[53] S. Zhao, Y. Ma, C. Ni, C. Zhang, H. Wang, T. H. Nguyen, K. Zhou, J. Q. Yip, D. Ng, and B. Ma. Mossformer2: Combining transformer and rnn-free recurrent network for enhanced time-domain monaural speech separation. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 10356–10360. IEEE, 2024.

[54] S. Zhao, Y. Ma, C. Ni, C. Zhang, H. Wang, T. Nguyen, K. Zhou, J. Yip, D. Ng, and B. Ma. Mossformer2: Combining transformer and rnn-free recurrent network for enhanced time-domain monaural speech separation. arxiv 2024. arXiv preprint arXiv:2312.11825, 2025.

[55] S. Zhao, Z. Pan, and B. Ma. Clearervoice-studio: Bridging advanced speech processing research and practical deployment. arXiv preprint arXiv:2506.19398, 2025.

[56] Z. Zhao, W. Chai, X. Wang, B. Li, S. Hao, S. Cao, T. Ye, and G. Wang. See and think: Embodied agent in virtual environment. In European Conference on Computer Vision, pages 187–204. Springer, 2024.