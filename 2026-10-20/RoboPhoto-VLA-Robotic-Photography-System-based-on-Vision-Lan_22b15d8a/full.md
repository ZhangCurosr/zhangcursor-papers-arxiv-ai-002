# RoboPhoto-VLA: Robotic Photography System based on Vision-Language-Action Model

Song-Li Wu<sup>1</sup>, Chubin Zhang<sup>1</sup>, Wenbo Ding<sup>1</sup>, Yansong Tang<sup>1,∗</sup>

<sup>1</sup>Shenzhen International Graduate School, Tsinghua University

{wsl24, zcb24}@mails.tsinghua.edu.cn

{ding.wenbo, tang.yansong}@sz.tsinghua.edu.cn

Abstract—Modern robotic photography requires understanding photographic intent, planning camera motions, and producing aesthetically pleasing images in complex environments. We present RoboPhoto-VLA, a unified Vision-Language-Action framework that jointly models perception, reasoning, and camera control for autonomous robotic photography. RoboPhoto-VLA incorporates PhotoCoT for structured action planning and Aesthetic Direct Preference Optimization (ADPO) to align camera behaviors with human aesthetic preferences. To facilitate training and evaluation, we construct OpenPhotography, a robotic photography dataset with language-guided demonstrations. Extensive real-world experiments demonstrate that RoboPhoto-VLA produces photographs better aligned with human aesthetic preferences while generalizing effectively to unseen subjects, viewpoints, poses, and complex photographic instructions.

## I. INTRODUCTION

Robotic photography is emerging as an important embodied AI application in creative production, event recording, and commercial media, where robots are expected to autonomously execute camera movements according to highlevel user instructions [1]–[3]. Unlike conventional robotic manipulation [4]–[6], whose objective is successful task execution, robotic photography requires balancing two complementary objectives: accurately understanding the semantic intent of the user while producing photographs that satisfy human aesthetic preferences. Achieving this objective demands not only precise camera control, but also reasoning about abstract photographic concepts such as composition, viewpoint, framing, and subject interaction. Developing robots capable of jointly perceiving scenes, interpreting photographic intent, and generating aesthetically pleasing camera behaviors therefore remains a fundamental challenge [7]–[9].

Existing robotic photography systems have explored reference-guided camera control, semantic event photography, and interactive computational photography [10]–[12]. However, these methods are typically built upon modular pipelines that decouple perception, language understanding, and camera control [13]–[15], limiting their ability to reason over abstract photographic intent and adapt to open-ended instructions, unseen subjects, and dynamic shooting environments. Recent Vision–Language–Action (VLA) models address this limitation by unifying perception, reasoning, and action within a single policy [16]–[21]. Nevertheless, existing VLAs are primarily optimized for task completion rather than photographic quality, lacking both photography-oriented reasoning and explicit optimization toward human aesthetic preferences. Moreover, the scarcity of embodied robotic photography datasets further limits their applicability to autonomous photography.

![](images/fcfd705bb0f9a7ae67d7ef59120bf17bdd5774377b257f01ec7c8588a33d94e5.jpg)  
Fig. 1. RoboPhoto-VLA first receives the user’s language request alongside a target scene (upper left: the user asks to capture a photo of themselves eating an apple). The system then detects the user, identifies incomplete framing of the pose, and initiates adjustments to its position and angle (upper right). Next, RoboPhoto-VLA precisely adjusts its robotic arm and camera orientation to align with the desired composition (lower left). The lower-right panel displays the final unretouched photograph captured by RoboPhoto-VLA.

To address these challenges, we formulate robotic photography as an aesthetic-aware Vision–Language–Action learning problem and propose RoboPhoto-VLA, a unified framework that jointly models perception, reasoning, and embodied camera control. Specifically, we introduce PhotoCoT, a photography-oriented reasoning strategy that grounds abstract photographic intent into structured camera operations, including viewpoint selection, framing adjustment, and camera motion planning. To further optimize photographic quality, we propose Aesthetical Direct Preference Optimization (ADPO), which explicitly aligns camera behaviors with human aesthetic preferences through preference learning rather than conventional task-success supervision. To facilitate training and evaluation, we further construct OpenPhotography, the first open-ended robotic photography dataset containing languageguided demonstrations together with aesthetic preference pairs.

![](images/bb1de42596b6935b2975b1c3fcb370dd024eae8ef28f9db45119b5a02c440a12.jpg)  
Fig. 2. RoboPhoto-VLA model architecture. Given an image observation and a language instruction, the model predicts executable robot camera actions. The architecture follows a two-stage training paradigm: (1) Supervised Fine-Tuning (SFT) with PhotoCoT for structured photographic grounding and multi-step action planning; (2) Reinforcement Learning (RL) with ADPO for refining action selection to better align with human aesthetic preferences.

Extensive real-world experiments demonstrate that RoboPhoto-VLA consistently produces photographs that better align with human aesthetic preferences while generalizing effectively across unseen subjects, viewpoints, poses, and diverse photographic scenarios.

Our contributions are threefold: we propose RoboPhoto-VLA for aesthetic-aware robotic photography, introduce PhotoCoT and ADPO for structured reasoning and aesthetic alignment, and construct OpenPhotography for real-world evaluation under diverse unseen settings.

## II. RELATED WORKS

Recent Vision–Language Models (VLMs) have achieved remarkable progress in multimodal understanding by combining pretrained vision encoders with large language models, enabling strong performance on visual reasoning, captioning, and question answering tasks [22]–[27]. Building upon these advances, Vision–Language–Action (VLA) models integrate perception, language, and robot control into a unified policy, allowing pretrained VLMs to directly predict robot actions for embodied manipulation and long-horizon planning [18], [28]– [33]. Representative models such as RT-2-X and OpenVLA demonstrate impressive generalization across robotic manipulation tasks but are primarily designed for task execution rather than photography-oriented reasoning and aesthetic decisionmaking. Robotic photography has traditionally focused on camera control, viewpoint selection, framing, and target tracking through rule-based or modular pipelines [10]–[12]. Although recent systems incorporate language understanding and scene perception, they generally lack unified reasoning over visual observations and abstract photographic intent, as well as explicit optimization toward human aesthetic preferences. In contrast, our work formulates robotic photography as an aesthetic-aware Vision–Language–Action problem. Built upon a unified VLA framework, RoboPhoto-VLA introduces photography-oriented reasoning and preference optimization to bridge abstract photographic intent with executable camera actions, enabling aesthetically aligned robotic photography.

## III. METHOD

We formulate robotic photography as an embodied vision– language–action (VLA) learning problem. Given an RGB observation I, a natural-language photography instruction x, and a camera action trajectory $A = \{ a _ { 1 } , \cdots , a _ { T } \}$ , the objective is to learn a policy $\pi _ { \boldsymbol { \theta } } ( A | I , \boldsymbol { x } )$ , that not only executes feasible camera motions but also produces photographs aligned with human aesthetic preferences. To this end, we propose RoboPhoto-VLA, which extends a general-purpose VLA policy with photography-oriented reasoning and preference optimization. The overall framework consists of three components: an OpenVLA backbone for end-to-end policy learning, PhotoCoT for structured photographic reasoning, and Aesthetical Direct Preference Optimization (ADPO) for aesthetic alignment.

## A. OpenVLA-based Policy

We adopt OpenVLA-7B as the backbone policy due to its strong embodied reasoning capability. Following OpenVLA, the policy consists of a dual visual encoder, a lightweight projector, and a large language model (LLM). Given an observation image, visual features are extracted by SigLIP and DINOv2, $z _ { v } ~ = ~ f _ { v } ( I )$ , while the language instruction is encoded as $z _ { l } = f _ { l } ( x )$ . The projected visual tokens and language tokens are concatenated and processed by the LLM to obtain a joint representation $h = f _ { \mathrm { L L M } } \big ( \big [ z _ { v } , z _ { l } \big ] \big )$ , from which the policy autoregressively predicts the camera trajectory $A \sim \pi _ { \theta } ( A | I , x )$ . Following OpenVLA, each continuous action dimension is independently discretized into 256 bins according to the 1% and 99% quantiles of the training distribution and represented as language tokens, allowing robot control to be optimized using the standard next-token prediction objective. Unlike OpenVLA, RoboPhoto-VLA explicitly introduces photography-oriented reasoning and aesthetic preference learning, enabling the policy to optimize photographic quality rather than only task completion.

## B. Photography-Oriented Chain-of-Thought

General-purpose VLA models directly predict actions from observations, making it difficult to interpret abstract photographic instructions. We therefore introduce PhotoCoT, which decomposes policy learning into photographic grounding and camera planning. Given an observation and instruction, Photo-CoT first generates a grounding representation $g = f _ { g } ( I , x ) =$ $\{ o , v , c , s \}$ , where $o , \ v , \ c ,$ and s denote the target subject, viewpoint, composition, and scene constraints, respectively. Instead of directly producing actions, the policy first predicts an intermediate reasoning sequence $r = f _ { r } ( g ) = \{ r _ { 1 } , \cdot \cdot \cdot , r _ { K } \}$ which explicitly describes the multi-step camera adjustment process. The final camera trajectory is then generated as $A = f _ { a } ( r )$

## C. Aesthetic Direct Preference Optimization

Although supervised fine-tuning enables reliable camera control, it does not explicitly optimize photographic aesthetics. We therefore formulate aesthetic alignment as a trajectorylevel preference learning problem. For each observation– instruction pair (I, x), multiple candidate camera trajectories are executed to capture corresponding images. These images are evaluated using a multimodal aesthetic assessment model, and the trajectories with higher and lower aesthetic scores are used to construct preference pairs $( y ^ { + } , y ^ { - } )$ . We initialize preference optimization from the PhotoCoT-supervised policy. The reference policy $\pi _ { \mathrm { r e f } }$ is a frozen copy of this policy, while π denotes the policy optimized during preference learning. Following DPO, the preference margin is defined as

$$
\Delta = \beta \left[ \log \frac { \pi _ { \theta } ( y ^ { + } \mid I , x ) } { \pi _ { \mathrm { r e f } } ( y ^ { + } \mid I , x ) } - \log \frac { \pi _ { \theta } ( y ^ { - } \mid I , x ) } { \pi _ { \mathrm { r e f } } ( y ^ { - } \mid I , x ) } \right] ,\tag{1}
$$

where $\beta$ controls the strength of deviation from the reference policy. The standard DPO objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D P O } } = - \mathbb { E } _ { ( I , x , y ^ { + } , y ^ { - } ) } \left[ \log \sigma ( \Delta ) \right] . } \end{array}\tag{2}
$$

Standard DPO treats all preference pairs equally, although their reliability may vary. We therefore introduce an adaptive confidence weight

$$
w = \sigma \left( \alpha \Delta _ { \mathrm { a e s } } - \gamma \Delta _ { \mathrm { l p i p s } } \right) ,\tag{3}
$$

where $\Delta _ { \mathrm { a e s } }$ denotes the aesthetic-score difference between the images produced by $y ^ { + }$ and $y ^ { - }$ , and $\Delta _ { \mathrm { l p i p s } }$ denotes their LPIPS perceptual distance. The coefficients α and $\gamma$ control the contributions of aesthetic confidence and perceptual variation, respectively. Consequently, preference pairs with a clear aesthetic margin receive larger weights, whereas pairs with large perceptual differences are down-weighted to reduce the influence of unrelated visual-content changes.

The resulting ADPO objective is

$$
\mathcal { L } _ { \mathrm { A D P O } } = - \mathbb { E } _ { ( I , x , y ^ { + } , y ^ { - } ) } \left[ w \log \sigma ( \Delta ) \right] .\tag{4}
$$

The resulting optimization objective becomes

$$
\mathcal { L } _ { \mathrm { A D P O } } = - \mathbb { E } \left[ w \log \sigma ( \Delta ) \right] .\tag{5}
$$

Finally, RoboPhoto-VLA is trained using a two-stage paradigm. We first perform supervised fine-tuning with Photo-CoT reasoning traces to learn photography-oriented grounding and camera planning. The resulting policy is subsequently optimized using ADPO to align camera trajectories with human aesthetic preferences. The overall objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { S F T } } + \lambda \mathcal { L } _ { \mathrm { A D P O } } , } \end{array}\tag{6}
$$

where λ balances supervised policy learning and preference optimization. During inference, RoboPhoto-VLA directly predicts an executable camera trajectory from an RGB observation and a natural-language instruction through a single forward pass.

## IV. EXPERIMENTS

## A. Experimental Setup

All experiments are conducted on a UR-series robotic arm equipped with an Intel RealSense D435i RGB-D camera. The balancing coefficient λ is set to 0.1 based on validation performance, balancing camera-control capability and aesthetic alignment. OpenPhotography contains 600 demonstrations covering 42 instructions, diverse subjects, viewpoints, camera heights, and human poses. Each demonstration includes an RGB observation, an instruction, a manually verified Qwen3- VL-generated PhotoCoT trace [34], a camera trajectory, and the resulting photograph. The demonstrations are split into training, validation, and test sets at a ratio of 6:2:2. For OOD evaluation, samples are further divided into in-distribution and OOD subsets at a ratio of 7:3, with all OOD samples excluded from training and preference optimization. Crosssubject evaluation separately considers unseen single-person and multi-person scenes; cross-view evaluation uses 45<sup>◦</sup>, 90<sup>◦</sup>, and 135<sup>◦</sup> viewpoints; and cross-posture evaluation considers unseen human poses. For preference learning, multiple SFTpolicy trajectories are executed and their photographs are scored by a multimodal aesthetic assessment model [35]. After removing failed or ambiguous comparisons, 200 trajectorylevel preference pairs are retained, with the higher- and lowerscoring trajectories labeled $y ^ { + }$ and $y ^ { - }$ , respectively.

We compare RoboPhoto-VLA with three representative general-purpose VLA models (RT-2-X, $\pi _ { 0 } ,$ and OpenVLA) and two representative robotic photography systems (ANSEL and Snapbot). The VLA baselines are fine-tuned on OpenPhotography using their official implementations, while ANSEL and Snapbot are evaluated using their original photography pipelines. Following prior work, we report three complementary metrics. Quality Score (QS) [36] measures overall image quality, including sharpness, exposure, and framing. Aesthetic Score (AS) [37] evaluates photographic composition and visual attractiveness using a pretrained aesthetic assessment model. Instruction–Image Alignment (IIA) [38] computes CLIP similarity between the generated image and the corresponding language instruction. All reported numbers are averaged over the testing set, and higher values indicate better performance.

## B. Comparison with State-of-the-Art Methods

Table I reports the comparison with representative VLA models and robotic photography systems. RoboPhoto-VLA consistently achieves the best performance across all crosssubject, cross-view, and cross-posture settings. Compared with general-purpose VLAs (RT-2-X, π<sub>0</sub>, and OpenVLA), our method significantly improves both aesthetic quality (AS) and instruction alignment (IIA), demonstrating the importance of photography-oriented reasoning and preference alignment. Compared with traditional photobots (ANSEL and Snapbot), RoboPhoto-VLA further produces higher-quality images while preserving accurate instruction following. The largest gains are observed in multi-person scenes and challenging viewpoints, where structured PhotoCoT reasoning and ADPO jointly enable robust subject grounding, multi-step camera planning, and aesthetically consistent image capture.

TABLE I. PERFORMANCE COMPARISON OF ROBOPHOTO-VLA WITH OTHER MODELS FOR PHOTOGRAPHY TASKS. ”HOP” DENOTES HUMAN-ONLYPHOTOGRAPHY AND ”HOP” DENOTES HUMAN–OBJECT PHOTOGRAPHY
<table><tr><td rowspan="2">Experiment</td><td rowspan="2">Subtask</td><td colspan="2">RT-2-X</td><td colspan="2"></td><td colspan="2">π0</td><td colspan="3">OpenVLA</td><td colspan="3">ANSEL</td><td colspan="2">Snapbot</td><td colspan="3">RoboPhoto-VLA</td></tr><tr><td>QS</td><td>AS</td><td>IIA</td><td>QS</td><td>AS</td><td>IIA</td><td>QS</td><td>AS</td><td>IIA</td><td>QS</td><td>AS</td><td>IIA</td><td>QS</td><td>AS IIA</td><td>QS</td><td>AS</td><td>IIA</td></tr><tr><td rowspan="3">Cross-subject</td><td>Single-Person</td><td>0.09</td><td>0.05</td><td>0.02</td><td>0.37</td><td>0.48</td><td>0.35</td><td>0.29</td><td>0.41</td><td>0.32</td><td>0.63</td><td>0.78</td><td>0.65 0.71</td><td>0.72 0.64</td><td>0.61 0.69</td><td>0.86</td><td>0.91</td><td>0.87</td></tr><tr><td>Multi-Person</td><td>0.03</td><td>0.07</td><td>0.01</td><td>0.21</td><td>0.31</td><td>0.32</td><td>0.27</td><td>0.36</td><td>0.38</td><td>0.75</td><td>0.66</td><td></td><td>0.77</td><td>0.68</td><td>0.89</td><td>0.90</td><td>0.91</td></tr><tr><td>45°</td><td>0.03</td><td>0.04</td><td>0.06</td><td>0.22</td><td>0.46</td><td>0.30</td><td>0.21</td><td>0.23</td><td>0.37</td><td>0.62</td><td>0.73</td><td>0.70</td><td>0.67</td><td>0.76 0.65</td><td>0.73</td><td>0.85</td><td>0.89</td></tr><tr><td rowspan="3">Cross-view</td><td>90°</td><td>0.07</td><td>0.08</td><td>0.09</td><td>0.35</td><td>0.49</td><td>0.36</td><td>0.20</td><td>0.45</td><td>0.39</td><td>0.74</td><td>0.63</td><td>0.78</td><td>0.66</td><td>0.71 0.69</td><td>0.88</td><td>0.81</td><td>0.86</td></tr><tr><td>135°</td><td>0.00</td><td>0.03</td><td>0.03</td><td>0.24</td><td>0.40</td><td>0.31</td><td>0.22</td><td>0.31</td><td>0.30</td><td>0.64</td><td>0.75 0.62</td><td>0.73</td><td>0.70</td><td>0.67</td><td>0.85</td><td>0.83</td><td>0.85</td></tr><tr><td>HoP</td><td>0.05</td><td>0.06</td><td>0.08</td><td>0.38</td><td>0.42</td><td>0.33</td><td>0.25</td><td>0.48</td><td></td><td></td><td>0.65</td><td>0.72</td><td>0.63</td><td>0.77</td><td>0.83</td><td>0.86</td><td>0.84</td></tr><tr><td>Cross-Posture</td><td>HOP</td><td>0.04</td><td>0.02</td><td>0.00</td><td>0.26</td><td>0.47</td><td>0.34</td><td>0.29</td><td>0.44</td><td>0.35 0.32</td><td>0.76 0.66</td><td>0.68 0.71</td><td>0.69</td><td>0.64</td><td>0.74 0.62</td><td>0.74</td><td>0.80</td><td>0.73</td></tr></table>

TABLE II. ABLATION STUDY
<table><tr><td rowspan="2">Experiment</td><td rowspan="2">Subtask</td><td colspan="3">OpenVLA</td><td colspan="3">w/ SFT</td><td colspan="3">W/ ADPO</td><td colspan="3">w/ DPO</td><td colspan="3">RoboPhoto-VLA</td></tr><tr><td>QS</td><td>AS</td><td>IIA</td><td>QS</td><td>AS</td><td>IIA</td><td>QS</td><td>AS</td><td>IIA</td><td>QS</td><td>AS</td><td>IIA</td><td>QS AS</td><td>IIA</td></tr><tr><td>Cross-subject</td><td>Single-Person</td><td>0.29</td><td>0.41 0.32</td><td></td><td>0.53</td><td>0.57 0.47</td><td></td><td>0.73 0.81</td><td></td><td>0.74</td><td>0.67 0.61</td><td>0.65</td><td>0.76</td><td>0.84</td><td>0.79</td></tr><tr><td rowspan="3">Cross-view</td><td>Multi-Person</td><td>0.27</td><td>0.36</td><td>0.38</td><td>0.38</td><td>0.42</td><td>0.56</td><td>0.74 0.72</td><td></td><td>0.75</td><td>0.68 0.52</td><td>0.63</td><td>0.81</td><td>0.83</td><td>0.84</td></tr><tr><td>45°</td><td>0.21</td><td>0.23 0.37</td><td>0.26</td><td>0.52</td><td>0.41</td><td>0.63</td><td>0.72</td><td>0.81</td><td>0.56</td><td>0.53</td><td>0.69</td><td>0.73</td><td>0.85</td><td>0.89</td></tr><tr><td>90°</td><td>0.20</td><td>0.45 0.39</td><td>0.46</td><td>0.62</td><td>0.44</td><td>0.78</td><td>0.77</td><td>0.73</td><td>0.71</td><td>0.63</td><td>0.66</td><td>0.88</td><td>0.81</td><td>0.86</td></tr><tr><td></td><td>135°</td><td>0.22</td><td>0.31</td><td>0.30</td><td>0.31</td><td>0.58</td><td>0.42</td><td>0.71</td><td>0.75</td><td>0.78</td><td>0.63 0.55</td><td>0.63</td><td>0.85</td><td>0.83</td><td>0.85</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=2>Photo</td><td rowspan=1 colspan=1>QS</td><td rowspan=1 colspan=1>AS</td><td rowspan=1 colspan=1>IIA</td></tr><tr><td rowspan=1 colspan=1>RT-2-X</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>OpenVLA</td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>0.37</td></tr><tr><td rowspan=1 colspan=1>π0</td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=1 colspan=1>ANSEL</td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>0.39</td></tr><tr><td rowspan=1 colspan=1>Snapbot</td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>0.60</td><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>0.88</td></tr><tr><td rowspan=1 colspan=1>RoboPhoto-VLA</td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td></tr></table>

Fig. 3. Instruction: Photograph me looking serious.

## C. Cross-Domain Generalization

We further evaluate out-of-distribution generalization on unseen subjects, viewpoints, and human poses. As shown in Table I, existing VLA models exhibit noticeable performance degradation under all distribution shifts, particularly for unseen identities and viewpoints. In contrast, RoboPhoto-VLA consistently maintains high QS, AS, and IIA scores across all settings. The results indicate that PhotoCoT learns transferable photographic reasoning beyond subject appearance, while ADPO improves the robustness of camera planning and aesthetic composition under diverse real-world scenarios.

## D. Qualitative Results

Representative results are shown in Fig. 3. Compared with OpenVLA and Snapbot, RoboPhoto-VLA better understands photographic intent, producing more accurate framing, viewpoint selection, and facial expression. As illustrated in Fig. 4, RoboPhoto-VLA also generalizes well to unseen subjects and viewpoints, generating consistently coherent and aesthetically pleasing photographs.

## E. Ablation Study

Table II summarizes the ablation results. Removing Photo-CoT leads to the largest drop in IIA, demonstrating the importance of structured photographic reasoning. Replacing ADPO with supervised fine-tuning or standard DPO mainly reduces AS, confirming that ADPO effectively aligns camera actions with human aesthetic preferences. Combining PhotoCoT and ADPO consistently achieves the best overall performance.

![](images/66ba3e83dbdbb57ce2a12601e0ea1eaf372977daf14fcb8184caa3980b8e459f.jpg)  
Fig. 4. We evaluate the model under cross-subject and cross-view generalization settings. For cross-subject evaluation, we employ the instructions “Capture me smiling naturally” and “Photograph me laughing with a friend” to assess performance in single-person and multi-person scenarios, respectively. For cross-view evaluation, we use the instruction “Take a photo of me drinking water” to examine robustness to viewpoint variations.

## V. CONCLUSION

We present RoboPhoto-VLA, a unified vision–language– action framework for autonomous robotic photography that bridges high-level photographic intent and low-level camera control. By combining PhotoCoT with aesthetic preference optimization via ADPO, our framework enables robots to reason about composition, plan coherent camera trajectories, and capture images aligned with human visual preferences within a unified policy. Extensive real-world experiments across diverse subjects, viewpoints, and poses demonstrate consistent improvements over general-purpose VLA models and existing robotic photography systems. We also introduce OpenPhotography, an open benchmark for advancing research in embodied robotic photography. We hope this work establishes robotic photography as a valuable testbed for embodied intelligence and advances research on aesthetic reasoning, human–robot creative collaboration, and preference-aware action learning.

[1] Y. Ma, Z. Song, Y. Zhuang, J. Hao, and I. King, “A survey on vision-language-action models for embodied ai,” arXiv preprint arXiv:2405.14093, 2024.

[2] G. Lu, Z. Wang, C. Liu, J. Lu, and Y. Tang, “Thinkbot: Embodied instruction following with thought chain reasoning,” arXiv preprint arXiv:2312.07062, 2023.

[3] C. Zhang, J. Wang, Z. Gao, Y. Su, T. Dai, C. Zhou, J. Lu, and Y. Tang, “Clap: Contrastive latent action pretraining for learning vision-language-action models from human videos,” arXiv preprint arXiv:2601.04061, 2026.

[4] G. Lu, W. Guo, C. Zhang, Y. Zhou, H. Jiang, Z. Gao, Y. Tang, and Z. Wang, “Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning,” arXiv preprint arXiv:2505.18719, 2025.

[5] G. Lu, B. Jia, P. Li, Y. Chen, Z. Wang, Y. Tang, and S. Huang, “Gwm: Towards scalable gaussian world models for robotic manipulation,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 9263–9274.

[6] W. Guo, G. Lu, H. Deng, Z. Wu, Y. Tang, and Z. Wang, “Vla-reasoner: Empowering vision-language-action models with reasoning via online monte carlo tree search,” in 2026 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2026, pp. 4990–4997.

[7] H. Luo, Q. Sun, S. Wu, C. Xu, W. Deng, H. Hu, and Y. Tang, “Stare: Surprisal-guided token-level advantage reweighting for policy entropy stability,” arXiv preprint arXiv:2606.19236, 2026.

[8] G. Lu, S. Zhang, Z. Wang, C. Liu, J. Lu, and Y. Tang, “Manigaussian: Dynamic gaussian splatting for multi-task robotic manipulation,” in European Conference on Computer Vision. Springer, 2024, pp. 349– 366.

[9] S.-L. Wu, J. Wang, Z. Du, and W. Gan, “Activemem: Dynamic latent memory trees for long-horizon agents,” arXiv preprint arXiv:2609.33244, 2026.

[10] O. Limoyo, J. Li, D. Rivkin, J. Kelly, and G. Dudek, “Photobot: Reference-guided interactive photography via natural language,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 2479–2486.

[11] D. Rivkin, G. Dudek, N. Kakodkar, D. Meger, O. Limoyo, X. Liu, and F. Hogan, “Ansel photobot: A robot event photographer with semantic intelligence,” arXiv preprint arXiv:2302.07931, 2023.

[12] C. Choi, J. Kim, Y. Nam, and Y. Lee, “Snapbot: enabling dynamic human robot interactions for real-time computational photography,” in Companion of the 2024 ACM/IEEE International Conference on Human-Robot Interaction, 2024, pp. 327–331.

[13] K. Kawaharazuka, J. Oh, J. Yamada, I. Posner, and Y. Zhu, “Visionlanguage-action models for robotics: A review towards real-world applications,” IEEE Access, 2025.

[14] K. Q. Lin, L. Li, D. Gao, Z. Yang, S. Wu, Z. Bai, S. W. Lei, L. Wang, and M. Z. Shou, “Showui: One vision-language-action model for gui visual agent,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 19 498–19 508.

[15] A. Adilkhanov, A. Yelenov, A. Seitzhanov, A. Mazhitov, A. Abdikarimov, D. Sandykbayeva, D. Kenzhebek, D. Mukashev, I. Umurbekov, J. Chumakov et al., “Survey on vision-language-action models,” arXiv preprint arXiv:2502.06851, 2025.

[16] S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta, “R3m: A universal visual representation for robot manipulation,” 2022. [Online]. Available: https://arxiv.org/abs/2203.12601

[17] S. Karamcheti, S. Nair, A. S. Chen, T. Kollar, C. Finn, D. Sadigh, and P. Liang, “Language-driven representation learning for robotics,” 2023. [Online]. Available: https://arxiv.org/abs/2302.12766

[18] A. Stone, T. Xiao, Y. Lu, K. Gopalakrishnan, K.-H. Lee, Q. Vuong, P. Wohlhart, S. Kirmani, B. Zitkovich, F. Xia et al., “Open-world object manipulation using pre-trained vision-language models,” arXiv preprint arXiv:2303.00905, 2023.

[19] S.-L. Wu, H. Jiang, J. Fan, Y. Huo, C. Zhang, and Y. Tang, “Dpamixersr: An efficient degradation-pattern-aware model for image superresolution,” arXiv preprint arXiv:2609.32705, 2026.

[20] J. Wang, A. Laurenzi, and N. Tsagarakis, “Autonomous behavior planning for humanoid loco-manipulation through grounded language model,” 2024. [Online]. Available: https://arxiv.org/abs/2408.08282

[21] J. Wang, R. Dai, W. Wang, L. Rossini, F. Ruscelli, and N. Tsagarakis, “HYPERmotion: Learning hybrid behavior planning for autonomous loco-manipulation,” in 8th Annual Conference on Robot Learning, 2024. [Online]. Available: https://openreview.net/forum?id=ma7McOiCZY

[22] D. A. Hudson and C. D. Manning, “Gqa: A new dataset for realworld visual reasoning and compositional question answering,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 6700–6709.

[23] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach, “Towards vqa models that can read,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 8317–8326.

[24] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19 730–19 742.

[25] H. Laurenc¸on, L. Saulnier, L. Tronchon, S. Bekman, A. Singh, A. Lozhkov, T. Wang, S. Karamcheti, A. Rush, D. Kiela et al., “Obelics: An open web-scale filtered dataset of interleaved imagetext documents,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[26] J. Zhang, J. Huang, S. Jin, and S. Lu, “Vision-language models for vision tasks: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[27] S.-L. Wu, Z. Du, X. Wang, and J. Wang, “Capedit: Capabilitypreserving lifelong knowledge editing for language models,” in Findings of the Association for Computational Linguistics: ACL 2026, 2026, pp. 15 710–15 752.

[28] J. Wang, L. Zhu, T. Weng, S.-L. Wu, H. Tan, J. Chen, C. Tao, H. Bai, L. Hou, L. Shang et al., “Grpo-vps: Enhancing group relative policy optimization with verifiable process supervision for effective reasoning,” arXiv preprint arXiv:2604.20659, 2026.

[29] A. Padalkar, A. Pooley, A. Jain, A. Bewley, A. Herzog, A. Irpan, A. Khazatsky, A. Rai, A. Singh, A. Brohan et al., “Open xembodiment: Robotic learning datasets and rt-x models,” arXiv preprint arXiv:2310.08864, 2023.

[30] X. Li, M. Liu, H. Zhang, C. Yu, J. Xu, H. Wu, C. Cheang, Y. Jing, W. Zhang, H. Liu et al., “Vision-language foundation models as effective robot imitators,” arXiv preprint arXiv:2311.01378, 2023.

[31] S.-L. Wu, J. Wang, Z. Du, W. Gan, and W. Liu, “Codeskill: Latent skill abstraction for long-horizon code agents,” arXiv preprint arXiv:2609.33243, 2026.

[32] S. Karamcheti, S. Nair, A. Balakrishna, P. Liang, T. Kollar, and D. Sadigh, “Prismatic vlms: Investigating the design space of visuallyconditioned language models,” arXiv preprint arXiv:2402.07865, 2024.

[33] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi et al., “Openvla: An open-source vision-language-action model,” arXiv preprint arXiv:2406.09246, 2024.

[34] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[35] T. Du, H. He, and Y. Peng, “Venus: Benchmarking and empowering multimodal large language models for aesthetic guidance and cropping,” arXiv preprint arXiv:2602.23980, 2026.

[36] H. Wu, Z. Zhang, W. Zhang, C. Chen, L. Liao, C. Li, Y. Gao, A. Wang, E. Zhang, W. Sun et al., “Q-align: Teaching lmms for visual scoring via discrete text-defined levels,” arXiv preprint arXiv:2312.17090, 2023.

[37] Y. Huang, X. Sheng, Z. Yang, Q. Yuan, Z. Duan, P. Chen, L. Li, W. Lin, and G. Shi, “Aesexpert: Towards multi-modality foundation model for image aesthetics perception,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 5911–5920.

[38] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.